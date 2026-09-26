"""
Proximal Policy Optimisation (PPO) in plain numpy.

Small enough to read in one sitting: a tanh MLP with hand-written backprop, an Adam
optimiser, generalised advantage estimation and the clipped PPO update. The brain that
uses it lives in ``aisoccer/brains/PPOBrain.py`` and the training loop in
``train_ppo.py``.
"""

import numpy as np


class MLP:
    """Fully connected network with tanh hidden layers and a linear output layer."""

    def __init__(self, sizes, rng=None, out_scale=1.0):
        rng = rng or np.random.default_rng()
        self.params = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = i == len(sizes) - 2
            # Orthogonal-ish init: scaled Gaussian. A small output layer keeps the initial
            # policy close to zero mean, which PPO implementations find helps.
            scale = (out_scale if last else 1.0) / np.sqrt(n_in)
            self.params.append(rng.normal(0.0, scale, (n_in, n_out)))
            self.params.append(np.zeros(n_out))

    def forward(self, x):
        """Return the output and the activations needed by ``backward``."""
        acts = [x]
        h = x
        n_layers = len(self.params) // 2
        for i in range(n_layers):
            w, b = self.params[2 * i], self.params[2 * i + 1]
            h = h @ w + b
            if i < n_layers - 1:
                h = np.tanh(h)
            acts.append(h)
        return h, acts

    def __call__(self, x):
        return self.forward(x)[0]

    def backward(self, acts, d_out):
        """Gradients of the loss with respect to every parameter, given dLoss/dOutput."""
        grads = [np.empty(0)] * len(self.params)
        n_layers = len(self.params) // 2
        d = d_out
        for i in reversed(range(n_layers)):
            h_in = acts[i]
            grads[2 * i] = h_in.T @ d
            grads[2 * i + 1] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.params[2 * i].T) * (1.0 - h_in**2)
        return grads


class Adam:
    def __init__(self, params, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        c1 = 1.0 - self.beta1**self.t
        c2 = 1.0 - self.beta2**self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def clip_grads(grads, max_norm):
    norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
    if norm > max_norm:
        grads = [g * (max_norm / norm) for g in grads]
    return grads, norm


def gaussian_log_prob(actions, mean, log_std):
    """Log density of ``actions`` under a diagonal Gaussian, summed over action dims."""
    std = np.exp(log_std)
    z = (actions - mean) / std
    return (
        -0.5 * (z * z).sum(axis=-1)
        - log_std.sum()
        - 0.5 * len(log_std) * np.log(2 * np.pi)
    )


def gae(rewards, values, dones, last_value, gamma, lam):
    """
    Generalised advantage estimation over one trajectory.

    ``dones[t]`` marks that the episode ended after step t, so nothing is bootstrapped
    past it. Returns (advantages, returns).
    """
    n = len(rewards)
    adv = np.zeros(n)
    running = 0.0
    next_value = last_value
    for t in reversed(range(n)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        running = delta + gamma * lam * not_done * running
        adv[t] = running
        next_value = values[t]
    return adv, adv + values


class PPO:
    """The PPO learner: owns the policy and value networks and their optimisers."""

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden=(128, 128),
        lr=3e-4,
        clip=0.2,
        entropy_coef=0.0,
        epochs=4,
        minibatch=4096,
        max_grad_norm=0.5,
        init_log_std=-0.5,
        min_log_std=-3.0,
        target_kl=None,
        seed=None,
    ):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.policy = MLP([obs_dim, *hidden, act_dim], rng, out_scale=0.01)
        self.value = MLP([obs_dim, *hidden, 1], rng, out_scale=1.0)
        self.log_std = np.full(act_dim, init_log_std)
        self.policy_opt = Adam(self.policy.params + [self.log_std], lr=lr)
        self.value_opt = Adam(self.value.params, lr=lr)
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.minibatch = minibatch
        self.max_grad_norm = max_grad_norm
        # Floor on the exploration noise, so the policy cannot stop exploring early.
        self.min_log_std = min_log_std
        # Stop an update's epochs once the policy has moved this far (approximate KL).
        self.target_kl = target_kl

    def weights(self):
        """Everything a brain needs to act (and a worker needs to estimate values)."""
        return {
            "policy": [p.copy() for p in self.policy.params],
            "value": [p.copy() for p in self.value.params],
            "log_std": self.log_std.copy(),
        }

    def set_lr(self, lr):
        self.policy_opt.lr = lr
        self.value_opt.lr = lr

    def update(
        self, obs, actions, old_log_probs, advantages, returns, train_policy=True
    ):
        """
        Run the clipped PPO update over one batch of experience and return stats.

        With ``train_policy=False`` only the value network learns. That is used to fit
        the critic to a pre-trained (e.g. behaviour cloned) policy before PPO starts
        changing it, since advantages from an untrained critic are noise.
        """
        n = len(obs)
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        stats = {"policy_loss": [], "value_loss": [], "clip_frac": [], "approx_kl": []}

        stats["epochs"] = []
        for epoch in range(self.epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, self.minibatch):
                end = start + self.minibatch
                idx = order[start:end]
                if train_policy:
                    self._policy_step(
                        obs[idx], actions[idx], old_log_probs[idx], adv[idx], stats
                    )
                self._value_step(obs[idx], returns[idx], stats)
            stats["epochs"] = [epoch + 1]
            if train_policy and self.target_kl is not None:
                log_probs = gaussian_log_prob(actions, self.policy(obs), self.log_std)
                if float((old_log_probs - log_probs).mean()) > 1.5 * self.target_kl:
                    break

        return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}

    def clone(self, obs, actions, epochs=10, lr=1e-3):
        """
        Behaviour cloning: fit the policy mean to demonstrated actions (mean squared
        error). Returns the loss after each epoch.
        """
        opt = Adam(self.policy.params, lr=lr)
        losses = []
        for _ in range(epochs):
            order = self.rng.permutation(len(obs))
            epoch_loss = []
            for start in range(0, len(obs), self.minibatch):
                end = start + self.minibatch
                idx = order[start:end]
                mean, acts = self.policy.forward(obs[idx])
                err = mean - actions[idx]
                grads = self.policy.backward(acts, err / len(idx))
                grads, _ = clip_grads(grads, self.max_grad_norm)
                opt.step(grads)
                epoch_loss.append(float((err * err).sum(axis=1).mean()))
            losses.append(float(np.mean(epoch_loss)))
        return losses

    def _policy_step(self, obs, actions, old_log_probs, adv, stats):
        m = len(obs)
        mean, acts = self.policy.forward(obs)
        std = np.exp(self.log_std)
        log_probs = gaussian_log_prob(actions, mean, self.log_std)
        ratio = np.exp(log_probs - old_log_probs)
        clipped = np.clip(ratio, 1.0 - self.clip, 1.0 + self.clip)
        loss = -np.minimum(ratio * adv, clipped * adv).mean()

        # The min() picks the unclipped term (and so passes gradient) unless the ratio
        # has already moved past the clip range in the direction the advantage favours.
        active = ~(
            ((adv > 0) & (ratio > 1.0 + self.clip))
            | ((adv < 0) & (ratio < 1.0 - self.clip))
        )
        d_log_prob = np.where(active, -adv * ratio, 0.0) / m

        z = (actions - mean) / std
        d_mean = d_log_prob[:, None] * z / std
        d_log_std = (d_log_prob[:, None] * (z * z - 1.0)).sum(axis=0)
        d_log_std -= self.entropy_coef  # entropy of a Gaussian is sum(log_std) + const

        grads = self.policy.backward(acts, d_mean) + [d_log_std]
        grads, _ = clip_grads(grads, self.max_grad_norm)
        self.policy_opt.step(grads)
        # Keep exploration noise in a sane range.
        np.clip(self.log_std, self.min_log_std, 0.5, out=self.log_std)

        stats["policy_loss"].append(loss)
        stats["clip_frac"].append(float((np.abs(ratio - 1.0) > self.clip).mean()))
        stats["approx_kl"].append(float((old_log_probs - log_probs).mean()))

    def _value_step(self, obs, returns, stats):
        pred, acts = self.value.forward(obs)
        err = pred[:, 0] - returns
        grads = self.value.backward(acts, (err / len(obs))[:, None])
        grads, _ = clip_grads(grads, self.max_grad_norm)
        self.value_opt.step(grads)
        stats["value_loss"].append(float(0.5 * (err * err).mean()))
