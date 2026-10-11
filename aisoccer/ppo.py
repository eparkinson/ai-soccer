"""
Proximal Policy Optimisation (PPO) in plain numpy.

Small enough to read in one sitting: a tanh MLP with hand-written backprop, an Adam
optimiser, generalised advantage estimation and the clipped PPO update. The brain that
uses it lives in ``aisoccer/brains/PPOBrain.py`` and the training loop in
``train_ppo.py``.

The learner trains in float32 by default (``PPO(dtype=...)``): the update's matrix
products and tanh run about twice as fast as in float64, and float32 is ample for
gradients that Adam rescales anyway. What leaves the learner (``PPO.weights()``, and
so every saved file) is float64, as before, so brains load and play exactly as they
always have.
"""

import math

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
            h = h @ w  # a new array, so the in-place steps below never touch x
            h += b
            if i < n_layers - 1:
                np.tanh(h, out=h)
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
                # d @ W.T * (1 - h^2), with one temporary instead of three.
                w = self.params[2 * i]
                # With one output (the critic) d @ W.T is an outer product, which
                # broadcasting computes exactly and several times faster than BLAS.
                d = d * w[:, 0] if w.shape[1] == 1 else d @ w.T
                slope = np.multiply(h_in, h_in)
                np.subtract(1.0, slope, out=slope)
                d *= slope
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
        self._scratch = {}

    def _buffer(self, i, like):
        buf = self._scratch.get(i)
        if buf is None or buf.shape != like.shape or buf.dtype != like.dtype:
            buf = self._scratch[i] = np.empty_like(like)
        return buf

    def step(self, grads):
        # The same arithmetic, in the same order, as
        #   m = b1 m + (1 - b1) g;  v = b2 v + (1 - b2) g g
        #   p -= lr (m / c1) / (sqrt(v / c2) + eps)
        # but in reused buffers. Scalars are Python floats so that float32 state stays
        # float32 (a numpy float64 scalar would promote it).
        self.t += 1
        lr, b1, b2, eps = float(self.lr), float(self.beta1), float(self.beta2), float(self.eps)
        c1 = 1.0 - b1**self.t
        c2 = 1.0 - b2**self.t
        for i, (p, g, m, v) in enumerate(zip(self.params, grads, self.m, self.v)):
            a = self._buffer(2 * i, p)
            b = self._buffer(2 * i + 1, p)
            m *= b1
            np.multiply(1.0 - b1, g, out=a)
            m += a
            v *= b2
            np.multiply(1.0 - b2, g, out=a)
            a *= g
            v += a
            np.divide(m, c1, out=a)
            a *= lr
            np.divide(v, c2, out=b)
            np.sqrt(b, out=b)
            b += eps
            a /= b
            p -= a


def clip_grads(grads, max_norm):
    """Scale the gradients (in place) so their global norm is at most ``max_norm``."""
    norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
    if norm > max_norm:
        scale = float(max_norm / norm)
        for g in grads:
            g *= scale
    return grads, norm


def gaussian_log_prob(actions, mean, log_std):
    """
    Log density of ``actions`` under a diagonal Gaussian, summed over action dims.
    ``log_std`` is one row (act_dim,) for every action, or one row per player
    (players, act_dim) for (..., players, act_dim) actions (see train_roles.py).
    """
    std = np.exp(log_std)
    z = (actions - mean) / std
    return (
        -0.5 * (z * z).sum(axis=-1)
        - log_std.sum(axis=-1)
        # A Python float, so float32 inputs give float32 log probs (a numpy float64
        # scalar would promote them).
        - 0.5 * log_std.shape[-1] * math.log(2 * math.pi)
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
        dtype=np.float32,
    ):
        """
        :param dtype: precision of the update (networks, loss, gradients and Adam
            state). float32 is about twice as fast as float64; ``weights()`` returns
            float64 either way. The initial weights are drawn in float64 and then
            rounded, so a seed gives the same networks (to float32 rounding) in both.
        """
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.dtype = np.dtype(dtype)
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
        self.sync_dtype()

    def sync_dtype(self):
        """
        Bring the parameters and the optimiser state to ``self.dtype`` and point the
        optimisers at the current parameter arrays.

        update() and clone() call this first, so code that swaps in parameter arrays of
        another precision (e.g. float64 arrays from a saved file, as train_ppo.py's
        --resume does) keeps working. Arrays already in the right precision are kept
        as they are, so in-place references to them stay valid.
        """
        dt = self.dtype
        for net in (self.policy, self.value):
            net.params[:] = [np.asarray(p, dtype=dt) for p in net.params]
        self.log_std = np.asarray(self.log_std, dtype=dt)
        for opt, params in (
            (self.policy_opt, self.policy.params + [self.log_std]),
            (self.value_opt, self.value.params),
        ):
            opt.params = params
            opt.m = [np.asarray(m, dtype=dt) for m in opt.m]
            opt.v = [np.asarray(v, dtype=dt) for v in opt.v]

    def weights(self):
        """
        Everything a brain needs to act (and a worker needs to estimate values), as
        float64 copies whatever the training precision, so saved files and the games
        played with them are unchanged.
        """
        return {
            "policy": [p.astype(np.float64) for p in self.policy.params],
            "value": [p.astype(np.float64) for p in self.value.params],
            "log_std": self.log_std.astype(np.float64),
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

        With ``target_kl`` set, the epochs stop early once an epoch's mean approximate
        KL exceeds 1.5 * target_kl. That mean is the sample-weighted mean of the
        per-minibatch KLs measured during the epoch, each just before its own minibatch
        step, so no extra pass over the batch is needed. (It used to be measured on
        the whole batch after the epoch. Each sample's KL is now taken up to one
        epoch's worth of steps earlier, so while the KL is rising the measure reads
        somewhat lower and an update may run one more epoch before stopping.)
        """
        self.sync_dtype()
        dt = self.dtype
        n = len(obs)
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # One cast of the whole batch, rather than one per minibatch.
        obs = np.asarray(obs, dtype=dt)
        actions = np.asarray(actions, dtype=dt)
        old_log_probs = np.asarray(old_log_probs, dtype=dt)
        adv = np.asarray(adv, dtype=dt)
        returns = np.asarray(returns, dtype=dt)
        stats = {"policy_loss": [], "value_loss": [], "clip_frac": [], "approx_kl": []}

        stats["epochs"] = []
        for epoch in range(self.epochs):
            order = self.rng.permutation(n)
            kl_sum = 0.0
            for start in range(0, n, self.minibatch):
                end = start + self.minibatch
                idx = order[start:end]
                mb_obs = obs[idx]  # gathered once for both networks
                if train_policy:
                    kl = self._policy_step(
                        mb_obs, actions[idx], old_log_probs[idx], adv[idx], stats
                    )
                    kl_sum += kl * len(idx)
                self._value_step(mb_obs, returns[idx], stats)
            stats["epochs"] = [epoch + 1]
            if train_policy and self.target_kl is not None:
                if kl_sum / n > 1.5 * self.target_kl:
                    break

        return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}

    def clone(self, obs, actions, epochs=10, lr=1e-3):
        """
        Behaviour cloning: fit the policy mean to demonstrated actions (mean squared
        error). Returns the loss after each epoch.
        """
        self.sync_dtype()
        obs = np.asarray(obs, dtype=self.dtype)
        actions = np.asarray(actions, dtype=self.dtype)
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
        """One clipped policy-gradient step on a minibatch. Returns its approximate KL
        (measured before the step)."""
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

        kl = float((old_log_probs - log_probs).mean())
        stats["policy_loss"].append(float(loss))
        stats["clip_frac"].append(float((np.abs(ratio - 1.0) > self.clip).mean()))
        stats["approx_kl"].append(kl)
        return kl

    def _value_step(self, obs, returns, stats):
        pred, acts = self.value.forward(obs)
        err = pred[:, 0] - returns
        grads = self.value.backward(acts, (err / len(obs))[:, None])
        grads, _ = clip_grads(grads, self.max_grad_norm)
        self.value_opt.step(grads)
        stats["value_loss"].append(float(0.5 * (err * err).mean()))
