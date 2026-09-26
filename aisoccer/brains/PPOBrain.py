from pathlib import Path

import numpy as np

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.constants import Constants
from aisoccer.ppo import MLP, gaussian_log_prob

NUM = Constants.NUM_PLAYERS
L = Constants.FIELD_LENGTH
H = Constants.FIELD_HEIGHT
REL_SCALE = 500.0  # pixels, for relative positions

# For each player, the indices of its four teammates.
TEAMMATES = np.array([[j for j in range(NUM) if j != i] for i in range(NUM)])
ROLES = np.eye(NUM)


ROWS = np.arange(NUM)[:, None]


def player_features(
    my_pos, my_vel, opp_pos, opp_vel, ball_pos, ball_vel, my_score, opp_score, game_time
):
    """
    One observation row per player, from that player's point of view (a 5 x OBS_DIM
    matrix). All five players share one policy network, so each row carries the player's
    role (its index) and everything relative to the player itself.

    Runs every decision in every game, so it fills one preallocated array with direct
    indexing (tests check it matches the straightforward version bit for bit).
    """
    my_pos = np.asarray(my_pos, dtype=float)
    my_vel = np.asarray(my_vel, dtype=float)
    opp_pos = np.asarray(opp_pos, dtype=float)
    opp_vel = np.asarray(opp_vel, dtype=float)
    ball_pos = np.asarray(ball_pos, dtype=float).reshape(2)
    ball_vel = np.asarray(ball_vel, dtype=float).reshape(2)
    n = my_pos.shape[0]
    out = np.empty((n, OBS_DIM))

    ball_rel = ball_pos[None, :] - my_pos
    ball_dist = np.sqrt((ball_rel * ball_rel).sum(axis=1))
    opp_rel = opp_pos[None, :, :] - my_pos[:, None, :]
    order = np.argsort((opp_rel**2).sum(axis=2), axis=1)  # nearest opponent first

    out[:, 0] = my_pos[:, 0] / L * 2 - 1
    out[:, 1] = my_pos[:, 1] / H * 2 - 1
    out[:, 2:4] = my_vel / Constants.MAX_PLAYER_VELOCITY
    out[:, 4:6] = ball_rel / REL_SCALE
    out[:, 6] = ball_pos[0] / L * 2 - 1
    out[:, 7] = ball_pos[1] / H * 2 - 1
    out[:, 8:10] = ball_vel / Constants.MAX_BALL_VELOCITY
    out[:, 10:18] = (my_pos[TEAMMATES[:n]] - my_pos[:, None, :]).reshape(n, -1) / REL_SCALE
    out[:, 18:26] = my_vel[TEAMMATES[:n]].reshape(n, -1) / Constants.MAX_PLAYER_VELOCITY
    out[:, 26:36] = opp_rel[ROWS[:n], order].reshape(n, -1) / REL_SCALE
    out[:, 36:46] = opp_vel[order].reshape(n, -1) / Constants.MAX_PLAYER_VELOCITY
    out[:, 46:51] = ROLES[:n]
    out[:, 51] = (ball_dist[None, :] < ball_dist[:, None]).sum(axis=1) / (NUM - 1)
    out[:, 52] = np.clip(my_score - opp_score, -3, 3) / 3.0
    out[:, 53] = game_time
    return out


OBS_DIM = 54
ACT_DIM = 2


# Training reward: goals are worth +/-"goal"; the other entries weight the terms of the
# shaping potential below. A goal is worth several times the largest shaping swing.
DEFAULT_REWARD = {"goal": 2.0, "progress": 0.3, "control": 0.15, "chase": 0.1}
CONTROL_SCALE = 100.0  # pixels: how much closer to the ball counts as control
TYPICAL_SPREAD = 300.0  # pixels: a typical mean distance between teammates


def potential_terms(my_pos, opp_pos, ball_pos, ball_vel=(0.0, 0.0)):
    """
    The terms of the shaping potential, each roughly in -1 .. 1:

    - progress: the ball is further up the field (towards their goal),
    - control: our nearest player is closer to the ball than theirs, so winning the
      ball or passing it forward to a teammate raises it and losing it lowers it,
    - chase: our nearest player is close to the ball,
    - final_third: the ball is deep in their half (0 outside their final third),
    - shot: the ball is heading into their goal mouth at speed, more so when close,
    - spread: our players are spread out rather than bunched.

    The last three come from analyse_stats.py: within a matchup, shots, the ball in
    the final third and team spread go with winning games.
    """
    my_pos = np.asarray(my_pos, dtype=float)
    ball_pos = np.asarray(ball_pos, dtype=float)
    ball_vel = np.asarray(ball_vel, dtype=float)
    ours = np.linalg.norm(my_pos - ball_pos, axis=1).min()
    theirs = np.linalg.norm(np.asarray(opp_pos) - ball_pos, axis=1).min()

    goal_line = L - 1 - Constants.GOAL_DEPTH
    shot = 0.0
    if ball_vel[0] > 0:
        y_at_line = ball_pos[1] + ball_vel[1] * (goal_line - ball_pos[0]) / ball_vel[0]
        if Constants.GOAL_Y_MIN < y_at_line < Constants.GOAL_Y_MAX:
            closeness = 1.0 - np.clip((goal_line - ball_pos[0]) / L, 0.0, 1.0)
            shot = min(ball_vel[0] / Constants.MAX_BALL_VELOCITY, 1.0) * closeness

    pairs = np.linalg.norm(my_pos[:, None, :] - my_pos[None, :, :], axis=2)
    n = len(my_pos)
    spread = pairs.sum() / (n * (n - 1))

    return {
        "progress": (ball_pos[0] / L - 0.5) * 2,  # -1 at our goal, +1 at theirs
        "control": np.tanh((theirs - ours) / CONTROL_SCALE),
        "chase": -ours / L,
        "final_third": float(np.clip((ball_pos[0] - 2 * L / 3) / (L / 3), 0.0, 1.0)),
        "shot": shot,
        "spread": float(np.clip(spread / TYPICAL_SPREAD - 1.0, -1.0, 1.0)),
    }


def shaping_potential(
    my_pos, opp_pos, ball_pos, reward=DEFAULT_REWARD, ball_vel=(0.0, 0.0)
):
    """
    Potential for reward shaping: the reward-weighted sum of ``potential_terms``.

    The reward is the change in this potential, so it cannot be farmed: moving the ball
    back and forth, or passing in circles, gives back exactly what it gained. Shaping of
    this form speeds up learning without changing which policy is best.
    """
    terms = potential_terms(my_pos, opp_pos, ball_pos, ball_vel)
    return sum(reward.get(name, 0.0) * value for name, value in terms.items())


def network(params):
    """An MLP holding these parameters; its layer sizes are read from their shapes."""
    params = [np.asarray(p, dtype=float) for p in params]
    sizes = [params[0].shape[0]] + [w.shape[1] for w in params[0::2]]
    net = MLP(sizes, np.random.default_rng(0))
    net.params = params
    return net


class PPOBrain(AbstractBrain):
    """
    A neural network policy trained with PPO (see ``aisoccer/ppo.py`` and
    ``train_ppo.py``).

    One small network is shared by all five players: it maps a player's view of the game
    to that player's acceleration. The brain picks a new action every ACTION_REPEAT
    ticks and holds it in between. Trained weights are loaded from
    ``aisoccer/brains/weights/PPOBrain.npz``; without them the brain plays an untrained
    (near-stationary) random network.
    """

    ACTION_REPEAT = 2
    WEIGHTS_FILE = Path(__file__).parent / "weights" / "PPOBrain.npz"

    def __init__(
        self, name=None, weights=None, deterministic=True, training=False, reward=None
    ):
        """
        :param weights: dict with "policy" (and for training "value") parameter lists
            and "log_std". Defaults to the saved weights file. A mixed team (see
            coach.py) also has "role_policies": one policy parameter list per player
            role, so each player can come from a different trained brain.
        :param deterministic: act with the policy's mean rather than sampling.
        :param training: sample actions and record a trajectory for PPO.
        :param reward: training reward weights, overriding ``DEFAULT_REWARD``.
        """
        super().__init__(name=name)
        if weights is None:
            weights = self.load_weights(self.WEIGHTS_FILE)
        self.policy = MLP([OBS_DIM, 128, 128, ACT_DIM], np.random.default_rng(0), 0.01)
        if weights is not None:
            self.policy = network(weights["policy"])
            self.log_std = np.asarray(weights["log_std"], dtype=float)
        else:
            self.log_std = np.full(ACT_DIM, -0.5)
        self.role_policies = None
        if weights is not None and weights.get("role_policies"):
            self.role_policies = []
            self.role_policies = [
                network(params) for params in weights["role_policies"]
            ]
        # Older saved versions may have been trained with a different action repeat.
        self.action_repeat = int(
            (weights or {}).get("action_repeat", self.ACTION_REPEAT)
        )
        self.value = None
        if training:
            self.value = network(weights["value"])
        self.deterministic = deterministic and not training
        self.training = training
        self.reward = {**DEFAULT_REWARD, **(reward or {})}
        self.ticks_until_decision = 0
        self.action = np.zeros((NUM, ACT_DIM))
        self.trajectory: dict[str, list] = {
            k: [] for k in ("obs", "act", "logp", "val", "phi", "goal")
        }

    @staticmethod
    def load_weights(path):
        if not Path(path).exists():
            return None
        data = np.load(path)
        n = len([k for k in data.files if k.startswith("policy_")])
        weights = {
            "policy": [data[f"policy_{i}"] for i in range(n)],
            "log_std": data["log_std"],
        }
        if "action_repeat" in data.files:
            weights["action_repeat"] = int(data["action_repeat"])
        if "role0_policy_0" in data.files:
            weights["role_policies"] = [
                [data[f"role{r}_policy_{i}"] for i in range(n)] for r in range(NUM)
            ]
        n_value = len([k for k in data.files if k.startswith("value_")])
        if n_value:
            weights["value"] = [data[f"value_{i}"] for i in range(n_value)]
        return weights

    @staticmethod
    def save_weights(path, weights):
        arrays = {f"policy_{i}": p for i, p in enumerate(weights["policy"])}
        arrays.update({f"value_{i}": p for i, p in enumerate(weights.get("value", []))})
        for r, params in enumerate(weights.get("role_policies") or []):
            arrays.update({f"role{r}_policy_{i}": p for i, p in enumerate(params)})
        arrays["log_std"] = weights["log_std"]
        arrays["action_repeat"] = np.array(
            weights.get("action_repeat", PPOBrain.ACTION_REPEAT)
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

    def do_move(self):
        if self.ticks_until_decision <= 0:
            self.decide()
            self.ticks_until_decision = self.action_repeat
        self.ticks_until_decision -= 1
        return self.action.copy()

    def decide(self):
        obs = player_features(
            self.my_players_pos,
            self.my_players_vel,
            self.opp_players_pos,
            self.opp_players_vel,
            self.ball_pos,
            self.ball_vel,
            self.my_score,
            self.opp_score,
            self.game_time,
        )
        if self.role_policies:
            mean = np.stack(
                [net(obs[[r]])[0] for r, net in enumerate(self.role_policies)]
            )
        else:
            mean = self.policy(obs)
        if self.deterministic:
            self.action = mean
            return

        action = mean + np.exp(self.log_std) * self.rng.standard_normal(mean.shape)
        self.action = action
        if self.training:
            t = self.trajectory
            t["obs"].append(obs)
            t["act"].append(action)
            t["logp"].append(gaussian_log_prob(action, mean, self.log_std))
            t["val"].append(self.value(obs)[:, 0])
            t["phi"].append(
                shaping_potential(
                    self.my_players_pos,
                    self.opp_players_pos,
                    self.ball_pos,
                    self.reward,
                    self.ball_vel,
                )
            )
            t["goal"].append(0.0)

    def on_goal_scored(self, team, game_state):
        self.goal(self.reward["goal"])

    def on_goal_conceded(self, team, game_state):
        self.goal(-self.reward["goal"])

    def goal(self, reward):
        # Everyone is back at kick-off, so pick a fresh action straight away.
        self.ticks_until_decision = 0
        if self.training and self.trajectory["goal"]:
            self.trajectory["goal"][-1] += reward

    def finish_trajectory(self, gamma):
        """
        Return this game's experience as arrays: obs (T, 5, D), act (T, 5, 2),
        logp (T, 5), val (T, 5) and the team reward per decision rew (T,).
        """
        t = self.trajectory
        phi = np.array(t["phi"])
        next_phi = np.append(phi[1:], phi[-1] / gamma if len(phi) else 0.0)
        rewards = np.array(t["goal"]) + gamma * next_phi - phi
        return {
            "obs": np.array(t["obs"]),
            "act": np.array(t["act"]),
            "logp": np.array(t["logp"]),
            "val": np.array(t["val"]),
            "rew": rewards,
        }
