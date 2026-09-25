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


def player_features(
    my_pos, my_vel, opp_pos, opp_vel, ball_pos, ball_vel, my_score, opp_score, game_time
):
    """
    One observation row per player, from that player's point of view (a 5 x OBS_DIM
    matrix). All five players share one policy network, so each row carries the player's
    role (its index) and everything relative to the player itself.
    """
    my_pos = np.asarray(my_pos, dtype=float)
    my_vel = np.asarray(my_vel, dtype=float)
    opp_pos = np.asarray(opp_pos, dtype=float)
    opp_vel = np.asarray(opp_vel, dtype=float)
    ball_pos = np.asarray(ball_pos, dtype=float).reshape(2)
    ball_vel = np.asarray(ball_vel, dtype=float).reshape(2)

    def norm_pos(p):
        return np.stack([p[..., 0] / L * 2 - 1, p[..., 1] / H * 2 - 1], axis=-1)

    ball_rel = ball_pos[None, :] - my_pos  # (5, 2)
    ball_dist = np.linalg.norm(ball_rel, axis=1)

    team_rel = my_pos[None, :, :] - my_pos[:, None, :]  # [i, j] = pos j - pos i
    team_rel = np.take_along_axis(team_rel, TEAMMATES[:, :, None], axis=1)
    team_vel = my_vel[TEAMMATES]

    opp_rel = opp_pos[None, :, :] - my_pos[:, None, :]
    order = np.argsort((opp_rel**2).sum(axis=2), axis=1)  # nearest opponent first
    opp_rel = np.take_along_axis(opp_rel, order[:, :, None], axis=1)
    opp_v = opp_vel[order]

    closer_teammates = (ball_dist[None, :] < ball_dist[:, None]).sum(axis=1) / (NUM - 1)
    score_diff = np.clip(my_score - opp_score, -3, 3) / 3.0

    n = my_pos.shape[0]
    return np.concatenate(
        [
            norm_pos(my_pos),
            my_vel / Constants.MAX_PLAYER_VELOCITY,
            ball_rel / REL_SCALE,
            np.broadcast_to(norm_pos(ball_pos), (n, 2)),
            np.broadcast_to(ball_vel / Constants.MAX_BALL_VELOCITY, (n, 2)),
            team_rel.reshape(n, -1) / REL_SCALE,
            team_vel.reshape(n, -1) / Constants.MAX_PLAYER_VELOCITY,
            opp_rel.reshape(n, -1) / REL_SCALE,
            opp_v.reshape(n, -1) / Constants.MAX_PLAYER_VELOCITY,
            ROLES[:n],
            closer_teammates[:, None],
            np.full((n, 1), score_diff),
            np.full((n, 1), game_time),
        ],
        axis=1,
    )


OBS_DIM = 54
ACT_DIM = 2


def shaping_potential(my_pos, opp_pos, ball_pos):
    """
    Potential for reward shaping. It is higher when:

    - the ball is further up the field (progress towards their goal),
    - we control the ball: our nearest player is closer to it than theirs, so winning
      the ball or passing it forward to a teammate raises it and losing it lowers it,
    - our nearest player is close to the ball.

    The reward is the change in this potential, so it cannot be farmed: moving the ball
    back and forth, or passing in circles, gives back exactly what it gained. Shaping of
    this form speeds up learning without changing which policy is best.
    """
    ball_pos = np.asarray(ball_pos, dtype=float)
    ball_progress = (ball_pos[0] / L - 0.5) * 2  # -1 at our goal, +1 at theirs
    ours = np.linalg.norm(np.asarray(my_pos) - ball_pos, axis=1).min()
    theirs = np.linalg.norm(np.asarray(opp_pos) - ball_pos, axis=1).min()
    control = np.tanh((theirs - ours) / PPOBrain.CONTROL_SCALE)  # -1 .. +1
    return (
        PPOBrain.BALL_PROGRESS_WEIGHT * ball_progress
        + PPOBrain.CONTROL_WEIGHT * control
        - PPOBrain.CHASE_WEIGHT * ours / L
    )


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

    # Training reward: +2 per goal scored, -2 per goal conceded, plus shaping (see
    # shaping_potential). A goal is worth several times the largest shaping swing.
    GOAL_REWARD = 2.0
    BALL_PROGRESS_WEIGHT = 0.3
    CONTROL_WEIGHT = 0.15
    CONTROL_SCALE = 100.0  # pixels: how much closer to the ball counts as control
    CHASE_WEIGHT = 0.1

    def __init__(self, name=None, weights=None, deterministic=True, training=False):
        """
        :param weights: dict with "policy" (and for training "value") parameter lists
            and "log_std". Defaults to the saved weights file.
        :param deterministic: act with the policy's mean rather than sampling.
        :param training: sample actions and record a trajectory for PPO.
        """
        super().__init__(name=name)
        if weights is None:
            weights = self.load_weights(self.WEIGHTS_FILE)
        self.policy = MLP([OBS_DIM, 128, 128, ACT_DIM], np.random.default_rng(0), 0.01)
        if weights is not None:
            self.policy.params = [np.asarray(p, dtype=float) for p in weights["policy"]]
            self.log_std = np.asarray(weights["log_std"], dtype=float)
        else:
            self.log_std = np.full(ACT_DIM, -0.5)
        # Older saved versions may have been trained with a different action repeat.
        self.action_repeat = int(
            (weights or {}).get("action_repeat", self.ACTION_REPEAT)
        )
        self.value = None
        if training:
            self.value = MLP([OBS_DIM, 128, 128, 1])
            self.value.params = [np.asarray(p, dtype=float) for p in weights["value"]]
        self.deterministic = deterministic and not training
        self.training = training
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
        n_value = len([k for k in data.files if k.startswith("value_")])
        if n_value:
            weights["value"] = [data[f"value_{i}"] for i in range(n_value)]
        return weights

    @staticmethod
    def save_weights(path, weights):
        arrays = {f"policy_{i}": p for i, p in enumerate(weights["policy"])}
        arrays.update({f"value_{i}": p for i, p in enumerate(weights.get("value", []))})
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
                    self.my_players_pos, self.opp_players_pos, self.ball_pos
                )
            )
            t["goal"].append(0.0)

    def on_goal_scored(self, team, game_state):
        self.goal(self.GOAL_REWARD)

    def on_goal_conceded(self, team, game_state):
        self.goal(-self.GOAL_REWARD)

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
