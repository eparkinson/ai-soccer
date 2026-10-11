"""
The PPO-tactician: TacticsV1's players and skills, with the team strategy (Layer 1)
chosen per situation by a small policy network trained with PPO (train_tactician.py).

TacticsV1 plays by a strategy vector of 15 numbers in [0, 1] (STRATEGY: how high the
line sits, how many players press, how far out to shoot, ...). It always plays one
fixed vector, DEFAULT_STRATEGY. Here, every DECISION_INTERVAL ticks, a network looks
at the game (team_features) and outputs the vector to play until the next decision.

The action is the strategy before squashing: the network's output u (15 numbers) is
mapped to the strategy by sigmoid(u). PPO works on u, a plain Gaussian, so nothing in
aisoccer/ppo.py changes (the squash is part of the environment; the probability ratio
PPO uses is the same either way).

Start: the network's output bias is logit(DEFAULT_STRATEGY) and its output weights are
tiny, so an untrained tactician plays DEFAULT_STRATEGY in every situation: it is plain
TacticsV1 and learns to adapt from there. (DEFAULT_STRATEGY's "opportunistic" is 1.0, a
switch at 0.5; it starts at 0.98, which plays the same.)
"""

import json
from pathlib import Path

import numpy as np

from aisoccer.brains.PPOBrain import DEFAULT_REWARD, network, shaping_potential
from aisoccer.constants import Constants
from aisoccer.ppo import MLP, gaussian_log_prob
from aisoccer.tactics.v1 import DEFAULT_STRATEGY, STRATEGY, TacticsV1

L = Constants.FIELD_LENGTH
H = Constants.FIELD_HEIGHT
NUM = Constants.NUM_PLAYERS

ACT_DIM = len(STRATEGY)  # 15
# Ticks between strategy decisions. TacticsV1 re-assigns its players' jobs every 10
# ticks, so a strategy has to last a few re-assignments to have an effect the critic
# can see, and it must be a multiple of 10 so every decision lands on a re-assignment.
# 30 ticks is 60 decisions per 1800-tick game: still quick enough to switch between
# attacking and defending when the ball changes hands (the ball crosses the field in
# about 100 ticks at full speed).
DECISION_INTERVAL = 30
HIDDEN = (64, 64)
# Opponent style, averaged over the game so far: how quickly the average forgets.
STYLE_DECAY = 0.9
EPS = 0.02  # the initial strategy is kept this far inside [0, 1] so its logit is finite


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-np.asarray(u, dtype=float)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


DEFAULT_VECTOR = np.array([DEFAULT_STRATEGY[k] for k in STRATEGY])
DEFAULT_LOGITS = logit(DEFAULT_VECTOR)


def team_features(my_pos, my_vel, opp_pos, opp_vel, ball_pos, ball_vel, my_score, opp_score, style):
    """
    One observation for the whole team (the strategy is team-wide), in the brain's own
    frame (it defends x = 0), each entry roughly in -1 .. 1:

    - the ball: position and velocity (4),
    - score difference (clipped at 3) (1); no game time: training games (1800 ticks)
      and league matches (36000) differ in length, so time would mean different things,
    - both teams' positions, players sorted by x so the order means "deepest first"
      (10 + 10),
    - both teams' mean velocity (2 + 2),
    - the race for the ball: our and their nearest player's distance to it, and
      who is nearer (tanh of the difference, the shaping reward's "control") (3),
    - the opponent's style over the game so far (``style``, 3): their mean x, and how
      spread out they are along and across the field.
    """
    my_pos = np.asarray(my_pos, dtype=float)
    opp_pos = np.asarray(opp_pos, dtype=float)
    ball = np.asarray(ball_pos, dtype=float).reshape(2)
    ours = np.linalg.norm(my_pos - ball, axis=1).min()
    theirs = np.linalg.norm(opp_pos - ball, axis=1).min()
    scale = np.array([L, H])
    return np.concatenate(
        [
            ball / scale * 2 - 1,
            np.asarray(ball_vel, dtype=float).reshape(2) / Constants.MAX_BALL_VELOCITY,
            [np.clip(my_score - opp_score, -3, 3) / 3.0],
            (my_pos[np.argsort(my_pos[:, 0])] / scale * 2 - 1).ravel(),
            (opp_pos[np.argsort(opp_pos[:, 0])] / scale * 2 - 1).ravel(),
            np.asarray(my_vel, dtype=float).mean(axis=0) / Constants.MAX_PLAYER_VELOCITY,
            np.asarray(opp_vel, dtype=float).mean(axis=0) / Constants.MAX_PLAYER_VELOCITY,
            [ours / L, theirs / L, np.tanh((theirs - ours) / 100.0)],
            style,
        ]
    )


def opponent_style(opp_pos):
    """This moment's opponent shape: mean x (-1 .. 1), and spread in x and y."""
    opp_pos = np.asarray(opp_pos, dtype=float)
    return np.array(
        [opp_pos[:, 0].mean() / L * 2 - 1, opp_pos[:, 0].std() / (L / 4), opp_pos[:, 1].std() / (H / 4)]
    )


OBS_DIM = 35


def initial_policy(rng=None, hidden=HIDDEN, obs_dim=OBS_DIM):
    """An untrained policy network that outputs DEFAULT_LOGITS (plus a tiny variation)."""
    net = MLP([obs_dim, *hidden, ACT_DIM], rng or np.random.default_rng(0), out_scale=0.01)
    net.params[-1][:] = DEFAULT_LOGITS
    return net


class TacticianBrain(TacticsV1):
    """TacticsV1 whose strategy a PPO-trained network chooses (see the module docstring)."""

    def __init__(
        self, name=None, policy=None, log_std=None, interval=DECISION_INTERVAL, weights=None,
        deterministic=True, training=False, value=None, reward=None,
    ):
        """
        :param policy: the policy network's parameters (a list of arrays or nested
            lists, as in the entrant .json). Default: the untrained policy, i.e.
            DEFAULT_STRATEGY everywhere.
        :param weights: alternatively, a weights .npz path (save_weights) or dict.
        :param deterministic: play the policy's mean (how the league plays it).
        :param training: sample strategies and record a trajectory for PPO; needs
            ``value`` (the critic's parameters).
        """
        super().__init__(name=name)
        if isinstance(weights, (str, Path)):
            weights = self.load_weights(weights)
        weights = dict(weights or {})
        policy = policy if policy is not None else weights.get("policy")
        log_std = log_std if log_std is not None else weights.get("log_std")
        value = value if value is not None else weights.get("value")
        interval = int(weights.get("interval", interval))
        self.policy = network(policy) if policy is not None else initial_policy()
        self.log_std = np.asarray(log_std if log_std is not None else np.full(ACT_DIM, -0.7), dtype=float)
        self.interval = interval
        self.training = training
        self.deterministic = deterministic and not training
        self.value = network(value) if training else None
        self.reward = {**DEFAULT_REWARD, **(reward or {})}
        self.style = None
        self.trajectory: dict[str, list] = {k: [] for k in ("obs", "act", "logp", "val", "phi", "goal")}

    # ------------------------------------------------------------ layer 1

    def choose_strategy(self):
        # TacticsV1.do_move asks every tick and re-assigns every 10 ticks (its tick
        # count restarts at each kick-off); decide on every interval'th of those ticks.
        if self.ticks % self.interval == 0:
            self.decide()
        return self.strategy

    def decide(self):
        now = opponent_style(self.opp_players_pos)
        self.style = now if self.style is None else STYLE_DECAY * self.style + (1 - STYLE_DECAY) * now
        obs = team_features(
            self.my_players_pos, self.my_players_vel, self.opp_players_pos, self.opp_players_vel,
            self.ball_pos, self.ball_vel, self.my_score, self.opp_score, self.style,
        )
        mean = self.policy(obs[None, :])[0]
        u = mean if self.deterministic else mean + np.exp(self.log_std) * self.rng.standard_normal(ACT_DIM)
        self.strategy = dict(zip(STRATEGY, sigmoid(u)))
        if self.training:
            t = self.trajectory
            t["obs"].append(obs[None, :])
            t["act"].append(u[None, :])
            t["logp"].append(gaussian_log_prob(u[None, :], mean[None, :], self.log_std))
            t["val"].append(self.value(obs[None, :])[:, 0])
            t["phi"].append(
                shaping_potential(self.my_players_pos, self.opp_players_pos, self.ball_pos, self.reward, self.ball_vel)
            )
            t["goal"].append(0.0)

    def on_goal_scored(self, team, game_state):
        super().on_goal_scored(team, game_state)  # restarts the tick count: decide at kick-off
        self.goal(self.reward["goal"])

    def on_goal_conceded(self, team, game_state):
        super().on_goal_conceded(team, game_state)
        self.goal(-self.reward["goal"])

    def goal(self, reward):
        if self.training and self.trajectory["goal"]:
            self.trajectory["goal"][-1] += reward

    def finish_trajectory(self, gamma):
        """
        This game's experience in train_ppo.build_batch's layout (one "player": the
        team): obs (T, 1, OBS_DIM), act (T, 1, 15), logp (T, 1), val (T, 1), and the
        reward per decision rew (T,): goals plus the change in the shaping potential,
        as in PPOBrain.
        """
        t = self.trajectory
        phi = np.array(t["phi"])
        next_phi = np.append(phi[1:], phi[-1] / gamma if len(phi) else 0.0)
        return {
            "obs": np.array(t["obs"]).reshape(-1, 1, OBS_DIM),
            "act": np.array(t["act"]).reshape(-1, 1, ACT_DIM),
            "logp": np.array(t["logp"]).reshape(-1, 1),
            "val": np.array(t["val"]).reshape(-1, 1),
            "rew": np.array(t["goal"]) + gamma * next_phi - phi,
        }

    # ------------------------------------------------------------ files

    @staticmethod
    def save_weights(path, weights):
        """An .npz with policy_i, value_i, log_std and interval (for training resumes;
        the league plays the .json from entrant_spec, since its .npz files are PPOBrains)."""
        arrays = {f"policy_{i}": p for i, p in enumerate(weights["policy"])}
        arrays.update({f"value_{i}": p for i, p in enumerate(weights.get("value") or [])})
        arrays["log_std"] = np.asarray(weights["log_std"])
        arrays["interval"] = np.array(weights.get("interval", DECISION_INTERVAL))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

    @staticmethod
    def load_weights(path):
        data = np.load(path)
        n = len([k for k in data.files if k.startswith("policy_")])
        n_value = len([k for k in data.files if k.startswith("value_")])
        weights = {"policy": [data[f"policy_{i}"] for i in range(n)], "log_std": data["log_std"]}
        if n_value:
            weights["value"] = [data[f"value_{i}"] for i in range(n_value)]
        if "interval" in data.files:
            weights["interval"] = int(data["interval"])
        return weights

    @staticmethod
    def entrant_spec(weights, **extra):
        """
        A self-contained brain spec (see aisoccer/brainspec.py): {"class", "kwargs"}
        with the policy inline, so a copy of the file (the league's snapshot) is a
        frozen version of the brain. ``extra`` keys (e.g. the iteration) are added
        for readers; load_brain ignores them.
        """
        return {
            "class": f"{__name__}.TacticianBrain",
            "kwargs": {
                "policy": [np.asarray(p, dtype=float).tolist() for p in weights["policy"]],
                "log_std": np.asarray(weights["log_std"], dtype=float).tolist(),
                "interval": int(weights.get("interval", DECISION_INTERVAL)),
            },
            **extra,
        }

    @classmethod
    def write_entrant(cls, path, weights, **extra):
        """Write entrant_spec to a .json atomically (readers never see half a file)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.json")
        tmp.write_text(json.dumps(cls.entrant_spec(weights, **extra)))
        tmp.replace(path)
