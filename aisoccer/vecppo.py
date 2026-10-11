"""
PPOBrain decisions for many games at once (see ``aisoccer.vecgame.VecGame``).

``BatchedPPO`` plays a set of PPOBrain instances, one per seat (a team in one game).
Each decision tick it builds every due seat's observations with one vectorised
``batch_player_features`` call and runs each distinct network once over all the seats
that use it. For training brains it records what ``PPOBrain.decide`` records (and the
goal rewards ``PPOBrain.goal`` adds), and ``finish`` writes that into each brain's own
``trajectory`` lists, so ``finish_trajectory`` works unchanged.

Every function here repeats the elementwise operations of its PPOBrain counterpart
over a leading seat axis, so the results are bit-identical (tests/test_vecgame.py).
Seats whose networks read obs v2 get PPOBrain's extra_features, with the tick count
and score trend kept here per seat as each PPOBrain keeps them (tests/test_obs_v2.py).
"""

import numpy as np

from aisoccer.brains.PPOBrain import (
    CONTROL_SCALE,
    OBS_DIM,
    RAW_DIM,
    REL_SCALE,
    ROLES,
    TEAMMATES,
    TYPICAL_SPREAD,
    H,
    L,
    ScoreTrend,
    batch_chance,
    batch_raw_features,
    extra_features,
    time_left,
)
from aisoccer.constants import Constants

NUM = Constants.NUM_PLAYERS
# The terms of the shaping potential, in the order shaping_potential sums them.
TERMS = ("progress", "control", "chase", "final_third", "shot", "spread")
LOG_2PI = np.log(2 * np.pi)
FIELD = np.array([L, H], dtype=float)  # to normalise (x, y) positions


def batch_player_features(
    my_pos, my_vel, opp_pos, opp_vel, ball_pos, ball_vel, my_score, opp_score, game_time
):
    """
    ``player_features`` for K seats at once: (K, 5, 2) players, (K, 2) ball, (K,) scores
    and a scalar or (K,) game time give (K, 5, OBS_DIM) observations.
    """
    k = my_pos.shape[0]
    out = np.empty((k, NUM, OBS_DIM))

    ball_rel = ball_pos[:, None, :] - my_pos
    ball_dist = np.sqrt((ball_rel * ball_rel).sum(axis=-1))
    opp_rel = opp_pos[:, None, :, :] - my_pos[:, :, None, :]
    order = np.argsort((opp_rel**2).sum(axis=-1), axis=-1)  # nearest opponent first
    nearest = (
        order + (NUM * np.arange(k))[:, None, None]
    )  # rows of the flattened opponents

    out[:, :, 0:2] = my_pos / FIELD * 2 - 1
    out[:, :, 2:4] = my_vel / Constants.MAX_PLAYER_VELOCITY
    out[:, :, 4:6] = ball_rel / REL_SCALE
    out[:, :, 6:8] = (ball_pos / FIELD * 2 - 1)[:, None, :]
    out[:, :, 8:10] = (ball_vel / Constants.MAX_BALL_VELOCITY)[:, None, :]
    out[:, :, 10:18] = (my_pos[:, TEAMMATES] - my_pos[:, :, None, :]).reshape(
        k, NUM, -1
    ) / REL_SCALE
    out[:, :, 18:26] = (
        my_vel[:, TEAMMATES].reshape(k, NUM, -1) / Constants.MAX_PLAYER_VELOCITY
    )
    nearest_rel = opp_pos.reshape(-1, 2)[nearest] - my_pos[:, :, None, :]
    out[:, :, 26:36] = nearest_rel.reshape(k, NUM, -1) / REL_SCALE
    out[:, :, 36:46] = (
        opp_vel.reshape(-1, 2)[nearest].reshape(k, NUM, -1)
        / Constants.MAX_PLAYER_VELOCITY
    )
    out[:, :, 46:51] = ROLES
    out[:, :, 51] = (ball_dist[:, None, :] < ball_dist[:, :, None]).sum(axis=-1) / (
        NUM - 1
    )
    out[:, :, 52] = (
        np.clip(np.asarray(my_score) - np.asarray(opp_score), -3, 3) / 3.0
    )[:, None]
    out[:, :, 53] = np.broadcast_to(np.asarray(game_time, dtype=float), (k,))[:, None]
    return out


def batch_potential_terms(my_pos, opp_pos, ball_pos, ball_vel):
    """``potential_terms`` for K seats at once: a dict of (K,) arrays."""
    ours = np.sqrt(np.add.reduce((my_pos - ball_pos[:, None, :]) ** 2, axis=-1)).min(
        axis=-1
    )
    theirs = np.sqrt(np.add.reduce((opp_pos - ball_pos[:, None, :]) ** 2, axis=-1)).min(
        axis=-1
    )
    bx, by = ball_pos[:, 0], ball_pos[:, 1]
    vx, vy = ball_vel[:, 0], ball_vel[:, 1]

    goal_line = L - 1 - Constants.GOAL_DEPTH
    forward = vx > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        y_at_line = by + vy * (goal_line - bx) / np.where(forward, vx, 1.0)
    on_target = (
        forward
        & (Constants.GOAL_Y_MIN < y_at_line)
        & (y_at_line < Constants.GOAL_Y_MAX)
    )
    closeness = 1.0 - np.clip((goal_line - bx) / L, 0.0, 1.0)
    shot = np.where(
        on_target, np.minimum(vx / Constants.MAX_BALL_VELOCITY, 1.0) * closeness, 0.0
    )

    rel = my_pos[:, :, None, :] - my_pos[:, None, :, :]
    pairs = np.sqrt(np.add.reduce(rel * rel, axis=-1))
    spread = pairs.reshape(len(pairs), -1).sum(axis=-1) / (NUM * (NUM - 1))

    return {
        "progress": (bx / L - 0.5) * 2,
        "control": np.tanh((theirs - ours) / CONTROL_SCALE),
        "chase": -ours / L,
        "final_third": np.clip((bx - 2 * L / 3) / (L / 3), 0.0, 1.0),
        "shot": shot,
        "spread": np.clip(spread / TYPICAL_SPREAD - 1.0, -1.0, 1.0),
    }


def batch_log_prob(actions, mean, log_std):
    """
    ``gaussian_log_prob`` with one log_std row per seat (K, 2), or per seat and player
    (K, 5, 2): (K, 5, 2) actions give (K, 5).
    """
    if log_std.ndim == 2:
        log_std = log_std[:, None, :]
    std = np.exp(log_std)
    z = (actions - mean) / std
    return (
        -0.5 * (z * z).sum(axis=-1)
        - log_std.sum(axis=-1)
        - 0.5 * log_std.shape[-1] * LOG_2PI
    )


PRECISIONS = ("exact", "float64", "float32")


def forward(params, x, precision="exact"):
    """
    An MLP (``aisoccer.ppo.MLP``) over (K, rows, D) inputs, returning float64.

    - "exact": each seat's rows are multiplied separately (one stacked matmul), which is
      bit-identical to calling the MLP on one seat at a time as PPOBrain does,
    - "float64": all rows in one matrix product; the last bit of an output can differ
      (a larger product sums in a different order),
    - "float32": the same in single precision (pass float32 ``params``), several times
      faster; outputs differ from float64 by about 1e-7.
    """
    shape = x.shape
    h = x if precision == "exact" else x.reshape(-1, shape[-1])
    if precision == "float32":
        h = h.astype(np.float32)
    n_layers = len(params) // 2
    for i in range(n_layers):
        h = h @ params[2 * i] + params[2 * i + 1]
        if i < n_layers - 1:
            h = np.tanh(h)
    return h.reshape(*shape[:-1], h.shape[-1]).astype(float, copy=False)


def _net_key(net):
    return tuple(id(p) for p in net.params)


class BatchedPPO:
    """
    Decides for many PPOBrain seats at once. Seats whose brains share parameter arrays
    (e.g. every learner built from one weights dict) share one network pass.
    """

    def __init__(self, brains, precision="exact"):
        """:param precision: how to run the networks, see ``forward``."""
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        self.brains = brains
        self.precision = precision
        self._params: dict[tuple, list] = (
            {}
        )  # network -> its parameters at that precision
        k = len(brains)
        self.repeat = np.array([b.action_repeat for b in brains], dtype=int)
        self.until_decision = np.array(
            [b.ticks_until_decision for b in brains], dtype=int
        )
        self.actions = np.array([b.action for b in brains], dtype=float).reshape(
            k, NUM, 2
        )
        # (K, 5, 2): a brain's log_std is one row for the team or one per player role.
        self.log_std = np.array(
            [np.broadcast_to(np.asarray(b.log_std, dtype=float), (NUM, 2)) for b in brains]
        )
        # Observation width: the widest any seat's networks read (obs v2 adds columns
        # after the v1 ones, so narrower networks read the leading columns).
        self.obs_dims = [getattr(b, "obs_dim", OBS_DIM) for b in brains]
        self.width = max(self.obs_dims, default=OBS_DIM)
        self.tick = 0  # ticks played, as each PPOBrain's ObsMemory counts them
        # Obs v3 (raw) is not a superset of v1, so it cannot share a batch with other versions.
        self.raw = RAW_DIM in self.obs_dims
        if self.raw and any(d != RAW_DIM for d in self.obs_dims):
            raise ValueError("obs v3 (raw) networks cannot play in a batch with other observation versions")
        self.trend = ScoreTrend(k) if self.width > OBS_DIM else None
        self.sampling = np.array([not b.deterministic for b in brains])
        self.training = np.array([b.training for b in brains])
        self.log: list[tuple] = []  # every training decision, flushed by finish()
        self.decisions = np.zeros(k, dtype=int)  # training decisions so far, per seat
        self.goals: dict[tuple, float] = {}  # (seat, decision): goal reward on it
        self.goal_reward = np.array([b.reward["goal"] for b in brains], dtype=float)
        self.weights = {
            name: np.array([b.reward.get(name, 0.0) for b in brains], dtype=float)
            for name in TERMS
        }
        # The "chance" term (PPOBrain.chance_potential), which shaping_potential adds
        # after the TERMS for brains whose reward has a "chance" weight.
        self.chance_weight = np.array(
            [b.reward.get("chance", 0.0) for b in brains], dtype=float
        )
        self.xg_models = [b.reward.get("xg_model") for b in brains]

        # Seats grouped by the network(s) that pick their actions, and training seats by
        # their value network.
        policy_groups, value_groups = {}, {}
        for i, b in enumerate(brains):
            nets = tuple(b.role_policies) if b.role_policies else (b.policy,)
            key = tuple(_net_key(net) for net in nets)
            policy_groups.setdefault(key, (nets, []))[1].append(i)
            if b.training:
                role_values = getattr(b, "role_values", None)
                nets = tuple(role_values) if role_values else (b.value,)
                key = tuple(_net_key(net) for net in nets)
                value_groups.setdefault(key, (nets, []))[1].append(i)
        self.policy_groups = [
            (nets, np.array(seats)) for nets, seats in policy_groups.values()
        ]
        self.value_groups = [
            (nets, np.array(seats)) for nets, seats in value_groups.values()
        ]

    def act(self, views, game_time):
        """
        Every seat's acceleration matrix (K, 5, 2) this tick, as PPOBrain.do_move gives
        it. ``views(seats)`` returns the view (as VecGame.views) of the given seats.
        """
        due = self.until_decision <= 0
        if due.any():
            seats = np.flatnonzero(due)
            self.decide(seats, *views(seats), game_time)
            self.until_decision[due] = self.repeat[due]
        self.until_decision -= 1
        self.tick += 1
        return self.actions.copy()

    def decide(
        self,
        seats,
        my_pos,
        my_vel,
        opp_pos,
        opp_vel,
        ball_pos,
        ball_vel,
        my_score,
        opp_score,
        game_time,
    ):
        """PPOBrain.decide for the given seats (indices), whose views are passed row by row."""
        features = batch_raw_features if self.raw else batch_player_features
        obs = features(
            my_pos,
            my_vel,
            opp_pos,
            opp_vel,
            ball_pos,
            ball_vel,
            my_score,
            opp_score,
            game_time,
        )
        if self.trend is not None:  # obs v2 (PPOBrain.observation)
            n = len(seats)
            extra = extra_features(
                my_pos,
                my_vel,
                opp_pos,
                opp_vel,
                ball_pos,
                ball_vel,
                time_left(
                    np.full(n, self.tick),
                    np.broadcast_to(np.asarray(game_time, dtype=float), (n,)),
                ),
                self.trend.update(
                    seats, self.tick, np.asarray(my_score) - np.asarray(opp_score)
                ),
            )
            obs = np.concatenate([obs, extra], axis=-1)
        row = np.full(len(self.brains), -1)
        row[seats] = np.arange(len(seats))

        mean = np.empty((len(seats), NUM, 2))
        for nets, members in self.policy_groups:
            rows = row[members]
            rows = rows[rows >= 0]
            if not len(rows):
                continue
            if len(nets) == 1:
                mean[rows] = self.forward(nets[0], obs[rows])
            else:  # a mixed team: each player's role has its own policy
                for r, net in enumerate(nets):
                    mean[rows, r] = self.forward(net, obs[rows, r][:, None])[:, 0]

        action = mean.copy()
        sampling = np.flatnonzero(self.sampling[seats])
        if len(sampling):
            noise = np.stack(
                [self.brains[seats[i]].exploration_noise((NUM, 2)) for i in sampling]
            )
            action[sampling] = (
                mean[sampling]
                + np.exp(self.log_std[seats[sampling]]) * noise
            )
        self.actions[seats] = action

        learning = np.flatnonzero(self.training[seats])
        if not len(learning):
            return
        values = np.empty((len(seats), NUM))
        for nets, members in self.value_groups:
            rows = row[members]
            rows = rows[rows >= 0]
            if not len(rows):
                continue
            if len(nets) == 1:
                values[rows] = self.forward(nets[0], obs[rows])[:, :, 0]
            else:  # one value network per player role (train_roles.py)
                for r, net in enumerate(nets):
                    values[rows, r] = self.forward(net, obs[rows, r][:, None])[:, 0, 0]
        log_std = self.log_std[seats[learning]]
        logp = batch_log_prob(action[learning], mean[learning], log_std)
        terms = batch_potential_terms(
            my_pos[learning], opp_pos[learning], ball_pos[learning], ball_vel[learning]
        )
        phi = 0
        for name in TERMS:  # summed in shaping_potential's order
            phi = phi + self.weights[name][seats[learning]] * terms[name]
        who = seats[learning]
        if self.chance_weight[who].any():  # as shaping_potential adds it
            phi = np.where(
                self.chance_weight[who] != 0,
                phi
                + self.chance(
                    who,
                    my_pos[learning],
                    opp_pos[learning],
                    ball_pos[learning],
                    ball_vel[learning],
                ),
                phi,
            )
        self.log.append(
            (who, obs[learning], action[learning], logp, values[learning], phi)
        )
        self.decisions[who] += 1

    def chance(self, who, my_pos, opp_pos, ball_pos, ball_vel):
        """
        Each seat's (``who``, with its views row by row) weighted "chance" potential:
        weight * PPOBrain.chance_potential with the seat's own xG model, 0 for seats
        without a "chance" weight.
        """
        out = np.zeros(len(who))
        weight = self.chance_weight[who]
        rows = np.flatnonzero(weight)
        models = {}  # seats sharing an xG model (usually all) are done together
        for r in rows:
            models.setdefault(id(self.xg_models[who[r]]), []).append(r)
        for group in models.values():
            group = np.array(group)
            out[group] = weight[group] * batch_chance(
                my_pos[group],
                opp_pos[group],
                ball_pos[group],
                ball_vel[group],
                self.xg_models[who[group[0]]],
            )
        return out

    def forward(self, net, x):
        width = net.params[0].shape[0]
        if x.shape[-1] != width:  # an older observation version: its leading columns
            x = np.ascontiguousarray(x[..., :width])
        params = self._params.get(_net_key(net))
        if params is None:
            dtype = np.float32 if self.precision == "float32" else float
            params = self._params[_net_key(net)] = [
                p.astype(dtype, copy=False) for p in net.params
            ]
        return forward(params, x, self.precision)

    def goal(self, seats, reward):
        """
        PPOBrain.goal for the seats (a boolean mask) of a game that just had a goal:
        ``reward`` (per seat) goes on each training seat's latest decision, and every
        seat decides afresh at the kick-off.
        """
        self.until_decision[seats] = 0
        for i in np.flatnonzero(seats & self.training & (self.decisions > 0)):
            key = (i, self.decisions[i] - 1)
            self.goals[key] = self.goals.get(key, 0.0) + reward[i]

    def finish(self):
        """Write every training seat's recorded decisions into its brain's trajectory lists."""
        if not self.log:
            return
        who = np.concatenate([entry[0] for entry in self.log])
        columns = [
            np.concatenate([entry[c] for entry in self.log]) for c in range(1, 6)
        ]
        for i in np.flatnonzero(self.training):
            rows = np.flatnonzero(who == i)  # in decision order
            goal = [0.0] * len(rows)
            for (seat, step), reward in self.goals.items():
                if seat == i:
                    goal[step] += reward
            t = self.brains[i].trajectory
            for name, column in zip(("obs", "act", "logp", "val", "phi"), columns):
                if name == "obs" and column.shape[-1] != self.obs_dims[i]:
                    column = column[..., : self.obs_dims[i]]  # its own observation width
                t[name].extend(column[rows])
            t["goal"].extend(goal)
        self.log = []
