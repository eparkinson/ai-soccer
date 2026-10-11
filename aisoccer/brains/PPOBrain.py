import json
from collections import deque
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


# ---------------------------------------------------------------- observation v2
#
# Obs v2 is opt-in: a network trained on it says so in its weights file ("obs_version"),
# and PPOBrain builds whichever features its network reads. Its rows are the 54 v1
# features followed by the OBS_V2_NAMES features, all computed directly from the current
# view (distances, directions, speeds, one straight-line extrapolation of the ball) plus
# two things remembered within the game: the tick count (for the time left) and the
# score history (for the recent trend). No rollouts, bounces, planning or searches.
OBS_V2_NAMES = (
    "time_left",  # min(ticks left, TIME_WINDOW) / TIME_WINDOW
    "score_trend",  # goal difference over the last TIME_WINDOW ticks, clipped +-3, / 3
    "goal_dist",  # player -> centre of their goal mouth, / L
    "goal_dir_x",  # unit vector player -> their goal centre
    "goal_dir_y",
    "post_lo_sin",  # y part of the unit vector player -> their post at GOAL_Y_MIN
    "post_hi_sin",  # ... and to their post at GOAL_Y_MAX
    "own_goal_dist",  # player -> centre of our goal mouth, / L
    "own_goal_dir_x",
    "own_goal_dir_y",
    "ball_goal_dist",  # ball -> their goal centre, / L
    "ball_goal_dir_x",
    "ball_goal_dir_y",
    "ball_speed",  # |ball velocity| / MAX_BALL_VELOCITY
    "ball_speed_to_goal",  # ball velocity . ball_goal_dir / MAX_BALL_VELOCITY
    "ball_to_their_mouth",  # 1 if going straight on it crosses their goal line in the mouth
    "ball_to_our_mouth",  # 1 if going straight on it crosses our goal line in the mouth
    "ball_cross_y",  # where it crosses the line it heads for, (y - H/2) / (H/2), clipped
    "ball_line_ticks",  # ticks until it reaches that line, capped at PATH_TICKS, / PATH_TICKS
    "closing_speed",  # (player vel - ball vel) . unit(player -> ball) / (max speeds summed)
    "reach_ticks",  # (distance to touching the ball) / MAX_PLAYER_VELOCITY, capped, scaled
    "behind_ball_cos",  # unit(player -> ball) . unit(ball -> their goal): 1 = right behind it
    "behind_ball_sin",  # the cross product of the same two unit vectors
    "opp_ball_dist",  # nearest opponent to the ball, min(distance, PRESSURE_SCALE) / scale
    "opp_closing_speed",  # that opponent's closing speed on the ball, as closing_speed
    "shot_lane",  # nearest opponent to the segment ball -> their goal centre, capped, scaled
)
# ---------------------------------------------------------------- observation v3 (raw)
#
# Obs v3 is the brain interface itself, scaled but otherwise untouched: nothing relative,
# sorted, counted or computed. Each player's row is which player it is (one-hot: all five
# share one network, so each needs to know which output it is), then every position and
# velocity in the order the game gives them (ours, theirs, the ball), both scores and the
# game time. Positions / FIELD * 2 - 1, velocities / their maximum, scores / 10.
RAW_DIM = NUM + 4 * NUM + 4 * NUM + 4 + 3  # 52


def batch_raw_features(my_pos, my_vel, opp_pos, opp_vel, ball_pos, ball_vel, my_score, opp_score, game_time):
    """Obs v3 for K views at once: (K, 5, 2) players, (K, 2) ball, (K,) scores, scalar or (K,) time -> (K, 5, RAW_DIM)."""
    my_pos, my_vel, opp_pos, opp_vel = (np.asarray(a, dtype=float) for a in (my_pos, my_vel, opp_pos, opp_vel))
    k = my_pos.shape[0]
    field = np.array([L, H], dtype=float)
    team = np.concatenate([
        (my_pos / field * 2 - 1).reshape(k, -1),
        (my_vel / Constants.MAX_PLAYER_VELOCITY).reshape(k, -1),
        (opp_pos / field * 2 - 1).reshape(k, -1),
        (opp_vel / Constants.MAX_PLAYER_VELOCITY).reshape(k, -1),
        np.asarray(ball_pos, dtype=float).reshape(k, 2) / field * 2 - 1,
        np.asarray(ball_vel, dtype=float).reshape(k, 2) / Constants.MAX_BALL_VELOCITY,
        np.broadcast_to(np.asarray(my_score, dtype=float), (k,))[:, None] / 10.0,
        np.broadcast_to(np.asarray(opp_score, dtype=float), (k,))[:, None] / 10.0,
        np.broadcast_to(np.asarray(game_time, dtype=float), (k,))[:, None],
    ], axis=1)
    out = np.empty((k, NUM, RAW_DIM))
    out[:, :, :NUM] = np.eye(NUM)[None]
    out[:, :, NUM:] = team[:, None, :]
    return out


OBS_DIMS = {1: OBS_DIM, 2: OBS_DIM + len(OBS_V2_NAMES), 3: RAW_DIM}
TIME_WINDOW = 1800  # ticks (one training game): time-left cap and score-trend window
PATH_TICKS = 120.0  # cap on ball_line_ticks (a full-speed ball crosses the field in 120)
REACH_TICKS = 240.0  # cap on reach_ticks (a full-speed player crosses the field in 240)
PRESSURE_SCALE = 500.0  # px
LANE_SCALE = 200.0  # px
GOAL_X = L - 1.0 - Constants.GOAL_DEPTH  # their goal line (they always defend the right)
OWN_GOAL_X = float(Constants.GOAL_DEPTH)  # ours; the two mirror each other exactly
# The points extra_features measures each player against: their goal centre, their posts
# (GOAL_Y_MIN, GOAL_Y_MAX) and our goal centre.
TARGETS_X = np.array([GOAL_X, GOAL_X, GOAL_X, OWN_GOAL_X])
TARGETS_Y = np.array([H / 2, Constants.GOAL_Y_MIN, Constants.GOAL_Y_MAX, H / 2])
NEVER = np.iinfo(np.int64).max


def _unit(dx, dy):
    """Length and unit vector of (dx, dy); a zero vector gives (0, 0)."""
    d = np.sqrt(dx * dx + dy * dy)
    safe = np.where(d > 0, d, 1.0)
    return d, dx / safe, dy / safe


def _clip(x, lo, hi):
    """np.clip, the same values (for numbers, not NaN) at a fraction of its call overhead."""
    return np.minimum(np.maximum(x, lo), hi)


def time_left(ticks, game_time):
    """
    Obs v2's time_left for (K,) tick counts and game times: min(ticks left,
    TIME_WINDOW) / TIME_WINDOW, so it means the same in 1800-tick training games and
    36000-tick league matches (1 until the last TIME_WINDOW ticks, then down to 0).

    Brains are told game_time = tick / game length, not the length, so the length is
    recovered from the brain's own tick count: round(tick / game_time) (exact: game_time
    is one division away from it). At tick 0, and in endless games (game_time always 0),
    the time left counts as TIME_WINDOW or more.
    """
    ticks = np.asarray(ticks, dtype=float)
    game_time = np.asarray(game_time, dtype=float)
    started = game_time > 0
    length = np.rint(ticks / np.where(started, game_time, 1.0))
    left = np.where(started, length - ticks, float(TIME_WINDOW))
    return _clip(left, 0.0, float(TIME_WINDOW)) / TIME_WINDOW


class ScoreTrend:
    """
    Obs v2's score trend for K seats: the goal difference now minus the goal difference
    TIME_WINDOW ticks ago (score_trend clips it to +-3). A change in the difference is
    noted at the decision that first sees it; a goal always forces a decision at the
    kick-off, so that is the tick after the goal, the same in Game and VecGame.
    """

    def __init__(self, k=1):
        self.last = np.zeros(k, dtype=np.int64)  # difference at the last update
        self.base = np.zeros(k, dtype=np.int64)  # difference TIME_WINDOW ticks ago
        self.pending = [deque() for _ in range(k)]  # (tick, difference) still in the window
        self.expires = np.full(k, NEVER)  # when the oldest pending change leaves it

    def update(self, seats, tick, diff):
        """
        The raw trend (ints) of the given seats (indices) at ``tick`` (scalar or per
        seat), whose goal differences are now ``diff``.
        """
        seats = np.asarray(seats, dtype=int)
        diff = np.asarray(diff, dtype=np.int64)
        tick = np.broadcast_to(np.asarray(tick, dtype=np.int64), seats.shape)
        for i in np.flatnonzero(diff != self.last[seats]):  # goals: rare
            s = seats[i]
            if not self.pending[s]:
                self.expires[s] = tick[i] + TIME_WINDOW
            self.pending[s].append((int(tick[i]), int(diff[i])))
        self.last[seats] = diff
        for i in np.flatnonzero(self.expires[seats] <= tick):
            s, queue = seats[i], self.pending[seats[i]]
            while queue and queue[0][0] + TIME_WINDOW <= tick[i]:
                self.base[s] = queue.popleft()[1]
            self.expires[s] = queue[0][0] + TIME_WINDOW if queue else NEVER
        return diff - self.base[seats]


def extra_features(
    my_pos, my_vel, opp_pos, opp_vel, ball_pos, ball_vel, time_left_now, trend
):
    """
    The OBS_V2_NAMES features for K views at once: (K, n, 2) players, (K, 2) ball, (K,)
    time_left values and (K,) raw score trends give (K, n, len(OBS_V2_NAMES)).

    Every operation is elementwise, or a min / argmin over one view's opponents, so a
    view gets exactly the same values alone as in a batch (PPOBrain and BatchedPPO both
    call this). Each value is roughly in -1 .. 1 (see OBS_V2_NAMES).
    """
    vp, vb = Constants.MAX_PLAYER_VELOCITY, Constants.MAX_BALL_VELOCITY
    k, n = my_pos.shape[:2]
    out = np.empty((k, n, len(OBS_V2_NAMES)))
    px, py = my_pos[..., 0], my_pos[..., 1]
    vx, vy = my_vel[..., 0], my_vel[..., 1]
    bx, by = ball_pos[:, 0:1], ball_pos[:, 1:2]  # (K, 1): the same for every player
    bvx, bvy = ball_vel[:, 0:1], ball_vel[:, 1:2]

    out[..., 0] = np.asarray(time_left_now, dtype=float)[:, None]
    out[..., 1] = (_clip(np.asarray(trend), -3, 3) / 3.0)[:, None]

    # The player and the goals: their goal's centre and posts, and our goal's centre.
    d, ux, uy = _unit(TARGETS_X - px[..., None], TARGETS_Y - py[..., None])  # (K, n, 4)
    out[..., 2], out[..., 3], out[..., 4] = d[..., 0] / L, ux[..., 0], uy[..., 0]
    out[..., 5:7] = uy[..., 1:3]
    out[..., 7], out[..., 8], out[..., 9] = d[..., 3] / L, ux[..., 3], uy[..., 3]

    # The ball and their goal, and the ball's straight-line path to a goal line.
    goal_d, gx, gy = _unit(GOAL_X - bx, H / 2 - by)
    out[..., 10], out[..., 11], out[..., 12] = goal_d / L, gx, gy
    out[..., 13] = np.sqrt(bvx * bvx + bvy * bvy) / vb
    out[..., 14] = (bvx * gx + bvy * gy) / vb
    forward, backward = bvx > 1e-6, bvx < -1e-6  # heading for their line / ours
    moving = forward | backward
    line_x = np.where(backward, OWN_GOAL_X, GOAL_X)
    ticks = (line_x - bx) / np.where(moving, bvx, 1.0)
    y_cross = by + bvy * ticks
    mouth = (Constants.GOAL_Y_MIN < y_cross) & (y_cross < Constants.GOAL_Y_MAX)
    out[..., 15] = forward & mouth
    out[..., 16] = backward & mouth
    out[..., 17] = np.where(moving, _clip((y_cross - H / 2) / (H / 2), -1.0, 1.0), 0.0)
    out[..., 18] = np.where(moving, _clip(ticks, 0.0, PATH_TICKS), PATH_TICKS) / PATH_TICKS

    # The player and the ball.
    d, ux, uy = _unit(bx - px, by - py)
    out[..., 19] = ((vx - bvx) * ux + (vy - bvy) * uy) / (vp + vb)
    reach = d - (Constants.PLAYER_RADIUS + Constants.BALL_RADIUS)
    out[..., 20] = _clip(reach / vp, 0.0, REACH_TICKS) / REACH_TICKS
    out[..., 21] = ux * gx + uy * gy
    out[..., 22] = ux * gy - uy * gx

    # Pressure: their nearest player to the ball, and their players near the shot lane.
    ox, oy = opp_pos[..., 0], opp_pos[..., 1]
    od, oux, ouy = _unit(bx - ox, by - oy)  # (K, 5)
    j = np.argmin(od, axis=1)
    rows = np.arange(k)
    out[..., 23] = (np.minimum(od[rows, j], PRESSURE_SCALE) / PRESSURE_SCALE)[:, None]
    ovx, ovy = opp_vel[rows, j, 0], opp_vel[rows, j, 1]
    closing = (ovx - bvx[:, 0]) * oux[rows, j] + (ovy - bvy[:, 0]) * ouy[rows, j]
    out[..., 24] = (closing / (vp + vb))[:, None]
    sx, sy = GOAL_X - bx, H / 2 - by
    length_sq = sx * sx + sy * sy
    ax, ay = ox - bx, oy - by
    s = _clip((ax * sx + ay * sy) / np.where(length_sq > 0, length_sq, 1.0), 0.0, 1.0)
    ex, ey = ax - s * sx, ay - s * sy
    lane = np.sqrt(ex * ex + ey * ey).min(axis=1)
    out[..., 25] = (np.minimum(lane, LANE_SCALE) / LANE_SCALE)[:, None]
    return out


class ObsMemory:
    """
    What obs v2 remembers within a game for one team: its tick count (``tick``: ticks
    seen before the current one, which is the game's tick) and the score trend. A game
    clock that goes backwards means a new game: the memory starts again.
    """

    def __init__(self):
        self.new_game()

    def new_game(self):
        self.tick = 0
        self.last_time = -np.inf
        self.trend = ScoreTrend(1)

    def start_tick(self, game_time):
        """Call every tick, before any decision."""
        if game_time < self.last_time:
            self.new_game()
        self.last_time = game_time

    def end_tick(self):
        self.tick += 1


def observation(view, version=1, memory=None):
    """
    The observation rows (5 x OBS_DIMS[version]) of a brain's view (the arguments of
    AbstractBrain.move, in order). Version 2 needs the team's ObsMemory, at this tick.
    """
    if version == 3:
        my_pos, my_vel, opp_pos, opp_vel, ball_pos, ball_vel, my_score, opp_score, game_time = view
        return batch_raw_features(
            np.asarray(my_pos, dtype=float)[None], np.asarray(my_vel, dtype=float)[None],
            np.asarray(opp_pos, dtype=float)[None], np.asarray(opp_vel, dtype=float)[None],
            np.asarray(ball_pos, dtype=float)[None], np.asarray(ball_vel, dtype=float)[None],
            my_score, opp_score, game_time,
        )[0]
    obs = player_features(*view)
    if version == 1:
        return obs
    my_pos, my_vel, opp_pos, opp_vel, ball_pos, ball_vel, my_score, opp_score, game_time = view
    extra = extra_features(
        *(np.asarray(a, dtype=float)[None] for a in (my_pos, my_vel, opp_pos, opp_vel)),
        np.asarray(ball_pos, dtype=float).reshape(1, 2),
        np.asarray(ball_vel, dtype=float).reshape(1, 2),
        time_left(np.array([memory.tick]), np.array([game_time], dtype=float)),
        memory.trend.update(np.zeros(1, dtype=int), memory.tick, [my_score - opp_score]),
    )
    return np.concatenate([obs, extra[0]], axis=1)


def obs_version_of(weights):
    """
    The observation version a weights dict's networks read: its "obs_version" (checked
    against the policy's input size), else the version whose width matches the widest
    policy input (files from before obs v2 are all 54 wide: version 1). A mixed team
    (role_policies) is built at its widest version; narrower role networks read the
    leading columns, which are exactly the older version's features.
    """
    nets = [weights["policy"]] + list(weights.get("role_policies") or [])
    width = max(np.shape(net[0])[0] for net in nets)
    version = weights.get("obs_version")
    if version is None:
        return next((v for v, dim in OBS_DIMS.items() if dim == width), 1)
    version = int(version)
    if OBS_DIMS.get(version) != width:
        raise ValueError(f"obs_version {version} but the policy reads {width} inputs")
    return version


def widen_inputs(weights, version):
    """
    A copy of a weights dict whose networks read OBS_DIMS[version] inputs and act as
    before: the first layers get zero weights for the added features (outputs equal up
    to float rounding), so e.g. a v1 champion can start an obs-v2 learner.
    """
    width = OBS_DIMS[version]

    def pad(params):
        params = [np.asarray(p, dtype=float) for p in params]
        w = params[0]
        if w.shape[0] > width:
            raise ValueError(f"a {w.shape[0]}-input network cannot read {width} inputs")
        extra = np.zeros((width - w.shape[0], w.shape[1]))
        return [np.concatenate([w, extra])] + params[1:]

    out = dict(weights, obs_version=version)
    for key in ("policy", "value"):
        if key in weights:
            out[key] = pad(weights[key])
    for key in ("role_policies", "role_values"):
        if weights.get(key):
            out[key] = [pad(p) for p in weights[key]]
    return out


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

    A reward with a "chance" weight adds that times ``chance_potential``, using the xG
    model in reward["xg_model"] (a load_xg_model dict) if there is one.
    """
    terms = potential_terms(my_pos, opp_pos, ball_pos, ball_vel)
    total = sum(reward.get(name, 0.0) * value for name, value in terms.items())
    if reward.get("chance"):
        total = total + reward["chance"] * chance_potential(
            my_pos, opp_pos, ball_pos, ball_vel, reward.get("xg_model")
        )
    return total


# The "chance" reward term: the chance to score of the ball's current path (see
# chance_potential). The potential is our chance minus theirs, so creating a chance, or
# stopping theirs, is rewarded when it happens and given back when the ball is touched
# again or goes in (then the goal reward pays).
CHANCE_DISTANCE = 450.0  # px: the hand-made chance of a full-speed ball this far out is 1/e
# A defender whose centre is this close to the ball's path can put its body in the way
# with a step of one player radius.
PATH_REACH = 2 * Constants.PLAYER_RADIUS + Constants.BALL_RADIUS
XG_FEATURES = ("distance", "offset", "speed", "blockers", "keeper")


def shot_features(ball_pos, ball_vel, defenders):
    """
    Whether each ball is on target for the goal on the right, and that chance's
    features: (K, 2) ball positions and velocities and the (K, 5, 2) positions of the
    defending players give an on-target mask (K,) and features (K, 5) in XG_FEATURES
    order:

    - distance: px from the ball to the goal line (0 once over it),
    - offset: px from the middle of the goal mouth to where the ball crosses the line,
    - speed: the ball's speed towards the goal line, px per tick,
    - blockers: defenders ahead of the ball within PATH_REACH of its path to the line,
    - keeper: px from the crossing point to the nearest defender.

    On target: moving towards the goal line and, going straight on (the ball has no
    friction), crossing it inside the goal mouth, directly or after one bounce off the
    top or bottom wall. Posts and players are ignored here (blockers counts players).
    Features of balls that are not on target are finite but meaningless. Every
    operation is elementwise or over one row's players, so a ball gets exactly the same
    values alone as in a batch.
    """
    ball_pos = np.asarray(ball_pos, dtype=float)
    ball_vel = np.asarray(ball_vel, dtype=float)
    defenders = np.asarray(defenders, dtype=float)
    bx, by = ball_pos[:, 0], ball_pos[:, 1]
    vx, vy = ball_vel[:, 0], ball_vel[:, 1]
    goal_line = L - 1 - Constants.GOAL_DEPTH
    lo = float(Constants.BALL_RADIUS)  # the ball's centre bounces off the walls here
    hi = H - float(Constants.BALL_RADIUS)

    forward = vx > 0
    dx = goal_line - bx
    y_line = by + vy * dx / np.where(forward, vx, 1.0)  # where it meets the line, no walls
    below, above = y_line < lo, y_line > hi
    y_cross = np.where(below, 2 * lo - y_line, np.where(above, 2 * hi - y_line, y_line))
    on_target = (
        forward & (Constants.GOAL_Y_MIN < y_cross) & (y_cross < Constants.GOAL_Y_MAX)
    )

    # The path unfolded at the wall: a straight segment from the ball to (goal_line,
    # y_line). For a path that bounces, a defender's mirror image in that wall stands
    # where the defender is relative to the path after the bounce.
    sx, sy = dx[:, None], (y_line - by)[:, None]
    length_sq = sx * sx + sy * sy
    length_sq = np.where(length_sq > 0, length_sq, 1.0)
    px, py = defenders[..., 0], defenders[..., 1]
    blocked = np.zeros(px.shape, dtype=bool)
    for image_y, used in (
        (py, True),
        (2 * lo - py, below[:, None]),
        (2 * hi - py, above[:, None]),
    ):
        rx, ry = px - bx[:, None], image_y - by[:, None]
        along = rx * sx + ry * sy
        t = np.clip(along / length_sq, 0.0, 1.0)
        ex, ey = rx - t * sx, ry - t * sy
        blocked |= used & (along > 0) & (ex * ex + ey * ey < PATH_REACH * PATH_REACH)
    blockers = blocked.sum(axis=-1).astype(float)

    kx, ky = px - goal_line, py - y_cross[:, None]
    keeper = np.sqrt(kx * kx + ky * ky).min(axis=-1)

    features = np.stack(
        [
            np.maximum(dx, 0.0),
            np.abs(y_cross - H / 2),
            np.where(forward, vx, 0.0),
            blockers,
            keeper,
        ],
        axis=-1,
    )
    return on_target, features


def hand_chance(features):
    """
    The hand-made chance of on-target balls (the fallback when there is no xG model):
    exp(-d / CHANCE_DISTANCE) where d is the distance to the line scaled by how much
    slower than full speed the ball approaches it (so d / MAX_BALL_VELOCITY is the time
    to the line), halved for every blocker. A deflection near the goal is a bigger
    chance than a long shot, a strike a bigger one than a slow roll.
    """
    distance, speed, blockers = features[:, 0], features[:, 2], features[:, 3]
    scaled = distance * Constants.MAX_BALL_VELOCITY / np.where(speed > 0, speed, 1.0)
    return np.exp(-scaled / CHANCE_DISTANCE) * 0.5**blockers


def xg_column(features, name):
    """
    One input of the xG model from shot_features' columns: a feature of XG_FEATURES, or
    one of XG_LOG_FEATURES: log_distance = log(1 + distance), log_speed = log(speed)
    (at least 0.1 px/tick). On a log scale, doubling the distance or the speed changes
    the odds by the same factor wherever it starts.
    """
    if name == "log_distance":
        return np.log1p(features[:, 0])
    if name == "log_speed":
        return np.log(np.maximum(features[:, 2], 0.1))
    return features[:, XG_FEATURES.index(name)]


XG_LOG_FEATURES = ("log_distance", "log_speed")


def xg_probability(features, model):
    """
    The xG model's (fit_xg.py) probability that each on-target ball goes in before it is
    touched again: a logistic regression on the standardised features it names.
    """
    z = np.full(len(features), float(model["intercept"]))
    for name, mean, scale, coef in zip(
        model["features"], model["mean"], model["scale"], model["coef"]
    ):
        x = xg_column(features, name)
        z = z + float(coef) * ((x - float(mean)) / float(scale))
    with np.errstate(over="ignore", under="ignore"):
        return 1.0 / (1.0 + np.exp(-z))


def chance_values(ball_pos, ball_vel, defenders, model=None):
    """The chance to score in the goal on the right: 0 off target, else xG (or hand-made)."""
    on_target, features = shot_features(ball_pos, ball_vel, defenders)
    value = hand_chance(features) if model is None else xg_probability(features, model)
    return np.where(on_target, value, 0.0)


def batch_chance(my_pos, opp_pos, ball_pos, ball_vel, model=None):
    """
    The "chance" potential for K views at once (see chance_potential): (K, 5, 2)
    players and (K, 2) ball give (K,).
    """
    my_pos = np.asarray(my_pos, dtype=float)
    ball_pos = np.asarray(ball_pos, dtype=float)
    ball_vel = np.asarray(ball_vel, dtype=float)
    ours = chance_values(ball_pos, ball_vel, opp_pos, model)
    # Theirs: the same, mirrored (flip_pos, flip_vel) so that they attack to the right.
    ball_m, vel_m, my_m = ball_pos.copy(), ball_vel.copy(), my_pos.copy()
    ball_m[:, 0] = L - 1 - ball_m[:, 0]
    vel_m[:, 0] = -vel_m[:, 0]
    my_m[..., 0] = L - 1 - my_m[..., 0]
    theirs = chance_values(ball_m, vel_m, my_m, model)
    return ours - theirs


def chance_potential(my_pos, opp_pos, ball_pos, ball_vel=(0.0, 0.0), model=None):
    """
    The potential of the "chance" reward term: our chance to score minus theirs, each in
    0 .. 1. A chance is 0 unless the ball's path is on target (see shot_features); on
    target it is the xG model's probability that the ball goes in before anyone touches
    it (``model``, see load_xg_model), or without a model the hand-made ``hand_chance``.
    """
    return float(
        batch_chance(
            np.asarray(my_pos, dtype=float)[None],
            np.asarray(opp_pos, dtype=float)[None],
            np.asarray(ball_pos, dtype=float)[None],
            np.asarray(ball_vel, dtype=float)[None],
            model,
        )[0]
    )


def load_xg_model(path):
    """
    The xG model fit_xg.py wrote to ``path`` (JSON) as a plain dict for chance_potential,
    or None if there is no such file (then the chance is the hand-made one).
    """
    if not path or not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text())
    model = {k: data[k] for k in ("features", "mean", "scale", "coef", "intercept")}
    unknown = set(model["features"]) - set(XG_FEATURES) - set(XG_LOG_FEATURES)
    if unknown:
        raise ValueError(f"{path}: unknown xG features {sorted(unknown)}")
    return model


def inputs(obs, net):
    """
    The observation columns a network reads: all of them, or for a network of an older
    observation version in a mixed team, the leading ones (contiguous, so it multiplies
    exactly as a narrower observation would).
    """
    width = net.params[0].shape[0]
    if obs.shape[-1] == width:
        return obs
    return np.ascontiguousarray(obs[..., :width])


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

    The network's inputs are the observation version its weights record (obs_version,
    see OBS_DIMS; files without one are version 1, the 54 player_features).
    """

    ACTION_REPEAT = 2
    # Exploration noise held for this many decisions before a fresh sample (1: a fresh sample
    # every decision, as always). Holding it keeps decisions every tick but makes the
    # exploration coherent over a few ticks, so it is not averaged away (raw-PPO, 2026-10-01).
    NOISE_HOLD = 1
    WEIGHTS_FILE = Path(__file__).parent / "weights" / "PPOBrain.npz"

    def __init__(
        self, name=None, weights=None, deterministic=True, training=False, reward=None
    ):
        """
        :param weights: dict with "policy" (and for training "value") parameter lists
            and "log_std". Defaults to the saved weights file. A mixed team (see
            coach.py) also has "role_policies": one policy parameter list per player
            role, so each player can come from a different trained brain. For training
            per-role networks (train_roles.py) it may also have "role_values", and
            "log_std" may have one row per role (5 x 2).
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
        # The observation the networks read (obs_version_of) and its width.
        self.obs_version = 1 if weights is None else obs_version_of(weights)
        self.obs_dim = OBS_DIMS.get(self.obs_version, OBS_DIM)
        self.memory = ObsMemory() if self.obs_version == 2 else None
        self.last_obs = None  # the latest decision's observation rows
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
        self.role_values = None
        if training:
            self.value = network(weights["value"])
            if weights.get("role_values"):  # one value network per role (train_roles.py)
                self.role_values = [network(params) for params in weights["role_values"]]
        self.deterministic = deterministic and not training
        self.training = training
        self.reward = {**DEFAULT_REWARD, **(reward or {})}
        self.ticks_until_decision = 0
        self.noise_hold = max(int(PPOBrain.NOISE_HOLD), 1)
        self._noise, self._noise_age = None, 0
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
        if "obs_version" in data.files:
            weights["obs_version"] = int(data["obs_version"])
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
        version = obs_version_of(weights)  # v1 files are saved exactly as before
        if version != 1:
            arrays["obs_version"] = np.array(version)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

    def do_move(self):
        if self.memory is not None:
            self.memory.start_tick(self.game_time)
        if self.ticks_until_decision <= 0:
            self.decide()
            self.ticks_until_decision = self.action_repeat
        self.ticks_until_decision -= 1
        if self.memory is not None:
            self.memory.end_tick()
        return self.action.copy()

    def features(self):
        """This decision's observation rows, at the brain's observation version."""
        view = (
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
        if self.obs_version == 3:
            return observation(view, 3)
        if self.memory is None:
            return player_features(*view)
        return observation(view, self.obs_version, self.memory)

    def decide(self):
        obs = self.last_obs = self.features()
        if self.role_policies:
            mean = np.stack(
                [net(inputs(obs[[r]], net))[0] for r, net in enumerate(self.role_policies)]
            )
        else:
            mean = self.policy(inputs(obs, self.policy))
        if self.deterministic:
            self.action = mean
            return

        action = mean + np.exp(self.log_std) * self.exploration_noise(mean.shape)
        self.action = action
        if self.training:
            t = self.trajectory
            t["obs"].append(obs)
            t["act"].append(action)
            t["logp"].append(gaussian_log_prob(action, mean, self.log_std))
            if self.role_values:
                t["val"].append(
                    np.array(
                        [net(inputs(obs[[r]], net))[0, 0] for r, net in enumerate(self.role_values)]
                    )
                )
            else:
                t["val"].append(self.value(inputs(obs, self.value))[:, 0])
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

    def exploration_noise(self, shape=(NUM, ACT_DIM)):
        """This decision's standard normal exploration noise: a fresh sample every noise_hold decisions."""
        if self._noise is None or self._noise_age >= self.noise_hold or self._noise.shape != tuple(shape):
            self._noise, self._noise_age = self.rng.standard_normal(shape), 0
        self._noise_age += 1
        return self._noise

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
