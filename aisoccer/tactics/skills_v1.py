"""
Player skills, version 1 (frozen; see aisoccer/tactics/__init__.py), built on the
game's own physics. Nothing here changes the rules: it only predicts the ball and
chooses accelerations.

Everything is in a brain's own frame: it defends x = 0 and attacks x = FIELD_LENGTH - 1.
All functions return an acceleration request for one player; the game caps its
magnitude at 1 and player speed at MAX_PLAYER_VELOCITY.

How the ball moves (aisoccer/physics.py):

- There is no friction: the ball keeps its velocity until it hits a wall, a post or a
  player. Walls and posts reflect it.
- When a player hits the ball, the ball's velocity changes by
  KICK_FACTOR * ((v_player - v_ball) . n) n, where n is the unit vector from the
  player's centre to the ball's. So the kick direction is set by where the player
  strikes the ball, and the power by how fast it is moving along that line.

Speed: these run every tick for several players, so the hot ones use plain floats for
scalar arithmetic and work on all players at once where they can. Such changes must
give bit-identical results (a version never changes how it plays): element-wise numpy
arithmetic and Python floats are the same IEEE operations, but a matrix product's
rounding can depend on the shape of its arguments, so those are kept as they were.
"""

import math

import numpy as np

from aisoccer.constants import Constants

L = Constants.FIELD_LENGTH - 1
H = Constants.FIELD_HEIGHT
BALL_R = Constants.BALL_RADIUS
PLAYER_R = Constants.PLAYER_RADIUS
CONTACT = BALL_R + PLAYER_R
MAX_SPEED = Constants.MAX_PLAYER_VELOCITY
MAX_BALL_SPEED = Constants.MAX_BALL_VELOCITY
GOAL_LINE = L - Constants.GOAL_DEPTH
OWN_GOAL_LINE = Constants.GOAL_DEPTH
GOAL_Y_MIN = Constants.GOAL_Y_MIN
GOAL_Y_MAX = Constants.GOAL_Y_MAX
GOAL_CENTRE = np.array([L, H / 2])
OWN_GOAL_CENTRE = np.array([0.0, H / 2])

_player_mass = PLAYER_R**2
_ball_mass = BALL_R**2
KICK_FACTOR = 2 * _player_mass / (_player_mass + _ball_mass)  # about 1.8


def length(dx, dy):
    """Length of the vector (dx, dy) in plain floats: np.linalg.norm's result, bit for bit."""
    return math.sqrt(dx * dx + dy * dy)


def unit(v):
    n = length(*np.asarray(v, dtype=float).tolist())
    return v / n if n > 1e-9 else np.zeros(2)


def norms(v):
    """Lengths of the 2-vectors along v's last axis: np.linalg.norm(v, axis=-1), bit for bit."""
    return np.sqrt(v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1])


# ---------------------------------------------------------------- prediction


def predict_ball(pos, vel, ticks):
    """
    Ball positions for the next `ticks` ticks (row t = after t+1 ticks), bouncing off
    the side walls and the goal lines outside the goal mouths. Players are ignored.
    """
    # Plain floats: the same arithmetic as 2-vectors, without numpy's per-call cost.
    x, y = np.asarray(pos, dtype=float).tolist()
    vx, vy = np.asarray(vel, dtype=float).tolist()
    xs, ys = [0.0] * ticks, [0.0] * ticks
    for t in range(ticks):
        in_mouth = GOAL_Y_MIN < y < GOAL_Y_MAX
        min_x = (0.0 if in_mouth else OWN_GOAL_LINE) + BALL_R
        max_x = (L if in_mouth else GOAL_LINE) - BALL_R
        if (x < min_x and vx < 0) or (x > max_x and vx > 0):
            vx = -vx
        if (y < BALL_R and vy < 0) or (y > H - BALL_R and vy > 0):
            vy = -vy
        x += vx
        y += vy
        xs[t], ys[t] = x, y
    out = np.empty((ticks, 2))
    out[:, 0], out[:, 1] = xs, ys
    return out


def time_to_reach(pos, vel, target):
    """Rough ticks for a player to reach `target` (accelerating at 1, capped speed)."""
    offset = np.asarray(target, dtype=float) - pos
    d = np.linalg.norm(offset)
    if d < 1e-6:
        return 0.0
    direction = offset / d
    v0 = float(np.dot(vel, direction))
    sideways = np.linalg.norm(vel - v0 * direction)
    t_full = max(MAX_SPEED - v0, 0.0)  # ticks to reach full speed
    d_full = (v0 + MAX_SPEED) / 2 * t_full
    if d <= d_full:
        t = -v0 + math.sqrt(max(v0 * v0 + 2 * d, 0.0))
    else:
        t = t_full + (d - d_full) / MAX_SPEED
    return t + sideways  # time spent cancelling sideways motion


def times_to_reach(pos, vel, targets):
    """Vectorised time_to_reach for many targets (an N x 2 array)."""
    pos = np.asarray(pos, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return _times_to_reach(pos[None], [np.asarray(vel, dtype=float)], targets[None])[0]


def _times_to_reach(positions, velocities, targets):
    """
    times_to_reach for several players at once: row k of `positions` (P x 2) and
    velocities[k] (a list of P 2-vectors) with the targets targets[k] (P x N x 2).
    """
    offset = targets - positions[:, None, :]
    d = norms(offset)
    direction = offset / np.maximum(d, 1e-9)[..., None]
    # One matrix-vector product per player, as for a single player: its rounding can
    # depend on the arrays' shapes, so a single product for everyone could differ.
    # (Everything else is element-wise, so the same whether done at once or apart.)
    v0 = np.array([direction[k] @ vel for k, vel in enumerate(velocities)])
    sideways = norms(np.asarray(velocities)[:, None, :] - v0[..., None] * direction)
    t_full = np.maximum(MAX_SPEED - v0, 0.0)
    d_full = (v0 + MAX_SPEED) / 2 * t_full
    accelerating = -v0 + np.sqrt(np.maximum(v0 * v0 + 2 * d, 0.0))
    cruising = t_full + (d - d_full) / MAX_SPEED
    return np.where(d <= d_full, accelerating, cruising) + sideways


def intercept(pos, vel, ball_path, margin=0.0):
    """
    Earliest tick (index into ball_path) at which the player can be touching the
    ball, and the ball's position then. Falls back to the end of the path.
    """
    t = intercepts([pos], [vel], ball_path, margin)[0]
    return t, ball_path[t]


def intercepts(positions, velocities, ball_path, margin=0.0):
    """intercept's tick for several players at once (row k of positions and velocities)."""
    positions = np.asarray(positions, dtype=float)
    velocities = [np.asarray(vel, dtype=float) for vel in velocities]
    offsets = ball_path - positions[:, None, :]
    dist = norms(offsets)
    reach_points = ball_path - offsets / np.maximum(dist, 1e-9)[..., None] * (
        CONTACT + margin
    )
    ok = _times_to_reach(positions, velocities, reach_points) <= np.arange(
        1, len(ball_path) + 1
    )
    first, found = np.argmax(ok, axis=1).tolist(), ok.any(axis=1).tolist()
    return [t if f else len(ball_path) - 1 for t, f in zip(first, found)]


# ---------------------------------------------------------------- movement


def move_to(pos, vel, target, arrive_speed=0.0):
    """
    Accelerate towards `target`, braking so as to arrive at about `arrive_speed`
    instead of overshooting (with acceleration 1, speed sqrt(2d) stops in d).
    """
    offset = np.asarray(target, dtype=float) - pos
    d = length(*offset.tolist())
    if d < 1.0:
        return -vel  # stop
    speed = min(MAX_SPEED, math.sqrt(2 * d + arrive_speed**2))
    return offset / d * speed - vel


def run_through(pos, vel, direction, speed=MAX_SPEED):
    """Accelerate so the velocity becomes `speed` along `direction`."""
    return unit(direction) * speed - vel


# ---------------------------------------------------------------- ball skills


def kick(pos, vel, ball, ball_vel, direction, power=1.0, ball_path=None, ball_then=None):
    """
    Send the ball in `direction` at up to `power` x full strength.

    The player's centre must meet the ball at the contact point behind it (the ball
    leaves along the line through both centres). From roughly behind the ball (within
    about 60 degrees of the kick direction) the player drives straight through the
    contact point; a small angle only bends the kick a little. From the wrong side it
    runs round the ball first. The approach speed sets the kick's strength.

    `ball_then`, if known, is where the player meets the ball (intercept's position).
    """
    d = unit(direction)
    if ball_then is None:
        if ball_path is None:
            ball_path = predict_ball(ball, ball_vel, 60)
        _, ball_then = intercept(pos, vel, ball_path)
    ball_then = np.asarray(ball_then)
    contact = ball_then - d * CONTACT
    speed = float(
        np.clip(
            (power * MAX_BALL_SPEED + float(np.dot(ball_vel, d))) / KICK_FACTOR,
            1.5,
            MAX_SPEED,
        )
    )

    approach = unit(ball_then - pos)
    if np.dot(approach, d) > 0.5:
        # Behind the ball: aim a little past the contact point so we arrive moving.
        return move_to(pos, vel, contact + d * 10, arrive_speed=speed)

    # Wrong side: run round the ball on the nearer side, then come back through it.
    side = np.array([-d[1], d[0]])
    if np.dot(pos - ball_then, side) < 0:
        side = -side
    waypoint = ball_then + side * (CONTACT + 25) - d * (CONTACT + 15)
    return move_to(pos, vel, waypoint, arrive_speed=4.0)


def tackle(pos, vel, ball, ball_vel, ball_path=None, ball_then=None):
    """
    Win a contested ball: run straight at it and knock it towards the opponent's goal.
    `ball_then`, if known, is where the player meets the ball (intercept's position).
    """
    if ball_then is None:
        if ball_path is None:
            ball_path = predict_ball(ball, ball_vel, 60)
        _, ball_then = intercept(pos, vel, ball_path)
    away = unit(np.asarray(ball_then) - OWN_GOAL_CENTRE)
    return move_to(
        pos, vel, np.asarray(ball_then) - away * (CONTACT - 6), arrive_speed=MAX_SPEED
    )


def shot_direction(ball, target_y=None, opponents=None):
    """Direction from the ball to the best point in the opponent's goal mouth."""
    if target_y is None:
        target_y = open_goal_y(ball, opponents)
    target = np.array([L + 10, target_y])
    return unit(target - ball)


def open_goal_y(ball, opponents=None, samples=9):
    """The point in the goal mouth whose shooting line passes furthest from opponents."""
    margin = BALL_R + Constants.POST_RADIUS + 6
    ys = np.linspace(GOAL_Y_MIN + margin, GOAL_Y_MAX - margin, samples)
    if opponents is None or len(opponents) == 0:
        return float(np.clip(ball[1], ys[0], ys[-1]))
    best, best_clear = ys[samples // 2], -1.0
    ends = np.column_stack([np.full(samples, float(GOAL_LINE)), ys])
    for y, clear in zip(ys, lane_clearances(ball, ends, opponents)):
        if clear > best_clear:
            best, best_clear = y, clear
    return float(best)


def lane_clearance(start, end, others):
    """Smallest distance from any of `others` to the segment start -> end."""
    seg = end - start
    length_sq = float(np.dot(seg, seg)) or 1.0
    others = np.asarray(others)
    rel = others - start
    t = np.clip(rel @ seg / length_sq, 0.0, 1.0)
    closest = start + t[:, None] * seg
    return float(norms(others - closest).min())


def lane_clearances(start, ends, others):
    """lane_clearance for several segments at once: from start to each row of `ends`."""
    segs = ends - start
    length_sq = np.array([float(np.dot(seg, seg)) or 1.0 for seg in segs])
    others = np.asarray(others)
    rel = others - start
    # One matrix-vector product per segment, as for a single one (see _times_to_reach).
    t = np.clip(np.array([rel @ seg for seg in segs]) / length_sq[:, None], 0.0, 1.0)
    closest = start + t[:, :, None] * segs[:, None, :]
    return norms(others - closest).min(axis=1).tolist()


# ---------------------------------------------------------------- positional skills


def mark(pos, vel, opponent, distance=60.0):
    """Stand between an opponent and our goal, `distance` from the opponent."""
    goal_side = opponent + unit(OWN_GOAL_CENTRE - opponent) * distance
    return move_to(pos, vel, goal_side)


def block_lane(pos, vel, ball, fraction=0.3):
    """Stand on the line from the ball to the middle of our goal, `fraction` of the way."""
    point = ball + (OWN_GOAL_CENTRE - ball) * fraction
    return move_to(pos, vel, point)


def keep_goal(pos, vel, ball, ball_vel, depth=0.0, ball_path=None):
    """
    Goalkeeper: stay on (or `depth` px in front of) our goal line, at the height the
    ball will cross it, or covering the ball's angle when it is not coming our way.
    `ball_path`, if known, is predict_ball(ball, ball_vel, ...): its first ticks are
    used instead of predicting them again.
    """
    x = OWN_GOAL_LINE + PLAYER_R + depth
    if ball_vel[0] < -0.5:
        t = (ball[0] - x) / -ball_vel[0]
        ticks = int(min(max(t, 1), 200))
        if ball_path is None or len(ball_path) < ticks:
            ball_path = predict_ball(ball, ball_vel, ticks)
        y = ball_path[ticks - 1][1]
    else:
        y = H / 2 + (ball[1] - H / 2) * 0.5
    y = float(np.clip(y, GOAL_Y_MIN + PLAYER_R * 0.5, GOAL_Y_MAX - PLAYER_R * 0.5))
    return move_to(pos, vel, np.array([x, y]))
