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


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(2)


# ---------------------------------------------------------------- prediction


def predict_ball(pos, vel, ticks):
    """
    Ball positions for the next `ticks` ticks (row t = after t+1 ticks), bouncing off
    the side walls and the goal lines outside the goal mouths. Players are ignored.
    """
    p = np.array(pos, dtype=float)
    v = np.array(vel, dtype=float)
    out = np.empty((ticks, 2))
    for t in range(ticks):
        in_mouth = GOAL_Y_MIN < p[1] < GOAL_Y_MAX
        min_x = (0.0 if in_mouth else OWN_GOAL_LINE) + BALL_R
        max_x = (L if in_mouth else GOAL_LINE) - BALL_R
        if (p[0] < min_x and v[0] < 0) or (p[0] > max_x and v[0] > 0):
            v[0] = -v[0]
        if (p[1] < BALL_R and v[1] < 0) or (p[1] > H - BALL_R and v[1] > 0):
            v[1] = -v[1]
        p = p + v
        out[t] = p
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
    offset = np.asarray(targets, dtype=float) - pos
    d = np.linalg.norm(offset, axis=1)
    direction = offset / np.maximum(d, 1e-9)[:, None]
    v0 = direction @ vel
    sideways = np.linalg.norm(vel[None, :] - v0[:, None] * direction, axis=1)
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
    pos = np.asarray(pos, dtype=float)
    offsets = ball_path - pos
    dist = np.linalg.norm(offsets, axis=1)
    reach_points = ball_path - offsets / np.maximum(dist, 1e-9)[:, None] * (
        CONTACT + margin
    )
    ok = times_to_reach(pos, np.asarray(vel, dtype=float), reach_points) <= np.arange(
        1, len(ball_path) + 1
    )
    t = int(np.argmax(ok)) if ok.any() else len(ball_path) - 1
    return t, ball_path[t]


# ---------------------------------------------------------------- movement


def move_to(pos, vel, target, arrive_speed=0.0):
    """
    Accelerate towards `target`, braking so as to arrive at about `arrive_speed`
    instead of overshooting (with acceleration 1, speed sqrt(2d) stops in d).
    """
    offset = np.asarray(target, dtype=float) - pos
    d = np.linalg.norm(offset)
    if d < 1.0:
        return -vel  # stop
    speed = min(MAX_SPEED, math.sqrt(2 * d + arrive_speed**2))
    return offset / d * speed - vel


def run_through(pos, vel, direction, speed=MAX_SPEED):
    """Accelerate so the velocity becomes `speed` along `direction`."""
    return unit(direction) * speed - vel


# ---------------------------------------------------------------- ball skills


def kick(pos, vel, ball, ball_vel, direction, power=1.0, ball_path=None):
    """
    Send the ball in `direction` at up to `power` x full strength.

    The player's centre must meet the ball at the contact point behind it (the ball
    leaves along the line through both centres). From roughly behind the ball (within
    about 60 degrees of the kick direction) the player drives straight through the
    contact point; a small angle only bends the kick a little. From the wrong side it
    runs round the ball first. The approach speed sets the kick's strength.
    """
    d = unit(direction)
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


def tackle(pos, vel, ball, ball_vel, ball_path=None):
    """Win a contested ball: run straight at it and knock it towards the opponent's goal."""
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
    for y in ys:
        clear = lane_clearance(ball, np.array([GOAL_LINE, y]), opponents)
        if clear > best_clear:
            best, best_clear = y, clear
    return float(best)


def lane_clearance(start, end, others):
    """Smallest distance from any of `others` to the segment start -> end."""
    seg = end - start
    length_sq = float(np.dot(seg, seg)) or 1.0
    rel = np.asarray(others) - start
    t = np.clip(rel @ seg / length_sq, 0.0, 1.0)
    closest = start + t[:, None] * seg
    return float(np.min(np.linalg.norm(np.asarray(others) - closest, axis=1)))


# ---------------------------------------------------------------- positional skills


def mark(pos, vel, opponent, distance=60.0):
    """Stand between an opponent and our goal, `distance` from the opponent."""
    goal_side = opponent + unit(OWN_GOAL_CENTRE - opponent) * distance
    return move_to(pos, vel, goal_side)


def block_lane(pos, vel, ball, fraction=0.3):
    """Stand on the line from the ball to the middle of our goal, `fraction` of the way."""
    point = ball + (OWN_GOAL_CENTRE - ball) * fraction
    return move_to(pos, vel, point)


def keep_goal(pos, vel, ball, ball_vel, depth=0.0):
    """
    Goalkeeper: stay on (or `depth` px in front of) our goal line, at the height the
    ball will cross it, or covering the ball's angle when it is not coming our way.
    """
    x = OWN_GOAL_LINE + PLAYER_R + depth
    if ball_vel[0] < -0.5:
        t = (ball[0] - x) / -ball_vel[0]
        y = predict_ball(ball, ball_vel, int(min(max(t, 1), 200)))[-1][1]
    else:
        y = H / 2 + (ball[1] - H / 2) * 0.5
    y = float(np.clip(y, GOAL_Y_MIN + PLAYER_R * 0.5, GOAL_Y_MAX - PLAYER_R * 0.5))
    return move_to(pos, vel, np.array([x, y]))
