"""
LLMBrain v11: the kick planner. The first version under the new rules (improvement first;
large changes allowed), so a rewrite rather than a step from v8.

What I saw: in recorded games the ball is at or near its top speed (15 px a tick, twice a
player's) about 70% of the time, and nothing slows it down: it flies about the field,
bouncing off the walls, and is touched only about once every 15-30 ticks. So this is
closer to air hockey than to football: what matters is where each touch sends the ball.
v8's players run at a spot and bump the ball wherever the contact happens to send it: in a
test with a randomly moving ball, 46% of v8's first touches went forwards and 23% towards
their goal. Steering to "the spot behind the ball" by geometry alone (my first drafts) did
little better, 78% and 46%, because the ball moves while the player gets there.

What I changed: everything.
- The ball's path is predicted ahead, with its bounces off the walls.
- The kick planner: each of our players tries 49 moves (run in one of 24 directions at full
  or half speed, or stop), follows each for 40 ticks with the game's own rules for
  acceleration and top speed, finds where it would first meet the ball and, from the
  collision rule, where the ball would then go. Each result is scored: forwards is good,
  on target (straight or off a side wall) is better, and better still when no opponent can
  get into its path in time; towards our own goal is very bad; sooner is better. The best
  kick of the best-placed player is taken, if it is good enough.
- Otherwise the player who can reach the ball's path first heads for the spot behind the
  ball, going round it rather than through it.
- The keeper stays on the goal line and moves to where the ball's path will cross it; when
  the ball is close and it has a good kick, it clears.
- One player waits by their far post for rebounds while the ball is in their half; the
  other two hold a line behind the ball to meet clearances.

Why: the planner turns the one thing that decides this game, where each touch sends the
ball, from luck into a choice. In the same test one player using it sent 97% of its first
touches forwards and 94% towards their goal.
"""

import numpy as np

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.constants import Constants as C

# The field, for the ball's centre. Outside the goal mouths it bounces off the end walls
# at X_MIN and X_MAX; inside a mouth it goes in.
TOUCH = C.PLAYER_RADIUS + C.BALL_RADIUS  # distance between centres when a player touches the ball
X_MIN = C.GOAL_DEPTH + C.BALL_RADIUS
X_MAX = C.FIELD_LENGTH - 1 - C.GOAL_DEPTH - C.BALL_RADIUS
Y_MIN = C.BALL_RADIUS
Y_MAX = C.FIELD_HEIGHT - C.BALL_RADIUS
MID_Y = C.FIELD_HEIGHT / 2
THEIR_GOAL = np.array([C.FIELD_LENGTH - 1 - C.GOAL_DEPTH, MID_Y])
OUR_GOAL = np.array([C.GOAL_DEPTH, MID_Y])
# In a collision the ball (mass ~ radius squared) gains 1.8 times the closing speed along
# the line between the centres: 2 m_player / (m_player + m_ball).
KICK_GAIN = 2 * C.PLAYER_RADIUS**2 / (C.PLAYER_RADIUS**2 + C.BALL_RADIUS**2)

LOOK_AHEAD = 120  # ticks of predicted ball path
TICKS = np.arange(LOOK_AHEAD + 1, dtype=float)
PLAN_TICKS = 40  # ticks each planned move is followed
DIRECTIONS = 24  # the planner's running directions
_ANGLES = np.linspace(0, 2 * np.pi, DIRECTIONS, endpoint=False)
_UNIT_MOVES = np.concatenate(
    [
        np.stack([np.cos(_ANGLES), np.sin(_ANGLES)], 1),  # full speed
        0.5 * np.stack([np.cos(_ANGLES), np.sin(_ANGLES)], 1),  # half speed
        np.zeros((1, 2)),  # stop
    ]
)

# Scoring a kick
ON_TARGET_OPEN = 1.3  # bonus for a ball heading into their goal that no opponent can reach first
ON_TARGET_BLOCKED = 0.6  # bonus when an opponent can get into its path in time
POST_MARGIN = 15  # a shot must cross the goal line this far inside a post
OWN_GOAL_RISK = 3.0  # penalty for a ball heading at our goal (or near it)
OWN_GOAL_MARGIN = 40
TIME_COST = 0.02  # per tick until the kick
GOOD_KICK = 0.3  # a planned kick must score above this to be taken

# Positions
KEEPER_X = C.GOAL_DEPTH + C.PLAYER_RADIUS + 6
KEEPER_CLEARS_INSIDE = 200  # the keeper may leave its line for a kick when the ball is this close
BEHIND = 30  # the chaser lines up this far behind the ball
SWITCH_MARGIN = 4  # ticks: another player takes over as chaser only if this much sooner
LINE_GAP = 220  # the line stands this far behind the ball...
LINE_MIN_X, LINE_MAX_X = 260, 1300  # ...but no deeper or higher than this
POACHER_FROM_X = 900  # the poacher goes forward once the ball is past this
POACHER_SPOT = (1600, 120)  # a poacher's x, and its distance from mid-goal towards its post
POACHERS = 1


def fold(u, lo, hi):
    """Where a point moving freely to u ends up between two walls at lo and hi that reflect it."""
    width = hi - lo
    w = np.mod(u - lo, 2 * width)
    return lo + np.where(w > width, 2 * width - w, w)


def fold_sign(u, lo, hi):
    """+1 where the reflected motion runs the original way at u, -1 where it has turned."""
    width = hi - lo
    return np.where(np.mod(u - lo, 2 * width) > width, -1.0, 1.0)


def predict_ball(pos, vel):
    """
    The ball's positions and velocities for the next LOOK_AHEAD ticks, bouncing off the walls
    and ignoring players, and whether it goes into a goal (+1 theirs, -1 ours, 0 neither).
    """
    ux, uy = pos[0] + vel[0] * TICKS, pos[1] + vel[1] * TICKS
    x, y = fold(ux, X_MIN, X_MAX), fold(uy, Y_MIN, Y_MAX)
    vx = vel[0] * fold_sign(ux, X_MIN, X_MAX)
    vy = vel[1] * fold_sign(uy, Y_MIN, Y_MAX)
    goal = 0
    at_end = (ux < X_MIN) | (ux > X_MAX)
    if at_end.any():
        t = int(np.argmax(at_end))  # the first time it reaches an end wall
        if C.GOAL_Y_MIN < y[t] < C.GOAL_Y_MAX:
            goal = 1 if ux[t] > X_MAX else -1
            x[t:], y[t:], vx[t:], vy[t:] = ux[t], y[t], 0.0, 0.0
    return np.stack([x, y], 1), np.stack([vx, vy], 1), goal


def cross_end_line(pos, vel, x_line):
    """Where (y, after side-wall bounces) and when balls (N x 2) moving freely reach x = x_line."""
    vx = vel[:, 0]
    t = (x_line - pos[:, 0]) / np.where(np.abs(vx) < 1e-6, 1e-6, vx)
    return fold(pos[:, 1] + vel[:, 1] * t, Y_MIN, Y_MAX), t


def reach_time(pos, vel, targets):
    """
    Roughly how many ticks each player (pos, vel: P x 2) needs to get to each target (T x 2):
    accelerate towards it at full rate up to top speed, plus time to cancel sideways motion.
    """
    vmax, acc = C.MAX_PLAYER_VELOCITY, C.MAX_ACCELERATION
    dx = targets[None, :, 0] - pos[:, None, 0]
    dy = targets[None, :, 1] - pos[:, None, 1]
    d = np.hypot(dx, dy)
    towards = (dx * vel[:, None, 0] + dy * vel[:, None, 1]) / np.maximum(d, 1e-9)
    sideways = np.sqrt(np.maximum((vel**2).sum(1)[:, None] - towards**2, 0.0))
    accelerating = (vmax**2 - towards**2) / (2 * acc)  # distance covered getting to top speed
    t_short = (-towards + np.sqrt(np.maximum(towards**2 + 2 * acc * d, 0.0))) / acc
    t_long = (vmax - towards) / acc + (d - accelerating) / vmax
    return np.where(d <= accelerating, t_short, t_long) + 0.5 * sideways / acc


def steer(pos, vel, target, arrive=True):
    """Acceleration towards target at top speed; with arrive, slowing so as to stop there."""
    diff = target - pos
    d = np.hypot(diff[0], diff[1])
    if d < 1e-6:
        return -vel
    speed = C.MAX_PLAYER_VELOCITY
    if arrive:
        speed = min(speed, np.sqrt(1.6 * C.MAX_ACCELERATION * d), d)
    return diff / d * speed - vel


def plan_kicks(pos, vel, traj, tvel):
    """
    The kick planner's rollout. For each of our players (P) and each move (M): follow the
    move for PLAN_TICKS ticks and find the first contact with the ball on its predicted
    path. Returns the contact time (P x M, inf if none) and the ball's velocity after the
    contact (P x M x 2), from the game's collision rule.

    A move is a wanted velocity u. Each tick the player accelerates towards u at the full
    rate, so its velocity slides along the straight line from its velocity v0 to u by
    MAX_ACCELERATION a tick, never above top speed: v_k = v0 + (u - v0) min(1, a k / |u - v0|).
    """
    vmax, acc = C.MAX_PLAYER_VELOCITY, C.MAX_ACCELERATION
    moves = _UNIT_MOVES * vmax
    P, M, K = len(pos), len(moves), PLAN_TICKS
    hit_t = np.full((P, M), np.inf)
    ball_after = np.zeros((P, M, 2))
    k = np.arange(1, K + 1, dtype=float)
    # skip players who cannot get to the ball's path in time, whatever they do
    gap = np.hypot(traj[None, 1:K + 1, 0] - pos[:, None, 0], traj[None, 1:K + 1, 1] - pos[:, None, 1])
    near = np.nonzero((gap - TOUCH - vmax * k - 4 <= 0).any(axis=1))[0]
    if not len(near):
        return hit_t, ball_after
    pos, vel = pos[near], vel[near]
    dux = moves[None, :, 0] - vel[:, None, 0]
    duy = moves[None, :, 1] - vel[:, None, 1]
    share = np.minimum(1.0, acc * k / np.maximum(np.hypot(dux, duy), 1e-9)[..., None])  # P x M x K
    vx = vel[:, None, None, 0] + dux[..., None] * share
    vy = vel[:, None, None, 1] + duy[..., None] * share
    px = pos[:, None, None, 0] + np.cumsum(vx, axis=2)
    py = pos[:, None, None, 1] + np.cumsum(vy, axis=2)
    # the ball relative to the player at the start of each tick (r0) and its change in the tick
    r0x, r0y = traj[:K, 0] - (px - vx), traj[:K, 1] - (py - vy)
    drx, dry = (traj[1:K + 1, 0] - px) - r0x, (traj[1:K + 1, 1] - py) - r0y
    # first moment s in the tick (0..1) when they are TOUCH apart: |r0 + s dr| = TOUCH
    a = drx * drx + dry * dry
    b = 2 * (r0x * drx + r0y * dry)
    c = r0x * r0x + r0y * r0y - TOUCH**2
    disc = b * b - 4 * a * c
    s = (-b - np.sqrt(np.maximum(disc, 0))) / np.maximum(2 * a, 1e-9)
    touch = (disc >= 0) & (s <= 1) & ((s >= 0) | (c <= 0))
    pi, mi = np.nonzero(touch.any(axis=2))
    if len(pi):
        ki = touch.argmax(axis=2)[pi, mi]
        si = np.clip(s[pi, mi, ki], 0, 1)
        nx = r0x[pi, mi, ki] + drx[pi, mi, ki] * si
        ny = r0y[pi, mi, ki] + dry[pi, mi, ki] * si
        norm = np.maximum(np.hypot(nx, ny), 1e-9)
        nx, ny = nx / norm, ny / norm  # the line between the centres at contact
        bvx, bvy = tvel[ki, 0], tvel[ki, 1]
        closing = np.maximum((vx[pi, mi, ki] - bvx) * nx + (vy[pi, mi, ki] - bvy) * ny, 0)
        newx, newy = bvx + KICK_GAIN * closing * nx, bvy + KICK_GAIN * closing * ny
        cap = np.minimum(1.0, C.MAX_BALL_VELOCITY / np.maximum(np.hypot(newx, newy), 1e-9))
        rows = near[pi]
        ball_after[rows, mi, 0], ball_after[rows, mi, 1] = newx * cap, newy * cap
        hit_t[rows, mi] = ki + si
    return hit_t, ball_after


class LLMBrainV11(AbstractBrain):
    def __init__(self, name=None):
        super().__init__(name)
        self.chaser = None  # the chaser keeps the job unless another is clearly sooner

    # ---------- judging and choosing kicks ----------
    def kick_value(self, ball_pos, ball_vel, t_kick):
        """How good it is to send the ball from ball_pos with ball_vel (N x 2), t_kick ticks from now."""
        vx = ball_vel[:, 0]
        value = vx / C.MAX_BALL_VELOCITY  # forwards is good
        y_end, t_end = cross_end_line(ball_pos, ball_vel, X_MAX)
        on_target = (vx > 1) & (y_end > C.GOAL_Y_MIN + POST_MARGIN) & (y_end < C.GOAL_Y_MAX - POST_MARGIN)
        if on_target.any():
            i = np.nonzero(on_target)[0]
            # can an opponent get into the ball's path before the ball passes?
            after = t_end[i, None] * np.linspace(0.15, 1.0, 8)[None, :]
            px = ball_pos[i, 0, None] + ball_vel[i, 0, None] * after
            py = fold(ball_pos[i, 1, None] + ball_vel[i, 1, None] * after, Y_MIN, Y_MAX)
            points = np.stack([px, py], -1).reshape(-1, 2)
            t_them = reach_time(self.opp_players_pos, self.opp_players_vel, points).min(axis=0)
            spare = t_them.reshape(px.shape) - TOUCH / C.MAX_PLAYER_VELOCITY - after - t_kick[i, None]
            bonus = np.where((spare > 0).all(axis=1), ON_TARGET_OPEN, ON_TARGET_BLOCKED)
            speed = np.hypot(vx[i], ball_vel[i, 1]) / C.MAX_BALL_VELOCITY
            value[i] += bonus * (0.8 + 0.2 * speed)
        y_our, _ = cross_end_line(ball_pos, ball_vel, X_MIN)
        at_our_goal = (
            (vx < -0.5) & (y_our > C.GOAL_Y_MIN - OWN_GOAL_MARGIN) & (y_our < C.GOAL_Y_MAX + OWN_GOAL_MARGIN)
        )
        return value - OWN_GOAL_RISK * at_our_goal

    def best_kicks(self, traj, tvel):
        """For each of our players: the acceleration for its best planned kick, and its score."""
        me, mv = self.my_players_pos, self.my_players_vel
        hit_t, ball_after = plan_kicks(me, mv, traj, tvel)
        score = np.full(hit_t.shape, -np.inf)
        hit = np.isfinite(hit_t)
        if hit.any():
            where = traj[np.minimum(hit_t[hit].astype(int), LOOK_AHEAD)]
            score[hit] = self.kick_value(where, ball_after[hit], hit_t[hit]) - TIME_COST * hit_t[hit]
        best = score.argmax(axis=1)
        rows = np.arange(len(me))
        return _UNIT_MOVES[best] * C.MAX_PLAYER_VELOCITY - mv, score[rows, best]

    # ---------- the team ----------
    def do_move(self):
        me, mv = self.my_players_pos, self.my_players_vel
        ball = np.asarray(self.ball_pos, dtype=float)
        traj, tvel, goal = predict_ball(ball, np.asarray(self.ball_vel, dtype=float))
        kick_acc, kick_score = self.best_kicks(traj, tvel)
        acc = np.zeros((5, 2))
        free = [1, 2, 3, 4]  # outfield players without a job yet

        # the striker: the outfield player with the best planned kick, if it is good enough
        striker = max(free, key=lambda i: kick_score[i])
        if kick_score[striker] > GOOD_KICK:
            acc[striker] = kick_acc[striker]
            self.chaser = striker
        else:
            # the chaser: first to the ball's path heads for the spot behind the ball
            late = reach_time(me, mv, traj) - TOUCH / C.MAX_PLAYER_VELOCITY - TICKS[None, :]
            first = np.where((late <= 1).any(axis=1), (late <= 1).argmax(axis=1), late.argmin(axis=1))
            striker = min(free, key=lambda i: first[i])
            if self.chaser in free and first[self.chaser] <= first[striker] + SWITCH_MARGIN:
                striker = self.chaser
            self.chaser = striker
            acc[striker] = self.go_behind(striker, traj[int(first[striker])], ball)
        free.remove(striker)

        acc[0] = self.keeper(traj, goal, kick_acc[0], kick_score[0])

        # the poachers wait by their posts, far post first, while the ball is in their half
        if ball[0] > POACHER_FROM_X:
            far_post = 1.0 if ball[1] < MID_Y else -1.0
            for post in (far_post, -far_post)[:POACHERS]:
                poacher = max(free, key=lambda i: me[i][0])
                spot = np.array([POACHER_SPOT[0], MID_Y + post * POACHER_SPOT[1]])
                acc[poacher] = steer(me[poacher], mv[poacher], spot)
                free.remove(poacher)

        # the rest hold a line behind the ball, spread across the field
        line_x = np.clip(ball[0] - LINE_GAP, LINE_MIN_X, LINE_MAX_X)
        shares = {1: [0.5], 2: [0.33, 0.67], 3: [0.22, 0.5, 0.78]}[len(free)]
        ys = np.array(shares) * C.FIELD_HEIGHT + (ball[1] - MID_Y) * 0.4
        for i, y in zip(sorted(free, key=lambda j: me[j][1]), ys):
            acc[i] = steer(me[i], mv[i], np.array([line_x, y]))
        return acc

    def go_behind(self, i, ball_then, ball_now):
        """Run for the spot just behind the ball (from their goal), passing round the ball if it is in the way."""
        pos, vel = self.my_players_pos[i], self.my_players_vel[i]
        to_goal = THEIR_GOAL - ball_then
        spot = ball_then - to_goal / max(np.hypot(*to_goal), 1e-9) * (TOUCH + BEHIND)
        for b in (ball_now, ball_then):
            path = spot - pos
            along = np.clip((b - pos) @ path / max(path @ path, 1e-9), 0, 1)
            miss = b - (pos + path * along)
            if 0 < along < 1 and miss @ miss < (TOUCH + 14) ** 2:
                side = np.array([-path[1], path[0]]) / max(np.hypot(*path), 1e-9)
                if (b - pos) @ side > 0:
                    side = -side  # pass on the side we are already on
                return steer(pos, vel, b + side * (TOUCH + 26), arrive=False)
        return steer(pos, vel, spot, arrive=False)

    def keeper(self, traj, goal, kick_acc, kick_score):
        """Stay on the goal line where the ball's path will cross it; clear a close ball."""
        pos, vel = self.my_players_pos[0], self.my_players_vel[0]
        ball = traj[0]
        if ball[0] < KEEPER_CLEARS_INSIDE and kick_score > GOOD_KICK:
            return kick_acc
        if goal == -1 or self.ball_vel[0] < -1:
            crossing = np.nonzero(traj[:, 0] <= KEEPER_X + 5)[0]
            if len(crossing):
                y = traj[crossing[0], 1]
                if C.GOAL_Y_MIN - 30 < y < C.GOAL_Y_MAX + 30:
                    y = np.clip(y, C.GOAL_Y_MIN + 10, C.GOAL_Y_MAX - 10)
                    return steer(pos, vel, np.array([KEEPER_X, y]))
        # otherwise on the line from the middle of our goal to the ball
        d = ball - OUR_GOAL
        y = MID_Y + d[1] * (KEEPER_X - OUR_GOAL[0]) / max(d[0], 30.0)
        return steer(pos, vel, np.array([KEEPER_X, np.clip(y, C.GOAL_Y_MIN + 20, C.GOAL_Y_MAX - 20)]))
