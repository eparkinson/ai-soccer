"""
LLM-v11-calc: LLM-v11 (aisoccer/llm/v11.py) with its simulated kick planner replaced by a
calculated one. Everything else (keeper, chaser, poacher, line, kick scoring) is v11's own
code, so a comparison with v11 isolates the planner.

v11's planner simulates 49 moves per player for 40 ticks and looks for the first contact.
This one works backwards instead: pick a tick K and a contact point, then solve in closed
form for the move that puts the player there at tick K, the velocity it then has, and the
contact normal that sends the ball where we want. The maths, in four steps:

1. The ball's path. With no friction the ball moves in straight lines, reflected at the
   walls: position after k ticks = fold(pos + k vel) (v11's predict_ball). So B(K) and the
   ball's velocity b(K) are known for every K.

2. Where can a player be at tick K, and how fast is it then going? A move is v11's kind of
   move: a wanted velocity u (|u| <= top speed V). Each tick the player's velocity slides
   from v0 towards u by the acceleration cap A, so with D = |u - v0| and e = (u - v0) / D
       v_k = v0 + e min(D, A k),     p(K) = p0 + K v0 + e G_K(D),
       G_K(D) = sum_{k=1..K} min(D, A k) = A m (m + 1) / 2 + D (K - m),   m = floor(D / A).
   So the displacement beyond drifting, p(K) - p0 - K v0, points along e and has length
   G_K(D), which grows with D (piecewise linearly) up to A K (K + 1) / 2 (full acceleration
   all the way). To be at X at tick K: e is the direction of Delta = X - p0 - K v0, and D
   solves G_K(D) = |Delta|, a quadratic on each linear piece (step 2 of solve_move). The
   move is possible if |Delta| < A K (K + 1) / 2 and |u| <= V. This is the rollout's family of
   moves inverted exactly, with any direction and speed rather than 49 of them.

3. The contact. With n the unit vector from the player's centre to the ball's at contact
   and w = v_player - v_ball, the collision rule gives the ball
       b' = b + g (w . n) n,   g = 2 m_p / (m_p + m_b) = 1.8   (v11's KICK_GAIN)
   when w . n > 0 (closing). For the ball to leave along a wanted direction d, the part of
   b' across d must vanish. Writing n = (cos t, sin t) in axes along and across d, with
   b = (a, c) and w = (wx, wy) in the same axes:
       c + g (wx cos t + wy sin t) sin t = 0
       <=>  sin(2t - phi) = -(c + g wy / 2) / (g |w| / 2),   phi = atan2(wy, wx),
   a closed form with two roots for t; the one with cos(2t - phi) >= 0 sends the ball faster
   along d. If the right-hand side is outside [-1, 1], d cannot be reached; the nearest
   direction is used and the kick is scored for where it really goes.

   Steps 2 and 3 depend on each other (n moves the contact point, and the contact point sets
   the player's arrival velocity), but weakly: a shift of up to 2 TOUCH in the contact point
   against a run of hundreds of pixels. So they are iterated (fixed point, from n = d).
   The touch is planned a quarter of a tick before the end of tick K (the player then ends
   tick K at B(K) - TOUCH n + (v_K - b) / 4): planned for the very end of the tick, float
   rounding put half the touches into the next tick, when the replanning player had nothing
   sensible left to plan for (as first built, 19 of 132 first touches in a test fell back to
   v11's "run behind the ball" in the last tick).

4. Choosing. For every player, contact ticks K from the earliest it could reach the ball's
   path (and the tick it planned for last time, so a plan can be carried on) and 21 aims
   (three points across their goal, two banks off the side walls and 16 directions all
   round), steps 2-3 give the move, the contact and the ball's velocity after it, all in
   closed form. The plans are scored quickly (v11's kick_value without its lane check);
   each player's best 8 are checked for an earlier touch (the player's position at every
   tick is also closed form, and within a tick both bodies move straight, so the check is
   exact: does the ball pass within TOUCH of the player before tick K?); the best 3 that
   pass get v11's own kick_value (forwards, on target, lane open, own-goal risk) less v11's
   time cost, and each player takes its best, steering by u - v0 this tick (the game caps it
   at A). Everything is recomputed every tick (receding horizon); LLMBrainV11CalcLite
   carries an exact plan on for a few ticks instead.

Cost: about 250 numpy operations on arrays of a thousand or so numbers, against v11's ~50 on
arrays of ~10,000. In numpy the cost is mostly per call, so per tick the whole brain costs
about 1.7 ms against v11's 1.1 (the Lite variant: 0.9).

Result (analysis/league3/v11calc/results.txt): in a first-touch test it matches the rollout
(96% of first touches forwards, 93% on target; v11 95% and 92%) and touches the ball sooner,
but in 9000-tick games it plays worse than v11: it lost 0-8 to v11 by about 2-3 goals a game
and beat Tactics and GP-v2 by less than v11 does.
"""

import numpy as np

from aisoccer.constants import Constants as C
from aisoccer.llm import v11
from aisoccer.llm.v11 import KICK_GAIN, MID_Y, PLAN_TICKS, TIME_COST, TOUCH, X_MAX, Y_MAX, Y_MIN, LLMBrainV11

K_TICKS = np.arange(1, PLAN_TICKS + 1, dtype=float)
# Contact ticks tried, counted from the earliest tick each player could reach the ball
K_OFFSETS = np.array([0, 1, 2, 3, 4, 6, 8, 11, 15, 20])
FIXED_POINT_STEPS = 2  # rounds of steps 2-3 after the first guess
EARLY_TOUCH = 0.25  # ticks: plan the touch this long before the end of tick K, so it falls inside the tick
CHECK = 8  # plans per player (best by a quick score) checked for an earlier touch...
RESCORE = 3  # ...and of those that pass, the best few given v11's full kick_value (with its lane check)
EARLY_MARGIN = 1.0  # px: passing closer than TOUCH + this before the planned tick counts as a touch

# Aims: points the ball's centre should cross their goal line at (straight or mirrored in a
# side wall for a bank shot), then directions all round (every 22.5 degrees: first built with
# only forwards and 40 degrees either side, it lost 2-3 more goals a game; see the analysis).
GOAL_X = X_MAX + 10
_GOAL_AIM_Y = np.array([MID_Y - 120, MID_Y, MID_Y + 120, 2 * Y_MIN - MID_Y, 2 * Y_MAX - MID_Y])
_CLEAR_ANGLES = np.radians(np.arange(-180.0, 180.0, 22.5))
N_AIMS = len(_GOAL_AIM_Y) + len(_CLEAR_ANGLES)
_DIR_X = np.repeat(np.cos(_CLEAR_ANGLES)[None, :], PLAN_TICKS + 1, 0)
_DIR_Y = np.repeat(np.sin(_CLEAR_ANGLES)[None, :], PLAN_TICKS + 1, 0)


def displacement_limit(K, acc=None):
    """How far (beyond drifting) a player can get in K ticks: A K (K + 1) / 2."""
    acc = C.MAX_ACCELERATION if acc is None else acc
    return acc * K * (K + 1) / 2


def solve_move(px, py, vx, vy, X, Y, K, acc=None, vmax=None):
    """
    The move (wanted velocity u) that brings a player at (px, py) with velocity (vx, vy) to
    (X, Y) after exactly K ticks, accelerating towards u at the full rate each tick (step 2 of
    the module docstring). Arguments are arrays (or numbers) that broadcast.

    Returns (ux, uy, vKx, vKy, ok): the wanted velocity, the player's velocity at tick K and
    whether the move is possible (reachable in K ticks with |u| <= top speed).
    """
    acc = C.MAX_ACCELERATION if acc is None else acc
    vmax = C.MAX_PLAYER_VELOCITY if vmax is None else vmax
    dx, dy = X - px - K * vx, Y - py - K * vy
    L = np.hypot(dx, dy)
    inv = 1.0 / np.maximum(L, 1e-9)
    ex, ey = dx * inv, dy * inv
    # G_K(D) ~ D (K + 1/2) - D^2 / (2A) (exact at multiples of A): its root is a first guess
    Kh = K + 0.5
    D = acc * (Kh - np.sqrt(np.maximum(Kh * Kh - 2 * L / acc, 0.0)))
    # then solve exactly on the linear piece the guess lies on (if the guess fell just short of
    # a piece's start, the neighbouring piece's line is used: off by under A / 8 px at worst)
    m = np.minimum(np.floor(D / acc), K - 1)
    D = np.maximum((L - acc * m * (m + 1) / 2) / (K - m), 0.0)
    ux, uy = vx + ex * D, vy + ey * D
    ok = (L < displacement_limit(K, acc)) & (ux * ux + uy * uy <= vmax * vmax + 1e-6)
    DK = np.minimum(D, acc * K)
    return ux, uy, vx + ex * DK, vy + ey * DK, ok


def contact_normal(bx, by, wx, wy, dx, dy, gain=KICK_GAIN):
    """
    The unit contact normal n (player centre to ball centre) for which a ball with velocity
    b, hit by a player with relative velocity w = v_player - b, leaves along the unit
    direction d (step 3 of the module docstring). Where no normal can do it, the nearest.

    Of the two roots, 2t - phi = alpha or pi - alpha, the ball's speed along d afterwards,
    a + (g / 2)(wx + |w| cos(2t - phi)), is larger for the one with cos(alpha) >= 0.
    No trigonometric functions: with sin(alpha) = s and cos(alpha) = sqrt(1 - s^2), cos 2t and
    sin 2t follow from the angle-sum rule, and (cos t, sin t) is (1 + cos 2t, sin 2t) or
    (sin 2t, 1 - cos 2t) normalised (the half-angle rule; either sign will do).
    """
    c = by * dx - bx * dy  # b across d
    Wx, Wy = wx * dx + wy * dy, wy * dx - wx * dy  # w along and across d
    R = np.maximum(np.hypot(Wx, Wy), 1e-9)
    cphi, sphi = Wx / R, Wy / R
    s = np.minimum(np.maximum(-(c + 0.5 * gain * Wy) / (0.5 * gain * R), -1.0), 1.0)
    ca = np.sqrt(1.0 - s * s)
    C2, S2 = ca * cphi - s * sphi, s * cphi + ca * sphi
    hx, hy = np.where(C2 >= 0, 1 + C2, S2), np.where(C2 >= 0, S2, 1 - C2)
    h = 1.0 / np.maximum(np.hypot(hx, hy), 1e-12)
    ct, st = hx * h, hy * h
    nx, ny = ct * dx - st * dy, ct * dy + st * dx
    sign = np.where(wx * nx + wy * ny < 0, -1.0, 1.0)  # point it so that we close
    return nx * sign, ny * sign


def plan_kicks_calc(pos, vel, traj, tvel, keep=None):
    """
    The calculated planner (module docstring). Returns flat arrays over the plans tried:
    the player, the contact tick K, the move (ux, uy), whether the plan holds, where the
    ball is at contact and its velocity afterwards.
    """
    vmax = C.MAX_PLAYER_VELOCITY
    # 1. the earliest tick at which each player might reach the ball (optimistic)
    Bx, By = traj[1:PLAN_TICKS + 1, 0], traj[1:PLAN_TICKS + 1, 1]
    k = K_TICKS[None, :]
    drift = np.hypot(Bx[None] - pos[:, 0, None] - k * vel[:, 0, None], By[None] - pos[:, 1, None] - k * vel[:, 1, None])
    direct = np.hypot(Bx[None] - pos[:, 0, None], By[None] - pos[:, 1, None])
    reach = (drift - TOUCH < displacement_limit(k)) & (direct - TOUCH < vmax * k)
    players = np.nonzero(reach.any(axis=1))[0]
    empty = np.zeros(0)
    if not len(players):
        return (np.zeros(0, int),) + (empty,) * 7 + (np.zeros(0, bool),)
    kmin = reach[players].argmax(axis=1) + 1
    # 2. the plans: player x contact tick x aim, flattened
    # (the last column: the tick a player planned for last time, now one nearer, so that a
    # plan can be carried on; see LLMBrainV11Calc.best_kicks)
    kept = kmin if keep is None else np.where(keep[players] >= 1, keep[players], kmin)
    Kp = np.minimum(np.concatenate([kmin[:, None] + K_OFFSETS[None, :], kept[:, None]], 1), PLAN_TICKS)
    pl = np.repeat(players, Kp.shape[1] * N_AIMS)
    K = np.repeat(Kp.ravel(), N_AIMS)
    aim = np.tile(np.arange(N_AIMS), Kp.size)
    Kf = K.astype(float)
    Bx, By = traj[K, 0], traj[K, 1]
    bx, by = tvel[K - 1, 0], tvel[K - 1, 1]
    # the aims' directions from the ball at every tick (ticks x aims), then each plan's
    to_x = np.concatenate([np.repeat(GOAL_X - traj[:PLAN_TICKS + 1, 0:1], len(_GOAL_AIM_Y), 1), _DIR_X], 1)
    to_y = np.concatenate([_GOAL_AIM_Y[None, :] - traj[:PLAN_TICKS + 1, 1:2], _DIR_Y], 1)
    inv = 1.0 / np.hypot(to_x, to_y)
    row = K * N_AIMS + aim
    dx, dy = (to_x * inv).ravel()[row], (to_y * inv).ravel()[row]
    px, py, vx, vy = pos[pl, 0], pos[pl, 1], vel[pl, 0], vel[pl, 1]
    # 3. steps 2-3 as a fixed point, starting from n = d
    # The touch is planned for time K - EARLY_TOUCH, when the centres are TOUCH apart along n:
    # the player is then at B(K) - TOUCH n + EARLY_TOUCH (v_K - b) at the end of tick K.
    nx, ny, wx, wy = dx, dy, 0.0, 0.0
    for _ in range(FIXED_POINT_STEPS):
        X, Y = Bx - TOUCH * nx + EARLY_TOUCH * wx, By - TOUCH * ny + EARLY_TOUCH * wy
        _, _, vKx, vKy, _ = solve_move(px, py, vx, vy, X, Y, Kf)
        wx, wy = vKx - bx, vKy - by
        nx, ny = contact_normal(bx, by, wx, wy, dx, dy)
    X, Y = Bx - TOUCH * nx + EARLY_TOUCH * wx, By - TOUCH * ny + EARLY_TOUCH * wy
    ux, uy, vKx, vKy, ok = solve_move(px, py, vx, vy, X, Y, Kf)
    closing = (vKx - bx) * nx + (vKy - by) * ny
    ok &= closing > 0.05
    gain = KICK_GAIN * np.maximum(closing, 0.0)
    ax, ay = bx + gain * nx, by + gain * ny
    cap = np.minimum(1.0, C.MAX_BALL_VELOCITY / np.maximum(np.hypot(ax, ay), 1e-9))
    ax, ay = ax * cap, ay * cap
    return pl, Kf, ux, uy, Bx, By, ax, ay, ok


def touches_early(px, py, vx, vy, ux, uy, K, traj, acc=None):
    """
    Whether each plan (flat arrays) would touch the ball before its tick K. The player's
    position at every tick k is known in closed form, p(k) = p0 + k v0 + e G_k(D) (step 2),
    and within a tick both bodies move in straight lines, so each tick k < K is checked
    exactly: does the segment the ball follows relative to the player pass within TOUCH?
    """
    acc = C.MAX_ACCELERATION if acc is None else acc
    k = np.arange(PLAN_TICKS, dtype=float)[None, :]  # ticks 0 .. PLAN_TICKS - 1
    Dx, Dy = (ux - vx)[:, None], (uy - vy)[:, None]
    D = np.hypot(Dx, Dy)
    m = np.minimum(np.floor(D / acc), k)
    G = (acc * m * (m + 1) / 2 + D * (k - m)) / np.maximum(D, 1e-9)
    rx = traj[None, :PLAN_TICKS, 0] - (px[:, None] + k * vx[:, None] + Dx * G)
    ry = traj[None, :PLAN_TICKS, 1] - (py[:, None] + k * vy[:, None] + Dy * G)
    sx, sy = rx[:, 1:] - rx[:, :-1], ry[:, 1:] - ry[:, :-1]  # the ball's motion relative to us in tick k
    s = np.minimum(np.maximum(-(rx[:, :-1] * sx + ry[:, :-1] * sy) / np.maximum(sx * sx + sy * sy, 1e-9), 0), 1)
    close = np.hypot(rx[:, :-1] + s * sx, ry[:, :-1] + s * sy) < TOUCH + EARLY_MARGIN
    before = k[:, 1:] < K[:, None]  # ticks 1 .. K - 1
    return (close & before).any(axis=1)


def quick_value(bx, by, vx, vy, t):
    """v11's kick_value without its lane check (on-target kicks counted as open), less the time cost."""
    value = vx / C.MAX_BALL_VELOCITY - TIME_COST * t
    slope = vy / np.where(np.abs(vx) < 1e-6, 1e-6, vx)
    y_end = v11.fold(by + slope * (X_MAX - bx), Y_MIN, Y_MAX)  # where it crosses their goal line
    on = (vx > 1) & (y_end > C.GOAL_Y_MIN + v11.POST_MARGIN) & (y_end < C.GOAL_Y_MAX - v11.POST_MARGIN)
    value += on * v11.ON_TARGET_OPEN * (0.8 + 0.2 * np.hypot(vx, vy) / C.MAX_BALL_VELOCITY)
    y_our = v11.fold(by + slope * (v11.X_MIN - bx), Y_MIN, Y_MAX)  # ...and ours
    ours = (vx < -0.5) & (y_our > C.GOAL_Y_MIN - v11.OWN_GOAL_MARGIN) & (y_our < C.GOAL_Y_MAX + v11.OWN_GOAL_MARGIN)
    return value - v11.OWN_GOAL_RISK * ours


class LLMBrainV11Calc(LLMBrainV11):
    """LLM-v11 with the calculated kick planner (module docstring)."""

    REPLAN_EVERY = 1  # ticks; see LLMBrainV11CalcLite

    def __init__(self, name=None):
        super().__init__(name)
        self._keep = None  # each player's planned contact tick, as seen from the next tick
        self._plan = None  # the last plan, for LLMBrainV11CalcLite

    def best_kicks(self, traj, tvel):
        """For each of our players: the acceleration for its best planned kick, and its score."""
        me, mv = self.my_players_pos, self.my_players_vel
        plan = self._plan
        if plan is not None and self.REPLAN_EVERY > 1:
            # carry the plan on while everything has gone as planned: the ball where it was
            # predicted to be, the striker where its move puts it, and not too long ago
            u, score, K, at, bpos, bvel, age = plan
            i = self.chaser
            striking = i is not None and score[i] > v11.GOOD_KICK
            on_plan = not striking or (K[i] > age + 2 and np.abs(me[i] - at[i]).max() < 0.5)
            if (age + 1 < self.REPLAN_EVERY and on_plan
                    and np.abs(traj[0] - bpos).max() < 0.5 and np.abs(tvel[0] - bvel).max() < 0.05):
                self._plan = (u, score + TIME_COST, K, *self._next(u, traj, tvel), age + 1)
                return u - mv, score + TIME_COST
        acc, score, K = self._best_kicks_now(traj, tvel)
        u = acc + mv
        self._plan = (u, score, K, *self._next(u, traj, tvel), 0)
        return acc, score

    def _next(self, u, traj, tvel):
        """Where our players (following u) and the ball will be next tick, and the ball's velocity."""
        mv = self.my_players_vel
        a = u - mv
        n = np.hypot(a[:, 0], a[:, 1])[:, None]
        a = np.where(n > C.MAX_ACCELERATION, a / np.maximum(n, 1e-9) * C.MAX_ACCELERATION, a)
        return self.my_players_pos + mv + a, traj[1], tvel[1]

    def _best_kicks_now(self, traj, tvel):
        """best_kicks, planned afresh; also returns each player's planned contact tick."""
        me, mv = self.my_players_pos, self.my_players_vel
        P = len(me)
        acc, best_score = np.zeros((P, 2)), np.full(P, -np.inf)
        Kbest = np.zeros(P)
        pl, K, ux, uy, Bx, By, ax, ay, ok = plan_kicks_calc(me, mv, traj, tvel, self._keep)
        self._keep = None
        if not ok.any():
            return acc, best_score, Kbest
        # a quick score for every plan; v11's full score for each player's best few
        quick = np.where(ok, quick_value(Bx, By, ax, ay, K), -np.inf).reshape(-1, (len(K_OFFSETS) + 1) * N_AIMS)
        top = np.argsort(-quick, axis=1)[:, :CHECK]
        idx = (top + np.arange(len(quick))[:, None] * quick.shape[1]).ravel()
        idx = idx[ok[idx]]
        idx = idx[~touches_early(me[pl[idx], 0], me[pl[idx], 1], mv[pl[idx], 0], mv[pl[idx], 1],
                                 ux[idx], uy[idx], K[idx], traj)]
        if not len(idx):
            return acc, best_score, Kbest
        order = np.lexsort((-quick.ravel()[idx], pl[idx]))  # by player, best first
        rank = np.arange(len(idx)) - np.searchsorted(pl[idx][order], pl[idx][order])
        idx = idx[order][rank < RESCORE]
        where, after = np.stack([Bx[idx], By[idx]], 1), np.stack([ax[idx], ay[idx]], 1)
        score = self.kick_value(where, after, K[idx]) - TIME_COST * K[idx]
        order = np.lexsort((score, pl[idx]))  # by player, best last
        last = np.r_[pl[idx][order][1:] != pl[idx][order][:-1], True]
        chosen = idx[order][last]
        best_score[pl[chosen]] = score[order][last]
        self._keep = np.zeros(P, int)
        self._keep[pl[chosen]] = K[chosen].astype(int) - 1
        Kbest[pl[chosen]] = K[chosen]
        acc[pl[chosen], 0], acc[pl[chosen], 1] = ux[chosen] - mv[pl[chosen], 0], uy[chosen] - mv[pl[chosen], 1]
        return acc, best_score, Kbest


class LLMBrainV11CalcLite(LLMBrainV11Calc):
    """
    LLM-v11-calc planning less often. A calculated plan is exact: followed open-loop, the
    move u lands the touch where it was planned. So while everything goes as planned (the
    ball where it was predicted, a striker where its move puts it), the last plan is
    carried on, and a fresh one is made only every REPLAN_EVERY ticks or when something
    changes (a touch, a bump, a bounce the prediction got slightly wrong, the plan's last
    ticks). The other players' plans and all kick scores are then up to REPLAN_EVERY - 1
    ticks old (they and the opponents may have moved).
    """

    REPLAN_EVERY = 4
