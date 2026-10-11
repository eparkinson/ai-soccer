"""
GPBrain v3: GPBrain v2 (aisoccer/gp/v2.py) with real roles, better kicks and more words.

GP-v2 plateaued on "the nearest player clears through the ball, the other four clump
200 px in front of goal": `role` (the player's index) was only ever a few-pixel nudge,
so one program could not give players different jobs (analysis/league3/gpdive/RESULTS.md).

A v3 program is a list of roles. Each role has two evolvable expressions:

    ["roles",
        ["role", BY, TARGET],     # role 1
        ["role", BY, TARGET],     # role 2
        ...]                      # up to MAX_ROLES roles

- BY (a number per player): who fills the role. Each tick the roles are filled in
  order; each role takes the free player for whom BY is lowest (the first by index on
  a tie). The last role filled takes every player still free, so its BY is unused: with
  3 roles, roles 1 and 2 get one player each and role 3 gets the other three. Only
  the first five roles can be filled; roles further down are dormant (their code does
  nothing until an operator moves them up), and parsimony counts them.
- TARGET (a point): where a player in the role goes (v2's controller: full speed to
  the point, stopping there).

Both are ordinary typed GP trees over the whole vocabulary, so subtree crossover,
subtree mutation and point mutation work on them as before, one role at a time, and
evolve_gp.py adds role-level operators (see `structural`): add a random role, delete
one, duplicate and mutate one, move one in the order, take a whole role from the other
parent, and make a new role by crossing two roles (the BY of one with the TARGET of
the other, or subtree crossover between their expressions).

A program prints as readable Python:

    # roles are filled in order: each takes the free player with the lowest `by`;
    # the last role filled takes everyone left
    def role_1():  # 1 player, by: ball_dist
        return hit(opp_goal)
    def role_2():  # 1 player, by: x(me)
        return save_point
    def role_3():  # the other 3 players
        return point(x(ball) - 200 - 150 * rank, lane_y)

Every v2 (and v1) program is a v3 program (from_v2): with `nearest_to_ball`, it becomes
role 1 "BY 0 if nearest_to_ball else 1" playing the program with nearest_to_ball
fixed to yes, and role 2 (everyone else) playing it with nearest_to_ball fixed to no.
The targets are bit-identical, so GP-v3 starts exactly where GP-v2 is.

New words (in the brain's own frame, attacking x = 1799; v1's and v2's are kept):

  Points
    clear         the point to run at to kick the ball forward into the most open space:
                  of seven directions from straight ahead to 70 degrees either side,
                  the one whose first CLEAR_LENGTH px is furthest from any opponent
                  (capped, with a small preference for straight ahead), among those I
                  can hit without going round the ball; kicked with `hit`
    pass          the point to run at to pass (with `hit`) to the teammate (not me)
                  furthest forward whose lane from the ball is open (no opponent within
                  PASS_OPEN px of it), aiming where it will be when the ball gets there;
                  `clear` if nobody is open
    open_goal     the point in the opponent's goal mouth (of five) whose line from the
                  ball is furthest from any opponent: where to shoot
    threat        the most dangerous opponent: nearest the ball if the ball is in my
                  half, else nearest my goal
  Numbers
    rank          my depth rank among the players in my role (0 = deepest, by x);
                  inside BY (before roles are filled) among the whole team
    lane          my height rank among the players in my role (0 = top, smallest y);
                  inside BY among the whole team
    lane_y        my lane's height: FIELD_HEIGHT x (lane + 1) / (players in the role + 1)
                  (inside BY: over the team of 5)
    ball_dist     my distance to the ball
    reach         ticks until I can meet the ball (the model behind `intercept`)
    goal_dist     my distance to the centre of my goal
    time_left     min(ticks left, 1800) / 1800: 1 until the last 1800 ticks, so it means
                  the same in 1800-tick training games and 36000-tick matches
    score_diff    my goals minus theirs, clipped to +-3, / 3
  Yes/no
    we_have_ball  our player nearest the ball is nearer than their nearest
    shot_coming   the ball's straight path (no bounces) crosses my goal line inside the
                  mouth within HORIZON ticks
  Functions
    hit(aim)      strike(aim) at full speed: strike runs through the contact point only
                  within 45 degrees of the aim and otherwise first stops at a line-up
                  point STAGE px behind it (the review's suspect for giving the ball
                  away); hit runs through within 60 degrees and heads for the line-up or
                  round-the-ball points without slowing (see hit_point)
    mark(p)       the goal-side marking spot of p: MARK_GAP px from p towards my goal
    between(a, b, t)   a + t (b - a), t clipped to [0, 1]

Per-tick cost: the role assignment is one loop over the free players per BY; the new
words are computed only if a live part of the program uses them (clear and pass check
about 35 and 25 ball-to-point lanes against five opponents).
"""

import copy
import json
import math

import numpy as np

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.constants import Constants
from aisoccer.gp import v1, v2
from aisoccer.gp.v1 import FIELD_X, FIELD_Y, NUM, OPP_GOAL, OWN_GOAL, _clip, depth, size  # noqa: F401
from aisoccer.gp.v2 import CONTACT, HORIZON, STAGE, _TICKS, ball_in, predict_xy, reach_times, strike_point

MAX_ROLES = 8
HIT_THROUGH = 60.0  # hit: how far past the contact point to aim, so the touch is at full speed
HIT_CONE = 0.5  # hit: run through when my line to the contact point is within 60 degrees of the aim
HIT_RUN = 120.0  # hit: detour points at least this far away, so the run never slows
CLEAR_LENGTH = 350.0  # clear: how much of each direction must be open
CLEAR_CAP = 150.0  # clear: openness beyond this counts the same
CLEAR_FORWARD = 25.0  # clear: bonus x cos(angle), preferring straight ahead among equally open
CLEAR_ANGLES = (0, -20, 20, -45, 45, -70, 70)
CLEAR_DIRS = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in CLEAR_ANGLES]
PASS_OPEN = float(Constants.PLAYER_RADIUS + Constants.BALL_RADIUS + 16)
GOAL_AIM_YS = [Constants.FIELD_HEIGHT / 2 + d for d in (0.0, -70.0, 70.0, -140.0, 140.0)]
GOAL_CAP = 150.0
MARK_GAP = 2.0 * Constants.PLAYER_RADIUS + 2.0
TIME_WINDOW = 1800
SCORE_CAP = 3
OWN_LINE_X = float(Constants.GOAL_DEPTH)
ROLE_CONTEXT = ("rank", "lane", "lane_y")  # terminals whose value depends on the role filled

TYPE_NAMES = {**v1.TYPE_NAMES, "r": "role", "P": "program"}
GENE_TYPES = ("v", "n", "b")  # the types ordinary GP operators work on


# --- words on numpy arrays (the reference interpreter's) ---


def p_mark(p):
    d = OWN_GOAL - p
    n = np.sqrt(np.sum(d * d, axis=-1, keepdims=True))
    safe = np.where(n > 1e-9, n, 1.0)
    return np.where(n > 1e-9, p + d * (np.minimum(MARK_GAP, n) / safe), p)


def p_between(a, b, t):
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return a + v1._col(t) * (b - a)


NEW_FUNCTIONS = {
    "hit": ("v", ("v",), None, "hit"),
    "mark": ("v", ("v",), p_mark, "mark"),
    "between": ("v", ("v", "v", "n"), p_between, "between"),
}
NEW_TERMINALS = {
    "clear": "v",
    "pass": "v",
    "open_goal": "v",
    "threat": "v",
    "rank": "n",
    "lane": "n",
    "lane_y": "n",
    "ball_dist": "n",
    "reach": "n",
    "goal_dist": "n",
    "time_left": "n",
    "score_diff": "n",
    "we_have_ball": "b",
    "shot_coming": "b",
}
FUNCTIONS = {**v2.FUNCTIONS, **NEW_FUNCTIONS}
TERMINALS = {**v2.TERMINALS, **NEW_TERMINALS}
IMPLICIT = {**v2.IMPLICIT, "hit": ("me", "intercept")}
# How often random programs and mutations draw each word (evolve_gp.Vocab repeats a
# word this many times in its lists; unlisted words count once).
TERMINAL_WEIGHTS = {"intercept": 3, "ball": 2, "clear": 2, "save_point": 2, "me": 2, "ball_dist": 2, "rank": 2}
FUNCTION_WEIGHTS = {"hit": 2, "if_v": 2, "behind": 2, "point": 2}
# BY expressions for new random roles: half the time one of these, else a random tree.
CLAIMS = ["ball_dist", "reach", "goal_dist", ["x", "me"], ["sub", 0.0, ["x", "me"]], ["y", "me"], "role", "rank"]
ROOT_TYPE = "P"


# --- the new words' geometry, on plain floats ---


def hit_point(bx, by, px, py, ax, ay):
    """
    Where a player at (px, py) runs to kick a ball at (bx, by) towards (ax, ay), never
    slowing down: strike_point's plan (through the contact point C when lined up; else
    line up behind C, or go round the ball from in front) with a wider cone (HIT_CONE,
    60 degrees, not 45) and HIT_THROUGH past C, and with every detour point pushed out
    to at least HIT_RUN px from the player, so it keeps full speed on the way round
    instead of stopping at the line-up point as strike does. (A run through C from
    wider than about 60 degrees is a glancing blow: the ball leaves slowly.)
    """
    dx, dy = ax - bx, ay - by
    n = math.sqrt(dx * dx + dy * dy)
    if n < 1e-9:
        return bx, by
    dx, dy = dx / n, dy / n
    cx, cy = bx - dx * CONTACT, by - dy * CONTACT
    wx, wy = cx - px, cy - py
    n = math.sqrt(wx * wx + wy * wy)
    wx, wy = (wx / n, wy / n) if n > 1e-9 else (dx, dy)
    if wx * dx + wy * dy > HIT_CONE:
        return cx + wx * HIT_THROUGH, cy + wy * HIT_THROUGH
    if (px - bx) * dx + (py - by) * dy < -CONTACT:
        tx, ty = cx - dx * STAGE, cy - dy * STAGE
    else:
        sx, sy = -dy, dx
        if (px - bx) * sx + (py - by) * sy < 0:
            sx, sy = -sx, -sy
        tx, ty = bx + sx * (CONTACT + 40) - dx * (CONTACT + 15), by + sy * (CONTACT + 40) - dy * (CONTACT + 15)
    ox, oy = tx - px, ty - py
    n = math.sqrt(ox * ox + oy * oy)
    if n < HIT_RUN:
        if n < 1e-9:
            return tx, ty
        ox, oy = ox * HIT_RUN / n, oy * HIT_RUN / n
    return px + ox, py + oy


def lane_clearance(bx, by, ex, ey, opps):
    """The distance from the segment (bx, by)-(ex, ey) to the nearest of the points opps."""
    sx, sy = ex - bx, ey - by
    length = sx * sx + sy * sy
    if length < 1e-9:
        length = 1.0
    best = math.inf
    for ox, oy in opps:
        rx, ry = ox - bx, oy - by
        t = (rx * sx + ry * sy) / length
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        ex_, ey_ = rx - t * sx, ry - t * sy
        d = ex_ * ex_ + ey_ * ey_
        if d < best:
            best = d
    return math.sqrt(best)


def clear_points(ball, meets, me, opps):
    """Every player's `clear` point (see the module docstring)."""
    bx, by = ball
    scored = []
    for k, (dx, dy) in enumerate(CLEAR_DIRS):
        open_ = lane_clearance(bx, by, bx + dx * CLEAR_LENGTH, by + dy * CLEAR_LENGTH, opps)
        scored.append((-(min(open_, CLEAR_CAP) + CLEAR_FORWARD * dx), k))
    order = [CLEAR_DIRS[k] for _, k in sorted(scored)]
    out = []
    for (kx, ky), (px, py) in zip(meets, me):
        chosen = order[0]
        for dx, dy in order:
            if (kx - dx * CONTACT - px) * dx + (ky - dy * CONTACT - py) * dy > 0.0:
                chosen = dx, dy
                break
        out.append(hit_point(kx, ky, px, py, kx + chosen[0] * 100.0, ky + chosen[1] * 100.0))
    return out


def pass_points(ball, meets, me, vel, opps, clears):
    """Every player's `pass` point (see the module docstring); `clears` are the fallbacks."""
    bx, by = ball
    speed = Constants.MAX_BALL_VELOCITY
    open_mates = []
    for j, ((mx, my), (vx, vy)) in enumerate(zip(me, vel)):
        lead = math.sqrt((mx - bx) ** 2 + (my - by) ** 2) / speed
        ax, ay = mx + vx * lead, my + vy * lead
        if lane_clearance(bx, by, ax, ay, opps) > PASS_OPEN:
            open_mates.append((-ax, j, ax, ay))
    open_mates.sort()
    out = []
    for i, ((kx, ky), (px, py)) in enumerate(zip(meets, me)):
        target = next(((ax, ay) for _, j, ax, ay in open_mates if j != i), None)
        out.append(clears[i] if target is None else hit_point(kx, ky, px, py, *target))
    return out


def open_goal_point(ball, opps):
    bx, by = ball
    best, best_y = -1.0, GOAL_AIM_YS[0]
    for y in GOAL_AIM_YS:
        c = min(lane_clearance(bx, by, float(OPP_GOAL[0]), y, opps), GOAL_CAP)
        if c > best:
            best, best_y = c, y
    return float(OPP_GOAL[0]), best_y


def threat_point(ball, opps):
    bx, by = ball
    gx, gy = (bx, by) if bx < FIELD_X / 2 else (float(OWN_GOAL[0]), float(OWN_GOAL[1]))
    return opps[v1._nearest(opps, gx, gy)]


def shot_coming(bx, by, vx, vy):
    if not (vx < 0.0 and bx > OWN_LINE_X):
        return False
    t = (bx - OWN_LINE_X) / -vx
    y = by + vy * t
    return t <= HORIZON and Constants.GOAL_Y_MIN < y < Constants.GOAL_Y_MAX


def reach_ticks(ball, ball_vel, me, vel):
    """Every player's `reach`: ticks until it can meet the ball (as `intercept` finds it)."""
    xs, ys = predict_xy(*ball, *ball_vel)
    me, vel = np.asarray(me, dtype=float), np.asarray(vel, dtype=float)
    if xs[0] == xs[1] and ys[0] == ys[1]:
        return reach_times(me, vel, xs[:1], ys[:1])[:, 0].tolist()
    late = reach_times(me, vel, xs, ys) - _TICKS[: len(xs)]
    ok = late <= 0.0
    index = np.where(ok.any(axis=1), ok.argmax(axis=1), late.argmin(axis=1))
    rows = np.arange(len(me))
    return (index + np.maximum(late[rows, index], 0.0)).astype(float).tolist()


def time_left(tick, game_time):
    """min(ticks left, TIME_WINDOW) / TIME_WINDOW; the game length is round(tick / game_time)."""
    if not game_time or game_time <= 0:
        return 1.0
    left = round(tick / game_time) - tick
    return (0.0 if left < 0 else (TIME_WINDOW if left > TIME_WINDOW else left)) / TIME_WINDOW


def ranks(values, members):
    """Each member's rank by value among the members (ties by index), as {member: rank}."""
    order = sorted(members, key=lambda i: (values[i], i))
    return {i: k for k, i in enumerate(order)}


def role_ranks(job, me):
    """rank, lane and lane_y of every player among the players in its role."""
    rank, lane, lane_y = [0.0] * NUM, [0.0] * NUM, [0.0] * NUM
    groups = {}
    for i, j in enumerate(job):
        groups.setdefault(j, []).append(i)
    xs, ys = [p[0] for p in me], [p[1] for p in me]
    for members in groups.values():
        rx, ry = ranks(xs, members), ranks(ys, members)
        for i in members:
            rank[i], lane[i] = float(rx[i]), float(ry[i])
            lane_y[i] = FIELD_Y * (ry[i] + 1) / (len(members) + 1)
    return rank, lane, lane_y


TEAM_JOB = [0] * NUM  # inside BY, rank / lane / lane_y are over the whole team


# --- the program structure ---


def roles_of(program):
    return program[1:]


def active_count(program):
    """How many roles can be filled: the first five at most."""
    return min(len(program) - 1, NUM)


def node_type(node):
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return "n"
    if isinstance(node, str):
        return TERMINALS[node]
    if node[0] == "roles":
        return "P"
    if node[0] == "role":
        return "r"
    return FUNCTIONS[node[0]][0]


def _validate_tree(node, want):
    if isinstance(node, bool):
        raise ValueError("booleans are not constants")
    if isinstance(node, (int, float)):
        if not math.isfinite(node):
            raise ValueError("non-finite constant")
    elif isinstance(node, str):
        if node not in TERMINALS:
            raise ValueError(f"unknown terminal {node!r}")
    elif isinstance(node, list) and node and node[0] in FUNCTIONS:
        _, args, _, _ = FUNCTIONS[node[0]]
        if len(node) != len(args) + 1:
            raise ValueError(f"{node[0]} takes {len(args)} arguments")
        for child, t in zip(node[1:], args):
            _validate_tree(child, t)
    else:
        raise ValueError(f"not a program node: {node!r}")
    if node_type(node) != want:
        raise ValueError(f"{node!r} is a {TYPE_NAMES[node_type(node)]}, not a {TYPE_NAMES[want]}")


def validate(node, want="P"):
    """Raise ValueError unless node is a well-typed v3 program (or, with want, subtree)."""
    if want == "P":
        if not (isinstance(node, list) and node and node[0] == "roles"):
            raise ValueError(f"a v3 program is ['roles', role, ...], not {str(node)[:60]}")
        if not 1 <= len(node) - 1 <= MAX_ROLES:
            raise ValueError(f"a program has 1 to {MAX_ROLES} roles, not {len(node) - 1}")
        for r in node[1:]:
            validate(r, "r")
    elif want == "r":
        if not (isinstance(node, list) and len(node) == 3 and node[0] == "role"):
            raise ValueError(f"a role is ['role', by, target], not {str(node)[:60]}")
        _validate_tree(node[1], "n")
        _validate_tree(node[2], "v")
    else:
        _validate_tree(node, want)


def _terminals(node, found):
    if isinstance(node, str):
        found.add(node)
    elif isinstance(node, list):
        found.update(IMPLICIT.get(node[0], ()))
        for c in node[1:]:
            _terminals(c, found)
    return found


def live_parts(program):
    """(the BY expressions that fill roles, the TARGETs of the roles that can be filled)."""
    roles, n = roles_of(program), active_count(program)
    return [r[1] for r in roles[: n - 1]], [r[2] for r in roles[:n]]


def terminals_used(node, found=None):
    """The terminals the live code reads (dormant roles and the last BY are skipped)."""
    found = set() if found is None else found
    if isinstance(node, list) and node and node[0] == "roles":
        claims, targets = live_parts(node)
        for part in claims + targets:
            _terminals(part, found)
        return found
    return _terminals(node, found)


# --- v2 programs as v3 programs ---


def _fold(node, near):
    """node with nearest_to_ball replaced by the Python bool `near`, folded away."""
    if node == "nearest_to_ball":
        return near
    if not isinstance(node, list):
        return node
    name, args = node[0], [_fold(c, near) for c in node[1:]]
    if name in ("if_v", "if_n") and isinstance(args[0], bool):
        return args[1] if args[0] else args[2]
    if name == "not" and isinstance(args[0], bool):
        return not args[0]
    if name in ("and", "or"):
        a, b = args
        stop = name == "or"  # the value that decides it: yes for or, no for and
        if (isinstance(a, bool) and a == stop) or (isinstance(b, bool) and b == stop):
            return stop
        if isinstance(a, bool):
            return b
        if isinstance(b, bool):
            return a
    return [name] + args


def from_v2(program):
    """
    A v1 or v2 program as a v3 program that moves every player identically: with
    nearest_to_ball, role 1 (BY: 0 if nearest_to_ball else 1, so the nearest player,
    the first on a tie, as v2) plays it with nearest_to_ball = yes and role 2 (everyone
    else) with nearest_to_ball = no; without, one role plays it for all.
    """
    v2.validate(program)
    if "nearest_to_ball" not in v1.terminals_used(program):
        return ["roles", ["role", 0.0, copy.deepcopy(program)]]
    return [
        "roles",
        ["role", ["if_n", "nearest_to_ball", 0.0, 1.0], _fold(program, True)],
        ["role", 0.0, _fold(program, False)],
    ]


def convert(program):
    """A v3 program as it is; a v1 or v2 program translated (from_v2)."""
    if isinstance(program, list) and program and program[0] == "roles":
        return program
    return from_v2(program)


# --- terminal values for one tick ---


def python_terminals(brain, names):
    """The terminal values for one tick as plain Python lists, one entry per player."""
    old = [n for n in names if n in v2.TERMINALS or n == "ball_path"]
    T = v2.python_terminals(brain, old)
    new = [n for n in names if n not in T]
    if not new:
        return T
    me = brain.my_players_pos.tolist()
    ball = tuple(float(v) for v in np.asarray(brain.ball_pos).reshape(2))
    ball_vel = tuple(float(v) for v in np.asarray(brain.ball_vel).reshape(2))
    opps = brain.opp_players_pos.tolist()
    meets = T.get("intercept")
    if meets is None and ("clear" in new or "pass" in new):
        meets = v2.intercept_points(*predict_xy(*ball, *ball_vel), brain.my_players_pos, brain.my_players_vel)
    clears = None
    for name in new:
        if name in ("clear", "pass"):
            if clears is None:
                clears = clear_points(ball, meets, me, opps)
            T[name] = clears if name == "clear" else pass_points(
                ball, meets, me, brain.my_players_vel.tolist(), opps, clears
            )
        elif name == "open_goal":
            T[name] = [open_goal_point(ball, opps)] * NUM
        elif name == "threat":
            T[name] = [tuple(threat_point(ball, opps))] * NUM
        elif name in ROLE_CONTEXT:
            rank, lane, lane_y = role_ranks(TEAM_JOB, me)
            T[name] = {"rank": rank, "lane": lane, "lane_y": lane_y}[name]
        elif name == "ball_dist":
            T[name] = [math.sqrt((x - ball[0]) ** 2 + (y - ball[1]) ** 2) for x, y in me]
        elif name == "reach":
            T[name] = reach_ticks(ball, ball_vel, brain.my_players_pos, brain.my_players_vel)
        elif name == "goal_dist":
            gx, gy = float(OWN_GOAL[0]), float(OWN_GOAL[1])
            T[name] = [math.sqrt((x - gx) ** 2 + (y - gy) ** 2) for x, y in me]
        elif name == "time_left":
            T[name] = [time_left(getattr(brain, "_tick", 0), brain.game_time)] * NUM
        elif name == "score_diff":
            d = (brain.my_score or 0) - (brain.opp_score or 0)
            T[name] = [max(-SCORE_CAP, min(SCORE_CAP, d)) / SCORE_CAP] * NUM
        elif name == "we_have_ball":
            ours = min((x - ball[0]) ** 2 + (y - ball[1]) ** 2 for x, y in me)
            theirs = min((x - ball[0]) ** 2 + (y - ball[1]) ** 2 for x, y in opps)
            T[name] = [ours < theirs] * NUM
        elif name == "shot_coming":
            T[name] = [shot_coming(*ball, *ball_vel)] * NUM
    return T


# --- the reference interpreter (numpy, all players at once) ---


def interpret(node, values):
    """Evaluate a v3 expression (not a whole program) on numpy arrays; `values` by terminal."""
    if isinstance(node, (int, float)):
        return float(node)
    if isinstance(node, str):
        return values[node]
    name = node[0]
    if name in ("strike", "hit"):
        aim = np.broadcast_to(interpret(node[1], values), (NUM, 2))
        me, meet, f = values["me"], values["intercept"], (strike_point if name == "strike" else hit_point)
        return np.array([f(*meet[i], *me[i], *aim[i]) for i in range(NUM)])
    if name == "ball_in":
        n = np.broadcast_to(interpret(node[1], values), (NUM,))
        return np.array([ball_in(values["ball_path"], float(k)) for k in n])
    impl = FUNCTIONS[name][2]
    return impl(*(interpret(c, values) for c in node[1:]))


def terminal_arrays(brain, names):
    values = v1.terminal_values(brain, [n for n in names if n in v1.TERMINALS])
    for name, value in python_terminals(brain, [n for n in names if n not in v1.TERMINALS]).items():
        if name == "ball_path":
            values[name] = value
        else:
            values[name] = np.array(value, dtype=bool if TERMINALS[name] == "b" else float)
    return values


def assign(program, values):
    """The role index of every player (the interpreter's version of the filling rule)."""
    claims, _ = live_parts(program)
    n = active_count(program)
    job = [n - 1] * NUM
    free = list(range(NUM))
    for k, claim in enumerate(claims):
        c = np.broadcast_to(np.asarray(interpret(claim, values), dtype=float), (NUM,))
        best = free[int(np.argmin(c[free]))]
        job[best] = k
        free.remove(best)
    return job


def interpret_targets(program, brain):
    """Every player's target point by the interpreter, clipped to the field."""
    names = terminals_used(program) | {"me"}
    values = terminal_arrays(brain, names)
    job = assign(program, values)
    rank, lane, lane_y = role_ranks(job, values["me"].tolist())
    role_values = dict(values, rank=np.array(rank), lane=np.array(lane), lane_y=np.array(lane_y))
    target = np.zeros((NUM, 2))
    _, targets = live_parts(program)
    jobs = np.array(job)
    for k, t in enumerate(targets):
        t = np.broadcast_to(np.asarray(interpret(t, role_values), dtype=float), (NUM, 2))
        target[jobs == k] = t[jobs == k]
    return np.clip(target, (0.0, 0.0), (FIELD_X, FIELD_Y))


# --- compiling ---


class _CodeGen(v2._CodeGen):
    """v2's compiler, plus v3's words."""

    def emit(self, node, pad):
        if isinstance(node, str):
            return (f"{node}_x", f"{node}_y") if TERMINALS[node] == "v" else node
        if isinstance(node, list) and node[0] in NEW_FUNCTIONS:
            args = [self.emit(c, pad) for c in node[1:]]
            t = self.new()
            if node[0] == "hit":
                call = f"_hit(intercept_x, intercept_y, me_x, me_y, {args[0][0]}, {args[0][1]})"
            elif node[0] == "mark":
                call = f"_mark({args[0][0]}, {args[0][1]})"
            else:
                call = f"_between({args[0][0]}, {args[0][1]}, {args[1][0]}, {args[1][1]}, {args[2]})"
            self.lines.append(pad + f"{t}x, {t}y = {call}")
            return t + "x", t + "y"
        return super().emit(node, pad)


def _mark(px, py):
    dx, dy = float(OWN_GOAL[0]) - px, float(OWN_GOAL[1]) - py
    n = math.sqrt(dx * dx + dy * dy)
    if n > 1e-9:
        s = min(MARK_GAP, n) / n
        return px + dx * s, py + dy * s
    return px, py


def _between(ax, ay, bx, by, t):
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return ax + t * (bx - ax), ay + t * (by - ay)


def _unpack(names, pad, role_context):
    lines = []
    for name in sorted(names):
        if name == "ball_path":
            continue
        source = f"{name}_r[i]" if role_context and name in ROLE_CONTEXT else f"{name}_all[i]"
        if TERMINALS[name] == "v":
            lines.append(f"{pad}{name}_x, {name}_y = {source}")
        else:
            lines.append(f"{pad}{name} = {source}")
    return lines


def program_source(program):
    """The Python source of the compiled program (see v1.program_source): fill the roles, then targets."""
    claims, targets = live_parts(program)
    n = active_count(program)
    names = sorted(terminals_used(program) | {"me"})
    lines = ["def _program(T):"]
    for name in names:
        lines.append("    ball_path = T['ball_path']" if name == "ball_path" else f"    {name}_all = T[{name!r}]")
    lines += [f"    job = [{n - 1}] * {NUM}", f"    free = list(range({NUM}))"]
    for k, claim in enumerate(claims):
        gen = _CodeGen()
        value = gen.emit(claim, "        ")
        lines += ["    bv = _INF", "    best = -1", "    for i in free:"]
        lines += _unpack(_terminals(claim, set()), "        ", False)
        lines += gen.lines
        lines += [f"        if {value} < bv:", f"            bv = {value}", "            best = i"]
        lines += ["    if best < 0:", "        best = free[0]", f"    job[best] = {k}", "    free.remove(best)"]
    target_names = set()
    for t in targets:
        _terminals(t, target_names)
    if target_names & set(ROLE_CONTEXT):
        lines.append("    rank_r, lane_r, lane_y_r = _role_ranks(job, me_all)")
    lines += ["    out = []", f"    for i in range({NUM}):", "        j = job[i]"]
    for k, t in enumerate(targets):
        gen = _CodeGen()
        value = gen.emit(t, "            ")
        lines.append(f"        {'if' if k == 0 else 'elif'} j == {k}:")
        lines += _unpack(_terminals(t, set()), "            ", True)
        lines += gen.lines
        lines.append(f"            x = {value[0]}; y = {value[1]}")
    lines += [
        f"        out.append((0.0 if x < 0.0 else ({FIELD_X!r} if x > {FIELD_X!r} else x),"
        f" 0.0 if y < 0.0 else ({float(FIELD_Y)!r} if y > {float(FIELD_Y)!r} else y)))",
        "    return out",
    ]
    return "\n".join(lines) + "\n"


_CACHE = {}


def compile_program(program):
    """The program as a Python function and the terminals it reads; cached."""
    key = json.dumps(program)
    fn = _CACHE.get(key)
    if fn is None:
        scope = {
            "_clip": _clip, "_sqrt": math.sqrt, "_strike": strike_point, "_ball_in": ball_in, "_hit": hit_point,
            "_mark": _mark, "_between": _between, "_role_ranks": role_ranks, "_INF": math.inf,
        }
        exec(compile(program_source(program), "<gp v3 program>", "exec"), scope)
        fn = (scope["_program"], sorted(terminals_used(program) | {"me"}))
        if len(_CACHE) > 4096:
            _CACHE.clear()
        _CACHE[key] = fn
    return fn


# --- printing ---


def expression(node, parent_prec=-1):
    """A v3 expression as one Python-like expression."""
    if isinstance(node, (int, float)):
        text = v1._fmt_const(node)
        return f"({text})" if text.startswith("-") and parent_prec >= 0 else text
    if isinstance(node, str):
        return node
    name, args = node[0], node[1:]
    form = FUNCTIONS[name][3]
    if form == "if":
        text = f"{expression(args[1], 2)} if {expression(args[0], 2)} else {expression(args[2], 2)}"
        return f"({text})" if parent_prec >= 0 else text
    if form == "not":
        return f"not {expression(args[0], 2)}"
    if isinstance(form, tuple):
        op, prec = form
        text = f"{expression(args[0], prec)} {op} {expression(args[1], prec + 1)}"
        return f"({text})" if prec < parent_prec else text
    return f"{form}({', '.join(expression(a) for a in args)})"


def _statements(node, indent):
    pad = "    " * indent
    if isinstance(node, list) and node[0] == "if_v":
        return [f"{pad}if {expression(node[1])}:"] + _statements(node[2], indent + 1) + _statements(node[3], indent)
    return [f"{pad}return {expression(node)}"]


def to_text(program):
    """The program as readable Python: one function per role, in the order they are filled."""
    roles, n = roles_of(program), active_count(program)
    lines = [
        "# Roles are filled in order: each takes the free player with the lowest `by`;",
        "# the last role filled takes everyone left. Every player runs to its role's point.",
    ]
    for k, (_, by, target) in enumerate(roles):
        if k < n - 1:
            note = f"1 player, by: {expression(by)}"
        elif k == n - 1:
            note = "everyone left" if k else "all players"
        else:
            note = "dormant (only 5 roles can be filled)"
        lines.append(f"def role_{k + 1}():  # {note}")
        lines += _statements(target, 1)
    return "\n".join(lines)


# --- random programs and role-level operators (for evolve_gp.py) ---


def random_role(rng, tree, max_depth):
    """A new role: a BY (half the time one of CLAIMS) and a random TARGET; tree(type, max_depth) makes subtrees."""
    by = copy.deepcopy(rng.choice(CLAIMS)) if rng.random() < 0.5 else tree("n", min(max_depth, 3))
    return ["role", by, tree("v", max_depth)]


def random_program(rng, max_depth, full, tree):
    """A random program of 2 to 5 roles; tree(type, max_depth, full) makes random subtrees."""
    return ["roles"] + [
        random_role(rng, lambda t, d: tree(t, d, full), max_depth) for _ in range(rng.randint(2, 5))
    ]


def role_operators():
    return ["add", "delete", "duplicate", "move", "take", "combine"]


def structural(rng, a, b, tree, mutate, cross, max_roles=MAX_ROLES):
    """
    One role-level change to program a (b is another parent), as (child, operator name):

    - add: a new random role at a random place in the order,
    - delete: remove one role (if there are two or more),
    - duplicate: copy a role and mutate the copy (mutate(program) is an ordinary
      mutation), placed right after the original or at a random place,
    - move: move a role to another place in the order,
    - take: a whole role of b replaces one of a's roles or is inserted,
    - combine: a new role crossing two roles (of a, or one of a and one of b): the BY
      of one and the TARGET of the other, or cross(p, q), ordinary subtree crossover,
      between the two roles; inserted at a random place.

    Operators that would break the 1..max_roles limit fall back to another.
    """
    a = copy.deepcopy(a)
    roles = a[1:]
    ops = role_operators()
    if len(roles) >= max_roles:
        ops = [o for o in ops if o not in ("add", "duplicate", "combine")]
    if len(roles) <= 1:
        ops = [o for o in ops if o not in ("delete", "move")]
    op = rng.choice(ops)
    if op == "add":
        roles.insert(rng.randint(0, len(roles)), random_role(rng, tree, rng.randint(2, 5)))
    elif op == "delete":
        del roles[rng.randrange(len(roles))]
    elif op == "duplicate":
        k = rng.randrange(len(roles))
        new = mutate(["roles", copy.deepcopy(roles[k])])[1]
        roles.insert(k + 1 if rng.random() < 0.5 else rng.randint(0, len(roles)), new)
    elif op == "move":
        k = rng.randrange(len(roles))
        role = roles.pop(k)
        roles.insert(rng.choice([p for p in range(len(roles) + 1) if p != k]), role)
    elif op == "take":
        donor = copy.deepcopy(rng.choice(b[1:]))
        if len(roles) < max_roles and rng.random() < 0.5:
            roles.insert(rng.randint(0, len(roles)), donor)
        else:
            roles[rng.randrange(len(roles))] = donor
    else:  # combine
        p = copy.deepcopy(rng.choice(roles))
        q = copy.deepcopy(rng.choice(b[1:] if rng.random() < 0.5 else roles))
        if rng.random() < 0.5:
            new = ["role", p[1], q[2]]
        else:
            new = cross(["roles", p], ["roles", q])[1]
        roles.insert(rng.randint(0, len(roles)), new)
    return ["roles"] + roles, op


# --- the brain ---


class GPBrainV3(v1.GPBrain):
    """GPBrain (same controller) playing a v3 program; v1 and v2 programs are translated (from_v2)."""

    def __init__(self, name=None, program="ball"):
        AbstractBrain.__init__(self, name)
        program = convert(program)
        validate(program)
        self.program = program
        self._fn, self._names = compile_program(program)
        self._tick = 0
        self._last_time = -math.inf

    def targets(self):
        return self._fn(python_terminals(self, self._names))

    def do_move(self):
        if self.game_time is not None:
            if self.game_time < self._last_time:  # a new game
                self._tick = 0
            self._last_time = self.game_time
        moves = super().do_move()
        self._tick += 1
        return moves

    def __str__(self):
        return to_text(self.program)
