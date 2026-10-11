"""
GPBrain v2: GPBrain v1 (aisoccer/gp/v1.py) with seven more words, so evolution can
express meeting a moving ball, aiming a kick and keeping goal, which v1 cannot (v1 can only say
"go to this point", and league 3's GP plateaued on "nearest player chases the ball").

Everything else is v1: one program returns every player's target point, and every
player runs at full speed towards its target (the same controller). Every v1 program
is a v2 program and plays exactly the same moves under GPBrainV2, so v2 can be seeded
with v1's best.

New words (all per player, in the brain's own frame: it attacks x = 1799):

  Points
    intercept     where I can first meet the ball: the earliest point on the ball's
                  predicted path (straight lines, bouncing off the walls; the ball has
                  no friction) that I can reach, touching the ball, by the time it
                  gets there; if I cannot catch it within HORIZON ticks, the point I
                  come closest to catching it
    front_mate    the teammate (not me) nearest the goal I attack: someone to pass to
    bank_top      the goal centre mirrored in the top wall: kicked at, the ball bounces
                  off the top wall into the goal
    bank_bottom   the same off the bottom wall
    save_point    where a keeper should stand: on the keeper line (a player radius
                  and a bit in front of my goal line), where the ball's predicted
                  path crosses it if the ball is heading at my goal, else on the line
                  from my goal's centre to the ball; kept inside the posts
  Functions
    strike(aim)   the point to run at to kick the ball (where I can meet it,
                  `intercept`) towards `aim`: from behind the ball, a point straight
                  through the ball's back, so I hit it at full speed; otherwise round
                  the ball to behind it first
    ball_in(n)    where the ball will be in n ticks (0 to HORIZON) if nobody touches it

Why strike returns a point *through* the ball rather than the contact point: the v1
controller slows down to stop at its target, so running to the contact point would
only tap the ball. A target past the contact point, on the line from me through it,
means I am still at full speed when I touch the ball, and the first touch happens at
the contact point as long as I come from its back half (see strike_point), so the
ball leaves towards `aim`. The controller stays v1's, so v1 programs are unaffected.

The ball prediction and time-to-reach models are aisoccer/tactics/skills_v1.py's
(predict_ball, times_to_reach), rewritten to run every tick for all players at once
(see predict_xy and reach_times), with the game's current speed and acceleration caps
(skills_v1 copied them at import and assumes acceleration 1).
"""

import json
import math

import numpy as np

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.constants import Constants
from aisoccer.gp import v1
from aisoccer.gp.v1 import BIG, FIELD_X, FIELD_Y, NUM, OPP_GOAL, OWN_GOAL, TYPE_NAMES, _clip, depth, size  # noqa: F401
from aisoccer.tactics.skills_v1 import BALL_R, CONTACT, GOAL_LINE, OWN_GOAL_LINE
from aisoccer.tactics.skills_v1 import H as WALL_Y

HORIZON = 120  # ticks of ball prediction
THROUGH = 30.0  # strike: how far past the contact point to aim, to hit at full speed
STAGE = 60.0  # strike: how far behind the contact point to line up
CONE = 0.7  # strike: run through when my line to the contact point is within 45 degrees of the aim
BANK_TOP = (float(OPP_GOAL[0]), 2 * BALL_R - float(OPP_GOAL[1]))
BANK_BOTTOM = (float(OPP_GOAL[0]), 2 * (FIELD_Y - BALL_R) - float(OPP_GOAL[1]))
# save_point: the keeper line, a player's centre just clear of the goal line, and the
# range of heights on it inside the posts. A keeper PLAYER_RADIUS inside a post still
# blocks a ball grazing the post (it reaches PLAYER_RADIUS + BALL_RADIUS from its centre).
KEEPER_X = float(Constants.GOAL_DEPTH + Constants.PLAYER_RADIUS + 4)
KEEPER_Y_MIN = float(Constants.GOAL_Y_MIN + Constants.PLAYER_RADIUS)
KEEPER_Y_MAX = float(Constants.GOAL_Y_MAX - Constants.PLAYER_RADIUS)
SHOT_WIDE = float(Constants.PLAYER_RADIUS)  # a path this far outside a post is still a shot

NEW_FUNCTIONS = {
    "strike": ("v", ("v",), None, "strike"),
    "ball_in": ("v", ("n",), None, "ball_in"),
}
NEW_TERMINALS = {"intercept": "v", "front_mate": "v", "bank_top": "v", "bank_bottom": "v", "save_point": "v"}
FUNCTIONS = {**v1.FUNCTIONS, **NEW_FUNCTIONS}
TERMINALS = {**v1.TERMINALS, **NEW_TERMINALS}
# Values a word reads besides its arguments (the compiled code needs them per player).
# "ball_path" is hidden: not a word, just the prediction ball_in reads.
IMPLICIT = {"strike": ("me", "intercept"), "ball_in": ("ball_path",)}


# --- the new words' geometry, on plain floats ---


_TICKS = np.arange(HORIZON + 1, dtype=float)
_X_LO, _X_HI = OWN_GOAL_LINE + BALL_R, GOAL_LINE - BALL_R
_Y_LO, _Y_HI = BALL_R, WALL_Y - BALL_R


def _fold(p, lo, hi):
    """Positions on an unbounded line folded into [lo, hi], as if reflected by walls at lo and hi."""
    w = hi - lo
    return (lo + w) - np.abs(np.mod(p - lo, 2 * w) - w)


def predict_xy(bx, by, vx, vy):
    """
    The ball's x and y now and after each of the next HORIZON ticks (two arrays of
    HORIZON + 1), moving in straight lines (there is no friction) and reflecting off
    the side walls and the goal lines, ignoring players. That is
    skills_v1.predict_ball's model with exact reflections: it bounces a ball as soon
    as it passes a wall, so the two differ by up to a tick's travel after each bounce.
    A ball crossing a goal line in the mouth is a goal, so its path beyond is moot.
    """
    if vx == 0.0 and vy == 0.0:
        return np.full(HORIZON + 1, bx), np.full(HORIZON + 1, by)
    return _fold(bx + vx * _TICKS, _X_LO, _X_HI), _fold(by + vy * _TICKS, _Y_LO, _Y_HI)


def predict_path(bx, by, vx, vy):
    """predict_xy as a list of (x, y) points."""
    xs, ys = predict_xy(bx, by, vx, vy)
    return list(zip(xs.tolist(), ys.tolist()))


def reach_times(me, vel, xs, ys):
    """
    Rough ticks for each player (rows of the arrays me, vel) to be touching a ball at
    each point (xs[k], ys[k]) (columns): skills_v1.times_to_reach's model (accelerate
    to top speed, then run; plus time to cancel sideways velocity) with the game's
    current caps, simplified to its cruising case d / top + (top - v0)^2 / (2 acc top),
    which overestimates short runs by at most top / (2 acc) = 2.5 ticks.
    Written on x and y separately: it runs every tick, for every player and point.
    """
    acc, top = Constants.MAX_ACCELERATION, Constants.MAX_PLAYER_VELOCITY
    vx, vy = vel[:, :1], vel[:, 1:]
    ox, oy = xs - me[:, :1], ys - me[:, 1:]
    dist = np.sqrt(ox * ox + oy * oy)
    np.maximum(dist, 1e-9, out=dist)
    v0 = (ox * vx + oy * vy) / dist  # my speed towards the point
    sideways = np.sqrt(np.maximum((vx * vx + vy * vy) - v0 * v0, 0.0))
    np.minimum(v0, top, out=v0)
    v0 -= top
    need = np.maximum(dist - CONTACT, 0.0)
    need *= 1 / top
    need += v0 * v0 * (1 / (2 * acc * top)) + sideways * (1 / acc)
    need[dist <= CONTACT] = 0.0
    return need


def intercept_points(xs, ys, me, vel):
    """
    Every player's intercept (see the module docstring): the first predicted point
    (index = ticks from now) the player can reach in time, else the one it misses by
    the fewest ticks.
    """
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if xs[0] == xs[1] and ys[0] == ys[1]:  # a still ball stays put
        return [(float(xs[0]), float(ys[0]))] * len(me)
    late = reach_times(np.asarray(me, dtype=float), np.asarray(vel, dtype=float), xs, ys)
    late -= _TICKS[: len(xs)]
    ok = late <= 0.0
    index = np.where(ok.any(axis=1), ok.argmax(axis=1), late.argmin(axis=1))
    return list(zip(xs[index].tolist(), ys[index].tolist()))


def strike_point(bx, by, px, py, ax, ay):
    """
    Where a player at (px, py) should run to kick a ball at (bx, by) towards (ax, ay).

    The ball leaves along the line from the player's centre to the ball's at the
    touch, so the player must touch it at the contact point C, CONTACT behind the
    ball on the aim line. Running along any line through C that approaches from C's
    side (at less than 90 degrees to the aim) touches the ball first at C. Within 45
    degrees (CONE), the player runs at a point THROUGH past C on that line, so it is
    still at full speed there and hits the ball hard.
    Behind the ball but at a wider angle, it lines up STAGE behind C first; beside or
    in front of the ball, it goes round the ball on its own side.
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
    if wx * dx + wy * dy > CONE:
        return cx + wx * THROUGH, cy + wy * THROUGH
    if (px - bx) * dx + (py - by) * dy < -CONTACT:
        return cx - dx * STAGE, cy - dy * STAGE
    sx, sy = -dy, dx
    if (px - bx) * sx + (py - by) * sy < 0:
        sx, sy = -sx, -sy
    return bx + sx * (CONTACT + 40) - dx * (CONTACT + 15), by + sy * (CONTACT + 40) - dy * (CONTACT + 15)


def ball_in(path, n):
    """The predicted ball position n ticks ahead (n clipped to 0..HORIZON, rounded down); path = (xs, ys)."""
    k = 0 if not n > 0 else (HORIZON if n >= HORIZON else int(n))
    return path[0][k], path[1][k]


def front_mates(me):
    """For each player, the teammate (not itself) nearest the goal it attacks."""
    gx, gy = OPP_GOAL.tolist()
    order = sorted(range(NUM), key=lambda i: (me[i][0] - gx) ** 2 + (me[i][1] - gy) ** 2)
    first, second = me[order[0]], me[order[1]]
    return [second if i == order[0] else first for i in range(NUM)]


def save_point(bx, by, vx, vy):
    """
    Where a keeper should stand (see the module docstring). A shot is a ball that the
    prediction (predict_xy's straight lines and wall bounces, within HORIZON ticks)
    takes across the keeper line and then to the goal line within SHOT_WIDE of the
    mouth; the keeper goes where it crosses the keeper line, which for an angled shot
    is well away from where it crosses the goal line. Otherwise it cuts the ball's
    angle to the goal. Either way the height is clipped to KEEPER_Y_MIN..KEEPER_Y_MAX.
    Whether the keeper can get there in time is not considered.
    """
    y = None
    if vx < 0.0 and bx > KEEPER_X and (bx - KEEPER_X) <= -vx * HORIZON:
        # heading left, it reaches x = KEEPER_X, then the goal line, before any x bounce
        at_keeper = float(_fold(by + vy * (bx - KEEPER_X) / -vx, _Y_LO, _Y_HI))
        at_goal = float(_fold(by + vy * (bx - _X_LO) / -vx, _Y_LO, _Y_HI))
        if Constants.GOAL_Y_MIN - SHOT_WIDE <= at_goal <= Constants.GOAL_Y_MAX + SHOT_WIDE:
            y = at_keeper
    if y is None:
        gx, gy = float(OWN_GOAL[0]), float(OWN_GOAL[1])
        y = gy + (by - gy) * (KEEPER_X - gx) / (bx - gx) if bx > KEEPER_X else by
    return KEEPER_X, (KEEPER_Y_MIN if y < KEEPER_Y_MIN else (KEEPER_Y_MAX if y > KEEPER_Y_MAX else y))


# --- the language (v1's functions work unchanged on v2 programs where they are generic) ---


def node_type(node):
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return "n"
    if isinstance(node, str):
        return TERMINALS[node]
    return FUNCTIONS[node[0]][0]


def terminals_used(node, found=None):
    """The terminals a program reads, including those its words read implicitly."""
    found = set() if found is None else found
    if isinstance(node, str):
        found.add(node)
    elif isinstance(node, list):
        found.update(IMPLICIT.get(node[0], ()))
        for c in node[1:]:
            terminals_used(c, found)
    return found


def validate(node, want="v"):
    """Raise ValueError unless node is a well-typed program returning `want`."""
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
            validate(child, t)
    else:
        raise ValueError(f"not a program node: {node!r}")
    if node_type(node) != want:
        raise ValueError(f"{node!r} is a {TYPE_NAMES[node_type(node)]}, not a {TYPE_NAMES[want]}")


# --- terminal values for one tick ---


def python_terminals(brain, names):
    """The terminal values for one tick as plain Python lists, one entry per player."""
    T = v1.python_terminals(brain, [n for n in names if n in v1.TERMINALS])
    if not any(n not in v1.TERMINALS for n in names):
        return T
    path = None
    if "ball_path" in names or "intercept" in names:
        bx, by = (float(v) for v in np.asarray(brain.ball_pos).reshape(2))
        vx, vy = (float(v) for v in np.asarray(brain.ball_vel).reshape(2))
        path = predict_xy(bx, by, vx, vy)
        if "ball_path" in names:
            T["ball_path"] = (path[0].tolist(), path[1].tolist())
    if "intercept" in names:
        T["intercept"] = intercept_points(*path, brain.my_players_pos, brain.my_players_vel)
    if "front_mate" in names:
        T["front_mate"] = front_mates(brain.my_players_pos.tolist())
    if "bank_top" in names:
        T["bank_top"] = [BANK_TOP] * NUM
    if "bank_bottom" in names:
        T["bank_bottom"] = [BANK_BOTTOM] * NUM
    if "save_point" in names:
        bx, by = (float(v) for v in np.asarray(brain.ball_pos).reshape(2))
        vx, vy = (float(v) for v in np.asarray(brain.ball_vel).reshape(2))
        T["save_point"] = [save_point(bx, by, vx, vy)] * NUM
    return T


def interpret(node, values):
    """
    The numpy reference interpreter (v1.interpret) extended to the new functions,
    for testing the compiler. `values` is terminal_values' dict.
    """
    if isinstance(node, list) and node[0] == "strike":
        aim = np.broadcast_to(interpret(node[1], values), (NUM, 2))
        me, meet = values["me"], values["intercept"]
        return np.array([strike_point(*meet[i], *me[i], *aim[i]) for i in range(NUM)])
    if isinstance(node, list) and node[0] == "ball_in":
        n = np.broadcast_to(interpret(node[1], values), (NUM,))
        return np.array([ball_in(values["ball_path"], float(k)) for k in n])
    if isinstance(node, list):
        impl = FUNCTIONS[node[0]][2]
        return impl(*(interpret(c, values) for c in node[1:]))
    return v1.interpret(node, values)


def interpret_targets(program, brain):
    """Every player's target point by the interpreter, clipped to the field."""
    names = terminals_used(program)
    values = v1.terminal_values(brain, [n for n in names if n in v1.TERMINALS])
    for name, value in python_terminals(brain, [n for n in names if n not in v1.TERMINALS]).items():
        values[name] = value if name == "ball_path" else np.array(value, dtype=float)
    target = np.broadcast_to(np.asarray(interpret(program, values), dtype=float), (NUM, 2))
    return np.clip(target, (0.0, 0.0), (FIELD_X, FIELD_Y))


# --- compiling ---


class _CodeGen(v1._CodeGen):
    """v1's compiler, plus the new words (calls to strike_point and ball_in)."""

    def emit(self, node, pad):
        if isinstance(node, str):
            return (f"{node}_x", f"{node}_y") if TERMINALS[node] == "v" else node
        if isinstance(node, list) and node[0] in NEW_FUNCTIONS:
            arg = self.emit(node[1], pad)
            t = self.new()
            if node[0] == "strike":
                call = f"_strike(intercept_x, intercept_y, me_x, me_y, {arg[0]}, {arg[1]})"
            else:
                call = f"_ball_in(ball_path, {arg})"
            self.lines.append(pad + f"{t}x, {t}y = {call}")
            return t + "x", t + "y"
        return super().emit(node, pad)


def program_source(program):
    """The Python source of the compiled program (see v1.program_source)."""
    gen = _CodeGen()
    result = gen.emit(program, "        ")
    names = sorted(terminals_used(program))
    head = ["def _program(T):"]
    per_player = []
    for name in names:
        if name == "ball_path":
            head.append("    ball_path = T['ball_path']")
            continue
        head.append(f"    {name}_all = T[{name!r}]")
        if TERMINALS[name] == "v":
            per_player.append(f"        {name}_x, {name}_y = {name}_all[i]")
        else:
            per_player.append(f"        {name} = {name}_all[i]")
    body = [
        "    out = []",
        f"    for i in range({NUM}):",
        *per_player,
        *gen.lines,
        f"        x = {result[0]}; y = {result[1]}",
        f"        out.append((0.0 if x < 0.0 else ({FIELD_X!r} if x > {FIELD_X!r} else x),"
        f" 0.0 if y < 0.0 else ({float(FIELD_Y)!r} if y > {float(FIELD_Y)!r} else y)))",
        "    return out",
    ]
    return "\n".join(head + body) + "\n"


_CACHE = {}


def compile_program(program):
    """The program as a Python function and the terminals it reads; cached."""
    key = json.dumps(program)
    fn = _CACHE.get(key)
    if fn is None:
        scope = {"_clip": _clip, "_sqrt": math.sqrt, "_strike": strike_point, "_ball_in": ball_in}
        exec(compile(program_source(program), "<gp v2 program>", "exec"), scope)
        fn = (scope["_program"], sorted(terminals_used(program)))
        if len(_CACHE) > 4096:
            _CACHE.clear()
        _CACHE[key] = fn
    return fn


# --- printing (v1's, over v2's words) ---


def expression(node, parent_prec=-1):
    """A program node as one Python-like expression."""
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
    """The program as readable Python-like rules: where each player goes."""
    return "\n".join(["def target():"] + _statements(program, 1))


# --- the brain ---


class GPBrainV2(v1.GPBrain):
    """GPBrain (same controller) playing a program in the v2 vocabulary."""

    def __init__(self, name=None, program="ball"):
        AbstractBrain.__init__(self, name)
        validate(program)
        self.program = program
        self._fn, self._names = compile_program(program)

    def targets(self):
        return self._fn(python_terminals(self, self._names))

    def __str__(self):
        return to_text(self.program)
