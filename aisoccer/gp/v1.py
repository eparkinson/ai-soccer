"""
GPBrain v1: a brain whose whole play is one short program, evolved by genetic
programming (evolve_gp.py) from random programs.

The program answers one question for each player: "where should I go?". It returns a
target point, and every player runs at full speed towards its target and stops there.
The same program runs for all five players; `role` (the player's index, 0 to 4) and
`home` (its kick-off position) let different players do different things.

The vocabulary is deliberately low-level geometry, the same raw material LLMBrain's
code works from (positions, velocities, distances, comparisons), with no football
skills built in: no passing, shooting, marking or formations. Anything like that has
to be discovered.

Programs are nested lists, so they can be stored as JSON:

    ["if_v", "nearest_to_ball", ["behind", "ball", "opp_goal", 30], "home"]

prints as

    def target():
        if nearest_to_ball:
            return behind(ball, opp_goal, 30)
        return home

A program is compiled once to straight-line Python on plain floats (numpy is slow on
arrays of five), so a GPBrain adds only about 5% to the game's own physics time.
The numpy interpreter (`interpret`) is the slow reference the compiler is tested
against.

Vocabulary (V = point or vector, N = number, B = yes/no; all per player):

  Points and vectors
    me            my position
    ball          the ball's position
    ball_vel      the ball's velocity (pixels per tick, at most 10)
    own_goal      the centre of the goal I defend: (20, 400)
    opp_goal      the centre of the goal I attack: (1779, 400)
    home          my kick-off position
    mate          the nearest teammate's position
    opp           the nearest opponent's position
    opp_at_ball   the position of the opponent nearest the ball
  Numbers
    role          my index in the team, 0 to 4 (0, 1 start deepest; 4 up front)
    constants     e.g. 30, 0.5, 900
  Yes/no
    nearest_to_ball    I am my team's player nearest the ball
    ball_in_own_half   the ball is in my half

  Functions
    a + b, a - b          add or subtract points/vectors, or numbers
    v * n                 scale a vector
    unit(v)               v scaled to length 1 (or (0, 0))
    behind(a, b, n)       the point n beyond a on the line from b through a:
                          behind(ball, opp_goal, 30) is 30 behind the ball, facing goal
    point(x, y)           the point (x, y)
    a * b, min(a, b), max(a, b)          numbers
    x(v), y(v), dist(a, b)               numbers from points
    a < b, a and b, a or b, not a        yes/no
    (a if c else b)                      choose, for points or numbers

Every number and vector result is kept finite (multiplications are clipped to
+-10000), and the target is clipped to the field, so a program can never produce
NaN or infinite accelerations.
"""

import json
import math

import numpy as np

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.constants import Constants

NUM = Constants.NUM_PLAYERS
FIELD_X = Constants.FIELD_LENGTH - 1
FIELD_Y = Constants.FIELD_HEIGHT
MID_Y = Constants.FIELD_HEIGHT / 2
BIG = 1e4  # numbers and vector components from * are clipped to +-BIG
ROLE = np.arange(NUM, dtype=float)
HOME = np.array(Constants.STARTING_POSITIONS[0], dtype=float)
OWN_GOAL = np.array([Constants.GOAL_DEPTH, MID_Y])
OPP_GOAL = np.array([FIELD_X - Constants.GOAL_DEPTH, MID_Y])

# --- primitive implementations on numpy arrays (the reference interpreter's) ---


def _col(n):
    """A per-player number (or a plain number) as a column, to combine with vectors."""
    return np.reshape(n, (-1, 1))


def p_scale(v, n):
    return np.clip(v * _col(n), -BIG, BIG)


def p_unit(v):
    norm = np.sqrt(np.sum(np.square(v), axis=-1, keepdims=True))
    return np.where(norm > 1e-9, v / np.maximum(norm, 1e-9), 0.0)


def p_behind(a, b, n):
    return a + p_scale(p_unit(a - b), n)


def p_point(x, y):
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return np.stack([x, y], axis=-1)


def p_if_v(c, a, b):
    return np.where(_col(c), a, b)


def p_mul(a, b):
    return np.clip(np.multiply(a, b), -BIG, BIG)


def p_x(v):
    return v[..., 0]


def p_y(v):
    return v[..., 1]


def p_dist(a, b):
    d = np.subtract(a, b)
    return np.sqrt(np.sum(d * d, axis=-1))


# --- the language ---
# name: (return type, argument types, implementation, print form)
# Print forms: an infix operator with its precedence, or a function name.

FUNCTIONS = {
    "add_v": ("v", ("v", "v"), np.add, ("+", 4)),
    "sub_v": ("v", ("v", "v"), np.subtract, ("-", 4)),
    "scale": ("v", ("v", "n"), p_scale, ("*", 5)),
    "unit": ("v", ("v",), p_unit, "unit"),
    "behind": ("v", ("v", "v", "n"), p_behind, "behind"),
    "point": ("v", ("n", "n"), p_point, "point"),
    "if_v": ("v", ("b", "v", "v"), p_if_v, "if"),
    "add": ("n", ("n", "n"), np.add, ("+", 4)),
    "sub": ("n", ("n", "n"), np.subtract, ("-", 4)),
    "mul": ("n", ("n", "n"), p_mul, ("*", 5)),
    "min": ("n", ("n", "n"), np.minimum, "min"),
    "max": ("n", ("n", "n"), np.maximum, "max"),
    "x": ("n", ("v",), p_x, "x"),
    "y": ("n", ("v",), p_y, "y"),
    "dist": ("n", ("v", "v"), p_dist, "dist"),
    "if_n": ("n", ("b", "n", "n"), np.where, "if"),
    "lt": ("b", ("n", "n"), np.less, ("<", 3)),
    "and": ("b", ("b", "b"), np.logical_and, ("and", 1)),
    "or": ("b", ("b", "b"), np.logical_or, ("or", 0)),
    "not": ("b", ("b",), np.logical_not, "not"),
}

TERMINALS = {
    "me": "v",
    "ball": "v",
    "ball_vel": "v",
    "own_goal": "v",
    "opp_goal": "v",
    "home": "v",
    "mate": "v",
    "opp": "v",
    "opp_at_ball": "v",
    "role": "n",
    "nearest_to_ball": "b",
    "ball_in_own_half": "b",
}

TYPE_NAMES = {"v": "point", "n": "number", "b": "yes/no"}


def node_type(node):
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return "n"
    if isinstance(node, str):
        return TERMINALS[node]
    return FUNCTIONS[node[0]][0]


def size(node):
    if isinstance(node, list):
        return 1 + sum(size(c) for c in node[1:])
    return 1


def depth(node):
    if isinstance(node, list):
        return 1 + max(depth(c) for c in node[1:])
    return 1


def terminals_used(node, found=None):
    found = set() if found is None else found
    if isinstance(node, str):
        found.add(node)
    elif isinstance(node, list):
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


# --- terminal values for one tick, all players at once (for the interpreter) ---


def terminal_values(brain, names):
    me = np.asarray(brain.my_players_pos, dtype=float)
    ball = np.asarray(brain.ball_pos, dtype=float).reshape(2)
    values = {}
    ball_d = None
    for name in names:
        if name == "me":
            values[name] = me
        elif name == "ball":
            values[name] = np.broadcast_to(ball, (NUM, 2))
        elif name == "ball_vel":
            values[name] = np.broadcast_to(np.asarray(brain.ball_vel, dtype=float).reshape(2), (NUM, 2))
        elif name == "own_goal":
            values[name] = np.broadcast_to(OWN_GOAL, (NUM, 2))
        elif name == "opp_goal":
            values[name] = np.broadcast_to(OPP_GOAL, (NUM, 2))
        elif name == "home":
            values[name] = HOME
        elif name == "mate":
            d = me[:, None, :] - me[None, :, :]
            d2 = np.sum(d * d, axis=-1)
            np.fill_diagonal(d2, np.inf)
            values[name] = me[np.argmin(d2, axis=1)]
        elif name == "opp":
            opp = np.asarray(brain.opp_players_pos, dtype=float)
            d = me[:, None, :] - opp[None, :, :]
            values[name] = opp[np.argmin(np.sum(d * d, axis=-1), axis=1)]
        elif name == "opp_at_ball":
            opp = np.asarray(brain.opp_players_pos, dtype=float)
            d = opp - ball
            values[name] = np.broadcast_to(opp[np.argmin(np.sum(d * d, axis=1))], (NUM, 2))
        elif name == "role":
            values[name] = ROLE
        elif name == "nearest_to_ball":
            if ball_d is None:
                d = me - ball
                ball_d = np.sum(d * d, axis=1)
            values[name] = ROLE == np.argmin(ball_d)
        elif name == "ball_in_own_half":
            values[name] = np.full(NUM, ball[0] < FIELD_X / 2)
    return values


# --- interpreting and compiling ---


def interpret(node, values):
    """
    Evaluate a program node on numpy arrays (all players at once). Slow: it is the
    reference that the compiled programs are tested against.
    """
    if isinstance(node, (int, float)):
        return float(node)
    if isinstance(node, str):
        return values[node]
    impl = FUNCTIONS[node[0]][2]
    return impl(*(interpret(c, values) for c in node[1:]))


def interpret_targets(program, brain):
    """Every player's target point by the interpreter, clipped to the field."""
    values = terminal_values(brain, terminals_used(program))
    target = np.broadcast_to(np.asarray(interpret(program, values), dtype=float), (NUM, 2))
    return np.clip(target, (0.0, 0.0), (FIELD_X, FIELD_Y))


def _clip(x):
    return -BIG if x < -BIG else (BIG if x > BIG else x)


class _CodeGen:
    """
    Compiles a program to straight-line Python on floats, one player at a time.
    Numpy is slow on arrays of five, and plain Python lets 'if' run only the branch
    it needs. A point becomes two variables (x and y); numbers and yes/no one each.
    """

    def __init__(self):
        self.lines = []
        self.count = 0

    def new(self):
        self.count += 1
        return f"t{self.count}"

    def emit(self, node, pad):
        """Emit code computing node; return its value: (x, y) names, or one name/literal."""
        add = lambda line: self.lines.append(pad + line)  # noqa: E731
        if isinstance(node, (int, float)):
            return repr(float(node))
        if isinstance(node, str):
            return (f"{node}_x", f"{node}_y") if TERMINALS[node] == "v" else node
        name, args = node[0], node[1:]
        if name in ("if_v", "if_n"):
            c = self.emit(args[0], pad)
            t = self.new()
            out = (t + "x", t + "y") if name == "if_v" else (t,)
            add(f"if {c}:")
            for k, branch in enumerate((args[1], args[2])):
                if k == 1:
                    add("else:")
                value = self.emit(branch, pad + "    ")
                value = value if isinstance(value, tuple) else (value,)
                self.lines.append(pad + "    " + f"{', '.join(out)} = {', '.join(value)}")
            return out if name == "if_v" else t
        if name == "and" or name == "or":  # evaluate both sides, like the interpreter
            a, b = self.emit(args[0], pad), self.emit(args[1], pad)
            t = self.new()
            add(f"{t} = bool({a}) {name} bool({b})")
            return t
        vals = [self.emit(a, pad) for a in args]
        t = self.new()
        tx, ty = t + "x", t + "y"
        if name in ("add_v", "sub_v"):
            op = "+" if name == "add_v" else "-"
            (ax, ay), (bx, by) = vals
            add(f"{tx} = {ax} {op} {bx}; {ty} = {ay} {op} {by}")
            return tx, ty
        if name == "scale":
            (ax, ay), n = vals
            add(f"{tx} = _clip({ax} * {n}); {ty} = _clip({ay} * {n})")
            return tx, ty
        if name in ("unit", "behind"):
            (ax, ay) = vals[0]
            if name == "behind":
                (bx, by) = vals[1]
                add(f"{t}dx = {ax} - {bx}; {t}dy = {ay} - {by}")
                dx, dy = t + "dx", t + "dy"
            else:
                dx, dy = ax, ay
            add(f"{t}n = _sqrt({dx} * {dx} + {dy} * {dy})")
            add(f"if {t}n > 1e-9: {tx} = {dx} / {t}n; {ty} = {dy} / {t}n")
            add(f"else: {tx} = {ty} = 0.0")
            if name == "behind":
                n = vals[2]
                add(f"{tx} = {ax} + _clip({tx} * {n}); {ty} = {ay} + _clip({ty} * {n})")
            return tx, ty
        if name == "point":
            add(f"{tx} = {vals[0]}; {ty} = {vals[1]}")
            return tx, ty
        if name in ("add", "sub"):
            add(f"{t} = {vals[0]} {'+' if name == 'add' else '-'} {vals[1]}")
        elif name == "mul":
            add(f"{t} = _clip({vals[0]} * {vals[1]})")
        elif name in ("min", "max"):
            add(f"{t} = {name}({vals[0]}, {vals[1]})")
        elif name in ("x", "y"):
            return vals[0][0 if name == "x" else 1]
        elif name == "dist":
            (ax, ay), (bx, by) = vals
            add(f"{t}dx = {ax} - {bx}; {t}dy = {ay} - {by}")
            add(f"{t} = _sqrt({t}dx * {t}dx + {t}dy * {t}dy)")
        elif name == "lt":
            add(f"{t} = {vals[0]} < {vals[1]}")
        elif name == "not":
            add(f"{t} = not {vals[0]}")
        else:
            raise ValueError(name)
        return t


def program_source(program):
    """
    The Python source of the compiled program: a function of the tick's terminal
    values (see python_terminals) returning every player's target point, clipped to
    the field.
    """
    gen = _CodeGen()
    result = gen.emit(program, "        ")
    names = sorted(terminals_used(program))
    head = ["def _program(T):"]
    per_player = []
    for name in names:
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


def python_terminals(brain, names):
    """The terminal values for one tick as plain Python lists, one entry per player."""
    me = brain.my_players_pos.tolist()
    bx, by = (float(v) for v in np.asarray(brain.ball_pos).reshape(2))
    T = {}
    for name in names:
        if name == "me":
            T[name] = me
        elif name == "ball":
            T[name] = [(bx, by)] * NUM
        elif name == "ball_vel":
            T[name] = [tuple(float(v) for v in np.asarray(brain.ball_vel).reshape(2))] * NUM
        elif name == "own_goal":
            T[name] = [tuple(OWN_GOAL.tolist())] * NUM
        elif name == "opp_goal":
            T[name] = [tuple(OPP_GOAL.tolist())] * NUM
        elif name == "home":
            T[name] = HOME_LIST
        elif name == "mate":
            T[name] = [me[_nearest(me, x, y, skip=i)] for i, (x, y) in enumerate(me)]
        elif name == "opp":
            opp = brain.opp_players_pos.tolist()
            T[name] = [opp[_nearest(opp, x, y)] for x, y in me]
        elif name == "opp_at_ball":
            opp = brain.opp_players_pos.tolist()
            T[name] = [opp[_nearest(opp, bx, by)]] * NUM
        elif name == "role":
            T[name] = ROLE_LIST
        elif name == "nearest_to_ball":
            k = _nearest(me, bx, by)
            T[name] = [i == k for i in range(NUM)]
        elif name == "ball_in_own_half":
            T[name] = [bx < FIELD_X / 2] * NUM
    return T


def _nearest(points, x, y, skip=-1):
    """Index of the point nearest (x, y) (the first if tied), skipping index `skip`."""
    best, best_d = -1, math.inf
    for j, (px, py) in enumerate(points):
        if j == skip:
            continue
        dx, dy = px - x, py - y
        d = dx * dx + dy * dy
        if d < best_d:
            best, best_d = j, d
    return best


HOME_LIST = [tuple(p) for p in HOME.tolist()]
ROLE_LIST = ROLE.tolist()
_CACHE = {}


def compile_program(program):
    """The program as a Python function (see program_source) and the terminals it uses; cached."""
    key = json.dumps(program)
    fn = _CACHE.get(key)
    if fn is None:
        scope = {"_clip": _clip, "_sqrt": math.sqrt}
        exec(compile(program_source(program), "<gp program>", "exec"), scope)
        fn = (scope["_program"], sorted(terminals_used(program)))
        if len(_CACHE) > 4096:
            _CACHE.clear()
        _CACHE[key] = fn
    return fn


# --- printing ---


def _fmt_const(c):
    c = float(c)
    if abs(c) >= 10 or c == int(c):
        return str(int(round(c)))
    return f"{c:.2f}".rstrip("0")


def expression(node, parent_prec=-1):
    """A program node as one Python-like expression."""
    if isinstance(node, (int, float)):
        text = _fmt_const(node)
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
        # left-associative: the right operand needs brackets at equal precedence
        text = f"{expression(args[0], prec)} {op} {expression(args[1], prec + 1)}"
        return f"({text})" if prec < parent_prec else text
    return f"{form}({', '.join(expression(a) for a in args)})"


def _statements(node, indent):
    pad = "    " * indent
    if isinstance(node, list) and node[0] == "if_v":
        lines = [f"{pad}if {expression(node[1])}:"]
        lines += _statements(node[2], indent + 1)
        rest = node[3]
        lines += _statements(rest, indent)  # "return" above makes the else implicit
        return lines
    return [f"{pad}return {expression(node)}"]


def to_text(program):
    """The program as readable Python-like rules: where each player goes."""
    return "\n".join(["def target():"] + _statements(program, 1))


# --- the brain ---


class GPBrain(AbstractBrain):
    """
    Plays one evolved program (see the module docstring). Each tick every player's
    target point comes from the program, and the player accelerates to reach it at
    full speed and stop there: desired velocity = (target - me), capped at the top
    speed; acceleration = desired velocity - current velocity (the game caps it at 1).
    """

    def __init__(self, name=None, program="ball"):
        super().__init__(name)
        validate(program)
        self.program = program
        self._fn, self._names = compile_program(program)

    def targets(self):
        """Every player's target point this tick (a list of five (x, y))."""
        return self._fn(python_terminals(self, self._names))

    def do_move(self):
        speed = Constants.MAX_PLAYER_VELOCITY
        vel = self.my_players_vel.tolist()
        moves = []
        for (tx, ty), (x, y), (vx, vy) in zip(self.targets(), self.my_players_pos.tolist(), vel):
            dx, dy = tx - x, ty - y
            norm = math.sqrt(dx * dx + dy * dy)
            if norm > speed:
                dx, dy = dx * speed / norm, dy * speed / norm
            moves.append((dx - vx, dy - vy))
        return np.array(moves)

    def __str__(self):
        return to_text(self.program)
