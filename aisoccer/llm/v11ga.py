"""
LLM-v11-GA: LLM-v11 (aisoccer/llm/v11.py) with its numbers tuned by evolution (tune_v11.py).

The code that plays is v11's own, unchanged: every function and method of v11 is re-bound
to a private copy of v11's module constants with this instance's overrides (and the
values derived from them) applied. So LLM-v11 in the same process is untouched, two
instances with different numbers do not interfere, and with no overrides the brain plays
exactly as LLM-v11.

The numbers that can be overridden (PARAMS: name -> v11's value). Two differ in form from
v11's constants: KEEPER_OFFSET is how far in front of the goal line the keeper stands
(v11: KEEPER_X = GOAL_DEPTH + PLAYER_RADIUS + 6), and POACHER_SPOT_X / POACHER_SPOT_Y are
v11's POACHER_SPOT pair.

A brain spec for aisoccer/brainspec.py load_brain:
    {"class": "aisoccer.llm.v11ga.LLMBrainV11GA", "kwargs": {"params": {"GOOD_KICK": 0.25, ...}}}
"""

import json
import types

import numpy as np

from aisoccer.constants import Constants as C
from aisoccer.llm import v11
from aisoccer.llm.v11 import LLMBrainV11

KEEPER_BASE = C.GOAL_DEPTH + C.PLAYER_RADIUS  # v11's KEEPER_X less its offset
PARAMS = {
    # scoring a kick
    "ON_TARGET_OPEN": v11.ON_TARGET_OPEN,
    "ON_TARGET_BLOCKED": v11.ON_TARGET_BLOCKED,
    "POST_MARGIN": v11.POST_MARGIN,
    "OWN_GOAL_RISK": v11.OWN_GOAL_RISK,
    "OWN_GOAL_MARGIN": v11.OWN_GOAL_MARGIN,
    "TIME_COST": v11.TIME_COST,
    "GOOD_KICK": v11.GOOD_KICK,
    # positions
    "KEEPER_OFFSET": v11.KEEPER_X - KEEPER_BASE,
    "KEEPER_CLEARS_INSIDE": v11.KEEPER_CLEARS_INSIDE,
    "BEHIND": v11.BEHIND,
    "SWITCH_MARGIN": v11.SWITCH_MARGIN,
    "LINE_GAP": v11.LINE_GAP,
    "LINE_MIN_X": v11.LINE_MIN_X,
    "LINE_MAX_X": v11.LINE_MAX_X,
    "POACHER_FROM_X": v11.POACHER_FROM_X,
    "POACHER_SPOT_X": v11.POACHER_SPOT[0],
    "POACHER_SPOT_Y": v11.POACHER_SPOT[1],
    "POACHERS": v11.POACHERS,
    # the planner and prediction (these set the brain's cost per tick)
    "PLAN_TICKS": v11.PLAN_TICKS,
    "DIRECTIONS": v11.DIRECTIONS,
    "LOOK_AHEAD": v11.LOOK_AHEAD,
}
INTEGER = {"PLAN_TICKS", "DIRECTIONS", "LOOK_AHEAD", "POACHERS"}
METHODS = ("kick_value", "best_kicks", "do_move", "go_behind", "keeper")

_NAMESPACES = {}  # params (as sorted JSON) -> the re-bound functions and constants


def check_params(params):
    """The overrides, validated and typed (integers where v11 needs them)."""
    unknown = set(params) - set(PARAMS)
    if unknown:
        raise ValueError(f"unknown LLM-v11 parameters: {sorted(unknown)}")
    out = {k: int(round(v)) if k in INTEGER else float(v) for k, v in params.items()}
    merged = {**PARAMS, **out}
    if merged["LOOK_AHEAD"] < merged["PLAN_TICKS"] + 1:
        raise ValueError("LOOK_AHEAD must be at least PLAN_TICKS + 1 (the planner follows the predicted path)")
    if not 0 <= merged["POACHERS"] <= 2:
        raise ValueError("POACHERS must be 0, 1 or 2 (two outfield players are left for the striker and line)")
    if merged["DIRECTIONS"] < 4 or merged["PLAN_TICKS"] < 1:
        raise ValueError("DIRECTIONS must be at least 4 and PLAN_TICKS at least 1")
    return out


def _constants(params):
    """v11's module globals with the overrides and everything derived from them."""
    g = dict(vars(v11))
    for k, v in params.items():
        if k not in ("KEEPER_OFFSET", "POACHER_SPOT_X", "POACHER_SPOT_Y"):
            g[k] = v
    if "KEEPER_OFFSET" in params:
        g["KEEPER_X"] = KEEPER_BASE + params["KEEPER_OFFSET"]
    if "POACHER_SPOT_X" in params or "POACHER_SPOT_Y" in params:
        g["POACHER_SPOT"] = (
            params.get("POACHER_SPOT_X", v11.POACHER_SPOT[0]),
            params.get("POACHER_SPOT_Y", v11.POACHER_SPOT[1]),
        )
    if "LOOK_AHEAD" in params:  # as v11 derives them
        g["TICKS"] = np.arange(g["LOOK_AHEAD"] + 1, dtype=float)
    if "DIRECTIONS" in params:
        angles = np.linspace(0, 2 * np.pi, g["DIRECTIONS"], endpoint=False)
        g["_ANGLES"] = angles
        g["_UNIT_MOVES"] = np.concatenate(
            [
                np.stack([np.cos(angles), np.sin(angles)], 1),
                0.5 * np.stack([np.cos(angles), np.sin(angles)], 1),
                np.zeros((1, 2)),
            ]
        )
    return g


def _rebind(f, g):
    return types.FunctionType(f.__code__, g, f.__name__, f.__defaults__, f.__closure__)


def namespace(params):
    """The re-bound module functions and methods for one set of (checked) overrides, shared by equal sets."""
    key = json.dumps(params, sort_keys=True)
    if key not in _NAMESPACES:
        if len(_NAMESPACES) >= 256:  # a tuner meets many parameter sets: keep the cache small
            _NAMESPACES.clear()
        g = _constants(params)
        for name, f in list(g.items()):  # module functions: fold, predict_ball, plan_kicks, ...
            if isinstance(f, types.FunctionType) and f.__module__ == v11.__name__:
                g[name] = _rebind(f, g)
        methods = {m: _rebind(getattr(LLMBrainV11, m), g) for m in METHODS}
        _NAMESPACES[key] = (g, methods)
    return _NAMESPACES[key]


class LLMBrainV11GA(LLMBrainV11):
    def __init__(self, name=None, params=None):
        super().__init__(name)
        self.params = check_params(params or {})
        self._constants, self._methods = namespace(self.params)

    def constant(self, name):
        """The value of one of v11's module constants for this brain."""
        return self._constants[name]

    # v11's own methods, run with this brain's constants
    def kick_value(self, ball_pos, ball_vel, t_kick):
        return self._methods["kick_value"](self, ball_pos, ball_vel, t_kick)

    def best_kicks(self, traj, tvel):
        return self._methods["best_kicks"](self, traj, tvel)

    def do_move(self):
        return self._methods["do_move"](self)

    def go_behind(self, i, ball_then, ball_now):
        return self._methods["go_behind"](self, i, ball_then, ball_now)

    def keeper(self, traj, goal, kick_acc, kick_score):
        return self._methods["keeper"](self, traj, goal, kick_acc, kick_score)

    # re-bound functions do not pickle: rebuild them from the params
    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_constants", None)
        state.pop("_methods", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._constants, self._methods = namespace(self.params)
