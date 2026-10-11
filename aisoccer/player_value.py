"""
Rating individual players inside a team, whatever kind of brain controls them.

A value map says, for the ball in each part of the pitch, how much more likely the next
goal is to be ours than theirs (+1: certainly ours, -1: certainly theirs), from our own
point of view (we defend x = 0). It is learned from recorded games: every tick is
labelled with who scored next.

Each time a player touches the ball, it is credited with the change in value from just
before the touch to TOUCH_AFTER ticks later: a clearance out of our box earns credit, a
push into theirs earns credit, a touch that gifts the ball to them costs credit. A
player's rating is its total credit per 9000 ticks. This is the idea behind "possession
value" ratings in football analytics.
"""

import numpy as np

from aisoccer.constants import Constants

L, H = Constants.FIELD_LENGTH - 1, Constants.FIELD_HEIGHT
CONTACT = Constants.BALL_RADIUS + Constants.PLAYER_RADIUS + 3
X_BINS, Y_BINS = 12, 4
TOUCH_AFTER = 15
PRIOR = 20  # pseudo-counts pulling sparse cells towards "no advantage"


def perspective(positions, side):
    """Positions (ticks x 11 x 2: blue 0-4, red 5-9, ball 10) as seen by `side`: we defend x = 0."""
    p = np.array(positions, dtype=float)
    if side == "red":
        p[..., 0] = L - p[..., 0]
        p = p[:, [5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 10]]
    return p


def cell(ball):
    x = np.clip((ball[..., 0] / L * X_BINS).astype(int), 0, X_BINS - 1)
    y = np.clip((ball[..., 1] / H * Y_BINS).astype(int), 0, Y_BINS - 1)
    return x, y


def next_goal_labels(scores):
    """Per tick, +1 if the next goal is blue's, -1 if red's, 0 if no goal follows (scores: ticks x 2)."""
    scores = np.asarray(scores)
    labels = np.zeros(len(scores))
    change = np.diff(scores, axis=0, prepend=scores[:1])
    upcoming = 0
    for t in range(len(scores) - 1, -1, -1):
        labels[t] = upcoming
        if change[t, 0] > 0:
            upcoming = 1
        elif change[t, 1] > 0:
            upcoming = -1
    return labels


def fit_value_map(games):
    """games: list of (positions, scores). Returns the value map (X_BINS x Y_BINS)."""
    total = np.zeros((X_BINS, Y_BINS))
    count = np.zeros((X_BINS, Y_BINS))
    for positions, scores in games:
        labels = next_goal_labels(scores)
        for side, sign in (("blue", 1), ("red", -1)):
            ball = perspective(positions, side)[:, 10]
            x, y = cell(ball)
            keep = labels != 0
            np.add.at(total, (x[keep], y[keep]), sign * labels[keep])
            np.add.at(count, (x[keep], y[keep]), 1)
    return total / (count + PRIOR)


def player_credits(positions, value_map, side, scores=None):
    """
    Total touch credit of each of `side`'s five players in one game, and the game's ticks.
    With the scores, a touch followed by a goal within TOUCH_AFTER ticks is valued at +1
    (ours) or -1 (theirs): after a goal the ball is back on the centre spot, which would
    otherwise make a scoring touch look worthless.
    """
    p = perspective(positions, side)
    ball, players = p[:, 10], p[:, :5]
    v = value_map[cell(ball)].astype(float)
    if scores is not None:
        scores = np.asarray(scores)
        change = np.diff(scores, axis=0, prepend=scores[:1])
        ours, theirs = (0, 1) if side == "blue" else (1, 0)
        outcome = np.zeros(len(v))  # +1 / -1 at the tick of a goal for / against us
        outcome[change[:, ours] > 0] = 1
        outcome[change[:, theirs] > 0] = -1
    else:
        outcome = np.zeros(len(v))
    near = np.linalg.norm(players - ball[:, None, :], axis=2) <= CONTACT  # ticks x 5
    credits = np.zeros(5)
    starts = near[1:] & ~near[:-1]  # a new contact at tick t + 1
    for t, i in zip(*np.nonzero(starts)):
        after = min(t + 1 + TOUCH_AFTER, len(v) - 1)
        goals = np.nonzero(outcome[t + 1 : after + 1])[0]  # noqa: E203
        end = outcome[t + 1 + goals[0]] if len(goals) else v[after]
        credits[i] += end - v[t]
    return credits, len(p)


def ratings(recorded, value_map):
    """
    recorded: list of (brain name, side, positions, scores). Returns {name: per-role credit
    per 9000 ticks (length-5 array)}.
    """
    totals, ticks = {}, {}
    for name, side, positions, scores in recorded:
        credits, n = player_credits(positions, value_map, side, scores)
        totals[name] = totals.get(name, 0) + credits
        ticks[name] = ticks.get(name, 0) + n
    return {name: totals[name] * 9000 / ticks[name] for name in totals}


# ---------------------------------------------------------------- football statistics

GOAL_LINE = {"blue": L - Constants.GOAL_DEPTH, "red": Constants.GOAL_DEPTH}  # the goal each side attacks
MOUTH = (Constants.GOAL_Y_MIN, Constants.GOAL_Y_MAX)
CREDIT_WINDOW = 400  # ticks: a goal is only credited to a touch this recent
STAT_NAMES = ["goals", "assists", "shots", "saves", "own_goals"]


def touches(positions):
    """New contacts between a player and the ball, in order: (tick, side, player)."""
    p = np.asarray(positions, dtype=float)
    ball = p[:, 10]
    near = np.linalg.norm(p[:, :10] - ball[:, None, :], axis=2) <= CONTACT
    starts = near[1:] & ~near[:-1]
    return [(t + 1, "blue" if k < 5 else "red", k % 5) for t, k in zip(*np.nonzero(starts))]


def on_target(positions, side):
    """Per tick: is the ball heading into the goal `side` attacks, at speed and within range?"""
    ball = np.asarray(positions, dtype=float)[:, 10]
    vel = np.diff(ball, axis=0, prepend=ball[:1])
    sign = 1 if side == "blue" else -1
    distance = np.abs(GOAL_LINE[side] - ball[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        y_at_goal = ball[:, 1] + vel[:, 1] * distance / np.maximum(np.abs(vel[:, 0]), 1e-9)
    return (vel[:, 0] * sign > 3) & (distance < 700) & (y_at_goal > MOUTH[0]) & (y_at_goal < MOUTH[1])


def match_events(positions, scores):
    """Goals, assists, shots on target, saves and own goals per player: {(side, player): counts}."""
    stats = {(side, i): dict.fromkeys(STAT_NAMES, 0) for side in ("blue", "red") for i in range(5)}
    events = touches(positions)
    ticks = [t for t, _, _ in events]
    scores = np.asarray(scores)
    change = np.diff(scores, axis=0, prepend=scores[:1])
    for goal_tick in np.nonzero(change.any(axis=1))[0]:
        scorer_side = "blue" if change[goal_tick, 0] > 0 else "red"
        k = np.searchsorted(ticks, goal_tick) - 1  # the last touch before the goal
        if k < 0 or goal_tick - ticks[k] > CREDIT_WINDOW:
            continue
        _, side, player = events[k]
        if side != scorer_side:
            stats[(side, player)]["own_goals"] += 1
            continue
        stats[(side, player)]["goals"] += 1
        if k >= 1:
            t0, side0, player0 = events[k - 1]
            if side0 == side and player0 != player and ticks[k] - t0 <= CREDIT_WINDOW:
                stats[(side, player0)]["assists"] += 1
    goal_ticks = np.nonzero(change.any(axis=1))[0]
    for side, defender in (("blue", "red"), ("red", "blue")):
        target = on_target(positions, side)
        starts = np.nonzero(target[1:] & ~target[:-1])[0] + 1
        for s in starts:  # a shot: credit the attacker who touched the ball last, if recently
            k = np.searchsorted(ticks, s + 1) - 1
            if k >= 0 and s - ticks[k] <= 30 and events[k][1] == side:
                stats[(side, events[k][2])]["shots"] += 1
        for t, who, player in events:  # a save: a defender touches a ball on target, no goal follows
            if who == defender and t >= 1 and target[t - 1]:
                later = goal_ticks[(goal_ticks >= t) & (goal_ticks <= t + 60)]
                if not any(change[g, 0 if side == "blue" else 1] > 0 for g in later):
                    stats[(who, player)]["saves"] += 1
    return stats


def event_stats(recorded):
    """
    recorded: list of (brain name, side, positions, scores). Returns {name: {stat: per-role
    counts per 9000 ticks (length-5 list)}}.
    """
    totals, ticks = {}, {}
    for name, side, positions, scores in recorded:
        game = match_events(positions, scores)
        t = totals.setdefault(name, {s: np.zeros(5) for s in STAT_NAMES})
        for i in range(5):
            for stat in STAT_NAMES:
                t[stat][i] += game[(side, i)][stat]
        ticks[name] = ticks.get(name, 0) + len(positions)
    return {
        name: {stat: [round(float(v), 2) for v in totals[name][stat] * 9000 / ticks[name]] for stat in STAT_NAMES}
        for name in totals
    }
