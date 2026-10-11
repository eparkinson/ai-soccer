"""
Match reports for showcase replays: facts measured from a recorded game, and a short
description for the dashboard and the blog.

Everything comes from the recorded positions (players and ball, every tick):
territory, how many players crowd the ball, whether a team keeps someone back in goal,
team spacing, shots on goal, and when the goals came. Hand-written commentary can be
added per round in <league>/commentary.json ({"3": "text", ...}).
"""

import json

import numpy as np

from aisoccer.constants import Constants

L, H = Constants.FIELD_LENGTH - 1, Constants.FIELD_HEIGHT
MOUTH = (Constants.GOAL_Y_MIN, Constants.GOAL_Y_MAX)


def facts(replay_path):
    data = np.load(replay_path, allow_pickle=True)
    pos = data["positions"].astype(float)  # ticks x 11 x 2: blue 0-4, red 5-9, ball 10
    scores = data["scores"]
    names = [str(n) for n in data["names"]]
    ball = pos[:, 10]
    teams = {"blue": pos[:, 0:5], "red": pos[:, 5:10]}
    ticks = len(pos)
    out = {"names": names, "score": [int(v) for v in scores[-1]], "ticks": ticks, "teams": {}}

    velocity = np.diff(ball, axis=0, prepend=ball[:1])
    for side, players in teams.items():
        own_x = 0.0 if side == "blue" else L
        attack_sign = 1 if side == "blue" else -1
        dist_to_ball = np.linalg.norm(players - ball[:, None, :], axis=2)
        near_goal = (np.abs(players[:, :, 0] - own_x) < 200) & (
            np.abs(players[:, :, 1] - H / 2) < (MOUTH[1] - MOUTH[0]) / 2 + 100
        )
        pairs = np.linalg.norm(players[:, :, None, :] - players[:, None, :, :], axis=3)
        # Shots: the ball starts heading into the opponent's goal mouth at speed, in range.
        vx = velocity[:, 0] * attack_sign
        goal_x = L - Constants.GOAL_DEPTH if side == "blue" else Constants.GOAL_DEPTH
        distance = np.abs(goal_x - ball[:, 0])
        with np.errstate(divide="ignore", invalid="ignore"):
            y_at_goal = ball[:, 1] + velocity[:, 1] * distance / np.maximum(np.abs(velocity[:, 0]), 1e-9)
        on_target = (vx > 3) & (distance < 700) & (y_at_goal > MOUTH[0]) & (y_at_goal < MOUTH[1])
        shots = int(np.sum(on_target[1:] & ~on_target[:-1]))
        out["teams"][side] = {
            "name": names[0 if side == "blue" else 1],
            "territory": float(np.mean((ball[:, 0] > L / 2) if side == "blue" else (ball[:, 0] < L / 2))),
            "crowd": float(np.mean(np.sum(dist_to_ball < 150, axis=1))),
            "keeper": float(np.mean(np.any(near_goal, axis=1))),
            "spread": float(pairs.sum(axis=(1, 2)).mean() / 20),
            "shots": shots,
            "run": float(np.linalg.norm(np.diff(players, axis=0), axis=2).sum() / 5 / ticks * 1000),
        }
    goal_ticks = np.nonzero(np.any(np.diff(scores, axis=0) != 0, axis=1))[0] + 1
    out["goals"] = [
        {"minute": int(t / ticks * 90) + 1, "team": names[0] if scores[t][0] > scores[t - 1][0] else names[1]}
        for t in goal_ticks
    ]
    return out


def describe(f, previous=None):
    """A short match report. `previous` is the facts of the earlier showcases, oldest first."""
    a, b = f["teams"]["blue"], f["teams"]["red"]
    (ga, gb), (na, nb) = f["score"], (a["name"], b["name"])
    if ga == gb:
        result = f"{na} and {nb} drew {ga}–{gb}."
    else:
        winner, loser = (na, nb) if ga > gb else (nb, na)
        result = f"{winner} beat {loser} {max(ga, gb)}–{min(ga, gb)}."
    if f["goals"]:
        minutes = ", ".join(f"{g['minute']}' {g['team']}" for g in f["goals"][:8])
        result += f" Goals: {minutes}{'…' if len(f['goals']) > 8 else ''}."

    notes = []
    for t in (a, b):
        style = []
        if t["crowd"] >= 2.5:
            style.append(f"swarms the ball ({t['crowd']:.1f} players on it at once)")
        elif t["crowd"] < 0.5:
            style.append("is rarely near the ball")
        if t["spread"] < 250:
            style.append("moves as a tight pack")
        elif t["spread"] > 500:
            style.append("is scattered all over the pitch")
        if t["keeper"] >= 0.6:
            style.append(f"keeps a goalkeeper back {t['keeper']:.0%} of the time")
        elif t["keeper"] <= 0.15:
            style.append("leaves its goal empty")
        speed = t["run"] / 1000
        if speed < 1.0:
            style.append("barely moves")
        elif speed > 3.8:
            style.append("runs flat out almost all the time")
        if style:
            notes.append(f"{t['name']} " + ", ".join(style[:-1]) + (" and " if len(style) > 1 else "") + style[-1] + ".")
    territory = max((a, b), key=lambda t: t["territory"])
    notes.append(
        f"{territory['name']} kept the ball in the opposition half {territory['territory']:.0%} of the time; "
        f"shots on goal {a['shots']}–{b['shots']}."
    )

    if previous:
        firsts = []
        keeper_before = max(max(p["teams"]["blue"]["keeper"], p["teams"]["red"]["keeper"]) for p in previous)
        if max(a["keeper"], b["keeper"]) >= 0.6 and keeper_before < 0.6:
            firsts.append("the first showcase with a real goalkeeper")
        crowd_before = min(min(p["teams"]["blue"]["crowd"], p["teams"]["red"]["crowd"]) for p in previous)
        if min(a["crowd"], b["crowd"]) < crowd_before - 0.3:
            firsts.append("the least ball-swarming team so far: players are starting to hold positions")
        goals_before = max(sum(p["score"]) for p in previous)
        if sum(f["score"]) > goals_before:
            firsts.append(f"the most goals in a showcase so far ({sum(f['score'])})")
        shots_before = max(p["teams"]["blue"]["shots"] + p["teams"]["red"]["shots"] for p in previous)
        if a["shots"] + b["shots"] > shots_before:
            firsts.append(f"the most shots so far ({a['shots'] + b['shots']})")
        if firsts:
            notes.append("New: " + "; ".join(firsts) + ".")
    return result, " ".join(notes)


def commentary(league_dir, round_number):
    path = league_dir / "commentary.json"
    if path.exists():
        return json.loads(path.read_text()).get(str(round_number))
    return None
