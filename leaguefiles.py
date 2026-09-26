"""
Read the league's state from its files (runs/league), for the dashboard and the
live-match window. Nothing here changes the league.
"""

import json
import os
import re
import time
from pathlib import Path

LEAGUE_DIR = Path(os.environ.get("LEAGUE_DIR", "runs/league"))
HISTORY_DIR = Path(os.environ.get("LEAGUE_HISTORY", "aisoccer/brains/weights/history"))

SWISS_ROW = re.compile(
    r"^\s+(\d+)([* ])\s?(\S+)\s+(\d+)\s+(\d+)-(\d+)-(\d+)\s+([+-]\d+\.\d+) ±(\d+\.\d+)\s+(\d+\.\d+) ±(\d+\.\d+)"
)
H2H_ROW = re.compile(
    r"^    (\S+)\s+(\d+)\s+(\d+)-(\d+)-(\d+)\s+([+-]\d+\.\d+) ±(\d+\.\d+)\s+([\d.\s]+)$"
)
STYLE_KEYS = ["control", "own_half", "final_third", "shots", "spread", "passes"]
EVENT = re.compile(r"(NEW CHAMPION|PROMOTED|ON COURSE|PENDING|REPLACED|RETIRED|ROSTER|VERDICT|SUCCESS|Not yet significantly|exited)")


def latest_round(log_text=None):
    """The most recent completed league round: number, time, Swiss rows, head to heads, notes."""
    text = log_text if log_text is not None else (LEAGUE_DIR / "progress.log").read_text()
    starts = [m.start() for m in re.finditer(r"^LEAGUE ROUND ", text, re.M)]
    if not starts:
        return None
    block = text[starts[-1] :].splitlines()
    head = re.match(r"LEAGUE ROUND (\d+)\s+\((\d+:\d+)\)", block[0])
    swiss, h2h, notes = [], [], []
    for line in block[1:]:
        if line.startswith("LEAGUE ROUND") or line.startswith("---"):
            break
        if m := SWISS_ROW.match(line):
            swiss.append(
                {
                    "rank": int(m[1]),
                    "entrant": m[2] == "*",
                    "name": m[3],
                    "games": int(m[4]),
                    "w": int(m[5]),
                    "d": int(m[6]),
                    "l": int(m[7]),
                    "gd": float(m[8]),
                    "gd_ci": float(m[9]),
                    "pts": float(m[10]),
                    "pts_ci": float(m[11]),
                }
            )
        elif m := H2H_ROW.match(line):
            style = [float(v) for v in m[8].split()]
            h2h.append(
                {
                    "name": m[1],
                    "games": int(m[2]),
                    "w": int(m[3]),
                    "d": int(m[4]),
                    "l": int(m[5]),
                    "gd": float(m[6]),
                    "gd_ci": float(m[7]),
                    "style": dict(zip(STYLE_KEYS, style)),
                }
            )
        elif line.startswith("  ") and not line.startswith("    ") and "Swiss tournament" not in line and "Head to head" not in line:
            notes.append(line.strip())
    return {"number": int(head[1]), "time": head[2], "swiss": swiss, "h2h": h2h, "notes": notes}


def events(limit=25):
    lines = (LEAGUE_DIR / "progress.log").read_text().splitlines()
    found, current_round = [], None
    for line in lines:
        if m := re.match(r"LEAGUE ROUND (\d+)", line):
            current_round = int(m[1])
        elif EVENT.search(line):
            found.append((current_round, line.strip()))
    return found[-limit:]


def champion():
    champions = list(HISTORY_DIR.glob("*champ-*.*"))
    return max(champions, key=lambda p: int(p.stem.split("-")[-1])) if champions else None


def learner_progress():
    """name -> latest training iteration, for every learner directory."""
    progress = {}
    for path in sorted(LEAGUE_DIR.glob("*/progress.log")):
        if path.parent.name in ("coach", "ga", "tactics_ga", "neuro"):
            continue
        last = [line for line in path.read_text().splitlines() if line.startswith("it ")]
        if last:
            progress[path.parent.name] = int(last[-1].split()[1])
    return progress


def helper_progress():
    out = {}
    for name, path in (
        ("GA", "ga/best.json"),
        ("Tactics-v1-GA", "tactics_ga/best.json"),
        ("Neuro-ES", "neuro/best.json"),
    ):
        p = LEAGUE_DIR / path
        if p.exists():
            data = json.loads(p.read_text())
            out[name] = f"generation {data.get('generation')}, fitness {data.get('fitness', 0):.2f}"
    team = LEAGUE_DIR / "coach" / "team.json"
    if team.exists():
        out["Coach"] = "team: " + " / ".join(json.loads(team.read_text()))
    return out


def next_round_eta(round_minutes=30):
    """Seconds until the next round starts, estimated from the last round's log time."""
    snapshots = sorted((LEAGUE_DIR / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime)
    if not snapshots:
        return None
    return snapshots[-1].stat().st_mtime + round_minutes * 60 + 60 * 8 - time.time()


def brain_spec(name, round_number):
    """The brain file (or spec) a Swiss entry refers to, for a given round."""
    snapshots = list((LEAGUE_DIR / "snapshots").glob(f"{name}-r{round_number:03d}.*"))
    if snapshots:
        return str(snapshots[0])
    for path in HISTORY_DIR.glob(f"{name}.*"):
        return str(path)
    roster = json.loads((LEAGUE_DIR / "roster.json").read_text())
    if name in roster.get("fixed_entrants", {}):
        return roster["fixed_entrants"][name]
    return name  # a heuristic brain


def all_rounds():
    """Every completed league round in the log, oldest first (see latest_round)."""
    text = (LEAGUE_DIR / "progress.log").read_text()
    starts = [m.start() for m in re.finditer(r"^LEAGUE ROUND ", text, re.M)] + [len(text)]
    return [latest_round(text[a:b]) for a, b in zip(starts, starts[1:])]


def round_results(number=None):
    """A saved round (rounds/round-NNN.json): the latest one by default."""
    files = sorted((LEAGUE_DIR / "rounds").glob("round-*.json"))
    if number is not None:
        files = [f for f in files if f.stem == f"round-{number:03d}"]
    return json.loads(files[-1].read_text()) if files else None


def compute_usd(results=None, lineage=True):
    """
    Compute cost per brain in dollars: CPU-hours (cpu_hours) at cpu_extra.json's
    usd_per_cpu_hour, plus anything costed directly in dollars ("usd", e.g. LLMBrain's
    token bills).
    """
    extra_file = LEAGUE_DIR / "cpu_extra.json"
    extra = json.loads(extra_file.read_text()) if extra_file.exists() else {}
    price = extra.get("usd_per_cpu_hour", 0.04)
    out = {n: h * price for n, h in cpu_hours(results, lineage).items()}
    for name, usd in extra.get("usd", {}).items():
        out[name] = out.get(name, 0.0) + usd
    if lineage:  # a new version's own bill plus its predecessors' (like "inherited")
        for name, usd in extra.get("usd_inherited", {}).items():
            if name in out:
                out[name] += usd
    return out


DRAW_MARGIN = 0.25  # goals per game of 9000 ticks (scaled for longer games)


def cpu_hours(results=None, lineage=True):
    """
    CPU-hours per brain as of a round (the latest by default): the league's ledger for
    running brains, champions' compute when crowned, and <league>/cpu_extra.json:
    {"hours": {name: h}} for brains measured elsewhere (champions crowned before the
    ledger, LLMBrain versions' token costs), {"corrections": {name: delta}} for fixes to
    the ledger's estimates, and {"inherited": {name: h}}: a branch's parent's compute
    when it branched, counted in the brain's own total (lineage=True) but not when
    adding up a method's total, where the parent already counts it.
    """
    results = results or round_results() or {}
    out = {**results.get("cpu_hours", {}), **results.get("champion_cpu_hours", {})}
    extra = LEAGUE_DIR / "cpu_extra.json"
    if extra.exists():
        extra = json.loads(extra.read_text())
        out = {**extra.get("hours", {}), **out}
        adjustments = {**extra.get("corrections", {})}
        if lineage:
            for name, hours in extra.get("inherited", {}).items():
                adjustments[name] = adjustments.get(name, 0.0) + hours
        for name, delta in adjustments.items():
            if name in out:
                out[name] += delta
    return out


def football_table(results):
    """
    The round as a football league table. Each Swiss pairing is one match: a series of
    games whose score is the average score per game. A match is won by more than
    DRAW_MARGIN goals per game, otherwise drawn. GF and GA add up the match scores;
    Pts are 3 for a win and 1 for a draw.
    """
    rows = {}
    cpu = compute_usd(results)
    margin = DRAW_MARGIN * results.get("match_ticks", 9000) / 9000  # per 9000 ticks of play
    for m in results["matches"]:
        for me, them, sign in (("a", "b", 1), ("b", "a", -1)):
            name = m[me]
            row = rows.setdefault(name, {"name": name, "p": 0, "w": 0, "d": 0, "l": 0, "gf": 0.0, "ga": 0.0})
            row["p"] += 1
            gf, ga = m[f"goals_{me}"] / m["games"], m[f"goals_{them}"] / m["games"]
            row["gf"] += gf
            row["ga"] += ga
            # A draw is a series the teams finish within a quarter of a goal per game:
            # "no real difference", like a real 1-1.
            if gf - ga >= margin:
                row["w"] += 1
            elif ga - gf >= margin:
                row["l"] += 1
            else:
                row["d"] += 1
    for row in rows.values():
        row["gd"] = row["gf"] - row["ga"]
        row["pts"] = 3 * row["w"] + row["d"]
        row["entrant"] = row["name"] in results["entrants"]
        row["cpu"] = cpu.get(row["name"])
    table = sorted(rows.values(), key=lambda r: (-r["pts"], -r["gd"], -r["gf"]))
    for position, row in enumerate(table, 1):
        row["pos"] = position
    return table


def videos():
    """Rendered showcase videos, newest first: round, file, names, final score."""
    import numpy as np

    out = []
    video_dir = Path("videos") / LEAGUE_DIR.name
    for video in sorted(video_dir.glob("round-*.mp4"), reverse=True):
        if ".tmp" in video.name:
            continue
        replay = LEAGUE_DIR / "replays" / f"{video.stem}.npz"
        entry = {"round": int(video.stem.split("-")[1]), "file": video.name}
        if replay.exists():
            data = np.load(replay, allow_pickle=True)
            entry["names"] = [str(n) for n in data["names"]]
            entry["score"] = [int(v) for v in data["scores"][-1]]
        out.append(entry)
    return out


def champion_crowned():
    """(champion name, round it was crowned in) from the event log."""
    crowned = None
    for round_number, text in events(1000):
        if text.startswith("NEW CHAMPION"):
            brain, _, saved = text.removeprefix("NEW CHAMPION:").partition("->")
            crowned = (saved.split("(")[0].strip(), round_number, brain.strip())
    return crowned
