"""
How a rule change affected each brain: Swiss results in the rounds before and after it
(points, goals for and against per game, per 9000 ticks of play), from the saved rounds.

    LEAGUE_DIR=runs/league2 poetry run python rule_effect.py --before 13 14 15 --after 16
"""

import argparse

import numpy as np

import leaguefiles as lf


def per_brain(numbers):
    """name -> points per game, goals for and against per 9000 ticks, over the given rounds."""
    out = {}
    for number in numbers:
        results = lf.round_results(number)
        if results is None:
            continue
        scale = results.get("match_ticks", 9000) / 9000
        points = {s["name"]: (s["points"], s["n"]) for s in results["standings"]}
        for m in results["matches"]:
            for me, them in (("a", "b"), ("b", "a")):
                row = out.setdefault(m[me], {"gf": 0.0, "ga": 0.0, "games": 0, "pts": 0.0, "pts_games": 0})
                row["gf"] += m[f"goals_{me}"] / scale
                row["ga"] += m[f"goals_{them}"] / scale
                row["games"] += m["games"]
        for name, (p, n) in points.items():
            if name in out:
                out[name]["pts"] += p * n
                out[name]["pts_games"] += n
    return {
        n: {"pts": r["pts"] / max(r["pts_games"], 1), "gf": r["gf"] / r["games"], "ga": r["ga"] / r["games"]}
        for n, r in out.items() if r["games"]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--before", type=int, nargs="+", required=True)
    parser.add_argument("--after", type=int, nargs="+", required=True)
    parser.add_argument("--brains", nargs="*", help="only these brains (default: all in both periods)")
    args = parser.parse_args()
    before, after = per_brain(args.before), per_brain(args.after)
    names = args.brains or sorted(set(before) & set(after), key=lambda n: -after[n]["pts"])
    print(f"Rounds {args.before} -> {args.after}; goals per 9000 ticks of play")
    print("| Brain | Pts/game before | after | Conceded before | after | Scored before | after |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for n in names:
        if n in before and n in after:
            b, a = before[n], after[n]
            cells = [b["pts"], a["pts"], b["ga"], a["ga"], b["gf"], a["gf"]]
            print(f"| {n} | " + " | ".join(f"{v:.2f}" for v in cells) + " |")
    goals = [np.mean([r["gf"] for r in period.values()]) for period in (before, after)]
    print(f"\nLeague average goals per team per 9000 ticks: {goals[0]:.2f} before, {goals[1]:.2f} after")


if __name__ == "__main__":
    main()
