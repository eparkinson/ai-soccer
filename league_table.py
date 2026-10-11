"""
A round's full league table as Markdown: Swiss standings, the head to head with the
champion, compute cost, and the round's events.

    LEAGUE_DIR=runs/league2 poetry run python league_table.py [round]
"""

import re
import sys

import leaguefiles as lf

EVENTS = re.compile(
    r"(PROMOTED|NEW CHAMPION|TITLE RACE|ON COURSE|ADMITTED|WAITING|SPARRING|REPLACED|RETIRED|SETTINGS|SWISS)"
)


def main():
    number = int(sys.argv[1]) if len(sys.argv) > 1 else None
    results = lf.round_results(number)
    if results is None:
        print("No saved rounds yet.")
        return
    usd = lf.compute_usd(results)
    ticks = results.get("match_ticks", 9000)
    h2h = results.get("h2h", {})
    race = results.get("title_race", {})
    print(f"Round {results['round']} ({results['time']}), champion {results['champion']}; "
          f"games of {ticks} ticks, goals per game")
    print()
    measure = "Title rating (3 rounds)" if race else "vs champion (GD)"
    print(f"| # | Brain | W-D-L | Pts/game | GD/game | {measure} | Compute |")
    print("| ---: | --- | --- | ---: | ---: | ---: | ---: |")
    for rank, s in enumerate(results["standings"], 1):
        name, v = s["name"], (race or h2h).get(s["name"])
        vs = f"{v['gd']:+.1f} ±{v['gd_ci']:.1f}" if v else ("—" if name == results["champion"] and not race else "")
        cost = f"${usd[name]:.2f}" if name in usd else ""
        print(
            f"| {rank} | {name} | {int(s['wins'])}-{int(s['draws'])}-{int(s['losses'])} | {s['points']:.2f} | "
            f"{s['gd']:+.2f} | {vs} | {cost} |"
        )
    waiting = results.get("qualifying", {})
    if waiting:
        print()
        print("Waiting: " + ", ".join(
            f"{n} {w['gd']:+.1f} ±{w['gd_ci']:.1f}" + (f" (${usd[n]:.2f})" if n in usd else "")
            for n, w in waiting.items()
        ))
    events = [n for n in results.get("notes", []) if EVENTS.match(n) and not n.startswith("WAITING")]
    if events:
        print()
        for event in events:
            print(f"- {event}")


if __name__ == "__main__":
    main()
