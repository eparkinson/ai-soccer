"""
Keep the blog draft (blog.md) up to date with the league.

After every league round: render the new showcase replay to videos/<league>/round-NNN.mp4
and regenerate the round-by-round section of blog.md (between the AUTO markers). The
rest of blog.md is hand-written and never touched.

    LEAGUE_DIR=runs/league2 poetry run python blog_watch.py
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np

import leaguefiles as lf
from render_replay import Renderer, render

START, END = "<!-- AUTO:ROUNDS START -->", "<!-- AUTO:ROUNDS END -->"
# Rounds left out of the blog: in rounds 3-4 hand-structured brains were admitted too
# early by mistake (see progress.log). The blog tells the story of the league as intended.
HIDDEN = {3, 4}
BRIDGE = {3: "### Rounds 3–4\n\nThe learners keep training; PPO-champ-0 stays champion.\n"}


def round_section(round_, replay, video, blog_dir):
    lines = []
    if round_ is None:  # round 0: untrained brains, before any league round
        data = np.load(replay, allow_pickle=True)
        names, final = [str(n) for n in data["names"]], data["scores"][-1]
        lines += [
            "### Round 0: untrained",
            "",
            f"Two brains that have never trained: **{names[0]}** and **{names[1]}**, both random "
            "neural networks, moving on nothing but the random exploration noise every learner "
            "starts with. Any goals are accidents.",
            "",
        ]
    else:
        data = np.load(replay, allow_pickle=True) if replay.exists() else None
        lines += [f"### Round {round_['number']} ({round_['time']})", ""]
        if data is not None:
            names, final = [str(n) for n in data["names"]], data["scores"][-1]
    if replay.exists():
        import showcase_facts as sf

        previous = [
            sf.facts(p)
            for p in sorted(replay.parent.glob("round-*.npz"))
            if p.name < replay.name and int(p.stem.split("-")[1]) not in HIDDEN
        ]
        report_result, report_notes = sf.describe(sf.facts(replay), previous or None)
        note = sf.commentary(lf.LEAGUE_DIR, 0 if round_ is None else round_["number"])
        link = video.relative_to(blog_dir) if video.exists() else None
        result = "a draw" if final[0] == final[1] else f"{names[0] if final[0] > final[1] else names[1]} wins"
        lines.append(
            f"**Showcase: {names[0]} {final[0]} – {final[1]} {names[1]}** ({result}). "
            + (f"Video: [{video.name}]({link})" if link else "Video: rendering…")
        )
        lines += ["", f"{report_result} {report_notes}", ""]
        if note:
            lines += [f"*{note}*", ""]
    results = lf.round_results(round_["number"]) if round_ is not None else None
    if results:
        table = lf.football_table(results)
        lines += [
            "| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for t in table[:6]:
            lines.append(
                f"| {t['pos']} | {t['name']} | {t['p']} | {t['w']} | {t['d']} | {t['l']} | "
                f"{t['gf']:.1f} | {t['ga']:.1f} | {t['gd']:+.1f} | **{t['pts']}** | "
                + (f"{t['cpu']:.2f} |" if t.get("cpu") is not None else "– |")
            )
        lines.append("")
    elif round_ is not None and round_["swiss"]:
        top = ", ".join(f"{r['rank']}. {r['name']} ({r['pts']:.2f} pts/game)" for r in round_["swiss"][:5])
        lines += [f"Top of the Swiss: {top}.", ""]
        events = [n for n in round_["notes"] if re.match(r"(NEW CHAMPION|PROMOTED|ON COURSE|PENDING|REPLACED|RETIRED|ADMITTED)", n)]
        for event in events:
            lines.append(f"- {event}")
        if events:
            lines.append("")
    return "\n".join(lines)


COMPUTE_START, COMPUTE_END = "<!-- AUTO:COMPUTE START -->", "<!-- AUTO:COMPUTE END -->"


FAMILIES = [  # (family, name prefix); the first match wins
    ("Coach (mixes PPO roles)", "Coach"),
    ("PPO", "PPO-"),
    ("Evolution strategies", "Neuro-ES"),
    ("Genetic algorithm on networks", "GA-net"),
    ("Tactics GA (hand-structured)", "Tactics"),
    ("Genetic algorithm on a readable brain (hand-structured)", "GA"),
    ("LLMBrain (written by Claude)", "LLM-"),
]


def family(name):
    return next((f for f, prefix in FAMILIES if name.startswith(prefix)), name)


def family_table(data):
    """Each method's best brain this round beside the method's total compute so far."""
    raw = {  # branches not double counted; champions are snapshots of a counted line
        n: h for n, h in lf.compute_usd(data, lineage=False).items() if not n.startswith("PPO-champ-")
    }
    strength = {**data.get("qualifying", {}), **data.get("h2h", {})}
    swiss = {r["name"]: r for r in data["standings"]}
    groups = {}
    for name, hours in raw.items():
        groups.setdefault(family(name), {"cpu": 0.0, "members": []})
        groups[family(name)]["cpu"] += hours
        groups[family(name)]["members"].append(name)
    rows = []
    for fam, g in sorted(groups.items(), key=lambda kv: -kv[1]["cpu"]):
        ranked = [n for n in g["members"] if n in swiss]
        best = max(ranked, key=lambda n: swiss[n]["points"]) if ranked else None
        vs = strength.get(best) if best else None
        rows.append(
            f"| {fam} | {len(g['members'])} | ${g['cpu']:.2f} | {best or 'waiting'} | "
            + (f"{swiss[best]['points']:.2f}" if best else "–")
            + " | "
            + (f"{vs['gd']:+.2f} ±{vs['gd_ci']:.2f}" if vs else "–")
            + " |"
        )
    return [
        "",
        "By method: the total compute each approach has used (all its brains), and its best brain this round.",
        "",
        "| Method | Brains | Compute $ (total) | Best brain | Swiss pts/game | Goals/game vs champion |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
        *rows,
    ]


def compute_section():
    """Compute used so far by every brain, beside how strong it is, from the latest round."""
    import json

    files = sorted((lf.LEAGUE_DIR / "rounds").glob("round-*.json"))
    data = next((d for d in (json.loads(f.read_text()) for f in reversed(files)) if d.get("cpu_hours")), None)
    if data is None:
        return "*The compute ledger starts in round 7; the table appears here after that round.*"
    swiss = {r["name"]: r for r in data["standings"]}
    strength = {**data.get("qualifying", {}), **data.get("h2h", {})}
    rows = []
    for name, hours in sorted(lf.compute_usd(data).items(), key=lambda kv: -kv[1]):
        vs = strength.get(name)
        pts = swiss.get(name, {}).get("points")
        rows.append(
            f"| {name} | ${hours:.2f} | "
            + (f"{vs['gd']:+.2f} ±{vs['gd_ci']:.2f}" if vs else "(champion)" if name == data["champion"] else "–")
            + " | "
            + (f"{pts:.2f}" if pts is not None else "waiting")
            + " |"
        )
    return "\n".join(
        [
            f"After round {data['round']} (champion: {data['champion']}). Compute is measured as CPU time per brain, "
            "workers included (estimated from each brain's measured rate before round 7), and priced at "
            "cloud rates: $0.04 per CPU-hour.",
            "",
            "| Brain | Compute $ | Goals/game vs champion | Swiss pts/game |",
            "| --- | ---: | ---: | ---: |",
            *rows,
            *family_table(data),
        ]
    )


def update_blog(blog, league_label):
    replays = lf.LEAGUE_DIR / "replays"
    videos = Path("videos") / lf.LEAGUE_DIR.name
    rounds = {r["number"]: r for r in lf.all_rounds()} if (lf.LEAGUE_DIR / "progress.log").exists() else {}
    sections = []
    zero = replays / "round-000.npz"
    if zero.exists():
        sections.append(round_section(None, zero, videos / "round-000.mp4", blog.parent))
    for number in sorted(rounds):
        if number in HIDDEN:
            if number in BRIDGE:
                sections.append(BRIDGE[number])
            continue
        replay = replays / f"round-{number:03d}.npz"
        sections.append(round_section(rounds[number], replay, videos / f"round-{number:03d}.mp4", blog.parent))
    generated = f"{START}\n\n" + "\n".join(sections) + f"\n{END}"
    text = blog.read_text()
    text = text[: text.index(START)] + generated + text[text.index(END) + len(END) :]  # noqa: E203
    if COMPUTE_START in text:
        compute = f"{COMPUTE_START}\n\n{compute_section()}\n\n{COMPUTE_END}"
        head, tail = text.index(COMPUTE_START), text.index(COMPUTE_END) + len(COMPUTE_END)
        text = text[:head] + compute + text[tail:]
    blog.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--blog", type=Path, default=Path("blog.md"))
    parser.add_argument("--league-label", default="League 2")
    parser.add_argument("--every", type=float, default=60, help="seconds between checks")
    args = parser.parse_args()
    videos = Path("videos") / lf.LEAGUE_DIR.name
    videos.mkdir(parents=True, exist_ok=True)
    renderer = Renderer()
    while True:
        for replay in sorted((lf.LEAGUE_DIR / "replays").glob("round-*.npz")):
            video = videos / f"{replay.stem}.mp4"
            if not video.exists():
                tmp = video.with_suffix(".tmp.mp4")
                render(replay, tmp, renderer, league=args.league_label)
                tmp.rename(video)
                print(f"rendered {video}", flush=True)
        update_blog(args.blog, args.league_label)
        time.sleep(args.every)


if __name__ == "__main__":
    main()
