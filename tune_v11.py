"""
LLM-v11-GA: LLM-v11's numbers (aisoccer/llm/v11.py) tuned by CMA-ES against the league
field, on a small, capped compute budget. The brain is aisoccer/llm/v11ga.py: v11's own
code with the tuned numbers.

Why CMA-ES, not a GA: the search is over about a dozen continuous numbers, from a known
good start (v11's values), on a noisy fitness and with only a few hundred evaluations to
spend. CMA-ES (Hansen's standard (mu/mu_w, lambda) algorithm, written out here) makes the
most of that: it moves a search distribution's mean with a weighted recombination of the
better half of each generation (which averages out some of the noise), learns which
directions and scales matter (the covariance) and adapts its own step size, with no
crossover or mutation rates to tune.

- Every number is scaled to [0, 1] between its bounds (TUNED); the search starts at
  v11's values with step size --sigma (in those units). Samples outside the bounds are
  reflected back in; whole-number parameters are rounded.
- Each generation plays --lam samples plus the distribution's mean (which is not used in
  the update: it is the estimate of the best numbers, measured every generation) against
  every brain in --panel-dir (snapshotted into <out>/panel first, as evolve_gp.py does;
  --exclude leaves out the own entry; exact duplicates count once; LLM-v11 itself is
  added if the field has no copy of it), on the same kick-offs for every candidate
  (common random numbers): --ticks-per-opponent ticks per opponent in pairs of games
  on one seed with the colours swapped.
- Fitness: goal difference per 1000 ticks against each opponent; with --families (the
  league's roster.json, re-read every generation) weighted equally per strategy family
  and split equally within a family (evolve_gp.py's family_weights: roster families,
  state.json's champion_names, class inference for champions); without it, equally per
  opponent.
- Every --reference-every generations LLM-v11's own numbers are played too, on the same
  games, as a yardstick for the mean.

Outputs in --out: best.json (the entrant: the mean of the latest generation, from the
second generation on, when it has moved off v11's values), state.json (the CMA-ES state
and random generator, to resume after a restart: an interrupted generation is replayed
from its start), cpu.json (CPU used so far, all runs) and progress.log.

Compute cap: CPU seconds of this process and its workers (reaped and live) are added up
across restarts in cpu.json. A generation is only started if the last one's CPU still
fits within --cpu-hours-budget; after that the process stays alive but idle (sleeping,
best.json kept), so the league does not keep restarting it. Raising the budget and
restarting carries on the search.

    poetry run python tune_v11.py --workers 1 --out runs/league3/v11ga --panel-dir runs/league3/field \\
        --families runs/league3/roster.json --cpu-hours-budget 6
"""

import argparse
import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from aisoccer.brainspec import load_brain
from aisoccer.game import Game
from aisoccer.llm.v11ga import PARAMS, LLMBrainV11GA, check_params
from evolve_gp import describe_weights, family_weights, field_panel, game_plan, read_league, spec_key

BRAIN = "aisoccer.llm.v11ga.LLMBrainV11GA"
V11 = "aisoccer.llm.v11.LLMBrainV11"
NAME = "LLM-v11-GA"

# The tuned numbers and their bounds. v11's values (aisoccer/llm/v11ga.PARAMS) are the start.
TUNED = {
    # scoring a kick
    "ON_TARGET_OPEN": (0.4, 3.0),  # v11 1.3
    "ON_TARGET_BLOCKED": (0.0, 1.6),  # 0.6
    "POST_MARGIN": (0.0, 45.0),  # 15
    "OWN_GOAL_RISK": (0.5, 6.0),  # 3.0
    "OWN_GOAL_MARGIN": (0.0, 120.0),  # 40
    "TIME_COST": (0.0, 0.06),  # 0.02
    "GOOD_KICK": (-0.3, 1.0),  # 0.3
    # positions
    "KEEPER_OFFSET": (0.0, 60.0),  # 6
    "KEEPER_CLEARS_INSIDE": (80.0, 450.0),  # 200
    "BEHIND": (0.0, 80.0),  # 30
    "LINE_GAP": (60.0, 450.0),  # 220
    "POACHER_FROM_X": (500.0, 1400.0),  # 900
    # the planner's horizon: its cost per tick grows in proportion (48 is 1.2 times v11's 40)
    "PLAN_TICKS": (30, 48),  # 40
}
INTEGER = {"PLAN_TICKS", "DIRECTIONS", "LOOK_AHEAD", "POACHERS"}


def log(out, message):
    print(message, flush=True)
    with open(out / "progress.log", "a") as f:
        f.write(message + "\n")


def write_json(path, data):
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


# --- parameters <-> the search space [0, 1]^n ---


def bounds(names):
    lo = np.array([float(TUNED[k][0]) for k in names])
    hi = np.array([float(TUNED[k][1]) for k in names])
    return lo, hi


def encode(params, names):
    lo, hi = bounds(names)
    return (np.array([float(params[k]) for k in names]) - lo) / (hi - lo)


def reflect(u):
    """Points folded back into [0, 1] off its ends (a sample just outside a bound lands just inside)."""
    w = np.mod(u, 2.0)
    return np.where(w > 1.0, 2.0 - w, w)


def decode(u, names):
    """The params dict for a point of the search space (rounded, so it is exactly what is written)."""
    lo, hi = bounds(names)
    v = lo + np.clip(u, 0, 1) * (hi - lo)
    return {k: int(round(x)) if k in INTEGER else round(float(x), 4) for k, x in zip(names, v)}


# --- CMA-ES (maximising) ---


class CMAES:
    """The standard CMA-ES (Hansen, "The CMA Evolution Strategy: A Tutorial", 2016), with its default settings."""

    def __init__(self, mean, sigma, lam):
        n = len(mean)
        self.n, self.lam, self.mu = n, lam, lam // 2
        w = np.log((lam + 1) / 2) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
        self.mueff = 1.0 / (self.weights**2).sum()
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.mean = np.array(mean, dtype=float)
        self.sigma = float(sigma)
        self.C = np.eye(n)
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.updates = 0

    def ask(self, rng):
        """lam samples, reflected into [0, 1]."""
        vals, B = np.linalg.eigh(self.C)
        D = np.sqrt(np.maximum(vals, 1e-20))
        z = rng.standard_normal((self.lam, self.n))
        return reflect(self.mean + self.sigma * (z * D) @ B.T)

    def tell(self, xs, fitness):
        """Update from the samples (as evaluated) and their fitness (higher is better)."""
        xs = np.asarray(xs, dtype=float)
        order = np.argsort(-np.asarray(fitness), kind="stable")[: self.mu]
        old = self.mean
        ys = (xs[order] - old) / self.sigma
        yw = self.weights @ ys
        self.mean = old + self.sigma * yw
        vals, B = np.linalg.eigh(self.C)
        inv_sqrt = B @ np.diag(1 / np.sqrt(np.maximum(vals, 1e-20))) @ B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * inv_sqrt @ yw
        self.updates += 1
        norm = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.updates))
        hsig = float(norm / self.chin < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        rank_mu = (ys.T * self.weights) @ ys
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
            + self.cmu * rank_mu
        )
        self.C = (self.C + self.C.T) / 2
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chin - 1))
        self.sigma = float(np.clip(self.sigma, 0.005, 0.5))

    def state(self):
        return {
            "mean": self.mean.tolist(), "sigma": self.sigma, "C": self.C.tolist(),
            "pc": self.pc.tolist(), "ps": self.ps.tolist(), "updates": self.updates, "lam": self.lam,
        }

    @classmethod
    def from_state(cls, s):
        es = cls(s["mean"], s["sigma"], s["lam"])
        es.C, es.pc, es.ps = np.array(s["C"]), np.array(s["pc"]), np.array(s["ps"])
        es.updates = s["updates"]
        return es


# --- CPU used, across restarts ---


def own_cpu_seconds():
    """
    CPU of this process and all its descendants: the reaped children (os.times) and every
    live descendant from /proc with the children it has reaped (the pool's workers are
    children of multiprocessing's forkserver, not of this process).
    """
    t = os.times()
    total = t.user + t.system + t.children_user + t.children_system
    parent, cpu = {}, {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            stat = open(f"/proc/{entry.name}/stat").read()
        except OSError:
            continue
        fields = stat[stat.rindex(")") + 2:].split()
        pid = int(entry.name)
        parent[pid] = int(fields[1])
        cpu[pid] = sum(int(v) for v in fields[11:15])  # user, system, and reaped children
    me = os.getpid()
    tick = os.sysconf("SC_CLK_TCK")
    for pid in cpu:
        p, seen = parent.get(pid), 0
        while p is not None and p != me and p > 1 and seen < 64:
            p, seen = parent.get(p), seen + 1
        if p == me and pid != me:
            total += cpu[pid] / tick
    return total


class CpuLedger:
    """CPU seconds used by all runs so far, kept in cpu.json."""

    def __init__(self, path):
        self.path = path
        self.before = json.loads(path.read_text())["seconds"] if path.exists() else 0.0
        self.last_saved = 0.0
        self.peak = 0.0

    def total(self):
        self.peak = max(self.peak, own_cpu_seconds())  # a worker reaped between the two reads is not lost
        return self.before + self.peak

    def save(self, every=0.0):
        if time.time() - self.last_saved >= every:
            write_json(self.path, {"seconds": self.total(), "hours": self.total() / 3600})
            self.last_saved = time.time()


# --- fitness ---


def play(task):
    """(params or None for LLM-v11, opponent spec, candidate plays blue, seed, ticks) -> (for, against)."""
    params, opponent, blue_side, seed, ticks = task
    a = load_brain("brain:" + V11, name=NAME) if params is None else LLMBrainV11GA(NAME, params=params)
    b = load_brain(opponent)
    blue, red = (a, b) if blue_side else (b, a)
    score = Game(blue, red, game_length=ticks, quiet_mode=True, seed=seed).play()
    return (score["blue"], score["red"]) if blue_side else (score["red"], score["blue"])


def evaluate(workers, candidates, panel, seeds, ticks, ledger):
    """Goal difference per 1000 ticks, candidates x opponents, on shared kick-offs (seeds: opponents x pairs)."""
    games = 2 * seeds.shape[1]
    tasks = [
        (c, opp, g % 2 == 0, int(seeds[o, g // 2]), ticks)
        for c in candidates
        for o, opp in enumerate(panel)
        for g in range(games)
    ]
    results = workers.imap(play, tasks) if workers is not None else map(play, tasks)
    scores = []
    for r in results:
        scores.append(r)
        ledger.save(every=60)
    scores = np.array(scores, dtype=float).reshape(len(candidates), len(panel), games, 2)
    return (scores[..., 0] - scores[..., 1]).sum(axis=2) * 1000.0 / (games * ticks)


def with_v11(panel, snapshot):
    """The panel, with LLM-v11 added (as <snapshot>/LLM-v11.json) if no brain in it is exactly LLM-v11."""
    target = spec_key(json.dumps({"class": V11}), ".json")
    for p in panel:
        if p.endswith(".json") and spec_key(Path(p).read_text(), ".json") == target:
            return panel
    path = Path(snapshot) / "LLM-v11.json"
    path.write_text(json.dumps({"class": V11}))
    return panel + [str(path)]


def entrant_spec(params, generation, fitness, measure):
    return {
        "class": BRAIN,
        "kwargs": {"params": params},
        "generation": generation,
        "fitness": round(float(fitness), 4),
        "fitness_measure": measure,
    }


# --- the run ---


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--panel-dir", required=True, help="the league field (read only; snapshotted into <out>/panel)")
    parser.add_argument("--exclude", nargs="*", default=[NAME], help="brain names to leave out (the own entry)")
    parser.add_argument("--families", help="the league's roster.json: weight opponents equally per family")
    parser.add_argument("--cpu-hours-budget", type=float, default=6.0, help="stop evolving at this much CPU (all runs)")
    parser.add_argument("--lam", type=int, default=10, help="samples per generation (plus the mean)")
    parser.add_argument("--sigma", type=float, default=0.15, help="first step size, in units of each bound's range")
    parser.add_argument("--ticks-per-opponent", type=int, default=3600)
    parser.add_argument("--ticks", type=int, default=1800, help="most ticks per game")
    parser.add_argument("--reference-every", type=int, default=5, help="play LLM-v11 every N generations (0: never)")
    parser.add_argument("--params", nargs="*", default=list(TUNED), help="which of TUNED to tune")
    parser.add_argument("--generations", type=int, default=0, help="stop after this many in this run (0: no limit)")
    parser.add_argument("--exit-when-done", action="store_true", help="exit instead of idling when done")
    parser.add_argument("--seed", type=int, help="random seed of a new run (default: random)")
    return parser.parse_args(argv)


def idle(out, args, ledger, reason):
    log(out, f"{reason}; idle from now on (best.json kept)")
    ledger.save()
    if args.exit_when_done:
        return
    while True:
        time.sleep(600)


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    names = list(args.params)
    unknown = [k for k in names if k not in TUNED]
    if unknown:
        raise SystemExit(f"not tunable: {unknown} (see TUNED)")
    start = {k: PARAMS[k] for k in names}
    check_params(start)
    state_file = out / "state.json"
    ledger = CpuLedger(out / "cpu.json")
    budget = args.cpu_hours_budget * 3600
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if state["params"] != names:
            raise SystemExit(f"{state_file} tunes {state['params']}, not {names}: use another --out")
        es = CMAES.from_state(state["cma"])
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng"]
        generation = state["generation"]
        last_cpu = state.get("last_generation_cpu", 0.0)
        log(out, f"resumed after generation {generation}; {ledger.total() / 3600:.2f} CPU-hours used so far")
    else:
        seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
        rng = np.random.default_rng(seed)
        es = CMAES(encode(start, names), args.sigma, args.lam)
        generation, last_cpu = 0, 0.0
        log(
            out,
            f"{NAME} started (seed {seed}): CMA-ES over {len(names)} of LLM-v11's numbers from v11's values, "
            f"lambda {args.lam} + the mean, sigma {args.sigma}; bounds "
            + ", ".join(f"{k} {TUNED[k][0]}..{TUNED[k][1]} (v11 {PARAMS[k]})" for k in names),
        )
    games, game_ticks = game_plan(args.ticks_per_opponent, args.ticks)
    measure = "goal difference per 1000 ticks, " + (
        "equal weight per family (split within a family)" if args.families else "equal weight per opponent"
    )
    log(out, f"fitness: {measure}; {games} games of {game_ticks} ticks per opponent, same kick-offs for all")
    if args.families:
        league = read_league(args.families)
    workers = Pool(args.workers) if args.workers > 1 else None
    shown, done_here = None, 0
    try:
        while True:
            used = ledger.total()
            if used + last_cpu > budget:
                return idle(
                    out, args, ledger,
                    f"CPU budget: {used / 3600:.2f} of {args.cpu_hours_budget} hours used"
                    + (f", the next generation (~{last_cpu / 3600:.2f} h) would overrun it" if used < budget else ""),
                )
            if args.generations and done_here >= args.generations:
                return idle(out, args, ledger, f"{done_here} generations done in this run (--generations)")
            began = ledger.total()
            generation += 1
            panel = field_panel(args.panel_dir, set(args.exclude), snapshot=out / "panel")
            panel = with_v11(panel, out / "panel")
            samples = es.ask(rng)
            seeds = rng.integers(2**31, size=(len(panel), games // 2))
            candidates = [decode(es.mean, names)] + [decode(x, names) for x in samples]
            reference = args.reference_every and generation % args.reference_every == 0
            if reference:
                candidates.append(None)  # LLM-v11 itself
            per_opponent = evaluate(workers, candidates, panel, seeds, game_ticks, ledger)
            if args.families:
                try:
                    league = read_league(args.families)
                except (OSError, ValueError) as e:
                    log(out, f"  could not read {args.families} ({e}): families as last generation")
                weights, members = family_weights(panel, *league)
                text = describe_weights(panel, weights, members)
                if text != shown:
                    log(out, f"  weights by family (from generation {generation}): {text}")
                    shown = text
            else:
                weights = np.full(len(panel), 1.0 / len(panel))
            fitness = per_opponent @ weights
            mean_params, mean_fit = candidates[0], fitness[0]
            sample_fit = fitness[1:1 + len(samples)]
            es.tell(samples, sample_fit)
            if mean_params != decode(encode(start, names), names):  # not yet a copy of v11
                write_json(out / "best.json", entrant_spec(mean_params, generation, mean_fit, measure))
            last_cpu = ledger.total() - began
            ledger.save()
            write_json(
                state_file,
                {
                    "params": names, "generation": generation, "cma": es.state(),
                    "rng": rng.bit_generator.state, "last_generation_cpu": last_cpu,
                },
            )
            done_here += 1
            log(
                out,
                f"generation {generation:3d}: mean {mean_fit:+.3f}, samples best {sample_fit.max():+.3f} "
                f"median {np.median(sample_fit):+.3f}, sigma {es.sigma:.3f}; "
                f"CPU {last_cpu / 60:.1f} min, {ledger.total() / 3600:.2f} of {args.cpu_hours_budget} h",
            )
            log(out, "  mean's numbers: " + ", ".join(f"{k} {v}" for k, v in mean_params.items()))
            log(
                out,
                "  mean's goal difference per 1000 ticks: "
                + ", ".join(f"{Path(o).stem} {v:+.2f}" for o, v in zip(panel, per_opponent[0])),
            )
            if reference:
                log(out, f"  LLM-v11's own numbers on the same games: {fitness[-1]:+.3f} "
                         f"(mean minus v11: {mean_fit - fitness[-1]:+.3f})")
    finally:
        if workers is not None:
            workers.close()
            workers.join()
        ledger.save()


if __name__ == "__main__":
    run(parse_args())
