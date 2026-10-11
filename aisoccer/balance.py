"""
Balanced training populations: weight a learner's training opponents so that, on
average, the learner is about as good as the opponents it trains against (a small edge,
`target`), while still meeting weaker and stronger ones.

Given the learner's recent goal difference against each opponent, the weights are the
base weights (e.g. equal, or by family) tilted exponentially, w ~ base x exp(-lam x gd):
lam > 0 shifts weight to opponents the learner does worst against, lam < 0 to those it
does best against. lam is chosen so that the tilted weights' mean goal difference equals
the target, which is the smallest change to the base weights that does so. A share
`floor` of the weights then stays on the base, so every opponent is still met now and
then; the floor is for variety and is not counted towards the target (otherwise a floor
share of a few blowouts forces the rest onto the single strongest opponent). When the
target is out of reach (the learner beats, or loses to, everybody), the tilt goes as far
as it can: most weight on the hardest (or easiest) opponents.
"""

import numpy as np

LAMBDA_MAX = 20.0  # per goal per 1000 ticks: beyond this the tilt is all on one opponent


def tilt(base, gd, target=0.1, floor=0.2):
    """Weights (summing to 1): (1 - floor) x the tilt of `base` whose mean of `gd` is `target`, + floor x base."""
    base = np.asarray(base, float)
    base = base / base.sum()
    gd = np.asarray(gd, float)
    if len(gd) == 0:
        return base

    def tilted(lam):
        z = -lam * (gd - gd.mean())
        t = base * np.exp(z - z.max())
        return t / t.sum()

    def weights(lam):
        return (1.0 - floor) * tilted(lam) + floor * base

    def mean(lam):
        return float(tilted(lam) @ gd)

    lo, hi = -LAMBDA_MAX, LAMBDA_MAX  # mean(lam) falls as lam rises
    if mean(lo) <= target:
        return weights(lo)
    if mean(hi) >= target:
        return weights(hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mean(mid) > target:
            lo = mid
        else:
            hi = mid
    return weights(0.5 * (lo + hi))


class Record:
    """
    The learner's recent goal difference per 1000 ticks against each opponent, faded over
    roughly the last `memory` observations: {name: [faded sum, faded count]}. An opponent
    not seen yet counts as `prior` (the target: neither easy nor hard).
    """

    def __init__(self, data=None, memory=40, prior=0.1):
        self.data = dict(data or {})
        self.memory = memory
        self.prior = prior

    def add(self, name, gd_per_1000):
        fade = 1.0 - 1.0 / self.memory
        total, count = self.data.get(name, (0.0, 0.0))
        self.data[name] = [total * fade + float(gd_per_1000), count * fade + 1.0]

    def value(self, name):
        total, count = self.data.get(name, (0.0, 0.0))
        return (total + self.prior) / (count + 1.0)  # one pseudo-observation of the prior

    def values(self, names):
        return np.array([self.value(n) for n in names])
