"""
One way to name any brain, so tournaments, the league and training can use them all.

A brain spec is a string:

- a heuristic brain's class name, e.g. "DefendersAndAttackers",
- "brain:<module>.<Class>", e.g. "brain:aisoccer.tactics.v1.TacticsV1",
  created with no arguments,
- a path to a .npz file: PPOBrain weights,
- "explore:<path to .npz>": the same PPOBrain sampling its exploration noise,
- a path to a .json file: either a GeneticBrain chromosome ({"chromosome": [...]})
  or any brain class with arguments ({"class": "<module>.<Class>", "kwargs": {...}}).
"""

import importlib
import json
from pathlib import Path

from aisoccer.brains.AdaptiveChaser import AdaptiveChaser
from aisoccer.brains.BehindAndTowards import BehindAndTowards
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.brains.GeneticBrain import GeneticBrain
from aisoccer.brains.LearningBrain import LearningBrain
from aisoccer.brains.PPOBrain import PPOBrain
from aisoccer.brains.RandomWalk import RandomWalk
from aisoccer.brains.SimpleBrain import SimpleBrain
from aisoccer.brains.StrategicPlanner import StrategicPlanner

HEURISTICS = {
    "DefendersAndAttackers": DefendersAndAttackers,
    "BehindAndTowards": BehindAndTowards,
    "StrategicPlanner": StrategicPlanner,
    "AdaptiveChaser": AdaptiveChaser,
    "SimpleBrain": SimpleBrain,
    "LearningBrain": LearningBrain,
    "RandomWalk": RandomWalk,
}


def load_class(path):
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def load_brain(spec, name=None):
    """Create the brain a spec describes (see the module docstring)."""
    spec = str(spec)
    if spec in HEURISTICS:
        return HEURISTICS[spec](name) if name else HEURISTICS[spec]()
    if spec.startswith("explore:"):
        # An untrained learner as it really plays: sampling its exploration noise.
        path = Path(spec.removeprefix("explore:"))
        return PPOBrain(name or path.stem, weights=PPOBrain.load_weights(path), deterministic=False)
    if spec.startswith("brain:"):
        return load_class(spec.removeprefix("brain:"))(name=name)
    path = Path(spec)
    name = name or path.stem
    if path.suffix == ".npz":
        return PPOBrain(name, weights=PPOBrain.load_weights(path))
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if "chromosome" in data:
            return GeneticBrain(name, data["chromosome"])
        return load_class(data["class"])(name=name, **data.get("kwargs", {}))
    raise ValueError(f"not a brain spec: {spec}")


def panel_from_dir(directory, size, rng):
    """Up to `size` random brain files (.npz or .json) from a directory, as specs."""
    files = sorted(
        str(p) for p in Path(directory).glob("*") if p.suffix in (".npz", ".json") and ".tmp" not in p.name
    )
    if len(files) <= size:
        return files
    return [files[i] for i in rng.choice(len(files), size=size, replace=False)]


def spec_label(spec):
    spec = str(spec)
    if spec in HEURISTICS:
        return spec
    if spec.startswith("brain:"):
        return spec.rsplit(".", 1)[-1]
    return Path(spec).stem
