import pytest
import importlib
import inspect
import numpy as np
from pathlib import Path
from aisoccer.abstractbrain import AbstractBrain
from abc import ABCMeta

# Path to the brains folder
BRAINS_FOLDER = Path(__file__).parent.parent / "aisoccer" / "brains"

# Dynamically find all brain implementations
brain_classes = []
for file in BRAINS_FOLDER.glob("*.py"):
    if file.name == "__init__.py":
        continue
    module_name = f"aisoccer.brains.{file.stem}"
    module = importlib.import_module(module_name)
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, AbstractBrain)
            and obj is not AbstractBrain
            and not isinstance(obj, ABCMeta)  # Exclude abstract classes
        ):
            brain_classes.append(obj)

@pytest.mark.parametrize("BrainClass", brain_classes)
def test_brain_implementation(BrainClass):
    """Test that each brain implementation behaves correctly."""
    # Instantiate the brain
    brain = BrainClass()

    # Check that the brain implements do_move
    assert hasattr(brain, "do_move"), f"{BrainClass.__name__} does not implement do_move."
    assert callable(getattr(brain, "do_move")), f"{BrainClass.__name__}.do_move is not callable."

    # Mock inputs for the move method
    my_players_pos = np.zeros((5, 2))
    my_players_vel = np.zeros((5, 2))
    opp_players_pos = np.ones((5, 2))
    opp_players_vel = np.ones((5, 2)) * -1
    ball_pos = np.array([0.5, 0.5])
    ball_vel = np.array([0.1, 0.1])
    my_score = 0
    opp_score = 0
    game_time = 0.5

    # Call the move method
    result = brain.move(
        my_players_pos,
        my_players_vel,
        opp_players_pos,
        opp_players_vel,
        ball_pos,
        ball_vel,
        my_score,
        opp_score,
        game_time
    )

    # Validate the result
    assert isinstance(result, np.ndarray), f"{BrainClass.__name__}.move did not return a numpy array."
    assert result.shape == (5, 2), f"{BrainClass.__name__}.move returned an array with incorrect shape: {result.shape}."
    assert np.isfinite(result).all(), f"{BrainClass.__name__}.move returned non-finite values."

def test_brain_instantiation():
    """Test that all brain implementations can be instantiated."""
    for BrainClass in brain_classes:
        try:
            brain = BrainClass()
        except Exception as e:
            pytest.fail(f"Failed to instantiate {BrainClass.__name__}: {e}")
