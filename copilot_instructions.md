# Copilot Instructions

## Project Overview
This repository implements an AI-driven soccer simulation framework. It includes:
- Core game logic (`aisoccer/game.py`, `aisoccer/physics.py`)
- AI brains for teams and players (`aisoccer/brains/`)
- Graphics rendering (`aisoccer/graphics/`)
- Demos and tests

## Typical Workflow

### 1. Adding a New Brain
- Create a new Python file in `aisoccer/brains/` (e.g., `MyBrain.py`).
- Subclass `AbstractBrain` and implement the `do_move` method.
- Ensure the `do_move` method returns a `(5, 2)` numpy array of accelerations.
- Example:

```python
from aisoccer.abstractbrain import AbstractBrain
import numpy as np

class MyBrain(AbstractBrain):
    def do_move(self) -> np.array:
        return np.zeros((5, 2))  # Example: all players stay still
```

### 2. Testing Brains
- Add unit tests for your brain in `tests/` if needed.
- Run `tests/test_brains.py` to validate all brain implementations:

```bash
poetry run pytest tests/test_brains.py
```
- The test suite ensures:
  - The brain implements `do_move`.
  - The `move` method returns a valid `(5, 2)` numpy array with finite values.

### 3. Running Demos
- Use the demo scripts to test your brain in action:

```bash
poetry run python demo_game.py
poetry run python demo_tournament.py
```

### 4. Using Poetry
- Install dependencies and set up the environment:

```bash
poetry install
poetry shell
```
- Run commands directly using `poetry run`:

```bash
poetry run python <script>.py
```

### 5. Debugging and Iteration
- Use `pytest` to run all tests:

```bash
poetry run pytest
```
- Check for errors and iterate on your implementation.

## Example Prompts for Copilot
- "Create a new brain that moves all players toward the ball."
- "Write a test for a brain that ensures it returns a `(5, 2)` array."
- "Add a CLI flag to `demo_tournament.py` to limit worker processes."

## Notes
- Abstract classes like `BaseBrainUtils` are excluded from tests.
- Follow PEP8 and add docstrings to new classes and methods.
- Keep brain logic simple and focused on computing accelerations.
