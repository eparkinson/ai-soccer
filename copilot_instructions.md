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



## Lessons Learned and Common Errors

1. **Method Signatures**:
   - Ensure that the `do_move` method in custom brains like `AdaptiveChaser` aligns with the `AbstractBrain` class design. Use instance variables set by the `move` method instead of expecting arguments.

2. **Testing Defensive and Offensive Strategies**:
   - Add unit tests to validate both defensive and offensive strategies for new brains.

3. **Error Handling**:
   - Address `TypeError` issues by carefully matching method signatures and expected arguments.

4. **Tournament Validation**:
   - Always run a full tournament to validate the integration of new brains and analyze their performance.

5. **Constructor Design**:
   - Ensure that all brain classes accept an optional `name` parameter in their constructors to avoid `TypeError` when adding them to the tournament script.