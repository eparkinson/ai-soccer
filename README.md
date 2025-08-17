
# ai-soccer
Physics based, soccer game sandbox to pit AI agents against each other

## Quick Start
```shell
git clone https://github.com/eparkinson/ai-soccer.git
cd ai-soccer
pip install -r requirements.txt
python demo_game.py
```

This sets up a game between two very basic agents (or 'brains') - DefendersAndAttackers2 and BehindAndTowards

```shell
python demo_nographics_game.py
```

This sets up a game without any graphics - perfect for quickly pitting two brains against each other to see the score without having to watch the game visually.

## Example game
https://www.youtube.com/watch?v=YipEvWC1kt4

This pits two simple heuristic algorithms against each other.

BehindAndTowards (Red): Employs an extremely simple (yet surprisingly simple) strategy of getting behind the ball and then pushing towards the goal.

AttackersAndDefenders (Blue): Employs a more complicated strategy with defenders hanging back waiting for the ball and more aggressive attackers attacking the ball.

## Roadmap
Current status:
 - Prototype

Next milestone: Alpha release
- Unit tests (I know!)
- At least one Reinforcement learning agent
- Improved graphics for game. At the very least the score needs to be displayed
- Publish as python package
- (possibly) Improved physics engine - although a few glitches to exploit might be more interesting
- (possibly) Improved physics engine performance. Important for faster training times.

## Contributing

See https://github.com/eparkinson/ai-soccer/blob/main/CONTRIBUTING.md

## Code of Conduct

See https://github.com/eparkinson/ai-soccer/blob/main/CODE_OF_CONDUCT.md

## AI Brains Overview

### Existing Brains

1. **BehindAndTowards** (Red):
   - Strategy: Gets behind the ball and pushes towards the goal.
   - Simplicity: Extremely simple yet surprisingly effective.

2. **DefendersAndAttackers** (Blue):
   - Strategy: Defenders hang back to protect the goal, while attackers aggressively pursue the ball.
   - Complexity: More sophisticated than BehindAndTowards.

3. **RandomWalk**:
   - Strategy: Moves randomly on the field.
   - Use Case: Baseline for testing other brains.

### New Brains

4. **LearningBrain**:
   - Strategy: Uses reinforcement learning with a Q-table to adapt its actions based on rewards.
   - Features: Can save and load its state for persistent learning.

5. **Heuristic-Based Brain**:
   - Strategy: Employs predefined rules and heuristics to make decisions.
   - Simplicity: Focuses on interpretability and ease of debugging.

6. **Genetic Algorithm Brain**:
   - Strategy: Evolves strategies over generations using genetic algorithms.
   - Features: Uses selection, crossover, and mutation to optimize performance.

## Developer Guide: Implementing Your Own Brain

1. **Create a New Brain Class**:
   - Place your brain implementation in `aisoccer/brains/`.
   - Inherit from the `BaseBrain` class in `BaseBrainUtils.py`.

2. **Implement Required Methods**:
   - `act(self, state)`: Define how your brain decides actions based on the game state.

3. **Test Your Brain**:
   - Use `demo_game.py` or `demo_nographics_game.py` to test your brain against existing ones.

4. **Register Your Brain**:
   - Add your brain to the appropriate demo scripts for testing.

## Testing Brains in a Tournament

1. **Run a Tournament**:
   ```shell
   python demo_tournament.py
   ```
   - This pits multiple brains against each other in a round-robin format.

2. **Visualize a Tournament**:
   - By default, `demo_tournament.py` includes graphical output to watch games.

## Training the LearningBrain

1. **Run Training**:
   ```shell
   python demo_learning_tournament.py
   ```
   - This script trains the `LearningBrain` by playing multiple games and updating its Q-table.

2. **Save and Load State**:
   - The `LearningBrain` automatically saves its state periodically during training.
   - Use the saved state to resume training or for evaluation.

## Design Documents

The following design documents provide detailed insights into various aspects of the AI Soccer framework:

1. [Heuristic-Based Learning Design](docs/heuristic_based_learning.md):
   - Outlines the design and implementation plan for heuristic-based decision-making.

2. [Genetic Algorithm Learning Design](docs/genetic_algorithm_learning.md):
   - Details the use of genetic algorithms to evolve strategies over generations.

3. [Performance Improvement Ideas](docs/performance_ideas.md):
   - Explores strategies to optimize performance for non-visual games.

## Brain Performance Summary

The table below summarizes the performance scores of each brain based on the last full tournament results:


## Todo: quality improvements

- **Add/expand unit tests** for brains and edge cases.
- **Add error handling** for file I/O and invalid states.
- **Add docstrings** to all public classes and methods.
- **Centralize magic numbers/constants** in a config or constants module.
- **Consider code coverage in CI** for better test quality tracking.
- **Refactor code duplication** in brains and utilities for maintainability.
- **Review type safety**: minimize use of `Any` and `type: ignore` where possible.
- **Constructor consistency**: ensure all brain classes accept an optional `name` parameter.
- **Method signatures**: double-check that all overridden methods match their base class signatures.
- **Profile and optimize performance** if scaling up tournaments or training.
