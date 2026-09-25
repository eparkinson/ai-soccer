
# ai-soccer
Physics based, soccer game sandbox to pit AI agents against each other

## Quick Start
```shell
git clone https://github.com/eparkinson/ai-soccer.git
cd ai-soccer
poetry install          # or: pip install -e .
poetry run python demo_game.py
```

This sets up a game between two simple heuristic brains, DefendersAndAttackers and SimpleBrain.

```shell
poetry run python demo_nographics_game.py
```

This plays a game without any graphics - perfect for quickly pitting two brains against each other to see the score without having to watch the game visually.

## Example game
https://www.youtube.com/watch?v=YipEvWC1kt4

This pits two simple heuristic algorithms against each other. (The video predates the goal mouths and posts.)

BehindAndTowards (Red): Employs an extremely simple (yet surprisingly effective) strategy of getting behind the ball and then pushing towards the goal.

DefendersAndAttackers (Blue): Employs a more complicated strategy with defenders hanging back waiting for the ball and more aggressive attackers attacking the ball.

## Rules of the game
- Two teams of 5 players. Blue defends the left goal, red the right.
- Each goal is a 300 pixel gap (the goal mouth) in the goal line, with a post at each end. Outside the mouth the goal line is a wall. A goal is scored when the whole ball crosses the line inside the mouth.
- Every tick, each brain returns an acceleration for each of its players. Accelerations are capped at 1 and speeds at 5 (players) and 10 (ball).
- Collisions are elastic, with mass proportional to radius squared. Posts do not move.
- After a goal, play restarts from the kick-off positions with the ball given a small random nudge.

All of these values live in `aisoccer/constants.py`.

### Reproducible games
Pass a `seed` to `Game` (or `Tournament`) and the game plays out identically every time. The seed drives the kick-off and each brain's `self.rng`, so brains that need randomness should use `self.rng` rather than the `random` module.

```python
from aisoccer.game import Game
from aisoccer.brains.BehindAndTowards import BehindAndTowards
from aisoccer.brains.RandomWalk import RandomWalk

score = Game(RandomWalk(), BehindAndTowards(), quiet_mode=True, seed=42).play()
```

## Roadmap
Current status:
 - Prototype

Next milestone: Alpha release
- At least one reinforcement learning agent (a Gym-style environment wrapper is the next step)
- Publish as a Python package
- (possibly) Further physics engine performance work. Important for faster training times.

## Contributing

See https://github.com/eparkinson/ai-soccer/blob/main/CONTRIBUTING.md

## Code of Conduct

See https://github.com/eparkinson/ai-soccer/blob/main/CODE_OF_CONDUCT.md

## AI Brains Overview

All brains live in `aisoccer/brains/`.

| Brain | Strategy |
| --- | --- |
| **BehindAndTowards** | Gets behind the ball, then pushes it towards the goal. Extremely simple, yet surprisingly effective. |
| **DefendersAndAttackers** | Defenders hang back to protect the goal and intercept; attackers pursue the ball. |
| **StrategicPlanner** | Fixed roles: a goalkeeper on the goal line, two defenders between ball and goal, a midfielder and an attacker. |
| **AdaptiveChaser** | Chases the ball while level or behind, and falls back to defend when winning. |
| **SimpleBrain** | Every player runs at the ball. A minimal, well-commented example to copy. |
| **PPOBrain** | A neural network trained with reinforcement learning (PPO). The strongest brain; see [PPOBrain](#ppobrain-reinforcement-learning) below. |
| **LearningBrain** | A placeholder learning brain: a coarse state-to-action table nudged by goal rewards. Not a real RL agent yet. |
| **RandomWalk** | Moves randomly. A baseline for testing other brains. |

Heuristic-based and genetic algorithm brains are designed but not built yet; see the design documents below.

## Developer Guide: Implementing Your Own Brain

1. **Create a new brain class** in `aisoccer/brains/`. Inherit from `AbstractBrain`, or from `BaseBrainUtils` for helpers such as `run_towards`, `is_behind_ball` and `distance_to_ball`.

2. **Implement `do_move(self)`**. It must return a 5 x 2 numpy array: one acceleration vector per player. Before it is called, the brain's attributes are filled in:
   - `my_players_pos`, `my_players_vel`, `opp_players_pos`, `opp_players_vel` (5 x 2)
   - `ball_pos`, `ball_vel` (2,)
   - `my_score`, `opp_score`, `game_time` (0 to 1)

   Everything is from your team's point of view: **you always defend the goal at x = 0 and attack the goal at x = 1799**, whichever colour you are playing.

   ```python
   import numpy as np

   from aisoccer.abstractbrain import AbstractBrain


   class MyBrain(AbstractBrain):
       def do_move(self) -> np.ndarray:
           return self.ball_pos - self.my_players_pos  # everyone chases the ball
   ```

3. **React to goals (optional)** by overriding `on_goal_scored(team, game_state)` and `on_goal_conceded(team, game_state)`. `game_state["ticks_elapsed"]` is the number of ticks since the last kick-off.

4. **Test your brain**. `tests/test_brains.py` automatically checks every brain in `aisoccer/brains/` returns a valid move. Use `demo_game.py` or `demo_nographics_game.py` to watch or score it against existing brains.

## Testing Brains in a Tournament

```shell
poetry run python demo_tournament.py
```

`Tournament(brains, rounds=0)` plays a round robin; `rounds=N` plays N Swiss rounds. Each pairing plays `legs` games (default 2, home and away) so both brains play each side. Games run in parallel, one process per CPU by default; pass `processes=1` to play them in the current process. Pass `seed` for a reproducible tournament.

## Recording Games

`Game(..., record_game=True)` records one row per team per tick, from that team's point of view, including the capped accelerations each brain chose. `game.save_game("game.csv")` writes it to CSV, which is handy as training data for imitation learning.

## PPOBrain: Reinforcement Learning

`PPOBrain` is a small neural network policy trained with Proximal Policy Optimisation (PPO), written in plain numpy (`aisoccer/ppo.py`). One network is shared by all five players: it takes one player's view of the game (its role, its own state, the ball, its teammates, and its opponents sorted by distance) and returns that player's acceleration. The brain picks a new action every 2 ticks. Trained weights are in `aisoccer/brains/weights/PPOBrain.npz`.

```shell
poetry run python demo_ppo_game.py     # watch PPOBrain (blue) play DefendersAndAttackers
```

### How it is trained

`train_ppo.py` runs the whole pipeline on all CPU cores:

1. **Behaviour cloning.** The network first imitates DefendersAndAttackers, then refines that with DAgger: the clone plays, and DefendersAndAttackers labels the positions the clone actually reaches. PPO from scratch plateaued well below the heuristic brains; starting from a clone, PPO starts level and improves from there.
2. **Critic warm-up.** A few iterations train only the value network, so the first policy updates are not driven by an untrained critic.
3. **PPO.** Each iteration plays 48 games in parallel and updates the policy. Opponents are a mix of the current policy (self-play), earlier snapshots, the heuristic brains, and earlier saved PPOBrain versions (`aisoccer/brains/weights/history/`). No single opponent dominates, so the policy has to beat a variety of strategies.

The reward is +2 per goal scored and -2 per goal conceded, plus shaping for moving the ball up the field, controlling the ball (our nearest player closer to it than theirs), and getting to the ball. The shaping is potential-based: it rewards the *change* in a score of the position, so it cannot be farmed by moving the ball back and forth, and it does not change which policy is best.

Updates stop early once the policy has moved a set distance (a KL target), and the exploration noise has a floor, which keeps training stable.

```shell
poetry run python train_ppo.py --iterations 400    # train from scratch (clone, then PPO)
poetry run python train_ppo.py --resume            # continue from runs/ppo/latest.npz
tail -f runs/ppo/progress.log                      # follow progress
```

Every 10 iterations the policy plays every fixed opponent, and each opponent's record against it is logged with 95% confidence intervals. The version with the best worst-case matchup is saved as the new `PPOBrain.npz`.

### Measuring strength: games are noisy

Goals are rare and random. Between the strongest brains there are only about 0.6 goals per 1800-tick game, goals arrive roughly as a Poisson process, and 55-65% of games are draws. So a handful of games proves nothing:

- The scoring rate does not depend on game length (goal difference per 1800 ticks is the same in 1800- and 5400-tick games), so what matters is the total number of ticks played. More games and longer games carry the same information.
- With 16 games, the 95% confidence interval on a goal difference is about +/-0.4 goals per game, several times the real gaps between the top brains.
- Measuring a 0.1 goals per game difference reliably takes about 250-300 games per pairing.

This is why the training evaluations use 48-144 games per opponent and why `ppo_tournament.py` plays 300 games per pairing. When one checkpoint is picked as the best of many evaluations, some of its apparent lead is luck, so the final check always uses fresh games.

### Result

`poetry run python ppo_tournament.py` plays a round robin of every brain, including earlier PPOBrain versions, with 300 games per pairing (3300 games per brain). Result for the committed weights, `seed=2026`:

| Brain | Points per game (95% CI) | W | L | GD |
| --- | ---: | ---: | ---: | ---: |
| **PPOBrain** | **2.07 +/- 0.04** | 1913 | 289 | 3786 |
| PPO-it120 (earlier version) | 2.00 +/- 0.04 | 1830 | 353 | 3589 |
| DefendersAndAttackers | 1.97 +/- 0.04 | 1783 | 362 | 3093 |
| PPO-it80 (earlier version) | 1.88 +/- 0.04 | 1694 | 486 | 2902 |
| PPO-clone (before PPO) | 1.72 +/- 0.04 | 1523 | 657 | 2054 |
| StrategicPlanner | 1.60 +/- 0.04 | 1286 | 597 | 1411 |
| BehindAndTowards | 1.29 +/- 0.04 | 1153 | 1363 | -256 |
| AdaptiveChaser | 1.04 +/- 0.04 | 942 | 1757 | -2348 |
| SimpleBrain | 0.91 +/- 0.04 | 785 | 1860 | -2232 |
| PPO-scratch (PPO without cloning) | 0.80 +/- 0.04 | 607 | 1875 | -2511 |
| RandomWalk | 0.51 +/- 0.03 | 338 | 2286 | -4671 |
| LearningBrain | 0.50 +/- 0.03 | 332 | 2301 | -4817 |

PPOBrain tops the table with significantly more points per game than any other brain, and is significantly stronger head to head than every brain except DefendersAndAttackers. Against DefendersAndAttackers it is level: a goal difference of +0.08 +/- 0.09 per game in PPOBrain's favour (66 wins, 181 draws, 53 losses), not yet significant.

## Design Documents

The following design documents provide detailed insights into various aspects of the AI Soccer framework:

1. [Heuristic-Based Learning Design](docs/heuristic_based_learning.md):
   - Outlines the design and implementation plan for heuristic-based decision-making.

2. [Genetic Algorithm Learning Design](docs/genetic_algorithm_learning.md):
   - Details the use of genetic algorithms to evolve strategies over generations.

3. [Performance Improvement Ideas](docs/performance_ideas.md):
   - Explores strategies to optimize performance for non-visual games.

## Brain Performance Summary

Round robin between one copy of each brain, 10 legs per pairing (60 games each), `seed=2026`:

| Brain | P | W | L | GF | GA | GD | Points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DefendersAndAttackers | 60 | 40 | 4 | 111 | 12 | 99 | 136 |
| StrategicPlanner | 60 | 36 | 4 | 72 | 15 | 57 | 128 |
| BehindAndTowards | 60 | 25 | 23 | 88 | 64 | 24 | 87 |
| SimpleBrain | 60 | 24 | 29 | 73 | 78 | -5 | 79 |
| AdaptiveChaser | 60 | 22 | 27 | 44 | 70 | -26 | 77 |
| LearningBrain | 60 | 13 | 35 | 32 | 93 | -61 | 51 |
| RandomWalk | 60 | 5 | 43 | 26 | 114 | -88 | 27 |

## Todo: quality improvements

- **Add error handling** for file I/O and invalid states.
- **Add docstrings** to all public classes and methods.
- **Consider code coverage in CI** for better test quality tracking.
- **Refactor code duplication** in brains and utilities for maintainability.
- **Review type safety**: minimize use of `Any` and `type: ignore` where possible.
