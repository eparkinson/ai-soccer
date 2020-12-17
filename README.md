# ai-soccer
Physics based, soccer game sandbox to pit AI agents against each other

## Quick Start
```shell
git clone https://github.com/eparkinson/ai-soccer.git
cd ai-soccer
pip install -r requirements.txt
python test_game.py
```

This sets up a game between two very basic agents (or 'brains') - DefendersAndAttackers2 and BehindAndTowards

```shell
python test_nographics_game.py
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

