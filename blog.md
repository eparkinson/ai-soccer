# From random flailing to real football: a league of learning soccer brains

*Draft. Videos are local files for now (`videos/league2/`); they will be uploaded to YouTube before publishing.*

## The idea

[ai-soccer](https://github.com/eparkinson/ai-soccer) is a small, physics-based soccer sandbox: two teams of five circular players on a flat pitch, a ball with no friction, elastic collisions, and goal mouths in the walls at each end. Every tick, a *brain* decides how hard each of its five players accelerates, and in which direction. That is all a brain controls.

This post follows a league of brains that all start knowing nothing. Nobody is shown how to play, and none of the hand-written heuristic brains in the project take part. The brains only ever play each other, and every so often the league holds a tournament to see who is improving. Every round we record a game between the two best brains of that round, so you can watch the league learn, from the first aimless games to (hopefully) something that looks like football.

## The competitors

Every brain starts random, and each one learns in a different way:

- **Six reinforcement learners** (PPO, a standard policy-gradient method). Each is a small neural network that sees one player's view of the game and outputs that player's acceleration; all five players share it. They learn from rewards: a goal is worth +2, a goal conceded -2, and most also get small rewards for moving the ball forward and getting to it first. Each makes a different bet:
  - *steady*: standard settings,
  - *explorer*: more random exploration and faster learning,
  - *goals*: rewarded for goals only, nothing else,
  - *stats*: extra rewards for shots and for spreading out, the two things that best predicted winning in an earlier analysis,
  - *big*: a network twice as wide,
  - *long*: values goals further in the future.
- **Neuro-ES**: the same kind of neural network, but trained by *evolution strategies* instead of rewards: it tries lots of small random changes to all its weights and keeps moving in the direction of the changes that won more games.
- **GA-net**: a *genetic algorithm* on the same network's weights: a population of 24 random networks; each generation the best quarter survive and the rest are replaced by mutated copies of them.
- **Coach**: builds a mixed team, taking each of the five player roles from whichever learner plays that role best.

Two other kinds of brain are waiting in the wings. Both learn, but on top of *hand-written* structure, so even a random one plays purposeful football, which would spoil the "from zero" start:

- **Tactics-v2-GA**: a genetic algorithm tuning a *hierarchical* brain. The team structure is built in: players take roles (win the ball, support, mark, hold a formation position), and a "team strategy" of 15 numbers shapes it. Its players' skills are not built in: four skill genes start at zero and have to be discovered. They are looking ahead to where the ball is going, getting behind the ball to aim it, the full kicking repertoire (shots at the open corner, bank shots off the walls, passes), and keeping a goalkeeper. At the start its players chase where the ball *is* and knock it anywhere, about as well as the naive LLMBrain. It evolves on half the compute of the other learners.
- **GA**: a genetic algorithm evolving a simple readable brain whose behaviour is set by 40 numbers ("genes") such as where each player waits and how far it chases the ball.

They evolve in the background against the league, and each round they play the current champion. They only join the league once the learners have caught up with them. Each round they play 84 games against the champion. Once they are at most 2 goals per game better, they become *sparring partners*: the learners train against them, but they don't play in the league. They join the league once they are level with the champion in two rounds running: no more than half a goal per game better on average, and not significantly better (the 95% confidence interval includes zero). The same rule applies to a third waiting brain, **LLM-v1**, a deliberately naive hand-written brain ("everyone run at the ball") that Claude improves with one small, explainable change at a time once it joins, but only in rounds where it isn't leading. Each new version costs compute (Claude's tokens), so it's only spent when it's needed, and each time the aim is the smallest change that gets it back on top, or near it.

## How the league works

Everything trains at the same time, on one 24-core machine. Every 20 minutes the league pauses training and plays a round:

1. A **Swiss tournament** among all the competitors and past champions: four rounds in which brains with similar scores meet each other. Each pairing plays 4 games of 36,000 ticks, about 6 minutes of play each. That is enough to spot a brain that is clearly ahead of the field; no tournament we could afford would separate brains that are close.
2. A **head to head** of every competitor against the current champion: 5 games, and 16 more for the three best. This is the test that matters, so it gets the most play.

(Rounds 1–7 used shorter games: five Swiss rounds of 20 games of 9,000 ticks per pairing, and 20 + 64 games against the champion. The total play is similar; longer games just make each result read more like a real match. Goals per game are four times higher from round 8, so compare goal differences between the two periods per 9,000 ticks.)

Goals are rare and random in this game (between good brains, most games end in a draw), so a handful of games proves nothing. A brain only becomes champion by beating the current one convincingly over three rounds: across all its games against the champion in the last three rounds, it must be significantly better (with a 95% confidence interval), it must have been ahead in each of those rounds, and it must finish at least as high as the champion in the latest Swiss. Pooling three rounds uses all the evidence, so a brain that really is better gets through, while a lucky round can't carry anyone. (Until round 9 the rule was two significant wins in a row.) The first champion is simply the winner of round 1.

The showcase game each round is played between the top two of that round's Swiss. The showcase games are 9,000 ticks long (about 90 seconds of video). From round 6 on, the pair play five recorded games and the blog shows the most *typical* one: the game whose goal difference is closest to their usual margin in the Swiss that round. That way the videos show how the brains normally play, not a freak result.

## Predictions (made early, before the outcome is known)

Written down early so they can be scored honestly at the end. Percentages are my confidence.

1. The hand-structured brains (GA, Tactics-v2-GA) join once the learners catch up, and have a strong spell when they do (50%).
2. A PPO learner becomes the dominant champion within 5-15 rounds (60%), most likely from the PPO-stats or PPO-steady line.
3. LLMBrain v1 is admitted within 3-6 rounds (65%), then improves quickly through obvious fixes (keeper, defenders, attackers who get behind the ball).
4. The evolution methods (Neuro-ES, GA-net) fall behind the PPO learners (70%).
5. PPO-goals, rewarded only for goals, stays in the bottom half (75%).
6. Champion at the end: PPO lineage 55%, LLMBrain 30%, anything else 15%.
7. Games get tighter as everyone improves: fewer goals per match and more draws (70%).
8. The Coach tops the Swiss now and then but never becomes champion (60%).

Wild card (25%): LLMBrain ends up champion with its final version still under 100 readable lines.

## What we are seeing

*(To be written as the league runs.)*

## Learning per unit of compute

Winning is only half the comparison. The other half is what it cost: a brain that gets good on a tenth of the compute has learned something more efficiently. The league measures the CPU time of every brain's training (its own processes and all their workers) and prices it the way you would pay for it: at a cloud rate of about $0.04 per CPU-hour. Every round sets that cost against how strong the brain is. Dollars are a common currency for very different kinds of compute: a CPU-hour of simulation and a thousand tokens of Claude can be compared directly.

A few things to keep in mind when reading it:

- Every training process gets about two CPU cores, so wall-clock time and compute grow at roughly the same rate for everyone. The exception is Tactics-v2-GA, which runs on one worker.
- The Coach's CPU covers only the mixing and matching of player roles. The skills it mixes were paid for by the learners.
- The hand-structured brains (Tactics, GA) start with expert knowledge that cost no CPU at all: it was written by hand. Their CPU is only what it took to *tune* that knowledge.
- LLMBrain's local compute is only the diagnostic games Claude reads before each new version. Its real cost is Claude's inference: the tokens used to write each version, counted exactly per version. To put that on the same scale, both are simply priced in dollars: what the tokens cost, and what the CPU time would cost in the cloud. Both prices already include the cost of building what you are paying for: Anthropic's token price covers training Claude, and a cloud CPU-hour covers the hardware. So a version's token bill and a learner's CPU bill are a fair like-for-like comparison.
- In the final tournament, every contender's total compute is shown beside its result: a champion snapshot counts everything its line of training had used when it was crowned, and LLMBrain counts the diagnostics and token bills of every version up to the one playing.
- The one cost no one pays for in CPU is human (or Claude) design time: the hand-written skills in the Tactics and GA brains, and the reward shaping and network design of the learners.

<!-- AUTO:COMPUTE START -->

After round 9 (champion: PPO-champ-1). Compute is measured as CPU time per brain, workers included (estimated from each brain's measured rate before round 7), and priced at cloud rates: $0.04 per CPU-hour.

| Brain | Compute $ | Goals/game vs champion | Swiss pts/game |
| --- | ---: | ---: | ---: |
| LLM-v2 | $0.38 | – | waiting |
| GA | $0.24 | +13.24 ±1.88 | waiting |
| Neuro-ES | $0.24 | +1.67 ±2.39 | 1.69 |
| Coach | $0.24 | -2.20 ±3.57 | 1.50 |
| GA-net | $0.24 | +0.40 ±6.35 | 1.75 |
| PPO-steady | $0.17 | +2.48 ±2.34 | 2.69 |
| PPO-long | $0.17 | -0.80 ±4.16 | 1.19 |
| PPO-stats | $0.17 | +2.10 ±2.81 | 1.50 |
| PPO-explorer | $0.17 | -0.80 ±6.65 | 1.06 |
| PPO-goals | $0.17 | +2.20 ±3.38 | 0.75 |
| PPO-big | $0.16 | +1.60 ±4.62 | 1.75 |
| PPO-champ-1 | $0.11 | (champion) | 1.94 |
| LLM-v1 | $0.10 | -6.00 ±9.78 | 1.00 |
| Tactics-v2-GA | $0.05 | +14.29 ±3.47 | waiting |
| PPO-champ-0 | $0.02 | – | 0.50 |

By method: the total compute each approach has used (all its brains), and its best brain this round.

| Method | Brains | Compute $ (total) | Best brain | Swiss pts/game | Goals/game vs champion |
| --- | ---: | ---: | --- | ---: | ---: |
| PPO | 6 | $1.01 | PPO-steady | 2.69 | +2.48 ±2.34 |
| LLMBrain (written by Claude) | 2 | $0.38 | LLM-v1 | 1.00 | -6.00 ±9.78 |
| Genetic algorithm on a readable brain (hand-structured) | 1 | $0.24 | waiting | – | – |
| Evolution strategies | 1 | $0.24 | Neuro-ES | 1.69 | +1.67 ±2.39 |
| Coach (mixes PPO roles) | 1 | $0.24 | Coach | 1.50 | -2.20 ±3.57 |
| Genetic algorithm on networks | 1 | $0.24 | GA-net | 1.75 | +0.40 ±6.35 |
| Tactics GA (hand-structured) | 1 | $0.05 | waiting | – | – |

<!-- AUTO:COMPUTE END -->

## Round by round

<!-- AUTO:ROUNDS START -->

### Round 0: untrained

Two brains that have never trained: **untrained-1** and **untrained-2**, both random neural networks, moving on nothing but the random exploration noise every learner starts with. Any goals are accidents.

**Showcase: untrained-1 6 – 4 untrained-2** (untrained-1 wins). Video: [round-000.mp4](videos/league2/round-000.mp4)

untrained-1 beat untrained-2 6–4. Goals: 3' untrained-2, 13' untrained-1, 20' untrained-1, 24' untrained-1, 39' untrained-1, 48' untrained-1, 53' untrained-2, 69' untrained-2…. untrained-1 is rarely near the ball, is scattered all over the pitch and runs flat out almost all the time. untrained-2 is rarely near the ball, is scattered all over the pitch and runs flat out almost all the time. untrained-1 kept the ball in the opposition half 69% of the time; shots on goal 5–4.

*Pure noise. With no friction on the pitch, the random pushes of untrained players add up, so everyone drifts around at full speed, bouncing off the walls. The ten goals are accidents: the ball gets knocked into a goal mouth by whoever happens to be passing.*

### Round 1 (06:11)

**Showcase: PPO-stats 2 – 1 PPO-steady** (PPO-stats wins). Video: [round-001.mp4](videos/league2/round-001.mp4)

PPO-stats beat PPO-steady 2–1. Goals: 31' PPO-steady, 32' PPO-stats, 45' PPO-stats. PPO-stats is rarely near the ball, moves as a tight pack and leaves its goal empty. PPO-steady is rarely near the ball and keeps a goalkeeper back 96% of the time. PPO-stats kept the ball in the opposition half 64% of the time; shots on goal 6–0. New: the first showcase with a real goalkeeper.

*After 20 minutes of learning, the first 'strategy' is camping, not football. Watch where the players are: the ball drifts around an empty midfield while all ten players sit at one end. PPO-stats goal-hangs, its whole team parked in the corner beside the opponent's goal, waiting for the ball to drift over (with no friction, it eventually does). PPO-steady has parked the bus: its players pack into and around their own goal mouth. It never took a shot, but it did discover the goalkeeper.*

Top of the Swiss: 1. PPO-stats (1.51 pts/game), 2. PPO-steady (1.34 pts/game), 3. GA-net (1.30 pts/game), 4. Neuro-ES (1.11 pts/game), 5. Coach (1.09 pts/game).

- NEW CHAMPION: PPO-stats -> PPO-champ-0

### Round 2 (06:47)

**Showcase: PPO-stats 3 – 4 Coach** (Coach wins). Video: [round-002.mp4](videos/league2/round-002.mp4)

Coach beat PPO-stats 4–3. Goals: 22' Coach, 29' PPO-stats, 34' Coach, 41' Coach, 43' PPO-stats, 70' Coach, 72' PPO-stats. PPO-stats is rarely near the ball, leaves its goal empty and runs flat out almost all the time. Coach is rarely near the ball, is scattered all over the pitch and keeps a goalkeeper back 82% of the time. Coach kept the ball in the opposition half 57% of the time; shots on goal 3–3.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Coach | 5 | 4 | 0 | 1 | 16.2 | 14.4 | +1.9 | **12** | – |
| 2 | PPO-champ-0 | 5 | 3 | 1 | 1 | 22.3 | 19.2 | +3.0 | **10** | 0.02 |
| 3 | GA-net | 5 | 3 | 1 | 1 | 13.6 | 12.2 | +1.4 | **10** | – |
| 4 | Neuro-ES | 5 | 3 | 0 | 2 | 12.3 | 11.2 | +1.2 | **9** | – |
| 5 | PPO-stats | 5 | 2 | 1 | 2 | 22.2 | 20.2 | +2.0 | **7** | – |
| 6 | PPO-explorer | 5 | 2 | 1 | 2 | 11.9 | 13.0 | -1.2 | **7** | – |

### Rounds 3–4

The learners keep training; PPO-champ-0 stays champion.

### Round 5 (08:09)

**Showcase: PPO-stats 0 – 2 Neuro-ES** (Neuro-ES wins). Video: [round-005.mp4](videos/league2/round-005.mp4)

Neuro-ES beat PPO-stats 2–0. Goals: 39' Neuro-ES, 75' Neuro-ES. PPO-stats is rarely near the ball, leaves its goal empty and runs flat out almost all the time. Neuro-ES is rarely near the ball, is scattered all over the pitch, keeps a goalkeeper back 99% of the time and runs flat out almost all the time. Neuro-ES kept the ball in the opposition half 54% of the time; shots on goal 1–3.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Neuro-ES | 5 | 4 | 1 | 0 | 10.1 | 7.2 | +2.9 | **13** | – |
| 2 | PPO-stats | 5 | 3 | 1 | 1 | 21.2 | 16.9 | +4.2 | **10** | – |
| 3 | GA-net | 5 | 3 | 1 | 1 | 18.0 | 15.7 | +2.3 | **10** | – |
| 4 | PPO-big | 5 | 3 | 0 | 2 | 20.1 | 18.3 | +1.8 | **9** | – |
| 5 | PPO-explorer | 5 | 2 | 1 | 2 | 14.6 | 13.7 | +1.0 | **7** | – |
| 6 | PPO-steady | 5 | 2 | 1 | 2 | 16.5 | 17.1 | -0.6 | **7** | – |

### Round 6 (08:37)

**Showcase: GA-net 2 – 2 PPO-stats** (a draw). Video: [round-006.mp4](videos/league2/round-006.mp4)

GA-net and PPO-stats drew 2–2. Goals: 3' PPO-stats, 55' GA-net, 73' GA-net, 88' PPO-stats. GA-net is rarely near the ball, moves as a tight pack, leaves its goal empty and runs flat out almost all the time. PPO-stats is rarely near the ball, is scattered all over the pitch, leaves its goal empty and runs flat out almost all the time. PPO-stats kept the ball in the opposition half 59% of the time; shots on goal 2–2.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Neuro-ES | 5 | 4 | 0 | 1 | 13.5 | 11.6 | +1.9 | **12** | – |
| 2 | PPO-steady | 5 | 3 | 1 | 1 | 17.1 | 15.0 | +2.1 | **10** | – |
| 3 | GA-net | 5 | 3 | 1 | 1 | 19.9 | 18.2 | +1.7 | **10** | – |
| 4 | PPO-stats | 5 | 2 | 2 | 1 | 20.3 | 16.8 | +3.5 | **8** | – |
| 5 | Coach | 5 | 1 | 3 | 1 | 12.6 | 12.1 | +0.5 | **6** | – |
| 6 | PPO-big | 5 | 1 | 2 | 2 | 16.4 | 16.8 | -0.4 | **5** | – |

### Round 7 (09:04)

**Showcase: PPO-champ-1 4 – 6 Neuro-ES** (Neuro-ES wins). Video: [round-007.mp4](videos/league2/round-007.mp4)

Neuro-ES beat PPO-champ-1 6–4. Goals: 2' PPO-champ-1, 13' Neuro-ES, 46' Neuro-ES, 50' Neuro-ES, 60' Neuro-ES, 62' PPO-champ-1, 66' Neuro-ES, 75' Neuro-ES…. PPO-champ-1 is rarely near the ball, is scattered all over the pitch, leaves its goal empty and runs flat out almost all the time. Neuro-ES is rarely near the ball, keeps a goalkeeper back 98% of the time and runs flat out almost all the time. PPO-champ-1 kept the ball in the opposition half 56% of the time; shots on goal 6–6. New: the most shots so far (12).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Neuro-ES | 5 | 5 | 0 | 0 | 16.1 | 9.8 | +6.2 | **15** | 0.19 |
| 2 | PPO-stats | 5 | 3 | 0 | 2 | 19.6 | 16.9 | +2.7 | **9** | 0.13 |
| 3 | PPO-champ-1 | 5 | 3 | 0 | 2 | 20.1 | 18.8 | +1.3 | **9** | 0.11 |
| 4 | PPO-long | 5 | 2 | 1 | 2 | 16.3 | 15.4 | +0.9 | **7** | 0.13 |
| 5 | PPO-steady | 5 | 2 | 1 | 2 | 15.8 | 16.1 | -0.3 | **7** | 0.13 |
| 6 | Coach | 4 | 2 | 0 | 2 | 13.9 | 14.1 | -0.2 | **6** | 0.19 |

### Round 8 (09:43)

**Showcase: Neuro-ES 2 – 1 PPO-long** (Neuro-ES wins). Video: [round-008.mp4](videos/league2/round-008.mp4)

Neuro-ES beat PPO-long 2–1. Goals: 39' Neuro-ES, 41' PPO-long, 49' Neuro-ES. Neuro-ES is rarely near the ball, moves as a tight pack and keeps a goalkeeper back 100% of the time. PPO-long is rarely near the ball, is scattered all over the pitch and leaves its goal empty. PPO-long kept the ball in the opposition half 56% of the time; shots on goal 2–7.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GA-net | 4 | 3 | 1 | 0 | 46.2 | 43.5 | +2.8 | **10** | 0.21 |
| 2 | Neuro-ES | 4 | 2 | 2 | 0 | 43.8 | 22.8 | +21.0 | **8** | 0.22 |
| 3 | Coach | 3 | 2 | 1 | 0 | 38.0 | 31.8 | +6.2 | **7** | 0.21 |
| 4 | PPO-steady | 4 | 2 | 1 | 1 | 45.5 | 47.8 | -2.2 | **7** | 0.15 |
| 5 | PPO-stats | 4 | 1 | 3 | 0 | 59.8 | 51.0 | +8.8 | **6** | 0.15 |
| 6 | PPO-explorer | 4 | 2 | 0 | 2 | 44.5 | 42.8 | +1.8 | **6** | 0.15 |

### Round 9 (10:21)

**Showcase: PPO-champ-1 3 – 4 PPO-steady** (PPO-steady wins). Video: [round-009.mp4](videos/league2/round-009.mp4)

PPO-steady beat PPO-champ-1 4–3. Goals: 7' PPO-steady, 22' PPO-champ-1, 26' PPO-champ-1, 46' PPO-steady, 49' PPO-champ-1, 55' PPO-steady, 79' PPO-steady. PPO-champ-1 is rarely near the ball, leaves its goal empty and runs flat out almost all the time. PPO-steady is rarely near the ball, is scattered all over the pitch, leaves its goal empty and runs flat out almost all the time. PPO-steady kept the ball in the opposition half 56% of the time; shots on goal 3–5.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-steady | 4 | 4 | 0 | 0 | 71.8 | 51.0 | +20.8 | **12** | 0.17 |
| 2 | Coach | 4 | 3 | 0 | 1 | 51.5 | 46.8 | +4.8 | **9** | 0.24 |
| 3 | GA-net | 4 | 2 | 1 | 1 | 51.8 | 46.0 | +5.8 | **7** | 0.24 |
| 4 | PPO-champ-1 | 4 | 2 | 1 | 1 | 57.5 | 54.8 | +2.8 | **7** | 0.11 |
| 5 | PPO-stats | 4 | 2 | 1 | 1 | 41.2 | 40.5 | +0.8 | **7** | 0.17 |
| 6 | Neuro-ES | 4 | 1 | 3 | 0 | 33.0 | 25.8 | +7.2 | **6** | 0.24 |

<!-- AUTO:ROUNDS END -->
