# From random flailing to real football: a league of learning soccer brains

*Draft. Videos are local files for now (`videos/league2/`); they will be uploaded to YouTube before publishing.*

## The idea

[ai-soccer](https://github.com/eparkinson/ai-soccer) is a small, physics-based soccer sandbox: two teams of five circular players on a flat pitch, a ball with no friction, elastic collisions, and goal mouths in the walls at each end. Every tick, a *brain* decides how hard each of its five players accelerates, and in which direction. That is all a brain controls.

This post follows a league of brains that all start knowing nothing. Nobody is shown how to play, and none of the hand-written heuristic brains in the project take part. The brains only ever play each other, and every so often the league holds a tournament to see who is improving. Every round we record a game between the two best brains of that round, so you can watch the league learn, from the first aimless games to (hopefully) something that looks like football.

## The competitors

Every brain starts random, and each one learns in a different way:

- **Reinforcement learners** (PPO, a standard policy-gradient method). Each is a small neural network that sees one player's view of the game and outputs that player's acceleration; all five players share it. They learn from rewards: a goal is worth +2, a goal conceded -2, and most also get small rewards for moving the ball forward and getting to it first. Each makes a different bet:
  - *steady*: standard settings,
  - *goals*: rewarded for goals only, nothing else,
  - *stats*: extra rewards for shots and for spreading out, the two team statistics that most often go with winning,
  - *big*: a network twice as wide,
  - *long*: values goals further in the future.
  - Later branches (from round 9): *bigger* (big, widened again to 512x512), *longer* (long, looking even further ahead), and *repeat4* (stats, deciding every 4 ticks instead of 2, which a side experiment showed learns faster per CPU-hour). A sixth original learner, *explorer* (more exploration), was retired after round 9 to make room.
- **Neuro-ES**: the same kind of neural network, but trained by *evolution strategies* instead of rewards: it tries lots of small random changes to all its weights and keeps moving in the direction of the changes that won more games. From round 9 it has two branches: *bold* (bigger steps) and *wide* (twice as many directions tried per generation).
- **GA-net**: a *genetic algorithm* on the same network's weights: a population of 24 random networks; each generation the best quarter survive and the rest are replaced by mutated copies of them.
- **Coach**: picks a team from the best *players* in the league, of any kind: a goalkeeper from one brain, defenders from another, a striker from a third. Every round it scouts all the brains and rates each player like a football analyst would: every touch of the ball is credited with how much it changed the team's chance of scoring next (a clearance off the line counts as much as a shot), alongside goals, assists and saves. It then tries the best-rated player for each position and keeps a swap only if the new team beats the old one. (Until round 13 it could only mix player roles from the reinforcement learners.)
- **PPO-roles** (from round 10): PPO with a separate network for each of the five players, to see whether positions emerge when each player has its own brain.
- **GP** (from round 10): *genetic programming*, evolving small readable rules ("go to the point behind the ball...") from random programs, using only basic geometry: no built-in skills. The machine's answer to LLMBrain.

One brain is written rather than learned:

- **LLMBrain**: a deliberately naive brain written by Claude (version 1: "everyone run at the ball"), which Claude improves with one small, explainable change at a time, but only in rounds where it isn't leading. Each new version costs compute (Claude's tokens), so it's only spent when it's needed, and each time the aim is the smallest change that gets it back on top, or near it.

New brains wait outside the league until they are ready. Each round they play 84 games' worth of play against the champion. Once they are at most 2 goals per 9,000 ticks better, they become *sparring partners*: the learners train against them, but they don't play in the league. They join the league once they are level with the champion in two rounds running: no more than half a goal per 9,000 ticks better on average, and not significantly better (the 95% confidence interval includes zero). LLMBrain, PPO-roles and GP all came in through this door.

## How the league works

Everything trains at the same time, on one 24-core machine. Every 20 minutes the league pauses training and plays a round:

1. A **Swiss tournament** among all the competitors and past champions: four rounds in which brains with similar scores meet each other. Each pairing plays 6 games of 36,000 ticks, about 6 minutes of play each. That is enough to spot a brain that is clearly ahead of the field; no tournament we could afford would separate brains that are close.
2. **Qualifying games** for brains waiting to join: each plays the champion, to see whether it is ready.

(Rounds 1–7 used shorter games: five Swiss rounds of 20 games of 9,000 ticks per pairing. The total play is similar; longer games just make each result read more like a real match. Goals per game are four times higher from round 8, so compare goal differences between the two periods per 9,000 ticks.)

Goals are rare and random in this game (between good brains, most games end in a draw), so a handful of games proves nothing.

### Round winners and the champion

- **The round winner** is the brain that tops that round's tournament. It is a snapshot of one round, and one round is a small sample: a brain can top one round and finish eighth in the next.
- **The champion** is the best tournament performer over the last three rounds. Every match of those three tournaments goes into a rating: a brain's average goal difference against an average opponent, allowing for who it played (the Swiss pairs strong brains with strong brains, so raw points would punish the leaders). A brain takes the title when it has played all three rounds, tops that rating, and is above the current champion with 95% confidence.

Think of the round winner as this week's result and the champion as the form table: a lucky week doesn't win the title, but a brain that keeps beating the field does. The champion also serves as the league's fixed yardstick: brains waiting to join qualify by how they do against it. The first champion was simply the winner of round 1.

The showcase game each round is played between the top two of that round's Swiss. The showcase games are 9,000 ticks long (about 90 seconds of video). From round 6 on, the pair play five recorded games and the blog shows the most *typical* one: the game whose goal difference is closest to their usual margin in the Swiss that round. That way the videos show how the brains normally play, not a freak result.

## Predictions (made early, before the outcome is known)

Written down early so they can be scored honestly at the end. Percentages are my confidence.

1. A PPO learner becomes the dominant champion within 5-15 rounds (60%), most likely from the PPO-stats or PPO-steady line.
2. LLMBrain v1 is admitted within 3-6 rounds (65%), then improves quickly through obvious fixes (keeper, defenders, attackers who get behind the ball).
3. The evolution methods (Neuro-ES, GA-net) fall behind the PPO learners (70%).
4. PPO-goals, rewarded only for goals, stays in the bottom half (75%).
5. Champion at the end: PPO lineage 55%, LLMBrain 30%, anything else 15%.
6. Games get tighter as everyone improves: fewer goals per match and more draws (70%).
7. The Coach tops the Swiss now and then but never becomes champion (60%).

Wild card (25%): LLMBrain ends up champion with its final version still under 100 readable lines.

## What we are seeing

*As of round 19. This section is updated as the league runs.*

**From noise to camping to football.** Untrained brains just drift around at full speed, and goals are accidents. After 20 minutes of training the first "strategy" is camping: one team parks all five players in the corner beside the opponent's goal and waits for the ball to drift over; another packs its own goal mouth. Only after that do the brains start to contest the ball, shoot and spread out.

**The goalkeeper is the big discovery, and evolution found it first.** In every showcase it plays, Neuro-ES keeps one player in its goal mouth almost the whole game (98-100%). No other learning brain does: the reward-trained networks (PPO) never keep a keeper for more than a tenth of the game. Neuro-ES topped the league in rounds 7 and 8 and took the title from PPO-stats in round 9. LLMBrain reached the same conclusion from the other direction: its first version, "everyone run at the ball", conceded 98% of its goals into an empty net, and its one change for version 2 was a goalkeeper. Evolution found the keeper by trial and error; Claude found it by reading the diagnostics.

**Filling the goal, and changing the rules.** By round 14 the evolved brains had found a simple strong defence: put all five players in the goal mouth. GP, evolving readable rules from scratch, won its first league round doing exactly that, and both evolution-strategies brains did the same. The reason is geometry: five players side by side are 240 px wide, the goal was 300 px, and the gaps left were narrower than the ball, so a tidy wall could not be beaten by a shot. A single good goalkeeper should be enough to stop long shots anyway, so after round 15 the goal was widened to 400 px. Now five players can't seal it. In the first round with the wider goal, goals across the league rose by 42%, GP conceded three times as many as before, and LLM-v2, whose lone keeper now guards a much wider line, conceded twice as many. Within two rounds the answer began to appear, from an unexpected brain: the Coach, which picks the best-rated players from all the others, kept four of GP's wall players and added one striker, and topped the league in rounds 16 and 17, while GP's all-in wall dropped to 16th. The question was whether evolution would adapt on its own. It did, within four rounds: GP's rules evolved from "everyone stand in the goal" into "everyone follow the ball's height along our goal line", a wall that moves with the ball. The new program is 13 nodes:

```python
def target():
    return point(x(opp_at_ball - opp_goal), y(ball) if ball_in_own_half else y(ball_vel - ball))
```

and it took GP from 16th back to 3rd.

**Reward shaping matters, at least early on.** The first champion, PPO-stats, is the learner with the most shaping: small rewards for shots, spreading out, moving the ball forward and getting to it first. PPO-goals, rewarded for goals and nothing else, has been at or near the bottom all along: goals are too rare a signal to learn from quickly.

**There is no single ladder.** In round 8, PPO-big dominated its games against a saved version of the champion, yet finished 9th in the tournament: it had found a way to beat that one brain, and little else. Styles counter each other, which is why the title is decided by results against the whole field, over three rounds.

**Single rounds are noisy.** PPO-steady went from 8th to 1st in one round. That is why the videos show a typical game rather than a lucky one, and why a brain has to be better over three rounds to become champion.

**It is all very cheap.** Each learner's training so far has cost about $0.15-0.22 of compute at cloud prices. LLMBrain's second version alone cost more than that in tokens (about $0.28): the LLM is the most expensive competitor per step, and so far it has taken very few steps.

## Learning per unit of compute

Winning is only half the comparison. The other half is what it cost: a brain that gets good on a tenth of the compute has learned something more efficiently. The league measures the CPU time of every brain's training (its own processes and all their workers) and prices it the way you would pay for it: at a cloud rate of about $0.04 per CPU-hour. Every round sets that cost against how strong the brain is. Dollars are a common currency for very different kinds of compute: a CPU-hour of simulation and a thousand tokens of Claude can be compared directly.

A few things to keep in mind when reading it:

- Every training process gets about two CPU cores, so wall-clock time and compute grow at roughly the same rate for everyone.
- The Coach's CPU covers only the mixing and matching of player roles. The skills it mixes were paid for by the learners.
- LLMBrain's local compute is only the diagnostic games Claude reads before each new version. Its real cost is Claude's inference: the tokens used to write each version, counted exactly per version. To put that on the same scale, both are simply priced in dollars: what the tokens cost, and what the CPU time would cost in the cloud. Both prices already include the cost of building what you are paying for: Anthropic's token price covers training Claude, and a cloud CPU-hour covers the hardware. So a version's token bill and a learner's CPU bill are a fair like-for-like comparison.
- In the final tournament, every contender's total compute is shown beside its result: a champion snapshot counts everything its line of training had used when it was crowned, and LLMBrain counts the diagnostics and token bills of every version up to the one playing.
- The one cost no one pays for in CPU is human (or Claude) design time: LLMBrain's rules, and the reward shaping and network design of the learners.

<!-- AUTO:COMPUTE START -->

After round 23 (champion: champ-3). Compute is measured as CPU time per brain, workers included (estimated from each brain's measured rate before round 7), and priced at cloud rates: $0.04 per CPU-hour.

| Brain | Compute $ | Goals/game vs champion | Swiss pts/game |
| --- | ---: | ---: | ---: |
| Neuro-ES | $0.52 | – | 2.46 |
| Neuro-ES-wide | $0.51 | – | 1.42 |
| Coach | $0.50 | – | 1.83 |
| Neuro-ES-bold | $0.50 | – | 1.62 |
| GA-net | $0.44 | – | waiting |
| LLM-v2 | $0.38 | – | 2.21 |
| PPO-steady | $0.34 | – | 1.25 |
| PPO-goals | $0.34 | – | 1.58 |
| PPO-long | $0.34 | – | 1.58 |
| PPO-stats | $0.34 | – | 0.92 |
| PPO-longer | $0.34 | – | 1.83 |
| PPO-repeat4 | $0.33 | – | 1.00 |
| GP | $0.32 | – | 2.12 |
| PPO-bigger | $0.32 | – | 0.29 |
| PPO-big | $0.30 | – | waiting |
| PPO-roles | $0.27 | – | 1.12 |
| PPO-champ-2 | $0.17 | – | 1.29 |
| PPO-explorer | $0.15 | – | waiting |
| PPO-champ-1 | $0.11 | – | 1.00 |
| LLM-v1 | $0.10 | – | waiting |
| PPO-champ-0 | $0.02 | – | waiting |
| champ-3 | $0.00 | (champion) | waiting |

By method: the total compute each approach has used (all its brains), and its best brain this round.

| Method | Brains | Compute $ (total) | Best brain | Swiss pts/game | Goals/game vs champion |
| --- | ---: | ---: | --- | ---: | ---: |
| PPO | 10 | $2.62 | PPO-longer | 1.83 | – |
| Evolution strategies | 3 | $1.10 | Neuro-ES | 2.46 | – |
| Coach (mixes PPO roles) | 1 | $0.50 | Coach | 1.83 | – |
| Genetic algorithm on networks | 1 | $0.44 | waiting | – | – |
| LLMBrain (written by Claude) | 2 | $0.38 | LLM-v2 | 2.21 | – |
| GP | 1 | $0.32 | GP | 2.12 | – |
| champ-3 | 1 | $0.00 | waiting | – | – |

<!-- AUTO:COMPUTE END -->

## Round by round

<!-- AUTO:ROUNDS START -->

### Round 0: untrained

Two brains that have never trained: **untrained-1** and **untrained-2**, both random neural networks, moving on nothing but the random exploration noise every learner starts with. Any goals are accidents.

**Showcase: untrained-1 6 – 4 untrained-2** (untrained-1 wins). Video: [round-000.mp4](videos/league2/round-000.mp4)

untrained-1 beat untrained-2 6–4. Goals: 3' untrained-2, 13' untrained-1, 20' untrained-1, 24' untrained-1, 39' untrained-1, 48' untrained-1, 53' untrained-2, 69' untrained-2…. untrained-1 is rarely near the ball, is scattered all over the pitch and runs flat out almost all the time. untrained-2 is rarely near the ball, is scattered all over the pitch and runs flat out almost all the time. untrained-1 kept the ball in the opposition half 69% of the time; shots on goal 6–4.

*Pure noise. With no friction on the pitch, the random pushes of untrained players add up, so everyone drifts around at full speed, bouncing off the walls. The ten goals are accidents: the ball gets knocked into a goal mouth by whoever happens to be passing.*

### Round 1 (06:11)

**Showcase: PPO-stats 2 – 1 PPO-steady** (PPO-stats wins). Video: [round-001.mp4](videos/league2/round-001.mp4)

PPO-stats beat PPO-steady 2–1. Goals: 31' PPO-steady, 32' PPO-stats, 45' PPO-stats. PPO-stats is rarely near the ball, moves as a tight pack and leaves its goal empty. PPO-steady is rarely near the ball and keeps a goalkeeper back 98% of the time. PPO-stats kept the ball in the opposition half 64% of the time; shots on goal 9–0. New: the first showcase with a real goalkeeper.

*After 20 minutes of learning, the first 'strategy' is camping, not football. Watch where the players are: the ball drifts around an empty midfield while all ten players sit at one end. PPO-stats goal-hangs, its whole team parked in the corner beside the opponent's goal, waiting for the ball to drift over (with no friction, it eventually does). PPO-steady has parked the bus: its players pack into and around their own goal mouth. It never took a shot, but it did discover the goalkeeper.*

Top of the Swiss: 1. PPO-stats (1.51 pts/game), 2. PPO-steady (1.34 pts/game), 3. GA-net (1.30 pts/game), 4. Neuro-ES (1.11 pts/game), 5. Coach (1.09 pts/game).

- NEW CHAMPION: PPO-stats -> PPO-champ-0

### Round 2 (06:47)

**Showcase: PPO-stats 3 – 4 Coach** (Coach wins). Video: [round-002.mp4](videos/league2/round-002.mp4)

Coach beat PPO-stats 4–3. Goals: 22' Coach, 29' PPO-stats, 34' Coach, 41' Coach, 43' PPO-stats, 70' Coach, 72' PPO-stats. PPO-stats is rarely near the ball, leaves its goal empty and runs flat out almost all the time. Coach is rarely near the ball, is scattered all over the pitch and keeps a goalkeeper back 87% of the time. Coach kept the ball in the opposition half 57% of the time; shots on goal 4–4.

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

Neuro-ES beat PPO-stats 2–0. Goals: 39' Neuro-ES, 75' Neuro-ES. PPO-stats is rarely near the ball, leaves its goal empty and runs flat out almost all the time. Neuro-ES is rarely near the ball, is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Neuro-ES kept the ball in the opposition half 54% of the time; shots on goal 3–3.

*Neuro-ES wins with something no other brain has yet: a goalkeeper. One of its players stays in its goal mouth for 99% of the game, while everyone else on the pitch runs flat out. PPO-stats has more of the ball but leaves its goal empty.*

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

GA-net and PPO-stats drew 2–2. Goals: 3' PPO-stats, 55' GA-net, 73' GA-net, 88' PPO-stats. GA-net is rarely near the ball, moves as a tight pack, leaves its goal empty and runs flat out almost all the time. PPO-stats is rarely near the ball, is scattered all over the pitch, leaves its goal empty and runs flat out almost all the time. PPO-stats kept the ball in the opposition half 59% of the time; shots on goal 3–3.

*The first video chosen as the most typical of five games: a 2-2 between the two most attacking networks. Neither keeps a goalkeeper. GA-net moves as a tight pack (average spacing 183) while PPO-stats spreads wide (629) and pushes into the opposition half 59% of the time.*

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

Neuro-ES beat PPO-champ-1 6–4. Goals: 2' PPO-champ-1, 13' Neuro-ES, 46' Neuro-ES, 50' Neuro-ES, 60' Neuro-ES, 62' PPO-champ-1, 66' Neuro-ES, 75' Neuro-ES…. PPO-champ-1 is rarely near the ball, is scattered all over the pitch, leaves its goal empty and runs flat out almost all the time. Neuro-ES is rarely near the ball, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. PPO-champ-1 kept the ball in the opposition half 56% of the time; shots on goal 7–7. New: the most shots so far (14).

*Neuro-ES again, and again with a permanent goalkeeper (98% of the game), beats PPO-champ-1, a saved version of champion PPO-stats, 6-4. It went on to top the Swiss by a distance: the keeper is worth more than any trick the networks trained with rewards have found.*

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

Neuro-ES beat PPO-long 2–1. Goals: 39' Neuro-ES, 41' PPO-long, 49' Neuro-ES. Neuro-ES is rarely near the ball, moves as a tight pack and keeps a goalkeeper back 100% of the time. PPO-long is rarely near the ball, is scattered all over the pitch and leaves its goal empty. PPO-long kept the ball in the opposition half 56% of the time; shots on goal 6–10. New: the most shots so far (16).

*The first round of long matches. Neuro-ES wins 2-1 with its keeper back the whole game and its team compact (spacing 170), while PPO-long is spread all over the pitch (935) and takes 7 shots to Neuro-ES's 2 but can't beat the keeper.*

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

PPO-steady beat PPO-champ-1 4–3. Goals: 7' PPO-steady, 22' PPO-champ-1, 26' PPO-champ-1, 46' PPO-steady, 49' PPO-champ-1, 55' PPO-steady, 79' PPO-steady. PPO-champ-1 is rarely near the ball, leaves its goal empty and runs flat out almost all the time. PPO-steady is rarely near the ball, is scattered all over the pitch and runs flat out almost all the time. PPO-steady kept the ball in the opposition half 56% of the time; shots on goal 3–7.

*PPO-steady, 8th a round earlier, beats PPO-champ-1, a saved version of PPO-stats, 4-3. Still no goalkeeper on either side: none of the reward-trained networks has found one. PPO-steady's players are less drawn to the ball than the champion's, and it wins the territory battle (56%).*

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-steady | 4 | 4 | 0 | 0 | 71.8 | 51.0 | +20.8 | **12** | 0.17 |
| 2 | Coach | 4 | 3 | 0 | 1 | 51.5 | 46.8 | +4.8 | **9** | 0.24 |
| 3 | GA-net | 4 | 2 | 1 | 1 | 51.8 | 46.0 | +5.8 | **7** | 0.24 |
| 4 | PPO-champ-1 | 4 | 2 | 1 | 1 | 57.5 | 54.8 | +2.8 | **7** | 0.11 |
| 5 | PPO-stats | 4 | 2 | 1 | 1 | 41.2 | 40.5 | +0.8 | **7** | 0.17 |
| 6 | Neuro-ES | 4 | 1 | 3 | 0 | 33.0 | 25.8 | +7.2 | **6** | 0.24 |

### Round 10 (11:05)

**Showcase: Neuro-ES-wide 2 – 1 LLM-v2** (Neuro-ES-wide wins). Video: [round-010.mp4](videos/league2/round-010.mp4)

Neuro-ES-wide beat LLM-v2 2–1. Goals: 14' Neuro-ES-wide, 44' Neuro-ES-wide, 76' LLM-v2. Neuro-ES-wide is rarely near the ball, moves as a tight pack, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v2 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v2 kept the ball in the opposition half 51% of the time; shots on goal 14–7. New: the most shots so far (21).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Neuro-ES-wide | 4 | 3 | 1 | 0 | 47.5 | 25.5 | +22.0 | **10** | 0.24 |
| 2 | Neuro-ES | 4 | 3 | 1 | 0 | 43.5 | 26.0 | +17.5 | **10** | 0.24 |
| 3 | PPO-stats | 4 | 3 | 1 | 0 | 64.0 | 52.2 | +11.8 | **10** | 0.17 |
| 4 | LLM-v2 | 3 | 3 | 0 | 0 | 57.5 | 31.8 | +25.8 | **9** | 0.38 |
| 5 | PPO-longer | 4 | 2 | 1 | 1 | 50.0 | 53.0 | -3.0 | **7** | 0.17 |
| 6 | PPO-champ-1 | 4 | 2 | 0 | 2 | 67.2 | 70.2 | -3.0 | **6** | 0.11 |

### Round 11 (11:49)

**Showcase: LLM-v2 2 – 1 Neuro-ES-wide** (LLM-v2 wins). Video: [round-011.mp4](videos/league2/round-011.mp4)

LLM-v2 beat Neuro-ES-wide 2–1. Goals: 6' Neuro-ES-wide, 35' LLM-v2, 38' LLM-v2. LLM-v2 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Neuro-ES-wide is rarely near the ball, moves as a tight pack, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v2 kept the ball in the opposition half 57% of the time; shots on goal 18–7. New: the most shots so far (25).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v2 | 4 | 3 | 1 | 0 | 72.2 | 48.2 | +24.0 | **10** | 0.38 |
| 2 | Neuro-ES | 4 | 3 | 1 | 0 | 47.8 | 34.8 | +13.0 | **10** | 0.26 |
| 3 | Neuro-ES-wide | 4 | 3 | 0 | 1 | 44.5 | 27.2 | +17.2 | **9** | 0.25 |
| 4 | PPO-long | 4 | 3 | 0 | 1 | 46.2 | 49.0 | -2.8 | **9** | 0.18 |
| 5 | PPO-stats | 4 | 2 | 1 | 1 | 44.8 | 42.5 | +2.2 | **7** | 0.18 |
| 6 | Neuro-ES-bold | 4 | 2 | 1 | 1 | 29.2 | 27.0 | +2.2 | **7** | 0.26 |

### Round 12 (12:36)

**Showcase: LLM-v2 3 – 2 Neuro-ES** (LLM-v2 wins). Video: [round-012.mp4](videos/league2/round-012.mp4)

LLM-v2 beat Neuro-ES 3–2. Goals: 2' LLM-v2, 19' Neuro-ES, 26' LLM-v2, 37' LLM-v2, 47' Neuro-ES. LLM-v2 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Neuro-ES is rarely near the ball, moves as a tight pack, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v2 kept the ball in the opposition half 52% of the time; shots on goal 15–53. New: the most shots so far (68).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v2 | 4 | 4 | 0 | 0 | 84.2 | 53.5 | +30.8 | **12** | 0.38 |
| 2 | Neuro-ES | 4 | 3 | 1 | 0 | 24.2 | 18.2 | +6.0 | **10** | 0.28 |
| 3 | GA-net | 4 | 3 | 0 | 1 | 62.2 | 61.2 | +1.0 | **9** | 0.28 |
| 4 | PPO-big | 4 | 3 | 0 | 1 | 62.5 | 62.2 | +0.2 | **9** | 0.19 |
| 5 | Neuro-ES-wide | 4 | 2 | 1 | 1 | 27.0 | 21.0 | +6.0 | **7** | 0.27 |
| 6 | PPO-champ-2 | 4 | 2 | 1 | 1 | 50.8 | 47.5 | +3.2 | **7** | 0.17 |

### Round 13 (13:04)

**Showcase: LLM-v2 3 – 2 Neuro-ES-wide** (LLM-v2 wins). Video: [round-013.mp4](videos/league2/round-013.mp4)

LLM-v2 beat Neuro-ES-wide 3–2. Goals: 22' Neuro-ES-wide, 43' LLM-v2, 65' LLM-v2, 80' Neuro-ES-wide, 85' LLM-v2. LLM-v2 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Neuro-ES-wide is rarely near the ball, moves as a tight pack, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v2 kept the ball in the opposition half 53% of the time; shots on goal 16–31.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v2 | 4 | 3 | 1 | 0 | 66.5 | 40.5 | +26.0 | **10** | 0.38 |
| 2 | Neuro-ES | 4 | 3 | 1 | 0 | 18.3 | 12.2 | +6.2 | **10** | 0.30 |
| 3 | Neuro-ES-wide | 4 | 3 | 0 | 1 | 39.0 | 30.8 | +8.2 | **9** | 0.29 |
| 4 | Neuro-ES-bold | 4 | 2 | 1 | 1 | 38.7 | 27.0 | +11.7 | **7** | 0.30 |
| 5 | PPO-champ-2 | 4 | 2 | 1 | 1 | 62.2 | 60.8 | +1.3 | **7** | 0.17 |
| 6 | PPO-roles | 4 | 2 | 1 | 1 | 65.5 | 65.7 | -0.2 | **7** | 0.05 |

### Round 14 (13:31)

**Showcase: GP 3 – 3 LLM-v2** (a draw). Video: [round-014.mp4](videos/league2/round-014.mp4)

GP and LLM-v2 drew 3–3. Goals: 3' LLM-v2, 5' LLM-v2, 10' GP, 25' LLM-v2, 51' GP, 73' GP. GP is rarely near the ball, moves as a tight pack, keeps a goalkeeper back 97% of the time and runs flat out almost all the time. LLM-v2 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v2 kept the ball in the opposition half 60% of the time; shots on goal 9–26.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v2 | 4 | 3 | 1 | 0 | 57.8 | 43.3 | +14.5 | **10** | 0.38 |
| 2 | champ-3 | 4 | 3 | 0 | 1 | 76.3 | 42.3 | +34.0 | **9** | 0.00 |
| 3 | Neuro-ES | 4 | 3 | 0 | 1 | 45.2 | 30.8 | +14.3 | **9** | 0.33 |
| 4 | GP | 4 | 3 | 0 | 1 | 21.3 | 14.7 | +6.7 | **9** | 0.10 |
| 5 | PPO-big | 4 | 3 | 0 | 1 | 62.5 | 63.5 | -1.0 | **9** | 0.21 |
| 6 | Neuro-ES-wide | 4 | 2 | 1 | 1 | 39.8 | 34.7 | +5.2 | **7** | 0.31 |

### Round 15 (13:58)

**Showcase: Coach 3 – 3 champ-3** (a draw). Video: [round-015.mp4](videos/league2/round-015.mp4)

Coach and champ-3 drew 3–3. Goals: 9' champ-3, 10' champ-3, 24' Coach, 58' champ-3, 83' Coach, 90' Coach. Coach is rarely near the ball, moves as a tight pack, keeps a goalkeeper back 95% of the time and runs flat out almost all the time. champ-3 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. champ-3 kept the ball in the opposition half 56% of the time; shots on goal 8–17.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | champ-3 | 4 | 4 | 0 | 0 | 81.3 | 52.5 | +28.8 | **12** | 0.00 |
| 2 | Coach | 4 | 3 | 0 | 1 | 24.8 | 15.0 | +9.8 | **9** | 0.32 |
| 3 | GP | 4 | 3 | 0 | 1 | 23.0 | 15.0 | +8.0 | **9** | 0.13 |
| 4 | LLM-v2 | 4 | 2 | 1 | 1 | 67.7 | 45.5 | +22.2 | **7** | 0.38 |
| 5 | Neuro-ES-wide | 4 | 2 | 1 | 1 | 26.8 | 20.2 | +6.7 | **7** | 0.33 |
| 6 | Neuro-ES-bold | 4 | 2 | 1 | 1 | 16.8 | 11.2 | +5.7 | **7** | 0.34 |

### Round 16 (14:27)

**Showcase: Coach 5 – 3 champ-3** (Coach wins). Video: [round-016.mp4](videos/league2/round-016.mp4)

Coach beat champ-3 5–3. Goals: 6' Coach, 39' Coach, 45' Coach, 47' Coach, 67' Coach, 71' champ-3, 74' champ-3, 84' champ-3. Coach is rarely near the ball, is scattered all over the pitch, keeps a goalkeeper back 87% of the time and runs flat out almost all the time. champ-3 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. champ-3 kept the ball in the opposition half 58% of the time; shots on goal 20–23.

*The first round with the wider goal (400 px instead of 300). Goals across the league jump by 42%, and the brains that relied on filling the goal feel it most: GP, whose five players all stand in their own goal mouth, concedes three times as many goals as in the three rounds before (3.1 per 90 seconds of play instead of 0.9), and LLM-v2's keeper, now guarding a much wider line on its own, lets in twice as many (5.5 instead of 2.7). Both still score more than they concede. The Coach tops the table with a telling team: four of GP's wall players plus one striker, PPO-longer's player 4, whom its scouting rated the league's best attacker under the new goal. GP itself, with all five in the wall, came 6th.*

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Coach | 4 | 4 | 0 | 0 | 68.3 | 42.0 | +26.3 | **12** | 0.34 |
| 2 | champ-3 | 4 | 3 | 0 | 1 | 99.2 | 64.8 | +34.3 | **9** | 0.00 |
| 3 | LLM-v2 | 4 | 3 | 0 | 1 | 108.0 | 88.3 | +19.7 | **9** | 0.38 |
| 4 | Neuro-ES-wide | 4 | 3 | 0 | 1 | 30.5 | 24.7 | +5.8 | **9** | 0.35 |
| 5 | PPO-goals | 4 | 3 | 0 | 1 | 74.5 | 75.5 | -1.0 | **9** | 0.24 |
| 6 | Neuro-ES | 4 | 2 | 1 | 1 | 37.8 | 30.5 | +7.3 | **7** | 0.37 |

### Round 17 (14:54)

**Showcase: LLM-v2 3 – 2 Coach** (LLM-v2 wins). Video: [round-017.mp4](videos/league2/round-017.mp4)

LLM-v2 beat Coach 3–2. Goals: 11' Coach, 14' LLM-v2, 17' LLM-v2, 60' Coach, 83' LLM-v2. LLM-v2 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach is rarely near the ball, is scattered all over the pitch, keeps a goalkeeper back 83% of the time and runs flat out almost all the time. LLM-v2 kept the ball in the opposition half 59% of the time; shots on goal 9–6.

*The wall without a striker collapses: GP, all five players in its goal mouth, drops to 16th and concedes six goals per 90 seconds of play, up from under one before the goal was widened. The Coach's version, four of GP's players plus PPO-longer's striker, tops the league again and leads the title race. LLM-v2's swarm-and-keeper is 2nd. Under the new goal, what wins is a wall of four with someone up front.*

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Coach | 4 | 3 | 1 | 0 | 64.0 | 33.8 | +30.2 | **10** | 0.36 |
| 2 | LLM-v2 | 4 | 3 | 1 | 0 | 105.2 | 77.2 | +28.0 | **10** | 0.38 |
| 3 | champ-3 | 4 | 3 | 0 | 1 | 91.0 | 67.7 | +23.3 | **9** | 0.00 |
| 4 | Neuro-ES-wide | 4 | 3 | 0 | 1 | 44.8 | 32.2 | +12.7 | **9** | 0.37 |
| 5 | Neuro-ES-bold | 4 | 3 | 0 | 1 | 57.7 | 48.8 | +8.8 | **9** | 0.35 |
| 6 | PPO-stats | 4 | 3 | 0 | 1 | 94.7 | 91.5 | +3.2 | **9** | 0.25 |

### Round 18 (15:21)

**Showcase: Neuro-ES-wide 2 – 2 LLM-v2** (a draw). Video: [round-018.mp4](videos/league2/round-018.mp4)

Neuro-ES-wide and LLM-v2 drew 2–2. Goals: 10' LLM-v2, 48' Neuro-ES-wide, 50' LLM-v2, 75' Neuro-ES-wide. Neuro-ES-wide is rarely near the ball, moves as a tight pack, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v2 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v2 kept the ball in the opposition half 55% of the time; shots on goal 12–11.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v2 | 4 | 4 | 0 | 0 | 100.2 | 77.0 | +23.2 | **12** | 0.38 |
| 2 | PPO-goals | 4 | 3 | 1 | 0 | 36.2 | 27.2 | +9.0 | **10** | 0.27 |
| 3 | PPO-roles | 4 | 3 | 0 | 1 | 83.8 | 80.7 | +3.2 | **9** | 0.15 |
| 4 | Neuro-ES-wide | 4 | 2 | 1 | 1 | 66.3 | 47.5 | +18.8 | **7** | 0.39 |
| 5 | Coach | 4 | 2 | 1 | 1 | 109.0 | 94.2 | +14.8 | **7** | 0.38 |
| 6 | Neuro-ES | 4 | 2 | 1 | 1 | 47.2 | 38.8 | +8.3 | **7** | 0.41 |

### Round 19 (15:49)

**Showcase: Neuro-ES-bold 2 – 6 LLM-v2** (LLM-v2 wins). Video: [round-019.mp4](videos/league2/round-019.mp4)

LLM-v2 beat Neuro-ES-bold 6–2. Goals: 4' LLM-v2, 14' LLM-v2, 57' LLM-v2, 59' LLM-v2, 60' LLM-v2, 71' Neuro-ES-bold, 72' LLM-v2, 76' Neuro-ES-bold. Neuro-ES-bold is rarely near the ball, moves as a tight pack, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v2 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v2 kept the ball in the opposition half 60% of the time; shots on goal 4–14.

*GP is back: 3rd, after 16th two rounds ago. Evolution has rewritten its rules in response to the wider goal. Its new 13-node program sends all five players to its own goal line at the ball's height while the ball is in its half, so the whole team slides up and down in front of the goal with the ball, and back towards a corner of its own end when the ball is far away. A static wall became a moving one. LLM-v2 tops the table with 21 wins from 24 and keeps the title. This is the last round before the game is sped up.*

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v2 | 4 | 4 | 0 | 0 | 100.5 | 69.2 | +31.3 | **12** | 0.38 |
| 2 | Neuro-ES | 4 | 3 | 1 | 0 | 37.8 | 21.5 | +16.3 | **10** | 0.43 |
| 3 | Neuro-ES-wide | 4 | 3 | 0 | 1 | 76.7 | 62.3 | +14.3 | **9** | 0.41 |
| 4 | PPO-steady | 4 | 3 | 0 | 1 | 80.7 | 82.5 | -1.8 | **9** | 0.28 |
| 5 | Neuro-ES-bold | 4 | 2 | 2 | 0 | 36.8 | 19.3 | +17.5 | **8** | 0.40 |
| 6 | PPO-goals | 4 | 2 | 1 | 1 | 46.3 | 49.3 | -3.0 | **7** | 0.28 |

### Round 20 (16:18)

**Showcase: Coach 2 – 3 GP** (GP wins). Video: [round-020.mp4](videos/league2/round-020.mp4)

GP beat Coach 3–2. Goals: 34' GP, 39' GP, 50' GP, 62' Coach, 86' Coach. Coach is rarely near the ball, is scattered all over the pitch and keeps a goalkeeper back 100% of the time. GP is rarely near the ball, moves as a tight pack, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach kept the ball in the opposition half 81% of the time; shots on goal 12–7.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GP | 4 | 4 | 0 | 0 | 28.5 | 14.2 | +14.3 | **12** | 0.24 |
| 2 | Coach | 4 | 3 | 0 | 1 | 130.2 | 65.0 | +65.2 | **9** | 0.42 |
| 3 | PPO-roles | 4 | 3 | 0 | 1 | 153.2 | 151.7 | +1.5 | **9** | 0.19 |
| 4 | Neuro-ES | 4 | 2 | 2 | 0 | 39.7 | 23.2 | +16.5 | **8** | 0.45 |
| 5 | LLM-v2 | 4 | 2 | 1 | 1 | 131.7 | 115.0 | +16.7 | **7** | 0.38 |
| 6 | Neuro-ES-bold | 4 | 2 | 1 | 1 | 33.0 | 17.2 | +15.8 | **7** | 0.42 |

### Round 21 (16:47)

**Showcase: LLM-v2 5 – 5 GP** (a draw). Video: [round-021.mp4](videos/league2/round-021.mp4)

LLM-v2 and GP drew 5–5. Goals: 9' GP, 16' GP, 33' LLM-v2, 35' GP, 36' LLM-v2, 73' GP, 74' LLM-v2, 75' LLM-v2…. LLM-v2 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP is rarely near the ball, moves as a tight pack, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v2 kept the ball in the opposition half 50% of the time; shots on goal 16–9.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GP | 4 | 4 | 0 | 0 | 137.3 | 87.2 | +50.2 | **12** | 0.27 |
| 2 | Neuro-ES | 4 | 3 | 0 | 1 | 63.5 | 36.5 | +27.0 | **9** | 0.48 |
| 3 | LLM-v2 | 4 | 3 | 0 | 1 | 116.2 | 93.7 | +22.5 | **9** | 0.38 |
| 4 | Neuro-ES-bold | 4 | 2 | 2 | 0 | 55.8 | 30.3 | +25.5 | **8** | 0.45 |
| 5 | Neuro-ES-wide | 4 | 2 | 1 | 1 | 76.8 | 56.5 | +20.3 | **7** | 0.46 |
| 6 | Coach | 4 | 2 | 1 | 1 | 29.2 | 16.8 | +12.3 | **7** | 0.45 |

### Round 22 (17:09)

**Showcase: Coach 2 – 3 Neuro-ES-wide** (Neuro-ES-wide wins). Video: [round-022.mp4](videos/league2/round-022.mp4)

Neuro-ES-wide beat Coach 3–2. Goals: 31' Neuro-ES-wide, 34' Coach, 37' Neuro-ES-wide, 68' Neuro-ES-wide, 73' Coach. Coach is rarely near the ball, is scattered all over the pitch and keeps a goalkeeper back 100% of the time. Neuro-ES-wide is rarely near the ball, moves as a tight pack, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach kept the ball in the opposition half 61% of the time; shots on goal 8–8.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Coach | 4 | 4 | 0 | 0 | 94.2 | 53.3 | +40.8 | **12** | 0.47 |
| 2 | LLM-v2 | 4 | 3 | 0 | 1 | 119.5 | 100.0 | +19.5 | **9** | 0.38 |
| 3 | Neuro-ES-wide | 4 | 3 | 0 | 1 | 39.2 | 23.8 | +15.3 | **9** | 0.48 |
| 4 | GP | 4 | 3 | 0 | 1 | 29.7 | 16.8 | +12.8 | **9** | 0.29 |
| 5 | PPO-roles | 4 | 3 | 0 | 1 | 131.8 | 144.5 | -12.7 | **9** | 0.25 |
| 6 | PPO-goals | 4 | 2 | 1 | 1 | 50.8 | 49.2 | +1.7 | **7** | 0.32 |

### Round 23 (17:33)

**Showcase: Neuro-ES 6 – 6 LLM-v2** (a draw). Video: [round-023.mp4](videos/league2/round-023.mp4)

Neuro-ES and LLM-v2 drew 6–6. Goals: 2' LLM-v2, 12' Neuro-ES, 21' Neuro-ES, 40' Neuro-ES, 44' LLM-v2, 65' LLM-v2, 66' LLM-v2, 71' LLM-v2…. Neuro-ES is rarely near the ball, moves as a tight pack, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v2 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v2 kept the ball in the opposition half 56% of the time; shots on goal 20–17. New: the most goals in a showcase so far (12).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Neuro-ES | 4 | 4 | 0 | 0 | 97.8 | 60.7 | +37.2 | **12** | 0.52 |
| 2 | LLM-v2 | 4 | 3 | 0 | 1 | 120.5 | 89.7 | +30.8 | **9** | 0.38 |
| 3 | Coach | 4 | 3 | 0 | 1 | 95.7 | 73.8 | +21.8 | **9** | 0.50 |
| 4 | PPO-longer | 4 | 2 | 2 | 0 | 34.2 | 23.8 | +10.3 | **8** | 0.34 |
| 5 | GP | 4 | 2 | 1 | 1 | 74.3 | 36.7 | +37.7 | **7** | 0.32 |
| 6 | Neuro-ES-bold | 4 | 2 | 0 | 2 | 82.3 | 57.5 | +24.8 | **6** | 0.50 |

<!-- AUTO:ROUNDS END -->
