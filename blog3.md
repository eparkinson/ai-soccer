# From zero to football: a league of learning soccer brains

*Draft, league 3. Videos are local files for now (`videos/league3/`).*

## The idea

Six brains start knowing nothing and learn to play soccer by playing each other. Every round the league plays a tournament, records a typical game between the two best brains, and measures the round's winner against a fixed yardstick: a strong hand-written brain that nobody in the league ever trains against. The chart of that yardstick, round by round, is the story.

## The competitors

*(to be written)*

## How the league works

*(to be written)*

## What we are seeing

*(to be written as the league runs)*

## How GP thinks: a language for "where should I go?"

GP, short for genetic programming, was the league's most successful learner for most of league 3. It evolves one short program that answers a single question for every player, every tick: **where should I go?** The program returns a point, and the player runs there at full speed and slows to stop on it. All five players run the same program. Two words let them behave differently: `role` (their number, 0 to 4) and `home` (their kick-off spot).

**The language.** GP starts with no football in it at all: no passing, no shooting, no formations. It has only low-level geometry:

- **places:** `me`, `ball`, `own_goal`, `opp_goal`, `home`, the nearest teammate `mate`, the nearest opponent `opp`, and `opp_at_ball`, the opponent nearest the ball;
- **numbers and questions:** `role`, constants, `nearest_to_ball` (am I our player closest to the ball?), `ball_in_own_half`;
- **ways to combine them:** add and subtract points, scale, measure distances, compare, choose (`a if c else b`), and one handy word, `behind(a, b, n)`: the point n pixels past a on the line from b. So `behind(ball, opp_goal, 30)` is "30 px behind the ball, facing their goal".

A program is a small tree of these words. A tiny example:

```
def target():
    if nearest_to_ball:
        return behind(ball, opp_goal, 30)
    return home
```

This reads: the player nearest the ball goes just behind it, facing their goal; everyone else goes home.

**How it evolves.** Each generation has 64 programs. Every program plays the same opponents on the same kick-offs, and is scored on the result. The best two survive unchanged. The rest of the next generation is bred from winners picked in small tournaments:

- **crossover** swaps a branch of one parent's tree for a branch of another's;
- **mutation** replaces a branch with a random new one, or swaps one word for another of the same kind.

A small penalty per word keeps programs from growing without purpose.

**What it found.** After about ten rounds, evolution settled on an idea no one gave it: *the player nearest the ball chases it; the other four stand together about 200 px in front of their own goal, between the goal and the ball.* That is "the wall". In GP's language it is only a few words, and it held the title for most of the league.

**Why it got stuck.** The wall is about the best you can do when all you can say is *where to stand*. GP's language had no way to say *how to hit the ball*: not where the ball will be, not which way a kick will send it. It could not discover an aimed shot, because no program in its language could express one. Its plateau was a problem of vocabulary, not search.

**GP-v2: a bigger vocabulary.** So we gave evolution new words and kept everything else the same:

- `intercept`: the first point on the ball's predicted path I can reach in time;
- `strike(aim)`: where to run so that I hit the ball, at speed, towards `aim`;
- `ball_in(n)`: where the ball will be in n ticks;
- `front_mate`: my teammate furthest forward, a pass target;
- `bank_top`, `bank_bottom`: aim points that bounce the ball off a side wall into the goal;
- `save_point`: where a keeper should stand. That is on its own line, one player's width in front of the goal, where the ball's path will cross it. A keeper can't put its centre on the goal line, so aiming there leaves a gap.

GP-v2 started from GP's best program. Every old program is still a valid GP-v2 program, so a single mutation can turn "run at the ball" into "strike the ball towards their goal". Whether evolution finds and keeps those words is one of the questions of the last rounds.

**Evolution needs a designer.** GP is often described as learning from nothing, but that is only half true. Its programs start random, but its *language* was designed, and the design carries a lot of knowledge about the game:

- that a player's decision can be reduced to "where should I go?";
- that a controller will handle the running;
- that the ball, the goals and the nearest opponent are the things worth naming;
- and, in GP-v2, that intercepting, striking and keeping goal are useful ideas.

Each of those is a human judgement about soccer. Evolution only searches the space the language opens up. When GP stalled, the fix was not more evolution; it was better design.

PPO is the opposite end of the spectrum. It gets raw numbers in (positions and velocities) and produces raw accelerations out, and it has to discover everything else, including how to run to a spot and stop. That is the purest form of learning in the league, and also why it learned slowest. PPO-tactician, which learns to *adjust* Tactics' hand-designed strategy rather than play from scratch, shows the other side: learning goes much further when it is given good building blocks.

**Training against everyone.** Late in the league we also changed how GP is scored. It had been tested each generation against four brains picked at random, so the strongest brains rarely showed up. Now every program plays every brain in the tournament for the same number of ticks, and its score is goal difference per 1,000 ticks. Training measures what the tournament rewards.

## Round 30: the outsider

By round 29 the league had gone quiet. GP, a program grown by evolution, had found a simple idea (the player nearest the ball chases it, the other four stand together in front of their own goal) and nobody could get past it. Its snapshots and the Coach, who picked GP's players for his team, took turns at the top, and their games ended 0–0. The LLM line had stalled too, so we froze it.

The LLM line was an LLM writing a brain in small, careful steps. It started naive (everyone chases the ball) and after each round it could make **one small change backed by the round's evidence**. It kept a change only if it did at least as well in a quick check, and stayed around a hundred lines. It learned to keep a goalkeeper (v2) and a sweeper (v4, which led rounds 14 and 15). It learned to hit the ball from behind towards goal (v6), to slow down as it reaches its spot (v7), and to keep a second sweeper (v8). Four ideas failed their checks and never played. After ten versions and $4.68 of compute, it sat sixth, and every new idea made it worse.

Then, in round 30, we let in an outsider: **Tactics**, a brain Claude wrote in an earlier league in one go, from a full brief. It is about 600 lines with real skills: it predicts where the ball is going, meets it there, aims its kicks (at the goal, off a side wall, at a teammate), passes, and has a proper goalkeeper. Same author as the LLM line, different brief.

It won the round 39-0-3, by 32 goals a game. It beat GP 4-0-2 and the LLM line 6-0-0 (149–33). Against our yardstick it scored **+7.2 goals per 9,000 ticks**, where the best GP ever managed +1.4.

### What this says about prompting

We had assumed the careful line and the full design would end up in the same place. They didn't, and the reason is in the brief, not the model.

- **The unit of change was too small.** An aimed shot needs four things at once: predict the ball, pick a target, find the contact point behind the ball, arrive there at speed. Each on its own does nothing or makes things worse. One of our failed versions (v10) added "arrive at speed" without the aim, and conceded more. A process that keeps only changes that help right away is hill climbing, and it cannot cross a valley. The line found the best way to *stand* and stopped there.
- **The evidence shaped the ideas.** Our diagnostics measured positions: possession, shots, how many players were caught upfield. So every version "fixed" the formation. Nothing in the report said "your shots go where you don't aim them", so no version tried to aim.
- **The rules we chose for a clean story set its ceiling.** "Smallest change", "don't jump ahead to known designs" and "about a hundred lines" made every step explainable, and made a skill impossible to build.
- **Evolution hit the same wall for the same reason.** GP's building blocks could only say *where to go*, never *how to hit the ball*. Its plateau was a vocabulary problem, not a search problem.

The first lesson reaches well beyond soccer. In software, small pull requests are generally considered good and large ones bad, and with AI writing more of the code that instinct is only getting stronger. Small changes are easier to review, easier to test, and easier to roll back. But our LLM line shows **there is such a thing as too small**. The trouble starts when every change must pay for itself on arrival. A capability that needs several parts before any of them is useful can then never be built: each part alone shows no gain, or even a loss, and gets rejected. What should stay small is the review, not the capability. Land the parts behind a flag, or judge the whole feature once it's complete. Don't demand that each step improve the product on its own.

Small steps have real virtues too, and the LLM line shows those as well. Its first eight versions climbed a clean staircase against our yardstick (−10, −7, −4.4, −3.3, −1.1): every step explainable, cheap and safe. That is how most AI-assisted engineering should work most of the time. The skill is noticing when the staircase has flattened. By round 33 the signs were all there: two versions in a row that failed their checks, an evolved one-line program (15 building blocks) beating our 41 lines of carefully iterated code, and an outsider (Tactics) showing how much room there was above everyone. Small steps are the right default. When they stop paying, the next step has to be a bigger one, taken on purpose: our "v11" jumped from −1 to +20 in a single version.

### Human-AI collaboration

The rules that trapped the LLM line were ours: one small change per round, don't jump ahead, keep it short. We set them for good reasons: every step explainable, compute kept low, a clean story. Nobody noticed that together they put a ceiling on what the line could ever become. The automated process followed them faithfully. It never questioned them, because questioning its own brief was not part of its brief.

The way out did not come from inside the loop either. After round 29, Claude's own recommendation was to freeze the LLM line and wrap up the league. What broke the deadlock was a sideways question from the human running the experiment: *"How do you think your old Tactics brain might do here?"* Letting an outsider in, then reopening the frozen line under new rules, took a step back from the process, not another step inside it.

Two lessons follow.

- **Guardrails have unintended consequences.** Each of our rules made sense on its own and served a real goal. Together they defined which solutions could exist. In an automated process, a guardrail is also a boundary on the search space, and the process will not tell you when it is standing against it. It will just report, round after round, that nothing helps.
- **Know which rules serve the goal and which serve something else.** "Keep it explainable" served the story; "one small change" served cost and clarity. None of them served winning. When progress stops, the useful question is not "what should the next small step be?" but "which of my rules are arbitrary with respect to what I actually want?" Breaking those deliberately, and saying so, is not cheating. That is how the LLM line was reopened: improvement first, with the rule change written into its changelog.

An honest footnote belongs here. We call these "learning brains", and between redesigns they did learn unattended, round after round, for days. But every contender that got anywhere was also helped over a plateau at least once by the two of us working on it together:

- **The LLM line** was prompted by a person at every version, and v11 exists only because a person changed its rules.
- **GP** evolves in a language we designed. When it stalled we gave it new words (GP-v2), changed what it was scored on (goal difference against the whole field), and after a review of why it had flatlined, rebuilt it as GP-v3 with evolvable roles and less noisy selection. The role ideas came from the human; the diagnosis and the build came from Claude.
- **PPO-v2** started as an imitation of v11, and was given extra inputs and a bigger network.
- **PPO-goals** is the purest: it learned from zero, and all we ever changed were its training settings. It is also the one that never caught up.

The two biggest jumps in the league had the same shape. LLM versions v1 to v10 were written by the LLM on its own, one small step at a time, and stalled. v11 came after a person changed the rules. GP spent about 60 rounds losing almost every game to v11, then beat it within about 10 rounds of the redesign into GP-v3. In both cases, a long unattended plateau ended soon after a person and an AI sat down together to rethink it.

That is one experiment, not a proof. But the pattern held all the way through. The loops did the grinding: thousands of games, generations and updates nobody could run by hand. The decisions that changed direction came from outside the loop, out of a conversation: what to try next, which rule to break, and when to stop waiting for a plateau to fix itself. Neither side would have got there alone. The person usually noticed the rut and asked the sideways question; Claude usually found the cause and built the fix, and sometimes it was the other way round. So we don't call this human intervention. It was human-AI collaboration, and it did something neither the unattended loops nor either of us alone was doing.

### A bigger, slower field

Tactics' win started a burst of new experiments, each aimed at the wall:

- **GP-v2:** evolution with new building blocks: where I'll meet the ball, and where to hit it so it goes towards the goal, off a wall or to a teammate.
- **PPO-tactician:** a learner that sets Tactics' strategy as the game goes.
- **PPO-exploiter:** a learner that trains mostly against the champion, to find a hole in it.
- **PPO-chance:** a learner rewarded for the chance to score, measured by a model fitted to 12,544 past shots.
- **Stepping stones for PPO-shaped:** it trains against opponents it can beat about half the time.

The field grew from 8 brains to 12 and more, and the newcomers think harder per move than a wall of four does. Tactics was 4–5 times slower per game than the other brains. A rewrite made it 2.2 times faster while playing exactly the same moves: 150,000 moves checked, all identical.

A full round robin with 12 brains is 11 waves of matches, and one round took the best part of an hour. So from round 32 the league plays a **4-round Swiss** instead. The first wave is drawn at random; after that, brains are paired with others on the same points, winners against winners. It's 4 waves instead of 11, so rounds are about half as long. The price is noisier ratings in the middle and bottom of the table. The Swiss spends its smaller budget where it matters: the top brains still usually meet each other by the last waves, and most of the lopsided games (Tactics beat PPO-shaped 606–13 in round 30) drop out.

### Not giving up

After Tactics' debut we reopened the frozen LLM line under new rules: improvement comes first, big changes are allowed, it may add its own measurements, and it may go backwards to experiment. It was told only *that* better play existed (Tactics' scores), never *how*. It could play Tactics as often as it liked, but not read its code.

The result, **LLM-v11**, entered in round 34 and won all 24 of its games. It beat Tactics 152–4 over six matches and scored **+20.4 against our yardstick**, almost three times Tactics' +7.2. What makes it interesting is how it got there.

**It measured before it built.** It recorded games and found that the ball is at or near top speed 70% of the time, nothing slows it down, and it is touched only once every 15–30 ticks. *This is closer to air hockey than to football*: what decides a game is where each touch sends the ball. So it invented its own yardstick, a "first-touch test": a random moving ball, and a check of where one player's first touch sends it. The old v8 sent 46% of its first touches forwards and 23% on target.

**Its first idea failed.** Steering to the spot behind the ball by geometry got only 78% forwards and 46% on target, because the ball moves while the player lines up. That draft lost 0–8 to Tactics. The failure pointed to the answer: *simulate instead of calculate.*

**The kick planner.** Every tick, each player tries 49 moves (24 directions at full or half speed, or stop). It follows each move 40 ticks ahead with the game's own rules for acceleration and speed, finds where it would first meet the ball, and uses the game's collision rule to work out where the ball goes next. It scores each result: forwards is good, on target is better, on target with no defender able to reach the path in time is best, and towards our own goal is very bad. The best kick wins. One player using it sends 97% of its first touches forwards and 94% on target.

**Everything else stays simple.** A keeper steps to where the ball's path will cross its line, one player's width in front of the goal, not on the goal line. One poacher waits at the far post for rebounds. Two players hold a line 220 px behind the ball to meet clearances. Each of these survived an experiment; a second poacher, a deeper line and longer planning did not.

Watching v11 play GP-v2, the best evolved team, shows another reason it wins. Neither team does anything silly: GP-v2 is a well-organised side, and v11 is simply better. But GP-v2 attacks with a single striker while four players hold the wall, so nearly all its chances are long-range shots. Those are exactly what v11's lone goalkeeper handles best. It stands on its own line, one player's width in front of the goal, where the ball's path will cross, and a long shot gives it all the time it needs to get there. One well-placed keeper is enough against a team that only shoots from distance. It would not be enough against a team that attacks in numbers and scores from short range.

At the other end it's the reverse. GP-v2's wall of four is built to block, but v11 attacks with its planner-driven kicks, a poacher at the far post and a line close behind the ball. Its shots are fast and from short range, and some are deflections off the wall itself. A crowd of defenders in front of goal cannot react to a ball that changes direction a few body-lengths away at twice their top speed. So the matchup is lopsided in both directions: v11's one keeper faces only the shots it is best at, and GP-v2's many defenders face exactly the shots they are worst at.

Two things stand out. First, v11 beats Tactics because it doesn't approximate: Tactics estimates where it will meet the ball, while v11 runs the real physics forward, and against a wall of defenders a small error in aim is a blocked shot. Second, "plan ahead with the simulator" was one of the learning approaches we had listed ourselves as worth trying. Nobody told v11; it got there from its own measurements.

It was also by far the most expensive brain in the league. Its development cost **$17.62**, almost all of it tokens: the model re-reading its own growing notebook of experiments about a hundred times. That takes the LLM family to about $24 against $2–4 for the other approaches. So we froze it, and Tactics, and let the other approaches train with the cores weighted towards whoever is furthest behind, to see whether they catch up.

## Conclusion

*(draft; final numbers after the experiment rounds)*

Looking back over three leagues, every real step forward had the same four ingredients.

1. **Iteration.** No brain was good first time. GP needed a hundred generations; the LLM line needed eleven versions. Improvement came from many rounds of trying, measuring and adjusting.
2. **Evolution, in steps that aren't too small.** Keeping only what works is what drives progress. But the steps must be big enough to build something new: an aimed shot, a kick planner, a vocabulary that can say "strike". One small change at a time kept the LLM line tidy and stuck. Evolution only reached new skills once its building blocks could express them.
3. **An honest way to measure progress.** The tournament, the three-round title rating and the fixed yardstick against DefendersAndAttackers told us, round after round, what was working and what wasn't. They told us when we were stuck, too, even when the story we wanted said otherwise.
4. **Human-AI collaboration, just outside the loop.** The automated processes followed their rules faithfully, including the rules that trapped them. The breakthroughs came when a person and an AI stepped back together and asked a sideways question ("how would your old Tactics brain do here?"), changed the rules deliberately, or insisted on a fairer test. Every contender that made real progress was helped over at least one plateau this way; the only one whose design was never changed made the least.

None of these is enough alone. Iteration without evaluation wanders; evaluation without big enough steps gets stuck; and even a well-run loop needs someone to notice when it's in a rut.

### The takeaway for AI-assisted engineering

The most useful lesson from all this may have nothing to do with soccer:

1. **Default to small steps.** They are reviewable, testable, cheap and safe, and most of the time they make steady progress.
2. **Watch for the plateau.** The signs were all there before our LLM line broke through: repeated attempts that failed their checks, a far simpler solution (a one-line evolved program) outperforming ours, and evidence from elsewhere (Tactics) that there was plenty of room above.
3. **Then take a big step on purpose,** and judge it on the whole capability, not step by step. Sometimes the rules that kept the process safe are what hold it back, and changing them is the step.

## Learning per unit of compute

<!-- AUTO:COMPUTE START -->

After round 150 (champion: PPO-champ-11). Compute is measured as CPU time per brain, workers included (estimated from each brain's measured rate before round 7), and priced at cloud rates: $0.04 per CPU-hour.

| Brain | Compute $ | Goals/game vs champion | Swiss pts/game |
| --- | ---: | ---: | ---: |
| LLM-v11-GA | $25.04 | – | waiting |
| GP-v3 | $22.32 | – | 0.79 |
| LLM-v11 | $22.30 | – | 1.79 |
| PPO-v3 | $19.67 | – | 0.00 |
| PPO-v2 | $18.22 | – | 2.50 |
| PPO-champ-11 | $11.43 | (champion) | 2.38 |
| GP-v2 | $10.69 | – | waiting |
| PPO-goals | $9.41 | – | waiting |
| PPO-champ-10 | $9.31 | – | waiting |
| PPO-champ-9 | $6.62 | – | waiting |
| LLM-v8 | $4.68 | – | waiting |
| GP | $4.51 | – | waiting |
| LLM-v7 | $2.82 | – | waiting |
| PPO-shaped | $2.29 | – | waiting |
| LLM-v6 | $2.23 | – | waiting |
| Tactics | $2.00 | – | waiting |
| LLM-v4 | $1.74 | – | waiting |
| PPO-tactician | $1.51 | – | waiting |
| PPO-exploiter | $1.47 | – | waiting |
| PPO-chance | $1.41 | – | waiting |
| champ-6 | $1.04 | – | waiting |
| champ-5 | $0.99 | – | waiting |
| Coach | $0.96 | – | waiting |
| Neuro-ES | $0.95 | – | waiting |
| champ-4 | $0.86 | – | waiting |
| LLM-v2 | $0.81 | – | waiting |
| PPO-roles | $0.71 | – | waiting |
| PPO-v11clone | $0.65 | – | waiting |
| champ-3 | $0.54 | – | waiting |
| PPO-champ-2 | $0.30 | – | waiting |
| LLM-v1 | $0.10 | – | waiting |
| champ-1 | $0.08 | – | waiting |
| champ-0 | $0.02 | – | waiting |
| champ-7 | $0.00 | – | waiting |
| champ-8 | $0.00 | – | waiting |

By method: the total compute each approach has used (all its brains), and its best brain this round.

| Method | Brains | Compute $ (total) | Best brain | Swiss pts/game | Goals/game vs champion |
| --- | ---: | ---: | --- | ---: | ---: |
| PPO | 9 | $44.93 | PPO-v2 | 2.50 | – |
| LLMBrain (written by Claude) | 8 | $25.03 | LLM-v11 | 1.79 | – |
| GP-v2 | 1 | $10.69 | waiting | – | – |
| GP-v3 | 1 | $7.12 | GP-v3 | 0.79 | – |
| GP | 1 | $4.51 | waiting | – | – |
| Tactics | 1 | $2.00 | waiting | – | – |
| champ-6 | 1 | $1.04 | waiting | – | – |
| champ-5 | 1 | $0.99 | waiting | – | – |
| Coach (mixes PPO roles) | 1 | $0.96 | waiting | – | – |
| Evolution strategies | 1 | $0.95 | waiting | – | – |
| champ-4 | 1 | $0.86 | waiting | – | – |
| champ-3 | 1 | $0.54 | waiting | – | – |
| champ-1 | 1 | $0.08 | waiting | – | – |
| champ-0 | 1 | $0.02 | waiting | – | – |
| champ-7 | 1 | $0.00 | waiting | – | – |
| champ-8 | 1 | $0.00 | waiting | – | – |

<!-- AUTO:COMPUTE END -->

**How we count compute.** Each contender is charged for its own lineage: everything spent on the versions it was built from. LLM-v11 carries the bills of every earlier version, failed attempts included, and GP-v3 carries GP and GP-v2, since it began as an exact translation of GP-v2's population. PPO-goals learned from zero, so the other PPO experiments count in the PPO family's total, not in its own. PPO-v2 is a hybrid: it started from the weights of a clone of v11 (so the clone's $0.65 is in its lineage), but its design (new inputs, a bigger network, two-tick decisions) was new. The fair question is how good each brain got for what went into it, so in the final stretch we gave every contender roughly the same budget, $20–25 each, with the lagging PPOs getting most of the cores.

## Round by round

<!-- AUTO:ROUNDS START -->

### Round 0: untrained

Two brains that have never trained: **untrained-1** and **untrained-2**, both random neural networks, moving on nothing but the random exploration noise every learner starts with. Any goals are accidents.

**Showcase: untrained-1 5 – 2 untrained-2** (untrained-1 wins). Video: [round-000.mp4](videos/league3/round-000.mp4)

untrained-1 beat untrained-2 5–2. Goals: 23' untrained-2, 32' untrained-1, 33' untrained-1, 54' untrained-1, 57' untrained-2, 74' untrained-1, 81' untrained-1. untrained-1 is rarely near the ball, is scattered all over the pitch and runs flat out almost all the time. untrained-2 is rarely near the ball, is scattered all over the pitch and runs flat out almost all the time. untrained-2 kept the ball in the opposition half 52% of the time; shots on goal 7–5.

### Round 1 (18:13)

**Showcase: PPO-shaped 7 – 7 GP** (a draw). Video: [round-001.mp4](videos/league3/round-001.mp4)

PPO-shaped and GP drew 7–7. Goals: 2' GP, 12' PPO-shaped, 14' GP, 17' GP, 20' PPO-shaped, 27' GP, 57' PPO-shaped, 73' PPO-shaped…. PPO-shaped is rarely near the ball, is scattered all over the pitch and runs flat out almost all the time. GP is rarely near the ball, moves as a tight pack, leaves its goal empty and runs flat out almost all the time. GP kept the ball in the opposition half 53% of the time; shots on goal 8–10. New: the most goals in a showcase so far (14); the most shots so far (18).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GP | 4 | 4 | 0 | 0 | 133.3 | 97.2 | +36.2 | **12** | 0.02 |
| 2 | PPO-shaped | 4 | 3 | 0 | 1 | 126.8 | 106.8 | +20.0 | **9** | 0.02 |
| 3 | PPO-roles | 4 | 2 | 0 | 2 | 132.2 | 123.8 | +8.3 | **6** | 0.02 |
| 4 | PPO-goals | 4 | 1 | 0 | 3 | 73.0 | 102.8 | -29.8 | **3** | 0.02 |
| 5 | Neuro-ES | 4 | 0 | 0 | 4 | 76.5 | 111.2 | -34.7 | **0** | 0.02 |

### Round 2 (18:35)

**Showcase: PPO-roles 8 – 9 PPO-shaped** (PPO-shaped wins). Video: [round-002.mp4](videos/league3/round-002.mp4)

PPO-shaped beat PPO-roles 9–8. Goals: 7' PPO-shaped, 14' PPO-shaped, 19' PPO-shaped, 21' PPO-roles, 24' PPO-shaped, 25' PPO-roles, 30' PPO-shaped, 34' PPO-shaped…. PPO-roles is rarely near the ball, is scattered all over the pitch, keeps a goalkeeper back 65% of the time and runs flat out almost all the time. PPO-shaped is rarely near the ball, is scattered all over the pitch, keeps a goalkeeper back 62% of the time and runs flat out almost all the time. PPO-shaped kept the ball in the opposition half 53% of the time; shots on goal 11–12. New: the first showcase with a real goalkeeper; the most goals in a showcase so far (17); the most shots so far (23).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-roles | 6 | 5 | 1 | 0 | 278.5 | 196.3 | +82.2 | **16** | 0.05 |
| 2 | PPO-shaped | 6 | 5 | 1 | 0 | 247.2 | 199.5 | +47.7 | **16** | 0.04 |
| 3 | GP | 6 | 4 | 0 | 2 | 457.8 | 281.7 | +176.2 | **12** | 0.05 |
| 4 | champ-0 | 6 | 3 | 0 | 3 | 189.3 | 242.2 | -52.8 | **9** | 0.02 |
| 5 | Coach | 6 | 2 | 0 | 4 | 180.0 | 218.2 | -38.2 | **6** | 0.01 |
| 6 | PPO-goals | 6 | 1 | 0 | 5 | 123.3 | 245.2 | -121.8 | **3** | 0.04 |

### Round 3 (19:01)

**Showcase: PPO-roles 8 – 9 PPO-shaped** (PPO-shaped wins). Video: [round-003.mp4](videos/league3/round-003.mp4)

PPO-shaped beat PPO-roles 9–8. Goals: 5' PPO-shaped, 7' PPO-roles, 9' PPO-shaped, 14' PPO-roles, 16' PPO-roles, 21' PPO-roles, 24' PPO-shaped, 27' PPO-roles…. PPO-roles is rarely near the ball, is scattered all over the pitch and runs flat out almost all the time. PPO-shaped is rarely near the ball, is scattered all over the pitch, keeps a goalkeeper back 69% of the time and runs flat out almost all the time. PPO-shaped kept the ball in the opposition half 52% of the time; shots on goal 10–12.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-roles | 6 | 5 | 1 | 0 | 256.0 | 206.3 | +49.7 | **16** | 0.07 |
| 2 | PPO-shaped | 6 | 3 | 3 | 0 | 252.7 | 193.8 | +58.8 | **12** | 0.07 |
| 3 | Coach | 6 | 4 | 0 | 2 | 324.7 | 288.0 | +36.7 | **12** | 0.02 |
| 4 | GP | 6 | 3 | 0 | 3 | 343.2 | 327.8 | +15.3 | **9** | 0.08 |
| 5 | champ-0 | 6 | 2 | 0 | 4 | 195.0 | 291.5 | -96.5 | **6** | 0.02 |
| 6 | PPO-goals | 6 | 1 | 1 | 4 | 181.3 | 203.2 | -21.8 | **4** | 0.07 |

### Round 4 (19:26)

**Showcase: Coach 10 – 13 GP** (GP wins). Video: [round-004.mp4](videos/league3/round-004.mp4)

GP beat Coach 13–10. Goals: 4' GP, 7' GP, 20' Coach, 25' Coach, 26' Coach, 28' GP, 30' GP, 35' Coach…. Coach is scattered all over the pitch and runs flat out almost all the time. GP is rarely near the ball, leaves its goal empty and runs flat out almost all the time. GP kept the ball in the opposition half 57% of the time; shots on goal 13–23. New: the most goals in a showcase so far (23); the most shots so far (36).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GP | 7 | 7 | 0 | 0 | 291.7 | 196.0 | +95.7 | **21** | 0.13 |
| 2 | Coach | 7 | 5 | 0 | 2 | 398.0 | 327.2 | +70.8 | **15** | 0.03 |
| 3 | PPO-shaped | 7 | 5 | 0 | 2 | 280.5 | 244.7 | +35.8 | **15** | 0.10 |
| 4 | PPO-roles | 7 | 4 | 1 | 2 | 279.2 | 256.8 | +22.3 | **13** | 0.12 |
| 5 | champ-1 | 7 | 2 | 0 | 5 | 380.0 | 382.8 | -2.8 | **6** | 0.08 |
| 6 | champ-0 | 7 | 2 | 0 | 5 | 227.5 | 342.2 | -114.7 | **6** | 0.02 |

### Round 5 (19:53)

**Showcase: Coach 11 – 10 GP** (Coach wins). Video: [round-005.mp4](videos/league3/round-005.mp4)

Coach beat GP 11–10. Goals: 3' GP, 5' GP, 11' Coach, 12' Coach, 14' GP, 15' GP, 17' GP, 23' GP…. Coach is rarely near the ball, is scattered all over the pitch, keeps a goalkeeper back 63% of the time and runs flat out almost all the time. GP is scattered all over the pitch, leaves its goal empty and runs flat out almost all the time. GP kept the ball in the opposition half 51% of the time; shots on goal 18–24. New: the most shots so far (42).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GP | 7 | 6 | 0 | 1 | 392.8 | 306.0 | +86.8 | **18** | 0.18 |
| 2 | Coach | 7 | 6 | 0 | 1 | 297.5 | 233.5 | +64.0 | **18** | 0.06 |
| 3 | PPO-shaped | 7 | 3 | 2 | 2 | 278.5 | 253.2 | +25.3 | **11** | 0.14 |
| 4 | Neuro-ES | 7 | 3 | 1 | 3 | 243.8 | 272.8 | -29.0 | **10** | 0.18 |
| 5 | PPO-roles | 7 | 3 | 0 | 4 | 271.8 | 246.0 | +25.8 | **9** | 0.16 |
| 6 | champ-1 | 7 | 2 | 1 | 4 | 363.2 | 361.7 | +1.5 | **7** | 0.08 |

### Round 6 (20:19)

**Showcase: PPO-roles 8 – 11 Neuro-ES** (Neuro-ES wins). Video: [round-006.mp4](videos/league3/round-006.mp4)

Neuro-ES beat PPO-roles 11–8. Goals: 9' Neuro-ES, 16' Neuro-ES, 27' PPO-roles, 31' Neuro-ES, 33' Neuro-ES, 34' PPO-roles, 35' Neuro-ES, 43' PPO-roles…. PPO-roles is rarely near the ball, is scattered all over the pitch, keeps a goalkeeper back 85% of the time and runs flat out almost all the time. Neuro-ES is rarely near the ball, leaves its goal empty and runs flat out almost all the time. Neuro-ES kept the ball in the opposition half 61% of the time; shots on goal 10–12.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-roles | 7 | 5 | 1 | 1 | 253.0 | 182.0 | +71.0 | **16** | 0.21 |
| 2 | PPO-shaped | 7 | 4 | 0 | 3 | 268.3 | 212.3 | +56.0 | **12** | 0.18 |
| 3 | Neuro-ES | 7 | 3 | 3 | 1 | 228.5 | 203.5 | +25.0 | **12** | 0.23 |
| 4 | champ-1 | 7 | 4 | 0 | 3 | 397.2 | 374.8 | +22.3 | **12** | 0.08 |
| 5 | PPO-goals | 7 | 3 | 2 | 2 | 164.7 | 167.0 | -2.3 | **11** | 0.18 |
| 6 | GP | 7 | 2 | 2 | 3 | 124.7 | 143.3 | -18.7 | **8** | 0.23 |

### Round 7 (20:45)

**Showcase: PPO-shaped 10 – 13 GP** (GP wins). Video: [round-007.mp4](videos/league3/round-007.mp4)

GP beat PPO-shaped 13–10. Goals: 1' PPO-shaped, 3' GP, 9' GP, 11' GP, 16' PPO-shaped, 17' GP, 24' PPO-shaped, 27' PPO-shaped…. PPO-shaped is rarely near the ball, is scattered all over the pitch and runs flat out almost all the time. GP is scattered all over the pitch, leaves its goal empty and runs flat out almost all the time. GP kept the ball in the opposition half 57% of the time; shots on goal 18–18.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GP | 7 | 6 | 0 | 1 | 370.0 | 319.7 | +50.3 | **18** | 0.28 |
| 2 | PPO-roles | 7 | 5 | 0 | 2 | 286.5 | 235.8 | +50.7 | **15** | 0.25 |
| 3 | PPO-shaped | 7 | 4 | 1 | 2 | 281.0 | 234.8 | +46.2 | **13** | 0.23 |
| 4 | Neuro-ES | 7 | 3 | 1 | 3 | 245.0 | 256.5 | -11.5 | **10** | 0.28 |
| 5 | champ-1 | 7 | 3 | 0 | 4 | 357.7 | 370.8 | -13.2 | **9** | 0.08 |
| 6 | Coach | 7 | 2 | 1 | 4 | 190.0 | 207.2 | -17.2 | **7** | 0.11 |

### Round 8 (21:12)

**Showcase: PPO-roles 11 – 11 Coach** (a draw). Video: [round-008.mp4](videos/league3/round-008.mp4)

PPO-roles and Coach drew 11–11. Goals: 10' Coach, 13' Coach, 19' PPO-roles, 23' PPO-roles, 34' Coach, 36' PPO-roles, 38' PPO-roles, 40' PPO-roles…. PPO-roles is rarely near the ball, is scattered all over the pitch, keeps a goalkeeper back 81% of the time and runs flat out almost all the time. Coach is rarely near the ball, is scattered all over the pitch and runs flat out almost all the time. Coach kept the ball in the opposition half 54% of the time; shots on goal 15–17.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-roles | 7 | 7 | 0 | 0 | 308.2 | 221.2 | +87.0 | **21** | 0.30 |
| 2 | PPO-shaped | 7 | 5 | 1 | 1 | 295.0 | 228.7 | +66.3 | **16** | 0.27 |
| 3 | Coach | 7 | 5 | 0 | 2 | 387.0 | 294.3 | +92.7 | **15** | 0.12 |
| 4 | Neuro-ES | 7 | 4 | 1 | 2 | 297.8 | 272.7 | +25.2 | **13** | 0.33 |
| 5 | champ-1 | 7 | 2 | 0 | 5 | 335.8 | 402.0 | -66.2 | **6** | 0.08 |
| 6 | GP | 7 | 1 | 1 | 5 | 350.2 | 384.8 | -34.7 | **4** | 0.33 |

### Round 9 (22:17)

**Showcase: Coach 11 – 11 PPO-champ-2** (a draw). Video: [round-009.mp4](videos/league3/round-009.mp4)

Coach and PPO-champ-2 drew 11–11. Goals: 13' PPO-champ-2, 15' Coach, 19' Coach, 22' PPO-champ-2, 25' PPO-champ-2, 27' Coach, 29' Coach, 30' Coach…. Coach is scattered all over the pitch and runs flat out almost all the time. PPO-champ-2 is rarely near the ball, is scattered all over the pitch, keeps a goalkeeper back 77% of the time and runs flat out almost all the time. Coach kept the ball in the opposition half 54% of the time; shots on goal 21–15.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Coach | 7 | 7 | 0 | 0 | 387.8 | 310.5 | +77.3 | **21** | 0.15 |
| 2 | PPO-champ-2 | 7 | 3 | 3 | 1 | 279.0 | 246.8 | +32.2 | **12** | 0.30 |
| 3 | PPO-shaped | 7 | 3 | 3 | 1 | 273.5 | 244.7 | +28.8 | **12** | 0.29 |
| 4 | GP | 7 | 3 | 1 | 3 | 346.7 | 361.8 | -15.2 | **10** | 0.37 |
| 5 | PPO-roles | 7 | 2 | 1 | 4 | 265.5 | 257.5 | +8.0 | **7** | 0.33 |
| 6 | Neuro-ES | 7 | 2 | 1 | 4 | 281.8 | 274.5 | +7.3 | **7** | 0.36 |

### Round 10 (22:45)

**Showcase: PPO-shaped 6 – 6 PPO-roles** (a draw). Video: [round-010.mp4](videos/league3/round-010.mp4)

PPO-shaped and PPO-roles drew 6–6. Goals: 2' PPO-shaped, 4' PPO-roles, 6' PPO-shaped, 28' PPO-roles, 30' PPO-shaped, 40' PPO-shaped, 51' PPO-roles, 59' PPO-shaped…. PPO-shaped is rarely near the ball, is scattered all over the pitch, keeps a goalkeeper back 84% of the time and runs flat out almost all the time. PPO-roles is rarely near the ball, is scattered all over the pitch and runs flat out almost all the time. PPO-shaped kept the ball in the opposition half 54% of the time; shots on goal 10–9.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-shaped | 7 | 6 | 1 | 0 | 285.0 | 225.0 | +60.0 | **19** | 0.34 |
| 2 | PPO-roles | 7 | 3 | 2 | 2 | 272.8 | 239.2 | +33.7 | **11** | 0.38 |
| 3 | Neuro-ES | 7 | 3 | 2 | 2 | 272.5 | 272.0 | +0.5 | **11** | 0.42 |
| 4 | PPO-champ-2 | 7 | 3 | 1 | 3 | 272.2 | 244.3 | +27.8 | **10** | 0.30 |
| 5 | GP | 7 | 3 | 1 | 3 | 342.8 | 364.3 | -21.5 | **10** | 0.42 |
| 6 | Coach | 7 | 3 | 0 | 4 | 342.5 | 317.8 | +24.7 | **9** | 0.17 |

### Round 11 (23:15)

**Showcase: GP 7 – 3 PPO-shaped** (GP wins). Video: [round-011.mp4](videos/league3/round-011.mp4)

GP beat PPO-shaped 7–3. Goals: 12' PPO-shaped, 15' GP, 18' GP, 24' PPO-shaped, 35' GP, 40' GP, 53' GP, 57' PPO-shaped…. GP is rarely near the ball, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. PPO-shaped is rarely near the ball, is scattered all over the pitch, keeps a goalkeeper back 91% of the time and runs flat out almost all the time. PPO-shaped kept the ball in the opposition half 55% of the time; shots on goal 15–22.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GP | 9 | 9 | 0 | 0 | 475.7 | 200.8 | +274.8 | **27** | 0.48 |
| 2 | PPO-goals | 9 | 6 | 1 | 2 | 340.5 | 291.2 | +49.3 | **19** | 0.38 |
| 3 | PPO-roles | 9 | 6 | 1 | 2 | 354.7 | 307.8 | +46.8 | **19** | 0.43 |
| 4 | PPO-shaped | 9 | 6 | 0 | 3 | 345.0 | 288.0 | +57.0 | **18** | 0.38 |
| 5 | PPO-champ-2 | 9 | 5 | 0 | 4 | 340.2 | 302.8 | +37.3 | **15** | 0.30 |
| 6 | Coach | 9 | 4 | 1 | 4 | 453.2 | 417.0 | +36.2 | **13** | 0.19 |

### Round 12 (23:44)

**Showcase: GP 9 – 9 Coach** (a draw). Video: [round-012.mp4](videos/league3/round-012.mp4)

GP and Coach drew 9–9. Goals: 1' GP, 18' GP, 21' Coach, 23' Coach, 26' GP, 38' Coach, 42' GP, 45' Coach…. GP is rarely near the ball, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach is rarely near the ball, is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP kept the ball in the opposition half 53% of the time; shots on goal 20–24. New: the most shots so far (44).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GP | 9 | 9 | 0 | 0 | 447.8 | 199.2 | +248.7 | **27** | 0.54 |
| 2 | Coach | 9 | 8 | 0 | 1 | 409.7 | 208.8 | +200.8 | **24** | 0.19 |
| 3 | LLM-v2 | 9 | 7 | 0 | 2 | 363.8 | 273.5 | +90.3 | **21** | 0.81 |
| 4 | PPO-roles | 9 | 5 | 1 | 3 | 300.2 | 296.5 | +3.7 | **16** | 0.48 |
| 5 | PPO-shaped | 9 | 4 | 0 | 5 | 286.7 | 292.7 | -6.0 | **12** | 0.43 |
| 6 | Neuro-ES | 9 | 3 | 2 | 4 | 300.7 | 331.8 | -31.2 | **11** | 0.54 |

### Round 13 (00:10)

**Showcase: GP 7 – 7 champ-3** (a draw). Video: [round-013.mp4](videos/league3/round-013.mp4)

GP and champ-3 drew 7–7. Goals: 4' GP, 7' GP, 10' GP, 14' champ-3, 16' GP, 20' champ-3, 32' champ-3, 40' champ-3…. GP is rarely near the ball, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. champ-3 is rarely near the ball, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP kept the ball in the opposition half 51% of the time; shots on goal 17–25.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | champ-3 | 9 | 8 | 1 | 0 | 380.2 | 215.8 | +164.3 | **25** | 0.54 |
| 2 | GP | 9 | 7 | 2 | 0 | 378.3 | 236.7 | +141.7 | **23** | 0.60 |
| 3 | Coach | 9 | 7 | 1 | 1 | 366.7 | 218.8 | +147.8 | **22** | 0.20 |
| 4 | LLM-v2 | 9 | 6 | 0 | 3 | 326.2 | 288.0 | +38.2 | **18** | 0.81 |
| 5 | PPO-goals | 9 | 5 | 0 | 4 | 287.8 | 273.2 | +14.7 | **15** | 0.47 |
| 6 | Neuro-ES | 9 | 3 | 1 | 5 | 278.0 | 351.5 | -73.5 | **10** | 0.59 |

### Round 14 (00:36)

**Showcase: champ-3 2 – 6 LLM-v4** (LLM-v4 wins). Video: [round-014.mp4](videos/league3/round-014.mp4)

LLM-v4 beat champ-3 6–2. Goals: 10' LLM-v4, 13' champ-3, 26' LLM-v4, 47' LLM-v4, 60' LLM-v4, 82' champ-3, 85' LLM-v4, 87' LLM-v4. champ-3 is rarely near the ball and keeps a goalkeeper back 100% of the time. LLM-v4 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. champ-3 kept the ball in the opposition half 53% of the time; shots on goal 33–23. New: the most shots so far (56).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v4 | 9 | 9 | 0 | 0 | 325.3 | 189.7 | +135.7 | **27** | 1.74 |
| 2 | champ-3 | 9 | 8 | 0 | 1 | 372.0 | 216.8 | +155.2 | **24** | 0.54 |
| 3 | Coach | 9 | 7 | 0 | 2 | 363.0 | 237.2 | +125.8 | **21** | 0.22 |
| 4 | GP | 9 | 6 | 0 | 3 | 345.0 | 231.5 | +113.5 | **18** | 0.65 |
| 5 | PPO-goals | 9 | 5 | 0 | 4 | 287.8 | 275.7 | +12.2 | **15** | 0.51 |
| 6 | Neuro-ES | 9 | 4 | 0 | 5 | 286.7 | 345.0 | -58.3 | **12** | 0.64 |

### Round 15 (01:01)

**Showcase: GP 2 – 4 LLM-v4** (LLM-v4 wins). Video: [round-015.mp4](videos/league3/round-015.mp4)

LLM-v4 beat GP 4–2. Goals: 21' LLM-v4, 22' GP, 29' GP, 42' LLM-v4, 58' LLM-v4, 63' LLM-v4. GP is rarely near the ball and keeps a goalkeeper back 100% of the time. LLM-v4 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v4 kept the ball in the opposition half 51% of the time; shots on goal 41–18. New: the most shots so far (59).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v4 | 9 | 9 | 0 | 0 | 327.5 | 186.0 | +141.5 | **27** | 1.74 |
| 2 | GP | 9 | 8 | 0 | 1 | 365.5 | 231.8 | +133.7 | **24** | 0.70 |
| 3 | Coach | 9 | 7 | 0 | 2 | 346.8 | 233.8 | +113.0 | **21** | 0.24 |
| 4 | champ-3 | 9 | 6 | 0 | 3 | 351.3 | 214.0 | +137.3 | **18** | 0.54 |
| 5 | PPO-goals | 9 | 4 | 1 | 4 | 294.7 | 289.3 | +5.3 | **13** | 0.55 |
| 6 | Neuro-ES | 9 | 3 | 2 | 4 | 285.5 | 350.2 | -64.7 | **11** | 0.70 |

### Round 16 (01:29)

**Showcase: GP 6 – 4 champ-3** (GP wins). Video: [round-016.mp4](videos/league3/round-016.mp4)

GP beat champ-3 6–4. Goals: 7' GP, 11' GP, 18' GP, 66' GP, 70' champ-3, 72' champ-3, 74' GP, 79' champ-3…. GP is rarely near the ball, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. champ-3 is rarely near the ball and keeps a goalkeeper back 100% of the time. champ-3 kept the ball in the opposition half 58% of the time; shots on goal 13–18.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v4 | 9 | 8 | 1 | 0 | 307.0 | 198.3 | +108.7 | **25** | 1.74 |
| 2 | GP | 9 | 8 | 0 | 1 | 265.3 | 123.7 | +141.7 | **24** | 0.75 |
| 3 | Coach | 9 | 6 | 2 | 1 | 254.8 | 135.8 | +119.0 | **20** | 0.26 |
| 4 | champ-3 | 9 | 6 | 1 | 2 | 365.0 | 227.5 | +137.5 | **19** | 0.54 |
| 5 | PPO-goals | 9 | 5 | 0 | 4 | 266.8 | 257.2 | +9.7 | **15** | 0.58 |
| 6 | PPO-shaped | 9 | 3 | 1 | 5 | 245.5 | 273.3 | -27.8 | **10** | 0.59 |

### Round 17 (01:56)

**Showcase: Coach 7 – 4 LLM-v4** (Coach wins). Video: [round-017.mp4](videos/league3/round-017.mp4)

Coach beat LLM-v4 7–4. Goals: 23' LLM-v4, 24' Coach, 25' Coach, 27' Coach, 33' Coach, 37' Coach, 44' LLM-v4, 69' Coach…. Coach is rarely near the ball, is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v4 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach kept the ball in the opposition half 54% of the time; shots on goal 26–20.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Coach | 9 | 9 | 0 | 0 | 227.8 | 92.8 | +135.0 | **27** | 0.28 |
| 2 | LLM-v4 | 9 | 8 | 0 | 1 | 296.3 | 185.2 | +111.2 | **24** | 1.74 |
| 3 | champ-3 | 9 | 7 | 0 | 2 | 350.2 | 210.2 | +140.0 | **21** | 0.54 |
| 4 | GP | 9 | 6 | 0 | 3 | 362.5 | 217.0 | +145.5 | **18** | 0.80 |
| 5 | PPO-goals | 9 | 5 | 0 | 4 | 263.7 | 251.7 | +12.0 | **15** | 0.63 |
| 6 | PPO-shaped | 9 | 4 | 0 | 5 | 275.8 | 277.7 | -1.8 | **12** | 0.63 |

### Round 18 (02:20)

**Showcase: GP 3 – 1 LLM-v6** (GP wins). Video: [round-018.mp4](videos/league3/round-018.mp4)

GP beat LLM-v6 3–1. Goals: 5' LLM-v6, 9' GP, 42' GP, 44' GP. GP keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v6 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v6 kept the ball in the opposition half 56% of the time; shots on goal 20–26.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GP | 7 | 7 | 0 | 0 | 257.3 | 70.2 | +187.2 | **21** | 0.86 |
| 2 | LLM-v6 | 7 | 6 | 0 | 1 | 228.3 | 104.5 | +123.8 | **18** | 2.23 |
| 3 | Coach | 7 | 5 | 0 | 2 | 167.5 | 128.0 | +39.5 | **15** | 0.31 |
| 4 | champ-3 | 7 | 4 | 0 | 3 | 221.0 | 173.2 | +47.8 | **12** | 0.54 |
| 5 | PPO-goals | 7 | 3 | 0 | 4 | 152.3 | 206.0 | -53.7 | **9** | 0.67 |
| 6 | PPO-shaped | 7 | 1 | 1 | 5 | 158.7 | 248.0 | -89.3 | **4** | 0.68 |

### Round 19 (02:43)

**Showcase: GP 3 – 1 champ-4** (GP wins). Video: [round-019.mp4](videos/league3/round-019.mp4)

GP beat champ-4 3–1. Goals: 14' GP, 26' champ-4, 71' GP, 74' GP. GP is rarely near the ball, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. champ-4 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP kept the ball in the opposition half 55% of the time; shots on goal 28–17.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GP | 7 | 7 | 0 | 0 | 197.5 | 38.3 | +159.2 | **21** | 0.91 |
| 2 | champ-4 | 7 | 6 | 0 | 1 | 206.2 | 67.3 | +138.8 | **18** | 0.86 |
| 3 | Coach | 7 | 5 | 0 | 2 | 193.7 | 76.0 | +117.7 | **15** | 0.32 |
| 4 | LLM-v6 | 7 | 4 | 0 | 3 | 172.2 | 108.3 | +63.8 | **12** | 2.23 |
| 5 | champ-3 | 7 | 3 | 0 | 4 | 166.3 | 184.3 | -18.0 | **9** | 0.54 |
| 6 | PPO-goals | 7 | 2 | 0 | 5 | 128.8 | 224.7 | -95.8 | **6** | 0.71 |

### Round 20 (03:04)

**Showcase: Coach 1 – 1 GP** (a draw). Video: [round-020.mp4](videos/league3/round-020.mp4)

Coach and GP drew 1–1. Goals: 16' Coach, 23' GP. Coach keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP kept the ball in the opposition half 53% of the time; shots on goal 18–45. New: the most shots so far (63).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GP | 7 | 7 | 0 | 0 | 187.3 | 39.3 | +148.0 | **21** | 0.95 |
| 2 | Coach | 7 | 6 | 0 | 1 | 179.0 | 40.7 | +138.3 | **18** | 0.33 |
| 3 | champ-4 | 7 | 4 | 1 | 2 | 185.2 | 65.3 | +119.8 | **13** | 0.86 |
| 4 | LLM-v7 | 7 | 4 | 1 | 2 | 164.0 | 73.5 | +90.5 | **13** | 2.82 |
| 5 | champ-3 | 7 | 3 | 0 | 4 | 148.0 | 185.7 | -37.7 | **9** | 0.54 |
| 6 | PPO-goals | 7 | 2 | 0 | 5 | 108.7 | 215.2 | -106.5 | **6** | 0.74 |

### Round 21 (03:26)

**Showcase: Coach 0 – 0 GP** (a draw). Video: [round-021.mp4](videos/league3/round-021.mp4)

Coach and GP drew 0–0. Coach is rarely near the ball, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP is rarely near the ball, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach kept the ball in the opposition half 50% of the time; shots on goal 3–2.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Coach | 7 | 7 | 0 | 0 | 182.3 | 34.8 | +147.5 | **21** | 0.35 |
| 2 | GP | 7 | 6 | 0 | 1 | 176.0 | 32.3 | +143.7 | **18** | 0.99 |
| 3 | champ-4 | 7 | 5 | 0 | 2 | 166.0 | 65.3 | +100.7 | **15** | 0.86 |
| 4 | LLM-v8 | 7 | 4 | 0 | 3 | 154.7 | 64.7 | +90.0 | **12** | 4.68 |
| 5 | champ-3 | 7 | 3 | 0 | 4 | 138.8 | 179.0 | -40.2 | **9** | 0.54 |
| 6 | PPO-goals | 7 | 2 | 0 | 5 | 113.8 | 185.5 | -71.7 | **6** | 0.77 |

### Round 22 (03:48)

**Showcase: GP 0 – 0 Coach** (a draw). Video: [round-022.mp4](videos/league3/round-022.mp4)

GP and Coach drew 0–0. GP keeps a goalkeeper back 96% of the time and runs flat out almost all the time. Coach keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP kept the ball in the opposition half 52% of the time; shots on goal 14–27.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GP | 7 | 6 | 1 | 0 | 139.2 | 13.5 | +125.7 | **19** | 1.04 |
| 2 | Coach | 7 | 6 | 1 | 0 | 141.7 | 18.8 | +122.8 | **19** | 0.36 |
| 3 | champ-5 | 7 | 5 | 0 | 2 | 130.0 | 39.0 | +91.0 | **15** | 0.99 |
| 4 | LLM-v8 | 7 | 4 | 0 | 3 | 114.2 | 66.3 | +47.8 | **12** | 4.68 |
| 5 | champ-4 | 7 | 3 | 0 | 4 | 119.5 | 76.3 | +43.2 | **9** | 0.86 |
| 6 | champ-3 | 7 | 2 | 0 | 5 | 96.8 | 183.8 | -87.0 | **6** | 0.54 |

### Round 23 (04:09)

**Showcase: Coach 0 – 0 GP** (a draw). Video: [round-023.mp4](videos/league3/round-023.mp4)

Coach and GP drew 0–0. Coach keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP keeps a goalkeeper back 96% of the time and runs flat out almost all the time. Coach kept the ball in the opposition half 54% of the time; shots on goal 20–23.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | champ-6 | 7 | 5 | 2 | 0 | 107.8 | 17.7 | +90.2 | **17** | 1.04 |
| 2 | Coach | 7 | 5 | 2 | 0 | 104.3 | 16.3 | +88.0 | **17** | 0.38 |
| 3 | GP | 7 | 4 | 3 | 0 | 100.0 | 18.5 | +81.5 | **15** | 1.08 |
| 4 | champ-5 | 7 | 4 | 1 | 2 | 99.7 | 34.0 | +65.7 | **13** | 0.99 |
| 5 | LLM-v8 | 7 | 2 | 1 | 4 | 90.0 | 64.8 | +25.2 | **7** | 4.68 |
| 6 | champ-4 | 7 | 2 | 1 | 4 | 90.2 | 73.7 | +16.5 | **7** | 0.86 |

### Round 24 (04:29)

**Showcase: Coach 0 – 0 champ-6** (a draw). Video: [round-024.mp4](videos/league3/round-024.mp4)

Coach and champ-6 drew 0–0. Coach keeps a goalkeeper back 100% of the time and runs flat out almost all the time. champ-6 keeps a goalkeeper back 96% of the time and runs flat out almost all the time. champ-6 kept the ball in the opposition half 54% of the time; shots on goal 37–30. New: the most shots so far (67).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Coach | 7 | 6 | 1 | 0 | 105.2 | 18.2 | +87.0 | **19** | 0.39 |
| 2 | champ-6 | 7 | 6 | 1 | 0 | 105.8 | 20.5 | +85.3 | **19** | 1.04 |
| 3 | GP | 7 | 4 | 1 | 2 | 99.5 | 35.3 | +64.2 | **13** | 1.12 |
| 4 | champ-5 | 7 | 4 | 1 | 2 | 97.3 | 39.2 | +58.2 | **13** | 0.99 |
| 5 | LLM-v8 | 7 | 2 | 1 | 4 | 91.0 | 61.5 | +29.5 | **7** | 4.68 |
| 6 | champ-4 | 7 | 2 | 1 | 4 | 89.8 | 76.5 | +13.3 | **7** | 0.86 |

### Round 25 (04:49)

**Showcase: Coach 0 – 0 GP** (a draw). Video: [round-025.mp4](videos/league3/round-025.mp4)

Coach and GP drew 0–0. Coach is rarely near the ball, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP keeps a goalkeeper back 95% of the time and runs flat out almost all the time. Coach kept the ball in the opposition half 50% of the time; shots on goal 20–19.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Coach | 7 | 6 | 1 | 0 | 92.7 | 15.8 | +76.8 | **19** | 0.40 |
| 2 | GP | 7 | 5 | 1 | 1 | 100.5 | 17.5 | +83.0 | **16** | 1.16 |
| 3 | champ-6 | 7 | 4 | 3 | 0 | 96.7 | 21.2 | +75.5 | **15** | 1.04 |
| 4 | champ-5 | 7 | 4 | 1 | 2 | 90.8 | 35.2 | +55.7 | **13** | 0.99 |
| 5 | LLM-v8 | 7 | 2 | 1 | 4 | 87.0 | 66.2 | +20.8 | **7** | 4.68 |
| 6 | champ-4 | 7 | 2 | 1 | 4 | 82.7 | 82.3 | +0.3 | **7** | 0.86 |

### Round 26 (05:09)

**Showcase: Coach 1 – 1 champ-6** (a draw). Video: [round-026.mp4](videos/league3/round-026.mp4)

Coach and champ-6 drew 1–1. Goals: 14' Coach, 81' champ-6. Coach keeps a goalkeeper back 100% of the time and runs flat out almost all the time. champ-6 keeps a goalkeeper back 96% of the time and runs flat out almost all the time. Coach kept the ball in the opposition half 53% of the time; shots on goal 43–15.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Coach | 7 | 6 | 1 | 0 | 93.5 | 17.2 | +76.3 | **19** | 0.41 |
| 2 | champ-6 | 7 | 5 | 2 | 0 | 93.3 | 18.3 | +75.0 | **17** | 1.04 |
| 3 | GP | 7 | 5 | 1 | 1 | 98.5 | 19.3 | +79.2 | **16** | 1.20 |
| 4 | champ-5 | 7 | 4 | 0 | 3 | 88.3 | 40.2 | +48.2 | **12** | 0.99 |
| 5 | champ-4 | 7 | 3 | 0 | 4 | 83.3 | 81.0 | +2.3 | **9** | 0.86 |
| 6 | LLM-v8 | 7 | 2 | 0 | 5 | 79.7 | 62.2 | +17.5 | **6** | 4.68 |

### Round 27 (05:29)

**Showcase: Coach 0 – 0 GP** (a draw). Video: [round-027.mp4](videos/league3/round-027.mp4)

Coach and GP drew 0–0. Coach keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP keeps a goalkeeper back 96% of the time and runs flat out almost all the time. GP kept the ball in the opposition half 60% of the time; shots on goal 19–22.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GP | 7 | 7 | 0 | 0 | 89.2 | 15.7 | +73.5 | **21** | 1.24 |
| 2 | champ-6 | 7 | 5 | 1 | 1 | 81.8 | 17.3 | +64.5 | **16** | 1.04 |
| 3 | Coach | 7 | 5 | 1 | 1 | 83.0 | 19.2 | +63.8 | **16** | 0.43 |
| 4 | champ-5 | 7 | 4 | 0 | 3 | 74.3 | 40.0 | +34.3 | **12** | 0.99 |
| 5 | LLM-v8 | 7 | 2 | 1 | 4 | 71.3 | 62.0 | +9.3 | **7** | 4.68 |
| 6 | champ-4 | 7 | 2 | 1 | 4 | 74.7 | 74.8 | -0.2 | **7** | 0.86 |

### Round 28 (05:50)

**Showcase: GP 0 – 0 champ-6** (a draw). Video: [round-028.mp4](videos/league3/round-028.mp4)

GP and champ-6 drew 0–0. GP keeps a goalkeeper back 94% of the time and runs flat out almost all the time. champ-6 keeps a goalkeeper back 97% of the time and runs flat out almost all the time. GP kept the ball in the opposition half 55% of the time; shots on goal 40–31. New: the most shots so far (71).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GP | 7 | 6 | 1 | 0 | 92.7 | 17.2 | +75.5 | **19** | 1.28 |
| 2 | champ-6 | 7 | 6 | 1 | 0 | 89.7 | 18.8 | +70.8 | **19** | 1.04 |
| 3 | Coach | 7 | 4 | 1 | 2 | 78.3 | 26.5 | +51.8 | **13** | 0.45 |
| 4 | champ-5 | 7 | 4 | 1 | 2 | 77.2 | 36.3 | +40.8 | **13** | 0.99 |
| 5 | LLM-v8 | 7 | 3 | 0 | 4 | 71.8 | 57.7 | +14.2 | **9** | 4.68 |
| 6 | champ-4 | 7 | 2 | 0 | 5 | 78.7 | 75.0 | +3.7 | **6** | 0.86 |

### Round 29 (06:10)

**Showcase: champ-6 0 – 0 GP** (a draw). Video: [round-029.mp4](videos/league3/round-029.mp4)

champ-6 and GP drew 0–0. champ-6 keeps a goalkeeper back 97% of the time and runs flat out almost all the time. GP keeps a goalkeeper back 98% of the time and runs flat out almost all the time. champ-6 kept the ball in the opposition half 52% of the time; shots on goal 23–21.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | champ-6 | 7 | 6 | 1 | 0 | 79.3 | 18.0 | +61.3 | **19** | 1.04 |
| 2 | GP | 7 | 5 | 2 | 0 | 84.0 | 19.5 | +64.5 | **17** | 1.32 |
| 3 | Coach | 7 | 5 | 1 | 1 | 72.3 | 25.5 | +46.8 | **16** | 0.46 |
| 4 | champ-5 | 7 | 4 | 0 | 3 | 82.2 | 41.0 | +41.2 | **12** | 0.99 |
| 5 | LLM-v8 | 7 | 2 | 1 | 4 | 73.8 | 64.7 | +9.2 | **7** | 4.68 |
| 6 | champ-4 | 7 | 2 | 1 | 4 | 71.7 | 70.7 | +1.0 | **7** | 0.86 |

### Round 30 (06:47)

**Showcase: Tactics 1 – 1 GP** (a draw). Video: [round-030.mp4](videos/league3/round-030.mp4)

Tactics and GP drew 1–1. Goals: 17' GP, 81' Tactics. Tactics is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP moves as a tight pack, keeps a goalkeeper back 96% of the time and runs flat out almost all the time. Tactics kept the ball in the opposition half 83% of the time; shots on goal 56–11.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Tactics | 7 | 7 | 0 | 0 | 244.0 | 21.3 | +222.7 | **21** | 2.00 |
| 2 | GP | 7 | 5 | 1 | 1 | 74.8 | 24.5 | +50.3 | **16** | 1.36 |
| 3 | champ-6 | 7 | 4 | 1 | 2 | 73.2 | 23.7 | +49.5 | **13** | 1.04 |
| 4 | Coach | 7 | 3 | 3 | 1 | 65.8 | 44.2 | +21.7 | **12** | 0.48 |
| 5 | champ-5 | 7 | 3 | 1 | 3 | 71.8 | 48.3 | +23.5 | **10** | 0.99 |
| 6 | LLM-v8 | 7 | 2 | 0 | 5 | 64.7 | 78.0 | -13.3 | **6** | 4.68 |

### Round 31 (07:46)

**Showcase: PPO-tactician 4 – 4 Tactics** (a draw). Video: [round-031.mp4](videos/league3/round-031.mp4)

PPO-tactician and Tactics drew 4–4. Goals: 18' PPO-tactician, 27' PPO-tactician, 30' Tactics, 61' Tactics, 75' Tactics, 79' PPO-tactician, 81' PPO-tactician, 88' Tactics. PPO-tactician is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Tactics keeps a goalkeeper back 100% of the time and runs flat out almost all the time. PPO-tactician kept the ball in the opposition half 70% of the time; shots on goal 48–24. New: the most shots so far (72).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-tactician | 11 | 11 | 0 | 0 | 366.5 | 47.7 | +318.8 | **33** | 0.03 |
| 2 | Tactics | 11 | 10 | 0 | 1 | 375.8 | 48.3 | +327.5 | **30** | 2.00 |
| 3 | Coach | 11 | 7 | 2 | 2 | 173.0 | 53.2 | +119.8 | **23** | 0.51 |
| 4 | GP-v2 | 11 | 7 | 2 | 2 | 108.7 | 35.2 | +73.5 | **23** | 0.03 |
| 5 | GP | 11 | 7 | 2 | 2 | 105.3 | 37.0 | +68.3 | **23** | 1.39 |
| 6 | champ-6 | 11 | 6 | 0 | 5 | 108.7 | 39.2 | +69.5 | **18** | 1.04 |

### Round 32 (08:12)

**Showcase: Tactics 2 – 2 PPO-tactician** (a draw). Video: [round-032.mp4](videos/league3/round-032.mp4)

Tactics and PPO-tactician drew 2–2. Goals: 5' PPO-tactician, 16' Tactics, 21' Tactics, 49' PPO-tactician. Tactics keeps a goalkeeper back 100% of the time and runs flat out almost all the time. PPO-tactician keeps a goalkeeper back 100% of the time and runs flat out almost all the time. PPO-tactician kept the ball in the opposition half 59% of the time; shots on goal 30–50. New: the most shots so far (80).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Tactics | 4 | 4 | 0 | 0 | 35.8 | 19.7 | +16.2 | **12** | 2.00 |
| 2 | PPO-tactician | 4 | 3 | 0 | 1 | 97.5 | 26.5 | +71.0 | **9** | 0.10 |
| 3 | Coach | 4 | 2 | 1 | 1 | 51.8 | 24.5 | +27.3 | **7** | 0.57 |
| 4 | GP | 4 | 2 | 1 | 1 | 23.8 | 12.8 | +11.0 | **7** | 1.46 |
| 5 | champ-6 | 4 | 2 | 0 | 2 | 21.0 | 12.3 | +8.7 | **6** | 1.04 |
| 6 | GP-v2 | 4 | 2 | 0 | 2 | 22.3 | 18.5 | +3.8 | **6** | 0.10 |

### Round 33 (08:40)

**Showcase: PPO-tactician 3 – 2 Tactics** (PPO-tactician wins). Video: [round-033.mp4](videos/league3/round-033.mp4)

PPO-tactician beat Tactics 3–2. Goals: 37' Tactics, 52' PPO-tactician, 55' PPO-tactician, 75' PPO-tactician, 79' Tactics. PPO-tactician keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Tactics is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Tactics kept the ball in the opposition half 70% of the time; shots on goal 28–53. New: the most shots so far (81).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-tactician | 4 | 3 | 1 | 0 | 47.5 | 23.0 | +24.5 | **10** | 0.13 |
| 2 | Tactics | 4 | 3 | 1 | 0 | 46.8 | 26.0 | +20.8 | **10** | 2.00 |
| 3 | Coach | 4 | 2 | 1 | 1 | 59.7 | 23.5 | +36.2 | **7** | 0.60 |
| 4 | GP-v2 | 4 | 2 | 1 | 1 | 28.2 | 15.0 | +13.2 | **7** | 0.13 |
| 5 | GP | 4 | 2 | 1 | 1 | 24.5 | 13.5 | +11.0 | **7** | 1.49 |
| 6 | LLM-v8 | 4 | 2 | 0 | 2 | 43.5 | 50.2 | -6.7 | **6** | 4.68 |

### Round 34 (09:10)

**Showcase: LLM-v11 10 – 1 PPO-tactician** (LLM-v11 wins). Video: [round-034.mp4](videos/league3/round-034.mp4)

LLM-v11 beat PPO-tactician 10–1. Goals: 3' LLM-v11, 7' LLM-v11, 13' LLM-v11, 24' LLM-v11, 29' LLM-v11, 44' LLM-v11, 51' LLM-v11, 56' PPO-tactician…. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. PPO-tactician keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 81% of the time; shots on goal 157–11. New: the most shots so far (168).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 4 | 0 | 0 | 232.2 | 2.2 | +230.0 | **12** | 22.30 |
| 2 | PPO-tactician | 4 | 3 | 0 | 1 | 69.7 | 45.7 | +24.0 | **9** | 0.17 |
| 3 | Tactics | 4 | 3 | 0 | 1 | 35.7 | 38.2 | -2.5 | **9** | 2.00 |
| 4 | Coach | 4 | 2 | 1 | 1 | 59.2 | 52.8 | +6.3 | **7** | 0.64 |
| 5 | champ-6 | 4 | 2 | 0 | 2 | 29.2 | 13.8 | +15.3 | **6** | 1.04 |
| 6 | GP | 4 | 2 | 0 | 2 | 20.8 | 13.8 | +7.0 | **6** | 1.53 |

### Round 35 (10:01)

**Showcase: GP-v2 0 – 13 LLM-v11** (LLM-v11 wins). Video: [round-035.mp4](videos/league3/round-035.mp4)

LLM-v11 beat GP-v2 13–0. Goals: 12' LLM-v11, 19' LLM-v11, 22' LLM-v11, 31' LLM-v11, 34' LLM-v11, 48' LLM-v11, 54' LLM-v11, 65' LLM-v11…. GP-v2 moves as a tight pack, keeps a goalkeeper back 93% of the time and runs flat out almost all the time. LLM-v11 is rarely near the ball, is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 87% of the time; shots on goal 3–130.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 4 | 0 | 0 | 165.0 | 5.8 | +159.2 | **12** | 22.30 |
| 2 | Coach | 4 | 3 | 0 | 1 | 120.3 | 52.5 | +67.8 | **9** | 0.70 |
| 3 | GP-v2 | 4 | 3 | 0 | 1 | 36.5 | 13.7 | +22.8 | **9** | 0.23 |
| 4 | Tactics | 4 | 2 | 0 | 2 | 69.7 | 42.3 | +27.3 | **6** | 2.00 |
| 5 | GP | 4 | 2 | 0 | 2 | 34.3 | 20.7 | +13.7 | **6** | 1.59 |
| 6 | LLM-v8 | 4 | 2 | 0 | 2 | 39.3 | 53.5 | -14.2 | **6** | 4.68 |

### Round 36 (10:48)

**Showcase: LLM-v11 4 – 3 Coach** (LLM-v11 wins). Video: [round-036.mp4](videos/league3/round-036.mp4)

LLM-v11 beat Coach 4–3. Goals: 20' LLM-v11, 24' Coach, 57' LLM-v11, 63' LLM-v11, 75' Coach, 78' LLM-v11, 90' Coach. LLM-v11 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach kept the ball in the opposition half 58% of the time; shots on goal 67–128. New: the most shots so far (195).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 4 | 0 | 0 | 145.0 | 11.5 | +133.5 | **12** | 22.30 |
| 2 | Coach | 4 | 3 | 0 | 1 | 188.0 | 20.0 | +168.0 | **9** | 0.72 |
| 3 | Tactics | 4 | 2 | 1 | 1 | 94.3 | 45.5 | +48.8 | **7** | 2.00 |
| 4 | PPO-tactician | 4 | 2 | 1 | 1 | 65.2 | 48.3 | +16.8 | **7** | 0.26 |
| 5 | GP-v2 | 4 | 2 | 1 | 1 | 26.7 | 66.7 | -40.0 | **7** | 0.33 |
| 6 | PPO-shaped | 4 | 2 | 0 | 2 | 24.2 | 30.0 | -5.8 | **6** | 1.29 |

### Round 37 (11:36)

**Showcase: LLM-v11 5 – 3 Coach** (LLM-v11 wins). Video: [round-037.mp4](videos/league3/round-037.mp4)

LLM-v11 beat Coach 5–3. Goals: 12' LLM-v11, 18' Coach, 20' LLM-v11, 23' Coach, 29' Coach, 46' LLM-v11, 58' LLM-v11, 73' LLM-v11. LLM-v11 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach kept the ball in the opposition half 54% of the time; shots on goal 59–91.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 4 | 0 | 0 | 167.7 | 9.0 | +158.7 | **12** | 22.30 |
| 2 | Coach | 4 | 3 | 0 | 1 | 143.0 | 20.7 | +122.3 | **9** | 0.75 |
| 3 | Tactics | 4 | 3 | 0 | 1 | 88.3 | 20.7 | +67.7 | **9** | 2.00 |
| 4 | champ-6 | 4 | 3 | 0 | 1 | 28.3 | 74.7 | -46.3 | **9** | 1.04 |
| 5 | GP | 4 | 2 | 0 | 2 | 33.3 | 14.3 | +19.0 | **6** | 1.79 |
| 6 | PPO-tactician | 4 | 2 | 0 | 2 | 75.2 | 78.2 | -3.0 | **6** | 0.32 |

### Round 38 (12:24)

**Showcase: Coach 2 – 2 LLM-v11** (a draw). Video: [round-038.mp4](videos/league3/round-038.mp4)

Coach and LLM-v11 drew 2–2. Goals: 38' Coach, 53' LLM-v11, 62' Coach, 64' LLM-v11. Coach is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach kept the ball in the opposition half 54% of the time; shots on goal 92–81.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 3 | 1 | 0 | 145.5 | 11.0 | +134.5 | **10** | 22.30 |
| 2 | Coach | 4 | 3 | 1 | 0 | 128.3 | 18.5 | +109.8 | **10** | 0.78 |
| 3 | PPO-tactician | 4 | 3 | 0 | 1 | 66.7 | 44.2 | +22.5 | **9** | 0.36 |
| 4 | Tactics | 4 | 2 | 0 | 2 | 82.3 | 46.7 | +35.7 | **6** | 2.00 |
| 5 | GP | 4 | 2 | 0 | 2 | 28.5 | 11.8 | +16.7 | **6** | 1.91 |
| 6 | champ-6 | 4 | 2 | 0 | 2 | 24.3 | 75.3 | -51.0 | **6** | 1.04 |

### Round 39 (13:11)

**Showcase: LLM-v11 2 – 2 Coach** (a draw). Video: [round-039.mp4](videos/league3/round-039.mp4)

LLM-v11 and Coach drew 2–2. Goals: 42' LLM-v11, 47' Coach, 52' LLM-v11, 71' Coach. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 58% of the time; shots on goal 106–67.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 4 | 0 | 0 | 157.0 | 9.8 | +147.2 | **12** | 22.30 |
| 2 | Coach | 4 | 3 | 0 | 1 | 165.0 | 11.2 | +153.8 | **9** | 0.80 |
| 3 | GP-v2 | 4 | 3 | 0 | 1 | 43.7 | 40.3 | +3.3 | **9** | 0.73 |
| 4 | GP | 4 | 3 | 0 | 1 | 29.0 | 72.5 | -43.5 | **9** | 2.02 |
| 5 | PPO-tactician | 4 | 2 | 0 | 2 | 88.5 | 35.0 | +53.5 | **6** | 0.42 |
| 6 | Tactics | 4 | 2 | 0 | 2 | 58.7 | 41.8 | +16.8 | **6** | 2.00 |

### Round 40 (13:56)

**Showcase: Coach 1 – 1 LLM-v11** (a draw). Video: [round-040.mp4](videos/league3/round-040.mp4)

Coach and LLM-v11 drew 1–1. Goals: 39' Coach, 82' LLM-v11. Coach is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach kept the ball in the opposition half 50% of the time; shots on goal 101–99. New: the most shots so far (200).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 4 | 0 | 0 | 129.3 | 8.5 | +120.8 | **12** | 22.30 |
| 2 | Coach | 4 | 3 | 0 | 1 | 213.8 | 12.0 | +201.8 | **9** | 0.83 |
| 3 | PPO-tactician | 4 | 3 | 0 | 1 | 67.0 | 40.7 | +26.3 | **9** | 0.48 |
| 4 | GP-v2 | 4 | 3 | 0 | 1 | 57.7 | 44.2 | +13.5 | **9** | 0.87 |
| 5 | Tactics | 4 | 2 | 0 | 2 | 77.2 | 43.2 | +34.0 | **6** | 2.00 |
| 6 | PPO-exploiter | 4 | 2 | 0 | 2 | 18.3 | 84.7 | -66.3 | **6** | 0.48 |

### Round 41 (14:42)

**Showcase: LLM-v11 1 – 1 Coach** (a draw). Video: [round-041.mp4](videos/league3/round-041.mp4)

LLM-v11 and Coach drew 1–1. Goals: 22' LLM-v11, 67' Coach. LLM-v11 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach kept the ball in the opposition half 58% of the time; shots on goal 74–122.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 4 | 0 | 0 | 155.3 | 11.5 | +143.8 | **12** | 22.30 |
| 2 | Coach | 4 | 3 | 0 | 1 | 139.7 | 14.8 | +124.8 | **9** | 0.85 |
| 3 | PPO-tactician | 4 | 3 | 0 | 1 | 69.2 | 36.8 | +32.3 | **9** | 0.56 |
| 4 | GP | 4 | 3 | 0 | 1 | 41.7 | 14.0 | +27.7 | **9** | 2.25 |
| 5 | Tactics | 4 | 2 | 0 | 2 | 108.5 | 44.8 | +63.7 | **6** | 2.00 |
| 6 | GP-v2 | 4 | 2 | 0 | 2 | 40.3 | 65.2 | -24.8 | **6** | 1.00 |

### Round 42 (15:27)

**Showcase: Coach 2 – 3 LLM-v11** (LLM-v11 wins). Video: [round-042.mp4](videos/league3/round-042.mp4)

LLM-v11 beat Coach 3–2. Goals: 5' LLM-v11, 21' LLM-v11, 39' Coach, 51' LLM-v11, 61' Coach. Coach is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach kept the ball in the opposition half 56% of the time; shots on goal 96–69.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 4 | 0 | 0 | 163.3 | 11.0 | +152.3 | **12** | 22.30 |
| 2 | Coach | 4 | 3 | 0 | 1 | 162.7 | 12.2 | +150.5 | **9** | 0.88 |
| 3 | PPO-tactician | 4 | 3 | 0 | 1 | 58.7 | 39.3 | +19.3 | **9** | 0.59 |
| 4 | GP-v2 | 4 | 2 | 1 | 1 | 67.3 | 36.0 | +31.3 | **7** | 1.14 |
| 5 | champ-6 | 4 | 2 | 1 | 1 | 32.0 | 72.8 | -40.8 | **7** | 1.04 |
| 6 | Tactics | 4 | 2 | 0 | 2 | 58.0 | 42.7 | +15.3 | **6** | 2.00 |

### Round 43 (16:20)

**Showcase: LLM-v11 0 – 2 Coach** (Coach wins). Video: [round-043.mp4](videos/league3/round-043.mp4)

Coach beat LLM-v11 2–0. Goals: 50' Coach, 90' Coach. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach kept the ball in the opposition half 50% of the time; shots on goal 98–91.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Coach | 4 | 4 | 0 | 0 | 127.7 | 10.2 | +117.5 | **12** | 0.90 |
| 2 | LLM-v11 | 4 | 3 | 0 | 1 | 160.0 | 13.8 | +146.2 | **9** | 22.30 |
| 3 | GP-v2 | 4 | 3 | 0 | 1 | 65.8 | 36.0 | +29.8 | **9** | 1.24 |
| 4 | PPO-tactician | 4 | 2 | 1 | 1 | 53.0 | 43.3 | +9.7 | **7** | 0.65 |
| 5 | GP | 4 | 2 | 0 | 2 | 15.8 | 11.0 | +4.8 | **6** | 2.44 |
| 6 | PPO-exploiter | 4 | 2 | 0 | 2 | 14.8 | 52.3 | -37.5 | **6** | 0.65 |

### Round 44 (17:06)

**Showcase: LLM-v11 1 – 1 Coach** (a draw). Video: [round-044.mp4](videos/league3/round-044.mp4)

LLM-v11 and Coach drew 1–1. Goals: 80' LLM-v11, 89' Coach. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. Coach is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 52% of the time; shots on goal 107–86.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 4 | 0 | 0 | 128.5 | 11.7 | +116.8 | **12** | 22.30 |
| 2 | Coach | 4 | 3 | 0 | 1 | 209.0 | 11.8 | +197.2 | **9** | 0.93 |
| 3 | GP-v2 | 4 | 2 | 1 | 1 | 50.2 | 38.0 | +12.2 | **7** | 1.37 |
| 4 | GP | 4 | 2 | 1 | 1 | 10.2 | 2.5 | +7.7 | **7** | 2.54 |
| 5 | PPO-tactician | 4 | 2 | 0 | 2 | 56.7 | 44.3 | +12.3 | **6** | 0.69 |
| 6 | Tactics | 4 | 2 | 0 | 2 | 42.7 | 35.3 | +7.3 | **6** | 2.00 |

### Round 45 (17:50)

**Showcase: LLM-v11 8 – 2 GP-v2** (LLM-v11 wins). Video: [round-045.mp4](videos/league3/round-045.mp4)

LLM-v11 beat GP-v2 8–2. Goals: 3' LLM-v11, 16' LLM-v11, 23' GP-v2, 28' LLM-v11, 33' LLM-v11, 46' LLM-v11, 58' GP-v2, 67' LLM-v11…. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP-v2 moves as a tight pack, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 80% of the time; shots on goal 150–15.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 4 | 0 | 0 | 141.3 | 2.8 | +138.5 | **12** | 22.30 |
| 2 | PPO-tactician | 4 | 3 | 0 | 1 | 46.5 | 15.3 | +31.2 | **9** | 0.74 |
| 3 | GP-v2 | 4 | 3 | 0 | 1 | 45.2 | 33.5 | +11.7 | **9** | 1.51 |
| 4 | Tactics | 4 | 2 | 0 | 2 | 83.3 | 41.8 | +41.5 | **6** | 2.00 |
| 5 | GP | 4 | 2 | 0 | 2 | 7.5 | 18.7 | -11.2 | **6** | 2.65 |
| 6 | PPO-goals | 4 | 2 | 0 | 2 | 17.2 | 40.7 | -23.5 | **6** | 1.73 |

### Round 46 (18:34)

**Showcase: LLM-v11 8 – 0 GP-v2** (LLM-v11 wins). Video: [round-046.mp4](videos/league3/round-046.mp4)

LLM-v11 beat GP-v2 8–0. Goals: 11' LLM-v11, 30' LLM-v11, 36' LLM-v11, 53' LLM-v11, 57' LLM-v11, 68' LLM-v11, 81' LLM-v11, 85' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 99% of the time and runs flat out almost all the time. GP-v2 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 72% of the time; shots on goal 126–10.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 3 | 1 | 0 | 126.7 | 3.0 | +123.7 | **10** | 22.30 |
| 2 | Tactics | 4 | 3 | 0 | 1 | 44.7 | 25.5 | +19.2 | **9** | 2.00 |
| 3 | GP-v2 | 4 | 2 | 1 | 1 | 42.2 | 36.7 | +5.5 | **7** | 1.61 |
| 4 | PPO-tactician | 4 | 2 | 0 | 2 | 95.8 | 41.0 | +54.8 | **6** | 0.82 |
| 5 | GP | 4 | 1 | 3 | 0 | 4.7 | 1.8 | +2.8 | **6** | 2.75 |
| 6 | PPO-chance | 4 | 1 | 2 | 1 | 19.2 | 65.7 | -46.5 | **5** | 0.80 |

### Round 47 (19:19)

**Showcase: GP-v2 1 – 6 LLM-v11** (LLM-v11 wins). Video: [round-047.mp4](videos/league3/round-047.mp4)

LLM-v11 beat GP-v2 6–1. Goals: 5' LLM-v11, 19' LLM-v11, 23' LLM-v11, 37' LLM-v11, 62' LLM-v11, 73' LLM-v11, 83' GP-v2. GP-v2 moves as a tight pack, keeps a goalkeeper back 60% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 78% of the time; shots on goal 20–182. New: the most shots so far (202).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 4 | 0 | 0 | 113.0 | 4.2 | +108.8 | **12** | 22.30 |
| 2 | GP-v2 | 4 | 3 | 0 | 1 | 73.0 | 26.5 | +46.5 | **9** | 1.74 |
| 3 | Tactics | 4 | 3 | 0 | 1 | 64.5 | 19.0 | +45.5 | **9** | 2.00 |
| 4 | GP | 4 | 2 | 1 | 1 | 13.8 | 2.2 | +11.7 | **7** | 2.89 |
| 5 | PPO-tactician | 4 | 2 | 1 | 1 | 34.7 | 31.2 | +3.5 | **7** | 0.86 |
| 6 | champ-5 | 4 | 2 | 0 | 2 | 21.0 | 28.5 | -7.5 | **6** | 0.99 |

### Round 48 (20:04)

**Showcase: LLM-v11 6 – 1 GP-v2** (LLM-v11 wins). Video: [round-048.mp4](videos/league3/round-048.mp4)

LLM-v11 beat GP-v2 6–1. Goals: 11' LLM-v11, 14' LLM-v11, 40' LLM-v11, 48' GP-v2, 54' LLM-v11, 59' LLM-v11, 82' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP-v2 moves as a tight pack and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 74% of the time; shots on goal 174–20.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 4 | 0 | 0 | 114.8 | 1.7 | +113.2 | **12** | 22.30 |
| 2 | GP-v2 | 4 | 3 | 1 | 0 | 52.3 | 6.2 | +46.2 | **10** | 1.86 |
| 3 | PPO-tactician | 4 | 3 | 0 | 1 | 99.2 | 27.7 | +71.5 | **9** | 0.91 |
| 4 | GP | 4 | 2 | 1 | 1 | 9.3 | 1.5 | +7.8 | **7** | 3.00 |
| 5 | Tactics | 4 | 2 | 0 | 2 | 55.2 | 42.7 | +12.5 | **6** | 2.00 |
| 6 | champ-5 | 4 | 2 | 0 | 2 | 23.8 | 26.8 | -3.0 | **6** | 0.99 |

### Round 49 (21:23)

**Showcase: LLM-v11 3 – 1 LLM-v11-GA** (LLM-v11 wins). Video: [round-049.mp4](videos/league3/round-049.mp4)

LLM-v11 beat LLM-v11-GA 3–1. Goals: 17' LLM-v11, 47' LLM-v11, 69' LLM-v11, 84' LLM-v11-GA. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11-GA keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 54% of the time; shots on goal 97–67.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 4 | 0 | 0 | 78.3 | 10.7 | +67.7 | **12** | 22.30 |
| 2 | PPO-tactician | 4 | 3 | 1 | 0 | 52.7 | 12.8 | +39.8 | **10** | 0.94 |
| 3 | LLM-v11-GA | 4 | 3 | 0 | 1 | 127.0 | 17.7 | +109.3 | **9** | 24.84 |
| 4 | GP | 4 | 2 | 1 | 1 | 11.3 | 2.3 | +9.0 | **7** | 3.09 |
| 5 | Tactics | 4 | 2 | 0 | 2 | 66.7 | 56.3 | +10.3 | **6** | 2.00 |
| 6 | champ-5 | 4 | 2 | 0 | 2 | 29.3 | 36.3 | -7.0 | **6** | 0.99 |

### Round 50 (22:11)

**Showcase: LLM-v11 1 – 0 LLM-v11-GA** (LLM-v11 wins). Video: [round-050.mp4](videos/league3/round-050.mp4)

LLM-v11 beat LLM-v11-GA 1–0. Goals: 86' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11-GA keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 52% of the time; shots on goal 95–90.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 3 | 1 | 0 | 147.0 | 15.7 | +131.3 | **10** | 22.30 |
| 2 | LLM-v11-GA | 4 | 3 | 1 | 0 | 98.3 | 12.8 | +85.5 | **10** | 24.86 |
| 3 | Tactics | 4 | 3 | 0 | 1 | 57.0 | 39.8 | +17.2 | **9** | 2.00 |
| 4 | GP-v2 | 4 | 2 | 1 | 1 | 42.2 | 30.8 | +11.3 | **7** | 2.08 |
| 5 | GP | 4 | 2 | 1 | 1 | 7.5 | 3.3 | +4.2 | **7** | 3.22 |
| 6 | PPO-exploiter | 4 | 2 | 1 | 1 | 28.7 | 49.5 | -20.8 | **7** | 1.02 |

### Round 51 (22:58)

**Showcase: LLM-v11-GA 2 – 3 LLM-v11** (LLM-v11 wins). Video: [round-051.mp4](videos/league3/round-051.mp4)

LLM-v11 beat LLM-v11-GA 3–2. Goals: 29' LLM-v11, 31' LLM-v11-GA, 42' LLM-v11-GA, 46' LLM-v11, 81' LLM-v11. LLM-v11-GA keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 53% of the time; shots on goal 85–104.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 4 | 0 | 0 | 123.2 | 13.0 | +110.2 | **12** | 22.30 |
| 2 | LLM-v11-GA | 4 | 3 | 0 | 1 | 98.2 | 19.0 | +79.2 | **9** | 24.89 |
| 3 | champ-6 | 4 | 3 | 0 | 1 | 30.5 | 73.5 | -43.0 | **9** | 1.04 |
| 4 | champ-5 | 4 | 2 | 1 | 1 | 36.8 | 23.7 | +13.2 | **7** | 0.99 |
| 5 | GP-v2 | 4 | 2 | 1 | 1 | 47.3 | 34.5 | +12.8 | **7** | 2.20 |
| 6 | PPO-v2 | 4 | 2 | 0 | 2 | 66.0 | 53.2 | +12.8 | **6** | 1.12 |

### Round 52 (23:44)

**Showcase: LLM-v11 3 – 4 LLM-v11-GA** (LLM-v11-GA wins). Video: [round-052.mp4](videos/league3/round-052.mp4)

LLM-v11-GA beat LLM-v11 4–3. Goals: 4' LLM-v11-GA, 9' LLM-v11, 14' LLM-v11-GA, 39' LLM-v11-GA, 53' LLM-v11, 75' LLM-v11-GA, 88' LLM-v11. LLM-v11 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11-GA keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11-GA kept the ball in the opposition half 53% of the time; shots on goal 80–89.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 4 | 0 | 0 | 136.0 | 13.3 | +122.7 | **12** | 22.30 |
| 2 | LLM-v11-GA | 4 | 3 | 0 | 1 | 98.3 | 13.3 | +85.0 | **9** | 24.92 |
| 3 | GP-v2 | 4 | 3 | 0 | 1 | 42.5 | 32.5 | +10.0 | **9** | 2.33 |
| 4 | champ-5 | 4 | 3 | 0 | 1 | 32.2 | 22.8 | +9.3 | **9** | 0.99 |
| 5 | PPO-tactician | 4 | 2 | 1 | 1 | 51.3 | 19.7 | +31.7 | **7** | 1.10 |
| 6 | PPO-v2 | 4 | 2 | 0 | 2 | 53.5 | 41.2 | +12.3 | **6** | 1.16 |

### Round 53 (00:29)

**Showcase: LLM-v11-GA 3 – 5 LLM-v11** (LLM-v11 wins). Video: [round-053.mp4](videos/league3/round-053.mp4)

LLM-v11 beat LLM-v11-GA 5–3. Goals: 12' LLM-v11, 17' LLM-v11-GA, 43' LLM-v11, 46' LLM-v11-GA, 51' LLM-v11, 61' LLM-v11-GA, 67' LLM-v11, 83' LLM-v11. LLM-v11-GA keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 53% of the time; shots on goal 90–88.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 4 | 0 | 0 | 109.8 | 13.8 | +96.0 | **12** | 22.30 |
| 2 | LLM-v11-GA | 4 | 3 | 0 | 1 | 205.5 | 18.7 | +186.8 | **9** | 24.94 |
| 3 | GP-v2 | 4 | 3 | 0 | 1 | 39.7 | 31.3 | +8.3 | **9** | 2.46 |
| 4 | GP | 4 | 2 | 1 | 1 | 8.0 | 19.0 | -11.0 | **7** | 3.60 |
| 5 | Tactics | 4 | 2 | 1 | 1 | 46.3 | 57.8 | -11.5 | **7** | 2.00 |
| 6 | PPO-v11clone | 4 | 2 | 0 | 2 | 55.2 | 70.8 | -15.7 | **6** | 0.35 |

### Round 54 (01:15)

**Showcase: LLM-v11-GA 3 – 4 LLM-v11** (LLM-v11 wins). Video: [round-054.mp4](videos/league3/round-054.mp4)

LLM-v11 beat LLM-v11-GA 4–3. Goals: 4' LLM-v11, 15' LLM-v11, 23' LLM-v11-GA, 39' LLM-v11-GA, 45' LLM-v11-GA, 52' LLM-v11, 61' LLM-v11. LLM-v11-GA keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 60% of the time; shots on goal 60–110.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11-GA | 4 | 3 | 1 | 0 | 133.0 | 17.5 | +115.5 | **10** | 24.97 |
| 2 | LLM-v11 | 4 | 3 | 1 | 0 | 126.2 | 12.8 | +113.3 | **10** | 22.30 |
| 3 | Tactics | 4 | 3 | 0 | 1 | 57.0 | 18.7 | +38.3 | **9** | 2.00 |
| 4 | PPO-v2 | 4 | 2 | 1 | 1 | 70.8 | 59.5 | +11.3 | **7** | 1.24 |
| 5 | PPO-tactician | 4 | 2 | 1 | 1 | 32.2 | 39.7 | -7.5 | **7** | 1.20 |
| 6 | GP-v2 | 4 | 2 | 1 | 1 | 27.0 | 36.2 | -9.2 | **7** | 2.59 |

### Round 55 (02:01)

**Showcase: LLM-v11 2 – 1 LLM-v11-GA** (LLM-v11 wins). Video: [round-055.mp4](videos/league3/round-055.mp4)

LLM-v11 beat LLM-v11-GA 2–1. Goals: 25' LLM-v11, 37' LLM-v11-GA, 78' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11-GA keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 52% of the time; shots on goal 98–93.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 3 | 1 | 0 | 132.2 | 13.7 | +118.5 | **10** | 22.30 |
| 2 | LLM-v11-GA | 4 | 3 | 1 | 0 | 105.2 | 11.7 | +93.5 | **10** | 24.99 |
| 3 | GP-v2 | 4 | 3 | 0 | 1 | 42.2 | 31.0 | +11.2 | **9** | 2.72 |
| 4 | PPO-tactician | 4 | 2 | 1 | 1 | 35.2 | 18.7 | +16.5 | **7** | 1.25 |
| 5 | Tactics | 4 | 2 | 1 | 1 | 36.0 | 33.7 | +2.3 | **7** | 2.00 |
| 6 | PPO-v2 | 4 | 2 | 0 | 2 | 53.5 | 65.0 | -11.5 | **6** | 1.29 |

### Round 56 (02:47)

**Showcase: LLM-v11-GA 3 – 3 LLM-v11** (a draw). Video: [round-056.mp4](videos/league3/round-056.mp4)

LLM-v11-GA and LLM-v11 drew 3–3. Goals: 8' LLM-v11-GA, 17' LLM-v11, 20' LLM-v11, 57' LLM-v11-GA, 62' LLM-v11-GA, 70' LLM-v11. LLM-v11-GA keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 58% of the time; shots on goal 77–116.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 3 | 1 | 0 | 155.7 | 15.2 | +140.5 | **10** | 22.30 |
| 2 | LLM-v11-GA | 4 | 3 | 1 | 0 | 93.3 | 13.7 | +79.7 | **10** | 25.02 |
| 3 | PPO-tactician | 4 | 3 | 0 | 1 | 49.3 | 33.5 | +15.8 | **9** | 1.30 |
| 4 | GP | 4 | 2 | 1 | 1 | 4.0 | 1.7 | +2.3 | **7** | 3.99 |
| 5 | Tactics | 4 | 2 | 1 | 1 | 41.7 | 49.3 | -7.7 | **7** | 2.00 |
| 6 | GP-v2 | 4 | 2 | 1 | 1 | 18.7 | 31.2 | -12.5 | **7** | 2.85 |

### Round 57 (03:34)

**Showcase: LLM-v11 3 – 1 LLM-v11-GA** (LLM-v11 wins). Video: [round-057.mp4](videos/league3/round-057.mp4)

LLM-v11 beat LLM-v11-GA 3–1. Goals: 40' LLM-v11, 74' LLM-v11-GA, 84' LLM-v11, 90' LLM-v11. LLM-v11 keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11-GA is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11-GA kept the ball in the opposition half 50% of the time; shots on goal 81–94.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 3 | 1 | 0 | 107.7 | 15.8 | +91.8 | **10** | 22.30 |
| 2 | Tactics | 4 | 3 | 1 | 0 | 70.7 | 15.7 | +55.0 | **10** | 2.00 |
| 3 | LLM-v11-GA | 4 | 3 | 0 | 1 | 111.7 | 21.2 | +90.5 | **9** | 25.04 |
| 4 | PPO-tactician | 4 | 3 | 0 | 1 | 48.8 | 38.8 | +10.0 | **9** | 1.36 |
| 5 | GP | 4 | 2 | 2 | 0 | 6.2 | 1.3 | +4.8 | **8** | 4.13 |
| 6 | champ-5 | 4 | 2 | 0 | 2 | 26.7 | 33.8 | -7.2 | **6** | 0.99 |

### Round 58 (04:20)

**Showcase: LLM-v11 2 – 1 LLM-v11-GA** (LLM-v11 wins). Video: [round-058.mp4](videos/league3/round-058.mp4)

LLM-v11 beat LLM-v11-GA 2–1. Goals: 32' LLM-v11, 61' LLM-v11, 87' LLM-v11-GA. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11-GA is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 50% of the time; shots on goal 83–78.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 4 | 0 | 0 | 115.2 | 10.7 | +104.5 | **12** | 22.30 |
| 2 | LLM-v11-GA | 4 | 3 | 0 | 1 | 118.5 | 20.0 | +98.5 | **9** | 25.04 |
| 3 | PPO-tactician | 4 | 3 | 0 | 1 | 52.0 | 18.7 | +33.3 | **9** | 1.41 |
| 4 | GP-v2 | 4 | 3 | 0 | 1 | 47.0 | 30.3 | +16.7 | **9** | 3.12 |
| 5 | GP | 4 | 2 | 1 | 1 | 7.3 | 2.0 | +5.3 | **7** | 4.27 |
| 6 | Tactics | 4 | 2 | 1 | 1 | 32.5 | 44.5 | -12.0 | **7** | 2.00 |

### Round 59 (05:07)

**Showcase: LLM-v11 3 – 3 LLM-v11-GA** (a draw). Video: [round-059.mp4](videos/league3/round-059.mp4)

LLM-v11 and LLM-v11-GA drew 3–3. Goals: 17' LLM-v11, 39' LLM-v11, 44' LLM-v11-GA, 61' LLM-v11-GA, 64' LLM-v11, 86' LLM-v11-GA. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11-GA keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 62% of the time; shots on goal 105–58.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 4 | 3 | 1 | 0 | 107.8 | 13.0 | +94.8 | **10** | 22.30 |
| 2 | LLM-v11-GA | 4 | 3 | 0 | 1 | 108.0 | 21.0 | +87.0 | **9** | 25.04 |
| 3 | PPO-tactician | 4 | 3 | 0 | 1 | 49.7 | 20.5 | +29.2 | **9** | 1.47 |
| 4 | GP | 4 | 2 | 2 | 0 | 9.8 | 1.3 | +8.5 | **8** | 4.40 |
| 5 | GP-v2 | 4 | 2 | 1 | 1 | 25.2 | 32.5 | -7.3 | **7** | 3.26 |
| 6 | Tactics | 4 | 2 | 0 | 2 | 63.5 | 40.7 | +22.8 | **6** | 2.00 |

### Round 60 (05:55)

**Showcase: GP-v2 0 – 5 LLM-v11** (LLM-v11 wins). Video: [round-060.mp4](videos/league3/round-060.mp4)

LLM-v11 beat GP-v2 5–0. Goals: 7' LLM-v11, 29' LLM-v11, 34' LLM-v11, 41' LLM-v11, 86' LLM-v11. GP-v2 moves as a tight pack and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 76% of the time; shots on goal 20–145.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 5 | 5 | 0 | 0 | 155.0 | 4.8 | +150.2 | **15** | 22.30 |
| 2 | GP-v2 | 5 | 4 | 0 | 1 | 53.7 | 28.2 | +25.5 | **12** | 3.37 |
| 3 | PPO-v2 | 5 | 3 | 0 | 2 | 42.3 | 47.0 | -4.7 | **9** | 1.51 |
| 4 | champ-7 | 5 | 2 | 0 | 3 | 42.7 | 50.3 | -7.7 | **6** | 0.00 |
| 5 | PPO-v11clone | 5 | 1 | 0 | 4 | 23.3 | 76.2 | -52.8 | **3** | 0.65 |
| 6 | PPO-goals | 5 | 0 | 0 | 5 | 7.7 | 118.2 | -110.5 | **0** | 2.32 |

### Round 61 (06:42)

**Showcase: GP-v2 1 – 7 LLM-v11** (LLM-v11 wins). Video: [round-061.mp4](videos/league3/round-061.mp4)

LLM-v11 beat GP-v2 7–1. Goals: 28' LLM-v11, 43' LLM-v11, 45' LLM-v11, 49' GP-v2, 66' LLM-v11, 68' LLM-v11, 71' LLM-v11, 78' LLM-v11. GP-v2 runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 71% of the time; shots on goal 33–139.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 5 | 5 | 0 | 0 | 184.2 | 3.3 | +180.8 | **15** | 22.30 |
| 2 | GP-v2 | 5 | 4 | 0 | 1 | 46.2 | 30.5 | +15.7 | **12** | 3.61 |
| 3 | PPO-v2 | 5 | 3 | 0 | 2 | 32.8 | 42.3 | -9.5 | **9** | 1.56 |
| 4 | champ-7 | 5 | 2 | 0 | 3 | 30.3 | 49.5 | -19.2 | **6** | 0.00 |
| 5 | champ-6 | 5 | 1 | 0 | 4 | 18.7 | 93.2 | -74.5 | **3** | 1.04 |
| 6 | PPO-goals | 5 | 0 | 0 | 5 | 8.7 | 102.0 | -93.3 | **0** | 2.43 |

### Round 62 (07:29)

**Showcase: GP-v2 2 – 6 LLM-v11** (LLM-v11 wins). Video: [round-062.mp4](videos/league3/round-062.mp4)

LLM-v11 beat GP-v2 6–2. Goals: 3' GP-v2, 7' LLM-v11, 12' LLM-v11, 22' LLM-v11, 25' LLM-v11, 31' LLM-v11, 35' LLM-v11, 40' GP-v2. GP-v2 moves as a tight pack and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 74% of the time; shots on goal 23–146.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 5 | 5 | 0 | 0 | 179.3 | 3.8 | +175.5 | **15** | 22.30 |
| 2 | GP-v2 | 5 | 4 | 0 | 1 | 40.8 | 30.7 | +10.2 | **12** | 3.95 |
| 3 | PPO-v2 | 5 | 3 | 0 | 2 | 34.7 | 42.3 | -7.7 | **9** | 1.63 |
| 4 | champ-7 | 5 | 2 | 0 | 3 | 20.5 | 54.7 | -34.2 | **6** | 0.00 |
| 5 | champ-6 | 5 | 1 | 0 | 4 | 13.5 | 86.0 | -72.5 | **3** | 1.04 |
| 6 | PPO-goals | 5 | 0 | 0 | 5 | 3.7 | 75.0 | -71.3 | **0** | 2.54 |

### Round 63 (08:08)

**Showcase: LLM-v11 9 – 2 GP-v2** (LLM-v11 wins). Video: [round-063.mp4](videos/league3/round-063.mp4)

LLM-v11 beat GP-v2 9–2. Goals: 9' LLM-v11, 12' LLM-v11, 21' GP-v2, 27' GP-v2, 32' LLM-v11, 38' LLM-v11, 46' LLM-v11, 62' LLM-v11…. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP-v2 moves as a tight pack and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 75% of the time; shots on goal 139–12.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 84.8 | 3.0 | +81.8 | **9** | 22.30 |
| 2 | GP-v2 | 3 | 2 | 0 | 1 | 21.3 | 30.3 | -9.0 | **6** | 4.28 |
| 3 | PPO-v2 | 3 | 1 | 0 | 2 | 19.2 | 25.5 | -6.3 | **3** | 1.70 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 1.5 | 68.0 | -66.5 | **0** | 2.69 |

### Round 64 (08:47)

**Showcase: GP-v2 0 – 7 LLM-v11** (LLM-v11 wins). Video: [round-064.mp4](videos/league3/round-064.mp4)

LLM-v11 beat GP-v2 7–0. Goals: 4' LLM-v11, 9' LLM-v11, 14' LLM-v11, 27' LLM-v11, 31' LLM-v11, 62' LLM-v11, 75' LLM-v11. GP-v2 moves as a tight pack and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 99% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 71% of the time; shots on goal 14–138.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 91.8 | 4.3 | +87.5 | **9** | 22.30 |
| 2 | GP-v2 | 3 | 2 | 0 | 1 | 22.3 | 32.7 | -10.3 | **6** | 4.58 |
| 3 | PPO-v2 | 3 | 1 | 0 | 2 | 15.5 | 23.0 | -7.5 | **3** | 1.76 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 2.5 | 72.2 | -69.7 | **0** | 2.82 |

### Round 65 (09:27)

**Showcase: LLM-v11 4 – 0 PPO-v2** (LLM-v11 wins). Video: [round-065.mp4](videos/league3/round-065.mp4)

LLM-v11 beat PPO-v2 4–0. Goals: 7' LLM-v11, 10' LLM-v11, 40' LLM-v11, 51' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. PPO-v2 moves as a tight pack, keeps a goalkeeper back 91% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 84% of the time; shots on goal 159–11.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 88.7 | 3.5 | +85.2 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 1 | 1 | 1 | 24.0 | 22.7 | +1.3 | **4** | 1.83 |
| 3 | GP-v2 | 3 | 1 | 1 | 1 | 13.3 | 35.0 | -21.7 | **4** | 4.87 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 2.2 | 67.0 | -64.8 | **0** | 2.95 |

### Round 66 (10:06)

**Showcase: LLM-v11 9 – 1 GP-v2** (LLM-v11 wins). Video: [round-066.mp4](videos/league3/round-066.mp4)

LLM-v11 beat GP-v2 9–1. Goals: 27' LLM-v11, 35' LLM-v11, 40' LLM-v11, 44' LLM-v11, 47' LLM-v11, 50' LLM-v11, 57' LLM-v11, 59' LLM-v11…. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP-v2 moves as a tight pack, keeps a goalkeeper back 61% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 80% of the time; shots on goal 165–17.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 78.3 | 3.0 | +75.3 | **9** | 22.30 |
| 2 | GP-v2 | 3 | 2 | 0 | 1 | 14.2 | 32.3 | -18.2 | **6** | 5.16 |
| 3 | PPO-v2 | 3 | 1 | 0 | 2 | 16.8 | 32.5 | -15.7 | **3** | 1.88 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 3.3 | 44.8 | -41.5 | **0** | 3.05 |

### Round 67 (10:46)

**Showcase: GP-v2 1 – 9 LLM-v11** (LLM-v11 wins). Video: [round-067.mp4](videos/league3/round-067.mp4)

LLM-v11 beat GP-v2 9–1. Goals: 8' LLM-v11, 13' LLM-v11, 19' LLM-v11, 23' LLM-v11, 26' LLM-v11, 31' LLM-v11, 52' LLM-v11, 59' LLM-v11…. GP-v2 moves as a tight pack and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 76% of the time; shots on goal 8–148.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 84.5 | 3.7 | +80.8 | **9** | 22.30 |
| 2 | GP-v2 | 3 | 2 | 0 | 1 | 14.0 | 35.0 | -21.0 | **6** | 5.46 |
| 3 | PPO-v2 | 3 | 1 | 0 | 2 | 17.2 | 31.0 | -13.8 | **3** | 1.94 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 5.5 | 51.5 | -46.0 | **0** | 3.17 |

### Round 68 (11:26)

**Showcase: LLM-v11 7 – 0 GP-v2** (LLM-v11 wins). Video: [round-068.mp4](videos/league3/round-068.mp4)

LLM-v11 beat GP-v2 7–0. Goals: 4' LLM-v11, 8' LLM-v11, 10' LLM-v11, 33' LLM-v11, 51' LLM-v11, 65' LLM-v11, 81' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP-v2 moves as a tight pack and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 76% of the time; shots on goal 165–15.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 76.3 | 3.0 | +73.3 | **9** | 22.30 |
| 2 | GP-v2 | 3 | 2 | 0 | 1 | 14.5 | 28.5 | -14.0 | **6** | 5.68 |
| 3 | PPO-v2 | 3 | 1 | 0 | 2 | 17.7 | 25.0 | -7.3 | **3** | 2.02 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 5.2 | 57.2 | -52.0 | **0** | 3.31 |

### Round 69 (12:06)

**Showcase: GP-v2 0 – 5 LLM-v11** (LLM-v11 wins). Video: [round-069.mp4](videos/league3/round-069.mp4)

LLM-v11 beat GP-v2 5–0. Goals: 21' LLM-v11, 36' LLM-v11, 48' LLM-v11, 50' LLM-v11, 73' LLM-v11. GP-v2 moves as a tight pack and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 99% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 74% of the time; shots on goal 15–163.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 65.5 | 4.2 | +61.3 | **9** | 22.30 |
| 2 | GP-v2 | 3 | 2 | 0 | 1 | 14.7 | 25.3 | -10.7 | **6** | 5.96 |
| 3 | PPO-v2 | 3 | 1 | 0 | 2 | 15.0 | 22.8 | -7.8 | **3** | 2.11 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 1.7 | 44.5 | -42.8 | **0** | 3.49 |

### Round 70 (12:46)

**Showcase: LLM-v11 4 – 1 PPO-v2** (LLM-v11 wins). Video: [round-070.mp4](videos/league3/round-070.mp4)

LLM-v11 beat PPO-v2 4–1. Goals: 19' LLM-v11, 30' LLM-v11, 48' LLM-v11, 50' PPO-v2, 67' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 84% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 73% of the time; shots on goal 148–16.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 75.0 | 4.7 | +70.3 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 1 | 1 | 1 | 18.5 | 23.3 | -4.8 | **4** | 2.21 |
| 3 | GP-v2 | 3 | 1 | 1 | 1 | 16.3 | 29.3 | -13.0 | **4** | 6.25 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 3.2 | 55.7 | -52.5 | **0** | 3.67 |

### Round 71 (13:27)

**Showcase: LLM-v11 5 – 0 PPO-v2** (LLM-v11 wins). Video: [round-071.mp4](videos/league3/round-071.mp4)

LLM-v11 beat PPO-v2 5–0. Goals: 19' LLM-v11, 36' LLM-v11, 42' LLM-v11, 45' LLM-v11, 53' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 86% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 75% of the time; shots on goal 137–13.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 93.7 | 4.3 | +89.3 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 2 | 0 | 1 | 25.8 | 31.3 | -5.5 | **6** | 2.30 |
| 3 | GP-v2 | 3 | 1 | 0 | 2 | 12.7 | 47.7 | -35.0 | **3** | 6.53 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 5.7 | 54.5 | -48.8 | **0** | 3.85 |

### Round 72 (14:07)

**Showcase: LLM-v11 6 – 0 GP-v2** (LLM-v11 wins). Video: [round-072.mp4](videos/league3/round-072.mp4)

LLM-v11 beat GP-v2 6–0. Goals: 5' LLM-v11, 48' LLM-v11, 58' LLM-v11, 68' LLM-v11, 84' LLM-v11, 87' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP-v2 moves as a tight pack, keeps a goalkeeper back 66% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 81% of the time; shots on goal 164–11.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 62.3 | 5.2 | +57.2 | **9** | 22.30 |
| 2 | GP-v2 | 3 | 2 | 0 | 1 | 12.3 | 27.0 | -14.7 | **6** | 6.81 |
| 3 | PPO-v2 | 3 | 1 | 0 | 2 | 17.3 | 23.3 | -6.0 | **3** | 2.40 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 2.3 | 38.8 | -36.5 | **0** | 4.03 |

### Round 73 (14:48)

**Showcase: LLM-v11 6 – 0 GP-v2** (LLM-v11 wins). Video: [round-073.mp4](videos/league3/round-073.mp4)

LLM-v11 beat GP-v2 6–0. Goals: 8' LLM-v11, 21' LLM-v11, 24' LLM-v11, 32' LLM-v11, 62' LLM-v11, 72' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP-v2 runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 66% of the time; shots on goal 145–17.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 77.5 | 3.7 | +73.8 | **9** | 22.30 |
| 2 | GP-v2 | 3 | 2 | 0 | 1 | 14.7 | 27.2 | -12.5 | **6** | 7.10 |
| 3 | PPO-v2 | 3 | 1 | 0 | 2 | 16.3 | 33.7 | -17.3 | **3** | 2.49 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 4.8 | 48.8 | -44.0 | **0** | 4.21 |

### Round 74 (15:29)

**Showcase: LLM-v11 7 – 1 GP-v2** (LLM-v11 wins). Video: [round-074.mp4](videos/league3/round-074.mp4)

LLM-v11 beat GP-v2 7–1. Goals: 5' LLM-v11, 31' GP-v2, 43' LLM-v11, 66' LLM-v11, 71' LLM-v11, 74' LLM-v11, 77' LLM-v11, 88' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP-v2 runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 63% of the time; shots on goal 123–24.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 80.3 | 3.3 | +77.0 | **9** | 22.30 |
| 2 | GP-v2 | 3 | 2 | 0 | 1 | 11.2 | 32.7 | -21.5 | **6** | 7.39 |
| 3 | PPO-v2 | 3 | 1 | 0 | 2 | 17.0 | 25.2 | -8.2 | **3** | 2.59 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 3.3 | 50.7 | -47.3 | **0** | 4.40 |

### Round 75 (16:10)

**Showcase: LLM-v11 7 – 0 GP-v2** (LLM-v11 wins). Video: [round-075.mp4](videos/league3/round-075.mp4)

LLM-v11 beat GP-v2 7–0. Goals: 10' LLM-v11, 12' LLM-v11, 19' LLM-v11, 35' LLM-v11, 64' LLM-v11, 73' LLM-v11, 84' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP-v2 moves as a tight pack and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 73% of the time; shots on goal 232–29. New: the most shots so far (261).

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 79.3 | 2.8 | +76.5 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 1 | 1 | 1 | 17.8 | 22.0 | -4.2 | **4** | 2.68 |
| 3 | GP-v2 | 3 | 1 | 1 | 1 | 8.2 | 30.2 | -22.0 | **4** | 7.68 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 2.2 | 52.5 | -50.3 | **0** | 4.58 |

### Round 76 (16:50)

**Showcase: LLM-v11 8 – 1 GP-v2** (LLM-v11 wins). Video: [round-076.mp4](videos/league3/round-076.mp4)

LLM-v11 beat GP-v2 8–1. Goals: 6' LLM-v11, 24' LLM-v11, 26' GP-v2, 33' LLM-v11, 37' LLM-v11, 41' LLM-v11, 43' LLM-v11, 72' LLM-v11…. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP-v2 moves as a tight pack and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 71% of the time; shots on goal 138–22.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 87.5 | 3.8 | +83.7 | **9** | 22.30 |
| 2 | GP-v2 | 3 | 2 | 0 | 1 | 14.8 | 29.7 | -14.8 | **6** | 7.96 |
| 3 | PPO-v2 | 3 | 1 | 0 | 2 | 18.7 | 29.2 | -10.5 | **3** | 2.78 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 3.3 | 61.7 | -58.3 | **0** | 4.76 |

### Round 77 (17:29)

**Showcase: PPO-v2 0 – 3 LLM-v11** (LLM-v11 wins). Video: [round-077.mp4](videos/league3/round-077.mp4)

LLM-v11 beat PPO-v2 3–0. Goals: 36' LLM-v11, 64' LLM-v11, 84' LLM-v11. PPO-v2 keeps a goalkeeper back 94% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 69% of the time; shots on goal 13–131.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 78.8 | 5.7 | +73.2 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 1 | 1 | 1 | 21.3 | 26.7 | -5.3 | **4** | 2.87 |
| 3 | GP-v2 | 3 | 1 | 1 | 1 | 15.3 | 26.7 | -11.3 | **4** | 8.23 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 1.7 | 58.2 | -56.5 | **0** | 4.93 |

### Round 78 (18:08)

**Showcase: GP-v2 2 – 6 LLM-v11** (LLM-v11 wins). Video: [round-078.mp4](videos/league3/round-078.mp4)

LLM-v11 beat GP-v2 6–2. Goals: 17' LLM-v11, 30' LLM-v11, 41' LLM-v11, 51' GP-v2, 58' LLM-v11, 78' GP-v2, 82' LLM-v11, 85' LLM-v11. GP-v2 moves as a tight pack and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 70% of the time; shots on goal 19–149.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 101.3 | 5.5 | +95.8 | **9** | 22.30 |
| 2 | GP-v2 | 3 | 2 | 0 | 1 | 17.8 | 23.8 | -6.0 | **6** | 8.51 |
| 3 | PPO-v2 | 3 | 1 | 0 | 2 | 20.7 | 25.2 | -4.5 | **3** | 2.96 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 1.2 | 86.5 | -85.3 | **0** | 5.11 |

### Round 79 (18:47)

**Showcase: LLM-v11 5 – 0 GP-v2** (LLM-v11 wins). Video: [round-079.mp4](videos/league3/round-079.mp4)

LLM-v11 beat GP-v2 5–0. Goals: 24' LLM-v11, 42' LLM-v11, 46' LLM-v11, 76' LLM-v11, 86' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP-v2 moves as a tight pack, keeps a goalkeeper back 60% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 78% of the time; shots on goal 174–13.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 85.5 | 6.2 | +79.3 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 1 | 1 | 1 | 21.5 | 23.8 | -2.3 | **4** | 3.05 |
| 3 | GP-v2 | 3 | 1 | 1 | 1 | 16.5 | 29.0 | -12.5 | **4** | 8.78 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 1.7 | 66.2 | -64.5 | **0** | 5.28 |

### Round 80 (19:26)

**Showcase: LLM-v11 6 – 0 GP-v2** (LLM-v11 wins). Video: [round-080.mp4](videos/league3/round-080.mp4)

LLM-v11 beat GP-v2 6–0. Goals: 6' LLM-v11, 11' LLM-v11, 19' LLM-v11, 21' LLM-v11, 61' LLM-v11, 63' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP-v2 moves as a tight pack and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 76% of the time; shots on goal 171–15.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 86.5 | 3.7 | +82.8 | **9** | 22.30 |
| 2 | GP-v2 | 3 | 2 | 0 | 1 | 12.5 | 30.2 | -17.7 | **6** | 9.06 |
| 3 | PPO-v2 | 3 | 1 | 0 | 2 | 17.0 | 27.0 | -10.0 | **3** | 3.14 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 2.0 | 57.2 | -55.2 | **0** | 5.46 |

### Round 81 (20:06)

**Showcase: GP-v2 2 – 4 LLM-v11** (LLM-v11 wins). Video: [round-081.mp4](videos/league3/round-081.mp4)

LLM-v11 beat GP-v2 4–2. Goals: 5' LLM-v11, 26' GP-v2, 32' LLM-v11, 37' LLM-v11, 41' LLM-v11, 71' GP-v2. GP-v2 moves as a tight pack, leaves its goal empty and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 67% of the time; shots on goal 27–148.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 57.7 | 7.3 | +50.3 | **9** | 22.30 |
| 2 | GP-v2 | 3 | 2 | 0 | 1 | 12.8 | 19.5 | -6.7 | **6** | 9.33 |
| 3 | PPO-v2 | 3 | 1 | 0 | 2 | 22.2 | 19.3 | +2.8 | **3** | 3.23 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 1.3 | 47.8 | -46.5 | **0** | 5.63 |

### Round 82 (20:46)

**Showcase: LLM-v11 2 – 0 GP-v2** (LLM-v11 wins). Video: [round-082.mp4](videos/league3/round-082.mp4)

LLM-v11 beat GP-v2 2–0. Goals: 38' LLM-v11, 67' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP-v2 moves as a tight pack and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 68% of the time; shots on goal 152–16.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 64.0 | 5.7 | +58.3 | **9** | 22.30 |
| 2 | GP-v2 | 3 | 2 | 0 | 1 | 13.0 | 15.7 | -2.7 | **6** | 9.60 |
| 3 | PPO-v2 | 3 | 1 | 0 | 2 | 24.3 | 23.2 | +1.2 | **3** | 3.32 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 2.8 | 59.7 | -56.8 | **0** | 5.80 |

### Round 83 (21:26)

**Showcase: LLM-v11 4 – 0 GP-v2** (LLM-v11 wins). Video: [round-083.mp4](videos/league3/round-083.mp4)

LLM-v11 beat GP-v2 4–0. Goals: 3' LLM-v11, 5' LLM-v11, 25' LLM-v11, 49' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP-v2 moves as a tight pack and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 75% of the time; shots on goal 166–9.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 63.0 | 7.2 | +55.8 | **9** | 22.30 |
| 2 | GP-v2 | 3 | 2 | 0 | 1 | 12.7 | 20.0 | -7.3 | **6** | 9.87 |
| 3 | PPO-v2 | 3 | 1 | 0 | 2 | 22.0 | 22.0 | -0.0 | **3** | 3.42 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 1.8 | 50.3 | -48.5 | **0** | 5.97 |

### Round 84 (22:06)

**Showcase: LLM-v11 4 – 0 PPO-v2** (LLM-v11 wins). Video: [round-084.mp4](videos/league3/round-084.mp4)

LLM-v11 beat PPO-v2 4–0. Goals: 18' LLM-v11, 25' LLM-v11, 79' LLM-v11, 90' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 90% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 62% of the time; shots on goal 105–19.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 71.8 | 4.8 | +67.0 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 1 | 1 | 1 | 22.5 | 25.2 | -2.7 | **4** | 3.51 |
| 3 | GP-v2 | 3 | 1 | 1 | 1 | 11.3 | 30.5 | -19.2 | **4** | 10.15 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 1.5 | 46.7 | -45.2 | **0** | 6.15 |

### Round 85 (22:46)

**Showcase: LLM-v11 4 – 0 GP-v2** (LLM-v11 wins). Video: [round-085.mp4](videos/league3/round-085.mp4)

LLM-v11 beat GP-v2 4–0. Goals: 46' LLM-v11, 70' LLM-v11, 82' LLM-v11, 85' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP-v2 moves as a tight pack, leaves its goal empty and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 67% of the time; shots on goal 138–23.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 65.0 | 4.5 | +60.5 | **9** | 22.30 |
| 2 | GP-v2 | 3 | 2 | 0 | 1 | 9.7 | 18.2 | -8.5 | **6** | 10.42 |
| 3 | PPO-v2 | 3 | 1 | 0 | 2 | 21.7 | 21.8 | -0.2 | **3** | 3.60 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 2.7 | 54.5 | -51.8 | **0** | 6.32 |

### Round 86 (23:25)

**Showcase: PPO-v2 2 – 4 LLM-v11** (LLM-v11 wins). Video: [round-086.mp4](videos/league3/round-086.mp4)

LLM-v11 beat PPO-v2 4–2. Goals: 12' PPO-v2, 15' PPO-v2, 45' LLM-v11, 48' LLM-v11, 52' LLM-v11, 72' LLM-v11. PPO-v2 keeps a goalkeeper back 90% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 70% of the time; shots on goal 25–122.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 2 | 2 | 0 | 0 | 54.8 | 4.5 | +50.3 | **6** | 22.30 |
| 2 | PPO-v2 | 2 | 1 | 0 | 1 | 20.7 | 19.0 | +1.7 | **3** | 3.69 |
| 3 | PPO-goals | 2 | 0 | 0 | 2 | 2.0 | 54.0 | -52.0 | **0** | 6.49 |

### Round 87 (00:05)

**Showcase: PPO-v2 1 – 5 LLM-v11** (LLM-v11 wins). Video: [round-087.mp4](videos/league3/round-087.mp4)

LLM-v11 beat PPO-v2 5–1. Goals: 5' LLM-v11, 9' LLM-v11, 18' LLM-v11, 29' PPO-v2, 35' LLM-v11, 47' LLM-v11. PPO-v2 keeps a goalkeeper back 84% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 67% of the time; shots on goal 25–126.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 78.0 | 14.0 | +64.0 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 2 | 0 | 1 | 29.8 | 32.0 | -2.2 | **6** | 3.78 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 39.7 | 42.7 | -3.0 | **3** | 15.42 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 4.2 | 63.0 | -58.8 | **0** | 6.67 |

### Round 88 (00:46)

**Showcase: LLM-v11 4 – 0 PPO-v2** (LLM-v11 wins). Video: [round-088.mp4](videos/league3/round-088.mp4)

LLM-v11 beat PPO-v2 4–0. Goals: 18' LLM-v11, 61' LLM-v11, 64' LLM-v11, 68' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 79% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 72% of the time; shots on goal 141–22.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 75.2 | 14.0 | +61.2 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 2 | 0 | 1 | 35.2 | 34.3 | +0.8 | **6** | 3.87 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 41.8 | 38.0 | +3.8 | **3** | 15.70 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 4.3 | 70.2 | -65.8 | **0** | 6.85 |

### Round 89 (01:27)

**Showcase: LLM-v11 5 – 1 PPO-v2** (LLM-v11 wins). Video: [round-089.mp4](videos/league3/round-089.mp4)

LLM-v11 beat PPO-v2 5–1. Goals: 4' LLM-v11, 21' LLM-v11, 29' LLM-v11, 33' LLM-v11, 67' LLM-v11, 82' PPO-v2. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 78% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 68% of the time; shots on goal 117–38.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 70.3 | 13.5 | +56.8 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 2 | 0 | 1 | 26.8 | 29.5 | -2.7 | **6** | 3.97 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 35.5 | 29.0 | +6.5 | **3** | 15.98 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 2.8 | 63.5 | -60.7 | **0** | 7.03 |

### Round 90 (02:07)

**Showcase: PPO-v2 2 – 6 LLM-v11** (LLM-v11 wins). Video: [round-090.mp4](videos/league3/round-090.mp4)

LLM-v11 beat PPO-v2 6–2. Goals: 18' LLM-v11, 21' LLM-v11, 29' LLM-v11, 42' LLM-v11, 58' PPO-v2, 63' LLM-v11, 67' LLM-v11, 75' PPO-v2. PPO-v2 keeps a goalkeeper back 86% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 54% of the time; shots on goal 25–108.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 63.3 | 10.0 | +53.3 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 2 | 0 | 1 | 29.8 | 31.0 | -1.2 | **6** | 4.06 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 31.3 | 25.3 | +6.0 | **3** | 16.25 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 4.2 | 62.3 | -58.2 | **0** | 7.21 |

### Round 91 (02:46)

**Showcase: PPO-v2 0 – 3 LLM-v11** (LLM-v11 wins). Video: [round-091.mp4](videos/league3/round-091.mp4)

LLM-v11 beat PPO-v2 3–0. Goals: 39' LLM-v11, 69' LLM-v11, 80' LLM-v11. PPO-v2 keeps a goalkeeper back 88% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 69% of the time; shots on goal 33–139.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 72.7 | 13.5 | +59.2 | **9** | 22.30 |
| 2 | GP-v3 | 3 | 1 | 1 | 1 | 40.8 | 26.5 | +14.3 | **4** | 16.52 |
| 3 | PPO-v2 | 3 | 1 | 1 | 1 | 29.7 | 27.2 | +2.5 | **4** | 4.15 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 2.8 | 78.8 | -76.0 | **0** | 7.38 |

### Round 92 (03:26)

**Showcase: LLM-v11 4 – 1 PPO-v2** (LLM-v11 wins). Video: [round-092.mp4](videos/league3/round-092.mp4)

LLM-v11 beat PPO-v2 4–1. Goals: 16' LLM-v11, 23' LLM-v11, 33' LLM-v11, 37' LLM-v11, 65' PPO-v2. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 88% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 62% of the time; shots on goal 132–35.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 70.0 | 10.8 | +59.2 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 2 | 0 | 1 | 31.3 | 23.5 | +7.8 | **6** | 4.25 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 36.3 | 30.5 | +5.8 | **3** | 16.80 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 2.7 | 75.5 | -72.8 | **0** | 7.55 |

### Round 93 (04:06)

**Showcase: PPO-v2 1 – 5 LLM-v11** (LLM-v11 wins). Video: [round-093.mp4](videos/league3/round-093.mp4)

LLM-v11 beat PPO-v2 5–1. Goals: 20' LLM-v11, 36' LLM-v11, 64' LLM-v11, 70' LLM-v11, 76' PPO-v2, 82' LLM-v11. PPO-v2 keeps a goalkeeper back 86% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 73% of the time; shots on goal 25–145.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 2 | 1 | 0 | 68.8 | 11.2 | +57.7 | **7** | 22.30 |
| 2 | PPO-v2 | 3 | 2 | 0 | 1 | 28.8 | 26.5 | +2.3 | **6** | 4.34 |
| 3 | GP-v3 | 3 | 1 | 1 | 1 | 44.5 | 18.5 | +26.0 | **4** | 17.07 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 1.7 | 87.7 | -86.0 | **0** | 7.73 |

### Round 94 (04:46)

**Showcase: LLM-v11 4 – 0 PPO-v2** (LLM-v11 wins). Video: [round-094.mp4](videos/league3/round-094.mp4)

LLM-v11 beat PPO-v2 4–0. Goals: 9' LLM-v11, 22' LLM-v11, 49' LLM-v11, 79' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 87% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 58% of the time; shots on goal 99–35.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 79.5 | 11.5 | +68.0 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 2 | 0 | 1 | 30.2 | 33.0 | -2.8 | **6** | 4.43 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 51.5 | 34.8 | +16.7 | **3** | 17.34 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 4.7 | 86.5 | -81.8 | **0** | 7.90 |

### Round 95 (05:26)

**Showcase: PPO-v2 0 – 5 LLM-v11** (LLM-v11 wins). Video: [round-095.mp4](videos/league3/round-095.mp4)

LLM-v11 beat PPO-v2 5–0. Goals: 12' LLM-v11, 21' LLM-v11, 68' LLM-v11, 80' LLM-v11, 84' LLM-v11. PPO-v2 keeps a goalkeeper back 82% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 68% of the time; shots on goal 20–122.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 78.3 | 11.5 | +66.8 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 2 | 0 | 1 | 33.3 | 31.8 | +1.5 | **6** | 4.52 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 56.0 | 36.2 | +19.8 | **3** | 17.62 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 3.0 | 91.2 | -88.2 | **0** | 8.08 |

### Round 96 (06:07)

**Showcase: GP-v3 4 – 3 LLM-v11** (GP-v3 wins). Video: [round-096.mp4](videos/league3/round-096.mp4)

GP-v3 beat LLM-v11 4–3. Goals: 7' GP-v3, 14' GP-v3, 15' LLM-v11, 45' GP-v3, 70' LLM-v11, 83' GP-v3, 90' LLM-v11. GP-v3 keeps a goalkeeper back 99% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 99% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 58% of the time; shots on goal 56–96.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GP-v3 | 3 | 3 | 0 | 0 | 60.7 | 25.0 | +35.7 | **9** | 17.90 |
| 2 | LLM-v11 | 3 | 2 | 0 | 1 | 66.7 | 20.0 | +46.7 | **6** | 22.30 |
| 3 | PPO-v2 | 3 | 1 | 0 | 2 | 32.7 | 38.5 | -5.8 | **3** | 4.62 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 3.7 | 80.2 | -76.5 | **0** | 8.26 |

### Round 97 (06:48)

**Showcase: GP-v3 6 – 4 LLM-v11** (GP-v3 wins). Video: [round-097.mp4](videos/league3/round-097.mp4)

GP-v3 beat LLM-v11 6–4. Goals: 7' GP-v3, 9' LLM-v11, 12' LLM-v11, 20' LLM-v11, 42' GP-v3, 50' GP-v3, 74' GP-v3, 79' GP-v3…. GP-v3 keeps a goalkeeper back 97% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 60% of the time; shots on goal 36–125.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GP-v3 | 3 | 3 | 0 | 0 | 54.5 | 19.2 | +35.3 | **9** | 18.17 |
| 2 | LLM-v11 | 3 | 2 | 0 | 1 | 58.8 | 16.5 | +42.3 | **6** | 22.30 |
| 3 | PPO-v2 | 3 | 1 | 0 | 2 | 25.3 | 31.5 | -6.2 | **3** | 4.71 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 4.2 | 75.7 | -71.5 | **0** | 8.44 |

### Round 98 (07:28)

**Showcase: LLM-v11 3 – 3 GP-v3** (a draw). Video: [round-098.mp4](videos/league3/round-098.mp4)

LLM-v11 and GP-v3 drew 3–3. Goals: 10' LLM-v11, 14' GP-v3, 20' GP-v3, 30' LLM-v11, 48' LLM-v11, 53' GP-v3. LLM-v11 keeps a goalkeeper back 99% of the time and runs flat out almost all the time. GP-v3 keeps a goalkeeper back 85% of the time and runs flat out almost all the time. GP-v3 kept the ball in the opposition half 53% of the time; shots on goal 89–67.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 69.8 | 19.0 | +50.8 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 2 | 0 | 1 | 35.7 | 32.5 | +3.2 | **6** | 4.80 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 63.5 | 32.2 | +31.3 | **3** | 18.45 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 3.2 | 88.5 | -85.3 | **0** | 8.62 |

### Round 99 (08:09)

**Showcase: LLM-v11 3 – 3 GP-v3** (a draw). Video: [round-099.mp4](videos/league3/round-099.mp4)

LLM-v11 and GP-v3 drew 3–3. Goals: 25' GP-v3, 28' LLM-v11, 53' GP-v3, 54' LLM-v11, 59' LLM-v11, 62' GP-v3. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 99% of the time and runs flat out almost all the time. GP-v3 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 54% of the time; shots on goal 115–56.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 75.7 | 18.5 | +57.2 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 2 | 0 | 1 | 33.8 | 30.5 | +3.3 | **6** | 4.90 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 57.8 | 32.2 | +25.7 | **3** | 18.73 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 4.0 | 90.2 | -86.2 | **0** | 8.80 |

### Round 100 (08:57)

**Showcase: LLM-v11 4 – 1 PPO-v2** (LLM-v11 wins). Video: [round-100.mp4](videos/league3/round-100.mp4)

LLM-v11 beat PPO-v2 4–1. Goals: 34' LLM-v11, 40' LLM-v11, 52' LLM-v11, 73' LLM-v11, 80' PPO-v2. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 89% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 57% of the time; shots on goal 108–36.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 76.2 | 23.2 | +53.0 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 2 | 0 | 1 | 44.0 | 32.7 | +11.3 | **6** | 5.07 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 65.7 | 54.7 | +11.0 | **3** | 18.83 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 7.3 | 82.7 | -75.3 | **0** | 8.93 |

### Round 101 (09:38)

**Showcase: LLM-v11 3 – 3 GP-v3** (a draw). Video: [round-101.mp4](videos/league3/round-101.mp4)

LLM-v11 and GP-v3 drew 3–3. Goals: 23' GP-v3, 50' LLM-v11, 71' GP-v3, 84' GP-v3, 87' LLM-v11, 90' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP-v3 keeps a goalkeeper back 98% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 67% of the time; shots on goal 119–30.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 2 | 0 | 1 | 77.3 | 20.0 | +57.3 | **6** | 22.30 |
| 2 | GP-v3 | 3 | 2 | 0 | 1 | 59.8 | 25.5 | +34.3 | **6** | 18.97 |
| 3 | PPO-v2 | 3 | 2 | 0 | 1 | 39.3 | 27.8 | +11.5 | **6** | 5.21 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 2.8 | 106.0 | -103.2 | **0** | 9.04 |

### Round 102 (10:19)

**Showcase: LLM-v11 1 – 3 GP-v3** (GP-v3 wins). Video: [round-102.mp4](videos/league3/round-102.mp4)

GP-v3 beat LLM-v11 3–1. Goals: 28' LLM-v11, 44' GP-v3, 63' GP-v3, 82' GP-v3. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. GP-v3 keeps a goalkeeper back 99% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 62% of the time; shots on goal 103–43.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 2 | 0 | 1 | 74.5 | 21.5 | +53.0 | **6** | 22.30 |
| 2 | GP-v3 | 3 | 2 | 0 | 1 | 57.3 | 21.2 | +36.2 | **6** | 19.12 |
| 3 | PPO-v2 | 3 | 2 | 0 | 1 | 36.2 | 25.5 | +10.7 | **6** | 5.40 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 2.5 | 102.3 | -99.8 | **0** | 9.18 |

### Round 103 (11:00)

**Showcase: PPO-v2 0 – 2 LLM-v11** (LLM-v11 wins). Video: [round-103.mp4](videos/league3/round-103.mp4)

LLM-v11 beat PPO-v2 2–0. Goals: 60' LLM-v11, 88' LLM-v11. PPO-v2 keeps a goalkeeper back 79% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 61% of the time; shots on goal 32–124.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 2 | 0 | 1 | 80.5 | 20.0 | +60.5 | **6** | 22.30 |
| 2 | GP-v3 | 3 | 2 | 0 | 1 | 52.2 | 22.8 | +29.3 | **6** | 19.23 |
| 3 | PPO-v2 | 3 | 2 | 0 | 1 | 41.8 | 24.2 | +17.7 | **6** | 5.54 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 3.2 | 110.7 | -107.5 | **0** | 9.28 |

### Round 104 (11:40)

**Showcase: PPO-v2 2 – 2 LLM-v11** (a draw). Video: [round-104.mp4](videos/league3/round-104.mp4)

PPO-v2 and LLM-v11 drew 2–2. Goals: 15' PPO-v2, 23' PPO-v2, 37' LLM-v11, 62' LLM-v11. PPO-v2 keeps a goalkeeper back 92% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 64% of the time; shots on goal 31–129.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 68.8 | 11.2 | +57.7 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 2 | 0 | 1 | 38.7 | 18.0 | +20.7 | **6** | 5.72 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 38.2 | 19.3 | +18.8 | **3** | 19.36 |
| 4 | PPO-goals | 3 | 0 | 0 | 3 | 1.5 | 98.7 | -97.2 | **0** | 9.41 |

### Round 105 (12:19)

**Showcase: PPO-v2 1 – 2 LLM-v11** (LLM-v11 wins). Video: [round-105.mp4](videos/league3/round-105.mp4)

LLM-v11 beat PPO-v2 2–1. Goals: 10' LLM-v11, 23' PPO-v2, 89' LLM-v11. PPO-v2 keeps a goalkeeper back 84% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 64% of the time; shots on goal 44–106.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 238.7 | 15.8 | +222.8 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 2 | 0 | 1 | 124.7 | 21.2 | +103.5 | **6** | 5.82 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 102.3 | 27.2 | +75.2 | **3** | 19.47 |
| 4 | PPO-v3 | 3 | 0 | 0 | 3 | 4.5 | 406.0 | -401.5 | **0** | 9.54 |

### Round 106 (12:59)

**Showcase: PPO-v2 1 – 2 LLM-v11** (LLM-v11 wins). Video: [round-106.mp4](videos/league3/round-106.mp4)

LLM-v11 beat PPO-v2 2–1. Goals: 4' LLM-v11, 63' LLM-v11, 83' PPO-v2. PPO-v2 keeps a goalkeeper back 87% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 71% of the time; shots on goal 27–143.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 2 | 1 | 0 | 233.8 | 20.2 | +213.7 | **7** | 22.30 |
| 2 | PPO-v2 | 3 | 2 | 0 | 1 | 123.2 | 23.5 | +99.7 | **6** | 5.89 |
| 3 | GP-v3 | 3 | 1 | 1 | 1 | 342.8 | 27.7 | +315.2 | **4** | 19.55 |
| 4 | PPO-v3 | 3 | 0 | 0 | 3 | 5.2 | 633.7 | -628.5 | **0** | 9.64 |

### Round 107 (13:38)

**Showcase: LLM-v11 4 – 4 PPO-v2** (a draw). Video: [round-107.mp4](videos/league3/round-107.mp4)

LLM-v11 and PPO-v2 drew 4–4. Goals: 3' PPO-v2, 6' PPO-v2, 22' PPO-v2, 25' LLM-v11, 61' LLM-v11, 66' LLM-v11, 69' PPO-v2, 85' LLM-v11. LLM-v11 keeps a goalkeeper back 99% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 66% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 53% of the time; shots on goal 72–66.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 2 | 1 | 0 | 69.8 | 17.7 | +52.2 | **7** | 22.30 |
| 2 | PPO-v2 | 3 | 2 | 0 | 1 | 46.3 | 24.0 | +22.3 | **6** | 6.07 |
| 3 | GP-v3 | 3 | 1 | 1 | 1 | 45.7 | 29.0 | +16.7 | **4** | 19.66 |
| 4 | PPO-v3 | 3 | 0 | 0 | 3 | 3.3 | 94.5 | -91.2 | **0** | 9.64 |

### Round 108 (14:19)

**Showcase: PPO-v2 2 – 3 LLM-v11** (LLM-v11 wins). Video: [round-108.mp4](videos/league3/round-108.mp4)

LLM-v11 beat PPO-v2 3–2. Goals: 5' PPO-v2, 14' LLM-v11, 20' LLM-v11, 31' PPO-v2, 57' LLM-v11. PPO-v2 keeps a goalkeeper back 79% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 56% of the time; shots on goal 39–88.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 74.8 | 17.3 | +57.5 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 2 | 0 | 1 | 46.8 | 20.7 | +26.2 | **6** | 6.30 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 41.3 | 33.8 | +7.5 | **3** | 19.81 |
| 4 | PPO-v3 | 3 | 0 | 0 | 3 | 3.3 | 94.5 | -91.2 | **0** | 9.78 |

### Round 109 (14:59)

**Showcase: LLM-v11 2 – 2 PPO-v2** (a draw). Video: [round-109.mp4](videos/league3/round-109.mp4)

LLM-v11 and PPO-v2 drew 2–2. Goals: 19' PPO-v2, 22' PPO-v2, 30' LLM-v11, 86' LLM-v11. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 84% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 61% of the time; shots on goal 113–41.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 3 | 3 | 0 | 0 | 51.7 | 19.5 | +32.2 | **9** | 6.53 |
| 2 | LLM-v11 | 3 | 2 | 0 | 1 | 58.8 | 21.8 | +37.0 | **6** | 22.30 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 48.2 | 34.7 | +13.5 | **3** | 19.95 |
| 4 | PPO-v3 | 3 | 0 | 0 | 3 | 4.8 | 87.5 | -82.7 | **0** | 9.96 |

### Round 110 (15:40)

**Showcase: PPO-v2 2 – 1 LLM-v11** (PPO-v2 wins). Video: [round-110.mp4](videos/league3/round-110.mp4)

PPO-v2 beat LLM-v11 2–1. Goals: 55' PPO-v2, 64' PPO-v2, 68' LLM-v11. PPO-v2 keeps a goalkeeper back 85% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 64% of the time; shots on goal 39–121.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LLM-v11 | 3 | 3 | 0 | 0 | 91.2 | 9.5 | +81.7 | **9** | 22.30 |
| 2 | PPO-v2 | 3 | 2 | 0 | 1 | 40.5 | 16.5 | +24.0 | **6** | 6.76 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 38.8 | 17.5 | +21.3 | **3** | 20.09 |
| 4 | PPO-v3 | 3 | 0 | 0 | 3 | 4.5 | 131.5 | -127.0 | **0** | 10.14 |

### Round 111 (16:20)

**Showcase: PPO-v2 3 – 2 LLM-v11** (PPO-v2 wins). Video: [round-111.mp4](videos/league3/round-111.mp4)

PPO-v2 beat LLM-v11 3–2. Goals: 19' LLM-v11, 43' PPO-v2, 61' PPO-v2, 76' PPO-v2, 87' LLM-v11. PPO-v2 keeps a goalkeeper back 81% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 55% of the time; shots on goal 50–89.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 3 | 3 | 0 | 0 | 50.8 | 15.2 | +35.7 | **9** | 6.99 |
| 2 | LLM-v11 | 3 | 2 | 0 | 1 | 109.8 | 21.2 | +88.7 | **6** | 22.30 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 62.7 | 24.0 | +38.7 | **3** | 20.23 |
| 4 | PPO-v3 | 3 | 0 | 0 | 3 | 4.5 | 167.5 | -163.0 | **0** | 10.32 |

### Round 112 (17:01)

**Showcase: LLM-v11 1 – 3 PPO-v2** (PPO-v2 wins). Video: [round-112.mp4](videos/league3/round-112.mp4)

PPO-v2 beat LLM-v11 3–1. Goals: 45' PPO-v2, 48' PPO-v2, 53' LLM-v11, 57' PPO-v2. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 82% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 62% of the time; shots on goal 119–41.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 3 | 3 | 0 | 0 | 53.0 | 20.0 | +33.0 | **9** | 7.22 |
| 2 | LLM-v11 | 3 | 2 | 0 | 1 | 110.5 | 18.2 | +92.3 | **6** | 22.30 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 48.8 | 26.0 | +22.8 | **3** | 20.37 |
| 4 | PPO-v3 | 3 | 0 | 0 | 3 | 4.7 | 152.8 | -148.2 | **0** | 10.50 |

### Round 113 (17:42)

**Showcase: PPO-v2 2 – 2 LLM-v11** (a draw). Video: [round-113.mp4](videos/league3/round-113.mp4)

PPO-v2 and LLM-v11 drew 2–2. Goals: 21' LLM-v11, 27' LLM-v11, 42' PPO-v2, 61' PPO-v2. PPO-v2 keeps a goalkeeper back 67% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 58% of the time; shots on goal 51–110.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 3 | 3 | 0 | 0 | 56.7 | 16.5 | +40.2 | **9** | 7.44 |
| 2 | LLM-v11 | 3 | 2 | 0 | 1 | 108.7 | 26.3 | +82.3 | **6** | 22.30 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 50.7 | 46.0 | +4.7 | **3** | 20.51 |
| 4 | PPO-v3 | 3 | 0 | 0 | 3 | 7.5 | 134.7 | -127.2 | **0** | 10.68 |

### Round 114 (18:24)

**Showcase: PPO-v2 2 – 1 LLM-v11** (PPO-v2 wins). Video: [round-114.mp4](videos/league3/round-114.mp4)

PPO-v2 beat LLM-v11 2–1. Goals: 38' LLM-v11, 73' PPO-v2, 83' PPO-v2. PPO-v2 keeps a goalkeeper back 80% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 61% of the time; shots on goal 51–126.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 3 | 3 | 0 | 0 | 61.8 | 15.0 | +46.8 | **9** | 7.62 |
| 2 | LLM-v11 | 3 | 2 | 0 | 1 | 89.7 | 14.5 | +75.2 | **6** | 22.30 |
| 3 | GP-v3 | 3 | 1 | 0 | 2 | 40.5 | 30.0 | +10.5 | **3** | 20.62 |
| 4 | PPO-v3 | 3 | 0 | 0 | 3 | 1.2 | 133.7 | -132.5 | **0** | 10.82 |

### Round 115 (19:09)

**Showcase: PPO-champ-9 1 – 2 PPO-v2** (PPO-v2 wins). Video: [round-115.mp4](videos/league3/round-115.mp4)

PPO-v2 beat PPO-champ-9 2–1. Goals: 4' PPO-v2, 10' PPO-champ-9, 64' PPO-v2. PPO-champ-9 keeps a goalkeeper back 74% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 70% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 53% of the time; shots on goal 59–89.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 57.2 | 17.3 | +39.8 | **12** | 7.85 |
| 2 | PPO-champ-9 | 4 | 3 | 0 | 1 | 54.7 | 20.3 | +34.3 | **9** | 6.62 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 95.0 | 27.2 | +67.8 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 37.8 | 30.5 | +7.3 | **3** | 20.77 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 2.5 | 151.8 | -149.3 | **0** | 11.01 |

### Round 116 (19:55)

**Showcase: PPO-champ-9 0 – 1 PPO-v2** (PPO-v2 wins). Video: [round-116.mp4](videos/league3/round-116.mp4)

PPO-v2 beat PPO-champ-9 1–0. Goals: 54' PPO-v2. PPO-champ-9 keeps a goalkeeper back 78% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 67% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 59% of the time; shots on goal 57–55.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 3 | 1 | 0 | 64.0 | 22.3 | +41.7 | **10** | 8.10 |
| 2 | PPO-champ-9 | 4 | 2 | 2 | 0 | 59.5 | 21.8 | +37.7 | **8** | 6.62 |
| 3 | LLM-v11 | 4 | 2 | 1 | 1 | 116.3 | 27.0 | +89.3 | **7** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 51.5 | 32.7 | +18.8 | **3** | 20.93 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 3.0 | 190.5 | -187.5 | **0** | 11.21 |

### Round 117 (20:39)

**Showcase: PPO-champ-9 1 – 1 PPO-v2** (a draw). Video: [round-117.mp4](videos/league3/round-117.mp4)

PPO-champ-9 and PPO-v2 drew 1–1. Goals: 54' PPO-champ-9, 69' PPO-v2. PPO-champ-9 keeps a goalkeeper back 85% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 84% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 58% of the time; shots on goal 41–95.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 65.0 | 21.5 | +43.5 | **12** | 8.35 |
| 2 | PPO-champ-9 | 4 | 3 | 0 | 1 | 61.2 | 21.8 | +39.3 | **9** | 6.62 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 95.3 | 31.2 | +64.2 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 65.0 | 49.3 | +15.7 | **3** | 21.08 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 2.7 | 165.3 | -162.7 | **0** | 11.41 |

### Round 118 (21:22)

**Showcase: PPO-v2 1 – 1 PPO-champ-9** (a draw). Video: [round-118.mp4](videos/league3/round-118.mp4)

PPO-v2 and PPO-champ-9 drew 1–1. Goals: 26' PPO-v2, 43' PPO-champ-9. PPO-v2 keeps a goalkeeper back 64% of the time and runs flat out almost all the time. PPO-champ-9 keeps a goalkeeper back 77% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 66% of the time; shots on goal 100–38.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 63.7 | 21.0 | +42.7 | **12** | 8.59 |
| 2 | PPO-champ-9 | 4 | 3 | 0 | 1 | 53.3 | 22.3 | +31.0 | **9** | 6.62 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 84.2 | 29.8 | +54.3 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 50.0 | 37.7 | +12.3 | **3** | 21.24 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 3.7 | 144.0 | -140.3 | **0** | 11.60 |

### Round 119 (22:06)

**Showcase: PPO-v2 1 – 1 PPO-champ-9** (a draw). Video: [round-119.mp4](videos/league3/round-119.mp4)

PPO-v2 and PPO-champ-9 drew 1–1. Goals: 20' PPO-v2, 76' PPO-champ-9. PPO-v2 keeps a goalkeeper back 78% of the time and runs flat out almost all the time. PPO-champ-9 keeps a goalkeeper back 69% of the time and runs flat out almost all the time. PPO-champ-9 kept the ball in the opposition half 52% of the time; shots on goal 53–73.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 66.3 | 20.7 | +45.7 | **12** | 8.83 |
| 2 | PPO-champ-9 | 4 | 3 | 0 | 1 | 60.7 | 20.7 | +40.0 | **9** | 6.62 |
| 3 | GP-v3 | 4 | 2 | 0 | 2 | 68.8 | 44.8 | +24.0 | **6** | 21.39 |
| 4 | LLM-v11 | 4 | 1 | 0 | 3 | 85.5 | 36.2 | +49.3 | **3** | 22.30 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 3.5 | 162.5 | -159.0 | **0** | 11.80 |

### Round 120 (22:49)

**Showcase: PPO-champ-9 2 – 1 PPO-v2** (PPO-champ-9 wins). Video: [round-120.mp4](videos/league3/round-120.mp4)

PPO-champ-9 beat PPO-v2 2–1. Goals: 28' PPO-champ-9, 51' PPO-champ-9, 74' PPO-v2. PPO-champ-9 keeps a goalkeeper back 81% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 76% of the time and runs flat out almost all the time. PPO-champ-9 kept the ball in the opposition half 51% of the time; shots on goal 57–54.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-champ-9 | 4 | 4 | 0 | 0 | 64.8 | 20.2 | +44.7 | **12** | 6.62 |
| 2 | PPO-v2 | 4 | 3 | 0 | 1 | 70.7 | 25.0 | +45.7 | **9** | 9.07 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 96.0 | 33.7 | +62.3 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 68.3 | 45.7 | +22.7 | **3** | 21.54 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 4.3 | 179.7 | -175.3 | **0** | 11.99 |

### Round 121 (23:33)

**Showcase: PPO-champ-9 1 – 1 PPO-v2** (a draw). Video: [round-121.mp4](videos/league3/round-121.mp4)

PPO-champ-9 and PPO-v2 drew 1–1. Goals: 40' PPO-champ-9, 80' PPO-v2. PPO-champ-9 keeps a goalkeeper back 75% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 74% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 52% of the time; shots on goal 55–112.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 77.3 | 22.3 | +55.0 | **12** | 9.32 |
| 2 | PPO-champ-9 | 4 | 3 | 0 | 1 | 64.5 | 20.0 | +44.5 | **9** | 6.62 |
| 3 | GP-v3 | 4 | 2 | 0 | 2 | 72.3 | 45.3 | +27.0 | **6** | 21.69 |
| 4 | LLM-v11 | 4 | 1 | 0 | 3 | 90.3 | 36.5 | +53.8 | **3** | 22.30 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 3.8 | 184.2 | -180.3 | **0** | 12.19 |

### Round 122 (00:18)

**Showcase: PPO-champ-9 1 – 1 PPO-v2** (a draw). Video: [round-122.mp4](videos/league3/round-122.mp4)

PPO-champ-9 and PPO-v2 drew 1–1. Goals: 25' PPO-v2, 43' PPO-champ-9. PPO-champ-9 keeps a goalkeeper back 75% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 88% of the time and runs flat out almost all the time. PPO-champ-9 kept the ball in the opposition half 53% of the time; shots on goal 75–58.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 62.8 | 25.7 | +37.2 | **12** | 9.56 |
| 2 | PPO-champ-9 | 4 | 3 | 0 | 1 | 59.0 | 25.2 | +33.8 | **9** | 6.62 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 84.0 | 36.7 | +47.3 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 72.7 | 46.3 | +26.3 | **3** | 21.85 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 5.2 | 149.8 | -144.7 | **0** | 12.39 |

### Round 123 (01:03)

**Showcase: PPO-champ-9 1 – 2 PPO-v2** (PPO-v2 wins). Video: [round-123.mp4](videos/league3/round-123.mp4)

PPO-v2 beat PPO-champ-9 2–1. Goals: 8' PPO-champ-9, 19' PPO-v2, 29' PPO-v2. PPO-champ-9 keeps a goalkeeper back 68% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 85% of the time and runs flat out almost all the time. PPO-champ-9 kept the ball in the opposition half 57% of the time; shots on goal 91–48.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 3 | 1 | 0 | 65.8 | 24.0 | +41.8 | **10** | 9.81 |
| 2 | PPO-champ-9 | 4 | 3 | 1 | 0 | 60.5 | 23.3 | +37.2 | **10** | 6.62 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 89.7 | 35.3 | +54.3 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 71.0 | 46.0 | +25.0 | **3** | 22.01 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 3.3 | 161.7 | -158.3 | **0** | 12.59 |

### Round 124 (01:47)

**Showcase: PPO-v2 3 – 2 LLM-v11** (PPO-v2 wins). Video: [round-124.mp4](videos/league3/round-124.mp4)

PPO-v2 beat LLM-v11 3–2. Goals: 13' PPO-v2, 35' LLM-v11, 39' PPO-v2, 51' PPO-v2, 85' LLM-v11. PPO-v2 keeps a goalkeeper back 82% of the time and runs flat out almost all the time. LLM-v11 is scattered all over the pitch, keeps a goalkeeper back 100% of the time and runs flat out almost all the time. LLM-v11 kept the ball in the opposition half 54% of the time; shots on goal 70–98.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 54.2 | 21.7 | +32.5 | **12** | 10.06 |
| 2 | LLM-v11 | 4 | 2 | 1 | 1 | 78.2 | 31.7 | +46.5 | **7** | 22.30 |
| 3 | PPO-champ-9 | 4 | 2 | 1 | 1 | 59.5 | 24.3 | +35.2 | **7** | 6.62 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 53.5 | 48.7 | +4.8 | **3** | 22.16 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 5.2 | 124.2 | -119.0 | **0** | 12.79 |

### Round 125 (02:33)

**Showcase: PPO-champ-9 1 – 2 PPO-v2** (PPO-v2 wins). Video: [round-125.mp4](videos/league3/round-125.mp4)

PPO-v2 beat PPO-champ-9 2–1. Goals: 41' PPO-v2, 67' PPO-v2, 70' PPO-champ-9. PPO-champ-9 keeps a goalkeeper back 69% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 88% of the time and runs flat out almost all the time. PPO-champ-9 kept the ball in the opposition half 51% of the time; shots on goal 47–55.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 58.5 | 19.2 | +39.3 | **12** | 10.31 |
| 2 | PPO-champ-9 | 4 | 3 | 0 | 1 | 59.5 | 24.5 | +35.0 | **9** | 6.62 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 75.3 | 32.5 | +42.8 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 55.2 | 51.5 | +3.7 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 3.2 | 124.0 | -120.8 | **0** | 12.99 |

### Round 126 (03:16)

**Showcase: PPO-v2 0 – 1 PPO-champ-10** (PPO-champ-10 wins). Video: [round-126.mp4](videos/league3/round-126.mp4)

PPO-champ-10 beat PPO-v2 1–0. Goals: 75' PPO-champ-10. PPO-v2 keeps a goalkeeper back 88% of the time and runs flat out almost all the time. PPO-champ-10 keeps a goalkeeper back 82% of the time and runs flat out almost all the time. PPO-champ-10 kept the ball in the opposition half 56% of the time; shots on goal 47–98.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-champ-10 | 4 | 4 | 0 | 0 | 52.5 | 21.7 | +30.8 | **12** | 9.31 |
| 2 | PPO-v2 | 4 | 3 | 0 | 1 | 56.3 | 16.8 | +39.5 | **9** | 10.54 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 77.8 | 36.0 | +41.8 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 53.5 | 53.2 | +0.3 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 4.3 | 116.8 | -112.5 | **0** | 13.18 |

### Round 127 (04:00)

**Showcase: PPO-champ-10 0 – 1 PPO-v2** (PPO-v2 wins). Video: [round-127.mp4](videos/league3/round-127.mp4)

PPO-v2 beat PPO-champ-10 1–0. Goals: 7' PPO-v2. PPO-champ-10 keeps a goalkeeper back 86% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 81% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 56% of the time; shots on goal 39–67.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 54.3 | 18.2 | +36.2 | **12** | 10.86 |
| 2 | PPO-champ-10 | 4 | 3 | 0 | 1 | 51.2 | 20.2 | +31.0 | **9** | 9.31 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 99.0 | 28.2 | +70.8 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 52.3 | 53.0 | -0.7 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 4.8 | 142.2 | -137.3 | **0** | 13.45 |

### Round 128 (04:44)

**Showcase: PPO-champ-10 0 – 1 PPO-v2** (PPO-v2 wins). Video: [round-128.mp4](videos/league3/round-128.mp4)

PPO-v2 beat PPO-champ-10 1–0. Goals: 59' PPO-v2. PPO-champ-10 keeps a goalkeeper back 89% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 85% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 55% of the time; shots on goal 58–88.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 59.5 | 21.2 | +38.3 | **12** | 11.18 |
| 2 | PPO-champ-10 | 4 | 3 | 0 | 1 | 59.8 | 19.3 | +40.5 | **9** | 9.31 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 106.5 | 32.7 | +73.8 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 57.2 | 59.0 | -1.8 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 5.3 | 156.2 | -150.8 | **0** | 13.72 |

### Round 129 (05:27)

**Showcase: PPO-champ-10 1 – 1 PPO-v2** (a draw). Video: [round-129.mp4](videos/league3/round-129.mp4)

PPO-champ-10 and PPO-v2 drew 1–1. Goals: 12' PPO-v2, 49' PPO-champ-10. PPO-champ-10 keeps a goalkeeper back 79% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 84% of the time and runs flat out almost all the time. PPO-champ-10 kept the ball in the opposition half 50% of the time; shots on goal 94–76.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-champ-10 | 4 | 3 | 1 | 0 | 57.5 | 18.2 | +39.3 | **10** | 9.31 |
| 2 | PPO-v2 | 4 | 3 | 1 | 0 | 55.0 | 19.2 | +35.8 | **10** | 11.50 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 107.2 | 34.8 | +72.3 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 59.5 | 51.8 | +7.7 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 3.3 | 158.5 | -155.2 | **0** | 13.99 |

### Round 130 (06:10)

**Showcase: PPO-v2 1 – 1 PPO-champ-10** (a draw). Video: [round-130.mp4](videos/league3/round-130.mp4)

PPO-v2 and PPO-champ-10 drew 1–1. Goals: 18' PPO-champ-10, 78' PPO-v2. PPO-v2 keeps a goalkeeper back 89% of the time and runs flat out almost all the time. PPO-champ-10 keeps a goalkeeper back 88% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 55% of the time; shots on goal 80–105.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 56.7 | 16.2 | +40.5 | **12** | 11.81 |
| 2 | PPO-champ-10 | 4 | 3 | 0 | 1 | 53.2 | 19.3 | +33.8 | **9** | 9.31 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 105.0 | 33.3 | +71.7 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 53.0 | 54.5 | -1.5 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 3.3 | 147.8 | -144.5 | **0** | 14.25 |

### Round 131 (06:52)

**Showcase: PPO-champ-10 1 – 0 PPO-v2** (PPO-champ-10 wins). Video: [round-131.mp4](videos/league3/round-131.mp4)

PPO-champ-10 beat PPO-v2 1–0. Goals: 21' PPO-champ-10. PPO-champ-10 keeps a goalkeeper back 90% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 87% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 57% of the time; shots on goal 59–74.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 65.3 | 14.5 | +50.8 | **12** | 12.12 |
| 2 | PPO-champ-10 | 4 | 3 | 0 | 1 | 62.5 | 21.8 | +40.7 | **9** | 9.31 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 113.2 | 38.2 | +75.0 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 52.2 | 63.5 | -11.3 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 3.5 | 158.7 | -155.2 | **0** | 14.52 |

### Round 132 (07:34)

**Showcase: PPO-champ-10 1 – 2 PPO-v2** (PPO-v2 wins). Video: [round-132.mp4](videos/league3/round-132.mp4)

PPO-v2 beat PPO-champ-10 2–1. Goals: 13' PPO-v2, 40' PPO-champ-10, 63' PPO-v2. PPO-champ-10 keeps a goalkeeper back 91% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 85% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 58% of the time; shots on goal 42–92.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 3 | 1 | 0 | 64.7 | 15.5 | +49.2 | **10** | 12.43 |
| 2 | PPO-champ-10 | 4 | 3 | 1 | 0 | 60.8 | 19.8 | +41.0 | **10** | 9.31 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 109.8 | 38.0 | +71.8 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 49.0 | 60.8 | -11.8 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 5.5 | 155.7 | -150.2 | **0** | 14.78 |

### Round 133 (08:17)

**Showcase: PPO-champ-11 1 – 1 PPO-v2** (a draw). Video: [round-133.mp4](videos/league3/round-133.mp4)

PPO-champ-11 and PPO-v2 drew 1–1. Goals: 2' PPO-v2, 32' PPO-champ-11. PPO-champ-11 keeps a goalkeeper back 91% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 89% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 56% of the time; shots on goal 53–98.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 65.3 | 17.0 | +48.3 | **12** | 12.75 |
| 2 | PPO-champ-11 | 4 | 3 | 0 | 1 | 71.3 | 20.0 | +51.3 | **9** | 11.43 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 121.0 | 37.8 | +83.2 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 59.0 | 60.8 | -1.8 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 4.5 | 185.5 | -181.0 | **0** | 15.04 |

### Round 134 (09:00)

**Showcase: PPO-v2 0 – 0 PPO-champ-11** (a draw). Video: [round-134.mp4](videos/league3/round-134.mp4)

PPO-v2 and PPO-champ-11 drew 0–0. PPO-v2 keeps a goalkeeper back 80% of the time and runs flat out almost all the time. PPO-champ-11 keeps a goalkeeper back 91% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 56% of the time; shots on goal 98–56.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 3 | 1 | 0 | 62.3 | 18.7 | +43.7 | **10** | 13.06 |
| 2 | PPO-champ-11 | 4 | 3 | 1 | 0 | 61.0 | 17.3 | +43.7 | **10** | 11.43 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 125.0 | 37.7 | +87.3 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 60.2 | 56.5 | +3.7 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 5.0 | 183.3 | -178.3 | **0** | 15.31 |

### Round 135 (09:44)

**Showcase: PPO-champ-11 1 – 0 PPO-v2** (PPO-champ-11 wins). Video: [round-135.mp4](videos/league3/round-135.mp4)

PPO-champ-11 beat PPO-v2 1–0. Goals: 68' PPO-champ-11. PPO-champ-11 keeps a goalkeeper back 78% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 85% of the time and runs flat out almost all the time. PPO-champ-11 kept the ball in the opposition half 56% of the time; shots on goal 96–38.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 68.5 | 15.8 | +52.7 | **12** | 13.38 |
| 2 | PPO-champ-11 | 4 | 3 | 0 | 1 | 63.8 | 17.0 | +46.8 | **9** | 11.43 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 122.8 | 33.2 | +89.7 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 62.0 | 55.5 | +6.5 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 4.0 | 199.7 | -195.7 | **0** | 15.58 |

### Round 136 (10:28)

**Showcase: PPO-champ-11 1 – 1 PPO-v2** (a draw). Video: [round-136.mp4](videos/league3/round-136.mp4)

PPO-champ-11 and PPO-v2 drew 1–1. Goals: 75' PPO-v2, 82' PPO-champ-11. PPO-champ-11 keeps a goalkeeper back 88% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 63% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 57% of the time; shots on goal 59–93.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 70.3 | 23.3 | +47.0 | **12** | 13.70 |
| 2 | PPO-champ-11 | 4 | 3 | 0 | 1 | 69.5 | 19.0 | +50.5 | **9** | 11.43 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 132.7 | 36.0 | +96.7 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 67.3 | 61.7 | +5.7 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 5.5 | 205.3 | -199.8 | **0** | 15.85 |

### Round 137 (11:12)

**Showcase: PPO-champ-11 1 – 1 PPO-v2** (a draw). Video: [round-137.mp4](videos/league3/round-137.mp4)

PPO-champ-11 and PPO-v2 drew 1–1. Goals: 33' PPO-champ-11, 84' PPO-v2. PPO-champ-11 keeps a goalkeeper back 85% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 79% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 66% of the time; shots on goal 30–114.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 65.7 | 18.5 | +47.2 | **12** | 14.03 |
| 2 | PPO-champ-11 | 4 | 3 | 0 | 1 | 64.7 | 17.3 | +47.3 | **9** | 11.43 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 123.2 | 37.0 | +86.2 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 70.0 | 55.2 | +14.8 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 4.7 | 200.2 | -195.5 | **0** | 16.13 |

### Round 138 (11:58)

**Showcase: PPO-champ-11 0 – 0 PPO-v2** (a draw). Video: [round-138.mp4](videos/league3/round-138.mp4)

PPO-champ-11 and PPO-v2 drew 0–0. PPO-champ-11 keeps a goalkeeper back 82% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 63% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 53% of the time; shots on goal 101–73.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-champ-11 | 4 | 3 | 1 | 0 | 67.3 | 14.8 | +52.5 | **10** | 11.43 |
| 2 | PPO-v2 | 4 | 3 | 1 | 0 | 68.8 | 20.8 | +48.0 | **10** | 14.35 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 109.3 | 39.5 | +69.8 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 56.0 | 59.0 | -3.0 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 7.0 | 174.3 | -167.3 | **0** | 16.40 |

### Round 139 (12:46)

**Showcase: PPO-v2 2 – 1 PPO-champ-11** (PPO-v2 wins). Video: [round-139.mp4](videos/league3/round-139.mp4)

PPO-v2 beat PPO-champ-11 2–1. Goals: 4' PPO-v2, 22' PPO-v2, 44' PPO-champ-11. PPO-v2 keeps a goalkeeper back 73% of the time and runs flat out almost all the time. PPO-champ-11 keeps a goalkeeper back 81% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 54% of the time; shots on goal 65–74.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 3 | 1 | 0 | 73.7 | 19.8 | +53.8 | **10** | 14.68 |
| 2 | PPO-champ-11 | 4 | 3 | 1 | 0 | 68.5 | 15.5 | +53.0 | **10** | 11.43 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 108.0 | 38.7 | +69.3 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 53.8 | 59.2 | -5.3 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 6.0 | 176.8 | -170.8 | **0** | 16.68 |

### Round 140 (13:31)

**Showcase: PPO-v2 1 – 1 PPO-champ-11** (a draw). Video: [round-140.mp4](videos/league3/round-140.mp4)

PPO-v2 and PPO-champ-11 drew 1–1. Goals: 53' PPO-v2, 64' PPO-champ-11. PPO-v2 keeps a goalkeeper back 79% of the time and runs flat out almost all the time. PPO-champ-11 keeps a goalkeeper back 74% of the time and runs flat out almost all the time. PPO-champ-11 kept the ball in the opposition half 60% of the time; shots on goal 60–75.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 61.8 | 21.7 | +40.2 | **12** | 15.03 |
| 2 | PPO-champ-11 | 4 | 3 | 0 | 1 | 60.5 | 19.0 | +41.5 | **9** | 11.43 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 101.7 | 40.7 | +61.0 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 49.0 | 57.5 | -8.5 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 5.0 | 139.2 | -134.2 | **0** | 16.98 |

### Round 141 (14:14)

**Showcase: PPO-v2 1 – 1 PPO-champ-11** (a draw). Video: [round-141.mp4](videos/league3/round-141.mp4)

PPO-v2 and PPO-champ-11 drew 1–1. Goals: 17' PPO-v2, 38' PPO-champ-11. PPO-v2 keeps a goalkeeper back 77% of the time and runs flat out almost all the time. PPO-champ-11 keeps a goalkeeper back 70% of the time and runs flat out almost all the time. PPO-champ-11 kept the ball in the opposition half 53% of the time; shots on goal 65–80.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 64.8 | 18.0 | +46.8 | **12** | 15.36 |
| 2 | PPO-champ-11 | 4 | 3 | 0 | 1 | 57.0 | 17.8 | +39.2 | **9** | 11.43 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 104.0 | 38.7 | +65.3 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 53.5 | 53.8 | -0.3 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 5.8 | 156.8 | -151.0 | **0** | 17.25 |

### Round 142 (14:58)

**Showcase: PPO-v2 1 – 1 PPO-champ-11** (a draw). Video: [round-142.mp4](videos/league3/round-142.mp4)

PPO-v2 and PPO-champ-11 drew 1–1. Goals: 3' PPO-v2, 58' PPO-champ-11. PPO-v2 keeps a goalkeeper back 88% of the time and runs flat out almost all the time. PPO-champ-11 keeps a goalkeeper back 87% of the time and runs flat out almost all the time. PPO-champ-11 kept the ball in the opposition half 51% of the time; shots on goal 68–66.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-champ-11 | 4 | 4 | 0 | 0 | 61.3 | 17.5 | +43.8 | **12** | 11.43 |
| 2 | PPO-v2 | 4 | 3 | 0 | 1 | 62.2 | 19.5 | +42.7 | **9** | 15.67 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 113.2 | 36.0 | +77.2 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 52.5 | 55.0 | -2.5 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 6.2 | 167.3 | -161.2 | **0** | 17.51 |

### Round 143 (15:40)

**Showcase: PPO-champ-11 1 – 1 PPO-v2** (a draw). Video: [round-143.mp4](videos/league3/round-143.mp4)

PPO-champ-11 and PPO-v2 drew 1–1. Goals: 10' PPO-v2, 20' PPO-champ-11. PPO-champ-11 keeps a goalkeeper back 81% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 92% of the time and runs flat out almost all the time. PPO-champ-11 kept the ball in the opposition half 66% of the time; shots on goal 120–39.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-champ-11 | 4 | 3 | 1 | 0 | 65.2 | 16.8 | +48.3 | **10** | 11.43 |
| 2 | PPO-v2 | 4 | 3 | 1 | 0 | 62.3 | 19.0 | +43.3 | **10** | 15.98 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 106.2 | 36.5 | +69.7 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 51.8 | 58.2 | -6.3 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 3.8 | 158.8 | -155.0 | **0** | 17.78 |

### Round 144 (16:23)

**Showcase: PPO-champ-11 0 – 0 PPO-v2** (a draw). Video: [round-144.mp4](videos/league3/round-144.mp4)

PPO-champ-11 and PPO-v2 drew 0–0. PPO-champ-11 keeps a goalkeeper back 82% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 93% of the time and runs flat out almost all the time. PPO-champ-11 kept the ball in the opposition half 62% of the time; shots on goal 113–67.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-champ-11 | 4 | 3 | 1 | 0 | 60.5 | 18.8 | +41.7 | **10** | 11.43 |
| 2 | PPO-v2 | 4 | 3 | 1 | 0 | 58.5 | 18.3 | +40.2 | **10** | 16.30 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 94.5 | 33.5 | +61.0 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 52.7 | 49.5 | +3.2 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 5.3 | 151.3 | -146.0 | **0** | 18.04 |

### Round 145 (17:06)

**Showcase: PPO-champ-11 1 – 1 PPO-v2** (a draw). Video: [round-145.mp4](videos/league3/round-145.mp4)

PPO-champ-11 and PPO-v2 drew 1–1. Goals: 10' PPO-v2, 26' PPO-champ-11. PPO-champ-11 keeps a goalkeeper back 87% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 84% of the time and runs flat out almost all the time. PPO-champ-11 kept the ball in the opposition half 55% of the time; shots on goal 90–63.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 3 | 1 | 0 | 63.5 | 17.5 | +46.0 | **10** | 16.61 |
| 2 | PPO-champ-11 | 4 | 3 | 1 | 0 | 55.5 | 15.0 | +40.5 | **10** | 11.43 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 93.8 | 34.5 | +59.3 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 52.7 | 56.5 | -3.8 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 8.8 | 150.8 | -142.0 | **0** | 18.31 |

### Round 146 (17:51)

**Showcase: PPO-v2 1 – 1 PPO-champ-11** (a draw). Video: [round-146.mp4](videos/league3/round-146.mp4)

PPO-v2 and PPO-champ-11 drew 1–1. Goals: 75' PPO-champ-11, 79' PPO-v2. PPO-v2 keeps a goalkeeper back 96% of the time and runs flat out almost all the time. PPO-champ-11 keeps a goalkeeper back 78% of the time and runs flat out almost all the time. PPO-champ-11 kept the ball in the opposition half 62% of the time; shots on goal 45–97.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-champ-11 | 4 | 3 | 1 | 0 | 60.5 | 15.0 | +45.5 | **10** | 11.43 |
| 2 | PPO-v2 | 4 | 3 | 1 | 0 | 57.5 | 17.2 | +40.3 | **10** | 16.93 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 97.5 | 34.0 | +63.5 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 49.2 | 55.0 | -5.8 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 3.7 | 147.2 | -143.5 | **0** | 18.57 |

### Round 147 (18:35)

**Showcase: PPO-v2 0 – 0 PPO-champ-11** (a draw). Video: [round-147.mp4](videos/league3/round-147.mp4)

PPO-v2 and PPO-champ-11 drew 0–0. PPO-v2 keeps a goalkeeper back 83% of the time and runs flat out almost all the time. PPO-champ-11 keeps a goalkeeper back 86% of the time and runs flat out almost all the time. PPO-champ-11 kept the ball in the opposition half 58% of the time; shots on goal 47–101.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-champ-11 | 4 | 3 | 1 | 0 | 61.3 | 13.7 | +47.7 | **10** | 11.43 |
| 2 | PPO-v2 | 4 | 3 | 1 | 0 | 54.5 | 20.3 | +34.2 | **10** | 17.26 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 92.5 | 38.3 | +54.2 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 54.3 | 48.0 | +6.3 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 6.2 | 148.5 | -142.3 | **0** | 18.85 |

### Round 148 (19:19)

**Showcase: PPO-champ-11 0 – 0 PPO-v2** (a draw). Video: [round-148.mp4](videos/league3/round-148.mp4)

PPO-champ-11 and PPO-v2 drew 0–0. PPO-champ-11 keeps a goalkeeper back 88% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 71% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 61% of the time; shots on goal 38–73.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-champ-11 | 4 | 3 | 1 | 0 | 55.8 | 16.5 | +39.3 | **10** | 11.43 |
| 2 | PPO-v2 | 4 | 3 | 1 | 0 | 57.8 | 21.2 | +36.7 | **10** | 17.58 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 83.0 | 37.3 | +45.7 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 39.8 | 56.8 | -17.0 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 9.0 | 113.7 | -104.7 | **0** | 19.12 |

### Round 149 (20:02)

**Showcase: PPO-v2 0 – 0 PPO-champ-11** (a draw). Video: [round-149.mp4](videos/league3/round-149.mp4)

PPO-v2 and PPO-champ-11 drew 0–0. PPO-v2 keeps a goalkeeper back 90% of the time and runs flat out almost all the time. PPO-champ-11 keeps a goalkeeper back 84% of the time and runs flat out almost all the time. PPO-v2 kept the ball in the opposition half 51% of the time; shots on goal 64–59.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-v2 | 4 | 4 | 0 | 0 | 60.2 | 19.3 | +40.8 | **12** | 17.91 |
| 2 | PPO-champ-11 | 4 | 3 | 0 | 1 | 67.2 | 20.7 | +46.5 | **9** | 11.43 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 83.0 | 40.5 | +42.5 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 45.7 | 59.2 | -13.5 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 6.3 | 122.7 | -116.3 | **0** | 19.40 |

### Round 150 (20:46)

**Showcase: PPO-champ-11 2 – 2 PPO-v2** (a draw). Video: [round-150.mp4](videos/league3/round-150.mp4)

PPO-champ-11 and PPO-v2 drew 2–2. Goals: 15' PPO-champ-11, 26' PPO-v2, 61' PPO-champ-11, 90' PPO-v2. PPO-champ-11 keeps a goalkeeper back 81% of the time and runs flat out almost all the time. PPO-v2 keeps a goalkeeper back 83% of the time and runs flat out almost all the time. PPO-champ-11 kept the ball in the opposition half 57% of the time; shots on goal 73–39.

| Pos | Team | P | W | D | L | GF | GA | GD | Pts | Compute $ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | PPO-champ-11 | 4 | 3 | 1 | 0 | 60.5 | 16.8 | +43.7 | **10** | 11.43 |
| 2 | PPO-v2 | 4 | 3 | 1 | 0 | 60.7 | 20.5 | +40.2 | **10** | 18.22 |
| 3 | LLM-v11 | 4 | 2 | 0 | 2 | 94.2 | 36.2 | +58.0 | **6** | 22.30 |
| 4 | GP-v3 | 4 | 1 | 0 | 3 | 50.8 | 57.0 | -6.2 | **3** | 22.32 |
| 5 | PPO-v3 | 4 | 0 | 0 | 4 | 4.3 | 140.0 | -135.7 | **0** | 19.67 |

<!-- AUTO:ROUNDS END -->
