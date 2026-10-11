# LLMBrain changelog

Each version is written by Claude after reading the previous version's diagnostics
(`llm_report.py`). One small, explainable change at a time, and only after a round in which it didn't lead:
what I saw, what I changed, why. The aim each time is the smallest change likely to
get it back on top, or near it, not the biggest possible improvement. Each version records its compute: the diagnostic games'
CPU and the tokens spent writing it.

## v1: everyone runs at the ball

**What I saw:** nothing yet: this is the starting point.

**What I changed:** every player runs straight at the ball.

**Why:** it is the simplest possible idea, and the kind of first attempt most people write.
First diagnostics against the league (before admission): it swarms the ball and outshoots the
young learners 25 to 15 per match, but when it concedes, its goal is empty 95% of the time and
nearly all five players are caught upfield of the ball.

## v2: one player stays in goal

**What I saw:** v1 finished 10th of 12 in round 9. When it conceded, our goal was empty 98%
of the time, all five players were upfield of the ball, and the nearest was a median 475 px
away; the decisive touches were mid-range (median 264 px). Its attack was fine: it out-shot
its opponents 17.5 to 16.2 per match.

**What I changed:** player 0 stops chasing. It stands just in front of our goal line and
slides along it to stay level with the ball, never leaving the goal mouth. The other four
still run at the ball.

**Why:** an empty net was behind nearly every goal conceded. A keeper is the smallest change
that fixes that, and costs only one of five chasers in an attack that already out-shoots
everyone.

## v3: the keeper comes off the line

**What I saw:** v2 finished 3rd of 10 in its first round (league 3, round 12). The keeper did
its job of being there: our goal was never empty when we conceded. But GP and the Coach still
scored 8-9 a match against us (v2 lost 7 of 10 to GP and all 10 to the Coach), and the
decisive touches came from a median 301 px out. A keeper standing on the goal line can only
slide sideways, and in a 400 px goal it cannot reach a shot taken from closer than about 360 px.

**What I changed:** the keeper leaves the goal line. It stands on the straight line from the
middle of our goal to the ball, halfway to the ball, but never more than half a goal-width
out. The other four still run at the ball.

**Why:** stepping towards the shooter "narrows the angle": the closer the keeper is to the
ball, the less of the goal is left to aim at, so the same sideways speed covers more of it.
It keeps v2's one-keeper, four-chaser shape and changes only where the keeper stands.

**Sanity check (10 games of 9000 ticks, 62 CPU seconds):** vs GP W-D-L 1-0-5, goals 43-73
(7.2-12.2 per match, v2 had 6.3-9.1); vs v2 1-0-3, goals 23-30. A small sample, but it
points the wrong way: more goals conceded, not fewer.

## v4: one player stays between the ball and our goal

**What I saw:** v2 finished 3rd then 4th of 10 (league 3, rounds 12 and 13), behind GP and the
Coach (v2 lost 7 of 10 to GP, 91 goals conceded, and all 10 to the Coach). When we conceded,
our goal was never empty, but on average 4.1 of our 5 players were upfield of the ball: the
keeper faced every attack alone. v3 moved the keeper off its line and conceded more, so the
keeper is not the problem; nobody else being back is.

**What I changed:** built on v2, not v3. Player 1 stops chasing: it stands halfway between the
ball and the middle of our goal. The keeper and the other three chasers are exactly as in v2.

**Why:** a player on the straight line from the ball to our goal blocks the most direct shot
and dribble, and is always goal-side of the ball, so the keeper is no longer alone. It is the
smallest change that puts a second player behind the ball, and it leaves the keeper untouched.

**Sanity check (10 games of 9000 ticks, 66 CPU seconds):** vs GP W-D-L 4-0-2, goals 34-30
(5.7-5.0 per match; v2 had 6.3-9.1); vs v2 4-0-0, goals 18-9. A small sample, but both point
the right way: far fewer goals conceded, and scoring barely dropped.

## v5: the sweeper closes down the ball

**What I saw:** v4 won league-3 rounds 14 and 15, then finished 3rd in round 16 behind GP and
its champion copy (v4 vs GP 6-0-4, vs the Coach 2-4-4, conceding 6.5 a match to the Coach).
When we conceded, the decisive touch came from a median 348 px out (301 px for v2), and the
nearest of our players was a median 190 px from the ball: the sweeper, standing halfway back.
The shooter had the ball to himself.

**What I changed:** built on v4. The sweeper no longer stands halfway between the ball and our
goal: it stands right in front of the ball, touching it, on the ball's side facing our goal.
The keeper and the three chasers are exactly as in v4.

**Why:** the sweeper already stood on the right line, from the ball to our goal, just too far
down it. Against the ball it blocks every direct shot and dribble at the source, instead of a
third of the goal from halfway. It is the smallest change that puts someone on the shooter.

**Sanity check (12 games of 9000 ticks, 74.6 CPU seconds):** vs v4 W-D-L 0-0-4, goals 14-38;
vs champ-3 (GP) 0-1-3, goals 22-31 (v4 had 4.8-4.7 per match); vs the Coach 1-0-3, goals
19-29. It points clearly the wrong way: far more goals conceded. A sweeper glued to the ball
is beaten by one touch and then far from goal, so the keeper is alone again; depth mattered
more than pressure. Like v3, v5 should not replace v4.

## v6: the chasers run at the back of the ball

**What I saw:** v4 won league-3 rounds 14 and 15, then finished 3rd and 2nd (rounds 16 and
17) behind GP and the Coach (round 16: v4 vs GP 6-0-4, vs the Coach 2-4-4, vs champ-3 6-0-4).
We had the ball 66% of the time but took fewer shots than our opponents (33.9 against 45.0 a
match), and ran nearly twice as far (305503 px against 168702). The three chasers all run at
the same point, the ball itself, from whichever side they are on, so a chaser coming from
upfield knocks the ball back towards our own goal. v3 and v5 pulled defenders forward and
conceded more, so this time the defence is left alone.

**What I changed:** built on v4. The three chasers no longer run at the middle of the ball but
at its back: the point just behind the ball, touching it, on the line from the middle of the
opponents' goal through the ball. The keeper and the sweeper are exactly as in v4.

**Why:** a player that reaches the ball from behind pushes it towards the opponents' goal, so
our control turns into shots instead of pushes in random directions, some towards our own
goal. It changes only where the chasers aim, not who defends, which v3 and v5 showed matters.

**Sanity check (12 games of 9000 ticks, 82.2 CPU seconds):** vs v4 W-D-L 2-1-1, goals 20-15;
vs champ-3 (GP) 4-0-0, goals 27-13 (v4 had 4.8-4.7 per match in round 16); vs the Coach
3-1-0, goals 24-11 (v4 had 6.2-6.5). A small sample, but all three point the right way: more
goals scored and fewer conceded. v6 should replace v4.

## v7: players stop at their spot instead of running past it

**What I saw:** v6 finished 2nd in league-3 round 18 (35-0-7), then 4th in round 19 (24-2-16),
behind GP, its champion copy champ-4 and the Coach (diagnostics: v6 vs GP 1-1-8, goals 16-37;
vs champ-4 1-2-7, goals 21-49). Our goal was never empty when we conceded, yet the shots went
in from a median 455 px out. Our players ran 66% further than theirs (303969 px against 183357
a match): 6.8 px a tick on average against a top speed of 7.5, so even the keeper and the
sweeper, whose spots move slowly, are nearly always at full speed. Each player accelerates
towards its spot and nothing slows it down, so it runs past, turns and runs past again: the
keeper swings to and fro across its line. GP's players steer differently: each picks the
velocity that takes it to its spot, slowing as it arrives. (Reading GP's generation-52 program:
the player nearest the ball runs at it, the other four stand together about 160 px out from
the middle of their goal, towards the ball.)

**What I changed:** built on v6. Every player now steers like GP's: it wants to move towards
its spot at top speed, or just fast enough to get there this tick when it is close, and
accelerates by the difference between that and how it is moving now. Every spot is exactly as
in v6.

**Why:** a player that stops on its spot is on its spot: the keeper stands still in front of
the shot and the sweeper stays on the line to our goal. It changes how players move, not where
they go, so the shape that v3 and v5 showed matters is kept.

**Sanity check (12 games of 9000 ticks, 70.1 CPU seconds):** vs v6 W-D-L 3-1-0, goals 14-4;
vs champ-4 (GP) 1-0-3, goals 9-14 (v6 had 2.1-4.9 per match in its diagnostics); vs GP
(snapshot r019) 0-3-1, goals 7-10 (v6 had 1.6-3.7). A small sample: it beats v6 and concedes
far fewer against GP, but still does not beat GP. v7 should replace v6.

## v8: a second player joins the sweeper

**What I saw:** v7 finished 3rd in league-3 round 20 (29-1-12), behind GP and the Coach
(diagnostics: v7 vs GP 1-2-7, goals 13-33; vs the Coach 2-2-6, goals 10-23). When we conceded,
our goal was never empty, the shot came from a median 399 px out, the nearest of our players
was 233 px from the ball, and 3.1 of our 5 players were upfield of it: the keeper and the
sweeper faced the attack alone while the three chasers were still behind the play. We had the
ball more (58%) and shot more (51 against 36 a match), so the attack is not short of players.
GP chases with one player and keeps the other four together in front of its goal.

**What I changed:** built on v7. Player 2 stops chasing and stands with the sweeper, halfway
between the ball and the middle of our goal. The keeper, the two remaining chasers and the way
every player steers are exactly as in v7.

**Why:** it is v4's idea again, one more player goal-side of the ball, the change that helped
most so far. Two players on the line from the ball to our goal block twice the width of one,
and two chasers still outnumber GP's single chaser. It keeps defenders back, which v3 and v5
showed matters, and moves nobody forward.

**Sanity check (12 games of 9000 ticks, 59.4 CPU seconds):** vs v7 W-D-L 3-0-1, goals 7-3;
vs the Coach 1-1-2, goals 4-7 (v7 had 1.0-2.3 per match in its diagnostics); vs GP (snapshot
r020) 1-1-2, goals 4-7 (v7 had 1.3-3.3). A small sample: it beats v7 and concedes fewer
against GP and the Coach, scoring about the same, but still does not beat either. v8 should
replace v7.

## v9: the two sweepers stand shoulder to shoulder

**What I saw:** v8 finished 4th in league-3 round 21 (25-1-16, +12.9 a game), behind GP (the
champion) and the Coach (diagnostics: v8 vs GP 0-2-8, goals 9-27; vs the Coach 0-3-7, goals
7-24). The second sweeper got players back: when we conceded, 2.1 of our 5 players were
upfield of the ball (v7: 3.1). But the goals still came from long shots (median 459 px out)
with our goal never empty, so our players were back yet not in the way of the shot. Both
sweepers aim at the very same spot, halfway between the ball and the middle of our goal, and
players bounce off each other, so they knock each other aside: two players there cover about
the width of one.

**What I changed:** built on v8. The two sweepers stand side by side, touching, across the
line from the ball to the middle of our goal, one on each side of it, instead of both on it.
Their midpoint is v8's spot. The keeper, the chasers and the way every player steers are
exactly as in v8.

**Why:** a shot has to pass the sweepers to reach the goal. Shoulder to shoulder they make a
wall two players wide in front of the shooter and stop pushing each other off the line. It
moves nobody forward (v3, v5) and keeps v8's shape.

**Sanity check (12 games of 9000 ticks, 63.5 CPU seconds):** vs v8 W-D-L 1-1-2, goals 4-6;
vs GP (snapshot r021) 1-1-2, goals 3-8 (v8 had 0.9-2.7 per match in its diagnostics); vs
champ-4 2-1-1, goals 8-5 (v8 had 2.6-3.0). A small sample and a mixed one: better than v8's
diagnostics against both GPs, but it lost to v8 head to head. Not clearly at least as good as
v8, so v9 should not replace v8.

## v10: the chasers strike the ball instead of stopping against it

**What I saw:** v8 finished 4th in league-3 rounds 21 and 22 (+12.9 and +6.8 goals a game),
behind GP (the champion since round 18) and the Coach (diagnostics after round 22: v8 vs GP
0-3-7, goals 8-19; vs the Coach 0-1-9, goals 2-29). We have the ball more (54%) and pass more,
yet they shoot more (45.6 against 38.9 a match), and against GP we score under one goal a match,
almost only from scrambles right at their goal (median 35 px out). Since v7 every player slows
as it reaches its spot, and a chaser's spot is the back of the ball, touching it: so a chaser
arrives there at almost no speed and nudges the ball along instead of hitting it. A player
weighs 9 times as much as the ball, so one running into it at top speed (7.5 px a tick) sends
it off at about 13.5, close to the ball's top speed of 15: a shot. GP's chaser runs at the ball
without slowing down.

**What I changed:** built on v8. The two chasers no longer slow down as they reach the back of
the ball; they run at it at top speed. Where every player goes, and how the keeper and the two
sweepers move, are exactly as in v8.

**Why:** running through the back of the ball at full speed turns our possession into shots
towards their goal from further out, instead of slow dribbles that end in scrambles. It changes
only how the chasers arrive, so the defence that v3, v5 and v9 showed matters is left alone.

**Sanity check (24 games of 9000 ticks for v10 and 16 for the v8 baseline on the same seeds,
244.0 CPU seconds):** vs v8 W-D-L 2-1-5, goals 14-25; vs GP (snapshot r022) 1-2-9, goals 11-26
(v8 on the same 12 seeds: 2-4-6, goals 12-25); vs the Coach 0-0-4, goals 0-14 (v8: 0-0-4,
goals 0-9). Striking the ball did not bring more goals: v10 scored no more than v8 against GP
and the Coach, conceded more, and lost clearly to v8 head to head. v10 should not replace v8.

## Line frozen after round 29 (league 3)

After v9 and v10 both failed their checks, the pure line was judged stalled: since round 19 it
finished 3rd to 6th at +1 to +13 goals per game, while GP and its snapshots held the top. The user
chose to freeze it: v8 stays in the league as the line's final entry and no further versions are
written. Total cost of the line in league 3: $4.42, against $1.0-1.3 for GP.

## v11: the kick planner (new rules: improvement first; large changes allowed)

**The rules changed.** After round 30 the user lifted the line's rules: improvement comes first,
and large changes are allowed (several at once, new capabilities, a rewrite), found from my own
evidence. The code must still be mine, readable and explainable: no networks, no evolved
programs, and no reading of the Tactics or GP code (studying them through games is fine).
v11 is a rewrite, not a step from v8. Code: `aisoccer/llm/v11.py` (333 lines).

**What I saw:** in round 30 Tactics won 39-0-3 and beat v8 6-0-0, 149-33; v8 was 6th of 8.
I recorded games and measured them myself (scripts in the session scratchpad, not in the repo):
- The ball is at or near its top speed (15 px a tick, twice a player's) about 70% of the time,
  and nothing slows it down; it is touched only about once every 15-30 ticks. The game is
  closer to air hockey than football: what matters is where each touch sends the ball.
- Tactics vs v8: the ball spent 80% of the time in v8's half; Tactics' outfield players stood
  goal-side of the ball 4.6 of 5 on average, and 60% of its goals were last touched by v8's own
  crowded defenders.
- A first-touch test (random ball and players, idle opponents parked at the edges): only 46% of
  v8's first touches went forwards and 23% towards the goal.

**What I changed:** everything. The ball's path is predicted with its wall bounces. A kick
planner tries 49 moves for each player (24 directions at full or half speed, or stop), follows
each for 40 ticks under the game's acceleration and speed rules, finds the first contact with
the ball and, from the collision rule, where the ball then goes; it scores that (forwards good;
on target, straight or off a side wall, better; better still if no opponent can get into its
path in time; at our goal very bad; sooner better) and the best-placed player takes the best
kick if it is good enough. Otherwise the first player to the ball's path runs for the spot
behind the ball, going round it. The keeper moves to where the ball's path will cross its line
and clears close balls with the planner. One player waits by their far post while the ball is
in their half; the other two hold a line 220 px behind the ball.

**Why:** the planner turns where each touch sends the ball from luck into a choice. In the
first-touch test one planning player sent 97% of its first touches forwards and 94% towards
the goal (v8: 46% and 23%).

**Experiments** (4500-tick games unless stated; W-D-L and goals per game vs Tactics, GP r030,
champ-6; differences under about 1 goal per game were noise at these sample sizes):
- Draft 1, geometry only (intercept the ball's path, line up behind it, run through it): lost
  0-8 to Tactics and 3-8 to GP over 4 games each; in the test it managed 78% forwards and 46% on
  target. Discarded: the ball moves while the player lines up, and contacts go wrong.
- Draft 2, the rollout planner: 11-1-0, 38-0 against v8, Tactics and GP; then 24-0-0, 200-5 in
  9000-tick games against v8, champ-6, GP and Tactics. Kept: this is the core of v11.
- One striker instead of two: better in both samples (+1.9 and +1.7 goals a game). Kept.
- Lane check (on-target kicks worth less when an opponent can cut them out): +1.1, -0.6, +0.7,
  +0.25 over four samples. Small but kept, since 58% of our on-target kicks against Tactics were blocked by outfield
  players (208 of 358 in two games) and only 3 went straight in.
- A poacher at the far post: over 12 games against each, +1.3 goals a game on average, better
  against all three (Tactics 3.5 vs 2.8, GP 7.8 vs 6.5, champ-6 8.5 vs 6.6). Kept.
- Two poachers: -2.5 a game (much worse). Discarded. Poacher deeper or wider: noise.
- Line 220 px behind the ball instead of 320: +0.8, +0.2 and +0.75 in three samples. Kept.
  150 px, or letting the line go higher: noise or worse.
- Planning 25 or 60 ticks ahead instead of 40, and kick thresholds 0.0 or 0.6 instead of 0.3:
  noise. Unchanged.

**Check (9000-tick games, colours alternated, seeds 20000+, CPU_SECONDS 710.6):**

| opponent | games | W-D-L | goals | per game |
|---|---|---|---|---|
| LLM-v8 | 12 | 12-0-0 | 162-1 | 13.5-0.1 |
| champ-6 (GP) | 12 | 12-0-0 | 190-2 | 15.8-0.2 |
| Tactics (now champ-7) | 12 | 12-0-0 | 83-3 | 6.9-0.3 |
| GP r032 | 4 | 4-0-0 | 57-0 | 14.3-0.0 |
| GP-v2 r032 | 2 | 2-0-0 | 33-1 | 16.5-0.5 |
| Coach r032 | 2 | 2-0-0 | 29-0 | 14.5-0.0 |
| PPO-goals r032 | 2 | 2-0-0 | 68-0 | 34.0-0.0 |
| PPO-shaped r032 | 2 | 2-0-0 | 68-0 | 34.0-0.0 |

v11 clearly beats v8, and every other opponent tried, and should enter the league.

**Compute:** about 5700 metered CPU seconds of games for development and the check (plus a few
hundred unmetered for recordings and the first-touch test); v11 itself costs
about 0.9 ms a tick (v8: 0.09), about 32 CPU seconds per 36000-tick league match.

**What I would try next** (from what the league shows after its first rounds):
- Tactics is the only opponent holding us under 10 a game, by crowding its goal; most of our
  shots at it are blocked. Try planning a kick that aims at a rebound spot (the planner already
  knows where the ball goes; it could follow it one bounce further, off a player).
- The planner ignores all other players between our player and the ball; add opponents (and
  team-mates) as obstacles in the rollout.
- Let a second player take a planned kick only when it is much sooner than the striker's.
- If opponents start to copy the planner, defence will matter again: study goals conceded then.
- Cost: plan only every second tick, or fewer directions, if the league needs it cheaper.
