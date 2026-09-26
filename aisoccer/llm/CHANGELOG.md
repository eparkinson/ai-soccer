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
