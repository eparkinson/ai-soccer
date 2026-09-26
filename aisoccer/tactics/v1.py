import numpy as np

from aisoccer.abstractbrain import AbstractBrain
from aisoccer.constants import Constants
from aisoccer.tactics import skills_v1 as skills

NUM = Constants.NUM_PLAYERS
L, H = skills.L, skills.H

# The team strategy: every entry in [0, 1]. Layer 1 chooses these numbers; the rest of
# the brain plays according to them. A learner can set them per situation, so new
# strategies can emerge from combinations nobody named.
STRATEGY = [
    "line",  # how far up the field the formation sits
    "push",  # how far the formation moves up and down with the ball
    "width",  # how far apart the formation spreads across the field
    "press",  # how many players go for the ball (1 to 3)
    "shoot_range",  # how far out the ball winner shoots (300 to 1300 px from goal)
    "pass_bias",  # preference for passing over carrying the ball
    "power",  # kick strength for passes and carries
    "keeper_depth",  # how far off the goal line the keeper stands
    "mark",  # tendency of outfield players to mark opponents when defending
    # How the ball winner values kicks it can reach without repositioning:
    "opportunistic",  # above 0.5: take the best reachable kick; below: line up first
    "w_goal",  # value of a direct shot
    "w_bank",  # value of a shot off a side wall
    "w_pass",  # value of a pass
    "w_space",  # value of knocking the ball into space
    "max_angle",  # widest strike angle used (30 to 90 degrees off straight)
]
DEFAULT_STRATEGY = {
    "line": 0.35,
    "push": 0.5,
    "width": 0.6,
    "press": 0.3,
    "shoot_range": 0.5,
    "pass_bias": 0.5,
    "power": 0.7,
    "keeper_depth": 0.1,
    "mark": 0.5,
    "opportunistic": 1.0,
    "w_goal": 0.6,
    "w_bank": 0.4,
    "w_pass": 0.5,
    "w_space": 0.4,
    "max_angle": 0.5,
}
DECISION_TICKS = 10
PATH_TICKS = 90


class TacticsV1(AbstractBrain):
    """
    TacticsBrain version 1 (frozen: improvements go in a new version, never here).

    A hierarchical brain:

    1. Team strategy: a vector of numbers (STRATEGY) such as how high the line sits
       and how many players press. `choose_strategy` picks it from the situation;
       by default it is fixed.
    2. Player skills: every DECISION_TICKS ticks each player is given a skill with a
       target: keep goal, win the ball, support the press, mark an opponent, block
       the shooting lane, or hold a formation position. The ball winner decides
       whether to shoot, pass or carry.
    3. Skill execution: every tick, using aisoccer/tactics/skills_v1.py.
    """

    def __init__(self, name=None, strategy=None):
        super().__init__(name=name)
        self.strategy = {**DEFAULT_STRATEGY, **(strategy or {})}
        self.ticks = 0
        self.plan = [("hold", None)] * NUM

    # ------------------------------------------------------------ layer 1

    def choose_strategy(self):
        """The strategy for the current situation. Subclasses can learn this."""
        return self.strategy

    # ------------------------------------------------------------ layer 2

    def assign(self, s, ball_path):
        mine, opp = self.my_players_pos, self.opp_players_pos
        my_times = [
            skills.intercept(mine[i], self.my_players_vel[i], ball_path)[0]
            for i in range(NUM)
        ]
        opp_times = [
            skills.intercept(opp[i], self.opp_players_vel[i], ball_path)[0]
            for i in range(NUM)
        ]
        plan: list = [None] * NUM

        keeper = int(np.argmin(mine[:, 0]))  # deepest player keeps goal
        plan[keeper] = ("keep", None)
        outfield = [i for i in range(NUM) if i != keeper]

        by_time = sorted(outfield, key=lambda i: my_times[i])
        pressers = by_time[: 1 + int(round(s["press"] * 2))]
        winner = pressers[0]
        plan[winner] = ("ball", None)
        for i in pressers[1:]:
            plan[i] = ("support", None)

        # Everyone else: formation, or marking when the opponent will reach the ball first.
        rest = [i for i in outfield if plan[i] is None]
        we_win = my_times[winner] <= min(opp_times) + 3
        slots = self.formation(s, ball_path[0], len(rest))
        dangerous = sorted(
            range(NUM), key=lambda j: opp[j][0]
        )  # deepest into our half first
        marked = set()
        for slot, i in zip(slots, sorted(rest, key=lambda i: mine[i][1])):
            if not we_win and self.rng.random() < s["mark"]:
                target = next((j for j in dangerous if j not in marked), None)
                if target is not None:
                    marked.add(target)
                    plan[i] = ("mark", target)
                    continue
            plan[i] = ("hold", slot)
        return plan

    def formation(self, s, ball, n):
        """Home positions for n outfield players, shifted with the ball."""
        base_x = L * (0.15 + 0.5 * s["line"]) + (ball[0] - L / 2) * s["push"]
        spread = H * (0.3 + 0.6 * s["width"])
        ys = (
            np.linspace(H / 2 - spread / 2, H / 2 + spread / 2, max(n, 1))
            if n > 1
            else [H / 2]
        )
        # Stagger: alternate players a little further forward.
        return [
            np.array(
                [
                    np.clip(base_x + (80 if k % 2 else -80), 60, L - 60),
                    float(np.clip(y, 40, H - 40)),
                ]
            )
            for k, y in enumerate(ys)
        ][:n]

    # ------------------------------------------------------------ layer 3

    def ball_action(self, i, s, ball_path):
        """The ball winner: shoot, pass or carry."""
        pos, vel = self.my_players_pos[i], self.my_players_vel[i]
        ball, ball_vel = np.asarray(self.ball_pos, float), np.asarray(
            self.ball_vel, float
        )
        opp = self.opp_players_pos
        to_goal = skills.GOAL_LINE - ball[0]

        # Contested: an opponent gets there about as soon as we do. No time to line up.
        mine_t, _ = skills.intercept(pos, vel, ball_path)
        theirs_t = min(
            skills.intercept(opp[j], self.opp_players_vel[j], ball_path)[0]
            for j in range(NUM)
        )
        if theirs_t <= mine_t + 4:
            return skills.tackle(pos, vel, ball, ball_vel, ball_path)

        if s["opportunistic"] >= 0.5:
            _, ball_then = skills.intercept(pos, vel, ball_path)
            best = self.best_reachable_kick(i, s, np.asarray(ball_then))
            if best is not None:
                direction, power = best
                return skills.kick(
                    pos, vel, ball, ball_vel, direction, power, ball_path
                )
            # Nothing worth kicking at from here: reposition, heading for a shot.
            return skills.kick(
                pos,
                vel,
                ball,
                ball_vel,
                skills.shot_direction(ball, None, opp),
                1.0,
                ball_path,
            )

        if to_goal < 300 + 1000 * s["shoot_range"]:
            target_y = skills.open_goal_y(ball, opp)
            if (
                skills.lane_clearance(ball, np.array([skills.GOAL_LINE, target_y]), opp)
                > 30
            ):
                return skills.kick(
                    pos,
                    vel,
                    ball,
                    ball_vel,
                    skills.shot_direction(ball, target_y),
                    1.0,
                    ball_path,
                )

        if (
            self.rng.random() < s["pass_bias"] * 0.2
        ):  # reconsidered every tick, so keep it rare
            options = []
            for j in range(NUM):
                mate = self.my_players_pos[j]
                if j == i or mate[0] < ball[0] + 50:
                    continue  # only forward passes
                lead = mate + self.my_players_vel[j] * 20
                clear = skills.lane_clearance(ball, lead, opp)
                if clear > 70:
                    options.append((mate[0], lead))
            if options:
                lead = max(options, key=lambda o: o[0])[1]
                distance = np.linalg.norm(lead - ball)
                power = float(np.clip(distance / 900, 0.35, 1.0)) * (
                    0.6 + 0.4 * s["power"]
                )
                return skills.kick(
                    pos, vel, ball, ball_vel, lead - ball, power, ball_path
                )

        # Carry: towards goal, angled away from the nearest opponent.
        direction = skills.unit(skills.GOAL_CENTRE - ball)
        nearest = opp[np.argmin(np.linalg.norm(opp - ball, axis=1))]
        away = skills.unit(ball - nearest)
        direction = skills.unit(
            direction + 0.3 * away * (np.linalg.norm(nearest - ball) < 150)
        )
        return skills.kick(
            pos, vel, ball, ball_vel, direction, 0.3 + 0.4 * s["power"], ball_path
        )

    def best_reachable_kick(self, i, s, ball):
        """
        The most valuable kick the ball winner can make from where it already is.

        The ball leaves along the line from the player's centre to the ball's at
        contact, so from the current approach every direction within about 60 degrees
        is reachable by striking the ball off-centre, without repositioning. Wider
        angles lose power and make the direction too sensitive, so they are not used.
        Candidates: the goal (direct, or off a side wall), a teammate, or space.
        Returns (direction, power), or None if nothing reachable is worth it.
        """
        pos = self.my_players_pos[i]
        opp = self.opp_players_pos
        approach = skills.unit(ball - pos)
        candidates = []  # (value, target point, power)

        shoot_reach = 300 + 1000 * s["shoot_range"]
        margin = skills.BALL_R + Constants.POST_RADIUS + 8
        for y in np.linspace(skills.GOAL_Y_MIN + margin, skills.GOAL_Y_MAX - margin, 5):
            goal = np.array([skills.GOAL_LINE + 20, y])
            distance = np.linalg.norm(goal - ball)
            if distance < shoot_reach:
                clear = skills.lane_clearance(ball, goal, opp)
                value = (
                    5
                    * s["w_goal"]
                    * (1 - distance / shoot_reach)
                    * min(clear / 60, 1.0)
                )
                candidates.append((value, goal, 1.0))
            # Bank shot: aim at the goal's mirror image beyond a side wall.
            for wall in (skills.BALL_R, H - skills.BALL_R):
                mirror = np.array([goal[0], 2 * wall - y])
                path = np.linalg.norm(mirror - ball)
                if path < shoot_reach:
                    frac = (
                        (wall - ball[1]) / (mirror[1] - ball[1])
                        if mirror[1] != ball[1]
                        else 2.0
                    )
                    if 0 < frac < 1:
                        bounce = ball + (mirror - ball) * frac
                        clear = min(
                            skills.lane_clearance(ball, bounce, opp),
                            skills.lane_clearance(bounce, goal, opp),
                        )
                        value = (
                            5
                            * s["w_bank"]
                            * (1 - path / shoot_reach)
                            * min(clear / 60, 1.0)
                        )
                        candidates.append((value, mirror, 1.0))

        for j in range(NUM):
            if j == i:
                continue
            lead = self.my_players_pos[j] + self.my_players_vel[j] * 15
            gain = (lead[0] - ball[0]) / L
            if gain < -0.05:
                continue
            clear = skills.lane_clearance(ball, lead, opp)
            if clear > 50:
                value = (
                    2
                    * s["w_pass"]
                    * (0.2 + 2 * gain)
                    * (0.5 + s["pass_bias"])
                    * min(clear / 120, 1.0)
                )
                power = float(np.clip(np.linalg.norm(lead - ball) / 900, 0.3, 1.0)) * (
                    0.6 + 0.4 * s["power"]
                )
                candidates.append((value, lead, power))

        for angle in np.radians(np.arange(-60, 61, 20)):
            target = ball + np.array([np.cos(angle), np.sin(angle)]) * 300
            target = np.clip(target, [40, 40], [L - 40, H - 40])
            free = np.min(np.linalg.norm(opp - target, axis=1))
            value = s["w_space"] * ((target[0] - ball[0]) / 300) * min(free / 200, 1.0)
            candidates.append((value, target, 0.3 + 0.4 * s["power"]))

        best, best_score = None, 0.05
        widest = np.cos(np.radians(30 + 60 * s["max_angle"]))
        for value, target, power in candidates:
            direction = skills.unit(target - ball)
            straightness = float(np.dot(approach, direction))
            if straightness < widest:
                continue  # not reachable without repositioning
            score = value * straightness**2
            if score > best_score:
                best, best_score = (direction, power), score
        return best

    def do_move(self):
        s = self.choose_strategy()
        ball = np.asarray(self.ball_pos, float)
        ball_path = skills.predict_ball(ball, self.ball_vel, PATH_TICKS)
        if self.ticks % DECISION_TICKS == 0:
            self.plan = self.assign(s, ball_path)
        self.ticks += 1

        moves = np.zeros((NUM, 2))
        for i, (skill, target) in enumerate(self.plan):
            pos, vel = self.my_players_pos[i], self.my_players_vel[i]
            if skill == "keep":
                moves[i] = skills.keep_goal(
                    pos, vel, ball, self.ball_vel, depth=150 * s["keeper_depth"]
                )
            elif skill == "ball":
                moves[i] = self.ball_action(i, s, ball_path)
            elif skill == "support":
                # Cover the space behind the ball winner, between the ball and our goal.
                moves[i] = skills.block_lane(pos, vel, ball, fraction=0.25)
            elif skill == "mark":
                moves[i] = skills.mark(pos, vel, self.opp_players_pos[target])
            else:
                moves[i] = skills.move_to(pos, vel, target)
        return moves

    def on_goal_scored(self, team, game_state):
        self.ticks = 0  # re-plan at kick-off

    def on_goal_conceded(self, team, game_state):
        self.ticks = 0
