"""
Many games at once: the rules and physics of ``aisoccer.game.Game`` stepped for N
independent games with (N, bodies, 2) arrays.

Game k of a VecGame plays out like ``Game(blue_brains[k], red_brains[k], seed=seeds[k])``:
the same kick-off (and random ball nudge from the same seeded generator), the same
brain generators, collisions, walls, goal mouths, posts, speed and acceleration caps,
goal detection and kick-off resets, and the same game length. Every operation is the
one Game does, applied elementwise over the games, so positions match Game's bit for
bit (tests/test_vecgame.py checks this).

The only structural difference: Game spends one ``tick()`` call on a goal (reset, no
physics), while a VecGame resets a game that scored and goes straight on to its next
physics step in the same call. Physics ticks, and so the game clock, stay in step
across all games, and each game sees exactly the sequence of states Game produces.

PPOBrain seats are played together by ``aisoccer.vecppo.BatchedPPO`` (one feature
computation and one network pass per decision for all games); any other brain is
called one game at a time with the same view Game gives it.
"""

import numpy as np

from aisoccer.brains.PPOBrain import PPOBrain
from aisoccer.constants import Constants
from aisoccer.vecppo import BatchedPPO

NUM = Constants.NUM_PLAYERS
# Body order in every game's arrays, the same as in Game's physics state: blue players,
# red players, the four posts, the ball.
PLAYERS = slice(0, 2 * NUM)
POSTS = slice(2 * NUM, 2 * NUM + 4)
BALL = 2 * NUM + 4
N_BODIES = BALL + 1

POST_POSITIONS = [
    [x, y]
    for x in (Constants.GOAL_DEPTH, Constants.FIELD_LENGTH - 1 - Constants.GOAL_DEPTH)
    for y in (Constants.GOAL_Y_MIN, Constants.GOAL_Y_MAX)
]
KICK_OFF = np.array(
    Constants.STARTING_POSITIONS[0]
    + Constants.STARTING_POSITIONS[1]
    + POST_POSITIONS
    + [[(Constants.FIELD_LENGTH - 1) / 2, Constants.FIELD_HEIGHT / 2]],
    dtype=float,
)
RADIUS = np.array(
    [float(Constants.PLAYER_RADIUS)] * 2 * NUM
    + [float(Constants.POST_RADIUS)] * 4
    + [float(Constants.BALL_RADIUS)]
)
FIXED = np.zeros(N_BODIES, dtype=bool)
FIXED[POSTS] = True
SPEED_CAPS = np.full(N_BODIES, np.inf)
SPEED_CAPS[PLAYERS] = Constants.MAX_PLAYER_VELOCITY
SPEED_CAPS[BALL] = Constants.MAX_BALL_VELOCITY

# Collision constants (see PhyState._collide).
_MASS = RADIUS**2
REACH_SQ = (RADIUS[:, None] + RADIUS[None, :]) ** 2
IMPULSE = np.where(
    FIXED[None, :], 2.0, 2.0 * _MASS[None, :] / (_MASS[:, None] + _MASS[None, :])
)
CAN_HIT = ~np.eye(N_BODIES, dtype=bool)
CAN_HIT[FIXED, :] = False  # fixed bodies are never pushed

MAX_X = Constants.FIELD_LENGTH - 1  # PhyState's maxX
MAX_Y = Constants.FIELD_HEIGHT


class VecGame:
    """
    N independent games stepped together. ``blue_brains[k]`` and ``red_brains[k]`` play
    game k, seeded with ``seeds[k]``; every brain must be its own instance.
    """

    def __init__(
        self,
        blue_brains,
        red_brains,
        seeds,
        game_length=Constants.GAME_LENGTH,
        precision="exact",
    ):
        """
        :param precision: how PPOBrain networks run (see ``aisoccer.vecppo.forward``):
            "exact" gives every action bit-identical to Game's; "float32" is several
            times faster, its network outputs differing from float64 by about 1e-7.
        """
        n = len(seeds)
        if not len(blue_brains) == len(red_brains) == n:
            raise ValueError("need one blue brain, one red brain and one seed per game")
        if game_length <= 0:
            raise ValueError("a VecGame needs a finite game length")
        brains = [b for pair in zip(blue_brains, red_brains) for b in pair]
        if len({id(b) for b in brains}) != len(brains):
            raise ValueError("every game needs its own brain instances")
        self.n = n
        self.game_length = game_length
        self.brains = [list(blue_brains), list(red_brains)]  # [side][game]
        self.rngs = [np.random.default_rng(s) for s in seeds]
        for rng, blue, red in zip(self.rngs, blue_brains, red_brains):
            blue.rng, red.rng = rng.spawn(2)  # as Game.seed_brains

        self.pos = np.repeat(KICK_OFF[None], n, axis=0)
        self.vel = np.zeros((n, N_BODIES, 2))
        self.score = np.zeros((n, 2), dtype=int)  # blue, red
        self.last_goal_tick = np.zeros(n, dtype=int)
        self.ticks = 0
        for g in range(n):
            self.kick_off(g)

        # PPOBrains (exactly that class: subclasses may change how they decide) play
        # together; any other brain is asked one game at a time.
        ppo_seats = [
            (g, s)
            for s in (0, 1)
            for g in range(n)
            if type(self.brains[s][g]) is PPOBrain
        ]
        self.ppo = (
            BatchedPPO([self.brains[s][g] for g, s in ppo_seats], precision=precision)
            if ppo_seats
            else None
        )
        self.ppo_games = np.array([g for g, _ in ppo_seats], dtype=int)
        self.ppo_sides = np.array([s for _, s in ppo_seats], dtype=int)
        self.batched = set(ppo_seats)
        self.other_seats = [
            (g, s) for g in range(n) for s in (0, 1) if (g, s) not in self.batched
        ]

    def kick_off(self, g):
        """Everyone back to the kick-off positions and a small random nudge to the ball (Game.start)."""
        self.pos[g] = KICK_OFF
        self.vel[g] = 0.0
        self.vel[g, BALL] = self.rngs[g].random(2) - 0.5

    def play(self):
        """Play every game to the end and return their scores as Game.play does."""
        while self.ticks < self.game_length:
            self.tick()
        self.finish()
        return self.scores()

    def finish(self):
        """Hand the training PPOBrains their trajectories (``play`` does this at the end)."""
        if self.ppo is not None:
            self.ppo.finish()

    def scores(self):
        return [{"blue": int(b), "red": int(r)} for b, r in self.score]

    def tick(self):
        """
        One physics step of every game. A game whose ball went in on the previous step
        first scores and restarts (what Game does in a tick of its own), then plays on.
        """
        self.check_goals()
        self.apply_moves(self.run_brains())
        self.limit_velocities()
        self.step_physics()
        self.ticks += 1

    def check_goals(self):
        x, y = self.pos[:, BALL, 0], self.pos[:, BALL, 1]
        mouth = (Constants.GOAL_Y_MIN < y) & (y < Constants.GOAL_Y_MAX)
        red = (x < Constants.GOAL_DEPTH - Constants.BALL_RADIUS) & mouth
        goal_line = Constants.FIELD_LENGTH - 1 - Constants.GOAL_DEPTH
        blue = ~red & (x > goal_line + Constants.BALL_RADIUS) & mouth
        if not (red.any() or blue.any()):
            return
        for g in np.flatnonzero(red | blue):
            scorer = 0 if blue[g] else 1
            ticks_elapsed = self.ticks - int(self.last_goal_tick[g])
            self.last_goal_tick[g] = self.ticks
            self.score[g, scorer] += 1
            side_name = ("blue", "red")[scorer]
            game_state = {"ticks_elapsed": ticks_elapsed}
            for s in (0, 1):  # blue's brain first, as in Game.notify_brains_goal
                if (g, s) in self.batched:
                    continue
                brain = self.brains[s][g]
                if s == scorer:
                    brain.on_goal_scored(side_name, game_state)
                else:
                    brain.on_goal_conceded(side_name, game_state)
            if self.ppo is not None:  # PPOBrain.on_goal_scored / on_goal_conceded
                sign = np.where(self.ppo_sides == scorer, 1.0, -1.0)
                self.ppo.goal(self.ppo_games == g, sign * self.ppo.goal_reward)
            self.kick_off(g)

    def views(self, games, sides):
        """
        The views of the given seats (arrays of game and side): positions and velocities
        of the seat's own players, the opponents and the ball, mirrored for red as in
        Game, and the two scores.
        """
        k = len(games)
        rows = np.arange(k)
        players = self.pos[games, PLAYERS].reshape(k, 2, NUM, 2)
        players_vel = self.vel[games, PLAYERS].reshape(k, 2, NUM, 2)
        my_pos, opp_pos = players[rows, sides], players[rows, 1 - sides]
        my_vel, opp_vel = players_vel[rows, sides], players_vel[rows, 1 - sides]
        ball_pos, ball_vel = self.pos[games, BALL], self.vel[games, BALL]
        red = sides == 1
        if red.any():  # flip_pos / flip_vel
            for p in (my_pos, opp_pos):
                p[red, :, 0] = Constants.FIELD_LENGTH - 1 - p[red, :, 0]
            for v in (my_vel, opp_vel):
                v[red, :, 0] = -v[red, :, 0]
            ball_pos[red, 0] = Constants.FIELD_LENGTH - 1 - ball_pos[red, 0]
            ball_vel[red, 0] = -ball_vel[red, 0]
        my_score, opp_score = self.score[games, sides], self.score[games, 1 - sides]
        return my_pos, my_vel, opp_pos, opp_vel, ball_pos, ball_vel, my_score, opp_score

    def run_brains(self):
        """Every seat's acceleration matrix, [game, side, player, xy], in its own view."""
        game_time = float(self.ticks) / float(self.game_length)
        moves = np.zeros((self.n, 2, NUM, 2))
        if self.ppo is not None:
            g, s = self.ppo_games, self.ppo_sides
            moves[g, s] = self.ppo.act(
                lambda seats: self.views(g[seats], s[seats]), game_time
            )
        if self.other_seats:
            games = np.array([g for g, _ in self.other_seats])
            sides = np.array([s for _, s in self.other_seats])
            view = self.views(games, sides)
            for i, (g, s) in enumerate(self.other_seats):
                (
                    my_pos,
                    my_vel,
                    opp_pos,
                    opp_vel,
                    ball_pos,
                    ball_vel,
                    my_score,
                    opp_score,
                ) = (v[i] for v in view)
                moves[g, s] = self.brains[s][g].move(
                    my_pos,
                    my_vel,
                    opp_pos,
                    opp_vel,
                    ball_pos,
                    ball_vel,
                    int(my_score),
                    int(opp_score),
                    game_time,
                )
        return moves

    def apply_moves(self, moves):
        """Team.apply_move for every seat: scrub NaN/inf, cap each acceleration at 1, add to velocity."""
        moves[:, 1, :, 0] = -moves[
            :, 1, :, 0
        ]  # red's moves back into the real field (flip_acc)
        if not np.isfinite(moves).all():
            moves = np.nan_to_num(moves, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(moves, axis=-1, keepdims=True)
        capped = np.where(norms > 1, moves / np.maximum(norms, 1), moves)
        self.vel[:, PLAYERS] = self.vel[:, PLAYERS] + capped.reshape(self.n, 2 * NUM, 2)

    def limit_velocities(self):
        vel = self.vel
        speed = np.hypot(vel[..., 0], vel[..., 1])
        too_fast = speed > SPEED_CAPS
        if too_fast.any():
            caps = np.broadcast_to(SPEED_CAPS, speed.shape)
            vel[too_fast] *= (caps[too_fast] / speed[too_fast])[:, None]

    def step_physics(self):
        """PhyState.tick for every game: collisions, then walls, then move."""
        pos = self.pos
        vel = self.collide(pos, self.vel)
        vel = self.bounce_walls(pos, vel)
        vel[:, FIXED] = 0.0
        self.pos = pos + vel
        self.vel = vel

    @staticmethod
    def collide(pos, vel):
        """
        PhyState._collide over a leading game axis: elastic collisions resolved
        simultaneously. The same arithmetic, done only for the pairs that touch.
        """
        pdiff = pos[:, :, None, :] - pos[:, None, :, :]
        dist_sq = np.einsum("nijk,nijk->nij", pdiff, pdiff)
        near = (dist_sq <= REACH_SQ) & CAN_HIT
        if not near.any():
            return vel
        g, i, j = np.nonzero(near)
        pdiff, dist_sq = pdiff[g, i, j], dist_sq[g, i, j]
        vdiff = vel[g, i] - vel[g, j]
        next_pdiff = pdiff + vdiff
        hit = (
            np.einsum("pk,pk->p", next_pdiff, next_pdiff) < dist_sq
        )  # moving towards each other
        if not hit.any():
            return vel
        g, i, j, pdiff, vdiff, dist_sq = (
            g[hit],
            i[hit],
            j[hit],
            pdiff[hit],
            vdiff[hit],
            dist_sq[hit],
        )
        safe_dist_sq = np.where(dist_sq == 0, 1.0, dist_sq)
        scale = IMPULSE[i, j] * np.einsum("pk,pk->p", vdiff, pdiff) / safe_dist_sq
        impulse = np.zeros_like(vel)
        np.add.at(impulse, (g, i), scale[:, None] * pdiff)
        return vel - impulse

    @staticmethod
    def bounce_walls(pos, vel):
        """PhyState._bounce_walls: walls, with the gap of the goal mouths at each end."""
        vel = vel.copy()
        x, y = pos[..., 0], pos[..., 1]
        in_mouth = (y > Constants.GOAL_Y_MIN) & (y < Constants.GOAL_Y_MAX)
        min_x = np.where(in_mouth, 0.0, Constants.GOAL_DEPTH) + RADIUS
        max_x = np.where(in_mouth, MAX_X, MAX_X - Constants.GOAL_DEPTH) - RADIUS
        vx, vy = vel[..., 0], vel[..., 1]
        flip_x_ = ((x < min_x) & (vx < 0)) | ((x > max_x) & (vx > 0))
        flip_y = ((y < RADIUS) & (vy < 0)) | ((y > MAX_Y - RADIUS) & (vy > 0))
        vel[..., 0] = np.where(flip_x_, -vx, vx)
        vel[..., 1] = np.where(flip_y, -vy, vy)
        return vel
