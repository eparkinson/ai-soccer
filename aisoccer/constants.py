class Constants:
    # Constants related to the core physics of the game
    FIELD_LENGTH = 1800
    FIELD_HEIGHT = 800
    BALL_RADIUS = 8
    NUM_PLAYERS = 5
    PLAYER_RADIUS = 24
    GAME_LENGTH = 1800

    # Game speed. Speeds and accelerations are in px per tick; SPEED_SCALE multiplies all
    # of them (the same game, played SPEED_SCALE times faster per tick). Each tick's
    # motion is split into SUBSTEPS equal sub-steps, with collisions, walls and goals
    # checked at every one, so fast bodies cannot tunnel through posts, players or walls.
    # SPEED_SCALE = 1 with SUBSTEPS = 1 is exactly the original game. Change them with
    # set_speed (it re-derives the values below).
    SPEED_SCALE = 1.5
    SUBSTEPS = 2
    MAX_BALL_VELOCITY = 10 * SPEED_SCALE
    MAX_PLAYER_VELOCITY = 5 * SPEED_SCALE
    MAX_ACCELERATION = 1 * SPEED_SCALE  # per tick; each player's move is capped at this

    # Goals. Each end of the field has a goal line GOAL_DEPTH in from the edge. The goal
    # mouth is a GOAL_MOUTH_HEIGHT gap in that line, centred vertically, with a post at
    # each end. Outside the mouth the goal line acts as a wall. A goal is scored when the
    # whole ball crosses the line inside the mouth.
    GOAL_DEPTH = 20
    GOAL_MOUTH_HEIGHT = 400  # 300 until league-2 round 15: five players could wall off a 300 px goal
    GOAL_Y_MIN = (FIELD_HEIGHT - GOAL_MOUTH_HEIGHT) / 2
    GOAL_Y_MAX = (FIELD_HEIGHT + GOAL_MOUTH_HEIGHT) / 2
    POST_RADIUS = 4

    # Starting positions for [blue, red]. Blue defends the left goal, red the right.
    # Red's positions are blue's mirrored (x -> FIELD_LENGTH - 1 - x), so both teams see
    # the same kick-off from their own point of view.
    STARTING_POSITIONS = [
        [[200, 200], [200, 400], [500, 200], [500, 400], [800, 300]],
        [[1599, 200], [1599, 400], [1299, 200], [1299, 400], [999, 300]],
    ]

    @classmethod
    def set_speed(cls, speed_scale, substeps):
        """
        Set the game speed for this process: SPEED_SCALE, SUBSTEPS and the speed and
        acceleration caps derived from them. Returns the previous (speed_scale, substeps).
        Modules that copied a cap at import time (e.g. aisoccer/tactics/skills_v1.py)
        keep the old value.
        """
        previous = (cls.SPEED_SCALE, cls.SUBSTEPS)
        if int(substeps) != substeps or substeps < 1:
            raise ValueError("SUBSTEPS must be a positive integer")
        cls.SPEED_SCALE = speed_scale
        cls.SUBSTEPS = int(substeps)
        cls.MAX_BALL_VELOCITY = 10 * speed_scale
        cls.MAX_PLAYER_VELOCITY = 5 * speed_scale
        cls.MAX_ACCELERATION = 1 * speed_scale
        return previous
