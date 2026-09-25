class Constants:
    # Constants related to the core physics of the game
    FIELD_LENGTH = 1800
    FIELD_HEIGHT = 800
    BALL_RADIUS = 8
    NUM_PLAYERS = 5
    PLAYER_RADIUS = 24
    GAME_LENGTH = 1800
    MAX_BALL_VELOCITY = 10
    MAX_PLAYER_VELOCITY = 5

    # Goals. Each end of the field has a goal line GOAL_DEPTH in from the edge. The goal
    # mouth is a GOAL_MOUTH_HEIGHT gap in that line, centred vertically, with a post at
    # each end. Outside the mouth the goal line acts as a wall. A goal is scored when the
    # whole ball crosses the line inside the mouth.
    GOAL_DEPTH = 20
    GOAL_MOUTH_HEIGHT = 300
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
