"""
How fast games are shown on screen and in videos.

The original game (SPEED_SCALE 1) is shown at 2 ticks per 0.02 s frame: 100 ticks a
second. A game played SPEED_SCALE times faster per tick is shown at 100 / SPEED_SCALE
ticks a second, so players and ball cross the screen at the same pace as before.
"""

from fractions import Fraction

from aisoccer.constants import Constants

BASE_TICKS_PER_SECOND = 100  # at SPEED_SCALE 1
BASE_TICKS_PER_FRAME = 2


def display_pace(speed_scale=None):
    """
    (ticks per frame, frames per second) for showing a game played at ``speed_scale``
    (default Constants.SPEED_SCALE). Frames per second is an exact Fraction, e.g.
    (2, 50) at speed 1 (as before) and (1, 200/3) at speed 1.5.
    """
    if speed_scale is None:
        speed_scale = Constants.SPEED_SCALE
    scale = Fraction(str(speed_scale)).limit_denominator(1000)
    ticks_per_second = BASE_TICKS_PER_SECOND / scale
    ticks_per_frame = max(1, round(BASE_TICKS_PER_FRAME / scale))
    return ticks_per_frame, ticks_per_second / ticks_per_frame
