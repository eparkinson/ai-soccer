"""
Render recorded league games (runs/<league>/replays/round-NNN.npz) to MP4 videos.

    poetry run python render_replay.py runs/league2/replays/round-001.npz
    poetry run python render_replay.py runs/league2/replays/*.npz --out videos/

Frames are drawn offscreen with pyglet (the same look as the live match window) and
piped to ffmpeg (from the imageio-ffmpeg package: poetry install --with video).
"""

import argparse
import subprocess
from pathlib import Path

import ctypes

import imageio_ffmpeg
import numpy as np
import pyglet
from pyglet import gl
from pyglet.math import Mat4, Vec3

from aisoccer.constants import Constants
from aisoccer.graphics.field import Field

L, H = Constants.FIELD_LENGTH, Constants.FIELD_HEIGHT
HEADER = 110
BLUE, RED = (110, 140, 255, 255), (255, 110, 110, 255)
WHITE, GREY, GOLD = (240, 240, 240, 255), (170, 180, 175, 255), (242, 193, 78, 255)


class Renderer:
    def __init__(self, scale=2 / 3):
        # Draw in field coordinates, scaled down; frames are read straight from OpenGL.
        self.width, self.height = int(L * scale) // 2 * 2, int((H + HEADER) * scale) // 2 * 2
        self.window = pyglet.window.Window(self.width, self.height, visible=False)
        self.window.view = Mat4.from_scale(Vec3(scale, scale, 1))
        self.pixels = (ctypes.c_ubyte * (self.width * self.height * 3))()
        self.labels = {}

    def label(self, text, size, colour, x, y, anchor, anchor_y="baseline", bold=False):
        """Text labels are slow to lay out, so reuse them."""
        key = (text, size, colour, x, y, anchor, anchor_y, bold)
        if key not in self.labels:
            if len(self.labels) > 400:
                self.labels.clear()
            self.labels[key] = pyglet.text.Label(
                text, font_size=size, color=colour, x=x, y=y, anchor_x=anchor, anchor_y=anchor_y,
                weight="bold" if bold else "normal",
            )
        return self.labels[key]

    def frame(self, positions, score, names, title, progress, flash=None, card=None):
        w = self.window
        w.switch_to()
        w.clear()
        pyglet.shapes.Rectangle(0, 0, L, H, Field.BACKGROUND_COLOUR).draw()
        mouth = Constants.GOAL_Y_MAX - Constants.GOAL_Y_MIN
        right = L - 1 - Constants.GOAL_DEPTH
        pyglet.shapes.Rectangle(0, Constants.GOAL_Y_MIN, Constants.GOAL_DEPTH, mouth, Field.BLUE_GOAL_COLOUR).draw()
        pyglet.shapes.Rectangle(right, Constants.GOAL_Y_MIN, Constants.GOAL_DEPTH, mouth, Field.RED_GOAL_COLOUR).draw()
        for x in (Constants.GOAL_DEPTH, right):
            for y0, y1 in ((0, Constants.GOAL_Y_MIN), (Constants.GOAL_Y_MAX, H)):
                pyglet.shapes.Line(x, y0, x, y1, thickness=2, color=Field.FIELD_COLOUR).draw()
            for y in (Constants.GOAL_Y_MIN, Constants.GOAL_Y_MAX):
                pyglet.shapes.Circle(x, y, Constants.POST_RADIUS, color=Field.POST_COLOUR).draw()
        pyglet.shapes.Line(L / 2, 0, L / 2, H, thickness=4, color=Field.FIELD_COLOUR).draw()
        pyglet.shapes.Circle(L / 2, H / 2, 6, color=Field.FIELD_COLOUR).draw()

        if positions is not None:
            for k, (x, y) in enumerate(positions[:10]):
                colour = Field.TEAM_COLOURS[0 if k < 5 else 1]
                pyglet.shapes.Circle(x, y, Constants.PLAYER_RADIUS, color=colour).draw()
            bx, by = positions[10]
            pyglet.shapes.Circle(bx, by, Constants.BALL_RADIUS, color=Field.BALL_COLOUR).draw()

        mid = L // 2
        labels = [
            (names[0], 34, BLUE, mid - 150, H + 58, "right"),
            (f"{score[0]}  –  {score[1]}", 42, WHITE, mid, H + 54, "center"),
            (names[1], 34, RED, mid + 150, H + 58, "left"),
            (title, 16, GREY, 20, H + 12, "left"),
        ]
        for text, size, colour, x, y, anchor in labels:
            self.label(text, size, colour, x, y, anchor, bold=size > 30).draw()
        # Game clock: a full-width track that fills with gold as the game is played.
        pyglet.shapes.Rectangle(0, H, L, 6, color=(60, 70, 64)).draw()
        pyglet.shapes.Rectangle(0, H, L * progress, 6, color=GOLD[:3]).draw()
        self.label(f"{100 * progress:3.0f}% played", 16, GOLD, L - 20, H + 12, "right").draw()
        if flash:
            text, colour = flash
            self.label(text, 64, colour, mid, H // 2, "center", "center", bold=True).draw()
        if card:
            pyglet.shapes.Rectangle(0, 0, L, H, (0, 0, 0)).draw()
            for k, (text, size, colour) in enumerate(card):
                self.label(text, size, colour, mid, H // 2 + 60 - 90 * k, "center", "center", bold=k == 0).draw()
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, self.pixels)
        return bytes(self.pixels)


def render(replay_path, out_path, renderer, league="League 2", ticks_per_frame=2, fps=50, width=1280):
    data = np.load(replay_path, allow_pickle=True)
    names = [str(n) for n in data["names"]]
    positions, scores = data["positions"], data["scores"]
    round_number = int(data["round"])
    title = f"{league} · round {round_number} · the top two of the round's Swiss"
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{renderer.width}x{renderer.height}",
        "-r", str(fps), "-i", "-",
        "-vf", f"vflip,scale={width}:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(out_path),
    ]
    ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    card = [
        (f"{league} · Round {round_number}", 56, GOLD),
        (f"{names[0]}  vs  {names[1]}", 40, WHITE),
        ("every brain here started from random play", 22, GREY),
    ]
    for _ in range(fps * 3):  # three-second title card
        ffmpeg.stdin.write(renderer.frame(None, (0, 0), names, title, 0.0, card=card))

    flash, flash_until, previous = None, -1, (0, 0)
    total = len(positions)
    for t in range(0, total, ticks_per_frame):
        score = tuple(int(v) for v in scores[t])
        if score != previous:
            scorer = names[0] if score[0] > previous[0] else names[1]
            flash = (f"GOAL!  {scorer}", BLUE if score[0] > previous[0] else RED)
            flash_until, previous = t + 80, score
        ffmpeg.stdin.write(
            renderer.frame(positions[t], score, names, title, t / total, flash if t < flash_until else None)
        )
    final = tuple(int(v) for v in scores[-1])
    result = "draw" if final[0] == final[1] else f"{names[0] if final[0] > final[1] else names[1]} wins"
    end_card = [(f"Full time: {final[0]} – {final[1]}", 56, GOLD), (result, 40, WHITE)]
    for _ in range(fps * 2):
        ffmpeg.stdin.write(renderer.frame(positions[-1], final, names, title, 1.0, card=end_card))
    ffmpeg.stdin.close()
    ffmpeg.wait()
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("videos"))
    parser.add_argument("--league", default="League 2")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    renderer = Renderer()
    for replay in args.replays:
        out = args.out / (replay.stem + ".mp4")
        final = render(replay, out, renderer, league=args.league)
        print(f"{out}  (final score {final[0]}-{final[1]})", flush=True)


if __name__ == "__main__":
    main()
