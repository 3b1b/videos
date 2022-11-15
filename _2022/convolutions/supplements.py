from manim_imports_ext import *
OUTPUT_DIRECTORY = ""


class JuliaVideoFrame(VideoWrapper):
    title = "Lecture on convolutions for image processing"


class SideBySideForContinuousConv(InteractiveScene):
    def construct(self):
        self.add(FullScreenRectangle())
        squares = Square().replicate(2)
        squares.set_fill(BLACK, 1)
        squares.set_stroke(WHITE, 2)
        squares.set_height(6)
        squares.arrange(RIGHT, buff=0.5)
        squares.set_width(FRAME_WIDTH - 1)
        squares.to_edge(DOWN)
        self.add(squares)
