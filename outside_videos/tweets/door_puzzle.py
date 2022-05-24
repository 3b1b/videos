from manim_imports_ext import *


class DoorPuzzle(InteractiveScene):
    def construct(self):
        # Setup
        squares = Square().get_grid(10, 10)
        squares.set_stroke(WHITE, 1)
        squares.set_height(FRAME_HEIGHT - 1)
        labels = VGroup(*(
            Integer(n, font_size=24).move_to(square)
            for n, square in zip(it.count(1), squares)
        ))

        for square in squares:
            square.n_hits = 0

        self.add(squares, labels)

        # Run operation
        for n in range(1, len(squares) + 1):
            to_toggle = squares[n - 1::n]
            outlines = to_toggle.copy()
            squares.generate_target()
            for square in to_toggle:
                target = squares.target[squares.submobjects.index(square)]
                square.n_hits += 1
                if square.n_hits % 2 == 0:
                    target.set_fill(BLACK, 0)
                else:
                    target.set_fill(BLUE, 0.5)
            outlines = to_toggle.copy()
            outlines.set_fill(opacity=0)
            outlines.set_stroke(YELLOW, 3)

            time_per_anim = min(2, 4 / n)
            self.play(Write(outlines, run_time=time_per_anim))

            self.add(squares, labels)
            self.play(
                MoveToTarget(squares, lag_ratio=0.5),
                FadeOut(outlines),
                run_time=time_per_anim
            )
            self.wait(time_per_anim)
        self.wait(3)
