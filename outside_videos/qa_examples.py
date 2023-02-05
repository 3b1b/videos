from manim_imports_ext import *


class DemoScene(InteractiveScene):
    def construct(self):
        # Demo animation
        square = Square()
        square.set_fill(BLUE, 0.5)
        square.set_stroke(WHITE, 1)

        grid = square.get_grid(10, 10, buff=0.5)
        grid.set_height(7)

        labels = index_labels(grid)

        self.add(grid)
        self.add(labels)

        # Animations
        def flip(square):
            if square.get_fill_color() == BLUE:
                target_color = GREY_C
            else:
                target_color = BLUE
            return square.animate.set_color(target_color).flip(RIGHT)

        for n in range(2, 100):
            highlights = grid[::n].copy()
            highlights.set_stroke(RED, 3)
            highlights.set_fill(opacity=0)
            self.play(
                ShowCreation(highlights, lag_ratio=0.05),
                run_time = 1 / math.sqrt(n),
            )
            self.wait()
            self.remove(labels)
            self.play(
                LaggedStartMap(flip, grid[::n], lag_ratio=0.05),
                FadeOut(highlights),
                Animation(labels),
                run_time = 1 / math.sqrt(n),
            )

        # New
        randy = PiCreature()
        self.play(randy.change("hooray"))
