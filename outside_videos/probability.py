from manim_imports_ext import *


class StacksApproachBellCurve(Scene):
    CONFIG = {
        "n_iterations": 70,
    }

    def construct(self):
        bar = Square(side_length=1)
        bar.set_fill(BLUE, 1)
        bar.set_stroke(BLUE, 1)
        bars = VGroup(bar)

        max_width = FRAME_WIDTH - 2
        max_height = FRAME_Y_RADIUS - 1.5

        for x in range(self.n_iterations):

            bars_copy = bars.copy()

            #Copy and shift
            for mob, vect in (bars, DOWN), (bars_copy, UP):
                mob.generate_target()
                if mob.target.get_height() > max_height:
                    mob.target.stretch_to_fit_height(max_height)
                if mob.target.get_width() > max_width:
                    lx1 = mob.target[1].get_left()[0]
                    rx0 = mob.target[0].get_right()[0]
                    curr_buff = lx1 - rx0
                    mob.target.arrange(
                        RIGHT, buff=0.9 * curr_buff,
                        aligned_edge=DOWN
                    )
                    mob.target.stretch_to_fit_width(max_width)
                mob.target.next_to(ORIGIN, vect, MED_LARGE_BUFF)
            colors = color_gradient([BLUE, YELLOW], len(bars) + 1)
            for color, bar in zip(colors, bars.target):
                bar.set_color(color)
            for color, bar in zip(colors[1:], bars_copy.target):
                bar.set_color(color)
            bars_copy.set_fill(opacity=0)
            bars_copy.set_stroke(width=0)
            if x == 0:
                distance = 1.5
            else:
                cx1 = bars.target[-1].get_center()[0]
                cx0 = bars.target[0].get_center()[0]
                distance = (cx1 - cx0) / (len(bars) - 1)
            self.play(*list(map(MoveToTarget, [bars, bars_copy])))
            self.play(
                bars.shift, distance * LEFT / 2,
                bars_copy.shift, distance * RIGHT / 2,
            )

            # Stack
            bars_copy.generate_target()
            for i in range(len(bars) - 1):
                top_bar = bars_copy.target[i]
                low_bar = bars[i + 1]
                top_bar.move_to(low_bar.get_top(), DOWN)
            bars_copy.target[-1].align_to(bars, DOWN)

            self.play(MoveToTarget(
                bars_copy, lag_ratio=0.5,
                run_time=np.sqrt(x + 1)
            ))

            # Resize lower bars
            for top_bar, low_bar in zip(bars_copy[:-1], bars[1:]):
                bottom = low_bar.get_bottom()
                low_bar.replace(
                    VGroup(low_bar, top_bar),
                    stretch=True
                )
                low_bar.move_to(bottom, DOWN)
            bars.add(bars_copy[-1])
            self.remove(bars_copy)
            self.add(bars)
