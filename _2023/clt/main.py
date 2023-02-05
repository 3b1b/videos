from __future__ import annotations

from manim_imports_ext import *

from _2022.convolutions.continuous import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
    from manimlib.typing import ManimColor


class ChartBars(VGroup):
    def __init__(
        self,
        axes,
        values: Sequence[float | int],
        xs: Sequence[float] | None = None,
        width_ratio: float = 1.0,
        offset: float = 0.5,
        fill_color: ManimColor = BLUE,
        fill_opacity: float = 0.5,
        stroke_color: ManimColor = WHITE,
        stroke_width: float = 1.0
    ):
        xs = xs if xs is not None else np.arange(*axes.x_range)

        self.x_to_index = dict(zip(xs, it.count()))
        x_step = xs[1] - xs[0]
        x_unit = axes.x_axis.get_unit_size()
        y_unit = axes.y_axis.get_unit_size()

        width = width_ratio * x_unit * x_step

        # Create a list of rectangles arranged side by side,
        # one for each x value
        rects = []
        epsilon = 1e-8
        for x, y in zip(xs, values):
            rect = Rectangle(
                width=width,
                height=max(y * y_unit, epsilon),
                fill_color=fill_color,
                fill_opacity=fill_opacity,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
            )
            rect.move_to(axes.c2p(x + offset * x_step, 0), DOWN)
            rects.append(rect)
        super().__init__(*rects)
        self.axes = axes
        self.xs = xs
        self.set_values(values)

    def set_values(self, values: Iterable[float | int]):
        y_unit = self.axes.y_axis.get_unit_size()
        for rect, value in zip(self, values):
            rect.set_height(
                y_unit * value,
                stretch=True,
                about_edge=DOWN,
            )


def get_die_distribution_chart(
    dist: list[float],
    die_config=dict(),
    axes_config=dict(
        width=4,
        height=3,
    ),
    max_value: int = 6
):
    axes = Axes((0, max_value), (0, 1, 0.2), **axes_config)
    axes.y_axis.add_numbers(font_size=16, num_decimal_places=1)
    bars = ChartBars(axes, dist)

    dice = VGroup()
    for n in range(1, max_value + 1):
        die = DieFace(n, **die_config)
        die.set_width(0.5 * axes.x_axis.get_unit_size())
        die.next_to(axes.c2p(n - 0.5, 0), DOWN, SMALL_BUFF)
        dice.add(die)

    return VGroup(axes, bars, dice)

# Scenes


class HistogramTest(InteractiveScene):
    def construct(self):
        # Test
        axes = Axes((0, 10), (0, 1, 0.25), width=12, height=4)
        axes.y_axis.add_numbers(num_decimal_places=2)
        axes.x_axis.add_numbers(num_decimal_places=0)

        data = np.array([5, 5, 3, 4, 2, 3, 1])
        x_range = np.arange(1, 7)
        values = np.array([sum(data == x) for x in x_range], dtype=float)
        values /= values.sum()
        bars = ChartBars(axes, values, x_range)

        chart = VGroup(axes, bars)
        self.add(chart)

        # Test animation
        self.play(Write(chart))
        self.play(chart.animate.set_width(5).to_corner(UL), run_time=2)

        # Second insert
        randy = Randolph()
        self.play(FadeIn(randy))
        self.play(randy.change("pondering"))
        self.play(Blink(randy))


class DiceSimulations(InteractiveScene):
    n_dice = 10
    n_samples = 100
    distribution = [1 / 6] * 6
    # samples = 4

    def construct(self):
        # Die distribution (TODO, animate in?)
        die_dist = get_die_distribution_chart(self.distribution)
        die_dist.set_height(3)
        die_dist.to_corner(UL)

        axes, bars, dice = die_dist

        self.play(
            Write(axes),
            LaggedStartMap(GrowFromEdge, bars, edge=DOWN),
            Write(dice),
        )

        self.add(die_dist)


class GaussConvolutions(Convolutions):
    conv_y_stretch_factor = 1.0

    def construct(self):
        super().construct()

    def f(self, x):
        return np.exp(-x**2)

    def g(self, x):
        return np.exp(-x**2)
