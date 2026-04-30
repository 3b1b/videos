from __future__ import annotations

import math

from manim_imports_ext import *


class StackedProbDistribution(VGroup):
    def __init__(
        self,
        distribution,
        width=12,
        height=0.5,
        fill_colors=(BLUE_E, TEAL_E),
        fill_opacity=1,
        stroke_width=1,
        stroke_color=WHITE,
        labels=None,
        label_height_ratio=0.7,
        label_width_ratio=0.8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.distribution = distribution
        self.bar_color_bounds = fill_colors
        self.label_height_ratio = label_height_ratio
        self.label_width_ratio = label_width_ratio

        # Set bars with dummy values
        self.bars = VGroup(
            Rectangle().set_stroke(stroke_color, stroke_width).set_fill(color, fill_opacity)
            for color in fill_colors
        )
        self.bars.arrange(RIGHT, buff=0)
        self.bars.set_shape(width, height)

        # Initialize labels as empty
        self.labels = VGroup()

        self.add(self.bars, self.labels)

        self.set_distribution(distribution)
        if labels is not None:
            self.set_labels(labels)

    def set_labels(self, labels: VMobject):
        self.labels.set_submobjects(labels)
        self.original_labels = labels.copy()
        self.reposition_labels()
        return self

    def reposition_labels(self):
        self.labels.become(self.original_labels)
        self.labels.set_height(self.bars.get_height() * self.label_height_ratio)
        self.labels.move_to(self.bars)
        for label, bar in zip(self.labels, self.bars):
            label.match_x(bar)
            fill_opacity = float(label.get_width() < bar.get_width() * self.label_width_ratio)
            label.set_fill(opacity=fill_opacity)
        return self

    def set_distribution(self, distribution):
        center = self.bars.get_center().copy()
        width, height = self.bars.get_shape()[:2]
        n_bars = len(distribution)
        bar_style = self.bars[0].get_style()

        if len(self.bars) != n_bars:
            self.bars.set_submobjects([Rectangle() for n in range(n_bars)])

        color_range = color_gradient(self.bar_color_bounds, len(distribution))

        for bar, prob, color in zip(self.bars, distribution, color_range):
            bar.set_shape(width * prob, height)
            bar.set_style(**bar_style)
            bar.set_fill(color)
        self.bars.arrange(RIGHT, buff=0)
        self.bars.move_to(center)

        if len(self.labels) > 0:
            self.reposition_labels()

        return self

    def highlight(self, index, color=None, other_bar_opacity=0.35):
        self.bars.set_fill(opacity=other_bar_opacity)
        self.bars[index].set_fill(color, opacity=1)
        return self

    def renormalize_around(self, index: int):
        width, height = self.get_shape()[:2]
        center = self.get_center().copy()
        bar_width = self.bars[index].get_width()
        self.bars.stretch(width / bar_width, 0)
        self.bars.shift(center - self.bars[index].get_center())
        self.reposition_labels()
        return self

    def stretch(self, factor, dim, **kwargs):
        super().stretch(factor, dim, **kwargs)
        if dim == 0:
            self.reposition_labels()
        return self


class DynamicInterval(UnitInterval):
    def __init__(
        self,
        x_range=(0, 1),
        width=12,
        subdivisions=10,
        include_numbers=True,
        include_endpoint_numbers=True,
        width_fade_range=(4, 6),
        number_font_size=24,
        **kwargs
    ):
        self.width_fade_range = width_fade_range

        x_min, x_max = x_range[0], x_range[1]
        step = (x_max - x_min) / subdivisions
        num_decimal_places = max(int(np.round(-np.log(step) / np.log(subdivisions))), 0)

        super().__init__(
            x_range=(x_min, x_max, step),
            width=width,
            big_tick_numbers=[x_min, x_max],
            **kwargs
        )
        label_values = [x_min + step * n for n in range(subdivisions + 1)]
        if not include_endpoint_numbers:
            label_values = label_values[1:-1]
        self.add_numbers(
            label_values,
            direction=UP,
            num_decimal_places=num_decimal_places,
            font_size=number_font_size,
        )
        self.original_numbers = self.numbers.copy()

    def reposition_labels(self):
        self.numbers.become(self.original_numbers)
        for label, original in zip(self.numbers, self.original_numbers):
            label.match_x(self.n2p(original.get_value()))
        return self

    def update_opacity_from_width(self):
        w_min, w_max = self.width_fade_range
        opacity = clip(inverse_interpolate(w_min, w_max, self.get_width()), 0, 1)
        self.set_stroke(opacity=opacity)
        self.numbers.set_fill(opacity=opacity)
        return self

    def stretch(self, factor, dim, **kwargs):
        super().stretch(factor, dim, **kwargs)
        if dim == 0:
            self.reposition_labels()
        self.update_opacity_from_width()
        return self
