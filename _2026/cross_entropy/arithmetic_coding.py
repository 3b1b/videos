import math

from manim_imports_ext import *
from _2026.cross_entropy.distribution import DynamicInterval
from _2026.cross_entropy.distribution import StackedProbDistribution
from _2026.cross_entropy.next_char import CHAR_ALPHABET
from _2026.cross_entropy.next_char import get_next_char_distribution


def get_random_distribution(length):
    dist = np.random.random(30)
    dist /= sum(dist)
    return dist


class ArithmeticCodingDiagram(Group):
    def __init__(
        self,
        width=12,
        buff_to_bars=0.25,
        show_decimal_labels=True,
        interval_subdivisions=10,
        char_alphabet=CHAR_ALPHABET,
        highlight_colors=(GREEN_E, GREEN_D),
        context=" ",
    ):
        self.layers = VGroup()
        self.intervals = VGroup()
        self.full_width = width
        self.show_decimal_labels = show_decimal_labels
        self.interval_subdivisions = interval_subdivisions
        self.char_alphabet = char_alphabet
        self.context = context

        self.highlight_color_iter = it.cycle(highlight_colors)

        self.unit_interval = DynamicInterval(width=width)
        self.intervals.add(self.unit_interval)
        self.interval_keys = {(1, 0)}  # Of the form (step_size, lower_bound_int)

        self.curr_text = ""  # Start with a neutral context
        self.char_labels_template = Text(char_alphabet, font_size=24)
        self.layers.add(self.get_new_layer(buff=SMALL_BUFF))

        super().__init__(self.intervals, Point(), self.layers)

    def get_letter_bar(self, char, layer_index=-1):
        index = self.char_alphabet.index(char)
        return self.layers[layer_index].bars[index]

    def get_letter_label(self, char, layer_index=-1):
        index = self.char_alphabet.index(char)
        return self.layers[layer_index].labels[index]

    def populate_intervals(self, x_min, x_max):
        min_scale = int(np.round(np.log(x_max - x_min) / np.log(self.interval_subdivisions)))

        for scale in range(-1, min_scale - 1, -1):
            step = self.interval_subdivisions**(scale)
            for lower_bound_int in range(int(x_min / step) - 1, int(x_max / step) + 1):
                interval_key = (step, lower_bound_int)
                if interval_key in self.interval_keys:
                    continue
                low = lower_bound_int * step
                high = (lower_bound_int + 1) * step
                interval = DynamicInterval(
                    x_range=(low, high),
                    width=get_norm(self.unit_interval.n2p(high) - self.unit_interval.n2p(low)),
                    subdivisions=self.interval_subdivisions,
                    include_numbers=self.show_decimal_labels,
                    include_endpoint_numbers=False,
                    number_font_size=24 + scale * 2,
                )
                interval.shift(self.unit_interval.n2p(low) - interval.n2p(low))
                self.intervals.add(interval)
                self.interval_keys.add(interval_key)
        return self

    def get_new_layer(self, buff=0.1):
        if len(self.layers) == 0:
            mob_above = self.unit_interval[0]
        else:
            mob_above = self.get_letter_bar(self.curr_text[-1])

        distribution = get_next_char_distribution(self.context + self.curr_text)
        layer = StackedProbDistribution(
            distribution,
            labels=self.char_labels_template.copy(),
            width=mob_above.get_width()
        )
        layer.next_to(mob_above, DOWN, buff=buff)
        return layer

    def get_conditional_probability(self, char, layer_index=-1):
        idx = self.char_alphabet.index(char)
        return self.layers[layer_index].distribution[idx]

    def get_absolute_information(self, text):
        result = 0
        for layer, char in zip(self.layers, text):
            result += -math.log2(layer.distribution[self.char_alphabet.index(char)])
        return result

    # Animations
    def renormalize_animation(
        self,
        x_min,
        x_max,
        run_time=3,
        center=ORIGIN,
        center_curr_text=False,
        **kwargs
    ):
        big_interval = self.intervals[0]
        x_mid = (x_min + x_max) / 2
        p_left, p_mid, p_right = [big_interval.n2p(x) for x in (x_min, x_mid, x_max)]

        stretch_factor = self.full_width / get_norm(p_right - p_left)
        x_shift = (center - p_mid)[0] * RIGHT

        self.populate_intervals(x_min, x_max)
        for interval in self.intervals:
            interval.update_opacity_from_width()

        target = self.copy()

        for interval in target.intervals:
            interval.shift(x_shift)
            interval.stretch(stretch_factor, 0, about_point=ORIGIN)
        for layer in target.layers:
            layer.shift(x_shift)
            layer.stretch(stretch_factor, 0, about_point=ORIGIN)
            layer.reposition_labels()
        if center_curr_text:
            for layer, char in zip(target.layers, self.curr_text):
                label = layer[1][self.char_alphabet.index(char)]
                label.match_x(center)

        return Transform(self, target, run_time=run_time, **kwargs)

    def highlight_letter(self, char, color=None, layer_index=-1, add_to_text=False):
        if color is None:
            color = next(self.highlight_color_iter)
        index = self.char_alphabet.index(char)
        animation = self.layers[layer_index].animate.highlight(index, color)

        if add_to_text:
            self.curr_text += char

        return animation

    def zoom_in_on_letter(self, char, layer_index=-1, add_to_text=True, **kwargs):
        bar = self.get_letter_bar(char, layer_index)
        x_min = self.unit_interval.p2n(bar.get_left())
        x_max = self.unit_interval.p2n(bar.get_right())
        if add_to_text:
            self.curr_text += char
        return self.renormalize_animation(x_min, x_max, center_curr_text=True, **kwargs)

    def fade_in_new_layer(self, char=None, buff=0):
        layer = self.get_new_layer()
        self.layers.add(layer)
        return FadeIn(layer)


class DiagramTest(InteractiveScene):
    interval_width = 12

    def construct(self):
        # Initialize layers list
        diagram = ArithmeticCodingDiagram()

        self.add(diagram)
        for letter in "mathematic":
            self.play(diagram.highlight_letter(letter))
            self.play(diagram.zoom_in_on_letter(letter))
            self.play(diagram.fade_in_new_layer())

        # A few tests
        diagram.get_conditional_probability("h")
        diagram.get_absolute_information("math")
        bar = diagram.get_letter_bar("e")
        -np.log2(bar.get_width() / diagram.unit_interval[0].get_width())
        self.play(diagram.zoom_in_on_letter("p", add_to_text=False))

        # Some custom bounds
        self.play(diagram.renormalize_animation(0.4, 0.5))
        self.play(diagram.renormalize_animation(0.45, 0.46))
        self.play(diagram.renormalize_animation(0.4, 0.5))
        self.play(diagram.renormalize_animation(0, 1))
