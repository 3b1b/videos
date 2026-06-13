import math

from manim_imports_ext import *
from _2026.cross_entropy.distribution import DynamicInterval
from _2026.cross_entropy.distribution import StackedProbDistribution
from _2026.cross_entropy.next_char import CHAR_ALPHABET
from _2026.cross_entropy.next_char import get_next_char_distribution
from _2026.cross_entropy.next_char import total_information


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
        label_to_bar_height_ratio=0.5,
        context=" ",
    ):
        self.layers = VGroup()
        self.intervals = VGroup()
        self.full_width = width
        self.show_decimal_labels = show_decimal_labels
        self.interval_subdivisions = interval_subdivisions
        self.char_alphabet = char_alphabet
        self.label_to_bar_height_ratio = label_to_bar_height_ratio
        self.context = context

        self.highlight_color_iter = it.cycle(highlight_colors)

        self.unit_interval = DynamicInterval(width=width)
        self.intervals.add(self.unit_interval)
        self.interval_keys = {(1, 0)}  # Of the form (step_size, lower_bound_int)

        self.curr_text = ""  # Start with a neutral context
        self.char_labels_template = Text(char_alphabet)
        self.layers.add(self.get_new_layer(buff=SMALL_BUFF))

        super().__init__(self.intervals, Point(), self.layers)

    def get_new_layer(self, buff=0.1):
        if len(self.layers) == 0:
            mob_above = self.unit_interval[0]
        else:
            mob_above = self.get_letter_bar(self.curr_text[-1])

        full_context = self.context + self.curr_text
        distribution = get_next_char_distribution(full_context)
        if len(full_context.strip()) == 0:
            # Artificially suppress punctuation
            distribution[26:] *= 1e-3
            distribution /= sum(distribution)

        layer = StackedProbDistribution(
            distribution,
            labels=self.char_labels_template.copy(),
            width=mob_above.get_width(),
            label_height_ratio=self.label_to_bar_height_ratio
        )
        layer.next_to(mob_above, DOWN, buff=buff)
        return layer

    def get_letter_bar(self, char, layer_index=-1):
        index = self.char_alphabet.index(char)
        return self.layers[layer_index].bars[index]

    def get_letter_label(self, char, layer_index=-1):
        index = self.char_alphabet.index(char)
        return self.layers[layer_index].labels[index]

    def populate_intervals(self, x_min, x_max):
        log_val = np.log(x_max - x_min) / np.log(self.interval_subdivisions)
        min_scale = int(np.ceil(np.round(log_val, 1)))

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

    def highlight_letter(
        self,
        char,
        color=None,
        layer_index=-1,
        other_bar_opacity=0.35,
        add_to_text=False
    ):
        if color is None:
            color = next(self.highlight_color_iter)
        index = self.char_alphabet.index(char)
        animation = self.layers[layer_index].animate.highlight(
            index,
            color,
            other_bar_opacity=other_bar_opacity
        )

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


class ProbababilityOfAWord(InteractiveScene):
    interval_width = 12

    def construct(self):
        # Initialize layers list
        diagram = ArithmeticCodingDiagram()

        # Show bar chart (TODO, have a model above feeding into this)
        layer = diagram.layers[0]
        bars = layer.bars.copy()
        char_labels = Text(diagram.char_alphabet.replace(" ", "_"), font_size=36)
        for n, label in enumerate(char_labels):
            label.set_x(n * 0.4)
        dec_labels = VGroup(
            DecimalNumber(100 * x, font_size=16, unit="%", num_decimal_places=1)
            for x in layer.distribution
        )
        dec_labels.set_fill(GREY_B)

        char_labels.move_to(2 * DOWN)

        for char, bar, dec in zip(char_labels, bars, dec_labels):
            bar.rotate(90 * DEG)
            bar.stretch(0.5, 0)
            bar.stretch(2, 1)
            bar.match_x(char)
            bar.align_to(char_labels.get_top(), DOWN).shift(SMALL_BUFF * UP)
            dec.next_to(bar, UP, SMALL_BUFF)

        self.add(bars)
        self.add(char_labels)
        self.add(dec_labels)
        self.wait()  # Do something here?

        # Transition to horizontal stack
        frame = self.frame
        new_dec_labels = VGroup(
            DecimalNumber(x, font_size=12, num_decimal_places=2)
            for x in layer.distribution
        )
        new_dec_labels.set_fill(GREY_B)
        for dec, bar in zip(new_dec_labels, layer.bars):
            dec.next_to(bar, DOWN, buff=SMALL_BUFF)
            if dec.get_width() > bar.get_width():
                dec.set_opacity(0)

        kw = dict(run_time=3, lag_ratio=0.05)
        self.play(
            LaggedStart(
                (FadeTransform(dec1, dec2)
                for dec1, dec2 in zip(dec_labels, new_dec_labels)),
                group_type=Group,
                **kw
            ),
            ReplacementTransform(char_labels, layer.labels, **kw),
            ReplacementTransform(bars, layer.bars, **kw),
            frame.animate.set_width(12.5).set_anim_args(**kw),
        )
        self.add(layer)
        self.wait()

        # Show full width
        over_brace = Brace(layer, UP)
        over_brace.refresh_bounding_box()
        over_brace.save_state()
        over_brace.stretch(1e-4, 0, about_edge=LEFT)

        width_label = DecimalNumber(0.00)

        def update_width_label(label):
            label.set_value(over_brace.get_width() / layer.get_width())
            label.next_to(over_brace, UP, buff=0.35)
            label.set_opacity(clip(over_brace.get_width() / 5, 0, 1))

        new_dec_rects = VGroup(
            SurroundingRectangle(dec, buff=0.05).set_stroke(YELLOW, 1, opacity=dec.get_opacity())
            for dec in new_dec_labels
        )

        self.play(
            Restore(over_brace),
            UpdateFromFunc(width_label, update_width_label),
            run_time=2
        )
        self.wait()
        self.add(layer.bars, Point(), layer.labels)
        self.play(
            layer.bars.animate.set_fill(YELLOW, 0.7).set_anim_args(rate_func=there_and_back, lag_ratio=0.05, run_time=3),
            FadeIn(new_dec_rects, rate_func=there_and_back, lag_ratio=0.05, run_time=3),
        )
        self.wait()

        # Show unit interval (Actually, do this later)
        brace_group = VGroup(over_brace, width_label)
        self.play(
            Write(diagram.unit_interval, lag_ratio=0.01),
            brace_group.animate.next_to(diagram.unit_interval[0], UP, MED_LARGE_BUFF),
            frame.animate.set_width(13),
        )
        self.play(FadeOut(brace_group))
        self.wait()

        # Highlight "t" and "p" bars
        def udpate_new_dec_labels(labels):
            for dec, bar in zip(new_dec_labels, layer.bars):
                dec.match_x(bar)

        new_dec_labels.clear_updaters()
        new_dec_labels.add_updater(udpate_new_dec_labels)

        t_bar, p_bar, q_bar = VGroup(
            diagram.get_letter_bar(char)
            for char in "tpq"
        )

        t_brace = Brace(t_bar, DOWN)

        t_brace_label = DecimalNumber(diagram.get_conditional_probability("t"), font_size=36)
        t_brace_label.always.next_to(t_brace, DOWN)

        self.add(new_dec_labels)
        self.add(t_brace)
        self.play(
            diagram.renormalize_animation(0.6, 1.0),
            UpdateFromFunc(t_brace, lambda m: m.become(Brace(t_bar, DOWN))),
            VFadeIn(t_brace),
            UpdateFromFunc(new_dec_labels, udpate_new_dec_labels),
            VFadeOut(new_dec_labels),
            VFadeIn(t_brace_label),
            frame.animate.set_width(FRAME_WIDTH),
            run_time=2,
        )
        self.play(diagram.highlight_letter("t", TEAL, other_bar_opacity=0.5))
        self.wait()

        p_brace = Brace(p_bar, DOWN)
        pre_p_brace = t_brace.copy()
        p_brace_label = t_brace_label.copy()
        p_brace_label.clear_updaters()

        self.play(
            ReplacementTransform(pre_p_brace, p_brace),
            ChangeDecimalToValue(p_brace_label, diagram.get_conditional_probability("p")),
            UpdateFromFunc(Mobject(), lambda m: p_brace_label.match_x(pre_p_brace)),
            diagram.highlight_letter("p", TEAL, other_bar_opacity=0.5)
        )
        self.wait()

        # Show q
        self.play(
            diagram.renormalize_animation(0.6, 0.65),
            UpdateFromFunc(t_brace, lambda m: m.become(Brace(t_bar, DOWN))),
            UpdateFromFunc(p_brace, lambda m: m.become(Brace(p_bar, DOWN))),
            UpdateFromFunc(t_brace_label, lambda m: m.match_x(t_brace)),
            UpdateFromFunc(p_brace_label, lambda m: m.match_x(p_brace)),
        )
        self.remove(t_brace, t_brace_label)
        self.wait()

        q_brace = Brace(q_bar, DOWN)
        q_brace_label = DecimalNumber(
            diagram.get_conditional_probability("q"),
            num_decimal_places=3,
            font_size=36,
        )
        q_brace_label.next_to(q_brace, DOWN)

        self.play(
            TransformFromCopy(p_brace, q_brace),
            FadeTransform(p_brace_label.copy(), q_brace_label),
            diagram.highlight_letter("q", TEAL),
        )
        self.wait()
        self.add(layer.bars, Point(), layer.labels)
        self.play(
            FadeOut(VGroup(p_brace, q_brace, p_brace_label, q_brace_label)),
            layer.bars.animate.set_fill(opacity=1).set_submobject_colors_by_gradient(BLUE_E, TEAL_E).set_stroke(WHITE, 1),
        )
        self.play(diagram.renormalize_animation(0, 1), run_time=1)
        self.wait()

        # Cycle through some letters, ask about P("math")
        def get_letter_prob_label(char):
            brace = Brace(diagram.get_letter_bar(char), DOWN, buff=SMALL_BUFF)
            label = brace.get_tex(f"P(``{char}\")", font_size=36, buff=SMALL_BUFF)
            return VGroup(brace, label)

        prob_label = get_letter_prob_label("a")
        self.play(
            GrowFromCenter(prob_label[0]),
            Write(prob_label[1]),
        )
        for idx in range(1, 13):
            char = diagram.char_alphabet[idx]
            new_label = get_letter_prob_label(char)
            self.play(
                ReplacementTransform(prob_label[0], new_label[0]),
                FadeTransformPieces(prob_label[1], new_label[1]),
            )
            prob_label = new_label
        self.wait()

        p_math = Tex("P(``math\")?")
        p_math.to_edge(UP)
        self.play(FadeIn(p_math, UP))
        self.wait()

        # Zoom in on the "m"
        self.play(diagram.highlight_letter("m"))
        self.play(FadeOut(prob_label))
        self.play(
            diagram.zoom_in_on_letter("m"),
            p_math.animate.set_opacity(0.5).to_corner(UL),
        )
        self.play(diagram.fade_in_new_layer())
        self.wait()

        # Earlier tests
        self.add(diagram)
        for letter in "mathematics":
            self.play(diagram.highlight_letter(letter, color=TEAL))
            self.play(diagram.zoom_in_on_letter(letter))
            self.play(
                diagram.fade_in_new_layer(),
                self.frame.animate.shift(0.1 * DOWN)
            )

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


class SimpleZoom2(InteractiveScene):
    def construct(self):
        # Test
        diagram = ArithmeticCodingDiagram()
        word = "compress"
        self.add(diagram)
        for letter in word:
            self.play(diagram.highlight_letter(letter, color=BLUE_E))
            self.play(
                diagram.zoom_in_on_letter(letter),
                run_time=1
            )
            self.play(
                diagram.fade_in_new_layer(),
                self.frame.animate.shift(0.5 * diagram.layers[0].get_height() * DOWN)
            )

        # Calculate
