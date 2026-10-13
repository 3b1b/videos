import math
from functools import lru_cache

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


@lru_cache(maxsize=None)
def cached_next_char_distribution(context):
    return np.array(get_next_char_distribution(context), dtype=float)


class RenormalizeAnimation(MoveToTarget):
    def begin(self):
        self.mobject.is_renormalizing = True
        self.mobject.ids_in_transform = {id(mob) for mob in self.mobject.get_family()}
        super().begin()

    def finish(self):
        super().finish()
        self.mobject.is_renormalizing = False
        self.mobject.ids_in_transform = set()
        self.mobject.target = None


def move_tracker_through_zoom(diagram, tracker, end_value, run_time, rate_func=smooth):
    def screen_x(unit_interval, value):
        return unit_interval.n2p(value)[0]

    start_x = screen_x(diagram.unit_interval, tracker.get_value())
    end_x = screen_x(diagram.target.unit_interval, end_value)

    def update(mob, alpha):
        x = interpolate(start_x, end_x, alpha)
        x0 = screen_x(diagram.unit_interval, 0)
        x1 = screen_x(diagram.unit_interval, 1)
        mob.set_value((x - x0) / (x1 - x0))

    return UpdateFromAlphaFunc(tracker, update, run_time=run_time, rate_func=rate_func)


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
        self.branches = VGroup()
        self.layer_by_prefix = dict()
        self.min_branch_width = 0
        self.is_renormalizing = False
        self.ids_in_transform = set()
        self.intervals_hidden = False
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

        super().__init__(self.intervals, Point(), self.branches, self.layers)

    def get_distribution(self, prefix):
        return cached_next_char_distribution(self.context + prefix).copy()

    def get_new_layer(self, buff=0.1):
        if len(self.layers) == 0:
            mob_above = self.unit_interval[0]
        else:
            mob_above = self.get_letter_bar(self.curr_text[-1])
        return self.make_layer(self.curr_text, mob_above, buff=buff)

    def make_layer(self, prefix, mob_above, buff=0.1, build_width=None):
        distribution = self.get_distribution(prefix)
        target_width = mob_above.get_width()
        layer = StackedProbDistribution(
            distribution,
            labels=self.char_labels_template.copy(),
            width=build_width or target_width,
            label_height_ratio=self.label_to_bar_height_ratio
        )
        if build_width is not None:
            layer.stretch(target_width / layer.bars.get_width(), 0)
            layer.reposition_labels()
        layer.next_to(mob_above, DOWN, buff=buff)

        layer.prefix = prefix
        layer.default_bar_style = [
            (bar.get_fill_color(), bar.get_fill_opacity())
            for bar in layer.bars
        ]
        first_bar = layer.bars[0]
        layer.default_stroke_style = (
            first_bar.get_stroke_color(),
            float(first_bar.get_stroke_width()),
            float(first_bar.get_stroke_opacity()),
        )
        self.layer_by_prefix[prefix] = layer
        return layer

    def make_branch_layer(self, prefix, parent_bar, buff=0.1):
        layer = self.make_layer(prefix, parent_bar, buff=buff, build_width=self.full_width)
        self.update_bar_strokes(layer)
        layer.highlight_state = "default"
        layer.is_branch = True
        layer.is_visible = False
        return layer

    def update_visible_path(self, value):
        x = self.unit_interval.n2p(value)[0]
        n_rows = len(self.layers)
        path = {""}
        prefix = ""
        layer = self.layer_by_prefix[""]
        while True:
            index = int(np.clip(layer.x_value_to_index(x), 0, len(layer.bars) - 1))
            child_prefix = prefix + self.char_alphabet[index]
            if len(child_prefix) >= n_rows:
                break
            child = self.layer_by_prefix.get(child_prefix)
            if child is None:
                bar = layer.bars[index]
                if bar.get_width() <= self.min_branch_width:
                    break
                child = self.make_branch_layer(child_prefix, bar)
            self.set_layer_visibility(child, True)
            if self.is_renormalizing and id(child.bars) not in self.ids_in_transform:
                self.fit_layer_to_parent(child)
            path.add(child_prefix)
            prefix, layer = child_prefix, child

        for other_prefix, other_layer in self.layer_by_prefix.items():
            if other_prefix not in path:
                self.set_layer_visibility(other_layer, False)

    def show_all_layers(self):
        for prefix in sorted(self.layer_by_prefix, key=len):
            if prefix:
                self.set_layer_visibility(self.layer_by_prefix[prefix], True)

    def set_layer_visibility(self, layer, visible):
        if getattr(layer, "is_visible", True) == visible:
            return
        is_branch = getattr(layer, "is_branch", False)
        if visible:
            if is_branch:
                self.branches.add(layer)
            else:
                layer.set_submobjects(layer.stashed_submobjects)
            self.fit_layer_to_parent(layer)
        else:
            self.reset_layer_style(layer)
            layer.highlight_state = "default"
            if is_branch:
                self.branches.remove(layer)
            else:
                layer.stashed_submobjects = list(layer.submobjects)
                layer.set_submobjects([])
        layer.is_visible = visible

    def fit_layer_to_parent(self, layer):
        parent = self.layer_by_prefix[layer.prefix[:-1]]
        bar = parent.bars[self.char_alphabet.index(layer.prefix[-1])]
        return self.align_layer_to_bar(layer, bar)

    def align_layer_to_bar(self, layer, bar, update_strokes=True):
        layer.stretch(bar.get_width() / layer.bars.get_width(), 0)
        layer.shift((bar.get_left()[0] - layer.bars.get_left()[0]) * RIGHT)
        layer.reposition_labels()
        if update_strokes:
            self.update_bar_strokes(layer)
        return layer

    def update_bar_strokes(self, layer, min_width=0.05, sliver_stroke_width=1.0):
        stroke_color, stroke_width, stroke_opacity = layer.default_stroke_style
        for bar in layer.bars:
            alpha = clip(bar.get_width() / min_width, 0, 1)
            bar.set_stroke(
                color=interpolate_color(bar.get_fill_color(), stroke_color, alpha),
                width=interpolate(sliver_stroke_width, stroke_width, alpha),
                opacity=interpolate(bar.get_fill_opacity(), stroke_opacity, alpha),
            )
        return layer

    def get_sweep_range(self, mob, buff=0.01):
        view = self.target if getattr(self, "target", None) is not None else self
        view_mob = view.get_family()[self.get_family().index(mob)]
        low = view.unit_interval.p2n(view_mob.get_left() + buff * RIGHT)
        high = view.unit_interval.p2n(view_mob.get_right() + buff * LEFT)
        return low, high

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

    def get_text_interval(self, text):
        low, high = 0.0, 1.0
        for i, char in enumerate(text):
            dist = self.get_distribution(text[:i])
            dist = dist / dist.sum()
            cumulative = np.cumsum(dist)
            index = self.char_alphabet.index(char)
            low, high = (
                low + (high - low) * (cumulative[index] - dist[index]),
                low + (high - low) * cumulative[index],
            )
        return low, high

    @staticmethod
    def get_binary_code(low, high, max_bits=60, contain_interval=False):
        for n_bits in range(1, max_bits + 1):
            k = math.ceil(low * 2**n_bits)
            end = (k + 1) / 2**n_bits if contain_interval else k / 2**n_bits
            if end <= high if contain_interval else end < high:
                return format(k, f"0{n_bits}b")
        raise ValueError("Interval too small to encode")

    def show_text_path(self, text):
        path = {""}
        for i in range(1, len(text) + 1):
            prefix = text[:i]
            if prefix not in self.layer_by_prefix:
                parent = self.layer_by_prefix[text[:i - 1]]
                self.make_branch_layer(prefix, parent.bars[self.char_alphabet.index(text[i - 1])])
            self.set_layer_visibility(self.layer_by_prefix[prefix], True)
            path.add(prefix)
        for prefix, layer in self.layer_by_prefix.items():
            if prefix not in path:
                self.set_layer_visibility(layer, False)
        return [self.layer_by_prefix[prefix] for prefix in sorted(path, key=len)]

    @staticmethod
    def get_binary_digits(value, n_bits):
        value = clip(value, 0, 1 - 1e-12)
        bits = ""
        for _ in range(n_bits):
            value *= 2
            bit = int(value)
            bits += str(bit)
            value -= bit
        return bits

    def get_layer_chain(self, value):
        x = self.unit_interval.n2p(value)[0]
        chain = []
        prefix = ""
        while prefix in self.layer_by_prefix:
            layer = self.layer_by_prefix[prefix]
            if not getattr(layer, "is_visible", True):
                break
            index = int(np.clip(layer.x_value_to_index(x), 0, len(layer.bars) - 1))
            chain.append((layer, index))
            prefix += self.char_alphabet[index]
        return chain, prefix

    def decode_value(self, value, low, high, prefix="", n_chars=1):
        result = ""
        for _ in range(n_chars):
            dist = self.get_distribution(prefix + result)
            dist = dist / dist.sum()
            cumulative = np.cumsum(dist)
            alpha = (value - low) / (high - low) if high > low else 0
            index = int(min(np.searchsorted(cumulative, alpha, side="right"), len(dist) - 1))
            low, high = (
                low + (high - low) * (cumulative[index] - dist[index]),
                low + (high - low) * cumulative[index],
            )
            result += self.char_alphabet[index]
        return result

    def get_word_at_value(self, value, n_chars=None):
        if n_chars is None:
            n_chars = len(self.layers)
        chain, prefix = self.get_layer_chain(value)
        prefix = prefix[:n_chars]
        if len(prefix) < n_chars:
            parent, index = chain[-1]
            bar = parent.bars[index]
            low, high = (self.unit_interval.p2n(point) for point in (bar.get_left(), bar.get_right()))
            prefix += self.decode_value(value, low, high, prefix, n_chars - len(prefix))
        return prefix

    def reset_layer_style(self, layer):
        for bar, (color, opacity) in zip(layer.bars, layer.default_bar_style):
            bar.set_fill(color, opacity)

    def hide_intervals(self):
        self.intervals_hidden = True
        self.intervals.suspend_updating()
        return self.intervals.animate.set_opacity(0)

    def fade_out_layers(self, layers=None):
        if layers is None:
            layers = self.layers
        for layer in layers:
            layer.is_faded = True
        return VGroup(*layers).animate.set_opacity(0)

    def restyle_layer(self, layer, value, unit_interval=None, other_bar_opacity=0.35, center=None):
        if unit_interval is None:
            unit_interval = self.unit_interval
        layer.set_opacity(1)
        self.reset_layer_style(layer)
        layer.reposition_labels()
        x = unit_interval.n2p(value)[0]
        index = int(np.clip(layer.x_value_to_index(x), 0, len(layer.bars) - 1))
        layer.highlight(index, other_bar_opacity=other_bar_opacity)
        if center is not None:
            bar = layer.bars[index]
            if bar.get_left()[0] <= center[0] <= bar.get_right()[0]:
                layer[1][index].match_x(center)
        self.update_bar_strokes(layer)
        return index

    def restore_layer(self, layer, value, other_bar_opacity=0.35, center=None, **kwargs):
        if isinstance(layer, int):
            layer = self.layers[layer]
        layer.is_faded = False
        restyled = layer.copy()
        layer.highlight_state = self.restyle_layer(
            restyled, value, other_bar_opacity=other_bar_opacity, center=center
        )
        return Transform(layer, restyled, **kwargs)

    def restore_layers_in_target(self, layers, value, other_bar_opacity=0.35, center=None):
        target_of = dict(zip(map(id, self.get_family()), self.target.get_family()))
        for layer in layers:
            layer.is_faded = False
            layer.highlight_state = self.restyle_layer(
                target_of[id(layer)], value, self.target.unit_interval,
                other_bar_opacity=other_bar_opacity, center=center,
            )

    def highlight_value(self, value, color=None, other_bar_opacity=0.35):
        chain, _ = self.get_layer_chain(value)
        chain_indices = {id(layer): index for layer, index in chain}
        for layer in self.layer_by_prefix.values():
            if not getattr(layer, "is_visible", True) or getattr(layer, "is_faded", False):
                continue
            new_state = chain_indices.get(id(layer), "default")
            if new_state == "default" and getattr(layer, "highlight_state", None) == "default":
                continue
            self.reset_layer_style(layer)
            if new_state != "default":
                if color is None:
                    layer.highlight(new_state, other_bar_opacity=other_bar_opacity)
                else:
                    layer.highlight(new_state, color, other_bar_opacity=other_bar_opacity)
                self.update_bar_strokes(layer)
            layer.highlight_state = new_state

    # Animations
    def renormalize_animation(
        self,
        x_min,
        x_max,
        run_time=3,
        center=ORIGIN,
        center_curr_text=False,
        center_text=None,
        stretch_factor=None,
        fade_thin_strokes=False,
        point_func=None,
        **kwargs
    ):
        if point_func is None:
            point_func = self.intervals[0].n2p
        x_mid = (x_min + x_max) / 2
        p_left, p_mid, p_right = [point_func(x) for x in (x_min, x_mid, x_max)]

        if stretch_factor is None:
            stretch_factor = self.full_width / get_norm(p_right - p_left)
        x_shift = (center - p_mid)[0] * RIGHT

        if not self.intervals_hidden:
            self.populate_intervals(x_min, x_max)
            for interval in self.intervals:
                interval.update_opacity_from_width()

        self.generate_target()
        for interval in self.target.intervals:
            interval.shift(x_shift)
            interval.stretch(stretch_factor, 0, about_point=ORIGIN)
        for layer in [*self.target.layers, *self.target.branches]:
            if not getattr(layer, "is_visible", True):
                continue
            layer.shift(x_shift)
            layer.stretch(stretch_factor, 0, about_point=ORIGIN)
            layer.reposition_labels()
            if fade_thin_strokes:
                self.update_bar_strokes(layer)

        target_of = dict(zip(map(id, self.get_family()), self.target.get_family()))
        for prefix in sorted(self.layer_by_prefix, key=len):
            if not prefix:
                continue
            layer = self.layer_by_prefix[prefix]
            parent = self.layer_by_prefix.get(prefix[:-1])
            if parent is None or not getattr(layer, "is_visible", True) or not getattr(parent, "is_visible", True):
                continue
            parent_bar = parent.bars[self.char_alphabet.index(prefix[-1])]
            if id(layer) not in target_of or id(parent_bar) not in target_of:
                continue
            self.align_layer_to_bar(
                target_of[id(layer)], target_of[id(parent_bar)],
                update_strokes=fade_thin_strokes,
            )

        if center_curr_text:
            center_text = self.curr_text
        if center_text is not None:
            for i, char in enumerate(center_text):
                layer = self.layer_by_prefix.get(center_text[:i])
                if layer is None or not getattr(layer, "is_visible", True) or id(layer) not in target_of:
                    continue
                target_layer = target_of[id(layer)]
                index = self.char_alphabet.index(char)
                bar = target_layer.bars[index]
                if bar.get_left()[0] <= center[0] <= bar.get_right()[0]:
                    target_layer[1][index].match_x(center)

        for layer in [*self.target.layers, *self.target.branches]:
            if getattr(layer, "is_faded", False):
                layer.set_opacity(0)
        if self.intervals_hidden:
            self.target.intervals.set_opacity(0)

        return RenormalizeAnimation(self, run_time=run_time, **kwargs)

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
        if add_to_text:
            idx = layer_index % len(self.layers)
            if idx >= len(self.curr_text) or self.curr_text[idx] != char:
                self.curr_text = self.curr_text[:idx] + char
        return self.zoom_in_on_letter_range((char, char), layer_index=layer_index, **kwargs)

    def zoom_in_on_letter_range(self, letter_range, layer_index=-1, **kwargs):
        char1, char2 = letter_range
        bar1 = self.get_letter_bar(char1, layer_index)
        bar2 = self.get_letter_bar(char2, layer_index)
        x_min = self.unit_interval.p2n(bar1.get_left())
        x_max = self.unit_interval.p2n(bar2.get_right())
        if x_min > x_max:
            raise ValueError("Second letter in range must be alphabetically after first letter")
        return self.renormalize_animation(x_min, x_max, center_curr_text=True, **kwargs)

    def fade_in_new_layer(self, char=None, buff=0):
        layer = self.get_new_layer()
        self.layers.add(layer)
        return FadeIn(layer)


class IntroduceCharacterModel(InteractiveScene):

    def construct(self):
        # Show a few example distributions
        frame = self.frame
        input_phrase = "compression is "
        n_initial_letters = 1
        char_labels = CHAR_ALPHABET.replace(" ", "_")
        distribution = get_next_char_distribution(input_phrase[:n_initial_letters])
        bar_chart = self.get_bar_chart(distribution, char_labels)

        out_arrow = Line(3 * UP, 3 * RIGHT, path_arc=-90 * DEG, stroke_width=8, buff=0)
        out_arrow.add_tip(width=0.5, length=0.5)
        out_arrow.set_color(GREY_B)
        out_arrow.shift(bar_chart.get_bottom() - out_arrow.get_end() + 5 * UP)
        model = ImageMobject(Path(
            self.file_writer.get_output_file_rootname().parent.parent,
            "Paul Assets", "BigModel.png"
        ))
        model.set_height(5)
        model.rotate(-2 * DEG)
        model.next_to(out_arrow.get_start(), LEFT)
        in_arrow = Vector(RIGHT, thickness=5)
        in_arrow.next_to(model, LEFT)
        in_arrow.scale(1.5, about_edge=LEFT)

        def get_in_text(s, font_size=72):
            result = Text(f"“{s}”", font_size=font_size)
            result.next_to(in_arrow, LEFT, buff=0.15)
            result.shift(0.05 * UP)
            return result

        def get_bar_chart(in_text):
            return self.get_bar_chart(
                distribution=get_next_char_distribution(in_text),
                labels=char_labels
            )

        in_text_mob = get_in_text(input_phrase[:n_initial_letters])

        frame.reorient(0, 0, 0, (-4.50, 2.56, 0.00), 12.89)
        self.add(model, in_text_mob, in_arrow, out_arrow, bar_chart)
        self.wait(0.5)
        for n in range(n_initial_letters + 1, len(input_phrase) + 1):
            in_text_mob.set_submobjects(get_in_text(input_phrase[:n]))
            new_chart = get_bar_chart(input_phrase[:n])
            self.play(ReplacementTransform(bar_chart, new_chart), run_time=0.25)
            bar_chart = new_chart
            self.wait(0.5)

        # Highlight distribution and limited vocabulary
        wide_char_rect = SurroundingRectangle(bar_chart.labels, buff=0.05)
        wide_char_rect.set_stroke(YELLOW, 2)
        char_rects = VGroup(
            wide_char_rect.copy().set_width(label.get_width() + 0.1, stretch=True).match_x(label)
            for label in bar_chart.labels
        )
        alphabet_rects = char_rects[:26]
        punc_rects = char_rects[26:]

        in_rect = SurroundingRectangle(in_text_mob)
        out_rect = SurroundingRectangle(bar_chart)
        in_rect.set_stroke(TEAL, 3)
        out_rect.set_stroke(TEAL, 3)
        out_arrow.set_z_index(1)

        self.play(ShowCreation(in_rect))
        self.wait()
        self.play(
            frame.animate(time_span=(0.5, 2.5)).to_default_state().set_y(-1),
            ReplacementTransform(in_rect, out_rect, run_time=1)
        )
        self.wait()
        self.play(
            FadeOut(out_rect),
            ShowCreation(alphabet_rects, lag_ratio=0.05, run_time=2),
        )
        self.wait()
        self.play(
            FadeOut(alphabet_rects, lag_ratio=0.01),
            ShowCreation(punc_rects, lag_ratio=0.1),
        )
        self.wait()
        self.play(
            frame.animate(run_time=3).reorient(0, 0, 0, (-4.50, 2.56, 0.00), 12.89),
            FadeOut(punc_rects, time_span=(1, 2)),
        )

        # Show blank space input
        for n in range(len(input_phrase), -1, -1):
            in_text_mob.set_submobjects(get_in_text(input_phrase[:n]))
            new_chart = get_bar_chart(input_phrase[:n] or " ")
            self.play(ReplacementTransform(bar_chart, new_chart), run_time=0.1)
            bar_chart = new_chart
            self.wait(0.1)
        self.wait()

        self.play(
            frame.animate.reorient(0, 0, 0, (-1.69, 1.39, 0.00), 9.99),
            LaggedStart(*(
                bar.animate(rate_func=there_and_back).set_fill(YELLOW)
                for bar in bar_chart.bars
            ), lag_ratio=0.25),
            LaggedStart(*(
                FadeIn(rect, rate_func=there_and_back)
                for rect in char_rects
            ), lag_ratio=0.25),
            run_time=7
        )

        # Transition to horizontal stack
        diagram = ArithmeticCodingDiagram()
        layer = diagram.layers[0]
        new_dec_labels = self.get_stacked_distribution_dec_labels(layer)

        for bar in bar_chart.bars:
            og = bar.copy()
            bar.rotate(90 * DEG)
            bar.replace(og, stretch=True)

        kw = dict(run_time=3, lag_ratio=0.05)
        self.play(
            LaggedStart(
                (FadeTransform(dec1, dec2)
                 for dec1, dec2 in zip(bar_chart.dec_labels, new_dec_labels)),
                group_type=Group,
                **kw
            ),
            ReplacementTransform(bar_chart.bars, layer.bars, **kw),
            ReplacementTransform(bar_chart.labels, layer.labels, **kw),
            LaggedStartMap(FadeOut, Group(in_text_mob, in_arrow, model, out_arrow), shift=UP, time_span=(1, 3)),
            frame.animate(run_time=3).to_default_state().set_y(-1),
        )
        self.add(layer)
        self.wait()

    def get_bar_chart(
        self,
        distribution,
        labels,
        bar_colors=(BLUE_E, TEAL),
        spacing=0.15,
        bar_width=0.25,
        unit_height=15,
        max_height=5,
        label_font_size=36,
        dec_font_size=16,
        bottom=2 * DOWN
    ):
        bars = VGroup(
            Rectangle(width=bar_width, height=prob * unit_height)
            for prob in distribution
        )
        bars.arrange(RIGHT, buff=spacing, aligned_edge=DOWN)
        bars.set_fill(bar_colors, 1, gradient_direction=RIGHT)
        bars.set_stroke(WHITE, 1)
        bars.set_max_height(max_height, stretch=True)
        bars.center()

        dec_labels = VGroup(
            DecimalNumber(100 * prob, font_size=dec_font_size, unit="%", num_decimal_places=1)
            for prob in distribution
        )
        dec_labels.set_fill(GREY_B)
        char_labels = Text(labels, font_size=label_font_size)
        char_labels.next_to(bars, DOWN, SMALL_BUFF)

        for bar, char, dec in zip(bars, char_labels, dec_labels):
            char.match_x(bar)
            dec.next_to(bar, UP, SMALL_BUFF)

        result = VGroup(bars, char_labels, dec_labels)
        result.labels = char_labels
        result.dec_labels = dec_labels
        result.bars = bars
        result.move_to(bottom, DOWN)
        return result

    def get_stacked_distribution_dec_labels(self, layer, font_size=12, num_decimal_places=2):
        dec_labels = VGroup(
            DecimalNumber(x, font_size=font_size, num_decimal_places=num_decimal_places)
            for x in layer.distribution
        )
        dec_labels.set_fill(GREY_B)
        for dec, bar in zip(dec_labels, layer.bars):
            dec.next_to(bar, DOWN, buff=SMALL_BUFF)
            if dec.get_width() > bar.get_width():
                dec.set_opacity(0)
        return dec_labels

    def get_stacked_distribution_pct_labels(self, layer, font_size=12, num_decimal_places=2):
        pct_labels = VGroup(
            DecimalNumber(x * 100, font_size=font_size, num_decimal_places=num_decimal_places, unit=R"\%")
            for x in layer.distribution
        )
        pct_labels.set_fill(GREY_B)
        for pct, bar in zip(pct_labels, layer.bars):
            pct.next_to(bar, DOWN, buff=SMALL_BUFF)
            if pct.get_width() > bar.get_width():
                pct.set_opacity(0)
        return pct_labels

    def old_animations(self):
        for group in bar_chart:
            for mob in group:
                mob.save_state()
                mob.scale(0)
                mob.set_opacity(0)
                mob.move_to(model.get_right())

        frame.reorient(0, 0, 0, (-3.64, 2.97, 0.00), 11.64)
        self.add(model, in_text, in_arrow, out_arrow)
        kw = dict(lag_ratio=0.02, path_arc=-45 * DEG, run_time=2)
        self.play(*(
            LaggedStartMap(Restore, group, **kw)
            for group in bar_chart
        ))
        self.wait()


class ArithmeticCodingExplanation(IntroduceCharacterModel):
    interval_width = 12

    def construct(self):
        # Add diagram
        frame = self.frame
        diagram = ArithmeticCodingDiagram()
        layer = diagram.layers[0]
        dec_labels = self.get_stacked_distribution_dec_labels(layer)

        frame.set_y(-1)
        self.add(layer, dec_labels)

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
            for dec in dec_labels
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

        # Show unit interval
        brace_group = VGroup(over_brace, width_label)
        unit_interval = diagram.unit_interval
        self.play(
            Write(unit_interval, lag_ratio=0.01),
            brace_group.animate.next_to(unit_interval[0], UP, MED_LARGE_BUFF),
            frame.animate.set_width(13).set_y(0),
        )
        unit_interval.numbers.set_backstroke(BLACK, 3)
        self.play(FadeOut(brace_group))
        self.wait()

        # Show a value over this interval
        x_tracker = ValueTracker()
        get_x = x_tracker.get_value
        x_arrow = Vector(DOWN, thickness=5)
        x_arrow.add_updater(lambda m: m.move_to(unit_interval.n2p(get_x()), DOWN))
        x_arrow.set_z_index(-1)
        x_dec = DecimalNumber(0, num_decimal_places=3)
        x_dec.f_always.set_value(get_x)
        x_dec.always.next_to(x_arrow, UP, SMALL_BUFF)
        layer.add_updater(lambda m: m.highlight(m.float_to_index(get_x())))

        self.add(layer)
        self.play(
            VFadeIn(x_arrow, time_span=(0, 1)),
            VFadeIn(x_dec, time_span=(0, 1)),
            x_tracker.animate.set_value(1),
            run_time=6
        )

        # Highlight "t"
        t_bar, p_bar, q_bar = VGroup(
            diagram.get_letter_bar(char)
            for char in "tpq"
        )

        t_brace = Brace(t_bar, DOWN)
        p_brace = Brace(p_bar, DOWN)
        t_brace_label = DecimalNumber(diagram.get_conditional_probability("t"), font_size=24)
        t_brace_label.next_to(t_brace, DOWN, SMALL_BUFF)
        p_brace_label = DecimalNumber(diagram.get_conditional_probability("p"), font_size=24)
        p_brace_label.next_to(p_brace, DOWN, SMALL_BUFF)

        pre_dec_label = dec_labels[ord("t") - ord("a")]
        dec_labels.remove(pre_dec_label)

        self.play(
            VFadeOut(x_dec),
            x_tracker.animate.set_value(unit_interval.p2n(layer.bars[19].get_center())),
            run_time=2
        )
        self.play(
            x_tracker.animate(rate_func=wiggle, run_time=4).set_value(unit_interval.p2n(layer.bars[19].get_left() + 0.2 * LEFT)),
            GrowFromCenter(t_brace),
            TransformFromCopy(pre_dec_label, t_brace_label),
            FadeOut(dec_labels)
        )
        layer.clear_updaters()
        self.play(FadeOut(x_arrow))
        self.wait()

        # Highlight "p"
        self.add(diagram)
        self.play(
            diagram.highlight_letter("p", TEAL, other_bar_opacity=0.25),
            ReplacementTransform(t_brace, p_brace),
            ReplacementTransform(t_brace_label, p_brace_label),
        )
        self.wait()

        # Show "q"
        self.play(
            diagram.zoom_in_on_letter_range(("p", "q"), stretch_factor=18),
            UpdateFromFunc(p_brace, lambda m: m.become(Brace(p_bar, DOWN))),
            UpdateFromFunc(p_brace_label, lambda m: m.next_to(p_brace, DOWN, SMALL_BUFF))
        )
        self.wait()

        q_brace = Brace(q_bar, DOWN)
        q_brace_label = DecimalNumber(
            diagram.get_conditional_probability("q"),
            num_decimal_places=3,
            font_size=24,
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
        self.wait()
        self.play(diagram.renormalize_animation(0, 1), run_time=2)
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
        p_math.to_edge(UP, buff=1)
        self.play(FadeIn(p_math, UP))
        self.wait()

        # Zoom in on the "m"
        self.play(diagram.highlight_letter("m"))
        self.play(FadeOut(prob_label))
        self.play(
            diagram.zoom_in_on_letter("m"),
            p_math.animate(path_arc=PI * 0.3).set_opacity(0.5).to_corner(DL, buff=0.5).fix_in_frame(),
        )
        self.play(diagram.fade_in_new_layer(), self.camera.frame.animate.match_y(diagram))
        self.wait()

        # Show the domination of the vowels
        pct_labels = self.get_stacked_distribution_pct_labels(diagram.layers[1], font_size=25)

        def get_letter_pct_label(char):
            brace = Brace(diagram.get_letter_bar(char), DOWN, buff=SMALL_BUFF)
            label = pct_labels[diagram.char_alphabet.index(char)].next_to(brace, DOWN, buff=SMALL_BUFF)
            return VGroup(brace, label)

        pct_label = get_letter_pct_label("a")
        self.play(
            GrowFromCenter(pct_label[0]),
            Write(pct_label[1]),
        )
        for letter in "eiou":
            self.wait(0.5)
            new_label = get_letter_pct_label(letter)
            self.play(
                ReplacementTransform(pct_label[0], new_label[0]),
                FadeTransformPieces(pct_label[1], new_label[1]),
            )
            pct_label = new_label
        self.wait(0.5)
        self.play(FadeOut(pct_label))
        self.wait(1)

        # Zoom out to show the full width of the diagram
        self.play(diagram.renormalize_animation(0, 1))
        brace = Brace(diagram.layers[0], DOWN)
        self.play(GrowFromEdge(brace, UP), FadeOut(diagram.layers[1]))

        # Show the width of just the "m"
        new_brace = Brace(diagram.get_letter_bar("m", layer_index=0))
        label = new_brace.get_tex("P(``m\")", font_size=35)
        self.play(AnimationGroup(brace.animate.become(new_brace), FadeIn(label), lag_ratio=0.6), run_time=2.5)

        # Bring back the second layer
        self.play(
            AnimationGroup(
                VGroup(brace, label).animate.align_to(diagram.layers[1].get_bottom() + DOWN * 0.1, UP),
                FadeIn(diagram.layers[1]),
                lag_ratio=0.3
            )
        )

        # Renormalize around the "m" again
        renormalize_anim = diagram.zoom_in_on_letter("m", layer_index=0)
        new_brace = Brace(diagram.target.get_letter_bar("m", layer_index=0)).align_to(diagram.layers[1].get_bottom() + DOWN * 0.1, UP)
        new_label = new_brace.get_tex("P(``m\")", font_size=35)
        self.play(
            renormalize_anim,
            brace.animate(run_time=3).become(new_brace),
            label.animate(run_time=3).become(new_label)
        )

        # Show the chain rule
        new_brace = Brace(diagram.get_letter_bar("a"), DOWN, buff=SMALL_BUFF)
        new_label = new_brace.get_tex(f"P(``m\") \\cdot P(``a\" | ``m\")", font_size=36, buff=SMALL_BUFF)
        self.play(
            ReplacementTransform(brace, new_brace),
            TransformMatchingShapes(label, new_label),
            run_time=1.7
        )
        brace, label = new_brace, new_label
        self.wait(1)

        # Show the two letter probabilities
        def get_two_letter_prob_label(char):
            brace = Brace(diagram.get_letter_bar(char), DOWN, buff=SMALL_BUFF)
            label = brace.get_tex(f"P(``m{char}\")", font_size=36, buff=SMALL_BUFF)
            return VGroup(brace, label)
        two_letter_prob_label = get_two_letter_prob_label("a")
        self.play(FadeOut(VGroup(brace, label)), FadeIn(two_letter_prob_label))
        self.wait(1)
        for letter in "eiou":
            self.wait(0.5)
            new_label = get_two_letter_prob_label(letter)
            self.play(
                ReplacementTransform(two_letter_prob_label[0], new_label[0]),
                FadeTransformPieces(two_letter_prob_label[1], new_label[1]),
            )
            two_letter_prob_label = new_label
        self.wait(0.5)
        self.play(FadeOut(two_letter_prob_label))
        self.wait(1)

        # Zoom in on the "a"
        self.play(diagram.highlight_letter("a"))
        self.play(diagram.zoom_in_on_letter("a"))
        self.play(diagram.fade_in_new_layer(), self.camera.frame.animate.match_y(diagram))

        # Show the probabilities for the third layer
        def get_three_letter_prob_label(char):
            brace = Brace(diagram.get_letter_bar(char), DOWN, buff=SMALL_BUFF)
            label = brace.get_tex(f"P(``ma{char}\")", font_size=36, buff=SMALL_BUFF)
            return VGroup(brace, label)
        three_letter_prob_label = get_three_letter_prob_label("d")
        self.play(FadeIn(three_letter_prob_label))
        for letter in "lnt":
            self.wait(0.5)
            new_label = get_three_letter_prob_label(letter)
            self.play(
                ReplacementTransform(three_letter_prob_label[0], new_label[0]),
                FadeTransformPieces(three_letter_prob_label[1], new_label[1]),
            )
            three_letter_prob_label = new_label
        self.wait(1)

        # Zoom in on the "t"
        self.play(FadeOut(three_letter_prob_label), diagram.highlight_letter("t"))
        self.play(diagram.zoom_in_on_letter("t"))
        self.play(diagram.fade_in_new_layer(), self.camera.frame.animate.match_y(diagram))

        # Zoom in on the "h"
        self.play(diagram.highlight_letter("h"))
        self.play(diagram.zoom_in_on_letter("h"))

        # Attach the probability label
        brace = Brace(diagram.get_letter_bar("h"), DOWN, buff=SMALL_BUFF)
        label = brace.get_tex(f"P(``math\")", font_size=40, buff=SMALL_BUFF)
        self.play(GrowFromEdge(brace, UP, run_time=2), TransformMatchingShapes(p_math, label, run_time=2))
        self.wait(1)
        self.play(FadeOut(VGroup(brace, label)))

        # Show the slider going over the range again
        h_bar = diagram.get_letter_bar("h")
        h_min, h_max = diagram.get_sweep_range(h_bar)
        x_tracker = ValueTracker(h_min)
        get_x = x_tracker.get_value
        x_arrow = Vector(DOWN, thickness=5)
        x_arrow.add_updater(lambda m: m.move_to(unit_interval.n2p(get_x()), DOWN))
        x_arrow.set_z_index(-1)
        x_dec = DecimalNumber(x_tracker.get_value(), num_decimal_places=7, font_size=17)
        x_dec.f_always.set_value(get_x)
        x_dec.always.next_to(x_arrow, UP, SMALL_BUFF)
        for layer in diagram.layers:
            layer.add_updater(lambda m: m.highlight(m.x_value_to_index(x_arrow.get_x())))

        self.play(
            VFadeIn(x_arrow, time_span=(0, 1)),
            VFadeIn(x_dec, time_span=(0, 1)),
            x_tracker.animate.set_value(h_max),
            run_time=6
        )
        self.play(x_tracker.animate.set_value(h_min), run_time=6)

        # Track the word as the x value changes
        def get_word_mob(word):
            mob = Text(f"|{word.replace(" ", "_")}|").set_height(0.7)
            mob.to_edge(UP, buff=1.3)
            mob.remove(mob[0], mob[-1])
            return mob

        word_tracker = always_redraw(
            lambda: get_word_mob(diagram.get_word_at_value(get_x()).replace(" ", "_"))
        )
        word_tracker.suspend_updating()
        self.play(self.camera.frame.animate.shift(UP), FadeIn(word_tracker, shift=DOWN), run_time=2)
        word_tracker.resume_updating()

        t_zoom = diagram.zoom_in_on_letter("t", layer_index=2, run_time=2)
        t_min, t_max = diagram.get_sweep_range(diagram.get_letter_bar("t", layer_index=2))
        self.play(t_zoom, x_tracker.animate(run_time=9).set_value(t_min))
        self.play(x_tracker.animate(run_time=5).set_value(t_max))

        # Zoom out to the "ma" range
        for layer in diagram.layers:
            layer.clear_updaters()

        def update_diagram_from_tracker(mob):
            diagram.update_visible_path(get_x())
            diagram.highlight_value(get_x())

        highlighter = Mobject()
        highlighter.add_updater(update_diagram_from_tracker)
        self.add(highlighter)

        math_value = unit_interval.p2n(h_bar.get_center())
        a_zoom = diagram.zoom_in_on_letter(
            "a", layer_index=1,
            add_to_text=False,
            fade_thin_strokes=True,
            run_time=3,
        )
        a_min, a_max = diagram.get_sweep_range(diagram.get_letter_bar("a", layer_index=1))
        self.play(a_zoom, move_tracker_through_zoom(diagram, x_tracker, a_min, run_time=6))

        # Sweep across every four letter word starting with "ma"
        self.play(x_tracker.animate.set_value(a_max), run_time=4)

        # Zoom all the way out to the unit interval
        full_zoom = diagram.renormalize_animation(0, 1, fade_thin_strokes=True, run_time=2)
        self.play(full_zoom, move_tracker_through_zoom(diagram, x_tracker, 0, run_time=8))

        # Sweep across every possible four letter word
        self.play(x_tracker.animate.set_value(1), run_time=8)
        self.play(x_tracker.animate.set_value(math_value), run_time=2)
        self.wait()

        # Clarify the decoding process
        highlighter.suspend_updating()
        word_tracker.suspend_updating()
        self.play(diagram.fade_out_layers(), word_tracker.animate.set_opacity(0))
        self.remove(word_tracker)
        self.wait(1)

        string = "math"
        decoded = VGroup()
        for i, letter in enumerate(string):
            self.play(diagram.restore_layer(i, get_x()))
            self.wait(0.5)
            new_decoded = get_word_mob(string[:i + 1])
            self.play(
                *(ReplacementTransform(old, new) for old, new in zip(decoded, new_decoded)),
                FadeIn(new_decoded[i], shift=0.25 * DOWN),
            )
            decoded = new_decoded
            self.play(diagram.zoom_in_on_letter(letter, layer_index=i))

        # Zoom back out to the full unit interval
        self.play(diagram.renormalize_animation(0, 1, fade_thin_strokes=True, run_time=3))
        self.wait()

        # Hand the readout back to the live word tracker
        self.remove(*decoded)
        word_tracker.set_opacity(1)
        self.add(word_tracker)

        # Choose random values
        highlighter.resume_updating()
        word_tracker.resume_updating()
        random.seed(1)
        for _ in range(50):
            x_tracker.set_value(random.random())
            self.wait(1)

        # Set the tracker back to "math"
        self.play(x_tracker.animate.set_value(math_value), run_time=2)

        # Show the meaning of the first decimal digit
        x_dec.clear_updaters()
        self.play(x_dec.animate.scale(2).align_to(x_dec, DOWN))
        self.play(x_dec[2].animate.set_color(YELLOW))
        intervals = VGroup(*[
            Line(unit_interval.n2p(0.1 * i), unit_interval.n2p(0.1 * (i + 1))).set_color(YELLOW)
            for i in range(10)
        ])
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4]:
            self.remove(intervals)
            self.add(intervals[i])
            self.wait(0.1)
        self.wait(1)
        self.play(FadeOut(intervals[i]), x_dec[2].animate.set_color(WHITE))

        # Helpers for converting to binary
        highlighter.suspend_updating()
        word_tracker.suspend_updating()
        n2p = unit_interval.n2p
        line_y = n2p(0)[1]
        highlight_range = [0, 0.5]

        def value_to_x(value):
            ref_low, ref_high = highlight_range
            x_low = n2p(ref_low)[0]
            x_high = n2p(ref_high)[0]
            return x_low + (value - ref_low) * (x_high - x_low) / (ref_high - ref_low)

        def put_on_value_range(line, low_value, high_value, margin=1):
            frame_left = frame.get_left()[0] - margin
            frame_right = frame.get_right()[0] + margin
            left = clip(value_to_x(low_value), frame_left, frame_right)
            right = clip(value_to_x(high_value), frame_left, frame_right)
            right = max(right, left + 1e-3)
            line.set_points_by_ends(np.array([left, line_y, 0]), np.array([right, line_y, 0]))

        binary_line = Line(n2p(0), n2p(1)).set_stroke(GREY_B, 2)
        binary_line.add_updater(lambda m: put_on_value_range(m, 0, 1))

        label_style = ["fraction"]

        def get_dyadic_tex(numer, denom):
            if numer == 0:
                return "0"
            if numer == denom:
                return "1"
            if label_style[0] == "fraction":
                return Rf"\frac{{{numer}}}{{{denom}}}"
            power = int(math.log2(denom))
            while numer % 2 == 0:
                numer //= 2
                power -= 1
            if numer == 1:
                return Rf"2^{{-{power}}}"
            return Rf"{numer} \cdot 2^{{-{power}}}"

        def get_dyadic_mark(numer, denom):
            level = max(int(math.log2(denom)) - 1, 0)
            tick_height = max(0.3 * 0.85**level, 0.12)
            font_size = 30
            point = n2p(numer / denom)
            tick = Line(point + 0.5 * tick_height * DOWN, point + 0.5 * tick_height * UP)
            tick.set_stroke(GREY_B, 2)
            mark = VGroup(tick)
            if label_style[0] == "fraction":
                label = Tex(get_dyadic_tex(numer, denom), font_size=font_size)
                label.next_to(tick, UP, SMALL_BUFF)
                label.set_backstroke(BLACK, 3)
                mark.add(label)
            mark.numer = numer
            mark.denom = denom
            mark.value = numer / denom

            def update_mark(mark):
                mark[0].move_to(n2p(mark.value))
                if len(mark) > 1:
                    mark[1].next_to(mark[0], UP, SMALL_BUFF)

            mark.add_updater(update_mark)
            mark.set_z_index(1)
            return mark

        def get_width_braces(level, dynamic=True, fade=1, margin=1, max_braces=16, max_opacity=0.75):
            group = VGroup()
            group.fade = ValueTracker(fade)
            group.release = ValueTracker(0)
            cell_width = 2.0**-level
            label_template = Tex(Rf"2^{{-{level}}}", font_size=26)

            def update_braces(group):
                ref_low, ref_high = highlight_range
                x_ref_low = value_to_x(ref_low)
                slope = (value_to_x(ref_high) - x_ref_low) / (ref_high - ref_low)
                frame_left = frame.get_left()[0] - margin
                frame_right = frame.get_right()[0] + margin
                opacity = max_opacity * group.fade.get_value()
                if dynamic:
                    rel_width = slope * cell_width / frame.get_width()
                    fade_in = clip((rel_width - 0.2) / 0.1, 0, 1)
                    fade_out = clip((0.65 - rel_width) / 0.2, 0, 1)
                    opacity *= fade_in * interpolate(1, fade_out, group.release.get_value())
                n_used = 0
                if opacity > 0:
                    v_left = ref_low + (frame_left - x_ref_low) / slope
                    v_right = ref_low + (frame_right - x_ref_low) / slope
                    i_low = max(math.floor(v_left / cell_width), 0)
                    i_high = min(math.floor(v_right / cell_width), 2**level - 1, i_low + max_braces - 1)
                    indices = list(range(i_low, i_high + 1))
                    while len(group) < len(indices):
                        group.add(VGroup(VMobject(), label_template.copy()))
                    for i, width_brace in zip(indices, group):
                        x_low = x_ref_low + (i * cell_width - ref_low) * slope
                        x_high = x_low + cell_width * slope
                        if x_high - x_low > 100 * frame.get_width():
                            width_brace.set_opacity(0)
                            continue
                        line = Line(np.array([x_low, line_y, 0]), np.array([x_high, line_y, 0]))
                        width_brace[0].become(Brace(line, UP, buff=0).stretch(0.7, 1, about_edge=DOWN))
                        width_brace[1].next_to(width_brace[0], UP, SMALL_BUFF)
                        width_brace[1].set_backstroke(BLACK, 3)
                        width_brace.set_opacity(opacity)
                    n_used = len(indices)
                for width_brace in group[n_used:]:
                    width_brace.set_opacity(0)

            group.add_updater(update_braces)
            group.update()
            return group

        def get_visible_value_range(margin=0.5):
            x0 = n2p(0)[0]
            x1 = n2p(1)[0]
            half_width = frame.get_width() / 2 + margin
            return [(frame.get_x() + sign * half_width - x0) / (x1 - x0) for sign in (-1, 1)]

        def get_dyadic_marks(level):
            denom = 2**level
            low, high = get_visible_value_range()
            numers = range(max(math.ceil(low * denom), 0), min(math.floor(high * denom), denom) + 1)
            return VGroup(
                get_dyadic_mark(numer, denom)
                for numer in numers
                if level == 1 or numer % 2 == 1
            )

        def get_dyadic_ruler(fade=0, ref_spacing=0.35, max_height=0.45, height_ratio=0.7, max_depth=5, n_ticks=500, margin=0.5):
            ticks = VGroup(Line(ORIGIN, UP) for _ in range(n_ticks))
            ticks.set_stroke(GREY_B, 2)
            ticks.fade = ValueTracker(fade)
            ticks.set_z_index(1)

            def update_ruler(ticks):
                ref_low, ref_high = highlight_range
                x_ref_low = value_to_x(ref_low)
                slope = (value_to_x(ref_high) - x_ref_low) / (ref_high - ref_low)
                frame_width = frame.get_width()
                v_left = ref_low + (frame.get_left()[0] - margin - x_ref_low) / slope
                v_right = ref_low + (frame.get_right()[0] + margin - x_ref_low) / slope
                fade = ticks.fade.get_value()
                n_used = 0
                level = 0
                while n_used < len(ticks):
                    depth = math.log2(ref_spacing * frame_width / (slope * 2.0**-level))
                    if depth >= max_depth:
                        break
                    opacity = fade * clip(max_depth - depth, 0, 1)
                    height = max_height * height_ratio**clip(depth, 0, max_depth)
                    stroke_width = interpolate(1, 2.5, height / max_height)
                    denom = 2**level
                    i_low = max(math.ceil(v_left * denom), 0)
                    i_high = min(math.floor(v_right * denom), denom)
                    for i in range(i_low, i_high + 1):
                        if (level > 0 and i % 2 == 0) or n_used >= len(ticks):
                            continue
                        x = x_ref_low + (i / denom - ref_low) * slope
                        tick = ticks[n_used]
                        tick.put_start_and_end_on(
                            np.array([x, line_y - height / 2, 0]),
                            np.array([x, line_y + height / 2, 0]),
                        )
                        tick.set_stroke(width=stroke_width, opacity=opacity)
                        n_used += 1
                    level += 1
                for tick in ticks[n_used:]:
                    tick.set_stroke(opacity=0)

            ticks.add_updater(update_ruler)
            ticks.update()
            return ticks

        def zoom_to_interval(low, high, screen_fraction=0.7, run_time=2, **kwargs):
            mid = (low + high) / 2
            span = (high - low) * diagram.full_width / (screen_fraction * frame.get_width())
            return diagram.renormalize_animation(mid - span / 2, mid + span / 2, run_time=run_time, **kwargs)

        def get_half(bit, low=0, high=1):
            mid = (low + high) / 2
            return (low, mid) if bit == "0" else (mid, high)

        highlight = Line(n2p(0), n2p(0.5)).set_stroke(YELLOW, 5)
        highlight.add_updater(lambda m: put_on_value_range(m, *highlight_range))

        bin_scale = x_dec.get_height() / Text(".0", font="Consolas").get_height()
        bin_center_x = x_dec.get_x()
        bin_bottom_y = x_dec.get_bottom()[1] + UP

        def get_binary_number(bits, highlight_last=True):
            mob = Text("." + bits, font="Consolas").scale(bin_scale)
            mob.set_x(bin_center_x)
            mob.shift((bin_bottom_y - mob.get_bottom()[1]) * UP)
            if highlight_last:
                mob[-1].set_color(YELLOW)
            return mob

        level_marks = [get_dyadic_marks(1)]
        brace_levels = []

        # Convert everything to binary
        self.play(
            FadeOut(x_dec),
            FadeOut(x_arrow),
            FadeOut(word_tracker),
            diagram.fade_out_layers(),
            diagram.hide_intervals(),
            FadeIn(binary_line),
            FadeIn(level_marks[0]),
            run_time=2,
        )

        word = "math"
        word_path = diagram.show_text_path(word[:-1])
        for layer in word_path[1:]:
            layer.is_faded = True
            layer.set_opacity(0)
        last_bar = word_path[-1].bars[diagram.char_alphabet.index(word[-1])]
        word_low = unit_interval.p2n(last_bar.get_left())
        word_high = unit_interval.p2n(last_bar.get_right())
        word_bits = diagram.get_binary_code(word_low, word_high)
        n_bits = len(word_bits)
        self.wait()

        # Choose the first bit
        binary_number = get_binary_number("0")
        self.add(binary_number, highlight)
        for bit in ("1", word_bits[0]) if word_bits[0] == "0" else ("1",):
            self.wait(0.6)
            self.remove(binary_number)
            binary_number = get_binary_number(bit)
            self.add(binary_number)
            highlight_range[:] = get_half(bit)

        bits = word_bits[0]
        low, high = get_half(bits)

        # Choose the remaining bits
        for k in range(2, n_bits + 2):
            step_time = 0.75 if k <= 3 else max(0.75 * 0.5**(k - 3), 0.1)
            if label_style[0] == "power" and k <= n_bits:
                for group in brace_levels:
                    group.release.set_value(1)
                new_braces = get_width_braces(k)
                brace_levels.append(new_braces)
                self.add(new_braces)
            self.play(zoom_to_interval(low, high, run_time=2 * step_time))
            if k > n_bits:
                break

            if label_style[0] == "fraction":
                marks = get_dyadic_marks(k)
                level_marks.append(marks)
                self.play(FadeIn(marks), run_time=step_time)
            else:
                self.wait(step_time)

            bit = word_bits[k - 1]
            bits += bit
            low, high = get_half(bit, low, high)
            self.remove(binary_number)
            binary_number = get_binary_number(bits)
            self.add(binary_number)
            highlight_range[:] = [low, high]
            self.wait(0.2)

            if k == 3:
                label_style[0] = "power"
                old_marks = VGroup(*level_marks)
                ruler = get_dyadic_ruler(fade=0)
                new_braces = get_width_braces(k, fade=0)
                brace_levels.append(new_braces)
                self.add(ruler, new_braces)
                self.play(
                    UpdateFromAlphaFunc(Mobject(), lambda m, a: old_marks.set_opacity(1 - a)),
                    ruler.fade.animate.set_value(1),
                    new_braces.fade.animate.set_value(1),
                )
                self.remove(*level_marks)

        self.remove(binary_number)
        binary_number = get_binary_number(bits, highlight_last=False)
        self.add(binary_number)
        self.wait()

        binary_number_line = VGroup(binary_line, *level_marks)

        # Show the range of real numbers
        def get_aligned_binary_text(string):
            text = Text(string, font="Consolas").scale(bin_scale)
            text.align_to(binary_number, LEFT).align_to(binary_number, DOWN)
            return text

        long_text = get_aligned_binary_text("." + bits + "0" * 40)
        n_trail = sum(glyph.get_left()[0] < frame.get_right()[0] for glyph in long_text[n_bits + 1:]) + 1
        zeros = get_aligned_binary_text("." + bits + "0" * n_trail)[n_bits + 1:]
        ones = get_aligned_binary_text("." + bits + "1" * n_trail)[n_bits + 1:]
        continuation = VGroup(VGroup(zero, one) for zero, one in zip(zeros, ones))
        continuation.set_color(TEAL)

        arrow_tracker = ValueTracker(low)
        trail_opacity_tracker = ValueTracker(0)
        n_shown_tracker = ValueTracker(n_trail)

        def get_n_shown():
            return math.ceil(n_shown_tracker.get_value())

        def get_shown_value():
            scale = 2**(n_bits + get_n_shown())
            return math.floor(arrow_tracker.get_value() * scale) / scale

        def update_continuation(group):
            extra = diagram.get_binary_digits(get_shown_value(), n_bits + n_trail)[n_bits:]
            n_shown = get_n_shown()
            for i, (slot, bit) in enumerate(zip(group, extra)):
                opacity = trail_opacity_tracker.get_value() * (1 - i * 0.04) if i < n_shown else 0
                slot[0].set_opacity(opacity if bit == "0" else 0)
                slot[1].set_opacity(opacity if bit == "1" else 0)

        arrow_tip_y = unit_interval.get_y()

        range_arrow = Vector(DOWN, thickness=5)
        range_arrow.set_z_index(-1)
        range_arrow.add_updater(lambda m: m.move_to(
            np.array([n2p(get_shown_value())[0], arrow_tip_y, 0]), DOWN
        ))
        arrow_opacity_tracker = ValueTracker(0)
        range_arrow.add_updater(lambda m: m.set_opacity(arrow_opacity_tracker.get_value()))
        continuation.add_updater(update_continuation)
        self.add(range_arrow, continuation)
        self.play(arrow_tracker.animate(run_time=2).set_value(high), arrow_opacity_tracker.animate.set_value(1))
        self.play(arrow_tracker.animate(run_time=2).set_value(low))
        interval_width = n2p(high)[0] - n2p(low)[0]
        n_significant = sum(interval_width / 2**(i + 1) >= 0.02 for i in range(n_trail))
        self.play(
            arrow_tracker.animate(run_time=6).set_value(low + 0.93 * (high - low)),
            trail_opacity_tracker.animate.set_value(0.8),
            n_shown_tracker.animate(time_span=(4.5, 7), rate_func=rush_from).set_value(0),
        )

        # Bring back the diagram
        self.remove(continuation)

        word_zoom = diagram.renormalize_animation(word_low, word_high, center_text=word, run_time=3)
        diagram.restore_layers_in_target(word_path, (word_low + word_high) / 2, center=ORIGIN)
        binary_number.generate_target()
        bits_brace = Brace(binary_number.target[1:], DOWN, buff=SMALL_BUFF)
        bits_label = bits_brace.get_text(f"{n_bits} bits", font_size=30)
        bits_shift = range_arrow.get_top()[1] + 0.35 - bits_label.get_bottom()[1]
        VGroup(binary_number.target, bits_brace, bits_label).shift(bits_shift * UP)
        word_mob = get_word_mob(word)
        word_mob.next_to(binary_number.target, UP, buff=MED_SMALL_BUFF)
        word_mob.match_x(binary_number.target)
        label_x_min = frame.get_left()[0] - 0.5
        label_x_max = frame.get_right()[0] + 0.5
        for layer in word_path:
            for label in layer.labels:
                label.set_x(clip(label.get_x(), label_x_min, label_x_max))
        self.play(
            word_zoom,
            MoveToTarget(binary_number, run_time=2, time_span=(1, 3)),
            FadeIn(word_mob, shift=0.5 * DOWN, run_time=2, time_span=(1, 3)),
        )
        self.wait()

        # Show the number of bits used for the word
        self.play(
            GrowFromCenter(bits_brace),
            FadeIn(bits_label, shift=0.25 * DOWN),
        )
        self.wait()

        # Show that one fewer bit is not enough
        short_numer = int(word_bits[:-1], 2)
        red_range = [short_numer / 2**(n_bits - 1), (short_numer + 1) / 2**(n_bits - 1)]
        old_range = list(highlight_range)

        def update_highlight_range(mob, alpha):
            highlight_range[:] = [interpolate(a, b, alpha) for a, b in zip(old_range, red_range)]

        short_brace = Brace(binary_number[1:-1], DOWN, buff=SMALL_BUFF)
        short_label = short_brace.get_text(f"{n_bits - 1} bits", font_size=30)
        self.play(
            zoom_to_interval(*red_range, run_time=2, center_text=word),
            UpdateFromAlphaFunc(highlight, update_highlight_range, run_time=2),
            binary_number[-1].animate.set_opacity(0.2),
            ReplacementTransform(bits_brace, short_brace, run_time=2),
            FadeTransformPieces(bits_label, short_label, run_time=2),
        )
        bits_brace, bits_label = short_brace, short_label

        target_lines = VGroup(
            DashedLine(
                np.array([x, last_bar.get_bottom()[1], 0]),
                np.array([x, range_arrow.get_top()[1], 0]),
            )
            for x in (last_bar.get_left()[0], last_bar.get_right()[0])
        )
        target_lines.set_stroke(WHITE, 3)
        red_brace = get_width_braces(n_bits - 1, dynamic=False, fade=0)
        self.add(red_brace)
        self.play(
            UpdateFromAlphaFunc(highlight, lambda m, a: m.set_stroke(interpolate_color(YELLOW, RED, a))),
            ShowCreation(target_lines),
            brace_levels[-1].fade.animate.set_value(0),
            red_brace.fade.animate.set_value(1),
        )
        self.wait()

        # Try the endpoints of the red interval
        arrow_value = ValueTracker(get_shown_value())
        range_arrow.clear_updaters()
        range_arrow.add_updater(lambda m: m.move_to(
            np.array([n2p(arrow_value.get_value())[0], arrow_tip_y, 0]), DOWN
        ))
        number_rect = SurroundingRectangle(binary_number[1:-1], buff=0.05)
        number_rect.set_stroke(YELLOW, 3)
        self.play(arrow_value.animate.set_value(red_range[0]), FadeIn(number_rect))
        self.wait()

        endpoint_numbers = [
            get_aligned_binary_text("." + format(numer, f"0{n_bits - 1}b") + word_bits[-1])
            for numer in (short_numer, short_numer + 1)
        ]
        for number in endpoint_numbers:
            number[-1].set_opacity(0.2)

        for i, side in enumerate([1, 0, 1, 0, 1]):
            self.remove(binary_number)
            binary_number = endpoint_numbers[side]
            self.add(binary_number)
            arrow_value.set_value(red_range[side])
            self.wait(1 if i == 0 else 0.25)
        self.wait()
        self.play(FadeOut(number_rect))
        self.wait()

        # Bring back the last bit
        full_number = get_aligned_binary_text("." + word_bits)
        full_number[-1].set_opacity(0.2)
        self.remove(binary_number)
        binary_number = full_number
        self.add(binary_number)

        def update_yellow_highlight(mob, alpha):
            highlight_range[:] = [interpolate(a, b, alpha) for a, b in zip(red_range, old_range)]
            mob.set_stroke(interpolate_color(RED, YELLOW, alpha))

        full_brace = Brace(binary_number[1:], DOWN, buff=SMALL_BUFF)
        full_label = full_brace.get_text(f"{n_bits} bits", font_size=30)
        self.play(
            brace_levels[-1].fade.animate.set_value(1),
            red_brace.fade.animate.set_value(0),
            UpdateFromAlphaFunc(highlight, update_yellow_highlight),
            binary_number[-1].animate.set_opacity(1),
            ReplacementTransform(bits_brace, full_brace),
            FadeTransformPieces(bits_label, full_label),
            arrow_value.animate.set_value(int(word_bits, 2) / 2**n_bits),
            run_time=1.5,
        )
        bits_brace, bits_label = full_brace, full_label

        # Pin the trackers to the screen
        full_word = "mathematics"
        fixed_center = frame.get_center().copy()
        fixed_scale = FRAME_HEIGHT / frame.get_height()

        def fix_to_screen(mob):
            mob.shift(-fixed_center)
            mob.scale(fixed_scale, about_point=ORIGIN)
            mob.fix_in_frame()
            return mob

        bin_ref = binary_number.copy()
        fix_to_screen(binary_number)
        fix_to_screen(word_mob)
        fix_to_screen(VGroup(bits_brace, bits_label))

        # Show the avg bits per character
        avg_frac = Tex(R"\displaystyle\frac{14 \text{ bits}}{|\text{``math''}|}", font_size=32)
        avg_value = Tex(R"= 3.5 \text{ bits/char}", font_size=32)
        avg_value.next_to(avg_frac, DOWN, buff=0.25, aligned_edge=LEFT)
        avg_bits_per_char = VGroup(avg_frac, avg_value)
        fix_to_screen(avg_bits_per_char)
        avg_bits_per_char.align_to(word_mob, UP).to_edge(RIGHT, buff=1)
        copy = bits_label.copy()
        self.play(
            AnimationGroup(
                copy.animate(path_arc=PI * 0.3).match_width(avg_frac[:6]).move_to(avg_frac[:6]),
                GrowFromCenter(avg_frac[6]),
                AnimationGroup(
                    TransformFromCopy(word_mob, avg_frac[10:14], path_arc=PI * 0.3),
                    FadeIn(VGroup(avg_frac[7:10], avg_frac[14:17])),
                    lag_ratio=0.6
                ),
                Write(avg_value),
                lag_ratio=0.5
            )
        )
        self.add(avg_bits_per_char)
        self.remove(copy)
        self.wait(1)
        self.play(FadeOut(VGroup(bits_brace, bits_label, avg_bits_per_char)))

        # Helpers for extending the word
        def get_world_binary(bit_str):
            text = Text("." + bit_str, font="Consolas").scale(bin_scale)
            text.match_x(bin_ref).align_to(bin_ref, DOWN)
            return text

        def get_screen_binary(bit_str, n_new=0):
            text = get_world_binary(bit_str)
            if n_new > 0:
                text[-n_new:].set_color(YELLOW)
            return fix_to_screen(text)

        def get_screen_word(text):
            mob = get_word_mob(text)
            mob.next_to(bin_ref, UP, buff=MED_SMALL_BUFF)
            mob.match_x(bin_ref)
            return fix_to_screen(mob)

        x_ref = {"bar": last_bar, "range": (word_low, word_high)}

        def value_to_x(value):
            bar = x_ref["bar"]
            ref_low, ref_high = x_ref["range"]
            x_low = bar.get_left()[0]
            x_high = bar.get_right()[0]
            return x_low + (value - ref_low) * (x_high - x_low) / (ref_high - ref_low)

        def n2p(value):
            return np.array([value_to_x(value), line_y, 0])

        arrow_top_y = range_arrow.get_top()[1]
        line_bars = {"start": last_bar, "end": last_bar}
        line_alpha = ValueTracker(1)

        def get_line_params(bar):
            return np.array([bar.get_left()[0], bar.get_right()[0], bar.get_bottom()[1]])

        def update_target_lines(lines):
            x_left, x_right, y_bottom = interpolate(
                get_line_params(line_bars["start"]),
                get_line_params(line_bars["end"]),
                line_alpha.get_value(),
            )
            new_lines = VGroup(
                DashedLine(np.array([x, y_bottom, 0]), np.array([x, arrow_top_y, 0]))
                for x in (x_left, x_right)
            )
            new_lines.set_stroke(WHITE, 3)
            lines.become(new_lines)

        target_lines.set_z_index(10)
        target_lines.add_updater(update_target_lines)
        self.add(line_alpha)

        def zoom_to_world_width(low, high, world_width, **kwargs):
            mid = (low + high) / 2
            span = (high - low) * diagram.full_width / world_width
            return diagram.renormalize_animation(
                mid - span / 2, mid + span / 2, point_func=n2p, run_time=1.5, **kwargs
            )

        def get_cell(bit_str):
            numer = int(bit_str, 2)
            denom = 2**len(bit_str)
            return [numer / denom, (numer + 1) / denom]

        def get_cell_update(start_range, end_range):
            def update(mob, alpha):
                highlight_range[:] = [interpolate(a, b, alpha) for a, b in zip(start_range, end_range)]
                mob.set_stroke(interpolate_color(RED, YELLOW, alpha))
            return update

        # Extend to the full word
        prefix = word
        word_range = (word_low, word_high)
        target_bar = last_bar
        shown_bits = word_bits
        view = (tuple(red_range), 0.7 * frame.get_width())
        for char in full_word[len(word):]:
            new_layer = diagram.make_layer(prefix, target_bar)
            diagram.align_layer_to_bar(new_layer, target_bar)
            diagram.layers.add(new_layer)
            index = diagram.char_alphabet.index(char)
            target_bar = new_layer.bars[index]
            x0 = new_layer.bars.get_left()[0]
            x1 = new_layer.bars.get_right()[0]
            range_low, range_high = word_range
            word_range = tuple(
                range_low + (range_high - range_low) * (x - x0) / (x1 - x0)
                for x in (target_bar.get_left()[0], target_bar.get_right()[0])
            )
            prefix += char
            code = diagram.get_binary_code(*word_range)
            needs_bits = len(code) > len(shown_bits)

            camera_anims = []
            frame_width = frame.get_width()
            needed_bottom = new_layer.get_bottom()[1] - 0.75
            frame_bottom = frame.get_y() - frame.get_height() / 2
            if needed_bottom < frame_bottom:
                zoom = (arrow_top_y - needed_bottom) / (arrow_top_y - frame_bottom)
                frame_width *= zoom
                camera_anims.append(frame.animate.set_height(zoom * frame.get_height()).set_y(
                    arrow_top_y - zoom * (arrow_top_y - frame.get_y())
                ))

            new_layer.set_opacity(0)
            new_layer.highlight_state = index
            x_ref["bar"] = target_bar
            x_ref["range"] = word_range
            if not needs_bits:
                view = (word_range, 0.6 * frame_width)
            layer_zoom = zoom_to_world_width(*view[0], view[1], center_text=prefix)
            target_of = dict(zip(map(id, diagram.get_family()), diagram.target.get_family()))
            target_layer = target_of[id(new_layer)]
            target_layer.set_opacity(1)
            diagram.reset_layer_style(target_layer)
            target_layer.reposition_labels()
            target_layer.highlight(index, other_bar_opacity=0.35)
            target_bar_copy = target_layer.bars[index]
            if target_bar_copy.get_left()[0] <= 0 <= target_bar_copy.get_right()[0]:
                target_layer[1][index].match_x(ORIGIN)
            diagram.update_bar_strokes(target_layer)

            line_bars["start"] = line_bars["end"]
            line_bars["end"] = target_bar
            line_alpha.set_value(0)
            new_word = get_screen_word(prefix)
            self.play(
                layer_zoom,
                *camera_anims,
                line_alpha.animate.set_value(1),
                *(ReplacementTransform(a, b) for a, b in zip(word_mob, new_word)),
                FadeIn(new_word[-1], shift=0.25 * DOWN),
                run_time=1.5,
            )
            line_bars["start"] = target_bar
            self.remove(*new_word)
            self.add(new_word)
            word_mob = new_word

            for n_shown in range(len(shown_bits) + 1, len(code) + 1):
                step_bits = code[:n_shown]
                new_braces = get_width_braces(n_shown, fade=0)
                self.add(new_braces)
                red_anims = [
                    UpdateFromAlphaFunc(highlight, lambda m, a: m.set_stroke(interpolate_color(YELLOW, RED, a))),
                    *(group.release.animate.set_value(1) for group in brace_levels),
                    new_braces.fade.animate.set_value(1),
                ]
                brace_levels.append(new_braces)
                self.play(*red_anims, run_time=0.5)

                n_same = next(
                    (i for i, (a, b) in enumerate(zip(shown_bits, step_bits)) if a != b),
                    len(shown_bits),
                )
                old_number = binary_number
                new_number = get_screen_binary(step_bits, n_new=len(step_bits) - n_same)
                new_cell = get_cell(step_bits)
                self.play(
                    UpdateFromAlphaFunc(highlight, get_cell_update(list(highlight_range), new_cell)),
                    arrow_value.animate.set_value(new_cell[0]),
                    *(ReplacementTransform(a, b) for a, b in zip(old_number, new_number)),
                    FadeIn(new_number[len(old_number):]),
                    run_time=0.75,
                )
                self.remove(old_number, *new_number)
                binary_number = new_number
                self.add(binary_number)
                shown_bits = step_bits

            if needs_bits:
                view = (word_range, 0.6 * frame.get_width())
                self.play(zoom_to_world_width(*view[0], view[1], center_text=prefix))
            self.wait(0.5)

        new_number = get_screen_binary(shown_bits)
        self.remove(binary_number)
        binary_number = new_number
        self.add(binary_number)
        self.wait()

        # Show the number of bits for the full word
        total_bits = len(shown_bits)
        world_number = get_world_binary(shown_bits)
        bits_brace = Brace(world_number[1:], DOWN, buff=SMALL_BUFF)
        bits_label = bits_brace.get_text(f"{total_bits} bits", font_size=30)
        fix_to_screen(VGroup(bits_brace, bits_label))
        self.play(
            GrowFromCenter(bits_brace),
            FadeIn(bits_label, shift=0.25 * DOWN),
        )
        self.wait()

        # Show the avg bits per character for the full word
        avg_number = total_bits / len(full_word)
        avg_str = f"{avg_number:.2f}"
        relation = "=" if abs(avg_number - float(avg_str)) < 1e-9 else R"\approx"
        full_frac = Tex(
            Rf"\displaystyle\frac{{{total_bits} \text{{ bits}}}}{{|\text{{``{full_word}''}}|}}",
            font_size=32,
        )
        full_value = Tex(Rf"{relation} {avg_str} \text{{ bits/char}}", font_size=32)
        full_value.next_to(full_frac, DOWN, buff=0.25, aligned_edge=LEFT)
        full_avg = VGroup(full_frac, full_value)
        fix_to_screen(full_avg)
        full_avg.align_to(word_mob, UP).to_edge(RIGHT, buff=1)
        num_end = len(str(total_bits)) + 4
        word_start = num_end + 4
        word_end = word_start + len(full_word)
        label_copy = bits_label.copy()
        self.play(
            AnimationGroup(
                label_copy.animate(path_arc=PI * 0.3).match_width(full_frac[:num_end]).move_to(full_frac[:num_end]),
                GrowFromCenter(full_frac[num_end]),
                AnimationGroup(
                    TransformFromCopy(word_mob, full_frac[word_start:word_end], path_arc=PI * 0.3),
                    FadeIn(VGroup(full_frac[num_end + 1:word_start], full_frac[word_end:])),
                    lag_ratio=0.6
                ),
                Write(full_value),
                lag_ratio=0.5
            )
        )
        self.add(full_avg)
        self.remove(label_copy)
        self.wait()


class GeneralizeBitCount(InteractiveScene):
    def construct(self):
        # Set up the number line
        frame = self.frame
        line_width = 12
        line_y = 0
        p = 0.7 * 2**-5
        hit_level = 4
        hit_value = 5 / 16
        slid_low = 10 / 32 + (2**-5 - p) / 2

        view_center = ValueTracker(0.5)
        view_log_width = ValueTracker(0)
        k_tracker = ValueTracker(0)
        target_low = ValueTracker(hit_value - 0.35 * p)

        def get_scale():
            return line_width / 2**view_log_width.get_value()

        def value_to_x(value):
            return (value - view_center.get_value()) * get_scale()

        def n2p(value, y=line_y):
            return np.array([value_to_x(value), y, 0])

        def get_target_bounds():
            low = target_low.get_value()
            return low, low + p

        number_line = Line(LEFT, RIGHT).set_stroke(GREY_B, 2)

        def update_number_line(line):
            left = max(value_to_x(0), frame.get_left()[0] - 1)
            right = min(value_to_x(1), frame.get_right()[0] + 1)
            line.put_start_and_end_on(np.array([left, line_y, 0]), np.array([right, line_y, 0]))

        number_line.add_updater(update_number_line)
        number_line.update()

        def get_end_label(value):
            label = Tex(str(value))
            label.add_updater(lambda m: m.move_to(n2p(value) + 0.5 * DOWN))
            label.update()
            return label

        end_labels = VGroup(get_end_label(0), get_end_label(1))

        def get_ticks(ref_spacing=0.35, max_height=0.45, height_ratio=0.7, n_ticks=400, margin=0.5):
            ticks = VGroup(Line(ORIGIN, UP) for _ in range(n_ticks))
            ticks.set_stroke(GREY_B, 2)
            ticks.set_z_index(1)
            ticks.last_key = None
            ticks.n_used = len(ticks)

            def update_ticks(ticks):
                key = (view_center.get_value(), view_log_width.get_value(), k_tracker.get_value(), frame.get_width())
                if key == ticks.last_key:
                    return
                ticks.last_key = key
                scale = get_scale()
                frame_width = frame.get_width()
                v_left = view_center.get_value() + (frame.get_left()[0] - margin) / scale
                v_right = view_center.get_value() + (frame.get_right()[0] + margin) / scale
                k_value = k_tracker.get_value()
                n_used = 0
                for level in range(math.floor(k_value) + 2):
                    opacity = clip(k_value - level + 1, 0, 1)
                    if opacity <= 0:
                        break
                    depth = math.log2(ref_spacing * frame_width / (scale * 2.0**-level))
                    height = max_height * height_ratio**clip(depth, 0, 6)
                    stroke_width = interpolate(1, 2.5, height / max_height)
                    denom = 2**level
                    for i in range(max(math.ceil(v_left * denom), 0), min(math.floor(v_right * denom), denom) + 1):
                        if (level > 0 and i % 2 == 0) or n_used >= len(ticks):
                            continue
                        x = value_to_x(i / denom)
                        tick = ticks[n_used]
                        tick.put_start_and_end_on(
                            np.array([x, line_y - height / 2, 0]),
                            np.array([x, line_y + height / 2, 0]),
                        )
                        tick.set_stroke(width=stroke_width, opacity=opacity)
                        n_used += 1
                for tick in ticks[n_used:ticks.n_used]:
                    tick.set_stroke(opacity=0)
                ticks.n_used = n_used

            ticks.add_updater(update_ticks)
            ticks.update()
            return ticks

        ticks = get_ticks()

        self.play(
            ShowCreation(number_line),
            FadeIn(ticks),
            FadeIn(end_labels),
        )
        self.wait()

        # Show the target region
        goalpost_bottom = line_y - 0.75
        goalposts = VGroup(
            DashedLine(np.array([0, goalpost_bottom, 0]), np.array([0, line_y + 1.5, 0]))
            for _ in range(2)
        )
        goalposts.set_stroke(WHITE, 3)

        def update_goalposts(group):
            for post, value in zip(group, get_target_bounds()):
                post.set_x(value_to_x(value))

        goalposts.add_updater(update_goalposts)
        goalposts.update()

        def update_brace(group, low, high, direction, buff, y=line_y, grow=1):
            if not hasattr(group, "label_template"):
                group.label_template = group[1].copy()
            ends = (value_to_x(low), value_to_x(high))
            last_ends = getattr(group, "last_ends", None)
            last_grow = getattr(group, "last_grow", 1)
            if last_ends == ends and last_grow == grow:
                return
            same_width = last_ends is not None and np.isclose(ends[1] - ends[0], last_ends[1] - last_ends[0])
            if same_width and grow == 1 and last_grow == 1:
                group.shift((ends[0] - last_ends[0]) * RIGHT)
            else:
                group[0].become(Brace(Line(n2p(low, y), n2p(high, y)), direction, buff=buff))
                group[1].become(group.label_template)
                group[1].next_to(group[0], direction, SMALL_BUFF)
                if grow < 1:
                    group.scale(max(grow, 1e-3), about_edge=-direction)
            group.last_ends = ends
            group.last_grow = grow

        p_brace = VGroup(VMobject(), Tex("p", font_size=36))

        def update_p_brace(group):
            update_brace(group, *get_target_bounds(), DOWN, 0.1, y=goalpost_bottom)

        p_brace.add_updater(update_p_brace)
        p_brace.update()

        self.add(goalposts)
        self.play(AnimationGroup(*[ShowCreation(line) for line in goalposts]), GrowFromEdge(p_brace, UP))
        self.wait()

        # Bring up the bit count
        k_label = Tex("k = 0", font_size=60)
        k_number = k_label.make_number_changeable("0")
        k_label.to_edge(UP, buff=1)
        self.play(FadeIn(k_label))
        self.wait()

        # Helpers for adding bits
        width_low = ValueTracker(0)
        width_high = ValueTracker(1)
        width_brace = VGroup(VMobject(), Tex(R"2^{-k}", font_size=36))
        width_brace[1].set_backstroke(BLACK, 3)
        width_grow = ValueTracker(0)

        def update_width_brace(group):
            update_brace(group, width_low.get_value(), width_high.get_value(), UP, 0.3, grow=width_grow.get_value())

        width_brace.add_updater(update_width_brace)

        def get_closest_interval(level, low, high):
            spacing = 2.0**-level
            candidates = range(max(math.floor(low / spacing) - 1, 0), min(math.ceil(high / spacing) + 1, 2**level - 1) + 1)
            j = min(candidates, key=lambda j: (max(low - j * spacing, j * spacing - high, 0), j))
            return j * spacing, (j + 1) * spacing

        def get_step_anims(k):
            log_width = min(0, math.log2(4 * 2.0**-k))
            low, high = get_target_bounds()
            focus = (low + high) / 2
            center = focus + (0.5 - focus) * 4**log_width
            return [
                k_tracker.animate.set_value(k),
                view_log_width.animate.set_value(log_width),
                view_center.animate.set_value(center),
            ]

        def get_brace_anims(k):
            new_low, new_high = get_closest_interval(k, *get_target_bounds())
            return [width_low.animate.set_value(new_low), width_high.animate.set_value(new_high)]

        # Add bits until a tick lands in the target
        for k in range(1, hit_level + 1):
            k_number.set_value(k)
            anims = get_step_anims(k)
            if k > 3:
                anims += get_brace_anims(k)
            if k == 3:
                width_low.set_value(get_closest_interval(k, *get_target_bounds())[0])
                width_high.set_value(get_closest_interval(k, *get_target_bounds())[1])
                width_brace.update()
                self.add(width_brace)
                anims.append(width_grow.animate.set_value(1))
            self.play(*anims, run_time=1.5 if k >= 3 else 0.1)
            self.wait(1)

        # Slide the target so no tick lands inside
        slid_interval = get_closest_interval(hit_level, slid_low, slid_low + p)
        self.play(
            target_low.animate.set_value(slid_low),
            width_low.animate.set_value(slid_interval[0]),
            width_high.animate.set_value(slid_interval[1]),
            run_time=3,
        )
        self.wait()

        # Keep adding bits until the spacing is at most p
        k = hit_level
        while 2.0**-k > p:
            k += 1
            k_number.set_value(k)
            self.play(*get_step_anims(k), *get_brace_anims(k), run_time=1.5)
            self.wait()

        bound = Tex(R"2^{-k} \le p", font_size=48).to_edge(DOWN, buff=1)

        # Show the inequality
        self.play(
            AnimationGroup(
                TransformMatchingShapes(width_brace[1].copy(), bound["2^{-k}"], path_arc=PI * 0.3),
                Write(bound[R"\le"]),
                TransformMatchingShapes(p_brace[1].copy(), bound["p"], path_arc=-PI * 0.3),
                lag_ratio=0.4,
                run_time=2
            )
        )

        # Do the algebra
        algebra = Tex(R"2^{-k} \le p\ \implies\ -k \le \log_2 p\ \implies\ k \ge -\log_2 p", font_size=48).move_to(bound)
        self.play(TransformMatchingShapes(bound, algebra[R"2^{-k} \le p\ "]), run_time=1)
        self.play(FadeIn(algebra[5:15]))
        self.wait(1)
        self.play(FadeIn(algebra[15:]))
        self.wait(2)
        rect = SurroundingRectangle(algebra[19:], fill_opacity=0, stroke_width=3, stroke_color=YELLOW, buff=0.15)
        self.play(ShowCreation(rect), run_time=2)
        information_text = TexText("Information", font_size=34).next_to(algebra[19:], UP, buff=0.32)
        self.play(Write(information_text))