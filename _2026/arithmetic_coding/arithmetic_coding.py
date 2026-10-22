import math
from functools import lru_cache
from fractions import Fraction

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
        self.view_half_width = FRAME_WIDTH / 2 + 1
        self.full_width = width
        self.show_decimal_labels = show_decimal_labels
        self.interval_subdivisions = interval_subdivisions
        self.char_alphabet = char_alphabet
        self.label_to_bar_height_ratio = label_to_bar_height_ratio
        self.context = context

        self.unit_interval = DynamicInterval(width=width)
        self.intervals.add(self.unit_interval)
        self.interval_keys = {(1, 0)}  # Of the form (step_size, lower_bound_int)

        self.curr_text = ""  # Start with a neutral context
        self.char_labels_template = Text(char_alphabet)
        self.layers.add(self.get_new_layer(buff=SMALL_BUFF))

        super().__init__(self.intervals, Point(), self.branches, self.layers)

    def get_distribution(self, prefix):
        return cached_next_char_distribution(self.context + prefix).copy()

    def get_text_range(self, text):
        low, high = 0.0, 1.0
        for i, char in enumerate(text):
            dist = self.get_distribution(text[:i])
            cdf = np.cumsum(dist) / dist.sum()
            index = self.char_alphabet.index(char)
            a = cdf[index - 1] if index > 0 else 0.0
            b = cdf[index]
            low, high = low + (high - low) * a, low + (high - low) * b
        return low, high

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
        layer.shift((mob_above.get_left()[0] - layer.bars.get_left()[0]) * RIGHT)

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

    def add_interval(self, scale, lower_bound_int):
        step = self.interval_subdivisions**(scale)
        interval_key = (step, lower_bound_int)
        if interval_key in self.interval_keys:
            return
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

    def populate_intervals(self, x_min, x_max):
        log_val = np.log(x_max - x_min) / np.log(self.interval_subdivisions)
        min_scale = int(np.ceil(np.round(log_val, 1)))

        for scale in range(-1, min_scale - 1, -1):
            step = self.interval_subdivisions**(scale)
            for lower_bound_int in range(int(x_min / step) - 1, int(x_max / step) + 1):
                self.add_interval(scale, lower_bound_int)
        return self

    def populate_visible_intervals(self, x_left, x_right):
        min_width = self.unit_interval.width_fade_range[0]
        unit_width = x_right - x_left
        v_left = (-self.view_half_width - x_left) / unit_width
        v_right = (self.view_half_width - x_left) / unit_width
        scale = -1
        while self.interval_subdivisions**scale * unit_width > min_width:
            step = self.interval_subdivisions**(scale)
            i_low = max(math.floor(v_left / step) - 1, 0)
            i_high = min(math.floor(v_right / step) + 1, round(1 / step) - 1)
            for lower_bound_int in range(i_low, i_high + 1):
                self.add_interval(scale, lower_bound_int)
            scale -= 1
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
        center_on_text_bar=False,
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
            self.populate_visible_intervals(*(
                (self.unit_interval.n2p(value)[0] + x_shift[0]) * stretch_factor
                for value in (0, 1)
            ))
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
            path = []
            for i, char in enumerate(center_text):
                layer = self.layer_by_prefix.get(center_text[:i])
                if layer is None or not getattr(layer, "is_visible", True) or id(layer) not in target_of:
                    continue
                target_layer = target_of[id(layer)]
                index = self.char_alphabet.index(char)
                path.append((target_layer, index))
            if path:
                deepest_layer, deepest_index = path[-1]
                if center_on_text_bar:
                    label_x = deepest_layer.bars[deepest_index].get_center()[0]
                else:
                    label_x = center[0]
                for target_layer, index in path:
                    bar = target_layer.bars[index]
                    if bar.get_left()[0] <= label_x <= bar.get_right()[0]:
                        target_layer[1][index].set_x(label_x)

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
        # input_phrase = "mathematics"
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

        frame.reorient(0, 0, 0, (-5.60, 2.73, 0.00), 12.89)
        frame.save_state()
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
            frame.animate(time_span=(0.5, 5)).to_default_state().set_y(-1),
            ReplacementTransform(in_rect, out_rect, run_time=3)
        )
        self.wait()
        self.play(
            FadeOut(out_rect),
            ShowCreation(alphabet_rects, lag_ratio=0.05, run_time=3),
        )
        self.wait()
        self.play(
            FadeOut(alphabet_rects, lag_ratio=0.01),
            ShowCreation(punc_rects, lag_ratio=0.1),
        )
        self.wait()
        self.play(
            frame.animate(run_time=3).restore(),
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
    def show_model_producing_layer(self, diagram):
        frame = self.frame
        layer = diagram.get_new_layer()
        diagram.layers.add(layer)
        target = layer.copy()
        layer.set_opacity(0)

        end_frame = frame.copy().match_y(diagram)
        model_group = self.get_decoding_model_group(diagram.curr_text, layer.distribution)
        model_group.set_width(12)
        if model_group.get_height() > 3.5:
            model_group.set_height(3.5)
        model_group.next_to(target, DOWN, buff=0.75)
        model_group.set_x(end_frame.get_x())
        in_text, in_arrow, model, out_arrow, chart = model_group

        frame_top = end_frame.get_top()[1]
        frame_bottom = min(end_frame.get_bottom()[1], model_group.get_bottom()[1] - 0.4)
        self.play(
            frame.animate.set_height(frame_top - frame_bottom).set_y((frame_top + frame_bottom) / 2),
            FadeIn(model),
            GrowArrow(in_arrow),
            FadeIn(in_text),
        )
        self.play(
            GrowArrow(out_arrow),
            LaggedStart(*(GrowFromEdge(bar, DOWN) for bar in chart.bars), lag_ratio=0.02),
            FadeIn(chart.labels, lag_ratio=0.02),
            FadeIn(chart.dec_labels, lag_ratio=0.02),
            run_time=1.5,
        )
        self.wait()

        for bar in chart.bars:
            original = bar.copy()
            bar.rotate(90 * DEG)
            bar.replace(original, stretch=True)
        kw = dict(run_time=2, lag_ratio=0.02)
        self.play(
            ReplacementTransform(chart.bars, target.bars, **kw),
            ReplacementTransform(chart.labels, target.labels, **kw),
            FadeOut(chart.dec_labels, run_time=1),
            FadeOut(Group(in_text, in_arrow, model, out_arrow), shift=DOWN, run_time=1),
            frame.animate(run_time=2).set_height(end_frame.get_height()).move_to(end_frame),
        )
        self.remove(target.bars, target.labels)
        layer.become(target)
        self.add(layer)

    def get_decoding_model_group(self, prefix, distribution):
        chart = self.get_bar_chart(distribution, CHAR_ALPHABET.replace(" ", "_"))
        model = ImageMobject(Path(
            self.file_writer.get_output_file_rootname().parent.parent,
            "Paul Assets", "BigModel.png"
        ))
        model.set_height(0.8 * chart.get_height())
        model.rotate(-2 * DEG)
        model.next_to(chart, LEFT, buff=2)
        out_arrow = Arrow(model.get_right(), chart.get_left(), buff=0.3, thickness=5)
        out_arrow.set_color(GREY_B)
        in_arrow = Vector(1.5 * RIGHT, thickness=5)
        in_arrow.next_to(model, LEFT)
        in_text = Text(f"“{prefix}”", font_size=72)
        in_text.next_to(in_arrow, LEFT, buff=0.15)
        return Group(in_text, in_arrow, model, out_arrow, chart)

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

        self.add(layer.bars, Point(), layer.labels)
        self.play(
            layer.bars.animate.set_fill(YELLOW, 0.7).set_anim_args(rate_func=there_and_back, lag_ratio=0.05, run_time=3),
            FadeIn(new_dec_rects, rate_func=there_and_back, lag_ratio=0.05, run_time=3),
        )
        self.wait()
        self.play(
            Restore(over_brace),
            UpdateFromFunc(width_label, update_width_label),
            run_time=2
        )
        self.wait()

        # Show unit interval
        brace_group = VGroup(over_brace, width_label)
        unit_interval = diagram.unit_interval
        self.play(
            Write(unit_interval, lag_ratio=0.01),
            brace_group.animate.next_to(unit_interval[0], UP, MED_LARGE_BUFF),
            frame.animate.set_width(13).scale(1.1).set_y(0),
            run_time=2
        )
        unit_interval.numbers.set_backstroke(BLACK, 3)
        self.play(FadeOut(brace_group), run_time=2)
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

        self.play(
            frame.animate.scale(1/1.1),
            VFadeOut(x_dec),
            x_tracker.animate.set_value(unit_interval.p2n(layer.bars[19].get_center())),
            run_time=2
        )
        self.play(
            x_tracker.animate(rate_func=wiggle, run_time=2).set_value(unit_interval.p2n(layer.bars[19].get_left() + 0.2 * LEFT)),
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
            diagram.highlight_letter("p", other_bar_opacity=0.25),
            ReplacementTransform(t_brace, p_brace),
            ReplacementTransform(t_brace_label, p_brace_label),
        )
        self.wait()

        # Show "q"
        self.play(
            diagram.zoom_in_on_letter_range(("p", "q"), stretch_factor=18),
            UpdateFromFunc(p_brace, lambda m: m.become(Brace(p_bar, DOWN))),
            UpdateFromFunc(p_brace_label, lambda m: m.next_to(p_brace, DOWN, SMALL_BUFF)),
            run_time=1.3
        )

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
            diagram.highlight_letter("q"),
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
        # def get_letter_prob_label(char):
        #     brace = Brace(diagram.get_letter_bar(char), DOWN, buff=SMALL_BUFF)
        #     label = brace.get_tex(f"P(``\\text{{{char}}}\")", font_size=36, buff=SMALL_BUFF)
        #     return VGroup(brace, label)

        # prob_label = get_letter_prob_label("a")
        # self.play(
        #     GrowFromCenter(prob_label[0]),
        #     Write(prob_label[1]),
        # )
        # for idx in range(1, 13):
        #     char = diagram.char_alphabet[idx]
        #     new_label = get_letter_prob_label(char)
        #     self.play(
        #         ReplacementTransform(prob_label[0], new_label[0]),
        #         FadeTransformPieces(prob_label[1], new_label[1]),
        #     )
        #     prob_label = new_label
        # self.wait()

        p_math = Tex("P(``\\text{math}\")?")
        p_math.to_edge(UP, buff=1)
        self.play(FadeIn(p_math, UP))
        self.wait()


        # Zoom in on the "m"
        show_model = True
        self.play(diagram.highlight_letter("m"))
        self.play(
            diagram.zoom_in_on_letter("m"),
            p_math.animate(path_arc=PI * 0.3).set_opacity(0.5).to_corner(DL, buff=0.5).fix_in_frame(),
        )
        if show_model:
            self.show_model_producing_layer(diagram)
        else:
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
        label = new_brace.get_tex("P(``\\text{m}\")", font_size=35)
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
        new_label = new_brace.get_tex("P(``\\text{m}\")", font_size=35)
        self.play(
            renormalize_anim,
            brace.animate(run_time=3).become(new_brace),
            label.animate(run_time=3).become(new_label)
        )

        # Show the chain rule
        new_brace = Brace(diagram.get_letter_bar("a"), DOWN, buff=SMALL_BUFF)
        new_label = new_brace.get_tex(f"P(``\\text{{m}}\") \\cdot P(``\\text{{a}}\" | ``\\text{{m}}\")", font_size=36, buff=SMALL_BUFF)
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
            label = brace.get_tex(f"P(``\\text{{m{char}}}\")", font_size=36, buff=SMALL_BUFF)
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
        if show_model:
            self.show_model_producing_layer(diagram)
        else:
            self.play(diagram.fade_in_new_layer(), self.camera.frame.animate.match_y(diagram))

        # Show the probabilities for the third layer
        def get_three_letter_prob_label(char):
            brace = Brace(diagram.get_letter_bar(char), DOWN, buff=SMALL_BUFF)
            label = brace.get_tex(f"P(``\\text{{ma{char}}}\")", font_size=36, buff=SMALL_BUFF)
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
        if show_model:
            self.show_model_producing_layer(diagram)
        else:
            self.play(diagram.fade_in_new_layer(), self.camera.frame.animate.match_y(diagram))

        # Attach the probability label
        brace = Brace(diagram.get_letter_bar("h"), DOWN, buff=SMALL_BUFF)
        label = brace.get_tex(f"P(``\\text{{math}}\")", font_size=40, buff=SMALL_BUFF)
        self.play(GrowFromEdge(brace, UP, run_time=2), TransformMatchingShapes(p_math, label, run_time=2))
        self.wait(2)

        # Zoom in on the "h"
        math_bar = diagram.get_letter_bar("h")
        brace.add_updater(lambda m: m.become(Brace(math_bar, DOWN, buff=SMALL_BUFF)))
        label.add_updater(lambda m: brace.put_at_tip(m, buff=SMALL_BUFF))
        self.play(diagram.highlight_letter("h"))
        self.play(diagram.zoom_in_on_letter("h"))
        brace.clear_updaters()
        label.clear_updaters()

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

        self.wait(1)
        self.play(
            VFadeIn(x_arrow, time_span=(0, 1)),
            VFadeIn(x_dec, time_span=(0, 1)),
            x_tracker.animate.set_value(h_max),
            run_time=16.25
        )
        self.play(x_tracker.animate.set_value(h_min), run_time=16.25)

        # Track the word as the x value changes
        word_opacity_tracker = ValueTracker(0)
        def get_word_mob(word):
            mob = Text(f"|{word.replace(" ", "_")}|").set_height(0.7)
            mob.to_edge(UP, buff=1.3)
            mob.remove(mob[0], mob[-1])
            mob.set_opacity(word_opacity_tracker.get_value())
            return mob

        word_tracker = always_redraw(
            lambda: get_word_mob(diagram.get_word_at_value(get_x()).replace(" ", "_"))
        )
        self.add(word_tracker)

        t_zoom = diagram.zoom_in_on_letter("t", layer_index=2, run_time=1)
        t_min, t_max = diagram.get_sweep_range(diagram.get_letter_bar("t", layer_index=2))
        self.play(
            FadeOut(VGroup(brace, label)),
            self.camera.frame.animate.shift(UP),
            word_opacity_tracker.animate.set_value(1),
            t_zoom,
            x_tracker.animate(run_time=6).set_value(t_min)
        )

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
        self.play(a_zoom, move_tracker_through_zoom(diagram, x_tracker, a_max, run_time=4))
        self.play(x_tracker.animate(run_time=3.8).set_value(a_min))

        # Zoom all the way out to the unit interval
        full_zoom = diagram.renormalize_animation(0, 1, fade_thin_strokes=True, run_time=2)
        self.play(full_zoom, move_tracker_through_zoom(diagram, x_tracker, 1, run_time=7))

        # Sweep across every possible four letter word
        self.play(x_tracker.animate(run_time=6).set_value(0))
        word_tracker.suspend_updating()
        word_tracker.become(get_word_mob("aaaa"))
        self.wait(0.3)
        word_tracker.resume_updating()
        self.play(x_tracker.animate.set_value(math_value), run_time=4)
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
            layer = diagram.layers[i]
            restyled = diagram.restore_layer(i, get_x()).target_mobject
            model_group = self.get_decoding_model_group(string[:i], layer.distribution)
            model_group.set_width(12)
            if model_group.get_height() > 3.5:
                model_group.set_height(3.5)
            model_group.next_to(restyled, DOWN, buff=0.75)
            model_group.set_x(frame.get_x())
            in_text, in_arrow, model, out_arrow, chart = model_group

            frame.save_state()
            frame_top = frame.get_top()[1]
            frame_bottom = min(frame.get_bottom()[1], model_group.get_bottom()[1] - 0.4)
            in_text_anims = [FadeIn(VGroup(in_text[0], in_text[-1]))]
            if len(decoded) > 0:
                in_text_anims.append(TransformFromCopy(decoded, in_text[1:-1]))
            self.play(
                frame.animate.set_height(frame_top - frame_bottom).set_y((frame_top + frame_bottom) / 2),
                FadeIn(model),
                GrowArrow(in_arrow),
                *in_text_anims,
            )
            self.play(
                GrowArrow(out_arrow),
                LaggedStart(*(GrowFromEdge(bar, DOWN) for bar in chart.bars), lag_ratio=0.02),
                FadeIn(chart.labels, lag_ratio=0.02),
                FadeIn(chart.dec_labels, lag_ratio=0.02),
                run_time=1.5,
            )
            self.wait()

            for bar in chart.bars:
                original = bar.copy()
                bar.rotate(90 * DEG)
                bar.replace(original, stretch=True)
            kw = dict(run_time=2, lag_ratio=0.02)
            self.play(
                ReplacementTransform(chart.bars, restyled.bars, **kw),
                ReplacementTransform(chart.labels, restyled.labels, **kw),
                FadeOut(chart.dec_labels, run_time=1),
                FadeOut(Group(in_text, in_arrow, model, out_arrow), shift=DOWN, run_time=1),
                Restore(frame, run_time=2),
            )
            self.remove(restyled.bars, restyled.labels)
            layer.become(restyled)
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

        tick_glyph_template = Text("0.1", font="Consolas", font_size=20)
        tick_glyph_advance = (tick_glyph_template[2].get_x() - tick_glyph_template[0].get_x()) / 2
        tick_glyph_height = tick_glyph_template[0].get_height()
        tick_glyph_origin = tick_glyph_template[0].get_center()
        tick_glyphs = {
            char: (glyph, glyph.get_center() - tick_glyph_origin - i * tick_glyph_advance * RIGHT)
            for i, (char, glyph) in enumerate(zip("0.1", tick_glyph_template))
        }
        tick_label_cache = dict()

        def get_tick_label(string):
            if string not in tick_label_cache:
                label = VGroup()
                for i, char in enumerate(string):
                    glyph, offset = tick_glyphs[char]
                    label.add(glyph.copy().move_to(offset + (i - (len(string) - 1) / 2) * tick_glyph_advance * RIGHT))
                label.anchor = np.zeros(3)
                label.last_opacity = None
                tick_label_cache[string] = label
            return tick_label_cache[string]

        def get_tick_string(numer, level):
            if level == 0:
                return str(numer)
            return "." + bin(numer)[2:].zfill(level)

        def get_dyadic_ruler(fade=0, ref_spacing=0.35, max_height=0.45, height_ratio=0.7, max_depth=5, n_ticks=500, margin=0.5):
            ticks = VGroup(Line(ORIGIN, UP) for _ in range(n_ticks))
            ticks.set_stroke(GREY_B, 2)
            ticks.fade = ValueTracker(fade)
            ticks.set_z_index(1)
            ticks.labels = VGroup()
            ticks.labels.set_z_index(1)
            ticks.level_fits = dict()

            def update_ruler(ticks, dt):
                ref_low, ref_high = highlight_range
                x_ref_low = value_to_x(ref_low)
                slope = (value_to_x(ref_high) - x_ref_low) / (ref_high - ref_low)
                frame_width = frame.get_width()
                v_left = ref_low + (frame.get_left()[0] - margin - x_ref_low) / slope
                v_right = ref_low + (frame.get_right()[0] + margin - x_ref_low) / slope
                fade = ticks.fade.get_value()
                label_y = line_y + 0.3 + tick_glyph_height / 2
                shown_labels = []
                n_used = 0
                level = 0
                while n_used < len(ticks):
                    depth = math.log2(ref_spacing * frame_width / (slope * 2.0**-level))
                    if depth >= max_depth:
                        break
                    opacity = fade * clip(max_depth - depth, 0, 1)
                    height = max_height * height_ratio**clip(depth, 0, max_depth)
                    stroke_width = interpolate(1, 2.5, height / max_height)
                    spacing = slope * 2.0**-level
                    label_length = level + 1 if level > 0 else 1
                    width_fit = clip((spacing / (label_length * tick_glyph_advance) - 1.05) / 0.25, 0, 1)
                    smooth_fit = width_fit * clip((spacing / frame_width - 0.2) / 0.12, 0, 1)
                    target_fit = float(smooth_fit >= 0.5)
                    level_fit = ticks.level_fits.get(level, target_fit)
                    level_fit += clip(target_fit - level_fit, -dt / 0.4, dt / 0.4)
                    ticks.level_fits[level] = level_fit
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
                        label_opacity = opacity * level_fit
                        if label_opacity <= 1e-3:
                            continue
                        label = get_tick_label(get_tick_string(i, level))
                        position = np.array([x, label_y, 0])
                        label.shift(position - label.anchor)
                        label.anchor = position
                        if label.last_opacity != label_opacity:
                            label.set_fill(opacity=label_opacity)
                            label.last_opacity = label_opacity
                        shown_labels.append(label)
                    level += 1
                for tick in ticks[n_used:]:
                    tick.set_stroke(opacity=0)
                if ticks.labels.submobjects != shown_labels:
                    ticks.labels.set_submobjects(shown_labels)

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
        bin_template = Text("0.1", font="Consolas").scale(bin_scale)
        bin_advance = (bin_template[2].get_x() - bin_template[0].get_x()) / 2
        bin_glyphs = {
            char: (glyph, glyph.get_center() - bin_template[0].get_center() - i * bin_advance * RIGHT)
            for i, (char, glyph) in enumerate(zip("0.1", bin_template))
        }

        def get_binary_text(string):
            mob = VGroup()
            for i, char in enumerate(string):
                glyph, offset = bin_glyphs[char]
                mob.add(glyph.copy().move_to(offset + i * bin_advance * RIGHT))
            return mob

        def get_binary_number(bits, highlight_last=True):
            mob = get_binary_text("." + bits)
            mob.set_x(bin_center_x)
            mob.shift((bin_bottom_y - mob.get_bottom()[1]) * UP)
            if highlight_last:
                mob[-1].set_color(YELLOW)
            return mob

        level_marks = [get_dyadic_marks(1)]

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
        word_low, word_high = diagram.get_text_range(word)
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
                self.add(ruler, ruler.labels)
                self.play(
                    UpdateFromAlphaFunc(Mobject(), lambda m, a: old_marks.set_opacity(1 - a)),
                    ruler.fade.animate.set_value(1),
                )
                self.remove(*level_marks)

        self.remove(binary_number)
        binary_number = get_binary_number(bits, highlight_last=False)
        self.add(binary_number)
        self.wait()

        binary_number_line = VGroup(binary_line, *level_marks)

        # Show the range of real numbers
        def get_aligned_binary_text(string):
            text = get_binary_text(string)
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
            zoom_to_interval(*red_range, run_time=2, center_text=word, center_on_text_bar=True),
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
        self.play(
            UpdateFromAlphaFunc(highlight, lambda m, a: m.set_stroke(interpolate_color(YELLOW, RED, a))),
            ShowCreation(target_lines),
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
        def get_avg_bits_mobs(n_bits_used, text):
            avg_number = n_bits_used / len(text)
            avg_str = f"{avg_number:.2f}".rstrip("0").rstrip(".")
            relation = "=" if abs(avg_number - float(avg_str)) < 1e-9 else R"\approx"
            frac = Tex(
                Rf"\displaystyle\frac{{{n_bits_used} \text{{ bits}}}}{{|\text{{``{text}''}}|}}",
                font_size=32,
            )
            value = Tex(Rf"{relation} {avg_str} \text{{ bits/char}}", font_size=32)
            value.next_to(frac, DOWN, buff=0.25, aligned_edge=LEFT)
            num_end = len(str(n_bits_used)) + 4
            word_start = num_end + 4
            word_end = word_start + len(text)
            return frac, value, num_end, word_start, word_end

        avg_frac, avg_value, num_end, word_start, word_end = get_avg_bits_mobs(n_bits, word)
        avg_bits_per_char = VGroup(avg_frac, avg_value)
        fix_to_screen(avg_bits_per_char)
        avg_bits_per_char.align_to(word_mob, UP).to_edge(RIGHT, buff=1)
        copy = bits_label.copy()
        self.play(
            AnimationGroup(
                copy.animate(path_arc=PI * 0.3).match_width(avg_frac[:num_end]).move_to(avg_frac[:num_end]),
                GrowFromCenter(avg_frac[num_end]),
                AnimationGroup(
                    TransformFromCopy(word_mob, avg_frac[word_start:word_end], path_arc=PI * 0.3),
                    FadeIn(VGroup(avg_frac[num_end + 1:word_start], avg_frac[word_end:])),
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
            text = get_binary_text("." + bit_str)
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
        new_chars = full_word[len(word):]
        for char_index, char in enumerate(new_chars):
            is_last = char_index == len(new_chars) - 1
            new_layer = diagram.make_layer(prefix, target_bar)
            diagram.align_layer_to_bar(new_layer, target_bar)
            diagram.layers.add(new_layer)
            index = diagram.char_alphabet.index(char)
            target_bar = new_layer.bars[index]
            prefix += char
            word_range = diagram.get_text_range(prefix)
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
            if not needs_bits and not is_last:
                view = (word_range, 0.6 * frame_width)
            layer_zoom = zoom_to_world_width(*view[0], view[1], center_text=prefix, center_on_text_bar=True)
            target_of = dict(zip(map(id, diagram.get_family()), diagram.target.get_family()))
            target_layer = target_of[id(new_layer)]
            target_layer.set_opacity(1)
            diagram.reset_layer_style(target_layer)
            target_layer.reposition_labels()
            target_layer.highlight(index, other_bar_opacity=0.35)
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
                self.play(
                    UpdateFromAlphaFunc(highlight, lambda m, a: m.set_stroke(interpolate_color(YELLOW, RED, a))),
                    run_time=0.5,
                )

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

            if needs_bits and not is_last:
                view = (word_range, 0.6 * frame.get_width())
                self.play(zoom_to_world_width(*view[0], view[1], center_text=prefix, center_on_text_bar=True))
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
        full_frac, full_value, num_end, word_start, word_end = get_avg_bits_mobs(total_bits, full_word)
        full_avg = VGroup(full_frac, full_value)
        fix_to_screen(full_avg)
        full_avg.align_to(word_mob, UP).to_edge(RIGHT, buff=1)
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
    word = "info"

    def construct(self):
        # Measure the target word
        frame = self.frame
        word = self.word
        line_width = 12

        char_labels_template = Text(CHAR_ALPHABET)
        layer_cache = dict()

        def get_layer(prefix):
            if prefix not in layer_cache:
                layer = StackedProbDistribution(
                    cached_next_char_distribution(" " + prefix).copy(),
                    labels=char_labels_template.copy(),
                    width=line_width,
                    label_height_ratio=0.5,
                )
                bars = layer.bars
                cdf = np.cumsum(np.array(layer.distribution, dtype=float))
                cdf /= cdf[-1]
                layer.edges = list(zip([0.0, *cdf[:-1]], [*cdf[:-1], 1.0]))
                layer.default_fills = [(bar.get_fill_color(), bar.get_fill_opacity()) for bar in bars]
                layer.default_stroke_opacities = [bar.get_stroke_opacity() for bar in bars]
                layer.path_index = None
                top = layer.get_top()[1]
                widest = max(bars, key=lambda bar: bar.get_width())
                unit_points = widest.get_points().astype(float)
                unit_points[:, 0] = (unit_points[:, 0] - widest.get_left()[0]) / widest.get_width()
                unit_points[:, 1] -= top
                layer.bar_unit_points = unit_points
                layer.label_points = [
                    [
                        sm.get_points().astype(float) - np.array([label.get_x(), top, 0])
                        for sm in label.get_family()
                    ]
                    for label in layer.labels
                ]
                layer.label_widths = [label.get_width() for label in layer.labels]
                layer_cache[prefix] = layer
            return layer_cache[prefix]

        def decode(value, n_chars):
            ranges = [(0.0, 1.0)]
            indices = []
            prefix = ""
            for _ in range(n_chars):
                edges = get_layer(prefix).edges
                low, high = ranges[-1]
                alpha = (value - low) / (high - low)
                index = next((i for i, (a, b) in enumerate(edges) if alpha < b), len(edges) - 1)
                a, b = edges[index]
                ranges.append((low + (high - low) * a, low + (high - low) * b))
                indices.append(index)
                prefix += CHAR_ALPHABET[index]
            return prefix, ranges, indices

        word_ranges = [(0.0, 1.0)]
        for i, char in enumerate(word):
            a, b = get_layer(word[:i]).edges[CHAR_ALPHABET.index(char)]
            range_low, range_high = word_ranges[-1]
            word_ranges.append((range_low + (range_high - range_low) * a, range_low + (range_high - range_low) * b))
        word_low, word_high = word_ranges[-1]

        # Choose target positions based on the word
        p = word_high - word_low
        final_level = math.ceil(-math.log2(p) - 1e-9)
        hit_level = final_level - 2
        brace_level = min(3, hit_level)
        region_level = min(brace_level + 4, hit_level)
        word_center = (word_low + word_high) / 2

        hit_denom = 2**hit_level
        hit_numer = round(word_center * hit_denom)
        if hit_numer % 2 == 0:
            hit_numer += 1 if word_center * hit_denom > hit_numer else -1
        hit_value = hit_numer / hit_denom

        if len(ArithmeticCodingDiagram.get_binary_code(word_low, word_high)) == final_level:
            slid_low = word_low
        else:
            slid_spacing = 2.0**-(final_level - 1)
            slid_low = math.floor(word_center / slid_spacing) * slid_spacing + (slid_spacing - p) / 2

        # Set up the number line
        view_center = ValueTracker(0.5)
        view_log_width = ValueTracker(0)
        k_tracker = ValueTracker(0)
        label_k_tracker = ValueTracker(0)
        target_low = ValueTracker(hit_value - 0.35 * p)
        p_tracker = ValueTracker(p)
        p_label_shift = ValueTracker(0)
        line_y_tracker = ValueTracker(0)
        ruler_alpha = ValueTracker(0)

        def get_line_y():
            return line_y_tracker.get_value()

        def get_scale():
            return line_width / 2**view_log_width.get_value()

        def value_to_x(value):
            return (value - view_center.get_value()) * get_scale()

        def n2p(value, y=None):
            return np.array([value_to_x(value), get_line_y() if y is None else y, 0])

        modes = {"bits": "code", "diagram": "word"}

        def get_target_bounds():
            if modes["bits"] == "value":
                return decode(get_arrow_value(), len(word))[1][-1]
            low = target_low.get_value() - (p_tracker.get_value() - p) / 2
            return low, low + p_tracker.get_value()

        word_bits = ArithmeticCodingDiagram.get_binary_code(word_low, word_high)
        value_tracker = ValueTracker(0)

        n_shown_bits = max(final_level, len(word_bits))

        def get_code_bits():
            if modes["bits"] == "value":
                return ArithmeticCodingDiagram.get_binary_digits(value_tracker.get_value(), n_shown_bits)
            return ArithmeticCodingDiagram.get_binary_code(*get_target_bounds())

        def get_shown_bits():
            return get_code_bits().ljust(n_shown_bits, "0")

        def get_arrow_value():
            bits = get_shown_bits()
            return int(bits, 2) / 2**len(bits)

        def set_view(start_center, start_log_width, new_center, new_log_width, focus, alpha):
            start_x = (focus - start_center) * line_width / 2**start_log_width
            end_x = (focus - new_center) * line_width / 2**new_log_width
            log_width = interpolate(start_log_width, new_log_width, alpha)
            x = interpolate(start_x, end_x, alpha)
            view_log_width.set_value(log_width)
            view_center.set_value(focus - x * 2**log_width / line_width)

        def get_view_anim(new_center, new_log_width, focus, **kwargs):
            start_center = view_center.get_value()
            start_log_width = view_log_width.get_value()

            def update_view(mob, alpha):
                set_view(start_center, start_log_width, new_center, new_log_width, focus, alpha)

            return UpdateFromAlphaFunc(view_center, update_view, **kwargs)

        number_line = Line(LEFT, RIGHT).set_stroke(GREY_B, 2)

        def update_number_line(line):
            left = max(value_to_x(0), frame.get_left()[0] - 1)
            right = min(value_to_x(1), frame.get_right()[0] + 1)
            line.put_start_and_end_on(np.array([left, get_line_y(), 0]), np.array([right, get_line_y(), 0]))

        number_line.add_updater(update_number_line)
        number_line.update()

        glyph_template = Text("0.1", font="Consolas", font_size=20)
        glyph_advance = (glyph_template[2].get_x() - glyph_template[0].get_x()) / 2
        glyph_height = glyph_template[0].get_height()
        glyph_origin = glyph_template[0].get_center()
        glyphs = {
            char: (glyph, glyph.get_center() - glyph_origin - i * glyph_advance * RIGHT)
            for i, (char, glyph) in enumerate(zip("0.1", glyph_template))
        }
        tick_label_cache = dict()
        tick_label_fade = ValueTracker(0)
        tick_label_boost = ValueTracker(0)
        other_label_fade = ValueTracker(1)
        tick_labels = VGroup()

        def get_tick_label(string):
            if string not in tick_label_cache:
                label = VGroup()
                for i, char in enumerate(string):
                    glyph, offset = glyphs[char]
                    label.add(glyph.copy().move_to(offset + (i - (len(string) - 1) / 2) * glyph_advance * RIGHT))
                label.anchor = np.zeros(3)
                label.last_opacity = None
                tick_label_cache[string] = label
            return tick_label_cache[string]

        def get_tick_string(numer, level):
            if level == 0:
                return str(numer)
            return "." + bin(numer)[2:].zfill(level)

        def get_ticks(ref_spacing=0.35, max_height=0.45, height_ratio=0.7, ruler_depth=5, n_ticks=400, margin=0.5):
            ticks = VGroup(Line(ORIGIN, UP) for _ in range(n_ticks))
            ticks.set_stroke(GREY_B, 2)
            ticks.set_z_index(1)
            ticks.last_key = None
            ticks.n_used = len(ticks)

            def update_ticks(ticks):
                key = (
                    view_center.get_value(), view_log_width.get_value(), k_tracker.get_value(),
                    frame.get_width(), get_line_y(), ruler_alpha.get_value(), tick_label_fade.get_value(),
                    tick_label_boost.get_value(), other_label_fade.get_value(), label_k_tracker.get_value(),
                )
                if key == ticks.last_key:
                    return
                ticks.last_key = key
                scale = get_scale()
                frame_width = frame.get_width()
                line_y = get_line_y()
                v_left = view_center.get_value() + (frame.get_left()[0] - margin) / scale
                v_right = view_center.get_value() + (frame.get_right()[0] + margin) / scale
                k_value = k_tracker.get_value()
                alpha = ruler_alpha.get_value()
                label_fade = tick_label_fade.get_value()
                label_boost = tick_label_boost.get_value()
                label_y = line_y - 0.3 - glyph_height / 2
                shown_labels = []
                n_used = 0
                for level in range(64):
                    depth = math.log2(ref_spacing * frame_width / (scale * 2.0**-level))
                    bit_opacity = clip(k_value - level + 1, 0, 1)
                    ruler_opacity = clip(ruler_depth - depth, 0, 1)
                    if bit_opacity <= 0 and (alpha <= 0 or ruler_opacity <= 0):
                        break
                    opacity = interpolate(bit_opacity, ruler_opacity, alpha)
                    if opacity <= 0:
                        continue
                    height = max_height * height_ratio**clip(depth, 0, 6)
                    stroke_width = interpolate(1, 2.5, height / max_height)
                    level_label_fade = interpolate(clip(label_k_tracker.get_value() - level + 1, 0, 1), 1, alpha)
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
                        string = get_tick_string(i, level)
                        label_width = len(string) * glyph_advance
                        fit = clip((scale * 2.0**-level / label_width - 1.2) / 0.5, 0, 1)
                        label_opacity = min(opacity * fit * interpolate(1, 20, label_boost), 1) * label_fade * level_label_fade
                        if string != "0":
                            label_opacity *= other_label_fade.get_value()
                        if label_opacity <= 0:
                            continue
                        label = get_tick_label(string)
                        position = np.array([x, label_y, 0])
                        label.shift(position - label.anchor)
                        label.anchor = position
                        if label.last_opacity != label_opacity:
                            label.set_fill(opacity=label_opacity)
                            label.last_opacity = label_opacity
                        shown_labels.append(label)
                for tick in ticks[n_used:ticks.n_used]:
                    tick.set_stroke(opacity=0)
                ticks.n_used = n_used
                if tick_labels.submobjects != shown_labels:
                    tick_labels.set_submobjects(shown_labels)

            ticks.add_updater(update_ticks)
            ticks.update()
            return ticks

        ticks = get_ticks()
        self.add(tick_labels)

        # Create number line
        self.play(
            FadeIn(number_line),
            FadeIn(ticks),
            tick_label_fade.animate.set_value(1),
        )
        self.wait()

        # Set up goal posts and bit count
        goalpost_bottom = ValueTracker(-0.75)
        goalpost_top = ValueTracker(1.5)

        def get_goalpost(bottom, top):
            return DashedLine(np.array([0, bottom, 0]), np.array([0, top, 0])).set_stroke(WHITE, 3)

        goalpost_color = ValueTracker(0)
        target_fade = ValueTracker(0)
        goalposts = VGroup(get_goalpost(-0.75, 1.5) for _ in range(2))
        goalposts.set_z_index(2)
        goalposts.style_key = None

        def update_goalposts(group):
            ends = (goalpost_bottom.get_value(), get_line_y() + goalpost_top.get_value())
            style_key = (ends, goalpost_color.get_value(), target_fade.get_value())
            if style_key != group.style_key:
                for post in group:
                    post.become(get_goalpost(*ends))
                    post.set_stroke(interpolate_color(WHITE, GREEN, style_key[1]), opacity=style_key[2])
                group.style_key = style_key
            for post, value in zip(group, get_target_bounds()):
                post.set_x(value_to_x(value))

        goalposts.add_updater(update_goalposts)
        goalposts.update()

        def update_brace(group, low, high, direction, buff, y=None, grow=1, label_shift=0, opacity=1):
            if not hasattr(group, "label_template"):
                group.label_template = group[1].copy()
            y = get_line_y() if y is None else y
            ends = (value_to_x(low), value_to_x(high))
            state = (grow, label_shift, y, opacity)
            last_ends = getattr(group, "last_ends", None)
            last_state = getattr(group, "last_state", (1, 0, y, 1))
            if last_ends == ends and last_state == state:
                return
            same_width = last_ends is not None and np.isclose(ends[1] - ends[0], last_ends[1] - last_ends[0])
            if same_width and state == last_state and grow == 1:
                group.shift((ends[0] - last_ends[0]) * RIGHT)
            else:
                group[0].become(Brace(Line(n2p(low, y), n2p(high, y)), direction, buff=buff))
                group[1].become(group.label_template)
                group[1].next_to(group[0], direction, SMALL_BUFF)
                group[1].shift(label_shift * RIGHT)
                if opacity < 1:
                    group.set_opacity(opacity)
                if grow < 1:
                    group.scale(max(grow, 1e-3), about_edge=-direction)
            group.last_ends = ends
            group.last_state = state

        p_brace = VGroup(VMobject(), Tex("p", font_size=36))
        p_brace_fade = ValueTracker(1)
        p_label_fade = ValueTracker(0)

        def update_p_brace(group):
            opacity = target_fade.get_value() * p_brace_fade.get_value()
            update_brace(
                group, *get_target_bounds(), DOWN, 0.1,
                y=goalpost_bottom.get_value(),
                label_shift=p_label_shift.get_value(),
                opacity=opacity,
            )
            group[1].set_opacity(opacity * p_label_fade.get_value())

        k_label = Tex("k = 0", font_size=60)
        k_number = k_label.make_number_changeable("0")
        k_label.to_edge(UP, buff=1)
        k_label.fade_time = 0

        def fade_in_k_label(mob, dt):
            mob.fade_time += dt
            mob.set_opacity(clip(mob.fade_time / 2, 0, 1))

        width_low = ValueTracker(0)
        width_high = ValueTracker(1)
        width_brace = VGroup(VMobject(), Tex(R"2^{-k}", font_size=36))
        width_brace[1].set_backstroke(BLACK, 3)
        width_grow = ValueTracker(0)
        width_fade = ValueTracker(1)

        def update_width_brace(group):
            update_brace(
                group, width_low.get_value(), width_high.get_value(), UP, 0.3,
                grow=width_grow.get_value(), opacity=width_fade.get_value(),
            )

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
                get_view_anim(center, log_width, focus),
                label_k_tracker.animate.set_value(k),
            ]

        def get_brace_anims(k):
            new_low, new_high = get_closest_interval(k, *get_target_bounds())
            return [width_low.animate.set_value(new_low), width_high.animate.set_value(new_high)]

        # Add bits until a tick lands in the target
        k_label.add_updater(fade_in_k_label)
        k_label.update(0)
        self.add(k_label)
        for k in range(1, hit_level + 1):
            k_number.set_value(k)
            anims = get_step_anims(k)
            if k > brace_level:
                self.play(*anims, *get_brace_anims(k), run_time=2.5)
                self.wait(0.5)
            else:
                self.play(
                    anims[1],
                    k_tracker.animate(time_span=(0, 1)).set_value(k),
                    label_k_tracker.animate(time_span=(0, 1)).set_value(k),
                    run_time=2,
                )

            if k == brace_level:
                k_label.remove_updater(fade_in_k_label)
                k_label.set_opacity(1)
                width_low.set_value(get_closest_interval(k, *get_target_bounds())[0])
                width_high.set_value(get_closest_interval(k, *get_target_bounds())[1])
                width_brace.update()
                self.add(width_brace)
                self.play(width_grow.animate.set_value(1), run_time=1)
                self.wait()

            if k == region_level:
                target_center = target_low.get_value() + p / 2
                start_p = min(0.3 * 2**view_log_width.get_value(), 1.8 * min(target_center, 1 - target_center))

                def shrink_target(mob, alpha):
                    p_tracker.set_value(start_p * (p / start_p)**alpha)

                p_tracker.set_value(start_p)
                p_brace.add_updater(update_p_brace)
                p_brace.update()
                goalposts.update()
                self.add(goalposts, p_brace)
                self.play(
                    UpdateFromAlphaFunc(p_tracker, shrink_target, run_time=5),
                    target_fade.animate(time_span=(0, 1.5)).set_value(1),
                )
                p_tracker.set_value(p)
                self.wait(1)
                self.play(p_label_fade.animate.set_value(1))
                self.wait()

        # Slide the target so no tick lands inside
        self.play(target_low.animate.set_value(slid_low), run_time=3)
        self.wait()

        # Keep adding bits until the spacing is at most p
        k = hit_level
        while 2.0**-k > p:
            k += 1
            k_number.set_value(k)
            self.play(*get_step_anims(k), *get_brace_anims(k), run_time=1)

        # Highlight the tick inside the range
        highlight_level = k
        tick_highlight_fade = ValueTracker(0)
        tick_highlights = VGroup(Line(ORIGIN, UP) for _ in range(3))
        tick_highlights.set_stroke(YELLOW, 4)
        tick_highlights.set_z_index(2)

        def update_tick_highlights(group):
            low, high = get_target_bounds()
            denom = 2**highlight_level
            scale = get_scale()
            numers = list(range(math.ceil(low * denom), math.floor(high * denom) + 1))
            for i, line in enumerate(group):
                if i >= len(numers):
                    line.set_stroke(opacity=0)
                    continue
                numer, level = numers[i], highlight_level
                while level > 0 and numer % 2 == 0:
                    numer //= 2
                    level -= 1
                depth = math.log2(0.35 * frame.get_width() / (scale * 2.0**-level))
                height = 0.45 * 0.7**clip(depth, 0, 6)
                x = value_to_x(numers[i] / denom)
                line.put_start_and_end_on(
                    np.array([x, get_line_y() - height / 2, 0]),
                    np.array([x, get_line_y() + height / 2, 0]),
                )
                line.set_stroke(opacity=tick_highlight_fade.get_value())

        tick_highlights.add_updater(update_tick_highlights)
        tick_highlights.update()
        self.add(tick_highlights)

        original_low = target_low.get_value()
        spacing = 2.0**-highlight_level
        view_left = view_center.get_value() - 2**view_log_width.get_value() / 2
        view_right = view_center.get_value() + 2**view_log_width.get_value() / 2
        right_shift = min(1.5 * spacing, view_right - 0.25 * spacing - (original_low + p))
        left_shift = min(1.5 * spacing, original_low - view_left - 0.25 * spacing)
        self.play(
            target_low.animate(run_time=1.5).set_value(original_low + right_shift),
            tick_highlight_fade.animate(run_time=0.5).set_value(1)
        )
        self.play(target_low.animate.set_value(original_low - left_shift), run_time=1.5)
        self.play(target_low.animate.set_value(original_low), run_time=1.5)
        self.play(tick_highlight_fade.animate.set_value(0))
        self.remove(tick_highlights)
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
        self.play(TransformMatchingShapes(bound, algebra[:5]), run_time=1)
        log_version = Tex(R"\log_2(2^{-k}) \le \log_2 p", font_size=48).match_y(bound).align_to(algebra[7], LEFT)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    GrowFromEdge(algebra[5:7], LEFT),
                    TransformFromCopy(algebra[:3], log_version[5:8]),
                    TransformFromCopy(algebra[3], log_version[9]),
                    TransformFromCopy(algebra[4], log_version[14]),
                    run_time=2
                ),
                FadeIn(VGroup(log_version[:5], log_version[8], log_version[10:14])),
                lag_ratio=0.7
            )
        )
        self.play(
            FadeOut(VGroup(log_version[:6], log_version[8])),
            TransformMatchingShapes(log_version[6], algebra[7]),
            TransformMatchingShapes(log_version[7], algebra[8]),
            TransformMatchingShapes(log_version[8:], algebra[9:15])
        )
        self.wait(1)
        copy = algebra[9].copy()
        self.play(
            GrowFromEdge(algebra[15:17], LEFT),
            AnimationGroup(
                TransformFromCopy(algebra[7], algebra[19], path_arc=-PI * 0.3),
                TransformFromCopy(algebra[8], algebra[17], path_arc=-PI * 0.3),
                copy.animate(path_arc=-PI * 0.3).move_to(algebra[18]).rotate(PI, axis=UP),
                TransformFromCopy(algebra[10:15], algebra[20:], path_arc=-PI * 0.3),
                lag_ratio=0.1,
                run_time=2
            )
        )
        self.remove(copy)
        self.add(algebra)
        rect = SurroundingRectangle(algebra[19:], fill_opacity=0, stroke_width=3, stroke_color=YELLOW, buff=0.1)
        self.play(ShowCreation(rect), run_time=2)
        information_text = TexText("Information", font_size=34).next_to(algebra[19:], UP, buff=0.32)
        self.play(Write(information_text))
        self.wait(2)

        # Bring it to the center
        inequality = algebra[17:]
        self.play(
            FadeOut(algebra[:17], run_time=1),
            VGroup(inequality, rect, information_text).animate(run_time=1.8).set_x(0).scale(1.2)
        )
        self.wait(2)
        self.play(FadeOut(VGroup(rect, information_text)))

        # Show non-whole numbers of bits
        p_decimals = max(2, 2 - math.floor(math.log10(p)))
        p_value = VGroup(Tex("=", font_size=36), DecimalNumber(p, num_decimal_places=p_decimals, font_size=36))
        p_value.arrange(RIGHT, buff=SMALL_BUFF)
        label_shift = -(p_value.get_width() + SMALL_BUFF) / 2
        p_value.next_to(p_brace[1], RIGHT, buff=SMALL_BUFF).shift(label_shift * RIGHT)

        def update_p_value(group):
            group[1].set_value(p_tracker.get_value())
            group.next_to(p_brace[1], RIGHT, buff=SMALL_BUFF)

        info_value = VGroup(
            Tex("="),
            DecimalNumber(-math.log2(p), num_decimal_places=2),
            TexText("bits"),
        )
        info_value.arrange(RIGHT, buff=0.2)
        info_value[2].align_to(info_value[1], DOWN)
        info_value.scale(0.9)
        info_value.next_to(rect, RIGHT, buff=0.1)

        def update_info_value(group):
            group[1].set_value(-math.log2(p_tracker.get_value()))

        self.play(
            p_label_shift.animate.set_value(label_shift),
            FadeIn(p_value, shift=LEFT * 0.8),
            FadeIn(info_value),
        )
        p_value.add_updater(update_p_value)
        info_value.add_updater(update_info_value)

        # Explore the range of p values
        p_max = 0.95 * 2.0**-(k - 1)
        p_min = 1.05 * 2.0**-k
        spacing = 2.0**-k
        rng = random.Random(10)
        scale = get_scale()
        screen_margin = 1
        waypoints = []
        x = value_to_x(slid_low + p / 2)
        for _ in range(4):
            p_value_target = rng.uniform(p_min, p_max)
            half_width = p_value_target * scale / 2
            x_min = frame.get_left()[0] + screen_margin + half_width
            x_max = frame.get_right()[0] - screen_margin - half_width
            step = rng.choice([-1, 1]) * rng.uniform(1, 3.5)
            if not x_min <= x + step <= x_max:
                step = -step
            x = clip(x + step, x_min, x_max)
            center = view_center.get_value() + x / scale
            waypoints.append((p_value_target, center - p / 2, 1.5))
        waypoints.append((p, slid_low, 1.75))
        for p_value_target, low_target, run_time in waypoints:
            self.play(
                p_tracker.animate.set_value(p_value_target),
                target_low.animate.set_value(low_target),
                run_time=run_time,
            )

        # Round up to a whole number of bits
        p_value.clear_updaters()
        info_value.clear_updaters()
        ceil_inequality = Tex(R"k = \lceil -\log_2 p \rceil", font_size=48).scale(1.2)
        ceil_inequality.move_to(inequality)
        self.play(
            FadeOut(p_value, shift=RIGHT * 0.8),
            FadeOut(info_value),
            p_label_shift.animate.set_value(0),
            TransformMatchingShapes(inequality[0], ceil_inequality[0]),
            TransformMatchingShapes(inequality[1], ceil_inequality[1]),
            TransformMatchingShapes(inequality[2:], ceil_inequality[3:-1]),
            FadeIn(VGroup(ceil_inequality[2], ceil_inequality[-1])),
            run_time=1.5,
        )
        self.wait(2)

        # Helpers for the word diagram
        layer_buff = 0.1
        layer_height = get_layer("").get_height()
        first_layer_gap = 0.75
        below_word = 0.1 + 0.3 + 0.45
        stack_height = first_layer_gap + len(word) * (layer_height + layer_buff)
        new_line_y = frame.get_bottom()[1] + 0.5 + below_word + stack_height
        bits_bottom_y = new_line_y + 1.2
        new_goalpost_top = min(1.5, bits_bottom_y - 0.25 - new_line_y)
        new_goalpost_bottom = new_line_y - stack_height + layer_buff

        def style_layer(layer, index):
            if layer.path_index == index:
                return
            for bar, (color, opacity), stroke_opacity in zip(layer.bars, layer.default_fills, layer.default_stroke_opacities):
                bar.set_fill(color, opacity)
                bar.set_stroke(opacity=stroke_opacity)
            layer.highlight(index, other_bar_opacity=0.35)
            layer.path_index = index
            layer.base_opacities = [(bar.get_fill_opacity(), bar.get_stroke_opacity()) for bar in layer.bars]
            layer.applied_fade = None

        def get_diagram_value():
            return word_center if modes["diagram"] == "word" else get_arrow_value()

        layers_fade = ValueTracker(0)
        word_layer_group = VGroup()
        word_layer_group.last_key = None

        def place_layer(layer, left, right, top, path_index, fade):
            x_min = frame.get_left()[0] - 1
            x_max = frame.get_right()[0] + 1
            unit_points = layer.bar_unit_points
            gradient_start = np.array([left, top, 0])
            gradient_end = np.array([max(right, left + 1e-6), top, 0])
            for j, (bar, label, (a, b)) in enumerate(zip(layer.bars, layer.labels, layer.edges)):
                bar_left = left + (right - left) * a
                bar_right = left + (right - left) * b
                l = clip(bar_left, x_min, x_max)
                r = clip(bar_right, x_min, x_max)
                points = unit_points.copy()
                points[:, 0] = l + (r - l) * unit_points[:, 0]
                points[:, 1] += top
                bar.set_points(points)
                if "gradient_start" in bar.uniforms:
                    bar.uniforms["gradient_start"] = gradient_start.copy()
                    bar.uniforms["gradient_end"] = gradient_end.copy()
                x = (bar_left + bar_right) / 2
                if j == path_index:
                    visible_left = max(bar_left, frame.get_x() - line_width / 2)
                    visible_right = min(bar_right, frame.get_x() + line_width / 2)
                    if visible_left < visible_right:
                        x = (visible_left + visible_right) / 2
                offset = np.array([clip(x, x_min - 1, x_max + 1), top, 0])
                for sm, base_points in zip(label.get_family(), layer.label_points[j]):
                    if len(base_points) > 0:
                        sm.set_points(base_points + offset)
                fits = layer.label_widths[j] < 0.8 * (bar_right - bar_left)
                label.set_fill(opacity=fade * float(fits))

        def update_word_layers(group):
            value = get_diagram_value()
            key = (
                value, view_center.get_value(), view_log_width.get_value(),
                get_line_y(), layers_fade.get_value(), frame.get_x(), frame.get_width(),
            )
            if key == group.last_key:
                return
            group.last_key = key
            prefix, ranges, indices = decode(value, len(word))
            layers = [get_layer(prefix[:i]) for i in range(len(word))]
            if list(group.submobjects) != layers:
                group.set_submobjects(layers)
            fade = layers_fade.get_value()
            for i, layer in enumerate(layers):
                style_layer(layer, indices[i])
                if layer.applied_fade != fade:
                    for bar, (fill_opacity, stroke_opacity) in zip(layer.bars, layer.base_opacities):
                        bar.set_fill(opacity=fill_opacity * fade)
                        bar.set_stroke(opacity=stroke_opacity * fade)
                    layer.applied_fade = fade
                top = get_line_y() - first_layer_gap - i * (layer_height + layer_buff)
                place_layer(
                    layer,
                    value_to_x(ranges[i][0]),
                    value_to_x(ranges[i][1]),
                    top,
                    indices[i],
                    fade,
                )

        word_layer_group.add_updater(update_word_layers)
        word_layer_group.update()

        max_bits = n_shown_bits
        zeros = Text("." + "0" * max_bits, font="Consolas", font_size=48)
        ones = Text("." + "1" * max_bits, font="Consolas", font_size=48)
        ones.move_to(zeros)
        char_advance = zeros[2].get_x() - zeros[1].get_x()
        bit_slots = VGroup(VGroup(zero, one) for zero, one in zip(zeros[1:], ones[1:]))
        bit_string = VGroup(zeros[0], bit_slots)
        bit_string.match_x(k_label)
        bit_string.set_y(bits_bottom_y + bit_string.get_height() / 2)
        bit_fade = ValueTracker(0)
        bit_string.last_key = None

        def update_bit_string(group):
            bits = get_shown_bits()
            fade = bit_fade.get_value()
            key = (bits, fade)
            if key == group.last_key:
                return
            group.last_key = key
            group.match_x(k_label)
            group[0].set_opacity(fade)
            n_significant = bits.rfind("1") + 1
            for i, (bit, slot) in enumerate(zip(bits, group[1])):
                slot.set_color(WHITE if i < n_significant else GREY_D)
                slot[0].set_opacity(fade if bit == "0" else 0)
                slot[1].set_opacity(fade if bit == "1" else 0)

        bit_string.add_updater(update_bit_string)
        bit_string.update()

        word_cache = dict()
        word_fade = ValueTracker(0)
        word_tracker = VGroup()
        word_tracker.last_key = None
        word_tracker.frozen_value = None

        def get_word_text(string):
            if string not in word_cache:
                mob = Text(f"|{string.replace(' ', '_')}|").set_height(0.7)
                mob.next_to(bit_string, UP, buff=MED_SMALL_BUFF)
                mob.match_x(k_label)
                mob.remove(mob[0], mob[-1])
                word_cache[string] = mob
            return word_cache[string]

        def update_word_tracker(group):
            value = group.frozen_value if group.frozen_value is not None else get_arrow_value()
            key = (value, word_fade.get_value())
            if key == group.last_key:
                return
            group.last_key = key
            mob = get_word_text(decode(key[0], len(word))[0])
            if group.submobjects != [mob]:
                group.set_submobjects([mob])
            mob.set_opacity(key[1])

        word_tracker.add_updater(update_word_tracker)
        word_tracker.update()

        arrow_opacity = ValueTracker(0)
        value_arrow = Vector(DOWN, thickness=5)
        value_arrow.set_z_index(3)

        def update_value_arrow(arrow):
            arrow.move_to(n2p(get_arrow_value()), DOWN)
            arrow.set_opacity(arrow_opacity.get_value())

        value_arrow.add_updater(update_value_arrow)
        value_arrow.update()

        highlight_fade = ValueTracker(0)
        highlight = Line(LEFT, RIGHT).set_stroke(YELLOW, 5)

        def update_highlight(line):
            bits = get_code_bits()
            low = int(bits, 2) / 2**len(bits)
            high = low + 2.0**-len(bits)
            left = clip(value_to_x(low), frame.get_left()[0] - 1, frame.get_right()[0] + 1)
            right = clip(value_to_x(high), frame.get_left()[0] - 1, frame.get_right()[0] + 1)
            line.put_start_and_end_on(
                np.array([left, get_line_y(), 0]),
                np.array([max(right, left + 1e-3), get_line_y(), 0]),
            )
            line.set_stroke(opacity=highlight_fade.get_value())

        highlight.add_updater(update_highlight)
        highlight.update()

        # Show things in terms of the diagram
        last_low, last_high = word_ranges[-2]
        self.add(word_layer_group, highlight, value_arrow, bit_string, word_tracker)
        self.play(
            ceil_inequality.animate.scale(0.5).to_edge(DOWN, buff=0.4),
            word_fade.animate(time_span=(2, 3)).set_value(1),
            line_y_tracker.animate.set_value(new_line_y),
            goalpost_bottom.animate.set_value(new_goalpost_bottom),
            goalpost_top.animate.set_value(new_goalpost_top),
            target_low.animate.set_value(word_low),
            get_view_anim((last_low + last_high) / 2, math.log2(last_high - last_low), word_center),
            ruler_alpha.animate.set_value(1),
            layers_fade.animate.set_value(1),
            FadeOut(k_label, time_span=(0, 1.5)),
            width_fade.animate(time_span=(0, 1.5)).set_value(0),
            arrow_opacity.animate.set_value(1),
            highlight_fade.animate.set_value(1),
            tick_label_boost.animate(time_span=(0, 0.3)).set_value(1),
            bit_fade.animate(time_span=(2, 3)).set_value(1),
            run_time=3,
        )
        self.remove(width_brace)
        self.wait(2)

        # Show a luckier position
        scale = get_scale()
        half_width = p * scale / 2
        word_x = value_to_x(word_center)
        x_min = word_x - 6
        x_max = word_x - 2 * half_width
        desired_x = interpolate(word_x, x_min, 0.6)
        candidates = []
        for level in range(1, len(word_bits)):
            denom = 2**level
            for numer in range(math.ceil((view_center.get_value() + x_min / scale) * denom), denom + 1):
                x = value_to_x(numer / denom)
                if x > x_max:
                    break
                if numer % 2 == 1:
                    candidates.append((level, numer / denom, x))
        if not candidates:
            raise ValueError("No tick with fewer bits is visible to the left of the target")
        lucky_level, lucky_value, lucky_x = min(
            candidates,
            key=lambda c: (abs(c[2] - desired_x) > 1.5, c[0], abs(c[2] - desired_x)),
        )

        original_arrow = value_arrow.copy().clear_updaters().set_opacity(0.3)
        self.add(original_arrow)
        word_tracker.frozen_value = get_arrow_value()
        self.play(
            layers_fade.animate(run_time=0.7).set_value(0.2),
            highlight_fade.animate(run_time=0.7).set_value(0.2),
            word_fade.animate(run_time=0.7).set_value(0.2),
            target_low.animate(run_time=2.5).set_value(lucky_value - p / 2),
            goalpost_color.animate(run_time=1, time_span=(1.5, 2.5)).set_value(1)
        )
        self.wait()
        self.play(
            target_low.animate.set_value(word_low),
            goalpost_color.animate.set_value(0),
            layers_fade.animate.set_value(1),
            highlight_fade.animate.set_value(1),
            word_fade.animate.set_value(1),
            run_time=2,
        )
        word_tracker.frozen_value = None
        self.play(FadeOut(original_arrow))

        # Slide the number down to zero
        value_tracker.set_value(get_arrow_value())
        modes["bits"] = "value"
        modes["diagram"] = "arrow"
        zero_low, zero_high = decode(0, len(word))[1][-1]
        start_value = value_tracker.get_value()
        start_center = view_center.get_value()
        start_log_width = view_log_width.get_value()
        final_center = (zero_low + zero_high) / 2
        final_log_width = math.log2(zero_high - zero_low)
        zoom_out_time = 3
        slide_start = 1
        zoom_in_start = zoom_out_time
        follow_time = 2.5
        follow_start = 0.8
        finish_boost = 15
        finish_ease = 0.12
        settle_time = zoom_in_start + follow_time
        settle_log_width = max(final_log_width, min(6 - n_shown_bits, -2))
        extra_log_width = settle_log_width - final_log_width
        follow_speed = (2 - follow_start) * -settle_log_width / follow_time
        peak_speed = finish_boost * follow_speed
        if extra_log_width > 0.5:
            finish_time = extra_log_width / (
                (1 - finish_ease) * (follow_speed + peak_speed) / 2 + finish_ease * peak_speed / 2
            )
        else:
            settle_log_width = final_log_width
            finish_time = 0
        total_time = settle_time + finish_time

        def get_log_width(t):
            tau = t - zoom_in_start
            if finish_time == 0:
                return final_log_width * smooth(clip(tau / follow_time, 0, 1))
            if tau <= follow_time:
                x = clip(tau / follow_time, 0, 1)
                end_slope = 2 - follow_start
                ramp = 2 * (1 - 1 / end_slope)
                if ramp <= 0:
                    return settle_log_width * x
                if x < ramp:
                    u = x / ramp
                    return settle_log_width * end_slope * ramp * (u**3 - u**4 / 2)
                return settle_log_width * end_slope * (x - ramp / 2)
            u = min(tau - follow_time, finish_time)
            ramp_time = (1 - finish_ease) * finish_time
            if u <= ramp_time:
                dist = follow_speed * u + (peak_speed - follow_speed) * u**2 / (2 * ramp_time)
            else:
                r = u - ramp_time
                dist = (follow_speed + peak_speed) * ramp_time / 2 + peak_speed * r - peak_speed * r**2 / (2 * finish_ease * finish_time)
            return settle_log_width - dist

        dots_alpha = ValueTracker(0)
        bit_dots = VGroup(bit_string[0].copy().set_color(GREY_D) for _ in range(3))

        def update_bit_dots(dots):
            last_x = bit_slots[-1][0].get_x()
            for j, dot in enumerate(dots):
                dot.set_x(last_x + (0.75 + 0.45 * j) * char_advance)
                dot.match_y(bit_string[0])
                dot.set_opacity(clip(dots_alpha.get_value() - j, 0, 1))

        bit_dots.add_updater(update_bit_dots)
        bit_dots.update()
        self.add(bit_dots)

        def update_slide(mob, alpha):
            t = alpha * total_time
            if t < zoom_in_start:
                set_view(start_center, start_log_width, 0.5, 0, word_center, smooth(min(t / zoom_out_time, 1)))
                zoom_factor = 1
            else:
                set_view(0.5, 0, final_center, final_log_width, 0, get_log_width(t) / final_log_width)
                zoom_factor = 2**view_log_width.get_value()
            x = clip((t - slide_start) / (settle_time - slide_start), 0, 1)
            slide_alpha = 3 * x**2 - 2 * x**3
            value_tracker.set_value(start_value * (1 - slide_alpha) * zoom_factor)
            dots_alpha.set_value(3 * clip((t - settle_time + 3.5) / 3.5, 0, 1))
            other_label_fade.set_value(1 - smooth(clip((t - settle_time + 2) / 2, 0, 1)))

        self.play(
            UpdateFromAlphaFunc(view_center, update_slide, rate_func=linear, run_time=total_time),
            tick_label_boost.animate(time_span=(0, 2)).set_value(0),
            p_brace_fade.animate(time_span=(0, 1.5)).set_value(0),
            FadeOut(ceil_inequality, time_span=(0, 1.5)),
        )
        self.remove(p_brace)
        dots_alpha.set_value(3)
        other_label_fade.set_value(0)
        self.wait(2)


class ShowEncodingScene(InteractiveScene):
    text = "Euclid's Elements"
    highlight = None
    model_name = "qwen"
    qwen_start_text = "\n"
    nanogpt_start_text = " "
    nanogpt_alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,'!?"
    column_width = 6
    area_top = 2.5
    area_bottom = -2.6
    max_raw_font_size = 40
    max_bit_font_size = 36
    context_char_time = 0.04
    highlight_char_time = 0.15
    plain_char_time = 0.05

    def construct(self):
        # Compute the encoding
        text = " ".join(self.text.split())
        if self.model_name == "qwen":
            codes, char_ends = self.get_qwen_codes(text)
        elif self.model_name == "nanogpt":
            codes, char_ends = self.get_nanogpt_codes(text)
        else:
            raise ValueError(f"Unknown model {self.model_name!r}, use 'qwen' or 'nanogpt'")
        max_bits = max(len(code) for code in codes)

        highlight = " ".join(self.highlight.split()) if self.highlight else ""
        highlight_start = text.rfind(highlight) if highlight else -1
        has_highlight = highlight_start >= 0
        if has_highlight:
            highlight_token = max(t for t, end in enumerate(char_ends) if end <= highlight_start)
        else:
            highlight_start = len(text)
            highlight_token = 0

        # Fit the text to the columns
        area_height = self.area_top - self.area_bottom
        left_x = -self.column_width - 0.6
        right_x = 0.6

        raw_glyphs, char_glyphs = self.get_raw_glyphs(text, left_x, area_height, highlight_start)

        probe = Text("00", font="Consolas", font_size=48)
        advance_48 = probe[1].get_x() - probe[0].get_x()
        height_48 = probe[0].get_height()
        bit_size = self.max_bit_font_size
        while bit_size > 10:
            bit_advance = advance_48 * bit_size / 48
            bit_row_height = 1.6 * height_48 * bit_size / 48
            bits_per_row = int(self.column_width / bit_advance) - 1
            if math.ceil(max_bits / bits_per_row) * bit_row_height <= area_height:
                break
            bit_size -= 1
        bit_dot, bit_slots = self.get_bit_slots(
            max_bits, bit_size, right_x, bits_per_row, bit_row_height, height_48 * bit_size / 48,
        )

        # Set up the columns
        raw_title = Text("Raw", font_size=48)
        encoded_title = Text("Encoded", font_size=48)
        raw_title.move_to(np.array([left_x + self.column_width / 2, 3.25, 0]))
        encoded_title.move_to(np.array([right_x + self.column_width / 2, 3.25, 0]))
        divider = Line(FRAME_HEIGHT * 0.5 * UP, FRAME_HEIGHT * 0.5 * DOWN).set_stroke(GREY_B, 2)
        arrow = Arrow(0.45 * LEFT, 0.45 * RIGHT, buff=0, thickness=4)
        arrow.set_y((self.area_top + self.area_bottom) / 2)

        char_count = VGroup(Integer(0, font_size=36), Text("characters", font_size=36))
        bit_count = VGroup(Integer(0, font_size=36), Text("bits", font_size=36))
        counter_y = self.area_bottom - 0.6

        def update_counters(group):
            for counter in (char_count, bit_count):
                counter.arrange(RIGHT, buff=0.15, aligned_edge=DOWN)
            char_count.move_to(np.array([raw_title.get_x(), counter_y, 0]))
            bit_count.move_to(np.array([encoded_title.get_x(), counter_y, 0]))

        counters = VGroup(char_count, bit_count)
        counters.add_updater(update_counters)
        counters.update()

        self.add(raw_title, encoded_title, divider, arrow, counters)
        self.wait()

        # Type out the text
        n_tracker = ValueTracker(0)
        added_start = len(codes[highlight_token]) if has_highlight else None
        encoding_group = VGroup(raw_glyphs, bit_dot, bit_slots)
        encoding_group.add_updater(self.get_encoding_updater(
            n_tracker, codes, char_ends, char_glyphs, bit_dot, bit_slots, char_count, bit_count, added_start,
        ))
        encoding_group.update()
        self.add(encoding_group)

        if highlight_token > 0:
            self.play(
                n_tracker.animate.set_value(highlight_token),
                rate_func=linear,
                run_time=char_ends[highlight_token] * self.context_char_time,
            )
            self.wait()
        char_time = self.highlight_char_time if has_highlight else self.plain_char_time
        self.play(
            n_tracker.animate.set_value(len(codes) - 1),
            rate_func=linear,
            run_time=(len(text) - char_ends[highlight_token]) * char_time,
        )
        self.wait(2)

    def get_encoding_updater(self, n_tracker, codes, char_ends, char_glyphs, bit_dot, bit_slots, char_count, bit_count, added_start):
        state = {"n": None}

        def update_encoding(group):
            n = int(n_tracker.get_value() + 1e-6)
            if n == state["n"]:
                return
            state["n"] = n
            n_chars = char_ends[n]
            for i, glyph in enumerate(char_glyphs):
                if glyph is not None:
                    glyph.set_fill(opacity=float(i < n_chars))
            code = codes[n]
            bit_dot.set_fill(opacity=float(len(code) > 0))
            for i, (zero, one) in enumerate(bit_slots):
                bit = code[i] if i < len(code) else None
                color = YELLOW if added_start is not None and i >= added_start else WHITE
                zero.set_fill(color, opacity=float(bit == "0"))
                one.set_fill(color, opacity=float(bit == "1"))
            char_count[0].set_value(n_chars)
            bit_count[0].set_value(len(code))

        return update_encoding

    def get_code(self, low, high, n_bits):
        while True:
            scale = 2**n_bits
            numer = math.ceil(low * scale)
            if numer < high * scale:
                return (format(numer, "b").zfill(n_bits) if n_bits > 0 else ""), n_bits
            n_bits += 1

    def get_qwen_codes(self, text):
        import torch
        import _2026.cross_entropy.next_char as next_char

        next_char._ensure_qwen_loaded()
        tokenizer = next_char._qwen_tokenizer
        model = next_char._qwen_model
        start_ids = tokenizer.encode(self.qwen_start_text, add_special_tokens=False)
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        pieces = [tokenizer.decode([token_id]) for token_id in text_ids]
        if "".join(pieces) != text:
            raise ValueError("The tokens do not decode back to the text")
        char_ends = [0]
        for piece in pieces:
            char_ends.append(char_ends[-1] + len(piece))

        low = Fraction(0)
        width = Fraction(1)
        codes = [""]
        n_bits = 0
        for i, token_id in enumerate(text_ids):
            input_ids = torch.tensor([start_ids + text_ids[:i]])
            with torch.no_grad():
                logits = model(input_ids).logits[0, -1].double()
            probs = torch.softmax(logits, dim=-1)
            total = Fraction(float(probs.sum()))
            low += width * Fraction(float(probs[:token_id].sum())) / total
            width *= Fraction(float(probs[token_id])) / total
            code, n_bits = self.get_code(low, low + width, n_bits)
            codes.append(code)
        return codes, char_ends

    def get_nanogpt_codes(self, text):
        alphabet = self.nanogpt_alphabet
        missing = sorted(set(text) - set(alphabet))
        if missing:
            raise ValueError(f"Characters not in the nanoGPT alphabet: {missing}")

        low = Fraction(0)
        width = Fraction(1)
        codes = [""]
        n_bits = 0
        for i, char in enumerate(text):
            dist = [Fraction(float(x)) for x in get_next_char_distribution(self.nanogpt_start_text + text[:i], alphabet)]
            index = alphabet.index(char)
            if dist[index] == 0:
                raise ValueError(f"The model gives {char!r} zero probability, so check how next_char.py maps cases")
            total = sum(dist)
            low += width * sum(dist[:index]) / total
            width *= dist[index] / total
            code, n_bits = self.get_code(low, low + width, n_bits)
            codes.append(code)
        return codes, list(range(len(text) + 1))

    def get_raw_glyphs(self, text, left_x, area_height, highlight_start):
        source = Text(text, font_size=48)
        glyphs = list(source)
        words = text.split(" ")
        if len(glyphs) != sum(len(word) for word in words):
            raise ValueError("Glyph count does not match the text")

        word_glyphs = []
        word_starts = []
        index = 0
        char_index = 0
        for word in words:
            word_glyphs.append(glyphs[index:index + len(word)])
            word_starts.append(char_index)
            index += len(word)
            char_index += len(word) + 1
        word_lefts = [group[0].get_left()[0] for group in word_glyphs]
        word_widths = [group[-1].get_right()[0] - group[0].get_left()[0] for group in word_glyphs]
        gaps = [
            word_lefts[i + 1] - word_glyphs[i][-1].get_right()[0]
            for i in range(len(words) - 1)
        ]
        space_width = sum(gaps) / len(gaps) if gaps else 0
        line_height = 1.3 * source.get_height()

        def wrap(scale):
            lines = [[0]]
            width = word_widths[0] * scale
            for i in range(1, len(words)):
                new_width = width + (space_width + word_widths[i]) * scale
                if new_width > self.column_width:
                    lines.append([i])
                    width = word_widths[i] * scale
                else:
                    lines[-1].append(i)
                    width = new_width
            return lines

        font_size = self.max_raw_font_size
        while font_size > 12:
            scale = font_size / 48
            lines = wrap(scale)
            if len(lines) * line_height * scale <= area_height:
                break
            font_size -= 1
        scale = font_size / 48

        source.scale(scale, about_point=ORIGIN)
        word_lefts = [left * scale for left in word_lefts]
        word_widths = [width * scale for width in word_widths]
        top = source.get_top()[1]

        char_glyphs = [None] * len(text)
        for row, line in enumerate(lines):
            line_width = sum(word_widths[i] for i in line)
            gap = space_width * scale
            if row < len(lines) - 1 and len(line) > 1:
                gap = (self.column_width - line_width) / (len(line) - 1)
            x = left_x
            dy = self.area_top - top - row * line_height * scale
            for i in line:
                for j, glyph in enumerate(word_glyphs[i]):
                    glyph.shift((x - word_lefts[i]) * RIGHT + dy * UP)
                    color = YELLOW if word_starts[i] + j >= highlight_start else WHITE
                    glyph.set_fill(color, opacity=0)
                    char_glyphs[word_starts[i] + j] = glyph
                x += word_widths[i] + gap
        return VGroup(*glyphs), char_glyphs

    def get_bit_slots(self, max_bits, font_size, left_x, per_row, row_height, digit_height):
        zeros = Text("0" * max_bits, font="Consolas", font_size=font_size)
        ones = Text("1" * max_bits, font="Consolas", font_size=font_size)
        dot_source = Text(".0", font="Consolas", font_size=font_size)
        advance = self.column_width / (per_row + 1)
        top_y = self.area_top - digit_height
        dot = dot_source[0]
        dot.move_to(np.array([
            left_x + 0.5 * advance,
            top_y + dot_source[0].get_y() - dot_source[1].get_y(),
            0,
        ]))
        dot.set_fill(opacity=0)
        slots = VGroup()
        for i in range(max_bits):
            row, col = divmod(i, per_row)
            position = np.array([left_x + (col + 1.5) * advance, top_y - row * row_height, 0])
            zero = zeros[i].move_to(position)
            one = ones[i].move_to(position)
            slots.add(VGroup(zero, one))
        slots.set_fill(opacity=0)
        return dot, slots


# class EuclidAloneNanoGPT(ShowEncodingScene):
#     text = "Euclid's Elements"
#     highlight = "Euclid's Elements"
#     model_name = "nanogpt"


# class EuclidAloneQwen(ShowEncodingScene):
#     text = "Euclid's Elements"
#     highlight = "Euclid's Elements"
#     model_name = "qwen"


class EuclidInContextNanoGPT(ShowEncodingScene):
    text = """
        The concept of proof and its associated mathematical rigor began in
        ancient Greek mathematics, exemplified in Euclid's Elements
    """
    highlight = "Euclid's Elements"
    model_name = "nanogpt"


class EuclidInContextQwen(ShowEncodingScene):
    text = """
        The concept of proof and its associated mathematical rigor began in
        ancient Greek mathematics, exemplified in Euclid's Elements
    """
    highlight = "Euclid's Elements"
    model_name = "qwen"


class USConstitutionQwen(ShowEncodingScene):
    text = """
        We the People of the United States,
        in Order to form a more perfect Union,
        establish Justice, insure domestic Tranquility,
        provide for the common defence,
        promote the general Welfare,
        and secure the Blessings of Liberty to ourselves and our Posterity,
        do ordain and establish this Constitution for the United States of America.
    """
    highlight = """
        in Order to form a more perfect Union,
        establish Justice, insure domestic Tranquility,
        provide for the common defence,
        promote the general Welfare,
        and secure the Blessings of Liberty to ourselves and our Posterity,
        do ordain and establish this Constitution for the United States of America.
    """
    model_name = "qwen"