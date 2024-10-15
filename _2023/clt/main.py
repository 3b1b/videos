from __future__ import annotations

from manim_imports_ext import *

from _2023.convolutions2.continuous import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
    from manimlib.typing import ManimColor

EXP_DISTRIBUTION = [0.41, 0.25, 0.15, 0.10, 0.06, 0.03]
U_SHAPED_DISTRIBUTION = [0.3, 0.15, 0.05, 0.05, 0.15, 0.3]
STEEP_U_SHAPED_DISTRIBUTION = [0.4, 0.075, 0.025, 0.025, 0.075, 0.4]


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
    font_size=25,
    y_range=(0, 1, 0.2),
    max_value: int = 6,
    bar_colors=(BLUE, TEAL)
):
    axes = Axes((0, max_value), y_range, **axes_config)
    ndp = len(str(y_range[2])) - 2
    axes.y_axis.add_numbers(font_size=font_size, num_decimal_places=ndp)
    bars = ChartBars(axes, dist)
    bars.set_submobject_colors_by_gradient(*bar_colors)

    dice = VGroup()
    for n in range(1, max_value + 1):
        die = DieFace(n, **die_config)
        die.set_width(0.5 * axes.x_axis.get_unit_size())
        die.next_to(axes.c2p(n - 0.5, 0), DOWN, SMALL_BUFF)
        dice.add(die)

    result = VGroup(axes, bars, dice)
    result.axes = axes
    result.bars = bars
    result.dice = dice
    return result


def get_sample_markers(bars, samples):
    width = 0.25 * bars[0].get_width()
    tips = ArrowTip(angle=-90 * DEGREES).replicate(len(samples))
    tips.set_width(width)
    tips.set_height(0.5 * width, stretch=True)
    tips.set_fill(YELLOW, 1)
    tip_map = [VGroup() for b in bars]
    for sample, tip in zip(samples, tips):
        tip_map[sample - 1].add(tip)

    for value, group in enumerate(tip_map):
        group.arrange(UP, buff=0.5 * width)
        group.next_to(bars[value], UP, SMALL_BUFF)

    return tips


def get_dist(bars):
    result = np.array([bar.get_height() for bar in bars])
    return result / result.sum()


def get_mean_and_sd(dist, x_min=1):
    mean = sum(p * (n + x_min) for n, p in enumerate(dist))
    sd = math.sqrt(sum(p * (n + x_min - mean)**2 for n, p in enumerate(dist)))
    return mean, sd


def gauss_func(x, mu, sigma):
    pre_factor = 1 / (sigma * math.sqrt(2 * PI))
    return pre_factor * np.exp(-0.5 * ((x - mu) / sigma)**2)


# Composite distributions


class RandomDieRolls(InteractiveScene):
    def construct(self):
        n_runs = 30
        n_movements_per_run = 30

        faces = VGroup(*(
            DieFace(n, fill_color=BLUE_E, dot_color=WHITE)
            for n in range(1, 7)
        ))

        for x in range(n_runs):
            random_faces = VGroup(*(
                random.choice(faces).copy()
                for x in range(n_movements_per_run)
            ))
            for face in random_faces:
                face.shift(np.random.uniform(-0.1, 0.1, 3))
            self.play(ShowSubmobjectsOneByOne(random_faces, run_time=1))
            self.wait()
            self.remove(random_faces)


class DiceSumDistributions(InteractiveScene):
    distribution = [1 / 6] * 6
    max_plot_width = 10
    n_examples = 7
    values = list(range(1, 7))

    def construct(self):
        # Setup all plots
        plots = VGroup(*(
            self.get_sum_distribution_plot(
                n,
                y_max=(0.2 if n > 1 else 0.5)
            )
            for n in range(1, self.n_examples + 1)
        ))
        plots.scale(0.9)
        plots.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        plots.to_corner(UL)

        labels = VGroup(*(
            self.get_plot_label(plot, n, die_face=True)
            for plot, n in zip(plots, it.count(1))
        ))

        # n = 1 distribution
        plot1 = plots[0]
        plot1.save_state()
        plot1.set_height(4)
        plot1.center()

        dice = self.get_dice_axis_labels(plot1)

        axes, bars = plot1

        prob_labels = Tex(r"1 / 6").replicate(6)
        for label, bar in zip(prob_labels, bars):
            label.set_width(0.5 * bar.get_width())
            label.next_to(bar, UP)

        self.add(plot1, dice)
        self.play(Write(prob_labels))
        self.wait()

        axes.add(dice, prob_labels)
        plot1.generate_target()
        plot1.target.replace(plot1.saved_state)

        # Grid of dice to bar chart
        grid = Square().get_grid(6, 6, buff=0)
        grid.set_stroke(GREY, 2)
        grid.set_height(5.5)
        grid.to_corner(UR)

        plot2 = plots[1]
        axes, bars = plot2
        plot2.save_state()
        plot2.set_width(10)
        plot2.center().to_edge(DOWN)
        axes.y_axis.numbers.set_opacity(0)

        die_groups = [VGroup() for n in range(6)]
        columns = VGroup(*(VGroup() for n in range(11)))
        for n, square in enumerate(grid):
            i = n % 6
            j = n // 6
            die1 = dice[i].copy()
            die2 = dice[j].copy()
            die2.set_fill(RED_E)
            die2.dots.set_fill(WHITE)
            pair = VGroup(die1, die2)
            pair.arrange(RIGHT, buff=SMALL_BUFF)
            pair.move_to(square)
            square.dice = pair
            columns[i + j].add(square)
            die_groups[i].add(die1)
            die_groups[j].add(die2)

        self.play(
            MoveToTarget(plot1),
            LaggedStart(*(
                TransformFromCopy(VGroup(dice[n]), die_groups[n])
                for n in range(6)
            ), lag_ratio=0.1),
            Write(grid),
            run_time=3
        )
        self.wait()

        for square in grid:
            square.add(square.dice)

        column_targets = VGroup()
        for column in columns:
            column.target = column.generate_target()
            for square in column.target:
                square.stretch(0.5, 1)
                square.dice.stretch(2, 1)
            column.target.arrange(DOWN, buff=0)
            column_targets.add(column.target)
        column_targets.arrange(RIGHT, buff=0, aligned_edge=DOWN)
        column_targets.match_width(bars[1:])
        column_targets.move_to(bars[1:], DOWN)

        bars.match_height(column_targets, stretch=True, about_edge=DOWN)
        bars.set_opacity(0.25)

        axes = plot2[0]
        axes.save_state()
        axes.set_opacity(0)
        self.add(axes)
        for column in columns:
            ct = column.target
            self.play(column.animate.set_stroke(YELLOW, 2))
            self.play(
                Transform(column, ct, run_time=1, lag_ratio=0.001),
                Restore(axes),
            )
        self.add(plot2[1], columns)
        self.play(FadeIn(plot2[1]))
        self.add(plot2, columns)
        self.wait()

        # n = 2 prob labels
        prob_labels = VGroup()
        for column in columns:
            label = Tex("1 / 36")
            numer = label.make_number_changeable("1")
            numer.edge_to_fix = RIGHT
            numer.set_value(len(column))
            label.next_to(column, UP, SMALL_BUFF)
            label.set_width(0.6 * column.get_width())
            prob_labels.add(label)

        self.play(Write(prob_labels))
        self.wait()
        labelled_bars = VGroup(bars, columns, prob_labels)
        labelled_bars.save_state()
        highlights = [
            VGroup(columns[n - 2], prob_labels[n - 2]).copy()
            for n in [6, 10]
        ]
        last = VMobject()
        for highlight in highlights:
            self.play(
                FadeOut(last),
                FadeIn(highlight),
                labelled_bars.animate.set_opacity(0.075)
            )
            last = highlight
            self.wait()
        self.play(
            FadeOut(last),
            Restore(labelled_bars),
        )

        # Restore
        columns.generate_target()
        pre_bars = plot2.saved_state[1][1:]
        columns.target.replace(pre_bars, stretch=True)
        columns.target.set_opacity(0)
        prob_labels.target = prob_labels.generate_target()
        for bar, label in zip(pre_bars, prob_labels.target):
            label.set_width(0.6 * bar.get_width())
            label.next_to(bar, UP, 0.5 * SMALL_BUFF)

        self.play(
            Restore(plot2),
            MoveToTarget(columns, remover=True),
            MoveToTarget(prob_labels),
            FadeIn(labels[:2], lag_ratio=0.2),
        )
        axes.add(prob_labels)
        self.wait()

        # Pan through the rest
        frame = self.frame
        frame.target = frame.generate_target()
        for n in range(1, len(plots) - 1):
            frame.target.match_y(plots[n])
            min_width = VGroup(plots[n + 1], labels[n + 1]).get_width() + 1
            frame.target.set_width(
                max(FRAME_WIDTH, min_width),
                about_edge=LEFT
            )
            self.play(LaggedStart(
                MoveToTarget(frame, run_time=2),
                FadeTransform(
                    plots[n].copy().set_opacity(0),
                    plots[n + 1], run_time=2
                ),
                TransformMatchingShapes(labels[n].copy(), labels[n + 1]),
                lag_ratio=0.2
            ))
            self.wait()

        frame.target.set_height(FRAME_HEIGHT)
        frame.target.center()
        frame.target.set_height(plots.get_height() + 1, about_edge=UL)
        self.play(MoveToTarget(frame, run_time=2))

        # Flash through
        plot_groups = VGroup(*(
            VGroup(plot, label)
            for plot, label in zip(plots, labels)
        ))
        for pg in plot_groups:
            pg.save_state()

        for pg1 in plot_groups:
            anims = []
            for pg2 in plot_groups:
                if pg2 is pg1:
                    anims.append(Restore(pg2))
                else:
                    anims.append(pg2.animate.restore().fade(0.85))
            self.play(*anims)
        self.play(LaggedStartMap(Restore, plot_groups))

        # Comment on center of mass straying to the right?
        mean_labels = self.get_mean_labels(plots)

        self.play(frame.animate.set_height(4).move_to(plots[0], DOWN).shift(DOWN))
        self.play(Write(mean_labels[0], run_time=1))
        self.wait()

        prob_labels.set_opacity(0)
        for n in range(6):
            frame.target = frame.generate_target()
            frame.target.set_height(plots[:n + 2].get_height() + 2)
            frame.target.move_to(plots[n + 1], DL).shift(0.75 * DL)

            self.play(LaggedStart(
                MoveToTarget(frame),
                TransformFromCopy(*mean_labels[n:n + 2]),
                lag_ratio=0.5
            ))
            self.wait()

        # Comment on distributions becoming more spread out?
        sd_labels = self.get_sd_labels(plots)
        for label in sd_labels:
            label.save_state()
            label.stretch(0, 0)
            label.set_opacity(0)

        self.play(
            LaggedStartMap(Restore, sd_labels),
            LaggedStart(*(
                FadeOut(label[1])
                for label in mean_labels
            )),
            run_time=3
        )
        self.wait()
        self.play(*(
            FadeOut(label[2])
            for label in sd_labels
        ))
        self.wait()

        mean_lines = VGroup(*(ml[0] for ml in mean_labels))
        sd_lines = VGroup(*(sl[:2] for sl in sd_labels))
        self.add(mean_lines, sd_lines)

        # Realign
        all_axes = VGroup(*(plot[0] for plot in plots))
        all_bars = VGroup(*(plot[1] for plot in plots))

        self.play(FadeOut(all_axes))
        self.realign_distributions(all_bars, labels, mean_lines, sd_lines)
        self.wait()

        # Rescale
        self.rescale_distributions(all_bars, mean_lines, sd_lines)

        # Draw up against normal curve
        axes = Axes(
            (-3, 3), (0, 1, 0.25),
            width=0.45 * frame.get_width(),
            height=0.18 * frame.get_height(),
        )
        axes.next_to(frame.get_left(), RIGHT, buff=1)
        axes.x_axis.add_numbers()
        axes.y_axis.add_numbers(np.arange(0.25, 1.25, 0.25), num_decimal_places=2)
        graph = axes.get_graph(lambda x: np.exp(-x**2))
        graph.set_stroke(YELLOW, 3)

        graph_label = Tex(R"e^{-x^2}")
        graph_label.set_height(0.5 * graph.get_height())
        graph_label.move_to(graph, UL)

        self.play(
            labels.animate.shift(1.5 * RIGHT),
            all_bars.animate.shift(4.5 * RIGHT),
            FadeOut(mean_lines, 4.5 * RIGHT),
            FadeOut(sd_lines, 4.5 * RIGHT),
            FadeIn(axes),
            ShowCreation(graph, run_time=2),
            Write(graph_label)
        )
        self.wait()

        # Rescale normal curve
        alt_graph_label = Tex(R"{1 \over \sigma \sqrt{2 \pi}} e^{-x^2 / 2\sigma^2}")
        alt_graph_label[R"{1 \over \sigma \sqrt{2 \pi}}"].scale(0.75, about_edge=RIGHT)
        alt_graph_label.set_height(1.25 * graph_label.get_height())
        alt_graph_label.move_to(graph_label, DL)
        mu, sigma = self.get_mean_and_standard_deviation(self.distribution)
        alt_graph = axes.get_graph(
            lambda x: (1 / sigma / math.sqrt(2 * PI)) * math.exp(-x**2 / 2 * sigma**2)
        )
        alt_graph.match_style(graph)
        self.play(
            Transform(graph, alt_graph),
            TransformMatchingTex(graph_label, alt_graph_label),
            run_time=2
        )
        axes.generate_target()
        sf = 3.5
        axes.target.y_axis.stretch(sf, 1, about_edge=DOWN)
        for number in axes.target.y_axis.numbers:
            number.stretch(1 / sf, 1)
        self.play(
            MoveToTarget(axes),
            graph.animate.stretch(sf, 1, about_edge=DOWN),
        )
        self.wait()

        # Map bell curve over bars
        graph_copies = VGroup(*(
            graph.copy().replace(
                bars[n:],
                dim_to_match=1
            ).set_stroke(width=1.5)
            for n, bars in zip(it.count(1), all_bars[1:])
        ))
        self.play(
            LaggedStart(*(
                TransformFromCopy(graph, graph_copy)
                for graph_copy in graph_copies
            )),
            run_time=3
        )
        self.wait()

    def get_plot_label(self, plot, n, die_face=False, height=0.35):
        # Test
        if die_face:
            die = DieFace(1).set_width(height)
            die.set_fill(BLUE_E)
            die.remove(*die.dots)
            terms = die.replicate(min(n, 5))
        else:
            term = Tex("X_0")
            term.make_number_changeable("0")
            terms = term.replicate(min(n, 5))
            for i, term in enumerate(terms):
                term[1].set_value(i + 1)


        if n > 5:
            terms.replace_submobject(2, Tex(R"\cdots"))
            if not die_face:
                terms[-2][1].set_value(n - 1)
                terms[-1][1].set_value(n)

        label = VGroup()
        for term in terms[:-1]:
            label.add(term)
            label.add(Tex("+"))
        label.add(terms[-1])
        label.arrange(RIGHT, buff=SMALL_BUFF)

        if n > 5 and die_face:
            brace = Brace(label, DOWN, SMALL_BUFF)
            count = brace.get_text(str(n), buff=SMALL_BUFF)
            label.add(brace, count)

        label.move_to(plot.get_corner(UR), UL)

        return label

    def get_sum_distribution_plot(self, n, **kwargs):
        return self.get_distribution_plot(
            self.get_sum_distribution(n),
            **kwargs
        )

    def get_distribution_plot(
        self,
        distribution,
        x_unit=0.5,
        height=1.5,
        bar_colors=(BLUE, TEAL),
        y_max=None,
    ):
        n = len(distribution)
        if y_max is not None:
            y_step = 0.1
        elif n == len(self.distribution):
            y_max = 0.5
            y_step = 0.1
        else:
            y_max = 0.05 * int(20 * max(distribution) + 2)
            y_step = 0.1 if y_max > 0.2 else 0.05

        axes = Axes(
            (0, n), (0, y_max, y_step),
            width=n * x_unit,
            height=height
        )
        axes.set_stroke(width=1)
        axes.center()
        axes.y_axis.add_numbers(
            num_decimal_places=len(str(y_step)) - 2,
            font_size=16
        )
        axes.x_axis.add_numbers(
            range(1, n + 1),
            num_decimal_places=0,
            font_size=int(axes.x_axis.get_unit_size() * 40),
            buff=0.1
        )
        axes.x_axis.numbers.shift(0.5 * axes.x_axis.get_unit_size() * LEFT)
        bars = ChartBars(axes, distribution)
        bars.set_submobject_colors_by_gradient(*bar_colors)
        bars.set_stroke(WHITE, 1)

        plot = VGroup(axes, bars)
        return plot

    def get_sum_distribution(self, n):
        dist0 = [0, *self.distribution]
        dist = np.array(dist0)
        for _ in range(n - 1):
            dist = np.convolve(dist, dist0)
        return dist[1:]

    def get_dice_axis_labels(self, plot):
        axes, bars = plot
        numbers = axes.x_axis.numbers
        die_faces = VGroup(*(
            DieFace(
                n + 1,
                fill_color=BLUE_E,
                dot_color=WHITE,
            ).replace(number, dim_to_match=1)
            for n, number in enumerate(numbers)
        ))
        for die in die_faces:
            die.dots.set_stroke(WHITE, 1)
        die_faces.set_stroke(width=1)

        return die_faces

    def get_mean_and_standard_deviation(self, distribution):
        prob_value_pairs = list(zip(distribution, it.count(1)))
        mu = sum(p * x for p, x in prob_value_pairs)
        sigma = math.sqrt(sum(p * (x - mu)**2 for p, x in prob_value_pairs))
        return mu, sigma

    def get_mean_labels(self, plots, color=PINK, num_decimal_places=1):
        mu = self.get_mean_and_standard_deviation(self.distribution)[0]
        mean_labels = VGroup()
        for n, plot in zip(it.count(1), plots):
            axes = plot[0]
            label = Tex(fR"{n}\mu = {np.round(n * mu, num_decimal_places)}")
            if n == 1:
                label[0].scale(0, about_edge=RIGHT)
            v_line = Line(*axes.y_axis.get_start_and_end())
            v_line.move_to(axes.c2p(n * mu - 0.5, 0), DOWN)
            label.shift(
                v_line.get_top() + MED_SMALL_BUFF * UP - label[:2].get_bottom()
            )
            mean_labels.add(VGroup(v_line, label))

        mean_labels.set_color(color)
        return mean_labels

    def get_sd_labels(self, plots, color=RED):
        mu, sigma = self.get_mean_and_standard_deviation(self.distribution)

        sd_labels = VGroup()
        for n, plot in zip(it.count(1), plots):
            axes = plot[0]
            v_lines = Line(*axes.y_axis.get_start_and_end()).replicate(2)
            v_lines[0].move_to(axes.c2p(n * mu - math.sqrt(n) * sigma - 0.5, 0), DOWN)
            v_lines[1].move_to(axes.c2p(n * mu + math.sqrt(n) * sigma - 0.5, 0), DOWN)

            arrows = VGroup(
                FillArrow(v_lines.get_center(), v_lines[0].get_center()),
                FillArrow(v_lines.get_center(), v_lines[1].get_center()),
            )
            for arrow in arrows:
                arrow.scale(0.75)
            arrows.move_to(v_lines)

            if n == 1:
                sigma_label = Tex(fR"\sigma")
            else:
                sigma_label = Tex(fR"\sqrt{n} \cdot \sigma")
            sigma_label.scale(0.8)
            sigma_label.next_to(arrows[1], UP, SMALL_BUFF)

            sd_labels.add(VGroup(
                v_lines,
                arrows,
                sigma_label,
            ))
        sd_labels.set_color(color)

        return sd_labels

    def realign_distributions(self, all_bars, labels, mean_lines, sd_lines):
        frame = self.frame

        bar_groups = VGroup()
        for bars, mean_line, sd_line in zip(all_bars, mean_lines, sd_lines):
            bar_group = VGroup(bars, mean_line, sd_line)
            bar_group.target = bar_group.generate_target()
            bar_group.target.shift(
                (frame.get_center()[0] - mean_line.get_x()) * RIGHT
            )
            bar_groups.add(bar_group)

        labels.target = labels.generate_target()
        for label, bars in zip(labels.target, all_bars):
            label.match_y(bars)
            label.set_x(0)
        labels.target.set_x(frame.get_right()[0] - 1, RIGHT)

        self.play(
            MoveToTarget(labels),
            LaggedStartMap(MoveToTarget, bar_groups, lag_ratio=0.01),
            run_time=2
        )

    def rescale_distributions(self, all_bars, mean_lines, sd_lines):
        arrows = VGroup(*(
            self.get_rescaling_arrows(bars, n)
            for n, bars in zip(it.count(2), all_bars[1:])
        ))

        self.play(Write(arrows, lag_ratio=0.01))

        def get_factor(n):
            return math.sqrt(n)

        self.play(
            LaggedStart(*(
                bars.animate.stretch(
                    1 / get_factor(n), 0,
                    about_point=lines.get_center()
                ).stretch(get_factor(n), 1, about_edge=DOWN)
                for n, bars, lines in zip(it.count(2), all_bars[1:], sd_lines)
            )),
            LaggedStart(*(
                lines.animate.stretch(
                    1 / get_factor(n), 0,
                )
                for n, lines in zip(it.count(2), sd_lines[1:])
            )),
            Animation(mean_lines),
            LaggedStart(*(
                arrow.animate.shift(arrow.get_vector())
                for arrow_pair in arrows
                for arrow in arrow_pair
            )),
            run_time=3
        )
        self.wait()
        self.play(FadeOut(arrows))

    def get_center_of_mass_base(self, bars):
        dist = np.array([bar.get_height() for bar in bars])
        dist /= dist.sum()
        mu = sum(p * x for p, x in zip(dist, it.count()))
        return bars.get_corner(DL) + mu * bars[0].get_width() * RIGHT

    def get_rescaling_arrows(self, bars, n):
        dist = np.array([bar.get_height() for bar in bars])
        dist /= dist.sum()

        mu = sum(p * x for p, x in zip(dist, it.count()))
        sigma = math.sqrt(sum(p * (x - mu)**2 for p, x in zip(dist, it.count())))
        bar_width = bars[0].get_width()
        length = sigma * bar_width
        center = bars.get_corner(DL) + mu * bar_width * RIGHT
        center += bars.get_height() * UP / 2.0

        kw = dict(fill_color=RED)

        arrows = VGroup(*(
            FillArrow(ORIGIN, length * vect, **kw).move_to(
                center - 3.0 * sigma * bar_width * vect,
                -vect
            )
            for vect in [LEFT, RIGHT]
        ))
        labels = Tex(Rf"\sigma \sqrt{{{n}}}").replicate(2)
        for arrow, label in zip(arrows, labels):
            label.next_to(arrow, UP, buff=0)
            label.set_max_width(arrow.get_width())
            label[R"\sigma"].set_color(RED)
            arrow.add(label)
        return arrows

    def show_sample_sum(self, plot, distribution, n, low_plot=None):
        axes, bars = plot

        samples = np.random.choice(self.values, size=n, p=distribution)
        bar_highlights = VGroup(*(
            bars[sample - 1].copy()
            for sample in samples
        ))
        bar_highlights.set_fill(YELLOW, 1)

        if low_plot:
            low_bar_highlight = low_plot[1][sum(samples) - 1].copy()
            low_bar_highlight.set_fill(YELLOW, 1)
        else:
            low_bar_highlight = VMobject()

        tips = get_sample_markers(bars, samples)

        sum_expr = VGroup()
        for sample in samples:
            sum_expr.add(Integer(sample))
            sum_expr.add(Tex("+"))
        sum_expr.remove(sum_expr[-1])
        sum_expr.add(Tex("="))
        sum_expr.add(Integer(sum(samples), color=YELLOW))
        sum_expr.arrange(RIGHT, buff=0.15)

        sum_expr.scale(1.25)
        sum_expr.next_to(plot, RIGHT, buff=2.5)

        kw = dict(int_func=np.ceil)
        self.play(
            ShowIncreasingSubsets(tips, **kw),
            ShowSubmobjectsOneByOne(bar_highlights, remover=True, **kw),
            ShowIncreasingSubsets(sum_expr[0:-1:2], **kw),
            ShowIncreasingSubsets(sum_expr[1::2], **kw),
            run_time=0.5
        )
        self.wait(0.1)
        self.add(sum_expr)
        self.add(low_bar_highlight)
        self.wait()
        self.play(LaggedStartMap(FadeOut, VGroup(tips, sum_expr, low_bar_highlight)), run_time=0.5)


class DiceSumDistributionsExp(DiceSumDistributions):
    distribution = EXP_DISTRIBUTION


class TransitionToSkewDistribution(DiceSumDistributions):
    def construct(self):
        # Setup all plots
        self.distribution = [1 / 6] * 6
        plot1 = self.get_sum_distribution_plot(1, y_max=0.5)
        self.distribution = EXP_DISTRIBUTION
        plot2 = self.get_sum_distribution_plot(1, y_max=0.5)

        for plot in plot1, plot2:
            plot.set_height(4)
            plot.center()
            plot[0].x_axis.numbers.set_opacity(0)
        axes, bars = plot1

        dice = self.get_dice_axis_labels(plot1)
        for die in dice:
            die.set_width(0.6, about_edge=UP)

        prob_labels = Tex(r"1 / 6").replicate(6)
        exp_prob_labels = VGroup(*(DecimalNumber(p) for p in self.distribution))
        for bars, labels in [(plot1[1], prob_labels), (plot2[1], exp_prob_labels)]:
            for bar, label in zip(bars, labels):
                label.set_width(0.5 * bar.get_width())
                label.next_to(bar, UP)

        self.add(plot1, dice)
        for bar in plot1[1]:
            bar.save_state()
            bar.stretch(0, 1, about_edge=DOWN)
        self.play(
            LaggedStartMap(FadeIn, prob_labels, shift=UP),
            LaggedStartMap(Restore, plot1[1]),
        )
        self.wait()
        self.play(
            ReplacementTransform(plot1, plot2, run_time=2),
            LaggedStart(*(
                FadeTransform(*labels)
                for labels in zip(prob_labels, exp_prob_labels)
            ), lag_ratio=0.05, run_time=2)
        )
        self.wait()
        self.play(
            VGroup(plot2, exp_prob_labels, dice).animate.set_height(2).to_corner(UL)
        )
        self.wait()

        axes.add(dice, prob_labels)


class ExpDistSumDistributions(DiceSumDistributions):
    n_examples = 7
    distribution = EXP_DISTRIBUTION

    def construct(self):
        # Setup all plots
        distributions = [
            [1 / 6] * 6,
            list(self.distribution),
        ]

        plots_list = VGroup()
        for dist in distributions:
            self.distribution = dist
            plots = VGroup(*(
                self.get_sum_distribution_plot(
                    n,
                    y_max=(0.2 if n > 1 else 0.4)
                )
                for n in range(1, self.n_examples + 1)
            ))
            plots.scale(0.9)
            plots.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
            plots.to_corner(UL)

            plots_list.add(plots)

        plots = plots_list[0]
        labels, die_labels = [
            VGroup(*(
                self.get_plot_label(plot, n, die_face)
                for plot, n in zip(plots, it.count(1))
            ))
            for die_face in [False, True]
        ]

        self.add(plots)
        self.add(die_labels)

        # Show new distribution
        self.play(
            Transform(plots, plots_list[1], run_time=3, lag_ratio=0.001),
            die_labels[0].animate.shift(0.25 * RIGHT),
        )
        self.wait()

        # Show pairs of samples
        self.play(
            FadeOut(plots[2:]),
            FadeOut(die_labels[2:]),
        )

        for _ in range(10):
            self.show_sample_sum(plots[0], distributions[1], 2, plots[1])

        self.play(ShowCreationThenFadeAround(VGroup(plots[1], die_labels[1])))
        self.wait()

        # Show triplet of samples
        plots[1].save_state()
        die_labels[1].save_state()
        self.play(
            FadeIn(plots[2], DOWN),
            FadeIn(die_labels[2], DOWN),
            plots[1].animate.set_opacity(0.2),
            die_labels[1].animate.set_opacity(0.2),
        )

        for _ in range(10):
            self.show_sample_sum(plots[0], distributions[1], 3, plots[2])
        self.play(Restore(plots[1]), Restore(die_labels[1]))

        # Replace labels
        labels[0].move_to(die_labels[0], LEFT)
        self.play(
            LaggedStartMap(FadeOut, die_labels, shift=UP, lag_ratio=0.25),
            LaggedStartMap(FadeIn, labels, shift=UP, lag_ratio=0.25),
        )
        self.wait()

        # Show more plots
        frame = self.frame
        self.add(plots)
        self.play(
            frame.animate.set_height(plots.get_height() + 2).move_to(plots, LEFT).shift(LEFT),
            run_time=4
        )
        self.wait()

        # Show means and standard deviation lines
        mu, sigma = self.get_mean_and_standard_deviation(self.distribution)
        mean_labels = self.get_mean_labels(plots, num_decimal_places=2)
        mean_lines = VGroup(*(ml[0] for ml in mean_labels))
        mu_labels = VGroup(*(ml[1] for ml in mean_labels))

        sd_labels = self.get_sd_labels(plots)
        sd_lines = VGroup(*(sdl[0] for sdl in sd_labels))
        sd_arrows = VGroup(*(sdl[1] for sdl in sd_labels))
        sigma_labels = VGroup(*(sdl[2] for sdl in sd_labels))

        all_axes = VGroup(*(plot[0] for plot in plots))
        arrows = VGroup(*(
            FillArrow(axes.c2p(0, 0), axes.c2p(n * mu, 0)).move_to(
                axes.y_axis.pfp(0.75), LEFT
            ).scale(0.7)
            for n, axes in zip(it.count(1), all_axes[1:])
        ))
        arrows.set_color(PINK)

        self.play(
            LaggedStartMap(GrowArrow, arrows, lag_ratio=0.2),
            LaggedStartMap(FadeIn, mean_lines, shift=RIGHT, lag_ratio=0.2),
        )
        self.wait()

        for line in sd_lines:
            line.save_state()
            line.stretch(0, 0)
            line.set_opacity(0)

        self.play(
            FadeOut(arrows),
            LaggedStartMap(Restore, sd_lines),
            LaggedStartMap(FadeIn, sd_arrows)
        )
        self.wait()

        # Quantify means
        frame.target = frame.generate_target()
        frame.target.set_height(5)
        frame.target.move_to(plots[0], DOWN)
        frame.target.shift(DOWN)
        frame_around_plot0 = frame.target.copy()

        self.play(
            MoveToTarget(frame),
            FadeOut(sd_lines, lag_ratio=0.1),
            FadeOut(sd_arrows, lag_ratio=0.1),
            run_time=5
        )
        self.play(Write(mu_labels[0]))
        self.wait()

        for n in range(6):
            frame.target = frame.generate_target()
            frame.target.set_height(plots[:n + 2].get_height() + 2)
            frame.target.move_to(plots[n + 1], DL).shift(0.75 * DL)

            self.play(LaggedStart(
                MoveToTarget(frame),
                TransformFromCopy(*mu_labels[n:n + 2]),
                lag_ratio=0.5
            ))
            self.wait()

        # Quantify standard deviation
        # sigma = 1.38, if you're curious
        VGroup(sd_arrows, sigma_labels).shift(0.25 * UP)
        for label, arrows in zip(sigma_labels, sd_arrows):
            label.set_max_width(arrows[1].get_width())

        self.play(
            LaggedStartMap(FadeIn, sd_lines, scale=2),
            LaggedStartMap(FadeIn, sd_arrows, scale=2),
            FadeOut(mu_labels, lag_ratio=0.01),
        )
        self.play(
            frame.animate.become(frame_around_plot0).set_anim_args(run_time=5),
            Write(sigma_labels[0]),
        )
        self.wait()

        for n in range(6):
            frame.target = frame.generate_target()
            frame.target.set_height(plots[:n + 2].get_height() + 2)
            frame.target.move_to(plots[n + 1], DL).shift(0.75 * DL)

            self.play(LaggedStart(
                MoveToTarget(frame),
                TransformFromCopy(*sigma_labels[n:n + 2]),
                lag_ratio=0.5
            ))
            self.wait()

        self.play(
            FadeOut(sigma_labels, lag_ratio=0.1),
            FadeOut(sd_arrows, lag_ratio=0.1),
        )
        self.add(mean_lines, sd_lines)

        # Realign and rescale
        all_axes = VGroup(*(plot[0] for plot in plots))
        all_bars = VGroup(*(plot[1] for plot in plots))
        all_bars[1].stretch(0.8, 1, about_edge=DOWN)

        self.play(FadeOut(all_axes))
        self.realign_distributions(all_bars, labels, mean_lines, sd_lines)

        big_mean_line = Line(mean_lines.get_top(), mean_lines.get_bottom())
        big_mean_line.set_stroke(PINK, 5)
        big_mean_line.insert_n_curves(20)
        self.play(VShowPassingFlash(big_mean_line, run_time=2, time_width=2.0))

        self.wait()
        self.rescale_distributions(all_bars, mean_lines, sd_lines)

        big_sd_lines = Line(UP, DOWN).replicate(2)
        big_sd_lines.arrange(RIGHT)
        big_sd_lines.replace(sd_lines, stretch=True)
        big_sd_lines.set_stroke(RED, 5)
        big_sd_lines.insert_n_curves(20)
        self.play(*(
            VShowPassingFlash(line, run_time=2, time_width=2.0)
            for line in big_sd_lines
        ))

        self.wait()
        self.play(
            FadeOut(mean_lines),
            FadeOut(sd_lines),
            lag_ratio=0.1
        )

        # Show bell curves
        axes = Axes(
            (-5, 5), (0, 1, 0.25),
            width=0.45 * frame.get_width(),
            height=0.5 * frame.get_height(),
        )
        axes.next_to(frame.get_left(), RIGHT, buff=1)
        axes.x_axis.add_numbers(font_size=36)
        axes.y_axis.add_numbers(np.arange(0.25, 1.25, 0.25), font_size=36, num_decimal_places=2)
        graph = axes.get_graph(lambda x: np.exp(-0.5 * x**2) / math.sqrt(2 * PI))
        graph.set_stroke(YELLOW, 3)

        graph_label = Tex(R"{1 \over \sqrt{2\pi}} e^{-{1 \over 2}x^2}")
        graph_label.set_height(0.8 * graph.get_height())
        graph_label.move_to(graph.get_corner(UL), DL)

        box = SurroundingRectangle(graph_label)
        box.set_stroke(YELLOW, 1)
        words = Text("We'll unpack this\n in a moment")
        words.match_width(box)
        words.next_to(box, UP, buff=LARGE_BUFF)

        bar_shift = 5 * RIGHT

        self.play(LaggedStart(
            FadeIn(axes),
            all_bars.animate.shift(bar_shift),
            ShowCreation(graph),
            Write(graph_label),
            lag_ratio=0.25
        ))
        self.wait()
        self.play(
            Write(words),
            ShowCreation(box)
        )
        self.wait()
        self.play(FadeOut(words), FadeOut(box))

        mean_lines.shift(bar_shift)
        sd_lines.shift(bar_shift)

        # Map bell curve over plots
        target_curves = VGroup()
        for bars, mean_line in zip(all_bars, mean_lines):
            curve = graph.copy()
            curve.replace(bars, dim_to_match=1)
            curve.set_width(sd_lines[0].get_width() * 4, stretch=True)
            curve.match_x(mean_line)
            target_curves.add(curve)

        self.play(LaggedStart(*(
            TransformFromCopy(graph, curve)
            for curve in target_curves
        )))
        self.wait()


class UDistSumDistributions(ExpDistSumDistributions):
    distribution = U_SHAPED_DISTRIBUTION


# Mean and standard deviation

class MeanAndStandardDeviation(InteractiveScene):
    def construct(self):
        # Setup axes
        dist = np.array([3, 2, 1, 0.5, 1, 0.75])
        dist /= dist.sum()
        original_dist = dist

        die_config = dict(
            fill_color=BLUE_E,
            dot_color=WHITE,
        )
        chart = get_die_distribution_chart(
            dist,
            y_range=(0, 0.5, 0.1),
            die_config=die_config
        )
        chart.to_edge(LEFT)
        self.add(*chart)

        axes, bars, die_labels = chart
        assert(isinstance(axes, Axes))

        # Functions
        def set_dist(dist, **kwargs):
            new_bars = ChartBars(axes, dist)
            new_bars.match_style(bars)
            self.play(Transform(bars, new_bars, **kwargs))

        def get_dist():
            result = np.array([bar.get_height() for bar in bars])
            return result / result.sum()

        def get_mu():
            dist = get_dist()
            return sum(p * (n + 1) for n, p in enumerate(dist))

        def get_sigma():
            dist = get_dist()
            mu = get_mu()
            return math.sqrt(sum(p * (n + 1 - mu)**2 for n, p in enumerate(dist)))

        # Mean line
        tex_kw = dict(
            t2c={R"\mu": PINK, R"\sigma": RED},
            font_size=48
        )

        mean_line = Line(axes.c2p(0, 0), axes.c2p(0, 0.5))
        mean_line.set_stroke(PINK, 3)
        mean_line.add_updater(lambda m: m.move_to(axes.c2p(get_mu() - 0.5, 0), DOWN))
        mu_label = Tex(R"\mu = 1.00", **tex_kw)
        mu_number = mu_label.make_number_changeable("1.00")
        mu_number.add_updater(lambda m: m.set_value(get_mu()))
        mu_number.scale(0.75, about_edge=LEFT)
        mu_label.add_updater(lambda m: m.next_to(mean_line, UP))

        mean_name = Text("Mean")
        mean_name.next_to(mu_label, UL, buff=1.0)
        mean_name.shift_onto_screen()
        mean_arrow = Arrow(mean_name, mu_label)
        mean_arrow.add_updater(lambda m: m.put_start_and_end_on(
            mean_name.get_bottom(), mu_label.get_corner(UL)
        ).scale(0.7))

        self.play(
            ShowCreation(mean_line),
            VFadeIn(mu_label),
        )
        self.wait()
        self.play(
            ShowCreation(mean_arrow),
            FadeIn(mean_name, 0.5 * UL)
        )

        # Show a few means
        dists = [
            EXP_DISTRIBUTION[::-1],
            EXP_DISTRIBUTION,
            U_SHAPED_DISTRIBUTION,
            original_dist
        ]
        for dist in dists:
            set_dist(dist, run_time=2)
            self.wait()

        # Mean equation
        mean_eq = Tex(R"\mu = E[X] = \sum_{x} P(X = x) \cdot x =", **tex_kw)
        mean_eq[R"\mu"].scale(1.2, about_edge=RIGHT)
        rhs = VGroup()
        for n in range(1, 7):
            term = Tex(Rf"P(\square) \cdot {n} \, + ", **tex_kw)
            die = DieFace(n, **die_config)
            die.replace(term[R"\square"])
            term.replace_submobject(2, die)
            term.add(die)
            rhs.add(term)

        rhs.arrange(DOWN, buff=0.35)
        rhs[-1][-1].set_opacity(0)
        rhs.scale(0.75)
        mean_eq.next_to(rhs[0], LEFT, submobject_to_align=mean_eq[-1])
        VGroup(mean_eq, rhs).to_corner(UR)

        mean_eq[-1].set_opacity(0)
        self.play(Write(mean_eq))
        self.wait()

        # Highlight terms
        highlights = chart.bars.copy()
        highlights.set_stroke(YELLOW, 3)
        highlights.set_fill(opacity=0)

        for highlight, die, part in zip(highlights, chart.dice, rhs):
            self.play(
                ShowCreation(highlight),
                mean_eq[-1].animate.set_opacity(1),
            )
            self.play(
                TransformMatchingShapes(die.copy(), part),
                FadeOut(highlight),
                run_time=1
            )

        # Show a few other distributions again
        for dist in dists:
            set_dist(dist, run_time=2)
            self.wait()

        # Reorganize mean labels
        mu_label.target = mu_label.generate_target()
        mu_label.target[1:].scale(0, about_point=mu_label[0].get_right())
        mu_label.target.match_x(mean_line)
        mu_number.clear_updaters()
        self.play(
            MoveToTarget(mu_label),
            LaggedStart(
                FadeOut(mean_name),
                FadeOut(mean_arrow),
                FadeOut(rhs),
                FadeOut(mean_eq[-1]),
            )
        )
        mu_label.remove(*mu_label[1:])

        # Standard deviation
        sd_lines = mean_line.copy().replicate(2)
        sd_lines.set_stroke(RED, 3)
        sd_lines.clear_updaters()
        sd_lines[0].add_updater(lambda m: m.move_to(
            axes.c2p(get_mu() - get_sigma() - 0.5, 0), DOWN)
        )
        sd_lines[1].add_updater(lambda m: m.move_to(
            axes.c2p(get_mu() + get_sigma() - 0.5, 0), DOWN)
        )
        sd_arrows = VGroup(Vector(LEFT), Vector(RIGHT))
        sd_arrows.set_color(RED)
        sd_arrows.arrange(RIGHT, buff=0.35)
        sd_arrows.add_updater(lambda m: m.set_width(0.8 * sd_lines.get_width()))
        sd_arrows.add_updater(lambda m: m.move_to(sd_lines).shift(UP))

        sd_lines.save_state()
        sd_lines.stretch(0.5, 0).set_opacity(0)

        self.play(
            Restore(sd_lines),
            ShowCreation(sd_arrows, lag_ratio=0),
        )
        self.wait()

        # Contrast low and high variance
        var_dists = [
            np.array([1, 3, 12, 4, 2, 1], dtype='float'),
            np.array([10, 2, 1, 1, 2, 10], dtype='float'),
        ]

        for dist in var_dists:
            dist /= dist.sum()
            set_dist(dist, run_time=3)
            self.wait()

        # Variance equation
        var_eq = VGroup(
            Tex(R"\text{Var}(X) &= E[(X - \mu)^2]", **tex_kw),
            Tex(R"= \sum_x P(X = x) \cdot (x - \mu)^2", **tex_kw),
        )
        var_eq[1].next_to(var_eq[0]["="], DOWN, LARGE_BUFF, aligned_edge=LEFT)

        var_eq.next_to(mean_eq["E[X]"], DOWN, buff=1.5, aligned_edge=LEFT)

        variance_name = Text("Variance")
        variance_name.next_to(var_eq[0]["Var"], DOWN, buff=1.25).shift(0.5 * LEFT)
        variance_name.set_color(YELLOW)
        variance_arrow = Arrow(variance_name, var_eq[0]["Var"])
        variance_arrow.match_color(variance_name)

        self.play(
            FadeIn(var_eq[0], DOWN),
            mean_eq.animate.set_opacity(0.5),
        )
        self.play(
            Write(variance_name),
            ShowCreation(variance_arrow)
        )
        self.wait()
        self.play(Write(var_eq[1]))
        self.wait()

        partial_square_opacity_tracker = ValueTracker(0)

        # Show squares
        partial_square_opacity_tracker.set_value(0.5)
        new_dist = np.array([10, 2, 1, 3, 4, 13], dtype='float')
        new_dist /= new_dist.sum()
        sd_group = VGroup(sd_lines, sd_arrows)

        def get_squares(bars):
            result = VGroup()
            for bar in bars:
                prob = axes.y_axis.p2n(bars[0].get_top())
                line = Line(bar.get_bottom(), mean_line.get_bottom())
                square = Square(line.get_width())
                square.move_to(line, DOWN)
                square.match_y(bar.get_top(), DOWN)
                square.set_stroke(RED, 1)
                square.set_fill(RED, 0.2)
                p_square = square.copy()
                p_square.stretch(prob, 1, about_edge=DOWN)
                p_square.set_opacity(partial_square_opacity_tracker.get_value())
                result.add(VGroup(square, p_square))
            return result

        squares = get_squares(bars)
        labels = VGroup(*(Tex(Rf"P({n}) \cdot ({n} - \mu)^2", **tex_kw) for n in range(1, 7)))
        labels.scale(0.5)
        for label, square in zip(labels, squares):
            label.square = square
            label.add_updater(lambda m: m.set_width(0.7 * m.square.get_width()).next_to(m.square.get_bottom(), UP, SMALL_BUFF))

        label = labels[0]
        square = squares[0]
        square[1].set_fill(opacity=0)
        part1 = label[R"P(1) \cdot"]
        part2 = label[R"(1 - \mu)^2"]
        part2.save_state()
        part2.match_x(squares[0])
        self.play(
            FadeOut(sd_group),
            FadeIn(square, lag_ratio=0.8),
            FadeIn(part2),
        )
        self.wait()
        self.play(
            square[1].animate.set_fill(opacity=0.5),
            FadeIn(part1),
            Restore(part2),
        )
        self.add(label)

        # Show other squares
        last_group = VGroup(square, label)
        for new_label, new_square in zip(labels[1:], squares[1:]):
            new_group = VGroup(new_square, new_label)
            self.play(FadeOut(last_group), FadeIn(new_group))
            last_group = new_group
        self.play(
            FadeOut(last_group),
            FadeIn(square), FadeIn(label),
        )
        self.wait()

        squares = always_redraw(lambda: get_squares(bars[:1]))
        self.remove(square)
        self.add(squares, label)

        # Again, toggle between low and  high variance
        for dist in [var_dists[0], EXP_DISTRIBUTION, var_dists[1]]:
            set_dist(dist, run_time=3)
            self.wait(2)

        sd_group.update()
        self.play(FadeOut(squares), FadeOut(label), FadeIn(sd_group))
        self.wait()

        # Standard deviation
        sd_equation = Tex(R"\sigma = \sqrt{\text{Var}(X)}", **tex_kw)
        sd_equation.next_to(var_eq, DOWN, LARGE_BUFF, aligned_edge=LEFT)

        sd_name = Text("Standard deviation")
        sd_name.set_color(RED)
        sd_name.next_to(sd_equation, DOWN, LARGE_BUFF).shift_onto_screen()
        sd_name.shift(2 * LEFT)
        sd_arrow = Arrow(sd_name, sd_equation[R"\sigma"], buff=0.1)
        sd_arrow.match_color(sd_name)

        sigma_labels = Tex(R"\sigma", **tex_kw).replicate(2)
        for label, arrow in zip(sigma_labels, sd_arrows):
            label.arrow = arrow
            label.add_updater(lambda m: m.next_to(m.arrow, UP))

        self.play(
            FadeTransformPieces(variance_name, sd_name),
            ReplacementTransform(variance_arrow, sd_arrow),
            Write(sd_equation[:-len("Var(X)")]),
            TransformFromCopy(
                var_eq[0][R"\text{Var}(X)"],
                sd_equation[R"\text{Var}(X)"],
            ),
        )
        self.wait()
        self.play(*(
            TransformFromCopy(sd_equation[R"\sigma"][0], label)
            for label in sigma_labels
        ))
        self.wait()

        # Show a few distributions
        for dist in [*dists, *var_dists]:
            set_dist(dist, run_time=2)
            self.wait()


# Build up Gaussian


class BuildUpGaussian(InteractiveScene):
    def construct(self):
        # Axes and graph
        axes = self.get_axes()
        bell_graph = axes.get_graph(lambda x: gauss_func(x, 0, 0.5))
        bell_graph.set_stroke(YELLOW, 3)

        def get_graph(func, **kwargs):
            return axes.get_graph(func, **kwargs).match_style(bell_graph)

        self.add(axes)
        self.add(bell_graph)

        # Peel off from full Gaussian
        kw = dict(
            tex_to_color_map={
                R"\mu": PINK,
                R"\sigma": RED,
            },
            font_size=72
        )

        formulas = VGroup(*(
            Tex(tex, **kw)
            for tex in [
                "e^x",
                "e^{-x}",
                "e^{-x^2}",
                R"e^{-{1 \over 2} x^2}",
                R"{1 \over \sqrt{2\pi}} e^{-{1 \over 2} x^2}",
                R"{1 \over \sigma \sqrt{2\pi}} e^{-{1 \over 2} \left({x \over \sigma}\right)^2}",
                R"{1 \over \sigma \sqrt{2\pi}} e^{-{1 \over 2} \left({x - \mu \over \sigma}\right)^2}",
            ]
        ))
        saved_formulas = formulas.copy()

        morty = Mortimer()
        morty.to_edge(DOWN).shift(3 * RIGHT)

        formulas[-1].set_max_height(1.25)
        formulas[-1].to_edge(UP)
        self.add(formulas[-1])
        self.play(FlashAround(formulas[-1], run_time=2, time_width=1.5))
        self.wait()
        self.play(
            FadeOut(axes, 2 * DOWN),
            FadeOut(bell_graph, DOWN),
            VFadeIn(morty),
            morty.change("raise_right_hand"),
            formulas[-1].animate.next_to(morty.get_corner(UL), UP)
        )

        for form1, form2 in zip(formulas[::-1], formulas[-2::-1]):
            form2.next_to(morty.get_corner(UL), UP)
            self.play(TransformMatchingTex(form1, form2), run_time=1)

        self.play(
            formulas[0].animate.center().to_edge(UP),
            FadeIn(axes, 3 * UP),
            FadeOut(morty, DOWN)
        )
        self.wait()

        # Show decay
        self.remove(formulas[0])
        formulas = saved_formulas
        for form in formulas:
            form.center().to_edge(UP)
        self.add(formulas[0])

        def decay_anim(graph):
            dot = GlowDot(color=WHITE)
            dot.move_to(graph.pfp(0.5))
            sub_graph = VMobject()
            sub_graph.set_points(graph.get_points()[graph.get_num_points() // 2::])
            return AnimationGroup(
                MoveAlongPath(dot, sub_graph),
                UpdateFromAlphaFunc(
                    VectorizedPoint(),
                    lambda m, a: dot.set_radius(0.2 * clip(10 * there_and_back(a), 0, 1)),
                    remover=True
                ),
                run_time=1.5,
                remover=True
            )

        graph = get_graph(np.exp)

        self.play(ShowCreation(graph))
        self.wait()
        self.play(LaggedStart(
            TransformMatchingTex(formulas[0], formulas[1]),
            graph.animate.stretch(-1, 0).set_anim_args(run_time=2),
            lag_ratio=0.5
        ))
        graph.become(get_graph(lambda x: np.exp(-x)))
        self.play(LaggedStart(*(
            decay_anim(graph)
            for x in range(8)
        ), lag_ratio=0.15, run_time=4))

        # Decay in both directions
        abs_formula = Tex("e^{-|x|}", **kw)
        abs_formula.to_edge(UP)
        abs_graph = get_graph(
            lambda x: np.exp(-np.abs(x)),
        )
        abs_graph.make_jagged()
        smooth_graph = get_graph(lambda x: np.exp(-x**2))

        self.play(
            TransformMatchingTex(formulas[1], abs_formula),
            Transform(graph, abs_graph)
        )
        self.wait()
        self.play(Flash(axes.c2p(0, 1)))
        self.wait()
        self.play(
            TransformMatchingTex(abs_formula, formulas[2]),
            Transform(graph, smooth_graph)
        )
        self.wait()

        # Tweak the spread
        form_with_const = Tex("e^{-c x^2}", **kw)
        form_with_const["c"].set_color(RED)
        form_with_const.move_to(formulas[2])

        c_display, c_tracker = self.get_variable_display("c", RED, (0, 5))
        base_tracker = ValueTracker(math.exp(1))

        get_c = c_tracker.get_value
        get_base = base_tracker.get_value

        axes.bind_graph_to_func(
            graph,
            lambda x: np.exp(-get_c() * np.log(get_base()) * x**2)
        )

        self.play(
            TransformMatchingTex(formulas[2], form_with_const),
            FadeIn(c_display),
        )
        self.wait()
        for value in [5.0, 0.2, 1.0]:
            self.play(c_tracker.animate.set_value(value), run_time=2)
            self.wait()

        # Show effective base
        rhs = Tex(R"= \left(e^c\right)^{-x^2}", **kw)
        rhs["c"].set_color(RED)
        rhs.next_to(form_with_const)

        self.play(
            Write(rhs.shift(2 * LEFT)),
            form_with_const.animate.shift(2 * LEFT)
        )
        self.wait()

        alt_base_form = Tex(R"=(2.718)^{-x^2}", **kw)
        alt_base_form.next_to(rhs, RIGHT)
        base = alt_base_form.make_number_changeable("2.718")
        base_width = base.get_width()
        base.set_color(TEAL)
        base.edge_to_fix = ORIGIN
        base.add_updater(lambda m: m.set_value(
            np.exp(get_c() * np.log(get_base()))
        ))
        base.add_updater(lambda m: m.set_width(base_width))

        self.play(
            FadeIn(alt_base_form, RIGHT),
        )
        self.wait()
        for value in [2.2, 0.1, 1.0]:
            self.play(c_tracker.animate.set_value(value), run_time=2)
            self.wait()

        # Writing this family with "e" is a choice that we're making
        es = VGroup(form_with_const[0], rhs[2])
        pis, twos, threes = [
            VGroup(*(
                Tex(tex, **kw).move_to(e, DR)
                for e in es
            ))
            for tex in [R"\pi", "2", "3"]
        ]

        note = TexText("$e$ is not special here")
        note.next_to(es[0], DOWN, buff=1.5, aligned_edge=RIGHT)
        note.set_backstroke(width=5)
        arrow = Arrow(note, es[0])

        self.play(
            FlashAround(es[0], run_time=2),
            FadeIn(note),
            GrowArrow(arrow),
        )
        self.wait()
        last_mobs = es
        for mobs, value in [(pis, PI), (twos, 2), (threes, 3), (es, math.exp(1))]:
            base.suspend_updating()
            self.play(
                FadeOut(last_mobs, 0.25 * UP, lag_ratio=0.2),
                FadeIn(mobs, 0.25 * UP, lag_ratio=0.2),
                c_tracker.animate.set_value(1 / math.log(value)),
                base_tracker.animate.set_value(value),
            )
            base.resume_updating()
            self.wait()
            for c in [3, 0.5, 1]:
                self.play(c_tracker.animate.set_value(c), run_time=2)
                self.wait()

            last_mobs = mobs

        # Introduce sigma
        self.play(
            form_with_const.animate.set_x(0),
            FadeOut(rhs, RIGHT),
            FadeOut(alt_base_form, RIGHT),
            FadeOut(note, RIGHT),
            FadeOut(arrow, RIGHT),
        )
        self.play(FlashAround(form_with_const["c"], color=RED))

        form_with_sigma = Tex(
            R"e^{-{1 \over 2} \left({x / \sigma}\right)^2}",
            **kw
        )
        form_with_sigma.to_edge(UP)

        sigma_display, sigma_tracker = self.get_variable_display(R"\sigma", RED, (0, 3))
        sigma_tracker.set_value(math.sqrt(0.5))
        get_sigma = sigma_tracker.get_value
        sigma_display.update()
        graph.clear_updaters()
        axes.bind_graph_to_func(graph, lambda x: np.exp(
            -0.5 * (x / get_sigma())**2
        ))
        graph.update()

        self.play(
            TransformMatchingTex(form_with_const, form_with_sigma),
            FadeOut(c_display, UP),
            FadeIn(sigma_display, UP),
            run_time=1
        )
        self.wait()

        # Show standard deviation
        v_lines = Line(axes.c2p(0, 0), axes.c2p(0, 0.75 * axes.y_axis.x_max)).replicate(2)
        v_lines.set_stroke(RED, 2)
        v_lines[0].add_updater(lambda m: m.move_to(axes.c2p(-get_sigma(), 0), DOWN))
        v_lines[1].add_updater(lambda m: m.move_to(axes.c2p(get_sigma(), 0), DOWN))

        arrows = VGroup(Vector(LEFT), Vector(RIGHT))
        arrows.arrange(RIGHT, buff=0.25)
        arrows.match_color(v_lines)
        arrows.add_updater(lambda m: m.set_width(max(0.9 * v_lines.get_width(), 1e-5)))
        arrows.add_updater(lambda m: m.move_to(axes.c2p(0, 0.575 * axes.y_axis.x_max)))

        sigma_labels = Tex(R"\sigma").replicate(2)
        sigma_labels.set_color(RED)
        sigma_labels[0].add_updater(lambda m: m.next_to(arrows[0], UP, 0.15))
        sigma_labels[1].add_updater(lambda m: m.next_to(arrows[1], UP, 0.15))

        v_lines.suspend_updating()
        v_lines.save_state()
        v_lines.stretch(0.1, 0)
        v_lines.set_opacity(0)
        self.play(
            Restore(v_lines),
            ShowCreation(arrows, lag_ratio=0),
            FadeIn(sigma_labels)
        )
        v_lines.resume_updating()
        sd_group = VGroup(v_lines, arrows, sigma_labels)
        self.wait()
        for value in [2, math.sqrt(0.5)]:
            self.play(sigma_tracker.animate.set_value(value), run_time=3)
            self.wait()

        # Back to simple e^(-x^2)
        simple_form = Tex("e^{-x^2}", **kw)
        simple_form.to_edge(UP)

        self.remove(form_with_sigma)
        sd_group.save_state()
        self.play(
            FadeOut(sd_group),
            FadeOut(sigma_display),
            TransformMatchingTex(form_with_sigma.copy(), simple_form)
        )

        # We want the area to be 1
        ab_tracker = ValueTracker(np.array([-4, 4]))
        area = graph.copy()
        area.set_stroke(width=0)
        area.set_fill(YELLOW, 0.5)
        area.set_shading(0.5, 0.5)

        def update_area(area):
            a, b = ab_tracker.get_value()
            x_min, x_max = axes.x_range[:2]
            area.pointwise_become_partial(
                graph,
                (a - x_min) / (x_max - x_min),
                (b - x_min) / (x_max - x_min),
            )
            area.add_line_to(axes.c2p(b, 0))
            area.add_line_to(axes.c2p(a, 0))
            area.add_line_to(area.get_start())

        area.add_updater(update_area)

        note = Text("We want this area\nto be 1")
        note.next_to(axes.get_top(), DOWN)
        note.match_x(axes.c2p(2, 0))
        note.set_backstroke(width=5)
        note_arrow = Arrow(note.get_bottom() + LEFT, area.get_center())

        prob_label = Tex("p(a < x < b)")
        prob_label.move_to(axes.c2p(-2, 1.5))
        prob_arrow = Arrow(LEFT, RIGHT)
        prob_arrow.add_updater(lambda m: m.put_start_and_end_on(
            prob_label.get_bottom() + SMALL_BUFF * DOWN,
            area.get_center(),
        ))

        self.add(area, graph)
        self.play(
            FadeIn(area),
            Write(note, run_time=1),
            ShowCreation(note_arrow),
        )
        self.wait()
        self.play(
            Write(prob_label),
            ShowCreation(prob_arrow),
            ab_tracker.animate.set_value([-1, 0]),
            FadeOut(note),
            FadeOut(note_arrow),
        )
        self.wait()
        for value in [(-0.5, 0.5), (-1.5, -0.5), (0, 1), (-2, 1), (-4, 4)]:
            self.play(ab_tracker.animate.set_value(value), run_time=3)
            self.wait()
        self.play(
            FadeOut(prob_label),
            FadeOut(prob_arrow),
            FadeIn(note),
            FadeIn(note_arrow),
        )
        self.wait()

        self.remove(area)
        graph.set_fill(area.get_fill_color(), area.get_fill_opacity())

        # Normalize
        area_label = TexText(R"Area = $\sqrt{\pi}$", **kw)
        area_label.next_to(note_arrow.get_start(), UR, SMALL_BUFF)
        area_label.set_backstroke(width=5)
        graph.clear_updaters()

        normalized_form = Tex(R"{1 \over \sqrt{\pi}} e^{-x^2}", **kw)
        normalized_form[R"{1 \over \sqrt{\pi}}"].scale(0.8, about_edge=RIGHT)
        normalized_form.to_edge(UP, buff=MED_SMALL_BUFF)

        one = Tex("1", **kw)
        one.move_to(area_label[R"\sqrt{\pi}"], LEFT)
        one.align_to(area_label[0], DOWN)

        self.play(
            FadeOut(note, 0.5 * UP),
            FadeIn(area_label, 0.5 * UP),
        )
        self.wait()
        self.play(
            TransformMatchingTex(simple_form, normalized_form, run_time=1),
            FadeTransform(area_label[R"\sqrt{\pi}"], normalized_form[R"\sqrt{\pi}"]),
            graph.animate.stretch(1 / math.sqrt(PI), 1, about_edge=DOWN),
            note_arrow.animate.put_start_and_end_on(
                note_arrow.get_start(),
                note_arrow.get_end() + 0.5 * DOWN,
            )
        )
        self.play(Write(one))
        self.play(FlashAround(one))
        self.wait()

        # Show normalized form with sigma
        sigma_forms = [
            Tex(R"{1 \over \sqrt{\pi}} e^{-{1 \over 2}(x / \sigma)^2}", **kw),
            Tex(R"{1 \over \sigma \sqrt{2}} {1 \over \sqrt{\pi}} e^{-{1 \over 2}(x / \sigma)^2}", **kw),
            Tex(R"{1 \over \sigma \sqrt{2\pi}} e^{-{1 \over 2}(x / \sigma)^2}", **kw),
        ]
        for form in sigma_forms:
            e = form["e^"][0][0]
            form[:form.submobjects.index(e)].scale(0.7, about_edge=RIGHT)
            form.to_edge(UP, buff=MED_SMALL_BUFF)

        sqrt2_sigma = Tex(R"\sigma \sqrt{2}", **kw)
        sqrt2_sigma.move_to(one, LEFT)

        sigma_display.update()
        sd_group.restore().update()
        self.play(
            TransformMatchingTex(normalized_form, sigma_forms[0]),
            FadeIn(sigma_display),
            FadeIn(sd_group),
        )
        self.play(
            FadeOut(one, UP),
            FadeIn(sqrt2_sigma, UP),
            sigma_tracker.animate.set_value(1),
            graph.animate.stretch(math.sqrt(2), 0),
        )
        self.wait()
        self.play(
            TransformMatchingTex(*sigma_forms[:2], run_time=1),
            FadeTransform(sqrt2_sigma, sigma_forms[1][R"\sigma \sqrt{2}"][0]),
            graph.animate.stretch(1 / math.sqrt(2) / get_sigma(), 1, about_edge=DOWN),
            FadeIn(one),
        )
        self.wait()
        self.play(TransformMatchingTex(*sigma_forms[1:3]))
        self.play(FlashAround(sigma_forms[2][R"{1 \over \sigma \sqrt{2\pi}}"], run_time=3))
        self.wait()
        self.play(FlashAround(sigma_forms[2], run_time=3, time_width=1.0))

        axes.bind_graph_to_func(
            graph, lambda x: gauss_func(x, 0, get_sigma())
        )
        for value in [0.2, 0.5, math.sqrt(0.5)]:
            self.play(sigma_tracker.animate.set_value(value), run_time=3)
            self.wait()

        area_label[R"\sqrt{\pi}"].set_opacity(0)
        self.play(LaggedStartMap(FadeOut, VGroup(area_label, one, note_arrow)))
        self.wait()

        # Show standard form
        curr_form = sigma_forms[2].copy()
        self.remove(*sigma_forms)
        self.add(curr_form)
        standard_form = Tex(R"{1 \over \sqrt{2\pi}} e^{-{1 \over 2} x^2}", **kw)
        standard_form[R"{1 \over \sqrt{2\pi}}"].scale(0.7, about_edge=RIGHT)
        standard_form.move_to(curr_form)

        rect = SurroundingRectangle(standard_form)
        rect.set_stroke(BLUE, 2)
        std_words = Text("Standard\nnormal\ndistribution", alignment="LEFT")
        std_words.match_height(rect)
        std_words.scale(0.9)
        std_words.next_to(rect, RIGHT, buff=MED_LARGE_BUFF)
        std_words.set_color(BLUE)

        self.play(sigma_tracker.animate.set_value(1))
        self.play(TransformMatchingTex(curr_form, standard_form, run_time=1))

        one_labels = Integer(1).replicate(2)
        for ol, sl in zip(one_labels, sigma_labels):
            ol.match_style(sl)
            ol.move_to(sl)
            ol.shift(SMALL_BUFF * UP)

        self.play(
            FadeOut(sigma_labels, 0.1 * UP, lag_ratio=0.1),
            FadeIn(one_labels, 0.1 * UP, lag_ratio=0.1),
        )
        self.wait()
        self.play(
            ShowCreation(rect),
            Write(std_words)
        )
        self.wait()

        self.play(
            FadeOut(rect),
            FadeOut(std_words),
            FadeOut(one_labels, 0.2 * DOWN),
            FadeIn(sigma_labels, 0.2 * DOWN),
            TransformMatchingTex(standard_form, sigma_forms[2])
        )

        # Add the mean
        final_form = formulas[-1].copy()
        final_form[:final_form.submobjects.index(final_form["e^"][0][0])].scale(0.75, about_edge=RIGHT)
        final_form.to_edge(UP, buff=MED_SMALL_BUFF)

        mu_display, mu_tracker = self.get_variable_display(R"\mu", PINK, (-2, 2))
        mu_tracker.set_value(0)
        mu_display.update()
        get_mu = mu_tracker.get_value
        mu_display.to_corner(UR, buff=MED_SMALL_BUFF)

        mu_line = v_lines[0].copy()
        mu_line.clear_updaters()
        mu_line.set_stroke(PINK, 3)
        mu_line.add_updater(lambda m: m.move_to(axes.c2p(get_mu(), 0), DOWN))
        mu_label = Tex(R"\mu")
        mu_label.match_color(mu_line)
        mu_label.set_backstroke(width=10)
        mu_label.add_updater(lambda m: m.next_to(mu_line, UP))
        sd_group.add_updater(lambda m: m.match_x(mu_line))
        self.add(sd_group)

        axes.bind_graph_to_func(graph, lambda x: gauss_func(x, get_mu(), get_sigma()))

        self.add(mu_label)
        self.play(
            TransformMatchingTex(
                sigma_forms[2], final_form,
                matched_pairs=[
                    (sigma_forms[2][R"/ \sigma"], final_form[R"\over \sigma"][1]),
                ],
                match_animation=FadeTransform,
                mismatch_animation=FadeTransform,
            ),
            ShowCreation(mu_line),
            FadeIn(mu_display),
        )
        self.wait()

        for value in [2, -2, 1]:
            self.play(mu_tracker.animate.set_value(value), run_time=3)
            self.wait()
        for value in [0.5, 1.5, 0.8]:
            self.play(sigma_tracker.animate.set_value(value), run_time=2)
            self.wait()

        # Added animations for an opening scene
        mu_tracker.set_value(0)
        sigma_tracker.set_value(0.5)

        ms_mobs = VGroup(
            mu_display,
            sigma_display,
            mu_line, mu_label,
            sd_group
        )

        self.remove(ms_mobs)
        self.wait()
        self.play(LaggedStartMap(FadeIn, ms_mobs, lag_ratio=0.2))
        pairs = [(1, 1.5), (1, 1.0), (1.5, 1.0), (1, 0.7), (0, 1)]
        for mu, sigma in pairs:
            self.play(
                mu_tracker.animate.set_value(mu),
                sigma_tracker.animate.set_value(sigma),
                run_time=3
            )
            self.wait()

    def get_variable_display(self, name, color, value_range):
        eq = Tex(f"{name} = 1.00", t2c={name: color})
        number = eq.make_number_changeable("1.00")

        tracker = ValueTracker(1)
        get_value = tracker.get_value
        number.add_updater(lambda c: c.set_value(get_value()))

        number_line = NumberLine(
            value_range,
            width=1.25,
            tick_size=0.05
        )
        number_line.rotate(90 * DEGREES)
        number_line.next_to(eq, RIGHT, MED_LARGE_BUFF)
        number_line.add_numbers(font_size=14, direction=LEFT, buff=0.15)
        slider = ArrowTip(angle=PI).set_width(0.15)
        slider.set_color(color)
        slider.add_updater(lambda m: m.move_to(number_line.n2p(get_value()), LEFT))

        display = VGroup(eq, number_line, slider)
        display.to_corner(UL, buff=MED_SMALL_BUFF)
        return display, tracker

    def get_axes(
        self,
        x_range=(-4, 4),
        y_range=(-1.0, 2.0, 1.0),
    ):
        axes = NumberPlane(
            x_range, y_range,
            width=FRAME_WIDTH,
            height=6,
            background_line_style=dict(
                stroke_color=GREY_A,
                stroke_width=2,
                stroke_opacity=0.5
            )
        )
        axes.shift(BOTTOM - axes.c2p(0, y_range[0]))
        axes.x_axis.add_numbers()
        axes.y_axis.add_numbers(num_decimal_places=1)
        return axes


# Show limiting distributions


class LimitingDistributions(InteractiveScene):
    distribution = EXP_DISTRIBUTION
    x_min = 1
    bar_colors = (BLUE, TEAL)
    bar_opacity = 0.75
    y_range = (0, 0.5, 0.1)
    max_n = 50
    normal_y_range = (0, 0.5, 0.25)
    normal_axes_height = 2

    def construct(self):
        # Dividing line
        h_line = Line(LEFT, RIGHT)
        h_line.set_width(FRAME_WIDTH)
        h_line.set_stroke(GREY, 2)
        self.add(h_line)

        # Starting distribution
        dist = np.array(self.distribution)
        dist /= dist.sum()
        top_plot = self.get_top_distribution_plot(dist)
        top_label = TexText(R"Random variable \\ $X$", font_size=36)
        top_label[-1].scale(1.2, about_edge=UP).shift(0.25 * DOWN)
        top_label.move_to(top_plot).to_edge(LEFT, buff=0.25).shift(0.35 * UP)
        self.add(top_plot)
        self.add(top_label)

        # Lower plot
        N_tracker = ValueTracker(2)
        get_N = lambda: int(N_tracker.get_value())
        s_plot = self.get_sum_plot(dist, get_N(), top_plot)  # TODO
        self.add(s_plot)

        # Mean and s.d. labels
        bars = top_plot.bars
        mean, sd = get_mean_and_sd(get_dist(bars))
        top_mean_sd_labels = self.get_mean_sd_labels(
            R"\mu", R"\sigma", mean, sd
        )
        sum_mean_sd_labels = self.get_mean_sd_labels(
            R"\mu \cdot N", R"\sigma \cdot \sqrt{N}",
            mean * 10, sd * math.sqrt(10),
        )
        scaled_sum_mean_sd_labels = self.get_mean_sd_labels(R"", R"", 0, 1)

        sum_mean_sd_labels.next_to(h_line, DOWN)
        sum_mean_sd_labels.to_edge(RIGHT)
        scaled_sum_mean_sd_labels.next_to(h_line, DOWN)
        scaled_sum_mean_sd_labels.align_to(sum_mean_sd_labels, LEFT)
        top_mean_sd_labels.next_to(TOP, DOWN)
        top_mean_sd_labels.align_to(sum_mean_sd_labels, LEFT)

        mu_label, sigma_label = self.get_mu_sigma_annotations(top_plot)

        # Lower distribution labels
        sum_label = self.get_sum_label(get_N())
        self.add(sum_label)

        # Increase N
        for n in range(3, 11):
            new_s_plot = self.get_sum_plot(dist, n, top_plot)
            new_sum_label = self.get_sum_label(n, dots=False)
            N_tracker.set_value(n)
            self.play(
                ReplacementTransform(s_plot.bars, new_s_plot.bars[:len(s_plot.bars)]),
                GrowFromEdge(new_s_plot.bars[len(s_plot.bars):], DOWN),
                ReplacementTransform(s_plot.axes, new_s_plot.axes),
                FadeTransform(sum_label, new_sum_label),
                run_time=0.5,
            )
            s_plot = new_s_plot
            sum_label = new_sum_label
            self.wait(0.5)

        self.wait()

        # Show mean and standard deviation
        self.play(Write(mu_label, stroke_color=PINK), run_time=1)
        self.play(TransformMatchingTex(mu_label[1].copy(), top_mean_sd_labels[0]), run_time=1)
        self.wait()

        sigma_label.save_state()
        sigma_label.stretch(0, 0).set_opacity(0)
        self.play(Restore(sigma_label))
        self.play(TransformMatchingTex(sigma_label[2].copy(), top_mean_sd_labels[1]), run_time=1)
        self.wait()

        low_mu_label, low_sigma_label = self.get_mu_sigma_annotations(s_plot)
        low_lines = VGroup(low_mu_label[0], low_sigma_label[0])
        low_lines.shift((get_N() - 1.5)* s_plot.axes.x_axis.get_unit_size() * RIGHT)
        low_lines.stretch(0.5, 1, about_edge=DOWN)
        new_sum_label = self.get_sum_label(get_N())

        sum_mean_sd_labels.update()
        self.play(
            TransformMatchingTex(top_mean_sd_labels[0].copy(), sum_mean_sd_labels[0]),
            TransformMatchingTex(sum_label, new_sum_label),
            TransformFromCopy(mu_label[:1], low_mu_label[:1]),
            run_time=1.5
        )
        sum_label = new_sum_label
        self.wait()
        self.play(
            TransformMatchingTex(top_mean_sd_labels[1].copy(), sum_mean_sd_labels[1]),
            TransformFromCopy(sigma_label[:1], low_sigma_label[:1]),
            run_time=1.5
        )
        self.wait()

        # Write formula
        formula = Tex(
            R"{1 \over \sigma \sqrt{2\pi N}} e^{-{1 \over 2} \left({x - \mu N \over \sigma \sqrt{N}} \right)^2}",
            t2c={
                R"\mu": PINK,
                R"\sigma": RED,
                R"N": YELLOW,
            },
            font_size=36
        )
        N = get_N()
        mu, sigma = get_mean_and_sd(get_dist(s_plot.bars), x_min=N)
        graph = s_plot.axes.get_graph(lambda x: gauss_func(x, mu - 1, sigma))
        graph.set_stroke(WHITE, 3)
        formula.next_to(graph, UP).align_to(graph, LEFT).shift(0.5 * RIGHT)

        self.play(
            ShowCreation(graph),
            FadeIn(formula, UP),
        )
        self.wait()
        self.play(LaggedStartMap(
            FadeOut,
            VGroup(graph, *low_lines, formula),
            lag_ratio=0.5
        ))

        # Transition to rescaled version
        scaled_sum_label = self.get_sum_label(get_N(), scaled=True)
        ss_plot = self.get_scaled_sum_plot(dist, get_N())
        x_shift = 2 * RIGHT

        self.play(FlashAround(sum_label, time_width=1.5, run_time=2))
        self.play(LaggedStart(
            Transform(
                sum_label,
                scaled_sum_label[sum_label.get_string()][0].copy(),
                remover=True,
            ),
            Write(scaled_sum_label),
            s_plot.animate.shift(x_shift),
            lag_ratio=0.5
        ))
        self.wait()

        sub_part = scaled_sum_label[R"- \decimalmob \cdot \mu"]
        rect = SurroundingRectangle(sum_mean_sd_labels[0], color=RED)
        target_x = ss_plot.axes.c2p(0, 0)[0]
        self.play(ShowCreation(rect))
        self.play(
            FadeTransform(sum_mean_sd_labels[0].copy(), sub_part),
            rect.animate.set_points(SurroundingRectangle(sub_part).get_points()),
            s_plot.bars.animate.shift((target_x - low_lines[0].get_x() - x_shift) * RIGHT),
            s_plot.axes.animate.shift((target_x - s_plot.axes.c2p(0, 0)[0]) * RIGHT),
        )
        self.play(
            FadeOut(rect),
            FadeOut(sum_mean_sd_labels[0], RIGHT),
            FadeIn(scaled_sum_mean_sd_labels[0], RIGHT)
        )
        self.wait()

        div_part = scaled_sum_label[R"\sigma \cdot \sqrt{\decimalmob}"]
        rect = SurroundingRectangle(sum_mean_sd_labels[1], color=RED)
        self.play(ShowCreation(rect))
        self.play(
            FadeTransform(sum_mean_sd_labels[1].copy(), div_part),
            rect.animate.set_points(SurroundingRectangle(div_part).get_points()),
        )
        self.play(FadeOut(rect))
        self.wait()

        # Swap to new axes
        stretch_group = VGroup(ss_plot.axes.y_axis, ss_plot.bars)
        stretch_factor = s_plot.bars.get_height() / ss_plot.bars.get_height()
        stretch_group.stretch(stretch_factor, dim=1, about_edge=DOWN)

        self.play(
            FadeOut(s_plot.axes),
            FadeIn(ss_plot.axes),
            ReplacementTransform(s_plot.bars, ss_plot.bars),
            FadeOut(sum_label)
        )
        self.play(stretch_group.animate.stretch(1 / stretch_factor, dim=1, about_edge=DOWN))
        self.wait()

        # Show sd of 1
        sd_lines = Line(DOWN, UP).replicate(2)
        sd_lines[0].move_to(ss_plot.axes.c2p(1, 0), DOWN)
        sd_lines[1].move_to(ss_plot.axes.c2p(-1, 0), DOWN)
        sd_lines.set_stroke(RED, 3)
        sd_lines.save_state()
        sd_lines.stretch(0, 0).set_opacity(0)
        self.play(
            Restore(sd_lines),
            FadeOut(sum_mean_sd_labels[1], RIGHT),
            FadeIn(scaled_sum_mean_sd_labels[1], RIGHT)
        )
        self.wait()
        self.play(FadeOut(sd_lines))

        # Readable meaning
        full_screen_rect = FullScreenFadeRectangle()
        full_screen_rect.set_opacity(0.7)
        top_rect = full_screen_rect.copy().stretch(0.5, 1, about_edge=UP)
        top_rect.set_fill(BLACK, 0.7)
        words = Text("Highly readable meaning:")
        words.next_to(scaled_sum_label, UP, LARGE_BUFF, aligned_edge=LEFT)
        meaning = TexText(
            R"How many std devs away from the mean is $X_1 + \cdots + X_{10}$",
            t2c={"std devs": RED, "mean": PINK, },
            font_size=40
        )
        meaning.move_to(words, LEFT)

        self.add(full_screen_rect, scaled_sum_label)
        self.play(FadeIn(full_screen_rect))
        self.wait()
        self.play(
            FadeIn(top_rect), FadeIn(words)
        )
        self.wait()
        self.play(
            words.animate.shift(UP),
            Write(meaning)
        )
        self.wait()

        # Example bar
        bar = ss_plot.bars[9].copy()
        ss_plot.bars.save_state()

        sum_tex = Tex(R"0 = 19")
        die = DieFace(1, fill_color=BLUE_E)
        die.dots.set_opacity(0)
        dice = die.get_grid(2, 5)
        dice.set_height(0.75)
        dice.set_stroke(width=1)
        dice.move_to(sum_tex[0], RIGHT)
        sum_tex.replace_submobject(0, dice)
        sum_tex.move_to(bar).to_edge(LEFT)

        arrow = Arrow(sum_tex, bar, buff=0.1)

        ss_plot.bars.set_opacity(0.2)
        self.play(
            FadeOut(full_screen_rect),
            top_rect.animate.set_opacity(1),
            FadeIn(bar)
        )
        self.wait()
        self.play(
            FadeIn(sum_tex, lag_ratio=0.2, run_time=2),
            GrowArrow(arrow),
        )
        self.wait()
        self.play(
            arrow.animate.become(Vector(0.5 * UP).next_to(bar, DOWN, SMALL_BUFF))
        )
        self.wait()
        self.play(LaggedStart(
            FadeOut(top_rect),
            Restore(ss_plot.bars),
            FadeOut(bar),
            FadeOut(words),
            FadeOut(meaning),
            FadeOut(sum_tex),
            FadeOut(arrow),
        ))

        # Comment on meaning of bars
        words = Text("Probability = Area", font_size=36)
        words.next_to(bar, LEFT, buff=1.5, aligned_edge=UP)
        arrow = Arrow(words.get_right(), bar.get_center())

        y_axis_label = Text("Probability\ndensity", font_size=24)
        y_axis_label.set_color(GREY_B)
        y_axis_label.next_to(ss_plot.axes.c2p(0, 0.5), RIGHT, SMALL_BUFF)

        bar_range = ss_plot.bars[8:16].copy()

        self.play(
            ss_plot.bars.animate.set_opacity(0.2),
            FadeIn(bar)
        )
        self.play(FlashAround(scaled_sum_label))
        self.wait()
        self.play(Write(words), ShowCreation(arrow))
        self.wait()
        self.play(FadeIn(y_axis_label, 0.2 * UP))
        self.wait()
        self.play(
            FadeIn(bar_range, lag_ratio=0.2),
            FadeOut(bar)
        )
        self.wait()
        self.play(
            Restore(ss_plot.bars),
            FadeOut(bar_range)
        )
        self.wait()
        self.play(FadeOut(words), FadeOut(arrow), FadeOut(y_axis_label))

        # Roll back to 3
        for n in range(9, 2, -1):
            new_scaled_sum_label = self.get_sum_label(n, scaled=True)
            new_ss_plot = self.get_scaled_sum_plot(dist, n)
            N_tracker.set_value(n)

            self.remove(ss_plot.axes)
            self.add(new_ss_plot.axes)
            self.play(
                FadeTransform(scaled_sum_label, new_scaled_sum_label),
                FadeTransform(ss_plot.bars, new_ss_plot.bars),
                run_time=0.1
            )
            self.wait(0.4)

            scaled_sum_label = new_scaled_sum_label
            ss_plot = new_ss_plot

        # Mess around with the distribution
        alt_dists = [
            np.array(U_SHAPED_DISTRIBUTION),
            np.random.random(6),
            np.random.random(6),
            np.array(STEEP_U_SHAPED_DISTRIBUTION),
            np.random.random(6),
        ]
        for dist in alt_dists:
            dist /= dist.sum()

        words = Text("Change this\ndistribution")
        words.set_color(YELLOW)
        words.move_to(top_plot, UR)

        self.play(
            FadeOut(mu_label),
            FadeOut(sigma_label),
            Write(words),
        )
        self.change_distribution(
            alt_dists, top_plot, ss_plot,
            top_mean_sd_labels, get_N()
        )
        self.play(FadeOut(words))

        # Show limit
        for n in range(4, self.max_n + 1):
            N_tracker.set_value(n)
            new_ss_plot = self.get_scaled_sum_plot(self.distribution, n)
            new_scaled_sum_label = self.get_sum_label(n, scaled=True)

            if n < 5:
                rt = 0.5
                wt = 0.25
            elif n < 15:
                rt = 0.1
                wt = 0.1
            else:
                rt = 0
                wt = 0.1

            if rt > 0:
                self.play(
                    FadeOut(ss_plot.bars),
                    FadeIn(new_ss_plot.bars),
                    ReplacementTransform(ss_plot.axes, new_ss_plot.axes),
                    FadeTransform(scaled_sum_label, new_scaled_sum_label),
                    run_time=rt
                )
            else:
                self.remove(ss_plot, scaled_sum_label)
                self.add(new_ss_plot, new_scaled_sum_label)
            self.wait(wt)
            ss_plot = new_ss_plot
            scaled_sum_label = new_scaled_sum_label

            if n in [10, self.max_n]:
                # More switching
                self.wait()
                self.change_distribution(
                    [EXP_DISTRIBUTION, *alt_dists],
                    top_plot,
                    ss_plot,
                    top_mean_sd_labels, get_N(),
                )

        self.wait()

        # Show standard normal graph
        graph = ss_plot.axes.get_graph(
            lambda x: gauss_func(x, 0, 1),
            x_range=(-8, 8)
        )
        graph.set_stroke(YELLOW, 3)
        label = Tex(R"{1 \over \sqrt{2\pi}} e^{-x^2 / 2}")
        label[R"{1 \over \sqrt{2\pi}}"].scale(0.8, about_edge=RIGHT)
        label.next_to(graph.pfp(0.55), UR)

        self.play(
            ShowCreation(graph),
            Write(label),
            scaled_sum_mean_sd_labels.animate.to_edge(RIGHT)
        )
        self.play(FlashAround(label, run_time=2))
        self.wait()

        random_dists = [normalize(np.random.random(6))**2 for x in range(8)]
        self.change_distribution(
            [*random_dists, alt_dists[-1]],
            top_plot,
            ss_plot,
            top_mean_sd_labels, get_N(),
            run_time=1.5
        )
        self.wait()

        # Formal statement
        gen_scaled_sum_label = self.get_sum_label("N", scaled=True)
        gen_scaled_sum_label.replace(new_scaled_sum_label)
        rect = SurroundingRectangle(gen_scaled_sum_label)
        rect.set_stroke(BLUE, 2)
        statement = Tex(
            R"\lim_{N \to \infty} P(a < \text{This value} < b) = \int_a^b " + label.get_string() + "dx",
            font_size=36
        )
        statement["N"].set_color(YELLOW)
        statement.next_to(rect, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
        arrow = Arrow(statement["This value"], rect, buff=0.1)
        arrow.set_color(BLUE)

        bars = ss_plot.bars
        mid_i = np.argmax([bar.get_height() for bar in bars])
        to_fade = VGroup(*bars[:mid_i - 5], *bars[mid_i + 10:])

        shift_group = VGroup(ss_plot, graph)

        self.play(ShowCreation(rect))
        self.play(ReplacementTransform(new_scaled_sum_label, gen_scaled_sum_label))
        self.wait()
        self.play(
            FadeIn(statement[R"P(a < \text{This value} < b)"], 0.5 * DOWN)
        )
        self.play(
            statement["This value"].animate.set_color(BLUE),
            ShowCreation(arrow)
        )
        self.play(
            to_fade.animate.set_fill(opacity=0.1).set_stroke(width=0),
            graph.animate.set_stroke(width=1.5),
        )
        self.wait()
        self.play(Write(statement[R"\lim_{N \to \infty}"]))
        self.wait()
        self.play(LaggedStart(
            shift_group.animate.shift(3 * RIGHT),
            FadeTransform(label, statement[label.get_string()]),
            Write(statement[R" = \int_a^b "]),
            Write(statement[R"dx"]),
            lag_ratio=0.4,
            run_time=3,
        ))
        self.wait()

        self.play(LaggedStart(*(
            Indicate(bar, color=BLUE)
            for bar in bars[mid_i - 5:mid_i + 10]
        )))
        self.wait()

    def get_top_distribution_plot(
        self,
        dist,
        width=5,
        height=2.5,
        y_range=(0, 0.5, 0.1)
    ):
        # Axes and bars
        x_min = self.x_min

        axes = Axes(
            (x_min - 1, x_min + len(dist) - 1),
            y_range,
            width=width,
            height=height,
        )
        axes.x_axis.add_numbers(font_size=24, excluding=[0])
        axes.x_axis.numbers.shift(0.5 * axes.x_axis.get_unit_size() * LEFT)
        axes.y_axis.add_numbers(num_decimal_places=1, font_size=24, excluding=[0])

        bars = ChartBars(axes, dist)
        bars.set_submobject_colors_by_gradient(*self.bar_colors)
        bars.set_opacity(self.bar_opacity)

        plot = VGroup(axes, bars)
        plot.move_to(LEFT).to_edge(UP)

        plot.axes = axes
        plot.bars = bars

        return plot

    def get_mu_sigma_annotations(self, plot, mu_tex=R"\mu", sigma_tex=R"\sigma", min_height=2.0, x_min=1):
        bars = plot.bars
        axes = plot.axes
        dist = get_dist(bars)

        mu, sigma = get_mean_and_sd(dist, x_min)
        mean_line = Line(DOWN, UP)
        mean_line.set_height(max(bars.get_height() + 0.25, min_height))
        mean_line.set_stroke(PINK, 2)
        mean_line.move_to(axes.c2p(mu - 0.5, 0), DOWN)

        sd_lines = Line(DOWN, UP).replicate(2)
        sd_lines.arrange(RIGHT)
        sd_lines.set_stroke(RED, 2)
        sd_lines.match_height(mean_line)
        sd_lines[0].move_to(axes.c2p(mu - sigma - 0.5, 0), DOWN)
        sd_lines[1].move_to(axes.c2p(mu + sigma - 0.5, 0), DOWN)

        sd_arrows = VGroup(Vector(LEFT, stroke_width=3), Vector(RIGHT, stroke_width=3))
        sd_arrows.arrange(RIGHT, buff=0.25)
        sd_arrows.set_width(0.85 * sd_lines.get_width())
        sd_arrows.move_to(mean_line.pfp(0.75))
        sd_arrows.set_color(RED)

        mu_label = Tex(mu_tex, color=PINK, font_size=30)
        mu_label.next_to(mean_line, UP, SMALL_BUFF)
        sigma_label = Tex(sigma_tex, color=RED, font_size=30)
        sigma_label.set_max_width(sd_arrows[1].get_width() * 0.8)
        sigma_label.next_to(sd_arrows[1], UP, SMALL_BUFF)

        return (
            VGroup(mean_line, mu_label),
            VGroup(sd_lines, sd_arrows, sigma_label),
        )

    def get_sum_label(self, n: str | int, dots=True, scaled: bool = False):
        if isinstance(n, int) and n not in [2, 3]:
            if len(str(n)) == 1:
                n_str = "0"
            else:
                n_str = "1" + "0" * (len(str(n)) - 1)
        else:
            n_str = str(n)

        if n_str == "2":
            sum_tex = R"X_1 + X_2"
        elif n_str == "3":
            sum_tex = R"X_1 + X_2 + X_3"
        elif dots:
            sum_tex = Rf"X_1 + \cdots + X_{{{n_str}}}"
        else:
            sum_tex = " + ".join(f"X_{{{k}}}" for k in range(1, int(n) + 1))

        if scaled:
            sum_tex = "(" + sum_tex + ")" + Rf" - {n_str} \cdot \mu \over \sigma \cdot \sqrt{{{n_str}}}"

        sum_label = Tex(
            sum_tex,
            t2c={
                R"\mu": PINK,
                R"\sigma": RED,
                n_str: YELLOW,
            },
            font_size=40
        )

        if scaled:
            sum_label.scale(0.8)

        if isinstance(n, int) and n_str in sum_label.get_string():
            n_parts = sum_label.make_number_changeable(n_str, replace_all=True)
            for part in n_parts:
                part.set_value(n)

        sum_label.next_to(ORIGIN, DOWN)
        sum_label.to_edge(LEFT)
        return sum_label

    def get_scaled_sum_plot(self, dist, n):
        sum_dist = np.array(dist)
        for _ in range(n - 1):
            sum_dist = np.convolve(sum_dist, dist)

        axes = self.get_normal_plot_axes()

        mu, sigma = get_mean_and_sd(dist)
        x_min = n * self.x_min
        unscaled_xs = np.arange(x_min, x_min + len(sum_dist))
        xs = (unscaled_xs - n * mu) / (sigma * math.sqrt(n))

        bars = ChartBars(axes, sum_dist, xs=xs)
        bars.shift(0.5 * bars[0].get_width() * LEFT)
        bars.set_submobject_colors_by_gradient(*self.bar_colors)
        bars.set_opacity(self.bar_opacity)

        unit_area = axes.x_axis.get_unit_size() * axes.y_axis.get_unit_size()
        bar_area = sum(bar.get_width() * bar.get_height() for bar in bars)
        bars.stretch(unit_area / bar_area, 1, about_edge=DOWN)

        plot = VGroup(axes, bars)
        plot.set_stroke(background=True)

        plot.bars = bars
        plot.axes = axes

        return plot

    def get_normal_plot_axes(self):
        sum_axes = Axes(
            (-6, 6), self.normal_y_range,
            width=12, height=self.normal_axes_height
        )
        sum_axes.center().move_to(FRAME_HEIGHT * DOWN / 4)
        sum_axes.x_axis.add_numbers(font_size=16)
        sum_axes.y_axis.add_numbers(num_decimal_places=1, font_size=16, excluding=[0])
        return sum_axes

    def get_sum_plot(
        self,
        dist,
        n,
        top_plot,
        x_range=(0, 40),
        y_range=(0, 0.2, 0.1),
        x_num_range=(5, 40, 5),
        max_width=10,
    ):
        sum_dist = np.array(dist)
        for _ in range(n - 1):
            sum_dist = np.convolve(sum_dist, dist)

        x_min = n * self.x_min

        x_max = x_range[1]
        axes = Axes(
            x_range,
            y_range,
            width=top_plot.axes.x_axis.get_unit_size() * x_max,
            height=self.normal_axes_height,
            axis_config=dict(tick_size=0.05),
        )
        axes.x_axis.set_max_width(max_width, stretch=True, about_point=axes.c2p(0, 0))
        axes.shift(top_plot.axes.c2p(0, 0) - axes.c2p(0, 0))
        axes.to_edge(DOWN)
        axes.x_axis.add_numbers(np.arange(*x_num_range), font_size=16, excluding=[0])
        axes.x_axis.numbers.shift(0.5 * axes.x_axis.get_unit_size() * LEFT)
        axes.y_axis.add_numbers(
            num_decimal_places=len(str(y_range[2])) - 2,
            font_size=24,
            excluding=[0]
        )

        bars = ChartBars(axes, sum_dist, xs=range(x_min - 1, x_min + len(sum_dist) - 1))
        bars.set_submobject_colors_by_gradient(*self.bar_colors)
        bars.set_opacity(self.bar_opacity)

        plot = VGroup(axes, bars)
        plot.axes = axes
        plot.bars = bars

        return plot

    def get_mean_sd_labels(self, mu_tex, sigma_tex, mean, sd):
        label_kw = dict(
            t2c={
                R"\mu": PINK,
                R"\sigma": RED,
                R"N": YELLOW,
            },
            font_size=36
        )
        if len(mu_tex) > 0:
            mu_tex = "= " + mu_tex
        if len(sigma_tex) > 0:
            sigma_tex = "= " + sigma_tex

        labels = VGroup(
            Tex(Rf"\text{{mean}} {mu_tex} = 0.00", **label_kw),
            Tex(Rf"\text{{std dev}} {sigma_tex} = 0.00", **label_kw),
        )
        labels.arrange(DOWN, aligned_edge=LEFT)
        values = [mean, sd]
        colors = [PINK, RED]
        for label, value, color in zip(labels, values, colors):
            num = label.make_number_changeable("0.00")
            num.set_value(value)
            num.set_color(color)

        return labels

    def change_distribution(
        self,
        alt_dists,
        top_plot,
        ss_plot,
        top_mean_sd_labels,
        n,
        run_time=1,
    ):
        for dist in alt_dists:
            new_top_plot = self.get_top_distribution_plot(dist)
            new_top_plot.bars.align_to(top_plot.bars, DOWN)
            new_ss_plot = self.get_scaled_sum_plot(dist, n)
            mean, sd = get_mean_and_sd(dist)
            self.play(
                Transform(top_plot.bars, new_top_plot.bars),
                UpdateFromFunc(
                    ss_plot.bars,
                    lambda m: m.set_submobjects(self.get_scaled_sum_plot(
                        get_dist(top_plot.bars), n
                    ).bars)
                ),
                ChangeDecimalToValue(top_mean_sd_labels[0][-1], mean),
                ChangeDecimalToValue(top_mean_sd_labels[1][-1], sd),
             )
            self.wait()
            self.distribution = dist


class HowVarianceAdds(LimitingDistributions):
    def construct(self):
        # Define two distributions
        dist1 = EXP_DISTRIBUTION
        dist2 = np.convolve(dist1, U_SHAPED_DISTRIBUTION)

        plot1 = self.get_top_distribution_plot(dist1)
        plot2 = self.get_top_distribution_plot(dist2, y_range=(0, 0.301, 0.1))
        plot1.to_corner(UL)
        plot2.to_corner(UR)
        plot2.bars.set_submobject_colors_by_gradient(TEAL_E, GREEN_D)
        top_plots = VGroup(plot1, plot2)

        label_kw = dict(
            font_size=60,
            t2c={"X": BLUE, "Y": GREEN}
        )
        top_labels = VGroup(
            Tex("X", **label_kw),
            Tex("Y", **label_kw),
        )
        for plot, label in zip(top_plots, top_labels):
            label.move_to(plot, UR).shift(LEFT)

        h_line = Line(LEFT, RIGHT)
        h_line.set_width(FRAME_WIDTH)
        h_line.set_stroke(GREY_A, 1)

        # Define the sum
        sum_dist = np.convolve(dist1, dist2)
        sum_plot = self.get_top_distribution_plot(
            sum_dist,
            y_range=(0, 0.301, 0.1),
            width=10
        )
        sum_plot.to_corner(DL)
        sum_plot.bars.set_color_by_gradient(GREEN, YELLOW_E)

        sum_label = Tex("X + Y", **label_kw)
        sum_label.move_to(sum_plot, UL)
        sum_label.shift(RIGHT)

        # Define annotations
        top_annotations1 = VGroup(*(self.get_mu_sigma_annotations(plot1, "", "")))
        top_annotations2 = VGroup(*(self.get_mu_sigma_annotations(plot2, "", "")))
        sum_annotations = VGroup(*(self.get_mu_sigma_annotations(sum_plot, "", "")))
        for annotations in [top_annotations1, top_annotations2, sum_annotations]:
            annotations[0].set_color(GREY_C)
            annotations[1].set_color(GREY_A)
        top_annotations = VGroup(top_annotations1, top_annotations2)

        # Define variance formula
        var_form = Tex(
            R"\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)",
            **label_kw
        )
        var_form.scale(0.75)
        var_form.next_to(h_line, DOWN)
        var_form.to_edge(RIGHT, buff=1.0)
        note = TexText("(Assuming $X$ and $Y$ are independent!)", font_size=36)
        for key, color in label_kw["t2c"].items():
            note[key].set_color(color)
        note.next_to(var_form, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        # Add them all!
        self.add(h_line)
        self.play(
            LaggedStartMap(FadeIn, top_plots, lag_ratio=0.7, run_time=2),
            LaggedStartMap(FadeIn, top_labels, lag_ratio=0.7, run_time=2),
            LaggedStartMap(FadeIn, top_annotations, lag_ratio=0.7, run_time=2),
        )

        self.play(
            TransformFromCopy(plot1, sum_plot),
            TransformFromCopy(plot2, sum_plot.copy().fade(1)),
            TransformFromCopy(top_labels[0], sum_label[:1]),
            TransformFromCopy(top_labels[1], sum_label[1:]),
            TransformFromCopy(top_annotations1, sum_annotations),
            TransformFromCopy(top_annotations2, sum_annotations),
        )

        # Show variance formulas
        self.play(LaggedStart(
            TransformMatchingShapes(
                sum_label.copy(),
                var_form[R"\text{Var}(X + Y) = "]
            ),
            TransformMatchingShapes(
                top_labels[0].copy(),
                var_form[R"\text{Var}(X)"]
            ),
            TransformMatchingShapes(
                top_labels[1].copy(),
                var_form[R"+ \text{Var}(Y)"]
            ),
            FadeIn(note, 0.25 * DOWN),
            lag_ratio=0.25
        ))
        self.wait()

        # Sigma equation
        sigma_form = Tex(
            R"\sigma_{X + Y}^2 = \sigma_X^2 + \sigma_Y^2",
            **label_kw
        )
        sigma_form.scale(0.75)
        sigma_form.move_to(var_form, LEFT)

        self.play(LaggedStart(
            FadeIn(sigma_form, 0.5 * DOWN),
            var_form.animate.shift(DOWN),
            note.animate.shift(DOWN).set_opacity(0.7),
            lag_ratio=0.2,
            run_time=2
        ))
        self.wait()

        # Add many X_n
        plot1_group = VGroup(plot1, top_labels[0], top_annotations1)
        new_top_groups = plot1_group.replicate(4)
        for n, group in zip([1, 2, 3, "N"], new_top_groups):
            label = group[1]
            substr = Tex(str(n))
            substr.match_color(label)
            substr.set_height(label.get_height() * 0.5)
            substr.next_to(label.get_corner(DR), RIGHT, buff=0.05)
            label.add(substr)

        new_top_groups.scale(0.5)
        dots = Tex(R"\dots", font_size=90)
        arranger = VGroup(*new_top_groups[:3], dots, *new_top_groups[3:])
        arranger.arrange(RIGHT, buff=LARGE_BUFF)
        arranger.set_width(FRAME_WIDTH - 1)
        arranger.set_y(2.5)

        sum_dist = dist1
        for _ in range(6):
            sum_dist = np.convolve(dist1, sum_dist)

        new_sum_plot = self.get_top_distribution_plot(
            sum_dist,
            y_range=(0, 0.2, 0.1),
            width=10
        )
        new_sum_plot.axes.x_axis.remove(new_sum_plot.axes.x_axis.numbers)
        new_sum_plot.to_corner(DL)
        new_sum_label = Tex(
            R"X_1 + \cdots + X_n",
            t2c={"X_1": BLUE, "X_n": BLUE}
        )
        new_sum_label.next_to(new_sum_plot.axes.c2p(0, 0.2), UR)

        rules = VGroup(sigma_form, var_form, note)

        self.play(
            FadeOut(plot2, RIGHT),
            FadeOut(top_labels[1], RIGHT),
            FadeOut(top_annotations2, RIGHT),
        )
        self.remove(plot1_group)
        self.play(
            FadeTransformPieces(
                plot1_group.replicate(4),
                new_top_groups,
            ),
            Write(dots),
            h_line.animate.shift(UP),
            FadeOut(sum_plot),
            FadeOut(sum_annotations),
            FadeOut(sum_label),
            rules.animate.scale(0.5).next_to(new_sum_plot.axes.c2p(0, 0), UP).to_edge(RIGHT),
            FadeIn(new_sum_plot),
            FadeIn(new_sum_label),
        )
        self.wait()

        # New variance formula
        t2c = {"X_1": BLUE, "X_n": BLUE}
        new_var_form = Tex(
            # R"Var(X_1 + \cdots + X_n) = Var(X_1) + \cdots + Var(X_n) = n \cdot Var(X_1)",
            # R"\sigma_{X_1 + \cdots + X_n}^2 = \sigma_{X_1}^2 + \cdots + \sigma_{X_n}^2 = n \cdot \sigma_{X_1}^2",
            R"\sigma_{X_1 + \cdots + X_n}^2 = n \cdot \sigma_{X_1}^2",
            t2c=t2c
        )
        new_sigma_form = Tex(
            R"\sigma_{X_1 + \cdots + X_n} = \sqrt{n} \cdot \sigma_{X_1}",
            t2c=t2c
        )

        new_var_form.next_to(h_line, DOWN)
        new_var_form.to_edge(RIGHT, buff=2.5)
        new_sigma_form.next_to(new_var_form, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        rects = VGroup(
            SurroundingRectangle(new_var_form[R"\sigma_{X_1 + \cdots + X_n}^2"]),
            SurroundingRectangle(new_var_form[R"\sigma_{X_1}^2"]),
        )
        var_labels = VGroup(
            Tex(R"\text{Var}(X_1 + \cdots + X_n)", t2c=t2c),
            Tex(R"\text{Var}(X_1)", t2c=t2c),
        )
        for label, rect in zip(var_labels, rects):
            label.set_max_width(rect.get_width())
            label.next_to(rect, DOWN)
            rect.add(label)

        self.play(FadeTransform(new_sum_label.copy(), new_var_form))
        self.wait()

        # Test
        self.play(FadeIn(rects[0]))
        self.wait()
        self.play(FadeOut(rects[0]), FadeIn(rects[1]))
        self.wait()
        self.play(FadeOut(rects[1]))
        self.wait()

        self.play(TransformMatchingTex(new_var_form.copy(), new_sigma_form))
        self.play(FlashAround(new_sigma_form[R"\sqrt{n}"], time_width=1.5, run_time=2))
        self.wait()


class InfiniteVariance(LimitingDistributions):
    def construct(self):
        # Test
        max_n = 300
        full_dist = np.array([(1 / n**1.5) for n in range(3, max_n)])
        old_plot = VGroup()

        var_label = Tex(R"\text{Var}(X) = 0.00", font_size=60)
        var_label.make_number_changeable("0.00")
        var_label.move_to(2 * UP)
        var_label["X"].set_color(BLUE)
        self.add(var_label)

        for n in range(6, len(full_dist)):
            dist = full_dist[:n] / full_dist[:n].sum()
            plot = self.get_top_distribution_plot(
                dist,
                width=min(n * (5 / 6), 12),
                height=4,
                y_range=(0, 0.301, 0.1)
            )
            plot.axes.x_axis.remove(plot.axes.x_axis.numbers)
            plot.axes.x_axis.add_numbers(range(0, n, 10), font_size=16)
            plot.axes.x_axis.ticks.stretch(0.2, 1)
            plot.bars.set_height(4, about_edge=DOWN, stretch=True)
            plot.shift(3 * DOWN + 6 * LEFT - plot.axes.c2p(0, 0))
            annotations = VGroup(*(self.get_mu_sigma_annotations(plot, "", "")))
            annotations.stretch(0.75, 1, about_edge=DOWN)
            annotations.set_opacity((max_n - n) / max_n)
            plot.add(annotations)

            mu, sigma = get_mean_and_sd(dist)
            var = sigma**2

            if n < 12:
                self.play(
                    FadeOut(old_plot),
                    FadeIn(plot),
                    ChangeDecimalToValue(var_label[-1], var),
                    run_time=0.5
                )
                self.wait(0.5)
            else:
                self.remove(old_plot)
                self.add(plot)
                var_label[-1].set_value(var)
                self.wait(1 / 30)
            old_plot = plot

        inf = Tex(R"\infty", font_size=60)
        inf.move_to(var_label[-1], LEFT)
        self.remove(var_label[-1])
        self.add(inf)
        self.wait()


class RuleOfThumb(BuildUpGaussian):
    def construct(self):
        # Title
        colors = color_gradient([YELLOW, RED], 3)
        t2c = {
            "68": colors[0],
            "95": colors[1],
            "99.7": colors[2],
        }
        title = Text(
            "The 68–95–99.7 rule",
            font_size=60,
            t2c=t2c
        )
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        underline = Underline(title)
        underline.set_stroke(GREY, [0, 3, 3, 3, 0]).scale(1.2)

        self.add(title, underline)

        # Axes and graph
        axes = NumberPlane(
            (-4, 4), (0, 1.0, 0.1),
            width=0.5 * FRAME_WIDTH,
            height=5.5,
            background_line_style=dict(
                stroke_color=GREY_A,
                stroke_width=2,
                stroke_opacity=0.5
            )
        )
        axes.to_edge(LEFT, buff=0)
        axes.to_edge(DOWN)
        axes.x_axis.add_numbers()
        axes.y_axis.add_numbers(
            np.arange(0.2, 1.2, 0.2),
            num_decimal_places=1,
            direction=DL,
            font_size=16,
        )

        graph = axes.get_graph(lambda x: gauss_func(x, 0, 1))
        graph.set_stroke(YELLOW, 3)

        self.add(axes, graph)

        # Function for changeable area
        area = VMobject()
        area.set_stroke(width=0)
        area.set_fill(YELLOW, 0.5)
        ab_tracker = ValueTracker(np.array([0, 0]))

        def update_area(area, dx=0.01):
            a, b = ab_tracker.get_value()
            if a == b:
                area.clear_points()
                return
            xs = np.arange(a, b, dx)
            ys = gauss_func(xs, 0, 1)
            samples = axes.c2p(xs, ys)
            area.set_points_as_corners([
                *samples,
                axes.c2p(b, 0),
                axes.c2p(a, 0),
                samples[0],
            ])

        area.add_updater(update_area)
        self.add(area)

        # Area decimal
        area_label = TexText("Area = 0.000", font_size=40)
        area_label.set_backstroke(width=8)
        num = area_label.make_number_changeable("0.000")
        from scipy.stats import norm

        def get_area():
            a, b = ab_tracker.get_value()
            return norm.cdf(b) - norm.cdf(a)

        num.add_updater(lambda m: m.set_value(get_area()))
        area_label.next_to(graph.get_top(), UR, buff=0.5)
        area_arrow = Arrow(area_label.get_bottom(), axes.c2p(0, 0.2))

        # Normal label
        func_label = VGroup(
            Text("Standard normal\ndistribution", font_size=30),
            Tex(R"\frac{1}{\sqrt{2\pi}} e^{-x^2 / 2}").set_color(YELLOW)
        )
        func_label[1][R"\frac{1}{\sqrt{2\pi}}"].scale(0.75, about_edge=RIGHT)
        func_label.arrange(DOWN)
        func_label.next_to(graph, UP).shift(DOWN)
        func_label.to_edge(LEFT)
        func_label.set_backstroke(width=8)

        self.play(
            FadeIn(func_label[0]),
            Write(func_label[1]),
        )
        self.wait()

        # Rule of thumb labels
        labels = VGroup()
        for n, num, color in zip(it.count(1), [68, 95, 99.7], colors):
            label = Text(f"""
                {num}% of values fall within
                {n} standard deviations of the mean
            """, font_size=40)
            label[f"{n}"].set_color(color)
            label[f"{num}%"].set_color(color)
            labels.add(label)

        labels.arrange(DOWN, aligned_edge=LEFT, buff=LARGE_BUFF)
        labels.move_to(midpoint(axes.get_right(), RIGHT_SIDE))

        # Show successive regions
        for n in range(3):
            anims = [
                FadeIn(labels[n], DOWN),
                ab_tracker.animate.set_value([-n - 1, n + 1]),
            ]
            if n == 0:
                anims.extend([VFadeIn(area_label), FadeIn(area_arrow)])
            self.play(*anims)
            self.wait()


class AnalyzeHundreDiceQuestion(LimitingDistributions):
    distribution = [1 / 6] * 6

    def construct(self):
        # Setup
        h_line = Line(LEFT, RIGHT)
        h_line.set_width(FRAME_WIDTH)
        h_line.set_stroke(GREY, 2)
        self.add(h_line)

        N = 100
        dist = self.distribution
        top_plot = self.get_top_distribution_plot(dist)
        top_plot.to_edge(LEFT)

        sum_plot_config = dict(
            x_range=(0, 500, 10),
            y_range=(0, 0.05, 0.01),
            x_num_range=(0, 600, 100),
            max_width=14,
        )
        sum_plot = self.get_sum_plot(dist, N, top_plot, **sum_plot_config)

        sum_label = self.get_sum_label(N)

        self.add(top_plot)
        self.add(sum_plot)
        self.add(sum_label)

        # Add fair die labels
        sixths = Tex("1 / 6").replicate(6)
        sixths.set_width(0.4 * top_plot.bars[0].get_width())
        for sixth, bar in zip(sixths, top_plot.bars):
            sixth.next_to(bar, UP, SMALL_BUFF)

        fair_label = Text("Fair die")
        fair_label.move_to(top_plot, UP)

        dot_config = dict(fill_color=BLUE_E, dot_color=WHITE, stroke_width=1)
        dice = VGroup(*(
            DieFace(value, **dot_config)
            for value in range(1, 7)
        ))
        for die, number in zip(dice, top_plot.axes.x_axis.numbers):
            die.match_height(number).scale(1.5)
            die.move_to(number)

        self.add(fair_label)
        self.add(sixths)
        self.add(dice)

        # Animate in plot
        for n in range(5, 101):
            sum_label.become(self.get_sum_label(n))
            sum_plot.become(self.get_sum_plot(dist, n, top_plot, **sum_plot_config))

            self.wait(1 / 30)

        n = 100

        # Compute mean and standard deviation
        kw = dict(
            t2c={
                "100": YELLOW,
                R"\mu": PINK,
                R"\mu_s": PINK,
                R"\mu_a": PINK,
                R"\sigma": RED,
                R"\sigma_s": RED,
                R"\sigma_a": RED,
                "3.5": PINK,
                "1.71": RED,
            },
            font_size=36,
        )
        top_mean = Tex(
            R"\mu = \frac{1}{6}\big(1 + 2 + 3 + 4 + 5 + 6\big) = 3.5",
            **kw
        )
        top_var = Tex(
            R"\text{Var}(X) = \frac{1}{6}\Big((1 - 3.5)^2 + \cdots + (6 - 3.5)^2 \Big) = 2.92",
            **kw
        )
        top_var.scale(0.8)
        top_var[R"\text{Var}(X) = "].scale(1 / 0.8, about_edge=RIGHT)
        top_var["= 2.92"].scale(1 / 0.8, about_edge=LEFT)
        top_var.refresh_bounding_box()

        top_sd = Tex(R"\sigma = \sqrt{\text{Var}(X)} = 1.71", **kw)

        top_eqs = VGroup(top_mean, top_var, top_sd)
        top_eqs.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        top_eqs.to_corner(UR, buff=0.25)

        top_mean_simple = Tex(R"\mu = 3.5", **kw).scale(1.25)
        top_sd_simple = Tex(R"\sigma = 1.71", **kw).scale(1.25)
        simp_eqs = VGroup(top_mean_simple, top_sd_simple)
        simp_eqs.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        simp_eqs.next_to(top_plot.bars, RIGHT, MED_LARGE_BUFF, aligned_edge=DOWN)

        self.play(Write(top_mean))
        self.wait()
        self.play(Write(top_var))
        self.wait()
        self.play(
            Write(top_sd),
            Transform(top_var[R"\text{Var}(X)"].copy(), top_sd[R"\text{Var}(X)"].copy(), remover=True)
        )
        self.wait()

        self.play(
            TransformMatchingTex(top_mean.copy(), top_mean_simple),
            TransformMatchingTex(top_sd.copy(), top_sd_simple),
            top_eqs.animate.scale(0.75, about_edge=UR).set_opacity(0.5),
        )
        self.wait()

        # Sum mean and sd
        low_mean = Tex(R"\mu_s = 100 \cdot \mu = 350", **kw)
        low_sd = Tex(R"\sigma_s = \sqrt{100} \cdot \sigma = 17.1", **kw)
        low_eqs = VGroup(low_mean, low_sd)
        low_eqs.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        low_eqs.next_to(h_line, DOWN)
        low_eqs.align_to(top_mean_simple, LEFT)

        mean_label, sd_label = self.get_mu_sigma_annotations(
            sum_plot, "350", "17.1", min_height=1.0, x_min=n
        )

        self.play(TransformMatchingTex(top_mean_simple.copy(), low_mean))
        self.play(Write(mean_label))
        self.wait()
        self.play(TransformMatchingTex(top_sd_simple.copy(), low_sd, key_map={"1.71": "17.1"}))
        sd_label.save_state()
        sd_label.stretch(0, 0).set_opacity(0)
        self.play(Restore(sd_label))
        self.wait()

        # Show range
        sd_lines, sd_arrows, sd_num = sd_label
        sd_line_ghosts = sd_lines.copy().set_opacity(0.5)
        sd_line_ghosts.stretch(0.75, 1, about_edge=DOWN)
        two_times = Tex(R"2 \, \cdot")
        two_times.match_height(sd_num)
        new_sd_num = VGroup(two_times, sd_num.copy())
        new_sd_num.arrange(RIGHT, buff=SMALL_BUFF)
        new_sd_num.set_color(RED)
        new_sd_num.next_to(sd_arrows.get_right(), UP, SMALL_BUFF)
        new_sd_num.scale(1.25, about_edge=DOWN)

        bound_equations = VGroup(
            Tex(R"\mu_s - 2 \sigma_s = 350 - 2 \cdot 17.1 \approx 316", **kw),
            Tex(R"\mu_s + 2 \sigma_s = 350 + 2 \cdot 17.1 \approx 384", **kw),
        )
        bound_equations.arrange(DOWN, aligned_edge=LEFT)
        bound_equations.move_to(sum_plot.axes.c2p(0, 0.035), UL)
        bound_equations.shift(RIGHT)

        self.add(sd_line_ghosts)
        self.play(
            sd_lines.animate.stretch(2, 0),
            sd_arrows.animate.stretch(2, 0),
            TransformMatchingShapes(sd_num, new_sd_num, run_time=1)
        )
        new_sd_num.set_backstroke(BLACK, width=5)
        self.add(new_sd_num)
        self.wait()
        self.play(LaggedStartMap(FadeIn, bound_equations, shift=0.5 * UP, lag_ratio=0.75))
        self.wait()

        rhss = VGroup(bound_equations[0]["316"], bound_equations[1]["384"])
        rhs_rect = SurroundingRectangle(rhss)
        rhs_rect.set_stroke(YELLOW, 2)
        self.play(ShowCreation(rhs_rect))
        self.wait()

        # Add range values to diagram
        new_rhss = rhss.copy()
        new_rhss.set_color(RED)
        new_rhss.scale(mean_label[1].get_height() / rhss[0].get_height())
        for rhs, line in zip(new_rhss, sd_lines):
            rhs.next_to(line, UP, SMALL_BUFF)
            rhs.align_to(mean_label[1], DOWN)

        for rhs, new_rhs in zip(rhss, new_rhss):
            self.play(TransformFromCopy(rhs, new_rhs))
        self.wait()
        self.play(LaggedStartMap(
            FadeOut, VGroup(*bound_equations, rhs_rect), shift=DOWN
        ))

        # Transition to sample mean case
        sample_mean_expr = Tex(R"X_1 + \cdots + X_{100} \over 100", **kw)
        sample_mean_expr.move_to(sum_label, UP)

        number_swaps = []
        x_numbers = VGroup()
        new_x_labels = VGroup(*(Integer(k, font_size=36) for k in range(1, 6)))
        for old_label, new_label in zip(sum_plot.axes.x_axis.numbers, new_x_labels):
            faded_old_label = old_label
            old_label = old_label.copy()
            faded_old_label.set_opacity(0)
            self.add(old_label)
            new_label.move_to(old_label)
            number_swaps.append(FadeIn(new_label, 0.5 * UP))
            number_swaps.append(FadeOut(old_label, 0.5 * UP))
            x_numbers.add(old_label)

        diagram_labels = VGroup(new_rhss[0], mean_label[1], new_rhss[1], new_sd_num)
        new_diagram_labels = VGroup(
            DecimalNumber(3.16),
            DecimalNumber(3.50),
            DecimalNumber(3.84),
            Tex(R"2 \cdot 0.171")
        )
        for old_label, new_label in zip(diagram_labels, new_diagram_labels):
            new_label.match_style(old_label.family_members_with_points()[0])
            new_label.replace(old_label, dim_to_match=0)
            number_swaps.append(FadeOut(old_label, 0.5 * UP))
            number_swaps.append(FadeIn(new_label, 0.5 * UP))
            x_numbers.add(old_label)
        
        self.play(WiggleOutThenIn(sum_label))
        length = len(sum_label) + 2
        self.play(
            FadeTransform(sum_label, sample_mean_expr[:length]),
            Write(sample_mean_expr[length:]),
        )
        self.wait()
        self.play(LaggedStartMap(FlashAround, x_numbers, lag_ratio=0.1, time_width=1.5))
        self.play(
            LaggedStart(*number_swaps),
            low_eqs.animate.set_opacity(0.4),
        )

        # Show dice
        rect = SurroundingRectangle(sample_mean_expr)
        rect.set_stroke(BLUE, 2)
        avg_words = Text("Average of\n100 rolls")
        avg_words.next_to(rect, RIGHT)

        dice = VGroup(*(
            DieFace(random.randint(1, 6), **dot_config)
            for x in range(100)
        ))
        dice.arrange_in_grid(10, 10)
        dice.set_height(2)
        dice.next_to(sum_plot.axes.c2p(0, 0), UR, SMALL_BUFF)
        dice.match_x(avg_words)

        self.play(
            ShowCreation(rect),
            Write(avg_words),
            FadeIn(dice, lag_ratio=0.1),
        )
        self.wait()

        # Mean mean and sd label
        avg_mean = Tex(R"\mu_a = 3.5", **kw)
        avg_sd = Tex(R"\sigma_a = \sigma / \sqrt{100} = 0.171", **kw)
        avg_eqs = VGroup(avg_mean, avg_sd)
        avg_eqs.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        avg_eqs.move_to(low_eqs, UL)
        avg_eqs.shift(0.2 * LEFT)

        self.play(
            FadeIn(avg_mean, RIGHT),
            low_mean.animate.to_edge(RIGHT),
        )
        self.wait()
        self.play(
            FadeIn(avg_sd, RIGHT),
            low_sd.animate.to_edge(RIGHT)
        )
        self.wait()
        self.play(
            FlashAround(low_sd),
            low_sd.animate.set_opacity(1),
        )
        self.wait()
        self.play(low_sd.animate.set_opacity(0.5))
        self.wait()


class Thumbnail(LimitingDistributions):
    def construct(self):
        # Tests
        dist = [1 / 6] * 6
        ss_plot = self.get_scaled_sum_plot(dist, 50)
        for axis in ss_plot.axes:
            axis.remove(axis.numbers)
            for tick in axis.ticks:
                tick.scale(0.5)
        ss_plot.axes.y_axis.set_opacity(0)
        ss_plot.scale(2)
        ss_plot.bars.set_submobject_colors_by_gradient(
            *3 * [BLUE], *3 * [YELLOW]
        )
        ss_plot.shift(2 * DOWN - ss_plot.axes.c2p(0, 0))
        ss_plot.to_edge(DOWN)
        self.add(ss_plot)

        # Top plots
        top_plots = VGroup()
        np.random.seed(3)
        dists = [
            EXP_DISTRIBUTION,
            normalize(np.random.random(6))**2,
            normalize(np.random.random(6))**2,
            U_SHAPED_DISTRIBUTION,
        ]
        for dist in dists:
            top_plot = self.get_top_distribution_plot(dist, width=3, height=2, y_range=(0, 0.401, 0.1))
            for axis in top_plot.axes:
                axis.remove(axis.numbers)
            top_plots.add(top_plot)
            top_plot.bars.set_submobject_colors_by_gradient(BLUE_D, TEAL)
            top_plot.bars.set_stroke(WHITE, 1)
        top_plots.arrange(RIGHT, buff=2.0, aligned_edge=DOWN)
        top_plots.set_width(FRAME_WIDTH - 1)
        top_plots.to_edge(UP)

        arrows = VGroup(*(
            Arrow(
                tp.get_bottom(),
                ss_plot.bars.get_top() + vect,
                buff=0.3,
                stroke_width=10,
                stroke_color=YELLOW
            )
            for tp, vect in zip(top_plots, np.linspace(2 * LEFT, 2 * RIGHT, len(top_plots)))
        ))

        self.add(top_plots)
        self.add(arrows)

        # Words
        words = Text("Bizarrely\nUniversal", font_size=90)
        words.move_to(ss_plot.axes.c2p(0, 0.03), DOWN).to_edge(LEFT)
        words.set_x(0)
        words.set_backstroke()
        # self.add(words)

        # Formula
        form = Tex(R"{1 \over \sqrt{2\pi}} e^{-x^2 / 2}", font_size=90)
        form.move_to(words, DOWN).to_edge(RIGHT)



