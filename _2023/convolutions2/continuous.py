from manim_imports_ext import *
from _2023.clt.main import *
from _2022.convolutions.discrete import *

import scipy.stats


def wedge_func(x):
    return np.clip(-np.abs(x) + 1, 0, 1)


def double_lump(x):
    return 0.45 * np.exp(-6 * (x - 0.5)**2) + np.exp(-6 * (x + 0.5)**2)


def uniform(x):
    return 1.0 * (-0.5 < x) * (x < 0.5)


def get_conv_graph(axes, f, g, dx=0.1):
    dx = 0.1
    x_min, x_max = axes.x_range[:2]
    x_samples = np.arange(x_min, x_max + dx, dx)
    f_samples = np.array([f(x) for x in x_samples])
    g_samples = np.array([g(x) for x in x_samples])
    full_conv = np.convolve(f_samples, g_samples)
    x0 = len(x_samples) // 2 - 1  # TODO, be smarter about this
    conv_samples = full_conv[x0:x0 + len(x_samples)]
    conv_graph = VMobject()
    conv_graph.set_stroke(TEAL, 2)
    conv_graph.set_points_smoothly(axes.c2p(x_samples, conv_samples * dx))
    return conv_graph


class TransitionToContinuousProbability(InteractiveScene):
    def construct(self):
        # Setup axes and initial graph
        axes = Axes((0, 12), (0, 1, 0.2), width=14, height=5)
        axes.to_edge(LEFT, LARGE_BUFF)
        axes.to_edge(DOWN, buff=1.25)

        def pd(x):
            return (x**4) * np.exp(-x) / 8.0

        graph = axes.get_graph(pd)
        graph.set_stroke(WHITE, 2)
        bars = axes.get_riemann_rectangles(graph, dx=1, x_range=(0, 6), input_sample_type="right")
        bars.set_stroke(WHITE, 3)

        y_label = Text("Probability", font_size=48)
        y_label.next_to(axes.y_axis, UP, SMALL_BUFF)
        y_label.shift_onto_screen()

        self.add(axes)
        self.add(y_label)
        self.add(*bars)

        self.frame.move_to(0.5 * DOWN)

        # Label as die probabilities
        dice = get_die_faces(fill_color=BLUE_E, dot_color=WHITE, stroke_width=1)
        dice.set_height(0.5)
        for bar, die in zip(bars, dice):
            die.next_to(bar, DOWN)

        self.play(FadeIn(dice, 0.1 * UP, lag_ratio=0.05, rate_func=overshoot))
        self.wait()
        self.play(FadeOut(dice, RIGHT, rate_func=running_start, run_time=1, path_arc=-PI / 5, lag_ratio=0.01))

        # Make continuous
        all_rects = VGroup(*(
            axes.get_riemann_rectangles(
                graph,
                x_range=(0, min(6 + n, 12)),
                dx=(1 / n),
                input_sample_type="right",
            ).set_stroke(WHITE, width=(2.0 / n), opacity=(2.0 / n), background=False)
            for n in (*range(1, 10), *range(10, 20, 2), *range(20, 100, 5))
        ))
        area = all_rects[-1]
        area.set_stroke(width=0)

        self.remove(bars)
        self.play(
            ShowSubmobjectsOneByOne(all_rects, rate_func=bezier([0, 0, 0, 0, 1, 1])),
            FadeOut(y_label),
            run_time=5
        )
        self.remove(all_rects)
        self.add(area, graph)
        self.play(ShowCreation(graph))
        self.wait()

        # Show continuous value
        x_tracker = ValueTracker(0)
        get_x = x_tracker.get_value
        tip = ArrowTip(angle=PI / 2)
        tip.set_height(0.25)
        tip.add_updater(lambda m: m.move_to(axes.c2p(get_x(), 0), UP))
        x_label = DecimalNumber(font_size=36)
        x_label.add_updater(lambda m: m.set_value(get_x()))
        x_label.add_updater(lambda m: m.next_to(tip, DOWN, buff=0.2))

        self.play(FadeIn(tip), FadeIn(x_label))
        self.play(x_tracker.animate.set_value(12), run_time=6)

        # Labels
        x_labels = VGroup(*(
            Text(text)
            for text in [
                "Temperature tomorrow at noon",
                "Value of XYZ next year",
                "Time before the next bus comes",
            ]
        ))
        for x_label in x_labels:
            x_label.next_to(axes.c2p(4, 0), DOWN, buff=0.2)

        self.play(Write(x_labels[0], run_time=1))
        for xl1, xl2 in zip(x_labels, x_labels[1:]):
            self.wait()
            self.play(
                FadeOut(xl1, 0.5 * UP),
                FadeIn(xl2, 0.5 * UP),
            )
        self.wait()
        self.play(
            FadeOut(x_labels[-1]),
            x_tracker.animate.set_value(0).set_anim_args(run_time=3),
        )
        self.play(
            x_tracker.animate.set_value(12), run_time=5
        )
        self.remove(tip, x_label)

        # Label density
        density = Text("Probability density")
        density.match_height(y_label)
        density.move_to(y_label, LEFT)
        cross = Cross(y_label)
        cross.set_stroke(RED, width=(0, 8, 8, 8, 0))

        self.play(FadeIn(y_label))
        self.play(ShowCreation(cross))
        self.wait()
        self.play(
            VGroup(y_label, cross).animate.shift(0.75 * UP),
            FadeIn(density),
            self.frame.animate.set_y(0),
        )
        self.wait()

        # Interpretation
        range_tracker = ValueTracker([0, 12])
        sub_area_opacity_tracker = ValueTracker(0)

        def get_subarea():
            result = axes.get_area_under_graph(
                graph, range_tracker.get_value()
            )
            result.set_stroke(width=0)
            result.set_fill(TEAL, sub_area_opacity_tracker.get_value())
            return result

        sub_area = always_redraw(get_subarea)

        v_lines = Line(DOWN, UP).replicate(2)
        v_lines.set_stroke(GREY_A, 1)
        v_lines.set_height(FRAME_HEIGHT)

        def update_v_lines(v_lines):
            values = range_tracker.get_value()
            for value, line in zip(values, v_lines):
                line.move_to(axes.c2p(value, 0), DOWN)

        v_lines.add_updater(update_v_lines)

        bound_labels = Tex("ab")
        bound_labels[0].add_updater(lambda m: m.move_to(v_lines[0], DOWN).shift(0.5 * DOWN))
        bound_labels[1].add_updater(lambda m: m.move_to(v_lines[1], DOWN).shift(0.5 * DOWN))
        bound_labels.add_updater(lambda m: m.set_opacity(sub_area_opacity_tracker.get_value()))

        prob_label = Tex(R"P(a < x < b) = \text{This area}")
        prob_label.move_to(2 * UR)
        rhs = prob_label[R"\text{This area}"]
        prob_arrow = Arrow(LEFT, RIGHT)
        prob_arrow.add_updater(lambda m: m.put_start_and_end_on(
            rhs.get_bottom() + 0.1 * DOWN,
            sub_area.get_center(),
        ))

        self.add(area, sub_area, graph, bound_labels)
        self.play(
            area.animate.set_opacity(0.1),
            range_tracker.animate.set_value([3, 4.5]),
            sub_area_opacity_tracker.animate.set_value(1),
            VFadeIn(v_lines),
            FadeIn(prob_label),
            VFadeIn(prob_arrow),
            run_time=2,
        )
        self.wait()
        for pair in [(5, 6), (1, 3), (2.5, 3), (4, 7)]:
            self.play(range_tracker.animate.set_value(pair), run_time=2)
            self.wait()

        # Name the pdf
        long_name = Text("probability\ndensity\nfunction", alignment="LEFT")
        short_name = Text("pdf")
        long_name.move_to(axes.c2p(1, 0.75), UL)
        short_name.move_to(axes.c2p(2, 0.5))

        self.play(FadeIn(long_name, lag_ratio=0.1))
        self.wait()
        self.play(TransformMatchingStrings(long_name, short_name, lag_ratio=0.01, run_time=1))
        self.wait()

        # Show integral
        int_rhs = Tex(R"\int_a^b p_X(x) \, dx")
        int_rhs.move_to(rhs, LEFT)

        self.play(
            rhs.animate.set_opacity(0).shift(0.5 * DOWN + 1.0 * LEFT),
            FadeIn(int_rhs, DL)
        )
        self.wait()

        # Ambient range changing
        for pair in [(2.5, 7), (8, 10), (3.5, 9), (3, 4.5)]:
            self.play(range_tracker.animate.set_value(pair), run_time=3)
            self.wait()


class CompareFormulas(InteractiveScene):
    def construct(self):
        # Setup division
        v_line = Line(DOWN, UP).set_height(FRAME_HEIGHT)
        kw = dict(font_size=60)
        disc_title, cont_title = titles = VGroup(
            Text("Discrete case", **kw),
            Text("Continuous case", **kw),
        )
        for vect, title in zip([LEFT, RIGHT], titles):
            title.move_to(vect * FRAME_WIDTH * 0.25)
            title.to_edge(UP, buff=MED_SMALL_BUFF)
            underline = Underline(title, stretch_factor=1.5)
            underline.set_stroke(GREY_B)
            title.add(underline)

        self.add(v_line, titles)

        # Discrete diagram (pre-made image)
        discrete_diagram = ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/convolutions2/dice/DiscreteDistributionSum.png")
        discrete_diagram.set_height(5)
        discrete_diagram.match_x(disc_title)
        discrete_diagram.to_edge(DOWN, buff=0.5)
        self.add(discrete_diagram)

        # Continuous diagrams
        kw = dict(y_range=(0, 1, 0.5), width=5, height=1.5)
        all_axes = VGroup(
            Axes((-2, 2), **kw),
            Axes((-2, 2), **kw),
            Axes((-3, 3), **kw),
        )
        all_axes[:2].arrange(RIGHT, buff=1.5)
        all_axes[2].set_width(10)
        all_axes[2].next_to(all_axes[:2], UP, buff=1.5)
        all_axes.set_width(FRAME_WIDTH * 0.5 - 0.5)
        all_axes.match_x(cont_title)
        all_axes.to_edge(DOWN, buff=0.75)

        graphs = VGroup(
            all_axes[0].get_graph(
                wedge_func,
                use_smoothing=False
            ).set_stroke(BLUE),
            all_axes[1].get_graph(double_lump).set_stroke(RED),
            get_conv_graph(all_axes[2], wedge_func, double_lump).set_stroke(TEAL)
        )
        graphs.set_stroke(width=2)

        tex_kw = dict()
        labels = VGroup(
            Tex("f(x)", font_size=24),
            Tex("g(y)", font_size=24),
            Tex("[f * g](s)", font_size=36),
        )
        for label, axes in zip(labels, all_axes):
            label.move_to(midpoint(axes.get_corner(UR), axes.get_top()), UP)

        plots = VGroup(*(
            VGroup(*tup)
            for tup in zip(all_axes, graphs, labels)
        ))

        self.add(plots)

        # Formulae
        disc_formula = Tex(
            R"\big[P_X * P_Y\big](s) = \sum_{x = 1}^N P_X(x) \cdot P_Y(s - x)",
            font_size=36,
            t2c={"X": BLUE, "Y": RED},
        )
        disc_formula.next_to(disc_title, DOWN, MED_LARGE_BUFF)

        cont_formula = Tex(
            R"\big[f * g \big](s) = \int_{-\infty}^\infty f(x) g(s - x) \, dx",
            font_size=36,
        )
        cont_formula.match_x(cont_title)
        cont_formula.match_y(disc_formula)

        self.play(
            FadeIn(disc_formula, run_time=2, lag_ratio=0.1),
            FlashAround(disc_formula, time_width=1.5, run_time=2),
        )
        self.wait()

        rect = SurroundingRectangle(disc_formula)
        rect.set_stroke(YELLOW, 2, opacity=0)
        target_rects = VGroup(*(
            SurroundingRectangle(disc_formula[s][0])
            for s in ["P_X", "P_Y", "P_X * P_Y", re.compile(R"\\sum.*")]
        ))
        target_rects.set_stroke(YELLOW, 2)
        for target_rect in target_rects:
            self.play(rect.animate.become(target_rect), run_time=0.5)
            self.wait()
        self.play(FadeOut(rect))
        self.play(
            TransformMatchingTex(
                disc_formula.copy(), cont_formula,
                lag_ratio=-0.001,
                path_arc=-0.1 * PI
            )
        )
        self.wait()

        # Out of context
        fade_rect = FullScreenFadeRectangle()
        randy = Randolph()
        randy.next_to(cont_formula, DL, LARGE_BUFF)

        self.add(fade_rect, cont_formula)
        self.play(
            FadeIn(fade_rect),
            VFadeIn(randy),
            randy.change("horrified", cont_formula),
        )
        self.play(Blink(randy))
        self.wait()
        self.play(randy.change("pondering", cont_formula))
        self.play(Blink(randy))
        self.wait()

# Random samples from continuous distributions


class RepeatedSamplesFromContinuousDistributions(InteractiveScene):
    sigma1 = 1.0
    sigma2 = 1.0

    graph_colors = [BLUE, RED, TEAL]
    graph_stroke_width = 2

    dot_fade_factor = 0.25

    def setup(self):
        super().setup()
        self.random_variables = self.get_random_variables()
        self.all_dots = Group()
        self.add(self.all_dots)

    def get_plots(self):
        # Axes and graphs
        all_axes = self.get_axes()
        left_axes = all_axes[:2]
        left_axes.arrange(DOWN, buff=1.5)
        left_axes.to_edge(LEFT)
        all_axes[2].center().to_edge(RIGHT)

        for axes in all_axes:
            axes.x_axis.add_numbers(font_size=16)
            axes.y_axis.set_stroke(opacity=0.5)

        pdfs = self.get_pdfs()
        graphs = VGroup(*(
            axes.get_graph(func).set_stroke(color)
            for axes, func, color in zip(
                all_axes,
                self.get_pdfs(),
                self.graph_colors
            )
        ))
        graphs.add(self.get_sum_graph(all_axes[2]))
        graphs.set_stroke(width=self.graph_stroke_width)

        # Labels
        labels = self.get_axes_labels(all_axes)

        plots = VGroup(*(
            VGroup(*tup)
            for tup in zip(all_axes, graphs, labels)
        ))
        return plots

    def get_axes(self):
        return VGroup(*(
            Axes(
                (-5, 5), (0, 0.5, 0.25),
                width=5.5,
                height=2,
            )
            for x in range(3)
        ))

    def get_axes_labels(self, all_axes):
        a1, a2, a3 = all_axes
        return VGroup(
            Tex("X").move_to(midpoint(a1.get_corner(UR), a1.get_top())),
            Tex("Y").move_to(midpoint(a2.get_corner(UR), a2.get_top())),
            Tex("X + Y").next_to(a3, UP, buff=1.0)
        )

    def repeated_samples(self, plots, n_repetitions, **kwargs):
        for n in range(n_repetitions):
            self.animate_samples(plots, **kwargs)

    def animate_samples(
        self,
        plots,
        time_between_samples=0.25,
        time_before_fade=1.0,
        animate=True,
    ):
        # Setup
        xy_samples = np.round(self.get_samples(), 2)
        sample_sum = sum(xy_samples)
        samples = [*xy_samples[:2], sample_sum]
        dots = Group()
        labels = VGroup()
        lines = VGroup()
        for sample, plot in zip(samples, plots):
            axes, graph, sym_label = plot
            dot = GlowDot(axes.c2p(sample, 0))
            label = DecimalNumber(sample)
            label.next_to(sym_label, DOWN)
            label.scale(0.75, about_edge=DOWN)
            label.set_fill(GREY_A)

            line = axes.get_v_line_to_graph(sample, graph, line_func=Line)
            line.set_stroke(YELLOW, 2)

            dots.add(dot)
            labels.add(label)
            lines.add(line)

        if len(plots) > 2:
            sum_label = VGroup(
                DecimalNumber(samples[0]),
                Tex("+") if samples[1] > 0 else Tex("-"),
                DecimalNumber(abs(samples[1])),
                Tex("="),
                DecimalNumber(samples[2]),
            )
            sum_label.arrange(RIGHT, buff=0.15)
            sum_label[-1].align_to(sum_label[0], DOWN)
            sum_label.match_height(labels[2])
            sum_label.match_style(labels[2])
            sum_label.move_to(labels[2], DL)
            labels.remove(labels[2])
            labels.add(sum_label)
            sum_label.shift((plots[2][2]["+"].get_x() - sum_label[1].get_x()) * RIGHT)

        # Animate
        for i in range(min(2, len(plots))):
            self.add(dots[i], labels[i], lines[i])
            if len(plots) > 2:
                self.add(sum_label[:2 * i + 1])
            self.wait(time_between_samples)
        if len(plots) > 2:
            self.play(LaggedStart(
                Transform(dots[0].copy(), dots[2].copy().set_opacity(0.5), remover=True),
                Transform(dots[1].copy(), dots[2].copy().set_opacity(0.5), remover=True),
                FadeTransform(sum_label[:3].copy(), sum_label[3:]),
                run_time=1.0 if animate else 0,
            ))
            self.add(sum_label)
            self.add(dots[2])
        self.wait(time_before_fade)
        kw = dict(run_time=0.25 if animate else 0)
        self.play(
            LaggedStart(*(
                dot.animate.set_radius(0.1).set_opacity(self.dot_fade_factor)
                for dot in dots
            ), **kw),
            LaggedStartMap(FadeOut, labels, **kw),
            LaggedStartMap(FadeOut, lines[:2], **kw),
        )
        self.all_dots.add(*dots)
        self.add(self.all_dots)

    def get_random_variables(self):
        return [
            scipy.stats.norm(0, self.sigma1),
            scipy.stats.norm(0, self.sigma2),
        ]

    def get_samples(self):
        return [
            np.round(var.rvs(), 2)
            for var in self.random_variables
        ]

    def get_pdfs(self):
        return [var.pdf for var in self.random_variables]

    def get_sum_graph(self, axes):
        graph = get_conv_graph(axes, *self.get_pdfs())
        graph.set_stroke(self.graph_colors[2])
        return graph


class SampleTwoNormals(RepeatedSamplesFromContinuousDistributions):
    random_seed = 1
    sigma1 = 1
    sigma2 = 1.5

    annotations = False

    def construct(self):
        # Setup plots
        plots = self.get_plots()
        plots.to_edge(UP, buff=1.0)
        sum_axes, sum_graph, sum_label = plots[2]
        sum_axes.y_axis.set_opacity(0)
        sum_graph.set_opacity(0)
        sum_label.shift(DOWN)

        normal_parameters = VGroup(*(
            self.get_normal_parameter_labels(plot, 0, sigma)
            for plot, sigma in zip(plots, [self.sigma1, self.sigma2])
        ))
        normal_words = VGroup(*(
            Text("Normal\ndistribution", font_size=30, alignment="LEFT").next_to(
                parameters, UP, MED_LARGE_BUFF, LEFT
            )
            for parameters in normal_parameters
        ))

        if self.annotations:
            plots.set_opacity(0)

        # Repeated samples of X
        frame = self.frame
        frame.move_to(plots[0])
        frame.set_height(plots[0].get_height() + 2)

        self.add(plots[0])

        if self.annotations:
            # Describe X
            axes, graph, label = plots[0]
            label_rect = SurroundingRectangle(label, buff=0.05)
            label_rect.set_stroke(YELLOW, 2)
            sample_point = label.get_center() + label.get_height() * DOWN

            rv_words = Text("Random variable", font_size=24)
            rv_words.next_to(label, UR, buff=0.5)
            rv_arrow = Arrow(rv_words, label, buff=0.2, stroke_color=YELLOW)

            sample_words = Text("Samples", font_size=24)
            sample_words.next_to(sample_point, DOWN, LARGE_BUFF)
            sample_words.match_x(rv_words)
            sample_arrow = Arrow(sample_words, sample_point + 0.25 * DR, buff=0.2)
            sample_arrow.set_stroke(BLUE)

            self.play(
                FadeIn(rv_words, lag_ratio=0.1),
                ShowCreation(label_rect),
                GrowArrow(rv_arrow),
            )
            self.wait()
            self.play(
                FadeTransform(rv_words.copy(), sample_words),
                TransformFromCopy(rv_arrow, sample_arrow),
                FadeOut(label_rect),
            )
            self.wait()

            # Describe normal distribution
            curve_copy = graph.copy()
            curve_copy.set_stroke(TEAL, 7, 1)

            self.play(
                LaggedStartMap(FadeOut, VGroup(
                    sample_words, sample_arrow, rv_arrow, rv_words,
                )),
                Write(normal_words[0], run_time=2),
                FadeIn(normal_parameters[0]),
                VShowPassingFlash(curve_copy, time_width=0.7, time_span=(0.5, 5)),
            )
            self.wait()

            # Show area
            bound_tracker = ValueTracker([-1, -1])
            area = always_redraw(lambda: axes.get_area_under_graph(
                graph, bound_tracker.get_value()
            ))

            self.add(area)
            self.play(
                bound_tracker.animate.set_value([-1, 2]),
                run_time=3
            )
            self.wait()
            self.play(FadeOut(area))

        else:
            self.repeated_samples(plots[:1], 30, time_before_fade=0.5)

        # Show Y
        frame.generate_target()
        frame.target.set_height(plots[:2].get_height() + 2)
        frame.target.move_to(plots[:2])

        self.play(
            MoveToTarget(frame),
            FadeIn(plots[1]),
            FadeOut(self.all_dots),
        )
        self.all_dots.clear()

        if self.annotations:
            self.play(TransformFromCopy(*normal_words))
            self.play(
                LaggedStartMap(FadeIn, normal_parameters[1], lag_ratio=0.25),
                LaggedStartMap(
                    FlashAround, normal_parameters[1],
                    stroke_width=1,
                    time_width=1.0,
                    lag_ratio=0.25,
                ),
            )
            self.wait()
        else:
            self.repeated_samples(plots[:2], 10, time_before_fade=0.5)

        # Show sum
        self.play(
            frame.animate.to_default_state(),
            FadeIn(plots[2]),
            FadeOut(self.all_dots),
        )
        self.all_dots.clear()

        if self.annotations:
            # Show multiple graphs
            axes, graph, label = plots[2]
            graphs = VGroup(
                graph.copy(),
                axes.get_graph(lambda x: 0.3 * np.exp(-0.1 * x**4)),
                axes.get_graph(lambda x: 0.3 * (1 / (1 + x**2))),
            )
            graphs.set_stroke(TEAL, 2, 1)

            kw = dict(font_size=24)
            words = [
                Text("Another\nnormal?", **kw),
                Text("Something\nnew?", **kw),
                Text("Maybe this?", **kw),
            ]
            for word in words:
                word.move_to(axes)
                word.align_to(axes.x_axis.get_start(), LEFT)

            curr_graph = graphs[0].copy()
            self.play(
                ShowCreation(curr_graph),
                FadeIn(words[0], 0.5 * UP),
            )
            self.wait()
            for i in range(2):
                self.play(
                    FadeOut(words[i], 0.5 * UP),
                    FadeIn(words[i + 1], 0.5 * UP),
                    Transform(curr_graph, graphs[i + 1])
                )
                self.wait()
            self.wait()
            self.play(
                FadeOut(words[-1]),
                Transform(curr_graph, graphs[0])
            )
            self.wait()
            self.play(FadeOut(curr_graph, run_time=3))
        else:
            self.repeated_samples(
                plots, 10,
                time_between_samples=0.25,
                time_before_fade=1.0,
            )
            # More! Faster!
            self.repeated_samples(
                plots[:3], 100,
                time_between_samples=1 / 30,
                time_before_fade=0.2,
                animate=False
            )

    def get_normal_parameter_labels(self, plot, mean, sigma, font_size=18, color=GREY_A):
        kw = dict(font_size=font_size)
        labels = VGroup(
            Tex(R"\text{Mean} = 0.0", **kw),
            Tex(R"\text{Std. Dev.} = 0.0", **kw),
        )
        for label, value in zip(labels, [mean, sigma]):
            number = label.make_number_changable("0.0")
            number.set_value(value)

        labels.arrange(DOWN, aligned_edge=LEFT)
        labels.move_to(plot, LEFT)
        labels.shift(0.1 * plot.get_height() * DOWN)
        labels.align_to(plot[0].x_axis.get_start(), LEFT)
        labels.set_color(color)

        return labels

    def get_sum_graph(self, axes):
        # Todo, it would be better to directly convolve the first
        # two graphs
        var = scipy.stats.norm(0, np.sqrt(self.sigma1**2 + self.sigma2**2))
        return axes.get_graph(var.pdf, color=self.graph_colors[2])


class IntroAnnotations(SampleTwoNormals):
    annotations = True


class AddTwoGammaDistributions(RepeatedSamplesFromContinuousDistributions):
    dot_fade_factor = 0.75

    def construct(self):
        # Plots
        plots = self.get_plots()
        self.add(plots)

        # Add graph labels
        kw = dict(font_size=30)
        graph_labels = VGroup(
            Tex("e^{-x}", **kw),
            Tex(R"\frac{1}{2} x^2 \cdot e^{-x}", **kw),
            Tex(R"\frac{1}{6} x^3 \cdot e^{-x}", **kw),
        )
        for plot, label, x in zip(plots, graph_labels, [1, 2, 3]):
            axes, graph, var_label = plot
            label.next_to(axes.i2gp(x, graph), UP, SMALL_BUFF)
            label.match_color(graph)
        graph_labels[0].shift(0.3 * UR)

        self.add(graph_labels)

        # Initial samples
        self.repeated_samples(
            plots, 40,
            animate=False,
            time_between_samples=0.1,
            time_before_fade=0.5
        )

        # Graph equation
        frame = self.frame
        fs_rect = FullScreenRectangle()
        fs_rect.set_stroke(GREY_B, 1)
        fs_rect.set_fill(BLACK, 1)
        fuller_rect = FullScreenRectangle()
        fuller_rect.set_fill(GREY_E, 1)
        fuller_rect.scale(3)
        self.add(fuller_rect, fs_rect, *self.mobjects)

        graph_groups = VGroup(*(
            VGroup(plot[1], label).copy()
            for plot, label in zip(plots, graph_labels)
        ))
        graph_groups.generate_target()
        for graph_group in graph_groups.target:
            graph_group[0].stretch(0.5, 0, about_edge=LEFT)
            graph_group[0].set_stroke(width=4)
            graph_group[1].shift(SMALL_BUFF * UP)

        kw = dict(font_size=96)
        lp, rp = parens = Tex("()", **kw)
        parens.stretch(1.5, 1)
        parens.match_height(graph_groups.target[0])
        equation = VGroup(
            lp.copy(), graph_groups.target[0], rp.copy(),
            Tex("*", **kw),
            lp.copy(), graph_groups.target[1], rp.copy(),
            Tex("=", **kw),
            graph_groups.target[2],
        )
        equation.arrange(RIGHT, buff=0.5)
        equation[:3].space_out_submobjects(0.9)
        equation[4:7].space_out_submobjects(0.9)
        equation.next_to(plots, UP, buff=1.5)
        symbols = VGroup(*(
            mob for mob in equation
            if mob not in graph_groups.target
        ))

        self.play(
            frame.animate.set_height(13, about_point = 3 * DOWN),
            FadeIn(fuller_rect),
            FadeIn(fs_rect),
            MoveToTarget(graph_groups, run_time=2),
            Write(symbols, run_time=2),
        )
        self.wait()

        # Label convolution
        conv_label = Text("Convolution", font_size=72)
        arrow = Vector(DOWN)
        arrow.next_to(equation[3], UP)
        conv_label.next_to(arrow, UP)
        VGroup(conv_label, arrow).set_color(YELLOW)
        self.play(Write(conv_label), GrowArrow(arrow))

        # More repeated samples
        self.repeated_samples(
            plots, 50,
            animate=False,
            time_between_samples=0.1,
            time_before_fade=0.5
        )

    def get_axes(self):
        kw = dict(width=5.5, height=2,)
        return VGroup(
            Axes((0, 10), (0, 1.0, 0.25), **kw),
            Axes((0, 10), (0, 0.5, 0.25), **kw),
            Axes((0, 10), (0, 0.5, 0.25), **kw),
        )

    def get_random_variables(self):
        return [
            scipy.stats.gamma(1),
            scipy.stats.gamma(3),
        ]

    def get_sum_graph(self, axes):
        var = scipy.stats.gamma(4)
        return axes.get_graph(
            var.pdf,
            color=self.graph_colors[2]
        )


class SampleWedgePlusDoubleLump(RepeatedSamplesFromContinuousDistributions):
    def construct(self):
        # Plots
        plots = self.get_plots()
        plots[0][1].make_jagged()
        self.add(plots)

        # Initial samples
        self.repeated_samples(
            plots, 50,
            animate=False,
            time_between_samples=0.1,
            time_before_fade=0.5
        )

    def get_axes(self):
        return VGroup(*(
            Axes(
                (-2, 2), (0, 1, 0.25),
                width=5.5,
                height=2.5,
            )
            for x in range(3)
        ))

    def get_samples(self):
        x1 = sum(np.random.uniform(-0.5, 0.5, 2))
        # Hack
        x2 = np.random.normal(0, 0.5)
        x2 += (0.5 if random.random() < 0.3 else -0.5)

        return [x1, x2]

    def get_pdfs(self):
        return [wedge_func, double_lump]


class ContinuousSampleAnnotations(SampleWedgePlusDoubleLump):
    def construct(self):
        plots = self.get_plots()
        plots[0][1].make_jagged()
        # self.add(plots)

        thick_graphs = VGroup(*(
            plot[1].copy().set_stroke(width=10)
            for plot in plots
        ))

        # Labels
        func_labels = VGroup(Tex("f(x)"), Tex("g(y)"))
        func_labels.scale(0.75)
        for graph, label in zip(thick_graphs, func_labels):
            label.next_to(graph.pfp(0.4), LEFT)
            self.play(
                Write(label, time_span=(1, 2)),
                VShowPassingFlash(graph, time_width=1.5, run_time=3),
            )

        # Question
        question = Text("What is this?")
        question.move_to(plots[2][0].get_corner(UL)).shift(0.5 * RIGHT)
        question.set_color(TEAL_A)
        arrow = Arrow(question.get_bottom(), plots[2][1].pfp(0.25), buff=0.2)
        arrow.set_color(TEAL_A)

        self.play(FadeIn(question), GrowArrow(arrow))
        self.wait()


class UniformSamples(RepeatedSamplesFromContinuousDistributions):
    def construct(self):
        # Plots
        plots = self.get_plots()
        funcs = [uniform, uniform, wedge_func]

        for plot, func in zip(plots, funcs):
            axes, graph, label = plot
            axes.y_axis.add_numbers(
                np.arange(0.5, 2.5, 0.5),
                font_size=12,
                buff=0.15,
                num_decimal_places=1,
            )
            new_graph = axes.get_graph(
                func,
                x_range=(-2, 2, 0.01),
                use_smoothing=False
            )
            new_graph.match_style(graph)
            graph.match_points(new_graph)

        plots[2][1].set_opacity(0)
        plots[2][0].y_axis.set_opacity(0)

        self.add(plots)

        # Samples
        self.repeated_samples(
            plots, 50,
            animate=False,
            time_between_samples=0.1,
            time_before_fade=0.5
        )

    def get_axes(self):
        return VGroup(*(
            Axes(
                (-2, 2), (0, 2, 0.5),
                width=5.5,
                height=2,
            )
            for x in range(3)
        ))

    def get_samples(self):
        return np.random.uniform(-0.5, 0.5, 2)

    def get_pdfs(self):
        return [uniform, uniform]


class WedgeAndExpSamples(SampleWedgePlusDoubleLump):
    def get_axes(self):
        return VGroup(
            Axes(
                (-2, 2), (0, 1, 0.25),
                width=5.5,
                height=2,
            ),
            Axes(
                (-2, 5), (0, 1.0, 0.25),
                width=5.5,
                height=2,
            ),
            Axes(
                (-3, 6), (0, 1.0, 0.25),
                width=5.5,
                height=2,
            ),
        )

    def get_random_variables(self):
        return [
            scipy.stats.gamma(1),
            scipy.stats.gamma(3),
        ]

    def get_samples(self):
        wedge_sum = np.random.uniform(-0.5, 0.5, 2).sum()
        exp_value = np.clip(self.random_variables[0].rvs() - 2, -2, 5)
        return [wedge_sum, exp_value]

    def get_pdfs(self):
        return [wedge_func, lambda x: self.random_variables[0].pdf(x + 2)]


# Sliding window view of convolutions


class Convolutions(InteractiveScene):
    axes_config = dict(
        x_range=(-3, 3, 1),
        y_range=(-1, 1, 1.0),
        width=6,
        height=2,
    )
    f_graph_style = dict(stroke_color=BLUE, stroke_width=2)
    g_graph_style = dict(stroke_color=YELLOW, stroke_width=2)
    fg_graph_style = dict(stroke_color=GREEN, stroke_width=4)
    conv_graph_style = dict(stroke_color=TEAL, stroke_width=2)
    f_graph_x_step = 0.1
    g_graph_x_step = 0.1
    f_label_tex = "f(x)"
    g_label_tex = "g(s - x)"
    fg_label_tex = R"f(x) \cdot g(s - x)"
    conv_label_tex = R"[f * g](s) = \int_{-\infty}^\infty f(x) \cdot g(s - x) dx"
    label_config = dict(font_size=36)
    t_color = TEAL
    area_line_dx = 0.05
    jagged_product = True
    jagged_convolution = True
    g_is_rect = False
    conv_y_stretch_factor = 2.0

    def setup(self):
        super().setup()
        if self.g_is_rect:
            k1_tracker = self.k1_tracker = ValueTracker(1)
            k2_tracker = self.k2_tracker = ValueTracker(1)

        # Add axes
        all_axes = self.all_axes = self.get_all_axes()
        f_axes, g_axes, fg_axes, conv_axes = all_axes
        x_min, x_max = self.axes_config["x_range"][:2]

        self.disable_interaction(*all_axes)
        self.add(*all_axes)

        # Add f(x)
        f_graph = self.f_graph = f_axes.get_graph(self.f, x_range=(x_min, x_max, self.f_graph_x_step))
        f_graph.set_style(**self.f_graph_style)
        f_label = self.f_label = self.get_label(self.f_label_tex, f_axes)
        if self.jagged_product:
            f_graph.make_jagged()

        self.add(f_graph)
        self.add(f_label)

        # Add g(s - x)
        self.toggle_selection_mode()  # So triangle is highlighted
        s_indicator = self.s_indicator = ArrowTip().rotate(90 * DEGREES)
        s_indicator.set_height(0.15)
        s_indicator.set_fill(self.t_color, 0.8)
        s_indicator.move_to(g_axes.get_origin(), UP)
        s_indicator.add_updater(lambda m: m.align_to(g_axes.get_origin(), UP))

        def get_s():
            return g_axes.x_axis.p2n(s_indicator.get_center())

        self.get_s = get_s
        g_graph = self.g_graph = g_axes.get_graph(lambda x: 0, x_range=(x_min, x_max, self.g_graph_x_step))
        g_graph.set_style(**self.g_graph_style)
        if self.g_is_rect:
            x_min = g_axes.x_axis.x_min
            x_max = g_axes.x_axis.x_max
            g_graph.add_updater(lambda m: m.set_points_as_corners([
                g_axes.c2p(x, y)
                for s in [get_s()]
                for k1 in [k1_tracker.get_value()]
                for k2 in [k2_tracker.get_value()]
                for x, y in [
                    (x_min, 0), (-0.5 / k1 + s, 0), (-0.5 / k1 + s, k2), (0.5 / k1 + s, k2), (0.5 / k1 + s, 0), (x_max, 0)
                ]
            ]))
        else:
            g_axes.bind_graph_to_func(g_graph, lambda x: self.g(get_s() - x), jagged=self.jagged_product)

        g_label = self.g_label = self.get_label(self.g_label_tex, g_axes)

        s_label = self.s_label = VGroup(*Tex("s = "), DecimalNumber())
        s_label.arrange(RIGHT, buff=SMALL_BUFF)
        s_label.scale(0.5)
        s_label.set_backstroke(width=8)
        s_label.add_updater(lambda m: m.next_to(s_indicator, DOWN, buff=0.15))
        s_label.add_updater(lambda m: m[-1].set_value(get_s()))

        self.add(g_graph)
        self.add(g_label)
        self.add(s_indicator)
        self.add(s_label)

        # Show integral of f(x) * g(s - x)
        def prod_func(x):
            k1 = self.k1_tracker.get_value() if self.g_is_rect else 1
            k2 = self.k2_tracker.get_value() if self.g_is_rect else 1
            return self.f(x) * self.g((get_s() - x) * k1) * k2

        fg_graph = fg_axes.get_graph(lambda x: 0, x_range=(x_min, x_max, self.g_graph_x_step))
        pos_graph = fg_graph.copy()
        neg_graph = fg_graph.copy()
        for graph in f_graph, g_graph, fg_graph, pos_graph, neg_graph:
            self.disable_interaction(graph)
        fg_graph.set_style(**self.fg_graph_style)
        VGroup(pos_graph, neg_graph).set_stroke(width=0)
        pos_graph.set_fill(BLUE, 0.5)
        neg_graph.set_fill(RED, 0.5)

        get_discontinuities = None
        if self.g_is_rect:
            def get_discontinuities():
                k1 = self.k1_tracker.get_value()
                return [get_s() - 0.5 / k1, get_s() + 0.5 / k1]

        kw = dict(
            jagged=self.jagged_product,
            get_discontinuities=get_discontinuities,
        )
        fg_axes.bind_graph_to_func(fg_graph, prod_func, **kw)
        fg_axes.bind_graph_to_func(pos_graph, lambda x: np.clip(prod_func(x), 0, np.inf), **kw)
        fg_axes.bind_graph_to_func(neg_graph, lambda x: np.clip(prod_func(x), -np.inf, 0), **kw)

        self.prod_graphs = VGroup(fg_graph, pos_graph, neg_graph)

        fg_label = self.fg_label = self.get_label(self.fg_label_tex, fg_axes)

        self.add(pos_graph, neg_graph, fg_axes, fg_graph)
        self.add(fg_label)

        # Show convolution
        conv_graph = self.conv_graph = self.get_conv_graph(conv_axes)
        if self.jagged_convolution:
            conv_graph.make_jagged()
        conv_graph.set_style(**self.conv_graph_style)

        graph_dot = self.graph_dot = GlowDot(color=WHITE)
        graph_dot.add_updater(lambda d: d.move_to(conv_graph.quick_point_from_proportion(
            inverse_interpolate(x_min, x_max, get_s())
        )))
        graph_line = self.graph_line = Line(stroke_color=WHITE, stroke_width=1)
        graph_line.add_updater(lambda l: l.put_start_and_end_on(
            graph_dot.get_center(),
            [graph_dot.get_x(), conv_axes.get_y(), 0],
        ))
        self.conv_graph_dot = graph_dot
        self.conv_graph_line = graph_line

        conv_label = self.conv_label = Tex(self.conv_label_tex, **self.label_config)
        conv_label.match_x(conv_axes)
        conv_label.set_y(np.mean([conv_axes.get_y(UP), FRAME_HEIGHT / 2]))

        self.add(conv_graph)
        self.add(graph_dot)
        self.add(graph_line)
        self.add(conv_label)

    def get_all_axes(self):
        all_axes = VGroup(*(Axes(**self.axes_config) for x in range(4)))
        all_axes[:3].arrange(DOWN, buff=0.75)
        all_axes[3].next_to(all_axes[:3], RIGHT, buff=1.5)
        all_axes[3].y_axis.stretch(
            self.conv_y_stretch_factor, 1
        )
        all_axes.to_edge(LEFT)
        all_axes.to_edge(DOWN, buff=0.1)

        for i, axes in enumerate(all_axes):
            x_label = Tex("x" if i < 3 else "s", font_size=24)
            x_label.next_to(axes.x_axis.get_right(), UP, MED_SMALL_BUFF)
            axes.x_label = x_label
            axes.x_axis.add(x_label)
            axes.y_axis.ticks.set_opacity(0)
            axes.x_axis.ticks.stretch(0.5, 1)

        return all_axes

    def get_label(self, tex, axes):
        label = Tex(tex, **self.label_config)
        label.move_to(midpoint(axes.get_origin(), axes.get_right()))
        label.match_y(axes.get_top())
        return label

    def get_conv_graph(self, conv_axes):
        return get_conv_graph(conv_axes, self.f, self.g)

    def get_conv_s_indicator(self):
        g_s_indicator = VGroup(self.s_indicator, self.s_label)
        f_axes, g_axes, fg_axes, conv_axes = self.all_axes

        def get_s():
            return g_axes.x_axis.p2n(self.s_indicator.get_x())

        conv_s_indicator = g_s_indicator.copy()
        conv_s_indicator.add_updater(lambda m: m.become(g_s_indicator))
        conv_s_indicator.add_updater(lambda m: m.shift(
            conv_axes.c2p(get_s(), 0) - g_axes.c2p(get_s(), 0)
        ))
        return conv_s_indicator

    def f(self, x):
        return 0.5 * np.exp(-0.8 * x**2) * (0.5 * x**3 - 3 * x + 1)

    def g(self, x):
        return np.exp(-x**2) * np.sin(2 * x)


class ProbConvolutions(Convolutions):
    jagged_product = True

    def construct(self):
        # Hit most of previous setup
        f_axes, g_axes, fg_axes, conv_axes = self.all_axes
        f_graph, g_graph, prod_graphs, conv_graph = self.f_graph, self.g_graph, self.prod_graphs, self.conv_graph
        f_label, g_label, fg_label, conv_label = self.f_label, self.g_label, self.fg_label, self.conv_label
        s_indicator = self.s_indicator
        s_label = self.s_label
        self.remove(s_indicator, s_label)

        f_axes.x_axis.add_numbers(font_size=16, buff=0.1, excluding=[0])
        self.remove(f_axes, f_graph, f_label)

        y_label = Tex("y").replace(g_axes.x_label)
        g_label.shift(0.2 * UP)
        gy_label = Tex("g(y)", **self.label_config).replace(g_label, dim_to_match=1)
        gmx_label = Tex("g(-x)", **self.label_config).replace(g_label, dim_to_match=1)
        g_axes.x_label.set_opacity(0)
        self.remove(g_axes, g_graph, g_label)

        alt_fg_label = Tex(R"p_X(x) \cdot g(-x)", **self.label_config)
        alt_fg_label.move_to(fg_label)

        conv_label.shift_onto_screen()
        sum_label = Tex("[f * g](s)", **self.label_config)
        sum_label.move_to(conv_label)
        self.remove(fg_axes, prod_graphs, fg_label)
        conv_cover = SurroundingRectangle(conv_axes, buff=0.25)
        conv_cover.set_stroke(width=0)
        conv_cover.set_fill(BLACK, 0.5)
        self.add(conv_cover)

        # Show f
        f_term = conv_label["f(x)"][0]
        f_rect = SurroundingRectangle(f_term)
        f_rect.set_stroke(YELLOW, 2)

        self.play(ShowCreation(f_rect))
        self.play(
            TransformFromCopy(f_term, f_label),
            FadeIn(f_axes),
        )
        self.play(
            ShowCreation(f_graph),
            VShowPassingFlash(f_graph.copy().set_stroke(width=5)),
            run_time=2
        )
        self.wait()

        # Show g
        true_g_graph = g_axes.get_graph(self.g)
        true_g_graph.match_style(g_graph)

        g_term = conv_label["g"][1]
        g_rect = SurroundingRectangle(g_term, buff=0.05)
        g_rect.match_style(f_rect)

        self.play(ReplacementTransform(f_rect, g_rect))
        self.play(
            TransformFromCopy(g_term, gy_label),
            FadeIn(g_axes),
            FadeIn(y_label),
        )
        self.play(
            ShowCreation(true_g_graph),
            VShowPassingFlash(true_g_graph.copy().set_stroke(width=5)),
            run_time=2
        )
        self.wait()

        # Range over pairs of values
        int_rect = SurroundingRectangle(conv_label[re.compile(R"\\int.*")])
        x_rects = VGroup(*(
            SurroundingRectangle(x, buff=0.05)
            for x in conv_label["x"]
        ))
        VGroup(int_rect, *x_rects).match_style(g_rect)

        const_sum = 0.3
        x_tracker = ValueTracker(-1.0)
        y_tracker = ValueTracker()
        x_term = DecimalNumber(include_sign=True, edge_to_fix=RIGHT)
        y_term = DecimalNumber(include_sign=True)
        s_term = DecimalNumber(const_sum)
        equation = VGroup(x_term, y_term, Tex("="), s_term)
        VGroup(x_term, s_term).shift(0.05 * RIGHT)
        equation.arrange(RIGHT, buff=SMALL_BUFF)
        equation.match_x(conv_label)

        x_brace, y_brace, s_brace = braces = VGroup(*(
            Brace(term, UP, SMALL_BUFF)
            for term in [x_term, y_term, s_term]
        ))
        x_brace.add(x_brace.get_tex("x").set_color(BLUE))
        y_brace.add(y_brace.get_tex("y").set_color(YELLOW))
        s_brace.add(s_brace.get_tex("s").set_color(GREY_B))
        y_brace[-1].align_to(x_brace[-1], UP)
        alt_y_label = Tex("s - x")
        alt_y_label.space_out_submobjects(0.8)
        alt_y_label.move_to(y_brace[-1], UP)
        alt_y_label.set_color_by_tex_to_color_map({"s": GREY_B, "x": BLUE})

        def get_x():
            return x_tracker.get_value()

        def get_y():
            return const_sum - get_x()

        f_always(y_tracker.set_value, get_y)
        f_always(x_term.set_value, get_x)
        f_always(y_term.set_value, get_y)

        Axes.get_v_line_to_graph
        x_line = always_redraw(lambda: f_axes.get_v_line_to_graph(
            get_x(), f_graph, line_func=Line, color=WHITE
        ))
        y_line = always_redraw(lambda: g_axes.get_v_line_to_graph(
            get_y(), true_g_graph, line_func=Line, color=WHITE
        ))
        x_dot = GlowDot(color=BLUE)
        y_dot = GlowDot(color=YELLOW)
        f_always(x_dot.move_to, x_line.get_end)
        f_always(y_dot.move_to, y_line.get_end)

        self.play(ReplacementTransform(g_rect, int_rect))
        self.wait()
        self.play(LaggedStart(
            conv_cover.animate.set_opacity(1),
            FadeIn(equation),
            FadeIn(braces),
            VFadeIn(x_line),
            VFadeIn(y_line),
            FadeIn(x_dot),
            FadeIn(y_dot),
        ))
        for x in [1.0, -1.0]:
            self.play(x_tracker.animate.set_value(x), run_time=8)

        self.wait()
        self.remove(int_rect)
        self.play(*(
            ReplacementTransform(int_rect.copy(), x_rect)
            for x_rect in x_rects
        ))
        self.wait()
        self.play(FadeOut(x_rects, lag_ratio=0.5))
        self.play(
            FadeTransform(conv_label["s - x"].copy(), alt_y_label),
            y_brace[-1].animate.set_opacity(0)
        )
        self.remove(alt_y_label)
        y_brace[-1].become(alt_y_label)

        for x in [1.0, -1.0]:
            self.play(x_tracker.animate.set_value(x), run_time=8)

        self.play(LaggedStart(*map(FadeOut, [
            x_line, x_dot, y_line, y_dot,
            *equation, *braces
        ])), lag_ratio=0.2)

        # Flip g
        gsmx_rect = SurroundingRectangle(conv_label["g(s - x)"], buff=0.05)
        gsmx_rect.match_style(g_rect)

        g_axes_copy = g_axes.copy()
        g_axes_copy.add(y_label)
        true_group = VGroup(g_axes_copy, gy_label, true_g_graph)

        self.play(ShowCreation(gsmx_rect))
        self.wait()
        self.play(
            true_group.animate.to_edge(DOWN, buff=MED_SMALL_BUFF),
        )
        self.add(*true_group)
        g_axes.generate_target()
        g_axes.target.x_label.set_opacity(1),
        self.play(
            TransformMatchingShapes(gy_label.copy(), gmx_label),
            true_g_graph.copy().animate.flip().move_to(g_graph).set_anim_args(remover=True),
            MoveToTarget(g_axes),
        )
        self.add(g_graph)
        self.wait()
        self.play(FadeOut(true_group))

        # Show the parameter s
        self.play(
            s_indicator.animate.match_x(g_axes.c2p(2, 0)).set_anim_args(run_time=3),
            VFadeIn(s_indicator),
            VFadeIn(s_label),
            TransformMatchingTex(gmx_label, g_label, run_time=1),
        )
        self.wait()  # Play with the slider
        self.play(
            s_indicator.animate.match_x(g_axes.c2p(0.3, 0))
        )

        # Show product
        fg_rect = SurroundingRectangle(conv_label[R"f(x) \cdot g(s - x)"])
        fg_rect.match_style(g_rect)

        self.play(ReplacementTransform(gsmx_rect, fg_rect))
        self.play(LaggedStart(
            FadeTransform(f_axes.copy(), fg_axes),
            FadeTransform(g_axes.copy(), fg_axes),
            Transform(f_graph.copy(), prod_graphs[0].copy(), remover=True),
            Transform(g_graph.copy(), prod_graphs[0].copy(), remover=True),
            TransformFromCopy(
                VGroup(*f_label, *g_label),
                fg_label
            ),
            FadeOut(fg_rect),
            run_time=2,
        ))
        self.add(*prod_graphs)
        self.play(FadeIn(prod_graphs[1]))
        self.add(prod_graphs)
        # Play with the slider
        self.wait()
        self.play(
            s_indicator.animate.match_x(g_axes.c2p(-0.8, 0))
        )

        # Show convolution
        def get_s():
            return g_axes.x_axis.p2n(s_indicator.get_x())

        conv_s_indicator = self.get_conv_s_indicator()

        self.play(FadeOut(conv_cover))
        self.play(Transform(
            VGroup(indicator, s_label).copy().clear_updaters(),
            conv_s_indicator.copy().clear_updaters(),
            remover=True
        ))
        self.add(conv_s_indicator)
        # Play with the slider
        self.wait()
        self.play(s_indicator.animate.match_x(g_axes.c2p(-0.4, 0)))

    def highlight_several_regions(self, highlighted_xs=None, s=0, reference=None):
        # Highlight a few regions
        if highlighted_xs is None:
            highlighted_xs = np.arange(-1, 1.1, 0.1)

        g_axes = self.all_axes[1]
        highlight_rect = Rectangle(width=0.1, height=FRAME_HEIGHT / 2)
        highlight_rect.set_stroke(width=0)
        highlight_rect.set_fill(TEAL, 0.5)
        highlight_rect.move_to(g_axes.get_origin(), DOWN)
        highlight_rect.set_opacity(0.5)
        self.add(highlight_rect)

        last_label = VMobject()
        for x in highlighted_xs:
            x_tex = f"{{{np.round(x, 1)}}}"
            diff_tex = f"{{{np.round(s - x, 1)}}}"
            label = Tex(
                fR"p_X({x_tex}) \cdot p_Y({diff_tex})",
                tex_to_color_map={diff_tex: YELLOW, x_tex: BLUE},
                font_size=36
            )
            if reference:
                label.next_to(reference, UP, MED_LARGE_BUFF)
            else:
                label.next_to(ORIGIN, DOWN, LARGE_BUFF)

            highlight_rect.set_x(g_axes.c2p(x, 0)[0]),
            self.add(label)
            self.remove(last_label)
            self.wait(0.25)
            last_label = label
        self.play(FadeOut(last_label), FadeOut(highlight_rect))

    def f(self, x):
        return wedge_func(x)

    def g(self, x):
        return double_lump(x)


class ConvolveTwoUniforms(Convolutions):
    jagged_product = True
    jagged_convolution = True
    axes_config = dict(
        x_range=(-2, 2, 0.5),
        y_range=(-1, 1, 1.0),
        width=6,
        height=2,
    )
    f_graph_x_step = 0.025
    g_graph_x_step = 0.025
    conv_y_stretch_factor = 1.0

    def construct(self):
        self.all_axes[0].x_axis.add_numbers(
            font_size=16, num_decimal_places=1,
            excluding=[0], buff=0.1,
        )
        self.g_label.shift(MED_LARGE_BUFF * UP)
        self.fg_label.shift(MED_LARGE_BUFF * UP)
        self.add(self.get_conv_s_indicator())

        # Show it all
        self.wait(60)

    def f(self, x):
        return uniform(x)

    def g(self, x):
        return uniform(x)


class ConvolveUniformWithWedge(Convolutions):
    f_graph_x_step = 0.025
    g_graph_x_step = 0.025

    def construct(self):
        self.conv_graph.shift(0.025 * LEFT)

        # Play around with it
        self.wait(20)

    def f(self, x):
        return wedge_func(x)

    def g(self, x):
        return uniform(x)


class ConvolveTwoNormals(Convolutions):
    def construct(self):
        # Play around with it
        self.wait(20)

    def f(self, x):
        return gauss_func(x, 0, 0.5)

    def g(self, x):
        return gauss_func(x, 0, 0.5)


class ProbConvolutionControlled(ProbConvolutions):
    t_time_pairs = [(-2.5, 4), (2.5, 10), (-1, 6)]
    initial_t = 0

    def construct(self):
        s_indicator = self.s_indicator
        g_axes = self.all_axes[1]

        def set_t(t):
            return s_indicator.animate.set_x(g_axes.c2p(t, 0)[0])

        s_indicator.set_x(g_axes.c2p(self.initial_t, 0)[0])
        for t, time in self.t_time_pairs:
            self.play(set_t(t), run_time=time)


class ProbConvolutionControlledToMatchSlices(ProbConvolutionControlled):
    t_time_pairs = [(-1.5, 20), (1.5, 20), (-0.5, 20)]
    initial_t = 2


class AltSyncedConvolution(ProbConvolutionControlledToMatchSlices):
    t_time_pairs = [(1.0, 15), (-1.5, 10), (-3.0, 10), (1.0, 20)]
    initial_t = -3.0

    def f(self, x):
        return (x > -2) * np.exp(-2 - x)

    def g(self, x):
        return wedge_func(x)


class ThumbnailGraphs(AltSyncedConvolution):
    def construct(self):
        super().construct()
        s_indicator = self.s_indicator
        f_axes, g_axes = self.all_axes[:2]

        for axes in [f_axes, g_axes]:
            axes.set_stroke(width=4)
            axes.x_axis[1].set_stroke(width=0)
        for graph in [self.f_graph, self.g_graph, self.prod_graphs]:
            graph.set_stroke(width=8)

        s = -1.68
        s_indicator.set_x(g_axes.c2p(s, 0)[0])


class AltConvolutions(Convolutions):
    jagged_product = True

    def construct(self):
        s_indicator = self.s_indicator
        g_axes = self.all_axes[1]

        # Sample values
        for t in [3, -3, -1.0]:
            self.play(s_indicator.animate.set_x(g_axes.c2p(t, 0)[0]), run_time=3)
            self.wait()

    def f(self, x):
        if x < -2:
            return -0.5
        elif x < -1:
            return x + 1.5
        elif x < 1:
            return -0.5 * x
        else:
            return 0.5 * x - 1

    def g(self, x):
        return np.exp(-3 * x**2)


class MovingAverageAsConvolution(Convolutions):
    g_graph_x_step = 0.1
    jagged_product = True
    g_is_rect = True

    def construct(self):
        # Setup
        super().construct()
        s_indicator = self.s_indicator
        f_axes, g_axes, fg_axes, conv_axes = self.all_axes
        self.g_label.shift(0.25 * UP)
        self.fg_label.shift(0.25 * UP)

        y_axes = VGroup(*(axes.y_axis for axes in self.all_axes[1:3]))
        fake_ys = y_axes.copy()
        for fake_y in fake_ys:
            fake_y.stretch(1.2, 1)
        self.add(*fake_ys, *self.mobjects)

        conv_axes.y_axis.match_height(f_axes.y_axis)
        VGroup(conv_axes).match_y(f_axes)
        self.conv_graph.match_points(get_conv_graph(conv_axes, self.f, self.g))
        self.conv_label.next_to(conv_axes, DOWN, MED_LARGE_BUFF)

        # Sample values
        def set_t(t):
            return s_indicator.animate.set_x(g_axes.c2p(t, 0)[0])

        self.play(set_t(-2.5), run_time=2)
        self.play(set_t(2.5), run_time=8)
        self.wait()
        self.play(set_t(-1), run_time=3)
        self.wait()

        # Isolate to slice
        top_line, side_line = Line().replicate(2)
        top_line.add_updater(lambda l: l.put_start_and_end_on(*self.g_graph.get_anchors()[4:6]))
        side_line.add_updater(lambda l: l.put_start_and_end_on(*self.g_graph.get_anchors()[2:4]))

        top_line.set_stroke(width=0)
        self.add(top_line)

        left_rect, right_rect = fade_rects = FullScreenFadeRectangle().replicate(2)
        left_rect.add_updater(lambda m: m.set_x(top_line.get_left()[0], RIGHT))
        right_rect.add_updater(lambda m: m.set_x(top_line.get_right()[0], LEFT))

        self.play(FadeIn(fade_rects))
        self.play(set_t(-2), run_time=3)
        self.play(set_t(-0.5), run_time=3)
        self.wait()
        self.play(FadeOut(fade_rects))

        # Show rect dimensions
        get_k1 = self.k1_tracker.get_value
        get_k2 = self.k2_tracker.get_value
        top_label = DecimalNumber(1, font_size=24)
        top_label.add_updater(lambda m: m.set_value(1.0 / get_k1()))
        top_label.add_updater(lambda m: m.next_to(top_line, UP, SMALL_BUFF))
        side_label = DecimalNumber(1, font_size=24)
        side_label.add_updater(lambda m: m.set_value(get_k2()))
        side_label.add_updater(lambda m: m.next_to(side_line, LEFT, SMALL_BUFF))

        def change_ks(k1, k2, run_time=3):
            new_conv_graph = get_conv_graph(
                self.all_axes[3], self.f, lambda x: self.g(k1 * x) * k2,
            )
            new_conv_graph.match_style(self.conv_graph)
            self.play(
                self.k1_tracker.animate.set_value(k1),
                self.k2_tracker.animate.set_value(k2),
                Transform(self.conv_graph, new_conv_graph),
                run_time=run_time
            )

        top_line.set_stroke(WHITE, 3)
        side_line.set_stroke(RED, 3)
        self.play(
            ShowCreation(side_line),
            VFadeIn(side_label)
        )
        self.wait()
        self.play(
            ShowCreation(top_line),
            VFadeIn(top_label),
        )
        self.wait()

        # Change dimensions
        change_ks(0.5, 1)
        self.wait()
        change_ks(0.5, 0.5)
        self.play(set_t(-1.5), run_time=2)
        self.play(set_t(-0.25), run_time=2)
        self.wait()
        change_ks(2, 0.5)
        self.wait()
        change_ks(2, 2)
        self.wait()
        change_ks(4, 4)
        change_ks(1, 1)
        self.play(*map(FadeOut, [top_label, top_line, side_label, side_line]))

        # Show area
        rect = Rectangle()
        rect.set_fill(YELLOW, 0.5)
        rect.set_stroke(width=0)
        rect.set_gloss(1)
        rect.add_updater(lambda m: m.set_width(g_axes.x_axis.unit_size / get_k1(), stretch=True))
        rect.add_updater(lambda m: m.set_height(g_axes.y_axis.unit_size * get_k2(), stretch=True))
        rect.add_updater(lambda m: m.set_x(s_indicator.get_x()))
        rect.add_updater(lambda m: m.set_y(g_axes.get_origin()[1], DOWN))

        area_label = Tex(R"\text{Area } = 1", font_size=36)
        area_label.next_to(rect, UP, MED_LARGE_BUFF)
        area_label.to_edge(LEFT)
        arrow = Arrow(area_label.get_bottom(), rect.get_center())

        avg_label = TexText(R"Average value of\\$f(x)$ in the window", font_size=24)
        avg_label.move_to(area_label, DL)
        shift_value = self.all_axes[2].get_origin() - g_axes.get_origin() + 0.5 * DOWN
        avg_label.shift(shift_value)
        arrow2 = arrow.copy().shift(shift_value)

        self.play(
            Write(area_label, stroke_color=WHITE),
            ShowCreation(arrow),
            FadeIn(rect)
        )
        self.wait()
        self.play(
            FadeIn(avg_label, lag_ratio=0.1),
            ShowCreation(arrow2)
        )
        self.wait()
        for k in [1.4, 0.8, 1.0, 4.0, 10.0, 1.0]:
            change_ks(k, k)
        self.play(*map(FadeOut, [area_label, arrow, avg_label, arrow2]))

        # More ambient variation
        self.play(set_t(-2.5), run_time=3)
        self.play(set_t(2.5), run_time=8)
        self.play(set_t(0), run_time=4)
        change_ks(20, 20)
        self.wait()
        change_ks(10, 10)
        self.wait()
        change_ks(0.2, 0.2, run_time=12)
        self.wait()

    def f(self, x):
        return kinked_function(x)

    def g(self, x):
        return rect_func(x)


class GaussianConvolution(ProbConvolutionControlled):
    jagged_product = True
    t_time_pairs = [(-3.0, 4), (3.0, 10), (-1, 10), (1, 5)]
    conv_y_stretch_factor = 1.0

    def f(self, x):
        return 1.5 * np.exp(-x**2) / np.sqrt(PI)

    def g(self, x):
        return 1.5 * np.exp(-x**2) / np.sqrt(PI)


class GaussConvolutions(Convolutions):
    conv_y_stretch_factor = 1.0

    def construct(self):
        super().construct()

    def f(self, x):
        return np.exp(-x**2)

    def g(self, x):
        return np.exp(-x**2)


class RepeatedConvolution(MovingAverageAsConvolution):
    resolution = 0.01
    n_iterations = 12
    when_to_renormalize = 5
    f_label_tex = "f_1(x)"
    g_label_tex = "f_1(s - x)"
    fg_label_tex = R"f_1(x) \cdot f_1(s - x)"
    conv_label_tex = R"f_2(s) = [f_1 * f_1](s)"
    conv_y_stretch_factor = 1.0
    convolution_creation_time = 5
    pre_rescale_factor = 1.5
    lower_graph_target_area = 1.0

    def construct(self):
        # Clean the board
        dx = self.resolution
        axes1, axes2, axes3, conv_axes = self.all_axes
        g_graph = self.g_graph

        x_min, x_max = axes1.x_range[:2]
        x_samples = np.arange(x_min, x_max + dx, dx)
        f_samples = np.array([self.f(x) for x in x_samples])
        g_samples = np.array([self.g(x) for x in x_samples])

        self.remove(
            self.f_graph,
            self.prod_graphs,
            self.conv_graph,
            self.conv_graph_dot,
            self.conv_graph_line,
        )
        for axes in self.all_axes[:3]:
            axes.x_label.set_opacity(0)

        # New f graph
        f_graph = g_graph.copy()
        f_graph.clear_updaters()
        f_graph.set_stroke(BLUE, 3)
        f_graph.shift(axes1.get_origin() - axes2.get_origin())

        self.add(f_graph)

        # New prod graph
        def update_prod_graph(prod_graph):
            s = self.get_s()
            prod_samples = np.array([
                f_sample * self.g(s - x)
                for f_sample, x in zip(f_samples, x_samples)
            ])
            prod_graph.set_points_as_corners(
                axes3.c2p(x_samples, prod_samples)
            )

        prod_graph = VMobject()
        prod_graph.set_stroke(GREEN, 2)
        prod_graph.set_fill(BLUE_E, 1)
        prod_graph.add_updater(update_prod_graph)

        self.fg_label.shift(0.35 * UP)
        self.g_label.shift(0.35 * UP)

        self.add(prod_graph)
        self.add(self.fg_label)

        # Move convolution axes
        conv_axes.match_y(axes1)
        self.remove(self.conv_label)
        conv_label = self.get_conv_label(2)
        self.conv_label = conv_label
        self.add(conv_label)

        # Show repeated convolutions
        for n in range(1, self.n_iterations + 1):
            conv_samples, conv_graph = self.create_convolution(
                x_samples, f_samples, g_samples, conv_axes,
            )
            if n == self.when_to_renormalize:
                self.add_rescale_arrow()
            if n >= self.when_to_renormalize:
                self.rescale_conv(conv_axes, conv_graph, n)
            self.swap_graphs(f_graph, conv_graph, axes1, conv_axes, n)
            self.swap_labels(n, conv_graph)
            f_samples[:] = conv_samples

    def create_convolution(self, x_samples, f_samples, g_samples, conv_axes):
        # Prepare
        self.set_s(x_samples[0], animate=False)

        conv_samples, conv_graph = self.get_conv(
            x_samples, f_samples, g_samples, conv_axes
        )
        endpoint_dot = GlowDot(color=WHITE)
        endpoint_dot.add_updater(lambda m: m.move_to(conv_graph.get_points()[-1]))

        # Sweep
        self.play(
            self.set_s(x_samples[-1]),
            ShowCreation(conv_graph),
            UpdateFromAlphaFunc(
                endpoint_dot, lambda m, a: m.move_to(conv_graph.get_end()).set_opacity(min(6 * a, 1)),
            ),
            run_time=self.convolution_creation_time,
            rate_func=bezier([0, 0, 1, 1])
        )
        self.play(FadeOut(endpoint_dot, run_time=0.5))

        return conv_samples, conv_graph

    def swap_graphs(self, f_graph, conv_graph, f_axes, conv_axes, n):
        shift_value = f_axes.get_origin() - conv_axes.get_origin()
        conv_axes_copy = conv_axes.deepcopy()

        f_label = self.f_label
        new_f_label = Tex(f"f_{{{n + 1}}}(x)", **self.label_config)
        new_f_label.replace(self.conv_label[:len(new_f_label)])
        new_f_label[-2].set_opacity(0)

        f_group = VGroup(f_axes, f_graph, f_label)
        new_f_graph = conv_graph.copy()
        self.add(conv_axes_copy, new_f_graph)
        anims = [
            Transform(conv_axes_copy, f_axes, remover=True),
            new_f_graph.animate.shift(shift_value).match_style(f_graph),
            FadeOut(f_group, shift_value),
            new_f_label.animate.replace(f_label, dim_to_match=1).set_opacity(1),
        ]
        self.play(LaggedStart(*anims))
        self.remove(new_f_label, new_f_graph)
        f_graph.become(new_f_graph)
        f_label.become(new_f_label)
        self.add(f_axes, f_graph, f_label)

    def swap_labels(self, n, conv_graph):
        # Test
        new_conv_label = self.get_conv_label(n + 2)
        new_conv_label.replace(self.conv_label)
        prod_rhs = self.fg_label[6:]
        new_prod_rhs = Tex(f"f_{{{n + 1}}}(s - x)")
        new_prod_rhs.replace(prod_rhs, dim_to_match=1)

        to_remove = VGroup(
            self.conv_label[f"f_{{{n + 1}}}"],
            self.conv_label[f"f_{{{n}}}"],
            prod_rhs,
        )
        to_add = VGroup(
            new_conv_label[f"f_{{{n + 2}}}"],
            new_conv_label[f"f_{{{n + 1}}}"],
            new_prod_rhs,
        )
        anims = [
            LaggedStartMap(FadeOut, to_remove, shift=0.5 * UP),
            LaggedStartMap(FadeIn, to_add, shift=0.5 * UP),
            FadeOut(conv_graph),
        ]
        if hasattr(self, "rescaled_graphs"):
            anims.append(self.rescaled_graphs.animate.set_stroke(width=0.5, opacity=0.5))
        self.play(*anims, run_time=1)

        self.remove(self.conv_label)
        self.remove(new_prod_rhs)
        self.conv_label = new_conv_label
        prod_rhs.become(new_prod_rhs)
        self.add(self.conv_label)
        self.add(prod_rhs)

    def add_rescale_arrow(self):
        arrow = Vector(1.5 * DOWN)
        label = TexText(R"Rescale so that\\std. dev. = 1", font_size=30)
        label.next_to(arrow)
        self.rescale_label = VGroup(label, arrow)
        self.rescale_label.next_to(self.conv_label, DOWN)

        self.play(
            GrowArrow(arrow),
            FadeIn(label)
        )
        self.add(self.rescale_label)

    def rescale_conv(self, conv_axes, conv_graph, n):
        anims = []
        if not hasattr(self, "rescaled_axes"):
            self.rescaled_axes = conv_axes.copy()
            self.rescaled_axes.next_to(self.rescale_label, DOWN)
            self.rescaled_graphs = VGroup()
            anims.append(TransformFromCopy(conv_axes, self.rescaled_axes))

        factor = self.pre_rescale_factor / math.sqrt(n + 1)
        stretched_graph = conv_graph.copy()
        stretched_graph.stretch(factor, 0)
        stretched_graph.stretch(1 / factor, 1, about_edge=DOWN)

        new_graph = VMobject()
        new_graph.start_new_path(conv_axes.x_axis.get_start())
        new_graph.add_line_to(stretched_graph.get_start())
        new_graph.append_vectorized_mobject(stretched_graph)
        new_graph.add_line_to(conv_axes.x_axis.get_end())
        new_graph.match_style(stretched_graph)
        new_graph.shift(self.rescaled_axes.x_axis.pfp(0.5) - conv_axes.c2p(0, 0))
        area = get_norm(new_graph.get_area_vector())
        new_graph.stretch(self.lower_graph_target_area / area, 1, about_edge=DOWN)

        anims.append(TransformFromCopy(conv_graph, new_graph))
        self.play(*anims)

        self.rescaled_graphs.add(new_graph)
        self.add(self.rescaled_graphs)

    def get_conv(self, x_samples, f_samples, g_samples, axes):
        """
        Returns array of samples and graph
        """
        conv_samples = self.resolution * scipy.signal.fftconvolve(
            f_samples, g_samples, mode='same'
        )
        conv_graph = VMobject()
        conv_graph.set_points_as_corners(axes.c2p(x_samples, conv_samples))
        conv_graph.set_stroke(TEAL, 3)
        return conv_samples, conv_graph

    def get_s(self):
        return self.all_axes[1].x_axis.p2n(self.s_indicator.get_center())

    def set_s(self, s, animate=True):
        if animate:
            mob = self.s_indicator.animate
        else:
            mob = self.s_indicator
        return mob.set_x(self.all_axes[1].c2p(s)[0])

    def get_conv_label(self, n):
        lhs = f"f_{{{n}}}(s)"
        last = f"f_{{{n - 1}}}"
        result = Tex(lhs, "=", R"\big[", last, "*", "f_1", R"\big]", "(s)")
        result.set_height(0.5)
        result.next_to(self.all_axes[3], DOWN, MED_LARGE_BUFF)
        return result

    def f(self, x):
        return uniform(x)


class RepeatedConvolutionDoubleLump(RepeatedConvolution):
    n_iterations = 12
    when_to_renormalize = 1
    g_is_rect = False
    axes_config = dict(
        x_range=(-5, 5, 1),
        y_range=(-1, 1, 1.0),
        width=6,
        height=2,
    )
    pre_rescale_factor = 0.75
    lower_graph_target_area = 0.5

    def f(self, x):
        x *= 1.5
        return 1.5 * 0.69 * (np.exp(-6 * (x - 0.8)**2) + np.exp(-6 * (x + 0.8)**2))

    def g(self, x):
        return self.f(x)


class RepeatedConvolutionExp(RepeatedConvolutionDoubleLump):
    pre_rescale_factor = 1.0
    axes_config = dict(
        x_range=(-10, 10, 1),
        y_range=(-1, 1, 1.0),
        width=6,
        height=2,
    )

    def f(self, x):
        return np.exp(-(x + 1)) * (x > -1)


class RepeatedConvolutionGaussian(RepeatedConvolution):
    g_is_rect = False
    when_to_renormalize = 1
    axes_config = dict(
        x_range=(-7, 7, 1),
        y_range=(-0.5, 0.5, 0.5),
        width=6,
        height=2,
    )
    convolution_creation_time = 2
    pre_rescale_factor = 0.95

    def f(self, x):
        return gauss_func(x, 0,1)

    def g(self, x):
        return gauss_func(x, 0, 1)


# Old rect material

class MovingAverageOfRectFuncs(Convolutions):
    f_graph_x_step = 0.01
    g_graph_x_step = 0.01
    jagged_product = True

    def construct(self):
        super().construct()
        s_indicator = self.s_indicator
        g_axes = self.all_axes[1]
        self.all_axes[3].y_axis.match_height(g_axes.y_axis)
        self.conv_graph.set_height(0.5 * g_axes.y_axis.get_height(), about_edge=DOWN, stretch=True)

        for t in [3, -3, 0]:
            self.play(s_indicator.animate.set_x(g_axes.c2p(t, 0)[0]), run_time=5)
        self.wait()

    def f(self, x):
        return rect_func(x / 2)

    def g(self, x):
        return 1.5 * rect_func(1.5 * x)


class RectConvolutionsNewNotation(MovingAverageOfRectFuncs):
    def construct(self):
        # Setup axes
        x_min, x_max = -1.0, 1.0
        all_axes = axes1, axes2, axes3 = VGroup(*(
            Axes(
                (x_min, x_max, 0.5), (0, 5),
                width=3.75, height=4
            )
            for x in range(3)
        ))
        all_axes.arrange(RIGHT, buff=LARGE_BUFF, aligned_edge=DOWN)
        for axes in all_axes:
            axes.x_axis.add_numbers(font_size=12, num_decimal_places=1)
        axes2.y_axis.add_numbers(font_size=12, num_decimal_places=0, direction=DL, buff=0.05)
        all_axes.move_to(DOWN)

        self.add(all_axes)

        # Prepare convolution graphs
        dx = 0.01
        xs = np.arange(x_min, x_max + dx, dx)
        k_range = list(range(3, 9, 2))
        conv_graphs = self.get_all_convolution_graphs(xs, rect_func(xs), axes3, k_range)
        VGroup(*conv_graphs).set_stroke(TEAL, 3)

        rect_defs = VGroup(
            self.get_rect_func_def(),
            *(self.get_rect_k_def(k) for k in k_range)
        )
        rect_defs.scale(0.75)
        rect_defs.next_to(axes2, UP)
        rect_defs[0][9:].scale(0.7, about_edge=LEFT)
        rect_defs[0].next_to(axes1, UP).shift_onto_screen()

        conv_labels = VGroup(
            Tex(R"\big[\text{rect} * \text{rect}_3\big](x)"),
            Tex(R"\big[\text{rect} * \text{rect}_3 * \text{rect}_5\big](x)"),
            Tex(R"\big[\text{rect} * \text{rect}_3 * \text{rect}_5 * \text{rect}_7 \big](x)"),
        )
        conv_labels.scale(0.75)
        conv_labels.match_x(axes3).match_y(rect_defs)

        # Show rect_1 * rect_3
        rect_graphs = VGroup(*(
            self.get_rect_k_graph(axes2, k)
            for k in [1, *k_range]
        ))
        rect_graphs[0].set_color(BLUE)
        rect_graphs[0].match_x(axes1)

        rect = Rectangle(axes2.x_axis.unit_size / 3, axes2.y_axis.unit_size * 3)
        rect.set_stroke(width=0)
        rect.set_fill(YELLOW, 0.5)
        rect.move_to(axes2.get_origin(), DOWN)

        self.add(*rect_graphs[:2])
        self.add(*rect_defs[:2])
        self.add(conv_graphs[0])

        self.play(FadeIn(rect))
        self.wait()

        self.play(
            Transform(rect_defs[0][:4].copy(), conv_labels[0][0][1:5], remover=True, path_arc=-PI / 3),
            Transform(rect_defs[1][:5].copy(), conv_labels[0][0][6:11], remover=True, path_arc=-PI / 3),
            FadeIn(conv_labels[0][0], lag_ratio=0.1, time_span=(1.5, 2.5)),
            FadeOut(rect),
            run_time=2
        )
        self.wait()

        # Show the rest
        for n in range(2):
            left_graph = rect_graphs[n] if n == 0 else conv_graphs[n - 1]
            lefs_label = rect_defs[n] if n == 0 else conv_labels[n - 1]
            k = 2 * n + 5
            new_rect = Rectangle(axes2.x_axis.unit_size / k, axes2.y_axis.unit_size * k)
            new_rect.set_stroke(width=0)
            new_rect.set_fill(YELLOW, 0.5)
            new_rect.move_to(axes2.get_origin(), DOWN)
            self.play(
                FadeOut(left_graph, 1.5 * LEFT),
                FadeOut(lefs_label, 1.5 * LEFT),
                FadeOut(rect_defs[n + 1]),
                FadeOut(rect_graphs[n + 1]),
                conv_labels[n].animate.match_x(axes1),
                conv_graphs[n].animate.match_x(axes1),
            )
            self.play(
                Write(rect_defs[n + 2], stroke_color=WHITE),
                ShowCreation(rect_graphs[n + 2]),
                FadeIn(new_rect),
                run_time=1,
            )
            self.wait()
            left_conv = conv_labels[n][0][1:-4]
            r = len(left_conv) + 1
            self.play(
                Transform(left_conv.copy(), conv_labels[n + 1][0][1:r], remover=True, path_arc=-PI / 3),
                Transform(rect_defs[2][:5].copy(), conv_labels[n + 1][0][r + 1:r + 6], remover=True, path_arc=-PI / 3),
                FadeIn(conv_labels[n + 1][0], lag_ratio=0.1, time_span=(0.5, 1.5)),
                ShowCreation(conv_graphs[n + 1]),
            )
            self.play(FadeOut(new_rect))
            self.wait()

    def get_rect_k_graph(self, axes, k):
        x_range = axes.x_axis.x_range
        x_range[2] = 1 / k
        return axes.get_graph(
            lambda x: k * rect_func(k * x),
            discontinuities=(-1 / (2 * k), 1 / (2 * k)),
            stroke_color=YELLOW,
            stroke_width=3,
        )

    def get_rect_k_def(self, k):
        return Tex(Rf"\text{{rect}}_{{{k}}}(x) := {k} \cdot \text{{rect}}({k}x)")[0]


class RectConvolutionFacts(InteractiveScene):
    def construct(self):
        # Equations
        equations = VGroup(
            Tex(R"\text{rect}", "(0)", "=", "1.0"),
            Tex(
                R"\big[",
                R"\text{rect}", "*",
                R"\text{rect}_3",
                R"\big]", "(0)", "=", "1.0"
            ),
            Tex(
                R"\big[",
                R"\text{rect}", "*",
                R"\text{rect}_3", "*",
                R"\text{rect}_5",
                R"\big]", "(0)", "=", "1.0"
            ),
            Tex(R"\vdots"),
            Tex(
                R"\big[",
                R"\text{rect}", "*",
                R"\text{rect}_3", "*", R"\cdots", "*",
                R"\text{rect}_{13}",
                R"\big]", "(0)", "=", "1.0"
            ),
            Tex(
                R"\big[",
                R"\text{rect}", "*",
                R"\text{rect}_3", "*", R"\cdots", "*",
                R"\text{rect}_{13}", "*",
                R"\text{rect}_{15}",
                R"\big]", "(0)", "=", SUB_ONE_FACTOR + R"\dots"
            ),
        )

        for eq in equations:
            eq.set_color_by_tex(R"\text{rect}", BLUE)
            eq.set_color_by_tex("_3", TEAL)
            eq.set_color_by_tex("_5", GREEN)
            eq.set_color_by_tex("_{13}", YELLOW)
            eq.set_color_by_tex("_{15}", RED_B)

        equations.arrange(DOWN, buff=0.75, aligned_edge=RIGHT)
        equations[3].match_x(equations[2][-1])
        equations[-1][:-1].align_to(equations[-2][-2], RIGHT)
        equations[-1][-1].next_to(equations[-1][:-1], RIGHT)
        equations.set_width(FRAME_WIDTH - 4)
        equations.center()

        # Show all (largely copy pasted...)
        self.add(equations[0])
        for i in range(4):
            if i < 3:
                src = equations[i].copy()
            else:
                src = equations[i + 1].copy()

            if i < 2:
                target = equations[i + 1]
            elif i == 2:
                target = VGroup(*equations[i + 1], *equations[i + 2])
            else:
                target = equations[i + 2]
            self.play(TransformMatchingTex(src, target))
            self.wait(0.5)

        self.wait()
