from manim_imports_ext import *
from _2022.convolutions.main import *


class ConvolveDiscreteDistributions(InteractiveScene):
    def construct(self):
        # Set up two distributions
        dist1 = np.array([np.exp(-0.25 * (x - 3)**2) for x in range(6)])
        dist2 = np.array([1.0 / (x + 1)**1.2 for x in range(6)])
        for dist in dist1, dist2:
            dist /= dist.sum()

        top_bars = dist_to_bars(dist1, bar_colors=(BLUE_D, TEAL_D))
        low_bars = dist_to_bars(dist2, bar_colors=(RED_D, GOLD_E))
        all_bars = VGroup(top_bars, low_bars)
        all_bars.arrange(DOWN, buff=1.5)
        all_bars.move_to(4.5 * LEFT)

        add_labels_to_bars(top_bars, dist1)
        add_labels_to_bars(low_bars, dist2)

        for bars, color in (top_bars, BLUE_E), (low_bars, RED_E):
            for i, bar in zip(it.count(1), bars):
                die = DieFace(i, fill_color=color, stroke_width=1, dot_color=WHITE)
                die.set_width(bar.get_width() * 0.7)
                die.next_to(bar, DOWN, SMALL_BUFF)
                bar.die = die
                bar.add(die)
                bar.index = i

        # V lines
        v_lines = get_bar_dividing_lines(top_bars)
        VGroup()
        for bar in top_bars:
            v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
            v_line.set_stroke(GREY_C, 1, 0.75)
            v_line.set_x(bar.get_left()[0])
            v_line.set_y(0)
            v_lines.add(v_line)
        v_lines.add(v_lines[-1].copy().set_x(top_bars.get_right()[0]))
        # v_lines.set_stroke(opacity=0)

        # Set up new distribution
        conv_dist = np.convolve(dist1, dist2)
        conv_bars = dist_to_bars(conv_dist, bar_colors=(GREEN_E, YELLOW_E))
        conv_bars.to_edge(RIGHT)

        add_labels_to_bars(conv_bars, conv_dist)

        for n, bar in zip(it.count(2), conv_bars):
            sum_sym = VGroup(
                top_bars[0].die.copy().scale(0.7),
                Tex("+", font_size=16),
                low_bars[0].die.copy().scale(0.7),
                Tex("=", font_size=24).rotate(PI / 2),
                Tex(str(n), font_size=24),
            )
            sum_sym[0].remove(sum_sym[0][1])
            sum_sym[2].remove(sum_sym[2][1])
            sum_sym.arrange(DOWN, buff=SMALL_BUFF)
            sum_sym[:2].shift(0.05 * DOWN)
            sum_sym[:1].shift(0.05 * DOWN)
            sum_sym.next_to(bar, DOWN, buff=SMALL_BUFF)
            bar.add(sum_sym)

        # Dist labels
        plabel_kw = dict(tex_to_color_map={"X": BLUE, "Y": RED})
        PX = MTex("P_X", **plabel_kw)
        PY = MTex("P_Y", **plabel_kw)
        PXY = MTex("P_{X + Y}", **plabel_kw)

        PX.next_to(top_bars.get_corner(UR), DR)
        PY.next_to(low_bars.get_corner(UR), DR)
        PXY.next_to(conv_bars, UP, LARGE_BUFF)

        # Add distributions
        self.play(
            FadeIn(top_bars, lag_ratio=0.1),
            FadeIn(v_lines, lag_ratio=0.2),
            Write(PX),
        )
        self.wait()
        self.play(
            FadeIn(low_bars, lag_ratio=0.1),
            Write(PY),
        )
        self.wait()

        self.play(
            FadeIn(conv_bars),
            FadeTransform(PX.copy(), PXY),
            FadeTransform(PY.copy(), PXY),
        )
        self.wait()

        # March!
        self.play(low_bars.animate.arrange(LEFT, aligned_edge=DOWN, buff=0).move_to(low_bars))

        last_rects = VGroup()
        for n in range(2, 13):
            conv_bars.generate_target()
            conv_bars.target.set_opacity(0.35)
            conv_bars.target[n - 2].set_opacity(1.0)

            self.play(
                get_row_shift(top_bars, low_bars, n),
                MaintainPositionRelativeTo(PY, low_bars),
                FadeOut(last_rects),
                MoveToTarget(conv_bars),
            )
            pairs = get_aligned_pairs(top_bars, low_bars, n)

            label_pairs = VGroup(*(VGroup(m1.value_label, m2.value_label) for m1, m2 in pairs))
            rects = VGroup(*(
                SurroundingRectangle(lp, buff=0.05).set_stroke(YELLOW, 2).round_corners()
                for lp in label_pairs
            ))
            rects.set_stroke(YELLOW, 2)

            self.play(
                FadeIn(rects, lag_ratio=0.5),
                # Restore(bar[0], time_span=(0.5, 1.0)),
                # Write(bar[2], time_span=(0.5, 1.0)),
            )

            self.play(*(
                FadeTransform(label.copy(), conv_bars[n - 2].value_label)
                for lp in label_pairs
                for label in lp
            ))
            self.wait(0.5)

            last_rects = rects

        conv_bars.target.set_opacity(1.0)
        self.play(
            FadeOut(last_rects),
            get_row_shift(top_bars, low_bars, 7),
            MaintainPositionRelativeTo(PY, low_bars),
            MoveToTarget(conv_bars),
        )

        # Emphasize that these are also functions
        func_label = Text("Function", font_size=36)
        func_label.next_to(PX, UP, LARGE_BUFF, aligned_edge=LEFT)
        func_label.shift_onto_screen(buff=SMALL_BUFF)
        arrow = Arrow(func_label, PX.get_top(), buff=0.2)
        VGroup(func_label, arrow).set_color(YELLOW)
        x_args = VGroup(*(
            MTex(
                f"({x}) = {np.round(dist1[x - 1], 2)}"
            ).next_to(PX, RIGHT, SMALL_BUFF)
            for x in range(1, 7)
        ))
        die_rects = VGroup()
        value_rects = VGroup()
        for index, x_arg in enumerate(x_args):
            x_die = top_bars[index].die
            value_label = top_bars[index].value_label
            die_rect = SurroundingRectangle(x_die, buff=SMALL_BUFF)
            value_rect = SurroundingRectangle(value_label, buff=SMALL_BUFF)
            for rect in die_rect, value_rect:
                rect.set_stroke(YELLOW, 2).round_corners()
            die_rects.add(die_rect)
            value_rects.add(value_rect)

        index = 2
        x_arg = x_args[index]
        die_rect = die_rects[index]
        value_rect = value_rects[index]
        x_die = top_bars[index].die
        value_label = top_bars[index].value_label

        self.play(Write(func_label), ShowCreation(arrow))
        self.wait()
        self.play(ShowCreation(die_rect))
        self.play(FadeTransform(x_die.copy(), x_arg[:3]))
        self.play(TransformFromCopy(die_rect, value_rect))
        self.play(FadeTransform(value_label.copy(), x_arg[3:]))
        self.wait()
        for i in range(6):
            self.remove(*die_rects, *value_rects, *x_args)
            self.add(die_rects[i], value_rects[i], x_args[i])
            self.wait(0.5)

        func_group = VGroup(func_label, arrow)
        func_group_copies = VGroup(
            func_group.copy().shift(PXY.get_center() - PX.get_center()),
            func_group.copy().shift(PY.get_center() - PX.get_center()),
        )
        self.play(*(
            TransformFromCopy(func_group, func_group_copy)
            for func_group_copy in func_group_copies
        ))
        self.wait()
        self.play(LaggedStartMap(FadeOut, VGroup(
            func_group, *func_group_copies, die_rects[-1], value_rects[-1], *x_args[-1]
        )))

        # State definition again
        conv_def = MTex(
            R"\big[P_X * P_Y\big](s) = \sum_{x = 1}^6 P_X(x) \cdot P_Y(s - x)",
            font_size=36,
            **plabel_kw,
        )
        conv_def.next_to(conv_bars, UP, buff=MED_LARGE_BUFF)

        PXY.generate_target()
        lhs = conv_def[:10]
        PXY.target.next_to(lhs, UP, LARGE_BUFF).shift_onto_screen(buff=SMALL_BUFF)
        eq = Tex("=").rotate(90 * DEGREES)
        eq.move_to(midpoint(PXY.target.get_bottom(), lhs.get_top()))

        self.play(LaggedStart(
            MoveToTarget(PXY),
            Write(eq),
            TransformFromCopy(PX, lhs[1:3]),
            TransformFromCopy(PY, lhs[4:6]),
            Write(VGroup(lhs[0], lhs[3], *lhs[6:])),
        ))
        self.wait()
        self.play(Write(conv_def[10:]))
        self.wait()


# Continuous case


class TransitionToContinuousProbability(InteractiveScene):
    def construct(self):
        # Setup axes and initial graph
        axes = Axes((0, 12), (0, 1, 0.2), width=14, height=5)
        axes.to_edge(LEFT, LARGE_BUFF)

        def pd(x):
            return (x**4) * np.exp(-x) / 8.0

        graph = axes.get_graph(pd)
        graph.set_stroke(WHITE, 2)
        bars = axes.get_riemann_rectangles(graph, dx=1, x_range=(0, 6), input_sample_type="right")
        bars.set_stroke(WHITE, 3)

        y_label = Text("Probability", font_size=24)
        y_label.next_to(axes.y_axis, UP, SMALL_BUFF)

        self.add(axes)
        self.add(y_label)
        self.add(*bars)

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
        self.play(ShowSubmobjectsOneByOne(all_rects, rate_func=bezier([0, 0, 0, 0, 1, 1]), run_time=5))
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
        x_label.add_updater(lambda m: m.next_to(tip, DOWN, buff=0.2, aligned_edge=LEFT))

        self.play(FadeIn(tip), FadeIn(x_label))
        self.play(x_tracker.animate.set_value(12), run_time=6)
        self.remove(tip, x_label)

        # Labels
        x_label = Text("Value of XYZ next year")
        x_label.next_to(axes.c2p(4, 0), DOWN, buff=0.45)

        density = Text("Probability density")
        density.match_height(y_label)
        density.move_to(y_label, LEFT)
        cross = Cross(y_label)

        self.play(Write(x_label))
        self.wait()
        self.play(ShowCreation(cross))
        self.play(
            VGroup(y_label, cross).animate.shift(0.5 * UP),
            FadeIn(density)
        )
        self.wait()

        # Interpretation
        range_tracker = ValueTracker([0, 12])

        def update_area(area):
            values = range_tracker.get_value()
            x1, x2 = axes.x_axis.n2p(values)[:, 0]
            for bar in area:
                if x1 < bar.get_x() < x2:
                    bar.set_opacity(1)
                else:
                    bar.set_opacity(0.25)

        area.add_updater(update_area)

        v_lines = Line(DOWN, UP).replicate(2)
        v_lines.set_stroke(GREY_A, 1)
        v_lines.set_height(FRAME_HEIGHT)

        def update_v_lines(v_lines):
            values = range_tracker.get_value()
            for value, line in zip(values, v_lines):
                line.move_to(axes.c2p(value, 0), DOWN)

        v_lines.add_updater(update_v_lines)

        self.play(
            range_tracker.animate.set_value([3, 5]),
            VFadeIn(v_lines),
            run_time=2,
        )
        self.wait()
        for pair in [(5, 6), (1, 3), (2.5, 3), (2, 7), (4, 5), (0, 12)]:
            self.play(range_tracker.animate.set_value(pair), run_time=2)
            self.wait()


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
    g_label_tex = "g(t - x)"
    fg_label_tex = R"f(x) \cdot g(t - x)"
    t_color = TEAL
    area_line_dx = 0.05
    jagged_product = True
    g_is_rect = False

    def setup(self):
        super().setup()
        if self.g_is_rect:
            k_tracker = self.k_tracker = ValueTracker(1)

        # Add axes
        all_axes = self.all_axes = self.get_all_axes()
        f_axes, g_axes, fg_axes, conv_axes = all_axes
        x_min, x_max = self.axes_config["x_range"][:2]

        self.disable_interaction(*all_axes)
        self.add(*all_axes)

        # Add f(x)
        f_graph = self.f_graph = f_axes.get_graph(self.f, x_range=(x_min, x_max, self.f_graph_x_step))
        f_graph.set_style(**self.f_graph_style)
        f_label = self.get_label(self.f_label_tex, f_axes)
        if self.jagged_product:
            f_graph.make_jagged()

        self.add(f_graph)
        self.add(f_label)

        # Add g(t - x)
        self.toggle_selection_mode()  # So triangle is highlighted
        t_indicator = self.t_indicator = ArrowTip().rotate(90 * DEGREES)
        t_indicator.set_height(0.15)
        t_indicator.set_fill(self.t_color, 0.8)
        t_indicator.move_to(g_axes.get_origin(), UP)
        t_indicator.add_updater(lambda m: m.align_to(g_axes.get_origin(), UP))

        def get_t():
            return g_axes.x_axis.p2n(t_indicator.get_center())

        g_graph = self.g_graph = g_axes.get_graph(lambda x: 0, x_range=(x_min, x_max, self.g_graph_x_step))
        g_graph.set_style(**self.g_graph_style)
        if self.g_is_rect:
            x_min = g_axes.x_axis.x_min
            x_max = g_axes.x_axis.x_max
            g_graph.add_updater(lambda m: m.set_points_as_corners([
                g_axes.c2p(x, y)
                for t in [get_t()]
                for k in [k_tracker.get_value()]
                for x, y in [
                    (x_min, 0), (-0.5 / k + t, 0), (-0.5 / k + t, k), (0.5 / k + t, k), (0.5 / k + t, 0), (x_max, 0)
                ]
            ]))
        else:
            g_axes.bind_graph_to_func(g_graph, lambda x: self.g(get_t() - x), jagged=self.jagged_product)

        g_label = self.g_label = self.get_label(self.g_label_tex, g_axes)

        t_label = VGroup(*Tex("t = ")[0], DecimalNumber())
        t_label.arrange(RIGHT, buff=SMALL_BUFF)
        t_label.scale(0.5)
        t_label.set_backstroke(width=8)
        t_label.add_updater(lambda m: m.next_to(t_indicator, DOWN, submobject_to_align=m[0], buff=0.15))
        t_label.add_updater(lambda m: m.shift(m.get_width() * LEFT / 2))
        t_label.add_updater(lambda m: m[-1].set_value(get_t()))

        self.add(g_graph)
        self.add(g_label)
        self.add(t_indicator)
        self.add(t_label)

        # Show integral of f(x) * g(t - x)
        def prod_func(x):
            k = self.k_tracker.get_value() if self.g_is_rect else 1
            return self.f(x) * self.g((get_t() - x) * k) * k

        fg_graph, pos_graph, neg_graph = (
            fg_axes.get_graph(lambda x: 0, x_range=(x_min, x_max, self.g_graph_x_step))
            for x in range(3)
        )
        fg_graph.set_style(**self.fg_graph_style)
        VGroup(pos_graph, neg_graph).set_stroke(width=0)
        pos_graph.set_fill(BLUE, 0.5)
        neg_graph.set_fill(RED, 0.5)

        get_discontinuities = None
        if self.g_is_rect:
            def get_discontinuities():
                k = self.k_tracker.get_value()
                return [get_t() - 0.5 / k, get_t() + 0.5 / k]

        kw = dict(
            jagged=self.jagged_product,
            get_discontinuities=get_discontinuities,
        )
        fg_axes.bind_graph_to_func(fg_graph, prod_func, **kw)
        fg_axes.bind_graph_to_func(pos_graph, lambda x: max(prod_func(x), 0), **kw)
        fg_axes.bind_graph_to_func(neg_graph, lambda x: min(prod_func(x), 0), **kw)

        self.prod_graphs = VGroup(fg_graph, pos_graph, neg_graph)

        fg_label = self.fg_label = self.get_label(self.fg_label_tex, fg_axes)

        self.add(pos_graph, neg_graph, fg_axes, fg_graph)
        self.add(fg_label)

        # Show convolution
        conv_graph = self.conv_graph = self.get_conv_graph(conv_axes, self.f, self.g)

        graph_dot = GlowDot(color=WHITE)
        graph_dot.add_updater(lambda d: d.move_to(conv_graph.quick_point_from_proportion(
            inverse_interpolate(x_min, x_max, get_t())
        )))
        graph_line = Line(stroke_color=WHITE, stroke_width=1)
        graph_line.add_updater(lambda l: l.put_start_and_end_on(
            graph_dot.get_center(),
            [graph_dot.get_x(), conv_axes.get_y(), 0],
        ))
        self.conv_graph_dot = graph_dot
        self.conv_graph_line = graph_line

        conv_label = Tex(
            R"(f * g)(t) := \int_{-\infty}^\infty f(x) \cdot g(t - x) dx",
            font_size=36
        )
        conv_label.next_to(conv_axes, UP)

        self.add(conv_graph)
        self.add(graph_dot)
        self.add(graph_line)
        self.add(conv_label)

        # Now play!

    def get_all_axes(self):
        all_axes = VGroup(*(Axes(**self.axes_config) for x in range(4)))
        all_axes[:3].arrange(DOWN, buff=0.75)
        all_axes[3].next_to(all_axes[:3], RIGHT, buff=1.5)
        all_axes[3].y_axis.stretch(2, 1)
        all_axes.to_edge(LEFT)
        all_axes.to_edge(DOWN, buff=0.1)

        for i, axes in enumerate(all_axes):
            x_label = Tex("x" if i < 3 else "t", font_size=24)
            x_label.next_to(axes.x_axis.get_right(), UP, MED_SMALL_BUFF)
            axes.x_label = x_label
            axes.x_axis.add(x_label)
            axes.y_axis.ticks.set_opacity(0)
            axes.x_axis.ticks.stretch(0.5, 1)

        return all_axes

    def get_label(self, tex, axes):
        label = Tex(tex, font_size=36)
        label.move_to(midpoint(axes.get_origin(), axes.get_right()))
        label.match_y(axes.get_top())
        return label

    def get_conv_graph(self, axes, f, g, dx=0.1):
        dx = 0.1
        x_min, x_max = axes.x_range[:2]
        x_samples = np.arange(x_min, x_max + dx, dx)
        f_samples = np.array([f(x) for x in x_samples])
        g_samples = np.array([g(x) for x in x_samples])
        conv_samples = np.convolve(f_samples, g_samples, mode='same')
        conv_graph = VMobject().set_style(**self.conv_graph_style)
        conv_graph.set_points_smoothly(axes.c2p(x_samples, conv_samples * dx))
        return conv_graph

    def f(self, x):
        return 0.5 * np.exp(-0.8 * x**2) * (0.5 * x**3 - 3 * x + 1)

    def g(self, x):
        return np.exp(-x**2) * np.sin(2 * x)


class ProbConvolutions(Convolutions):
    jagged_product = True

    def f(self, x):
        return max(-abs(x) + 1, 0)

    def g(self, x):
        return 0.5 * np.exp(-6 * (x - 0.5)**2) + np.exp(-6 * (x + 0.5)**2)


class ProbConvolutionControlled(ProbConvolutions):
    t_time_pairs = [(-2.5, 4), (2.5, 10), (-1, 6)]
    initial_t = 0

    def construct(self):
        t_indicator = self.t_indicator
        g_axes = self.all_axes[1]

        def set_t(t):
            return t_indicator.animate.set_x(g_axes.c2p(t, 0)[0])

        t_indicator.set_x(g_axes.c2p(self.initial_t, 0)[0])
        for t, time in self.t_time_pairs:
            self.play(set_t(t), run_time=time)
            self.wait()


class ProbConvolutionControlledToMatch3D(ProbConvolutionControlled):
    t_time_pairs = [(1.5, 4), (-0.5, 8), (1.0, 8)]
    initial_t = 0.5


class AltConvolutions(Convolutions):
    jagged_product = True

    def construct(self):
        t_indicator = self.t_indicator
        g_axes = self.all_axes[1]

        # Sample values
        for t in [3, -3, -1.0]:
            self.play(t_indicator.animate.set_x(g_axes.c2p(t, 0)[0]), run_time=3)
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
        t_indicator = self.t_indicator
        g_axes = self.all_axes[1]
        self.g_label.shift(0.25 * UP)

        y_axes = VGroup(*(axes.y_axis for axes in self.all_axes[1:3]))
        fake_ys = y_axes.copy()
        for fake_y in fake_ys:
            fake_y.stretch(1.2, 1)
        self.add(*fake_ys, *self.mobjects)

        # Sample values
        def set_t(t):
            return t_indicator.animate.set_x(g_axes.c2p(t, 0)[0])

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
        get_k = self.k_tracker.get_value
        top_label = DecimalNumber(1, font_size=24)
        top_label.add_updater(lambda m: m.set_value(1 / get_k()))
        top_label.add_updater(lambda m: m.next_to(top_line, UP, SMALL_BUFF))
        side_label = DecimalNumber(1, font_size=24)
        side_label.add_updater(lambda m: m.set_value(get_k()))
        side_label.add_updater(lambda m: m.next_to(side_line, LEFT, SMALL_BUFF))

        def change_k(k, run_time=3):
            new_conv_graph = self.get_conv_graph(
                self.all_axes[3], self.f, lambda x: self.g(k * x) * k,
            )
            self.play(
                self.k_tracker.animate.set_value(k),
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
        change_k(0.5)
        self.wait()
        self.play(set_t(-1.5), run_time=3)
        self.wait()
        change_k(2)
        self.wait()
        change_k(1)
        self.play(*map(FadeOut, [top_label, top_line, side_label, side_line]))

        # Show area
        rect = Rectangle()
        rect.set_fill(YELLOW, 0.5)
        rect.set_stroke(width=0)
        rect.set_gloss(1)
        rect.add_updater(lambda m: m.set_width(g_axes.x_axis.unit_size / get_k(), stretch=True))
        rect.add_updater(lambda m: m.set_height(g_axes.y_axis.unit_size * get_k(), stretch=True))
        rect.add_updater(lambda m: m.set_x(t_indicator.get_x()))
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
        for k in [1.4, 0.8, 1.0]:
            change_k(k)
        self.play(*map(FadeOut, [area_label, arrow, avg_label, arrow2]))

        # Slide once more
        self.play(set_t(-2.5), run_time=3)
        self.play(set_t(2.5), run_time=8)

    def f(self, x):
        return kinked_function(x)

    def g(self, x):
        return rect_func(x)


class GaussianConvolution(Convolutions):
    jagged_product = True

    def f(self, x):
        return np.exp(-x**2) / np.sqrt(PI)

    def g(self, x):
        return np.exp(-x**2) / np.sqrt(PI)


class DiagonalSlices(ProbConvolutions):
    def setup(self):
        InteractiveScene.setup(self)

    def construct(self):
        # Add axes
        frame = self.camera.frame
        axes = self.axes = ThreeDAxes(
            (-2, 2), (-2, 2), (0, 1),
            height=7, width=7, depth=2
        )
        axes.z_axis.apply_depth_test()
        axes.add_axis_labels(z_tex="", font_size=36)
        plane = NumberPlane(
            (-2, 2), (-2, 2), height=7, width=7,
            axis_config=dict(
                stroke_width=1,
                stroke_opacity=0.5,
            ),
            background_line_style=dict(
                stroke_color=GREY_B, stroke_opacity=0.5,
                stroke_width=1,
            )
        )

        self.add(axes, axes.z_axis)
        self.add(plane)

        # Graph
        surface = axes.get_graph(lambda x, y: self.f(x) * self.g(y))
        surface.always_sort_to_camera(self.camera)

        surface_mesh = SurfaceMesh(surface, resolution=(21, 21))
        surface_mesh.set_stroke(WHITE, 0.5, 0.5)

        func_name = Tex(R"f(x) \cdot g(y)")
        func_name.to_corner(UL)
        func_name.fix_in_frame()

        self.add(surface)
        self.add(surface_mesh)
        self.add(func_name)

        # Slicer
        t_tracker = ValueTracker(0.5)
        slice_shadow = self.get_slice_shadow(t_tracker)
        slice_graph = self.get_slice_graph(t_tracker)

        equation = VGroup(MTex("x + y = "), DecimalNumber(color=YELLOW))
        equation[1].next_to(equation[0][-1], RIGHT, buff=0.2)
        equation.to_corner(UR)
        equation.fix_in_frame()
        equation[1].add_updater(lambda m: m.set_value(t_tracker.get_value()))

        set_label = MTex(R"\{(x, t - x): x \in \mathds{R}\}", tex_to_color_map={"t": YELLOW}, font_size=30)
        set_label.next_to(equation, DOWN, MED_LARGE_BUFF, aligned_edge=RIGHT)
        set_label.fix_in_frame()

        self.play(frame.animate.reorient(20, 70), run_time=5)
        self.wait()
        self.play(frame.animate.reorient(0, 0))
        self.wait()

        self.add(slice_shadow, slice_graph, axes.z_axis, axes.axis_labels, plane)
        self.play(
            FadeIn(slice_shadow),
            ShowCreation(slice_graph),
            Write(equation),
            FadeOut(surface_mesh),
            FadeOut(axes.z_axis),
        )
        self.wait()
        self.play(
            FadeIn(set_label, 0.5 * DOWN),
            MoveAlongPath(GlowDot(), slice_graph, run_time=5, remover=True)
        )
        self.wait()
        self.play(frame.animate.reorient(114, 75), run_time=3)
        self.wait()

        # Change t  (Fade out surface mesh?)
        def change_t_anims(t):
            return [
                t_tracker.animate.set_value(t),
                UpdateFromFunc(slice_shadow, lambda m: m.become(self.get_slice_shadow(t_tracker))),
                UpdateFromFunc(slice_graph, lambda m: m.become(self.get_slice_graph(t_tracker))),
            ]

        self.play(*change_t_anims(1.5), run_time=4)
        self.wait()
        self.play(
            *change_t_anims(-0.5),
            frame.animate.reorient(140, 50).set_anim_args(time_span=(0, 4)),
            run_time=8
        )
        self.wait()
        self.play(*change_t_anims(1.0), frame.animate.reorient(99, 77), run_time=8)
        self.wait()

    def get_slice_shadow(self, t_tracker, u_max=5.0, v_range=(-4.0, 4.0)):
        xu = self.axes.x_axis.unit_size
        yu = self.axes.y_axis.unit_size
        zu = self.axes.z_axis.unit_size
        x0, y0, z0 = self.axes.get_origin()
        t = t_tracker.get_value()

        return Surface(
            uv_func=lambda u, v: [
                xu * (u - v) / 2 + x0,
                yu * (u + v) / 2 + y0,
                zu * self.f((u - v) / 2) * self.g((u + v) / 2) + z0 + 2e-2
            ],
            u_range=(t, t + u_max),
            v_range=v_range,
            resolution=(201, 201),
            color=BLACK,
            opacity=1,
            gloss=0,
            reflectiveness=0,
            shadow=0,
        )

    def get_slice_graph(self, t_tracker, color=WHITE, stroke_width=4):
        t = t_tracker.get_value()
        x_min, x_max = self.axes.x_range[:2]
        y_min, y_max = self.axes.y_range[:2]

        if t > 0:
            x_range = (t - y_max, x_max)
        else:
            x_range = (x_min, t - y_min)

        return ParametricCurve(
            lambda x: self.axes.c2p(x, t - x, self.f(x) * self.g(t - x)),
            x_range,
            stroke_color=color,
            stroke_width=stroke_width,
            fill_color=TEAL_D,
            fill_opacity=0.5,
        )


class RepeatedConvolution(MovingAverageAsConvolution):
    resolution = 0.01
    n_iterations = 12

    def construct(self):
        # Clean the board
        dx = self.resolution
        axes1, axes2, axes3, conv_axes = self.all_axes
        conv_axes.y_axis.stretch(1.5 / 2.0, 1)
        g_graph = self.g_graph

        x_min, x_max = axes1.x_range[:2]
        x_samples = np.arange(x_min, x_max + dx, dx)
        f_samples = np.array([self.f(x) for x in x_samples])
        g_samples = np.array([self.g(x) for x in x_samples])

        self.remove(self.f_graph)
        self.remove(self.prod_graphs)
        self.remove(self.conv_graph)
        self.remove(self.conv_graph_dot)
        self.remove(self.conv_graph_line)
        for axes in self.all_axes[:3]:
            axes.x_label.set_opacity(0)

        # New f graph
        f_graph = g_graph.deepcopy()
        f_graph.clear_updaters()
        f_graph.set_stroke(BLUE)
        f_graph.shift(axes1.get_origin() - axes2.get_origin())

        self.add(f_graph)

        # New prod graph
        t_indicator = self.t_indicator

        def get_t():
            return axes2.x_axis.p2n(t_indicator.get_center())

        def set_t(t):
            return t_indicator.animate.set_x(axes2.c2p(t)[0])

        def update_prod_graph(prod_graph):
            prod_samples = f_samples.copy()
            t = get_t()
            prod_samples[x_samples < t - 0.5] = 0
            prod_samples[x_samples > t + 0.5] = 0
            prod_graph.set_points_as_corners(
                axes3.c2p(x_samples, prod_samples)
            )

        prod_graph = VMobject()
        prod_graph.set_stroke(GREEN, 2)
        prod_graph.set_fill(BLUE_E, 1)
        prod_graph.add_updater(update_prod_graph)

        self.add(prod_graph)
        self.add(self.fg_label)

        # Convolution
        conv_samples, conv_graph = self.get_conv(
            x_samples, f_samples, g_samples, conv_axes
        )
        endpoint_dot = GlowDot(color=WHITE)
        endpoint_dot.add_updater(lambda m: m.move_to(conv_graph.get_points()[-1]))

        self.add(conv_graph)

        # Show new convolutions
        for n in range(self.n_iterations):
            t_indicator.set_x(axes2.c2p(-3, 0)[0])
            self.play(
                set_t(3),
                ShowCreation(conv_graph),
                UpdateFromAlphaFunc(
                    endpoint_dot, lambda m, a: m.set_opacity(a),
                    time_span=(0, 0.5),
                ),
                run_time=5,
                rate_func=bezier([0, 0, 1, 1])
            )
            self.play(FadeOut(endpoint_dot))
            shift_value = axes1.get_origin() - conv_axes.get_origin()
            cg_anim = conv_graph.animate.stretch(1 / 1.5, 1, about_point=conv_axes.get_origin())
            cg_anim.shift(shift_value)
            cg_anim.match_style(f_graph)
            self.play(
                cg_anim,
                FadeOut(f_graph, shift_value),
                FadeOut(axes1, shift_value),
                Transform(conv_axes.deepcopy(), axes1, remover=True)
            )
            self.add(axes1, conv_graph)

            f_samples[:] = conv_samples
            f_graph = conv_graph
            conv_samples, conv_graph = self.get_conv(
                x_samples, f_samples, g_samples, conv_axes
            )

    def get_conv(self, x_samples, f_samples, g_samples, axes):
        """
        Returns array of samples and graph
        """
        conv_samples = self.resolution * scipy.signal.fftconvolve(
            f_samples, g_samples, mode='same'
        )
        conv_graph = VMobject().set_points_as_corners(
            axes.c2p(x_samples, conv_samples)
        )
        conv_graph.set_stroke(TEAL, 2)
        return conv_samples, conv_graph

    def f(self, x):
        return rect_func(x)


# Final
class FunctionAverage(InteractiveScene):
    def construct(self):
        # Axes and graph
        def f(x):
            return 0.5 * np.exp(-0.8 * x**2) * (0.5 * x**3 - 3 * x + 1)


# Old rect material


class MovingAverageOfRectFuncs(Convolutions):
    f_graph_x_step = 0.01
    g_graph_x_step = 0.01
    jagged_product = True

    def construct(self):
        super().construct()
        t_indicator = self.t_indicator
        g_axes = self.all_axes[1]
        self.all_axes[3].y_axis.match_height(g_axes.y_axis)
        self.conv_graph.set_height(0.5 * g_axes.y_axis.get_height(), about_edge=DOWN, stretch=True)

        for t in [3, -3, 0]:
            self.play(t_indicator.animate.set_x(g_axes.c2p(t, 0)[0]), run_time=5)
        self.wait()

    def f(self, x):
        return rect_func(x / 2)

    def g(self, x):
        return 1.5 * rect_func(1.5 * x)


class RectConvolutionsNewNotation(MovingAverages):
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
            left_label = rect_defs[n] if n == 0 else conv_labels[n - 1]
            k = 2 * n + 5
            new_rect = Rectangle(axes2.x_axis.unit_size / k, axes2.y_axis.unit_size * k)
            new_rect.set_stroke(width=0)
            new_rect.set_fill(YELLOW, 0.5)
            new_rect.move_to(axes2.get_origin(), DOWN)
            self.play(
                FadeOut(left_graph, 1.5 * LEFT),
                FadeOut(left_label, 1.5 * LEFT),
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
