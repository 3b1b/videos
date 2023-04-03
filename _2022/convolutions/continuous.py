from manim_imports_ext import *
from _2022.convolutions.discrete import *


# Continuous case


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
    conv_label_tex = R"(f * g)(s) = \int_{-\infty}^\infty f(x) \cdot g(s - x) dx"
    label_config = dict(font_size=36)
    t_color = TEAL
    area_line_dx = 0.05
    jagged_product = True
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

        s_label = self.s_label = VGroup(*Tex("s = ")[0], DecimalNumber())
        s_label.arrange(RIGHT, buff=SMALL_BUFF)
        s_label.scale(0.5)
        s_label.set_backstroke(width=8)
        s_label.add_updater(lambda m: m.next_to(s_indicator, DOWN, submobject_to_align=m[0], buff=0.15))
        s_label.add_updater(lambda m: m.shift(m.get_width() * LEFT / 2))
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
        conv_graph = self.conv_graph = self.get_conv_graph(conv_axes, self.f, self.g)

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
    f_label_tex = "p_X(x)"
    g_label_tex = "p_Y(s - x)"
    fg_label_tex = R"p_X(x) \cdot p_Y(s - x)"
    conv_label_tex = R"(p_X * p_Y)(s) := \int_{-\infty}^\infty p_X(x) \cdot p_Y(s - x) dx"
    label_config = dict(
        font_size=36,
        tex_to_color_map={"X": BLUE, "Y": YELLOW}
    )

    def construct(self):
        # Hit most of previous setup
        f_axes, g_axes, fg_axes, conv_axes = self.all_axes
        f_graph, g_graph, prod_graphs, conv_graph = self.f_graph, self.g_graph, self.prod_graphs, self.conv_graph
        f_label, g_label, fg_label, conv_label = self.f_label, self.g_label, self.fg_label, self.conv_label
        s_indicator = self.s_indicator
        s_label = self.s_label
        self.remove(s_indicator, s_label)

        f_axes.x_axis.add_numbers(font_size=16, buff=0.1)
        self.add(f_axes)

        y_label = Tex("y").replace(g_axes.x_label)
        g_label.shift(0.2 * UP)
        gy_label = Tex("p_Y(y)", **self.label_config).replace(g_label, dim_to_match=1)
        gmx_label = Tex("p_Y(-x)", **self.label_config).replace(g_label, dim_to_match=1)
        g_axes.x_label.set_opacity(0)
        self.remove(g_label)
        self.add(y_label)
        self.add(gy_label)

        alt_fg_label = Tex(R"p_X(x) \cdot p_Y(-x)", **self.label_config)
        alt_fg_label.move_to(fg_label)

        conv_label.shift_onto_screen()
        sum_label = Tex("p_{X + Y}(s)", **self.label_config)
        sum_label.move_to(conv_label)
        self.remove(fg_axes, prod_graphs, fg_label)
        self.remove(conv_label)
        self.remove(conv_axes, conv_graph, self.graph_dot, self.graph_line)

        # Show f and g
        true_g_graph = g_graph.copy()
        true_g_graph.clear_updaters()
        true_g_graph.flip()
        true_g_graph.reverse_points()

        self.remove(g_graph, f_graph)
        self.play(LaggedStart(*(
            AnimationGroup(
                ShowCreation(graph),
                VShowPassingFlash(graph.copy().set_stroke(width=5)),
                run_time=2
            )
            for graph in (f_graph, true_g_graph)
        ), lag_ratio=0.25))
        self.wait()
        self.play(
            Transform(f_graph.copy(), self.conv_graph.deepcopy(), remover=True),
            Transform(true_g_graph.copy(), self.conv_graph.deepcopy(), remover=True),
            FadeIn(conv_axes),
            TransformMatchingShapes(VGroup(*f_label, *gy_label).copy(), sum_label),
        )
        self.add(self.conv_graph)

        # Flip g
        right_rect = FullScreenFadeRectangle()
        right_rect.stretch(0.5, 0, about_edge=RIGHT)
        g_axes_copy = g_axes.copy()
        g_axes_copy.add(y_label)
        true_group = VGroup(g_axes_copy, gy_label, true_g_graph)

        self.play(
            true_group.animate.to_edge(DOWN, buff=MED_SMALL_BUFF),
            FadeIn(right_rect)
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

        # Show product
        self.play(
            FadeTransform(f_axes.copy(), fg_axes),
            FadeTransform(g_axes.copy(), fg_axes),
            Transform(f_graph.copy(), prod_graphs[0].copy(), remover=True),
            Transform(g_graph.copy(), prod_graphs[0].copy(), remover=True),
            TransformFromCopy(
                VGroup(*f_label, *gmx_label),
                alt_fg_label
            ),
        )
        self.add(fg_axes, prod_graphs[0])
        self.wait()
        self.add(*prod_graphs)
        self.play(DrawBorderThenFill(prod_graphs[1]))
        self.wait()

        # Show constant sums
        self.highlight_several_regions(reference=alt_fg_label)
        self.play(
            FadeIn(s_indicator), FadeIn(s_label),
            FadeOut(gmx_label, 0.5 * UP),
            FadeIn(g_label, 0.5 * UP),
            FadeTransform(alt_fg_label, fg_label),
        )
        self.wait()
        self.play(s_indicator.animate.match_x(g_axes.c2p(-1, 0)), run_time=2)
        self.highlight_several_regions(s=self.get_s(), reference=fg_label)
        self.wait()

        # Show convolution
        lhs = conv_label[:len("(px*py)(s)")]
        rhs = conv_label[len("(px*py)(s)"):]

        self.play(
            FadeOut(right_rect),
            FadeIn(self.graph_dot),
            FadeIn(self.graph_line),
        )
        self.play(s_indicator.animate.match_x(g_axes.c2p(1.0, 0)), run_time=3)
        self.wait()

        self.play(
            sum_label.animate.move_to(lhs, RIGHT),
            Write(rhs)
        )
        self.wait()
        self.play(
            FlashAround(rhs[6:]),
            FlashAround(fg_label),
            time_width=2.0,
            run_time=3,
        )
        self.wait()

        # Move p_{X + Y}
        equals = Tex("=").rotate(PI / 2)
        equals.next_to(lhs, UP)

        self.play(
            sum_label.animate.next_to(equals, UP, MED_SMALL_BUFF),
            Write(lhs),
            Write(equals),
        )
        self.wait()

        # Slow panning
        for s in [-2, 2]:
            self.play(s_indicator.animate.match_x(g_axes.c2p(s, 0)), run_time=8)
        self.wait()

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
        return max(-abs(x) + 1, 0)

    def g(self, x):
        return 0.5 * np.exp(-6 * (x - 0.5)**2) + np.exp(-6 * (x + 0.5)**2)


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
            self.wait()


class ProbConvolutionControlledToMatch3D(ProbConvolutionControlled):
    t_time_pairs = [(1.5, 4), (-0.5, 8), (1.0, 8)]
    initial_t = 0.5


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
        self.conv_graph.become(self.get_conv_graph(conv_axes, self.f, self.g))
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
            new_conv_graph = self.get_conv_graph(
                self.all_axes[3], self.f, lambda x: self.g(k1 * x) * k2,
            )
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

        func_name = Tex(
            R"p_X(x) \cdot p_Y(y)",
            tex_to_color_map={"X": BLUE, "Y": YELLOW},
            font_size=42,
        )
        func_name.to_corner(UL, buff=0.25)
        func_name.fix_in_frame()

        self.add(surface)
        self.add(surface_mesh)
        self.add(func_name)

        # Slicer
        t_tracker = ValueTracker(0.5)
        slice_shadow = self.get_slice_shadow(t_tracker)
        slice_graph = self.get_slice_graph(t_tracker)

        equation = VGroup(Tex("x + y = "), DecimalNumber(color=YELLOW))
        equation[1].next_to(equation[0][-1], RIGHT, buff=0.2)
        equation.to_corner(UR)
        equation.fix_in_frame()
        equation[1].add_updater(lambda m: m.set_value(t_tracker.get_value()))

        ses_label = Tex(R"\{(x, s - x): x \in \mathds{R}\}", tex_to_color_map={"s": YELLOW}, font_size=30)
        ses_label.next_to(equation, DOWN, MED_LARGE_BUFF, aligned_edge=RIGHT)
        ses_label.fix_in_frame()

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
            FadeIn(ses_label, 0.5 * DOWN),
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
        xu = self.axes.x_axis.get_unit_size()
        yu = self.axes.y_axis.get_unit_size()
        zu = self.axes.z_axis.get_unit_size()
        x0, y0, z0 = self.axes.get_origin()
        t = t_tracker.get_value()

        return ParametricSurface(
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
            shading=(0, 0, 0),
        )

    def get_slice_graph(self, t_tracker, color=WHITE, stroke_width=4):
        t = t_tracker.get_value()
        x_min, x_max = self.axes.x_range[:2]
        y_min, y_max = self.axes.y_range[:2]

        dt = 0.1
        if t > 0:
            x_range = (t - y_max, x_max, dt)
        else:
            x_range = (x_min, t - y_min, dt)

        return ParametricCurve(
            lambda x: self.axes.c2p(x, t - x, self.f(x) * self.g(t - x)),
            x_range,
            stroke_color=color,
            stroke_width=stroke_width,
            fill_color=TEAL_D,
            fill_opacity=0.5,
        )


class GaussianSlices(DiagonalSlices):
    def f(self, x):
        return np.exp(-x**2)

    def g(self, x):
        return np.exp(-x**2)


class RepeatedConvolution(MovingAverageAsConvolution):
    resolution = 0.01
    n_iterations = 8
    when_to_renormalize = 6
    f_label_tex = "f_1(x)"
    g_label_tex = "f_1(s - x)"
    fg_label_tex = R"f_1(x) \cdot f_1(s - x)"
    conv_label_tex = R"f_2(s) = [f_1 * f_1](s)"

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
        f_graph.set_stroke(BLUE, 3)
        f_graph.shift(axes1.get_origin() - axes2.get_origin())

        self.add(f_graph)

        # New prod graph
        def update_prod_graph(prod_graph):
            prod_samples = f_samples.copy()
            s = self.get_s()
            prod_samples[x_samples < s - 0.5] = 0
            prod_samples[x_samples > s + 0.5] = 0
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
        conv_axes.y_axis.match_height(axes1.y_axis)
        conv_axes.match_y(axes1)
        self.remove(self.conv_label)
        conv_label = self.conv_label = self.get_conv_label(2)
        self.add(conv_label)

        # Show repeated convolutions
        self.n = 2
        for x in range(self.n_iterations):
            conv_samples, conv_graph = self.create_convolution(
                x_samples, f_samples, g_samples, conv_axes,
                # TODO, account for renormalized version
            )
            if self.n == self.when_to_renormalize:
                self.renormalize()
            self.swap_graphs(f_graph, conv_graph, axes1, conv_axes)
            self.swap_labels()
            f_samples[:] = conv_samples
            self.n += 1

    def create_convolution(self, x_samples, f_samples, g_samples, conv_axes):
        # Test
        self.set_s(-3, animate=False)

        conv_samples, conv_graph = self.get_conv(
            x_samples, f_samples, g_samples, conv_axes
        )
        endpoint_dot = GlowDot(color=WHITE)
        endpoint_dot.add_updater(lambda m: m.move_to(conv_graph.get_points()[-1]))

        self.play(
            self.set_s(3),
            ShowCreation(conv_graph),
            UpdateFromAlphaFunc(
                endpoint_dot, lambda m, a: m.move_to(conv_graph.get_end()).set_opacity(min(6 * a, 1)),
            ),
            run_time=5,
            rate_func=bezier([0, 0, 1, 1])
        )
        self.play(FadeOut(endpoint_dot))

        return conv_samples, conv_graph

    def swap_graphs(self, f_graph, conv_graph, f_axes, conv_axes):
        shift_value = f_axes.get_origin() - conv_axes.get_origin()
        conv_axes_copy = conv_axes.deepcopy()

        f_label = self.f_label
        new_f_label = Tex(f"f_{{{self.n}}}(x)", **self.label_config)
        new_f_label.replace(self.conv_label[0])
        new_f_label[-2].set_opacity(0)

        f_group = VGroup(f_axes, f_graph, f_label)
        self.add(conv_axes_copy, conv_graph)
        self.play(LaggedStart(
            Transform(conv_axes_copy, f_axes, remover=True),
            conv_graph.animate.shift(shift_value).match_style(f_graph),
            FadeOut(f_group, shift_value),
            new_f_label.animate.replace(f_label, dim_to_match=1).set_opacity(1),
        ))
        self.remove(conv_graph, new_f_label)
        f_graph.become(conv_graph)
        f_label.become(new_f_label)
        self.add(f_axes, f_graph, f_label)

    def swap_labels(self):
        # Test
        new_conv_label = self.get_conv_label(self.n + 1)
        new_conv_label.replace(self.conv_label)
        prod_rhs = self.fg_label[6:]
        new_prod_rhs = Tex(f"f_{{{self.n}}}(s - x)")
        new_prod_rhs.replace(prod_rhs, dim_to_match=1)

        to_remove = VGroup(
            self.conv_label[0],
            self.conv_label[3],
            prod_rhs,
        )
        to_add = VGroup(
            new_conv_label[0],
            new_conv_label[3],
            new_prod_rhs,
        )
        self.play(
            LaggedStartMap(FadeOut, to_remove, shift=0.5 * UP),
            LaggedStartMap(FadeIn, to_add, shift=0.5 * UP),
        )

        self.remove(self.conv_label)
        self.remove(new_prod_rhs)
        self.conv_label = new_conv_label
        prod_rhs.become(new_prod_rhs)
        self.add(self.conv_label)
        self.add(prod_rhs)

    def renormalize(self):
        pass

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
        return rect_func(x)


# Supplements

class AsideOnVariance(InteractiveScene):
    def construct(self):
        pass


class RotateXplusYLine(InteractiveScene):
    def construct(self):
        pass


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
