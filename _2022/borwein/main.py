from manim_imports_ext import *


SUB_ONE_FACTOR = "0.99999999998529"


def sinc(x):
    return np.sinc(x / PI)


def multi_sinc(x, n):
    return np.prod([sinc(x / (2 * k + 1)) for k in range(n)])


def rect_func(x):
    result = np.zeros_like(x)
    result[(-0.5 < x) & (x < 0.5)] = 1.0
    return result


def get_fifteenth_frac_tex():
    return R"{467{,}807{,}924{,}713{,}440{,}738{,}696{,}537{,}864{,}469 \over 467{,}807{,}924{,}720{,}320{,}453{,}655{,}260{,}875{,}000}"


class ShowIntegrals(InteractiveScene):
    add_axis_labels = True

    def construct(self):
        # Setup axes
        axes = self.get_axes()

        graph = axes.get_graph(sinc, color=BLUE)
        graph.set_stroke(width=3)
        points = graph.get_anchors()
        right_sinc = VMobject().set_points_smoothly(points[len(points) // 2:])
        left_sinc = VMobject().set_points_smoothly(points[:len(points) // 2]).reverse_points()
        VGroup(left_sinc, right_sinc).match_style(graph).make_jagged()

        func_label = MTex(R"{\sin(x) \over x}")
        func_label.move_to(axes, UP).to_edge(LEFT)

        self.add(axes)
        self.play(
            Write(func_label),
            ShowCreation(right_sinc, remover=True, run_time=3),
            ShowCreation(left_sinc, remover=True, run_time=3),
        )
        self.add(graph)
        self.wait()

        # Discuss sinc function?
        sinc_label = Tex(R"\text{sinc}(x)")
        sinc_label.next_to(func_label, UR, buff=LARGE_BUFF)
        arrow = Arrow(func_label, sinc_label)

        one_over_x_graph = axes.get_graph(lambda x: 1 / x, x_range=(0.1, 8 * PI))
        one_over_x_graph.set_stroke(YELLOW, 2)
        one_over_x_label = Tex("1 / x")
        one_over_x_label.next_to(axes.i2gp(1, one_over_x_graph), RIGHT)
        sine_wave = axes.get_graph(np.sin, x_range=(0, 8 * PI)).set_stroke(TEAL, 3)
        half_sinc = axes.get_graph(sinc, x_range=(0, 8 * PI)).set_stroke(BLUE, 3)

        self.play(
            GrowArrow(arrow),
            FadeIn(sinc_label, UR)
        )
        self.wait()

        self.play(
            ShowCreation(sine_wave, run_time=2),
            graph.animate.set_stroke(width=1, opacity=0.5)
        )
        self.wait()
        self.play(
            ShowCreation(one_over_x_graph),
            FadeIn(one_over_x_label),
            Transform(sine_wave, half_sinc),
        )
        self.wait()
        self.play(
            FadeOut(one_over_x_graph),
            FadeOut(one_over_x_label),
            FadeOut(sine_wave),
            graph.animate.set_stroke(width=3, opacity=1),
        )

        # At 0
        hole = Dot()
        hole.set_stroke(BLUE, 2)
        hole.set_fill(BLACK, 1)
        hole.move_to(axes.c2p(0, 1))

        zero_eq = Tex(R"{\sin(0) \over 0} = ???")
        zero_eq.next_to(hole, UR)
        lim = Tex(R"\lim_{x \to 0} {\sin(x) \over x} = 1")
        lim.move_to(zero_eq, LEFT)
        x_tracker = ValueTracker(1.5 * PI)
        get_x = x_tracker.get_value
        dots = GlowDot().replicate(2)
        globals().update(locals())
        dots.add_updater(lambda d: d[0].move_to(axes.i2gp(-get_x(), graph)))
        dots.add_updater(lambda d: d[1].move_to(axes.i2gp(get_x(), graph)))
        dots.update()

        self.play(Write(zero_eq), FadeIn(hole, scale=0.35))
        self.wait()
        self.play(FadeTransform(zero_eq, lim))
        self.add(dots)
        self.play(
            x_tracker.animate.set_value(0).set_anim_args(run_time=2),
            UpdateFromAlphaFunc(dots, lambda m, a: m.set_opacity(a)),
        )
        self.wait()
        self.play(FadeOut(dots), FadeOut(hole), FadeOut(lim))
        self.play(FadeOut(arrow), FadeOut(sinc_label))

        # Area under curve
        area = self.get_area(axes, graph)
        int1 = self.get_integral(1)
        rhs = Tex(R"=\pi")
        rhs.next_to(int1, RIGHT)

        origin = axes.get_origin()
        pos_area = VGroup(*(r for r in area if r.get_center()[1] > origin[1]))
        neg_area = VGroup(*(r for r in area if r.get_center()[1] < origin[1]))

        self.play(
            Write(area, stroke_color=WHITE, lag_ratio=0.01),
            ReplacementTransform(func_label, int1[1]),
            Write(int1[::2]),
            graph.animate.set_stroke(width=1)
        )
        self.add(int1)
        self.wait()
        self.play(FadeOut(neg_area))
        self.wait()
        self.play(FadeIn(neg_area), FadeOut(pos_area))
        self.wait()
        self.play(FadeIn(pos_area))
        self.add(area)
        self.wait()
        self.play(Write(rhs))
        self.play(FlashAround(rhs, run_time=2))
        self.wait()

        # Show sin(x / 3) / (x / 3)
        frame = self.camera.frame
        midpoint = 1.5 * DOWN
        top_group = VGroup(axes, graph, area)
        top_group.generate_target().next_to(midpoint, UP)
        low_axes = self.get_axes(height=2.5)
        low_axes.next_to(midpoint, DOWN, buff=MED_LARGE_BUFF)
        low_sinc = low_axes.get_graph(sinc)
        low_sinc.set_stroke(BLUE, 2)
        low_graph = low_axes.get_graph(lambda x: sinc(x / 3))
        low_graph.set_stroke(WHITE, 2)
        low_label = Tex(R"{\sin(x / 3) \over x / 3}")
        low_label.match_x(int1[1]).shift(1.75 * LEFT).match_y(low_axes.get_top())

        self.play(
            MoveToTarget(top_group),
            FadeTransform(axes.copy(), low_axes),
            FadeIn(low_label),
            frame.animate.set_height(10),
            VGroup(int1, rhs).animate.shift(UP + 1.75 * LEFT),
        )
        self.wait()
        self.play(TransformFromCopy(graph, low_sinc))
        self.play(low_sinc.animate.stretch(3, 0).match_style(low_graph), run_time=2)
        self.remove(low_sinc)
        self.add(low_graph)
        self.wait()

        # Sequence of integrals
        curr_int = int1
        for n in range(2, 9):
            new_int = self.get_integral(n)
            if n == 4:
                new_int.scale(0.9)
            elif n == 5:
                new_int.scale(0.8)
            elif n > 5:
                new_int.scale(0.7)
            new_int.move_to(curr_int, LEFT)
            if n < 8:  # TODO
                new_rhs = MTex(R"=\pi")
            else:
                new_rhs = MTex(R"\approx \pi - 4.62 \times 10^{-11}")
            new_rhs.next_to(new_int, RIGHT)

            new_graph = axes.get_graph(lambda x, n=n: multi_sinc(x, n))
            new_graph.set_stroke(BLUE, 2)
            new_area = self.get_area(axes, new_graph)

            self.play(
                ReplacementTransform(curr_int[:n], new_int[:n]),
                TransformFromCopy(low_label[0], new_int[n]),
                ReplacementTransform(curr_int[-1], new_int[-1]),
                FadeOut(rhs),
                FadeOut(area),
                ReplacementTransform(graph, new_graph),
                TransformFromCopy(low_graph, new_graph.copy(), remover=True),
            )
            self.play(Write(new_area, stroke_color=WHITE, stroke_width=1, lag_ratio=0.01))
            self.wait(0.25)
            self.play(FadeIn(new_rhs, scale=0.7))
            self.play(FlashAround(new_rhs))
            self.wait()

            if n < 8:
                new_low_graph = low_axes.get_graph(lambda x, n=n: sinc(x / (2 * n + 1)))
                new_low_graph.match_style(low_graph)
                new_low_label = Tex(fR"{{\sin(x / {2 * n + 1}) \over x / {2 * n + 1}}}")
                new_low_label.move_to(low_label)
                if n == 4:
                    new_low_label.shift(0.5 * UP)

                self.play(
                    FadeOut(low_label, UP),
                    FadeIn(new_low_label, UP),
                    ReplacementTransform(low_graph, new_low_graph),
                )
                self.wait(0.25)

            curr_int = new_int
            rhs = new_rhs
            graph = new_graph
            area = new_area
            low_graph = new_low_graph
            low_label = new_low_label

        # More accurate rhs
        new_rhs = Tex(Rf"= {get_fifteenth_frac_tex()} \pi")
        new_rhs.next_to(curr_int, DOWN, buff=LARGE_BUFF)

        self.play(
            FadeOut(VGroup(low_axes, low_graph, low_label), DOWN),
            VGroup(axes, graph, area).animate.shift(2 * DOWN),
            Write(new_rhs)
        )
        self.wait()

    def get_axes(self,
                 x_range=(-10 * PI, 10 * PI, PI),
                 y_range=(-0.5, 1, 0.5),
                 width=1.3 * FRAME_WIDTH,
                 height=3.5,
                 ):
        axes = Axes(x_range, y_range, width=width, height=height)
        axes.center()
        if self.add_axis_labels:
            axes.y_axis.add_numbers(num_decimal_places=1, font_size=20)
            for u in -1, 1:
                axes.x_axis.add_numbers(
                    u * np.arange(PI, 15 * PI, PI),
                    unit=PI,
                    unit_tex=R"\pi",
                    font_size=20
                )
        return axes

    def get_area(self, axes, graph, dx=0.01 * PI, fill_opacity=0.5):
        rects = axes.get_riemann_rectangles(
            graph, dx=dx,
            colors=(BLUE, BLUE),
            negative_color=RED,
        )
        rects.set_fill(opacity=fill_opacity)
        rects.set_stroke(width=0)
        rects.sort(lambda p: abs(p[0]))
        return rects

    def get_integral(self, n):
        return Tex(
            R"\int_{-\infty}^\infty",
            R"{\sin(x) \over x}",
            *(
                Rf"{{\sin(x/{k}) \over x / {k} }}"
                for k in range(3, 2 * n + 1, 2)
            ),
            "dx"
        ).to_corner(UL)


class WriteOutIntegrals(InteractiveScene):
    def construct(self):
        # Integrals
        ints = self.get_integrals()

        ints.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=RIGHT)
        ints.set_height(FRAME_HEIGHT - 1)
        ints[-1][:-2].align_to(ints[:-1], RIGHT)
        ints[-1][-2:].next_to(ints[-1][:-2], RIGHT, SMALL_BUFF)
        ints[3].shift(SMALL_BUFF * LEFT)
        ints.center()

        for integral in ints:
            integral.set_color_by_tex("\sin", BLUE)
            integral.set_color_by_tex("x/3", TEAL)
            integral.set_color_by_tex("x/5", GREEN)
            integral.set_color_by_tex("x/13", YELLOW)
            integral.set_color_by_tex("x/15", RED_B)

        # Show all
        self.add(ints[0])
        ints[-1][-2].set_opacity(0)
        ints[-1][-1].save_state()
        ints[-1][-1].move_to(ints[-1][-2], LEFT)
        for i in range(4):
            if i < 3:
                src = ints[i].copy()
            else:
                src = ints[i + 1].copy()

            if i < 2:
                target = ints[i + 1]
            elif i == 2:
                target = VGroup(*ints[i + 1], *ints[i + 2])
            else:
                target = ints[i + 2]
            self.play(TransformMatchingTex(src, target))
            self.wait(0.5)

        ints[-1][-2].set_opacity(1)
        self.play(Write(ints[-1][-2]), ints[-1][-1].animate.restore())
        self.wait()

    def get_integrals(self):
        return VGroup(
            Tex(
                R"\int_{-\infty}^\infty",
                R"\frac{\sin(x)}{x}",
                R"dx = ", R"\pi"
            ),
            Tex(
                R"\int_{-\infty}^\infty",
                R"\frac{\sin(x)}{x}",
                R"\frac{\sin(x/3)}{x/3}",
                "dx = ", R"\pi"
            ),
            Tex(
                R"\int_{-\infty}^\infty",
                R"\frac{\sin(x)}{x}",
                R"\frac{\sin(x/3)}{x/3}",
                R"\frac{\sin(x/5)}{x/5}",
                "dx = ", R"\pi"
            ),
            Tex(R"\vdots"),
            Tex(
                R"\int_{-\infty}^\infty",
                R"\frac{\sin(x)}{x}",
                R"\frac{\sin(x/3)}{x/3}",
                R"\frac{\sin(x/5)}{x/5}",
                R"\dots",
                R"\frac{\sin(x/13)}{x/13}",
                "dx = ", R"\pi"
            ),
            Tex(
                R"\int_{-\infty}^\infty",
                R"\frac{\sin(x)}{x}",
                R"\frac{\sin(x/3)}{x/3}",
                R"\frac{\sin(x/5)}{x/5}",
                R"\dots",
                R"\frac{\sin(x/13)}{x/13}",
                R"\frac{\sin(x/15)}{x/15}",
                "dx = ", fR"({SUB_ONE_FACTOR}\dots)", R"\pi",
            ),
        )


class WriteOutIntegralsWithPi(WriteOutIntegrals):
    def get_integrals(self):
        result = VGroup(
            Tex(
                R"\int_{-\infty}^\infty",
                R"\frac{\sin(\pi x)}{\pi x}",
                R"dx = ", "1.0"
            ),
            Tex(
                R"\int_{-\infty}^\infty",
                R"\frac{\sin(\pi x)}{\pi x}",
                R"\frac{\sin(\pi x/3)}{\pi x/3}",
                R"dx = ", "1.0"
            ),
            Tex(
                R"\int_{-\infty}^\infty",
                R"\frac{\sin(\pi x)}{\pi x}",
                R"\frac{\sin(\pi x/3)}{\pi x/3}",
                R"\frac{\sin(\pi x/5)}{\pi x/5}",
                R"dx = ", "1.0"
            ),
            Tex("\vdots"),
            Tex(
                R"\int_{-\infty}^\infty",
                R"\frac{\sin(\pi x)}{\pi x}",
                R"\frac{\sin(\pi x/3)}{\pi x/3}",
                R"\frac{\sin(\pi x/5)}{\pi x/5}",
                R"\dots",
                R"\frac{\sin(\pi x/13)}{\pi x/13}",
                R"dx = ", "1.0"
            ),
            Tex(
                R"\int_{-\infty}^\infty",
                R"\frac{\sin(\pi x)}{\pi x}",
                R"\frac{\sin(\pi x/3)}{\pi x/3}",
                R"\frac{\sin(\pi x/5)}{\pi x/5}",
                R"\dots",
                R"\frac{\sin(\pi x/13)}{\pi x/13}",
                R"\frac{\sin(\pi x/15)}{\pi x/15}",
                R"dx = ", fR"{SUB_ONE_FACTOR}\dots", ".",
            ),
        )
        result[-1][-1].scale(0)
        return result


class MovingAverages(InteractiveScene):
    sample_resolution = 1000
    graph_color = BLUE
    window_color = GREEN
    window_opacity = 0.35
    n_iterations = 8
    rect_scalar = 1.0
    quick_transitions = False

    def construct(self):
        # Add rect graph
        axes = self.get_axes()
        rs = self.rect_scalar
        rect_graph = axes.get_graph(
            lambda x: rect_func(x / rs),
            discontinuities=[-0.5 * rs, 0.5 * rs]
        )
        rect_graph.set_stroke(self.graph_color, 3)
        rect_def = self.get_rect_func_def()
        rect_def.to_corner(UR)
        rect_graph.axes = axes

        drawing_dot = GlowDot()
        drawing_dot.path = rect_graph
        drawing_dot.add_updater(lambda m: m.move_to(m.path.pfp(1)))

        self.add(axes)
        self.add(rect_def)
        self.add(drawing_dot)
        self.play(ShowCreation(rect_graph), run_time=3, rate_func=linear)
        self.play(FadeOut(drawing_dot))
        self.wait()

        # Add f1 label
        f1_label = Tex("f_1(x) = ")
        f1_label.next_to(rect_def, LEFT, buff=0.2)

        self.play(Write(f1_label, stroke_color=WHITE))
        self.wait()

        f1_label_group = VGroup(f1_label, rect_def)
        rect_graph.func_labels = f1_label_group

        # Make room for new graph
        low_axes = self.get_axes().to_edge(DOWN, buff=MED_SMALL_BUFF)

        self.play(
            f1_label_group.animate.set_height(0.7, about_edge=UR),
            VGroup(axes, rect_graph).animate.to_edge(UP),
            TransformFromCopy(axes, low_axes)
        )
        self.wait()

        # Create all convolutions
        sample_resolution = self.sample_resolution
        xs = np.linspace(-rs, rs, int(2 * rs * sample_resolution + 1))
        k_range = list(range(3, self.n_iterations * 2 + 1, 2))
        graphs = [rect_graph, *self.get_all_convolution_graphs(
            xs, rect_func(xs / rs), low_axes, k_range
        )]
        for graph in graphs:
            graph.match_style(rect_graph)

        # Show all convolutions
        prev_graph = rect_graph
        prev_graph_label = f1_label_group

        for n in range(1, self.n_iterations):
            anim_time = (0.01 if self.quick_transitions else 1.0)
            # Show moving average, first with a pass, then with time to play
            window_group = self.show_moving_average(axes, low_axes, graphs[n], k_range[n - 1])

            # Label low graph
            fn_label = Tex(f"f_{{{n + 1}}}(x)", font_size=36)
            fn_label.move_to(low_axes.c2p(0.5 * rs, 1), UL)
            fn0_label = Tex(f"f_{{{n + 1}}}(0) = " + ("1.0" if n < self.n_iterations - 1 else "0.9999999999852937..."), font_size=36)
            fn0_label.next_to(fn_label, DOWN, aligned_edge=LEFT)

            self.play(Write(fn_label, stroke_color=WHITE), run_time=anim_time)
            self.wait(anim_time)

            # Note plateau width, color plateau
            width = max(rs - sum((1 / k for k in range(3, 2 * n + 3, 2))), 0)
            plateau = Line(low_axes.c2p(-width / 2, 1), low_axes.c2p(width / 2, 1))
            if width < 0:
                plateau.set_width(0)
            plateau.set_stroke(YELLOW, 3)
            brace = Brace(plateau, UP)
            brace_label = brace.get_tex(self.get_brace_label_tex(n))
            brace_label.scale(0.75, about_edge=DOWN)

            self.play(
                GrowFromCenter(brace),
                GrowFromCenter(plateau),
                Write(brace_label, stroke_color=WHITE),
                run_time=anim_time,
            )
            graphs[n].add(plateau)
            self.wait(anim_time)
            self.play(FadeIn(fn0_label, 0.5 * DOWN), run_time=anim_time)
            self.wait()

            # Remove window
            self.play(LaggedStartMap(FadeOut, window_group), run_time=anim_time)

            # Put lower graph up top
            new_low_axes = self.get_axes()
            new_low_axes.move_to(low_axes)
            shift_value = axes.c2p(0, 0) - low_axes.c2p(0, 0)

            if n < self.n_iterations - 1:
                anims = [
                    FadeOut(mob, shift_value)
                    for mob in [axes, prev_graph, prev_graph_label]
                ]
                anims.append(FadeIn(new_low_axes, shift_value))
            else:
                shift_value = ORIGIN
                anims = []

            anims.extend([
                *(
                    mob.animate.shift(shift_value)
                    for mob in [low_axes, graphs[n], fn_label, fn0_label]
                ),
                FadeOut(VGroup(brace, brace_label), shift_value),
            ])

            self.play(*anims)

            prev_graph = graphs[n]
            prev_graph_label = VGroup(fn_label, fn0_label)
            axes = low_axes
            low_axes = new_low_axes

            graphs[n].axes = axes
            graphs[n].func_labels = prev_graph_label

        # Show all graphs together
        graphs[0].add(Line(
            graphs[0].axes.c2p(-0.5, 1),
            graphs[0].axes.c2p(0.5, 1),
            stroke_color=YELLOW,
            stroke_width=3.0,
        ))
        graph_groups = VGroup(*(
            VGroup(graph.axes, graph, graph.func_labels[1])
            for graph in graphs[:8]
        ))
        graph_groups.save_state()
        graph_groups.generate_target()
        graph_groups[:-2].set_opacity(0)
        graph_groups.target.arrange_in_grid(
            4, 2, v_buff=2.0, h_buff=1.0,
            aligned_edge=LEFT,
            fill_rows_first=False
        )
        for group in graph_groups.target:
            group[2].scale(2, about_edge=DL)

        graph_groups.target.set_height(FRAME_HEIGHT - 1)
        graph_groups.target.center().to_edge(LEFT)

        self.play(
            MoveToTarget(graph_groups),
            FadeOut(graphs[7].func_labels[0]),
            FadeOut(graphs[6].func_labels[0]),
        )
        self.wait()

    def get_all_convolution_graphs(self, xs, ys, axes, k_range):
        result = []
        func_samples = [ys]
        for k in k_range:
            kernel = rect_func(k * xs)
            kernel /= sum(kernel)
            func_samples.append(np.convolve(
                func_samples[-1],  # Last function
                kernel,
                mode='same'
            ))
            new_graph = VMobject()
            new_graph.set_points_smoothly(axes.c2p(xs, func_samples[-1]))
            new_graph.origin = axes.c2p(0, 0)
            result.append(new_graph)
        result[0].make_jagged()
        return result

    def show_moving_average(self, top_axes, low_axes, low_graph, k):
        rs = self.rect_scalar
        window = Rectangle(
            width=top_axes.x_axis.unit_size / k,
            height=top_axes.y_axis.unit_size * 1.5,
        )
        window.set_stroke(width=0)
        window.set_fill(self.window_color, self.window_opacity)
        window.move_to(top_axes.c2p(-rs, 0), DOWN)
        arrow_buff = min(SMALL_BUFF, window.get_width() / 2)
        arrow = Arrow(window.get_left(), window.get_right(), stroke_width=4, buff=arrow_buff)
        arrows = VGroup(arrow, arrow.copy().flip())
        arrows[0].shift(0.5 * arrow_buff * RIGHT)
        arrows[1].shift(0.5 * arrow_buff * LEFT)
        arrows.next_to(window.get_bottom(), UP, buff=0.5)
        width_label = Tex(f"1 / {k}", font_size=24)
        width_label.set_backstroke(width=4)
        width_label.set_max_width(min(arrows.get_width(), 0.5))
        width_label.next_to(arrows, UP)
        window.add(arrows, width_label)
        window.add_updater(lambda w: w.align_to(top_axes.get_origin(), DOWN))

        v_line = DashedLine(
            window.get_top(),
            low_axes.c2p(low_axes.x_axis.p2n(window.get_bottom()), 0)
        )
        v_line.add_updater(lambda m, w=window: m.move_to(w.get_top(), UP))
        v_line.set_stroke(WHITE, 1)

        self.add(window)
        self.add(v_line)

        drawing_dot = GlowDot(color=window.get_color())
        drawing_dot.add_updater(lambda m, w=window, g=low_graph.copy(): m.move_to(
            g.quick_point_from_proportion(
                (low_axes.x_axis.p2n(w.get_center()) + rs) / (2 * rs)
            )
        ))
        drawing_dot.set_glow_factor(1)
        drawing_dot.set_radius(0.15)
        self.add(drawing_dot)
        self.play(
            ShowCreation(low_graph),
            window.animate.move_to(top_axes.c2p(rs, 0), DOWN),
            rate_func=bezier([0, 0, 1, 1]),
            run_time=(2.5 if self.quick_transitions else 5),
        )
        self.wait()
        return Group(window, v_line, drawing_dot)

    def get_brace_label_tex(self, n):
        result = "1"
        for k in range(3, 2 * n + 3, 2):
            result += "- " + f"1 / {k}"
        return result

    def get_rect_func_def(self):
        return MTex(
            R"""\text{rect}(x) :=
            \begin{cases}
                1 & \text{if } \text{-}\frac{1}{2} < x < \frac{1}{2} \
                0 & \text{otherwise}
            \end{cases}"""
        )

    def get_axes(self, x_range=(-1, 1, 0.25), y_range=(0, 1, 0.25), width=13.0, height=2.5):
        axes = Axes(x_range=x_range, y_range=y_range, width=width, height=height)
        axes.add_coordinate_labels(
            x_values=np.arange(*x_range)[::2],
            y_values=np.arange(0, 1.0, 0.5),
            num_decimal_places=2, font_size=20
        )
        return axes


class LongerTimescaleMovingAverages(MovingAverages):
    sample_resolution = 4000
    n_iterations = 58
    # n_iterations = 16
    rect_scalar = 2.0
    quick_transitions = True

    def get_rect_func_def(self):
        return MTex(
            R"""\text{rect}_2(x) :=
            \begin{cases}
                1 & \text{if } \text{-}1 < x < 1 \\
                0 & \text{otherwise}
            \end{cases}"""
        )

    def get_axes(self, x_range=(-2, 2, 0.25), *args, **kwargs):
        return super().get_axes(x_range, *args, **kwargs)

    def get_brace_label_tex(self, n):
        result = "2"
        for k in range(3, min(2 * n + 3, 9), 2):
            result += "- " + f"1 / {k}"
        if n > 3:
            result += fR" - \cdots - 1 / {2 * n + 1}"
        return result


class ShowReciprocalSums(InteractiveScene):
    def construct(self):
        # Equations
        equations = VGroup(*(
            self.get_sum(n)
            for n in range(1, 8)
        ))
        for equation in equations:
            t2c = dict(zip(
                [f"1 / {2 * k + 1}" for k in range(1, 10)],
                color_gradient([BLUE, YELLOW, RED], 8)
            ))
            equation.set_color_by_tex_to_color_map(t2c)
        equations[-1][-1].set_color(RED)
        equations.arrange(DOWN, aligned_edge=RIGHT, buff=MED_LARGE_BUFF)
        equations.center()

        self.add(equations[0])
        for eq1, eq2 in zip(equations, equations[1:]):
            self.play(TransformMatchingTex(eq1.copy(), eq2, fade_transform_mismatches=True))
            self.wait()
        self.wait()

    def get_sum(self, n):
        tex_parts = []
        tally = 0
        for k in range(1, n + 1):
            tex_parts.append(f"1 / {2 * k + 1}")
            tex_parts.append("+")
            tally += 1 / (2 * k + 1)
        tex_parts[-1] = "="
        tex_parts.append(R"{:.06f}\dots".format(tally))
        return Tex(*tex_parts)


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
    fg_label_tex = "f(x) \cdot g(t - x)"
    t_color = TEAL
    area_line_dx = 0.05
    jagged_product = False

    def construct(self):
        # Add axes
        all_axes = self.all_axes = self.get_all_axes()
        f_axes, g_axes, fg_axes, conv_axes = all_axes
        x_min, x_max = self.axes_config["x_range"][:2]

        self.disable_interaction(*all_axes)
        self.add(*all_axes)

        # Add f(x)
        f_graph = f_axes.get_graph(self.f, x_range=(x_min, x_max, self.f_graph_x_step))
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

        g_graph = g_axes.get_graph(lambda x: 0, x_range=(x_min, x_max, self.g_graph_x_step))
        g_graph.set_style(**self.g_graph_style)
        g_axes.bind_graph_to_func(g_graph, lambda x: self.g(get_t() - x), jagged=self.jagged_product)

        g_label = self.g_label = self.get_label(self.g_label_tex, g_axes)

        t_label = VGroup(*Tex("t = ")[0], DecimalNumber())
        t_label.arrange(RIGHT, buff=SMALL_BUFF)
        t_label.scale(0.5)
        t_label.set_backstroke(width=8)
        t_label.add_updater(lambda m: m.next_to(t_indicator, DOWN, submobject_to_align=m[0], buff=0.15))
        t_label.add_updater(lambda m: m[-1].set_value(get_t()))

        self.add(g_graph)
        self.add(g_label)
        self.add(t_indicator)
        self.add(t_label)

        # Show integral of f(x) * g(t - x)
        def prod_func(x):
            return self.f(x) * self.g(get_t() - x)

        fg_graph = fg_axes.get_graph(lambda x: 0, x_range=(x_min, x_max, self.g_graph_x_step))
        fg_graph.set_style(**self.fg_graph_style)
        fg_axes.bind_graph_to_func(fg_graph, prod_func, jagged=self.jagged_product)
        fg_label = self.get_label(self.fg_label_tex, fg_axes)

        x_values = np.arange(x_min, x_max, self.area_line_dx)
        area_lines = Line().replicate(len(x_values))
        area_lines.set_stroke(BLUE, 3)

        def update_area_lines(lines):
            for x, line in zip(x_values, lines):
                y = prod_func(x)
                line.put_start_and_end_on(fg_axes.c2p(x, 0), fg_axes.c2p(x, y))
                if y > 0:
                    line.set_color(BLUE)
                else:
                    line.set_color(RED)

        area_lines.add_updater(update_area_lines)

        self.add(area_lines)
        self.add(fg_graph)
        self.add(fg_label)

        # Show convolution
        dx = 0.1
        x_samples = np.arange(x_min, x_max + dx, dx)
        f_samples = self.f(x_samples)
        g_samples = self.g(x_samples)
        conv_samples = np.convolve(f_samples, g_samples, mode='same')
        conv_graph = self.conv_graph = VMobject().set_style(**self.conv_graph_style)
        conv_graph.set_points_as_corners(conv_axes.c2p(x_samples, conv_samples * dx))
        if not self.jagged_product:
            conv_graph.make_approximately_smooth()

        graph_dot = GlowDot(color=WHITE)
        graph_dot.add_updater(lambda d: d.move_to(conv_graph.quick_point_from_proportion(
            inverse_interpolate(x_min, x_max, get_t())
        )))
        graph_line = Line(stroke_color=WHITE, stroke_width=1)
        graph_line.add_updater(lambda l: l.put_start_and_end_on(
            graph_dot.get_center(),
            [graph_dot.get_x(), conv_axes.get_y(), 0],
        ))

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
            axes.x_axis.add(x_label)
            axes.y_axis.ticks.set_opacity(0)
            axes.x_axis.ticks.stretch(0.5, 1)

        return all_axes

    def get_label(self, tex, axes):
        label = Tex(tex, font_size=36)
        label.move_to(midpoint(axes.get_origin(), axes.get_right()))
        label.match_y(axes.get_top())
        return label

    def f(self, x):
        return 0.5 * np.exp(-0.8 * x**2) * (0.5 * x**3 - 3 * x + 1)

    def g(self, x):
        return np.exp(-x**2) * np.sin(2 * x)


class ShowFlippingOfGraph(InteractiveScene):
    def construct(self):
        pass


class MovingAverageAsConvolution(Convolutions):
    g_graph_x_step = 0.01
    jagged_product = True

    def construct(self):
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
        for t in [3, -3, -1.0]:
            self.play(t_indicator.animate.set_x(g_axes.c2p(t, 0)[0]), run_time=3)
            self.wait()

        # Show area
        rect = Rectangle(g_axes.x_axis.unit_size, g_axes.y_axis.unit_size)
        rect.move_to(g_axes.c2p(-1, 0), DOWN)
        rect.set_stroke(width=0)
        rect.set_fill(YELLOW, 0.5)
        rect.set_gloss(1)
        area_label = Tex(R"\text{Area } = 1", font_size=36)
        area_label.next_to(rect, UL)
        arrow = Arrow(area_label.get_bottom(), rect.get_center())

        self.play(
            Write(area_label, stroke_color=WHITE),
            ShowCreation(arrow),
            FadeIn(rect)
        )
        self.wait()

        # Rescale
        y_axes = VGroup(*(axes.y_axis for axes in self.all_axes[1:3]))
        fake_ys = y_axes.copy()
        self.add(*fake_ys, *self.mobjects)

        for sf in [0.3, 4.0, 0.5]:
            globals().update(locals())
            self.play(
                *(
                    y_axis.animate.stretch(sf, 1)
                    for y_axis in y_axes
                ),
                self.conv_graph.animate.stretch(sf, 1, about_point=self.all_axes[3].get_origin()),
                rect.animate.stretch(sf, 1, about_edge=DOWN),
                area_label.animate.set_opacity(0),
                arrow.animate.set_opacity(0),
                run_time=2
            )
            self.wait()

    def g(self, x):
        return rect_func(x)


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
        for k in range(2):
            left_graph = rect_graphs[k] if k == 0 else conv_graphs[k - 1]
            left_label = rect_defs[k] if k == 0 else conv_labels[k - 1]
            self.play(
                FadeOut(left_graph, 1.5 * LEFT),
                FadeOut(left_label, 1.5 * LEFT),
                FadeOut(rect_defs[k + 1]),
                FadeOut(rect_graphs[k + 1]),
                conv_labels[k].animate.match_x(axes1),
                conv_graphs[k].animate.match_x(axes1),
            )
            self.play(
                Write(rect_defs[k + 2], stroke_color=WHITE),
                ShowCreation(rect_graphs[k + 2]),
                run_time=1,
            )
            left_conv = conv_labels[k][0][1:-4]
            r = len(left_conv) + 1
            self.play(
                Transform(left_conv.copy(), conv_labels[k + 1][0][1:r], remover=True, path_arc=-PI / 3),
                Transform(rect_defs[2][:5].copy(), conv_labels[k + 1][0][r + 1:r + 6], remover=True, path_arc=-PI / 3),
                FadeIn(conv_labels[k + 1][0], lag_ratio=0.1, time_span=(0.5, 1.5)),
                ShowCreation(conv_graphs[k + 1]),
            )
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
        return Tex(fR"\text{{rect}}_{{{k}}}(x) := {k} \cdot \text{{rect}}({k}x)")[0]


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
                R"\text{rect}_3", "*",
                R"\text{rect}_5", "*", R"\cdots", "*",
                R"\text{rect}_{13}",
                R"\big]", "(0)", "=", "1.0"
            ),
            Tex(
                R"\big[",
                R"\text{rect}", "*",
                R"\text{rect}_3", "*",
                R"\text{rect}_5", "*", R"\cdots", "*",
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
        equations.set_width(FRAME_WIDTH - 1)
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


class ReplaceXWithPiX(InteractiveScene):
    def construct(self):
        # Setup graphs
        axes = Axes((-int(8 * PI), int(8 * PI)), (-0.5, 1.0, 0.5), width=FRAME_WIDTH * PI + 1, height=4)
        axes.shift(DOWN)
        axes.x_axis.add_numbers(num_decimal_places=0, font_size=20)
        axes.y_axis.add_numbers(num_decimal_places=1, font_size=20)
        sinc_graph = axes.get_graph(sinc)
        sinc_graph.set_stroke(BLUE, 1)
        sinc_pi_graph = sinc_graph.copy().stretch(1 / PI, 0, about_point=axes.get_origin())

        dx = 0.01
        sinc_area = axes.get_riemann_rectangles(
            sinc_graph,
            dx=dx,
            colors=(BLUE, BLUE),
            fill_opacity=0.5,
        )
        sinc_area.sort(lambda p: abs(p[0]))
        sinc_pi_area = sinc_area.copy().stretch(1 / PI, 0, about_point=axes.get_origin())

        partial_area = sinc_area[:len(sinc_area) // 3]
        self.add(partial_area, axes, sinc_graph)

        # Setup labels
        sinc_label = MTex(R"\int_{-\infty}^\infty \frac{\sin(x)}{x} dx = \pi")
        sinc_label.next_to(axes, UP).to_edge(LEFT)
        kw = dict(tex_to_color_map={R"\pi": TEAL})
        sinc_pi_label = MTex(
            R"\int_{-\infty}^\infty \frac{\sin(\pi x)}{\pi x} dx = 1.0",
            **kw
        )
        sinc_pi_label.move_to(sinc_label).to_edge(RIGHT)

        instead_of = Text("Instead of", color=YELLOW, font_size=60)
        instead_of.next_to(sinc_label, UP, buff=0.7, aligned_edge=LEFT)
        focus_on = Text("Focus on", color=YELLOW, font_size=60)
        focus_on.next_to(sinc_pi_label, UP, buff=0.7, aligned_edge=LEFT)

        self.add(instead_of, sinc_label)
        self.play(Write(partial_area, stroke_width=1.0))
        self.add(sinc_area, axes, sinc_graph)
        self.wait()

        # Squish
        x_to_pix = MTex(R"x \rightarrow \pi x", **kw)
        x_to_pix.match_y(instead_of)

        squish_arrows = VGroup(Vector(RIGHT), Vector(LEFT))
        squish_arrows.arrange(RIGHT, buff=1.5)
        squish_arrows.move_to(axes.c2p(0, 0.5))

        rect_kw = dict(buff=MED_SMALL_BUFF, stroke_width=1.5)
        rect = SurroundingRectangle(sinc_label, **rect_kw)
        sinc_graph.save_state()
        sinc_area.save_state()

        self.play(LaggedStart(
            FadeIn(x_to_pix),
            TransformMatchingShapes(sinc_label.copy(), sinc_pi_label),
            FadeTransform(instead_of.copy(), focus_on)
        ))
        self.wait()
        self.play(
            Transform(sinc_graph, sinc_pi_graph),
            Transform(sinc_area, sinc_pi_area),
            FadeIn(squish_arrows, scale=0.35),
            run_time=2
        )
        self.wait()
        self.play(
            ShowCreation(rect),
            FadeOut(squish_arrows, scale=3),
            sinc_area.animate.restore(),
            sinc_graph.animate.restore(),
        )
        self.play(FlashAround(sinc_label[-1], run_time=2))
        self.wait()
        self.play(
            rect.animate.become(SurroundingRectangle(sinc_pi_label, **rect_kw)),
            Transform(sinc_graph, sinc_pi_graph, run_time=2),
            Transform(sinc_area, sinc_pi_area, run_time=2),
            FadeIn(squish_arrows, scale=0.35, run_time=2),

        )
        self.play(FlashAround(sinc_pi_label[-3:], run_time=2))
        self.wait()


class FourierWrapper(VideoWrapper):
    title = "Fourier Transforms"


class FourierProblemSolvingSchematic(InteractiveScene):
    def construct(self):
        pass


class WhatWeNeedToShow(InteractiveScene):
    def construct(self):
        # Title
        title = Text("What we must show", font_size=60)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        title.set_backstroke(width=5)
        underline = Line(LEFT, RIGHT)
        underline.set_width(6)
        underline.set_stroke(GREY_A, width=(0, 3, 3, 3, 3, 0))
        underline.insert_n_curves(100)
        underline.next_to(title, DOWN, buff=0.05)

        self.add(underline, title)

        # Expressions
        t2c = {
            R"\mathcal{F}": TEAL,
            R"{t}": BLUE,
            R"{\omega}": YELLOW,
            R"{k}": RED,
        }
        kw = dict(tex_to_color_map=t2c, font_size=36)
        expressions = VGroup(
            MTex(R"\mathcal{F}\left[\frac{\sin(\pi {t})}{\pi {t}} \right]({\omega}) = \text{rect}({\omega})", **kw),
            MTex(R"\mathcal{F}\left[\frac{\sin(\pi {t} / {k})}{{t} / {k}} \right]({\omega}) = {k} \cdot \text{rect}({k}{\omega})", **kw),
            MTex(R"\int_{-\infty}^\infty f({t}) dt = \mathcal{F}\left[ f({t}) \right](0)", **kw),
            MTex(R"\int_{-\infty}^\infty \frac{\sin(\pi {t})}{\pi {t}} dt = \text{rect}(0) = 1", **kw),
            MTex(R"\mathcal{F}\left[ f({t}) \cdot g({t}) \right] = \mathcal{F}[f({t})] * \mathcal{F}[g({t})]", **kw),
            MTex(
                R"""\mathcal{F}\left[ \frac{\sin(\pi {t})}{\pi {t}} \cdot \frac{\sin(\pi {t} / 3)}{\pi {t} / 3} \right]
                = \big[ \text{rect} * \text{rect}_3 \big]""",
                **kw
            ),
        )
        expressions.set_stroke(width=0)
        key_facts = expressions[0::2]
        examples = expressions[1::2]
        key_facts.arrange(DOWN, buff=1.5, aligned_edge=LEFT)
        key_facts.next_to(underline, DOWN, MED_LARGE_BUFF).to_edge(LEFT)
        for fact, example in zip(key_facts, examples):
            example.next_to(fact, RIGHT, buff=2.0)

        ft_sinc, int_to_eval, conv_theorem = key_facts
        ft_sinck, sinc_int_to_rect_0, conv_theorem_ex = examples

        # FT of sinc
        ft_sinc.next_to(underline, DOWN, MED_LARGE_BUFF)

        width = FRAME_WIDTH / 2 - 1
        axes1 = Axes((-4, 4), (-1, 1), width=width, height=3)
        axes2 = Axes((-1, 1, 0.25), (0, 2), width=width, height=1.5)

        axes1.to_corner(DL)
        axes2.shift(axes1.get_origin() - axes2.get_origin())
        axes2.to_edge(RIGHT)

        axes1.add(Tex("t", color=BLUE, font_size=24).next_to(axes1.x_axis.get_right(), UP, 0.2))
        axes2.add(Tex(R"\omega", color=YELLOW, font_size=24).next_to(axes2.x_axis.get_right(), UP, 0.2))
        axes1.add_coordinate_labels(font_size=20)
        axes2.add_coordinate_labels(x_values=np.arange(-1, 1.5, 0.5), font_size=20, num_decimal_places=1)

        k_tracker = ValueTracker(1)
        get_k = k_tracker.get_value
        globals().update(locals())

        graph1 = axes1.get_graph(lambda x: 0, color=BLUE)
        axes1.bind_graph_to_func(graph1, lambda x: np.sinc(x / get_k()))

        graph2 = VMobject().set_stroke(YELLOW, 3)

        def update_graph2(graph):
            k = get_k()
            graph.set_points_as_corners([
                axes2.c2p(-1, 0),
                axes2.c2p(-0.5 / k, 0),
                axes2.c2p(-0.5 / k, k),
                axes2.c2p(0.5 / k, k),
                axes2.c2p(0.5 / k, 0),
                axes2.c2p(1, 0),
            ])
            return graph

        graph2.add_updater(update_graph2)

        graph1_label = MTex(R"{\sin(\pi {t}) \over \pi {t} }", **kw)
        graph2_label = MTex(R"\text{rect}({\omega})", **kw)
        graph1_label.move_to(axes1.c2p(-2, 1))
        graph2_label.move_to(axes2.c2p(0.5, 2))

        arrow = Arrow(axes1.c2p(2, 0.5), axes2.c2p(-0.5, 1), path_arc=-PI / 3)
        arrow.set_color(TEAL)
        arrow_copy = arrow.copy()
        arrow_copy.rotate(PI, about_point=midpoint(axes1.c2p(4, 0), axes2.c2p(-1, 0)))
        arrow_label = MTex(R"\mathcal{F}", color=TEAL)
        arrow_label.next_to(arrow, UP, SMALL_BUFF)
        arrow_label_copy = arrow_label.copy()
        arrow_label_copy.next_to(arrow_copy.pfp(0.5), UP)

        self.play(
            FadeIn(axes1),
            ShowCreation(graph1),
            FadeIn(graph1_label, UP)
        )
        self.wait()
        self.play(
            ShowCreation(arrow),
            FadeIn(arrow_label, RIGHT + 0.2 * UP),
            FadeIn(axes2),
        )
        self.play(
            Write(graph2_label),
            ShowCreation(graph2)
        )
        self.wait()
        self.play(
            TransformFromCopy(arrow, arrow_copy, path_arc=PI / 2),
            TransformFromCopy(arrow_label, arrow_label_copy, path_arc=PI / 2),
        )
        self.wait()

        self.play(LaggedStart(
            FadeTransform(arrow_label.copy(), ft_sinc[0]),
            FadeTransform(graph1_label.copy(), ft_sinc[2:12]),
            Write(VGroup(ft_sinc[1], ft_sinc[12])),
        ))
        self.wait()
        self.play(Write(ft_sinc[13:17]))
        self.play(
            FadeTransform(graph2_label.copy(), ft_sinc[17:])
        )
        self.add(ft_sinc)
        self.wait()

        # Generalize
        graph1_gen_label = MTex(R"{\sin(\pi {t} / {k}) \over \pi {t} / {k} }", **kw)
        graph2_gen_label = MTex(R"{k} \cdot \text{rect}({k} {\omega})", **kw)
        graph1_gen_label.move_to(graph1_label)
        graph2_gen_label.move_to(graph2_label)
        ft_sinck.move_to(ft_sinc)

        self.play(LaggedStart(
            FadeOut(graph1_label, UP),
            FadeIn(graph1_gen_label, UP),
            FadeOut(graph2_label, UP),
            FadeIn(graph2_gen_label, UP),
        ))
        self.play(
            FadeOut(ft_sinc, UP),
            FadeIn(ft_sinck, UP)
        )
        self.wait()
        self.play(k_tracker.animate.set_value(3), run_time=3)
        self.wait()
        self.play(k_tracker.animate.set_value(1), run_time=3)
        self.wait()
        self.play(ft_sinck.animate.set_height(0.5).to_corner(UL))
        self.wait()

        # Area to evaluate
        k_tracker.set_value(1)
        int_to_eval.next_to(underline, DOWN, MED_LARGE_BUFF)
        sinc_int_to_rect_0.move_to(int_to_eval)

        area = axes1.get_riemann_rectangles(
            axes1.get_graph(np.sinc),
            colors=(BLUE, BLUE),
            dx=0.01
        )
        area.set_stroke(WHITE, 0)
        x0 = axes1.get_origin()[0]
        area.sort(lambda p, x0=x0: abs(p[0] - x0))

        dot = GlowDot(color=BLUE)
        dot.set_glow_factor(0.5)
        dot.set_radius(0.1)
        dot.move_to(int_to_eval[11:].get_center())

        self.play(FadeIn(int_to_eval, DOWN))
        self.play(FlashAround(int_to_eval[:10], run_time=2, time_width=1.5))
        self.play(Write(area))
        self.wait()
        self.play(FlashAround(int_to_eval[11:], run_time=2, time_width=1.5))
        self.play(dot.animate.move_to(axes2.c2p(0, 1)), run_time=1.5)
        self.wait()
        self.play(
            int_to_eval.animate.set_height(0.5).next_to(ft_sinck, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
        )
        self.play(FadeIn(sinc_int_to_rect_0, RIGHT))
        self.wait()
        self.play(FadeOut(sinc_int_to_rect_0))

        # Convolution fact
        conv_theorem.set_height(0.45)
        conv_theorem.next_to(underline, DOWN, MED_LARGE_BUFF)
        conv_theorem_ex.next_to(underline, DOWN, MED_LARGE_BUFF)
        conv_theorem_name = TexText("``Convolution theorem''", font_size=60)
        conv_theorem_name.next_to(conv_theorem, DOWN, buff=MED_LARGE_BUFF)
        conv_theorem_name.set_color(YELLOW)

        self.play(FadeIn(conv_theorem, DOWN))
        self.wait()
        self.play(
            FadeIn(conv_theorem_ex, DOWN),
            FadeOut(conv_theorem, DOWN),
        )
        self.wait()
        self.play(
            FadeOut(conv_theorem_ex, UP),
            FadeIn(conv_theorem, UP),
        )
        self.play(Write(conv_theorem_name))

        # Reorganize
        top_row = VGroup(ft_sinck, int_to_eval)
        top_row.generate_target()
        top_row.target.scale(1.7)
        top_row.target.arrange(RIGHT, buff=LARGE_BUFF)
        top_row.target.next_to(underline, DOWN, MED_LARGE_BUFF)

        self.play(LaggedStart(
            MoveToTarget(top_row),
            conv_theorem.animate.next_to(top_row.target, DOWN, MED_LARGE_BUFF),
            FadeOut(conv_theorem_name),
        ))
        self.wait()
