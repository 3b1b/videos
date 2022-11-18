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


def get_sinc_tex(k=1):
    div_k = f"/ {k}" if k > 1 else ""
    return Rf"{{\sin(x {div_k}) \over x {div_k}}}"


def get_multi_sinc_integral(ks=[1], dots_at=None, rhs="", insertion=""):
    result = Tex(
        R"\int_{-\infty}^\infty",
        insertion,
        *(
            get_sinc_tex(k) if k != dots_at else R"\dots"
            for k in ks
        ),
        "dx",
        rhs,
    )
    t2c = {
        R"\sin": BLUE,
        "x / 3": TEAL,
        "x / 5": GREEN_B,
        "x / 7": GREEN_C,
        "x / 9": interpolate_color(GREEN, YELLOW, 1 / 3),
        "x / 11": interpolate_color(GREEN, YELLOW, 2 / 3),
        "x / 13": YELLOW,
        "x / 15": RED_B,
    }
    for tex, color in t2c.items():
        result.set_color_by_tex(tex, color)
    return result


class ShowIntegrals(InteractiveScene):
    # add_axis_labels = True
    add_axis_labels = False

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

        self.add(area, axes, graph)
        area.set_fill(opacity=1)
        self.play(
            Write(area, stroke_color=WHITE, lag_ratio=0.01, run_time=4),
            ReplacementTransform(func_label, int1[1]),
            Write(int1[::2]),
            graph.animate.set_stroke(width=1),
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
            colors=(BLUE_D, BLUE_D),
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


class SineLimit(InteractiveScene):
    def construct(self):
        axes = Axes((-4, 4), (-2, 2), width=14, height=7, axis_config=dict(tick_size=0))
        radius = axes.x_axis.unit_size
        circle = Circle(radius=radius)
        circle.move_to(axes.get_origin())
        circle.set_stroke(WHITE, 1)

        self.add(axes, circle)

        # Set up components
        x_tracker = ValueTracker(1)
        get_x = x_tracker.get_value
        origin = axes.get_origin()

        def get_point():
            x = get_x()
            return axes.c2p(math.cos(x), math.sin(x))

        arc = VMobject().set_stroke(YELLOW, 3)
        arc.add_updater(lambda m: m.match_points(Arc(
            radius=radius,
            arc_center=origin,
            start_angle=0,
            angle=get_x(),
        )))

        h_line = Line()
        h_line.set_stroke(RED)
        h_line.add_updater(lambda m: m.put_start_and_end_on(
            get_point(),
            get_point()[0] * RIGHT,
        ))

        radial_line = Line()
        radial_line.add_updater(lambda m: m.put_start_and_end_on(origin, get_point()))

        one_label = Integer(1, font_size=24)
        one_label.add_updater(lambda m: m.next_to(
            radial_line.get_center(),
            normalize(rotate_vector(radial_line.get_vector(), PI / 2)),
            buff=SMALL_BUFF,
        ))
        sine_label = Tex(R"\sin(x)", font_size=24)
        sine_label.set_backstroke()
        sine_label.add_updater(lambda m: m.set_max_height(0.2 * h_line.get_height()))
        sine_label.add_updater(lambda m: m.next_to(h_line, LEFT, buff=0.25 * m.get_width()))

        x_label = Tex("x", font_size=24)
        x_label.add_updater(lambda m: m.match_width(sine_label[0][4]))
        x_label.add_updater(lambda m: m.next_to(arc.pfp(0.5), RIGHT, buff=m.get_height()))

        self.add(arc, h_line, radial_line)
        self.add(one_label)
        self.add(sine_label)
        self.add(x_label)

        value_label = VGroup(
            Tex(R"\frac{\sin(x)}{x} = "),
            DecimalNumber(1, num_decimal_places=4)
        )
        value_label.arrange(RIGHT, buff=SMALL_BUFF)
        value_label.fix_in_frame()
        value_label.to_corner(UR)
        value_label.add_updater(lambda m: m[1].set_value(math.sin(get_x()) / get_x()))
        self.add(value_label)

        # Zoom in
        frame = self.camera.frame
        frame.set_height(4)
        x_tracker.add_updater(lambda m: m.set_value((1 / (self.time**1.5 + 1.5))))
        self.add(x_tracker)

        target_frame = frame.deepcopy()
        target_frame.add_updater(lambda m: m.set_height(4 * h_line.get_height()).move_to(h_line.get_bottom()))
        self.add(target_frame)

        alpha_tracker = ValueTracker(0)
        get_alpha = alpha_tracker.get_value

        self.play(
            UpdateFromFunc(
                frame, lambda m: m.interpolate(m, target_frame, get_alpha()),
                run_time=15,
            ),
            alpha_tracker.animate.set_value(1 / 30).set_anim_args(run_time=1)
        )


class WriteFullIntegrals(InteractiveScene):
    def construct(self):
        # Integrals
        ints = VGroup(*(
            get_multi_sinc_integral(range(1, n, 2), rhs=R"= \pi")
            for n in range(3, 19, 2)
        ))
        ints.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        ints.center().to_corner(UL)
        for inter in ints:
            inter[-1].scale(1.5, about_edge=LEFT)

        q_marks = Tex("???", color=RED).scale(2)
        q_marks.next_to(ints[-1], RIGHT, buff=MED_LARGE_BUFF)
        correction = Tex("- 0.0000000000462...").scale(1.25)
        correction.next_to(ints[-1], RIGHT, SMALL_BUFF)

        # Show all
        frame = self.camera.frame
        ds = ValueTracker(0)
        frame.add_updater(lambda m, dt: m.scale(1 + 0.02 * dt, about_edge=UL).shift(ds.get_value() * dt * DOWN))

        self.add(ints[0], ds)
        for i in range(len(ints) - 1):
            self.wait(2)
            anims = [TransformMatchingTex(ints[i].copy(), ints[i + 1])]
            if i < 6:
                anims.append(ds.animate.increment_value(0.1))
            self.play(*anims)
        frame.clear_updaters()
        self.play(Write(q_marks), frame.animate.scale(1.1, about_edge=UL).shift(2 * DOWN), run_time=2)
        self.wait()
        self.play(FadeTransform(q_marks, correction))
        self.wait()


class WriteMoreFullIntegrals(InteractiveScene):
    def construct(self):
        # Unfortunate copy pasting, but I'm in a rush
        # Integrals
        ints = VGroup(*(
            get_multi_sinc_integral(range(1, n, 2), rhs=R"= \pi")
            for n in range(3, 17, 2)
        ))
        ints.add(get_multi_sinc_integral(range(1, 17, 2), rhs=R"= (0.99999999998529)\pi"))
        ints.add(get_multi_sinc_integral(range(1, 19, 2), rhs=R"= (0.99999998807962)\pi"))
        ints.add(get_multi_sinc_integral(range(1, 21, 2), rhs=R"= (0.99999990662610)\pi"))
        ints.add(get_multi_sinc_integral(range(1, 23, 2), rhs=R"= (0.99999972286332)\pi"))

        ints.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        ints.center().to_corner(UL)
        for inter in ints:
            inter[-1].scale(1.5, about_edge=LEFT)

        # Show all
        frame = self.camera.frame
        ds = ValueTracker(0)
        frame.add_updater(lambda m, dt: m.scale(1 + 0.04 * dt, about_edge=UL).shift(ds.get_value() * dt * DOWN))

        key_map = dict(
            (ints[n][-1].get_tex(), ints[n + 1][-1].get_tex())
            for n in range(-4, -1, 1)
        )

        self.add(ints[0], ds)
        for i in range(len(ints) - 1):
            self.wait()
            anims = [TransformMatchingTex(ints[i].copy(), ints[i + 1], key_map=key_map)]
            if i < 6:
                anims.append(ds.animate.increment_value(0.1))
            self.play(*anims)
        self.wait(3)
        frame.clear_updaters()


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
            get_multi_sinc_integral([1], rhs=R"= \pi"),
            get_multi_sinc_integral([1, 3], rhs=R"= \pi"),
            get_multi_sinc_integral([1, 3, 5], rhs=R"= \pi"),
            Tex(R"\vdots"),
            get_multi_sinc_integral([1, 3, 5, 7, 13], dots_at=7, rhs=R"= \pi"),
            get_multi_sinc_integral([1, 3, 5, 7, 13, 15], dots_at=7, rhs=Rf"= ({SUB_ONE_FACTOR}\dots)\pi"),
        )


class WriteOutIntegralsWithPi(WriteOutIntegrals):
    def get_integrals(self):
        t2c = {
            R"\sin": BLUE,
            "x/3": TEAL,
            "x/5": GREEN_B,
            "x/7": GREEN_C,
            "x/9": interpolate_color(GREEN, YELLOW, 1 / 3),
            "x/11": interpolate_color(GREEN, YELLOW, 2 / 3),
            "x/13": YELLOW,
            "x/15": RED_B,
        }
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
            Tex(R"\vdots"),
            Tex(
                R"\int_{-\infty}^\infty",
                R"\frac{\sin(\pi x)}{\pi x}",
                R"\frac{\sin(\pi x/3)}{\pi x/3}",
                # R"\frac{\sin(\pi x/5)}{\pi x/5}",
                R"\dots",
                R"\frac{\sin(\pi x/13)}{\pi x/13}",
                R"dx = ", "1.0"
            ),
            Tex(
                R"\int_{-\infty}^\infty",
                R"\frac{\sin(\pi x)}{\pi x}",
                R"\frac{\sin(\pi x/3)}{\pi x/3}",
                # R"\frac{\sin(\pi x/5)}{\pi x/5}",
                R"\dots",
                R"\frac{\sin(\pi x/13)}{\pi x/13}",
                R"\frac{\sin(\pi x/15)}{\pi x/15}",
                R"dx = ", fR"{SUB_ONE_FACTOR}\dots", "."
            ),
        )
        result[-1][-1].scale(0)
        for mob in result:
            for tex, color in t2c.items():
                mob.set_color_by_tex(tex, color)
        return result


class InsertTwoCos(InteractiveScene):
    dx = 0.025

    def construct(self):
        # Formulas
        n_range = list(range(3, 17, 2))
        integrals = VGroup(
            get_multi_sinc_integral([1], rhs=R"= \pi"),
            *(
                get_multi_sinc_integral(range(1, n, 2), insertion=R"2\cos(x)", rhs=R"= \pi")
                for n in n_range
            )
        )
        integrals.scale(0.75)
        for inter in integrals:
            inter.to_edge(UP)

        self.add(integrals[0])

        # Graphs
        axes = Axes((-4 * PI, 4 * PI, PI), (-2, 2, 1), width=FRAME_WIDTH + 1, height=6)
        axes.to_edge(DOWN, buff=SMALL_BUFF)
        axes.x_axis.add_numbers(unit_tex=R"\pi", unit=PI)

        graphs = VGroup(
            axes.get_graph(sinc),
            *(
                axes.get_graph(
                    lambda x, n=n: 2 * math.cos(x) * multi_sinc(x, n + 1),
                    x_range=(-4 * PI, 4 * PI, 0.01)
                )
                for n in range(len(n_range))
            )
        )
        graphs.set_stroke(WHITE, 2)

        areas = VGroup(*(
            axes.get_riemann_rectangles(
                graph,
                dx=self.dx,
                colors=(BLUE_D, BLUE_D),
                fill_opacity=1
            )
            for graph in graphs
        ))

        self.add(areas[0], axes, graphs[0])

        # Progress
        for i in range(len(integrals) - 1):
            self.play(
                TransformMatchingTex(integrals[i], integrals[i + 1]),
                ReplacementTransform(graphs[i], graphs[i + 1]),
                ReplacementTransform(areas[i], areas[i + 1]),
            )
            self.add(areas[i + 1], axes, graphs[i + 1])
            if i == 0:
                self.play(FlashAround(integrals[i + 1][1], run_time=2, time_width=1))
            self.wait()


class WriteTwoCosPattern(InteractiveScene):
    def construct(self):
        # Integrals
        integrals = VGroup(
            *(
                get_multi_sinc_integral(range(1, n, 2), insertion=R"2\cos(x)", rhs=R"= \pi")
                for n in range(1, 7, 2)
            ),
            Tex(R"\vdots"),
            get_multi_sinc_integral([1, 3, 5, 111], dots_at=5, insertion=R"2\cos(x)", rhs=R"= \pi"),
            get_multi_sinc_integral([1, 3, 5, 111, 113], dots_at=5, insertion=R"2\cos(x)", rhs=R"= \pi"),
        )
        integrals.scale(0.75)
        for inter in integrals:
            inter[-1].scale(1.5, about_edge=LEFT)

        integrals.arrange(DOWN, buff=0.8, aligned_edge=RIGHT)
        integrals.set_height(FRAME_HEIGHT - 1)
        integrals.move_to(2 * LEFT)

        integrals[-1][-3].set_color(RED)

        self.add(integrals[0])
        for i in [0, 1, 2, 4]:
            j = i + 2 if i == 2 else i + 1
            anims = [TransformMatchingTex(integrals[i].copy(), integrals[j])]
            if i == 2:
                anims.append(Write(integrals[3], run_time=1))
            self.play(*anims)
            self.wait(0.5)
        self.add(integrals)

        # RHS
        text = Text("A tiny-but-definitely-positive number\nthat my computer couldn't evaluate\nin a reasonable amount of time")
        text.scale(0.5)
        parens = Tex("()")[0]
        parens.match_height(text)
        parens[0].next_to(text, LEFT, SMALL_BUFF)
        parens[1].next_to(text, RIGHT, SMALL_BUFF)
        minus = Tex("-").next_to(parens, LEFT)
        rhs = VGroup(minus, parens, text)

        rhs.next_to(integrals[-1], RIGHT, buff=SMALL_BUFF)
        self.play(Write(rhs))
        self.wait()


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
                1 & \text{if } \text{-}\frac{1}{2} < x < \frac{1}{2} \\
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
            R"""\text{long\_rect}(x) :=
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
    max_shown_parts = 10

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
        msp = self.max_shown_parts
        for k in range(1, n + 1):
            new_parts = [f"1 / {2 * k + 1}", "+"]
            if n > msp:
                if k < msp - 1 or k == n:
                    tex_parts.extend(new_parts)
                elif k == msp - 1:
                    tex_parts.extend([R"\cdots", "+"])
            else:
                tex_parts.extend(new_parts)
            tally += 1 / (2 * k + 1)
        tex_parts[-1] = "="
        tex_parts.append(R"{:.06f}\dots".format(tally))
        return Tex(*tex_parts)


class LongerReciprocalSums(ShowReciprocalSums):
    max_shown_parts = 5

    def construct(self):
        # Test
        equations = VGroup(*(
            self.get_sum(n)
            for n in [1, 2, 3, 4, 54, 55, 56]
        ))
        dots = Tex(R"\vdots")
        group = VGroup(*equations[:4], dots, *equations[4:])
        group.arrange(DOWN, buff=0.35, aligned_edge=RIGHT)
        dots.shift(2 * LEFT)
        for eq in equations:
            eq.set_color_by_tex_to_color_map({
                "1 / 3": BLUE,
                "1 / 5": interpolate_color(BLUE, GREEN, 1 / 3),
                "1 / 7": interpolate_color(BLUE, GREEN, 2 / 3),
                "1 / 9": GREEN,
                "1 / 109": YELLOW,
                "1 / 111": interpolate_color(YELLOW, RED, 1 / 2),
                "1 / 113": RED,
            })

        self.add(equations[0])
        for eq1, eq2 in zip(equations, equations[1:]):
            anims = [FadeTransformPieces(eq1.copy(), eq2)]
            if eq1 is equations[3]:
                anims.append(Write(dots))
            self.play(*anims)
            self.wait(0.5)


class MoreGeneralFact(InteractiveScene):
    dx = 0.025

    def construct(self):
        # Equation
        inters = VGroup(*(
            get_multi_sinc_integral(range(1, n, 2), rhs=R"= \pi")
            for n in range(3, 11, 2)
        ))
        inters.scale(0.75)
        inters.to_edge(UP)

        # Graph
        axes = Axes((-4 * PI, 4 * PI, PI), (-0.5, 1, 0.5), width=FRAME_WIDTH + 1, height=4)
        axes.x_axis.add_numbers(unit=PI, unit_tex=R"\pi")
        axes.to_edge(DOWN)

        graphs = VGroup(*(
            axes.get_graph(lambda x, n=n: multi_sinc(x, n))
            for n in range(1, 5)
        ))
        graphs.set_stroke(WHITE, 2)
        areas = VGroup(*(
            axes.get_riemann_rectangles(
                graph, colors=(BLUE_D, BLUE_D), dx=self.dx, fill_opacity=1,
            )
            for graph in graphs
        ))
        for area in areas:
            area.sort(lambda p: abs(p[0]))

        self.add(areas[0], axes, graphs[0], inters[0])

        for i in range(3):
            self.play(
                ReplacementTransform(areas[i], areas[i + 1]),
                TransformMatchingTex(inters[i], inters[i + 1]),
                ReplacementTransform(graphs[i], graphs[i + 1]),
            )
            self.add(areas[i + 1], axes, graphs[i + 1], inters[i + 1])
        self.wait()

        # Generalize
        inner_group = inters[-1][2:5]
        rect = SurroundingRectangle(inner_group, buff=SMALL_BUFF).set_stroke(YELLOW, 1)
        brace = Brace(rect)
        brace_text = brace.get_text("Nothing special", font_size=36)

        general_group = Tex(
            R"\frac{\sin(a_1 x)}{a_1 x}",
            R"\cdots",
            R"\frac{\sin(a_n x)}{a_n x}",
        )
        general_group.set_submobject_colors_by_gradient(TEAL, GREEN)
        general_group.match_height(inner_group)
        general_group.move_to(inner_group, LEFT)

        self.play(
            ShowCreation(rect),
            GrowFromCenter(brace),
            FadeIn(brace_text, 0.5 * DOWN),
        )
        self.wait()
        self.play(
            FadeOut(inner_group, UP),
            FadeIn(general_group, UP),
            rect.animate.match_points(SurroundingRectangle(general_group, buff=SMALL_BUFF)),
            brace.animate.become(Brace(general_group, DOWN, buff=MED_SMALL_BUFF)),
            FadeOut(brace_text),
            inters[-1][5:].animate.next_to(general_group, RIGHT, SMALL_BUFF),
        )
        self.wait()

        # Sum condition
        sum_tex = brace.get_tex(R"a_1 + \cdots + a_n < 1")[0]
        lt = Tex("<")
        eq = inters[-1][-1][0]
        lt.move_to(eq)

        self.play(FadeIn(sum_tex, 0.5 * DOWN))
        self.play(FlashAround(inters[-1][-1], run_time=2, time_width=1))
        self.wait()
        self.play(Rotate(sum_tex[-2], PI))
        self.play(
            FadeOut(eq, 0.5 * UP),
            FadeIn(lt, 0.5 * UP),
        )
        self.wait()


class WaysToCombineFunctions(InteractiveScene):
    def construct(self):
        # Axes
        axes1, axes2, axes3 = all_axes = VGroup(*(
            Axes((-5, 5), (0, 2), width=FRAME_WIDTH - 2, height=FRAME_HEIGHT / 3 - 1.0)
            for x in range(3)
        ))
        all_axes.arrange(DOWN, buff=1.0)

        self.add(all_axes)

        # Functions
        def f(x):
            return np.sin(x) + 1

        def g(x):
            return np.exp(-x**2)

        f_graph = axes1.get_graph(f, color=BLUE)
        g_graph = axes2.get_graph(g, color=YELLOW)

        f_label = Tex(R"f(x) = \sin(x) + 1", color=BLUE)[0]
        g_label = Tex(R"g(x) = e^{-x^2}", color=YELLOW)[0]
        f_label.move_to(axes1.c2p(-1.5, 1.5))
        g_label.move_to(axes2.c2p(-1.5, 1.5))

        self.play(
            LaggedStart(FadeIn(f_graph), FadeIn(g_graph)),
            LaggedStart(FadeIn(f_label), FadeIn(g_label)),
        )
        self.wait()

        # Combinations
        sum_graph = axes3.get_graph(lambda x: f(x) + g(x), color=GREEN)
        prod_graph = axes3.get_graph(lambda x: f(x) * g(x), color=GREEN)
        dx = 0.1
        x_samples = np.arange(*axes1.x_range[:2], dx)
        conv_samples = np.convolve(f(x_samples), g(x_samples), mode="same") * dx * 0.5
        conv_graph = VMobject().set_points_smoothly(axes3.c2p(x_samples, conv_samples))
        conv_graph.match_style(prod_graph)
        graphs = (sum_graph, prod_graph, conv_graph)

        sum_label = Tex("[f + g](x)")
        prod_label = Tex(R"[f \cdot g](x)")
        conv_label = Tex(R"[f * g](x)")
        labels = (sum_label, prod_label, conv_label)
        for label in labels:
            label.move_to(axes3.c2p(-1.5, 1.5))

        words = VGroup(*map(Text, ["Addition", "Multiplication", "Convolution"]))
        for word, label in zip(words, labels):
            word.next_to(label, UP)

        for graph, label, word in zip(graphs, labels, words):
            self.play(
                Transform(f_graph.copy(), graph.copy(), remover=True),
                TransformFromCopy(g_graph, graph),
                TransformMatchingShapes(VGroup(*f_label[:4], *g_label[:4]).copy(), label),
                FadeIn(word)
            )
            self.wait()
            self.play(*map(FadeOut, [graph, label, word]))


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
            colors=(BLUE_E, BLUE_E),
            negative_color=RED_E,
            fill_opacity=1.0,
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

        eq_pi = sinc_label[-2:]
        eq_one = sinc_pi_label[-4:]

        pi_rect = SurroundingRectangle(eq_pi).set_stroke(BLUE, 2)
        one_rect = SurroundingRectangle(eq_one).set_stroke(BLUE, 2)
        want_to_show = Text("want to show", font_size=36)
        want_to_show.next_to(pi_rect, DOWN, aligned_edge=LEFT)
        want_to_show.set_color(BLUE)

        instead_of = Text("Instead of", color=YELLOW, font_size=60)
        instead_of.next_to(sinc_label, UP, buff=0.7, aligned_edge=LEFT)
        focus_on = Text("Focus on", color=YELLOW, font_size=60)
        focus_on.next_to(sinc_pi_label, UP, buff=0.7, aligned_edge=LEFT)

        self.add(instead_of, sinc_label)
        self.play(Write(partial_area, stroke_width=1.0))
        self.add(sinc_area, axes, sinc_graph)
        self.wait()
        self.play(
            ShowCreation(pi_rect),
            FadeIn(want_to_show, 0.5 * DOWN)
        )
        self.wait()

        # Squish
        x_to_pix = MTex(R"x \rightarrow \pi x", **kw)
        x_to_pix.match_y(instead_of)

        squish_arrows = VGroup(Vector(RIGHT), Vector(LEFT))
        squish_arrows.arrange(RIGHT, buff=1.5)
        squish_arrows.move_to(axes.c2p(0, 0.5))

        rect_kw = dict(buff=MED_SMALL_BUFF, stroke_width=1.5)
        rect = SurroundingRectangle(sinc_pi_label, **rect_kw)
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
        self.play(ShowCreation(one_rect))
        self.wait()
        self.play(ShowCreation(rect))
        self.wait()
        self.play(
            FadeOut(squish_arrows, scale=3),
            sinc_area.animate.restore(),
            sinc_graph.animate.restore(),
            rect.animate.become(SurroundingRectangle(VGroup(sinc_label, want_to_show), **rect_kw)),
        )
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
            MTex(R"\mathcal{F}\left[\frac{\sin(\pi {t} / {k})}{\pi {t} / {k}} \right]({\omega}) = {k} \cdot \text{rect}({k}{\omega})", **kw),
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

        # Many dots
        dx = 0.1
        dots = Group(*(
            GlowDot(
                axes2.c2p(x, rect_func(x)),
                color=TEAL
            )
            for x in np.arange(-1, 1 + dx, dx)
        ))
        thick_graph = VGroup(
            axes1.get_graph(np.sinc, x_range=(-1, 4)),
            axes1.get_graph(np.sinc, x_range=(-4, 1)).reverse_points(),
        )
        thick_graph.set_stroke(YELLOW, 6)

        self.play(FadeIn(dots, DOWN, lag_ratio=0.5, run_time=5))
        self.wait()
        self.play(
            VShowPassingFlash(thick_graph[0], run_time=4, time_width=1),
            VShowPassingFlash(thick_graph[1], run_time=4, time_width=1),
            FadeOut(area)
        )
        self.wait()
        self.play(FadeOut(dots))

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
        facts = VGroup(ft_sinck, int_to_eval, conv_theorem)
        facts.generate_target()
        facts.target[:2].scale(1.7)
        facts.target.scale(0.8)
        facts.target.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        facts.target.next_to(ORIGIN, RIGHT).to_edge(UP, buff=MED_SMALL_BUFF)
        bullets = VGroup(*(
            Dot().next_to(fact, LEFT)
            for fact in facts.target
        ))

        self.play(
            MoveToTarget(facts),
            title.animate.next_to(facts.target, LEFT, LARGE_BUFF),
            Uncreate(underline),
            FadeOut(conv_theorem_name),
            Write(bullets)
        )
        self.wait()


class ConvolutionTheoremDiagram(InteractiveScene):
    def construct(self):
        # Axes
        width = FRAME_WIDTH / 2 - 1
        height = 1.5
        left_axes = VGroup(*(
            Axes((-4, 4), (-0.5, 1, 0.5), width=width, height=height)
            for x in range(3)
        ))
        right_axes = VGroup(*(
            Axes((-1, 1, 0.5), (0, 1), width=width, height=2 * height / 3)
            for x in range(3)
        ))

        left_axes.arrange(DOWN, buff=1.0)
        left_axes[-1].to_edge(DOWN, buff=MED_SMALL_BUFF)
        left_axes.set_x(-FRAME_WIDTH / 4)
        for a1, a2 in zip(left_axes, right_axes):
            a2.shift(a1.get_origin() - a2.get_origin())
        right_axes.set_x(FRAME_WIDTH / 4)

        # Graphs
        left_graphs = VGroup(
            left_axes[0].get_graph(np.sinc, color=BLUE),
            left_axes[1].get_graph(lambda x: np.sinc(x / 2), color=YELLOW),
            left_axes[2].get_graph(lambda x: np.sinc(x) * np.sinc(x / 2), color=GREEN),
        )
        left_graphs.set_stroke(width=2)
        right_graphs = VGroup(
            VMobject().set_points_as_corners([
                right_axes[0].c2p(x, y) for x, y in [
                    (-1, 0), (-0.5, 0), (-0.5, 1),
                    (0.5, 1), (0.5, 0), (1, 0),
                ]
            ]).set_stroke(BLUE, 2),
            VMobject().set_points_as_corners([
                right_axes[1].c2p(x, y) for x, y in [
                    (-1, 0), (-0.5 / 2, 0), (-0.5 / 2, 2),
                    (0.5 / 2, 2), (0.5 / 2, 0), (1, 0),
                ]
            ]).set_stroke(YELLOW, 2),
            VMobject().set_points_as_corners([
                right_axes[2].c2p(x, y) for x, y in [
                    (-1, 0), (-0.75, 0), (-0.25, 1),
                    (0.25, 1), (0.75, 0), (1, 0),
                ]
            ]).set_stroke(GREEN, 2),
        )

        left_plots = VGroup(*(VGroup(axes, graph) for axes, graph in zip(left_axes, left_graphs)))
        right_plots = VGroup(*(VGroup(axes, graph) for axes, graph in zip(right_axes, right_graphs)))

        # Labels
        left_labels = VGroup(
            Tex(R"\frac{\sin(\pi x)}{\pi x}")[0],
            Tex(R"\frac{\sin(\pi x / 2)}{\pi x / 2}")[0],
            Tex(R"\frac{\sin(\pi x)}{\pi x} \cdot \frac{\sin(\pi x / 2)}{\pi x / 2}")[0],
        )
        right_labels = VGroup(*(
            Tex(
                Rf"\mathcal{{F}}\left[{ll.get_tex()}\right]",
                tex_to_color_map={R"\mathcal{F}": TEAL}
            )
            for ll in left_labels
        ))
        VGroup(left_labels, right_labels).scale(0.5)

        for label_group, axes_group, x in (left_labels, left_axes, -2), (right_labels, right_axes, -0.85):
            for label, axes in zip(label_group, axes_group):
                label.move_to(axes.c2p(x, 1))
        VGroup(left_labels[2], right_labels[2]).shift(0.5 * UP)

        ft_arrows = VGroup(*(
            Arrow(l1.get_right(), l2.get_left(), buff=0.2, path_arc=arc, color=TEAL, stroke_width=3)
            for l1, l2, arc in zip(left_labels, right_labels, (-0.3, -0.75, -1.0))
        ))

        # Left animations
        self.play(
            LaggedStartMap(FadeIn, left_plots[:2], lag_ratio=0.5),
            LaggedStartMap(FadeIn, left_labels[:2], lag_ratio=0.5),
            run_time=1
        )
        self.wait()
        self.play(
            Transform(left_plots[0].copy(), left_plots[2].copy(), remover=True),
            TransformFromCopy(left_plots[1], left_plots[2], remover=True),
            FadeTransform(left_labels[0].copy(), left_labels[2][:len(left_labels[0])]),
            FadeTransform(left_labels[1].copy(), left_labels[2][len(left_labels[0]):]),
        )
        self.add(left_plots[2])
        self.wait()
        self.play(
            ShowCreation(ft_arrows[2]),
            FadeTransform(left_labels[2].copy(), right_labels[2]),
            FadeIn(right_plots[2]),
        )
        self.wait()

        # Right animations
        for i in range(2):
            self.play(
                ShowCreation(ft_arrows[i]),
                FadeTransform(left_labels[i].copy(), right_labels[i]),
                FadeIn(right_plots[i]),
            )
        self.wait()

        right_labels[2].generate_target()
        equation = VGroup(
            right_labels[2].target,
            Tex("=").scale(0.75),
            right_labels[0].copy(),
            Tex("*").scale(0.75),
            right_labels[1].copy(),
        )
        equation.arrange(RIGHT, buff=SMALL_BUFF)
        equation.next_to(right_axes[2], UP, SMALL_BUFF)

        self.play(LaggedStart(
            ft_arrows[2].animate.put_start_and_end_on(
                ft_arrows[2].get_start(),
                right_labels[2].target.get_left() + SMALL_BUFF * UL,
            ),
            MoveToTarget(right_labels[2]),
            Write(equation[1]),
            FadeTransform(right_labels[0].copy(), equation[2]),
            Write(equation[3]),
            FadeTransform(right_labels[1].copy(), equation[4]),
            run_time=2
        ))

        # Show convolution
        x_unit = right_axes[1].x_axis.unit_size
        y_unit = right_axes[1].y_axis.unit_size
        rect = Rectangle(width=x_unit / 2, height=2 * y_unit)
        rect.set_stroke(YELLOW, 1)
        rect.set_fill(YELLOW, 0)
        rect.move_to(right_axes[1].get_origin(), DOWN)

        dot = GlowDot(color=GREEN)
        dot.move_to(right_graphs[2].get_start())

        self.add(rect, right_plots)
        self.play(
            rect.animate.move_to(right_axes[0].c2p(-1, 0), DOWN).set_fill(opacity=0.5),
            FadeIn(dot),
        )
        self.play(
            MoveAlongPath(dot, right_graphs[2]),
            UpdateFromFunc(rect, lambda m: m.match_x(dot)),
            run_time=8,
        )
        self.play(FadeOut(rect), FadeOut(dot))

        # Show signed area
        area = left_axes[2].get_riemann_rectangles(
            left_graphs[2],
            dx=0.01,
            colors=(BLUE_D, BLUE_D),
            fill_opacity=1.0,
        )
        o = area.get_center()
        area.sort(lambda p: abs(p[0] - o[0]))

        self.add(area, left_axes[2], left_graphs[2])
        self.play(Write(area, stroke_width=1, run_time=2))
        self.wait()
        rect.set_fill(opacity=0.2)
        rect.set_stroke(width=0)
        self.play(
            MoveAlongPath(dot, right_graphs[2], rate_func=lambda t: smooth(1 - 0.5 * t)),
            MoveAlongPath(dot.copy(), right_graphs[2], rate_func=lambda t: smooth(0.5 * t)),
            # UpdateFromFunc(rect, lambda m: m.match_x(dot)),
            run_time=1,
        )
        self.play(FlashAround(dot, buff=0, run_time=2, time_width=1))
        self.wait()


class MultiplyBigNumbers(InteractiveScene):
    def construct(self):
        # Numbers
        numbers = VGroup(
            MTex("3{,}141{,}592{,}653{,}589{,}793{,}238"),
            MTex("2{,}718{,}281{,}828{,}459{,}045{,}235"),
        )
        numbers.arrange(DOWN, aligned_edge=RIGHT)
        numbers.scale(1.5)
        numbers.move_to(1.0 * DOWN)
        underline = Underline(numbers).set_stroke(WHITE, 2)
        underline.stretch(1.2, 0, about_edge=RIGHT)
        times = Tex(R"\times")
        times.scale(1.5)
        times.next_to(underline.get_left(), UR)

        self.add(numbers)
        self.add(underline, times)

        # Prep run time
        d_label = TexText("Two $N$-digit numbers", tex_to_color_map={"$N$": YELLOW})
        d_label.next_to(numbers, UP, buff=LARGE_BUFF)
        d2_label = Tex(R"\mathcal{O}(N^2)", font_size=60, tex_to_color_map={"N": YELLOW})
        dlogd_label = Tex(R"\mathcal{O}({N} \cdot \text{log}({N}))", font_size=60, tex_to_color_map={"{N}": YELLOW})
        os = VGroup(d2_label, dlogd_label)
        os.arrange(RIGHT, buff=2.5)
        os.next_to(d_label, UP, buff=LARGE_BUFF)
        # cross = Exmark().scale(2).next_to(d2_label, RIGHT)
        cross = Cross(d2_label)
        cross.insert_n_curves(50).set_stroke(RED, (0, 8, 8, 8, 8, 0))
        check = Checkmark().scale(2).next_to(dlogd_label, RIGHT)
        q_marks = Tex("?", font_size=72)
        q_marks.next_to(d2_label, RIGHT)

        # Square run time
        for num in numbers:
            num.digits = num[::-1]
            num.digits.remove(*num[-4::-4])
            num.digit_highlights = VGroup(*(
                VHighlight(digit, color_bounds=(YELLOW, YELLOW_E), max_stroke_addition=8)
                for digit in num.digits
            ))

        self.add(d_label)
        self.play(FadeIn(d2_label, UP), FadeIn(q_marks, UP))
        for dh2 in numbers[1].digit_highlights:
            self.add(dh2, numbers[1])
            self.play(
                ShowSubmobjectsOneByOne(
                    numbers[0].digit_highlights.copy(),
                    remover=True
                ),
                run_time=1.0
            )
            self.remove(dh2)

        # d * log(d)
        self.play(
            ShowCreation(cross),
            FadeOut(q_marks),
            FadeTransform(d2_label.copy(), dlogd_label)
        )
        self.play(Write(check))
        self.wait()
