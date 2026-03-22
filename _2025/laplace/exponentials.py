from manim_imports_ext import *
from _2025.laplace.shm import ShowFamilyOfComplexSolutions


S_COLOR = YELLOW
T_COLOR = BLUE


def get_exp_graph_icon(s, t_range=(0, 7), y_max=4, pos_real_scalar=0.1, neg_real_scalar=0.2, width=1, height=1):
    axes = Axes(
        t_range,
        (-y_max, y_max),
        width=width,
        height=height,
        axis_config=dict(tick_size=0.035, stroke_width=1)
    )
    scalar = pos_real_scalar if s.real > 0 else neg_real_scalar
    new_s = complex(s.real * scalar, s.imag)
    graph = axes.get_graph(lambda t: np.exp(new_s * t).real)
    graph.set_stroke(YELLOW, 2)
    rect = SurroundingRectangle(axes)
    rect.set_fill(BLACK, 1)
    rect.set_stroke(WHITE, 1)
    return VGroup(rect, axes, graph)


class IntroduceEulersFormula(InteractiveScene):
    def construct(self):
        # Add plane
        plane = ComplexPlane(
            (-2, 2), (-2, 2),
            width=6, height=6,
        )
        plane.background_lines.set_stroke(BLUE, 1)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.5)
        plane.to_edge(LEFT)

        plane.add_coordinate_labels([1, -1])
        i_labels = VGroup(
            Tex(R"i", font_size=36).next_to(plane.n2p(1j), UL, SMALL_BUFF),
            Tex(R"-i", font_size=36).next_to(plane.n2p(-1j), DL, SMALL_BUFF),
        )
        plane.add(i_labels)

        self.add(plane)

        # Show pi
        pi_color = RED
        arc = Arc(0, PI, radius=plane.x_axis.get_unit_size(), arc_center=plane.n2p(0))
        arc.set_stroke(pi_color, 5)
        t_tracker = ValueTracker(0)
        t_dec = DecimalNumber(0)
        t_dec.set_color(pi_color)
        t_dec.add_updater(lambda m: m.set_value(t_tracker.get_value()))
        t_dec.add_updater(lambda m: m.move_to(plane.n2p(1.3 * np.exp(0.9 * t_tracker.get_value() * 1j))))

        pi = Tex(R"\pi", font_size=72)
        pi.set_color(pi_color)
        pi.set_backstroke(BLACK, 3)

        self.play(
            ShowCreation(arc),
            t_tracker.animate.set_value(PI),
            VFadeIn(t_dec, time_span=(0, 1)),
            run_time=2
        )
        pi.move_to(t_dec, DR)
        self.play(
            FadeOut(t_dec),
            FadeIn(pi),
            run_time=0.5
        )

        # Write formula
        formula = Tex(R"e^{\pi i} = -1", font_size=90, t2c={R"\pi": RED, "i": BLUE})
        formula.set_x(FRAME_WIDTH / 4).to_edge(UP)
        cliche = Text("Cliché?", font_size=72)
        cliche.next_to(formula, DOWN, LARGE_BUFF)

        randy = Randolph(height=2)
        randy.next_to(plane, RIGHT, LARGE_BUFF, aligned_edge=DOWN)
        randy.body.set_backstroke(BLACK)

        self.play(LaggedStart(
            TransformFromCopy(pi, formula[R"\pi"][0]),
            FadeTransform(i_labels[0].copy(), formula["i"][0]),
            Write(formula["="][0]),
            TransformFromCopy(plane.coordinate_labels[1], formula["-1"][0]),
            Write(formula["e"][0]),
            lag_ratio=0.2,
            run_time=3
        ))
        self.wait()

        self.play(
            LaggedStart(
                *(
                    # TransformFromCopy(formula[c1][0], cliche[c2][0])
                    FadeTransform(formula[c1][0].copy(), cliche[c2][0])
                    for c1, c2 in zip(
                        ["e", "1", "i", "e", R"\pi", "e", "1"],
                        "Cliché?",
                    )
                ),
                lag_ratio=0.1,
                run_time=3
            ),
            VFadeIn(randy),
            randy.says("This again?", mode="sassy", bubble_direction=LEFT)
        )
        self.play(Blink(randy))
        self.add(cliche)

        # Show many thumbnails
        plane_group = VGroup(plane, arc, pi)
        plane_group.set_z_index(-1)
        thumbnails = Group(
            Group(ImageMobject(f"https://img.youtube.com/vi/{slug}/maxresdefault.jpg"))
            for slug in [
                "-dhHrg-KbJ0",  # Mathologer
                "f8CXG7dS-D0",  # Welch Labs
                "ZxYOEwM6Wbk",  # 3b1b
                "LE2uwd9V5vw",  # Khan Academy
                "CRj-sbi2i2I",  # Numberphile
                "v0YEaeIClKY",  # Other 3b1b
                "sKtloBAuP74",
                "IUTGFQpKaPU",  # Po shen lo
            ]
        )
        thumbnails.set_width(4)
        thumbnails.arrange(DOWN, buff=-0.8)
        thumbnails[4:].align_to(thumbnails, UP).shift(0.5 * DOWN)
        thumbnails.to_corner(UL)
        for n, tn in enumerate(thumbnails):
            tn.add_to_back(SurroundingRectangle(tn, buff=0).set_stroke(WHITE, 1))
            tn.shift(0.4 * n * RIGHT)

        self.play(
            FadeOut(randy.bubble, time_span=(0, 1)),
            randy.change("raise_left_hand", thumbnails).set_anim_args(time_span=(0, 1)),
            plane_group.animate.set_width(3.5).next_to(formula, DOWN, MED_LARGE_BUFF).set_anim_args(time_span=(0, 2)),
            FadeOut(cliche, 3 * RIGHT, lag_ratio=-0.02, time_span=(0.5, 2.0)),
            LaggedStartMap(FadeIn, thumbnails, shift=UP, lag_ratio=0.5),
            run_time=3
        )

        # Fail to explain
        thumbnails.generate_target()
        for tn, vect in zip(thumbnails.target, compass_directions(len(thumbnails))):
            vect[0] *= 1.5
            tn.set_height(1.75)
            tn.move_to(3 * vect)

        formula.generate_target()
        q_marks = Tex(R"???", font_size=90)
        VGroup(formula.target, q_marks).arrange(DOWN, buff=MED_LARGE_BUFF).center()

        self.play(
            MoveToTarget(thumbnails, lag_ratio=0.01, run_time=2),
            FadeOut(randy, DOWN),
            FadeOut(plane_group, DOWN),
            MoveToTarget(formula),
            Write(q_marks)
        )
        self.wait()
        self.play(LaggedStartMap(FadeOut, thumbnails, shift=DOWN, lag_ratio=0.5, run_time=4))
        self.wait()

        # Show constant meanings
        e_copy = formula["e"][0].copy()

        circle = Circle(radius=1)
        circle.to_edge(UP, buff=LARGE_BUFF)
        circle.set_stroke(WHITE, 1)
        arc = circle.copy().pointwise_become_partial(circle, 0, 0.5)
        arc.set_stroke(pi_color, 5)
        radius = Line(circle.get_center(), circle.get_right())
        radius.set_stroke(WHITE, 1)
        radius_label = Tex(R"1", font_size=24)
        radius_label.next_to(radius, DOWN, SMALL_BUFF)
        pi_label = Tex(R"\pi").set_color(pi_color)
        pi_label.next_to(circle, UP, buff=SMALL_BUFF)
        circle_group = VGroup(circle, arc, radius_label, radius, pi_label)

        i_eq = Tex(R"i^2 = -1", t2c={"i": BLUE}, font_size=90)
        i_eq.move_to(circle).set_x(5)

        self.play(
            formula.animate.shift(DOWN),
            FadeOut(e_copy, 3 * UP + 5 * LEFT),
            FadeOut(q_marks, DOWN)
        )
        self.play(
            TransformFromCopy(formula[R"\pi"][0], pi_label),
            LaggedStartMap(FadeIn, VGroup(circle, radius, radius_label)),
            ShowCreation(arc),
        )
        self.play(
            FadeTransform(formula["i"][0].copy(), i_eq["i"][0]),
            Write(i_eq[1:], time_span=(0.75, 1.75)),
        )
        self.wait()

        # Question marks over i
        i_rect = SurroundingRectangle(formula["i"], buff=0.05)
        i_rect.set_stroke(YELLOW, 2)
        q_marks = Tex(R"???", font_size=24)
        q_marks.match_color(i_rect)
        q_marks.next_to(i_rect, UP, SMALL_BUFF)

        self.play(
            ShowCreation(i_rect),
            FadeIn(q_marks, 0.25 * UP, lag_ratio=0.25)
        )
        self.wait()

        # Who cares (To overlay)
        frame = self.frame
        back_rect = FullScreenRectangle()
        back_rect.fix_in_frame()
        back_rect.set_z_index(-1),

        self.play(
            LaggedStartMap(FadeOut, VGroup(circle_group, i_eq, VGroup(i_rect, q_marks))),
            FadeOut(circle_group),
            frame.animate.set_y(-3.5),
            FadeIn(back_rect),
            formula.animate.set_fill(WHITE),
            run_time=2
        )
        self.wait()


class ExpGraph(InteractiveScene):
    def construct(self):
        # Set up graph
        axes = Axes((-1, 4), (0, 20), width=10, height=6)
        axes.to_edge(RIGHT)
        x_axis_label = Tex("t")
        x_axis_label.next_to(axes.x_axis.get_right(), UL, MED_SMALL_BUFF)
        axes.add(x_axis_label)

        graph = axes.get_graph(np.exp)
        graph.set_stroke(BLUE, 3)

        title = Tex(R"\frac{d}{dt} e^t = e^t", t2c={"t": GREY_B}, font_size=60)
        title.to_edge(UP)
        title.match_x(axes.c2p(1.5, 0))

        self.add(axes)
        self.add(graph)
        self.add(title)

        # Add height tracker
        t_tracker = ValueTracker(1)
        get_t = t_tracker.get_value
        v_line = always_redraw(
            lambda: axes.get_v_line_to_graph(get_t(), graph, line_func=Line).set_stroke(RED, 3)
        )
        height_label = Tex(R"e^t", font_size=42)
        height_label.always.next_to(v_line, RIGHT, SMALL_BUFF)
        height_label_height = height_label.get_height()
        height_label.add_updater(lambda m: m.set_height(
            min(height_label_height, 0.7 * v_line.get_height())
        ))

        self.play(
            ShowCreation(v_line, suspend_mobject_updating=True),
            FadeIn(height_label, UP, suspend_mobject_updating=True),
        )
        self.wait()

        # Add tangent line
        tangent_line = always_redraw(
            lambda: axes.get_tangent_line(get_t(), graph, length=10).set_stroke(BLUE_A, 1)
        )
        unit_size = axes.x_axis.get_unit_size()
        unit_line = Line(axes.c2p(0, 0), axes.c2p(1, 0))
        unit_line.add_updater(lambda m: m.move_to(v_line.get_end(), LEFT))
        unit_line.set_stroke(WHITE, 2)
        unit_label = Integer(1, font_size=24)
        unit_label.add_updater(lambda m: m.next_to(unit_line.pfp(0.6), UP, 0.5 * SMALL_BUFF))
        tan_v_line = always_redraw(
            lambda: v_line.copy().shift(v_line.get_vector() + unit_size * RIGHT)
        )

        deriv_label = Tex(R"\frac{d}{dt} e^t = e^t", font_size=42)
        deriv_label[R"\frac{d}{dt}"].scale(0.75, about_edge=RIGHT)
        deriv_label_height = deriv_label.get_height()
        deriv_label.add_updater(lambda m: m.set_height(
            min(deriv_label_height, 0.8 * v_line.get_height())
        ))
        deriv_label.always.next_to(tan_v_line, RIGHT, SMALL_BUFF)

        self.play(ShowCreation(tangent_line, suspend_mobject_updating=True))
        self.play(
            VFadeIn(unit_line),
            VFadeIn(unit_label),
            VFadeIn(tan_v_line, suspend_mobject_updating=True),
            TransformFromCopy(title, deriv_label),
        )
        self.play(
            ReplacementTransform(v_line.copy().clear_updaters(), tan_v_line, path_arc=45 * DEG),
            FadeTransform(height_label.copy(), deriv_label["e^t"][1], path_arc=45 * DEG, remover=True),
        )
        self.wait()

        # Move it around
        for t in [2.35, 0, 1, 2]:
            self.play(t_tracker.animate.set_value(t), run_time=5)


class DefiningPropertyOfExp(InteractiveScene):
    def construct(self):
        # Key property
        tex_kw = dict(t2c={"{t}": GREY_B, "x": BLUE})
        equation = Tex(R"\frac{d}{d{t}} e^{t} = e^{t}", font_size=90, **tex_kw)

        exp_parts = equation["e^{t}"]
        ddt = equation[R"\frac{d}{d{t}}"]

        self.play(Write(exp_parts[0]))
        self.wait()
        self.play(FadeIn(ddt, scale=2))
        self.play(
            Write(equation["="]),
            TransformFromCopy(*exp_parts, path_arc=PI / 2),
        )
        self.wait()

        # Differential Equation
        ode = Tex(R"x'(t) = x(t)", font_size=72, **tex_kw)
        ode.move_to(equation).to_edge(UP)
        ode_label = Text("Differential\nEquation", font_size=36)
        ode_label.next_to(ode, LEFT, LARGE_BUFF, aligned_edge=DOWN)

        self.play(
            FadeTransform(equation.copy(), ode),
            FadeIn(ode_label)
        )
        self.wait()

        # Initial condition
        frame = self.frame
        abs_ic = Tex(R"x(0) = 1", font_size=72, **tex_kw)
        exp_ic = Tex(R"e^{0} = 1", font_size=90, t2c={"0": GREY_B})
        abs_ic.next_to(ode, RIGHT, buff=2.0)
        exp_ic.match_x(abs_ic).match_y(equation).shift(0.1 * UP)
        ic_label = Text("Initial\nCondition", font_size=36)
        ic_label.next_to(abs_ic, RIGHT, buff=0.75)

        self.play(
            FadeIn(abs_ic, RIGHT),
            FadeIn(exp_ic, RIGHT),
            frame.animate.set_x(2),
            Write(ic_label)
        )
        self.wait()

        # Scroll down
        self.play(frame.animate.set_y(-2.5), run_time=2)
        self.wait()


class ExampleExponentials(InteractiveScene):
    def construct(self):
        # Show the family
        pass

        # Highlight -1 + i term

        # Show e^t as its own derivative


class ImaginaryInputsToTheTaylorSeries(InteractiveScene):
    def construct(self):
        # Add complex plane
        plane = ComplexPlane(
            (-6, 6),
            (-4, 4),
            background_line_style=dict(stroke_color=BLUE, stroke_width=1),
            faded_line_style=dict(stroke_color=BLUE, stroke_width=0.5, stroke_opacity=0.5),
        )
        plane.set_height(5)
        plane.to_edge(DOWN, buff=0)
        plane.add_coordinate_labels(font_size=16)

        self.add(plane)

        # Add πi dot
        dot = GlowDot(color=YELLOW)
        dot.move_to(plane.n2p(PI * 1j))
        pi_i_label = Tex(R"\pi i", font_size=30).set_color(YELLOW)
        pi_i_label.next_to(dot, RIGHT, buff=-0.1).align_to(plane.n2p(3j), DOWN)

        self.add(dot, pi_i_label)

        # Show false equation
        false_eq = Tex(R"e^x = e \cdot e \cdots e \cdot e", t2c={"x": BLUE}, font_size=60)
        false_eq.to_edge(UP).shift(2 * LEFT)
        brace = Brace(false_eq[3:], DOWN)
        brace_tex = brace.get_tex(R"x \text{ times}")
        brace_tex[0].set_color(BLUE)

        nonsense = TexText(R"Nonsense if $x$ \\ is complex")
        nonsense.next_to(VGroup(false_eq, brace_tex), RIGHT, LARGE_BUFF)
        nonsense.set_color(RED)

        self.add(false_eq)
        self.play(GrowFromCenter(brace), FadeIn(brace_tex, lag_ratio=0.1))
        self.play(FadeIn(nonsense, lag_ratio=0.1))
        self.wait()

        # Make it the real equation
        gen_poly = self.get_series("x")
        gen_poly.to_edge(LEFT).to_edge(UP, MED_SMALL_BUFF)

        epii = self.get_series(R"\pi i", use_parens=True, in_tex_color=YELLOW)
        epii.next_to(gen_poly, DOWN, aligned_edge=LEFT)

        self.remove(false_eq)
        self.play(
            TransformFromCopy(false_eq[:2], gen_poly[0]),
            FadeOut(false_eq[2:], 0.5 * DOWN, lag_ratio=0.05),
            FadeOut(nonsense),
            FadeOut(brace, 0.5 * DOWN),
            FadeOut(brace_tex, 0.25 * DOWN),
            Write(gen_poly[1:])
        )
        self.wait()

        # Plug in πi
        vectors = self.get_spiral_vectors(plane, PI)
        buff = 0.5 * SMALL_BUFF
        labels = VGroup(
            Tex(R"\pi i", font_size=30).next_to(vectors[1], RIGHT, buff),
            Tex(R"(\pi^2 / 2) \cdot i^2", font_size=30).next_to(vectors[2], UP, buff),
            Tex(R"(\pi^3 / 6) \cdot i^3", font_size=30).next_to(vectors[3], LEFT, buff),
            Tex(R"(\pi^4 / 24) \cdot i^4", font_size=30).next_to(vectors[4], DOWN, buff),
        )
        labels.set_color(YELLOW)
        labels.set_backstroke(BLACK, 5)

        for n in range(0, len(gen_poly), 2):
            anims = [
                LaggedStart(
                    TransformMatchingTex(gen_poly[n].copy(), epii[n], run_time=1),
                    TransformFromCopy(gen_poly[n + 1], epii[n + 1]),
                    gen_poly[n + 2:].animate.align_to(epii[n + 2:], LEFT),
                    lag_ratio=0.05
                ),
            ]
            k = (n - 1) // 2
            if k >= 0:
                anims.append(GrowArrow(vectors[k]))
            if k == 1:
                anims.append(FadeTransform(pi_i_label, labels[0]))
            elif 2 <= k <= len(labels):
                anims.append(FadeIn(labels[k - 1]))
            if k >= 1:
                anims.append(dot.animate.set_width(0.5).move_to(vectors[k].get_end()))
            self.play(*anims)
        for vector in vectors[7:]:
            self.play(GrowArrow(vector), dot.animate.move_to(vector.get_end()))
        self.wait()

        # Step through terms individually
        labels.add_to_back(VectorizedPoint().move_to(vectors[0]))
        for n in range(5):
            rect1 = SurroundingRectangle(epii[2 * n + 2])
            rect2 = SurroundingRectangle(VGroup(vectors[n], labels[n]))
            self.play(
                FadeIn(rect1),
                self.fade_all_but(epii, 2 * n + 2),
                self.fade_all_but(vectors, n),
                self.fade_all_but(labels, n),
                dot.animate.set_opacity(0.1),
            )
            self.play(Transform(rect1, rect2))
            self.play(FadeOut(rect1))
        self.play(*(
            mob.animate.set_fill(opacity=1)
            for mob in [epii, vectors, labels]
        ))
        self.wait()

        # Swap out i for t
        e_to_it = self.get_series("it", use_parens=True, in_tex_color=GREEN)
        for sm1, sm2 in zip(e_to_it, epii):
            sm1.move_to(sm2)

        t_tracker = ValueTracker(PI)
        get_t = t_tracker.get_value

        t_label = Tex(R"t = 3.14", t2c={"t": GREEN})
        t_label.next_to(e_to_it, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
        t_rhs = t_label.make_number_changeable("3.14")
        t_rhs.add_updater(lambda m: m.set_value(get_t()))

        vectors.add_updater(lambda m: m.become(self.get_spiral_vectors(plane, get_t(), 20)))
        dot.f_always.move_to(vectors[-1].get_end)

        max_theta = TAU
        semi_circle = Arc(0, max_theta, radius=plane.x_axis.get_unit_size(), arc_center=plane.n2p(0))
        semi_circle.set_stroke(TEAL, 3)

        self.play(
            ReplacementTransform(epii, e_to_it, lag_ratio=0.01, run_time=2),
            FadeOut(labels),
            FadeIn(t_label),
        )
        self.add(vectors)
        self.play(t_tracker.animate.set_value(0), run_time=5)
        self.play(
            t_tracker.animate.set_value(max_theta),
            ShowCreation(semi_circle),
            run_time=12
        )
        self.play(t_tracker.animate.set_value(PI), run_time=6)
        self.wait()

    def get_series(self, in_tex="x", use_parens=False, in_tex_color=BLUE, buff=0.2):
        paren_tex = f"({in_tex})" if use_parens else in_tex
        kw = dict(t2c={in_tex: in_tex_color})
        terms = VGroup(
            Tex(fR"e^{{{in_tex}}}", **kw),
            Tex("="),
            Tex(fR"1"),
            Tex(R"+"),
            Tex(fR"{in_tex}", **kw),
            Tex(R"+"),
            Tex(fR"\frac{{1}}{{2}} {paren_tex}^2", **kw),
            Tex(R"+"),
            Tex(fR"\frac{{1}}{{6}} {paren_tex}^3", **kw),
            Tex(R"+"),
            Tex(fR"\frac{{1}}{{24}} {paren_tex}^4", **kw),
            Tex(R"+"),
            Tex(R"\cdots", **kw),
            Tex(R"+"),
            Tex(fR"\frac{{1}}{{n!}} {paren_tex}^n", **kw),
            Tex(R"+"),
        )
        terms.arrange(RIGHT, buff=buff)
        terms[0].scale(1.25, about_edge=DR)
        return terms

    def get_spiral_vectors(
        self,
        plane,
        t,
        n_terms=10,
        # colors=[GREEN, YELLOW, GREEN_E, YELLOW_E]
        colors=[GREEN_E, GREEN_C, GREEN_B, GREEN_A],
    ):
        values = [(t * 1j)**n / math.factorial(n) for n in range(n_terms)]
        vectors = VGroup(
            Arrow(plane.n2p(0), plane.n2p(value), buff=0, fill_color=color)
            for value, color in zip(values, it.cycle(colors))
        )
        for v1, v2 in zip(vectors, vectors[1:]):
            v2.shift(v1.get_end() - v2.get_start())
        return vectors

    def fade_all_but(self, group, index):
        group.target = group.generate_target()
        group.target.set_fill(opacity=0.4)
        group.target[index].set_fill(opacity=1)
        return MoveToTarget(group)


class ComplexExpGraph(InteractiveScene):
    s_value = 1j
    orientation1 = (-77, -1, 0, (1.01, -0.1, 3.21), 7.55)
    orientation2 = (-33, -2, 0, (1.68, -0.09, 3.79), 10.88)

    def construct(self):
        # Set up parts
        self.set_floor_plane("xz")
        frame = self.frame

        plane = ComplexPlane((-2, 2), (-2, 2))
        plane.scale(0.75)
        moving_plane = plane.copy()

        t_axis = NumberLine((0, 12))
        t_axis.rotate(90 * DEG, DOWN)
        t_axis.shift(-t_axis.n2p(0))

        self.add(plane)
        self.add(t_axis)

        # Trackers and graph
        t_tracker = ValueTracker(0)
        get_t = t_tracker.get_value
        s = self.s_value

        def get_z():
            return np.exp(s * get_t())

        def z_to_point(z):
            return plane.n2p(z) + get_t() * OUT

        moving_plane.add_updater(lambda m: m.move_to(t_axis.n2p(get_t())))
        point = GlowDot()
        point.add_updater(lambda m: m.move_to(z_to_point(get_z())))
        vector = Vector()
        vector.add_updater(lambda m: m.put_start_and_end_on(
            z_to_point(0),
            z_to_point(get_z())
        ))
        graph = TracedPath(vector.get_end, stroke_color=TEAL)

        t_label = Tex("t = 0.00", font_size=30)
        t_label_rhs = t_label.make_number_changeable("0.00")
        t_label_rhs.add_updater(lambda m: m.set_value(get_t()))
        t_label.add_updater(lambda m: m.next_to(moving_plane, UP, SMALL_BUFF))

        self.add(t_tracker, moving_plane, vector, point, graph)
        self.add(t_label)
        frame.reorient(*self.orientation1)
        self.play(
            frame.animate.reorient(*self.orientation2),
            t_tracker.animate.set_value(12).set_anim_args(rate_func=linear),
            VFadeIn(t_axis, time_span=(0, 1)),
            run_time=12
        )
        self.play(
            frame.animate.reorient(0, -89, -90, (0.06, -0.62, 5.27), 7.42).set_field_of_view(1 * DEG),
            FadeOut(plane),
            FadeOut(moving_plane),
            FadeOut(t_label),
            FadeOut(point),
            FadeOut(vector),
            run_time=3
        )


class AltComplexExpGraph(ComplexExpGraph):
    s_value = -0.2 + 1j
    orientation1 = (-37, -1, 0, (0.08, 0.1, 0.08), 6)
    orientation2 = (-21, -5, 0, (1.47, -0.44, 3.88), 12.29)


class SPlane(InteractiveScene):
    tex_to_color_map = {"s": YELLOW, "t": BLUE, R"\omega": PINK}
    s_plane_x_range = (-2, 2)
    s_label_font_size = 36
    s_label_config = dict(
        hide_zero_components_on_complex=True,
        include_sign=True,
        num_decimal_places=1,
    )

    def construct(self):
        # Trackers
        s_tracker = self.s_tracker = ComplexValueTracker(-1)
        t_tracker = self.t_tracker = ValueTracker(0)
        get_s = s_tracker.get_value
        get_t = t_tracker.get_value

        # Add s plane
        s_plane = self.get_s_plane()
        s_dot, s_label = self.get_s_dot_and_label(s_plane, get_s)
        self.add(s_plane, s_dot, s_label)

        # Add exp plane
        exp_plane = self.get_exp_plane()
        exp_plane_label = self.get_exp_plane_label(exp_plane)
        output_dot, output_label = self.get_output_dot_and_label(exp_plane, get_s, get_t)
        output_path = self.get_output_path(exp_plane, get_t, get_s)

        self.add(exp_plane, exp_plane_label, output_path, output_dot, output_label)

        # Add e^{st} graph
        axes = self.get_graph_axes()
        graph = self.get_dynamic_exp_graph(axes, get_s)
        v_line = self.get_graph_v_line(axes, get_t, get_s)

        self.add(axes, graph, v_line)

        # Move s around, end at i
        s_tracker.set_value(-1)
        self.play(s_tracker.animate.set_value(0.2), run_time=4)
        self.play(s_tracker.animate.set_value(0), run_time=2)
        self.play(s_tracker.animate.set_value(1j), run_time=3)
        self.wait()

        # Let time tick forward
        frame = self.frame
        self.play_time_forward(
            3 * TAU,
            added_anims=[frame.animate.set_x(3).set_height(12).set_anim_args(time_span=(6, 15))],
        )
        self.wait()
        self.play(
            t_tracker.animate.set_value(0),
            frame.animate.set_x(1.5).set_height(10),
            run_time=3
        )

        # Set s to 2i, then add vectors
        t2c = {"2i": YELLOW, **self.tex_to_color_map}
        exp_2it_label = Tex(R"e^{2i t}", t2c=t2c)
        exp_2it_label.move_to(exp_plane_label, RIGHT)

        self.play(s_tracker.animate.set_value(2j), run_time=2)
        self.play(
            FadeOut(exp_plane_label, 0.5 * UP),
            FadeIn(exp_2it_label, 0.5 * UP),
        )
        self.play_time_forward(TAU)

        # Show the derivative
        exp_plane.target = exp_plane.generate_target()
        exp_plane.target.align_to(axes.c2p(0, 0), LEFT)

        deriv_expression = Tex(R"\frac{d}{dt} e^{2i t} = 2i \cdot e^{2i t}", t2c=t2c)
        deriv_expression.next_to(exp_plane.target, RIGHT, aligned_edge=UP)

        self.play(
            LaggedStart(
                ReplacementTransform(exp_2it_label, deriv_expression["e^{2i t}"][0], path_arc=-90 * DEG),
                MoveToTarget(exp_plane),
                Write(deriv_expression[R"\frac{d}{dt}"]),
                lag_ratio=0.2
            ),
            frame.animate.reorient(0, 0, 0, (1, 0, 0.0), 9.25),
        )
        self.play(LaggedStart(
            Write(deriv_expression["="]),
            TransformFromCopy(*deriv_expression["e^{2i t}"], path_arc=-90 * DEGREES),
            FadeTransform(deriv_expression["e^{2i t}"][0][1:3].copy(), deriv_expression[R"2i"][1], path_arc=-90 * DEG),
            Write(deriv_expression[R"\cdot"]),
            lag_ratio=0.25,
            run_time=1.5
        ))
        self.add(deriv_expression)
        self.wait()

        # Step through derivative parts
        v_part, p_part, i_part, two_part = parts = VGroup(
            deriv_expression[R"\frac{d}{dt} e^{2i t}"][0],
            deriv_expression[R"e^{2i t}"][1],
            deriv_expression[R"i"][1],
            deriv_expression[R"2"][1],
        )
        colors = [GREEN, BLUE, YELLOW, YELLOW]
        labels = VGroup(Text("Velocity"), Text("Position"), Tex(R"90^{\circ}"), Text("Stretch"))
        for part, color, label in zip(parts, colors, labels):
            part.rect = SurroundingRectangle(part, buff=0.05)
            part.rect.set_stroke(color, 2)
            label.set_color(color)
            label.next_to(part.rect, DOWN)
            part.label = label

        p_vect, v_vect = vectors = self.get_pv_vectors(exp_plane, get_t, get_s)
        for vector in vectors:
            vector.suspend_updating()
        p_vect_copy = p_vect.copy().clear_updaters()

        self.play(
            ShowCreation(v_part.rect),
            FadeIn(v_part.label),
            GrowArrow(v_vect),
        )
        self.wait()
        self.play(
            ReplacementTransform(v_part.rect, p_part.rect),
            FadeTransform(v_part.label, p_part.label),
            GrowArrow(p_vect),
        )
        self.wait()
        self.play(
            ReplacementTransform(p_part.rect, i_part.rect),
            FadeTransform(p_part.label, i_part.label),
            p_vect_copy.animate.rotate(90 * DEG, about_point=exp_plane.n2p(0)).shift(p_vect.get_vector())
        )
        self.wait()
        self.play(
            ReplacementTransform(i_part.rect, two_part.rect),
            FadeTransform(i_part.label, two_part.label),
            Transform(p_vect_copy, v_vect, remover=True)
        )
        self.wait()
        self.play(FadeOut(two_part.rect), FadeOut(two_part.label))

        vectors.resume_updating()
        self.play_time_forward(TAU)

        # Label this angular frequency with omega
        imag_exp = Tex(R"e^{i \omega t}", t2c=self.tex_to_color_map, font_size=60)
        imag_exp.move_to(deriv_expression, LEFT)

        self.play(
            FadeOut(deriv_expression, 0.5 * UP),
            FadeIn(imag_exp, 0.5 * UP),
        )
        t_tracker.set_value(0)
        output_path.suspend_updating()
        self.play(s_tracker.animate.set_value(1.5j), run_time=3)
        output_path.resume_updating()
        self.play_time_forward(TAU * 4 / 3)

        # Move to other complex values, end at -0.5 + i
        t_max_tracker = ValueTracker(20 * TAU)
        new_output_path = self.get_output_path(exp_plane, t_max_tracker.get_value, get_s)
        output_path.match_updaters(new_output_path)
        t_tracker.set_value(0)

        self.play(
            FadeOut(imag_exp, time_span=(0, 1)),
            s_tracker.animate.set_value(-0.2 + 1.5j),
            run_time=2
        )
        self.play(s_tracker.animate.set_value(-0.2 + 1j), run_time=2)
        self.play(s_tracker.animate.set_value(-0.5 + 1j), run_time=2)
        self.play_time_forward(TAU)

        # Split up the exponential to e^{-0.5t} * e^{it}
        t2c = {"-0.5": YELLOW, "i": YELLOW, "t": BLUE}
        lines = VGroup(
            Tex(R"e^{(-0.5 + i)t}", t2c=t2c),
            Tex(R"\left(e^{-0.5t} \right) \left(e^{it} \right)", t2c=t2c)
        )
        lines.arrange(DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
        lines.next_to(exp_plane, RIGHT, LARGE_BUFF, aligned_edge=UP)

        dec_brace = Brace(lines[1][R"\left(e^{-0.5t} \right)"], DOWN, SMALL_BUFF)
        rot_brace = Brace(lines[1][R"\left(e^{it} \right)"], DOWN, SMALL_BUFF)
        dec_label = dec_brace.get_text("Decay")
        rot_label = rot_brace.get_text("Rotation")

        self.play(
            FadeIn(lines[0], time_span=(0.5, 1)),
            FadeTransform(s_label[-1].copy(), lines[0]["-0.5 + i"])
        )
        self.wait()
        self.play(
            TransformMatchingTex(lines[0].copy(), lines[1], run_time=1, lag_ratio=0.01)
        )
        self.wait()
        self.play(
            GrowFromCenter(dec_brace),
            FadeIn(dec_label)
        )
        self.wait()
        self.play(
            ReplacementTransform(dec_brace, rot_brace),
            FadeTransform(dec_label, rot_label),
        )
        self.wait()
        self.play(
            FadeOut(rot_brace),
            FadeOut(rot_label),
            t_tracker.animate.set_value(0).set_anim_args(run_time=3)
        )

        # Show multiplication by s
        s_vect = Arrow(s_plane.n2p(0), s_plane.n2p(get_s()), buff=0, fill_color=YELLOW)
        one_vect = Arrow(s_plane.n2p(0), s_plane.n2p(1), buff=0, fill_color=BLUE)
        arc = Line(
            s_plane.n2p(0.3),
            s_plane.n2p(0.3 * get_s()),
            path_arc=s_vect.get_angle(),
            buff=0.1
        )
        arc.add_tip(length=0.25, width=0.25)
        times_s_label = Tex(R"\times s")
        times_s_label.next_to(arc.pfp(0.5), UR, SMALL_BUFF)

        self.play(LaggedStart(
            FadeIn(one_vect),
            FadeIn(arc),
            FadeIn(times_s_label),
            FadeIn(s_vect),
        ))
        self.wait()
        self.play(
            TransformFromCopy(one_vect, s_vect, path_arc=s_vect.get_angle()),
            run_time=2
        )
        self.wait()
        self.play(
            TransformFromCopy(one_vect, p_vect),
            TransformFromCopy(s_vect, v_vect),
            run_time=2
        )

        # Show spiraling inward
        self.play_time_forward(2 * TAU)

        self.play(FadeOut(VGroup(arc, times_s_label, lines)))
        t_tracker.set_value(0)

        s_vect.add_updater(lambda m: m.put_start_and_end_on(s_plane.n2p(0), s_plane.n2p(get_s())))
        self.add(s_vect)

        # Tour various values on the s plane
        values = [
            -0.1 + 2j,
            -0.1 - 2j,
            -0.1 + 0.5j,
            +0.05 + 0.5j,
            -0.5 + 0.5j,
            -0.1 + 0.5j,
        ]
        for value in values:
            self.play(s_tracker.animate.set_value(value), run_time=5)
            if value == values[0]:
                self.play_time_forward(TAU)
                self.t_tracker.set_value(0)

        self.play_time_forward(4 * TAU)

    def get_s_plane(self):
        s_plane = ComplexPlane(self.s_plane_x_range, self.s_plane_x_range)
        s_plane.set_width(7)
        s_plane.to_edge(LEFT, buff=SMALL_BUFF)
        s_plane.add_coordinate_labels(font_size=16)
        return s_plane

    def get_s_dot_and_label(self, s_plane, get_s):
        s_dot = Group(
            Dot(radius=0.05, fill_color=YELLOW),
            GlowDot(color=YELLOW),
        )
        s_dot.add_updater(lambda m: m.move_to(s_plane.n2p(get_s())))

        s_label = Tex(R"s = +0.5", font_size=self.s_label_font_size)
        s_rhs = s_label.make_number_changeable("+0.5", **self.s_label_config)
        s_rhs.f_always.set_value(get_s)
        s_label.set_color(S_COLOR)
        s_label.set_backstroke(BLACK, 5)
        s_label.always.next_to(s_dot[0], UR, SMALL_BUFF)

        return Group(s_dot, s_label)

    def get_exp_plane(self, x_range=(-2, 2)):
        exp_plane = ComplexPlane(x_range, x_range)
        exp_plane.background_lines.set_stroke(width=1)
        exp_plane.faded_lines.set_stroke(opacity=0.25)
        exp_plane.set_width(4)
        exp_plane.to_corner(DR).shift(0.5 * LEFT)

        return exp_plane

    def get_exp_plane_label(self, exp_plane, font_size=60):
        label = Tex(R"e^{st}", font_size=font_size, t2c=self.tex_to_color_map)
        label.set_backstroke(BLACK, 5)
        label.next_to(exp_plane.get_corner(UL), DL, 0.2)
        return label

    def get_output_dot_and_label(self, exp_plane, get_s, get_t, label_direction=UR, s_tex="s"):
        output_dot = Group(
            TrueDot(color=GREEN),
            GlowDot(color=GREEN)
        )
        output_dot.add_updater(lambda m: m.move_to(exp_plane.n2p(np.exp(get_s() * get_t()))))

        output_label = Tex(Rf"e^{{{s_tex} \cdot 0.00}}", font_size=36, t2c=self.tex_to_color_map)
        t_label = output_label.make_number_changeable("0.00")
        t_label.set_color(BLUE)
        s_term = output_label[s_tex][0][0]
        t_label.set_height(s_term.get_height() * 1.2, about_edge=LEFT)
        t_label.f_always.set_value(get_t)
        t_label.always.match_y(s_term, DOWN)
        output_label.always.next_to(output_dot, label_direction, buff=SMALL_BUFF, aligned_edge=LEFT, index_of_submobject_to_align=0),
        output_label.set_backstroke(BLACK, 3)

        return Group(output_dot, output_label)

    def get_graph_axes(
        self,
        x_range=(0, 24),
        y_range=(-2, 2),
        width=15,
        height=2,
    ):
        axes = Axes(x_range, y_range, width=width, height=height)
        x_axis_label = Tex(R"t", font_size=36, t2c=self.tex_to_color_map)
        y_axis_label = Tex(R"\text{Re}\left[e^{st}\right]", font_size=36, t2c=self.tex_to_color_map)
        x_axis_label.next_to(axes.x_axis.get_right(), UP, buff=0.15)
        y_axis_label.next_to(axes.y_axis.get_top(), UP, SMALL_BUFF)
        axes.add(x_axis_label)
        axes.add(y_axis_label)
        axes.next_to(ORIGIN, RIGHT, MED_LARGE_BUFF)
        axes.to_edge(UP, buff=0.5)
        x_axis_label.shift_onto_screen(buff=MED_LARGE_BUFF)
        return axes

    def get_dynamic_exp_graph(self, axes, get_s, delta_t=0.1, stroke_color=TEAL, stroke_width=3):
        graph = Line().set_stroke(stroke_color, stroke_width)
        t_samples = np.arange(*axes.x_range[:2], 0.1)

        def update_graph(graph):
            s = get_s()
            values = np.exp(s * t_samples)
            xs = values.astype(np.complex128).real
            graph.set_points_smoothly(axes.c2p(t_samples, xs))

        graph.add_updater(update_graph)
        return graph

    def get_graph_v_line(self, axes, get_t, get_s):
        v_line = Line(DOWN, UP)
        v_line.set_stroke(WHITE, 2)
        v_line.f_always.put_start_and_end_on(
            lambda: axes.c2p(get_t(), 0),
            lambda: axes.c2p(get_t(), np.exp(get_s() * get_t()).real),
        )
        return v_line

    def get_output_path(self, exp_plane, get_t, get_s, delta_t=1 / 30, color=TEAL, stroke_width=2):
        path = VMobject()
        path.set_points([ORIGIN])
        path.set_stroke(color, stroke_width)

        def get_path_points():
            t_range = np.arange(0, get_t(), delta_t)
            if len(t_range) == 0:
                t_range = np.array([0])
            values = np.exp(t_range * get_s())
            return np.array([exp_plane.n2p(z) for z in values])

        path.f_always.set_points_smoothly(get_path_points)
        return path

    def play_time_forward(self, time, added_anims=[]):
        self.t_tracker.set_value(0)
        self.play(
            self.t_tracker.animate.set_value(time).set_anim_args(rate_func=linear),
            *added_anims,
            run_time=time,
        )

    def get_pv_vectors(self, exp_plane, get_t, get_s, thickness=3, colors=[BLUE, YELLOW]):
        p_vect = Vector(RIGHT, fill_color=colors[0], thickness=thickness)
        v_vect = Vector(RIGHT, fill_color=colors[1], thickness=thickness)
        p_vect.add_updater(lambda m: m.put_start_and_end_on(
            exp_plane.n2p(0),
            exp_plane.n2p(np.exp(get_t() * get_s()))
        ))
        v_vect.add_updater(lambda m: m.put_start_and_end_on(
            exp_plane.n2p(0),
            exp_plane.n2p(get_s() * np.exp(get_t() * get_s()))
        ).shift(p_vect.get_vector()))

        return VGroup(p_vect, v_vect)

    ###

    def setup_for_square_frame(self):
        # For an insert
        axes.next_to(s_plane, UP, LARGE_BUFF, aligned_edge=LEFT)
        exp_plane.match_height(s_plane).next_to(s_plane, RIGHT, buff=1.5)
        exp_plane_label.set_height(1).next_to(exp_plane, RIGHT, aligned_edge=UP)
        output_label.set_fill(opacity=0).set_stroke(opacity=0)
        self.add(exp_plane_label)

        axes[:-2].stretch(2.5, 1, about_edge=DOWN)
        axes.x_axis.ticks.stretch(1 / 2.5, 1)
        axes[-2].next_to(axes.x_axis.get_right(), UP, MED_SMALL_BUFF)
        axes[-1].next_to(axes.y_axis.get_top(), RIGHT, MED_SMALL_BUFF)

        self.frame.reorient(0, 0, 0, (1.0, 2.68, 0.0), 16.00)

    def old_material(self):
        # Collapse the graph
        output_dot = GlowDot(color=GREEN)
        output_dot.move_to(axes.x_axis.n2p(1))
        output_label = Tex(R"e^{s \cdot 0.00}", **tex_kw, font_size=36)
        t_tracker.set_value(0)
        t_label = output_label.make_number_changeable("0.00")
        t_label.set_color(BLUE)
        t_label.match_height(output_label["s"], about_edge=LEFT)
        t_label.f_always.set_value(get_t)
        output_label.add_updater(lambda m: m.next_to(output_dot, UP, SMALL_BUFF, LEFT).shift(0.2 * DR))

        graph.clear_updaters()
        self.remove(axes)
        self.play(LaggedStart(
            FadeOut(VGroup(axes.x_axis, x_axis_label, graph)),
            AnimationGroup(
                Rotate(axes.y_axis, -90 * DEG),
                TransformMatchingTex(y_axis_label, output_label, run_time=1.5),
            ),
            FadeIn(output_dot),
            lag_ratio=0.5
        ))


class FamilyOfRealExp(InteractiveScene):
    def construct(self):
        # Graphs
        axes = Axes((-1, 8), (-1, 5))
        axes.set_height(FRAME_HEIGHT - 1)

        s_tracker = ValueTracker(0.5)
        get_s = s_tracker.get_value
        graph = axes.get_graph(lambda t: np.exp(t))
        graph.set_stroke(BLUE)
        axes.bind_graph_to_func(graph, lambda t: np.exp(get_s() * t))

        label = Tex(R"e^{st}", font_size=90)
        label.move_to(UP)
        label["s"].set_color(YELLOW)

        self.add(axes, label)
        self.play(ShowCreation(graph, suspend_mobject_updating=True))
        self.play(
            s_tracker.animate.set_value(-1),
            graph.animate.set_color(YELLOW),
            run_time=4
        )
        self.wait()


class ForcedOscillatorSolutionForm(InteractiveScene):
    def construct(self):
        # Create linear combination
        exp_texs = [Rf"e^{{s_{n} t}}" for n in range(1, 5)]
        const_texs = [Rf"c_{n}" for n in range(1, 5)]
        terms = [" ".join(pair) for pair in zip(const_texs, exp_texs)]
        solution = Tex("x(t) = " + " + ".join(terms), isolate=[*exp_texs, *const_texs])
        solution.to_edge(RIGHT)

        solution[re.compile(r's_\w+')].set_color(YELLOW)
        solution[re.compile(r'c_\w+')].set_color(BLUE)

        cut_index = solution.submobjects.index(solution["+"][1][0])
        first_two = solution[:cut_index]
        last_two = solution[cut_index:]

        first_two.save_state()
        first_two.to_edge(RIGHT, buff=1.5)

        self.add(first_two)

        # Not this
        ex_mark = Exmark(font_size=72).set_color(RED)
        checkmark = Checkmark(font_size=72).set_color(GREEN)
        ex_mark.next_to(first_two, UP, MED_LARGE_BUFF, aligned_edge=LEFT)
        checkmark.next_to(first_two.saved_state, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        nope = Text("Nope!", font_size=60).set_fill(border_width=4)
        nope.match_color(ex_mark)
        nope.next_to(ex_mark, RIGHT)

        actually = Text("Actually...", font_size=60)
        actually.set_fill(border_width=2)
        actually.match_color(checkmark)
        actually.next_to(checkmark, RIGHT, SMALL_BUFF, aligned_edge=DOWN, index_of_submobject_to_align=0)

        self.play(Write(ex_mark), Write(nope))

        # Freely tune coefficients
        c_trackers = ValueTracker(0).replicate(2)

        def get_c_values():
            return [tracker.get_value() for tracker in c_trackers]

        number_lines = VGroup(
            NumberLine((-3, 3), width=2).rotate(90 * DEG).next_to(solution[c_tex], DOWN)
            for c_tex in const_texs[:2]
        )
        for line in number_lines:
            line.set_width(0.1, stretch=True)
            line.add_numbers(font_size=12, direction=LEFT, buff=0.1)

        tips = ArrowTip().rotate(PI).set_height(0.2).replicate(2)
        tips.set_color(BLUE)

        def update_tips(tips):
            for tip, line, value in zip(tips, number_lines, get_c_values()):
                tip.move_to(line.n2p(value), LEFT)
            return tips

        tips.add_updater(update_tips)

        c_labels = VGroup(DecimalNumber(0, font_size=24) for _ in range(2))

        def update_c_labels(c_labels):
            for label, tip, value in zip(c_labels, tips, get_c_values()):
                label.set_value(value)
                label.next_to(tip, RIGHT, SMALL_BUFF)

        c_labels.add_updater(update_c_labels)

        def random_tuning_animation(run_time=2, lag_ratio=0.25):
            return LaggedStart(
                *(
                    tracker.animate.set_value(random.uniform(-3, 3))
                    for tracker in c_trackers
                ),
                lag_ratio=lag_ratio,
                run_time=run_time,
            )

        self.play(
            FadeIn(number_lines),
            VFadeIn(tips),
            VFadeIn(c_labels),
            random_tuning_animation()
        )
        for _ in range(6):
            self.play(random_tuning_animation())
        self.wait()

        # Show four particulcar exponentials
        plane = ComplexPlane((-3, 3), (-2, 2))
        plane.set_height(3.25)
        plane.to_corner(UL)
        plane.add_coordinate_labels(font_size=16)
        plane.coordinate_labels[-1].set_opacity(0)

        s_values = [1.5j, -1.5j, -0.3 + 1.0j, -0.3 - 1.0j]
        s_dots = Group(
            GlowDot(plane.n2p(s))
            for s in s_values
        )
        s_labels = VGroup(
            Tex(Rf"s_{n}", font_size=24).set_color(YELLOW).next_to(dot, vect, buff=-0.1)
            for n, dot, vect in zip(it.count(1), s_dots, [RIGHT, RIGHT, LEFT, LEFT])
        )

        self.play(LaggedStart(
            FadeOut(number_lines, lag_ratio=0.1),
            FadeOut(tips, lag_ratio=0.1),
            FadeOut(c_labels, lag_ratio=0.1),
            FadeOut(VGroup(ex_mark, nope), LEFT),
            FadeIn(VGroup(checkmark, actually), LEFT),
            Restore(first_two),
            FadeIn(last_two, LEFT),
            run_time=2
        ))
        self.play(
            FadeIn(plane),
            LaggedStartMap(FadeIn, s_dots),
            LaggedStart(
                *(
                    FadeTransform(solution[f"s_{n + 1}"].copy(), s_labels[n])
                    for n in range(4)
                )
            ),
        )
        self.wait()

        # Comment on constants
        const_rects = VGroup(
            SurroundingRectangle(solution[c_tex], buff=0.075)
            for c_tex in const_texs
        )
        const_rects.set_stroke(BLUE, 2)

        underlines = VGroup(
            Line(c1.get_bottom(), c2.get_bottom(), path_arc=40 * DEG)
            for c1, c2 in it.combinations(const_rects, 2)
        )
        underlines.set_stroke(TEAL, 2)
        underlines.insert_n_curves(10)

        underlines = VGroup(
            Vector(0.75 * UP, thickness=4).next_to(rect, DOWN, buff=0)
            for rect in const_rects
        )
        underlines.set_fill(BLUE)

        constraint_words = TexText("Only specific $c_n$ work")
        constraint_words.set_fill(BLUE, border_width=1)
        constraint_words.match_width(underlines)
        constraint_words.next_to(underlines, DOWN, buff=SMALL_BUFF)

        self.play(
            FadeIn(constraint_words, lag_ratio=0.1),
            FadeOut(checkmark),
            FadeOut(actually),
            LaggedStartMap(ShowCreation, const_rects, lag_ratio=0.25),
            LaggedStartMap(GrowArrow, underlines),
        )
        self.play(FadeOut(const_rects, lag_ratio=0.1))

        # Add exponential parts
        if False:
            # For an insertion
            for term, s in zip(exp_texs, s_values):
                exp_diagram = self.get_exponential_diagram(solution[term], s)
                self.add(exp_diagram)
            self.wait(24)

        # Ask about each part
        term_rects = VGroup(
            SurroundingRectangle(solution[term], buff=0.1).set_stroke(TEAL, 2)
            for term in terms
        )
        s_rects = VGroup(
            SurroundingRectangle(solution[exp_tex][0][1:3], buff=0.05).set_stroke(YELLOW, 2)
            for exp_tex in exp_texs
        )

        moving_rects = const_rects.copy()
        self.remove(const_rects)

        anim_kw = dict(lag_ratio=0.25, run_time=1.5)
        self.play(
            FadeOut(constraint_words),
            FadeOut(underlines),
            Transform(moving_rects, term_rects, **anim_kw)
        )
        self.wait()
        self.play(Transform(moving_rects, s_rects, **anim_kw))
        self.wait()
        self.play(Transform(moving_rects, const_rects, **anim_kw))
        self.wait()
        self.play(FadeOut(moving_rects, **anim_kw))

    def get_exponential_diagram(self, term, s, c=1.0, color=PINK):
        plane = ComplexPlane((-1, 1), (-1, 1))
        plane.set_width(1.25)
        plane.next_to(term, UP)

        t_tracker = ValueTracker()
        get_t = t_tracker.get_value
        t_tracker.add_updater(lambda m, dt: m.increment_value(dt))

        vector = Vector(thickness=2, fill_color=color)
        vector.add_updater(lambda m: m.put_start_and_end_on(
            plane.n2p(0),
            plane.n2p(c * np.exp(s * get_t())),
        ))

        tail = TracingTail(vector.get_end, stroke_color=color, time_traced=2, stroke_width=(0, 4))
        path = TracedPath(vector.get_end, stroke_color=color, stroke_width=1, stroke_opacity=0.75)

        return Group(plane, t_tracker, vector, tail, path)


class BreakingDownFunctions(ForcedOscillatorSolutionForm):
    def construct(self):
        # A s plane on the left, output plane on the right
        s_plane, out_plane = planes = VGroup(
            ComplexPlane((-2, 2), (-2, 2)),
            ComplexPlane((-2, 2), (-2, 2)),
        )
        for plane in planes:
            plane.set_height(5)
            plane.add_coordinate_labels(font_size=16)

        out_plane.center().to_edge(DOWN)
        s_plane.move_to(out_plane).to_edge(LEFT)

        self.add(out_plane)

        # Write a function as a sum, above the output plane
        n_range = list(range(1, 6))
        exp_texs = [Rf"e^{{s_{n} t}}" for n in n_range]
        const_texs = [Rf"c_{n}" for n in n_range]
        terms = [" ".join(pair) for pair in zip(const_texs, exp_texs)]
        solution = Tex("x(t) = " + " + ".join(terms), isolate=[*exp_texs, *const_texs])

        solution[re.compile(r's_\w+')].set_color(YELLOW)
        solution[re.compile(r'c_\w+')].set_color(BLUE)

        solution.next_to(out_plane, UP)

        self.add(solution)

        # Exp animations
        s_values = [-0.2 + 2j, -0.2 - 2j, -0.1 + 1j, -0.1 - 1j, -0.2]
        c_values = [1j, -1j, 0.8, 0.8, -0.75]
        colors = color_gradient([PINK, MAROON_B], len(n_range), interp_by_hsl=False)
        exp_diagrams = Group()
        for term, s, c, color in zip(terms, s_values, c_values, colors):
            part = solution[term]
            if len(part) == 0:
                continue
            exp_diagram = self.get_exponential_diagram(part, s, c, color)
            self.add(*exp_diagram)
            exp_diagrams.add(exp_diagram)

        # Set up the output
        center_point = VectorizedPoint(out_plane.n2p(0))
        all_vects = Vector().replicate(len(n_range))
        scale_factor = out_plane.x_axis.get_unit_size() / exp_diagrams[0][0].x_axis.get_unit_size()

        for diagram, vect, previous in zip(exp_diagrams, all_vects, [center_point, *all_vects]):
            vect.clone = diagram[2]
            vect.previous = previous
            vect.add_updater(lambda m: m.become(m.clone).scale(scale_factor))
            vect.add_updater(lambda m: m.shift(m.previous.get_end() - m.get_start()))

        self.add(all_vects)

        # Add output graph
        graph = VMobject()
        graph.set_stroke(RED, 3)
        graph.start_new_path(all_vects[-1].get_end())

        def update_graph(graph, dt):
            graph.shift(0.25 * dt * DOWN)
            graph.add_line_to(all_vects[-1].get_end())

        graph.add_updater(update_graph)

        self.add(graph)

        self.wait(30)
        self.play(VFadeOut(graph), VFadeOut(all_vects))
        exp_diagrams.suspend_updating()

        # Collapse to the s plane
        out_plane.generate_target()
        plane_group = VGroup(s_plane, out_plane.target)
        plane_group.arrange(RIGHT, buff=2)
        plane_group.to_edge(DOWN)
        compact_equation = Tex(
            R"x(t) = \sum_{n=1}^{N} c_n e^{{s_n} t}",
            t2c={"c_n": BLUE, "s_n": YELLOW},
            isolate=["n=1", "N"]
        )
        compact_equation.next_to(out_plane.target, UP)
        compact_equation_start = compact_equation[:-4]

        s_dot = GlowDot()
        s_dot.move_to(s_plane.n2p(s_values[2]))
        s_label = Tex(R"s").set_color(YELLOW)
        s_label.always.next_to(s_dot, UL, buff=-0.1)
        s_plane_title = Text(R"S-plane", font_size=60)
        s_plane_title.next_to(s_plane, UP)
        s_plane_title.set_color(YELLOW)

        exp_graph = VMobject()
        exp_graph.set_stroke(PINK, 2)

        def update_exp_graph(exp_graph):
            s = s_plane.p2n(s_dot.get_center())
            anchors = np.array([
                out_plane.n2p(np.exp(s * t))
                for t in np.arange(0, 100, 0.1)
            ])
            exp_graph.set_points_as_corners(anchors)

        exp_graph.add_updater(update_exp_graph)

        self.play(LaggedStart(
            MoveToTarget(out_plane),
            LaggedStart(*(
                exp_diagram.animate.scale(scale_factor).move_to(out_plane.target)
                for exp_diagram in exp_diagrams
            ), lag_ratio=0.01),
            AnimationGroup(*(
                ReplacementTransform(solution[t1], compact_equation[t2])
                for t1, t2 in [
                    ("x(t) = ", "x(t) = "),
                    (re.compile(r'c_\w+'), "c_n"),
                    (re.compile(r's_\w+'), "s_n"),
                    ("e", "e"),
                    ("t", "t"),
                    ("+", R"\sum_{n=1}^{N}"),
                ]
            )),
            lag_ratio=0.2,
            run_time=2
        ))
        exp_graph.update()
        self.play(
            FadeIn(s_plane),
            TransformFromCopy(compact_equation["s_n"][0], s_label),
            FadeTransform(compact_equation["s_n"][0].copy(), s_dot),
            Write(s_plane_title),
            FadeOut(exp_diagrams[2]),
            FadeIn(exp_graph, suspend_mobject_updating=True),
            compact_equation_start.animate.set_opacity(0.4)
        )
        self.remove(exp_diagrams)
        self.wait()

        # Growth, decay and oscillation
        arrows = VGroup(
            Arrow(s_plane.n2p(0), s_plane.n2p(z), thickness=4, fill_color=GREY_A)
            for z in [2, -2, 2j, -2j]
        )
        arrows.set_fill(GREY_A, 1)
        arrow_labels = VGroup(
            Text("Growth", font_size=36).next_to(arrows[0], UP, buff=0),
            Text("Decay", font_size=36).next_to(arrows[1], UP, buff=0),
            Text("Oscillation", font_size=36).rotate(-90 * DEG).next_to(arrows[3], RIGHT, buff=SMALL_BUFF),
        )
        rot_vect = Vector(2 * RIGHT, fill_color=RED, thickness=5)
        rot_vect.shift(out_plane.n2p(0) - rot_vect.get_start())
        rot_vect.rotate(-45 * DEG, about_point=out_plane.n2p(0))
        rot_vect.add_updater(lambda m, dt: m.rotate(2 * dt, about_point=out_plane.n2p(0)))

        self.play(
            s_dot.animate.shift(0.2 * RIGHT).set_anim_args(run_time=1),
            GrowArrow(arrows[0]),
            FadeIn(arrow_labels[0]),
        )
        self.play(
            s_dot.animate.shift(0.2 * LEFT).set_anim_args(run_time=1),
            GrowArrow(arrows[1]),
            FadeIn(arrow_labels[1]),
        )
        self.play(
            # s_dot.animate.shift(3 * DOWN).set_anim_args(run_time=4, rate_func=there_and_back),
            GrowArrow(arrows[2]),
            GrowArrow(arrows[3]),
            FadeIn(arrow_labels[2]),
            VFadeIn(rot_vect)
        )
        self.wait()
        self.play(VFadeOut(rot_vect))

        # Show multiple s
        frame = self.frame
        s_values[:2] = [-1.5 + 0.5j, -1.5 - 0.5j]
        s_values.extend([-0.8 + 1.5j, -0.8 - 1.5j])
        s_dots = Group(GlowDot(s_plane.n2p(s)) for s in s_values)

        self.play(
            FadeOut(arrows),
            FadeOut(arrow_labels),
            FadeOut(out_plane),
            FadeOut(exp_graph),
            FadeIn(s_dots, lag_ratio=0.5),
            FadeOut(s_dot),
            FadeOut(s_label),
            compact_equation.animate.set_height(2.0).set_opacity(1).next_to(s_plane, RIGHT, LARGE_BUFF),
            frame.animate.match_y(s_plane),
        )
        self.wait()

        # Infinite
        inf = Tex(R"\infty", font_size=60)
        N = compact_equation["N"][0]
        inf.move_to(N)

        dot_line = Group(
            GlowDot(s_plane.n2p(complex(-0.5, b)))
            for b in np.linspace(-2, 2, 25)
        )

        self.play(
            FlashAround(N, time_width=1),
            # Transform(N, inf, path_arc=20 * DEG),
            FadeTransform(N, inf, path_arc=20 * DEG),
            ShowIncreasingSubsets(dot_line, run_time=5, rate_func=linear),
            ReplacementTransform(s_dots, dot_line[:len(s_dots)].copy().set_opacity(0))
        )

        # Continuous range
        integral_eq = Tex(
            R"x(t) = \int_{\gamma} c(s) e^{st} ds",
            t2c={"s": YELLOW, R"\gamma": YELLOW}
        )
        integral_eq.replace(compact_equation, dim_to_match=1)
        integral_eq.shift(0.5 * RIGHT)
        line = Line(dot_line.get_bottom(), dot_line.get_top())
        line.set_stroke(YELLOW, 2)
        thick_line = line.copy().set_stroke(width=6)
        thick_line.insert_n_curves(100)

        self.play(
            LaggedStart(*(
                FadeTransform(compact_equation[t1], integral_eq[t2])
                for t1, t2 in [
                    ("x(t) = ", "x(t) ="),
                    (R"\sum_{n=1}^{N}", R"\int_{\gamma}"),
                    (R"n=1", R"\gamma"),
                    ("c_n", "c(s)"),
                    (R"e^{{s_n} t}", R"e^{st}"),
                ]
            )),
            FadeIn(integral_eq[R"ds"]),
            LaggedStartMap(FadeOut, dot_line, lag_ratio=0.5, scale=0.25, time_span=(0.3, 2)),
            VShowPassingFlash(thick_line, run_time=2),
            ShowCreation(line, run_time=2),
        )
        self.wait()


class Thumbnail(InteractiveScene):
    def construct(self):
        # Spiral
        spiral_color = TEAL
        s = -0.15 + 2j
        max_t = 3
        thick_stroke_width = (5, 25)

        plane = ComplexPlane(
            (-4, 4),
            (-2, 2)
        )
        plane.set_height(11)
        plane.center()
        plane.add_coordinate_labels(font_size=24)

        plane.axes.set_stroke(WHITE, 5)
        plane.background_lines.set_stroke(BLUE, 3)
        plane.faded_lines.set_stroke(BLUE, 2, 0.25)

        curve = ParametricCurve(
            lambda t: plane.n2p(np.exp(s * t)),
            t_range=(0, 40, 0.1)
        )
        partial_curve = ParametricCurve(
            lambda t: plane.n2p(np.exp(s * t)),
            t_range=(0, max_t, 0.1)
        )
        curve.set_stroke(spiral_color, 2, 0.5)
        partial_curve.set_stroke(spiral_color, width=thick_stroke_width, opacity=(0.5, 1))

        dot = Group(TrueDot(radius=0.2), GlowDot(radius=0.75))
        dot.set_color(spiral_color)
        dot.move_to(partial_curve.get_end())

        self.add(plane, curve)
        self.add(partial_curve)
        self.add(dot)

        vectors = VGroup(
            self.get_vector(plane, s, t, color=BLUE)
            for t in np.linspace(0, max_t, 10)
        )
        self.add(vectors)

    def get_vector(self, plane, s, t, scale_factor=0.5, thickness=5, color=YELLOW):
        vect = Vector(RIGHT, thickness=thickness, fill_color=color)
        vect.put_start_and_end_on(plane.n2p(0), scale_factor * plane.n2p(s * np.exp(s * t)))
        vect.shift(plane.n2p(np.exp(s * t)) - plane.n2p(0))
        return vect

    def get_formula(self):
        # Formula
        formula = Tex(R"e^{st}", t2c={"s": YELLOW, "t": BLUE}, font_size=200)
        formula.next_to(plane, LEFT, buff=2.0)

        s_rect = SurroundingRectangle(formula["s"], buff=0.1)
        s_rect.set_stroke(WHITE, 2)

        abi = Tex("a + bi", font_size=72)
        abi.next_to(formula, UP, LARGE_BUFF)
        abi.to_edge(LEFT, buff=LARGE_BUFF)

        arrow = Arrow(s_rect.get_top(), abi.get_corner(DR), buff=0.05)

        self.add(formula)
        self.add(s_rect)
        self.add(arrow)
        self.add(abi)


class Thumbnail2(InteractiveScene):
    def construct(self):
        # Test
        theta = 140 * DEG
        z = np.exp(theta * 1j)

        plane_color = BLUE
        path_color = YELLOW
        vect_color = WHITE

        plane = ComplexPlane(
            (-4, 4),
            (-2, 2)
        )
        plane.set_height(10)
        plane.center()
        plane.add_coordinate_labels(font_size=24)
        plane.axes.set_stroke(WHITE, 3)
        plane.background_lines.set_stroke(plane_color, 4)
        plane.faded_lines.set_stroke(plane_color, 3, 0.35)

        unit_size = plane.x_axis.get_unit_size()

        arrow = Arrow(
            plane.n2p(1),
            plane.n2p(z),
            buff=0,
            path_arc=theta,
            thickness=12,
            fill_color=YELLOW,
        )
        arrow.scale(0.965, about_point=plane.n2p(1))
        arrow.set_fill(border_width=3)

        path = Arc(0, theta, radius=unit_size)
        path.set_stroke(path_color, width=(2, 30))
        dot = Group(TrueDot(radius=0.2), GlowDot(radius=0.75))
        dot.set_color(path_color)
        dot.move_to(path.get_end())

        n_vects = 8
        vectors = VGroup(
            Vector(2.5 * UP, thickness=8).set_fill(vect_color, opacity**2).put_start_on(plane.n2p(1)).rotate(phi, about_point=plane.n2p(0))
            for phi, opacity in zip(np.linspace(0, theta, n_vects), np.linspace(0.5, 1, n_vects))
        )
        vectors.set_fill(border_width=3)

        circle = Circle(radius=unit_size)
        circle.set_stroke(GREY, 5)

        d_line = DashedLine(plane.n2p(0), plane.n2p(z))
        d_line.set_stroke(WHITE, 3)
        arc = Arc(0, theta, radius=0.5)
        arc.set_stroke(WHITE, 5)
        theta_label = Tex(R"\theta", font_size=72)
        theta_label.next_to(arc.pfp(0.5), UR, SMALL_BUFF)

        self.add(plane)
        self.add(circle)
        self.add(d_line)
        self.add(vectors)
        self.add(path)
        self.add(dot)
        self.add(arc)
        self.add(theta_label)


### For Main Laplace video


class RecapSPlane(SPlane):
    def construct(self):
        # Trackers
        s_tracker = self.s_tracker = ComplexValueTracker(-1)
        t_tracker = self.t_tracker = ValueTracker(0)
        get_s = s_tracker.get_value
        get_t = t_tracker.get_value

        # Add s plane
        s_plane = self.get_s_plane()
        s_dot, s_label = self.get_s_dot_and_label(s_plane, get_s)
        self.add(s_plane, s_dot, s_label)

        # Add exp plane
        exp_plane = self.get_exp_plane()
        exp_plane_label = self.get_exp_plane_label(exp_plane)
        output_dot, output_label = self.get_output_dot_and_label(exp_plane, get_s, get_t)
        output_path = self.get_output_path(exp_plane, get_t, get_s)
        output_path.set_clip_plane(RIGHT, -s_plane.get_x(RIGHT))

        max_t = 50
        output_path_preview = self.get_output_path(exp_plane, lambda: max_t, get_s)
        output_path_preview.set_stroke(opacity=0.5)
        for path in [output_path, output_path_preview]:
            path.set_clip_plane(RIGHT, -s_plane.get_x(RIGHT))

        self.add(exp_plane, exp_plane_label, output_path_preview, output_path, output_dot, output_label)

        # Add e^{st} graph
        axes = self.get_graph_axes()
        axes.x_axis.scale(0.5, 0, about_edge=LEFT)
        graph = self.get_dynamic_exp_graph(axes, get_s)
        v_line = self.get_graph_v_line(axes, get_t, get_s)
        graph.set_clip_plane(UP, -axes.get_y(DOWN))
        v_line.set_clip_plane(UP, -axes.get_y(DOWN))
        axes_background = BackgroundRectangle(axes)
        axes_background.set_fill(BLACK, 1)
        axes_background.align_to(s_plane.get_right(), LEFT).shift(1e-2 * RIGHT)
        axes_background.stretch(1.2, 1, about_edge=DOWN)

        self.add(axes_background, axes, graph, v_line)

        # Pre-preamble
        s_tracker.set_value(-0.3)
        self.play(s_tracker.animate.increment_value(1.5j), run_time=4)
        self.play(t_tracker.animate.set_value(TAU / 1.5), run_time=5, rate_func=linear)
        self.play(s_tracker.animate.set_value(1.5j), run_time=3)
        t_tracker.set_value(0)

        # Play around
        s_tracker.set_value(0)
        self.play(s_tracker.animate.increment_value(1.5j), run_time=4)
        t_tracker.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(t_tracker)
        self.wait(4.5)
        self.play(s_tracker.animate.increment_value(-0.2))
        self.wait(2)
        for step in [-0.8, 1.2, -0.4]:
            self.play(s_tracker.animate.increment_value(step), run_time=3)
        self.wait()


class DefineSPlane(SPlane):
    def construct(self):
        # Trackers
        s_tracker = self.s_tracker = ComplexValueTracker(-1)
        t_tracker = self.t_tracker = ValueTracker(0)
        get_s = s_tracker.get_value
        get_t = t_tracker.get_value

        # Add s plane
        s_plane = self.get_s_plane()
        s_dot, s_label = self.get_s_dot_and_label(s_plane, get_s)
        self.add(s_plane, s_dot, s_label)

        # Add exp plane
        exp_plane = self.get_exp_plane()
        exp_plane_label = self.get_exp_plane_label(exp_plane)
        # output_dot, output_label = self.get_output_dot_and_label(exp_plane, get_s, get_t)
        output_dot, output_label = self.get_output_dot_and_label(exp_plane, get_s, get_t, s_tex="a")
        output_path = self.get_output_path(exp_plane, get_t, get_s)
        output_path.set_clip_plane(RIGHT, -s_plane.get_x(RIGHT))

        max_t = 50
        output_path_preview = self.get_output_path(exp_plane, lambda: max_t, get_s)
        output_path_preview.set_stroke(opacity=0.5)
        for path in [output_path, output_path_preview]:
            path.set_clip_plane(RIGHT, -s_plane.get_x(RIGHT))

        self.add(exp_plane, exp_plane_label, output_path_preview, output_path, output_dot, output_label)

        # Add e^{st} graph
        axes = self.get_graph_axes()
        axes.x_axis.scale(0.5, 0, about_edge=LEFT)
        graph = self.get_dynamic_exp_graph(axes, get_s)
        v_line = self.get_graph_v_line(axes, get_t, get_s)
        graph.set_clip_plane(UP, -axes.get_y(DOWN))
        v_line.set_clip_plane(UP, -axes.get_y(DOWN))
        axes_background = BackgroundRectangle(axes)
        axes_background.set_fill(BLACK, 1)
        axes_background.align_to(s_plane.get_right(), LEFT).shift(1e-2 * RIGHT)
        axes_background.stretch(1.2, 1, about_edge=DOWN)

        self.add(axes_background, axes, graph, v_line)

        # Test
        s_tracker.set_value(-0.1 + 1.5j)
        t_tracker.clear_updaters()
        t_tracker.set_value(0)
        t_tracker.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(t_tracker)
        self.wait(20)

        # Go!
        t_tracker.clear_updaters()
        t_tracker.set_value(0)
        t_tracker.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(t_tracker)
        s_tracker.set_value(0)
        self.play(s_tracker.animate.set_value(-0.1 + 1.5j), run_time=3)
        self.wait(3)

        # Name the plane
        frame = self.frame
        s_plane_name = Text("S-plane", font_size=90)
        s_plane_name.next_to(s_plane, UP)
        s_plane_name.set_color(YELLOW)

        self.play(
            frame.animate.reorient(0, 0, 0, (0.49, 0.34, 0.0), 9.65).set_anim_args(time_span=(0, 2)),
            FlashAround(s_plane, time_width=1.5, buff=MED_SMALL_BUFF, stroke_width=5),
            run_time=4,
        )
        self.wait()
        self.play(s_tracker.animate.set_value(-0.2 - 1j), run_time=3)
        self.wait()
        self.play(Write(s_plane_name))
        self.wait(3)

        # Show exp pieces
        s_samples = [complex(a, b) for a in range(-2, 3) for b in range(-2, 3)]
        exp_pieces = VGroup(
            get_exp_graph_icon(s).move_to(s_plane.n2p(0.85 * s))
            for s in s_samples
        )

        dots = VGroup(Dot(radius=0.1).move_to(piece) for piece in exp_pieces)
        dots.set_color(YELLOW)

        self.play(
            LaggedStartMap(FadeIn, dots, scale=5, lag_ratio=0.05, run_time=2),
            VFadeOut(output_label),
            FadeOut(output_dot),
            VFadeOut(s_label),
            FadeOut(s_dot),
        )
        t_tracker.clear_updaters()
        self.wait()
        self.play(LaggedStart(
            (FadeTransform(dot, piece)
            for dot, piece in zip(dots, exp_pieces)),
            lag_ratio=0.05,
            run_time=3,
            group_type=Group,
        ))
        self.wait()

        # Associate with complex
        rect = SurroundingRectangle(exp_pieces[6], buff=SMALL_BUFF)
        rect.set_stroke(TEAL, 3)

        self.play(ShowCreation(rect))
        self.wait()
        self.play(rect.animate.surround(VGroup(exp_plane, exp_plane_label)))
        self.play(FadeOut(rect))

        # Go through imaginary partsg
        rows = VGroup(exp_pieces[n::5] for n in range(0, 5))
        for row in rows:
            row.save_state()
        self.play(rows.animate.fade(0.7))
        self.play(Restore(rows[2]))
        self.play(
            rows[2].animate.fade(0.7),
            Restore(rows[1]),
            Restore(rows[3]),
        )
        self.play(
            Restore(rows[0]),
            Restore(rows[4]),
            rows[1].animate.fade(0.7),
            rows[3].animate.fade(0.7),
        )
        self.wait()
        self.play(*(Restore(row) for row in rows))

        # Show columns
        cols = VGroup(exp_pieces[n:n + 5] for n in range(0, 25, 5))
        for col in cols:
            col.save_state()
        last = VectorizedPoint()
        self.play(cols.animate.fade(0.7))
        for col in cols:
            self.play(
                last.animate.fade(0.7),
                Restore(col),
            )
            last = col

        self.play(*(Restore(col) for col in cols))
        self.wait()


class BreakingDownCosine(ShowFamilyOfComplexSolutions):
    tex_to_color_map = {"+i": YELLOW, "-i": YELLOW}

    def construct(self):
        # Set up various planes
        left_planes, left_plane_labels = self.get_left_planes(label_texs=[R"e^{+it}", R"e^{-it}"])
        rot_vects, tails, t_tracker = self.get_rot_vects(left_planes)

        right_plane = self.get_right_plane(x_range=(-2, 2))
        right_plane.next_to(left_planes, RIGHT, buff=2.0)
        right_plane.add_coordinate_labels(font_size=16)
        vect_sum = self.get_rot_vect_sum(right_plane, t_tracker)
        for vect in vect_sum:
            vect.coef_tracker.set_value(0.5)

        output_dot = Group(TrueDot(), GlowDot()).set_color(YELLOW)
        output_dot.f_always.move_to(vect_sum[1].get_end)

        self.add(t_tracker)
        self.add(left_planes, left_plane_labels)
        self.add(rot_vects, tails, t_tracker)

        self.add(right_plane)
        self.add(vect_sum)
        self.add(output_dot)
        self.wait(12)

        # Show each part
        for vect in vect_sum:
            vect.coef_tracker.set_value(1)

        inner_arrows = VGroup(Arrow(2 * v, v, buff=0) for v in compass_directions(8))
        inner_arrows.set_fill(YELLOW)
        inner_arrows.move_to(right_plane)

        self.remove(vect_sum, output_dot)
        for i in [0, 1]:
            # self.play(ReplacementTransform(rot_vects[i].copy().clear_updaters(), vect_sum[i]))
            self.play(TransformFromCopy(rot_vects[i], vect_sum[i], suspend_mobject_updating=True))
            self.wait(3)

        self.wait(15)
        self.play(
            *(vs.coef_tracker.animate.set_value(0.5) for vs in vect_sum),
            LaggedStartMap(GrowArrow, inner_arrows, lag_ratio=1e-2, run_time=1)
        )
        self.play(FadeOut(inner_arrows))
        self.wait(12)
