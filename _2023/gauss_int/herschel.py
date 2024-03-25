from manim_imports_ext import *
from _2023.clt.main import *


class TwoDGaussianAsADistribution(InteractiveScene):
    n_points = 2000
    n_dots_per_moment = 10

    def construct(self):
        # Setup
        frame = self.frame
        plane = self.get_plane()
        plane.set_flat_stroke(False)
        self.add(plane)

        dartboard = self.get_dartboard(plane)
        dartboard.save_state()
        dartboard.set_opacity(0)
        self.add(dartboard)

        self.add_random_points_anim(plane)
        self.wait(10)
        self.play(Restore(dartboard))
        self.wait(10)

        # Graph
        def func(u, v, sigma=1):
            return np.exp(-(u**2 + v**2) / sigma**2) / sigma

        graphs = []
        for sigma in [0.8, 1.0, 0.6]:
            graph = ParametricSurface(lambda u, v: [u, v, func(u, v, sigma)], u_range=(-2, 2), v_range=(-2, 2))
            graph.match_width(plane.axes)
            graph.set_color(BLUE_E, 0.5)
            graph.move_to(plane.axes, IN)
            graph.always_sort_to_camera(self.camera)

            mesh = VGroup(*plane.background_lines.copy(), plane.faded_lines.copy())
            mesh.insert_n_curves(50)
            mesh.start = mesh.copy().set_opacity(0)
            mesh.save_state()
            mesh.saved_state.set_opacity(0)
            unit_size = plane.x_axis.get_unit_size()
            for submob in mesh.family_members_with_points():
                submob.set_points([
                    p + unit_size * func(*plane.p2c(p), sigma) * OUT
                    for p in submob.get_points()
                ])
                submob.set_stroke(
                    WHITE,
                    width=0.5 * submob.get_stroke_width(),
                    opacity=0.5 * submob.get_stroke_opacity()
                )
            mesh.shift(0.01 * OUT)
            graphs.append(Group(graph, mesh))

        graph1, graph2, graph3 = graphs

        # Test
        self.add(graph1)
        graph1[0].set_opacity(0)
        self.play(
            frame.animate.reorient(20, 70),
            graph1[0].animate.set_opacity(0.5),
            TransformFromCopy(graph1[1].start, graph1[1]),
            run_time=3
        )
        self.play(
            frame.animate.reorient(-20),
            run_time=7,
        )
        graph1.save_state()
        self.play(Transform(graph1, graph2), run_time=2)
        self.play(Transform(graph1, graph3), run_time=2)
        self.play(Restore(graph1), run_time=2)
        self.wait(2)

        self.play(
            FadeOut(graph1[0]),
            Restore(graph1[1]),
            frame.animate.reorient(0, 0),
            run_time=3,
        )

        # Radial symmetry
        blob = Circle(radius=0.2)
        blob.set_stroke(TEAL, 2)
        blob.set_fill(TEAL, 0.5)
        blob.move_to(plane.c2p(0.5, 0.5))

        arrow = FillArrow(ORIGIN, DL)
        arrow.next_to(blob, UR, buff=0)
        arrow.set_fill(RED, 1)
        arrow.set_backstroke(width=2)

        radial_line = DashedLine(plane.c2p(0, 0), blob.get_center(), dash_length=0.025)
        radial_line.set_stroke(TEAL, 2)

        self.play(
            dartboard.animate.set_opacity(0.2),
            DrawBorderThenFill(blob),
            GrowArrow(arrow)
        )
        self.wait()
        self.play(ShowCreation(radial_line))
        self.play(
            Rotate(blob, TAU, about_point=plane.get_origin(), run_time=6),
            Rotate(radial_line, TAU, about_point=plane.get_origin(), run_time=6),
            MaintainPositionRelativeTo(arrow, blob),
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeOut, VGroup(blob, arrow, radial_line)),
            dartboard.animate.set_opacity(0.75)
        )
        self.play(
            self.frame.animate.set_gamma(-0.75 * PI),
            rate_func=there_and_back,
            run_time=6,
        )
        self.wait()

        # Ambient randomness
        self.wait(0.1 * self.n_points)

    def get_plane(self):
        plane = NumberPlane(
            (-2, 2), (-2, 2),
            background_line_style=dict(stroke_color=GREY, stroke_width=1, stroke_opacity=1)
        )
        plane.set_height(7)
        plane.to_edge(DOWN, buff=0.25)
        plane.add(Tex("x").next_to(plane.x_axis.get_right(), RIGHT, SMALL_BUFF))
        plane.add(Tex("y").next_to(plane.y_axis.get_top(), UP, SMALL_BUFF))
        return plane

    def get_dartboard(self, plane):
        dartboard = Dartboard()
        dartboard.match_height(plane.axes)
        dartboard.move_to(plane.axes)
        dartboard.set_opacity(0.75)
        return dartboard

    def add_random_points_anim(self, plane):
        coords = np.random.normal(0, 0.5, (self.n_points, 2))
        dots = Group(*(
            GlowDot(plane.c2p(x, y), glow_factor=4.0, radius=0.3)
            for x, y in coords
        ))

        anim = LaggedStart(*(
            FadeIn(dot, rate_func=there_and_back)
            for dot in dots
        ), lag_ratio=1 / self.n_dots_per_moment, run_time=self.n_points / self.n_dots_per_moment)

        anim_mob = turn_animation_into_updater(anim)
        self.add(anim_mob)


class FaintDartboard(TwoDGaussianAsADistribution):
    n_points = 1000
    n_dots_per_moment = 5

    def construct(self):
        frame = self.frame
        plane = self.get_plane()
        plane.set_flat_stroke(False)
        self.add(plane)

        dartboard = self.get_dartboard(plane)
        dartboard.set_opacity(0.15)
        self.add(dartboard)

        self.add_random_points_anim(plane)
        self.wait(0.1 * self.n_points)


class ShowXYCoordinate(TwoDGaussianAsADistribution):
    def construct(self):
        plane = self.get_plane()
        # self.add(plane)  # Remove

        # Test
        x, y = (-1.5, 0.5)
        dot = Dot(plane.c2p(x, y))
        dot.set_color(TEAL)

        r_line = Line(plane.get_origin(), dot.get_center())
        r_line.set_stroke(RED, 3)
        x_line = Line(plane.get_origin(), plane.c2p(x, 0))
        x_line.set_stroke(BLUE, 5)
        y_line = Line(plane.c2p(x, 0), plane.c2p(x, y))
        y_line.set_stroke(YELLOW, 5)

        x_label = Tex("x", color=BLUE).next_to(x_line, DOWN, SMALL_BUFF)
        y_label = Tex("y", color=YELLOW).next_to(y_line, LEFT, SMALL_BUFF)
        r_label = Tex("r", color=RED).next_to(r_line.get_center(), UR, SMALL_BUFF)

        self.play(
            FadeIn(dot),
            ShowCreation(x_line),
            FadeIn(x_label, 0.5 * LEFT),
        )
        self.add(y_line, dot)
        self.play(
            ShowCreation(y_line),
            FadeIn(y_label, 0.5 * UP),
        )
        self.wait()
        self.add(r_line, dot)
        self.play(
            ShowCreation(r_line),
            FadeIn(r_label, shift=0.5 * normalize(r_line.get_vector()))
        )
        self.wait()


class IndependentCoordinates(TwoDGaussianAsADistribution):
    n_iterations = 20
    random_seed = 1

    def construct(self):
        frame = self.frame
        plane = self.get_plane()
        dartboard = self.get_dartboard(plane)
        dartboard.set_opacity(0.15)
        self.add(plane, dartboard)

        x_tip = ArrowTip(angle=90 * DEGREES).scale(0.5).set_color(BLUE)
        y_tip = ArrowTip(angle=0).scale(0.5).set_color(YELLOW)
        x_tip.move_to(plane.get_origin(), UP)
        y_tip.move_to(plane.get_origin(), RIGHT)
        for tip in x_tip, y_tip:
            tip.set_opacity(0)
            tip.save_state()

        for n in range(self.n_iterations):
            # Test
            x_tip.restore()
            y_tip.restore()

            # xs = np.random.normal(0, 1, 10)
            # ys = np.random.normal(0, 1, 10)
            # x = xs[-2]
            # y = ys[-2]
            x = np.random.normal(0, 0.5)
            y = np.random.normal(0, 0.5)

            if y < 0:
                x_tip.flip(RIGHT, about_point=plane.get_origin())
            if x < 0:
                y_tip.flip(UP, about_point=plane.get_origin())

            lines = VGroup(
                DashedLine(plane.c2p(x, 0), plane.c2p(x, y), dash_length=0.05),
                DashedLine(plane.c2p(0, y), plane.c2p(x, y), dash_length=0.05),
            )
            lines.set_stroke(WHITE, 2)

            dot = GlowDot().move_to(plane.c2p(x, y))

            self.play(LaggedStart(
                x_tip.animate.match_x(dot).set_opacity(1),
                y_tip.animate.match_y(dot).set_opacity(1),
                lag_ratio=0.5,
            ))
            self.play(
                *map(ShowCreation, lines),
                FadeIn(dot),
                run_time=0.5
            )
            self.wait(0.5)
            # self.play(UpdateFromAlphaFunc(x_tip, lambda m, a: m.match_x(
            #     plane.c2p(xs[integer_interpolate(0, len(xs) - 1, a)[0]], 0)
            # )))
            # self.play(UpdateFromAlphaFunc(y_tip, lambda m, a: m.match_y(
            #     plane.c2p(0, ys[integer_interpolate(0, len(ys) - 1, a)[0]])
            # )))
            # self.add(lines, dot)
            # self.wait()
            self.play(LaggedStartMap(FadeOut, Group(
                x_tip, y_tip, lines, dot
            )), run_time=1)


class ShowPointR0(TwoDGaussianAsADistribution):
    def construct(self):
        plane = self.get_plane()
        plane.axes.set_stroke(width=1)
        dartboard = self.get_dartboard(plane)
        dartboard.set_opacity(0.15)
        self.add(plane, dartboard)

        # Show point
        x, y = (0.7, 0.5)
        r = get_norm([x, y])
        dot = Dot(plane.c2p(x, y), radius=0.05)
        r_line = Line(plane.get_origin(), dot.get_center())
        r_line.set_stroke(RED, 3)
        coord_label = Tex("(x, y)", t2c={"x": BLUE, "y": YELLOW})
        coord_label.next_to(dot, UR, buff=SMALL_BUFF)
        coord_label.set_backstroke()
        new_coord_label = Tex("(r, 0)", t2c={"r": RED, "0": YELLOW})
        new_coord_label.next_to(plane.c2p(r, 0), UR, SMALL_BUFF)

        angle = math.atan2(y, x)

        self.add(r_line, dot, coord_label)
        self.wait()
        self.play(
            Rotate(r_line, -angle, about_point=plane.get_origin()),
            Rotate(dot, -angle, about_point=plane.get_origin()),
            ReplacementTransform(coord_label, new_coord_label, path_arc=-angle, time_span=(1, 2)),
            run_time=3
        )
        self.wait()


class RescaleG(InteractiveScene):
    def construct(self):
        def g(x):
            return 0.5 * math.exp(-x**4 + x**2)

        # Setup
        axes = Axes((-2, 2), (0, 1, 0.25), width=6, height=3)
        axes.x_axis.add_numbers()
        axes.y_axis.add_numbers(num_decimal_places=2, excluding=[0], font_size=16)
        self.add(axes)

        curve = axes.get_graph(g)
        curve.make_smooth()
        curve.set_stroke(BLUE, 3)
        curve.save_state()
        curve.generate_target()
        curve.target.stretch(1 / g(0), 1, about_edge=DOWN)

        label1 = TexText(R"$g(0) \ne 1$", font_size=36)
        label2 = TexText(R"$g(0) = 1$", font_size=36)
        label3 = TexText(R"Later we \\ re-scale anyway", font_size=36)
        labels = [label1, label2, label3]
        curves = [curve, curve.target, curve]
        for label, crv in zip(labels, curves):
            label.next_to(crv.get_top(), UR)

        self.play(
            ShowCreation(curve),
            FadeIn(label1, lag_ratio=0.1),
        )
        self.wait()
        self.play(
            MoveToTarget(curve),
            FadeTransform(label1, label2),
        )
        self.wait(2)
        self.play(
            Restore(curve),
            FadeTransform(label2, label3),
        )
        self.play(curve.animate.set_fill(BLUE, 0.5))
        self.wait()


class ManyDifferentFs(InteractiveScene):
    def construct(self):
        axes = Axes((0, 4), (0, 1, 0.25), width=6, height=3)
        axes.x_axis.add_numbers()
        axes.y_axis.add_numbers(num_decimal_places=2, excluding=[0], font_size=16)
        self.add(axes)

        # Many curves
        curves = VGroup(
            axes.get_graph(lambda x: math.exp(-x**2)),
            axes.get_graph(lambda x: 1 / (1 + x**2)),
            axes.get_graph(lambda x: math.exp(-x)),
            axes.get_graph(lambda x: 0.5 * math.exp(-x**4 + x**2)),
            axes.get_graph(lambda x: math.cos(x)**2 / (x + 1)),
            axes.get_graph(lambda x: (1 / 2) * (x**2) * np.exp(-x)),
        )
        curves.set_stroke(RED, 3)
        func_names = VGroup(
            Tex("e^{-x^2}"),
            Tex(R"1 \over (1 + x^2)"),
            Tex("e^{-x}"),
            Tex(R"\frac{1}{2} e^{-x^4 + x^2}"),
            Tex(R"\cos^2(x) \over (x + 1)"),
            Tex(R"\frac{1}{2} x^2 e^{-x}"),
        )
        func_names.move_to(axes.get_top())

        curve = curves[0]
        name = func_names[0]
        self.play(
            ShowCreation(curve),
            FadeIn(name, 0.5 * UP)
        )
        self.wait()
        for new_curve, new_name in zip(curves[1:], func_names[1:]):
            self.play(
                Transform(curve, new_curve),
                TransformMatchingTex(name, new_name, run_time=1)
            )
            name = new_name
            self.wait()


class VariableInputs(InteractiveScene):
    def construct(self):
        equation = Tex(R"f(\sqrt{(1.00)^2 + (0.00)^2}) = f(1.00)f(0.00)")
        xs = equation.make_number_changeable("1.00", replace_all=True)
        ys = equation.make_number_changeable("0.00", replace_all=True)

        xs.set_color(BLUE)
        ys.set_color(YELLOW)

        x_tracker = ValueTracker(1.0)
        y_tracker = ValueTracker(1.0)

        for mob in xs:
            mob.add_updater(lambda m: m.set_value(x_tracker.get_value()))

        for mob in ys:
            mob.add_updater(lambda m: m.set_value(y_tracker.get_value()))

        self.add(equation)
        for n in range(30):
            self.play(x_tracker.animate.set_value(random.random() * 10))
            self.play(y_tracker.animate.set_value(random.random() * 10))
            self.wait(0.5)


class RationalNumbers(InteractiveScene):
    def construct(self):
        # Interval
        interval = UnitInterval((0, 1, 1))
        interval.center()
        interval.add_numbers(font_size=36, num_decimal_places=0)
        self.add(interval)

        # Add rational points
        pairs = []
        max_n = 100
        for n in range(2, max_n):
            for k in range(1, n):
                if math.gcd(n, k) == 1:
                    pairs.append((k, n))

        lines = VGroup()
        line_groups = VGroup(*(VGroup() for n in range(max_n - 2)))
        labels = VGroup()
        frac_template = Tex(R"1 \over 2")
        frac_template.make_number_changeable("1")
        frac_template.make_number_changeable("2")

        for pair in pairs:
            k, n = pair
            line = Line(DOWN, UP)
            line.set_height(2.0 / n)
            line.set_stroke(TEAL, width=4.0 / math.sqrt(n))
            line.move_to(interval.n2p(k / n))
            lines.add(line)
            line_groups[n - 2].add(line)

            if n < 15:
                frac = frac_template.copy()
                frac[0].set_value(k)
                frac[2].set_value(n)
                frac[1].match_width(frac[2])
                frac.set_height(1.5 / n)
                frac.next_to(line, UP, SMALL_BUFF)
                labels.add(frac)

        line_groups.set_submobject_colors_by_gradient(BLUE, TEAL)

        for i, j in [(0, 9), (9, 27), (27, len(labels))]:
            self.play(
                LaggedStartMap(FadeIn, lines[i:j], lag_ratio=0.75),
                LaggedStartMap(FadeIn, labels[i:j], lag_ratio=0.75),
                rate_func=linear,
                run_time=3,
            )
        self.play(
            LaggedStartMap(FadeIn, lines[27:], lag_ratio=0.75),
            run_time=10,
            rate_func=rush_into,
        )
        self.wait()

        # Transition to real
        real_line = Line(interval.n2p(0), interval.n2p(1))
        real_line.insert_n_curves(50)
        real_line.set_stroke([TEAL, BLUE, TEAL], width=[0, 3, 3, 0])
        self.play(
            LaggedStart(*(Rotate(line, -90 * DEGREES) for line in lines), lag_ratio=1 / len(lines), run_time=3),
            FadeOut(labels, lag_ratio=0.1, run_time=1),
            ShowCreation(real_line, time_span=(2, 3)),
        )
        self.wait()


class TwoProperties(InteractiveScene):
    def construct(self):
        # Name properties
        properties = VGroup(
            VGroup(
                Text("Property 1"),
                TexText(R"""
                    The probability (density) depends \\
                    only on the distance from the origin
                """, alignment="", font_size=36, color=GREY_A),
            ),
            VGroup(
                Text("Property 2"),
                TexText(R"""
                    The $x$ and $y$ coordinates are \\
                    independent from each other.
                """, alignment="", font_size=36, color=GREY_A),
            ),
        )
        for prop in properties:
            prop.arrange(DOWN, aligned_edge=LEFT)
        properties.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        properties.to_corner(UL)

        prop_boxes = VGroup(*(
            SurroundingRectangle(prop[1]).set_fill(GREY_E, 1).set_stroke(RED, 1, 0.5)
            for prop in properties
        ))

        for prop, box in zip(properties, prop_boxes):
            self.play(FadeIn(prop[0], lag_ratio=0.1), FadeIn(box))

        # Formula
        implies = Tex(R"\Downarrow", font_size=72)
        implies.next_to(properties, DOWN, MED_LARGE_BUFF)

        kw = dict(
            t2c={"x": BLUE, "y": YELLOW, R"\sigma": RED, "{r}": RED}
        )
        form1, form2, form3 = forms = VGroup(
            Tex(R"f_2(x, y) = e^{-(x^2 + y^2)}", **kw),
            Tex(R"f_2(x, y) = e^{-(x^2 + y^2) / 2 \sigma^2}", **kw),
            Tex(R"f_2(x, y) = {1 \over 4 \sigma^2 \pi} e^{-(x^2 + y^2) / 2 \sigma^2}", **kw),
        )
        forms.next_to(implies, DOWN, MED_SMALL_BUFF)

        form1.save_state()
        self.play(
            Write(implies),
            FadeIn(form1, DOWN)
        )
        self.wait()
        self.play(TransformMatchingTex(form1, form2, run_time=1, lag_ratio=0.05))
        self.wait(2)
        self.play(TransformMatchingTex(form2, form3, run_time=1, lag_ratio=0.05))
        self.wait(2)
        form1.restore()
        self.play(TransformMatchingTex(form3, form1, run_time=2, lag_ratio=0.05))
        self.wait(2)

        # Property 1
        lhs = form1["f_2(x, y)"][0]

        self.add(properties[0][1], prop_boxes[0])
        self.play(
            prop_boxes[0].animate.stretch(0, 0, about_edge=RIGHT).set_opacity(0),
            FadeOut(implies),
            FadeOut(form1["= e^{-(x^2 + y^2)}"]),
        )
        self.remove(prop_boxes[0])
        self.add(lhs)
        self.wait()
        phrase = properties[0][1]["only on the distance"]
        self.play(
            FlashUnder(phrase, color=TEAL, buff=0),
            phrase.animate.set_color(TEAL),
        )
        self.wait(2)

        # Function of radius
        lhs.generate_target()
        lhs.target.to_edge(LEFT)
        radial_rhs = Tex(R"= f({r})", **kw)
        full_radial_rhs = Tex(R"= f(\sqrt{x^2 + y^2})", **kw)
        radial_rhs.next_to(lhs.target, RIGHT, SMALL_BUFF)
        full_radial_rhs.next_to(radial_rhs, RIGHT, MED_SMALL_BUFF)
        full_radial_rhs.shift((radial_rhs["="].get_y() - full_radial_rhs["="].get_y()) * UP)

        lhs_rect = SurroundingRectangle(lhs)
        f_rect = SurroundingRectangle(radial_rhs["f"], buff=0.05)
        f_rect.set_stroke(BLUE, 2)
        f_words = Text("Some single-variable function", font_size=36)
        f_words.next_to(f_rect, UP, SMALL_BUFF, aligned_edge=LEFT)
        f_words.match_color(f_rect)

        self.play(ShowCreation(lhs_rect))
        self.wait()
        self.play(lhs_rect.animate.replace(lhs[1], stretch=True).set_stroke(width=1).scale(1.1))
        self.play(FadeOut(lhs_rect))
        self.wait()
        self.play(
            MoveToTarget(lhs),
            Write(radial_rhs),
        )
        self.play(
            ShowCreation(f_rect),
            FadeIn(f_words, lag_ratio=0.1)
        )
        self.wait(2)
        self.play(FadeOut(f_words), FadeOut(f_rect))
        self.play(TransformMatchingTex(radial_rhs.copy(), full_radial_rhs))
        self.wait(2)

        # Property 2
        self.add(properties[1][1], prop_boxes[1])
        self.play(
            prop_boxes[1].animate.stretch(0, 0, about_edge=RIGHT).set_opacity(0),
        )
        self.remove(prop_boxes[0])
        self.wait()

        phrase = properties[1][1]["independent"]
        self.play(
            FlashUnder(phrase, color=TEAL, buff=0),
            phrase.animate.set_color(TEAL)
        )
        self.wait()

        # Factored expression
        lhs.generate_target()
        lhs.target.next_to(properties, DOWN, buff=0.7, aligned_edge=LEFT)
        radial_rhss = VGroup(radial_rhs, full_radial_rhs)

        factored_rhs = Tex(R"= g(x) h(y)", **kw)
        simpler_rhs = Tex(R"= g(x) g(y)", **kw)
        for rhs in factored_rhs, simpler_rhs:
            rhs.next_to(lhs.target, RIGHT)

        g_box = SurroundingRectangle(factored_rhs["g(x)"], buff=0.05).set_stroke(BLUE, 2)
        h_box = SurroundingRectangle(factored_rhs["h(y)"], buff=0.05).set_stroke(YELLOW, 2)
        g_words = TexText("Distribution of $x$", font_size=30).next_to(g_box, DOWN, 0.2)
        h_words = TexText("Distribution of $y$", font_size=30).next_to(h_box, DOWN, 0.2)

        self.play(
            MoveToTarget(lhs),
            radial_rhss.animate.to_edge(LEFT).shift(0.5 * DOWN).set_opacity(0.35),
        )
        self.play(
            TransformMatchingShapes(lhs.copy(), factored_rhs)
        )
        self.wait()
        self.play(
            ShowCreation(g_box),
            FadeIn(g_words)
        )
        self.wait()
        self.play(
            ReplacementTransform(g_box, h_box),
            ReplacementTransform(g_words, h_words),
        )
        self.wait()
        self.play(FadeOut(h_box), FadeOut(h_words))
        self.wait()
        self.play(
            FadeOut(factored_rhs["h(y)"], 0.5 * UP),
            FadeIn(simpler_rhs["g(y)"], 0.5 * UP),
        )
        self.wait()
        self.remove(factored_rhs)
        self.add(simpler_rhs)

        # Show proportionality
        arrow = Arrow(simpler_rhs, radial_rhs)
        self.play(
            GrowArrow(arrow),
            radial_rhs.animate.set_opacity(1),
            FadeOut(full_radial_rhs),
        )
        self.wait()
        radial_rhs.generate_target()
        radial_rhs.target.next_to(lhs, RIGHT),
        self.play(
            MoveToTarget(radial_rhs),
            simpler_rhs.animate.next_to(radial_rhs.target, RIGHT),
            Uncreate(arrow),
        )
        self.wait()

        xs = VGroup(lhs[3], simpler_rhs["x"][0][0])
        ys = VGroup(lhs[5], simpler_rhs["y"][0][0])
        rs = Tex("{r}", **kw).replicate(2)
        zeros = Tex("0", **kw).replicate(2)
        zeros.set_color(YELLOW)
        for x, r in zip(xs, rs):
            r.move_to(x, DOWN)
        for x, y, zero in zip(xs, ys, zeros):
            zero.move_to(y)
            zero.align_to(x, DOWN)

        const_rect = SurroundingRectangle(simpler_rhs["g(y)"], buff=0.05)
        const_rect.set_stroke(YELLOW, 1)
        const_words = Text("Some constant", font_size=36)
        const_words.match_color(const_rect)
        const_words.next_to(const_rect, DOWN)

        self.play(
            LaggedStartMap(FadeOut, VGroup(*xs, *ys), shift=0.5 * UP),
            LaggedStartMap(FadeIn, VGroup(*rs, *zeros), shift=0.5 * UP),
        )
        self.wait()
        self.play(
            ShowCreation(const_rect),
            FadeIn(const_words)
        )
        self.wait()

        # Assume this constant is 1
        assumption = TexText("Assume this is 1", font_size=36)
        assumption.move_to(const_words)
        assumption.match_color(const_words)

        f_eq_g = Tex("f = g", **kw)
        f_eq_g.next_to(radial_rhs, DOWN, LARGE_BUFF)

        f_rhs = Tex(R"= f(x)f(y)", **kw)
        f_rhs.move_to(simpler_rhs, LEFT)
        gs = simpler_rhs["g"]
        fs = f_rhs["f"]

        self.play(
            FadeIn(assumption, 0.5 * DOWN),
            FadeOut(const_words, 0.5 * DOWN),
        )
        self.wait()

        self.play(
            TransformFromCopy(
                VGroup(radial_rhs[1], *simpler_rhs[:2]),
                f_eq_g
            )
        )
        self.wait()

        self.play(
            LaggedStartMap(FadeIn, VGroup(*xs, *ys), shift=0.5 * DOWN),
            LaggedStartMap(FadeOut, VGroup(*rs, *zeros), shift=0.5 * DOWN),
            FadeOut(const_rect),
            FadeOut(assumption),
        )
        self.wait()
        self.play(
            TransformMatchingShapes(f_eq_g[0].copy(), fs),
            ReplacementTransform(simpler_rhs, f_rhs),
        )
        self.wait()

        # Highlight key equation
        key_equation = VGroup(*radial_rhs[1:], *f_rhs)

        self.play(
            key_equation.animate.set_x(0.25 * FRAME_WIDTH).to_edge(UP),
            FadeOut(f_eq_g),
            FadeOut(lhs),
            FadeOut(radial_rhs[0]),
        )
        self.play(FlashAround(key_equation, time_width=1, run_time=2))

        full_radial_rhs.set_opacity(1)
        full_radial_rhs.move_to(key_equation).shift(LEFT)

        self.play(
            GrowFromCenter(full_radial_rhs, lag_ratio=0.02),
            radial_rhs[1:].animate.next_to(full_radial_rhs, LEFT, aligned_edge=DOWN),
            key_equation[4:].animate.next_to(full_radial_rhs, RIGHT),
        )
        self.wait()

        # Name as a functional equation
        func_eq_name = Text("Functional\nequation")
        func_eq_name.to_corner(UL)
        func_eq_name.match_y(key_equation)
        arrow = Arrow(func_eq_name, radial_rhs[1].get_left(), buff=0.25)
        func_eq_name.to_edge(UP)

        self.play(LaggedStartMap(FadeOut, properties, shift=LEFT, lag_ratio=0.2))
        self.play(
            Write(func_eq_name),
            GrowArrow(arrow)
        )
        self.wait()

        # Example
        example_box = Rectangle(4, 3)
        example_box.set_stroke(TEAL, 2)
        example_box.set_fill(TEAL, 0.35)
        example_box.to_corner(DL, buff=0)
        example_word = Text("For example", font_size=30)
        example_word.next_to(example_box.get_top(), DOWN, SMALL_BUFF)
        example_f = Tex(R"f({r}) = e^{-{r}^2}", **kw)
        example_f.scale(0.75)
        example_f.next_to(example_word, DOWN, MED_LARGE_BUFF)

        self.play(
            FadeIn(example_box),
            FadeIn(example_word, 0.5 * DOWN)
        )
        self.play(
            TransformFromCopy(key_equation[:4], example_f[:4]),
            GrowFromPoint(example_f[4:], key_equation.get_left()),
        )
        self.wait()

        # Define h
        let = Text("Let")
        h_def = Tex(R"h({x}) = f(\sqrt{{x}})", **kw)
        h_def.next_to(key_equation, DOWN, LARGE_BUFF)
        let.next_to(h_def, LEFT, MED_LARGE_BUFF)
        h_def2 = Tex(R"h({x}^2) = f({x})", **kw)
        h_def2.next_to(h_def["h"], DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        h_eq = Tex(R"h(x^2 + y^2) = h(x^2) h(y^2)", **kw)
        h_eq.to_corner(UR)
        h_eq.to_edge(RIGHT, buff=1.25)

        example_h = Tex(R"h({r}) = e^{-{r}}", **kw)
        example_h.scale(0.75)
        example_h.next_to(example_f, DOWN, MED_LARGE_BUFF)

        self.play(FadeIn(h_def, DOWN), Write(let))
        self.wait()
        self.play(TransformMatchingTex(
            h_def.copy(), h_def2,
            key_map={R"\sqrt": "^2"},
            run_time=1
        ))
        self.wait()
        self.play(
            TransformFromCopy(h_def, example_h)
        )
        self.wait(2)

        self.play(
            VGroup(key_equation, full_radial_rhs).animate.scale(0.8).to_edge(LEFT),
            VGroup(let, h_def, h_def2).animate.scale(0.8).to_edge(LEFT),
            FadeOut(func_eq_name, LEFT),
            Uncreate(arrow),
        )
        self.play(
            TransformMatchingShapes(
                VGroup(*full_radial_rhs[1:], *f_rhs).copy(),
                h_eq
            )
        )
        self.wait()

        # Exponential property
        sum_box = SurroundingRectangle(h_eq["x^2 + y^2"])
        prod_box = SurroundingRectangle(h_eq["h(x^2) h(y^2)"])
        sum_words = Text("Adding inputs", font_size=30)
        sum_words.next_to(sum_box, DOWN)
        prod_words = Text("Multiplying outputs", font_size=30)
        prod_words.next_to(prod_box, DOWN)

        VGroup(sum_box, prod_box).set_stroke(TEAL, 2)
        VGroup(sum_words, prod_words).set_color(TEAL)

        self.play(
            ShowCreation(sum_box),
            FadeIn(sum_words, lag_ratio=0.1),
        )
        self.wait()
        self.play(
            ReplacementTransform(sum_box, prod_box),
            FadeTransform(sum_words, prod_words),
        )
        self.wait()
        self.play(FadeOut(prod_box), FadeOut(prod_words))
        self.wait()

        # Multi-input property
        implies = Tex(R"\Downarrow", font_size=72)
        implies.next_to(h_eq, DOWN)
        full_h_eq = Tex(R"h(x_1 + x_2 + \cdots + x_n) = h(x_1)h(x_2) \cdots h(x_n)")
        for s, color in zip(["1", "2", "n"], color_gradient([BLUE, YELLOW], 3)):
            full_h_eq[f"x_{s}"].set_color(color)
        full_h_eq.scale(0.75)
        full_h_eq.next_to(implies, DOWN)

        self.play(
            Write(implies),
            FadeIn(full_h_eq, DOWN),
        )
        self.wait()

        # Whole numbers
        implies2 = implies.copy()
        implies2.next_to(full_h_eq, DOWN, buff=MED_LARGE_BUFF)

        five_eq = Tex(R"h(5) &= h(1 + 1 + 1 + 1 + 1) \\ &= h(1)h(1)h(1)h(1)h(1) = h(1)^5")
        five_eq.next_to(implies2, DOWN)
        five_eq.to_edge(RIGHT)

        n_eq = Tex(R"h(n) = h(1 + \cdots + 1) = h(1) \cdots h(1) = h(1)^n")
        n_eq.scale(0.75)
        n_eq.next_to(implies2, DOWN, MED_LARGE_BUFF)
        sum_brace = Brace(n_eq[R"1 + \cdots + 1"], UP, SMALL_BUFF)
        sum_tex = sum_brace.get_tex(R"n \text{ times}", buff=SMALL_BUFF).scale(0.5, about_edge=DOWN)
        prod_brace = Brace(n_eq[R"h(1) \cdots h(1)"], UP, SMALL_BUFF)
        prod_tex = prod_brace.get_tex(R"n \text{ times}", buff=SMALL_BUFF).scale(0.5, about_edge=DOWN)

        for tex in n_eq, sum_tex, prod_tex:
            tex["n"].set_color(BLUE)

        self.play(Write(five_eq["h(5)"]))
        self.wait()
        self.play(
            TransformFromCopy(five_eq["h("][0], five_eq["h("][1]),
            TransformFromCopy(five_eq[")"][0], five_eq[")"][1]),
            Write(five_eq["="][0]),
        )
        self.play(ShowIncreasingSubsets(five_eq["1 + 1 + 1 + 1 + 1"][0]))
        self.wait()
        self.play(
            FadeTransform(
                five_eq["= h(1 + 1 + 1 + 1 + 1)"].copy(),
                five_eq["= h(1)h(1)h(1)h(1)h(1)"],
            )
        )
        self.wait()
        self.play(Write(five_eq["= h(1)^5"]))
        self.wait()

        self.play(FadeOut(five_eq), FadeIn(n_eq), FadeIn(implies2))
        self.play(LaggedStart(
            GrowFromCenter(sum_brace),
            FadeIn(sum_tex, 0.25 * DOWN),
            GrowFromCenter(prod_brace),
            FadeIn(prod_tex, 0.25 * DOWN),
        ))
        self.wait()

        # Exponential equation
        exp_eq1 = Tex(R"h(n) = h(1)^n")
        exp_eq2 = Tex(R"h(n) = b^n")
        for eq in exp_eq1, exp_eq2:
            eq["n"].set_color(BLUE)
        exp_eq1.next_to(n_eq, DOWN, MED_LARGE_BUFF)
        exp_eq2.move_to(exp_eq1, LEFT)
        h1_rect = SurroundingRectangle(exp_eq1["h(1)"], buff=0.05)
        h1_rect.set_stroke(YELLOW, 1)
        h1_words = Text("Some number", font_size=30)
        h1_words.match_color(h1_rect)
        h1_words.next_to(h1_rect, DOWN, SMALL_BUFF)

        self.play(
            TransformFromCopy(n_eq["h(n)"], exp_eq1["h(n)"]),
            TransformFromCopy(n_eq["= h(1)^n"], exp_eq1["= h(1)^n"]),
        )
        self.wait()
        self.play(ShowCreation(h1_rect), FadeIn(h1_words))
        self.wait()
        self.play(
            TransformMatchingTex(exp_eq1, exp_eq2),
            FadeOut(h1_rect, scale=0.5),
            FadeOut(h1_words, scale=0.5),
        )
        self.wait()

        # Show exercises
        self.play(
            exp_eq2.animate.next_to(implies2, DOWN),
            FadeOut(VGroup(n_eq, sum_brace, sum_tex, prod_brace, prod_tex), UP),
        )
        rational_eq = Tex(R"h(p / q) = b^{\,p / q}")
        rational_eq["p / q"].set_color(RED)
        rational_eq.move_to(exp_eq2)

        exercise = TexText(R"Exercise: Show this is also true for rational inputs, $p / q$")
        exercise["p / q"].set_color(RED)
        hint = TexText(R"Hint, think about $h\left(\frac{p}{q} + \cdots + \frac{p}{q} \right)$")

        exercise.next_to(rational_eq, DOWN, LARGE_BUFF)
        exercise.to_edge(RIGHT)
        hint.scale(0.7)
        hint.set_fill(GREY_A)
        hint.next_to(exercise, DOWN)

        self.play(
            Write(exercise),
            FadeOut(VGroup(example_box, example_word, example_f, example_h), shift=DL),
        )
        self.wait()
        pq_target = rational_eq["p / q"].copy()
        self.play(
            TransformMatchingTex(exp_eq2, rational_eq),
            TransformMatchingShapes(exercise["p / q"].copy(), pq_target),
        )
        self.remove(pq_target)
        self.wait()
        self.play(FadeIn(hint, DOWN))
        self.wait(2)
        self.play(LaggedStart(
            FadeOut(exercise, 0.5 * DOWN),
            FadeOut(hint, 0.5 * DOWN),
            lag_ratio=0.25,
        ))

        # Continuity
        assumption = TexText(R"Assuming $f$ (and hence also $h$) \\ is continuous...", font_size=36)
        assumption.next_to(rational_eq, LEFT, buff=2.0, aligned_edge=UP)
        assumption.shift(0.5 * DOWN)

        hx_eq = Tex("h(x) = b^x", **kw)
        hx_eq.move_to(rational_eq)
        range1 = TexText(R"For all $x \in \mathds{R}$", **kw)
        range2 = TexText(R"For all $x \in \mathds{R}^+$", **kw)
        for ran in range1, range2:
            ran.scale(0.75)
            ran.next_to(hx_eq, DOWN, MED_LARGE_BUFF)

        arrow = Arrow(assumption, hx_eq)

        self.play(FadeIn(assumption, lag_ratio=0.1))
        self.wait()
        self.play(
            GrowArrow(arrow),
            TransformMatchingTex(rational_eq, hx_eq)
        )
        self.play(FadeIn(range1, DOWN))
        self.wait()
        self.play(FadeTransform(range1, range2))
        self.wait()

        # Swap out for e
        hx_eq2 = Tex(R"h(x) = e^{{c} x}", **kw)
        hx_eq2.move_to(hx_eq)
        hx_eq2["c"].set_color(RED)

        b_rect = SurroundingRectangle(hx_eq["b"], buff=0.05)
        b_rect.set_stroke(PINK, 2)
        b_words = Text("Some constant", font_size=30)
        b_words.next_to(b_rect, DOWN, SMALL_BUFF, LEFT)
        b_words.match_color(b_rect)

        self.play(
            FadeOut(assumption, LEFT),
            Uncreate(arrow),
            FadeOut(range2, LEFT),
            ShowCreation(b_rect),
            Write(b_words, run_time=1)
        )
        self.wait()
        c = hx_eq2["{c}"][0][0]
        c_copy = c.copy()
        c.set_opacity(0)
        self.play(
            ReplacementTransform(b_rect, c_copy),
            TransformMatchingTex(hx_eq, hx_eq2, key_map={"b": "e"}),
            FadeOut(b_words, 0.2 * DOWN),
        )
        self.remove(c_copy)
        c.set_opacity(1)
        self.add(hx_eq2)
        self.wait()

        # Write final form for f
        implies3 = implies2.copy()
        implies3.rotate(-90 * DEGREES)
        implies3.next_to(hx_eq2, LEFT)
        f_form = Tex(R"f(x) = e^{cx^2}", **kw)
        f_form["c"].set_color(RED)
        f_form.next_to(implies3, LEFT)
        f_form.align_to(hx_eq2, DOWN)

        self.play(
            Write(implies3),
            TransformMatchingTex(hx_eq2.copy(), f_form, run_time=1)
        )
        self.wait()
        f_form.generate_target()
        f_form.target.scale(1.5, about_edge=RIGHT)
        rect = SurroundingRectangle(f_form.target)
        rect.set_stroke(YELLOW, 2)
        self.play(
            MoveToTarget(f_form),
            FlashAround(f_form.target, time_width=1, run_time=2, stroke_width=5),
            ShowCreation(rect, run_time=2),
        )
        self.wait()


class VariableC(InteractiveScene):
    c_values = [1.0, 0.5, -1.0, -0.7, -0.5, 0.25, -0.2, -0.4, -0.9, -0.1, 0.5, 0.3, 0.1]

    def construct(self):
        axes = self.get_axes()
        self.add(axes)

        curve = axes.get_graph(lambda x: self.func(x, 1))
        curve.set_stroke(RED, 3)
        self.add(curve)

        label = self.get_label(axes)
        self.add(label)

        c_tracker, c_interval, c_tip, c_label = self.get_c_group()
        get_c = c_tracker.get_value

        c_interval.move_to(axes, UR)
        c_interval.shift(0.5 * DOWN)
        self.add(c_interval, c_tip, c_label)

        axes.bind_graph_to_func(curve, lambda x: self.func(x, get_c()))

        # Animate
        for c in self.c_values:
            self.play(c_tracker.animate.set_value(c), run_time=2)
            self.wait()

    def get_c_group(self):
        c_tracker = ValueTracker(1)
        get_c = c_tracker.get_value

        c_interval = NumberLine(
            (-1, 1, 0.25), width=3, tick_size=0.05, big_tick_numbers=[-1, 0, 1],
        )
        c_interval.set_stroke(WHITE, 1)
        c_interval.add_numbers([-1, 0, 1], num_decimal_places=1, font_size=16)
        c_tip = ArrowTip(angle=-90 * DEGREES)
        c_tip.scale(0.5)
        c_tip.set_fill(RED)
        c_tip.add_updater(lambda m: m.move_to(c_interval.n2p(get_c()), DOWN))

        c_label = Tex("c = 1.00", t2c={"c": RED}, font_size=36)
        c_label.make_number_changeable("1.00")
        c_label[-1].scale(0.8, about_edge=LEFT)
        c_label.add_updater(lambda m: m[-1].set_value(get_c()))
        c_label.add_updater(lambda m: m.next_to(c_tip, UP, aligned_edge=LEFT))

        return [c_tracker, c_interval, c_tip, c_label]

    def get_axes(self):
        axes = Axes(
            (-1, 5), (0, 4),
            width=6, height=4,
        )
        return axes

    def func(self, x, c):
        return np.exp(c * x)

    def get_label(self, axes):
        label = Tex("e^{cx}", t2c={"c": RED})
        label.next_to(axes.c2p(0, 2.7), RIGHT)
        return label


class VariableCWithF(VariableC):
    def get_axes(self):
        axes = Axes(
            (-4, 4), (0, 2),
            width=8, height=3,
        )
        axes.add(VectorizedPoint(axes.c2p(0, 3)))
        axes.center()
        return axes

    def func(self, x, c):
        return np.exp(c * x * x)

    def get_label(self, axes):
        label = Tex("e^{cx^2}", t2c={"c": RED})
        label.next_to(axes.c2p(0, 2), LEFT)
        return label


class TalkAboutSignOfConstant3D(VariableCWithF):
    def construct(self):
        # Setup
        frame = self.frame
        frame.add_updater(lambda m: m.reorient(20 * math.cos(0.1 * self.time), 75))

        axes = ThreeDAxes((-4, 4), (-4, 4), (0, 1), depth=2)
        axes.set_width(10)
        axes.set_depth(2, stretch=True)
        axes.center()
        self.add(axes)

        label = Tex("f(r) = e^{cr^2}", t2c={"c": RED}, font_size=72)
        label.next_to(ORIGIN, LEFT)
        label.to_edge(UP)
        label.fix_in_frame()

        c_tracker, c_interval, c_tip, c_label = self.get_c_group()
        get_c = c_tracker.get_value
        c_interval.next_to(label, RIGHT, LARGE_BUFF)
        c_interval.shift(0.5 * DOWN)
        c_tracker.set_value(-1)

        c_group = VGroup(c_interval, c_tip, c_label)
        c_group.fix_in_frame()

        # Graph
        def get_graph(c):
            surface = axes.get_graph(lambda x, y: np.exp(c * (x**2 + y**2)))
            surface.always_sort_to_camera(self.camera)
            surface.set_color(BLUE_E, 0.5)
            mesh = SurfaceMesh(surface, (31, 31))
            mesh.set_stroke(WHITE, 0.5, 0.5)
            mesh.shift(0.001 * OUT)
            x_slice = ParametricCurve(
                lambda t: axes.c2p(t, 0, np.exp(c * t**2)),
                t_range=(-4, 4, 0.1)
            )
            x_slice.set_stroke(RED, 2)
            x_slice.set_flat_stroke(False)
            return Group(mesh, surface, x_slice)

        graph = get_graph(-1)

        self.add(graph)
        self.add(c_group)
        self.add(label)

        # Animations
        for value in [-0.5, -0.8, -0.3, -1.0, -0.3, 0.05, 0.1, -0.3, -1.0, -0.7, -1.0]:
            new_graph = get_graph(value)
            self.play(
                c_tracker.animate.set_value(value),
                Transform(graph, new_graph),
                run_time=5
            )
            self.wait()


class OldTalkAboutSignOfConstantScraps(InteractiveScene):
    def construct(self):
        plane.bind_graph_to_func(graph, lambda x: self.func(x, get_c()))
        # Area
        area = VMobject()
        area.set_fill(RED, 0.5)
        area.set_stroke(width=0)

        def update_area(area):
            area.set_points_as_corners([
                plane.c2p(-4, 0),
                *graph.get_anchors(),
                plane.c2p(4, 0)
            ])

        area.add_updater(update_area)

class OldTwoKeyProperties(InteractiveScene):
    def construct(self):
        # Setup equations
        kw = dict(
            t2c={"x": BLUE, "y": YELLOW, "{r}": RED, "{c}": PINK}
        )
        one_var = Tex("f_1(x) = e^{-x^2}", **kw)
        two_var = Tex("f_2(x, y) = e^{-(x^2 + y^2)}", **kw)
        factored = Tex("= f_1(x)f_1(y)", **kw)
        factored_exp = Tex("= e^{-x^2} e^{-y^2}", **kw)
        radial_exp = Tex(R"= e^{-{r}^2}", **kw)
        radial = Tex(R"= f_1({r})", **kw)
        radial_full = Tex(R"= f_1(\sqrt{x^2 + y^2})", **kw)

        expressions = VGroup(
            one_var,
            two_var,
            factored_exp,
            factored,
            radial_exp,
            radial,
            radial_full,
        )

        expressions.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        expressions.to_corner(UL)
        expressions.set_backstroke()

        rhs1 = VGroup(factored_exp, factored)
        rhs2 = VGroup(radial_exp, radial, radial_full)
        for rhs in [rhs1, rhs2]:
            rhs.arrange(RIGHT)
            rhs.next_to(two_var, RIGHT)

        for mob in expressions[2:]:
            mob.shift((two_var["="].get_y() - mob["="].get_y()) * UP)

        # From one to two
        self.add(one_var, two_var)
        self.wait()

        rects = VGroup(SurroundingRectangle(one_var), SurroundingRectangle(two_var))
        rects.set_stroke(TEAL, 2)
        rect_words = VGroup(Text("1 variable"), Text("2 variable"))
        for rect, words in zip(rects, rect_words):
            words.next_to(rect, RIGHT)

        arrow = Arrow(
            one_var.get_right(),
            two_var.get_corner(UR) + 0.5 * LEFT,
            path_arc=-PI,
        )
        words = Text("Two interpretations")
        words.next_to(arrow, RIGHT)

        self.play(
            FadeIn(rects[0]),
            FadeIn(rect_words[0]),
        )
        self.wait()
        self.play(
            ReplacementTransform(*rects),
            FadeTransform(*rect_words),
        )
        self.wait()

        two_var_copy = two_var.copy()
        self.play(
            FadeOut(rects[1]),
            FadeOut(rect_words[1]),
            ShowCreation(arrow),
            TransformMatchingTex(one_var.copy(), two_var_copy, run_time=1),
            Write(words, run_time=1)
        )
        self.remove(two_var_copy)
        self.wait()

        # Factored
        self.play(LaggedStart(
            Write(factored_exp["="]),
            TransformFromCopy(one_var["e^{-x^2}"], factored_exp["e^{-x^2}"]),
            TransformFromCopy(one_var["e^{-x^2}"], factored_exp["e^{-y^2}"]),
            run_time=2,
            lag_ratio=0.4,
        ))
        self.wait()
        self.play(LaggedStart(
            Write(factored["="]),
            TransformFromCopy(one_var["f_1(x)"], factored["f_1(x)"], path_arc=PI / 4),
            TransformFromCopy(one_var["f_1(x)"], factored["f_1(y)"], path_arc=PI / 4),
            run_time=3,
            lag_ratio=0.5,
        ))
        self.wait()

        # Rearrange
        rhs1.generate_target()
        rhs1.target.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        rhs1.target.next_to(two_var["="], DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
        rhs1.target.set_fill(opacity=0.5)
        self.play(MoveToTarget(rhs1, path_arc=-PI / 2))

        # Radial
        self.play(
            Write(radial_exp["="]),
            TransformFromCopy(one_var["e^{-x^2}"], radial_exp["e^{-{r}^2}"]),
        )
        self.wait()
        self.play(
            Write(radial["="]),
            TransformFromCopy(one_var["f_1(x)"], radial["f_1({r})"]),
        )
        self.wait()
        self.play(FadeIn(radial_full, lag_ratio=0.1))
        self.wait()

        # Consolidate
        prop = VGroup(factored, radial)
        prop.generate_target()
        prop.target.arrange(RIGHT)
        prop.target.center().move_to(UP)
        prop.target.set_opacity(1)
        prop.target.scale(1.5)
        prop.target[0]["="].set_opacity(0).scale(0, about_edge=RIGHT)
        prop.target.shift(0.5 * LEFT)

        f2_eq = VGroup(two_var, radial_exp, factored_exp)
        f2_eq.generate_target()
        f2_eq.target.set_fill(opacity=0.5)
        f2_eq.target[-1].next_to(f2_eq.target[-2], RIGHT)
        f2_eq.target.to_corner(UR)

        one_var.generate_target()
        one_var.target.scale(1.5, about_edge=UL)

        self.play(LaggedStart(
            FadeOut(arrow),
            FadeOut(words),
            MoveToTarget(prop),
            MoveToTarget(f2_eq),
            FadeOut(radial_full),
            lag_ratio=0.15,
            run_time=3,
        ))
        self.wait()
        self.play(
            FlashAround(one_var.target, time_width=1.0, run_time=2),
            MoveToTarget(one_var),
        )
        self.wait()
        self.play(FlashUnder(factored[1:]))
        self.wait()
        self.play(FlashUnder(radial))
        self.wait()

        radial_full.set_opacity(1)
        radial_full.scale(1.5)
        radial_full.move_to(radial, LEFT)
        radial_full.shift(0.1 * UP)
        left_shift = 2.0 * LEFT
        radial_full.shift(left_shift)

        self.play(
            FadeIn(radial_full),
            radial.animate.next_to(radial_full, RIGHT),
            factored.animate.shift(left_shift)
        )
        self.wait()

        # Flip the question
        prop = VGroup(factored, radial_full, radial)
        prop.generate_target()
        prop.target.scale(1 / 1.5).to_edge(UP).shift(1.5 * RIGHT)

        question = Text("What are all the functions \n with this property?")
        question.next_to(prop.target, DOWN, buff=1.5)
        question.to_edge(LEFT)
        arrow = Arrow(question, prop.target[0], buff=0.5)

        self.play(
            one_var.animate.scale(1 / 1.5).to_corner(DL).set_opacity(0.5),
            FadeOut(f2_eq, UP),
            MoveToTarget(prop),
        )
        self.play(
            FadeIn(question, lag_ratio=0.1),
            GrowArrow(arrow),
        )
        self.wait()

        # Substitute h
        h_eq = Tex(R"h(x^2) h(y^2) = h(x^2 + y^2)", **kw)
        h_eq.next_to(prop, DOWN, buff=MED_LARGE_BUFF)
        h_eq.shift((prop[1]["="].get_x() - h_eq["="].get_x()) * RIGHT)
        h_def = Tex(R"h(x) = f_1(\sqrt{x})", **kw)
        h_def.to_edge(LEFT).match_y(h_eq)
        h_def2 = Tex(R"h(x^2) = f_1(x)", **kw)
        h_def2.next_to(h_def, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        example_box = Rectangle(4, 4)
        example_box.set_stroke(TEAL, 2)
        example_box.set_fill(TEAL, 0.2)
        example_box.to_corner(DL, buff=0)
        example_word = Text("For example", font_size=30)
        example_word.next_to(example_box.get_top(), DOWN, SMALL_BUFF)
        one_var.generate_target()
        one_var.target.set_opacity(1)
        one_var.target.next_to(example_word, DOWN, MED_LARGE_BUFF)
        one_var.target.to_edge(LEFT)
        h_example = Tex(R"h(x) = e^{-x}", **kw)
        h_example.next_to(one_var.target, DOWN, MED_LARGE_BUFF)
        h_example.to_edge(LEFT)

        self.play(
            FadeOut(question, DOWN),
            FadeOut(arrow, DOWN),
            Write(h_def)
        )
        self.wait()
        self.play(TransformMatchingTex(h_def.copy(), h_def2))
        self.wait()

        self.add(example_box, one_var)
        self.play(
            FadeIn(example_box),
            FadeIn(example_word),
            MoveToTarget(one_var),
        )
        self.wait()
        self.play(FadeIn(h_example, DOWN))
        self.wait()

        self.play(FlashAround(prop[:2], time_width=1, run_time=2))
        self.play(
            TransformFromCopy(factored, h_eq["h(x^2) h(y^2)"][0]),
            TransformFromCopy(radial_full, h_eq["= h(x^2 + y^2)"][0]),
        )
        self.wait()

        # Exponential property
        h_box = SurroundingRectangle(h_eq)
        exp_words = Text("Exponential property!")
        exp_words.next_to(h_box, DOWN)

        sum_box = SurroundingRectangle(h_eq["x^2 + y^2"])
        prod_box = SurroundingRectangle(h_eq["h(x^2) h(y^2)"])
        sum_words = Text("Adding inputs", font_size=30)
        sum_words.next_to(sum_box, DOWN)
        prod_words = Text("Multiplying outputs", font_size=30)
        prod_words.next_to(prod_box, DOWN)

        VGroup(h_box, sum_box, prod_box).set_stroke(TEAL, 2)
        VGroup(exp_words, sum_words, prod_words).set_color(TEAL)

        self.play(
            ShowCreation(sum_box),
            FadeIn(sum_words, lag_ratio=0.1),
        )
        self.wait()
        self.play(
            ReplacementTransform(sum_box, prod_box),
            FadeTransform(sum_words, prod_words),
        )
        self.wait()
        self.play(
            ReplacementTransform(prod_box, h_box),
            FadeTransform(prod_words, exp_words),
        )
        self.wait()

        # Exponent
        implies = Tex(R"\Downarrow", font_size=72)
        implies.next_to(h_eq, DOWN)
        implies.rotate(PI)
        h_exp = Tex(R"h(x) = a \cdot b^x", **kw)
        h_exp.next_to(implies, DOWN)
        h_exp2 = Tex(R"h(x) = a \cdot e^{{c}x}", **kw)
        h_exp2.move_to(h_exp)
        h_exp0 = Tex(R"h(x) = b^x", **kw)
        h_exp0.move_to(h_exp)
        assumption = TexText("Assuming $h$ is continuous", font_size=24)
        assumption.next_to(implies, RIGHT)

        b_rect = SurroundingRectangle(h_exp["b"], buff=SMALL_BUFF)
        b_rect.set_stroke(PINK, 2)
        b_words = Text("Some constant", font_size=30)
        b_words.next_to(b_rect, DOWN, SMALL_BUFF, LEFT)
        b_words.match_color(b_rect)

        self.play(
            FadeTransform(h_box, implies),
            FadeTransform(exp_words, h_exp0),
        )
        self.wait()
        self.play(TransformMatchingTex(h_exp0, h_exp, run_time=1))
        self.wait()
        self.play(
            Rotate(implies, PI),
            FadeIn(assumption, lag_ratio=0.1)
        )
        self.wait()

        self.play(
            ShowCreation(b_rect),
            Write(b_words, run_time=1)
        )
        self.wait()
        c = h_exp2["{c}"][0][0]
        c_copy = c.copy()
        c.set_opacity(0)
        self.play(
            ReplacementTransform(b_rect, c_copy),
            TransformMatchingTex(h_exp, h_exp2),
            FadeOut(b_words, 0.2 * DOWN),
        )
        self.remove(c_copy)
        c.set_opacity(1)
        self.add(h_exp2)
        self.wait()

        # Final form
        implies2 = implies.copy()
        implies2.next_to(h_exp2, DOWN, MED_LARGE_BUFF)
        f_eq = Tex(R"f_1(x) = a \cdot e^{{c}x^2}", **kw)
        f_eq.next_to(implies2, DOWN)
        rect = SurroundingRectangle(f_eq)
        rect.set_stroke(YELLOW, 1)

        self.play(
            TransformMatchingTex(h_exp2.copy(), f_eq),
            Write(implies2, run_time=1)
        )
        self.wait()
        self.play(
            ShowCreation(rect, run_time=2),
            FlashAround(f_eq, time_width=1, run_time=2, stroke_width=5),
        )
        self.wait()
