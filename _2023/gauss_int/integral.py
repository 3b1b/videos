from manim_imports_ext import *
from _2023.convolutions2.continuous import *
from _2023.clt.main import *


class LastTime(VideoWrapper):
    wait_time = 8
    title = "Normal Distribution"


class AltBuildUpGaussian(BuildUpGaussian):
    pass


class BellCurveArea(InteractiveScene):
    def construct(self):
        # Setup
        axes = NumberPlane(
            (-4, 4), (0, 1.5, 0.5),
            width=14, height=5,
            background_line_style=dict(
                stroke_color=GREY_C,
                stroke_width=2,
                stroke_opacity=0.5
            )
        )
        axes.x_axis.add_numbers(font_size=24)
        axes.y_axis.add_numbers(num_decimal_places=1, excluding=[0])
        axes.to_edge(DOWN)
        graph = axes.get_graph(lambda x: np.exp(-x**2))
        graph.set_stroke(BLUE, 3)

        t2c = {"x": BLUE}
        graph_label = Tex("e^{-x^2}", font_size=72, t2c=t2c)
        graph_label.next_to(graph.pfp(0.6), UR)

        self.add(axes)
        self.play(ShowCreation(graph))
        self.play(Write(graph_label))
        self.wait()

        # Show integral
        integral = Tex(R"\int_{-\infty}^\infty e^{-x^2} dx", t2c=t2c)
        integral.to_edge(UP)

        self.play(graph.animate.set_fill(BLUE, 0.5))
        self.wait()
        self.play(
            Write(integral[R"\int_{-\infty}^\infty"]),
            FadeTransform(graph_label.copy(), integral["e^{-x^2}"])
        )
        self.play(TransformFromCopy(integral["x"][0], integral["dx"]))
        self.wait()

        # Show rectangles
        colors = (BLUE_E, BLUE_D, TEAL_D, TEAL_E)
        rects = axes.get_riemann_rectangles(graph, dx=0.2, colors=colors)
        rects.set_stroke(WHITE, 1)
        rects.set_fill(opacity=0.75)
        rect = rects[len(rects) // 2 - 2].copy()
        rect.set_opacity(1)
        graph_label.set_backstroke(width=5)

        brace = Brace(rect, UP, SMALL_BUFF)
        brace.set_backstroke(width=3)
        dx_label = brace.get_tex("dx", buff=SMALL_BUFF)
        dx_label["x"].set_color(BLUE)

        axes.generate_target()
        axes.target.y_axis.numbers.set_opacity(0)

        self.play(
            FadeIn(rects, lag_ratio=0.1, run_time=3),
            graph.animate.set_fill(opacity=0).set_anim_args(time_span=(1, 2)),
            graph_label.animate.shift(SMALL_BUFF * UR).set_anim_args(time_span=(1, 2)),
        )
        self.wait()
        self.play(
            rects.animate.set_opacity(0.1),
            MoveToTarget(axes),
            FadeIn(rect),
        )
        self.wait()
        self.play(graph_label.animate.set_height(0.5).next_to(rect, LEFT, SMALL_BUFF))
        self.play(FlashAround(integral["e^{-x^2}"], time_width=1, run_time=1.5))
        self.wait()
        self.play(
            GrowFromCenter(brace),
            FadeIn(dx_label, 0.5 * UP),
        )
        self.play(FlashAround(integral["dx"], time_width=1, run_time=1.5))
        self.wait()

        # Show addition
        rects.set_fill(opacity=0.8)
        rects.set_stroke(WHITE, 1)
        self.play(
            graph_label.animate.set_height(0.7).next_to(graph.pfp(0.4), UL),
            rects.animate.set_opacity(0.75),
            FadeOut(rect)
        )
        self.wait()
        self.play(
            LaggedStart(*(
                r.animate.shift(0.25 * UP).set_color(YELLOW).set_anim_args(rate_func=there_and_back)
                for r in rects
            ), run_time=5, lag_ratio=0.1),
            LaggedStart(
                FlashAround(integral[2:4], time_width=1),
                FlashAround(integral[1], time_width=1),
                lag_ratio=0.25,
                run_time=5,
            )
        )
        self.wait()

        # Thinner rectangles
        for dx in [0.1, 0.075, 0.05, 0.03, 0.02, 0.01, 0.005]:
            new_rects = axes.get_riemann_rectangles(graph, dx=dx, colors=colors)
            new_rects.set_stroke(WHITE, 1)
            new_rects.set_fill(opacity=0.7)
            self.play(
                Transform(rects, new_rects),
                brace.animate.set_width(dx * axes.x_axis.get_unit_size(), about_edge=LEFT),
                MaintainPositionRelativeTo(dx_label, brace),
            )
        self.add(graph)
        self.play(
            FadeOut(brace), FadeOut(dx_label),
            ShowCreation(graph),
        )

        # Indefinite integral
        frame = self.frame
        equals = Tex("=")
        equals.move_to(integral)
        equals.shift(0.5 * UP)
        answer_box = SurroundingRectangle(integral["e^{-x^2} dx"])
        answer_box.next_to(equals, RIGHT)
        answer_box.set_stroke(TEAL, 2)
        answer_box.set_fill(GREY_E, 1)
        q_marks = Tex("???")
        q_marks.set_height(0.6 * answer_box.get_height())
        q_marks.move_to(answer_box)
        answer_box.add(q_marks)

        self.play(
            frame.animate.set_height(9, about_edge=DOWN),
            integral.animate.next_to(equals, LEFT),
            FadeIn(equals),
            Write(answer_box),
        )

        integral.save_state()
        integral.generate_target()
        integral.target[1:4].stretch(0, 0, about_edge=RIGHT).set_opacity(0)
        integral.target[0].move_to(integral[:4], RIGHT)
        self.play(MoveToTarget(integral))

        # Arrows
        int_box = SurroundingRectangle(integral["e^{-x^2} dx"])
        int_box.set_stroke(BLUE, 2)
        arc = -0.5 * PI
        low_arrow = Arrow(answer_box.get_bottom(), int_box.get_bottom(), path_arc=arc)
        top_arrow = Arrow(int_box.get_top(), answer_box.get_top(), path_arc=arc)

        low_words = Text("Derivative", font_size=30)
        low_words.next_to(low_arrow, DOWN, MED_SMALL_BUFF)
        top_words = Text("Antiderivative", font_size=30)
        top_words.next_to(top_arrow, UP, MED_SMALL_BUFF)

        self.play(
            ShowCreation(low_arrow),
            FadeIn(low_words, 0.5 * LEFT),
            FadeTransform(answer_box.copy(), int_box, path_arc=arc)
        )
        self.wait()
        self.play(
            ShowCreation(top_arrow),
            FadeIn(top_words, 0.5 * RIGHT),
        )
        self.wait()

        # Impossible
        impossible = Text("Impossible!", font_size=72, color=RED)
        impossible.next_to(answer_box, RIGHT)

        functions = VGroup(
            Tex(R"a_n x^n + \cdots a_1 x + a_0", t2c=t2c),
            Tex(R"\sin(x), \cos(x), \tan(x)", t2c=t2c),
            Tex(R"b^x", t2c=t2c),
            Tex(R"\vdots")
        )
        functions.arrange(DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
        functions.set_height(2.5)
        functions.next_to(impossible, RIGHT, buff=LARGE_BUFF)

        self.play(FadeIn(impossible, scale=0.5, rate_func=rush_into))
        self.wait()
        self.play(
            LaggedStartMap(FadeIn, functions, shift=DOWN, lag_ratio=0.5),
            frame.animate.shift(4 * RIGHT),
            run_time=3
        )
        self.wait()


class AntiDerivative(InteractiveScene):
    def construct(self):
        # Add both planes
        x_min, x_max = (-3, 3)
        planes = VGroup(*(
            NumberPlane(
                (x_min, x_max), (0, 2),
                width=5.5, height=2.75,
                background_line_style=dict(stroke_color=GREY, stroke_width=1, stroke_opacity=1.0),
                faded_line_ratio=3,
            )
            for x in range(2)
        ))
        planes.arrange(DOWN, buff=LARGE_BUFF)
        planes.to_corner(UL)
        self.add(planes)

        # Titles
        titles = VGroup(
            Tex("f(x) = e^{-x^2}", font_size=66),
            Tex(R"F(x) = \int_0^x e^{-t^2} dt"),
        )
        for title, plane in zip(titles, planes):
            title.next_to(plane, RIGHT)

        ad_word = Text("Antiderivative")
        ad_word.next_to(titles[1], UP, MED_LARGE_BUFF)
        VGroup(ad_word, titles[1]).match_y(planes[1])

        self.add(titles)
        self.add(ad_word)

        # High graph
        x_tracker = ValueTracker(0)
        get_x = x_tracker.get_value
        high_graph = planes[0].get_graph(lambda x: np.exp(-x**2))
        high_graph.set_stroke(BLUE, 3)

        high_area = high_graph.copy()

        def update_area(area: VMobject):
            x = get_x()
            area.become(high_graph)
            area.set_stroke(width=0)
            area.set_fill(BLUE, 0.5)
            area.pointwise_become_partial(
                high_graph, 0, inverse_interpolate(x_min, x_max, x)
            )
            area.add_line_to(planes[0].c2p(x, 0))
            area.add_line_to(planes[0].c2p(x_min, 0))
            return area

        high_area.add_updater(update_area)

        self.add(high_graph, high_area)

        # Low graph
        dist = scipy.stats.norm(0, 1)
        low_graph = planes[1].get_graph(lambda x: math.sqrt(PI) * dist.cdf(x))
        low_graph.set_stroke(YELLOW, 2)
        low_dot = GlowDot()
        low_dot.add_updater(lambda m: m.move_to(planes[1].i2gp(get_x(), low_graph)))

        low_line = always_redraw(lambda: DashedLine(
            planes[1].c2p(get_x(), 0), planes[1].i2gp(get_x(), low_graph),
        ).set_stroke(WHITE, 2))

        self.add(low_graph, low_dot, low_line)

        # Animations
        for value in [1.5, -2, -1, 1, -0.5, 0.5, 3.0, -1.5]:
            self.play(x_tracker.animate.set_value(value), run_time=3)
            self.wait()


class UsualFunctionTypes(InteractiveScene):
    def construct(self):
        t2c = {"x": YELLOW}
        functions = VGroup(
            Tex(R"a_n x^n + \cdots a_1 x + a_0", t2c=t2c),
            Tex(R"\sin(x), \cos(x), \arctan(x)", t2c=t2c),
            Tex(R"b^x, \log(x), \cosh(x)", t2c=t2c),
            Tex(R"\vdots")
        )
        functions.arrange(DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
        functions.set_height(2.5)
        functions.to_edge(RIGHT, buff=0.6)

        box = SurroundingRectangle(functions, buff=MED_LARGE_BUFF)
        box.set_stroke(RED, 2)

        words = Text("Cannot be expressed \n in terms of these:")
        words.next_to(box, UP)
        words.set_fill(RED)

        self.play(
            FadeIn(words, lag_ratio=0.1),
            FadeIn(box),
            LaggedStartMap(FadeIn, functions, shift=DOWN, lag_ratio=0.5, run_time=3),
        )
        self.wait()


class GaussianIntegral(ThreeDScene, InteractiveScene):
    def func(self, x, y):
        return np.exp(-x**2 - y**2)

    def get_axes(
        self,
        x_range=(-3, 3),
        y_range=(-3, 3),
        z_range=(0, 1.5, 0.5),
        width=8,
        height=8,
        depth=3,
        center=0.5 * IN,
        include_plane=False
    ):
        axes = ThreeDAxes(
            x_range, y_range, z_range,
            width=width, height=height, depth=depth
        )
        axes.set_stroke(GREY_C)
        if include_plane:
            plane = NumberPlane(
                x_range, y_range,
                width=width, height=height,
                background_line_style=dict(
                    stroke_color=GREY_C,
                    stroke_width=1,
                ),
            )
            plane.faded_lines.set_stroke(opacity=0.5)
            plane.shift(0.01 * IN)
            axes.plane = plane
            axes.add(plane)

        x, y, z = axis_labels = VGroup(*map(Tex, "xyz"))
        axis_labels.use_winding_fill(False)
        x.next_to(axes.x_axis, RIGHT)
        y.next_to(axes.y_axis, UP)
        z.rotate(90 * DEGREES, RIGHT)
        z.next_to(axes.z_axis, OUT)
        axes.labels = axis_labels
        axes.add(axis_labels)

        axes.shift(center - axes.c2p(0, 0, 0))
        axes.set_flat_stroke(False)
        return axes

    def get_gaussian_graph(
        self,
        axes,
        color=interpolate_color(BLUE_E, BLACK, 0.6),
        opacity=1.0,
        shading=(0.2, 0.2, 0.4),
    ):
        graph = axes.get_graph(self.func)
        graph.set_color(color)
        graph.set_opacity(opacity)
        graph.set_shading(*shading)
        return graph

    def get_dynamic_cylinder(self, axes, r_init=1):
        cylinder = self.get_cylinder(axes, r_init)
        r_tracker = ValueTracker(r_init)
        cylinder.add_updater(lambda m: self.set_cylinder_r(
            m, axes, r_tracker.get_value()
        ))
        return cylinder, r_tracker

    def get_cylinder(
        self, axes, r,
        color=BLUE_E,
        opacity=1
    ):
        cylinder = Cylinder(color=color, opacity=opacity)
        self.set_cylinder_r(cylinder, axes, r)
        return cylinder

    def set_cylinder_r(self, cylinder, axes, r):
        r = max(r, 1e-5)
        cylinder.set_width(2 * r * axes.x_axis.get_unit_size())
        cylinder.set_depth(
            self.func(r, 0) * axes.z_axis.get_unit_size(),
            stretch=True
        )
        cylinder.move_to(axes.c2p(0, 0, 0), IN)
        return cylinder

    def get_thick_cylinder(self, cylinder, delta_r):
        radius = 0.5 * cylinder.get_width()
        outer_cylinder = cylinder.copy()
        factor = (radius + delta_r) / radius
        outer_cylinder.stretch(factor, 0)
        outer_cylinder.stretch(factor, 1)

        annulus = ParametricSurface(
            lambda u, v: (radius + u * delta_r) * np.array([math.cos(v), math.sin(v), 0]),
            u_range=(0, 1),
            v_range=(0, TAU),
        )
        annulus.match_color(cylinder)
        annulus.move_to(cylinder, OUT)

        result = Group(cylinder.copy(), annulus, outer_cylinder)
        result.clear_updaters()
        return result

    def get_x_slice(self, axes, y, x_range=(-3, 3.1, 0.1)):
        xs = np.arange(*x_range)
        ys = np.ones(len(xs)) * y
        points = axes.c2p(xs, ys, self.func(xs, y))
        graph = VMobject().set_points_smoothly(points)
        graph.use_winding_fill(False)
        return graph

    def get_dynamic_slice(
        self,
        axes,
        stroke_color=BLUE,
        stroke_width=2,
        fill_color=BLUE_E,
        fill_opacity=0.5,
    ):
        y_tracker = ValueTracker(0)
        get_y = y_tracker.get_value

        z_unit = axes.z_axis.get_unit_size()
        x_slice = self.get_x_slice(axes, 0)
        x_slice.set_stroke(stroke_color, stroke_width)
        x_slice.set_fill(fill_color, fill_opacity)
        x_slice.add_updater(
            lambda m: m.set_depth(self.func(0, get_y()) * z_unit, stretch=True)
        )
        x_slice.add_updater(lambda m: m.move_to(axes.c2p(0, get_y(), 0), IN))

        return x_slice, y_tracker


class CylinderSlices(GaussianIntegral):
    def construct(self):
        # Setup
        frame = self.frame
        axes = self.get_axes()

        graph = self.get_gaussian_graph(axes)
        graph.set_opacity(0.8)
        graph.always_sort_to_camera(self.camera)

        graph_mesh = SurfaceMesh(graph, resolution=(21, 21))
        graph_mesh.set_stroke(WHITE, 0.5, opacity=0.25)
        graph_mesh.set_flat_stroke(False)

        self.add(axes)

        # Animate in by rotating e^{-x^2}
        bell_halves = Group(*(
            axes.get_parametric_surface(
                lambda r, theta: np.array(
                    [r * np.cos(theta), r * np.sin(theta), np.exp(-r**2)
                ]),
                u_range=(0, 3),
                v_range=v_range,
            )
            for v_range in [(0, PI), (PI, TAU)]
        ))
        for half in bell_halves:
            half.match_style(graph)
            half.set_opacity(0.5)

        bell2d = self.get_x_slice(axes, 0)
        bell2d.set_stroke(TEAL, 3)
        kw = dict(t2c={"x": BLUE, "y": YELLOW})
        label2d, label3d = func_labels = VGroup(
            Tex("f_1(x) = e^{-x^2}", **kw),
            Tex("f_2(x, y) = e^{-(x^2 + y^2)}", **kw),
        )
        for label in func_labels:
            label.fix_in_frame()
            label.move_to(4 * LEFT + 2 * UP)

        axes.save_state()
        frame.reorient(0, 90)
        frame.move_to(OUT + 2 * UP)
        axes.y_axis.set_opacity(0)
        axes.labels.set_opacity(0)
        self.play(
            ShowCreation(bell2d),
            Write(label2d)
        )
        self.wait()

        self.play(
            ShowCreation(bell_halves[0]),
            ShowCreation(bell_halves[1]),
            Rotate(bell2d, PI, axis=OUT, about_point=axes.c2p(0, 0, 0)),
            frame.animate.move_to(ORIGIN).reorient(-20, 70),
            Restore(axes),
            TransformMatchingTex(label2d.copy(), label3d, time_span=(0, 2)),
            label2d.animate.next_to(label3d, UP, MED_LARGE_BUFF, LEFT),
            run_time=6
        )
        self.wait()
        self.play(
            FadeOut(bell_halves, 0.01 * IN),
            FadeOut(bell2d, 0.1 * IN),
            FadeIn(graph, 0.01 * IN),
        )
        self.play(Write(graph_mesh, stroke_width=1, lag_ratio=0.01))
        self.wait()

        # Rotate the frame
        self.play(
            frame.animate.set_theta(20 * DEGREES),
            rate_func=there_and_back,
            run_time=30,
        )

        # Reposition to 2d view
        frame.save_state()
        graph_mesh.save_state()
        func_labels.use_winding_fill(False)
        self.play(
            frame.animate.reorient(0, 0).set_height(10).move_to(1.5 * LEFT).set_field_of_view(1 * DEGREES),
            graph.animate.set_opacity(0.25),
            func_labels.animate.scale(0.75).to_corner(UL),
            graph_mesh.animate.set_stroke(width=1),
            run_time=3,
        )

        # Explain meaning of r
        x, y = (1.5, 0.75)
        dot = Dot(axes.c2p(x, y), fill_color=RED)
        dot.set_stroke(WHITE, 0.5)
        coords = Tex("(x, y)", font_size=36)
        coords.next_to(dot, UR, SMALL_BUFF)

        x_line = Line(axes.get_origin(), axes.c2p(x, 0, 0))
        y_line = Line(axes.c2p(x, 0, 0), axes.c2p(x, y, 0))
        r_line = Line(axes.c2p(x, y, 0), axes.get_origin())
        x_line.set_stroke(BLUE, 3)
        y_line.set_stroke(YELLOW, 3)
        r_line.set_stroke(RED, 3)
        lines = VGroup(x_line, y_line, r_line)
        labels = VGroup(*map(Tex, "xyr"))
        for label, line in zip(labels, lines):
            label.match_color(line)
            label.scale(0.85)
            label.next_to(line.get_center(), rotate_vector(line.get_vector(), -90 * DEGREES), SMALL_BUFF)

        self.add(dot, coords, set_depth_test=False)
        self.play(
            FadeIn(dot, scale=0.5),
            FadeIn(coords),
        )
        for line, label in zip(lines, labels):
            self.add(line, label, dot, set_depth_test=False)
            self.play(
                ShowCreation(line),
                Write(label),
            )

        # Plug in r
        r_label_rect = SurroundingRectangle(labels[2], buff=SMALL_BUFF)
        r_label_rect.set_stroke(RED, 2)
        arrow = Arrow(r_label_rect, axes.c2p(-3, 3, 0) + 3.2 * LEFT + 0.25 * UP, path_arc=45 * DEGREES)
        arrow.set_stroke(RED)

        self.always_depth_test = False
        self.play(ShowCreation(r_label_rect))
        self.play(ShowCreation(arrow))
        self.wait()

        # Show Pythagorean equations
        r_func = Tex("= e^{-r^2}", t2c={"r": RED})
        r_func.match_height(label2d["= e^{-x^2}"])
        r_func.next_to(label3d, RIGHT, MED_SMALL_BUFF, UP)
        r_func.fix_in_frame()

        r_rect = SurroundingRectangle(r_func["r^2"], buff=0.025)
        xy_rect = SurroundingRectangle(label3d["x^2 + y^2"], buff=0.025)
        VGroup(r_rect, xy_rect).set_stroke(TEAL, 1)
        VGroup(r_rect, xy_rect).fix_in_frame()

        pythag = Tex("x^2 + y^2 = r^2", t2c={"x": BLUE, "y": YELLOW, "r": RED})
        pythag.next_to(label3d, DOWN, buff=2.0, aligned_edge=LEFT)
        pythag.fix_in_frame()

        self.play(
            FadeTransform(label2d["= e^{-x^2}"].copy(), r_func),
            FadeOut(arrow, scale=0.8, shift=DR + RIGHT),
            FadeOut(r_label_rect)
        )
        self.wait()
        line_copies = lines.copy()
        self.add(*line_copies, set_depth_test=False)
        self.play(
            *(
                VShowPassingFlash(line.insert_n_curves(20).set_stroke(width=8), time_width=1.5)
                for line in line_copies
            ),
            *map(ShowCreation, lines)
        )
        self.play(
            FadeTransform(r_func["r^2"][0].copy(), pythag["r^2"]),
            FadeTransform(label3d["x^2 + y^2"][0].copy(), pythag["x^2 + y^2"]),
            Write(pythag["="]),
        )
        self.wait()
        self.wait()
        self.play(ShowCreation(xy_rect))
        self.wait()
        self.play(Transform(xy_rect, r_rect))
        self.play(FadeOut(xy_rect))

        functions = VGroup(label2d, label3d, r_func)
        functions.fix_in_frame()

        # Emphasize rotational symmetry
        self.always_depth_test = True
        x_label, y_label, r_label = labels
        self.play(
            *map(FadeOut, [x_line, y_line, x_label, y_label, pythag])
        )

        def get_circle(point, z_shift=0.02):
            origin = axes.c2p(0, 0, 0)
            point[2] = origin[2]
            radius = get_norm(point - origin)
            circle = Circle(radius=radius, n_components=96)
            x = axes.x_axis.p2n(point)
            y = axes.y_axis.p2n(point)
            circle.move_to(axes.c2p(0, 0, self.func(x, y) + z_shift))
            circle.set_stroke(RED, 2)
            circle.rotate(np.arctan2(y, x))
            circle.set_flat_stroke(False)
            return circle

        r_label.add_updater(lambda m: m.next_to(r_line.get_center(), UL, SMALL_BUFF))
        dot.add_updater(lambda m: m.move_to(r_line.get_start()))
        coords.add_updater(lambda m: m.next_to(dot, UR, SMALL_BUFF))
        circle = get_circle(r_line.get_start())

        self.play(
            Rotate(r_line, TAU, about_point=axes.get_origin()),
            ShowCreation(circle),
            frame.animate.reorient(30, 60).move_to(ORIGIN).set_height(8).set_field_of_view(45 * DEGREES),
            Restore(graph_mesh),
            run_time=7,
        )
        self.wait()
        self.play(
            r_line.animate.scale(0.1, about_point=axes.get_origin()),
            UpdateFromFunc(circle, lambda c: c.replace(get_circle(r_line.get_start()))),
            rate_func=there_and_back,
            run_time=8,
        )
        self.wait()
        self.play(*map(FadeOut, [r_line, dot, r_label, coords, circle]))

        # Dynamic cylinder
        cylinder, r_tracker = self.get_dynamic_cylinder(axes)
        delta_r = 0.1
        cylinders = Group(*(
            self.get_cylinder(axes, r, opacity=0.5)
            for r in np.arange(0, 3, delta_r)
        ))

        r_tracker.set_value(0)
        self.add(cylinder, cylinders, graph, graph_mesh)
        self.play(
            graph.animate.set_opacity(0.1).set_anim_args(time_span=(0, 2)),
            FadeIn(cylinders, lag_ratio=0.9),
            r_tracker.animate.set_value(3).set_anim_args(
                rate_func=linear,
                time_span=(0.5, 10),
            ),
            frame.animate.reorient(-15, 75).set_height(5.5),
            run_time=10,
        )
        self.wait()

        # Isolate one particular cylinder
        self.play(
            r_tracker.animate.set_value(0.7),
            cylinders.animate.set_opacity(0.1),
            frame.animate.reorient(-27, 71),
            run_time=3,
        )

        # Unwrap cylinder
        axes.labels[2].set_opacity(0)
        R = cylinder.get_width() / 2
        rect = Square3D(resolution=cylinder.resolution)
        rect.set_width(TAU * R)
        rect.set_height(cylinder.get_depth(), stretch=True)
        rect.match_color(cylinder)
        rect_top = Line(rect.get_corner(UL), rect.get_corner(UR))
        rect_top.set_stroke(RED, 3)
        rect_side = Line(rect.get_corner(DL), rect.get_corner(UL))
        rect_side.set_stroke(PINK, 3)
        VGroup(rect_top, rect_side).set_flat_stroke(False)
        rect_group = Group(rect, rect_top, rect_side)
        rect_group.apply_matrix(frame.get_orientation().as_matrix())
        rect_group.next_to(cylinder, [1, 0, 1], LARGE_BUFF)

        eq_kw = dict(
            font_size=35,
            t2c={"{r}": RED},
        )
        area_eq1 = TexText("Area = (Circumference)(Height)", **eq_kw)
        area_eq2 = TexText(R"Area = $2 \pi {r} \cdot e^{-{r}^2}$", **eq_kw)
        for eq in area_eq1, area_eq2:
            eq.fix_in_frame()
            eq.to_corner(UL)
        area_eq1.shift(area_eq2[0].get_center() - area_eq1[0].get_center())

        self.add(functions)
        functions.fix_in_frame()
        functions.deactivate_depth_test()
        functions.use_winding_fill(False)
        self.play(
            FadeIn(area_eq1, DOWN),
            functions.animate.shift(1.5 * DOWN).scale(0.7, about_edge=DL).set_fill(opacity=0.75)
        )
        self.wait()

        pre_rect = cylinder.copy()
        pre_rect.clear_updaters()
        self.add(pre_rect, graph)
        self.play(
            pre_rect.animate.scale(0.95).next_to(cylinder, OUT, buff=1.0),
            frame.animate.set_height(7).move_to([1.0, 0.15, 1.0]),
            run_time=2,
        )
        self.play(ReplacementTransform(pre_rect, rect), run_time=2)
        self.wait()

        # Show cylinder area
        circle = get_circle(cylinder.get_points()[0], z_shift=0)
        height_line = Line(cylinder.get_corner(IN + DOWN), cylinder.get_corner(OUT + DOWN))
        height_line.set_stroke(PINK, 3)
        height_line.set_flat_stroke(False)

        circ_brace = Brace(area_eq2[R"2 \pi {r}"], DOWN, SMALL_BUFF)
        height_brace = Brace(area_eq2[R"e^{-{r}^2}"], DOWN, SMALL_BUFF)
        VGroup(circ_brace, height_brace).fix_in_frame()

        circ_word = area_eq1["(Circumference)"]
        height_word = area_eq1["(Height)"]

        self.add(circle, set_depth_test=False)
        self.play(
            ShowCreation(circle),
            ShowCreation(rect_top),
        )
        self.play(
            FadeIn(circ_brace),
            circ_word.animate.scale(0.75).next_to(circ_brace, DOWN, SMALL_BUFF),
            Write(area_eq2[R"2 \pi {r}"]),
            height_word.animate.next_to(area_eq2[R"2 \pi {r}"], RIGHT)
        )
        self.wait()
        self.add(height_line, set_depth_test=False)
        self.play(
            FadeOut(circle),
            FadeOut(rect_top),
            ShowCreation(rect_side),
            ShowCreation(height_line),
        )
        self.play(
            FadeIn(height_brace),
            height_word.animate.scale(0.75).next_to(height_brace, DOWN, SMALL_BUFF, aligned_edge=LEFT),
            circ_word.animate.align_to(circ_brace, RIGHT),
            FadeInFromPoint(area_eq2[R"\cdot e^{-{r}^2}"], r_func[1:].get_center()),
        )
        self.remove(area_eq1)
        self.add(area_eq2, circ_word, height_word)
        self.wait()
        self.play(
            frame.animate.center().reorient(-15, 66).set_height(4).set_anim_args(run_time=15),
            *map(FadeOut, [rect, rect_side, height_line]),
        )

        # Show thickness
        volume_word = Text("Volume", **eq_kw)
        volume_word.fix_in_frame()
        volume_word.move_to(area_eq2, DL)
        area_part = area_eq2[R"= $2 \pi {r} \cdot e^{-{r}^2}$"]
        annotations = VGroup(circ_brace, height_brace, circ_word, height_word)
        dr_tex = Tex("d{r}", **eq_kw)
        dr_tex.fix_in_frame()

        thick_cylinder = self.get_thick_cylinder(cylinder, delta_r * axes.x_axis.get_unit_size())
        thin_cylinder = self.get_thick_cylinder(cylinder, 0.1 * delta_r * axes.x_axis.get_unit_size())
        _, annulus, outer_cylinder = thick_cylinder

        dr_brace = Brace(
            Line(axes.get_origin(), axes.c2p(delta_r, 0, 0)), UP
        )
        dr_brace.stretch(0.5, 1)
        brace_label = dr_brace.get_tex("d{r}", buff=SMALL_BUFF)
        brace_label["r"].set_color(RED)
        brace_label.scale(0.35, about_edge=DOWN)
        dr_brace.add(brace_label)
        dr_brace.rotate(90 * DEGREES, RIGHT)
        dr_brace.move_to(thick_cylinder.get_corner(OUT + LEFT), IN + LEFT)

        self.remove(cylinder)
        self.add(thin_cylinder, cylinders, graph, graph_mesh)
        self.play(Transform(thin_cylinder, thick_cylinder))
        self.add(dr_brace, set_depth_test=False)
        self.play(Write(dr_brace))
        self.wait()

        self.play(
            LaggedStartMap(FadeOut, annotations, shift=DOWN, run_time=1),
            FadeOut(area_eq2["Area"], DOWN),
            FadeIn(volume_word, DOWN),
            area_part.animate.next_to(volume_word, RIGHT, SMALL_BUFF, DOWN),
        )
        dr_tex.next_to(area_part, RIGHT, SMALL_BUFF, DOWN)
        self.play(FadeIn(dr_tex))
        self.wait()

        # Show all cylinders
        integrand = VGroup(*area_part[0][1:], *dr_tex)
        integrand.fix_in_frame()
        integral = Tex(R"\int_0^\infty", **eq_kw)
        integral.fix_in_frame()
        integral.move_to(volume_word, LEFT)

        thick_cylinders = Group(*(
            self.get_thick_cylinder(cyl, delta_r * axes.x_axis.get_unit_size())
            for cyl in cylinders
        ))
        thick_cylinders.set_opacity(0.8)
        thick_cylinders.set_shading(0.25, 0.25, 0.25)
        small_dr = 0.02
        thin_cylinders = Group(*(
            self.get_thick_cylinder(self.get_cylinder(axes, r), small_dr)
            for r in np.arange(0, 5, small_dr)
        ))
        thin_cylinders.set_opacity(0.5)

        self.play(
            FadeOut(volume_word, LEFT),
            FadeOut(area_part[0][0], LEFT),
            FadeIn(integral, LEFT),
            integrand.animate.next_to(integral, RIGHT, buff=0),
        )

        self.add(thick_cylinders, cylinders, graph, graph_mesh)
        self.play(ShowIncreasingSubsets(thick_cylinders, run_time=8))
        self.play(FadeOut(thick_cylinders, 0.1 * IN))
        self.wait()

        self.add(dr_brace[:-1], dr_brace[-1], set_depth_test=False)
        self.play(
            Transform(thin_cylinder, thin_cylinders[int(np.round(r_tracker.get_value() / small_dr))]),
            dr_brace[:-1].animate.stretch(small_dr / delta_r, 0, about_edge=RIGHT),
            UpdateFromFunc(dr_brace[-1], lambda m: m.next_to(dr_brace[:-1], OUT, SMALL_BUFF)),
            run_time=3,
        )
        self.add(thin_cylinders, cylinders, graph, graph_mesh)
        self.add(dr_brace)
        dr_brace.deactivate_depth_test()
        self.play(
            ShowIncreasingSubsets(thin_cylinders),
            frame.animate.reorient(20, 70).set_height(8).move_to(OUT),
            FadeOut(dr_brace, time_span=(0, 2)),
            FadeOut(thin_cylinder, 2 * IN, time_span=(0, 2)),
            FadeOut(functions, time_span=(0, 2)),
            FadeOut(integral, time_span=(0, 2)),
            FadeOut(integrand, time_span=(0, 2)),
            run_time=20,
        )
        self.wait()

        # Ambient rotation
        t0 = self.time
        frame.add_updater(lambda m: m.reorient(20 * math.cos(0.1 * (self.time - t0))))
        self.wait(30)


class CylinderIntegral(InteractiveScene):
    def construct(self):
        # Set up equations
        kw = dict(
            font_size=48,
            t2c={
                "{r}": RED,
                "{0}": RED,
                R"{\infty}": RED,
            }
        )
        exprs = VGroup(
            Tex(R"\int_0^\infty 2\pi {r} \cdot e^{-{r}^2} \, d{r}", **kw),
            Tex(R"\pi \int_0^\infty 2 {r} \cdot e^{-{r}^2}\,  d{r}", **kw),
            Tex(R"= \pi \left[ -e^{-{\infty}^2} - \left(-e^{-{0}^2} \right) \right]", **kw),
            Tex(R"= \pi", **kw),
        )
        exprs[1:].arrange(RIGHT, buff=SMALL_BUFF)
        exprs[1:].to_corner(UL)
        exprs[0].move_to(exprs[1], LEFT)

        # Factor out
        self.add(exprs[0])
        self.wait()
        self.play(TransformMatchingTex(*exprs[:2], run_time=1, path_arc=30 * DEGREES))
        self.wait()

        # Show antiderivative
        integrand = exprs[1][R"2 {r} \cdot e^{-{r}^2}"]
        integrand_rect = SurroundingRectangle(integrand, buff=SMALL_BUFF)
        integrand_rect.set_stroke(YELLOW, 2)
        anti_derivative = Tex(R"-e^{-{r}^2}", **kw)
        anti_derivative.next_to(integrand_rect, DOWN, buff=1.5)
        arrow = Arrow(anti_derivative, integrand_rect)
        arrow.set_color(YELLOW)
        arrow_label = Tex(R"d \over d{r}", **kw)
        arrow_label.scale(0.75)
        arrow_label.next_to(arrow, RIGHT, MED_SMALL_BUFF)

        self.play(
            ShowCreation(integrand_rect),
            GrowArrow(arrow),
            FadeIn(arrow_label, UP),
        )
        self.wait()
        self.play(TransformMatchingShapes(integrand.copy(), anti_derivative))
        self.wait()

        # Evaluate
        self.play(
            Write(exprs[2]["="]),
            TransformFromCopy(exprs[1][R"\pi"], exprs[2][R"\pi"]),
            TransformFromCopy(exprs[1][R"\int"], exprs[2][R"["]),
            TransformFromCopy(exprs[1][R"\int"], exprs[2][R"]"]),
        )
        self.wait()
        self.play(LaggedStart(
            FadeTransform(anti_derivative.copy(), exprs[2][R"-e^{-{\infty}^2}"]),
            FadeIn(VGroup(*exprs[2][R"- \left("], exprs[2][R"\right)"])),
            FadeTransform(anti_derivative.copy(), exprs[2][R"-e^{-{0}^2}"]),
            lag_ratio=0.7
        ))

        self.play(
            TransformMatchingShapes(
                VGroup(
                    *exprs[1][R"\pi \int_0^\infty"],
                    *anti_derivative
                ).copy(),
                exprs[2]
            )
        )
        self.wait()
        self.play(TransformMatchingTex(exprs[2].copy(), exprs[3]))
        self.wait()

        # Simplify
        rects = VGroup(
            SurroundingRectangle(exprs[2][R"-e^{-{\infty}^2}"]),
            SurroundingRectangle(exprs[2][R"-e^{-{0}^2}"]),
        )
        rects.set_stroke(TEAL, 1)
        values = VGroup(*map(Integer, [0, -1]))
        for value, rect in zip(values, rects):
            value.next_to(rect, DOWN)
            value.match_color(rect)

        zero, one = values

        self.play(
            TransformFromCopy(integrand_rect, rects[0]),
            integrand_rect.animate.set_stroke(YELLOW, 1, 0.5)
        )
        self.play(FadeIn(zero, 0.5 * DOWN))
        self.wait()
        self.play(TransformFromCopy(*rects))
        self.play(FadeIn(one, 0.5 * DOWN))
        self.wait()
        self.play(Write(exprs[3]))
        self.wait()

        # Highlight answer
        answer = exprs[3][R"\pi"]
        self.play(
            LaggedStartMap(FadeOut, VGroup(
                integrand_rect, arrow, arrow_label, anti_derivative,
                *rects, *values
            )),
            answer.animate.scale(2, about_edge=LEFT)
        )
        self.play(FlashAround(answer, run_time=2, time_width=1.5, color=TEAL))
        self.wait()


class CartesianSlices(GaussianIntegral):
    def construct(self):
        # Setup
        frame = self.frame
        axes = self.get_axes()

        graph = self.get_gaussian_graph(axes)
        graph_mesh = SurfaceMesh(graph, resolution=(21, 21))
        graph_mesh.set_stroke(WHITE, 0.5, opacity=0.25)
        graph_mesh.set_flat_stroke(False)

        self.add(axes, graph, graph_mesh)

        # Dynamic slice
        x_slice, y_tracker = self.get_dynamic_slice(axes)
        y_unit = axes.y_axis.get_unit_size()
        graph.add_updater(lambda m: m.set_clip_plane(UP, -y_tracker.get_value() * y_unit))

        x_max = axes.x_range[1]
        y_tracker.set_value(x_max)
        self.add(x_slice)
        self.play(
            y_tracker.animate.set_value(-x_max),
            run_time=5,
            rate_func=linear,
        )
        self.wait()

        # Show many slices
        def get_x_slices(dx=0.25):
            original_y_value = y_tracker.get_value()
            x_slices = VGroup()
            x_min, x_max = axes.x_range[:2]
            for y in np.arange(x_max, x_min, -dx):
                y_tracker.set_value(y)
                x_slice.update()
                x_slices.add(x_slice.copy().clear_updaters())
            x_slices.use_winding_fill(False)
            x_slices.deactivate_depth_test()
            x_slices.set_stroke(BLUE, 2, 0.5)
            x_slices.set_flat_stroke(False)
            y_tracker.set_value(original_y_value)
            return x_slices

        x_slices = get_x_slices(dx=0.25)
        self.add(x_slice, x_slices, graph, graph_mesh)
        self.play(
            FadeOut(graph, time_span=(0, 1)),
            FadeOut(x_slice, time_span=(0, 1)),
            FadeIn(x_slices, 0.1 * OUT, lag_ratio=0.1),
            axes.labels[2].animate.set_opacity(0),
            frame.animate.reorient(-80),
            run_time=4
        )
        self.play(
            frame.animate.reorient(-100),
            run_time=3,
        )
        self.wait()
        y_tracker.set_value(-x_max)
        self.add(x_slice, x_slices, graph, graph_mesh)
        self.play(
            FadeOut(x_slices, 0.1 * IN, time_span=(0, 2.5)),
            FadeIn(graph, time_span=(0, 2.5)),
            VFadeIn(x_slice),
            frame.animate.reorient(-15).set_height(6),
            y_tracker.animate.set_value(0),
            run_time=5,
        )

        # Discuss area of each slice
        tex_kw = dict(
            font_size=42,
            t2c={"x": BLUE, "y": YELLOW}
        )
        get_y = y_tracker.get_value
        x_slice_label = Tex("0.00 e^{-x^2}", **tex_kw)
        coef = x_slice_label.make_number_changeable("0.00")
        coef.set_color(YELLOW)
        coef.add_updater(lambda m: m.set_value(math.exp(-get_y()**2)).rotate(90 * DEGREES, RIGHT))

        x_term = x_slice_label[1:]
        brace = Brace(coef, UP, MED_SMALL_BUFF)
        y_term = Tex("e^{-y^2}", **tex_kw)
        y_term.next_to(brace, UP, SMALL_BUFF)

        y0_label = Tex("y = 0", **tex_kw)
        y0_label.rotate(90 * DEGREES, RIGHT)
        y0_label.next_to(x_slice.pfp(0.35), OUT + LEFT)

        x_slice_label.add(brace, y_term)
        x_slice_label.rotate(90 * DEGREES, RIGHT)
        x_slice_label.add_updater(lambda m: m.next_to(x_slice.pfp(0.6), OUT + RIGHT))

        x_slice_label.save_state()
        y_term.next_to(x_term, LEFT, SMALL_BUFF, aligned_edge=DOWN)
        brace.scale(0, about_edge=IN)
        coef.scale(0, about_edge=IN)
        swap = Swap(x_term, y_term)
        swap.begin()
        swap.update(1)

        func_label = Tex(R"e^{-(x^2 + y^2)}", **tex_kw)
        func_label.rotate(90 * DEGREES, RIGHT)
        func_label.next_to(x_slice_label, OUT, MED_LARGE_BUFF)

        fx0 = Tex(R"e^{-(x^2 + 0^2)} = e^{-x^2}", **tex_kw)
        fx0.rotate(90 * DEGREES, RIGHT)
        fx0.next_to(func_label, IN, MED_LARGE_BUFF, aligned_edge=LEFT)
        fx0["0"].set_color(YELLOW)

        self.always_depth_test = False
        self.play(
            *(
                VShowPassingFlash(mob, time_width=1.5, run_time=3)
                for mob in [
                    x_slice.copy().set_stroke(YELLOW, 8).set_fill(opacity=0).shift(0.02 * OUT),
                    Line(*axes.x_axis.get_start_and_end()).set_stroke(YELLOW, 8).insert_n_curves(40),
                ]
            ),
            Write(y0_label)
        )
        self.wait()

        self.play(FadeIn(func_label))
        self.wait()
        self.play(TransformMatchingTex(func_label.copy(), fx0, lag_ratio=0.025))
        self.wait()
        self.play(FadeOut(fx0, RIGHT, rate_func=running_start))

        self.play(TransformMatchingShapes(func_label.copy(), x_slice_label))
        self.wait()
        self.play(Swap(x_term, y_term, path_arc=0.5 * PI))
        self.play(
            Restore(x_slice_label),
            FadeOut(func_label, OUT),
        )
        self.wait()

        # Note the area
        def get_area_label():
            area_label = TexText(R"Area = $0.00 \cdot C$", font_size=30)
            area_label["C"].set_color(RED)
            num = area_label.make_number_changeable("0.00")
            num.set_value(coef.get_value())
            area_label.rotate(90 * DEGREES, RIGHT)
            area_label.move_to(interpolate(x_slice.get_zenith(), x_slice.get_nadir(), 0.66))
            area_label.shift(0.1 * DOWN)
            return area_label

        self.play(FadeIn(get_area_label(), run_time=3, rate_func=there_and_back_with_pause))

        # Move the slice
        y0_slice_copy = x_slice.copy()
        y0_slice_copy.clear_updaters()
        y0_slice_copy.set_fill(opacity=0)
        self.play(FadeOut(y0_label))
        for value in [-0.5, -0.75, -1]:
            self.play(y_tracker.animate.set_value(value), run_time=3)
            slice_copy = y0_slice_copy.copy().set_opacity(0)
            area_label = get_area_label()
            self.play(FadeIn(area_label))
            self.play(slice_copy.animate.match_y(x_slice).set_stroke(YELLOW, 3, 1))
            self.wait(0.25)
            self.play(slice_copy.animate.match_depth(x_slice, stretch=True, about_edge=IN).set_opacity(0))
            self.wait()
            self.play(FadeOut(area_label))

        # Go back to finer slices
        x_slices = get_x_slices(dx=0.1)
        y_tracker.set_value(-1)

        self.add(x_slices, graph, graph_mesh)
        self.play(
            FadeIn(x_slices, 0.1 * OUT, lag_ratio=0.1, run_time=4),
            FadeOut(graph),
            FadeOut(x_slice, time_span=(3, 4)),
            FadeOut(x_slice_label, time_span=(3, 4)),
            frame.animate.reorient(-83, 72, 0).set_height(8).center().set_anim_args(run_time=5)
        )

        # Ambient rotation
        t0 = self.time
        theta0 = frame.get_theta()
        frame.clear_updaters()
        frame.add_updater(lambda m: m.set_theta(
            theta0 + -0.2 * math.sin(0.1 * (self.time - t0))
        ))
        self.wait(10)

        # Show slice width
        mid_index = len(x_slices) // 2 - 3
        line = Line(x_slices[mid_index].get_zenith(), x_slices[mid_index + 1].get_zenith())
        brace = Brace(Line().set_width(line.get_length()), UP)
        brace.stretch(0.5, 1)
        brace.add(brace.get_tex("dy", buff=SMALL_BUFF).scale(0.75, about_edge=DOWN))
        brace.rotate(90 * DEGREES, RIGHT)
        brace.rotate(90 * DEGREES, IN)
        brace.next_to(line, OUT, buff=0)
        brace.use_winding_fill(False)
        self.play(FadeIn(brace))
        self.wait(60)


class CartesianSliceOverlay(InteractiveScene):
    def construct(self):
        # Show integral
        kw = dict(
            t2c={"{x}": BLUE, "{y}": YELLOW, "C": RED},
            font_size=48,
        )
        integral1 = Tex(R"\int_{-\infty}^\infty C \cdot e^{-{y}^2} d{y}", **kw)
        integral2 = Tex(R"= C \int_{-\infty}^\infty e^{-{y}^2}d{y}", **kw)
        rhs = Tex(R"= C^2", **kw)
        top_eq = VGroup(integral1, integral2, rhs)
        top_eq.arrange(RIGHT, buff=MED_SMALL_BUFF)
        top_eq.to_corner(UL)
        for part in top_eq:
            part.shift((integral1[0].get_y() - part[0].get_y()) * UP)

        self.play(FadeIn(integral1))
        self.wait()

        # Spell out meanings of each part
        area_part = integral1[R"C \cdot e^{-{y}^2}"]
        volume_part = integral1[R"C \cdot e^{-{y}^2} d{y}"]
        area_rect = SurroundingRectangle(area_part, buff=0.05)
        volume_rect = SurroundingRectangle(volume_part, buff=0.05)
        rects = VGroup(area_rect, volume_rect)
        rects.set_stroke(TEAL, 1)
        rects.set_fill(TEAL, 0.25)

        area_word = Text("Area of a slice")
        volume_word = Text("Volume of a slice")
        words = VGroup(area_word, volume_word)
        arrows = VGroup()
        for word, rect in zip(words, rects):
            word.next_to(rect, DOWN, LARGE_BUFF, LEFT)
            arrows.add(Arrow(rect, word))

        self.add(area_rect, integral1)
        self.play(
            FadeIn(area_rect),
            GrowArrow(arrows[0]),
            FadeIn(area_word)
        )
        self.wait()
        self.play(
            Transform(*rects),
            Transform(*arrows),
            TransformMatchingStrings(area_word, volume_word, run_time=1),
        )
        self.wait()
        self.play(*map(FadeOut, [area_rect, arrows[0], volume_word]))
        self.wait()

        # Show meaning of C
        sub_int = Tex(R"C = \int_{-\infty}^\infty e^{-{x}^2} d{x}", **kw)
        sub_int.next_to(top_eq, DOWN, LARGE_BUFF, aligned_edge=LEFT)
        box = SurroundingRectangle(sub_int)
        box.set_stroke(RED, 2)

        arrow = Arrow(integral1["C"], box, stroke_color=RED)

        for mob in box, sub_int:
            mob.save_state()
            mob.replace(integral1["C"], stretch=True)
            mob.set_opacity(0)

        self.play(
            ShowCreation(arrow),
            Restore(box),
            Restore(sub_int),
        )
        self.wait()
        self.play(TransformMatchingTex(integral1.copy(), integral2, path_arc=30 * DEGREES))
        self.wait()

        # Expand
        box2 = SurroundingRectangle(integral2[2:], buff=SMALL_BUFF)
        box2.set_stroke(RED, 1)
        self.play(TransformFromCopy(box, box2))
        self.wait()
        self.play(
            FadeOut(box2),
            Write(rhs)
        )
        self.wait()

        # Emphasize C^2
        C2 = rhs["C^2"]
        everything = VGroup(integral1, integral2, sub_int)
        self.play(C2.animate.scale(2, about_point=C2.get_left() + 0.1 * LEFT))
        self.play(
            FlashAround(C2, time_width=1.5, run_time=3),
            everything.animate.set_opacity(0.7),
        )
        self.wait()