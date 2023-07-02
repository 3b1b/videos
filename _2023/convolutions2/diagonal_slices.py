from manim_imports_ext import *
from _2023.convolutions2.continuous import *


class Introduce3DGraph(InteractiveScene):
    plane_config = dict(
        x_range=(-2, 2),
        y_range=(-2, 2),
        width=6.0,
        height=6.0,
    )
    plane_width = 6.0
    z_axis_height = 2.0
    plane_line_style = dict(
        stroke_color=GREY_C,
        stroke_width=1,
        stroke_opacity=1,
    )
    graph_resolution = (101, 101)

    def construct(self):
        # Initial axes and graphs
        f_axes, g_axes = all_axes = VGroup(*(
            Axes((-2, 2), (0, 1, 0.5), width=5, height=2)
            for n in range(2)
        ))
        all_axes.arrange(DOWN, buff=1.5)
        all_axes.to_edge(LEFT)
        self.frame.move_to(all_axes)

        for char, axes in zip("xy", all_axes):
            axis_label = Tex(char, font_size=24)
            axis_label.next_to(axes.x_axis.get_right(), UP)
            axes.add(axis_label)

        f_graph = f_axes.get_graph(self.f, use_smoothing=False)
        f_graph.set_stroke(BLUE, 3)
        g_graph = g_axes.get_graph(self.g)
        g_graph.set_stroke(YELLOW, 3)

        f_label, g_label = func_labels = VGroup(
            Tex("f(x)", font_size=36),
            Tex("g(y)", font_size=36)
        )
        for label, axes in zip(func_labels, all_axes):
            label.move_to(axes, UL)

        self.add(f_axes, f_graph, f_label)
        self.add(g_axes, g_graph, g_label)

        # Hook up trackers
        x_tracker = ValueTracker()
        y_tracker = ValueTracker()

        get_x = x_tracker.get_value
        get_y = y_tracker.get_value

        x_indicator, y_indicator = indicators = ArrowTip(90 * DEGREES).replicate(2)
        indicators.scale(0.5)
        indicators.set_fill(GREY_B)
        x_indicator.add_updater(lambda m: m.move_to(f_axes.c2p(get_x(), 0), UP))
        y_indicator.add_updater(lambda m: m.move_to(g_axes.c2p(get_y(), 0), UP))

        x_label, y_label = DecimalNumber(font_size=24).replicate(2)
        x_label.add_updater(lambda m: m.set_value(get_x()).next_to(x_indicator, DOWN, SMALL_BUFF).fix_in_frame())
        y_label.add_updater(lambda m: m.set_value(get_y()).next_to(y_indicator, DOWN, SMALL_BUFF).fix_in_frame())

        Axes.get_v_line_to_graph
        x_line = Line().set_stroke(WHITE, 1)
        y_line = Line().set_stroke(WHITE, 1)
        x_line.add_updater(lambda m: m.put_start_and_end_on(
            f_axes.c2p(get_x(), 0), f_axes.i2gp(get_x(), f_graph)
        ))
        y_line.add_updater(lambda m: m.put_start_and_end_on(
            g_axes.c2p(get_y(), 0), g_axes.i2gp(get_y(), g_graph)
        ))

        x_dot = GlowDot(color=BLUE)
        y_dot = GlowDot(color=YELLOW)
        x_dot.add_updater(lambda m: m.move_to(f_axes.i2gp(get_x(), f_graph)))
        y_dot.add_updater(lambda m: m.move_to(g_axes.i2gp(get_y(), g_graph)))

        # Ask about analog
        question = Text("What is analgous to this?")
        question.move_to(FRAME_WIDTH * RIGHT / 4)
        question.to_edge(UP)
        arrow = Vector(DOWN).next_to(question, DOWN)

        self.play(
            Write(question),
            GrowArrow(arrow),
            self.frame.animate.center().set_anim_args(run_time=2)
        )
        self.wait()

        # Scan over inputs
        x_tracker.set_value(-2)
        y_tracker.set_value(-2)
        self.add(x_indicator, x_label, x_line, x_dot)
        self.add(y_indicator, y_label, y_line, y_dot)
        self.play(LaggedStart(
            x_tracker.animate.set_value(0.31),
            y_tracker.animate.set_value(0.41),
            run_time=5,
            lag_ratio=0.2,
        ))
        self.wait()

        # Show the xy-plane
        plane = self.get_plane()
        plane.to_edge(RIGHT)

        x_indicator2 = x_indicator.copy().clear_updaters()
        y_indicator2 = y_indicator.copy().clear_updaters()
        y_indicator2.rotate(-90 * DEGREES)
        VGroup(x_indicator2, y_indicator2).scale(0.8)

        x_indicator2.add_updater(lambda m: m.move_to(plane.c2p(get_x()), UP))
        y_indicator2.add_updater(lambda m: m.move_to(plane.c2p(0, get_y()), RIGHT))

        self.play(
            FadeOut(question, UP),
            Uncreate(arrow),
            TransformFromCopy(f_axes.x_axis, plane.x_axis),
            TransformFromCopy(g_axes.x_axis, plane.y_axis),
            TransformFromCopy(x_indicator, x_indicator2),
            TransformFromCopy(y_indicator, y_indicator2),
            TransformFromCopy(f_axes[-1], plane.axis_labels[0]),
            TransformFromCopy(g_axes[-1], plane.axis_labels[1]),
        )
        self.play(
            Write(plane.background_lines, stroke_width=0.5, lag_ratio=0.01),
            Write(plane.faded_lines, stroke_width=0.5, lag_ratio=0.01),
        )
        self.add(plane, x_indicator2, y_indicator2)

        # Add plane lines
        h_line = Line().set_stroke(BLUE, 1)
        v_line = Line().set_stroke(YELLOW, 1)
        h_line.add_updater(lambda l: l.put_start_and_end_on(
            plane.c2p(0, get_y()), plane.c2p(get_x(), get_y())
        ))
        v_line.add_updater(lambda l: l.put_start_and_end_on(
            plane.c2p(get_x(), 0), plane.c2p(get_x(), get_y())
        ))

        dot = GlowDot(color=GREEN)
        dot.add_updater(lambda m: m.move_to(plane.c2p(get_x(), get_y())))
        xy_label = Tex("(x, y)", font_size=30)
        xy_label.add_updater(lambda m: m.next_to(dot, UR, buff=-SMALL_BUFF))

        self.play(LaggedStart(
            VFadeIn(h_line),
            VFadeIn(v_line),
            FadeIn(dot),
            VFadeIn(xy_label),
        ))
        self.wait()
        self.play(x_tracker.animate.set_value(1), run_time=2)
        self.play(y_tracker.animate.set_value(0.9), run_time=2)
        self.play(x_tracker.animate.set_value(0.2), run_time=2)
        self.wait()

        # Note probability density at a single point
        rect = SurroundingRectangle(xy_label, buff=0.05)
        rect.set_stroke(TEAL, 1)
        label = TexText("Probability density = $f(x)g(y)$", font_size=36)
        label.next_to(rect, UP)
        label.set_backstroke()

        prob_word = label["Probability"]
        equals = label["="]
        prob_word.save_state()
        prob_word.next_to(equals, LEFT)

        self.play(
            FadeIn(rect),
            FadeIn(prob_word, lag_ratio=0.1),
            FadeIn(equals),
        )
        self.wait()
        self.play(
            prob_word.animate.restore(),
            FadeIn(label["density"])
        )
        self.wait()

        self.play(LaggedStart(
            FadeTransform(f_label.copy(), label["f(x)"][0]),
            FadeTransform(g_label.copy(), label["g(y)"][0]),
            lag_ratio=0.3,
            run_time=2
        ))
        self.add(label)
        self.play(FadeOut(rect))
        self.wait()

        # Draw 3d graph
        to_fix = [
            f_axes, f_graph, f_label, x_indicator, x_label, x_line, x_dot,
            g_axes, g_graph, g_label, y_indicator, y_label, y_line, y_dot,
            label,
        ]
        for mobject in to_fix:
            mobject.fix_in_frame()
        plane.set_flat_stroke(False)

        three_d_axes = self.get_three_d_axes(plane)
        surface = three_d_axes.get_graph(
            lambda x, y: self.f(x) * self.g(y),
            resolution=self.graph_resolution,
        )

        self.play(
            FadeIn(surface),
            label.animate.set_x(FRAME_WIDTH / 4).to_edge(UP),
            self.frame.animate.reorient(-27, 78, 0).move_to([0.36, -0.62, 0.71]).set_height(5.66).set_anim_args(run_time=4),
        )
        surface.always_sort_to_camera(self.camera)
        self.play(
            self.frame.animate.reorient(68, 77, 0).move_to([-0.13, -1.12, -0.27]).set_height(9.37),
            run_time=5,
        )

        # Show two perspectives
        self.play(
            self.frame.animate.reorient(3, 83, 0).move_to([1.09, -0.82, -0.54]).set_height(6.91),
            run_time=4,
        )
        self.wait()
        self.play(
            self.frame.animate.reorient(89, 95, 0).move_to([0.63, -2.19, 2.56]).set_height(9.41),
            run_time=4,
        )
        self.wait()
        self.play(
            self.frame.animate.reorient(69, 75, 0).move_to([1.07, -1.37, -0.19]).set_height(7.64),
            run_time=5,
        )

    def get_plane(self):
        plane = NumberPlane(
            **self.plane_config,
            background_line_style=self.plane_line_style,
        )
        axis_labels = VGroup(
            Tex("x", font_size=24).next_to(plane.x_axis, RIGHT, SMALL_BUFF),
            Tex("y", font_size=24).next_to(plane.y_axis, UP, SMALL_BUFF),
        )
        axis_labels.insert_n_curves(100)
        axis_labels.make_jagged()
        plane.axis_labels = axis_labels
        plane.add(*axis_labels)
        return plane

    def get_three_d_axes(self, plane):
        axes = ThreeDAxes(
            plane.x_range,
            plane.y_range,
            (0, 1),
            width=plane.x_axis.get_width(),
            height=plane.y_axis.get_height(),
            depth=self.z_axis_height
        )
        axes.shift(plane.c2p(0, 0) - axes.c2p(0, 0, 0))
        axes.z_axis.apply_depth_test()
        return axes

    def f(self, x):
        return wedge_func(x)

    def g(self, y):
        return double_lump(y)


class DiagonalSlices(Introduce3DGraph):
    mesh_resolution = (21, 21)
    shadow_opacity = 0.25
    add_shadow = True
    shadow_bump = 0.01
    clip_plane_unit_coord = 0.45

    def setup(self):
        super().setup()
        plane, axes = self.add_plane_and_axes()
        self.s_tracker = ValueTracker(-2 * plane.x_range[1])
        get_s = self.s_tracker.get_value
        self.add_surface_group(axes, get_s)
        self.add_slice_graph(get_s)

        self.init_func_name()
        self.init_line_labels(get_s)

        self.add(self.func_name)

    def construct(self):
        # Get some nice local variables
        frame = self.camera.frame
        plane = self.plane
        axes = self.axes
        s_tracker = self.s_tracker
        slice_graph = self.slice_graph

        # Initial orientation
        self.frame.reorient(88, 90, 0).move_to([-0.31, -2.14, 2.16])
        self.play(frame.animate.reorient(40, 70).move_to(ORIGIN), run_time=10)
        self.play(
            s_tracker.animate.set_value(0.5),
            frame.animate.reorient(0, 0),
            VFadeIn(self.equation),
            FadeOut(axes.z_axis),
            run_time=6,
        )
        self.wait()

        # Show x + y = s slice
        self.play(
            FadeIn(self.ses_label, 0.5 * DOWN),
            MoveAlongPath(GlowDot(), slice_graph, run_time=5, remover=True)
        )
        self.wait()
        self.play(
            self.frame.animate.reorient(-22, 74, 0).move_to([-0.12, -0.16, 0.04]).set_height(5.45),
            run_time=3
        )
        self.wait()

        # Change s
        self.play(
            s_tracker.animate.set_value(1.5),
            self.frame.animate.reorient(-45, 75, 0).move_to([0.18, -0.14, 0.49]).set_height(3.0),
            run_time=6,
        )
        self.play(
            s_tracker.animate.set_value(-2.0),
            self.frame.animate.reorient(-5, 66, 0).move_to([-0.03, -0.18, 0.14]).set_height(6.35),
            run_time=20,
        )
        self.play(
            s_tracker.animate.set_value(2.0),
            self.frame.animate.reorient(16, 73, 0).move_to([-0.03, -0.18, 0.14]).set_height(6.35),
            run_time=15,
        )

    def add_plane_and_axes(self):
        frame = self.camera.frame
        frame.reorient(20, 70)
        plane = self.plane = self.get_plane()
        plane.axes.set_stroke(GREY_B)
        plane.set_flat_stroke(False)
        plane.remove(plane.faded_lines)
        axes = self.axes = self.get_three_d_axes(plane)

        self.add(axes, axes.z_axis)
        self.add(plane)

        self.plane = plane
        self.axes = axes

        return plane, axes

    def add_surface_group(self, axes, get_s):
        # Surface
        surface = axes.get_graph(
            lambda x, y: self.f(x) * self.g(y),
            resolution=self.graph_resolution
        )
        vect = axes.c2p(*2 * [self.clip_plane_unit_coord], 0)  # Why?
        surface.add_updater(lambda m: m.set_clip_plane(vect, -get_s()))
        surface.always_sort_to_camera(self.camera)

        surface_mesh = SurfaceMesh(surface, resolution=self.mesh_resolution)
        surface_mesh.set_stroke(WHITE, width=1, opacity=0.1)

        surface_group = Group(surface, surface_mesh)

        # Add shadow
        if self.add_shadow:
            surface_shadow = surface.copy()
            surface_shadow.set_opacity(self.shadow_opacity)
            surface_shadow.shift(self.shadow_bump * IN)
            self.add(surface_shadow)

            surface_group.add(surface_shadow)

        self.surface_group = surface_group
        self.add(surface_group)
        return surface_group

    def add_slice_graph(
        self, get_s,
        stroke_color=WHITE,
        stroke_width=2,
        fill_color=TEAL_D,
        fill_opacity=0.5,
        dx=0.01
    ):
        axes = self.axes

        def get_points(s):
            x_min, x_max = axes.x_range[:2]
            y_min, y_max = axes.y_range[:2]

            if s > 0:
                xs = np.arange(s - y_max, x_max, dx)
            else:
                xs = np.arange(x_min, s - y_min, dx)

            return axes.c2p(xs, s - xs, self.f(xs) * self.g(s - xs))

        graph = VMobject()
        graph.set_flat_stroke(False)
        graph.set_stroke(stroke_color, stroke_width)
        graph.set_fill(fill_color, fill_opacity)
        graph.add_updater(lambda m: m.set_points_as_corners(get_points(get_s())))

        self.add(graph)
        self.slice_graph = graph

    def init_func_name(self):
        self.func_name = Tex(
            R"f(x) \cdot g(y)",
            font_size=42,
        )
        self.func_name.to_corner(UL, buff=0.25)
        self.func_name.fix_in_frame()
        return self.func_name

    def init_line_labels(self, get_s):
        equation = Tex("x + y = 0.00")
        s_label = equation.make_number_changable("0.00")
        s_label.add_updater(lambda m: m.set_value(get_s()))
        equation.to_corner(UR)
        equation.fix_in_frame()

        ses_label = Tex(R"\{(x, s - x): x \in \mathds{R}\}", tex_to_color_map={"s": YELLOW}, font_size=30)
        ses_label.next_to(equation, DOWN, MED_LARGE_BUFF, aligned_edge=RIGHT)
        ses_label.fix_in_frame()

        self.equation = equation
        self.ses_label = ses_label

        return equation, ses_label


class SyncedSlices(DiagonalSlices):
    initial_s = 2
    add_shadow = False

    def setup(self):
        super().setup()
        self.func_name.set_x(-3)
        self.func_name.align_to(self.equation, UP)
        self.s_tracker.set_value(self.initial_s)
        self.equation.set_x(2)
        self.add(self.equation)

    def construct(self):
        s_tracker = self.s_tracker
        self.play(
            s_tracker.animate.set_value(-1.5),
            self.frame.animate.reorient(-28, 77, 0).move_to([0.17, -0.28, 0.25]).set_height(5.29),
            run_time=20,
        )
        self.play(
            s_tracker.animate.set_value(1.5),
            self.frame.animate.reorient(-10, 73, 0).move_to([0.15, -0.28, 0.22]).set_height(5.04),
            run_time=20,
        )
        self.play(
            s_tracker.animate.set_value(-0.5),
            self.frame.animate.reorient(-39, 79, 0).move_to([0.15, -0.28, 0.21]).set_height(5.04),
            run_time=20,
        )


class SyncedSlicesExpAndRect(SyncedSlices):
    initial_s = -3.0
    graph_resolution = (201, 201)
    add_shadow = True

    def construct(self):
        # Test
        s_tracker = self.s_tracker
        self.frame.reorient(-31, 68, 0).move_to([-1.51, 0.77, 0.22]).set_height(6.35)

        self.play(
            s_tracker.animate.set_value(1.0),
            self.frame.animate.reorient(-38, 72, 0).move_to([0.53, -0.5, 0.34]).set_height(6.75),
            run_time=15,
        )
        self.play(
            s_tracker.animate.set_value(-1.5),
            self.frame.animate.reorient(-45, 80, 0).move_to([-1.75, 0.32, 0.2]).set_height(5.04),
            run_time=10,
        )
        self.play(
            s_tracker.animate.set_value(-3.0),
            self.frame.animate.reorient(-28, 73, 0).move_to([-1.75, 0.32, 0.2]).set_height(5.04),
            run_time=10,
        )
        self.play(
            s_tracker.animate.set_value(1.0),
            self.frame.animate.reorient(-39, 65, 0).move_to([0.35, -0.53, 0.06]).set_height(6.38),
            run_time=20,
        )

    def f(self, x):
        return (x > -2) * np.exp(-2 - x)

    def g(self, x):
        return wedge_func(x)


class CleanExpAndRect(SyncedSlicesExpAndRect):
    def construct(self):
        self.remove(self.equation, self.func_name)

        s_tracker = self.s_tracker
        s_tracker.set_value(-1.5)

        surface, mesh, surface_shadow = self.surface_group
        mesh.make_jagged()
        surface_shadow.set_opacity(0.5)
        self.remove(self.surface_group)
        self.add(surface, surface_shadow, mesh, self.slice_graph)


class SyncedSlicesGaussian(SyncedSlices):
    add_shadow = True

    def construct(self):
        # Test
        self.func_name.set_x(0)
        self.equation.to_edge(RIGHT)

        s_tracker = self.s_tracker
        s_tracker.set_value(0)
        self.frame.reorient(0, 53, 0).move_to([0.05, 0.28, -0.23]).set_height(6.70)

        self.play(
            s_tracker.animate.set_value(-2.0),
            self.frame.animate.reorient(-38, 72, 0).move_to([0.53, -0.5, 0.34]).set_height(6.75),
            run_time=8,
        )
        self.play(
            s_tracker.animate.set_value(1.5),
            self.frame.animate.reorient(-7, 61, 0).move_to([0.5, -0.06, 0.4]).set_height(3.85),
            run_time=10,
        )
        self.play(
            s_tracker.animate.set_value(-1.5),
            self.frame.animate.reorient(-28, 73, 0).move_to([-0.06, -0.56, 0.26]).set_height(5.04),
            run_time=20,
        )

    def f(self, x):
        return gauss_func(x, 0, 0.5)

    def g(self, x):
        return gauss_func(x, 0, 0.5)


class AnalyzeStepAlongDiagonalLine(DiagonalSlices):
    initial_s = 0.5
    dx = 1 / 8

    def construct(self):
        # Setup
        s_tracker = self.s_tracker
        frame = self.frame
        surface, mesh, shadow = self.surface_group

        s_tracker.set_value(self.initial_s)
        frame.reorient(-27, 73, 0)

        self.slice_graph.update()
        self.slice_graph.clear_updaters()

        # Focus on line
        line = self.plane.get_graph(lambda x: self.initial_s - x)
        line.set_stroke(WHITE, 3)

        self.play(
            frame.animate.reorient(0, 0, 0).center().set_height(7),
            run_time=2
        )
        self.play(
            FadeOut(surface),
            FadeOut(mesh),
            FadeOut(shadow),
            FadeOut(self.slice_graph),
            FadeIn(line),
        )
        self.wait()

        # Show small step
        segment = line.copy()
        segment.pointwise_become_partial(line, 12 * self.dx / 4, 13 * self.dx / 4)
        segment.set_stroke(YELLOW, 3)

        segment.rotate(45 * DEGREES).scale(3)
        brace = Brace(segment, DOWN, buff=SMALL_BUFF)
        center = segment.get_center()
        VGroup(brace, segment).scale(1 / 3, about_point=center).rotate(-45 * DEGREES, about_point=center)

        tex_kw = dict(font_size=12)
        step_word = Text("Step", **tex_kw)
        step_word.next_to(brace.get_center(), DL, SMALL_BUFF)
        step_word.shift(0.05 * UP)
        step_word.set_backstroke()

        dx_line = DashedLine(segment.get_corner(UL), segment.get_corner(UR), dash_length=0.01)
        dy_line = DashedLine(segment.get_corner(UR), segment.get_corner(DR), dash_length=0.01)
        dx_line.set_stroke(RED)
        dy_line.set_stroke(GREEN)

        dx_label = Tex(R"\Delta x", **tex_kw)
        dx_label.match_color(dx_line)
        dx_label.next_to(dx_line, UP, buff=0.05)

        dy_label = Tex(R"\Delta y", **tex_kw)
        dy_label.next_to(dy_line, RIGHT, buff=0.05)
        dy_label.match_color(dy_line)

        self.play(
            ShowCreation(segment),
            line.animate.set_stroke(width=1),
            GrowFromCenter(brace),
            Write(step_word),
            frame.animate.scale(0.3, about_point=1.5 * UL).set_anim_args(run_time=2),
        )
        self.wait()
        self.play(
            ShowCreation(dx_line),
            ShowCreation(dy_line),
            FadeIn(dx_label),
            FadeIn(dy_label),
        )
        self.wait()

        # Show equation
        rhs = Tex(R"= \sqrt{2} \cdot \Delta x", **tex_kw)
        rhs.move_to(step_word, RIGHT)
        rhs.set_backstroke()

        self.play(
            FadeTransform(dx_label.copy(), rhs[R"\Delta x"][0]),
            Write(rhs[R"= \sqrt{2} \cdot "]),
            step_word.animate.next_to(rhs, LEFT, buff=0.05).shift(0.025 * DOWN)
        )
        self.wait()

        # Show graph again
        segment.set_flat_stroke(False)
        line.set_flat_stroke(False)
        self.add(surface, mesh)
        self.play(
            FadeIn(surface),
            FadeIn(mesh),
            FadeIn(self.slice_graph),
            FadeOut(self.equation),
            FadeOut(self.ses_label),
            FadeOut(self.func_name),
            self.frame.animate.reorient(-42, 79, 0).move_to([-0.15, 0.73, 1.18]).set_height(3.50),
            run_time=2,
        )

        # Show riemann rectangles
        rects = VGroup()
        x_unit = self.plane.x_axis.get_unit_size()
        z_unit = self.axes.z_axis.get_unit_size()
        dx = self.dx
        for x in np.arange(-2, 2, dx):
            y = self.initial_s - x
            if not (-2 <= y <= 2):
                continue
            rect = Rectangle(
                width=math.sqrt(2) * x_unit * dx,
                height=self.f(x) * self.g(y) * z_unit
            )
            rect.rotate(90 * DEGREES, RIGHT)
            rect.rotate(-45 * DEGREES, OUT)
            rect.shift(self.plane.c2p(x, y) - rect.get_corner([-1, 1, -1]))
            rects.add(rect)

        rects.set_fill(TEAL, 0.5)
        rects.set_stroke(WHITE, 1)
        rects.set_flat_stroke(False)

        self.play(
            Write(rects),
            self.slice_graph.animate.set_fill(opacity=0.1)
        )
        self.add(rects)
        self.wait()


class FullGaussianExample(DiagonalSlices):
    plane_config = dict(
        x_range=(-3, 3),
        y_range=(-3, 3),
        width=8.0,
        height=8.0,
    )
    mesh_resolution = (25, 25)
    clip_plane_unit_coord = 0.565

    def construct(self):
        # Variables
        frame = self.frame
        axes = self.axes
        s_tracker = self.s_tracker
        surface_group = self.surface_group
        slice_graph = self.slice_graph
        self.func_name.set_x(0)

        # Emphasize rotational symmetry
        # frame.add_updater(lambda m: m.set_theta(30 * math.sin(0.1 * self.time) * DEGREES))
        curve = VMobject()
        curve.set_stroke(YELLOW, 3)
        dx = 0.1
        xs = np.arange(*axes.x_range, dx)
        curve.set_points_smoothly(axes.c2p(xs, np.zeros(xs.size), self.f(xs)))
        curve.set_flat_stroke(False)

        self.add(surface_group, curve)
        self.play(ShowCreation(curve))
        self.play(Rotate(curve, TAU, about_point=ORIGIN, run_time=7))
        self.wait()


    def f(self, x):
        return np.exp(-x**2)

    def g(self, x):
        return np.exp(-x**2)
