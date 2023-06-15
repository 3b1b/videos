from manim_imports_ext import *
from _2023.convolutions2.continuous import *


class SetUpThreeDGraph(InteractiveScene):
    def construct(self):
        pass


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

