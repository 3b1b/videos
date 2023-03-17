from manim_imports_ext import *
from _2022.convolutions.continuous import *
from _2023.clt.main import *

class GaussianIntegral(ThreeDScene, InteractiveScene):
    def func(self, x, y):
        return np.exp(-x**2 - y**2)

    def get_axes(
        self,
        x_range=(-3, 3),
        y_range=(-3, 3),
        z_range=(0, 2, 0.5),
        width=8,
        height=8,
        depth=4,
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
        graph.suspend_updating()

        graph_mesh = SurfaceMesh(graph, resolution=(21, 21))
        graph_mesh.set_stroke(WHITE, 0.5, opacity=0.25)
        graph_mesh.set_flat_stroke(False)

        self.add(axes)

        # Animate in by rotating e^{-x^2} (TODO, needs label)
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

        axes.save_state()
        frame.reorient(0, 90)
        frame.move_to(OUT + 2 * UP)
        axes.y_axis.set_opacity(0)
        self.play(ShowCreation(bell2d))
        self.wait()

        self.play(
            ShowCreation(bell_halves[0]),
            ShowCreation(bell_halves[1]),
            Rotate(bell2d, PI, axis=OUT, about_point=axes.c2p(0, 0, 0)),
            frame.animate.move_to(ORIGIN).reorient(-20, 70),
            Restore(axes),
            run_time=3
        )
        self.wait()
        self.play(
            FadeOut(bell_halves, 0.01 * IN),
            FadeOut(bell2d),
            FadeIn(graph, 0.01 * IN),
            FadeIn(graph_mesh, 0.01 * IN),
        )
        self.wait()

        # Show the meaning of r

        # Dynamic cylinder
        graph.set_opacity(0.1)  # When to do this?
        cylinder, r_tracker = self.get_dynamic_cylinder(axes)
        cylinders = Group(*(
            self.get_cylinder(axes, r, opacity=0.2)
            for r in np.arange(0, 3, 0.1)
        ))

        r_tracker.set_value(0)
        self.add(*cylinders, cylinder)
        self.play(
            FadeIn(cylinders, lag_ratio=0.9),
            r_tracker.animate.set_value(3).set_anim_args(rate_func=linear),
            run_time=5,
        )
        self.add(*cylinders)

        # Show the dimensions of one particular cylinder


class CartesianSlices(GaussianIntegral):
    def construct(self):
        # Setup
        frame = self.frame
        axes = self.get_axes()

        graph = self.get_gaussian_graph(axes)
        graph_mesh = SurfaceMesh(graph, resolution=(21, 21))
        graph_mesh.set_stroke(WHITE, 0.5, opacity=0.25)
        graph_mesh.set_flat_stroke(False)

        self.add(axes)
        self.add(graph)
        self.add(graph_mesh)

        # Dynamic slice
        x_slice, y_tracker = self.get_dynamic_slice(axes)
        y_unit = axes.y_axis.get_unit_size()
        graph.add_updater(lambda m: m.set_clip_plane(UP, -y_tracker.get_value() * y_unit))

        y_tracker.set_value(-3)
        self.add(x_slice)
        self.play(y_tracker.animate.set_value(0), run_time=3)
        self.wait()

        # Show many slices
        x_slices = VGroup()
        for y in np.arange(3, -3, -0.25):
            y_tracker.set_value(y)
            x_slice.update()
            x_slices.add(x_slice.copy().clear_updaters())
        y_tracker.set_value(0)

        x_slices.use_winding_fill(False)
        x_slices.deactivate_depth_test()
        x_slices.set_stroke(BLUE, 2, 0.5)
        x_slices.set_flat_stroke(False)

        self.add(x_slice, x_slices, graph)
        self.play(
            FadeOut(graph, time_span=(0, 1)),
            FadeOut(x_slice, time_span=(0, 1)),
            FadeIn(x_slices, lag_ratio=0.1),
            frame.animate.reorient(-80),
            run_time=4
        )
        self.wait(2)
        self.add(x_slices, graph, x_slice, graph_mesh)
        graph.set_opacity(0.25)
        x_slice.deactivate_depth_test()
        self.play(
            FadeOut(x_slices, lag_ratio=0.1),
            FadeIn(graph),
            FadeIn(x_slice),
            frame.animate.reorient(-25).set_height(6).set_anim_args(run_time=3),
        )

        # Discuss area of each slice
        get_y = y_tracker.get_value
        x_slice_label = Tex("0.00 e^{-x^2}", t2c={"x": BLUE})
        coef = x_slice_label.make_number_changable("0.00")
        coef.set_color(YELLOW)
        coef.add_updater(lambda m: m.set_value(math.exp(-get_y()**2)).rotate(90 * DEGREES, RIGHT))

        x_term = x_slice_label[1:]
        y_term = Tex("e^{-y^2}", t2c={"y": YELLOW})
        y_term.move_to(coef, RIGHT)
        y_term.next_to(x_term, LEFT, SMALL_BUFF, aligned_edge=DOWN)

        brace = Brace(coef, UP, MED_SMALL_BUFF)

        x_slice_label.add(brace, y_term)
        x_slice_label.use_winding_fill(True)
        x_slice_label.rotate(90 * DEGREES, RIGHT)
        x_slice_label.add_updater(lambda m: m.next_to(x_slice.pfp(0.6), OUT + RIGHT))

        self.play(
            Write(y_term),
            Write(x_term),
            frame.animate.reorient(-15),
        )
        self.wait()
        self.play(
            y_term.animate.next_to(brace, OUT, SMALL_BUFF),
            GrowFromCenter(brace),
            FadeIn(coef),
        )
        self.add(x_slice_label)

        self.play(y_tracker.animate.set_value(-0.5), run_time=3)
        self.wait()
        self.play(y_tracker.animate.set_value(-1), run_time=4)
        self.wait()


        # Integrate slices