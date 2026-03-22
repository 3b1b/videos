from manim_imports_ext import *


class FlattenCone(InteractiveScene):
    def construct(self):
        # Add surfaces
        frame = self.frame
        radius = 3.0
        axes = ThreeDAxes()
        axes.set_stroke(WHITE, width=2, opacity=0.5)
        axes.set_anti_alias_width(5)
        axes.apply_depth_test()
        axes.set_z_index(1)
        self.add(axes)

        kw = dict(radius=0.1)
        tracking_dots = Group(
            TrueDot(2 * RIGHT, color=RED, **kw),
            TrueDot(2 * LEFT, color=RED, **kw),
            TrueDot(2 * UP, color=GREEN, **kw),
            TrueDot(2 * DOWN, color=GREEN, **kw),
            TrueDot(2 * IN, color=BLUE, **kw),
            TrueDot(2 * OUT, color=BLUE, **kw),
        )
        for dot in tracking_dots:
            dot.make_3d()
        tracking_dots.set_z_index(2)
        tracking_dots.deactivate_depth_test()
        self.add(tracking_dots)

        theta = TAU * math.sin(TAU / 8)  # Angle off slice of paper

        def cone_func(u, v):
            return np.array([
                u * math.cos(TAU * v),
                u * math.sin(TAU * v),
                0.5 * radius - u
            ])

        def flat_cone_func(u, v):
            return np.array([
                u * math.cos(theta * v + 0.5 * (TAU - theta)),
                u * math.sin(theta * v + 0.5 * (TAU - theta)),
                0,
            ])

        range_kw = dict(
            u_range=(0, radius),
            v_range=(0, 1)
        )
        cone = ParametricSurface(cone_func, **range_kw)
        flat_cone = ParametricSurface(flat_cone_func, **range_kw)

        for surface in [cone, flat_cone]:
            surface.set_color(GREY_D)
            surface.set_shading(0.5, 0.25, 0.5)

        frame.reorient(-25, 69, 0)
        frame.set_x(1e-1)
        self.play(
            frame.animate.reorient(50, 80, 0),
            ShowCreation(cone, time_span=(0, 2)),
            run_time=4
        )

        # Add line
        def get_line(uv_func):
            line = Line().set_stroke(RED, 5)
            line.set_points_as_corners([
                uv_func(radius, 0.75),
                uv_func(0, 0.75),
                uv_func(radius, 0.25),
            ])
            line.apply_depth_test()
            line.shift(1e-2 * OUT)
            return line

        def get_div_line(uv_func):
            line = DashedLine(uv_func(radius, 0.5), uv_func(0, 0)).set_stroke(YELLOW, 4)
            line.apply_depth_test()
            line.shift(1e-2 * OUT)
            return line

        cone_line = get_line(cone_func)
        flat_line = get_line(flat_cone_func)

        cone_div_line = get_div_line(cone_func)
        flat_div_line = get_div_line(flat_cone_func)

        self.play(
            ShowCreation(cone_line, time_span=(0, 3)),
            frame.animate.reorient(0, 2, 0),
            run_time=4,
        )
        self.play(ShowCreation(cone_div_line))
        self.wait()

        # Flatten
        kw = dict(time_span=(1.5, 3))
        self.play(
            Transform(cone, flat_cone, **kw),
            Transform(cone_line, flat_line, **kw),
            Transform(cone_div_line, flat_div_line, **kw),
            frame.animate.reorient(21, 74, 0),
            run_time=3
        )
        self.play(frame.animate.reorient(0, 0, 0), run_time=5)
        self.wait()


class SquareOnASphere(InteractiveScene):
    def construct(self):
        # Add sphere
        frame = self.frame
        self.camera.light_source.set_y(5)
        sphere = Sphere()
        sphere.set_color(GREY_D)
        sphere.set_shading(0.5, 0.25, 0.25)
        mesh = SurfaceMesh(sphere, resolution=(41, 21), normal_nudge=1e-3)
        mesh.set_stroke(WHITE, 1, 0.25)

        frame.reorient(0, 84, 0, ORIGIN, 2.50)
        self.add(sphere, mesh)

        # Show "square" lines
        arc_len = 40 * DEG
        u0 = 270 * DEG
        v0 = 100 * DEG

        line1 = ParametricCurve(lambda t: sphere.uv_func(u0, v0 + arc_len * t))
        line1.set_stroke(RED_D, 3)
        all_lines = VGroup(line1)

        self.play(
            ShowCreation(line1, time_span=(0, 2)),
            frame.animate.reorient(5, 57, 0),
            run_time=3
        )

        orientations = [
            (32, 65, 0),
            (23, 85, 0),
            (6, 84, 0),
        ]

        for orientation in orientations:
            last_line = all_lines[-1]
            elbow = self.get_elbow(last_line)
            new_line = last_line.copy()
            new_line = self.get_rotated_arc(last_line, 90 * DEG)
            new_line.reverse_points()
            self.play(
                ShowCreation(new_line, time_span=(0, 2)),
                ShowCreation(elbow, time_span=(0, 1)),
                frame.animate.reorient(*orientation),
                run_time=3
            )
            all_lines.add(new_line)

        # Show transitions
        for line in all_lines[:3]:
            anim = UpdateFromAlphaFunc(
                line.copy(),
                lambda m, a: m.match_points(self.get_rotated_arc(line, a * 90 * DEG)),
                run_time=3,
                time_span=(0, 2)
            )
            if line is all_lines[0]:
                self.play(
                    anim,
                    frame.animate.reorient(7, 61, 0, (-0.02, -0.01, -0.01), 2.68).set_anim_args(run_time=3)
                )
                frame.add_ambient_rotation(2 * DEG)
            else:
                self.play(anim)
        self.wait(5)

    def get_rotated_arc(self, arc, angle):
        return arc.copy().rotate(angle, about_point=arc.get_end(), axis=arc.get_end())

    def get_elbow(self, arc, prop=0.1):
        corner = arc.get_end()
        rot_arc = self.get_rotated_arc(arc, 90 * DEG)
        v1 = arc.get_points()[-2] - corner
        v2 = rot_arc.get_points()[-2] - corner
        elbow = VMobject()
        elbow.set_points_as_corners([corner + v1, corner + v1 + v2, corner + v2])
        elbow.set_stroke(WHITE, 2)
        return elbow
