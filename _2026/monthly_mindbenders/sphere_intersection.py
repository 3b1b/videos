from manim_imports_ext import *
from scipy.spatial.transform import Rotation, Slerp


class Spheres(InteractiveScene):
    def construct(self):
        # Set up camera
        self.camera.frame.reorient(-23, 73, 0, (-0.14, -0.02, -0.23), 13.36)
        self.camera.light_source.move_to(OUT * 2 + LEFT * 2)

        # Add spheres
        R = 2
        sphere_shell_1 = Sphere(radius=R).set_color(BLUE).set_opacity(0.8)
        sphere_shell_2 = sphere_shell_1.copy()
        sphere_mesh_opacity = 0.3
        sphere1 = SurfaceMesh(sphere_shell_1, stroke_width=0.4, stroke_opacity=sphere_mesh_opacity, stroke_color=WHITE, resolution=(100, 30))
        sphere1.remove(sphere1[0])
        sphere2 = sphere1.copy()
        sphere_shell_1.add_updater(lambda m: m.move_to(sphere1))
        sphere_shell_2.add_updater(lambda m: m.move_to(sphere2))
        Group(sphere1, sphere2).arrange(buff=-0.75)
        self.play(
            AnimationGroup(
                ShowCreation(sphere_shell_1),
                ShowCreation(sphere_shell_2.reverse_points()),
                lag_ratio=0.2,
                run_time=2
            ),
            self.camera.frame.animate(run_time=3).reorient(30, 51, 0, (0.05, 0.20, -0.09), 13.62)
        )

        # Show the radii
        arrow1 = Arrow(sphere1.get_center(), sphere1.get_center() + R * DL / np.linalg.norm(DL))
        label1 = Tex("1")
        label1.apply_matrix(self.camera.frame.get_orientation().as_matrix())
        label1.next_to(arrow1, OUT)
        arrow2 = Arrow(sphere2.get_center(), sphere2.get_center() + R * UR / np.linalg.norm(UR))
        label2 = label1.copy().next_to(arrow2, OUT)
        self.play(
            Group(sphere_shell_1, sphere_shell_2).animate.set_opacity(0.4),
            GrowArrow(arrow1),
            Write(label1),
            GrowArrow(arrow2),
            Write(label2)
        )
        self.wait(0.7)
        self.play(
            FadeOut(VGroup(arrow1, label1, arrow2, label2)),
            Group(sphere_shell_1, sphere_shell_2).animate.set_opacity(0.8),
            run_time=0.6
        )

        # Move them to random locations
        self.set_camera_target_position(55, 49, 0, (-0.51, -0.61, -0.09), 13.62, drift_time=2.9)

        def get_random_sphere_location():
            gaussian_point = np.random.normal(0, 1, 3)
            return gaussian_point / np.linalg.norm(gaussian_point) * R * (random.random() ** (1 / 3))
        VGroup(sphere1, sphere2).set_opacity(0)
        np.random.seed(2)
        for _ in range(3):
            self.play(
                sphere1.animate.move_to(get_random_sphere_location()),
                sphere2.animate.move_to(get_random_sphere_location()),
                run_time=1
            )

        # Add the intersection region
        def get_intersection():
            w = sphere2.get_center() - sphere1.get_center()
            d = np.linalg.norm(w)
            w = w / d
            w_perp_1 = np.cross(w, (0, 0, 1))
            w_perp_2 = np.cross(w, w_perp_1)

            def get_coords(u, v):
                circle_center = sphere1.get_center() + w * u
                if u < d / 2:
                    r = np.sqrt(R**2 - (d - u)**2)
                else:
                    r = np.sqrt(R**2 - u**2)
                offset = r * np.cos(v) * w_perp_1 + r * math.sin(v) * w_perp_2
                return circle_center + offset

            return ParametricSurface(
                get_coords,
                u_range=(d - R, R),
                v_range=(0, 2 * PI)
            ).set_color(RED)
        intersection = always_redraw(get_intersection)
        intersection.suspend_updating()
        self.add(intersection, sphere_shell_1, sphere_shell_2, sphere1, sphere2)
        self.play(
            FadeIn(intersection),
            VGroup(sphere1, sphere2).animate.set_stroke(opacity=sphere_mesh_opacity),
            FadeOut(Group(sphere_shell_1, sphere_shell_2)),
            self.camera.frame.animate.reorient(-47, 83, 0, (0.16, -0.09, 0.90), 10.62),
            run_time=2
        )
        intersection.resume_updating()

        # Keep moving them around
        np.random.seed(0)
        self.set_camera_target_position(17, 62, 0, (1.17, -0.10, 0.25), 9.32, drift_time=3)
        for _ in range(3):
            self.play(
                sphere1.animate.move_to(get_random_sphere_location()),
                sphere2.animate.move_to(get_random_sphere_location()),
                run_time=1.08
            )

        # Add axes
        axes = ThreeDAxes(
            x_range=(-2, 2, 1),
            y_range=(-2, 2, 1),
            z_range=(-2, 2, 1)
        ).set_width(4 * R)
        axes_opacity_tracker = ValueTracker(0)
        axes.add_updater(lambda m: m.set_opacity(axes_opacity_tracker.get_value()))

        def look_at_camera(m):
            template = m.copy()

            def update_m(m):
                center = m.get_center()
                m.become(template)
                m.apply_matrix(
                    self.camera.frame.get_orientation().as_matrix(),
                    about_point=template.get_center(),
                )
                m.move_to(center)
                m.set_opacity(axes_opacity_tracker.get_value())
            m.add_updater(update_m)

        coordinate_labels = axes.add_coordinate_labels()
        for axis_labels in coordinate_labels:
            for label in axis_labels:
                look_at_camera(label)
        self.add(axes, coordinate_labels)
        def get_random_sphere_location_2(): return get_random_sphere_location() * 2
        sphere1, sphere2 = sphere2, sphere1
        intersection_opacity_tracker = ValueTracker(1)
        intersection.add_updater(lambda m: m.set_opacity(intersection_opacity_tracker.get_value()))
        sphere2.add_updater(lambda m: m.set_stroke(opacity=sphere_mesh_opacity * intersection_opacity_tracker.get_value()))
        self.play(
            self.camera.frame.animate.reorient(17, 62, 0, (0.09, -0.33, 0.43), 15.50),
            AnimationGroup(
                axes_opacity_tracker.animate.set_value(1),
                AnimationGroup(
                    sphere1.animate.move_to(axes.c2p(0, 0, 0)),
                    FadeOut(sphere2),
                    intersection_opacity_tracker.animate.set_value(0)
                ),
                lag_ratio=0.3
            )
        )
        self.remove(intersection)

        # Show a cloud of dots
        self.set_camera_target_position(49, 48, 0, (0.23, -0.45, 0.32), 9.28, drift_time=30)
        dot = Sphere(radius=0.025).set_color(YELLOW)

        cloud = Group(dot, *[dot.copy().move_to(get_random_sphere_location_2()) for _ in range(1500)]).set_opacity(0.4)
        self.play(
            AnimationGroup(
                *[
                    GrowFromCenter(dot)
                    for dot in cloud
                ],
                lag_ratio=0.001
            ),
            run_time=1.5
        )

        # Pick one of them
        sphere2.move_to([1.7, 1.6, 1.2])
        dot_opacity_tracker = ValueTracker(1)

        def get_line():
            v = sphere2.get_center() - axes.c2p(0, 0, 0)
            v = v / np.linalg.norm(v) * np.linalg.norm(axes.c2p(2, 0, 0))
            return DashedLine(axes.c2p(0, 0, 0), v, buff=0, stroke_width=1)
        line = always_redraw(get_line)

        def get_line_label():
            v = sphere2.get_center() - axes.c2p(0, 0, 0)
            v = v / np.linalg.norm(v) * np.linalg.norm(axes.c2p(2, 0, 0))
            tex = Tex(R"d \leq 2").set_z_index(100)
            return tex.next_to(
                line.get_end(), OUT, buff=0.43
            ).apply_matrix(
                self.camera.frame.get_orientation().as_matrix(),
                about_point=tex.get_center(),
            ).set_opacity(
                dot_opacity_tracker.get_value()
            )
        line_label = always_redraw(get_line_label)
        dot.add_updater(lambda m: m.move_to(sphere2).set_opacity(dot_opacity_tracker.get_value()))
        line.add_updater(lambda m: m.set_opacity(dot_opacity_tracker.get_value()))
        self.play(
            FadeOut(cloud[1:]),
            dot.animate.scale(2).set_opacity(1),
            ShowCreation(line, suspend_mobject_updating=True),
            Write(line_label, suspend_mobject_updating=True),
            run_time=1.9
        )

        # Keep moving around the second sphere to random locations with the first one centered at the origin
        for i in range(5):
            anims = [sphere2.animate.move_to(get_random_sphere_location_2())]
            if i == 0:
                self.add(intersection, sphere1, sphere2)
                anims.append(intersection_opacity_tracker.animate.set_value(1))
            if i == 2:
                anims.append(dot_opacity_tracker.animate.set_value(0))
            self.play(*anims, run_time=1.85)
            if i == 2:
                self.remove(dot, line, line_label)

        for _ in range(60):
            sphere2.move_to(get_random_sphere_location_2())
            self.wait(0.3)

    def set_camera_target_position(
        self,
        theta_degrees=None,
        phi_degrees=None,
        gamma_degrees=None,
        center=None,
        height=None,
        drift_time=2.0,
    ):
        frame = self.camera.frame
        frame.clear_updaters()
        initial_orientation = frame.get_orientation()
        initial_height = frame.get_height()
        initial_eye = frame.get_implied_camera_location()
        target_frame = frame.copy()
        target_frame.reorient(theta_degrees, phi_degrees, gamma_degrees, center, height)
        target_orientation = target_frame.get_orientation()
        target_height = target_frame.get_height()
        target_eye = target_frame.get_implied_camera_location()
        fovy = frame.get_field_of_view()
        slerp = Slerp([0, 1], Rotation.concatenate([initial_orientation, target_orientation]))
        drift_time = max(drift_time, 1e-4)
        elapsed = 0.0

        def update_camera(f, dt):
            nonlocal elapsed
            elapsed += dt
            t = min(elapsed / drift_time, 1.0)
            alpha = smooth(t)
            current_orientation = slerp(alpha)
            current_height = interpolate(initial_height, target_height, alpha)
            current_eye = interpolate(initial_eye, target_eye, alpha)
            focal_distance = 0.5 * current_height / np.tan(0.5 * fovy)
            to_camera = current_orientation.as_matrix().T[2]
            current_center = current_eye - focal_distance * to_camera
            f.set_orientation(current_orientation)
            f.move_to(current_center)
            f.set_height(current_height)
            if t >= 1.0:
                f.remove_updater(update_camera)
        frame.add_updater(update_camera)
