from manim_imports_ext import *
from scipy.spatial.transform import Rotation, Slerp


class Spheres(InteractiveScene):
    def construct(self):
        # Add spheres
        self.camera.frame.reorient(-23, 73, 0, (-0.14, -0.02, -0.23), 13.36)
        self.camera.light_source.move_to(OUT * 2 + LEFT * 2)

        R = 2
        sphere1 = SurfaceMesh(Sphere(radius=R), stroke_opacity=0.5, stroke_color=BLUE_D, resolution=(100, 30))
        sphere2 = sphere1.copy()
        Group(sphere1, sphere2).arrange(buff=-0.75)
        self.play(
            AnimationGroup(
                ShowCreation(sphere1),
                ShowCreation(sphere2.reverse_points()),
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
            VGroup(sphere1, sphere2).animate.set_stroke(opacity=0.2),
            GrowArrow(arrow1),
            Write(label1),
            GrowArrow(arrow2),
            Write(label2)
        )
        self.wait(0.7)
        self.play(
            FadeOut(VGroup(arrow1, label1, arrow2, label2)),
            VGroup(sphere1, sphere2).animate.set_stroke(opacity=0.5),
            run_time=0.6
        )

        # Move them to random locations
        self.set_camera_target_position(111, 59, 0, (0.07, 0.28, -0.18), 15.19, drift_time=2.9)
        def get_random_sphere_location():
            gaussian_point = np.random.normal(0, 1, 3)
            return gaussian_point / np.linalg.norm(gaussian_point) * R * (random.random() ** (1/3))
        for _ in range(3):
            self.play(
                sphere1.animate.move_to(get_random_sphere_location()),
                sphere2.animate.move_to(get_random_sphere_location()),
                run_time=1
            )

        # Add the caps
        def get_cap():
            w = sphere2.get_center() - sphere1.get_center()
            d = np.linalg.norm(w)
            w = w / d
            w_perp_1 = np.cross(w, (0, 0, 1))
            w_perp_2 = np.cross(w, w_perp_1)
            def get_coords(u, v):
                circle_center = sphere1.get_center() + w*u
                if u < d / 2:
                    r = np.sqrt(R**2 - (d - u)**2)
                else:
                    r = np.sqrt(R**2 - u**2)
                offset = r*np.cos(v)*w_perp_1 + r*math.sin(v)*w_perp_2
                return circle_center + offset

            return ParametricSurface(
                get_coords,
                u_range=(d - R, R),
                v_range=(0, 2 * PI)
            ).set_color(YELLOW)
        cap = always_redraw(get_cap)
        cap.suspend_updating()
        self.add(cap, sphere1, sphere2)
        self.play(
            FadeIn(cap),
            VGroup(sphere1, sphere2).animate.set_stroke(opacity=0.3),
            self.camera.frame.animate.reorient(-19, 94, 0, (0.44, 0.07, 0.66), 7.12),
            run_time=2
        )
        cap.resume_updating()

        # Keep moving them around
        self.set_camera_target_position(18, 63, 0, (0.01, -0.59, 0.02), 9.32, drift_time=4)
        for _ in range(2):
            self.play(
                sphere1.animate.move_to(get_random_sphere_location()),
                sphere2.animate.move_to(get_random_sphere_location()),
                run_time=1
            )

        # Add axes
        axes = ThreeDAxes(
            x_range=(-2, 2, 1),
            y_range=(-2, 2, 1),
            z_range=(-2, 2, 1)
        ).set_width(4*R)
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
        get_random_sphere_location_2 = lambda: get_random_sphere_location() * 2
        self.play(
            AnimationGroup(
                axes_opacity_tracker.animate.set_value(1),
                sphere2.animate.move_to(axes.c2p(0, 0, 0)),
                sphere1.animate.move_to(get_random_sphere_location_2()),
                lag_ratio=0.3
            )
        )

        # Keep moving around the second sphere to random locations with the first one centered at the origin
        self.set_camera_target_position(54, 59, 0, (0.20, -0.13, 0.56), 9.28, drift_time=35.9)
        for _ in range(18):
            self.play(sphere1.animate.move_to(get_random_sphere_location_2()), run_time=2)


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
