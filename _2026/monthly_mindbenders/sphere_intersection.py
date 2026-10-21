from manim_imports_ext import *


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
            self.camera.frame.animate(run_time=4).reorient(30, 51, 0, (0.05, 0.20, -0.09), 13.62)
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
            self.camera.frame.animate.reorient(30, 51, 0, (0.05, 0.09, -0.20), 7.12),
            run_time=2.5
        )
        cap.resume_updating()

        # Move them together
        self.play(Group(sphere1, sphere2).animate.arrange(buff=1))
        self.play(Group(sphere1, sphere2).animate.arrange(buff=0))
        self.play(Group(sphere1, sphere2).animate.arrange(buff=-1))
        self.play(sphere2.animate.shift(OUT * 0.3))
