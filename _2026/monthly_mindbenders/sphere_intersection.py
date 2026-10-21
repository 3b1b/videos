from manim_imports_ext import *


class Spheres(InteractiveScene):
    def construct(self):
        # Add spheres and the caps
        R = 1
        sphere1 = Sphere(radius=R).set_color(BLUE).set_opacity(0.3)
        sphere2 = sphere1.copy()
        Group(sphere1, sphere2).arrange(buff=-0.5)

        def get_caps():
            w = sphere2.get_center() - sphere1.get_center()
            d = np.linalg.norm(w)
            w = w / d
            w_perp_1 = np.cross(w, (0, 0, 1))
            w_perp_2 = np.cross(w, w_perp_1)
            u_min = d - R
            u_max = d / 2
            def get_coords_1(u, v):
                circle_center = sphere1.get_center() + w*u
                r = np.sqrt(R**2 - (d - u)**2)
                offset = r*np.cos(v)*w_perp_1 + r*math.sin(v)*w_perp_2
                return circle_center + offset

            def get_coords_2(u, v):
                circle_center = sphere2.get_center() - w*u
                r = np.sqrt(R**2 - (d - u)**2)
                offset = r*np.cos(v)*w_perp_1 + r*math.sin(v)*w_perp_2
                return circle_center + offset

            cap1 = ParametricSurface(
                get_coords_1,
                u_range=(u_min, u_max),
                v_range=(0, 2 * PI)
            ).set_color(YELLOW).set_opacity(0.5)
            cap2 = ParametricSurface(
                get_coords_2,
                u_range=(u_min, u_max),
                v_range=(0, 2 * PI)
            ).set_color(YELLOW).set_opacity(0.5)
            return Group(cap1, cap2)
        caps = always_redraw(get_caps)
        self.add(caps, sphere1, sphere2)

        # Move them together
        self.play(Group(sphere1, sphere2).animate.arrange(buff=1))
        self.play(Group(sphere1, sphere2).animate.arrange(buff=0))
        self.play(Group(sphere1, sphere2).animate.arrange(buff=-1))