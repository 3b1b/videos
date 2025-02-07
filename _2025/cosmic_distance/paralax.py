from manim_imports_ext import *
from _2025.cosmic_distance.planets import *


class SimpleDotsParalax(InteractiveScene):
    show_pi_perspective = False

    def construct(self):
        # Add stars
        frame = self.frame
        self.set_floor_plane("xz")

        height = 4
        cube = VCube(height)
        cube.set_fill(opacity=0)
        cube.set_stroke(BLUE, 2)

        n_stars = 200
        stars = GlowDots(np.random.uniform(-1, 1, (n_stars, 3)))
        stars.scale(height / 2)
        stars.set_color(WHITE)
        stars.set_glow_factor(2)
        stars.set_radii(np.random.uniform(0, 0.075, n_stars))

        self.add(cube)
        self.add(stars)

        self.play(ShowCreation(stars, run_time=4))

        # Add randy
        randy = Randolph(height=1)
        randy.next_to(cube, LEFT, buff=1)

        self.play(
            VFadeIn(randy),
            randy.change("pondering", look_at=RIGHT)
        )

        if self.show_pi_perspective:
            self.play(
                frame.animate.reorient(-89, -4, 0, (0.01, 0.21, 0.0), 3.05),
                randy.animate.set_opacity(0),
                cube.animate.set_stroke(width=5).set_anti_alias_width(10),
                run_time=3,
            )
            frame.always.match_z(randy)
        else:
            self.play(frame.animate.reorient(-40, -26, 0), run_time=3)

        # Move up and down
        for dy in [1, -2, 2, -2, 1]:
            self.play(randy.animate.shift(dy * 1.5 * IN), run_time=5)

        return

        # Show some triangle
        star_points = np.array(list(filter(lambda p: get_norm(p) < 1, stars.get_points())))
        star_points[0] *= 10
        random.seed(1)
        verts = random.sample(list(star_points), 3)
        triangle = Polygon(*verts)
        triangle.set_color(RED)
        red_stars = DotCloud(verts)
        red_stars.set_radius(0.02)
        red_stars.set_color(RED)

        self.play(
            ShowCreation(triangle),
            FadeIn(red_stars),
        )


class SimpleDotsFromPerspective(SimpleDotsParalax):
    show_pi_perspective = True


class ParalxInSolarSystem(InteractiveScene):
    show_celestial_sphere = False

    def construct(self):
        # Add sun and earth
        frame = self.frame
        light_source = self.camera.light_source

        earth = get_earth(radius=0.01)
        earth.rotate(EARTH_TILT_ANGLE, UP)
        earth_axis = rotate_vector(OUT, EARTH_TILT_ANGLE, UP)
        sun = get_sun(radius=0.07, big_glow_ratio=10)

        orbit_radius = 5
        sun.move_to(ORIGIN)
        light_source.move_to(sun)

        orbit = Circle(radius=orbit_radius, n_components=100)
        orbit.set_stroke(BLUE, width=(0, 3))
        orbit.rotate(-30 * DEG)
        orbit.add_updater(lambda m, dt: m.rotate(10 * dt * DEG))
        orbit.set_stroke(flat=False)
        orbit.set_anti_alias_width(5)
        orbit.apply_depth_test()

        earth.add_updater(lambda m: m.move_to(orbit.get_end()))
        # earth.add_updater(lambda m, dt: m.rotate(2 * TAU * dt, axis=earth_axis))

        self.add(orbit, earth, sun)
        self.wait(3)

        # Add stars
        if self.show_celestial_sphere:
            n_stars = 0
            celestial_sphere = get_celestial_sphere()
            self.add(celestial_sphere)
        else:
            n_stars = 3000
        points = np.random.uniform(-1, 1, (n_stars, 3))
        points = normalize_along_axis(points, 1)
        distances = np.random.uniform(10, 50, n_stars)
        radii = np.random.uniform(0, 0.2, n_stars)
        points = points * distances[:, np.newaxis]
        stars = GlowDots(points)
        stars.set_color(WHITE)
        stars.set_radii(radii)

        self.add(stars, orbit, earth)
        self.play(
            ShowCreation(stars),
            frame.animate.reorient(0, 71, 0, (2.29, -0.66, 0.52), 43.56),
            run_time=4,
        )
        self.wait(3)

        self.play(
            frame.animate.set_euler_angles(-10 * DEG, 86 * DEG, 0).set_height(0.25),
            # frame.animate.set_height(0.1),
            UpdateFromAlphaFunc(sun, lambda m, a: frame.move_to(interpolate(frame.get_center(), earth.get_center(), min(2 * a, 1)))),
            run_time=5
        )
        frame.clear_updaters()
        frame.add_updater(lambda m: m.move_to(earth))

        self.wait(30)

        # Zoom back out
        self.play(
            frame.animate.reorient(7, 74, 0, ORIGIN, 120),
            stars.animate.set_radii(2 * radii),
            run_time=3,
        )
        new_points = 1000 * normalize_along_axis(stars.get_points(), 1)
        self.play(
            stars.animate.set_points(new_points).set_radii(20 * radii),
            run_time=4,
        )
        self.play(
            frame.animate.reorient(-89, 74, 0),
            run_time=12
        )


class ShowConstellationsDuringOrbit(ParalxInSolarSystem):
    show_celestial_sphere = True
