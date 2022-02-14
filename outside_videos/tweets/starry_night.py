from manim_imports_ext import *


def stereo_project_point(point, axis=0, r=1, max_norm=10000):
    point = fdiv(point * r, point[axis] + r)
    point[axis] = 0
    norm = get_norm(point)
    if norm > max_norm:
        point *= max_norm / norm
    return point


class StarryStarryNight(Scene):
    def construct(self):
        n_points = int(1e4)
        dots = DotCloud(np.random.random((n_points, 3)))
        dots.set_width(FRAME_WIDTH)
        dots.set_height(FRAME_WIDTH, stretch=True)
        dots.set_depth(FRAME_HEIGHT, stretch=True)
        dots.set_radius(0.25).set_glow_factor(5)
        dots.set_color(WHITE)
        dots.set_radii(
            0.2 * np.random.random(dots.get_num_points())**3
        )

        frame = self.camera.frame
        frame.add_updater(lambda m, dt: m.increment_theta(0.02 * dt))
        frame.reorient(-20, 80)
        self.add(frame)
        self.add(dots)

        self.play(dots.animate.apply_function(lambda p: 3 * normalize(p)), run_time=7)
        self.wait()
        self.play(dots.animate.apply_function(lambda p: stereo_project_point(p, r=3, axis=2)), run_time=7)
        self.wait()
        self.play(
            dots.animate.apply_complex_function(lambda z: z**2 / 5),
            run_time=7
        )
        self.play(Rotate(dots, PI / 2, UP, about_point=ORIGIN), run_time=5)
        self.play(
            dots.animate.apply_complex_function(np.exp),
            run_time=7
        )
        self.play(dots.animate.apply_function(lambda p: 3 * normalize(p)), run_time=10)
        self.wait(5)

        dots.generate_target()
        random.shuffle(dots.target.get_points())
        dots.target.apply_function(lambda p: (10 + 100 * random.random()) * normalize(p))
        self.play(MoveToTarget(dots, run_time=20))

        self.embed()
