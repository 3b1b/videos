from manim_imports_ext import *


class RandomWalks(InteractiveScene):
    def construct(self):
        # Add reward landscape
        start_dot = Dot()

        reward_regions = VGroup(
            self.get_reward_region(6 * RIGHT, label="+10"),
            self.get_reward_region(2 * RIGHT + 2 * UP),
            self.get_reward_region(2 * RIGHT + 2 * DOWN),
        )

        self.add(start_dot)
        self.add(reward_regions)

        # Test random walk
        def uniform_angle():
            return random.uniform(0, TAU)

        def biased_angle(mu=0, sigma=1):
            return random.normalvariate(mu, sigma)

        def end_condition(point):
            return any(
                get_norm(point - circ.get_center()) < circ.get_radius()
                for circ in reward_regions
            )

        n_paths = 20
        n_steps = 50
        step_size = 0.5
        walk_func = biased_angle
        show_one_first = False

        paths = VGroup(
            self.get_random_walk(
                walk_func,
                step_size=step_size,
                n_steps=n_steps,
                end_condition=end_condition
            )
            for n in range(n_paths)
        )
        paths.set_stroke(BLUE, 1, 0.5)
        good_paths = VGroup(path for path in paths if end_condition(path.get_end()))
        bad_paths = VGroup(path for path in paths if not end_condition(path.get_end()))

        dots = Group(map(self.get_end_dot, paths))

        if show_one_first:
            self.add(dots[0])
            self.play(ShowCreation(paths[0], run_time=10, rate_func=linear))
            self.wait()
        self.add(dots)
        self.play(LaggedStartMap(ShowCreation, paths, lag_ratio=1 / len(paths), run_time=10))
        self.wait()
        self.play(
            FadeOut(dots),
            good_paths.animate.set_stroke(GREEN, 2),
            bad_paths.animate.set_stroke(WHITE, 1, 0.1)
        )
        self.wait()

    def get_random_walk(
        self,
        angle_func,
        step_size=0.2,
        n_steps=300,
        stroke_color=BLUE,
        stroke_width=1,
        stroke_opacity=1,
        end_condition=None
    ):
        result = VMobject()
        result.start_new_path(ORIGIN)
        for n in range(n_steps):
            step = step_size * rotate_vector(RIGHT, angle_func())
            result.add_line_to(result.get_end() + step)

            # Check end condition
            if (end_condition is not None) and end_condition(result.get_end()):
                break

        result.set_stroke(stroke_color, stroke_width, stroke_opacity)
        return result

    def get_reward_region(
        self,
        location,
        radius=0.5,
        fill_color=GREEN_E,
        fill_opacity=1,
        stroke_color=WHITE,
        stroke_width=1,
        label="+1",
        font_size=36
    ):
        region = Circle(radius=radius)
        region.set_fill(fill_color, fill_opacity)
        region.set_stroke(stroke_color, stroke_width)
        label_mob = Tex(label, font_size=font_size)
        label_mob.set_max_width(radius)
        label_mob.move_to(region)
        region.add(label_mob)
        region.move_to(location)
        return region

    def get_end_dot(self, path):
        dot = GlowDot()
        dot.match_color(path)
        dot.add_updater(lambda m: m.move_to(path.get_end()))
        return dot
