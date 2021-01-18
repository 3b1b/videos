from manim_imports_ext import *


class PowersOfTwo(MovingCameraScene):
    def construct(self):
        R = 3
        circle = Circle(radius=R)
        circle.set_stroke(BLUE, 2)
        n = 101
        dots = VGroup()
        numbers = VGroup()
        points = [
            rotate_vector(R * DOWN, k * TAU / n)
            for k in range(n)
        ]
        dots = VGroup(*[
            Dot(point, radius=0.03, color=YELLOW)
            for point in points
        ])
        numbers = VGroup(*[
            Integer(k).scale(0.2).move_to((1.03) * point)
            for k, point in enumerate(points)
        ])
        lines = VGroup(*[
            Line(points[k], points[(2 * k) % n])
            for k in range(n)
        ])
        lines.set_stroke(RED, 2)
        arrows = VGroup(*[
            Arrow(
                n1.get_bottom(), n2.get_bottom(),
                path_arc=90 * DEGREES,
                buff=0.02,
                max_tip_length_to_length_ratio=0.1,
            )
            for n1, n2 in zip(numbers, numbers[::2])
        ])
        transforms = [
            Transform(
                numbers[k].copy(), numbers[(2 * k) % n].copy(),
            )
            for k in range(n)
        ]

        title = Tex(
            "\\mathds{Z} / (101 \\mathds{Z})"
        )
        title.scale(2)

        frame = self.camera_frame
        frame.save_state()

        self.add(circle, title)
        self.play(
            LaggedStart(*map(FadeInFromLarge, dots)),
            LaggedStart(*[
                FadeIn(n, -normalize(n.get_center()))
                for n in numbers
            ]),
            run_time=2,
        )
        self.play(
            frame.scale, 0.25,
            {"about_point": circle.get_bottom() + SMALL_BUFF * DOWN}
        )
        n_examples = 6
        for k in range(1, n_examples):
            self.play(
                ShowCreation(lines[k]),
                transforms[k],
                ShowCreation(arrows[k])
            )
        self.play(
            frame.restore,
            FadeOut(arrows[:n_examples])
        )
        self.play(
            LaggedStart(*map(ShowCreation, lines[n_examples:])),
            LaggedStart(*transforms[n_examples:]),
            FadeOut(title, rate_func=squish_rate_func(smooth, 0, 0.5)),
            run_time=10,
            lag_ratio=0.01,
        )
        self.play(
            LaggedStart(*[
                ShowCreationThenFadeOut(line.copy().set_stroke(PINK, 3))
                for line in lines
            ]),
            run_time=3
        )
        self.wait(4)


class Cardiod(Scene):
    def construct(self):
        r = 1
        big_circle = Circle(color=BLUE, radius=r)
        big_circle.set_stroke(width=1)
        big_circle.rotate(-PI / 2)
        time_tracker = ValueTracker()
        get_time = time_tracker.get_value

        def get_lil_circle():
            lil_circle = big_circle.copy()
            lil_circle.set_color(YELLOW)
            time = get_time()
            lil_circle.rotate(time)
            angle = 0.5 * time
            lil_circle.shift(
                rotate_vector(UP, angle) * 2 * r
            )
            return lil_circle
        lil_circle = always_redraw(get_lil_circle)

        cardiod = ParametricCurve(
            lambda t: op.add(
                rotate_vector(UP, 0.5 * t) * (2 * r),
                -rotate_vector(UP, 1.0 * t) * r,
            ),
            t_min=0,
            t_max=(2 * TAU),
        )
        cardiod.set_color(MAROON_B)

        dot = Dot(color=RED)
        dot.add_updater(lambda m: m.move_to(lil_circle.get_start()))

        self.add(big_circle, lil_circle, dot, cardiod)
        for color in [RED, PINK, MAROON_B]:
            self.play(
                ShowCreation(cardiod.copy().set_color(color)),
                time_tracker.increment_value, TAU * 2,
                rate_func=linear,
                run_time=6,
            )
