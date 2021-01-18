from manim_imports_ext import *
from _2016.zeta import zeta


def approx_exp(x, n):
    return sum([
        x**k / math.factorial(k)
        for k in range(n + 1)
    ])


class ExpPlay(Scene):
    def construct(self):
        for n in [2, 4, 6, 8, 10]:
            self.show_sum_up_to(n)
            self.clear()
        self.show_sum_up_to(40, 10, include_dots=True)
        self.wait(2)

    def show_sum_up_to(self, n, shown_n=None, include_dots=False):
        plane = ComplexPlane()
        plane.add_coordinate_labels()
        self.add(plane)
        plane.shift(DOWN)

        if shown_n is None:
            shown_n = n

        t_tracker = ValueTracker(0)
        vectors = always_redraw(
            lambda: self.get_sum_vectors(
                t_tracker.get_value(),
                n,
                center=plane.n2p(0),
            )
        )

        path = ParametricCurve(
            lambda t: plane.n2p(approx_exp(complex(0, t), n)),
            t_min=0,
            t_max=TAU,
        )
        path.set_color(WHITE)

        formula = Tex(
            "e^{it} \\approx",
            *[
                "{(ti)^{%d} \\over %d!} +" % (k, k)
                for k in range(shown_n + 1)
            ],
            "\\cdots" if include_dots else "",
        )
        for vect, part in zip(vectors, formula[1:]):
            part.match_color(vect)
            part[-1].set_color(WHITE)

        if include_dots:
            formula[-1].set_color(WHITE)
        else:
            formula[-1][-1].set_opacity(0)

        if formula.get_width() > FRAME_WIDTH - 1:
            formula.set_width(FRAME_WIDTH - 1)
        formula.to_edge(UP, buff=MED_SMALL_BUFF)
        formula.set_stroke(BLACK, 5, background=True)
        formula.add_background_rectangle(buff=MED_SMALL_BUFF)

        number_line = NumberLine(
            x_min=0,
            x_max=7.5,
        )
        number_line.set_width(5)
        number_line.add_numbers()
        number_line.move_to(1.25 * UP)
        number_line.to_edge(RIGHT)
        rect = BackgroundRectangle(number_line, buff=MED_SMALL_BUFF)
        rect.stretch(2, 1, about_edge=DOWN)
        rect.set_width(7, stretch=True)
        tip = ArrowTip()
        tip.rotate(-PI / 2)
        tip.set_color(WHITE)
        tip.add_updater(
            lambda m: m.move_to(
                number_line.n2p(t_tracker.get_value()),
                DOWN,
            )
        )
        t_eq = VGroup(
            Tex("t="),
            DecimalNumber(0),
        )
        t_eq.add_updater(lambda m: m.arrange(RIGHT, buff=SMALL_BUFF))
        t_eq.add_updater(lambda m: m.next_to(tip, UP, SMALL_BUFF))
        t_eq.add_updater(lambda m: m[1].set_value(t_tracker.get_value()))

        self.add(rect, number_line, tip, t_eq)

        self.add(vectors, path, formula)
        self.play(
            t_tracker.set_value, TAU,
            ShowCreation(path),
            run_time=4,
            rate_func=bezier([0, 0, 1, 1]),
        )
        self.wait()

    def get_sum_vectors(self, t, n, center=ORIGIN):
        vectors = VGroup()
        template_vect = Vector(RIGHT)
        last_tip = center
        for k in range(n + 1):
            vect = template_vect.copy()
            vect.rotate(k * PI / 2)
            vect.scale(t**k / math.factorial(k))
            vect.set_stroke(
                width=max(2, vect.get_stroke_width())
            )
            # vect.tip.set_stroke(width=0)
            vect.shift(last_tip - vect.get_start())
            last_tip = vect.get_end()
            vectors.add(vect)
        self.color_vectors(vectors)
        return vectors

    def color_vectors(self, vectors, colors=[BLUE, YELLOW, RED, PINK]):
        for vect, color in zip(vectors, it.cycle(colors)):
            vect.set_color(color)

    def get_paths(self):
        paths = VGroup(*[
            ParametricCurve(
                lambda t: plane.n2p(approx_exp(complex(0, t), n)),
                t_min=-PI,
                t_max=PI,
            )
            for n in range(10)
        ])
        paths.set_color_by_gradient(BLUE, YELLOW, RED)
        return paths

        # for path in paths:
        #     self.play(ShowCreation(path))
        # self.wait()


class ZetaSum(Scene):
    def construct(self):
        plane = ComplexPlane()
        self.add(plane)
        plane.scale(0.2)

        s = complex(0.5, 14.135)
        N = int(1e6)
        lines = VGroup()
        color = it.cycle([BLUE, RED])
        r = int(1e3)
        for k in range(1, N + 1, r):
            c = sum([
                L**(-s) * (N - L + 1) / N
                for L in range(k, k + r)
            ])
            line = Line(plane.n2p(0), plane.n2p(c))
            line.set_color(next(color))
            if len(lines) > 0:
                line.shift(lines[-1].get_end())
            lines.add(line)

        self.add(lines)
        self.add(
            Dot(lines[-1].get_end(), color=YELLOW),
            Dot(
                center_of_mass([line.get_end() for line in lines]),
                color=YELLOW
            ),
        )


class ZetaSpiral(Scene):
    def construct(self):
        max_t = 50
        spiral = VGroup(*[
            ParametricCurve(
                lambda t: complex_to_R3(
                    zeta(complex(0.5, t))
                ),
                t_min=t1,
                t_max=t2 + 0.1,
            )
            for t1, t2 in zip(it.count(), range(1, max_t))
        ])
        spiral.set_stroke(width=0, background=True)
        # spiral.set_color_by_gradient(BLUE, GREEN, YELLOW, RED)
        # spiral.set_color_by_gradient(BLUE, YELLOW)
        spiral.set_color(YELLOW)

        width = 10
        for piece in spiral:
            piece.set_stroke(width=width)
            width *= 0.98
            dot = Dot()
            dot.scale(0.25)
            dot.match_color(piece)
            dot.set_stroke(BLACK, 1, background=True)
            dot.move_to(piece.get_start())
            # piece.add(dot)

        label = Tex(
            "\\zeta\\left(0.5 + i{t}\\right)",
            tex_to_color_map={"{t}": YELLOW},
            background_stroke_width=0,
        )
        label.scale(1.5)
        label.next_to(spiral, DOWN)

        group = VGroup(spiral, label)
        group.set_height(FRAME_HEIGHT - 1)
        group.center()

        self.add(group)


class SumRotVectors(Scene):
    CONFIG = {
        "n_vects": 100,
    }

    def construct(self):
        plane = ComplexPlane()
        circle = Circle(color=YELLOW)

        t_tracker = ValueTracker(0)
        get_t = t_tracker.get_value

        vects = always_redraw(lambda: self.get_vects(get_t()))

        self.add(plane, circle)
        self.add(t_tracker, vects)
        self.play(
            t_tracker.set_value, 1,
            run_time=10,
            rate_func=linear,
        )
        self.play(ShowIncreasingSubsets(vects, run_time=5))

    def get_vects(self, t):
        vects = VGroup()
        last_tip = ORIGIN
        for n in range(self.n_vects):
            vect = Vector(RIGHT)
            vect.rotate(n * TAU * t, about_point=ORIGIN)
            vect.shift(last_tip)
            last_tip = vect.get_end()
            vects.add(vect)

        vects.set_submobject_colors_by_gradient(BLUE, GREEN, YELLOW, RED)
        vects.set_opacity(0.5)
        return vects


class Spirals(Scene):
    CONFIG = {
        "n_lines": 1200
    }

    def construct(self):
        s_tracker = ComplexValueTracker(complex(2, 1))
        get_s = s_tracker.get_value
        spiral = always_redraw(
            lambda: self.get_spiral(get_s())
        )
        s_dot = always_redraw(
            lambda: Dot(s_tracker.get_center(), color=YELLOW)
        )
        s_label = always_redraw(
            lambda: DecimalNumber(get_s()).to_corner(UR)
        )

        self.add(ComplexPlane().set_stroke(width=0.5))
        self.add(spiral, s_dot, s_label)
        self.play(s_tracker.set_value, complex(1, 1), run_time=3)
        sigma = 0.5
        zero_ts = [
            14.134725,
            21.022040,
            25.010858,
            30.424876,
            32.935062,
            37.586178,
        ]
        self.wait()
        self.play(s_tracker.set_value, complex(sigma, 1), run_time=3)
        self.wait()
        for zero_t in zero_ts:
            self.play(
                s_tracker.set_value, complex(sigma, zero_t),
                run_time=3
            )
            updaters = spiral.get_updaters()
            spiral.clear_updaters()
            self.play(FadeOut(spiral))
            self.play(ShowCreation(spiral, run_time=3))
            for updater in updaters:
                spiral.add_updater(updater)
            self.wait()

    def get_spiral(self, s, colors=[RED, BLUE]):
        n_lines = self.n_lines
        lines = VGroup()
        colors = it.cycle(colors)
        for n in range(1, n_lines + 1):
            z = self.n_to_z(n, s, n_lines)
            if abs(z) == 0:
                continue
            line = Line(ORIGIN, complex_to_R3(z))
            line.set_stroke(
                colors.__next__(),
                width=3 * abs(z)**0.1
            )
            if len(lines) > 0:
                line.shift(lines[-1].get_end())
            lines.add(line)
        return lines

    def n_to_z(self, n, s, n_lines):
        # if is_prime(n):
        #     return -np.log(1 - n**(-s))
        # else:
        #     return 0
        # return n**(-s)
        return (n**(-s)) * (n_lines - n) / n_lines
        # factors = factorize(n)
        # if len(set(factors)) != 1:
        #     return 0
        # else:
        #     return (1.0 / len(factors)) * n**(-s)
