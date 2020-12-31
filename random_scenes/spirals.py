from big_ol_pile_of_manim_imports import *


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