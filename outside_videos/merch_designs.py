from manim_imports_ext import *
from manimlib.once_useful_constructs.fractals import *


# Fractal posters
class ShowHilbertCurve(Scene):
    CONFIG = {
        "FractalClass": HilbertCurve,
        "orders": [3, 5, 7],
        "stroke_widths": [20, 15, 7],
    }

    def construct(self):
        curves = VGroup(*[
            self.FractalClass(
                order=order,
            ).scale(scale_factor)
            for order, scale_factor in zip(
                self.orders,
                np.linspace(1, 2, 3)
            )
        ])
        for curve, stroke_width in zip(curves, self.stroke_widths):
            curve.set_stroke(width=stroke_width)
        curves.arrange(DOWN, buff=LARGE_BUFF)
        curves.set_height(FRAME_HEIGHT - 1)
        self.add(*curves)


class ShowFlowSnake(ShowHilbertCurve):
    CONFIG = {
        "FractalClass": FlowSnake,
        "orders": [2, 3, 4],
        "stroke_widths": [20, 15, 10],
    }


class FlippedSierpinski(Sierpinski):
    def __init__(self, *args, **kwargs):
        Sierpinski.__init__(self, *args, **kwargs)
        self.rotate(np.pi, RIGHT, about_point=ORIGIN)


class ShowSierpinski(ShowHilbertCurve):
    CONFIG = {
        "FractalClass": FlippedSierpinski,
        "orders": [3, 6, 9],
        "stroke_widths": [20, 15, 6],
    }


# Socks
class SquareWave(Scene):
    def construct(self):
        L = FRAME_WIDTH / 4
        waves = VGroup(*[
            FunctionGraph(
                lambda x: self.approx_square_wave(x, L, n_terms),
                x_range=[-2 * L, 2 * L, L / 50],
            )
            for n_terms in range(1, 5)
        ])
        waves.set_color_by_gradient(BLUE, YELLOW, RED)
        stroke_widths = np.linspace(3, 1, len(waves))
        for wave, stroke_width in zip(waves, stroke_widths):
            # wave.set_stroke(width=stroke_width)
            wave.set_stroke(width=3)
        # waves.to_edge(UP)

        tex_mob = Tex("""
            {4 \\over \\pi}
            \\sum_{n=0}^\\infty {\\sin\\big((2n + 1)\\pi x\\big) \\over (2n + 1)}
        """)
        tex_mob.scale(1.5)
        # tex_mob.next_to(waves, DOWN, MED_LARGE_BUFF)

        self.add(waves)
        # self.add(tex_mob)

    def approx_square_wave(self, x, L, n_terms=3):
        return 1.0 * sum([
            (1.0 / n) * np.sin(n * PI * x / L)
            for n in range(1, 2 * n_terms + 1, 2)
        ])


class PendulumPhaseSpace(Scene):
    def construct(self):
        axes = Axes(
            x_min=-PI,
            x_max=PI,
            y_min=-2,
            y_max=2,
            x_axis_config={
                "tick_frequency": PI / 4,
                "unit_size": FRAME_WIDTH / TAU,
                "include_tip": False,
            },
            y_axis_config={
                "unit_size": 4,
                "tick_frequency": 0.25,
            },
        )

        def func(point, mu=0.1, k=0.5):
            theta, theta_dot = axes.p2c(point)
            return axes.c2p(
                theta_dot,
                -mu * theta_dot - k * np.sin(theta),
            )

        field = VectorField(
            func,
            delta_x=1.5 * FRAME_WIDTH / 12,
            delta_y=1.5,
            y_min=-6,
            y_max=6,
            length_func=lambda norm: 1.25 * sigmoid(norm),
            max_magnitude=4,
            vector_config={
                "tip_length": 0.75,
                "max_tip_length_to_length_ratio": 0.35,
            },
        )
        field.set_stroke(width=12)
        colors = list(Color(BLUE).range_to(RED, 5))
        for vect in field:
            mag = get_norm(1.18 * func(vect.get_start()))
            vect.set_color(colors[min(int(mag), len(colors) - 1)])

        line = VMobject()
        line.start_new_path(axes.c2p(
            -3 * TAU / 4,
            1.75,
        ))

        dt = 0.1
        t = 0
        total_time = 60

        while t < total_time:
            t += dt
            last_point = line.get_last_point()
            new_point = last_point + dt * func(last_point)
            if new_point[0] > FRAME_WIDTH / 2:
                new_point = last_point + FRAME_WIDTH * LEFT
                line.start_new_path(new_point)
            else:
                line.add_smooth_curve_to(new_point)

        line.set_stroke(WHITE, 6)

        # self.add(axes)
        self.add(field)
        # line.set_stroke(BLACK)
        # self.add(line)
