from big_ol_pile_of_manim_imports import *


class SquareWave(Scene):
    CONFIG = {
        "L": FRAME_WIDTH / 4,
    }

    def construct(self):
        kwargs = {
            "x_min": -2 * self.L,
            "x_max": 2 * self.L,
        }
        waves = VGroup(*[
            FunctionGraph(
                lambda x: self.approx_square_wave(x, n_terms),
                **kwargs
            )
            for n_terms in range(1, 5)
        ])
        waves.set_color_by_gradient(BLUE, YELLOW, RED)
        stroke_widths = np.linspace(3, 1, len(waves))
        for wave, stroke_width in zip(waves, stroke_widths):
            wave.set_stroke(width=stroke_width)
        waves.to_edge(UP)
        self.add(waves)

        tex_mob = TexMobject("""
            {4 \\over \\pi}
            \\sum_{n=1,3,5,\\dots} {1 \\over n}\\sin(n\\pi x)
        """)
        tex_mob.scale(1.5)
        tex_mob.next_to(waves, DOWN, MED_LARGE_BUFF)
        self.add(tex_mob)

    def approx_square_wave(self, x, n_terms=3):
        L = self.L
        return 1.0 * sum([
            (1.0 / n) * np.sin(n * PI * x / L)
            for n in range(1, 2 * n_terms + 1, 2)
        ])

