from manim_imports_ext import *


class PodcastIntro(Scene):
    def construct(self):
        tower = self.get_radio_tower()

        n_rings = 15
        min_radius = 0.5
        max_radius = 9
        max_width = 20
        min_width = 0
        max_opacity = 1
        min_opacity = 0
        rings = VGroup(*(
            self.get_circle(radius=r)
            for r in np.linspace(min_radius, max_radius, n_rings)
        ))
        tuples = zip(
            rings,
            np.linspace(max_width, min_width, n_rings),
            np.linspace(max_opacity, min_opacity, n_rings),
        )
        for ring, width, opacity in tuples:
            ring.set_stroke(width=width, opacity=opacity)
            ring.save_state()
            ring.scale(0)
            ring.set_stroke(WHITE, width=2)

        self.play(
            ShowCreation(tower[0], lag_ratio=0.1),
            run_time=3
        )
        self.play(
            FadeIn(tower[1], scale=10, run_time=1),
            LaggedStart(
                *(
                    Restore(ring, rate_func=linear)
                    for ring in reversed(rings)
                ),
                run_time=4,
                lag_ratio=0.08
            )
        )

    def get_radio_tower(self):
        base = VGroup()
        line1 = Line(DL, UP)
        line2 = Line(DR, UP)
        base.add(line1, line2)
        base.set_width(2, stretch=True)
        base.set_height(4, stretch=True)
        base.to_edge(DOWN, buff=1.5)
        # alphas = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85]
        values = np.array([0, *(1 / n for n in range(1, 11))])
        alphas = np.cumsum(values)
        alphas /= alphas[-1]
        for a1, a2 in zip(alphas, alphas[1:]):
            base.add(
                Line(line1.pfp(a1), line2.pfp(a2)),
                Line(line2.pfp(a1), line1.pfp(a2)),
            )
        base.set_stroke(GREY_A, width=2)
        VGroup(line1, line2).set_stroke(width=4)

        dot = Dot(line1.get_end(), radius=0.125)
        dot.set_color(WHITE)
        dot.set_gloss(0.5)
        tower = VGroup(base, dot)
        tower.set_height(3)
        tower.shift(-line1.get_end())
        tower.set_stroke(background=True)

        return tower

    def get_circle(self, center=ORIGIN, radius=1):
        arc1 = Arc(PI, 3 * PI / 2)
        arc2 = Arc(PI / 2, PI / 2)
        arc1.set_color(BLUE)
        arc2.set_color(GREY_BROWN)
        circle = VGroup(arc1, arc2)
        circle.set_width(2 * radius)
        circle.move_to(center)
        return circle
