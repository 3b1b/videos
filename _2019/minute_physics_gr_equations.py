from manim_imports_ext import *


class GREquations(Scene):
    CONFIG = {
        "mu_color": BLUE_E,
        "nu_color": RED_E,
        "camera_config": {
            "background_color": WHITE,
        },
        "tex_config": {
            "color": BLACK,
            "background_stroke_width": 0,
        },
    }

    def construct(self):
        eq1 = self.get_field_eq("\\mu", "\\nu")
        indices = list(filter(
            lambda t: t[0] <= t[1],
            it.product(range(4), range(4))
        ))
        sys1, sys2 = [
            VGroup(*[
                self.get_field_eq(i, j, simple=simple)
                for i, j in indices
            ])
            for simple in (True, False)
        ]
        for sys in sys1, sys2:
            sys.arrange(DOWN, buff=MED_LARGE_BUFF)
            sys.set_height(FRAME_HEIGHT - 0.5)
            sys2.center()

        sys1.next_to(ORIGIN, RIGHT)

        eq1.generate_target()
        group = VGroup(eq1.target, sys1)
        group.arrange(RIGHT, buff=2)
        arrows = VGroup(*[
            Arrow(
                eq1.target.get_right(), eq.get_left(),
                buff=0.2,
                color=BLACK,
                stroke_width=2,
                tip_length=0.2,
            )
            for eq in sys1
        ])

        self.play(FadeIn(eq1, DOWN))
        self.wait()
        self.play(
            MoveToTarget(eq1),
            LaggedStart(*[
                GrowArrow(arrow)
                for arrow in arrows
            ]),
        )
        self.play(
            LaggedStart(*[
                TransformFromCopy(eq1, eq)
                for eq in sys1
            ], lag_ratio=0.2),
        )
        self.wait()

        #
        sys1.generate_target()
        sys1.target.to_edge(LEFT)
        sys2.to_edge(RIGHT)
        new_arrows = VGroup(*[
            Arrow(
                e1.get_right(), e2.get_left(),
                buff=SMALL_BUFF,
                color=BLACK,
                stroke_width=2,
                tip_length=0.2,
            )
            for e1, e2 in zip(sys1.target, sys2)
        ])
        self.play(
            MoveToTarget(sys1),
            MaintainPositionRelativeTo(arrows, sys1),
            MaintainPositionRelativeTo(eq1, sys1),
            VFadeOut(arrows),
            VFadeOut(eq1),
        )

        #
        sys1_rects, sys2_rects = [
            VGroup(*map(self.get_rects, sys))
            for sys in [sys1, sys2]
        ]
        self.play(
            LaggedStartMap(FadeIn, sys1_rects),
            LaggedStartMap(GrowArrow, new_arrows),
            run_time=1,
        )
        self.play(
            TransformFromCopy(sys1_rects, sys2_rects),
            TransformFromCopy(sys1, sys2),
        )
        self.play(
            FadeOut(sys1_rects),
            FadeOut(sys2_rects),
        )
        self.wait()

    def get_field_eq(self, mu, nu, simple=True):
        mu = "{" + str(mu) + " }"  # Deliberate space
        nu = "{" + str(nu) + "}"
        config = dict(self.tex_config)
        config["tex_to_color_map"] = {
            mu: self.mu_color,
            nu: self.nu_color,
        }
        if simple:
            tex_args = [
                ("R_{%s%s}" % (mu, nu),),
                ("-{1 \\over 2}",),
                ("g_{%s%s}" % (mu, nu),),
                ("R",),
                ("=",),
                ("8\\pi T_{%s%s}" % (mu, nu),),
            ]
        else:
            tex_args = [
                (
                    "\\left(",
                    "\\partial_\\rho \\Gamma^{\\rho}_{%s%s} -" % (mu, nu),
                    "\\partial_%s \\Gamma^{\\rho}_{\\rho%s} +" % (nu, mu),
                    "\\Gamma^{\\rho}_{\\rho\\lambda}",
                    "\\Gamma^{\\lambda}_{%s%s} -" % (nu, mu),
                    "\\Gamma^{\\rho}_{%s \\lambda}" % nu,
                    "\\Gamma^{\\lambda}_{\\rho %s}" % mu,
                    "\\right)",
                ),
                ("-{1 \\over 2}",),
                ("g_{%s%s}" % (mu, nu),),
                (
                    "g^{\\alpha \\beta}",
                    "\\left(",
                    "\\partial_\\rho \\Gamma^{\\rho}_{\\beta \\alpha} -"
                    "\\partial_\\beta \\Gamma^{\\rho}_{\\rho\\alpha} +",
                    "\\Gamma^{\\rho}_{\\rho\\lambda}",
                    "\\Gamma^{\\lambda}_{\\beta\\alpha} -"
                    "\\Gamma^{\\rho}_{\\beta \\lambda}",
                    "\\Gamma^{\\lambda}_{\\rho \\alpha}",
                    "\\right)",
                ),
                ("=",),
                ("8\\pi T_{%s%s}" % (mu, nu),),
            ]

        result = VGroup(*[
            Tex(*args, **config)
            for args in tex_args
        ])
        result.arrange(RIGHT, buff=SMALL_BUFF)
        return result

    def get_rects(self, equation):
        return VGroup(*[
            SurroundingRectangle(
                equation[i],
                buff=0.025,
                color=color,
                stroke_width=1,
            )
            for i, color in zip(
                [0, 3],
                [GREY, GREY]
            )
        ])
