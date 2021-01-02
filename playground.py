#!/usr/bin/env python

from imports_3b1b import *

from _2016.zeta import zeta


class LogarithmicSpiral(Scene):
    def construct(self):
        exp_n_tracker = ValueTracker(1)
        group = VGroup()

        def update_group(gr):
            n = 3 * int(np.exp(exp_n_tracker.get_value()))
            gr.set_submobjects(self.get_group(n))

        group.add_updater(update_group)

        self.add(group)
        self.play(
            exp_n_tracker.set_value, 7,
            # exp_n_tracker.set_value, 4,
            run_time=10,
            rate_func=linear,
        )
        self.wait()

    def get_group(self, n, n_spirals=50):
        n = int(n)
        theta = TAU / n
        R = 10

        lines = VGroup(*[
            VMobject().set_points_as_corners([ORIGIN, R * v])
            for v in compass_directions(n)
        ])
        lines.set_stroke(WHITE, min(1, 25 / n))

        # points = [3 * RIGHT]
        # transform = np.array(rotation_matrix_transpose(90 * DEGREES + theta, OUT))
        # transform *= math.sin(theta)
        points = [RIGHT]
        transform = np.array(rotation_matrix_transpose(90 * DEGREES, OUT))
        transform *= math.tan(theta)

        for x in range(n_spirals * n):
            p = points[-1]
            dp = np.dot(p, transform)
            points.append(p + dp)

        vmob = VMobject()
        vmob.set_points_as_corners(points)

        vmob.scale(math.tan(theta), about_point=ORIGIN)

        vmob.set_stroke(BLUE, clip(1000 / n, 1, 3))

        return VGroup(lines, vmob)


class FakeAreaManipulation(Scene):
    CONFIG = {
        "unit": 0.5
    }

    def construct(self):
        unit = self.unit
        group1, group2 = groups = self.get_diagrams()
        for group in groups:
            group.set_width(10 * unit, stretch=True)
            group.set_height(12 * unit, stretch=True)
            group.move_to(3 * DOWN, DOWN)
            group[2].append_points(3 * [group[2].get_left() + LEFT])
            group[3].append_points(3 * [group[3].get_right() + RIGHT])

        grid = NumberPlane(
            x_range=(-30, 30),
            y_range=(-30, 30),
            faded_line_ratio=0,
        )
        grid.set_stroke(width=1)
        grid.scale(unit)
        grid.shift(3 * DOWN - grid.c2p(0, 0))

        vertex_dots = VGroup(
            Dot(group1.get_top()),
            Dot(group1.get_corner(DR)),
            Dot(group1.get_corner(DL)),
        )

        self.add(grid)
        self.add(group1)
        self.add(vertex_dots)

        # group1.save_state()

        kw = {
            "lag_ratio": 0.1,
            "run_time": 2,
            "rate_func": bezier([0, 0, 1, 1]),
        }
        path_arc_factors = [-1, 1, 0, 0, -1, 1]
        for target in (group2, group1.copy()):
            self.play(group1.space_out_submobjects, 1.2)
            self.play(*[
                Transform(
                    sm1, sm2,
                    path_arc=path_arc_factors[i] * 60 * DEGREES,
                    **kw
                )
                for i, sm1, sm2 in zip(it.count(), group1, target)
            ])
            self.wait(2)

        lines = VGroup(
            Line(group1.get_top(), group1.get_corner(DR)),
            Line(group1.get_top(), group1.get_corner(DL)),
        )
        lines.set_stroke(YELLOW, 2)

        frame = self.camera.frame
        frame.save_state()

        self.play(ShowCreation(lines, lag_ratio=0))
        self.play(
            frame.scale, 0.15,
            frame.move_to, group1[1].get_corner(DR),
            run_time=4,
        )
        self.wait(3)
        self.play(Restore(frame, run_time=2))

        # Another switch
        self.play(*[
            Transform(sm1, sm2, **kw)
            for i, sm1, sm2 in zip(it.count(), group1, group2)
        ])

        # Another zooming
        self.play(
            frame.scale, 0.15,
            frame.move_to, group1[1].get_corner(UL),
            run_time=4,
        )
        self.wait(2)
        self.play(Restore(frame, run_time=4))

        self.embed()

    def get_diagrams(self):
        unit = self.unit

        tri1 = Polygon(2 * LEFT, ORIGIN, 5 * UP)
        tri2 = tri1.copy()
        tri2.flip()
        tri2.next_to(tri1, RIGHT, buff=0)
        tris = VGroup(tri1, tri2)
        tris.scale(unit)
        tris.move_to(3 * UP, UP)
        tris.set_stroke(width=0)
        tris.set_fill(BLUE_D)
        tris[1].set_color(BLUE_C)

        ell = Polygon(
            ORIGIN,
            4 * RIGHT,
            4 * RIGHT + 2 * UP,
            2 * RIGHT + 2 * UP,
            2 * RIGHT + 5 * UP,
            5 * UP,
        )
        ell.scale(unit)
        ells = VGroup(ell, ell.copy().rotate(PI).shift(2 * unit * UP))
        ells.next_to(tris, DOWN, buff=0)

        ells.set_stroke(width=0)
        ells.set_fill(GREY)
        ells[1].set_fill(GREY_BROWN)

        big_tri = Polygon(ORIGIN, 3 * LEFT, 7 * UP)
        big_tri.set_stroke(width=0)
        big_tri.scale(unit)

        big_tri.move_to(ells.get_corner(DL), DR)
        big_tris = VGroup(big_tri, big_tri.copy().rotate(PI, UP, about_point=ORIGIN))

        big_tris[0].set_fill(RED_E, 1)
        big_tris[1].set_fill(RED_C, 1)
        full_group = VGroup(*tris, *ells, *big_tris)
        full_group.set_height(5, about_edge=UP)

        alt_group = full_group.copy()

        alt_group[0].move_to(alt_group, DL)
        alt_group[1].move_to(alt_group, DR)
        alt_group[4].move_to(alt_group[0].get_corner(UR), DL)
        alt_group[5].move_to(alt_group[1].get_corner(UL), DR)
        alt_group[2].rotate(90 * DEGREES)
        alt_group[2].move_to(alt_group[1].get_corner(DL), DR)
        alt_group[2].rotate(-90 * DEGREES)
        alt_group[2].move_to(alt_group[0].get_corner(DR), DL)
        alt_group[3].move_to(alt_group[1].get_corner(DL), DR)

        full_group.set_opacity(0.75)
        alt_group.set_opacity(0.75)

        return full_group, alt_group


class CurrBanner(Banner):
    CONFIG = {
        "camera_config": {
            "pixel_height": 1440,
            "pixel_width": 2560,
        },
        "pi_height": 1.25,
        "pi_bottom": 0.25 * DOWN,
        "use_date": False,
        "date": "Wednesday, March 15th",
        "message_scale_val": 0.9,
        "add_supporter_note": False,
        "pre_date_text": "Next video on ",
    }

    def construct(self):
        super().construct()
        for pi in self.pis:
            pi.set_gloss(0.1)


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


class BigCross(Scene):
    def construct(self):
        rect = FullScreenFadeRectangle()
        big_cross = Cross(rect)
        big_cross.set_stroke(width=30)
        self.add(big_cross)


class Eoc1Thumbnail(GraphScene):
    CONFIG = {

    }

    def construct(self):
        title = TextMobject(
            "The Essence of\\\\Calculus",
            tex_to_color_map={
                "\\emph{you}": YELLOW,
            },
        )
        subtitle = TextMobject("Chapter 1")
        subtitle.match_width(title)
        subtitle.scale(0.75)
        subtitle.next_to(title, DOWN)
        # title.add(subtitle)
        title.set_width(FRAME_WIDTH - 2)
        title.to_edge(UP)
        title.set_stroke(BLACK, 8, background=True)
        # answer = TextMobject("...yes")
        # answer.to_edge(DOWN)

        axes = Axes(
            x_min=-1,
            x_max=5,
            y_min=-1,
            y_max=5,
            y_axis_config={
                "include_tip": False,
            },
            x_axis_config={
                "unit_size": 2,
            },
        )
        axes.set_width(FRAME_WIDTH - 1)
        axes.center().to_edge(DOWN)
        axes.shift(DOWN)
        self.x_axis = axes.x_axis
        self.y_axis = axes.y_axis
        self.axes = axes

        graph = self.get_graph(self.func)
        rects = self.get_riemann_rectangles(
            graph,
            x_min=0, x_max=4,
            dx=0.2,
        )
        rects.set_submobject_colors_by_gradient(BLUE, GREEN)
        rects.set_opacity(1)
        rects.set_stroke(BLACK, 1)

        self.add(axes)
        self.add(graph)
        self.add(rects)
        self.add(title)
        # self.add(answer)

    def func(slef, x):
        return 0.35 * ((x - 2)**3 - 2 * (x - 2) + 6)


class HenryAnimation(Scene):
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
            TexMobject(*args, **config)
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


class PrimePiEPttern(Scene):
    def construct(self):
        self.add(FullScreenFadeRectangle(fill_color=WHITE, fill_opacity=1))

        tex0 = TexMobject(
            "\\frac{1}{1^2}", "+"
            "\\frac{1}{2^2}", "+"
            "\\frac{1}{3^2}", "+"
            "\\frac{1}{4^2}", "+"
            "\\frac{1}{5^2}", "+"
            "\\frac{1}{6^2}", "+"
            # "\\frac{1}{7^2}", "+"
            # "\\frac{1}{8^2}", "+"
            # "\\frac{1}{9^2}", "+"
            # "\\frac{1}{10^2}", "+"
            # "\\frac{1}{11^2}", "+"
            # "\\frac{1}{12^2}", "+"
            "\\cdots",
            "=",
            "\\frac{\\pi^2}{6}",
        )
        self.alter_tex(tex0)
        # self.add(tex0)

        tex1 = TexMobject(
            "\\underbrace{\\frac{1}{1^2}}_{\\text{kill}}", "+",
            "\\underbrace{\\frac{1}{2^2}}_{\\text{keep}}", "+",
            "\\underbrace{\\frac{1}{3^2}}_{\\text{keep}}", "+",
            "\\underbrace{\\frac{1}{4^2}}_{\\times (1 / 2)}", "+",
            "\\underbrace{\\frac{1}{5^2}}_{\\text{keep}}", "+",
            "\\underbrace{\\frac{1}{6^2}}_{\\text{kill}}", "+",
            "\\underbrace{\\frac{1}{7^2}}_{\\text{keep}}", "+",
            "\\underbrace{\\frac{1}{8^2}}_{\\times (1 / 3)}", "+",
            "\\underbrace{\\frac{1}{9^2}}_{\\times (1 / 2)}", "+",
            "\\underbrace{\\frac{1}{10^2}}_{\\text{kill}}", "+",
            "\\underbrace{\\frac{1}{11^2}}_{\\text{keep}}", "+",
            "\\underbrace{\\frac{1}{12^2}}_{\\text{kill}}", "+",
            "\\cdots",
            # "=",
            # "\\frac{\\pi^2}{6}"
        )
        self.alter_tex(tex1)
        tex1.set_color_by_tex("kill", RED)
        tex1.set_color_by_tex("keep", GREEN_E)
        tex1.set_color_by_tex("times", BLUE_D)

        self.add(tex1)
        return

        # tex1 = TexMobject(
        #     "\\underbrace{\\frac{1}{1}}_{\\text{kill}}", "+",
        #     "\\underbrace{\\frac{-1}{3}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{1}{5}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{-1}{7}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{1}{9}}_{\\times (1 / 2)}", "+",
        #     "\\underbrace{\\frac{-1}{11}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{1}{13}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{-1}{15}}_{\\text{kill}}", "+",
        #     "\\underbrace{\\frac{1}{17}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{-1}{19}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{1}{21}}_{\\text{kill}}", "+",
        #     "\\underbrace{\\frac{-1}{23}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{1}{25}}_{\\times (1 / 2)}", "+",
        #     "\\underbrace{\\frac{-1}{27}}_{\\times (1 / 3)}", "+",
        #     "\\cdots",
        #     "=",
        #     "\\frac{\\pi}{4}"
        # )
        # self.alter_tex(tex1)
        # VGroup(
        #     tex1[2 * 0],
        #     tex1[2 * 7],
        #     tex1[2 * 10],
        # ).set_color(RED)
        # VGroup(
        #     tex1[2 * 1],
        #     tex1[2 * 2],
        #     tex1[2 * 3],
        #     tex1[2 * 5],
        #     tex1[2 * 6],
        #     tex1[2 * 8],
        #     tex1[2 * 9],
        #     tex1[2 * 11],
        # ).set_color(GREEN_E)
        # VGroup(
        #     tex1[2 * 4],
        #     tex1[2 * 12],
        #     tex1[2 * 13],
        # ).set_color(BLUE_D)

        # self.add(tex1)

        # tex2 = TexMobject(
        #     "\\frac{-1}{3}", "+",
        #     "\\frac{1}{5}", "+",
        #     "\\frac{-1}{7}", "+",
        #     "\\frac{1}{2}", "\\cdot", "\\frac{1}{9}", "+",
        #     "\\frac{-1}{11}", "+",
        #     "\\frac{1}{13}", "+",
        #     "\\frac{1}{17}", "+",
        #     "\\frac{-1}{19}", "+",
        #     "\\frac{-1}{23}", "+",
        #     "\\frac{1}{2}", "\\cdot", "\\frac{1}{25}", "+",
        #     "\\frac{1}{3}", "\\cdot", "\\frac{-1}{27}", "+",
        #     "\\cdots",
        # )
        # self.alter_tex(tex2)
        # VGroup(
        #     tex2[2 * 0],
        #     tex2[2 * 1],
        #     tex2[2 * 2],
        #     tex2[2 * 5],
        #     tex2[2 * 6],
        #     tex2[2 * 7],
        #     tex2[2 * 8],
        #     tex2[2 * 9],
        # ).set_color(GREEN_E)
        # VGroup(
        #     tex2[2 * 3],
        #     tex2[2 * 4],
        #     tex2[2 * 10],
        #     tex2[2 * 11],
        #     tex2[2 * 12],
        #     tex2[2 * 13],
        # ).set_color(BLUE_D)

        tex2 = TexMobject(
            "\\frac{1}{2^2}", "+",
            "\\frac{1}{3^2}", "+",
            "\\frac{1}{2}", "\\cdot", "\\frac{1}{4^2}", "+",
            "\\frac{1}{5^2}", "+",
            "\\frac{1}{7^2}", "+",
            "\\frac{1}{3}", "\\cdot", "\\frac{1}{8^2}", "+",
            "\\frac{1}{2}", "\\cdot", "\\frac{1}{9^2}", "+",
            "\\frac{1}{11^2}", "+",
            "\\frac{1}{13^2}", "+",
            "\\frac{1}{4}", "\\cdot", "\\frac{1}{16^2}", "+",
            "\\cdots",
        )
        self.alter_tex(tex2)
        VGroup(
            tex2[2 * 0],
            tex2[2 * 1],
            tex2[2 * 4],
            tex2[2 * 5],
            tex2[2 * 10],
            tex2[2 * 11],
        ).set_color(GREEN_E)
        VGroup(
            tex2[2 * 2],
            tex2[2 * 3],
            tex2[2 * 6],
            tex2[2 * 7],
            tex2[2 * 8],
            tex2[2 * 9],
            tex2[2 * 12],
            tex2[2 * 13],
        ).set_color(BLUE_D)
        self.add(tex2)

        exp = TexMobject(
            "e^{\\left(",
            "0" * 30,
            "\\right)}",
            "= \\frac{\\pi^2}{6}"
        )
        self.alter_tex(exp)
        exp[1].set_opacity(0)
        tex2.replace(exp[1], dim_to_match=0)

        self.add(exp, tex2)

    def alter_tex(self, tex):
        tex.set_color(BLACK)
        tex.set_stroke(BLACK, 0, background=True)
        tex.set_width(FRAME_WIDTH - 1)


class NewMugThumbnail(Scene):
    def construct(self):
        title = TexMobject(
            "V - E + F = 0",
            tex_to_color_map={"0": YELLOW},
        )
        title.scale(3)
        title.to_edge(UP)
        image = ImageMobject("sci_youtubers_thumbnail")
        image.set_height(5.5)
        image.next_to(title, DOWN)
        self.add(title, image)


class Vertical3B1B(Scene):
    def construct(self):
        words = TextMobject(
            "3", "Blue", "1", "Brown",
        )
        words.scale(2)
        words[::2].scale(1.2)
        buff = 0.2
        words.arrange(
            DOWN,
            buff=buff,
            aligned_edge=LEFT,
        )
        words[0].match_x(words[1][0])
        words[2].match_x(words[3][0])
        self.add(words)

        logo = Logo()
        logo.next_to(words, LEFT)
        self.add(logo)

        VGroup(logo, words).center()


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

        label = TexMobject(
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


class TenDThumbnail(Scene):
    def construct(self):
        square = Square()
        square.set_height(3.5)
        square.set_stroke(YELLOW, 5)
        r = square.get_width() / 2
        circles = VGroup(*[
            Circle(radius=r).move_to(corner)
            for corner in square.get_vertices()
        ])
        circles.set_stroke(BLUE, 5)
        circles.set_fill(BLUE, 0.5)
        circles.set_sheen(0.5, UL)
        lil_circle = Circle(
            radius=(np.sqrt(2) - 1) * r
        )
        lil_circle.set_stroke(YELLOW, 3)
        lil_circle.set_fill(YELLOW, 0.5)

        group = VGroup(circles, lil_circle, square)
        group.to_edge(LEFT)
        square.scale(2)

        words = TextMobject(
            "What\\\\"
            "about\\\\"
            "in 10D?\\\\"
            # "dimensions?"
        )
        words.set_height(5)
        words.to_edge(RIGHT)

        arrow = Arrow(
            words[0][0].get_left(),
            lil_circle.get_center(),
            path_arc=90 * DEGREES,
            buff=0.5,
        )
        arrow.set_color(RED)
        arrow.set_stroke(width=12)
        arrow_group = VGroup(
            arrow.copy().set_stroke(BLACK, 16),
            arrow,
        )

        self.add(group)
        self.add(words)
        self.add(arrow_group)


class WhyPi(Scene):
    def construct(self):
        title = TextMobject("Why $\\pi$?")
        title.scale(3)
        title.to_edge(UP)

        formula1 = TexMobject(
            "1 +"
            "\\frac{1}{4} +"
            "\\frac{1}{9} +"
            "\\frac{1}{16} +"
            "\\frac{1}{25} + \\cdots"
            "=\\frac{\\pi^2}{6}"
        )
        formula1.set_color(YELLOW)
        formula1.set_width(FRAME_WIDTH - 2)
        formula1.next_to(title, DOWN, MED_LARGE_BUFF)

        formula2 = TexMobject(
            "1 -"
            "\\frac{1}{3} +"
            "\\frac{1}{5} -"
            "\\frac{1}{7} +"
            "\\frac{1}{9} - \\cdots"
            "=\\frac{\\pi}{4}"
        )
        formula2.set_color(BLUE_C)
        formula2.set_width(FRAME_WIDTH - 2)
        formula2.next_to(formula1, DOWN, LARGE_BUFF)

        self.add(title)
        self.add(formula1)
        self.add(formula2)


class GeneralExpositionIcon(Scene):
    def construct(self):
        title = TextMobject("What is \\underline{\\qquad \\qquad}?")
        title.scale(3)
        title.to_edge(UP)
        randy = Randolph()
        randy.change("pondering")
        randy.set_height(4.5)
        randy.to_edge(DOWN)
        randy.look_at(title[0][0])

        self.add(title)
        self.add(randy)


class GeometryIcon(Scene):
    def construct(self):
        im = ImageMobject("/Users/grant/Desktop/maxresdefault (9).jpg")
        im.set_height(FRAME_HEIGHT)
        im.scale(0.9, about_edge=DOWN)
        word = TextMobject("Geometry")
        word.scale(3)
        word.to_edge(UP)
        self.add(im, word)


class PhysicsIcon(Scene):
    def construct(self):
        im = ImageMobject("/Users/grant/Desktop/maxresdefault (10).png")
        im.set_height(FRAME_HEIGHT)
        im.shift(UP)
        title = TextMobject("Physics")
        title.scale(3)
        title.to_edge(UP)

        self.add(im)
        self.add(title)


class SupportIcon(Scene):
    def construct(self):
        randy = Randolph(mode="coin_flip_2")
        morty = Mortimer(mode="gracious")
        pis = VGroup(randy, morty)
        pis.arrange(RIGHT, buff=3)
        pis.to_edge(DOWN)
        randy.make_eye_contact(morty)
        heart = SuitSymbol("hearts")
        heart.set_height(1)
        heart.next_to(randy, UR, buff=-0.5)
        heart.shift(0.5 * RIGHT)

        # rect = FullScreenFadeRectangle(opacity=0.85)

        # self.add(rect)
        self.add(pis)
        self.add(heart)


class SupportPitch1(Scene):
    CONFIG = {
        "camera_config": {
            "background_opacity": 0.85,
        },
        "mode1": "happy",
        "mode2": "hooray",
        "words1": "So what do\\\\you do?",
        "words2": "Oh, I make\\\\videos about\\\\math.",
    }

    def construct(self):
        randy = Randolph()
        randy.to_corner(DL)
        morty = Mortimer()
        morty.to_corner(DR)

        randy.change(self.mode1, morty.eyes)
        morty.change(self.mode2, randy.eyes)

        b1 = randy.get_bubble(
            self.words1,
            bubble_class=SpeechBubble,
            height=3,
            width=4,
        )
        b1.add(b1.content)
        b1.shift(0.25 * UP)
        b2 = morty.get_bubble(
            self.words2,
            bubble_class=SpeechBubble,
            height=3,
            width=4,
        )
        # b2.content.scale(0.9)
        b2.add(b2.content)
        b2.shift(0.25 * DOWN)

        self.add(randy)
        self.add(morty)
        self.add(b2)
        self.add(b1)


class SupportPitch2(SupportPitch1):
    CONFIG = {
        "mode1": "confused",
        "mode2": "speaking",
        "words1": "Wait, how does\\\\that work?",
        "words2": "People pay\\\\for them.",
    }


class SupportPitch3(SupportPitch1):
    CONFIG = {
        "mode1": "hesitant",
        "mode2": "coin_flip_2",
        "words1": "Oh, so like\\\\a paid course?",
        "words2": "Well, no,\\\\everything\\\\is free.",
    }


class SupportPitch4(SupportPitch1):
    CONFIG = {
        "mode1": "confused",
        "mode2": "hesitant",
        "words1": "Wait, what?",
        "words2": "I know,\\\\it's weird...",
    }


class RantPage(Scene):
    CONFIG = {
    }

    def construct(self):
        squares = VGroup(Square(), Square())
        squares.arrange(DOWN, buff=MED_SMALL_BUFF)
        squares.set_height(FRAME_HEIGHT - 0.5)
        squares.set_width(5, stretch=True)
        squares.set_stroke(WHITE, 2)
        squares.set_fill(BLACK, opacity=0.75)
        s1, s2 = squares

        # Group1
        morty = Mortimer(mode="maybe")
        for eye, pupil in zip(morty.eyes, morty.pupils):
            pupil.move_to(eye)
        morty.shift(MED_SMALL_BUFF * UL)
        words = TextMobject(
            "What were you\\\\expecting to be here?"
        )
        bubble = SpeechBubble(direction=RIGHT)
        bubble.match_style(s1)
        bubble.add_content(words)
        bubble.resize_to_content()
        bubble.add(bubble.content)
        bubble.pin_to(morty)
        group1 = VGroup(morty, bubble)
        group1.set_height(s1.get_height() - MED_SMALL_BUFF)
        group1.next_to(s1.get_corner(DR), UL, SMALL_BUFF)

        # Group 2
        morty = Mortimer(mode="surprised")
        morty.shift(MED_SMALL_BUFF * UL)
        words = TextMobject(
            "Go on!\\\\Give the rant!"
        )
        bubble = SpeechBubble(direction=RIGHT)
        bubble.match_style(s1)
        bubble.add_content(words)
        bubble.resize_to_content()
        bubble.add(bubble.content)
        bubble.pin_to(morty)
        group2 = VGroup(morty, bubble)
        group2.set_height(s2.get_height() - MED_SMALL_BUFF)
        group2.next_to(s2.get_corner(DR), UL, SMALL_BUFF)

        self.add(squares)
        self.add(group1)
        self.add(group2)


#####

# TODO, this seems useful.  Put it somewhere
def get_count_mobs_on_points(points, height=0.1):
    result = VGroup()
    for n, point in enumerate(points):
        count = Integer(n)
        count.set_height(height)
        count.move_to(point)
        result.add(count)
    return result


def exp(A, max_n=30):
    factorial = 1
    A = np.array(A)
    result = np.zeros(A.shape)
    power = np.identity(A.shape[0])
    for k in range(max_n):
        result = result + (power / factorial)
        power = np.dot(power, A)
        factorial *= (k + 1)
    return result


def get_arrows(squares):
    result = VGroup()
    for square in squares:
        corners = square.get_vertices()
        if len(corners) >= 2:
            p0, p1 = corners[:2]
            angle = angle_of_vector(p1 - p0)
        else:
            angle = 0
        arrow = Vector(RIGHT)
        arrow.rotate(angle)
        arrow.move_to(square.get_center() + RIGHT)
        arrow.match_color(square)
        result.add(arrow)
    return result


class ClipsLogo(Scene):
    def construct(self):
        logo = Logo()
        logo.set_height(FRAME_HEIGHT - 0.5)
        square = Square(stroke_width=0, fill_color=BLACK, fill_opacity=1)
        square.scale(5)
        square.rotate(45 * DEGREES)
        square.move_to(ORIGIN, LEFT)
        self.add(logo, square)


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

        title = TexMobject(
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


class UsualBanner(Banner):
    CONFIG = {
        # "date": "Wednesday, April 3rd",
        # "use_date": True,
    }


class HindiBanner(Banner):
    CONFIG = {
        "message_scale_val": 0.9,
    }

    def get_probabalistic_message(self):
        # return TextMobject("3Blue1Brown")
        result = TextMobject("3Blue1Brown", "XXX")
        result[1].set_opacity(0)
        return result


class HindiLogo(Scene):
    def construct(self):
        logo = Logo()
        logo.set_height(3)
        words = TextMobject("")
        words.set_width(logo.get_width() * 0.9)
        words.move_to(logo)
        words.shift(SMALL_BUFF * UP)
        words.set_stroke(BLACK, 5, background=True)
        self.add(logo, words)


class PatronBanner(Banner):
    CONFIG = {
        "date": "Sunday, December 22nd",
        "use_date": True,
        "add_supporter_note": True,
    }


def vs_func(point):
    x, y, z = point
    return np.array([
        np.cos(y) * x,
        y - x,
        0,
    ])


class ColorTest(Scene):
    CONFIG = {
        "color_group_width": 2,
        "color_group_height": 2,
    }

    def construct(self):
        self.add(TextMobject("Some title").scale(1.5).to_edge(UP))

        blues = self.get_color_group("BLUE")
        greens = self.get_color_group("GREEN")
        browns = self.get_color_group("BROWN")
        teals = self.get_color_group("TEAL")
        pinks = self.get_color_group("PINK")
        reds = self.get_color_group("RED")
        yellows = self.get_color_group("YELLOW")
        greys = self.get_color_group("GREY")

        color_groups = VGroup(
            blues, teals, greens, greys,
            reds, pinks, yellows, browns,
        )
        color_groups.arrange_in_grid(n_rows=2, buff=LARGE_BUFF)
        self.add(color_groups)

        # tone_groups = VGroup(*[
        #     VGroup(*[group[i] for group in color_groups])
        #     for i in range(5)
        # ])
        # for group in tone_groups:
        #     group.arrange(DOWN, buff=0.0)

        # tone_groups.arrange(RIGHT, buff=MED_LARGE_BUFF)

        # for group in color_groups:
        #     self.add_equation(group)

        # self.add(tone_groups)

    def add_equation(self, group):
        eq = TexMobject("e^{\\pi\\sqrt{163}}")
        eq.match_color(group[2])
        eq.next_to(group, DOWN, buff=SMALL_BUFF)
        eq.set_stroke(width=0, background=True)
        self.add(eq)

    def get_color_group(self, name):
        colors = [
            globals().get(name + "_{}".format(c))
            for c in "ABCDE"
        ]
        group = VGroup(*[
            Rectangle(
                stroke_width=0,
                fill_opacity=1,
            ).set_color(color)
            for color in colors
        ])
        group.arrange(DOWN, buff=0.1)
        group.set_width(self.color_group_width, stretch=True)
        group.set_height(self.color_group_height, stretch=True)
        return group
