from manim_imports_ext import *
from _2022.quintic.roots_and_coefs import *


class CubicFormula(RootCoefScene):
    coefs = [1, -1, 0, 1]
    coef_plane_config = {
        "x_range": (-2.0, 2.0),
        "y_range": (-2.0, 2.0),
        "background_line_style": {
            "stroke_color": GREY,
            "stroke_width": 1.0,
        }
    }
    root_plane_config = {
        "x_range": (-2.0, 2.0),
        "y_range": (-2.0, 2.0),
        "background_line_style": {
            "stroke_color": BLUE_E,
            "stroke_width": 1.0,
        }
    }
    sqrt_plane_config = {
        "x_range": (-1.0, 1.0),
        "y_range": (-1.0, 1.0),
        "background_line_style": {
            "stroke_color": GREY_B,
            "stroke_width": 1.0,
        },
        "height": 3.0,
        "width": 3.0,
    }
    crt_plane_config = {
        "x_range": (-2.0, 2.0),
        "y_range": (-2.0, 2.0),
        "background_line_style": {
            "stroke_color": BLUE_E,
            "stroke_width": 1.0,
        },
        "height": 3.0,
        "width": 3.0,
    }
    cf_plane_config = {
        "x_range": (-2.0, 2.0),
        "y_range": (-2.0, 2.0),
        "background_line_style": {
            "stroke_color": BLUE_E,
            "stroke_width": 1.0,
        },
        "height": 3.0,
        "width": 3.0,
    }
    plane_height = 3.0
    plane_buff = 1.0
    planes_center = 1.6 * UP
    lower_planes_height = 2.75
    lower_planes_buff = 2.0

    sqrt_dot_color = GREEN
    crt_dot_colors = (RED, BLUE)
    cf_dot_color = YELLOW

    def add_planes(self):
        super().add_planes()
        self.root_plane_label.next_to(self.root_plane, -self.plane_arrangement)
        self.coef_plane_label.next_to(self.coef_plane, self.plane_arrangement)
        self.add_lower_planes()

    def add_lower_planes(self):
        sqrt_plane = ComplexPlane(**self.sqrt_plane_config)
        crt_plane = ComplexPlane(**self.crt_plane_config)
        cf_plane = ComplexPlane(**self.cf_plane_config)

        planes = VGroup(sqrt_plane, crt_plane, cf_plane)
        for plane in planes:
            plane.add_coordinate_labels(font_size=16)
        planes.set_height(self.lower_planes_height)
        planes.arrange(RIGHT, buff=self.lower_planes_buff)
        planes.to_edge(DOWN, buff=SMALL_BUFF)

        kw = dict(
            font_size=24,
            tex_to_color_map={
                "\\delta_1": GREEN,
                "\\delta_2": GREEN,
            },
            background_stroke_width=3,
            background_stroke_color=3,
        )

        sqrt_label = Tex(
            "\\delta_1, \\delta_2 = \\sqrt{ \\frac{q^2}{4} + \\frac{p^3}{27}}",
            **kw
        )
        sqrt_label.set_backstroke()
        sqrt_label.next_to(sqrt_plane, UP, SMALL_BUFF)

        crt_labels = VGroup(
            Tex("\\cdot", "= \\sqrt[3]{-\\frac{q}{2} + \\delta_1}", **kw),
            Tex("\\cdot", "= \\sqrt[3]{-\\frac{q}{2} + \\delta_2}", **kw),
        )
        for label, color in zip(crt_labels, self.crt_dot_colors):
            label[0].scale(4, about_edge=RIGHT)
            label[0].set_color(color)
            label.set_backstroke()
        crt_labels.arrange(RIGHT, buff=MED_LARGE_BUFF)
        crt_labels.next_to(crt_plane, UP, SMALL_BUFF)

        cf_label = Tex(
            "\\sqrt[3]{ -\\frac{q}{2} + \\delta_1 } +",
            "\\sqrt[3]{ -\\frac{q}{2} + \\delta_2 }",
            # **kw  # TODO, What the hell is going on here...
            font_size=24,
        )
        cf_label.set_backstroke()
        cf_label.next_to(cf_plane, UP, SMALL_BUFF)

        self.add(planes)
        self.add(sqrt_label)
        self.add(crt_labels)
        self.add(cf_label)

        self.sqrt_plane = sqrt_plane
        self.crt_plane = crt_plane
        self.cf_plane = cf_plane
        self.sqrt_label = sqrt_label
        self.crt_labels = crt_labels
        self.cf_label = cf_label

    def get_coef_poly(self):
        return Tex(
            "x^3 + {0}x^2 + {p}x + {q}",
            tex_to_color_map={
                "{0}": self.coef_color,
                "{p}": self.coef_color,
                "{q}": self.coef_color,
            }
        )

    def add_c_labels(self):
        self.c_dot_labels = self.add_dot_labels(
            VGroup(*map(Tex, ["q", "p", "0"])),
            self.coef_dots
        )

    def get_c_symbols(self, coef_poly):
        return VGroup(*(
            coef_poly.get_part_by_tex(tex)
            for tex in ["q", "p", "0"]
        ))

    #
    def add_dots(self):
        super().add_dots()
        self.add_sqrt_dots()
        self.add_crt_dots()
        self.add_cf_dots()

    def add_sqrt_dots(self):
        sqrt_dots = Dot(**self.dot_style).replicate(2)
        sqrt_dots.set_color(self.sqrt_dot_color)

        def update_sqrt_dots(dots):
            q, p, zero, one = self.get_coefs()
            disc = (q**2 / 4) + (p**3 / 27)
            roots = get_nth_roots(disc, 2)
            optimal_transport(dots, map(self.sqrt_plane.n2p, roots))
            return dots

        sqrt_dots.add_updater(update_sqrt_dots)

        self.sqrt_dots = sqrt_dots
        self.add(sqrt_dots)
        self.add(self.get_tracers(sqrt_dots))

        # Labels
        self.delta_labels = self.add_dot_labels(
            VGroup(Tex("\\delta_1"), Tex("\\delta_2")),
            sqrt_dots
        )

    def get_deltas(self):
        return list(map(self.sqrt_plane.p2n, (d.get_center() for d in self.sqrt_dots)))

    def add_crt_dots(self):
        crt_dots = Dot(**self.dot_style).replicate(3).replicate(2)
        for dots, color in zip(crt_dots, self.crt_dot_colors):
            dots.set_color(color)

        def update_crt_dots(dot_triples):
            q, p, zero, one = self.get_coefs()
            deltas = self.get_deltas()

            for delta, triple in zip(deltas, dot_triples):
                roots = get_nth_roots(-q / 2 + delta, 3)
                optimal_transport(triple, map(self.crt_plane.n2p, roots))
            return dot_triples

        crt_dots.add_updater(update_crt_dots)

        self.add(crt_dots)
        self.add(*(self.get_tracers(triple) for triple in crt_dots))

        self.crt_dots = crt_dots

    def get_cube_root_values(self):
        return [
            [
                self.crt_plane.p2n(d.get_center())
                for d in triple
            ]
            for triple in self.crt_dots
        ]

    def add_crt_lines(self):
        crt_lines = VGroup(*(
            Line(stroke_color=color, stroke_width=1).replicate(3)
            for color in self.crt_dot_colors
        ))

        def update_crt_lines(crt_lines):
            cube_root_values = self.get_cube_root_values()
            origin = self.crt_plane.n2p(0)
            for lines, triple in zip(crt_lines, cube_root_values):
                for line, value in zip(lines, triple):
                    line.put_start_and_end_on(origin, self.crt_plane.n2p(value))

        crt_lines.add_updater(update_crt_lines)

        self.add(crt_lines)
        self.crt_lines = crt_lines

    def add_cf_dots(self):
        cf_dots = Dot(**self.dot_style).replicate(9)
        cf_dots.set_fill(self.root_color, opacity=0.5)

        def update_cf_dots(dots):
            cube_root_values = self.get_cube_root_values()
            for dot, (z1, z2) in zip(dots, it.product(*cube_root_values)):
                dot.move_to(self.cf_plane.n2p(z1 + z2))
            return dots

        cf_dots.add_updater(update_cf_dots)

        alt_root_dots = GlowDot()
        alt_root_dots.add_updater(lambda m: m.set_points(
            list(map(self.cf_plane.n2p, self.get_roots()))
        ))

        self.cf_dots = cf_dots
        self.alt_root_dots = alt_root_dots

        self.add(cf_dots)
        self.add(self.get_tracers(cf_dots, stroke_width=0.5))
        self.add(alt_root_dots)

    def add_cf_lines(self):
        cf_lines = VGroup(
            Line(stroke_color=self.crt_dot_colors[0]).replicate(9),
            Line(stroke_color=self.crt_dot_colors[1]).replicate(3),
        )
        cf_lines.set_stroke(width=1)

        def update_cf_lines(cf_lines):
            cube_root_values = self.get_cube_root_values()
            for z1, line in zip(cube_root_values[1], cf_lines[1]):
                line.put_start_and_end_on(
                    self.cf_plane.n2p(0),
                    self.cf_plane.n2p(z1),
                )
            for line, (z1, z2) in zip(cf_lines[0], it.product(*cube_root_values)):
                line.put_start_and_end_on(
                    self.cf_plane.n2p(z2),
                    self.cf_plane.n2p(z1 + z2),
                )

        cf_lines.add_updater(update_cf_lines)

        self.cf_lines = cf_lines
        self.add(cf_lines)


class CubicFormulaTest(CubicFormula):
    def construct(self):
        pass
        # self.embed()
