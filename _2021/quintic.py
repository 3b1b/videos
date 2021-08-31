from manim_imports_ext import *


def roots_to_coefficients(roots):
    n = len(list(roots))
    return [
        ((-1)**(n - k)) * sum(
            np.prod(tup)
            for tup in it.combinations(roots, k + 1)
        )
        for k in range(n - 1, -1, -1)
    ] + [1]


def poly(x, coefs):
    return sum(coefs[k] * x**k for k in range(len(coefs)))


def dpoly(x, coefs):
    return sum(k * coefs[k] * x**(k - 1) for k in range(1, len(coefs)))


def find_root(func, dfunc, seed=complex(1, 1), tol=1e-8, max_steps=100):
    # Use newton's method
    last_seed = np.inf
    for n in range(max_steps):
        if abs(seed - last_seed) < tol:
            break
        last_seed = seed
        seed = seed - func(seed) / dfunc(seed)
    return seed


def coefficients_to_roots(coefs):
    # Find a root, divide out by (x - root), repeat
    roots = []
    for i in range(len(coefs) - 1):
        root = find_root(
            lambda x: poly(x, coefs),
            lambda x: dpoly(x, coefs),
        )
        roots.append(root)
        new_reversed_coefs, rem = np.polydiv(coefs[::-1], [1, -root])
        coefs = new_reversed_coefs[::-1]
    return roots


def get_nth_roots(z, n):
    base_root = z**(1 / n)
    return [
        base_root * np.exp(complex(0, k * TAU / n))
        for k in range(n)
    ]


class RootCoefScene(Scene):
    coefs = [3, 2, 1, 0, -1, 1]
    root_plane_config = {
        "x_range": (-2.0, 2.0),
        "y_range": (-2.0, 2.0),
        "background_line_style": {
            "stroke_color": BLUE_E,
        }
    }
    coef_plane_config = {
        "x_range": (-4, 4),
        "y_range": (-4, 4),
        "background_line_style": {
            "stroke_color": GREY,
        }
    }
    plane_height = 5.5
    plane_buff = 1.5
    planes_center = ORIGIN

    root_color = YELLOW
    coef_color = RED_B

    dot_style = {
        "radius": 0.05,
        "stroke_color": BLACK,
        "stroke_width": 3,
        "draw_stroke_behind_fill": True,
    }
    include_tracers = True
    include_labels = True
    label_font_size = 30
    coord_label_font_size = 18

    def setup(self):
        self.add_planes()
        self.add_dots()
        if self.include_tracers:
            self.add_tracers()
        if self.include_labels:
            self.add_r_labels()
            self.add_c_labels()

    def add_planes(self):
        # Planes
        planes = VGroup(
            ComplexPlane(**self.root_plane_config),
            ComplexPlane(**self.coef_plane_config),
        )
        for plane in planes:
            plane.set_height(self.plane_height)
        planes.arrange(RIGHT, buff=self.plane_buff)
        planes.move_to(self.planes_center)

        for plane in planes:
            plane.add_coordinate_labels(font_size=self.coord_label_font_size)
            plane.coordinate_labels.set_opacity(0.8)

        root_plane, coef_plane = planes

        # Lower labels
        root_plane_label = Text("Roots")
        coef_plane_label = Text("Coefficients")

        root_plane_label.next_to(root_plane, DOWN)
        coef_plane_label.next_to(coef_plane, DOWN)

        # Upper labels
        root_poly = self.get_root_poly()
        self.get_r_symbols(root_poly).set_color(self.root_color)
        root_poly.next_to(root_plane, UP)
        root_poly.set_max_width(root_plane.get_width())

        coef_poly = self.get_coef_poly()
        self.get_c_symbols(coef_poly).set_color(self.coef_color)
        coef_poly.set_max_width(coef_plane.get_width())
        coef_poly.next_to(coef_plane, UP)
        coef_poly.match_y(root_poly)

        equals = Tex("=")
        equals.move_to(midpoint(root_poly.get_right(), coef_poly.get_left()))

        self.add(planes)
        self.add(root_plane_label, coef_plane_label)
        self.add(root_poly, coef_poly)
        self.add(equals)

        self.root_plane = root_plane
        self.coef_plane = coef_plane
        self.root_plane_label = root_plane_label
        self.coef_plane_label = coef_plane_label
        self.root_poly = root_poly
        self.coef_poly = coef_poly
        self.poly_equal_sign = equals

    def get_degree(self):
        return len(self.coefs) - 1

    def get_coef_poly(self):
        degree = self.get_degree()
        return Tex(
            f"x^{degree}", *(
                f" + c_{n} x^{n}"
                for n in range(degree - 1, -1, -1)
            )
        )

    def get_root_poly(self):
        return Tex(*(
            f"(x - r_{i})"
            for i in range(self.get_degree())
        ))

    def add_dots(self):
        self.root_dots = VGroup()
        self.coef_dots = VGroup()
        roots = coefficients_to_roots(self.coefs)
        self.add_root_dots(roots)
        self.add_coef_dots(self.coefs)

    #
    def get_all_dots(self):
        return (*self.root_dots, *self.coef_dots)

    def get_r_symbols(self, root_poly):
        return VGroup(*(part[3:5] for part in root_poly))

    def get_c_symbols(self, coef_poly):
        return VGroup(*(part[1:3] for part in coef_poly[:0:-1]))

    def get_random_root(self):
        return complex(
            interpolate(*self.root_plane.x_range[:2], random.random()),
            interpolate(*self.root_plane.y_range[:2], random.random()),
        )

    def get_random_roots(self):
        return [self.get_random_root() for x in range(self.degree)]

    def get_roots_of_unity(self):
        return [np.exp(complex(0, TAU * n / self.degree)) for n in range(self.degree)]

    def set_roots(self, roots):
        self.root_dots.set_submobjects(
            Dot(
                self.root_plane.n2p(root),
                color=self.root_color,
                **self.dot_style,
            )
            for root in roots
        )

    def set_coefs(self, coefs):
        self.coef_dots.set_submobjects(
            Dot(
                self.coef_plane.n2p(coef),
                color=self.coef_color,
                **self.dot_style,
            )
            for coef in coefs[:-1]  # Exclude highest term
        )

    def add_root_dots(self, roots=None):
        if roots is None:
            roots = self.get_roots_of_unity()
        self.set_roots(roots)
        self.add(self.root_dots)

    def add_coef_dots(self, coefs=None):
        if coefs is None:
            coefs = [0] * self.degree + [1]
        self.set_coefs(coefs)
        self.add(self.coef_dots)

    def get_roots(self):
        return [
            self.root_plane.p2n(root_dot.get_center())
            for root_dot in self.root_dots
        ]

    def get_coefs(self):
        return [
            self.coef_plane.p2n(coef_dot.get_center())
            for coef_dot in self.coef_dots
        ] + [1.0]

    def tie_coefs_to_roots(self, clear_updaters=True):
        if clear_updaters:
            self.root_dots.clear_updaters()
            self.coef_dots.clear_updaters()

        def update_coef_dots(cdots):
            coefs = roots_to_coefficients(self.get_roots())
            for dot, coef in zip(cdots, coefs):
                dot.move_to(self.coef_plane.n2p(coef))

        self.coef_dots.add_updater(update_coef_dots)
        self.add(self.coef_dots)

    def tie_roots_to_coefs(self, clear_updaters=True):
        if clear_updaters:
            self.root_dots.clear_updaters()
            self.coef_dots.clear_updaters()

        def update_root_dots(rdots):
            old_roots = self.get_roots()
            unordered_roots = coefficients_to_roots(self.get_coefs())
            # Sort them to match the old_roots
            roots = []
            for old_root in old_roots:
                if len(unordered_roots) == 0:
                    break
                distances = [abs(old_root - ur) for ur in unordered_roots]
                root = unordered_roots[np.argmin(distances)]
                unordered_roots.remove(root)
                roots.append(root)
            for dot, root in zip(rdots, roots):
                dot.move_to(self.root_plane.n2p(root))

        self.root_dots.add_updater(update_root_dots)
        self.add(self.root_dots)

    def add_tracers(self, time_traced=2.0, **kwargs):
        self.tracers = VGroup(*(
            TracingTail(
                dot,
                stroke_color=dot.get_fill_color(),
                time_traced=time_traced,
                **kwargs
            )
            for dot in self.get_all_dots()
        ))
        self.add(self.tracers)

    def get_tracking_lines(self, dots, syms, stroke_width=1, stroke_opacity=0.5):
        lines = VGroup(*(
            Line(
                stroke_color=root.get_fill_color(),
                stroke_width=stroke_width,
                stroke_opacity=stroke_opacity,
            )
            for root in dots
        ))

        def update_lines(lines):
            for sym, dot, line in zip(syms, dots, lines):
                line.put_start_and_end_on(
                    sym.get_bottom(),
                    dot.get_center()
                )

        lines.add_updater(update_lines)
        return lines

    def add_root_lines(self, **kwargs):
        self.root_lines = self.get_tracking_lines(
            self.root_dots,
            self.get_r_symbols(self.root_poly),
            **kwargs
        )
        self.add(self.root_lines)

    def add_coef_lines(self, **kwargs):
        self.coef_lines = self.get_tracking_lines(
            self.coef_dots,
            self.get_c_symbols(self.coef_poly),
            **kwargs
        )
        self.add(self.coef_lines)

    def add_dot_labels(self, labels, dots, buff=0.05):
        for label, dot in zip(labels, dots):
            label.scale(self.label_font_size / label.font_size)
            label.set_fill(dot.get_fill_color())
            label.set_stroke(BLACK, 3, background=True)
            label.dot = dot
            label.add_updater(lambda m: m.next_to(m.dot, UR, buff=buff))
        self.add(*labels)
        return labels

    def add_r_labels(self):
        self.r_dot_labels = self.add_dot_labels(
            VGroup(*(
                Tex(f"r_{i}")
                for i in range(self.get_degree())
            )),
            self.root_dots
        )

    def add_c_labels(self):
        self.c_dot_labels = self.add_dot_labels(
            VGroup(*(
                Tex(f"c_{i}")
                for i in range(self.get_degree())
            )),
            self.coef_dots
        )

    def add_value_label(self):
        pass  # TODO

    # Animations
    def play(self, *anims, **kwargs):
        movers = list(it.chain(*(anim.mobject.get_family() for anim in anims)))
        roots_move = any(rd in movers for rd in self.root_dots)
        coefs_move = any(cd in movers for cd in self.coef_dots)
        if roots_move and not coefs_move:
            self.tie_coefs_to_roots()
        elif coefs_move and not roots_move:
            self.tie_roots_to_coefs()
        super().play(*anims, **kwargs)

    def get_root_swap_arrows(self, i, j,
                             path_arc=90 * DEGREES,
                             stroke_width=5,
                             stroke_opacity=0.7,
                             buff=0.3,
                             **kwargs):
        di = self.root_dots[i].get_center()
        dj = self.root_dots[j].get_center()
        kwargs["path_arc"] = path_arc
        kwargs["stroke_width"] = stroke_width
        kwargs["stroke_opacity"] = stroke_opacity
        kwargs["buff"] = buff
        return VGroup(
            Arrow(di, dj, **kwargs),
            Arrow(dj, di, **kwargs),
        )

    def swap_roots(self, *indices, run_time=2, wait_time=1, **kwargs):
        self.play(CyclicReplace(
            *(
                self.root_dots[i]
                for i in indices
            ),
            run_time=run_time,
            **kwargs
        ))
        self.wait(wait_time)

    def rotate_coefs(self, indicies, center_z=0, run_time=5, wait_time=1, **kwargs):
        self.play(*(
            Rotate(
                self.coef_dots[i], TAU,
                about_point=self.coef_plane.n2p(center_z),
                run_time=run_time,
                **kwargs
            )
            for i in indicies
        ))
        self.wait(wait_time)

    def rotate_coef(self, i, **kwargs):
        self.rotate_coefs([i], **kwargs)

    # Interaction

    def on_mouse_press(self, point, button, mods):
        try:
            super().on_mouse_press(point, button, mods)
            mob = self.point_to_mobject(
                point,
                search_set=self.get_all_dots(),
                buff=0.1
            )
            if mob is None:
                return
            if mob in self.root_dots:
                self.tie_coefs_to_roots()
                self.add(*self.root_dots)
            elif mob in self.coef_dots:
                self.tie_roots_to_coefs()
                self.add(*self.coef_dots)
            self.mouse_drag_point.move_to(point)
            mob.add_updater(lambda m: m.move_to(self.mouse_drag_point))
            self.unlock_mobject_data()
            self.lock_static_mobject_data()
        except Exception as e:
            print(e)

    def on_mouse_release(self, point, button, mods):
        super().on_mouse_release(point, button, mods)
        self.root_dots.clear_updaters()
        self.coef_dots.clear_updaters()


class CubicFormula(RootCoefScene):
    coefs = [1, -1, 0, 1]
    root_plane_config = {
        "x_range": (-2.0, 2.0),
        "y_range": (-2.0, 2.0),
        "background_line_style": {
            "stroke_color": BLUE_E,
            "stroke_width": 1.0,
        }
    }
    coef_plane_config = {
        "x_range": (-2.0, 2.0),
        "y_range": (-2.0, 2.0),
        "background_line_style": {
            "stroke_color": GREY,
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
    plane_height = 3.0
    plane_buff = 1.0
    planes_center = 1.5 * UP

    sqrt_dot_config = {
        "color": GREEN,
        "radius": 0.05,
        "stroke_color": BLACK,
        "stroke_width": 4,
        "draw_stroke_behind_fill": True,
    }

    cr_dot_config = {
        "color": YELLOW_D,
        "radius": 0.05,
        "stroke_color": BLACK,
        "stroke_width": 4,
        "draw_stroke_behind_fill": True,
    }

    def add_planes(self):
        super().add_planes()
        self.root_plane_label.next_to(self.root_plane, LEFT)
        self.coef_plane_label.next_to(self.coef_plane, RIGHT)
        self.add_sqrt_plane()
        self.add_crt_plane()

    def add_sqrt_plane(self):
        sqrt_plane = ComplexPlane(**self.sqrt_plane_config)
        sqrt_plane.move_to(self.root_plane)
        sqrt_plane.to_edge(DOWN)

        label = Tex(
            "\\delta = \\sqrt{ \\frac{q^2}{4} + \\frac{p^3}{27}}",
            font_size=30,
            tex_to_color_map={"\\delta": GREEN}
        )
        label.next_to(sqrt_plane, LEFT, aligned_edge=DOWN)

        self.add(sqrt_plane, label)
        self.sqrt_plane = sqrt_plane
        self.sqrt_plane_label = label

    def add_crt_plane(self):
        crt_plane = ComplexPlane(**self.crt_plane_config)
        label = Tex(
            # "\\frac{1}{8}\\left("
            "\\sqrt[3]{-\\frac{q}{2} + \\delta} +",
            "\\sqrt[3]{-\\frac{q}{2} - \\delta}",
            # "\\right)",
            font_size=30,
            tex_to_color_map={"\\delta": GREEN}
        )
        crt_plane.move_to(self.coef_plane)
        crt_plane.to_edge(DOWN)

        label.next_to(crt_plane, RIGHT, aligned_edge=DOWN)

        self.add(crt_plane, label)
        self.crt_plane = crt_plane
        self.crt_plane_label = label

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

    def add_sqrt_dots(self):
        sqrt_dots = Dot(**self.sqrt_dot_config).replicate(2)

        def update_sqrt_dots(dots):
            q, p, zero, one = self.get_coefs()
            disc = (q**2 / 4) + (p**3 / 27)
            roots = get_nth_roots(disc, 2)
            points = map(self.sqrt_plane.n2p, roots)
            # TODO: Align with previous points
            for dot, point in zip(dots, points):
                dot.move_to(point)
            return dot

        sqrt_dots.add_updater(update_sqrt_dots)

        self.sqrt_dots = sqrt_dots
        self.add(sqrt_dots)

        # Labels
        self.delta_labels = self.add_dot_labels(
            Tex("\\delta").replicate(2), sqrt_dots
        )

    def add_crt_dots(self):
        crt_dots = Dot(**self.sqrt_dot_config).replicate(2)

        # TODO, refactor to prevent code duplication here
        def update_crt_dots(dots):
            q, p, zero, one = self.get_coefs()
            disc = (q**2 / 4) + (p**3 / 27)
            deltas = get_nth_roots(disc, 2)
            # TODO add all the cube roots

            points = map(self.crt_plane.n2p, roots)
            # TODO: Align with previous points
            for dot, point in zip(dots, points):
                dot.move_to(point)
            return dot

        crt_dots.add_updater(update_crt_dots)

        self.crt_dots = crt_dots
        self.add(crt_dots)



# Scenes

class AmbientRootSwapping(RootCoefScene):
    n_swaps = 20

    def construct(self):
        for x in range(self.n_swaps):
            k = random.randint(2, 5)
            indices = random.choice(list(it.combinations(range(5), k)))
            self.swap_roots(*indices)
            self.wait()


class CubicFormulaTest(CubicFormula):
    def construct(self):
        self.embed()
