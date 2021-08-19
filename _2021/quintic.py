from manim_imports_ext import *


def roots_to_coefficients(roots):
    n = len(list(roots))
    return [
        ((-1)**(n - k)) * sum(np.prod(tup) for tup in it.combinations(roots, k + 1))
        for k in range(n - 1, -1, -1)
    ]


def poly(x, coefs):
    return sum(coefs[k] * x**k for k in range(len(coefs)))


def dpoly(x, coefs):
    return sum(k * coefs[k] * x**(k - 1) for k in range(1, len(coefs)))


def find_root(func, dfunc, seed=complex(1, 1), tol=1e-6, max_steps=100):
    # Use newton's method
    last_seed = np.inf
    for n in range(max_steps):
        if abs(seed - last_seed) < tol:
            break
        last_seed = seed
        seed = seed - func(seed) / dfunc(seed)
    return seed


def coefficients_to_roots(coefs):
    coefs = [*coefs, 1]  # Make the monomial term explicit

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


class RootCoefScene(Scene):
    root_plane_config = {
        "x_range": (-2, 2),
        "y_range": (-2, 2),
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

    degree = 5

    root_color = YELLOW
    coef_color = RED_B

    dot_style = {
        "radius": 0.05,
        "stroke_color": BLACK,
        "stroke_width": 3,
        "draw_stroke_behind_fill": True,
    }

    def add_planes(self):
        # Planes
        planes = VGroup(
            ComplexPlane(**self.root_plane_config),
            ComplexPlane(**self.coef_plane_config),
        )
        for plane in planes:
            plane.set_height(self.plane_height)
        planes.arrange(RIGHT, buff=self.plane_buff)

        for plane in planes:
            plane.add_coordinate_labels(font_size=18)
            plane.coordinate_labels.set_opacity(0.8)

        root_plane, coef_plane = planes

        # Lower labels
        root_plane_label = Text("Roots")
        coef_plane_label = Text("Coefficients")

        root_plane_label.next_to(root_plane, DOWN)
        coef_plane_label.next_to(coef_plane, DOWN)

        # Upper labels
        root_poly = Tex(*(
            f"(x - r_{i})"
            for i in range(self.degree)
        ))
        self.get_r_symbols(root_poly).set_color(self.root_color)
        root_poly.next_to(root_plane, UP)
        root_poly.set_max_width(root_plane.get_width())

        coef_poly = Tex(
            f"x^{self.degree}", *(
                f" + c_{n} x^{n}"
                for n in range(self.degree - 1, -1, -1)
            )
        )
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
        self.poly_equal_sign = equals

    def get_r_symbols(self, root_poly):
        return VGroup(*it.chain(*(
            part[3:5] for part in root_poly
        )))

    def get_c_symbols(self, coef_poly):
        return VGroup(*(it.chain(*(
            part[1:3] for part in coef_poly[1:]
        ))))

    def get_random_root(self):
        return complex(
            interpolate(*self.root_plane.x_range[:2], random.random()),
            interpolate(*self.root_plane.y_range[:2], random.random()),
        )

    def get_random_roots(self):
        return [self.get_random_root() for x in range(self.degree)]

    def get_roots_of_unity(self):
        return [np.exp(complex(0, TAU * n / self.degree)) for n in range(self.degree)]

    def create_root_dots(self, roots):
        return VGroup(*(
            Dot(
                self.root_plane.n2p(root),
                color=self.root_color,
                **self.dot_style,
            )
            for root in roots
        ))

    def create_coef_dots(self, coefs):
        return VGroup(*(
            Dot(
                self.coef_plane.n2p(coef),
                color=self.coef_color,
                **self.dot_style,
            )
            for coef in coefs
        ))

    def add_root_dots(self, roots=None):
        if roots is None:
            roots = self.get_roots_of_unity()
        self.root_dots = self.create_root_dots(roots)
        self.add(self.root_dots)

    def add_coef_dots(self, coefs=None):
        if coefs is None:
            coefs = [0] * self.degree
        self.coef_dots = self.create_coef_dots(coefs)
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
        ]

    def tie_coefs_to_roots(self):
        def update_coef_dots(cdots):
            coefs = roots_to_coefficients(self.get_roots())
            for dot, coef in zip(cdots, coefs):
                dot.move_to(self.coef_plane.n2p(coef))

        self.coef_dots.add_updater(update_coef_dots)

    def tie_roots_to_coefs(self):
        def update_root_dots(rdots):
            roots = coefficients_to_roots(self.get_coefs())
            for dot, root in zip(rdots, roots):
                dot.move_to(self.root_plane.n2p(root))

        self.root_dots.add_updater(update_root_dots)


class Test(RootCoefScene):
    def construct(self):
        self.add_planes()
        coefs = list(range(-2, 3, 1))
        roots = coefficients_to_roots(coefs)
        self.add_root_dots(roots)
        self.add_coef_dots(coefs)

        self.tie_coefs_to_roots()

        self.root_dots[0].set_color(RED)

        self.play(Swap(*self.root_dots[:2]), run_time=3)

        self.embed()
