from manim_imports_ext import *

# Helpers


def roots_to_coefficients(roots):
    n = len(list(roots))
    return [
        ((-1)**(n - k)) * sum(
            np.prod(tup)
            for tup in it.combinations(roots, n - k)
        )
        for k in range(n)
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
    if len(coefs) == 0:
        return []
    elif coefs[-1] == 0:
        return coefficients_to_roots(coefs[:-1])
    roots = []
    # Find a root, divide out by (x - root), repeat
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


def sort_to_minimize_distances(unordered_points, reference_points):
    """
    Sort the initial list of points in R^n so that the sum
    of the distances between corresponding points in both lists
    is smallest
    """
    ordered_points = []
    unused_points = list(unordered_points)

    for ref_point in reference_points:
        distances = [get_norm(ref_point - up) for up in unused_points]
        index = np.argmin(distances)
        ordered_points.append(unused_points.pop(index))
    return ordered_points


def optimal_transport(dots, target_points):
    """
    Move the dots to the target points such that each dot moves a minimal distance
    """
    points = sort_to_minimize_distances(target_points, [d.get_center() for d in dots])
    for dot, point in zip(dots, points):
        dot.move_to(point)
    return dots


def x_power_tex(power, base="x"):
    if power == 0:
        return ""
    elif power == 1:
        return base
    else:
        return f"{base}^{{{power}}}"


def poly_tex(coefs, prefix="P(x) = ", coef_color=RED_B):
    n = len(coefs) - 1
    coefs = [f"{{{coef}}}" for coef in coefs]
    terms = [prefix, x_power_tex(n)]
    for k in range(n - 1, -1, -1):
        coef = coefs[k]
        if not coef[1] == "-":
            terms.append("+")
        terms.append(str(coef))
        terms.append(x_power_tex(k))
    t2c = dict([(coef, coef_color) for coef in coefs])
    return Tex(*terms, tex_to_color_map=t2c)


def factored_poly_tex(roots, prefix="P(x) = ", root_colors=[YELLOW, YELLOW]):
    roots = list(roots)
    root_colors = color_gradient(root_colors, len(roots))
    root_texs = [str(r) for r in roots]
    parts = []
    if prefix:
        parts.append(prefix)
    for root_tex in root_texs:
        parts.extend(["(", "x", "-", root_tex, ")"])
    t2c = dict((
        (rt, root_color)
        for rt, root_color in zip(root_texs, root_colors)
    ))
    return Tex(*parts, tex_to_color_map=t2c)


def sym_poly_tex_args(roots, k, abbreviate=False):
    result = []
    subsets = list(it.combinations(roots, k))
    if k in [1, len(roots)]:
        abbreviate = False
    if abbreviate:
        subsets = [*subsets[:2], subsets[-1]]
    for subset in subsets:
        if abbreviate and subset is subsets[-1]:
            result.append(" \\cdots ")
            result.append("+")
        for r in subset:
            result.append(str(r))
            result.append(" \\cdot ")
        result.pop()
        result.append("+")
    result.pop()
    return result


def expanded_poly_tex(roots, vertical=True, root_colors=[YELLOW, YELLOW], abbreviate=False):
    roots = list(roots)
    root_colors = color_gradient(root_colors, len(roots))
    n = len(roots)
    kw = dict(
        tex_to_color_map=dict((
            (str(r), root_color)
            for r, root_color in zip(roots, root_colors)
        )),
        arg_separator=" "
    )
    result = VGroup()
    result.add(Tex(f"x^{{{n}}}"))
    for k in range(1, n + 1):
        sym_poly = sym_poly_tex_args(
            roots, k,
            abbreviate=abbreviate
        )
        line = Tex(
            "+" if k % 2 == 0 else "-",
            "\\big(", *sym_poly, "\\big)",
            x_power_tex(n - k),
            **kw,
        )
        result.add(line)
    for line in result:
        line[-1].set_color(WHITE)
    if vertical:
        result.arrange(DOWN, aligned_edge=LEFT)
    else:
        result.arrange(RIGHT, buff=SMALL_BUFF)
        result[0].shift(result[0].get_height() * UP / 4)
    return result


def get_symmetric_system(lhss,
                         roots=None,
                         root_colors=[YELLOW, YELLOW],
                         lhs_color=RED_B,
                         abbreviate=False,
                         signed=False,
                         ):
    lhss = [f"{{{lhs}}}" for lhs in lhss]
    if roots is None:
        roots = [f"r_{{{i}}}" for i in range(len(lhss))]
    root_colors = color_gradient(root_colors, len(roots))
    t2c = dict([
        (root, root_color)
        for root, root_color in zip(roots, root_colors)
    ])
    t2c.update(dict([
        (str(lhs), lhs_color)
        for lhs in lhss
    ]))
    kw = dict(tex_to_color_map=t2c)
    equations = VGroup(*(
        Tex(
            lhs, "=",
            "-(" if neg else "",
            *sym_poly_tex_args(roots, k, abbreviate=abbreviate),
            ")" if neg else "",
            **kw
        )
        for k, lhs in zip(it.count(1), lhss)
        for neg in [signed and k % 2 == 1]
    ))
    equations.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
    for eq in equations:
        eq.shift((equations[0][1].get_x() - eq[1].get_x()) * RIGHT)
    return equations


def get_quadratic_formula(lhs="", **tex_config):
    return MTex(
        lhs + "{-{b} \\pm \\sqrt{ {b}^2 - 4{a}{c} } \\over 2{a} }",
        **tex_config
    )


def get_full_cubic_formula(lhs="", **tex_config):
    # Thanks to Mathologer and MathPix here...
    return Tex(lhs + """
        &\\sqrt[3]{\\left(-{ {b}^{3} \\over 27 {a}^{3}}+{ {b} {c} \\over 6 {a}^{2}}
            -{ {d} \\over 2 {a} }\\right)-\\sqrt{\\left(-{ {b}^{3} \\over 27 {a}^{3}}
            +{ {b} {c} \\over 6 {a}^{2}}-{ {d} \\over 2 {a}}\\right)^{2}
            +\\left({ {c} \\over 3 {a} }-{ {b}^{2} \\over 9 {a}^{2}}\\right)^{3}}} \\\\
        +&\\sqrt[3]{\\left(-{ {b}^{3} \\over 27 {a}^{3}}+{ {b} {c} \\over 6 {a}^{2}}
            -{ {d} \\over 2 {a} }\\right)+\\sqrt{\\left(-{ {b}^{3} \\over 27 {a}^{3}}
            +{ {b} {c} \\over 6 {a}^{2}}-{ {d} \\over 2 {a}}\\right)^{2}
            +\\left({ {c} \\over 3 {a} }-{ {b}^{2} \\over 9 {a}^{2} }\\right)^{3}}} \\\\
        -&{ {b} \\over 3 {a} }
    """, **tex_config)


def get_cubic_formula(lhs="", **tex_config):
    return MTex(
        lhs + """
         \\sqrt[3]{-{q \\over 2}-\\sqrt{\\left({q \\over 2}\\right)^{2}+\\left({p \\over 3}\\right)^{3}}}
        +\\sqrt[3]{-{q \\over 2}+\\sqrt{\\left({q \\over 2}\\right)^{2}+\\left({p \\over 3}\\right)^{3}}}
        """,
        **tex_config
    )


def get_quartic_formula(lhs="", **tex_config):
    pass


# General scene types


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
    plane_arrangement = LEFT

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
    continuous_roots = True
    show_equals = True

    def setup(self):
        self.lock_coef_imag = False
        self.lock_coef_norm = False
        self.add_planes()
        self.add_dots()
        self.active_dot_aura = Group()
        self.add(self.active_dot_aura)
        self.prepare_cycle_interaction()
        if self.include_tracers:
            self.add_all_tracers()
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
        planes.arrange(self.plane_arrangement, buff=self.plane_buff)
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

        self.add(planes)
        self.add(root_plane_label, coef_plane_label)
        self.add(root_poly, coef_poly)

        if self.show_equals:
            equals = Tex("=")
            equals.move_to(midpoint(root_poly.get_right(), coef_poly.get_left()))
            self.add(equals)
            self.poly_equal_sign = equals

        self.root_plane = root_plane
        self.coef_plane = coef_plane
        self.root_plane_label = root_plane_label
        self.coef_plane_label = coef_plane_label
        self.root_poly = root_poly
        self.coef_poly = coef_poly

    def get_degree(self):
        return len(self.coefs) - 1

    def get_coef_poly(self):
        degree = self.get_degree()
        return Tex(
            f"x^{degree}",
            *(
                f" + c_{n} x^{n}"
                for n in range(degree - 1, 1, -1)
            ),
            " + c_{1} x",
            " + c_{0}",
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
        self.coef_dots.add_updater(self.update_coef_dots_by_roots)
        self.add(self.coef_dots)
        self.add(*self.root_dots)

    def update_coef_dots_by_roots(self, coef_dots):
        coefs = roots_to_coefficients(self.get_roots())
        for dot, coef in zip(coef_dots, coefs):
            dot.move_to(self.coef_plane.n2p(coef))
        return coef_dots

    def tie_roots_to_coefs(self, clear_updaters=True):
        if clear_updaters:
            self.root_dots.clear_updaters()
            self.coef_dots.clear_updaters()
        self.root_dots.add_updater(self.update_root_dots_by_coefs)
        self.add(self.root_dots)
        self.add(*self.coef_dots)

    def update_root_dots_by_coefs(self, root_dots):
        new_roots = coefficients_to_roots(self.get_coefs())
        new_root_points = map(self.root_plane.n2p, new_roots)
        if self.continuous_roots:
            optimal_transport(root_dots, new_root_points)
        else:
            for dot, point in zip(root_dots, new_root_points):
                dot.move_to(point)
        return root_dots

    def get_tracers(self, dots, time_traced=2.0, **kwargs):
        tracers = VGroup()
        for dot in dots:
            dot.tracer = TracingTail(
                dot,
                stroke_color=dot.get_fill_color(),
                time_traced=time_traced,
                **kwargs
            )
            tracers.add(dot.tracer)
        return tracers

    def add_all_tracers(self, **kwargs):
        self.tracers = self.get_tracers(self.get_all_dots())
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
    def add_dot_auroa(self, dot):
        glow_dot = GlowDot(color=WHITE)
        always(glow_dot.move_to, dot)
        self.active_dot_aura.add(glow_dot)

    def remove_dot_aura(self):
        if len(self.active_dot_aura) > 0:
            self.play(FadeOut(self.active_dot_aura), run_time=0.5)
            self.active_dot_aura.set_submobjects([])
            self.add(self.active_dot_aura)

    def prepare_cycle_interaction(self):
        self.dots_awaiting_cycle = []
        self.dot_awaiting_loop = None

    def handle_cycle_preparation(self, dot):
        if dot in self.root_dots and dot not in self.dots_awaiting_cycle:
            self.dots_awaiting_cycle.append(dot)
        if dot in self.coef_dots and dot is not self.dot_awaiting_loop:
            self.dot_awaiting_loop = dot
        self.add(dot)
        self.refresh_locked_data()

    def carry_out_cycle(self):
        if self.dots_awaiting_cycle:
            self.tie_coefs_to_roots()
            self.unlock_mobject_data()
            self.play(CyclicReplace(*self.dots_awaiting_cycle, run_time=5))
            self.remove_dot_aura()
        if self.dot_awaiting_loop is not None:
            self.tie_roots_to_coefs()
            self.unlock_mobject_data()
            self.play(Rotate(
                self.dot_awaiting_loop,
                angle=TAU,
                about_point=self.mouse_point.get_center().copy(),
                run_time=8
            ))
            self.remove_dot_aura()
        self.prepare_cycle_interaction()

    def on_mouse_release(self, point, button, mods):
        super().on_mouse_release(point, button, mods)
        if self.root_dots.has_updaters or self.coef_dots.has_updaters:
            # End the interaction where a dot is tied to the mouse
            self.root_dots.clear_updaters()
            self.coef_dots.clear_updaters()
            self.remove_dot_aura()
            return
        dot = self.point_to_mobject(point, search_set=self.get_all_dots(), buff=0.1)
        if dot is None:
            return
        self.add_dot_auroa(dot)
        if self.window.is_key_pressed(ord("c")):
            self.handle_cycle_preparation(dot)
            return

        # Make sure other dots are updated accordingly
        if dot in self.root_dots:
            self.tie_coefs_to_roots()
        elif dot in self.coef_dots:
            self.tie_roots_to_coefs()

        # Otherwise, have this dot track with the mouse
        dot.last_norm = get_norm(self.coef_plane.p2c(dot.get_center()))
        dot.last_y = dot.get_y()
        dot.add_updater(lambda m: m.move_to(self.mouse_point))
        if self.lock_coef_imag or self.window.is_key_pressed(ord("r")):
            # Fix the imaginary value
            dot.add_updater(lambda d: d.set_y(d.last_y))
        elif (self.lock_coef_norm or self.window.is_key_pressed(ord("a"))) and dot in self.coef_dots:
            # Fix the norm
            dot.add_updater(lambda d: d.move_to(self.coef_plane.c2p(
                *d.last_norm * normalize(self.coef_plane.p2c(d.get_center()))
            )))
        self.refresh_locked_data()

    def on_key_release(self, symbol, modifiers):
        super().on_key_release(symbol, modifiers)
        char = chr(symbol)
        if char == "c":
            self.carry_out_cycle()

    #
    def update_mobjects(self, dt):
        # Go in reverse order, since dots are often re-added
        # once they become interactive
        for mobject in reversed(self.mobjects):
            mobject.update(dt)


class RadicalScene(RootCoefScene):
    n = 3
    c = 1.5
    root_plane_config = {
        "x_range": (-2.0, 2.0),
        "y_range": (-2.0, 2.0),
        "background_line_style": {
            "stroke_color": BLUE_E,
        }
    }
    coef_plane_config = {
        "x_range": (-2, 2),
        "y_range": (-2, 2),
        "background_line_style": {
            "stroke_color": GREY,
        }
    }
    plane_height = 4.0
    plane_buff = 3.0
    planes_center = 1.5 * DOWN
    show_equals = False

    def setup(self):
        self.coefs = [-self.c, *[0] * (self.n - 1), 1]
        super().setup()
        self.remove(self.coef_plane_label)
        self.remove(self.root_plane_label)
        self.sync_roots(self.root_dots[0])

    def get_radical_labels(self):
        left = self.coef_plane.get_right()
        right = self.root_plane.get_left()
        arrow_kw = dict(
            stroke_width=5,
            stroke_color=GREY_A,
            buff=0.5,
        )
        r_arrow = Arrow(left, right, **arrow_kw).shift(UP)
        l_arrow = Arrow(right, left, **arrow_kw).shift(DOWN)

        r_label = Tex(f"\\sqrt[{self.n}]{{c}}")[0]
        l_label = Tex(f"r_i^{{{self.n}}}")[0]
        r_label[3].set_color(self.coef_color)
        l_label[-3::2].set_color(self.root_color)

        if self.n == 2:
            r_label[0].set_opacity(0)

        r_label.next_to(r_arrow, UP)
        l_label.next_to(l_arrow, UP)

        return VGroup(
            VGroup(r_arrow, l_arrow),
            VGroup(r_label, l_label),
        )

    def get_angle_label(self, dot, plane, sym, get_theta):
        line = Line()
        line.set_stroke(dot.get_fill_color(), width=2)
        line.add_updater(lambda m: m.put_start_and_end_on(
            plane.get_origin(), dot.get_center()
        ))

        arc = always_redraw(lambda: ParametricCurve(
            lambda t: plane.n2p((0.25 + 0.01 * t) * np.exp(complex(0, t))),
            t_range=[0, get_theta() + 1e-5, 0.025],
            stroke_width=2,
        ))

        tex_mob = Tex(sym, font_size=24)
        tex_mob.set_backstroke(width=8)

        def update_sym(tex_mob):
            tex_mob.set_opacity(min(1, 3 * get_theta()))
            point = arc.t_func(0.5 * get_theta())
            origin = plane.get_origin()
            w = tex_mob.get_width()
            tex_mob.move_to(origin + (1.3 + 2 * w) * (point - origin))
            return tex_mob

        tex_mob.add_updater(update_sym)

        return VGroup(line, arc, tex_mob)

    def get_c(self):
        return -self.get_coefs()[0]

    # Updates to RootCoefScene methods
    def get_coef_poly(self):
        degree = self.get_degree()
        return Tex(f"x^{degree}", "-", "c")

    def get_c_symbols(self, coef_poly):
        return VGroup(coef_poly[-1])

    def add_c_labels(self):
        self.c_dot_labels = self.add_dot_labels(
            VGroup(Tex("c")),
            VGroup(self.coef_dots[0]),
        )

    def get_coefs(self):
        c = self.coef_plane.p2n(self.coef_dots[0].get_center())
        return [-c, *[0] * (self.n - 1), 1]

    def set_coefs(self, coefs):
        super().set_coefs(coefs)
        self.coef_dots[0].move_to(self.coef_plane.n2p(-coefs[0]))
        self.coef_dots[1:].set_opacity(0)

    def tie_coefs_to_roots(self, *args, **kwargs):
        super().tie_coefs_to_roots(*args, **kwargs)
        # Hack
        for dot in self.root_dots:
            dot.add_updater(lambda m: m)

    def update_coef_dots_by_roots(self, coef_dots):
        controlled_roots = [
            d for d in self.root_dots
            if len(d.get_updaters()) > 1
        ]
        root_dot = controlled_roots[0] if controlled_roots else self.root_dots[0]
        root = self.root_plane.p2n(root_dot.get_center())
        coef_dots[0].move_to(self.coef_plane.n2p(root**self.n))
        # Update all the root dots
        if controlled_roots:
            self.sync_roots(controlled_roots[0])
        return coef_dots

    def sync_roots(self, anchor_root_dot):
        root = self.root_plane.p2n(anchor_root_dot.get_center())
        anchor_index = self.root_dots.submobjects.index(anchor_root_dot)
        for i, dot in enumerate(self.root_dots):
            if i != anchor_index:
                zeta = np.exp(complex(0, (i - anchor_index) * TAU / self.n))
                dot.move_to(self.root_plane.n2p(zeta * root))


class QuadraticFormula(RootCoefScene):
    coefs = [-1, 0, 1]
    coef_plane_config = {
        "x_range": (-2.0, 2.0),
        "y_range": (-2.0, 2.0),
        "background_line_style": {
            "stroke_color": GREY_B,
            "stroke_width": 1.0,
        }
    }
    sqrt_plane_config = {
        "x_range": (-2.0, 2.0),
        "y_range": (-2.0, 2.0),
        "background_line_style": {
            "stroke_color": BLUE_D,
            "stroke_width": 1.0,
        }
    }
    plane_height = 3.5
    plane_arrangement = RIGHT
    plane_buff = 1.0
    planes_center = 2 * LEFT + DOWN

    def add_planes(self):
        super().add_planes()
        self.coef_plane_label.match_y(self.root_plane_label)
        self.add_sqrt_plane()

    def add_sqrt_plane(self):
        plane = ComplexPlane(**self.sqrt_plane_config)
        plane.next_to(self.coef_plane, self.plane_arrangement, self.plane_buff)
        plane.set_height(self.plane_height)
        plane.add_coordinate_labels(font_size=24)

        label = Tex(
            "-{b \\over 2} \\pm \\sqrt{{b^2 \\over 4} - c}",
            font_size=30,
        )[0]
        for i in [1, 7, 12]:
            label[i].set_color(self.coef_color)
        label.next_to(plane, UP)

        self.sqrt_plane = plane
        self.sqrt_label = label
        self.add(plane)
        self.add(label)

    def add_dots(self):
        super().add_dots()
        dots = self.root_dots.copy().clear_updaters()
        dots.set_color(GREEN)

        def update_dots(dots):
            for dot, root in zip(dots, self.get_roots()):
                dot.move_to(self.sqrt_plane.n2p(root))
            return dots

        dots.add_updater(update_dots)

        self.sqrt_dots = dots
        self.add(dots)
        self.add(self.get_tracers(dots))

    def get_coef_poly(self):
        return Tex(
            "x^2", "+ b x", "+ c"
        )

    def add_c_labels(self):
        self.c_dot_labels = self.add_dot_labels(
            VGroup(Tex("c"), Tex("b")),
            self.coef_dots
        )

    def get_c_symbols(self, coef_poly):
        return VGroup(*(part[1] for part in coef_poly[:0:-1]))


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


# Introduction

class IntroduceUnsolvability(Scene):
    def construct(self):
        pass


class TableOfContents(Scene):
    def construct(self):
        pass


# Preliminaries on polynomials

class ConstructPolynomialWithGivenRoots(Scene):
    root_color = YELLOW

    def construct(self):
        # Add axes
        axes = self.add_axes()

        # Add challenge
        challenge = VGroup(
            Text("Can you construct a cubic polynomial"),
            Tex(
                "P(x) = x^3 + c_2 x^2 + c_1 x + c_0",
                tex_to_color_map={
                    "c_2": RED_B,
                    "c_1": RED_B,
                    "c_0": RED_B,
                }
            ),
            TexText(
                "with roots at $x = 1$, $x = 2$, and $x = 4$?",
                tex_to_color_map={
                    "$x = 1$": self.root_color,
                    "$x = 2$": self.root_color,
                    "$x = 4$": self.root_color,
                }
            )
        )
        challenge.scale(0.7)
        challenge.arrange(DOWN, buff=MED_LARGE_BUFF)
        challenge.to_corner(UL)

        self.add(challenge)

        # Add graph
        roots = [1, 2, 4]
        coefs = roots_to_coefficients(roots)
        graph = axes.get_graph(lambda x: poly(x, coefs))
        graph.set_color(BLUE)

        root_dots = Group(*(GlowDot(axes.c2p(x, 0)) for x in roots))
        root_dots.set_color(self.root_color)

        x_terms = challenge[2].get_parts_by_tex("x = ")

        self.wait()
        self.play(
            LaggedStart(*(
                FadeTransform(x_term.copy(), dot)
                for x_term, dot in zip(x_terms, root_dots)
            ), lag_ratio=0.7, run_time=3)
        )
        self.add(graph, root_dots)
        self.play(ShowCreation(graph, run_time=3, rate_func=linear))
        self.wait()

        # Show factored solution
        factored = factored_poly_tex(roots)
        factored.match_height(challenge[1])
        factored.next_to(challenge, DOWN, LARGE_BUFF)

        rects = VGroup(*(
            SurroundingRectangle(
                factored[i:i + 5],
                stroke_width=1,
                stroke_color=BLUE,
                buff=0.05
            )
            for i in range(1, 12, 5)
        ))
        arrows = VGroup(*(
            Vector(DOWN).next_to(dot, UP, buff=0)
            for dot in root_dots
        ))
        zeros_eqs = VGroup(*(
            Tex(
                f"P({r}) = 0",
                font_size=24
            ).next_to(rect, UP, SMALL_BUFF)
            for r, rect in zip(roots, rects)
        ))

        self.play(FadeIn(factored, DOWN))
        self.wait()
        to_fade = VGroup()
        for rect, arrow, eq in zip(rects, arrows, zeros_eqs):
            self.play(
                ShowCreation(rect),
                FadeIn(eq),
                ShowCreation(arrow),
                FadeOut(to_fade)
            )
            self.wait(2)
            to_fade = VGroup(rect, arrow, eq)
        self.play(FadeOut(to_fade))

        # Expand solution
        x_terms = factored[2::5]
        root_terms = VGroup(*(
            VGroup(m1, m2)
            for m1, m2 in zip(factored[3::5], factored[4::5])
        ))

        expanded = Tex(
            "&x^3 ",
            "-1x^2", "-2x^2", "-4x^2 \\\\",
            "&+(-1)(-2)x", "+(-1)(-4)x", "+(-2)(-4)x\\\\",
            "&+(-1)(-2)(-4)",
        )
        for i, part in enumerate(expanded):
            if i in [1, 2, 3]:
                part[:2].set_color(self.root_color)
            elif i in [4, 5, 6, 7]:
                part[2:4].set_color(self.root_color)
                part[6:8].set_color(self.root_color)
            if i == 7:
                part[10:12].set_color(self.root_color)

        expanded.scale(0.7)
        expanded.next_to(factored[1], DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
        equals = factored[0][-1].copy()
        equals.match_y(expanded[0][0])

        self.add(equals)
        expanded_iter = iter(expanded)
        for k in range(4):
            for tup in it.combinations(range(3), k):
                factored[1:].set_opacity(0.5)
                rects = VGroup()
                for i in range(3):
                    mob = root_terms[i] if (i in tup) else x_terms[i]
                    mob.set_opacity(1)
                    rect = SurroundingRectangle(mob, buff=SMALL_BUFF)
                    rect.set_min_height(0.45, about_edge=DOWN)
                    rects.add(rect)
                rects.set_stroke(BLUE, 2)
                expanded_term = next(expanded_iter)
                expanded_rect = SurroundingRectangle(
                    expanded_term, buff=SMALL_BUFF
                )
                expanded_rect.match_style(rects)

                self.add(rects, expanded_rect)
                self.add(expanded_term)
                self.wait()
                self.remove(rects, expanded_rect)
        factored.set_opacity(1)
        self.add(expanded)
        self.wait()

        # Cleaner expansion
        cleaner_expanded = expanded_poly_tex(roots, vertical=False)
        cleaner_expanded.scale(0.7)
        cleaner_expanded.shift(expanded[0][0].get_center() - cleaner_expanded[0][0][0].get_center())

        self.play(
            FadeTransform(expanded[0], cleaner_expanded[0]),
            TransformMatchingShapes(
                expanded[1:4],
                cleaner_expanded[1],
            ),
            expanded[4:].animate.next_to(cleaner_expanded[1], DOWN, aligned_edge=LEFT)
        )
        self.wait()
        self.play(
            TransformMatchingShapes(
                expanded[4:7],
                cleaner_expanded[2],
            )
        )
        self.wait()
        self.play(
            TransformMatchingShapes(
                expanded[7],
                cleaner_expanded[3],
            )
        )
        back_rect = BackgroundRectangle(cleaner_expanded, buff=SMALL_BUFF)
        self.add(back_rect, cleaner_expanded)
        self.play(FadeIn(back_rect))
        self.wait()

        # Evaluate
        answer = Tex(
            "= x^3 -7x^2 + 14x -8",
            tex_to_color_map={
                "-7": RED_B,
                "14": RED_B,
                "-8": RED_B,
            }
        )
        answer.scale(0.7)
        answer.next_to(equals, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        self.play(FadeIn(answer, DOWN))
        self.wait()

        # Note the symmetry
        randy = Randolph(height=1)
        randy.to_corner(DL, buff=MED_SMALL_BUFF)

        randy.change("tease")
        randy.save_state()
        randy.change("plain").set_opacity(0)

        bubble = SpeechBubble(width=3, height=1, stroke_width=2)
        bubble.move_to(randy.get_corner(UR), LEFT)
        bubble.shift(0.45 * UP + 0.1 * LEFT)
        bubble.add_content(Text("Note the symmetry!"))

        self.play(Restore(randy))
        self.play(ShowCreation(bubble), Write(bubble.content))
        self.play(Blink(randy))
        self.wait()

        factored.save_state()
        cleaner_expanded.save_state()
        for alt_roots in [(2, 4, 1), (4, 2, 1), (1, 4, 2), (1, 2, 4)]:
            alt_factored = factored_poly_tex(alt_roots)
            alt_factored.replace(factored)
            alt_expanded = expanded_poly_tex(alt_roots, vertical=False)
            alt_expanded.replace(cleaner_expanded)
            globals().update(locals())
            movers, targets = [
                VGroup(*(
                    group.get_parts_by_tex(str(root))
                    for root in alt_roots
                    for group in groups
                ))
                for groups in [(factored, *cleaner_expanded), (alt_factored, *alt_expanded)]
            ]

            self.play(
                TransformMatchingShapes(movers, targets, path_arc=PI / 2, run_time=1.5),
                randy.animate.look_at(movers),
            )
            self.remove(targets, factored, cleaner_expanded)
            factored.become(alt_factored)
            cleaner_expanded.become(alt_expanded)
            self.add(factored, cleaner_expanded)
            self.wait()
        factored.restore()
        cleaner_expanded.restore()
        self.play(
            FadeOut(randy),
            FadeOut(bubble),
            FadeOut(bubble.content),
        )

        # Reverse question
        top_lhs = Tex("P(x)").match_height(factored)
        top_lhs.next_to(answer, LEFT).align_to(factored, LEFT)
        top_lhs.set_opacity(0)
        coef_poly = VGroup(top_lhs, answer)
        coef_poly.generate_target()
        coef_poly.target.set_opacity(1).to_edge(UP)

        full_factored = VGroup(back_rect, factored, equals, cleaner_expanded)
        full_factored.generate_target()
        full_factored.target.next_to(coef_poly.target, DOWN, buff=0.75, aligned_edge=LEFT)
        full_factored.target.set_opacity(0.5)

        self.add(full_factored, coef_poly)
        self.play(
            FadeOut(challenge, UP),
            MoveToTarget(full_factored),
            MoveToTarget(coef_poly),
        )

        new_challenge = Text("Find the roots!")
        new_challenge.add_background_rectangle(buff=0.1)
        arrow = Vector(LEFT)
        arrow.next_to(coef_poly, RIGHT)
        new_challenge.next_to(arrow, RIGHT)

        self.play(
            ShowCreation(arrow),
            FadeIn(new_challenge, 0.5 * RIGHT),
        )
        self.wait()

        # Show general expansion
        rs = [f"r_{i}" for i in range(3)]
        gen_factored = factored_poly_tex(rs, root_colors=[YELLOW, GREEN])
        gen_expanded = expanded_poly_tex(rs, vertical=False, root_colors=[YELLOW, GREEN])
        for gen, old in (gen_factored, factored), (gen_expanded, cleaner_expanded):
            gen.match_height(old)
            gen.move_to(old, LEFT)

        self.play(FadeTransformPieces(factored, gen_factored))
        self.wait()
        for i in range(1, 4):
            self.play(
                cleaner_expanded[0].animate.set_opacity(1),
                equals.animate.set_opacity(1),
                FadeTransformPieces(cleaner_expanded[i], gen_expanded[i]),
                cleaner_expanded[i + 1:].animate.next_to(gen_expanded[i], RIGHT, SMALL_BUFF)
            )
            self.wait()
        self.remove(cleaner_expanded)
        self.add(gen_expanded)

        full_factored = VGroup(back_rect, gen_factored, equals, gen_expanded)

        # Show system of equations
        system = get_symmetric_system([7, 14, 8], root_colors=[YELLOW, GREEN])
        system.next_to(full_factored, DOWN, LARGE_BUFF, aligned_edge=LEFT)

        coef_terms = answer[1::2]
        rhss = [term[2:-2] for term in gen_expanded[1:]]

        for coef, rhs, eq in zip(coef_terms, rhss, system):
            self.play(
                FadeTransform(coef.copy(), eq[0]),
                FadeIn(eq[1]),
                FadeTransform(rhs.copy(), eq[2:]),
            )
            self.wait()

        cubic_example = VGroup(coef_poly, full_factored, system)

        # Show quintic
        q_roots = [-1, 1, 2, 4, 6]
        q_coefs = roots_to_coefficients(q_roots)
        q_poly = poly_tex(q_coefs)
        q_poly_factored = factored_poly_tex(
            [f"r_{i}" for i in range(5)],
            root_colors=[YELLOW, GREEN]
        )
        VGroup(q_poly, q_poly_factored).scale(0.8)
        q_poly.to_corner(UL)
        q_poly_factored.next_to(q_poly, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        self.play(
            FadeOut(cubic_example, DOWN),
            FadeOut(VGroup(arrow, new_challenge), DOWN),
            FadeIn(q_poly, DOWN)
        )

        y_scale_factor = 0.1
        new_graph = axes.get_graph(
            lambda x: y_scale_factor * poly(x, q_coefs),
            x_range=(-1.2, 6.2)
        )
        new_root_dots = Group(*(
            GlowDot(axes.c2p(x, 0))
            for x in q_roots
        ))
        new_graph.match_style(graph)
        axes.save_state()
        graph.save_state()
        root_dots.save_state()
        self.play(
            Transform(graph, new_graph),
            Transform(root_dots, new_root_dots),
        )
        self.wait()

        root_terms = q_poly_factored.get_parts_by_tex("r_")
        self.play(
            FadeIn(q_poly_factored, lag_ratio=0.1, run_time=2),
            LaggedStart(*(
                FadeTransform(dot.copy(), term, remover=True)
                for dot, term in zip(root_dots, root_terms)
            ), lag_ratio=0.5, run_time=3)
        )
        self.wait()

        # Quintic system
        signed_coefs = [
            (-1)**k * c for
            k, c in zip(it.count(1), q_coefs[-2::-1])
        ]
        q_system, q_system_full = [
            get_symmetric_system(
                signed_coefs,
                abbreviate=abbrev,
                root_colors=[YELLOW, GREEN],
            )
            for abbrev in [True, False]
        ]
        for mob in q_system, q_system_full:
            mob.scale(0.8)
            mob.next_to(q_poly_factored, DOWN, LARGE_BUFF, aligned_edge=LEFT)

        root_tuple_groups = VGroup(*(
            VGroup(*(
                VGroup(*tup)
                for tup in it.combinations(root_terms, k)
            ))
            for k in range(1, 6)
        ))

        for equation, tuple_group in zip(q_system, root_tuple_groups):
            self.play(FadeIn(equation))
            self.wait(0.25)

            rects_group = VGroup(*(
                VGroup(*(
                    SurroundingRectangle(term).set_stroke(BLUE, 2)
                    for term in tup
                ))
                for tup in tuple_group
            ))
            terms_column = VGroup(*(
                VGroup(*tup).copy().arrange(RIGHT, buff=SMALL_BUFF)
                for tup in tuple_group
            ))
            terms_column.arrange(DOWN)
            terms_column.move_to(4 * RIGHT).to_edge(UP)

            anims = [
                ShowSubmobjectsOneByOne(rects_group, rate_func=linear),
                ShowIncreasingSubsets(terms_column, rate_func=linear, int_func=np.ceil),
            ]
            if equation is q_system[1]:
                anims.append(
                    Group(axes, graph, root_dots).animate.scale(
                        0.5, about_point=axes.c2p(5, -3)
                    )
                )
            self.play(*anims, run_time=0.25 * len(terms_column))
            self.remove(rects_group)
            self.wait()
            self.play(FadeOut(terms_column))
            self.wait()
        self.wait()

        frame = self.camera.frame
        frame.save_state()
        self.play(
            frame.animate.replace(q_system_full, dim_to_match=0).scale(1.1),
            FadeIn(q_system_full, lag_ratio=0.1),
            FadeOut(q_system),
            Group(axes, graph, root_dots).animate.shift(2 * DOWN),
            run_time=2,
        )
        self.wait(2)

        # Back to cubic
        self.play(
            Restore(axes),
            Restore(graph),
            Restore(root_dots),
            FadeOut(q_system_full, 2 * DOWN),
            FadeOut(q_poly, 2 * DOWN),
            FadeOut(q_poly_factored, 2 * DOWN),
            FadeIn(cubic_example, 2 * DOWN),
            Restore(frame),
            run_time=2,
        )
        self.wait()

        # Can you always factor?
        question = Text("Is this always possible?")
        question.add_background_rectangle(buff=0.1)
        question.next_to(gen_factored, RIGHT, buff=2)
        question.to_edge(UP, buff=MED_SMALL_BUFF)
        arrow = Arrow(question.get_left(), gen_factored.get_corner(UR))

        self.play(
            FadeIn(question),
            ShowCreation(arrow),
            FlashAround(gen_factored, run_time=3)
        )
        self.wait()
        self.play(FadeOut(question), FadeOut(arrow))

        const_dec = DecimalNumber(8)
        top_const_dec = const_dec.copy()
        for dec, mob, vect in (const_dec, system[2][0], RIGHT), (top_const_dec, answer[-1][1], LEFT):
            dec.match_height(mob)
            dec.move_to(mob, vect)
            dec.set_color(RED)
            mob.set_opacity(0)
            self.add(dec)
        answer[-1][0].set_color(RED)

        top_const_dec.add_updater(lambda m: m.set_value(const_dec.get_value()))

        def get_coefs():
            return [-const_dec.get_value(), 14, -7, 1]

        def get_roots():
            return coefficients_to_roots(get_coefs())

        def update_graph(graph):
            graph.become(axes.get_graph(lambda x: poly(x, get_coefs())))
            graph.set_stroke(BLUE, 3)

        def update_root_dots(dots):
            roots = get_roots()
            for root, dot in zip(roots, dots):
                if abs(root.imag) > 1e-8:
                    dot.set_opacity(0)
                else:
                    dot.move_to(axes.c2p(root.real, 0))
                    dot.set_opacity(1)

        graph.add_updater(update_graph)
        self.remove(*root_dots, *new_root_dots)
        root_dots = root_dots[:3]
        root_dots.add_updater(update_root_dots)
        self.add(root_dots)

        example_constants = [5, 6, 9, 6.28]
        for const in example_constants:
            self.play(
                ChangeDecimalToValue(const_dec, const),
                run_time=3,
            )
            self.wait()

        # Show complex plane
        plane = ComplexPlane(
            (-1, 6), (-3, 3)
        )
        plane.replace(axes.x_axis.ticks, dim_to_match=0)
        plane.add_coordinate_labels(font_size=24)
        plane.save_state()
        plane.rotate(PI / 2, LEFT)
        plane.set_opacity(0)

        real_label = Text("Real numbers")
        real_label.next_to(root_dots, UP, SMALL_BUFF)
        complex_label = Text("Complex numbers")
        complex_label.set_backstroke()
        complex_label.next_to(plane.saved_state.get_corner(UR), DL, SMALL_BUFF)

        graph.clear_updaters()
        root_dots.clear_updaters()
        axes.generate_target(use_deepcopy=True)
        axes.target.y_axis.set_opacity(0)
        axes.target.x_axis.numbers.set_opacity(1)
        self.play(
            Uncreate(graph),
            Write(real_label),
            MoveToTarget(axes),
        )
        self.wait(2)
        self.add(plane, root_dots, real_label)
        self.play(
            Restore(plane),
            FadeOut(axes.x_axis),
            FadeTransform(real_label, complex_label),
            run_time=2,
        )
        self.wait(2)

        self.play(
            VGroup(coef_poly, top_const_dec).animate.next_to(plane, UP),
            gen_factored.animate.next_to(plane, UP, buff=1.2),
            FadeOut(equals),
            FadeOut(gen_expanded),
            frame.animate.shift(DOWN),
            run_time=2,
        )
        self.wait()

        eq_zero = Tex("= 0")
        eq_zero.scale(0.7)
        eq_zero.next_to(top_const_dec, RIGHT, SMALL_BUFF)
        eq_zero.shift(0.2 * LEFT)
        self.play(
            Write(eq_zero),
            VGroup(coef_poly, top_const_dec).animate.shift(0.2 * LEFT),
        )
        self.wait()

        # Show constant tweaking again
        def update_complex_roots(root_dots):
            for root, dot in zip(get_roots(), root_dots):
                dot.move_to(plane.n2p(root))

        root_dots.add_updater(update_complex_roots)

        self.play(
            FlashAround(const_dec),
            FlashAround(top_const_dec),
            run_time=2,
        )

        self.play(
            ChangeDecimalToValue(const_dec, 4),
            run_time=3,
        )
        self.wait()
        root_eqs = VGroup(*(
            VGroup(Tex(f"r_{i} ", "="), DecimalNumber(root, num_decimal_places=3)).arrange(RIGHT)
            for i, root in enumerate(get_roots())
        ))
        root_eqs.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        for eq in root_eqs:
            eq[0][0].set_color(YELLOW)
        root_eqs.next_to(system, UP)
        root_eqs.align_to(gen_factored, UP)
        self.play(
            FadeIn(root_eqs),
            VGroup(system, const_dec).animate.next_to(root_eqs, DOWN, LARGE_BUFF),
        )
        self.wait(2)
        self.play(FadeOut(root_eqs))

        example_constants = [4, 7, 9, 5]
        for const in example_constants:
            self.play(
                ChangeDecimalToValue(const_dec, const),
                run_time=3,
            )
            self.wait()

    def add_axes(self):
        x_range = (-1, 6)
        y_range = (-3, 11)
        axes = Axes(
            x_range, y_range,
            axis_config=dict(include_tip=False, numbers_to_exclude=[]),
            widith=abs(op.sub(*x_range)),
            height=abs(op.sub(*y_range)),
        )
        axes.set_height(FRAME_HEIGHT - 1)
        axes.to_edge(RIGHT)
        axes.x_axis.add_numbers(font_size=24)
        axes.x_axis.numbers[1].set_opacity(0)

        self.add(axes)
        return axes


class FactsAboutRootsToCoefficients(RootCoefScene):
    coefs = [-5, 14, -7, 1]
    coef_plane_config = {
        "x_range": (-15.0, 15.0, 5.0),
        "y_range": (-10, 10, 5),
        "background_line_style": {
            "stroke_color": GREY,
            "stroke_width": 1.0,
        },
        "height": 20,
        "width": 30,
    }
    root_plane_config = {
        "x_range": (-1.0, 6.0),
        "y_range": (-3.0, 3.0),
        "background_line_style": {
            "stroke_color": BLUE_E,
            "stroke_width": 1.0,
        }
    }
    plane_height = 3.5
    planes_center = 1.5 * DOWN

    def construct(self):
        # Play with coefficients, confined to real axis
        self.wait()
        self.add_constant_decimals()
        self.add_graph()
        self.lock_coef_imag = True
        self.wait(note="Move around c0")
        self.lock_coef_imag = False

        self.decimal_poly.clear_updaters()
        self.play(
            FadeOut(self.decimal_poly, DOWN),
            FadeOut(self.graph_group, DOWN),
        )

        # Show the goal
        self.add_system()
        self.add_solver_functions()

        # Why that's really weird
        self.play(
            self.coef_system.animate.set_opacity(0.2),
            self.root_system[1:].animate.set_opacity(0.2),
        )
        self.wait(note="Show loops with c0")

        # Why something like this must be possible
        brace = Brace(self.coef_system, RIGHT)
        properties = VGroup(
            Text("Continuous"),
            Text("Symmetric"),
        )
        properties.arrange(DOWN, buff=MED_LARGE_BUFF)
        properties.next_to(brace, RIGHT)

        self.play(
            GrowFromCenter(brace),
            self.root_system.animate.set_opacity(0),
            self.coef_system.animate.set_opacity(1),
        )
        self.wait()
        for words in properties:
            self.play(Write(words, run_time=1))
            self.wait()

        self.swap_root_symbols()
        self.wait(note="Physically swap roots")

        # What this implies about our functions
        brace.generate_target()
        brace.target.rotate(PI)
        brace.target.next_to(self.root_system, LEFT)
        left_group = VGroup(properties, self.coef_system)
        left_group.generate_target()
        left_group.target.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        left_group.target.set_height(1)
        left_group.target.to_corner(UL)
        left_group.target.set_opacity(0.5)

        self.play(
            MoveToTarget(brace, path_arc=PI / 2),
            MoveToTarget(left_group),
            self.root_system.animate.set_opacity(1)
        )
        self.wait()

        restriction = VGroup(
            Text("Cannot(!) be both"),
            Text("Continuous and single-valued", t2c={
                "Continuous": YELLOW,
                "single-valued": BLUE,
            })
        )
        restriction.scale(0.8)
        restriction.arrange(DOWN)
        restriction.next_to(brace, LEFT)

        self.play(FadeIn(restriction))
        self.wait(note="Move c0, emphasize multiplicity of outputs")

        # Impossibility result
        words = Text("Cannot be built from ")
        symbols = Tex(
            "+,\\,", "-,\\,", "\\times,\\,", "/,\\,", "\\text{exp}\\\\",
            "\\sin,\\,", "\\cos,\\,", "| \\cdot |,\\,", "\\dots",
        )
        impossibility = VGroup(words, symbols)
        impossibility.arrange(RIGHT)
        impossibility.match_width(restriction)
        impossibility.next_to(restriction, DOWN, aligned_edge=RIGHT)
        impossible_rect = SurroundingRectangle(impossibility)
        impossible_rect.set_stroke(RED, 2)

        arrow = Tex("\\Downarrow", font_size=36)
        arrow.next_to(impossible_rect, UP, SMALL_BUFF)
        restriction.generate_target()
        restriction.target.scale(1.0).next_to(arrow, UP, SMALL_BUFF)

        self.play(
            FadeIn(impossibility[0]),
            FadeIn(arrow),
            ShowCreation(impossible_rect),
            MoveToTarget(restriction),
        )
        for symbol in symbols:
            self.wait(0.25)
            self.add(symbol)
        self.wait()

        # Show discontinuous example
        to_fade = VGroup(
            restriction[0],
            restriction[1].get_part_by_text("Continuous and"),
            arrow,
            impossibility,
            impossible_rect,
        )
        to_fade.save_state()
        self.play(*(m.animate.fade(0.8) for m in to_fade))

        root_tracers = VGroup(*(d.tracer for d in self.root_dots))
        self.remove(root_tracers)
        self.continuous_roots = False
        self.root_dots[0].set_fill(BLUE)
        self.r_dot_labels[0].set_fill(BLUE)
        self.root_dots[1].set_fill(GREEN)
        self.r_dot_labels[1].set_fill(GREEN)
        self.wait(note="Show discontinuous behavior")
        self.add(self.get_tracers(self.root_dots))
        self.wait(note="Turn tracers back on")
        self.continuous_roots = True

        # Represent as a multivalued function
        f_name = "\\text{cubic\\_solve}"
        t2c = dict([
            (f"{sym}_{i}", color)
            for i in range(3)
            for sym, color in [
                ("r", self.root_color),
                ("c", self.coef_color),
            ]
        ])
        t2c[f_name] = GREY_A
        mvf = Tex(
            f"{f_name}(c_0, c_1, c_2)\\\\", "=\\\\", "\\left\\{r_0, r_1, r_2\\right\\}",
            tex_to_color_map=t2c
        )
        mvf.get_part_by_tex("=").rotate(PI / 2).match_x(mvf.slice_by_tex(None, "="))
        mvf.slice_by_tex("left").match_x(mvf.get_part_by_tex("="))
        mvf.move_to(self.root_system, LEFT)

        self.play(
            TransformMatchingShapes(self.root_system, mvf),
            restriction[1].get_part_by_text("single-valued").animate.fade(0.8),
        )
        self.wait(note="Labeling is an artifact")
        self.play(FadeOut(self.r_dot_labels))
        self.wait()

    def add_c_labels(self):
        super().add_c_labels()
        self.c_dot_labels[2].clear_updaters()
        self.c_dot_labels[2].add_updater(
            lambda l: l.next_to(l.dot, DL, buff=0)
        )
        return self.c_dot_labels

    def add_constant_decimals(self):
        dummy = "+10.00"
        polynomial = Tex(
            f"x^3 {dummy}x^2 {dummy}x {dummy}",
            isolate=[dummy],
            font_size=40,
        )
        polynomial.next_to(self.coef_poly, UP, LARGE_BUFF)
        decimals = DecimalNumber(100, include_sign=True, edge_to_fix=LEFT).replicate(3)
        for dec, part in zip(decimals, polynomial.get_parts_by_tex(dummy)):
            dec.match_height(part)
            dec.move_to(part, LEFT)
            part.set_opacity(0)
            polynomial.add(dec)
        polynomial.decimals = decimals

        def update_poly(polynomial):
            for dec, coef in zip(polynomial.decimals, self.get_coefs()[-2::-1]):
                dec.set_value(coef.real)
            polynomial.decimals.set_fill(RED, 1)
            return polynomial

        update_poly(polynomial)
        VGroup(polynomial[0], decimals[0]).next_to(
            polynomial[2], LEFT, SMALL_BUFF, aligned_edge=DOWN
        )

        self.play(FadeIn(polynomial, UP, suspend_updating=True))
        polynomial.add_updater(update_poly)
        self.decimal_poly = polynomial

    def add_graph(self):
        self.decimal_poly
        axes = Axes(
            (0, 6), (-4, 10),
            axis_config=dict(tick_size=0.025),
            width=3, height=2,
        )
        axes.set_height(2)
        axes.move_to(self.root_plane)
        axes.to_edge(UP, buff=SMALL_BUFF)

        graph = always_redraw(
            lambda: axes.get_graph(
                lambda x: poly(x, self.get_coefs()).real
            ).set_stroke(BLUE, 2)
        )

        root_dots = GlowDot()
        root_dots.add_updater(lambda d: d.set_points([
            axes.c2p(r.real, 0)
            for r in self.get_roots()
            if abs(r.imag) < 1e-5
        ]))

        arrow = Arrow(self.decimal_poly.get_right(), axes)

        graph_group = Group(axes, graph, root_dots)

        self.play(
            ShowCreation(arrow),
            FadeIn(graph_group, shift=UR),
        )

        graph_group.add(arrow)
        self.graph_group = graph_group

    def add_system(self):
        c_parts = self.get_c_symbols(self.coef_poly)
        system = get_symmetric_system(
            (f"c_{i}" for i in reversed(range(len(self.coef_dots)))),
            signed=True,
        )
        system.scale(0.8)
        system.next_to(self.coef_poly, UP, LARGE_BUFF)
        system.align_to(self.coef_plane, LEFT)

        self.add(system)

        kw = dict(lag_ratio=0.8, run_time=2.5)
        self.play(
            LaggedStart(*(
                TransformFromCopy(c, line[0])
                for c, line in zip(c_parts, system)
            ), **kw),
            LaggedStart(*(
                FadeIn(line[1:], lag_ratio=0.1)
                for line in system
            ), **kw)
        )
        self.add(system)
        self.coef_system = system
        self.wait()

    def add_solver_functions(self):
        func_name = "\\text{cubic\\_solve}"
        t2c = dict((
            (f"{sym}_{i}", color)
            for i in range(3)
            for sym, color in [
                ("c", self.coef_color),
                ("r", self.root_color),
                (func_name, GREY_A),
            ]
        ))
        kw = dict(tex_to_color_map=t2c)
        lines = VGroup(*(
            Tex(f"r_{i} = {func_name}_{i}(c_0, c_1, c_2)", **kw)
            for i in range(3)
        ))
        lines.scale(0.8)
        lines.arrange(DOWN, aligned_edge=LEFT)
        lines.match_y(self.coef_system)
        lines.align_to(self.root_plane, LEFT)

        kw = dict(lag_ratio=0.7, run_time=2)
        self.play(
            LaggedStart(*(
                TransformFromCopy(r, line[0])
                for r, line in zip(self.get_r_symbols(self.root_poly), lines)
            ), **kw),
            LaggedStart(*(
                FadeIn(line[1:], lag_ratio=0.1)
                for line in lines
            ), **kw),
        )
        self.add(lines)
        self.root_system = lines
        self.wait()

    def swap_root_symbols(self):
        system = self.coef_system
        cs = [f"c_{i}" for i in reversed(range(len(self.coef_dots)))]
        rs = [f"r_{{{i}}}" for i in range(len(self.root_dots))]

        for tup in [(1, 2, 0), (2, 0, 1), (0, 1, 2)]:
            rs = [f"r_{{{i}}}" for i in tup]
            alt_system = get_symmetric_system(cs, roots=rs, signed=True)
            alt_system.replace(system)
            self.play(*(
                TransformMatchingTex(
                    l1, l2,
                    path_arc=PI / 2,
                    lag_ratio=0.01,
                    run_time=2
                )
                for l1, l2 in zip(system, alt_system)
            ))
            self.remove(system)
            system = alt_system
            self.add(system)
            self.wait()
        self.coef_system = system


class ComplicatedSingleValuedFunction(Scene):
    def construct(self):
        pass


class SolvabilityChart(Scene):
    def construct(self):
        # Preliminary terms
        frame = self.camera.frame
        frame.set_height(10)

        words = self.get_words(frame)
        equations = self.get_equations(words)
        s_words = self.get_solvability_words(equations)
        gen_form_words = Text("General form")
        gen_form_words.match_x(equations, LEFT)
        gen_form_words.match_y(s_words, UP)
        lines = self.get_lines(
            rows=VGroup(s_words, *words),
            cols=VGroup(words, equations, *s_words),
        )
        row_lines, col_lines = lines
        marks = self.get_marks(equations, s_words)

        # Shift colums
        marks[1].save_state()
        s_words[1].save_state()
        frame.save_state()
        frame.set_height(9, about_edge=DL)
        frame.shift(LEFT)
        VGroup(marks[1], s_words[1]).next_to(col_lines[1], RIGHT, MED_LARGE_BUFF)

        solvable_word = TexText("Can you solve\\\\for $x$?")
        solvable_word.move_to(s_words[1], DOWN)

        # Cover rects
        cover_rect = Rectangle()
        cover_rect.set_fill(BLACK, 1)
        cover_rect.set_stroke(BLACK, 0)
        cover_rect.replace(frame, stretch=True)
        cover_rect.add(VectorizedPoint(cover_rect.get_top() + 0.025 * UP))
        cover_rect.move_to(row_lines[1], UL).shift(LEFT)
        right_cover_rect = cover_rect.copy()
        right_cover_rect.next_to(s_words[1], RIGHT, buff=MED_LARGE_BUFF)
        right_cover_rect.match_y(frame)

        self.add(words, equations, solvable_word)
        self.add(row_lines, col_lines[:2])
        self.add(right_cover_rect, cover_rect)

        # Axes
        axes = self.get_axes(frame)
        coefs = np.array([1, 0.5, 0, 0, 0, 0])
        coef_tracker = ValueTracker(coefs)
        get_coefs = coef_tracker.get_value
        graph = always_redraw(lambda: axes.get_graph(
            lambda x: poly(x, get_coefs()),
            stroke_color=BLUE,
            stroke_width=2,
        ))
        root_dots = GlowDot()
        root_dots.add_updater(lambda m: m.set_points([
            axes.c2p(r.real, 0)
            for r in coefficients_to_roots(get_coefs())
            if abs(r.imag) < 1e-5 and abs(r.real) < 5
        ]))
        self.add(axes)

        # Linear equation
        tex_kw = dict(tex_to_color_map=self.get_tex_to_color_map())
        lin_solution = Tex("x = {-{b} \\over {a}}", **tex_kw)
        lin_solution.scale(1.2)
        lin_solution.next_to(equations[0], DOWN, buff=2.0)

        self.wait()
        self.play(
            ShowCreation(graph),
            FadeIn(root_dots, rate_func=squish_rate_func(smooth, 0.3, 0.4)),
        )
        self.wait()
        self.play(TransformMatchingShapes(
            equations[0].copy(), lin_solution
        ))
        self.play(Write(marks[1][0]))
        self.wait()

        # Quadratic
        quadratic_formula = get_quadratic_formula(lhs="x = ", **tex_kw)
        quadratic_formula.next_to(equations[1], DOWN, buff=2.0)
        new_coefs = 0.2 * np.array([*roots_to_coefficients([-3, 2]), 0, 0, 0])

        self.play(
            cover_rect.animate.move_to(row_lines[2], UL).shift(LEFT),
            FadeOut(lin_solution, DOWN),
        )
        self.play(coef_tracker.animate.set_value(new_coefs))
        self.wait()
        self.play(TransformMatchingShapes(
            equations[1].copy(), quadratic_formula,
        ))
        self.play(Write(marks[1][1]))
        self.wait()

        # Cubic
        key_to_color = dict([
            (TransformMatchingShapes.get_mobject_key(Tex(c)[0][0]), color)
            for c, color in self.get_tex_to_color_map().items()
        ])
        full_cubic = get_full_cubic_formula(lhs="x = ")
        full_cubic.set_width(9)
        full_cubic.next_to(equations[2], DOWN, buff=1.0).shift(LEFT)
        for sm in full_cubic[0]:
            key = TransformMatchingShapes.get_mobject_key(sm)
            sm.set_color(key_to_color.get(key, WHITE))
        new_coefs = 0.05 * np.array([*roots_to_coefficients([-4, -1, 3]), 0, 0])

        self.play(
            cover_rect.animate.move_to(row_lines[3], UL).shift(LEFT),
            FadeOut(quadratic_formula, DOWN),
        )
        self.play(coef_tracker.animate.set_value(new_coefs))
        self.wait()
        self.play(TransformMatchingShapes(
            equations[2].copy(), full_cubic,
            run_time=2
        ))
        self.wait()

        # Embed
        self.embed()

    def get_words(self, frame):
        words = VGroup(*map(Text, (
            "Linear",
            "Quadratic",
            "Cubic",
            "Quartic",
            "Quintic",
            "Sextic",
        )))
        words.add(Tex("\\vdots"))
        words.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        words.next_to(frame.get_corner(DL), UR, buff=1.0)
        words.shift(0.5 * LEFT)
        words[-1].match_x(words[-2])
        return words

    def get_equations(self, words):
        kw = dict(tex_to_color_map=self.get_tex_to_color_map())
        equations = VGroup(
            Tex("{a}x + {b} = 0", **kw),
            Tex("{a}x^2 + {b}x + {c} = 0", **kw),
            Tex("{a}x^3 + {b}x^2 + {c}x + {d} = 0", **kw),
            Tex("{a}x^4 + \\cdots + {d}x + {e} = 0", **kw),
            Tex("{a}x^5 + \\cdots + {e}x + {f} = 0", **kw),
            Tex("{a}x^6 + \\cdots + {f}x + {g} = 0", **kw),
            Tex("\\vdots", **kw),
        )
        equations.arrange(DOWN, aligned_edge=LEFT)
        equations.next_to(words, RIGHT, LARGE_BUFF)
        for eq, word in zip(equations, words):
            dy = word[-1].get_bottom()[1] - eq[0][0].get_bottom()[1]
            eq.shift(dy * UP)
        equations[-1].match_y(words[-1])
        equations[-1].match_x(equations[-2])
        return equations

    def get_solvability_words(self, equations):
        operations = ["+", "-", "\\times", "/", "\\sqrt[n]{\\quad}"]
        arith, radicals = (
            "$" + " ,\\, ".join(operations[s]) + "$"
            for s in (slice(None, -1), slice(None))
        )
        s_words = VGroup(
            TexText("Solvable", " using\\\\", arith),
            TexText("Solvable", " using\\\\", radicals),
            TexText("Solvable\\\\", "numerically"),
        )
        s_words.arrange(RIGHT, buff=LARGE_BUFF, aligned_edge=UP)
        s_words.next_to(equations, UR, buff=MED_LARGE_BUFF)
        s_words.shift(MED_LARGE_BUFF * RIGHT)

        return s_words

    def get_lines(self, rows, cols, color=GREY_A, width=2):
        row_line = Line(cols.get_left(), cols.get_right())
        row_lines = row_line.replicate(len(rows) - 1)
        for r1, r2, rl in zip(rows, rows[1:], row_lines):
            rl.match_y(midpoint(r1.get_bottom(), r2.get_top()))

        col_line = Line(rows.get_top(), rows.get_bottom())
        col_lines = col_line.replicate(len(cols) - 1)
        for c1, c2, cl in zip(cols, cols[1:], col_lines):
            cl.match_x(midpoint(c1.get_right(), c2.get_left()))

        col_lines[0].match_height(Group(row_lines, Point(col_lines.get_bottom())), about_edge=DOWN)

        lines = VGroup(row_lines, col_lines)
        lines.set_stroke(color, width)
        return lines

    def get_marks(self, equations, solvability_words):
        pre_marks = [
            "cxxxxxx",
            "ccccxxx",
            "ccccccc",
        ]
        marks = VGroup(*(
            VGroup(*(
                Checkmark() if pm == 'c' else Exmark()
                for pm in pm_list
            ))
            for pm_list in pre_marks
        ))
        for mark_group, s_word in zip(marks, solvability_words):
            mark_group.match_x(s_word)
            for mark, eq in zip(mark_group, equations):
                mark.match_y(eq)
        return marks

    def get_axes(self, frame):
        axes = Axes((-5, 5), (-5, 5), height=10, width=10)
        axes.set_width(4)
        axes.next_to(frame.get_corner(DR), UL)
        axes.add(Tex("x", font_size=24).next_to(axes.x_axis.get_right(), DOWN, SMALL_BUFF))
        axes.add(Tex("y", font_size=24).next_to(axes.y_axis.get_top(), LEFT, SMALL_BUFF))
        return axes

    def get_tex_to_color_map(self):
        chars = "abcdefg"
        colors = color_gradient([RED_B, RED_C, RED_D], len(chars))
        return dict(
            (f"{{{char}}}", color)
            for char, color in zip(chars, colors)
        )


class StudySqrt(RadicalScene):
    n = 2
    c = 2.0

    def construct(self):
        # Show simple equation
        kw = dict(tex_to_color_map={"c": self.coef_color})
        equations = VGroup(
            Tex("x^2 - c = 0", **kw),
            Tex("x =", "\\sqrt{c}", **kw),
        )
        equations.arrange(DOWN, buff=MED_LARGE_BUFF)
        equations.to_edge(UP)

        self.wait()
        self.play(FadeIn(equations[0], UP))
        self.wait()
        self.play(
            TransformMatchingShapes(
                equations[0].copy(),
                equations[1],
                path_arc=PI / 2,
            )
        )
        self.wait()

        sqrt_label = equations[1][1:].copy()

        # Add decimal labels, show square roots of real c
        c_label = VGroup(
            Tex("c = ", tex_to_color_map={"c": self.coef_color}),
            DecimalNumber(self.c),
        )
        c_label.arrange(RIGHT, aligned_edge=DOWN)
        c_label.next_to(self.coef_poly, UP, buff=1.5)
        c_label[1].add_updater(lambda d: d.set_value(self.get_c().real))

        def update_root_dec(root_dec):
            c_real = self.get_c().real
            root_dec.unit = "" if c_real > 0 else "i"
            root_dec.set_value((-1)**root_dec.index * math.sqrt(abs(c_real)))

        r_labels = VGroup(*(
            VGroup(Tex(f"r_{i}", "="), DecimalNumber(self.c, include_sign=True))
            for i in range(2)
        ))
        for i, r_label in enumerate(r_labels):
            r_label.arrange(RIGHT)
            r_label[1].align_to(r_label[0][0][0], DOWN)
            r_label[0][0].set_color(self.root_color)
            r_label[1].index = i
            r_label[1].add_updater(update_root_dec)

        r_labels.arrange(DOWN, buff=0.75)
        r_labels.match_x(self.root_plane)
        r_labels.match_y(c_label)

        sqrt_arrow = Arrow(self.coef_plane, self.root_plane)
        sqrt_arrow.match_y(c_label)

        self.play(
            FadeIn(c_label),
            ShowCreation(sqrt_arrow),
            sqrt_label.animate.next_to(sqrt_arrow, UP),
            FadeOut(equations),
        )
        self.play(FadeIn(r_labels))
        self.lock_coef_imag = True
        self.wait(note="Move c along real line")
        self.lock_coef_imag = False

        # Focus just on one root
        root_dots = self.root_dots
        root_tracers = VGroup(*(d.tracer for d in root_dots))
        root_labels = self.r_dot_labels
        self.play(
            r_labels[0].animate.match_y(c_label),
            FadeOut(r_labels[1], DOWN),
            root_dots[1].animate.set_opacity(0.5),
            root_dots[1].tracer.animate.set_stroke(opacity=0.5),
            root_labels[1].animate.set_opacity(0.5),
        )
        self.wait()
        r_label = r_labels[0]

        # Vary the angle of c
        self.show_angle_variation(c_label, r_label, sqrt_arrow)
        self.play(
            sqrt_arrow.animate.set_width(1.75).match_y(self.root_plane),
            MaintainPositionRelativeTo(sqrt_label, sqrt_arrow)
        )

        # Discontinuous square root
        option = VGroup(
            TexText("One option:", color=BLUE),
            TexText("\\\\Make", " $\\sqrt{\\quad}$", " single-valued, but discontinuous", font_size=36),
        )
        option.arrange(DOWN, buff=0.5)
        option[1][1].align_to(option[1][0], DOWN)
        option.to_edge(UP, buff=MED_SMALL_BUFF)

        np_tex = Code("Python: numpy.sqrt(c)")
        np_tex.match_width(self.root_plane)
        np_tex.next_to(self.root_plane, UP)

        sqrt_dot = Dot(**self.dot_style)
        sqrt_dot.set_color(BLUE)
        sqrt_dot.add_updater(lambda d: d.move_to(self.root_plane.n2p(np.sqrt(self.get_c()))))
        sqrt_label = Code("sqrt(c)")
        sqrt_label.scale(0.75)
        sqrt_label.add_updater(lambda m: m.next_to(sqrt_dot, UR, buff=0))

        self.play(FadeIn(option, UP))
        self.wait()
        self.remove(root_tracers)
        self.play(
            self.root_dots.animate.set_opacity(0),
            FadeOut(self.root_poly),
            FadeOut(root_labels),
            FadeIn(np_tex),
            FadeIn(sqrt_dot),
            FadeIn(sqrt_label),
        )
        self.wait(note="Show discontinuity")

        # Taylor series
        taylor_series = Tex(
            "\\sqrt{x} \\approx",
            "1",
            "+ \\frac{1}{2}(x - 1)",
            "- \\frac{1}{8}(x - 1)^2",
            "+ \\frac{1}{16}(x - 1)^3",
            "- \\frac{5}{128}(x - 1)^4",
            "+ \\cdots",
            font_size=36,
        )
        ts_title = Text("What about a Taylor series?")
        ts_title.set_color(GREEN)
        ts_group = VGroup(ts_title, taylor_series)
        ts_group.arrange(DOWN, buff=MED_LARGE_BUFF)
        ts_group.to_edge(UP, buff=MED_SMALL_BUFF)

        def f(x, n):
            return sum((
                gen_choose(1 / 2, k) * (x - 1)**k
                for k in range(n)
            ))

        brace = Brace(taylor_series[1:-1], DOWN, buff=SMALL_BUFF)
        upper_f_label = brace.get_tex("f_4(x)", buff=SMALL_BUFF)
        upper_f_label.set_color(GREEN)

        f_dot = Dot(**self.dot_style)
        f_dot.set_color(GREEN)
        f_dot.add_updater(lambda d: d.move_to(self.root_plane.n2p(f(self.get_c(), 4))))
        f_label = Tex("f_4(x)", font_size=24, color=GREEN)
        f_label.add_updater(lambda m: m.next_to(f_dot, DL, buff=SMALL_BUFF))

        self.play(
            FadeOut(np_tex),
            FadeIn(ts_group, UP),
            FadeOut(option, UP),
        )
        self.wait()
        self.play(
            GrowFromCenter(brace),
            FadeIn(upper_f_label, 0.5 * DOWN),
        )
        self.wait()
        self.play(
            TransformFromCopy(upper_f_label, f_label),
            GrowFromPoint(f_dot, upper_f_label.get_center()),
        )
        self.wait()

        anims = [
            brace.animate.become(Brace(taylor_series[1:], DOWN, buff=SMALL_BUFF))
        ]
        for label in (upper_f_label, f_label):
            new_label = Tex("f_{50}(x)")
            new_label.replace(label, 1)
            new_label.match_style(label)
            anims.append(Transform(label, new_label, suspend_updating=False))
        self.play(*anims)
        f_dot.clear_updaters()
        f_dot.add_updater(lambda d: d.move_to(self.root_plane.n2p(f(self.get_c(), 50))))
        self.wait()

        disc = Circle(radius=self.root_plane.x_axis.get_unit_size())
        disc.move_to(self.root_plane.n2p(1))
        disc.set_stroke(BLUE_B, 2)
        disc.set_fill(BLUE_B, 0.2)
        self.play(FadeIn(disc))
        self.wait()

        # Back to normal
        ts_group.add(brace, upper_f_label)
        root_labels.set_opacity(1)
        self.play(
            FadeOut(ts_group, UP),
            *map(FadeOut, (disc, f_label, f_dot, sqrt_label, sqrt_dot)),
            FadeIn(root_labels),
            FadeIn(self.root_poly),
            root_dots.animate.set_opacity(1),
        )
        self.add(root_tracers, *root_labels)
        self.wait()

    def show_angle_variation(self, c_label, r_label, arrow):
        angle_color = TEAL
        self.last_theta = 0

        def get_theta():
            angle = np.log(self.get_c()).imag
            diff = angle - self.last_theta
            diff = (diff + PI) % TAU - PI
            self.last_theta += diff
            return self.last_theta

        circle = Circle(radius=self.coef_plane.x_axis.get_unit_size())
        circle.set_stroke(angle_color, 1)
        circle.move_to(self.coef_plane.get_origin())

        left_exp_label, right_exp_label = (
            self.get_exp_label(
                get_theta=func,
                color=angle_color
            ).move_to(label[-1], DL)
            for label, func in [
                (c_label, get_theta),
                (r_label, lambda: get_theta() / self.n),
            ]
        )

        below_arrow_tex = MTex(
            "e^{x} \\rightarrow e^{x /" + str(self.n) + "}",
            font_size=36,
            tex_to_color_map={"\\theta": angle_color},
        )
        below_arrow_tex.next_to(arrow, DOWN)

        angle_labels = VGroup(
            self.get_angle_label(self.coef_dots[0], self.coef_plane, "\\theta", get_theta),
            self.get_angle_label(
                self.root_dots[0], self.root_plane,
                f"\\theta / {self.n}",
                lambda: get_theta() / self.n,
            ),
        )

        self.add(circle, self.coef_dots)
        self.lock_coef_norm = True
        self.tie_roots_to_coefs()
        self.play(
            FadeIn(circle),
            FadeIn(angle_labels),
            self.coef_dots[0].animate.move_to(self.coef_plane.n2p(1)),
        )
        self.wait()
        self.play(
            FadeOut(c_label[-1], UP),
            FadeOut(r_label[-1], UP),
            FadeIn(left_exp_label, UP),
            FadeIn(right_exp_label, UP),
        )
        self.wait(note="Rotate c a bit")
        self.play(Write(below_arrow_tex))
        self.wait(note="Show full rotation, then two rotations")

        # Remove stuff
        self.play(LaggedStart(*map(FadeOut, (
            circle, angle_labels,
            left_exp_label, right_exp_label,
            c_label[:-1], r_label[:-1],
            below_arrow_tex,
        ))))
        self.lock_coef_norm = False

    def get_exp_label(self, get_theta, color=GREEN):
        result = Tex("e^{", "2\\pi i \\cdot", "0.00}")
        decimal = DecimalNumber()
        decimal.replace(result[2], dim_to_match=1)
        result.replace_submobject(2, decimal)
        result.add_updater(lambda m: m.assemble_family())
        result.add_updater(lambda m: m[-1].set_color(color))
        result.add_updater(lambda m: m[-1].set_value(get_theta() / TAU))
        return result


class CubeRootBehavior(StudySqrt):
    n = 3
    c = 1.0

    def construct(self):
        arrows, labels = self.get_radical_labels()
        self.add(arrows, labels)

        c_label = Tex("c = ", "1.00", tex_to_color_map={"c": self.coef_color})
        r_label = Tex("r_0 = ", "1.00", tex_to_color_map={"r_0": self.root_color})
        c_label.match_x(self.coef_plane)
        c_label.to_edge(UP, buff=1.0)
        r_label.match_x(self.root_plane)
        r_label.match_y(c_label)
        right_arrow_group = VGroup(arrows[0], labels[0])
        right_arrow_group.save_state()

        self.wait()
        self.wait()
        self.play(
            right_arrow_group.animate.to_edge(UP),
            *map(FadeIn, (c_label, r_label))
        )
        self.show_angle_variation(c_label, r_label, right_arrow_group[0])
        self.play(Restore(right_arrow_group))

    def add_labeled_arrow(self):
        pass


class FifthRootBehavior(CubeRootBehavior):
    n = 5


class SummarizeRootsToCyclesBehavior(Scene):
    def construct(self):
        pass


# Analyze the cubic formula


class Cubic(RootCoefScene):
    coefs = [1, -1, 0, 1]

    def construct(self):
        pass


class AmbientRootSwapping(RootCoefScene):
    n_swaps = 0

    def construct(self):
        for x in range(self.n_swaps):
            k = random.randint(2, 5)
            indices = random.choice(list(it.combinations(range(5), k)))
            self.swap_roots(*indices)
            self.wait()

        self.embed()


class CubicFormulaTest(CubicFormula):
    def construct(self):
        pass
        # self.embed()
