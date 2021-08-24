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
        "x_range": (-2.0, 2.0),
        "y_range": (-2.0, 2.0),
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

    def setup(self):
        self.root_dots = VGroup()
        self.coef_dots = VGroup()

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
        self.root_poly = root_poly
        self.coef_poly = coef_poly
        self.poly_equal_sign = equals

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
            for coef in coefs
        )

    def add_root_dots(self, roots=None):
        if roots is None:
            roots = self.get_roots_of_unity()
        self.set_roots(roots)
        self.add(self.root_dots)

    def add_coef_dots(self, coefs=None):
        if coefs is None:
            coefs = [0] * self.degree
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
        ]

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
                root = unordered_roots[np.argmin([
                    abs(old_root - ur)
                    for ur in unordered_roots
                ])]
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
            for dot in (*self.root_dots, *self.coef_dots)
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

    def add_dot_labels(self, sym, dots, buff=0.05):
        labels = VGroup()
        for i, dot in enumerate(dots):
            label = Tex(f"{sym}_{i}", font_size=24)
            label.set_fill(dot.get_fill_color())
            label.set_stroke(BLACK, 3, background=True)
            label.dot = dot
            label.add_updater(lambda m: m.next_to(m.dot, UR, buff=buff))
            labels.add(label)
        self.add(labels)
        return labels

    def add_r_labels(self):
        self.r_dot_labels = self.add_dot_labels("r", self.root_dots)

    def add_c_labels(self):
        self.c_dot_labels = self.add_dot_labels("c", self.coef_dots)

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

    def swap_roots(self, i, j, run_time=2, wait_time=1):
        self.play(Swap(
            self.root_dots[i],
            self.root_dots[j],
            run_time=3
        ))
        self.wait(wait_time)

    def rotate_coefs(self, indicies, center_z=0, run_time=5, wait_time=1):
        self.play(*(
            Rotate(
                self.coef_dots[i], TAU,
                about_point=self.coef_plane.n2p(center_z),
                run_time=run_time
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
                search_set=(*self.root_dots, *self.coef_dots),
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


class TestRootCoefScene(RootCoefScene):
    def construct(self):
        self.add_planes()

        # Add dots
        coefs = [3, 2, 1, 0, -1]
        # coefs = [-1, 0]
        roots = coefficients_to_roots(coefs)
        self.add_root_dots(roots)
        self.add_coef_dots(coefs)
        self.add_tracers()

        self.add_r_labels()
        self.add_c_labels()

        # Animate
        # self.swap_roots(0, 1)
        # self.rotate_coef(0)

        # self.add_root_lines()
        # self.add_coef_lines()

        # self.tie_roots_to_coefs()

        # Sample animations
        # self.rotate_coef(0)

        # self.tie_coefs_to_roots()
        # for i, j in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2)]:
        #     arrows = self.get_root_swap_arrows(i, j)
        #     self.play(*map(ShowCreation, arrows))
        #     self.swap_roots(i, j)
        #     self.play(FadeOut(arrows))

        # self.embed()


# Scenes

class ComplexNewtonsMethod(Scene):
    coefs = [1, 1, 1, 0, 0]
    poly_tex = "x^5 + x^2 + x + 1"
    plane_config = {
        "x_range": (-2, 2),
        "y_range": (-2, 2),
        "height": 8,
        "width": 8,
    }
    rule_font_size = 42
    seed = complex(1, 1)
    guess_color = YELLOW
    step_arrow_width = 5

    def construct(self):
        self.add_plane()
        self.add_title()
        self.add_z0_def()
        self.add_rule()
        self.find_root()

        self.embed()

    def add_plane(self):
        plane = ComplexPlane(**self.plane_config)
        plane.add_coordinate_labels(font_size=24)
        plane.to_edge(RIGHT, buff=0)
        self.plane = plane
        self.add(plane)

    def add_title(self):
        title = TexText("Newton's method", font_size=60)
        title.move_to(midpoint(self.plane.get_left(), LEFT_SIDE))
        title.to_edge(UP)

        poly = Tex("P(x) = ", self.poly_tex, "= 0 ")
        poly.match_width(title)
        poly.next_to(title, DOWN, buff=MED_LARGE_BUFF)
        poly.set_fill(GREY_A)
        title.add(poly)

        self.title = title
        self.add(title)

    def add_z0_def(self):
        seed_text = Text("(Arbitrary seed)")
        z0_def = Tex(
            "z_0 = 1 + i",
            tex_to_color_map={"z_0": self.guess_color},
            font_size=self.rule_font_size
        )
        z0_group = VGroup(seed_text, z0_def)
        z0_group.arrange(DOWN)
        z0_group.next_to(self.title, DOWN, buff=LARGE_BUFF)

        guess_dot = Dot(self.plane.n2p(self.seed), color=self.guess_color)

        guess = DecimalNumber(self.seed, num_decimal_places=3)
        # guess.set_stroke(BLACK, 8, background=True)
        guess.add_updater(
            lambda m: m.set_value(self.plane.p2n(
                guess_dot.get_center()
            )).set_fill(self.guess_color).add_background_rectangle()
        )
        guess.add_updater(lambda m: m.next_to(guess_dot, UP))

        self.play(
            Write(seed_text, run_time=1),
            FadeIn(z0_def),
        )
        self.play(
            FadeTransform(z0_def[0].copy(), guess_dot),
            FadeIn(guess),
        )
        self.wait()

        self.z0_def = z0_def
        self.guess_dot = guess_dot
        self.guess = guess

    def add_rule(self):
        rule = Tex(
            """
                z_1 =
                z_0 - {P(z_0) \\over P'(z_0)}
            """,
            tex_to_color_map={
                "z_1": self.guess_color,
                "z_0": self.guess_color
            },
            font_size=self.rule_font_size,
        )
        rule.next_to(self.z0_def, DOWN, buff=LARGE_BUFF)

        rule.n = 0
        rule.zns = rule.get_parts_by_tex("z_0")
        rule.znp1 = rule.get_parts_by_tex("z_1")

        self.rule = rule

        self.play(
            FadeTransformPieces(self.z0_def[0].copy(), rule.zns),
            FadeIn(rule),
        )
        self.wait()

    def find_root(self):
        for x in range(5):
            self.root_search_step(self.guess_dot)

    def root_search_step(self, dot, tol=1e-2):
        dot_step_anims = self.get_dot_step_anims(dot)
        diff_rect = SurroundingRectangle(
            self.rule.slice_by_tex("-"),
            buff=0.1,
            stroke_color=GREY_A,
            stroke_width=1,
        )

        self.play(
            ShowCreation(diff_rect),
            dot_step_anims[0],
        )
        self.play(
            dot_step_anims[1],
            FadeOut(diff_rect),
            *self.cycle_rule_entries_anims(),
            run_time=2
        )
        self.wait()

    def get_dot_step_anims(self, dot):
        plane = self.plane
        z0 = plane.p2n(dot.get_center())
        z1 = z0 - poly(z0, self.coefs) / dpoly(z0, self.coefs)

        arrow = Arrow(
            plane.n2p(z0), plane.n2p(z1),
            buff=0,
            stroke_width=self.step_arrow_width
        )

        return [
            ShowCreation(arrow),
            AnimationGroup(
                dot.animate.move_to(plane.n2p(z1)),
                FadeOut(arrow),
            )
        ]

    def cycle_rule_entries_anims(self):
        rule = self.rule
        rule.n += 1
        zns = VGroup(*(
            Tex(
                f"z_{rule.n}", font_size=self.rule_font_size
            ).replace(old_zn).match_color(old_zn)
            for old_zn in rule.zns
        ))
        znp1 = Tex(f"z_{rule.n + 1}", font_size=self.rule_font_size)
        znp1.move_to(rule.znp1)
        znp1.match_color(rule.znp1[0])

        result = (
            FadeOut(rule.zns),
            FadeTransformPieces(rule.znp1, zns),
            FadeIn(znp1, 0.5 * RIGHT)
        )
        rule.zns = zns
        rule.znp1 = znp1
        return result


class ComplexNewtonsMethodManySeeds(Scene):
    def construct(self):
        self.add_plane()
        self.add_title()
        self.add_z0_def()
        self.add_rule()
        self.find_root()

    def add_z0_def(self):
        pass

    def root_search_step(self):
        pass
