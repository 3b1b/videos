from manim_imports_ext import *

from _2022.quintic import coefficients_to_roots
from _2022.quintic import roots_to_coefficients
from _2022.quintic import dpoly
from _2022.quintic import poly


ROOT_COLORS_BRIGHT = [RED, GREEN, BLUE, YELLOW, MAROON_B]
ROOT_COLORS_DEEP = ["#440154", "#3b528b", "#21908c", "#5dc963", "#29abca"]
CUBIC_COLORS = [RED_E, TEAL_E, BLUE_E]


def glow_dot(point, r_min=0.05, r_max=0.15, color=YELLOW, n=20, opacity_mult=1.0):
    result = VGroup(*(
        Dot(point, radius=interpolate(r_min, r_max, a))
        for a in np.linspace(0, 1, n)
    ))
    result.set_fill(color, opacity=opacity_mult / n)
    return result


def get_newton_rule(font_size=36, var="z", **kwargs):
    terms = [f"{var}_n", f"{var}_{{n + 1}}"]
    t0, t1 = terms
    return Tex(
        t1, "=", t0, "-",
        "{P(", t0, ")", "\\over ", "P'(", t0, ")}",
        font_size=36,
        **kwargs
    )


def coefs_to_poly_string(coefs):
    n = len(coefs) - 1
    tex_str = "" if coefs[-1] == 1 else str(int(coefs[-1]))
    tex_str += f"z^{{{n}}}"
    for c, k in zip(coefs[-2::-1], it.count(n - 1, -1)):
        if c == 0:
            continue
        if isinstance(c, complex):
            num_str = "({:+}".format(int(c.real))
            num_str += "+ {:+})".format(int(c.imag))
        else:
            num_str = "{:+}".format(int(c))
        if abs(c) == 1 and k > 0:
            num_str = num_str[:-1]
        tex_str += num_str
        if k == 0:
            continue
        elif k == 1:
            tex_str += "z"
        else:
            tex_str += f"z^{{{k}}}"
    return tex_str


def get_figure(image_name, person_name, year_text, height=3, label_direction=DOWN):
    image = ImageMobject(image_name)
    image.set_height(height)
    rect = SurroundingRectangle(image, buff=0)
    rect.set_stroke(WHITE, 2)
    name = Text(f"{person_name}", font_size=36)
    name.set_color(GREY_A)
    year_label = Text(f"{year_text}", font_size=30)
    year_label.match_color(name)
    year_label.next_to(name, DOWN, buff=0.2)
    VGroup(name, year_label).next_to(image, label_direction)
    return Group(rect, image, name, year_label)


class NewtonFractal(Mobject):
    CONFIG = {
        "shader_folder": "newton_fractal",
        "shader_dtype": [
            ('point', np.float32, (3,)),
        ],
        "colors": ROOT_COLORS_DEEP,
        "coefs": [1.0, -1.0, 1.0, 0.0, 0.0, 1.0],
        "scale_factor": 1.0,
        "offset": ORIGIN,
        "n_steps": 30,
        "julia_highlight": 0.0,
        "max_degree": 5,
        "saturation_factor": 0.0,
        "opacity": 1.0,
        "black_for_cycles": False,
        "is_parameter_space": False,
    }

    def __init__(self, plane, **kwargs):
        super().__init__(
            scale_factor=plane.get_x_unit_size(),
            offset=plane.n2p(0),
            **kwargs,
        )
        self.replace(plane, stretch=True)

    def init_data(self):
        self.data = {
            "points": np.array([UL, DL, UR, DR]),
        }

    def init_uniforms(self):
        super().init_uniforms()
        self.set_colors(self.colors)
        self.set_julia_highlight(self.julia_highlight)
        self.set_coefs(self.coefs)
        self.set_scale(self.scale_factor)
        self.set_offset(self.offset)
        self.set_n_steps(self.n_steps)
        self.set_saturation_factor(self.saturation_factor)
        self.set_opacity(self.opacity)
        self.uniforms["black_for_cycles"] = float(self.black_for_cycles)
        self.uniforms["is_parameter_space"] = float(self.is_parameter_space)

    def set_colors(self, colors):
        self.uniforms.update({
            f"color{n}": np.array(color_to_rgba(color))
            for n, color in enumerate(colors)
        })
        return self

    def set_julia_highlight(self, value):
        self.uniforms["julia_highlight"] = value

    def set_coefs(self, coefs, reset_roots=True):
        full_coefs = [*coefs] + [0] * (self.max_degree - len(coefs) + 1)
        self.uniforms.update({
            f"coef{n}": np.array([coef.real, coef.imag], dtype=np.float64)
            for n, coef in enumerate(map(complex, full_coefs))
        })
        if reset_roots:
            self.set_roots(coefficients_to_roots(coefs), False)
        self.coefs = coefs
        return self

    def set_roots(self, roots, reset_coefs=True):
        self.uniforms["n_roots"] = float(len(roots))
        full_roots = [*roots] + [0] * (self.max_degree - len(roots))
        self.uniforms.update({
            f"root{n}": np.array([root.real, root.imag], dtype=np.float64)
            for n, root in enumerate(map(complex, full_roots))
        })
        if reset_coefs:
            self.set_coefs(roots_to_coefficients(roots), False)
        self.roots = roots
        return self

    def set_scale(self, scale_factor):
        self.uniforms["scale_factor"] = scale_factor
        return self

    def set_offset(self, offset):
        self.uniforms["offset"] = np.array(offset)
        return self

    def set_n_steps(self, n_steps):
        self.uniforms["n_steps"] = float(n_steps)
        return self

    def set_saturation_factor(self, saturation_factor):
        self.uniforms["saturation_factor"] = float(saturation_factor)
        return self

    def set_opacities(self, *opacities):
        for n, opacity in enumerate(opacities):
            self.uniforms[f"color{n}"][3] = opacity
        return self

    def set_opacity(self, opacity, recurse=True):
        self.set_opacities(*len(self.roots) * [opacity])
        return self


class MetaNewtonFractal(NewtonFractal):
    CONFIG = {
        "coefs": [-1.0, 0.0, 0.0, 1.0],
        "colors": [*ROOT_COLORS_DEEP[::2], BLACK, BLACK],
        "fixed_roots": [-1, 1],
        "n_roots": 3,
        "black_for_cycles": True,
        "is_parameter_space": True,
        "n_steps": 300,
    }

    def init_uniforms(self):
        super().init_uniforms()
        self.set_fixed_roots(self.fixed_roots)

    def set_fixed_roots(self, roots):
        super().set_roots(roots, reset_coefs=False)
        self.uniforms["n_roots"] = 3.0


# Scenes


class AmbientRootFinding(Scene):
    def construct(self):
        pass


class PragmaticOrigins(Scene):
    title = "Pragmatic origins"
    include_pi = False

    def construct(self):
        # Title
        title = Text(self.title, font_size=72)
        title.set_stroke(BLACK, 5, background=True)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        underline = Underline(title, buff=-0.05)
        underline.insert_n_curves(30)
        underline.set_stroke(BLUE, width=[0, 3, 3, 3, 0])
        underline.scale(1.5)

        # Axes
        axes = NumberPlane(
            x_range=(-3, 3),
            y_range=(-4, 4),
            width=6,
            height=8,
            background_line_style={
                "stroke_color": GREY_A,
                "stroke_width": 1,
            }
        )
        axes.set_height(5.0)
        axes.to_corner(DL)
        axes.shift(0.5 * UP)

        coefs = np.array([2, -3, 1, -2, -1, 1], dtype=np.float)
        roots = [
            r.real
            for r in coefficients_to_roots(coefs)
            if abs(r.imag) < 1e-2
        ]
        roots.sort()
        coefs *= 0.2

        solve = TexText("Solve $f(x) = 0$", font_size=36)
        solve.next_to(axes, UP, aligned_edge=LEFT)
        expr = Tex("f(x) = x^5 - x^4 - 2x^3 + x^2 -3x + 2")
        expr.match_width(axes)
        expr.next_to(axes, DOWN)
        graph_x_range = (-2, 2.4)
        graph = axes.get_graph(
            lambda x: poly(x, coefs),
            x_range=graph_x_range
        )
        graph.set_stroke(BLUE, [0, *50 * [4], 0])
        root_dots = VGroup(*(
            glow_dot(axes.c2p(root, 0))
            for root in roots
        ))
        root_eqs = VGroup()
        root_groups = VGroup()
        for i, root, dot in zip(it.count(1), roots, root_dots):
            lhs = Tex(f"x_{i} = ")
            rhs = DecimalNumber(root, num_decimal_places=3)
            rhs.set_color(YELLOW)
            eq = VGroup(lhs, rhs)
            eq.arrange(RIGHT, aligned_edge=DOWN)
            rhs.align_to(lhs.family_members_with_points()[0], DOWN)
            root_eqs.add(eq)
            root_groups.add(VGroup(eq, dot))
        root_eqs.arrange(RIGHT, buff=LARGE_BUFF)
        root_eqs.next_to(axes, RIGHT, aligned_edge=UP)

        self.add(axes)
        self.add(solve)
        self.add(expr)

        # Pi
        if self.include_pi:
            morty = Mortimer(height=2)
            morty.to_corner(DR)
            self.play(PiCreatureSays(
                morty, "How do you\nfind theses?",
                target_mode="tease",
                bubble_config={
                    "width": 4,
                    "height": 2.5,
                }
            ))

        # Animations
        self.add(underline, title)
        self.play(
            ShowCreation(underline),
        )
        self.wait()

        alphas = [inverse_interpolate(*graph_x_range, root) for root in roots]
        self.play(
            ShowCreation(graph, rate_func=linear),
            *(
                FadeIn(
                    rg,
                    rate_func=squish_rate_func(rush_from, a, min(a + 0.2, 1))
                )
                for rg, a in zip(root_groups, alphas)
            ),
            run_time=4,
        )
        self.wait()


class SeekingRoots(PragmaticOrigins):
    title = "Seeking roots"
    include_pi = True


class AskAboutComplexity(Scene):
    def construct(self):
        self.add(FullScreenRectangle())

        question = Text("What does this complexity reflect?")
        question.set_width(FRAME_WIDTH - 2)
        question.to_edge(UP)
        self.add(question)

        screen = ScreenRectangle()
        screen.set_height(6.0)
        screen.set_fill(BLACK, 1)
        screen.next_to(question, DOWN)
        self.add(screen)


class WhoCares(TeacherStudentsScene):
    def construct(self):
        self.students.refresh_triangulation()
        screen = self.screen
        screen.set_height(4, about_edge=UL)
        screen.set_fill(BLACK, 1)
        image = ImageMobject("RealNewtonStill")
        image.replace(screen)

        self.add(screen)
        self.add(image)

        self.wait()
        self.play(LaggedStart(
            PiCreatureSays(
                self.students[1], "Ooh, quintics...",
                target_mode="thinking",
                look_at=self.screen,
                bubble_config={
                    "direction": LEFT,
                    "width": 4,
                    "height": 2,
                }
            ),
            self.teacher.change("happy"),
            self.students[0].change("thinking", screen),
            self.students[2].change("sassy", screen),
            lag_ratio=0.1,
        ))
        self.wait(3)
        self.play(LaggedStart(
            PiCreatureSays(
                self.students[2], "Who cares?",
                target_mode="tired",
                bubble_config={
                    "direction": LEFT,
                    "width": 4,
                    "height": 3,
                }
            ),
            self.teacher.change("guilty"),
            self.students[0].change("confused", screen),
            RemovePiCreatureBubble(
                self.students[1],
                look_at=self.students[2].eyes,
                target_mode="erm",
            ),
            lag_ratio=0.1,
        ))
        self.wait(2)
        self.teacher_says(
            "Who doesn't",
            target_mode="hooray",
            bubble_config={"height": 3, "width": 4},
            added_anims=[self.change_students("pondering", "pondering", "confused")]
        )
        self.wait(3)


class SphereExample(Scene):
    def construct(self):
        # Shape
        axes = ThreeDAxes(z_range=(-4, 4))
        axes.shift(IN)
        sphere = Sphere(radius=1.0)
        # sphere = TexturedSurface(sphere, "EarthTextureMap", "NightEarthTextureMap")
        sphere.move_to(axes.c2p(0, 0, 0))
        sphere.set_gloss(1.0)
        sphere.set_opacity(0.5)
        sphere.sort_faces_back_to_front(DOWN)
        mesh = SurfaceMesh(sphere, resolution=(21, 11))
        mesh.set_stroke(BLUE, 0.5, 0.5)
        sphere = Group(sphere, mesh)

        frame = self.camera.frame
        frame.reorient(20, 80)
        frame.move_to(2 * RIGHT)
        light = self.camera.light_source

        self.add(axes)
        self.add(sphere)

        frame.add_updater(
            lambda m, dt: m.increment_theta(1 * dt * DEGREES)
        )

        # Expression
        equation = Tex(
            "1.00", "\\,x^2", "+",
            "1.00", "\\,y^2", "+",
            "1.00", "\\,z^2", "=",
            "1.00",
        )
        decimals = VGroup()
        for i in range(0, len(equation), 3):
            decimal = DecimalNumber(1.0, edge_to_fix=RIGHT)
            decimal.replace(equation[i])
            equation.replace_submobject(i, decimal)
            decimals.add(decimal)
            decimal.add_updater(lambda m: m.fix_in_frame())
        equation.fix_in_frame()
        equation.to_corner(UR)
        self.add(equation)

        # Animations
        light.move_to([-10, -10, 20])
        self.wait()
        self.play(
            ChangeDecimalToValue(decimals[3], 9.0),
            VFadeInThenOut(SurroundingRectangle(decimals[3]).fix_in_frame()),
            sphere.animate.scale(3),
            run_time=3
        )
        self.wait()
        self.play(
            ChangeDecimalToValue(decimals[2], 4.0),
            VFadeInThenOut(SurroundingRectangle(decimals[2]).fix_in_frame()),
            sphere.animate.stretch(0.5, 2),
            run_time=3
        )
        self.wait()
        self.play(
            ChangeDecimalToValue(decimals[0], 9.0),
            VFadeInThenOut(SurroundingRectangle(decimals[0]).fix_in_frame()),
            sphere.animate.stretch(1 / 3, 0),
            run_time=3
        )
        self.wait(10)


class ExamplePixels(Scene):
    def construct(self):
        pixels = Square().get_grid(5, 5, buff=0)
        pixels.set_height(2)
        pixels.to_corner(UL)
        pixels.set_stroke(WHITE, 1)
        pixels.set_fill(BLACK, 1)
        self.add(pixels)

        y, x = 1066, 1360

        endpoint = np.array([x, -y, 0], dtype=np.float)
        endpoint *= FRAME_HEIGHT / 2160
        endpoint += np.array([-FRAME_WIDTH / 2, FRAME_HEIGHT / 2, 0])
        lines = VGroup(
            Line(pixels.get_corner(UR), endpoint),
            Line(pixels.get_corner(DL), endpoint),
        )
        lines.set_stroke(WHITE, 2)
        self.add(lines)

        def match_values(pixels, values):
            for pixel, value in zip(pixels, it.chain(*values)):
                value = value[::-1]
                pixel.set_fill(rgb_to_color(value / 255))

        values = np.load(
            os.path.join(get_directories()["data"], "sphere_pixel_values.npy")
        )
        match_values(pixels, values[0])
        # for value in values[60::60]:
        for value in values[1:]:
            # pixels.generate_target()
            # match_values(pixels.target, value)
            # self.play(MoveToTarget(pixels))
            match_values(pixels, value)
            self.wait(1 / 60)


class CurvesDefiningFonts(Scene):
    def construct(self):
        # Setup
        frame = self.camera.frame

        chars = TexText("When a computer\\\\renders text...")[0]
        chars.set_width(FRAME_WIDTH - 3)
        chars.refresh_unit_normal()
        chars.refresh_triangulation()
        filled_chars = chars.copy()
        filled_chars.insert_n_curves(50)
        chars.set_stroke(WHITE, 0.5)
        chars.set_fill(opacity=0.0)

        dot_groups = VGroup()
        line_groups = VGroup()
        for char in chars:
            dots = VGroup()
            lines = VGroup()
            for a1, h, a2 in char.get_bezier_tuples():
                for pair in (a1, h), (h, a2):
                    lines.add(Line(
                        *pair,
                        stroke_width=0.25,
                        # dash_length=0.0025,
                        stroke_color=YELLOW,
                    ))
                for point in (a1, h, a2):
                    dots.add(Dot(point, radius=0.005))
            dot_groups.add(dots)
            line_groups.add(lines)

        dot_groups.set_fill(BLUE, opacity=0)

        self.play(ShowIncreasingSubsets(filled_chars, run_time=1, rate_func=linear))
        self.wait()

        # Zoom in on one letter
        char_index = 2
        char = chars[char_index]
        lines = line_groups[char_index]
        dots = dot_groups[char_index]
        char.refresh_bounding_box()
        frame.generate_target()
        frame.target.set_height(char.get_height() * 2)
        frame.target.move_to(char.get_bottom(), DOWN)
        frame.target.shift(0.1 * char.get_height() * DOWN)
        self.play(
            MoveToTarget(frame),
            filled_chars.animate.set_opacity(0.2),
            FadeIn(chars),
            ShowCreation(line_groups, rate_func=linear),
            dot_groups.animate.set_opacity(1),
            run_time=5,
        )
        for group in (line_groups, dot_groups):
            group.remove(*group[0:char_index - 1])
            group.remove(*group[char_index + 2:])
        self.wait()

        # Pull out one curve
        char.become(CurvesAsSubmobjects(char))

        index = 26
        curve = char[index]
        sublines = lines[2 * index:2 * index + 2]
        subdots = dots[3 * index:3 * index + 3]

        curve_group = VGroup(curve, sublines, subdots)
        curve_group.set_stroke(background=True)
        curve_group.generate_target()
        curve_group.save_state()
        curve_group.target.scale(3)
        curve_group.target.next_to(frame.get_top(), DOWN, buff=0.15)
        curve_group.target.shift(0.3 * LEFT)
        for dot in curve_group.target[2]:
            dot.scale(1 / 2)

        labels = VGroup(*(
            Tex(f"P_{i}").set_height(0.05)
            for i in range(3)
        ))
        for label, dot, vect in zip(labels, curve_group.target[2], [LEFT, UP, UP]):
            label.insert_n_curves(20)
            label.next_to(dot, vect, buff=0.025)
            label.match_color(dot)

        self.play(
            MoveToTarget(curve_group),
            *(
                GrowFromPoint(label, curve_group.get_center())
                for label in labels
            )
        )

        equation = Tex(
            "(1-t)^{2} P_0 +2(1-t)t P_1 +t^2 P_2",
            tex_to_color_map={
                "P_0": BLUE,
                "P_1": BLUE,
                "P_2": BLUE,
            }
        )
        equation.set_height(0.07)
        equation.next_to(curve_group, RIGHT, buff=0.25)
        equation.insert_n_curves(20)

        poly_label = Text("Polynomial")
        poly_label.insert_n_curves(20)
        poly_label.set_width(2)
        poly_label.apply_function(
            lambda p: [
                p[0],
                p[1] - 0.2 * p[0]**2,
                p[2],
            ]
        )
        poly_label.rotate(30 * DEGREES)
        poly_label.match_height(curve_group)
        poly_label.scale(0.8)
        poly_label.move_to(curve, DR)
        poly_label.shift(0.01 * UL)

        self.play(
            ShowCreationThenDestruction(curve.copy().set_color(PINK), run_time=2),
            Write(poly_label, stroke_width=0.5)
        )
        self.play(
            LaggedStart(*(
                TransformFromCopy(
                    labels[i],
                    equation.get_part_by_tex(f"P_{i}").copy(),
                    remover=True
                )
                for i in range(3)
            )),
            FadeIn(equation, rate_func=squish_rate_func(smooth, 0.5, 1)),
            run_time=2,
        )
        self.wait()
        self.add(curve_group.copy())
        self.play(Restore(curve_group))
        self.wait()


class PlayingInFigma(ExternallyAnimatedScene):
    pass


class RasterizingBezier(Scene):
    def construct(self):
        # Add curve and pixels
        self.add(FullScreenRectangle())

        curve = SVGMobject("bezier_example")[0]
        curve.set_width(FRAME_WIDTH - 3)
        curve.set_stroke(WHITE, width=1.0)
        curve.set_fill(opacity=0)
        curve.to_edge(DOWN, buff=1)
        curve.insert_n_curves(10)  # To better uniformize it

        thick_curve = curve.copy()
        thick_curve.set_stroke(YELLOW, 30.0)
        thick_curve.reverse_points()

        pixels = Square().get_grid(90 // 2, 160 // 2, buff=0, fill_rows_first=False)
        pixels.set_height(FRAME_HEIGHT)
        pixels.set_stroke(WHITE, width=0.25)

        # I fully recognize the irony is implementing this without
        # solving polynomials, but I'm happy to be inificient on runtime
        # to just code up the quickest thing I can think of.
        samples = np.array([curve.pfp(x) for x in np.linspace(0, 1, 100)])
        sw_tracker = ValueTracker(0.15)
        get_sw = sw_tracker.get_value

        for pixel in pixels:
            diffs = samples - pixel.get_center()
            dists = np.apply_along_axis(lambda p: np.dot(p, p), 1, diffs)
            index = np.argmin(dists)
            if index == 0 or index == len(samples) - 1:
                pixel.dist = np.infty
            else:
                pixel.dist = dists[index]

        def update_pixels(pixels):
            for pixel in pixels:
                pixel.set_fill(
                    YELLOW,
                    0.5 * clip(10 * (get_sw() - pixel.dist), 0, 1)
                )

        update_pixels(pixels)

        fake_pixels = pixels.copy()
        fake_pixels.set_stroke(width=0)
        fake_pixels.set_fill(GREY_E, 1)

        self.add(thick_curve)
        self.wait()
        self.add(fake_pixels, pixels)
        self.play(
            FadeIn(fake_pixels),
            ShowCreation(pixels),
            lag_ratio=10 / len(pixels),
            run_time=4
        )
        self.remove(thick_curve)
        self.wait()

        # Pixel
        pixel = pixels[725].deepcopy()
        pixel.set_fill(opacity=0)
        label = TexText("Pixel $\\vec{\\textbf{p}}$")
        label.refresh_triangulation()
        label.set_fill(YELLOW)
        label.set_stroke(BLACK, 4, background=True)
        label.next_to(pixel, UL, buff=LARGE_BUFF)
        label.shift_onto_screen()
        arrow = Arrow(label, pixel, buff=0.1, stroke_width=3.0)
        arrow.set_color(YELLOW)

        self.play(
            FadeIn(label),
            ShowCreation(arrow),
            pixel.animate.set_stroke(YELLOW, 2.0),
        )
        pixels.add_updater(update_pixels)
        self.play(sw_tracker.animate.set_value(2.0), run_time=2)
        self.play(sw_tracker.animate.set_value(0.2), run_time=2)
        pixels.suspend_updating()
        self.play(ShowCreation(curve))

        # Show P(t) value
        ct = VGroup(Tex("\\vec{\\textbf{c}}(")[0], DecimalNumber(0), Tex(")")[0])
        ct.arrange(RIGHT, buff=0)
        ct.add_updater(lambda m: m.set_stroke(BLACK, 4, background=True))
        t_tracker = ValueTracker(0)
        get_t = t_tracker.get_value
        P_dot = Dot(color=GREEN)
        globals().update(locals())
        ct[1].add_updater(lambda m: m.set_value(get_t()))
        ct[1].next_to(ct[0], RIGHT, buff=0)
        P_dot.add_updater(lambda m: m.move_to(curve.pfp(get_t() / 2)))
        ct.add_updater(lambda m: m.move_to(P_dot).shift(
            (0.3 - 0.5 * get_t() * (1 - get_t())) * rotate_vector(np.array([-3, 1, 0]), -0.8 * get_t() * PI)
        ))
        curve_copy = curve.copy()
        curve_copy.pointwise_become_partial(curve, 0, 0.5)
        curve_copy.set_points(curve_copy.get_points_without_null_curves())
        curve_copy.set_stroke(YELLOW, 3.0)

        self.play(
            VFadeIn(ct),
            ApplyMethod(t_tracker.set_value, 1.0, run_time=3),
            ShowCreation(curve_copy, run_time=3),
            VFadeIn(P_dot),
        )
        new_ct = Tex("\\vec{\\textbf{c}}(", "t", ")")
        new_ct.move_to(ct, LEFT)
        new_ct.set_stroke(BLACK, 4, background=True)
        self.play(FadeTransformPieces(ct, new_ct))
        ct = new_ct
        self.wait()

        # Show distance
        graph_group = self.get_corner_graph_group(pixel, curve)
        bg_rect, axes, y_label, graph = graph_group

        t_tracker = ValueTracker(0)
        dist_line = Line()
        dist_line.set_stroke(TEAL, 5)
        dist_line.add_updater(lambda l: l.put_start_and_end_on(
            pixel.get_center(),
            curve_copy.pfp(t_tracker.get_value())
        ))

        dist_lines = VGroup()
        graph_v_lines = VGroup()
        for t in np.linspace(0, 1, 20):
            t_tracker.set_value(t)
            dist_lines.add(dist_line.update().copy().clear_updaters())
            graph_v_lines.add(axes.get_v_line(
                axes.input_to_graph_point(t, graph)
            ))
        dist_lines.set_stroke(RED, 1, opacity=1.0)
        graph_v_lines.set_stroke(RED, 1, opacity=1.0)
        t_tracker.set_value(0)

        self.play(
            *map(FadeIn, graph_group[:-1]),
        )
        self.play(
            FadeIn(dist_lines, lag_ratio=1),
            FadeIn(graph_v_lines, lag_ratio=1),
            run_time=4
        )
        self.wait()
        t_tracker.set_value(0.0)
        self.play(
            VFadeIn(dist_line, rate_func=squish_rate_func(smooth, 0, 0.25)),
            ApplyMethod(t_tracker.set_value, 1.0),
            ShowCreation(graph),
            run_time=3,
        )
        self.play(dist_line.animate.set_stroke(RED, 1.0))
        self.wait()

        # Show width again
        pixels.resume_updating()
        self.play(sw_tracker.animate.set_value(1.5), run_time=2)
        self.play(sw_tracker.animate.set_value(0.5), run_time=1)
        pixels.suspend_updating()
        self.wait()

        # Show derivative
        deriv_graph_group = self.get_deriv_graph_group(graph_group)
        d_graph = deriv_graph_group[-1]
        d_graph.set_points_smoothly([d_graph.pfp(x) for x in np.linspace(0, 1, 20)])
        deriv_axes = deriv_graph_group[1]

        t_tracker = ValueTracker(0)
        get_t = t_tracker.get_value
        tan_line = always_redraw(
            lambda: axes.get_tangent_line(
                get_t(), graph, length=3,
            ).set_stroke(
                color=MAROON_B,
                width=1.0,
                opacity=clip(20 * get_t() * (1 - get_t()), 0, 1)
            )
        )

        self.play(*map(FadeIn, deriv_graph_group[:-1]))
        self.add(tan_line)
        self.play(
            t_tracker.animate.set_value(1),
            ShowCreation(d_graph),
            run_time=4
        )
        self.remove(tan_line)
        self.wait()

        points = graph.get_points()
        min_point = points[np.argmin([p[1] for p in points])]
        min_line = Line(min_point, [min_point[0], deriv_axes.c2p(0, 0)[1], 0])
        min_line.set_stroke(WHITE, 1)

        question = Text("What is\nthis value?", font_size=30)
        question.to_corner(DR)
        arrow = Arrow(
            question.get_left(), min_line.get_bottom(), stroke_width=3,
            buff=0.1
        )

        self.play(ShowCreation(min_line))
        self.play(
            Write(question),
            ShowCreation(arrow),
        )
        self.wait()

    def get_corner_graph_group(self, pixel, curve, t_range=(0, 0.5)):
        axes = Axes(
            x_range=(0, 1, 0.2),
            y_range=(0, 20, 5),
            height=3,
            width=5,
            axis_config={"include_tip": False}
        )
        axes.to_corner(UR, buff=SMALL_BUFF)
        y_label = Tex(
            "&\\text{Distance}^2\\\\",
            "&||\\vec{\\textbf{p}} - \\vec{\\textbf{c}}(t)||^2",
            font_size=24,
        )
        # For future transition
        y_label = VGroup(VectorizedPoint(y_label.get_left()), *y_label)
        y_label.next_to(axes.y_axis.get_top(), RIGHT, aligned_edge=UP)
        y_label.shift_onto_screen(buff=MED_SMALL_BUFF)

        graph = axes.get_graph(lambda t: get_norm(
            pixel.get_center() - curve.pfp(interpolate(*t_range, t))
        )**2)
        graph.set_stroke(RED, 2)

        bg_rect = BackgroundRectangle(axes, buff=SMALL_BUFF)
        result = VGroup(bg_rect, axes, y_label, graph)

        return result

    def get_deriv_graph_group(self, graph_group):
        top_bg_rect, top_axes, top_y_label, top_graph = graph_group

        axes = Axes(
            x_range=top_axes.x_range,
            y_range=(-60, 60, 10),
            height=top_axes.get_height(),
            width=top_axes.get_width(),
            axis_config={"include_tip": False}
        )
        axes.to_corner(DR, buff=SMALL_BUFF)
        axes.shift((top_axes.c2p(0, 0) - axes.c2p(0, 0))[0] * RIGHT)
        dt = 1e-5
        f = top_graph.underlying_function
        globals().update(locals())
        graph = axes.get_graph(lambda t: (f(t + dt) - f(t)) / dt)
        graph.set_stroke(MAROON_B)
        # Dumb hack, not sure why it's needed
        graph.get_points()[:133] += 0.015 * UP

        y_label = VGroup(Tex("\\frac{d}{dt}", font_size=24), top_y_label[2].copy())
        y_label.arrange(RIGHT, buff=0.05)
        y_label.next_to(axes.y_axis.get_top(), RIGHT, buff=2 * SMALL_BUFF)

        bg_rect = BackgroundRectangle(VGroup(axes, graph), buff=SMALL_BUFF)
        bg_rect.stretch(1.05, 1, about_edge=DOWN)

        result = VGroup(bg_rect, axes, y_label, graph)

        return result


class WriteThisIsPolynomial(Scene):
    def construct(self):
        text = TexText("(Some polynomial in $t$)", font_size=24)
        self.play(Write(text))
        self.wait()


class DontWorryAboutDetails(TeacherStudentsScene):
    CONFIG = {
        "background_color": BLACK,
    }

    def construct(self):
        screen = self.screen
        screen.set_height(4, about_edge=UL)
        screen.set_fill(BLACK, 1)
        image1, image2 = [
            ImageMobject(f"RasterizingBezier_{i}").replace(screen)
            for i in range(1, 3)
        ]

        frame = self.camera.frame
        frame.save_state()
        frame.replace(image1)

        self.add(screen, image1)

        self.play(Restore(frame))

        # Student asks about what the function is.
        self.student_says(
            TexText("Wait, what is that\\\\function exactly?"),
            look_at=image1,
            index=2,
            added_anims=[
                self.students[0].change("confused", image1),
                self.students[1].change("confused", image1),
            ]
        )
        self.play(self.teacher.change("tease"))
        self.wait(2)
        self.play(
            self.students[0].change("maybe", image1),
        )
        self.play(
            self.students[1].change("erm", image1),
        )
        self.wait(3)

        self.teacher_says(
            TexText("Just some\\\\polynomial"),
            bubble_config={
                "width": 4,
                "height": 3,
            },
            added_anims=[self.change_students("confused", "maybe", "pondering")]
        )
        self.wait()
        self.look_at(image1)
        self.play(
            frame.animate.replace(image1),
            RemovePiCreatureBubble(self.teacher),
            run_time=2
        )
        self.wait()

        # Image 2
        self.remove(image1)
        self.add(image2)
        self.play(Restore(frame))

        self.play_all_student_changes(
            "confused",
            look_at=image1,
        )
        self.teacher_says(
            Tex("P(x) = 0"),
            target_mode="tease",
            bubble_config={
                "width": 3,
                "height": 3,
            }
        )
        self.wait(4)
        self.play(
            RemovePiCreatureBubble(self.teacher, target_mode="raise_right_hand", look_at=image1),
            self.change_students(
                *3 * ["pondering"],
                look_at=image1,
            ),
            FadeOut(image2),
        )
        self.wait(4)


class ShowManyGraphs(Scene):
    def construct(self):
        # Add plots
        root_groups = [
            (-2, 6),
            (-5, 0, 3),
            (-7, -2, 3, 8),
            (-5, 1, 5, complex(0, 1), complex(0, -1)),
        ]
        coef_groups = list(map(roots_to_coefficients, root_groups))
        scalars = [0.5, 0.2, 0.01, -0.01]
        colors = [BLUE_C, BLUE_D, BLUE_B, RED]
        plots = Group(*(
            self.get_plot(coefs, scalar, color)
            for coefs, scalar, color in zip(coef_groups, scalars, colors)
        ))
        plots.arrange_in_grid(v_buff=0.5)
        axes, graphs, root_dots = [
            Group(*(plot[i] for plot in plots))
            for i in range(3)
        ]

        self.play(
            LaggedStartMap(FadeIn, axes, lag_ratio=0.3),
            LaggedStartMap(ShowCreation, graphs, lag_ratio=0.3),
            run_time=3,
        )
        self.play(
            LaggedStart(*(
                FadeIn(dot, scale=0.1)
                for dot in it.chain(*root_dots)
            ), lag_ratio=0.1)
        )

        self.add(plots)
        self.wait()

        quadratic, cubic, quartic, quintic = plots
        for plot in plots:
            plot.save_state()

        # Show quadratic
        kw = {"tex_to_color_map": {
            "{a}": BLUE_B,
            "{b}": BLUE_C,
            "{c}": BLUE_D,
            "{d}": TEAL_E,
            "{e}": TEAL_D,
            "{f}": TEAL_C,
            "{p}": BLUE_B,
            "{q}": BLUE_C,
            "\\text{root}": YELLOW,
            "r_1": YELLOW,
            "r_2": YELLOW,
            "+": WHITE,
            "-": WHITE,
        }}
        quadratic.generate_target()
        quadratic.target.set_height(6)
        quadratic.target.center().to_edge(LEFT)
        equation = Tex("{a}x^2 + {b}x + {c} = 0", **kw)
        equation.next_to(quadratic.target, UP)
        form = Tex(
            "r_1, r_2 = {-{b} \\pm \\sqrt{\\,{b}^2 - 4{a}{c}} \\over 2{a}}",
            **kw
        )
        form.next_to(quadratic.target, RIGHT, buff=MED_LARGE_BUFF)
        form_name = Text("Quadratic formula")
        form_name.match_width(form)
        form_name.next_to(form, UP, LARGE_BUFF)

        randy = Randolph(height=2)
        randy.flip()
        randy.next_to(form, RIGHT)
        randy.align_to(quadratic.target, DOWN)
        randy.shift_onto_screen()

        self.play(
            MoveToTarget(quadratic),
            Write(equation),
            *map(FadeOut, plots[1:]),
            FadeIn(randy),
        )
        self.play(randy.change("hooray"))
        self.play(
            TransformMatchingShapes(
                VGroup(*(
                    equation.get_part_by_tex(f"{{{c}}}")
                    for c in "abc"
                )).copy(),
                form,
                lag_ratio=0,
                run_time=2,
            ),
            randy.animate.look_at(form),
            FadeIn(form_name),
            FlashAround(form_name),
        )
        self.play(Blink(randy))
        self.wait()

        # Coco sidenote
        form_group = VGroup(form_name, form)
        form_group.save_state()
        form_group.set_stroke(BLACK, 5, background=True)
        plot_group = Group(quadratic, equation)
        plot_group.save_state()

        self.play(
            plot_group.animate.shift(4 * LEFT).set_opacity(0),
            form_group.animate.to_corner(UR),
            FadeOut(randy),
        )

        pixar_image = ImageMobject("PixarCampus")
        pixar_image.set_height(FRAME_HEIGHT + 4)
        pixar_image.to_corner(UL, buff=0)
        pixar_image.shift(LEFT)
        pixar_image.add_updater(lambda m, dt: m.shift(0.1 * dt * LEFT))

        coco_logo = ImageMobject("Coco_logo")
        coco_logo.set_width(4)
        coco_logo.match_y(form)
        coco_logo.to_edge(RIGHT, buff=LARGE_BUFF)
        arrow = Arrow(form.copy().to_edge(LEFT), coco_logo, buff=0.3, stroke_width=10)

        self.add(pixar_image, *self.mobjects)
        self.play(FadeIn(pixar_image))
        self.wait(6)
        self.add(coco_logo, *self.mobjects)
        self.play(
            FadeOut(pixar_image),
            form_group.animate.to_corner(UL),
            FadeIn(randy),
            ShowCreation(arrow),
            FadeIn(coco_logo),
        )

        over_trillion = Tex("> 1{,}000{,}000{,}000{,}000")[0]
        over_trillion.next_to(form, RIGHT)
        over_trillion.shift(3 * DOWN)
        form_copies = form[4:].replicate(50)
        self.play(
            ShowIncreasingSubsets(over_trillion, run_time=1),
            randy.change("thinking", over_trillion),
            LaggedStart(*(
                FadeOut(form_copy, 4 * DOWN)
                for form_copy in form_copies
            ), lag_ratio=0.15, run_time=5)
        )
        self.play(
            FadeOut(over_trillion),
            FadeOut(coco_logo),
            FadeOut(arrow),
            randy.change("happy"),
            Restore(form_group),
            Restore(plot_group),
        )

        self.embed()

        # Cubic
        low_fade_rect = BackgroundRectangle(
            Group(quartic, quintic),
            buff=0.01,
            fill_opacity=0.95,
        )
        cubic_eq = Tex("x^3 + {p}x + {q} = 0", **kw)
        cubic_eq.next_to(cubic, LEFT, LARGE_BUFF, aligned_edge=UP)
        cubic_eq.shift_onto_screen()
        cubic_name = TexText("Cubic\\\\", "Formula")
        cubic_name.to_corner(UL)
        cubic_form = Tex(
            "\\text{root}", "=",
            "\\sqrt[3]{\\,-{{q} \\over 2} + \\sqrt{\\, {{q}^2 \\over 4} + {{p}^3 \\over 27}} }+",
            "\\sqrt[3]{\\,-{{q} \\over 2} - \\sqrt{\\, {{q}^2 \\over 4} + {{p}^3 \\over 27}} }",
            **kw,
        )
        cubic_form.set_width(7)
        cubic_form.next_to(cubic_eq, DOWN, buff=1.25)
        cubic_form.to_edge(LEFT)
        cubic_arrow = Arrow(
            cubic_eq, cubic_form,
            stroke_width=5,
            buff=0.1,
        )

        self.add(*plots, randy)
        self.play(
            Restore(quadratic),
            *map(FadeIn, plots[1:]),
            FadeOut(form),
            FadeOut(form_name),
            FadeOut(equation),
            randy.change("plain"),
        )
        self.play(randy.change("erm", cubic))
        self.wait()
        self.play(
            FadeOut(quadratic),
            FadeIn(low_fade_rect),
            Write(cubic_eq),
            FadeIn(cubic_name),
        )
        self.play(
            ShowCreation(cubic_arrow),
            FadeIn(cubic_form, DOWN),
            randy.change("confused", cubic_name),
        )
        self.play(Blink(randy))

        # Quartic
        quartic_name = TexText("Quartic ", "Formula")
        quartic_name.move_to(quartic).to_edge(UP)
        cubic_fade_rect = BackgroundRectangle(cubic, buff=0.01, fill_opacity=0.95)
        quartic_eq = Tex("{a}x^4 + {b}x^3 + {c}x^2 + {d}x + {e} = 0", **kw)
        quartic_eq.next_to(quartic, UP)

        main_form = Tex(r"r_{i}&=-\frac{b}{4 a}-S \pm \frac{1}{2} \sqrt{-4 S^{2}-2 p \pm \frac{q}{S}}")
        details = Tex(r"""
            &\text{Where}\\\\
            p&=\frac{8 a c-3 b^{2}}{8 a^{2}} \qquad \qquad\\\\
            q&=\frac{b^{3}-4 a b c+8 a^{2} d}{8 a^{3}}\\\\
            S&=\frac{1}{2} \sqrt{-\frac{2}{3} p+\frac{1}{3 a}\left(Q+\frac{\Delta_{0}}{Q}\right)}\\\\
            Q&=\sqrt[3]{\frac{\Delta_{1}+\sqrt{\Delta_{1}^{2}-4 \Delta_{0}^{3}}}{2}}\\\\
            \Delta_{0}&=c^{2}-3 b d+12 a e\\\\
            \Delta_{1}&=2 c^{3}-9 b c d+27 b^{2} e+27 a d^{2}-72 a c e\\\\
        """)
        main_form.match_width(quartic_eq)
        main_form.move_to(VGroup(quartic_name, quartic_eq))
        details.scale(0.5)
        details.to_corner(UR)
        details.set_stroke(BLACK, 3, background=True)

        self.play(
            FadeOut(cubic_eq),
            FadeOut(cubic_form),
            FadeOut(cubic_arrow),
            FadeIn(cubic_fade_rect),
            FadeTransform(cubic_name[0], quartic_name[0]),
            FadeTransform(cubic_name[1], quartic_name[1]),
            randy.change("erm", quartic_name),
            low_fade_rect.animate.replace(quintic, stretch=True).scale(1.01),
            FadeIn(quartic_eq),
        )
        self.play(Write(main_form))
        self.wait()
        self.play(
            randy.change("horrified", details),
            Write(details, run_time=5)
        )
        self.play(randy.animate.look_at(details.get_bottom()))
        self.play(Blink(randy))
        self.wait()

        # Quintic
        quintic.generate_target()
        quintic.target.set_height(5)
        quintic.target.to_corner(UL).shift(DOWN)
        quintic_eq = Tex(
            "{a}x^5 + {b}x^4 + {c}x^3 + {d}x^2 + {e}x + {f}",
            **kw
        )
        quintic_eq.match_width(quintic.target)
        quintic_eq.next_to(quintic.target, UP)
        quintic_name = Text("Quintic formula?", font_size=60)
        quintic_name.move_to(3 * RIGHT)
        quintic_name.to_edge(UP)

        subwords = VGroup(
            TexText("There is none.", "$^*$"),
            TexText("And there never can be."),
        )

        subwords.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        subwords.next_to(quintic_name, DOWN, LARGE_BUFF, aligned_edge=LEFT)
        footnote = Tex(
            "^*\\text{Using }",
            "+,\\,",
            "-,\\,",
            "\\times,\\,",
            "/,\\,",
            "\\sqrt[n]{\\quad},\\,",
            "\\text{exp},\\,",
            "\\log,\\,",
            "\\sin,\\,",
            "\\cos,\\,",
            "etc.\\\\",
            font_size=36,
            alignment="",
        )
        footnote.set_color(GREY_A)
        footnote.next_to(subwords, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
        footnote.shift_onto_screen(buff=MED_SMALL_BUFF)

        self.play(
            FadeOut(cubic),
            FadeOut(quartic),
            FadeOut(quartic_eq),
            FadeOut(main_form),
            FadeOut(details),
            FadeTransform(quartic_name, quintic_name),
            MoveToTarget(quintic),
            UpdateFromFunc(
                low_fade_rect,
                lambda m: m.replace(quintic, stretch=True),
            ),
            VFadeOut(low_fade_rect),
            randy.change("tease", quintic_name),
            FadeIn(quintic_eq),
        )
        self.play(Blink(randy))
        self.wait()
        self.play(
            FadeIn(subwords[0][0], 0.5 * DOWN),
            randy.change("erm", subwords),
        )
        self.wait()
        self.play(FadeIn(subwords[1], 0.5 * DOWN))
        self.wait()
        self.play(
            FadeIn(subwords[0][1]),
            LaggedStartMap(FadeIn, footnote, run_time=6, lag_ratio=0.5),
            randy.change("pondering", footnote)
        )
        self.play(Blink(randy))
        self.wait()

    def get_plot(self, coefs, scalar=1.0, color=YELLOW, stroke_width=3, height=3.5, bound=10):
        axes = NumberPlane(
            (-bound, bound, 5), (-bound, bound, 5),
            faded_line_ratio=4,
            background_line_style={
                "stroke_width": 1.0,
                "stroke_color": GREY_A,
            }
        )
        axes.set_height(height)
        axes.add_coordinate_labels(
            x_values=[-5, 0, 5, 10],
            y_values=[-5, 5, 10],
            font_size=16,
            excluding=[],
        )

        def f(x):
            return scalar * poly(x, coefs)

        x_min = binary_search(
            lambda x: abs(f(x)), bound, -bound, 0
        )
        x_max = binary_search(
            lambda x: abs(f(x)), bound, 0, bound,
        )

        graph = axes.get_graph(f, x_range=(x_min, x_max))
        graph.set_stroke(color, stroke_width)

        roots = [
            root.real
            for root in coefficients_to_roots(coefs)
            if np.isclose(root.imag, 0)
        ]

        def get_glow_dot(point):
            result = DotCloud([point] * 10)
            result.set_radii([
                interpolate(0.03, 0.06, t**2)
                for t in np.linspace(0, 1, 10)
            ])
            result.set_opacity(0.2)
            result.set_color(YELLOW)
            return result

        root_dots = Group(*(
            get_glow_dot(axes.c2p(root, 0))
            for root in roots
        ))

        result = Group(axes, graph, root_dots)
        return result


class ComingVideoWrapper(VideoWrapper):
    animate_boundary = False
    title = "Unsolvability of the Quintic (future topic?)"


class QuinticAppletPlay(ExternallyAnimatedScene):
    pass


class AskAboutFractals(TeacherStudentsScene):
    def construct(self):
        self.screen.set_height(4, about_edge=UL)
        self.screen.set_fill(BLACK, 1)
        self.add(self.screen)
        self.student_says(
            "Fractals?",
            target_mode="raise_right_hand",
            index=2,
            added_anims=[
                self.students[0].change("confused"),
                self.students[1].change("sassy"),
            ]
        )
        self.wait()
        self.teacher_says(
            TexText("We're getting\\\\there"),
            bubble_config={
                "height": 3,
                "width": 4,
            },
            target_mode="happy"
        )
        self.play_all_student_changes(
            "pondering",
            look_at=self.screen
        )
        self.wait(2)


class RealNewtonsMethod(Scene):
    coefs = [-0.2, -1, 1, 0, 0, 1]
    poly_tex = "x^5 + x^2 - x - 0.2"
    dpoly_tex = "5x^4 + 2x - 1"
    seed = 1.3
    graph_x_range = (-1.5, 1.5)
    axes_config = {
        "x_range": (-2, 2, 0.2),
        "y_range": (-2, 6, 0.2),
        "height": 8,
        "width": 8,
        "axis_config": {
            "tick_size": 0.05,
            "longer_tick_multiple": 2.0,
            "tick_offset": 0,
            # Change name
            "numbers_with_elongated_ticks": list(range(-2, 3)),
            "include_tip": False,
        }
    }
    graph_color = BLUE_C
    guess_color = YELLOW
    rule_font_size = 42
    n_search_steps = 5

    def construct(self):
        self.add_graph()
        self.add_title(self.axes)
        self.draw_graph()
        self.highlight_roots()
        self.preview_iterative_root_finding()
        self.introduce_step()
        self.find_root()

    def add_graph(self):
        axes = self.axes = Axes(**self.axes_config)
        axes.to_edge(RIGHT)
        axes.add_coordinate_labels(
            np.arange(*self.axes.x_range[:2]),
            np.arange(self.axes.y_range[0] + 1, self.axes.y_range[1]),
        )
        self.add(axes)

        graph = self.graph = axes.get_graph(
            lambda x: poly(x, self.coefs),
            x_range=self.graph_x_range,
        )
        graph.set_color(self.graph_color)

        self.add(graph)

    def add_title(self, axes, opacity=0):
        title = TexText("Newton's method", font_size=60)
        title.move_to(midpoint(axes.get_left(), LEFT_SIDE))
        title.to_edge(UP)
        title.set_opacity(opacity)

        poly = Tex(f"P({self.poly_tex[0]}) = ", self.poly_tex, "= 0 ")
        poly.match_width(title)
        poly.next_to(title, DOWN, buff=MED_LARGE_BUFF)
        poly.set_fill(GREY_A)
        title.add(poly)

        self.title = title
        self.poly = poly
        self.add(title)

    def draw_graph(self):
        rect = SurroundingRectangle(self.poly[:-1])
        rect.set_stroke(self.graph_color, 2)

        self.play(
            FlashAround(self.poly[:-1], color=self.graph_color, run_time=2),
            ShowCreation(rect, run_time=2),
            ShowCreation(self.graph, run_time=4),
        )
        self.wait()
        self.play(
            rect.animate.replace(self.poly[-1], stretch=True).scale(1.2)
        )
        self.wait()
        self.play(FadeOut(rect))

    def highlight_roots(self):
        roots = coefficients_to_roots(self.coefs)
        real_roots = [
            root.real for root in roots
            if abs(root.imag) < 1e-6
        ]
        real_roots.sort()

        dots = VGroup(*(
            # Dot(self.axes.c2p(r, 0), radius=0.05)
            glow_dot(self.axes.c2p(r, 0))
            for r in real_roots
        ))
        squares = VGroup(*[
            Square().set_height(0.25).move_to(dot)
            for dot in dots
        ])
        squares.set_stroke(YELLOW, 3)
        squares.set_fill(opacity=0)

        self.play(
            LaggedStart(
                *[
                    FadeIn(dot, scale=0.1)
                    for dot in dots
                ] + [
                    VShowPassingFlash(square, time_width=2.0, run_time=2)
                    for square in squares
                ],
                lag_ratio=0.15
            ),
        )
        self.wait()

        # Show values numerically
        root_strs = ["{0:.4}".format(root) for root in real_roots]
        equations = VGroup(*(
            Tex(
                "P(", root_str, ")", "=", "0",
                font_size=self.rule_font_size
            ).set_color_by_tex(root_str, YELLOW)
            for root_str in root_strs
        ))
        equations.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        equations.next_to(self.poly, DOWN, LARGE_BUFF, aligned_edge=LEFT)
        question = Text("How do you\ncompute these?")
        question.next_to(equations, RIGHT, buff=LARGE_BUFF)
        question.set_color(YELLOW)

        arrows = VGroup(*(
            Arrow(
                question.get_corner(UL) + 0.2 * DL,
                eq[1].get_corner(UR) + 0.25 * LEFT,
                path_arc=arc, stroke_width=3,
                buff=0.2,
            )
            for eq, arc in zip(equations, [0.7 * PI, 0.5 * PI, 0.0 * PI])
        ))
        arrows.set_color(YELLOW)

        self.play(
            LaggedStartMap(FadeIn, equations, lag_ratio=0.25),
            LaggedStart(*(
                FadeTransform(dot.copy(), eq[1])
                for dot, eq in zip(dots, equations)
            ), lag_ratio=0.25)
        )
        self.wait()
        self.play(
            Write(question),
            Write(arrows)
        )
        self.wait()

        self.play(LaggedStart(
            FadeOut(dots),
            FadeOut(question),
            FadeOut(arrows),
            FadeOut(equations),
            lag_ratio=0.25
        ))
        self.wait()

    def preview_iterative_root_finding(self):
        axes = self.axes
        axis = axes.x_axis
        coefs = self.coefs
        n_steps = 5

        root_seekers = VGroup(*(
            ArrowTip().set_height(0.2).rotate(-PI / 2).move_to(axis.n2p(x), DOWN)
            for x in np.arange(-2, 2.0, 0.2)[:-1]
        ))
        root_seekers.set_stroke(YELLOW, 2, opacity=0.5)
        root_seekers.set_fill(YELLOW, opacity=0.3)

        words = Text("Approximate\nSolutions", alignment="\\flushleft")
        words.move_to(axes.c2p(0, 3))
        words.align_to(axis, LEFT)
        words.set_color(YELLOW)

        self.play(
            FadeIn(root_seekers, lag_ratio=0.1),
            Write(words),
        )

        for n in range(n_steps):
            for rs in root_seekers:
                rs.generate_target()
                x = axis.p2n(rs.get_center())
                if n == 0 and abs(x - 0.4) < 0.1:
                    x = 0.6
                new_x = x - poly(x, coefs) / dpoly(x, coefs)
                rs.target.set_x(axis.n2p(new_x)[0])
            self.play(*map(MoveToTarget, root_seekers), run_time=1.0)
        self.wait()

        values = VGroup(*(
            DecimalNumber(
                axis.p2n(rs.get_center()),
                num_decimal_places=5,
                show_ellipsis=True,
            ).next_to(rs, UP, SMALL_BUFF)
            for rs in root_seekers[0::len(root_seekers) // 2]
        ))
        values.set_fill(YELLOW)
        values.set_stroke(BLACK, 8, background=True)
        last_value = VMobject()
        for value in values:
            self.play(
                FadeIn(value),
                FadeOut(last_value)
            )
            self.wait(0.5)
            last_value = value
        self.play(FadeOut(last_value))
        self.play(
            FadeOut(words),
            FadeOut(root_seekers),
        )

    def introduce_step(self):
        axes = self.axes
        graph = self.graph

        # Add labels
        guess_label = Tex(
            "\\text{Guess: } x_0 = " + f"{self.seed}",
            tex_to_color_map={"x_0": YELLOW}
        )
        guess_label.next_to(self.poly, DOWN, LARGE_BUFF)
        guess_marker, guess_value, guess_tracker = self.get_guess_group()
        get_guess = guess_tracker.get_value

        self.play(self.title.animate.set_opacity(1))
        self.wait()
        self.play(Write(guess_label))
        self.play(
            FadeTransform(
                guess_label[1].copy(),
                VGroup(guess_marker, guess_value)
            )
        )
        self.wait()

        # Add lines
        v_line = axes.get_v_line(axes.i2gp(get_guess(), graph))
        tan_line = self.get_tan_line(get_guess())

        v_line_label = Tex("P(x_0)", font_size=30, fill_color=GREY_A)
        v_line_label.next_to(v_line, RIGHT, SMALL_BUFF)

        self.add(v_line, guess_marker, guess_value)
        self.play(ShowCreation(v_line))
        self.play(FadeIn(v_line_label, 0.2 * RIGHT))
        self.wait()
        self.play(
            ShowCreation(tan_line),
            graph.animate.set_stroke(width=2),
        )

        # Mention next guess
        next_guess_label = Text("Next guess", font_size=30)
        next_guess_label.set_color(RED)
        next_guess_label.next_to(axes.c2p(0, 0), RIGHT, MED_LARGE_BUFF)
        next_guess_label.shift(UP)
        next_guess_arrow = Arrow(next_guess_label, tan_line.get_start(), buff=0.1)
        next_guess_arrow.set_stroke(RED, 3)

        coord = axes.coordinate_labels[0][-1]
        coord_copy = coord.copy()
        coord.set_opacity(0)
        self.play(
            coord_copy.animate.scale(0),
            ShowCreation(next_guess_arrow),
            FadeIn(next_guess_label),
        )
        self.wait()

        # Show derivative
        dpoly = Tex("P'(x) = ", self.dpoly_tex)
        dpoly.match_height(self.poly)
        dpoly.match_style(self.poly)
        dpoly.next_to(self.poly, DOWN, aligned_edge=LEFT)

        self.play(
            FadeIn(dpoly, 0.5 * DOWN),
            guess_label.animate.shift(0.25 * DOWN)
        )
        self.play(FlashAround(dpoly))
        self.wait()

        # Show step
        step_arrow = Arrow(v_line.get_start(), tan_line.get_start(), buff=0)
        step_arrow.set_stroke(GREY_A, 3)
        step_arrow.shift(0.1 * UP)
        step_word = Text("Step", font_size=24)
        step_word.set_stroke(BLACK, 3, background=True)
        step_word.next_to(step_arrow, UP, SMALL_BUFF)

        self.play(
            ShowCreation(step_arrow),
            FadeIn(step_word)
        )
        self.wait()

        # Show slope
        slope_eq_texs = [
            "P'(x_0) = {P(x_0) \\over -\\text{Step}}",
            "\\text{Step} = -{P(x_0) \\over P'(x_0)}",
        ]
        slope_eqs = [
            Tex(
                tex,
                isolate=[
                    "P'(x_0)",
                    "P(x_0)",
                    "\\text{Step}",
                    "-"
                ],
                font_size=self.rule_font_size,
            )
            for tex in slope_eq_texs
        ]
        for slope_eq in slope_eqs:
            slope_eq.set_fill(GREY_A)
            slope_eq.set_color_by_tex("Step", WHITE)
            slope_eq.next_to(guess_label, DOWN, LARGE_BUFF)

        rule = self.rule = self.get_update_rule()
        rule.next_to(guess_label, DOWN, LARGE_BUFF)

        for line in [v_line, Line(tan_line.get_start(), v_line.get_start())]:
            self.play(
                VShowPassingFlash(
                    Line(line.get_start(), line.get_end()).set_stroke(YELLOW, 10).insert_n_curves(20),
                    time_width=1.0,
                    run_time=1.5
                )
            )
        self.wait()
        self.play(
            FadeTransform(v_line_label.copy(), slope_eqs[0].get_part_by_tex("P(x_0)")),
            FadeTransform(step_word.copy(), slope_eqs[0].get_part_by_tex("\\text{Step}")),
            FadeIn(slope_eqs[0][3:5]),
        )
        self.wait()
        self.play(FadeIn(slope_eqs[0][:2]))
        self.wait()
        self.play(TransformMatchingTex(*slope_eqs, path_arc=PI / 2))
        self.wait()
        self.play(
            FadeIn(rule),
            slope_eqs[1].animate.to_edge(DOWN)
        )
        self.wait()

        # Transition to x1
        self.add(tan_line, guess_value)
        self.play(
            FadeOut(next_guess_label),
            FadeOut(next_guess_arrow),
            FadeOut(step_word),
            FadeOut(step_arrow),
            FadeOut(v_line),
            FadeOut(v_line_label),
            guess_tracker.animate.set_value(self.get_next_guess(get_guess())),
        )
        self.play(FadeOut(tan_line))

    def find_root(self, cycle_run_time=1.0):
        for n in range(self.n_search_steps):
            self.play(*self.cycle_rule_entries_anims(), run_time=cycle_run_time)
            self.step_towards_root()

    def step_towards_root(self, fade_tan_with_vline=False, added_anims=None):
        guess = self.guess_tracker.get_value()
        next_guess = self.get_next_guess(guess)

        v_line = self.axes.get_v_line(self.axes.i2gp(guess, self.graph))
        tan_line = self.get_tan_line(guess)

        self.add(v_line, tan_line, self.guess_marker, self.guess_value)
        self.play(
            ShowCreation(v_line),
            GrowFromCenter(tan_line)
        )
        anims = [
            FadeOut(v_line),
            self.guess_tracker.animate.set_value(next_guess)
        ]
        if added_anims is not None:
            anims += added_anims
        tan_fade = FadeOut(tan_line)
        if fade_tan_with_vline:
            self.play(*anims, tan_fade)
        else:
            self.play(*anims)
            self.play(tan_fade)

    #
    def get_guess_group(self):
        axes = self.axes
        guess_tracker = ValueTracker(self.seed)
        get_guess = guess_tracker.get_value

        guess_marker = Triangle(start_angle=PI / 2)
        guess_marker.set_height(0.1)
        guess_marker.set_width(0.1, stretch=True)
        guess_marker.set_fill(self.guess_color, 1)
        guess_marker.set_stroke(width=0)
        guess_marker.add_updater(lambda m: m.move_to(
            axes.c2p(get_guess(), 0), UP
        ))
        guess_value = DecimalNumber(0, num_decimal_places=3, font_size=24)

        def update_guess_value(gv):
            gv.set_value(get_guess())
            gv.next_to(guess_marker, DOWN, SMALL_BUFF)
            gv.set_fill(self.guess_color)
            gv.set_stroke(BLACK, 3, background=True)
            return gv

        guess_value.add_updater(update_guess_value)

        self.guess_tracker = guess_tracker
        self.guess_marker = guess_marker
        self.guess_value = guess_value

        return (guess_marker, guess_value, guess_tracker)

    def get_next_guess(self, curr_guess):
        x = curr_guess
        return x - poly(x, self.coefs) / dpoly(x, self.coefs)

    def get_tan_line(self, curr_guess):
        next_guess = self.get_next_guess(curr_guess)
        start = self.axes.c2p(next_guess, 0)
        end = self.axes.i2gp(curr_guess, self.graph)
        line = Line(start, start + 2 * (end - start))
        line.set_stroke(RED, 3)
        return line

    def get_update_rule(self, char="x"):
        rule = Tex(
            """
                z_1 =
                z_0 - {P(z_0) \\over P'(z_0)}
            """.replace("z", char),
            tex_to_color_map={
                f"{char}_1": self.guess_color,
                f"{char}_0": self.guess_color
            },
            font_size=self.rule_font_size,
        )

        rule.n = 0
        rule.zns = rule.get_parts_by_tex(f"{char}_0")
        rule.znp1 = rule.get_parts_by_tex(f"{char}_1")
        return rule

    def cycle_rule_entries_anims(self):
        rule = self.rule
        rule.n += 1
        char = rule.get_tex().strip()[0]
        zns = VGroup()
        for old_zn in rule.zns:
            zn = Tex(f"{char}_{{{rule.n}}}", font_size=self.rule_font_size)
            zn[0][1:].set_max_width(0.2, about_edge=DL)
            zn.move_to(old_zn)
            zn.match_color(old_zn)
            zns.add(zn)
        znp1 = Tex(f"{char}_{{{rule.n + 1}}}", font_size=self.rule_font_size)
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


class FasterNewtonExample(RealNewtonsMethod):
    coefs = [0.1440, -1.0, 1.2, 1]
    poly_tex = "x^3 + 1.2x^2 - x + 0.144"
    dpoly_tex = "3x^2 + 2.4x - 1"
    n_search_steps = 6
    graph_x_range = (-2, 2)
    seed = 1.18
    axes_config = {
        "x_range": (-2, 2, 0.2),
        "y_range": (-1, 3, 0.2),
        "height": 8,
        "width": 8,
        "axis_config": {
            "tick_size": 0.05,
            "longer_tick_multiple": 2.0,
            "tick_offset": 0,
            # Change name
            "numbers_with_elongated_ticks": list(range(-2, 3)),
            "include_tip": False,
        }
    }

    def construct(self):
        self.add_graph()
        self.add_title(self.axes)
        self.draw_graph()
        self.introduce_step()
        self.find_root()

    def find_root(self, cycle_run_time=1.0):
        for n in range(self.n_search_steps):
            self.step_towards_root(
                added_anims=self.cycle_rule_entries_anims(),
                fade_tan_with_vline=True
            )


class AssumingItsGood(TeacherStudentsScene):
    def construct(self):
        self.pi_creatures.refresh_triangulation()
        self.teacher_says(
            TexText("Assuming this\\\\approximation\\\\is decent...", font_size=42),
            bubble_config={
                "height": 3, "width": 4,
            }
        )
        self.play_student_changes(
            "pondering", "pondering", "tease",
            look_at=self.screen
        )
        self.pi_creatures.refresh_triangulation()
        self.wait(3)


class PauseAndPonder(TeacherStudentsScene):
    def construct(self):
        self.teacher_says("Pause and\nponder", target_mode="hooray")
        self.play_all_student_changes("thinking", look_at=self.screen)
        self.wait(4)


class AltPauseAndPonder(Scene):
    def construct(self):
        morty = Mortimer(height=2)
        morty.flip().to_corner(DL)
        self.play(PiCreatureSays(
            morty, TexText("Pause and\\\\Ponder", font_size=36),
            target_mode="hooray",
            bubble_config={
                "height": 2,
                "width": 3,
            }
        ))
        self.play(Blink(morty))
        self.wait(2)
        self.play(morty.change("thinking"))
        self.play(Blink(morty))
        self.wait()


class WhatIsThis(Scene):
    def construct(self):
        words = Text("What is this", color=RED)
        arrow = Vector(UR)
        arrow.set_color(RED)
        words.next_to(ORIGIN, DOWN)
        self.play(FadeIn(words, lag_ratio=0.1), ShowCreation(arrow))
        self.wait()


class GutCheckFormula(RealNewtonsMethod):
    seed = 5.0

    def construct(self):
        self.add_axes_and_graph()
        self.add_rule()
        self.add_guess()
        self.sample_values()

    def add_axes_and_graph(self):
        axes = NumberPlane(
            (-2, 15), (-2, 8),
            faded_line_ratio=1,
            background_line_style={
                "stroke_opacity": 0.5,
                "stroke_color": GREY,
            }
        )
        axes.to_corner(DL, buff=0)
        axes.add_coordinate_labels(font_size=16, fill_opacity=0.5)
        axes.x_axis.numbers.next_to(axes.x_axis, UP, buff=0.05)
        self.add(axes)

        roots = [-1, 3, 4.5]
        coefs = 0.1 * np.array(roots_to_coefficients(roots))
        graph = axes.get_graph(lambda x: poly(x, coefs))
        graph.set_stroke(BLUE, 3)
        self.add(graph)

        self.root_point = axes.c2p(roots[-1], 0)

        self.axes = axes
        self.graph = graph

    def add_rule(self):
        rule = Tex(
            "x_{n + 1}", "=",
            "x_{n}", " - ", "{P(x) ", "\\over ", "P'(x)}"
        )
        rule.set_stroke(BLACK, 5, background=True)
        rule.to_corner(UR)

        step_box = SurroundingRectangle(rule[3:], buff=0.1)
        step_box.set_stroke(YELLOW, 1.0)
        step_word = Text("Step size", font_size=36)
        step_word.set_color(YELLOW)
        step_word.next_to(step_box, DOWN)

        self.add(rule)
        self.add(step_box)
        self.add(step_word)

        self.rule = rule
        self.step_box = step_box
        self.step_word = step_word

    def add_guess(self, include_px=True):
        guess_group = self.get_guess_group()
        marker, value, tracker = guess_group
        self.guess_tracker = tracker

        def update_v_line(v_line):
            x = tracker.get_value()
            graph_point = self.graph.pfp(
                inverse_interpolate(*self.graph.x_range[:2], x)
            )
            v_line.put_start_and_end_on(
                self.axes.c2p(x, 0),
                graph_point,
            )

        v_line = Line()
        v_line.set_stroke(WHITE, 2)
        v_line.add_updater(update_v_line)

        self.add(*guess_group)
        self.add(v_line)

        if include_px:
            px_label = Tex("P(x)", font_size=36)
            px_label.add_updater(lambda m: m.next_to(v_line, RIGHT, buff=0.05))
            self.add(px_label)

    def sample_values(self):
        box = self.step_box
        rule = self.rule
        tracker = self.guess_tracker
        graph = self.graph

        words = Text("Gut check!")
        words.next_to(self.step_word, DOWN, LARGE_BUFF)
        words.shift(2 * LEFT)
        arrow = Arrow(words, self.rule)

        self.play(
            Write(words, run_time=1),
            ShowCreation(arrow),
        )
        self.wait()
        self.play(
            FadeOut(words),
            FadeOut(arrow),
            FadeOut(self.step_word),
            box.animate.replace(rule[4], stretch=True).scale(1.2).set_stroke(width=2.0),
        )
        self.play(
            tracker.animate.set_value(6.666),
            run_time=3,
        )

        arrow = Arrow(
            self.axes.c2p(tracker.get_value(), 0),
            self.root_point,
            buff=0,
            stroke_color=RED,
        )
        self.play(ShowCreation(arrow))
        self.wait()

        # Large p_prime
        self.play(
            FadeOut(arrow),
            tracker.animate.set_value(5.0),
        )
        self.play(
            graph.animate.stretch(8, 1, about_point=self.axes.c2p(0, 0)),
            box.animate.replace(self.rule[-1]).scale(1.2),
            run_time=3
        )
        self.wait()

        tan_line = self.get_tan_line(graph, tracker.get_value(), 15)
        self.play(ShowCreation(tan_line))
        self.wait()

    def get_tan_line(self, graph, x, length=5, epsilon=1e-3):
        alpha = inverse_interpolate(*graph.x_range[:2], x)
        tan_line = Line(
            graph.pfp(alpha - epsilon),
            graph.pfp(alpha + epsilon),
        )
        tan_line.set_length(length)
        tan_line.set_stroke(RED, 5)
        return tan_line


class HistoryWithNewton(Scene):
    def construct(self):
        # Add title
        title = Text("Newton's method", font_size=60)
        title.to_edge(UP)
        self.add(title)

        # Add timeline
        time_range = (1620, 2020)
        timeline = NumberLine(
            (*time_range, 1),
            tick_size=0.025,
            longer_tick_multiple=4,
            numbers_with_elongated_ticks=range(*time_range, 10),
        )
        timeline.stretch(0.2 / timeline.get_unit_size(), 0)
        timeline_center = 2 * DOWN
        timeline.move_to(timeline_center)
        timeline.to_edge(RIGHT)
        timeline.add_numbers(
            range(*time_range, 10),
            group_with_commas=False,
        )
        timeline.shift(timeline_center - timeline.n2p(1680))

        self.add(timeline)

        # Newton
        newton = get_figure("Newton", "Isaac Newton", "1669")
        newton.next_to(title, DOWN, buff=0.5)
        newton.to_edge(LEFT, buff=1.5)
        newton_point = timeline.n2p(1669)
        newton_arrow = Arrow(newton_point, newton[0].get_right() + DOWN, path_arc=PI / 3)

        newton_words = Text("Overly\ncomplicated", font_size=36)
        newton_words.next_to(newton[0], RIGHT)

        raphson_point = timeline.n2p(1690)
        raphson = get_figure("Newton", "Joseph Raphson", "1690")
        raphson.move_to(newton)
        raphson.set_x(raphson_point[0] + 2)
        raphson[1].set_opacity(0)
        raphson_arrow = Arrow(raphson_point, raphson[0].get_left() + DOWN, path_arc=-PI / 3)
        raphson_word = Text("Simplified", font_size=36)
        raphson_word.next_to(raphson[0], LEFT)

        no_image_group = VGroup(
            Text("No image"),
            Text("(sorry)"),
            # Randolph(mode="shruggie", height=1)
        )
        no_image_group[:2].set_fill(GREY)
        no_image_group.arrange(DOWN, buff=0.5)
        no_image_group.set_width(raphson[0].get_width() - 0.5)
        no_image_group.move_to(raphson[0])

        self.add(newton, newton_arrow)

        frame = self.camera.frame
        frame.save_state()
        title.fix_in_frame()
        frame.move_to(timeline, RIGHT)
        self.play(
            frame.animate.match_width(timeline).set_x(timeline.get_center()[0]),
            run_time=2
        )
        self.play(Restore(frame, run_time=2))

        # self.play(
        #     GrowFromPoint(newton, newton_point),
        #     ShowCreation(newton_arrow)
        # )
        self.wait()
        self.play(Write(newton_words))
        self.wait()
        self.play(
            GrowFromPoint(raphson, raphson_point),
            ShowCreation(raphson_arrow),
        )
        self.play(LaggedStartMap(FadeIn, no_image_group, lag_ratio=0.2))
        self.play(FadeIn(raphson_word))
        self.wait()

        new_title = Text("Newton-Raphson method", font_size=60)
        new_title.to_edge(UP)
        self.play(
            FadeOut(title),
            TransformFromCopy(
                newton[2].get_part_by_text("Newton"),
                new_title.get_part_by_text("Newton"),
            ),
            TransformFromCopy(
                raphson[2].get_part_by_text("Raphson"),
                new_title.get_part_by_text("Raphson"),
            ),
            TransformFromCopy(
                title.get_part_by_text("method"),
                new_title.get_part_by_text("method"),
            ),
            FadeIn(new_title.get_part_by_text("-"))
        )
        self.play(FlashAround(new_title, run_time=2))
        self.wait()


class CalcHomework(GutCheckFormula):
    seed = 3.0

    def construct(self):
        # Title
        old_title = Text("Newton-Raphson method", font_size=60)
        old_title.to_edge(UP)
        title = Text("Calc 1", font_size=72)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        line = Underline(title)
        line.scale(2)
        line.set_stroke(WHITE, 2)
        self.add(old_title)

        # Axes
        axes = NumberPlane(
            x_range=(-5, 5, 1),
            y_range=(-8, 10, 2),
            height=6.5,
            width=FRAME_WIDTH,
            faded_line_ratio=4,
            background_line_style={
                "stroke_color": GREY_C,
                "stroke_width": 1,
            }
        )
        axes.to_edge(DOWN, buff=0)
        axes.add_coordinate_labels(font_size=18)

        self.add(axes)

        # Homework
        hw = TexText(
            "Homework:\\\\",
            "\\quad Approximate $\\sqrt{7}$ by hand using\\\\",
            "\\quad the ", "Newton-Raphson method.",
            alignment="",
            font_size=36,
            color=GREY_A,
        )
        hw[1:].shift(MED_SMALL_BUFF * RIGHT + SMALL_BUFF * DOWN)
        hw.add_to_back(
            BackgroundRectangle(hw, fill_opacity=0.8, buff=0.25)
        )
        hw.move_to(axes, UL)
        hw.to_edge(LEFT, buff=0)

        self.wait()
        self.play(
            FadeIn(hw, lag_ratio=0.1, run_time=2),
            FadeTransform(
                old_title,
                hw[-1]
            ),
            FadeIn(title),
            ShowCreation(line),
        )
        self.wait()

        # Graph
        graph = axes.get_graph(
            lambda x: x**2 - 7,
            x_range=(-math.sqrt(17), math.sqrt(17))
        )
        graph.set_stroke(BLUE, 2)
        graph_label = Tex("x^2 - 7", font_size=36)
        graph_label.set_color(BLUE)
        graph_label.next_to(graph.pfp(0.99), LEFT)

        self.add(graph, hw)
        self.play(ShowCreation(graph, run_time=3))
        self.play(FadeIn(graph_label))
        self.wait()

        # Marker
        axes.x_axis.numbers.remove(axes.x_axis.numbers[-3])
        self.axes = axes
        self.graph = graph
        self.add_guess(include_px=False)
        self.wait()

        # Update
        tan_line = self.get_tan_line(graph, 3)
        tan_line.set_stroke(width=3)
        update_tex = Tex(
            "3 \\rightarrow 3 - {3^2 - 7 \\over 2 \\cdot 3}",
            tex_to_color_map={"3": YELLOW},
            font_size=28
        )
        update_tex.next_to(axes.c2p(1.2, 0), UR, buff=SMALL_BUFF)

        self.add(tan_line, self.guess_marker, self.guess_value)
        self.play(
            GrowFromCenter(tan_line),
            FadeIn(update_tex),
        )
        self.wait()
        self.play(
            self.guess_tracker.animate.set_value(8 / 3),
            run_time=2
        )


class RealNewtonsMethodHigherGraph(FasterNewtonExample):
    coefs = [1, -1, 1, 0, 0, 0.99]
    poly_tex = "x^5 + x^2 - x + 1"
    n_search_steps = 20


class FactorPolynomial(RealNewtonsMethodHigherGraph):
    def construct(self):
        self.add_graph()
        self.add_title(self.axes)
        self.show_factors()

    def show_factors(self):
        poly = self.poly
        colors = color_gradient((BLUE, YELLOW), 5)
        factored = Tex(
            "P(x) = ", *(
                f"(x - r_{n})"
                for n in range(5)
            ),
            tex_to_color_map={
                f"r_{n}": color
                for n, color in enumerate(colors)
            }
        )
        factored.match_height(poly[0])
        factored.next_to(poly, DOWN, LARGE_BUFF, LEFT)

        self.play(
            FadeTransform(poly.copy(), factored)
        )
        self.wait()

        words = TexText("Potentially complex\\\\", "$r_n = a_n + b_n i$")
        words.set_color(GREY_A)
        words.next_to(factored, DOWN, buff=1.5)
        words.shift(LEFT)
        lines = VGroup(*(
            Line(words, part, buff=0.15).set_stroke(part.get_color(), 2)
            for n in range(5)
            for part in [factored.get_part_by_tex(f"r_{n}")]
        ))

        self.play(
            FadeIn(words[0]),
            Write(lines),
        )
        self.play(FadeIn(words[1], 0.5 * DOWN))
        self.wait()


class TransitionToComplexPlane(RealNewtonsMethodHigherGraph):
    poly_tex = "z^5 + z^2 - z + 1"

    def construct(self):
        self.add_graph()
        self.add_title(self.axes)
        self.poly.save_state()
        self.poly.to_corner(UL)
        self.center_graph()
        self.show_example_point()
        self.separate_input_and_output()
        self.move_input_around_plane()

    def center_graph(self):
        shift_vect = DOWN - self.axes.c2p(0, 0)

        self.play(
            self.axes.animate.shift(shift_vect),
            self.graph.animate.shift(shift_vect),
        )
        self.wait()

    def show_example_point(self):
        axes = self.axes

        input_tracker = ValueTracker(1)
        get_x = input_tracker.get_value

        def get_px():
            return poly(get_x(), self.coefs)

        def get_graph_point():
            return axes.c2p(get_x(), get_px())

        marker = ArrowTip().set_height(0.1)
        input_marker = marker.copy().rotate(PI / 2)
        input_marker.set_color(YELLOW)
        output_marker = marker.copy()
        output_marker.set_color(MAROON_B)
        input_marker.add_updater(lambda m: m.move_to(axes.x_axis.n2p(get_x()), UP))
        output_marker.add_updater(lambda m: m.shift(axes.y_axis.n2p(get_px()) - m.get_start()))

        v_line = always_redraw(
            lambda: axes.get_v_line(get_graph_point(), line_func=Line).set_stroke(YELLOW, 1)
        )
        h_line = always_redraw(
            lambda: axes.get_h_line(get_graph_point(), line_func=Line).set_stroke(MAROON_B, 1)
        )

        self.add(
            input_tracker,
            input_marker,
            output_marker,
            v_line,
            h_line,
        )

        self.play(input_tracker.animate.set_value(-0.5), run_time=3)
        self.play(input_tracker.animate.set_value(1.0), run_time=3)
        self.play(ShowCreationThenFadeOut(
            axes.get_tangent_line(get_x(), self.graph).set_stroke(RED, 3)
        ))

        self.input_tracker = input_tracker
        self.input_marker = input_marker
        self.output_marker = output_marker
        self.v_line = v_line
        self.h_line = h_line

    def separate_input_and_output(self):
        axes = self.axes
        x_axis, y_axis = axes.x_axis, axes.y_axis
        graph = self.graph
        input_marker = self.input_marker
        output_marker = self.output_marker
        v_line = self.v_line
        h_line = self.h_line

        in_plane = ComplexPlane(
            (-2, 2),
            (-2, 2),
            height=5,
            width=5,
        )
        in_plane.add_coordinate_labels(font_size=18)
        in_plane.to_corner(DL)
        out_plane = in_plane.deepcopy()
        out_plane.to_corner(DR)

        input_word = Text("Input")
        output_word = Text("Output")

        input_word.next_to(in_plane.x_axis, UP)
        output_word.rotate(PI / 2)
        output_word.next_to(out_plane.y_axis, RIGHT, buff=0.5)

        cl_copy = axes.coordinate_labels.copy()
        axes.coordinate_labels.set_opacity(0)
        self.play(
            *map(FadeOut, (v_line, h_line, graph, cl_copy)),
        )

        for axis1, axis2 in [(x_axis, in_plane.x_axis), (y_axis, out_plane.y_axis)]:
            axis1.generate_target()
            axis1.target.scale(axis2.get_unit_size() / axis1.get_unit_size())
            axis1.target.shift(axis2.n2p(0) - axis1.target.n2p(0))
        self.play(
            MoveToTarget(x_axis),
            MoveToTarget(y_axis),
            FadeIn(input_word),
            FadeIn(output_word),
        )
        self.wait()
        self.add(in_plane, input_marker)
        self.play(
            input_word.animate.next_to(in_plane, UP),
            x_axis.animate.set_stroke(width=0),
            Write(in_plane, lag_ratio=0.03),
        )
        self.play(
            Rotate(
                VGroup(y_axis, output_word, output_marker),
                -PI / 2,
                about_point=out_plane.n2p(0)
            )
        )
        self.add(out_plane, output_marker)
        self.play(
            output_word.animate.next_to(out_plane, UP),
            y_axis.animate.set_stroke(width=0),
            Write(out_plane, lag_ratio=0.03),
        )
        self.wait()

        self.in_plane = in_plane
        self.out_plane = out_plane
        self.input_word = input_word
        self.output_word = output_word

    def move_input_around_plane(self):
        in_plane = self.in_plane
        out_plane = self.out_plane
        input_marker = self.input_marker
        output_marker = self.output_marker

        in_dot, out_dot = [
            Dot(radius=0.05).set_fill(marker.get_fill_color()).move_to(marker.get_start())
            for marker in (input_marker, output_marker)
        ]
        in_dot.set_fill(YELLOW, 1)

        in_tracer = TracingTail(in_dot, stroke_color=in_dot.get_color())
        out_tracer = TracingTail(out_dot, stroke_color=out_dot.get_color())
        self.add(in_tracer, out_tracer)

        out_dot.add_updater(lambda m: m.move_to(out_plane.n2p(
            poly(in_plane.p2n(in_dot.get_center()), self.coefs)
        )))

        z_label = Tex("z", font_size=24)
        z_label.set_fill(YELLOW)
        z_label.add_background_rectangle()
        z_label.add_updater(lambda m: m.next_to(in_dot, UP, SMALL_BUFF))
        pz_label = Tex("P(z)", font_size=24)
        pz_label.set_fill(MAROON_B)
        pz_label.add_background_rectangle()
        pz_label.add_updater(lambda m: m.next_to(out_dot, UP, SMALL_BUFF))

        self.play(
            *map(FadeOut, (input_marker, output_marker)),
            *map(FadeIn, (in_dot, out_dot)),
            FadeIn(z_label),
            FlashAround(z_label),
        )
        self.play(
            FadeTransform(z_label.copy(), pz_label)
        )
        z_values = [
            complex(-0.5, 0.5),
            complex(-0.5, -0.5),
            complex(-0.25, 0.25),
            complex(0.5, -0.5),
            complex(0.5, 0.5),
            complex(1, 0.25),
        ]
        for z in z_values:
            self.play(
                in_dot.animate.move_to(in_plane.n2p(z)),
                run_time=2,
                path_arc=PI / 2
            )
            self.wait()

        self.remove(in_tracer, out_tracer)
        in_plane.generate_target()
        in_dot.generate_target()
        group = VGroup(in_plane.target, in_dot.target)
        group.set_height(8).center().to_edge(RIGHT, buff=0),
        self.play(
            MoveToTarget(in_plane),
            MoveToTarget(in_dot),
            FadeOut(self.input_word),
            FadeOut(self.output_word),
            FadeOut(out_plane),
            FadeOut(out_dot),
            FadeOut(pz_label),
            self.poly.animate.restore().shift(0.32 * RIGHT),
        )


class ComplexNewtonsMethod(RealNewtonsMethod):
    coefs = [1, -1, 1, 0, 0, 1]
    poly_tex = "z^5 + z^2 - z + 1"
    plane_config = {
        "x_range": (-2, 2),
        "y_range": (-2, 2),
        "height": 8,
        "width": 8,
    }
    seed = complex(-0.5, 0.5)
    seed_tex = "-0.5 + 0.5i"
    guess_color = YELLOW
    pz_color = MAROON_B
    step_arrow_width = 5
    step_arrow_opacity = 1.0
    step_arrow_len = None
    n_search_steps = 9

    def construct(self):
        self.add_plane()
        self.add_title()
        self.add_z0_def()
        self.add_pz_dot()
        self.add_rule()
        self.find_root()

    def add_plane(self):
        plane = ComplexPlane(**self.plane_config)
        plane.add_coordinate_labels(font_size=24)
        plane.to_edge(RIGHT, buff=0)
        self.plane = plane
        self.add(plane)

    def add_title(self, opacity=1):
        super().add_title(self.plane, opacity)

    def add_z0_def(self):
        seed_text = Text("(Arbitrary seed)")
        z0_def = Tex(
            f"z_0 = {self.seed_tex}",
            tex_to_color_map={"z_0": self.guess_color},
            font_size=self.rule_font_size
        )
        z0_group = VGroup(seed_text, z0_def)
        z0_group.arrange(DOWN)
        z0_group.next_to(self.title, DOWN, buff=LARGE_BUFF)

        guess_dot = Dot(self.plane.n2p(self.seed), color=self.guess_color)

        guess = DecimalNumber(self.seed, num_decimal_places=3, font_size=30)
        guess.add_updater(
            lambda m: m.set_value(self.plane.p2n(
                guess_dot.get_center()
            )).set_fill(self.guess_color).add_background_rectangle()
        )
        guess.add_updater(lambda m: m.next_to(guess_dot, UP, buff=0.15))

        self.play(
            Write(seed_text, run_time=1),
            FadeIn(z0_def),
        )
        self.play(
            FadeTransform(z0_def[0].copy(), guess_dot),
            FadeIn(guess),
        )
        self.wait()

        self.z0_group = z0_group
        self.z0_def = z0_def
        self.guess_dot = guess_dot
        self.guess = guess

    def add_pz_dot(self):
        plane = self.plane
        guess_dot = self.guess_dot

        def get_pz():
            z = plane.p2n(guess_dot.get_center())
            return poly(z, self.coefs)

        pz_dot = Dot(color=self.pz_color)
        pz_dot.add_updater(lambda m: m.move_to(plane.n2p(get_pz())))
        pz_label = Tex("P(z)", font_size=24)
        pz_label.set_color(self.pz_color)
        pz_label.add_background_rectangle()
        pz_label.add_updater(lambda m: m.next_to(pz_dot, UL, buff=0))

        self.play(
            FadeTransform(self.poly[0].copy(), pz_label),
            FadeIn(pz_dot),
        )
        self.wait()

    def add_rule(self):
        self.rule = rule = self.get_update_rule("z")
        rule.next_to(self.z0_group, DOWN, buff=LARGE_BUFF)

        self.play(
            FadeTransformPieces(self.z0_def[0].copy(), rule.zns),
            FadeIn(rule),
        )
        self.wait()

    def find_root(self):
        for x in range(self.n_search_steps):
            self.root_search_step()

    def root_search_step(self):
        dot = self.guess_dot
        dot_step_anims = self.get_dot_step_anims(VGroup(dot))
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

    def get_dot_step_anims(self, dots):
        plane = self.plane
        arrows = VGroup()
        dots.generate_target()
        for dot, dot_target in zip(dots, dots.target):
            try:
                z0 = plane.p2n(dot.get_center())
                pz = poly(z0, self.coefs)
                dpz = dpoly(z0, self.coefs)
                if abs(pz) < 1e-3:
                    z1 = z0
                else:
                    if dpz == 0:
                        dpz = 0.1  # ???
                    z1 = z0 - pz / dpz

                if np.isnan(z1):
                    z1 = z0

                arrow = Arrow(
                    plane.n2p(z0), plane.n2p(z1),
                    buff=0,
                    stroke_width=self.step_arrow_width,
                    storke_opacity=self.step_arrow_opacity,
                )
                if self.step_arrow_len is not None:
                    if arrow.get_length() > self.step_arrow_len:
                        arrow.set_length(self.step_arrow_len, about_point=arrow.get_start())

                if not hasattr(dot, "history"):
                    dot.history = [dot.get_center().copy()]
                dot.history.append(plane.n2p(z1))

                arrows.add(arrow)
                dot_target.move_to(plane.n2p(z1))
            except ValueError:
                pass
        return [
            ShowCreation(arrows, lag_ratio=0),
            AnimationGroup(
                MoveToTarget(dots),
                FadeOut(arrows),
            )
        ]


class OutputIsZero(Scene):
    def construct(self):
        words = TexText("Output $\\approx 0$")
        words.set_stroke(BLACK, 5, background=True)
        arrow = Vector(0.5 * UL)
        words.next_to(arrow, DR)
        words.shift(0.5 * LEFT)

        self.play(
            Write(words),
            ShowCreation(arrow)
        )
        self.wait()


class FunPartWords(Scene):
    def construct(self):
        text = TexText("Now here's \\\\ the fun part", font_size=72)
        self.add(text)


class ComplexNewtonsMethodManySeeds(ComplexNewtonsMethod):
    dot_radius = 0.035
    dot_color = WHITE
    dot_opacity = 0.8
    step_arrow_width = 3
    step_arrow_opacity = 0.1
    step_arrow_len = 0.15

    plane_config = {
        "x_range": (-2, 2),
        "y_range": (-2, 2),
        "height": 8,
        "width": 8,
    }
    step = 0.2
    n_search_steps = 20
    colors = ROOT_COLORS_BRIGHT

    def construct(self):
        self.add_plane()
        self.add_title()
        self.add_z0_def()
        self.add_rule()
        self.add_true_root_circles()
        self.find_root()
        self.add_color()

    def add_z0_def(self):
        seed_text = Text("Many seeds: ")
        z0_def = Tex(
            "z_0",
            tex_to_color_map={"z_0": self.guess_color},
            font_size=self.rule_font_size
        )
        z0_group = VGroup(seed_text, z0_def)
        z0_group.arrange(RIGHT)
        z0_group.next_to(self.title, DOWN, buff=LARGE_BUFF)

        x_range = self.plane_config["x_range"]
        y_range = self.plane_config["y_range"]
        step = self.step
        x_vals = np.arange(x_range[0], x_range[1] + step, step)
        y_vals = np.arange(y_range[0], y_range[1] + step, step)
        guess_dots = VGroup(*(
            Dot(
                self.plane.c2p(x, y),
                radius=self.dot_radius,
                fill_opacity=self.dot_opacity,
            )
            for i, x in enumerate(x_vals)
            for y in (y_vals if i % 2 == 0 else reversed(y_vals))
        ))
        guess_dots.set_submobject_colors_by_gradient(WHITE, GREY_B)
        guess_dots.set_fill(opacity=self.dot_opacity)
        guess_dots.set_stroke(BLACK, 2, background=True)

        self.play(
            Write(seed_text, run_time=1),
            FadeIn(z0_def),
        )
        self.play(
            LaggedStart(*(
                FadeTransform(z0_def[0].copy(), guess_dot)
                for guess_dot in guess_dots
            ), lag_ratio=0.1 / len(guess_dots)),
            run_time=3
        )
        self.add(guess_dots)
        self.wait()

        self.z0_group = z0_group
        self.z0_def = z0_def
        self.guess_dots = guess_dots

    def add_true_root_circles(self):
        roots = coefficients_to_roots(self.coefs)
        root_points = list(map(self.plane.n2p, roots))
        colors = self.colors

        root_circles = VGroup(*(
            Dot(radius=0.1).set_fill(color, opacity=0.75).move_to(rp)
            for rp, color in zip(root_points, colors)
        ))

        self.play(
            LaggedStart(*(
                FadeIn(rc, scale=0.5)
                for rc in root_circles
            ), lag_ratio=0.7, run_time=1),
        )
        self.wait()

        self.root_circles = root_circles

    def root_search_step(self):
        dots = self.guess_dots
        dot_step_anims = self.get_dot_step_anims(dots)

        self.play(dot_step_anims[0], run_time=0.25)
        self.play(
            dot_step_anims[1],
            *self.cycle_rule_entries_anims(),
            run_time=1
        )

    def add_color(self):
        root_points = [circ.get_center() for circ in self.root_circles]
        colors = [circ.get_fill_color() for circ in self.root_circles]

        dots = self.guess_dots
        dots.generate_target()
        for dot, dot_target in zip(dots, dots.target):
            dc = dot.get_center()
            dot_target.set_color(colors[
                np.argmin([get_norm(dc - rp) for rp in root_points])
            ])

        rect = SurroundingRectangle(self.rule)
        rect.set_fill(BLACK, 1)
        rect.set_stroke(width=0)

        self.play(
            FadeIn(rect),
            MoveToTarget(dots)
        )
        self.wait()

        len_history = max([len(dot.history) for dot in dots if hasattr(dot, "history")], default=0)
        for n in range(len_history):
            dots.generate_target()
            for dot, dot_target in zip(dots, dots.target):
                try:
                    dot_target.move_to(dot.history[len_history - n - 1])
                except Exception:
                    pass
            self.play(MoveToTarget(dots, run_time=0.5))


class ZeroStepColoring(ComplexNewtonsMethodManySeeds):
    n_search_steps = 0


class ComplexNewtonsMethodManySeedsHigherRes(ComplexNewtonsMethodManySeeds):
    step = 0.05


class IntroNewtonFractal(Scene):
    coefs = [1.0, -1.0, 1.0, 0.0, 0.0, 1.0]
    plane_config = {
        "x_range": (-4, 4),
        "y_range": (-4, 4),
        "height": 16,
        "width": 16,
        "background_line_style": {
            "stroke_color": GREY_A,
            "stroke_width": 1.0,
        },
        "axis_config": {
            "stroke_width": 1.0,
        }
    }
    n_steps = 30

    def construct(self):
        self.init_fractal(root_colors=ROOT_COLORS_BRIGHT)
        fractal, plane, root_dots = self.group

        # Transition from last scene
        frame = self.camera.frame
        frame.shift(plane.n2p(2) - RIGHT_SIDE)

        blocker = BackgroundRectangle(plane, fill_opacity=1)
        blocker.move_to(plane.n2p(-2), RIGHT)
        self.add(blocker)

        self.play(
            frame.animate.center(),
            FadeOut(blocker),
            run_time=2,
        )
        self.wait()
        self.play(
            fractal.animate.set_colors(ROOT_COLORS_DEEP),
            *(
                dot.animate.set_fill(interpolate_color(color, WHITE, 0.2))
                for dot, color in zip(root_dots, ROOT_COLORS_DEEP)
            )
        )
        self.wait()

        # Zoom in
        fractal.set_n_steps(40)
        zoom_points = [
            [-3.12334879, 1.61196545, 0.],
            [1.21514006, 0.01415811, 0.],
        ]
        for point in zoom_points:
            self.play(
                frame.animate.set_height(2e-3).move_to(point),
                run_time=25,
                rate_func=bezier(2 * [0] + 6 * [1])
            )
            self.wait()
            self.play(
                frame.animate.center().set_height(8),
                run_time=10,
                rate_func=bezier(6 * [0] + 2 * [1])
            )

        # Allow for play
        self.tie_fractal_to_root_dots(fractal)
        fractal.set_n_steps(12)

    def init_fractal(self, root_colors=ROOT_COLORS_DEEP):
        plane = self.get_plane()
        fractal = self.get_fractal(
            plane,
            colors=root_colors,
            n_steps=self.n_steps,
        )
        root_dots = self.get_root_dots(plane, fractal)
        self.tie_fractal_to_root_dots(fractal)

        self.plane = plane
        self.fractal = fractal
        self.group = Group(fractal, plane, root_dots)
        self.add(*self.group)

    def get_plane(self):
        plane = ComplexPlane(**self.plane_config)
        plane.add_coordinate_labels(font_size=24)
        self.plane = plane
        return plane

    def get_fractal(self, plane, colors=ROOT_COLORS_DEEP, n_steps=30):
        return NewtonFractal(
            plane,
            colors=colors,
            coefs=self.coefs,
            n_steps=n_steps,
        )

    def get_root_dots(self, plane, fractal):
        self.root_dots = VGroup(*(
            Dot(plane.n2p(root), color=color)
            for root, color in zip(
                coefficients_to_roots(fractal.coefs),
                fractal.colors
            )
        ))
        self.root_dots.set_stroke(BLACK, 5, background=True)
        return self.root_dots

    def tie_fractal_to_root_dots(self, fractal):
        fractal.add_updater(lambda f: f.set_roots([
            self.plane.p2n(dot.get_center())
            for dot in self.root_dots
        ]))

    def on_mouse_press(self, point, button, mods):
        super().on_mouse_press(point, button, mods)
        mob = self.point_to_mobject(point, search_set=self.root_dots)
        if mob is None:
            return
        self.mouse_drag_point.move_to(point)
        mob.add_updater(lambda m: m.move_to(self.mouse_drag_point))
        self.unlock_mobject_data()
        self.lock_static_mobject_data()

    def on_mouse_release(self, point, button, mods):
        super().on_mouse_release(point, button, mods)
        self.root_dots.clear_updaters()


class ChaosOnBoundary(TeacherStudentsScene):
    def construct(self):
        self.teacher_says(
            TexText("Chaos at\\\\the boundary"),
            bubble_config={
                "height": 3,
                "width": 3,
            }
        )
        self.play_all_student_changes("pondering", look_at=self.screen)
        self.wait(3)


class DeepZoomFractal(IntroNewtonFractal):
    coefs = [-1.0, 0.0, 0.0, 1.0, 0.0, 1.0]

    plane_config = {
        "x_range": (-4, 4),
        "y_range": (-4, 4),
        "height": 16 * 1,
        "width": 16 * 1,
        "background_line_style": {
            "stroke_color": GREY_A,
            "stroke_width": 1.0,
        },
        "axis_config": {
            "stroke_width": 1.0,
        }
    }

    def construct(self):
        self.init_fractal(root_colors=ROOT_COLORS_DEEP)
        fractal, plane, root_dots = self.group

        he_tracker = ValueTracker(0)
        frame = self.camera.frame
        zoom_point = np.array([
            # -1.91177811, 0.52197285, 0.
            0.72681252, -0.66973296, 0.
        ], dtype=np.float64)

        initial_fh = FRAME_HEIGHT
        frame.add_updater(lambda m: m.set_height(
            initial_fh * 2**(-he_tracker.get_value()),
        ))
        # rd_height = root_dots.get_height()
        # root_dots.add_updater(lambda m: m.set_height(
        #     rd_height * 2**(he_tracker.get_value() / 8),
        #     about_point=zoom_point
        # ))

        self.add(frame)
        self.play(
            UpdateFromAlphaFunc(
                frame,
                lambda m, a: m.move_to(zoom_point * a),
                run_time=15,
            ),
            ApplyMethod(
                he_tracker.set_value, 14,
                run_time=30,
                rate_func=bezier([0, 0, 1, 1]),
            ),
        )
        self.wait()


class IncreasingStepsNewtonFractal(IntroNewtonFractal):
    play_mode = False

    def construct(self):
        self.init_fractal()
        fractal, plane, root_dots = self.group
        fractal.set_n_steps(0)

        steps_label = VGroup(Integer(0, edge_to_fix=RIGHT), Text("Steps"))
        steps_label.arrange(RIGHT, aligned_edge=UP)
        steps_label.next_to(ORIGIN, UP).to_edge(LEFT)
        steps_label.set_stroke(BLACK, 5, background=True)
        self.add(steps_label)

        step_tracker = ValueTracker(0)
        get_n_steps = step_tracker.get_value
        fractal.add_updater(lambda m: m.set_n_steps(int(get_n_steps())))
        steps_label[0].add_updater(
            lambda m: m.set_value(int(get_n_steps()))
        )
        steps_label[0].add_updater(lambda m: m.set_stroke(BLACK, 5, background=True))

        if self.play_mode:
            self.wait(20)
            for n in range(20):
                step_tracker.set_value(n)
                if n == 1:
                    self.wait(15)
                elif n == 2:
                    self.wait(10)
                else:
                    self.wait()
        else:
            self.play(
                step_tracker.animate.set_value(20),
                run_time=10
            )


class ManyQuestions(Scene):
    def construct(self):
        self.add(FullScreenRectangle())

        questions = VGroup(
            Text("Lower order polynomials?"),
            Text("Do points ever cycle?"),
            Text("Fractal dimension?"),
            Text("Connection to Mandelbrot?"),
        )
        screens = VGroup(*(ScreenRectangle() for q in questions))
        screens.arrange_in_grid(
            v_buff=1.5,
            h_buff=3.0,
        )
        screens.set_fill(BLACK, 1)

        questions.match_width(screens[0])
        for question, screen in zip(questions, screens):
            question.next_to(screen, UP)
            screen.add(question)

        screens.set_height(FRAME_HEIGHT - 0.5)
        screens.center()

        self.play(LaggedStartMap(
            FadeIn, screens,
            lag_ratio=0.9,
        ), run_time=8)
        self.wait()


class WhatsGoingOn(TeacherStudentsScene):
    def construct(self):
        self.screen.set_height(4, about_edge=UL)
        self.screen.set_fill(BLACK, 1)
        self.add(self.screen)

        self.student_says(
            "What the %$!* is\ngoing on?",
            target_mode="angry",
            look_at=self.screen,
            index=2,
            added_anims=[LaggedStart(*(
                pi.change("guilty", self.students[2].eyes)
                for pi in [self.teacher, *self.students[:2]]
            ), run_time=2)]
        )
        self.wait(4)


class EquationToFrame(Scene):
    def construct(self):
        self.add(FullScreenRectangle())
        screens = self.get_screens()
        arrow = Arrow(*screens)

        equation = get_newton_rule()
        equation.next_to(screens[0], UP)

        title = TexText("Unreasonable intricacy")
        title.next_to(screens[1], UP)

        self.wait()
        self.add(screens)
        self.add(equation)
        self.play(
            ShowCreation(arrow),
            FadeTransform(equation.copy(), title),
        )
        self.wait()

    def get_screens(self):
        screens = Square().get_grid(1, 2)
        screens.set_height(6)
        screens.set_width(FRAME_WIDTH - 1, stretch=True)
        screens.set_stroke(WHITE, 3)
        screens.set_fill(BLACK, 1)
        screens.arrange(RIGHT, buff=2.0)
        screens.to_edge(DOWN)
        return screens


class RepeatedNewton(Scene):
    coefs = [1.0, -1.0, 1.0, 0.0, 0.0, 1.0]
    plane_config = {
        "x_range": (-4, 4),
        "y_range": (-2, 2),
        "height": 8,
        "width": 16,
    }
    dots_config = {
        "radius": 0.05,
        "color": GREY_A,
        "gloss": 0.4,
        "shadow": 0.1,
        "opacity": 0.5,
    }
    arrow_style = {
        "stroke_color": WHITE,
        "stroke_opacity": 0.5,
    }
    dot_density = 5.0
    points_scalar = 1.0
    n_steps = 10
    colors = ROOT_COLORS_BRIGHT
    show_coloring = True
    show_arrows = True
    highlight_equation = False
    corner_group_height = 2.0
    step_run_time = 1.0
    show_fractal_background = False

    def construct(self):
        self.add_plane()
        self.add_true_roots()
        self.add_labels()
        if self.show_fractal_background:
            self.add_fractal_background()
        self.add_dots()
        self.run_iterations()
        if self.show_coloring:
            self.color_points()
            self.revert_to_original_positions()

    def add_plane(self):
        plane = self.plane = ComplexPlane(**self.plane_config)
        plane.add_coordinate_labels(font_size=24)

        self.add(plane)

    def add_labels(self):
        eq_label = self.eq_label = Tex(
            "P(z) = " + coefs_to_poly_string(self.coefs),
            font_size=36
        )

        rule_label = self.rule_label = get_newton_rule()
        rule_label.next_to(eq_label, DOWN, MED_LARGE_BUFF)

        corner_rect = SurroundingRectangle(
            VGroup(eq_label, rule_label),
            buff=MED_SMALL_BUFF
        )
        corner_rect.set_fill(BLACK, 0.9)
        corner_rect.set_stroke(WHITE, 1)

        self.corner_group = VGroup(
            corner_rect,
            eq_label,
            rule_label,
        )
        self.corner_group.set_height(self.corner_group_height)
        self.corner_group.to_corner(UL, buff=0)
        self.add(self.corner_group)

    def add_true_roots(self):
        roots = self.roots = coefficients_to_roots(self.coefs)
        root_dots = self.root_dots = VGroup(*(
            glow_dot(self.plane.n2p(root), color=color, opacity_mult=2.0)
            for root, color in zip(roots, self.colors)
        ))

        self.add(root_dots)

    def add_dots(self):
        dots = self.dots = DotCloud(
            self.get_original_points(), **self.dots_config
        )

        self.add(dots, self.corner_group)
        self.play(ShowCreation(dots))

    def get_original_points(self):
        step = 1.0 / self.dot_density
        return self.points_scalar * np.array([
            self.plane.c2p(x, y)
            for x in np.arange(*self.plane.x_range[:2], step)
            for y in np.arange(*self.plane.y_range[:2], step)
        ])

    def run_iterations(self):
        self.points_history = []
        for x in range(self.n_steps):
            self.points_history.append(self.dots.get_points().copy())
            self.take_step(run_time=self.step_run_time)

    def update_z(self, z, epsilon=1e-6):
        denom = dpoly(z, self.coefs)
        if abs(denom) < epsilon:
            denom = epsilon
        return z - poly(z, self.coefs) / denom

    def take_step(self, run_time=1.0):
        plane = self.plane
        points = self.dots.get_points()

        zs = map(plane.p2n, points)
        new_zs = map(self.update_z, zs)
        new_points = list(map(plane.n2p, new_zs))

        added_anims = []
        if self.show_arrows:
            arrows = []
            max_len = 0.5 * plane.get_x_unit_size() / self.dot_density
            for p1, p2 in zip(points, new_points):
                vect = p2 - p1
                norm = get_norm(vect)
                if norm > max_len:
                    vect = normalize(vect) * max_len
                arrows.append(Vector(vect, **self.arrow_style).shift(p1))
            arrows = VGroup(*arrows)
            self.add(arrows, self.dots, self.corner_group)
            self.play(ShowCreation(arrows, lag_ratio=0))
            added_anims.append(FadeOut(arrows))

        self.play(
            self.dots.animate.set_points(new_points),
            *added_anims,
            run_time=run_time,
        )
        self.dots.filter_out(lambda p: get_norm(p) > FRAME_WIDTH)

    def color_points(self):
        root_points = [rd.get_center() for rd in self.root_dots]
        rgbas = list(map(color_to_rgba, self.colors))

        def get_rgba(point):
            norms = [get_norm(point - rp) for rp in root_points]
            return rgbas[np.argmin(norms)]

        rgbas = list(map(get_rgba, self.dots.get_points()))

        fractal = NewtonFractal(
            self.plane,
            coefs=self.coefs,
            colors=self.colors,
            n_steps=0,
        )
        fractal.set_opacity(0)

        self.add(fractal, self.plane, self.dots, self.corner_group)
        radius = self.dots.get_radius()
        self.play(
            fractal.animate.set_opacity(0.5),
            self.dots.animate.set_rgba_array(rgbas).set_radius(1.5 * radius),
        )
        self.play(
            fractal.animate.set_opacity(0),
            self.dots.animate.set_radius(radius),
        )
        self.remove(fractal)

    def revert_to_original_positions(self):
        for ph in self.points_history[::-1]:
            self.play(
                self.dots.animate.set_points(ph),
                run_time=0.5,
            )

    def reveal_fractal(self, **kwargs):
        plane = self.plane

        fractal = self.fractal = self.get_fractal(**kwargs)
        root_dot_backs = VGroup(*(Dot(rd.get_center(), radius=0.1) for rd in self.root_dots))
        root_dot_backs.set_stroke(BLACK, 2)
        root_dot_backs.set_fill(opacity=0)

        plane.generate_target(use_deepcopy=True)
        for lines in plane.target.background_lines, plane.target.faded_lines:
            lines.set_stroke(WHITE)
            for line in lines.family_members_with_points():
                line.set_opacity(line.get_stroke_opacity() * 0.5)

        self.root_dots.generate_target()
        for rd, color in zip(self.root_dots.target, fractal.colors):
            rd.set_fill(color)

        self.add(fractal, *self.mobjects, root_dot_backs)
        self.play(
            FadeIn(fractal),
            FadeOut(self.dots),
            FadeIn(root_dot_backs),
            MoveToTarget(plane),
            MoveToTarget(self.root_dots),
        )
        self.wait()

    def get_fractal(self, **kwargs):
        if "colors" not in kwargs:
            kwargs["colors"] = self.colors
        self.fractal = NewtonFractal(self.plane, coefs=self.coefs, **kwargs)
        return self.fractal

    def add_fractal_background(self):
        fractal = self.get_fractal()
        fractal.set_opacity(0.1)
        fractal.set_n_steps(12)
        boundary = self.fractal_boundary = fractal.copy()
        boundary.set_colors(5 * [WHITE])
        boundary.set_julia_highlight(1e-4)
        boundary.set_opacity(0.25)
        self.add(fractal, boundary, *self.mobjects)


class AmbientQuinticSolving(RepeatedNewton):
    coefs = [-23.125, -11.9375, -6.875, 0.3125, 2.5, 1]
    show_fractal_background = True
    dots_config = {
        "radius": 0.03,
        "color": GREY_A,
        "gloss": 0.4,
        "shadow": 0.1,
        "opacity": 0.5,
    }
    dot_density = 10.0

    def add_labels(self):
        super().add_labels()
        self.corner_group.set_opacity(0)


class WhyNotThisWrapper(VideoWrapper):
    title = "Why not something like this?"
    animate_boundary = False
    title_config = {
        "font_size": 60,
        "color": RED,
    }
    wait_time = 2


class SimplyTendingToNearestRoot(RepeatedNewton):
    def update_z(self, z):
        norms = [abs(r - z) for r in self.roots]
        nearest_root = self.roots[np.argmin(norms)]
        norm = min(norms)
        step_size = np.log(1 + norm * 3) / 3
        return z + step_size * (nearest_root - z)


class UnrelatedIdeas(TeacherStudentsScene):
    def construct(self):
        self.screen.set_height(4, about_edge=UL)
        self.add(self.screen)

        self.play_student_changes(
            "tease", "thinking", "raise_right_hand",
            look_at=self.screen,
            added_anims=[self.teacher.change("happy")]
        )
        self.wait(2)
        self.teacher_says(
            TexText("Unrelated\\\\ideas"),
            bubble_config={
                "height": 3,
                "width": 4,
            },
            added_anims=[
                s.change("sassy", self.teacher.eyes)
                for s in self.students
            ]
        )
        self.play(LaggedStart(
            self.students[2].change("angry"),
            self.teacher.change("guilty"),
            lag_ratio=0.7,
        ))
        self.wait(2)

        self.embed()


class RepeatedNewtonCubic(RepeatedNewton):
    coefs = [-1, 0, 0, 1]
    # colors = [RED_E, GREEN_E, BLUE_E]
    colors = ROOT_COLORS_DEEP[::2]

    def construct(self):
        super().construct()
        self.reveal_fractal()

        frame = self.camera.frame
        self.play(
            frame.animate.move_to([0.86579359, -0.8322599, 0.]).set_height(0.0029955),
            rate_func=bezier([0, 0, 1, 1, 1, 1, 1, 1]),
            run_time=10,
        )


class RepeatedNewtonQuadratic(RepeatedNewton):
    coefs = [-1, 0, 1]
    colors = [RED, BLUE]
    n_steps = 10


class SimpleFractalScene(IntroNewtonFractal):
    colors = ROOT_COLORS_DEEP
    display_polynomial_label = False
    display_root_values = False
    n_steps = 25

    def construct(self):
        self.init_fractal(root_colors=self.colors)
        if self.display_polynomial_label:
            self.add_polynomial_label()
        if self.display_root_values:
            self.add_root_labels()

    def add_polynomial_label(self):
        n = len(self.fractal.roots)
        t2c = {
            f"r_{i + 1}": interpolate_color(self.colors[i], WHITE, 0.5)
            for i in range(n)
        }
        label = Tex(
            "p(z) = ", *(
                f"(z - r_{i})"
                for i in range(1, n + 1)
            ),
            tex_to_color_map=t2c,
            font_size=36
        )
        label.to_corner(UL)
        label.set_stroke(BLACK, 5, background=True)
        self.add(label)

    def add_root_labels(self):
        for n, root_dot in zip(it.count(1), self.root_dots):
            self.add(self.get_root_label(root_dot, n))

    def get_root_label(self, root_dot, n):
        def get_z():
            return self.plane.p2n(root_dot.get_center())
        label = VGroup(
            Tex(f"r_{n} = "),
            DecimalNumber(get_z(), include_sign=True),
        )
        label.scale(0.5)
        label.set_stroke(BLACK, 3, background=True)

        def update_label(label):
            label.arrange(RIGHT, buff=0.1)
            label[0].shift(0.1 * label[0].get_height() * DOWN)
            label.next_to(root_dot, UR, SMALL_BUFF)
            label[1].set_value(get_z())

        label.add_updater(update_label)
        return label


class TwoRootFractal(SimpleFractalScene):
    coefs = [-1.0, 0.0, 1.0]
    colors = [ROOT_COLORS_DEEP[0], ROOT_COLORS_DEEP[4]]
    n_steps = 0  # Doesn't really matter, does it?


class TwoRootFractalWithLabels(TwoRootFractal):
    display_polynomial_label = True
    display_root_values = True


class ThreeRootFractal(SimpleFractalScene):
    coefs = [-1.0, 0.0, 0.0, 1.0]
    colors = ROOT_COLORS_DEEP[::2]
    n_steps = 30


class ThreeRootFractalWithLabels(ThreeRootFractal):
    display_polynomial_label = True
    display_root_values = True


class FromTwoToThree(EquationToFrame):
    def construct(self):
        self.add(FullScreenRectangle())
        screens = self.get_screens()
        arrow = Arrow(*screens)

        quadratic = Tex("x^2 + c_1 x + c_0")
        cubic = Tex("x^3 + c_2 x^2 + c_1 x + c_0")
        quadratic.next_to(screens[0], UP)
        cubic.next_to(screens[1], UP)

        self.add(screens)
        self.add(quadratic, cubic)
        self.play(ShowCreation(arrow))
        self.wait()


class StudentAsksAboutComplexity(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            TexText("Why is it\\\\so complicated?"),
            index=0,
            bubble_config={
                "height": 3,
                "width": 4,
            },
            added_anims=[
                self.students[1].change("confused", self.teacher.eyes),
                self.students[2].change("erm", self.teacher.eyes),
            ],
        )
        self.wait()
        self.play(
            self.teacher.change("shruggie"),
        )
        self.wait()
        self.play(LaggedStart(
            PiCreatureSays(
                self.teacher, TexText("Math is what\\\\it is"),
                target_mode="well",
                bubble_config={
                    "height": 3,
                    "width": 4,
                }
            ),
            self.students[1].change("maybe"),
            self.students[2].change("sassy"),
            lag_ratio=0.7,
        ))
        self.wait(2)

        why = self.students[0].bubble.content[0][:3]
        question = Text("Is this meaningful?")
        question.to_corner(UL)
        question.set_color(YELLOW)
        arrow = Arrow(question, why)
        arrow.set_stroke(YELLOW, 5)

        self.play(
            why.animate.set_color(YELLOW),
            Write(question),
            ShowCreation(arrow),
            LaggedStart(*(
                pi.change(mode, question)
                for pi, mode in zip(self.pi_creatures, ("well", "erm", "sassy", "hesitant"))
            ))
        )
        self.wait(2)

        cross = Cross(question)
        cross.set_stroke(RED, [1, *4 * [8], 1])
        words = Text("Surprisingly answerable!")
        words.next_to(question, RIGHT, LARGE_BUFF)
        new_arrow = Arrow(words[:10], why)
        new_arrow.set_stroke(WHITE, 5)
        self.play(
            RemovePiCreatureBubble(self.teacher, target_mode="erm"),
            ShowCreation(cross),
            FadeIn(words),
            ShowCreation(new_arrow),
        )
        self.wait(2)


class NextVideoWrapper(VideoWrapper):
    title = "Next video"


class PeculiarBoundaryProperty(Scene):
    coefs = [-1, 0, 0, 1]
    colors = [RED_E, TEAL_E, BLUE_E]

    def construct(self):
        # Title
        title = Text("Peculiar property", font_size=60)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        title.set_stroke(BLACK, 5, background=True)
        underline = Underline(title, buff=-0.05)
        underline.set_width(title.get_width() + 1)
        underline.insert_n_curves(20)
        underline.set_stroke(BLUE, [1, *5 * [3], 1])

        subtitle = TexText(
            "Boundary of one color",
            " = "
            "Boundary of any other",
            tex_to_color_map={
                "one color": BLUE_D,
                "any other": RED_D,
            }
        )
        subtitle.next_to(underline, DOWN, MED_LARGE_BUFF)

        # Setup for planes
        grid = VGroup(*(
            ComplexPlane(
                x_range=(-3, 3),
                y_range=(-2, 2),
            )
            for n in range(6)
        ))

        grid.arrange_in_grid(2, 3, v_buff=2, h_buff=3)
        grid.set_width(FRAME_WIDTH - 2)
        grid.to_edge(DOWN, buff=MED_LARGE_BUFF)

        arrows = VGroup()
        bound_words = VGroup()
        for p1, p2 in zip(grid[:3], grid[3:]):
            arrow = Arrow(p1, p2, stroke_width=4, buff=0.1)
            arrows.add(arrow)
            bound_word = Text("Boundary", font_size=24)
            bound_word.next_to(arrow, RIGHT, buff=SMALL_BUFF)
            bound_words.add(bound_word)

        low_equals = VGroup(
            Tex("=").move_to(grid[3:5]),
            Tex("=").move_to(grid[4:6]),
        )

        # Fractals
        fractals = Group(*(
            NewtonFractal(plane, coefs=self.coefs, colors=self.colors)
            for plane in grid
        ))
        alpha = 0.2
        for k in 0, 3:
            fractals[0 + k].set_opacities(alpha, 1, alpha)
            fractals[1 + k].set_opacities(alpha, alpha, 1)
            fractals[2 + k].set_opacities(1, alpha, alpha)

        boxes = VGroup(*(
            SurroundingRectangle(fractal, buff=0)
            for fractal in fractals
        ))
        boxes.set_stroke(GREY_B, 1)

        # Initial fractal
        big_plane = grid[0].deepcopy()
        big_plane.set_height(6.5)
        big_plane.center().to_edge(DOWN)
        big_fractal = NewtonFractal(big_plane, coefs=self.coefs, colors=self.colors)
        big_julia = big_fractal.copy()
        big_julia.set_julia_highlight(1e-3)
        big_julia.set_colors(3 * [WHITE])

        self.add(big_fractal)

        # Animations
        def get_show_border_anims(fractal):
            f_copy = fractal.copy()
            fractal.set_julia_highlight(5e-3)
            fractal.set_colors(3 * [WHITE])
            return (FadeOut(f_copy), GrowFromCenter(fractal))

        def high_to_low_anims(index):
            return (
                ShowCreation(arrows[index]),
                FadeIn(bound_words[index]),
                TransformFromCopy(fractals[index], fractals[index + 3]),
                TransformFromCopy(boxes[index], boxes[index + 3]),
            )

        self.add(underline, title)
        self.play(
            ShowCreation(underline),
            GrowFromCenter(big_julia, run_time=4)
        )
        self.play(
            big_julia.animate.set_julia_highlight(0.02).set_colors(CUBIC_COLORS).set_opacity(0)
        )
        self.wait()

        self.play(
            big_fractal.animate.set_opacities(alpha, alpha, 1)
        )
        self.wait()

        self.play(
            ReplacementTransform(big_fractal, fractals[1]),
            FadeIn(subtitle[:2]),
            ReplacementTransform(
                boxes[1].copy().replace(big_fractal).set_opacity(0),
                boxes[1],
            ),
        )
        self.play(*high_to_low_anims(1))
        self.play(*get_show_border_anims(fractals[4]))
        self.wait(2)

        subtitle[2:].set_opacity(0)
        self.add(subtitle[2:])
        for i in 2, 0:
            self.play(
                FadeIn(fractals[i]),
                FadeIn(boxes[i]),
                subtitle[2:].animate.set_opacity(1),
            )
            self.play(*high_to_low_anims(i))
            self.play(*get_show_border_anims(fractals[i + 3]))
            self.wait()

        self.play(Write(low_equals))


class DefineBoundary(Scene):
    def construct(self):
        # Add set
        blob = VMobject()
        blob.set_fill(BLUE_E, 1)
        blob.set_stroke(width=0)
        blob.set_points_as_corners([
            (1 + 0.3 * random.random()) * p
            for p in compass_directions(12)
        ])
        blob.close_path()
        blob.set_height(3)
        blob.set_width(1.0, stretch=True)
        blob.move_to(2 * RIGHT)
        blob.apply_complex_function(np.exp)
        blob.make_smooth()
        blob.rotate(90 * DEGREES)
        blob.center()
        blob.set_height(4)
        blob.insert_n_curves(50)

        set_text = Text("Set", font_size=72)
        set_text.set_stroke(BLACK, 3, background=True)
        set_text.move_to(interpolate(blob.get_top(), blob.get_bottom(), 0.35))

        self.add(blob)
        self.add(set_text)

        # Preview boundary
        point = Dot(radius=0.05)
        point.move_to(blob.get_start())

        boundary_word = Text("Boundary")
        boundary_word.set_color(YELLOW)
        boundary_word.next_to(blob, LEFT)
        outline = blob.copy()
        outline.set_fill(opacity=0)
        outline.set_stroke(YELLOW, 2)

        self.add(point)

        kw = {
            "rate_func": bezier([0, 0, 1, 1]),
            "run_time": 5,
        }
        self.play(
            FadeIn(boundary_word),
            ShowCreation(outline, **kw),
            MoveAlongPath(point, blob, **kw)
        )
        self.play(FadeOut(outline))

        # Mention formality
        boundary_word.generate_target()
        boundary_word.target.to_corner(UL)
        formally_word = Text("More formally")
        formally_word.next_to(boundary_word.target, DOWN, aligned_edge=LEFT)

        self.play(
            MoveToTarget(boundary_word),
            FadeTransform(boundary_word.copy(), formally_word)
        )
        self.wait()

        # Draw circle
        circle = Circle()
        circle.move_to(point)
        circle.set_stroke(TEAL, 3.0)

        self.play(
            ShowCreation(circle),
            point.animate.scale(0.5),
        )
        self.wait()
        group = VGroup(blob, set_text)
        self.add(group, point, circle)
        self.play(
            ApplyMethod(
                group.scale, 2, {"about_point": point.get_center()},
                run_time=4
            ),
            ApplyMethod(
                circle.set_height, 0.5,
                run_time=2,
            ),
        )

        # Labels
        inside_words = Text("Points inside", font_size=36)
        outside_words = Text("Points outside", font_size=36)
        inside_words.next_to(circle, DOWN, buff=0.5).shift(0.5 * LEFT)
        outside_words.next_to(circle, UP, buff=0.5).shift(0.5 * RIGHT)
        inside_arrow = Arrow(
            inside_words, point,
            stroke_width=3,
            buff=0.1,
        )
        outside_arrow = Arrow(
            outside_words, point,
            stroke_width=3,
            buff=0.1,
        )

        self.play(
            FadeIn(inside_words),
            ShowCreation(inside_arrow)
        )
        self.play(
            FadeIn(outside_words),
            ShowCreation(outside_arrow)
        )
        self.wait()

        # Show interior
        point_group = VGroup(point, circle)

        self.play(
            point_group.animate.shift(circle.get_height() * DOWN / 4),
            LaggedStartMap(
                FadeOut, VGroup(inside_words, inside_arrow, outside_words, outside_arrow)
            )
        )
        self.wait()
        self.play(circle.animate.set_height(0.2))
        self.wait()

        # Show exterior
        point_group.generate_target()
        point_group.target.move_to(blob.get_start() + 0.25 * UP)
        point_group.target[1].set_height(1.0)

        self.play(MoveToTarget(point_group))
        self.wait()
        self.play(circle.animate.set_height(0.2))
        self.wait()

        # Back to boundary
        self.play(point_group.animate.move_to(blob.get_start()))
        frame = self.camera.frame
        frame.generate_target()
        frame.target.set_height(0.2)
        frame.target.move_to(point)
        point_group.generate_target()
        point_group.target.set_height(0.2 / 8)
        point_group.target[1].set_stroke(width=0.1)

        self.play(MoveToTarget(point_group))
        self.play(
            MoveToTarget(frame),
            run_time=4
        )


class VariousCirclesOnTheFractal(SimpleFractalScene):
    coefs = [-1.0, 0.0, 0.0, 1.0]
    colors = CUBIC_COLORS
    sample_density = 0.02

    def construct(self):
        super().construct()
        frame = self.camera.frame
        plane = self.plane
        fractal = self.fractal
        frame.save_state()

        # Setup samples
        n_steps = 20
        density = self.sample_density
        samples = np.array([
            [complex(x, y), 0]
            for x in np.arange(0, 2, density)
            for y in np.arange(0, 2, density)
        ])
        roots = coefficients_to_roots(self.coefs)
        for i in range(len(samples)):
            z = samples[i, 0]
            for n in range(n_steps):
                z = z - poly(z, self.coefs) / dpoly(z, self.coefs)
            norms = [abs(z - root) for root in roots]
            samples[i, 1] = np.argmin(norms)

        unit_size = plane.get_x_unit_size()

        circle = Circle()
        circle.set_stroke(WHITE, 3.0)
        circle.move_to(2 * UR)

        words = VGroup(
            Text("#Colors inside: "),
            Integer(3),
        )
        words.arrange(RIGHT)
        words[1].align_to(words[0][-2], DOWN)
        height_ratio = words.get_height() / FRAME_HEIGHT

        def get_interior_count(circle):
            radius = circle.get_height() / 2 / unit_size
            norms = abs(samples[:, 0] - plane.p2n(circle.get_center()))
            true_result = len(set(samples[norms < radius, 1]))
            # In principle this would work, but the samples are not perfect
            return 3 if true_result > 1 else 1

        def get_frame_ratio():
            return frame.get_height() / FRAME_HEIGHT

        def update_words(words):
            words.set_height(height_ratio * frame.get_height())
            ratio = get_frame_ratio()
            words.next_to(circle, UP, buff=SMALL_BUFF * ratio)
            count = get_interior_count(circle)
            words[1].set_value(count)
            words.set_stroke(BLACK, 5 * ratio, background=True)
            return words

        words.add_updater(update_words)

        circle.add_updater(lambda m: m.set_stroke(width=3.0 * get_frame_ratio()))

        self.play(ShowCreation(circle))
        self.play(FadeIn(words))
        self.wait()

        self.play(circle.animate.set_height(0.25))
        self.wait()
        point = plane.c2p(0.5, 0.5)
        self.play(circle.animate.move_to(point))
        self.play(frame.animate.set_height(2).move_to(point))
        self.wait()
        point = plane.c2p(0.25, 0.4)
        self.play(circle.animate.move_to(point).set_height(0.1))
        self.wait()
        for xy in (0.6, 0.4), (0.2, 0.6):
            self.play(
                circle.animate.move_to(plane.c2p(*xy)),
                run_time=4
            )
            self.wait()

        # Back to larger
        self.play(
            Restore(frame),
            circle.animate.set_height(0.5)
        )
        self.wait()

        # Show smooth boundary
        count_tracker = ValueTracker(3)
        words.add_updater(lambda m: m[1].set_value(count_tracker.get_value()))

        def change_count_at(new_value, alpha):
            curr_value = count_tracker.get_value()
            return UpdateFromAlphaFunc(
                count_tracker,
                lambda m, a: m.set_value(curr_value if a < alpha else new_value)
            )

        fractal.set_n_steps(10)
        self.play(
            fractal.animate.set_n_steps(3),
            run_time=2
        )
        self.play(
            circle.animate.move_to(plane.c2p(0, 0.3)),
            change_count_at(2, 0.75),
            run_time=2
        )
        self.wait()
        self.play(
            circle.animate.move_to(plane.c2p(0, 0)),
            change_count_at(3, 0.5),
            run_time=2
        )
        self.wait()
        self.play(
            circle.animate.move_to(plane.c2p(-0.6, 0.2)),
            Succession(
                change_count_at(2, 0.9),
                change_count_at(3, 0.7),
            ),
            run_time=3
        )
        self.play(
            circle.animate.set_height(0.1).move_to(plane.c2p(-0.6, 0.24)),
            change_count_at(2, 0.8),
            frame.animate.set_height(2.5).move_to(plane.c2p(-0.5, 0.5)),
            run_time=3
        )
        self.wait(2)
        self.play(
            fractal.animate.set_n_steps(20),
            change_count_at(3, 0.1),
            run_time=3,
        )
        self.wait()

        # Just show boundary
        boundary = fractal.copy()
        boundary.set_colors(3 * [WHITE])
        boundary.add_updater(
            lambda m: m.set_julia_highlight(get_frame_ratio() * 1e-3)
        )
        boundary.set_n_steps(50)

        frame.generate_target()
        frame.target.set_height(0.0018),
        frame.target.move_to([-1.15535091, 0.23001433, 0.])
        self.play(
            FadeOut(circle),
            FadeOut(words),
            FadeOut(self.root_dots),
            GrowFromCenter(boundary, run_time=3),
            fractal.animate.set_opacity(0.35),
            MoveToTarget(
                frame,
                run_time=10,
                rate_func=bezier([0, 0, 1, 1, 1, 1, 1])
            ),
        )
        self.wait()


class ArtPuzzle(Scene):
    def construct(self):
        words = VGroup(
            Text("Art Puzzle:", font_size=60),
            TexText("- Use $\\ge 3$ colors"),
            TexText("- Boundary of one color = Boundary of all"),
        )
        words.set_color(BLACK)
        words.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        words[1:].shift(0.5 * DOWN + 0.5 * RIGHT)
        words.to_corner(UL)

        for word in words:
            self.play(FadeIn(word, lag_ratio=0.1))
            self.wait()


class ZoomInOnCubic(ThreeRootFractal):
    colors = CUBIC_COLORS
    coefs = [complex(0, -1), 0, 0, 1]
    n_steps = 30

    def construct(self):
        super().construct()
        frame = self.camera.frame

        height_exp_tracker = ValueTracker()
        get_height_exp = height_exp_tracker.get_value
        center_tracker = VectorizedPoint(ORIGIN)

        frame.add_updater(lambda m: m.move_to(center_tracker))
        frame.add_updater(lambda m: m.set_height(FRAME_HEIGHT * 2**(-get_height_exp())))

        self.play(
            ApplyMethod(center_tracker.move_to, [0.2986952, 1.11848235, 0], run_time=4),
            ApplyMethod(
                height_exp_tracker.set_value, 7,
                run_time=15,
                rate_func=bezier([0, 0, 1, 1]),
            ),
        )
        self.wait()


class BlobsOnBlobsOnBlobs(Scene):
    def construct(self):
        words = TexText(
            "Blobs", *(
                " on blobs " + ("\\\\" if n == 2 else "")
                for n in range(6)
            ),
            "..."
        )
        words.set_width(FRAME_WIDTH - 2)
        words.to_edge(UP)
        words.set_color(BLACK)
        self.add(words[0])
        for word in words[1:]:
            self.play(FadeIn(word, 0.25 * UP))
        self.wait()


class FractalDimensionWords(Scene):
    def construct(self):
        text = TexText("Fractal dimension $\\approx$ 1.44", font_size=60)
        text.to_corner(UL)
        self.play(Write(text))
        self.wait()


class ThinkAboutWhatPropertyMeans(TeacherStudentsScene):
    def construct(self):
        self.screen.set_height(4, about_edge=UL)
        self.add(self.screen)
        image = ImageMobject("NewtonBoundaryProperty")
        image.replace(self.screen)
        self.add(image)

        self.teacher_says(
            TexText("Think about what\\\\this tells us."),
            bubble_config={
                "height": 3,
                "width": 4,
            }
        )
        self.play_student_changes(
            "pondering", "thinking", "pondering",
            look_at=self.screen
        )
        self.wait(4)


class InterpretBoundaryProperty(RepeatedNewton):
    plane_config = {
        "x_range": (-4, 4),
        "y_range": (-2, 2),
        "height": 12,
        "width": 24,
    }
    n_steps = 15

    def construct(self):
        self.add_plane()
        plane = self.plane
        plane.shift(2 * RIGHT)
        self.add_true_roots()
        self.add_labels()
        self.add_fractal_background()

        # Show sensitive point
        point = plane.c2p(-0.8, 0.4)
        dots = self.dots = DotCloud()
        dots.set_points([
            [r * math.cos(theta), r * math.sin(theta), 0]
            for r in np.linspace(0, 1, 20)
            for theta in np.linspace(0, TAU, int(r * 20)) + random.random() * TAU
        ])
        dots.set_height(2).center()
        dots.filter_out(lambda p: get_norm(p) > 1)
        dots.set_height(0.3)
        dots.set_radius(0.04)
        dots.make_3d()
        dots.set_color(GREY_A)
        dots.move_to(point)

        sensitive_words = Text("Sensitive area")
        sensitive_words.next_to(dots, RIGHT, buff=SMALL_BUFF)
        sensitive_words.set_stroke(BLACK, 5, background=True)

        def get_arrows():
            root_dots = self.root_dots
            if plane.p2n(dots.get_center()).real < -1.25:
                root_dots = [root_dots[4]]
            return VGroup(*(
                Arrow(
                    dots, root_dot,
                    buff=0.1,
                    stroke_color=root_dot[0].get_color()
                )
                for root_dot in root_dots
            ))

        arrows = get_arrows()

        self.play(
            FadeIn(dots, scale=2),
            FadeIn(sensitive_words, shift=0.25 * UP)
        )
        self.wait()
        self.play(ShowCreation(arrows[2]))
        self.play(ShowCreation(arrows[4]))
        self.wait()
        self.play(
            FadeOut(sensitive_words),
            LaggedStartMap(ShowCreation, VGroup(*(
                arrows[i] for i in (0, 1, 3)
            )))
        )
        self.wait()
        arrows.add_updater(lambda m: m.become(get_arrows()))
        self.add(arrows)

        self.play(dots.animate.move_to(plane.c2p(-1.4, 0.4)), run_time=3)
        self.wait()
        self.play(dots.animate.move_to(point), run_time=3)
        self.wait()

        not_allowed = Text("Not allowed!")
        not_allowed.set_color(RED)
        not_allowed.set_stroke(BLACK, 8, background=True)
        not_allowed.next_to(dots, RIGHT, SMALL_BUFF)

        arrows.clear_updaters()
        self.play(
            arrows[:2].animate.set_opacity(0),
            FadeIn(not_allowed, scale=0.7)
        )
        self.wait()
        self.play(FadeOut(arrows), FadeOut(not_allowed))

        # For fun
        self.run_iterations()


class CommentsOnNaming(Scene):
    def construct(self):
        self.setup_table()
        self.show_everyone()

    def setup_table(self):
        titles = VGroup(
            TexText("How it started", font_size=60),
            TexText("How it's going", font_size=60),
        )
        titles.to_edge(UP, buff=MED_SMALL_BUFF)
        titles.set_color(GREY_A)
        titles[0].set_x(-FRAME_WIDTH / 4)
        titles[1].set_x(FRAME_WIDTH / 4)
        h_line = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        h_line.next_to(titles, DOWN).set_x(0)
        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        lines = VGroup(h_line, v_line)
        lines.set_stroke(WHITE, 2)

        self.left_point = [-FRAME_WIDTH / 4, -1, 0]
        self.right_point = [FRAME_WIDTH / 4, -1, 0]

        self.add(titles, lines)

    def show_everyone(self):
        # Newton
        newton = get_figure(
            "Newton", "Isaac Newton", "1643-1727",
            height=4,
        )
        newton.move_to(self.left_point)

        newton_formula = get_newton_rule(var="x")
        newton_formula.next_to(newton, UP)

        nf_label = TexText("``Newton's'' fractal")
        nf_label.align_to(newton_formula, UP)
        nf_label.set_x(self.right_point[0])

        self.play(
            FadeIn(newton_formula),
            LaggedStartMap(FadeIn, newton)
        )
        self.wait()
        self.play(Write(nf_label))
        self.wait(2)

        # Hamilton
        hamilton = get_figure(
            "Hamilton", "William Rowan Hamilton", "1805 - 1865",
            height=4,
        )
        hamilton.move_to(self.left_point)
        hamiltons_equation = Tex(
            r"\frac{\mathrm{d} \boldsymbol{q}}{\mathrm{d} t}=\frac{\partial \mathcal{H}}{\partial \boldsymbol{p}}, \quad \frac{\mathrm{d} \boldsymbol{p}}{\mathrm{d} t}=-\frac{\partial \mathcal{H}}{\partial \boldsymbol{q}}"
        )
        hamiltons_equation.match_width(hamilton[0])
        hamiltons_equation.next_to(hamilton, UP)

        hamiltonians = Text("Hamiltonians")
        hamiltonians.move_to(nf_label)

        self.play(
            LaggedStart(
                FadeOut(newton, shift=0.25 * LEFT),
                FadeOut(newton_formula, shift=0.25 * LEFT),
                FadeOut(nf_label, shift=0.25 * RIGHT),
            ),
            LaggedStart(
                FadeIn(hamilton, shift=0.25 * LEFT),
                FadeIn(hamiltons_equation, shift=0.25 * LEFT),
                FadeIn(hamiltonians, shift=0.25 * RIGHT),
            )
        )
        self.wait(2)

        # Fourier
        fourier = get_figure(
            "Joseph Fourier", "Joseph Fourier", "1768-1830",
            height=4
        )
        fourier.move_to(self.left_point)
        fourier_transform = Tex(
            r"f(t)=\int_{0}^{\infty}(a(\lambda) \cos (2 \pi \lambda t)+b(\lambda) \sin (2 \pi \lambda t)) d \lambda"
        )
        fourier_transform.set_width(fourier.get_width() * 1.5)
        fourier_transform.next_to(fourier, UP)

        FFT = Text("FFT")
        FFT.move_to(hamiltonians)

        FFT_diagram = ImageMobject("FFT_Diagram")
        FFT_diagram.move_to(self.right_point),

        self.play(
            LaggedStart(
                FadeOut(hamilton, shift=0.25 * LEFT),
                FadeOut(hamiltons_equation, shift=0.25 * LEFT),
                FadeOut(hamiltonians, shift=0.25 * RIGHT),
            ),
            LaggedStart(
                FadeIn(fourier, shift=0.25 * LEFT),
                FadeIn(fourier_transform, shift=0.25 * LEFT),
                FadeIn(FFT, shift=0.25 * RIGHT),
            ),
            FadeIn(FFT_diagram),
        )
        self.wait(2)

        # Everyone
        people = Group(newton, hamilton, fourier)
        people.generate_target()
        people.target.arrange(DOWN, buff=LARGE_BUFF)
        people.target.set_height(6.4)
        people.target.move_to(self.left_point)
        people.target.to_edge(DOWN, buff=SMALL_BUFF)

        self.play(
            FadeOut(fourier_transform),
            FadeOut(FFT),
            MoveToTarget(people, run_time=2),
            FFT_diagram.animate.scale(1 / 3).match_y(people.target[2]),
        )

        arrow = Arrow(
            fourier, FFT_diagram,
            buff=1.0,
            stroke_width=8
        )
        arrows = VGroup(
            arrow.copy().match_y(newton),
            arrow.copy().match_y(hamilton),
            arrow,
        )
        self.play(LaggedStartMap(ShowCreation, arrows, lag_ratio=0.5, run_time=3))
        self.wait()


class MakeFunOfNextVideo(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            TexText("``Next part''...I've\\\\heard that before."),
            target_mode="sassy",
            index=2,
            added_anims=[LaggedStart(
                self.teacher.change("guilty"),
                self.students[0].change("sassy"),
                self.students[1].change("hesitant"),
            )]
        )
        self.wait()
        self.teacher_says(
            TexText("Wait, for real\\\\this time!"),
            bubble_config={
                "height": 3,
                "width": 3,
            },
            target_mode="speaking",
            added_anims=[
                self.students[0].change("hesitant"),
            ]
        )
        self.wait(3)


class Part1EndScroll(PatreonEndScreen):
    CONFIG = {
        "title_text": "",
        "scroll_time": 60,
        "show_pis": False,
    }


class Thanks(Scene):
    def construct(self):
        morty = Mortimer(mode="happy")
        thanks = Text("Thank you")
        thanks.next_to(morty, LEFT)
        self.play(
            morty.change("gracious"),
            FadeIn(thanks, lag_ratio=0.1)
        )
        for n in range(5):
            self.play(morty.animate.look([DL, DR][n % 2]))
            self.wait(random.random() * 5)
            self.play(Blink(morty))


class HolomorphicDynamics(Scene):
    def construct(self):
        self.ask_about_property()
        self.repeated_functions()

    def ask_about_property(self):
        back_plane = FullScreenRectangle()
        self.add(back_plane)

        image = ImageMobject("NewtonBoundaryProperty")
        border = SurroundingRectangle(image, buff=0)
        border.set_stroke(WHITE, 2)
        image = Group(border, image)
        image.set_height(FRAME_HEIGHT)

        image.generate_target()
        image.target.set_height(6)
        image.target.to_corner(DL)

        question = Text("Why is this true?")
        question.to_corner(UR)
        arrow = Arrow(
            question.get_left(), image.target.get_top() + RIGHT,
            path_arc=45 * DEGREES
        )

        self.play(
            image.animate.set_height(6).to_corner(DL),
            Write(question),
            ShowCreation(arrow, rate_func=squish_rate_func(smooth, 0.5, 1), run_time=2)
        )
        self.wait()

        title = self.title = Text("Holomorphic Dynamics", font_size=60)
        title.to_edge(UP)

        self.play(
            image.animate.set_height(1).to_corner(DL),
            FadeOut(question, shift=DL, scale=0.2),
            FadeOut(arrow, shift=DL, scale=0.2),
            FadeIn(title, shift=3 * DL, scale=0.5),
            FadeOut(back_plane),
        )
        self.wait()

        self.image = image

    def repeated_functions(self):
        basic_expr = Tex(
            "z", "\\rightarrow ", " f(z)"
        )
        fz = basic_expr.get_part_by_tex("f(z)")
        basic_expr.next_to(self.title, DOWN, LARGE_BUFF)
        basic_expr.to_edge(LEFT, buff=LARGE_BUFF)
        brace = Brace(fz, DOWN)
        newton = Tex("z - {P(z) \\over P'(z)}")
        newton.next_to(brace, DOWN)
        newton.align_to(basic_expr[1], LEFT)
        newton_example = Tex("z - {z^3 + z - 1 \\over 3z^2 + 1}")
        eq = Tex("=").rotate(PI / 2)
        eq.next_to(newton, DOWN)
        newton_example.next_to(eq, DOWN)

        newton_group = VGroup(newton, eq, newton_example)
        newton_group.generate_target()
        newton_group.target[1].rotate(-PI / 2)
        newton_group.target.arrange(RIGHT, buff=0.2)
        newton_group.target[2].shift(SMALL_BUFF * UP)
        newton_group.target.scale(0.7)
        newton_group.target.to_corner(DL)

        mandelbrot = Tex("z^2 + c")
        mandelbrot.next_to(brace, DOWN)

        exponential = Tex("a^z")
        exponential.next_to(brace, DOWN)

        self.play(
            FadeIn(basic_expr),
            FadeOut(self.image)
        )
        self.wait()
        self.describe_holomorphic(fz, brace)
        self.wait()
        self.play(
            FadeIn(newton),
        )
        self.play(
            FadeIn(eq),
            FadeIn(newton_example),
        )
        self.wait()
        self.play(
            MoveToTarget(newton_group),
            FadeIn(mandelbrot, DOWN),
        )
        self.wait()
        self.play(
            mandelbrot.animate.scale(0.7).next_to(newton, UP, LARGE_BUFF, LEFT),
            FadeIn(exponential, DOWN)
        )
        self.wait()

        # Show fractals
        rhss = VGroup(exponential, mandelbrot, newton)
        f_eqs = VGroup()
        lhss = VGroup()
        for rhs in rhss:
            rhs.generate_target()
            if rhs is not exponential:
                rhs.target.scale(1 / 0.7)
            lhs = Tex("f(z) = ")
            lhs.next_to(rhs.target, LEFT)
            f_eqs.add(VGroup(lhs, rhs.target))
            lhss.add(lhs)
        f_eqs.arrange(RIGHT, buff=1.5)
        f_eqs.next_to(self.title, DOWN, MED_LARGE_BUFF)

        rects = ScreenRectangle().replicate(3)
        rects.arrange(DOWN, buff=0.5)
        rects.set_height(6.5)
        rects.next_to(ORIGIN, RIGHT, MED_LARGE_BUFF)
        rects.to_edge(DOWN, MED_SMALL_BUFF)
        rects.set_stroke(WHITE, 1)
        arrows = VGroup()
        for rect, f_eq in zip(rects, f_eqs):
            arrow = Vector(0.7 * RIGHT)
            arrow.next_to(rect, LEFT)
            arrows.add(arrow)
            f_eq.next_to(arrow, LEFT)

        self.play(
            LaggedStartMap(MoveToTarget, rhss),
            LaggedStartMap(Write, lhss),
            LaggedStartMap(FadeIn, rects),
            LaggedStartMap(ShowCreation, arrows),
            FadeOut(brace),
            basic_expr.animate.to_edge(UP),
            FadeOut(newton_group[1:]),
        )
        self.wait()

    def describe_holomorphic(self, fz, brace):
        self.title.set_stroke(BLACK, 5, background=True)
        word = self.title.get_part_by_text("Holomorphic")
        underline = Underline(word, buff=-0.05)
        underline.scale(1.2)
        underline.insert_n_curves(40)
        underline.set_stroke(YELLOW, [1, *6 * [3], 1])

        self.add(underline, self.title)
        self.play(
            word.animate.set_fill(YELLOW),
            ShowCreation(underline)
        )

        in_words = Text("Complex\ninputs", font_size=36)
        in_words.to_corner(UL)
        in_arrow = Arrow(
            in_words.get_right(),
            fz[2].get_top(),
            path_arc=-80 * DEGREES,
            buff=0.2,
        )
        VGroup(in_words, in_arrow).set_color(YELLOW)

        out_words = Text("Complex\noutputs", font_size=36)
        out_words.next_to(brace, DOWN)
        out_words.set_color(YELLOW)

        f_prime = TexText("$f'(z)$ exists")
        f_prime.set_color(YELLOW)
        f_prime.next_to(underline, DOWN, MED_LARGE_BUFF)
        f_prime.match_y(fz)

        self.wait()
        self.play(
            Write(in_words),
            ShowCreation(in_arrow),
            run_time=1,
        )
        self.play(
            GrowFromCenter(brace),
            FadeIn(out_words, lag_ratio=0.05)
        )
        self.wait()
        self.play(FadeIn(f_prime, 0.5 * DOWN))
        self.wait()

        self.play(
            LaggedStartMap(FadeOut, VGroup(
                in_words, in_arrow, out_words, f_prime, underline,
            )),
            word.animate.set_fill(WHITE)
        )


class AmbientRepetition(Scene):
    n_steps = 30

    def construct(self):
        plane = ComplexPlane((-2, 2), (-2, 2))
        plane.set_height(FRAME_HEIGHT)
        plane.add_coordinate_labels(font_size=24)
        self.add(plane)

        font_size = 36

        z0 = complex(0, 0)
        dot = Dot(color=BLUE)
        dot.move_to(plane.n2p(z0))
        z_label = Tex("z", font_size=font_size)
        z_label.set_stroke(BLACK, 5, background=True)
        z_label.next_to(dot, UP, SMALL_BUFF)
        self.add(dot, z_label)

        def func(z):
            return z**2 + complex(-0.6436875, -0.441)

        def get_new_point():
            z = plane.p2n(dot.get_center())
            return plane.n2p(func(z))

        for n in range(self.n_steps):
            new_point = get_new_point()
            arrow = Arrow(dot.get_center(), new_point, buff=dot.get_height() / 2)

            dot_copy = dot.copy()
            dot_copy.move_to(new_point)
            dot_copy.set_color(YELLOW)
            fz_label = Tex("f(z)", font_size=font_size)
            fz_label.set_stroke(BLACK, 8, background=True)
            fz_label.next_to(dot_copy, UP, SMALL_BUFF)

            self.add(dot, dot_copy, arrow, z_label)
            self.play(
                ShowCreation(arrow),
                TransformFromCopy(dot, dot_copy),
                FadeInFromPoint(fz_label, z_label.get_center()),
            )
            self.wait(0.5)
            to_fade = VGroup(
                dot.copy(), z_label.copy(),
                dot_copy, arrow, fz_label,
            )
            dot.move_to(dot_copy)
            z_label.next_to(dot, UP, SMALL_BUFF)
            self.remove(z_label)
            self.play(
                *map(FadeOut, to_fade),
                FadeIn(z_label),
            )

        self.embed()


class BriefMandelbrot(Scene):
    n_iterations = 30

    def construct(self):
        self.add_plane()
        self.add_process_description()
        self.show_iterations()
        self.wait(10)  # Time to play
        self.add_mandelbrot_image()

    def add_plane(self):
        plane = self.plane = ComplexPlane((-2, 1), (-2, 2))

        plane.set_height(4)
        plane.scale(FRAME_HEIGHT / 2.307)
        plane.next_to(2 * LEFT, RIGHT, buff=0)
        plane.add_coordinate_labels(font_size=24)
        self.add(plane)

    def add_process_description(self):
        kw = {
            "tex_to_color_map": {
                "{c}": YELLOW,
            }
        }
        terms = self.terms = VGroup(
            Tex("z_{n + 1} = z_n^2 + {c}", **kw),
            Tex("z_0 = 0", **kw),
            Tex("{c} \\text{ can be changed}", **kw),
        )
        terms.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        terms.next_to(self.plane, LEFT, MED_LARGE_BUFF)

        self.add(terms)

    def show_iterations(self):
        plane = self.plane

        c0 = complex(-0.2, 0.95)

        c_dot = self.c_dot = Dot()
        c_dot.set_fill(YELLOW)
        c_dot.set_stroke(BLACK, 5, background=True)
        c_dot.move_to(plane.n2p(c0))

        lines = VGroup()
        lines.set_stroke(background=True)

        def get_c():
            return plane.p2n(c_dot.get_center())

        def update_lines(lines):
            z1 = 0
            c = get_c()
            new_lines = []

            for n in range(self.n_iterations):
                try:
                    z2 = z1**2 + c
                    new_lines.append(Line(
                        plane.n2p(z1),
                        plane.n2p(z2),
                        stroke_color=GREY,
                        stroke_width=2,
                    ))
                    new_lines.append(Dot(
                        plane.n2p(z2),
                        fill_color=YELLOW,
                        fill_opacity=0.5,
                        radius=0.05,
                    ))
                    z1 = z2
                except Exception:
                    pass

            lines.set_submobjects(new_lines)

        update_lines(lines)
        self.add(lines[:2], c_dot)
        last_dot = Dot(plane.n2p(0)).scale(0)
        for line, dot in zip(lines[0:20:2], lines[1:20:2]):
            self.add(line, dot, c_dot)
            self.play(
                ShowCreation(line),
                TransformFromCopy(last_dot, dot)
            )
            last_dot = dot
        self.remove(*lines)
        lines.add_updater(update_lines)
        self.add(lines, c_dot)

    def add_mandelbrot_image(self):
        image = ImageMobject("MandelbrotSet")
        image.set_height(FRAME_HEIGHT)
        image.shift(self.plane.n2p(-0.7) - image.get_center())

        rect = FullScreenFadeRectangle()
        rect.set_fill(BLACK, 1)
        rect.next_to(self.plane, LEFT, buff=0)

        self.add(image, rect, *self.mobjects)
        self.play(
            FadeIn(image, run_time=2),
            self.plane.animate.set_opacity(0.5)
        )

    def on_mouse_press(self, point, button, mods):
        # TODO, copy-pasted, should factor out
        super().on_mouse_press(point, button, mods)
        mob = self.point_to_mobject(point, search_set=[self.c_dot])
        if mob is None:
            return
        self.mouse_drag_point.move_to(point)
        mob.add_updater(lambda m: m.move_to(self.mouse_drag_point))
        self.unlock_mobject_data()
        self.lock_static_mobject_data()

    def on_mouse_release(self, point, button, mods):
        super().on_mouse_release(point, button, mods)
        self.c_dot.clear_updaters()


class CyclicAttractor(RepeatedNewton):
    coefs = [2, -2, 0, 1]
    n_steps = 20
    show_coloring = False

    def construct(self):
        super().construct()

    def add_plane(self):
        super().add_plane()
        self.plane.axes.set_stroke(GREY_B, 1)

    def add_labels(self):
        super().add_labels()
        eq = self.corner_group[1]
        self.play(FlashAround(eq, run_time=3))

    def get_original_points(self):
        return [
            (r * np.cos(theta), r * np.sin(theta), 0)
            for r in np.linspace(0, 0.2, 10)
            for theta in np.linspace(0, TAU, int(50 * r)) + TAU * np.random.random()
        ]


class HighlightedJulia(IntroNewtonFractal):
    coefs = [-1.0, 0.0, 0.0, 1.0, 0.0, 1.0]

    def construct(self):
        # self.init_fractal(root_colors=ROOT_COLORS_DEEP[0::2])
        self.init_fractal(root_colors=ROOT_COLORS_DEEP)
        fractal = self.fractal

        def get_height_ratio():
            return self.camera.frame.get_height() / FRAME_HEIGHT

        fractal.set_colors(5 * [WHITE])
        fractal.add_updater(lambda m: m.set_julia_highlight(get_height_ratio() * 1e-3))
        fractal.set_n_steps(50)
        # self.play(
        #     fractal.animate.set_julia_highlight(1e-3),
        #     run_time=5
        # )

        # self.embed()


class MontelCorrolaryScreenGrab(Scene):
    def construct(self):
        pass


class MetaFractal(IntroNewtonFractal):
    fixed_roots = [-1, 1]
    z0 = complex(0.5, 0)
    n_steps = 200

    def construct(self):
        colors = ROOT_COLORS_DEEP[0::2]
        self.plane_config["faded_line_ratio"] = 3
        plane = self.get_plane()
        root_dots = self.root_dots = VGroup(*(
            Dot(plane.n2p(root), color=color)
            for root, color in zip(self.fixed_roots, colors)
        ))
        root_dots.set_stroke(BLACK, 3)
        fractal = MetaNewtonFractal(
            plane,
            fixed_roots=self.fixed_roots,
            colors=colors,
            n_steps=self.n_steps,
        )
        fractal.add_updater(lambda f: f.set_fixed_roots([
            plane.p2n(dot.get_center())
            for dot in root_dots
        ]))

        self.add(fractal, plane)
        self.add(root_dots)

        point1 = np.array([1.62070862, 1.68700851, 0.])
        point2 = np.array([0.81263967, 2.84042313, 0.])
        height1 = 0.083
        height2 = 0.035

        frame = self.camera.frame
        frame.generate_target()
        frame.target.move_to(point1)
        frame.target.set_height(height1)

        self.play(
            MoveToTarget(frame),
            run_time=10,
            rate_func=bezier([0, 0, 1, 1])
        )
        self.wait()
        self.play(
            UpdateFromAlphaFunc(
                frame,
                lambda m, a: m.set_height(
                    interpolate(
                        interpolate(height1, 2, a),
                        interpolate(2, height2, a),
                        a,
                    ),
                ).move_to(
                    interpolate(point1, point2, a)
                )
            ),
            run_time=10
        )


class Thumbnail2(SimpleFractalScene):
    def construct(self):
        super().construct()
        fractal = self.fractal
        fractal.set_saturation_factor(4.5)
        self.remove(self.plane)
        self.remove(self.root_dots)

        frame = self.camera.frame
        frame.set_height(4)

        fc = fractal.copy()
        fc.set_saturation_factor(2)
        fc.set_julia_highlight(0.01)
        self.add(fc)

        # self.clear()
        # back = fractal.copy()
        # back.set_saturation_factor(0)
        # back.set_opacity(0.1)
        # self.add(back)

        # N = 20
        # for x in np.linspace(np.log(1e-3), np.log(0.1), N):
        #     jh = np.exp(x)
        #     fc = fractal.copy()
        #     fc.set_saturation_factor(1)
        #     fc.set_julia_highlight(jh)
        #     fc.set_opacity(2 / N)
        #     self.add(fc)

        self.embed()
