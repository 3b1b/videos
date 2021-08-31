from manim_imports_ext import *

from _2021.quintic import coefficients_to_roots
from _2021.quintic import roots_to_coefficients
from _2021.quintic import dpoly
from _2021.quintic import poly


ROOT_COLORS_BRIGHT = [RED, GREEN, BLUE, YELLOW, MAROON_B]
ROOT_COLORS_DEEP = ["#440154", "#3b528b", "#21908c", "#5dc963", "#29abca"]


class PolyFractal(Mobject):
    CONFIG = {
        "shader_folder": "poly_fractal",
        "shader_dtype": [
            ('point', np.float32, (3,)),
        ],
        "colors": ROOT_COLORS_DEEP,
        "coefs": [1.0, -1.0, 1.0, 0.0, 0.0, 1.0],
        "scale_factor": 1.0,
        "offset": ORIGIN,
        "n_steps": 30,
    }

    def init_data(self):
        self.data = {
            "points": np.array([UL, DL, UR, DR]),
        }

    def init_uniforms(self):
        super().init_uniforms()
        self.set_colors(self.colors)
        self.set_coefs(self.coefs)
        self.set_scale(self.scale_factor)
        self.set_offset(self.offset)
        self.set_n_steps(self.n_steps)

    def set_colors(self, colors):
        self.uniforms.update({
            f"color{n}": np.array(color_to_rgba(color))
            for n, color in enumerate(colors)
        })
        return self

    def set_coefs(self, coefs, reset_roots=True):
        self.uniforms["n_roots"] = float(len(coefs))
        self.uniforms.update({
            f"coef{n}": np.array([coef.real, coef.imag])
            for n, coef in enumerate(map(complex, coefs))
        })
        if reset_roots:
            self.set_roots(coefficients_to_roots(coefs), False)
        return self

    def set_roots(self, roots, reset_coefs=True):
        self.uniforms["n_roots"] = float(len(roots))
        self.uniforms.update({
            f"root{n}": np.array([root.real, root.imag])
            for n, root in enumerate(map(complex, roots))
        })
        if reset_coefs:
            self.set_coefs(roots_to_coefficients(roots), False)
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


# Scenes

class HelloPatrons(Scene):
    def construct(self):
        self.add(FullScreenRectangle())
        morty = Mortimer()
        morty.to_edge(DOWN).shift(3 * RIGHT)
        self.play(PiCreatureSays(morty, "Hello patrons!", target_mode="wave_1"))
        self.play(Blink(morty))


class ComingVideoWrapper(VideoWrapper):
    animate_boundary = False
    title = "Upcoming: Unsolvabillity of the Quintic"


class RealNewtonsMethod(Scene):
    coefs = [-0.2, -1, 1, 0, 0, 1]
    poly_tex = "x^5 + x^2 - x - 0.2"
    dpoly_tex = "5x^4 + 2x - 1"
    seed = 1.3
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
            lambda x: poly(x, self.coefs)
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
        underline = Underline(self.poly[:-1])
        underline.match_style(self.graph)

        self.play(
            FlashAround(self.poly[:-1], color=self.graph_color),
            ShowCreation(self.graph),
            run_time=3
        )
        self.wait()

    def highlight_roots(self):
        roots = coefficients_to_roots(self.coefs)
        real_roots = [
            root.real for root in roots
            if abs(root.imag) < 1e-6
        ]
        real_roots.sort()

        dots = VGroup(*(
            Dot(self.axes.c2p(r, 0), radius=0.05)
            for r in real_roots
        ))
        dots.set_fill(YELLOW, 1)
        dots.set_stroke(BLACK, 2, background=True)
        squares = VGroup(*[
            Square().set_height(0.25).move_to(dot)
            for dot in dots
        ])
        squares.set_stroke(YELLOW, 3)
        squares.set_fill(opacity=0)

        self.play(
            LaggedStart(
                *[
                    FadeIn(dot, shift=DOWN, scale=0.25)
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

    def step_towards_root(self, fade_tan_with_vline=False):
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
            gv.add_background_rectangle()
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


class AssumingItsGood(TeacherStudentsScene):
    def construct(self):
        self.pi_creatures.refresh_triangulation()
        self.teacher_says(
            TexText("Assuming this\\\\approximation\\\\is decent...", font_size=42),
            bubble_kwargs={
                "height": 3, "width": 4,
            }
        )
        self.change_student_modes(
            "pondering", "pondering", "tease",
            look_at_arg=self.screen
        )
        self.pi_creatures.refresh_triangulation()
        self.wait(3)


class RealNewtonsMethodHigherGraph(RealNewtonsMethod):
    coefs = [1, -1, 1, 0, 0, 0.99]
    poly_tex = "x^5 + x^2 - x + 1"
    n_search_steps = 20

    def find_root(self, cycle_run_time=0.5):
        super().find_root(cycle_run_time)

    def step_towards_root(self, fade_tan_with_vline=True):
        super().step_towards_root(fade_tan_with_vline)


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
            Restore(self.poly),
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

        self.play(LaggedStartMap(DrawBorderThenFill, root_circles))
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

        len_history = max([len(dot.history) for dot in dots if hasattr(dot, "history")])
        for n in range(len_history):
            dots.generate_target()
            for dot, dot_target in zip(dots, dots.target):
                try:
                    dot_target.move_to(dot.history[len_history - n - 1])
                except Exception:
                    pass
            self.play(MoveToTarget(dots, run_time=0.5))


class ComplexNewtonsMethodManySeedsHigherRes(ComplexNewtonsMethodManySeeds):
    step = 0.05


class IntroPolyFractal(Scene):
    def construct(self):
        plane = self.get_plane()
        fractal = self.get_fractal(plane)
        root_dots = self.get_root_dots(plane, fractal)

        self.add(fractal)
        self.add(plane)
        self.add(root_dots)

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

    def get_plane(self):
        plane = ComplexPlane(
            x_range=(-4, 4),
            y_range=(-4, 4),
            height=16,
            width=16,
            background_line_style={
                "stroke_color": GREY_A,
                "stroke_width": 1.0,
            },
            axis_config={
                "stroke_width": 1.0,
            }
        )
        plane.add_coordinate_labels(font_size=24)
        self.plane = plane
        return plane

    def get_fractal(self, plane, colors=ROOT_COLORS_BRIGHT):
        fractal = PolyFractal(
            scale_factor=get_norm(plane.n2p(1) - plane.n2p(0)),
            offset=plane.n2p(0),
            colors=colors,
        )
        fractal.replace(plane, stretch=True)
        return fractal

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


class IncreasingStepsPolyFractal(IntroPolyFractal):
    play_mode = False

    def construct(self):
        plane = self.get_plane()
        fractal = self.get_fractal(plane, colors=ROOT_COLORS_DEEP)
        fractal.set_n_steps(0)
        root_dots = self.get_root_dots(plane, fractal)
        self.tie_fractal_to_root_dots(fractal)

        steps_label = VGroup(Integer(0, edge_to_fix=RIGHT), Text("Steps"))
        steps_label.arrange(RIGHT, aligned_edge=UP)
        steps_label.next_to(ORIGIN, UP).to_edge(LEFT)
        steps_label.set_stroke(BLACK, 5, background=True)

        self.add(fractal)
        self.add(plane)
        self.add(root_dots)
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
