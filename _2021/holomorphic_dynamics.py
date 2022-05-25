from manim_imports_ext import *
from _2022.newton_fractal import *


MANDELBROT_COLORS = [
    "#00065c",
    "#061e7e",
    "#0c37a0",
    "#205abc",
    "#4287d3",
    "#D9EDE4",
    "#F0F9E4",
    "#BA9F6A",
    "#573706",
]


def get_c_dot_label(dot, get_c, font_size=24, direction=UP):
    c_label = VGroup(
        Tex("c = ", font_size=font_size),
        DecimalNumber(get_c(), font_size=font_size, include_sign=True)
    ).arrange(RIGHT, buff=0.075)
    c_label[0].shift(0.02 * DOWN)
    c_label.set_color(YELLOW)
    c_label.set_stroke(BLACK, 5, background=True)
    c_label.add_updater(lambda m: m.next_to(dot, direction, SMALL_BUFF))
    c_label.add_updater(lambda m: m[1].set_value(get_c()))
    return c_label


def get_iteration_label(font_size=36):
    kw = {
        "tex_to_color_map": {
            "z_0": BLUE_C,
            "z_1": BLUE_D,
            "z_2": GREEN_D,
            "z_3": GREEN_E,
            "z_{n + 1}": GREEN_D,
            "z_n": BLUE,
            "\\longrightarrow": WHITE,
        },
        "font_size": font_size,
    }
    iterations = Tex(
        """
        z_0 \\longrightarrow
        z_1 \\longrightarrow
        z_2 \\longrightarrow
        z_3 \\longrightarrow
        \\cdots
        """,
        **kw
    )
    for part in iterations.get_parts_by_tex("\\longrightarrow"):
        f = Tex("f", **kw)
        f.scale(0.5)
        f.next_to(part, UP, buff=0)
        part.add(f)

    rule = Tex("z_{n + 1} &= f(z_n)", **kw)
    result = VGroup(rule, iterations)
    result.arrange(DOWN, buff=MED_LARGE_BUFF)
    return result


class MandelbrotFractal(NewtonFractal):
    CONFIG = {
        "shader_folder": "mandelbrot_fractal",
        "shader_dtype": [
            ('point', np.float32, (3,)),
        ],
        "scale_factor": 1.0,
        "offset": ORIGIN,
        "colors": MANDELBROT_COLORS,
        "n_colors": 9,
        "parameter": complex(0, 0),
        "n_steps": 300,
        "mandelbrot": True,
    }

    def init_uniforms(self):
        Mobject.init_uniforms(self)
        self.uniforms["mandelbrot"] = float(self.mandelbrot)
        self.set_parameter(self.parameter)
        self.set_opacity(self.opacity)
        self.set_scale(self.scale_factor)
        self.set_colors(self.colors)
        self.set_offset(self.offset)
        self.set_n_steps(self.n_steps)

    def set_parameter(self, c):
        self.uniforms["parameter"] = np.array([c.real, c.imag])
        return self

    def set_opacity(self, opacity):
        self.uniforms["opacity"] = opacity
        return self

    def set_colors(self, colors):
        for n in range(len(colors)):
            self.uniforms[f"color{n}"] = color_to_rgb(colors[n])
        return self


class JuliaFractal(MandelbrotFractal):
    CONFIG = {
        "n_steps": 100,
        "mandelbrot": False,
    }

    def set_c(self, c):
        self.set_parameter(c)


# Scenes


class MandelbrotSetPreview(Scene):
    def construct(self):
        plane = ComplexPlane(
            (-2, 1), (-2, 2),
            background_line_style={
                "stroke_color": GREY_B,
                "stroke_opacity": 0.5,
            }
        )
        plane.set_width(0.7 * FRAME_WIDTH)
        plane.axes.set_stroke(opacity=0.5)
        plane.add_coordinate_labels(font_size=18)

        mandelbrot = MandelbrotFractal(plane)
        mandelbrot.set_n_steps(0)

        self.add(mandelbrot, plane)
        self.play(
            mandelbrot.animate.set_n_steps(300),
            rate_func=lambda a: a**3,
            run_time=10,
        )
        self.wait()


class HolomorphicDynamics(Scene):
    def construct(self):
        self.show_goals()
        self.complex_functions()
        self.repeated_functions()
        self.example_fractals()

    def show_goals(self):
        background = FullScreenRectangle()
        self.add(background)

        title = self.title = Text("Holomorphic Dynamics", font_size=60)
        title.to_edge(UP)
        title.set_stroke(BLACK, 3, background=True)
        underline = Underline(title, buff=-0.05)
        underline.scale(1.2)
        underline.insert_n_curves(20)
        underline.set_stroke(YELLOW, [1, *5 * [3], 1])
        self.add(title)
        self.add(underline, title)
        self.play(ShowCreation(underline))
        self.wait()

        frames = Square().replicate(2)
        frames.set_height(5)
        frames.set_width(6, stretch=True)
        frames.set_stroke(WHITE, 2)
        frames.set_fill(BLACK, 1)
        frames.arrange(RIGHT, buff=1)
        frames.to_edge(DOWN)

        goals = VGroup(
            # TexText("Newton's fractal $\\leftrightarrow$ Mandelbrot"),
            # TexText("Tie up loose ends"),
            TexText("Goal 1: Other Mandelbrot occurrences"),
            TexText("Goal 2: Tie up loose ends"),
        )
        goals.set_width(frames[0].get_width())
        goals.set_fill(GREY_A)
        for goal, frame in zip(goals, frames):
            goal.next_to(frame, UP)
            goal.align_to(goals[0], UP)

        self.play(
            FadeIn(frames[0]),
            FadeIn(goals[0], 0.5 * UP),
        )
        self.wait()
        self.play(
            FadeIn(frames[1]),
            FadeIn(goals[1], 0.5 * UP),
        )
        self.wait()

        # Transition
        rect = SurroundingRectangle(title.get_part_by_text("Holomorphic"))
        rect.set_stroke(YELLOW, 2)
        self.play(
            ReplacementTransform(underline, rect),
            FadeOut(background),
            LaggedStartMap(FadeOut, VGroup(
                *goals, *frames,
            ))
        )

        self.title_rect = rect

    def complex_functions(self):
        kw = {
            "tex_to_color_map": {
                "\\mathds{C}": BLUE,
                "z": YELLOW,
            }
        }
        f_def = VGroup(
            Tex("f : \\mathds{C} \\rightarrow \\mathds{C}", **kw),
            Tex("f'(z) \\text{ exists}", **kw)
        )
        f_def.arrange(RIGHT, aligned_edge=DOWN, buff=LARGE_BUFF)
        f_def.next_to(self.title, DOWN, buff=MED_LARGE_BUFF)

        for part in f_def:
            self.play(Write(part, stroke_width=1))
            self.wait()

        # Examples
        examples = VGroup(
            Tex("f(z) = z^2 + 1", **kw),
            Tex("f(z) = e^z", **kw),
            Tex("f(z) = \\sin\\left(z\\right)", **kw),
            Tex("\\vdots")
        )
        examples.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        examples[-1].shift(0.25 * RIGHT)
        examples.set_width(2.5)
        examples.to_corner(UL)

        self.play(LaggedStartMap(
            FadeIn, examples,
            shift=0.25 * DOWN,
            run_time=3,
            lag_ratio=0.5
        ))

        # Transition
        rect = self.title_rect
        new_rect = SurroundingRectangle(self.title.get_part_by_text("Dynamics"))
        new_rect.match_style(rect)

        self.play(
            FadeOut(examples, lag_ratio=0.1),
            FadeOut(f_def, lag_ratio=0.1),
            Transform(rect, new_rect)
        )
        self.wait()

    def repeated_functions(self):
        words = TexText("For some function $f(z)$,")
        rule, iterations = get_iteration_label()
        group = VGroup(words, iterations, rule)
        group.arrange(DOWN, buff=MED_LARGE_BUFF)
        group.next_to(self.title, DOWN, LARGE_BUFF)
        group.to_edge(LEFT)

        self.play(
            FadeIn(words),
            FadeIn(iterations, lag_ratio=0.2, run_time=2)
        )
        self.play(FadeIn(rule, 0.5 * DOWN))
        self.wait()
        self.play(LaggedStartMap(FadeOut, VGroup(
            words, iterations, rule, self.title_rect
        )))

    def example_fractals(self):
        newton = Tex("z - {P(z) \\over P'(z)}")
        mandelbrot = Tex("z^2 + c")
        exponential = Tex("a^z")

        rhss = VGroup(newton, mandelbrot, exponential)
        f_eqs = VGroup()
        lhss = VGroup()
        for rhs in rhss:
            rhs.generate_target()
            lhs = Tex("f(z) = ")
            lhs.next_to(rhs, LEFT)
            f_eqs.add(VGroup(lhs, rhs))
            lhss.add(lhs)
        VGroup(exponential, mandelbrot).shift(0.05 * UP)
        f_eqs.arrange(RIGHT, buff=1.5)
        f_eqs.next_to(self.title, DOWN, MED_LARGE_BUFF)

        rects = Square().replicate(3)
        rects.arrange(RIGHT, buff=0.2 * rects.get_width())
        rects.set_width(FRAME_WIDTH - 1)
        rects.center().to_edge(DOWN, buff=LARGE_BUFF)
        rects.set_stroke(WHITE, 1)
        arrows = VGroup()
        for rect, f_eq in zip(rects, f_eqs):
            arrow = Vector(0.5 * DOWN)
            arrow.next_to(rect, UP)
            arrows.add(arrow)
            f_eq.next_to(arrow, UP)
        f_eqs[0].match_y(f_eqs[1])

        self.play(
            FadeOut(self.title, UP),
            LaggedStartMap(Write, rhss),
            LaggedStartMap(FadeIn, lhss),
            LaggedStartMap(FadeIn, rects),
            LaggedStartMap(ShowCreation, arrows),
        )
        self.wait()


class HolomorphicPreview(Scene):
    def construct(self):
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

        input_word.next_to(in_plane, UP)
        output_word.next_to(out_plane, UP)

        self.add(in_plane, out_plane, input_word, output_word)

        # Show tiny neighborhood
        tiny_plane = ComplexPlane(
            (-2, 2),
            (-2, 2),
            height=0.5,
            width=0.5,
            axis_config={
                "stroke_width": 1.0,
            },
            background_line_style={
                "stroke_width": 1.0,
            },
            faded_line_ratio=1,
        )
        tiny_plane.move_to(in_plane.c2p(1, 1))

        for plane in in_plane, out_plane:
            plane.generate_target()
            for mob in plane.target.family_members_with_points():
                mob.set_opacity(mob.get_opacity() * 0.25)

        self.play(
            ShowCreation(tiny_plane),
            MoveToTarget(in_plane),
        )
        self.wait()

        def f(z):
            w = z - complex(1, 1)
            return complex(-1, 0.5) + complex(-1, 1) * w + 0.2 * w**2

        tiny_plane.prepare_for_nonlinear_transform()
        tiny_plane_image = tiny_plane.copy()
        tiny_plane_image.apply_function(
            lambda p: out_plane.n2p(f(in_plane.p2n(p)))
        )

        arrow = Arrow(
            tiny_plane, tiny_plane_image,
            path_arc=-PI / 4,
            stroke_width=5,
        )
        f_label = Tex("f(z)")
        f_label.next_to(arrow, UP, SMALL_BUFF)

        words = Text("Looks roughly like\nscaling + rotating")
        words.set_width(2.5)
        words.move_to(VGroup(in_plane, out_plane))

        self.play(
            ShowCreation(arrow),
            FadeIn(f_label, 0.1 * UP),
        )
        self.play(
            TransformFromCopy(tiny_plane, tiny_plane_image),
            MoveToTarget(out_plane),
            FadeIn(words),
            run_time=2,
        )
        self.wait(2)


class AmbientRepetition(Scene):
    n_steps = 30
    # c = -0.6436875 + -0.441j
    c = -0.5436875 + -0.641j
    show_labels = True

    def construct(self):
        plane = ComplexPlane((-2, 2), (-2, 2))
        plane.set_height(6)
        plane.add_coordinate_labels(font_size=18)
        plane.to_corner(DR, buff=SMALL_BUFF)
        self.add(plane)

        font_size = 30

        z0 = complex(0, 0)
        dot = Dot(color=BLUE)
        dot.move_to(plane.n2p(z0))
        z_label = Tex("z", font_size=font_size)
        z_label.set_stroke(BLACK, 5, background=True)
        z_label.next_to(dot, UP, SMALL_BUFF)
        if not self.show_labels:
            z_label.set_opacity(0)
        self.add(dot, z_label)

        self.add(TracedPath(dot.get_center, stroke_width=1))

        func = self.func

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
            fz_label.next_to(dot_copy, normalize(new_point - dot.get_center()), buff=0)
            if not self.show_labels:
                fz_label.set_opacity(0)

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

    def func(self, z):
        return 2 * ((z / 2)**2 + self.c)


class AmbientRepetitionLimitPoint(AmbientRepetition):
    n_steps = 30
    c = complex()
    c = 0.234 + 0.222j
    show_labels = False


class AmbientRepetitionInfty(AmbientRepetition):
    c = -0.7995 + 0.3503j
    n_steps = 12


class AmbientRepetitionChaos(AmbientRepetition):
    def func(self, c):
        return complex(
            random.random() * 4 - 2,
            random.random() * 4 - 2,
        )


class Recap(VideoWrapper):
    title = "Newton's fractal quick recap"
    animate_boundary = False
    screen_height = 6.3


class RepeatedNewtonPlain(RepeatedNewton):
    n_steps = 20
    colors = ROOT_COLORS_DEEP


class RationalFunctions(Scene):
    def construct(self):
        # Show function
        equation = Tex(
            "f(z)",
            "=", "z - {P(z) \\over P'(z)}",
            "=", "z - {z^3 - 1 \\over 3z^2}",
            "=", "{2z^3 + 1 \\over 3z^2}"
        )
        iter_brace = Brace(equation[2], UP)
        iter_text = iter_brace.get_text("What's being iterated")
        VGroup(iter_brace, iter_text).set_color(BLUE_D)
        example_brace = Brace(equation[4], DOWN)
        example_text = example_brace.get_text("For example")
        VGroup(example_brace, example_text).set_color(TEAL_D)

        self.play(
            FadeIn(equation[:3]),
            GrowFromCenter(iter_brace),
            FadeIn(iter_text, 0.5 * UP)
        )
        self.wait()
        self.play(
            FadeIn(equation[3:5]),
            GrowFromCenter(example_brace),
            FadeIn(example_text, 0.5 * DOWN),
        )
        self.wait()
        self.play(
            Write(equation[5]),
            LaggedStart(*(
                FadeTransform(equation[4][i].copy(), equation[6][j])
                for i, j in zip(
                    [1, 0, *range(2, 10)],
                    [0, 0, *range(1, 9)],
                )
            ), lag_ratio=0.02)
        )
        self.add(equation[6])
        self.play(
            LaggedStart(*(
                FadeTransform(equation[4][i].copy().set_opacity(0), equation[6][j].copy().set_opacity(0))
                for i, j in zip(
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                )
            ), lag_ratio=0.02)
        )
        self.add(equation)
        self.wait()

        # Name rational function
        box = SurroundingRectangle(equation[6], buff=SMALL_BUFF)
        box.set_stroke(YELLOW, 2)
        rational_name = TexText("``Rational function''")
        rational_name.next_to(box, UP, buff=1.5)
        arrow = Arrow(rational_name, box)

        self.play(
            Write(rational_name),
            ShowCreation(arrow),
            ShowCreation(box),
        )
        self.wait()

        # Forget about the Newton's method origins
        self.play(
            equation[:2].animate.next_to(ORIGIN, LEFT, SMALL_BUFF),
            equation[2:6].animate.set_opacity(0.5).scale(0.5).to_corner(DR),
            VGroup(equation[6], box).animate.next_to(ORIGIN, RIGHT, SMALL_BUFF),
            MaintainPositionRelativeTo(
                VGroup(rational_name, arrow),
                box,
            ),
            FadeOut(
                VGroup(example_brace, example_text, iter_brace, iter_text),
                shift=2 * RIGHT + DOWN
            )
        )
        self.wait()

        # Other rational functions
        frame = self.camera.frame

        rhs = equation[6]
        functions = VGroup(
            Tex("3z^4 + z^3 + 4 \\over z^5 + 5z + 9"),
            Tex("2z^6 + 7z^4 + 1z \\over 8z^3 + 2z^2 + 8"),
            Tex("(z^2 + 1)^2 \\over 4z(z^2 - 1)"),
            Tex("az + b \\over cz + d"),
            Tex("z^2 + az + b \\over z^2 + cz + d"),
            Tex(
                "a_n z^n + \\cdots + a_0 \\over "
                "b_m z^m + \\cdots + b_0"
            ),
            Tex("z^2 + c \\over 1"),
        )

        for function in functions:
            function.replace(rhs, dim_to_match=1)
            function.move_to(rhs, LEFT)
            function.set_max_width(rhs.get_width() + 2)

        for n, function in enumerate(functions):
            self.play(
                FadeOut(rhs, lag_ratio=0.1),
                FadeIn(function, lag_ratio=0.1),
                box.animate.set_opacity(0),
            )
            self.remove(box)
            self.wait()
            if n == 0:
                self.play(
                    rational_name.animate.next_to(box, UP),
                    arrow.animate.scale(0, about_edge=DOWN),
                    ApplyMethod(frame.shift, 2 * DOWN, run_time=2),
                    FadeOut(equation[2:6]),
                )
            else:
                self.wait()
            rhs = function[0]

        self.play(
            FadeOut(box),
            FadeOut(rational_name),
            FadeOut(rhs[4:]),
            rhs[:4].animate.next_to(ORIGIN, RIGHT, SMALL_BUFF).shift(0.07 * UP),
            frame.animate.shift(DOWN),
        )
        self.wait()


class ShowFatouAndJulia(Scene):
    def construct(self):
        time_range = (1900, 2020)
        timeline = NumberLine(
            (*time_range, 1),
            tick_size=0.025,
            longer_tick_multiple=4,
            numbers_with_elongated_ticks=range(*time_range, 10),
        )
        timeline.stretch(0.25, 0)
        timeline.add_numbers(
            range(*time_range, 10),
            group_with_commas=False,
        )
        timeline.set_y(-3)
        timeline.to_edge(LEFT, buff=0)

        line = Line(timeline.n2p(1917), timeline.n2p(1920))
        line.scale(2)
        brace = Brace(line, UP)
        brace.stretch(1 / 2, 0)
        line.stretch(0.6, 0)
        line.insert_n_curves(20)
        line.set_stroke(BLUE, [1, *3 * [5], 1])

        kw = {"label_direction": UP, "height": 3}
        figures = Group(
            get_figure("Pierre_Fatou", "Pierre Fatou", "1878-1929", **kw),
            get_figure("Gaston_Julia", "Gaston Julia", "1893-1978", **kw),
        )
        figures.set_height(3)
        figures.arrange(RIGHT, buff=LARGE_BUFF)
        figures.next_to(brace, UP)

        self.add(timeline)
        self.add(figures)
        for figure in figures:
            self.remove(*figure[2:])

        frame = self.camera.frame
        frame.save_state()
        frame.align_to(timeline, RIGHT)
        self.play(Restore(frame, run_time=3))
        self.play(LaggedStart(*(
            Write(VGroup(*figure[2:]))
            for figure in figures
        ), lag_ratio=0.7))
        self.play(
            GrowFromCenter(brace),
            ShowCreation(line),
        )
        self.wait()

        # Names
        names = VGroup(*(figure[2][-5:] for figure in figures))
        rects = VGroup(*(SurroundingRectangle(name, buff=0.05) for name in names))
        rects.set_stroke(BLUE, 2)

        self.play(LaggedStartMap(ShowCreation, rects))


class IveSeenThis(TeacherStudentsScene):
    def construct(self):
        equation = Tex("f(z) = z^2 + c")
        equation.to_edge(UP)
        self.add(equation)

        mandelbrot_outline = ImageMobject("Mandelbrot_boundary")
        mandelbrot_outline.set_height(3)
        mandelbrot_outline.move_to(self.students[2].get_corner(UR))
        mandelbrot_outline.shift(1.2 * UP)

        self.student_says(
            "I've seen this one",
            target_mode="surprised",
            look_at=equation,
            added_anims=[
                self.students[0].change("tease", equation),
                self.students[1].change("happy", equation),
                self.teacher.change("happy", equation),
            ]
        )
        self.play(self.teacher.change("tease"))
        self.wait(2)
        self.add(mandelbrot_outline, *self.mobjects)
        self.play(
            FadeOut(self.background),
            RemovePiCreatureBubble(
                self.students[2],
                target_mode="raise_right_hand",
                look_at=mandelbrot_outline,
            ),
            FadeIn(mandelbrot_outline, 0.5 * UP, scale=2),
            self.students[0].change("pondering", mandelbrot_outline),
            self.students[1].change("thinking", mandelbrot_outline),
            self.teacher.change("happy", self.students[2].eyes),
        )
        self.wait(4)

        self.embed()


class MandelbrotIntro(Scene):
    n_iterations = 30

    def construct(self):
        self.add_process_description()
        self.add_plane()
        self.show_iterations()
        self.add_mandelbrot_image()

    def add_process_description(self):
        kw = {
            "tex_to_color_map": {
                "{c}": YELLOW,
            }
        }
        terms = self.terms = VGroup(
            Tex("z_{n + 1} = z_n^2 + {c}", **kw),
            Tex("{c} \\text{ can be changed}", **kw),
            Tex("z_0 = 0", **kw),
        )
        terms.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        terms.to_corner(UL)

        equation = Tex("f(z) = z^2 + c")
        equation.to_edge(UP)

        self.process_terms = terms
        self.add(equation)
        self.wait()
        self.play(FadeTransform(equation, terms[0]))

    def add_plane(self):
        plane = self.plane = ComplexPlane((-2, 1), (-2, 2))
        plane.set_height(4)
        plane.set_height(1.5 * FRAME_HEIGHT)
        plane.next_to(2 * LEFT, RIGHT, buff=0)
        plane.add_coordinate_labels(font_size=24)
        self.add(plane)

    def show_iterations(self):
        plane = self.plane

        # c0 = complex(-0.2, 0.95)
        c0 = complex(-0.6, 0.4)

        c_dot = self.c_dot = Dot()
        c_dot.set_fill(YELLOW)
        c_dot.set_stroke(BLACK, 5, background=True)
        c_dot.move_to(plane.n2p(c0))
        c_dot.add_updater(lambda m: m)  # Null

        n_iter_tracker = ValueTracker(1)

        def get_n_iters():
            return int(n_iter_tracker.get_value())

        def get_c():
            return plane.p2n(c_dot.get_center())

        def update_lines(lines):
            z1 = 0
            c = get_c()
            new_lines = []

            for n in range(get_n_iters()):
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

        c_label = get_c_dot_label(c_dot, get_c)

        lines = VGroup()
        lines.set_stroke(background=True)
        lines.add_updater(update_lines)
        self.add(lines, c_dot, c_label)

        def increase_step(run_time=1.0):
            n_iter_tracker.increment_value(1)
            lines.update()
            lines.suspend_updating()
            self.add(*lines, c_dot, c_label)
            self.play(
                ShowCreation(lines[-2]),
                TransformFromCopy(lines[-3], lines[-1]),
                run_time=run_time
            )
            self.add(lines, c_dot, c_label)
            lines.resume_updating()

        kw = {
            "tex_to_color_map": {
                "c": YELLOW,
            }
        }
        new_lines = VGroup(
            Tex("z_1 = 0^2 + c = c", **kw),
            Tex("z_2 = c^2 + c", **kw),
            Tex("z_3 = (c^2 + c)^2 + c", **kw),
            Tex("z_4 = ((c^2 + c)^2 + c)^2 + c", **kw),
            Tex("\\vdots", **kw),
        )
        new_lines.arrange(DOWN, aligned_edge=LEFT)
        new_lines[-2].scale(0.8, about_edge=LEFT)
        new_lines.next_to(self.process_terms[2], DOWN, aligned_edge=LEFT)
        new_lines[-1].match_x(new_lines[-2][0][2])

        # Show c
        self.wait()
        self.play(Write(self.process_terms[1]))
        self.wait(10)

        # Show first step
        dot = Dot(plane.n2p(0))
        self.play(FadeIn(self.process_terms[2], 0.5 * DOWN))
        self.play(FadeIn(dot, scale=0.2, run_time=2))
        self.play(FadeOut(dot))

        self.play(FadeIn(new_lines[0], 0.5 * DOWN))
        self.play(ShowCreationThenFadeOut(
            lines[0].copy().set_stroke(BLUE, 5)
        ))
        self.wait(3)

        # Show second step
        self.play(FadeIn(new_lines[1], 0.5 * DOWN))
        increase_step()
        self.wait(10)

        # Show 3rd to nth steps
        self.play(FadeIn(new_lines[2], 0.5 * DOWN))
        increase_step()
        self.play(FadeIn(new_lines[3], 0.5 * DOWN))
        increase_step()
        self.wait(5)
        self.play(FadeIn(new_lines[4]))
        for n in range(self.n_iterations):
            increase_step(run_time=0.25)

        # Play around
        self.wait(15)

    def add_mandelbrot_image(self):
        mandelbrot_set = MandelbrotFractal(self.plane)

        self.add(mandelbrot_set, *self.mobjects)
        self.play(
            FadeIn(mandelbrot_set, run_time=2),
            # self.plane.animate.set_opacity(0.5)
            self.plane.animate.set_stroke(WHITE, opacity=0.25)
        )

    # Listeners
    def on_mouse_motion(self, point, d_point):
        super().on_mouse_motion(point, d_point)
        if self.window.is_key_pressed(ord(" ")):
            self.c_dot.move_to(point)


class BenSparksVideoWrapper(VideoWrapper):
    title = "Heavily inspired from Ben Sparks"


class AckoNet(VideoWrapper):
    title = "Acko.net, How to Fold a Julia Fractal"


class ParameterSpaceVsSeedSpace(Scene):
    def construct(self):
        boxes = Square().get_grid(2, 2, buff=0)
        boxes.set_stroke(WHITE, 2)
        boxes.set_height(6)
        boxes.set_width(10, stretch=True)
        boxes.to_corner(DR)
        self.add(boxes)

        f_labels = VGroup(
            Tex("f(z) = z^2 + c"),
            Tex("f(z) = z - {P(z) \\over P'(z)}"),
        )
        kw = {
            "tex_to_color_map": {
                "function": YELLOW,
                "seed": BLUE_D,
            }
        }
        top_labels = VGroup(
            TexText("One seed\\\\Pixel $\\leftrightarrow$ function", **kw),
            TexText("One function\\\\Pixel $\\leftrightarrow$ seed", **kw),
        )
        for f_label, box in zip(f_labels, boxes[::2]):
            f_label.set_max_width(3.25)
            f_label.next_to(box, LEFT)
        for top_label, box in zip(top_labels, boxes):
            top_label.next_to(box, UP, buff=0.2)

        for f_label in f_labels:
            self.play(FadeIn(f_label, 0.25 * LEFT))
        self.wait()
        self.play(FadeIn(top_labels[0]))
        self.wait()
        self.play(TransformMatchingTex(
            top_labels[0].copy(), top_labels[1]
        ))
        self.wait()


class MandelbrotStill(Scene):
    def construct(self):
        plane = ComplexPlane((-3, 2), (-1.3, 1.3))
        plane.set_height(FRAME_HEIGHT)
        fractal = MandelbrotFractal(plane)
        self.add(fractal)


class JuliaStill(Scene):
    def construct(self):
        plane = ComplexPlane((-4, 4), (-1.5, 1.5))
        plane.set_height(FRAME_HEIGHT)
        fractal = JuliaFractal(plane)
        fractal.set_c(-0.03 + 0.74j)
        fractal.set_n_steps(100)
        self.add(fractal)


class ClassicJuliaSetDemo(MandelbrotIntro):
    def construct(self):
        # Init planes
        kw = {
            "background_line_style": {
                "stroke_width": 0.5,
            }
        }
        planes = VGroup(
            ComplexPlane((-2, 1), (-1.6, 1.6), **kw),
            ComplexPlane((-2, 2), (-2, 2), **kw),
        )
        for plane, corner in zip(planes, [DL, DR]):
            plane.set_stroke(WHITE, opacity=0.5)
            plane.set_height(6)
            plane.to_corner(corner, buff=MED_SMALL_BUFF)
            plane.to_edge(DOWN, SMALL_BUFF)

        planes[1].add_coordinate_labels(font_size=18)
        planes[0].add_coordinate_labels(
            (-1, 0, 1, 1j, -1j),
            font_size=18
        )

        # Init fractals
        mandelbrot = MandelbrotFractal(planes[0])
        julia = JuliaFractal(planes[1])
        fractals = Group(mandelbrot, julia)

        self.add(*fractals, *planes)

        # Add c_dot
        c_dot = self.c_dot = Dot(radius=0.05)
        c_dot.set_fill(YELLOW, 1)
        c_dot.move_to(planes[0].c2p(-0.5, 0.5))
        c_dot.add_updater(lambda m: m)

        def get_c():
            return planes[0].p2n(c_dot.get_center())

        c_label = get_c_dot_label(c_dot, get_c, direction=UR)
        julia.add_updater(lambda m: m.set_c(get_c()))

        self.add(c_dot, c_label)

        # Add labels
        kw = {
            "tex_to_color_map": {
                "{z_0}": GREY_A,
                "{c}": YELLOW,
                "\\text{Pixel}": BLUE_D,
            },
        }

        title = TexText("Iterate\\\\$z^2 + c$")
        title.move_to(Line(*planes))
        title[0][-1].set_color(YELLOW)
        self.add(title)

        labels = VGroup(
            VGroup(
                Tex("{c} \\leftrightarrow \\text{Pixel}", **kw),
                Tex("z_0 = 0", **kw),
            ),
            VGroup(
                Tex("{c} = \\text{const.}", **kw),
                Tex("z_0 \\leftrightarrow \\text{Pixel}", **kw),
            )
        )

        for label, plane in zip(labels, planes):
            label.arrange(DOWN, aligned_edge=LEFT)
            label.next_to(plane, UP)

        space_labels = VGroup(
            Tex("{c}\\text{-space}", **kw),
            Tex("{z_0}\\text{-space}", **kw),
        )
        for label, plane in zip(space_labels, planes):
            label.scale(0.5)
            label.move_to(plane, UL).shift(SMALL_BUFF * DR)
        self.add(space_labels)

        # Animations
        self.add(labels[0])
        self.wait(2)
        self.play(
            TransformFromCopy(
                labels[0][0][0],
                labels[1][0][0],
            ),
            FadeIn(labels[1][0][1]),
        )
        self.wait(2)
        self.play(
            TransformFromCopy(
                labels[0][1].get_part_by_tex("z_0"),
                labels[1][1].get_part_by_tex("z_0"),
            ),
            FadeTransform(
                labels[0][0][1:].copy(),
                labels[1][1][1:],
            ),
        )


class AskAboutGeneralTheory(TeacherStudentsScene):
    def construct(self):
        self.teacher_says(
            TexText("Think about constructing\\\\a general theory"),
            added_anims=[self.change_students(
                "pondering", "thinking", "pondering",
                look_at=UP,
            )]
        )
        self.wait(3)
        self.teacher_says(
            TexText("What questions would\\\\you ask?"),
            target_mode="tease",
        )
        self.play_student_changes("thinking", "pondering")
        self.wait(3)

        self.embed()


class NewtonRuleLabel(Scene):
    def construct(self):
        rule = get_newton_rule()
        rule.scale(1.5)
        rule.set_stroke(BLACK, 5, background=True)
        rule.to_corner(UL)

        box = SurroundingRectangle(rule, buff=0.2)
        box.set_fill(BLACK, 0.9)
        box.set_stroke(WHITE, 1)
        VGroup(box, rule).to_corner(UL, buff=0)

        self.add(box, rule)


class FixedPoints(Scene):
    def construct(self):
        # Set the stage
        iter_label = self.add_labels()
        rule, iterations = iter_label
        plane = self.add_plane()
        z_dot, z_label = self.add_z_dot()

        # Ask question
        question = TexText("When does $z$ stay fixed in place?")
        question.next_to(plane, RIGHT, MED_LARGE_BUFF, aligned_edge=UP)

        arrow = self.get_arrow_loop(z_dot)

        # f(z) = z
        t2c = {"z": BLUE}
        kw = {
            "tex_to_color_map": t2c,
            "isolate": ["=", "\\Rightarrow", "A(", "B(", ")"],
        }
        equation = Tex("f(z) = z", **kw)
        equation.next_to(question, DOWN, MED_LARGE_BUFF)

        newton_example = Tex(
            "z - {P(z) \\over P'(z)} = z",
            "\\quad \\Leftrightarrow \\quad ",
            "P(z) = 0",
            **kw,
        )
        newton_example.next_to(equation, DOWN, buff=LARGE_BUFF)
        mandelbrot_example = Tex(
            "\\text{Exercise 1a: Find the fixed points}\\\\",
            "\\text{of }", "f(z) = z^2 + c",
            alignment="\\centering",
            **kw
        )
        mandelbrot_example[1:].match_x(mandelbrot_example[0])
        mandelbrot_example.move_to(newton_example)

        fixed_point = Text("Fixed point")
        fixed_point.next_to(equation, DOWN, LARGE_BUFF)
        fixed_point.to_edge(RIGHT)
        fp_arrow = Arrow(
            fixed_point.get_left(),
            equation[1].get_bottom(),
            path_arc=-PI / 4,
        )
        fp_group = VGroup(fixed_point, fp_arrow)
        fp_group.set_color(YELLOW)

        self.play(FadeIn(equation, DOWN))
        self.wait(2)
        self.play(
            FadeIn(fixed_point),
            ShowCreation(fp_arrow),
        )
        self.wait()
        self.play(
            FadeIn(newton_example, shift=0.5 * DOWN),
            FadeOut(fp_group),
        )
        self.wait(2)
        self.play(
            FadeOut(newton_example, 0.5 * DOWN),
            FadeIn(mandelbrot_example, 0.5 * DOWN),
        )
        self.wait(2)

        # Rational function
        question_group = VGroup(question, equation)
        question_group.generate_target()
        iterations.generate_target()
        VGroup(question_group.target, iterations.target).to_edge(UP)

        rational_parts = VGroup(
            Tex("{A(z) \\over B(z)} = z", **kw),
            Tex("A(z) = z \\cdot B(z)", **kw),
            Tex("A(z) - z \\cdot B(z) = 0", **kw),
        )
        rational_parts.arrange(DOWN, buff=MED_LARGE_BUFF)
        for part, tex in zip(rational_parts[1:], ("=", "-")):
            curr_x = part.get_part_by_tex(tex).get_x()
            target_x = rational_parts[0].get_part_by_tex("=").get_x()
            part.shift((target_x - curr_x) * RIGHT)
        rational_parts.next_to(question_group.target, DOWN, LARGE_BUFF)

        self.play(
            FadeOut(rule, UP),
            MoveToTarget(question_group),
            MoveToTarget(iterations),
            FadeOut(mandelbrot_example),
            FadeIn(rational_parts[0])
        )
        self.wait()
        for p1, p2 in zip(rational_parts, rational_parts[1:]):
            self.play(
                TransformMatchingTex(
                    p1.copy(), p2,
                    path_arc=PI / 2,
                    run_time=2,
                    fade_transform_mismatches=True,
                )
            )
            self.wait(2)

        rect = SurroundingRectangle(rational_parts[-1])
        solution_words = Text("Must have\nsolutions!", font_size=36)
        solution_words.set_color(YELLOW)
        solution_words.next_to(rect, RIGHT)
        solution_words.shift_onto_screen()

        self.play(
            ShowCreation(rect),
            Write(solution_words, run_time=1),
        )
        self.wait()

        example_roots = [
            -1.5 + 0.5j, -1.5 - 0.5j,
            -1.0 + 1.2j, -1.0 - 1.2j,
            1.0 + 1.0j, 1.0 - 1.0j,
            0.5, 1.7,
        ]
        glow_dots = VGroup(*(
            glow_dot(plane.n2p(root))
            for root in example_roots
        ))

        self.play(LaggedStartMap(
            FadeIn, glow_dots,
            scale=0.5,
            lag_ratio=0.2,
        ))
        self.wait()

        # Ask about stability
        def get_arrows(point, inward=False):
            arrows = VGroup(*(
                Arrow(ORIGIN, vect, buff=0.3)
                for vect in compass_directions(8)
            ))
            arrows.set_height(1)
            arrows.move_to(point)
            if inward:
                for arrow in arrows:
                    arrow.rotate(PI)
            return arrows

        outward_arrows = get_arrows(glow_dots[2])
        inward_arrows = get_arrows(glow_dots[4], inward=True)
        arrow_groups = VGroup(inward_arrows, outward_arrows)

        stability_words = VGroup(
            Text("Attracting"),
            Text("Repelling"),
        )
        for words, arrows in zip(stability_words, arrow_groups):
            words.scale(0.7)
            words.next_to(arrows, UP, SMALL_BUFF)
            words.set_color(GREY_A)
            words.set_stroke(BLACK, 5, background=True)

        stability_question = Text(
            "When are fixed points stable?"
        )
        stability_question.move_to(question)
        stable_underline = Underline(
            stability_question.get_part_by_text("stable")
        )
        stable_underline.insert_n_curves(20)
        stable_underline.scale(1.2)
        stable_underline.set_stroke(MAROON_B, [1, *4 * [4], 1])

        self.play(
            FadeOut(question, RIGHT),
            FadeIn(stability_question, RIGHT),
            FadeOut(arrow),
            FadeOut(z_dot),
            FadeOut(z_label),
        )
        self.play(ShowCreation(stable_underline))
        self.wait()

        for words, arrows in zip(stability_words, arrow_groups):
            self.play(
                FadeIn(words),
                ShowCreation(arrows, lag_ratio=0.2, run_time=3)
            )
            self.wait()

        # Show derivative condition
        morty = Mortimer(height=2)
        morty.to_corner(DR)

        deriv_ineq = Tex("|f'(z)| < 1", **kw)
        deriv_ineq.next_to(equation, DOWN, MED_LARGE_BUFF)

        equation.generate_target()
        group = VGroup(equation.target, deriv_ineq)
        group.arrange(RIGHT, buff=LARGE_BUFF)
        group.move_to(equation)

        attracting_condition = deriv_ineq.copy()
        repelling_condition = Tex("|f'(z)| > 1", **kw)
        conditions = VGroup(attracting_condition, repelling_condition)
        for condition, words in zip(conditions, stability_words):
            condition.scale(0.7)
            condition.set_stroke(BLACK, 5, background=True)
            condition.move_to(words, DOWN)
            if conditions is conditions[0]:
                condition.shift(SMALL_BUFF * UP)
            words.generate_target()
            words.target.next_to(condition, UP, buff=MED_SMALL_BUFF)

        self.play(
            LaggedStartMap(FadeOut, VGroup(
                *rational_parts, rect, solution_words,
            )),
            VFadeIn(morty),
            morty.change("tease"),
        )
        self.play(PiCreatureSays(
            morty, "Use derivatives!",
            target_mode="hooray",
            bubble_config={
                "height": 2,
                "width": 4,
            }
        ))
        self.play(Blink(morty))
        self.wait()
        self.play(RemovePiCreatureBubble(morty, target_mode="happy"))
        for condition, words in zip(conditions, stability_words):
            self.play(
                Write(condition),
                MoveToTarget(words),
            )
            self.wait()
        self.play(
            FadeInFromPoint(deriv_ineq, morty.get_corner(UL)),
            MoveToTarget(equation),
            morty.change("raise_right_hand")
        )
        self.play(Blink(morty))

        # Newton derivative examples
        newton_case = VGroup(
            Tex("f(z) = z - {P(z) \\over P'(z)}", **kw),
            Tex("f'(z) = {P(z)P''(z) \\over P'(z)^2}", **kw),
            Tex("P(z) = 0 \\quad \\Rightarrow \\quad f'(z) = 0", **kw),
        )
        newton_case.arrange(DOWN, aligned_edge=LEFT, buff=MED_LARGE_BUFF)
        newton_case.scale(0.8)
        newton_case.next_to(equation, DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)

        alt_line1 = Tex("f'(z) = 1 - {P'(z)P'(z) - P(z)P''(z) \\over P'(z)^2}", **kw)
        alt_line1.match_height(newton_case[1])
        alt_line1.move_to(newton_case[1], LEFT)

        self.play(
            TransformMatchingTex(equation.copy()[:3], newton_case[0]),
            morty.change("pondering", newton_case[0]),
        )
        self.wait()
        self.play(
            TransformMatchingTex(
                deriv_ineq[:-1].copy(),
                alt_line1[:4],
            )
        )
        self.wait()
        self.play(
            FadeIn(alt_line1[4:], lag_ratio=0.1, run_time=1.5),
            morty.animate.scale(0.8, about_edge=DR).change("sassy", alt_line1),
        )
        self.play(Blink(morty))
        self.wait()
        self.play(
            TransformMatchingTex(alt_line1[4:], newton_case[1][4:]),
            morty.change("tease", alt_line1),
        )
        self.remove(alt_line1)
        self.add(newton_case[1])
        self.wait()
        self.play(Blink(morty))
        self.wait()
        self.play(
            morty.change("pondering", newton_case[2]),
            FadeIn(newton_case[2])
        )
        self.play(Blink(morty))
        self.wait()

        # Show super-attraction
        super_arrows = VGroup(*(
            get_arrows(dot, inward=True)
            for dot in glow_dots
        ))
        attraction_anims = []
        for cluster in super_arrows:
            for arrow in cluster:
                new_arrow = arrow.copy()
                new_arrow.scale(1.5)
                new_arrow.set_stroke(YELLOW, 8)
                attraction_anims.append(
                    ShowCreationThenFadeOut(new_arrow)
                )

        rect = SurroundingRectangle(newton_case[2][6:])
        super_words = TexText("``Superattracting''", font_size=36)
        super_words.set_color(YELLOW)
        super_words.next_to(rect, DOWN, SMALL_BUFF)

        self.play(
            morty.change("thinking"),
            ShowCreation(rect),
            FadeIn(super_words)
        )
        self.play(
            LaggedStartMap(FadeOut, VGroup(
                *conditions, *stability_words,
                outward_arrows, inward_arrows,
            )),
            LaggedStartMap(FadeIn, super_arrows, scale=0.5)
        )
        self.play(
            LaggedStart(*attraction_anims, lag_ratio=0.02),
            run_time=3
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeOut, VGroup(
                *super_arrows, *newton_case,
                rect, super_words, morty,
                glow_dots,
            ))
        )

        # Mandelbrot exercise
        part1 = mandelbrot_example
        part1.set_height(0.9)
        part2 = TexText(
            "Exercise 1b: Determine when at least\\\\"
            "one fixed point is attracting.",
        )
        part3 = TexText(
            "Exercise 1c$^{**}$: Show that the set of values\\\\",
            "$c$ satisfying this form a cardioid.",
            tex_to_color_map={"$c$": YELLOW}
        )
        parts = VGroup(part1, part2, part3)
        for part in part2, part3:
            part.scale(part1[0][0].get_height() / part[0][0].get_height())
        parts.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        parts.next_to(
            VGroup(equation, deriv_ineq),
            DOWN, buff=LARGE_BUFF,
        )

        mandelbrot = MandelbrotFractal(plane)

        R = 0.25
        cardioid = ParametricCurve(
            lambda t: plane.c2p(
                2 * R * math.cos(t) - R * math.cos(2 * t),
                2 * R * math.sin(t) - R * math.sin(2 * t),
            ),
            t_range=(0, TAU)
        )
        cardioid.set_stroke(YELLOW, 4)

        self.add(mandelbrot, plane)
        plane.generate_target(use_deepcopy=True)
        plane.target.set_stroke(WHITE, opacity=0.5)
        self.play(
            MoveToTarget(plane),
            FadeIn(mandelbrot),
            FadeIn(part1),
        )
        self.wait()
        self.play(Write(part2))
        self.wait()
        print(self.num_plays)
        self.play(Write(part3))
        self.play(ShowCreation(cardioid, run_time=4, rate_func=linear))
        self.play(
            cardioid.animate.set_fill(YELLOW, 0.25),
            run_time=2
        )
        self.wait()

    def add_labels(self):
        iter_label = get_iteration_label(48)
        iter_label.to_edge(UP)
        self.add(iter_label)
        return iter_label

    def add_plane(self):
        plane = self.plane = ComplexPlane(
            (-2, 2), (-2, 2),
            background_line_style={
                "stroke_width": 1,
            }
        )
        plane.set_height(5)
        plane.to_corner(DL)
        plane.add_coordinate_labels(font_size=16)
        self.add(plane)
        return plane

    def add_z_dot(self, z=1 + 1j, z_tex="z"):
        z_dot = Dot(radius=0.05, color=BLUE)
        z_dot.move_to(self.plane.n2p(z))
        z_label = Tex(z_tex, font_size=30)
        z_label.next_to(z_dot, UL, buff=0)
        self.add(z_dot, z_label)
        return z_dot, z_label

    def get_arrow_loop(self, dot):
        arrow = Line(
            dot.get_bottom(),
            dot.get_right(),
            path_arc=330 * DEGREES,
            buff=0.05,
        )
        arrow.add_tip(width=0.15, length=0.15)
        arrow.set_color(GREY_A)
        return arrow


class UseNewton(TeacherStudentsScene):
    def construct(self):
        self.teacher_says(
            TexText(
                "You could solve\\\\ $A(z) - z\\cdot B(z) = 0$ \\\\",
                "using Newton's method"
            ),
            bubble_config={
                "width": 4,
                "height": 3,
            },
            target_mode="hooray",
            added_anims=[
                self.change_students("confused", "erm", "maybe")
            ]
        )
        self.wait(2)
        self.student_says(
            Text("Too meta..."),
            target_mode="sassy",
            index=2,
        )
        self.wait(3)


class DescribeDerivative(Scene):
    zoom_in_frame = False
    z = 1
    z_str = "{1}"
    fz_str = "1"
    fpz_str = "2"

    def construct(self):
        # Add plane
        plane = ComplexPlane((-4, 4), (-2, 2))
        plane.set_height(FRAME_HEIGHT)
        plane.add_coordinate_labels(font_size=18)
        self.add(plane)

        # Add function labels
        z_str = self.z_str
        fz_str = self.fz_str
        fpz_str = self.fpz_str
        kw = {
            "tex_to_color_map": {
                "z": GREY_A,
                z_str: YELLOW,
            },
            "isolate": ["f", "(", ")", "=", z_str, fz_str, fpz_str],
        }
        f_label = Tex("f(z) = z^2", **kw)
        df_label = Tex("f'(z) = 2z", **kw)
        labels = VGroup(f_label, df_label)
        labels.arrange(DOWN)
        labels.to_corner(UL)
        labels.set_stroke(BLACK, 5, background=True)
        corner_rect = SurroundingRectangle(labels, buff=0.25)
        corner_rect.set_stroke(WHITE, 2)
        corner_rect.set_fill(BLACK, 0.9)
        corner_group = VGroup(corner_rect, *labels)
        corner_group.to_corner(UL, buff=0)

        df_question = Text("Derivative?")
        df_question.set_stroke(BLACK, 5, background=True)
        df_arrow = Vector(LEFT)
        df_arrow.next_to(f_label, RIGHT, SMALL_BUFF)
        df_question.next_to(df_arrow, RIGHT, SMALL_BUFF)

        self.add(corner_rect, f_label)
        self.play(
            FadeIn(df_question, 0.2 * RIGHT),
            ShowCreation(df_arrow)
        )
        self.wait()
        self.play(
            FadeIn(df_label, shift=0.5 * DOWN),
            FadeOut(df_question),
            Uncreate(df_arrow),
        )

        # Add dots
        density = 10
        dot_radius = 0.025
        dots = DotCloud([
            plane.c2p(x, y)
            for x in np.arange(-3.7, 3.7, 1.0 / density)
            for y in np.arange(-2.0, 2.1, 1.0 / density)
        ])
        dots.set_radius(dot_radius)
        dots.set_color(GREY_B)
        dots.set_gloss(0.2)
        dots.set_opacity(0.5)
        dots.add_updater(lambda m: m)

        epsilon = 5e-3
        tiny_dots = DotCloud([
            plane.n2p(self.z + epsilon * complex(x, y))
            for x in np.arange(-20, 20)
            for y in np.arange(-10, 10)
        ])
        tiny_dots.set_radius(dot_radius * 15 * epsilon)
        tiny_dots.set_opacity(0.75)
        tiny_dots.set_color_by_gradient(YELLOW, BLUE)
        tiny_dots.set_gloss(0.2)

        self.add(dots, corner_group)
        self.play(ShowCreation(dots), run_time=2)
        self.wait()

        # Show function evaluation
        ex_labels = VGroup(
            Tex(f"f({z_str}) = {fz_str}", **kw),
            Tex(f"f'({z_str}) = {fpz_str}", **kw),
        )
        for ex_label, gen_label in zip(ex_labels, labels):
            ex_label.next_to(gen_label, RIGHT, MED_LARGE_BUFF)

        ex_labels[0].align_to(ex_labels[1], LEFT)

        corner_rect.generate_target()
        corner_rect.target.set_width(
            VGroup(ex_labels, labels).get_width() + 0.5,
            about_edge=LEFT,
            stretch=True,
        )

        self.add(*corner_group)
        self.play(
            MoveToTarget(corner_rect),
            FadeIn(ex_labels),
        )
        self.wait()

        # Apply function
        fade_anims = []
        if self.zoom_in_frame:
            rect = self.camera.frame
            plane.generate_target()
            fade_anims = [
                ApplyMethod(dots.set_opacity, 0, rate_func=squish_rate_func(smooth, 0.5, 1.0)),
                plane.animate.set_stroke(width=0.25),
            ]
        else:
            rect = ScreenRectangle()
            rect.set_height(FRAME_HEIGHT)
            rect.set_stroke(WHITE, 2)

        self.play(
            rect.animate.replace(tiny_dots, 1).move_to(plane.n2p(self.z)),
            *fade_anims,
            FadeIn(tiny_dots),
            run_time=3,
        )
        self.wait()

        # def func(p):
        #     z = plane.p2n(p)
        #     return plane.n2p(z**2)

        def homotopy(x, y, z, t):
            z = plane.p2n([x, y, z])
            return plane.n2p(z**(1 + t))

        rc = rect.get_center()
        path = ParametricCurve(lambda t: homotopy(*rc, t))

        self.play(
            Homotopy(homotopy, dots),
            Homotopy(homotopy, tiny_dots),
            MoveAlongPath(rect, path),
            # dots.animate.apply_function(func),
            # tiny_dots.animate.apply_function(func),
            # rect.animate.move_to(func(rect.get_center())),
            run_time=5,
        )
        self.wait()


class DescribeDerivativeInnerFrame(DescribeDerivative):
    zoom_in_frame = True


class DescribeDerivativeIExample(DescribeDerivative):
    z = 1j
    z_str = "{i}"
    fz_str = "-1"
    fpz_str = "2i"


class DescribeDerivativeIExampleInnerFrame(DescribeDerivativeIExample):
    zoom_in_frame = True


class LooksLikeTwoMult(Scene):
    const = "2"

    def construct(self):
        tex = TexText(f"Looks like $z \\rightarrow {self.const}\\cdot z$")
        tex.set_stroke(BLACK, 5, background=True)
        self.play(FadeIn(tex, lag_ratio=0.1))
        self.wait()


class LooksLikeTwoiMult(LooksLikeTwoMult):
    const = "2i"


class Cycles(FixedPoints):
    def construct(self):
        # Set the stage
        iter_label = self.add_labels()
        rule, iterations = iter_label
        self.remove(rule)
        iterations.to_edge(UP)

        plane = self.add_plane()
        z0_dot, z0_label = self.add_z_dot(complex(-1.1, 0.6), "z_0")
        z1_dot, z1_label = self.add_z_dot(complex(0.2, -0.5), "z_1")
        z1_label.next_to(z1_dot, UR, SMALL_BUFF)
        z_dots = VGroup(z0_dot, z1_dot)
        z_labels = VGroup(z0_label, z1_label)

        # Ask question
        question = TexText("When does $z$ cycle?")
        question.next_to(plane, UP, MED_SMALL_BUFF)

        kw = {"path_arc": PI / 3, "buff": 0.1}
        arrows = VGroup(
            Arrow(z0_dot, z1_dot, **kw),
            Arrow(z1_dot, z0_dot, **kw),
        )
        arrows.set_stroke(opacity=0.75)
        z_dot = z0_dot.copy()
        z_dot.set_color(YELLOW)

        self.add(question)
        self.play(
            ShowCreation(arrows[0]),
            TransformFromCopy(z0_dot, z1_dot, path_arc=-PI / 3),
            TransformFromCopy(z0_label, z1_label, path_arc=-PI / 3),
        )
        self.wait()
        self.play(
            TransformFromCopy(z1_dot, z_dot, path_arc=-PI / 3),
            ShowCreation(arrows[1]),
        )
        self.wait()
        for n in range(1, 5):
            self.play(z_dot.move_to, z_dots[n % 2], path_arc=PI / 3)
            self.wait()

        # Show formula
        kw = {
            "tex_to_color_map": {"z": BLUE},
            "isolate": ["f"],
        }

        f2_equation = Tex("f(f(z)) = z", **kw)
        f2_equation.next_to(plane, RIGHT, MED_LARGE_BUFF, aligned_edge=UP)

        julia_fractal = JuliaFractal(plane)
        julia_fractal.set_c(-0.18 + 0.77j)

        z2c = Tex("f(z) = z^2 + c", **kw)
        z2c.next_to(f2_equation, RIGHT, LARGE_BUFF)

        self.play(FadeIn(f2_equation, 0.25 * DOWN))
        self.wait()
        self.add(julia_fractal, plane, z_dots, z_dot, z_labels, arrows, z_dot)
        julia_fractal.set_opacity(0)
        self.play(
            julia_fractal.animate.set_opacity(0.75),
            Write(z2c),
        )
        self.wait()

        # Example with z^2 + c
        julia_f2_eqs = VGroup(
            Tex("(z^2 + c)^2 + c = z", **kw),
            Tex("z^4 + 2cz^2 -z + c^2 + c = 0", **kw),
        )
        julia_f2_eqs.arrange(DOWN, buff=0.7, aligned_edge=LEFT)
        julia_f2_eqs.next_to(f2_equation, DOWN, buff=1.0, aligned_edge=LEFT)

        eq_arrows = VGroup(
            Arrow(f2_equation.get_bottom(), julia_f2_eqs.get_top()),
            Arrow(z2c.get_bottom(), julia_f2_eqs.get_top()),
        )

        self.play(
            *map(ShowCreation, eq_arrows),
            FadeIn(julia_f2_eqs[0], 0.5 * DOWN)
        )
        self.wait()
        self.play(
            TransformMatchingShapes(
                julia_f2_eqs[0].copy(),
                julia_f2_eqs[1]
            )
        )
        self.wait()

        # Add fixed_points
        fixed_dots = VGroup(
            Dot(plane.c2p(0.8, -0.5), color=GREY_A),
            Dot(plane.c2p(-0.9, -0.6), color=GREY_A),
        )
        arrow_loops = VGroup(*(
            self.get_arrow_loop(dot)
            for dot in fixed_dots
        ))

        for dot, loop in zip(fixed_dots, arrow_loops):
            loop.set_color(GREY_B)
            self.play(
                FadeIn(dot, scale=0.3),
                ShowCreation(loop),
            )
        self.wait()
        self.play(
            LaggedStartMap(FadeOut, VGroup(
                eq_arrows, *julia_f2_eqs,
                *fixed_dots, *arrow_loops,
            ))
        )

        # N cycles
        fn_eq = Tex(
            "f(f(\\cdots f(z) \\cdots)) = z",
            **kw
        )
        fn_eq.move_to(f2_equation, LEFT)
        fn_eq.shift(SMALL_BUFF * DOWN)
        brace = Brace(
            fn_eq[:fn_eq.index_of_part_by_tex("z")],
            DOWN
        )
        brace_tex = brace.get_tex("n \\text{ times}", buff=SMALL_BUFF)
        brace_tex.scale(0.7, about_edge=UP)

        for z in [-0.2 + 0.6j, 1.1 - 0.6j, 0.4 + 0.2j]:
            dot = z0_dot.copy()
            dot.set_fill(BLUE_D)
            dot.move_to(plane.n2p(z))
            z_dots.add(dot)

        n_arrows = VGroup()
        for d1, d2 in adjacent_pairs(z_dots):
            arrow = Arrow(d1, d2, buff=0.1, path_arc=PI / 8)
            arrow.set_stroke(WHITE, opacity=0.7)
            n_arrows.add(arrow)

        dot_anims = []
        n = len(z_dots)
        for k in range(1, n + 1):
            dot_anims.append(
                ApplyMethod(z_dot.move_to, z_dots[k % n], path_arc=PI / 8)
            )

        self.play(
            TransformMatchingShapes(f2_equation, fn_eq),
            FadeOut(z2c),
            ReplacementTransform(arrows, n_arrows),
            *map(GrowFromCenter, z_dots[2:])
        )
        self.play(
            GrowFromCenter(brace),
            FadeIn(brace_tex, SMALL_BUFF * DOWN),
            Succession(*dot_anims, run_time=3),
        )
        self.wait()

        # Ask about how many solutions
        morty = Mortimer(height=2)
        morty.to_corner(DR)
        z2c.next_to(fn_eq, UP, buff=MED_LARGE_BUFF)

        self.play(
            PiCreatureSays(
                morty, "How many solutions?",
                bubble_config={"height": 2, "width": 4}
            ),
        )
        self.play(Blink(morty))
        self.wait()
        self.play(
            FadeIn(z2c),
            RemovePiCreatureBubble(
                morty,
                target_mode="pondering",
                look_at=z2c,
            )
        )
        self.wait()

        # 1,000,000-cycles
        million = Integer(1e6, font_size=36)
        million.next_to(brace, DOWN)
        million.set_value(0)
        mega_poly = Tex(
            "z^{2^{1{,}000{,}000}} +",
            "\\cdots \\text{(nightmare)} \\cdots",
            "= 0",
            **kw
        )
        mega_poly.next_to(million, DOWN, buff=0.75)
        mega_poly.align_to(fn_eq, LEFT)

        expr = self.show_composition(million, morty, **kw)
        self.play(
            ChangeDecimalToValue(million, 1e6, run_time=2),
            VFadeIn(million),
            FadeOut(brace_tex),
            morty.change("raise_right_hand", fn_eq)
        )
        self.play(Blink(morty))
        self.play(
            FadeOut(expr),
            Write(mega_poly),
            morty.animate.set_height(1.8, about_edge=DR).change("horrified", mega_poly),
        )
        self.play(morty.animate.look_at(mega_poly.get_right()))
        self.wait()

        # Show "million" dots
        N = 5000
        points = np.random.random((N, 3))
        points[:, 2] = 0
        dots = DotCloud(points, radius=0)
        dots.replace(plane)
        dots.set_radius(0.01)
        dots.set_color(GREY_B)
        dots.set_opacity(1)
        dots.add_updater(lambda m: m)

        self.play(
            FadeOut(z_labels),
            FadeOut(z_dots),
            FadeOut(n_arrows),
            morty.change("erm", dots),
            ShowCreation(dots, run_time=5),
        )
        self.wait()
        self.play(
            FadeOut(dots),
            FadeOut(mega_poly),
            FadeOut(morty),
            FadeOut(z2c),
        )

        # Rational map
        rational = Tex("f(z) = {A(z) \\over B(z)}", **kw)
        rational.next_to(million, DOWN, LARGE_BUFF)
        rational.align_to(fn_eq, LEFT)

        self.play(FadeIn(rational))
        for arrow, dot in zip(n_arrows, [*z_dots[1:], z_dots[0]]):
            self.play(
                ShowCreation(arrow),
                z_dot.animate.move_to(dot),
                path_arc=PI / 6,
            )
            self.add(dot, z_dot)

        # Ask about attracting cycle
        new_question = Text("When is a cycle attracting?")
        new_question.get_part_by_text("attracting").set_color(YELLOW)
        new_question.next_to(question, RIGHT, LARGE_BUFF)

        self.play(
            Write(new_question, run_time=1),
            fn_eq.animate.shift(0.5 * DOWN),
            FadeOut(brace),
            FadeOut(million),
        )

        circle = Circle(radius=0.5)
        circle.set_stroke(YELLOW, 1, 1)
        circle.set_fill(YELLOW, 0.25)
        h_tracker = ValueTracker(1.0)
        circle.add_updater(lambda m: m.set_height(h_tracker.get_value()))
        circle.add_updater(lambda m: m.move_to(z_dot))

        multipliers = [0.9, 0.9, 1.2, 0.5, 1.1]

        self.add(circle, z_dot)
        self.play(GrowFromCenter(circle))

        for n in range(3):
            for mult, dot in zip(multipliers, [*z_dots[1:], z_dots[0]]):
                self.play(
                    ApplyMethod(z_dot.move_to, dot, path_arc=PI / 6),
                    h_tracker.animate.set_value(h_tracker.get_value() * mult),
                )

        # Possibly add on a bit for Fatou's theorem?
        theorem = TexText(
            "Theorem (Fatou 1919): If $f(z)$ has an\\\\",
            "attracting cycle, then at least one solution\\\\",
            "to $f'(z) = 0$ will fall into it.",
            font_size=36,
        )
        theorem.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        theorem.next_to(fn_eq, DOWN, LARGE_BUFF, aligned_edge=LEFT)

        self.play(
            rational.animate.set_height(0.8).next_to(fn_eq, RIGHT, LARGE_BUFF),
            FadeIn(theorem),
        )

    def show_composition(self, ref_mob, morty, **kwargs):
        tex = "z^2 + c"
        polys = VGroup(Tex(tex, **kwargs))
        for n in range(20):
            new_tex_parts = ["\\left(", tex, "\\right)^2 + c"]
            polys.add(Tex(*new_tex_parts))
            if n < 3:
                tex = "".join(new_tex_parts)

        for poly in polys:
            poly.set_max_width(5)
            poly.next_to(ref_mob, DOWN, LARGE_BUFF, aligned_edge=LEFT)

        degree = VGroup(Text("Degree: "), Integer(2))
        degree.arrange(RIGHT)
        degree.next_to(polys, DOWN, aligned_edge=LEFT)

        curr_poly = polys[0]
        self.play(
            FadeIn(curr_poly),
            FadeIn(degree),
            morty.change("tease", curr_poly),
        )
        for n, poly in enumerate(polys[1:]):
            anims = []
            if n == 4:
                anims.append(morty.change("erm"))
            self.play(
                curr_poly.animate.replace(poly[1]),
                FadeIn(poly[::2]),
                UpdateFromAlphaFunc(
                    degree, lambda m, a: m[1].set_value(
                        (2**(n + 1) if a < 0.5 else 2**(n + 2))
                    )
                ),
                *anims,
                run_time=(1 if n < 5 else 0.25)
            )
            poly.replace_submobject(1, curr_poly)
            self.add(poly)
            curr_poly = poly
        self.wait()
        self.play(FadeOut(degree))
        return poly


class TwoToMillionPoints(Scene):
    c = -0.18 + 0.77j
    plane_height = 7

    def construct(self):
        plane, julia_fractal = self.get_plane_and_fractal()

        words = TexText("$\\approx 2^{1{,}000{,}000}$ solutions!")
        words.set_stroke(BLACK, 8, background=True)
        words.move_to(plane, UL)
        words.shift(MED_SMALL_BUFF * DR)

        points = self.get_julia_set_points(plane, 100000, 1000)
        dots = DotCloud(points)
        dots.set_color(YELLOW)
        dots.set_opacity(1)
        dots.set_radius(0.025)
        dots.add_updater(lambda m: m)
        dots.make_3d()

        self.add(julia_fractal, plane, words)
        self.play(ShowCreation(dots, run_time=10))

    def get_plane_and_fractal(self):
        plane = ComplexPlane((-2, 2), (-2, 2))
        plane.set_height(self.plane_height)
        fractal = JuliaFractal(plane)
        fractal.set_c(self.c)
        return plane, fractal

    def get_julia_set_points(self, plane, n_points, n_steps):
        values = np.array([
            complex(math.cos(x), math.sin(x))
            for x in np.linspace(0, TAU, n_points)
        ])

        c = self.c
        for n in range(n_steps):
            units = -1 + 2 * np.random.randint(0, 2, len(values))
            values[:] = (units * np.sqrt(values[:])) - c
        values += c

        return np.array(list(map(plane.n2p, values)))


class CyclesHaveSolutions(Scene):
    def construct(self):
        text = VGroup(
            Tex(
                "f^n(z) = z \\text{ has solutions}",
                tex_to_color_map={"z": BLUE},
            ),
            Tex("\\sim D^n \\text{ of them...}"),
        )
        text.arrange(DOWN)

        for part in text:
            self.play(FadeIn(part))
            self.wait()


class MandelbrotFunctions(Scene):
    def construct(self):
        kw = {"tex_to_color_map": {"c": YELLOW}}
        group = VGroup(
            Tex("f(z) = z^2 + c", **kw),
            Tex("f'(z) = 2z"),
        )
        group.arrange(DOWN, aligned_edge=LEFT)
        self.add(group)


class AmbientNewtonRepetition(RepeatedNewton):
    coefs = [-4, 0, -3, 0, 1]
    show_fractal_background = True
    show_coloring = False
    n_steps = 20
    dot_density = 10.0
    points_scalar = 2.0
    dots_config = {
        "radius": 0.025,
        "color": GREY_A,
        "gloss": 0.4,
        "shadow": 0.1,
        "opacity": 0.5,
    }


class AmbientNewtonBoundary(AmbientNewtonRepetition):
    def construct(self):
        self.add_plane()
        fractal = self.get_fractal()
        self.remove(self.plane)

        fractal.set_julia_highlight(1e-4)
        fractal.set_colors(5 * [WHITE])
        self.play(GrowFromPoint(
            fractal,
            fractal.get_corner(UL),
            run_time=5
        ))
        self.wait()


class CyclicAttractor(RepeatedNewton):
    coefs = [2, -2, 0, 1]
    n_steps = 20
    show_coloring = False
    cluster_radius = 0.5

    def add_plane(self):
        super().add_plane()
        self.plane.axes.set_stroke(GREY_B, 1)
        self.plane.scale(1.7)

    def add_labels(self):
        super().add_labels()
        eq = self.corner_group[1]
        self.play(FlashAround(eq, run_time=3))

    def get_original_points(self):
        return [
            (r * np.cos(theta), r * np.sin(theta), 0)
            for r in np.linspace(0, self.cluster_radius, 10)
            for theta in np.linspace(0, TAU, int(50 * r)) + TAU * np.random.random()
        ]


class CyclicAttractorSmallRadius(CyclicAttractor):
    cluster_radius = 0.25
    colors = ROOT_COLORS_DEEP[0::2]

    def construct(self):
        super().construct()

        fractal = NewtonFractal(
            self.plane,
            coefs=self.coefs,
            colors=self.colors,
            black_for_cycles=True,
        )
        dots = VGroup(*(
            Dot(rd.get_center()) for rd in self.root_dots
        ))
        dots.set_stroke(BLACK, 3)
        dots.set_fill(opacity=0)

        self.add(fractal, *self.mobjects, dots)
        self.play(
            FadeIn(fractal),
            self.plane.animate.fade(0.5)
        )
        self.wait()


class CyclicExercise(Scene):
    def construct(self):
        words = TexText(
            "Exercise 2: If $f(z) = z - {z^3 - 2z + 2 \\over 3z^2 - 2}$,\\\\",
            "and $g(z) = f(f(z))$, confirm that $|g'(0)| < 1$."
        )
        words[1].shift(SMALL_BUFF * DOWN)
        box = SurroundingRectangle(words, buff=0.45)
        box.set_stroke(WHITE, 2)
        box.set_fill(BLACK, 1)
        group = VGroup(box, words)
        group.to_edge(UP, buff=0)

        hint = TexText(
            "Hint: Don't expand out $g(z)$. Use\\\\",
            "the chain rule: $g'(0) = f'(f(0))f'(0)$"
        )
        hint.scale(0.8)
        hint.set_color(GREY_A)
        hint_box = SurroundingRectangle(hint, buff=0.25)
        hint_box.match_style(box)
        hint_group = VGroup(hint_box, hint)
        hint_group.next_to(group, DOWN, buff=0)

        self.add(group)
        self.wait(2)
        self.play(FadeIn(hint_group))
        self.wait()


class AskHowOftenThisHappensAlt(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            TexText("How often does\\\\this happen?"),
            bubble_config={
                "height": 3,
                "width": 4,
            },
            index=0,
        )
        self.play(
            self.teacher.change("raise_right_hand", 3 * UR),
            self.change_students(
                "raise_left_hand", "pondering", "pondering",
                look_at=3 * UR,
            )
        )
        self.play_student_changes(
            "raise_left_hand", "erm", "erm",
            look_at=3 * UR,
        )
        self.wait()
        self.play(self.teacher.change("tease", 3 * UR))
        self.play_student_changes(
            "confused", "pondering", "thinking",
            look_at=3 * UR,
        )
        self.wait(8)
        return

        self.wait(2)
        self.play(
            PiCreatureSays(
                self.teacher, "You'll like this",
                target_mode="tease",
                run_time=1,
            ),
            self.students[1].change("thinking", self.teacher.eyes),
            self.students[2].change("thinking", self.teacher.eyes),
        )
        self.wait(2)


class WhatDoesBlackMean(Scene):
    def construct(self):
        lhs = TexText("$z_n$ never gets\\\\near a root")
        rhs = TexText("$\\Rightarrow$ ", "Black ", )
        rhs.next_to(lhs, RIGHT)
        words = VGroup(lhs, rhs)
        words.next_to(ORIGIN, UR, MED_LARGE_BUFF)
        words.to_edge(RIGHT)
        words.set_stroke(BLACK, 5, background=True)
        # lhs[0].set_color(BLACK)
        # lhs[0].set_stroke(width=0)

        arrow = Arrow(ORIGIN, words.get_left(), buff=0.1)

        self.play(
            ShowCreation(arrow),
            Write(lhs),
        )
        self.wait()
        self.play(FadeIn(rhs))
        self.wait()


class PlayWithRootsSeekingCycles(ThreeRootFractal):
    coefs = [2, -2, 0, 1]

    def construct(self):
        super().construct()
        self.fractal.uniforms["black_for_cycles"] = 1.0


class ShowCenterOfMassPoint(PlayWithRootsSeekingCycles):
    display_root_values = True
    only_center = False

    def construct(self):
        super().construct()

        mean_dot = glow_dot(ORIGIN)
        mean_dot.add_updater(lambda m: m.move_to(
            sum([rd.get_center() for rd in self.root_dots]) / 3
        ))
        self.add(mean_dot)

        if self.only_center:
            mean_dot.set_opacity(0)
            self.fractal.replace(mean_dot, stretch=True)
            self.fractal.add_updater(lambda m: m.move_to(mean_dot))
            window = Square()
            window.set_stroke(WHITE, 1)
            window.set_fill(BLACK, 0)
            window.replace(self.fractal, stretch=True)
            window.add_updater(lambda m: m.move_to(mean_dot))
            self.add(window)

        mean_label = Tex("(r_1 + r_2 + r_3) / 3", font_size=24)
        mean_label.set_stroke(BLACK, 2, background=True)
        mean_label.add_updater(lambda m: m.next_to(mean_dot, UP, buff=0.1))
        self.add(mean_label)

        circle = Circle(radius=2)
        circle.rotate(PI)
        circle.stretch(0.9)
        circle.move_to(self.root_dots, LEFT)

        self.play(
            MoveAlongPath(
                self.root_dots[2], circle,
                rate_func=linear,
                run_time=20
            )
        )
        self.play(
            self.root_dots[2].animate.move_to(self.plane.n2p(-3)),
            rate_func=there_and_back,
            run_time=10,
        )


class ShowCenterOfMassPointFocusIn(ShowCenterOfMassPoint):
    only_center = True


class CenterOfMassStatement(Scene):
    def construct(self):
        words = TexText(
            "If there's an attracting cycle, the seed\\\\",
            "$z_0 = (r_1 + r_2 + r_3) / 3$ will fall into it."
        )
        words.set_stroke(BLACK, 8, background=True)
        words.to_corner(UL)

        self.play(Write(words))
        self.wait()


class GenerateCubicParameterPlot(Scene):
    colors = ROOT_COLORS_DEEP[0::2]

    def construct(self):
        # Title
        colors = self.colors
        kw = {
            "tex_to_color_map": {
                "\\lambda": colors[2],
            }
        }
        title = Tex(
            "z_{n+1} = z_n - {P(z_n) \\over P'(z_n)} \\qquad\\qquad ",
            "P(z) = (z - 1)(z + 1)(z - \\lambda)",
            font_size=30,
            **kw
        )
        title.to_edge(UP)
        self.add(title)

        # Planes
        planes = VGroup(*(
            ComplexPlane(
                (-2, 2), (-2, 2),
                background_line_style={
                    "stroke_color": GREY_B,
                    "stroke_opacity": 0.5,
                },
            )
            for x in range(2)
        ))
        for plane, vect in zip(planes, [DL, DR]):
            plane.add_coordinate_labels(font_size=18)
            plane.set_height(5)
            plane.to_corner(vect, buff=MED_SMALL_BUFF)

        root_dots = VGroup(*(
            Dot(planes[0].n2p(z), color=color)
            for color, z in zip(colors, [-1, 1, 1j])
        ))
        root_dots.set_stroke(BLACK, 2)

        lambda_label = Tex("\\lambda", font_size=36)
        lambda_label.set_color(interpolate_color(colors[2], WHITE, 0.75))
        lambda_label.set_stroke(BLACK, 3, background=True)
        lambda_label.add_updater(lambda m: m.next_to(
            root_dots[2], UR, buff=SMALL_BUFF,
        ))

        # Fractals
        left_fractal = NewtonFractal(
            planes[0], coefs=[-1, 0, 0, 1],
            colors=colors,
            black_for_cycles=True,
        )
        left_fractal.add_updater(lambda m: m.set_roots([
            planes[0].p2n(rd.get_center())
            for rd in root_dots
        ]))

        right_fractal = MetaNewtonFractal(
            planes[1],
            colors=colors,
            fixed_roots=[-1, 1],
        )
        row_meta_fractal = right_fractal.deepcopy()
        col_meta_fractal = right_fractal.deepcopy()

        self.add(left_fractal, planes[0], planes, root_dots, lambda_label)

        # Plane titles
        plane_titles = VGroup(
            TexText("Pixel $\\leftrightarrow z_0$"),
            TexText("Pixel $\\leftrightarrow$ ", "$\\lambda$"),
        )
        plane_titles[1][-1].set_color(colors[2])
        for plane_title, plane in zip(plane_titles, planes):
            plane_title.next_to(plane, UP, MED_SMALL_BUFF)
        plane_titles[0].align_to(plane_titles[1], UP)

        self.add(plane_titles[0])

        # Show left plane
        pins = VGroup()
        for rd in root_dots[:2]:
            pin = SVGMobject("push_pin")
            pin.set_fill(GREY_C)
            pin.set_stroke(width=0)
            pin.set_gloss(0.5)
            pin.set_height(0.3)
            pin.rotate(10 * DEGREES)
            pin.move_to(rd.get_center(), DR)
            pins.add(pin)
            self.play(
                FadeIn(pin, 0.25 * DR),
                FlashAround(rd),
            )

        self.wait()
        circle = Circle()
        circle.scale(0.5)
        circle.rotate(PI / 2)
        circle.move_to(root_dots[2].get_center(), UP)

        self.play(MoveAlongPath(root_dots[2], circle, run_time=5))
        self.play(
            root_dots[2].animate.move_to(planes[0].get_corner(UL)),
            run_time=2
        )
        self.wait()

        # Center of mass dot
        com_dot = Dot()
        com_dot.set_fill(opacity=0)
        com_dot.set_stroke(YELLOW, 3)
        com_dot.add_updater(lambda m: m.move_to(
            np.array([rd.get_center() for rd in root_dots]).mean(0)
        ))
        self.play(FadeIn(com_dot, scale=0.5), FadeOut(pins))
        self.wait()

        # Right plane
        self.play(Write(plane_titles[1]))
        self.wait()

        # Show filling process
        step = 0.1
        square = Square()
        square.set_stroke(WHITE, 2)
        arrow = Arrow(LEFT, RIGHT, stroke_width=3)
        self.add(row_meta_fractal, col_meta_fractal, planes[1])
        self.add(square, arrow)

        x_range = np.arange(-2, 2 + step, step)
        y_range = np.arange(2, -2, -step)

        thin_height = plane.get_y_unit_size() * step
        col_meta_fractal.set_height(thin_height)
        square.set_height(thin_height)

        epsilon = 1e-6
        for y, back in zip(y_range, it.cycle([False, True])):
            height = max(epsilon, plane.get_y_unit_size() * abs(2 - y))
            row_meta_fractal.set_height(height, stretch=True)
            row_meta_fractal.move_to(planes[1], UP)
            x0 = (2 if back else -2)
            for x in (x_range[::-1] if back else x_range):
                width = max(epsilon, planes[1].get_x_unit_size() * abs(x - x0))
                col_meta_fractal.set_width(width, stretch=True)
                col_meta_fractal.next_to(row_meta_fractal, DOWN, buff=0)
                col_meta_fractal.align_to(row_meta_fractal, RIGHT if back else LEFT)

                root_dots[2].move_to(planes[0].c2p(x, y))
                square.move_to(
                    col_meta_fractal, DOWN + (LEFT if back else RIGHT)
                )
                self.update_mobjects(0)
                arrow.put_start_and_end_on(
                    com_dot.get_center(),
                    square.get_center(),
                )
                self.wait(1 / 15)

        self.play(FadeOut(arrow), FadeOut(square))
        self.wait()

        # Zoom in to meta fractal
        frame = self.camera.frame
        self.play(
            frame.animate.replace(planes[1], 1),
            run_time=3,
        )
        self.wait()


class Z0RuleLabel(Scene):
    def construct(self):
        label = Tex("z_0 = (r_1 + r_2 + r_3) / 3")
        self.add(label)


class WhyFractals(Scene):
    def construct(self):
        words = Text("Why fractals?")
        words.set_stroke(BLACK, 5, background=True)
        self.play(Write(words))
        self.wait()


class SmallCircleProperty(Scene):
    def construct(self):
        # Titles
        rule = TexText(
            "For any rational map, color points\\\\based on their limiting behavior...\\\\",
            "(which limit point, which limit cycle, etc.)",
            font_size=36,
        )
        rule.to_corner(UL)
        rule[-1].shift(SMALL_BUFF * DOWN)
        rule[-1].set_color(GREY_B)

        r_map = Tex("z_{n + 1} = A(z_n) / B(z_n)")
        r_map.to_corner(UR)

        # Fractal
        colors = [
            MANDELBROT_COLORS[0], BLUE_C, BLUE_E, ROOT_COLORS_DEEP[1],
        ]
        plane = ComplexPlane((-2, 2), (-2, 2))
        plane.set_height(5)
        plane.to_corner(DR)
        fractal = NewtonFractal(plane, coefs=[5, 4j, 3, 2, 1], colors=colors)

        # Circles
        circles = Circle(radius=0.6).get_grid(3, 1, buff=0.5)
        circles.replace(plane, 1)
        circles.set_x(-1)
        circles.set_stroke(WHITE, 2)

        circles[0].set_fill(colors[2], 1)
        semis = circles[1].copy().pointwise_become_partial(circles[1], 0, 0.5).replicate(2)
        semis[1].rotate(PI, about_point=circles[1].get_center())
        semis[0].set_fill(colors[0], 1)
        semis[1].set_fill(colors[1], 1)
        circles[1].add(semis)

        multi_color_image = ImageMobject("MulticoloredNewtonsMapCircle")
        multi_color_image.add_updater(lambda m: m.replace(circles[2]))

        # Circle label
        circle_labels = VGroup(
            Text("One color"),
            Text("Some colors"),
            Text("All colors"),
        )
        marks = VGroup(Checkmark(), Exmark(), Checkmark())
        for label, mark, circle in zip(circle_labels, marks, circles):
            mark.set_height(0.5 * circle.get_height())
            mark.next_to(circle, LEFT, MED_LARGE_BUFF)
            label.next_to(mark, LEFT, MED_LARGE_BUFF)

        # Little circles
        lil_circles = circles[0].replicate(2)
        lil_circles.set_height(0.2)
        lil_circles[0].move_to(plane.get_corner(UL) + MED_SMALL_BUFF * DR)
        lil_circles[1].move_to(plane).shift([0.22, -0.12, 0])
        lil_circles[1].set_fill(opacity=0)

        lines = VGroup()
        for big, lil in zip(circles[::2], lil_circles):
            vect = normalize(lil.get_center() - big.get_center())
            v1 = rotate_vector(vect, 75 * DEGREES)
            v2 = rotate_vector(vect, -75 * DEGREES)
            big.insert_n_curves(20)
            lines.add(VGroup(
                Line(lil.get_top(), big.get_boundary_point(v1)),
                Line(lil.get_bottom(), big.get_boundary_point(v2)),
            ))
        lines.set_stroke(WHITE, 1)

        # Intro anims
        self.add(rule[0])
        self.play(FadeIn(r_map))
        self.wait()
        fractal.set_opacity(0)
        self.add(fractal)
        for i in range(1, 5):
            opacities = np.zeros(4)
            opacities[:i] = 1
            self.play(fractal.animate.set_opacities(*opacities))
        self.wait()
        self.play(FadeIn(rule[-1], lag_ratio=0.1, run_time=2))
        self.wait()

        # Show circles
        self.add(lil_circles[0])
        self.play(
            TransformFromCopy(lil_circles[0], circles[0]),
            *map(ShowCreation, lines[0]),
            FadeIn(circle_labels[0]),
        )
        self.play(Write(marks[0]))
        self.wait()

        self.add(multi_color_image)
        self.add(lil_circles[1])
        self.play(
            TransformFromCopy(lil_circles[1], circles[2]),
            *map(ShowCreation, lines[1]),
            FadeIn(circle_labels[2]),
        )
        self.play(Write(marks[2]))
        self.wait()

        self.play(
            FadeIn(circle_labels[1]),
            FadeIn(circles[1]),
        )
        self.play(Write(marks[1]))
        self.wait()


class MentionFatouSetsAndJuliaSets(Scene):
    colors = [RED_E, BLUE_E, TEAL_E, MAROON_E]

    def construct(self):
        # Introduce terms
        f_group, j_group = self.get_fractals()
        f_name, j_name = VGroup(
            Text("Fatou set"),
            Text("Julia set"),
        )
        f_name.next_to(f_group, UP, MED_LARGE_BUFF)
        j_name.next_to(j_group, UP, MED_LARGE_BUFF)

        self.play(
            Write(j_name),
            GrowFromCenter(j_group)
        )
        self.wait()
        self.play(
            Write(f_name),
            *map(GrowFromCenter, f_group)
        )
        self.wait()

        # Define Fatou set
        fatou_condition = self.get_fatou_condition()
        fatou_condition.set_width(FRAME_WIDTH - 1)
        fatou_condition.center().to_edge(UP, buff=1.0)
        lhs, arrow, rhs = fatou_condition
        f_line = Line(LEFT, RIGHT)
        f_line.match_width(fatou_condition)
        f_line.next_to(fatou_condition, DOWN)
        f_line.set_stroke(WHITE, 1)

        self.play(
            FadeOut(j_name, RIGHT),
            FadeOut(j_group, RIGHT),
            Write(lhs)
        )
        self.wait()
        for words in lhs[-1]:
            self.play(FlashUnder(
                words,
                buff=0,
                time_width=1.5
            ))
        self.play(Write(arrow))
        self.play(LaggedStart(
            FadeTransform(f_name.copy(), rhs[1][:8]),
            FadeIn(rhs),
            lag_ratio=0.5
        ))
        self.wait()

        # Show Julia set
        otherwise = Text("Otherwise...")
        otherwise.next_to(rhs, DOWN, LARGE_BUFF)
        j_condition = TexText("$z_0 \\in$", " Julia set", " of $f$")
        j_condition.match_height(rhs)
        j_condition.next_to(otherwise, DOWN, LARGE_BUFF)

        j_group.set_height(4.0)
        j_group.to_edge(DOWN)
        j_group.set_x(-1.0)
        j_name = j_condition.get_part_by_tex("Julia set")
        j_underline = Underline(j_name, buff=0.05)
        j_underline.set_color(YELLOW)
        arrow = Arrow(
            j_name.get_bottom(),
            j_group.get_right(),
            path_arc=-45 * DEGREES,
        )
        arrow.set_stroke(YELLOW, 5)

        julia_set = j_group[0]
        julia_set.update()
        julia_set.suspend_updating()
        julia_copy = julia_set.copy()
        julia_copy.clear_updaters()
        julia_copy.set_colors(self.colors)
        julia_copy.set_julia_highlight(0)

        mover = f_group[:-4]
        mover.generate_target()
        mover.target.match_width(rhs)
        mover.target.next_to(rhs, UP, MED_LARGE_BUFF)
        mover.target.shift_onto_screen(buff=SMALL_BUFF)

        self.play(
            ShowCreation(f_line),
            FadeOut(f_name),
            MoveToTarget(mover),
        )
        self.play(
            Write(otherwise),
            FadeIn(j_condition, 0.5 * DOWN)
        )
        self.wait()
        self.play(
            ShowCreation(j_underline),
            ShowCreation(arrow),
            FadeIn(j_group[1]),
            FadeIn(julia_copy)
        )
        self.play(
            GrowFromPoint(julia_set, julia_set.get_corner(UL), run_time=2),
            julia_copy.animate.set_opacity(0.2)
        )
        self.wait()

    def get_fractals(self, jy=1.5, fy=-2.5):
        coefs = roots_to_coefficients([-1.5, 1.5, 1j, -1j])
        n = len(coefs) - 1
        colors = self.colors
        f_planes = VGroup(*(self.get_plane() for x in range(n)))
        f_planes.arrange(RIGHT, buff=LARGE_BUFF)
        plusses = Tex("+").replicate(n - 1)
        f_group = Group(*it.chain(*zip(f_planes, plusses)))
        f_group.add(f_planes[-1])
        f_group.arrange(RIGHT)
        fatou = Group(*(
            NewtonFractal(f_plane, coefs=coefs, colors=colors)
            for f_plane in f_planes
        ))
        for i, fractal in enumerate(fatou):
            opacities = n * [0.2]
            opacities[i] = 1
            fractal.set_opacities(*opacities)
        f_group.add(*fatou)
        f_group.set_y(fy)

        j_plane = self.get_plane()
        j_plane.set_y(jy)
        julia = NewtonFractal(j_plane, coefs=coefs, colors=5 * [GREY_A])
        julia.set_julia_highlight(1e-3)
        j_group = Group(julia, j_plane)

        for fractal, plane in zip((*fatou, julia), (*f_planes, j_plane)):
            fractal.plane = plane
            fractal.add_updater(
                lambda m: m.set_offset(
                    m.plane.get_center()
                ).set_scale(
                    m.plane.get_x_unit_size()
                ).replace(m.plane)
            )

        fractals = Group(f_group, j_group)
        return fractals

    def get_plane(self):
        plane = ComplexPlane(
            (-2, 2), (-2, 2),
            background_line_style={"stroke_width": 1, "stroke_color": GREY}
        )
        plane.set_height(2)
        plane.set_opacity(0)
        box = SurroundingRectangle(plane, buff=0)
        box.set_stroke(WHITE, 1)
        plane.add(box)
        return plane

    def get_fatou_condition(self):
        zn = Tex(
            "z_0", "\\overset{f}{\\longrightarrow}",
            "z_1", "\\overset{f}{\\longrightarrow}",
            "z_2", "\\overset{f}{\\longrightarrow}",
            "\\dots",
            "\\longrightarrow"
        )
        words = VGroup(
            TexText("Stable fixed point"),
            TexText("Stable cycle"),
            TexText("$\\infty$"),
        )
        words.arrange(DOWN, aligned_edge=LEFT)
        brace = Brace(words, LEFT)
        zn.next_to(brace, LEFT)
        lhs = VGroup(zn, brace, words)

        arrow = Tex("\\Rightarrow")
        arrow.scale(2)
        arrow.next_to(lhs, RIGHT, MED_LARGE_BUFF)
        rhs = Tex("z_0 \\in", " \\text{Fatou set of $f$}")
        rhs.next_to(arrow, RIGHT, buff=MED_LARGE_BUFF)

        result = VGroup(lhs, arrow, rhs)

        return result


class ShowJuliaSetPoint(TwoToMillionPoints):
    plane_height = 14
    show_disk = False
    n_steps = 60
    disk_radius = 0.02

    def construct(self):
        # Background
        plane, fractal = self.get_plane_and_fractal()

        plane.add_coordinate_labels(font_size=24)
        for mob in plane.family_members_with_points():
            if isinstance(mob, Line):
                mob.set_stroke(opacity=0.5 * mob.get_stroke_opacity())
        self.add(fractal, plane)

        # Points
        points = list(self.get_julia_set_points(plane, n_points=1, n_steps=1000))

        def func(p):
            z = plane.p2n(p)
            return plane.n2p(z**2 + self.c)

        for n in range(100):
            points.append(func(points[-1]))

        dot = Dot(points[0])
        dot.set_color(YELLOW)

        self.add(dot)

        if self.show_disk:
            dot.scale(0.5)
            disk = dot.copy()
            disk.insert_n_curves(10000)
            disk.set_height(plane.get_x_unit_size() * self.disk_radius)
            disk.set_fill(YELLOW, 0.25)
            disk.set_stroke(YELLOW, 2, 1)
            self.add(disk, dot)

        frame = self.camera.frame
        path_arc = 30 * DEGREES
        point = dot.get_center().copy()
        for n in range(self.n_steps):
            new_point = func(point)
            arrow = Arrow(point, new_point, path_arc=path_arc, buff=0)
            arrow.set_stroke(WHITE, opacity=0.9)
            self.add(dot.copy().set_opacity(0.5))
            anims = []
            if self.show_disk:
                disk.generate_target()
                disk.target.apply_function(func)
                disk.target.make_approximately_smooth()
                anims.append(MoveToTarget(disk, path_arc=path_arc))
                if disk.target.get_height() > frame.get_height():
                    anims.extend([
                        mob.animate.scale(2.0)
                        for mob in [frame, fractal]
                    ])
            self.play(
                ApplyMethod(dot.move_to, new_point, path_arc=path_arc),
                ShowCreation(arrow),
                *anims,
            )
            self.play(FadeOut(arrow))
            point = new_point


class ShowJuliaSetPointWithDisk(ShowJuliaSetPoint):
    show_disk = True
    n_steps = 11


class AboutFatouDisks(Scene):
    disk_style = {
        "fill_color": YELLOW,
        "fill_opacity": 0.5,
        "stroke_color": YELLOW,
        "stroke_width": 1,
    }

    def construct(self):
        words = Text("(Small enough) disks around points in the Fatou set...")
        words.to_edge(UP)

        disks, arrows = self.get_disks_and_arrow()
        arrows[-1].add(Tex("\\dots").next_to(arrows[-1], RIGHT))
        group = VGroup(disks, arrows)
        group.next_to(words, DOWN)

        shrink_words = Text("...eventually shrink to 0")
        shrink_words.next_to(group, DOWN, aligned_edge=RIGHT)

        self.add(words)
        self.play_disk_progression(disks, arrows)
        self.play(
            FadeIn(arrows[-1]),
            Write(shrink_words, run_time=2)
        )
        self.wait()

    def play_disk_progression(self, disks, arrows):
        self.add(disks[0])
        for d1, d2, arrow in zip(disks, disks[1:], arrows):
            self.play(
                TransformFromCopy(d1.copy().fade(1), d2),
                FadeIn(arrow),
            )

    def get_disks(self):
        radii = [
            *np.linspace(0.5, 1, 3),
            *np.linspace(1, 0, 7)**2 + 0.05,
        ]
        disks = VGroup(*(Circle(radius=r, **self.disk_style) for r in radii))

        for disk in disks:
            disk.add(Dot(disk.get_center(), radius=0.01))

        return disks

    def get_disks_and_arrow(self):
        disks = self.get_disks()
        arrows = Tex("\\rightarrow").replicate(len(disks))
        group = VGroup(*it.chain(*zip(disks, arrows)))
        group.arrange(RIGHT)
        group.set_width(FRAME_WIDTH - 2)
        return disks, arrows


class AboutFatouDisksJustWords(Scene):
    def construct(self):
        words = TexText(
            "(Small enough) disks around points in the Fatou set...\\\\",
            "...eventually shrink to 0"
        )
        words.arrange(DOWN, aligned_edge=LEFT)
        words.to_corner(UL)
        words.set_stroke(BLACK, 6, background=True)

        self.play(FadeIn(words[0], lag_ratio=0.1))
        self.wait()
        self.play(FadeIn(words[1], lag_ratio=0.1))
        self.wait()


class ShowFatouDiskExample(Scene):
    disk_radius = 0.1
    n_steps = 14

    def construct(self):
        c = -1.06 + 0.11j
        plane = ComplexPlane((-3, 3), (-2, 2))
        for line in plane.family_members_with_points():
            line.set_stroke(opacity=0.5 * line.get_stroke_opacity())
        plane.set_height(1.8 * FRAME_HEIGHT)
        plane.add_coordinate_labels(font_size=18)
        fractal = JuliaFractal(plane, parameter=c)

        # z0 = -1.1 + 0.1j
        z0 = -0.3 + 0.2j

        dot = Dot(plane.n2p(z0), radius=0.025)
        dot.set_fill(YELLOW)

        disk = dot.copy()
        disk.set_height(2 * self.disk_radius * plane.get_x_unit_size())
        disk.set_fill(YELLOW, 0.5)
        disk.set_stroke(YELLOW, 1.0)
        disk.insert_n_curves(1000)

        def func(point):
            return plane.n2p(plane.p2n(point)**2 + c)

        self.add(fractal, plane)
        self.add(disk, dot)
        self.play(DrawBorderThenFill(disk))

        path_arc = 10 * DEGREES
        for n in range(self.n_steps):
            point = dot.get_center()
            new_point = func(point)
            arrow = Arrow(point, new_point, path_arc=path_arc, buff=0.1)
            self.play(
                dot.animate.move_to(new_point),
                disk.animate.apply_function(func),
                ShowCreation(arrow),
                path_arc=path_arc,
            )
            self.play(FadeOut(arrow))

        self.embed()


class AboutJuliaDisks(AboutFatouDisks):
    def construct(self):
        words1 = Text("Any tiny disk around a Julia set point...")
        words1.to_edge(UP)
        disks, arrows = self.get_disks_and_arrow()
        group = VGroup(disks, arrows)
        group.next_to(words1, DOWN)
        arrows[-1].add(disks[-1].copy().scale(5).next_to(arrows[-1], RIGHT))

        words2 = Text("...eventually hits every point in the plane,")
        words2.next_to(disks, DOWN, aligned_edge=RIGHT)
        words2.get_part_by_text("every point in the plane").set_color(BLUE)

        words3 = Text("with at most two exceptions.")
        words3.next_to(words2, DOWN, MED_LARGE_BUFF, aligned_edge=RIGHT)

        words4 = TexText(
            "``Stuff goes everywhere'' principle of Julia sets",
            font_size=60
        )
        words4.to_edge(DOWN, LARGE_BUFF)

        VGroup(words1, words2, words3, words4).set_stroke(BLACK, 5, background=True)

        plane = ComplexPlane((-100, 100), (-50, 50))
        plane.scale(2)
        plane.add_coordinate_labels()
        plane.add(BackgroundRectangle(plane, opacity=0.25))
        plane.set_stroke(background=True)
        plane.add_updater(lambda m, dt: m.scale(1 - 0.15 * dt))

        self.add(words1)
        self.play_disk_progression(disks, arrows)
        self.add(plane, *self.mobjects)
        self.play(
            FadeIn(arrows[-1]),
            FadeIn(words2, run_time=2, lag_ratio=0.1),
            VFadeIn(plane, suspend_updating=False),
        )
        self.wait()
        self.play(
            FadeIn(words3, run_time=2, lag_ratio=0.1),
        )
        self.wait(2)
        self.play(Write(words4))
        self.wait(6)
        self.play(VFadeOut(plane, suspend_updating=False))

    def get_disks(self):
        c = -0.18 + 0.77j
        z = -0.491 - 0.106j
        plane = ComplexPlane()
        disk = Circle(radius=0.1, **self.disk_style)
        disk.move_to(plane.n2p(z))
        disk.insert_n_curves(200)

        disks = VGroup(disk)
        for n in range(5):
            new_disk = disks[-1].copy()
            new_disk.apply_complex_function(lambda z: z**2 + c)
            new_disk.make_approximately_smooth()
            disks.add(new_disk)

        for disk in disks:
            disk.center()

        disks.set_height(1)

        return disks


class MontelCorrolaryScreenGrab(ExternallyAnimatedScene):
    pass


class DescribeChaos(Scene):
    def construct(self):
        j_point = 3 * LEFT
        j_value = -0.36554 - 0.29968j

        plane = ComplexPlane((-3, 3), (-2, 2))
        plane.scale(50)
        plane.shift(j_point - plane.n2p(j_value))
        fractal = JuliaFractal(plane)
        fractal.set_c(-0.5 + 0.5j)
        self.add(fractal, plane)

        j_dot = Dot(color=YELLOW, radius=0.05)
        j_dot.move_to(j_point)

        j_label = Text("Julia set point", color=YELLOW)
        j_label.next_to(j_dot, UP, buff=1.0).shift(LEFT)
        j_arrow = Line(j_label.get_bottom(), j_dot.get_center(), buff=0.1)
        j_arrow.set_stroke(width=3)
        j_arrow.set_color(YELLOW)

        surrounding_dots = VGroup(*(
            Dot(radius=0.05).move_to(j_dot.get_center() + buff * vect)
            for n, buff in [(6, 0.2), (12, 0.4)]
            for vect in compass_directions(n)
        ))
        surrounding_dots.set_color(GREY_B)
        # for dot in surrounding_dots:
        #     dot.shift(0.1 * (random.random() - 0.5))
        dots_label = Text("Immediate neighbors")
        dots_label.next_to(surrounding_dots, DOWN)
        dots_label.set_color(GREY_A)
        sublabel = Text("drift far away")
        sublabel.set_color(GREY_A)
        sublabel.next_to(dots_label, DOWN)
        fa = sublabel.get_part_by_text("far away")
        strike = Line(LEFT, RIGHT)
        strike.set_stroke(RED, 10)
        strike.replace(fa, 0)
        new_words = Text("everywhere!")
        new_words.next_to(fa, DOWN)
        new_words.set_color(RED)

        all_words = VGroup(j_label, dots_label, sublabel, new_words)
        all_words.set_stroke(BLACK, 5, background=True)

        arrows = VGroup()
        for dot in surrounding_dots[-12:]:
            point = dot.get_center()
            vect = 0.6 * normalize(point - j_dot.get_center())
            arrows.add(Arrow(point, point + vect, buff=0.1))
        arrows.set_stroke(RED)

        self.add(j_dot)
        self.add(surrounding_dots)
        self.add(j_label, j_arrow)
        self.add(dots_label)

        frame = self.camera.frame
        frame.save_state()
        frame.replace(plane)
        self.play(Restore(frame, run_time=5))
        self.wait()
        self.play(FadeIn(sublabel, lag_ratio=0.1))
        self.wait()
        self.play(
            ShowCreation(strike),
            FadeIn(new_words, shift=0.25 * DOWN)
        )

        self.add(arrows, all_words, strike)
        self.play(LaggedStartMap(ShowCreation, arrows))
        self.wait()


class SimulationOfTinyDisk(RepeatedNewton):
    coefs = [1, 1, 0, 1]
    plane_config = {
        "x_range": (-4, 4),
        "y_range": (-2, 2),
        "height": 12,
        "width": 24,
    }
    colors = ROOT_COLORS_DEEP[::2]
    n_steps = 20

    def construct(self):
        frame = self.camera.frame
        frame.save_state()

        def get_height_ratio():
            return frame.get_height() / FRAME_HEIGHT

        self.add_plane()
        self.add_true_roots()
        self.add_labels()
        self.add_fractal_background()

        fractal = self.fractal
        julia_set = self.fractal_boundary
        fractal.set_n_steps(40)
        julia_set.set_n_steps(40)
        fractal.set_opacity(0.3)
        julia_set.set_opacity(1)
        julia_set.add_updater(lambda m: m.set_julia_highlight(
            get_height_ratio() * 1e-3
        ))
        julia_set.set_opacity(0.5)

        # Generate dots
        point = [1.10049904, 1.38962415, 0.]
        target_height = 0.0006
        cluster_radius = target_height / 50

        dots = self.dots = DotCloud()
        n_radii = 200
        dots.set_points([
            [cluster_radius * r * math.cos(theta), cluster_radius * r * math.sin(theta), 0]
            for r in np.linspace(1, 0, n_radii)
            for theta in np.linspace(0, TAU, int(r * 20)) + random.random() * TAU
        ])
        dots.set_height(0.3)
        dots.set_gloss(0.5)
        dots.set_shadow(0.5)
        dots.set_color(GREY_A)
        dots.move_to(point)
        dots.add_updater(lambda m: m.set_radius(0.05 * get_height_ratio()))

        self.play(
            frame.animate.set_height(target_height).move_to(point),
            run_time=6,
            rate_func=lambda a: smooth(a**0.5),
        )

        self.play(ShowCreation(dots))
        self.wait()
        self.play(Restore(frame, run_time=6, rate_func=lambda a: smooth(a**3)))
        fractal.set_n_steps(12)
        julia_set.set_n_steps(12)

        self.run_iterations()


class SimulationAnnotations(Scene):
    def construct(self):
        dots = self.dots = DotCloud()
        n_radii = 200
        cluster_radius = 0.5
        dots.set_points([
            [cluster_radius * r * math.cos(theta), cluster_radius * r * math.sin(theta), 0]
            for r in np.linspace(1, 0, n_radii)
            for theta in np.linspace(0, TAU, int(r * 20)) + random.random() * TAU
        ])

        n_points_label = TexText("$\\sim 2{,}000$ points")
        n_points_label.next_to(dots, UP)

        brace = Brace(
            # Line(dots.get_bottom(), dots.get_corner(DR)),
            Line(dots.get_center(), dots.get_right()),
            DOWN, buff=0
        )
        brace_tex = Tex("\\text{Radius } \\approx 1 / 1{,}000{,}000")
        brace_tex.next_to(dots, DOWN)

        group = VGroup(n_points_label, brace, brace_tex)
        group.set_stroke(BLACK, 5, background=True)

        self.play(Write(n_points_label))
        self.play(
            GrowFromCenter(brace),
            FadeIn(brace_tex, DOWN)
        )
        self.wait()
        self.play(FadeOut(group))


class LattesExample(TeacherStudentsScene):
    def construct(self):
        example = VGroup(
            TexText("Lattè's example: "),
            Tex(r"L(z)=\frac{\left(z^{2}+1\right)^{2}}{4 z\left(z^{2}-1\right)}"),
        )
        example.arrange(RIGHT)
        example[0].shift(SMALL_BUFF * DOWN)
        example.move_to(self.hold_up_spot, DOWN)
        example.set_x(0)

        j_fact = TexText("Julia set of $L(z)$ is all of $\\mathds{C}$")
        j_fact.move_to(example)
        subwords = TexText("(and the point at $\\infty$)", font_size=36)
        subwords.set_fill(GREY_A)
        subwords.next_to(j_fact, DOWN)

        self.play(
            self.teacher.change("raise_right_hand", 3 * UR),
            self.change_students(
                "pondering", "happy", "tease",
                look_at=3 * UR
            )
        )
        self.wait(3)
        self.play(
            self.teacher.change("sassy", example),
            Write(example)
        )
        self.play_student_changes(
            "pondering", "pondering", "pondering",
            look_at=example,
        )

        self.play(
            example.animate.to_edge(UP),
            FadeIn(j_fact),
            self.change_students(
                "erm", "erm", "erm",
                look_at=j_fact,
            ),
            self.teacher.change("raise_right_hand", j_fact)
        )
        self.play(FadeIn(subwords))
        self.wait(3)


class JFunctionMention(Scene):
    def construct(self):
        image = ImageMobject("j_invariant")
        image.set_height(5)
        name = TexText("Klein's $j$ function")
        name.next_to(image, UP)
        words = Text("A whole story...")
        words.next_to(image, RIGHT)

        self.play(
            FadeIn(image),
            Write(name)
        )
        self.wait()
        self.play(Write(words))
        self.wait()


class LinksBelow(TeacherStudentsScene):
    def construct(self):
        self.pi_creatures.flip().flip()
        self.teacher_says("Links below")
        self.play_student_changes(
            "pondering", "thinking", "pondering",
            look_at=FRAME_HEIGHT * DOWN,
        )
        self.wait(2)
        self.play(self.teacher.change("happy"))
        self.wait(4)


class MoreAmbientChaos(TwoToMillionPoints):
    def construct(self):
        plane = ComplexPlane((-2, 2), (-2, 2))
        plane.set_height(7)
        self.add(plane)

        point = self.get_julia_set_points(plane, n_points=1, n_steps=1000)[0]
        epsilon = 1e-6
        dots = DotCloud([
            point + epsilon * rotate_vector(RIGHT, random.random() * TAU)
            for x in range(10)
        ])
        dots.set_gloss(0.5)
        dots.set_shadow(0.5)
        dots.set_radius(0.075)
        dots.set_color(YELLOW)

        def func(p):
            z = plane.p2n(p)
            return plane.n2p(z**2 + self.c)

        n_steps = 100

        for n in range(n_steps):
            points = dots.get_points()

            values = list(map(plane.p2n, points))
            new_values = np.array(list(map(lambda z: z**2 + self.c, values)))

            new_points = list(map(plane.n2p, new_values))

            nn_points = []
            for p in new_points:
                if p[0] < plane.get_left()[0]:
                    p[0] += 2 * (plane.get_left()[0] - p[0])
                if p[0] > plane.get_right()[0]:
                    p[0] -= 2 * (p[0] - plane.get_right()[0])
                if p[1] < plane.get_bottom()[1]:
                    p[1] += 2 * (plane.get_bottom()[1] - p[1])
                if p[1] > plane.get_top()[1]:
                    p[1] -= 2 * (p[1] - plane.get_top()[1])
                nn_points.append(p)
            new_points = nn_points

            lines = VGroup(*(
                Line(p1, p2)
                for p1, p2 in zip(points, new_points)
            ))
            lines.set_stroke(WHITE, 1)

            self.play(
                dots.animate.set_points(new_points),
                ShowCreation(lines, lag_ratio=0),
            )
            self.add(lines[0].copy().set_opacity(0.25))
            for line in lines:
                line.rotate(PI)
            self.play(FadeOut(lines))


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
            # z0=self.z0,
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
        frame.save_state()
        frame.generate_target()
        frame.target.move_to(point1)
        frame.target.set_height(height1)

        fractal.set_saturation_factor(2)
        plane.remove(plane.coordinate_labels)

        self.play(
            MoveToTarget(frame),
            run_time=8,
            rate_func=bezier([0, 0, 1, 1])
        )
        self.play(
            fractal.animate.set_saturation_factor(4),
            run_time=3
        )
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
        self.wait(2)
        self.play(
            Restore(frame),
            fractal.animate.set_saturation_factor(0),
            run_time=7
        )
        self.wait()


class Part1EndScroll(PatreonEndScreen):
    CONFIG = {
        # "title_text": "",
        "scroll_time": 30,
        # "show_pis": False,
    }


class AmbientJulia(Scene):
    def construct(self):
        plane = ComplexPlane(
            (-4, 4), (-2, 2),
            background_line_style={
                "stroke_color": GREY_A,
                "stroke_width": 1,
            }
        )
        plane.axes.set_stroke(width=1, opacity=0.5)
        plane.set_height(14)
        fractal = JuliaFractal(plane)
        fractal.set_n_steps(100)

        R = 0.25
        cardioid = ParametricCurve(
            lambda t: plane.c2p(
                2 * R * math.cos(t) - R * math.cos(2 * t),
                2 * R * math.sin(t) - R * math.sin(2 * t),
            ),
            t_range=(0, TAU)
        )

        t_tracker = ValueTracker(0)
        get_t = t_tracker.get_value

        fractal.add_updater(lambda m: m.set_c(
            plane.p2n(cardioid.pfp(get_t()))
        ))

        self.add(fractal, plane)
        self.play(
            t_tracker.animate.set_value(1),
            rate_func=linear,
            run_time=300
        )
