from manim_imports_ext import *


def get_matrix_exponential(matrix, height=1.5, scalar_tex="t", **matrix_config):
    elem = matrix[0][0]
    if isinstance(elem, str):
        mat_class = Matrix
    elif isinstance(elem, int) or isinstance(elem, np.int64):
        mat_class = IntegerMatrix
    else:
        mat_class = DecimalMatrix

    matrix = mat_class(matrix, **matrix_config)
    base = Tex("e")
    base.set_height(0.4 * height)
    matrix.set_height(0.6 * height)
    matrix.move_to(base.get_corner(UR), DL)
    result = VGroup(base, matrix)
    if scalar_tex:
        scalar = Tex(scalar_tex)
        scalar.set_height(0.7 * base.get_height())
        scalar.next_to(matrix, RIGHT, buff=SMALL_BUFF, aligned_edge=DOWN)
        result.add(scalar)
    return result


def get_vector_field_and_stream_lines(func, coordinate_system,
                                      magnitude_range=(0.5, 4),
                                      vector_opacity=0.75,
                                      vector_thickness=0.03,
                                      color_by_magnitude=False,
                                      line_color=GREY_A,
                                      line_width=3,
                                      line_opacity=0.75,
                                      sample_freq=5,
                                      n_samples_per_line=10,
                                      arc_len=3,
                                      time_width=0.3,
                                      ):
    vector_field = VectorField(
        func, coordinate_system,
        magnitude_range=magnitude_range,
        vector_config={
            "fill_opacity": vector_opacity,
            "thickness": vector_thickness,
        }
    )
    stream_lines = StreamLines(
        func, coordinate_system,
        step_multiple=1.0 / sample_freq,
        n_samples_per_line=n_samples_per_line,
        arc_len=arc_len,
        magnitude_range=magnitude_range,
        color_by_magnitude=color_by_magnitude,
        stroke_color=line_color,
        stroke_width=line_width,
        stroke_opacity=line_opacity,
    )
    animated_lines = AnimatedStreamLines(
        stream_lines,
        line_anim_config={
            "time_width": time_width,
        },
    )

    return vector_field, animated_lines


def mat_exp(matrix, N=100):
    curr = np.identity(len(matrix))
    curr_sum = curr
    for n in range(1, N):
        curr = np.dot(curr, matrix) / n
        curr_sum += curr
    return curr_sum


def get_1d_equation(r="r"):
    return Tex(
        "{d \\over d{t}} x({t}) = {" + r + "} \\cdot x({t})",
        tex_to_color_map={
            "{t}": GREY_B,
            "{" + r + "}": BLUE,
            "=": WHITE,
        }
    )


def get_2d_equation(matrix=[["a", "b"], ["c", "d"]]):
    deriv = Tex("d \\over dt", tex_to_color_map={"t": GREY_B})
    vect = Matrix(
        [["x(t)"], ["y(t)"]],
        bracket_h_buff=SMALL_BUFF,
        bracket_v_buff=SMALL_BUFF,
        element_to_mobject_config={
            "tex_to_color_map": {"t": GREY_B},
            "isolate": ["(", ")"]
        }
    )
    deriv.match_height(vect)
    equals = Tex("=")
    matrix_mob = Matrix(matrix, h_buff=0.8)
    matrix_mob.set_color(TEAL)
    matrix_mob.match_height(vect)
    equation = VGroup(deriv, vect, equals, matrix_mob, vect.deepcopy())
    equation.arrange(RIGHT)
    return equation


class VideoWrapper(Scene):
    title = ""

    def construct(self):
        self.add(FullScreenRectangle())
        screen_rect = ScreenRectangle(height=6)
        screen_rect.set_stroke(BLUE_D, 1)
        screen_rect.set_fill(BLACK, 1)
        screen_rect.to_edge(DOWN)
        self.add(screen_rect)

        title = TexText(self.title, font_size=90)
        if title.get_width() > screen_rect.get_width():
            title.set_width(screen_rect.get_width())
        title.next_to(screen_rect, UP)

        self.play(Write(title))
        self.wait()


# Video scenes

class ArnoldBookClip(ExternallyAnimatedScene):
    pass


class ZoomInOnProblem(Scene):
    def construct(self):
        # Highlight problem
        image = ImageMobject("mat_exp_exercise.png")
        image.set_height(FRAME_HEIGHT)

        prob_rect = Rectangle(3.25, 0.35)
        prob_rect.move_to([-2.5, -2.5, 0])
        prob_rect.set_stroke(BLUE, 2)

        examples_rect = Rectangle(2.0, 0.8)
        examples_rect.move_to([01.8, 2.8, 0.0])
        examples_rect.set_stroke(YELLOW, 3)
        answer_rect = Rectangle(3.0, 2.0)
        answer_rect.move_to(examples_rect, UR)
        answer_rect.shift(0.1 * RIGHT)

        full_rect = FullScreenRectangle()
        full_rect.flip()
        full_rect.set_fill(BLACK, opacity=0.6)
        full_rect.append_vectorized_mobject(prob_rect)
        full_rect2 = full_rect.copy()
        full_rect3 = full_rect.copy()
        full_rect.append_vectorized_mobject(examples_rect.copy().scale(1e-6))
        full_rect2.append_vectorized_mobject(examples_rect)
        full_rect3.append_vectorized_mobject(answer_rect)

        self.add(image)

        # Write problem
        problem = TexText(
            "Compute the {{matrix}} {{$e^{At}$}}\\\\if the {{matrix A}} has the form..."
        )
        problem.to_corner(UL)
        problem.set_stroke(BLACK, 5, background=True)
        prob_arrow = Arrow(prob_rect, problem)
        prob_arrow.set_fill(BLUE)
        mat_underline = Underline(problem.get_part_by_tex("matrix"))
        mat_underline.set_color(YELLOW)

        self.play(
            ShowCreation(prob_rect),
            FadeIn(full_rect),
            FadeTransform(prob_rect.copy(), problem),
            GrowArrow(prob_arrow),
        )
        self.wait()
        for part in ["e^{At}", "matrix A"]:
            self.play(FlashAround(problem.get_part_by_tex(part), run_time=2, time_width=4))
            self.wait()
        self.play(ShowCreation(mat_underline))
        self.wait()
        mat_underline.rotate(PI)
        self.play(Uncreate(mat_underline))
        self.wait()

        # Show inputs
        examples_label = TexText("Various matrices to\\\\plug in for $A$", font_size=40)
        examples_label.next_to(examples_rect)

        lhs = Tex("A = ", font_size=96)
        lhs.set_stroke(BLACK, 5, background=True)

        matrices = VGroup(
            IntegerMatrix([[1, 0], [0, 2]]),
            IntegerMatrix([[0, 1], [0, 0]]),
            IntegerMatrix([[0, 1], [-1, 0]]),
            IntegerMatrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
        )
        eq = VGroup(lhs, matrices[-1])
        eq.arrange(RIGHT)
        eq.next_to(examples_rect, DOWN, LARGE_BUFF, aligned_edge=LEFT)
        eq.shift_onto_screen()
        for matrix in matrices:
            matrix.move_to(matrices[-1], LEFT)
            matrix.set_stroke(BLACK, 5, background=True)

        self.play(
            Transform(full_rect, full_rect2)
        )
        self.play(
            ShowCreation(examples_rect),
            FadeIn(examples_label),
            FadeIn(lhs),
            FadeIn(matrices[0]),
        )
        self.wait()
        for m1, m2 in zip(matrices, matrices[1:]):
            self.play(FadeTransform(m1, m2))
            self.wait()


class LeadToPhysicsAndQM(Scene):
    def construct(self):
        de_words = TexText("Differential\\\\equations", font_size=60)
        de_words.set_x(-3).to_edge(UP)
        mat_exp = get_matrix_exponential([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
        mat_exp[1].set_color(TEAL)
        mat_exp.next_to(de_words, DOWN, buff=3)

        qm_words = TexText("Quantum\\\\mechanics", font_size=60)
        qm_words.set_x(3).to_edge(UP)
        physics_words = TexText("Physics", font_size=60)
        physics_words.move_to(qm_words)

        qm_exp = Tex("e^{-i \\hat{H} t / \\hbar}")
        qm_exp.scale(2)
        qm_exp.refresh_bounding_box()
        qm_exp[0][0].set_height(mat_exp[0].get_height(), about_edge=UR)
        qm_exp[0][0].shift(SMALL_BUFF * DOWN)
        qm_exp.match_x(qm_words)
        qm_exp.align_to(mat_exp, DOWN)
        qm_exp[0][3:5].set_color(TEAL)

        de_arrow = Arrow(de_words, mat_exp)
        qm_arrow = Arrow(qm_words, qm_exp)
        top_arrow = Arrow(de_words, qm_words)

        self.add(de_words)
        self.play(
            GrowArrow(de_arrow),
            FadeIn(mat_exp, shift=DOWN),
        )
        self.wait()
        self.play(
            GrowArrow(top_arrow),
            FadeIn(physics_words, RIGHT)
        )
        self.wait()
        self.play(
            FadeOut(physics_words, UP),
            FadeIn(qm_words, UP),
        )
        self.play(
            TransformFromCopy(de_arrow, qm_arrow),
            FadeTransform(mat_exp.copy(), qm_exp),
        )
        self.wait()


class LaterWrapper(VideoWrapper):
    title = "Later..."


class PlanForThisVideo(TeacherStudentsScene):
    def construct(self):
        s0, s1, s2 = self.students
        self.play(
            PiCreatureSays(s0, TexText("But what\\\\is $e^{M}$?"), target_mode="raise_left_hand"),
            s1.change("erm", UL),
            s2.change("pondering", UL),
        )
        self.wait()
        s0.bubble = None
        self.play(
            PiCreatureSays(
                s2, TexText("And who cares?"), target_mode="sassy",
                bubble_config={"direction": LEFT},
            ),
            s1.change("hesitant", UL),
            self.teacher.change("guilty")
        )
        self.wait(3)


class IntroduceTheComputation(Scene):
    def construct(self):
        # Matrix in exponent
        base = Tex("e")
        base.set_height(1.0)
        matrix = IntegerMatrix(
            [[3, 1, 4],
             [1, 5, 9],
             [2, 6, 5]],
        )
        matrix.move_to(base.get_corner(UR), DL)
        matrix_exp = VGroup(base, matrix)
        matrix_exp.set_height(2)
        matrix_exp.to_corner(UL)
        matrix_exp.shift(3 * RIGHT)

        randy = Randolph()
        randy.set_height(2)
        randy.to_corner(DL)

        matrix.save_state()
        matrix.center()
        matrix.set_height(2.5)

        self.add(randy)
        self.play(
            randy.change("pondering", matrix),
            Write(matrix.get_brackets()),
            ShowIncreasingSubsets(matrix.get_entries()),
        )
        self.play(
            matrix.animate.restore(),
            Write(base),
            randy.change("erm", base),
        )
        self.play(Blink(randy))

        # Question the repeated multiplication implication
        rhs = Tex("= e \\cdot e \\dots e \\cdot e")
        rhs.set_height(0.75 * base.get_height())
        rhs.next_to(matrix_exp, RIGHT)
        rhs.align_to(base, DOWN)
        brace = Brace(rhs[0][1:], DOWN)
        matrix_copy = matrix.copy()
        matrix_copy.scale(0.5)
        brace_label = VGroup(
            matrix.copy().scale(0.5),
            Text("times?")
        )
        brace_label.arrange(RIGHT)
        brace_label.next_to(brace, DOWN, SMALL_BUFF)

        bubble = randy.get_bubble(
            TexText("I'm sorry,\\\\what?!").scale(0.75),
            height=2,
            width=3,
            bubble_type=SpeechBubble,
        )

        self.play(
            TransformMatchingParts(
                base.copy(), rhs,
                path_arc=10 * DEGREES,
                lag_ratio=0.01,
            ),
            GrowFromCenter(brace),
            ReplacementTransform(
                matrix.copy(), brace_label[0],
                path_arc=30 * DEGREES,
                run_time=2,
                rate_func=squish_rate_func(smooth, 0.3, 1),
            ),
            Write(
                brace_label[1],
                run_time=2,
                rate_func=squish_rate_func(smooth, 0.5, 1),
            ),
            randy.change("angry", rhs),
            ShowCreation(bubble),
            Write(bubble.content, run_time=1),
        )
        self.wait()

        false_equation = VGroup(
            matrix_exp, rhs, brace, brace_label
        )

        # This is nonsense.
        morty = Mortimer()
        morty.refresh_triangulation()
        morty.match_height(randy)
        morty.to_corner(DR)
        morty.set_opacity(0)
        false_equation.generate_target()
        false_equation.target.scale(0.5)
        false_equation.target.next_to(morty, UL)
        fe_rect = SurroundingRectangle(false_equation.target)
        fe_rect.set_color(GREY_BROWN)
        cross = Cross(false_equation.target[1])
        cross.insert_n_curves(1)
        cross.set_stroke(RED, width=[1, 5, 1])
        nonsense = Text("This would be nonsense")
        nonsense.match_width(fe_rect)
        nonsense.next_to(fe_rect, UP)
        nonsense.set_color(RED)

        randy.bubble = bubble
        self.play(
            MoveToTarget(false_equation),
            RemovePiCreatureBubble(randy, target_mode="hesitant"),
            morty.animate.set_opacity(1).change("raise_right_hand"),
            ShowCreation(fe_rect),
        )
        self.play(
            ShowCreation(cross),
            FadeIn(nonsense),
        )
        self.play(Blink(morty))
        self.wait()

        false_group = VGroup(false_equation, fe_rect, cross, nonsense)

        # Show Taylor series
        real_equation = Tex(
            "e^x = x^0 + x^1 + \\frac{1}{2} x^2 + \\frac{1}{6} x^3 + \\cdots + \\frac{1}{n!} x^n + \\cdots",
            isolate=["x"]
        )
        xs = real_equation.get_parts_by_tex("x")
        xs.set_color(YELLOW)
        real_equation.set_width(FRAME_WIDTH - 2.0)
        real_equation.to_edge(UP)
        real_rhs = real_equation[3:]

        real_label = Text("Real number", color=YELLOW, font_size=24)
        # real_label.next_to(xs[0], DOWN, buff=0.8)
        # real_label.to_edge(LEFT, buff=MED_SMALL_BUFF)
        # real_arrow = Arrow(real_label, xs[0], buff=0.1, fill_color=GREY_B, thickness=0.025)
        real_label.to_corner(UL, buff=MED_SMALL_BUFF)
        real_arrow = Arrow(real_label, real_equation[1], buff=0.1)

        taylor_brace = Brace(real_rhs, DOWN)
        taylor_label = taylor_brace.get_text("Taylor series")

        self.play(
            TransformFromCopy(base, real_equation[0]),
            FadeTransform(matrix.copy(), real_equation[1]),
            FadeIn(real_label, UR),
            GrowArrow(real_arrow),
            randy.change("thinking", real_label),
            morty.animate.look_at(real_label),
        )
        self.wait()
        self.play(
            Write(real_equation[2], lag_ratio=0.2),
            FadeTransformPieces(xs[:1].copy(), xs[1:], path_arc=20 * DEGREES),
            LaggedStart(*(
                FadeIn(part)
                for part in real_equation[4:]
                if part not in xs
            )),
            randy.change("pondering", real_equation),
            morty.change("pondering", real_equation),
        )
        self.add(real_equation)
        self.play(Blink(morty))
        self.play(
            false_group.animate.scale(0.7).to_edge(DOWN),
            GrowFromCenter(taylor_brace),
            FadeIn(taylor_label, 0.5 * DOWN)
        )
        self.wait()

        # Taylor series example
        ex_rhs = Tex(
            """
            {2}^0 +
            {2}^1 +
            { {2}^2 \\over 2} +
            { {2}^3 \\over 6} +
            { {2}^4 \\over 24} +
            { {2}^5 \\over 120} +
            { {2}^6 \\over 720} +
            { {2}^7 \\over 5040} +
            \\cdots
            """,
            tex_to_color_map={"{2}": YELLOW, "+": WHITE},
        )
        ex_rhs.next_to(real_equation[3:], DOWN, buff=0.75)

        ex_parts = VGroup(*(
            ex_rhs[i:j] for i, j in [
                (0, 2),
                (3, 5),
                (6, 8),
                (9, 11),
                (12, 14),
                (15, 17),
                (18, 20),
                (21, 23),
                (24, 25),
            ]
        ))
        term_brace = Brace(ex_parts[0], DOWN)
        frac = Tex("1", font_size=36)
        frac.next_to(term_brace, DOWN, SMALL_BUFF)

        rects = VGroup(*(
            Rectangle(height=2**n / math.factorial(n), width=1)
            for n in range(11)
        ))
        rects.arrange(RIGHT, buff=0, aligned_edge=DOWN)
        rects.set_fill(opacity=1)
        rects.set_submobject_colors_by_gradient(BLUE, GREEN)
        rects.set_stroke(WHITE, 1)
        rects.set_width(7)
        rects.to_edge(DOWN)

        self.play(
            ReplacementTransform(taylor_brace, term_brace),
            FadeTransform(real_equation[3:].copy(), ex_rhs),
            FadeOut(false_group, shift=DOWN),
            FadeOut(taylor_label, shift=DOWN),
            FadeIn(frac),
        )
        term_values = VGroup()
        for n in range(11):
            rect = rects[n]
            fact = math.factorial(n)
            ex_part = ex_parts[min(n, len(ex_parts) - 1)]
            value = DecimalNumber(2**n / fact)
            value.set_color(GREY_A)
            max_width = 0.6 * rect.get_width()
            if value.get_width() > max_width:
                value.set_width(max_width)
            value.next_to(rects[n], UP, SMALL_BUFF)
            new_brace = Brace(ex_part, DOWN)
            if fact == 1:
                new_frac = Tex(f"{2**n}", font_size=36)
            else:
                new_frac = Tex(f"{2**n} / {fact}", font_size=36)
            new_frac.next_to(new_brace, DOWN, SMALL_BUFF)
            self.play(
                term_brace.animate.become(new_brace),
                FadeTransform(frac, new_frac),
            )
            frac = new_frac
            rect.save_state()
            rect.stretch(0, 1, about_edge=DOWN)
            rect.set_opacity(0)
            value.set_value(0)
            self.play(
                Restore(rect),
                ChangeDecimalToValue(value, 2**n / math.factorial(n)),
                UpdateFromAlphaFunc(value, lambda m, a: m.next_to(rect, UP, SMALL_BUFF).set_opacity(a)),
                randy.animate.look_at(rect),
                morty.animate.look_at(rect),
            )
            term_values.add(value)
        self.play(FadeOut(frac))

        new_brace = Brace(ex_rhs, DOWN)
        sum_value = DecimalNumber(math.exp(2), num_decimal_places=4, font_size=36)
        sum_value.next_to(new_brace, DOWN)
        self.play(
            term_brace.animate.become(new_brace),
            randy.change("thinking", sum_value),
            morty.change("tease", sum_value),
            *(FadeTransform(dec.copy().set_opacity(0), sum_value) for dec in term_values)
        )
        self.play(Blink(randy))

        lhs = Tex("e \\cdot e =")
        lhs.match_height(real_equation[0])
        lhs.next_to(ex_rhs, LEFT)
        self.play(Write(lhs))
        self.play(Blink(morty))
        self.play(Blink(randy))

        # Increment input
        twos = ex_rhs.get_parts_by_tex("{2}")
        threes = VGroup(*(
            Tex("3").set_color(YELLOW).replace(two)
            for two in twos
        ))
        new_lhs = Tex("e \\cdot e \\cdot e = ")
        new_lhs.match_height(lhs)
        new_lhs[0].space_out_submobjects(0.8)
        new_lhs[0][-1].shift(SMALL_BUFF * RIGHT)
        new_lhs.move_to(lhs, RIGHT)

        anims = []
        unit_height = 0.7 * rects[0].get_height()
        for n, rect, value_mob in zip(it.count(0), rects, term_values):
            rect.generate_target()
            new_value = 3**n / math.factorial(n)
            rect.target.set_height(unit_height * new_value, stretch=True, about_edge=DOWN)
            value_mob.rect = rect
            anims += [
                MoveToTarget(rect),
                ChangeDecimalToValue(value_mob, new_value),
                UpdateFromFunc(value_mob, lambda m: m.next_to(m.rect, UP, SMALL_BUFF))
            ]

        self.play(
            FadeOut(twos, 0.5 * UP),
            FadeIn(threes, 0.5 * UP),
        )
        twos.set_opacity(0)
        self.play(
            ChangeDecimalToValue(sum_value, math.exp(3)),
            *anims,
        )
        self.play(
            FadeOut(lhs, 0.5 * UP),
            FadeIn(new_lhs, 0.5 * UP),
        )
        self.wait()

        # Isolate polynomial
        real_lhs = VGroup(real_equation[:3], real_label, real_arrow)

        self.play(
            LaggedStartMap(FadeOut, VGroup(
                *new_lhs, *threes, *ex_rhs,
                term_brace, sum_value,
                *rects, *term_values,
            )),
            real_lhs.animate.set_opacity(0.2),
            randy.change("erm", real_equation),
            morty.change("thinking", real_equation),
            run_time=1,
        )
        self.play(Blink(morty))

        # Alternate inputs
        rhs_tex = "X^0 + X^1 + \\frac{1}{2} X^2 + \\frac{1}{6} X^3 + \\cdots + \\frac{1}{n!} X^n + \\cdots"
        pii_rhs = Tex(
            rhs_tex.replace("X", "(\\pi i)"),
            tex_to_color_map={"(\\pi i)": BLUE},
        )
        pii_rhs.match_width(real_rhs)

        mat_tex = "\\left[ \\begin{array}{ccc} 3 & 1 & 4 \\\\ 1 & 5 & 9 \\\\ 2 & 6 & 5 \\end{array} \\right]"
        mat_rhs = Tex(
            rhs_tex.replace("X", mat_tex),
            tex_to_color_map={mat_tex: TEAL},
        )
        mat_rhs.scale(0.5)

        pii_rhs.next_to(real_rhs, DOWN, buff=0.7)
        mat_rhs.next_to(pii_rhs, DOWN, buff=0.7)

        self.play(FlashAround(real_rhs))
        self.wait()
        self.play(
            morty.change("raise_right_hand", pii_rhs),
            FadeTransformPieces(real_rhs.copy(), pii_rhs),
        )
        self.play(Blink(randy))
        self.play(
            FadeTransformPieces(real_rhs.copy(), mat_rhs),
        )
        self.play(
            randy.change("maybe", mat_rhs),
        )
        self.wait()

        why = Text("Why?", font_size=36)
        why.next_to(randy, UP, aligned_edge=LEFT)
        self.play(
            randy.change("confused", mat_rhs.get_corner(UL)),
            Write(why),
        )
        self.play(Blink(randy))

        reassurance = VGroup(
            Text("I know it looks complicated.", font_size=24),
            Text("Don't panic.", font_size=24),
        )
        reassurance.arrange(DOWN)
        reassurance.next_to(morty, LEFT, aligned_edge=UP)
        reassurance.set_color(GREY_A)

        for words in reassurance:
            self.play(FadeIn(words))
        self.play(Blink(morty))

        # Describe exp
        to_right_group = VGroup(real_lhs, real_rhs, mat_rhs, pii_rhs)
        to_right_group.generate_target()
        to_right_group.target.to_edge(RIGHT, buff=SMALL_BUFF)
        to_right_group.target[2].to_edge(RIGHT, buff=SMALL_BUFF)
        self.play(
            MoveToTarget(to_right_group),
            FadeOut(why),
            FadeOut(reassurance),
            randy.change("pondering", mat_rhs),
            morty.change("tease"),
        )

        pii_lhs = Tex("\\text{exp}\\left(\\pi i \\right) = ")[0]
        pii_lhs.next_to(pii_rhs, LEFT)
        mat_lhs = Tex("\\text{exp}\\left(" + mat_tex + "\\right) = ")[0]
        mat_lhs.match_height(mat_rhs)
        mat_lhs[:3].match_height(pii_lhs[:3])
        mat_lhs[:3].next_to(mat_lhs[3:5], LEFT, SMALL_BUFF)
        mat_lhs.next_to(mat_rhs, LEFT)

        pii_lhs_pi_part = pii_lhs[4:6]
        pii_lhs_pi_part.set_color(BLUE)
        mat_lhs_mat_part = mat_lhs[5:18]
        mat_lhs_mat_part.set_color(TEAL)

        self.play(
            FadeIn(pii_lhs),
            randy.change("thinking", pii_lhs),
            randy.change("tease", pii_lhs),
        )
        self.play(FadeIn(mat_lhs))
        self.play(Blink(randy))
        self.play(
            LaggedStart(
                FlashAround(pii_lhs[:3]),
                FlashAround(mat_lhs[:3]),
                lag_ratio=0.3,
                run_time=2
            ),
            randy.change("raise_left_hand", pii_lhs),
        )
        self.wait()

        # Transition to e^x notation
        crosses = VGroup(*(
            Cross(lhs[:3], stroke_width=[0, 3, 3, 3, 0]).scale(1.3)
            for lhs in [pii_lhs, mat_lhs]
        ))
        bases = VGroup()
        powers = VGroup()
        equals = VGroup()
        for part, lhs in (pii_lhs_pi_part, pii_lhs), (mat_lhs_mat_part, mat_lhs):
            power = part.copy()
            part.set_opacity(0)
            self.add(power)
            base = Tex("e", font_size=60)
            equal = Tex(":=")
            power.generate_target()
            if power.target.get_height() > 0.7:
                power.target.set_height(0.7)
            power.target.next_to(base, UR, buff=0.05)
            group = VGroup(base, power.target, equal)
            equal.next_to(group[:2], RIGHT, MED_SMALL_BUFF)
            equal.match_y(base)
            if lhs is mat_lhs:
                equal.shift(0.1 * UP)
            group.shift(lhs.get_right() - equal.get_right())
            bases.add(base)
            powers.add(power)
            equals.add(equal)

        self.play(
            ShowCreation(crosses),
            randy.change("hesitant", crosses),
        )
        self.play(Blink(randy))
        self.play(real_lhs.animate.set_opacity(1))

        self.play(
            FadeOut(pii_lhs),
            FadeOut(mat_lhs),
            FadeOut(crosses),
            *(MoveToTarget(power) for power in powers),
            *(TransformFromCopy(real_equation[0], base) for base in bases),
            Write(equals),
            randy.change("sassy", powers),
        )
        self.wait()

        # Theorem vs. definition
        real_part = VGroup(real_lhs, real_rhs)
        pii_part = VGroup(bases[0], powers[0], equals[0], pii_rhs)
        mat_part = VGroup(bases[1], powers[1], equals[1], mat_rhs)
        def_parts = VGroup(pii_part, mat_part)

        self.play(
            FadeOut(randy, DOWN),
            FadeOut(morty, DOWN),
            real_part.animate.set_x(0).shift(DOWN),
            def_parts.animate.set_x(0).to_edge(DOWN),
        )

        real_rect = SurroundingRectangle(real_part)
        real_rect.set_stroke(YELLOW, 2)
        theorem_label = Text("Theorem")
        theorem_label.next_to(real_rect, UP)

        def_rect = SurroundingRectangle(def_parts)
        def_rect.set_stroke(BLUE, 2)
        def_label = Text("Definition")
        def_label.next_to(def_rect, UP)

        self.play(
            ShowCreation(real_rect),
            FadeIn(theorem_label, 0.5 * UP),
        )
        self.wait()
        self.play(
            ShowCreation(def_rect),
            FadeIn(def_label, 0.5 * UP),
        )
        self.wait()

        # Abuse?  Or the beauty of discovery...
        randy2 = Randolph()
        randy2.set_height(1.5)
        randy2.next_to(def_rect, UP, SMALL_BUFF, aligned_edge=LEFT)

        self.play(
            ApplyMethod(randy2.change, "angry", mat_rhs),
            UpdateFromAlphaFunc(randy2, lambda m, a: m.set_opacity(a)),
        )
        self.play(Blink(randy2))

        discovery_label = Text("Discovery")
        discovery_label.move_to(theorem_label, DOWN)
        invention_label = Text("Invention")
        invention_label.move_to(def_label, DOWN)

        self.play(
            FadeOut(theorem_label, UP),
            FadeIn(discovery_label, UP),
            randy2.change("hesitant", theorem_label),
        )
        self.play(
            FadeOut(def_label, UP),
            FadeIn(invention_label, UP),
        )
        self.wait(2)

        # Isolate matrix right hand side
        to_fade = VGroup(
            randy2, discovery_label, invention_label,
            real_rect, def_rect,
            real_arrow, real_label, real_part, pii_part,
            bases, powers, equals,
        )

        self.play(
            mat_rhs.animate.set_width(FRAME_WIDTH - 1).center().to_edge(UP),
            LaggedStartMap(FadeOut, to_fade),
            FadeIn(VGroup(randy, morty), run_time=2, rate_func=squish_rate_func(smooth, 0.5, 1))
        )

        # Matrix powers
        mat = mat_rhs[4]
        mat_brace = Brace(VGroup(mat, mat_rhs[5][0]), DOWN, buff=SMALL_BUFF)
        matrix = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
        matrix_square = np.dot(matrix, matrix)
        result = IntegerMatrix(matrix_square, h_buff=1.3, v_buff=0.7)
        result.match_height(mat)
        square_eq = VGroup(mat.copy(), mat.copy(), Tex("="), result)
        square_eq.arrange(RIGHT, buff=SMALL_BUFF)
        square_eq.next_to(mat_brace, DOWN)

        self.play(GrowFromCenter(mat_brace))
        self.play(
            LaggedStart(
                TransformFromCopy(mat, square_eq[0], path_arc=45 * DEGREES),
                TransformFromCopy(mat, square_eq[1]),
                Write(square_eq[2]),
                Write(result.brackets),
            ),
            randy.change("pondering", square_eq),
        )
        self.show_mat_mult(matrix, matrix, square_eq[0][2:11], square_eq[1][2:11], result.elements)

        # Show matrix cubed
        mat_brace.generate_target()
        mat_brace.target.next_to(mat_rhs[6], DOWN, SMALL_BUFF)

        mat_squared = result
        mat_cubed = IntegerMatrix(
            np.dot(matrix, matrix_square),
            h_buff=1.8, v_buff=0.7,
            element_alignment_corner=ORIGIN,
        )
        mat_cubed.match_height(mat)
        cube_eq = VGroup(
            VGroup(mat.copy(), mat.copy(), mat.copy()).arrange(RIGHT, buff=SMALL_BUFF),
            Tex("=").rotate(90 * DEGREES),
            VGroup(mat.copy(), mat_squared.deepcopy()).arrange(RIGHT, buff=SMALL_BUFF),
            Tex("=").rotate(90 * DEGREES),
            mat_cubed
        )
        cube_eq.arrange(DOWN)
        cube_eq.next_to(mat_brace.target, DOWN)

        self.play(
            MoveToTarget(mat_brace),
            ReplacementTransform(square_eq[0], cube_eq[0][1]),
            ReplacementTransform(square_eq[1], cube_eq[0][2]),
            ReplacementTransform(square_eq[2], cube_eq[1]),
            ReplacementTransform(square_eq[3], cube_eq[2][1]),
            randy.change("happy", cube_eq),
        )
        self.play(
            LaggedStart(
                FadeIn(cube_eq[0][0]),
                FadeIn(cube_eq[2][0]),
                FadeIn(cube_eq[3]),
                FadeIn(cube_eq[4].brackets),
            ),
            randy.change("tease", cube_eq),
        )
        self.show_mat_mult(
            matrix, matrix_square,
            cube_eq[2][0][2:11],
            cube_eq[2][1].get_entries(),
            cube_eq[4].get_entries(),
            0.1, 0.1,
        )
        self.play(Blink(morty))

        # Scaling
        example_matrix = Matrix([
            ["a", "b", "c"],
            ["d", "e", "f"],
            ["g", "h", "i"],
        ])
        example_scaled_matrix = Matrix([
            ["a / n!", "b / n!", "c / n!"],
            ["d / n!", "e / n!", "f / n!"],
            ["g / n!", "h / n!", "i / n!"],
        ])
        factor = Tex("1 \\over n!")
        factor.scale(1.5)
        factor.next_to(example_matrix, LEFT, MED_SMALL_BUFF)

        self.play(
            LaggedStartMap(FadeOut, VGroup(mat_brace, *cube_eq[:-1])),
            FadeIn(factor),
            FadeTransformPieces(cube_eq[-1], example_matrix),
        )
        self.wait()
        self.play(
            TransformMatchingShapes(
                VGroup(*factor, *example_matrix),
                example_scaled_matrix,
            ),
            randy.change("pondering", example_scaled_matrix),
        )
        self.wait()

        # Adding
        mat1 = np.array([[2, 7, 1], [8, 2, 8], [1, 8, 2]])
        mat2 = np.array([[8, 4, 5], [9, 0, 4], [5, 2, 3]])

        sum_eq = VGroup(
            IntegerMatrix(mat1),
            Tex("+"),
            IntegerMatrix(mat2),
            Tex("="),
            Matrix(
                np.array([
                    f"{m1} + {m2}"
                    for m1, m2 in zip(mat1.flatten(), mat2.flatten())
                ]).reshape((3, 3)),
                h_buff=1.8,
            )
        )
        sum_eq.set_height(1.5)
        sum_eq.arrange(RIGHT)
        sum_eq.center()

        self.play(
            FadeOut(example_scaled_matrix, UP),
            FadeIn(sum_eq[:-1], UP),
            FadeIn(sum_eq[-1].brackets, UP),
            morty.change("raise_right_hand", sum_eq),
            randy.change("thinking", sum_eq),
        )

        last_rects = VGroup()
        for e1, e2, e3 in zip(sum_eq[0].elements, sum_eq[2].elements, sum_eq[4].elements):
            rects = VGroup(SurroundingRectangle(e1), SurroundingRectangle(e2))
            self.add(e3, rects)
            self.play(FadeOut(last_rects), run_time=0.2)
            self.wait(0.1)
            last_rects = rects
        self.play(FadeOut(last_rects))

        # Ask about infinity
        bubble = randy.get_bubble(TexText("But...going\\\\to $\\infty$?"))
        bubble.shift(SMALL_BUFF * RIGHT)
        self.play(
            Write(bubble),
            Write(bubble.content),
            FadeOut(sum_eq, UP),
            randy.change("sassy", mat_rhs),
            morty.change("guilty", randy.eyes),
        )
        self.play(Blink(randy))
        self.wait()
        self.play(
            FadeOut(bubble),
            bubble.content.animate.next_to(randy, RIGHT, aligned_edge=UP),
            randy.change("pondering", mat_rhs),
            morty.change("pondering", mat_rhs),
        )

        # Replace matrix
        pi_mat_tex = ""
        pi_mat_tex = "\\left[ \\begin{array}{cc} 0 & -\\pi \\\\ \\pi & 0 \\end{array} \\right]"
        pi_mat_rhs = Tex(
            rhs_tex.replace("X", pi_mat_tex),
            tex_to_color_map={pi_mat_tex: BLUE},
        )
        pi_mat_rhs.match_width(mat_rhs)
        pi_mat_rhs.move_to(mat_rhs)

        pi_mat = pi_mat_rhs.get_part_by_tex(pi_mat_tex).copy()
        pi_mat.scale(1.5)
        pi_mat.next_to(morty, UL)

        self.play(
            morty.change("raise_right_hand"),
            FadeIn(pi_mat, UP)
        )
        self.play(Blink(morty))
        self.play(
            FadeTransformPieces(mat_rhs, pi_mat_rhs),
            Transform(
                VGroup(pi_mat),
                pi_mat_rhs.get_parts_by_tex(pi_mat_tex),
                remover=True,
            ),
            morty.change("tease"),
        )
        self.wait()

        # Show various partial sum values
        matrix = np.array([[0, -np.pi], [np.pi, 0]])
        curr_matrix = np.identity(2)
        curr_sum = np.identity(2)
        curr_sum_mob = IntegerMatrix(curr_matrix)
        curr_sum_mob.set_height(1.5)
        mat_parts = pi_mat_rhs.get_parts_by_tex(pi_mat_tex)

        brace = Brace(mat_parts[0], DOWN)
        brace.stretch(1.1, 0, about_edge=LEFT)
        curr_sum_mob.next_to(brace, DOWN)
        curr_sum_mob.shift_onto_screen()

        self.play(
            GrowFromCenter(brace),
            FadeTransform(mat_parts[0].copy(), curr_sum_mob),
            randy.change("erm", curr_sum_mob),
        )
        self.wait()

        last_n_label = VMobject()
        partial_sum_mobs = [curr_sum_mob]
        for n in range(1, 18):
            if n < 5:
                new_brace = Brace(mat_parts[:n + 1])
                new_brace.set_width(new_brace.get_width() + 0.2, about_edge=LEFT)
                brace.generate_target()
                brace.target.become(new_brace)
                anims = [
                    MoveToTarget(brace),
                ]
            else:
                n_label = Tex(f"n = {n}", font_size=24)
                n_label.next_to(brace.get_corner(DR), DL, SMALL_BUFF)
                anims = [
                    FadeIn(n_label),
                    FadeOut(last_n_label),
                ]
                last_n_label = n_label

            curr_matrix = np.dot(curr_matrix, matrix) / n
            curr_sum += curr_matrix
            nd = min(n + 1, 4)
            if n < 2:
                h_buff = 1.3
            else:
                sample = DecimalMatrix(curr_sum[0], num_decimal_places=nd)
                sample.replace(curr_sum_mob.get_entries()[0], 1)
                h_buff = 1.3 * sample.get_width()
            new_sum_mob = DecimalMatrix(
                curr_sum,
                element_alignment_corner=RIGHT,
                element_to_mobject_config={
                    "num_decimal_places": nd,
                    "font_size": 36,
                },
                h_buff=h_buff,
            )
            new_sum_mob.match_height(curr_sum_mob)
            new_sum_mob.next_to(brace.target, DOWN)

            self.play(
                FadeOut(curr_sum_mob),
                FadeIn(new_sum_mob),
                randy.animate.look_at(new_sum_mob),
                *anims,
                run_time=(1 if n < 5 else 1 / 60)
            )
            self.wait()
            curr_sum_mob = new_sum_mob
            partial_sum_mobs.append(new_sum_mob)
        self.play(
            FadeOut(last_n_label),
            randy.change("confused", curr_sum_mob),
        )

        # Ask why
        why = Text("Why?")
        why.move_to(bubble.content, UL)
        epii = Tex("e^{\\pi i} = -1")
        epii.next_to(morty, UL)
        later_text = Text("...but that comes later", font_size=24)
        later_text.set_color(GREY_A)
        later_text.next_to(epii, DOWN, aligned_edge=RIGHT)

        self.play(
            randy.change("maybe"),
            FadeIn(why, UP),
            FadeOut(bubble.content, UP),
        )
        self.wait()
        self.play(
            morty.change("raise_right_hand"),
            FadeIn(epii, UP),
        )
        self.play(Blink(morty))
        self.play(
            Write(later_text, run_time=1),
            randy.change("hesitant", morty.eyes)
        )

        # Show partial sums
        new_mat_rhs = Tex(
            rhs_tex.replace("X", mat_tex),
            tex_to_color_map={mat_tex: TEAL},
            isolate=["+"]
        )
        new_mat_rhs.replace(mat_rhs)
        self.play(
            FadeOut(pi_mat_rhs),
            FadeIn(new_mat_rhs),
            FadeOut(new_sum_mob, DOWN),
            brace.animate.become(Brace(new_mat_rhs, DOWN)),
            LaggedStartMap(
                FadeOut, VGroup(
                    why, epii, later_text,
                ),
                shift=DOWN,
            ),
            randy.change("pondering", new_mat_rhs),
            morty.change("pondering", new_mat_rhs),
        )

        matrix = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
        partial_sum_mobs = VGroup()
        curr_matrix = np.identity(3)
        partial_sum = np.array(curr_matrix)
        for n in range(50):
            psm = DecimalMatrix(
                partial_sum,
                element_to_mobject_config={"num_decimal_places": 2},
                element_alignment_corner=ORIGIN,
                h_buff=1.5 * DecimalNumber(partial_sum[0, 0]).get_width(),
                v_buff=1.0,
            )
            psm.next_to(brace, DOWN, MED_LARGE_BUFF)
            partial_sum_mobs.add(psm)
            curr_matrix = np.dot(curr_matrix, matrix) / (n + 1)
            partial_sum += curr_matrix

        new_mat_rhs[2:].set_opacity(0.1)
        self.add(partial_sum_mobs[0])
        self.wait(0.5)
        for n, k in zip(it.count(1), [5, 9, 13, 19, 21]):
            self.remove(partial_sum_mobs[n - 1])
            self.add(partial_sum_mobs[n])
            new_mat_rhs[:k].set_opacity(1)
            self.wait(0.5)
        brace.become(Brace(new_mat_rhs, DOWN))
        n_label = VGroup(Tex("n = "), Integer(n))
        n_label[1].set_height(n_label[0].get_height() * 1.2)
        n_label.arrange(RIGHT, SMALL_BUFF)
        n_label.set_color(GREY_B)
        n_label.next_to(brace.get_corner(DR), DL, SMALL_BUFF)
        self.add(n_label)
        for n in range(6, 50):
            self.remove(partial_sum_mobs[n - 1])
            self.add(partial_sum_mobs[n])
            n_label[1].set_value(n)
            n_label[1].set_color(GREY_B)
            n_label[1].next_to(n_label[0], RIGHT, SMALL_BUFF)
            self.wait(0.1)
        self.play(
            randy.change("erm"),
            morty.change("tease"),
            # LaggedStartMap(
            #     FadeOut, VGroup(brace, n_label, partial_sum_mobs[n]),
            #     shift=DOWN,
            # )
        )
        self.play(Blink(morty))
        self.play(Blink(randy))
        self.wait()

    def show_mat_mult(self, m1, m2, m1_terms, m2_terms, rhs_terms, per_term=0.1, between_terms=0.35):
        dim = m1.shape[0]
        m1_color = m1_terms[0].get_fill_color()
        m2_color = m2_terms[0].get_fill_color()
        for n in range(dim * dim):
            i = n // dim
            j = n % dim
            row = m1_terms[dim * i:dim * i + dim]
            col = m2_terms[j::dim]
            row_rect = SurroundingRectangle(row, buff=0.05)
            col_rect = SurroundingRectangle(col, buff=0.05)
            row_rect.set_stroke(YELLOW, 2)
            col_rect.set_stroke(YELLOW, 2)
            right_elem = Integer(0, edge_to_fix=ORIGIN)
            right_elem.replace(rhs_terms[n], dim_to_match=1)
            right_elem.set_value(0)

            self.add(row_rect, col_rect, right_elem)
            for k in range(dim):
                self.wait(per_term)
                right_elem.increment_value(m1[i, k] * m2[k, j])
                right_elem.scale(rhs_terms[0][0].get_height() / right_elem[-1].get_height())
                row[k].set_color(YELLOW)
                col[k].set_color(YELLOW)
            self.remove(right_elem)
            self.add(rhs_terms[n])
            self.wait(between_terms)
            m1_terms.set_color(m1_color)
            m2_terms.set_color(m2_color)
            self.remove(row_rect, col_rect)


class ShowHigherMatrixPowers(IntroduceTheComputation):
    matrix = [[3, 1, 4], [1, 5, 9], [2, 6, 5]]
    per_term = 0.1
    between_terms = 0.1
    N_powers = 10

    def construct(self):
        # Show many matrix powers
        def get_mat_mob(matrix):
            term = Integer(matrix[0, 0])
            return IntegerMatrix(
                matrix,
                h_buff=max(0.8 + 0.3 * len(term), 1.0)
            )

        N = self.N_powers
        matrix = np.matrix(self.matrix)
        matrix_powers = [np.identity(len(matrix)), matrix]
        for x in range(N):
            matrix_powers.append(np.dot(matrix, matrix_powers[-1]))

        mat_mobs = [get_mat_mob(mat) for mat in matrix_powers]
        for mob in mat_mobs:
            mob.set_height(1)
        mat_mobs[1].set_color(TEAL)

        equation = VGroup(
            mat_mobs[1].deepcopy(),
            Integer(2, font_size=18),
            Tex("="),
            mat_mobs[1].deepcopy(),
            mat_mobs[1].deepcopy(),
            Tex("="),
            mat_mobs[2].deepcopy()
        )
        equation.arrange(RIGHT)
        equation.to_edge(LEFT)
        equation[1].set_y(equation[0].get_top()[1])
        equation[0].set_x(equation[1].get_left()[0] - SMALL_BUFF, RIGHT)
        self.add(equation)

        exp = equation[1]
        m1, m2, eq, rhs = equation[-4:]
        self.remove(*rhs.get_entries())
        for n in range(3, N):
            self.show_mat_mult(
                matrix_powers[1], matrix_powers[n - 2],
                m1.get_entries(), m2.get_entries(), rhs.get_entries(),
                between_terms=self.between_terms,
                per_term=self.per_term
            )
            self.wait(0.5)
            rhs.generate_target()
            eq.generate_target()
            rhs.target.move_to(m2, LEFT)
            eq.target.next_to(rhs.target, RIGHT)
            new_rhs = mat_mobs[n].deepcopy()
            new_rhs.next_to(eq.target, RIGHT)
            new_exp = Integer(n)
            new_exp.replace(exp, dim_to_match=1)

            self.play(
                MoveToTarget(rhs, path_arc=PI / 2),
                MoveToTarget(eq),
                FadeOut(m2, DOWN),
                FadeIn(new_rhs.get_brackets()),
                FadeIn(new_exp, 0.5 * DOWN),
                FadeOut(exp, 0.5 * DOWN),
            )
            m2, rhs = rhs, new_rhs
            exp = new_exp


class Show90DegreePowers(ShowHigherMatrixPowers):
    matrix = [[0, -1], [1, 0]]
    per_term = 0.0
    between_terms = 0.2
    N_powers = 16


class WhyTortureMatrices(TeacherStudentsScene):
    def construct(self):
        self.play_student_changes(
            "maybe", "confused", "erm",
            look_at=self.screen,
        )
        q_marks = VGroup()
        for student in self.students:
            marks = Tex("???")
            marks.next_to(student, UP)
            q_marks.add(marks)
        self.play(FadeIn(q_marks, 0.25 * UP, lag_ratio=0.1, run_time=2))
        self.wait(2)
        self.student_says(
            TexText("Why...would you\\\\ever want\\\\to do that?"),
            index=2,
            added_anims=[FadeOut(q_marks)],
        )
        self.play(
            self.change_students("confused", "pondering", "raise_left_hand", look_at=self.screen),
            self.teacher.change("tease", self.screen)
        )
        self.wait(2)
        self.play(self.students[0].change("erm"))
        self.wait(7)


class DefinitionFirstVsLast(Scene):
    show_love_and_quantum = True

    def construct(self):
        # Setup objects
        top_title = Text("Textbook progression")
        low_title = Text("Discovery progression")

        top_prog = VGroup(
            TexText("Definition", color=BLUE),
            TexText("Theorem"),
            TexText("Proof"),
            TexText("Examples"),
        )
        low_prog = VGroup(
            TexText("Specific\n\nproblem"),
            TexText("General\n\nproblems"),
            TexText("Helpful\n\nconstructs"),
            TexText("Definition", color=BLUE),
        )
        progs = VGroup(top_prog, low_prog)
        for progression in progs:
            progression.arrange(RIGHT, buff=1.2)
        progs.arrange(DOWN, buff=3)
        progs.set_width(FRAME_WIDTH - 2)

        for progression in progs:
            arrows = VGroup()
            for m1, m2 in zip(progression[:-1], progression[1:]):
                arrows.add(Arrow(m1, m2))
            progression.arrows = arrows

        top_dots = Tex("\\dots", font_size=72)
        top_dots.next_to(top_prog.arrows[0], RIGHT)
        low_dots = top_dots.copy()
        low_dots.next_to(low_prog.arrows[-1], LEFT)

        top_rect = SurroundingRectangle(top_prog, buff=MED_SMALL_BUFF)
        top_rect.set_stroke(TEAL, 2)
        top_title.next_to(top_rect, UP)
        top_title.match_color(top_rect)
        low_rect = SurroundingRectangle(low_prog, buff=MED_SMALL_BUFF)
        low_rect.set_stroke(YELLOW, 2)
        low_title.next_to(low_rect, UP)
        low_title.match_color(low_rect)
        versus = Text("vs.")

        # Show progressions
        self.add(top_prog[0])
        self.play(
            GrowArrow(top_prog.arrows[0]),
            FadeIn(top_dots, 0.2 * RIGHT, lag_ratio=0.1),
        )
        self.wait()
        kw = {"path_arc": -90 * DEGREES}
        self.play(
            LaggedStart(
                TransformFromCopy(top_prog[0], low_prog[-1], **kw),
                TransformFromCopy(top_prog.arrows[0], low_prog.arrows[-1], **kw),
                TransformFromCopy(top_dots, low_dots, **kw),
            ),
            Write(versus)
        )
        self.wait()

        self.play(
            ShowCreation(top_rect),
            FadeIn(top_title, 0.25 * UP)
        )
        self.play(
            FadeOut(top_dots),
            FadeIn(top_prog[1]),
        )
        for arrow, term in zip(top_prog.arrows[1:], top_prog[2:]):
            self.play(
                GrowArrow(arrow),
                FadeIn(term, shift=0.25 * RIGHT),
            )
        self.wait()

        self.play(
            ShowCreation(low_rect),
            FadeIn(low_title, 0.25 * UP),
            versus.animate.move_to(midpoint(low_title.get_top(), top_rect.get_bottom())),
        )
        self.wait()
        self.play(FadeIn(low_prog[0]))
        self.play(
            GrowArrow(low_prog.arrows[0]),
            FadeIn(low_prog[1], shift=0.25 * RIGHT),
        )
        self.play(
            GrowArrow(low_prog.arrows[1]),
            FadeIn(low_prog[2], shift=0.25 * RIGHT),
            FadeOut(low_dots),
        )
        self.wait()

        # Highlight specific example
        full_rect = FullScreenRectangle()
        full_rect.set_fill(BLACK, opacity=0.75)
        sp, gp, hc = low_prog[:3].copy()
        self.add(full_rect, sp)
        self.play(FadeIn(full_rect))
        self.wait()

        # Go to general
        if not self.show_love_and_quantum:
            self.play(FadeIn(gp))
            self.play(FlashAround(gp, color=BLUE, run_time=2))
            self.wait()
            return

        # Love and quantum
        love = SVGMobject("hearts")
        love.set_height(1)
        love.set_fill(RED, 1)
        love.set_stroke(MAROON_B, 1)

        quantum = Tex("|\\psi\\rangle")
        quantum.set_color(BLUE)
        quantum.match_height(love)
        group = VGroup(quantum, love)
        group.arrange(RIGHT, buff=MED_LARGE_BUFF)
        group.next_to(sp, UP, MED_LARGE_BUFF)

        love.save_state()
        love.match_x(sp)
        self.play(Write(love))
        self.wait()
        self.play(
            Restore(love),
            FadeIn(quantum, 0.5 * LEFT)
        )
        self.wait()
        self.play(
            love.animate.center().scale(1.5),
            FadeOut(quantum),
            FadeOut(sp),
            full_rect.animate.set_fill(opacity=1)
        )
        self.wait()


class DefinitionFirstVsLastGP(DefinitionFirstVsLast):
    show_love_and_quantum = False


class RomeoAndJuliet(Scene):
    def construct(self):
        # Add Romeo and Juliet
        romeo, juliet = lovers = self.get_romeo_and_juliet()
        lovers.set_height(2)
        lovers.arrange(LEFT, buff=1)
        lovers.move_to(0.5 * DOWN)

        self.add(*lovers)
        self.make_romeo_and_juliet_dynamic(romeo, juliet)
        romeo.love_tracker.set_value(1.5)
        juliet.love_tracker.set_value(1.5)
        get_romeo_juilet_name_labels(lovers)

        for creature in lovers:
            self.play(
                creature.love_tracker.animate.set_value(2.5),
                Write(creature.name_label, run_time=1),
            )
        self.wait()

        # Add their scales
        juliet_scale = self.get_love_scale(juliet, LEFT, "x", BLUE_B)
        romeo_scale = self.get_love_scale(romeo, RIGHT, "y", BLUE)
        scales = [juliet_scale, romeo_scale]

        scale_labels = VGroup(
            TexText("Juliet's love for Romeo", font_size=30),
            TexText("Romeo's love for Juliet", font_size=30),
        )
        scale_arrows = VGroup()
        for scale, label in zip(scales, scale_labels):
            var = scale[2][0][0]
            label.next_to(var, UP, buff=0.7)
            arrow = Arrow(var, label, buff=0.1, thickness=0.025)
            scale_arrows.add(arrow)
            label.set_color(var.get_fill_color())

        for lover, scale, arrow, label, final_love in zip(reversed(lovers), scales, scale_arrows, scale_labels, [1, -1]):
            self.add(scale)
            self.play(FlashAround(scale[2][0][0]))
            self.play(
                lover.love_tracker.animate.set_value(5),
                GrowArrow(arrow),
                FadeIn(label, 0.5 * UP),
            )
            self.play(lover.love_tracker.animate.set_value(final_love), run_time=2)
            self.wait()

        # Juliet's rule
        frame = self.camera.frame
        equations = VGroup(
            Tex("{dx \\over dt} {{=}} -{{y(t)}}"),
            Tex("{dy \\over dt} {{=}} {{x(t)}}"),
        )
        juliet_eq, romeo_eq = equations
        juliet_eq.next_to(scale_labels[0], UR)
        juliet_eq.shift(0.5 * UP)

        self.play(
            frame.animate.move_to(0.7 * UP),
            Write(equations[0]),
        )
        self.wait()
        self.play(FlashAround(juliet_eq[0]))
        self.wait()
        y_rect = SurroundingRectangle(juliet_eq.get_parts_by_tex("y(t)"), buff=0.05)
        y_rect_copy = y_rect.copy()
        y_rect_copy.replace(romeo.scale_mob.dot, stretch=True)
        self.play(FadeIn(y_rect))
        self.wait()
        self.play(TransformFromCopy(y_rect, y_rect_copy))
        y_rect_copy.add_updater(lambda m: m.move_to(romeo.scale_mob.dot))
        self.wait()
        self.play(romeo.love_tracker.animate.set_value(-3))

        big_arrow = Arrow(
            juliet.scale_mob.number_line.get_bottom(),
            juliet.scale_mob.number_line.get_top(),
        )
        big_arrow.set_color(GREEN)
        big_arrow.next_to(juliet.scale_mob.number_line, LEFT)

        self.play(
            FadeIn(big_arrow),
            ApplyMethod(juliet.love_tracker.set_value, 5, run_time=3, rate_func=linear),
        )
        self.wait()
        self.play(romeo.love_tracker.animate.set_value(5))
        self.play(
            big_arrow.animate.rotate(PI).set_color(RED),
            path_arc=PI,
            run_time=0.5,
        )
        self.play(juliet.love_tracker.animate.set_value(-5), rate_func=linear, run_time=5)
        self.play(FadeOut(y_rect), FadeOut(y_rect_copy))

        # Romeo's rule
        romeo_eq.next_to(scale_labels[1], UL)
        romeo_eq.shift(0.5 * UP)
        self.play(
            juliet_eq.animate.to_edge(LEFT),
            FadeOut(big_arrow),
        )
        self.play(FadeIn(romeo_eq, UP))
        self.wait()

        dy_rect = SurroundingRectangle(romeo_eq.get_part_by_tex("dy"))
        x_rect = SurroundingRectangle(romeo_eq.get_part_by_tex("x(t)"), buff=0.05)
        x_rect_copy = x_rect.copy()
        x_rect_copy.replace(juliet.scale_mob.dot, stretch=True)
        self.play(ShowCreation(dy_rect))
        self.wait()
        self.play(TransformFromCopy(dy_rect, x_rect))
        self.play(TransformFromCopy(x_rect, x_rect_copy))
        self.wait()

        big_arrow.next_to(romeo.scale_mob.number_line, RIGHT)
        self.play(FadeIn(big_arrow), LaggedStartMap(FadeOut, VGroup(dy_rect, x_rect)))
        self.play(romeo.love_tracker.animate.set_value(-3), run_time=4, rate_func=linear)
        x_rect_copy.add_updater(lambda m: m.move_to(juliet.scale_mob.dot))
        juliet.love_tracker.set_value(5)
        self.wait()
        self.play(
            big_arrow.animate.rotate(PI).set_color(GREEN),
            path_arc=PI,
            run_time=0.5,
        )
        self.play(romeo.love_tracker.animate.set_value(5), rate_func=linear, run_time=5)
        self.play(FadeOut(x_rect_copy))
        self.wait()

        # Show constant change
        left_arrow = Arrow(UP, DOWN)
        left_arrow.character = juliet
        left_arrow.get_rate = lambda: -romeo.love_tracker.get_value()

        right_arrow = Arrow(DOWN, UP)
        right_arrow.character = romeo
        right_arrow.get_rate = lambda: juliet.love_tracker.get_value()

        def update_arrow(arrow):
            nl = arrow.character.scale.number_line
            rate = arrow.get_rate()
            if rate == 0:
                rate = 1e-6
            arrow.put_start_and_end_on(nl.n2p(0), nl.n2p(rate))
            arrow.next_to(nl, np.sign(nl.get_center()[0]) * RIGHT)
            if rate > 0:
                arrow.set_color(GREEN)
            else:
                arrow.set_color(RED)

        left_arrow.add_updater(update_arrow)
        right_arrow.add_updater(update_arrow)

        self.play(
            VFadeIn(left_arrow),
            ApplyMethod(big_arrow.scale, 0, remover=True, run_time=3),
            ApplyMethod(juliet.love_tracker.set_value, 0, run_time=3),
        )

        ps_point = Point(5 * UP)
        curr_time = self.time
        ps_point.add_updater(lambda m: m.move_to([
            -5 * np.sin(0.5 * (self.time - curr_time)),
            5 * np.cos(0.5 * (self.time - curr_time)),
            0,
        ]))
        juliet.love_tracker.add_updater(lambda m: m.set_value(ps_point.get_location()[0]))
        romeo.love_tracker.add_updater(lambda m: m.set_value(ps_point.get_location()[1]))
        self.add(ps_point)
        self.add(right_arrow)

        self.play(
            equations.animate.arrange(RIGHT, buff=LARGE_BUFF).to_edge(UP, buff=0),
            run_time=2,
        )
        # Just let this play out for a long time while other animations are played on top
        self.wait(5 * TAU)

    def get_romeo_and_juliet(self):
        romeo = PiCreature(color=BLUE_E, flip_at_start=True)
        juliet = PiCreature(color=BLUE_B)
        return VGroup(romeo, juliet)

    def make_romeo_and_juliet_dynamic(self, romeo, juliet):
        cutoff_values = [-5, -3, -1, 0, 1, 3, 5]
        modes = ["angry", "sassy", "hesitant", "plain", "happy", "hooray", "surprised"]
        self.make_character_dynamic(romeo, juliet, cutoff_values, modes)
        self.make_character_dynamic(juliet, romeo, cutoff_values, modes)

    def get_romeo_juilet_name_labels(self, lovers, font_size=36, spacing=1.2, buff=MED_SMALL_BUFF):
        name_labels = VGroup(*(
            Text(name, font_size=font_size)
            for name in ["Romeo", "Juliet"]
        ))
        for label, creature in zip(name_labels, lovers):
            label.next_to(creature, DOWN, buff)
            creature.name_label = label
        name_labels.space_out_submobjects(spacing)
        return name_labels

    def make_character_dynamic(self, pi_creature, lover, cutoff_values, modes):
        height = pi_creature.get_height()
        bottom = pi_creature.get_bottom()
        copies = [
            pi_creature.deepcopy().change(mode).set_height(height).move_to(bottom, DOWN)
            for mode in modes
        ]
        pi_creature.love_tracker = ValueTracker()

        def update_func(pi):
            love = pi.love_tracker.get_value()

            if love < cutoff_values[0]:
                pi.become(copies[0])
            elif love >= cutoff_values[-1]:
                pi.become(copies[-1])
            else:
                i = 1
                while cutoff_values[i] < love:
                    i += 1
                copy1 = copies[i - 1]
                copy2 = copies[i]

                alpha = inverse_interpolate(cutoff_values[i - 1], cutoff_values[i], love)
                s_alpha = squish_rate_func(smooth, 0.25, 0.75)(alpha)

                # if s_alpha > 0:
                copy1.align_data_and_family(copy2)
                pi.align_data_and_family(copy1)
                pi.align_data_and_family(copy2)
                fam = pi.family_members_with_points()
                f1 = copy1.family_members_with_points()
                f2 = copy2.family_members_with_points()
                for sm, sm1, sm2 in zip(fam, f1, f2):
                    sm.interpolate(sm1, sm2, s_alpha)

            pi.look_at(lover.get_top())
            if love < cutoff_values[1]:
                # Look away from the lover
                pi.look_at(2 * pi.eyes.get_center() - lover.eyes.get_center() + DOWN)

            return pi

        pi_creature.add_updater(update_func)

        def update_eyes(heart_eyes):
            love = pi_creature.love_tracker.get_value()
            l_alpha = np.clip(
                inverse_interpolate(cutoff_values[-1] - 0.5, cutoff_values[-1], love),
                0, 1
            )
            pi_creature.eyes.set_opacity(1 - l_alpha)
            heart_eyes.set_opacity(l_alpha)
            # heart_eyes.move_to(pi_creature.eyes)
            heart_eyes.match_x(pi_creature.mouth)

        heart_eyes = self.get_heart_eyes(pi_creature)
        heart_eyes.add_updater(update_eyes)
        pi_creature.heart_eyes = heart_eyes
        self.add(heart_eyes)
        return pi_creature

    def get_heart_eyes(self, creature):
        hearts = VGroup()
        for eye in creature.eyes:
            heart = SVGMobject("hearts")
            heart.set_fill(RED)
            heart.match_width(eye)
            heart.move_to(eye)
            heart.scale(1.25)
            heart.set_stroke(BLACK, 1)
            hearts.add(heart)
        hearts.set_opacity(0)
        return hearts

    def get_love_scale(self, creature, direction, var_name, color):
        number_line = NumberLine((-5, 5))
        number_line.rotate(90 * DEGREES)
        number_line.set_height(1.5 * creature.get_height())
        number_line.next_to(creature, direction, buff=MED_LARGE_BUFF)
        number_line.add_numbers(
            range(-4, 6, 2),
            font_size=18,
            color=GREY_B,
            buff=0.1,
            direction=LEFT,
        )

        dot = Dot(color=color)
        dot.add_updater(lambda m: m.move_to(number_line.n2p(creature.love_tracker.get_value())))

        label = VGroup(Tex(var_name, "=", font_size=36), DecimalNumber(font_size=24))
        label.set_color(color)
        label[0].shift(label[1].get_left() + SMALL_BUFF * LEFT - label[0][1].get_right())
        label.next_to(number_line, UP)
        label[1].add_updater(lambda m: m.set_value(creature.love_tracker.get_value()).set_color(color))

        result = VGroup(number_line, dot, label)
        result.set_stroke(background=True)
        result.number_line = number_line
        result.dot = dot
        result.label = label
        creature.scale_mob = result

        return result


class DiscussSystem(Scene):
    def construct(self):
        # Setup equations
        equations = VGroup(
            Tex("{dx \\over dt} {{=}} -{{y(t)}}"),
            Tex("{dy \\over dt} {{=}} {{x(t)}}"),
        )
        equations.arrange(RIGHT, buff=LARGE_BUFF)
        equations.to_edge(UP, buff=1.5)

        eq_rect = SurroundingRectangle(equations, stroke_width=2, buff=0.25)
        sys_label = Text("System of differential equations")
        sys_label.next_to(eq_rect, UP)

        self.add(equations)

        self.play(
            FadeIn(sys_label, 0.5 * UP),
            ShowCreation(eq_rect),
        )
        style = {"color": BLUE, "time_width": 3, "run_time": 2}
        self.play(LaggedStart(
            FlashAround(sys_label.get_part_by_text("differential"), **style),
            FlashAround(equations[0].get_part_by_tex("dx"), **style),
            FlashAround(equations[1].get_part_by_tex("dy"), **style),
        ))
        self.wait()

        # Ask for explicit solutions
        solutions = VGroup(
            Tex("x(t) {{=}} (\\text{expression with } t)"),
            Tex("y(t) {{=}} (\\text{expression with } t)"),
        )
        for solution in solutions:
            solution.set_color_by_tex("expression", GREY_B)
        solutions.arrange(DOWN, buff=0.5)
        solutions.move_to(equations)
        solutions.set_x(3)

        self.play(
            sys_label.animate.match_width(eq_rect).to_edge(LEFT),
            VGroup(equations, eq_rect).animate.to_edge(LEFT),
            LaggedStartMap(FadeIn, solutions, shift=0.5 * UP, lag_ratio=0.3),
        )
        self.wait()

        # Show a guess
        guess_rhss = VGroup(
            Tex("\\cos(t)", color=GREY_B)[0],
            Tex("\\sin(t)", color=GREY_B)[0],
        )
        temp_rhss = VGroup()
        for rhs, solution in zip(guess_rhss, solutions):
            temp_rhss.add(solution[2])
            rhs.move_to(solution[2], LEFT)

        bubble = ThoughtBubble(height=4, width=4)
        bubble.flip()
        bubble.set_fill(opacity=0)
        bubble[:3].rotate(30 * DEGREES, about_point=bubble[3].get_center() + 0.2 * RIGHT)
        bubble.shift(solutions.get_left() + 0.7 * LEFT - bubble[3].get_left())

        self.remove(temp_rhss)
        self.play(
            ShowCreation(bubble),
            *(
                TransformMatchingShapes(temp_rhs.copy(), guess_rhs)
                for temp_rhs, guess_rhs in zip(temp_rhss, guess_rhss)
            ),
        )
        self.wait()

        # Not enough!
        not_enough = Text("Not enough!", font_size=40)
        not_enough.next_to(bubble[3].get_corner(UR), DR)
        not_enough.set_color(RED)

        self.play(LaggedStartMap(FadeIn, not_enough, run_time=1, lag_ratio=0.1))
        self.wait()
        self.remove(guess_rhss)
        self.play(
            LaggedStartMap(FadeOut, VGroup(*bubble, *not_enough)),
            *(
                TransformMatchingShapes(guess_rhs.copy(), temp_rhs)
                for temp_rhs, guess_rhs in zip(temp_rhss, guess_rhss)
            ),
        )

        # Initial condition
        solutions.generate_target()
        initial_conditions = VGroup(
            Tex("x(0) = x_0"),
            Tex("y(0) = y_0"),
        )
        full_requirement = VGroup(*solutions.target, *initial_conditions)
        full_requirement.arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        full_requirement.scale(0.8)
        full_requirement.move_to(solutions)
        full_requirement.to_edge(UP)

        self.play(
            MoveToTarget(solutions),
            LaggedStartMap(FadeIn, initial_conditions, shift=0.1 * UP, lag_ratio=0.3),
        )
        self.wait()

        ic_label = Text("Initial condition", font_size=30)
        ic_label.set_color(BLUE)
        ic_label.next_to(initial_conditions, RIGHT, buff=1.0)
        ic_arrows = VGroup(*(
            Arrow(ic_label.get_left(), eq.get_right(), buff=0.1, fill_color=BLUE, thickness=0.025)
            for eq in initial_conditions
        ))

        self.play(
            FadeIn(ic_label),
            LaggedStartMap(GrowArrow, ic_arrows, run_time=1)
        )
        self.wait()


class MoreGeneralSystem(Scene):
    def construct(self):
        kw = {
            "tex_to_color_map": {
                "x": RED,
                "y": GREEN,
                "z": BLUE,
                "{t}": GREY_B,
            }
        }
        equations = VGroup(
            Tex("{dx \\over d{t} } = a\\cdot x({t}) + b\\cdot y({t}) + c\\cdot z({t})", **kw),
            Tex("{dy \\over d{t} } = d\\cdot x({t}) + e\\cdot y({t}) + f\\cdot z({t})", **kw),
            Tex("{dz \\over d{t} } = g\\cdot x({t}) + h\\cdot y({t}) + i\\cdot z({t})", **kw),
        )
        equations.arrange(DOWN, buff=LARGE_BUFF)

        self.add(equations)
        self.play(LaggedStartMap(FadeIn, equations, shift=UP, lag_ratio=0.5, run_time=3))
        self.wait()


class HowExampleLeadsToMatrixExponents(Scene):
    def construct(self):
        # Screen
        self.add(FullScreenRectangle())
        screen = ScreenRectangle()
        screen.set_height(3)
        screen.set_fill(BLACK, 1)
        screen.set_stroke(BLUE_B, 2)
        screen.to_edge(LEFT)
        self.add(screen)

        # Mat exp
        mat_exp = get_matrix_exponential(
            [["a", "b"], ["c", "d"]],
            height=2,
            h_buff=0.95, v_buff=0.75
        )
        mat_exp.set_x(FRAME_WIDTH / 4)

        def get_arrow():
            return Arrow(screen, mat_exp[0])

        arrow = get_arrow()

        self.play(
            GrowArrow(arrow),
            FadeIn(mat_exp, RIGHT)
        )
        self.wait()

        # New screen
        screen2 = screen.copy()
        screen2.set_stroke(GREY_BROWN, 2)
        screen2.to_corner(DR)

        mat_exp.generate_target()
        mat_exp.target.to_edge(UP)
        mat_exp.target.match_x(screen2)
        double_arrow = VGroup(
            Arrow(mat_exp.target, screen2),
            Arrow(screen2, mat_exp.target),
        )
        for mob in double_arrow:
            mob.scale(0.9, about_point=mob.get_end())

        self.play(
            MoveToTarget(mat_exp),
            GrowFromCenter(double_arrow),
            arrow.animate.become(Arrow(screen, screen2)),
            FadeIn(screen2, DOWN),
        )
        self.wait()


class RomeoJulietVectorSpace(RomeoAndJuliet):
    def construct(self):
        # Set up Romeo and Juliet
        romeo, juliet = lovers = self.get_romeo_and_juliet()
        lovers.set_height(2.0)
        lovers.arrange(LEFT, buff=3)
        name_labels = self.get_romeo_juilet_name_labels(lovers, font_size=36, spacing=1.1)
        self.make_romeo_and_juliet_dynamic(*lovers)

        self.add(*lovers)
        self.add(*name_labels)

        # Scales
        juliet_scale = self.get_love_scale(juliet, LEFT, "x", BLUE_B)
        romeo_scale = self.get_love_scale(romeo, RIGHT, "y", BLUE)
        scales = [juliet_scale, romeo_scale]
        self.add(*scales)

        # Animate in
        psp_tracker = Point()

        def get_psp():
            # Get phase space point
            return psp_tracker.get_location()

        juliet.love_tracker.add_updater(lambda m: m.set_value(get_psp()[0]))
        romeo.love_tracker.add_updater(lambda m: m.set_value(get_psp()[1]))
        self.add(romeo.love_tracker, juliet.love_tracker)

        psp_tracker.move_to([1, -3, 0])
        self.play(
            Rotate(psp_tracker, 90 * DEGREES, about_point=ORIGIN, run_time=3, rate_func=linear)
        )

        # Transition to axes
        axes = Axes(
            x_range=(-5, 5),
            y_range=(-5, 5),
            height=7,
            width=7,
            axis_config={
                "include_tip": False,
                "numbers_to_exclude": [],
            }
        )
        axes.set_x(-3)

        for axis in axes:
            axis.add_numbers(range(-4, 6, 2), color=GREY_B)
            axis.numbers[2].set_opacity(0)

        for pi in lovers:
            pi.clear_updaters()
            pi.generate_target()
            pi.target.set_height(0.75)
            pi.name_label.generate_target()
            pi.name_label.target.scale(0.5)
            group = VGroup(pi.target, pi.name_label.target)
            group.arrange(DOWN, buff=SMALL_BUFF)
            pi.target_group = group
            pi.scale_mob[2].clear_updaters()
            self.add(*pi.scale_mob)
        juliet.target_group.next_to(axes.x_axis.get_end(), RIGHT)
        romeo.target_group.next_to(axes.y_axis.get_corner(UR), RIGHT)
        romeo.target_group.shift_onto_screen(buff=MED_SMALL_BUFF)
        romeo.target.flip()
        juliet.target.flip()
        juliet.target.make_eye_contact(romeo.target)

        self.play(LaggedStart(
            juliet.scale_mob.number_line.animate.become(axes.x_axis),
            FadeOut(juliet.scale_mob.label),
            MoveToTarget(juliet),
            MoveToTarget(juliet.name_label),
            romeo.scale_mob.number_line.animate.become(axes.y_axis),
            FadeOut(romeo.scale_mob.label),
            MoveToTarget(romeo),
            MoveToTarget(romeo.name_label),
            run_time=3
        ))
        self.add(*romeo.scale_mob[:2], *juliet.scale_mob[:2])

        # Reset pi creatures
        self.remove(lovers)
        self.remove(romeo.heart_eyes)
        self.remove(juliet.heart_eyes)
        new_lovers = self.get_romeo_and_juliet()
        for new_pi, pi in zip(new_lovers, lovers):
            new_pi.flip()
            new_pi.replace(pi)
            new_pi.scale_mob = pi.scale_mob
        lovers = new_lovers
        romeo, juliet = new_lovers
        self.add(romeo, juliet)
        self.make_romeo_and_juliet_dynamic(romeo, juliet)
        juliet.love_tracker.add_updater(lambda m: m.set_value(get_psp()[0]))
        romeo.love_tracker.add_updater(lambda m: m.set_value(get_psp()[1]))
        self.add(romeo.love_tracker, juliet.love_tracker)

        # h_line and v_line
        ps_dot = Dot(color=BLUE)
        ps_dot.add_updater(lambda m: m.move_to(axes.c2p(*get_psp()[:2])))
        v_line = Line().set_stroke(BLUE_D, 2)
        h_line = Line().set_stroke(BLUE_B, 2)
        v_line.add_updater(lambda m: m.put_start_and_end_on(
            axes.x_axis.n2p(get_psp()[0]),
            axes.c2p(*get_psp()[:2]),
        ))
        h_line.add_updater(lambda m: m.put_start_and_end_on(
            axes.y_axis.n2p(get_psp()[1]),
            axes.c2p(*get_psp()[:2]),
        ))
        x_dec = DecimalNumber(0, font_size=24)
        x_dec.next_to(h_line, UP, SMALL_BUFF)
        y_dec = DecimalNumber(0, font_size=24)
        y_dec.next_to(v_line, RIGHT, SMALL_BUFF)

        romeo.scale_mob.dot.clear_updaters()
        juliet.scale_mob.dot.clear_updaters()
        self.play(
            ShowCreation(h_line.copy().clear_updaters(), remover=True),
            ShowCreation(v_line.copy().clear_updaters(), remover=True),
            ReplacementTransform(romeo.scale_mob.dot, ps_dot),
            ReplacementTransform(juliet.scale_mob.dot, ps_dot),
            ChangeDecimalToValue(x_dec, get_psp()[0]),
            VFadeIn(x_dec),
            ChangeDecimalToValue(y_dec, get_psp()[1]),
            VFadeIn(y_dec),
        )
        self.add(h_line, v_line, ps_dot)

        # Add coordinates
        equation = VGroup(
            Matrix([["x"], ["y"]], bracket_h_buff=SMALL_BUFF),
            Tex("="),
            DecimalMatrix(
                np.reshape(get_psp()[:2], (2, 1)),
                element_to_mobject_config={
                    "num_decimal_places": 2,
                    "font_size": 36,
                    "include_sign": True,
                }
            ),
        )
        equation[0].match_height(equation[2])
        equation.arrange(RIGHT)
        equation.to_corner(UR)
        equation.shift(MED_SMALL_BUFF * LEFT)

        self.play(
            FadeIn(equation[:2]),
            FadeIn(equation[2].get_brackets()),
            TransformFromCopy(x_dec, equation[2].get_entries()[0]),
            TransformFromCopy(y_dec, equation[2].get_entries()[1]),
        )
        equation[2].get_entries()[0].add_updater(lambda m: m.set_value(get_psp()[0]))
        equation[2].get_entries()[1].add_updater(lambda m: m.set_value(get_psp()[1]))

        self.play(FadeOut(x_dec), FadeOut(y_dec))

        # Play around in state space
        self.play(psp_tracker.move_to, [3, -2, 0], path_arc=120 * DEGREES, run_time=3)
        self.wait()
        self.play(psp_tracker.move_to, [-5, -2, 0], path_arc=0 * DEGREES, run_time=3, rate_func=there_and_back)
        self.wait()
        self.play(psp_tracker.move_to, [3, 5, 0], path_arc=0 * DEGREES, run_time=3, rate_func=there_and_back)
        self.wait()
        self.play(psp_tracker.move_to, [5, 3, 0], path_arc=-120 * DEGREES, run_time=2)
        self.wait()

        # Arrow vs. dot
        arrow = Arrow(axes.get_origin(), ps_dot.get_center(), buff=0, fill_color=BLUE)
        arrow.set_stroke(BLACK, 2, background=True)
        arrow_outline = arrow.copy()
        arrow_outline.set_fill(opacity=0)
        arrow_outline.set_stroke(YELLOW, 1)
        self.play(LaggedStart(
            FadeIn(arrow),
            FadeOut(ps_dot),
            ShowPassingFlash(arrow_outline, run_time=1, time_width=0.5),
            lag_ratio=0.5,
        ))
        self.wait()
        self.play(LaggedStart(
            FadeIn(ps_dot),
            FadeOut(arrow),
            FlashAround(ps_dot, buff=0.05),
        ))
        self.wait()
        self.play(FlashAround(equation))
        self.play(psp_tracker.move_to, [4, 3, 0], run_time=2)
        self.wait()

        # Function of time
        new_lhs = Matrix([["x(t)"], ["y(t)"]])
        new_lhs.match_height(equation[0])
        new_lhs.move_to(equation[0], RIGHT)

        self.play(
            FadeTransformPieces(equation[0], new_lhs),
        )
        self.remove(equation[0])
        self.add(new_lhs)
        equation.replace_submobject(0, new_lhs)

        # Initialize rotation
        curr_time = self.time
        curr_psp = get_psp()
        psp_tracker.add_updater(lambda m: m.move_to(np.dot(
            curr_psp,
            np.transpose(rotation_about_z(0.25 * (self.time - curr_time))),
        )))
        self.wait(5)

        # Rate of change
        deriv_lhs = Matrix([["x'(t)"], ["y'(t)"]], bracket_h_buff=SMALL_BUFF)
        deriv_lhs.match_height(equation[0])
        deriv_lhs.move_to(equation[0])
        deriv_lhs.set_color(RED_B)
        deriv_label = Text("Rate of change", font_size=24)
        deriv_label.match_width(deriv_lhs)
        deriv_label.match_color(deriv_lhs)
        deriv_label.next_to(deriv_lhs, DOWN, SMALL_BUFF)

        self.play(
            FadeIn(deriv_lhs),
            Write(deriv_label, run_time=1),
            equation.animate.shift(2.0 * deriv_lhs.get_height() * DOWN)
        )
        self.wait(5)

        deriv_vect = Arrow(fill_color=RED_B)
        deriv_vect.add_updater(
            lambda m: m.put_start_and_end_on(
                axes.get_origin(),
                axes.c2p(-0.5 * get_psp()[1], 0.5 * get_psp()[0])
            ).shift(
                ps_dot.get_center() - axes.get_origin()
            )
        )
        pre_vect = Arrow(LEFT, RIGHT)
        pre_vect.replace(deriv_label, dim_to_match=0)
        pre_vect.set_fill(RED_B, 0)
        moving_vect = pre_vect.copy()
        deriv_vect.set_opacity(0)
        self.add(deriv_vect)
        self.play(
            UpdateFromAlphaFunc(
                moving_vect,
                lambda m, a: m.interpolate(pre_vect, deriv_vect, a).set_fill(opacity=a),
                remover=True
            )
        )
        deriv_vect.set_fill(opacity=1)
        self.add(deriv_vect, ps_dot)
        self.wait(8)

        # Show equation
        rhs = VGroup(
            Tex("="),
            Matrix([["-y(t)"], ["x(t)"]], bracket_h_buff=SMALL_BUFF)
        )
        rhs.match_height(deriv_lhs)
        rhs.arrange(RIGHT)
        rhs.next_to(deriv_lhs, RIGHT)

        self.play(FadeIn(rhs))
        self.wait()
        for i in range(2):
            self.play(FlashAround(
                VGroup(deriv_lhs.get_entries()[i], rhs[1].get_entries()[i]),
                run_time=3,
                time_width=4,
            ))
        self.wait(2)

        # Write with a matrix
        deriv_lhs.generate_target()
        new_eq = VGroup(
            deriv_lhs.target,
            Tex("="),
            IntegerMatrix([[0, -1], [1, 0]], bracket_v_buff=MED_LARGE_BUFF),
            Matrix([["x(t)"], ["y(t)"]], bracket_h_buff=SMALL_BUFF),
        )
        new_eq[2].match_height(new_eq[0])
        new_eq[3].match_height(new_eq[0])
        new_eq.arrange(RIGHT)
        new_eq.to_corner(UR)

        self.play(
            MoveToTarget(deriv_lhs),
            MaintainPositionRelativeTo(deriv_label, deriv_lhs),
            ReplacementTransform(rhs[0], new_eq[1]),
            ReplacementTransform(rhs[1].get_brackets(), new_eq[3].get_brackets()),
            FadeIn(new_eq[2], scale=2),
            FadeTransform(rhs[1].get_entries()[1], new_eq[3].get_entries()[0]),
            FadeTransform(rhs[1].get_entries()[0], new_eq[3].get_entries()[1]),
        )
        self.wait(3)

        row_rect = SurroundingRectangle(new_eq[2].get_entries()[:2], buff=SMALL_BUFF)
        col_rect = SurroundingRectangle(new_eq[3].get_entries(), buff=SMALL_BUFF)
        both_rects = VGroup(row_rect, col_rect)
        both_rects.set_stroke(YELLOW, 2)

        self.play(*map(ShowCreation, both_rects))
        self.wait(3)
        self.play(row_rect.animate.move_to(new_eq[2].get_entries()[2:4]))
        self.wait(3)
        self.play(FadeOut(both_rects))

        # Write general form
        general_form = Tex(
            "{d \\over dt}",
            "\\vec{\\textbf{v} }",
            "(t)",
            "=",
            "\\textbf{M}",
            "\\vec{\\textbf{v} }",
            "(t)",
        )
        general_form.set_color_by_tex("d \\over dt", RED_B)
        general_form.set_color_by_tex("\\textbf{v}", GREY_B)
        general_form.scale(1.2)
        general_form.next_to(new_eq, DOWN, LARGE_BUFF)
        general_form.shift(0.5 * RIGHT)
        gf_rect = SurroundingRectangle(general_form, buff=MED_SMALL_BUFF)
        gf_rect.set_stroke(YELLOW, 2)

        equation.clear_updaters()
        self.play(
            FadeIn(general_form),
            FadeOut(equation),
        )
        self.wait()
        self.play(ShowCreation(gf_rect))
        self.wait(4 * TAU)

        # Fade all else out
        self.play(FadeOut(VGroup(gf_rect, general_form, new_eq, deriv_lhs, deriv_label)))
        self.wait(4 * TAU)
        print(self.num_plays)


class From2DTo1D(Scene):
    show_solution = False

    def construct(self):
        # (Setup vector equation)
        equation = get_2d_equation()
        equation.center()
        equation.to_edge(UP, buff=1.0)
        deriv, vect_sym, equals, matrix_mob, vect_sym2 = equation

        vect_sym.save_state()

        # (Setup plane)
        plane = NumberPlane(
            x_range=(-4, 4),
            y_range=(-2, 2),
            height=4,
            width=8,
        )
        plane.to_edge(DOWN)

        point = Point(plane.c2p(2, 0.5))
        vector = Arrow(plane.get_origin(), point.get_location(), buff=0)
        vector.set_color(YELLOW)

        # Show vector
        vect_sym.set_x(0)
        static_vect_sym = vect_sym.deepcopy()
        for entry in static_vect_sym.get_entries():
            entry[1:].set_opacity(0)
            entry[:1].move_to(entry)
        static_vect_sym.get_brackets().space_out_submobjects(0.7)
        vector.save_state()
        vector.put_start_and_end_on(
            static_vect_sym.get_corner(DL),
            static_vect_sym.get_corner(UR),
        )
        vector.set_opacity(0)

        self.add(plane, static_vect_sym)
        self.play(Restore(vector))
        self.wait()

        # Changing with time
        matrix = np.array([[0.5, -3], [1, -0.5]])

        def func(x, y):
            return 0.2 * np.dot([x, y], matrix.T)

        move_points_along_vector_field(point, func, plane)
        vector.add_updater(lambda m: m.put_start_and_end_on(
            plane.get_origin(), point.get_location(),
        ))
        deriv_vector = Vector(fill_color=RED, thickness=0.03)
        deriv_vector.add_updater(
            lambda m: m.put_start_and_end_on(
                plane.get_origin(),
                plane.c2p(*func(*plane.p2c(point.get_location()))),
            ).shift(vector.get_vector())
        )

        self.add(point)
        self.play(ReplacementTransform(static_vect_sym, vect_sym))
        self.wait(3)

        # Show matrix equation
        deriv_underline = Underline(VGroup(deriv, vect_sym.saved_state))
        deriv_underline.set_stroke(RED, 3)
        alt_line = deriv_underline.deepcopy()

        self.play(
            Restore(vect_sym),
            FadeIn(deriv),
        )
        self.wait()
        self.play(
            ShowCreation(deriv_underline),
        )
        self.play(
            VFadeIn(deriv_vector, rate_func=squish_rate_func(smooth, 0.8, 1.0)),
            UpdateFromAlphaFunc(
                alt_line,
                lambda m, a: m.put_start_and_end_on(
                    interpolate(deriv_underline.get_start(), deriv_vector.get_start(), a),
                    interpolate(deriv_underline.get_end(), deriv_vector.get_end(), a),
                ),
                remover=True,
            ),
        )
        self.wait(4)

        self.play(
            LaggedStartMap(FadeIn, equation[2:4], shift=RIGHT, lag_ratio=0.3),
            TransformFromCopy(
                equation[1], equation[4], path_arc=-45 * DEGREES,
                run_time=2,
                rate_func=squish_rate_func(smooth, 0.3, 1.0)
            )
        )
        self.wait(8)

        # Highlight equation
        deriv_rect = SurroundingRectangle(equation[:2])
        deriv_rect.set_stroke(RED, 2)
        rhs_rect = SurroundingRectangle(equation[-1])
        rhs_rect.set_stroke(YELLOW, 2)

        self.play(ShowCreation(deriv_rect))
        self.wait()
        self.play(ReplacementTransform(deriv_rect, rhs_rect, path_arc=-45 * DEGREES))

        # Draw vector field
        vector_field = VectorField(
            func, plane,
            magnitude_range=(0, 1.2),
            opacity=0.5,
            vector_config={"thickness": 0.02}
        )
        vector_field.sort(lambda p: get_norm(p - plane.get_origin()))

        self.add(vector_field, deriv_vector, vector)
        VGroup(vector, deriv_vector).set_stroke(BLACK, 5, background=True)
        self.play(
            FadeOut(rhs_rect),
            LaggedStartMap(GrowArrow, vector_field, lag_ratio=0)
        )
        self.wait(14)

        # flow_lines = AnimatedStreamLines(StreamLines(func, plane, step_multiple=0.25))
        # self.add(flow_lines)
        # self.wait(4)
        # self.play(VFadeOut(flow_lines))

        # self.wait(10)

        # Show solution
        equation.add(deriv_underline)
        mat_exp = get_matrix_exponential(
            [["a", "b"], ["c", "d"]],
            h_buff=0.75,
            v_buff=0.75,
        )
        mat_exp[1].set_color(TEAL)
        if self.show_solution:
            equation.generate_target()
            equation.target.to_edge(LEFT)
            implies = Tex("\\Rightarrow")
            implies.next_to(equation.target, RIGHT)
            solution = VGroup(
                equation[1].copy(),
                Tex("="),
                mat_exp,
                Matrix(
                    [["x(0)"], ["y(0)"]],
                    bracket_h_buff=SMALL_BUFF,
                    bracket_v_buff=SMALL_BUFF,
                )
            )
            solution[2].match_height(solution[0])
            solution[3].match_height(solution[0])
            solution.arrange(RIGHT, buff=MED_SMALL_BUFF)
            solution.next_to(implies, RIGHT, MED_LARGE_BUFF)
            solution.align_to(equation[1], DOWN)
            solution_rect = SurroundingRectangle(solution, buff=MED_SMALL_BUFF)

            self.play(
                MoveToTarget(equation),
            )
            self.play(LaggedStart(
                Write(implies),
                ShowCreation(solution_rect),
                TransformFromCopy(equation[4], solution[0], path_arc=30 * DEGREES),
            ))
            self.wait()
            self.play(LaggedStart(
                TransformFromCopy(equation[3], solution[2][1]),
                FadeIn(solution[2][0]),
                FadeIn(solution[2][2]),
                FadeIn(solution[1]),
                FadeIn(solution[3]),
                lag_ratio=0.1,
            ))
            self.wait(10)
            return
        else:
            # Show relation with matrix exp
            mat_exp.move_to(equation[0], DOWN)
            mat_exp.to_edge(RIGHT, buff=LARGE_BUFF)
            equation.generate_target()
            equation.target.to_edge(LEFT, buff=LARGE_BUFF)
            arrow1 = Arrow(equation.target.get_corner(UR), mat_exp.get_corner(UL), path_arc=-45 * DEGREES)
            arrow2 = Arrow(mat_exp.get_corner(DL), equation.target.get_corner(DR), path_arc=-45 * DEGREES)
            arrow1.shift(0.2 * RIGHT)

            self.play(MoveToTarget(equation))
            self.play(
                FadeIn(mat_exp[0::2]),
                TransformFromCopy(equation[3], mat_exp[1]),
                Write(arrow1, run_time=1),
            )
            self.wait(4)
            self.play(Write(arrow2, run_time=2))
            self.wait(4)

            normal_exp = Tex("e^{rt}")[0]
            normal_exp.set_height(1.0)
            normal_exp[1].set_color(BLUE)
            normal_exp.move_to(mat_exp)
            self.play(
                FadeTransformPieces(mat_exp, normal_exp),
                FadeOut(VGroup(arrow1, arrow2))
            )
            self.wait(2)

        # Transition to 1D
        max_x = 50
        mult = 50
        number_line = NumberLine((0, max_x), width=max_x)
        number_line.add_numbers()
        number_line.move_to(plane)
        number_line.to_edge(LEFT)
        nl = number_line
        nl2 = NumberLine((0, mult * max_x, mult), width=max_x)
        nl2.add_numbers()
        nl2.set_width(nl.get_width() * mult)
        nl2.shift(nl.n2p(0) - nl2.n2p(0))
        nl2.set_opacity(0)
        nl.add(nl2)

        new_equation = Tex(
            "{d \\over dt}", "x(t)", "=", "r \\cdot ", "x(t)",
        )
        new_equation[0][3].set_color(GREY_B)
        new_equation[1][2].set_color(GREY_B)
        new_equation[4][2].set_color(GREY_B)
        new_equation[3][0].set_color(BLUE)
        new_equation.match_height(equation)
        new_equation.move_to(equation)

        self.remove(point)
        vector.clear_updaters()
        deriv_vector.clear_updaters()

        self.remove(vector_field)
        plane.add(vector_field)
        self.add(number_line, deriv_vector, vector)
        self.play(
            normal_exp.animate.scale(0.5).to_corner(UR),
            # Plane to number line
            vector.animate.put_start_and_end_on(nl.n2p(0), nl.n2p(1)),
            deriv_vector.animate.put_start_and_end_on(nl.n2p(1), nl.n2p(1.5)),
            plane.animate.shift(nl.n2p(0) - plane.get_origin()).set_opacity(0),
            FadeIn(number_line, rate_func=squish_rate_func(smooth, 0.5, 1)),
            # Equation
            TransformMatchingShapes(equation[0], new_equation[0]),
            Transform(equation[1].get_entries()[0], new_equation[1]),
            FadeTransform(equation[2], new_equation[2]),
            FadeTransform(equation[3], new_equation[3]),
            FadeTransform(equation[4].get_entries()[0], new_equation[4]),
            FadeOut(equation[1].get_brackets()),
            FadeOut(equation[1].get_entries()[1]),
            FadeOut(equation[4].get_brackets()),
            FadeOut(equation[4].get_entries()[1]),
            FadeOut(deriv_underline),
            run_time=2,
        )

        vt = ValueTracker(1)
        vt.add_updater(lambda m, dt: m.increment_value(0.2 * dt * m.get_value()))

        vector.add_updater(lambda m: m.put_start_and_end_on(nl.n2p(0), nl.n2p(vt.get_value())))
        deriv_vector.add_updater(lambda m: m.set_width(0.5 * vector.get_width()).move_to(vector.get_right(), LEFT))

        self.add(vt)
        self.wait(11)
        self.play(
            number_line.animate.scale(0.3, about_point=nl.n2p(0)),
        )
        self.wait(4)
        number_line.generate_target()
        number_line.target.scale(0.1, about_point=nl.n2p(0)),
        number_line.target[-1].set_opacity(1)
        self.play(
            MoveToTarget(number_line)
        )
        self.wait(11)
        self.play(number_line.animate.scale(0.2, about_point=nl.n2p(0)))
        self.wait(10)


class SchroedingersEquationIntro(Scene):
    def construct(self):
        # Show equation
        title = Text("Schrödinger equation", font_size=72)
        title.to_edge(UP)
        self.add(title)

        t2c = {
            "|\\psi \\rangle": BLUE,
            "{H}": GREY_A,
            "=": WHITE,
            "i\\hbar": WHITE,
        }
        original_equation = Tex(
            "i\\hbar \\frac{\\partial}{\\partial t} |\\psi \\rangle = {H} |\\psi \\rangle",
            tex_to_color_map=t2c
        )
        equation = Tex(
            "\\frac{\\partial}{\\partial t} |\\psi \\rangle = \\frac{1}{i\\hbar} {H} |\\psi \\rangle",
            tex_to_color_map=t2c
        )
        VGroup(original_equation, equation).scale(1.5)

        psis = original_equation.get_parts_by_tex("\\psi")
        state_label = TexText("State of a system \\\\ as a vector", font_size=36)
        state_label.next_to(psis, DOWN, buff=1.5)
        state_label.shift(0.5 * RIGHT)
        state_arrows = VGroup(*(Arrow(state_label, psi) for psi in psis))
        state_label.match_color(psis[0])
        state_arrows.match_color(psis[0])
        psis.set_color(WHITE)

        randy = Randolph(height=2.0, color=BLUE_C)
        randy.to_corner(DL)
        randy.set_opacity(0)

        self.play(Write(original_equation, run_time=3))
        self.wait()
        self.play(
            randy.animate.set_opacity(1).change("horrified", original_equation)
        )
        self.play(Blink(randy))
        self.play(
            randy.change("pondering", state_label),
            psis.animate.match_color(state_label),
            FadeIn(state_label, 0.25 * DOWN),
            *map(GrowArrow, state_arrows),
        )
        self.wait()
        self.play(Blink(randy))
        self.wait()

        self.play(
            ReplacementTransform(original_equation[1:4], equation[0:3]),
            Write(equation[3]),
            ReplacementTransform(original_equation[0], equation[4], path_arc=90 * DEGREES),
            ReplacementTransform(original_equation[4:], equation[5:]),
            state_arrows.animate.become(
                VGroup(*(Arrow(state_label, psi) for psi in equation.get_parts_by_tex("\\psi")))
            ),
            randy.change("hesitant", equation)
        )
        self.play(FlashAround(equation[0], time_width=2, run_time=2))
        self.play(Blink(randy))

        mat_rect = SurroundingRectangle(equation[3:6], buff=0.05, color=TEAL)
        mat_label = Text("A certain matrix", font_size=36)
        mat_label.next_to(mat_rect, UP)
        mat_label.match_color(mat_rect)
        self.play(
            ShowCreation(mat_rect),
            FadeIn(mat_label, 0.25 * UP),
        )
        self.wait()
        self.play(randy.change("confused", equation))
        self.play(Blink(randy))
        self.wait()

        # Complicating factors
        psi_words = TexText(
            "Often this is a function.\\\\",
            "(But whatever functions are really\\\\just infinite-dimensional vectors)"
        )
        psi_words[0].match_width(psi_words[1], about_edge=DOWN)
        psi_words[1].shift(0.1 * DOWN)
        psi_words.scale(0.75)
        psi_words.move_to(state_label, UP)
        psi_words[0].set_color(RED)
        psi_words[1].set_color(RED_D)

        mat_line = Line(LEFT, RIGHT)
        mat_line.set_stroke(RED, 5)
        mat_line.replace(mat_label.get_part_by_text("matrix"), dim_to_match=0)

        operator_word = Text("operator", font_size=36)
        operator_word.next_to(mat_label, UP, buff=SMALL_BUFF)
        operator_word.align_to(mat_line, LEFT)
        operator_word.set_color(RED)

        complex_valued = Text("Complex-valued", font_size=30)
        complex_valued.set_color(RED)
        complex_valued.next_to(equation, RIGHT)
        complex_valued.to_edge(RIGHT)
        cv_arrow = Arrow(complex_valued, equation, fill_color=RED)

        self.play(LaggedStart(
            FadeOut(state_label),
            FadeIn(psi_words),
            ShowCreation(mat_line),
            Write(operator_word, run_time=1),
            GrowArrow(cv_arrow),
            FadeIn(complex_valued),
            randy.change("horrified", equation),
            lag_ratio=0.5
        ))
        self.play(Blink(randy))
        self.wait()


class SimpleDerivativeOfExp(TeacherStudentsScene):
    def construct(self):
        eq = Tex(
            "{d \\over dt}", "e^{rt}", "=", "r", "e^{rt}",
        )
        eq.set_color_by_tex("r", TEAL, substring=False)
        for part in eq.get_parts_by_tex("e^{rt}"):
            part[1].set_color(TEAL)

        s0, s1, s2 = self.students
        morty = self.teacher
        bubble = s2.get_bubble(eq)

        self.play(
            s2.change("pondering", eq),
            FadeIn(bubble, lag_ratio=0.2),
        )
        self.play(
            LaggedStart(
                s0.change("hesitant", eq),
                s1.change("erm", eq),
                morty.change("tease"),
            ),
            Write(eq[:2])
        )
        self.wait()
        self.play(
            TransformFromCopy(*eq.get_parts_by_tex("e^{rt}"), path_arc=45 * DEGREES),
            Write(eq.get_part_by_tex("=")),
            *(
                pi.animate.look_at(eq[4])
                for pi in self.pi_creatures
            )
        )
        self.play(
            FadeTransform(eq[4][1].copy(), eq[3][0], path_arc=90 * DEGREES)
        )
        self.play(
            LaggedStart(
                s0.change("thinking"),
                s1.change("tease"),
            )
        )
        self.wait(2)

        rect = ScreenRectangle(height=3.5)
        rect.set_fill(BLACK, 1)
        rect.set_stroke(BLUE_B, 2)
        rect.to_corner(UR)
        self.play(
            morty.change("raise_right_hand", rect),
            self.change_students("pondering", "pondering", "pondering", look_at=rect),
            FadeIn(rect, UP)
        )
        self.wait(10)


class ETitleCard(Scene):
    def construct(self):
        title = TexText("A brief review of $e$\\\\and exponentials")
        title.scale(2)
        self.add(title)


class GraphAndHistoryOfExponential(Scene):
    def construct(self):
        # Setup
        axes = Axes(
            x_range=(0, 23, 1),
            y_range=(0, 320, 10),
            height=160,
            width=13,
            axis_config={"include_tip": False}
        )
        axes.y_axis.add(*(axes.y_axis.get_tick(x, size=0.05) for x in range(20)))
        axes.to_corner(DL)
        r = 0.25
        exp_graph = axes.get_graph(lambda t: np.exp(r * t))
        exp_graph.set_stroke([BLUE_E, BLUE, YELLOW])
        graph_template = exp_graph.copy()
        graph_template.set_stroke(width=0)
        axes.add(graph_template)

        equation = self.get_equation()
        solution = Tex("x({t}) = e^{r{t} }", tex_to_color_map={"{t}": GREY_B, "r": BLUE})
        solution.next_to(equation, DOWN, MED_LARGE_BUFF)

        self.add(axes)
        self.add(axes.get_x_axis_label("t"))
        self.add(axes.get_y_axis_label("x"))
        self.add(equation)

        curr_time = self.time
        exp_graph.add_updater(lambda m: m.pointwise_become_partial(
            graph_template, 0, (self.time - curr_time) / 20,
        ))
        dot = Dot(color=BLUE_B, radius=0.04)
        dot.add_updater(lambda d: d.move_to(exp_graph.get_end()))
        vect = Arrow(DOWN, UP, fill_color=YELLOW, thickness=0.025)
        vect.add_updater(lambda v: v.put_start_and_end_on(
            axes.get_origin(),
            axes.y_axis.get_projection(exp_graph.get_end()),
        ))
        h_line = always_redraw(lambda: DashedLine(
            vect.get_end(), dot.get_left(),
            stroke_width=1,
            stroke_color=GREY_B,
        ))
        v_line = always_redraw(lambda: DashedLine(
            axes.x_axis.get_projection(dot.get_bottom()),
            dot.get_bottom(),
            stroke_width=1,
            stroke_color=GREY_B,
        ))

        def stretch_axes(factor):
            axes.generate_target(use_deepcopy=True)
            axes.target.stretch(factor, 1, about_point=axes.get_origin()),
            axes.target.x_axis.stretch(1 / factor, 1, about_point=axes.get_origin()),
            self.play(MoveToTarget(axes))

        self.add(exp_graph, dot, vect, h_line, v_line)
        # equation.scale(2, about_edge=UP)###
        self.wait(2)

        # Highlight equation parts
        index = equation.index_of_part_by_tex("=")
        lhs = equation[:index]
        rhs = equation[index + 1:]

        for part in (rhs, lhs):
            self.play(FlashAround(part, time_width=3, run_time=2))
            self.wait()
        # self.wait(4)

        stretch_axes(0.2)

        self.wait(6)
        stretch_axes(0.23)
        self.wait(4)

        exp_graph.clear_updaters()
        exp_graph.become(graph_template)
        exp_graph.set_stroke(width=3)
        self.add(exp_graph)
        self.play(FadeOut(vect), FadeOut(dot))
        self.wait()

        # Write exponential growth
        words = Text("Exponential growth")
        words.next_to(axes.c2p(0, 0), UR, buff=0)
        original_words = words.deepcopy()
        original_words.set_opacity(0)

        def func(p):
            t, x = axes.p2c(p)
            angle = axes.angle_of_tangent(t, exp_graph)
            vect = rotate_vector(RIGHT, angle + PI / 2)
            graph_point = axes.input_to_graph_point(t, exp_graph)
            y = (axes.y_axis.get_projection(p) - axes.get_origin())[1]
            return graph_point + y * vect

        fill_tracker = ValueTracker(0)
        words.add_updater(lambda m: m.become(original_words).set_fill(opacity=fill_tracker.get_value()).apply_function(func))

        self.add(words)
        self.play(
            ApplyMethod(original_words.next_to, axes.c2p(14, 0), UP, SMALL_BUFF, run_time=2),
            fill_tracker.animate.set_value(1),
        )
        self.remove(original_words, fill_tracker)
        words.clear_updaters()
        self.wait()

        # Introduce 2.71828
        solution_with_number = Tex(
            "{d \\over dt}",
            "(2.71828...)^{", "r", "t", "}",
            "=",
            "r", "\\cdot",
            "(2.71828...)^{", "r", "t", "}",
            tex_to_color_map={
                "2.71828...": TEAL,
            }
        )
        solution_with_e = Tex(
            "{d \\over dt}",
            "{e}^{", "r", "t", "}",
            "=",
            "r", "\\cdot",
            "{e}^{", "r", "t", "}",
            tex_to_color_map={
                "{e}": TEAL,
            }
        )
        for eq in solution_with_number, solution_with_e:
            eq.set_color_by_tex("r", BLUE, substring=False)
            eq.set_color_by_tex("t", GREY_B, substring=False)
            eq.next_to(equation, DOWN, buff=MED_LARGE_BUFF)
        lhs = solution_with_number[:6]
        lhs.save_state()
        lhs.match_x(equation)

        self.play(FadeIn(lhs[1:], DOWN))
        self.wait()
        self.play(Write(lhs[0]))

        dot1 = Dot(radius=0.05, color=RED)
        dot2 = dot1.copy()
        for sec_dot in dot1, dot2:
            sec_dot.x_tracker = ValueTracker(15)
            sec_dot.add_updater(lambda d: d.move_to(axes.input_to_graph_point(
                d.x_tracker.get_value(), exp_graph
            )))
        line = Line(LEFT, RIGHT, stroke_color=GREY_A, stroke_width=2)
        line.add_updater(lambda l: l.put_start_and_end_on(
            dot1.get_center(), dot2.get_center()
        ).set_length(10))

        dot2.x_tracker.set_value(17)
        dot1.x_tracker.set_value(15)
        self.play(
            FadeOut(words),
            FadeIn(dot1),
            FadeIn(dot2),
            FadeIn(line),
        )
        self.play(
            dot2.x_tracker.animate.set_value(15 + 1e-6),
            run_time=3
        )
        self.play(
            dot1.x_tracker.animate.set_value(18),
            dot2.x_tracker.animate.set_value(18 + 1e-6),
            run_time=3
        )
        self.play(
            FadeOut(dot1),
            FadeOut(dot2),
            FadeOut(line),
        )
        self.wait()

        self.play(Restore(lhs))
        self.play(
            FadeIn(solution_with_number[6]),
            TransformFromCopy(lhs[1:], solution_with_number[8:], path_arc=30 * DEGREES),
        )
        self.play(TransformFromCopy(
            solution_with_number[12], solution_with_number[7],
            path_arc=-90 * DEGREES,
        ))
        self.wait()

        # Historical letters
        correspondants = Group()
        for name1, name2, letter, year in [("Leibniz", "Huygens", "b", "1690"), ("Euler", "Goldbach", "e", "1731")]:
            im1 = ImageMobject(name1, height=2.5)
            im2 = ImageMobject(name2, height=2.5)

            lines = VGroup(*(Line(LEFT, RIGHT) for x in range(11)))
            lines.set_width(1.6)
            lines.arrange(DOWN, buff=0.2)
            lines.set_stroke(GREY_A, 2)
            lines[-1].stretch(0.5, 0, about_edge=LEFT)

            eq = Tex(letter, "= 2.71828\\dots")
            eq[0].scale(1.5, about_edge=RIGHT)
            eq[0].align_to(eq[1], DOWN)
            eq.set_color(TEAL)
            eq.match_width(lines)
            eq.move_to(lines[5])
            lines.remove(*lines[4:7])
            lines.add(eq)
            box = SurroundingRectangle(lines, buff=0.25)
            box.set_stroke(GREY_C, 2)
            note = VGroup(box, lines)
            note.match_height(im1)

            group = Group()
            arrow = Vector(0.5 * RIGHT, fill_color=GREY_A)
            group.add(im1, arrow, note, arrow.copy(), im2)
            group.arrange(RIGHT)
            note.align_to(im1, UP)
            group.next_to(solution_with_number, DOWN, buff=LARGE_BUFF)

            for im, name, index in [(im1, name1, 0), (im2, name2, 4)]:
                name_label = Text(name, font_size=24)
                name_label.set_color(GREY_A)
                name_label.next_to(im, DOWN)
                rect = SurroundingRectangle(im, buff=0, stroke_color=GREY_A, stroke_width=1)
                group.replace_submobject(index, Group(im, rect, name_label))

            date = Text(year, font_size=24)
            date.next_to(box, UP)
            note.add(date)
            group.eq = eq
            group.eq.set_opacity(0)

            correspondants.add(group)

        b_group, e_group = correspondants
        self.play(LaggedStartMap(FadeIn, b_group, shift=0.5 * UP, lag_ratio=0.2, run_time=1))
        b_group.eq.set_fill(opacity=1)
        self.play(Write(b_group.eq, run_time=1))
        self.wait()
        self.play(
            LaggedStartMap(FadeOut, b_group, shift=0.5 * UP),
            LaggedStartMap(FadeIn, e_group, shift=0.5 * UP),
        )
        e_group.eq.set_fill(opacity=1)
        self.play(Write(e_group.eq, run_time=1))
        self.wait()

        # Euler's book
        e_usage = e_group.eq.copy()
        book = ImageMobject("Introductio_in_Analysin_infinitorum")
        book.match_width(e_group[0][0])
        book.move_to(e_group[2][0])
        date = Text("1748", font_size=24)
        date.next_to(book, UP, SMALL_BUFF)

        self.play(
            LaggedStartMap(FadeOut, e_group[2:], shift=0.5 * RIGHT),
            FadeIn(book),
            FadeIn(date),
            e_usage.animate.scale(1.4).next_to(book, RIGHT, aligned_edge=UP).shift(0.25 * DOWN)
        )
        self.wait()

        pi_eq = Tex("\\pi = 3.1415\\dots", fill_color=GREEN, font_size=36)
        func_eq = Tex("f(x)", font_size=36)
        pi_eq.next_to(e_usage, DOWN, buff=0.5, aligned_edge=LEFT)
        func_eq.next_to(pi_eq, DOWN, buff=0.5, aligned_edge=LEFT)

        self.play(Write(pi_eq, run_time=1))
        self.wait()
        self.play(Write(func_eq, run_time=1))
        self.wait()

        # Final notation
        self.play(FadeTransformPieces(solution_with_number, solution_with_e))
        self.wait()

    def get_equation(self, r="r"):
        return get_1d_equation(r).to_edge(UP)


class BernoullisThoughts(Scene):
    def construct(self):
        im = ImageMobject("Jacob_Bernoulli", height=4)
        name = Text("Jacob Bernoulli")
        name.match_width(im)
        name.next_to(im, DOWN, SMALL_BUFF)
        jacob = Group(im, name)
        jacob.to_corner(DR)

        bubble = ThoughtBubble(height=3.5, width=5)
        bubble.pin_to(jacob)
        bubble.shift(0.25 * RIGHT + 0.75 * DOWN)

        dollars = Text("$$$")
        dollars.set_height(1)
        dollars.set_color(GREEN)
        dollars.move_to(bubble.get_bubble_center())

        dollar = dollars[0].copy()
        continuous_money = VGroup(*(
            dollar.copy().shift(smooth(x) * RIGHT * 2.0 + np.exp(smooth(x) - 1) * UP)
            for x in np.linspace(0, 1, 300)
        ))
        continuous_money.set_fill(opacity=0.1)
        continuous_money.center()
        continuous_money.move_to(dollars)

        self.play(FadeIn(jacob, shift=RIGHT))
        self.play(
            ShowCreation(bubble),
            Write(dollars)
        )
        self.wait()
        self.add(continuous_money)
        self.play(
            FadeOut(dollars),
            Write(continuous_money, stroke_color=GREEN, stroke_width=1, run_time=3),
        )
        self.play(
            FadeOut(continuous_money, lag_ratio=0.01),
            FadeIn(dollars)
        )
        self.wait()

        limit = Tex("\\left(1 + \\frac{r}{n}\\right)^{nt}")
        limit.move_to(bubble.get_bubble_center())
        limit.scale(1.5)

        self.play(
            FadeOut(dollars),
            FadeIn(limit),
        )
        self.wait()


class CompoundInterestPopulationAndEpidemic(Scene):
    def construct(self):
        N = 16000
        points = 2 * np.random.random((N, 3)) - 1
        points[:, 2] = 0
        points = points[[get_norm(p) < 1 for p in points]]
        points *= 2 * FRAME_WIDTH
        points = np.array(list(sorted(
            points, key=lambda p: get_norm(p) + 0.5 * random.random()
        )))

        dollar = Tex("\\$")
        dollar.set_fill(GREEN)
        person = SVGMobject("person")
        person.set_fill(GREY_B, 1)
        virus = SVGMobject("virus")
        virus.set_fill([RED, RED_D])
        virus.remove(virus[1:])
        virus[0].set_points(virus[0].get_subpaths()[0])
        templates = [dollar, person, virus]
        mob_height = 0.25

        for mob in templates:
            mob.set_stroke(BLACK, 1, background=True)
            mob.set_height(mob_height)

        dollars, people, viruses = groups = [
            VGroup(*(mob.copy().move_to(point) for point in points))
            for mob in templates
        ]

        dollars.set_submobjects(dollars[:20])
        people.set_submobjects(people[:500])

        start_time = self.time

        def get_n():
            time = self.time - start_time
            return int(math.exp(0.75 * time))

        def update_group(group):
            group.set_opacity(0)
            group[:get_n()].set_opacity(0.9)

        def update_height(group, alpha):
            for mob in group:
                mob.set_height(max(alpha * mob_height, 1e-4))

        for group in groups:
            group.add_updater(update_group)

        frame = self.camera.frame
        frame.set_height(2)
        frame.add_updater(lambda m, dt: m.set_height(m.get_height() * (1 + 0.2 * dt)))
        self.add(frame)

        self.add(dollars)
        self.wait(3)
        self.play(
            UpdateFromAlphaFunc(people, update_height),
            UpdateFromAlphaFunc(dollars, update_height, rate_func=lambda t: smooth(1 - t), remover=True),
        )
        self.wait(4)
        self.play(
            UpdateFromAlphaFunc(viruses, update_height),
            UpdateFromAlphaFunc(people, update_height, rate_func=lambda t: smooth(1 - t), remover=True),
        )
        self.wait(4)


class CovidPlot(ExternallyAnimatedScene):
    pass


class Compare1DTo2DEquations(Scene):
    def construct(self):
        eq1d = get_1d_equation()
        eq2d = get_2d_equation()
        eq1d.match_height(eq2d)

        equations = VGroup(eq1d, eq2d)
        equations.arrange(DOWN, buff=2.0)
        equations.to_edge(LEFT, buff=1.0)

        solutions = VGroup(
            Tex("e^{rt}", tex_to_color_map={"r": BLUE}),
            get_matrix_exponential(
                [["a", "b"], ["c", "d"]],
                h_buff=0.75,
                v_buff=0.5,
                bracket_h_buff=0.25,
            )
        )
        solutions[1][1].set_color(TEAL)
        solutions[0].scale(2)
        arrows = VGroup()
        for eq, sol in zip(equations, solutions):
            sol.next_to(eq[-1], RIGHT, index_of_submobject_to_align=0)
            sol.set_x(4)
            arrows.add(Arrow(eq, sol[0], buff=0.5))

        sol0, sol1 = solutions
        sol0.save_state()
        sol0.center()
        self.add(sol0)
        self.wait()
        self.play(
            Restore(sol0),
            TransformFromCopy(Arrow(eq1d, sol0, fill_opacity=0), arrows[0]),
            FadeIn(eq1d),
        )
        self.wait(2)
        self.play(
            TransformMatchingShapes(sol0.copy(), sol1, fade_transform_mismatches=True),
        )
        self.wait()
        self.play(
            TransformFromCopy(*arrows),
            LaggedStart(*(
                FadeTransform(m1.copy(), m2)
                for m1, m2 in zip(
                    [eq1d[:2], eq1d[2:5], eq1d[5], eq1d[6], eq1d[7:]],
                    eq2d
                )
            ), lag_ratio=0.05)
        )
        self.wait()


class EVideoWrapper(VideoWrapper):
    title = "Video on $e^x$"


class ManyExponentialForms(ExternallyAnimatedScene):
    pass


class EBaseMisconception(Scene):
    def construct(self):
        randy = Randolph()
        randy.flip()
        randy.to_corner(DR)
        self.add(randy)
        self.play(
            PiCreatureSays(
                randy, TexText("This function is\\\\about about $e$", tex_to_color_map={"$e$": BLUE}),
                target_mode="thinking",
                bubble_config={"height": 3, "width": 4},
            )
        )
        self.play(Blink(randy))
        self.wait(2)
        self.play(randy.change("tease"))
        self.play(Blink(randy))
        cross = Cross(randy.bubble.content, stroke_width=[0, 5, 5, 5, 0])
        for line in cross:
            line.insert_n_curves(10)
        cross.scale(1.25)
        self.play(
            ShowCreation(cross),
            randy.change("guilty")
        )
        self.wait()


class OneFinalPoint(TeacherStudentsScene):
    def construct(self):
        self.teacher_says(
            TexText("One final point\\\\about one-dimension"),
            bubble_config={"height": 3, "width": 4},
        )
        self.play_student_changes(
            "happy", "hesitant", "tease",
            added_anims=[self.teacher.animate.look_at(self.students[2])]
        )
        self.wait(3)


class ManySolutionsDependingOnInitialCondition(VideoWrapper):
    def construct(self):
        # Setup graphs and equations
        axes = Axes(
            x_range=(0, 12),
            y_range=(0, 10),
            width=12,
            height=6,
        )
        axes.add(axes.get_x_axis_label("t"))
        axes.add(axes.get_y_axis_label("x"))
        axes.x_axis.add_numbers()
        equation = get_1d_equation("0.3").to_edge(UP)

        def get_graph(x0, r=0.3):
            return axes.get_graph(lambda t: np.exp(r * t) * x0)

        step = 0.05
        graphs = VGroup(*(
            get_graph(x0)
            for x0 in np.arange(step, axes.y_range[1], step)
        ))
        graphs.set_submobject_colors_by_gradient(BLUE_E, BLUE, TEAL, YELLOW)
        graphs.set_stroke(width=1)
        graph = get_graph(1)
        graph.set_color(BLUE)

        solution = Tex(
            "x({t}) = e^{0.3 {t} } \\cdot x_0",
            tex_to_color_map={"{t}": GREY_B, "0.3": BLUE, "=": WHITE},
        )
        solution.next_to(equation, DOWN, LARGE_BUFF)
        solution.shift(0.25 * RIGHT)
        solution[-1][1:].set_fill(MAROON_B)

        group = VGroup(solution, equation)
        group.set_stroke(BLACK, 5, background=True)
        group.shift(2 * LEFT)

        labels = VGroup(
            Text("Differential equation", font_size=30),
            Text("Solution", font_size=30),
        )
        labels.set_stroke(BLACK, 5, background=True)
        arrows = VGroup()
        for label, eq in zip(labels, [equation, solution[:-1]]):
            label.next_to(eq, RIGHT, buff=1.5)
            label.align_to(labels[0], LEFT)
            arrows.add(Arrow(label, eq))

        eq_label, sol_label = labels
        eq_arrow, sol_arrow = arrows

        # Show many possible representations
        alt_rhs = Tex(
            "&= 2^{(0.4328\\dots) t }\\\\",
            "&= 3^{(0.2730\\dots) t }\\\\",
            "&= 4^{(0.2164\\dots) t }\\\\",
            "&= 5^{(0.1864\\dots) t }\\\\",
        )
        alt_rhs.next_to(solution.get_part_by_tex("="), DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        alt_rhs.set_stroke(BLACK, 5, background=True)

        choice_words = Text("This is a choice!", color=YELLOW, font_size=24)
        choice_words.next_to(solution[:-1], RIGHT, MED_LARGE_BUFF)
        choice_words.set_stroke(BLACK, 5, background=True)

        self.add(axes)
        self.add(solution[:-1])

        self.play(ShowCreation(graph, run_time=3))
        self.wait()
        self.play(FlashAround(solution.get_part_by_tex("e"), time_width=2, buff=0.05, run_time=2))
        self.wait()
        self.play(LaggedStartMap(FadeIn, alt_rhs, shift=0.5 * DOWN, lag_ratio=0.6, run_time=5))
        self.wait()
        self.play(
            alt_rhs.animate.set_opacity(0.4),
            FlashAround(solution[4:-1]),
            Write(choice_words, run_time=1)
        )
        self.wait()
        self.play(
            FadeTransform(solution[:3].copy(), equation[2:5]),  # x(t)
            FadeTransform(solution[:3].copy(), equation[7:10]),  # x(t)
            FadeTransform(solution[3].copy(), equation[5]),  # =
            FadeIn(equation[:2]),
            FadeOut(choice_words),
        )
        self.play(FadeTransform(solution[5].copy(), equation[6]))  # r
        self.wait()
        self.play(alt_rhs.animate.set_opacity(1.0))
        self.wait()
        self.play(LaggedStartMap(FadeOut, alt_rhs, shift=DOWN))
        for label, arrow in zip(labels, arrows):
            self.play(
                FadeIn(label, 0.5 * RIGHT),
                GrowArrow(arrow),
            )
            self.wait()

        # Just one solution
        one = Text("One", font_size=30)
        many = Text("out of many", font_size=30)
        one.move_to(sol_label.get_corner(UL), LEFT)
        sol_label.generate_target()
        group = VGroup(
            VGroup(one, sol_label.target).arrange(RIGHT, buff=MED_SMALL_BUFF, aligned_edge=DOWN),
            many
        )
        group.arrange(DOWN, buff=0.15)
        group.move_to(sol_label, LEFT)

        dot = Dot().scale(2 / 3)
        dot.move_to(axes.c2p(0, 0))

        self.add(graphs, graph, equation, solution[:-1], arrows, labels, one, many)
        self.play(
            MoveToTarget(sol_label, rate_func=squish_rate_func(smooth, 0, 0.5)),
            FadeIn(one),
            FadeIn(many),
            ShowIncreasingSubsets(graphs),
            dot.animate.move_to(axes.c2p(0, 10)),
            run_time=4,
        )
        self.wait()

        sol_label = VGroup(one, sol_label, many)

        # Initial conditions
        ic_label = Text("Initial conditions")
        ic_label.rotate(90 * DEGREES)
        ic_label.next_to(axes.y_axis, LEFT)

        x0_tracker = ValueTracker(1)
        get_x0 = x0_tracker.get_value
        x0_label = VGroup(
            Tex("x_0 = ", font_size=36),
            DecimalNumber(1, font_size=36),
        )
        x0_label.set_color(MAROON_B)
        x0_label[1].match_height(x0_label[0])
        x0_label[1].next_to(x0_label[0][0][2], RIGHT, SMALL_BUFF)
        x0_label.add_updater(lambda m: m[1].set_value(get_x0()).set_color(MAROON_B))
        sv = x0_label.get_width() + MED_SMALL_BUFF
        x0_label.add_updater(lambda m: m.move_to(axes.c2p(0, get_x0()), LEFT).shift(sv * LEFT))

        self.play(
            Write(ic_label),
            run_time=1,
        )
        self.play(
            dot.animate.move_to(axes.c2p(0, 1)),
            graphs.animate.set_stroke(opacity=0.5),
            run_time=2,
        )
        self.wait()

        graph.add_updater(lambda g: g.match_points(get_graph(get_x0())))
        dot.add_updater(lambda d: d.move_to(axes.c2p(0, get_x0())))

        rect = BackgroundRectangle(x0_label)
        rect.set_fill(BLACK, 1)
        self.add(x0_label, rect, ic_label)
        self.play(
            FadeOut(rect),
            FadeOut(ic_label),
            self.camera.frame.animate.shift(0.6 * LEFT)
        )
        self.wait()
        self.play(
            x0_tracker.animate.set_value(0.1),
        )
        self.wait()
        for x in [5, 9]:
            self.play(
                x0_tracker.animate.set_value(x),
                run_time=5
            )
            self.wait()
            self.play(
                x0_tracker.animate.set_value(1),
                run_time=3,
            )
        self.wait()

        # Show general solution
        new_sol_label = Text("General solution", font_size=36)
        new_sol_label.move_to(sol_label, LEFT)

        self.play(
            TransformFromCopy(x0_label[0][0][:2], solution[-1]),
            sol_arrow.animate.become(Arrow(sol_label, solution))
        )
        self.play(
            FadeOut(sol_label),
            FadeIn(new_sol_label),
        )
        self.wait()

        rhs = solution[4:].copy()
        rhs_rect = SurroundingRectangle(rhs, stroke_width=2)
        rhs.generate_target()
        index = equation.index_of_part(equation.get_parts_by_tex("x")[1])
        rhs.target.move_to(equation[index], LEFT)
        rhs.target.shift(0.05 * UP)

        self.play(ShowCreation(rhs_rect))
        self.play(
            MoveToTarget(rhs),
            MaintainPositionRelativeTo(rhs_rect, rhs),
            FadeOut(equation[index:]),
            FadeOut(eq_arrow),
            FadeOut(eq_label),
        )
        self.play(FadeOut(rhs_rect))
        self.wait()
        self.play(x0_tracker.animate.set_value(2.5), run_time=2)
        self.play(x0_tracker.animate.set_value(0.5), run_time=2)
        self.wait()
        self.play(
            FadeOut(rhs),
            FadeIn(equation[index:]),
            FadeIn(eq_arrow),
            FadeIn(eq_label),
        )
        self.wait()

        # Emphasize solution vs action
        exp_rect = SurroundingRectangle(solution[4:7], buff=0.05, stroke_width=2)

        words1 = Text("Don't think of this\n\n as a solution", font_size=30)
        words2 = Text(
            "It's something which\n\nacts on an initial condition\n\nto give a solution",
            t2s={"acts": ITALIC},
            t2c={"acts": YELLOW, "initial condition": MAROON_B},
            font_size=30
        )
        for words in [words1, words2]:
            words.next_to(exp_rect, DOWN, MED_LARGE_BUFF)
            words.set_stroke(BLACK, 5, background=True)

        self.play(
            ShowCreation(exp_rect),
            Write(words1, run_time=1)
        )
        self.wait(2)
        self.play(
            FadeOut(words1),
            FadeIn(words2),
        )
        self.wait(2)


class ExoticExponentsWithEBase(Scene):
    def construct(self):
        # Write exotic exponents
        exps = VGroup(
            Tex("e", "^{it}"),
            get_matrix_exponential([[3, 1], [4, 1]]),
            Tex("e", "^{((i + j + k) / \\sqrt{3})t}"),
            Tex("e", "^{\\left(\\frac{\\partial}{\\partial x}\\right)t}"),
        )
        for i in (0, 2, 3):
            exps[i][0].scale(1.25)
            exps[i][0].move_to(exps[i][1:].get_corner(DL), UR)
        exps[1].scale(exps[0][0].get_height() / exps[1][0].get_height())
        exps.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)

        labels = VGroup(
            Text("Complex numbers", font_size=30),
            Text("Matrices", font_size=30),
            Text("Quaternions", font_size=30),
            Text("Operators", font_size=30),
        )
        labels.set_submobject_colors_by_gradient(BLUE_B, BLUE_C)
        for label, exp in zip(labels, exps):
            label.move_to(exp, DL)
            label.align_to(labels[0], LEFT)
        labels.set_x(-2)

        exps.set_x(labels.get_right()[0] + 3)

        arrow = Arrow(labels.get_corner(UL), labels.get_corner(DL), buff=0)
        arrow.shift(0.5 * LEFT)
        arrow.set_color(TEAL)
        exotic_label = TexText("More\\\\exotic\\\\exponents", alignment="")
        exotic_label.next_to(arrow, LEFT)
        VGroup(arrow, exotic_label).set_opacity(0)

        e_mobs = VGroup()
        for label, exp in zip(labels, exps):
            e_mobs.add(exp[0])
            self.play(
                FadeIn(label),
                FadeIn(exp[0]),
                GrowFromPoint(exp[1:], exp[0].get_center()),
                exotic_label.animate.set_opacity(1),
                arrow.animate.set_opacity(1),
            )
        self.wait()
        self.play(LaggedStart(*(
            FlashAround(e, buff=0.05, time_width=3, run_time=2)
            for e in e_mobs
        ), lag_ratio=0.1))
        self.wait()

        # Analytic number theory exception
        zeta = Tex("\\sum_{n = 1}^\\infty \\frac{1}{n^s}")
        zeta.move_to(exps[0], LEFT)

        self.play(
            FadeOut(exps[0]),
            FadeIn(zeta),
        )
        self.wait()
        self.play(
            FadeOut(zeta),
            FadeIn(exps[0]),
        )
        self.wait()

        # Other bases
        n_mob_groups = VGroup()
        for n in ["2", "5", "\\pi", "1{,}729"]:
            n_mobs = VGroup()
            for e_mob in e_mobs:
                n_mob = Tex(n)
                n_mob.match_height(e_mob)
                n_mob.scale(1.2)
                n_mob.move_to(e_mob, UR)
                n_mob.shift(0.025 * DL)
                n_mobs.add(n_mob)
            n_mob_groups.add(n_mobs)
        for color, group in zip([YELLOW, GREEN_B, RED, MAROON_B], n_mob_groups):
            group.set_color(color)
        last_group = e_mobs
        for group in [*n_mob_groups, e_mobs]:
            self.play(
                LaggedStartMap(FadeOut, last_group, lag_ratio=0.2),
                LaggedStartMap(FadeIn, group, lag_ratio=0.2),
                run_time=1.5,
            )
            self.wait()
            last_group = group

        # Show which equations are solved.
        equations = VGroup(
            Tex("\\frac{dz}{dt}(t) = i \\cdot z(t)"),
            get_2d_equation([["3", "1"], ["4", "1"]]),
            Tex("\\frac{dq}{dt}(t) = \\frac{i + j + k}{ \\sqrt{3} } \\cdot q(t)"),
            Tex("{\\partial \\over \\partial t}f(x, t) = {\\partial \\over \\partial x}f(x, t)"),
        )
        for eq, exp in zip(equations, exps):
            eq.match_height(exp)
            eq.move_to(exp, DL)
            eq.align_to(equations[0], LEFT)

        equations.to_edge(RIGHT)

        self.play(
            FadeOut(arrow, 3 * LEFT),
            FadeOut(exotic_label, 3 * LEFT),
            VGroup(labels, exps).animate.to_edge(LEFT),
            LaggedStartMap(FadeIn, equations, run_time=3, lag_ratio=0.5)
        )
        self.wait()
        self.play(
            VGroup(
                labels[0], labels[2:],
                exps[0], exps[2:],
                equations[0], equations[2:]
            ).animate.set_opacity(0.3),
            equations[1].animate.scale(1.5, about_edge=RIGHT),
            exps[1].animate.scale(1.5),
            labels[1].animate.scale(1.5, about_edge=LEFT),
        )
        self.wait()


class TryToDefineExp(TeacherStudentsScene):
    CONFIG = {
        "background_color": BLACK,
    }

    def construct(self):
        words = TexText(
            "Try to define {{$e^{M}$}} to\\\\make sure this is true!",
        )
        words[1][1].set_color(TEAL)

        self.play(
            PiCreatureSays(self.teacher, words, target_mode="surprised"),
            self.change_students("pondering", "thinking", "pondering"),
        )
        for pi in self.students:
            for eye in pi.eyes:
                eye.refresh_bounding_box()
        self.look_at(4 * UP + 2 * RIGHT)
        self.wait(6)


class SolutionsToMatrixEquation(From2DTo1D):
    show_solution = True


class SolutionToRomeoJuliet(Scene):
    def construct(self):
        mat_exp = get_matrix_exponential(
            [["0", "-1"], ["1", "0"]],
            h_buff=1.0,
        )
        solution = VGroup(
            Matrix(
                [["x(t)"], ["y(t)"]],
                bracket_h_buff=SMALL_BUFF,
                bracket_v_buff=SMALL_BUFF,
            ),
            Tex("="),
            mat_exp,
            Matrix(
                [["x(0)"], ["y(0)"]],
                bracket_h_buff=SMALL_BUFF,
                bracket_v_buff=SMALL_BUFF,
            )
        )
        solution[2].match_height(solution[0])
        solution[3].match_height(solution[0])
        solution.arrange(RIGHT)

        self.add(solution)
        self.wait()
        rect = SurroundingRectangle(mat_exp, buff=0.1)
        rect_copy = rect.copy()
        self.play(ShowCreation(rect))
        self.wait()
        self.play(rect.animate.become(SurroundingRectangle(solution[-1], color=BLUE)))
        self.wait()
        rot_question = Text("Rotation?")
        rot_question.next_to(rect_copy, DOWN)
        rot_question.set_color(YELLOW)
        self.play(
            rect.animate.become(rect_copy),
            Write(rot_question)
        )
        self.wait()


class RotMatrixStill(Scene):
    def construct(self):
        mat = IntegerMatrix([[0, -1], [1, 0]])
        br = BackgroundRectangle(mat, buff=SMALL_BUFF)
        br.set_fill(BLACK, 1)
        self.add(br, mat)


class ExpRotMatrixComputation(Scene):
    def construct(self):
        # Plug Mt into series
        mat_exp = get_matrix_exponential([[0, -1], [1, 0]])
        mat_exp.to_corner(UL)
        mat_exp[1].set_color(TEAL)

        equation = Tex(
            "e^X", ":=", "X^0 + X^1 + \\frac{1}{2} X^2 + \\frac{1}{6} X^3 + \\cdots + \\frac{1}{n!} X^n + \\cdots",
            isolate=["X", "+"],
        )
        equation.set_width(FRAME_WIDTH - 1)
        equation.to_edge(UP)
        self.add(equation)

        rhs_tex = "{t}^0 X^0 + {t}^1 X^1 + \\frac{1}{2} {t}^2 X^2 + \\frac{1}{6} {t}^3 X^3 + \\cdots + \\frac{1}{n!} {t}^n X^n + \\cdots"
        mat_tex = "\\left[ \\begin{array}{cc} 0 & -1 \\\\ 1 & 0 \\end{array} \\right]"
        mat_rhs = Tex(
            rhs_tex.replace("X", mat_tex),
            tex_to_color_map={mat_tex: TEAL},
            isolate=["{t}", "+"],
        )
        mat_rhs.scale(0.5)
        mat_equals = Tex("=")

        mat_exp.match_height(mat_rhs)
        mat_equation = VGroup(mat_exp, mat_equals, mat_rhs)
        mat_equation.arrange(RIGHT)
        mat_exp.align_to(mat_rhs, DOWN)
        mat_equation.set_width(FRAME_WIDTH - 1)
        mat_equation.next_to(equation, DOWN, LARGE_BUFF)

        self.wait()
        self.play(LaggedStart(
            FadeTransform(equation[:2].copy(), mat_equation[0]),
            FadeTransform(equation[2].copy(), mat_equation[1]),
            *(
                FadeTransform(equation[i].copy(), mat_equation[2][j])
                for i, j in zip(
                    # Christ...
                    [3, 4, 3, 4, 5, 6, 7, 6, 7, 8, 9, 10, 11, 10, 11, 12, 13, 14, 15, 14, 15, 16, 17, 18, 19, 20, 21, 20, 21, 22, 23],
                    it.count()
                )
            ),
            run_time=2,
            lag_ratio=0.02
        ))
        self.wait()

        # Show first few powers
        new_eq = mat_equation.copy()
        new_eq.shift(1.5 * DOWN)

        kw = {"bracket_h_buff": 0.25}
        m0 = IntegerMatrix([[1, 0], [0, 1]], **kw)
        m1 = IntegerMatrix([[0, -1], [1, 0]], **kw)
        m2 = IntegerMatrix([[-1, 0], [0, -1]], **kw)
        m3 = IntegerMatrix([[0, 1], [-1, 0]], **kw)
        mat_powers = VGroup(m0, m1, m2, m3)
        mat_powers.set_submobject_colors_by_gradient(BLUE, GREEN, RED)

        indices = [2, 7, 13, 19]
        for mat_power, index in zip(mat_powers, indices):
            mat_power.replace(new_eq[2][index], dim_to_match=0)
            mat_power.source = mat_equation[2][index:index + 2]
            new_eq[2][index: index + 2].set_opacity(0)

        self.play(LaggedStart(
            TransformFromCopy(mat_equation[0], new_eq[0]),
            TransformFromCopy(mat_equation[1], new_eq[1]),
            *(
                TransformFromCopy(mat_equation[2][i], new_eq[2][i])
                for i in range(21)
                if i not in indices and i - 1 not in indices
            ),
            lag_ratio=0.01
        ))

        rect = SurroundingRectangle(mat_powers[0].source, buff=0.05)
        rect.set_stroke(width=0)
        for mat_power in mat_powers:
            self.play(rect.animate.become(SurroundingRectangle(mat_power.source, buff=0.05)), run_time=0.5)
            self.play(FadeTransform(mat_power.source.copy(), mat_power))
            self.wait(0.5)
        self.wait()
        self.play(rect.animate.move_to(mat_equation[2][27:29]))

        # Show cycling pattern
        rows = VGroup()
        for i in range(2):
            row = VGroup()
            for j in range(4):
                mat_power_copy = mat_powers[j].deepcopy()
                power = str(4 * (i + 1) + j)
                coef = Tex("+\\frac{1}{" + power + "!} t^{" + power + "}")
                coef.match_height(mat_equation[2][10])
                coef.next_to(mat_power_copy, LEFT, SMALL_BUFF)
                row.add(coef, mat_power_copy)
            row.shift(1.1 * (i + 1) * DOWN)
            rows.add(row)
        dots = Tex("+ \\cdots", font_size=24)
        dots.next_to(rows, DOWN, aligned_edge=LEFT)

        for row in rows:
            for coef, mp in zip(row[0::2], row[1::2]):
                new_rect = SurroundingRectangle(mp, buff=0.05)
                self.add(coef, mp, new_rect)
                self.wait(0.5)
                self.remove(new_rect)
        self.play(Write(dots))
        self.wait()

        # Setup for new rhs
        lhs = mat_equation[:2]
        self.play(
            lhs.animate.scale(2).to_corner(UL).shift(DOWN),
            FadeOut(VGroup(mat_equation[2], rect, equation))
        )

        # Show massive rhs
        rhs = Matrix(
            [
                [
                    "1 - \\frac{t^2}{2!} + \\frac{t^4}{4!} - \\frac{t^6}{6!} + \\cdots",
                    "-t + \\frac{t^3}{3!} - \\frac{t^5}{5!} + \\frac{t^7}{7!} - \\cdots",
                ],
                [
                    "t - \\frac{t^3}{3!} + \\frac{t^5}{5!} - \\frac{t^7}{7!} + \\cdots",
                    "1 - \\frac{t^2}{2!} + \\frac{t^4}{4!} - \\frac{t^6}{6!} + \\cdots",
                ],
            ],
            h_buff=6,
            v_buff=2,
        )
        rhs.set_width(10)
        rhs.next_to(lhs, RIGHT)

        power_entry_rects = VGroup(*(VGroup() for x in range(4)))
        for group in [mat_powers, rows[0][1::2], rows[1][1::2]]:
            for mat_power in group:
                for i, entry in enumerate(mat_power.get_entries()):
                    rect = SurroundingRectangle(entry, buff=0.05)
                    rect.set_stroke(YELLOW, 2)
                    power_entry_rects[i].add(rect)

        self.play(Write(rhs.get_brackets()))

        last_per = power_entry_rects[0].copy()
        last_per.set_opacity(0)
        last_rect = VMobject()
        for entry, per in zip(rhs.get_entries(), power_entry_rects):
            rect = VGroup(SurroundingRectangle(entry, stroke_width=1))
            self.play(
                ReplacementTransform(last_per, per),
                FadeIn(entry),
                FadeIn(rect),
                FadeOut(last_rect),
            )
            self.wait()
            last_per = per
            last_rect = rect
        self.play(FadeOut(last_per), FadeOut(last_rect))
        self.wait()

        # Show collapse to trig functions
        low_terms = VGroup(new_eq[2][:21], mat_powers, rows, dots)

        new_lhs = lhs.copy()
        new_lhs.next_to(rhs, DOWN, LARGE_BUFF)
        new_lhs.align_to(lhs, LEFT)
        final_result = Matrix(
            [["\\cos(t)", "-\\sin(t)"], ["\\sin(t)", "\\cos(t)"]],
            h_buff=2.0
        )
        final_result.next_to(new_lhs, RIGHT)

        anims = []
        colors = [BLUE_B, BLUE_D, BLUE_D, BLUE_B]
        for color, entry1, entry2 in zip(colors, rhs.get_entries(), final_result.get_entries()):
            anims.append(entry1.animate.set_color(color))
            entry2.set_color(color)

        self.play(
            *anims,
            FadeOut(low_terms, shift=2 * DOWN),
            ReplacementTransform(new_eq[:2], new_lhs),
            TransformFromCopy(rhs.get_brackets(), final_result.get_brackets()),
        )
        self.wait()
        last_rects = VGroup()
        for entry1, entry2 in zip(rhs.get_entries(), final_result.get_entries()):
            rect1 = SurroundingRectangle(entry1, stroke_width=1)
            rect2 = SurroundingRectangle(entry2, stroke_width=1)
            self.play(
                FadeIn(rect1),
                FadeIn(rect2),
                FadeIn(entry2),
                FadeOut(last_rects),
            )
            last_rects = VGroup(rect1, rect2)
        self.play(FadeOut(last_rects))
        self.wait()

        # Ask question
        question = TexText("What transformation\\\\is this?")
        question.next_to(final_result, RIGHT, buff=1.5)
        question.shift_onto_screen()
        arrow = Arrow(question, final_result, buff=SMALL_BUFF)

        self.play(
            FadeIn(question, shift=0.5 * RIGHT),
            GrowArrow(arrow)
        )
        self.wait()

        # Highlight result
        high_eq = VGroup(lhs, rhs)
        low_eq = VGroup(new_lhs, final_result)

        self.play(LaggedStart(
            FadeOut(high_eq, UP),
            FadeOut(VGroup(question, arrow)),
            low_eq.animate.center().to_edge(UP),
            lag_ratio=0.05
        ))
        self.wait()


class ThatsHorrifying(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            TexText("You want us to\\\\do what?"),
            index=2,
            target_mode="pleading",
            added_anims=[LaggedStart(
                self.students[0].change("tired"),
                self.students[1].change("horrified"),
                self.teacher.change("guilty"),
                lag_ratio=0.5
            )]
        )
        for pi in self.pi_creatures:
            for eye in pi.eyes:
                # Why?
                eye.refresh_bounding_box()
        self.look_at(self.screen)
        self.wait(2)
        self.teacher_says(
            "Just wait",
            bubble_config={"height": 3, "width": 3.5},
            target_mode="tease"
        )
        self.play_student_changes(
            "pondering", "thinking", "hesitant",
            look_at=self.screen,
        )
        self.wait(3)


class LinearAlgebraWrapper(VideoWrapper):
    title = "Matrices as linear transformations"


class HowBasisVectorMultiplicationPullsOutColumns(Scene):
    def construct(self):
        # Setup
        plane = NumberPlane()
        plane.scale(2.5)
        plane.shift(1.5 * DOWN)
        b_plane = plane.copy()
        b_plane.set_color(GREY_B)
        plane.add_coordinate_labels()
        self.add(b_plane, plane)

        matrix = Matrix(
            [["a", "b"], ["c", "d"]],
            h_buff=0.8,
        )
        matrix.to_corner(UL)
        matrix.to_edge(LEFT, buff=MED_SMALL_BUFF)
        matrix.add_to_back(BackgroundRectangle(matrix))
        self.add(matrix)

        basis_vectors = VGroup(
            Arrow(plane.get_origin(), plane.c2p(1, 0), buff=0, fill_color=GREEN),
            Arrow(plane.get_origin(), plane.c2p(0, 1), buff=0, fill_color=RED),
        )
        bhb = 0.2
        basis_labels = VGroup(
            Matrix([["1"], ["0"]], bracket_h_buff=bhb),
            Matrix([["0"], ["1"]], bracket_h_buff=bhb),
        )
        for vector, label, direction in zip(basis_vectors, basis_labels, [UR, RIGHT]):
            label.scale(0.7)
            label.match_color(vector)
            label.add_to_back(BackgroundRectangle(label))
            label.next_to(vector.get_end(), direction)

        # Show products
        basis_label_copies = basis_labels.deepcopy()
        rhss = VGroup(
            Matrix([["a"], ["c"]], bracket_h_buff=bhb),
            Matrix([["b"], ["d"]], bracket_h_buff=bhb),
        )
        colors = [GREEN, RED]

        def show_basis_product(index, matrix):
            basis_label_copies[index].match_height(matrix)
            basis_label_copies[index].next_to(matrix, RIGHT, SMALL_BUFF),
            equals = Tex("=")
            equals.next_to(basis_label_copies[index], RIGHT, SMALL_BUFF)
            rhss[index].next_to(equals, RIGHT, SMALL_BUFF)
            rhss[index].set_color(colors[index])
            rhs_br = BackgroundRectangle(rhss[index])

            self.play(
                FadeIn(basis_labels[index], RIGHT),
                GrowArrow(basis_vectors[index]),
                FadeIn(basis_label_copies[index]),
                FadeIn(equals),
                FadeIn(rhs_br),
                FadeIn(rhss[index].get_brackets()),
            )
            rect_kw = {"stroke_width": 2, "buff": 0.1}
            row_rects = [
                SurroundingRectangle(row, **rect_kw)
                for row in matrix.get_rows()
            ]
            col_rect = SurroundingRectangle(basis_label_copies[index].get_entries(), **rect_kw)
            col_rect.set_stroke(opacity=0)
            last_row_rect = VMobject()
            for e1, e2, row_rect in zip(matrix.get_columns()[index], rhss[index].get_entries(), row_rects):
                self.play(
                    col_rect.animate.set_stroke(opacity=1),
                    FadeIn(row_rect),
                    FadeOut(last_row_rect),
                    e1.animate.set_color(colors[index]),
                    FadeIn(e2),
                )
                last_row_rect = row_rect
            self.play(FadeOut(last_row_rect), FadeOut(col_rect))
            rhss[index].add_to_back(rhs_br)
            rhss[index].add(equals)

        low_matrix = matrix.deepcopy()
        show_basis_product(0, matrix)
        self.wait()
        self.play(low_matrix.animate.shift(2.5 * DOWN))
        show_basis_product(1, low_matrix)
        self.wait()


class ColumnsToBasisVectors(ExternallyAnimatedScene):
    pass


class ReadColumnsOfRotationMatrix(Scene):
    show_exponent = False

    def construct(self):
        # Setup
        plane = NumberPlane(faded_line_ratio=0)
        plane.scale(3.0)
        plane.shift(1.5 * DOWN)
        b_plane = plane.copy()
        b_plane.set_color(GREY_B)
        b_plane.set_stroke(opacity=0.5)

        angle = 50 * DEGREES
        plane2 = plane.copy().apply_matrix([
            [math.cos(angle), 0],
            [math.sin(angle), 1],
        ], about_point=plane.get_origin())
        plane3 = plane.copy().apply_matrix([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ], about_point=plane.get_origin())
        coords = plane.deepcopy().add_coordinate_labels((-2, -1, 1, 2), (-1, 1))
        self.add(b_plane, plane, coords)

        equation = VGroup(
            get_matrix_exponential([[0, -1], [1, 0]]),
            Tex("="),
            Matrix(
                [["\\cos(t)", "-\\sin(t)"], ["\\sin(t)", "\\cos(t)"]],
                h_buff=2.0
            )
        )
        exp, eq, matrix = equation
        exp[1].set_color(TEAL)
        exp[1].add_background_rectangle()
        matrix.add_background_rectangle()
        equation.arrange(RIGHT)
        equation.set_width(6)
        equation.to_corner(UL)

        if not self.show_exponent:
            equation.remove(*equation[:2])
            equation.set_height(1.5)
            equation.to_corner(UL)

        self.add(equation)

        basis_vectors = VGroup(
            Arrow(plane.get_origin(), plane.c2p(1, 0), buff=0, fill_color=GREEN),
            Arrow(plane.get_origin(), plane.c2p(0, 1), buff=0, fill_color=RED),
        )
        basis_shadows = basis_vectors.copy()
        basis_shadows.set_fill(opacity=0.5)
        self.add(basis_shadows, basis_vectors)

        self.play(FlashAround(matrix.get_columns()[0], color=GREEN))
        self.wait()

        # Show action on basis vectors
        rot_b0, rot_b1 = rot_basis_vectors = basis_vectors.copy()
        for rot_b in rot_basis_vectors:
            rot_b.rotate(angle, about_point=plane.get_origin())

        rbl0, rbl1 = rot_basis_labels = VGroup(
            Matrix([["\\cos(t)"], ["\\sin(t)"]]),
            Matrix([["-\\sin(t)"], ["\\cos(t)"]]),
        )
        for label, color, rot_b, direction in zip(rot_basis_labels, [GREEN, RED], rot_basis_vectors, [UR, LEFT]):
            label.set_color(color)
            label.scale(0.7)
            label.next_to(rot_b.get_end(), direction, SMALL_BUFF)
            label.add_background_rectangle()

        arcs = VGroup(
            Arc(0, angle, arc_center=plane.get_origin(), radius=0.5),
            Arc(PI / 2, angle, arc_center=plane.get_origin(), radius=0.5),
        )
        arcs.set_stroke(WHITE, 2)
        arc_label = TexText("$t$", " ")
        arc_label.next_to(arcs[0], RIGHT, SMALL_BUFF)
        arc_label.shift(SMALL_BUFF * UP)

        h_line = DashedLine(plane.get_origin(), plane.c2p(math.cos(angle), 0))
        v_line = DashedLine(plane.c2p(math.cos(angle), 0), plane.c2p(math.cos(angle), math.sin(angle)))
        cos_label = matrix.get_entries()[0].copy().next_to(h_line, DOWN, SMALL_BUFF)
        sin_label = matrix.get_entries()[2].copy().next_to(v_line, RIGHT, SMALL_BUFF)

        self.play(
            TransformFromCopy(matrix[0], rbl0[0]),  # Background rectangles
            TransformFromCopy(matrix.get_brackets(), rbl0.get_brackets()),
            TransformFromCopy(matrix.get_columns()[0], rbl0.get_entries()),
            Transform(plane, plane2, path_arc=angle),
            Transform(basis_vectors[0], rot_b0, path_arc=angle),
            Animation(matrix.get_columns()[1]),
            run_time=2,
        )
        rects = VGroup(*(
            SurroundingRectangle(entry, color=WHITE, stroke_width=2, buff=0.05)
            for entry in rbl0.get_entries()
        ))
        self.play(
            ShowCreation(h_line),
            FadeIn(rects[0]),
            FadeIn(cos_label),
        )
        self.play(
            ShowCreation(v_line),
            FadeOut(rects[0]),
            FadeIn(rects[1]),
            FadeIn(sin_label),
        )
        self.play(FadeOut(rects[1]))
        self.wait()
        self.play(FlashAround(matrix.get_columns()[1], color=RED))
        self.play(
            TransformFromCopy(matrix[0], rbl1[0]),  # Background rectangles
            TransformFromCopy(matrix.get_brackets(), rbl1.get_brackets()),
            TransformFromCopy(matrix.get_columns()[1], rbl1.get_entries()),
            Transform(plane, plane3, path_arc=angle),
            Transform(basis_vectors[1], rot_b1, path_arc=angle),
            Animation(matrix.get_columns()[0]),
            run_time=2,
        )
        self.wait()
        self.play(
            FadeIn(arc_label),
            *map(ShowCreation, arcs),
        )
        self.wait()


class ReadColumnsOfRotationMatrixWithExp(ReadColumnsOfRotationMatrix):
    show_exponent = True


class AnalyzeRomeoAndJulietSpace(RomeoJulietVectorSpace):
    def construct(self):
        # Add axes
        axes = self.get_romeo_juliet_axes()
        ps_dot = axes.ps_dot
        ps_dot.set_opacity(0)
        ps_dot.move_to(axes.c2p(4, 3))

        ps_arrow = self.get_ps_arrow(axes, add_shadow=True)
        ps_arrow.update()
        self.add(ps_arrow)

        # Add equation
        matrix = [["0", "-1"], ["1", "0"]]
        equation = get_2d_equation(matrix)
        equation.to_corner(UR)

        implies = Tex("\\Downarrow", font_size=72)
        implies.next_to(equation, DOWN, buff=0.3)

        initial_condition = Matrix([["x_0"], ["y_0"]], bracket_h_buff=0.1)
        initial_condition.match_height(equation)
        initial_condition.set_color(BLUE_B)
        ic_source = initial_condition.copy()
        ic_source.scale(0.5)
        ic_source.next_to(ps_arrow.get_end(), RIGHT, SMALL_BUFF)

        mat_exp = get_matrix_exponential(matrix, h_buff=0.8)
        solution = VGroup(equation[1].copy(), Tex("="), mat_exp, initial_condition)
        solution[2][1].set_color(TEAL)
        solution.arrange(RIGHT)
        solution.scale(0.8)
        solution.next_to(implies, DOWN, buff=0.3)

        rot_matrix = Matrix(
            [["\\cos(t)", "-\\sin(t)"], ["\\sin(t)", "\\cos(t)"]],
            bracket_h_buff=0.2,
            h_buff=2.0
        )
        for entry in rot_matrix.get_entries():
            entry[0][-2].set_color(GREY_B)
        rot_matrix.match_height(solution[0])
        solution2 = solution.deepcopy()
        solution2.replace_submobject(2, rot_matrix)
        solution2.arrange(RIGHT)
        solution2.next_to(solution, DOWN, buff=0.7)

        rot_matrix_brace = Brace(rot_matrix, DOWN)
        rot_label = TexText("Rotate by angle $t$")
        rot_label.next_to(rot_matrix_brace, DOWN, SMALL_BUFF)

        self.add(equation)
        self.play(FlashAround(equation, run_time=2))
        self.wait()
        self.play(
            Write(implies),
            TransformFromCopy(equation[1], solution[0], path_arc=-20 * DEGREES),
            TransformFromCopy(equation[2], solution[1]),
        )
        self.play(
            Write(mat_exp[0]),
            TransformFromCopy(equation[3], mat_exp[1]),
        )
        self.play(Write(mat_exp[2]))
        self.wait()
        self.play(
            FadeIn(ic_source),
            FlashAround(ic_source),
        )
        self.play(TransformFromCopy(ic_source, initial_condition))
        self.wait()
        self.play(
            *(
                TransformFromCopy(solution[i], solution2[i])
                for i in [0, 1, 3]
            ),
            FadeTransform(mat_exp.copy(), rot_matrix)
        )
        self.play(
            GrowFromCenter(rot_matrix_brace),
            FadeIn(rot_label),
        )
        self.wait()

        # Rotate vector
        start_angle = ps_arrow.get_angle()
        t_tracker = ValueTracker(0)
        get_t = t_tracker.get_value

        def get_arc():
            curr_time = get_t()
            arc = ParametricCurve(
                lambda s: axes.c2p(*tuple((1 + 0.2 * s) * np.array([
                    math.cos(start_angle + s * curr_time),
                    math.sin(start_angle + s * curr_time),
                ]))),
                t_range=(0, 1, 0.01)
            )
            arc.set_stroke(WHITE, 2)
            return arc

        arc = always_redraw(get_arc)
        t_label = VGroup(Tex("t = ", font_size=30), DecimalNumber(font_size=24))
        t_label.arrange(RIGHT, buff=SMALL_BUFF)

        def update_t_label(label):
            point = arc.pfp(0.5)
            label[1].set_value(get_t())
            label.set_stroke(BLACK, 5, background=True)
            label.set_opacity(min(2 * get_t(), 1))
            if get_t() < 3.65:
                target_point = point + 0.75 * (point - axes.get_origin())
                label.shift(target_point - label[1].get_center())

        t_label.add_updater(update_t_label)

        curr_xy = np.array(axes.p2c(ps_dot.get_center()))

        def update_ps_dot(dot):
            rot_M = np.array(rotation_matrix_transpose(get_t(), OUT))[:2, :2]
            new_xy = np.dot(curr_xy, rot_M)
            dot.move_to(axes.c2p(*new_xy))

        ps_dot.add_updater(update_ps_dot)

        t_tracker.add_updater(lambda m, dt: m.increment_value(0.5 * dt))
        self.add(arc, t_label, t_tracker, ps_dot)
        self.play(VFadeIn(t_label))
        self.wait(16 * PI - 1)

    def get_romeo_juliet_axes(self, height=5):
        # Largely copied from RomeoJulietVectorSpace
        romeo, juliet = lovers = self.get_romeo_and_juliet()
        axes = Axes(
            x_range=(-5, 5),
            y_range=(-5, 5),
            height=height,
            width=height,
            axis_config={
                "include_tip": False,
                "numbers_to_exclude": [],
            }
        )
        axes.to_edge(LEFT)

        for axis in axes:
            axis.add_numbers(range(-4, 6, 2), color=GREY_A, font_size=12)
            axis.numbers[2].set_opacity(0)

        lovers.set_height(0.5)
        lovers.flip()
        juliet.next_to(axes.x_axis.get_end(), RIGHT)
        romeo.next_to(axes.y_axis.get_end(), UP)

        ps_dot = Dot(radius=0.04, fill_color=BLUE)
        ps_dot.move_to(axes.c2p(3, 1))

        def get_xy():
            # Get phase space point
            return axes.p2c(ps_dot.get_center())

        self.make_romeo_and_juliet_dynamic(romeo, juliet)
        juliet.love_tracker.add_updater(lambda m: m.set_value(get_xy()[0]))
        romeo.love_tracker.add_updater(lambda m: m.set_value(get_xy()[1]))
        self.add(romeo.love_tracker, juliet.love_tracker)

        name_labels = self.get_romeo_juilet_name_labels(lovers, font_size=12, spacing=1.0, buff=0.05)

        axes.lovers = lovers
        axes.name_labels = name_labels
        axes.ps_dot = ps_dot
        axes.add(lovers, name_labels, ps_dot)

        self.add(axes)
        self.add(*lovers)
        self.add(*(pi.heart_eyes for pi in lovers))

        return axes

    def get_ps_arrow(self, axes, color=BLUE, add_shadow=False):
        arrow = Arrow(
            LEFT, RIGHT,
            fill_color=color,
            thickness=0.04
        )
        arrow.add_updater(lambda m: m.put_start_and_end_on(
            axes.get_origin(), axes.ps_dot.get_center(),
        ))

        if add_shadow:
            ps_arrow_shadow = arrow.copy()
            ps_arrow_shadow.clear_updaters()
            ps_arrow_shadow.set_fill(BLUE, 0.5)
            self.add(ps_arrow_shadow)
        return arrow


class GeometricReasoningForRomeoJuliet(AnalyzeRomeoAndJulietSpace):
    def construct(self):
        # Setup
        axes = self.get_romeo_juliet_axes()
        ps_dot = axes.ps_dot
        ps_dot.set_opacity(0)
        ps_dot.move_to(axes.c2p(4, 3))

        ps_arrow = self.get_ps_arrow(axes)
        self.add(ps_arrow)

        matrix = [["0", "-1"], ["1", "0"]]
        equation = get_2d_equation(matrix)
        ddt, xy1, eq, mat, xy2 = equation
        equation.to_corner(UR)
        self.add(equation)

        # Show 90 degree rotation
        mat_brace = Brace(mat, DOWN, buff=SMALL_BUFF)
        mat_label = mat_brace.get_text("90-degree rotation matrix", font_size=30)
        mat_label.set_color(TEAL)

        self.play(
            GrowFromCenter(mat_brace),
            Write(mat_label, run_time=1)
        )
        self.wait()

        # Spiral around to various points
        curve = VMobject()
        curve.set_points_smoothly(
            [
                axes.c2p(x, y)
                for x, y in [(4, 3), (0, 2), (0, -1), (-3, -4), (-3, 2), (3, 1)]
            ],
            true_smooth=True
        )
        self.play(MoveAlongPath(ps_dot, curve, run_time=5))
        self.wait()

        # Rate of change
        deriv_vect = ps_arrow.copy()
        deriv_vect.clear_updaters()
        deriv_vect.set_color(RED)
        deriv_vect.shift(ps_arrow.get_vector())

        deriv_rect = SurroundingRectangle(VGroup(ddt, xy1))
        deriv_rect.set_stroke(RED, 2)

        d_line = DashedLine(ps_arrow.get_start(), ps_arrow.get_end())
        d_line.shift(ps_arrow.get_vector())
        d_line.set_stroke(WHITE, 1)

        arc = Arc(
            start_angle=ps_arrow.get_angle(),
            angle=90 * DEGREES,
            arc_center=ps_arrow.get_end(),
            radius=0.25
        )
        elbow = VMobject()
        elbow.set_points_as_corners([RIGHT, UR, UP])
        elbow.scale(0.25, about_point=ORIGIN)
        elbow.rotate(ps_arrow.get_angle(), about_point=ORIGIN)
        elbow.shift(ps_arrow.get_end())
        VGroup(elbow, arc).set_stroke(WHITE, 2)

        self.play(
            TransformFromCopy(ps_arrow, deriv_vect, path_arc=30 * DEGREES),
            ShowCreation(deriv_rect),
        )
        self.add(d_line, deriv_vect)
        self.play(
            ShowCreation(arc),
            Rotate(deriv_vect, 90 * DEGREES, about_point=ps_arrow.get_end()),
            run_time=2,
            rate_func=rush_into,
        )
        self.remove(arc)
        self.add(elbow)
        self.wait()

        # Show rotation
        ps_arrow.clear_updaters()
        ps_dot.add_updater(lambda m: m.move_to(ps_arrow.get_end()))
        rot_group = VGroup(ps_arrow, d_line, elbow, deriv_vect)
        circle = Circle(radius=ps_arrow.get_length())
        circle.set_stroke(GREY, 1)
        circle.move_to(axes.get_origin())

        self.play(
            Rotate(
                rot_group,
                angle=TAU - ps_arrow.get_angle(),
                about_point=axes.get_origin(),
                run_time=6,
                rate_func=linear,
            ),
            FadeIn(circle),
        )
        self.wait()

        # Show matching length
        self.play(
            deriv_vect.animate.put_start_and_end_on(ps_arrow.get_start(), ps_arrow.get_end()).shift(0.2 * UP),
            run_time=3,
            rate_func=there_and_back_with_pause,
        )
        self.wait()

        # Show one radian of arc
        arc = Arc(
            0, 1,
            radius=circle.radius,
            arc_center=axes.get_origin(),
        )
        arc.set_stroke(YELLOW, 3)
        sector = Sector(
            start_angle=0,
            angle=1,
            outer_radius=circle.radius,
        )
        sector.shift(axes.get_origin())
        sector.set_stroke(width=0)
        sector.set_fill(GREY_D, 1)

        t_label = VGroup(Tex("t = ", font_size=30), DecimalNumber(0, font_size=20))
        t_label.arrange(RIGHT, buff=SMALL_BUFF)
        t_label.next_to(arc.pfp(0.5), UR, SMALL_BUFF)

        self.play(
            Rotate(rot_group, 1, about_point=axes.get_origin()),
            ShowCreation(arc),
            ChangeDecimalToValue(t_label[1], 1.0),
            VFadeIn(t_label),
            run_time=2,
            rate_func=linear,
        )
        self.add(sector, axes, ps_arrow, arc)
        self.play(FadeIn(sector))
        self.wait()

        radius = Line(axes.get_origin(), arc.get_start())
        radius.set_stroke(RED, 5)
        self.play(ShowCreation(radius))
        self.play(Rotate(radius, -PI / 2, about_point=radius.get_end()))
        radius.rotate(PI)
        self.play(radius.animate.match_points(arc))
        self.play(FadeOut(radius))
        self.wait()

        # One radian per unit time
        self.play(
            t_label.animate.shift(UP).scale(2),
        )
        t_label[1].data["font_size"] *= 2
        for x in range(5):
            VGroup(sector, arc).rotate(1, about_point=axes.get_origin())
            self.play(
                Rotate(rot_group, 1, about_point=axes.get_origin()),
                ChangeDecimalToValue(t_label[1], t_label[1].get_value() + 1),
                run_time=2,
                rate_func=linear,
            )
        self.play(
            FadeOut(VGroup(sector, arc)),
            Rotate(rot_group, TAU - 6, about_point=axes.get_origin()),
            ChangeDecimalToValue(t_label[1], TAU),
            run_time=2 * (TAU - 6),
            rate_func=linear,
        )
        self.wait(2)

        t_label[1].set_value(0)
        self.play(
            Rotate(rot_group, PI, about_point=axes.get_origin()),
            ChangeDecimalToValue(t_label[1], PI),
            run_time=2 * PI,
            rate_func=linear,
        )
        self.wait(2)


class Show90DegreeRotation(Scene):
    def construct(self):
        plane = NumberPlane()
        plane.scale(2.5)
        back_plane = plane.deepcopy()
        back_plane.set_stroke(color=GREY, opacity=0.5)
        back_plane.add_coordinate_labels()

        vects = VGroup(
            Arrow(plane.get_origin(), plane.c2p(1, 0), fill_color=GREEN, buff=0, thickness=0.075),
            Arrow(plane.get_origin(), plane.c2p(0, 1), fill_color=RED, buff=0, thickness=0.075),
        )
        plane.add(*vects)
        plane.set_stroke(background=True)

        self.add(back_plane, plane)
        self.wait()
        self.play(Rotate(plane, 90 * DEGREES, run_time=3))
        self.wait()


class Show90DegreeRotationColumnByColumn(Scene):
    def construct(self):
        plane = NumberPlane()
        plane.scale(2.5)
        back_plane = plane.deepcopy()
        back_plane.set_stroke(color=GREY, opacity=0.5)
        back_plane.add_coordinate_labels()

        plane2 = plane.copy()
        plane3 = plane.copy()
        plane2.apply_matrix([[0, 0], [1, 1]])
        plane3.apply_matrix([[0, -1], [1, 0]])

        vects = VGroup(
            Arrow(plane.get_origin(), plane.c2p(1, 0), fill_color=GREEN, buff=0, thickness=0.075),
            Arrow(plane.get_origin(), plane.c2p(0, 1), fill_color=RED, buff=0, thickness=0.075),
        )

        self.add(back_plane, plane, *vects)
        self.wait()
        self.play(
            Transform(plane, plane2, path_arc=90 * DEGREES),
            Rotate(vects[0], 90 * DEGREES, about_point=plane.get_origin()),
            run_time=2
        )
        self.wait()
        self.play(
            Transform(plane, plane3, path_arc=90 * DEGREES),
            Rotate(vects[1], 90 * DEGREES, about_point=plane.get_origin()),
            run_time=2
        )
        self.wait()

        arc = Arc(5 * DEGREES, 80 * DEGREES, buff=0.1, radius=1.5)
        arc.set_stroke(width=3)
        arc.add_tip(width=0.15, length=0.15)
        arc2 = arc.copy().rotate(PI, about_point=ORIGIN)
        arcs = VGroup(arc, arc2)
        arcs.set_color(GREY_B)
        self.play(*map(ShowCreation, arcs))
        self.wait()


class DistanceOverTimeEquation(Scene):
    def construct(self):
        equation = Tex(
            "{\\text{Distance}", " \\over", " \\text{Time} }", "=", "\\text{Radius}",
        )
        # equation[:3].set_color(RED_B)
        equation.add(SurroundingRectangle(equation[:3], color=RED, stroke_width=1))
        equation[4].set_color(BLUE)
        self.play(FadeIn(equation, UP))
        self.wait()


class ExplicitSolution(Scene):
    def construct(self):
        kw = {"tex_to_color_map": {
            "x_0": BLUE_D,
            "y_0": BLUE_B,
            "{t}": GREY_B,

        }}
        solutions = VGroup(
            Tex("x({t}) = \\cos(t) x_0 - \\sin(t) y_0", **kw),
            Tex("y({t}) = \\sin(t) x_0 + \\cos(t) y_0", **kw),
        )
        solutions.arrange(DOWN, buff=MED_LARGE_BUFF)
        for solution in solutions:
            self.play(Write(solution))
        self.wait()


class TwoDifferetViewsWrapper(Scene):
    def construct(self):
        self.add(FullScreenRectangle())
        screens = VGroup(*(ScreenRectangle() for x in range(2)))
        screens.arrange(RIGHT)
        screens.set_width(FRAME_WIDTH - 1)
        screens.set_fill(BLACK, 1)
        screens[0].set_stroke(BLUE, 2)
        screens[1].set_stroke(GREY_BROWN, 2)
        self.add(screens)

        titles = VGroup(
            Text("Geometric"),
            Text("Analytic"),
        )
        for title, screen in zip(titles, screens):
            title.next_to(screen, UP)
            title.align_to(titles[0], UP)
            self.play(Write(title))
            self.wait()
        self.wait()


class EulersFormulaWrapper(VideoWrapper):
    title = "Video on $e^{it}$"


class ImaginaryExponent(ExternallyAnimatedScene):
    pass


class ComplexEquation(Scene):
    def construct(self):
        # Equation
        equation = Tex(
            """
            \\frac{d}{d{t} } \\Big[ x(t) + {i}y(t)\\Big] =
            {i} \\cdot \\Big[ x(t) + {i}y(t) \\Big]
            """,
            tex_to_color_map={
                "x": BLUE_B,
                "y": BLUE_D,
                "{t}": GREY_B,
                "=": WHITE,
                "\\cdot": WHITE,
                "{i}": RED,
            }
        )
        equation.move_to(2 * UP)

        braces = VGroup(*(
            Brace(equation[i:j], UP, buff=SMALL_BUFF)
            for i, j in [(2, 8), (11, 17)]
        ))
        braces.set_fill(GREY_A, 1)
        for brace in braces:
            brace.zt = Tex("z(t)", tex_to_color_map={"z": BLUE, "t": GREY_B})
            brace.zt.next_to(brace, UP, SMALL_BUFF)

        self.add(equation)
        self.play(
            *(GrowFromCenter(b) for b in braces),
            *(FadeIn(b.zt, 0.25 * UP) for b in braces)
        )
        self.wait()

        # Show romeo and juilet
        x_part = equation.get_part_by_tex("x")
        y_part = equation.get_part_by_tex("y")

        juliet_label = TexText("Juliet's love", font_size=30)
        juliet_label.next_to(x_part, DOWN, LARGE_BUFF)
        romeo_label = TexText("Romeo's love", font_size=30)
        romeo_label.next_to(y_part, DOWN, LARGE_BUFF)
        juliet_label.align_to(romeo_label, DOWN)
        VGroup(juliet_label, romeo_label).space_out_submobjects(1.5)

        juliet_arrow = Arrow(x_part, juliet_label, buff=0.1, fill_color=BLUE_B)
        romeo_arrow = Arrow(y_part, romeo_label, buff=0.1, fill_color=BLUE_D)

        self.play(
            GrowArrow(juliet_arrow),
            FadeIn(juliet_label),
        )
        self.play(
            GrowArrow(romeo_arrow),
            FadeIn(romeo_label),
        )
        self.wait()

        # Describe i
        i_part = equation.get_parts_by_tex("{i}")[1]
        i_label = TexText("90-degree rotation", font_size=30)
        i_label.set_color(RED)
        i_label.next_to(romeo_label, RIGHT, MED_LARGE_BUFF, aligned_edge=DOWN)
        i_arrow = Arrow(i_part, i_label, fill_color=RED)

        self.play(
            GrowArrow(i_arrow),
            FadeIn(i_label, DR),
        )
        self.wait()

        all_labels = VGroup(romeo_label, juliet_label, i_label)

        # Show solution
        solution = Tex(
            "z(t) = e^{it} z_0",
            tex_to_color_map={
                "i": RED,
                "t": GREY_B,
                "z": BLUE,
                "=": WHITE,
            }
        )
        solution.scale(1.5)
        solution.next_to(all_labels, DOWN, LARGE_BUFF)
        solution[8:].set_color(BLUE_D)

        self.play(
            TransformFromCopy(braces[0].zt, solution[:4]),
            Write(solution[4]),
        )
        self.play(
            Write(solution[5]),
            TransformFromCopy(equation[9], solution[6]),
            Write(solution[7]),
        )
        self.play(
            FadeIn(solution[8:], 0.1 * DOWN)
        )
        self.wait()


class JulietChidingRomeo(Scene):
    def construct(self):
        juliet = PiCreature(color=BLUE_B)
        romeo = PiCreature(color=BLUE_D)
        romeo.flip()
        pis = VGroup(juliet, romeo)
        pis.set_height(2)
        pis.arrange(RIGHT, buff=LARGE_BUFF)
        pis.to_edge(DOWN)
        romeo.make_eye_contact(juliet)

        self.add(pis)
        self.play(
            PiCreatureSays(
                juliet, TexText("It just seems like \\\\ your feelings aren't \\\\ real"),
                bubble_config={"height": 2.5, "width": 3.5},
                target_mode="sassy",
            ),
            romeo.change("guilty"),
        )
        for x in range(2):
            self.play(Blink(juliet))
            self.play(Blink(romeo))
            self.wait()


class General90DegreeRotationExponents(Scene):
    def construct(self):
        # Setup
        exps = VGroup(
            get_matrix_exponential([["0", "-1"], ["1", "0"]]),
            Tex("e^{it}", tex_to_color_map={"i": RED}),
            Tex(
                "e^{(ai + bj + ck)t}",
                tex_to_color_map={
                    "i": RED,
                    "j": GREEN,
                    "k": BLUE,
                }
            ),
            Tex(
                "e^{i \\sigma_x t}, \\quad ",
                "e^{i \\sigma_y t}, \\quad ",
                "e^{i \\sigma_z t}",
                tex_to_color_map={
                    "\\sigma_x": RED,
                    "\\sigma_y": GREEN,
                    "\\sigma_z": BLUE,
                }
            ),
        )
        exps[0][1].set_color(TEAL)
        exps[0].scale(0.5)

        exps.arrange(DOWN, buff=1.5, aligned_edge=LEFT)

        labels = VGroup(
            TexText("$90^\\circ$ rotation matrix"),
            TexText("Imaginary numbers"),
            TexText("Quaternions"),
            TexText("Pauli matrices"),
        )
        for label, exp in zip(labels, exps):
            label.scale(0.75)
            label.next_to(exp, LEFT, LARGE_BUFF, aligned_edge=DOWN)
            label.align_to(labels[0], LEFT)

        VGroup(exps, labels).to_edge(LEFT)

        quat_note = Tex("(a^2 + b^2 + c^2 = 1)", font_size=24)
        quat_note.next_to(exps[2], DOWN, aligned_edge=LEFT)
        exps[2].add(quat_note)

        for exp, label in zip(exps, labels):
            self.play(
                FadeIn(exp),
                FadeIn(label),
            )
        self.wait()


class RotationIn3dPlane(Scene):
    def construct(self):
        # Axes and frame
        axes = ThreeDAxes()
        axes.set_stroke(width=2)
        axes.set_flat_stroke(False)

        frame = self.camera.frame
        frame.set_height(2.0 * FRAME_HEIGHT)
        frame.reorient(-40, 70)
        frame.add_updater(lambda f, dt: f.increment_theta(0.03 * dt))

        self.add(axes, frame)

        # Plane
        plane = Surface()
        plane.replace(VGroup(axes.x_axis, axes.y_axis), stretch=True)
        plane.set_color(GREY_C, opacity=0.75)
        plane.set_gloss(1)
        grid = NumberPlane((-6, 6), (-5, 5), faded_line_ratio=0)

        radius = 3
        p_vect = Vector(radius * RIGHT, fill_color=BLUE)
        v_vect = p_vect.copy()
        arc = Arc(0, PI / 2, radius=radius / 4)
        circle = Circle(radius=radius)
        circle.set_stroke(WHITE, 2)
        randy = Randolph(mode="pondering")
        randy.set_gloss(0.7)
        rot_group = Group(plane, grid, p_vect, v_vect, arc, circle, randy)

        normal = OUT
        for angle, axis in [(30 * DEGREES, RIGHT), (20 * DEGREES, UP)]:
            rot_group.rotate(angle, axis)
            normal = rotate_vector(normal, angle, axis)

        self.play(
            ShowCreation(plane),
            FadeIn(randy),
        )
        self.play(GrowArrow(p_vect))
        self.wait(2)
        self.play(
            Rotate(v_vect, PI / 2, axis=normal, about_point=axes.get_origin()),
            Rotate(randy, PI / 2, axis=normal, about_point=axes.get_origin()),
            UpdateFromAlphaFunc(v_vect, lambda m, a: m.set_fill(interpolate_color(BLUE, RED, a))),
            ShowCreation(arc),
            run_time=2,
        )
        self.wait(3)
        self.play(
            v_vect.animate.shift(p_vect.get_end() - v_vect.get_start()),
            FadeOut(arc),
            FadeOut(randy),
        )
        rot_group = VGroup(grid, p_vect, v_vect)
        rot_group.set_stroke(background=True)
        self.add(circle, rot_group)
        self.play(
            FadeIn(circle),
            Rotate(
                rot_group,
                2 * TAU,
                axis=normal,
                about_point=axes.get_origin(),
                run_time=4 * TAU,
                rate_func=linear,
            ),
            VFadeIn(grid),
        )


class StoryForAnotherTime(TeacherStudentsScene):
    def construct(self):
        self.teacher_says(
            TexText("The full story\\\\takes more time."),
            bubble_config={"height": 3, "width": 3.5},
            added_anims=[self.change_students("confused", "erm", "hesitant", look_at=self.screen)]
        )
        self.wait(5)


class SchrodingerSum(Scene):
    def construct(self):
        # Add title
        equation_label = VGroup(
            TexText("Schrödinger equation:"),
            Tex(
                "{i} \\hbar \\frac{\\partial}{\\partial t} |\\psi(t)\\rangle = \\hat{H} |\\psi(t)\\rangle",
                tex_to_color_map={
                    "|\\psi(t)\\rangle": BLUE,
                    "{i}": WHITE,
                    "\\frac{\\partial}{\\partial t}": GREY_A,
                },
                font_size=36
            )
        )
        equation_label.arrange(RIGHT, buff=MED_LARGE_BUFF)
        equation_label.set_width(FRAME_WIDTH - 1)
        equation_label.to_edge(UP)
        i_part = equation_label[1].get_part_by_tex("{i}")
        self.add(equation_label)

        # Axes
        x_range = np.array([-5, 5, 1])
        y_range = np.array([-1, 1, 0.5])
        axes = ThreeDAxes(
            x_range=x_range,
            y_range=y_range,
            z_range=y_range,
            height=1.25,
            depth=1.25,
            width=4,
            axis_config={"include_tip": False, "tick_size": 0.05},
        )
        # plane = ComplexPlane(y_range, y_range)
        # plane.rotate(PI / 2, DOWN)
        # plane.match_depth(axes)
        # axes.add(plane)
        # axes.y_axis.set_opacity(0)
        # axes.z_axis.set_opacity(0)

        lil_axes = VGroup(*(axes.deepcopy() for x in range(3)))
        lil_axes.arrange(DOWN, buff=MED_LARGE_BUFF)
        lil_axes.to_corner(DL)
        brace = Brace(lil_axes, RIGHT, buff=MED_LARGE_BUFF)
        axes.scale(2)
        axes.next_to(brace, RIGHT)

        for ax in [axes, *lil_axes]:
            ax.rotate(-30 * DEGREES, UP)
            ax.rotate(10 * DEGREES, RIGHT)
            ax.set_flat_stroke(False)

        # Graphs
        def func0(x, t):
            magnitude = np.exp(-x * x / 2)
            phase = np.exp(complex(0, 0.5 * t))
            return magnitude * phase

        def func1(x, t):
            magnitude = np.exp(-x * x / 2) * 2 * x
            magnitude *= 1 / math.sqrt(2)
            phase = np.exp(complex(0, 1.5 * t))
            return magnitude * phase

        def func2(x, t):
            magnitude = -1 * np.exp(-x * x / 2) * (2 - 4 * x * x)
            magnitude *= 1 / math.sqrt(8)
            phase = np.exp(complex(0, 2.5 * t))
            return magnitude * phase

        def comb_func(x, t):
            return (func0(x, t) + func1(x, t) + func2(x, t)) / 3

        def to_xyz(func, x):
            z = func(x, get_t())
            return (x, z.real, z.imag)

        t_tracker = ValueTracker()
        t_tracker.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(t_tracker)
        get_t = t_tracker.get_value

        def get_graph(axes, func, color):
            fade_tracker = ValueTracker(1)
            result = VGroup()
            graph = always_redraw(lambda: ParametricCurve(
                lambda x: axes.c2p(*to_xyz(func, x)),
                t_range=x_range[:2],
                color=color,
                stroke_opacity=fade_tracker.get_value(),
                flat_stroke=False,
            ))
            result.add(graph)
            for x in np.linspace(0, 1, 100):
                line = Line(stroke_color=color, stroke_width=2)
                line.x = x
                line.axes = axes
                line.graph = graph
                line.fade_tracker = fade_tracker
                line.add_updater(lambda m: m.set_points_as_corners([
                    m.axes.x_axis.pfp(m.x),
                    m.graph.pfp(m.x),
                ]).set_opacity(0.5 * m.fade_tracker.get_value()))
                result.add(line)
            result.fade_tracker = fade_tracker
            return result

        graph0, graph1, graph2, comb_graph = graphs = [
            get_graph(axes, func, color)
            for axes, func, color in zip(
                [*lil_axes, axes],
                [func0, func1, func2, comb_func],
                [BLUE, TEAL, GREEN, YELLOW]
            )
        ]

        lil_axes[1].save_state()
        lil_axes[1].scale(2)
        lil_axes[1].move_to(DOWN)

        self.add(lil_axes[1], graph1)
        self.wait(10)
        self.add(*graphs)
        self.play(
            FadeIn(lil_axes[::2]),
            FadeIn(axes),
            Restore(lil_axes[1]),
            GrowFromCenter(brace),
            *(
                UpdateFromAlphaFunc(g.fade_tracker, lambda m, a: m.set_value(a))
                for g in (*graphs[::2], comb_graph)
            )
        )
        self.wait(15)
        self.play(
            FlashAround(i_part, color=RED),
            i_part.animate.set_color(RED)
        )
        self.wait(30)


class BasicVectorFieldIdea(Scene):
    matrix = [[-1.5, -1], [3, 0.5]]

    def construct(self):
        # Equation
        v_tex = "\\vec{\\textbf{v} }(t)"
        equation = Tex(
            "{d \\over dt}", v_tex, "=", "M", v_tex,
            tex_to_color_map={
                "M": GREY_A,
                v_tex: YELLOW,
            }
        )
        equation.set_height(1.5)
        equation.to_corner(UL, buff=0.25)

        background_rect = SurroundingRectangle(equation, buff=SMALL_BUFF)
        background_rect.set_fill(BLACK, opacity=0.9).set_stroke(WHITE, 2)

        # Plane and field
        matrix = np.array(self.matrix)

        def func(x, y):
            return 0.15 * np.dot(matrix.T, [x, y])

        plane = NumberPlane()
        vector_field = VectorField(
            func, plane,
            magnitude_range=(0, 2),
            vector_config={"thickness": 0.025}
        )
        dots = VGroup(*(
            Dot(v.get_start(), radius=0.02, fill_color=YELLOW)
            for v in vector_field
        ))

        # Velocity and position
        vel_rect = SurroundingRectangle(equation[:2], stroke_color=RED)
        pos_rect = SurroundingRectangle(equation[4], stroke_color=YELLOW)
        pos_rect.match_height(vel_rect, stretch=True)
        pos_rect.match_y(vel_rect)

        vel_words = Text("Velocity", color=RED)
        vel_words.next_to(vel_rect, DOWN)
        vel_words.shift_onto_screen(buff=0.2)
        pos_words = Text("Position", color=YELLOW)
        pos_words.next_to(pos_rect, DOWN)

        for word in vel_words, pos_words:
            word.add_background_rectangle()

        self.add(plane, background_rect, equation)

        self.play(
            ShowCreation(vel_rect),
            FadeIn(vel_words, 0.5 * DOWN),
        )
        self.wait()
        self.play(
            ShowCreation(pos_rect),
            FadeIn(pos_words, 0.5 * DOWN),
        )
        self.wait()
        term_labels = VGroup(vel_rect, vel_words, pos_rect, pos_words)

        foreground = [background_rect, equation, term_labels]
        self.add(dots, *foreground)
        self.play(LaggedStartMap(GrowFromCenter, dots))
        self.add(dots, vector_field, *foreground)
        self.play(LaggedStartMap(GrowArrow, vector_field, lag_ratio=0.01))
        self.wait()
        self.play(
            FadeOut(dots),
            FadeOut(term_labels),
            vector_field.animate.set_opacity(0.25)
        )

        # Show Mv being attached to v
        index = 326
        lil_vect = vector_field[index]
        coords = plane.p2c(lil_vect.get_start())

        dot = Dot(color=YELLOW, radius=0.05)
        dot.move_to(plane.c2p(*coords))
        dot_label = Tex("\\vec{\\textbf{v}}", color=YELLOW)
        dot_label.next_to(dot, UR, SMALL_BUFF)

        vector = Arrow(plane.get_origin(), dot.get_center(), buff=0)
        vector.set_fill(YELLOW, opacity=1)
        vector.set_stroke(BLACK, 0.5)

        Mv = Arrow(plane.get_origin(), plane.c2p(*3 * func(*coords)), buff=0)
        Mv.set_fill(RED, 1)
        Mv_label = Tex("M", "\\vec{\\textbf{v}}")
        Mv_label[0].set_color(GREY_B)
        Mv_label.next_to(Mv.get_end(), DOWN)
        attached_Mv = lil_vect.copy().set_opacity(1)

        self.play(
            GrowFromPoint(dot, equation[-1].get_center()),
            FadeTransform(equation[-1][0].copy(), dot_label),
        )
        self.wait()
        self.play(
            FadeIn(vector),
            ReplacementTransform(vector.copy().set_opacity(0), Mv, path_arc=-45 * DEGREES),
            FadeTransform(equation[3:5].copy(), Mv_label),
        )
        self.wait()
        self.play(
            TransformFromCopy(Mv, attached_Mv)
        )
        self.remove(attached_Mv)
        lil_vect.set_opacity(1)
        self.wait()

        for x in range(100):
            index += 1
            lil_vect = vector_field[index]
            coords = plane.p2c(lil_vect.get_start())
            vector.put_start_and_end_on(plane.get_origin(), plane.c2p(*coords))
            Mv.put_start_and_end_on(plane.get_origin(), plane.c2p(*3 * func(*coords)))
            Mv_label.next_to(Mv.get_end(), DOWN)
            dot.move_to(vector.get_end())
            dot_label.next_to(dot, UR, SMALL_BUFF)
            lil_vect.set_opacity(1)

            if x < 20:
                self.wait(0.25)
            else:
                self.wait(0.1)

        self.play(
            LaggedStartMap(FadeOut, VGroup(Mv_label, dot_label, Mv, vector, dot)),
            vector_field.animate.set_opacity(1)
        )
        self.wait()

        # Show example initial condition evolving
        def get_flow_lines(step_multiple, arc_len):
            return StreamLines(
                func, plane,
                step_multiple=step_multiple,
                magnitude_range=(0, 2),
                color_by_magnitude=False,
                stroke_color=GREY_A,
                stroke_width=2,
                stroke_opacity=1,
                arc_len=arc_len,
            )

        dot = Dot()
        vect = Vector()
        vect.set_color(RED)

        def update_vect(vect):
            coords = plane.p2c(dot.get_center())
            end = plane.c2p(*func(*coords))
            vect.put_start_and_end_on(plane.get_origin(), end)
            vect.shift(dot.get_center() - plane.get_origin())

        vect.add_updater(update_vect)

        flow_line = get_flow_lines(4, 20)[10]
        flow_line.insert_n_curves(100)
        flow_line.set_stroke(width=3)
        dot.move_to(flow_line.get_start())

        self.add(flow_line, vect, dot)
        self.play(
            MoveAlongPath(dot, flow_line, run_time=10, rate_func=linear),
            ShowCreation(flow_line, run_time=10, rate_func=linear),
            VFadeIn(dot),
            VFadeIn(vect),
        )
        self.play(LaggedStartMap(FadeOut, VGroup(flow_line, dot, vect)))
        self.wait()

        # Show exponential solution
        solution = Tex(
            v_tex, "=", "e^{M t}", "\\vec{\\textbf{v} }(0)"
        )

        solution[0].set_color(YELLOW)
        solution[2][1].set_color(GREY_B)
        solution.match_width(equation)
        solution.next_to(equation, DOWN, MED_LARGE_BUFF)

        new_br = SurroundingRectangle(VGroup(equation, solution), buff=MED_SMALL_BUFF)
        new_br.match_style(background_rect)

        self.play(
            Transform(background_rect, new_br),
            FadeIn(solution, 0.5 * DOWN)
        )
        self.wait()
        foreground = VGroup(background_rect, equation, solution)
        self.play(FadeOut(foreground))

        # Show flow of all initial conditions
        initial_points = np.array([
            plane.c2p(x, y)
            for x in np.arange(-16, 16, 0.5)
            for y in np.arange(-6, 6, 0.5)
        ])
        dots = DotCloud(initial_points)
        dots.set_radius(0.06)
        dots.set_color(WHITE)
        initial_points = np.array(dots.get_points())

        self.add(dots)
        self.play(vector_field.animate.set_opacity(0.75))

        time_tracker = ValueTracker()

        def update_dots(dots):
            time = time_tracker.get_value()
            transformation = np.identity(3)
            transformation[:2, :2] = mat_exp(0.15 * matrix * time)
            dots.set_points(np.dot(initial_points, transformation))

        streaks = Group()

        def update_streaks(streaks):
            dc = dots.copy()
            dc.clear_updaters()
            dc.set_opacity(0.25)
            dc.set_radius(0.01)
            streaks.add(dc)

        dots.add_updater(update_dots)
        streaks.add_updater(update_streaks)

        self.add(plane, vector_field, streaks, dots)
        self.play(time_tracker.animate.set_value(3), run_time=6, rate_func=linear)

        # Flow
        # animated_flow = AnimatedStreamLines(get_flow_lines(0.25, 3))


class DefineVectorFieldWithHyperbolicFlow(BasicVectorFieldIdea):
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]]) / 0.45


class MoreShakesperianRomeoJuliet(RomeoAndJuliet):
    def construct(self):
        # Add plane/vector field
        plane = NumberPlane((-5, 5), (-5, 5), faded_line_ratio=0)
        plane.set_height(5)
        plane.to_corner(DL)
        self.add(plane)

        def func0(x, y):
            return (x, y)

        def func1(x, y):
            return (-y, x)

        def func2(x, y):
            return (y, x)

        vector_fields = VGroup(*(
            VectorField(
                func, plane,
                step_multiple=1,
                magnitude_range=(0, 8),
                vector_config={"thickness": 0.025},
                length_func=lambda norm: 0.9 * sigmoid(norm)
            )
            for func in [func0, func1, func2]
        ))

        # Put differential equation above it
        equations = VGroup(
            get_2d_equation([["0", "-1"], ["+1", "0"]]),
            get_2d_equation([["0", "+1"], ["+1", "0"]]),
        )
        equations.to_corner(UL)

        m1 = equations[0][3]
        m2 = equations[1][3]

        self.add(equations[0])
        vector_fields[0].set_opacity(0)
        vf = vector_fields[0]
        self.play(Transform(vf, vector_fields[1]))
        self.wait()

        # Add Romeo and Juliet
        romeo, juliet = lovers = self.get_romeo_and_juliet()
        lovers.set_height(2)
        lovers.arrange(LEFT, buff=0.5)
        lovers.to_corner(DR, buff=1.5)
        self.make_romeo_and_juliet_dynamic(romeo, juliet)

        scales = VGroup(
            self.get_love_scale(romeo, RIGHT, "y", BLUE_D),
            self.get_love_scale(juliet, LEFT, "x", BLUE_B),
        )

        x0, y0 = (5, 5)
        ps_point = Dot(color=BLUE_B)
        ps_point.move_to(plane.c2p(x0, y0))
        romeo.love_tracker.add_updater(lambda m: m.set_value(plane.p2c(ps_point.get_center())[0]))
        juliet.love_tracker.add_updater(lambda m: m.set_value(plane.p2c(ps_point.get_center())[1]))

        self.add(*lovers, scales, self.get_romeo_juilet_name_labels(lovers))
        self.add(*(pi.love_tracker for pi in lovers))
        self.add(*(pi.heart_eyes for pi in lovers))
        self.add(ps_point)
        self.wait()

        # Transition to alternate field
        self.play(
            Transform(vf, vector_fields[2]),
            FadeOut(m1, UP),
            FadeIn(m2, UP),
        )
        self.wait()

        last_rect = VMobject()
        for row in m2.get_rows():
            rect = SurroundingRectangle(row)
            self.play(FadeIn(rect), FadeOut(last_rect))
            self.wait()
            last_rect = rect
        self.play(FadeOut(last_rect))

        self.play(ShowIncreasingSubsets(vf, run_time=6, rate_func=linear))

        # Show flow
        def get_flow_line(x0, y0):
            line = ParametricCurve(
                lambda t: plane.c2p(
                    math.cosh(t) * x0 + math.sinh(t) * y0,
                    math.sinh(t) * x0 + math.cosh(t) * y0,
                ),
                t_range=(0, 4),
            )
            line.set_stroke(WHITE, 3)
            return line

        def move_along_line(line, run_time=8):
            self.add(line, ps_point)
            self.play(
                MoveAlongPath(ps_point, line.copy()),
                ShowCreation(line),
                rate_func=linear,
                run_time=run_time,
            )

        line1 = get_flow_line(4, -3)
        # line2 = get_flow_line(-3, 4)
        line3 = get_flow_line(-4, 3)

        # move_along_line(line1)
        # self.wait()

        ps_point.move_to(line1.get_start())
        self.remove(line1)
        low_vects = VGroup()
        mid_vects = VGroup()
        high_vects = VGroup()
        for vector in vf:
            x, y = plane.p2c(vector.get_start())
            if x + y < -1e-6:
                low_vects.add(vector)
            elif -1e-6 < x + y < 1e-6:
                mid_vects.add(vector)
            else:
                high_vects.add(vector)

        low_vects.set_opacity(0.1)
        mid_vects.set_opacity(0.5)
        self.wait()
        move_along_line(line1)
        self.wait()

        self.remove(line1)
        ps_point.move_to(line3)
        low_vects.set_opacity(1)
        high_vects.set_opacity(0.1)
        move_along_line(line3)
        self.wait()


class NotAllThatRomanticLabel(Scene):
    def construct(self):
        text = TexText("Not all that\n\n romantic!")
        arrow = Vector(LEFT)
        arrow.next_to(text, LEFT)
        text.set_color(RED)
        arrow.set_color(RED)

        self.add(text)
        self.play(ShowCreation(arrow))
        self.wait()


class TransitionWrapper(Scene):
    def construct(self):
        self.add(FullScreenRectangle())
        screens = VGroup(*(ScreenRectangle() for x in range(2)))
        screens.set_width(0.4 * FRAME_WIDTH)
        screens[0].to_edge(LEFT)
        screens[1].to_edge(RIGHT)
        screens.set_fill(BLACK, 1)
        screens[0].set_stroke(BLUE, 2)
        screens[1].set_stroke(GREY_BROWN, 2)
        self.add(screens)

        arrow = Arrow(*screens)
        exp = Tex("e^{Mt}")
        exp.next_to(arrow, UP)
        titles = VGroup(
            TexText("Time: $0$"),
            TexText("Time: $t$"),
        )
        for screen, title in zip(screens, titles):
            title.next_to(screen, UP)
            screen.add(title)

        vf_words = TexText("Vector field defined by $\\vec{\\textbf{v} } \\rightarrow M\\vec{\\textbf{v} }$")
        vf_words.match_width(screens[0])
        vf_words.next_to(screens[0], DOWN)
        self.add(vf_words)

        self.play(ShowCreation(arrow))
        self.play(Write(exp))
        self.wait()


class DerivativeOfExpMt(Scene):
    def construct(self):
        # For all tex
        v0_tex = "\\vec{\\textbf{v} }_0"
        kw = {
            "tex_to_color_map": {
                "M": GREY_B,
                "{t}": YELLOW,
                v0_tex: BLUE,
                "=": WHITE,
                "\\cdots": WHITE,
                "+": WHITE,
                "\\left(": WHITE,
                "\\right)": WHITE,
            }
        }

        # Show claim
        solution = Tex("\\vec{\\textbf{v} }({t}) = e^{M {t} } " + v0_tex, **kw)
        equation = Tex("{d \\over dt} \\vec{\\textbf{v} }({t}) = M \\vec{\\textbf{v} }({t})", **kw)
        arrow = Vector(1.5 * RIGHT)
        top_line = VGroup(solution, arrow, equation)
        top_line.arrange(RIGHT, buff=MED_LARGE_BUFF)
        top_line.set_width(FRAME_WIDTH - 1)

        solves = Text("Solves(?)", font_size=24)
        solves.next_to(arrow, UP, SMALL_BUFF)

        self.add(solution)
        self.play(
            GrowArrow(arrow),
            FadeIn(equation, RIGHT),
            FadeIn(solves, lag_ratio=0.1)
        )
        self.wait()

        arrow.add(solves)

        # Try it...

        # Show calculations...
        tex_expressions = [
            """
            e^{ {t}M } v_0 =
            \\left( {t}^0 M^0 +
            {t}^1 M^1 +
            { {t}^2 \\over 2} M^2 +
            { {t}^3 \\over 6} M^3 +
            \\cdots +
            { {t}^n \\over n!} M^n +
            \\cdots
            \\right) v_0
            """,
            """
            e^{ {t}M } v_0 =
            {t}^0 M^0 v_0 +
            {t}^1 M^1 v_0 +
            { {t}^2 \\over 2} M^2 v_0 +
            { {t}^3 \\over 6} M^3 v_0 +
            \\cdots +
            { {t}^n \\over n!} M^n v_0 +
            \\cdots
            """,
            """
            {d \\over dt}
            e^{ {t}M } v_0 =
            0 +
            1 \\cdot {t}^0 M^1 v_0 +
            {2 {t}^1 \\over 2} M^2 v_0 +
            {3 {t}^2 \\over 6} M^3 v_0 +
            \\cdots +
            {n {t}^{n - 1} \\over n!} M^n v_0 +
            \\cdots
            """,
            """
            {d \\over dt}
            e^{ {t}M } v_0 =
            0 +
            1 \\cdot {t}^0 M^1 v_0 +
            {t}^1 M^2 v_0 +
            { {t}^2 \\over 2} M^3 v_0 +
            \\cdots +
            { {t}^{n - 1} \\over (n - 1)!} M^n v_0 +
            \\cdots
            """,
            """
            {d \\over dt}
            e^{ {t}M } v_0 =
            M \\left(
            {t}^0 M^0 v_0 +
            {t}^1 M^1 v_0 +
            { {t}^2 \\over 2} M^2 v_0 +
            \\cdots +
            { {t}^{n - 1} \\over (n - 1)!} M^{n - 1} v_0 +
            \\cdots
            \\right)
            """,
            """
            {d \\over dt}
            e^{ {t}M } v_0 =
            M \\left( e^{ {t}M } v_0  \\right)
            """
        ]

        lines = VGroup(*(
            Tex(tex.replace("v_0", v0_tex), **kw)
            for tex in tex_expressions
        ))
        lines.set_width(FRAME_WIDTH - 1)
        max_height = 1.0
        for line in lines:
            line.set_width(FRAME_WIDTH - 2)
            if line.get_height() > max_height:
                line.set_height(max_height)
            line.center()

        def match_lines(l1, l2):
            eq_centers = [eq.get_part_by_tex("=").get_center() for eq in (l1, l2)]
            l1.shift(eq_centers[1] - eq_centers[0])

        # Line 0
        self.play(top_line.animate.set_width(6).to_edge(UP))
        lines[0].set_y(1)
        self.play(TransformMatchingTex(solution[4:].copy(), lines[0]), run_time=2)
        self.wait()

        # 0 -> 1
        match_lines(lines[1], lines[0])
        lines[1].set_y(-1)
        self.play(
            TransformMatchingTex(
                VGroup(*(
                    sm
                    for sm in lines[0][5:]
                    if sm.get_tex() not in ["\\left(", "\\right)"]
                )).copy(),
                lines[1][5:]
            ),
            TransformFromCopy(lines[0][:5], lines[1][:5])
        )
        self.play(
            lines[1].animate.set_y(1),
            FadeOut(lines[0], UP)
        )
        self.wait()

        # 1 -> 2 -> 3
        match_lines(lines[2], lines[1])
        match_lines(lines[3], lines[2])
        lines[2:4].set_y(-1)
        self.play(FadeIn(lines[2], DOWN))

        l1_indices, l2_indices, l3_indices = [
            [
                lines[i].index_of_part(part)
                for part in it.chain(lines[i].get_parts_by_tex("="), lines[i].get_parts_by_tex("+"))
            ]
            for i in (1, 2, 3)
        ]

        last_rects = VMobject()
        for l1i, l1j, l2i, l2j in zip(l1_indices, l1_indices[1:], l2_indices, l2_indices[1:]):
            if l1i is l1_indices[4]:
                continue
            r1 = SurroundingRectangle(lines[1][l1i + 1:l1j])
            r2 = SurroundingRectangle(lines[2][l2i + 1:l2j])
            rects = VGroup(r1, r2)
            self.play(FadeIn(rects), FadeOut(last_rects))
            self.wait(0.5)
            last_rects = rects
        self.play(FadeOut(last_rects))
        self.play(
            lines[2].animate.set_y(1),
            FadeOut(lines[1]),
            FadeIn(lines[3]),
        )
        self.wait()

        last_rects = VMobject()
        for l2i, l2j, l3i, l3j in zip(l2_indices, l2_indices[1:], l3_indices, l3_indices[1:]):
            # Such terrible style...please no on look
            if l2i in [l2_indices[0], l2_indices[1], l2_indices[4]]:
                continue
            r2 = SurroundingRectangle(lines[2][l2i + 1:l2j])
            r3 = SurroundingRectangle(lines[3][l3i + 1:l3j])
            rects = VGroup(r2, r3)
            self.play(FadeIn(rects), FadeOut(last_rects))
            self.wait(0.5)
            last_rects = rects
        self.play(FadeOut(last_rects))
        self.wait()

        # 3 -> 4
        match_lines(lines[4], lines[3])
        lines[4].set_y(-1)

        self.play(
            FadeIn(lines[1], DOWN),
            FadeOut(lines[2], DOWN)
        )
        self.wait()
        self.play(LaggedStart(*(
            FlashUnder(sm, time_width=2, run_time=1)
            for sm in lines[3].get_parts_by_tex("M")[1:]
        ), lag_ratio=0.2))
        self.wait()

        self.play(
            TransformMatchingTex(lines[3][:5], lines[4][:5]),
            FadeTransform(lines[3][5:], lines[4][7:34]),
            FadeIn(lines[4].get_part_by_tex("\\left(")),
            FadeIn(lines[4].get_part_by_tex("\\right)")),
            TransformFromCopy(
                lines[3].get_parts_by_tex("M")[1:],
                VGroup(lines[4][5]),
                path_arc=-45 * DEGREES,
            )
        )
        self.wait()

        # 4 -> 5
        match_lines(lines[5], lines[4])
        lines[5].set_y(-3)

        self.play(
            TransformFromCopy(lines[4][:7], lines[5][:7]),
            TransformFromCopy(lines[4][34], lines[5][11]),
            FadeTransform(lines[4][7:34].copy(), lines[5][7:11], stretch=False),
        )
        self.wait()


class TryIt(Scene):
    def construct(self):
        morty = Mortimer()
        morty.to_corner(DR)

        self.add(morty)

        self.play(PiCreatureSays(morty, "Try it!", target_mode="surprised", run_time=1))
        self.play(Blink(morty))
        self.wait(2)

        self.remove(morty.bubble, morty.bubble.content)
        self.play(PiCreatureSays(morty, "Brace yourself now...", target_mode="hesitant"))
        morty.look(ORIGIN)
        self.wait(2)


class TracePropertyAndComputation(TeacherStudentsScene):
    def construct(self):
        trace_eq = Tex(
            "\\text{Det}\\left(e^{Mt}\\right) = e^{\\text{Tr}(M) t}",
            tex_to_color_map={
                "\\text{Det}": GREEN_D,
                "\\text{Tr}": RED_B,
            }
        )
        trace_eq.move_to(self.hold_up_spot, DOWN)

        self.play(
            self.teacher.change("raise_right_hand", trace_eq),
            FadeIn(trace_eq, 0.5 * UP),
        )
        self.play_student_changes("pondering", "confused", "pondering", look_at=trace_eq)
        self.wait(2)

        text = TexText("Diagonalization $\\rightarrow$ Easier computation")
        text.move_to(self.hold_up_spot, DOWN)
        text.shift_onto_screen()

        self.play(
            trace_eq.animate.shift(UP),
            FadeIn(text, 0.5 * UP),
            self.change_students("erm", "tease", "maybe"),
        )
        for pi in self.pi_creatures:  # Why?
            pi.eyes[0].refresh_bounding_box()
            pi.eyes[1].refresh_bounding_box()
        self.look_at(text)
        self.wait(3)

        topics = VGroup(trace_eq, text)

        exp_deriv = Tex("e", "{d \\over dx}")
        exp_deriv[0].scale(2)
        exp_deriv[1].move_to(exp_deriv[0].get_corner(UR), DL)
        exp_deriv.move_to(self.hold_up_spot, DOWN)

        self.play(
            FadeIn(exp_deriv, scale=2),
            topics.animate.scale(0.5).to_corner(UL),
            self.teacher.change("tease", exp_deriv),
        )
        self.play_student_changes("confused", "sassy", "angry")
        self.wait(6)


class EndScreen(PatreonEndScreen):
    pass


# GENERIC flow scenes


class ExponentialPhaseFlow(Scene):
    CONFIG = {
        "field_config": {
            "color_by_magnitude": False,
            "magnitude_range": (0.5, 5),
            "arc_len": 5,
        },
        "plane_config": {
            "x_range": [-4, 4],
            "y_range": [-2, 2],
            "height": 8,
            "width": 16,
        },
        "matrix": [
            [1, 0],
            [0, 1],
        ],
        "label_height": 3,
        "run_time": 30,
        "slow_factor": 0.25,
    }

    def construct(self):
        mr = np.array(self.field_config["magnitude_range"])
        self.field_config["magnitude_range"] = self.slow_factor * mr
        plane = NumberPlane(**self.plane_config)
        plane.add_coordinate_labels()

        vector_field, animated_lines = get_vector_field_and_stream_lines(
            self.func, plane,
            **self.field_config,
        )

        box = Square()
        box.replace(Line(plane.c2p(-1, -1), plane.c2p(1, 1)), stretch=True)
        box.set_stroke(GREY_A, 1)
        box.set_fill(BLUE_E, 0.8)
        move_points_along_vector_field(box, self.func, plane)

        basis_vectors = VGroup(
            Vector(RIGHT, fill_color=GREEN),
            Vector(UP, fill_color=RED),
        )
        basis_vectors[0].add_updater(lambda m: m.put_start_and_end_on(
            plane.get_origin(),
            box.pfp(7 / 8)
        ))
        basis_vectors[1].add_updater(lambda m: m.put_start_and_end_on(
            plane.get_origin(),
            box.pfp(1 / 8)
        ))

        self.add(plane)
        self.add(vector_field)
        self.add(animated_lines)
        self.add(box)
        self.add(*basis_vectors)
        self.wait(self.run_time)

    def func(self, x, y):
        return self.slow_factor * np.dot([x, y], np.transpose(self.matrix))

    def get_label(self):
        exponential = get_matrix_exponential(self.matrix)
        changing_t = DecimalNumber(0, color=YELLOW)
        changing_t.match_height(exponential[2])
        changing_t.move_to(exponential[2], DL)
        exponential.replace_submobject(2, changing_t)

        equals = Tex("=")
        rhs = DecimalMatrix(
            np.zeros((2, 2)),
            element_to_mobject_config={"num_decimal_places": 3},
            h_buff=1.8,
        )
        rhs.match_height(exponential)

        equation = VGroup(
            exponential,
            equals,
            rhs,
        )
        equation.arrange(RIGHT)
        equation.to_corner(UL)


class ExponentialEvaluationWithTime(Scene):
    flow_scene_class = ExponentialPhaseFlow

    def construct(self):
        flow_scene_attrs = merge_dicts_recursively(
            ExponentialPhaseFlow.CONFIG,
            self.flow_scene_class.CONFIG,
        )
        matrix = np.array(flow_scene_attrs["matrix"])
        slow_factor = flow_scene_attrs["slow_factor"]

        def get_t():
            return slow_factor * self.time

        exponential = get_matrix_exponential(matrix)
        dot = Tex("\\cdot")
        dot.move_to(exponential[2], LEFT)
        changing_t = DecimalNumber(0)
        changing_t.match_height(exponential[2])
        changing_t.next_to(dot, RIGHT, SMALL_BUFF)
        changing_t.align_to(exponential[1], DOWN)
        changing_t.add_updater(lambda m: m.set_value(get_t()).set_color(YELLOW))
        lhs = VGroup(*exponential[:2], dot, changing_t)

        equals = Tex("=")
        rhs = DecimalMatrix(
            np.zeros((2, 2)),
            element_to_mobject_config={"num_decimal_places": 2},
            element_alignment_corner=ORIGIN,
            h_buff=2.0,
        )
        for mob in rhs.get_entries():
            mob.edge_to_fix = ORIGIN

        rhs.match_height(lhs)

        def update_rhs(rhs):
            result = mat_exp(matrix * get_t())
            for mob, value in zip(rhs.get_entries(), result.flatten()):
                mob.set_value(value)
            return rhs

        rhs.add_updater(update_rhs)

        equation = VGroup(lhs, equals, rhs)
        equation.arrange(RIGHT)
        equation.center()

        self.add(equation)
        self.wait(flow_scene_attrs["run_time"])
        self.embed()


class CircularPhaseFlow(ExponentialPhaseFlow):
    CONFIG = {
        "field_config": {
            "magnitude_range": (0.5, 8),
        },
        "matrix": [
            [0, -1],
            [1, 0],
        ]
    }


class CircularFlowEvaluation(ExponentialEvaluationWithTime):
    flow_scene_class = CircularPhaseFlow


class EllipticalPhaseFlow(ExponentialPhaseFlow):
    CONFIG = {
        "field_config": {
            "magnitude_range": (0.5, 8),
        },
        "matrix": [
            [0.5, -3],
            [1, -0.5],
        ]
    }


class EllipticalFlowEvaluation(ExponentialEvaluationWithTime):
    flow_scene_class = EllipticalPhaseFlow


class HyperbolicPhaseFlow(ExponentialPhaseFlow):
    CONFIG = {
        "field_config": {
            "sample_freq": 8,
        },
        "matrix": [
            [1, 0],
            [0, -1],
        ]
    }


class HyperbolicFlowEvaluation(ExponentialEvaluationWithTime):
    flow_scene_class = HyperbolicPhaseFlow


class ShearPhaseFlow(ExponentialPhaseFlow):
    CONFIG = {
        "field_config": {
            "sample_freq": 2,
            "magnitude_range": (0.5, 8),
        },
        "plane_config": {
            "x_range": [-8, 8],
            "y_range": [-4, 4],
        },
        "matrix": [
            [1, 1],
            [0, 1],
        ],
        "slow_factor": 0.1,
    }


class ShearFlowEvaluation(ExponentialEvaluationWithTime):
    flow_scene_class = ShearPhaseFlow


class HyperbolicTrigFlow(ExponentialPhaseFlow):
    CONFIG = {
        "field_config": {
            "sample_freq": 2,
            "magnitude_range": (0.5, 7),
        },
        "plane_config": {
            "x_range": [-8, 8],
            "y_range": [-4, 4],
        },
        "matrix": [
            [0, 1],
            [1, 0],
        ],
        "slow_factor": 0.1,
    }


class HyperbolicTrigFlowEvaluation(ExponentialEvaluationWithTime):
    flow_scene_class = HyperbolicTrigFlow


class DampedRotationPhaseFlow(ExponentialPhaseFlow):
    CONFIG = {
        "matrix": [
            [-1, -1],
            [1, 0],
        ],
    }


class DampedRotationFlowEvaluation(ExponentialEvaluationWithTime):
    flow_scene_class = DampedRotationPhaseFlow


class FrameForFlow(Scene):
    def construct(self):
        self.add(FullScreenRectangle(fill_color=GREY_D))
        screen_rect = ScreenRectangle()
        screen_rect.set_height(5.5)
        screen_rect.set_stroke(WHITE, 3)
        screen_rect.set_fill(BLACK, 1)
        screen_rect.to_edge(DOWN)
        self.add(screen_rect)


class ThumbnailBackdrop(DampedRotationPhaseFlow):
    CONFIG = {
        "run_time": 10,
    }

    def construct(self):
        super().construct()

        for mob in self.mobjects:
            if isinstance(mob, Square) or isinstance(mob, Arrow):
                self.remove(mob)
            if isinstance(mob, NumberPlane):
                self.remove(mob.coordinate_labels)


class Thumbnail(Scene):
    def construct(self):
        im = ImageMobject("ExpMatThumbnailBackdrop")
        im.set_height(FRAME_HEIGHT)
        im.set_opacity(0.7)
        self.add(im)

        # rect = FullScreenFadeRectangle()
        # rect.set_fill(opacity=0.3)
        # self.add(rect)

        exp = get_matrix_exponential([[-1, -1], [1, 0]], scalar_tex="")
        exp.set_height(5)
        exp.set_stroke(BLACK, 50, opacity=0.5, background=True)

        fuzz = VGroup()
        N = 100
        for w in np.linspace(150, 0, N):
            ec = exp.copy()
            ec.set_stroke(BLUE_E, width=w, opacity=(1 / N))
            ec.set_fill(opacity=0)
            fuzz.add(ec)

        self.add(fuzz, exp)

        self.embed()


# Older

class LetsSumUp(TeacherStudentsScene):
    def construct(self):
        self.teacher_says(
            "Let's review",
            added_anims=[self.change_students("thinking", "pondering", "thinking")]
        )
        self.wait(3)


class PrerequisitesWrapper(Scene):
    def construct(self):
        self.add(FullScreenRectangle())
        title = Text("Helpful background knowledge")
        title.to_edge(UP)
        self.add(title)

        screens = VGroup(*(ScreenRectangle() for x in range(2)))
        screens.arrange(RIGHT, buff=LARGE_BUFF)
        screens.set_width(FRAME_WIDTH - 1)
        screens.move_to(DOWN)
        screens.set_fill(BLACK, 1)
        screens.set_stroke(WHITE, 2)

        topics = VGroup(
            TexText("Basics of $e^x$"),
            TexText("How matrices act\\\\as transformations"),
        )
        for topic, screen in zip(topics, screens):
            topic.next_to(screen, UP)
            topic.set_color(WHITE)

        for topic, screen in zip(topics, screens):
            sc = screen.copy()
            sc.set_fill(opacity=0)
            sc.set_stroke(width=3)
            self.play(
                FadeIn(topic, 0.5 * UP),
                FadeIn(screen),
                VShowPassingFlash(sc, time_width=1.0, run_time=1.5),
            )
            self.wait(2)


class SchroedingersComplicatingFactors(TeacherStudentsScene):
    def construct(self):
        pass


class OldComputationCode(Scene):
    def construct(self):
        # Taylor series example
        ex_rhs = Tex(
            """
            {2}^0 +
            {2}^1 +
            { {2}^2 \\over 2} +
            { {2}^3 \\over 6} +
            { {2}^4 \\over 24} +
            { {2}^5 \\over 120} +
            { {2}^6 \\over 720} +
            { {2}^7 \\over 5040} +
            \\cdots
            """,
            tex_to_color_map={"{2}": YELLOW, "+": WHITE},
        )
        ex_rhs.next_to(real_equation[3:], DOWN, buff=0.75)

        ex_parts = VGroup(*(
            ex_rhs[i:j] for i, j in [
                (0, 2),
                (3, 5),
                (6, 8),
                (9, 11),
                (12, 14),
                (15, 17),
                (18, 20),
                (21, 23),
                (24, 25),
            ]
        ))
        term_brace = Brace(ex_parts[0], DOWN)
        frac = Tex("1", font_size=36)
        frac.next_to(term_brace, DOWN, SMALL_BUFF)

        rects = VGroup(*(
            Rectangle(height=2**n / math.factorial(n), width=1)
            for n in range(11)
        ))
        rects.arrange(RIGHT, buff=0, aligned_edge=DOWN)
        rects.set_fill(opacity=1)
        rects.set_submobject_colors_by_gradient(BLUE, GREEN)
        rects.set_stroke(WHITE, 1)
        rects.set_width(7)
        rects.to_edge(DOWN)

        self.play(
            ReplacementTransform(taylor_brace, term_brace),
            FadeTransform(real_equation[3:].copy(), ex_rhs),
            FadeOut(false_group, shift=DOWN),
            FadeOut(taylor_label, shift=DOWN),
            FadeIn(frac),
        )
        term_values = VGroup()
        for n in range(11):
            rect = rects[n]
            fact = math.factorial(n)
            ex_part = ex_parts[min(n, len(ex_parts) - 1)]
            value = DecimalNumber(2**n / fact)
            value.set_color(GREY_A)
            max_width = 0.6 * rect.get_width()
            if value.get_width() > max_width:
                value.set_width(max_width)
            value.next_to(rects[n], UP, SMALL_BUFF)
            new_brace = Brace(ex_part, DOWN)
            if fact == 1:
                new_frac = Tex(f"{2**n}", font_size=36)
            else:
                new_frac = Tex(f"{2**n} / {fact}", font_size=36)
            new_frac.next_to(new_brace, DOWN, SMALL_BUFF)
            self.play(
                term_brace.animate.become(new_brace),
                FadeTransform(frac, new_frac),
            )
            frac = new_frac
            rect.save_state()
            rect.stretch(0, 1, about_edge=DOWN)
            rect.set_opacity(0)
            value.set_value(0)
            self.play(
                Restore(rect),
                ChangeDecimalToValue(value, 2**n / math.factorial(n)),
                UpdateFromAlphaFunc(value, lambda m, a: m.next_to(rect, UP, SMALL_BUFF).set_opacity(a)),
                randy.animate.look_at(rect),
                morty.animate.look_at(rect),
            )
            term_values.add(value)
        self.play(FadeOut(frac))

        new_brace = Brace(ex_rhs, DOWN)
        sum_value = DecimalNumber(math.exp(2), num_decimal_places=4, font_size=36)
        sum_value.next_to(new_brace, DOWN)
        self.play(
            term_brace.animate.become(new_brace),
            randy.change("thinking", sum_value),
            morty.change("tease", sum_value),
            *(FadeTransform(dec.copy().set_opacity(0), sum_value) for dec in term_values)
        )
        self.play(Blink(randy))

        lhs = Tex("e \\cdot e =")
        lhs.match_height(real_equation[0])
        lhs.next_to(ex_rhs, LEFT)
        self.play(Write(lhs))
        self.play(Blink(morty))
        self.play(Blink(randy))

        # Increment input
        twos = ex_rhs.get_parts_by_tex("{2}")
        threes = VGroup(*(
            Tex("3").set_color(YELLOW).replace(two)
            for two in twos
        ))
        new_lhs = Tex("e \\cdot e \\cdot e = ")
        new_lhs.match_height(lhs)
        new_lhs[0].space_out_submobjects(0.8)
        new_lhs[0][-1].shift(SMALL_BUFF * RIGHT)
        new_lhs.move_to(lhs, RIGHT)

        anims = []
        unit_height = 0.7 * rects[0].get_height()
        for n, rect, value_mob in zip(it.count(0), rects, term_values):
            rect.generate_target()
            new_value = 3**n / math.factorial(n)
            rect.target.set_height(unit_height * new_value, stretch=True, about_edge=DOWN)
            value_mob.rect = rect
            anims += [
                MoveToTarget(rect),
                ChangeDecimalToValue(value_mob, new_value),
                UpdateFromFunc(value_mob, lambda m: m.next_to(m.rect, UP, SMALL_BUFF))
            ]

        self.play(
            FadeOut(twos, 0.5 * UP),
            FadeIn(threes, 0.5 * UP),
        )
        twos.set_opacity(0)
        self.play(
            ChangeDecimalToValue(sum_value, math.exp(3)),
            *anims,
        )
        self.play(
            FadeOut(lhs, 0.5 * UP),
            FadeIn(new_lhs, 0.5 * UP),
        )
        self.wait()


class PreviewVisualizationWrapper(Scene):
    def construct(self):
        background = FullScreenFadeRectangle(fill_color=GREY_E, fill_opacity=1)
        self.add(background)

        screen = ScreenRectangle(height=6)
        screen.set_fill(BLACK, 1)
        screen.set_stroke(GREY_A, 3)
        screen.to_edge(DOWN)
        self.add(screen)

        titles = VGroup(
            Text("How to think about matrix exponentiation"),
            Text(
                "How to visualize matrix exponentiation",
                t2s={"visualize": ITALIC},
            ),
            Text("What problems matrix exponentiation solves?"),
        )
        for title in titles:
            title.next_to(screen, UP)
            title.get_parts_by_text("matrix exponentiation").set_color(TEAL)

        self.play(FadeIn(titles[0], 0.5 * UP))
        self.wait(2)
        self.play(*(
            FadeTransform(
                titles[0].get_parts_by_text(w1),
                titles[1].get_parts_by_text(w2),
            )
            for w1, w2 in [
                ("How to", "How to"),
                ("think about", "visualize"),
                ("matrix exponentiation", "matrix exponentiation"),
            ]
        ))
        self.wait(2)
        self.play(
            *(
                FadeTransform(
                    titles[1].get_parts_by_text(w1),
                    titles[2].get_parts_by_text(w2),
                )
                for w1, w2 in [
                    ("How to visualize", "What problems"),
                    ("matrix exponentiation", "matrix exponentiation"),
                ]
            ),
            FadeIn(titles[2].get_parts_by_text("solves?"))
        )
        self.wait(2)
