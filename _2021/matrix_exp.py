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


# Generic fLow scenes


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
        "matrix": [
            [0, -1],
            [1, 0],
        ]
    }


class CircularFlowEvaluation(ExponentialEvaluationWithTime):
    flow_scene_class = CircularPhaseFlow


class EllipticalPhaseFlow(ExponentialPhaseFlow):
    CONFIG = {
        "matrix": [
            [0, -3],
            [1, 0],
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

# Video scenes


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
            randy.animate.change("pondering", matrix),
            Write(matrix.get_brackets()),
            ShowIncreasingSubsets(matrix.get_entries()),
        )
        self.play(
            matrix.animate.restore(),
            Write(base),
            randy.animate.change("erm", base),
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
            bubble_class=SpeechBubble,
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
        )
        self.play(
            randy.animate.change("angry", rhs),
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
        real_label.next_to(xs[0], DOWN, buff=0.8)
        real_label.to_edge(LEFT, buff=MED_SMALL_BUFF)
        real_arrow = Arrow(real_label, xs[0], buff=0.1, fill_color=GREY_B, thickness=0.025)

        taylor_brace = Brace(real_rhs, DOWN)
        taylor_label = taylor_brace.get_text("Taylor series")

        self.play(
            TransformFromCopy(base, real_equation[0]),
            FadeTransform(matrix.copy(), real_equation[1]),
            FadeIn(real_label, UR),
            GrowArrow(real_arrow),
            randy.animate.change("thinking", real_label),
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
            randy.animate.change("pondering", real_equation),
            morty.animate.change("pondering", real_equation),
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
            randy.animate.change("thinking", sum_value),
            morty.animate.change("tease", sum_value),
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
            randy.animate.change("erm", real_equation),
            morty.animate.change("thinking", real_equation),
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
            morty.animate.change("raise_right_hand", pii_rhs),
            FadeTransformPieces(real_rhs.copy(), pii_rhs),
        )
        self.play(Blink(randy))
        self.play(
            FadeTransformPieces(real_rhs.copy(), mat_rhs),
        )
        self.play(
            randy.animate.change("maybe", mat_rhs),
        )
        self.wait()

        why = Text("Why?", font_size=36)
        why.next_to(randy, UP, aligned_edge=LEFT)
        self.play(
            randy.animate.change("confused", mat_rhs.get_corner(UL)),
            Write(why),
        )
        self.play(Blink(randy))

        reassurance = VGroup(
            Text("I know it looks complicated,", font_size=24),
            Text("don't panic, it'll be okay", font_size=24),
        )
        reassurance.arrange(DOWN)
        reassurance.next_to(morty, LEFT, aligned_edge=UP)
        reassurance.set_color(GREY_A)

        for words in reassurance:
            self.play(FadeIn(words))
        self.play(Blink(morty))

        # Matrix powers
        top_parts = VGroup(real_lhs, real_rhs, pii_rhs)
        top_parts.save_state()
        original_mat_rhs = mat_rhs.copy()

        self.play(
            mat_rhs.animate.set_width(FRAME_WIDTH - 1).center().to_edge(UP),
            FadeOut(top_parts, shift=2 * UP),
            LaggedStartMap(FadeOut, VGroup(why, *reassurance), shift=0.1 * LEFT),
        )

        mat = mat_rhs[4]
        mat_brace = Brace(VGroup(mat, mat_rhs[5][0]), DOWN, buff=SMALL_BUFF)
        matrix = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
        matrix_square = np.dot(matrix, matrix)
        result = IntegerMatrix(matrix_square, h_buff=1.3, v_buff=0.7)
        result.match_height(mat)
        square_eq = VGroup(mat.copy(), mat.copy(), Tex("="), result)
        square_eq.arrange(RIGHT, buff=SMALL_BUFF)
        square_eq.next_to(mat_brace, DOWN)

        m1_terms = square_eq[0][2:11]
        m2_terms = square_eq[1][2:11]
        rhs_terms = result.elements

        self.play(GrowFromCenter(mat_brace))
        self.play(
            LaggedStart(
                TransformFromCopy(mat, square_eq[0], path_arc=45 * DEGREES),
                TransformFromCopy(mat, square_eq[1]),
                Write(square_eq[2]),
                Write(result.brackets),
            ),
            randy.animate.change("pondering", square_eq),
        )
        for n in range(9):
            i = n // 3
            j = n % 3
            row = m1_terms[3 * i:3 * i + 3]
            col = m2_terms[j::3]
            row_rect = SurroundingRectangle(row, buff=0.05)
            col_rect = SurroundingRectangle(col, buff=0.05)
            row_rect.set_stroke(YELLOW, 2)
            col_rect.set_stroke(YELLOW, 2)
            right_elem = Integer(matrix_square[i, j])
            right_elem.replace(rhs_terms[n], dim_to_match=1)
            right_elem.set_value(0)

            self.add(row_rect, col_rect, right_elem)
            for k in range(3):
                self.wait(0.1)
                right_elem.increment_value(matrix[i, k] * matrix[k, j])
                row[k].set_color(YELLOW)
                col[k].set_color(YELLOW)
            self.remove(right_elem)
            self.add(rhs_terms[n])
            self.wait(0.35)
            m1_terms.set_color(TEAL)
            m2_terms.set_color(TEAL)
            self.remove(row_rect, col_rect)

        # Show matrix cubed
        mat_brace.generate_target()
        mat_brace.target.next_to(mat_rhs[6], DOWN, SMALL_BUFF)

        mat_cubed = IntegerMatrix(
            np.dot(matrix, matrix_square),
            h_buff=1.8, v_buff=0.7,
            element_alignment_corner=ORIGIN,
        )
        mat_cubed.match_height(mat)
        cube_eq = VGroup(mat.copy(), mat.copy(), mat.copy(), Tex("="), mat_cubed)
        cube_eq.arrange(RIGHT, buff=SMALL_BUFF)
        cube_eq.next_to(mat_brace.target, DOWN)

        self.play(
            MoveToTarget(mat_brace),
            ReplacementTransform(square_eq, cube_eq),
        )
        self.play(randy.animate.change("tease"))
        self.play(morty.animate.change("happy"))
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
        )
        self.wait()

        # Adding
        mat1 = np.array([[2, 1, 7], [1, 8, 2], [8, 1, 8]])
        mat2 = np.array([[2, 8, 4], [5, 9, 0], [4, 5, 2]])

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
            morty.animate.change("raise_right_hand"),
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
            randy.animate.change("sassy", mat_rhs),
            morty.animate.change("guilty", randy.eyes),
        )
        self.play(Blink(randy))
        self.wait()
        self.play(
            FadeOut(bubble),
            bubble.content.animate.next_to(randy, RIGHT, aligned_edge=UP),
            randy.animate.change("pondering", mat_rhs),
            morty.animate.change("pondering", mat_rhs),
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
            morty.animate.change("raise_right_hand"),
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
            morty.animate.change("tease"),
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
            randy.animate.change("erm", curr_sum_mob),
        )

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
                h_buff = 1.2 * sample.get_width()
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
            randy.animate.change("confused", curr_sum_mob),
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
            randy.animate.change("maybe"),
            FadeIn(why, UP),
            FadeOut(bubble.content, UP),
        )
        self.wait()
        self.play(
            morty.animate.change("raise_right_hand"),
            FadeIn(epii, UP),
        )
        self.play(Blink(morty))
        self.play(
            Write(later_text, run_time=1),
            randy.animate.change("hesitant", morty.eyes)
        )

        # Show partial sums
        new_pi_mat_rhs = Tex(
            rhs_tex.replace("X", pi_mat_tex),
            tex_to_color_map={pi_mat_tex: BLUE},
            isolate=["+"]
        )
        new_pi_mat_rhs.replace(pi_mat_rhs)
        self.add(new_pi_mat_rhs)
        self.remove(pi_mat_rhs)

        for psm in partial_sum_mobs:
            psm.move_to(new_sum_mob)

        self.play(
            LaggedStartMap(
                FadeOut, VGroup(
                    why, epii, later_text,
                ),
                shift=DOWN,
            ),
            FadeOut(curr_sum_mob),
            FadeIn(partial_sum_mobs[0]),
            new_pi_mat_rhs[2:].animate.set_opacity(0.2),
            randy.animate.change("pondering", new_pi_mat_rhs),
            morty.animate.change("pondering", new_pi_mat_rhs),
        )
        for n, k in zip(it.count(1), [5, 9, 13, 19, 21]):
            self.remove(partial_sum_mobs[n - 1])
            self.add(partial_sum_mobs[n])
            new_pi_mat_rhs[:k].set_opacity(1)
            self.wait()
        brace.become(Brace(new_pi_mat_rhs, DOWN))
        for n in range(6, 18):
            self.remove(partial_sum_mobs[n - 1])
            self.add(partial_sum_mobs[n])
            self.wait(0.25)

        # Describe exp
        mat_rhs.become(original_mat_rhs)
        real_lhs.set_opacity(1)
        real_label.to_corner(UL, buff=MED_SMALL_BUFF)
        real_arrow.become(Arrow(real_label, real_equation[1], buff=0.1))

        VGroup(real_lhs, real_rhs, mat_rhs, pii_rhs).to_edge(RIGHT, buff=SMALL_BUFF)
        mat_rhs.to_edge(RIGHT, buff=SMALL_BUFF)

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
            FadeIn(real_equation),
            FadeIn(real_arrow),
            FadeIn(real_label),
            FadeIn(pii_rhs),
            FadeIn(mat_rhs),
            FadeOut(new_pi_mat_rhs),
            FadeOut(brace),
            FadeOut(partial_sum_mobs[n]),
        )
        self.play(
            FadeIn(pii_lhs),
            randy.animate.change("thinking", pii_lhs),
            randy.animate.change("tease", pii_lhs),
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
            randy.animate.change("raise_left_hand", pii_lhs),
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
            randy.animate.change("hesitant", crosses),
        )
        self.play(Blink(randy))

        self.play(
            FadeOut(pii_lhs),
            FadeOut(mat_lhs),
            FadeOut(crosses),
            *(MoveToTarget(power) for power in powers),
            *(TransformFromCopy(real_equation[0], base) for base in bases),
            Write(equals),
            randy.animate.change("sassy", powers),
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
        real_label = Text("Theorem")
        real_label.next_to(real_rect, UP)

        def_rect = SurroundingRectangle(def_parts)
        def_rect.set_stroke(BLUE, 2)
        def_label = Text("Definition")
        def_label.next_to(def_rect, UP)

        self.play(
            ShowCreation(real_rect),
            FadeIn(real_label, 0.5 * UP),
        )
        self.wait()
        self.play(
            ShowCreation(def_rect),
            FadeIn(def_label, 0.5 * UP),
        )
        self.wait()


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

        name_labels = VGroup(*(
            Text(name, font_size=36)
            for name in ["Romeo", "Juliet"]
        ))
        for label, creature in zip(name_labels, lovers):
            label.next_to(creature, DOWN)
            creature.name_label = label
        name_labels.space_out_submobjects(1.2)

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
        y_rect_copy.replace(romeo.scale.dot, stretch=True)
        self.play(FadeIn(y_rect))
        self.wait()
        self.play(TransformFromCopy(y_rect, y_rect_copy))
        y_rect_copy.add_updater(lambda m: m.move_to(romeo.scale.dot))
        self.wait()
        self.play(romeo.love_tracker.animate.set_value(-3))

        big_arrow = Arrow(
            juliet.scale.number_line.get_bottom(),
            juliet.scale.number_line.get_top(),
        )
        big_arrow.set_color(GREEN)
        big_arrow.next_to(juliet.scale.number_line, LEFT)

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
        x_rect_copy.replace(juliet.scale.dot, stretch=True)
        self.play(ShowCreation(dy_rect))
        self.wait()
        self.play(TransformFromCopy(dy_rect, x_rect))
        self.play(TransformFromCopy(x_rect, x_rect_copy))
        self.wait()

        big_arrow.next_to(romeo.scale.number_line, RIGHT)
        self.play(FadeIn(big_arrow), LaggedStartMap(FadeOut, VGroup(dy_rect, x_rect)))
        self.play(romeo.love_tracker.animate.set_value(-3), run_time=4, rate_func=linear)
        x_rect_copy.add_updater(lambda m: m.move_to(juliet.scale.dot))
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
        globals().update(locals())
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

            pi.look_at(lover.eyes)
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

    def make_romeo_and_juliet_dynamic(self, romeo, juliet):
        cutoff_values = [-5, -3, -1, 0, 1, 3, 5]
        modes = ["angry", "sassy", "hesitant", "plain", "happy", "hooray", "surprised"]
        self.make_character_dynamic(romeo, juliet, cutoff_values, modes)
        self.make_character_dynamic(juliet, romeo, cutoff_values, modes)

    def get_romeo_and_juliet(self):
        romeo = PiCreature(color=BLUE_E, flip_at_start=True)
        juliet = PiCreature(color=BLUE_B)
        return VGroup(romeo, juliet)

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
        creature.scale = result

        return result


class DiscussSystem(Scene):
    def construct(self):
        equations = VGroup(
            Tex("{dx \\over dt} {{=}} -{{y(t)}}"),
            Tex("{dy \\over dt} {{=}} {{x(t)}}"),
        )
        equations.arrange(RIGHT, buff=LARGE_BUFF)
        equations.to_edge(UP, buff=1.5)

        eq_rect = SurroundingRectangle(equation, stroke_width=2)
        sys_label = Text("System of differential equations")
        sys_label.next_to(eq_rect, UP)

        self.add(equations)

        self.embed()


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


class VectorChangingInTime(Scene):
    def construct(self):
        # (Setup vector equation)
        matrix = np.array([
            [0, -3],
            [1, 0],
        ])
        deriv = Tex("d \\over dt", tex_to_color_map={"t": GREY_B})
        vect_sym = Matrix(
            [["x(t)"], ["y(t)"]],
            bracket_h_buff=SMALL_BUFF,
            bracket_v_buff=SMALL_BUFF,
            element_to_mobject_config={
                "tex_to_color_map": {"t": GREY_B},
                "isolate": ["(", ")"]
            }
        )
        deriv.match_height(vect_sym)
        equals = Tex("=")
        matrix = IntegerMatrix(matrix, h_buff=0.8)
        matrix.set_color(TEAL)
        equation = VGroup(deriv, vect_sym, equals, matrix, vect_sym.deepcopy())
        equation.arrange(RIGHT)
        equation.center()
        equation.to_edge(UP, buff=1.0)

        vect_sym.save_state()

        # (Setup plane)
        plane = NumberPlane(
            x_range=(-4, 4),
            y_range=(-2, 2),
            height=4,
            width=8,
        )
        plane.to_edge(DOWN)

        point = Point(plane.c2p(3, 0.5))
        vector = Arrow(plane.get_origin(), point.get_location(), buff=0)
        vector.set_color(YELLOW)

        # In short, matrix exponentiation comes up whenever you have a vector
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

        # changing with time
        def func(x, y):
            return 0.2 * np.dot([x, y], matrix.T)

        move_points_along_vector_field(point, func, plane)
        globals().update(locals())
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

        # and the rate at which it changes, its derivative, looks like some matrix times itself.
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

        # This is a differential equation in its purest form; the rate at which some state
        # changes is dependent on where that state is.
        deriv_rect = SurroundingRectangle(equation[:2])
        deriv_rect.set_stroke(RED, 2)
        rhs_rect = SurroundingRectangle(equation[-1])
        rhs_rect.set_stroke(YELLOW, 2)

        self.play(ShowCreation(deriv_rect))
        self.wait()
        self.play(ReplacementTransform(deriv_rect, rhs_rect, path_arc=-45 * DEGREES))

        # One thing we’ll review later is how any differential equation can be thought of as a vector
        # field, with solutions represented as a kind of flow through this field.
        vector_field = VectorField(
            func, plane,
            magnitude_range=(0, 1.2),
            opacity=0.8,
            vector_config={"thickness": 0.02}
        )
        vector_field.sort(lambda p: get_norm(p - plane.get_origin()))

        self.add(vector_field, deriv_vector, vector)
        VGroup(vector, deriv_vector).set_stroke(BLACK, 5, background=True)
        self.play(
            FadeOut(rhs_rect),
            LaggedStartMap(GrowArrow, vector_field, lag_ratio=0)
        )
        self.wait(4)

        flow_lines = AnimatedStreamLines(StreamLines(func, plane, step_multiple=0.25))
        self.add(flow_lines)
        self.wait(4)
        self.play(VFadeOut(flow_lines))

        self.wait(10)

        # Show abstract form
        equation.add(deriv_underline)

        # Show solution
        equation.generate_target()
        equation.target.to_edge(LEFT)
        implies = Tex("\\Rightarrow")
        implies.next_to(equation.target, RIGHT)
        solution = VGroup(
            Matrix([["x(t)"], ["y(t)"]]),
            Tex("="),
            get_matrix_exponential(
                matrix.astype(int),
                v_buff=0.5,
                h_buff=0.5,
            ),
            Matrix([["x(0)"], ["y(0)"]])
        )
        solution.arrange(RIGHT, buff=SMALL_BUFF)
        solution.next_to(implies, RIGHT)

        self.play(
            MoveToTarget(equation),
            Write(implies),
        )
        self.play(
            TransformFromCopy(equation[1], solution[0].copy(), remover=True),
            TransformFromCopy(equation[4], solution[0]),
            TransformFromCopy(equation[3], solution[2][1]),
            FadeIn(solution[1]),
            FadeIn(solution[2][0]),
            FadeIn(solution[2][2]),
        )

        # But the simplest example, and a good place to warm up, is the one-dimensional case,
        # when you just have a single value changing, and its rate of change is proportional
        # to its own size.
        number_line = NumberLine((0, 40), width=40)
        number_line.add_numbers()
        number_line.move_to(plane)
        number_line.to_edge(LEFT)
        nl = number_line

        new_equation = Tex(
            "{dx \\over dt}(t)", "=", "r \\cdot", "x(t)",
        )
        new_equation[0][-2].set_color(GREY_B)
        new_equation[3][-2].set_color(GREY_B)
        new_equation[2][0].set_color(BLUE)
        new_equation.match_height(equation)
        new_equation.move_to(equation)

        self.remove(point)
        vector.clear_updaters()
        deriv_vector.clear_updaters()

        self.remove(vector_field)
        plane.add(vector_field)
        self.add(number_line, deriv_vector, vector)
        self.play(
            # Plane to number line
            vector.animate.put_start_and_end_on(nl.n2p(0), nl.n2p(1)),
            deriv_vector.animate.put_start_and_end_on(nl.n2p(1), nl.n2p(1.5)),
            plane.animate.shift(nl.n2p(0) - plane.get_origin()).set_opacity(0),
            FadeIn(number_line, rate_func=squish_rate_func(smooth, 0.5, 1)),
            # Equation
            TransformMatchingShapes(
                VGroup(equation[0], equation[1].get_entries()[0]),
                new_equation[0],
            ),
            FadeTransform(equation[2], new_equation[1]),
            FadeTransform(equation[3], new_equation[2]),
            FadeTransform(equation[4].get_entries()[0], new_equation[3]),
            FadeOut(equation[1].get_brackets()),
            FadeOut(equation[1].get_entries()[1]),
            FadeOut(equation[4].get_brackets()),
            FadeOut(equation[4].get_entries()[1]),
            FadeOut(deriv_underline),
            FadeOut(rhs_rect),
            run_time=2,
        )

        vt = ValueTracker(1)
        vt.add_updater(lambda m, dt: m.increment_value(0.2 * dt * m.get_value()))

        vector.add_updater(lambda m: m.put_start_and_end_on(nl.n2p(0), nl.n2p(vt.get_value())))
        deriv_vector.add_updater(lambda m: m.set_width(0.5 * vector.get_width()).move_to(vector.get_right(), LEFT))

        self.add(vt)
        self.wait(10)
        self.play(number_line.animate.scale(0.4, about_edge=LEFT))
        self.wait(5)


class SchroedingersEquationIntro(Scene):
    def construct(self):
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
            randy.animate.change("pondering", state_label),
            psis.animate.match_color(state_label),
            FadeIn(state_label, 0.25 * DOWN),
            *map(GrowArrow, state_arrows),
        )
        self.wait()
        self.play(
            FlashUnder(title, time_width=1.5, run_time=2),
            randy.animate.look_at(title),
        )
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
            randy.animate.change("hesitant", equation)
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
        self.play(randy.animate.change("confused", equation))
        self.play(Blink(randy))
        self.wait()
        self.play(FadeOut(randy))
        self.wait()


class SchroedingersComplicatingFactors(TeacherStudentsScene):
    def construct(self):
        pass



# Older

class DefinitionFirstVsLast(Scene):
    def construct(self):
        textbook_title = Text("How textbooks often present math")
        research_title = Text("How math is discovered")


        textbook_progression = VGroup(
            Text("Definition"),
            Text("Theorem"),
            Text("Proof"),
            Text("Examples"),
        )
        research_progression = VGroup(
            Text("Specific problem"),
            Text("General problems"),
            Text("General tactics"),
            Text("Definitions"),
        )
        for progression in [textbook_progression, research_progression]:
            pass

