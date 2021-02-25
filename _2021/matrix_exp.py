from manim_imports_ext import *


def get_integer_matrix_exponential(matrix):
    pass


def get_vector_field_and_stream_lines(func, coordinate_system,
                                      magnitude_range=(0.5, 4),
                                      vector_opacity=0.75,
                                      vector_thickness=0.03,
                                      color_by_magnitude=True,
                                      line_color=GREY_A,
                                      line_width=3,
                                      sample_freq=3,
                                      n_samples_per_line=8,
                                      arc_len=3,
                                      time_width=0.5,
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
    )
    animated_lines = AnimatedStreamLines(
        stream_lines,
        line_anim_config={
            "time_width": time_width,
        },
    )

    return vector_field, animated_lines


# Scenes

class ShowConfusionAtMatrixExponenent(Scene):
    def construct(self):
        # Sticking a matrix in an exponent like this might strike you as...
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
        self.play(Blink(randy))
        self.play(
            matrix.animate.restore(),
            Write(base),
            randy.animate.change("erm", base),
        )
        self.play(Blink(randy))

        # ...odd, to say the least.
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
            Write(bubble.content),
        )
        self.wait()

        false_equation = VGroup(
            matrix_exp, rhs, brace, brace_label
        )

        # The short version of this video would be to say the notation is misleading,
        # having little to do with the number e being multiplying it by itself.
        morty = Mortimer()
        morty.match_height(randy)
        morty.to_corner(DR)
        false_equation.generate_target()
        false_equation.target.scale(0.5)
        false_equation.target.next_to(morty, UL)
        fe_rect = SurroundingRectangle(false_equation.target)
        fe_rect.set_color(GREY_BROWN)
        cross = Cross(false_equation.target[1])
        cross.set_stroke(RED, 5)
        nonsense = Text("This would be nonsense")
        nonsense.match_width(fe_rect)
        nonsense.next_to(fe_rect, UP)
        nonsense.set_color(RED)

        randy.bubble = bubble
        self.play(
            MoveToTarget(false_equation),
            RemovePiCreatureBubble(randy, target_mode="hesitant"),
            morty.animate.change("raise_right_hand"),
            ShowCreation(fe_rect),
            VFadeIn(morty),
        )
        self.play(
            ShowCreation(cross),
            FadeIn(nonsense),
        )
        self.play(Blink(morty))
        self.wait()

        # Writing e with a matrix up top is shorthand for plugging in the matrix to a certain infinite polynomial.
        real_equation = Tex(
            "e^X := X^0 + X^1 + \\frac{1}{2} X^2 + \\frac{1}{6} X^3 + \\cdots + \\frac{1}{n!} X^n + \\cdots",
            isolate=["X"]
        )
        xs = real_equation.get_parts_by_tex("X")
        xs.set_color(TEAL)
        real_equation.set_width(FRAME_WIDTH - 2.0)
        real_equation.to_edge(UP)

        by_definition = Text("(by definition)", font_size=24)
        by_definition.set_fill(GREY_B)
        bd_arrow = Vector(0.5 * UP)
        bd_arrow.match_color(by_definition)
        bd_arrow.next_to(real_equation.get_part_by_tex("="), DOWN, SMALL_BUFF)
        by_definition.next_to(bd_arrow, DOWN)

        self.play(
            TransformFromCopy(base, real_equation[0]),
            FadeTransform(matrix.copy(), real_equation[1]),
        )
        self.play(
            GrowArrow(bd_arrow),
            FadeIn(by_definition, DOWN),
            Write(real_equation[2]),
            FadeTransformPieces(xs[:1].copy(), xs[1:], path_arc=20 * DEGREES),
            LaggedStart(*(
                FadeIn(part)
                for part in real_equation[4:]
                if part not in xs
            ))
        )
        self.add(real_equation)
        self.play(
            randy.animate.change("pondering", real_equation),
            morty.animate.change("pondering", real_equation),
        )
        self.wait()

        # Still, that’s a complicated thing to do, which takes some explaining in its own right,
        # and without context this definition does very little to explain how to think about this operation.
        rhs_tex = "X^0 + X^1 + \\frac{1}{2} X^2 + \\frac{1}{6} X^3 + \\cdots + \\frac{1}{n!} X^n + \\cdots"
        mat_tex = "\\left[ \\begin{array}{ccc} 3 & 1 & 4 \\\\ 1 & 5 & 9 \\\\ 2 & 6 & 5 \\end{array} \\right]"
        ex_rhs = Tex(
            rhs_tex.replace("X", mat_tex),
            tex_to_color_map={mat_tex: TEAL},
        )
        ex_rhs.scale(0.5)

        ex_eq = VGroup(matrix_exp.copy(), Tex("="), ex_rhs)
        ex_eq[0][1].set_color(TEAL)
        ex_eq.arrange(RIGHT)
        ex_rhs.align_to(ex_eq[0], DOWN)
        ex_eq = VGroup(ex_eq[0], ex_eq[1], *ex_rhs)
        ex_eq[1:].shift(0.1 * DOWN)
        ex_eq.next_to(real_equation, DOWN, buff=2, aligned_edge=DOWN)

        false_group = VGroup(false_equation, fe_rect, cross, nonsense)
        complicated = Text("Rather complicated!")
        complicated.set_color(YELLOW)
        complicated.next_to(ex_eq, DOWN, LARGE_BUFF)

        self.play(
            TransformFromCopy(matrix_exp, ex_eq[0]),
            FadeOut(false_group, DOWN),
            FadeTransformPieces(
                real_equation.copy()[2:], ex_eq[1:], run_time=2,
            ),
            randy.animate.change("hesitant"),
            morty.animate.change("raise_right_hand"),
        )
        self.play(Blink(randy))
        self.play(
            FadeIn(complicated, shift=DOWN),
            randy.animate.change("confused"),
            morty.animate.change("hesitant"),
        )
        self.wait()


class CircularPhaseFlow(Scene):
    def construct(self):
        plane = NumberPlane(
            x_range=[-4, 4],
            y_range=[-2, 2],
            height=8,
            width=16,
        )
        plane.add_coordinate_labels()

        vector_field, animated_lines = get_vector_field_and_stream_lines(
            self.func, plane
        )

        self.add(plane)
        self.add(vector_field)
        self.add(animated_lines)

        self.wait(10)

        #
        self.embed()

    def func(self, x, y):
        return (-y, x)

    def get_label(self):
        pass


class HyperbolicPhaseFlow(CircularPhaseFlow):
    def func(self, x, y):
        return (x, -y)


class WhyWedCare(Scene):
    def construct(self):
        pass
        # ...how to think about this operation, or more importantly why we’d care.

        # So if it’s alright with you I think we should hold off explaining this definition, or justifying the
        # abuse of notation, until it’s a bit clearer what problems it helps us to solve.


        # 
        self.embed()
