from manim_imports_ext import *
from _2023.convolutions2.continuous import *


class WhatDistributionDescribesThis(InteractiveScene):
    def construct(self):
        words = Text("What distribution\ndescribes this?")
        arrow = Arrow(words.get_bottom(), words.get_bottom() + DL)
        VGroup(words, arrow).set_color(TEAL)
        self.play(Write(words), ShowCreation(arrow))
        self.wait()


class GuessTheAnswer(TeacherStudentsScene):
    def construct(self):
        pass


class AlreadyCoveredConvolutions(TeacherStudentsScene):
    def construct(self):
        pass


class CountOutcomes(InteractiveScene):
    def construct(self):
        equation = Tex(R"6 \times 6 = 36 \text{ outcomes}")
        self.play(FadeIn(equation, lag_ratio=0.1))
        self.wait()


class AskAboutAddingThreeUniforms(InteractiveScene):
    def construct(self):
        # Set up equations
        h_line = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        h_line.set_stroke(GREY_A, 1)
        two_sum = Tex("X_1 + X_2", font_size=96)
        three_sum = Tex("X_1 + X_2 + X_3", font_size=96)
        two_sum.to_edge(UP)
        three_sum.next_to(h_line, DOWN, MED_SMALL_BUFF)
        VGroup(two_sum, three_sum).shift(LEFT)

        # Braces
        exprs = [two_sum, three_sum]
        term_braces = VGroup(*(
            Brace(xi, DOWN, SMALL_BUFF)
            for expr in exprs
            for xi in expr[re.compile("X_.")]
        ))
        expr_braces = VGroup(*(
            Brace(expr, DOWN)
            for expr in [*exprs]
        ))
        x12_brace = Brace(three_sum["X_1 + X_2"], DOWN)

        # Plots
        # (Christ this is confusingly written)
        all_braces = [*term_braces, *expr_braces, x12_brace]
        funcs = [*[uniform] * 5, *[wedge_func] * 3]
        colors = [*[BLUE] * 5, TEAL, YELLOW, TEAL]
        for brace, func, color in zip(all_braces, funcs, colors):
            x_range = (-1, 1) if brace in term_braces else (-2, 2)
            axes = Axes(
                x_range,
                (0, 1.5, 0.5),
                width=2.0,
                height=1.0,
                axis_config=dict(tick_size=0.025)
            )
            axes.next_to(brace, np.round(brace.get_direction(), 1))
            if brace is expr_braces[1]:
                graph = get_conv_graph(axes, uniform, wedge_func)
            else:
                graph = axes.get_graph(
                    func,
                    x_range=(*x_range, 0.025),
                    use_smoothing=False
                )
            graph.set_stroke(color, 3)

            brace.plot = VGroup(axes, graph)

        term_plots = VGroup(*(brace.plot for brace in term_braces))
        expr_plots = VGroup(*(brace.plot for brace in expr_braces))
        x12_plot = x12_brace.plot

        # Convolution equations
        symbols = VGroup()
        x_shift = (term_plots[1].get_center() - term_plots[0].get_center()) / 2
        for ch, plot in zip("*=**=", term_plots):
            symbol = Tex(ch, font_size=72)
            symbol.move_to(plot)
            symbol.shift(x_shift * RIGHT)
            if ch == "=":
                symbol.shift(0.25 * x_shift * RIGHT)
            symbols.add(symbol)

        expr_plots[0].move_to(term_plots[1], DOWN).shift(2.5 * x_shift * RIGHT)
        expr_plots[1].move_to(term_plots[4], DOWN).shift(2.5 * x_shift * RIGHT)

        q_marks = Tex("???", font_size=96)
        q_marks.move_to(expr_plots[1])

        # Show the first two
        self.add(two_sum)

        self.play(
            LaggedStartMap(GrowFromCenter, term_braces[:2], lag_ratio=0),
            LaggedStartMap(FadeIn, term_plots[:2], shift=DOWN),
            FadeIn(symbols[:2], lag_ratio=0.2),
            run_time=1,
        )
        self.play(
            TransformFromCopy(term_plots[0], expr_plots[0]),
            TransformFromCopy(term_plots[1], expr_plots[0]),
        )
        self.wait()

        # Transition from two_sum to three_sum
        self.play(LaggedStart(
            ShowCreation(h_line),
            TransformFromCopy(two_sum, three_sum["X_1 + X_2"][0]),
            TransformFromCopy(term_braces[:2], term_braces[2:4]),
            TransformFromCopy(term_plots[:2], term_plots[2:4]),
            TransformFromCopy(two_sum["+ X_2"][0], three_sum["+ X_3"][0]),
            TransformFromCopy(term_braces[1], term_braces[4]),
            TransformFromCopy(term_plots[1], term_plots[4]),
            TransformFromCopy(symbols[0].replicate(2), symbols[2:4]),
            TransformFromCopy(symbols[1], symbols[4]),
            lag_ratio=0.02,
        ))
        self.wait()
        self.play(Write(q_marks))
        self.wait()

        # Threat first two of three as a wedge
        self.play(
            TransformFromCopy(expr_plots[0], x12_plot),
            ReplacementTransform(term_braces[2], x12_brace),
            ReplacementTransform(term_braces[3], x12_brace),
            FadeOut(term_plots[2:4], DOWN),
            FadeOut(symbols[2], DOWN),
            symbols[3].animate.shift(0.5 * LEFT)
        )
        self.wait()
        self.play(
            TransformFromCopy(x12_plot, expr_plots[1]),
            TransformFromCopy(term_plots[4], expr_plots[1]),
            FadeOut(q_marks, RIGHT)
        )
        self.wait()
