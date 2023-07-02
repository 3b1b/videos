from manim_imports_ext import *
from _2023.convolutions2.continuous import *


class QuizTime(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(self.teacher.says("Quiz time!", mode="hooray"))
        self.play(
            self.change_students("happy", "tease", "hooray")
        )
        self.wait(2)


class IfYouReallyUnderstand(TeacherStudentsScene):
    def construct(self):
        # Initial comments
        morty = self.teacher
        self.play(
            morty.says(TexText(R"If you \emph{really} \\ understand"), run_time=1.5),
            self.change_students("pondering", "confused", "erm", look_at=self.screen)
        )
        self.wait()

        old_bubble = morty.bubble
        old_words = morty.bubble.content
        new_bubble = morty.get_bubble(
            TexText(R"You'll understand \\ why normal distributions \\ are special"),
            bubble_type=SpeechBubble,
            width=5, height=4,
        )
        new_bubble.content.scale(1.2)
        VGroup(new_bubble, new_bubble.content).shift(0.5 * RIGHT)
        self.play(
            Transform(old_bubble, new_bubble),
            Transform(old_words, new_bubble.content, run_time=1),
            morty.change("tease"),
        )
        self.remove(old_bubble, old_words)
        self.add(new_bubble, new_bubble.content)
        morty.bubble = new_bubble

        # Stare at it
        self.wait()
        self.play(self.change_students(None, "tease", "maybe", look_at=self.screen))
        self.wait(3)
        self.play(self.change_students("happy", "erm", "hesitant", look_at=self.screen, lag_ratio=0.2))
        self.wait(3)

        # The real lesson
        title = TexText("Today's lesson:", font_size=60)
        subtitle = Text("How to add random\nvariables, in general", font_size=48)
        title.set_x(FRAME_WIDTH / 4).to_edge(UP, buff=MED_SMALL_BUFF)
        subtitle.next_to(title, DOWN, buff=LARGE_BUFF)

        self.play(
            morty.debubble("raise_right_hand"),
            FadeIn(title, UP),
            self.change_students("pondering", "pondering", "hesitant", look_at=title),
        )
        self.play(FadeIn(subtitle, lag_ratio=0.1))
        self.wait(2)
        self.play(self.change_students("maybe", "hesitant", "well", look_at=self.screen))
        self.wait(4)


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
        # Ask
        morty = self.teacher
        stds = self.students
        self.screen.to_edge(UR)
        self.play(
            stds[2].says(
                "Wait, didn't you already\ncover convolutions?",
                mode="dance_1"
            ),
            morty.change("guilty"),
        )
        self.play(
            self.change_students("hesitant", "awe", None, look_at=self.screen)
        )
        self.wait(2)

        # Show video
        words = VGroup(
            Text("Only discrete examples", font_size=48),
            Text(
                """
                Probability
                Moving averages
                Image processing
                Polynomial multiplication
                Relationship with FFTs
                """,
                alignment="LEFT",
                font_size=36
            ),
        )
        words.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        words[1].shift(0.25 * RIGHT)
        words.to_corner(UL)
        words.to_edge(UP)

        self.play(LaggedStart(
            morty.change("raise_right_hand", look_at=self.screen),
            stds[2].debubble(mode="sassy", look_at=self.screen),
            stds[1].change("pondering", look_at=self.screen),
        ))
        self.wait(2)
        self.play(
            Write(words[0], run_time=1),
            self.change_students("pondering", "tease", "plain", look_at=words)
        )
        self.wait(3)
        self.play(FadeIn(words[1], run_time=1, lag_ratio=0.1))
        self.wait(3)

        # Awkward spot
        no_prereq = Text("Not a \n prerequisite", font_size=60)
        no_prereq.move_to(words).to_edge(LEFT, buff=1.0)
        no_prereq.set_color(RED)
        redundancy_words = Text("But some overlap\nwith this video")
        redundancy_words.move_to(no_prereq, LEFT)
        redundancy_words.match_color(no_prereq)
        arrow = Vector(2.5 * RIGHT)
        arrow.next_to(no_prereq, RIGHT)
        arrow.match_color(no_prereq)

        self.play(
            morty.change("guilty"),
            self.change_students("hesitant", "pondering", "hesitant", look_at=morty.eyes)
        )
        self.wait()
        self.look_at(no_prereq, added_anims=[
            FadeOut(words, LEFT),
            FadeIn(no_prereq, LEFT),
        ])
        self.play(
            GrowArrow(arrow),
            morty.change("raise_left_hand", self.screen)
        )
        self.wait(2)
        self.play(morty.change("maybe"))
        self.look_at(no_prereq, added_anims=[
            FadeOut(no_prereq, UP),
            FadeIn(redundancy_words, UP),
            arrow.animate.stretch(0.8, 0, about_edge=RIGHT),
        ])
        self.wait(3)
        self.play(
            morty.change("tease"),
            self.change_students("happy", "thinking", "happy", look_at=morty.eyes),
        )
        self.wait(4)


class PauseAndPonder(TeacherStudentsScene):
    def construct(self):
        self.play(self.teacher.says("Pause and\nponder!", mode="hooray", run_time=1))
        self.play(self.change_students("thinking", "pondering", "pondering", look_at=self.screen))
        self.wait(5)

        # State goal
        self.play(
            self.teacher.debubble(mode="raise_right_hand", look_at=3 * UP),
            self.change_students("tease", "thinking", "awe", look_at=3 * UP)
        )
        self.wait(8)


class CountOutcomes(InteractiveScene):
    def construct(self):
        equation = Tex(R"6 \times 6 = 36 \text{ outcomes}")
        self.play(FadeIn(equation, lag_ratio=0.1))
        self.wait()


class AssumingIndependence(TeacherStudentsScene):
    def construct(self):
        # Assuming...
        morty = self.teacher
        stds = self.students
        self.remove(self.background)

        self.play(
            morty.says("Assuming\nindependence!", mode="surprised", run_time=1),
            self.change_students("hesitant", "guilty", "plain")
        )
        self.wait(3)
        self.play(
            morty.debubble("tease"),
            stds[1].says(TexText(R"Isn't that a \\ little pedantic?"), mode="sassy"),
            stds[2].change("angry")
        )
        self.wait(3)


class IsntThatOvercomplicated(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.remove(self.background)

        # Show
        self.play(
            stds[0].change("erm", self.screen),
            stds[2].change("confused", self.screen),
            stds[1].says(
                "Isn't that...a little\nover-complicated?",
                mode="angry",
                bubble_config=dict(width=7, height=4),
            ),
            morty.change("shruggie")
        )
        self.wait(2)
        self.play(morty.change("guilty"))
        self.wait()
        self.play(
            morty.says("It's fun!", mode="hooray", bubble_config=dict(width=2, height=1.5)),
            stds[1].debubble()
        )
        self.play(self.change_students("hesitant", "tease", "pondering", look_at=self.screen))
        self.wait(3)

        self.play(
            morty.debubble(mode="raise_right_hand"),
            self.change_students("pondering", "erm", "pondering", look_at=self.screen)
        )
        self.wait(5)


class QuestionTheFormula(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.remove(self.background)

        self.play(morty.change("raise_right_hand", 3 * UR))
        self.play(
            stds[0].change("erm", 3 * UR),
            stds[1].says("Hang on...", mode="hesitant"),
            stds[2].change("confused", 3 * UR),
        )
        self.wait(3)


class RuleOfThumb(InteractiveScene):
    def construct(self):
        # Test
        title = Text("General rule of thumb", font_size=72)
        title.to_edge(UP)
        sigma, arrow, integral = group = VGroup(
            Tex(R"\sum \dots").set_height(1.5),
            Vector(2 * RIGHT, stroke_width=10),
            Tex(R"\int\dots dx").set_height(2.5),
        )
        group.arrange(RIGHT, buff=LARGE_BUFF)
        integral.shift(0.5 * LEFT)
        group.center()

        sigma.set_fill(RED)
        integral.set_fill(TEAL)
        VGroup(sigma, integral).set_stroke(WHITE, 1)
        sigma.save_state()
        sigma.center()

        self.add(title)
        self.play(Write(sigma, run_time=1))
        self.wait()
        self.play(
            TransformFromCopy(sigma, integral),
            sigma.animate.restore(),
            GrowFromCenter(arrow)
        )
        self.wait()


class CanWeSeeAnExample(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.remove(self.background)
        self.play(
            stds[1].says("Can we see\nan example?", mode="raise_left_hand"),
        )
        self.play(
            stds[0].change("confused", self.screen),
            stds[2].change("maybe", self.screen),
            morty.change("happy"),
        )
        self.wait(2)
        self.play(
            morty.change("raise_right_hand", 3 * UP),
            self.change_students("pondering", None, "pondering", look_at=3 * UP),
            stds[1].debubble()
        )
        self.look_at(3 * UP)
        self.wait(5)


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


class ConfusedAtThree(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.play(
            morty.change("raise_right_hand"),
            self.change_students("maybe", "confused", "sassy", look_at=self.screen)
        )
        self.wait(2)
        self.look_at(morty.get_corner(UL))
        self.play(self.change_students("hesitant", "maybe", "pondering", look_at=self.screen))
        self.wait(5)


class LikeAMovingAverage(InteractiveScene):
    def construct(self):
        words = Text("Kind of like\na moving average")
        top_words = words["Kind of like"]
        cross = Cross(top_words)

        self.play(FadeIn(words, lag_ratio=0.1))
        self.wait()
        self.play(ShowCreation(cross))
        self.wait()


class KeepGoing(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.play(
            stds[0].says("More\niterations!", mode="hooray", bubble_config=dict(width=4, height=2)),
            stds[1].change("tease", stds[0].eyes),
            stds[2].change("coin_flip_1", stds[0].eyes),
            morty.change("happy"),
            run_time=1
        )
        self.wait(5)


class SumOfThree(InteractiveScene):
    def construct(self):
        self.add(Tex("X_1 + X_2 + X_3"))


class SumOfFour(InteractiveScene):
    def construct(self):
        self.add(Tex("X_1 + X_2 + X_3 + X_4"))


class WhyNormals(InteractiveScene):
    def construct(self):
        # Add graph
        plane = NumberPlane(
            (-4, 4), (0, 0.75, 0.125),
            width=7, height=5,
            axis_config=dict(stroke_width=1),
            background_line_style=dict(stroke_color=GREY_B, stroke_width=1, stroke_opacity=0.5)
        )
        plane.to_edge(RIGHT)
        graph = plane.get_graph(lambda x: np.exp(-x**2 / 2) / math.sqrt(TAU))
        graph.set_stroke(BLUE, 3)

        formula = Tex(R"\frac{1}{\sqrt{2\pi}} e^{-x^2 / 2}")
        formula.next_to(graph.pfp(0.45), UL, buff=-0.25)
        formula.set_backstroke()
        name = Text("Standard normal distribution", font_size=48)
        name.next_to(plane.get_top(), DOWN, buff=0.2)
        name.set_backstroke()

        self.add(plane)
        self.add(graph)
        self.add(formula)
        self.add(name)

        # Ask question
        randy = Randolph(height=2)
        randy.move_to(interpolate(plane.get_left(), LEFT_SIDE, 0.75))
        randy.align_to(plane, DOWN)
        morty = Mortimer(height=2)
        morty.next_to(randy, RIGHT, buff=1.0)

        self.add(randy)
        self.play(LaggedStart(
            randy.thinks(
                Text("Why this\nfunction?", font_size=32),
                mode="pondering", look_at=formula, run_time=1
            ),
            FlashAround(formula, run_time=3, time_width=1.5)
        ))
        self.play(Blink(randy))
        self.wait()
        self.play(
            randy.debubble(mode="tease", look_at=morty.eyes),
            VFadeIn(morty),
            morty.says(
                Text("There's a very\nfun answer", font_size=32),
                look_at=randy.eyes,
                run_time=1
            ),
        )
        self.wait(4)


class AskIfItsTheArea(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.remove(self.background)

        # Test
        self.play(
            stds[0].says(
                TexText("Is $[f*g](s)$ the\narea of this slice?"),
                mode="raise_left_hand",
                look_at=self.screen,
            ),
            stds[1].change("pondering", self.screen),
            stds[2].change("pondering", self.screen),
            morty.change("tease", stds[0].eyes),
        )
        self.wait(3)
        self.play(morty.says("Almost!", mode="coin_flip_2"))
        self.play(
            self.change_students("confused", "erm", "hesitant", look_at=morty.eyes)
        )
        self.wait(5)


class TwoRects(InteractiveScene):
    height = 5.5
    width = 6.11
    buff = 1.0
    bottom_buff = MED_LARGE_BUFF

    def construct(self):
        rects = Rectangle(self.width, self.height).replicate(2)
        rects.set_stroke(WHITE, 2)
        rects.arrange(RIGHT, buff=self.buff)
        rects.to_edge(DOWN, buff=self.bottom_buff)
        self.add(rects)


class ShorterRects(TwoRects):
    height = 4.0
    width = 6.5
    buff = 0.25
    bottom_buff = 1.5


class IndicatingRectangle(InteractiveScene):
    def construct(self):
        self.play(FlashAround(Rectangle(9, 3), run_time=2, time_width=1.5, stroke_width=8))


class Sqrt2Correction(InteractiveScene):
    def construct(self):
        # Test
        words = TexText(R"Technically, this shape is $\sqrt{2}$ times wider than this shape")
        shape_words = words["this shape"]
        words.to_edge(UP, buff=0.5)

        points = [
            [-3.75, 0.25, 0],
            [2.5, -2.5, 0],
        ]
        colors = [YELLOW, TEAL]
        arrows = VGroup()
        for word, point, color in zip(shape_words, points, colors):
            word.set_color(color)
            arrow = FillArrow(word[:4].get_bottom(), point)
            arrow.scale(0.9)
            arrow.set_fill(color, 1)
            arrow.set_backstroke()
            arrows.add(arrow)

        self.play(FadeIn(words, lag_ratio=0.1))
        self.play(LaggedStartMap(GrowArrow, arrows, lag_ratio=0.5))
        self.wait()


class Sqrt2Explanation(InteractiveScene):
    def construct(self):
        # Test
        tex_kw = dict(font_size=30, alignment="")
        explanation = VGroup(
            TexText(R"""
                Think about what a Riemann sum approximating the area of \\
                the slice below would look like, where the slice is taken over \\
                the line $x + y = s$. If you have some samples of $x$, spaced \\
                apart by $\Delta x$, the sum of the areas of the rectangles \\
                below looks like\\

                $$
                \displaystyle \sum_{x \in \text{samples}}
                \overbrace{f(x) g(s - x)}^{\text{Rect. height}}
                \underbrace{\sqrt{2} \cdot \Delta x}_{\text{Rect. width}}
                $$\\
            """, **tex_kw),
            TexText(R"""
                This is \emph{almost} a Riemann approximation of the\\
                integral we care about \\

                $$\displaystyle [f * g](s) = \int_{-\infty}^\infty f(x)g(s - x)dx.$$\\

                The only difference is that factor of $\sqrt{2}$. That factor \\
                persists in the limit as $\Delta x \to 0$, so the area of the \\
                slice below is exactly $\sqrt{2} \cdot [f * g](s)$.\\
            """, **tex_kw),
        )
        explanation[0][196:].scale(1.25, about_edge=UP).match_x(explanation[0][:196])
        explanation[1][57:83].scale(1.25, about_edge=UP).match_x(explanation[1][:57])
        explanation.arrange(RIGHT, buff=LARGE_BUFF, aligned_edge=UP)
        explanation.set_width(FRAME_WIDTH - 1)
        h_line = Line(UP, DOWN).match_height(explanation)
        h_line.move_to(explanation)
        h_line.shift(0.5 * RIGHT)
        h_line.set_stroke(GREY_B, 1)
        explanation.add(h_line)
        explanation.to_edge(UP)

        self.add(explanation)


class SimpleQuestion(InteractiveScene):
    def construct(self):
        # Rectangles
        rects = ScreenRectangle().replicate(2)
        rects.set_height(0.45 * FRAME_HEIGHT)
        rects.arrange(RIGHT, buff=0.75).shift(0.5 * DOWN)

        # Equations
        equations = VGroup(
            Tex("X + Y", font_size=60),
            Tex(R"\int_{-\infty}^\infty f(x)g(s - x)dx"),
        )
        for equation, rect in zip(equations, rects):
            equation.next_to(rect, UP)

        # Words
        words = VGroup(
            Text("Such a \n simple question!"),
            Text(
                "Such a seemingly \n simple question.",
                t2s={"seemingly": ITALIC},
                t2c={"seemingly": YELLOW},
            )
        )
        for word in words:
            word.match_x(rects[0])
            word.to_edge(UP, buff=MED_SMALL_BUFF)

        self.play(
            Write(words[0], run_time=1),
            FadeIn(equations[0])
        )
        self.wait()
        self.play(TransformMatchingStrings(
            words[0], words[1],
            key_map={"!": "."},
            run_time=1,
        ))
        self.wait()
        # self.play(Write(equations[1]))
        self.play(TransformFromCopy(equations[0], equations[1]), run_time=2, lag_ratio=0.02)
        self.wait(2)


class ComplainAboutCalculation(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.remove(self.background)

        # Complain
        self.play(
            stds[2].says("But do these help\nwith actual computations?", mode="sassy"),
            morty.change("guilty")
        )
        self.play(self.change_students("erm", "frustrated", look_at=self.screen))
        self.wait(3)

        # Opening quiz
        self.play(
            stds[1].says("Like, the opening quiz?", mode="raise_right_hand"),
            stds[2].debubble(mode="hesitant", look_at=self.screen),
            stds[0].change("pondering"),
        )
        self.play(morty.change("well"))
        self.look_at(self.screen)
        self.wait(3)

        # React to ordinary approach
        self.play(
            morty.change("raise_left_hand", 4 * UR),
            stds[0].animate.look_at(3 * UR),
            stds[1].debubble(mode="pondering", look_at=3 * UR),
            stds[2].change("pondering", 3 * UR),
            self.frame.animate.set_height(12, about_edge=DOWN)
        )
        self.wait(4)
        self.play(
            morty.change("raise_right_hand", look_at=self.screen),
            self.change_students("hesitant", "pondering", "guilty", look_at=self.screen),
        )
        self.wait(2)
        self.play(
            self.change_students("horrified", "pleading", "concentrating", look_at=2 * UR)
        )
        self.play(morty.change("tease"))
        self.wait(4)
        self.play(
            morty.change("coin_flip_2"),
            self.change_students("guilty", "guilty", "sick", look_at=2 * UR)
        )
        self.wait()
        self.play(
            morty.change("hooray"),
            self.change_students("well", "tease", "happy", look_at=morty.eyes)
        )
        self.wait(4)


class OrdinaryApproach(InteractiveScene):
    def construct(self):
        # Title
        self.add(FullScreenRectangle().scale(2))
        title = Text("Ordinary (messy) approach")
        title.set_x(0.25 * FRAME_WIDTH)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        title.set_backstroke()
        underline = Underline(title, buff=-0.05)
        self.play(
            ShowCreation(underline),
            FadeIn(title, lag_ratio=0.1),
        )
        self.wait()

        # Equations
        tex1 = R"f(x) = \frac{1}{\sigma_1 \sqrt{2\pi}} e^{-x^2 / 2\sigma_1^2}"
        tex2 = R"g(y) = \frac{1}{\sigma_2 \sqrt{2\pi}} e^{-y^2 / 2\sigma_2^2}"
        tex_kw = dict(
            font_size=30,
            t2c={"x": BLUE, "y": YELLOW}
        )
        equations = VGroup(
            Tex(tex1, **tex_kw),
            Tex(tex2, **tex_kw),
        )
        equations.arrange(DOWN, buff=0.75, aligned_edge=LEFT)
        equations.next_to(title, DOWN, buff=0.75, aligned_edge=LEFT)

        brace = Brace(equations[0], RIGHT, tex_string=R"\underbrace{\qquad\qquad\qquad}")
        brace.stretch(1.5, 1)
        brace_text = brace.get_text(
            "Formula\nfor a normal\ndistribution",
            font_size=30,
            alignment="LEFT",
            buff=MED_SMALL_BUFF,
        )

        self.play(LaggedStartMap(FadeIn, equations, lag_ratio=0.5, run_time=1))
        self.play(
            GrowFromCenter(brace),
            FadeIn(brace_text, lag_ratio=0.1)
        )
        self.wait()

        # Convolution
        conv_equation = Tex(
            R"""
                [f * g](s) = \int_{-\infty}^\infty f(x)g(s - x) dx
                = \int_{-\infty}^\infty \frac{1}{\sigma_1 \sigma_2 2 \pi} e^{-x^2 / 2\sigma_1^2} e^{-(s - x)^2 / 2\sigma_2^2} \; dx
            """,
            t2c={"x": BLUE, "s": GREEN},
            font_size=36,
        )
        conv_equation.move_to(1.0 * DOWN)
        rhss = conv_equation[re.compile(r"=.*dx")]
        lhs = conv_equation["[f * g](s)"]

        self.play(FadeIn(lhs), FadeIn(rhss[0]))
        self.wait()
        self.play(
            LaggedStart(
                FadeTransform(equations[0].copy(), rhss[1]),
                FadeTransform(equations[1].copy(), rhss[1]),
                FadeTransform(rhss[0].copy(), rhss[1]),
                lag_ratio=0.05,
            ),
            self.frame.animate.move_to(DOWN),
            run_time=2
        )
        self.wait()

        # Low brace
        low_brace = Brace(rhss[1][1:], DOWN, buff=SMALL_BUFF)
        calc_text = low_brace.get_text(
            "Not prohibitively difficult,\nbut a bit clunky",
            font_size=30
        )
        self.play(
            GrowFromCenter(low_brace),
            FadeIn(calc_text, lag_ratio=0.1)
        )
        self.wait()

        # Toss out formula
        to_toss = VGroup(conv_equation, low_brace, calc_text)
        self.play(
            FadeOut(to_toss, 6 * RIGHT, path_arc=-30 * DEGREES, rate_func=running_start, run_time=1)
        )
        self.wait()


class GaussianFunctionAnnotations(InteractiveScene):
    def construct(self):
        # Test
        tex_kw = dict(font_size=36, t2c={"x": BLUE, "y": YELLOW})
        functions = VGroup(
            Tex(R"f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-x^2 / 2\sigma^2}", **tex_kw),
            Tex(R"g(y) = \frac{1}{\sigma \sqrt{2\pi}} e^{-y^2 / 2\sigma^2}", **tex_kw),
        )
        functions.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        functions.to_corner(UL)

        self.add(functions)


class PredictTheProof(InteractiveScene):
    def construct(self):
        morty = Mortimer(height=2)
        morty.to_corner(DR)
        self.play(morty.says("Try predicting\nthe proof!"))
        for x in range(3):
            self.play(Blink(morty))
            self.wait(4 * random.random())


class PiScene(InteractiveScene):
    def construct(self):
        morty = Mortimer(height=1.5)
        randy = Randolph(height=1.5)
        morty.set_x(3).to_edge(UP)
        randy.set_x(-3).to_edge(UP)

        self.play(
            morty.change("pondering", ORIGIN),
            randy.change("thinking", ORIGIN),
        )
        self.wait()


class ThumbnailMaterial(InteractiveScene):
    def construct(self):
        # Graphs
        axes = Axes((-2, 2), (0, 1, 0.25), width=7, height=3)

        exp_graph = axes.get_graph(lambda x: np.exp(-2 - x))
        exp_graph.set_stroke(BLUE, 5)
        exp_label = Tex("p_X(x)", font_size=72)
        exp_label.next_to(exp_graph.pfp(0.2), UR, SMALL_BUFF)

        wedge_graph = axes.get_graph(wedge_func, use_smoothing=False)
        wedge_graph.set_stroke(YELLOW, 5)
        wedge_label = Tex("p_Y(y)", font_size=72)
        wedge_label.next_to(wedge_graph.pfp(0.4), UL, SMALL_BUFF)

        # conv_label = Tex("[f * g](-1.5)", font_size=72)
        # self.add(conv_label)
        # return

        self.add(axes)
        self.add(exp_graph)
        self.add(exp_label)
        # self.add(wedge_graph)
        # self.add(wedge_label)


class EndScreen(PatreonEndScreen):
    pass
