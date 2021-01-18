from manim_imports_ext import *


class DoingMathVsHowMathIsPresented(Scene):
    def construct(self):
        titles = VGroup(
            TexText("How math is presented"),
            TexText("Actually doing math"),
        )
        for title, vect in zip(titles, [LEFT, RIGHT]):
            title.scale(1.2)
            title.to_edge(UP)
            title.shift(vect * FRAME_WIDTH / 4)

        blocks = VGroup(
            self.get_block(0.98),
            self.get_block(0.1),
        )

        self.add(titles)
        v_line = DashedLine(FRAME_WIDTH * UP / 2, FRAME_WIDTH * DOWN / 2)
        self.add(v_line)
        for block, title in zip(blocks, titles):
            block.next_to(title, DOWN)
            self.play(LaggedStartMap(FadeInFromLarge, block))
            self.wait(2)

    def get_block(self, prob):
        block = VGroup(*[
            self.get_mark(prob)
            for x in range(100)
        ])
        block.arrange_in_grid()
        block.set_width(5)
        return block

    def get_mark(self, prob):
        if random.random() < prob:
            mark = Tex("\\checkmark").set_color(GREEN)
        else:
            mark = Tex("\\times").set_color(RED)
        mark.set_height(1)
        mark.set_width(1, stretch=True)
        return mark


class PiCharts(Scene):
    def construct(self):
        # Add top lines
        equation = Tex(
            "\\frac{1}{10}", "+", "\\frac{2}{5}", "=", "\\; ?"
        )
        equation.scale(2)
        equation.to_edge(UP)
        vs = TexText("vs.")
        vs.scale(3)
        vs.next_to(equation, DOWN, LARGE_BUFF)
        self.add(equation, vs)

        # Add pi charts
        pi_equation = VGroup(
            self.get_pi_chart(10),
            Tex("+").scale(2),
            self.get_pi_chart(5),
            Tex("=").scale(2),
            Tex("?").scale(2),
        )
        pi_equation[0][0].set_fill(RED)
        pi_equation[2][:2].set_fill(RED)
        pi_equation.arrange(RIGHT)
        pi_equation.next_to(vs, DOWN)
        self.add(pi_equation)

        vs.shift(UL * MED_SMALL_BUFF)

        # Swap
        pi_equation.to_edge(UP)
        arrow = Tex("\\downarrow").scale(3)
        arrow.next_to(pi_equation[1], DOWN, LARGE_BUFF)
        equation.next_to(arrow, DOWN)
        equation.shift(0.7 * RIGHT)

        self.remove(vs)
        self.add(arrow)

    def get_pi_chart(self, n):
        result = VGroup(*[
            Sector(
                start_angle=TAU / 4 - k * TAU / n,
                angle=-TAU / n,
                stroke_color=WHITE,
                stroke_width=2,
                fill_color=BLUE_E,
                fill_opacity=1,
            )
            for k in range(n)
        ])
        result.scale(1.5)
        return result


class AskAboutCircleProportion(Scene):
    def construct(self):
        R = 2.5
        circle = Circle(radius=R)
        circles = VGroup(circle, circle.copy())
        circles[0].move_to(R * LEFT / 2)
        circles[1].move_to(R * RIGHT / 2)
        circles[0].set_stroke(WHITE, 2)
        circles[1].set_stroke(BLUE, 4)

        dots = VGroup()
        for circle in circles:
            dots.add(Dot(circle.get_center()))

        arc = Arc(
            radius=R,
            start_angle=TAU / 3,
            angle=TAU / 3,
        )
        arc.set_stroke(YELLOW, 4)
        arc.move_arc_center_to(circles[1].get_center())

        question = TexText("What proportion of the circle?")
        question.set_height(0.6)
        question.to_corner(UL)
        arrow = Arrow(
            question.get_bottom() + LEFT,
            arc.point_from_proportion(0.25),
        )

        self.add(circles)
        self.add(dots)
        self.add(arc)
        self.add(question)
        self.add(arrow)

        answer = Tex("1/3")
        answer.set_height(0.9)
        answer.set_color(YELLOW)
        answer.next_to(question, RIGHT, LARGE_BUFF)
        self.add(answer)


class BorweinIntegrals(Scene):
    def construct(self):
        ints = VGroup(
            Tex(
                "\\int_0^\\infty",
                "\\frac{\\sin(x)}{x}",
                "dx = \\frac{\\pi}{2}"
            ),
            Tex(
                "\\int_0^\\infty",
                "\\frac{\\sin(x)}{x}",
                "\\frac{\\sin(x/3)}{x/3}",
                "dx = \\frac{\\pi}{2}"
            ),
            Tex(
                "\\int_0^\\infty",
                "\\frac{\\sin(x)}{x}",
                "\\frac{\\sin(x/3)}{x/3}",
                "\\frac{\\sin(x/5)}{x/5}",
                "dx = \\frac{\\pi}{2}"
            ),
            Tex("\\vdots"),
            Tex(
                "\\int_0^\\infty",
                "\\frac{\\sin(x)}{x}",
                "\\frac{\\sin(x/3)}{x/3}",
                "\\frac{\\sin(x/5)}{x/5}",
                "\\dots",
                "\\frac{\\sin(x/13)}{x/13}",
                "dx = \\frac{\\pi}{2}"
            ),
            Tex(
                "\\int_0^\\infty",
                "\\frac{\\sin(x)}{x}",
                "\\frac{\\sin(x/3)}{x/3}",
                "\\frac{\\sin(x/5)}{x/5}",
                "\\dots",
                "\\frac{\\sin(x/13)}{x/13}",
                "\\frac{\\sin(x/15)}{x/15}",
                "dx = \\frac{\\pi}{2}",
                "- 0.0000000000231006..."
            ),
        )

        ints.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=RIGHT)
        ints.set_height(FRAME_HEIGHT - 1)
        ints[-1][:-1].align_to(ints[:-1], RIGHT)
        ints[-1][-1].next_to(ints[-1][:-1], RIGHT, SMALL_BUFF)
        ints[3].shift(SMALL_BUFF * LEFT)
        ints.center()

        for integral in ints:
            integral.set_color_by_tex("\\sin(x)", BLUE)
            integral.set_color_by_tex("x/3", TEAL)
            integral.set_color_by_tex("x/5", GREEN)
            integral.set_color_by_tex("x/13", YELLOW)
            integral.set_color_by_tex("x/15", RED_B)

        self.add(ints)
