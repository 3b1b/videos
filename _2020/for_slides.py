from imports_3b1b import *


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

        question = TextMobject("What proportion of the circle?")
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

        answer = TexMobject("1/3")
        answer.set_height(0.9)
        answer.set_color(YELLOW)
        answer.next_to(question, RIGHT, LARGE_BUFF)
        self.add(answer)


class BorweinIntegrals(Scene):
    def construct(self):
        ints = VGroup(
            TexMobject(
                "\\int_0^\\infty",
                "\\frac{\\sin(x)}{x}",
                "dx = \\frac{\\pi}{2}"
            ),
            TexMobject(
                "\\int_0^\\infty",
                "\\frac{\\sin(x)}{x}",
                "\\frac{\\sin(x/3)}{x/3}",
                "dx = \\frac{\\pi}{2}"
            ),
            TexMobject(
                "\\int_0^\\infty",
                "\\frac{\\sin(x)}{x}",
                "\\frac{\\sin(x/3)}{x/3}",
                "\\frac{\\sin(x/5)}{x/5}",
                "dx = \\frac{\\pi}{2}"
            ),
            TexMobject("\\vdots"),
            TexMobject(
                "\\int_0^\\infty",
                "\\frac{\\sin(x)}{x}",
                "\\frac{\\sin(x/3)}{x/3}",
                "\\frac{\\sin(x/5)}{x/5}",
                "\\dots",
                "\\frac{\\sin(x/13)}{x/13}",
                "dx = \\frac{\\pi}{2}"
            ),
            TexMobject(
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
