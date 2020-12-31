from big_ol_pile_of_manim_imports import *


class DoingMathVsHowMathIsPresented(Scene):
    def construct(self):
        titles = VGroup(
            TextMobject("How math is presented"),
            TextMobject("Actually doing math"),
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
            mark = TexMobject("\\checkmark").set_color(GREEN)
        else:
            mark = TexMobject("\\times").set_color(RED)
        mark.set_height(1)
        mark.set_width(1, stretch=True)
        return mark


class PiCharts(Scene):
    def construct(self):
        # Add top lines
        equation = TexMobject(
            "\\frac{1}{10}", "+", "\\frac{2}{5}", "=", "\\; ?"
        )
        equation.scale(2)
        equation.to_edge(UP)
        vs = TextMobject("vs.")
        vs.scale(3)
        vs.next_to(equation, DOWN, LARGE_BUFF)
        self.add(equation, vs)

        # Add pi charts
        pi_equation = VGroup(
            self.get_pi_chart(10),
            TexMobject("+").scale(2),
            self.get_pi_chart(5),
            TexMobject("=").scale(2),
            TexMobject("?").scale(2),
        )
        pi_equation[0][0].set_fill(RED)
        pi_equation[2][:2].set_fill(RED)
        pi_equation.arrange(RIGHT)
        pi_equation.next_to(vs, DOWN)
        self.add(pi_equation)

        vs.shift(UL * MED_SMALL_BUFF)

        # Swap
        pi_equation.to_edge(UP)
        arrow = TexMobject("\\downarrow").scale(3)
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
