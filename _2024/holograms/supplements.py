from manim_imports_ext import *


class DoubleSlitSupplementaryGraphs(InteractiveScene):
    def construct(self):
        # Setup all three axes, with labels

        # Show constructive interference

        # Show destructive interference
        ...



class DistApproximations(InteractiveScene):
    def construct(self):
        # Show sqrt(L^2 + x^2) approx L + x/(2L) approx L
        pass


class DiffractionEquation(InteractiveScene):
    def construct(self):
        # Add equation
        equation = Tex(R"{d} \cdot \sin(\theta) = \lambda", font_size=60)
        equation.set_backstroke(BLACK)
        arrow = Vector(DOWN, thickness=4)
        arrow.set_color(BLUE)

        globals().update(locals())
        d, theta, lam = syms = [equation[s][0] for s in [R"{d}", R"\theta", R"\lambda"]]
        colors = [BLUE, YELLOW, TEAL]

        arrow.next_to(d, UP, LARGE_BUFF)
        arrow.set_fill(opacity=0)

        self.add(equation)

        for sym, color in zip(syms, colors):
            self.play(
                FlashAround(sym, color=color, time_span=(0.25, 1.25)),
                sym.animate.set_fill(color),
                arrow.animate.next_to(sym, UP).set_fill(color, 1),
            )
            self.wait()
        self.play(FadeOut(arrow))
