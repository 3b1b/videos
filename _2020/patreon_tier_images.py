from manim_imports_ext import *


class CircleDivisionImage(Scene):
    CONFIG = {
        "random_seed": 0,
        "n": 0,
    }

    def construct(self):
        # tex = Tex("e^{\\tau i}")
        # tex = Tex("\\sin(2\\theta) \\over \\sin(\\theta)\\cos(\\theta)")
        # tex = Tex("")

        # tex.set_height(FRAME_HEIGHT - 2)
        # if tex.get_width() > (FRAME_WIDTH - 2):
        #     tex.set_width(FRAME_WIDTH - 2)
        # self.add(tex)

        n = self.n
        # angles = list(np.arange(0, TAU, TAU / 9))
        # for i in range(len(angles)):
        #     angles[i] += 1 * np.random.random()

        # random.shuffle(angles)
        # angles = angles[:n + 1]
        # angles.sort()

        # arcs = VGroup(*[
        #     Arc(
        #         start_angle=a1,
        #         angle=(a2 - a1),
        #     )
        #     for a1, a2 in zip(angles, angles[1:])
        # ])
        # arcs.set_height(FRAME_HEIGHT - 1)
        # arcs.set_stroke(YELLOW, 3)

        circle = Circle()
        circle.set_stroke(YELLOW, 5)
        circle.set_height(FRAME_HEIGHT - 1)

        alphas = np.arange(0, 1, 1 / 10)
        alphas += 0.025 * np.random.random(10)
        # random.shuffle(alphas)
        alphas = alphas[:n + 1]

        points = [circle.point_from_proportion(3 * alpha % 1) for alpha in alphas]

        dots = VGroup(*[Dot(point) for point in points])
        for dot in dots:
            dot.scale(1.5)
            dot.set_stroke(BLACK, 2, background=True)
        dots.set_color(BLUE)
        lines = VGroup(*[
            Line(p1, p2)
            for p1, p2 in it.combinations(points, 2)
        ])
        lines.set_stroke(WHITE, 3)

        self.add(circle, lines, dots)


class PatronImage1(CircleDivisionImage):
    CONFIG = {"n": 0}


class PatronImage2(CircleDivisionImage):
    CONFIG = {"n": 1}


class PatronImage4(CircleDivisionImage):
    CONFIG = {"n": 2}


class PatronImage8(CircleDivisionImage):
    CONFIG = {"n": 3}


class PatronImage16(CircleDivisionImage):
    CONFIG = {"n": 4}


class PatronImage31(CircleDivisionImage):
    CONFIG = {"n": 5}


class PatronImage57(CircleDivisionImage):
    CONFIG = {"n": 6}


class PatronImage99(CircleDivisionImage):
    CONFIG = {"n": 7}


class PatronImage163(CircleDivisionImage):
    CONFIG = {"n": 8}


class PatronImage256(CircleDivisionImage):
    CONFIG = {"n": 9}
