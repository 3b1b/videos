from manim_imports_ext import *

class Resistor(Line):
    def init_points(self):
        midpoints = [
            interpolate(self.start, self.end, alpha)
            for alpha in [0.25]+list(np.arange(0.3, 0.71, 0.1))+[0.75]
        ]
        perp = rotate_vector(self.end-self.start, np.pi/2)
        for midpoint, n in zip(midpoints[1:-1], it.count()):
            midpoint += 0.1*((-1)**n)*perp
        points = [self.start]+midpoints+[self.end]
        self.set_points_as_corners(points)


class LongResistor(Line):
    def init_points(self):
        mid1 = interpolate(self.start, self.end, 1./5)
        mid2 = interpolate(self.start, self.end, 4./5)
        self.add(Line(self.start, mid1))
        self.add(Resistor(mid1, mid2))
        self.add(Line(mid2, self.end))


class Source(VMobject):
    def init_points(self):
        self.add(Circle(color = self.color))
        self.add(Tex("+").scale(1.5).set_color(GREEN).shift(0.5*UP))
        self.add(Tex("-").scale(1.5).set_color(RED).shift(0.5*DOWN))
        self.set_height(1)        
        self.add(Line(self.get_top(), self.get_top()+UP))
        self.add(Line(self.get_bottom(), self.get_bottom()+DOWN))



class CircuitReduction(Scene):
    def construct(self):
        pos = dict([
            (x, {
                0 : (1.8*x-6)*RIGHT+2*UP,
                0.5 : (1.8*x-6)*RIGHT,
                1 : (1.8*x-6)*RIGHT+2*DOWN,
            })
            for x in range(8)
        ])

        source = Mobject(
            Line(0.5*UP, 2*UP),
            Source().scale(0.5),
            Line(2*DOWN, 0.5*DOWN)
        )
        source.shift(pos[0][0][0]*RIGHT)
        self.add(source)

        ohms = dict([
            (n, Tex("%d\\Omega"%int(n)).scale(0.75))
            for n in (1, 2, 3, 4, 6, 12, 1.1, 5, 10.1, 10, 8, 2.1)
        ])
        ohms[1].shift(0.5*pos[1][0]+0.5*pos[2][0]+0.7*UP)
        ohms[2].shift(pos[2][0.5]+0.7*LEFT)
        ohms[3].shift(pos[1][0.5]+0.7*LEFT)
        ohms[12].shift(pos[2][0.5]+0.7*LEFT)
        ohms[4].shift(pos[3][0.5]+0.7*LEFT)
        ohms[6].shift(pos[4][0.5]+0.7*LEFT)
        ohms[1.1].shift(0.5*pos[4][0]+0.5*pos[5][0]+0.7*UP)
        ohms[5].shift(pos[5][0.5]+0.7*LEFT)
        ohms[10.1].shift(pos[5][0.5]+0.7*LEFT)
        ohms[10].shift(pos[6][0.5]+0.7*LEFT)
        ohms[8].shift(pos[7][0.5]+0.7*LEFT)
        ohms[2.1].shift(0.5*pos[6][0]+0.5*pos[7][0]+0.7*UP)

        line1 = Line(pos[0][0], pos[1][0])
        line2 = Line(pos[0][1], pos[1][1])
        resistor = LongResistor(pos[1][0], pos[1][1])
        self.add(line1, line2, resistor, ohms[3])
        self.wait(3)

        combo_parts = [
            Resistor(pos[1][0], pos[2][0]),
            LongResistor(pos[2][0], pos[2][1]),
            Line(pos[2][1], pos[1][1])
        ]
        combo = Mobject(*combo_parts).ingest_submobjects()

        self.play(
            Transform(resistor, combo),
            Transform(ohms[3], ohms[2]),
            Transform(ohms[3].copy(), ohms[1])
        )
        self.wait(3)
        self.remove(ohms[3])
        self.add(ohms[2])
        self.remove(resistor)
        self.add(*combo_parts)
        resistor = combo_parts[1]


        top_point = Point(pos[2][0])
        line = Point(pos[2][0])
        bottom_point = Point(pos[2][1])
        self.play(
            Transform(top_point, Line(pos[2][0], pos[3][0])),
            Transform(line, Line(pos[3][0], pos[4][0])),
            Transform(bottom_point, Line(pos[2][1], pos[4][1])),
            Animation(resistor.copy()),
            Transform(resistor.copy(), LongResistor(pos[3][0], pos[3][1])),
            Transform(resistor, LongResistor(pos[4][0], pos[4][1])),
            Transform(ohms[2].copy(), ohms[12]),
            Transform(ohms[2].copy(), ohms[4]),
            Transform(ohms[2], ohms[6])
        )
        self.wait(3)
        self.remove(ohms[2])
        self.add(ohms[6])

        combo_parts = [
            Resistor(pos[4][0], pos[5][0]),
            LongResistor(pos[5][0], pos[5][1]),
            Line(pos[5][1], pos[4][1])
        ]
        combo = Mobject(*combo_parts).ingest_submobjects()
        self.play(
            # Transform(resistor, LongResistor(pos[5][0], pos[5][1])),
            # Transform(line, LongResistor(pos[3][0], pos[5][0])),
            # Transform(bottom_point, Line(pos[2][1], pos[5][1])),
            Transform(resistor, combo),
            Transform(ohms[6], ohms[5]),
            Transform(ohms[6].copy(), ohms[1.1])
        )
        self.wait(3)
        self.remove(ohms[6])
        self.add(ohms[5])
        self.remove(resistor)
        self.add(*combo_parts)
        resistor = combo_parts[1]

        line1 = Point(pos[5][0])
        line2 = Point(pos[5][1])

        self.play(
            Transform(line1, Line(pos[5][0], pos[6][0])),
            Transform(line2, Line(pos[5][1], pos[6][1])),
            Animation(resistor.copy()),
            Transform(resistor, LongResistor(pos[6][0], pos[6][1])),
            Transform(ohms[5].copy(), ohms[10.1]),
            Transform(ohms[5], ohms[10])
        )
        self.wait(3)
        self.remove(ohms[5])
        self.add(ohms[10])

        point1 = Point(pos[6][0])
        point2 = Point(pos[6][1])

        self.play(
            Transform(resistor, Mobject(
                Resistor(pos[6][0], pos[7][0]),
                LongResistor(pos[7][0], pos[7][1]),
                Line(pos[7][1], pos[6][1]),
            ).ingest_submobjects()),
            Transform(ohms[10], ohms[8]),
            Transform(ohms[10].copy(), ohms[2.1])
        )
        self.wait(3)

        # self.reverse_frames()
        self.invert_colors()


        # circuit = Mobject(*[
        #     source,
        #     #
        #     LongResistor(pos[0][0], pos[2][0]),
        #     Line(pos[0][1], pos[2][1]),
        #     #
        #     LongResistor(pos[2][0], pos[2][1]),
        #     Line(pos[2][0], pos[3][0]),
        #     Line(pos[2][1], pos[3][1]),
        #     LongResistor(pos[3][0], pos[3][1]),
        #     #
        #     LongResistor(pos[3][0], pos[5][0]),
        #     Line(pos[3][1], pos[5][1]),
        #     #
        #     LongResistor(pos[5][0], pos[5][1]),
        #     #
        #     Line(pos[5][0], pos[6][0]),
        #     Resistor(pos[6][0], pos[7][0]),            
        #     Line(pos[5][1], pos[7][1]),
        #     #
        #     LongResistor(pos[7][0], pos[7][1])
        # ])
















