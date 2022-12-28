from manim_imports_ext import *

class Cycloidify(Scene):
    def construct(self):
        def cart_to_polar(xxx_todo_changeme):
            (x, y, z) = xxx_todo_changeme
            return x*RIGHT+x*y*UP
        def polar_to_cycloid(point):
            epsilon = 0.00001
            t = get_norm(point)
            R = point[1]/(point[0]+epsilon)+epsilon
            return R*(t/R-np.sin(t/R))*RIGHT+R*(1-np.cos(t/R))*UP
        polar = Mobject(*[
            Line(ORIGIN, T*(np.cos(theta)*RIGHT+np.sin(theta)*UP))
            for R in np.arange(0.25, 4, 0.25)
            for theta in [np.arctan(R)]
            for T in [R*2*np.pi]
        ])
        polar.set_color(BLUE)
        cycloids = polar.copy().apply_function(polar_to_cycloid)
        cycloids.set_color(YELLOW)
        for mob in polar, cycloids:
            mob.rotate(np.pi, RIGHT)
            mob.to_corner(UP+LEFT)
        lines = polar.copy()

        self.add(lines)
        self.wait()
        self.play(Transform(
            lines, cycloids, 
            run_time = 3,
            path_func = path_along_arc(np.pi/2)
        ))
        self.wait()
        self.play(Transform(
            lines, polar,
            run_time = 3,
            path_func = path_along_arc(-np.pi/2)
        ))
        self.wait()

class PythagoreanTransformation(Scene):
    def construct(self):
        triangle = Mobject(
            Line(ORIGIN, 4*UP, color = "#FF69B4"),
            Line(4*UP, 2*RIGHT, color = YELLOW_C),
            Line(2*RIGHT, ORIGIN, color = BLUE_D)
        )
        arrangment1 = Mobject(*[
            triangle.copy().rotate(theta).shift(3*vect)
            for theta, vect in zip(
                np.arange(0, 2*np.pi, np.pi/2),
                compass_directions(4, DOWN+LEFT)
            )
        ])
        arrangment2 = Mobject(
            triangle.copy().rotate(np.pi, UP).rotate(-np.pi/2).shift(3*DOWN+3*LEFT),
            triangle.copy().rotate(np.pi, UP).rotate(np.pi/2).shift(DOWN+RIGHT),
            triangle.copy().shift(DOWN+RIGHT),
            triangle.copy().rotate(np.pi).shift(3*UP+3*RIGHT)
        )
        growth = 1.2
        a_region, b_region, c_region = regions = [
            MobjectFromRegion(
                region_from_polygon_vertices(
                    *compass_directions(4, growth*start)
                ),
                color
            ).scale(1/growth).shift(shift_val)
            for start, color, shift_val in [
                (DOWN+RIGHT, BLUE_D, 2*(DOWN+RIGHT)),
                (2*(DOWN+RIGHT), MAROON_B, UP+LEFT),
                (3*RIGHT+DOWN, YELLOW_E, ORIGIN),
            ]
        ]
        for mob, char in zip(regions, "abc"):
            mob.add(
                OldTex("%s^2"%char).shift(mob.get_center())
            )

        square = Square(side_length = 6, color = WHITE)
        mover = arrangment1.copy()

        self.add(square, mover)
        self.play(FadeIn(c_region))
        self.wait(2)
        self.remove(c_region)
        self.play(Transform(
            mover, arrangment2,
            run_time = 3,
            path_func = path_along_arc(np.pi/2)
        ))
        self.remove(c_region)
        self.play(*[
            FadeIn(region)
            for region in (a_region, b_region)
        ])
        self.wait(2)
        self.clear()
        self.add(mover, square)
        self.play(Transform(
            mover, arrangment1,
            run_time = 3,
            path_func = path_along_arc(-np.pi/2)
        ))
        self.wait()

class PullCurveStraight(Scene):
    def construct(self):
        start = -1.5
        end = 1.5
        parabola = ParametricCurve(
            lambda t : (t**3-t)*RIGHT + (2*np.exp(-t**2))*UP,
            start = start,
            end = end,
            color = BLUE_D,
            density = 2*DEFAULT_POINT_DENSITY_1D
        )
        from scipy import integrate
        integral = integrate.quad(
            lambda x : np.sqrt(1 + 4*x**2), start, end
        )
        length = integral[0]
        line = Line(
            0.5*length*LEFT,
            0.5*length*RIGHT,
            color = BLUE_D
        )
        brace = Brace(line, UP)
        label = OldTexText("What is this length?")
        label.next_to(brace, UP)

        self.play(ShowCreation(parabola))
        self.wait()
        self.play(Transform(
            parabola, line,
            path_func = path_along_arc(np.pi/2),
            run_time = 2
        ))
        self.play(
            GrowFromCenter(brace),
            ShimmerIn(label)
        )
        self.wait(3)

class StraghtenCircle(Scene):
    def construct(self):
        radius = 1.5
        radius_line = Line(ORIGIN, radius*RIGHT, color = RED_D)
        radius_brace = Brace(radius_line, UP)
        r = OldTex("r").next_to(radius_brace, UP)
        circle = Circle(radius = radius, color = BLUE_D)
        line = Line(
            np.pi*radius*LEFT, 
            np.pi*radius*RIGHT,
            color = circle.get_color()
        )
        line_brace = Brace(line, UP)
        two_pi_r = OldTex("2\\pi r").next_to(line_brace, UP)

        self.play(ShowCreation(radius_line))
        self.play(ShimmerIn(r), GrowFromCenter(radius_brace))
        self.wait()
        self.remove(r, radius_brace)
        self.play(
            ShowCreation(circle),
            Rotating(
                radius_line, 
                axis = OUT,
                rate_func = smooth,
                in_place = False
            ),
            run_time = 2
        )
        self.wait()
        self.remove(radius_line)
        self.play(Transform(
            circle, line,
            run_time = 2
        ))
        self.play(
            ShimmerIn(two_pi_r),
            GrowFromCenter(line_brace)
        )
        self.wait(3)

class SingleVariableFunc(Scene):
    def construct(self):
        start = OldTex("3").set_color(GREEN)
        start.scale(2).shift(5*LEFT+2*UP)
        end = OldTex("9").set_color(RED)
        end.scale(2).shift(5*RIGHT+2*DOWN)
        point = Point()
        func = OldTex("f(x) = x^2")
        circle = Circle(color = WHITE, radius = func.get_width()/1.5)

        self.add(start, circle, func)
        self.wait()
        self.play(Transform(
            start, point, 
            path_func = path_along_arc(np.pi/2)
        ))
        self.play(Transform(
            start, end,
            path_func = path_along_arc(-np.pi/2)
        ))
        self.wait()

class MultivariableFunc(Scene):
    def construct(self):
        start = OldTex("(1, 2)").set_color(GREEN)
        start.scale(1.5).shift(5*LEFT, 2*UP)
        end = OldTex("(5, -3)").set_color(RED)
        end.scale(1.5).shift(5*RIGHT, 2*DOWN)
        point = Point()
        func = OldTex("f(x, y) = (x^2+y^2, x^2-y^2)")
        circle = Circle(color = WHITE)
        circle.stretch_to_fit_width(func.get_width()+1)

        self.add(start, circle, func)
        self.wait()
        self.play(Transform(
            start, point, 
            path_func = path_along_arc(np.pi/2)
        ))
        self.play(Transform(
            start, end,
            path_func = path_along_arc(-np.pi/2)
        ))
        self.wait()        

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

class ShowSumOfSquaresPattern(Scene):
    def construct(self):
        dots = VGroup(*[
            VGroup(*[
                Dot() for x in range(2*n + 1)
            ]).arrange(DOWN, buff = SMALL_BUFF)
            for n in range(30)
        ]).arrange(RIGHT, buff = MED_LARGE_BUFF, aligned_edge = UP)
        dots = VGroup(*it.chain(*dots))
        dots.to_edge(UP)
        numbers = VGroup()
        for n, dot in enumerate(dots):
            factors = prime_factors(n)
            color = True
            factors_to_counts = dict()
            for factor in factors:
                if factor not in factors_to_counts:
                    factors_to_counts[factor] = 0
                factors_to_counts[factor] += 1

            for factor, count in list(factors_to_counts.items()):
                if factor%4 == 3 and count%2 == 1:
                    color = False
            n_mob = Integer(n)
            n_mob.replace(dot, dim_to_match = 1)
            if color:
                n_mob.set_color(RED)
            numbers.add(n_mob)

        self.add(numbers)