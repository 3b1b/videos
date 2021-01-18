from manim_imports_ext import *


class FluidFlow(Scene):
    CONFIG = {
        "vector_spacing" : 1,
        "dot_spacing" : 0.5,
        "dot_color" : BLUE_C,
        "text_color" : WHITE,
        "vector_color" : YELLOW,
        "vector_length" : 0.5,
        "points_height" : FRAME_Y_RADIUS,
        "points_width" : FRAME_X_RADIUS,
    }
    def use_function(self, function):
        self.function = function

    def get_points(self, spacing):
        x_radius, y_radius = [
            val-val%spacing
            for val in self.points_width, self.points_height
        ]
        return map(np.array, it.product(
            np.arange(-x_radius, x_radius+spacing, spacing),
            np.arange(-y_radius, y_radius+spacing, spacing),
            [0]
        ))


    def add_axes(self, show_creation = False):
        self.axes = Axes(color = WHITE, tick_frequency = 1)
        self.add(self.axes)
        if show_creation:
            self.play(ShowCreation(self.axes))
            self.wait()

    def add_dots(self, show_creation = False):
        points = self.get_points(self.dot_spacing)
        self.dots = VMobject(*map(Dot, points))
        self.dots.set_color(self.dot_color)
        self.add(self.dots)
        if show_creation:
            self.play(ShowCreation(self.dots))
            self.wait()

    def add_vectors(self, true_length = False, show_creation = False):
        if not hasattr(self, "function"):
            raise Exception("Must run use_function first")
        points = self.get_points(self.vector_spacing)
        points = filter(
            lambda p : get_norm(self.function(p)) > 0.01,
            points
        )
        directions = map(self.function, points)
        if not true_length:
            directions = [
                self.vector_length*d/get_norm(d)
                for d in directions
            ]
        self.vectors = VMobject(*[
            Vector(
                direction,
                color = self.vector_color,
                tip_length = 0.1,
            ).shift(point)
            for point, direction in zip(points, directions)
        ])
        self.add(self.vectors)
        if show_creation:
            self.play(ShowCreation(self.vectors))
            self.wait()

    def flow(self, **kwargs):
        if not hasattr(self, "function"):
            raise Exception("Must run use_function first")
        # Warning, this is now depricated
        self.play(ApplyToCenters(
            PhaseFlow,
            self.dots.split(),
            function = self.function,
            **kwargs
        ))

    def label(self, text, time = 5):
        mob = TexText(text)
        mob.scale(1.5)
        mob.to_edge(UP)
        mob.set_color(self.text_color)        
        rectangle = Polygon(*[
            mob.get_corner(vect) + 0.3*vect
            for vect in [
                UP+RIGHT,
                UP+LEFT,
                DOWN+LEFT,
                DOWN+RIGHT
            ]
        ])
        rectangle.set_fill(BLACK, 1.0)
        self.add(rectangle, mob)
        self.wait(time)
        self.remove(mob, rectangle)


class VectorFieldExample(FluidFlow):
    CONFIG = {
        "points_height" : 4,
        "points_width" : 4,
        "vector_spacing" : 0.5,
        "vector_length" : 0.3
    }
    def construct(self):
        self.use_function(
            lambda (x, y, z) : 0.5*((2*y)**3-9*(2*y))*RIGHT+0.5*((2*x)**3-9*(2*x))*UP
        )
        self.add_axes(show_creation = True)
        self.add_vectors(show_creation = True)
        self.add_dots(show_creation = True)
        self.show_frame()
        self.flow(run_time = 30, virtual_time = 3)

class VectorFieldExampleWithoutArrows(FluidFlow):
    def construct(self):
        self.use_function(
            lambda (x, y, z) : 0.5*(y**3-9*y)*RIGHT+0.5*(x**3-9*x)*UP
        )
        self.add_axes(show_creation = True)
        self.add_dots(show_creation = True)
        self.flow(run_time = 30, virtual_time = 3)


class VectorFieldExampleTwo(FluidFlow):
    CONFIG = {
        "points_width" : 3*FRAME_X_RADIUS,
        "points_height" : 1.4*FRAME_Y_RADIUS
    }
    def construct(self):
        self.use_function(
            lambda (x, y, z) : RIGHT+np.sin(x)*UP
        )
        self.add_axes()
        self.add_vectors()
        self.add_dots(show_creation = True)
        for x in range(10):
            self.flow(
                run_time = 1,
                rate_func=linear,
            )


class VectorFieldExampleThree(FluidFlow):
    def construct(self):
        self.use_function(
            lambda p : p/(2*get_norm(0.5*p)**0.5+0.01)
        )
        self.add_axes()
        self.add_vectors()  
        self.add_dots(show_creation = True)
        self.flow(run_time = 2, virtual_time = 2)
        self.wait(2)


class VectorFieldExampleFour(FluidFlow):
    CONFIG = {
        "points_height" : 1.FRAME_WIDTH,
        "points_width" : 1.FRAME_WIDTH,
    }
    def construct(self):
        self.use_function(
            lambda (x, y, z) : (x*UP - y*RIGHT)/5
        )
        self.add_axes()
        self.add_vectors(true_length = True)  
        self.add_dots(show_creation = True)
        self.show_frame()
        self.play(Rotating(
            self.dots, 
            run_time = 10, 
            axis = OUT
        ))
        self.wait(2)        


class FluxArticleExample(FluidFlow):
    CONFIG = {
        "vector_length" : 0.4,
        "vector_color" : BLUE_D,
        "points_height" : FRAME_Y_RADIUS,
        "points_width" : FRAME_X_RADIUS,
    }
    def construct(self):
        self.use_function(
            lambda (x, y, z) : (x**2+y**2)*((np.sin(x)**2)*RIGHT + np.cos(y)*UP)
        )
        # self.add_axes()
        self.add_vectors()
        self.show_frame()
        self.add_dots()        
        self.flow(run_time = 2, virtual_time = 0.1)
        self.wait(2)

class NegativeDivergenceExamlpe(FluidFlow):
    CONFIG = {
        "points_width" : FRAME_WIDTH,
        "points_height" : FRAME_HEIGHT,
    }
    def construct(self):
        circle = Circle(color = YELLOW_C)
        self.use_function(
            lambda p : -p/(2*get_norm(0.5*p)**0.5+0.01)
        )
        self.add_axes()
        self.add_vectors()
        self.play(ShowCreation(circle))
        self.wait()
        self.add_dots(show_creation = True)        
        self.flow(
            run_time = 1, 
            virtual_time = 1,
            rate_func = smooth
       )
        self.wait(2)


class PositiveDivergenceExample(FluidFlow):
    def construct(self):
        circle = Circle(color = YELLOW_C)
        self.use_function(
            lambda p : p/(2*get_norm(0.5*p)**0.5+0.01)
        )
        self.add_axes()
        self.add_vectors()
        self.play(ShowCreation(circle))
        self.wait()
        self.add_dots(show_creation = True)        
        self.flow(
            run_time = 1, 
            virtual_time = 1, 
            rate_func = smooth
        )
        self.wait(2)

class DivergenceArticleExample(FluidFlow):
    def construct(self):
        def raw_function((x, y, z)):
            return (2*x-y, y*y, 0)
        def normalized_function(p):
            result = raw_function(p)
            return result/(get_norm(result)+0.01)
        self.use_function(normalized_function)

        self.add_axes()
        self.add_vectors()
        self.add_dots()
        self.flow(
            virtual_time = 4,
            run_time = 5
        )

class QuadraticField(FluidFlow):
    def construct(self):
        self.use_function(
            lambda (x, y, z) : 0.25*((x*x-y*y)*RIGHT+x*y*UP)
        )
        self.add_axes()
        self.add_vectors()
        self.add_dots()
        self.flow(
            virtual_time = 10,
            run_time = 20,
            rate_func=linear
        )


class IncompressibleFluid(FluidFlow):
    CONFIG = {
        "points_width" : FRAME_WIDTH,
        "points_height" : 1.4*FRAME_Y_RADIUS
    }
    def construct(self):
        self.use_function(
            lambda (x, y, z) : RIGHT+np.sin(x)*UP
        )
        self.add_axes()
        self.add_vectors()
        self.add_dots()
        for x in range(8):
            self.flow(
                run_time = 1,
                rate_func=linear,
            )



class ConstantInwardFlow(FluidFlow):
    CONFIG = {
        "points_height" : FRAME_HEIGHT,
        "points_width" : FRAME_WIDTH,
    }
    def construct(self):
        self.use_function(
            lambda p : -3*p/(get_norm(p)+0.1)
        )
        self.add_axes()
        self.add_vectors()
        self.add_dots()
        for x in range(5):
            self.flow(
                run_time = 5,
                rate_func=linear,
            )




class ConstantOutwardFlow(FluidFlow):
    def construct(self):
        self.use_function(
            lambda p : p/(2*get_norm(0.5*p)**0.5+0.01)
        )
        self.add_axes()
        self.add_vectors()
        self.add_dots()
        for x in range(10):
            self.flow(rate_func=linear)
            dot = self.dots.split()[0].copy()
            dot.center()
            new_dots = [
                dot.copy().shift(0.5*vect)
                for vect in [
                    UP, DOWN, LEFT, RIGHT, 
                    UP+RIGHT, UP+LEFT, DOWN+RIGHT, DOWN+LEFT
                ]
            ]
            self.dots.add(*new_dots)


class ConstantPositiveCurl(FluidFlow):
    CONFIG = {
        "points_height" : FRAME_X_RADIUS,
    }
    def construct(self):
        self.use_function(
            lambda p : 0.5*(-p[1]*RIGHT+p[0]*UP)
        )
        self.add_axes()
        self.add_vectors(true_length = True)
        self.add_dots()
        for x in range(10):
            self.flow(
                rate_func=linear
            )



class ComplexCurlExample(FluidFlow):
    def construct(self):
        self.use_function(
            lambda (x, y, z) : np.cos(x+y)*RIGHT+np.sin(x*y)*UP
        )
        self.add_axes()
        self.add_vectors(true_length = True)
        self.add_dots()
        for x in range(4):
            self.flow(
                run_time = 5,
                rate_func=linear,
            )

class SingleSwirl(FluidFlow):
    CONFIG = {
        "points_height" : FRAME_X_RADIUS,
    }
    def construct(self):
        self.use_function(
            lambda p : (-p[1]*RIGHT+p[0]*UP)/get_norm(p)
        )
        self.add_axes()
        self.add_vectors()
        self.add_dots()
        for x in range(10):
            self.flow(rate_func=linear)


class CurlArticleExample(FluidFlow):
    CONFIG = {
        "points_height" : 3*FRAME_Y_RADIUS,
        "points_width" : 3*FRAME_X_RADIUS
    }
    def construct(self):
        self.use_function(
            lambda (x, y, z) : np.cos(0.5*(x+y))*RIGHT + np.sin(0.25*x*y)*UP
        )
        circle = Circle().shift(3*UP)
        self.add_axes()
        self.add_vectors()
        self.play(ShowCreation(circle))
        self.add_dots()
        self.show_frame()
        self.flow(
            rate_func=linear,
            run_time = 15,
            virtual_time = 10
        )


class FourSwirlsWithoutCircles(FluidFlow):
    CONFIG = {
        "points_height" : FRAME_X_RADIUS,
    }
    def construct(self):
        circles = [
            Circle().shift(3*vect)
            for vect in compass_directions()
        ]
        self.use_function(
            lambda (x, y, z) : 0.5*(y**3-9*y)*RIGHT+(x**3-9*x)*UP
        )
        self.add_axes()
        self.add_vectors()
        # for circle in circles:
        #     self.play(ShowCreation(circle))
        self.add_dots()
        self.add_extra_dots()
        self.flow(
            virtual_time = 2,
            run_time = 20,
            rate_func=linear
        )

    def add_extra_dots(self):
        dots = self.dots.split()
        for vect in UP+LEFT, DOWN+RIGHT:
            for n in range(5, 15):
                dots.append(
                    dots[0].copy().center().shift(n*vect)
                )
        self.dots = VMobject(*dots)


class CopyPlane(Scene):
    def construct(self):
        def special_rotate(mob):
            mob.rotate(0.9*np.pi/2, RIGHT, about_point = ORIGIN)
            mob.rotate(-np.pi/4, UP, about_point = ORIGIN)
            return mob
        plane = NumberPlane()
        copies = [
            special_rotate(plane.copy().shift(u*n*OUT))
            for n in range(1, 3)
            for u in -1, 1
        ]
        line = Line(4*IN, 4*OUT)


        self.add(plane)
        self.play(*[
            ApplyFunction(special_rotate, mob, run_time = 3)
            for mob in plane, line
        ])
        self.wait()
        for copy in copies:
            self.play(Transform(plane.copy(), copy))
        self.wait()


class DropletFlow(FluidFlow):
    def construct(self):
        seconds = 60*5
        droplets = Group(*[
            PointDot(x*RIGHT+y*UP, radius = 0.15, density = 120)
            for x in range(-7, 9)
            for y in range(-3, 4)
        ])
        droplets.set_color_by_gradient(BLUE, GREEN, YELLOW)
        self.use_function(
            lambda (x, y, z) : y*RIGHT+np.sin(2*np.pi*x)*UP,
        )
        self.add(NumberPlane().fade())
        self.play(ShowCreation(droplets))
        n_steps = int(seconds * self.camera.frame_rate)
        from tqdm import tqdm as ProgressDisplay
        for x in ProgressDisplay(range(n_steps)):
            for d in droplets:
                if x%10 == 0:
                    d.filter_out(
                        lambda p : abs(p[0]) > 1.5*FRAME_X_RADIUS or abs(p[1]) > 1.5*FRAME_Y_RADIUS
                    )
                for p in d.points:
                    p += 0.001*self.function(p)
            self.wait(1 / self.camera.frame_rate)


class AltDropletFlow(FluidFlow):
    def construct(self):
        self.use_function(lambda (x, y, z):
            (np.sin(x)+np.sin(y))*RIGHT+\
            (np.sin(x)-np.sin(y))*UP
        )
        self.add_dots()
        self.flow(
            rate_func=linear,
            run_time = 10,
            virtual_time = 2
        )













