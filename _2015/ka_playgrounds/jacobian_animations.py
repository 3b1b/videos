#!/usr/bin/env python
from manim_imports_ext import *

from zeta import *

class ExampleLinearTransformation(LinearTransformationScene):
    CONFIG = {
        "show_coordinates" : True
    }
    def construct(self):
        self.wait()
        self.apply_transposed_matrix([[2, 1], [-3, 1]])
        self.wait()

def example_function(point):
    x, y, z = point
    return np.array([
        x + np.sin(y),
        y + np.sin(x),
        0
    ])

class ExampleMultivariableFunction(LinearTransformationScene):
    CONFIG = {
        "show_basis_vectors" : False,
        "show_coordinates" : True,
    }
    def construct(self):
        self.wait()
        self.apply_nonlinear_transformation(example_function)
        self.wait()
        
class ExampleMultivariableFunctionWithZoom(ZoomedScene, ExampleMultivariableFunction):
    def construct(self):
        self.activate_zooming()
        self.little_rectangle.set_color(YELLOW)
        point = 2*LEFT + UP
        self.little_rectangle.move_to(point)
        dense_lines = self.get_dense_lines(point)

        self.play(ShowCreation(dense_lines))
        self.plane.add(dense_lines)
        self.wait()
        self.apply_nonlinear_transformation(
            example_function,
            added_anims = [ApplyMethod(
                self.little_rectangle.move_to,
                example_function(point),
                run_time = 3
            )]
        )
        self.wait()

    def get_dense_lines(self, point):
        radius = 0.4*self.little_rectangle.get_height()
        n_steps = 5

        vert_lines = VGroup(*[
            Line(DOWN, UP).scale(FRAME_Y_RADIUS).shift(x*RIGHT)
            for x in np.linspace(point[0]-radius, point[0]+radius, n_steps)
        ])
        horiz_lines = VGroup(*[
            Line(LEFT, RIGHT).scale(FRAME_X_RADIUS).shift(y*UP)
            for y in np.linspace(point[1]-radius, point[1]+radius, n_steps)            
        ])
        dense_lines = VGroup(vert_lines, horiz_lines)
        dense_lines.set_stroke(BLUE, width = 2)
        for group in vert_lines, horiz_lines:
            group[n_steps/2].set_color(WHITE)        
        return dense_lines

    def capture_mobjects_in_camera(self, mobjects, **kwargs):
        self.camera.capture_mobjects(mobjects, **kwargs)
        if self.zoom_activated:
            filter_mobjects = [m for m in mobjects if m not in self.background_plane.get_family()]
            self.zoomed_camera.capture_mobjects(
                filter_mobjects, **kwargs
            )

class ExampleMultivariableFunctionWithMuchZoom(ExampleMultivariableFunctionWithZoom):
    CONFIG = {
        "zoom_factor" : 20
    }

class ExampleDeterminantAnimation(LinearTransformationScene):
    CONFIG = {
        "show_coordinates" : True,
    }
    def construct(self):
        self.add_unit_square()
        self.wait()
        self.apply_transposed_matrix([[3, 0], [1, 2]])
        self.wait(2)

class JacobianDeterminantAnimation(ExampleMultivariableFunctionWithMuchZoom):
    CONFIG = {
        "point" : 2*LEFT+UP
    }
    def construct(self):
        self.activate_zooming()
        self.little_rectangle.set_color(YELLOW)
        point = self.point
        self.little_rectangle.move_to(point)
        dense_lines = self.get_dense_lines(point)

        self.add_unit_square()
        tiny_unit = get_norm(
            dense_lines[0][1].get_center()-dense_lines[0][0].get_center()
        )
        self.square.scale(tiny_unit)
        self.square.shift(point)

        self.play(ShowCreation(dense_lines))
        self.plane.add(dense_lines)
        self.wait()
        self.apply_nonlinear_transformation(
            example_function,
            added_anims = [ApplyMethod(
                self.little_rectangle.move_to,
                example_function(point),
                run_time = 3
            )]
        )
        self.wait()

class SmallJacobianDeterminant(JacobianDeterminantAnimation):
    CONFIG = {
        "point" : UP,
    }

































