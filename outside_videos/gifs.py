from manim_imports_ext import *


class HyperSlinky(Scene):
    def construct(self):
        self.play(
            ApplyPointwiseFunction(
                lambda x_y_z: (1 + x_y_z[1]) * np.array((
                    np.cos(2 * np.pi * x_y_z[0]),
                    np.sin(2 * np.pi * x_y_z[0]),
                    x_y_z[2]
                )),
                NumberPlane().prepare_for_nonlinear_transform(),
                rate_func=there_and_back,
                run_time=10,
            )
        )


class CircleAtaphogy(Scene):
    def construct(self):
        self.play(
            ApplyMethod(Circle(radius=3).repeat, 7),
            run_time=3.0
        )
