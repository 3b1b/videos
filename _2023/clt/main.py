from manim_imports_ext import *

from _2022.convolutions.continuous import *


class GaussConvolutions(Convolutions):
    conv_y_stretch_factor = 1.0

    def construct(self):
        super().construct()

    def f(self, x):
        return np.exp(-x**2)

    def g(self, x):
        return np.exp(-x**2)
