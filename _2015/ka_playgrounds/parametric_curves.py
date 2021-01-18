#!/usr/bin/env python

from manim_imports_ext import *

class VectorDraw(Animation):
    CONFIG = {
        "vector_color" : GREEN_B,
        "line_color"   : YELLOW,
        "t_min" : 0, 
        "t_max" : 10,
        "run_time" : 7,
    }
    def __init__(self, func, **kwargs):
        digest_config(self, kwargs, locals())
        self.curve = ParametricCurve(
            func, t_min = self.t_min, t_max = self.t_max,
            color = self.line_color
        )
        self.vector = Vector(
            func(self.t_min),
            color = self.vector_color
        )
        mobject = VMobject(self.vector, self.curve) ##First is filler
        Animation.__init__(self, mobject, **kwargs)

    def interpolate_mobject(self, alpha):
        t = alpha*self.t_max + (1 - alpha)*self.t_min
        self.vector.put_start_and_end_on(ORIGIN, self.func(t))        
        old_curve = self.starting_mobject.split()[1]
        self.curve.pointwise_become_partial(old_curve, 0, alpha)
        return self

class Test(Scene):
    def construct(self):
        axes = Axes()
        def func(t):
            return 0.5*(t*np.cos(t)*RIGHT + t*np.sin(t)*UP)

        words = TexText("Parametric functions")
        words.set_color(YELLOW)
        words.to_edge(UP+LEFT)
        self.add(words)
        # v = Vector(ORIGIN)
        # v.put_start_and_end_on(LEFT, RIGHT)
        # v.show()

        self.play(ShowCreation(axes))
        self.wait(2)
        self.play(VectorDraw(func))
        self.wait(3)
        self.clear()
        self.add(axes, words)
        self.wait()
        self.play(VectorDraw(func, rate_func = rush_from))
        self.wait()

