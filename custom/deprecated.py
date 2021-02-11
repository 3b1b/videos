# This file contains functions and classes which have been used in old
# videos, but which have since been supplanted in manim.  For example,
# there were previously various specifically named classes for version
# of fading which are now covered by arguments passed into FadeIn and
# FadeOut

from manim_imports_ext import *


class FadeInFromDown(FadeIn):
    def __init__(self, mobject, **kwargs):
        super().__init__(mobject, UP, **kwargs)


class FadeOutAndShiftDown(FadeOut):
    def __init__(self, mobject, **kwargs):
        super().__init__(mobject, DOWN, **kwargs)


class FadeInFromLarge(FadeIn):
    def __init__(self, mobject, scale_factor=2, **kwargs):
        super().__init__(mobject, scale=(1 / scale_factor), **kwargs)
