from manim_imports_ext import *


class RotatingButterflyCurve(Animation):
    """
    Pretty hacky, but should only be redone in the context
    of a broader 4d mobject class
    """
    CONFIG = {
        "space_epsilon": 0.002,
        "grid_line_sep": 0.2,
        "radius": 2,
        "radians": 2 * np.pi,
        "run_time": 15,
        "rate_func": linear,
    }

    def __init__(self, **kwargs):
        digest_config(self, kwargs)
        fine_range = np.arange(-self.radius, self.radius, self.space_epsilon)
        corse_range = np.arange(-self.radius, self.radius, self.grid_line_sep)
        self.points = np.array([
            [x, y, x * x, y * y]
            for a in fine_range
            for b in corse_range
            for x, y in [(a, b), (b, a)]
        ])
        graph_rgb = Color(TEAL).get_rgb()
        self.rgbas = np.array([graph_rgb] * len(self.points))
        colors = iter([YELLOW_A, YELLOW_B, YELLOW_C, YELLOW_D])
        for i in range(4):
            vect = np.zeros(4)
            vect[i] = 1
            line_range = np.arange(-FRAME_Y_RADIUS,
                                   FRAME_Y_RADIUS, self.space_epsilon)
            self.points = np.append(self.points, [
                x * vect
                for x in line_range
            ], axis=0)
            axis_rgb = Color(next(colors)).get_rgb()
            self.rgbas = np.append(
                self.rgbas, [axis_rgb] * len(line_range), axis=0)

        self.quads = [(0, 1, 2, 3), (0, 1, 0, 1), (3, 2, 1, 0)]
        self.multipliers = [0.9, 1, 1.1]
        Animation.__init__(self, Mobject(), **kwargs)

    def interpolate_mobject(self, alpha):
        angle = alpha * self.radians
        rot_matrix = np.identity(4)
        for quad, mult in zip(self.quads, self.multipliers):
            base = np.identity(4)
            theta = mult * angle
            base[quad[:2], quad[2]] = [np.cos(theta), np.sin(theta)]
            base[quad[:2], quad[3]] = [-np.sin(theta), np.cos(theta)]
            rot_matrix = np.dot(rot_matrix, base)
        points = np.dot(self.points, np.transpose(rot_matrix))
        self.mobject.points = points[:, :3]
        self.mobject.rgbas = self.rgbas


class RotatingFourDButterflyCurve(Scene):
    def construct(self):
        self.play(RotatingButterflyCurve())
