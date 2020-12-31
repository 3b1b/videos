
from big_ol_pile_of_manim_imports import *


class HyperSlinky(Scene):
    def construct(self):
        self.play(
            ApplyPointwiseFunction(
                lambda x_y_z: (1 + x_y_z[1]) * np.array((
                    np.cos(2 * np.pi * x_y_z[0]),
                    np.sin(2 * np.pi * x_y_z[0]),
                    x_y_z[2]
                )),
                NumberPlane(),
                rate_func=there_and_back,
                run_time=10,
            )
        )


class CircleAtaphogy(Scene):
    def construct(self):
        self.play(
            DelayByOrder(ApplyMethod(Circle(radius=3).repeat, 7)),
            run_time=3.0
        )


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


class PascalColored(Scene):
    CONFIG = {
        "colors": [BLUE_E, BLUE_D, BLUE_B],
        "dot_radius": 0.16,
        "n_layers": 2 * 81,
        "rt_reduction_factor": 0.5,
    }

    def construct(self):
        max_height = 6
        rt = 1.0

        layers = self.get_dots(self.n_layers)
        triangle = VGroup(layers[0])
        triangle.to_edge(UP, buff=LARGE_BUFF)
        self.add(triangle)
        last_layer = layers[0]
        for layer in layers[1:]:
            height = last_layer.get_height()
            layer.set_height(height)
            layer.next_to(last_layer, DOWN, 0.3 * height)
            for i, dot in enumerate(layer):
                pre_dots = VGroup(*last_layer[max(i - 1, 0):i + 1])
                self.play(*[
                    ReplacementTransform(
                        pre_dot.copy(), dot,
                        run_time=rt
                    )
                    for pre_dot in pre_dots
                ])
            last_layer = layer
            triangle.add(layer)
            if triangle.get_height() > max_height:
                self.play(
                    triangle.set_height, 0.5 * max_height,
                    triangle.to_edge, UP, LARGE_BUFF
                )
                rt *= self.rt_reduction_factor
                print(rt)
        self.wait()

    def get_pascal_point(self, n, k):
        return n * rotate_vector(RIGHT, -2 * np.pi / 3) + k * RIGHT

    def get_dot_layer(self, n):
        n_to_mod = len(self.colors)
        dots = VGroup()
        for k in range(n + 1):
            point = self.get_pascal_point(n, k)
            # p[0] *= 2
            nCk_residue = choose(n, k) % n_to_mod
            dot = Dot(
                point,
                radius=2 * self.dot_radius,
                color=self.colors[nCk_residue]
            )
            if n <= 9:
                num = TexMobject(str(nCk_residue))
                num.set_height(0.5 * dot.get_height())
                num.move_to(dot)
                dot.add(num)
            # num = DecimalNumber(choose(n, k), num_decimal_points = 0)
            # num.set_color(dot.get_color())
            # max_width = 2*dot.get_width()
            # max_height = dot.get_height()
            # if num.get_width() > max_width:
            #     num.set_width(max_width)
            # if num.get_height() > max_height:
            #     num.set_height(max_height)
            # num.move_to(dot, aligned_edge = DOWN)
            dots.add(dot)
        return dots

    def get_dots(self, n_layers):
        dots = VGroup()
        for n in range(n_layers + 1):
            dots.add(self.get_dot_layer(n))
        return dots


class StacksApproachBellCurve(Scene):
    CONFIG = {
        "n_iterations": 70,
    }

    def construct(self):
        bar = Square(side_length=1)
        bar.set_fill(BLUE, 1)
        bar.set_stroke(BLUE, 1)
        bars = VGroup(bar)

        max_width = FRAME_WIDTH - 2
        max_height = FRAME_Y_RADIUS - 1.5

        for x in range(self.n_iterations):

            bars_copy = bars.copy()

            #Copy and shift
            for mob, vect in (bars, DOWN), (bars_copy, UP):
                mob.generate_target()
                if mob.target.get_height() > max_height:
                    mob.target.stretch_to_fit_height(max_height)
                if mob.target.get_width() > max_width:
                    lx1 = mob.target[1].get_left()[0]
                    rx0 = mob.target[0].get_right()[0]
                    curr_buff = lx1 - rx0
                    mob.target.arrange(
                        RIGHT, buff=0.9 * curr_buff,
                        aligned_edge=DOWN
                    )
                    mob.target.stretch_to_fit_width(max_width)
                mob.target.next_to(ORIGIN, vect, MED_LARGE_BUFF)
            colors = color_gradient([BLUE, YELLOW], len(bars) + 1)
            for color, bar in zip(colors, bars.target):
                bar.set_color(color)
            for color, bar in zip(colors[1:], bars_copy.target):
                bar.set_color(color)
            bars_copy.set_fill(opacity=0)
            bars_copy.set_stroke(width=0)
            if x == 0:
                distance = 1.5
            else:
                cx1 = bars.target[-1].get_center()[0]
                cx0 = bars.target[0].get_center()[0]
                distance = (cx1 - cx0) / (len(bars) - 1)
            self.play(*list(map(MoveToTarget, [bars, bars_copy])))
            self.play(
                bars.shift, distance * LEFT / 2,
                bars_copy.shift, distance * RIGHT / 2,
            )

            # Stack
            bars_copy.generate_target()
            for i in range(len(bars) - 1):
                top_bar = bars_copy.target[i]
                low_bar = bars[i + 1]
                top_bar.move_to(low_bar.get_top(), DOWN)
            bars_copy.target[-1].align_to(bars, DOWN)

            self.play(MoveToTarget(
                bars_copy, lag_ratio=0.5,
                run_time=np.sqrt(x + 1)
            ))

            # Resize lower bars
            for top_bar, low_bar in zip(bars_copy[:-1], bars[1:]):
                bottom = low_bar.get_bottom()
                low_bar.replace(
                    VGroup(low_bar, top_bar),
                    stretch=True
                )
                low_bar.move_to(bottom, DOWN)
            bars.add(bars_copy[-1])
            self.remove(bars_copy)
            self.add(bars)
