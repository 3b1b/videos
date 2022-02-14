from manim_imports_ext import *


class PowersOfTwo(Scene):
    def construct(self):
        max_n = 22
        self.colors = [
            [BLUE_B, BLUE_D, BLUE_C, GREY_BROWN][n % 4]
            for n in range(max_n)
        ]

        def update_group(group, alpha):
            n = int(interpolate(0, max_n, alpha))
            group.set_submobjects([
                self.get_label(n),
                self.get_dots(n),
            ])

        label = self.get_label(0)
        dots = self.get_dots(0)
        for n in range(1, max_n + 1):
            new_label = self.get_label(n)
            new_dots = self.get_dots(n)
            self.play(
                FadeTransform(label, new_label),
                FadeTransform(dots, new_dots[0]),
                FadeIn(new_dots[1]),
                run_time=0.4,
            )
            self.remove(dots)
            self.add(new_dots)
            self.wait(0.1)

            label = new_label
            dots = new_dots

        self.wait(2)

    def get_label(self, n):
        lhs = MTex("2^{", "10", "} =")
        exp = Integer(n)
        exp.match_height(lhs[1])
        exp.move_to(lhs[1], LEFT)
        lhs.replace_submobject(1, exp)
        rhs = Integer(2**n)
        rhs.next_to(lhs, RIGHT)
        rhs.shift((lhs[0].get_bottom() - rhs[0].get_bottom())[1] * UP)
        result = VGroup(lhs, rhs)
        result.center().to_edge(UP)
        # result.set_x(-1, LEFT)
        return result

    def get_dots(self, n, height=6):
        if n == 0:
            result = self.get_marginal_dots(0)
        else:
            old_dots = self.get_dots(n - 1)
            new_dots = self.get_marginal_dots(n - 1)
            result = Group(old_dots, new_dots)

        for sm in result.get_family():
            if isinstance(sm, DotCloud):
                sm.set_radius(0)

        if len(result) > 0:
            result[0].replace(result[1], stretch=True)

        radius = min(0.3 * height / (2**(int(np.ceil(n / 2)))), 0.1)

        buff = 1
        if n % 2 == 0:
            result.arrange(DOWN, buff=buff)
        else:
            result.arrange(RIGHT, buff=buff)

        # result.set_height(min(1 + n / 2, 6))
        result.set_width(min(1 + n / 2, 6))
        result.move_to(0.5 * DOWN)

        for sm in result.get_family():
            if isinstance(sm, DotCloud):
                sm.set_radius(radius)

        return result

    def get_marginal_dots(self, n):
        dots = DotCloud()
        rows = 2**(int(np.floor(n / 2)))
        cols = 2**(int(np.ceil(n / 2)))
        dots.to_grid(rows, cols)
        # dots.set_glow_factor(0.25)
        dots.set_color(self.colors[n])
        dots.set_width(min(1 + n / 2, 10))
        return dots
