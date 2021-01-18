from manim_imports_ext import *


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
                num = Tex(str(nCk_residue))
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


class TriominoGrid(Scene):
    CONFIG = {
        "random_seed": 4,
        "n": 4,
    }

    def construct(self):
        n = self.n

        grid = VGroup(*[
            VGroup(*[
                Square()
                for x in range(2**n)
            ]).arrange(RIGHT, buff=0)
            for y in range(2**n)
        ]).arrange(UP, buff=0)
        for row in grid:
            for square in row:
                square.is_covered = False

        grid.set_fill(BLUE, 1)
        grid.set_stroke(WHITE, 1)

        covered_x = random.randint(0, 2**n - 1)
        covered_y = random.randint(0, 2**n - 1)
        covered = grid[covered_x][covered_y]
        covered.is_covered = True
        covered.set_fill(RED)

        grid.set_height(6)
        self.add(grid)

        self.triominos = VGroup()
        self.add(self.triominos)
        self.cover_grid(grid)
        colors = [
            BLUE_C,
            BLUE_E,
            BLUE_D,
            BLUE_B,
            MAROON_C,
            MAROON_E,
            MAROON_D,
            MAROON_B,
            YELLOW,
            GREY_BROWN,
            GREY_B,
            GREEN_C,
            GREEN_E,
            GREEN_D,
            GREEN_B,
        ]
        random.shuffle(colors)
        for triomino, color in zip(self.triominos, it.cycle(colors)):
            triomino.set_color(color)
            triomino.scale(0.95)

        self.play(ShowIncreasingSubsets(
            self.triominos,
            run_time=5
        ))

    def cover_grid(self, grid):
        N = len(grid)  # N = 2**n
        if N == 1:
            return
        q1 = VGroup(*[row[:N // 2] for row in grid[:N // 2]])
        q2 = VGroup(*[row[:N // 2] for row in grid[N // 2:]])
        q3 = VGroup(*[row[N // 2:] for row in grid[:N // 2]])
        q4 = VGroup(*[row[N // 2:] for row in grid[N // 2:]])
        quads = [q1, q2, q3, q4]

        for q in quads:
            squares = [
                square
                for row in q
                for square in row
            ]
            q.has_covered = any([s.is_covered for s in squares])
            corner_index = np.argmin([
                get_norm(s.get_center() - grid.get_center())
                for s in squares
            ])
            q.inner_corner = squares[corner_index]

        covered_quad_index = [q.has_covered for q in quads].index(True)
        covered_quad = quads[covered_quad_index]

        hugging_triomino = VGroup()
        for q in quads:
            if q is not covered_quad:
                hugging_triomino.add(q.inner_corner.copy())
                q.inner_corner.is_covered = True

        hugging_triomino.set_stroke(width=0)
        hugging_triomino.set_fill(random_color(), opacity=1.0)

        self.triominos.add(hugging_triomino)

        for q in quads:
            self.cover_grid(q)
