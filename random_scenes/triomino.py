from big_ol_pile_of_manim_imports import *


class GridTest(Scene):
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
            LIGHT_GREY,
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
