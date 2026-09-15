from PIL import Image
import tempfile
import os
import math
import random
from manim_imports_ext import *
from scipy.spatial.transform import Slerp


class Tile(Rectangle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, fill_opacity=0.9, fill_color=BLUE, stroke_width=8, stroke_color=WHITE, **kwargs)
        self.round_corners(0.05)
        self.set_scale_stroke_with_zoom(True)


class Hole(VGroup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.background = Square(
            side_length=1,
            fill_opacity=1,
            fill_color="#444444",
            stroke_width=0
        )
        self.cross = Cross(self.background, stroke_width=2).scale(0.95)
        points = [ORIGIN, RIGHT, UR, UP]
        self.border = VGroup(*[
            Line(points[i], points[(i + 1) % 4], stroke_width=4, stroke_color=YELLOW).scale(1.06)
            for i in range(len(points))
        ]).match_width(self.cross).scale(1.06).move_to(self.cross)
        self.add(self.background, self.cross, self.border)
        self.set_scale_stroke_with_zoom(False)


class Grid(VGroup):
    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.vertical_lines = VGroup(*[Line(RIGHT * i, RIGHT * i + DOWN * n) for i in range(n + 1)])
        self.horizontal_lines = VGroup(*[Line(DOWN * i, DOWN * i + RIGHT * n) for i in range(n + 1)])
        self.lines = VGroup(self.vertical_lines, self.horizontal_lines).set_stroke(width=4, color=WHITE, opacity=0.7)
        self.lines.set_scale_stroke_with_zoom(True)

        self.background = Square(
            side_length=self.horizontal_lines.get_width(),
            fill_opacity=1,
            fill_color=GREY_D,
            stroke_width=0
        ).move_to(
            self.lines
        )
        self.tiles = VGroup()
        self.holes = VGroup()

        self.add(self.background, self.lines, self.tiles, self.holes)
        self.center()

    def position_at_coordinates(self, tile_or_hole, i, j):
        tile_or_hole.align_to(self.vertical_lines[i], LEFT).align_to(self.horizontal_lines[j], UP)

    def add_tile(self, width, height, i, j, *args, **kwargs):
        unit_size = self.get_width() / self.n
        tile = Tile(width, height, *args, **kwargs).scale(unit_size)
        self.position_at_coordinates(tile, i, j)
        self.tiles.add(tile)

    def add_hole(self, i, j, *args, **kwargs):
        unit_size = self.get_width() / self.n
        hole = Hole(*args, **kwargs).scale(unit_size)
        hole.pos = (i, j)
        self.position_at_coordinates(hole, i, j)
        self.holes.add(hole)

    def get_reasonable_tiling(self):
        occupied = [[False] * self.n for _ in range(self.n)]
        for hole in self.holes:
            occupied[hole.pos[1]][hole.pos[0]] = True

        def find_candidate_rectangles():
            heights = [0] * self.n
            candidates = []
            for row in range(self.n):
                for col in range(self.n):
                    heights[col] = 0 if occupied[row][col] else heights[col] + 1
                stack = []
                for col in range(self.n + 1):
                    h = heights[col] if col < self.n else 0
                    start = col
                    while stack and stack[-1][1] >= h:
                        idx, height = stack.pop()
                        area = height * (col - idx)
                        candidates.append((area, row - height + 1, idx, row, col - 1))
                        start = idx
                    stack.append((start, h))
            return candidates

        empty_remaining = self.n * self.n - self.n
        while empty_remaining > 0:
            candidates = find_candidate_rectangles()
            max_area = max(c[0] for c in candidates)
            good = [c for c in candidates if c[0] >= max_area * 0.75]
            area, top, left, bottom, right = random.choice(good)

            self.add_tile(right - left + 1, bottom - top + 1, left, top)
            for r in range(top, bottom + 1):
                for c in range(left, right + 1):
                    occupied[r][c] = True
            empty_remaining -= area


class OptimalGrid(Grid):
    def __init__(self, k, *args, **kwargs):
        self.k = k
        n = k * k
        super().__init__(n, *args, **kwargs)

        # Holes
        for i in range(n):
            self.add_hole(k - 1 - i // k + (i % k) * k, i)

        # Main tiles
        for i in range((k - 1) * (k - 1)):
            self.add_tile(k, k, k - 1 - i // (k - 1) + (i % (k - 1)) * k, i + i // (k - 1) + 1)
        self.main_tiles = VGroup(*self.tiles)

        # Upper-right tiles
        for j in range(1, k):
            self.add_tile(k, j, j * k, 0)
        self.ur_tiles = VGroup(*self.tiles[len(self.main_tiles):])

        # Lower-right tiles
        for j in range(1, k):
            self.add_tile(j, k, n - j, j * k)
        self.dr_tiles = VGroup(*self.tiles[len(self.main_tiles) + len(self.ur_tiles):])

        # Lower-left tiles
        for j in range(1, k):
            self.add_tile(k, j, n - (j + 1) * k, n - j)
        self.dl_tiles = VGroup(*self.tiles[len(self.main_tiles) + len(self.ur_tiles) + len(self.dr_tiles):])

        # Upper-left tiles
        for j in range(1, k):
            self.add_tile(j, k, 0, n - (j + 1) * k)
        self.ul_tiles = VGroup(*self.tiles[len(self.main_tiles) + len(self.ur_tiles) + len(self.dr_tiles) + len(self.dl_tiles):])


class RandomGrid(Grid):
    def __init__(self, n, *args, **kwargs):
        super().__init__(n, *args, **kwargs)

        hole_col = list(range(n))
        random.shuffle(hole_col)
        for row in range(n):
            self.add_hole(hole_col[row], row)


class OptimalArrangementMotivation(InteractiveScene):
    def construct(self):
        # Add a bunch of tiles
        self.camera.frame.save_state()
        self.camera.frame.scale(2)
        tiles = VGroup(
            Tile(4, 2),
            Tile(2, 3),
            Tile(4, 3),
            Tile(1, 5),
            Tile(3, 3)
        )
        tiles[0].shift(LEFT * 8 + UP * 4.5)
        tiles[1].shift(RIGHT * 8 + UP * 4.5)
        tiles[3].shift(LEFT * 8 + DOWN * 3.5)
        tiles[4].shift(RIGHT * 8 + DOWN * 4.5)

        for i, tile in enumerate(tiles):
            tile.w_val = tile.get_width()
            tile.h_val = tile.get_height()
            phase = random.uniform(0, 2 * math.pi)
            amplitude = math.radians(random.uniform(3, 6))
            frequency = random.uniform(1, 1.2)

            init_angle = amplitude * math.sin(phase)
            tile.angle_tracker = ValueTracker(init_angle)
            tile.current_angle = init_angle
            tile.rotate(init_angle)

            def make_tile_updater(p, amp, freq):
                def updater(m, dt):
                    t = self.time
                    target_angle = amp * math.sin(freq * t + p)
                    m.angle_tracker.set_value(target_angle)

                    d_theta = target_angle - m.current_angle
                    m.rotate(d_theta)
                    m.current_angle = target_angle
                return updater

            tile.add_updater(make_tile_updater(phase, amplitude, frequency))

        shuffled_tiles = list(tiles)
        random.shuffle(shuffled_tiles)

        self.play(
            AnimationGroup(
                *[GrowFromCenter(tile) for tile in shuffled_tiles],
                lag_ratio=0.25
            )
        )
        self.wait(3)

        # Add some holes sliding around the sides
        holes = VGroup()
        for tile in tiles:
            w_val = tile.w_val
            h_val = tile.h_val

            for side_idx in range(4):
                hole = Hole()
                hole.border[side_idx].set_color(WHITE)
                hole.phase = random.uniform(0, 2 * math.pi)
                hole.freq = random.uniform(0.8, 1.5)

                if side_idx == 0 or side_idx == 2:
                    L = max(0.1, (w_val - 1) / 2)
                else:
                    L = max(0.1, (h_val - 1) / 2)

                s = L * math.sin(hole.phase)
                if side_idx == 0:
                    local_pos = np.array([s, h_val / 2 + 0.5, 0])
                elif side_idx == 1:
                    local_pos = np.array([-w_val / 2 - 0.5, s, 0])
                elif side_idx == 2:
                    local_pos = np.array([s, -h_val / 2 - 0.5, 0])
                else:
                    local_pos = np.array([w_val / 2 + 0.5, s, 0])

                init_angle = tile.angle_tracker.get_value()
                world_pos = tile.get_center() + rotate_vector(local_pos, init_angle)

                hole.rotate(init_angle)
                hole.current_angle = init_angle
                hole.move_to(world_pos)

                def make_hole_updater(t_ref, s_idx, l_val, p_val, f_val, w_v, h_v):
                    def updater(h, dt):
                        t = self.time
                        s_t = l_val * math.sin(f_val * t + p_val)
                        if s_idx == 0:
                            loc = np.array([s_t, h_v / 2 + 0.5, 0])
                        elif s_idx == 1:
                            loc = np.array([-w_v / 2 - 0.5, s_t, 0])
                        elif s_idx == 2:
                            loc = np.array([s_t, -h_v / 2 - 0.5, 0])
                        else:
                            loc = np.array([w_v / 2 + 0.5, s_t, 0])

                        target_angle = t_ref.angle_tracker.get_value()
                        w_pos = t_ref.get_center() + rotate_vector(loc, target_angle)
                        d_theta = target_angle - h.current_angle
                        h.rotate(d_theta)
                        h.current_angle = target_angle
                        h.move_to(w_pos)
                    return updater

                hole.add_updater(make_hole_updater(tile, side_idx, L, hole.phase, hole.freq, w_val, h_val))
                holes.add(hole)

        shuffled_holes = list(holes)
        random.shuffle(shuffled_holes)
        self.play(AnimationGroup(*[FadeIn(hole) for hole in shuffled_holes], lag_ratio=0.05))
        self.wait(9)

        # Focus on one of the tiles
        tiles[2].clear_updaters()

        def make_tracker_listener():
            def updater(m, dt):
                target_angle = m.angle_tracker.get_value()
                d_theta = target_angle - m.current_angle
                m.rotate(d_theta)
                m.current_angle = target_angle
            return updater

        tiles[2].add_updater(make_tracker_listener())

        self.play(
            self.camera.frame.animate.scale(0.87),
            VGroup(tiles[:2], tiles[3:], holes[:8], holes[12:]).animate.set_opacity(0),
            tiles[2].angle_tracker.animate.set_value(0), run_time=2)
        self.remove(tiles[:2], tiles[3:], holes[:8], holes[12:])
        tile = tiles[2]
        holes = holes[8:12]

        # Let the holes naturally slide into a cleaned up position
        damp_tracker = ValueTracker(1.0)

        for side_idx, hole in enumerate(holes):
            w_val = tile.w_val
            h_val = tile.h_val

            if side_idx == 0 or side_idx == 2:
                L = max(0.1, (w_val - 1) / 2)
            else:
                L = max(0.1, (h_val - 1) / 2)

            hole.clear_updaters()

            def make_dampening_hole_updater(t_ref, s_idx, l_val, p_val, f_val, w_v, h_v):
                def updater(h, dt):
                    t = self.time
                    scale = damp_tracker.get_value()
                    s_t = scale * l_val * math.sin(f_val * t + p_val)

                    if s_idx == 0:
                        loc = np.array([s_t, h_v / 2 + 0.5, 0])
                    elif s_idx == 1:
                        loc = np.array([-w_v / 2 - 0.5, s_t, 0])
                    elif s_idx == 2:
                        loc = np.array([s_t, -h_v / 2 - 0.5, 0])
                    else:
                        loc = np.array([w_v / 2 + 0.5, s_t, 0])

                    target_angle = t_ref.angle_tracker.get_value()
                    w_pos = t_ref.get_center() + rotate_vector(loc, target_angle)
                    d_theta = target_angle - h.current_angle
                    h.rotate(d_theta)
                    h.current_angle = target_angle
                    h.move_to(w_pos)
                return updater

            hole.add_updater(make_dampening_hole_updater(tile, side_idx, L, hole.phase, hole.freq, w_val, h_val))

        self.play(damp_tracker.animate.set_value(0), run_time=3)
        tile.clear_updaters()
        holes.clear_updaters()

        # Add tiles above and below the hole
        right_hole = holes[3]
        right_hole.add_updater(lambda m: self.bring_to_front(m))
        tile_above = Tile(3, 5).align_to(right_hole.get_corner(UL), DL)
        tile_below = Tile(4, 5).align_to(right_hole.get_corner(DL), UL)
        self.add(tile_above, tile_below, holes)
        self.play(
            FadeIn(tile_above, shift=DOWN),
            FadeIn(tile_below, shift=UP),
            holes[:3].animate.set_opacity(0.2),
            VGroup(right_hole.border[0], right_hole.border[2]).animate.set_color(WHITE), run_time=2)
        self.wait(2)

        # Add holes to those tiles
        inner_holes_above = VGroup(*[
            Hole().next_to(tile_above, [LEFT, UP, RIGHT][i], buff=0).shift(UP * 0.8 if i == 0 else 0)
            for i in range(3)
        ])
        for i, hole in enumerate(inner_holes_above):
            for j in range(len(hole.border)):
                if (-j + 1) % 4 == i:
                    hole.border[j].set_color(WHITE)
        inner_holes_below = VGroup(*[
            Hole().next_to(tile_below, [LEFT, RIGHT, DOWN][i], buff=0).shift(DOWN * 0.8 if i == 0 else 0)
            for i in range(3)
        ])
        for i, hole in enumerate(inner_holes_below):
            for j in range(len(hole.border)):
                if (i == 0 and j == 1 or i == 1 and j == 3 or i == 2 and j == 2):
                    hole.border[j].set_color(WHITE)
        hole_above = inner_holes_above[0]
        hole_below = inner_holes_below[0]
        shuffled_inner_holes = list(inner_holes_above) + list(inner_holes_below)
        random.shuffle(shuffled_inner_holes)
        self.play(AnimationGroup(*[FadeIn(hole) for hole in shuffled_inner_holes], lag_ratio=0.2))
        self.wait(2)
        dashed_line = DashedLine(hole_below, hole_above, dash_length=0.2, stroke_width=6).set_color(PURE_RED)
        self.play(
            ShowCreation(dashed_line),
            inner_holes_above[1:].animate.set_opacity(0.2),
            inner_holes_below[1:].animate.set_opacity(0.2)
        )
        self.wait(3)

        # Slide the hole up and down the side of the tile
        extra_tiles_and_holes_group = VGroup(tile_above, tile_below, inner_holes_above, inner_holes_below, dashed_line)
        right_hole.clear_updaters()
        self.play(VGroup(extra_tiles_and_holes_group, right_hole).animate.shift(UP * 0.3), run_time=3)
        self.play(VGroup(extra_tiles_and_holes_group, right_hole).animate.shift(DOWN * 0.6), run_time=3)
        self.play(FadeOut(VGroup(holes[:3], inner_holes_above[1:], inner_holes_below[1:])))
        self.wait(1)
        holes_group = VGroup(right_hole, hole_above, hole_below, dashed_line)
        holes_group.generate_target()
        holes_group.target.shift(UP * (tile.get_top()[1] - right_hole.get_top()[1]))
        tile_above.generate_target()
        tile_above.target.align_to(holes_group.target[0].get_top(), DOWN)
        tile_below.generate_target()
        tile_below.target.align_to(holes_group.target[0].get_bottom(), UP)
        self.play(
            MoveToTarget(holes_group),
            MoveToTarget(tile_above),
            MoveToTarget(tile_below),
            self.camera.frame.animate.move_to(holes_group.target[0]), run_time=3)
        right_hole.add_updater(lambda m: self.bring_to_front(m))
        self.play(
            FadeOut(dashed_line, shift=LEFT * 1.7),
            VGroup(tile_above, hole_above).animate.shift(LEFT * (tile_above.get_right()[0] - right_hole.get_right()[0])), run_time=2.5)
        self.wait(2)
        new_tile = Tile(5, 3).align_to(right_hole.get_corner(DR), DL)
        self.play(FadeIn(new_tile, shift=LEFT), right_hole.border[1].animate.set_color(WHITE))
        right_hole.clear_updaters()

        # Add some extra holes around the outer tiles
        outer_tiles = VGroup(tile, tile_above, new_tile, tile_below)
        outer_tiles.add_updater(lambda m: self.bring_to_back(m))
        middle_hole = right_hole
        inner_holes = VGroup(*[
            Hole().align_to(outer_tile.get_corner(direction1), direction2)
            for outer_tile, direction1, direction2 in zip(outer_tiles, [DR, DL, UL, UR], [UR, DR, DL, UL])
        ])
        for i in range(len(inner_holes)):
            VGroup(inner_holes[i].border[(-i + 1) % 4], inner_holes[i].border[(-i + 2) % 4]).set_color(WHITE)
        self.play(
            ReplacementTransform(hole_below, inner_holes[0]),
            ReplacementTransform(hole_above, inner_holes[1]),
            AnimationGroup(*[
                FadeIn(hole, shift=direction)
                for hole, direction in zip(inner_holes[2:], [DL, UL])
            ], lag_ratio=0.1), run_time=1.5)
        outer_holes = VGroup(*[
            Hole().align_to(outer_tile.get_corner(direction1), direction2)
            for outer_tile, direction1, direction2 in zip(outer_tiles, [DL, UL, UR, DR], [DR, DL, UL, UR])
        ])
        for i in range(len(inner_holes)):
            for j in range(4):
                if (-j + 1) % 4 == i:
                    outer_holes[i].border[j].set_color(WHITE)
        self.play(AnimationGroup(*[FadeIn(hole) for hole in outer_holes], lag_ratio=0.1))
        self.wait(0.5)
        checkmarks = VGroup(*[
            Checkmark().scale(1.5).set_color(PURE_GREEN).move_to(hole)
            for hole in [middle_hole] + list(inner_holes) + list(outer_holes)
        ])
        self.play(AnimationGroup(*[GrowFromCenter(checkmark) for checkmark in checkmarks], lag_ratio=0.2))
        self.wait(2)
        self.play(FadeOut(checkmarks))

        # Change the tiles into squares
        square_tiles = VGroup(*[Tile(3, 3) for _ in range(4)])
        for square_tile, outer_tile, direction in zip(square_tiles, outer_tiles, [UR, DR, DL, UL]):
            square_tile.align_to(outer_tile, direction)
        for hole, square_tile, direction1, direction2 in zip(inner_holes, square_tiles, [DR, DL, UL, UR], [UR, DR, DL, UL]):
            hole.generate_target()
            hole.target.align_to(square_tile.get_corner(direction1), direction2)
        for hole, square_tile, direction1, direction2 in zip(outer_holes, square_tiles, [DL, UL, UR, DR], [DR, DL, UL, UR]):
            hole.generate_target()
            hole.target.align_to(square_tile.get_corner(direction1), direction2)
        middle_hole.add_updater(lambda m: self.bring_to_front(m))
        self.play(
            AnimationGroup(*[
                ReplacementTransform(outer_tile, square_tile)
                for outer_tile, square_tile in zip(outer_tiles, square_tiles)
            ]),
            AnimationGroup(*[
                MoveToTarget(hole)
                for hole in inner_holes
            ]),
            AnimationGroup(*[
                MoveToTarget(hole)
                for hole in outer_holes
            ]), run_time=2)
        middle_hole.clear_updaters()
        self.clear()
        self.add(square_tiles, middle_hole, inner_holes, outer_holes)
        self.wait(2)

        # Focus on one of the puzzle pieces
        puzzle_piece = VGroup(square_tiles[2], middle_hole, inner_holes[2], outer_holes[2], inner_holes[3])

        time_start = self.time
        single_phase = 0
        single_amplitude = math.radians(4.5)
        single_frequency = 1.1

        single_init_angle = single_amplitude * math.sin(single_phase)
        puzzle_piece.angle_tracker = ValueTracker(single_init_angle)
        puzzle_piece.current_angle = single_init_angle
        puzzle_piece.rotate(single_init_angle)

        def make_piece_updater(p, amp, freq):
            def updater(m, dt):
                t = self.time - time_start
                target_angle = amp * math.sin(freq * t + p)
                m.angle_tracker.set_value(target_angle)

                d_theta = target_angle - m.current_angle
                m.rotate(d_theta)
                m.current_angle = target_angle

                for i in range(len(m[1:])):
                    for j in range(4):
                        if (-j + 1) % 4 != i:
                            m[1:][i].border[j].set_color(interpolate_color(m[1:][i].border[j].get_color(), YELLOW, 0.03))

            return updater

        puzzle_piece.add_updater(make_piece_updater(single_phase, single_amplitude, single_frequency))
        self.play(
            FadeOut(
                VGroup(
                    square_tiles[:2], square_tiles[3],
                    inner_holes[:2],
                    outer_holes[:2], outer_holes[3]
                ),
                shift=DL * 2
            ),
            self.camera.frame.animate.scale(0.9).move_to(puzzle_piece),
            puzzle_piece.animate.shift(0), run_time=2)

        # Show floating copies of the puzzle piece
        self.camera.frame.center()
        puzzle_piece.center()

        puzzle_pieces = VGroup(puzzle_piece, *[puzzle_piece.copy() for _ in range(4)])
        puzzle_pieces[1].move_to(LEFT * 7.5 + UP * 4.5)
        puzzle_pieces[2].move_to(RIGHT * 7.5 + UP * 4.5)
        puzzle_pieces[3].move_to(LEFT * 7.5 + DOWN * 4.5)
        puzzle_pieces[4].move_to(RIGHT * 7.5 + DOWN * 4.5)

        for i in range(1, len(puzzle_pieces)):
            piece = puzzle_pieces[i]
            phase = random.uniform(0, 2 * math.pi)
            amplitude = math.radians(random.uniform(3, 6))
            frequency = random.uniform(1, 1.2)

            init_angle = amplitude * math.sin(phase)
            piece.angle_tracker = ValueTracker(init_angle)
            piece.current_angle = init_angle
            piece.rotate(init_angle)

            piece.add_updater(make_piece_updater(phase, amplitude, frequency))

        shuffled_pieces = list(puzzle_pieces)
        random.shuffle(shuffled_pieces)

        self.play(
            self.camera.frame.animate.scale(1.35),
            AnimationGroup(
                *[GrowFromCenter(piece) for piece in shuffled_pieces if piece != puzzle_piece],
                lag_ratio=0.25
            ),
            run_time=2.5
        )
        self.wait(10)


class WindmillTilings(InteractiveScene):
    def construct(self):
        # Add a grid with k = 5
        min_k = 5
        max_k = 45

        grid = OptimalGrid(min_k).align_to(ORIGIN, DL)
        for hole in grid.holes:
            hole.border.set_color(WHITE)
        self.add(grid)
        self.camera.frame.move_to(grid).set_height(grid.get_height() * 1.4)
        self.camera.frame.save_state()
        self.wait(2)

        # Change it to k = 4
        new_grid = OptimalGrid(4).align_to(ORIGIN, DL)
        for hole in new_grid.holes:
            hole.border.set_color(WHITE)
        grid.holes.set_z_index(100)
        self.add(grid)
        self.play(
            self.camera.frame.animate(run_time=1).move_to(new_grid).set_height(new_grid.get_height() * 1.4),
            ReplacementTransform(grid.background, new_grid.background),
            ReplacementTransform(grid.lines, new_grid.lines),
            ReplacementTransform(grid.main_tiles, new_grid.main_tiles),
            ReplacementTransform(grid.ur_tiles, new_grid.ur_tiles),
            ReplacementTransform(grid.dr_tiles, new_grid.dr_tiles),
            ReplacementTransform(grid.dl_tiles, new_grid.dl_tiles),
            ReplacementTransform(grid.ul_tiles, new_grid.ul_tiles),
            ReplacementTransform(grid.holes, new_grid.holes)
        )
        grid = new_grid
        self.wait(2)

        # Show the new dimensions
        x_length_label = Tex("16", font_size=180).next_to(grid, DOWN, buff=1)
        y_length_label = x_length_label.copy().next_to(grid, LEFT, buff=1)
        brace1 = Brace(grid, DOWN)
        brace2 = Brace(grid, LEFT)
        self.play(GrowFromEdge(brace1, UP), GrowFromEdge(brace2, RIGHT), FadeIn(VGroup(x_length_label, y_length_label)))
        self.wait(2)

        # Count the holes
        circles = VGroup(*[
            Circle(radius=0.8, stroke_width=3, stroke_color=PURE_GREEN).move_to(hole)
            for hole in grid.holes
        ])
        hole_numbers = VGroup(*[
            Tex(str(i + 1), font_size=90).next_to(hole, UP, buff=0.7)
            for i, hole in enumerate(grid.holes)
        ])
        grid.save_state()
        self.play(
            AnimationGroup(
                AnimationGroup(
                    grid.background.animate.set_opacity(0.2),
                    grid.lines.animate.set_opacity(0.2),
                    grid.tiles.animate.set_opacity(0.2),
                    grid.holes.animate.shift(0)
                ),
                AnimationGroup(*[
                    AnimationGroup(
                        ShowCreation(circle),
                        FadeIn(num, shift=UP * 0.4), lag_ratio=0.1)
                    for circle, num in zip(circles, hole_numbers)
                ], lag_ratio=0.05), lag_ratio=0.1)
        )
        self.wait(2)
        self.play(
            FadeOut(VGroup(x_length_label, y_length_label, brace1, brace2, circles, hole_numbers)),
            grid.animate.restore()
        )

        # Change it to k = 3
        new_grid = OptimalGrid(3).align_to(ORIGIN, DL)
        for hole in new_grid.holes:
            hole.border.set_color(WHITE)
        grid.holes.set_z_index(100)
        self.add(grid)
        self.play(
            self.camera.frame.animate(run_time=1).move_to(new_grid).set_height(new_grid.get_height() * 1.4),
            ReplacementTransform(grid.background, new_grid.background),
            ReplacementTransform(grid.lines, new_grid.lines),
            ReplacementTransform(grid.main_tiles, new_grid.main_tiles),
            ReplacementTransform(grid.ur_tiles, new_grid.ur_tiles),
            ReplacementTransform(grid.dr_tiles, new_grid.dr_tiles),
            ReplacementTransform(grid.dl_tiles, new_grid.dl_tiles),
            ReplacementTransform(grid.ul_tiles, new_grid.ul_tiles),
            ReplacementTransform(grid.holes, new_grid.holes)
        )
        grid = new_grid
        self.wait(2)

        # Show the new dimensions again
        x_length_label = Tex("9", font_size=100).next_to(grid, DOWN, buff=0.7)
        y_length_label = x_length_label.copy().next_to(grid, LEFT, buff=0.7)
        brace1 = Brace(grid, DOWN)
        brace2 = Brace(grid, LEFT)
        self.play(GrowFromEdge(brace1, UP), GrowFromEdge(brace2, RIGHT), FadeIn(VGroup(x_length_label, y_length_label)))
        self.wait(2)
        self.play(FadeOut(VGroup(x_length_label, y_length_label, brace1, brace2)))

        # Generalize
        tile = grid.tiles[1]
        grid.save_state()
        k_label_1 = Tex("k", font_size=100).next_to(tile, DOWN, buff=0.7)
        k_label_2 = k_label_1.copy().next_to(tile, LEFT, buff=0.7)
        brace1 = Brace(tile, DOWN)
        brace2 = Brace(tile, LEFT)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    grid.background.animate.set_opacity(0.1),
                    grid.lines.animate.set_opacity(0.1),
                    *[t.animate.set_opacity(0.1 if t != tile else t.get_opacity()) for t in grid.tiles],
                    grid.holes.animate.set_opacity(0.1)
                ),
                AnimationGroup(
                    GrowFromEdge(brace1, UP),
                    GrowFromEdge(brace2, RIGHT),
                    FadeIn(VGroup(k_label_1, k_label_2))
                ), lag_ratio=0.4)
        )

        # Add a slider
        k_tracker = ValueTracker(3)
        x_range = [0, 50, 1]
        k_slider = NumberLine(
            x_range=x_range,
            width=3,
            include_numbers=True,
            numbers_to_exclude=[x for x in range(x_range[0], x_range[1], x_range[2]) if x % 10 != 0],
            longer_tick_multiple=10
        )
        for i, tick in enumerate(k_slider.ticks):
            if i % 10 != 0:
                tick.scale(0.4).set_stroke(width=1.5)
        k_display = Tex("k = 2").next_to(k_slider, UP, buff=0.7)
        k_value = k_display.make_number_changeable("2")
        k_value.add_updater(lambda m: m.set_value(round(k_tracker.get_value())))
        k_triangle = Triangle(fill_opacity=1, fill_color=TEAL, stroke_width=0).stretch(1.5, 1).set_width(0.2).rotate(PI)
        k_triangle.align_to(k_slider[0].get_center(), DOWN)
        k_triangle.add_updater(lambda m: m.set_x(k_slider.n2p(round(k_tracker.get_value()))[0]))
        rect = BackgroundRectangle(VGroup(k_slider, k_display, k_triangle), buff=0.1).round_corners(0.2)
        k_slider_group = VGroup(rect, k_slider, k_display, k_triangle)
        k_slider_group.fix_in_frame().set_anti_alias_width(0).to_corner(UL, buff=0.2).set_scale_stroke_with_zoom(True)
        self.play(FadeIn(k_slider_group))
        self.play(
            grid.animate.restore(),
            FadeOut(VGroup(k_label_1, k_label_2, brace1, brace2))
        )

        # Switch to dynamic updating
        def update_grid(grid):
            new_k = round(k_tracker.get_value())
            if new_k - grid.k != 0:
                grid.k = new_k
                grid.become(OptimalGrid(new_k).align_to(ORIGIN, DL))
                for hole in grid.holes:
                    hole.border.set_color(WHITE)

                for hole in grid.holes:
                    hole.border.set_stroke(width=5 * new_k**(-1 / 3), color=interpolate_color(WHITE, RED, min(1, (new_k - 3) / 7)))
                for tile in grid.tiles:
                    tile.add_updater(lambda m: m.set_stroke(width=1.5 * new_k**1.5))
        grid.add_updater(update_grid)

        # Increase k to 10
        next_k = 10
        self.play(
            k_tracker.animate(run_time=5).set_value(next_k),
            self.camera.frame.animate(run_time=5).move_to([next_k * next_k * 0.5, next_k * next_k * 0.5, 0]).set_height(1.4 * next_k * next_k)
        )
        grid.suspend_updating()

        # Show the new size: k^2 x k^2
        x_length_label = Tex(R"k^2", font_size=900).next_to(grid, DOWN, buff=5)
        y_length_label = x_length_label.copy().next_to(grid, LEFT, buff=5)
        self.play(Write(x_length_label), Write(y_length_label), run_time=2)
        self.wait(2)
        self.play(FadeOut(VGroup(brace1, brace2, x_length_label, y_length_label)))

        # Increase k to 45
        grid.resume_updating()
        self.play(
            k_tracker.animate(run_time=5).set_value(max_k),
            self.camera.frame.animate(run_time=5).reorient(-13, 40, 0, (654.42, 252.66, 461.53), 1315.09)
        )
        grid.suspend_updating()
        self.wait(2)

        # Show the new size
        x_length_label = Tex(R"2025\\=45^2", font_size=12000).next_to(grid, DOWN, buff=50)
        y_length_label = x_length_label.copy().next_to(grid, LEFT, buff=60)
        brace1 = Brace(grid, DOWN)
        brace2 = Brace(grid, LEFT)
        self.play(GrowFromEdge(brace1, UP), GrowFromEdge(brace2, RIGHT), Write(x_length_label["2025"]), Write(y_length_label["2025"]))
        self.wait(2)
        self.play(
            FadeIn(
                x_length_label["=45^2"].set_color(
                    YELLOW
                ).next_to(
                    x_length_label["2025"], RIGHT, buff=30
                ).align_to(
                    x_length_label["2025"], DOWN
                )
            ),
            FadeIn(
                y_length_label["=45^2"].set_color(
                    YELLOW
                ).next_to(
                    y_length_label["2025"], DOWN, buff=40
                )
            )
        )
        self.wait(2)
        self.play(FadeOut(VGroup(x_length_label, y_length_label, brace1, brace2)))

        # Pan the camera around
        grid.save_state()
        self.add(k_slider_group)
        grid.holes.set_scale_stroke_with_zoom(False)
        self.play(
            self.camera.frame.animate.reorient(7, 46, 0, (1110.17, 171.23, 372.81), 1109.67), run_time=10)
        self.play(
            grid.tiles.animate.set_stroke(width=40),
            grid.holes.animate.set_stroke(width=0.5),
            self.camera.frame.animate.reorient(-24, 62, 0, (214.22, 294.13, -91.55), 379.46), run_time=10)
        self.play(
            # grid.animate.restore(),
            grid.tiles.animate.set_stroke(width=80),
            grid.holes.animate.set_stroke(width=4),
            self.camera.frame.animate.reorient(-27, 65, 0, (631.07, 270.25, 216.58), 180.95), run_time=10)

        # Reset the camera to the original position and show the labels for k
        tile = grid.tiles[1596]
        k_label_1 = Tex("k", font_size=5000).next_to(tile, DOWN, buff=20)
        k_label_2 = k_label_1.copy().next_to(tile, LEFT, buff=20)

        x_length_label = Tex(R"k^2", font_size=12000).next_to(grid, DOWN, buff=50)
        y_length_label = x_length_label.copy().next_to(grid, LEFT, buff=50)
        self.add(x_length_label, y_length_label)

        k_slider_group.set_z_index(1000)
        self.add(k_slider_group)
        self.play(
            AnimationGroup(
                self.camera.frame.animate(run_time=10).reorient(-18, 58, 0, (973.79, 650.28, -62.57), 1754.60),
                AnimationGroup(
                    AnimationGroup(
                        grid.background.animate.set_opacity(0.1),
                        grid.lines.animate.set_opacity(0.1),
                        *[t.animate.set_opacity(0.1 if t != tile else t.get_opacity()) for t in grid.tiles],
                        grid.holes.animate.set_opacity(0.1).set_stroke(width=0.5)
                    ),
                    FadeIn(VGroup(k_label_1, k_label_2)), lag_ratio=0.4, run_time=2.5), lag_ratio=0.15)
        )
        grid.holes.set_scale_stroke_with_zoom(True)

        # Set k back to 5
        self.play(
            grid.animate.restore(),
            FadeOut(VGroup(k_label_1, k_label_2, x_length_label, y_length_label))
        )
        grid.resume_updating()
        self.play(
            k_tracker.animate(run_time=2).set_value(5),
            self.camera.frame.animate(run_time=2).restore()
        )
        grid.suspend_updating()
        self.wait(1)

        # Count the number of square tiles
        new_grid = OptimalGrid(k=5).match_width(grid).move_to(grid)
        for hole in new_grid.holes:
            hole.border.set_color(WHITE)
        self.remove(grid)
        grid = new_grid
        update_grid(grid)
        self.add(grid)
        main_tiles = grid.main_tiles
        edge_tiles = VGroup(
            *grid.ul_tiles,
            *grid.ur_tiles,
            *grid.dr_tiles,
            *grid.dl_tiles
        )
        main_tile_numbers = VGroup(*[
            Tex(str(i + 1), font_size=150).set_color(BLACK).move_to(tile)
            for i, tile in enumerate(main_tiles)
        ])
        grid.save_state()
        holes = grid.holes
        holes.add_updater(lambda m: self.bring_to_front(m))
        self.play(
            VGroup(grid.lines, *[tile for tile in edge_tiles]).animate.set_opacity(0.1),
            AnimationGroup(*[
                AnimationGroup(
                    tile.animate(rate_func=there_and_back).set_color(YELLOW).scale(1.1).set_fill(opacity=0.5),
                    GrowFromCenter(num, run_time=0.8)
                )
                for tile, num in zip(main_tiles, main_tile_numbers)
            ], lag_ratio=0.1)
        )
        holes.suspend_updating()
        self.wait(2)

        # Generalize
        self.play(
            AnimationGroup(*[
                num.animate.become(
                    Dot(radius=0.2).set_color(BLACK).move_to(tile) if i < 16 - 1 - 3 else
                    Tex("(k - 1)^2", font_size=120).set_color(BLACK).move_to(tile)
                )
                for i, (tile, num) in enumerate(zip(main_tiles[3:], main_tile_numbers[3:]))
            ])
        )
        self.wait(1)

        # Save the value (k - 1)^2
        main_tile_count = main_tile_numbers[-1].copy().scale(
            1.9
        ).align_to(
            grid, UP
        ).set_x(
            0.5 * (grid.get_right()[0] + self.camera.frame.get_right()[0])
        ).set_color(
            TEAL_A
        )
        self.play(TransformFromCopy(main_tile_numbers[-1], main_tile_count, path_arc=-PI * 0.2), run_time=1.5)
        self.wait(1)

        # Switch focus to the tiles around the edges
        holes.resume_updating()
        self.play(
            grid.animate.restore(),
            main_tiles.animate.set_opacity(0.5),
            FadeOut(main_tile_numbers), run_time=2)
        holes.suspend_updating()
        self.wait(2)

        # Count the edge tiles
        edge_tile_numbers = VGroup(*[
            Tex(str(i % 4 + 1), font_size=100).set_color(BLACK).move_to(tile)
            for i, tile in enumerate(edge_tiles)
        ])
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    tile.animate(rate_func=there_and_back).set_color(YELLOW).scale(1.1).set_fill(opacity=0.5),
                    GrowFromCenter(num, run_time=0.8)
                )
                for tile, num in zip(edge_tiles, edge_tile_numbers)
            ], lag_ratio=0.2)
        )
        self.wait(2)

        # Generalize the edge tile count
        self.play(
            AnimationGroup(*[
                num.animate.become(
                    Tex(R"\cdots", font_size=100).set_color(BLACK).move_to(tile) if i % 4 == 2 else
                    Tex("k - 1", font_size=100).set_color(BLACK).move_to(tile) if i % 4 == 3 else
                    num
                )
                for i, (tile, num) in enumerate(zip(edge_tiles, edge_tile_numbers))
            ])
        )
        self.wait(1)

        # Save the value 4(k - 1)
        edge_tile_count = Tex(
            R"+4(k - 1)"
        )
        edge_tile_count[1:].set_color(TEAL_D)
        edge_tile_count.scale(
            main_tile_count[0].get_height() / edge_tile_count[2].get_height()
        ).next_to(
            main_tile_count, DOWN, buff=0.6
        ).align_to(
            main_tile_count[-2], RIGHT
        )
        edge_tile_count[0].shift(LEFT * 0.16)
        k_minus_1_copies = VGroup(*[edge_tile_count[3:-1].copy() for _ in range(4)])
        self.play(
            AnimationGroup(*[
                TransformMatchingShapes(
                    edge_tile_numbers[4 * i + 3].copy(),
                    k_minus_1_copies[i],
                    path_arc=-PI * 0.2
                )
                for i in range(4)
            ]), run_time=1.5)
        self.play(FadeIn(VGroup(edge_tile_count[:3], edge_tile_count[-1])))
        self.play(FadeOut(k_minus_1_copies[1:]))
        self.remove(k_minus_1_copies)
        self.add(edge_tile_count)
        self.wait(1)

        # Focus on the formula
        formula_group = VGroup(main_tile_count, edge_tile_count)
        self.clear()
        self.add(k_slider_group, grid, edge_tile_numbers, formula_group)
        formula_group.generate_target()
        formula_group.target.arrange(buff=0.4)
        formula_group.target.match_y(self.camera.frame).shift(UP * 4).align_to(formula_group, RIGHT).shift(LEFT)
        formula_group.target[1].align_to(formula_group.target[0], DOWN)
        self.play(
            FadeOut(k_slider_group, shift=LEFT * 3),
            VGroup(grid, edge_tile_numbers).animate.shift(LEFT * 14),
            MoveToTarget(formula_group, path_arc=PI * 0.2), run_time=2)

        # Expand it
        expanded_version_intermediate = Tex("= k^2 - 2k + 1 + 4k - 4")
        expanded_version_intermediate.scale(
            formula_group[0][1].get_height() / expanded_version_intermediate[1].get_height()
        ).next_to(
            formula_group, DOWN, buff=0.8
        )
        expanded_version_intermediate[:8].align_to(formula_group[0], RIGHT).set_color(TEAL_A)
        expanded_version_intermediate[0].set_color(WHITE)
        expanded_version_intermediate[8].match_x(formula_group[1][0])
        expanded_version_intermediate[9:].align_to(formula_group[1][1:], LEFT).set_color(TEAL_D)
        self.play(
            AnimationGroup(
                FadeIn(expanded_version_intermediate[0]),
                TransformMatchingShapes(formula_group[0].copy(), expanded_version_intermediate[1:8], run_time=1.2),
                TransformMatchingShapes(formula_group[1][0].copy(), expanded_version_intermediate[8], run_time=1),
                TransformMatchingShapes(formula_group[1][1:].copy(), expanded_version_intermediate[9:], run_time=1.2), lag_ratio=0.2)
        )
        self.wait(1)

        # Simplify
        expanded_version = Tex("= k^2 + 2k - 3")
        expanded_version.match_height(
            expanded_version_intermediate
        ).next_to(
            expanded_version_intermediate, DOWN, buff=0.8
        ).align_to(
            expanded_version_intermediate, LEFT
        )
        self.play(
            AnimationGroup(
                TransformMatchingShapes(expanded_version_intermediate[:3].copy(), expanded_version[:3], run_time=1.2),
                TransformMatchingShapes(
                    VGroup(expanded_version_intermediate[3:6], expanded_version_intermediate[9:11]).copy(),
                    expanded_version[3:6], run_time=1.2),
                TransformMatchingShapes(
                    VGroup(expanded_version_intermediate[6:8], expanded_version_intermediate[11:13]).copy(),
                    expanded_version[6:8], run_time=1.2), lag_ratio=0.2)
        )

        # Circle the final count
        final_count = expanded_version[1:]
        rect = SurroundingRectangle(final_count, buff=0.4, fill_opacity=0, stroke_width=3, stroke_color=YELLOW)
        self.play(ShowCreation(rect, run_time=2), FadeOut(edge_tile_numbers), grid.animate.restore().move_to(grid))
        self.wait(2)

        # Compare the conjectured optimal arrangement with other random arrangements
        grid.generate_target()
        final_count.generate_target()
        VGroup(final_count.target, grid.target).arrange(DOWN, buff=2.5).match_x(grid).match_y(self.camera.frame)
        edge_tile_numbers.set_z_index(100)
        self.play(
            AnimationGroup(
                FadeOut(VGroup(formula_group, expanded_version_intermediate, expanded_version[0], rect)),
                AnimationGroup(
                    MoveToTarget(grid, run_time=1),
                    MoveToTarget(final_count, path_arc=PI * 0.4, run_time=1.5)
                ), lag_ratio=0.3)
        )

        num_iters = 100
        for i in range(num_iters):
            other_grid = RandomGrid(int(k_tracker.get_value()**2)).match_y(grid).set_x(2 * self.camera.frame.get_x() - grid.get_x())
            for hole in other_grid.holes:
                hole.border.set_stroke(color=WHITE)
            self.add(other_grid)
            if i == 0:
                qms = TexText("???").match_height(final_count).match_y(final_count).match_x(other_grid)
                self.play(FadeIn(VGroup(other_grid, qms)), run_time=0.4)
            elif i == 26:
                arrow1 = Arrow(ORIGIN, RIGHT * 4, thickness=17).set_color(YELLOW).next_to(final_count, LEFT, buff=1)
                arrow2 = arrow1.copy().rotate(PI).next_to(final_count, RIGHT, buff=1)
                self.play(FadeIn(arrow1, shift=RIGHT), FadeIn(arrow2, shift=LEFT))
            else:
                self.wait(0.4)
            if i < num_iters - 1:
                self.remove(other_grid)


class RandomGrids(InteractiveScene):
    def construct(self):
        # Show an animation cyclying through many random grids
        num_iters = 900
        for i in range(num_iters):
            other_grid = RandomGrid(25)
            other_grid.get_reasonable_tiling()
            self.camera.frame.set_height(other_grid.get_height() * 1.1)
            for hole in other_grid.holes:
                hole.border.set_stroke(color=WHITE)
            self.add(other_grid)
            self.wait(0.4)
            if i < num_iters - 1:
                self.remove(other_grid)


class RandomGridThumbnailTest(InteractiveScene):
    def construct(self):
        # Show a big random grid
        self.camera.frame.reorient(0, 0, 0, (-0.39, 3.47, 0.00), 38.34)
        grid = RandomGrid(70)
        grid.get_reasonable_tiling()
        self.add(grid)
        for hole in grid.holes:
            self.add(GlowDot(radius=3).move_to(hole).set_opacity(0.6))
            hole.border.set_stroke(color=WHITE)


class TexturedSphereExample(InteractiveScene):
    def construct(self):
        # Put the grid on a sphere
        self.camera.frame.reorient(55, 71, 0, (-0.02, -0.00, 0.87), 0.32)
        sphere = Sphere(
            radius=1,
            resolution=(70, 70),
            v_range=(PI * 0.9, PI)
        )
        textured_sphere = TexturedSurface(sphere, "RandomGrid.png")
        self.add(textured_sphere)


class ErdosSzekeres(InteractiveScene):
    def construct(self):
        # Add a grid
        n = 9
        grid = Grid(n).set_width(6)
        self.add(grid)
        hole_positions = [3, 4, 7, 5, 8, 0, 1, 2, 6]
        for i, j in enumerate(hole_positions):
            grid.add_hole(i, j)
        grid.add_tile(5, 3, 0, 0)
        grid.add_tile(3, 1, 6, 0)
        grid.add_tile(1, 2, 5, 1)
        grid.add_tile(2, 1, 7, 1)
        grid.add_tile(1, 1, 6, 2)
        grid.add_tile(1, 4, 8, 2)
        grid.add_tile(1, 1, 1, 3)
        grid.add_tile(6, 2, 2, 3)
        grid.add_tile(1, 5, 0, 4)
        grid.add_tile(2, 2, 1, 5)
        grid.add_tile(4, 3, 4, 5)
        grid.add_tile(1, 2, 3, 6)
        grid.add_tile(1, 1, 1, 7)
        grid.add_tile(1, 2, 8, 7)
        grid.add_tile(3, 1, 1, 8)
        grid.add_tile(3, 1, 5, 8)
        grid.tiles.set_stroke(width=3)
        for hole in grid.holes:
            hole.border.set_color(WHITE)

        # Number the holes according to their height
        nums_color = BLUE_B
        values = [n - j for j in hole_positions]
        nums = VGroup(*[
            Integer(j).set_color(BLUE_B).next_to(grid.holes[i], UP, buff=0.15)
            for i, j in enumerate(values)
        ])
        bar_color = BLUE
        column_highlights = VGroup(*[
            VGroup(*[
                Tile(1, 1).match_width(grid.holes[0])
                .set_fill(bar_color, opacity=0)
                .set_stroke(bar_color, opacity=0)
                for _ in range(height)
            ]).arrange(UP, buff=0).match_x(hole).align_to(hole, UP)
            for height, hole in zip(values, grid.holes)
        ])
        self.add(column_highlights)
        n_color = YELLOW
        brace = Brace(grid, UP, buff=0.8)
        label = brace.get_tex("N", font_size=60).set_color(n_color)
        self.camera.frame.save_state()
        self.play(
            grid.background.animate.fade(0.9),
            grid.lines.animate.fade(0.9),
            grid.tiles.animate.fade(0.9),
            AnimationGroup(
                AnimationGroup(*[
                    AnimationGroup(
                        Succession(
                            AnimationGroup(*[
                                square.copy().animate(rate_func=there_and_back).set_fill(GREEN, opacity=1)
                                for square in column
                            ], lag_ratio=0.1),
                            FadeOut(column)
                        ),
                        FadeIn(num, shift=UP * 0.2, run_time=0.7), lag_ratio=0.1)
                    for num, column in zip(nums, column_highlights)
                ], lag_ratio=0.2),
                AnimationGroup(
                    self.camera.frame.animate(run_time=1.5).scale(1.1).shift(UP * 0.7),
                    AnimationGroup(
                        GrowFromEdge(brace, DOWN),
                        Write(label)
                    ), lag_ratio=0.2), lag_ratio=0.5)
        )
        self.remove(column_highlights)

        # Make the bar chart
        base = Line(LEFT, RIGHT).set_width(nums.get_width() * 1.1).align_to(grid, DOWN)
        bars = VGroup(*[
            Rectangle(
                width=column.get_width() * 0.9,
                height=column.get_height(),
                fill_opacity=1,
                fill_color=bar_color,
                stroke_width=0
            ).match_x(column)
            for column in column_highlights
        ]).align_to(base, DOWN)
        for bar in bars:
            bar.align_to(base, DOWN)
        chart = VGroup(bars, base)

        for num, bar in zip(nums, bars):
            num.generate_target()
            num.target.next_to(bar, UP, buff=0.2)
            bar.save_state()
            bar.stretch_to_fit_height(0.001).align_to(base, DOWN)
        self.play(
            self.camera.frame.animate.restore(),
            FadeOut(VGroup(brace, label)),
            FadeOut(grid, run_time=3),
            AnimationGroup(*[
                AnimationGroup(
                    MoveToTarget(num),
                    bar.animate.restore(), lag_ratio=0.1, run_time=2)
                for num, bar in zip(nums, bars)
            ]),
            ShowCreation(base, run_time=1)
        )
        self.wait(2)

        # Define helpers for indicating LIS/LDS
        increasing_sequence_color = GREEN_D
        decreasing_sequence_color = RED_D
        marker_thickness = 0.5 * min(bar.get_height() for bar in bars)

        def marker_rect(bar, color, level):
            r = Rectangle(
                width=bar.get_width(), height=marker_thickness,
                fill_opacity=1, fill_color=color, stroke_width=0
            )
            r.match_x(bar)
            top = bar.get_top()[1]
            r.set_y(top - marker_thickness * (level + 0.5))
            return r

        # Show a permutation with a long increasing subsequence
        bars.save_state()
        nums.save_state()
        permutation = [4, 2, 3, 8, 0, 7, 1, 6, 5]
        for i, (bar, num) in enumerate(zip(bars, nums)):
            bars[permutation[i]].generate_target()
            nums[permutation[i]].generate_target()
            bars[permutation[i]].target.match_x(bar)
            nums[permutation[i]].target.match_x(bars[permutation[i]].target)
        self.play(
            AnimationGroup(*[MoveToTarget(bar) for bar in bars]),
            AnimationGroup(*[MoveToTarget(num) for num in nums])
        )

        # Highlight the increasing subsequence, then the decreasing one
        bars_left_to_right = VGroup(*sorted(bars, key=lambda bar: bar.get_x()))
        nums_left_to_right = VGroup(*sorted(nums, key=lambda num: num.get_x()))
        base.add_updater(lambda m: self.bring_to_front(m))
        increasing_indices = [0, 1, 2, 4, 5, 7, 8]
        increasing_sequence = VGroup(*[
            bars_left_to_right[i]
            for i in increasing_indices
        ])
        increasing_markers = VGroup(*[
            marker_rect(bar, increasing_sequence_color, 0)
            for bar in increasing_sequence
        ])
        self.play(
            AnimationGroup(*[
                FadeIn(marker, run_time=0.3)
                for marker in increasing_markers
            ], lag_ratio=0.3)
        )
        self.wait(0.4)
        decreasing_indices = [5, 6]
        decreasing_sequence = VGroup(*[
            bars_left_to_right[i]
            for i in decreasing_indices
        ])
        decreasing_markers = VGroup(*[
            marker_rect(bar, decreasing_sequence_color, 1 if i in increasing_indices else 0)
            for i, bar in zip(decreasing_indices, decreasing_sequence)
        ])
        self.play(
            AnimationGroup(*[
                FadeIn(marker, run_time=0.7)
                for marker in decreasing_markers
            ], lag_ratio=0.3)
        )
        self.wait(1)
        self.play(FadeOut(increasing_markers), FadeOut(decreasing_markers))

        # Show a permutation with a long decreasing subsequence
        permutation = [7, 8, 5, 4, 6, 1, 3, 2, 0]
        for i, (bar, num) in enumerate(zip(bars_left_to_right, nums_left_to_right)):
            bars_left_to_right[permutation[i]].generate_target()
            nums_left_to_right[permutation[i]].generate_target()
            bars_left_to_right[permutation[i]].target.match_x(bar)
            nums_left_to_right[permutation[i]].target.match_x(bars_left_to_right[permutation[i]].target)
        self.play(
            AnimationGroup(*[MoveToTarget(bar) for bar in bars_left_to_right]),
            AnimationGroup(*[MoveToTarget(num) for num in nums_left_to_right])
        )

        # Highlight the decreasing subsequence, then the increasing one
        bars_left_to_right = VGroup(*sorted(bars, key=lambda bar: bar.get_x()))
        decreasing_indices = [1, 2, 3, 4, 7, 8]
        decreasing_sequence = VGroup(*[
            bars_left_to_right[i]
            for i in decreasing_indices
        ])
        decreasing_markers_dict = {
            i: marker_rect(bars_left_to_right[i], decreasing_sequence_color, 0)
            for i in decreasing_indices
        }
        decreasing_markers = VGroup(*decreasing_markers_dict.values())
        self.play(
            AnimationGroup(*[
                FadeIn(marker, run_time=0.3)
                for marker in decreasing_markers
            ], lag_ratio=0.3)
        )
        self.wait(0.4)
        increasing_indices = [5, 6, 7]
        increasing_sequence = VGroup(*[
            bars_left_to_right[i]
            for i in increasing_indices
        ])
        increasing_markers = VGroup(*[
            marker_rect(bar, increasing_sequence_color, 0)
            for bar in increasing_sequence
        ])
        shared_indices = [i for i in increasing_indices if i in decreasing_indices]
        self.play(
            AnimationGroup(*[
                FadeIn(marker, run_time=0.3)
                for marker in increasing_markers
            ], lag_ratio=0.3),
            AnimationGroup(*[
                decreasing_markers_dict[i].animate.shift(DOWN * marker_thickness)
                for i in shared_indices
            ])
        )
        self.wait(1)

        # Flash through many other permutations, highlighting their LIS and LDS
        self.remove(increasing_markers, decreasing_markers)
        slot_xs = sorted(bar.get_x() for bar in bars)

        def get_extreme_subsequence_slots(seq, increasing=True):
            n = len(seq)
            lengths = [1] * n
            prev = [-1] * n
            for i in range(n):
                for j in range(i):
                    better = (seq[j] < seq[i]) if increasing else (seq[j] > seq[i])
                    if better and lengths[j] + 1 > lengths[i]:
                        lengths[i] = lengths[j] + 1
                        prev[i] = j
            end = max(range(n), key=lambda i: lengths[i])
            slots = []
            while end != -1:
                slots.append(end)
                end = prev[end]
            return set(slots)

        n_flashes = 30
        prev_perm = None
        markers = VGroup()
        self.add(markers)
        camera_shift_iter = 5
        inequality = Tex(
            R"\text{LIS} \cdot \text{LDS} \ge N",
            font_size=80,
            tex_to_color_map={"LIS": increasing_sequence_color, "LDS": decreasing_sequence_color, "N": n_color}
        ).shift(RIGHT * 7 + UP * 1)

        value_font_size = 80
        value_row_y = inequality.get_bottom()[1] - 1
        lis_value = None
        lds_value = None

        def make_lis_value(val):
            mob = Integer(val, font_size=value_font_size).set_color(increasing_sequence_color)
            mob.match_x(inequality["LIS"])
            mob.set_y(value_row_y)
            return mob

        def make_lds_value(val):
            mob = Integer(val, font_size=value_font_size).set_color(decreasing_sequence_color)
            mob.match_x(inequality["LDS"])
            mob.set_y(value_row_y)
            return mob

        for i in range(n_flashes):
            perm = np.random.permutation(9).tolist()
            while perm == prev_perm:
                perm = np.random.permutation(9).tolist()
            prev_perm = perm

            for slot, bar_index in enumerate(perm):
                bars[bar_index].set_x(slot_xs[slot])
                nums[bar_index].set_x(slot_xs[slot])

            heights_in_order = [values[bar_index] for bar_index in perm]
            lis_slots = get_extreme_subsequence_slots(heights_in_order, increasing=True)
            lds_slots = get_extreme_subsequence_slots(heights_in_order, increasing=False)
            lis_val = len(lis_slots)
            lds_val = len(lds_slots)

            self.remove(markers)
            markers = VGroup()
            for slot, bar_index in enumerate(perm):
                bar = bars[bar_index]
                is_lis = slot in lis_slots
                is_lds = slot in lds_slots
                if is_lis:
                    markers.add(marker_rect(bar, increasing_sequence_color, 0))
                if is_lds:
                    markers.add(marker_rect(bar, decreasing_sequence_color, 1 if is_lis else 0))
            self.add(markers)

            if lis_value is not None:
                self.remove(lis_value)
                lis_value = make_lis_value(lis_val)
                self.add(lis_value)
            if lds_value is not None:
                self.remove(lds_value)
                lds_value = make_lds_value(lds_val)
                self.add(lds_value)

            if i == camera_shift_iter:
                self.set_camera_target_position(0, 0, 0, (3.35, 0.38, 0.00), 9.25)
                lis_value = make_lis_value(lis_val)
                self.play(FadeIn(inequality["LIS"]), FadeIn(lis_value))
            elif i == camera_shift_iter + 3:
                cdot_value = Tex(R"\cdot", font_size=value_font_size)
                cdot_value.match_x(inequality[R"\cdot"])
                cdot_value.set_y(value_row_y)
                lds_value = make_lds_value(lds_val)
                self.play(
                    FadeIn(inequality[R"\cdot"]),
                    FadeIn(cdot_value),
                    FadeIn(inequality["LDS"]),
                    FadeIn(lds_value)
                )
            elif i == camera_shift_iter + 8:
                geq_value = Tex(R"\ge", font_size=value_font_size)
                geq_value.match_x(inequality[-2])
                geq_value.set_y(value_row_y)
                n_value = Tex(str(n), font_size=value_font_size).set_color(n_color)
                n_value.match_x(inequality[-1])
                n_value.set_y(value_row_y)
                self.play(
                    FadeIn(inequality[-2:]),
                    FadeIn(geq_value),
                    FadeIn(n_value),
                    GrowFromEdge(brace, UP),
                    Write(label)
                )
            else:
                self.wait(1)

        # Switch to the main example for the rest of the scene
        self.clear()
        self.camera.frame.restore()
        self.add(chart, nums)
        bars.restore()
        nums.restore()
        base.add_updater(lambda m: self.bring_to_front(m))
        self.wait(1)

        # Focus on one of the bars
        focus_index = 3
        focus_bar = bars[focus_index]
        arrow = Arrow(ORIGIN, DOWN * 1.5, thickness=5).set_color(YELLOW).next_to(focus_bar, UP, buff=1.5)
        self.play(
            AnimationGroup(*[
                VGroup(bar, num).animate.set_opacity(0.1)
                for bar, num in zip(bars[focus_index + 1:], nums[focus_index + 1:])
            ]),
            GrowArrow(arrow)
        )

        # Highlight its longest increasing and decreasing subsequences
        increasing_indices = [2, 3]
        increasing_sequence = VGroup(*[
            bars[i]
            for i in increasing_indices
        ])
        increasing_markers = VGroup(*[
            marker_rect(bar, increasing_sequence_color, 0)
            for bar in increasing_sequence
        ])
        self.play(
            AnimationGroup(*[
                FadeIn(marker)
                for marker in increasing_markers
            ], lag_ratio=0.3)
        )
        self.wait(1)
        lis_text = Tex(R"\text{LIS}: 2", font_size=110).set_color(increasing_sequence_color)
        lds_text = Tex(R"\text{LDS}: 3", font_size=110).set_color(decreasing_sequence_color)
        lds_text.next_to(lis_text, DOWN, buff=0.6).align_to(lis_text, LEFT)
        VGroup(lis_text, lds_text).set_y(0).to_edge(RIGHT, buff=1.5)
        base.suspend_updating()
        self.play(
            AnimationGroup(
                VGroup(chart, nums, arrow, increasing_markers).animate.to_edge(LEFT, buff=1.5),
                Write(lis_text), lag_ratio=0.6, run_time=1.5)
        )
        base.resume_updating()
        self.wait(1)
        decreasing_indices = [0, 1, 3]
        decreasing_sequence = VGroup(*[
            bars[i]
            for i in decreasing_indices
        ])
        decreasing_markers = VGroup(*[
            marker_rect(bar, decreasing_sequence_color, 1 if i in increasing_indices else 0)
            for i, bar in zip(decreasing_indices, decreasing_sequence)
        ])
        self.play(
            AnimationGroup(*[
                FadeIn(marker)
                for marker in decreasing_markers
            ], lag_ratio=0.3)
        )
        self.wait(1)
        self.play(Write(lds_text), run_time=1.5)
        self.wait(1)
        self.wait(2)

        # Save the values as a pair of numbers below the bar
        pair = Tex("(2, 3)", font_size=30).next_to(focus_bar, DOWN)
        pair[1].set_color(increasing_sequence_color)
        pair[3].set_color(decreasing_sequence_color)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    TransformFromCopy(lis_text[-1], pair[1]),
                    TransformFromCopy(lds_text[-1], pair[3]), run_time=2),
                FadeIn(VGroup(pair[0], pair[2], pair[4])), lag_ratio=0.7)
        )
        self.wait(2)

        # Switch focus back to the full chart
        base.clear_updaters()
        chart.generate_target()
        chart.target.set_opacity(1).stretch(1.5, 0).center()
        nums.generate_target()
        nums.target.set_opacity(1)
        for num, bar in zip(nums.target, chart.target[0]):
            num.match_x(bar)
        pair.generate_target()
        pair.target.match_x(chart.target[0][3]).scale(1.3)

        increasing_markers.set_z_index(100)
        decreasing_markers.set_z_index(100)
        self.play(
            FadeOut(VGroup(arrow, lis_text, lds_text), run_time=1),
            MoveToTarget(chart, run_time=2),
            MoveToTarget(nums, run_time=2),
            MoveToTarget(pair, run_time=2),
            FadeOut(increasing_markers, shift=RIGHT * 0.08, run_time=0.6),
            FadeOut(decreasing_markers, shift=RIGHT * 0.08, run_time=0.6)
        )
        base.add_updater(lambda m: self.bring_to_front(m))
        self.wait(2)

        # Show that all the numbers are distinct
        if False:
            self.remove(pair)
            self.wait(1)
            circles = VGroup(*[Circle(radius=0.35, fill_opacity=0, stroke_width=3, stroke_color=YELLOW).move_to(num) for num in nums])
            self.play(AnimationGroup(*[ShowCreation(circle) for circle in circles], lag_ratio=0.1))

        # Add the (LIS, LDS) pair for each bar
        lis_lds_lengths = [(1, 1), (1, 2), (1, 3), (2, 3), (1, 4), (3, 1), (3, 2), (3, 3), (2, 4)]
        pairs = VGroup(*[
            Tex(F"({lis}, {lds})").match_height(pair).match_y(pair).match_x(bar)
            for (lis, lds), bar in zip(lis_lds_lengths, bars)
        ])
        pair_4 = pair
        for i, pair in enumerate(pairs):
            pair[1].set_color(increasing_sequence_color)
            pair[3].set_color(decreasing_sequence_color)
            pair.save_state()
            if i != focus_index:
                pair.scale(1.2).set_opacity(0)
            else:
                pair.set_opacity(1)
        self.play(
            AnimationGroup(*[
                pair.animate.restore()
                for pair in list(pairs[:focus_index]) + list(pairs[focus_index + 1:])
            ], lag_ratio=0.2), run_time=3.6)
        self.remove(pair_4)
        self.add(pairs)
        self.wait(2)

        # Do another example
        bars.save_state()
        nums.save_state()
        pairs.save_state()
        focus_index = 6
        focus_bar = bars[focus_index]
        arrow = Arrow(ORIGIN, DOWN * 1.1, thickness=4).set_color(YELLOW).next_to(focus_bar, UP, buff=0.85)
        self.play(
            VGroup(
                *[
                    VGroup(bar, num)
                    for bar, num in zip(bars[focus_index + 1:], nums[focus_index + 1:])
                ],
                pairs[:focus_index],
                pairs[focus_index + 1:]
            ).animate.set_opacity(0.1),
            GrowArrow(arrow)
        )
        increasing_indices = [2, 3, 6]
        increasing_sequence = VGroup(*[
            bars[i]
            for i in increasing_indices
        ])
        increasing_markers = VGroup(*[
            marker_rect(bar, increasing_sequence_color, 0)
            for bar in increasing_sequence
        ])
        self.play(
            AnimationGroup(*[
                FadeIn(marker)
                for marker in increasing_markers
            ], lag_ratio=0.3)
        )
        self.wait(1)
        decreasing_indices = [5, 6]
        decreasing_sequence = VGroup(*[
            bars[i]
            for i in decreasing_indices
        ])
        decreasing_markers = VGroup(*[
            marker_rect(bar, decreasing_sequence_color, 1 if i in increasing_indices else 0)
            for i, bar in zip(decreasing_indices, decreasing_sequence)
        ])
        self.play(
            AnimationGroup(*[
                FadeIn(marker)
                for marker in decreasing_markers
            ], lag_ratio=0.3)
        )
        self.wait(1)
        self.play(
            bars.animate.restore(), nums.animate.restore(), pairs.animate.restore(),
            FadeOut(arrow), FadeOut(increasing_markers), FadeOut(decreasing_markers), run_time=2
        )
        self.wait(1)

        # Indicate pairs to show uniqueness
        self.play(AnimationGroup(*[Indicate(pair) for pair in pairs], lag_ratio=0.1), run_time=3)

        # Save the full chart
        original_chart_group = VGroup(chart, nums, pairs).copy()

        # Bring in an arbitrary pair of bars
        bar1 = bars[1].copy()
        bar2 = bar1.copy()
        VGroup(bar1, bar2).arrange(buff=2).align_to(bars[0], DOWN)
        self.play(
            AnimationGroup(
                FadeOut(VGroup(bars, nums, pairs)),
                FadeIn(VGroup(bar1, bar2)), lag_ratio=0.2), run_time=3)
        self.wait(2)

        # Write an arbitrary pair of values for the LIS and LDS for that bar
        pair = Tex("(x, y)", tex_to_color_map={"x": increasing_sequence_color, "y": decreasing_sequence_color}).match_height(pairs[0]).match_y(pairs[0]).match_x(bar1)
        self.play(FadeIn(pair))

        # Make the second bar taller
        stretch_factor = 1.2
        self.play(
            bar1.animate.stretch(1 / stretch_factor, 1).align_to(bar1, DOWN),
            bar2.animate.stretch(stretch_factor, 1).align_to(bar2, DOWN), run_time=0.7)
        self.play(
            bar1.animate.stretch(stretch_factor**2, 1).align_to(bar1, DOWN),
            bar2.animate.stretch(1 / stretch_factor**2, 1).align_to(bar2, DOWN), run_time=0.7)
        self.play(
            bar1.animate.stretch(1 / stretch_factor**3, 1).align_to(bar1, DOWN),
            bar2.animate.stretch(stretch_factor**3, 1).align_to(bar2, DOWN), run_time=2)
        pair2 = Tex("(x', y')", tex_to_color_map={"x'": increasing_sequence_color, "y'": decreasing_sequence_color}).match_height(pairs[0]).match_y(pairs[0]).match_x(bar2)
        self.play(FadeIn(pair2))
        self.wait(1)

        # Show a generic tail of bars behind the first bar
        heights = [2, 6, 5, 3]
        heights = [h * 0.6 for h in heights]
        tail = VGroup(*[
            bar1.copy().stretch_to_fit_width(0.5).stretch_to_fit_height(height)
            for height in heights
        ]).arrange(
            buff=0.1
        ).next_to(
            bar1, LEFT, buff=0.2
        )
        tail_opacity = 0.4
        for bar in tail:
            bar.align_to(
                bar1, DOWN
            ).set_opacity(
                tail_opacity
            )
        cdots = Tex(R"\cdots", font_size=100).match_y(bar1)

        self.play(
            AnimationGroup(
                *[
                    FadeIn(bar)
                    for bar in tail
                ],
                Write(cdots, run_time=1.5), lag_ratio=0.1),
        )
        self.wait(2)

        # Recompute marker thickness
        marker_thickness = 0.5 * min(
            b.get_height() for b in [tail[0], tail[1], tail[2], tail[3], bar1]
        )

        # Show the increasing subsequence of length x
        increasing_sequence = VGroup(tail[0], tail[3], bar1)
        increasing_markers = VGroup(*[
            marker_rect(bar, increasing_sequence_color, 0)
            for bar in increasing_sequence
        ])
        brace = Brace(increasing_sequence, UP).shift(UP * 0.1)
        label = brace.get_tex("x", font_size=60).set_color(increasing_sequence_color).shift(UP * 0.2)
        self.play(
            AnimationGroup(*[
                bar.animate.set_opacity(1)
                for bar in increasing_sequence
            ], lag_ratio=0.1),
            AnimationGroup(*[
                FadeIn(marker)
                for marker in increasing_markers
            ], lag_ratio=0.1),
            GrowFromEdge(brace, DOWN),
            Write(label)
        )
        self.wait(2)

        # Extend the increasing sequence to the second bar
        brace.generate_target()
        label.generate_target()
        extended_brace = Brace(VGroup(increasing_sequence, bar2), UP).shift(UP * 0.1)
        extended_label = extended_brace.get_tex(R"x' \ge x + 1", font_size=60).set_color(increasing_sequence_color).shift(UP * 0.2)
        part1 = extended_label[:3]
        part2 = extended_label[3:]
        part2.save_state()
        part2.match_x(extended_brace)
        bar2_increasing_marker = marker_rect(bar2, increasing_sequence_color, 0)
        increasing_markers.add(bar2_increasing_marker)
        self.play(
            TransformFromCopy(brace, extended_brace),
            TransformMatchingShapes(label.copy(), part2),
            FadeIn(bar2_increasing_marker), run_time=2)
        self.wait(1)
        self.play(part2.animate.restore(), FadeIn(part1, shift=RIGHT * 0.5))

        # Save the example
        case1 = VGroup(
            tail, bar1, bar2, increasing_markers, base, pair, pair2,
            brace, label, extended_brace, extended_label, cdots
        ).copy()

        # Make the second bar shorter
        increasing_markers.set_z_index(100)
        self.play(
            FadeOut(VGroup(brace, label, extended_brace, extended_label)),
            FadeOut(increasing_markers[:-1]),
            FadeOut(increasing_markers[-1], shift=DOWN * 3),
            tail.animate.set_opacity(tail_opacity),
            bar2.animate.stretch_to_fit_height(0.6 * bar1.get_height()).align_to(bar2, DOWN)
        )
        self.wait(2)

        # Show the decreasing subsequence of length 7
        decreasing_sequence = VGroup(tail[1], tail[2], bar1)
        decreasing_markers = VGroup(*[
            marker_rect(bar, decreasing_sequence_color, 0)
            for bar in decreasing_sequence
        ])
        brace = Brace(decreasing_sequence, UP)
        label = brace.get_tex("y", font_size=60).set_color(decreasing_sequence_color)
        self.play(
            AnimationGroup(*[
                bar.animate.set_opacity(1)
                for bar in decreasing_sequence
            ], lag_ratio=0.1),
            AnimationGroup(*[
                FadeIn(marker)
                for marker in decreasing_markers
            ], lag_ratio=0.1),
            GrowFromEdge(brace, DOWN),
            Write(label)
        )
        self.wait(2)

        # Extend the decreasing sequence to the second bar
        brace.generate_target()
        label.generate_target()
        extended_brace = Brace(VGroup(decreasing_sequence, bar2), UP).align_to(case1[-3], UP)
        extended_label = extended_brace.get_tex(R"y' \ge y + 1", font_size=60).set_color(decreasing_sequence_color).align_to(case1[-2], UP)
        part1 = extended_label[:3]
        part2 = extended_label[3:]
        part2.save_state()
        part2.match_x(extended_brace)
        bar2_decreasing_marker = marker_rect(bar2, decreasing_sequence_color, 0)
        decreasing_markers.add(bar2_decreasing_marker)
        self.play(
            TransformFromCopy(brace, extended_brace),
            TransformMatchingShapes(label.copy(), part2),
            FadeIn(bar2_decreasing_marker), run_time=2)
        self.wait(1)
        self.play(part2.animate.restore(), FadeIn(part1, shift=RIGHT * 0.5))

        # Save the second example
        case2 = VGroup(
            tail, bar1, bar2, decreasing_markers, base, pair, pair2,
            brace, label, extended_brace, extended_label, cdots
        )

        # Show both examples side by side
        case1.clear_updaters()
        base.clear_updaters()
        case1[1:3].set_stroke(width=3, color=YELLOW, behind=True)
        case1[1:4].set_z_index(400)
        case2.generate_target()
        case2.target[1:3].set_stroke(width=3, color=YELLOW)
        case2.set_stroke(behind=True)
        VGroup(case1, case2.target).scale(0.63).arrange(buff=0.7)
        case2.target.align_to(case1, DOWN)
        case2[1:4].set_z_index(400)
        case1_label = TexText("Case 1").next_to(case1, DOWN, buff=1)
        case2_label = TexText("Case 2").next_to(case2.target, DOWN, buff=1)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    FadeIn(case1, shift=RIGHT * 5),
                    MoveToTarget(case2)
                ),
                LaggedStartMap(FadeIn, VGroup(case1_label, case2_label), lag_ratio=0.2, shift=UP * 1.5, run_time=0.8), lag_ratio=0.1), run_time=3)

        # Bring back the original chart
        original_chart_group.clear_updaters()
        self.play(
            FadeOut(VGroup(case1, case2, case1_label, case2_label), shift=DOWN * 7),
            FadeIn(original_chart_group, shift=DOWN * 7), run_time=1.5)
        chart, nums, pairs = original_chart_group[0], original_chart_group[1], original_chart_group[2]
        self.wait(0.5)

        # Indicate pairs again to show uniqueness
        self.play(AnimationGroup(*[Indicate(pair) for pair in pairs], lag_ratio=0.1), run_time=3)
        self.wait(2)

        # Put each pair of numbers on a coordinate grid
        number_plane = NumberPlane(
            x_range=[0, 5],
            y_range=[0, 5]
        ).set_width(4.5).to_edge(RIGHT, buff=1)
        number_plane.remove(number_plane.faded_lines)
        x_labels = number_plane.add_coordinate_labels(x_values=[1, 2, 3, 4, 5], y_values=[], font_size=30, direction=DOWN)
        y_labels = number_plane.add_coordinate_labels(x_values=[], y_values=[1, 2, 3, 4, 5], font_size=30, direction=LEFT)
        x_labels.set_color(increasing_sequence_color)
        y_labels.set_color(decreasing_sequence_color)
        points = Group(*[
            Group(GlowDot(), TrueDot()).set_color(n_color).move_to(number_plane.c2p(x, y))
            for (x, y) in lis_lds_lengths
        ])
        point_labels = pairs.copy()
        for point, label in zip(points, point_labels):
            label.scale(0.6).next_to(point, UR, buff=-0.1)
        self.play(
            original_chart_group.animate(run_time=2).scale(0.55).to_edge(LEFT, buff=1),
            FadeIn(number_plane, shift=LEFT * 6, run_time=2),
            AnimationGroup(*[
                AnimationGroup(
                    TransformFromCopy(pair, label, path_arc=-PI * 0.2),
                    FadeIn(point), lag_ratio=0.6, run_time=2 + i * 0.2)
                for i, (point, pair, label) in enumerate(zip(points, pairs, point_labels))
            ])
        )
        self.wait(1)

        # Draw a rectangle bounding the points
        unit_size = number_plane.background_lines[1].get_y() - number_plane.background_lines[0].get_y()
        rect = Rectangle(
            width=unit_size * 3,
            height=unit_size * 4,
            fill_opacity=0.4,
            fill_color=TEAL,
            stroke_width=4,
            stroke_color=TEAL
        ).align_to(number_plane.c2p(0, 0), DL)
        self.bring_to_back(rect)
        self.play(
            DrawBorderThenFill(rect, stroke_width=6), run_time=2)
        self.wait(2)

        # Show the dimensions
        width_brace = Brace(rect, DOWN, buff=0.5)
        width_label = width_brace.get_tex(R"\text{LIS}").set_color(increasing_sequence_color)
        bars, base = chart
        base.add_updater(lambda m: self.bring_to_front(m))
        # Bars have since been scaled down (coordinate-grid step), so
        # recompute the marker thickness to match their current size.
        marker_thickness = 0.5 * min(bar.get_height() for bar in bars)

        increasing_indices = [2, 3, 7]
        decreasing_indices = [0, 1, 3, 8]
        increasing_markers = VGroup(*[
            marker_rect(bars[i], increasing_sequence_color, 0)
            for i in increasing_indices
        ])
        self.play(
            GrowFromEdge(width_brace, UP),
            Write(width_label),
            AnimationGroup(*[
                FadeIn(marker)
                for marker in increasing_markers
            ], lag_ratio=0.1)
        )
        self.wait(1)

        height_brace = Brace(rect, LEFT, buff=0.5)
        height_label = height_brace.get_tex(R"\text{LDS}").set_color(decreasing_sequence_color)
        decreasing_markers = VGroup(*[
            marker_rect(bars[i], decreasing_sequence_color, 1 if i in increasing_indices else 0)
            for i in decreasing_indices
        ])
        self.play(
            GrowFromEdge(height_brace, RIGHT),
            Write(height_label),
            AnimationGroup(*[
                FadeIn(marker)
                for marker in decreasing_markers
            ], lag_ratio=0.1)
        )
        self.wait(1)

        # Circle the lattice points
        lattice_points = VGroup()
        for i in range(3):
            for j in range(4):
                point = Circle(
                    radius=0.15, stroke_width=3, stroke_color=WHITE
                ).move_to(number_plane.c2p(i + 1, j + 1))
                lattice_points.add(point)
        self.play(AnimationGroup(*[ShowCreation(point) for point in lattice_points], lag_ratio=0.15))

        # Write the inequality up top
        inequality.scale(0.8).set_x(0).to_edge(UP, buff=0.6)
        self.play(
            AnimationGroup(
                TransformMatchingShapes(width_label.copy(), inequality["LIS"], path_arc=-PI * 0.35),
                TransformMatchingShapes(height_label.copy(), inequality["LDS"], path_arc=-PI * 0.2),
                GrowFromCenter(inequality[R"\cdot"], path_arc=-PI * 0.3),
                Write(inequality[R"\ge N"]), lag_ratio=0.3, run_time=2)
        )

    def set_camera_target_position(
        self,
        theta_degrees=None,
        phi_degrees=None,
        gamma_degrees=None,
        center=None,
        height=None,
        drift_time=2.0,
    ):
        frame = self.camera.frame
        frame.clear_updaters()
        initial_orientation = frame.get_orientation()
        initial_height = frame.get_height()
        initial_eye = frame.get_implied_camera_location()
        target_frame = frame.copy()
        target_frame.reorient(theta_degrees, phi_degrees, gamma_degrees, center, height)
        target_orientation = target_frame.get_orientation()
        target_height = target_frame.get_height()
        target_eye = target_frame.get_implied_camera_location()
        fovy = frame.get_field_of_view()
        slerp = Slerp([0, 1], Rotation.concatenate([initial_orientation, target_orientation]))
        drift_time = max(drift_time, 1e-4)
        elapsed = 0.0

        def update_camera(f, dt):
            nonlocal elapsed
            elapsed += dt
            t = min(elapsed / drift_time, 1.0)
            alpha = smooth(t)
            current_orientation = slerp(alpha)
            current_height = interpolate(initial_height, target_height, alpha)
            current_eye = interpolate(initial_eye, target_eye, alpha)
            focal_distance = 0.5 * current_height / np.tan(0.5 * fovy)
            to_camera = current_orientation.as_matrix().T[2]
            current_center = current_eye - focal_distance * to_camera
            f.set_orientation(current_orientation)
            f.move_to(current_center)
            f.set_height(current_height)
            if t >= 1.0:
                f.remove_updater(update_camera)
        frame.add_updater(update_camera)


class ErdosSzekeresV2(InteractiveScene):
    def construct(self):
        # Add the grid
        n = 16
        grid = Grid(n).set_width(6)
        self.add(grid)
        hole_positions = [7, 9, 15, 0, 2, 4, 12, 8, 6, 13, 5, 1, 14, 10, 3, 11]
        for i, j in enumerate(hole_positions):
            grid.add_hole(i, j)
        for hole in grid.holes:
            hole.border.set_color(WHITE)
        grid.get_reasonable_tiling()
        grid.tiles.set_stroke(width=3)
        self.wait(1)

        # Change it into the n = 12 version
        for n in range(15, 9, -1):
            self.wait(0.25)
            self.remove(grid)
            grid = Grid(n).set_width(6)
            self.add(grid)
            hole_positions = np.random.permutation(list(range(n)))
            if n == 10:
                hole_positions = [8, 7, 5, 3, 2, 9, 0, 6, 4, 1]
            for i, j in enumerate(hole_positions):
                grid.add_hole(i, j)
            for hole in grid.holes:
                hole.border.set_color(WHITE)
            grid.get_reasonable_tiling()
            grid.tiles.set_stroke(width=3)
        self.clear()
        self.add(grid)

        # Number the holes according to their height
        nums_color = BLUE_B
        values = [n - j for j in hole_positions]
        nums = VGroup(*[
            Integer(j, font_size=40).set_color(BLUE_B).next_to(grid.holes[i], UP, buff=0.15)
            for i, j in enumerate(values)
        ])
        bar_color = BLUE
        column_highlights = VGroup(*[
            VGroup(*[
                Tile(1, 1).match_width(grid.holes[0])
                .set_fill(bar_color, opacity=0)
                .set_stroke(bar_color, opacity=0)
                for _ in range(height)
            ]).arrange(UP, buff=0).match_x(hole).align_to(hole, UP)
            for height, hole in zip(values, grid.holes)
        ])
        self.add(column_highlights)
        n_color = YELLOW
        brace = Brace(grid, UP, buff=0.8)
        label = brace.get_tex("N", font_size=60).set_color(n_color)
        self.camera.frame.save_state()
        self.play(
            grid.background.animate.fade(0.9),
            grid.lines.animate.fade(0.9),
            grid.tiles.animate.fade(0.9),
            AnimationGroup(
                AnimationGroup(*[
                    AnimationGroup(
                        Succession(
                            AnimationGroup(*[
                                square.copy().animate(rate_func=there_and_back).set_fill(GREEN, opacity=1)
                                for square in column
                            ], lag_ratio=0.05),
                            FadeOut(column)
                        ),
                        FadeIn(num, shift=UP * 0.2, run_time=0.7), lag_ratio=0.1)
                    for num, column in zip(nums, column_highlights)
                ], lag_ratio=0.1),
                AnimationGroup(
                    self.camera.frame.animate(run_time=1.5).scale(1.1).shift(UP * 0.7),
                    AnimationGroup(
                        GrowFromEdge(brace, DOWN),
                        Write(label)
                    ), lag_ratio=0.2), lag_ratio=0.5)
        )
        self.remove(column_highlights)

        # Make the bar chart
        base = Line(LEFT, RIGHT).set_width(nums.get_width() * 1.1).align_to(grid, DOWN)
        bars = VGroup(*[
            Rectangle(
                width=column.get_width() * 0.9,
                height=column.get_height(),
                fill_opacity=1,
                fill_color=bar_color,
                stroke_width=0
            ).match_x(column)
            for column in column_highlights
        ]).align_to(base, DOWN)
        for bar in bars:
            bar.align_to(base, DOWN)
        chart = VGroup(bars, base)

        for bar in bars:
            bar.save_state()
            bar.stretch_to_fit_height(0.001).align_to(base, DOWN)
        self.play(
            FadeOut(grid, run_time=3),
            VGroup(brace, label).animate(run_time=2).shift(DOWN * 0.3),
            AnimationGroup(*[
                AnimationGroup(
                    FadeOut(num, shift=UP * 0.3),
                    bar.animate.restore(), lag_ratio=0.1, run_time=2)
                for num, bar in zip(nums, bars)
            ]),
            ShowCreation(base, run_time=1)
        )

        # Define helpers for indicating LIS/LDS
        increasing_sequence_color = GREEN_D
        decreasing_sequence_color = RED_D
        marker_thickness = 0.5 * min(bar.get_height() for bar in bars)

        def marker_rect(bar, color, level):
            r = Rectangle(
                width=bar.get_width(), height=marker_thickness,
                fill_opacity=1, fill_color=color, stroke_width=0
            )
            r.match_x(bar)
            top = bar.get_top()[1]
            r.set_y(top - marker_thickness * (level + 0.5))
            return r

        # Define subsequence helper functions
        increasing_sequence_color = GREEN_D
        decreasing_sequence_color = RED_D

        def longest_increasing_subsequence_indices(seq):
            n_terms = len(seq)
            lengths = [1] * n_terms
            prev = [-1] * n_terms
            for i in range(n_terms):
                for j in range(i):
                    if seq[j] < seq[i] and lengths[j] + 1 > lengths[i]:
                        lengths[i] = lengths[j] + 1
                        prev[i] = j
            end = max(range(n_terms), key=lambda i: lengths[i])
            indices = []
            while end != -1:
                indices.append(end)
                end = prev[end]
            return list(reversed(indices))

        def longest_decreasing_subsequence_indices(seq):
            return longest_increasing_subsequence_indices([-x for x in seq])

        def maximal_longest_increasing_subsequences(seq, include_length_1=False):
            remaining_indices = list(range(len(seq)))
            all_sequences = []
            while remaining_indices:
                remaining_values = [seq[i] for i in remaining_indices]
                local_lis = longest_increasing_subsequence_indices(remaining_values)
                global_lis = [remaining_indices[i] for i in local_lis]
                all_sequences.append(global_lis)
                used = set(global_lis)
                remaining_indices = [i for i in remaining_indices if i not in used]
            if not include_length_1:
                all_sequences = [s for s in all_sequences if len(s) > 1]
            return all_sequences

        def maximal_longest_decreasing_subsequences(seq, include_length_1=False):
            remaining_indices = list(range(len(seq)))
            all_sequences = []
            while remaining_indices:
                remaining_values = [seq[i] for i in remaining_indices]
                local_lds = longest_decreasing_subsequence_indices(remaining_values)
                global_lds = [remaining_indices[i] for i in local_lds]
                all_sequences.append(global_lds)
                used = set(global_lds)
                remaining_indices = [i for i in remaining_indices if i not in used]
            if not include_length_1:
                all_sequences = [s for s in all_sequences if len(s) > 1]
            return all_sequences

        left_to_right = sorted(zip(values, bars), key=lambda triple: triple[1].get_x())
        values, bars = [v for v, _ in left_to_right], VGroup(*[b for _, b in left_to_right])
        base.add_updater(lambda m: self.bring_to_front(m))

        # --- Dot-nesting registry -------------------------------------------------
        # Each sequence path now places a dot at every included bar (not just the
        # first), so it's unambiguous which bars are actually part of a run even
        # when several points happen to be collinear. When two paths share a bar,
        # whichever one's dot is *prepared* second (i.e. whichever gets drawn/added
        # second) is shrunk so it nests visibly inside the one already there.
        # Paths must release their claim on a bar (via release_path_dots /
        # release_dots_recursive) once they're faded out, so later, unrelated
        # paths on the same physical bar objects go back to full size.
        DOT_RADIUS = 0.1
        DOT_SHRINK_FACTOR = 0.6
        dot_stacks = {}  # id(bar) -> list of {"token", "radius"} dicts, in registration order
        _dot_token_counter = [0]

        def register_dot(bar):
            key = id(bar)
            stack = dot_stacks.setdefault(key, [])
            radius = DOT_RADIUS if not stack else min(e["radius"] for e in stack) * DOT_SHRINK_FACTOR
            _dot_token_counter[0] += 1
            token = _dot_token_counter[0]
            stack.append({"token": token, "radius": radius})
            return radius, token

        def release_dot(bar, token):
            key = id(bar)
            stack = dot_stacks.get(key, [])
            dot_stacks[key] = [e for e in stack if e["token"] != token]
            if key in dot_stacks and not dot_stacks[key]:
                del dot_stacks[key]

        def prepare_path_dots(path):
            # Resolves final dot sizes (nesting inside any dot already
            # registered on a shared bar) and resizes the dots in place.
            # Must be called exactly once per path, at the moment it's about
            # to appear on screen -- whether via path_creation_animation or
            # a direct self.add/FadeIn.
            dots, line, tip = path
            for dot in dots:
                radius, token = register_dot(dot.bar)
                dot.dot_token = token
                dot.set_width(radius * 2)

        def release_path_dots(path):
            dots, line, tip = path
            for dot in dots:
                token = getattr(dot, "dot_token", None)
                if token is not None:
                    release_dot(dot.bar, token)

        def release_dots_recursive(mobject):
            if getattr(mobject, "is_sequence_path", False):
                release_path_dots(mobject)
            else:
                for sub in mobject:
                    release_dots_recursive(sub)
        # ----------------------------------------------------------------------------

        def sequence_path(bars, indices, color):
            points = [bars[i].get_top() for i in indices]
            line = VMobject()
            tip = VGroup()
            if len(points) >= 2:
                tip_length = 0.3
                tip_width = 0.3
                direction = normalize(points[-1] - points[-2])
                perp = np.array([-direction[1], direction[0], 0])
                tip_base_center = points[-1] - tip_length * direction
                line.set_points_as_corners(points[:-1] + [tip_base_center])
                tip = Polygon(
                    points[-1],
                    tip_base_center + (tip_width / 2) * perp,
                    tip_base_center - (tip_width / 2) * perp,
                    fill_color=color,
                    fill_opacity=1,
                    stroke_width=0,
                )
                tip.direction = direction
            else:
                line.set_points_as_corners(points)
            line.set_stroke(color, width=9)
            dots = VGroup(*[
                Dot(radius=DOT_RADIUS).set_color(color).move_to(point)
                for point in points
            ])
            for dot, i in zip(dots, indices):
                dot.bar = bars[i]
            path = VGroup(dots, line, tip).set_stroke(opacity=0.85)
            path.is_sequence_path = True
            return path

        def path_creation_animation(path):
            dots, line, tip = path
            prepare_path_dots(path)
            animations = [AnimationGroup(*[GrowFromCenter(dot) for dot in dots]), ShowCreation(line)]
            if len(tip.get_points()) > 0:
                animations.append(FadeIn(tip, shift=tip.direction * 0.2))
            return AnimationGroup(*animations, lag_ratio=0.3)

        # Draw the subsequences
        self.play(
            FadeOut(VGroup(brace, label)),
            self.camera.frame.animate.restore(),
        )
        self.wait(2)
        all_increasing_indices = maximal_longest_increasing_subsequences(values)
        all_decreasing_indices = maximal_longest_decreasing_subsequences(values)

        increasing_paths = VGroup(*[
            sequence_path(bars, indices, increasing_sequence_color)
            for indices in all_increasing_indices
        ])
        decreasing_paths = VGroup(*[
            sequence_path(bars, indices, decreasing_sequence_color)
            for indices in all_decreasing_indices
        ])
        self.play(
            AnimationGroup(*[
                path_creation_animation(path)
                for path in decreasing_paths
            ], lag_ratio=0.1)
        )
        self.wait(0.4)
        self.play(
            AnimationGroup(*[
                path_creation_animation(path)
                for path in increasing_paths
            ], lag_ratio=0.1)
        )
        self.wait(1)
        self.play(FadeOut(increasing_paths), FadeOut(decreasing_paths))
        release_dots_recursive(VGroup(increasing_paths, decreasing_paths))

        # Show a permutation with a long decreasing subsequence but a short increasing one
        new_bar_heights = [9, 7, 4, 10, 1, 8, 6, 3, 2, 5]
        permutation = [values.index(i) for i in new_bar_heights]
        for i, bar in enumerate(bars):
            bars[permutation[i]].generate_target()
            bars[permutation[i]].target.match_x(bar)
        self.play(AnimationGroup(*[MoveToTarget(bar) for bar in bars]))

        # Highlight the decreasing sequences, then the increasing ones
        left_to_right = sorted(zip(values, bars), key=lambda triple: triple[1].get_x())
        values, bars = [v for v, _ in left_to_right], VGroup(*[b for _, b in left_to_right])
        base.add_updater(lambda m: self.bring_to_front(m))

        all_decreasing_indices_2 = maximal_longest_decreasing_subsequences(values)
        all_increasing_indices_2 = maximal_longest_increasing_subsequences(values)

        # Built in the order they'll actually be revealed (increasing first,
        # then decreasing), so any bar shared between the two gets its
        # decreasing dot correctly nested inside the increasing one.
        increasing_paths_2 = VGroup(*[
            sequence_path(bars, indices, increasing_sequence_color)
            for indices in all_increasing_indices_2
        ])
        decreasing_paths_2 = VGroup(*[
            sequence_path(bars, indices, decreasing_sequence_color)
            for indices in all_decreasing_indices_2
        ])

        self.play(
            AnimationGroup(*[
                path_creation_animation(path)
                for path in increasing_paths_2
            ], lag_ratio=0.1)
        )
        self.wait(0.4)
        self.play(
            AnimationGroup(*[
                path_creation_animation(path)
                for path in decreasing_paths_2
            ], lag_ratio=0.1)
        )
        self.wait(1)

        # Show some random permutations
        inequality = Tex(
            R"\frac{\text{LIS} + \text{LDS}}{2} \ge \sqrt{N}",
            font_size=80,
            tex_to_color_map={"LIS": increasing_sequence_color, "LDS": decreasing_sequence_color, "N": n_color}
        ).shift(RIGHT * 7 + UP * 1)

        value_expr = inequality.copy()
        value_expr.next_to(inequality, DOWN, buff=1.0)
        lis_placeholder = value_expr["LIS"].set_opacity(0)
        lds_placeholder = value_expr["LDS"].set_opacity(0)
        n_placeholder = value_expr["N"].set_opacity(0)
        lis_anchor = lis_placeholder.get_center()
        lds_anchor = lds_placeholder.get_center()
        n_anchor = n_placeholder.get_center()
        value_expr.remove(lis_placeholder, lds_placeholder, n_placeholder)

        def make_lis_value(val):
            mob = Integer(val, font_size=80).set_color(increasing_sequence_color)
            mob.move_to(lis_anchor)
            return mob

        def make_lds_value(val):
            mob = Integer(val, font_size=80).set_color(decreasing_sequence_color)
            mob.move_to(lds_anchor)
            return mob

        n_value = Integer(n, font_size=70).set_color(n_color)
        n_value.move_to(n_anchor)

        lis_value = make_lis_value(len(longest_increasing_subsequence_indices(values)))
        lds_value = make_lds_value(len(longest_decreasing_subsequence_indices(values)))

        self.set_camera_target_position(0, 0, 0, (3.35, 0.38, 0.00), 9.25)
        self.play(
            FadeIn(inequality),
            FadeIn(value_expr),
            FadeIn(lis_value),
            FadeIn(lds_value),
            FadeIn(n_value),
            FadeOut(decreasing_paths_2[1:]),
            FadeOut(increasing_paths_2[1:])
        )
        release_dots_recursive(VGroup(decreasing_paths_2[1:], increasing_paths_2[1:]))
        self.wait(1)
        self.play(FadeOut(VGroup(decreasing_paths_2[0], increasing_paths_2[0])))
        release_dots_recursive(VGroup(decreasing_paths_2[0], increasing_paths_2[0]))

        num_iters = 10
        for _ in range(num_iters):
            new_bar_heights = np.random.permutation(list(range(1, n + 1)))
            permutation = [values.index(i) for i in new_bar_heights]
            for i, bar in enumerate(bars):
                bars[permutation[i]].generate_target()
                bars[permutation[i]].target.match_x(bar)
            self.play(AnimationGroup(*[MoveToTarget(bar) for bar in bars]), run_time=0.7)
            left_to_right = sorted(zip(values, bars), key=lambda triple: triple[1].get_x())
            values, bars = [v for v, _ in left_to_right], VGroup(*[b for _, b in left_to_right])
            base.add_updater(lambda m: self.bring_to_front(m))

            lis_indices = longest_increasing_subsequence_indices(values)
            lds_indices = longest_decreasing_subsequence_indices(values)
            lis = sequence_path(bars, lis_indices, increasing_sequence_color)
            lds = sequence_path(bars, lds_indices, decreasing_sequence_color)

            new_lis_value = make_lis_value(len(lis_indices))
            new_lds_value = make_lds_value(len(lds_indices))
            self.play(
                path_creation_animation(lis),
                path_creation_animation(lds),
                FadeOut(lis_value), FadeIn(new_lis_value),
                FadeOut(lds_value), FadeIn(new_lds_value), run_time=1.2)
            lis_value, lds_value = new_lis_value, new_lds_value

            self.play(FadeOut(lis), FadeOut(lds), run_time=0.7)
            release_dots_recursive(VGroup(lis, lds))
        self.wait(1)
        self.play(
            self.camera.frame.animate.restore(),
            FadeOut(VGroup(inequality, value_expr, lis_value, lds_value))
        )

        # Show (LIS, LDS) = (N, 1)
        new_bar_heights = []
        for i in range(1, n + 1):
            new_bar_heights.append(i)
        permutation = [values.index(i) for i in new_bar_heights]
        for i, bar in enumerate(bars):
            bars[permutation[i]].generate_target()
            bars[permutation[i]].target.match_x(bar)
        self.play(AnimationGroup(*[MoveToTarget(bar) for bar in bars]))
        left_to_right = sorted(zip(values, bars), key=lambda triple: triple[1].get_x())
        values, bars = [v for v, _ in left_to_right], VGroup(*[b for _, b in left_to_right])
        base.add_updater(lambda m: self.bring_to_front(m))

        # List the numbers in order from 1 up to n
        nums = VGroup(*[
            (
                Tex(str(i)).set_color(nums_color) if i < n - 1 else
                Tex(R"\cdots").set_color(nums_color).scale(0.7) if i == n - 1 else
                Tex("N").set_color(YELLOW)
            ).next_to(bars[i - 1], UP, buff=0.35)
            for i in range(1, n + 1)
        ])
        self.play(AnimationGroup(*[FadeIn(num, shift=UP * 0.2) for num in nums], lag_ratio=0.2), run_time=3)
        self.wait(1)

        all_increasing_indices = maximal_longest_increasing_subsequences(values, include_length_1=True)
        all_decreasing_indices = maximal_longest_decreasing_subsequences(values, include_length_1=True)

        increasing_paths = VGroup(*[
            sequence_path(bars, indices, increasing_sequence_color)
            for indices in all_increasing_indices
        ])
        decreasing_paths = VGroup(*[
            sequence_path(bars, indices, decreasing_sequence_color)
            for indices in all_decreasing_indices
        ])

        self.play(
            AnimationGroup(*[
                path_creation_animation(path)
                for path in increasing_paths
            ], lag_ratio=0.1)
        )
        self.wait(0.4)
        self.play(
            AnimationGroup(*[
                path_creation_animation(path)
                for path in decreasing_paths
            ], lag_ratio=0.1)
        )

        # Save the pair (n, 1)
        pair1 = Tex("(N, 1)", font_size=37).to_corner(UL, buff=0.8)
        pair1[1:-3].set_color(increasing_sequence_color)
        pair1[-2].set_color(decreasing_sequence_color)
        table_label = Tex(
            R"(\text{LIS},\ \text{LDS})",
            tex_to_color_map={"LIS": increasing_sequence_color, "LDS": decreasing_sequence_color}
        ).set_width(
            pair1.get_width() * 1.2
        ).next_to(pair1, UP)
        self.play(Write(pair1), FadeIn(table_label))
        self.play(FadeOut(decreasing_paths), FadeOut(increasing_paths), FadeOut(nums))
        release_dots_recursive(VGroup(increasing_paths, decreasing_paths))

        # Show (LIS, LDS) = (N/2, 2)
        new_bar_heights = []
        for i in range(n // 2):
            new_bar_heights.append(2 * (i + 1))
        for i in range(n // 2):
            new_bar_heights.append(2 * i + 1)
        permutation = [values.index(i) for i in new_bar_heights]
        for i, bar in enumerate(bars):
            bars[permutation[i]].generate_target()
            bars[permutation[i]].target.match_x(bar)
        self.play(AnimationGroup(*[MoveToTarget(bar) for bar in bars]))

        # List the numbers
        nums = VGroup(*[
            Tex(str(i)).set_color(nums_color).next_to(bars[i - 1], UP, buff=0.35)
            for i in range(1, n + 1)
        ])
        nums_left_to_right = sorted(nums, key=lambda m: m.get_x())
        self.play(AnimationGroup(*[FadeIn(num, shift=UP * 0.2) for num in nums_left_to_right], lag_ratio=0.2), run_time=5)
        self.wait(1)

        left_to_right = sorted(zip(values, bars), key=lambda triple: triple[1].get_x())
        values, bars = [v for v, _ in left_to_right], VGroup(*[b for _, b in left_to_right])
        base.add_updater(lambda m: self.bring_to_front(m))

        # Show the new paths
        all_increasing_indices = maximal_longest_increasing_subsequences(values, include_length_1=True)
        all_decreasing_indices = maximal_longest_decreasing_subsequences(values, include_length_1=True)

        increasing_paths = VGroup(*[
            sequence_path(bars, indices, increasing_sequence_color)
            for indices in all_increasing_indices
        ])
        decreasing_paths = VGroup(*[
            sequence_path(bars, indices, decreasing_sequence_color)
            for indices in all_decreasing_indices
        ])

        length_labels_increasing = VGroup(*[
            Tex(R"\frac{N}{2}").set_color(GREEN).next_to(path.get_center(), DR).set_stroke(width=8, color=BLACK, behind=True)
            for path in increasing_paths
        ])
        self.play(
            AnimationGroup(*[
                path_creation_animation(path)
                for path in increasing_paths
            ], lag_ratio=0.1)
        )
        self.wait(1)
        self.play(
            AnimationGroup(*[
                FadeIn(label, shift=RIGHT * 0.3, run_time=2)
                for label in length_labels_increasing
            ], lag_ratio=0.1)
        )
        self.wait(0.4)
        length_labels_decreasing = VGroup(*[
            Tex(R"2").set_color(RED).next_to(path.get_center(), UR).set_stroke(width=8, color=BLACK, behind=True)
            for path in decreasing_paths
        ])
        self.play(
            AnimationGroup(*[
                path_creation_animation(path)
                for path in decreasing_paths
            ], lag_ratio=0.1)
        )
        self.wait(1)
        self.play(
            AnimationGroup(*[
                FadeIn(label, shift=UP * 0.3, run_time=2)
                for label in length_labels_decreasing
            ], lag_ratio=0.1)
        )

        # Save the pair (N/2, 2)
        pair2 = Tex(R"\left(\frac{N}{2}, 2\right)").match_width(pair1).next_to(pair1, DOWN)
        pair2[1:-3].set_color(increasing_sequence_color)
        pair2[-2].set_color(decreasing_sequence_color)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    TransformMatchingShapes(length_labels_increasing.copy(), pair2[1:-3]),
                    TransformMatchingShapes(length_labels_decreasing.copy(), pair2[-2])
                ),
                FadeIn(VGroup(pair2[0], pair2[-3], pair2[-1])), lag_ratio=0.7)
        )
        self.wait(1)
        self.play(FadeOut(VGroup(decreasing_paths, length_labels_decreasing)))
        release_dots_recursive(decreasing_paths)
        self.wait(2)

        # Show the case where N is odd
        bar_11 = bars[0].copy().stretch_to_fit_height(
            bars[5].get_height() * 11
        ).next_to(
            bars, RIGHT, buff=bars[1].get_left()[0] - bars[0].get_right()[0]
        ).align_to(
            bars, DOWN
        )
        bars.generate_target()
        bar_11.generate_target()
        VGroup(bars.target, bar_11.target).match_width(bars).match_x(bars).align_to(bars, DOWN)
        bar_11.set_opacity(0)
        bars.save_state()
        bar_11.save_state()

        all_increasing_indices_new = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9, 10]
        ]
        increasing_paths_new = VGroup(*[
            sequence_path(VGroup(*bars.target, bar_11.target), indices, increasing_sequence_color)
            for indices in all_increasing_indices_new
        ])
        length_labels_increasing_new = VGroup(*[
            Tex(R"\frac{N}{2}" if i == 0 else R"\left\lceil\frac{N}{2}\right\rceil").set_color(
                GREEN
            ).next_to(
                path.get_center(), DR
            ).set_stroke(
                width=8, color=BLACK, behind=True
            )
            for i, path in enumerate(increasing_paths_new)
        ])
        increasing_paths.set_z_index(100)
        new_values = [2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]
        nums = VGroup(*[sorted(nums, key=lambda m: m.get_x())])
        new_nums = VGroup(*[Tex(str(value)).set_color(nums_color).next_to(bar, UP) for value, bar in zip(new_values, list(bars.target) + [bar_11.target])])
        self.play(
            MoveToTarget(bars),
            MoveToTarget(bar_11),
            ReplacementTransform(increasing_paths, increasing_paths_new, run_time=1),
            TransformMatchingShapes(length_labels_increasing, length_labels_increasing_new, run_time=1),
            ReplacementTransform(nums, new_nums[:-1]),
            FadeIn(new_nums[-1], shift=DL * 0.5)
        )
        release_dots_recursive(increasing_paths)
        self.wait(2)
        bars = VGroup(*bars, bar_11)
        values += [11]
        self.play(FadeOut(VGroup(increasing_paths_new, length_labels_increasing_new, new_nums)))

        # Show (LIS, LDS) = (N/3, 3)
        n = 11
        k = 3
        new_bar_heights = []
        for j in range(k):
            new_bar_heights += [i for i in range(1, n + 1) if (i + j) % k == 0]

        permutation = [values.index(i) for i in new_bar_heights]
        for i, bar in enumerate(bars):
            bars[permutation[i]].generate_target()
            bars[permutation[i]].target.match_x(bar)
        self.play(AnimationGroup(*[MoveToTarget(bar) for bar in bars]))
        self.wait(1)

        left_to_right = sorted(zip(values, bars), key=lambda triple: triple[1].get_x())
        values, bars = [v for v, _ in left_to_right], VGroup(*[b for _, b in left_to_right])
        base.add_updater(lambda m: self.bring_to_front(m))

        all_increasing_indices = []
        section = []
        start_index = 0
        for i in range(k):
            finish_index = start_index + (n // k if i < (-n) % k else n // k + 1)
            section.append([new_bar_heights.index(h) for h in new_bar_heights[start_index:finish_index]])
            start_index = finish_index
        all_increasing_indices += section
        all_decreasing_indices = maximal_longest_decreasing_subsequences(values, include_length_1=True)

        increasing_paths = VGroup(*[
            sequence_path(bars, indices, increasing_sequence_color)
            for indices in all_increasing_indices
        ])
        decreasing_paths = VGroup(*[
            sequence_path(bars, indices, decreasing_sequence_color)
            for indices in all_decreasing_indices
        ])

        length_labels_increasing = VGroup(*[
            Tex(R"\frac{N}{" + str(k) + "}" if n % k > i + 1 else R"\left\lceil\frac{N}{" + str(k) + R"}\right\rceil", font_size=35).set_color(
                GREEN
            ).next_to(
                path.get_center(), LEFT
            ).set_stroke(
                width=8, color=BLACK, behind=True
            )
            for i, path in enumerate(increasing_paths)
        ])
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    path_creation_animation(path),
                    FadeIn(label, shift=RIGHT * 0.3), lag_ratio=0.3)
                for path, label in zip(increasing_paths, length_labels_increasing)
            ], lag_ratio=0.1)
        )
        self.wait(0.4)
        length_labels_decreasing = VGroup(*[
            Tex(str(len(indices))).set_color(RED).next_to(path.get_center(), UR).set_stroke(width=8, color=BLACK, behind=True)
            for path, indices in zip(decreasing_paths, all_decreasing_indices)
        ])
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    path_creation_animation(path),
                    FadeIn(label, shift=UP * 0.3), lag_ratio=0.3)
                for path, label in zip(decreasing_paths, length_labels_decreasing)
            ], lag_ratio=0.1)
        )
        self.wait(1)

        # Save the pair (N/3, 3)
        pair3 = Tex(R"\left(\frac{N}{" + str(k) + "}, " + str(k) + R"\right)").match_width(pair1).next_to(pair2, DOWN)
        pair3[1:-3].set_color(increasing_sequence_color)
        pair3[-2].set_color(decreasing_sequence_color)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    TransformMatchingShapes(length_labels_increasing.copy(), pair3[1:-3]),
                    TransformMatchingShapes(length_labels_decreasing.copy(), pair3[-2])
                ),
                FadeIn(VGroup(pair3[0], pair3[-3], pair3[-1])), lag_ratio=0.7)
        )
        self.wait(1)
        self.play(FadeOut(VGroup(decreasing_paths, length_labels_decreasing, increasing_paths, length_labels_increasing)))
        release_dots_recursive(VGroup(increasing_paths, decreasing_paths))
        self.wait(0.3)

        # Show the entire family of examples
        pairs = VGroup(table_label, pair1, pair2, pair3)
        for k in range(4, 7):
            self.remove(VGroup(decreasing_paths, increasing_paths))
            release_dots_recursive(VGroup(decreasing_paths, increasing_paths))
            new_bar_heights = []
            for j in range(k):
                new_bar_heights += [i for i in range(1, n + 1) if (i + j) % k == 0]

            permutation = [values.index(i) for i in new_bar_heights]
            bars.generate_target()
            for i, bar in enumerate(bars):
                bars.target[permutation[i]].match_x(bar)
            bars.become(bars.target)

            left_to_right = sorted(zip(values, bars), key=lambda triple: triple[1].get_x())
            values, bars = [v for v, _ in left_to_right], VGroup(*[b for _, b in left_to_right])
            base.add_updater(lambda m: self.bring_to_front(m))

            all_increasing_indices = []
            section = []
            start_index = 0
            for i in range(k):
                finish_index = start_index + (n // k if i < (-n) % k else n // k + 1)
                section.append([new_bar_heights.index(h) for h in new_bar_heights[start_index:finish_index]])
                start_index = finish_index
            all_increasing_indices += section
            all_decreasing_indices = maximal_longest_decreasing_subsequences(values, include_length_1=True)

            increasing_paths = VGroup(*[
                sequence_path(bars, indices, increasing_sequence_color)
                for indices in all_increasing_indices
            ])
            decreasing_paths = VGroup(*[
                sequence_path(bars, indices, decreasing_sequence_color)
                for indices in all_decreasing_indices
            ])
            for path in increasing_paths:
                prepare_path_dots(path)
            for path in decreasing_paths:
                prepare_path_dots(path)

            next_pair = Tex(R"\left(\frac{N}{" + str(k) + "}, " + str(k) + R"\right)").match_width(pair1).next_to(pairs[-1], DOWN)
            next_pair[1:-3].set_color(increasing_sequence_color)
            next_pair[-2].set_color(decreasing_sequence_color)
            pairs.add(next_pair)
            self.add(increasing_paths, decreasing_paths)
            self.play(FadeIn(next_pair), run_time=0.1)
            self.wait(1)
        vdots = Tex(R"\vdots").next_to(pairs, DOWN)
        self.play(Write(vdots, run_time=0.7))

        # Conjecture that LIS*LDS >= N
        base.clear_updaters()
        chart = VGroup(bars, base)
        self.play(
            FadeOut(VGroup(chart, increasing_paths, decreasing_paths), shift=RIGHT * 3),
            VGroup(pairs, vdots).animate.scale(1.2).set_y(0).set_x(1), run_time=2)
        release_dots_recursive(VGroup(increasing_paths, decreasing_paths))
        brace = Brace(pairs[1:], RIGHT)
        label = brace.get_tex(
            R"\text{LIS} \cdot \text{LDS} \ge N",
            tex_to_color_map={"LIS": increasing_sequence_color, "LDS": decreasing_sequence_color, "N": n_color},
            font_size=50,
            buff=0.3
        )
        self.play(GrowFromEdge(brace, LEFT), Write(label), run_time=2)
        self.wait(2)

        # Build a square k^2-bar chart in the residue-block pattern (shared by k = 3..6)
        frame = self.camera.frame
        bottom_margin = 0.5
        brace_headroom = 1.3  # room for the N-brace/label above the chart
        square_side = frame.get_height() - bottom_margin - brace_headroom
        bars_bottom_y = frame.get_bottom()[1] + bottom_margin

        def build_square_chart(k):
            n_local = k**2
            heights = []
            for j in range(k):
                heights += [i for i in range(1, n_local + 1) if (i + j) % k == 0]
            bw = square_side / (n_local - 0.1)
            new_bars = VGroup(*[
                Rectangle(
                    width=bw * 0.9,
                    height=(h / n_local) * square_side,
                    fill_opacity=1,
                    fill_color=bar_color,
                    stroke_width=0,
                )
                for h in heights
            ]).arrange(RIGHT, buff=bw * 0.1, aligned_edge=DOWN)
            new_bars.move_to(ORIGIN)
            new_bars.set_y(bars_bottom_y + square_side / 2)
            new_base = Line(LEFT, RIGHT).set_width(new_bars.get_width() * 1.05)
            new_base.match_x(new_bars)
            new_base.set_y(bars_bottom_y)
            new_chart = VGroup(new_bars, new_base)
            return new_chart, new_bars, new_base, heights

        def add_square_braces(new_bars):
            n_brace = Brace(new_bars, UP)
            n_label = n_brace.get_tex(
                "N = k^2",
                tex_to_color_map={"N": n_color, "k": TEAL},
                font_size=50
            )
            return VGroup(n_brace, n_label)

        def add_square_subsequences(new_bars, heights, k):
            # Each residue class is a contiguous block of exactly k bars
            # (since n_local = k^2 divides evenly by k), so every maximal
            # increasing run is just a consecutive chunk of size k, and by
            # construction every maximal decreasing run also has length k.
            all_increasing_indices = [
                list(range(block * k, (block + 1) * k))
                for block in range(k)
            ]
            all_decreasing_indices = maximal_longest_decreasing_subsequences(heights, include_length_1=True)

            increasing_paths = VGroup(*[
                sequence_path(new_bars, indices, increasing_sequence_color)
                for indices in all_increasing_indices
            ])
            decreasing_paths = VGroup(*[
                sequence_path(new_bars, indices, decreasing_sequence_color)
                for indices in all_decreasing_indices
            ])
            for path in increasing_paths:
                prepare_path_dots(path)
            for path in decreasing_paths:
                prepare_path_dots(path)

            return VGroup(increasing_paths, decreasing_paths)

        # Bring in the chart already arranged as n = 9 = 3^2 in the k=3 pattern
        base.clear_updaters()
        new_chart, bars, base, heights = build_square_chart(3)
        current_chart = new_chart
        current_braces = add_square_braces(bars)
        current_subsequences = add_square_subsequences(bars, heights, 3)
        values = heights
        n = 9
        self.play(
            AnimationGroup(
                FadeOut(VGroup(pairs, vdots, brace, label), shift=LEFT * 7, run_time=2),
                FadeIn(VGroup(new_chart, current_braces, current_subsequences), shift=LEFT * 7, run_time=1.5), lag_ratio=0.3)
        )
        self.wait(0.5)

        # Now show a square chart of k^2 bars in this same pattern, for k = 4, 5, 6
        for k in range(4, 7):
            new_chart, new_bars, new_base, heights = build_square_chart(k)
            self.remove(current_chart)
            self.add(new_chart)
            self.remove(current_braces)
            current_braces = add_square_braces(new_bars)
            self.add(current_braces)
            self.remove(current_subsequences)
            release_dots_recursive(current_subsequences)
            current_subsequences = add_square_subsequences(new_bars, heights, k)
            for sequences in current_subsequences:
                for s in sequences:
                    s[1].set_stroke(width=4)
            self.add(current_subsequences)
            current_chart = new_chart
            bars, base, values = new_bars, new_base, heights
            n = k**2
            self.wait(1)

        # Fade it to a grid
        grid = OptimalGrid(6).match_width(bars).align_to(bars, DOWN)
        grid.tiles.set_stroke(width=3)
        for hole in grid.holes:
            hole.border.set_stroke(width=3, color=WHITE)
        self.play(FadeIn(grid.holes.set_z_index(100)), FadeOut(current_subsequences), run_time=2)
        release_dots_recursive(current_subsequences)
        self.play(FadeOut(new_chart), FadeIn(VGroup(grid.background, grid.tiles)), run_time=2)
        self.wait(1)

        # Ambiently flip through random permutations, landing on the main example for the rest of the scene
        self.clear()
        dot_stacks.clear()
        self.camera.frame.restore()
        n = 9
        chart_width = 6
        chart_height = 5.8
        bar_width = chart_width / n
        final_heights = [6, 5, 2, 4, 1, 9, 8, 7, 3]

        values = list(np.random.permutation(list(range(1, n + 1))))
        bars = VGroup(*[
            Rectangle(
                width=bar_width * 0.9,
                height=(h / n) * chart_height,
                fill_opacity=1,
                fill_color=bar_color,
                stroke_width=0,
            )
            for h in values
        ]).arrange(RIGHT, buff=bar_width * 0.1, aligned_edge=DOWN)
        bars.move_to(ORIGIN).to_edge(DOWN, buff=1.0)
        base = Line(LEFT, RIGHT).set_width(bars.get_width() * 1.05).match_x(bars).next_to(bars, DOWN, buff=0)
        chart = VGroup(bars, base)

        self.add(chart)
        base.add_updater(lambda m: self.bring_to_front(m))
        self.wait(0.5)

        num_lead_in_iters = 45
        for i in range(num_lead_in_iters):
            is_last = (i == num_lead_in_iters - 1)
            if is_last:
                new_bar_heights = final_heights
            else:
                new_bar_heights = list(np.random.permutation(list(range(1, n + 1))))
                while new_bar_heights == values:
                    new_bar_heights = list(np.random.permutation(list(range(1, n + 1))))
            permutation = [values.index(h) for h in new_bar_heights]
            for i2, bar in enumerate(bars):
                bars[permutation[i2]].generate_target()
                bars[permutation[i2]].target.match_x(bar)
            self.play(AnimationGroup(*[MoveToTarget(bar) for bar in bars]), run_time=0.6)
            left_to_right = sorted(zip(values, bars), key=lambda pair: pair[1].get_x())
            values, bars = [v for v, _ in left_to_right], VGroup(*[b for _, b in left_to_right])
            base.add_updater(lambda m: self.bring_to_front(m))

            lis_indices = longest_increasing_subsequence_indices(values)
            lds_indices = longest_decreasing_subsequence_indices(values)
            lis_path = sequence_path(bars, lis_indices, increasing_sequence_color)
            lds_path = sequence_path(bars, lds_indices, decreasing_sequence_color)
            self.play(
                path_creation_animation(lis_path),
                path_creation_animation(lds_path), run_time=0.9)
            self.wait(0.5)
            self.play(FadeOut(lis_path), FadeOut(lds_path), run_time=0.5)
            release_dots_recursive(VGroup(lis_path, lds_path))

        heights = values
        nums = VGroup(*[
            Integer(h, font_size=40).set_color(nums_color).next_to(bar, UP, buff=0.38)
            for h, bar in zip(heights, bars)
        ])
        self.play(FadeIn(nums))
        base.add_updater(lambda m: self.bring_to_front(m))
        self.wait(1)

        # Focus on one of the bars
        focus_index = 3
        focus_bar = bars[focus_index]
        arrow = Arrow(ORIGIN, DOWN * 1.5, thickness=5).set_color(YELLOW).next_to(focus_bar, UP, buff=1.5)
        self.play(
            AnimationGroup(*[
                VGroup(bar, num).animate.set_opacity(0.1)
                for bar, num in zip(bars[focus_index + 1:], nums[focus_index + 1:])
            ]),
            GrowArrow(arrow)
        )

        # Highlight its longest increasing and decreasing subsequences
        increasing_indices = [2, 3]
        increasing_path = sequence_path(bars, increasing_indices, increasing_sequence_color)
        self.play(path_creation_animation(increasing_path))
        self.wait(1)
        lis_text = Tex(R"\text{LIS}: 2", font_size=110).set_color(increasing_sequence_color)
        lds_text = Tex(R"\text{LDS}: 3", font_size=110).set_color(decreasing_sequence_color)
        lds_text.next_to(lis_text, DOWN, buff=0.6).align_to(lis_text, LEFT)
        VGroup(lis_text, lds_text).set_y(0).to_edge(RIGHT, buff=1.5)
        base.suspend_updating()
        self.play(
            AnimationGroup(
                VGroup(chart, nums, arrow, increasing_path).animate.to_edge(LEFT, buff=1.5),
                Write(lis_text), lag_ratio=0.6, run_time=1.5)
        )
        base.resume_updating()
        self.wait(1)
        decreasing_indices = [0, 1, 3]
        decreasing_path = sequence_path(bars, decreasing_indices, decreasing_sequence_color)
        self.play(path_creation_animation(decreasing_path))
        self.wait(1)
        self.play(Write(lds_text), run_time=1.5)
        self.wait(1)
        self.wait(2)

        # Save the values as a pair of numbers below the bar
        pair = Tex("(2, 3)", font_size=30).next_to(focus_bar, DOWN)
        pair[1].set_color(increasing_sequence_color)
        pair[3].set_color(decreasing_sequence_color)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    TransformFromCopy(lis_text[-1], pair[1]),
                    TransformFromCopy(lds_text[-1], pair[3]), run_time=2),
                FadeIn(VGroup(pair[0], pair[2], pair[4])), lag_ratio=0.7)
        )
        self.wait(2)

        # Switch focus back to the full chart
        chart = VGroup(bars, base)
        base.clear_updaters()
        chart.generate_target()
        chart.target.set_opacity(1).stretch(1.5, 0).center()
        nums.generate_target()
        nums.target.set_opacity(1)
        for num, bar in zip(nums.target, chart.target[0]):
            num.match_x(bar)
        pair.generate_target()
        pair.target.match_x(chart.target[0][3]).scale(1.3)

        increasing_path.set_z_index(100)
        decreasing_path.set_z_index(100)
        self.play(
            FadeOut(arrow, shift=UP),
            FadeOut(VGroup(lis_text, lds_text), run_time=1),
            MoveToTarget(chart, run_time=2),
            MoveToTarget(nums, run_time=2),
            MoveToTarget(pair, run_time=2),
            FadeOut(increasing_path, shift=RIGHT * 0.08, run_time=0.6),
            FadeOut(decreasing_path, shift=RIGHT * 0.08, run_time=0.6)
        )
        release_dots_recursive(VGroup(increasing_path, decreasing_path))
        base.add_updater(lambda m: self.bring_to_front(m))
        self.wait(2)

        # Show that all the numbers are distinct
        self.remove(pair)
        self.wait(1)
        circles = VGroup(*[Circle(radius=0.35, fill_opacity=0, stroke_width=3, stroke_color=YELLOW).move_to(num) for num in nums])
        self.play(AnimationGroup(*[ShowCreation(circle) for circle in circles], lag_ratio=0.1))
        self.wait(2)
        self.play(FadeOut(circles))

        # Add the (LIS, LDS) pair for each bar
        lis_lds_lengths = [(1, 1), (1, 2), (1, 3), (2, 3), (1, 4), (3, 1), (3, 2), (3, 3), (2, 4)]
        pairs = VGroup(*[
            Tex(F"({lis}, {lds})").match_height(pair).match_y(pair).match_x(bar)
            for (lis, lds), bar in zip(lis_lds_lengths, bars)
        ])
        self.add(pairs[focus_index])
        for i, pair in enumerate(pairs):
            pair[1].set_color(increasing_sequence_color)
            pair[3].set_color(decreasing_sequence_color)
            pair.save_state()
            if i != focus_index:
                pair.scale(1.2).set_opacity(0)
            else:
                pair.set_opacity(1)
        self.play(
            AnimationGroup(*[
                pair.animate.restore()
                for pair in list(pairs[:focus_index]) + list(pairs[focus_index + 1:])
            ], lag_ratio=0.2), run_time=3.6)
        self.wait(2)

        # Do another example
        bars.save_state()
        nums.save_state()
        pairs.save_state()
        focus_index = 6
        focus_bar = bars[focus_index]
        arrow = Arrow(ORIGIN, DOWN * 1.1, thickness=4).set_color(YELLOW).next_to(focus_bar, UP, buff=0.85)
        self.play(
            VGroup(
                *[
                    VGroup(bar, num)
                    for bar, num in zip(bars[focus_index + 1:], nums[focus_index + 1:])
                ],
                pairs[:focus_index],
                pairs[focus_index + 1:]
            ).animate.set_opacity(0.1),
            GrowArrow(arrow)
        )
        increasing_indices = [2, 3, 6]
        increasing_path = sequence_path(bars, increasing_indices, increasing_sequence_color)
        self.play(path_creation_animation(increasing_path))
        self.wait(1)
        decreasing_indices = [5, 6]
        decreasing_path = sequence_path(bars, decreasing_indices, decreasing_sequence_color)
        self.play(path_creation_animation(decreasing_path))
        self.wait(1)
        self.play(
            bars.animate.restore(), nums.animate.restore(), pairs.animate.restore(),
            FadeOut(arrow), FadeOut(increasing_path), FadeOut(decreasing_path), run_time=2
        )
        release_dots_recursive(VGroup(increasing_path, decreasing_path))
        self.wait(1)

        # Indicate pairs to show uniqueness
        self.play(AnimationGroup(*[Indicate(pair) for pair in pairs], lag_ratio=0.1), run_time=3)

        # Save the full chart
        original_chart_group = VGroup(chart, nums, pairs).copy()

        # Bring in an arbitrary pair of bars
        bar1 = bars[1].copy()
        bar2 = bar1.copy()
        VGroup(bar1, bar2).arrange(buff=2).align_to(bars[0], DOWN)
        self.play(
            AnimationGroup(
                FadeOut(VGroup(bars, nums, pairs)),
                FadeIn(VGroup(bar1, bar2)), lag_ratio=0.2), run_time=3)
        self.wait(2)

        # Write an arbitrary pair of values for the LIS and LDS for that bar
        pair = Tex("(x, y)", tex_to_color_map={"x": increasing_sequence_color, "y": decreasing_sequence_color}).match_height(pairs[0]).match_y(pairs[0]).match_x(bar1)
        self.play(FadeIn(pair))

        # Make the second bar taller
        stretch_factor = 1.2
        self.play(
            bar1.animate.stretch(1 / stretch_factor, 1).align_to(bar1, DOWN),
            bar2.animate.stretch(stretch_factor, 1).align_to(bar2, DOWN), run_time=0.7)
        self.play(
            bar1.animate.stretch(stretch_factor**2, 1).align_to(bar1, DOWN),
            bar2.animate.stretch(1 / stretch_factor**2, 1).align_to(bar2, DOWN), run_time=0.7)
        self.play(
            bar1.animate.stretch(1 / stretch_factor**3, 1).align_to(bar1, DOWN),
            bar2.animate.stretch(stretch_factor**3, 1).align_to(bar2, DOWN), run_time=2)
        pair2 = Tex("(x', y')", tex_to_color_map={"x'": increasing_sequence_color, "y'": decreasing_sequence_color}).match_height(pairs[0]).match_y(pairs[0]).match_x(bar2)
        self.play(FadeIn(pair2))
        self.wait(1)

        # Show a generic tail of bars behind the first bar
        heights_tail = [2, 6, 5, 3]
        heights_tail = [h * 0.6 for h in heights_tail]
        tail = VGroup(*[
            bar1.copy().stretch_to_fit_width(0.5).stretch_to_fit_height(height)
            for height in heights_tail
        ]).arrange(
            buff=0.1
        ).next_to(
            bar1, LEFT, buff=0.2
        )
        tail_opacity = 0.4
        for bar in tail:
            bar.align_to(
                bar1, DOWN
            ).set_opacity(
                tail_opacity
            )
        cdots = Tex(R"\cdots", font_size=100).match_y(bar1)

        self.play(
            AnimationGroup(
                *[
                    FadeIn(bar)
                    for bar in tail
                ],
                Write(cdots, run_time=1.5), lag_ratio=0.1),
        )
        self.wait(2)

        # Show the increasing subsequence of length x
        increasing_sequence = VGroup(tail[0], tail[3], bar1)
        increasing_path = sequence_path(increasing_sequence, range(len(increasing_sequence)), increasing_sequence_color)
        brace = Brace(increasing_sequence, UP).shift(UP * 0.1)
        label = brace.get_tex("x", font_size=60).set_color(increasing_sequence_color).shift(UP * 0.2)
        self.play(
            AnimationGroup(*[
                bar.animate.set_opacity(1)
                for bar in increasing_sequence
            ], lag_ratio=0.1),
            path_creation_animation(increasing_path),
            GrowFromEdge(brace, DOWN),
            Write(label)
        )
        self.wait(2)

        # Extend the increasing sequence to the second bar
        brace.generate_target()
        label.generate_target()
        extended_brace = Brace(VGroup(increasing_sequence, bar2), UP).shift(UP * 0.1)
        extended_label = extended_brace.get_tex(R"x' \ge x + 1", font_size=60).set_color(increasing_sequence_color).shift(UP * 0.2)
        part1 = extended_label[:3]
        part2 = extended_label[3:]
        part2.save_state()
        part2.match_x(extended_brace)
        extended_increasing_sequence = VGroup(*increasing_sequence, bar2)
        extended_increasing_path = sequence_path(extended_increasing_sequence, range(len(extended_increasing_sequence)), increasing_sequence_color)
        self.play(
            TransformFromCopy(brace, extended_brace),
            TransformMatchingShapes(label.copy(), part2),
            Transform(increasing_path.set_z_index(100), extended_increasing_path.set_z_index(100)), run_time=2)
        self.wait(1)
        self.play(part2.animate.restore(), FadeIn(part1, shift=RIGHT * 0.5))

        # Save the example
        case1 = VGroup(
            tail, bar1, bar2, increasing_path, base, pair, pair2,
            brace, label, extended_brace, extended_label, cdots
        ).copy()

        # Make the second bar shorter
        increasing_path.set_z_index(100)
        self.play(
            FadeOut(VGroup(brace, label, extended_brace, extended_label)),
            FadeOut(increasing_path),
            tail.animate.set_opacity(tail_opacity),
            bar2.animate.stretch_to_fit_height(0.6 * bar1.get_height()).align_to(bar2, DOWN)
        )
        self.wait(2)

        # Show the decreasing subsequence of length y
        decreasing_sequence = VGroup(tail[1], tail[2], bar1)
        decreasing_path = sequence_path(decreasing_sequence, range(len(decreasing_sequence)), decreasing_sequence_color)
        brace = Brace(decreasing_sequence, UP)
        label = brace.get_tex("y", font_size=60).set_color(decreasing_sequence_color)
        self.play(
            AnimationGroup(*[
                bar.animate.set_opacity(1)
                for bar in decreasing_sequence
            ], lag_ratio=0.1),
            path_creation_animation(decreasing_path),
            GrowFromEdge(brace, DOWN),
            Write(label)
        )
        self.wait(2)

        # Extend the decreasing sequence to the second bar
        brace.generate_target()
        label.generate_target()
        extended_brace = Brace(VGroup(decreasing_sequence, bar2), UP).align_to(case1[-3], UP)
        extended_label = extended_brace.get_tex(R"y' \ge y + 1", font_size=60).set_color(decreasing_sequence_color).align_to(case1[-2], UP)
        part1 = extended_label[:3]
        part2 = extended_label[3:]
        part2.save_state()
        part2.match_x(extended_brace)
        extended_decreasing_sequence = VGroup(*decreasing_sequence, bar2)
        extended_decreasing_path = sequence_path(extended_decreasing_sequence, range(len(extended_decreasing_sequence)), decreasing_sequence_color)
        self.play(
            TransformFromCopy(brace, extended_brace),
            TransformMatchingShapes(label.copy(), part2),
            Transform(decreasing_path, extended_decreasing_path), run_time=2)
        self.wait(1)
        self.play(part2.animate.restore(), FadeIn(part1, shift=RIGHT * 0.5))

        # Save the second example
        case2 = VGroup(
            tail, bar1, bar2, decreasing_path, base, pair, pair2,
            brace, label, extended_brace, extended_label, cdots
        )

        # Show both examples side by side
        case1.clear_updaters()
        base.clear_updaters()
        case1[1:4].set_z_index(400)
        case2.generate_target()
        VGroup(case1, case2.target).scale(0.63).arrange(buff=0.7)
        case2.target.align_to(case1, DOWN)
        case2[1:4].set_z_index(400)
        case1_label = TexText("Case 1").next_to(case1, DOWN, buff=1)
        case2_label = TexText("Case 2").next_to(case2.target, DOWN, buff=1)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    FadeIn(case1, shift=RIGHT * 5),
                    MoveToTarget(case2)
                ),
                LaggedStartMap(FadeIn, VGroup(case1_label, case2_label), lag_ratio=0.2, shift=UP * 1.5, run_time=0.8), lag_ratio=0.1), run_time=3)

        # Bring back the original chart
        original_chart_group.clear_updaters()
        self.play(
            FadeOut(VGroup(case1, case2, case1_label, case2_label), shift=DOWN * 7),
            FadeIn(original_chart_group, shift=DOWN * 7), run_time=1.5)
        chart, nums, pairs = original_chart_group[0], original_chart_group[1], original_chart_group[2]
        self.wait(0.5)

        # Indicate pairs again to show uniqueness
        self.play(AnimationGroup(*[Indicate(pair) for pair in pairs], lag_ratio=0.1), run_time=3)
        self.wait(2)

        # Put each pair of numbers on a coordinate grid
        number_plane = NumberPlane(
            x_range=[0, 5],
            y_range=[0, 5]
        ).set_width(4.5).to_edge(RIGHT, buff=1)
        number_plane.remove(number_plane.faded_lines)
        x_labels = number_plane.add_coordinate_labels(x_values=[1, 2, 3, 4, 5], y_values=[], font_size=30, direction=DOWN)
        y_labels = number_plane.add_coordinate_labels(x_values=[], y_values=[1, 2, 3, 4, 5], font_size=30, direction=LEFT)
        x_labels.set_color(increasing_sequence_color)
        y_labels.set_color(decreasing_sequence_color)
        points = Group(*[
            Group(GlowDot(), TrueDot()).set_color(n_color).move_to(number_plane.c2p(x, y))
            for (x, y) in lis_lds_lengths
        ])
        point_labels = pairs.copy()
        for point, label in zip(points, point_labels):
            label.scale(0.6).next_to(point, UR, buff=-0.1)
        self.play(
            original_chart_group.animate(run_time=2).scale(0.55).to_edge(LEFT, buff=1),
            FadeIn(number_plane, shift=LEFT * 6, run_time=2),
            AnimationGroup(*[
                AnimationGroup(
                    TransformFromCopy(pair, label, path_arc=-PI * 0.2),
                    FadeIn(point), lag_ratio=0.6, run_time=2 + i * 0.2)
                for i, (point, pair, label) in enumerate(zip(points, pairs, point_labels))
            ])
        )
        self.wait(1)

        # Draw a rectangle bounding the points
        unit_size = number_plane.background_lines[1].get_y() - number_plane.background_lines[0].get_y()
        rect = Rectangle(
            width=unit_size * 3,
            height=unit_size * 4,
            fill_opacity=0.4,
            fill_color=TEAL,
            stroke_width=4,
            stroke_color=TEAL
        ).align_to(number_plane.c2p(0, 0), DL)
        self.bring_to_back(rect)
        self.play(
            DrawBorderThenFill(rect, stroke_width=6), run_time=2)
        self.wait(2)

        # Show the dimensions
        width_brace = Brace(rect, DOWN, buff=0.5)
        width_label = width_brace.get_tex(R"\text{LIS}").set_color(increasing_sequence_color)
        bars, base = chart
        base.add_updater(lambda m: self.bring_to_front(m))

        increasing_indices = [2, 3, 7]
        decreasing_indices = [0, 1, 3, 8]
        increasing_path = sequence_path(bars, increasing_indices, increasing_sequence_color)
        self.play(
            GrowFromEdge(width_brace, UP),
            Write(width_label),
            path_creation_animation(increasing_path)
        )
        self.wait(1)

        height_brace = Brace(rect, LEFT, buff=0.5)
        height_label = height_brace.get_tex(R"\text{LDS}").set_color(decreasing_sequence_color)
        decreasing_path = sequence_path(bars, decreasing_indices, decreasing_sequence_color)
        self.play(
            GrowFromEdge(height_brace, RIGHT),
            Write(height_label),
            path_creation_animation(decreasing_path)
        )
        self.wait(1)

        # Circle the lattice points
        lattice_points = VGroup()
        for i in range(3):
            for j in range(4):
                point = Circle(
                    radius=0.15, stroke_width=3, stroke_color=WHITE
                ).move_to(number_plane.c2p(i + 1, j + 1))
                lattice_points.add(point)
        self.play(AnimationGroup(*[ShowCreation(point) for point in lattice_points], lag_ratio=0.15))

        # Write the inequality up top
        inequality = Tex(
            R"\text{LIS} \cdot \text{LDS} \ge N",
            font_size=64,
            tex_to_color_map={"LIS": increasing_sequence_color, "LDS": decreasing_sequence_color, "N": n_color}
        ).to_edge(UP, buff=0.6)
        self.play(
            AnimationGroup(
                TransformMatchingShapes(width_label.copy(), inequality["LIS"], path_arc=-PI * 0.35),
                TransformMatchingShapes(height_label.copy(), inequality["LDS"], path_arc=-PI * 0.2),
                GrowFromCenter(inequality[R"\cdot"], path_arc=-PI * 0.3),
                Write(inequality[R"\ge N"]), lag_ratio=0.3, run_time=2)
        )

    def set_camera_target_position(
        self,
        theta_degrees=None,
        phi_degrees=None,
        gamma_degrees=None,
        center=None,
        height=None,
        drift_time=2.0,
    ):
        frame = self.camera.frame
        frame.clear_updaters()
        initial_orientation = frame.get_orientation()
        initial_height = frame.get_height()
        initial_eye = frame.get_implied_camera_location()
        target_frame = frame.copy()
        target_frame.reorient(theta_degrees, phi_degrees, gamma_degrees, center, height)
        target_orientation = target_frame.get_orientation()
        target_height = target_frame.get_height()
        target_eye = target_frame.get_implied_camera_location()
        fovy = frame.get_field_of_view()
        slerp = Slerp([0, 1], Rotation.concatenate([initial_orientation, target_orientation]))
        drift_time = max(drift_time, 1e-4)
        elapsed = 0.0

        def update_camera(f, dt):
            nonlocal elapsed
            elapsed += dt
            t = min(elapsed / drift_time, 1.0)
            alpha = smooth(t)
            current_orientation = slerp(alpha)
            current_height = interpolate(initial_height, target_height, alpha)
            current_eye = interpolate(initial_eye, target_eye, alpha)
            focal_distance = 0.5 * current_height / np.tan(0.5 * fovy)
            to_camera = current_orientation.as_matrix().T[2]
            current_center = current_eye - focal_distance * to_camera
            f.set_orientation(current_orientation)
            f.move_to(current_center)
            f.set_height(current_height)
            if t >= 1.0:
                f.remove_updater(update_camera)
        frame.add_updater(update_camera)


class OptimalErdosSzekeres(InteractiveScene):
    def construct(self):
        # Add a grid
        k = 3
        grid = OptimalGrid(k).set_width(6)
        self.add(grid)
        grid.tiles.set_stroke(width=3)
        for hole in grid.holes:
            hole.border.set_color(WHITE)

        # Label N = k^2
        n_color = YELLOW
        k_color = TEAL
        brace = Brace(grid, UP)
        label = brace.get_tex("N = k^2", font_size=60, tex_to_color_map={"N": n_color, "k": k_color}).shift(UP * 0.2)
        self.camera.frame.save_state()
        self.play(
            AnimationGroup(
                self.camera.frame.animate(run_time=1.5).scale(1.1).shift(UP * 0.5),
                AnimationGroup(
                    GrowFromEdge(brace, DOWN),
                    Write(label), lag_ratio=0.2), lag_ratio=0.2)
        )

        # Focus on the Xs
        self.play(
            grid.background.animate.fade(0.9),
            grid.lines.animate.fade(0.9),
            grid.tiles.animate.fade(0.9),
            run_time=2)
        self.wait(2)

        # Show all possible LISs
        increasing_sequence_color = GREEN_D
        decreasing_sequence_color = RED_D
        increasing_sequences = [
            [6, 3, 0],
            [6, 3, 1],
            [6, 3, 2],
            [6, 4, 1],
            [6, 4, 2],
            [6, 5, 2]
        ]
        paths = VGroup()
        for s in increasing_sequences:
            path = VMobject()
            for i in range(len(s) - 1):
                path.append_vectorized_mobject(Line(grid.holes[s[i]].get_center(), grid.holes[s[i + 1]].get_center()))
            paths.add(path)
            path.set_stroke(width=6, color=increasing_sequence_color)
        for path, s in zip(paths, increasing_sequences):
            self.add(path)
            grid.holes.save_state()
            for i in s:
                grid.holes[i].border.set_stroke(color=increasing_sequence_color)
            self.wait(1)
            self.remove(path)
            grid.holes.restore()
        self.wait(2)

        # Increase the size of the grid
        for k in range(4, 7):
            self.remove(grid)
            grid = OptimalGrid(k).set_width(6)
            grid.background.fade(0.9)
            grid.lines.fade(0.9)
            grid.tiles.fade(0.9)
            grid.tiles.set_stroke(width=3)
            grid.holes.set_stroke(width=2)
            for hole in grid.holes:
                hole.border.set_color(WHITE)
            self.add(grid)
            self.wait(0.1)
        self.wait(2)
        grid.save_state()

        # Divide the holes up into k groups of k
        holes_sorted_by_y = VGroup(*sorted(grid.holes, key=lambda hole: hole.get_y()))
        groups_increasing = VGroup(*[
            holes_sorted_by_y[i * k:(i + 1) * k]
            for i in range(k)
        ])
        rect = Rectangle(
            width=5.5,
            height=0.45,
            fill_opacity=0,
            stroke_width=3,
            stroke_color=k_color
        ).round_corners(0.2)
        x, y, _ = groups_increasing[0][-1].get_center() - groups_increasing[0][0].get_center()
        group_rects_increasing = VGroup(*[rect.copy().rotate(np.arctan2(y, x)).move_to(group) for group in groups_increasing])
        self.remove(grid.holes)
        self.add(grid.holes)
        self.play(AnimationGroup(*[FadeIn(rect) for rect in group_rects_increasing], lag_ratio=0.2))

        # Label the groups
        brace1 = Brace(groups_increasing, np.array([x, y, 0])).set_color(k_color)
        label1 = brace1.get_tex("k").set_color(k_color)
        brace_1_group = VGroup(brace1, label1)
        brace_1_group.save_state()
        self.play(FadeIn(brace_1_group))
        self.wait(1)
        brace2 = Brace(groups_increasing, np.array([-y, x, 0])).set_color(k_color)
        label2 = brace2.get_tex("k").set_color(k_color)
        brace_2_group = VGroup(brace2, label2)
        brace_2_group.save_state()
        self.play(FadeIn(brace_2_group))
        self.wait(2)
        self.play(brace_2_group.animate.fade(1))

        # Build up an increasing subsequence
        increasing_sequence = [30, 25, 21, 15, 9, 5]
        increasing_path = VGroup()
        for i in range(len(increasing_sequence) - 1):
            increasing_path.add(
                Line(
                    grid.holes[increasing_sequence[i]].get_center(), grid.holes[increasing_sequence[i + 1]].get_center()
                ).set_stroke(
                    width=5, color=increasing_sequence_color
                )
            )

        # for line in increasing_path:
        #     self.play(ShowCreation(line), run_time = 0.7)

        for line in increasing_path[:-1]:
            self.play(ShowCreation(line), run_time=2)
            self.wait(0.5)
        self.wait(2)

        # # Show a decreasing segment
        # decreasing_segment = Line(
        #     grid.holes[increasing_sequence[-2]].get_center(), grid.holes[increasing_sequence[-2] + 1].get_center()
        # ).set_stroke(
        #     width = 5, color = decreasing_sequence_color
        # )
        # self.play(ShowCreation(decreasing_segment), run_time = 2.2)
        # self.wait(1.5)

        # Complete the real path
        self.play(ShowCreation(increasing_path[-1]), run_time=2)
        self.wait(1)

        # Add labels of 1 through k on top of the holes
        labels_increasing = VGroup(*[
            Tex(
                str(i) if i < k - 1 else R"\cdots" if i == k - 1 else "k"
            ).next_to(
                grid.holes[increasing_sequence[i - 1]], UP + LEFT * 0.1 if i < k - 1 else UP, buff=0.2
            ).set_color(
                k_color
            ).set_stroke(
                width=10, color=BLACK, behind=True
            )
            for i in range(1, k + 1)
        ])
        brace_label_group = VGroup(brace, label)
        brace_label_group.save_state()
        self.play(
            grid.animate.fade(0.8),
            brace_label_group.animate.fade(0.8),
            group_rects_increasing.animate.fade(0.8),
            AnimationGroup(*[FadeIn(label, shift=UP * 0.2) for label in labels_increasing], lag_ratio=0.2)
        )
        self.wait(2)

        # Show the decreasing subsequence case
        holes_sorted_by_x = VGroup(*sorted(grid.holes, key=lambda hole: hole.get_x()))
        groups_decreasing = VGroup(*[
            holes_sorted_by_x[i * k:(i + 1) * k]
            for i in range(k)
        ])
        x, y, _ = groups_decreasing[0][-1].get_center() - groups_decreasing[0][0].get_center()
        group_rects_decreasing = VGroup(*[rect.copy().rotate(np.arctan2(y, x)).move_to(group) for group in groups_decreasing])
        increasing_path_group = VGroup(increasing_path, labels_increasing)
        increasing_path_group.save_state()
        self.remove(grid.holes)
        self.add(grid.holes)
        self.play(
            brace_label_group.animate.restore(),
            grid.animate.restore(),
            FadeOut(group_rects_increasing),
            brace_1_group.animate.fade(1),
            increasing_path_group.animate.fade(1),
            brace_2_group.animate.restore(),
            AnimationGroup(*[FadeIn(rect) for rect in group_rects_decreasing], lag_ratio=0.2)
        )

        # Build up an decreasing subsequence
        decreasing_sequence = [6, 7, 14, 27, 34, 35]
        decreasing_path = VMobject()
        for i in range(len(decreasing_sequence) - 1):
            decreasing_path.append_vectorized_mobject(
                Line(
                    grid.holes[decreasing_sequence[i]].get_center(), grid.holes[decreasing_sequence[i + 1]].get_center()
                )
            )
            decreasing_path.set_stroke(
                width=5, color=decreasing_sequence_color
            )

        labels_decreasing = VGroup(*[
            Tex(
                str(i) if i < k - 1 else R"\cdots" if i == k - 1 else "k"
            ).next_to(
                grid.holes[decreasing_sequence[i - 1]], UP + RIGHT * 0.1
            ).set_color(
                k_color
            ).set_stroke(
                width=10, color=BLACK, behind=True
            )
            for i in range(1, k + 1)
        ])
        self.play(
            ShowCreation(decreasing_path, run_time=3),
            grid.animate.fade(0.8),
            brace_label_group.animate.fade(0.8),
            group_rects_decreasing.animate.fade(0.8),
            AnimationGroup(*[FadeIn(label, shift=UP * 0.2) for label in labels_decreasing], lag_ratio=0.2, run_time=3)
        )
        self.wait(2)

        # Show the product
        equation = Tex(R"k \cdot k = k^2 = N", font_size=80).shift(RIGHT * 7 + UP * 0.7)
        equation["k"][0].set_color(increasing_sequence_color)
        equation["k"][1].set_color(decreasing_sequence_color)
        equation["k"][2].set_color(k_color)
        equation["N"].set_color(n_color)

        increasing_path_group.generate_target()
        increasing_path_group.target.restore()
        increasing_path_group.target[1].set_fill(color=increasing_sequence_color)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    FadeOut(VGroup(brace_1_group, brace_2_group)),
                    FadeOut(group_rects_decreasing),
                    brace_label_group.animate.restore(),
                    MoveToTarget(increasing_path_group),
                    labels_decreasing.animate.set_fill(color=decreasing_sequence_color),
                    grid.animate.restore(),
                    self.camera.frame.animate.shift(RIGHT * 3.2), run_time=2),
                AnimationGroup(
                    ReplacementTransform(labels_increasing[-1].copy().set_stroke(width=0), equation[0]),
                    GrowFromCenter(equation[1]),
                    ReplacementTransform(labels_decreasing[-1].copy().set_stroke(width=0), equation[2]),
                    FadeIn(equation[3:6]),
                    FadeIn(equation[6:]), lag_ratio=0.4), lag_ratio=0.3)
        )


class LISEqualsK(InteractiveScene):
    def construct(self):
        # Write the equation
        increasing_sequence_color = GREEN_D
        k_color = TEAL
        equation = Tex(
            R"\text{LIS} = 3 = k", font_size=70, tex_to_color_map={"LIS": increasing_sequence_color, "k": k_color}
        ).to_edge(RIGHT, buff=1)
        self.play(Write(equation), run_time=2)
        self.wait(2)

        # Generalize
        generalized_equation = Tex(
            R"\text{LIS} = k", font_size=70, tex_to_color_map={"LIS": increasing_sequence_color, "k": k_color}
        ).move_to(equation)
        self.play(TransformMatchingShapes(equation, generalized_equation))


class PiCreatureReactions(InteractiveScene):
    def construct(self):
        # Add a pi creature
        randy = Randolph(flip_at_start=True)
        self.add(randy)
        self.play(FadeIn(randy, shift=LEFT))

        # React to things
        self.play(randy.change("hooray"), run_time=1)
        self.wait(2)
        self.play(Blink(randy))
        self.wait(4)
        self.play(Blink(randy))
        self.wait(2)
        self.play(randy.change("confused", UP * 4 + LEFT))
        self.wait(1)
        self.play(Blink(randy))
        self.wait(1)
        self.play(randy.change("pondering", LEFT * 3))
        self.wait(2)
        self.play(randy.change("thinking", LEFT * 3))
        self.wait(1)
        self.play(Blink(randy))
        self.wait(3)


class PassingFlashes(InteractiveScene):
    def construct(self):
        # Do some passing flashes
        self.play(VShowPassingFlash(Rectangle(width=4, height=1, stroke_width=4, stroke_color=YELLOW).insert_n_curves(100), time_width=3), run_time=4)
        self.wait(1)
        self.play(VShowPassingFlash(Rectangle(width=4.5, height=1, stroke_width=4, stroke_color=YELLOW).insert_n_curves(100), time_width=3), run_time=4)


class IMODetails(InteractiveScene):
    def construct(self):
        # Write "International Math Olympiad"
        imo_text = TexText("International Math Olympiad", font_size=70).set_opacity(0.9).set_stroke(width=7, color=BLACK, behind=True)
        imo_logo = ImageMobject("IMO_logo").set_opacity(0.9)
        self.play(FadeIn(imo_logo, shift=OUT * 2), Write(imo_text, stroke_color=WHITE))
        self.wait(2)
        imo_text_shortened = TexText("IMO", font_size=100).set_stroke(width=7, color=BLACK, behind=True)
        self.play(TransformMatchingShapes(imo_text, imo_text_shortened), run_time=1.5)
        self.wait(2)

        # Shift it up to the top
        self.play(Group(imo_logo, imo_text_shortened).animate.scale(0.3).to_edge(UP, buff=0.4))
        self.wait(1)

        # Fade in boxes for the problems underneath
        problems = VGroup()
        for i in range(6):
            rect = Rectangle(width=6, height=1.5, fill_opacity=1, fill_color=TEAL_A, stroke_width=0).round_corners(0.2)
            label = TexText(R"\text{Problem }" + str(i + 1)).set_color(BLACK)
            label.set_z_index(1)
            problem = VGroup(rect, label)
            problems.add(problem)
        problems.arrange_in_grid(n_cols=2, h_buff=2, v_buff=0.5, fill_rows_first=False).set_width(10).to_edge(DOWN, buff=1)
        self.play(LaggedStartMap(FadeIn, problems, shift=UP * 0.2, lag_ratio=0.1))

        # Write labels for days underneath
        day1_label = TexText("Day 1").next_to(problems[:3], UP, buff=0.4)
        day2_label = TexText("Day 2").next_to(problems[3:], UP, buff=0.4)
        self.play(Write(day1_label), Write(day2_label))
        self.wait(2)

        # Pi Creatures react to each one's difficulty
        creatures = VGroup(*[
            PiCreature("pondering").match_height(problem).next_to(problem, LEFT).look_at(problem)
            for problem in problems
        ])
        self.play(LaggedStartMap(FadeIn, creatures, shift=RIGHT * 0.2, lag_ratio=0.1))
        self.wait(2)

        expressions = ["pondering", "confused", "horrified"]
        colors = [RED_B, interpolate_color(RED_B, RED_E, 0.5), RED_E]
        hard = TexText("Hard").set_color(colors[0]).match_y(problems[0])
        brutal = TexText("Brutal").set_color(colors[2]).match_y(problems[2])
        arrow = Arrow(ORIGIN, DOWN * 2.2, thickness=3).move_to(VGroup(hard, brutal))
        VGroup(hard, arrow, brutal).next_to(problems, RIGHT)
        self.play(
            AnimationGroup(*[
                creature.change(expressions[i % 3]).look_at(problem)
                for i, (creature, problem) in enumerate(zip(creatures, problems))
            ]),
            AnimationGroup(
                Write(hard),
                GrowArrow(arrow),
                Write(brutal), lag_ratio=0.4),
            AnimationGroup(*[
                problem[0].animate.set_color(colors[i % 3])
                for i, problem in enumerate(problems)
            ], run_time=2)
        )
        self.wait(2)

        # Change the label to "2025 IMO" and arrange the problems on the left
        year_label = TexText(
            "2025"
        ).match_height(
            imo_text_shortened
        ).set_stroke(
            width=7, color=BLACK, behind=True
        ).set_opacity(
            0
        ).next_to(
            imo_text_shortened, LEFT, buff=0.1
        )
        year_label.generate_target()
        year_label.target.set_opacity(1)
        imo_text_shortened.generate_target()
        imo_logo.generate_target()
        VGroup(year_label.target, imo_text_shortened.target).arrange(buff=0.1).move_to(imo_logo.target)
        Group(imo_logo.target, year_label.target, imo_text_shortened.target).scale(1.5).set_y(0).to_edge(RIGHT, buff=1)
        imo_logo.set_z_index(0)
        imo_text_shortened.set_z_index(1)
        year_label.set_z_index(1)

        for problem in problems:
            problem.generate_target()
        VGroup(*[problem.target for problem in problems]).scale(0.9).arrange(DOWN, buff=0.2).to_edge(LEFT, buff=0.6)
        problems[-1].target[0].set_fill(color=PURE_RED).set_stroke(width=4, color=YELLOW)

        self.play(
            MoveToTarget(imo_text_shortened, run_time=2),
            MoveToTarget(imo_logo, run_time=2),
            MoveToTarget(year_label, run_time=2),
            AnimationGroup(*[
                MoveToTarget(problem, path_arc=PI * 0.3 if problem.target.get_y() > 0 else -PI * 0.3)
                for problem in problems
            ], lag_ratio=0.06, run_time=2),
            FadeOut(creatures, shift=LEFT, run_time=1),
            FadeOut(day1_label),
            FadeOut(day2_label),
            FadeOut(VGroup(hard, arrow, brutal), shift=RIGHT)
        )

        # Show scores
        total_participants = 630
        scores_data = [368, 253, 102, 342, 215, 6]
        score_bars = VGroup()
        master_bar_width_tracker = ValueTracker(0)
        for score, problem in zip(scores_data, problems):
            skeleton = Rectangle(
                width=4,
                height=0.5,
                fill_opacity=0.1,
                fill_color=WHITE,
                stroke_width=3,
                stroke_color=WHITE,
                stroke_opacity=1
            ).round_corners(0.08).set_z_index(1)

            def get_bar(score=score, skeleton=skeleton):
                fraction = master_bar_width_tracker.get_value() * score / total_participants
                return Rectangle(
                    fill_opacity=1,
                    fill_color=TEAL_E,
                    stroke_width=0
                ).match_height(
                    skeleton
                ).stretch_to_fit_width(
                    skeleton.get_width() * fraction
                ).move_to(
                    skeleton
                ).align_to(
                    skeleton, LEFT
                ).round_corners(min(0.08, fraction)).set_z_index(0)
            bar = always_redraw(get_bar)
            score_bars.add(VGroup(bar, skeleton).next_to(problem, RIGHT, buff=0.5))
        self.add(score_bars)

        scores = VGroup(*[
            Integer(score, font_size=30).set_color(TEAL).next_to(bar, RIGHT)
            for score, bar in zip(scores_data, score_bars)
        ])
        self.play(
            AnimationGroup(
                AnimationGroup(
                    AnimationGroup(*[
                        FadeIn(skeleton)
                        for (_, skeleton) in score_bars
                    ], run_time=1),
                    master_bar_width_tracker.animate(run_time=2).set_value(1), lag_ratio=0.02),
                AnimationGroup(*[
                    FadeIn(score)
                    for score in scores
                ], lag_ratio=0.1, run_time=2), lag_ratio=0.2)
        )
        score_bars.clear_updaters()

        # Draw attention to the last bar
        arrow = Arrow(ORIGIN, LEFT * 1.5).set_color(YELLOW).next_to(scores[-1], RIGHT)
        self.play(GrowArrow(arrow))


class IMODetailsV2(InteractiveScene):
    def construct(self):
        # Write "International Math Olympiad"
        imo_logo = ImageMobject("IMO_logo").set_opacity(0.9)
        imo_text = TexText("International Math Olympiad", font_size=160).set_opacity(0.9).set_stroke(width=7, color=BLACK, behind=True).next_to(imo_logo, DOWN)
        Group(imo_logo, imo_text).scale(0.3).to_edge(UP, buff=0.3)
        imo_text_shortened = TexText("2025 IMO", font_size=30).set_stroke(width=7, color=BLACK, behind=True).move_to(imo_text)
        self.play(FadeIn(imo_logo, shift=OUT * 2), Write(imo_text, stroke_color=WHITE))

        # Fade in boxes for the problems underneath
        problems = VGroup()
        for i in range(6):
            rect = Rectangle(width=6, height=1.5, fill_opacity=1, fill_color=TEAL_A, stroke_width=0).round_corners(0.2)
            label = TexText(R"\text{Problem }" + str(i + 1)).set_color(BLACK)
            label.set_z_index(1)
            problem = VGroup(rect, label)
            problems.add(problem)
        problems.arrange_in_grid(n_cols=2, h_buff=2, v_buff=0.5, fill_rows_first=False).set_width(10).to_edge(DOWN, buff=0.7)
        day1_label = TexText("Day 1").next_to(problems[:3], UP, buff=0.4)
        day2_label = TexText("Day 2").next_to(problems[3:], UP, buff=0.4)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    TransformMatchingShapes(imo_text, imo_text_shortened, run_time=1),
                    LaggedStartMap(FadeIn, problems, shift=UP * 0.2, lag_ratio=0.1)
                ),
                AnimationGroup(
                    Write(day1_label),
                    Write(day2_label)
                ), lag_ratio=0.3)
        )
        self.wait(1)

        # Highlight problem 6
        self.play(problems[-1][0].animate.set_fill(color=RED).set_stroke(width=4, color=YELLOW))
        self.wait(2)

        # Arrange the problems on the left
        imo_text_shortened.generate_target()
        imo_logo.generate_target()
        Group(imo_logo.target, imo_text_shortened.target).scale(1.5).set_y(0).to_edge(RIGHT, buff=1)
        imo_logo.set_z_index(0)
        imo_text_shortened.set_z_index(1)

        for problem in problems:
            problem.generate_target()
        VGroup(*[problem.target for problem in problems]).scale(0.9).arrange(DOWN, buff=0.2).to_edge(LEFT, buff=0.6)

        self.play(
            MoveToTarget(imo_text_shortened, run_time=2),
            MoveToTarget(imo_logo, run_time=2),
            AnimationGroup(*[
                MoveToTarget(problem, path_arc=PI * 0.3 if problem.target.get_y() > 0 else -PI * 0.3)
                for problem in problems
            ], lag_ratio=0.06, run_time=2),
            FadeOut(VGroup(day1_label, day2_label))
        )

        # Show scores
        total_participants = 630
        scores_data = [368, 253, 102, 342, 215, 6]
        score_bars = VGroup()
        master_bar_width_tracker = ValueTracker(0)
        for score, problem in zip(scores_data, problems):
            skeleton = Rectangle(
                width=4,
                height=0.5,
                fill_opacity=0.1,
                fill_color=WHITE,
                stroke_width=3,
                stroke_color=WHITE,
                stroke_opacity=1
            ).round_corners(0.08).set_z_index(1)

            def get_bar(score=score, skeleton=skeleton):
                fraction = master_bar_width_tracker.get_value() * score / total_participants
                return Rectangle(
                    fill_opacity=1,
                    fill_color=TEAL_E,
                    stroke_width=0
                ).match_height(
                    skeleton
                ).stretch_to_fit_width(
                    skeleton.get_width() * fraction
                ).move_to(
                    skeleton
                ).align_to(
                    skeleton, LEFT
                ).round_corners(min(0.08, fraction)).set_z_index(0)
            bar = always_redraw(get_bar)
            score_bars.add(VGroup(bar, skeleton).next_to(problem, RIGHT, buff=0.5))
        self.add(score_bars)

        scores = VGroup(*[
            Integer(score, font_size=30).set_color(TEAL).next_to(bar, RIGHT)
            for score, bar in zip(scores_data, score_bars)
        ])

        def update_scores(s):
            for score_value, score_display in zip(scores_data, scores):
                score_display.set_value(
                    int(score_value * master_bar_width_tracker.get_value())
                ).set_opacity(
                    master_bar_width_tracker.get_value() * 3
                )
        scores.add_updater(update_scores)
        self.add(scores)
        arrow = Arrow(ORIGIN, LEFT * 1.5).set_color(YELLOW).next_to(scores[-1], RIGHT)
        bars_label = TexText("Number of perfect scores", font_size=30).next_to(score_bars[0], UP)
        self.play(
            FadeIn(bars_label, run_time=1.4),
            AnimationGroup(
                AnimationGroup(
                    AnimationGroup(*[
                        FadeIn(skeleton)
                        for (_, skeleton) in score_bars
                    ], run_time=1),
                    master_bar_width_tracker.animate(run_time=2).set_value(1), lag_ratio=0.02), lag_ratio=0.2)
        )
        score_bars.clear_updaters()
        scores.clear_updaters()
        self.wait(1)
        self.play(GrowArrow(arrow))
        self.wait(2)

        # Add text: "The Last IMO Problem that AI could not solve"
        checkx_and_xs = VGroup(*[
            (Checkmark().set_color(PURE_GREEN) if i < 5 else Exmark().set_color(PURE_RED)).next_to(problems[i], RIGHT)
            for i in range(6)
        ])
        self.play(
            FadeOut(Group(bars_label, score_bars, scores, arrow, imo_logo, imo_text_shortened), shift=RIGHT * 2),
            AnimationGroup(*[
                FadeIn(check_or_x, shift=UP * 0.1)
                for check_or_x in checkx_and_xs
            ], lag_ratio=0.1), run_time=2)
        last_imo_problem_text = """
            The last IMO problem that
            AI could not solve
        """
        last_imo_problem = Text(last_imo_problem_text, font_size=50).set_x(0.5 * (FRAME_WIDTH * 0.5 + checkx_and_xs.get_right()[0]))
        for word in last_imo_problem_text.split():
            self.add(last_imo_problem[word])
            self.wait(0.06 * len(word))
        self.wait(2)


class ThumbnailIdea1(InteractiveScene):
    def construct(self):
        # Add problems
        problems = VGroup()
        for i in range(6):
            rect = Rectangle(width=6, height=1.5, fill_opacity=1, fill_color=GREEN, stroke_width=10, stroke_color=BLACK).round_corners(0.3)
            label = TexText(R"\text{P}" + str(i + 1), font_size=120).set_color(BLACK)
            label.set_z_index(1)
            problem = VGroup(rect, label)
            problems.add(problem)
        problems[-1][0].set_fill(color=RED)
        problems.arrange(DOWN, buff=0.2).set_height(FRAME_HEIGHT * 0.9).to_edge(LEFT, buff=1).fix_in_frame().set_z_index(100)
        self.add(problems)

        # Add a random grid
        random.seed(2)
        self.camera.frame.reorient(-28, 55, 0, (-16.09, -0.07, -17.96), 49.76)
        grid = RandomGrid(100)
        grid.get_reasonable_tiling()
        self.add(grid)
        for hole in grid.holes:
            hole.border.set_opacity(0)
            hole.cross.set_opacity(0)
            hole.background.set_color(RED_D).set_stroke(width=2)


_desaturate_cache_dir = os.path.join(tempfile.gettempdir(), "manim_desaturate_cache")
os.makedirs(_desaturate_cache_dir, exist_ok=True)
_source_array_cache = {}  # source_path -> float64 rgb(a) array, decoded once per source


def desaturate_from_source(source_path: str, alpha: float, position_ref: ImageMobject) -> ImageMobject:
    alpha = round(float(np.clip(alpha, 0.0, 1.0)), 2)

    if source_path not in _source_array_cache:
        _source_array_cache[source_path] = np.array(Image.open(source_path)).astype(np.float64)
    arr = _source_array_cache[source_path]

    has_alpha_channel = arr.shape[-1] == 4
    rgb = arr[..., :3]
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    blended_rgb = (1 - alpha) * rgb + alpha * gray_rgb
    blended = (
        np.concatenate([blended_rgb, arr[..., 3:4]], axis=-1)
        if has_alpha_channel else blended_rgb
    )
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    base_name = os.path.splitext(os.path.basename(source_path))[0]
    out_path = os.path.join(_desaturate_cache_dir, f"{base_name}_{alpha:.2f}.png")
    if not os.path.exists(out_path):
        Image.fromarray(blended).save(out_path)

    new_image_mobject = ImageMobject(out_path)
    new_image_mobject.replace(position_ref)
    return new_image_mobject


class Timeline(Group):
    def __init__(self, start_year, end_year, initial_year, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_year = start_year
        self.end_year = end_year
        self.number_line = NumberLine(
            x_range=(self.start_year, self.end_year, 1 / 12),
            unit_size=2.6,
            tick_size=0.1,
            longer_tick_multiple=2,
            big_tick_spacing=1
        )
        self.add(self.number_line)
        self.year_labels = self.number_line.add_numbers(
            range(self.start_year, self.end_year),
            group_with_commas=False,
        )

        def update_year_label(label):
            year = label.get_value()
            label.next_to(self.number_line.n2p(year), DOWN, 0.4)
            focal_value = np.exp(-0.1 * label.get_x()**2)
            label.set_height(0.25 + 0.25 * focal_value, about_edge=UP)
            label.set_fill(opacity=(0.5 + 0.5 * focal_value), color=interpolate_color(WHITE, YELLOW, focal_value**2))
        for label in self.year_labels:
            label.add_updater(update_year_label)

        self.images = Group()
        self.add(self.images)
        self._image_entries = []
        self.images.add_updater(self._refresh_images)

        self.center().set_y(-2)
        self.year_tracker = ValueTracker(initial_year)
        self.add_updater(
            lambda tl: tl.shift(tl.number_line.n2p(tl.year_tracker.get_value())[0] * LEFT)
        )

        self.markers = Group()

    def _refresh_images(self, images_group):
        for entry in self._image_entries:
            img = entry["current"]
            focal_value = np.exp(-0.1 * img.get_x() ** 2)
            new_img = desaturate_from_source(entry["source_path"], 1 - focal_value**2, img)
            new_img.set_width(1.6 + focal_value**0.7 * 2.35, about_edge=DOWN)
            new_img.set_opacity(focal_value**0.5)
            images_group.remove(img)
            images_group.add(new_img)
            entry["current"] = new_img

    def add_image(self, file_path, year):
        image = ImageMobject(file_path).align_to(DOWN, DOWN).set_x(self.number_line.n2p(year)[0])
        self.images.add(image)
        self._image_entries.append({"current": image, "source_path": image.image_path})
        self._refresh_images(self.images)  # avoid a 1-frame flash of the raw original

    def center_on_year(self, year):
        return self.shift(self.number_line.n2p(year)[0] * LEFT)

    def get_year_label(self, year):
        return self.year_labels[year - self.start_year]

    def get_marker(self, marker_text, year, image, font_size=60, shift=0, image_position=UP, image_shift=0):
        text = TexText(marker_text, font_size=font_size).next_to(self.number_line.n2p(year), UP).shift(UP * 2.5 + shift)
        ul = Underline(text)
        heading = VGroup(text, ul)
        image.next_to(heading, image_position).shift(image_shift)

        dot = Dot(radius=0.12).move_to(self.number_line.n2p(year)).set_z_index(1)
        line = Line(dot.get_center(), ul.get_center(), buff=0, stroke_width=3)

        marker = Group(dot, line, heading, image)
        marker.create = AnimationGroup(
            AnimationGroup(
                FadeIn(dot),
                ShowCreation(line),
                Write(text),
                GrowFromPoint(ul, ul.get_left()), lag_ratio=0.2),
            FadeIn(image, shift=image_position * 0.2), lag_ratio=0.5)
        marker.year_tracker = ValueTracker(year)
        marker.add_updater(lambda m: m.shift(self.number_line.n2p(m.year_tracker.get_value()) - dot.get_center()))
        self.markers.add(marker)

        marker.text = text
        marker.ul = ul
        marker.heading = heading
        marker.image = image
        marker.dot = dot
        marker.line = line

        return marker


# class AIEvolution(InteractiveScene):
#     def construct(self):
#         # Add the timeline
#         timeline = Timeline(2017, 2030, 2023)
#         self.add(timeline)

#         # Add the images to the timeline
#         images_dir = "AI Evolution Timeline images"
#         image_names_and_years = {
#             "dwarkesh_thumbnail.png": 2023,
#             "deepmind_2024.jpg": 2024,
#             "deepmind_and_openai_gold_medal.webp": 2025
#         }
#         for file_name in image_names_and_years:
#             timeline.add_image(
#                 os.path.join(images_dir, file_name),
#                 image_names_and_years[file_name]
#             )

#         # Move to 2024
#         self.play(timeline.year_tracker.animate.set_value(2024), run_time = 2)
#         self.wait(1)

#         # Move to 2025
#         self.play(timeline.year_tracker.animate.set_value(2025), run_time = 2)
#         self.wait(1)

#         # Move to 2026
#         self.play(timeline.year_tracker.animate.set_value(2026), run_time = 2)
#         self.wait(1)

#         # Go back to 2025
#         self.play(timeline.year_tracker.animate.set_value(2025), run_time = 2)


class AIEvolutionV2(InteractiveScene):
    def construct(self):
        # Add the timeline
        timeline = Timeline(2010, 2040, 2026).set_y(-3)
        self.add(timeline)

        # Move to 2021
        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale(2, about_point=[0, timeline.get_y(), 0]),
            timeline.year_tracker.animate.set_value(2021), run_time=2.5)
        self.wait(1)

        # Add impressive results in games and natural language
        alphago_marker = timeline.get_marker(
            "AlphaGo beats Lee Sedol",
            2016 + 3 / 12 + 15 / 365,
            ImageMobject("AI Evolution Timeline images/alphago.png").set_height(2),
            shift=UP * 2.3 + RIGHT * 3,
            image_position=RIGHT
        )
        dota_marker = timeline.get_marker(
            "OpenAI 5 beats Dota 2 champs",
            2019 + 4 / 12 + 15 / 365,
            ImageMobject("AI Evolution Timeline images/dota.png").set_height(2),
            shift=UP * 0.5 + RIGHT * 2,
            image_position=RIGHT,
            image_shift=UP * 0.3
        )
        gpt_marker = timeline.get_marker(
            "GPT-3 released",
            2020 + 5 / 12 + 29 / 365,
            ImageMobject("AI Evolution Timeline images/gpt.png").set_height(2),
            shift=DOWN * 1.3 + RIGHT * 2.6,
            image_position=RIGHT,
            image_shift=DOWN * 0.2
        )
        self.play(
            AnimationGroup(
                alphago_marker.create,
                dota_marker.create, lag_ratio=0.5)
        )
        self.play(gpt_marker.create)
        self.wait(2)

        # Add imo marker
        imo_marker = timeline.get_marker(
            "IMO Gold?",
            2023.9,
            ImageMobject("IMO_logo").set_height(2),
            shift=UP + RIGHT * 0.3
        )
        self.play(FadeIn(imo_marker))
        self.play(imo_marker.year_tracker.animate.set_value(2025.6), run_time=2)
        self.play(imo_marker.year_tracker.animate.set_value(2023.6), run_time=2)
        self.play(
            AnimationGroup(
                imo_marker.year_tracker.animate(run_time=7).set_value(2034),
                self.camera.frame.animate(run_time=5).shift(RIGHT * 18), lag_ratio=0.2)
        )

        # Move back to 2023
        self.play(
            AnimationGroup(
                FadeOut(Group(alphago_marker, dota_marker, gpt_marker, imo_marker), run_time=1),
                AnimationGroup(
                    self.camera.frame.animate.restore(),
                    timeline.year_tracker.animate.set_value(2023), run_time=3), lag_ratio=0.4)
        )
        self.wait(1)

        # Move to 2024
        self.play(timeline.year_tracker.animate.set_value(2024), run_time=2)
        deepmind_marker = timeline.get_marker(
            "Deepmind solves 4/6 IMO problems",
            2024 + 7 / 12 + 25 / 365,
            ImageMobject("AI Evolution Timeline images/deepmind_2024.jpg").set_height(2),
            shift=DOWN * 0.5 + RIGHT * 2,
            font_size=36
        )
        self.play(deepmind_marker.create)
        self.wait(1)

        # Move to 2025
        deepmind_marker_opacity_tracker = ValueTracker(1)
        deepmind_marker.add_updater(lambda m: m.set_opacity(deepmind_marker_opacity_tracker.get_value()))
        timeline.add_updater(lambda m: deepmind_marker.update())
        self.play(
            timeline.year_tracker.animate.set_value(2025),
            deepmind_marker_opacity_tracker.animate.set_value(0), run_time=2)
        self.remove(deepmind_marker)
        deepmind_and_openai_marker = timeline.get_marker(
            R"Deepmind, OpenAI, Harmonic, and Bytedance \\ solve all problems except P6",
            2025 + 7 / 12 + 21 / 365,
            ImageMobject("AI Evolution Timeline images/multiple_models_win_gold.png").set_height(3),
            shift=DOWN * 0.5 + RIGHT * 1.2,
            font_size=33
        )
        deepmind_and_openai_marker.text["except P6"].set_color(RED)
        self.play(deepmind_and_openai_marker.create)

        # Move to 2026
        deepmind_and_openai_marker_opacity_tracker = ValueTracker(1)
        deepmind_and_openai_marker.add_updater(lambda m: m.set_opacity(deepmind_and_openai_marker_opacity_tracker.get_value()))
        timeline.add_updater(lambda m: deepmind_and_openai_marker.update())
        self.play(
            timeline.year_tracker.animate.set_value(2026),
            deepmind_and_openai_marker_opacity_tracker.animate.set_value(0), run_time=2)
        self.remove(deepmind_and_openai_marker)


class HumanitysLastStand(InteractiveScene):
    def construct(self):
        # Add the timeline
        timeline = Timeline(2010, 2040, 2030.7).set_y(-3)
        self.add(timeline)

        # Add impressive results in games and natural language
        alphago_marker = timeline.get_marker(
            R"AlphaGo beats Lee Sedol \\ every game except game 4 (of 5)",
            2016 + 3 / 12 + 15 / 365,
            ImageMobject("AI Evolution Timeline images/alphago.png").set_height(2),
            shift=UP * 2.3 + RIGHT * 3,
            image_position=RIGHT
        )
        alphago_marker.text["game 4"].set_color(GREEN)
        dota_marker = timeline.get_marker(
            "OpenAI 5 beats Dota 2 champs",
            2019 + 4 / 12 + 15 / 365,
            ImageMobject("AI Evolution Timeline images/dota.png").set_height(2),
            shift=UP * 0.5 + RIGHT * 2,
            image_position=RIGHT,
            image_shift=UP * 0.3
        )
        gpt_marker = timeline.get_marker(
            "GPT-3 released",
            2020 + 5 / 12 + 29 / 365,
            ImageMobject("AI Evolution Timeline images/gpt.png").set_height(2),
            shift=DOWN * 1.3 + RIGHT * 2.6,
            image_position=RIGHT,
            image_shift=DOWN * 0.2
        )
        self.add(alphago_marker, dota_marker, gpt_marker)

        # Add imo marker and shift the camera back
        imo_marker = timeline.get_marker(
            "IMO Gold?",
            2030.7,
            ImageMobject("IMO_logo").set_height(2),
            shift=UP + RIGHT * 0.3
        )
        self.add(imo_marker)
        self.camera.frame.scale(2, about_point=[0, timeline.get_y(), 0]).shift(RIGHT * 11)
        self.camera.frame.save_state()
        self.camera.frame.scale(0.7, about_point=[0, timeline.get_y(), 0]).shift(LEFT * 14)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    imo_marker.year_tracker.animate.set_value(2026.6),
                    timeline.year_tracker.animate.set_value(2016),
                    self.camera.frame.animate.restore()
                ),
                FadeOut(Group(dota_marker, gpt_marker)), lag_ratio=0.6), run_time=6)

        # Show the game
        alphago_marker.image.generate_target()
        alphago_marker.image.target.scale(2)
        vs = TexText("VS", font_size=100)
        sedol_image = ImageMobject("AI Evolution Timeline images/sedol.jpg").set_height(5)
        Group(sedol_image, vs, alphago_marker.image.target).arrange(buff=0.5).next_to(alphago_marker.text, UP, buff=1.3)

        self.play(
            AnimationGroup(
                FadeIn(sedol_image, shift=RIGHT),
                Write(vs),
                MoveToTarget(alphago_marker.image, path_arc=PI * 0.3),
                lag_ratio=0.1
            )
        )


class Game4(InteractiveScene):
    def construct(self):
        # Add the board
        BOARD_N = 19
        board_extent = 6.2
        spacing = board_extent / (BOARD_N - 1)
        board_center = np.array([0.0, 0.0, 0.0])

        def grid_point(col, row):
            return board_center + np.array([
                -board_extent / 2 + col * spacing,
                -board_extent / 2 + row * spacing,
                0.0,
            ])

        board_bg = Square(side_length=board_extent + spacing * 1.4)
        board_bg.set_fill(color=TEAL_D, opacity=1)
        board_bg.set_stroke(width=0)
        board_bg.move_to(board_center)

        grid_lines = VGroup(*[
            Line(grid_point(i, 0), grid_point(i, BOARD_N - 1),
                 stroke_color=GREY_D, stroke_width=3)
            for i in range(BOARD_N)
        ], *[
            Line(grid_point(0, i), grid_point(BOARD_N - 1, i),
                 stroke_color=GREY_D, stroke_width=3)
            for i in range(BOARD_N)
        ])
        hoshi = VGroup(*[
            Dot(grid_point(c, r), radius=0.045, color=BLACK)
            for c in (3, 9, 15) for r in (3, 9, 15)
        ])
        self.add(board_bg, grid_lines)

        # Moves data
        LETTERS = "ABCDEFGHJKLMNOPQRST"
        COORDS = [
            "Q16", "D4", "C16", "R4", "P4", "P3", "O3", "Q3", "C6", "F3", "N4", "Q5",
            "J3", "E17", "H16", "C13", "E16", "C10", "D17", "B4", "O17", "R11", "E4",
            "E5", "D9", "F4", "C9", "D10", "E10", "E11", "F11", "E12", "F12", "B10",
            "F9", "F13", "G13", "F14", "G14", "N17", "N16", "M17", "O18", "J16",
            "H17", "K13", "Q10", "Q11", "P10", "P11", "O11", "O12", "N12", "O13",
            "N13", "N11", "O10", "N14", "M11", "O15", "O16", "N10", "M14", "N9",
            "N15", "O14", "M12", "R10", "L9", "J9", "K11", "G12", "H10", "G15",
            "H15", "F16", "F17",
            "L11",
            "K10", "M10", "L12", "K12", "N8", "O9", "P8", "P9", "Q9", "Q8", "R9",
            "O8", "L10", "J11", "S9", "P7", "Q13", "R8", "C4", "C5", "P15",
            "S8", "T9", "S10", "H13", "J10", "L7", "G11", "F10", "K8", "L8", "G8",
            "F8", "G7", "C12", "E15", "E18", "B13", "D13", "E13", "E6", "F5", "D14",
            "D12", "J7", "H9", "B6", "J14", "G16", "F15", "H14", "J12", "B12", "C11",
            "H5", "G5", "P2", "S13", "D6", "C3", "Q2", "R2", "S14", "R13", "R14",
            "K17", "G2", "T14", "T15", "T13", "S16", "B8", "B9", "A9", "C8", "H6",
            "J6", "H4", "F2", "E2", "E1", "D1", "A12", "A11", "L16", "J15", "L17",
            "L18", "G9", "J18", "R12", "S12", "R1", "S1", "P12", "T8", "P14", "T10",
            "P5", "K4",
        ]
        NUMBERS = list(range(1, 177)) + [179, 180]
        assert len(COORDS) == len(NUMBERS) == 178

        board = {}

        def neighbors(p):
            c, r = p
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nc, nr = c + dc, r + dr
                if 0 <= nc < BOARD_N and 0 <= nr < BOARD_N:
                    yield (nc, nr)

        def group_and_liberties(start):
            color = board[start]
            stack, seen, group, libs = [start], set(), set(), set()
            while stack:
                p = stack.pop()
                if p in seen:
                    continue
                seen.add(p)
                group.add(p)
                for np_ in neighbors(p):
                    c = board.get(np_)
                    if c is None:
                        libs.add(np_)
                    elif c == color and np_ not in seen:
                        stack.append(np_)
            return group, libs

        stone_radius = spacing * 0.47
        stone_mobs = {}

        # Play the sequence of moves
        for n, coord_s in zip(NUMBERS, COORDS):
            color = "B" if n % 2 == 1 else "W"
            col = LETTERS.index(coord_s[0])
            row = int(coord_s[1:]) - 1
            pos = (col, row)

            board[pos] = color
            opp = "W" if color == "B" else "B"
            captured = []
            for np_ in neighbors(pos):
                if board.get(np_) == opp:
                    group, libs = group_and_liberties(np_)
                    if not libs:
                        captured.extend(group)
            captured = sorted(set(captured))
            for cp in captured:
                del board[cp]

            stone = Circle(radius=stone_radius)
            stone.set_fill(BLACK if color == "B" else WHITE, opacity=1)
            stone.set_stroke(width=0)
            stone.move_to(grid_point(*pos))

            fade_outs = [FadeOut(stone_mobs[cp]) for cp in captured if cp in stone_mobs]
            self.play(FadeIn(stone), *fade_outs, run_time=0.12)

            stone_mobs[pos] = stone
            for cp in captured:
                stone_mobs.pop(cp, None)

        self.wait(2)


class QuoteScene(InteractiveScene):
    quote_color = YELLOW
    settled_color = WHITE
    bg_color = "#111111"
    lag_ratio = 0.1
    wait_time = 0.15
    run_time_per_word = 0.25
    min_run_time = 1.0
    max_run_time = 3.0

    def get_quote_and_sections(self, raw_text):
        section_texts = raw_text.split("|")
        full_text = "".join(section_texts)

        quote = Text(full_text, alignment="left")
        aspect_ratio = FRAME_WIDTH / FRAME_HEIGHT
        if quote.get_width() / quote.get_height() > aspect_ratio:
            quote.set_width(FRAME_WIDTH * 0.8)
        else:
            quote.set_height(FRAME_HEIGHT * 0.8)
        quote.set_color(self.quote_color)

        sections = [quote[text] for text in section_texts]
        return quote, sections, section_texts

    def get_section_run_time(self, section_text):
        n_words = len(section_text.split())
        run_time = n_words * self.run_time_per_word
        return max(self.min_run_time, min(self.max_run_time, run_time))

    def play_quote(self, raw_text, run_times=None, wait_time=None):
        quote, sections, section_texts = self.get_quote_and_sections(raw_text)

        quote_bg = quote.copy().set_color(self.bg_color)
        self.add(quote_bg)

        if run_times is None:
            run_times = [self.get_section_run_time(t) for t in section_texts]
        if wait_time is None:
            wait_time = self.wait_time

        prev_section = None
        for section, run_time in zip(sections, run_times):
            anims = [FadeIn(section, lag_ratio=self.lag_ratio, run_time=run_time)]
            if prev_section is not None:
                anims.append(prev_section.animate.set_color(self.settled_color))
            self.play(*anims)
            self.wait(wait_time)
            prev_section = section

        self.play(prev_section.animate.set_color(self.settled_color))
        return quote


class LuongQuote(QuoteScene):
    def construct(self):
        # Show the quote
        raw_text = """
            ‘‘We didn’t really have a way to teach
            the model to be patient.| It didn’t take
            the time to understand the problem,|
            to get a feel for the problem,|
            to not try to solve the problem.”
            """
        self.play_quote(raw_text)


class PatreonQuote(QuoteScene):
    def construct(self):
        # Show the quote
        raw_text = """
            “This question doesn’t contribute to
            a deep understanding of mathematics,|
            nor is it particularly difficult (when
            compared with mathematical research).|
            Rather, the value of this question lies
            in the fact that it warmed my heart when
            I solved it,| and it still warms my heart
            more than a year later.| Like a good book or
            a touching song, the value here is human.|

            Call me humanist,| but I truly believe
            that the value of this question,| as a
            mathematical discovery,| exceeds that
            of the average PhD thesis.”
            """
        self.play_quote(raw_text)


class PiCreaturesWatchingPreview(TeacherStudentsScene):
    def construct(self):
        # Students watch the preview
        self.play(self.get_teacher().change("raise_right_hand", look_at=UP * 2))
        self.play(
            self.get_students()[0].change("happy", look_at=UP * 2),
            self.get_students()[1].change("pondering", look_at=UP * 2),
            self.get_students()[2].change("well", look_at=UP * 2)
        )
        self.wait(20)


class PiCreaturesWatchingPreview2(TeacherStudentsScene):
    def construct(self):
        # Students watch the preview
        self.play(self.get_teacher().change("raise_right_hand", look_at=UP * 2 + LEFT * 4))
        self.play(
            self.get_students()[0].change("happy", look_at=UP * 2 + LEFT * 4),
            self.get_students()[1].change("pondering", look_at=UP * 2 + LEFT * 4),
            self.get_students()[2].change("well", look_at=UP * 2 + LEFT * 4)
        )
        self.wait(3)

        # Teacher reminds students to be patient
        self.teacher_says(Text("You'll need to\nbe patient", font_size=40))
        self.play(
            self.get_students()[0].change("pondering", look_at=DOWN * 2),
            self.get_students()[1].change("thinking", look_at=DOWN * 2),
            self.get_students()[2].change("pondering", look_at=DOWN * 2)
        )

        self.wait(7)


class Headlines(InteractiveScene):
    MARGIN = 0.4
    H_GAP = 0.35
    V_GAP = 0.35
    N_COLS = 2
    N_ROWS = 3

    REVEAL_WIDTH_FRAC = 0.55
    REVEAL_HEIGHT_FRAC = 0.55

    GROW_TIME = 0.8
    BACKSWING_FRACTION = 0.4
    OVERSHOOT_SCALE = 1.12
    SHIFT_TIME = 1.1

    LAG_RATIO = 0.5

    PULSE_AMPLITUDE = 0.01
    PULSE_FREQ_RANGE = (0.25, 0.35)
    BREATHE_TIME = 12

    def scale_to_fit_box(self, mobject, max_width, max_height):
        width_scale = max_width / mobject.get_width()
        height_scale = max_height / mobject.get_height()
        mobject.scale(min(width_scale, height_scale))
        return mobject

    def get_headline_update_func(self, tiny_start, overshoot_target, grid_target,
                                 start_delay, rise_time, settle_time,
                                 breathe_amplitude, breathe_freq):
        rise_end = start_delay + rise_time
        settle_end = rise_end + settle_time

        def update_func(mob, alpha, total_run_time):
            t = alpha * total_run_time

            if t <= start_delay:
                mob.become(tiny_start)
            elif t < rise_end:
                p = (t - start_delay) / rise_time
                mob.interpolate(tiny_start, overshoot_target, rush_from(p))
            elif t < settle_end:
                p = (t - rise_end) / settle_time
                mob.interpolate(overshoot_target, grid_target, smooth(p))
            else:
                breathe_t = t - settle_end
                factor = 1 / (1 + breathe_amplitude * np.sin(TAU * breathe_freq * breathe_t))
                mob.become(grid_target)
                mob.scale(factor, about_point=grid_target.get_center())

        return update_func

    def construct(self):
        # Add a grid of images
        images_dir = "AI Evolution Timeline images"
        image_names = [
            "fel.png",
            "unit_distance.png",
            "erdos_problems.png",
            "jacobian.png",
            "non-sofic_group.png",
            "navier-stokes.png",
        ]

        raw_images = [
            ImageMobject(os.path.join(images_dir, name))
            for name in image_names
        ]

        grid_width = FRAME_WIDTH - 2 * self.MARGIN
        grid_height = FRAME_HEIGHT - 2 * self.MARGIN
        cell_width = (grid_width - (self.N_COLS - 1) * self.H_GAP) / self.N_COLS
        cell_height = (grid_height - (self.N_ROWS - 1) * self.V_GAP) / self.N_ROWS

        cell_centers = []
        for row in range(self.N_ROWS):
            for col in range(self.N_COLS):
                x = -grid_width / 2 + cell_width / 2 + col * (cell_width + self.H_GAP)
                y = grid_height / 2 - cell_height / 2 - row * (cell_height + self.V_GAP)
                cell_centers.append(np.array([x, y, 0.0]))

        reveal_w = FRAME_WIDTH * self.REVEAL_WIDTH_FRAC
        reveal_h = FRAME_HEIGHT * self.REVEAL_HEIGHT_FRAC

        overshoot_targets = []
        grid_targets = []
        for img, center in zip(raw_images, cell_centers):
            reveal = img.copy()
            self.scale_to_fit_box(reveal, reveal_w, reveal_h)
            reveal.move_to(ORIGIN)

            overshoot_targets.append(reveal.copy().scale(self.OVERSHOOT_SCALE))

            grid = img.copy()
            self.scale_to_fit_box(grid, cell_width, cell_height)
            grid.move_to(center)
            grid_targets.append(grid)

        rise_time = self.GROW_TIME * (1 - self.BACKSWING_FRACTION)
        settle_time = self.GROW_TIME * self.BACKSWING_FRACTION + self.SHIFT_TIME
        seq_time = self.GROW_TIME + self.SHIFT_TIME

        n = len(raw_images)
        max_delay = (n - 1) * self.LAG_RATIO * seq_time
        total_run_time = max_delay + seq_time + self.BREATHE_TIME

        # Pop up the headlines
        animations = []
        for i in range(n):
            tiny_start = overshoot_targets[i].copy().scale(0.001 / self.OVERSHOOT_SCALE).set_opacity(0)
            start_delay = i * self.LAG_RATIO * seq_time

            mob = tiny_start.copy()
            self.add(mob)

            update_func = self.get_headline_update_func(
                tiny_start,
                overshoot_targets[i],
                grid_targets[i],
                start_delay,
                rise_time,
                settle_time,
                self.PULSE_AMPLITUDE,
                random.uniform(*self.PULSE_FREQ_RANGE),
            )
            animations.append(
                UpdateFromAlphaFunc(
                    mob,
                    lambda m, a, f=update_func: f(m, a, total_run_time),
                    run_time=total_run_time,
                    rate_func=linear,
                )
            )

        self.play(*animations)
