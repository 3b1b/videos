import math
import random
from manim_imports_ext import *
from scipy.spatial.transform import Slerp


class Tile(Rectangle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, fill_opacity = 0.9, fill_color = BLUE, stroke_width = 8, stroke_color = WHITE, **kwargs)
        self.round_corners(0.05)
        self.set_scale_stroke_with_zoom(True)

class Hole(VGroup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.background = Square(
            side_length = 1,
            fill_opacity = 1,
            fill_color = "#444444",
            stroke_width = 0
        )
        self.cross = Cross(self.background, stroke_width = 2).scale(0.95)
        points = [ORIGIN, RIGHT, UR, UP]
        self.border = VGroup(*[
            Line(points[i], points[(i + 1) % 4], stroke_width = 4, stroke_color = YELLOW).scale(1.06)
            for i in range(len(points))
        ]).match_width(self.cross).scale(1.06).move_to(self.cross)
        self.add(self.background, self.cross, self.border)
        self.set_scale_stroke_with_zoom(False)

class Grid(VGroup):
    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.vertical_lines = VGroup(*[Line(RIGHT*i, RIGHT*i + DOWN*n) for i in range(n + 1)])
        self.horizontal_lines = VGroup(*[Line(DOWN*i, DOWN*i + RIGHT*n) for i in range(n + 1)])
        self.lines = VGroup(self.vertical_lines, self.horizontal_lines).set_stroke(width = 4, color = WHITE, opacity = 0.7)
        self.lines.set_scale_stroke_with_zoom(True)
        
        self.background = Square(
            side_length = self.horizontal_lines.get_width(),
            fill_opacity = 1,
            fill_color = GREY_D,
            stroke_width = 0
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
        unit_size = self.get_width()/self.n
        tile = Tile(width, height, *args, **kwargs).scale(unit_size)
        self.position_at_coordinates(tile, i, j)
        self.tiles.add(tile)

    def add_hole(self, i, j, *args, **kwargs):
        unit_size = self.get_width()/self.n
        hole = Hole(*args, **kwargs).scale(unit_size)
        self.position_at_coordinates(hole, i, j)
        self.holes.add(hole)

class OptimalGrid(Grid):
    def __init__(self, k, *args, **kwargs):
        self.k = k
        n = k*k
        super().__init__(n, *args, **kwargs)

        # Holes
        for i in range(n):
            self.add_hole(k - 1 - i//k + (i % k)*k, i)

        # Main tiles
        for i in range((k - 1)*(k - 1)):
            self.add_tile(k, k, k - 1 - i//(k - 1) + (i % (k - 1))*k, i + i//(k - 1) + 1)
        self.main_tiles = VGroup(*self.tiles)

        # Upper-right tiles
        for j in range(1, k):
            self.add_tile(k, j, j*k, 0)
        self.ur_tiles = VGroup(*self.tiles[len(self.main_tiles):])

        # Lower-right tiles
        for j in range(1, k):
            self.add_tile(j, k, n - j, j*k)
        self.dr_tiles = VGroup(*self.tiles[len(self.main_tiles) + len(self.ur_tiles):])

        # Lower-left tiles
        for j in range(1, k):
            self.add_tile(k, j, n - (j + 1)*k, n - j)
        self.dl_tiles = VGroup(*self.tiles[len(self.main_tiles) + len(self.ur_tiles) + len(self.dr_tiles):])

        # Upper-left tiles
        for j in range(1, k):
            self.add_tile(j, k, 0, n - (j + 1)*k)
        self.ul_tiles = VGroup(*self.tiles[len(self.main_tiles) + len(self.ur_tiles) + len(self.dr_tiles) + len(self.dl_tiles):])

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
        tiles[0].shift(LEFT*8 + UP*4.5)
        tiles[1].shift(RIGHT*8 + UP*4.5)
        tiles[3].shift(LEFT*8 + DOWN*3.5)
        tiles[4].shift(RIGHT*8 + DOWN*4.5)

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
        self.play(AnimationGroup(*[FadeIn(hole) for hole in shuffled_holes], lag_ratio = 0.05))
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
            tiles[2].angle_tracker.animate.set_value(0)
        , run_time = 2)
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

        self.play(damp_tracker.animate.set_value(0), run_time = 3)
        tile.clear_updaters()
        holes.clear_updaters()

        # Add tiles above and below the hole
        right_hole = holes[3]
        right_hole.add_updater(lambda m: self.bring_to_front(m))
        tile_above = Tile(3, 5).align_to(right_hole.get_corner(UL), DL)
        tile_below = Tile(4, 5).align_to(right_hole.get_corner(DL), UL)
        self.add(tile_above, tile_below, holes)
        self.play(
            FadeIn(tile_above, shift = DOWN),
            FadeIn(tile_below, shift = UP),
            holes[:3].animate.set_opacity(0.2),
            VGroup(right_hole.border[0], right_hole.border[2]).animate.set_color(WHITE)
        , run_time = 2)
        self.wait(2)

        # Add holes to those tiles
        inner_holes_above = VGroup(*[
            Hole().next_to(tile_above, [LEFT, UP, RIGHT][i], buff = 0).shift(UP*0.8 if i == 0 else 0)
            for i in range(3)
        ])
        for i, hole in enumerate(inner_holes_above):
            for j in range(len(hole.border)):
                if (-j + 1) % 4 == i:
                    hole.border[j].set_color(WHITE)
        inner_holes_below = VGroup(*[
            Hole().next_to(tile_below, [LEFT, RIGHT, DOWN][i], buff = 0).shift(DOWN*0.8 if i == 0 else 0)
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
        self.play(AnimationGroup(*[FadeIn(hole) for hole in shuffled_inner_holes], lag_ratio = 0.2))
        self.wait(2)
        dashed_line = DashedLine(hole_below, hole_above, dash_length = 0.2, stroke_width = 6).set_color(PURE_RED)
        self.play(
            ShowCreation(dashed_line),
            inner_holes_above[1:].animate.set_opacity(0.2),
            inner_holes_below[1:].animate.set_opacity(0.2)
        )
        self.wait(3)

        # Slide the hole up and down the side of the tile
        extra_tiles_and_holes_group = VGroup(tile_above, tile_below, inner_holes_above, inner_holes_below, dashed_line)
        right_hole.clear_updaters()
        self.play(VGroup(extra_tiles_and_holes_group, right_hole).animate.shift(UP*0.3), run_time = 3)
        self.play(VGroup(extra_tiles_and_holes_group, right_hole).animate.shift(DOWN*0.6), run_time = 3)
        self.play(FadeOut(VGroup(holes[:3], inner_holes_above[1:], inner_holes_below[1:])))
        self.wait(1)
        holes_group = VGroup(right_hole, hole_above, hole_below, dashed_line)
        holes_group.generate_target()
        holes_group.target.shift(UP*(tile.get_top()[1] - right_hole.get_top()[1]))
        tile_above.generate_target()
        tile_above.target.align_to(holes_group.target[0].get_top(), DOWN)
        tile_below.generate_target()
        tile_below.target.align_to(holes_group.target[0].get_bottom(), UP)
        self.play(
            MoveToTarget(holes_group),
            MoveToTarget(tile_above),
            MoveToTarget(tile_below),
            self.camera.frame.animate.move_to(holes_group.target[0])
        , run_time = 3)
        right_hole.add_updater(lambda m: self.bring_to_front(m))
        self.play(
            FadeOut(dashed_line, shift = LEFT*1.7),
            VGroup(tile_above, hole_above).animate.shift(LEFT*(tile_above.get_right()[0] - right_hole.get_right()[0]))
        , run_time = 2.5)
        self.wait(2)
        new_tile = Tile(5, 3).align_to(right_hole.get_corner(DR), DL)
        self.play(FadeIn(new_tile, shift = LEFT), right_hole.border[1].animate.set_color(WHITE))
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
                FadeIn(hole, shift = direction)
                for hole, direction in zip(inner_holes[2:], [DL, UL])
            ]
            , lag_ratio = 0.1)
        , run_time = 1.5)
        outer_holes = VGroup(*[
            Hole().align_to(outer_tile.get_corner(direction1), direction2)
            for outer_tile, direction1, direction2 in zip(outer_tiles, [DL, UL, UR, DR], [DR, DL, UL, UR])
        ])
        for i in range(len(inner_holes)):
            for j in range(4):
                if (-j + 1) % 4 == i:
                    outer_holes[i].border[j].set_color(WHITE)
        self.play(AnimationGroup(*[FadeIn(hole) for hole in outer_holes], lag_ratio = 0.1))
        self.wait(0.5)
        checkmarks = VGroup(*[
            Checkmark().scale(1.5).set_color(PURE_GREEN).move_to(hole)
            for hole in [middle_hole] + list(inner_holes) + list(outer_holes)
        ])
        self.play(AnimationGroup(*[GrowFromCenter(checkmark) for checkmark in checkmarks], lag_ratio = 0.2))
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
            ])
        , run_time = 2)
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
                shift = DL*2
            ),
            self.camera.frame.animate.scale(0.9).move_to(puzzle_piece),
            puzzle_piece.animate.shift(0)
        , run_time = 2)

        # Show floating copies of the puzzle piece
        self.camera.frame.center()
        puzzle_piece.center()
        
        puzzle_pieces = VGroup(puzzle_piece, *[puzzle_piece.copy() for _ in range(4)])
        puzzle_pieces[1].move_to(LEFT*7.5 + UP*4.5)
        puzzle_pieces[2].move_to(RIGHT*7.5 + UP*4.5)
        puzzle_pieces[3].move_to(LEFT*7.5 + DOWN*4.5)
        puzzle_pieces[4].move_to(RIGHT*7.5 + DOWN*4.5)

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
        self.camera.frame.move_to(grid).set_height(grid.get_height()*1.4)
        self.camera.frame.save_state()
        self.wait(2)

        # Change it to k = 4
        new_grid = OptimalGrid(4).align_to(ORIGIN, DL)
        for hole in new_grid.holes:
            hole.border.set_color(WHITE)
        grid.holes.set_z_index(100)
        self.add(grid)
        self.play(
            self.camera.frame.animate(run_time = 1).move_to(new_grid).set_height(new_grid.get_height()*1.4),
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
        x_length_label = Tex("16", font_size = 180).next_to(grid, DOWN, buff = 1)
        y_length_label = x_length_label.copy().next_to(grid, LEFT, buff = 1)
        brace1 = Brace(grid, DOWN)
        brace2 = Brace(grid, LEFT)
        self.play(GrowFromEdge(brace1, UP), GrowFromEdge(brace2, RIGHT), FadeIn(VGroup(x_length_label, y_length_label)))
        self.wait(2)

        # Count the holes
        circles = VGroup(*[
            Circle(radius = 0.8, stroke_width = 3, stroke_color = PURE_GREEN).move_to(hole)
            for hole in grid.holes
        ])
        hole_numbers = VGroup(*[
            Tex(str(i + 1), font_size = 90).next_to(hole, UP, buff = 0.7)
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
                        FadeIn(num, shift = UP*0.4)
                    , lag_ratio = 0.1)
                    for circle, num in zip(circles, hole_numbers)
                ], lag_ratio = 0.05)
            , lag_ratio = 0.1)
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
            self.camera.frame.animate(run_time = 1).move_to(new_grid).set_height(new_grid.get_height()*1.4),
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
        x_length_label = Tex("9", font_size = 100).next_to(grid, DOWN, buff = 0.7)
        y_length_label = x_length_label.copy().next_to(grid, LEFT, buff = 0.7)
        brace1 = Brace(grid, DOWN)
        brace2 = Brace(grid, LEFT)
        self.play(GrowFromEdge(brace1, UP), GrowFromEdge(brace2, RIGHT), FadeIn(VGroup(x_length_label, y_length_label)))
        self.wait(2)
        self.play(FadeOut(VGroup(x_length_label, y_length_label, brace1, brace2)))

        # Generalize
        tile = grid.tiles[1]
        grid.save_state()
        k_label_1 = Tex("k", font_size = 100).next_to(tile, DOWN, buff = 0.7)
        k_label_2 = k_label_1.copy().next_to(tile, LEFT, buff = 0.7)
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
                )
            , lag_ratio = 0.4)
        )

        # Add a slider
        k_tracker = ValueTracker(3)
        x_range = [0, 50, 1]
        k_slider = NumberLine(
            x_range = x_range,
            width = 3,
            include_numbers = True,
            numbers_to_exclude = [x for x in range(x_range[0], x_range[1], x_range[2]) if x % 10 != 0],
            longer_tick_multiple = 10
        )
        for i, tick in enumerate(k_slider.ticks):
            if i % 10 != 0:
                tick.scale(0.4).set_stroke(width = 1.5)
        k_display = Tex("k = 2").next_to(k_slider, UP, buff = 0.7)
        k_value = k_display.make_number_changeable("2")
        k_value.add_updater(lambda m: m.set_value(round(k_tracker.get_value())))
        k_triangle = Triangle(fill_opacity = 1, fill_color = TEAL, stroke_width = 0).stretch(1.5, 1).set_width(0.2).rotate(PI)
        k_triangle.align_to(k_slider[0].get_center(), DOWN)
        k_triangle.add_updater(lambda m: m.set_x(k_slider.n2p(round(k_tracker.get_value()))[0]))
        rect = BackgroundRectangle(VGroup(k_slider, k_display, k_triangle), buff = 0.1).round_corners(0.2)
        k_slider_group = VGroup(rect, k_slider, k_display, k_triangle)
        k_slider_group.fix_in_frame().set_anti_alias_width(0).to_corner(UL, buff = 0.2).set_scale_stroke_with_zoom(True)
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
                    hole.border.set_stroke(width = 5*new_k**(-1/3), color = interpolate_color(WHITE, RED, min(1, (new_k - 3)/7)))
                for tile in grid.tiles:
                    tile.add_updater(lambda m: m.set_stroke(width = 1.5*new_k**1.5))
        grid.add_updater(update_grid)

        # Increase k to 10
        next_k = 10
        self.play(
            k_tracker.animate(run_time = 5).set_value(next_k),
            self.camera.frame.animate(run_time = 5).move_to([next_k*next_k*0.5, next_k*next_k*0.5, 0]).set_height(1.4*next_k*next_k)
        )
        grid.suspend_updating()

        # Show the new size: k^2 x k^2
        x_length_label = Tex(R"k^2", font_size = 900).next_to(grid, DOWN, buff = 5)
        y_length_label = x_length_label.copy().next_to(grid, LEFT, buff = 5)
        self.play(Write(x_length_label), Write(y_length_label), run_time = 2)
        self.wait(2)
        self.play(FadeOut(VGroup(brace1, brace2, x_length_label, y_length_label)))

        # Increase k to 45
        grid.resume_updating()
        self.play(
            k_tracker.animate(run_time = 5).set_value(max_k),
            self.camera.frame.animate(run_time = 5).reorient(-13, 40, 0, (654.42, 252.66, 461.53), 1315.09)
        )
        grid.suspend_updating()
        self.wait(2)

        # Show the new size
        x_length_label = Tex(R"2025\\=45^2", font_size = 12000).next_to(grid, DOWN, buff = 50)
        y_length_label = x_length_label.copy().next_to(grid, LEFT, buff = 60)
        brace1 = Brace(grid, DOWN)
        brace2 = Brace(grid, LEFT)
        self.play(GrowFromEdge(brace1, UP), GrowFromEdge(brace2, RIGHT), Write(x_length_label["2025"]), Write(y_length_label["2025"]))
        self.wait(2)
        self.play(
            FadeIn(
                x_length_label["=45^2"].set_color(
                    YELLOW
                ).next_to(
                    x_length_label["2025"], RIGHT, buff = 30
                ).align_to(
                    x_length_label["2025"], DOWN
                )
            ),
            FadeIn(
                y_length_label["=45^2"].set_color(
                    YELLOW
                ).next_to(
                    y_length_label["2025"], DOWN, buff = 40
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
            self.camera.frame.animate.reorient(7, 46, 0, (1110.17, 171.23, 372.81), 1109.67)
        , run_time = 10)
        self.play(
            grid.tiles.animate.set_stroke(width = 40),
            grid.holes.animate.set_stroke(width = 0.5),
            self.camera.frame.animate.reorient(-24, 62, 0, (214.22, 294.13, -91.55), 379.46)
        , run_time = 10)
        self.play(
            # grid.animate.restore(),
            grid.tiles.animate.set_stroke(width = 80),
            grid.holes.animate.set_stroke(width = 4),
            self.camera.frame.animate.reorient(-27, 65, 0, (631.07, 270.25, 216.58), 180.95)
        , run_time = 10)

        # Reset the camera to the original position and show the labels for k
        tile = grid.tiles[1596]
        k_label_1 = Tex("k", font_size = 5000).next_to(tile, DOWN, buff = 20)
        k_label_2 = k_label_1.copy().next_to(tile, LEFT, buff = 20)

        x_length_label = Tex(R"k^2", font_size = 12000).next_to(grid, DOWN, buff = 50)
        y_length_label = x_length_label.copy().next_to(grid, LEFT, buff = 50)
        self.add(x_length_label, y_length_label)

        k_slider_group.set_z_index(1000)
        self.add(k_slider_group)
        self.play(
            AnimationGroup(
                self.camera.frame.animate(run_time = 10).reorient(-18, 58, 0, (973.79, 650.28, -62.57), 1754.60),
                AnimationGroup(
                    AnimationGroup(
                        grid.background.animate.set_opacity(0.1),
                        grid.lines.animate.set_opacity(0.1),
                        *[t.animate.set_opacity(0.1 if t != tile else t.get_opacity()) for t in grid.tiles],
                        grid.holes.animate.set_opacity(0.1).set_stroke(width = 0.5)
                    ),
                    FadeIn(VGroup(k_label_1, k_label_2))
                , lag_ratio = 0.4, run_time = 2.5)
            , lag_ratio = 0.15)
        )
        grid.holes.set_scale_stroke_with_zoom(True)

        # Set k back to 5
        self.play(
            grid.animate.restore(),
            FadeOut(VGroup(k_label_1, k_label_2, x_length_label, y_length_label))
        )
        grid.resume_updating()
        self.play(
            k_tracker.animate(run_time = 2).set_value(5),
            self.camera.frame.animate(run_time = 2).restore()
        )
        grid.suspend_updating()
        self.wait(1)

        # Count the number of square tiles
        new_grid = OptimalGrid(k = 5).match_width(grid).move_to(grid)
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
            Tex(str(i + 1), font_size = 150).set_color(BLACK).move_to(tile)
            for i, tile in enumerate(main_tiles)
        ])
        grid.save_state()
        holes = grid.holes
        holes.add_updater(lambda m: self.bring_to_front(m))
        self.play(
            VGroup(grid.lines, *[tile for tile in edge_tiles]).animate.set_opacity(0.1),
            AnimationGroup(*[
                AnimationGroup(
                    tile.animate(rate_func = there_and_back).set_color(YELLOW).scale(1.1).set_fill(opacity = 0.5),
                    GrowFromCenter(num, run_time = 0.8)
                )
                for tile, num in zip(main_tiles, main_tile_numbers)
            ], lag_ratio = 0.1)
        )
        holes.suspend_updating()
        self.wait(2)

        # Generalize
        self.play(
            AnimationGroup(*[
                num.animate.become(
                    Dot(radius = 0.2).set_color(BLACK).move_to(tile) if i < 16 - 1 - 3 else
                    Tex("(k - 1)^2", font_size = 120).set_color(BLACK).move_to(tile)
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
            0.5*(grid.get_right()[0] + self.camera.frame.get_right()[0])
        ).set_color(
            TEAL_A
        )
        self.play(TransformFromCopy(main_tile_numbers[-1], main_tile_count, path_arc = -PI*0.2), run_time = 1.5)
        self.wait(1)

        # Switch focus to the tiles around the edges
        holes.resume_updating()
        self.play(
            grid.animate.restore(),
            main_tiles.animate.set_opacity(0.5),
            FadeOut(main_tile_numbers)
        , run_time = 2)
        holes.suspend_updating()
        self.wait(2)

        # Count the edge tiles
        edge_tile_numbers = VGroup(*[
            Tex(str(i % 4 + 1), font_size = 100).set_color(BLACK).move_to(tile)
            for i, tile in enumerate(edge_tiles)
        ])
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    tile.animate(rate_func = there_and_back).set_color(YELLOW).scale(1.1).set_fill(opacity = 0.5),
                    GrowFromCenter(num, run_time = 0.8)
                )
                for tile, num in zip(edge_tiles, edge_tile_numbers)
            ], lag_ratio = 0.2)
        )
        self.wait(2)

        # Generalize the edge tile count
        self.play(
            AnimationGroup(*[
                num.animate.become(
                    Tex(R"\cdots", font_size = 100).set_color(BLACK).move_to(tile) if i % 4 == 2 else
                    Tex("k - 1", font_size = 100).set_color(BLACK).move_to(tile) if i % 4 == 3 else
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
            main_tile_count[0].get_height()/edge_tile_count[2].get_height()
        ).next_to(
            main_tile_count, DOWN, buff = 0.6
        ).align_to(
            main_tile_count[-2], RIGHT
        )
        edge_tile_count[0].shift(LEFT*0.16)
        k_minus_1_copies = VGroup(*[edge_tile_count[3:-1].copy() for _ in range(4)])
        self.play(
            AnimationGroup(*[
                TransformMatchingShapes(
                    edge_tile_numbers[4*i + 3].copy(),
                    k_minus_1_copies[i],
                    path_arc = -PI*0.2
                )
                for i in range(4)
            ])
        , run_time = 1.5)
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
        formula_group.target.arrange(buff = 0.4)
        formula_group.target.match_y(self.camera.frame).shift(UP*4).align_to(formula_group, RIGHT).shift(LEFT)
        formula_group.target[1].align_to(formula_group.target[0], DOWN)
        self.play(
            FadeOut(k_slider_group, shift = LEFT*3),
            VGroup(grid, edge_tile_numbers).animate.shift(LEFT*14),
            MoveToTarget(formula_group, path_arc = PI*0.2)
        , run_time = 2)

        # Expand it
        expanded_version_intermediate = Tex("= k^2 - 2k + 1 + 4k - 4")
        expanded_version_intermediate.scale(
            formula_group[0][1].get_height()/expanded_version_intermediate[1].get_height()
        ).next_to(
            formula_group, DOWN, buff = 0.8
        )
        expanded_version_intermediate[:8].align_to(formula_group[0], RIGHT).set_color(TEAL_A)
        expanded_version_intermediate[0].set_color(WHITE)
        expanded_version_intermediate[8].match_x(formula_group[1][0])
        expanded_version_intermediate[9:].align_to(formula_group[1][1:], LEFT).set_color(TEAL_D)
        self.play(
            AnimationGroup(
                FadeIn(expanded_version_intermediate[0]),
                TransformMatchingShapes(formula_group[0].copy(), expanded_version_intermediate[1:8], run_time = 1.2),
                TransformMatchingShapes(formula_group[1][0].copy(), expanded_version_intermediate[8], run_time = 1),
                TransformMatchingShapes(formula_group[1][1:].copy(), expanded_version_intermediate[9:], run_time = 1.2)
            , lag_ratio = 0.2)
        )
        self.wait(1)

        # Simplify
        expanded_version = Tex("= k^2 + 2k - 3")
        expanded_version.match_height(
            expanded_version_intermediate
        ).next_to(
            expanded_version_intermediate, DOWN, buff = 0.8
        ).align_to(
            expanded_version_intermediate, LEFT
        )
        self.play(
            AnimationGroup(
                TransformMatchingShapes(expanded_version_intermediate[:3].copy(), expanded_version[:3], run_time = 1.2),
                TransformMatchingShapes(
                    VGroup(expanded_version_intermediate[3:6], expanded_version_intermediate[9:11]).copy(),
                    expanded_version[3:6]
                , run_time = 1.2),
                TransformMatchingShapes(
                    VGroup(expanded_version_intermediate[6:8], expanded_version_intermediate[11:13]).copy(),
                    expanded_version[6:8]
                , run_time = 1.2)
            , lag_ratio = 0.2)
        )


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
        grid.tiles.set_stroke(width = 3)
        for hole in grid.holes:
            hole.border.set_color(WHITE)

        # Number the holes according to their height
        nums_color = BLUE_B
        values = [n - j for j in hole_positions]
        nums = VGroup(*[
            Integer(j).set_color(BLUE_B).next_to(grid.holes[i], UP, buff = 0.15)
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
        brace = Brace(grid, UP, buff = 0.8)
        label = brace.get_tex("N", font_size = 60).set_color(n_color)
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
                        FadeIn(num, shift=UP*0.2, run_time=0.7)
                    , lag_ratio=0.1)
                    for num, column in zip(nums, column_highlights)
                ], lag_ratio=0.2),
                AnimationGroup(
                    self.camera.frame.animate(run_time = 1.5).scale(1.1).shift(UP*0.7),
                    AnimationGroup(
                        GrowFromEdge(brace, DOWN),
                        Write(label)
                    )
                , lag_ratio = 0.2)
            , lag_ratio = 0.5)
        )
        self.remove(column_highlights)

        # Make the bar chart
        base = Line(LEFT, RIGHT).set_width(nums.get_width()*1.1).align_to(grid, DOWN)
        bars = VGroup(*[
            Rectangle(
                width = column.get_width()*0.9,
                height = column.get_height(),
                fill_opacity = 1,
                fill_color = bar_color,
                stroke_width = 0
            ).match_x(column)
            for column in column_highlights
        ]).align_to(base, DOWN)
        for bar in bars:
            bar.align_to(base, DOWN)
        chart = VGroup(bars, base)

        for num, bar in zip(nums, bars):
            num.generate_target()
            num.target.next_to(bar, UP, buff = 0.2)
            bar.save_state()
            bar.stretch_to_fit_height(0.001).align_to(base, DOWN)
        self.play(
            self.camera.frame.animate.restore(),
            FadeOut(VGroup(brace, label)),
            FadeOut(grid, run_time = 3),
            AnimationGroup(*[
                AnimationGroup(
                    MoveToTarget(num),
                    bar.animate.restore()
                , lag_ratio = 0.1, run_time = 2)
                for num, bar in zip(nums, bars)
            ]),
            ShowCreation(base, run_time = 1)
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
        bars_left_to_right = VGroup(*sorted(bars, key = lambda bar: bar.get_x()))
        nums_left_to_right = VGroup(*sorted(nums, key = lambda num: num.get_x()))
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
                FadeIn(marker, run_time = 0.3)
                for marker in increasing_markers
            ], lag_ratio = 0.3)
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
                FadeIn(marker, run_time = 0.7)
                for marker in decreasing_markers
            ], lag_ratio = 0.3)
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
        bars_left_to_right = VGroup(*sorted(bars, key = lambda bar: bar.get_x()))
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
                FadeIn(marker, run_time = 0.3)
                for marker in decreasing_markers
            ], lag_ratio = 0.3)
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
                FadeIn(marker, run_time = 0.3)
                for marker in increasing_markers
            ], lag_ratio = 0.3),
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
            font_size = 80,
            tex_to_color_map = {"LIS": increasing_sequence_color, "LDS": decreasing_sequence_color, "N": n_color}
        ).shift(RIGHT*7 + UP*1)

        value_font_size = 80
        value_row_y = inequality.get_bottom()[1] - 1
        lis_value = None
        lds_value = None

        def make_lis_value(val):
            mob = Integer(val, font_size = value_font_size).set_color(increasing_sequence_color)
            mob.match_x(inequality["LIS"])
            mob.set_y(value_row_y)
            return mob

        def make_lds_value(val):
            mob = Integer(val, font_size = value_font_size).set_color(decreasing_sequence_color)
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
                cdot_value = Tex(R"\cdot", font_size = value_font_size)
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
                geq_value = Tex(R"\ge", font_size = value_font_size)
                geq_value.match_x(inequality[-2])
                geq_value.set_y(value_row_y)
                n_value = Tex(str(n), font_size = value_font_size).set_color(n_color)
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
        arrow = Arrow(ORIGIN, DOWN*1.5, thickness = 5).set_color(YELLOW).next_to(focus_bar, UP, buff = 1.5)
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
            ], lag_ratio = 0.3)
        )
        self.wait(1)
        lis_text = Tex(R"\text{LIS}: 2", font_size = 110).set_color(increasing_sequence_color)
        lds_text = Tex(R"\text{LDS}: 3", font_size = 110).set_color(decreasing_sequence_color)
        lds_text.next_to(lis_text, DOWN, buff = 0.6).align_to(lis_text, LEFT)
        VGroup(lis_text, lds_text).set_y(0).to_edge(RIGHT, buff = 1.5)
        base.suspend_updating()
        self.play(
            AnimationGroup(
                VGroup(chart, nums, arrow, increasing_markers).animate.to_edge(LEFT, buff = 1.5),
                Write(lis_text)
            , lag_ratio = 0.6, run_time = 1.5)
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
            ], lag_ratio = 0.3)
        )
        self.wait(1)
        self.play(Write(lds_text), run_time = 1.5)
        self.wait(1)
        self.wait(2)

        # Save the values as a pair of numbers below the bar
        pair = Tex("(2, 3)", font_size = 30).next_to(focus_bar, DOWN)
        pair[1].set_color(increasing_sequence_color)
        pair[3].set_color(decreasing_sequence_color)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    TransformFromCopy(lis_text[-1], pair[1]),
                    TransformFromCopy(lds_text[-1], pair[3])                        
                , run_time = 2),
                FadeIn(VGroup(pair[0], pair[2], pair[4]))
            , lag_ratio = 0.7)
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
            FadeOut(VGroup(arrow, lis_text, lds_text), run_time = 1),
            MoveToTarget(chart, run_time = 2),
            MoveToTarget(nums, run_time = 2),
            MoveToTarget(pair, run_time = 2),
            FadeOut(increasing_markers, shift = RIGHT*0.08, run_time = 0.6),
            FadeOut(decreasing_markers, shift = RIGHT*0.08, run_time = 0.6)
        )
        base.add_updater(lambda m: self.bring_to_front(m))
        self.wait(2)

        # Show that all the numbers are distinct
        if False:
            self.remove(pair)
            self.wait(1)
            circles = VGroup(*[Circle(radius = 0.35, fill_opacity = 0, stroke_width = 3, stroke_color = YELLOW).move_to(num) for num in nums])
            self.play(AnimationGroup(*[ShowCreation(circle) for circle in circles], lag_ratio = 0.1))

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
            ], lag_ratio = 0.2)
        , run_time = 3.6)
        self.remove(pair_4)
        self.add(pairs)
        self.wait(2)

        # Do another example
        bars.save_state()
        nums.save_state()
        pairs.save_state()
        focus_index = 6
        focus_bar = bars[focus_index]
        arrow = Arrow(ORIGIN, DOWN*1.1, thickness = 4).set_color(YELLOW).next_to(focus_bar, UP, buff = 0.85)
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
            ], lag_ratio = 0.3)
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
            ], lag_ratio = 0.3)
        )
        self.wait(1)
        self.play(
            bars.animate.restore(), nums.animate.restore(), pairs.animate.restore(),
            FadeOut(arrow), FadeOut(increasing_markers), FadeOut(decreasing_markers), run_time = 2
        )
        self.wait(1)

        # Indicate pairs to show uniqueness
        self.play(AnimationGroup(*[Indicate(pair) for pair in pairs], lag_ratio = 0.1), run_time = 3)

        # Save the full chart
        original_chart_group = VGroup(chart, nums, pairs).copy()

        # Bring in an arbitrary pair of bars
        bar1 = bars[1].copy()
        bar2 = bar1.copy()
        VGroup(bar1, bar2).arrange(buff = 2).align_to(bars[0], DOWN)
        self.play(
            AnimationGroup(
                FadeOut(VGroup(bars, nums, pairs)),
                FadeIn(VGroup(bar1, bar2))
            , lag_ratio = 0.2)
        , run_time = 3)
        self.wait(2)

        # Write an arbitrary pair of values for the LIS and LDS for that bar
        pair = Tex("(x, y)", tex_to_color_map = {"x": increasing_sequence_color, "y": decreasing_sequence_color}).match_height(pairs[0]).match_y(pairs[0]).match_x(bar1)
        self.play(FadeIn(pair))

        # Make the second bar taller
        stretch_factor = 1.2
        self.play(
            bar1.animate.stretch(1/stretch_factor, 1).align_to(bar1, DOWN),
            bar2.animate.stretch(stretch_factor, 1).align_to(bar2, DOWN)
        , run_time = 0.7)
        self.play(
            bar1.animate.stretch(stretch_factor**2, 1).align_to(bar1, DOWN),
            bar2.animate.stretch(1/stretch_factor**2, 1).align_to(bar2, DOWN)
        , run_time = 0.7)
        self.play(
            bar1.animate.stretch(1/stretch_factor**3, 1).align_to(bar1, DOWN),
            bar2.animate.stretch(stretch_factor**3, 1).align_to(bar2, DOWN)
        , run_time = 2)
        pair2 = Tex("(x', y')", tex_to_color_map = {"x'": increasing_sequence_color, "y'": decreasing_sequence_color}).match_height(pairs[0]).match_y(pairs[0]).match_x(bar2)
        self.play(FadeIn(pair2))
        self.wait(1)

        # Show a generic tail of bars behind the first bar
        heights = [2, 6, 5, 3]
        heights = [h*0.6 for h in heights]
        tail = VGroup(*[
            bar1.copy().stretch_to_fit_width(0.5).stretch_to_fit_height(height)
            for height in heights
        ]).arrange(
            buff = 0.1
        ).next_to(
            bar1, LEFT, buff = 0.2
        )
        tail_opacity = 0.4
        for bar in tail:
            bar.align_to(
                bar1, DOWN
            ).set_opacity(
                tail_opacity
            )
        cdots = Tex(R"\cdots", font_size = 100).match_y(bar1)

        self.play(
            AnimationGroup(
                *[
                    FadeIn(bar)
                    for bar in tail
                ],
                Write(cdots, run_time = 1.5)
            , lag_ratio = 0.1),
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
        brace = Brace(increasing_sequence, UP).shift(UP*0.1)
        label = brace.get_tex("x", font_size = 60).set_color(increasing_sequence_color).shift(UP*0.2)
        self.play(
            AnimationGroup(*[
                bar.animate.set_opacity(1)
                for bar in increasing_sequence
            ], lag_ratio = 0.1),
            AnimationGroup(*[
                FadeIn(marker)
                for marker in increasing_markers
            ], lag_ratio = 0.1),
            GrowFromEdge(brace, DOWN),
            Write(label)
        )
        self.wait(2)

        # Extend the increasing sequence to the second bar
        brace.generate_target()
        label.generate_target()
        extended_brace = Brace(VGroup(increasing_sequence, bar2), UP).shift(UP*0.1)
        extended_label = extended_brace.get_tex(R"x' \ge x + 1", font_size = 60).set_color(increasing_sequence_color).shift(UP*0.2)
        part1 = extended_label[:3]
        part2 = extended_label[3:]
        part2.save_state()
        part2.match_x(extended_brace)
        bar2_increasing_marker = marker_rect(bar2, increasing_sequence_color, 0)
        increasing_markers.add(bar2_increasing_marker)
        self.play(
            TransformFromCopy(brace, extended_brace),
            TransformMatchingShapes(label.copy(), part2),
            FadeIn(bar2_increasing_marker)
        , run_time = 2)
        self.wait(1)
        self.play(part2.animate.restore(), FadeIn(part1, shift = RIGHT*0.5))

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
            FadeOut(increasing_markers[-1], shift = DOWN*3),
            tail.animate.set_opacity(tail_opacity),
            bar2.animate.stretch_to_fit_height(0.6*bar1.get_height()).align_to(bar2, DOWN)
        )
        self.wait(2)

        # Show the decreasing subsequence of length 7
        decreasing_sequence = VGroup(tail[1], tail[2], bar1)
        decreasing_markers = VGroup(*[
            marker_rect(bar, decreasing_sequence_color, 0)
            for bar in decreasing_sequence
        ])
        brace = Brace(decreasing_sequence, UP)
        label = brace.get_tex("y", font_size = 60).set_color(decreasing_sequence_color)
        self.play(
            AnimationGroup(*[
                bar.animate.set_opacity(1)
                for bar in decreasing_sequence
            ], lag_ratio = 0.1),
            AnimationGroup(*[
                FadeIn(marker)
                for marker in decreasing_markers
            ], lag_ratio = 0.1),
            GrowFromEdge(brace, DOWN),
            Write(label)
        )
        self.wait(2)

        # Extend the decreasing sequence to the second bar
        brace.generate_target()
        label.generate_target()
        extended_brace = Brace(VGroup(decreasing_sequence, bar2), UP).align_to(case1[-3], UP)
        extended_label = extended_brace.get_tex(R"y' \ge y + 1", font_size = 60).set_color(decreasing_sequence_color).align_to(case1[-2], UP)
        part1 = extended_label[:3]
        part2 = extended_label[3:]
        part2.save_state()
        part2.match_x(extended_brace)
        bar2_decreasing_marker = marker_rect(bar2, decreasing_sequence_color, 0)
        decreasing_markers.add(bar2_decreasing_marker)
        self.play(
            TransformFromCopy(brace, extended_brace),
            TransformMatchingShapes(label.copy(), part2),
            FadeIn(bar2_decreasing_marker)
        , run_time = 2)
        self.wait(1)
        self.play(part2.animate.restore(), FadeIn(part1, shift = RIGHT*0.5))

        # Save the second example
        case2 = VGroup(
            tail, bar1, bar2, decreasing_markers, base, pair, pair2,
            brace, label, extended_brace, extended_label, cdots
        )

        # Show both examples side by side
        case1.clear_updaters()
        base.clear_updaters()
        case1[1:3].set_stroke(width = 3, color = YELLOW, behind = True)
        case1[1:4].set_z_index(400)
        case2.generate_target()
        case2.target[1:3].set_stroke(width = 3, color = YELLOW)
        case2.set_stroke(behind = True)
        VGroup(case1, case2.target).scale(0.63).arrange(buff = 0.7)
        case2.target.align_to(case1, DOWN)
        case2[1:4].set_z_index(400)
        case1_label = TexText("Case 1").next_to(case1, DOWN, buff = 1)
        case2_label = TexText("Case 2").next_to(case2.target, DOWN, buff = 1)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    FadeIn(case1, shift = RIGHT*5),
                    MoveToTarget(case2)
                ),
                LaggedStartMap(FadeIn, VGroup(case1_label, case2_label), lag_ratio = 0.2, shift = UP*1.5, run_time = 0.8)
            , lag_ratio = 0.1)
        , run_time = 3)

        # Bring back the original chart
        original_chart_group.clear_updaters()
        self.play(
            FadeOut(VGroup(case1, case2, case1_label, case2_label), shift = DOWN*7),
            FadeIn(original_chart_group, shift = DOWN*7)
        , run_time = 1.5)
        chart, nums, pairs = original_chart_group[0], original_chart_group[1], original_chart_group[2]
        self.wait(0.5)

        # Indicate pairs again to show uniqueness
        self.play(AnimationGroup(*[Indicate(pair) for pair in pairs], lag_ratio = 0.1), run_time = 3)
        self.wait(2)

        # Put each pair of numbers on a coordinate grid
        number_plane = NumberPlane(
            x_range = [0, 5],
            y_range = [0, 5]
        ).set_width(4.5).to_edge(RIGHT, buff = 1)
        number_plane.remove(number_plane.faded_lines)
        x_labels = number_plane.add_coordinate_labels(x_values = [1, 2, 3, 4, 5], y_values = [], font_size = 30, direction = DOWN)
        y_labels = number_plane.add_coordinate_labels(x_values = [], y_values = [1, 2, 3, 4, 5], font_size = 30, direction = LEFT)
        x_labels.set_color(increasing_sequence_color)
        y_labels.set_color(decreasing_sequence_color)
        points = Group(*[
            Group(GlowDot(), TrueDot()).set_color(n_color).move_to(number_plane.c2p(x, y))
            for (x, y) in lis_lds_lengths
        ])
        point_labels = pairs.copy()
        for point, label in zip(points, point_labels):
            label.scale(0.6).next_to(point, UR, buff = -0.1)
        self.play(
            original_chart_group.animate(run_time = 2).scale(0.55).to_edge(LEFT, buff = 1),
            FadeIn(number_plane, shift = LEFT*6, run_time = 2),
            AnimationGroup(*[
                AnimationGroup(
                    TransformFromCopy(pair, label, path_arc = -PI*0.2),
                    FadeIn(point)
                , lag_ratio = 0.6, run_time = 2 + i*0.2)
                for i, (point, pair, label) in enumerate(zip(points, pairs, point_labels))
            ])
        )
        self.wait(1)

        # Draw a rectangle bounding the points
        unit_size = number_plane.background_lines[1].get_y() - number_plane.background_lines[0].get_y()
        rect = Rectangle(
            width = unit_size*3,
            height = unit_size*4,
            fill_opacity = 0.4,
            fill_color = TEAL,
            stroke_width = 4,
            stroke_color = TEAL
        ).align_to(number_plane.c2p(0, 0), DL)
        self.bring_to_back(rect)
        self.play(
            DrawBorderThenFill(rect, stroke_width = 6)
        , run_time = 2)
        self.wait(2)

        # Show the dimensions
        width_brace = Brace(rect, DOWN, buff = 0.5)
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
            ], lag_ratio = 0.1)
        )
        self.wait(1)

        height_brace = Brace(rect, LEFT, buff = 0.5)
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
            ], lag_ratio = 0.1)
        )
        self.wait(1)

        # Circle the lattice points
        lattice_points = VGroup()
        for i in range(3):
            for j in range(4):
                point = Circle(
                    radius = 0.15, stroke_width = 3, stroke_color = WHITE
                ).move_to(number_plane.c2p(i + 1, j + 1))
                lattice_points.add(point)
        self.play(AnimationGroup(*[ShowCreation(point) for point in lattice_points], lag_ratio = 0.15))

        # Write the inequality up top
        inequality.scale(0.8).set_x(0).to_edge(UP, buff = 0.6)
        self.play(
            AnimationGroup(
                TransformMatchingShapes(width_label.copy(), inequality["LIS"], path_arc = -PI*0.35),
                TransformMatchingShapes(height_label.copy(), inequality["LDS"], path_arc = -PI*0.2),
                GrowFromCenter(inequality[R"\cdot"], path_arc = -PI*0.3),
                Write(inequality[R"\ge N"])
            , lag_ratio = 0.3, run_time = 2)
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
        grid.tiles.set_stroke(width = 3)
        for hole in grid.holes:
            hole.border.set_color(WHITE)

        # Label N = k^2
        n_color = YELLOW
        k_color = TEAL
        brace = Brace(grid, UP)
        label = brace.get_tex("N = k^2", font_size = 60, tex_to_color_map = {"N": n_color, "k": k_color}).shift(UP*0.2)
        self.camera.frame.save_state()
        self.play(
            AnimationGroup(
                self.camera.frame.animate(run_time = 1.5).scale(1.1).shift(UP*0.5),
                AnimationGroup(
                    GrowFromEdge(brace, DOWN),
                    Write(label)
                , lag_ratio = 0.2)
            , lag_ratio = 0.2)
        )

        # Focus on the Xs
        self.play(
            grid.background.animate.fade(0.9),
            grid.lines.animate.fade(0.9),
            grid.tiles.animate.fade(0.9),
         run_time = 2)
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
            path.set_stroke(width = 6, color = increasing_sequence_color)
        for path, s in zip(paths, increasing_sequences):
            self.add(path)
            grid.holes.save_state()
            for i in s:
                grid.holes[i].border.set_stroke(color = increasing_sequence_color)
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
            grid.tiles.set_stroke(width = 3)
            grid.holes.set_stroke(width = 2)
            for hole in grid.holes:
                hole.border.set_color(WHITE)
            self.add(grid)
            self.wait(0.1)
        self.wait(2)
        grid.save_state()

        # Divide the holes up into k groups of k
        holes_sorted_by_y = VGroup(*sorted(grid.holes, key = lambda hole: hole.get_y()))
        groups_increasing = VGroup(*[
            holes_sorted_by_y[i*k:(i + 1)*k]
            for i in range(k)
        ])
        rect = Rectangle(
            width = 5.5,
            height = 0.45,
            fill_opacity = 0,
            stroke_width = 3,
            stroke_color = k_color
        ).round_corners(0.2)
        x, y, _ = groups_increasing[0][-1].get_center() - groups_increasing[0][0].get_center()
        group_rects_increasing = VGroup(*[rect.copy().rotate(np.arctan2(y, x)).move_to(group) for group in groups_increasing])
        self.remove(grid.holes)
        self.add(grid.holes)
        self.play(AnimationGroup(*[FadeIn(rect) for rect in group_rects_increasing], lag_ratio = 0.2))

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
                    width = 5, color = increasing_sequence_color
                )
            )

        for line in increasing_path[:-1]:
            self.play(ShowCreation(line), run_time = 2)
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
        self.play(ShowCreation(increasing_path[-1]), run_time = 2)
        self.wait(1)

        # Add labels of 1 through k on top of the holes
        labels_increasing = VGroup(*[
            Tex(
                str(i) if i < k - 1 else R"\cdots" if i == k - 1 else "k"
            ).next_to(
                grid.holes[increasing_sequence[i - 1]], UP + LEFT*0.1 if i < k - 1 else UP, buff = 0.2
            ).set_color(
                k_color
            ).set_stroke(
                width = 10, color = BLACK, behind = True
            )
            for i in range(1, k + 1)
        ])
        brace_label_group = VGroup(brace, label)
        brace_label_group.save_state()
        self.play(
            grid.animate.fade(0.8),
            brace_label_group.animate.fade(0.8),
            group_rects_increasing.animate.fade(0.8),
            AnimationGroup(*[FadeIn(label, shift = UP*0.2) for label in labels_increasing], lag_ratio = 0.2)
        )
        self.wait(2)

        # Show the decreasing subsequence case
        holes_sorted_by_x = VGroup(*sorted(grid.holes, key = lambda hole: hole.get_x()))
        groups_decreasing = VGroup(*[
            holes_sorted_by_x[i*k:(i + 1)*k]
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
            AnimationGroup(*[FadeIn(rect) for rect in group_rects_decreasing], lag_ratio = 0.2)
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
                width = 5, color = decreasing_sequence_color
            )

        labels_decreasing = VGroup(*[
            Tex(
                str(i) if i < k - 1 else R"\cdots" if i == k - 1 else "k"
            ).next_to(
                grid.holes[decreasing_sequence[i - 1]], UP + RIGHT*0.1
            ).set_color(
                k_color
            ).set_stroke(
                width = 10, color = BLACK, behind = True
            )
            for i in range(1, k + 1)
        ])
        self.play(
            ShowCreation(decreasing_path, run_time = 3),
            grid.animate.fade(0.8),
            brace_label_group.animate.fade(0.8),
            group_rects_decreasing.animate.fade(0.8),
            AnimationGroup(*[FadeIn(label, shift = UP*0.2) for label in labels_decreasing], lag_ratio = 0.2, run_time = 3)
        )
        self.wait(2)

        # Show the product
        equation = Tex(R"k \cdot k = k^2 = N", font_size = 80).shift(RIGHT*7 + UP*0.7)
        equation["k"][0].set_color(increasing_sequence_color)
        equation["k"][1].set_color(decreasing_sequence_color)
        equation["k"][2].set_color(k_color)
        equation["N"].set_color(n_color)

        increasing_path_group.generate_target()
        increasing_path_group.target.restore()
        increasing_path_group.target[1].set_fill(color = increasing_sequence_color)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    FadeOut(VGroup(brace_1_group, brace_2_group)),
                    FadeOut(group_rects_decreasing),
                    brace_label_group.animate.restore(),
                    MoveToTarget(increasing_path_group),
                    labels_decreasing.animate.set_fill(color = decreasing_sequence_color),
                    grid.animate.restore(),
                    self.camera.frame.animate.shift(RIGHT*3.2)
                , run_time = 2),
                AnimationGroup(
                    ReplacementTransform(labels_increasing[-1].copy().set_stroke(width = 0), equation[0]),
                    GrowFromCenter(equation[1]),
                    ReplacementTransform(labels_decreasing[-1].copy().set_stroke(width = 0), equation[2]),
                    FadeIn(equation[3:6]),
                    FadeIn(equation[6:])
                , lag_ratio = 0.4)
            , lag_ratio = 0.3)
        )


class LISEqualsK(InteractiveScene):
    def construct(self):
        # Write the equation
        increasing_sequence_color = GREEN_D
        k_color = TEAL
        equation = Tex(
            R"\text{LIS} = 3 = k", font_size = 70, tex_to_color_map = {"LIS": increasing_sequence_color, "k": k_color}
        ).to_edge(RIGHT, buff = 1)
        self.play(Write(equation), run_time = 2)
        self.wait(2)

        # Generalize
        generalized_equation = Tex(
            R"\text{LIS} = k", font_size = 70, tex_to_color_map = {"LIS": increasing_sequence_color, "k": k_color}
        ).move_to(equation)
        self.play(TransformMatchingShapes(equation, generalized_equation))

class PiCreatureReactions(InteractiveScene):
    def construct(self):
        # Add a pi creature
        randy = Randolph(flip_at_start = True)
        self.add(randy)
        self.play(FadeIn(randy, shift = LEFT))

        # React to things
        self.play(randy.change("hooray"), run_time = 1)
        self.wait(2)
        self.play(Blink(randy))
        self.wait(4)
        self.play(Blink(randy))
        self.wait(2)
        self.play(randy.change("confused", UP*4 + LEFT))
        self.wait(1)
        self.play(Blink(randy))
        self.wait(1)
        self.play(randy.change("pondering", LEFT*3))
        self.wait(2)
        self.play(randy.change("thinking", LEFT*3))
        self.wait(1)
        self.play(Blink(randy))
        self.wait(3)

class PassingFlashes(InteractiveScene):
    def construct(self):
        # Do some passing flashes
        self.play(VShowPassingFlash(Rectangle(width = 4, height = 1, stroke_width = 4, stroke_color = YELLOW).insert_n_curves(100), time_width = 3), run_time = 4)
        self.wait(1)
        self.play(VShowPassingFlash(Rectangle(width = 4.5, height = 1, stroke_width = 4, stroke_color = YELLOW).insert_n_curves(100), time_width = 3), run_time = 4)




class IMODetails(InteractiveScene):
    def construct(self):
        # Write "International Math Olympiad"
        imo_text = TexText("International Math Olympiad", font_size = 70).set_opacity(0.9).set_stroke(width = 7, color = BLACK, behind = True)
        imo_logo = ImageMobject("IMO_logo").set_opacity(0.9)
        self.play(FadeIn(imo_logo, shift = OUT*2), Write(imo_text, stroke_color = WHITE))
        self.wait(2)
        imo_text_shortened = TexText("IMO", font_size = 100).set_stroke(width = 7, color = BLACK, behind = True)
        self.play(TransformMatchingShapes(imo_text, imo_text_shortened), run_time = 1.5)
        self.wait(2)

        # Shift it up to the top
        self.play(Group(imo_logo, imo_text_shortened).animate.scale(0.3).to_edge(UP, buff = 0.4))
        self.wait(1)

        # Fade in boxes for the problems underneath
        problems = VGroup()
        for i in range(6):
            rect = Rectangle(width = 6, height = 1.5, fill_opacity = 1, fill_color = TEAL_A, stroke_width = 0).round_corners(0.2)
            label = TexText(R"\text{Problem }" + str(i + 1)).set_color(BLACK)
            label.set_z_index(1)
            problem = VGroup(rect, label)
            problems.add(problem)
        problems.arrange_in_grid(n_cols = 2, h_buff = 2, v_buff = 0.5, fill_rows_first = False).set_width(10).to_edge(DOWN, buff = 1)
        self.play(LaggedStartMap(FadeIn, problems, shift = UP*0.2, lag_ratio = 0.1))

        # Write labels for days underneath
        day1_label = TexText("Day 1").next_to(problems[:3], UP, buff = 0.4)
        day2_label = TexText("Day 2").next_to(problems[3:], UP, buff = 0.4)
        self.play(Write(day1_label), Write(day2_label))
        self.wait(2)

        # Pi Creatures react to each one's difficulty
        creatures = VGroup(*[
            PiCreature("pondering").match_height(problem).next_to(problem, LEFT).look_at(problem)
            for problem in problems
        ])
        self.play(LaggedStartMap(FadeIn, creatures, shift = RIGHT*0.2, lag_ratio = 0.1))
        self.wait(2)

        expressions = ["pondering", "confused", "horrified"]
        colors = [RED_B, interpolate_color(RED_B, RED_E, 0.5), RED_E]
        hard = TexText("Hard").set_color(colors[0]).match_y(problems[0])
        brutal = TexText("Brutal").set_color(colors[2]).match_y(problems[2])
        arrow = Arrow(ORIGIN, DOWN*2.2, thickness = 3).move_to(VGroup(hard, brutal))
        VGroup(hard, arrow, brutal).next_to(problems, RIGHT)
        self.play(
            AnimationGroup(*[
                creature.change(expressions[i % 3]).look_at(problem)
                for i, (creature, problem) in enumerate(zip(creatures, problems))
            ]),
            AnimationGroup(
                Write(hard),
                GrowArrow(arrow),
                Write(brutal)
            , lag_ratio = 0.4),
            AnimationGroup(*[
                problem[0].animate.set_color(colors[i % 3])
                for i, problem in enumerate(problems)
            ], run_time = 2)
        )
        self.wait(2)

        # Change the label to "2025 IMO"
        year_label = TexText("2025").match_height(imo_text_shortened).set_stroke(width = 7, color = BLACK, behind = True)
        imo_text_shortened.generate_target()
        VGroup(year_label, imo_text_shortened.target).arrange(buff = 1.7).match_y(imo_text_shortened)
        self.play(MoveToTarget(imo_text_shortened), FadeIn(year_label, shift = RIGHT*0.3), run_time = 2)