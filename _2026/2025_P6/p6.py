import math
import random
from manim_imports_ext import *


class Tile(Rectangle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, fill_opacity = 0.9, fill_color = BLUE, stroke_width = 4, stroke_color = WHITE, **kwargs)
        self.round_corners(0.05)
        self.set_scale_stroke_with_zoom(False)

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
        self.border = Square(
            side_length = 1,
            fill_opacity = 0,
            stroke_width = 4,
            stroke_color = YELLOW,
            stroke_opacity = 1
        )
        self.add(self.background, self.cross, self.border)
        self.set_scale_stroke_with_zoom(False)

class Grid(VGroup):
    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.vertical_lines = VGroup(*[Line(RIGHT*i, RIGHT*i + DOWN*n) for i in range(n + 1)])
        self.horizontal_lines = VGroup(*[Line(DOWN*i, DOWN*i + RIGHT*n) for i in range(n + 1)])
        self.grid = VGroup(self.vertical_lines, self.horizontal_lines).set_stroke(width = 4, color = WHITE, opacity = 0.7)
        self.grid.set_scale_stroke_with_zoom(True)
        
        self.tiles = VGroup()
        self.holes = VGroup()

        self.add(self.grid, self.tiles, self.holes)
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
        tiles = VGroup(*[Tile(random.randint(2, 4), random.randint(2, 4)) for _ in range(5)])
        tiles[0].shift(LEFT*8 + UP*5)
        tiles[1].shift(RIGHT*8 + UP*5)
        tiles[3].shift(LEFT*8 + DOWN*5)
        tiles[4].shift(RIGHT*8 + DOWN*5)

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
                hole.phase = random.uniform(0, 2 * math.pi)
                hole.freq = random.uniform(0.8, 1.5)

                if side_idx < 2:
                    L = max(0.1, (w_val - 1) / 2)
                else:
                    L = max(0.1, (h_val - 1) / 2)

                s = L * math.sin(hole.phase)
                if side_idx == 0:
                    local_pos = np.array([s, h_val / 2 + 0.5, 0])
                elif side_idx == 1:
                    local_pos = np.array([s, -h_val / 2 - 0.5, 0])
                elif side_idx == 2:
                    local_pos = np.array([-w_val / 2 - 0.5, s, 0])
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
                            loc = np.array([s_t, -h_v / 2 - 0.5, 0])
                        elif s_idx == 2:
                            loc = np.array([-w_v / 2 - 0.5, s_t, 0])
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

        # Focus on one of them
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

            if side_idx < 2:
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
                        loc = np.array([s_t, -h_v / 2 - 0.5, 0])
                    elif s_idx == 2:
                        loc = np.array([-w_v / 2 - 0.5, s_t, 0])
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
        tile_above = Tile(3, 5).align_to(right_hole.get_corner(UL), DL)
        tile_below = Tile(4, 5).align_to(right_hole.get_corner(DL), UL)
        self.add(tile_above, tile_below, holes)
        self.play(FadeIn(tile_above, shift = DOWN), FadeIn(tile_below, shift = UP), holes[:3].animate.set_opacity(0.2), run_time = 2)
        self.wait(2)

        # Add holes to those tiles
        extra_holes_above = VGroup(*[
            Hole().next_to(tile_above, [LEFT, UP, RIGHT][i], buff = 0).shift(UP*0.8 if i == 0 else 0)
            for i in range(3)
        ])
        extra_holes_below = VGroup(*[
            Hole().next_to(tile_below, [LEFT, RIGHT, DOWN][i], buff = 0).shift(DOWN*0.8 if i == 0 else 0)
            for i in range(3)
        ])
        hole_above = extra_holes_above[0]
        hole_below = extra_holes_below[0]
        shuffled_extra_holes = list(extra_holes_above) + list(extra_holes_below)
        random.shuffle(shuffled_extra_holes)
        self.play(AnimationGroup(*[FadeIn(hole) for hole in shuffled_extra_holes], lag_ratio = 0.2))
        self.wait(2)
        dashed_line = DashedLine(hole_below, hole_above, dash_length = 0.2, stroke_width = 6).set_color(PURE_RED)
        self.play(
            ShowCreation(dashed_line),
            extra_holes_above[1:].animate.set_opacity(0.2),
            extra_holes_below[1:].animate.set_opacity(0.2)
        )
        self.wait(3)

        # Slide the hole up and down the side of the tile
        extra_tiles_and_holes_group = VGroup(tile_above, tile_below, extra_holes_above, extra_holes_below, dashed_line)
        self.play(VGroup(right_hole, extra_tiles_and_holes_group).animate.shift(UP*0.3), run_time = 3)
        self.play(VGroup(right_hole, extra_tiles_and_holes_group).animate.shift(DOWN*0.6), run_time = 3)
        self.play(FadeOut(VGroup(holes[:3], extra_holes_above[1:], extra_holes_below[1:])))
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
        new_tile = Tile(5, 2).align_to(hole.get_corner(DR), DL)
        self.play(
            FadeOut(dashed_line, shift = LEFT*1.7),
            VGroup(tile_above, hole_above).animate.shift(LEFT*(tile_above.get_right()[0] - right_hole.get_right()[0]))
        , run_time = 2.5)
        self.wait(2)
        self.play(FadeIn(new_tile, shift = LEFT), ShrinkToCenter(hole_above), ShrinkToCenter(hole_below))

        # Change the tiles into squares
        hole = right_hole
        tiles = VGroup(*[Tile(3, 3) for _ in range(4)])
        tiles[0].align_to(tile, UR)
        tiles[1].align_to(tile_above, DR)
        tiles[2].align_to(new_tile, DL)
        tiles[3].align_to(tile_below, UL)
        self.play(
            AnimationGroup(*[
                ReplacementTransform(tile1, tile2)
                for tile1, tile2 in zip([tile, tile_above, new_tile, tile_below], tiles)
            ])
        , run_time = 2)

class WindmillTilings(InteractiveScene):
    def construct(self):
        # Add a grid with k = 5
        min_k = 5
        max_k = 45

        grid = OptimalGrid(min_k).align_to(ORIGIN, DL)
        self.add(grid)
        self.camera.frame.move_to(grid).set_height(grid.get_height()*1.4)
        self.camera.frame.save_state()
        self.wait(2)

        # Change it to k = 4
        new_grid = OptimalGrid(4).align_to(ORIGIN, DL)
        self.play(
            self.camera.frame.animate(run_time = 1).move_to(new_grid).set_height(new_grid.get_height()*1.4),
            ReplacementTransform(grid.grid, new_grid.grid),
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
                grid.animate.set_opacity(0.2),
                AnimationGroup(*[
                    AnimationGroup(
                        ShowCreation(circle),
                        FadeIn(num, shift = UP*0.4)
                    , lag_ratio = 0.1)
                    for circle, num in zip(circles, hole_numbers)
                ], lag_ratio = 0.05),
                grid.holes.animate.shift(0)
            , lag_ratio = 0.1)
        )
        self.wait(2)
        self.play(
            FadeOut(VGroup(x_length_label, y_length_label, brace1, brace2, circles, hole_numbers)),
            grid.animate.restore()
        )

        # Change it to k = 3
        new_grid = OptimalGrid(3).align_to(ORIGIN, DL)
        self.play(
            self.camera.frame.animate(run_time = 1).move_to(new_grid).set_height(new_grid.get_height()*1.4),
            ReplacementTransform(grid.grid, new_grid.grid),
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
        tile = grid[1][2]
        grid.save_state()
        k_label_1 = Tex("k", font_size = 100).next_to(tile, DOWN, buff = 0.7)
        k_label_2 = k_label_1.copy().next_to(tile, LEFT, buff = 0.7)
        brace1 = Brace(tile, DOWN)
        brace2 = Brace(tile, LEFT)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    grid.grid.animate.set_opacity(0.1),
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
        k_slider = NumberLine(
            x_range = [0, 50, 10],
            width = 3,
            include_numbers = True
        )
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
        k_tracker.current_k = int(k_tracker.get_value())
        grid_container = VGroup(OptimalGrid(k_tracker.current_k).align_to(ORIGIN, DL))
        self.clear()
        self.add(grid_container)
        def update_grid(m):
            new_k = round(m.get_value())
            if abs(new_k - m.current_k) == 1:
                m.current_k = new_k
                grid_container.set_submobjects([OptimalGrid(m.current_k).align_to(ORIGIN, DL)])
        k_tracker.add_updater(update_grid)


        # Increase k incrementally up to 45
        self.play(
            k_tracker.animate(run_time = 5).set_value(10),
            self.camera.frame.animate(run_time = 5).reorient(-6, 30, 0, (np.float32(43.11), np.float32(30.29), np.float32(14.27)), 85.80)
        )
        k_tracker.suspend_updating()
        self.wait(2)
        k_tracker.resume_updating()
        self.play(
            k_tracker.animate(run_time = 5).set_value(max_k),
            self.camera.frame.animate(run_time = 5).reorient(-13, 40, 0, (np.float32(654.42), np.float32(252.66), np.float32(461.53)), 1315.09)
        )
        k_tracker.suspend_updating()
        self.wait(2)

        # Show the new size
        x_length_label = Tex(R"2025\\=45^2", font_size = 12000).next_to(grid_container, DOWN, buff = 50)
        y_length_label = x_length_label.copy().next_to(grid_container, LEFT, buff = 60)
        brace1 = Brace(grid_container, DOWN)
        brace2 = Brace(grid_container, LEFT)
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
        self.play(
            self.camera.frame.animate.reorient(7, 46, 0, (np.float32(1110.17), np.float32(171.23), np.float32(372.81)), 1109.67)
        , run_time = 10)
        self.play(
            self.camera.frame.animate.reorient(-24, 62, 0, (np.float32(215.15), np.float32(285.09), np.float32(-107.24)), 249.79)
        , run_time = 10)
        self.play(
            self.camera.frame.animate.reorient(-27, 65, 0, (np.float32(631.07), np.float32(270.25), np.float32(216.58)), 180.95)
        , run_time = 10)
        self.play(
            self.camera.frame.animate.reorient(-18, 58, 0, (np.float32(973.79), np.float32(650.28), np.float32(-62.57)), 1754.60)
        , run_time = 10)

        # Set k back to 5
        self.wait(2)
        k_tracker.resume_updating()
        self.play(
            k_tracker.animate(run_time = 5).set_value(5),
            self.camera.frame.animate(run_time = 5).restore()
        )
        k_tracker.suspend_updating()
        self.wait(1)

        # Count the number of square tiles
        all_tiles = grid_container[0][1]
        square_tiles = all_tiles[:16]
        square_tile_numbers = VGroup(*[
            Tex(str(i + 1), font_size = 150).set_color(BLACK).move_to(tile)
            for i, tile in enumerate(square_tiles)
        ])
        grid_container.save_state()
        holes = grid_container[0][2]
        self.play(
            grid_container.animate.set_opacity(0.1),
            holes.animate.shift(0),
            AnimationGroup(*[
                AnimationGroup(
                    tile.animate(rate_func = there_and_back).set_color(YELLOW).scale(1.1).set_fill(opacity = 0.5),
                    GrowFromCenter(num, run_time = 0.8)
                )
                for tile, num in zip(square_tiles, square_tile_numbers)
            ], lag_ratio = 0.1)
        )
        self.wait(2)

        # Generalize
        self.play(
            AnimationGroup(*[
                num.animate.become(
                    Dot(radius = 0.2).set_color(BLACK).move_to(tile) if i < 16 - 1 - 3 else
                    Tex("(k - 1)^2", font_size = 120).set_color(BLACK).move_to(tile)
                )
                for i, (tile, num) in enumerate(zip(square_tiles[3:], square_tile_numbers[3:]))
            ]),
            holes.animate.shift(0)
        )
        self.wait(1)

        # Save the value (k - 1)^2
        square_tile_count = square_tile_numbers[-1].copy().scale(
            1.9
        ).align_to(
            grid_container, UP
        ).set_x(
            0.5*(grid_container.get_right()[0] + self.camera.frame.get_right()[0])
        ).set_color(
            TEAL_A
        )
        self.play(TransformFromCopy(square_tile_numbers[-1], square_tile_count, path_arc = -PI*0.2), run_time = 1.5)
        self.wait(1)

        # Switch focus to the tiles around the edges
        self.play(
            grid_container.animate.restore(),
            square_tiles.animate.set_opacity(0.5),
            FadeOut(square_tile_numbers)
        , run_time = 2)
        self.wait(2)

        # Count the edge tiles
        edge_tiles = all_tiles[16:]
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
            ]),
            holes.animate.shift(0),
        )
        self.wait(1)

        # Save the value 4(k - 1)
        edge_tile_count = Tex(
            R"+4(k - 1)"
        )
        edge_tile_count[1:].set_color(TEAL_D)
        edge_tile_count.scale(
            square_tile_count[0].get_height()/edge_tile_count[2].get_height()
        ).next_to(
            square_tile_count, DOWN, buff = 0.6
        ).align_to(
            square_tile_count[-2], RIGHT
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
        formula_group = VGroup(square_tile_count, edge_tile_count)
        formula_group.generate_target()
        formula_group.target.arrange(buff = 0.4)
        formula_group.target[1].align_to(formula_group.target[0], DOWN)
        formula_group.target.move_to(self.camera.frame).scale(1.2).shift(UP*5)
        self.play(
            FadeOut(k_slider_group, shift = LEFT*5),
            FadeOut(VGroup(grid_container, edge_tile_numbers), shift = LEFT*20),
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
                Tile(1, 1).match_width(grid.holes[0]).set_color(bar_color) for _ in range(height)
            ]).arrange(UP, buff = 0).match_x(hole).align_to(hole, UP)
            for height, hole in zip(values, grid.holes)
        ])
        column_highlights.set_opacity(0)
        self.add(column_highlights)
        self.play(
            grid.tiles.animate.fade(0.9),
            AnimationGroup(*[
                AnimationGroup(
                    Succession(
                        AnimationGroup(*[
                            square.animate(rate_func = there_and_back).set_opacity(1).set_color(GREEN)
                            for square in column
                        ], lag_ratio = 0.1),
                        FadeOut(column)
                    ) if len(column) > 0 else Point().animate.shift(0),
                    FadeIn(num, shift = UP*0.2, run_time = 0.7)
                , lag_ratio = 0.1)
                for num, column in zip(nums, column_highlights)
            ], lag_ratio = 0.2)
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

        # Show example increasing and decreasing subsequences
        increasing_sequence_color = GREEN_D
        decreasing_sequence_color = RED_D
        increasing_sequence = VGroup(*[
            bars[i]
            for i in [2, 3, 7]
        ])
        bars.save_state()
        self.play(
            AnimationGroup(*[
                bar.animate.set_color(increasing_sequence_color)
                for bar in increasing_sequence
            ], lag_ratio = 0.3)
        )
        self.wait(1)
        self.play(bars.animate.restore())
        self.wait(1)
        decreasing_sequence = VGroup(*[
            bars[i]
            for i in [0, 1, 3, 8]
        ])
        bars.save_state()
        self.play(
            AnimationGroup(*[
                bar.animate.set_color(decreasing_sequence_color)
                for bar in decreasing_sequence
            ], lag_ratio = 0.3)
        )
        self.wait(1)
        self.play(bars.animate.restore())
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
        increasing_sequence = VGroup(*[
            bars[i]
            for i in [2, 3]
        ])
        self.play(
            AnimationGroup(*[
                bar.animate.set_color(increasing_sequence_color)
                for bar in increasing_sequence
            ], lag_ratio = 0.3)
        )
        self.wait(1)
        lis_text = Tex(R"\text{LIS}: 2", font_size = 110).set_color(increasing_sequence_color)
        lds_text = Tex(R"\text{LDS}: 3", font_size = 110).set_color(decreasing_sequence_color)
        lds_text.next_to(lis_text, DOWN, buff = 0.6).align_to(lis_text, LEFT)
        VGroup(lis_text, lds_text).set_y(0).to_edge(RIGHT, buff = 1.5)
        self.play(
            AnimationGroup(
                VGroup(chart, nums, arrow).animate.to_edge(LEFT, buff = 1.5),
                Write(lis_text)
            , lag_ratio = 0.6, run_time = 1.5)
        )
        self.wait(1)


        self.play(bars.animate.set_color(bar_color))
        self.wait(1)
        decreasing_sequence = VGroup(*[
            bars[i]
            for i in [0, 1, 3]
        ])
        bars.save_state()
        self.play(
            AnimationGroup(*[
                bar.animate.set_color(decreasing_sequence_color)
                for bar in decreasing_sequence
            ], lag_ratio = 0.3)
        )
        self.wait(1)
        self.play(Write(lds_text), run_time = 1.5)
        self.wait(1)

        self.play(bars.animate.restore())
        self.wait(1)

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
        chart.generate_target()
        chart.target.set_opacity(1).stretch(1.5, 0).center()
        nums.generate_target()
        nums.target.set_opacity(1)
        for num, bar in zip(nums.target, chart.target[0]):
            num.match_x(bar)
        pair.generate_target()
        pair.target.match_x(chart.target[0][3]).scale(1.3)

        self.play(
            FadeOut(VGroup(arrow, lis_text, lds_text), run_time = 1),
            MoveToTarget(chart, run_time = 2),
            MoveToTarget(nums, run_time = 2),
            MoveToTarget(pair, run_time = 2)
        )
        self.wait(2)

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

        increasing_sequence = VGroup(*[
            bars[i]
            for i in [2, 3, 6]
        ])
        self.play(
            AnimationGroup(*[
                bar.animate.set_color(increasing_sequence_color)
                for bar in increasing_sequence
            ], lag_ratio = 0.3)
        )
        self.wait(1)
        self.play(bars.animate.set_color(bar_color))
        self.wait(1)

        decreasing_sequence = VGroup(*[
            bars[i]
            for i in [5, 6]
        ])
        self.play(
            AnimationGroup(*[
                bar.animate.set_color(decreasing_sequence_color)
                for bar in decreasing_sequence
            ], lag_ratio = 0.3)
        )
        self.wait(1)
        self.play(bars.animate.restore(), nums.animate.restore(), pairs.animate.restore(), FadeOut(arrow), run_time = 2)
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
        self.wait(2)

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

        # Show the increasing subsequence of length x
        increasing_sequence = VGroup(tail[0], tail[3], bar1)
        brace = Brace(increasing_sequence, UP).shift(UP*0.1)
        label = brace.get_tex("x", font_size = 60).set_color(increasing_sequence_color).shift(UP*0.2)
        self.play(
            AnimationGroup(*[
                bar.animate.set_opacity(1).set_color(increasing_sequence_color)
                for bar in increasing_sequence
            ], lag_ratio = 0.1),
            GrowFromEdge(brace, DOWN),
            Write(label)
        )
        self.wait(2)

        # Extend it to the second bar
        brace.generate_target()
        label.generate_target()
        extended_brace = Brace(VGroup(increasing_sequence, bar2), UP).shift(UP*0.1)
        extended_label = extended_brace.get_tex("x + 1", font_size = 60).set_color(increasing_sequence_color).shift(UP*0.2)
        self.play(
            TransformFromCopy(brace, extended_brace),
            TransformMatchingShapes(label.copy(), extended_label),
            bar2.animate.set_color(increasing_sequence_color)
        , run_time = 2)
        self.wait(2)

        # Save the example
        case1 = VGroup(tail, bar1, bar2, base, pair, brace, label, extended_brace, extended_label, cdots).copy()

        # Make the second bar shorter
        self.play(
            FadeOut(VGroup(brace, label, extended_brace, extended_label)),
            tail.animate.set_color(bar_color).set_opacity(tail_opacity),
            bar1.animate.set_color(bar_color),
            bar2.animate.set_color(bar_color).stretch_to_fit_height(0.6*bar1.get_height()).align_to(bar2, DOWN)
        )
        self.wait(2)

        # Show the decreasing subsequence of length 7
        decreasing_sequence = VGroup(tail[1], tail[2], bar1)
        brace = Brace(decreasing_sequence, UP)
        label = brace.get_tex("y", font_size = 60).set_color(decreasing_sequence_color)
        self.play(
            AnimationGroup(*[
                bar.animate.set_opacity(1).set_color(decreasing_sequence_color)
                for bar in decreasing_sequence
            ], lag_ratio = 0.1),
            GrowFromEdge(brace, DOWN),
            Write(label)
        )
        self.wait(2)

        # Extend it to the second bar
        brace.generate_target()
        label.generate_target()
        extended_brace = Brace(VGroup(decreasing_sequence, bar2), UP).align_to(case1[-3], UP)
        extended_label = extended_brace.get_tex("y + 1", font_size = 60).set_color(decreasing_sequence_color).align_to(case1[-2], UP)
        self.play(
            TransformFromCopy(brace, extended_brace),
            TransformMatchingShapes(label.copy(), extended_label),
            bar2.animate.set_color(decreasing_sequence_color)
        , run_time = 2)
        self.wait(2)

        # Save the example
        case2 = VGroup(tail, bar1, bar2, base, pair, brace, label, extended_brace, extended_label, cdots)

        # Show both examples side by side
        case1[1:3].set_stroke(width = 4, color = YELLOW)
        case2.generate_target()
        case2.target[1:3].set_stroke(width = 4, color = YELLOW)
        VGroup(case1, case2.target).scale(0.65).arrange(buff = 0.5).shift(RIGHT*0.5)
        case2.target.align_to(case1, DOWN)
        self.play(
            FadeIn(case1, shift = RIGHT*5),
            MoveToTarget(case2)
        , run_time = 2)
        self.wait(2)

        # Bring back the original chart
        self.play(
            FadeOut(VGroup(case1, case2), shift = DOWN*7),
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
        points_color = YELLOW
        points = Group(*[
            Group(GlowDot(), TrueDot()).set_color(points_color).move_to(number_plane.c2p(x, y))
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
            # self.camera.frame.animate.move_to(number_plane),
            # FadeOut(original_chart_group, shift = LEFT*5),
            DrawBorderThenFill(rect, stroke_width = 6)
        , run_time = 2)
        self.wait(2)

        # Show the dimensions
        width_brace = Brace(rect, DOWN, buff = 0.5)
        width_label = width_brace.get_tex(R"\text{LIS}").set_color(increasing_sequence_color)
        self.play(GrowFromEdge(width_brace, UP), Write(width_label))
        self.wait(1)

        height_brace = Brace(rect, LEFT, buff = 0.5)
        height_label = height_brace.get_tex(R"\text{LDS}").set_color(decreasing_sequence_color)
        self.play(GrowFromEdge(height_brace, RIGHT), Write(height_label))
        self.wait(2)

        # Write the final inequality up top
        final_inequality = Tex(
            R"\text{LIS} \cdot \text{LDS} \ge N",
            font_size = 60,
            tex_to_color_map = {"LIS": increasing_sequence_color, "LDS": decreasing_sequence_color, "N": points_color}
        ).to_edge(UP, buff = 0.6)
        self.play(
            AnimationGroup(
                TransformMatchingShapes(width_label.copy(), final_inequality["LIS"], path_arc = -PI*0.35),
                TransformMatchingShapes(height_label.copy(), final_inequality["LDS"], path_arc = -PI*0.2),
                GrowFromCenter(final_inequality[R"\cdot"], path_arc = -PI*0.3),
                Write(final_inequality[R"\ge N"])
            , lag_ratio = 0.3, run_time = 2)
        )