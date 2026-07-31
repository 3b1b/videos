import math
import random
from manim_imports_ext import *

class Grid(VGroup):
    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vertical_lines = VGroup(*[Line(RIGHT*i, RIGHT*i + DOWN*n) for i in range(n + 1)])
        self.horizontal_lines = VGroup(*[Line(DOWN*i, DOWN*i + RIGHT*n) for i in range(n + 1)])
        self.grid = VGroup(self.vertical_lines, self.horizontal_lines).set_stroke(width = 2, color = WHITE, opacity = 0.6)
        self.add(self.grid)
        self.n = n
        self.set_scale_stroke_with_zoom(True)

    def position_at_coordinates(self, tile_or_hole, i, j):
        tile_or_hole.align_to(self.vertical_lines[i], LEFT).align_to(self.horizontal_lines[j], UP)

class Tile(Rectangle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, fill_opacity = 0.9, fill_color = BLUE, stroke_width = 4, stroke_color = WHITE, **kwargs)
        self.round_corners(0.05)
        self.set_scale_stroke_with_zoom(False)

class Hole(VGroup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.square = Square(
            side_length = 1, fill_opacity = 1, fill_color = "#444444", stroke_width = 4, stroke_color = YELLOW, stroke_opacity = 0.8
        )
        self.cross = Cross(self.square, stroke_width = 2).scale(0.95)
        self.add(self.square, self.cross)
        self.set_scale_stroke_with_zoom(False)

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
        # Define a function to generate the optimal grid based on k
        def get_optimal_grid(k):
            n = k*k
            grid = Grid(n)

            holes = VGroup(*[Hole() for _ in range(n)])
            for i, hole in enumerate(holes):
                grid.position_at_coordinates(hole, k - 1 - i//k + (i % k)*k, i)

            main_tiles = VGroup(*[Tile(k, k) for _ in range((k - 1)*(k - 1))])
            for i, tile in enumerate(main_tiles):
                grid.position_at_coordinates(tile, k - 1 - i//(k - 1) + (i % (k - 1))*k, i + i//(k - 1) + 1)

            ur_tiles = VGroup(*[Tile(k, j) for j in range(1, k)])
            for i, tile in enumerate(ur_tiles):
                grid.position_at_coordinates(tile, (i + 1)*k, 0)

            dr_tiles = VGroup(*[Tile(j, k) for j in range(1, k)])
            for i, tile in enumerate(dr_tiles):
                grid.position_at_coordinates(tile, n - i - 1, (i + 1)*k)

            dl_tiles = VGroup(*[Tile(k, j) for j in range(1, k)])
            for i, tile in enumerate(dl_tiles):
                grid.position_at_coordinates(tile, n - (i + 2)*k, n - i - 1)

            ul_tiles = VGroup(*[Tile(j, k) for j in range(1, k)])
            for i, tile in enumerate(ul_tiles):
                grid.position_at_coordinates(tile, 0, n - (i + 2)*k)

            all_tiles = VGroup(*main_tiles, *ur_tiles, *dr_tiles, *dl_tiles, *ul_tiles)
            return VGroup(grid, all_tiles, holes)

        # Add a grid with k = 5
        min_k = 5
        max_k = 45

        k_tracker = ValueTracker(min_k)
        k_tracker.current_k = int(k_tracker.get_value())
        grid_container = VGroup(get_optimal_grid(k_tracker.current_k).align_to(ORIGIN, DL))
        self.add(grid_container)
        self.camera.frame.move_to(grid_container).set_height(grid_container.get_height()*1.4)
        self.camera.frame.save_state()
        self.wait(2)

        # Change it to k = 4
        def update_grid(m):
            new_k = round(m.get_value())
            if abs(new_k - m.current_k) == 1:
                m.current_k = new_k
                grid_container.set_submobjects([get_optimal_grid(m.current_k).align_to(ORIGIN, DL)])

        k_tracker.add_updater(update_grid)

        k_tracker.set_value(4)
        k_tracker.update()
        self.play(self.camera.frame.animate.move_to(grid_container).set_height(grid_container.get_height()*1.4))
        self.wait(2)

        # Show the new dimensions
        x_length_label = Tex("16", font_size = 180).next_to(grid_container, DOWN, buff = 1)
        y_length_label = x_length_label.copy().next_to(grid_container, LEFT, buff = 1)
        brace1 = Brace(grid_container, DOWN)
        brace2 = Brace(grid_container, LEFT)
        self.play(GrowFromEdge(brace1, UP), GrowFromEdge(brace2, RIGHT), FadeIn(VGroup(x_length_label, y_length_label)))
        self.wait(2)

        # Count the holes
        circles = VGroup(*[
            Circle(radius = 0.8, stroke_width = 3, stroke_color = PURE_GREEN).move_to(hole)
            for hole in grid_container[0][2]
        ])
        hole_numbers = VGroup(*[
            Tex(str(i + 1), font_size = 90).next_to(hole, UP, buff = 0.7)
            for i, hole in enumerate(grid_container[0][2])
        ])
        grid_container.save_state()
        self.play(
            AnimationGroup(
                grid_container.animate.set_opacity(0.2),
                AnimationGroup(*[
                    AnimationGroup(
                        ShowCreation(circle),
                        FadeIn(num, shift = UP*0.4)
                    , lag_ratio = 0.1)
                    for circle, num in zip(circles, hole_numbers)
                ], lag_ratio = 0.05),
                grid_container[0][2].animate.shift(0)
            , lag_ratio = 0.1)
        )
        self.wait(2)
        self.play(
            FadeOut(VGroup(x_length_label, y_length_label, brace1, brace2, circles, hole_numbers)),
            grid_container.animate.restore()
        )

        # Change it to k = 3
        k_tracker.set_value(3)
        k_tracker.update()
        self.play(self.camera.frame.animate.move_to(grid_container).set_height(grid_container.get_height()*1.4))
        self.wait(2)

        # Show the new dimensions again
        x_length_label = Tex("9", font_size = 100).next_to(grid_container, DOWN, buff = 0.7)
        y_length_label = x_length_label.copy().next_to(grid_container, LEFT, buff = 0.7)
        brace1 = Brace(grid_container, DOWN)
        brace2 = Brace(grid_container, LEFT)
        self.play(GrowFromEdge(brace1, UP), GrowFromEdge(brace2, RIGHT), FadeIn(VGroup(x_length_label, y_length_label)))
        self.wait(2)
        self.play(FadeOut(VGroup(x_length_label, y_length_label, brace1, brace2)))

        # Generalize
        tile = grid_container[0][1][2]
        grid_container.save_state()
        k_label_1 = Tex("k", font_size = 100).next_to(tile, DOWN, buff = 0.7)
        k_label_2 = k_label_1.copy().next_to(tile, LEFT, buff = 0.7)
        brace1 = Brace(tile, DOWN)
        brace2 = Brace(tile, LEFT)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    grid_container.animate.set_opacity(0.1),
                    tile.animate.shift(0)
                ),
                AnimationGroup(
                    GrowFromEdge(brace1, UP),
                    GrowFromEdge(brace2, RIGHT),
                    FadeIn(VGroup(k_label_1, k_label_2))
                )
            , lag_ratio = 0.4)
        )

        # Add a slider
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
            grid_container.animate.restore(),
            FadeOut(VGroup(k_label_1, k_label_2, brace1, brace2))
        )
        self.remove(grid_container)
        self.add(grid_container, k_slider_group)

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