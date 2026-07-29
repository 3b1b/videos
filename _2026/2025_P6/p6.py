import math
import random
from manim_imports_ext import *

class Grid(VGroup):
    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid = Square(side_length = 1, stroke_width = 0.1, stroke_color = WHITE, stroke_opacity = 0.4).get_grid(n, n, buff = 0)
        self.add(self.grid)
        self.n = n

    def position_at_coordinates(self, tile_or_hole, i, j):
        cell = self.grid[self.n*j + i]
        tile_or_hole.align_to(cell, UL)

class Tile(Rectangle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, fill_opacity = 0.9, fill_color = BLUE, stroke_width = 4, stroke_color = WHITE, **kwargs)
        self.round_corners(0.05)

class Hole(VGroup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.square = Square(
            side_length = 1, fill_opacity = 1, fill_color = "#444444", stroke_width = 4, stroke_color = YELLOW, stroke_opacity = 0.8
        )
        self.cross = Cross(self.square, stroke_width = 2).scale(0.95)
        self.add(self.square, self.cross)

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
        self.wait(1.5)

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
        self.wait(3)

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
            self.camera.frame.animate.scale(0.8),
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
        self.wait(2)

        # Add tiles above and below the hole on the right and show the conflict
        right_hole = holes[3]
        tile_above = Tile(3, 5).align_to(right_hole.get_corner(UL), DL)
        tile_below = Tile(4, 5).align_to(right_hole.get_corner(DL), UL)
        self.play(FadeIn(tile_above, shift = DOWN), FadeIn(tile_below, shift = UP), holes[:3].animate.set_opacity(0.2))
        self.wait(2)
        hole_above = Hole().next_to(tile_above, LEFT, buff = 0).shift(UP*0.8)
        hole_below = Hole().next_to(tile_below, LEFT, buff = 0).shift(DOWN*0.8)
        self.play(FadeIn(hole_above, shift = DOWN*0.3), FadeIn(hole_below, shift = UP*0.3))
        self.wait(2)
        dashed_line = DashedLine(hole_below, hole_above, dash_length = 0.2, stroke_width = 6).set_color(PURE_RED)
        self.play(ShowCreation(dashed_line), run_time = 1)
        self.wait(1)
        self.play(FadeOut(VGroup(dashed_line, holes[:3], hole_above, hole_below)))

        # Slide the hole up and down the side of the tile
        self.play(VGroup(right_hole, tile_above, tile_below).animate.shift(UP*0.3), run_time = 1.5)
        self.play(VGroup(right_hole, tile_above, tile_below).animate.shift(DOWN*0.6), run_time = 1.5)
        right_hole.generate_target()
        right_hole.target.align_to(tile, UP)
        tile_above.generate_target()
        tile_above.target.align_to(right_hole.target.get_top(), DOWN)
        tile_below.generate_target()
        tile_below.target.align_to(right_hole.target.get_bottom(), UP)
        self.play(
            MoveToTarget(right_hole),
            MoveToTarget(tile_above),
            MoveToTarget(tile_below),
            self.camera.frame.animate.move_to(right_hole.target)
        , run_time = 1.5)
        new_tile = Tile(5, 2).align_to(hole.get_corner(DR), DL)
        self.play(
            AnimationGroup(
                tile_above.animate.align_to(right_hole, RIGHT),
                FadeIn(new_tile, shift = LEFT)
            , lag_ratio = 0.4)
        )

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
            grid.set_scale_stroke_with_zoom(True)
            all_tiles.set_scale_stroke_with_zoom(True)
            holes.set_scale_stroke_with_zoom(True)
            return VGroup(grid, all_tiles, holes)

        # Add a grid
        min_k = 2
        max_k = 15

        k_tracker = ValueTracker(min_k)
        k_tracker.current_k = int(k_tracker.get_value())
        grid_container = VGroup(get_optimal_grid(k_tracker.current_k).align_to(ORIGIN, DL))
        self.add(grid_container)
        self.camera.frame.move_to(grid_container)
        self.wait(2)

        # Add a slider
        k_slider = NumberLine(
            x_range = [1, max_k],
            width = 3,
            include_numbers = True,
            numbers_to_exclude = [i for i in range(1, max_k + 1) if 1 < i < max_k]
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

        # Adjust the value of k
        def update_grid(m):
            new_k = round(m.get_value())
            if abs(new_k - m.current_k) == 1:
                m.current_k = new_k
                grid_container.set_submobjects([get_optimal_grid(m.current_k).align_to(ORIGIN, DL)])

        k_tracker.add_updater(update_grid)

        self.camera.frame.save_state()
        self.play(
            k_tracker.animate(run_time = 3).set_value(max_k),
            self.camera.frame.animate(run_time = 5).reorient(-14, 57, 0, (np.float32(110.7), np.float32(9.71), np.float32(33.65)), 131.68)
        )
        k_tracker.suspend_updating()
        self.wait(2)
        k_tracker.resume_updating()
        self.play(
            k_tracker.animate(run_time = 2).set_value(5),
            self.camera.frame.animate(run_time = 4).reorient(0, 0, 0, (np.float32(12), np.float32(12), np.float32(0.0)), 30)
        )
        k_tracker.suspend_updating()
        self.wait(3)
