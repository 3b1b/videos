from __future__ import annotations

from manim_imports_ext import *


def get_lozenge(side_length=1):
    verts = [math.sqrt(3) * LEFT, UP, math.sqrt(3) * RIGHT, DOWN]
    result = Polygon(*verts)
    result.scale(side_length / get_norm(verts[0] - verts[1]))
    return result


class ShowLozenge(InteractiveScene):
    def construct(self):
        # Add Lozenge
        lozenge = get_lozenge()
        lozenge.scale(4)
        lozenge.set_stroke(TEAL)

        arc1 = Arc(-30 * DEGREES, 60 * DEGREES, arc_center=lozenge.get_left(), radius=0.75)
        arc2 = Arc(-150 * DEGREES, 120 * DEGREES, arc_center=lozenge.get_top(), radius=0.5)

        VGroup(lozenge, arc1, arc2).stretch(0.95, 0)  # Remove

        # arc1_label = Tex(R"60^\circ")
        arc1_label = Tex(R"70.5^\circ")
        arc1_label.next_to(arc1, RIGHT, MED_SMALL_BUFF)
        # arc2_label = Tex(R"120^\circ")
        arc2_label = Tex(R"109.5^\circ")
        arc2_label.next_to(arc2, DOWN, MED_SMALL_BUFF)
        angle_labels = VGroup(
            arc1, arc1_label,
            arc2, arc2_label,
        )
        angle_labels.set_z_index(1)

        self.play(
            ShowCreation(lozenge, time_span=(1, 2.5)),
            VShowPassingFlash(lozenge.copy().insert_n_curves(20).set_stroke(width=5), time_width=2),
            run_time=3
        )
        self.play(
            Write(arc1_label),
            ShowCreation(arc1),
        )
        self.play(
            Write(arc2_label),
            ShowCreation(arc2),
        )
        self.add(angle_labels)
        self.wait()

        # Tile the plane
        verts = lozenge.get_anchors()[:4]
        v1 = verts[1] - verts[0]
        v2 = verts[-1] - verts[0]
        row = VGroup(lozenge.copy().shift(x * v1) for x in range(-10, 11))
        rows = VGroup(row.copy().shift(y * v2) for y in range(-10, 11))
        tiles = VGroup(*rows.family_members_with_points())
        tiles.sort(lambda p: get_norm(p))

        for mob in row, rows:
            mob.set_fill(GREY, 1)
            mob.set_stroke(WHITE, 2)
            mob.shift(-tiles[0].get_center())

        n_center_tiles = 501

        self.play(
            self.frame.animate.set_height(40),
            lozenge.animate.set_fill(GREY, 1),
            LaggedStart(
                (TransformFromCopy(lozenge, tile, path_arc=30 * DEGREES) for tile in row),
                lag_ratio=1.0 / len(row),
                time_span=(1, 3),
            ),
            run_time=4
        )
        self.play(
            LaggedStart(
                (TransformFromCopy(row, row2, path_arc=30 * DEGREES) for row2 in rows),
                lag_ratio=1.0 / len(rows),
                run_time=3,
            ),
        )
        self.clear()
        self.add(rows, angle_labels)

        # Squish it
        self.play(FadeOut(angle_labels))
        rows.save_state()
        self.play(rows.animate.stretch(2, 0), run_time=2)
        self.wait()
        self.play(Restore(rows), run_time=2)
        self.play(Write(angle_labels))
        self.wait()


class CubesAsHexagonTiling(InteractiveScene):
    n = 4
    colors = [GREY, GREY, GREY]

    def setup(self):
        super().setup()
        n = self.n

        # Set up axes and camera angle
        self.frame.set_field_of_view(1 * DEGREES)
        self.frame.reorient(135, 55, 0)
        self.axes = ThreeDAxes((-5, 5), (-5, 5), (-5, 5))

        # Add base
        self.base_cube = self.get_half_cube(
            side_length=n,
            shared_corner=[-1, -1, -1],
            grid=True
        )
        self.add(self.base_cube)
        self.add(Point())

        # Block pattern
        self.block_pattern = np.zeros((n, n, n))
        self.cubes = VGroup()

    def pre_populate(self):
        for x in range(self.n**3 // 2):
            new_cube = self.random_new_cube()
            self.add_cube(new_cube)

    def get_new_cube(self, x, y):
        zero_indices = np.where(self.block_pattern[x, y, :] == 0)
        if len(zero_indices) == 0:
            print("Column full")
            return
        min_z = np.min(zero_indices)
        min_y = np.min(np.where(self.block_pattern[x, :, min_z] == 0))
        min_x = np.min(np.where(self.block_pattern[:, min_y, min_z] == 0))

        return self.get_half_cube((min_x, min_y, min_z))

    def random_new_cube(self):
        empty_spaces = np.transpose(np.where(self.block_pattern[:, :, :] == 0))
        x, y, z = random.choice(empty_spaces)
        return self.get_new_cube(x, y)

    def get_random_cube_from_stack(self):
        filled_spaces = np.transpose(np.where(self.block_pattern[:, :, :] == 1))
        x, y, z = random.choice(filled_spaces)
        max_x = np.max(np.where(self.block_pattern[:, y, z] == 1))
        max_y = np.max(np.where(self.block_pattern[max_x, :, z] == 1))
        max_z = np.max(np.where(self.block_pattern[max_x, max_y, :] == 1))
        for cube in self.cubes:
            if all(cube.get_corner([-1, -1, -1]).astype(int) == (max_x, max_y, max_z)):
                return cube
        return self.cubes[-1]

    def add_cube(self, cube):
        self.cubes.add(cube)
        cube.spacer = Mobject()
        self.add(cube, cube.spacer)
        self.refresh_block_pattern()

    def remove_cube(self, cube):
        self.cubes.remove(cube)
        self.remove(cube, cube.spacer)
        self.refresh_block_pattern()

    def refresh_block_pattern(self):
        self.block_pattern[:, :, :] = 0
        for cube in self.cubes:
            coords = cube.get_corner([-1, -1, -1]).astype(int)
            self.block_pattern[*coords] = 1

    def get_half_cube(self, coords=(0, 0, 0), side_length=1, colors=None, shared_corner=[1, 1, 1], grid=False):
        if colors is None:
            colors = self.colors
        squares = Square(side_length).replicate(3)
        if grid:
            for square in squares:
                grid = Square(side_length=1).get_grid(side_length, side_length, buff=0)
                grid.move_to(square)
                square.add(grid)
        axes = [OUT, DOWN, LEFT]
        for square, color, axis in zip(squares, colors, axes):
            square.set_fill(color, 1)
            square.set_stroke(color, 0)
            square.rotate(90.1 * DEGREES, axis)  # Why 0.1 ?
            square.move_to(ORIGIN, shared_corner)
        squares.move_to(coords, np.array([-1, -1, -1]))
        squares.set_stroke(WHITE, 2)

        return squares

    def animate_in_with_rotation(self, cube, color=TEAL, run_time=2):
        cube.save_state()
        cube.rotate(-60 * DEGREES, axis=[1, 1, 1])
        cube.set_fill(color)
        blackness = cube.copy().set_color(BLACK)
        spacer = Mobject()
        self.play(FadeIn(cube))
        self.add(blackness, spacer, cube)
        self.play(Rotate(cube, 60 * DEGREES, axis=[1, 1, 1], run_time=run_time))
        self.play(Restore(cube))
        self.add_cube(cube)
        self.remove(blackness, spacer)

    def animate_out_with_rotation(self, cube, color=TEAL, run_time=2):
        blackness = cube.copy().set_color(BLACK)
        spacer = Mobject()
        self.play(cube.animate.set_fill(color))
        self.add(blackness, spacer, cube)
        self.play(Rotate(cube, 60 * DEGREES, axis=[1, 1, 1], run_time=run_time))
        self.remove(blackness, spacer)
        self.play(FadeOut(cube))
        self.remove_cube(cube)


class AmbientTilingChanges(CubesAsHexagonTiling):
    n = 10
    # n = 5

    def construct(self):
        # Hugely inefficient
        self.pre_populate()
        for x in range(10):
            new_cube = self.random_new_cube()
            self.animate_in_with_rotation(new_cube)
            old_cube = self.get_random_cube_from_stack()
            if old_cube is not new_cube:
                self.animate_out_with_rotation(old_cube)


class RotationMove(InteractiveScene):
    def construct(self):
        # Add hex
        lozenge = Polygon(math.sqrt(3) * LEFT, UP, math.sqrt(3) * RIGHT, DOWN)
        lozenge.move_to(ORIGIN, DOWN)
        hexagon = VGroup(
            lozenge,
            lozenge.copy().rotate(TAU / 3, about_point=ORIGIN),
            lozenge.copy().rotate(2 * TAU / 3, about_point=ORIGIN),
        )
        hexagon.set_fill(TEAL_E, 1)
        hexagon.set_stroke(WHITE, 3)
        hexagon.set_height(3)
        hexagon.move_to(3 * LEFT)

        rot_hex = hexagon.copy()
        hexagon.rotate(-60 * DEGREES)
        rot_hex.move_to(3 * RIGHT)

        arrow1 = Arrow(hexagon, rot_hex, thickness=5, path_arc=60 * DEGREES).shift(DOWN)
        arrow2 = Arrow(rot_hex, hexagon, thickness=5, path_arc=60 * DEGREES).shift(UP)

        self.add(hexagon, rot_hex, arrow1, arrow2)
        self.wait()
        self.play(LaggedStart(
            VShowPassingFlash(arrow1.copy().set_stroke(YELLOW, 3)),
            TransformFromCopy(hexagon, rot_hex, path_arc=60 * DEGREES),
            lag_ratio=0.1,
            run_time=2,
        ))
        self.wait()


class AmbientTilingChangesHexagonBound(AmbientTilingChanges):
    n = 4


class IntroduceHexagonFilling(InteractiveScene):
    N = 15
    tile_color = GREY_C
    highlight_color = TEAL
    drag_to_pan = False

    def setup(self):
        super().setup()
        # Create hexagonal tiling
        lozenge = get_lozenge()
        lozenge.set_stroke(WHITE, 1)
        lozenge.set_fill(self.tile_color, 1)
        lozenge.move_to(ORIGIN, DOWN)
        lozenges = VGroup(lozenge.copy().rotate(theta, about_point=ORIGIN) for theta in np.arange(0, TAU, TAU / 3))

        tiling = VGroup()
        for template in lozenges:
            v1 = template.get_vertices()[0]
            v2 = template.get_vertices()[2]
            for x, y in it.product(*2 * [range(self.N)]):
                tiling.add(template.copy().shift(x * v1 + y * v2))

        self.add(tiling)
        self.tiling = tiling

        # Add hexagon
        hexagon = RegularPolygon(6, radius=self.N, start_angle=90 * DEGREES)
        hexagon.set_stroke(YELLOW, 3)
        self.add(hexagon)
        self.hexagon = hexagon

        self.selected_set = VGroup()

    def construct(self):
        # Just play around
        self.wait(10)
        pass

    def rotate_selection(self):
        trip = self.selected_set
        trip.target = trip.generate_target()
        trip.target.rotate(TAU / 6)
        trip.target.set_fill(self.tile_color)
        self.add(trip)
        self.play(MoveToTarget(trip, path_arc=TAU / 6))
        self.selected_set.clear()
        self.add(self.tiling)

        for tile in self.tiling:
            tile.refresh_bounding_box()

    def fill_with_current_tiles(self):
        # Populate
        starter = self.tiling[0].copy()
        starter.to_corner(UL)

        random_order = VGroup(*self.tiling)
        random_order.shuffle()

        self.remove(self.tiling)
        self.play(
            LaggedStart(
                (TransformFromCopy(starter, tile)
                for tile in random_order),
                lag_ratio=1.0 / len(self.tiling),
                run_time=5
            )
        )
        self.add(self.tiling)

    def on_mouse_release(
        self,
        point: Vect3,
        button: int,
        mods: int
    ) -> None:
        super().on_mouse_release(point, button, mods)
        if len(self.selected_set) == 3:
            return
        mouse_center = self.mouse_point.get_center()
        dists = [get_norm(tile.get_center() - mouse_center) for tile in self.tiling]
        tile = self.tiling[np.argmin(dists)]
        tile.set_fill(self.highlight_color)
        if tile in self.selected_set:
            tile.set_fill(self.tile_color)
            self.selected_set.remove(tile)
        else:
            self.selected_set.add(tile)

    def on_key_release(
        self,
        symbol: int,
        modifiers: int
    ) -> None:
        super().on_key_release(symbol, modifiers)
        if chr(symbol) == "p":
            self.fill_with_current_tiles()

        if len(self.selected_set) != 3:
            return

        if chr(symbol) == "r":
            self.rotate_selection()


class HexagonStack(CubesAsHexagonTiling):
    n = 5
    colors = [BLUE_B, BLUE_D, BLUE_E]

    def construct(self):
        # Add hexagonal stack
        for x in range(self.n):
            for y in range(self.n - x):
                for z in range(self.n - x - y):
                    self.add_cube(self.get_new_cube(x, y))
        self.remove(self.base_cube)
        self.cubes.set_fill(BLUE_D)
        self.wait()


class DrawHexagon(IntroduceHexagonFilling):
    def construct(self):
        self.remove(self.tiling)

        # Test
        tiles = self.tiling
        tiles.add(*tiles.copy().rotate(60 * DEGREES))
        tiles.set_fill(opacity=0)
        tiles.set_stroke(WHITE, 0.5, 0.5)
        frame = self.frame
        hexagon = self.hexagon
        brace = Brace(Line(ORIGIN, 4 * UP), RIGHT)
        brace.next_to(hexagon, RIGHT, SMALL_BUFF)
        brace_label = brace.get_text("4")

        self.play(
            ShowCreation(hexagon),
            VShowPassingFlash(hexagon.copy().set_stroke(width=5).insert_n_curves(20), time_width=2),
            run_time=2
        )
        self.wait()
        self.play(
            GrowFromCenter(brace),
            Write(brace_label),
            FadeIn(tiles),
        )
        self.wait()


class ShowAsThreeD(CubesAsHexagonTiling):
    colors = [BLUE_B, BLUE_D, BLUE_E]

    def construct(self):
        # self.pre_populate()

        # Color
        grey_cubes = self.cubes.copy()
        grey_cubes.set_fill(GREY)
        full_grey = Group()
        for cube in grey_cubes:
            full_grey.add(cube, Point())
        self.add(full_grey)
        self.base_cube.save_state()
        self.base_cube.set_fill(GREY)

        self.wait()
        self.play(
            Restore(self.base_cube),
            FadeOut(full_grey),
        )

        # Change perspective
        self.pre_populate()
        self.pre_populate()
        self.play(
            # self.frame.animate.reorient(118, 79, 0, (-1.01, 0.75, 1.54), 8.00)
            self.frame.animate.reorient(118, 79, 0, (-1.01, 0.75, 1.54), 12.00).set_field_of_view(40 * DEGREES),
            run_time=3
        )
        self.play(
            # self.frame.animate.reorient(163, 83, 0, (1.82, -0.33, 1.89), 8.00),
            self.frame.animate.reorient(163, 83, 0, (1.82, -0.33, 1.89), 12.00),
            run_time=4
        )
        self.play(self.frame.animate.reorient(135, 55, 0, ORIGIN, 8).set_field_of_view(1 * DEGREES), run_time=3)

        # Rotation is adding
        new_cube = self.random_new_cube()
        self.animate_in_with_rotation(new_cube)
        self.wait()
        self.play(
            new_cube.animate.shift(5 * RIGHT + 3 * OUT),
            run_time=6,
            rate_func=there_and_back_with_pause
        )
        self.wait()

        # Add or remove a few
        for x in range(2):
            new_cube = self.random_new_cube()
            self.animate_in_with_rotation(new_cube, run_time=1)
            old_cube = self.get_random_cube_from_stack()
            if old_cube is not new_cube:
                self.animate_out_with_rotation(old_cube, run_time=1)

        # Remove all
        while len(self.cubes) > 0:
            cube = self.get_random_cube_from_stack()
            self.play(FadeOut(cube, 0.25 * OUT, run_time=0.25))
            self.remove_cube(cube)

        self.wait()
        for x in range(self.n**3):
            cube = self.random_new_cube()
            self.play(FadeIn(cube, shift=0.25 * IN, run_time=0.25))
            self.add_cube(cube)


class Project3DCube(InteractiveScene):
    def construct(self):
        # Set axes
        frame = self.frame
        light_source = self.camera.light_source

        frame.reorient(28, 68, 0, (0.99, 0.63, 0.66), 2.89)
        light_source.move_to([3, 5, 7])

        axes = ThreeDAxes(
            (-3, 3), (-3, 3), (-3, 3),
            axis_config=dict(tick_size=0.05)
        )
        axes.set_stroke(GREY_A, 1)
        plane = NumberPlane((-3, 3), (-3, 3))
        plane.axes.set_stroke(GREY_A, 1)
        plane.background_lines.set_stroke(BLUE_E, 0.5)
        plane.faded_lines.set_stroke(BLUE_E, 0.5, 0.25)

        self.add(plane, axes)

        # Add cube
        vertices = np.array(list(it.product(*3 * [[0, 1]])))
        vert_dots = DotCloud(vertices)
        vert_dots.make_3d()
        vert_dots.set_radius(0.025)
        vert_dots.set_color(TEAL)

        cube_shell = VGroup(
            Line(vertices[i], vertices[j])
            for i, p1 in enumerate(vertices)
            for j, p2 in enumerate(vertices[i + 1:], start=i + 1)
            if get_norm(p2 - p1) == 1
        )
        cube_shell.set_stroke(YELLOW, 1)
        cube_shell.set_anti_alias_width(1)
        cube_shell.set_width(1)
        cube_shell.move_to(ORIGIN, [-1, -1, -1])

        self.play(Write(cube_shell, lag_ratio=0.1, run_time=2))
        self.wait()

        # Show the coordinates
        labels = VGroup()
        for vert in vertices:
            coords = vert.astype(int)
            label = Tex(str(tuple(coords)), font_size=12)
            label.next_to(vert, DR, buff=0.05)
            label.rotate(45 * DEGREES, RIGHT, about_point=vert)
            label.set_backstroke(BLACK, 2)
            labels.add(label)

        self.play(
            LaggedStartMap(FadeIn, labels),
            FadeIn(vert_dots),
            frame.animate.reorient(10, 61, 0, (0.9, 0.51, 0.48), 2.44),
            run_time=3,
        )
        self.wait(note="Talk through the coordinates")

        # Show base and top square
        edges = VGroup(*cube_shell)
        edges.sort(lambda p: p[2])

        self.play(
            edges[4:].animate.set_stroke(width=0.5, opacity=0.25),
            labels[1::2].animate.set_opacity(0.1)
        )
        self.wait()
        self.play(
            edges[8:].animate.set_stroke(width=2, opacity=1),
            labels[1::2].animate.set_opacity(1),
            edges[:4].animate.set_stroke(width=0.5, opacity=0.25),
            labels[0::2].animate.set_opacity(0.1)
        )
        self.wait()
        self.play(
            edges.animate.set_stroke(width=1, opacity=1),
            labels.animate.set_opacity(1)
        )

        self.play(FadeOut(labels))

        # Orient to look down the corner
        self.play(frame.animate.reorient(135.795, 55.795, 0, (-0.02, -0.08, 0.05), 3.61), run_time=4)
        self.wait(2, note="Take a moment to look down the corner")
        self.play(frame.animate.reorient(50, 68, 0, (-0.46, 0.29, 0.23), 3.45), run_time=4)

        # Show the flat projection
        diag_vect = Vector([1, 1, 1], thickness=2)
        diag_vect.set_perpendicular_to_camera(frame)
        diag_label = labels[-1].copy()

        proj_mat = self.construct_proj_matrix()
        perp_plane = Square3D().set_width(20)
        perp_plane.set_color(GREY_E, 0.5)
        perp_plane.apply_matrix(proj_mat)

        proj_cube_shell = cube_shell.copy().apply_matrix(proj_mat)
        proj_vert_dots = vert_dots.copy().apply_matrix(proj_mat)

        self.play(
            GrowArrow(diag_vect),
            FadeIn(diag_label, shift=np.ones(3)),
            cube_shell.animate.set_stroke(opacity=0.25),
        )
        self.wait()
        self.play(
            TransformFromCopy(cube_shell, proj_cube_shell),
            TransformFromCopy(vert_dots, proj_vert_dots),
        )

        self.wait(10, note="Talk through projection")
        frame.save_state()
        self.play(
            frame.animate.reorient(134.75, 54.47, 0, (-0.46, 0.29, 0.23), 3.45).set_field_of_view(1 * DEGREES),
            run_time=4
        )
        self.wait()
        self.play(Restore(frame, run_time=3))
        self.wait()

        # Project more cubes down
        cube_grid = VGroup(
            cube_shell.copy().shift(vect)
            for vect in it.product(*3 * [[0, 1, 2]])
        )
        cube_grid.remove(cube_grid[0])
        proj_cube_grid = cube_grid.copy().apply_matrix(proj_mat)
        proj_cube_grid.set_stroke(YELLOW, 2, 0.5)

        ghost_cube = cube_shell.copy().set_opacity(0)
        self.play(
            LaggedStart(
                (TransformFromCopy(ghost_cube, new_cube)
                for new_cube in cube_grid),
                lag_ratio=0.05,
            ),
            frame.animate.reorient(40, 72, 0, (1.25, 1.69, 0.99), 5.10),
            run_time=5
        )
        self.wait()
        self.play(
            TransformFromCopy(cube_grid, proj_cube_grid),
            frame.animate.reorient(60, 68, 0, (0.81, 1.09, 0.94), 5.36),
            run_time=3
        )
        self.wait(note="Any commentary?")
        self.play(
            FadeOut(cube_grid),
            FadeOut(proj_cube_grid),
            FadeOut(diag_label),
            FadeOut(diag_vect),
            FadeOut(vert_dots),
            FadeOut(proj_vert_dots),
            frame.animate.reorient(42, 62, 0, (0.68, 0.48, 0.41), 2.34),
            run_time=2,
        )

        # Show cube faces
        cube = Cube()
        cube.set_color(BLUE_E, 1)
        cube.set_shading(0.75, 0.25, 0.5)
        cube.replace(cube_shell)
        cube.sort(lambda p: np.dot(p, np.ones(3)))
        inner_faces = cube[:3]
        outer_faces = cube[3:]

        for mob in [cube_shell, proj_cube_shell, plane]:  # No axes?
            mob.apply_depth_test()
        self.add(axes, cube, cube_shell, plane, proj_cube_shell)
        self.play(
            FadeIn(cube),
            proj_cube_shell.animate.set_stroke(width=1, opacity=0.2),
        )
        self.wait(10, note="Note the outer faces")
        self.add(axes, inner_faces, cube_shell, plane, proj_cube_shell)
        self.play(
            FadeOut(outer_faces),
            inner_faces.animate.set_submobject_colors_by_gradient(RED, GREEN, BLUE),
        )
        self.wait(10, note="Gesture at inner faces")
        inner_faces.save_state()
        self.play(inner_faces.animate.apply_matrix(proj_mat), run_time=2)
        self.play(inner_faces.animate.space_out_submobjects(1.2), rate_func=there_and_back, run_time=2)
        self.wait(10)

        # Shuffle faces around
        inner_proj_state = inner_faces.copy()
        self.wait()
        self.play(Restore(inner_faces), run_time=2)
        inner_faces.target = inner_faces.generate_target()
        for face, vect in zip(inner_faces.target, [UP, RIGHT, OUT]):
            face.shift(vect)
        outer_state = inner_faces.target.copy()
        self.play(MoveToTarget(inner_faces, lag_ratio=0.5, run_time=3))
        self.wait()
        self.play(inner_faces.animate.apply_matrix(proj_mat), run_time=2)
        self.play(inner_faces.animate.space_out_submobjects(1.2), rate_func=there_and_back, run_time=2)
        self.wait()

        for u in [-1, 1]:
            self.play(Rotate(inner_faces, u * PI / 3, axis=np.ones(3), run_time=2))
            self.wait()

        for group in inner_faces, inner_proj_state:
            for i, mob in enumerate(group):
                mob.shift(i * 0.0001 * IN)
        self.play(Transform(inner_faces, inner_proj_state, lag_ratio=0.5, run_time=3))
        self.wait()
        self.play(Restore(inner_faces), run_time=2)

        # Show coordinates for inner faces
        bases = np.identity(3, dtype=int)
        vects = VGroup(Vector(basis, thickness=2) for basis in bases)
        coord_labels = VGroup(
            Tex(str(tuple(basis)), font_size=16).next_to(basis, UR, buff=0.05).rotate(45 * DEGREES, RIGHT, about_point=basis)
            for basis in bases
        )

        self.play(
            axes.animate.set_stroke(width=0.5),
            plane.axes.animate.set_stroke(width=0.5),
            FadeOut(inner_faces),
            LaggedStartMap(GrowArrow, vects),
            run_time=2
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeIn, coord_labels),
            frame.animate.reorient(9, 63, 0, (1.03, 0.61, 0.56), 2.72),
            run_time=2
        )
        self.wait()

        # Emphasize pairs
        vects_state = vects.copy()
        labels_state = coord_labels.copy()
        last_face = VectorizedPoint()
        ordered_faces = Group(inner_faces[i] for i in [1, 0, 2])
        ordered_faces.set_opacity(0.8)
        ordered_faces.deactivate_depth_test()

        for i in range(3):
            vects_target = vects_state.copy()
            labels_target = labels_state.copy()
            vects_target[i].fade(0.8)
            labels_target[i].fade(0.8)
            self.add(ordered_faces[i], vects, coord_labels)
            self.play(
                Transform(vects, vects_target),
                Transform(coord_labels, labels_target),
                FadeIn(ordered_faces[i]),
                FadeOut(last_face),
            )
            self.wait()

            last_face = ordered_faces[i]

        # Project all the vectors
        proj_vects = VGroup(
            Vector(np.dot(basis, proj_mat.T), thickness=3)
            for basis in np.identity(3)
        )
        proj_coords = VGroup(
            Tex(f"P{str(tuple(basis))}", font_size=16)
            for basis in np.identity(3, dtype=int)
        )
        for label, vect in zip(proj_coords, proj_vects):
            label.move_to(vect.get_end() + 0.25 * vect.get_vector())
            label.rotate(45 * DEGREES, RIGHT)
            label.rotate(45 * DEGREES, OUT)
            vect.set_perpendicular_to_camera(self.frame)
        proj_coords[1].shift(0.25 * UP)
        faces = Group(ordered_faces[2], ordered_faces[0], ordered_faces[1])

        self.add(faces, vects, coord_labels)
        self.play(
            Transform(vects, vects_state),
            Transform(coord_labels, labels_state),
            FadeIn(faces),
            frame.animate.reorient(44, 55, 0, (1.03, 0.61, 0.56), 2.72),
            run_time=2
        )
        self.play(
            Transform(vects, proj_vects),
            Transform(coord_labels, proj_coords),
            faces.animate.apply_matrix(proj_mat),
        )
        self.play(frame.animate.reorient(56, 58, 0, (0.7, 0.32, 0.6), 2.72), run_time=5)

    def add_coordinate_labels(self, axes):
        coordinate_config = dict(font_size=12, buff=0.1)
        axes.add_coordinate_labels(**coordinate_config)
        axes.z_axis.add_numbers(
            **coordinate_config,
            excluding=[0],
            direction=LEFT
        )
        for number in axes.z_axis.numbers:
            number.scale(0.75, about_edge=RIGHT)
            number.rotate(90 * DEGREES, RIGHT)

    def construct_proj_matrix(self):
        diag = normalize(np.ones(3))
        id3 = np.identity(3)
        return np.array([self.project(basis, diag) for basis in id3]).T

    def gram_schmitt(self, vects):
        for i in range(len(vects)):
            for j in range(i):
                vects[i] = self.project(vects[i], vects[j])
            vects[i] = normalize(vects[i])
        return vects

    def project(self, vect, unit_norm):
        """
        Project v1 onto the orthogonal subspace of norm
        """
        return vect - np.dot(unit_norm, vect) * unit_norm


class Project4DCube(Project3DCube):
    def construct(self):
        # Get hypercube data
        frame = self.frame
        hypercube_points, edge_indices = self.get_hypercube_data()

        # Prepare pre-projectiong
        w_shift = 2 * RIGHT + UP + OUT

        cube_verts = np.array(list(it.product(*3 * [[0, 1]])))
        cube_shell = VGroup(
            Line(cube_verts[i], cube_verts[j])
            for i, p1 in enumerate(cube_verts)
            for j, p2 in enumerate(cube_verts[i + 1:], start=i + 1)
            if get_norm(p2 - p1) == 1
        )
        cube_shells = cube_shell.replicate(2)
        cube_shells[1].shift(w_shift)
        edge_connectors = VGroup(Line(v, v + w_shift) for v in cube_verts)

        cube_shells[0].set_stroke(BLUE, 2)
        cube_shells[1].set_stroke(YELLOW, 2)
        edge_connectors.set_stroke(WHITE, 1)

        coord_labels = VGroup()
        for point in hypercube_points:
            label = Tex(str(tuple(point)), font_size=12)
            point_3d = point[:3] + point[3] * w_shift
            label.next_to(point_3d, DR, buff=0.05)
            label.rotate(45 * DEGREES, RIGHT, about_point=point_3d)
            coord_labels.add(label)
        coord_labels.set_backstroke(BLACK, 2)

        low_labels = coord_labels[0::2]
        high_labels = coord_labels[1::2]
        low_labels.set_z_index(1)
        high_labels.set_z_index(1)
        for group in [low_labels, high_labels]:
            group.generate_target()
            for part in group.target:
                part[-2].set_fill(RED)

        # Show lists of coordinates
        titles = VGroup(Text(f"{n}D Cube Vertices") for n in [3, 4])
        coords3d = VGroup(Tex(str(tuple(coords))) for coords in it.product(*3 * [[0, 1]]))
        coords4d = VGroup(Tex(str(tuple(coords))) for coords in it.product(*4 * [[0, 1]]))

        coords3d.scale(0.75).arrange(DOWN, buff=MED_SMALL_BUFF)
        coords4d.scale(0.75).arrange_in_grid(8, 2, v_buff=MED_SMALL_BUFF, h_buff=0.5)

        for title, vect, coords in zip(titles, [LEFT, RIGHT], [coords3d, coords4d]):
            title.move_to(vect * FRAME_WIDTH / 4).to_edge(UP)
            title.add(Underline(title))
            coords.set_backstroke(BLACK, 2)
            coords.next_to(title, DOWN)

        self.add(titles)
        self.add(coords3d)
        self.play(LaggedStartMap(FadeIn, coords4d, shift=0.1 * DOWN, lag_ratio=0.1, run_time=3))
        self.wait()

        label_group3d = VGroup(titles[0], coords3d)
        label_group4d = VGroup(titles[1], coords4d)
        VGroup(label_group3d, label_group4d).fix_in_frame()

        # Show pre-projection
        pre_low_labels = coords4d[0::2].copy()
        pre_low_labels.unfix_from_frame()
        pre_low_labels.set_backstroke(BLACK, 2)

        label_group4d.target = label_group4d.generate_target()
        label_group4d.target.scale(0.5).to_corner(UL)
        label_group4d.target[1][1::2].set_opacity(0.2)

        self.play(
            Write(cube_shells[0]),
            TransformFromCopy(pre_low_labels, low_labels),
            frame.animate.reorient(11, 67, 0, (1.08, 0.47, 0.77), 3.22),
            FadeOut(label_group3d, 3 * LEFT),
            MoveToTarget(label_group4d),
            run_time=3
        )
        self.wait(6, note="Pan somewhat")
        self.play(
            MoveToTarget(low_labels),
            LaggedStart(
                (FlashUnder(label[-3:], color=RED)
                for label in low_labels),
                lag_ratio=0.05,
            )
        )
        self.wait()
        self.play(
            ShowCreation(edge_connectors, lag_ratio=0),
            TransformFromCopy(*cube_shells),
            TransformFromCopy(low_labels, high_labels),
            label_group4d[1][1::2].animate.set_opacity(1),
            run_time=3
        )
        self.wait()
        self.play(
            MoveToTarget(high_labels),
            LaggedStart(
                (FlashUnder(label[-3:], color=RED)
                for label in high_labels),
                lag_ratio=0.05,
            )
        )
        self.wait(20, note="Pan and gesture")

        # Put pre-projection in the corner
        axes = ThreeDAxes((-3, 3), (-3, 3), (-3, 3))
        axes.set_height(12)
        pre_proj_points = np.array([
            *hypercube_points[:8, 1:],
            *(hypercube_points[:8, 1:] + w_shift),
        ])
        pre_proj_frame = VGroup(
            Line(pre_proj_points[i], pre_proj_points[j])
            for i, j in edge_indices
        )
        pre_proj_frame.set_stroke(WHITE, 1)
        pre_proj_frame.generate_target()
        pre_proj_frame.target.fix_in_frame()
        pre_proj_frame.target.set_height(1.0)
        pre_proj_frame.target.rotate(60 * DEGREES, LEFT).rotate(45 * DEGREES, UP).rotate(15 * DEGREES, OUT)
        pre_proj_frame.target.to_corner(UL, buff=LARGE_BUFF)

        cloud = ThoughtBubble(Rectangle(2, 1.5))[0][3]
        cloud.set_fill(GREY_E, 1)
        cloud.to_corner(UL, buff=MED_SMALL_BUFF)
        cloud.fix_in_frame()
        cloud_label = Text("4D")
        cloud_label.next_to(cloud, DOWN)
        cloud_label.fix_in_frame()
        pre_proj_frame.target.move_to(cloud)

        arrow = Arrow(cloud.get_right(), UL, path_arc=-60 * DEGREES, thickness=5)
        arrow.set_fill(border_width=0.5)
        arrow.fix_in_frame()
        arrow_label = TexText("Project along [1, 1, 1, 1]", font_size=24)
        arrow_label.next_to(arrow.pfp(0.15), UR, buff=0.15)
        arrow_label.fix_in_frame()

        self.play(
            FadeOut(label_group4d),
            FadeOut(VGroup(cube_shells, edge_connectors, coord_labels)),
            FadeIn(pre_proj_frame),
        )
        self.add(cloud, pre_proj_frame),
        self.play(
            FadeIn(cloud, time_span=(2, 3)),
            Write(cloud_label, time_span=(2, 3)),
            MoveToTarget(pre_proj_frame),
            frame.animate.reorient(22, 76, 0, (-1.33, 0.51, 0.63), 7.64),
            run_time=3,
        )
        self.wait()
        self.play(
            GrowArrow(arrow, path_arc=-30 * DEGREES),
            Write(arrow_label),
            Write(axes),
        )
        self.wait()

        corner_group = VGroup(cloud, cloud_label, pre_proj_frame, arrow, arrow_label)

        # Project down
        proj_coords = self.project_along_diagonal(hypercube_points)
        proj_points = axes.c2p(*proj_coords.T)
        proj_frame = VGroup(
            Line(proj_points[i], proj_points[j])
            for i, j in edge_indices
        )
        proj_frame.set_stroke(YELLOW, 2)

        self.add(Point(), pre_proj_frame)
        self.play(Transform(pre_proj_frame.copy(), proj_frame.copy(), run_time=3, remover=True))
        self.add(Point(), proj_frame)
        self.wait()

        # Show solid faces
        inner_cells = self.get_rhombic_dodec(side_length=axes.x_axis.get_unit_size())
        inner_cells.set_color(BLUE_E, 1)

        axes.apply_depth_test()
        self.play(
            FadeOut(proj_frame),
            FadeIn(inner_cells),
        )
        self.wait()

        # Break up inner cells
        space_factor = 1.5
        ghost_cells = inner_cells.copy()
        ghost_cells.deactivate_depth_test()
        ghost_cells.set_opacity(0.1)
        inner_cells.target = inner_cells.generate_target()
        inner_cells.target.space_out_submobjects(space_factor)

        for group in [inner_cells.target, ghost_cells]:
            group.set_submobject_colors_by_gradient(RED_E, GREEN_E, BLUE_E, PINK)

        self.play(
            MoveToTarget(inner_cells),
            FadeOut(corner_group),
            run_time=2
        )
        self.wait()
        self.play(
            FadeOut(inner_cells),
            FadeIn(ghost_cells, scale=0.8),
        )

        # Projected bases
        proj_bases = self.construct_proj_matrix().T
        proj_basis_vectors = VGroup(
            Vector(axes.c2p(*basis))
            for basis in proj_bases
        )
        proj_basis_labels = VGroup(
            Tex(Rf"P{tuple(basis)}", font_size=24)
            for basis in np.identity(4).astype(int)
        )
        for vect, label in zip(proj_basis_vectors, proj_basis_labels):
            vect.set_perpendicular_to_camera(frame)  # Always?
            label.next_to(vect.get_end(), RIGHT, SMALL_BUFF)
            label.rotate(45 * DEGREES, about_point=vect.get_end(), axis=RIGHT)
        proj_basis_labels[0].shift(0.25 * DOWN) 

        self.play(
            axes.animate.set_stroke(width=1),
            LaggedStartMap(GrowArrow, proj_basis_vectors, suspend_mobject_updating=True),
            FadeIn(proj_basis_labels),
        )
        self.wait()

        self.play(
            ghost_cells.animate.space_out_submobjects(space_factor).set_opacity(0.5),
            run_time=2
        )

        # Iterate through triplets
        ordered_cells = Group(ghost_cells[i] for i in [0, 1, 2, 3])
        vect_groups = VGroup(
            VGroup(vect, label)
            for vect, label in zip(proj_basis_vectors, proj_basis_labels)
        )
        self.add(ordered_cells)
        for i in range(4):
            vect_groups.generate_target()
            vect_groups.target.set_fill(opacity=1)
            vect_groups.target[i].set_fill(opacity=0.1)
            ordered_cells.generate_target()
            ordered_cells.target.set_opacity(0.05)
            ordered_cells.target[i].set_opacity(0.5)
            self.play(
                MoveToTarget(vect_groups),
                MoveToTarget(ordered_cells),
            )
            self.wait()

        self.play(
            FadeOut(vect_groups),
            FadeOut(ordered_cells),
            FadeIn(inner_cells),
        )

        # Play more
        self.wait(5)
        self.play(inner_cells.animate.space_out_submobjects(1.0 / space_factor))
        self.wait(10)

        # Show inversion
        self.play(FadeOut(inner_cells[1:]))
        self.wait()
        self.play(
            inner_cells[0].animate.move_to(-inner_cells[0].get_center()),
            rate_func=there_and_back_with_pause,
            run_time=6,
        )
        self.wait()
        self.play(FadeIn(inner_cells[1:]))
        self.wait()

        inner_cells.save_state()
        self.play(
            LaggedStart(
                (cell.animate.move_to(-cell.get_center())
                for cell in inner_cells),
                group=inner_cells,
                group_type=Group,
                run_time=3,
                lag_ratio=0.25
            ),
        )
        self.wait()
        self.play(Restore(inner_cells))
        self.wait()

        # Tile space
        N = 4
        small_space_factor = 1.1
        tiling = Group()

        for i in range(4):
            indices = list(range(4))
            indices.remove(i)
            bases = proj_bases[indices]
            for coords in it.product(*3 * [list(range(N))]):
                vect = axes.c2p(*np.dot(coords, bases))
                new_cell = inner_cells[i].copy().shift(vect)
                tiling.add(new_cell)

        tiling.space_out_submobjects(small_space_factor)
        tiling.sort(lambda p: get_norm(p))
        colored_tiling = tiling.copy()
        tiling.set_color(BLUE_E)

        self.play(
            FadeOut(inner_cells[:2]),
            FadeOut(inner_cells[3:]),
            axes.animate.set_stroke(width=0, opacity=0),
        )
        self.wait(15)

        self.remove(inner_cells)
        self.play(
            LaggedStart(
                (TransformFromCopy(inner_cells[2], cell)
                for cell in tiling),
                group_type=Group,
                lag_ratio=0.05,
            ),
            frame.animate.reorient(19, 65, 0, (1.39, 1.51, 0.57), 21.55),
            run_time=8
        )
        self.wait(20)
        self.play(frame.animate.reorient(36, 66, 0, (-1.32, 0.25, -0.7), 22.55), run_time=3)
        self.play(frame.animate.increment_theta(PI), run_time=10)
        self.play(Transform(tiling, colored_tiling))
        self.wait()

    def get_hypercube_data(self):
        points = np.array(list(it.product(*4 * [[0, 1]])))
        edge_indices = [
            (i, j)
            for i, p1 in enumerate(points)
            for j, p2 in enumerate(points[i + 1:], start=i + 1)
            if get_norm(p2 - p1) == 1
        ]

        return points, edge_indices

    def project_along_diagonal(self, points):
        if not hasattr(self, "diag_4d_projection"):
            self.diag_4d_projection = self.construct_proj_matrix()
        return np.dot(points, self.diag_4d_projection.T)

    def construct_proj_matrix(self):
        diag = normalize(np.ones(4))
        id4 = np.identity(4)
        pre_basis = np.array([diag, id4[1] - id4[0], id4[2], id4[3]])
        basis = self.gram_schmitt(pre_basis)
        return basis[1:, :]

    def get_rhombic_dodec(self, side_length=1):
        cube = Cube()
        cube.set_width(side_length)
        cube.move_to(ORIGIN, -np.ones(3))

        proj_bases = self.project_along_diagonal(np.identity(4))
        cells = Group()
        for i in range(4):
            indices = list(range(4))
            indices.remove(i)
            mat = proj_bases[indices]
            cells.add(cube.copy().apply_matrix(mat.T, about_point=ORIGIN))

        cells.set_color(BLUE_E, 1)
        return cells


class ShowRhombicDodecTesselation(Project4DCube):
    def construct(self):
        # Create tiling pattern
        frame = self.frame
        proj_bases = self.project_along_diagonal(np.identity(4))
        dodec = self.get_rhombic_dodec()

        N = 6
        coords = [
            coords
            for coords in it.product(*4 * [list(range(N))])
            if sum(coords) == 8
        ]
        pieces = Group(
            dodec.copy().shift(np.dot(coord, proj_bases))
            for coord in coords
        )
        pieces.sort(lambda p: get_norm(p))
        for piece in pieces:
            piece.set_color(random_bright_color(hue_range=(0.5, 0.55), luminance_range=(0.25, 0.5)))
            piece.save_state()

        pieces.space_out_submobjects(1.25)
        pieces.set_opacity(0)

        frame.reorient(31, 77, 0).set_height(6)
        self.play(
            LaggedStart(
                (Restore(piece)
                for piece in pieces),
                lag_ratio=0.5,
                group_type=Group
            ),
            frame.animate.reorient(-93, 68, 0, (0.07, 0.22, 0.17), 15),
            run_time=12,
        )


class CubeToHypercubeAnalogy(InteractiveScene):
    def construct(self):
        # Vertices

        # Numbers of faces/cells

        # Which specific cells touch the origin
        pass


class AskStripQuestion(InteractiveScene):
    def construct(self):
        # Add circle
        radius = 2.5
        circle = Circle(radius=radius)
        circle.set_stroke(YELLOW, 2)
        radial_line = Line(circle.get_center(), circle.get_right())
        radial_line.set_stroke(WHITE, 2)
        radius_label = Integer(1)
        radius_label.next_to(radial_line, UP, SMALL_BUFF)

        self.play(
            ShowCreation(radial_line),
            FadeIn(radius_label, RIGHT)
        )
        self.play(
            Rotate(radial_line, 2 * PI, about_point=circle.get_center()),
            ShowCreation(circle),
            run_time=2
        )
        self.wait()

        # Show first strip
        r0_tracker = ValueTracker(0.2)
        r1_tracker = ValueTracker(0.8)
        strip1 = always_redraw(lambda: self.get_strip(
            circle,
            r0_tracker.get_value(), r1_tracker.get_value(),
            theta=TAU / 3,
            color=TEAL,
            include_arrow=True,
            label=""
        ))
        radius = radial_line.get_length()
        width_label = DecimalNumber(0)
        width_label.add_updater(lambda m: m.set_value(r1_tracker.get_value() - r0_tracker.get_value()))
        width_label.add_updater(lambda m: m.set_height(min(0.33, 0.5 * strip1.submobjects[0].get_height())))
        width_label.always.next_to(strip1.submobjects[0].get_center(), UR, SMALL_BUFF)

        d_label = Tex(R"d_1")
        d_label.move_to(width_label, DL)

        strip1.suspend_updating()
        self.animate_strip_in(strip1)
        self.wait()

        self.play(Write(width_label, suspend_mobject_updating=True))
        strip1.resume_updating()
        self.play(
            r0_tracker.animate.set_value(0.49),
            r1_tracker.animate.set_value(0.51),
            run_time=4,
            rate_func=there_and_back,
        )
        strip1.clear_updaters()
        width_label.clear_updaters()
        self.wait()
        self.play(ReplacementTransform(width_label, d_label))
        strip1.add(d_label)

        # Add first couple strips
        new_strips = VGroup(
            self.get_strip(
                circle, r0, r1, angle,
                color=color,
                include_arrow=True,
                label=f"d_{n}",
            )
            for n, r0, r1, angle, color in [
                (2, 0.5, 0.75, 2 * TAU / 3, GREEN),
                (3, 0.1, 0.3, 0.8 * TAU, BLUE_D),
                (4, 0.4, 0.7, 0.1 * TAU, BLUE_B),
            ]
        )

        for strip in new_strips:
            self.animate_strip_in(strip)

        # Cover in lots of strips
        np.random.seed(0)
        strips = VGroup(
            self.get_strip(
                circle,
                *sorted(np.random.uniform(-1, 1, 2)),
                TAU * np.random.uniform(0, TAU),
                opacity=0.25
            ).set_stroke(width=1)
            for n in range(10)
        )
        self.add(strips, strip1, new_strips, circle, radius_label)
        self.play(FadeIn(strips, lag_ratio=0.5, run_time=3))
        self.wait()

        # Add together all the widths
        frame = self.frame
        arrows = VGroup(strip.submobjects[0] for strip in (strip1, *new_strips))
        d_labels = VGroup(strip.submobjects[1] for strip in (strip1, *new_strips))

        top_expr = Tex(R"d_1 + d_2 + d_3 + d_4 + \cdots + d_n")
        top_expr.to_edge(UP, buff=0)
        d_labels.target = VGroup(
            top_expr[f"d_{n}"][0]
            for n in range(1, 5)
        )

        self.play(
            LaggedStart(
                MoveToTarget(d_labels, lag_ratio=0.01),
                Write(top_expr["+"]),
                Write(top_expr[R"\cdots"]),
                Write(top_expr[R"d_n"]),
                lag_ratio=0.5
            ),
            FadeOut(arrows),
            frame.animate.move_to(UP).set_anim_args(run_time=2)
        )
        self.remove(d_labels)
        self.add(top_expr)
        self.wait()

        # Compress sum
        short_expr = Tex(R"\min\left( \sum_i d_i \right)")
        short_expr.move_to(top_expr)

        self.play(
            LaggedStart(
                ReplacementTransform(top_expr[re.compile("d_.")], short_expr["d_i"]),
                ReplacementTransform(top_expr["+"], short_expr[R"\sum"]),
                ReplacementTransform(top_expr[R"\cdots"], short_expr["i"][1]),
                lag_ratio=0.25
            )
        )
        self.wait()
        self.play(LaggedStart(
            Write(short_expr[R"\min\left("]),
            Write(short_expr[R"\right)"]),
            lag_ratio=0.5
        ))
        self.wait()

        # Show various alternate coverings
        d_labels.set_opacity(0)
        arrows.set_opacity(0)
        curr_strips = VGroup(strip1, *new_strips, *strips)
        og_strips = curr_strips

        for _ in range(4):
            self.play(FadeOut(curr_strips))
            base_hue = random.random()
            curr_strips = VGroup(
                self.get_strip(
                    circle,
                    *sorted(np.random.uniform(-1, 1, 2)),
                    TAU * np.random.uniform(0, TAU),
                    color=random_bright_color(hue_range=(base_hue, base_hue + 0.2)),
                    opacity=0.25
                ).set_stroke(width=1)
                for n in range(15)
            )
            self.play(ShowIncreasingSubsets(curr_strips))

        self.play(FadeOut(curr_strips))
        self.play(ShowIncreasingSubsets(og_strips))

        # Show trivial covering
        fat_strip = self.get_strip(circle, -1, 1, 0, RED_B)
        fat_strip.rect.set_height(6, stretch=True)
        fat_strip.pre_rect.move_to(fat_strip.rect, DOWN)

        top_brace = Brace(fat_strip.rect, UP)
        top_label = top_brace.get_text("2")

        self.play(
            FadeOut(og_strips),
            short_expr.animate.next_to(circle, RIGHT, buff=LARGE_BUFF),
            frame.animate.move_to(0.5 * UP)
        )
        self.play(Transform(fat_strip.pre_rect, fat_strip.rect))
        self.play(GrowFromCenter(top_brace), Write(top_label))
        self.wait()

        # Subdivide trivial covering
        subdivision = sorted([-1, 1, *np.random.uniform(-1, 1, 10)])
        strips = VGroup(
            self.get_strip(circle, r0, r1, theta=0, color=random_bright_color(hue_range=(0.3, 0.5)))
            for r0, r1 in zip(subdivision, subdivision[1:])
        )

        self.play(
            FadeOut(fat_strip.pre_rect),
            FadeIn(strips, lag_ratio=0.5, run_time=2)
        )
        self.wait()

        # Show suggestive fan covering
        fan_covering = VGroup(
            self.get_strip(circle, -0.4, 0.4, theta=theta)
            for theta in np.arange(0, TAU, TAU / 3)
        )
        fan_covering.add(*(
            self.get_strip(circle, 0.6, 0.9, theta=theta)
            for theta in np.arange(TAU / 12, TAU, TAU / 3)
        ))

        self.play(FadeOut(strips))
        for strip in fan_covering:
            self.animate_strip_in(strip)
        self.wait()

    def get_strip(self, circle, r0, r1, theta, color=None, opacity=0.5, include_arrow=False, label="", rect_length=10.0):
        diam = circle.get_width()
        width = (r1 - r0) * diam / 2
        if color is None:
            color = random_bright_color(luminance_range=(0.5, 0.7))

        rect = Rectangle(width, rect_length)
        rect.move_to(
            interpolate(circle.get_center(), circle.get_right(), r0),
            LEFT,
        )
        rect.set_fill(color, opacity)
        rect.set_stroke(color, 1)
        pre_rect = rect.copy().stretch(0, 1, about_edge=DOWN)
        pre_rect.set_stroke(width=0)
        VGroup(rect, pre_rect).rotate(theta, about_point=circle.get_center())

        strip = Intersection(rect, circle)
        strip.match_style(rect)
        strip.rect = rect
        strip.pre_rect = pre_rect

        if include_arrow:
            arrow = Tex(R"\longleftrightarrow")
            arrow.set_width(width, stretch=True)
            arrow.rotate(theta)
            arrow.move_to(rect)
            strip.add(arrow)
        if len(label) > 0:
            label = Tex(label, font_size=36)
            label.move_to(rect.get_center())
            vect = 0.25 * rotate_vector(UP, theta)
            vect *= np.sign(vect[1])
            label.shift(vect)
            strip.add(label)

        return strip

    def animate_strip_in(self, strip):
        self.play(Transform(strip.pre_rect, strip.rect))
        self.play(LaggedStart(
            FadeIn(strip),
            FadeOut(strip.pre_rect),
            lag_ratio=0.5,
            run_time=1,
        ))


class StruggleWithStrips(AskStripQuestion):
    def construct(self):
        # Add circle
        radius = 2.5
        circle = Circle(radius=radius)
        circle.set_stroke(YELLOW, 2)
        radial_line = Line(circle.get_center(), circle.get_right())
        radial_line.set_stroke(WHITE, 2)
        radius_label = Integer(1)
        radius_label.next_to(radial_line, UP, SMALL_BUFF)

        self.add(circle, radial_line, radius_label)

        # Show fan strategy
        angles = [*np.arange(0, TAU, TAU / 3), *np.arange(TAU / 12, TAU, TAU / 3)]
        widths = [*3 * [0.8], *3 * [0.25]]
        strips = VGroup(
            self.get_strip(circle, -0.4, 0.4, theta=theta, include_arrow=True)
            for theta in angles[:3]
        )
        strips.add(*(
            self.get_strip(circle, 0.7, 0.95, theta=theta, include_arrow=True)
            for theta in angles[3:]
        ))
        arrows = VGroup()
        for strip in strips:
            arrow = strip[0]
            strip.remove(arrow)
            arrows.add(arrow)

        self.play(LaggedStart(
            (TransformFromCopy(strip.pre_rect, strip.rect)
            for strip in strips),
            lag_ratio=0.1,
        ))
        rects = VGroup(strip.rect for strip in strips)
        self.play(
            LaggedStartMap(FadeOut, rects),
            LaggedStartMap(FadeIn, strips),
        )

        # Show the sum
        sum_expr = Tex("0.00 + 0.00 + 0.00 + 0.00 + 0.00 + 0.00 = 0.00")
        sum_expr.to_edge(UP)
        decimals = sum_expr.make_number_changeable("0.00", replace_all=True)
        width_terms = decimals[:6]
        sum_term = decimals[6]
        plusses = sum_expr["+"]
        equals = sum_expr["="][0]
        plusses.add_to_back(VectorizedPoint(sum_expr.get_left()))

        sum_term.set_fill(RED)

        last_arrow = VGroup()
        for i in range(len(strips)):
            width_term = width_terms[i]
            width_term.set_value(widths[i])
            width_term.save_state()

            arrow = arrows[i]
            width_term.next_to(
                arrow.get_center(),
                rotate_vector(UP, angles[i])
            )

            strips.target = strips.generate_target()
            strips.target.set_opacity(0.2)
            strips.target[i].set_fill(opacity=0.5)
            strips.target[i].set_stroke(opacity=1)

            self.play(
                MoveToTarget(strips),
                FadeIn(width_term),
                FadeIn(arrow),
                FadeOut(last_arrow),
            )
            self.play(
                Restore(width_term),
                FadeIn(plusses[i])
            )

            last_arrow = arrow

        sum_term.set_value(sum(wt.get_value() for wt in width_terms))
        self.play(
            FadeOut(last_arrow),
            strips.animate.set_fill(opacity=0.5).set_stroke(opacity=1),
            Write(equals),
            FadeIn(sum_term),
        )
        self.wait()

        # Turn into parallel strips
        np.random.seed(3)
        subdivision = sorted([-1, 1, *np.random.uniform(-1, 1, 5)])
        r_pairs = list(zip(subdivision, subdivision[1:]))
        new_widths = [r1 - r0 for r0, r1 in r_pairs]

        new_strips = VGroup(
            self.get_strip(circle, r0, r1, theta=0)
            for r0, r1 in r_pairs
        )
        new_strips.match_style(strips)
        new_rects = VGroup(s.rect for s in new_strips)

        self.play(
            FadeOut(strips),
            FadeIn(rects),
        )
        self.play(
            # Transform(strips, new_strips),
            ReplacementTransform(rects, new_rects),
            *(
                ChangeDecimalToValue(width_term, new_width)
                for width_term, new_width in zip(width_terms, new_widths)
            ),
            ChangeDecimalToValue(sum_term, 2.0),
            run_time=2
        )
        self.play(FadeOut(new_rects), FadeIn(new_strips))
        self.wait()

        # Show sum of the area
        width_sum = Tex(R"\sum_{\text{strip}} \textbf{Width}(\text{strip})")
        area_sum = Tex(R"\sum_{\text{strip}} \textbf{Area}(\text{strip})")
        area_sum_rhs = Tex(R"\ge \pi r^2 = \pi")
        width_sum.to_corner(UR)
        area_sum.to_corner(UL)
        area_sum_rhs.next_to(area_sum[-1], RIGHT, MED_SMALL_BUFF)

        width_brace = Brace(width_sum, DOWN)
        width_annotation = width_brace.get_text("We want to\ncontrol this")
        width_annotation.set_color(YELLOW)

        self.play(FadeTransformPieces(sum_expr, width_sum))
        self.play(GrowFromCenter(width_brace), Write(width_annotation))
        self.wait()
        self.play(Write(area_sum))
        self.wait()
        self.play(Write(area_sum_rhs))
        self.wait()

        # Add area and width label for strip
        strip = new_strips[1]
        area_label = TexText(R"Area = $0.00$")
        area_dec = area_label.make_number_changeable("0.00")
        area_dec.add_updater(lambda m: m.set_value(
            get_norm(strip.get_area_vector()) / radius**2
        ))
        area_label.add_updater(lambda m: m.next_to(strip, LEFT))
        area_label.match_color(strip)

        arrow = Tex(R"\leftrightarrow").stretch(2, 0)
        arrow.match_width(strip)
        arrow.always.move_to(strip)

        width_label = VGroup(
            Text("Width"),
            Tex("=").rotate(90 * DEGREES),
            DecimalNumber(1),
        )
        width_label.arrange(DOWN)
        width_label.set_width(strip.get_width() * 0.8)
        width_label[2].add_updater(lambda m: m.set_value(strip.get_width() / radius))
        width_label.always.next_to(arrow, UP)

        self.play(
            FadeOut(new_strips[:1]),
            FadeOut(new_strips[2:]),
            FadeOut(radial_line),
            FadeOut(radius_label),
        )
        self.play(Write(area_label))
        self.wait()
        self.play(
            GrowFromCenter(arrow),
            FadeIn(width_label),
        )
        self.wait()

        # Show varying strip
        r0 = subdivision[1]
        delta_r = subdivision[2] - subdivision[1]
        delta_r_tracker = ValueTracker(delta_r)
        r0_tracker = ValueTracker(r0)

        strip.add_updater(lambda m: m.match_points(self.get_strip(
            circle,
            r0_tracker.get_value(),
            r0_tracker.get_value() + delta_r_tracker.get_value(),
            theta=0
        )))
        for value in [-1, 0.6, r0]:
            self.play(r0_tracker.animate.set_value(value), run_time=4)

        self.play(
            delta_r_tracker.animate.set_value(0.3),
            arrow.animate.scale(0.3 / 0.44),
            run_time=3
        )

        strip.clear_updaters()
        self.play(
            FadeOut(area_label),
            FadeOut(width_label),
            FadeOut(arrow),
            FadeIn(new_strips[:1]),
            FadeIn(new_strips[2:]),
        )

        # Show the dream of proportionality
        width_label_group = VGroup(width_sum, width_brace, width_annotation)
        circle_group = VGroup(circle, new_strips)

        dream_sum = Tex(R"\sum_{\text{strip}} {k} \cdot \textbf{Width}(\text{strip})")
        dream_sum[R"{k}"].set_color(YELLOW)
        dream_sum.next_to(area_sum, DOWN, buff=2.0)
        dream_sum.shift_onto_screen()

        down_arrow = Arrow(area_sum, dream_sum, thickness=5)
        arrow_words = Text("If only...")
        arrow_words.next_to(down_arrow, RIGHT, SMALL_BUFF)

        self.play(
            circle_group.animate.shift(3 * RIGHT),
            width_label_group.animate.scale(0.5, about_edge=UR),
        )
        self.wait()
        self.play(
            GrowArrow(down_arrow),
            FadeIn(arrow_words, lag_ratio=0.1)
        )
        self.play(TransformMatchingStrings(area_sum.copy(), dream_sum))
        self.wait()

    def get_strip(self, *args, **kwargs):
        kwargs["rect_length"] = kwargs.get("rect_length", 6.0)
        return super().get_strip(*args, **kwargs)


class SphereStrips(InteractiveScene):
    def construct(self):
        # Axes
        frame = self.frame
        frame.set_height(3)
        axes = ThreeDAxes((-2, 2), (-2, 2), (-2, 2))
        axes.set_stroke(width=1)
        plane = NumberPlane((-2, 2), (-2, 2))
        plane.fade(0.5)
        self.add(axes)
        self.add(plane)

        # Circle
        circle = Circle()
        circle.set_stroke(YELLOW, 3)
        circle.set_fill(BLACK, 0.0)
        self.add(circle)

        # Sphere
        sphere = ParametricSurface(
            lambda u, v: [
                np.sin(u) * np.cos(v),
                np.sin(u) * np.sin(v),
                np.cos(u)
            ],
            u_range=(0, PI),
            v_range=(0, 2 * PI)
        )
        sphere.set_opacity(0.5)
        sphere.set_shading(0.5, 0.5, 0.5)
        sphere.always_sort_to_camera(self.camera)
        sphere.set_clip_plane(OUT, 1e-3)

        # Show pre_strip
        delta_x = 0.25
        x0 = 0.5
        strip = self.get_strip(x0, x0 + delta_x, 0)
        pre_strip = strip.copy()
        pre_strip.stretch(1e-3, 2)
        pre_strip.set_z_index(1)
        circle.set_clip_plane(UP, 10)  # Why?
        plane.set_clip_plane(UP, 10)  # Why?

        self.play(ShowCreation(pre_strip, run_time=2))
        self.wait()

        # Expand
        pre_sphere = sphere.copy()
        pre_sphere.stretch(0, 2)
        pre_sphere.shift(1e-2 * IN)
        pre_sphere.set_opacity(0)

        strip.save_state()
        strip.become(pre_strip)
        sphere.save_state()
        sphere.become(pre_sphere)

        self.remove(pre_strip)
        self.add(strip, sphere)

        self.play(
            frame.animate.reorient(-34, 59, 0),
            run_time=2
        )
        self.wait()
        self.add(pre_sphere, pre_strip)
        self.play(
            Restore(strip),
            Restore(sphere),
            run_time=3
        )
        self.play(
            frame.animate.reorient(40, 59, 0),
            run_time=7
        )
        self.wait()

        # Note the area
        brace = Brace(pre_strip, UP)
        brace.add(brace.get_tex(R"d", font_size=24, buff=0.05))
        brace.rotate(90 * DEGREES, RIGHT)
        brace.next_to(strip, OUT, buff=0)

        area_label = TexText(R"Area = $\pi d$")
        area_label.to_corner(UR)
        area_label.fix_in_frame()

        self.play(
            GrowFromCenter(brace, time_span=(1, 2)),
            frame.animate.reorient(-2, 94, 0, (0.31, 0.11, 0.63), 2.35),
            run_time=3,
        )
        self.wait()
        self.play(
            Write(area_label),
            Transform(
                brace[-1].copy(),
                brace[-1].copy().scale(0.5).shift(1.5 * RIGHT + 0.25 * OUT).set_opacity(0),
                remover=True
            )
        )
        self.play(FlashAround(area_label, run_time=2))
        self.wait()

        # Move strip around
        x0_tracker = ValueTracker(x0)
        strip.add_updater(lambda m: m.become(self.get_strip(
            x0_tracker.get_value(),
            x0_tracker.get_value() + delta_x,
            theta=0
        )))
        brace.add_updater(lambda m: m.next_to(strip, OUT, buff=0))

        self.play(
            x0_tracker.animate.set_value(-0.99),
            frame.animate.reorient(-24, 82, 0, (0.4, 0.08, 0.63), 2.67),
            run_time=5,
        )
        self.play(
            x0_tracker.animate.set_value(0.5),
            frame.animate.reorient(2, 76, 0, (0.4, 0.08, 0.63), 2.67),
            run_time=5
        )
        strip.clear_updaters()
        brace.clear_updaters()
        self.play(FadeOut(brace), FadeOut(area_label))

        # Reorient and add make full sphere
        self.play(
            frame.animate.reorient(29, 74, 0, ORIGIN, 3.00).set_anim_args(run_time=2),
            sphere.animate.set_clip_plane(OUT, 1),
            strip.animate.set_clip_plane(OUT, 1),
        )
        self.play(
            frame.animate.reorient(7, 67, 0),
            Rotate(strip, PI / 2, DOWN, about_point=ORIGIN),
            run_time=2
        )
        self.wait()

        # Add cylinder
        clyinder = ParametricSurface(
            lambda u, v: [np.cos(v), np.sin(v), u],
            u_range=[-1, 1],
            v_range=[0, TAU],
        )
        cylinder_mesh = SurfaceMesh(clyinder, resolution=(33, 51))
        cylinder_mesh.set_stroke(WHITE, 1, 0.25)
        cylinder_mesh.set_clip_plane(UP, 20)
        cylinder_mesh.match_height(sphere)

        self.play(self.frame.animate.reorient(26, 69, 0, (-0.0, -0.0, 0.0), 3.00), run_time=3)
        self.play(ShowCreation(cylinder_mesh, lag_ratio=0.01))
        self.wait()

        # Project the strip
        def clyinder_projection(points):
            radii = np.apply_along_axis(np.linalg.norm, 1, points[:, :2])
            return np.transpose([points[:, 0] / radii, points[:, 1] / radii, points[:, 2]])

        def get_proj_strip(strip):
            return strip.copy().apply_points_function(clyinder_projection).set_opacity(0.8)

        proj_strip = get_proj_strip(strip)
        proj_strip.save_state()
        proj_strip.become(strip)

        self.add(proj_strip, cylinder_mesh)
        self.play(
            frame.animate.reorient(-28, 62, 0).set_anim_args(run_time=4),
            Restore(proj_strip, run_time=2),
        )
        self.wait()

        # Vary the height of the strip
        strip.add_updater(lambda m: m.become(
            self.get_strip(
                x0_tracker.get_value(),
                x0_tracker.get_value() + delta_x,
                theta=0,
            ).rotate(PI / 2, DOWN, about_point=ORIGIN)
        ))
        proj_strip.add_updater(lambda m: m.match_z(strip))
        sphere.set_clip_plane(UP, 20)

        self.add(sphere, cylinder_mesh)
        frame.add_ambient_rotation()
        for value in [0.75, 0, 0.5]:
            self.play(x0_tracker.animate.set_value(value), run_time=6)

        frame.clear_updaters()
        strip.clear_updaters()
        proj_strip.clear_updaters()

        # Go back to the hemisphere state
        self.play(
            FadeOut(cylinder_mesh),
            FadeOut(proj_strip, 5 * OUT),
        )
        strip.set_clip_plane(OUT, 0)
        sphere.set_clip_plane(OUT, 1)
        self.play(
            frame.animate.reorient(23, 68, 0),
            sphere.animate.set_clip_plane(OUT, 0),
            Rotate(strip, PI / 2, axis=UP, about_point=ORIGIN),
            run_time=3
        )
        self.wait()

        # Cover with more strips
        strips = Group(
            self.get_strip(
                *sorted(np.random.random(2)),
                theta=random.uniform(0, TAU),
                color=random_bright_color(),
            ).shift(x * 1e-3 * OUT)
            for x in range(1, 20)
        )
        strips.set_opacity(0.5)

        self.play(
            ShowCreation(strips, lag_ratio=0.9),
            frame.animate.reorient(-17, 31, 0),
            run_time=10
        )
        self.play(
            frame.animate.reorient(-24, 64, 0),
            run_time=8,
        )
        self.wait()

    def get_strip(self, x0, x1, theta, color=BLUE):
        strip = ParametricSurface(
            lambda u, v: [
                np.cos(u),
                np.sin(u) * np.cos(v),
                np.sin(u) * np.sin(v),
            ],
            u_range=(math.acos(x1), math.acos(x0)),
            v_range=(0, TAU),
        )
        strip.rotate(theta, OUT, about_point=ORIGIN)
        strip.scale(1.001, about_point=ORIGIN)
        strip.set_color(color)
        strip.set_shading(0.5, 0.5, 0.5)
        strip.set_clip_plane(OUT, 1e-3)
        return strip


class MongesTheorem(InteractiveScene):
    def construct(self):
        # Add circles
        circ1, circ2, circ3 = circles = self.get_initial_circles()

        self.play(LaggedStartMap(ShowCreation, circles, lag_ratio=0.5, run_time=2))
        self.wait()

        # Add tangents
        tangent_pairs = always_redraw(lambda: self.get_all_external_tangents(circles))
        intersection_dots = always_redraw(lambda: self.get_all_intersection_dots(tangent_pairs))

        dependents = Group(tangent_pairs, intersection_dots)
        dependents.suspend_updating()

        for tangents, dot, circle_pair in zip(tangent_pairs, intersection_dots, it.combinations(circles, 2)):
            c1, c2 = (c.copy() for c in circle_pair)
            self.play(*map(GrowFromCenter, tangents), run_time=1.5)
            self.play(
                LaggedStart(
                    c1.animate.scale(0, about_point=dot.get_center()),
                    c2.animate.scale(0, about_point=dot.get_center()),
                    FadeIn(dot),
                    lag_ratio=0.2,
                )
            )
            self.remove(c1, c2)
            self.wait()

        # Manipuate the circles
        dependents.resume_updating()
        circles.save_state()
        self.add(*dependents)

        self.add(*circles)
        self.wait(note="Play with circle positions. Be careful!")

        # Show the line between them
        monge_line = Line()
        monge_line.f_always.put_start_and_end_on(
            intersection_dots[0].get_center,
            intersection_dots[2].get_center,
        )
        monge_line.always.set_length(100)
        monge_line.set_stroke(WHITE, 3)
        monge_line.suspend_updating()

        self.play(GrowFromCenter(monge_line))
        self.wait()

        # Manipulate again
        monge_line.resume_updating()
        self.add(*circles)
        self.wait(30, note="Play with circle positions. Be careful!")
        self.play(Restore(circles), self.frame.animate.to_default_state(), run_time=3)

        dependents.suspend_updating()
        self.play(FadeOut(monge_line))
        self.wait()

        # Setup spheres and tangent groups
        plane = NumberPlane((-8, 8), (-8, 8))
        plane.background_lines.set_stroke(GREY, 1)
        plane.faded_lines.set_stroke(GREY, 1, 0.25)
        plane.axes.set_stroke(GREY, 1)

        spheres = self.get_spheres(circles)
        tangent_groups = always_redraw(lambda: self.get_tangent_groups(circles))

        tangent_groups.suspend_updating()

        # Show spheres
        frame = self.frame
        self.wait()
        self.play(
            frame.animate.reorient(-11, 69, 0),
            FadeIn(plane),
            FadeIn(spheres, lag_ratio=0.25),
            run_time=4
        )
        self.wait()

        # Reposition
        self.play(self.frame.animate.reorient(-41, 72, 0), run_time=5)

        # Show various external tangents
        self.play(
            frame.animate.reorient(-67, 76, 0),
            LaggedStartMap(GrowFromCenter, tangent_groups[2], lag_ratio=0.1),
            spheres[0].animate.set_opacity(0.05),
            run_time=3
        )
        self.wait(10, note="Emphasize how it's formed")

        self.play(
            frame.animate.reorient(-105, 46, 0),
            LaggedStartMap(GrowFromCenter, tangent_groups[1], lag_ratio=0.1),
            spheres[0].animate.set_opacity(0.5),
            spheres[1].animate.set_opacity(0.05),
            run_time=3
        )
        self.wait()
        self.play(
            frame.animate.reorient(-175, 51, 0, (1.2, 0.92, -0.26)),
            LaggedStartMap(GrowFromCenter, tangent_groups[0], lag_ratio=0.1),
            spheres[1].animate.set_opacity(0.5),
            spheres[2].animate.set_opacity(0.05),
            run_time=3
        )
        self.wait()
        self.play(
            frame.animate.reorient(-70, 59, 0, (0.22, 0.32, -1.5), 9.17),
            spheres[2].animate.set_opacity(0.5),
            run_time=6,
        )
        self.wait()

        # Show mutually tangent plane (Fudged, but it works)
        xy_plane = Square3D(resolution=(100, 100)).rotate(PI)
        xy_plane.set_color(BLUE_E, 0.35)
        xy_plane.replace(plane)

        inter_points = [dot.get_center() for dot in intersection_dots]
        blue_tip = self.get_cone_tips(circles[2:], angle=84 * DEGREES)[0]
        tangent_plane = self.get_plane_through_points([inter_points[2], inter_points[0], blue_tip])

        plane_lines = VGroup(
            tangent_groups[2][19].copy(),
            tangent_groups[0][31].copy(),
            tangent_groups[1][25].copy(),
        )
        plane_lines.set_stroke(width=4, opacity=1)

        self.play(
            frame.animate.reorient(-50, 74, 0, (0.22, 0.32, -1.5), 9.17),
            ShowCreation(tangent_plane, time_span=(0, 2)),
            run_time=6,
        )
        self.wait()
        self.play(
            frame.animate.reorient(-77, 63, 0, (0.22, 0.32, -1.5), 9.17),
            FadeOut(tangent_groups),
            run_time=4
        )
        for line in plane_lines:
            self.play(ShowCreation(line, run_time=2))
            self.wait()

        self.add(xy_plane, tangent_plane, plane_lines)
        self.play(ShowCreation(xy_plane, time_span=(0, 2)))
        self.wait()
        self.play(ShowCreation(monge_line, suspend_mobject_updating=True))
        self.wait()

        # Move circles to problem position
        self.play(
            FadeOut(xy_plane),
            FadeOut(tangent_plane),
            FadeOut(plane_lines),
            self.frame.animate.to_default_state(),
            run_time=2
        )

        dependents.resume_updating()
        self.add(dependents)
        self.play(
            circles[1].animate.move_to(2 * LEFT),
            circles[0].animate.move_to(0.2 * UP),
            circles[2].animate.move_to(3 * RIGHT),
            run_time=3
        )
        dependents.suspend_updating()
        self.wait()

        # Show the outside plane
        angle = abs(tangent_pairs[2][0].get_angle())
        partial_tangent_plane = xy_plane.copy()
        pivot_point = intersection_dots[2].get_center()
        partial_tangent_plane.rotate(angle, axis=DOWN, about_point=pivot_point)
        partial_tangent_plane.set_height(5, stretch=True)
        partial_tangent_plane.set_color(GREY_C, 0.5)
        partial_tangent_plane.set_shading(0.25, 0.25, 0.25)

        self.add(partial_tangent_plane)
        self.play(ShowCreation(partial_tangent_plane))
        self.wait()
        self.play(self.frame.animate.reorient(27, 75, 0))
        self.play(
            Rotating(partial_tangent_plane, PI / 2, axis=RIGHT, about_point=pivot_point),
            run_time=8,
            rate_func=there_and_back,
        )
        self.wait()
        self.play(
            FadeOut(partial_tangent_plane),
            self.frame.animate.to_default_state(),
            run_time=3
        )
        dependents.resume_updating()
        self.add(dependents)
        self.play(circles[0].animate.move_to(2 * UP), run_time=3)
        dependents.suspend_updating()

        # Show the cones
        cones = self.get_cones(circles)

        def upadte_cone_positions(cones):
            for cone, circle in zip(cones, circles):
                cone.match_width(circle)
                cone.move_to(circle, IN)

        self.play(
            self.frame.animate.reorient(-74, 72, 0, (-1.2, 0.14, -0.2), 8.00),
            run_time=3
        )
        spheres.clear_updaters()
        self.play(ReplacementTransform(spheres, cones, lag_ratio=0.5, run_time=2))
        self.wait()

        # Show the center of similarity
        def get_tip_lines():
            result = VGroup()
            for i, j, k in [(2, 2, 2), (1, 2, 1), (0, 1, 0)]:
                line = Line(intersection_dots[i].get_center(), cones[j].get_zenith())
                line.match_color(tangent_pairs[k][0])
                line.scale(2, about_point=line.get_start())
                result.add(line)
            return result
        tip_lines = always_redraw(get_tip_lines)
        tip_lines.suspend_updating()

        self.play(ShowCreation(tip_lines[0]))
        self.play(self.frame.animate.reorient(-1, 83, 0, (-1.2, 0.14, -0.2)), run_time=3)
        self.wait()

        cone_ghost = cones[2].copy().set_opacity(0.5)
        cone_ghost.deactivate_depth_test()
        self.add(cones, cone_ghost)
        self.play(FadeIn(cone_ghost))
        for x in range(2):
            self.play(
                cone_ghost.animate.scale(1e-2, about_point=intersection_dots[2].get_center()),
                run_time=8,
                rate_func=there_and_back
            )
            self.wait()
            self.play(self.frame.animate.reorient(0, 7, 0, (-1.92, 0.22, 0.0)), run_time=3)
        self.play(
            FadeOut(cone_ghost),
            self.frame.animate.reorient(-129, 75, 0, (-1.92, 0.22, 0.0)),
            run_time=4
        )
        self.play(ShowCreation(tip_lines[1:], lag_ratio=0.5, run_time=2))
        self.wait()

        # Add plane
        plane = always_redraw(lambda: self.get_plane_through_points([
            intersection_dots[2].get_center(),
            intersection_dots[0].get_center(),
            cones[2].get_zenith()
        ]))
        plane.suspend_updating()

        self.play(
            ShowCreation(plane),
            self.frame.animate.reorient(-74, 66, 0, (-1.92, 0.22, 0.0)),
            run_time=4
        )

        # Move the circles all about
        dependents.add(tip_lines, plane)
        dependents.resume_updating()
        cones.add_updater(upadte_cone_positions)
        self.add(cones, dependents)

        self.play(circles[0].animate.move_to(0.2 * UP), run_time=3)
        dependents.suspend_updating()
        self.play(self.frame.animate.reorient(-173, 69, 0, (-1.36, 0.7, 1.01), 7.14), run_time=10)
        self.wait(note="Reorient")
        dependents.resume_updating()
        self.play(
            circles[0].animate.move_to(2 * UP),
            self.frame.animate.reorient(-122, 54, 0, (-1.54, 0.75, 0.38), 8.65),
            run_time=4
        )
        self.manipulate_circle_positions(circles)
        dependents.suspend_updating()

    def get_initial_circles(self):
        centers = [[-3, 3, 0], [-6, -1.5, 0], [3, -1.5, 0]]
        colors = [RED, GREEN, BLUE]
        radii = [1, 2, 4]
        circles = VGroup(
            Circle(radius=radius).move_to(center).set_color(color)
            for radius, center, color in zip(radii, centers, colors)
        )
        circles.scale(0.5)
        circles.to_edge(RIGHT, buff=LARGE_BUFF)
        return circles

    def get_plane_through_points(self, points, color=GREY_B, opacity=0.5):
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        perp = normalize(cross(v2, v1))
        vert_angle = math.acos(perp[2])

        plane = Square3D(resolution=(100, 100))
        plane.set_width(get_norm(v1))
        plane.move_to(ORIGIN, DL)
        plane.rotate(angle_of_vector(v1), about_point=ORIGIN)
        plane.rotate(PI - vert_angle, axis=v1, about_point=ORIGIN)
        plane.shift(points[0])
        plane.scale(2, about_point=points[0])

        plane.set_color(color, opacity=opacity)

        return plane

    def get_cones(self, circles, angle=90 * DEGREES):
        cones = Group()
        for circle in circles:
            radius = circle.get_width() / 2
            cone = Cone(radius=radius, height=radius / math.tan(angle / 2))
            cone.move_to(circle, IN)
            cone.set_color(circle.get_color())
            cone.set_opacity(0.5)
            cone.always_sort_to_camera(self.camera)
            cones.add(cone)
        return cones

    def get_cone_tips(self, circles, angle=90 * DEGREES):
        points = []
        for circle in circles:
            radius = circle.get_width() / 2
            height = radius / math.tan(angle / 2)
            point = circle.get_center() + height * OUT
            points.append(point)
        return points

    def get_spheres(self, circles, opacity=0.5):
        spheres = Group()
        for circle in circles:
            sphere = Sphere(radius=circle.get_radius())
            sphere.set_color(circle.get_color(), opacity)
            sphere.circle = circle
            sphere.always_sort_to_camera(self.camera)
            sphere.always.match_width(circle)
            sphere.always.move_to(circle)
            spheres.add(sphere)
        return spheres

    def get_tangent_groups(self, circles, n_lines=24):
        tangent_groups = VGroup()
        for circ1, circ2 in it.combinations(circles, 2):
            tangent_pair = self.get_external_tangents(circ1, circ2)
            point = self.get_intersection(*tangent_pair)
            axis = circ2.get_center() - circ1.get_center()
            group = VGroup()
            for angle in np.arange(0, PI, PI / n_lines):
                group.add(*tangent_pair.copy().rotate(angle, axis=axis, about_point=point))
            for line in group:
                line.shift(point - line.get_start())
            group.set_stroke(width=1, opacity=0.5)
            tangent_groups.add(group)
        return tangent_groups

    def get_all_intersection_dots(self, line_pairs):
        return Group(
            GlowDot(self.get_intersection(*pair))
            for pair in line_pairs
        )

    def get_all_external_tangents(self, circles, **kwargs):
        return VGroup(
            self.get_external_tangents(circ1, circ2)
            for circ1, circ2 in it.combinations(circles, 2)
        )

    def get_external_tangents(self, circle1, circle2, length=100, color=None):
        c1 = circle1.get_center()
        c2 = circle2.get_center()
        r1 = circle1.get_radius()
        r2 = circle2.get_radius()

        if get_norm(c1 - c2) <= max(r1, r2):
            return VectorizedPoint().replicate(2)

        # Distance to intersection of external tangents
        L1 = get_norm(c1 - c2) / (1 - r2 / r1)
        intersection = c1 + L1 * normalize(c2 - c1)
        theta = math.asin(r1 / L1)

        line1 = Line(c1, c2)
        line1.insert_n_curves(20)
        line1.rotate(theta, about_point=intersection)
        line1.set_length(length)
        line2 = line1.copy().rotate(PI, axis=(c2 - c1), about_point=intersection)

        result = VGroup(line1, line2)
        if color is None:
            color = interpolate_color(circle1.get_color(), circle2.get_color(), 0.5)
        result.set_stroke(color, width=2)
        return result

    def get_intersection(self, line1, line2):
        try:
            return line_intersection(
                line1.get_start_and_end(),
                line2.get_start_and_end(),
            )
        except Exception:
            return midpoint(line1.get_end(), line2.get_end())

    def manipulate_circle_positions(self, circles):
        circ1, circ2, circ3 = circles
        # Example
        self.play(circ2.animate.shift(LEFT), run_time=2)
        self.play(circ2.animate.scale(0.75), run_time=2)
        self.play(circ1.animate.scale(0.5).shift(0.2 * DOWN), run_time=2)
        self.play(circ3.animate.scale(0.7).shift(0.2 * DOWN), run_time=4)
        self.wait()
        self.play(Restore(circles), run_time=3)
        self.wait()


class GeneralCentersOfSimilarity(MongesTheorem):
    def construct(self):
        # Show centers of similarity
        circles = self.get_initial_circles()
        cos_dots = always_redraw(lambda: self.get_center_of_similarity_dots(circles))
        similarity_lines = always_redraw(lambda: self.get_all_similarity_lines(circles))
        theorem_line = Line().set_stroke(WHITE, 2).insert_n_curves(20)
        theorem_line.add_updater(lambda m: m.put_start_and_end_on(
            cos_dots[0].get_center(), cos_dots[2].get_center()
        ).scale(100))
        labels = VGroup()

        self.add(circles)
        for lines, dot, pair in zip(similarity_lines, cos_dots, it.combinations(circles, 2)):
            bigger = pair[0] if pair[0].get_width() > pair[1].get_width() else pair[1]
            ghost = bigger.copy()
            label = Text("Center of similarity", font_size=36)
            label.next_to(dot, DL, SMALL_BUFF)
            label.shift_onto_screen()
            self.play(
                ghost.animate.scale(0, about_point=dot.get_center()),
                ShowCreation(lines, lag_ratio=0),
                FadeIn(dot),
                FadeIn(label)
            )
            self.remove(ghost)
            self.wait()
            labels.add(label)

        theorem_line.update()
        theorem_line.suspend_updating()
        self.play(
            FadeOut(labels),
            GrowFromCenter(theorem_line)
        )
        theorem_line.resume_updating()
        self.add(cos_dots)
        self.add(similarity_lines)
        self.add(theorem_line)
        self.add(*circles)

        # Play around
        self.wait(20, note="Play!")

        # Change shapes
        pis = VGroup(
            Tex(R"\pi")[0].match_style(circle).replace(circle)
            for circle in circles
        )

        self.play(Transform(circles, pis, lag_ratio=0.5, run_time=3))
        self.wait()
        self.add(*circles)

        # Play some more
        self.wait(15, note="Play!")

        # Show the cones
        frame = self.frame
        cones = VGroup(self.get_cone(shape) for shape in circles)
        cones.set_stroke(opacity=0.75)
        self.play(
            LaggedStart(
                (FadeIn(cone, lag_ratio=0.05)
                for cone in cones),
                lag_ratio=0.5,
            ),
            frame.animate.reorient(16, 70, 0),
            run_time=5
        )
        self.wait(5)

    def get_center_of_similarity_dots(self, shapes):
        return Group(
            GlowDot(self.get_center_of_similarity(*pair))
            for pair in it.combinations(shapes, 2)
        )

    def get_center_of_similarity(self, shape1, shape2):
        w1 = shape1.get_width()
        w2 = shape2.get_width()
        c1 = shape1.get_center()
        c2 = shape2.get_center()

        vect = c2 - c1
        dist = get_norm(vect)
        # Desired ratio: (x - dist) / x = w2 / w1
        # -------------> x - dist = x (w2 / w1)
        # -------------> x (1 - w2 / w1) = dist
        # -------------> result = c1 + x * (vect / dist)
        return c1 + vect / (1.0 - w2 / w1)

    def get_all_similarity_lines(self, shapes, **kwargs):
        return VGroup(
            self.get_similarity_lines(*pair, **kwargs)
            for pair in it.combinations(shapes, 2)
        )

    def get_similarity_lines(self, shape1, shape2, n_lines=25):
        point = self.get_center_of_similarity(shape1, shape2)
        big = shape1 if shape1.get_width() > shape2.get_width() else shape2
        color = interpolate_color(shape1.get_color(), shape2.get_color(), 0.5)

        if big.get_num_points() == 0:
            return VGroup()

        result = VGroup(
            Line(big.pfp(alpha), point)
            for alpha in np.linspace(0, 1, n_lines)
        )
        result.set_stroke(color, 1, 0.5)
        return result

    def get_cone(self, shape):
        top_z = 0.5 * shape.get_width()
        return VGroup(
            shape.copy().scale(a).set_z(z).set_stroke(width=1)
            for a, z in zip(np.linspace(1, 0), np.linspace(0, top_z))
        )


class SimilarDiagrams(MongesTheorem):
    def construct(self):
        # Test
        circle1, circle2 = circles = VGroup(
            Circle(radius=1).move_to(LEFT).set_color(GREEN),
            Circle(radius=2).move_to(4 * RIGHT).set_color(BLUE),
        )
        lines = self.get_external_tangents(*circles)
        int_point = self.get_intersection(*lines)

        angle = lines[0].get_angle()
        t_point1 = circle1.get_center() + rotate_vector(UP, -angle)
        t_point2 = t_point1.copy()
        t_point2[1] *= -1
        t_points = [t_point1, t_point2]
        radii = VGroup(
            Line(circle1.get_center(), t_point)
            for t_point in t_points
        )
        elbows = VGroup(
            Elbow(width=0.1).rotate(PI - angle, about_point=ORIGIN).shift(t_point1),
            Elbow(width=0.1).rotate(-1.5 * PI + angle, about_point=ORIGIN).shift(t_point2),
        )
        elbows.set_stroke(width=2)
        tangents = VGroup(Line(t_point, int_point) for t_point in t_points)
        tangents.set_color(TEAL)

        self.add(circle1, radii, tangents, elbows)
        self.wait()

        self.add(radii.copy(), elbows.copy())
        self.play(
            TransformFromCopy(circle1, circle2),
            VGroup(tangents, radii, elbows).animate.scale(2, about_point=int_point),
            run_time=3
        )
        self.wait()


class AskAboutVolumeOfParallelpiped(InteractiveScene):
    def construct(self):
        # Axes and plane
        frame = self.frame
        axes = ThreeDAxes((-8, 8), (-4, 4), (-4, 4))
        axes.set_stroke(WHITE, 2)
        plane = NumberPlane()
        frame.reorient(-43, 73, 0, (1.18, 0.21, 1.19), 5.96)
        self.add(plane, axes)

        # Tetrahedron
        verts = [
            (-2, 1, 1),
            (1, 0, 3),
            (3, 0, 0),
            (2, -2, 0),
        ]
        tetrahedron = VGroup(
            Polygon(*subset)
            for subset in it.combinations(verts, 3)
        )
        tetrahedron.set_stroke(WHITE, 1)
        tetrahedron.set_fill(TEAL_E, 0.5)
        tetrahedron.set_shading(0.5, 0.5, 0)
        dots = DotCloud(verts, radius=0.05)
        dots.make_3d()
        dots.set_color(WHITE)

        self.add(tetrahedron)
        self.add(dots)

        # Add vertex labels
        labels = VGroup(
            Tex(f"(x_{n}, y_{n}, z_{n})", font_size=36)
            for n in range(1, 5)
        )
        vects = [OUT + LEFT, OUT, OUT + RIGHT, OUT + RIGHT]
        for label, point, vect in zip(labels, dots.get_points(), vects):
            label.rotate(89 * DEGREES, RIGHT)
            label.next_to(point, vect, buff=SMALL_BUFF)

        frame.reorient(-33, 84, 0, (0.34, 0.8, 1.42), 7.12)
        frame.add_updater(lambda m: m.set_theta(-math.cos(7 * self.time * DEGREES) * 35 * DEGREES))

        self.play(LaggedStartMap(FadeIn, labels, shift=0.5 * OUT, lag_ratio=0.5, run_time=3))
        self.wait(30)


class TriangleAreaFormula(InteractiveScene):
    def construct(self):
        # Set up triangle
        plane = NumberPlane(faded_line_ratio=1)
        plane.add_coordinate_labels(font_size=16)
        plane.background_lines.set_stroke(opacity=0.75)
        plane.faded_lines.set_stroke(opacity=0.25)
        verts = [
            (1, 1, 0),
            (2, -1, 0),
            (3, 2, 0),
        ]
        triangle = Polygon(*verts)
        triangle.set_stroke(YELLOW, 3)
        dots = Group(TrueDot(vert, radius=0.1) for vert in verts)
        dots.set_color(WHITE)

        self.add(plane)
        self.add(triangle)
        self.add(dots)

        # Add labels
        labels = VGroup(
            Tex(Rf"(x_{n}, y_{n})", font_size=36)
            for n in [1, 2, 3]
        )
        labels.set_backstroke(BLACK, 3)
        for label, dot, vect in zip(labels, dots, [UP, DOWN, UP]):
            label.next_to(dot, vect, SMALL_BUFF)
        labels[0].shift(0.3 * LEFT)

        self.frame.reorient(0, 0, 0, (2.02, 0.72, 0.0), 4.49),
        self.play(
            LaggedStartMap(FadeIn, labels, shift=0.25 * UP, lag_ratio=0.5, run_time=2),
            self.frame.animate.to_default_state().set_anim_args(run_time=6),
        )
        self.wait()
        self.play(triangle.animate.set_fill(YELLOW, 0.5))
        self.wait()

        # Set up 3D labels
        labels_3d = VGroup(
            Tex(Rf"(x_{n}, y_{n}, 1)", font_size=36)
            for n in [1, 2, 3]
        )
        for label, dot, vect in zip(labels_3d, dots, [LEFT, UR, RIGHT]):
            label.rotate(89 * DEGREES, RIGHT)
            label.next_to(dot, vect + OUT, SMALL_BUFF)
            label.shift(OUT)

        # Move up to 3d
        frame = self.frame
        z_axis = NumberLine((-4, 4))
        z_axis.rotate(PI / 2, DOWN)
        z_axis.set_flat_stroke(False)
        ghost_plane = plane.copy()
        ghost_plane.fade(0.5)
        ghost_plane.shift(OUT)

        self.play(
            frame.animate.reorient(-13, 75, 0, (-0.14, 0.7, 1.48), 9.34).set_anim_args(run_time=2),
            FadeIn(z_axis),
        )
        self.play(
            frame.animate.reorient(15, 81, 0, (-0.56, 0.95, 1.43), 9.34).set_anim_args(run_time=8),
            TransformFromCopy(plane, ghost_plane),
            triangle.animate.shift(OUT),
            Transform(labels, labels_3d),
            *(
                dot.animate.shift(OUT).make_3d()
                for dot in dots
            ),
        )
        self.wait()

        # Show parallelpiped
        cube = VCube(side_length=1)
        cube.set_stroke(WHITE, 2)
        cube.set_fill(WHITE, 0.1)
        cube.deactivate_depth_test()
        cube.move_to(ORIGIN, [-1, -1, -1])
        cube.apply_matrix(np.transpose([
            dot.get_center()
            for dot in dots
        ]))

        self.play(
            frame.animate.reorient(-5, 63, 0, (-0.04, 0.69, 0.39), 7.73).set_anim_args(run_time=5),
            Write(cube),
        )
        self.wait()

        # Show tetrehedron
        tetrahedron = VGroup(
            triangle.copy(),
            Polygon(ORIGIN, dots[0].get_center(), dots[1].get_center()),
            Polygon(ORIGIN, dots[0].get_center(), dots[2].get_center()),
            Polygon(ORIGIN, dots[1].get_center(), dots[2].get_center()),
        )
        tetrahedron.set_stroke(width=0)
        tetrahedron.set_fill(YELLOW, 0.5)

        self.play(
            frame.animate.reorient(-4, 77, 0, (0.28, 0.78, 0.41), 7.73).set_anim_args(run_time=4),
            Write(tetrahedron)
        )
        self.wait()
        self.play(
            frame.animate.reorient(-5, 64, 0, (0.22, 0.72, -0.04), 7.14),
            run_time=8
        )
        self.wait()


class LogicForArea(InteractiveScene):
    def construct(self):
        # Test
        det_tex = Tex(R"""
            = \frac{1}{2}\det\left[\begin{array}{ccc}
            x_1 & x_2 & x_3 \\
            y_1 & y_2 & y_3 \\
            1 & 1 & 1
            \end{array}\right]
        """)
        equations = VGroup(
            TexText(R"Volume(Tetra.) = $\frac{1}{3}$ Area(Tri.) $\times 1$"),
            TexText(R"Volume(Tetra.) = $\frac{1}{6}$ Volume(Para.)"),
            Tex(R"\Downarrow"),
            TexText(R"Area(Tri.) = $\frac{1}{2}$ Volume(Para.)"),
        )
        for eq in [det_tex, *equations]:
            eq["Tetra."].set_color(YELLOW)
            eq["Tri."].set_color(YELLOW)
            eq["Para."].set_color(YELLOW)

        equations.arrange(DOWN, buff=LARGE_BUFF)
        equations.to_corner(UL)
        equations[2].scale(2)
        det_tex.next_to(equations[-1]["="], DOWN, LARGE_BUFF, aligned_edge=LEFT)

        self.frame.set_height(10)
        self.add(det_tex)
        self.add(equations)


class FourDDet(InteractiveScene):
    def construct(self):
        det_tex = Tex(R"""
            \frac{1}{6}\det\left[\begin{array}{cccc}
            x_1 & x_2 & x_3 & x_4 \\
            y_1 & y_2 & y_3 & y_4 \\
            z_1 & z_2 & z_3 & z_4 \\
            1 & 1 & 1 & 1
            \end{array}\right]
        """)
        group = VGroup(
            Text("Volume", font_size=72),
            Tex("=", font_size=72).rotate(90 * DEGREES),
            det_tex
        )
        group.arrange(DOWN, buff=LARGE_BUFF)
        self.add(group)


class RandomVectorStatistics(InteractiveScene):
    def construct(self):
        # Show 2d distribution
        chart = self.get_random_angle_data_histogram(2, step_size=2)

        label = VGroup(Integer(2, edge_to_fix=UR), Text("D"))
        label.arrange(RIGHT, buff=0.05)
        label.next_to(chart.get_corner(UL), DR)
        label.shift(RIGHT)

        self.add(label)
        self.add(chart)

        # Many random vectors
        center = chart.get_center() + 0.5 * UP
        for _ in range(0):
            vects = VGroup(
                Vector(
                    1.5 * rotate_vector(RIGHT, random.uniform(0, TAU)),
                    thickness=4
                ).set_fill(random_bright_color(), border_width=1)
                for _ in range(2)
            )
            vects.shift(center)
            angle = (vects[0].get_angle() - vects[1].get_angle()) % TAU
            if angle > PI:
                angle = TAU - angle

            bar = chart.bars[int(angle / DEGREES) // 2].copy()
            bar.set_color(YELLOW)

            self.add(vects, bar)
            self.wait(0.5)
            self.remove(vects, bar)

        # Animate an increase in charts
        dim_tracker = ValueTracker(2)

        def get_dim():
            return int(dim_tracker.get_value())

        label.add_updater(lambda m: m[0].set_value(get_dim()))

        self.play(
            dim_tracker.animate.set_value(1000).set_anim_args(rate_func=rush_into),
            UpdateFromFunc(chart, lambda m: m.become(
                self.get_random_angle_data_histogram(
                    get_dim(),
                    n_vects=50000000 // get_dim(),
                    # n_vects=500000 // get_dim(),
                    step_size=1 if get_dim() < 100 else 0.5
                )
            )),
            run_time=20
        )

    def get_random_angle_data_histogram(self, dim, n_vects=1000000, step_size=1):
        vects1, vects2 = all_vects = [np.random.normal(0, 1, (n_vects, dim)) for _ in range(2)]

        for vects in all_vects:
            norms = np.linalg.norm(vects, axis=1)
            vects /= norms[:, np.newaxis]

        angles = np.arccos((vects1 * vects2).sum(1)) / DEGREES

        return self.get_histogram(angles, step_size=step_size)

    def get_histogram(self, data, min_val=0, max_val=180, step_size=5, bar_color=BLUE_D):
        bins = np.arange(min_val, max_val + 1, step_size)
        bucket_counts, _ = np.histogram(data, bins=bins, range=(min_val, max_val))

        bin_width = step_size / (max_val - min_val)
        densities = (bucket_counts / bucket_counts.sum()) / (bin_width)

        y_max = 16.0
        if densities.max() > y_max:
            densities *= (y_max / densities.max())**0.5

        axes = Axes(
            x_range=(min_val, max_val, 5),
            y_range=(0, y_max, 2),  # TODO
            width=6, height=4
        )
        axes.x_axis.add_numbers(np.arange(min_val, max_val + 1, 45), font_size=24, unit_tex=R"^\circ")
        x_unit = axes.x_axis.get_unit_size()
        y_unit = axes.y_axis.get_unit_size()

        bar_width = x_unit * step_size

        bars = VGroup(
            Rectangle(width=bar_width, height=density * y_unit).move_to(axes.c2p(x, 0), DL)
            for x, density in zip(bins, densities)
        )
        bars.set_fill(bar_color, 1)
        bars.set_stroke(WHITE, 0.5, 0.5)

        chart = VGroup(axes, bars)
        chart.axes = axes
        chart.bars = bars
        return chart


class ProbabilityQuestion(InteractiveScene):
    N = 6

    def construct(self):
        # Number lines and trackers
        trackers = Group(ValueTracker(np.random.uniform(-1, 1)) for x in range(self.N))
        lines = VGroup(
            self.get_uniform_random_indicator(tracker, n)
            for n, tracker in enumerate(trackers, start=1)
        )
        lines.arrange(DOWN, buff=0.35)
        lines.to_corner(DL)

        self.add(lines)

        # Add x_i distribution label
        dist_label = TexText("$x_i$ uniform in [-1, 1]")
        dist_label.match_x(lines).to_edge(UP)

        self.add(dist_label)

        # Add question label
        # lhs = Tex(R"P\left(\sum_{i=0}^7 x_i^2 \le 1 \right) = ")
        lhs = Tex(R"P\left(x_1^2 + x_2^2 + x_3^2 + x_4^2 + x_5^2 + x_6^2 \le 1 \right)")
        rhs = Tex(R"\frac{\pi^3}{6} \cdot \frac{1}{2^6}")
        eq = VGroup(lhs, Tex("=").rotate(PI / 2), rhs)
        eq.arrange(DOWN)
        eq.center().to_edge(RIGHT)

        self.add(eq)

        # Add brace
        def get_sum():
            return sum(t.get_value()**2 for t in trackers)

        brace = Brace(lhs[R"x_1^2 + x_2^2 + x_3^2 + x_4^2 + x_5^2 + x_6^2"], UP)
        brace.set_color(BLUE_D)
        sum_value = DecimalNumber()
        sum_value.set_color(BLUE_D)
        sum_value.next_to(brace, UP)
        sum_value.f_always.set_value(get_sum)

        symbols = VGroup(Checkmark().set_color(GREEN), Exmark().set_color(RED))
        symbols.set_height(0.5)
        symbols.match_x(lhs["1"][-1]).align_to(sum_value, UP)

        def update_symbols(syms):
            if get_sum() < 1:
                syms[0].set_opacity(1)
                syms[1].set_opacity(0)
            else:
                syms[1].set_opacity(1)
                syms[0].set_opacity(0)

        symbols.add_updater(update_symbols)

        self.add(brace, sum_value, symbols)

        # Animate in 6D label
        rect = SurroundingRectangle(rhs[R"\frac{\pi^3}{6}"])
        rect.set_stroke(TEAL, 2)
        label = Text("Volume of a \n 6D unit ball")
        label.next_to(rect, DOWN)
        label.set_color(TEAL)

        self.play(ShowCreation(rect))
        self.play(Write(label))
        self.wait()

        # Go over many random values
        time_per_state = 0.2
        total_time = 25
        for _ in range(int(total_time / time_per_state)):
            for tracker in trackers:
                tracker.set_value(np.random.uniform(-1, 1))
            self.wait(time_per_state)


    def get_uniform_random_indicator(self, value_tracker, n):
        line = NumberLine((-1, 1, 0.1), width=4, big_tick_spacing=1, tick_size=0.05)
        line.set_stroke(width=2)
        line.add_numbers(
            np.arange(-1, 1.5, 0.5),
            font_size=12,
            num_decimal_places=1,
            buff=0.15
        )

        tip = ArrowTip(angle=-90 * DEGREES)
        tip.set_height(0.15)
        tip.set_fill(YELLOW, 1)
        tip.add_updater(lambda m: m.move_to(line.n2p(value_tracker.get_value()), DOWN))
        tip.add_updater(lambda m: m.set_fill(self.value_to_color(value_tracker.get_value())))

        x_label = Tex(Rf"x_{n}", font_size=30)
        x_label.always.next_to(tip, UP, buff=0.05)
        x_label.always.match_color(tip)

        return VGroup(line, tip, x_label)

    def value_to_color(self, value):
        return interpolate_color_by_hsl(
            GREY_B,
            BLUE if value > 0 else RED,
            abs(value)
        )


class IntersectingCircles(InteractiveScene):
    def construct(self):
        # Add circles
        circles = Circle(radius=2).replicate(4)
        circles.set_stroke(BLUE_B, 2)
        circles[3].set_stroke(YELLOW, 3)
        circles.tri_intersection = ORIGIN  # To change
        circles.pair_intersections = np.zeros((3, 3))  # To change

        vectors = [RIGHT, UL, DL]
        vector_trackers = VGroup(VectorizedPoint(vect) for vect in vectors)

        def update_circles(circles):
            self.place_circles_by_vectors(
                circles,
                [vt.get_center() for vt in vector_trackers]
            )

        circles.add_updater(update_circles)

        dots = GlowDots(circles.pair_intersections)
        dots.set_color(WHITE)
        dots.add_updater(lambda m: m.set_points(circles.pair_intersections))

        circles[3].set_opacity(0)
        self.play(LaggedStartMap(ShowCreation, circles, lag_ratio=0.7))
        self.play(FadeIn(dots))
        self.wait()
        circles[3].set_stroke(opacity=1)
        self.play(ShowCreation(circles[3]))
        self.add(circles)
        self.wait()

        self.play(
            vector_trackers[2].animate.move_to(LEFT + 0.5 * DOWN),
            run_time=2
        )
        self.wait()
        self.play(
            vector_trackers[0].animate.move_to(RIGHT + 0.5 * DOWN),
            run_time=2
        )
        self.wait()
        self.play(
            vector_trackers[0].animate.move_to(RIGHT),
            run_time=2
        )
        self.wait()
        self.play(circles[3].animate.set_opacity(0))
        self.wait()

        # Draw radial lines
        centers = Dot(radius=0.05).replicate(3)
        centers.set_color(WHITE)

        def update_centers(centers):
            for center, circle in zip(centers, circles):
                center.move_to(circle)

        centers.add_updater(update_centers)

        radial_lines = self.get_radial_lines(circles, [vt.get_center() for vt in vector_trackers])

        self.play(
            LaggedStartMap(FadeOut, circles[:3].copy(), lag_ratio=0.5, scale=0),
            FadeIn(centers, lag_ratio=0.5)
        )
        self.wait()
        self.play(ShowCreation(radial_lines[:3], lag_ratio=0.75))
        self.wait()
        for i in range(3, 8, 2):
            self.play(ShowCreation(radial_lines[i:i + 2], lag_ratio=0.5, run_time=1))
            self.wait()
        self.wait()
        self.play(
            LaggedStart(
                (FadeTransform(radial_lines[i1].copy(), radial_lines[i2])
                for i1, i2 in [(5, 9), (8, 10), (4, 11)]),
                lag_ratio=0.75,
                run_time=3
            ),
        )
        self.wait()

        # Animate about
        radial_lines.add_updater(lambda m: m.become(
            self.get_radial_lines(circles, [vt.get_center() for vt in vector_trackers])
        ))

        self.add(circles, centers, radial_lines)
        self.play(
            vector_trackers[1].animate.move_to(UP),
            run_time=3
        )
        self.play(
            vector_trackers[1].animate.move_to(UL),
            run_time=3
        )
        circles[3].set_stroke(opacity=1)
        self.play(ShowCreation(circles[3]))
        self.wait()

    def place_circles_by_vectors(self, circles, vectors):
        radius = circles[0].get_radius()
        radial_vectors = np.array([radius * normalize(vect) for vect in vectors])
        for circle, radial_vector in zip(circles, radial_vectors):
            circle.move_to(radial_vector)
        circles[3].move_to(sum(radial_vectors)),

        circles.tri_intersection = ORIGIN
        circles.pair_intersections = np.array([
            sum(pair) for pair in it.combinations(list(radial_vectors[:3]), 2)
        ])

        return circles

    def get_radial_lines(self, circles, vectors):
        radius = circles[0].get_radius()
        radial_vectors = np.array([radius * normalize(vect) for vect in vectors])

        result = VGroup()
        for vect in radial_vectors:
            result.add(Line(ORIGIN, vect))
        for v1 in radial_vectors:
            for v2 in radial_vectors:
                if np.all(v1 == v2):
                    continue
                result.add(Line(v1, v1 + v2))
        total_sum = sum(radial_vectors)
        for vect in radial_vectors:
            result.add(DashedLine(total_sum - vect, total_sum))

        result.set_stroke(WHITE, 2)
        result[-3:].set_stroke(RED, 2)
        return result


class TriPod(MongesTheorem):
    def construct(self):
        # Initialize frame
        frame = self.frame
        self.set_floor_plane("xz")
        axes = ThreeDAxes()
        axes.set_stroke(width=1)

        # Triangles
        proj_point = 3 * UP
        vects = [DOWN + 0.35 * LEFT + 0.5 * OUT, DOWN + 0.35 * RIGHT + 0.5 * OUT, DOWN + 0.5 * IN]
        factors1 = [2.0, 1.0, 2.0]
        factors2 = [4.5, 4.25, 4.0]
        tri1, tri2 = tris = VGroup(
            Polygon(*(proj_point + f * v for f, v in zip(factors, vects)))
            for factors in [factors1, factors2]
        )
        tri1.set_stroke(BLUE, 3)
        tri1.set_fill(BLUE, 0.5)
        tri2.set_stroke(RED, 3)
        tri2.set_fill(RED, 0.5)

        self.add(tris)

        # Labels
        tri1_labels = VGroup(map(Tex, "ABC"))
        tri2_labels = VGroup(map(Tex, ["A'", "B'", "C'"]))
        tri_labels = VGroup(tri1_labels, tri2_labels)
        tri_labels.scale(0.75)
        for tri, labels in zip(tris, tri_labels):
            center = tri.get_center()
            for label, vert in zip(labels, tri.get_vertices()):
                buff = normalize(vert - center)
                label.move_to(vert + 0.45 * buff)
        tri1_labels[2].scale(1.25).shift(0.15 * DR)
        tri2_labels[2].scale(1.25).shift(0.2 * UR)

        for labels in tri_labels:
            self.play(LaggedStartMap(Write, labels, lag_ratio=0.5, run_time=2))
            self.wait()

        # Tripod legs
        tripod_legs = VGroup(
            Line(vert, proj_point)
            for vert in tri2.get_vertices()
        )
        tripod_legs.set_stroke(WHITE, 1)
        for line in tripod_legs:
            line.scale(4)

        self.play(
            LaggedStartMap(GrowFromCenter, tripod_legs, lag_ratio=0.75, run_time=4),
            VGroup(g[2] for g in tri_labels).animate.shift(0.1 * RIGHT).set_anim_args(time_span=(3, 4))
        )
        self.wait()

        # Intersections
        v1 = tri1.get_vertices()
        v2 = tri2.get_vertices()
        side_pairs = VGroup(
            VGroup(
                Line(v1[i], v1[j]),
                Line(v2[i], v2[j]),
            )
            for (i, j) in [(1, 0), (0, 2), (1, 2)]
        )
        for pair, color in zip(side_pairs, [PINK, YELLOW, TEAL]):
            pair.set_stroke(color, 2)
            for line in pair:
                line.scale(20, about_point=line.get_start())

        side_pairs[1].set_stroke(width=(2, 20))

        int_dots = Group()
        for pair in side_pairs:
            point = self.get_intersection(*pair)
            dot = GlowDot(point)
            int_dots.add(dot)
            self.play(
                ShowCreation(pair, lag_ratio=0),
            )
            self.wait()

        # Add planes
        planes = Group()
        meshes = VGroup()
        for tri in tris:
            a, b, c = tri.get_vertices()
            mat = np.array([
                normalize(b - a),
                normalize(c - a),
                normalize(cross(b - a, c - a))
            ]).T
            plane = Square3D(side_length=50)
            plane.apply_matrix(mat)
            plane.shift(a)
            plane.set_color(GREY_C, 0.25)
            plane.set_shading(1, 0.5, 0.5)
            planes.add(plane)

            mesh = SurfaceMesh(plane, resolution=(101, 101))
            meshes.add(mesh)

        meshes.add(meshes[0].copy().shift(0.03 * DR))
        meshes.set_stroke(WHITE, 0.5, 0.25)

        self.play(
            FadeIn(planes),
            FadeIn(meshes)
        )
        frame.clear_updaters()
        frame.add_updater(lambda m, dt: m.increment_theta(math.sin(self.time / 10) * dt * 2 * DEGREES))
        self.wait(10)
