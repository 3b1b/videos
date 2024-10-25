from manim_imports_ext import *


class ShowLozenge(InteractiveScene):
    def construct(self):
        # Add Lozenge
        lozenge = Polygon(math.sqrt(3) * LEFT, UP, math.sqrt(3) * RIGHT, DOWN)
        lozenge.scale(2)
        lozenge.set_stroke(TEAL)

        arc1 = Arc(-30 * DEGREES, 60 * DEGREES, arc_center=lozenge.get_left(), radius=0.75)
        arc2 = Arc(-150 * DEGREES, 120 * DEGREES, arc_center=lozenge.get_top(), radius=0.5)

        arc1_label = Tex(R"60^\circ")
        arc1_label.next_to(arc1, RIGHT, MED_SMALL_BUFF)
        arc2_label = Tex(R"120^\circ")
        arc2_label.next_to(arc2, DOWN, MED_SMALL_BUFF)
        angle_labels = VGroup(
            arc1, arc1_label,
            arc2, arc2_label,
        )

        self.add(lozenge)
        self.add(angle_labels)

        # Tile the plane
        verts = lozenge.get_anchors()[:4]
        v1 = verts[1] - verts[0]
        v2 = verts[-1] - verts[0]
        tiles = VGroup(
            lozenge.copy().shift(x * v1 + y * v2)
            for x in range(-10, 11)
            for y in range(-10, 11)
        )
        tiles.sort(lambda p: get_norm(p))
        tiles.set_fill(GREY, 1)
        tiles.set_stroke(WHITE, 2)
        tiles.shift(-tiles[0].get_center())

        self.play(
            self.frame.animate.set_height(40).set_anim_args(run_time=5),
            lozenge.animate.set_fill(GREY, 1),
            LaggedStart(
                (FadeIn(tile) for tile in VGroup(*VectorizedPoint().replicate(len(tiles) // 4), *tiles[1:])),
                lag_ratio=0.01,
                time_span=(1, 5),
            )
        )
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
    # n = 10
    n = 5

    def construct(self):
        # Hugely inefficient
        self.pre_populate()
        for x in range(10):
            new_cube = self.random_new_cube()
            self.animate_in_with_rotation(new_cube)
            old_cube = self.get_random_cube_from_stack()
            if old_cube is not new_cube:
                self.animate_out_with_rotation(old_cube)


class AmbientTilingChangesHexagonBound(AmbientTilingChanges):
    n = 4


class ShowAsThreeD(CubesAsHexagonTiling):
    colors = [BLUE_B, BLUE_D, BLUE_E]

    def construct(self):
        self.pre_populate()

        # Color
        grey_cubes = self.cubes.copy()
        grey_cubes.set_fill(GREY)
        full_grey = Group()
        for cube in grey_cubes:
            full_grey.add(cube, Mobject())
        self.add(full_grey)
        self.base_cube.save_state()
        self.base_cube.set_fill(GREY)

        self.wait()
        self.play(
            Restore(self.base_cube),
            FadeOut(full_grey),
        )

        # Change perspective
        self.play(
            self.frame.animate.reorient(118, 79, 0, (-1.01, 0.75, 1.54), 8.00),
            run_time=3
        )
        self.play(
            self.frame.animate.reorient(163, 83, 0, (1.82, -0.33, 1.89), 8.00),
            run_time=4
        )
        self.play(self.frame.animate.reorient(135, 55, 0, ORIGIN), run_time=3)

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
        for x in range(self.n**3 // 2):
            cube = self.random_new_cube()
            self.play(FadeIn(cube, shift=0.25 * IN, run_time=0.25))
            self.add_cube(cube)


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

        # Add first couple strips
        strip1 = self.get_strip(
            circle, 0.2, 0.8, TAU / 3,
            color=TEAL,
            include_arrow=True,
            label="d_1"
        )
        strip2 = self.get_strip(
            circle, 0.5, 0.75, 2 * TAU / 3,
            color=GREEN,
            include_arrow=True,
            label="d_2",
        )
        strip3 = self.get_strip(
            circle, 0.1, 0.3, 0.8 * TAU,
            color=BLUE,
            include_arrow=True,
            label="d_3",
        )

        self.animate_strip_in(strip1)
        self.wait()
        self.animate_strip_in(strip2)
        self.animate_strip_in(strip3)

        # Cover in lots of strips
        np.random.seed(0)
        strips = VGroup(
            self.get_strip(
                circle,
                *sorted(np.random.uniform(-1, 1, 2)),
                TAU * np.random.uniform(0, TAU),
                opacity=0.5
            ).set_stroke(width=1)
            for n in range(10)
        )
        self.add(strips, strip1, strip2, strip3, circle, radius_label)
        self.play(FadeIn(strips, lag_ratio=0.25))
        self.wait()

        # Question
        syms = Tex(R"\min\left(\sum_i d_i \right)", font_size=60)
        syms.to_edge(RIGHT, buff=MED_SMALL_BUFF)
        self.play(Write(syms))
        self.wait()

    def get_strip(self, circle, r0, r1, theta, color=None, opacity=0.5, include_arrow=False, label=""):
        diam = circle.get_width()
        width = (r1 - r0) * diam / 2
        if color is None:
            color = random_bright_color(luminance_range=(0.5, 0.7))

        rect = Rectangle(width, 2 * diam)
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


class SphereStrips(InteractiveScene):
    def construct(self):
        # Axes
        frame = self.frame
        frame.set_height(3)
        axes = ThreeDAxes((-2, 2), (-2, 2), (-2, 2))
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
        strip = self.get_strip(0.5, 0.75, 0)
        pre_strip = strip.copy()
        pre_strip.stretch(1e-3, 2)

        self.play(ShowCreation(pre_strip))
        self.wait()

        # Expand
        pre_sphere = sphere.copy()
        pre_sphere.stretch(0, 2)
        pre_sphere.shift(1e-2 * IN)
        pre_sphere.set_opacity(0)
        self.play(
            frame.animate.reorient(-34, 59, 0),
            run_time=2
        )
        self.wait()
        self.add(pre_strip, pre_sphere)
        self.play(
            ReplacementTransform(pre_strip, strip),
            ReplacementTransform(pre_sphere, sphere),
            run_time=3
        )
        self.play(
            frame.animate.reorient(40, 59, 0),
            run_time=5
        )
        self.wait()

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
        cyliner = ParametricSurface(
            lambda u, v: [np.cos(v), np.sin(v), u],
            u_range=[-1, 1],
            v_range=[0, TAU],
        )
        cylinder_mesh = SurfaceMesh(cyliner, resolution=(33, 51))
        cylinder_mesh.set_stroke(WHITE, 1, 0.25)

        self.play(ShowCreation(cylinder_mesh, lag_ratio=0.01))
        self.wait()

        # Project the strip
        def clyinder_projection(points):
            radii = np.apply_along_axis(np.linalg.norm, 1, points[:, :2])
            return np.transpose([points[:, 0] / radii, points[:, 1] / radii, points[:, 2]])

        proj_strip = strip.copy().apply_points_function(clyinder_projection)
        proj_strip.set_opacity(0.8)
        proj_strip.save_state()
        proj_strip.become(strip)

        self.add(proj_strip, cylinder_mesh)
        self.play(
            frame.animate.reorient(-28, 62, 0).set_anim_args(run_time=4),
            Restore(proj_strip, run_time=2),
        )
        self.wait()

        # Show a patch of area (another time?)
        patch = ParametricSurface(
            lambda u, v: [
                np.sin(u) * np.cos(v),
                np.sin(u) * np.sin(v),
                np.cos(u)
            ],
            u_range=(45 * DEGREES, 50 * DEGREES),
            v_range=(-60 * DEGREES, -50 * DEGREES)
        )
        patch.set_color(RED)
        proj_patch = patch.copy().apply_points_function(clyinder_projection)
        # proj_patch.save_state()
        # proj_patch.become(patch)

        # self.remove(strip)
        # self.play(
        #     FadeIn(patch),
        #     proj_strip.animate.set_opacity(0.2),
        # )
        # self.add(proj_patch, proj_strip)
        # self.play(
        #     Restore(proj_patch)
        # )

        # Go back to the hemisphere state
        self.play(
            FadeOut(cylinder_mesh),
            FadeOut(proj_strip),
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
            run_time=4,
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
        centers = [[-3, 3, 0], [-6, -1.5, 0], [3, -1.5, 0]]
        colors = [RED, GREEN, BLUE]
        radii = [1, 2, 4]
        circles = VGroup(
            Circle(radius=radius).move_to(center).set_color(color)
            for radius, center, color in zip(radii, centers, colors)
        )
        circles.scale(0.5)
        circles.to_edge(RIGHT, buff=LARGE_BUFF)
        circ1, circ2, circ3 = circles

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

        # self.manipulate_circle_positions(circles)
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
        # self.manipulate_circle_positions(circles)
        self.wait(note="Play with circle positions. Be careful!")
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
        return line_intersection(
            line1.get_start_and_end(),
            line2.get_start_and_end(),
        )

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


class AskAboutVolumeOfParallelpiped(InteractiveScene):
    def construct(self):
        # Axes and plane
        frame = self.frame
        axes = ThreeDAxes((-8, 8), (-4, 4), (-4, 4))
        plane = NumberPlane()
        frame.reorient(-43, 73, 0, (1.18, 0.21, 1.19), 5.96)
        self.add(axes, plane)

        # Tetrahedron
        verts = [
            (0, 1, 1),
            (1, 0, 3),
            (3, 0, 0),
            (2, -2, 0),
        ]
        tetrahedron = VGroup(
            Polygon(*subset)
            for subset in it.combinations(verts, 3)
        )
        tetrahedron.set_stroke(WHITE, 1)
        tetrahedron.set_fill(BLUE, 0.5)
        tetrahedron.set_shading(1, 1, 0)
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
        vects = [LEFT, OUT, OUT + RIGHT, OUT + RIGHT]
        for label, point, vect in zip(labels, dots.get_points(), vects):
            label.rotate(89 * DEGREES, RIGHT)
            label.next_to(point, vect, buff=SMALL_BUFF)

        self.add(labels)

        self.play(
            frame.animate.reorient(4, 78, 0, (1.32, 0.28, 1.22), 5.16),
            run_time=12,
        )


class TriangleAreaFormula(InteractiveScene):
    def construct(self):
        # Set up triangle
        plane = NumberPlane(faded_line_ratio=1)
        plane.add_coordinate_labels(font_size=16)
        plane.background_lines.set_stroke(opacity=0.75)
        plane.faded_lines.set_stroke(opacity=0.25)
        verts = [
            (1, 1, 0),
            (2, -2, 0),
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

        self.add(labels)

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


class IntersectingCircles(InteractiveScene):
    def construct(self):
        # Test
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

        self.add(circles)
        circles[3].set_opacity(0)
        self.add(dots)

        self.play(
            vector_trackers[2].animate.move_to(LEFT + 0.5 * DOWN),
            run_time=3
        )
        circles[3].set_stroke(opacity=1)
        self.play(ShowCreation(circles[3]))
        self.add(circles)
        self.play(
            vector_trackers[0].animate.move_to(RIGHT + 0.5 * DOWN),
            run_time=3
        )
        self.play(
            vector_trackers[0].animate.move_to(RIGHT),
            run_time=3
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
        self.play(ShowCreation(radial_lines[3:9], lag_ratio=0.5, run_time=3))
        self.wait()
        self.play(
            LaggedStart(
                (TransformFromCopy(radial_lines[i1], radial_lines[i2])
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
            result.add(Line(total_sum - vect, total_sum))

        result.set_stroke(WHITE, 2)
        result[-3:].set_stroke(RED, 2)
        return result
