from manim_imports_ext import *


class CubesAsHexagonTiling(InteractiveScene):
    def setup(self):
        super().setup()
        # Set up axes and camera angle
        self.frame.set_field_of_view(1 * DEGREES)
        self.frame.reorient(135, 55)
        self.axes = ThreeDAxes((-5, 5), (-5, 5), (-5, 5))

    def construct(self):
        # Setup a half cube
        n = 4
        # colors = [RED_E, GREEN_E, BLUE_E]
        colors = [BLUE_B, BLUE_E, BLUE_D]

        base_cube = self.get_half_cube(side_length=4, shared_corner=[-1, -1, - 1], grid=True)
        base_cube.set_fill(GREY)
        base_cube.set_submobject_colors_by_gradient(*colors)
        base_cube.set_stroke(WHITE, 1)
        self.add(base_cube)
        self.add(Point())

        # Subcubes
        coords = np.zeros(3)
        cubes = Group()
        corners = [
            (0, 0, 0), (1, 0, 0), (2, 0, 0),
            (0, 1, 0), (1, 1, 0), (2, 1, 0),
            (0, 2, 0), (1, 2, 0), (2, 2, 0),
            (0, 3, 0), (1, 3, 0), (2, 3, 0),
            (3, 0, 0), (3, 1, 0),
            (0, 0, 1), (1, 0, 1), (2, 0, 1),
            (0, 1, 1), (1, 1, 1), (2, 1, 1),
            (0, 2, 1), (1, 2, 1), (2, 2, 1),
            (0, 0, 2), (1, 0, 2), (2, 0, 2),
            (0, 1, 2), (1, 1, 2), (2, 1, 2),
            (0, 0, 3),
        ]
        for corner in corners:
            cube = self.get_half_cube(corner, colors=colors)
            cubes.add(Point())
            cubes.add(cube)
        cubes.sort(lambda p: np.dot(p, [1, 1, 1]))

        last_cube = cubes[-1]

        cubes.set_shading(0.2, 0.1, 0.2)

        self.add(cubes)

        cubes.remove(last_cube)
        cubes.add(last_cube)

        # Bland colors
        for cube in cubes:
            if isinstance(cube, VMobject):
                cube.set_fill(GREY)
        base_cube.set_fill(GREY)

    def get_half_cube(self, coords=(0, 0, 0), side_length=1, colors=(BLUE_E, BLUE_D, BLUE_B), shared_corner=[1, 1, 1], grid=False):
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
            square.rotate(90.1 * DEGREES, axis)
            square.move_to(ORIGIN, shared_corner)
        squares.move_to(coords, np.array([-1, -1, -1]))
        squares.set_stroke(WHITE, 2)

        return squares



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

        # Hemisphere
        hemisphere = ParametricSurface(
            lambda u, v: [
                np.sin(u) * np.cos(v),
                np.sin(u) * np.sin(v),
                np.cos(u)
            ],
            u_range=(0, PI / 2),
            v_range=(0, 2 * PI)
        )
        hemisphere.set_opacity(0.5)
        hemisphere.set_shading(1, 1, 1)
        hemisphere.always_sort_to_camera(self.camera)

        self.add(hemisphere)

        # Strip
        x0 = 0.5
        x1 = 0.75
        strip = ParametricSurface(
            lambda x, y: [
                x,
                y * np.sqrt(1 - x**2),
                np.sqrt((1 - x**2) * (1 - y**2)),
            ],
            u_range=(x0, x1),
            v_range=(-1, 1),
        )
        strip.set_color(BLUE)
        strip.scale(1.01, about_point=ORIGIN)
        strip.set_shading(1, 1, 1)

        # Show pre_strip
        pre_strip = strip.copy()
        pre_strip.stretch(0, 2, about_edge=IN)

        self.play(ShowCreation(pre_strip))
        self.wait()

        # Expand
        pre_hemisphere = hemisphere.copy()
        pre_hemisphere.stretch(0, 2, about_edge=IN)
        pre_hemisphere.shift(1e-2 * IN)
        pre_hemisphere.set_opacity(0)
        self.play(
            frame.animate.reorient(28, 71, 1),
            run_time=0
        )
        self.wait()
        self.add(pre_strip, pre_hemisphere)
        self.play(
            Transform(pre_hemisphere, hemisphere),
            Transform(pre_strip, strip),
            run_time=3
        )
        self.play(
            frame.animate.reorient(40, 59, 0),
            run_time=5
        )