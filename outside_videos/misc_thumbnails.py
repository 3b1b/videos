from manim_imports_ext import *


class LinalgThumbnail(ThreeDScene):
    CONFIG = {
        "camera_config": {
            "anti_alias_width": 0,
        }
    }

    def construct(self):
        grid = NumberPlane((-10, 10), (-10, 10), faded_line_ratio=1)
        grid.set_stroke(width=6)
        grid.faded_lines.set_stroke(width=1)
        grid.apply_matrix([[3, 2], [1, -1]])
        # self.add(grid)

        frame = self.camera.frame
        frame.reorient(0, 75)

        cube = Cube()
        cube.set_color(BLUE)
        cube.set_opacity(0.5)

        edges = VGroup()
        for vect in [OUT, RIGHT, UP, LEFT, DOWN, IN]:
            face = Square()
            face.shift(OUT)
            face.apply_matrix(z_to_vector(vect))
            edges.add(face)
        for sm in edges.family_members_with_points():
            sm.flat_stroke = False
            sm.joint_type = "round"

        edges.set_stroke(WHITE, 4)
        edges.replace(cube)
        edges.apply_depth_test()
        cube = Group(cube, edges)

        cube2 = cube.copy().apply_matrix(
            [[1, 0, 1], [0, 1, 0], [0, 0, 1]]
        )
        # cube2.match_height(cube)
        arrow = Vector(RIGHT)
        arrow.rotate(PI / 2, RIGHT)
        group = Group(cube, arrow, cube2)
        group.arrange(RIGHT, buff=MED_LARGE_BUFF)
        self.add(group)

        # kw ={
        #     "thickness": 0.1,
        #     # "max_tip_length_to_length_ratio": 0.2,
        # }
        # self.add(Vector(grid.c2p(1, 0), fill_color=GREEN, **kw))
        # self.add(Vector(grid.c2p(0, 1), fill_color=RED, **kw))

        # self.add(FullScreenFadeRectangle(fill_opacity=0.1))


class CSThumbnail(Scene):
    def construct(self):
        self.add(self.get_background())

    def get_background(self, n=12, k=50, zero_color=GREY_C, one_color=GREY_B):
        choices = (Integer(0, color=zero_color), Integer(1, color=one_color))
        background = VGroup(*(
            random.choice(choices).copy()
            for x in range(n * k)
        ))
        background.arrange_in_grid(n, k)
        background.set_height(FRAME_HEIGHT)
        return background


class GroupThumbnail(ThreeDScene):
    def construct(self):
        cube = Cube()
        cubes = Group(cube)
        for axis in [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]]:
            for angle in [60 * DEGREES]:
                cubes.add(cube.copy().rotate(angle, axis))

        cubes.rotate(95 * DEGREES, RIGHT)
        cubes.rotate(30 * DEGREES, UP)
        cubes.set_height(6)
        cubes.center()
        # cubes.set_y(-0.5)
        cubes.set_color(BLUE_D)
        cubes.set_shadow(0.65)
        cubes.set_gloss(0.5)
        self.add(cubes)


class BaselThumbnail(Scene):
    def construct(self):
        # Lake
        lake_radius = 6
        lake_center = ORIGIN

        lake = Circle(
            fill_color=BLUE,
            fill_opacity=0.0,
            radius=lake_radius,
            stroke_color=BLUE_D,
            stroke_width=3,
        )
        lake.move_to(lake_center)

        R = 2
        light_template = VGroup()
        rs = np.linspace(0, 1, 100)
        for r1, r2 in zip(rs, rs[1:]):
            dot1 = Dot(radius=R * r1).flip()
            dot2 = Dot(radius=R * r2)
            dot2.append_vectorized_mobject(dot1)
            dot2.insert_n_curves(100)
            dot2.set_fill(YELLOW, opacity=0.5 * (1 - r1)**2)
            dot2.set_stroke(width=0)
            light_template.add(dot2)

        houses = VGroup()
        lights = VGroup()
        for i in range(16):
            theta = -TAU / 4 + (i + 0.5) * TAU / 16
            pos = lake_center + lake_radius * np.array([np.cos(theta), np.sin(theta), 0])
            house = Lighthouse()
            house.set_fill(GREY_B)
            house.set_stroke(width=0)
            house.set_height(0.5)
            house.move_to(pos)
            light = light_template.copy()
            light.move_to(pos)
            houses.add(house)
            lights.add(light)

        self.add(lake)
        self.add(houses)
        self.add(lights)

        # Equation
        equation = Tex(
            "1", "+", "{1 \\over 4}", "+",
            "{1 \\over 9}", "+", "{1 \\over 16}", "+",
            "{1 \\over 25}", "+", "\\cdots"
        )
        equation.scale(1.8)
        equation.move_to(2 * UP)
        answer = Tex("= \\frac{\\pi^2}{6}", color=YELLOW)
        answer.scale(3)
        answer.move_to(1.25 * DOWN)
        equation.add(answer)

        shadow = VGroup()
        for w in np.linspace(20, 0, 50):
            shadow.add(equation.copy().set_fill(opacity=0).set_stroke(BLACK, width=w, opacity=0.02))
        self.add(shadow)
        self.add(equation)

        self.wait()


class Eola1Thumbnail(Scene):
    def construct(self):
        plane = NumberPlane(
            x_range=(-2, 2),
            y_range=(-5, 5),
        )
        plane.set_width(FRAME_WIDTH / 3)
        plane.to_edge(LEFT, buff=0)
        plane.shift(1.5 * DOWN)
        vect = Arrow(
            plane.get_origin(), plane.c2p(1, 2),
            buff=0,
            thickness=0.1,
        )
        vect.set_color(YELLOW)
        self.add(plane, vect)

        coords = IntegerMatrix([[1], [2]])
        coords.set_height(3)
        coords.set_color(TEAL)
        coords.center()
        coords.match_y(vect)
        self.add(coords)

        symbol = Tex("\\vec{\\textbf{v} } \\in V")
        symbol.set_color(BLUE)
        symbol.set_width(FRAME_WIDTH / 3 - 1)
        symbol.set_x(FRAME_WIDTH / 3)
        symbol.match_y(vect)
        self.add(symbol)

        lines = VGroup(*(Line(DOWN, UP) for x in range(2)))
        lines.set_height(FRAME_HEIGHT)
        lines.arrange(RIGHT, buff=FRAME_WIDTH / 3)
        lines.set_stroke(GREY_A, 5)
        self.add(lines)

        title = Text("Vectors", font_size=120)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        shadow = VGroup()
        for w in np.linspace(50, 0, 100):
            shadow.add(title.copy().set_fill(opacity=0).set_stroke(BLACK, width=w, opacity=0.01))
        self.add(shadow)
        self.add(title)


def pendulum_vector_field_func(theta, omega, mu=0.3, g=9.8, L=3):
    return [omega, -np.sqrt(g / L) * np.sin(theta) - mu * omega]


class ODEThumbnail(Scene):
    def construct(self):
        plane = NumberPlane()
        field = VectorField(
            pendulum_vector_field_func, plane,
            step_multiple=0.5,
            magnitude_range=(0, 5),
            length_func=lambda norm: 0.35 * sigmoid(norm),
        )
        field.set_opacity(0.75)

        # self.add(plane)
        # self.add(field)
        # return

        # Solution curve

        dt = 0.1
        t = 0
        total_time = 50

        def func(point):
            return plane.c2p(*pendulum_vector_field_func(*plane.p2c(point)))

        points = [plane.c2p(-4 * TAU / 4, 4.0)]
        while t < total_time:
            t += dt
            points.append(points[-1] + dt * func(points[-1]))

        line = VMobject()
        line.set_points_smoothly(points, true_smooth=True)
        line.set_stroke([WHITE, WHITE, BLACK], width=[5, 1])
        # line.set_stroke((BLUE_C, BLUE_E), width=(10, 1))

        line_fuzz = VGroup()
        N = 50
        for width in np.linspace(50, 0, N):
            line_fuzz.add(line.copy().set_stroke(BLACK, width=width, opacity=2 / N))

        self.add(line_fuzz)
        self.add(line)
