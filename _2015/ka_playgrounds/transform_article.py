from manim_imports_ext import *

def half_plane():
    plane = NumberPlane(
        x_radius = FRAME_X_RADIUS/2,
        x_unit_to_spatial_width  = 0.5,
        y_unit_to_spatial_height = 0.5,
        x_faded_line_frequency = 0,
        y_faded_line_frequency = 0,
        density = 4*DEFAULT_POINT_DENSITY_1D,
    )
    plane.add_coordinates(
        x_vals = list(range(-6, 7, 2)),
        y_vals = list(range(-6, 7, 2))
    )
    return plane

class SingleVariableFunction(Scene):
    args_list = [
        (lambda x : x**2 - 3, "ShiftedSquare", True),
        (lambda x : x**2 - 3, "ShiftedSquare", False),
    ]

    @staticmethod
    def args_to_string(func, name, separate_lines):
        return name + ("SeparateLines" if separate_lines else "")

    def construct(self, func, name, separate_lines):
        base_line = NumberLine(color = "grey")
        moving_line = NumberLine(
            tick_frequency = 1, 
            density = 3*DEFAULT_POINT_DENSITY_1D
        )
        base_line.add_numbers()
        def point_function(xxx_todo_changeme):
            (x, y, z) = xxx_todo_changeme
            return (func(x), y, z)
        target = moving_line.copy().apply_function(point_function)

        transform_config = {
            "run_time" : 3,
            "path_func" : path_along_arc(np.pi/4)
        }

        if separate_lines:
            numbers = moving_line.get_number_mobjects(*list(range(-7, 7)))
            negative_numbers = []
            for number in numbers:
                number.set_color(GREEN_E)
                number.shift(-2*moving_line.get_vertical_number_offset())
                center = number.get_center()
                target_num = number.copy()
                target_num.shift(point_function(center) - center)
                target.add(target_num)
                if center[0] < -0.5:
                    negative_numbers.append(number)
            moving_line.add(*numbers)
            base_line.shift(DOWN)
            target.shift(DOWN)
            moving_line.shift(UP)

        self.add(base_line, moving_line)
        self.wait(3)
        self.play(Transform(moving_line, target, **transform_config))
        if separate_lines:
            self.play(*[
                ApplyMethod(mob.shift, 0.4*UP)
                for mob in negative_numbers
            ])
        self.wait(3)


class LineToPlaneFunction(Scene):
    args_list = [
        (lambda x : (np.cos(x), 0.5*x*np.sin(x)), "Swirl", []),
        (lambda x : (np.cos(x), 0.5*x*np.sin(x)), "Swirl", [
            ("0", "(1, 0)", 0),
            ("\\frac{\\pi}{2}",  "(0, \\pi / 4)", np.pi/2),
            ("\\pi", "(-1, 0)", np.pi),
        ])        
    ]

    @staticmethod
    def args_to_string(func, name, numbers_to_follow):
        return name + ("FollowingNumbers" if numbers_to_follow else "")

    def construct(self, func, name, numbers_to_follow):
        line = NumberLine(
            unit_length_to_spatial_width = 0.5,
            tick_frequency = 1,
            number_at_center = 6,
            numerical_radius = 6,
            numbers_with_elongated_ticks = [0, 12],
            density = 3*DEFAULT_POINT_DENSITY_1D
        )
        line.to_edge(LEFT)
        line_copy = line.copy()
        line.add_numbers(*list(range(0, 14, 2)))
        divider = Line(FRAME_Y_RADIUS*UP, FRAME_Y_RADIUS*DOWN)
        plane = half_plane()
        plane.submobjects = []
        plane.filter_out(
            lambda x_y_z2 : abs(x_y_z2[0]) > 0.1 and abs(x_y_z2[1]) > 0.1
        )
        plane.shift(0.5*FRAME_X_RADIUS*RIGHT)
        self.add(line, divider, plane)

        def point_function(point):
            x, y = func(line.point_to_number(point))
            return plane.num_pair_to_point((x, y))

        target = line_copy.copy().apply_function(point_function)
        target.set_color()
        anim_config = {"run_time" : 3}
        anims = [Transform(line_copy, target, **anim_config)]

        colors = iter([BLUE_B, GREEN_D, RED_D])
        for input_tex, output_tex, number in numbers_to_follow:
            center = line.number_to_point(number)
            dot = Dot(center, color = next(colors))
            anims.append(ApplyMethod(
                dot.shift, 
                point_function(center) - center, 
                **anim_config 
            ))
            label = Tex(input_tex)
            label.shift(center + 2*UP)
            arrow = Arrow(label, dot)
            self.add(label)
            self.play(ShowCreation(arrow), ShowCreation(dot))
            self.wait()
            self.remove(arrow, label)


        self.wait(2)
        self.play(*anims)
        self.wait()

        for input_tex, output_tex, number in numbers_to_follow:
            point = plane.num_pair_to_point(func(number))
            label = Tex(output_tex)
            side_shift = LEFT if number == np.pi else RIGHT
            label.shift(point, 2*UP, side_shift)
            arrow = Arrow(label, point)
            self.add(label)
            self.play(ShowCreation(arrow))
            self.wait(2)
            self.remove(arrow, label)

class PlaneToPlaneFunctionSeparatePlanes(Scene):
    args_list = [
        (lambda x_y3 : (x_y3[0]**2+x_y3[1]**2, x_y3[0]**2-x_y3[1]**2), "Quadratic")
    ]
    @staticmethod
    def args_to_string(func, name):
        return name

    def construct(self, func, name):
        shift_factor = 0.55
        in_plane  = half_plane().shift(shift_factor*FRAME_X_RADIUS*LEFT)
        out_plane = half_plane().shift(shift_factor*FRAME_X_RADIUS*RIGHT)
        divider = Line(FRAME_Y_RADIUS*UP, FRAME_Y_RADIUS*DOWN)
        self.add(in_plane, out_plane, divider)

        plane_copy = in_plane.copy()
        plane_copy.submobjects = []

        def point_function(point):
            result = np.array(func((point*2 + 2*shift_factor*FRAME_X_RADIUS*RIGHT)[:2]))
            result = np.append(result/2, [0])
            return result + shift_factor*FRAME_X_RADIUS*RIGHT

        target = plane_copy.copy().apply_function(point_function)
        target.set_color(GREEN_B)

        anim_config = {"run_time" : 5}

        self.wait()
        self.play(Transform(plane_copy, target, **anim_config))
        self.wait()

class PlaneToPlaneFunction(Scene):
    args_list = [
        (lambda x_y4 : (x_y4[0]**2+x_y4[1]**2, x_y4[0]**2-x_y4[1]**2), "Quadratic")
    ]
    @staticmethod
    def args_to_string(func, name):
        return name

    def construct(self, func, name):
        plane = NumberPlane()
        plane.prepare_for_nonlinear_transform()
        background = NumberPlane(color = "grey")
        background.add_coordinates()
        anim_config = {"run_time" : 3}

        def point_function(point):
            return np.append(func(point[:2]), [0])

        self.add(background, plane)
        self.wait(2)
        self.play(ApplyPointwiseFunction(point_function, plane, **anim_config))
        self.wait(3)

class PlaneToLineFunction(Scene):
    args_list = [
        (lambda x_y : x_y[0]**2 + x_y[1]**2, "Bowl"),
    ]

    @staticmethod
    def args_to_string(func, name):
        return name

    def construct(self, func, name):
        line = NumberLine(
            color = GREEN,
            unit_length_to_spatial_width = 0.5,
            tick_frequency = 1,
            number_at_center = 6,
            numerical_radius = 6,
            numbers_with_elongated_ticks = [0, 12],
        ).to_edge(RIGHT)
        line.add_numbers()
        plane = half_plane().to_edge(LEFT, buff = 0)

        divider = Line(FRAME_Y_RADIUS*UP, FRAME_Y_RADIUS*DOWN)
        line_left = line.number_to_point(0)
        def point_function(point):
            shifter = 0.5*FRAME_X_RADIUS*RIGHT
            return func((point+shifter)[:2])*RIGHT + line_left

        self.add(line, plane, divider)
        self.wait()
        plane.submobjects = []
        self.play(ApplyPointwiseFunction(point_function, plane))
        self.wait()



class PlaneToSpaceFunction(Scene):
    args_list = [
        (lambda x_y1 : (x_y1[0]*x_y1[0], x_y1[0]*x_y1[1], x_y1[1]*x_y1[1]), "Quadratic"),
    ]

    @staticmethod
    def args_to_string(func, name):
        return name

    def construct(self, func, name):
        plane = half_plane().shift(0.5*FRAME_X_RADIUS*LEFT)
        divider = Line(FRAME_Y_RADIUS*UP, FRAME_Y_RADIUS*DOWN)
        axes = XYZAxes()
        axes.filter_out(lambda p : get_norm(p) > 3)
        rot_kwargs = {
            "run_time" : 3,
            "radians"  : 0.3*np.pi,
            "axis"     : [0.1, 1, 0.1],
        }
        axes.to_edge(RIGHT).shift(DOWN)        
        dampening_factor = 0.1
        def point_function(xxx_todo_changeme5):
            (x, y, z) = xxx_todo_changeme5
            return dampening_factor*np.array(func((x, y)))
        target = NumberPlane().apply_function(point_function)
        target.set_color("yellow")
        target.shift(axes.get_center())

        self.add(plane, divider, axes)
        self.play(Rotating(axes, **rot_kwargs))

        target.rotate(rot_kwargs["radians"])
        self.play(
            TransformAnimations(
                Animation(plane.copy()),
                Rotating(target, **rot_kwargs),
                rate_func = smooth
            ),
            Rotating(axes, **rot_kwargs)
        )
        axes.add(target)
        self.clear()
        self.add(plane, divider, axes)
        self.play(Rotating(axes, **rot_kwargs))
        self.clear()
        for i in range(5):
            self.play(Rotating(axes, **rot_kwargs))


class SpaceToSpaceFunction(Scene):
    args_list = [
        (lambda x_y_z : (x_y_z[1]*x_y_z[2], x_y_z[0]*x_y_z[2], x_y_z[0]*x_y_z[1]), "Quadratic"),
    ]

    @staticmethod
    def args_to_string(func, name):
        return name

    def construct(self, func, name):
        space = SpaceGrid()
        rot_kwargs = {
            "run_time" : 10,
            "radians"  : 2*np.pi/5,
            "axis"     : [0.1, 1, 0.1],
            "in_place" : False,
        }
        axes = XYZAxes()
        target = space.copy().apply_function(func)

        self.play(
            TransformAnimations(
                Rotating(space, **rot_kwargs),
                Rotating(target, **rot_kwargs),
                rate_func = squish_rate_func(smooth, 0.3, 0.7)
            ),
            Rotating(axes, **rot_kwargs)
        )
        axes.add(space)
        self.play(Rotating(axes, **rot_kwargs))



























