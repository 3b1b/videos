from manim_imports_ext import *


class WaveMachine(Group):
    def __init__(
        self,
        n_arms=100,
        width=20,
        delta_angle=2 * TAU / 50,
        shaft_radius=0.1,
        arm_radius=0.025,
        arm_length=1.5,
        arm_cap_radius=0.035,
        shaft_color=GREY_D,
        arm_color=GREY_BROWN,
        cap_color=GREY_BROWN,
    ):
        shaft = Cylinder(
            height=width,
            radius=shaft_radius,
            color=shaft_color,
        )
        for point in [shaft.get_zenith(), shaft.get_nadir()]:
            disk = Circle(radius=shaft_radius)
            disk.set_stroke(width=0)
            disk.set_fill(shaft_color, 1)
            disk.move_to(point)
            shaft.add(disk)

        angle = 0
        arms = Group()
        z_range = np.linspace(
            shaft.get_z(IN) + arm_radius + 1e-2,
            shaft.get_z(OUT) - arm_radius - 1e-2,
            n_arms
        )
        for z in z_range:
            arm = Cylinder(
                height=arm_length,
                radius=arm_radius,
                color=arm_color,
            )
            arm.rotate(PI / 2, UP)
            arm.next_to(ORIGIN, RIGHT, buff=0)
            cap = Sphere(radius=arm_cap_radius, color=cap_color)
            cap.move_to(arm.get_right())
            # full_arm = Group(arm, cap)
            full_arm = arm
            full_arm.rotate(angle, about_point=ORIGIN)
            full_arm.shift(z * OUT)
            full_arm.angle = angle
            arms.add(full_arm)

            angle += delta_angle

        super().__init__(shaft, arms)
        self.shaft = shaft
        self.arms = arms
        self.rotate(PI / 2, UP)



class WaveMachineDemo(InteractiveScene, ThreeDScene):
    def construct(self):
        # Add axes (or don't!)
        frame = self.frame
        axes = ThreeDAxes(width=14)
        axes.set_stroke(width=1)

        plane = NumberPlane(
            background_line_style=dict(stroke_color=BLUE_D, stroke_width=1),
            faded_line_style=dict(stroke_color=BLUE_D, stroke_width=0.5, stroke_opacity=0.25),
        )
        plane.axes.set_stroke(BLUE_D, 1)
        plane.set_width(14)

        # Add machine
        machine = WaveMachine(width=14, delta_angle=2 * TAU / 25)
        self.add(machine)

        frame.set_field_of_view(0.2)
        frame.reorient(-68, 79, 0).move_to([-2.41, 1.04, -0.68])
        self.play(
            frame.animate.reorient(0, 78, 0).move_to([-0.0, 0.01, 0.02]).set_height(9.5),
            LaggedStartMap(ShowCreation, machine.arms, lag_ratio=0.1),
            run_time=9,
        )
        self.play(
            Rotating(machine, 5 * TAU, RIGHT, ORIGIN, run_time=20),
        )

        # Reposition
        machine2 = WaveMachine(width=14, delta_angle=2 * TAU / 100)

        self.play(
            frame.animate.reorient(-80, 88, 0).move_to([0.21, -0.56, 0.39]).set_height(11.11),
            run_time=2
        )
        self.play(*(
            Transform(arm1, arm2, path_arc=arm2.angle - arm1.angle, path_arc_axis=RIGHT, run_time=3)
            for arm1, arm2 in zip(machine.arms, machine2.arms)
        ))
        self.play(
            frame.animate.reorient(0, 78, 0).move_to([-0.0, 0.01, 0.02]).set_height(9.5),
            run_time=2
        )
        self.play(
            Rotating(machine, 5 * TAU, RIGHT, ORIGIN, run_time=20),
        )

        # Yet another
        machine3 = WaveMachine(width=14, delta_angle=2 * TAU / 200)
        self.play(*(
            Transform(arm1, arm3, path_arc=arm3.angle - arm2.angle, path_arc_axis=RIGHT)
            for arm1, arm2, arm3 in zip(machine.arms, machine2.arms, machine3.arms)
        ))

        self.play(
            Rotating(machine, 5 * TAU, RIGHT, ORIGIN, run_time=20),
        )

    def get_rotation_arrow_animations(self, x_value=6):
        rot_arrows = VGroup(
            Arrow(RIGHT, LEFT, path_arc=PI, stroke_width=8),
            Arrow(LEFT, RIGHT, path_arc=PI, stroke_width=8),
        )
        rot_arrows.set_height(3)
        rot_arrows.rotate(PI / 2, RIGHT)
        rot_arrows.rotate(PI / 2, OUT)
        rot_arrows.set_flat_stroke(True)
        rot_arrows.set_x(-x_value)
        rot_arrows.add(*rot_arrows.copy().set_x(x_value))

        return (
            Succession(
                ShowCreation(arrow),
                Animation(arrow),
                arrow.animate.set_stroke(BLACK, opacity=0),
                run_time=3,
                remover=True
            )
            for arrow in rot_arrows
        )
        

