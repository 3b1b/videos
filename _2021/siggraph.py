from manim_imports_ext import *


class ThreePis(Scene):
    def construct(self):
        pis = VGroup(*(PiCreature() for x in range(3)))
        for pi, color in zip(pis, (BLUE_E, BLUE_C, BLUE_D)):
            pi.set_color(color)
        pis.set_height(2)
        pis.arrange(RIGHT, buff=2)
        pis.to_corner(DR)
        pis[2].flip()

        words = ["Choas", "Linear\nalgebra", "Quaternions"]
        modes = ["speaking", "tease", "hooray"]
        for pi, word, mode in zip(pis, words, modes):
            bubble = pi.get_bubble(
                word,
                bubble_type=SpeechBubble,
                height=3, width=3, direction=RIGHT
            )
            bubble.add(bubble.content)
            bubble.next_to(pi, UL)
            bubble.shift(RIGHT)
            pi.change(mode, bubble.content)
            self.add(pi, bubble)
        self.wait()


class PendulumAxes(Scene):
    def construct(self):
        plane = NumberPlane(
            (-2, 2),
            (-2, 2),
            height=8,
            width=8,
            background_line_style={
                "stroke_color": WHITE,
            }
        )
        theta1 = Tex("\\theta_1").next_to(plane.c2p(2, 0), RIGHT, SMALL_BUFF)
        theta2 = Tex("\\theta_2").next_to(plane.c2p(0, 2), DR, SMALL_BUFF)
        labels = VGroup(theta1, theta2)

        x_labels = VGroup(Tex("-\\pi"), Tex("-{\\pi \\over 2}"), Tex("{\\pi \\over 2}"), Tex("\\pi"))
        y_labels = VGroup()
        for label, x in zip(x_labels, (-2, -1, 1, 2)):
            label.scale(0.5)
            label.set_stroke(BLACK, 5, background=True)
            label.next_to(plane.c2p(x, 0), DL, SMALL_BUFF)
            y_label = label.copy()
            y_label.next_to(plane.c2p(0, x), DL, SMALL_BUFF)
            y_labels.add(y_label)

        self.add(plane)
        self.add(labels)
        self.add(x_labels, y_labels)


class InterpolatingOrientations(ThreeDScene):
    matrix = True

    def construct(self):
        # Frame
        frame = self.camera.frame
        frame.set_height(5)
        self.camera.light_source.move_to((-5, -10, 5))

        # 3d model
        color = "#539ac3"
        logo = Cube(color=color)
        logo.replace_submobject(0, TexturedSurface(logo[0], "acm_logo"))
        logo.set_depth(0.5, stretch=True)
        logo.rotate(-45 * DEGREES)
        logo.set_height(2)
        logo.set_opacity(0.8)

        logo_template = logo.copy()

        # Axes
        axes = ThreeDAxes(
            (-3, 3),
            (-3, 3),
            (-2, 2),
            width=6,
            height=6,
            depth=4,
        )
        axes.set_stroke(width=1)
        axes.apply_depth_test()

        # Bases
        bases = DotCloud(np.identity(3))
        bases.set_color((RED, GREEN, BLUE))
        bases.make_3d()

        lines = VGroup(*(Line() for x in range(3)))
        lines.set_submobject_colors_by_gradient(RED, GREEN, BLUE)
        lines.set_stroke(width=2)
        lines.insert_n_curves(20)
        lines.apply_depth_test()

        def update_lines(lines):
            for line, point in zip(lines, bases.get_points()):
                line.put_start_and_end_on(ORIGIN, point)

        lines.add_updater(update_lines)

        def get_matrix():
            return bases.get_points().T

        # Tie logo to matrix
        logo = always_redraw(
            lambda: logo_template.copy().apply_matrix(get_matrix()).sort(lambda p: -p[1])
        )

        # Show matrix/quaternion label
        if self.matrix:
            mat_mob = DecimalMatrix(
                get_matrix(),
                element_to_mobject_config={
                    "num_decimal_places": 2,
                    "edge_to_fix": RIGHT
                }
            )
            mat_mob.to_corner(UL)

            def update_mat_mob(mat_mob):
                for entry, value in zip(mat_mob.get_entries(), get_matrix().flatten()):
                    entry.set_value(value)
                mat_mob.fix_in_frame()
                mat_mob.set_column_colors(RED, GREEN, BLUE)
                return mat_mob

            mat_mob.add_updater(update_mat_mob)

            self.add(mat_mob)
        else:
            quat_tracker = ValueTracker([1, 0, 0, 0])
            bases.add_updater(lambda m: m.set_points(
                rotation_matrix_from_quaternion(quat_tracker.get_value()).T
            ))

            kw = {"num_decimal_places": 3, "include_sign": True}
            quat_label = VGroup(
                VGroup(DecimalNumber(1, **kw)),
                VGroup(DecimalNumber(0, **kw), Tex("i")),
                VGroup(DecimalNumber(0, **kw), Tex("j")),
                VGroup(DecimalNumber(0, **kw), Tex("k")),
            )
            for mob in quat_label:
                mob.arrange(RIGHT, buff=SMALL_BUFF)
            quat_label.arrange(DOWN, aligned_edge=LEFT, buff=MED_SMALL_BUFF)
            quat_label.to_corner(UL)
            quat_label.shift(2 * RIGHT)

            def update_quat_label(quat_label):
                colors = [WHITE, RED, GREEN, BLUE]
                for mob, value, color in zip(quat_label, quat_tracker.get_value(), colors):
                    mob[0].set_value(value)
                    mob.set_color(color)
                quat_label.fix_in_frame()
                return quat_label

            quat_label.add_updater(update_quat_label)
            self.add(quat_label)

        # Show 3d scene
        self.add(axes)
        self.add(bases)
        self.add(lines)
        self.add(logo)

        def move_to_matrix(matrix, run_time=5):
            self.play(
                bases.animate.set_points(np.transpose(matrix)),
                run_time=run_time
            )
            self.wait()

        def move_to_quaternion(quaternion, run_time=5):
            self.play(
                quat_tracker.animate.set_value(quaternion),
                UpdateFromFunc(
                    quat_tracker,
                    lambda m: m.set_value(normalize(m.get_value()))
                ),
                run_time=run_time
            )
            self.wait()

        self.play(
            ShowCreation(axes),
            frame.animate.reorient(-20, 60), run_time=3
        )
        self.wait()

        # Using the matrix to interpolate
        if self.matrix:
            move_to_matrix(z_to_vector(RIGHT))
            move_to_matrix(z_to_vector(LEFT))
            move_to_matrix(z_to_vector(DOWN + RIGHT))
            move_to_matrix(z_to_vector(UP + LEFT + OUT))
        else:
            move_to_quaternion(quaternion_from_angle_axis(PI / 2, UP))
            move_to_quaternion(quaternion_from_angle_axis(-PI / 2, UP))
            move_to_quaternion(quaternion_from_angle_axis(PI / 2, UR))
            move_to_quaternion(quaternion_from_angle_axis(-0.3 * PI, UR))


class InterpolatingOrientationsWithQuaternions(InterpolatingOrientations):
    matrix = False


class FirstStepIsToCare(Scene):
    def construct(self):
        morty = Mortimer()
        morty.to_corner(DR)
        self.play(PiCreatureSays(morty, TexText("The first step\\\\is to care.")))


class NeverNeeded(Scene):
    def construct(self):
        formula = Tex("\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}")
        formula.set_height(2)
        formula.to_corner(UL)
        self.add(formula)

        morty = Mortimer()
        morty.to_corner(DR)
        self.play(PiCreatureSays(morty, TexText("Who actually\\\\uses it!"), target_mode="angry"))


class Thanks(Scene):
    def construct(self):
        morty = Mortimer()
        morty.to_corner(DR)
        self.play(PiCreatureSays(morty, TexText("Thanks"), target_mode="gracious"))
        morty.look_at(OUT)
