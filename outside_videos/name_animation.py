#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from manim_imports_ext import *
from _2019.diffyq.part2.fourier_series import FourierOfTexPaths


# I'm guessing most of this needs to be fixed...
# class ComplexMorphingNames(ComplexTransformationScene):
class ComplexMorphingNames(Scene):
    CONFIG = {
        "patron_name": "Janel",
        "function": lambda z: 0.2 * (z**3),
        "default_apply_complex_function_kwargs": {
            "run_time": 5,
        },
        "output_directory": os.path.join(get_output_dir(), "EightDollarPatrons"),
        "include_coordinate_labels": False,
        "vert_start_color": YELLOW,  # TODO
        "vert_end_color": PINK,
        "horiz_start_color": GREEN,
        "horiz_end_color": BLUE,
        "use_multicolored_plane": True,
        # "plane_config" : {
        #     "unit_size" : 1.5,
        # },
    }

    def construct(self):
        name = self.patron_name
        self.clear()
        self.frames = []
        self.setup()
        self.add_transformable_plane()
        self.plane.fade()

        name_mob = TexText(name)
        name_mob.set_width(4)
        name_mob.next_to(ORIGIN, UP, LARGE_BUFF)
        self.start_vect = name_mob.get_center()
        for submob in name_mob.family_members_with_points():
            submob.insert_n_curves(100)
        name_mob_copy = name_mob.copy()

        self.play(Write(name_mob))
        self.play(
            self.get_rotation(name_mob),
            run_time=5,
        )
        self.wait()
        self.add_transformable_mobjects(name_mob)
        self.apply_complex_function(self.function)
        self.wait()
        self.play(
            self.get_post_transform_rotation(name_mob, name_mob_copy),
            run_time=10
        )
        self.wait(3)

    def get_rotation(self, name_mob):
        return UpdateFromAlphaFunc(
            name_mob,
            lambda mob, alpha: mob.move_to(rotate_vector(
                self.start_vect, 2 * np.pi * alpha
            ))
        )

    def get_post_transform_rotation(self, name_mob, name_mob_copy):
        simple_rotation = self.get_rotation(name_mob_copy)

        def update(name_mob, alpha):
            simple_rotation.update(alpha)
            new_name = simple_rotation.mobject.copy()
            new_name.apply_complex_function(self.function)
            Transform(name_mob, new_name).update(1)
            return name_mob
        return UpdateFromAlphaFunc(name_mob, update)


class FlowNameAnimation(Scene):
    CONFIG = {
        "patron_name": "Test Name"
    }

    def construct(self):
        name_mob = TexText(self.patron_name)
        name_mob.scale(2)
        max_width = FRAME_WIDTH - 2
        if name_mob.get_width() > max_width:
            name_mob.set_width(max_width)
        name_strokes = VGroup()
        for mob in name_mob.family_members_with_points():
            mob.insert_n_curves(20)
            anchors1, handles1, handles2, anchors2 = mob.get_anchors_and_handles()
            for a1, h1, h2, a2 in zip(anchors1, handles1, handles2, anchors2):
                stroke = VMobject()
                stroke.set_points([a1, h1, h2, a2])
                stroke.set_stroke(WHITE, width=2)
                name_strokes.add(stroke)
                stroke.save_state()

        from _2017.eoc.div_curl import four_swirls_function
        from _2017.eoc.div_curl import VectorField
        from _2017.eoc.div_curl import move_submobjects_along_vector_field
        func = four_swirls_function
        vector_field = VectorField(func)
        vector_field.submobjects.sort(
            key=lambda a: a.get_length()
        )
        flow = move_submobjects_along_vector_field(name_strokes, func)

        self.add_foreground_mobjects(name_strokes)
        self.play(Write(name_strokes))
        self.play(LaggedStartMap(GrowArrow, vector_field))
        self.add(flow)
        self.wait(60)
        self.remove(flow)
        self.play(
            FadeOut(vector_field),
            LaggedStartMap(
                ApplyMethod, name_strokes,
                lambda m: (m.restore,),
                lag_ratio=0.2
            ),
            run_time=5,
        )
        self.wait()


class NameAnimationScene(Scene):
    CONFIG = {
        "animated_name": "Test name",
        "all_names": [
            "范英睿",
        ],
        "animate_all_names": True,
        "linger_after_completion": False,
    }

    def run(self):
        if self.animate_all_names:
            for name in self.all_names:
                self.__init__()
                self.camera.frame_rate = 30
                try:
                    self.file_writer.file_name = name.replace(" ", "") + self.__class__.__name__
                    self.file_writer.init_output_directories()
                    self.file_writer.begin()
                    self.num_plays = 0
                    # Allow subclasses to alter this name, for example by lengthening it.
                    name = self.edit_name_text(name)
                    self.animated_name = name
                    self.clear()
                    self.setup()
                    self.construct()
                    self.tear_down()
                except Exception as inst:
                    raise inst
                    # print(inst)
                    # print(f"Problem with {name}")
        else:
            super().__init__(**kwargs)

    def edit_name_text(self, name):
        return name


class RotatingNameLetters(NameAnimationScene):
    def edit_name_text(self, name):
        if len(name) < 8:
            name += " Is Delightful"
        return name

    def construct(self):
        diameter = 3.0
        radius = diameter / 2
        letter_scale = 1

        name = self.animated_name.replace(" ", "$\\cdot$")
        name += "$\\cdot$"
        text_mob = TexText(name)
        text_mob.set_stroke(BLACK, 2, background=True)
        # for part in text_mob.get_parts_by_tex("$\\cdot$"):
        #     part.set_opacity(0)
        letter_mobs = text_mob[0]
        nb_letters = len(letter_mobs)
        randy = PiCreature()
        randy.move_to(ORIGIN).set_height(0.5 * diameter)
        randy.set_color(BLUE_E)
        randy.look_at(UP + RIGHT)
        self.add(randy)
        dtheta = TAU / nb_letters
        angles = np.arange(TAU / 4, -3 * TAU / 4, -dtheta)
        name_mob = VGroup()
        for (letter_mob, angle) in zip(letter_mobs, angles):
            letter_mob.scale(letter_scale)
            pos = radius * np.cos(angle) * RIGHT + radius * np.sin(angle) * UP
            letter_mob.move_to(pos)
            name_mob.add(letter_mob)

        pos2 = radius * np.cos(angles[2]) * RIGHT + \
            radius * np.sin(angles[2]) * UP

        times_n_label = VGroup(
            Tex("\\times"),
            Integer(1)
        )
        times_n_label.arrange(RIGHT)
        times_n_label.shift(FRAME_WIDTH * RIGHT / 4)
        times_n_label.to_edge(UP)

        self.play(
            LaggedStartMap(FadeIn, name_mob, run_time=3),
            ApplyMethod(randy.change, "pondering", pos2, run_time=1),
            FadeIn(times_n_label)
        )

        for n in range(2, nb_letters + 2):

            group = []

            for (j, letter_mob) in enumerate(name_mob.submobjects):

                new_angle = TAU / 4 - n * j * dtheta
                new_pos = radius * np.cos(new_angle) * \
                    RIGHT + radius * np.sin(new_angle) * UP
                letter_mob.target = letter_mob.copy().move_to(new_pos)
                anim = MoveToTarget(letter_mob, path_arc=- j * dtheta)
                group.append(anim)
            new_n = Integer(n)
            new_n.move_to(times_n_label[1])
            self.play(
                AnimationGroup(*group, run_time=3),
                UpdateFromFunc(randy, lambda r: r.look_at(name_mob.submobjects[2])),
                FadeOut(times_n_label[1]),
                FadeIn(new_n)
            )
            times_n_label.submobjects[1] = new_n
            self.wait(0.5)

        thank_you = TexText("Thank You!").next_to(randy, DOWN)
        new_randy = randy.copy()
        new_randy.change("hooray")
        new_randy.set_color(BLUE_E)
        new_randy.look_at(ORIGIN)
        self.play(
            ReplacementTransform(name_mob, VGroup(*thank_you)),
            Transform(randy, new_randy)
        )
        self.play(Blink(randy))


class ModularMultiplicationNameAnimation(RotatingNameLetters):
    def construct(self):
        max_width = FRAME_WIDTH - 4
        char_radius = 3
        index_radius = 2.5
        text = TexText(self.animated_name)[0]
        N = len(text)

        text.scale(2)
        text.set_stroke(BLACK, 5, background=True)
        if text.get_width() > max_width:
            text.set_width(max_width)

        circle_text = text.copy()
        alphas = np.arange(0, TAU, TAU / N)
        for char, theta in zip(circle_text, alphas):
            char.move_to(char_radius * np.array([
                np.sin(theta),
                np.cos(theta),
                0,
            ]))

        index_mobs = VGroup()
        for i, char in enumerate(circle_text):
            index = Integer(i)
            index.move_to((index_radius / char_radius) * char.get_center())
            index.scale(0.5)
            index.set_color(YELLOW)
            char.index = 0
            index_mobs.add(index)

        self.play(FadeInFromDown(text))
        self.wait()
        self.play(
            Transform(text, circle_text),
            FadeIn(index_mobs),
            lag_ratio=0.2,
            run_time=3,
        )
        self.wait()

        # Multiplications
        # last_lines = VMobject()
        # last_label = VMobject()
        for k in range(2, N + 1):
            text.generate_target()
            text.save_state()
            for i, char in enumerate(text.target):
                char.move_to(circle_text[(i * k) % N])

            lines = VGroup(*[
                Line(
                    index_mobs[i],
                    index_mobs[(i * k) % N],
                    buff=SMALL_BUFF,
                ).add_tip(0.1).set_opacity(
                    0 if (i * k) % N == i else 1
                )
                for i in range(N)
            ])
            lines.set_color(MAROON_B)
            lines.set_stroke(width=1)

            label = VGroup(Tex("\\times"), Integer(k))
            label.arrange(RIGHT, buff=SMALL_BUFF)
            label.scale(2)
            label.next_to(circle_text, UR)
            label.shift_onto_screen()
            label.set_color(MAROON_B)

            kw = {
                "run_time": 5,
                "lag_ratio": 0.5,
                "rate_func": lambda t: smooth(t, 2),
            }
            self.play(
                MoveToTarget(text, **kw),
                ShowCreation(lines, **kw),
                FadeIn(label),
            )
            self.wait()

            text_copy = text.copy()
            self.add(text_copy)
            self.remove(text)
            text.restore()

            self.play(
                FadeOut(lines),
                FadeOut(label),
                FadeOut(text_copy),
                FadeIn(text),
            )


class FourierNameAnimation(FourierOfTexPaths, NameAnimationScene):
    pass


class QuaternionNameAnimation(Scene):
    CONFIG = {
        "R": 2,
    }

    def construct(self):
        surface = ParametricSurface(lambda u, v: (u, v, 0), resolution=16)
        surface.set_width(self.R * TAU)
        surface.set_height(1.8 * self.R, stretch=True)
        surface.center()
        surface.set_fill(opacity=0.5)
        name = TexText(self.name_text)
        name.set_width(self.R * TAU - 1)
        max_height = 0.4 * surface.get_height()
        if name.get_height() > max_height:
            name.set_height(max_height)
        name.next_to(surface.get_top(), DOWN)
        for letter in name:
            letter.add(VectorizedPoint(letter.get_center() + 2 * OUT))
            letter.set_shade_in_3d(True, z_index_as_group=True)
            # for submob in letter.family_members_with_points():
            #     submob.pre_function_handle_to_anchor_scale_factor = 0.001

        axes = self.get_axes()

        self.play(
            Write(surface),
            Write(name),
        )
        surface.add(name)
        self.wait()
        self.move_camera(
            phi=70 * DEGREES,
            theta=-140 * DEGREES,
            added_anims=[
                ApplyPointwiseFunction(self.plane_to_cylinder, surface),
                FadeIn(axes),
            ],
            run_time=3,
        )
        self.begin_ambient_camera_rotation(0.01)
        self.wait(2)
        self.play(
            ApplyPointwiseFunction(self.cylinder_to_sphere, surface),
            run_time=3
        )
        # self.play(Rotating(
        #     surface, angle=-TAU, axis=OUT,
        #     about_point=ORIGIN,
        #     run_time=4,
        #     rate_func=smooth
        # ))
        self.wait(2)
        for i in range(3):
            axis = np.zeros(3)
            axis[i] = 1
            self.play(Homotopy(
                self.get_quaternion_homotopy(axis),
                surface,
                run_time=10,
            ))
        self.wait(5)

    def plane_to_cylinder(self, p):
        x, y, z = p
        R = self.R + z
        return np.array([
            R * np.cos(x / self.R - 0),
            R * np.sin(x / self.R - 0),
            1.0 * y,
        ])

    def cylinder_to_sphere(self, p):
        x, y, z = p
        R = self.R
        r = np.sqrt(R**2 - z**2)
        return np.array([
            x * fdiv(r, R),
            y * fdiv(r, R),
            z
        ])

    def get_quaternion_homotopy(self, axis=[1, 0, 0]):
        def result(x, y, z, t):
            alpha = t
            quaternion = np.array([np.cos(TAU * alpha), 0, 0, 0])
            quaternion[1:] = np.sin(TAU * alpha) * np.array(axis)
            new_quat = q_mult(quaternion, [0, x, y, z])
            return new_quat[1:]
        return result
