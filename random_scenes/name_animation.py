#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from imports_3b1b import *
from from_3b1b.active.diffyq.part2.fourier_series import FourierOfTexPaths

NAME_WITH_SPACES = "Prime Meridian"


class NameAnimationScene(Scene):
    CONFIG = {
        "animated_name": "Test name",
        "all_names": [
            "William Wayne Smith",
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
        text_mob = TextMobject(name)
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
            TexMobject("\\times"),
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

        thank_you = TextMobject("Thank You!").next_to(randy, DOWN)
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
        text = TextMobject(self.animated_name)[0]
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

            label = VGroup(TexMobject("\\times"), Integer(k))
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
    CONFIG = {
        "camera_class": MovingCamera
    }
