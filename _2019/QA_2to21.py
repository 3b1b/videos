from manim_imports_ext import *

from _2017.mug import HappyHolidays


# For Q&A Video
class Questions(Scene):
    def construct(self):
        kw = {
            "alignment": ""
        }
        TexText.CONFIG.update(kw)
        questions = VGroup(
            TexText(
                "Who is your favorite mathematician?"
            ),
            TexText(
                "A teenage kid walks up to you and says they\\\\",
                "hate maths. What do you tell/show them?"
            ),
            TexText(
                "What advice would you want to give to give a\\\\",
                "math enthusiast suffering from an anxiety\\\\",
                "disorder, clinical depression and ADHD?",
            ),
            TexText(
                "Is Ben, Ben and Blue still a thing?"
            ),
            TexText(
                "Favorite podcasts?"
            ),
            TexText(
                "Hey Grant, if you had, both, the responsibility and\\\\",
                "opportunity to best introduce the world of mathematics\\\\",
                "to curious and intelligent minds before they are shaped\\\\",
                "by the antiquated, disempowering and demotivational\\\\",
                "education system of today, what would you do?  (Asking\\\\",
                "because I will soon be a father).\\\\",
            ),
            TexText(
                "What's something you think could've\\\\",
                "been discovered long before it was\\\\",
                "actually discovered?",
            ),
            TexText(
                "Can we fix math on Wikipedia? Really serious\\\\",
                "here. I constantly go there after your vids for\\\\",
                "a bit of deeper dive and learn - nothing more, ever.\\\\",
                "Compared to almost any topic in the natural sciences\\\\",
                "or physics where at least I get an outline of where\\\\",
                "to go next. It's such a shame."
            ),
        )

        last_question = VMobject()
        for question in questions:
            question.set_width(FRAME_WIDTH - 1)
            self.play(
                FadeInFromDown(question),
                FadeOut(last_question, UP)
            )
            last_question = question
            self.wait(2)


class MathematicianPlusX(Scene):
    def construct(self):
        text = TexText(
            "Side note:\\\\",
            "``The Mathematician + X''\\\\",
            "would make a great band name.",
        )
        text.set_width(FRAME_WIDTH - 1)
        self.add(text)


class NoClearCutPath(Scene):
    def construct(self):
        path1 = VMobject()
        path1.start_new_path(3 * LEFT)
        path1.add_line_to(ORIGIN)
        path1.append_points([
            ORIGIN,
            2.5 * RIGHT,
            2.5 * RIGHT + 3 * UP,
            5 * RIGHT + 3 * UP,
        ])
        path2 = path1.copy()
        path2.rotate(PI, axis=RIGHT, about_point=ORIGIN)
        paths = VGroup(path1, path2)
        paths.to_edge(LEFT)
        paths.set_stroke(WHITE, 2)

        labels = VGroup(
            TexText("Pure mathematicians"),
            TexText("Applied mathematicians"),
        )
        for label, path in zip(labels, paths):
            label.next_to(path.get_end(), RIGHT)

        animations = []
        n_animations = 20
        colors = [BLUE_C, BLUE_D, BLUE_E, GREY_BROWN]
        for x in range(n_animations):
            dot = self.get_dot(random.choice(colors))
            path = random.choice(paths)
            dot.move_to(path.get_start())
            anim = Succession(
                # FadeIn(dot),
                MoveAlongPath(dot, path, run_time=4, rate_func=lambda t: smooth(t, 2)),
                FadeOut(dot),
            )
            animations.append(anim)

        alt_path = VMobject()
        alt_path.start_new_path(paths.get_left())
        alt_path.add_line_to(paths.get_left() + 3 * RIGHT)
        alt_path.add_line_to(2 * UP)
        alt_path.add_line_to(2 * DOWN + RIGHT)
        alt_path.add_line_to(2 * RIGHT)
        alt_path.add_line_to(3 * RIGHT)
        alt_path.add_line_to(8 * RIGHT)
        alt_path.make_smooth()
        alt_path.set_stroke(YELLOW, 3)
        dashed_path = DashedVMobject(alt_path, num_dashes=100)

        alt_dot = self.get_dot(YELLOW)
        alt_dot.move_to(alt_path.get_start())
        alt_dot_anim = MoveAlongPath(
            alt_dot,
            alt_path,
            run_time=10,
            rate_func=linear,
        )

        self.add(paths)
        self.add(labels)
        self.play(
            LaggedStart(
                *animations,
                run_time=10,
                lag_ratio=1 / n_animations,
            ),
            ShowCreation(
                dashed_path,
                rate_func=linear,
                run_time=10,
            ),
            alt_dot_anim,
        )
        self.wait()

    def get_dot(self, color):
        dot = Dot()
        dot.scale(1.5)
        dot.set_color(color)
        dot.set_stroke(BLACK, 2, background=True)
        return dot


class Cumulative(Scene):
    def construct(self):
        colors = list(Color(BLUE_B).range_to(BLUE_D, 20))
        rects = VGroup(*[
            Rectangle(
                height=0.3, width=4,
                fill_color=random.choice(colors),
                fill_opacity=1,
                stroke_color=WHITE,
                stroke_width=1,
            )
            for i, color in zip(range(20), it.cycle(colors))
        ])
        rects.arrange(UP, buff=0)

        check = Tex("\\checkmark").set_color(GREEN)
        cross = Tex("\\times").set_color(RED)
        checks, crosses = [
            VGroup(*[
                mob.copy().next_to(rect, RIGHT, SMALL_BUFF)
                for rect in rects
            ])
            for mob in [check, cross]
        ]

        rects.set_fill(opacity=0)
        self.add(rects)

        for i in range(7):
            self.play(
                rects[i].set_fill, None, 1.0,
                Write(checks[i]),
                run_time=0.5,
            )
        self.play(FadeOut(rects[7], 2 * LEFT))
        self.play(
            LaggedStartMap(Write, crosses[8:]),
            LaggedStartMap(
                ApplyMethod, rects[8:],
                lambda m: (m.set_fill, None, 0.2),
            )
        )
        self.wait()
        rects[7].set_opacity(1)
        self.play(
            FadeIn(rects[7], 2 * LEFT),
            FadeIn(checks[7], 2 * LEFT),
        )
        self.play(
            LaggedStartMap(
                ApplyMethod, rects[8:],
                lambda m: (m.set_fill, None, 1.0),
            ),
            LaggedStart(*[
                ReplacementTransform(cross, check)
                for cross, check in zip(crosses[8:], checks[8:])
            ])
        )
        self.wait()


class HolidayStorePromotionTime(HappyHolidays):
    def construct(self):
        title = TexText("Holiday store promotion time!")
        title.set_width(FRAME_WIDTH - 1)
        title.to_edge(UP)
        self.add(title)

        HappyHolidays.construct(self)
