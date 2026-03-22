from manim_imports_ext import *
from _2025.laplace.derivatives import tex_to_color


def get_lt_group(src, trg, arrow_length=1.5, arrow_thickness=4, buff=MED_SMALL_BUFF, label_font_size=48):
    arrow = Vector(arrow_length * RIGHT, thickness=arrow_thickness)
    arrow.next_to(src, RIGHT, buff=buff)
    trg.next_to(arrow, RIGHT, buff=buff)

    label = Tex(R"\mathcal{L}", font_size=label_font_size)
    label.next_to(arrow, UP, buff=SMALL_BUFF)

    return VGroup(src, arrow, label, trg)


class AnnotateIntro(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph(height=1.0)
        left_section = Rectangle(5.25, 1.5).to_edge(LEFT)
        right_section = Rectangle(5, 1.3).next_to(left_section, RIGHT, buff=0)
        left_brace = Brace(left_section, UP)
        right_brace = Brace(right_section, RIGHT)

        q_marks = Tex(R"???", font_size=36)
        length_question = VGroup(Vector(LEFT), q_marks, Vector(RIGHT))
        length_question.arrange(RIGHT)
        length_question.match_width(left_section)
        length_question.move_to(left_brace, DOWN)

        randy.next_to(left_brace, UP, buff=0.25)

        self.play(
            randy.change("maybe", look_at=left_section),
            GrowFromCenter(left_brace)
        )
        self.play(Blink(randy))
        self.wait()
        self.play(
            randy.change("hesitant", look_at=left_section),
            FadeTransform(left_brace, length_question)
        )
        self.play(Blink(randy))
        self.wait()

        # Move to right section
        arrows = length_question[0::2]
        right_brace.save_state()
        right_brace.rotate(90 * DEG)
        right_brace.replace(arrows, stretch=True)
        right_brace.set_opacity(0)

        kw = dict(path_arc=-30 * DEG, run_time=2)
        self.play(
            Restore(right_brace, **kw),
            arrows.animate.rotate(90 * DEG).set_opacity(0).replace(right_brace.saved_state, stretch=True).set_anim_args(**kw),
            q_marks.animate.next_to(right_brace.saved_state, RIGHT).set_anim_args(**kw),
            randy.change("pondering", right_section),
        )

        h_lines = DashedLine(ORIGIN, 5 * LEFT).replicate(2)
        h_lines.set_stroke(WHITE, 1)
        h_lines[0].next_to(right_brace.get_corner(UL), LEFT, buff=0)
        h_lines[1].next_to(right_brace.get_corner(DL), LEFT, buff=0)
        self.play(*map(ShowCreation, h_lines))


class SimpleEToST(InteractiveScene):
    def construct(self):
        tex = Tex(R"e^{st}", t2c={"s": YELLOW, "t": GREY_A}, font_size=120)
        self.add(tex)


class MovingBrace(InteractiveScene):
    def construct(self):
        squares = Square().replicate(3)
        squares.arrange(RIGHT)
        squares.set_width(FRAME_WIDTH - 1)
        brace = Brace(squares[0], UP)
        self.play(GrowFromCenter(brace))
        self.wait()
        self.play(brace.animate.match_x(squares[1]))
        self.wait()


class TitleCard(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle().set_fill(GREY_E, 1))
        title = Text("Laplace Transform ", font_size=72)
        tex = Tex(
            R"F({s}) = \int^\infty_0 f({t})e^{\minus{s}{t}} d{t}",
            t2c=tex_to_color,
            font_size=60
        )
        # VGroup(title, tex).arrange(RIGHT, buff=MED_LARGE_BUFF).to_edge(UP)
        title.center().to_edge(UP)

        self.play(Write(title))
        # self.play(LaggedStart(
        #     FadeIn(title, 0.5 * LEFT, lag_ratio=0.01),
        #     FadeIn(tex, lag_ratio=0.1),
        #     lag_ratio=0.2
        # ))
        self.wait()


class QuickRecap(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer().flip()
        morty.to_edge(DOWN, buff=1.0).shift(LEFT)
        self.play(morty.says(Text("Quick Recap", font_size=72), bubble_direction=LEFT))
        self.play(Blink(morty))
        self.wait(2)


class KeyProperties(InteractiveScene):
    def construct(self):
        # Add title
        title = Text("Key Properties", font_size=72)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        title.set_backstroke(BLACK, 3)
        underline = Underline(title, buff=-0.05)
        underline.scale(1.25)
        self.add(underline, title)

        # Create
        t2c = dict(tex_to_color)
        number_labels = VGroup(Tex(Rf"{n})", font_size=72) for n in range(1, 4))
        number_labels.arrange(DOWN, aligned_edge=LEFT, buff=1.5)
        number_labels.next_to(title, DOWN, LARGE_BUFF)
        number_labels.to_edge(LEFT)

        properties = VGroup(
            get_lt_group(
                Tex(R"e^{a{t}}", t2c=t2c, font_size=60),
                Tex(R"{1 \over {s} - a}", t2c=t2c)
            ),
            get_lt_group(
                Tex(R"a \cdot f({t}) + b \cdot g({t})", t2c=t2c),
                Tex(R"a \cdot F({s}) + b \cdot G({s})", t2c=t2c),
            ),
            get_lt_group(
                Tex(R"f'({t})", t2c=t2c),
                Tex(R"{s} F({s}) - f(0)", t2c=t2c),
            ),
        )
        exp_prop, lin_prop, deriv_prop = properties
        properties.scale(1.25)
        for num, prop in zip(number_labels, properties):
            prop.shift(num.get_right() + MED_SMALL_BUFF * RIGHT - prop[0].get_left())
        exp_prop.shift(SMALL_BUFF * UP)

        # Show first properties
        self.play(
            LaggedStartMap(FadeIn, number_labels[:2], shift=UP, lag_ratio=0.25),
            ShowCreation(underline),
        )
        self.wait()

        self.play(Write(exp_prop[0]))
        self.play(LaggedStart(
            GrowArrow(exp_prop[1]),
            FadeIn(exp_prop[2], 0.25 * RIGHT),
            Transform(exp_prop[0]["a"][0].copy(), exp_prop[3]["a"][0].copy(), remover=True),
            Write(exp_prop[3]),
            lag_ratio=0.5
        ))
        self.wait()

        # Show linearity
        f_rects = VGroup(
            SurroundingRectangle(lin_prop[0]["f({t})"], buff=SMALL_BUFF),
            SurroundingRectangle(lin_prop[3]["F({s})"], buff=SMALL_BUFF),
        )
        g_rects = VGroup(
            SurroundingRectangle(lin_prop[0]["g({t})"], buff=SMALL_BUFF),
            SurroundingRectangle(lin_prop[3]["G({s})"], buff=SMALL_BUFF),
        )
        VGroup(f_rects, g_rects).set_stroke(TEAL, 2)

        self.play(
            Write(lin_prop[0])
        )
        self.play(LaggedStart(
            GrowArrow(lin_prop[1]),
            FadeIn(lin_prop[2], 0.25 * RIGHT),
            TransformMatchingTex(
                lin_prop[0].copy(),
                lin_prop[3],
                key_map={"{t}": "{s}", "f": "F", "g": "G"},
                path_arc=45 * DEG,
                lag_ratio=0.01,
            )
        ))
        self.wait()
        self.play(ShowCreation(f_rects, lag_ratio=0))
        self.wait()
        self.play(ReplacementTransform(f_rects, g_rects, lag_ratio=0))
        self.wait()
        self.play(FadeOut(g_rects))

        # (Edited in, show combination transformed)

        # Show third property
        frame = self.frame
        morty = Mortimer().flip()
        morty.next_to(number_labels[2], DR, LARGE_BUFF)

        self.play(
            VFadeIn(morty),
            morty.change("raise_left_hand", number_labels[2]),
            properties[:2].animate.set_fill(opacity=0.5),
            number_labels[:2].animate.set_fill(opacity=0.5),
            Write(number_labels[2]),
            frame.animate.set_height(12, about_edge=UP)
        )
        self.wait()
        self.play(LaggedStart(
            Write(deriv_prop[0]),
            GrowArrow(deriv_prop[1]),
            FadeIn(deriv_prop[2], 0.25 * RIGHT),
            morty.change("pondering", deriv_prop[0]),
        ))
        self.play(Blink(morty))
        self.wait()
        morty.body.insert_n_curves(500)
        self.play(
            Write(deriv_prop[3]),
            morty.change("raise_right_hand", deriv_prop[3])
        )
        self.play(Blink(morty))
        self.wait()

        # TODO


class SimpleLTArrow(InteractiveScene):
    def insertion(self):
        # Test
        group = get_lt_group(VGroup(), VGroup(), arrow_length=3, arrow_thickness=5)
        group.scale(1.5).center()
        self.play(
            GrowArrow(group[1]),
            FadeIn(group[2], RIGHT),
        )
        self.wait()


class CombinationOfExponentials(InteractiveScene):
    def construct(self):
        # Test
        t2c = dict(tex_to_color)
        t2c["c_n"] = TEAL
        kw = dict(t2c=t2c, font_size=72)
        combination = Tex(R"\sum_n c_n e^{a_n{t}}", **kw)
        result = Tex(R"\sum_n {c_n \over {s} - a_n}", **kw)

        group = get_lt_group(
            combination,
            result,
            arrow_length=4,
            label_font_size=60,
            arrow_thickness=6
        )
        group.center()

        self.play(FadeIn(combination, lag_ratio=0.1))
        self.wait()
        self.play(LaggedStart(
            GrowArrow(group[1]),
            FadeIn(group[2], RIGHT),
            *(
                Transform(combination[tex][0].copy(), result[tex][0].copy(), remover=True)
                for tex in ["c_n", "a_n", R"\sum_n"]
            ),
            Write(result[R"\over {s} - "][0]),
            lag_ratio=0.2,
            run_time=2
        ))
        self.add(group)
        self.wait()


class SimpleFrameForExpDeriv(InteractiveScene):
    def construct(self):
        rect = Rectangle(6, 4)
        rect.set_stroke(BLUE, 3)

        self.play(ShowCreation(rect))
        self.wait()


class MortyReferencingTwoThings(InteractiveScene):
    def construct(self):
        morty = Mortimer().flip()
        morty.to_edge(DOWN)
        self.play(morty.change("raise_left_hand", 3 * UL))
        self.play(Blink(morty))
        self.play(morty.change("raise_right_hand", 3 * UR))
        for _ in range(2):
            self.play(Blink(morty))
            self.wait(2)
        self.play(morty.change("tease", 3 * UR))
        self.play(Blink(morty))
        self.wait(2)


class StepByStep(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(1000)

        self.background.fade(0.5)

        self.play(
            morty.change("raise_right_hand"),
            self.change_students("confused", "maybe", "erm", look_at=self.screen)
        )
        self.wait()
        self.play(
            morty.change("tease"),
            self.change_students("pondering", "thinking", "pondering", look_at=morty.eyes)
        )
        self.wait()
        self.look_at(self.screen)
        self.wait(4)


class TryItAsAnExercise(InteractiveScene):
    def construct(self):
        morty = Mortimer()
        morty.body.insert_n_curves(500)
        self.play(morty.says("Try it as\nan exercise", mode="tease"))
        self.play(Blink(morty))
        self.wait()


class SimpleRect(InteractiveScene):
    def construct(self):
        rect = Rectangle(1.5, 1)
        rect.set_stroke(YELLOW, 3)
        self.play(ShowCreation(rect))
        self.wait()
        self.play(rect.animate.stretch(0.5, 1, about_edge=DOWN))
        self.wait()


class PolesAtOmegaI(InteractiveScene):
    def construct(self):
        t2c = {"{s}": YELLOW, R"\omega": PINK}
        pole_words = VGroup(
            Tex(R"\text{Pole at } {s} = +\omega i", t2c=t2c),
            Tex(R"\text{Pole at } {s} = -\omega i", t2c=t2c),
        )
        pole_words.arrange(DOWN, aligned_edge=LEFT)
        for words in pole_words:
            self.play(Write(words))
            self.wait()


class WhatIsTheAnswer(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            stds[1].says("Okay, okay, okay", mode="sassy"),
            stds[0].change("erm", stds[1].eyes),
            stds[2].change("hesitant", stds[1].eyes),
            morty.change("tease"),
        )
        self.wait(2)
        self.play(
            FadeOut(stds[1].bubble),
            stds[1].says("But...what is\nthe actual answer?", mode="maybe"),
            morty.change("guilty"),
        )
        self.wait()
        self.play(self.change_students("sassy", "erm", "confused", look_at=self.screen))
        self.wait(3)

        self.wait(2)
        self.play(
            FadeOut(stds[1].bubble),
            morty.change("raise_right_hand"),
            self.change_students("pondering", "pondering", "pondering", look_at=4 * UR)
        )
        self.wait(4)


class InvertArrow(InteractiveScene):
    def construct(self):
        # Add arrow
        arrow = Vector(1.5 * RIGHT, thickness=4)
        arrow.set_fill(border_width=2)
        label = Tex(R"\mathcal{L}", font_size=48)
        label.next_to(arrow, UP, SMALL_BUFF)
        inv_label = Tex(R"\mathcal{L}^{-1}")
        inv_label.next_to(arrow, DOWN, buff=0)

        self.play(
            GrowArrow(arrow),
            FadeIn(label, 0.5 * RIGHT)
        )
        self.wait()
        self.play(
            Rotate(arrow, PI),
            ReplacementTransform(label, inv_label)
        )
        self.wait()


class ReferenceHomework(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher

        self.play(
            morty.change("raise_right_hand"),
            self.change_students("tease", "pondering", "thinking", look_at=self.screen)
        )
        self.wait(3)
        self.play(morty.change("tease"))
        self.play(self.change_students("pondering", "hesitant", "pondering", look_at=self.screen))
        self.wait(4)


class ODEToAlgebra(InteractiveScene):
    def construct(self):
        # Test
        arrow_config = dict(arrow_length=2.5, buff=0.75)
        ode_group = get_lt_group(
            Text("Differential\nEquation"),
            Text("Algebra"),
            **arrow_config
        )
        ode_group.center()

        deriv_group = get_lt_group(
            Tex("d / dt", t2c={"t": BLUE}, font_size=72),
            Tex(R"\times s", t2c={"s": YELLOW}, font_size=72),
            **arrow_config
        )
        deriv_group.next_to(ode_group, DOWN, LARGE_BUFF)

        self.animate_group(ode_group)
        self.wait()
        self.animate_group(deriv_group)
        self.wait()

    def animate_group(self, group):
        src, arrow, label, trg = group
        self.play(FadeIn(src))
        self.play(LaggedStart(
            GrowArrow(arrow),
            FadeIn(label, 0.5 * arrow.get_vector()),
            FadeIn(trg, arrow.get_vector())
        ))


class ThreeExplanations(InteractiveScene):
    def construct(self):
        # Add terms
        t2c = {"{t}": BLUE, "{s}": YELLOW}
        title = Tex(R"\text{Why } \mathcal{L}\big\{f'({t})\big\} = {s} F({s}) - f(0)", t2c=t2c)
        title.to_edge(UP)
        title_underline = Underline(title)

        num_mobs = VGroup(Text(f"{n}) ") for n in range(1, 4))
        num_mobs.scale(1.25)
        num_mobs.arrange(DOWN, buff=1.25, aligned_edge=LEFT)
        num_mobs.next_to(title, DOWN, buff=1.25).to_edge(LEFT, buff=LARGE_BUFF)

        by_parts = Tex(R"\text{Integration by parts}", t2c=t2c)
        inversion = Tex(R"\frac{d}{d{t}} \Big(\text{Inverse Laplace Transform}\Big)", t2c=t2c)

        descriptors = VGroup(
            Text("[Elementary, but limited]"),
            Text("[General, but opaque]"),
            Text("[My favorite, but theoretical]"),
        )
        descriptors.set_fill(GREY_B)
        explanations = VGroup(
            Tex(R"\text{Start with } e^{a{t}}", t2c=t2c),
            by_parts,
            inversion,
        )

        for num, desc, expl in zip(num_mobs, descriptors, explanations):
            for mob in [desc, expl]:
                mob.next_to(num, RIGHT, buff=MED_LARGE_BUFF)
        explanations[0].shift(SMALL_BUFF * UP)

        self.add(title)
        self.play(ShowCreation(title_underline))
        self.wait()
        self.play(LaggedStartMap(FadeIn, num_mobs, shift=UP, lag_ratio=0.25))
        self.wait()
        for desc in descriptors:
            self.play(FadeIn(desc, lag_ratio=0.2))
            self.wait()

        # Show explanations
        for i, desc, expl in zip(it.count(), descriptors, explanations):
            self.play(
                FadeOut(desc, 0.5 * UP),
                FadeIn(expl, 0.5 * UP),
                num_mobs[:i].animate.set_opacity(0.25),
                explanations[:i].animate.set_opacity(0.25),
            )
            self.wait()

        # Show all of them
        num_mobs.set_fill(opacity=1)
        explanations.set_fill(opacity=1)
        self.play(
            LaggedStartMap(FadeIn, num_mobs, shift=UP, lag_ratio=0.25),
            LaggedStartMap(FadeIn, explanations, shift=UP, lag_ratio=0.25),
        )
        self.wait()
        fade_group = VGroup(title, title_underline, *num_mobs, *explanations)
        fade_group.sort(lambda p: np.dot(p, DOWN + 0.1 * RIGHT))
        self.play(LaggedStartMap(FadeOut, fade_group, shift=LEFT, lag_ratio=0.15, run_time=2))
        self.wait()


class ComplainAboutSpecificity(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            stds[1].says("But, that’s just\none example!", mode="angry"),
            stds[0].change("pondering", self.screen),
            stds[2].change("erm", self.screen),
            morty.change("guilty")
        )
        self.wait(3)
        self.play(
            stds[1].debubble("pondering"),
            morty.says("One with the\nseeds of generality"),
            stds[2].change("thinking")
        )
        self.wait(3)


class ExplanationOneTitle(InteractiveScene):
    def construct(self):
        t2c = dict(tex_to_color)
        title = VGroup(
            Text("Explanation #1: Try it for"),
            Tex(R"e^{a{t}}", t2c=t2c)
        )
        title.arrange(RIGHT)
        title[1].shift(SMALL_BUFF * UP)
        title.to_corner(UL)
        alt_formula = Tex(R"c_1 e^{a_1 {t}} + c_2 e^{a_2 {t}} + \cdots + c_n e^{a_n {t}}", t2c=t2c)
        alt_formula[re.compile(r"c_.")].set_color(TEAL_D)
        alt_formula.move_to(title[1], LEFT)
        ponder_words = Text("(Pause and ponder to taste)")
        ponder_words.set_fill(GREY_B)
        ponder_words.move_to(title)
        ponder_words.to_edge(RIGHT, buff=SMALL_BUFF)

        self.add(title)
        self.play(Write(ponder_words))
        self.wait()
        self.play(
            TransformMatchingTex(title[1], alt_formula),
            FadeOut(ponder_words, lag_ratio=0.1),
        )
        self.wait()


class IntuitionEvaporating(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(1000)

        words = Text("Intuition", font_size=36).replicate(12)
        words.move_to(stds.get_top() + 0.5 * DOWN)
        for word in words:
            word.shift(random.uniform(-4, 4) * RIGHT)

        self.play(
            morty.change("raise_right_hand"),
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen),
        )
        self.wait()
        y0 = words.get_y()
        self.play(
            self.change_students("tired", "tired", "tired", look_at=self.screen),
            morty.change("guilty"),
            LaggedStart(
                *(
                    UpdateFromAlphaFunc(word, lambda m, a: m.set_y(y0 + 2 * a).set_fill(opacity=there_and_back(a)))
                    for word in words
                ),
                lag_ratio=0.1,
                run_time=6
            ),
        )


class ContourIntegralReference(InteractiveScene):
    def construct(self):
        # Test
        inv_words = Text("Inverse Laplace Transfor")
        inv_rect = SurroundingRectangle(inv_words)
        inv_rect.set_fill(BLACK, 1)
        inv_rect.set_stroke(width=0)

        contour_words = Text("Contour Integral")
        contour_words.move_to(inv_rect)

        integral = Tex(R"\int_\gamma")
        integral.next_to(contour_words, UP, MED_LARGE_BUFF)
        integral.shift(0.25 * RIGHT)
        integral_rect = SurroundingRectangle(integral)
        integral_rect.set_stroke(YELLOW, 2)

        self.add(inv_rect)
        self.play(
            Write(contour_words),
            ShowCreation(integral_rect),
            VShowPassingFlash(integral_rect.copy().insert_n_curves(100).set_stroke(YELLOW, 3)),
            run_time=2
        )
        self.wait()


class SimpleBigRect(InteractiveScene):
    def construct(self):
        rect = Rectangle(4, 5)
        rect.set_stroke(YELLOW, 3)
        self.play(ShowCreation(rect))
        self.wait()