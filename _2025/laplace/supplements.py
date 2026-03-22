from manim_imports_ext import *


class IntroduceTrilogy(InteractiveScene):
    def construct(self):
        # Add definition
        self.add(FullScreenRectangle().fix_in_frame())
        frame = self.frame
        name = Text("Laplace Transform", font_size=60)
        name.to_edge(UP)
        t2c = {"s": YELLOW, R"{t}": BLUE}
        laplace = Tex(R"F(s) = \int_0^\infty f({t}) e^{\minus s{t}} d{t}", font_size=36, t2c=t2c)
        laplace.next_to(name, DOWN)

        frames = Square().replicate(3)
        frames.set_stroke(WHITE, 1).set_fill(BLACK, 1)
        frames.set_width(0.3 * FRAME_WIDTH)
        frames.arrange(RIGHT, buff=MED_LARGE_BUFF)
        frames.set_y(-1.0)
        frames.fix_in_frame()
        name.fix_in_frame()

        frame.match_x(laplace["f({t})"])

        self.play(
            Write(name),
            FadeIn(frames, lag_ratio=0.25, run_time=2)
        )
        self.play(
            Write(laplace["f({t})"]),
        )
        self.play(
            Write(laplace[R"e^{\minus s"]),
            TransformFromCopy(*laplace[R"{t}"][0:2]),
            frame.animate.match_x(laplace[R"f({t}) e^{\minus s{t}}"])
        )
        self.play(
            FadeIn(laplace[R"\int_0^\infty"], shift=0.25 * RIGHT, scale=1.5),
            FadeIn(laplace[R"d{t}"], shift=0.25 * LEFT, scale=1.5),
        )
        self.play(
            FadeTransform(laplace["f("].copy(), laplace["F("], path_arc=-PI / 2),
            TransformFromCopy(laplace[")"][1], laplace[")"][0], path_arc=-PI / 2),
            TransformFromCopy(laplace["s"][1], laplace["s"][0], path_arc=-PI / 4),
            Write(laplace["="]),
            frame.animate.center(),
        )
        self.wait()

        # Contrast the pair
        brace = Brace(frames[1:], UP)
        brace_ghost = brace.copy().set_fill(GREY_D)

        ilp = Tex(R"f({t}) = \frac{1}{2\pi i} \int_{a - i \infty}^{a + i \infty} F(s) e^{s{t}} d{s}", t2c=t2c, font_size=36)
        ilp.scale(0.75)
        ilp.next_to(frames[2], UP, MED_LARGE_BUFF)

        self.play(
            GrowFromCenter(brace),
            laplace.animate.scale(0.75).next_to(frames[1], UP, MED_LARGE_BUFF),
            name.animate.scale(0.75).next_to(frames[1:], UP, buff=1.75),
        )
        self.play(TransformMatchingTex(
            laplace.copy(), ilp,
            lag_ratio=1e-2,
            path_arc=-20 * DEG,
            matched_keys=["f({t})", "F(s)", "e^{s{t}}", R"\int"],
            key_map={"d{t}": "d{s}"},
        ))
        self.wait()
        self.add(brace_ghost, brace)
        self.play(
            brace.animate.match_width(frames[0], stretch=True).next_to(frames[0], UP).set_anim_args(path_arc=15 * DEG)
        )
        self.wait()

        # Clear the board
        self.play(
            LaggedStartMap(FadeOut, VGroup(*frames[1:], brace, brace_ghost, name, laplace, ilp), shift=RIGHT, lag_ratio=0.15),
            frames[0].animate.set_shape(16, 9).set_width(0.45 * FRAME_WIDTH).move_to(FRAME_WIDTH * LEFT / 4 + 0.5 * DOWN),
            run_time=2
        )
        self.wait()

        # Add new frame
        new_frame = frames[0].copy()
        new_frame.set_x(FRAME_WIDTH / 4)
        arc = -120 * DEG
        arrow = Arrow(frames[0].get_top(), new_frame.get_top(), path_arc=arc, thickness=6)
        arrow_line = Line(frames[0].get_top(), new_frame.get_top(), path_arc=arc, buff=0.2)
        arrow_line.pointwise_become_partial(arrow_line, 0, 0.95)
        arrow.set_fill(TEAL)
        arrow_line.set_stroke(TEAL, 10)
        self.play(
            ShowCreation(arrow_line, time_span=(0, 0.8)),
            FadeIn(arrow, time_span=(0.7, 1)),
            FadeIn(new_frame)
        )

    def old_material(self):
        # Show trilogy
        background = FullScreenRectangle().set_fill(GREY_E, 1)
        screens = ScreenRectangle().replicate(3)
        screens.set_fill(BLACK, 1)
        screens.set_stroke(WHITE, 2)
        screens.set_width(0.3 * FRAME_WIDTH)
        screens.arrange(RIGHT, buff=0.25 * (FRAME_WIDTH - 3 * screens[0].get_width()))
        screens.set_width(FRAME_WIDTH - 1)
        screens.to_edge(UP)

        screens[1].save_state()
        screens[1].replace(background)
        screens[1].set_stroke(width=0)
        screens.set_stroke(behind=True)

        terms = VGroup(name, laplace)

        self.add(background, screens, terms)
        self.play(
            FadeIn(background),
            Restore(screens[1]),
            terms.animate.scale(0.4).move_to(screens[1].saved_state),
        )
        self.wait()

        # Inverse
        ilp = VGroup(
            Text("Inverse Laplace Transform"),
            Tex(R"f({t}) = \frac{1}{2\pi i} \int_{a - i \infty}^{a + i \infty} F(s) e^{s{t}} ds", t2c={"s": YELLOW})
        )
        for mob1, mob2 in zip(ilp, terms):
            mob1.replace(mob2, dim_to_match=1)

        ilp.move_to(screens[2])

        self.play(
            TransformMatchingStrings(name.copy(), ilp[0], lag_ratio=1e-2, path_arc=-20 * DEG),
            TransformMatchingTex(
                laplace.copy(),
                ilp[1],
                lag_ratio=1e-2,
                path_arc=-20 * DEG,
                matched_keys=["f({t})", "F(s)", "e^{s{t}}", R"\int"],
                key_map={"dt": "ds"},
            ),
        )
        self.wait()


class DiscussTrilogy(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        screens = ScreenRectangle().replicate(3)
        screens.set_width(0.3 * FRAME_WIDTH)
        screens.arrange(RIGHT, buff=0.25 * (FRAME_WIDTH - 3 * screens[0].get_width()))
        screens.set_width(FRAME_WIDTH - 1)
        screens.to_edge(UP)

        # Reference last two
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(200)
        morty = self.teacher
        morty.change_mode("tease")
        brace1 = Brace(screens[0], DOWN)
        brace2 = Brace(screens[1:3], DOWN)

        self.wait(2)
        self.play(
            morty.change("raise_left_hand", look_at=brace2),
            self.change_students("pondering", "erm", "sassy", look_at=brace2),
            GrowFromCenter(brace2),
        )
        self.wait(3)
        self.play(
            morty.change("raise_right_hand"),
            self.change_students("thinking", "pondering", "pondering", look_at=brace1),
            ReplacementTransform(brace2, brace1, path_arc=-30 * DEG),
        )
        self.wait(6)


class WhoCares(TeacherStudentsScene):
    def construct(self):
        # Test
        self.remove(self.background)
        stds = self.students
        morty = self.teacher

        self.play(
            stds[2].says("Who cares?", mode="angry", look_at=3 * UP),
            morty.change("guilty", stds[2].eyes),
            stds[1].change("hesitant", 3 * UP),
            stds[0].change("erm", stds[2].eyes),
        )
        self.wait(3)


class MiniLessonTitle(InteractiveScene):
    def construct(self):
        title = Text("Visualizing complex exponents", font_size=72)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()


class WeGotThis(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        self.play(
            self.change_students("coin_flip_2", "tease", "hooray", look_at=3 * UP),
            morty.change("tease")
        )
        self.wait()
        self.play(
            self.change_students("tease", "happy", "well", look_at=morty.eyes)
        )
        self.wait(3)


class ConfusionAndWhy(TeacherStudentsScene):
    def construct(self):
        # Test
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(200)
        morty = self.teacher
        stds = self.students

        q_marks = Tex(R"???").replicate(3)
        q_marks.space_out_submobjects(1.5)
        for mark, student in zip(q_marks, stds):
            mark.next_to(student, UP, MED_SMALL_BUFF)
        self.play(
            self.change_students("confused", "pondering", "pleading", look_at=self.screen),
            FadeIn(q_marks, 0.2 * UP, lag_ratio=0.05),
            morty.change("raise_right_hand")
        )
        self.wait(3)
        self.play(morty.change("raise_left_hand", look_at=3 * UR))
        self.play(
            self.change_students("erm", "thinking", "hesitant", look_at=morty.get_top() + 2 * UP),
            FadeOut(q_marks)
        )
        self.wait(4)
        self.play(self.change_students("pondering"))
        self.wait(3)


class ArrowBetweenScreens(InteractiveScene):
    def construct(self):
        # Test
        screens = ScreenRectangle().replicate(2)
        screens.arrange(RIGHT, buff=MED_LARGE_BUFF)
        screens.set_width(FRAME_WIDTH - 2)
        screens.move_to(DOWN)
        arrow = Arrow(screens[0].get_top(), screens[1].get_top(), path_arc=-120 * DEG, thickness=6, buff=0.25)
        line = Line(screens[0].get_top(), screens[1].get_top(), path_arc=-120 * DEG, stroke_width=8, buff=0.25)
        VGroup(arrow, line).set_color(TEAL)
        self.play(
            ShowCreation(line),
            FadeIn(arrow, time_span=(0.75, 1))
        )
        self.wait()


class WhatAndWhy(InteractiveScene):
    def construct(self):
        words = VGroup(
            Tex(R"\text{1) Understanding } e^{i {t}} \\ \text{ intuitively}", t2c={R"{t}": GREY_B}),
            TexText(R"2) How they \\ \quad \quad naturally arise"),
        )
        words[0][R"intuitively"].align_to(words[0]["Understanding"], LEFT)
        words[1][R"naturally arise"].align_to(words[1]["How"], LEFT)
        words.refresh_bounding_box()
        words.scale(1.25)
        self.add(words)
        words.arrange(DOWN, aligned_edge=LEFT, buff=2.5)
        words.next_to(ORIGIN, RIGHT)
        words.set_opacity(0)
        for word, u in zip(words, [1, -1]):
            word.set_opacity(1)
            self.play(Write(word))
            self.wait()
        # Test


class PrequelToLaplace(InteractiveScene):
    def construct(self):
        # False goal of motivating the i
        pass

        # Swap out i and π for s and t


class OtherExponentialDerivatives(InteractiveScene):
    def construct(self):
        # Test
        kw = dict(t2c={"t": GREY_B})
        derivs = VGroup(
            Tex(R"\frac{d}{dt} 2^t = (0.693...)2^t", **kw),
            Tex(R"\frac{d}{dt} 3^t = (1.098...)3^t", **kw),
            Tex(R"\frac{d}{dt} 4^t = (1.386...)4^t", **kw),
            Tex(R"\frac{d}{dt} 5^t = (1.609...)5^t", **kw),
            Tex(R"\frac{d}{dt} 6^t = (1.791...)6^t", **kw),
        )
        derivs.scale(0.75)
        derivs.arrange(DOWN, buff=0.7)
        derivs.to_corner(UL)

        self.play(LaggedStartMap(FadeIn, derivs, shift=UP, lag_ratio=0.5, run_time=5))
        self.wait()


class VariousExponentials(InteractiveScene):
    def construct(self):
        # Test
        exp_st = Tex(R"e^{st}", t2c={"s": YELLOW, "t": BLUE}, font_size=90)
        gen_exp = Tex(R"e^{+0.50 t}", t2c={"+0.50": YELLOW, "t": BLUE}, font_size=90)
        exp_st.to_edge(UP, buff=MED_LARGE_BUFF)
        gen_exp.move_to(exp_st)

        num = gen_exp["+0.50"]
        num.set_opacity(0)
        gen_exp["t"].scale(1.25, about_edge=UL)

        s_num = DecimalNumber(-1.00, edge_to_fix=ORIGIN, include_sign=True)
        s_num.set_color(YELLOW)
        s_num.replace(num, dim_to_match=1)

        self.add(gen_exp, s_num)
        self.play(ChangeDecimalToValue(s_num, 0.5, run_time=4))
        self.wait()
        self.play(LaggedStart(
            ReplacementTransform(gen_exp["e"][0], exp_st["e"][0]),
            ReplacementTransform(s_num, exp_st["s"]),
            ReplacementTransform(gen_exp["t"][0], exp_st["t"][0]),
        ))
        self.wait()


class WhyToWhat(InteractiveScene):
    def construct(self):
        # Title text
        why = Text("Why", font_size=90)
        what = Text("Wait, what does this even mean?", font_size=72)
        VGroup(why, what).to_edge(UP)

        what_word = what["what"][0].copy()
        what["what"][0].set_opacity(0)

        arrow = Arrow(
            what["this"].get_bottom(),
            (2.5, 2, 0),
            thickness=5,
            fill_color=YELLOW
        )

        self.play(FadeIn(why, UP))
        self.wait()
        self.play(
            # FadeOut(why, UP),
            ReplacementTransform(why, what_word),
            FadeIn(what, lag_ratio=0.1),
        )
        self.play(
            GrowArrow(arrow),
            what["this"].animate.set_color(YELLOW)
        )
        self.wait()


class DerivativeOfExp(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        tex_kw = dict(t2c={"t": GREY_B, "s": YELLOW})
        equation = Tex(R"\frac{d}{dt} e^{st} = s \cdot e^{st}", font_size=90, **tex_kw)
        deriv_part = equation[R"\frac{d}{dt}"][0]
        exp_parts = equation[R"e^{st}"]
        equals = equation[R"="][0]
        s_dot = equation[R"s \cdot"][0]

        v_box = SurroundingRectangle(VGroup(deriv_part, exp_parts[0]))
        p_box = SurroundingRectangle(exp_parts[1])
        s_box = SurroundingRectangle(s_dot)
        s_box.match_height(p_box, stretch=True).match_y(p_box)
        boxes = VGroup(v_box, p_box, s_box)
        boxes.set_stroke(width=2)
        boxes.set_submobject_colors_by_gradient(GREEN, BLUE, YELLOW)

        v_label = Text("Velocity", font_size=48).match_color(v_box)
        p_label = Text("Position", font_size=48).match_color(p_box)
        s_label = Text("Modifier", font_size=48).match_color(s_box)
        v_label.next_to(v_box, UP, MED_SMALL_BUFF)
        p_label.next_to(p_box, UP, MED_SMALL_BUFF, aligned_edge=LEFT)
        s_label.next_to(s_box, DOWN, MED_SMALL_BUFF)
        labels = VGroup(v_label, p_label, s_label)

        frame.move_to(exp_parts[0])

        self.add(exp_parts[0])
        self.wait()
        self.play(Write(deriv_part))
        self.play(
            TransformFromCopy(*exp_parts, path_arc=90 * DEG),
            Write(equals),
            frame.animate.center(),
        )
        self.play(
            TransformFromCopy(exp_parts[1][1], s_dot[0], path_arc=90 * DEG),
            Write(s_dot[1]),
        )
        self.wait()

        # Show labels
        for box, label in zip(boxes, labels):
            self.play(ShowCreation(box), FadeIn(label))

        self.wait()
        full_group = VGroup(equation, boxes, labels)

        # Set s equal to 1
        s_eq_1 = Tex(R"s = 1", font_size=72, **tex_kw)
        simple_equation = Tex(R"\frac{d}{dt} e^{t} = e^{t}", font_size=72, **tex_kw)
        simple_equation.to_edge(UP).shift(2 * LEFT)
        s_eq_1.next_to(simple_equation, RIGHT, buff=2.5)
        arrow = Arrow(s_eq_1, simple_equation, thickness=5, buff=0.35).shift(0.05 * DOWN)

        self.play(
            Write(s_eq_1),
            GrowArrow(arrow),
            TransformMatchingTex(equation.copy(), simple_equation, run_time=1.5, lag_ratio=0.02),
            full_group.animate.shift(DOWN).scale(0.75).fade(0.15)
        )
        self.wait()


class HighlightRect(InteractiveScene):
    def construct(self):
        img = ImageMobject('/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2025/laplace/exponentials/DynamicExpIntuitionStill.png')
        img.set_height(FRAME_HEIGHT)
        self.add(img)

        # Rects
        rects = VGroup(
            Rectangle(2.25, 1).move_to((2.18, 2.74, 0)),
            Rectangle(2, 0.85).move_to((-5.88, -2.2, 0.0)),
        )
        rects.set_stroke(YELLOW, 2)

        self.play(ShowCreation(rects[0]))
        self.play(TransformFromCopy(*rects))
        self.play(FadeOut(rects))


class DefineI(InteractiveScene):
    def construct(self):
        eq = Tex(R"i = \sqrt{-1}", t2c={"i": YELLOW}, font_size=90)
        self.play(Write(eq))
        self.wait()


class WaitWhy(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.students[0].change("erm", self.screen),
            self.students[1].change("tease", self.screen),
            self.students[2].says("Wait, why?", "confused", look_at=self.screen, bubble_direction=LEFT),
        )
        self.wait(4)


class MultiplicationByI(InteractiveScene):
    def construct(self):
        # Example number
        plane = ComplexPlane(
            background_line_style=dict(stroke_color=BLUE, stroke_width=1),
            # faded_line_style=dict(stroke_color=BLUE, stroke_width=0.5, stroke_opacity=0.5),
        )
        plane.add_coordinate_labels(font_size=24)

        z = 3 + 2j
        tex_kw = dict(t2c={"a": YELLOW, "b": PINK})

        vect = Vector(plane.n2p(z), fill_color=WHITE, thickness=4)
        vect_label = Tex(R"a + bi", **tex_kw)
        vect_label.next_to(vect.get_end(), UR, SMALL_BUFF)
        vect_label.set_backstroke(BLACK, 5)

        lines = VGroup(
            Line(ORIGIN, plane.n2p(z.real)).set_color(YELLOW),
            Line(plane.n2p(z.real), plane.n2p(z)).set_color(PINK),
        )
        a_label, b_label = line_labels = VGroup(
            Tex(R"a", font_size=36, **tex_kw).next_to(lines[0], UP, SMALL_BUFF),
            Tex(R"bi", font_size=36, **tex_kw).next_to(lines[1], RIGHT, SMALL_BUFF),
        )
        line_labels.set_backstroke(BLACK, 5)

        self.add(plane, Point(), plane.coordinate_labels)
        self.add(vect)
        self.add(vect_label)
        for line, label in zip(lines, line_labels):
            self.play(
                ShowCreation(line),
                FadeIn(label, 0.25 * line.get_vector())
            )
        self.wait()

        # Multiply components by i
        new_lines = lines.copy()
        new_lines.rotate(90 * DEG, about_point=ORIGIN)
        new_lines[1].move_to(ORIGIN, RIGHT)

        new_a_label = Tex(R"ai", font_size=36, **tex_kw).next_to(new_lines[0], RIGHT, SMALL_BUFF)
        new_b_label = Tex(R"bi \cdot i", font_size=36, **tex_kw).next_to(new_lines[1], UP, SMALL_BUFF)
        neg_b_label = Tex(R"=-b", font_size=36, **tex_kw)
        neg_b_label.move_to(new_b_label.get_right())

        mult_i_label = Tex(R"\times i", font_size=90)
        mult_i_label.set_backstroke(BLACK, 5)
        mult_i_label.to_corner(UR, buff=MED_LARGE_BUFF).shift(0.2 * UP)

        self.play(Write(mult_i_label))
        self.wait()
        self.play(
            TransformFromCopy(lines[0], new_lines[0], path_arc=90 * DEG),
            TransformFromCopy(a_label[0], new_a_label[0], path_arc=90 * DEG),
            TransformFromCopy(mult_i_label[1], new_a_label[1]),
        )
        self.wait()
        self.play(
            TransformFromCopy(lines[1], new_lines[1], path_arc=90 * DEG),
            TransformFromCopy(b_label[0], new_b_label[:-1], path_arc=90 * DEG),
            TransformFromCopy(mult_i_label[1], new_b_label[-1]),
        )
        self.wait()
        self.play(
            FlashAround(VGroup(new_b_label, new_lines[1]), color=PINK, time_width=1.5, run_time=2),
            new_b_label.animate.next_to(neg_b_label, LEFT, SMALL_BUFF),
            FadeIn(neg_b_label, SMALL_BUFF * RIGHT),
        )
        self.wait()
        self.play(VGroup(new_lines[1], new_b_label, neg_b_label).animate.shift(new_lines[0].get_vector()))

        # New vector
        vect_copy = vect.copy()
        elbow = Elbow().rotate(vect.get_angle(), about_point=ORIGIN)
        self.play(
            Rotate(vect_copy, 90 * DEG, run_time=2, about_point=ORIGIN),
        )
        self.play(
            ShowCreation(elbow)
        )
        self.wait()

    def old_material(self):
        # Show the algebra
        algebra = VGroup(
            Tex(R"i \cdot (a + bi)", **tex_kw),
            Tex(R"ai + bi^2", **tex_kw),
            Tex(R"-b + ai", **tex_kw),
        )
        algebra.set_backstroke(BLACK, 8)
        algebra.arrange(DOWN, buff=0.35)
        algebra.to_corner(UL)

        self.play(
            TransformFromCopy(vect_label, algebra[0]["a + bi"][0]),
            FadeIn(algebra[0]),
        )
        self.play(LaggedStart(
            TransformFromCopy(algebra[0]["a"], algebra[1]["a"]),
            TransformFromCopy(algebra[0]["+ bi"], algebra[1]["+ bi"]),
            TransformFromCopy(algebra[0]["i"][0], algebra[1]["i"][0]),
            TransformFromCopy(algebra[0]["i"][0], algebra[1]["2"]),
            lag_ratio=0.25
        ))
        self.wait()
        self.play(LaggedStart(
            TransformFromCopy(algebra[1]["bi^2"], algebra[2]["-b"]),
            TransformFromCopy(algebra[1]["ai"], algebra[2]["ai"]),
            TransformFromCopy(algebra[1]["+"], algebra[2]["+"]),
            lag_ratio=0.25
        ))
        self.wait()

        # New lines
        new_lines = lines.copy()
        new_lines.rotate(90 * DEG)
        new_lines.refresh_bounding_box()
        new_lines[1].move_to(ORIGIN, RIGHT)
        new_lines[0].move_to(new_lines[1].get_left(), DOWN)

        neg_b_label = Tex(R"-b", fill_color=PINK, font_size=36).next_to(new_lines[1], UP, SMALL_BUFF)
        new_a_label = Tex(R"a", fill_color=YELLOW, font_size=36).next_to(new_lines[0], LEFT, SMALL_BUFF)

        self.play(
            TransformFromCopy(lines[1], new_lines[1]),
            FadeTransform(algebra[2]["-b"].copy(), neg_b_label),
        )
        self.play(
            TransformFromCopy(lines[0], new_lines[0]),
            FadeTransform(algebra[2]["a"].copy(), new_a_label),
        )
        self.wait()


class UnitArcLengthsOnCircle(InteractiveScene):
    def construct(self):
        # Moving sectors
        arc = Arc(0, 1, radius=2.5, stroke_color=GREEN, stroke_width=8)
        sector = Sector(angle=1, radius=2.5).set_fill(GREEN, 0.25)
        v_line = Line(ORIGIN, 2.5 * UP)
        v_line.match_style(arc)
        v_line.move_to(arc.get_start(), DOWN)

        self.add(v_line)
        self.play(
            FadeIn(sector),
            ReplacementTransform(v_line, arc),
        )

        group = VGroup(sector, arc)
        self.add(group)

        for n in range(5):
            self.wait(2)
            group.rotate(1, about_point=ORIGIN)

        return

        # Previous
        colors = [RED, BLUE]
        arcs = VGroup(
            Arc(n, 1, radius=2.5, stroke_color=colors[n % 2], stroke_width=8)
            for n in range(6)
        )
        for arc in arcs:
            one = Integer(1, font_size=24).move_to(1.0 * arc.get_center())
            self.play(ShowCreation(arc, rate_func=linear, run_time=2))
        self.wait()


class SimpleIndicationRect(InteractiveScene):
    def construct(self):
        rect = Rectangle(3, 2)
        # Test
        self.play(FlashAround(rect, time_width=2.0, run_time=2, color=WHITE))


class WriteSPlane(InteractiveScene):
    def construct(self):
        title = Text("S-plane", font_size=72)
        title.set_color(YELLOW)
        self.play(Write(title))
        self.wait()


class ODEStoExp(InteractiveScene):
    def construct(self):
        # Test
        odes, exp = words = VGroup(
            Text("Differential\nEquations"),
            Tex("e^{st}", t2c={"s": YELLOW}, font_size=72),
        )
        exp.match_height(odes)
        words.arrange(RIGHT, buff=3.0)
        words.to_edge(UP, buff=1.25)

        top_arrow, low_arrow = arrows = VGroup(
            Arrow(odes.get_corner(UR), exp.get_corner(UL), path_arc=-60 * DEG, thickness=5),
            Arrow(exp.get_corner(DL), odes.get_corner(DR), path_arc=-60 * DEG, thickness=5),
        )
        arrows.set_fill(TEAL)

        top_words = Tex(R"Explain", font_size=36).next_to(top_arrow, UP, SMALL_BUFF)
        low_words = Tex(R"Solves", font_size=36).next_to(low_arrow, DOWN, SMALL_BUFF)

        exp.shift(0.25 * UP + 0.05 * LEFT)

        self.add(words)
        self.wait()
        self.play(
            Write(top_arrow),
            Write(top_words),
        )
        self.wait()
        self.play(
            # Write(low_arrow),
            TransformFromCopy(top_arrow, low_arrow, path_arc=-PI),
            Write(low_words),
        )
        self.wait()


class GenLinearEquationToOscillator(InteractiveScene):
    def construct(self):
        # General equation
        a_texs = ["a_n", "a_2", "a_1", "a_0"]
        x_texs = ["x^{n}(t)", "x''(t)", "x'(t)", "x(t)"]
        x_colors = color_gradient([BLUE, TEAL], len(x_texs), interp_by_hsl=True)
        t2c = dict()
        t2c.update({a: WHITE for a in a_texs})
        t2c.update({x: color for x, color in zip(x_texs, x_colors)})
        ode = Tex(R"a_n x^{n}(t) + \cdots + a_2 x''(t) + a_1 x'(t) + a_0 x(t) = 0", t2c=t2c, font_size=60)
        ode.move_to(DOWN)
        ode_2nd = ode["a_2 x''(t) + a_1 x'(t) + a_0 x(t) = 0"]

        self.play(Write(ode))
        self.wait()
        self.play(
            FadeOut(ode[R"a_n x^{n}(t) + \cdots + "]),
            ode_2nd.animate.move_to(UP),
            self.frame.animate.set_height(7)
        )

        # Transition
        alt_consts = VGroup(Tex(R"m"), Tex(R"\mu"), Tex(R"k"))
        alt_consts.scale(60 / 48)
        a_parts = VGroup(ode[tex][0] for tex in a_texs[1:])
        for const, a_part in zip(alt_consts, a_parts):
            const.move_to(a_part, RIGHT)
            const.align_to(ode[-1], DOWN)
            if const is alt_consts[1]:
                const.shift(0.1 * DOWN)
            self.play(
                FadeOut(a_part, 0.25 * UP),
                FadeIn(const, 0.25 * UP),
            )
        self.wait()


class VLineOverZero(InteractiveScene):
    def construct(self):
        # Test
        rect = Square(0.25)
        rect.move_to(2.5 * DOWN)
        v_line = Line(rect.get_top(), 4 * UP, buff=0.1)
        v_line.set_stroke(YELLOW, 2)
        rect.match_style(v_line)

        self.play(
            ShowCreationThenFadeOut(rect),
            ShowCreationThenFadeOut(v_line),
        )
        self.wait()


class KIsSomeConstant(InteractiveScene):
    def construct(self):
        rect = SurroundingRectangle(Text("k"), buff=0.05)
        rect.set_stroke(YELLOW, 2)
        words = Text("Some constant", font_size=24)
        words.next_to(rect, UP, SMALL_BUFF)
        words.match_color(rect)

        self.play(ShowCreation(rect), FadeIn(words))
        self.wait()


class WriteMu(InteractiveScene):
    def construct(self):
        sym = Tex(R"\mu")
        rect = SurroundingRectangle(sym, buff=0.05)
        rect.set_stroke(YELLOW, 2)
        mu = TexText("``Mu''")
        mu.set_color(YELLOW)
        mu.next_to(rect, DOWN)
        self.play(
            Write(mu),
            ShowCreation(rect)
        )
        self.wait()


class ReferenceGuessingExp(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.remove(self.background)
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)

        # Student asks
        question = Tex(R"x(t) = ???")
        lhs = question["x(t)"][0]
        rhs = question["= ???"][0]
        bubble = stds[2].get_bubble(question, bubble_type=SpeechBubble, direction=LEFT)
        lhs.save_state()
        lhs.scale(0.25).move_to([-6.24, 2.38, 0])

        self.play(
            morty.change("hesitant", look_at=stds[2].eyes),
            self.change_students("erm", "confused", "maybe", look_at=self.screen)
        )
        self.wait()
        self.play(
            stds[2].change("raise_left_hand", morty.eyes),
            Write(bubble[0]),
            Write(rhs, time_span=(0.5, 1.0)),
            Restore(lhs),
        )
        self.wait()
        self.add(Point())
        self.play(
            morty.says("Here's a trick:", mode="tease", bubble_creation_class=FadeIn),
            self.change_students("pondering", "thinking", "hesitant", look_at=UL),
        )
        self.wait(2)

        # Teacher gestures to upper right, students look confused and hesitant
        eq_point = 5 * RIGHT + 3 * UP
        self.play(
            morty.change("raise_right_hand", look_at=eq_point),
            FadeOut(bubble),
            FadeOut(morty.bubble),
            self.change_students("confused", "thinking", "hesitant", look_at=eq_point),
        )
        self.wait()
        self.play(self.change_students("confused", "hesitant", "confused", look_at=eq_point, lag_ratio=0.1))
        self.wait()
        self.play(
            morty.change("shruggie", look_at=eq_point),
        )
        self.wait(2)
        self.play(
            self.change_students("angry", "hesitant", "erm", look_at=morty.eyes),
            morty.animate.look_at(stds)
        )
        self.wait(2)

        # Transition: flip and reposition morty to where stds are
        new_teacher_pos = stds[2].get_bottom()
        new_teacher = morty.copy()
        new_teacher.change_mode("raise_left_hand")
        new_teacher.look_at(3 * UR)
        new_teacher.body.set_color(GREY_C)

        self.play(
            morty.animate.scale(0.8).flip().change_mode("confused").look_at(5 * UR).move_to(new_teacher_pos, DOWN),
            LaggedStartMap(FadeOut, stds, shift=DOWN, lag_ratio=0.2, run_time=1),
            FadeIn(new_teacher, time_span=(0.5, 1.5)),
        )
        self.play(morty.change("pleading", 3 * UR))
        self.play(Blink(new_teacher))
        self.wait(2)
        self.play(LaggedStart(
            morty.change("erm", new_teacher.eyes),
            new_teacher.change("guilty", look_at=morty.eyes),
            lag_ratio=0.5,
        ))
        self.wait(3)

        # Reference a graph
        self.play(
            morty.change("angry", 2 * UR),
            new_teacher.change("tease", 2 * UR)
        )
        self.play(Blink(morty))
        self.play(Blink(new_teacher))
        self.wait()


class FromGuessToLaplace(InteractiveScene):
    def construct(self):
        # Words
        strategy = VGroup(
            Text("“Strategy”", fill_color=GREY_A, font_size=72),
            TexText("Guess $x(t) = e^{{s}t}$", t2c={"{s}": YELLOW}, fill_color=WHITE, font_size=72),
        )
        strategy.arrange(DOWN)
        self.add(strategy)
        return

        # Comment on it
        exp_rect = SurroundingRectangle(strategy[1]["x(t) = e^{{s}t}"], buff=SMALL_BUFF)
        exp_words = Text("Why?", font_size=42)
        exp_words.next_to(exp_rect, RIGHT, SMALL_BUFF)
        VGroup(exp_rect, exp_words).set_color(PINK)

        guess_rect = SurroundingRectangle(strategy[1]["Guess"], buff=SMALL_BUFF)
        guess_rect.match_height(exp_rect, stretch=True).match_y(exp_rect)
        guess_words = Text("Seems dumb", font_size=36)
        guess_words.next_to(guess_rect, DOWN, SMALL_BUFF)
        VGroup(guess_rect, guess_words).set_color(RED)

        self.play(LaggedStart(
            ShowCreation(guess_rect),
            FadeIn(guess_words, lag_ratio=0.1),
            ShowCreation(exp_rect),
            FadeIn(exp_words, lag_ratio=0.1),
            lag_ratio=0.25
        ))
        self.wait()

        # Transition to Laplace
        laplace = Tex(R"\int_0^\infty x(t) e^{-{s}t} dt", t2c={"{s}": YELLOW}, font_size=72)
        laplace.move_to(strategy[1])

        self.play(LaggedStart(
            LaggedStartMap(FadeOut, VGroup(strategy[1]["Guess"], guess_rect, guess_words), shift=DOWN, lag_ratio=0.1),
            TransformFromCopy(strategy[0]["S"][0], laplace[R"\int"][0]),
            TransformFromCopy(strategy[0]["e"][0], laplace[R"0"][0]),
            TransformFromCopy(strategy[0]["g"][0], laplace[R"\infty"][0]),
            FadeOut(strategy[0], lag_ratio=0.1),
            # Break
            FadeOut(VGroup(exp_rect, exp_words), 0.5 * LEFT, lag_ratio=0.1),
            FadeTransform(strategy[1]["x(t)"][0], laplace["x(t)"][0]),
            FadeTransform(strategy[1]["="][0], laplace["-"][0]),
            FadeTransform(strategy[1]["e"][-1], laplace["e"][0]),
            FadeTransform(strategy[1]["{s}t"][0], laplace["{s}t"][0]),
            Write(laplace["dt"][0]),
            lag_ratio=0.15,
            run_time=3,
        ))
        self.wait()

        # Label laplace
        laplace_rect = SurroundingRectangle(laplace)
        laplace_rect.set_color(BLUE)
        laplace_label = Text("Laplace Transform", font_size=72)
        laplace_label.next_to(laplace_rect, UP)
        laplace_label.match_color(laplace_rect)

        self.play(
            Write(laplace_label),
            ShowCreation(laplace_rect),
        )
        self.wait()


class JustAlgebra(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer(mode="tease")
        morty.body.insert_n_curves(100)
        self.play(morty.says("Just algebra!", mode="hooray", look_at=2 * UL))
        self.play(Blink(morty))
        self.wait()
        self.play(
            FadeOut(morty.bubble),
            morty.change("tease", look_at=2 * UL + UP)
        )
        self.play(Blink(morty))
        self.wait()


class BothPositiveNumbers(InteractiveScene):
    def construct(self):
        tex = Tex("k / m")
        self.add(tex)

        # Test
        rects = VGroup(SurroundingRectangle(tex[c], buff=0.05) for c in "km")
        rects.set_stroke(GREEN, 3)
        plusses = VGroup(Tex(R"+").next_to(rect, DOWN, SMALL_BUFF) for rect in rects)
        plusses.set_fill(GREEN)

        self.play(
            LaggedStartMap(ShowCreation, rects, lag_ratio=0.5),
            LaggedStartMap(FadeIn, plusses, shift=0.25 * DOWN, lag_ratio=0.5)
        )
        self.wait()


class ButSpringsAreReal(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.play(
            stds[0].change("maybe", self.screen),
            stds[1].says("But...springs are real", mode="confused", look_at=self.screen),
            stds[2].change("erm", self.screen),
            morty.change("tease", stds[2].eyes)
        )
        self.wait(4)


class ShowIncreaseToK(InteractiveScene):
    def construct(self):
        # Test
        k = Tex(R"k")

        box = SurroundingRectangle(k)
        box.set_stroke(GREEN, 5)
        arrow = Vector(UP, thickness=6)
        arrow.set_fill(GREEN)
        center = box.get_center()

        self.play(
            ShowCreation(box),
            UpdateFromAlphaFunc(
                arrow, lambda m, a: m.move_to(
                    center + interpolate(-1, 1, a) * UP
                ).set_fill(
                    opacity=there_and_back(a) * 0.7
                ),
                run_time=4
            ),
        )
        self.wait()


class PureMathEquation(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"x''(t)": RED, "x(t)": TEAL, R"\omega": PINK}
        physics_eq = Tex(R"m x''(t) + k x(t) = 0", t2c=t2c, font_size=72)
        math_eq = Tex(R"a_2 x''(t) + a_0 x(t) = 0", t2c=t2c, font_size=72)

        self.add(physics_eq)
        self.play(LaggedStart(
            *(
                ReplacementTransform(physics_eq[tex][0], math_eq[tex][0])
                for tex in ["x''(t) +", "x(t) = 0"]
            ),
            FadeOut(physics_eq["m"], 0.5 * UP),
            FadeIn(math_eq["a_2"], 0.5 * UP),
            FadeOut(physics_eq["k"], 0.5 * UP),
            FadeIn(math_eq["a_0"], 0.5 * UP),
            run_time=2,
            lag_ratio=0.15
        ))
        self.wait()

        # Show solution
        implies = Tex(R"\Downarrow", font_size=72)
        answer = Tex(R"e^{\pm i\omega t}", font_size=90, t2c=t2c)
        answer.next_to(implies, DOWN, MED_LARGE_BUFF)
        omega_eq = Tex(R"\text{Where } \omega = \sqrt{a_2 / a_0}", t2c=t2c)
        omega_eq.next_to(answer, DOWN, MED_LARGE_BUFF)

        self.play(LaggedStart(
            math_eq.animate.next_to(implies, UP, MED_LARGE_BUFF),
            Write(implies),
            FadeIn(answer, DOWN),
            lag_ratio=0.25
        ))
        self.play(FadeIn(omega_eq))
        self.wait()


class LinearityDefinition(InteractiveScene):
    def construct(self):
        # Base differential equation string
        eq_str = R"m x''(t) + k x(t) = 0"
        t2c = {"x_1": TEAL, "x_2": RED, "0.0": YELLOW, "2.0": YELLOW}

        base_eq = Tex(eq_str)
        base_eq.to_edge(UP)

        eq1, eq2, eq3, eq4 = equations = VGroup(
            Tex(eq_str.replace("x", "x_1"), t2c=t2c),
            Tex(eq_str.replace("x", "x_2"), t2c=t2c),
            Tex(R"m\Big(x_1''(t) + x_2''(t) \Big) + k \Big(x_1(t) + x_2(t)\Big) = 0", t2c=t2c),
            Tex(R"m\Big(0.0 x_1''(t) + 2.0 x_2''(t) \Big) + k \Big(0.0 x_1(t) + 2.0 x_2(t)\Big) = 0", t2c=t2c),
        )
        for eq in equations:
            eq.set_max_width(7)
        equations.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        equations.to_edge(RIGHT)
        equations.shift(DOWN)

        phrase1, phrase2, phrase3, phrase4 = phrases = VGroup(
            TexText("If $x_1$ solves it:", t2c=t2c),
            TexText("and $x_2$ solves it:", t2c=t2c),
            TexText("Then $(x_1 + x_2)$ solves it:", t2c=t2c),
            TexText("Then $(0.0 x_1 + 2.0 x_2)$ solves it:", t2c=t2c),
        )

        for phrase, eq in zip(phrases, equations):
            phrase.set_max_width(5)
            phrase.next_to(eq, LEFT, LARGE_BUFF)

        eq4.move_to(eq3)
        phrase4.move_to(phrase3)

        kw = dict(edge_to_fix=RIGHT)
        c1_terms = VGroup(phrase4.make_number_changeable("0.0", **kw), *eq4.make_number_changeable("0.0", replace_all=True, **kw))
        c2_terms = VGroup(phrase4.make_number_changeable("2.0", **kw), *eq4.make_number_changeable("2.0", replace_all=True, **kw))

        # Show base equation
        self.play(Write(phrase1), FadeIn(eq1))
        self.wait()
        self.play(
            TransformMatchingTex(eq1.copy(), eq2, key_map={"x_1": "x_2"}, run_time=1, lag_ratio=0.01),
            FadeTransform(phrase1.copy(), phrase2)
        )
        self.wait()
        self.play(
            FadeIn(phrase3, DOWN),
            FadeIn(eq3, DOWN),
        )
        self.wait()
        self.play(
            FadeOut(eq3, 0.5 * DOWN),
            FadeOut(phrase3, 0.5 * DOWN),
            FadeIn(eq4, 0.5 * DOWN),
            FadeIn(phrase4, 0.5 * DOWN),
        )
        for _ in range(8):
            new_c1 = random.random() * 10
            new_c2 = random.random() * 10
            self.play(*(
                ChangeDecimalToValue(c1, new_c1, run_time=1)
                for c1 in c1_terms
            ))
            self.wait(0.5)
            self.play(*(
                ChangeDecimalToValue(c2, new_c2, run_time=1)
                for c2 in c2_terms
            ))
            self.wait(0.5)


class ComplainAboutNeelessComplexity(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students

        # Complain
        self.play(
            stds[0].change("confused", self.screen),
            stds[1].says("That’s needlessly\ncomplicated!", mode="angry", look_at=self.screen),
            stds[2].change("maybe", self.screen),
            morty.change("guilty"),
        )
        self.wait(3)
        self.play(
            stds[0].change("erm", self.screen),
            stds[1].debubble(mode="raise_left_hand", look_at=self.screen),
            stds[2].change("sassy", self.screen),
            morty.change("tease"),
        )
        self.wait()
        self.play(
            stds[1].change("raise_right_hand", ORIGIN),
            stds[0].change("pondering", ORIGIN),
            stds[2].change("pondering", ORIGIN),
        )
        self.wait(5)


class LetsGeneralize(InteractiveScene):
    def construct(self):
        morty = Mortimer()
        morty.to_corner(DR)
        self.play(
            morty.says("Let’s\ngenerlize!", mode="hooray")
        )
        self.play(Blink(morty))
        self.wait(3)


class EquationRect(InteractiveScene):
    def construct(self):
        rect = Rectangle(5.25, 1)
        rect.set_stroke(YELLOW, 3)

        # Test
        self.play(ShowCreation(rect))
        self.wait()
        self.play(rect.animate.stretch(0.5, 0).shift(4 * RIGHT).set_opacity(0))
        self.wait()


class GeneralLinearEquation(InteractiveScene):
    def construct(self):
        # Set up equations
        a_texs = ["a_n", "a_2", "a_1", "a_0"]
        x_texs = ["x^{n}(t)", "x''(t)", "x'(t)", "x(t)"]
        x_colors = color_gradient([BLUE, TEAL], len(x_texs), interp_by_hsl=True)
        t2c = {"{s}": YELLOW}
        t2c.update({a: WHITE for a in a_texs})
        t2c.update({x: color for x, color in zip(x_texs, x_colors)})
        ode = Tex(R"a_n x^{n}(t) + \cdots + a_2 x''(t) + a_1 x'(t) + a_0 x(t) = 0", t2c=t2c)
        exp_version = Tex(
            R"a_n \left({s}^n e^{{s}t}\right) "
            R"+ \cdots "
            R"+ a_2 \left({s}^2 e^{{s}t}\right) "
            R"+ a_1 \left({s}e^{{s}t}\right) "
            R"+ a_0 e^{{s}t} = 0",
            t2c=t2c
        )
        factored = Tex(R"e^{{s}t} \left(a_n {s}^n + \cdots + a_2 {s}^2 + a_1 {s} + a_0 \right) = 0", t2c=t2c)

        ode.to_edge(UP)
        exp_version.next_to(ode, DOWN, MED_LARGE_BUFF)
        factored.move_to(exp_version)

        # Introduce ode
        index = ode.submobjects.index(ode["a_2"][0][0])

        right_part = ode[index:]
        left_part = ode[:index]
        right_part.save_state()
        right_part.set_x(0)

        self.play(FadeIn(right_part, UP))
        self.wait()
        self.play(LaggedStart(
            Restore(right_part),
            Write(left_part)
        ))
        self.add(ode)

        # Highlight equation parts
        x_arrows = VGroup(
            Arrow(UP, ode[x_tex].get_bottom(), fill_color=color)
            for x_tex, color in zip(x_texs, x_colors)
        )
        x_arrows.reverse_submobjects()

        x_rects = VGroup(SurroundingRectangle(ode[x_tex], buff=SMALL_BUFF) for x_tex in x_texs)
        a_rects = VGroup(SurroundingRectangle(ode[a_tex]) for a_tex in a_texs)
        full_rect = SurroundingRectangle(ode[:-2])
        zero_rect = SurroundingRectangle(ode[-2:])
        VGroup(x_rects, a_rects, full_rect, zero_rect).set_stroke(YELLOW, 2)

        self.play(LaggedStartMap(ShowCreation, x_rects))
        self.wait()
        self.play(ReplacementTransform(x_rects, a_rects, lag_ratio=0.2))
        self.wait()
        self.play(ReplacementTransform(a_rects, VGroup(full_rect)))
        self.wait()
        self.play(ReplacementTransform(full_rect, zero_rect))
        self.wait()
        self.play(FadeOut(zero_rect))

        # Plug in e^{st}
        key_map = {
            R"+ a_0 x(t) = 0": R"+ a_0 e^{{s}t} = 0",
            R"+ a_1 x'(t)": R"+ a_1 \left({s}e^{{s}t}\right)",
            R"+ a_2 x''(t)": R"+ a_2 \left({s}^2 e^{{s}t}\right)",
            R"+ \cdots": R"+ \cdots",
            R"a_n x^{n}(t)": R"a_n \left({s}^n e^{{s}t}\right)",
        }

        self.play(LaggedStart(*(
            FadeTransform(ode[k1].copy(), exp_version[k2])
            for k1, k2 in key_map.items()
        ), lag_ratio=0.6, run_time=4))
        self.wait()
        self.play(
            TransformMatchingTex(
                exp_version,
                factored,
                matched_keys=[R"e^{{s}t}", "{s}^n", "{s}^2", "{s}", "a_n", "a_2", "a_1", "a_0"],
                path_arc=45 * DEG
            )
        )
        self.wait()

        # Highlight the polynomail
        poly_rect = SurroundingRectangle(factored[R"a_n {s}^n + \cdots + a_2 {s}^2 + a_1 {s} + a_0"])
        poly_rect.set_stroke(YELLOW, 1)

        self.play(
            ShowCreation(poly_rect),
            FadeOut(factored["e^{{s}t}"]),
            FadeOut(factored[R"\left("]),
            FadeOut(factored[R"\right)"]),
        )

        # Show factored expression
        linear_term_texs = [
            R"({s} - s_1)",
            R"({s} - s_2)",
            R"({s} - s_3)",
            R"\cdots",
            R"({s} - s_n)",
        ]
        fully_factored = Tex(
            R"a_n" + " ".join(linear_term_texs),
            t2c=t2c,
            font_size=42,
            isolate=linear_term_texs
        )
        fully_factored.next_to(poly_rect, DOWN)
        linear_terms = VGroup(
            fully_factored[tex][0]
            for tex in linear_term_texs
        )

        self.play(
            Transform(factored["{s}"][1].copy().replicate(4), fully_factored["{s}"].copy(), remover=True),
            FadeIn(fully_factored, time_span=(0.25, 1)),
        )
        self.wait()

        # Plane
        plane = ComplexPlane((-3, 3), (-3, 3), width=6, height=6)
        plane.set_height(4.5)
        plane.next_to(poly_rect, DOWN, LARGE_BUFF)
        plane.set_x(0)
        plane.add_coordinate_labels(font_size=16)
        c_label = Tex(R"\mathds{C}", font_size=90, fill_color=BLUE)
        c_label.next_to(plane, LEFT, aligned_edge=UP).shift(0.5 * DOWN)

        self.play(
            Write(plane, run_time=1, lag_ratio=2e-2),
            Write(c_label),
        )

        # Show some random root collections
        for n in range(4):
            roots = []
            n_roots = random.randint(3, 7)
            for _ in range(n_roots):
                root = complex(random.uniform(-3, 3), random.uniform(-3, 3))
                if random.random() < 0.25:
                    roots.append(root.real)
                else:
                    roots.extend([root, root.conjugate()])
            dots = Group(GlowDot(plane.n2p(z)) for z in roots)

            self.play(ShowIncreasingSubsets(dots))
            self.play(FadeOut(dots))

        # Turn linear terms into
        roots = [0.2 + 1j, 0.2 - 1j, -0.5 + 3j, -0.5 - 3j, -2]
        root_dots = Group(GlowDot(plane.n2p(root)) for root in roots)

        root_labels = VGroup(
            Tex(Rf"s_{{{n + 1}}}", font_size=36).next_to(dot.get_center(), UR, SMALL_BUFF)
            for n, dot in enumerate(root_dots)
        )
        root_labels.set_color(YELLOW)

        root_intro_kw = dict(lag_ratio=0.3, run_time=4)
        self.play(
            LaggedStart(*(
                FadeTransform(term, dot)
                for term, dot in zip(linear_terms, root_dots)
            ), **root_intro_kw),
            LaggedStart(*(
                TransformFromCopy(term[3:5], label)
                for term, label in zip(linear_terms, root_labels)
            ), **root_intro_kw),
            FadeOut(fully_factored["a_n"][0]),
        )
        self.wait()

        # Show the solutions
        frame = self.frame
        axes = VGroup(
            Axes((0, 10), (-y_max, y_max), width=5, height=1.25)
            for root in roots
            for y_max in [3 if root.real > 0 else 1]
        )
        axes.arrange(DOWN, buff=0.75)
        axes.next_to(plane, RIGHT, buff=6)

        c_trackers = Group(ComplexValueTracker(1) for root in roots)
        graphs = VGroup(
            self.get_graph(axes, root, c_tracker.get_value)
            for axes, root, c_tracker in zip(axes, roots, c_trackers)
        )

        axes_labels = VGroup(
            Tex(Rf"e^{{s_{{{n + 1}}} t}}", font_size=60)
            for n in range(len(axes))
        )
        for label, ax in zip(axes_labels, axes):
            label.next_to(ax, LEFT, aligned_edge=UP)
            label[1:3].set_color(YELLOW)

        self.play(
            FadeIn(axes, lag_ratio=0.2),
            frame.animate.reorient(0, 0, 0, (4.67, -0.94, 0.0), 10.96),
            LaggedStart(
                (FadeTransform(m1.copy(), m2) for m1, m2 in zip(root_labels, axes_labels)),
                lag_ratio=0.05,
                group_type=Group
            ),
            run_time=2
        )

        rect = Square(side_length=1e-3).move_to(plane.n2p(0))
        rect.set_stroke(TEAL, 3)
        for root_label, graph in zip(root_labels, graphs):
            self.play(
                ShowCreation(graph, time_span=(0.5, 2.0), suspend_mobject_updating=True),
                rect.animate.surround(root_label, buff=0.1),
            )
        self.play(FadeOut(rect))
        self.wait()

        # Add on constants
        constant_labels = VGroup(
            Tex(Rf"c_{{{n + 1}}}", font_size=60).next_to(label[0], LEFT, SMALL_BUFF, aligned_edge=UP)
            for n, label in enumerate(axes_labels)
        )
        constant_labels.set_color(BLUE_B)
        target_values = [0.5, 0.25, 1.5, -1.5, -1]

        solution_rect = SurroundingRectangle(VGroup(axes_labels, axes, constant_labels), buff=MED_SMALL_BUFF)
        solution_rect.set_stroke(WHITE, 1)
        solution_words = Text("All Solutions", font_size=60)
        solution_words.next_to(solution_rect, UP)
        solution_word = solution_words["Solutions"][0]
        solution_word.save_state(0)
        solution_word.match_x(solution_rect)

        const_rects = VGroup(SurroundingRectangle(c_label) for c_label in constant_labels)
        const_rects.set_stroke(BLUE, 3)

        plusses = Tex("+").replicate(4)
        for l1, l2, plus in zip(axes_labels, axes_labels[1:], plusses):
            plus.move_to(VGroup(l1, l2)).shift(SMALL_BUFF * LEFT)

        self.play(
            ShowCreation(solution_rect),
            Write(solution_word),
        )
        self.play(
            LaggedStartMap(Write, constant_labels, lag_ratio=0.5),
            LaggedStart(*(
                c_tracker.animate.set_value(value)
                for c_tracker, value in zip(c_trackers, target_values)
            ), lag_ratio=0.5),
            run_time=4
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeIn, plusses),
            Write(solution_words["All"]),
            Restore(solution_word),
        )
        self.wait()

        # Play with constants
        self.play(LaggedStartMap(ShowCreation, const_rects, lag_ratio=0.15))
        value_sets = [
            [1, 1, 1, 1, 1],
            [1j, -1j, 1 + 1j, -1 + 1j, -0.5],
            [-0.5, 1j, 1j, 1 + 1j, -1],
        ]
        for values in value_sets:
            self.play(
                LaggedStart(*(
                    c_tracker.animate.set_value(value)
                    for c_tracker, value in zip(c_trackers, values)
                ), lag_ratio=0.25, run_time=3)
            )
            self.wait()
        self.play(LaggedStartMap(FadeOut, const_rects, lag_ratio=0.25))
        self.wait()

    def get_graph(self, axes, s, get_const):
        def func(t):
            return (get_const() * np.exp(s * t)).real

        graph = axes.get_graph(func, bind=True, stroke_color=TEAL, stroke_width=2)
        return graph


class HoldUpGeneralLinear(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(500)

        self.play(
            morty.change("raise_right_hand"),
            self.change_students("pondering", "thinking", "tease", look_at=3 * UR)
        )
        self.wait()
        self.play(
            morty.change("sassy", look_at=3 * UR),
            self.change_students("hesitant", "erm", "maybe")
        )
        self.wait(5)


class BigCross(InteractiveScene):
    def construct(self):
        cross = Cross(Rectangle(4, 1.5))
        cross.set_stroke(RED, width=(0, 8, 8, 8, 0))
        self.play(ShowCreation(cross))
        self.wait()


class DifferentialEquation(InteractiveScene):
    def construct(self):
        # ode to x
        x_term = Tex(R"x(t)", font_size=90)
        arrow = Vector(DOWN, thickness=5)
        arrow.move_to(ORIGIN, DOWN)
        words = Text("Differential Equation", font_size=72)
        words.next_to(arrow, UP)

        self.play(Write(x_term))
        self.wait()
        self.play(
            GrowArrow(arrow),
            FadeIn(words),
            x_term.animate.next_to(arrow, DOWN),
        )
        self.wait()


class DumbTrickAlgebra(InteractiveScene):
    def construct(self):
        pass


class LaplaceTransformAlgebra(InteractiveScene):
    def construct(self):
        # Add equation
        colors = color_gradient([TEAL, RED], 3, interp_by_hsl=True)
        t2c = {
            R"x(t)": colors[0],
            R"x'(t)": colors[1],
            R"x''(t)": colors[2],
            "{s}": YELLOW,
        }
        equation = Tex(
            R"{m} x''(t) + \mu x'(t) + k x(t) = F_0 \cos(\omega_l t)",
            t2c=t2c
        )
        equation.to_edge(UP, buff=1.5)

        arrow = Vector(1.25 * DOWN, thickness=6)
        arrow.next_to(equation, DOWN)
        arrow_label = Tex(R"\mathcal{L}", font_size=72)
        arrow_label.next_to(arrow, RIGHT, buff=SMALL_BUFF)

        self.add(equation)
        self.wait()
        self.play(
            GrowArrow(arrow),
            FadeIn(arrow_label, shift=0.5 * DOWN)
        )

        # Make transformed
        transformed_eq = Tex(
            R"{m} {s}^2 X({s}) + \mu {s} X({s}) + k X({s}) = \frac{F_0 {s}}{{s}^2 + \omega_l^2}",
            t2c=t2c
        )
        transformed_eq.next_to(arrow, DOWN)

        xt_texs = ["x(t)", "x'(t)", "x''(t)"]
        Xs_texs = ["X({s})", "{s} X({s})", "{s}^2 X({s})"]

        rects = VGroup()
        srcs = VGroup()
        trgs = VGroup()
        for t1, t2, color in zip(xt_texs, Xs_texs, colors):
            src = equation[t1][0]
            trg = transformed_eq[t2][-1]
            rect = SurroundingRectangle(src, buff=0.05)
            rect.set_stroke(color, 2)
            rect.target = rect.generate_target()
            rect.target.surround(trg, buff=0.05)

            rects.add(rect)
            srcs.add(src.copy())
            trgs.add(trg)

        self.play(LaggedStartMap(ShowCreation, rects, lag_ratio=0.25, run_time=1.5))
        self.play(
            LaggedStart(
                *(FadeTransform(src, trg)
                for src, trg in zip(srcs, trgs)),
                lag_ratio=0.25,
                group_type=Group,
                run_time=1.5
            ),
            LaggedStartMap(MoveToTarget, rects, lag_ratio=0.25, run_time=1.5)
        )
        self.play(
            LaggedStart(*(
                TransformFromCopy(equation[tex], transformed_eq[tex][:2])
                for tex in ["{m}", "+", R"\mu", "k", "="]
            )),
            TransformMatchingParts(
                equation[R"F_0 \cos(\omega_l t)"].copy(),
                transformed_eq[R"\frac{F_0 {s}}{{s}^2 + \omega_l^2}"]
            )
        )
        self.wait()

        # Factor it
        factored = Tex(
            R"X({s}) \left({m} {s}^2+ \mu {s} + k\right) = \frac{F_0 {s}}{{s}^2 + \omega_l^2}",
            t2c=t2c
        )
        factored.move_to(transformed_eq)
        left_rect = SurroundingRectangle(factored["X({s})"], buff=0.05)
        left_rect.set_stroke(YELLOW, 2)

        self.play(
            TransformMatchingTex(
                transformed_eq,
                factored,
                matched_keys=["X({s})"],
                path_arc=30 * DEG
            ),
            ReplacementTransform(rects, VGroup(left_rect), path_arc=30 * DEG),
            run_time=2
        )
        self.play(FadeOut(left_rect))
        self.wait()

        # Rearrange
        rearranged = Tex(
            R"X({s}) = \frac{F_0 {s}}{{s}^2 + \omega_l^2} \frac{1}{{m} {s}^2+ \mu {s} + k}",
            t2c=t2c
        )
        rearranged.next_to(factored, DOWN, LARGE_BUFF)

        self.play(
            TransformMatchingTex(
                factored.copy(),
                rearranged,
                matched_keys=["X({s})"],
            )
        )
