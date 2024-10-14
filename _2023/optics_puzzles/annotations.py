from manim_imports_ext import *
from _2023.optics_puzzles.objects import *


class AnnotateDemo(InteractiveScene):
    def construct(self):
        image = ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/barber_pole/images/DemoStill.jpg")
        image.set_height(FRAME_HEIGHT)
        plane = NumberPlane().fade(0.25)
        # self.add(image)
        # self.add(plane)

        # Label sugar
        sugar_label = Text("Sugar solution\n(0.75g sucrose/mL water)")
        sugar_label.move_to(2.5 * UP)
        sugar_label.set_backstroke(BLACK, 3)
        arrow_kw = dict(stroke_color=RED, stroke_width=10)
        sugar_arrow = Arrow(sugar_label, plane.c2p(0, 0.5), **arrow_kw)

        self.play(
            Write(sugar_label, lag_ratio=0.01, run_time=2),
            ShowCreation(sugar_arrow),
        )
        self.wait()

        # Label light
        light_label = Text("White light\n(unpolarized)")
        light_label.match_y(sugar_label)
        light_label.match_style(sugar_label)
        light_label.to_edge(RIGHT)
        light_arrow = Arrow(light_label, plane.c2p(4.75, 0.85), buff=0.1, **arrow_kw)

        self.play(
            FadeTransform(sugar_label, light_label),
            ReplacementTransform(sugar_arrow, light_arrow),
        )
        self.wait()

        # Label polarizer
        filter_label = Text("Linearly polarizing filter\n(variable angle)")
        filter_label.set_x(3.5).to_edge(UP)
        filter_label.match_style(sugar_label)
        filter_arrow = Arrow(filter_label, plane.c2p(3.4, 1.25), buff=0.1, **arrow_kw)

        self.play(
            FadeTransform(light_label, filter_label),
            ReplacementTransform(light_arrow, filter_arrow),
        )
        self.wait()
        self.play(
            filter_label.animate.set_x(-2.2).to_edge(UP, buff=0.5),
            filter_arrow.animate.set_x(-3.3),
        )
        self.wait()


class HoldUp(TeacherStudentsScene):
    def construct(self):
        arrow = Vector(LEFT, stroke_color=YELLOW)
        arrow.to_edge(UP, buff=1.0)
        # self.add(arrow)
        self.play(
            self.teacher.change("raise_right_hand", look_at=3 * UR),
            self.change_students("pondering", "erm", "sassy", look_at=self.screen)
        )
        self.wait(3)
        self.play(
            self.change_students("maybe", "pondering", "hesitant", look_at=3 * UR)
        )
        self.wait(3)


class SteveVideoWrapper(VideoWrapper):
    title = "Steve Mould: Why Sugar Always Twists Light"


class DidntSteveCover(TeacherStudentsScene):
    def construct(self):
        stds = self.students
        morty = self.teacher
        image = ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/barber_pole/chat with Steve/MouldThumbnail.jpg")
        image.set_height(3)
        thumbnail = Group(
            SurroundingRectangle(image, buff=0).set_stroke(WHITE, 3),
            image
        )
        thumbnail.next_to(stds[2].get_corner(UR), UP, buff=1.0)
        thumbnail.shift(RIGHT)

        # Test
        self.play(
            stds[2].says("Didn't Steve Mould\ncover this?", mode="raise_right_hand"),
            morty.change("tease"),
            self.change_students("pondering", "pondering", look_at=thumbnail),
            FadeIn(thumbnail, UP)
        )
        self.wait(4)


class FocusOnWall(InteractiveScene):
    def construct(self):
        # Test
        words = Text("Notice what color comes out", font_size=72)
        words.set_backstroke(BLACK, 5)
        words.to_edge(UP)
        arrow = Arrow(
            words["Notice what"].get_bottom(), 6.5 * LEFT + 0.5 * UP,
            buff=0.5,
            stroke_color=RED, stroke_width=14,
        )
        self.play(
            FadeIn(words, 0.25 * UP),
            GrowArrow(arrow),
        )
        self.wait()


class WhatWeNeedToUnderstand(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students

        # React
        self.play(
            self.change_students("pondering", "maybe", "confused", look_at=self.screen),
            morty.change("tease")
        )
        self.wait(3)
        self.play(
            self.change_students("erm", "sassy", "maybe", look_at=self.screen)
        )
        self.wait(3)

        # Core concepts
        concepts = VGroup(
            Text("Polarization"),
            Text("Scattering"),
            Text("The index of refraction"),
        )
        concepts.arrange(DOWN, aligned_edge=LEFT)
        concepts.next_to(self.hold_up_spot, UP)

        for i in range(3):
            concepts[i].align_to(concepts, DOWN)
            self.play(
                FadeIn(concepts[i], 0.5 * UP),
                concepts[:i].animate.shift(UP),
                morty.change("raise_right_hand", concepts),
                *(pi.animate.look_at(concepts) for pi in stds),
            )
            self.wait()
        self.wait()

        # Start with an overview
        self.play(
            morty.says("To begin:\nAn overview"),
            FadeOut(concepts, UP),
            self.change_students("tease", "happy", "well", look_at=morty.eyes)
        )
        self.look_at(self.screen)
        self.wait(3)


class ThisIsStillWhiteLight(InteractiveScene):
    def construct(self):
        # Test
        words = Text("This is still white light", font_size=60)
        words.to_edge(UP)
        arrow = Arrow(
            words["This"].get_bottom(), LEFT + 0.5 * UP,
            stroke_color=WHITE, stroke_width=8,
        )

        self.play(
            Write(words, run_time=1),
            FadeIn(arrow, RIGHT, run_time=4.0, rate_func=rush_into),
        )
        self.wait()


class Questions(InteractiveScene):
    def construct(self):
        # Title
        title = Text("Lingering questions", font_size=72)
        title.add(Underline(title))
        title.to_edge(UP, buff=0.25)
        self.add(title)

        # Questions
        questions = self.get_questions()

        last_questions = questions[1:]
        last_questions.save_state()
        last_questions.set_height(6, about_edge=DL)

        for question in last_questions:
            self.play(FadeIn(question, 0.25 * UP))
            self.wait()
        self.wait()
        self.play(
            FadeIn(questions[0], shift=0.25 * DOWN),
            Restore(last_questions)
        )
        self.wait()

        # Check marks
        marks = VGroup(*(
            Checkmark(font_size=60).next_to(question[0], RIGHT).shift(0.15 * UP)
            for question in questions
        ))
        for mark in marks:
            mark.align_to(marks, RIGHT)

        # Crossing off
        cross_lines = VGroup()
        for question in questions:
            full_question = question[1]
            phrases = full_question.get_text().split("\n")
            cross_lines.add(VGroup(*(
                Line(LEFT, RIGHT).set_stroke(YELLOW, 3).replace(full_question[phrase], dim_to_match=0)
                for phrase in phrases
            )))

        # Mark off Q0
        self.play(
            Write(marks[0], stroke_color=GREEN),
            ShowCreation(cross_lines[0]),
        )
        self.wait()

        # Highlight question 3
        rect = SurroundingRectangle(questions[3])
        rect.set_stroke(YELLOW, 2)
        arrow = Vector(LEFT, stroke_color=YELLOW)
        arrow.next_to(rect, RIGHT)
        self.play(
            questions[:3].animate.set_opacity(0.5),
            ShowCreation(rect),
            FadeInFromPoint(arrow, ORIGIN),
        )
        self.wait()

        self.play(
            FadeOut(rect),
            FadeOut(arrow),
            ShowCreation(cross_lines[3]),
            Write(marks[3], stroke_color=GREEN),
            questions[:3].animate.set_opacity(1),
        )
        self.wait()

        # Question 1
        rect.set_stroke(opacity=0)
        arrow.set_stroke(opacity=0)
        self.play(
            rect.animate.surround(questions[1]).set_stroke(opacity=1),
            arrow.animate.next_to(questions[1], RIGHT).set_stroke(opacity=1),
        )
        self.wait()
        marks[1].set_fill(opacity=0)
        marks[1].set_stroke(GREEN, 2)
        self.play(
            FadeOut(rect),
            Write(marks[1]),
        )
        self.wait()

        # Question 2
        self.play(
            arrow.animate.next_to(questions[2], RIGHT)
        )
        self.wait()

    def get_questions(self):
        kw = dict(alignment="LEFT")
        question_texts = [
            "What exactly is wiggling?",
            "Why does sugar make light twist?",
            "Why does the twisting rate\ndepend on frequency?",
            "Why do we see colors\nin diagonal stripes?",
        ]
        questions = VGroup(*(
            VGroup(
                Text(f"Question #{n}:", **kw).set_fill(GREY_A),
                Text(text, **kw),
            ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
            for n, text in enumerate(question_texts)
        ))
        questions.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        questions.set_height(6)
        questions.to_corner(DL)
        return questions


class TwoLines(InteractiveScene):
    def construct(self):
        h_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        v_line = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        lines = VGroup(h_line, v_line)
        lines.set_stroke(WHITE, 3)
        self.add(lines)


class CyclingQuestions(Questions):
    def construct(self):
        # Test
        questions = self.get_questions()[1:]
        for question in questions:
            question.scale(1.1)
            question.to_corner(UL, buff=0.5)

        self.add(questions[0])
        self.wait()
        for q1, q2 in zip(questions, questions[1:]):
            self.play(FadeOut(q1, UP), FadeIn(q2, UP))
            self.wait()


class FromOnHigh(InteractiveScene):
    def construct(self):
        rects = ScreenRectangle().replicate(3)
        rects.set_height(0.3 * FRAME_HEIGHT)
        rects.arrange(RIGHT)
        # self.add(rects)

        top_words = Text("Authoritative explainer")
        top_words.to_edge(UP)
        low_words = Text("Fundamental understanding")
        low_words.to_edge(DOWN)
        cross = Cross(top_words)

        top_arrows = VGroup(*(Arrow(top_words.get_bottom(), rect.get_top()) for rect in rects))
        low_arrows = VGroup(*(Arrow(low_words.get_top(), rect.get_bottom()) for rect in rects))

        # Test
        self.add(top_words)
        self.play(LaggedStartMap(GrowArrow, top_arrows))
        self.play(ShowCreation(cross))
        self.wait()
        self.play(
            Transform(top_arrows, low_arrows),
            FadeTransform(VGroup(top_words, cross), low_words),
        )
        self.wait()


class ThreeParts(InteractiveScene):
    def construct(self):
        # Setup
        self.add(FullScreenRectangle())
        parts = ScreenRectangle().replicate(3)
        parts.set_height(0.3 * FRAME_HEIGHT)
        parts.arrange(RIGHT)
        parts.arrange_to_fit_width(FRAME_WIDTH - 0.5)
        parts.set_fill(BLACK, 1)
        parts.set_stroke(width=0)
        self.add(parts)

        self.clear()

        rect = SurroundingRectangle(parts[0], buff=0)
        rect.set_stroke(WHITE, 4)

        self.add(rect)

        # Test
        self.wait()
        for part in parts[1:]:
            self.play(rect.animate.surround(part, buff=0))
            self.wait()


class UnitRVector(InteractiveScene):
    def construct(self):
        text = TexText(R"Unit vector in \\ the direction of $\vec{r}$")
        text.set_fill(YELLOW)
        text.set_backstroke(BLACK, 10)
        self.add(text)


class RSquaredVsR(InteractiveScene):
    def construct(self):
        # Axes
        axes = Axes((0, 10), (0, 1, 0.25), height=5, width=12)
        axes.x_axis.add_numbers()
        axes.y_axis.add_numbers(num_decimal_places=2)
        self.add(axes)

        # Add graphs
        graph1 = axes.get_graph(lambda x: 1 / x**2, x_range=(0.1, 10)).set_stroke(YELLOW, 5)
        graph2 = axes.get_graph(lambda x: 1 / x, x_range=(0.1, 10)).set_stroke(BLUE, 5)
        graphs = VGroup(graph1, graph2)
        labels = VGroup(
            Tex(R"f(r) = \frac{1}{r^2}").set_color(YELLOW),
            Tex(R"f(r) = \frac{1}{r}").set_color(BLUE),
        )
        labels.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        labels.move_to(axes, UR)

        for graph, label in zip(graphs, labels):
            self.play(
                ShowCreation(graph, run_time=3),
                FadeIn(label, 0.5 * UP),
            )
        self.wait(3)


class SimpleRect(InteractiveScene):
    def construct(self):
        rad = Text("rad")
        rect = SurroundingRectangle(rad)
        rect.set_stroke(YELLOW, 3)
        arrow = Vector(UP)
        arrow.set_stroke(YELLOW)
        arrow.next_to(rect, DOWN)
        self.play(ShowCreation(arrow), ShowCreation(rect))
        self.wait()


class ThisIsLight(InteractiveScene):
    def construct(self):
        morty = Mortimer(height=1.5)
        morty.to_corner(DR)
        self.play(morty.says("This is light!", mode="surprised"))
        self.play(Blink(morty))
        self.wait()


class MentionMaxwell(TeacherStudentsScene):
    def construct(self):
        # Introduce
        stds = self.students
        morty = self.teacher
        kw = dict(
            tex_to_color_map={
                R"\mathbf{E}": BLUE,
                R"\mathbf{B}": YELLOW,
            },
            font_size=30,
        )
        maxwells_equations = VGroup(
            Tex(R"\nabla \cdot \mathbf{E}=\frac{\rho}{\varepsilon_0}", **kw),
            Tex(R"\nabla \cdot \mathbf{B}=0", **kw),
            Tex(R"\nabla \times \mathbf{E}=-\frac{\partial \mathbf{B}}{\partial t}", **kw),
            Tex(R"\nabla \times \mathbf{B}=\mu_0\left(\mathbf{J}+\varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}\right)", **kw),
        )
        maxwells_equations.arrange(DOWN, aligned_edge=LEFT)
        equations_rect = SurroundingRectangle(maxwells_equations, buff=0.25)
        equations_rect.set_stroke(WHITE, 1)
        equations_rect.set_fill(BLACK, 1)
        maxwells_equations.add_to_back(equations_rect)

        maxwells_equations.next_to(stds[2].get_corner(UR), UP)
        maxwells_equations.shift(MED_SMALL_BUFF * UP)
        equations_title = Text("Maxwell's Equations")
        equations_title.next_to(maxwells_equations, UP, buff=0.25)

        self.play(
            stds[2].says(TexText(R"What about \\ Maxwell's equations?")),
            morty.change("tease"),
            self.change_students("confused", "pondering")
        )
        self.play(
            stds[2].change("raise_right_hand"),
            FadeIn(maxwells_equations, UP),
        )
        self.play(
            FadeTransform(stds[2].bubble.content["Maxwell's equations"].copy(), equations_title),
            stds[2].debubble(),
            *(pi.animate.look_at(equations_title) for pi in [morty, *stds[:2]]),
        )
        self.wait()
        self.play(self.change_students("maybe", "erm", look_at=maxwells_equations))
        self.wait(4)

        maxwells_equations.add(equations_title)

        # Lorentz law
        maxwells_equations.generate_target()
        maxwells_equations.target.scale(0.7).to_edge(LEFT)

        equation = Tex(R"""
            \vec{E}_{\text{rad}}(\vec{r}, t) = 
            {-q \over 4\pi \epsilon_0 c^2}
            {1 \over ||\vec{r}||}
            \vec{a}_\perp(t - ||\vec{r}|| / c)
        """, font_size=42)
        lhs = equation[R"\vec{E}_{\text{rad}}(\vec{r}, t)"]
        lhs.set_color(BLUE)
        equation[R"\vec{a}_\perp("].set_color(PINK)
        equation[R")"][1].set_color(PINK)

        equation.next_to(morty.get_corner(UL), UP)
        equation.shift_onto_screen()
        equation.match_y(maxwells_equations)
        eq_rect = SurroundingRectangle(equation)
        eq_rect.set_stroke(BLUE, 2)
        eq_rect.set_fill(BLUE, 0.1)

        implies_arrow = Arrow(eq_rect, maxwells_equations.target)
        implies_word = Text("Derived from these", font_size=30)
        implies_word.next_to(implies_arrow, UP)

        question = Text("How far will this take us?")
        question.next_to(equation, UP, buff=2.0)
        question.shift_onto_screen()
        arrow = Arrow(equation, question)

        self.play(
            morty.change("raise_right_hand"),
            FadeIn(equation, 0.5 * UP),
            MoveToTarget(maxwells_equations),
            GrowArrow(implies_arrow),
            FadeIn(implies_word, lag_ratio=0.1),
            self.change_students("well", "happy")
        )
        self.wait()
        self.add(eq_rect, equation)
        self.play(
            FadeIn(eq_rect),
            ShowCreation(arrow),
            FadeIn(question, lag_ratio=0.1),
            morty.change("tease", look_at=question),
            self.change_students("tease", "well", "happy", look_at=equation)
        )
        self.wait(4)

        # Remove
        self.remove(equation)
        everything = Group(*self.mobjects)

        self.play(
            LaggedStartMap(FadeOut, everything),
            equation.copy().animate.scale(36.0 / 42).to_corner(UR),
            run_time=1,
        )
        self.wait()


class BasicallyZ(InteractiveScene):
    def construct(self):
        rect = Rectangle(6.0, 2.0)
        rect.set_stroke(YELLOW, 2)
        rect.set_fill(YELLOW, 0.2)
        rect.to_edge(RIGHT, buff=0.25)
        words = Text("Essentially parallel\nto the z-axis")
        words.next_to(rect, UP)
        self.play(
            FadeIn(words, 0.25 * UP),
            FadeIn(rect)
        )
        self.wait()


class StrengthInDifferentDirectionsWithDecimal(InteractiveScene):
    def construct(self):
        line = Line(ORIGIN, 7 * RIGHT)
        line.set_stroke(TEAL, 4)
        arc = always_redraw(lambda: Arc(angle=line.get_angle(), radius=0.5))
        angle_label = Integer(0, unit=R"^\circ")
        angle_label.add_updater(lambda m: m.set_value(line.get_angle() / DEGREES))
        angle_label.add_updater(lambda m: m.set_height(clip(arc.get_height(), 0.01, 0.4)))
        angle_label.add_updater(lambda m: m.next_to(arc.pfp(0.3), normalize(arc.pfp(0.3)), SMALL_BUFF, aligned_edge=DOWN))

        strong_words = Text("Strongest in this direction")
        strong_words.next_to(line, UP)
        cos_temp_text = "cos(00*)=0.00"
        weak_words = Text(f"Weaker by a factor of {cos_temp_text}", font_size=36)
        cos_template = weak_words[cos_temp_text][0]
        cos_template.set_opacity(0)
        weak_words.next_to(line, UP)

        strong_words.set_backstroke(BLACK, 10)
        weak_words.set_backstroke(BLACK, 10)

        def get_cos_tex():
            cos_tex = Tex(R"\cos(10^\circ) = 0.00", font_size=36)
            cos_tex.make_number_changeable("10", edge_to_fix=RIGHT).set_value(line.get_angle() / DEGREES)
            cos_tex.make_number_changeable("0.00").set_value(math.cos(line.get_angle()))
            cos_tex.rotate(line.get_angle())
            cos_tex.move_to(weak_words[-len(cos_temp_text) + 1:])
            cos_tex.set_backstroke(BLACK, 10)
            return cos_tex

        cos_tex = always_redraw(get_cos_tex)

        # Test
        self.play(ShowCreation(line), Write(strong_words, run_time=1))
        self.wait(2)
        self.add(arc, angle_label, weak_words, cos_tex)
        self.remove(strong_words)
        rot_group = VGroup(line, weak_words)
        self.play(
            Rotate(rot_group, 30 * DEGREES, about_point=ORIGIN),
            run_time=3
        )
        self.wait(2)
        self.play(
            self.frame.animate.set_height(12),
            Rotate(rot_group, 30 * DEGREES, about_point=ORIGIN),
            run_time=3
        )
        self.wait(2)
        self.play(
            self.frame.animate.set_height(15),
            Rotate(rot_group, 30 * DEGREES, about_point=ORIGIN),
            run_time=3
        )
        self.wait(2)


class ERadEquation(InteractiveScene):
    def construct(self):
        equation = Tex(R"""
            \vec{E}_{\text{rad}}(\vec{r}, t) = 
            {-q \over 4\pi \epsilon_0 c^2}
            {1 \over ||\vec{r}||}
            \vec{a}_\perp(t - ||\vec{r}|| / c)
        """, font_size=36)
        lhs = equation[R"\vec{E}_{\text{rad}}(\vec{r}, t)"]
        lhs.set_color(BLUE)
        equation[R"\vec{a}_\perp("].set_color(PINK)
        equation[R")"][1].set_color(PINK)
        self.add(equation)


class XZLabel(InteractiveScene):
    def construct(self):
        xz_label = Tex("xz")
        x, z = xz_label
        x.next_to(ORIGIN, UP, SMALL_BUFF).to_edge(RIGHT, buff=0.2)
        z.next_to(ORIGIN, RIGHT, SMALL_BUFF).to_edge(UP, buff=0.2)
        self.add(xz_label)


class TransitionBeforePolarization(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.change_students("pondering", "thinking", "happy", look_at=self.screen),
        )
        self.wait(3)
        self.play(
            self.teacher.change("raise_left_hand", look_at=3 * UR),
            self.change_students("well", "tease", "happy", look_at=3 * UR),
        )
        self.wait(4)


class AskAboutQuantum(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students

        # Question
        self.play(
            stds[0].says("What about, like,\nphotons and quantum stuff?"),
            morty.change("well"),
            self.change_students(None, "sassy", "hesitant", look_at=stds[0].eyes),
        )
        self.wait(3)

        # Create T chart
        titles = VGroup(
            Text("Classical"),
            Text("Quantum"),
        )
        v_lines = Line(UP, DOWN).replicate(2).set_height(4.5)
        v_lines.arrange(RIGHT, buff=titles[0].get_width() + 0.75)
        v_lines.to_edge(UP, buff=0)
        v_lines.shift(1.5 * RIGHT)
        h_line = Line(LEFT, RIGHT)
        h_line.set_width(13.25)
        h_line.to_edge(UP, buff=titles.get_height() + 0.5)
        h_line.set_x(0)
        lines = VGroup(h_line, *v_lines)
        lines.set_stroke(WHITE, 2)

        titles[0].move_to(v_lines).to_edge(UP, buff=0.25)
        titles[1].next_to(v_lines, RIGHT, buff=0.5).match_y(titles[0])

        # Points
        points = VGroup(
            Text("Waves radiate away from changing charges"),
            Text("These waves have polarization"),
            Text("Energy scales continuously"),
        )
        points.scale(0.7)
        points.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        points.next_to(h_line, DOWN, buff=0.5, aligned_edge=LEFT)

        soft_h_lines = h_line.replicate(len(points))
        soft_h_lines.set_stroke(WHITE, 1, opacity=0.5)
        for point, soft_line in zip(points, soft_h_lines):
            soft_line.next_to(point, DOWN, buff=0.25)
            soft_line.align_to(h_line, LEFT)

        checks = VGroup(*(
            Checkmark().match_x(title).match_y(point)
            for point in points
            for title in titles
        ))
        cross = Exmark().move_to(checks[-1])

        last_q_point = Text(
            """
            Energy comes in
            discrete chunks
            (that is, in quanta)
            """,
            alignment="LEFT"
        )
        last_q_point.scale(0.7)
        last_q_point.next_to(cross, DOWN, buff=0.5)
        last_q_point.align_to(titles[1], LEFT)
        last_q_point.set_color(YELLOW)

        # Animate in the charge
        self.play(
            morty.change("raise_right_hand"),
            FadeTransform(stds[0].bubble.content["quantum"].copy(), titles[1]),
            FadeIn(titles[0], shift=0.25 * UP),
            ShowCreation(h_line),
            ShowCreation(v_lines),
            Write(soft_h_lines),
            stds[0].debubble(mode="pondering"),
            self.change_students(None, "pondering", "pondering", look_at=UP),
            run_time=2,
        )
        self.wait()

        # Points
        self.play(
            FadeIn(points[0], lag_ratio=0.1),
            Write(checks[:2], lag_ratio=0.7, stroke_color=GREEN),
            *(pi.animate.look_at(titles) for pi in self.pi_creatures)
        )
        self.wait()
        self.play(
            FadeIn(points[1], lag_ratio=0.1),
            Write(checks[2:4], lag_ratio=0.7, stroke_color=GREEN)
        )
        self.wait()

        self.play(
            FadeIn(points[2], lag_ratio=0.1),
            Write(checks[4], stroke_color=GREEN),
            FadeIn(cross, scale=0.5),
            morty.change("sassy", look_at=cross)
        )
        self.wait()

        self.play(
            FadeOut(self.pi_creatures, DOWN),
            FadeOut(soft_h_lines[-1]),
            # FadeIn(last_q_point, lag_ratio=0.1),
            v_lines.animate.set_height(7, about_edge=UP, stretch=True),
        )
        self.wait()


class ContinuousWave(InteractiveScene):
    def construct(self):
        wave = self.get_wave()
        points = wave.get_points().copy()
        wave.add_updater(lambda m: m.set_points(points).stretch(
            (1 - math.sin(self.time)), 1,
        ))
        self.add(wave)
        self.wait(20)

    def get_wave(self):
        axes = Axes((0, 2 * TAU), (0, 1))
        wave = axes.get_graph(np.sin)
        wave.set_stroke(BLUE, 4)
        wave.set_width(2.0)
        return wave


class DiscreteWave(ContinuousWave):
    def construct(self):
        # Test
        waves = self.get_wave().replicate(3)
        waves.scale(2)
        waves.arrange(DOWN, buff=1.0)
        labels = VGroup()
        for n, wave in zip(it.count(1), waves):
            wave.stretch(0.5 * n, 1)
            label = TexText(f"{n} $hf$" + ("" if n > 1 else ""))
            label.scale(0.5)
            label.next_to(wave, UP, buff=0.2)
            labels.add(label)
            self.play(
                FadeIn(wave),
                FadeIn(label),
            )

        dots = Tex(R"\vdots", font_size=60)
        dots.next_to(waves, DOWN)
        self.play(Write(dots))
        self.wait()


class ContinuousGraph(InteractiveScene):
    def construct(self):
        axes = Axes((0, 5), (0, 5), width=4, height=4)
        graph = axes.get_graph(lambda x: (x**2) / 5)
        graph.set_stroke(YELLOW, 3)
        self.add(axes)
        self.play(ShowCreation(graph))
        self.wait()


class DiscreteGraph(InteractiveScene):
    def construct(self):
        # Test
        axes = Axes((0, 5), (0, 5), width=4, height=4)
        graph = axes.get_graph(
            lambda x: np.floor(x) + 0.5,
            discontinuities=np.arange(0, 8),
            x_range=(0, 4.99),
        )
        graph.set_stroke(RED, 5)
        self.add(axes)
        self.play(ShowCreation(graph))
        self.wait()


class LightQuantumWrapper(VideoWrapper):
    title = "Some light quantum mechanics"


class RedFilter(InteractiveScene):
    def construct(self):
        image = ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/barber_pole/images/RedFilter.jpg")
        image.set_height(FRAME_HEIGHT)

        # self.add(image)

        plane = NumberPlane()
        plane.fade(0.75)

        # Pixel indicator
        indicator = self.get_pixel_indicator(image)
        indicator.move_to(plane.c2p(5.25, -0.25), DOWN)

        self.add(indicator)

        # Move around
        self.wait()
        self.play(
            indicator.animate.move_to(plane.c2p(1.5, -0.125), DOWN),
            run_time=6,
        )
        self.wait()
        self.play(
            indicator.animate.move_to(plane.c2p(-3.5, 0), DOWN),
            run_time=6,
        )
        self.wait()

    def get_pixel_indicator(self, image, vect_len=2.0, direction=DOWN, square_size=1.0):
        vect = Vector(vect_len * direction, stroke_color=WHITE)
        square = Square(side_length=square_size)
        square.set_stroke(WHITE, 1)
        square.next_to(vect, -direction)

        def get_color():
            points = vect.get_end() + 0.05 * compass_directions(12)
            rgbs = np.array([image.point_to_rgb(point) for point in points])
            return rgb_to_color(rgbs.mean(0))

        square.add_updater(lambda s: s.set_fill(get_color(), 1))
        return VGroup(square, vect)


class LengthsOnDifferentColors(InteractiveScene):
    def construct(self):
        # Setup
        image = ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/barber_pole/images/RainbowTubes.jpg")
        image.set_height(FRAME_HEIGHT)
        # self.add(image)

        plane = NumberPlane()
        plane.fade(0.75)
        # self.add(plane)

        # Rectangles
        rects = Rectangle().replicate(4)
        rects.set_stroke(width=0)
        rects.set_fill(BLACK, opacity=0)
        rects.set_height(2)
        rects.set_width(FRAME_WIDTH, stretch=True)
        rects.arrange(DOWN, buff=0)

        # Braces
        lines = VGroup(
            Line(plane.c2p(3.2, 2.8), plane.c2p(-0.5, 2.8)),
            Line(plane.c2p(3.2, 0.9), plane.c2p(0.6, 0.9)),
            Line(plane.c2p(3.2, -1.0), plane.c2p(1.0, -1.0)),
            Line(plane.c2p(3.2, -3.0), plane.c2p(1.3, -3.0)),
        )
        braces = VGroup(*(
            Brace(line, DOWN, buff=SMALL_BUFF)
            for line in lines
        ))
        braces.set_backstroke(BLACK, 3)
        numbers = VGroup(*(
            DecimalNumber(line.get_length() / 7, font_size=36, unit=R" \text{m}")
            for line in lines
        ))
        for number, brace in zip(numbers, braces):
            number.next_to(brace, DOWN, buff=SMALL_BUFF)

        # Show braces
        for brace, rect, number in zip(braces, rects, numbers):
            other_rects = VGroup(*(r for r in rects if r is not rect))
            self.play(
                GrowFromPoint(brace, brace.get_right()),
                CountInFrom(number, 0),
                UpdateFromFunc(VGroup(), lambda m: number.next_to(brace, DOWN, SMALL_BUFF)),
                rect.animate.set_opacity(0),
                other_rects.animate.set_opacity(0.8),
            )
            self.wait(2)
        self.play(FadeOut(rects))

        # Ribbons
        axes_3d = ThreeDAxes((0, 6))
        ribbons = Group()
        twist_rates = [1.0 / PI / line.get_length() for line in lines]
        twist_rates = [0.09, 0.11, 0.115, 0.12]
        for line, twist_rate in zip(lines, twist_rates):
            ribbon = TwistedRibbon(
                axes_3d,
                amplitude=0.25,
                twist_rate=twist_rate,
                color=rgb_to_color(image.point_to_rgb(line.get_start())),
            )
            ribbon.rotate(PI / 2, RIGHT)
            ribbon.set_opacity(0.75)
            ribbon.flip(UP)
            ribbon.next_to(line, UP, MED_LARGE_BUFF, aligned_edge=RIGHT)
            ribbons.add(ribbon)

        for ribbon in ribbons:
            self.play(ShowCreation(ribbon, run_time=2))
        self.wait()


class AskAboutDiagonal(InteractiveScene):
    def construct(self):
        randy = Randolph(height=2)
        self.play(randy.says("Why diagonal?", mode="maybe", look_at=DOWN))
        self.play(Blink(randy))
        self.wait()


class AskNoteVerticalVariation(RedFilter):
    def construct(self):
        image = ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/barber_pole/images/GreenFilter.jpg")
        image.set_height(FRAME_HEIGHT)
        # self.add(image)

        plane = NumberPlane()
        plane.fade(0.75)
        # self.add(plane)

        indicator = self.get_pixel_indicator(image)

        # Scan horizontally, then vertically
        indicator.move_to(plane.c2p(-0.5, -0.5), DOWN)

        lr_arrows = VGroup(Vector(LEFT), Vector(RIGHT))
        lr_arrows.arrange(RIGHT, buff=1.0)
        lr_arrows.move_to(plane.c2p(0.5, -1.5))
        ud_arrows = VGroup(Vector(UP), Vector(DOWN))
        ud_arrows.arrange(DOWN, buff=1.0)
        ud_arrows.move_to(plane.c2p(0, -0.6))

        self.add(indicator)
        self.play(
            FadeIn(lr_arrows, time_span=(0, 1)),
            indicator.animate.move_to(plane.c2p(3, -0.5), DOWN),
            run_time=3
        )
        self.play(indicator.animate.move_to(plane.c2p(1.5, -0.5), DOWN), run_time=3)
        self.wait()
        self.play(
            FadeIn(ud_arrows, time_span=(0, 1)),
            FadeOut(lr_arrows, time_span=(0, 1)),
            indicator.animate.move_to(plane.c2p(1.5, -0.2), DOWN),
            run_time=3
        )
        self.play(indicator.animate.move_to(plane.c2p(1.5, -0.9), DOWN), run_time=3)
        self.play(indicator.animate.move_to(plane.c2p(1.5, -0.5), DOWN), run_time=3)
        self.wait()


class CombineColors(InteractiveScene):
    def construct(self):
        # Get images
        folder = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/barber_pole/images/"
        images = Group(*(
            ImageMobject(os.path.join(folder, ext + "Still.jpg"))
            for ext in ["Red", "Orange", "Green", "Blue", "Rainbow"]
        ))
        colors = images[:4]
        rainbow = images[4]
        colors.set_height(FRAME_HEIGHT / 2)
        colors.arrange_in_grid(buff=0)
        rainbow.set_height(FRAME_HEIGHT)

        self.add(*colors)


class SteveMouldMention(InteractiveScene):
    def construct(self):
        morty = Mortimer(height=2)
        morty.to_edge(DOWN)
        self.play(
            morty.says("""
                This is the part that
                Steve Mould explains
                quite well, by the way
            """, mode="tease")
        )
        self.play(Blink(morty))
        self.wait()


class IndexOfRefraction(InteractiveScene):
    def construct(self):
        # Test
        equation = Tex(R"{\text{Speed in a vacuum} \over \text{Speed in water}} \approx 1.33")
        equation.shift(UP)
        rhs = equation[R"\approx 1.33"]
        equation.scale(0.75)
        rhs.scale(1 / 0.75, about_edge=LEFT)
        arrow = Vector(DOWN)
        arrow.next_to(rhs, DOWN, SMALL_BUFF)
        words = TexText("``Index of refraction''")
        words.next_to(arrow, DOWN, SMALL_BUFF)
        words.set_color(BLUE)

        self.play(FadeIn(equation, 0.5 * UP))
        self.wait()
        self.play(
            GrowArrow(arrow),
            Write(words)
        )
        self.wait()


class LayerKickBackLabel(InteractiveScene):
    def construct(self):
        layer_label = Text("Each layer kicks back the phase")
        layer_label.to_edge(UP)
        self.play(Write(layer_label))
        self.wait()


class SugarIsChiral(InteractiveScene):
    default_frame_orientation = (-33, 85)
    title = R"Sucrose $\text{C}_{12}\text{H}_{22}\text{O}_{11}$ "
    subtitle = "(D-Glucose + D-Fructose)"
    molecule_height = 3

    def construct(self):
        axes = ThreeDAxes()
        axes.set_stroke(opacity=0.5)
        # self.add(axes)

        # Set up
        frame = self.frame
        title = VGroup(
            TexText(self.title),
            TexText(self.subtitle),
        )
        title[1].scale(0.7).set_color(GREY_A)
        title.arrange(DOWN)
        title.fix_in_frame()
        title.to_edge(UP, buff=0.25)

        sucrose = Sucrose()
        sucrose.set_height(self.molecule_height)

        # Introduce
        frame.reorient(24, 74, 0)
        self.add(title)
        self.play(
            FadeIn(sucrose, scale=5),
        )
        self.play(
            self.frame.animate.reorient(-16, 75, 0).set_anim_args(run_time=6)
        )
        self.add(sucrose)
        self.wait()

        # Show mirror image
        mirror = Square(side_length=6)
        mirror.set_fill(BLUE, 0.35)
        mirror.set_stroke(width=0)
        mirror.rotate(PI / 2, UP)
        mirror.set_shading(0, 1, 0)
        mirror.stretch(3, 1)

        sucrose.target = sucrose.generate_target()
        sucrose.target.next_to(mirror, LEFT, buff=1.0)

        self.add(mirror, sucrose)
        self.play(
            frame.animate.move_to(0.75 * OUT),
            MoveToTarget(sucrose),
            FadeIn(mirror),
            title.animate.scale(0.75).match_x(sucrose.target).set_y(1.75),
            run_time=1.5,
        )

        mirror_image = sucrose.copy()
        mirror_image.target = mirror_image.generate_target()
        mirror_image.target.stretch(-1, 0, about_point=mirror.get_center())

        mirror_words = Text("(mirror image)", font_size=36, color=GREY_A)
        mirror_words.fix_in_frame()
        mirror_words.match_x(mirror_image.target)
        mirror_words.match_y(title)

        self.add(mirror_image, mirror, sucrose)
        self.play(
            MoveToTarget(mirror_image),
            FadeIn(mirror_words),
        )

        # Chiral definition
        definition = TexText(R"Chiral $\rightarrow$ Cannot be superimposed onto its mirror image")
        definition.fix_in_frame()
        definition.to_edge(UP)

        sucrose.add_updater(lambda m, dt: m.rotate(10 * DEGREES * dt, axis=OUT))
        mirror_image.add_updater(lambda m, dt: m.rotate(-10 * DEGREES * dt, axis=OUT))
        self.play(Write(definition))
        self.play(
            self.frame.animate.reorient(-8, 76, 0),
            run_time=15,
        )
        self.wait(15)


class SimplerChiralShape(InteractiveScene):
    default_frame_orientation = (0, 70)

    def construct(self):
        # Ribbon
        frame = self.frame
        frame.set_field_of_view(1 * DEGREES)
        axes = ThreeDAxes((-3, 3))
        ribbon = TwistedRibbon(axes, amplitude=1, twist_rate=-0.35)
        ribbon.rotate(PI / 2, DOWN)
        ribbon.set_color(RED)
        ribbon.set_opacity(0.9)
        ribbon.set_shading(0.5, 0.5, 0.5)
        always(ribbon.sort_faces_back_to_front, UP)
        ribbon.set_x(-4)

        mirror_image = ribbon.copy()
        mirror_image.stretch(-1, 0)
        mirror_image.set_x(-ribbon.get_x())
        mirror_image.set_color(YELLOW_C)
        mirror_image.set_opacity(0.9)

        # Title
        spiral_name = Text("Spiral")
        mirror_name = Text("Mirror image")
        for name, mob in [(spiral_name, ribbon), (mirror_name, mirror_image)]:
            name.fix_in_frame()
            name.to_edge(UP)
            name.match_x(mob)

        self.play(
            FadeIn(spiral_name, 0.5 * UP),
            ShowCreation(ribbon, run_time=3),
        )
        self.wait()
        self.play(
            ReplacementTransform(ribbon.copy().shift(0.1 * DOWN), mirror_image),
            FadeTransformPieces(spiral_name.copy(), mirror_name),
        )
        self.wait()

        # Reorient
        r_copy = ribbon.copy()
        self.play(r_copy.animate.next_to(mirror_image, LEFT))
        self.play(Rotate(r_copy, PI, RIGHT, run_time=2))
        self.play(Rotate(r_copy, PI, OUT, run_time=2))
        self.play(Rotate(r_copy, PI, UP, run_time=2))
        self.wait()


class SucroseAction(InteractiveScene):
    just_sucrose = False

    def construct(self):
        # Sucrose
        sucrose = Sucrose(height=1.5)
        sucrose.balls.scale_radii(0.5)
        sucrose.rotate(PI / 2, RIGHT)
        sucrose.to_edge(LEFT)
        sucrose.add_updater(lambda m, dt: m.rotate(10 * DEGREES * dt, UP))
        if self.just_sucrose:
            self.add(sucrose)
            self.wait(36)
            return

        # Arrows
        arrows = VGroup()
        words = VGroup(
            Text("The amount that sugar\nslows this light..."),
            Text("...is different from how\nit slows this light"),
        )
        words.scale(0.75)
        for sign, word in zip([+1, -1], words):
            arrow = Line(ORIGIN, 5 * RIGHT + 1.5 * sign * UP, path_arc=-sign * 60 * DEGREES)
            arrow.insert_n_curves(100)
            arrow.add_tip(length=0.2, width=0.2)
            arrow.shift(sucrose.get_right() + 0.5 * LEFT + sign * 1.25 * UP)
            arrows.add(arrow)
            word.next_to(arrow, sign * UP, buff=0.1)

            self.play(
                ShowCreation(arrow),
                FadeIn(word, 0.25 * sign * UP)
            )
            self.wait(2)


class SucroseActionSucrosePart(SucroseAction):
    just_sucrose = True


class ThatSeemsIrrelevant(TeacherStudentsScene):
    def construct(self):
        stds = self.students
        morty = self.teacher
        self.play(
            stds[0].says("That seems irrelevant", mode="sassy"),
            self.change_students(None, "erm", "hesitant", look_at=stds[0].eyes),
            morty.change("guilty"),
        )
        self.wait(2)
        self.play(
            stds[0].debubble(mode="raise_left_hand", look_at=self.screen),
            self.change_students(None, "pondering", "pondering", look_at=self.screen),
        )
        self.wait(5)


class BigPlus(InteractiveScene):
    def construct(self):
        brace = Brace(Line(2 * DOWN, 2 * UP), RIGHT)
        brace.set_height(7)
        brace.move_to(2 * LEFT)
        plus = Tex("+", font_size=90)
        plus.next_to(brace, LEFT, buff=2.5)
        equals = Tex("=", font_size=90)
        equals.next_to(brace, RIGHT, buff=1.0)

        self.add(plus, brace, equals)


class CurvyCurvyArrow(InteractiveScene):
    def construct(self):
        # Test
        kw = dict(path_arc=-PI, buff=0.2, stroke_width=30, tip_width_ratio=4)
        arrows = VGroup(
            Arrow(DOWN, UP, **kw),
            Arrow(UP, DOWN, **kw),
        )
        arrows.set_color(WHITE)
        arrows.set_height(5)
        self.frame.reorient(-9, 75, 90)
        self.frame.set_field_of_view(1 * DEGREES)
        self.add(arrows)


class GlowDot(InteractiveScene):
    def construct(self):
        self.add(GlowDot(radius=3, color=WHITE))

        mid_point = 0.85 * LEFT
        mask = VMobject().set_points_as_corners([
            UR, mid_point, DR, DL, UL,
        ])
        mask.set_fill(BLACK, 1)
        mask.set_stroke(width=0)
        mask.set_height(20, about_point=mid_point)
        self.add(mask)


class Randy(InteractiveScene):
    def construct(self):
        self.add(Randolph(mode="confused"))


class EndScreen(PatreonEndScreen):
    scroll_time = 25
    show_pis = False
    title_text = ""
