from manim_imports_ext import *


class ShowPastVideos(InteractiveScene):
    def construct(self):
        # Show the video
        title = Text("2019 Video:", font_size=72)
        title[-1].set_opacity(0)
        title.to_edge(UP)
        background = FullScreenFadeRectangle()
        background.set_fill(GREY_E, 1)
        screen = ScreenRectangle().set_height(6)
        screen.next_to(title, DOWN)
        screen.set_fill(BLACK, 1).set_stroke(WHITE, 2)

        vertical_frame = screen.copy().set_shape(7 * 9 / 16, 7)
        vertical_frame.to_corner(UR)
        vertical_frame.match_style(screen)

        self.add(background, screen)
        self.play(
            Write(title),
            VShowPassingFlash(screen.copy().set_fill(opacity=0).set_stroke(BLUE, 4).insert_n_curves(20), time_width=1.5, run_time=3),
        )
        self.wait()

        # Three versions
        versions = VGroup(
            Text("2019 video: "),
            Text("Adapted as a short:\n(separate channel) ", alignment="LEFT"),
            Text("Re-posted to this channel: "),
        )
        versions[1]["(separate channel)"].set_fill(GREY_B)
        versions.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        versions.to_corner(UL)

        counts = VGroup(
            Text("13.8M Views").set_color(YELLOW),
            Text("42.6M Views").set_color(ORANGE),
            Text("62.0M Views").set_color(RED),
        )
        for version, count in zip(versions, counts):
            count.next_to(version, RIGHT, aligned_edge=UP)

        explanation = VGroup(
            Text("My only reason for making shorts at all"),
            Text("is to pique the curiosity of people in the shorts feed and"),
            Text("encourage them to pop out for a full explanation. Originally I didn't"),
            Text("want to trash up this channel with shorts, but then the only way to link to"),
            Text("link to a long-form videos was if the short lived on the same channel.  ¯\\_(ツ)_/¯"),
        )
        for part in explanation:
            part.match_width(versions[2])
        explanation.arrange(DOWN, buff=0.15)
        explanation.set_fill(GREY_B)
        explanation.next_to(versions[2], DOWN, buff=0.5)

        self.play(
            Transform(title, versions[0]),
            FadeIn(counts[0]),
            Transform(screen, vertical_frame)
        )
        for i in [1, 2]:
            self.play(
                FadeIn(versions[i]),
                FadeIn(counts[i], lag_ratio=0.1,)
            )
        self.play(FadeIn(explanation, lag_ratio=0.01, run_time=1))
        self.wait()


class ConfettiSpiril(Animation):
    x_start = 0
    spiril_radius = 0.5
    num_spirals = 4
    run_time = 10
    rate_func = None

    def __init__(self, mobject, **kwargs):
        x_start = kwargs.pop("x_start", self.x_start)
        self.num_spirals = kwargs.pop("num_spirals", self.num_spirals)
        mobject.next_to(x_start * RIGHT + FRAME_Y_RADIUS * UP, UP)
        self.total_vert_shift = FRAME_HEIGHT + mobject.get_height() + 2 * MED_SMALL_BUFF

        super().__init__(mobject, **kwargs)

    def interpolate_submobject(self, submobject, starting_submobject, alpha):
        submobject.set_points(starting_submobject.get_points())

    def interpolate_mobject(self, alpha):
        Animation.interpolate_mobject(self, alpha)
        angle = alpha * self.num_spirals * TAU
        vert_shift = alpha * self.total_vert_shift

        start_center = self.mobject.get_center()
        self.mobject.shift(self.spiril_radius * OUT)
        self.mobject.rotate(angle, axis=UP, about_point=start_center + 0.5 * RIGHT)
        self.mobject.shift(vert_shift * DOWN)


class Confetti(InteractiveScene):
    def construct(self):
        # Test
        num_confetti_squares = 300
        colors = [RED, YELLOW, GREEN, BLUE, PURPLE, RED]
        confetti_squares = [
            Square(
                side_length=0.2,
                stroke_width=0,
                fill_opacity=0.75,
                fill_color=random.choice(colors),
            )
            for x in range(num_confetti_squares)
        ]
        confetti_spirils = [
            ConfettiSpiril(
                square,
                x_start=2 * random.random() * FRAME_X_RADIUS - FRAME_X_RADIUS,
                num_spirals=np.random.uniform(-5, 5),
            )
            for square in confetti_squares
        ]

        self.play(LaggedStart(*confetti_spirils, lag_ratio=1e-2, run_time=10))


class HappyPiDay(TeacherStudentsScene):
    def construct(self):
        # Test
        title = Text("Happy Pi Day!", font_size=72)
        title.to_edge(UP)
        morty = self.teacher
        morty.change_mode("surprised")

        self.play(
            Write(title),
            morty.change("hooray", look_at=title),
            self.change_students("hooray", "surprised", "jamming", look_at=title)
        )
        self.wait()
        self.play(
            morty.change("tease", 2 * UP),
            self.change_students("tease", "surprised", "well", look_at=3 * UR)
        )
        self.wait(4)


class Leftrightarrow(InteractiveScene):
    def construct(self):
        arrow = Tex(R"\longleftrightarrow", font_size=90)
        self.play(GrowFromCenter(arrow, run_time=2))
        self.wait(2)


class GroversAlgorithmLabel(InteractiveScene):
    def construct(self):
        labels = VGroup(
            TexText("Quantum Computing"),
            TexText("Grover's Algorithm"),
        )
        self.play(FadeIn(labels[0], 0.5 * UP))
        self.wait()
        self.play(
            labels[0].animate.shift(UP),
            FadeIn(labels[1], 0.5 * UP)
        )
        self.wait()


class UnsolvedReference(InteractiveScene):
    def construct(self):
        rect = Rectangle(8, 1.25)
        rect.set_stroke(RED, 4)
        label = Text("Unsolved", font_size=60)
        label.set_color(RED)
        label.next_to(rect, UP, buff=MED_SMALL_BUFF)

        self.play(ShowCreation(rect), FadeIn(label, 0.25 * UP))
        self.wait()


class Recap(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer()
        self.play(morty.says(TexText("Let's Recap"), look_at=DL))
        self.wait()


class RewindArrows(InteractiveScene):
    def construct(self):
        # Test
        arrows = ArrowTip(angle=PI).get_grid(1, 3)
        arrows.scale(2)
        self.play(LaggedStartMap(FadeIn, arrows, lag_ratio=0.1, shift=0.5 * LEFT, run_time=1))
        self.wait()


class CommentOnElastic(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.play(
            morty.says("Assume perfectly\nelastic collisions"),
            self.change_students("pondering", "erm", "tease", look_at=self.screen)
        )
        self.wait(2)
        self.play(
            stds[2].says(
                "Then there should\nbe no sound",
                mode="sassy",
                bubble_direction=LEFT,
                look_at=morty.eyes
            ),
            morty.debubble("guilty"),
        )
        self.play(self.change_students("erm", "angry", "sassy", look_at=morty.eyes))
        self.wait(3)


class WritePiDigits(InteractiveScene):
    def construct(self):
        eq = Tex(R"\pi = 3.14159265358 \dots")
        self.play(FadeIn(eq, lag_ratio=0.25, run_time=4))
        self.wait()


class ReactToQuantumComparisson(InteractiveScene):
    def construct(self):
        randy = Randolph(height=3)
        randy.to_edge(LEFT).shift(DOWN)
        randy.body.insert_n_curves(100)

        self.play(randy.change("sassy"))
        self.play(Blink(randy))
        self.wait(2)
        self.play(randy.change("awe"))
        self.play(Blink(randy))
        self.wait(2)
        self.play(randy.change("confused"))
        self.play(Blink(randy))
        self.wait()
        self.play(Blink(randy))
        self.wait()


class LoadSolutionIntoHead(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenFadeRectangle().set_fill(interpolate_color(GREY_E, BLACK, 0.5), 1))
        randy = Randolph()
        randy.to_edge(DOWN, buff=0.25)
        bubble = ThoughtBubble(direction=RIGHT, filler_shape=(4, 2.5)).pin_to(randy)
        bubble[0][-1].set_fill(GREEN_SCREEN, 1)

        self.play(
            ShowCreation(bubble),
            randy.change("concentrating", 3 * UR)
        )
        for _ in range(2):
            self.play(Blink(randy))
            self.wait(2)


class ShowMassRatioToCountChart(InteractiveScene):
    def construct(self):
        # Chart
        n_terms = 6
        v_line = Line(UP, DOWN).set_height(6)
        points = [v_line.pfp(a) for a in np.linspace(0, 1, n_terms + 2)]
        h_lines = VGroup(Line(LEFT, RIGHT).set_width(7).move_to(p) for p in points)

        VGroup(v_line, h_lines).set_stroke(WHITE, 2)
        self.add(v_line, h_lines[1:-1])

        # Content
        titles = VGroup(
            Text("Mass ratio"),
            Text("#Collisions"),
        )
        mass_ratios = VGroup(
            Integer(int(10**n), unit=":1", font_size=42)
            for n in range(0, 2 * n_terms, 2)
        )
        counts = VGroup(
            Integer(int(PI * 10**n), font_size=42)
            for n in range(0, n_terms)
        )
        for point, ratio, count in zip(points[2:], mass_ratios, counts):
            ratio.next_to(point, UL, buff=0.2).shift(0.1 * LEFT)
            count.next_to(point, UR, buff=0.2).shift(0.1 * RIGHT)

        titles[0].next_to(points[1], UL).shift(0.25 * LEFT)
        titles[1].next_to(points[1], UR).shift(0.25 * RIGHT)
        titles[0].align_to(titles[1], UP)

        self.add(titles)
        for ratio, count in zip(mass_ratios, counts):
            ratio.align_to(count, UP)
            self.play(FadeIn(ratio))
            self.wait()
            self.play(TransformFromCopy(ratio, count))
            self.wait()


class StateThePuzzle(InteractiveScene):
    def construct(self):
        # Test
        question = TexText(
            R"Given $m_1$ and $m_2$, how \\ many collisions take place?",
            t2c={R"$m_1$": BLUE, R"$m_2$": BLUE}
        )
        question.to_edge(UP)
        self.play(Write(question))
        self.wait()


class EnergyAndMomentumLaws(InteractiveScene):
    def construct(self):
        # Test
        laws = VGroup(
            Text("Conservation of Energy"),
            Text("Conservation of Momentum"),
        )
        laws.arrange(DOWN, buff=0.75)
        laws.move_to(3 * UP, UP)

        self.play(FadeIn(laws[0], UP))
        self.wait()
        self.play(
            TransformMatchingStrings(
                *laws,
                key_map={"Energy": "Momentum"},
                mismatch_animation=FadeTransform,
                run_time=1
            )
        )
        self.wait()


class ProblemSolvingPrinciplesWithPis(TeacherStudentsScene):
    def construct(self):
        # Prep
        morty = self.teacher
        stds = self.students
        for pi in [morty, *stds]:
            pi.body.insert_n_curves(100)

        # Title
        title = Text("Problem-solving principles", font_size=72)
        title.to_edge(UP)
        underline = Underline(title, buff=-0.1)
        VGroup(title, underline).set_color(BLUE)
        title.set_backstroke(width=3)

        # Items
        items = BulletedList(
            "Try a simpler version of the problem",
            "Use the defining features of the problem",
            "List any equations that might be relevant",
            "Seek symmetry",
            "Compute something (anything) to build intuition",
            "Run simulations (if possible) to build intuition",
            "Draw pictures!",
        )
        items.next_to(title, DOWN, buff=0.35)
        items.to_edge(LEFT, buff=0.75)
        items.save_state()

        # Have students toss up examples
        np.random.seed(0)
        for y, item in enumerate(items):
            item.set_height(0.25)
            item.set_opacity(0.8)
            item[0].set_opacity(0)
            item.next_to(underline, DOWN)
            item.shift(np.random.uniform(-5, 5) * RIGHT)
            item.shift(0.5 * y * DOWN)

        self.play(
            morty.change("raise_right_hand", underline),
            self.change_students("pondering", "pondering", "pondering", look_at=underline),
            FadeIn(title, lag_ratio=0.1),
            ShowCreation(underline),
        )
        self.play(morty.change("tease", stds))
        self.wait()
        index_mode_pairs = [
            (0, "raise_right_hand"),
            (2, "raise_right_hand"),
            (1, "raise_right_hand"),
            (2, "raise_left_hand"),
            (0, "raise_left_hand"),
            (1, "raise_left_hand"),
            (2, "raise_right_hand"),
        ]
        for item, pair in zip(items, index_mode_pairs):
            index, mode = pair
            if mode == "raise_left_hand":
                item.shift(2 * LEFT).shift_onto_screen()
            self.play(
                stds[index].change(mode, item),
                FadeIn(item, scale=3, shift=item.get_center() - stds[index].get_center())
            )
        self.wait(3)

        # Return to position
        self.play(
            Restore(items, run_time=2, lag_ratio=1e-3),
            FadeOut(self.pi_creatures),
            FadeOut(self.background),
        )
        self.wait()

        # Isolate a few items
        for index in [2, 6, 3]:
            self.play(
                items.animate.fade_all_but(index),
            )
            self.wait()


class StaysConstant(InteractiveScene):
    word = "Unchanged!"
    color = YELLOW

    def construct(self):
        # Note the change
        unchanged_label = Text(self.word)
        unchanged_label.set_color(self.color)
        unchanged_label.next_to(ORIGIN, DOWN, buff=1.5)
        unchanged_label.shift(0.5 * RIGHT)
        unchanged_arrow = Arrow(unchanged_label, ORIGIN, buff=0.25)
        unchanged_arrow.set_fill(self.color)

        self.play(
            FadeIn(unchanged_label),
            GrowArrow(unchanged_arrow),
        )
        self.wait()
        self.play(FadeOut(VGroup(unchanged_label, unchanged_arrow)))


class NoteChange(StaysConstant):
    word = "Changed!"
    color = RED


class SimpleArrow(InteractiveScene):
    def construct(self):
        arrow = Vector(0.5 * DL)
        self.play(GrowArrow(arrow))
        self.wait()


class KeyStep(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.teacher.says("This is the\nkey step"),
            self.change_students("pondering", "thinking", "erm", look_at=self.screen)
        )
        self.wait()
        self.play(self.change_students("thinking", "well", "pondering", look_at=self.screen))
        self.wait(3)


class StateSpaceLabel(InteractiveScene):
    def construct(self):
        # Test
        title = TexText("``State Space''", font_size=60)
        title.to_corner(UL)
        arrow = Arrow(title.get_bottom() + 0.1 * DOWN + 1.0 * RIGHT, 1.0 * UP, thickness=5)
        arrow.set_color(YELLOW)
        title.set_color(YELLOW)

        self.play(
            Write(title),
            GrowArrow(arrow)
        )
        self.wait()


class HoldUpEllipseVsCircle(InteractiveScene):
    def construct(self):
        # Show pi
        radius = 3
        circle = Circle(radius=radius)
        circle.set_stroke(WHITE, 3)

        radial_line = Line()
        radial_line.set_stroke(BLUE, 3)
        radial_line.f_always.put_start_and_end_on(lambda: ORIGIN, circle.get_end)
        radial_label = Tex(R"1", font_size=48)
        radial_label.add_updater(lambda m: m.move_to(
            radial_line.get_center() + 0.1 * rotate_vector(radial_line.get_vector(), -90 * DEG),
        ))

        theta_tracker = ValueTracker(0)
        get_theta = theta_tracker.get_value
        arc = always_redraw(lambda: Arc(0, get_theta(), radius=radius).set_stroke(RED, 5))
        arc_len_label = DecimalNumber(0, show_ellipsis=True, num_decimal_places=5)
        arc_len_label.set_color(RED)
        arc_len_label.f_always.set_value(get_theta)
        arc_len_label.f_always.move_to(lambda: 1.4 * arc.get_end())

        self.add(radial_line, radial_label)
        self.play(ShowCreation(circle, run_time=3))
        self.add(arc, arc_len_label)
        self.play(theta_tracker.animate.set_value(PI), run_time=3)
        self.wait()
        circle_group = VGroup(circle, radial_line, radial_label, arc, arc_len_label)
        circle_group.clear_updaters()

        # Make the comparison
        morty = Mortimer(mode="raise_right_hand").flip()
        morty.to_edge(DOWN)

        circle_group.target = circle_group.generate_target()
        circle_group.target.set_height(3)
        circle_group.target.next_to(morty.get_corner(UR), UP, buff=0.5)
        circle_group.target[1:].set_opacity(0)

        self.play(
            MoveToTarget(circle_group),
            VFadeIn(morty),
        )
        self.play(Blink(morty))
        self.wait()
        self.play(morty.change("raise_left_hand", look_at=2 * UL))
        for _ in range(2):
            self.play(Blink(morty))
            self.wait(2)


class AskWhy(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.students[2].says("...why?", mode="confused", bubble_direction=LEFT, look_at=self.screen),
            self.teacher.change("tease"),
            self.change_students("pondering", "thinking", look_at=self.screen)
        )
        self.wait()
        self.play(
            self.teacher.says("Symmetry breeds\ninsight!", mode="hooray"),
            self.students[2].debubble(look_at=self.teacher.eyes)
        )
        self.play(self.change_students("thinking", "tease", "erm", self.teacher.eyes))
        self.wait()


class PiTime1e5(InteractiveScene):
    def construct(self):
        # Test
        value = Integer(PI * 1e5, font_size=60)
        self.play(Write(value, lag_ratio=0.2))
        self.wait()


class HighlightTheSlope(TeacherStudentsScene):
    def construct(self):
        # Test
        slope = Tex(R"-\sqrt{m_1 \over m_2}", t2c={"m_1": BLUE, "m_2": BLUE}, font_size=60)
        slope.next_to(self.hold_up_spot, UP)
        rect = SurroundingRectangle(slope)

        sqrt_highlight = slope[R"\sqrt"][0].copy()
        sqrt_highlight.set_stroke(YELLOW, 4).set_fill(opacity=0)
        sqrt_highlight.insert_n_curves(100)

        self.play(
            self.teacher.change("raise_right_hand", slope),
            self.change_students("erm", "sassy", "well", look_at=slope),
            FadeIn(slope, UP),
        )
        self.wait()
        self.play(
            FlashAround(slope),
            ShowCreation(rect),
            self.change_students("pondering", "pondering", "thinking", look_at=slope)
        )
        self.wait()

        self.remove(rect)
        rect_copy = rect.replicate(2)
        self.play(Transform(rect_copy, sqrt_highlight))
        self.wait()
        self.play(FadeOut(rect_copy))
        self.wait(4)


class MostOfTheReasoning(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer(height=2.5).flip()
        morty.to_corner(DL)
        self.play(morty.says("This is most of\nthe physics"))
        self.play(Blink(morty))
        self.wait(2)


class StareAtDiagram(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            self.change_students("well", "happy", look_at=self.screen),
            stds[2].says("I can see\nmyself there", mode="tease", look_at=self.screen, bubble_direction=LEFT),
            morty.change("tease")
        )
        self.wait(4)
        self.play(
            stds[2].debubble(mode="pondering", look_at=3 * UP),
            self.change_students("pondering", "pondering", look_at=3 * UP),
            morty.change("raise_right_hand"),
            FadeOut(self.background),
        )
        self.wait()
        self.play(stds[1].change("concentrating", look_at=3 * UP))
        self.wait()
        self.play(LaggedStart(
            stds[1].change("hooray"),
            stds[0].change("erm", look_at=stds[1].eyes),
            stds[2].change("erm", look_at=stds[1].eyes),
            morty.change("tease", look_at=stds[1].eyes),
            lag_ratio=0.25,
        ))
        self.wait()


class AskHowThisIsHelpful(InteractiveScene):
    def construct(self):
        randy = Randolph()
        randy.to_corner(DL)
        self.play(randy.says("Why is this\nhelpful?"))
        self.play(Blink(randy))
        self.wait()
        self.play(Blink(randy))
        self.wait()


class ISeeWhereThisIsGoing(TeacherStudentsScene):
    def construct(self):
        # Setup
        morty = self.teacher
        stds = self.students
        std = stds[1]
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)

        self.play(
            morty.change("raise_right_hand"),
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen)
        )
        self.wait()
        self.play(std.says("I see!", mode="hooray"))
        self.play(morty.change("hesitant"))

        # Show the progression
        mass_terms = VGroup(
            Tex(R"m_1 = 100^n", t2c={"m_1": BLUE}),
            Tex(R"m_2 = 1", t2c={"m_2": BLUE}),
        )
        mass_terms.arrange(DOWN, aligned_edge=LEFT)
        mass_terms.next_to(std.get_corner(UL), UP, buff=0.5)

        implies = Tex(R"\Longrightarrow", font_size=72)
        implies.next_to(mass_terms, RIGHT)
        q_mark = Tex(R"?").next_to(implies, UP, SMALL_BUFF)
        theta_eq = Tex(R"\theta=(0.1)^n")
        theta_eq.next_to(implies, RIGHT)

        group = VGroup(mass_terms, implies, q_mark, theta_eq)

        self.play(
            std.debubble(mode="raise_left_hand", look_at=mass_terms),
            FadeIn(mass_terms, UL, scale=2),
            stds[0].change("hesitant", mass_terms),
            stds[2].change("erm", mass_terms),
        )
        self.wait()
        self.play(
            std.change("raise_right_hand", look_at=theta_eq),
            Write(implies),
            Write(q_mark),
            FadeTransform(mass_terms[0].copy(), theta_eq),
            stds[0].animate.look_at(theta_eq),
            stds[2].animate.look_at(theta_eq),
        )
        self.wait()
        self.play(
            std.change("hooray", group),
            group.animate.set_height(0.75).to_edge(UP),
        )
        self.wait(3)
        self.play(
            morty.says("Almost", mode="well"),
            self.change_students("pondering", "angry", "confused", look_at=morty.eyes)
        )
        self.wait(3)

        # Final statement
        self.play(
            FadeOut(morty.bubble),
            morty.says("One final bit\nof reasoning", mode="speaking"),
            self.change_students("thinking", "hesitant", "happy"),
        )
        self.wait(3)


class ReferenceSmallAngleApproximations(TeacherStudentsScene):
    def construct(self):
        self.add_title()
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)

        # Test
        equation1 = Tex(R"\arctan(x) \approx x")
        equation1[R"\arctan"].set_color(GREEN)
        equation2 = Tex(R"x \approx \tan(x)")
        equation2[R"\tan"].set_color(YELLOW)
        for mob in equation1, equation2:
            mob.move_to(self.hold_up_spot, DOWN)

        self.play(
            FadeIn(equation1, UP),
            self.teacher.change("raise_right_hand"),
            self.change_students(
                "erm", "sassy", "confused"
            )
        )
        self.look_at(3 * UL)
        self.play(equation1.animate.shift(UP))
        self.play(
            TransformMatchingTex(
                equation1.copy(),
                equation2,
                lag_ratio=0.01,
                key_map={R"\arctan": R"\tan"}
            )
        )
        self.play(self.change_students("confused", "erm", "sassy"))
        self.look_at(3 * UL)
        self.wait(3)

    def add_title(self):
        title = TexText("For small $x$")
        subtitle = TexText("(e.g. $x = 0.001$)")
        subtitle.scale(0.75)
        subtitle.next_to(title, DOWN)
        title.add(subtitle)
        title.move_to(self.hold_up_spot)
        title.to_edge(UP)
        self.add(title)


class ExplainSmallAngleApprox(InteractiveScene):
    def construct(self):
        # Setup
        frame = self.frame
        height = 7.5
        axes = Axes(
            (-2, 2),
            (-2, 2),
            height=height,
            width=height,
            axis_config=dict(include_tip=True)
        )
        axes.shift(-axes.get_origin())
        unit_size = axes.x_axis.get_unit_size()

        circle = Circle(radius=unit_size)
        circle.set_stroke(WHITE, 3)
        point = GlowDot(color=RED)
        point.move_to(circle.get_right())

        radial_line = Line()
        radial_line.set_stroke(BLUE, 3)
        radial_line.f_always.put_start_and_end_on(
            axes.get_origin,
            point.get_center,
        )
        radial_label = Tex(R"1", font_size=36)
        radial_label.add_updater(lambda m: m.move_to(
            radial_line.get_center() + 0.15 * rotate_vector(radial_line.get_vector(), -90 * DEG),
        ))

        self.add(axes)
        self.add(radial_line)
        self.add(radial_label)
        self.play(
            ShowCreation(circle),
            UpdateFromFunc(point, lambda m: m.move_to(circle.get_end())),
            frame.animate.set_height(5),
            run_time=3
        )
        self.wait()

        # Show angle
        theta_color = YELLOW
        x_color = RED
        y_color = GREEN

        h_radial_line = radial_line.copy().clear_updaters()
        radial_label.clear_updaters()
        theta_tracker = ValueTracker(1e-2)
        get_theta = theta_tracker.get_value
        point.add_updater(lambda m: m.move_to(unit_size * rotate_vector(RIGHT, get_theta())))

        arc = always_redraw(lambda: Arc(0, get_theta(), radius=0.5))
        theta_label = Tex(R"\theta", font_size=36)
        theta_label.set_color(theta_color)
        theta_label.f_always.set_height(lambda: clip(arc.get_height(), 1e-2, 0.30))
        theta_label.add_updater(lambda m: m.next_to(arc.pfp(0.65), RIGHT, buff=0.075))

        tan_eq = Tex(
            R"\tan(\theta) = {{y} \over {x}} \approx {\theta \over 1}",
            t2c={R"\theta": theta_color, "{x}": x_color, "{y}": y_color},
        )
        tan_eq.fix_in_frame()
        tan_eq.to_corner(UR).shift(LEFT)

        self.add(h_radial_line)
        self.add(arc)
        self.add(theta_label)
        self.play(
            theta_tracker.animate.set_value(30 * DEG),
            radial_line.animate.set_stroke(WHITE, 3),
            run_time=1
        )
        self.play(
            TransformFromCopy(theta_label.copy().clear_updaters(), tan_eq[R"\theta"][0]),
            Write(tan_eq[R"\tan("]),
            Write(tan_eq[R") ="]),
        )
        self.wait()

        # Show lines
        x_line = Line().set_stroke(x_color, 4)
        y_line = Line().set_stroke(y_color, 4)
        x_line.add_updater(lambda m: m.put_start_and_end_on(
            axes.get_origin(), axes.c2p(math.cos(get_theta()), 0)
        ))
        y_line.add_updater(lambda m: m.put_start_and_end_on(
            axes.c2p(math.cos(get_theta()), 0),
            axes.c2p(math.cos(get_theta()), math.sin(get_theta()))
        ))

        x_label = Tex(R"x", font_size=30).set_color(x_color)
        y_label = Tex(R"y", font_size=30).set_color(y_color)
        VGroup(x_label, y_label).set_backstroke(BLACK, 3)
        x_label.always.next_to(x_line, DOWN, SMALL_BUFF)
        y_label.always.next_to(y_line, RIGHT, buff=0.05)

        self.play(LaggedStart(
            ShowCreation(y_line, suspend_mobject_updating=True),
            VFadeIn(y_label),
            FadeOut(h_radial_line),
            FadeOut(radial_label),
            ShowCreation(x_line, suspend_mobject_updating=True),
            VFadeIn(x_label),
            lag_ratio=0.15
        ))
        self.play(LaggedStart(
            TransformFromCopy(y_label.copy().clear_updaters(), tan_eq["{y}"][0]),
            Write(tan_eq[R"\over"][0]),
            TransformFromCopy(x_label.copy().clear_updaters(), tan_eq["{x}"][0]),
            lag_ratio=0.5
        ))
        self.wait()

        # Shrink down
        self.play(theta_tracker.animate.set_value(0.1), run_time=3)
        self.wait()

        # Analogy
        lil_rect = SurroundingRectangle(tan_eq["{x}"])
        self.play(
            Write(tan_eq[R"\approx"]),
            TransformFromCopy(tan_eq[R"\over"][0], tan_eq[R"\over"][1]),
            ShowCreation(lil_rect),
        )
        self.play(
            TransformFromCopy(tan_eq["{x}"][0], tan_eq["1"][0]),
            lil_rect.animate.surround(tan_eq["1"]),
        )
        self.wait()
        self.play(
            lil_rect.animate.surround(tan_eq["{y}"]),
            y_label.animate.scale(0.5),
            x_label.animate.scale(0.5),
            point.animate.set_radius(0.25 * point.get_radius()),
            frame.animate.reorient(0, 0, 0, (1.46, -0.0, 0.0), 1.87).set_anim_args(run_time=3),
        )
        self.play(
            TransformFromCopy(tan_eq["{y}"][0], tan_eq[R"\theta"][1]),
            lil_rect.animate.surround(tan_eq[R"\theta"][1]),
        )
        self.play(FadeOut(lil_rect))

        # More shrinking!
        self.play(theta_tracker.animate.set_value(1e-2), run_time=8)


class AngryStudents(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.change_students("angry", "confused", "sassy"),
            self.teacher.change("guilty"),
        )
        self.wait(3)


class DigitsOfPi(InteractiveScene):
    def construct(self):
        # Test
        pi_text = Path(Path(__file__).parent, "digits_of_pi.txt").read_text()
        pi_text_lines = pi_text.split("\n")
        one_line_pi = Tex(R"\pi = " + pi_text_lines[0])
        one_line_pi[0].scale(1.5)
        self.play(FadeIn(one_line_pi, lag_ratio=0.25, run_time=5))
        self.wait()

        # Second half
        index = 25
        nines = Tex(R"\dots999999999999999999999 \dots")
        nines.next_to(one_line_pi[:index], RIGHT, buff=SMALL_BUFF, aligned_edge=DOWN)

        prefix_brace = Brace(VGroup(one_line_pi[2:index], nines[:3]), UP, MED_SMALL_BUFF)
        prefix_label = prefix_brace.get_text("First n digits")
        nines_brace = Brace(nines[3:], UP, MED_SMALL_BUFF)
        nines_label = nines_brace.get_text("Next n digits")

        self.play(
            FadeOut(one_line_pi[index:]),
            FadeIn(nines, lag_ratio=0.25, run_time=3),
        )
        self.wait()
        self.play(LaggedStart(
            GrowFromCenter(prefix_brace),
            Write(prefix_label),
            GrowFromCenter(nines_brace),
            Write(nines_label),
            lag_ratio=0.25
        ))
        self.wait()

        # Show full pi
        frame = self.frame
        pi_tex = pi_text[2:].replace("\n", R"\\&")
        full_pi = Tex(R"\pi = 3.&" + pi_tex)
        full_pi.to_edge(UP)
        dots = Tex(R"\vdots", font_size=72)
        dots.next_to(full_pi, DOWN)
        digits_per_line = len(pi_text_lines[-1].strip())
        big_nines = Text("\n".join([
            "9" * digits_per_line
            for line in pi_text_lines
        ]))
        big_nines.match_width(full_pi[-digits_per_line:])
        big_nines.next_to(dots, DOWN)
        big_nines.align_to(full_pi, RIGHT)
        big_nines.set_color(RED)

        self.play(
            ReplacementTransform(one_line_pi[:index], full_pi[:index]),
            FadeIn(full_pi[index:], lag_ratio=1e-3, time_span=(0.5, 3)),
            FadeOut(VGroup(prefix_brace, prefix_label, nines_brace, nines_label, nines)),
            FadeIn(dots, time_span=(2, 3))
        )
        self.wait()
        self.play(
            frame.animate.scale(1.8, about_edge=UP),
            FadeIn(big_nines, lag_ratio=1e-2),
            run_time=6
        )
        self.wait()


class WriteExactSolution(InteractiveScene):
    def construct(self):
        # Title
        title = Text("Answer", font_size=72)
        title.to_edge(UP)
        underline = Underline(title, stretch_factor=1.5)
        VGroup(title, underline).set_color(YELLOW)
        self.play(
            FadeIn(title, lag_ratio=0.1),
            ShowCreation(underline)
        )

        # Answer
        kw = dict(
            t2c={
                R"\theta": YELLOW,
                R"m_1": BLUE,
                R"m_2": BLUE,
            }
        )
        solution = VGroup(
            Tex(R"\#\text{Collisions} = \lceil \pi / \theta - 1 \rceil", **kw),
            Tex(R"\text{Where } \; \theta = \text{arctan}\left(\sqrt{m_2 / m_1}\right)", **kw),
        )
        solution.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        solution.next_to(underline, DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)

        self.play(LaggedStartMap(FadeIn, solution, shift=0.5 * DOWN, lag_ratio=0.5))
        self.wait()


class ExactSolutionForMatt(InteractiveScene):
    def construct(self):
        # Answer
        kw = dict(
            t2c={
                R"\theta": YELLOW,
                R"m_1": BLUE,
                R"m_2": BLUE,
                R"\pi": WHITE
            }
        )
        equations = VGroup(
            Tex(R"\#\text{Collisions} = {\pi \over \text{arctan}\left(\sqrt{m_2 / m_1}\right)}", **kw),
            Tex(R"\#\text{Collisions} \approx {\pi \over \sqrt{m_2 / m_1}}", **kw),
            Tex(R"\#\text{Collisions} \approx \pi \cdot \sqrt{m_1 / m_2}", **kw),
        )
        for eq in equations[:2]:
            eq[R"\pi"].scale(2, about_edge=DOWN)
        equations[2][R"\pi"].scale(1.5, about_edge=DOWN).shift(0.05 * RIGHT)

        rect = SurroundingRectangle(equations[0][len("#Collisions="):])
        rect.set_stroke(YELLOW, 2)
        words = VGroup(
            Text("If fractional, round down").set_color(YELLOW),
            TexText("If whole...").set_color(GREEN),
        )
        for word in words:
            word.next_to(rect, UP)

        minus_1 = Tex(R"-1", font_size=60)
        minus_1.next_to(equations[0][R"\over"], RIGHT)

        self.play(FadeIn(equations[0], lag_ratio=0.25))
        self.wait()
        self.play(ShowCreation(rect), FadeIn(words[0], 0.25 * UP))
        self.wait()
        self.play(
            rect.animate.set_color(GREEN),
            FadeTransformPieces(words[0], words[1], run_time=1),
        )
        self.play(Write(minus_1))
        self.wait()
        self.play(LaggedStart(FadeOut(rect), FadeOut(words[1]), FadeOut(minus_1)))

        # Pi creature
        randy = Randolph()
        pi = equations[0][R"\pi"][0][0]
        randy.scale(pi.get_height() / randy.body.get_height())
        randy.shift(pi.get_center() - randy.body.get_center())

        self.play(
            FadeOut(pi),
            FadeIn(randy)
        )
        self.play(randy.change("hooray", DOWN))
        self.play(Blink(randy))
        self.play(FadeOut(randy), FadeIn(pi))

        # Cross out arctan
        exmark = Cross(equations[0][R"\text{arctan}"])
        self.play(ShowCreation(exmark))
        self.wait()
        self.play(
            FadeOut(exmark),
            TransformMatchingTex(equations[0], equations[1]),
        )
        self.wait()
        self.play(
            TransformMatchingTex(
                equations[1],
                equations[2],
                matched_keys=["m_1", "m_2", R"\sqrt"],
                path_arc=45 * DEG
            )
        )
        self.wait()


class WhoCares(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.play(
            stds[2].says("Question...", mode="dance_3"),
            self.change_students("pondering", "pondering", look_at=stds[2].eyes),
            morty.change('tease'),
        )
        self.wait()
        old_bubble = stds[2].bubble
        bubble = stds[2].get_bubble("Who cares?", bubble_type=SpeechBubble)
        self.play(LaggedStart(
            FadeTransformPieces(old_bubble, bubble),
            stds[2].change("angry"),
            self.change_students("guilty", "hesitant"),
            morty.change("hesitant"),
            lag_ratio=0.25
        ))
        self.play(LaggedStart(
            stds[1].animate.look_at(stds[0].eyes),
            stds[0].change("maybe")
        ))
        self.play(Blink(stds[0]))
        self.wait(5)


class SimplifyingMessiness(InteractiveScene):
    samples = 4

    def construct(self):
        # Test
        circle = Circle(radius=2)
        circle.set_stroke(WHITE, 3)

        rough_points = np.array([np.random.uniform(0.95, 1.05) * circle.pfp(a) for a in np.linspace(0, 1, 200)])
        rough_points[-1] = rough_points[0]
        rough_circle = VMobject().set_points_as_corners(rough_points)

        circles = VGroup(rough_circle, circle)
        circles.arrange(RIGHT, buff=3.0)

        arrows = VGroup(
            Arrow(rough_circle.get_right(), circle.get_left(), path_arc=-sgn * 90 * DEG, thickness=5).shift(sgn * UP)
            for sgn in [+1, -1]
        )
        mid_arrow = Arrow(rough_circle, circle, buff=0.25, thickness=5, path_arc=1 * DEG)

        self.play(ShowCreation(rough_circle))
        self.wait()
        self.play(
            LaggedStartMap(GrowArrow, arrows, lag_ratio=0.7),
            TransformFromCopy(rough_circle, circle, run_time=2),
        )
        self.wait()
        self.play(ReplacementTransform(arrows[0], mid_arrow))
        self.wait()
        self.play(ReplacementTransform(mid_arrow, arrows[1]))
        self.play(
            circle.animate.center(),
            FadeOut(rough_circle, 2 * LEFT),
            FadeOut(arrows[1], 2 * LEFT),
        )
        self.wait()


class HiddenConnections(InteractiveScene):
    def construct(self):
        # Transition from circle
        circle = Circle(radius=2)
        circle.set_stroke(WHITE, 3)
        self.add(circle)
        self.wait()

        background = FullScreenFadeRectangle()
        background.set_fill(GREY_E, 0.5)
        background.set_z_index(-1)

        # Screens
        screens = ScreenRectangle(height=3.0).get_grid(2, 2, h_buff=2.0, v_buff=1.5)
        screens.set_fill(BLACK, 1)
        screens.set_stroke(WHITE, 1)
        screens.set_height(7.5)

        q_marks = VGroup(
            Tex(R"???", font_size=72).move_to(screen)
            for screen in screens
        )

        connections = VGroup()
        for s1, s2 in it.combinations(screens, 2):
            vect = s2.get_center() - s1.get_center()
            line = Line(s1.get_corner(vect), s2.get_corner(-vect))
            connections.add(line)
        connections.set_stroke(BLUE, 3)
        screens[1:].set_fill(GREY_E, 0.5)

        self.remove(circle)
        self.play(
            FadeIn(background),
            ReplacementTransform(circle.replicate(len(connections)), connections, lag_ratio=0.1),
            LaggedStartMap(FadeIn, screens, scale=1.25),
            Write(q_marks[1:], time_span=(1, 3)),
            run_time=3
        )
        self.wait()

        # Expose individual screens
        self.play(
            FadeOut(q_marks[1]),
            screens[1].animate.set_fill(BLACK, 1),
            connections[1:].animate.set_stroke(width=1, opacity=0.5),
        )
        self.wait()
        self.play(
            FadeOut(q_marks[2]),
            screens[2].animate.set_fill(BLACK, 1),
            connections[1:4:2].animate.set_stroke(width=3, opacity=1),
        )
        self.wait()
        self.play(
            FadeOut(q_marks[3]),
            screens[3].animate.set_fill(BLACK, 1),
            connections.animate.set_stroke(width=3, opacity=1),
        )
        self.wait()


class WebOfConnections(InteractiveScene):
    n_points = 1000

    def setup(self):
        super().setup()

        # Points
        points = np.random.uniform(-1, 1, (self.n_points, 3))

        curr_norms = np.linalg.norm(points, axis=1)
        target_norms = curr_norms**1.5
        points *= (target_norms / curr_norms)[:, np.newaxis]

        points = points[curr_norms < 1]
        points = np.vstack([[ORIGIN], points])  # Ensure origin is in there
        points *= 25

        dots = DotCloud(points)
        dots.stretch(0.2, 2, about_point=ORIGIN)
        dots.set_radius(0.02)
        self.dots = dots

        # Web of connections
        web = VGroup()
        for p1, p2 in it.combinations(dots.get_points(), 2):
            if random.random() < np.exp(-0.8 * get_norm(p1 - p2)):
                line = Line(p1, p2, stroke_color=WHITE, stroke_width=1, stroke_opacity=random.random())
                web.add(line)
        self.web = web

        # Zoom out over connections
        self.frame.add_updater(lambda m: m.set_height(2 + 0.5 * self.time))
        self.frame.add_updater(lambda m, dt: m.increment_theta(0.5 * dt * DEG))
        self.frame.add_updater(lambda m, dt: m.increment_phi(dt * DEG))


class CentralWebConnections(WebOfConnections):
    n_iterations = 13
    n_neighbors = 10
    n_examples = 5

    def construct(self):
        # Test
        point_list = list(self.dots.get_points())
        dots = GlowDot(ORIGIN, radius=0.15).replicate(self.n_examples)

        for dot in dots:
            path = TracedPath(dot.get_center, stroke_width=1, stroke_color=WHITE)
            path.set_stroke(opacity=0.5)
            tail = TracingTail(dot, time_traced=2.0)
            self.add(dot, path, tail)

        dots.set_opacity(0)
        dots[0].set_opacity(1)
        self.wait()

        for n in range(self.n_iterations):
            for dot in dots:
                dot_center = dot.get_center()
                indices = np.argsort([get_norm(p - dot_center) for p in point_list])
                choice = np.random.randint(0, self.n_neighbors - 1)
                new_center = point_list[indices[choice]]
                point_list.pop(indices[choice])

                dot.target = dot.generate_target()
                dot.target.move_to(new_center)
                dot.target.set_opacity(1)
                if n > 0:
                    self.add(dot.copy().set_opacity(0.5))

            self.play(LaggedStartMap(MoveToTarget, dots), lag_ratio=0.5, run_time=3)


class ShowSimpleWeb(WebOfConnections):
    def construct(self):
        # Test
        self.web.sort(lambda p: get_norm(p))

        self.add(self.dots, self.web)
        self.wait(30)

        # Dense web
        dense_web = VGroup()
        for p1, p2 in it.combinations(self.dots.get_points(), 2):
            if random.random() < np.exp(-0.2 * get_norm(p1 - p2)):
                line = Line(
                    p1,
                    p2,
                    stroke_color=WHITE,
                    stroke_width=0.5,
                    stroke_opacity=random.random()**10
                )
                dense_web.add(line)
        self.play(ShowCreation(dense_web, lag_ratio=1 / len(dense_web), run_time=5))
        self.wait(5)


class GrowingWhiteDot(InteractiveScene):
    def construct(self):
        # Test
        dot = GlowDot(radius=1)
        dot.set_color(WHITE)
        self.add(dot)
        self.play(
            dot.animate.set_radius(FRAME_WIDTH).set_glow_factor(0.5),
            run_time=35
        )


class EndScreen(PatreonEndScreen):
    pass
