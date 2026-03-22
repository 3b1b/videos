from manim_imports_ext import *


class WhyDoWeCare(TeacherStudentsScene):
    def construct(self):
        # Test
        stds = self.students
        morty = self.teacher

        self.play(
            self.change_students("confused", "erm", "concentrating", look_at=self.screen),
        )
        self.wait(3)
        self.play(
            stds[2].change("erm", stds[1].eyes),
            stds[1].says("I’m sorry, why\ndo we care?", mode="sassy"),
            stds[0].change("thinking", self.screen),
            morty.change("well"),
        )
        self.wait(2)
        self.play(self.change_students("pondering", "maybe", "pondering", look_at=self.screen))

        # Answer
        self.play(
            morty.says("Topology has\nmore subtle utility", mode="tease"),
            stds[0].animate.look_at(morty.eyes),
            stds[1].debubble(),
            stds[2].change("hesitant", morty.eyes)
        )
        self.wait(3)


class RenameTheorem(InteractiveScene):
    def construct(self):
        # Test
        name1, name2 = names = VGroup(
            Text("Hairy Ball Theorem"),
            Text("Sphere Vector Field Theorem"),
        )
        names.scale(1.25)
        names.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        names.to_edge(LEFT)

        lines = VGroup()
        for text in ["Hairy", "Ball"]:
            word = name1[text][0]
            line = Line(word.get_left(), word.get_right())
            line.set_stroke(RED, 8)
            lines.add(line)
        lines[0].align_to(lines[1], UP)

        self.add(name1)
        self.wait()
        self.play(
            ShowCreation(lines[1]),
            name1["Ball"].animate.set_opacity(0.5),
            FadeTransformPieces(name1["Ball"].copy(), name2["Sphere"]),
        )
        self.play(
            ShowCreation(lines[0]),
            name1["Hairy"].animate.set_opacity(0.5),
            FadeTransformPieces(name1["Hairy"].copy(), name2["Vector Field"]),
        )
        self.play(
            TransformFromCopy(name1["Theorem"], name2["Theorem"]),
        )
        self.wait()


class SimpleImplies(InteractiveScene):
    def construct(self):
        arrow = Tex(R"\Rightarrow", font_size=120)
        self.play(Write(arrow))
        self.wait()


class CommentOnForce(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        morty.body.insert_n_curves(1000)
        equation = Tex(R"m x''(t) = \text{Lift} + \text{Gravity}", t2c={R"x''(t)": RED, R"\text{Lift}": PINK, R"\text{Gravity}": BLUE})
        equation.move_to(self.hold_up_spot, DOWN)
        equation.shift_onto_screen()

        self.play(
            morty.change("tease"),
            self.change_students("thinking", "erm", "concentrating", look_at=self.screen),
        )
        self.wait(2)
        self.play(
            morty.change("raise_right_hand", equation),
            self.change_students("pondering", "confused", "hesitant", look_at=equation),
            Write(equation),
        )
        self.wait(3)


class WingVectCodeSnippet(InteractiveScene):
    def construct(self):
        # Test
        code = Code("""
            def wing_vect(heading_vect):
                \"\"\"
                Return 3d vector perpendicular
                to heading_vect
                \"\"\"
                ...
        """, alignment="LEFT")
        self.play(ShowIncreasingSubsets(code, run_time=2, rate_func=linear))
        self.wait()


class LazyPerpCodeSnippet(InteractiveScene):
    def construct(self):
        # Test
        code = Code("""
            def lazy_perp(heading):
                # Returns the normalized cross product
                # between (0, 0, 1) and heading
                # Note the division by 0 for x=y=0
                x, y, z = heading
                return np.array([-y, x, 0]) / np.sqrt(x * x + y * y)
        """, alignment="LEFT")
        code.to_corner(UL)
        self.play(Write(code))
        self.wait()


class StatementOfTheorem(InteractiveScene):
    def construct(self):
        # Add text
        title = Text("Hairy Ball Theorem", font_size=72)
        title.to_corner(UL)
        underline = Underline(title)

        self.add(title, underline)

        statement = Text("""
            Any continuous vector field
            on a sphere must have at least
            one null vector.
        """, alignment="LEFT")
        statement.next_to(underline, DOWN, buff=MED_LARGE_BUFF)
        statement.to_edge(LEFT)

        self.play(Write(statement, run_time=3, lag_ratio=1e-1))
        self.wait()

        statement.set_backstroke(BLACK, 5)

        # Highlight text
        for text, color in [("continuous", BLUE), ("one null vector", YELLOW)]:
            self.play(
                FlashUnder(statement[text], time_width=1.5, run_time=2, color=color),
                statement[text].animate.set_fill(color)
            )
            self.wait()


class WriteAntipode(InteractiveScene):
    def construct(self):
        # Test
        text1 = Text("“Antipodes”")
        text2 = Text("Antipode map")
        for text in [text1, text2]:
            text.scale(1.5)
            text.to_corner(UL)

        self.play(Write(text1), run_time=2)
        self.wait()
        self.play(TransformMatchingStrings(text1, text2), run_time=1)
        self.wait()


class Programmer(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle().fix_in_frame())
        laptop = Laptop()
        self.frame.reorient(60, 66, 0, (0.09, -0.5, 0.13), 4.12)

        randy = Randolph(height=5)
        randy.to_edge(LEFT)
        randy.add_updater(lambda m: m.fix_in_frame().look_at(4 * RIGHT))

        self.add(laptop)
        self.play(randy.change("hesitant"))
        self.play(Blink(randy))
        self.play(randy.change("concentrating"))
        self.play(Blink(randy))
        self.wait()


class PedanticStudent(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            morty.change('raise_right_hand'),
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen)
        )
        self.wait()
        self.play(LaggedStart(
            stds[2].says("But atmosphere\nis 3D!", mode="angry", look_at=morty.eyes, bubble_direction=LEFT),
            morty.change("guilty"),
            stds[0].change("hesitant", look_at=stds[2].eyes),
            stds[1].change("hesitant", look_at=stds[2].eyes),
        ))
        self.wait(2)
        self.look_at(self.screen)
        self.wait(3)


class YouAsAMathematician(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph(height=4)
        randy.move_to(3 * LEFT)
        label = VGroup(
            Text("You", font_size=72),
            Text("The mathematician").set_color(GREY_B)
        )
        label.arrange(DOWN)
        label.next_to(randy, DOWN)

        self.add(randy, label)
        self.play(randy.change("pondering", 3 * RIGHT))
        self.play(Blink(randy))
        self.play(randy.change("tease", 3 * RIGHT))
        self.wait(3)


class ThreeCases(InteractiveScene):
    def construct(self):
        # Test
        titles = VGroup(
            VGroup(Text("2 null points"), Text("Obvious")),
            VGroup(Text("1 null point"), Text("Clever")),
            VGroup(Text("0 null points"), Text("Very clever")),
        )
        for title in titles:
            title[0].set_color(GREY_B)
            title[1].scale(1.25)
            title.arrange(DOWN)
        titles.arrange(RIGHT, buff=1.5, aligned_edge=UP)
        titles.to_edge(UP)

        vc_cross = Cross(titles[2][1])

        why_not = Text("Why not?")
        why_not.next_to(title)
        why_not.set_color(YELLOW)
        why_not.next_to(titles[2], DOWN, aligned_edge=RIGHT)

        for title in titles:
            self.add(title[0])
        for title in titles:
            self.play(FadeIn(title[1], lag_ratio=0.1))
        self.wait()
        self.play(ShowCreation(vc_cross))
        self.play(Write(why_not))


class ProofOutline(InteractiveScene):
    def construct(self):
        # Add outline
        title = Text("Proof by Contradiction", font_size=72)
        title.to_edge(UP)
        background = FullScreenRectangle()

        frames = Square().replicate(2)
        frames.set_height(4.5)
        frames.arrange(RIGHT, buff=3.5)
        frames.next_to(title, DOWN, buff=1.5)
        frames.set_fill(BLACK, 1)
        frames.set_stroke(WHITE, 2)

        implies = Tex(R"\Longrightarrow", font_size=120)
        implies.move_to(frames)

        impossibility = Text("Impossibility", font_size=90)
        impossibility.next_to(implies, RIGHT, MED_LARGE_BUFF)
        impossibility.set_color(RED)

        assumption = Text("Assume there exists a non-zero\ncontinuous vector field", font_size=30)
        assumption.set_color(BLUE)
        assumption.next_to(frames[0], UP)

        self.add(background)
        self.play(Write(title), run_time=2)
        self.wait()
        self.play(
            FadeIn(frames[0]),
            # FadeIn(assumption, lag_ratio=0.01)
        )
        self.wait()
        implies.save_state()
        implies.stretch(0, 0, about_edge=LEFT)
        self.play(Restore(implies))
        self.play(FadeIn(impossibility, lag_ratio=0.1))
        self.wait()
        self.play(
            DrawBorderThenFill(frames[1]),
            impossibility.animate.scale(0.5).next_to(frames[1], UP)
        )
        self.wait()
        self.play(FadeOut(impossibility))

        # Next part
        words = VGroup(Text("Assume the impossible"), Text("Find a contradiction"))
        brace = Brace(frames[0], RIGHT)
        question = Text("What do we\nshow here?", font_size=72)
        question.next_to(brace, RIGHT)

        self.play(
            FadeOut(implies),
            FadeOut(frames[1]),
        )
        self.play(
            GrowFromCenter(brace),
            FadeIn(question, lag_ratio=0.1),
        )
        self.wait()


class AimingForRediscovery(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        goal = Text("Goal: A feeling\nof rediscovery")
        goal.move_to(self.hold_up_spot, DOWN)

        self.play(
            morty.change("tease"),
            self.change_students("pondering", "happy", "hooray", look_at=self.screen)
        )
        self.wait(3)
        self.play(
            FadeIn(goal, shift=UP),
            morty.change("raise_right_hand"),
            self.change_students("pondering", "pondering", "pondering", look_at=goal)
        )
        self.wait(2)
        self.play(self.change_students("thinking", "tease", "erm", look_at=self.screen))
        self.wait()


class TwoFactsForEachPoint(InteractiveScene):
    def construct(self):
        # Test
        features = VGroup(
            Tex(R"\text{1)  } p \rightarrow -p", font_size=72),
            Tex(R"\text{2)  } &\text{Motion varies }\\ &\text{continuously with } p", font_size=72),
        )
        features[1][2:].scale(0.8, about_edge=UL)
        features.arrange(DOWN, aligned_edge=LEFT, buff=1.5)
        features.to_edge(LEFT)

        for feature in features:
            self.add(feature[:2])
        self.wait()
        for feature in features:
            self.play(FadeIn(feature[2:], lag_ratio=0.1))
            self.wait()
        self.wait()


class WaitWhat(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph(height=4)
        randy.to_edge(LEFT, buff=2.5)
        randy.shift(DOWN)
        randy.body.insert_n_curves(1000)

        self.play(randy.says("Wait...what?", mode="confused", bubble_direction=RIGHT, look_at=RIGHT))
        self.play(Blink(randy))
        self.wait(3)


class TwoKeyFeatures(InteractiveScene):
    def construct(self):
        # Set up
        features = VGroup(
            # Text("1) Sphere turns\ninside out"),
            # Text("2) No point touches\nthe origin"),
            Text("1) Inside out"),
            Text("2) Avoids the origin"),
        )
        features[1]["the origin"].align_to(features[1]["No"], LEFT)
        features.arrange(DOWN, aligned_edge=LEFT, buff=1.5)
        features.to_edge(LEFT)

        for feature in features:
            self.play(FadeIn(feature, lag_ratio=0.1))
            self.wait()

        # Emphasize first point
        self.play(
            features[0].animate.scale(1.25, about_edge=LEFT),
            features[1].animate.scale(0.75, about_edge=LEFT).set_fill(opacity=0.5).shift(DOWN),
        )
        self.wait()

        # Ask why
        randy = Randolph(height=1.75)
        randy.next_to(features[0], DOWN, buff=MED_LARGE_BUFF)
        why = Text("Why?", font_size=36)
        why.next_to(randy, RIGHT, aligned_edge=UP)
        why.set_color(YELLOW)

        self.play(
            VFadeIn(randy),
            randy.change("maybe"),
            Write(why),
        )
        self.play(Blink(randy))
        self.wait()

        # Ask what "inside out" means
        rect = SurroundingRectangle(features[0][2:])
        rect.set_stroke(YELLOW, 2)
        self.play(
            randy.change("confused", rect),
            FadeTransform(why, rect)
        )
        self.play(Blink(randy))
        self.wait()
        self.play(FadeOut(randy), FadeOut(rect))

        # Inside out implication
        rect0 = SurroundingRectangle(features[0])
        rect0.set_stroke(BLUE, 2)

        implies0 = Tex(R"\Longrightarrow", font_size=72)
        implies0.next_to(rect0)
        net_flow_m1 = TexText("Final Flux = $-1.0$", t2c={"-1.0": RED}, font_size=60)
        net_flow_m1.next_to(implies0, RIGHT)

        self.play(
            ShowCreation(rect0),
            FadeIn(implies0, scale=2, shift=0.25 * RIGHT),
        )
        self.play(FadeIn(net_flow_m1, lag_ratio=0.1))
        self.wait()

        # No origin implication
        self.play(features[1].animate.scale(1.25 / 0.75, about_edge=UL).set_opacity(1))

        rect1 = SurroundingRectangle(features[1])
        rect1.match_style(rect0)
        implies1 = implies0.copy()
        implies1.next_to(rect1)
        net_flow_p1 = TexText(R"Final flux = $+1.0$", t2c={"+1.0": GREEN}, font_size=60)  
        net_flow_p1.next_to(implies1, RIGHT)

        self.play(
            ShowCreation(rect1),
            FadeIn(implies1, scale=2, shift=0.25 * RIGHT),
        )
        self.play(FadeIn(net_flow_p1, lag_ratio=0.1))
        self.wait()

        # Contradiction
        contra = Tex(R"\bot", font_size=90)
        contra.to_corner(DR)

        self.play(Write(contra))
        self.wait()


class DumbQuestion(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            stds[2].says("Isn’t it\nobvious?", mode="confused", look_at=self.screen, bubble_direction=LEFT),
            stds[1].change("angry", look_at=morty.eyes),
            stds[0].change("erm", look_at=morty.eyes),
            morty.change("guilty"),
        )
        self.wait(2)
        self.play(
            stds[0].change("confused", self.screen),
            stds[1].change("sassy", self.screen),
            stds[2].change("maybe", morty.eyes),
        )
        self.wait(4)


class InsideOutsideQuestion(InteractiveScene):
    def construct(self):
        # Test
        inside = Text("Inside?", font_size=72)
        outside = Text("Outside?", font_size=72)
        VGroup(inside, outside).set_backstroke(BLACK, 3)
        self.play(FadeIn(inside, lag_ratio=0.1))
        self.wait()
        self.play(
            FadeIn(outside, lag_ratio=0.1),
            FadeOut(inside, lag_ratio=0.1),
        )
        self.wait()


class WhatIsInsideAndOutside(TeacherStudentsScene):
    def construct(self):
        # Test
        stds = self.students
        morty = self.teacher

        self.play(
            stds[2].says(
                "Hang on, what\ndo you mean\n“paint the outside”?",
                mode="maybe",
                bubble_direction=LEFT
            ),
            stds[1].change("erm", self.screen),
            stds[0].change("pondering", self.screen),
            morty.change("tease")
        )
        self.wait(5)


class PToNegP(InteractiveScene):
    def construct(self):
        # Test
        p, to, neg_p = expression = VGroup(
            Tex(R"p"), Tex(R"\longrightarrow"), Tex(R"-p")
        )
        expression.arrange(RIGHT, buff=0.75)
        expression.scale(2.5)
        expression.to_edge(UP)

        to.save_state()
        to.stretch(0, 0, about_edge=LEFT)
        to.stretch(0.5, 1)

        self.play(Write(p))
        self.play(Restore(to))
        self.play(FadeTransformPieces(p.copy(), neg_p))
        self.wait()


class SimplerInsideOutProgression(InteractiveScene):
    def construct(self):
        # Test
        parts = VGroup(
            Tex(R"(x, y, z)"),
            Vector(0.75 * DOWN),
            Tex(R"(-x, -y, z)"),
            Vector(0.75 * DOWN),
            Tex(R"(-x, -y, -z)"),
        )
        parts.arrange(DOWN, buff=MED_SMALL_BUFF)
        self.add(parts[0])
        self.wait()

        for i in [0, 2]:
            src, arrow, trg = parts[i:i + 3]
            self.play(
                TransformMatchingStrings(src.copy(), trg),
                GrowArrow(arrow),
                run_time=1
            )
            self.wait()


class ReferenceInsideOutMovie(TeacherStudentsScene):
    def construct(self):
        # Complain
        morty = self.teacher
        stds = self.students
        self.screen.to_corner(UL)

        self.play(
            stds[0].change("pondering", self.screen),
            stds[1].change("erm", self.screen),
            stds[2].says(
                Text(
                    "Huh? I thought\nyou can turn a\nsphere inside out!",
                    t2s={"can": ITALIC},
                    font_size=42,
                ),
                mode="confused",
                look_at=self.screen,
                bubble_direction=LEFT
            ),
            morty.change("guilty")
        )
        self.wait(2)
        self.play(morty.change('tease'))
        self.wait(3)


class FluxDecimals(InteractiveScene):
    def construct(self):
        # Test
        label = TexText("Flux: +1.000 L/s", font_size=60)
        dec = label.make_number_changeable("+1.000", include_sign=True)
        label.to_corner(UR)
        dec.set_value(1)

        def update_color(dec, epsilon=1e-4):
            value = dec.get_value()
            if value > epsilon:
                dec.set_color(GREEN)
            elif abs(value) < epsilon:
                dec.set_color(YELLOW)
            else:
                dec.set_color(RED)

        dec.add_updater(update_color)

        self.add(label)
        self.wait()
        for value in [0.014, -0.014, 0.014]:
            self.play(ChangeDecimalToValue(dec, value))
            self.wait()
        self.play(ChangeDecimalToValue(dec, 1.0), run_time=3)
        self.wait()
        dec.set_value(0)
        self.wait()


class DivergenceTheorem(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        div_theorem = Tex(R"""
            \displaystyle \iiint_V(\nabla \cdot \mathbf{F}) \mathrm{d} V
            = \oiint_S(\mathbf{F} \cdot \hat{\mathbf{n}}) \mathrm{d} S
        """, t2c={R"\mathbf{F}": BLUE})
        div_theorem.next_to(morty, UP, buff=1.5)
        div_theorem.to_edge(RIGHT)

        div_theorem_name = Text("Divergence Theorem", font_size=72)
        div_theorem_name.next_to(div_theorem, UP, buff=0.5)

        for pi in self.pi_creatures:
            pi.body.insert_n_curves(200)

        self.play(
            self.change_students("pondering", "confused", "tease", look_at=self.screen),
            morty.change("tease"),
        )
        self.wait(3)
        self.play(
            morty.change("raise_right_hand", div_theorem),
            FadeIn(div_theorem, shift=UP),
            self.change_students("thinking", "confused", "happy", look_at=div_theorem),
        )
        self.wait()
        self.play(Write(div_theorem_name))
        self.wait(3)


class ThinkAboutOrigin(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        self.play(
            morty.says(Text("Think about why\ncrossing the origin\nis significant", font_size=36)),
            self.change_students('thinking', 'tease', 'pondering', look_at=self.screen)
        )
        self.wait(5)


class CommentOnContardiction(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer(height=3)
        morty.body.insert_n_curves(1000)
        morty.to_corner(DR)
        morty.shift(2 * LEFT)

        qed = Text("Q.E.D.")
        qed.next_to(morty.get_corner(UR), UP, MED_SMALL_BUFF)

        self.play(morty.says("Contradiction!", mode="hooray"))
        self.play(Blink(morty))
        self.wait()
        self.play(morty.change('raise_left_hand', look_at=qed), FadeIn(qed, 0.25 * UP))
        self.play(Blink(morty))
        self.play(morty.change('tease'))
        self.wait()


class FrameIntuitionVsExamples(InteractiveScene):
    def construct(self):
        titles = VGroup(
            Text("Intuitive idea"),
            Text("Counterexample"),
            Text("Clever proof"),
        )
        for x, title in zip([-1, 1, 1], titles):
            title.scale(1.5)
            title.move_to(x * FRAME_WIDTH * RIGHT / 4)
            title.to_edge(UP)
        h_line = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        h_line.to_edge(UP, buff=1.5)
        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        VGroup(h_line, v_line).set_stroke(WHITE, 2)

        ideas = VGroup(
            Text("Turning a sphere\ninside-out must crease it"),
            Text("All closed loops\nhave inscribed rectangles"),
        )
        for idea in ideas:
            idea.next_to(h_line, DOWN)
            idea.set_color(GREY_A)
            idea.shift(FRAME_WIDTH * LEFT / 4)

        self.add(v_line, h_line)
        self.add(titles[0])
        self.wait()

        # Test
        self.play(FadeIn(ideas[0]))
        self.play(
            FadeIn(titles[1], lag_ratio=0.1)
        )
        self.wait()
        self.play(
            FadeOut(ideas[0]),
            FadeIn(ideas[1]),
            FadeOut(titles[1], lag_ratio=0.1),
            FadeIn(titles[2], lag_ratio=0.1),
        )
        self.wait()


class DimensionGeneralization(InteractiveScene):
    def construct(self):
        # Set up grid
        row_labels = VGroup(
            Text("Dimension"),
            Text("Can you\ncomb a ball?"),
        )
        n_cols = 15
        cells = Square().get_grid(2, 1, buff=0).get_grid(1, n_cols, buff=0)
        cells.set_height(2.6)
        cells[0].set_width(row_labels.get_width() + MED_LARGE_BUFF, stretch=True, about_edge=RIGHT)
        cells.to_corner(UL, buff=LARGE_BUFF)
        for label, cell in zip(row_labels, cells[0]):
            label.move_to(cell)

        dim_labels = VGroup()
        mark_labels = VGroup()
        for n, cell in zip(it.count(2), cells[1:]):
            dim_label = Integer(n)
            mark_label = Checkmark().set_color(GREEN) if n % 2 == 0 else Exmark().set_color(RED)
            mark_label.set_height(0.5 * cell[1].get_height())
            dim_label.move_to(cell[0])
            mark_label.move_to(cell[1])

            dim_labels.add(dim_label)
            mark_labels.add(mark_label)

        self.add(cells[:3], row_labels, dim_labels[:2], mark_labels[1])
        self.play(
            LaggedStartMap(FadeIn, cells[3:], lag_ratio=0.5),
            LaggedStartMap(FadeIn, dim_labels[2:], lag_ratio=0.5),
            run_time=3
        )
        self.wait()

        # Show two
        self.play(Write(mark_labels[0]))
        self.wait()

        # General dimensions
        frame = self.frame
        for i in [0, 1]:
            self.play(
                LaggedStart(
                    (TransformFromCopy(mark_labels[i], mark_label, path_arc=30 * DEG)
                    for mark_label in mark_labels[i + 2::2]),
                    lag_ratio=0.25,
                ),
                frame.animate.reorient(0, 0, 0, (4.15, -2.53, 0.0), 12.34),
                run_time=3
            )
            self.wait()

        # Show determinants
        last_det = VGroup()
        highlight_rect = SurroundingRectangle(cells[1], buff=0)
        highlight_rect.set_opacity(0).shift(LEFT)
        for dim in range(2, 12):
            det_tex = self.get_det_neg_tex(dim)
            det_tex.scale(1.5)
            det_tex.move_to(5 * DOWN).to_edge(LEFT, LARGE_BUFF)
            rect = SurroundingRectangle(cells[dim - 1], buff=0)
            rect.set_stroke(YELLOW, 5)

            self.play(
                Transform(highlight_rect, rect),
                FadeIn(det_tex),
                FadeOut(last_det),
            )
            self.wait()
            last_det = det_tex

    def get_det_neg_tex(self, dim):
        mat = IntegerMatrix(-1 * np.identity(dim))
        det_text = Text("det")
        mat.set_max_height(5 * det_text.get_height())
        lp, rp = parens = Tex(R"()")
        parens.stretch(2, 1)
        parens.match_height(mat)
        lp.next_to(mat, LEFT, buff=0.1)
        rp.next_to(mat, RIGHT, buff=0.1)
        det_text.next_to(parens, LEFT, SMALL_BUFF)

        sign = ["+", "-"][dim % 2]
        rhs = Tex(Rf"= {sign}1")
        rhs.next_to(rp, RIGHT, SMALL_BUFF)
        rhs[1:].set_color([GREEN, RED][dim % 2])

        result = VGroup(det_text, lp, mat, rp, rhs)
        return result


class MoreRigorNeeded(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher

        words = Text("More rigor\nneeded", font_size=60)
        arrow = Vector(1.5 * LEFT, thickness=8)
        label = VGroup(arrow, words)
        label.arrange(RIGHT, SMALL_BUFF)
        label.next_to(self.screen, RIGHT)

        self.add(words)
        self.play(
            morty.change("hesitant"),
            self.change_students("confused", "sassy", "erm", look_at=self.screen),
            Write(words),
        )
        self.play(
            GrowArrow(arrow)
        )
        self.wait(4)


class AskAboutHomology(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        chain = Tex(R"\cdots C_{2} \xrightarrow{\partial_2} C_1 \xrightarrow{\partial_1} C_0 \xrightarrow{\partial_0} 0")
        chain.next_to(stds[2], UP, buff=MED_LARGE_BUFF)
        chain.align_to(stds[2].get_center(), RIGHT)

        self.play(
            morty.change("well", stds[2].eyes),
            stds[2].says(
                "What about\nusing homology?",
                mode="tease",
                bubble_direction=LEFT,
                look_at=morty.eyes
            )
        )
        self.play(
            stds[2].change("raise_left_hand", chain),
            self.change_students("confused", "erm"),
            Write(chain),
        )
        self.wait(3)


class RotationIn2D(InteractiveScene):
    def construct(self):
        # Test
        grid = NumberPlane()
        back_grid = grid.copy()
        back_grid.background_lines.set_stroke(GREY, 1)
        back_grid.axes.set_stroke(GREY, 1, 1)
        back_grid.faded_lines.set_stroke(GREY, 0.5, 0.5)

        basis_vectors = VGroup(
            Vector(RIGHT).set_color(GREEN),
            Vector(UP).set_color(RED),
        )

        self.frame.set_height(4)
        self.add(back_grid, grid, basis_vectors)
        self.wait()
        self.play(
            Rotate(basis_vectors, PI, about_point=ORIGIN),
            Rotate(grid, PI, about_point=ORIGIN),
            run_time=5
        )
        self.wait()


class InversionIn3d(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        frame.add_ambient_rotation(1 * DEG)
        axes = ThreeDAxes((-3, 3), (-3, 3), (-3, 3))
        coord_range = list(range(-3, 3))
        cubes = VGroup(
            VCube(side_length=1).move_to([x, y, z], DL + IN)
            for x, y, z in it.product(* 3 * [coord_range])
        )
        cubes.set_fill(opacity=0)
        cubes.set_stroke(WHITE, 1, 0.25)

        basis_vectors = VGroup(
            Vector(RIGHT).set_color(GREEN),
            Vector(UP).set_color(RED),
            Vector(OUT).set_color(BLUE),
        )
        # for vect in basis_vectors:
        #     vect.set_perpendicular_to_camera(frame)

        frame.reorient(29, 72, 0, ORIGIN, 5)
        self.add(axes, cubes, basis_vectors)

        # Rotate
        rot_group = VGroup(cubes, basis_vectors)
        self.wait()
        self.play(
            Rotate(rot_group, PI, about_point=ORIGIN, run_time=3)
        )
        self.play(
            rot_group.animate.stretch(-1, 2, about_point=ORIGIN),
            run_time=2
        )
        self.wait(3)


class HypersphereWords(InteractiveScene):
    def construct(self):
        # Test
        words = VGroup(
            Text("Hairs on a neatly-combed 4d hypersphere", font_size=60),
            Text("Represented via streographic projection into 3d space", font_size=48).set_color(GREY_A)
        )
        words.set_backstroke(BLACK, 10)
        words.arrange(DOWN)
        words.to_corner(UL)

        self.add(words[0])
        self.wait()
        self.play(FadeIn(words[1], lag_ratio=0.1, run_time=2))
        self.wait()


class EndScreen2(SideScrollEndScreen):
    scroll_time = 23
