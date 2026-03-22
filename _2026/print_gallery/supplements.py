from manim_imports_ext import *
from _2026.print_gallery.exponential import get_rectified_print_gallery


class PrintGalleryTitle(InteractiveScene):
    def construct(self):
        group = VGroup(
            Text("The Print Gallery", font_size=72),
            Text("M.C. Escher, 1956").set_fill(GREY_A),
        )
        group.arrange(DOWN)
        group.set_backstroke(BLACK, 2)
        self.play(FadeIn(group, lag_ratio=0.1, run_time=2))
        self.wait()


class Prentententoonsteling(InteractiveScene):
    def construct(self):
        text = Text("“Prentententoonstelling.”")
        self.play(Write(text), lag_ratio=0.02, run_time=2)
        self.wait()


class EscherToSonQuote(InteractiveScene):
    def construct(self):
        # Add quote
        quote = Text("""
            “...the most peculiar thing I
            have ever done...a young man
            looking with interest at a print
            on the wall of an exhibition
            that features himself.”
        """, alignment="LEFT")
        quote.to_edge(RIGHT)
        quote.set_color(GREY_B)

        underline = VGroup(
            Underline(quote["the most peculiar thing I"], buff=-0.05, stretch_factor=1),
            Underline(quote["have ever done"], buff=0.05, stretch_factor=1),
        )
        underline.set_stroke(TEAL, 2)

        self.play(FadeIn(quote, lag_ratio=5e-3, run_time=2))
        self.play(
            LaggedStart(
                quote["a young man"].animate.set_fill(YELLOW).set_anim_args(lag_ratio=0.1),
                quote["looking with interest at a print"].animate.set_fill(YELLOW).set_anim_args(lag_ratio=0.1),
                quote["that features himself"].animate.set_fill(YELLOW).set_anim_args(lag_ratio=0.1),
                lag_ratio=0.5
            )
        )
        self.wait()
        self.play(
            quote[20:].animate.set_fill(GREY),
            LaggedStart(
                quote["the most peculiar thing I"].animate.set_fill(TEAL).set_anim_args(lag_ratio=0.1),
                quote["have ever done"].animate.set_fill(TEAL).set_anim_args(lag_ratio=0.1),
                lag_ratio=0.5,
                run_time=2
            ),
            Write(underline, run_time=4, lag_ratio=0.5),
        )
        self.wait()


class MathVsIntuition(InteractiveScene):
    def construct(self):
        morty = Mortimer().to_edge(DOWN)
        morty.body.insert_n_curves(100)
        log_image = ImageMobject(Path(
            self.file_writer.get_output_file_rootname().parent.parent,
            'exponential/LogImage.png',
        ))
        log_image.set_height(3)
        log_image.next_to(morty, UP).to_edge(LEFT)

        self.play(
            morty.change("raise_right_hand", log_image),
            FadeIn(log_image, UP)
        )
        self.play(Blink(morty))
        self.wait()
        self.play(
            morty.change("raise_left_hand", 3 * UR),
            log_image.animate.scale(0.5, about_edge=LEFT).fade(0.5),
        )
        self.play(Blink(morty))
        self.wait()


class CyclidElementsQuote(InteractiveScene):
    def construct(self):
        # Test
        quote = Text("""
            “I quite intentionally chose serial
            types of objects, such, for instance,
            as a row of prints along the wall and
            blocks of houses in a town. Without the
            cyclic elements, it would be all the
            more difficult to get my meaning over
            to the random viewer.”
        """, alignment="LEFT")
        quote.to_edge(RIGHT)
        quote.shift(UP)
        attribution = Text(
            "From Magic Mirror of M.C. Escher\nby Bruno Ernst.",
            t2s={"Magic Mirror of M.C. Escher": ITALIC},
            font_size=30
        )
        attribution.set_fill(GREY_B)
        attribution.next_to(quote, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        self.play(FadeIn(quote, lag_ratio=0.005, run_time=2))
        self.wait()
        for phrase, color, u in [("row of prints", YELLOW, -1), ("blocks of houses", TEAL, +1)]:
            part = quote[phrase][0]
            underline = Underline(part, stretch_factor=1, buff=u * 0.05)
            underline.set_stroke(color, 3)
            self.play(
                part.animate.set_color(color).set_anim_args(lag_ratio=0.1),
                ShowCreation(underline)
            )
            self.wait()


class WhereDidThatPictureComeFrom(TeacherStudentsScene):
    def construct(self):
        # Ask
        self.remove(self.background)
        morty = self.teacher
        stds = self.students

        self.play(
            stds[2].says("Where does that\n come from?", "raise_left_hand", bubble_direction=LEFT, look_at=self.screen),
            stds[0].change("thinking", self.screen),
            stds[1].change("hesitant", self.screen),
        )
        self.play(morty.change("tease"))
        self.wait(3)

        # Show the mathematicians
        frame = self.frame
        mathematicians = VGroup(
            PiCreature(color=GREY_C).flip(),
            PiCreature(color=GREY_D).flip(),
        )
        for pi in mathematicians:
            pi.body.insert_n_curves(100)
        mathematicians.arrange(RIGHT)
        mathematicians.next_to(morty, RIGHT, buff=4.0, aligned_edge=DOWN)

        self.play(
            frame.animate.set_x(10),
            stds[2].debubble(),
            morty.change("gracious", look_at=mathematicians.get_top()),
            mathematicians[0].change("hooray"),
            mathematicians[1].change("tease"),
            run_time=2
        )
        self.play(mathematicians[0].change("raise_right_hand"))
        self.play(Blink(mathematicians[0]))
        self.play(Blink(mathematicians[1]))
        self.wait()
        pass


class WhereDoesTheGridComeFrom(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        randy.to_corner(DL)
        self.play(
            randy.says("Where does this\ncome from?", mode="confused")
        )
        self.play(Blink(randy))
        self.wait()


class SimpleArrow(InteractiveScene):
    def construct(self):
        vect = Vector(1.5 * RIGHT, thickness=8)
        vect.set_fill(TEAL)
        self.play(GrowArrow(vect))
        self.wait()


class SorryWhat(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        stds = self.students
        self.screen.center().to_edge(UP)
        self.play(
            stds[0].says("I'm sorry, what?", mode="maybe", look_at=self.screen),
            stds[2].change("maybe", look_at=self.screen),
            stds[1].change("confused", look_at=self.screen),
            self.teacher.change("tease")
        )
        self.wait(3)


class DistributeZoomOverlay(InteractiveScene):
    def construct(self):
        # Add frames
        frames = Square(side_length=6).replicate(2)
        frames.arrange(RIGHT, buff=1.5)
        frames.set_stroke(WHITE, 2)
        frames.set_fill(BLACK, 0)
        left_frame, right_frame = frames
        left_frame.set_stroke(BLACK)
        right_frame.set_stroke(opacity=0)

        self.add(left_frame)

        # Add zoom arrows
        small_left = left_frame.copy()
        small_left.scale(1 / 16)
        corner_arrows = VGroup(
            Arrow(
                left_frame.get_corner(vect),
                small_left.get_corner(vect),
                buff=0.1,
                thickness=8
            )
            for vect in compass_directions(4, DL)
        )
        corner_arrows.set_color(TEAL)
        zoom_label = Tex(R"\times 16", font_size=72)
        zoom_label.move_to(corner_arrows[:2])

        self.play(
            *map(GrowArrow, corner_arrows),
            TransformFromCopy(left_frame, small_left),
            Write(zoom_label)
        )
        self.wait()

        # Distributed
        corner_squares = Square(side_length=2).replicate(4)
        for corner, square in zip(compass_directions(4, DL), corner_squares):
            square.move_to(right_frame, corner)

        rot_arrows = VGroup(
            Arrow(s2, s1, buff=0.1, thickness=5, fill_color=TEAL)
            for s1, s2 in adjacent_pairs(corner_squares)
        )

        two_x_labels = VGroup(
            Tex(R"\times 2").next_to(arrow, np.round(rotate_vector(arrow.get_vector(), 90 * DEG)), buff=0)
            for arrow in rot_arrows
        )

        corner_arrow_ghosts = corner_arrows.copy().set_fill(opacity=0.35)
        zoom_label_ghost = zoom_label.copy().set_fill(opacity=0.35)

        self.remove(zoom_label)
        self.play(
            small_left.animate.set_stroke(opacity=0.35),
            FadeIn(right_frame),
            ReplacementTransform(corner_arrows, rot_arrows),
            *(
                FadeTransform(zoom_label.copy(), two_x_label)
                for two_x_label in two_x_labels
            ),
            run_time=2
        )
        self.wait()


class DistributeZoom256(InteractiveScene):
    def construct(self):
        # Add frames
        frames = Square(side_length=6).replicate(2)
        frames.arrange(RIGHT, buff=1.5)
        frames.set_stroke(WHITE, 2)
        frames.set_fill(BLACK, 0)
        left_frame, right_frame = frames
        left_frame.set_stroke(BLACK)
        right_frame.set_stroke(opacity=0)

        self.add(left_frame)

        # Add droste
        droste_image = get_rectified_print_gallery()
        droste_image.scale(175)
        droste_image.move_to(left_frame)
        droste_image.clip_to_box(left_frame)

        droste_image_ghost = droste_image.copy()
        droste_image_ghost.set_opacity(0.5)

        self.add(droste_image_ghost, Point(), droste_image)

        # Add zoom arrows
        small_left = left_frame.copy()
        small_left.scale(1 / 256)
        corner_arrows = VGroup(
            Arrow(
                left_frame.get_corner(vect),
                small_left.get_corner(vect),
                buff=0.1,
                thickness=8
            )
            for vect in compass_directions(4, DL)
        )
        corner_arrows.set_color(TEAL)
        zoom_label = Tex(R"\times 256", font_size=72)
        zoom_label.move_to(corner_arrows[:2])

        self.play(
            *map(GrowArrow, corner_arrows),
            Transform(left_frame, small_left),
            Write(zoom_label),
            droste_image.animate.clip_to_box(small_left),
            run_time=2
        )
        self.wait()

        # Distributed
        top_equation = Tex(R"256 = 4 \times 4 \times 4 \times 4")
        top_equation.to_edge(UP, buff=0.25)

        corner_squares = Square(side_length=2).replicate(4)
        for corner, square in zip(compass_directions(4, DL), corner_squares):
            square.move_to(right_frame, corner)

        rot_arrows = VGroup(
            Arrow(s2, s1, buff=0.1, thickness=5, fill_color=TEAL)
            for s1, s2 in adjacent_pairs(corner_squares)
        )

        two_x_labels = VGroup(
            Tex(R"\times 4").next_to(arrow, np.round(rotate_vector(arrow.get_vector(), 90 * DEG)), buff=0)
            for arrow in rot_arrows
        )

        corner_arrow_ghosts = corner_arrows.copy().set_fill(opacity=0.35)
        zoom_label_ghost = zoom_label.copy().set_fill(opacity=0.35)

        self.play(
            Write(top_equation[3:]),
            TransformFromCopy(zoom_label["256"], top_equation["256"]),
            run_time=1
        )
        self.play(
            small_left.animate.set_stroke(opacity=0.35),
            FadeIn(right_frame),
            TransformFromCopy(corner_arrows, rot_arrows),
            *(
                FadeTransform(zoom_label.copy(), two_x_label)
                for two_x_label in two_x_labels
            ),
            run_time=2
        )
        self.wait()


class ComplexConstantTerms(InteractiveScene):
    def construct(self):
        # Test
        kw = dict(t2c={"{c}": GREEN, "z": YELLOW})
        equations = VGroup(
            Tex(R"f(z) = {c} \cdot z", **kw),
            Tex(R"f(0) = {c} \cdot 0 = 0", **kw),
            Tex(R"f(1) = {c} \cdot 1 = {c}", **kw),
        )
        equations.arrange(RIGHT, buff=1.5)
        equations.to_edge(UP)

        self.add(equations[0])
        self.wait()
        for eq in equations[1:]:
            self.play(Write(eq))
            self.wait()


class LogEquation(InteractiveScene):
    def construct(self):
        group = VGroup(
            Tex(R"=").rotate(90 * DEG),
            Tex(R"\ln(w \cdot 16)", t2c={"w": RED})
        )
        group.arrange(DOWN)
        group.set_backstroke(BLACK, 3)
        self.play(Write(group))
        self.wait()


class ComplexDerivEquation(InteractiveScene):
    def construct(self):
        equation = Tex(
            R"\Delta f(z) \approx \Delta z \cdot c",
            t2c={
                R"\Delta f(z)": PINK,
                R"\Delta z": YELLOW,
                "c": GREEN
            },
            font_size=72
        )
        self.play(Write(equation))
        self.wait()


class WhyDerivatives(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students

        self.play(
            morty.change("well"),
            self.change_students("pondering", "hesitant", "guilty", look_at=self.screen)
        )
        self.wait(5)


class ReferenceDerivatives(InteractiveScene):
    def construct(self):
        # Test
        groups = VGroup(
            self.get_group("z^2", "2z"),
            self.get_group("z^3", "3z^2"),
            self.get_group("e^z", "e^z"),
            self.get_group(R"\ln(z)", R"{1 \over z}"),
        )
        groups.arrange(DOWN, buff=LARGE_BUFF)

        self.add(*(g[0] for g in groups))
        self.play(
            LaggedStart(
                (TransformMatchingTex(g[0].copy(), g[2])
                for g in groups),
                lag_ratio=0.2,
            ),
            LaggedStart(
                (GrowArrow(g[1]) for g in groups),
                lag_ratio=0.2,
            ),
            LaggedStart(
                (FadeIn(g[3], 0.25 * RIGHT) for g in groups),
                lag_ratio=0.2,
            ),
            run_time=3
        )
        self.wait()

    def get_group(self, in_tex, out_tex):
        kw = dict(font_size=60, t2c={"z": BLUE})
        group = VGroup(
            Tex(in_tex, **kw),
            Vector(1.5 * RIGHT, thickness=4),
            Tex(out_tex, **kw),
        )
        group.arrange(RIGHT)
        deriv_label = Tex(R"d / dz", **kw)
        deriv_label.scale(0.5)
        deriv_label.next_to(group[1], UP, SMALL_BUFF)
        group.add(deriv_label)
        return group


class ReferenceZSquared(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            morty.change("raise_right_hand"),
            self.change_students("pondering", "thinking", "pondering", look_at=self.screen)
        )
        self.wait(2)
        self.look_at(3 * UP, run_time=3)
        self.play(
            self.change_students("erm", "hesitant", "maybe", look_at=3 * UR),
            morty.change("tease"),
        )
        self.wait(5)


class TimeSpentOnExp(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students

        self.play(
            morty.change("raise_right_hand"),
            self.change_students("pondering", "confused", "erm", look_at=self.screen),
        )
        self.wait(2)
        self.play(
            morty.change("raise_left_hand", 3 * UR),
            self.change_students("hesitant", "pondering", "sassy", look_at=3 * UR)
        )
        self.wait(3)


class IKnowThisOne(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            stds[2].says("I know\nthis one!", look_at=morty.eyes, mode="hooray", bubble_direction=LEFT),
            stds[0].change("confused", self.screen),
            stds[1].change("happy", self.screen),
            morty.change("tease"),
        )
        self.wait(3)
        self.play(
            morty.change("raise_right_hand"),
            stds[2].debubble('pondering', self.screen),
        )
        self.wait(2)


class ReferencePastExpVideos(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)

        # Test
        morty = self.teacher
        stds = self.students
        self.play(
            stds[1].says("Why?", mode="confused", look_at=2 * UR),
            stds[2].change("erm", look_at=2 * UR),
            stds[0].change("pondering", look_at=2 * UL),
            morty.change("tease"),
        )
        self.wait(3)
        self.play(
            morty.change("raise_left_hand", look_at=3 * UR),
            stds[1].debubble("pondering", look_at=3 * UR),
            stds[2].change("tease"),
            stds[0].change("thinking"),
        )
        self.wait(5)


class ThinkAboutWhy(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.says("Think about\nwhy", mode="happy"),
            self.change_students("pondering", "thinking", "pondering", look_at=self.screen)
        )
        self.wait()
        self.play(self.teacher.change("raise_right_hand"))
        self.wait(5)


class PeriodicWords(InteractiveScene):
    def construct(self):
        # Test
        periodic = Text("Periodic")
        vertically = Text("Vertically")
        horizontally = Text("Horizontally")
        doubly = Text("Doubly")
        for word in [periodic, vertically, horizontally, doubly]:
            word.scale(1.5)
            word.set_backstroke(BLACK, 3)

        periodic.target = periodic.generate_target()

        group1 = VGroup(periodic, vertically)
        group2 = VGroup(doubly, periodic.target)
        for group in [group1, group2]:
            group.arrange(RIGHT, aligned_edge=UP, buff=0.35)
            group.to_edge(UP)
            group.set_backstroke(BLACK, 3)
        horizontally.move_to(vertically, UL)

        self.play(Write(group1, stroke_color=WHITE, run_time=2))
        self.wait()
        self.play(LaggedStart(
            FadeIn(horizontally, 0.75 * UP),
            FadeOut(vertically, 0.75 * UP),
            lag_ratio=0.2
        ))
        self.wait()
        self.play(
            MoveToTarget(periodic, path_arc=45 * DEG),
            FadeTransform(horizontally, doubly, path_arc=45 * DEG),
            run_time=2
        )
        self.wait()


class TransitionTo3b1bTalent(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(1000)

        words = Text("3b1b Talent")
        words.move_to(self.hold_up_spot, DOWN)

        self.play(
            self.change_students("confused", "tired", "concentrating", look_at=2 * UP),
            morty.change("tease")
        )
        self.wait(3)
        self.play(self.change_students("maybe", "erm", "tired"))
        self.wait(2)
        self.play(
            morty.change("raise_right_hand"),
            FadeIn(words, UP),
            self.change_students("pondering", "pondering", "pondering", look_at=morty.eyes),
        )
        self.wait(3)


class WaitWhat(TeacherStudentsScene):
    def construct(self):
        stds = self.students
        self.play(
            stds[2].says("Wait, what?", mode="confused", look_at=self.screen),
            stds[0].change("maybe", look_at=self.screen),
            stds[1].change("erm", look_at=self.screen),
            self.teacher.change("tease")
        )
        self.wait(3)


class UmWhy(TeacherStudentsScene):
    def construct(self):
        stds = self.students
        morty = self.teacher
        self.play(
            stds[1].says("Um, why?", mode="maybe", look_at=self.screen),
            stds[2].change("erm", look_at=self.screen),
            stds[0].change("pondering", look_at=self.screen),
        )
        self.play(morty.change("guilty"))
        self.wait(3)
        self.play(
            morty.says("Watch what\nhappens"),
            stds[1].debubble(),
            stds[0].change("hesitant", morty.eyes),
            stds[2].change("sassy", morty.eyes),
        )
        self.wait(3)


class SimplifyFormula(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        randy.to_edge(DOWN)
        randy.body.insert_n_curves(1000)

        kw = dict(t2c={"{c}": YELLOW, "{w}": PINK}, font_size=72)
        eq1 = Tex(R"e^{{c} \cdot \ln({w})}", **kw)
        eq2 = Tex(R"=\left(e^{\ln({w})}\right)^{c}", **kw)
        eq3 = Tex(R"=\left({w}\right)^{c}", **kw)
        eq4 = Tex(R"={w}^{c}", **kw)
        eq1.next_to(randy.get_corner(UL), UP, MED_LARGE_BUFF)
        for eq in [eq2, eq3, eq4]:
            eq.next_to(eq1, RIGHT, MED_SMALL_BUFF)

        self.play(
            randy.change("raise_left_hand", eq1),
            Write(eq1[R"\ln({w})"])
        )
        self.play(Write(eq1[R"{c} \cdot"]))
        self.play(Write(eq1[R"e"]), randy.change("pondering", eq1))
        self.play(Blink(randy))
        self.wait(2)

        # Simplify
        self.play(
            randy.change("raise_right_hand", eq2),
            TransformMatchingTex(eq1.copy(), eq2)
        )
        self.play(Blink(randy))
        self.wait()
        self.play(TransformMatchingTex(eq2, eq3))
        self.play(
            TransformMatchingTex(eq3, eq4),
            randy.change("tease")
        )
        self.play(Blink(randy))
        self.wait()

        # Add constants
        new_eq1 = Tex(R"e^{({c} \cdot \ln({w}) - z_0) + z_0}", **kw)
        new_eq1.move_to(eq1, RIGHT)
        new_eq2 = Tex(R"= A {w}^{c}", **kw)
        new_eq2.move_to(eq2, LEFT)

        self.play(
            randy.change("pondering", new_eq1),
            TransformMatchingTex(eq1, new_eq1),
            TransformMatchingTex(eq4, new_eq2),
        )
        self.play(Blink(randy))
        self.play(randy.animate.look_at(new_eq2))
        self.wait()

        # Highlight
        rect = SurroundingRectangle(new_eq2[1:], buff=0.2)
        rect.set_stroke(TEAL, 2)
        self.play(
            randy.change("raise_right_hand"),
            ShowCreation(rect),
            new_eq1.animate.set_opacity(0.5),
        )
        self.play(Blink(randy))
        self.wait()
        self.play(randy.change("confused", new_eq2))
        self.wait()
        self.play(randy.change("maybe", eq2))
        self.wait()


class ZoomInLine(InteractiveScene):
    def construct(self):
        # Test
        small_point = SMALL_BUFF * DL
        line = Line(16 * small_point, small_point)
        line.set_stroke(RED, width=(8, 3))
        dot = Dot(line.get_start())
        dot.set_color(RED)
        self.play(FadeIn(dot))
        self.play(
            ShowCreation(line),
            TransformFromCopy(dot, dot.copy().scale(0.25).move_to(line.get_end())),
            run_time=3
        )
        self.wait()


class CircleLoop(InteractiveScene):
    def construct(self):
        # Test
        circle = Circle(radius=3)
        circle.set_stroke(RED, 8)
        circle.insert_n_curves(100)
        circle.flip(RIGHT)
        circle.rotate(5 * PI / 4)
        dot = GlowDot(color=RED)
        dot.move_to(circle.get_start())
        tail = TracingTail(dot, stroke_color=RED, time_traced=3, stroke_width=(0, 8))
        self.add(tail)
        self.play(FadeIn(dot))
        self.play(
            MoveAlongPath(dot, circle),
            run_time=8
        )
        self.wait(3)


class HeadacheQuote(InteractiveScene):
    def construct(self):
        # Test
        text = Text(
            "The realization of this\nidea caused him “some\nalmighty headaches.”",
            font_size=48,
            alignment="LEFT"
        )
        line = Line(text.get_corner(UL), text.get_corner(DL))
        line.set_stroke(WHITE, 5)
        line.shift(MED_SMALL_BUFF * LEFT)
        line.scale(1.25)

        attribution = Text("From Bruno Ernst in\nThe Magic Mirror of M.C. Escher", font_size=30, alignment="LEFT")
        attribution.set_color(GREY_A)
        attribution.next_to(line, DOWN, LARGE_BUFF, aligned_edge=LEFT)

        self.add(line)
        self.play(
            FadeIn(text, lag_ratio=0.1, run_time=2),
            FadeIn(attribution, 0.25 * DOWN, time_span=(1, 2))
        )
        self.play(text["“some\nalmighty headaches.”"].animate.set_color(YELLOW))
        self.wait()


class AppreciatingWithMath(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        randy.to_edge(DOWN)

        path = Path(
            self.file_writer.get_output_file_rootname().parent.parent,
            'exponential/LogImage.png',
        )
        log_image = ImageMobject(path)
        log_image.set_width(7)
        log_image.to_corner(UL)
        log_label = Tex(R"\ln(z)", t2c={"z": BLUE})
        log_label.move_to(log_image).shift(0.45 * UP)
        log_group = Group(log_image, log_label)

        self.play(
            randy.change("raise_left_hand", 3 * UL),
            FadeIn(log_image, UP),
            Write(log_label),
        )
        self.play(Blink(randy))
        self.wait()
        self.play(randy.change("pondering", 3 * UR))
        self.play(Blink(randy))
        self.wait()

        # Add bubble
        rect = Rectangle(3, 1.5).set_opacity(0)
        bubble = randy.get_bubble(rect, direction=RIGHT)
        self.play(
            Write(bubble),
            log_group.animate.replace(rect, 0),
            randy.animate.look_at(bubble)
        )
        self.play(Blink(randy))
        self.play(randy.change("tease", 3 * UR))
        self.wait()
        self.play(Blink(randy))
        self.wait()


class EndScreen(SideScrollEndScreen):
    scroll_time = 30
