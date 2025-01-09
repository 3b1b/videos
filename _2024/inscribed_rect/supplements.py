from manim_imports_ext import *
from _2024.inscribed_rect.helpers import *


class UnsolvedQuestion(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.teacher.says("Here is an\nunsolved problem", mode="tease"),
            self.change_students("pondering", "sassy", "pondering"),
        )
        self.look_at(self.screen)
        self.wait(5)


class AskWhoCares(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            stds[0].change("guilty", stds[1].eyes),
            stds[1].says("Who cares?!", mode="angry", look_at=self.screen),
            stds[2].change("erm", stds[1].eyes),
            morty.change("guilty", stds[1].eyes),
        )
        self.wait(3)
        self.play(
            FadeOut(stds[1].bubble),
            stds[0].change("hesitant"),
            stds[1].says("What practial use\ndoes this have?", "sassy"),
        )
        self.wait(4)
        self.play(
            stds[0].change("thinking", morty.eyes),
            stds[1].debubble(),
            stds[2].change("pondering", morty.eyes),
            morty.says("Here's why\nI love it")
        )
        self.wait()


class PurePuzzleToApplication(InteractiveScene):
    def construct(self):
        # Set up pieces
        frame = self.frame
        puzzle_piece = SVGMobject("puzzle_piece")
        puzzle_piece.set_fill(BLUE_E, 1)
        puzzle_piece.set_height(2)
        puzzle_piece.insert_n_curves(50)

        gear = SVGMobject("gear")
        gear.set_fill(GREY_D, 1)
        gear.set_shading(0.4, 0.3, 0)
        gear.set_height(2)

        arrow = Vector(1.5 * RIGHT, thickness=8)

        pieces = VGroup(puzzle_piece, arrow, gear)
        pieces.arrange(RIGHT, buff=0.75)

        frame.match_x(puzzle_piece)
        frame.set_y(0.5)
        self.play(Write(puzzle_piece, run_time=3))

        # Insight
        lightbulb = SVGMobject("light_bulb")
        lightbulb.set_fill(YELLOW, 1)
        lightbulb.next_to(puzzle_piece, UP).shift(0.25 * UL)
        glow = GlowDot()
        glow.move_to(lightbulb)

        self.play(FadeIn(lightbulb, UP))
        self.play(glow.animate.set_radius(3))
        self.wait()

        # Carry over
        self.play(
            LaggedStart(
                TransformFromCopy(puzzle_piece, gear, run_time=2),
                GrowArrow(arrow),
                Group(lightbulb, glow).animate.match_x(gear).set_anim_args(run_time=2),
                lag_ratio=0.25,
            ),
            frame.animate.set_x(0),
            run_time=3
        )
        self.wait()


class WhatIsTopology(InteractiveScene):
    def construct(self):
        # Ask question
        question = Text("What is Topology?", font_size=72)
        question.to_edge(UP)
        underline = Underline(question, buff=-0.05)

        morty = Mortimer(height=3).flip()
        morty.move_to(DL)
        conf_morty = morty.copy().change_mode("confused")
        for pi in [morty, conf_morty]:
            for eye in pi.eyes:
                eye.scale(1.2, about_edge=DOWN)
        conf_morty.look_at(question)

        self.add(question)
        self.add(morty)
        self.play(
            Transform(morty, conf_morty),
            ShowCreation(underline)
        )
        self.play(Blink(morty))
        self.wait(2)

        # Answer 1
        answer1 = Text("Is it a study of bizarre shapes?", font_size=60)
        answer1.set_color(BLUE)
        answer1.next_to(question, DOWN, MED_LARGE_BUFF)

        self.play(
            FadeIn(answer1, lag_ratio=0.1, run_time=2),
            FadeOut(morty, DOWN)
        )
        self.wait()

        # Answer 2
        answer2 = TexText("Is it ``rubber sheet'' geometry?", font_size=60)
        answer2.set_color(BLUE)
        answer2.next_to(question, DOWN, MED_LARGE_BUFF)

        self.play(
            FadeIn(answer2, lag_ratio=0.1, run_time=2),
            answer1.animate.set_opacity(0.25).to_edge(DOWN)
        )
        self.wait()

        # Cross out both
        self.play(answer1.animate.set_opacity(1).next_to(answer2, DOWN, buff=MED_LARGE_BUFF), run_time=2)

        answers = VGroup(answer1, answer2)
        crosses = VGroup(Cross(a) for a in answers)
        crosses.set_stroke(RED, (0, 5, 5, 5, 0))

        self.play(ShowCreation(crosses, lag_ratio=0.25))
        self.wait()

        # How is this math?
        morty = Mortimer(height=2).flip()
        morty.to_corner(DL, buff=LARGE_BUFF)
        conf_morty = morty.copy().change_mode("maybe")
        for pi in [morty, conf_morty]:
            for eye in pi.eyes:
                eye.scale(1.2, about_edge=DOWN)
        bubble = morty.get_bubble("How is\nthis math?", SpeechBubble)

        self.play(
            FadeOut(answers),
            FadeOut(crosses),
            FadeIn(morty)
        )
        self.play(
            Transform(morty, conf_morty),
            Write(bubble),
        )
        self.play(Blink(morty))
        self.wait()


class ThreeShapes(InteractiveScene):
    def construct(self):
        # Test
        v_lines = Line(DOWN, UP).replicate(2)
        v_lines.set_height(FRAME_HEIGHT)
        v_lines.arrange(RIGHT, buff=FRAME_WIDTH / 3)
        v_lines.set_stroke(WHITE, 3)

        titles = VGroup(
            Text("Möbius Strip"),
            Text("Torus"),
            Text("Klein Bottle"),
        )
        for title, x in zip(titles, [-1, 0, 1]):
            title.scale(60 / 48)
            title.set_x(x * FRAME_WIDTH / 3)
            title.to_edge(UP)

        self.add(v_lines)
        self.add(titles)


class ThisIsTheSame(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer().flip()
        morty.to_edge(DOWN)
        self.play(morty.says("This is the\nsame question"))
        self.play(Blink(morty))
        self.wait()


class LabelMapping(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Mapping", font_size=72)
        title.to_edge(UP)
        title.set_color(BLUE)
        underline = Underline(title, buff=-0.1)
        underline.set_color(BLUE)

        body = VGroup(
            Text("Pair of\nloop points"),
            Vector(RIGHT),
            Text("3D space"),
        )
        body.arrange(RIGHT)
        body.next_to(title, DOWN, buff=0.5)

        self.play(FadeIn(title, 0.25 * UP), ShowCreation(underline))
        self.wait()
        self.play(FadeIn(body[0]))
        self.play(
            GrowArrow(body[1]),
            TransformFromCopy(*body[0::2])
        )
        self.wait()

        # Add continuity mention
        new_title = Text("Continuous Mapping", font_size=72)
        new_title.move_to(title, UP)
        new_title.match_style(title)

        self.play(
            Write(new_title["Continuous"]),
            Transform(title, new_title["Mapping"][0]),
            underline.animate.set_width(1.2 * new_title.get_width())
        )
        self.wait()


class ComplainAboutQuestion(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            morty.change("guilty"),
            stds[1].says("That's just a pretentious\nway to ask the same thing", "angry", look_at=self.screen),
            stds[0].change("hesitant", stds[1].eyes),
            stds[2].change("erm", stds[1].eyes),
        )
        self.wait(5)


class RememberThat(TeacherStudentsScene):
    def construct(self):
        self.play(self.teacher.says("Remember that"))
        self.play(
            self.change_students("pondering", "pondering", "thinking", look_at=self.screen)
        )
        self.wait(3)


class AskAboutACircle(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.play(
            stds[2].says("What does it look\nlike for a circle?", "maybe"),
            morty.change("tease", stds[2].eyes),
            stds[1].change("pondering", 4 * UR),
            stds[0].change("confused", 4 * UR),
        )
        self.wait(4)


class AskAboutProvingCollision(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        randy.to_edge(DOWN)
        self.play(randy.thinks("Why must there\nbe a collision?"))
        self.play(Blink(randy))
        self.play(randy.change("confused"))
        self.wait()


class GraphOfBellCurve(InteractiveScene):
    def construct(self):
        # Test
        axes = ThreeDAxes((-3, 3), (-3, 3), (0, 1))
        axes.set_depth(2, stretch=True)
        surface = axes.get_graph(lambda x, y: np.exp(-x**2 - y**2))
        surface.set_opacity(0.7)
        surface.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(surface, resolution=(21, 21))
        mesh.set_stroke(WHITE, 0.5, 0.5)

        self.frame.reorient(42, 67, 0, (0.42, 0.43, 0.1), 6.03)
        self.frame.add_ambient_rotation(3 * DEG)
        self.add(axes)
        self.add(surface, mesh)
        self.wait(30)


class GraphLabel(InteractiveScene):
    def construct(self):
        group = VGroup(
            Text("Graph of"),
            Tex(R"f(x, y) = e^{-x^2 - y^2}"),
        )
        group.arrange(DOWN)
        group.to_edge(UP)
        self.play(Write(group))
        self.wait()


class ChekhovsGun(InteractiveScene):
    def construct(self):
        # Add image
        image = ImageMobject("AntonChekhovPortrait")
        image.set_height(5)
        image.to_edge(LEFT, buff=1).to_edge(UP, buff=1)
        border = SurroundingRectangle(image, buff=0)
        border.set_stroke(GREY_B, 2)
        border.set_anti_alias_width(20)
        name = Text("Anton Chekhov")
        name.next_to(image, DOWN)

        self.play(
            Write(name),
            FadeIn(Group(image, border), UP)
        )
        self.wait()

        # Add quote
        quote = TexText(R"""
            ``If in the first act you \\
            have hung a pistol on the \\
            wall, then in the following \\
            one it should be fired.'' \\
        """, alignment="")
        quote.next_to(image, RIGHT, buff=2)

        self.play(Write(quote, run_time=4, lag_ratio=0.2))
        self.wait()


class XXLabel(InteractiveScene):
    def construct(self):
        tex = TexText(R"``Pairs'' of the form (X, X)", font_size=60)
        tex["X"].set_color(YELLOW)
        tex.to_edge(UP)
        self.play(Write(tex))


class CommutativeDiagram(InteractiveScene):
    samples = 4

    def construct(self):
        # Test
        objs = Square().set_height(3.5).replicate(3)
        objs[0].to_corner(UR).shift(0.5 * LEFT)
        objs[1].move_to(objs[0]).to_edge(DOWN)
        objs[2].next_to(objs[0], LEFT, buff=3)

        arrow_kw = dict(thickness=5)
        down_arrow = Arrow(
            objs[0].get_corner(DL) + UP,
            objs[1].get_left(),
            path_arc=PI,
            **arrow_kw
        )
        lr_arrow = VGroup(
            Arrow(objs[0], objs[2], **arrow_kw).shift(0.15 * UP),
            Arrow(objs[2], objs[0], **arrow_kw).shift(0.15 * DOWN),
        )

        words = Text("Another\nSurface", font_size=72)
        cloud = ThoughtBubble(words)[0][-1]
        cloud.add(words)
        cloud.move_to(objs[2], RIGHT).shift(0.25 * LEFT)

        dr_arrow = Arrow(cloud.get_bottom(), objs[1].get_left() + (-1, 0.25, 0), buff=0.3, **arrow_kw)

        self.play(Write(down_arrow))
        self.wait()
        self.play(
            *map(GrowArrow, lr_arrow),
            FadeIn(cloud, LEFT)
        )
        self.wait()
        self.play(GrowArrow(dr_arrow))
        self.wait()


class ContinuousAssociation(InteractiveScene):
    def construct(self):
        words = Text("Continuous\nAssociation", font_size=60)
        words.to_corner(UR)
        arrows = VGroup(
            Arrow(3 * LEFT, 3 * RIGHT, path_arc=PI / 2, thickness=6).shift(2.5 * DOWN),
            Arrow(3 * RIGHT, 3 * LEFT, path_arc=PI / 2, thickness=6).shift(2.5 * UP),
        )
        arrows.set_fill(TEAL)

        self.play(
            Write(words),
            *map(Write, arrows)
        )
        self.wait()


class ThinkLikeATopologist(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.says("Think like a\n topologist"),
            self.change_students("erm", "tease", "thinking", look_at=self.screen),
        )
        self.wait(5)


class RandyPuzzling(InteractiveScene):
    def construct(self):
        randy = Randolph()
        randy.to_edge(LEFT)
        self.play(randy.change("pondering", 2 * RIGHT))
        self.play(Blink(randy))
        self.wait()
        self.play(randy.change('confused', 2 * RIGHT))
        self.play(Blink(randy))
        self.wait(2)
        self.play(Blink(randy))
        self.play(randy.change('maybe', 2 * RIGHT))
        self.play(Blink(randy))
        self.wait()


class UnorderedPair(InteractiveScene):
    def construct(self):
        tex = TexText(R"Unordered pairs: $\{A, B\} = \{B, A\}$", font_size=60)
        tex.to_edge(UP)
        self.add(tex)


class XYtoYX(InteractiveScene):
    def construct(self):
        tex = Tex(R"(x, y) \leftrightarrow (y, x)")
        tex.to_edge(UP)
        self.add(tex)


class XXOnEdge(InteractiveScene):
    def construct(self):
        tex = TexText(R"``Pairs'' of the form $(X, X)$\\live on the edge")
        tex.to_corner(UL)
        self.add(tex)


class StripClaim(InteractiveScene):
    def construct(self):
        # Add title
        title = Text("Desired Claim", font_size=72)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        underline = Underline(title)
        VGroup(title, underline).set_color(BLUE)
        self.add(title, underline)

        # Add claim
        claim = Text("""
            A Möbius strip cannot be embedded in 3D
            with its edge on the xy-plane without
            intersecting itself.
        """, alignment="LEFT")
        claim.next_to(title, DOWN, buff=MED_LARGE_BUFF)
        underline.match_width(claim)
        self.play(Write(claim, run_time=2, lag_ratio=0.05))

        # False!
        morty = Mortimer()
        morty.to_corner(DR)
        self.play(
            VFadeIn(morty),
            morty.says("False!", mode="surprised"),
        )
        self.play(
            morty.bubble.content.animate.set_color(RED),
            Blink(morty)
        )
        self.wait()

        # Change the title
        new_title = Text("Revised Claim", font_size=72)
        new_title.set_color(TEAL)
        new_title.move_to(title)
        dots = Text("...")
        dots.next_to(claim["lane"], RIGHT, SMALL_BUFF, aligned_edge=DOWN)

        self.play(
            TransformMatchingStrings(
                title, new_title,
                key_map={"Desired": "Revised"},
                run_time=1,
                lag_ratio=0.5
            ),
            FadeOut(VGroup(morty, morty.bubble)),
            FadeOut(claim["without"], DOWN),
            FadeOut(claim["intersecting itself."], DOWN),
            Write(dots),
        )

        # New claim
        new_claim = Text("""
            A Möbius strip cannot be embedded in 3D
            with its edge on the xy-plane, and with
            its interior entirely above the xy-plane
            without intersecting itself.
        """, alignment="LEFT")
        new_claim.move_to(claim, UL)

        part1 = claim["with its edge on the xy-plane"]
        part2 = new_claim["its interior entirely above the xy-plane"]

        self.play(
            FlashUnder(part1, color=YELLOW, time_width=1.5),
            part1.animate.set_color(YELLOW),
        )
        self.wait()
        self.play(
            FadeOut(dots),
            FadeIn(new_claim[", and with"]),
            FadeIn(part2),
        )
        self.play(
            FlashUnder(part2, time_width=1.5, color=BLUE, run_time=2),
            part2.animate.set_color(BLUE)
        )
        self.wait()
        self.play(
            Write(new_claim["without intersecting itself."])
        )
        self.wait()

        # This one is true
        morty = Mortimer(height=1.5).to_corner(DR)

        self.play(
            VFadeIn(morty),
            morty.says("This one\nis true", mode="tease")
        )
        self.play(Blink(morty))
        self.wait()


class MobiusStripTextReflection(InteractiveScene):
    def construct(self):
        # Text
        text = Text("Möbius Strip", font_size=72)
        text.to_edge(LEFT)
        text.shift(UP)
        reflection = text.copy().flip(RIGHT)
        reflection.set_y(-text.get_y())
        self.add(text)
        self.wait()
        self.play(TransformFromCopy(text, reflection, run_time=3))
        self.wait()


class AskAboutUnsolved(InteractiveScene):
    def construct(self):
        randy = Randolph()
        randy.to_corner(DL)
        self.play(randy.says("What about the\nunsolved problem?", "maybe"))
        self.play(Blink(randy))
        self.wait(2)


class AskAbout4DEmbeddings(TeacherStudentsScene):
    def construct(self):
        # Hold up terms
        morty = self.teacher
        stds = self.students
        randy = stds[2]
        tup = Tex(R"(x, y, d, \theta)")
        tup.next_to(randy.get_corner(UL), UP, buff=1)

        four_d = Text("4D Space")
        cloud = ThoughtBubble(four_d)[0][-1]
        cloud.add(four_d)
        cloud.set_width(2.5)
        cloud.next_to(randy.get_corner(UR), UP, buff=0.7).shift(0.25 * RIGHT)

        strip = ParametricSurface(mobius_strip_func)
        strip.rotate(45 * DEG, LEFT)
        strip.set_color(BLUE_D, 0.5)
        strip.set_shading(0.3, 0.4, 0)
        strip.set_width(2)
        strip.match_x(cloud).to_edge(UP, buff=MED_SMALL_BUFF)
        arrow = Arrow(strip, cloud, buff=SMALL_BUFF)

        self.play(
            morty.change("tease", randy.eyes),
            self.change_students("thinking", "erm", "pondering", look_at=self.screen),
        )
        self.wait(2)
        self.play(
            stds[0].change("pondering", strip),
            stds[1].change("pondering", strip),
            randy.change("raise_right_hand", strip),
            ShowCreation(strip)
        )
        self.play(
            GrowArrow(arrow),
            FadeIn(cloud, DOWN)
        )
        self.wait()
        self.play(
            stds[0].change("pondering", tup),
            stds[1].change("pondering", tup),
            randy.change("raise_left_hand", tup),
            FadeIn(tup, UP)
        )
        self.wait(5)


class GreeneAndLobb(InteractiveScene):
    def construct(self):
        # Test
        names = VGroup(
            Text("Joshua Greene"),
            Text("Andrew Lobb"),
        )
        images = Group(
            ImageMobject("JoshuaGreene"),
            ImageMobject("AndrewLobb"),
        )
        for image in images:
            image.set_height(5)
        images.arrange(RIGHT, buff=1)
        for image, name in zip(images, names):
            name.next_to(image, DOWN)
            self.play(
                FadeIn(image, 0.5 * UP),
                Write(name)
            )
        self.wait()


class GreeneLobbTheorem(InteractiveScene):
    def construct(self):
        # Text
        text = Text("""
            Theorem by Greene and Lobb (2020):
            Every smooth closed curve contains
            inscribed rectangles of all possible
            aspect ratios.
        """, t2s={"smooth": ITALIC}, alignment="LEFT")
        text.to_corner(UL)

        title = text["Theorem by Greene and Lobb (2020):"][0]
        underline = Underline(title, stretch_factor=1.1, buff=-0.025)
        underline.set_color(TEAL)
        title.match_color(underline)
        body = text[len(title):]
        body.shift(0.1 * DOWN)

        self.play(
            FadeIn(title),
            ShowCreation(underline)
        )
        self.wait()
        self.play(Write(body, lag_ratio=0.05, run_time=3))
        self.wait()


class SmoothImplication(InteractiveScene):
    def construct(self):
        # Test
        smooth = Text("Smooth", font_size=72)
        implies = Tex(R"\Downarrow", font_size=120)
        tangent = Text("Every point has\na tangent line", font_size=60)
        words = VGroup(smooth, implies, tangent)
        words.arrange(DOWN, buff=0.5)
        words.to_edge(RIGHT)
        words.shift(UP)

        self.add(smooth)
        self.wait()
        self.play(
            Write(implies),
            FadeIn(tangent, 0.5 * DOWN),
        )
        self.wait()


class ProblemSolvingToRecreation(InteractiveScene):
    def construct(self):
        # Test
        words = VGroup(
            Text("Solving\nproblems"),
            Text("Helpful\nconstructs"),
            Text("Recreational\nmath"),
        )
        words.arrange(RIGHT, buff=2.5)
        words.set_y(-1.5)
        mind_bending = Text("(Sometimes Mind-bending)", font_size=24)
        mind_bending.next_to(words[1], DOWN)

        arrows = VGroup(
            Arrow(w1, w2, thickness=4, buff=0.45)
            for w1, w2 in zip(words, words[1:])
        )
        arrows.space_out_submobjects(1.1)

        self.play(FadeIn(words[0], lag_ratio=0.1))
        self.wait()
        self.play(
            FadeIn(words[1], lag_ratio=0.1),
            GrowArrow(arrows[0]),
        )
        self.wait()
        self.play(FadeIn(mind_bending, 0.25 * DOWN))
        self.wait()

        # Move mind-bending
        mb = mind_bending["Mind-bending"]
        helpful = words[1]["Helpful"]

        self.play(
            mb.animate.scale(2).move_to(helpful),
            FadeOut(helpful, 0.25 * UP),
            FadeOut(mind_bending["(Sometimes"][0]),
            FadeOut(mind_bending[")"][0]),
        )
        self.play(
            FadeIn(words[2], lag_ratio=0.1),
            GrowArrow(arrows[1]),
        )
        self.wait()
        self.play(
            words[0].animate.set_opacity(0.1),
            arrows[0].animate.set_opacity(0.1),
        )
        self.wait()


class WriteTopologicalSpace(InteractiveScene):
    def construct(self):
        # Test
        text = Text("Topological Space", font_size=72)
        text.to_edge(UP)
        self.play(Write(text))
        self.wait()


class AskAboutTopology(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students

        self.play(
            stds[0].says("What is a\ntopological space?", "raise_left_hand"),
            stds[1].change("pondering", stds[0].eyes),
            stds[2].change("pondering", stds[0].eyes),
            morty.change("tease"),
        )
        self.wait(2)
        self.play(
            stds[1].animate.look_at(stds[2].eyes),
            stds[2].says("What is\ntopology?", "maybe", look_at=morty.eyes, bubble_direction=LEFT)
        )
        self.wait()
        self.play(morty.change("guilty"))
        self.play(stds[0].change('confused'))
        self.play(morty.change("hesitant"))
        self.wait(4)


class PlaylistMention(InteractiveScene):
    def construct(self):
        # Test
        rects = ScreenRectangle().replicate(8)
        rects.set_fill(GREY_E, 1)
        rects.set_stroke(WHITE, 1)
        group = Group()
        for x, rect in zip(np.linspace(1, 0, len(rects)), rects):
            rect.shift(x * 0.75 * UR)
            group.add(rect)
            group.add(Point())

        thumbnail = ImageMobject("https://img.youtube.com/vi/OkmNXy7er84/maxresdefault.jpg")
        thumbnail.replace(rects[-1])
        group.add(thumbnail)

        group.set_height(2)
        group.to_edge(UP, buff=1.0)

        words = Text("More neat proofs\nand puzzle solutions")
        words.next_to(group, DOWN, buff=2.5)
        arrow = Arrow(words, group)
        self.play(
            FadeIn(group, UP),
            Write(words),
            GrowArrow(arrow)
        )
        self.wait()
        self.play(FadeOut(words), FadeOut(arrow))


class AskWhat(InteractiveScene):
    def construct(self):
        randy = Randolph()
        randy.to_edge(DOWN)

        # Test
        bubble = randy.get_bubble("Wait, what?")
        bubble.shift(0.5 * RIGHT + 0.25 * DR)
        self.play(
            randy.change("confused", UP),
            Write(bubble),
        )
        self.play(Blink(randy))
        for mode in ["maybe", "erm"]:
            self.play(randy.change(mode, UP))
            self.wait()
            self.play(Blink(randy))
            self.wait()


class EndScreen(PatreonEndScreen):
    title_text = ""
