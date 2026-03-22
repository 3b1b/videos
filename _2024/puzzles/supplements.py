from manim_imports_ext import *


class TableOfContents(InteractiveScene):
    def construct(self):
        # Test
        p_titles = VGroup(Text(f"Puzzle #{n}") for n in range(1, 6))
        subtitles = VGroup(map(Text, [
            "Twirling Tiles",
            "Tarski Plank Problem",
            "Monge's theorem",
            "Determining Det(M)",
            "The Hypercube Stack",
        ]))
        subtitles.set_color(GREY_B)
        titles = VGroup(VGroup(*pair) for pair in zip(p_titles, subtitles))
        for title in titles:
            title.arrange(DOWN, aligned_edge=LEFT)

        titles.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        titles.set_height(FRAME_HEIGHT - 1)
        titles.to_edge(LEFT)

        self.play(LaggedStartMap(FadeIn, titles, shift=LEFT, lag_ratio=0.25, run_time=2))
        self.wait()
        og_state = titles.copy()

        for i in range(5):
            target = og_state.copy()
            for j, title in enumerate(target):
                if i == j:
                    sf = 1.5
                    op = 1.0
                else:
                    sf = 0.75
                    op = 0.5
                title.scale(sf, about_edge=LEFT)
                title.set_opacity(op)
            target.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
            target.set_height(FRAME_HEIGHT - 1)
            target.to_edge(LEFT)
            self.play(Transform(titles, target))
            self.wait()

        self.play(Transform(titles, og_state))

        # Morty referencing
        rect = SurroundingRectangle(titles[:3])
        rect.set_stroke(YELLOW, 3)
        randy = Randolph(height=4).flip()
        randy.body.insert_n_curves(10)
        randy.next_to(titles, RIGHT, buff=1.5)
        randy.to_edge(DOWN)

        self.play(
            randy.change("tease"),
            VFadeIn(randy),
            ShowCreation(rect),
            titles[3:].animate.set_opacity(0.5)
        )
        self.play(Blink(randy))
        self.play(randy.change("hooray"))
        self.play(Blink(randy))
        self.wait()

        # Gather them up
        self.play(
            FadeOut(rect),
            FadeOut(randy),
            titles[3:].animate.set_opacity(1)
        )
        self.wait()


class AskHexagonQuestion(InteractiveScene):
    def construct(self):
        # Test
        question = Text("Can any tiling\n turn into any other?", font_size=60)
        question.to_corner(UL)

        arrow = Vector(2 * DOWN)
        arrow.next_to(question, DOWN)
        ifs = VGroup(Text("If no"), Text("If yes"))
        ifs.scale(0.75)
        ifs.next_to(arrow, RIGHT, SMALL_BUFF)
        ifs.set_submobject_colors_by_gradient(RED, GREEN)

        subquestions = VGroup(
            Text("How many connected\nsets are there?"),
            TexText(R"""
                Which two states are\\``farthest'' apart?
                \\ \quad \\
                How many moves\\away are they?
            """),
        )
        subquestions.next_to(arrow, DOWN)
        subquestions.set_color(GREY_A)

        self.play(Write(question))
        self.wait()
        self.play(
            GrowArrow(arrow),
            FadeIn(ifs[0], DOWN),
        )
        self.play(FadeIn(subquestions[0], lag_ratio=0.1))
        self.wait()
        self.play(
            FadeOut(VGroup(arrow.copy(), ifs[0], subquestions[0]), 3 * LEFT),
            GrowArrow(arrow),
            FadeIn(ifs[1], DOWN)
        )
        self.wait()
        self.play(FadeIn(subquestions[1], lag_ratio=0.1))
        self.wait()


class CounterTo64(InteractiveScene):
    def construct(self):
        value = Integer(64)
        value.scale(2)
        self.play(CountInFrom(value, 0), rate_func=linear, run_time=12)
        self.wait()


class NewNCubedArrow(InteractiveScene):
    def construct(self):
        arrow = Vector(1.5 * DOWN)
        label = Tex(R"N^3", font_size=72)
        label.next_to(arrow, RIGHT)
        self.play(GrowArrow(arrow))
        self.wait()
        self.play(Write(label))
        self.wait()


class SquintingPi(InteractiveScene):
    def construct(self):
        randy = Randolph(mode='pondering')
        randy.to_corner(LEFT)
        randy.look_at(ORIGIN)
        self.add(randy)
        self.play(Blink(randy))
        self.play(randy.change('concentrating'))
        self.play(Blink(randy))
        self.wait()
        self.play(randy.change('tease'))
        self.play(Blink(randy))
        self.wait()


class TwoDToThreeDInsight(InteractiveScene):
    def construct(self):
        # 2D -> 3D
        self.add(FullScreenRectangle())
        rects = Rectangle(4, 3).replicate(2)
        rects.set_width(FRAME_WIDTH // 2 - 1)
        rects.move_to(0.5 * DOWN)
        rects.arrange(RIGHT, buff=1)
        rects.set_fill(BLACK, 1)
        rects.set_stroke(WHITE, 2)

        titles = VGroup(Text("2D Puzzle"), Text("3D Insight"))
        for title, rect in zip(titles, rects):
            title.next_to(rect, UP)

        arrow = Arrow(titles[0].get_corner(UR), titles[1].get_corner(UL), path_arc=-45 * DEGREES)

        self.add(rects)
        self.add(titles)
        self.wait()
        self.play(GrowArrow(arrow, path_arc=-22 * DEGREES))
        self.wait()

        # Ask about four dimension
        new_titles = VGroup(Text("3D Puzzle"), Text("4D Insight"))
        for title, rect in zip(new_titles, rects):
            title.next_to(rect, UP)

        new_arrow = arrow.copy()

        left_shift = 8 * LEFT
        self.play(
            FadeOut(titles[0], left_shift),
            FadeOut(arrow, left_shift),
            TransformMatchingStrings(titles[1], new_titles[0])
        )
        self.play(
            GrowArrow(new_arrow, path_arc=-22 * DEGREES),
            FadeIn(new_titles[1], 4 * RIGHT)
        )
        self.wait()


class CanYouDoBetterThanTwo(InteractiveScene):
    def construct(self):
        morty = Mortimer().flip()
        morty.to_edge(LEFT)

        self.play(morty.says("Can you do\nbetter than 2?", mode="tease"))
        self.play(Blink(morty))
        self.wait()
        self.play(morty.change("pondering"))
        self.wait()


class AreWeSupposedToKnowThat(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.play(
            stds[0].change("confused", look_at=self.screen),
            stds[1].change("erm", look_at=self.screen),
            stds[2].says("Are we supposed\nto know that?", mode="pleading", look_at=self.screen, bubble_direction=LEFT),
            morty.change("guilty"),
        )
        self.wait(4)

        self.play(
            morty.change("raise_right_hand"),
            stds[2].debubble(mode="hesitant", look_at=morty.eyes),
            stds[0].change("maybe", morty.eyes),
            stds[1].change("pondering", morty.eyes),
        )
        self.wait(2)
        self.play(self.change_students("pondering", "pondering", "pondering", 2 * UR))
        self.wait(3)


class AddAreas(InteractiveScene):
    def construct(self):
        # Test
        parts = [
            R"\sum_i \textbf{Area}(\text{Hemisphere Strip}_i)",
            R"= \sum_i \pi d_i",
            R"= \pi \sum_i d_i",
            R"\ge 2\pi"
        ]
        top_eq = Tex(" ".join(parts))
        top_eq.to_edge(UP)

        for part in parts:
            self.play(FadeIn(top_eq[part][0], 0.5 * UP))
            self.wait()


class CylinderAreaAnnotation(InteractiveScene):
    def construct(self):
        # Test
        expr = Tex(R"\text{Area} = 2 \pi R \cdot d")
        expr.to_corner(UL)
        circum_brace = Brace(expr[R"2 \pi R"])
        circum_label = Text("Circumference", font_size=30)
        circum_label.next_to(circum_brace, DOWN, SMALL_BUFF)

        thickness_rect = SurroundingRectangle(expr["d"])
        thickness_rect.set_stroke(BLUE)
        thickness_label = Text("Thickness", font_size=30)
        thickness_label.set_color(BLUE)
        thickness_label.next_to(thickness_rect, RIGHT, SMALL_BUFF)

        self.add(expr)
        self.wait()
        self.play(
            GrowFromCenter(circum_brace),
            FadeIn(circum_label, lag_ratio=0.1),
        )
        self.play(
            ShowCreation(thickness_rect),
            FadeIn(thickness_label, lag_ratio=0.1),
        )
        self.wait()


class CircleWithinAnother(TeacherStudentsScene):
    def construct(self):
        stds = self.students
        morty = self.teacher
        self.play(
            stds[2].says("What if one circle\nis inside another?", look_at=morty.eyes, bubble_direction=LEFT),
            morty.change("tease"),
            stds[1].change("pondering", self.screen),
            stds[0].change("pondering", self.screen),
        )
        self.wait(4)


class MentionProjectiveGeometry(TeacherStudentsScene):
    def construct(self):
        stds = self.students
        morty = self.teacher
        self.play(
            stds[1].says("You could use\n projective geometry!", mode="hooray", look_at=morty.eyes),
            morty.change("happy"),
            stds[2].change("confused", stds[1].eyes),
            stds[0].change("erm", stds[1].eyes),
        )
        self.wait(4)


class SolicitMore(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        self.play(
            morty.says("Any other\nexamples?"),
            self.change_students("pondering", "pondering", "thinking")
        )
        self.wait(2)
        self.play(
            morty.debubble(mode='tease'),
            self.students[0].says("Monge's theorem!", mode="hooray"),
            self.students[1].animate.look_at(self.students[0].eyes),
            self.students[2].animate.look_at(self.students[0].eyes),
        )
        self.wait(4)

        # Point out a problem
        self.play(self.students[0].debubble(mode="pondering", look_at=self.screen))
        self.look_at(self.screen)
        self.wait(3)

        self.play(
            self.students[2].says("Hang on, that\ndoesn't work...", mode="sassy", look_at=self.screen, bubble_direction=LEFT),
            morty.change("guilty"),
        )
        self.wait(5)


class SlovokiaTeam(InteractiveScene):
    def construct(self):
        # Test
        talker = PiCreature(color=RED_E)
        talker.to_edge(DOWN).shift(3 * LEFT)
        flag = ImageMobject("flags/sk.png")
        flag.set_height(1)
        flag.next_to(talker, LEFT)
        name = Text("Akos Zahorsky", font_size=24)
        name.next_to(flag, DOWN, buff=SMALL_BUFF)

        morty = Mortimer()
        morty.to_edge(DOWN).shift(3 * RIGHT)

        self.play(
            talker.says("You can save\nthis proof.", mode="speaking", look_at=morty.eyes),
            morty.change("pondering", talker.eyes)
        )
        self.play(FadeIn(flag, UP), morty.change("tease"))
        self.play(Write(name))
        self.play(Blink(morty))
        self.play(Blink(talker))
        self.wait()


class HowIsThatHelpful(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.play(
            stds[2].says(
                "How is that\nhelpful?",
                mode="raise_left_hand",
                look_at=morty.eyes,
                bubble_direction=LEFT,
            ),
            stds[1].change("confused", self.screen),
            stds[0].change("pondering", self.screen),
            morty.change("guilty")
        )
        self.wait(3)


class AskTetrahedronQuestion(InteractiveScene):
    def construct(self):
        question = Text("What is the volume\nof this solid?")
        # question = TexText(R"""
        #     What is the volume in terms of \\
        #     $(x_1, y_1, z_1)$, $(x_2, y_2, z_2)$\\
        #     $(x_3, y_3, z_3)$, and $(x_4, y_4, z_4)$
        # """)
        question.to_corner(UL)

        self.play(Write(question))
        self.wait()


class WillNotAnswer(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(LaggedStart(
            morty.says("I won't give\nyou the answer"),
            self.change_students("angry", "erm", "sassy"),
            lag_ratio=0.5
        ))
        self.wait(3)

        # Spoiler
        det_expr = Tex(R"""
            \text{det}\left(\left[
            \begin{array}{cccc}
            \cdot & \cdot & \cdot & \cdot \\
            \cdot & \cdot & \cdot & \cdot \\
            \cdot & \cdot & \cdot & \cdot \\
            \cdot & \cdot & \cdot & \cdot \\
            \end{array}
            \right]\right)
        """)
        det_expr.next_to(self.hold_up_spot, UP)

        self.play(
            FadeIn(det_expr, UP),
            morty.debubble(mode="raise_right_hand", look_at=det_expr),
            self.change_students("pondering", "pondering", "pondering", look_at=det_expr),
        )
        self.wait(4)


class SpoilerAlert(InteractiveScene):
    def construct(self):
        # Test
        tri = RegularPolygon(3)
        tri.set_height(3)
        tri.round_corners(radius=0.25)
        tri.set_stroke(RED, 20)
        tri.move_to(UP)

        bang = Tex("!", font_size=240)
        bang.move_to(interpolate(tri.get_bottom(), tri.get_top(), 0.45))
        bang.match_color(tri)

        words = Text("Spoiler", font_size=120)
        words.next_to(tri, DOWN, buff=0.5)
        words.set_color(RED)

        self.add(tri)
        self.play(FadeIn(words, scale=1.25), Write(bang), run_time=1)
        self.wait()


class SolveThisExplainThat(InteractiveScene):
    def construct(self):
        self.add(FullScreenRectangle())
        screens = Rectangle(4, 3).replicate(2)
        screens.set_fill(BLACK, 1)
        screens.set_stroke(WHITE, 2)
        screens.set_height(4.5)
        screens.arrange(RIGHT, buff=1)
        screens.move_to(0.5 * DOWN)

        titles = VGroup(
            Text("Solving this..."),
            Text("...explains this"),
        )
        for title, screen in zip(titles, screens):
            title.next_to(screen, UP)

        self.add(screens)
        self.play(Write(titles[0]))
        self.wait()
        self.play(TransformMatchingStrings(titles[0].copy(), titles[1], path_arc=-45 * DEGREES))
        self.wait()


class DeterminantFormula(InteractiveScene):
    def construct(self):
        # Test
        det_expr = Tex(R"""
            \text{det}\left(\left[
            \begin{array}{ccc}
            a & b & c \\
            d & e & f \\
            g & h & i \\
            \end{array}
            \right]\right)
        """)
        det_expr.scale(1.25)
        det_expr.next_to(ORIGIN, LEFT).set_y(2)
        cols = VGroup(det_expr[k:16:3] for k in range(7, 10))
        col_syms = ["adg", "beh", "cfi"]

        equals = Tex(R"=", font_size=60)
        equals.next_to(det_expr, RIGHT)

        n = 3
        rhs_terms = VGroup()
        perms = list(it.permutations(range(n)))
        for perm in perms:
            parity = sum(int(j < i) for (i, j) in it.combinations(perm, 2))
            sgn = "+" if (parity % 2 == 0) else "-"
            term = Tex(sgn + "".join([col_syms[i][j] for i, j in enumerate(perm)]))
            rhs_terms.add(term)

        rhs_terms.scale(1.25)
        rhs_terms.arrange(DOWN)
        rhs_terms.next_to(equals, RIGHT, aligned_edge=UP)
        rhs_terms.shift(0.2 * UP)

        self.add(det_expr)
        self.add(equals)
        last_highlight = VGroup()
        for perm, term, sgn in zip(perms, rhs_terms, it.cycle([1, -1])):
            highlight = VGroup(cols[i][j] for i, j in enumerate(perm)).copy()
            highlight.set_fill(BLUE if sgn > 0 else RED, 1, border_width=2)
            self.play(
                FadeIn(highlight),
                FadeOut(last_highlight),
                cols.animate.set_opacity(0.5)
            )
            self.play(
                FadeIn(term[0]),
                TransformFromCopy(highlight, term[1:]),
            )

            last_highlight = highlight
        self.play(FadeOut(last_highlight), cols.animate.set_opacity(1))


class Seeking4DAnalog(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle())
        screens = Rectangle(4, 3).replicate(2)
        screens.set_fill(BLACK, 1)
        screens.set_stroke(WHITE, 2)
        screens.set_height(4.5)
        screens.arrange(RIGHT, buff=1)
        screens.move_to(0.5 * DOWN)

        titles = VGroup(
            TexText(R"2d tiling \\ $\downarrow$ \\ 3d cube stack"),
            TexText(R"3d tiling \\ $\downarrow$ \\ 4d hypercube stack"),
        )
        for title, screen in zip(titles, screens):
            title.next_to(screen, UP)

        self.add(screens)
        self.play(Write(titles[0]))
        self.wait()
        self.play(TransformMatchingStrings(
            titles[0].copy(), titles[1],
            key_map={"2d": "3d", "3d": "4d"}
        ))
        self.wait()


class ShowConfusion(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        self.play(
            morty.change("guilty"),
            self.change_students("concentrating", "droopy", "sad", look_at=self.screen)
        )
        self.wait(3)
        self.play(self.change_students("maybe", "tired", "concentrating", look_at=self.screen))
        self.wait(3)
        self.play(
            morty.says("A little mental gymnastics\nkeeps your mind flexible", mode="tease"),
            self.change_students("hesitant", "erm", "thinking", look_at=morty.eyes)
        )
        self.wait(5)


class ReversePuzzleAndInsight(InteractiveScene):
    def construct(self):
        # Add
        bulb = SVGMobject("light_bulb").set_color(YELLOW)
        puzzle = SVGMobject("puzzle_piece").set_color(BLUE_D)
        icons = VGroup(puzzle, bulb)
        icons.scale(0.75)
        titles = VGroup(Text("Puzzle"), Text("Insight"))
        titles.scale(1.25)
        objs = VGroup(VGroup(*pair) for pair in zip(titles, icons))
        for obj in objs:
            obj[1].set_height(1.5)
            obj.arrange(UP)
        objs.arrange(RIGHT, buff=3.5)
        puzzle.match_y(bulb).align_to(titles[0], LEFT)
        arrow = Arrow(puzzle, bulb, thickness=8, buff=0.3)

        self.play(LaggedStart(
            AnimationGroup(
                FadeIn(titles[0], lag_ratio=0.1),
                FadeIn(puzzle, 0.5 * UP),
            ),
            GrowArrow(arrow),
            AnimationGroup(
                FadeIn(titles[1], lag_ratio=0.1),
                FadeIn(bulb, scale=2),
            ),
            lag_ratio=0.7
        ))
        self.wait()

        # Swap
        self.play(LaggedStart(
            objs[0].animate.next_to(arrow, RIGHT, buff=0.3).shift(0.2 * DOWN).set_anim_args(path_arc=PI / 2),
            objs[1].animate.next_to(arrow, LEFT, buff=0.3).shift(0.2 * DOWN).set_anim_args(path_arc=PI / 2),
            lag_ratio=0.15
        ))
        self.wait()


class ProjectionFormula4D(InteractiveScene):
    def construct(self):
        # Test
        t2c = {R"\hat{\textbf{d}}": BLUE_D, R"\vec{\textbf{v}}": YELLOW}
        form = Tex(
            R"\text{Project}(\vec{\textbf{v}}) = \vec{\textbf{v}} - \left(\vec{\textbf{v}} \cdot \hat{\textbf{d}} \right) \hat{\textbf{d}}",
            t2c=t2c
        )
        # form.to_corner(UL)
        form.move_to(UP)

        definition = Tex(
            R"\text{Where } \hat{\textbf{d}} := \frac{1}{\sqrt{4}} \left[\begin{array}{c} 1 \\ 1 \\ 1 \\ 1 \end{array}\right]",
            font_size=36,
            t2c=t2c
        )
        definition.next_to(form, DOWN, buff=0.5)

        VGroup(form, definition).set_backstroke(BLACK, 5)

        self.add(form)
        self.play(FadeIn(definition, 0.25 * DOWN))
        self.wait()


class ConfusionThenAnswer(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.play(
            morty.change("guilty"),
            self.change_students("maybe", "tired", "concentrating", look_at=self.screen)
        )
        self.wait()

        # Answer
        answer = TexText(R"Maximum number \\ of moves: $N^4$")
        answer.next_to(stds[2].get_corner(UR), UP, buff=LARGE_BUFF)
        answer["N^4"].set_color(YELLOW)

        self.play(
            stds[2].change("raise_right_hand", answer)
        )
        self.play(
            self.change_students("pondering", "pondering", "raise_right_hand", look_at=answer),
            morty.change("tease"),
            FadeIn(answer, 0.5 * UP)
        )
        self.wait(5)


class IMOTitles(InteractiveScene):
    def construct(self):
        # Test
        logo = ImageMobject("IMO_logo")
        logo.set_height(1.25)
        title1 = Text("International Math Olympiad 2024")
        title2 = Text("IMO 2024", font_size=60)
        for title in title1, title2:
            title.to_edge(UP)
            title.shift(logo.get_width() * RIGHT / 2)
        logo.next_to(title1, LEFT)

        self.add(logo, title1)
        self.wait()
        self.play(
            TransformMatchingStrings(
                title1, title2,
                key_map={
                    "International": "I",
                    "Math": "M",
                    "Olympiad": "O",
                },
                run_time=1.5,
                lag_ratio=0.1
            ),
            logo.animate.next_to(title2, LEFT),
        )
        self.wait()


class DimensionsPointingToEachOther(InteractiveScene):
    def construct(self):
        # Show shapes
        shapes = VGroup(
            VGroup(Line(LEFT, RIGHT)),
            VGroup(Square(side_length=2)),
            self.get_cube(),
            self.get_hypercube(),
        )
        shapes.arrange(RIGHT, buff=1.0)
        shapes.set_submobject_colors_by_gradient(TEAL, YELLOW)

        point = VectorizedPoint()

        self.add(shapes[0])
        for s1, s2 in zip(shapes, shapes[1:]):
            point.move_to(s1).align_to(shapes, UP)
            arrow = Arrow(s2.get_top(), point.get_center(), buff=0.1, path_arc=120 * DEGREES, thickness=4)
            self.play(
                FadeIn(arrow),
                TransformFromCopy(s1, s2, lag_ratio=0.5 / len(s2), run_time=1.5),
            )
            self.wait(0.25)

    def get_cube(self, side_length=2.0):
        cube = VCube(side_length=side_length)
        cube.set_stroke(WHITE, 2)
        cube.set_fill(opacity=0)
        cube.rotate(20 * DEGREES, OUT).rotate(80 * DEGREES, LEFT)
        return cube

    def get_hypercube(self, side_length=2.0, theta=20 * DEGREES, phi=80 * DEGREES):
        cubes = VGroup(
            VCube().set_height(side_length),
            VCube().set_height(side_length / 2),
        )
        for cube in cubes:
            cube.sort(lambda p: p[2])
        edges = VGroup()
        for i in [0, -1]:
            edges.add(*(
                Line(*pair)
                for pair in zip(cubes[0][i].get_vertices(), cubes[1][i].get_vertices())
            ))

        result = VGroup(*cubes, edges)
        result.set_stroke(WHITE, 2)
        result.set_fill(opacity=0)
        result.rotate(theta, OUT)
        result.rotate(phi, LEFT)
        return VGroup(*result.family_members_with_points())


class SingleHypercube(DimensionsPointingToEachOther):
    samples = 4

    def construct(self):
        # Test
        frame = self.frame
        self.set_floor_plane("xz")
        frame.reorient(-23, -17, 0, (0.0, -0.4, 0.2), 9.23)

        cube = self.get_hypercube(theta=0, phi=0)
        cube.set_height(5)
        cube.set_stroke(YELLOW, 5)
        cube.set_flat_stroke(True)
        points = []
        for edge in cube.family_members_with_points():
            n_point = int(edge.get_arc_length() * 15)
            for a in np.linspace(0, 1, n_point):
                points.append(edge.pfp(a))

        fuzz = GlowDots(points)
        fuzz.set_radius(0.2)
        fuzz.set_opacity(0.2)
        self.add(cube)
        self.add(fuzz)


class SpherePacking(InteractiveScene):
    samples = 4

    def construct(self):
        # Test
        frame = self.frame
        sphere = TrueDot(radius=math.sqrt(2) / 2)
        sphere.make_3d()
        max_x = 12
        spheres = Group(
            sphere.copy().move_to(coords)
            for coords in it.product(*3 * [np.arange(0, max_x)])
            if sum(coords) % 2 == 0
            if get_norm(coords) < max_x
        )
        spheres.sort(lambda p: get_norm(p))
        for sphere in spheres:
            sphere.set_color(random_bright_color(hue_range=(0.50, 0.55), luminance_range=(0.25, 0.5)))

        frame.reorient(89, 71, 0, (0, 0, 1), 10)
        self.play(
            LaggedStart(
                (FadeIn(sphere, scale=1.2)
                for sphere in spheres),
                lag_ratio=0.25,
                group_type=Group
            ),
            frame.animate.reorient(170, 64, 0, (2.7, -1.16, 1.8), 22.64),
            run_time=25
        )
        self.wait()


class Shruggie(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer(mode="shruggie").flip()
        self.add(morty)
        random.seed(1)
        for _ in range(10):
            x = random.random()
            if x < 0.25:
                self.play(Blink(morty))
            else:
                self.wait()


class GolayCode(InteractiveScene):
    def construct(self):
        # Set up dots
        dots = Dot().get_grid(23, 12)
        dots.set_height(FRAME_HEIGHT - 1)
        dots.set_stroke(WHITE, 1)
        dots.set_fill(BLACK)
        self.add(dots)

        # Color
        for i in range(12):
            dots[12 * i + i].set_fill(GREY_B)
        pattern = np.array([
            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
        ])

        for i, row in zip(range(12, 23), pattern):
            for j, bit in enumerate(row):
                dots[12 * i + j].set_fill(GREY_B if bit else BLACK)

        self.play(Write(dots))
        self.wait()


class AskAboutTilingBijection(InteractiveScene):
    def construct(self):
        # Test
        arrows = VGroup(
            Tex(R"\longleftarrow", font_size=180),
            Tex(R"\longrightarrow", font_size=180),
        )
        arrows.stretch(1.5, 0)
        arrows.arrange(DOWN)
        top_text = Text("Each stack clearly\ngives a tiling.", font_size=42)
        low_text = Text("But does such tiling\nnecessarily have a stack?", font_size=42)
        top_text.next_to(arrows, UP)
        low_text.next_to(arrows, DOWN)

        self.add(arrows[0], top_text)
        self.wait()
        self.play(Write(low_text), GrowFromCenter(arrows[1]))
        self.wait()


class AnalysisIntuitionFraming(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle())
        squares = Rectangle(4, 3).replicate(2)
        squares.set_height(4.5)
        squares.set_fill(BLACK, 1)
        squares.set_stroke(WHITE, 2)
        squares.arrange(RIGHT, buff=1.0)
        squares.move_to(0.5 * DOWN)

        titles = VGroup(Text("Analysis"), Text("Intuition"))
        titles.scale(1.5)
        for title, square in zip(titles, squares):
            title.next_to(square, UP)
        titles[0].align_to(titles[1], UP)

        self.add(squares)
        self.play(LaggedStartMap(FadeIn, titles))
        self.wait()


class LightBulb(InteractiveScene):
    def construct(self):
        # Add bulb
        bulb = SVGMobject("light_bulb")
        bulb.set_fill(YELLOW, 0.1)
        bulb.set_height(3)
        fancy_bulb = VGroup()
        for part in bulb:
            for x in range(20):
                fancy_bulb.add(part.copy())
                part.scale(0.97)

        self.add(fancy_bulb)

        # Light
        N = 300
        radiation = VGroup(
            Circle(radius=interpolate(0.5, 10, a)).set_stroke(YELLOW, 2, interpolate(0.5, 0.2, a))
            for a in np.linspace(0, 1, N)
        )

        self.play(LaggedStartMap(VFadeInThenOut, radiation, lag_ratio=1.0 / N, run_time=5))


class PileOfEquations(InteractiveScene):
    def construct(self):
        # Test
        equations = VGroup(
            Tex(R"\mathbf{x} = \big\{(x_1, x_2, \ldots, x_8) \quad | x_i \in \mathbb{Z} \text{ or } x_i \in \mathbb{Z} + \frac{1}{2}\big\}"),
            Tex(R"\Delta(\Lambda) = \frac{V_n}{\det(\Lambda)}"),
            Tex(R"V_n = \frac{\pi^{n/2} r^n}{\Gamma(n/2 + 1)} \rightarrow V_8 = \frac{\pi^4}{24}"),
            Tex(R"\theta_{E_8}(z) = \sum_{\mathbf{x} \in E_8} e^{\pi i z |\mathbf{x}|^2}"),
            Tex(R"|\text{Aut}(E_8)| = 696,729,600 = 2^{14} \cdot 3^5 \cdot 5^2 \cdot 7"),
            Tex(R"\Delta_{\text{max}} \leq \frac{\pi^4}{384} + \epsilon"),
            Tex(R"\Delta \leq \frac{2^{-n}n}{e} \cdot \frac{\text{Vol}(B^n_1)}{\text{Vol}(V)}"),
        )
        equations.arrange(DOWN, aligned_edge=LEFT)
        equations.set_height(FRAME_HEIGHT - 1)
        self.play(LaggedStartMap(Write, equations, lag_ratio=0.5, run_time=7))
        self.wait()


class Obvious(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        self.play(randy.says("Isn't that\nobvious?", mode="confused", bubble_direction=LEFT))
        self.play(randy.animate.look_at(3 * UP))
        self.wait(5)


class BonusVideo(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer(mode="tease", height=4)
        morty.to_edge(DOWN, buff=1).shift(2 * RIGHT)
        self.play(Blink(morty))
        self.play(morty.says("Bonus video!", mode="hooray"))
        self.play(Blink(morty))
        self.wait(2)


class BonusVideoMention(InteractiveScene):
    def construct(self):
        pass


class EndScene(PatreonEndScreen):
    pass
