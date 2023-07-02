from manim_imports_ext import *


def moser(n):
    return choose(n, 4) + choose(n, 2) + 1


class Introduction(InteractiveScene):
    def construct(self):
        # Title
        tale = SVGMobject("cautionary_tale.svg")
        tale.set_fill(GREY_A, 1)
        tale.set_stroke(GREY_A, 2)
        tale.set_width(7)
        tale.center()
        tale.use_winding_fill(False)
        paths = VGroup(*(
            VMobject().set_points(path)
            for path in tale[0].get_subpaths()
        ))
        paths.set_fill(opacity=0)
        paths.set_stroke(width=0)
        paths.sort(lambda p: np.dot(p, RIGHT))
        rect = FullScreenRectangle()
        rect.set_fill(BLACK, 1)
        rect.set_stroke(WHITE, 0)
        rect.move_to(tale, LEFT)

        self.add(tale)
        self.play(
            rect.animate.next_to(tale, RIGHT, buff=LARGE_BUFF).set_anim_args(rate_func=linear),
            Write(paths, stroke_width=5),
            run_time=3,
        )
        self.remove(rect, paths)
        self.wait()

        # Name
        name = TexText(
            R"Moser's Circle Problem",
            font_size=72,
        )
        name.to_edge(UP)

        self.play(
            FadeOut(tale),
            FadeIn(name, UP)
        )
        self.wait(2)


class ShowPattern(InteractiveScene):
    def construct(self):
        # Show expression
        N = 11
        values = [moser(n) for n in range(1, N + 1)]
        expression = Tex(
            R",".join(map(str, values)) + R"\dots"
        )
        expression.set_width(FRAME_WIDTH - 3)
        expression.to_edge(UP)

        n = 0
        parts = VGroup()
        for value in values:
            new_n = n + len(str(value))
            parts.add(expression[n:new_n])
            self.play(FadeIn(expression[max(n - 1, 0):new_n], 0.25 * UP, run_time=0.5))
            self.wait(0.5)
            n = new_n + 1

        self.play(Write(expression[-3:]))
        self.wait()

        # Ask about expression
        brace1 = Brace(expression, DOWN)
        brace2 = Brace(expression["1,2,4,8,16"], DOWN)
        brace3 = Brace(expression["256"], DOWN)
        question = brace1.get_text("What is this pattern?")
        coincidence = brace2.get_text("Coincidence?")
        what = brace3.get_text("And what's with this?")

        VGroup(question, coincidence, what).set_color(BLUE)

        self.play(
            GrowFromCenter(brace1),
            FadeIn(question, 0.5 * DOWN),
        )
        self.wait()
        self.play(
            ReplacementTransform(brace1, brace2),
            FadeTransform(question, coincidence),
        )
        self.wait()
        self.play(
            ReplacementTransform(brace2, brace3),
            FadeTransform(coincidence, what),
        )
        self.wait()
        self.play(FadeOut(brace3), FadeOut(what))

        # Ask about the function
        fn = Tex(R"f(n) = \, ???", font_size=60)
        fn.next_to(expression, DOWN, buff=1.5)

        self.play(Write(fn))

        last_rect = VGroup()
        last_term = VGroup()
        for n, part in zip(it.count(1), parts):
            rect = SurroundingRectangle(part, buff=SMALL_BUFF)
            rect.set_stroke(YELLOW, 2)
            term = Tex(fR"f({n})", font_size=48)
            term.set_color(YELLOW)
            term.next_to(rect, DOWN)
            self.play(
                FadeIn(term), FadeIn(rect),
                FadeOut(last_term), FadeOut(last_rect),
                run_time=0.5
            )
            self.wait(0.5)
            last_term = term
            last_rect = rect


class AskAboutPosition(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.remove(self.background)

        self.play(LaggedStart(
            stds[1].says("Doesn't it depend on\nwhere the points are?", mode="sassy"),
            stds[0].change('confused', self.screen),
            stds[2].change('pondering', self.screen),
            morty.change("guilty", stds[1].eyes)
        ))
        self.wait(5)


class ExplainNChoose2(InteractiveScene):
    n = 7
    fudge_factor = 0.1

    def construct(self):
        # Setup circle
        circle, chords, dots, numbers = diagram = self.get_circle_diagram()
        self.add(*diagram)

        # Ask question
        question = Text("How many pairs of points?")
        question.to_edge(RIGHT, MED_LARGE_BUFF)
        question.to_edge(UP, MED_SMALL_BUFF)
        question.add(Underline(question))
        self.add(question)

        # Show all pairs
        indices = list(range(self.n))
        pair_labels = VGroup()
        pair_label_template = Tex("(0, 0)")
        last_label = VectorizedPoint()
        last_label.next_to(question, DOWN)
        last_label.shift(1.5 * LEFT)

        for i, j in it.combinations(indices, 2):
            label = pair_label_template.copy()
            values = label.make_number_changable("0", replace_all=True)
            values[0].set_value(i + 1)
            values[1].set_value(j + 1)
            label.next_to(last_label, DOWN)
            if label.get_y() < -3:
                label.next_to(pair_labels, RIGHT, MED_LARGE_BUFF)
                label.align_to(pair_labels, UP)
            pair_labels.add(label)

            chords.set_opacity(0.25)
            dots.set_opacity(0.25)
            numbers.set_opacity(0.25)
            temp_line = Line(dots[i].get_center(), dots[j].get_center())
            temp_line.set_stroke(BLUE_B, 3)
            for mob in [dots[i], dots[j], numbers[i], numbers[j]]:
                mob.set_opacity(1)

            self.add(pair_labels)
            self.add(temp_line)
            self.wait(0.25)
            self.remove(temp_line)

            last_label = label

        self.play(
            chords.animate.set_opacity(1),
            numbers.animate.set_opacity(1),
            dots.animate.set_opacity(1),
        )
        self.wait()

        # Show n choose 2
        nc2 = Tex(R"{n \choose 2}")
        nc2.set_color(YELLOW)
        nc2_label = TexText("``n choose 2''")

        group = VGroup(nc2, nc2_label)
        group.arrange(RIGHT, buff=MED_LARGE_BUFF)
        group.next_to(question, DOWN)

        self.play(
            FadeIn(nc2, DOWN),
            pair_labels.animate.set_height(4).to_edge(DOWN)
        )
        self.play(Write(nc2_label))
        self.wait()

        # Show counts
        n_value = Integer(1)
        n_value.move_to(nc2[1])
        n_value.set_color(YELLOW)

        number_rects = VGroup(*(SurroundingRectangle(number) for number in numbers))
        pair_rects = VGroup(*(
            SurroundingRectangle(pair, buff=SMALL_BUFF).set_stroke(YELLOW, 1)
            for pair in pair_labels
        ))
        rhs = Tex("= 0")
        rhs_num = rhs.make_number_changable("0")
        rhs.next_to(nc2, RIGHT)

        nc2[1].set_opacity(0)
        self.play(
            Write(number_rects, lag_ratio=0.5),
            ChangeDecimalToValue(n_value, self.n),
        )
        self.play(FadeOut(number_rects, lag_ratio=0.1))
        self.wait()
        self.play(
            VFadeIn(rhs),
            nc2_label.animate.next_to(nc2, DOWN),
            ChangeDecimalToValue(rhs_num, choose(self.n, 2), run_time=3),
            ShowIncreasingSubsets(pair_rects, run_time=3),
        )
        self.wait()
        self.play(FadeOut(pair_rects, lag_ratio=0.04))

        # Show how to calculate it
        new_rhs = Tex(R"= {7 \cdot (7 - 1) \over 2}")
        new_rhs.next_to(nc2, RIGHT)

        self.play(
            rhs.animate.next_to(new_rhs, RIGHT),
            Write(new_rhs[:2]),
            Write(new_rhs[R"\over"]),
        )
        self.wait()
        self.play(Write(new_rhs[R"\cdot (7 - 1)"]))
        self.wait()
        self.play(Write(new_rhs[R"2"]))
        self.wait()

    def get_circle_diagram(self):
        circle = Circle()
        circle.set_stroke(Color("red"), width=2)
        circle.set_height(6)
        circle.to_edge(LEFT)
        points = [
            circle.pfp(a + self.fudge_factor * np.random.uniform(-0., 0.5))
            for a in np.arange(0, 1, 1 / self.n)
        ]
        dots = VGroup(*(
            Dot(point, radius=0.04).set_fill(WHITE)
            for point in points
        ))

        chords = VGroup(*(
            Line(p1, p2).set_stroke(BLUE_B, 1)
            for p1, p2 in it.combinations(points, 2)
        ))

        numbers = VGroup()
        for n, point in zip(it.count(1), points):
            number = Integer(n, font_size=36)
            vect = normalize(point - circle.get_center())
            number.next_to(point, vect, buff=MED_SMALL_BUFF)
            numbers.add(number)

        return VGroup(circle, chords, dots, numbers)


class SimpleCircle(InteractiveScene):
    def construct(self):
        # Test
        circle = Circle()
        circle.set_height(7.75)
        circle.set_stroke(RED, width=3)
        circle.move_to(2.6 * RIGHT)
        radians = np.arange(1, 7)
        dots = VGroup(*(
            Dot(circle.pfp(radians[i] / TAU))
            for i in [0, 2, 4, 1, 3, 5]
        ))

        self.add(circle)
        for dot in dots:
            self.wait()
            self.add(dot)
        self.wait()


class LeftDiagram(ExplainNChoose2):
    n = 6
    fudge_factor = 0.5

    def construct(self):
        # Initialize
        circle = Circle()
        circle.set_height(7.75)
        circle.set_stroke(RED, width=3)
        circle.move_to(2.6 * RIGHT)
        radians = np.arange(1, 7)
        dots = VGroup(*(
            Dot(circle.pfp(radians[i] / TAU))
            for i in [0, 2, 4, 1, 3, 5]
        ))

        # First four diagrams
        diagrams = VGroup()
        for n in range(2, 6):
            diagram = VGroup(
                circle.copy(),
                self.get_chords(dots[:n]),
                dots[:n].copy(),
            )
            diagram.set_stroke(background=True)
            diagram.preimage = diagram.copy()
            diagrams.add(diagram)

        diagrams.arrange(DOWN, buff=2.5)
        diagrams.set_height(FRAME_HEIGHT - 1)
        diagrams.to_edge(LEFT)

        numbers = VGroup(*(
            Integer(2**n, font_size=60).next_to(diagram, RIGHT)
            for n, diagram in zip(it.count(1), diagrams)
        ))

        for diagram, number in zip(diagrams, numbers):
            for dot in diagram[2]:
                dot.scale(2)
            self.play(
                TransformFromCopy(diagram.preimage, diagram),
                FadeInFromPoint(number, circle.get_center()),
            )
            self.wait()

    def get_chords(self, dots):
        return VGroup(*(
            Line(
                d1.get_center(), d2.get_center(),
                stroke_width=2,
                stroke_color=BLUE_B,
            )
            for d1, d2 in it.combinations(dots, 2)
        ))

    def get_samples(self, circle, n=1000, color=YELLOW):
        radius = circle.get_radius()
        center = circle.get_center()
        samples = DotCloud()
        samples.to_grid(n, n)
        samples.replace(circle)
        samples.set_radius(8 / n)
        samples.filter_out(lambda p: np.linalg.norm(p - center) > radius)
        samples.set_color(color)
        return samples

    def get_regions(self, samples, chords):
        regions = Group()
        for signs in it.product(*len(chords) * [[-1, 1]]):
            sub_samples = samples.copy()
            for chord, sign in zip(chords, signs):
                vect = sign * rotate_vector(chord.get_vector(), 90 * DEGREES)
                sub_samples.filter_out(lambda p: np.dot(p, vect) > 0)
            if sub_samples.get_num_points() > 0:
                regions.add(sub_samples)
        return regions


class ProblemSolvingRule1(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students

        # Title
        title = Text(
            "Problem-solving rule #1",
            font_size=72
        )
        title.to_edge(UP)
        title.set_backstroke()
        underline = Underline(title, buff=-0.05)
        self.play(
            morty.change("raise_right_hand"),
            self.change_students("pondering", "thinking", "pondering", look_at=title),
            GrowFromPoint(title, morty.get_corner(UL))
        )
        self.add(title)
        self.add(underline, title)
        self.play(ShowCreation(underline))
        self.wait()

        # Words
        words = Text("""
            Try answering easier questions
            related to the problem at hand
        """, font_size=48)
        words.next_to(underline, DOWN, LARGE_BUFF)
        words.set_fill(YELLOW)

        self.play(
            FadeIn(words, lag_ratio=0.1, run_time=2),
            self.change_students("tease", "thinking", "well", look_at=words),
            morty.change("tease"),
        )
        self.wait(6)


class PairOfChords(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        self.play(
            self.students[0].change("confused", self.screen),
            self.students[1].says("Pairs of chords?", "raise_left_hand"),
            self.students[2].change("confused", self.screen),
            self.teacher.change("well")
        )
        self.wait(5)


class PauseAndPonder(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        self.play(LaggedStart(
            self.students[0].change("pondering", self.screen),
            self.students[1].change("thinking", self.screen),
            self.students[2].change("erm", self.screen),
            self.teacher.says("Pause and\nponder", mode="hooray")
        ))
        self.wait(5)


class CountIntersections(ExplainNChoose2):
    tuple_font_size = 28

    def construct(self):
        # Setup circle
        diagram = self.get_circle_diagram()
        self.add(*diagram)

        # Quad words
        quad_words = Text("Quadruplets of points")
        quad_words.to_edge(RIGHT, MED_LARGE_BUFF)
        quad_words.to_edge(UP, MED_SMALL_BUFF)
        quad_words.add(Underline(quad_words))
        self.add(quad_words)

        # Show all quadruplets
        int_dots, quad_labels = self.show_quadruplets(
            diagram, quad_words
        )

        # Show n choose 4
        nc4 = Tex(R"n \choose 4")
        nc4.next_to(quad_words, DOWN, MED_LARGE_BUFF)
        nc4.shift(2.0 * LEFT)
        nc4.set_color(YELLOW)
        nc4_label = TexText("``n choose 4''")
        nc4_label.next_to(nc4, DOWN)

        self.play(
            FadeIn(nc4, 0.5 * DOWN),
            Write(nc4_label),
            quad_labels.animate.set_height(4).to_edge(DOWN),
        )
        self.wait()

        # Show the count
        rhs = Tex("= 0")
        rhs_num = rhs.make_number_changable("0")
        rhs.next_to(nc4, RIGHT)

        quad_rects = VGroup(*(
            SurroundingRectangle(label, buff=SMALL_BUFF).set_stroke(YELLOW, 1)
            for label in quad_labels
        ))

        self.play(
            VFadeIn(rhs, time_span=(0, 1)),
            ShowIncreasingSubsets(quad_rects),
            ChangeDecimalToValue(rhs_num, choose(self.n, 4)),
            run_time=3
        )
        self.wait()
        self.play(
            FadeOut(quad_rects, lag_ratio=0.02),
            rhs.animate.set_opacity(0),
        )
        self.wait()

        # Bigger rhs
        full_rhs = Tex(
            R"= {n(n-1)(n-2)(n-3) \over 1 \cdot 2 \cdot 3 \cdot 4}",
        )
        nc4.generate_target()
        nc4.target.shift(1.25 * LEFT)
        full_rhs.next_to(nc4.target, RIGHT, SMALL_BUFF)

        self.play(
            FadeOut(diagram),
            FadeOut(int_dots),
            Write(full_rhs),
            MoveToTarget(nc4),
            FadeOut(rhs, 2 * RIGHT),
            MaintainPositionRelativeTo(nc4_label, nc4),
        )
        self.wait()

    def show_quadruplets(self, diagram, quad_words):
        circle, chords, dots, numbers = diagram
        indices = list(range(self.n))
        quad_labels = VGroup()
        quad_label_template = Tex(
            "(0, 0, 0, 0)",
            font_size=self.tuple_font_size
        )
        last_label = VectorizedPoint()
        last_label.next_to(quad_words, DOWN)
        last_label.shift(1.75 * LEFT)

        int_dots = VGroup()

        for sub_indices in it.combinations(indices, 4):
            label = quad_label_template.copy()
            values = label.make_number_changable("0", replace_all=True)
            for value, i in zip(values, sub_indices):
                value.set_value(i + 1)
            label.next_to(last_label, DOWN)
            if label.get_y() < -3.5:
                label.next_to(quad_labels, RIGHT, MED_LARGE_BUFF)
                label.align_to(quad_labels, UP)
            quad_labels.add(label)

            chords.set_opacity(0.25)
            dots.set_opacity(0.25)
            numbers.set_opacity(0.25)
            i, j, k, l = sub_indices
            temp_lines = VGroup(
                Line(dots[i].get_center(), dots[k].get_center()),
                Line(dots[j].get_center(), dots[l].get_center()),
            )
            int_dot = Dot(find_intersection(
                temp_lines[0].get_start(), temp_lines[0].get_vector(),
                temp_lines[1].get_start(), temp_lines[1].get_vector(),
            ), radius=0.04)
            int_dot.set_fill(YELLOW)

            temp_lines.set_stroke(BLUE_B, 2)
            for group in dots, numbers:
                for i in sub_indices:
                    group[i].set_opacity(1)

            self.add(quad_labels)
            self.add(temp_lines)
            self.play(LaggedStart(*(
                TransformFromCopy(dots[i], int_dot)
                for i in sub_indices
            )), lag_ratio=0.1)
            self.add(int_dot)
            self.wait(0.5)
            self.remove(temp_lines)

            int_dots.add(int_dot)
            int_dots.set_opacity(0.25)
            self.add(int_dots)

            last_label = label

        self.play(
            chords.animate.set_opacity(1),
            numbers.animate.set_opacity(1),
            dots.animate.set_opacity(1),
            int_dots.animate.set_opacity(1),
        )
        self.wait()

        return int_dots, quad_labels


class SimpleRect(InteractiveScene):
    def construct(self):
        self.play(FlashAround(
            Tex("1.2.3.4").scale(2),
            time_width=1.5,
            run_time=3,
            stroke_width=8,
        ))


class Clean4choose4(InteractiveScene):
    def construct(self):
        self.add(Tex(R"""
            {4 \choose 4}
            =\frac{4(4-1)(4-2)(4-3)}{1 \cdot 2 \cdot 3 \cdot 4}
            =1 \text { quadruplet }
        """).to_edge(UP))


class Clean6choose4(InteractiveScene):
    def construct(self):
        self.add(Tex(R"""
            {6 \choose 4}
            =\frac{6(6-1)(6-2)(6-3)}{1 \cdot 2 \cdot 3 \cdot 4}
            =15 \text { quadruplets }
        """).to_edge(UP))


class Clean100choose4(InteractiveScene):
    def construct(self):
        self.add(Tex(R"""
            {100 \choose 4}
            =\frac{100(100-1)(100-2)(100-3)}{1 \cdot 2 \cdot 3 \cdot 4}
            =3{,}921{,}225 \text { quadruplets }
        """).to_edge(UP))


class FillIn100PointDiagram(ExplainNChoose2):
    n = 100

    def construct(self):
        # Show all points
        circle, chords, dots, numbers = diagram = self.get_circle_diagram()
        diagram.center()
        chords.set_stroke(width=1, opacity=0.1)
        for dot in dots:
            dot.scale(0.75)

        self.add(circle, chords, dots)
        self.play(
            ShowCreation(chords),
            run_time=9,
        )


class PlanarNonPlanar(InteractiveScene):
    def construct(self):
        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        v_line.set_stroke(GREY_C, 2)
        self.add(v_line)

        titles = VGroup(
            Text("Planar graph"),
            Text("Non-planar graph"),
        )
        for vect, title in zip([LEFT, RIGHT], titles):
            title.shift(vect * FRAME_WIDTH / 4)
        titles.to_edge(UP)
        self.add(titles)


class NonPlanarGraph(InteractiveScene):
    def construct(self):
        # Test
        circle = Circle(radius=3)
        dots = Dot().get_grid(2, 3, h_buff=1.5, v_buff=3.0)
        lines = VGroup(*(
            Line(d1.get_center(), d2.get_center())
            for d1, d2 in it.product(dots[:3], dots[3:])
        ))
        lines.set_stroke(BLUE_B, 2)

        self.add(lines, dots)
        self.play(ShowCreation(lines, lag_ratio=0.1, run_time=2))
        self.wait()


class SimpleR(InteractiveScene):
    def construct(self):
        self.add(Tex("R").scale(2))


class SimpleVLine(InteractiveScene):
    def construct(self):
        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        v_line.set_stroke(GREY_C, 2)
        self.add(v_line)


class Polyhedra(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        frame.reorient(30, 70)
        cube = VCube()
        cube.set_stroke(WHITE, 2, 0.5)
        cube.set_flat_stroke(False)
        cube.set_fill(BLUE_D, 0.8)
        cube.set_shading(1, 0.5, 0)
        cube.apply_depth_test()
        cube.move_to(2.0 * OUT)

        dodec = Dodecahedron()
        dodec.match_style(cube)
        dodec.move_to(2.0 * IN)
        dodec.set_fill(TEAL_E)

        camera_point = frame.get_implied_camera_location()
        cube.add_updater(lambda m: m.sort(lambda p: -get_norm(p - camera_point)))
        dodec.add_updater(lambda m: m.sort(lambda p: -get_norm(p - camera_point)))

        self.add(cube, dodec)
        self.play(LaggedStart(
            Rotating(
                cube,
                axis=np.array([1, 1, 1]),
                angle=PI,
                rate_func=smooth,
                about_point=cube.get_center()
            ),
            Rotating(
                dodec,
                axis=np.array([1, -1, 0]),
                angle=PI,
                rate_func=smooth,
                about_point=dodec.get_center()
            ),
            run_time=16,
            lag_ratio=0.2
        ))


class BalancedEquation(InteractiveScene):
    def construct(self):
        # Equation
        full_rect = FullScreenRectangle()
        equation = Tex("V - E + F = 2", font_size=60)
        equation.to_edge(UP)
        variables = VGroup(*(
            equation[char][0]
            for char in "VEF"
        ))
        numbers = VGroup(*map(Integer, [1, 0, 1]))
        for number, variable in zip(numbers, variables):
            number.scale(1.25)
            number.next_to(variable, DOWN, MED_SMALL_BUFF)

        self.add(full_rect)
        self.add(equation)

        # Initialize Graph
        original_points = [ORIGIN, RIGHT, UL, DL, UR, 2 * LEFT]
        edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (2, 5),
            (2, 4),
            (5, 0),
            (3, 5),
            (1, 3),
        ]
        faces = [
            (0, 1, 4, 2),
            (0, 2, 5),
            (0, 3, 5),
            (0, 1, 3),
        ]
        dots = VGroup(*(Dot(point) for point in original_points))
        dots.space_out_submobjects(1.5)
        dots.move_to(DOWN)
        points = [dot.get_center() for dot in dots]

        edges = VGroup(*(
            Line(points[i], points[j])
            for i, j in edges
        ))
        edges.set_stroke(BLUE_B, 2)

        colors = [BLUE_E, BLUE_C, TEAL_C, GREY_BROWN]
        faces = VGroup(*(
            Polygon(*[points[i] for i in face]).set_fill(color, 1)
            for face, color in zip(faces, colors)
        ))
        faces.set_stroke(width=0)

        # Trivial graph
        arrow = Vector(DL)
        arrow.next_to(dots[0], UR, SMALL_BUFF)
        word = Text("Trivial graph")
        word.next_to(arrow.get_start(), UP)

        self.add(dots[0])
        self.play(
            Write(word, run_time=1),
            GrowArrow(arrow),
        )
        self.wait()

        self.play(
            FlashAround(equation["V"]),
            FlashAround(dots[0]),
            FadeIn(numbers[0], 0.5 * DOWN),
        )
        self.wait(0.5)
        self.play(
            FlashAround(equation["F"]),
            full_rect.animate.set_fill(BLUE_E, 0.5).set_anim_args(rate_func=there_and_back),
            FadeIn(numbers[2], 0.5 * DOWN)
        )
        self.wait()
        self.play(
            FlashAround(equation["E"]),
            FadeIn(numbers[1], 0.5 * DOWN)
        )
        self.wait()
        self.play(FadeOut(word), FadeOut(arrow))

        # Edges with new vertices
        number_y = numbers.get_y()
        increments = VGroup(
            Integer(1, include_sign=True),
            Integer(-1, include_sign=True),
            Integer(1, include_sign=True),
        )
        for number, incr in zip(numbers, increments):
            incr.next_to(number, DOWN)
            incr.set_color(YELLOW)

        def incr_anim(*ns):
            return LaggedStart(*(
                AnimationGroup(
                    UpdateFromAlphaFunc(
                        increments[n], lambda m, a: m.set_y(number_y - 0.75 * a).set_opacity(there_and_back(a))
                    ),
                    ChangeDecimalToValue(
                        numbers[n], numbers[n].get_value() + 1,
                        run_time=0.25
                    )
                )
                for n in ns
            ), lag_ratio=0.1)

        new_vert_word = TexText(R"New edge $\rightarrow$ New vertex")
        new_vert_word["New vertex"].set_color(YELLOW)
        new_vert_word.next_to(dots[1], UP)
        new_vert_word.set_backstroke(width=2)

        self.play(
            ShowCreation(edges[0]),
            FadeIn(new_vert_word[R"New edge $\rightarrow$"][0])
        )
        self.wait()
        self.play(
            FadeIn(dots[1], scale=0.5),
            FadeIn(new_vert_word["New vertex"][0], lag_ratio=0.1)
        )
        self.play(incr_anim(0, 1))
        for i in [2, 3, 5, 4]:
            self.add(edges[i - 1], new_vert_word)
            self.play(
                ShowCreation(edges[i - 1]),
                new_vert_word.animate.next_to(dots[i], UP),
                FadeIn(dots[i])
            )
            self.play(incr_anim(0, 1))

        # Edges with new faces
        new_face_word = TexText(R"New edge $\rightarrow$ New face")
        new_face_word["New face"].set_color(BLUE)
        new_face_word.next_to(faces[0], UP)

        self.play(
            FadeTransform(new_vert_word, new_face_word),
            ShowCreation(edges[5]),
        )
        self.play(Write(faces[0]))
        self.play(incr_anim(1, 2))
        self.wait()
        for i, vect in zip([1, 2, 3], [UP, DOWN, DOWN]):
            self.add(edges[i + 5], dots)
            self.play(
                ShowCreation(edges[i + 5]),
                new_face_word.animate.next_to(faces[i], vect),
            )
            self.add(faces[i], edges[:i + 5], dots, new_face_word)
            self.play(Write(faces[i]))
            self.play(incr_anim(1, 2))
        self.add(*faces, *edges, *dots)
        self.play(
            FadeOut(new_face_word),
            FadeOut(numbers),
        )

        # Name formula
        rect = SurroundingRectangle(equation)
        rect.set_stroke(YELLOW, 2)
        name = TexText("Euler's Characteristic Formula", font_size=60)
        name.next_to(rect, DOWN, buff=MED_LARGE_BUFF)

        self.play(
            ShowCreation(rect),
            FlashAround(equation),
            Write(name, run_time=2),
        )
        self.wait()
        self.play(FadeOut(rect), FadeOut(name))

        # Rearrange
        new_equation = Tex("F = E - V + 2")
        new_equation.match_height(equation)
        new_equation.move_to(equation)

        face_labels = index_labels(faces, label_height=0.2)
        face_labels.set_backstroke(BLACK, 3)
        face_labels[3].shift(0.3 * UP)
        outer_label = Integer(len(faces) + 1)
        outer_label.match_height(face_labels[0])
        outer_label.match_style(face_labels[0])
        outer_label.next_to(faces, RIGHT, MED_LARGE_BUFF)

        self.remove(faces)
        self.play(
            ShowIncreasingSubsets(faces),
            ShowIncreasingSubsets(face_labels),
            run_time=2,
        )
        self.add(edges, *dots)
        self.wait(0.2)
        self.add(outer_label)
        full_rect.set_fill(BLUE_E, 0.5)
        self.wait(0.5)
        full_rect.set_fill(GREY_E, 1)
        self.wait()
        self.play(TransformMatchingTex(
            equation, new_equation, path_arc=90 * DEGREES
        ))
        self.wait()


class OnDumbJoke(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        morty = self.teacher
        stds = self.students

        self.play(
            morty.change("raise_left_hand", 3 * UR),
            self.change_students("hesitant", "happy", "tease", look_at=3 * UR)
        )
        self.wait(3)
        self.play(
            stds[0].says("Um, no, Euler was\na person...", mode="sassy"),
            stds[1].change("hesitant", stds[0].eyes),
            stds[2].change("jamming", stds[0].eyes),
            morty.change("well", stds[0].eyes)
        )
        self.wait(4)


class SimpleFEq(InteractiveScene):
    def construct(self):
        F_eq = Tex("F = E - V + 2")
        F_eq.to_edge(UP)
        one = Tex("1")
        one.move_to(F_eq["2"], DOWN)

        self.add(F_eq)
        self.wait()
        self.play(
            FadeOut(F_eq["2"][0], 0.5 * UP),
            FadeIn(one, 0.5 * UP),
        )
        self.wait()


class NonPlanarComplaint(TeacherStudentsScene):
    def construct(self):
        stds = self.students
        morty = self.teacher
        self.remove(self.background)

        self.play(
            stds[2].says("But our graph\nis not planar!", mode="angry"),
            stds[1].change("sassy", 3 * UR),
            stds[0].change("hesitant", 3 * UR),
            morty.change("tease"),
        )
        self.wait(3)
        self.play(
            morty.change("raise_left_hand"),
            self.change_students("angry", "sassy", "hesitant")
        )
        self.wait(4)


class VEquation(InteractiveScene):
    def construct(self):
        eq = Tex(R"V = n + {n \choose 4}")
        eq.to_corner(UL)
        self.play(Write(eq))
        self.wait()


class EdgeAdditionFactor(InteractiveScene):
    def construct(self):
        eq = Tex(R"""
            E = (\#\text{Original lines}) + 2(\#\text{Intersection points})
        """)
        eq["Original lines"].set_color(BLUE_B)
        eq["Intersection points"].set_color(YELLOW)
        eq.to_edge(UP)
        self.add(eq)


class EEquation(InteractiveScene):
    def construct(self):
        eq = Tex(R"E = {n \choose 2} + 2 {n \choose 4} + n", font_size=72)
        eq.to_corner(UR)
        self.play(FadeIn(eq[:6], UP))
        self.wait()
        self.play(Write(eq[6:-2]))
        self.wait()
        self.play(Write(eq[-2:]))
        self.wait()


class AddArcComment(InteractiveScene):
    def construct(self):
        comment = TexText("$+n$ circular arcs")
        comment.to_corner(UL)
        comment.set_color(YELLOW)
        self.add(comment)


class FinalRearrangment(InteractiveScene):
    def construct(self):
        # Add equations
        top_eq = Tex("F = E - V + 1")
        top_eq.to_edge(UP, buff=LARGE_BUFF)

        V_rhs = R"n + {n \choose 4}"
        E_rhs = R"{n \choose 2} + 2{n \choose 4} + n"
        V_eq = Tex(Rf"V = {V_rhs}")
        E_eq = Tex(Rf"E = {E_rhs}")
        V_eq.next_to(top_eq, DOWN, buff=1.5)
        E_eq.next_to(top_eq, DOWN, buff=1.5)

        V_eq.set_color(TEAL)
        E_eq.set_color(BLUE_B)
        t2c = {
            Rf"\left({V_rhs}\right)": TEAL,
            Rf"\left({E_rhs}\right)": BLUE_B,
        }

        self.play(FadeIn(top_eq, UP))
        self.wait()

        # Substitute V
        top_eq2 = Tex(Rf"F = E - \left({V_rhs}\right) + 1", t2c=t2c)
        top_eq2.move_to(top_eq)

        self.play(
            top_eq["V"][0].animate.set_color(TEAL),
            FadeTransform(top_eq["V"][0].copy(), V_eq[0]),
            Write(V_eq[1:], run_time=1)
        )
        self.wait()
        self.play(
            FadeTransform(V_eq[2:].copy(), top_eq2[Rf"\left({V_rhs}\right)"]),
            FadeOut(top_eq["V"][0], UP),
            ReplacementTransform(
                top_eq["F = E - "],
                top_eq2["F = E - "],
            ),
            ReplacementTransform(
                top_eq[-2:],
                top_eq2[-2:],
            ),
        )
        self.wait()

        # Substitute E
        E_eq.set_color(BLUE_B)
        top_eq3 = Tex(
            Rf"F = \left({E_rhs}\right) - \left({V_rhs}\right) + 1",
            t2c=t2c
        )
        top_eq3.move_to(top_eq)

        self.play(
            top_eq2[2].animate.set_color(BLUE_B),
            # top_eq2[3:].animate.set_color(WHITE),
            FadeTransform(top_eq2[2].copy(), E_eq[0]),
            Write(E_eq[1:], run_time=1),
            V_eq.animate.shift(2.0 * DOWN),
        )
        self.wait()
        self.play(
            FadeTransform(E_eq[2:].copy(), top_eq3[Rf"\left({E_rhs}\right)"]),
            FadeOut(top_eq2[2], UP),
            ReplacementTransform(top_eq2[:2], top_eq3[:2]),
            ReplacementTransform(top_eq2[-11:], top_eq3[-11:]),
        )

        # Show cancellation
        final_eq = Tex(R"F = 1 + {n \choose 2} + {n \choose 4}")
        final_eq.move_to(top_eq)

        self.play(LaggedStart(
            FadeOut(E_eq, DOWN),
            FadeOut(V_eq, DOWN),
        ))
        self.play(
            # Ns
            FlashAround(top_eq3[14], color=RED),
            FlashAround(top_eq3[18], color=RED),
        )
        self.play(
            FadeOut(top_eq3[13:15], DOWN),
            FadeOut(top_eq3[18:20], DOWN),
        )
        self.play(
            # Ns
            FlashAround(top_eq3[9:13], color=RED),
            FlashAround(top_eq3[20:24], color=RED),
        )
        self.play(
            # N choose 4s
            FadeOut(top_eq3[8], DOWN),
            FadeOut(top_eq3[20:24], DOWN),
        )
        kw = dict(path_arc=90 * DEGREES)
        self.play(LaggedStart(
            Transform(top_eq3[:2], final_eq[:2], **kw),
            Transform(top_eq3[26], final_eq[2], **kw),
            Transform(top_eq3[25], final_eq[3], **kw),
            Transform(top_eq3[3:7], final_eq[4:8], **kw),
            Transform(top_eq3[7], final_eq[8], **kw),
            Transform(top_eq3[9:13], final_eq[9:13], **kw),
            *(FadeOut(top_eq3[i], DOWN) for i in [2, 15, 16, 17, 24]),
            run_time=2,
        ))
        self.wait()


class ExamplesOfFormula(InteractiveScene):
    def construct(self):
        # Test
        examples = VGroup(*(
            self.get_example(n)
            for n in range(1, 7)
        ))
        examples.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        examples.set_height(7)
        examples.to_edge(RIGHT)

        self.add(*(ex[0] for ex in examples))
        for ex in examples:
            self.play(FadeTransform(ex[0].copy(), ex[1]))
            self.wait(0.5)
        self.wait()

    def get_example(self, n):
        result = VGroup(
            Tex(fR"1 + {{{n} \choose 2 }} + {{{n} \choose 4 }}"),
            Tex(fR"= 1 + {choose(n, 2)} + {choose(n, 4)} = {1 + choose(n, 2) + choose(n, 4)}")
        )
        result.arrange(RIGHT)
        return result


class WhyThePowersOfTwo(TeacherStudentsScene):
    def construct(self):
        stds = self.students
        morty = self.teacher
        self.remove(self.background)

        self.play(LaggedStart(
            stds[0].change("confused", self.screen),
            stds[1].says("Why the powers \n of 2?"),
            stds[2].change("pondering", self.screen),
            morty.change("tease")
        ))
        self.wait(6)


class IllustrateNChooseK(InteractiveScene):
    n = 7
    k = 3
    show_count = False

    def construct(self):
        # Test2
        n = 7
        k = 3

        dots = Dot(radius=0.1).replicate(self.n)
        dots.arrange(RIGHT, buff=0.5)
        self.add(dots)

        rects = SurroundingRectangle(dots[0], buff=SMALL_BUFF).replicate(self.k)
        rects.set_stroke(BLUE)
        self.add(rects)

        if self.show_count:
            count = Integer(0, font_size=60)
            count.next_to(dots, DOWN, buff=MED_LARGE_BUFF)
            self.add(count)

        for subdots in it.combinations(dots, self.k):
            dots.set_color(WHITE)
            for dot, rect in zip(subdots, rects):
                rect.move_to(dot)
                dot.set_fill(BLUE)
            if self.show_count:
                count.set_value(count.get_value() + 1)
            self.wait(0.2)


class Illustrate5Choose3(InteractiveScene):
    n = 5
    k = 3

    def construct(self):
        # Test
        row = Dot().replicate(self.n)
        row.arrange(RIGHT)
        rows = row.replicate(choose(self.n, self.k))
        rows.arrange(DOWN)

        indices = list(range(self.n))
        for tup, row in zip(it.combinations(indices, self.k), rows):
            for i in tup:
                row[i].set_color(BLUE)
                rect = SurroundingRectangle(row[i], buff=SMALL_BUFF)
                rect.set_stroke(BLUE, 2)
                row[i].add(rect)

        brace = Brace(rows, RIGHT, buff=SMALL_BUFF)
        count = Integer(1, font_size=60)
        count.next_to(brace, RIGHT, SMALL_BUFF)
        count.add_updater(lambda m: m.set_value(len(rows)))

        self.add(count)
        self.play(
            ShowIncreasingSubsets(rows),
            run_time=2,
        )
        self.wait()


class OnScreenExplanationOfNChooseK(InteractiveScene):
    def construct(self):
        # Test
        kw = dict(
            tex_to_color_map={"{n}": BLUE, "{k}": TEAL},
            alignment=""
        )
        explanation = VGroup(
            TexText(R"""
            The claim is that if you look at the ${k}^\text{th}$ element of the
            ${n}^\text{th}$ row in Pascal's triangle, it equals ${n} \choose {k}$. To show this,
            it's enough to show that
                $$ {{n} \choose {k}} = {{n} - 1 \choose {k} - 1} + {{n} - 1 \choose {k}} $$
            Can you see why?
            """, **kw),
            TexText(R"""
                There's a nice counting argument which proves this equation. Imagine
                you have ${n}$ items, and you want to choose ${k}$ of them. On the one hand,
                by definition, there are ${n} \choose {k}$ ways to do this. But on the other
                hand, imagine marking one of the items as special. How many ways can you
                select ${k}$ unmarked items? How may ways can you select ${k}$ items where
                one of them is the marked one? What does this tell you?
            """, **kw)
        )
        explanation.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        explanation.set_width(FRAME_WIDTH - 1)
        self.add(explanation)


class ChallengeProblem(InteractiveScene):
    def construct(self):
        numbers = VGroup(*(
            Integer(1 + choose(n, 2) + choose(n, 4))
            for n in range(100)
        ))
        numbers.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=RIGHT)
        numbers.to_corner(UL)
        numbers.shift(DR)
        numbers.add_updater(lambda m, dt: m.shift(2 * dt * UP))

        question = VGroup(
            Text("Challenge question", font_size=72).set_fill(YELLOW),
            Text("Will there ever be another power of 2?")
        )
        question.arrange(DOWN, buff=MED_LARGE_BUFF)
        question.next_to(numbers, RIGHT, LARGE_BUFF)
        question.set_y(0)

        # Test
        self.add(numbers)
        self.play(Write(question))
        self.wait(20)


class AskAboutSome(InteractiveScene):
    def construct(self):
        randy = Randolph()
        morty = Mortimer()
        randy.next_to(ORIGIN, LEFT, LARGE_BUFF).to_edge(DOWN)
        morty.next_to(ORIGIN, RIGHT, LARGE_BUFF).to_edge(DOWN)

        morty.to_corner(DR)
        randy.next_to(morty, LEFT, LARGE_BUFF)
        self.add(morty)

        # SoME
        words = Text("The 3rd Summer of\nMath Exposition", font_size=72)
        words.next_to(randy, UP)
        words.to_edge(RIGHT)

        small_words = Text("SoME3", font_size=72)
        small_words.move_to(words, UP)
        small_words_copy = small_words.copy()
        small_words_copy.next_to(morty.get_corner(UL), UP)
        three_some = Text("3SoME", font_size=72)
        three_some.next_to(randy.get_corner(UR), UP)

        self.play(
            morty.change("raise_left_hand", words),
            FadeIn(words, 0.25 * UP, lag_ratio=0.01)
        )
        self.play(Blink(morty))
        self.wait(3)

        self.play(
            TransformMatchingStrings(words, small_words, path_arc=45 * DEGREES),
            morty.change("raise_right_hand")
        )
        self.play(Blink(morty))
        self.wait(2)
        self.play(
            VFadeIn(randy),
            randy.change("shruggie", three_some),
            morty.change("sassy", randy.eyes),
            TransformMatchingStrings(small_words, three_some, path_arc=45 * DEGREES)
        )
        self.play(morty.change("angry", randy.eyes))
        self.play(Blink(morty))
        self.play(
            TransformMatchingStrings(three_some, small_words_copy, run_time=1),
            morty.change("raise_right_hand"),
            randy.change("sassy")
        )
        self.play(Blink(morty))
        self.play(Blink(randy))
        self.wait(2)


class EndScreen(PatreonEndScreen):
    pass
