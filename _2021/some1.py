from manim_imports_ext import *


def get_tripple_underline(mobject, buff=0.1):
    ul1 = Underline(mobject, buff=buff).set_stroke(BLUE_C, 3)
    ul2 = Underline(ul1).scale(0.9).set_stroke(BLUE_D, 2)
    ul3 = Underline(ul2).scale(0.9).set_stroke(BLUE_E, 1)
    return VGroup(ul1, ul2, ul3)


def get_h_line():
    line = DashedLine(ORIGIN, FRAME_WIDTH * RIGHT)
    line.center()
    return line


# Scenes

class TableOfContents(Scene):
    def construct(self):
        # plane = NumberPlane()
        # self.add(plane, FullScreenFadeRectangle(opacity=0.8))

        items = VGroup(
            Text("Summer of Math Exposition"),
            Text("The Universal Advice"),
            Text("How to structure math explanations"),
            Text("Thoughts on animation software"),
            Text("The 3b1b Podcast/Second channel"),
        )
        items.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        items.to_edge(LEFT, buff=0.5)

        self.add(items)

        for i in range(len(items)):
            for item in items:
                item.generate_target()
                if item is items[i]:
                    height = 0.55
                    opacity = 1.0
                else:
                    height = 0.3
                    opacity = 0.25
                item.target.scale(
                    height / item[0].get_height(),
                    about_edge=LEFT
                )
                item.target.set_opacity(opacity)
            self.play(*map(MoveToTarget, items))
            self.wait()


class SoME1Name(Scene):
    def construct(self):
        strings = ["Summer", "of", "Math", "Exposition", "#1"]
        lens = list(map(lambda s: len(s) + 1, strings))
        indices = [0, *np.cumsum(lens)]
        phrase = Text(" ".join(strings))
        phrase.set_width(12)
        words = VGroup()
        for i1, i2 in zip(indices, indices[1:]):
            words.add(phrase[i1:i2])

        acronym = VGroup(*(word[0].copy() for word in words[:-1]))
        acronym.add(words[-1][-1].copy())
        acronym.generate_target()
        acronym.target.arrange(RIGHT, buff=SMALL_BUFF, aligned_edge=DOWN)
        acronym.target.move_to(DOWN)

        self.play(LaggedStartMap(FadeIn, words, lag_ratio=0.5, run_time=1))
        self.wait()
        self.play(
            # FadeOut(phrase),
            phrase.animate.move_to(UP),
            MoveToTarget(acronym)
        )
        self.wait()

        self.play(
            ShowCreation(get_tripple_underline(words[2])),
            words[2].animate.set_color(BLUE_B),
        )
        self.wait()


class LinkAndDate(Scene):
    def construct(self):
        words = link, date = VGroup(
            Text("https://3b1b.co/SoME1", font="CMU Serif"),
            Text("August 22nd", font="CMU Serif")
        )
        words.set_width(10)
        words.arrange(DOWN, buff=0.5)
        words.set_stroke(BLACK, 5, background=True)

        lines = get_tripple_underline(date)

        details = Text("(Full details here)")
        details.to_corner(UR)
        arrow = Arrow(details, link, buff=0.5)

        self.play(Write(link), run_time=1)
        self.wait()
        self.add(lines, words)
        self.play(
            FadeIn(date, scale=1.2, rate_func=squish_rate_func(smooth, 0.3, 0.7)),
            ShowCreation(lines, lag_ratio=0.25)
        )
        self.wait()

        self.play(FadeIn(details), GrowArrow(arrow))
        self.wait()


class Featuring(TeacherStudentsScene):
    def construct(self):
        screen = ScreenRectangle()
        screen.set_height(4)
        screen.to_corner(UL)
        screen.set_stroke(WHITE, 2)
        screen.set_fill(BLACK, 1)

        self.add(screen)

        words = TexText("Your work here")
        words.set_width(screen.get_width() - 1)
        words.move_to(screen)

        self.play(
            self.teacher.change("tease"),
            self.change_students(*3 * ["hooray"], look_at=screen),
            FadeIn(screen, UP)
        )
        self.play(Write(words))
        self.play(ShowCreation(get_tripple_underline(words)))
        self.wait(5)
        self.student_says(
            TexText("Anything\\\\else?"),
            target_mode="raise_left_hand",
            look_at=self.teacher,
        )
        self.play(self.teacher.change("sassy"))
        self.wait(2)


class Constraints(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            TexText("What are the\\\\constraints?"),
            added_anims=[self.teacher.change("tease")]
        )
        self.wait(3)


class TopicChoice(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            TexText("What kind of\\\\topics?"),
            added_anims=[self.teacher.change("tease")],
            index=1,
        )
        self.wait(3)


class PartialFraction(Scene):
    def construct(self):
        frac = Tex(
            r"\frac{x+9}{(x-3)(x+5)} = \frac{?}{x - 3} + \frac{?}{x + 5}"
        )
        frac.set_width(10)
        self.add(frac)


class TrigIdentity(Scene):
    def construct(self):
        plane = NumberPlane().scale(2)
        circle = Circle(radius=2)
        circle.set_stroke(YELLOW, 2)

        theta_tracker = ValueTracker(60 * DEGREES)
        get_theta = theta_tracker.get_value

        def get_point():
            return circle.pfp(get_theta() / TAU)

        dot = Dot()
        f_always(dot.move_to, get_point)
        radial_line = always_redraw(lambda: Line(plane.c2p(0, 0), get_point()))
        tan_line = always_redraw(
            lambda: Line(get_point(), plane.c2p(1 / math.cos(get_theta()), 0), stroke_color=RED)
        )
        sec_line = always_redraw(
            lambda: Line(plane.c2p(0, 0), tan_line.get_end(), stroke_color=PINK)
        )

        def get_one_label():
            one = Integer(1)
            one.next_to(radial_line.get_center(), UL, SMALL_BUFF)
            return one

        def get_tan_label():
            label = Tex("\\tan(\\theta)")
            label.set_color(RED)
            point = tan_line.get_center()
            label.next_to(point, UP, SMALL_BUFF)
            label.rotate(tan_line.get_angle(), about_point=point)
            label.set_stroke(BLACK, 3, background=True)
            return label

        def get_sec_label():
            label = Tex("\\sec(\\theta)")
            label.set_color(PINK)
            label.next_to(sec_line, DOWN, SMALL_BUFF)
            label.set_stroke(BLACK, 3, background=True)
            return label

        one_label = always_redraw(get_one_label)
        tan_label = always_redraw(get_tan_label)
        sec_label = always_redraw(get_sec_label)

        arc = always_redraw(lambda: Arc(0, get_theta(), radius=0.25))
        arc_label = Tex("\\theta", font_size=36)
        arc_label.next_to(arc, RIGHT, SMALL_BUFF, DOWN).shift(SMALL_BUFF * UP)

        equation = Tex(
            "\\tan^2(\\theta) + 1 = \\sec^2(\\theta)",
            tex_to_color_map={
                "\\tan": RED,
                "\\sec": PINK,
            },
            font_size=60
        )
        equation.to_corner(UL)
        equation.set_stroke(BLACK, 7, background=True)

        self.add(plane)
        self.add(circle)
        self.add(equation)

        self.add(radial_line)
        self.add(tan_line)
        self.add(sec_line)
        self.add(dot)

        self.add(one_label)
        self.add(tan_label)
        self.add(sec_label)

        self.add(arc)
        self.add(arc_label)

        angles = [40 * DEGREES, 70 * DEGREES]
        for angle in angles:
            self.play(theta_tracker.animate.set_value(angle), run_time=4)


class TeacherStudentPairing(Scene):
    def construct(self):
        randy = Randolph(height=2.5)
        morty = Mortimer(height=3.0)
        pis = VGroup(randy, morty)
        pis.arrange(RIGHT, buff=3, aligned_edge=DOWN)
        pis.to_edge(DOWN, buff=1.5)

        self.add(pis)
        self.add(Text("Teacher").next_to(morty, DOWN))
        self.add(Text("Student").next_to(randy, DOWN))

        teacher_label = Text("Delivers/writes lesson", font_size=36)
        student_label = Text("Produces/edits/animates", font_size=36)
        teacher_label.next_to(morty, UP, buff=1.5).shift(RIGHT)
        student_label.next_to(randy, UP, buff=1.5).shift(LEFT)
        teacher_arrow = Arrow(teacher_label, morty)
        student_arrow = Arrow(student_label, randy)

        self.play(
            Write(student_label),
            GrowArrow(student_arrow),
            randy.change("hooray", student_label)
        )
        self.wait()
        self.play(
            Write(teacher_label),
            GrowArrow(teacher_arrow),
            morty.change("pondering", teacher_label)
        )
        self.wait()
        for x in range(4):
            self.play(Blink(random.choice(pis)))
            self.wait(random.random())


class Grey(Scene):
    def construct(self):
        self.add(FullScreenRectangle().set_fill(GREY_D, 1))


class ButIHaveNoExperience(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            TexText("But I have no\\\\experience!"),
            index=0,
            target_mode="pleading",
        )
        self.play_student_changes(
            "pleading", "pondering", "pondering",
            look_at=self.students[0].bubble,
        )
        self.students[0].look_at(self.teacher.eyes)
        self.play(self.teacher.change("tease", self.students[0].eyes))

        self.wait(4)


class ContentAdvice(Scene):
    def construct(self):
        title = Text("Advice for structuring math explanations")
        title.set_width(10)
        title.to_edge(UP)
        underline = Underline(title).scale(1.2).set_stroke(GREY_B, 2)

        self.play(FadeIn(title, UP))
        self.play(ShowCreation(underline))

        points = VGroup(
            Text("1) Concrete before abstract"),
            Text("2) Topic choice > production quality"),
            Text("3) Be niche"),
            Text("4) Know your genre"),
            Text("5) Definitions are not the beginning"),
        )
        points.arrange(DOWN, buff=0.7, aligned_edge=LEFT)
        points.next_to(title, DOWN, buff=1.0)

        self.play(LaggedStartMap(
            FadeIn,
            VGroup(*(point[:2] for point in points)),
            shift=0.25 * RIGHT
        ))
        self.wait()

        for point in points:
            self.play(Write(point[2:]))

        self.wait()
        gt0 = Tex("> 0")
        gt0.next_to(points[1], RIGHT)
        self.play(
            VGroup(points[0], *points[2:]).animate.set_opacity(0.25)
        )
        self.play(Write(gt0))
        self.play(ShowCreation(get_tripple_underline(
            VGroup(points[1].get_part_by_text("production quality"), gt0)
        )))
        self.wait()


class LayersOfAbstraction(Scene):
    def construct(self):
        self.save_count = 0
        self.add_title()
        self.show_layers()
        self.show_pairwise_relations()
        self.circle_certain_pairs()

    def add_title(self):
        title = TexText("Layers of abstraction")
        title.scale(1.5)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        line = Line(LEFT, RIGHT)
        line.set_width(FRAME_WIDTH)
        line.next_to(title, DOWN, SMALL_BUFF)

        self.add(title, line)

    def show_layers(self):
        layers = self.layers = self.get_layers()

        for layer in layers:
            self.play(FadeIn(layer, shift=0.1 * UP))

    def show_pairwise_relations(self):
        p1, p2 = [layer.get_left() for layer in self.layers[2:4]]
        down_arrow = Arrow(p2, p1, path_arc=PI)
        down_words = TexText("``For example''")
        down_words.scale(0.8)
        down_words.next_to(down_arrow, LEFT)
        up_arrow = Arrow(p1, p2, path_arc=-PI)
        up_words = TexText("``In general''")
        up_words.scale(0.8)
        up_words.next_to(up_arrow, LEFT)

        VGroup(up_words, down_words).set_color(YELLOW)

        self.play(
            GrowArrow(down_arrow),
            FadeIn(down_words)
        )
        self.wait()
        self.play(
            ReplacementTransform(down_arrow, up_arrow, path_arc=PI),
            FadeTransform(down_words, up_words)
        )
        self.wait()
        self.play(FadeOut(up_arrow), FadeOut(up_words))

    def circle_certain_pairs(self):
        layers = self.layers

        for l1, l2 in zip(layers, layers[1:]):
            group = VGroup(l1, l2)
            group.save_state()
            layers.save_state()
            layers.fade(0.75)
            rect = SurroundingRectangle(group)
            rect.set_stroke(YELLOW, 5)
            group.restore()
            self.add(rect)
            self.wait()
            self.remove(rect)
            layers.restore()

    #

    def get_layers(self):
        layers = VGroup(*[
            VGroup(Rectangle(height=1, width=5))
            for x in range(6)
        ])
        layers.arrange(UP, buff=0)
        layers.set_stroke(GREY, 2)
        layers.set_gloss(1)

        # Layer 0: Quantities
        triangle = Triangle().set_height(0.25)
        tri_dots = VGroup(*[Dot(v) for v in triangle.get_vertices()])
        dots_rect = VGroup(*[Dot() for x in range(12)])
        dots_rect.arrange_in_grid(3, 4, buff=SMALL_BUFF)
        for i, color in enumerate([RED, GREEN, BLUE]):
            dots_rect[i::4].set_color(color)
        pi_chart = VGroup(*[
            Sector(start_angle=a, angle=TAU / 3)
            for a in np.arange(0, TAU, TAU / 3)
        ])
        pi_chart.set_fill(opacity=0)
        pi_chart.set_stroke(WHITE, 2)
        pi_chart[0].set_fill(BLUE, 1)
        pi_chart.rotate(PI / 3)
        pi_chart.match_height(dots_rect)
        quantities = VGroup(tri_dots, dots_rect, pi_chart)
        quantities.arrange(RIGHT, buff=LARGE_BUFF)

        # Layer 1: Numbers
        numbers = VGroup(
            Tex("3"),
            Tex("3 \\times 4"),
            Tex("1 / 3"),
        )
        for number, quantity in zip(numbers, quantities):
            number.move_to(quantity)

        # Layer 2: Algebra
        algebra = VGroup(
            Tex("x^2 - 1 = (x + 1)(x - 1)")
        )
        algebra.set_width(layers.get_width() - MED_LARGE_BUFF)

        # Layer 3: Functions
        functions = VGroup(
            Tex("f(x) = 0"),
            Tex("\\frac{df}{dx}"),
        )
        functions.set_height(layers[0].get_height() - 2 * SMALL_BUFF)
        functions.arrange(RIGHT, buff=LARGE_BUFF)
        # functions.match_width(algebra)

        # Layer 4: Vector space
        t2c_map = {
            "\\textbf{v}": YELLOW,
            "\\textbf{w}": PINK,
        }
        vector_spaces = VGroup(
            Tex(
                "\\textbf{v} + \\textbf{w} ="
                "\\textbf{w} + \\textbf{v}",
                tex_to_color_map=t2c_map,
            ),
            Tex(
                "s(\\textbf{v} + \\textbf{w}) ="
                "s\\textbf{v} + s\\textbf{w}",
                tex_to_color_map=t2c_map,
            ),
        )
        vector_spaces.arrange(DOWN, buff=MED_SMALL_BUFF)
        vector_spaces.set_height(layers[0].get_height() - MED_LARGE_BUFF)
        v, w = vectors = VGroup(
            Vector([2, 1, 0], color=YELLOW),
            Vector([1, 2, 0], color=PINK),
        )
        vectors.add(DashedLine(v.get_end(), v.get_end() + w.get_vector()))
        vectors.add(DashedLine(w.get_end(), w.get_end() + v.get_vector()))
        vectors.match_height(vector_spaces)
        vectors.next_to(vector_spaces, RIGHT)
        vectors.set_stroke(width=2)
        # vector_spaces.add(vectors)

        inner_product = Tex(
            "\\langle f, g \\rangle ="
            "\\int f(x)g(x)dx"
        )
        inner_product.match_height(vector_spaces)
        inner_product.next_to(vector_spaces, RIGHT)
        vector_spaces.add(inner_product)

        # Layer 5: Categories
        dots = VGroup(Dot(UL), Dot(UR), Dot(RIGHT))
        arrows = VGroup(
            Arrow(dots[0], dots[1], buff=SMALL_BUFF),
            Arrow(dots[1], dots[2], buff=SMALL_BUFF),
            Arrow(dots[0], dots[2], buff=SMALL_BUFF),
        )
        arrows.set_stroke(width=2)
        arrow_labels = VGroup(
            Tex("m_1").next_to(arrows[0], UP, SMALL_BUFF),
            Tex("m_2").next_to(arrows[1], RIGHT, SMALL_BUFF),
            Tex("m_2 \\circ m_1").rotate(-np.arctan(1 / 2)).move_to(
                arrows[2]
            ).shift(MED_SMALL_BUFF * DL)
        )
        categories = VGroup(dots, arrows, arrow_labels)
        categories.set_height(layers[0].get_height() - MED_SMALL_BUFF)

        # Put it all together
        all_content = [
            quantities, numbers, algebra,
            functions, vector_spaces, categories,
        ]

        for layer, content in zip(layers, all_content):
            content.move_to(layer)
            layer.add(content)
            layer.content = content

        layer_titles = VGroup(*map(TexText, [
            "Quantities",
            "Numbers",
            "Algebra",
            "Functions",
            "Vector spaces",
            "Categories",
        ]))
        for layer, title in zip(layers, layer_titles):
            title.next_to(layer, RIGHT)
            layer.add(title)
            layer.title = title
        layers.titles = layer_titles

        layers.center()
        layers.to_edge(DOWN)
        layers.shift(0.5 * RIGHT)
        return layers


class FractionsExample(Scene):
    def construct(self):
        expr = Tex("{2 \\over 3}", "+", "{1 \\over 5}")
        expr.scale(1.5)
        h_line = DashedLine(ORIGIN, FRAME_WIDTH * RIGHT).center()

        quantities = Text("Quantities")
        quantities.to_corner(UL, buff=MED_SMALL_BUFF)
        quantities.shift(FRAME_HEIGHT * DOWN / 2)
        numbers = Text("Numbers")
        numbers.to_corner(UL, buff=MED_SMALL_BUFF)

        def get_pie(numer, denom, color=BLUE, height=1.5):
            pie = VGroup(*(
                AnnularSector(
                    angle=TAU / denom,
                    start_angle=i * TAU / denom,
                    inner_radius=0.0,
                    outer_radius=1.0,
                )
                for i in range(denom)
            ))
            pie.set_stroke(WHITE, 2)
            pie.set_fill(BLACK, 1)
            pie[:numer].set_fill(color)
            pie.set_height(height)
            return pie

        pies = VGroup(
            get_pie(2, 3, color=BLUE_C),
            Tex("+"),
            get_pie(1, 5, color=BLUE_D),
        )
        pies.arrange(RIGHT)
        pies.move_to(FRAME_HEIGHT * DOWN / 4)

        self.add(expr)
        self.wait()
        self.play(
            expr.animate.move_to(FRAME_HEIGHT * UP / 4),
            ShowCreation(h_line),
            Write(numbers),
        )
        self.play(
            *(
                FadeTransform(expr[i].copy(), pies[i])
                for i in range(len(pies))
            ),
            FadeTransform(numbers.copy(), quantities)
        )
        self.wait()

        # Evaluate bottom
        bottom_eq = Tex("=")
        bottom_eq.move_to(pies[2])
        self.play(
            pies.animate.next_to(bottom_eq, LEFT),
            Write(bottom_eq),
        )

        bottom_rhs = VGroup(
            *pies[0][:2].copy(),
            pies[2][0].copy()
        )
        bottom_rhs.generate_target()
        bottom_rhs.target[2].shift(pies[0].get_center() - pies[2].get_center())
        bottom_rhs.target[2].rotate(TAU * 2 / 3, about_point=pies[0].get_center())
        bottom_rhs.target.next_to(bottom_eq, RIGHT)

        self.play(MoveToTarget(bottom_rhs))
        self.wait()

        # Evaluate top
        rhs = Tex(
            "=",
            "{10 \\over 15}",
            "+",
            "{3 \\over 15}",
            "=",
            "{13 \\over 15}"
        )
        rhs.match_height(expr)
        rhs.next_to(expr, RIGHT)
        expr.generate_target()
        VGroup(expr.target, rhs).set_x(0)

        self.play(
            MoveToTarget(expr),
            FadeIn(rhs, lag_ratio=0.1)
        )
        self.wait()

        # Show overlays
        overlays = VGroup()
        for pie in (*pies[::2], bottom_rhs):
            overlay = get_pie(0, 15)
            overlay.set_fill(BLACK, 0)
            overlay.set_stroke(WHITE, 1, opacity=0.5)
            overlay.move_to(pie)
            overlays.add(overlay)

        self.play(ShowCreation(overlays))
        self.wait()


class CalculusStatement(Scene):
    def construct(self):
        statement = VGroup(
            TexText(
                "If $f(x)$ has a local maximum or minimum\\\\"
                "at $x_0$, and $f$ is differentiable at $x_0$, then"
            ),
            Tex("\\frac{df}{dx}(x_0) = 0.")
        )
        statement.arrange(DOWN)
        statement[1].scale(1.5, about_edge=UP).shift(0.25 * DOWN)
        self.add(statement)


class ExamplesOfFunctions(Scene):
    def construct(self):
        plane = NumberPlane((-15, 15, 5), (-10, 10, 5))
        plane.scale(0.5)
        self.add(plane)

        graphs = VGroup(
            plane.get_graph(lambda x: -x * (x - 4)),
            plane.get_graph(lambda x: x / 3),
            plane.get_graph(lambda x: (1 / 3) * x**3 - 2 * x + 1),
            plane.get_graph(lambda x: -5 * x * np.exp(-x**2 / 2)),
        )
        graph_labels = VGroup(
            Tex("f(x) = ", "-x^2 - 4x"),
            Tex("f(x) = ", "x / 3"),
            Tex("f(x) = ", "x^3 / 3- 2x + 1"),
            Tex("f(x) = ", "-(5x) e^{-{1 \\over 2} x^2}"),
        )
        colors = [YELLOW, RED, TEAL, PINK]
        for graph, color, label in zip(graphs, colors, graph_labels):
            graph.set_stroke(color)
            label.scale(1.5)
            label[1].set_color(color)
            label.next_to(plane.c2p(0, 5), UR, SMALL_BUFF)
            label.set_stroke(BLACK, 3, background=True)

        self.play(
            FadeIn(graph_labels[0]), ShowCreation(graphs[0]),
            run_time=1.5
        )
        self.wait()
        for i in range(1, len(graphs)):
            self.play(
                ReplacementTransform(graphs[i - 1], graphs[i]),
                FadeTransform(graph_labels[i - 1], graph_labels[i]),
                run_time=0.5
            )
            self.wait(1.5)


class SpecificCases(Scene):
    def construct(self):
        self.add(FullScreenRectangle())
        title = Text("Applications for optimization", font_size=72)
        title.to_edge(UP)
        self.add(title)

        grid = Square().get_grid(1, 2, buff=0)
        grid.set_height(6)
        grid.set_width(13, stretch=True)
        grid.to_edge(DOWN)
        grid.set_fill(BLACK, 1)

        labels = VGroup(
            Text("Profit maximization"),
            Text("Distance between curves"),
        )
        for label, square in zip(labels, grid):
            label.next_to(square.get_top(), DOWN)

        self.add(grid)
        self.add(labels)

        # Fill 'em
        r_axes = Axes(
            x_range=(-3, 3), y_range=(-3, 3),
            width=6, height=4.5,
            axis_config={"include_tip": False},
        )
        r_axes.move_to(grid[1], DOWN).shift(0.25 * UP)
        circle = Circle()
        circle.set_height(get_norm(r_axes.c2p(0, 0) - r_axes.c2p(0, 2)))
        circle.move_to(r_axes.c2p(1, 1))
        circle.set_stroke(YELLOW)
        parabola = r_axes.get_graph(
            lambda x: 0.2 * (x**2 - 4)
        )
        parabola.set_stroke(RED_D)
        smallest_line = Line(
            circle.pfp(0.83),
            parabola.pfp(0.76),
            stroke_width=3
        )

        self.add(circle, parabola, smallest_line)

        l_axes = Axes(
            (0, 10),
            (-3, 8),
            width=6, height=4.5,
        )
        l_axes.move_to(grid[0], DOWN).shift(0.25 * UP)
        x_label = Text("production", font_size=24)
        x_label.next_to(l_axes.x_axis, DOWN, SMALL_BUFF, aligned_edge=RIGHT)
        y_label = Text("profit", font_size=24)
        y_label.next_to(l_axes.y_axis, RIGHT, SMALL_BUFF, aligned_edge=UP)
        graph = l_axes.get_graph(
            lambda x: -0.07 * x * (x - 4) * (x - 9.5)
        )
        graph.set_stroke(GREEN, 3)

        self.add(l_axes, x_label, y_label, graph)


class ConcreteToAbstract(Scene):
    def construct(self):
        self.add(
            get_h_line().move_to(FRAME_HEIGHT * UP / 6),
            get_h_line().move_to(FRAME_HEIGHT * DOWN / 6),
        )

        labels = VGroup(
            Text("Algebra"),
            Text("Numbers"),
            Text("Quantities"),
        )
        for i, label in enumerate(labels):
            label.scale(30 / label.font_size)
            label.to_corner(UL, buff=SMALL_BUFF)
            label.shift(i * FRAME_HEIGHT * DOWN / 3)

        self.add(labels)

        # Algebra
        expr = Tex(
            "x^2 - y^2 = (x + y)(x - y)",
            tex_to_color_map={"x": BLUE_D, "y": BLUE_B}
        )
        expr.scale(1.5)
        expr.move_to(FRAME_HEIGHT * UP / 3)
        expr.to_edge(LEFT, buff=LARGE_BUFF)
        self.add(expr)

        # Numbers
        tricks = VGroup(
            Tex("143 = (12 + 1)(12 - 1) = 13 \\cdot 11"),
            Tex("3{,}599 = (60 + 1)(60 - 1) = 61 \\cdot 59"),
            Tex("9{,}991 = (100 + 3)(100 - 3) = 103 \\cdot 97"),
        )
        tricks.arrange(DOWN, aligned_edge=LEFT)
        tricks[0].shift((tricks[1][0][5].get_x() - tricks[0][0][3].get_x()) * RIGHT)
        tricks.set_height(FRAME_HEIGHT / 3 - 1)
        tricks.center()
        brace = Brace(tricks, LEFT)
        words = TexText("Factoring\\\\tricks", font_size=30)
        words.set_color(GREY_A)
        words.next_to(brace, LEFT)
        VGroup(tricks, brace, words).to_edge(LEFT, buff=MED_LARGE_BUFF)

        self.add(tricks, brace, words)
        self.wait()

        # Arrows
        arrow = Arrow(
            ORIGIN, FRAME_HEIGHT * UP / 3,
            path_arc=60 * DEGREES,
            thickness=0.15,
            fill_color=YELLOW,
            stroke_color=YELLOW,
        )
        up_arrows = VGroup(arrow.copy().shift(FRAME_HEIGHT * DOWN / 3), arrow)
        up_arrows.to_edge(RIGHT, buff=1.5)

        self.play(Write(up_arrows, lag_ratio=0.5))
        self.wait()

        top_words = Text("Once you understand\n algebra...")
        top_words.next_to(up_arrows, UP)
        top_words.shift_onto_screen()
        top_words.set_color(YELLOW)

        self.play(FadeIn(top_words, lag_ratio=0.05))
        self.play(LaggedStartMap(Rotate, up_arrows, angle=PI, axis=RIGHT))
        up_arrows.refresh_unit_normal()
        self.wait()

        new_top_words = Text("If you're learning\n algebra.")
        new_top_words.move_to(top_words)
        new_top_words.match_color(top_words)
        self.play(
            FadeTransform(top_words, new_top_words),
            LaggedStartMap(Rotate, up_arrows, angle=PI, axis=RIGHT)
        )
        up_arrows.refresh_unit_normal()
        self.wait()

        self.embed()


class DifferenceOfSquares(Scene):
    def construct(self):
        x = 6
        y = 1
        squares = VGroup(*[
            VGroup(*[
                Square()
                for x in range(x)
            ]).arrange(RIGHT, buff=0)
            for y in range(x)
        ]).arrange(DOWN, buff=0)
        squares.set_height(4)
        squares.set_stroke(BLUE_D, 3)
        squares.set_fill(BLUE_D, 0.5)

        last_row_parts = VGroup()
        for row in squares[-y:]:
            row[-y:].set_color(GREY_E)
            row[:-y].set_color(BLUE_B)
            last_row_parts.add(row[:-y])
        squares.to_edge(LEFT)

        arrow = Vector(RIGHT, color=WHITE)
        arrow.shift(1.5 * LEFT)
        squares.next_to(arrow, LEFT)

        new_squares = squares[:-y].copy()
        new_squares.next_to(arrow, RIGHT)
        new_squares.align_to(squares, UP)

        x1 = Tex(str(x)).set_color(BLUE_D)
        x2 = x1.copy()
        x1.next_to(squares, UP)
        x2.next_to(squares, LEFT)
        y1 = Tex(str(y)).set_color(BLUE_B)
        y2 = y1.copy()
        y1.next_to(squares[-int(np.ceil(y / 2))], RIGHT)
        y2.next_to(squares[-1][-int(np.ceil(y / 2))], DOWN)

        xpy = Tex(str(x), "+", str(y))
        xmy = Tex(str(x), "-", str(y))
        for mob in xpy, xmy:
            mob[0].set_color(BLUE)
            mob[2].set_color(BLUE_B)
        xpy.next_to(new_squares, UP)
        # xmy.rotate(90 * DEGREES)
        xmy.next_to(new_squares, RIGHT)
        xmy.shift(squares[0][0].get_width() * RIGHT)

        self.add(squares, x1, x2, y1, y2)
        self.play(
            ReplacementTransform(
                squares[:-y].copy().set_fill(opacity=0),
                new_squares
            ),
            ShowCreation(arrow),
            lag_ratio=0,
        )
        last_row_parts = last_row_parts.copy()
        last_row_parts.save_state()
        last_row_parts.set_fill(opacity=0)
        self.play(
            last_row_parts.restore,
            last_row_parts.rotate, -90 * DEGREES,
            last_row_parts.next_to, new_squares, RIGHT, {"buff": 0},
            lag_ratio=0,
        )
        self.play(Write(xmy), Write(xpy))
        self.wait()


class AbstractVectorSpace(Scene):
    def construct(self):
        self.add(get_h_line())
        top_title = Text("Abstract vector space axioms")
        top_title.to_edge(UP, buff=MED_SMALL_BUFF)
        low_title = Text("Concrete vectors")
        low_title.next_to(ORIGIN, DOWN, buff=MED_SMALL_BUFF)
        self.add(top_title, low_title)

        # Vectors
        kw = {"bracket_h_buff": 0.1}
        columns = VGroup(
            Matrix([["1"], ["1"]], **kw).set_color(BLUE_B),
            Tex("+"),
            Matrix([["-2"], ["3"]], **kw).set_color(BLUE_D),
            Tex("="),
            Matrix([["1 - 2"], ["1 + 3"]], **kw).set_color(GREEN),
        )
        columns.arrange(RIGHT, buff=SMALL_BUFF)
        columns.next_to(low_title, DOWN, LARGE_BUFF)
        columns.to_edge(LEFT)

        arrows = VGroup(
            Arrow(ORIGIN, (1, 1), fill_color=BLUE_B, buff=0),
            Arrow((1, 1), (-1, 4), fill_color=BLUE_D, buff=0),
            Arrow(ORIGIN, (-1, 4), fill_color=GREEN, buff=0),
        )
        arrows.match_height(columns)
        arrows.scale(1.5)
        arrows.match_y(columns)
        arrows.match_x(low_title)

        funcs = Tex(
            "(f + g)(x) = f(x) + g(x)",
            tex_to_color_map={"f": BLUE_B, "g": BLUE_D}
        )
        funcs.next_to(arrows, RIGHT, LARGE_BUFF)

        self.add(columns)
        self.add(arrows)
        self.add(funcs)

        # Axioms
        u_tex = "\\vec{\\textbf{u}}"
        w_tex = "\\vec{\\textbf{w}}"
        v_tex = "\\vec{\\textbf{v}}"
        axioms = VGroup(*it.starmap(Tex, [
            (
                "1. \\,",
                u_tex, "+", "(", v_tex, "+", w_tex, ")=(",
                u_tex, "+", v_tex, ")+", w_tex
            ),
            (
                "2. \\,",
                v_tex, "+", w_tex, "=", w_tex, "+", v_tex
            ),
            (
                "3. \\,",
                "\\textbf{0}+", v_tex,
                "=", v_tex, "\\text{ for all }", v_tex
            ),
            (
                "4. \\,",
                "\\forall", v_tex, "\\;\\exists", w_tex,
                "\\text{ s.t. }", v_tex, "+", w_tex, "=\\textbf{0}"
            ),
            (
                "5. \\,",
                "a", "(", "b", v_tex, ")=(", "a", "b", ")", v_tex
            ),
            (
                "6. \\,",
                "1", v_tex, "=", v_tex
            ),
            (
                "7. \\,",
                "a", "(", v_tex, "+", w_tex, ")", "=",
                "a", v_tex, "+", "a", w_tex
            ),
            (
                "8. \\,",
                "(", "a", "+", "b", ")", v_tex, "=",
                "a", v_tex, "+", "b", v_tex
            ),
        ]))
        for axiom in axioms:
            axiom.set_color_by_tex_to_color_map({
                u_tex: BLUE_B,
                w_tex: BLUE_D,
                v_tex: GREEN,
            })
        axioms[:4].arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        axioms[4:].arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        axioms[4:].next_to(axioms[:4], RIGHT, buff=1.5)
        axioms.set_height(2.5)
        axioms.next_to(top_title, DOWN)

        for axiom in axioms:
            self.play(FadeIn(axiom))

        self.add(axioms)


class Nicheness(Scene):
    def construct(self):
        words = TexText(
            "Perceived ", "nicheness",
            " $>$ ",
            "Actual ", "nichness",
        )
        words[:2].set_color(BLUE_B)
        words[3:].set_color(BLUE_D)

        self.add(words[:2])
        self.wait()
        self.play(
            FadeTransformPieces(words[:2].copy(), words[3:], path_arc=PI / 2),
            GrowFromCenter(words[2]),
        )
        self.wait()


class TransitionToProductionQuality(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.students[0].change("pondering"),
            self.students[2].change("pondering"),
        )
        self.student_says(
            TexText("What parts of\\\\production quality matter?"),
            index=1
        )
        self.play(
            self.teacher.change("happy"),
        )
        self.wait(3)


class SurfaceExample(Scene):
    CONFIG = {
        "camera_class": ThreeDCamera,
    }

    def construct(self):
        torus1 = Torus(r1=1, r2=1)
        torus2 = Torus(r1=3, r2=1)
        sphere = Sphere(radius=2.5, resolution=torus1.resolution)
        # You can texture a surface with up to two images, which will
        # be interpreted as the side towards the light, and away from
        # the light.  These can be either urls, or paths to a local file
        # in whatever you've set as the image directory in
        # the custom_config.yml file

        day_texture = "EarthTextureMap"
        night_texture = "NightEarthTextureMap"

        surfaces = [
            TexturedSurface(surface, day_texture, night_texture)
            for surface in [sphere, torus1, torus2]
        ]

        for mob in surfaces:
            mob.mesh = SurfaceMesh(mob)
            mob.mesh.set_stroke(BLUE, 1, opacity=0.5)

        # Set perspective
        frame = self.camera.frame
        frame.set_euler_angles(
            theta=-30 * DEGREES,
            phi=70 * DEGREES,
        )

        surface = surfaces[0]

        self.play(
            FadeIn(surface),
            ShowCreation(surface.mesh, lag_ratio=0.01, run_time=3),
        )
        for mob in surfaces:
            mob.add(mob.mesh)
        surface.save_state()
        self.play(Rotate(surface, PI / 2), run_time=2)
        for mob in surfaces[1:]:
            mob.rotate(PI / 2)

        self.play(
            Transform(surface, surfaces[1]),
            run_time=3
        )

        self.play(
            Transform(surface, surfaces[2]),
            # Move camera frame during the transition
            frame.animate.increment_phi(-10 * DEGREES),
            frame.animate.increment_theta(-20 * DEGREES),
            run_time=3
        )
        # Add ambient rotation
        frame.add_updater(lambda m, dt: m.increment_theta(-0.1 * dt))

        # Play around with where the light is

        light = self.camera.light_source
        self.add(light)
        light.save_state()
        self.play(light.animate.move_to(3 * IN), run_time=5)
        self.play(light.animate.shift(10 * OUT), run_time=5)


class Spotlight(Scene):
    def construct(self):
        self.add(FullScreenRectangle())
        screen = ScreenRectangle()
        screen.set_height(6.0)
        screen.set_stroke(WHITE, 2)
        screen.set_fill(BLACK, 1)
        screen.to_edge(DOWN)
        animated_screen = AnimatedBoundary(screen)
        self.add(screen, animated_screen)
        self.wait(16)


class BadManimExample(Scene):
    def construct(self):
        words = TexText(
            "Does any of this ",
            "need to be ",
            "animated?\\\\",
            "Much less programatically?"
        )
        words[-1].shift(MED_SMALL_BUFF * DOWN)
        words.set_width(FRAME_WIDTH - 2)

        self.play(Write(words[0]), run_time=1)
        self.play(FadeIn(words[1], scale=10, shift=0.25 * UP))
        self.play(TransformMatchingShapes(words[0].copy(), words[2], path_arc=PI / 2))
        self.wait()
        self.play(Transform(
            VGroup(*words[0], *words[1], *words[2]).copy(),
            words[3],
            lag_ratio=0.03,
            run_time=1
        ))
        self.wait()


class WhereCanIEngageWithOthers(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            TexText("Where can I find\\\\others joining SoME1?"),
            index=0,
            added_anims=[
                self.students[1].change("pondering", UL),
                self.students[2].change("pondering", UL),
            ]
        )
        self.play(
            self.teacher.change("tease"),
        )
        self.wait(4)


class BeTheFirst(Scene):
    def construct(self):
        circ = Circle(color=RED)
        circ.stretch(5, 0)
        circ.set_height(0.4)

        circ.to_edge(LEFT)
        words = Text("Be the first!")
        words.next_to(circ, RIGHT, aligned_edge=UP)
        words.set_color(RED)

        self.play(ShowCreation(circ))
        self.play(Write(words))
        self.wait()


class SoME1EndScreen(PatreonEndScreen):
    pass