from manim_imports_ext import *
from _2023.clt.main import *
import sympy


class GaltonBoardName(InteractiveScene):
    def construct(self):
        # Test
        name = Text("Galton \nBoard", font_size=120)
        name.next_to(ORIGIN, RIGHT).shift(2 * UP)
        point = name.get_bottom() + 1 * DOWN
        arrow = Arrow(
            point,
            point + FRAME_WIDTH * LEFT * 0.25,
            stroke_width=10
        )

        self.add(name)
        self.play(GrowArrow(arrow))
        self.wait()


class NormalName(InteractiveScene):
    def construct(self):
        # Names
        names = VGroup(
            Text("Normal distribution"),
            Text("Bell curve"),
            Text("Gaussian distribution"),
        )
        point = 2 * LEFT + 1.5 * UP
        names.next_to(point, UP)
        names.set_color(GREY_A)

        shift = 0.75 * UP
        names[0].scale(2, about_edge=DOWN)
        self.play(Write(names[0]), run_time=1)
        self.wait()
        self.play(
            names[0].animate.scale(0.5, about_edge=DOWN).shift(shift),
            FadeIn(names[1], shift=0.5 * shift),
        )
        self.wait()
        self.play(
            names[:2].animate.shift(shift),
            FadeIn(names[2], shift=0.5 * shift)
        )
        self.wait()

        # In a moment
        words = TexText("We'll unpack this in a bit", font_size=36)
        words.set_fill(GREY_A)
        words.next_to(point, UP)
        words.shift(0.5 * UL)
        arrow = Arrow(
            words.get_bottom() + LEFT, words.get_bottom() + DOWN,
            stroke_width=3, stroke_color=GREY_A,
        )

        self.play(LaggedStart(
            FadeOut(names[1:]),
            FadeIn(words, lag_ratio=0.1),
            ShowCreation(arrow),
            lag_ratio=0.25
        ))
        self.wait()


class ErdosKac(InteractiveScene):
    def construct(self):
        # Title
        h_line = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        VGroup(h_line, v_line).set_stroke(GREY_A, 1)
        h_line.to_edge(UP, buff=1.0)
        titles = VGroup(
            Tex("N"),
            TexText(R"\# of distinct prime factors of $N$"),
        )
        titles.next_to(h_line, UP)
        titles[0].set_x(-FRAME_WIDTH * 0.25)
        titles[1].set_width(6)
        titles[1].to_edge(RIGHT)
        v_line.set_x(titles[1].get_x(LEFT) - 0.25)

        self.add(titles)
        self.add(h_line)

        # Number line
        number_line = NumberLine((0, 10), width=10)
        number_line.add_numbers(excluding=[0])
        number_line.numbers.shift(number_line.get_unit_size() * LEFT / 2)
        number_line.to_edge(DOWN, buff=LARGE_BUFF)
        self.add(number_line)

        stacks = VGroup(*(
            VGroup(VectorizedPoint(number_line.n2p(x - 0.5)))
            for x in range(1, 20)    
        ))
        self.add(stacks)

        rect_template = Rectangle(width=number_line.get_unit_size(), height=0.1)
        rect_template.set_stroke(BLUE_E, 0.5)
        rect_template.set_fill(TEAL_E, 0.5)

        # Add normal curve
        mean = math.log(math.log(1e18))
        sd = math.sqrt(mean)
        curve = FunctionGraph(lambda x: 10 * gauss_func(x, mean, sd), x_range=(0, 10, 0.25))
        curve.set_stroke(RED, 2)
        curve.match_width(number_line)
        curve.move_to(number_line.n2p(0), DL)
        self.add(curve)

        # Show many numbers
        num = int(1e18)
        group = VGroup()

        for x in range(0, 100):
            rect = rect_template.copy()
            nf = len(sympy.factorint(num + x))
            stack = stacks[nf - 1]
            rect.next_to(stack, UP, buff=0)
            rect.set_fill(YELLOW)

            self.remove(group)
            group = self.get_factor_group(num + x, h_line)
            self.add(group, rect)
            if x < 20:
                self.wait(0.5)
            else:
                self.wait(0.1)

            rect.set_fill(TEAL_E)
            stack.add(rect)
            self.add(stacks)

    def get_factor_group(self, num, h_line, font_size=36):
        # Test
        num_mob = Integer(num, font_size=font_size)
        num_mob.next_to(h_line, DOWN, MED_LARGE_BUFF).to_edge(LEFT, buff=0.25)
        rhs = self.get_rhs(num_mob)

        omega_n = len(sympy.factorint(num))
        omega_n_mob = Integer(omega_n, color=YELLOW)
        omega_n_mob.match_y(num_mob)
        omega_n_mob.to_edge(RIGHT, buff=1)

        arrow = Arrow(rhs, omega_n_mob, buff=0.5)
        group = VGroup(num_mob, rhs, arrow, omega_n_mob)

        return group

    def get_rhs(self, num_mob):
        if not isinstance(num_mob, Integer):
            return VGroup()

        kw = dict(font_size=num_mob.get_font_size())
        parts = [Tex("=", **kw)]
        for factor, exp in sympy.factorint(num_mob.get_value()).items():
            base = Integer(factor, **kw)
            underline = Underline(base, stretch_factor=1)
            underline.set_stroke(YELLOW, 2)
            underline.set_y(base[0].get_y(DOWN) - 0.05)
            base.add(underline)
            if exp == 1:
                parts.append(base)
            else:
                exp_mob = Integer(exp, **kw)
                exp_mob.scale(0.75)
                exp_mob.next_to(base.get_corner(UR), RIGHT, buff=0.05)
                parts.append(VGroup(*base, *exp_mob))
            parts.append(Tex(R"\times", **kw))
        result = VGroup(*parts[:-1])
        result.arrange(RIGHT, buff=SMALL_BUFF)
        result.next_to(num_mob, buff=SMALL_BUFF)
        target_y = num_mob[0].get_y()
        for part in result:
            part.shift((target_y - part[0].get_y()) * UP)

        self.add(result)

        return result


class PopulationHeights(InteractiveScene):
    random_seed = 2

    def construct(self):
        # Test
        func = lambda x: gauss_func(x, 1, 0.175)

        stacks = VGroup()
        all_pis = VGroup()
        for height in np.arange(0.6, 1.5, 0.1):
            n = int(10 * func(height))
            randys = VGroup(*(self.get_pi(height) for _ in range(n)))
            randys.arrange(RIGHT, buff=SMALL_BUFF)
            stacks.add(randys)
            all_pis.add(*randys)
        stacks.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        for stack, color in zip(stacks, color_gradient([BLUE_C, BLUE_E], len(stacks))):
            for pi in stack:
                pi.body.set_color(color)
        stacks.set_height(FRAME_HEIGHT - 1)
        stacks.to_edge(LEFT)

        for pi in all_pis:
            pi.save_state()

        all_pis.shuffle()
        all_pis.arrange_in_grid(n_cols=15)
        all_pis.shuffle()


        self.play(FadeIn(all_pis, lag_ratio=0.01, run_time=4))
        self.wait()
        self.play(LaggedStartMap(Restore, all_pis, lag_ratio=0.01, run_time=5))
        self.wait()

    def get_pi(self, height=1):
        pi = Randolph(
            mode=random.choice(["happy", "hooray", "pondering", "tease"]),
            height=1
        )
        pi.body.stretch(math.sqrt(height), 1, about_edge=UP)
        diff = height - pi.get_height()
        new_points = np.array(pi.body.get_points())
        new_points[3:26] += diff * DOWN
        new_points[63:82] += diff * DOWN
        pi.body.set_points(new_points)

        return pi


class VideoPlan(InteractiveScene):
    def construct(self):
        # Rectangles
        full_rect = FullScreenRectangle()
        screen_rect = ScreenRectangle()
        screen_rect.set_height(5.5)
        screen_rect.to_edge(DOWN)
        screen_rect.set_fill(BLACK, 1)
        screen_rect.set_stroke(width=2)
        rects = VGroup(full_rect, screen_rect)
        self.add(*rects)

        # CLT title
        big_name = Text("The Central\nLimit Theorem", font_size=120)
        name = Text("The Central Limit Theorem", font_size=60)
        name.to_edge(UP)
        subtitle = Text("from the basics")
        subtitle.set_color(GREY_A)
        subtitle.next_to(name, DOWN)

        rects.set_opacity(0)
        self.play(FadeIn(big_name, lag_ratio=0.1))
        self.wait()
        self.play(
            TransformMatchingStrings(big_name, name, run_time=1),
            full_rect.animate.set_opacity(1)
        )
        self.play(FadeIn(subtitle, 0.5 * DOWN))
        self.wait(2)

        # Next part
        group1 = VGroup(name, subtitle, screen_rect)
        group1.target = group1.generate_target()
        group1.target.set_width(FRAME_WIDTH * 0.5 - 1)
        group1.target.move_to(FRAME_WIDTH * LEFT * 0.25)

        screen2 = group1.target[2].copy()
        screen2.set_opacity(1)
        name2 = Text("Diving deeper")
        name2.scale(group1.target[0][0].get_height() / name2[0].get_height())
        name2.move_to(group1.target[0])
        group2 = VGroup(name2, screen2)
        group2.move_to(FRAME_WIDTH * RIGHT * 0.25)

        self.play(MoveToTarget(group1))
        self.play(
            FadeTransform(screen_rect.copy(), screen2),
            FadeTransform(name.copy(), name2),
        )
        self.wait()

        # Also follow-on to convolutions
        group1.target = group1.generate_target()
        group1.target[2].scale(0.75, about_edge=UP)
        group1.target[1].scale(0)
        group1.target.arrange(DOWN, buff=SMALL_BUFF)
        group1.target.to_corner(UL)

        conv_rect = group1.target[2].copy()
        conv_rect.set_stroke(WHITE, 3, 1)
        conv_image = ImageMobject("ConvolutionThumbnail")
        conv_image.replace(conv_rect)
        conv_name = Text("Convolutions")
        conv_name.replace(group1.target[0], dim_to_match=1)
        conv_group = Group(conv_name, conv_rect, conv_image)
        conv_group.to_corner(DL)

        arrows = VGroup(
            Arrow(group1.target[2].get_right(), screen2.get_left() + UP),
            Arrow(conv_rect.get_right(), screen2.get_left() + DOWN),
        )
        arrows.set_color(BLUE)

        self.play(LaggedStart(
            MoveToTarget(group1),
            FadeIn(conv_group, DOWN),
            name2.animate.next_to(screen2, UP),
            lag_ratio=0.25
        ))
        self.play(*map(GrowArrow, arrows))
        self.wait()


class NextVideoInlay(InteractiveScene):
    def construct(self):
        # Graph
        plane = NumberPlane(
            (-5, 5), (-0.25, 1.0, 0.25),
            width=2.5, height=1.25,
            background_line_style=dict(
                stroke_color=GREY_B,
                stroke_width=1,
            ),
            faded_line_style=dict(
                stroke_color=GREY_B,
                stroke_width=1,
                stroke_opacity=0.25,
            ),
            faded_line_ratio=4
        )
        plane.set_height(FRAME_HEIGHT)

        graph = plane.get_graph(lambda x: gauss_func(x, 0, 1))
        graph.set_stroke(YELLOW, 3)

        self.add(plane)
        self.play(
            VShowPassingFlash(graph.copy().set_stroke(TEAL, 10), time_width=1.5),
            ShowCreation(graph),
            run_time=2
        )

        # Function name
        expr = Tex(
            R"{1 \over \sqrt{2\pi}} e^{-x^2 / 2}",
        )
        expr.set_height(3)
        expr.next_to(plane.c2p(0, 0.25), UP)
        expr.to_edge(LEFT)
        expr.set_backstroke(width=20)

        self.play(Write(expr, lag_ratio=0.1))
        self.wait()

        # Questions
        rects = VGroup(
            SurroundingRectangle(expr["e^{-x^2 / 2}"]).set_stroke(TEAL, 5),
            SurroundingRectangle(expr[R"\pi"]).set_stroke(RED, 5),
        )
        questions = [
            Text("Why this function?"),
            Text("Where's\nthe circle?", alignment="LEFT"),
        ]
        for question, rect, vect in zip(questions, rects, [UP, DOWN]):
            question.scale(2)
            question.next_to(rect, vect, MED_LARGE_BUFF, aligned_edge=LEFT)
            question.match_color(rect)
            question.set_backstroke(width=20)

        self.play(
            FadeIn(questions[0]),
            ShowCreation(rects[0]),
        )
        self.wait()
        self.play(
            FadeIn(questions[1]),
            ShowCreation(rects[1]),
        )
        self.wait()


class SumsOfSizeFive(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students

        self.play(
            morty.change("raise_right_hand", self.screen),
            self.change_students("pondering", "happy", "tease", look_at=self.screen)
        )
        self.wait()

        # Show sums
        def get_sum():
            values = [
                Integer(random.choice([-1, 1]), include_sign=True)
                for x in range(5)
            ]
            lhs = Tex(R"\text{Sum} = ")
            rhs = Integer(sum(v.get_value() for v in values))
            result = VGroup(lhs, *values, Tex(R" = "), rhs)
            result.arrange(RIGHT, buff=0.15)
            result.next_to(self.screen, RIGHT, buff=-0.5)
            return result

        curr_sum = get_sum()

        brace = Brace(curr_sum[1:-2], UP)
        brace_text = brace.get_text("5 values")
        brace_text.set_color(YELLOW)

        self.add(curr_sum)
        self.play(GrowFromCenter(brace), FadeIn(brace_text))
        for x in range(7):
            new_sum = get_sum()
            self.play(
                FadeOut(curr_sum[1:], lag_ratio=0.1, shift=0.2 * UP),
                FadeIn(new_sum[1:], lag_ratio=0.1, shift=0.2 * UP),
                *(pi.animate.look_at(new_sum) for pi in self.pi_creatures)
            )
            self.wait()
            curr_sum = new_sum

        self.wait()


class HighLevelCLTDescription(InteractiveScene):
    def construct(self):
        # Title
        title = Text("General idea of the Central Limit Theorem", font_size=60)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        title["Central Limit Theorem"].set_color(YELLOW)
        underline = Underline(title)
        underline.scale(1.2)
        underline.set_stroke(WHITE, [0, 3, 3, 0])
        self.add(title)
        self.play(ShowCreation(underline))
        self.wait()

        # Random variables
        kw = dict(font_size=42)
        words = TexText("Start with a random variable: $X$", **kw)
        words["random variable: $X$"].set_color(BLUE)
        sub_words = Text("""
            (a random process, where each outcome
            is associated with some number)
        """, font_size=32)
        sub_words.next_to(words, DOWN)
        sub_words.set_color(GREY_A)

        point1 = VGroup(words, sub_words)
        point1.next_to(underline, DOWN, buff=0.7)
        point1.to_edge(LEFT, buff=MED_SMALL_BUFF)

        example_boxes = Square().replicate(2)
        example_boxes.set_height(1.75)
        example_boxes.set_width(2.25, stretch=True)
        example_boxes.arrange(RIGHT, buff=1)
        example_boxes.set_stroke(WHITE, 1)
        example_boxes.match_y(point1)
        example_boxes.to_edge(RIGHT)

        self.play(
            Write(words, run_time=1, stroke_width=1),
            LaggedStartMap(FadeIn, example_boxes, lag_ratio=0.5),
        )
        self.wait()
        self.play(FadeIn(sub_words, 0.25 * DOWN))
        self.wait()

        # Sample many
        point2 = TexText(R"""
            Add $N$ samples of this variable \\
            $X_1 + X_2 + \cdots + X_N$
        """, **kw)
        point2["$N$ samples"].set_color(RED)
        point2[R"$X_1 + X_2 + \cdots + X_N$"].set_color(BLUE).shift(0.25 * DOWN)
        point2.next_to(point1, DOWN, buff=1.1, aligned_edge=LEFT)

        example_boxes2 = example_boxes.copy()
        example_boxes2.match_y(point2)

        self.play(
            FadeIn(point2, DOWN),
            FadeIn(example_boxes2, DOWN),
        )
        self.wait()

        # Distribution
        point3 = TexText(R"""
            The distribution of this sum looks \\
            more like a bell curve as $N \to \infty$
        """, **kw)
        point3["this sum"].set_color(BLUE)
        point3["bell curve"].set_color(YELLOW)
        point3[R"$N \to \infty$"].set_color(RED)
        point3.next_to(point2, DOWN, buff=1.25, aligned_edge=LEFT)
        example_boxes3 = example_boxes.copy()
        example_boxes3.match_y(point3)

        self.play(
            FadeIn(point3, DOWN),
            FadeIn(example_boxes3, DOWN),
        )
        self.wait()


class GoalsThisLesson(TeacherStudentsScene):
    def construct(self):
        self.add(self.screen)
        self.screen.set_stroke(width=2)
        morty = self.teacher
        stds = self.students

        self.play(
            morty.change("tease"),
            self.change_students("confused", "thinking", "hesitant", look_at=self.screen)
        )
        self.wait(2)

        # Goals
        title = Text("Goals of this lesson", font_size=60)
        title.to_corner(UR)
        underline = Underline(title)
        underline.scale(1.2)
        underline.set_color(GREY_B)

        goals = BulletedList(
            "Make this quantitative",
            "Put formulas to it",
            "Use it to predict",
            buff=0.35,
            font_size=48
        )
        goals.next_to(underline, DOWN, MED_LARGE_BUFF)
        goals.align_to(title, LEFT)

        self.play(
            FadeIn(title),
            ShowCreation(underline),
            morty.change("raise_right_hand", title),
            self.change_students("erm", "thinking", "tease", look_at=title)
        )
        self.wait()

        for goal in goals:
            self.play(FadeIn(goal, UP))
            self.wait()

        # Three assumptions
        new_title = VGroup(
            Text("3 assumptions underlie the"),
            Text("Central Limit Theorem"),
        )
        for line in new_title:
            line.match_width(title)
        new_title.arrange(DOWN)
        new_title.move_to(title, UP)
        new_title.set_color(RED)

        numbers = VGroup(Tex("1."), Tex("2."), Tex("3."))
        numbers.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        numbers.next_to(new_title, DOWN, buff=0.75, aligned_edge=LEFT)
        numbers.set_color(RED)

        boxes = VGroup(*(
            Rectangle(height=1.2 * num.get_height(), width=4).next_to(num)
            for num in numbers
        ))
        boxes.set_stroke(width=0)
        boxes.set_fill(interpolate_color(RED_E, BLACK, 0.8), 1)
        for box in boxes:
            box.save_state()
            box.stretch(0, 0, about_edge=LEFT)

        self.play(
            FadeOut(title, 0.5 * DOWN),
            FadeIn(new_title, 0.5 * DOWN),
            underline.animate.next_to(new_title, DOWN, SMALL_BUFF).set_color(RED_E),
            LaggedStartMap(FadeOut, goals, shift=DOWN),
            morty.change("tease", new_title),
            self.change_students("hesitant", "pondering", "skeptical", new_title),
        )
        self.play(
            FadeIn(numbers, lag_ratio=0.25),
            morty.change("raise_left_hand")
        )
        self.wait(2)
        self.play(
            LaggedStartMap(Restore, boxes),
            LaggedStart(
                morty.change("tease"),
                stds[0].change("angry"),
                stds[1].change("hesitant"),
                stds[2].change("erm"),
                lag_ratio=0.2
            )
        )
        self.wait(4)

        # # Reference again and look down
        # self.remove(self.screen)
        # self.remove(self.background)

        # self.play(
        #     morty.change("raise_left_hand", boxes),
        #     self.change_students(look_at=boxes)
        # )
        # self.wait(2)
        # self.play(
        #     self.change_students("pondering", "pondering", "pondering", look_at=BOTTOM + 3 * LEFT),
        #     morty.change("tease", stds)
        # )
        # self.wait(3)


class CommentOnSpikeyBellCurve(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        morty = self.teacher
        stds = self.students

        # Test
        self.play(
            morty.says("Kind of a\nbell curve", mode="shruggie"),
            self.change_students("erm", "hesitant", "hesitant", look_at=self.screen)
        )
        self.wait(3)
        self.play(
            stds[1].says("Is it...supposed to\nlook like that?"),
            morty.debubble(),
        )
        self.look_at(self.screen)
        self.wait(2)
        self.play(
            stds[2].says("How many samples\nuntil we're sure?"),
            stds[1].debubble(),
        )
        self.wait(5)


class ConvolutionsWrapper(VideoWrapper):
    title = "Convolutions"
    wait_time = 8


class WhatElseDoYouNotice(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        self.remove(self.background)
        self.play(
            morty.says("What else do\nyou notice?"),
            self.change_students("pondering", "erm", "pondering", look_at=self.screen)
        )
        self.wait(2)
        self.play(self.change_students("erm", "pondering", "tease", look_at=self.screen))
        self.wait(2)


class ReferenceMeanAndSD(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.remove(self.background)

        self.play(
            morty.says("We must put\nnumbers to those!", mode="surprised"),
            self.change_students("hesitant", "concerned_musician", "tease")
        )
        self.look_at(self.screen)
        self.wait(3)

        # Mean and sd
        kw = dict(t2c={R"\mu": PINK, R"\sigma": RED}, font_size=42)
        terms = VGroup(
            Tex(R"\text{Mean: } \mu = E[X]", **kw),
            Tex(R"\text{Std dev: } \sigma = \sqrt{E[(X - \mu)^2]}", **kw),
        )

        terms.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        terms.next_to(morty, UP, LARGE_BUFF)
        terms.to_edge(RIGHT)

        self.play(
            morty.debubble(mode="raise_right_hand", look_at=terms),
            self.change_students("pondering", "hesitant", "happy", look_at=terms),
            LaggedStartMap(FadeIn, terms, shift=UP, lag_ratio=0.5)
        )
        self.wait(3)
        self.play(
           stds[2].says("Great!", mode="hooray"),
           morty.change("happy")
        )
        self.play(
            stds[0].says("Wait, can you\nremind me?", mode="guilty"),
            morty.change("tease")
        )
        self.wait(5)


class AskWhy(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students

        self.play(
            stds[1].says("Why?", mode="raise_left_hand")
        )
        self.play(
            morty.change("tease"),
            self.change_students("pondering", "raise_left_hand", "confused", look_at=morty.get_top() + 3 * UP)
        )
        self.wait(8)


class PDFWrapper(VideoWrapper):
    title = "Probability Density Functions"
    wait_time = 8


class AskAboutPi(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.remove(self.background)

        self.play(LaggedStart(
            morty.change("happy"),
            stds[1].says("Wait, what?!", mode="surprised"),
            stds[0].change("confused", self.screen),
            stds[2].change("hesitant", self.screen),
        ))
        self.wait()
        self.play(
            stds[2].says("Where's the\ncircle?", mode="raise_left_hand")
        )
        self.wait(3)
        self.play(
            stds[1].debubble(),
            stds[2].debubble(),
            morty.says(TexText(R"We'll cover \\ that later"))
        )
        self.wait(2)


class WhyPiQuestion(InteractiveScene):
    def construct(self):
        # Axes
        axes = Axes((-3, 3), (0, 2, 0.5), width=10, height=5)
        axes.x_axis.add_numbers(font_size=16)
        axes.y_axis.add_numbers(num_decimal_places=1, font_size=16)
        graph = axes.get_graph(lambda x: math.exp(-x**2))
        graph.set_stroke(BLUE, 2)
        graph.set_fill(TEAL, 0.5)

        self.add(axes, graph)

        # Labels
        graph_label = Tex("e^{-x^2}", font_size=72)
        graph_label.next_to(graph.pfp(0.4), UL)
        self.add(graph_label)

        area_label = Tex(R"\text{Area} = \sqrt{\pi}")
        area_label.move_to(graph, UR)
        area_label.shift(UL)
        arrow = Arrow(area_label.get_bottom(), graph.get_center() + 0.5 * RIGHT)

        self.add(area_label, arrow)

        question = Text("But where's the circle?", font_size=30)
        question.set_color(YELLOW)
        question.next_to(area_label, DOWN)
        question.to_edge(RIGHT)
        self.add(question)


class WeCanBeMoreElegant(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.says("We can be\nmore elegant"),
            self.change_students("pondering", "confused", "sassy", look_at=self.screen)
        )
        self.wait()
        self.play(self.students[2].change("tease", look_at=self.teacher.eyes))
        self.wait(3)


class OneMoreNuance(InteractiveScene):
    def construct(self):
        morty = Mortimer()
        morty.to_edge(DOWN)

        self.play(morty.says("One more\nquick nuance", mode="speaking"))
        self.play(Blink(morty))
        self.play(morty.change("tease", look_at=BOTTOM))
        self.wait()
        for x in range(2):
            self.play(Blink(morty))
            self.wait()


class LetsHaveFun(InteractiveScene):
    def construct(self):
        morty = Mortimer()
        morty.to_edge(DOWN)

        self.play(morty.says("Let's have\nsome fun", mode="hooray", look_at=BOTTOM))
        for x in range(2):
            self.play(Blink(morty))
            self.wait()


class AskAboutFormalStatement(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        question = Text(
            "But what is the\ncentral limit\ntheorem",
            t2s={"theorem": ITALIC},
            t2c={"theorem": YELLOW},
        )

        self.play(
            stds[2].says(question, mode="sassy", bubble_direction=LEFT)
        )
        self.play(morty.change("guilty"))
        self.play(
            stds[0].change("pondering"),
            stds[1].change("erm"),
        )
        self.wait(8)


class TrueTheoremWords(InteractiveScene):
    def construct(self):
        words = "Actual rigorous\nno-jokes-this-time\nCentral Limit Theorem"
        text = Text(words)
        text.set_height(0.3 * FRAME_HEIGHT)
        text.to_edge(UP)

        for word in ["Actual", "rigorous", "no-","jokes-", "this-", "time", "Central", "Limit", "Theorem"]:
            self.add(text[word])
            self.wait(0.1 * len(word))
        self.wait()


class ExampleQuestion(TeacherStudentsScene):
    def construct(self):
        # Setup
        question = Text("""
            Consider rolling a fair die 100 times,
            and adding the results.

            Find a range of values such that you're
            95% sure the sum will fall within
            this range.
        """, alignment="LEFT")
        question.to_corner(UL)

        die_values = np.random.randint(1, 7, 100)
        dice = VGroup(*(
            DieFace(
                value,
                fill_color=BLUE_E,
                stroke_width=1,
                dot_color=WHITE
            )
            for value in die_values
        ))
        dice.arrange_in_grid()
        dice.set_height(3.0)
        dice.next_to(question, RIGHT, buff=1)
        dice.to_edge(UP, buff=1.0)

        sum_label = TexText("Sum = 100")
        sum_label.next_to(dice, UP)
        num = sum_label.make_number_changeable("100")
        n_tracker = ValueTracker(0)
        num.add_updater(lambda m: m.set_value(sum(die_values[:int(n_tracker.get_value())])))

        # Ask for an example
        morty = self.teacher
        stds = self.students

        self.play(
            stds[2].says("Can we see a\nconcrete example?"),
            morty.change("happy")
        )
        self.wait()

        # Add up dice
        die_highlights = dice.copy()
        die_highlights.set_stroke(YELLOW, 1)

        part1 = re.findall(r"Consider .*\n.* results.", question.get_string())[0]
        self.play(
            morty.change("raise_right_hand", question),
            self.change_students("pondering", "pondering", "tease", look_at=question),
            FadeIn(question[part1], lag_ratio=0.1, run_time=1.5),
            FadeOut(stds[2].bubble),
            FadeOut(stds[2].bubble.content),
        )
        self.play(
            FlashUnder(question["fair die"]),
            question["fair die"].animate.set_color(YELLOW),
            FadeIn(dice, lag_ratio=0.03, run_time=2),
        )
        self.play(
            *(pi.animate.look_at(dice) for pi in [morty, *stds])
        )
        n_tracker.set_value(0)
        self.play(
            VFadeIn(sum_label),
            n_tracker.animate.set_value(100).set_anim_args(run_time=2),
            ShowIncreasingSubsets(die_highlights, run_time=2)
        )
        self.play(FadeOut(die_highlights))
        self.wait()

        # Find a range
        part2 = re.findall(r"Find .*\n.*\n.* range.", question.get_string())[0]
        self.play(
            morty.change("tease", question),
            self.change_students("erm", "hesitant", "pondering", look_at=question),
            FadeIn(question[part2], lag_ratio=0.1, run_time=1.5),
        )
        self.wait()
        self.play(
            FlashUnder(question[r"95% sure"], color=TEAL),
            question[r"95% sure"].animate.set_color(TEAL)
        )
        self.wait(2)
        self.play(self.change_students("confused", "erm", "tease", look_at=question))
        self.wait(3)


class ExampleQuestionNoPiCreatures(InteractiveScene):
    def construct(self):
        # Setup
        question = [
            Text("""
                Consider rolling a die
                100 times, and adding the
                results.
            """, alignment="LEFT"),
            Text("""
                Find a range of values such
                that you're 95% sure the sum
                will fall within this range.
            """, alignment="LEFT")
        ]
        question[0].to_corner(UL)
        question[1].to_corner(UR)

        # Dice
        def get_dice():
            die_values = np.random.randint(1, 7, 100)
            dice = VGroup(*(
                DieFace(
                    value,
                    fill_color=BLUE_E,
                    stroke_width=1,
                    dot_color=WHITE
                )
                for value in die_values
            ))
            dice.arrange_in_grid()
            dice.set_height(4.5)
            dice.next_to(question[0], DOWN, buff=1)
            return dice

        dice = get_dice()

        sum_label = TexText("Sum = 100", font_size=60)
        sum_label.next_to(dice, RIGHT, buff=LARGE_BUFF)
        num = sum_label.make_number_changeable("100")
        num.add_updater(lambda m: m.set_value(sum(
            die.value for die in dice
        )))

        # Add up dice
        self.play(FadeIn(question[0], lag_ratio=0.1, run_time=1.5))
        self.play(
            VFadeIn(sum_label),
            ShowIncreasingSubsets(dice, run_time=2)
        )
        self.wait()

        # Find a range
        self.play(
            FadeIn(question[1], lag_ratio=0.1, run_time=1.5),
        )
        self.wait()
        self.play(
            FlashUnder(question[1][r"95% sure"], color=TEAL),
            question[1][r"95% sure"].animate.set_color(TEAL)
        )

        # More sums
        for x in range(5):
            old_dice = dice.copy()
            dice.set_submobjects(list(get_dice()))
            self.play(ShowIncreasingSubsets(dice, run_time=2), FadeOut(old_dice, run_time=0.5))
            self.wait()


class AverageDiceValues(InteractiveScene):
    def construct(self):
        # Test
        avg_label = TexText("Average value = 0.00")
        avg_label.make_number_changeable("0.00")
        avg_label.next_to(ORIGIN, RIGHT)
        avg_label.set_y(FRAME_HEIGHT / 4)
        self.add(avg_label)

        self.old_dice = VGroup()
        for _ in range(20):
            self.show_sample(avg_label, 100)
            self.wait(2)

    def show_sample(self, avg_label, n_dice, added_anims=[]):
        dice = VGroup(*(
            DieFace(random.randint(1, 6), fill_color=BLUE_E, dot_color=WHITE, stroke_width=1)
            for x in range(n_dice)
        ))
        dice.arrange_in_grid()
        dice.set_height(3.5)
        dice.next_to(avg_label, LEFT, buff=LARGE_BUFF)
        dice.to_edge(UP, buff=MED_SMALL_BUFF)

        self.play(LaggedStart(
            FadeOut(self.old_dice, run_time=0.5),
            ShowIncreasingSubsets(dice, run_time=1, int_func=np.ceil),
            UpdateFromFunc(avg_label[-1], lambda m: m.set_value(
                np.mean([die.value for die in dice])
            )),
            *added_anims
        ))
        avg_label[-1].set_color(YELLOW)

        self.old_dice = dice


class DoesThisMakeSense(InteractiveScene):
    def construct(self):
        rect = SurroundingRectangle(Tex(R"\sigma_a = \sigma / \sqrt{100} = 0.171"))
        words = Text("Does this make sense?")
        words.next_to(rect, DOWN)
        words.align_to(rect.get_center(), RIGHT)

        rect.set_stroke(RED, 2)
        words.set_color(RED)

        self.play(
            ShowCreation(rect),
            Write(words)
        )
        self.wait()


class OneMoreSideNote(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.change_students("pondering", "pondering", "erm", look_at=self.screen)
        )
        self.wait()
        self.play(
            self.teacher.says(
                "Will you tolerate\none more\nside note?",
                mode="speaking",
            ),
        )
        self.play(
            self.change_students("tired", "pondering", "hesitant")
        )
        self.wait(4)


class ThreeAssumptions(InteractiveScene):
    def construct(self):
        # Title
        title = Text("Three assumptions", font_size=60)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        title.set_color(RED)
        underline = Underline(title, stretch_factor=1.5)
        underline.match_color(title)
        underline.shift(0.15 * UP)


        # Points
        points = [
            TexText(R"1. All $X_i$'s are independent \\ from each other."),
            TexText(R"2. Each $X_i$ is drawn from \\ the same distribution."),
            TexText(R"3. $0 < \text{Var}(X_i) < \infty$"),
        ]
        points[0]["from each other."].align_to(points[0]["All"], LEFT)
        points[1]["the same distribution."].align_to(points[1]["Each"], LEFT)
        VGroup(*points).arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT).to_edge(LEFT)
        for point in points:
            point["X_i"].set_color(BLUE)

        rects = VGroup(*(SurroundingRectangle(point[2:]) for point in points))
        rects.set_stroke(width=0)
        rects.set_fill(interpolate_color(RED_E, BLACK, 0.9), 1.0)

        # Add title and numbers
        self.add(title)
        self.play(ShowCreation(underline, run_time=2))
        self.play(
            LaggedStart(*(
                Write(point[:2])
                for point in points
            ), lag_ratio=0.7),
            LaggedStartMap(FadeIn, rects, lag_ratio=0.7)
        )
        self.add(*points, rects)
        self.wait()

        # Add first two points
        self.play(rects[0].animate.stretch(0, 0, about_edge=RIGHT))
        self.remove(rects[0])
        self.wait()

        word = points[0]["independent"]
        self.play(
            FlashUnder(word, color=TEAL),
            word.animate.set_color(TEAL),
        )
        self.wait()

        self.play(rects[1].animate.stretch(0, 0, about_edge=RIGHT))
        self.remove(rects[1])
        self.wait()
        word = points[1]["same distribution"]
        self.play(
            FlashUnder(word, color=YELLOW),
            word.animate.set_color(YELLOW),
        )
        self.wait()

        # Mention iid
        box = SurroundingRectangle(VGroup(*points[:2]))
        box.set_stroke(GREY_C, 2)
        iid_words = TexText(R"i.i.d. $\rightarrow$ independent and identically distributed")
        iid_words.next_to(box, UP, MED_LARGE_BUFF, aligned_edge=LEFT)

        title.add(underline)

        self.play(
            FadeOut(title),
            ShowCreation(box),
        )
        self.play(Write(iid_words))
        self.wait(2)
        self.play(
            FadeOut(box),
            FadeOut(iid_words),
            FadeIn(title)
        )
        self.wait()

        # Mention generalizations
        gen_words = TexText(R"These can be relaxed (see Lindeberg CLT, $\alpha$-mixing, etc.)")
        gen_words.scale(0.7)
        gen_words.set_color(GREY_A, 1)
        gen_words.next_to(box, UP, aligned_edge=LEFT)
        self.play(
            FadeOut(title),
            FadeIn(gen_words),
            FadeIn(box)
        )
        self.wait(2)
        self.play(
            FadeOut(box),
            FadeOut(gen_words),
            FadeIn(title),
        )
        self.wait()

        # Add last point
        self.play(rects[2].animate.stretch(0, 0, about_edge=RIGHT))
        self.remove(rects[2])
        self.wait(2)


class VariableSum(InteractiveScene):
    def construct(self):
        expr = Tex(R"X_1 + X_2 + \cdots + X_n")
        for c in "12n":
            expr["X_" + c].set_color(BLUE)
        self.add(expr)


class AssumingNormality(InteractiveScene):
    def construct(self):
        # Setup
        randy = Randolph().to_corner(DL)
        morty = Mortimer().to_corner(DR)

        randy.shift(2 * RIGHT)
        thought = ThoughtBubble(height=2, width=3, direction=RIGHT)
        thought.pin_to(randy)
        curve = FunctionGraph(lambda x: 5 * gauss_func(x, 0, 1), x_range=(-3, 3, 0.1))
        curve.set_stroke(YELLOW, 3)
        curve.scale(0.35)
        curve.move_to(thought.get_bubble_center() + 0.15 * UL)

        self.add(FullScreenRectangle().set_fill(GREY_E, 0.5))
        self.add(randy, morty)
        self.add(thought, curve)

        # Words
        self.play(
            randy.says(
                TexText(
                    R"""
                    It's a $3\sigma$ event, \\
                    so $p < 0.003$
                    """,
                    t2c={R"\sigma": RED}
                ),
                look_at=morty.eyes,
                mode="tease"
            ),
            morty.change("hesitant", randy.eyes)
        )
        self.play(Blink(morty))
        self.wait()
        self.play(Blink(randy))
        self.wait()
        self.play(
            morty.says("Is it though?", look_at=randy.eyes, mode="sassy"),
            randy.change("guilty", morty.eyes)
        )
        self.wait()
        self.play(Blink(randy))
        self.wait()


class FiniteExpectations(InteractiveScene):
    def construct(self):
        tex = TexText(R"And finite $E[X]$ for that matter", t2c={"X": BLUE})
        self.add(tex)


class EndScreen(PatreonEndScreen):
    scroll_time = 30
