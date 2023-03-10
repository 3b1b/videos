from manim_imports_ext import *


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
            is assocaited with some number)
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


class AskAboutFormalStatement(InteractiveScene):
    def construct(self):
        pass


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
        num = sum_label.make_number_changable("100")
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


class AverageDiceValues(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        avg_label = TexText("Average value = 0.00")
        avg_label.make_number_changable("0.00")
        avg_label.move_to(FRAME_WIDTH * RIGHT * 0.25)
        avg_label.to_edge(UP)
        self.add(avg_label)

        self.show_sample(avg_label, 100, added_anims=[
            morty.change("raise_right_hand", avg_label),
            self.change_students("pondering", "thinking", "pondering", look_at=avg_label)
        ])
        for n in range(8):
            self.show_sample(avg_label, 100)

    def show_sample(self, avg_label, n_dice, added_anims=[]):
        dice = VGroup(*(
            DieFace(random.randint(1, 6), fill_color=BLUE_E, dot_color=WHITE, stroke_width=1)
            for x in range(n_dice)
        ))
        dice.arrange_in_grid()
        dice.set_height(3)
        dice.next_to(avg_label, DOWN)
        dice.shift_onto_screen()

        self.play(
            ShowIncreasingSubsets(dice, run_time=1),
            UpdateFromFunc(avg_label[-1], lambda m: m.set_value(
                np.mean([die.value for die in dice])
            )),
            *added_anims
        )
        avg_label[-1].set_color(YELLOW)
        self.wait(2)
        self.play(
            FadeOut(dice, RIGHT),
            avg_label.animate.set_color(WHITE)
        )


class FormalStatementOfCLT(InteractiveScene):
    def construct(self):
        # Title
        title = Text("Formal statement of the Central Limit Theorem", font_size=60)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        title["Central Limit Theorem"].set_color(YELLOW)
        underline = Underline(title)
        underline.scale(1.2)
        underline.set_stroke(WHITE, [0, 3, 3, 0])
        self.add(title)
        self.play(ShowCreation(underline))
        self.wait()

        # Words
        VGroup(
            TexText(R"""
                Let $X_1, X_2, \dots, X_n, \dots$ be \\
                independent and identically distributed
                random variables.
            """)
        )