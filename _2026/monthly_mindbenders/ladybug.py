from manim_imports_ext import *


class Ladybug(InteractiveScene):
    # random_seed = 3 # ends with 6
    # random_seed = 6 # ends with 3
    # random_seed = 7 # ends with 10
    # random_seed = 8 # ends with 4
    # random_seed = 9 # ends with 9
    # random_seed = 10 # ends with 2
    # random_seed = 11 # ends with 11
    # random_seed = 12 # ends with 7
    # random_seed = 14 # ends with 1
    # random_seed = 15 # ends with 5
    # random_seed = 18 # ends with 2
    random_seed = 19 # ends with 3

    def construct(self):
        # Add clock
        clock = self.get_clock()
        clock_points = [tick.get_start() for tick in clock.ticks]
        clock_anim = cycle_animation(ClockPassesTime(clock, 12 * 60, 12))
        self.add(clock_anim)

        # Lady bug lands on it
        ladybug = SVGMobject("ladybug")
        ladybug.set_height(0.7)
        ladybug.set_color(GREY_A)
        ladybug.set_shading(0.5, 0.5, 0)
        circle = Dot(fill_color=RED_E, radius=0.36 * ladybug.get_height())
        circle.move_to(ladybug, DOWN)
        bug = Group(circle, Point(), ladybug)
        bug.move_to(clock.ticks[0].get_start())

        path = VMobject()
        path.start_new_path(ORIGIN)
        for n in range(5):
            step = rotate_vector(RIGHT, PI * random.random())
            path.add_line_to(path.get_end() + step)
        path.make_smooth()
        path.put_start_and_end_on(7 * LEFT, clock_points[0])

        self.play(MoveAlongPath(bug, path, run_time=3))
        self.play(clock.numbers[0].animate.set_color(RED))

        bug.shift(UP)

        # Run simulation
        curr_number = 0
        covered_numbers = {0}
        while len(covered_numbers) < 12:
            step = random.choice([+1, -1])
            next_number = curr_number + step
            path_arc = -step * TAU / 12
            arrow = Arrow(
                1.2 * clock_points[curr_number],
                1.2 * clock_points[next_number],
                buff=0,
                fill_color=YELLOW,
                path_arc=path_arc,
                thickness=5,
            )

            end_color = RED
            if len(covered_numbers) == 11 and next_number not in covered_numbers:
                end_color = TEAL
            self.play(
                VFadeInThenOut(arrow),
                bug.animate.move_to(clock_points[next_number]).set_anim_args(path_arc=path_arc, time_span=(0, 0.5)),
                clock.numbers[next_number].animate.set_color(end_color)
            )
            curr_number = next_number
            covered_numbers.add(curr_number)

    def get_clock(self, radius=2):
        # Add clock (Todo, add these modifications as options to the Clock class)
        clock = Clock()
        clock.set_height(2 * radius)
        for line in [clock.hour_hand, clock.minute_hand, *clock.ticks]:
            line.scale(0.75, about_point=line.get_start())

        numbers = VGroup(Integer(n) for n in [12, *range(1, 12)])
        for number, theta in zip(numbers, np.arange(0, TAU, TAU / 12)):
            number.move_to(0.75 * radius * rotate_vector(UP, -theta))

        clock.numbers = numbers
        clock.add(numbers)
        return clock


class Question(InteractiveScene):
    def construct(self):
        text = Text("""
            What is the probability that
            the last number painted is 6?
        """)
        text.to_edge(UP)
        self.play(Write(text))
        self.wait()

class Simulation(Ladybug):
    def construct(self):
        random.seed(8)

        num_trials = 1000
        trialTracker = ValueTracker(0)
        trialCounter = Integer(1, color = YELLOW, font_size = 35)
        trialCounter.add_updater(lambda c: c.set_value(trialTracker.get_value()))
        trialLabel = VGroup(
            Text("Trial: ", color = WHITE, font_size = 35),
            trialCounter
        )
        results = [int(0) for _ in range(11)]
        def get_hist():
            proportions = [x/(max(trialTracker.get_value(), 1)) for x in results]
            hist = BarChart(
                values = proportions,
                n_ticks = 0,
                bar_names = [str(x) for x in list(range(1, 12))],
                bar_colors = [BLUE]
            )
            hist.y_axis_labels.set_opacity(0)
            yAxisLabel = Tex(r"\%\ \text{occurrence}", font_size = 30).next_to(hist.y_axis, LEFT)
            hist.add(yAxisLabel)

            hist.center().scale(0.6).to_edge(DOWN, buff = 1.5).shift(LEFT*0.3)
            return hist
        hist = get_hist()
        self.add(hist, trialLabel)
        trialLabel.add_updater(lambda l: l.arrange(buff = 0.3).next_to(hist.x_axis, DOWN, buff = 0.6))
 
        clock = self.get_clock()
        clock_points = [tick.get_start() for tick in clock.ticks]
        ladybug = SVGMobject("ladybug")
        ladybug.set_height(0.7)
        ladybug.set_color(GREY_A)
        ladybug.set_shading(0.5, 0.5, 0)
        ladybug.set_z_index(100)
        circle = Dot(fill_color=RED_E, radius=0.36 * ladybug.get_height())
        circle.move_to(ladybug, DOWN)
        bug = Group(circle, Point(), ladybug)
        self.add(Group(clock, bug).scale(0.6).to_edge(UP, buff = 1).set_opacity(0))


        for trial in range(num_trials):
            trialTracker.increment_value(1)
            final_number = None
            covered_numbers = {0}
            curr_number = 0
            while len(covered_numbers) < 12:
                if random.random() > 0.5:
                    curr_number = (curr_number + 1) % 12
                else:
                    curr_number = (curr_number - 1) % 12
                covered_numbers.add(curr_number)
            final_number = curr_number
            if trial >= 2:
                bug.move_to(clock.ticks[final_number].get_start())
                clock.numbers.set_color(RED)
                clock.numbers[final_number].set_color(TEAL)
                Group(clock, bug).set_opacity(1)
                clock[0].set_fill(opacity = 0)

            results[final_number - 1] += 1
            new_hist = get_hist()
            self.play(hist.animate.become(new_hist))
        self.wait(3)

class ProbabilityCalculation(Ladybug):
    def construct(self):
        phase1 = Group()
        clock = self.get_clock()

        clock_points = [tick.get_start() for tick in clock.ticks]
        # clock_anim = cycle_animation(ClockPassesTime(clock, 12 * 60, 12))
        # self.add(clock_anim)

        # Lady bug lands on it
        ladybug = SVGMobject("ladybug")
        ladybug.set_height(0.7)
        ladybug.set_color(GREY_A)
        ladybug.set_shading(0.5, 0.5, 0)
        ladybug.set_z_index(100)
        circle = Dot(fill_color=RED_E, radius=0.36 * ladybug.get_height())
        circle.move_to(ladybug, DOWN)
        bug = Group(circle, Point(), ladybug)
        bug.move_to(clock.ticks[0].get_start())
        self.add(bug)
        clock.numbers[0].set_color(RED)
        phase1.add(clock, Point(), bug)
        ending_number = 6
        # ending_number = 3
        phase2a = phase1.copy()
        phase2a[-1].move_to(phase2a[0].ticks[ending_number - 1].get_start())
        phase2a[0].numbers[:ending_number].set_color(RED)
        phase2a[0].numbers[10:].set_color(RED)
        phase2b = phase1.copy()
        phase2b[-1].move_to(phase2b[0].ticks[ending_number + 1].get_start())
        phase2b[0].numbers[:2].set_color(RED)
        phase2b[0].numbers[ending_number + 1:].set_color(RED)
        Group(phase2a, phase2b).arrange(buff = 2)
        phase2 = Group(phase2a, Text("OR").set_color(TEAL), phase2b)
        phase3 = phase1.copy()
        phase3[-1].move_to(phase3[0].ticks[ending_number].get_start())
        phase3[0].numbers.set_color(RED)
        Group(phase1, phase2, phase3).scale(0.35).arrange(DOWN, buff = 1).center()
        self.add(phase1)
        arrow1 = Arrow(phase1, phase2[1]).set_color(YELLOW)
        self.play(
            AnimationGroup(
                ReplacementTransform(
                    Group(phase1.copy(), phase1.copy()),
                    Group(phase2a, phase2b)
                ),
                GrowArrow(arrow1),
                FadeIn(phase2[1])
            , lag_ratio = 0.2)
        , run_time = 3)
        self.wait(1)
        self.play(
            AnimationGroup(
                Flash(phase2a[0].numbers[ending_number - 1]),
                Flash(phase2b[0].numbers[ending_number + 1])
            )
        , run_time = 2, rate_func = there_and_back_with_pause)
        arrow2 = Arrow(phase2[1], phase3).set_color(YELLOW)
        phase3Copy = phase3.copy()
        self.play(
            AnimationGroup(
                ReplacementTransform(
                    Group(phase2a, phase2b).copy(),
                    Group(phase3, phase3Copy)
                ),
                GrowArrow(arrow2)
            , lag_ratio = 0.2)
        , run_time = 3)
        self.remove(phase3Copy)
        probabilityQuestionmark = Tex(
            # had to do mathrm for the text to render
            r"\mathrm{Probability}=\ ?", font_size = 20
        ).next_to(arrow2, RIGHT, buff = 0.1).shift(DOWN*0.25).set_color(PINK)
        self.play(Write(probabilityQuestionmark))
        self.wait(1)
        probability1 = Tex(
            # had to do mathrm for the text to render
            r"\mathrm{Probability}=\ 1", font_size = 20
        ).next_to(arrow1, RIGHT, buff = 0.1).shift(UP*0.25).set_color(PINK)
        self.play(Write(probability1))
        self.wait(4)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    FadeOut(phase1),
                    FadeOut(arrow1),
                    FadeOut(probability1),
                    FadeOut(phase2[0]),
                    FadeOut(phase2[1])
                ),
                AnimationGroup(
                    Group(phase2b, arrow2, phase3).animate.scale(1.4).arrange(DOWN),
                    probabilityQuestionmark.animate.scale(1.1).align_to(
                        probabilityQuestionmark, LEFT
                    ).set_y(0)
                )
            , lag_ratio = 0.4)
        , run_time = 3)
        self.wait(2)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    FadeOut(arrow2),
                    FadeOut(probabilityQuestionmark),
                    FadeOut(phase3)
                ),
                phase2b.animate.set_height(2.7).center()
            , lag_ratio = 0.4)
        )
        clock = phase2b[0]
        bug = phase2b[-1]
        self.play(CircleIndicate(clock.numbers[ending_number - 1]), run_time = 2, rate_func = there_and_back_with_pause)
        self.play(CircleIndicate(clock.numbers[ending_number]), run_time = 2, rate_func = there_and_back_with_pause)
        self.wait(2)
        clock_points = [tick.get_start() for tick in clock.ticks]
        arcTracker = ValueTracker(0)
        final_angle = -10*TAU/12
        def get_clockwise_arrow():
            return Arrow(
                clock_points[ending_number + 1],
                clock.get_center() + [
                    clock.get_width()*0.5*math.cos(arcTracker.get_value() - (4*TAU/12 if ending_number == 6 else TAU/12)),
                    clock.get_width()*0.5*math.sin(arcTracker.get_value() - (4*TAU/12 if ending_number == 6 else TAU/12)),
                    0
                ],
                buff=0,
                fill_color=YELLOW,
                path_arc=arcTracker.get_value(),
                thickness=5,
            )
        clockwiseArrow = always_redraw(get_clockwise_arrow)
        self.remove(bug)
        self.add(clockwiseArrow, Point(), bug)
        plus10 = Tex("+10").set_color(
            YELLOW
        ).next_to(
            clock_points[ending_number - 1],
            RIGHT if ending_number == 6 else UP,
            buff = 0.4
        )
        self.play(
            AnimationGroup(
                arcTracker.animate.set_value(final_angle),
                bug.animate.move_to(bug), # weird trick to get z index to work
                Write(plus10)
            )
        )
        counterclockwiseArrow = Arrow(
            clock_points[ending_number + 1]*0.6,
            clock_points[ending_number]*0.6,
            buff=0,
            fill_color=YELLOW,
            path_arc=-PI*0.5,
            thickness=3,
        )
        minus1 = Tex("-1", font_size = 25).set_color(
            YELLOW
        ).next_to(
            counterclockwiseArrow,
            UP if ending_number == 6 else LEFT,
            buff = 0.05
        )
        self.remove(bug)
        self.add(clockwiseArrow, Point(), bug)
        self.play(
            AnimationGroup(
                GrowArrow(counterclockwiseArrow),
                bug.animate.move_to(bug), # weird trick to get z index to work
                Write(minus1)
            )
        )
        self.wait(2)
        clockwiseArrow = get_clockwise_arrow()
        self.clear()
        self.add(clock, bug)
        self.play(
            AnimationGroup(
                FadeOut(clockwiseArrow),
                bug.animate.move_to(bug), # weird trick to get z index to work
                FadeOut(plus10),
                FadeOut(counterclockwiseArrow),
                FadeOut(minus1)
            )
        )

        clock_anim = cycle_animation(ClockPassesTime(clock, 12 * 60, 12))
        self.remove(bug)
        self.add(clock_anim, bug)
        curr_number = ending_number + 1
        covered_numbers = {0, 1, 2, 7, 8, 9, 10, 11}

        predetermined_steps = [
            +1, +1, -1, +1, -1, +1, +1, -1, -1, +1,
            +1, -1, +1, +1, +1, -1, +1, -1, +1, +1,
            -1, +1, +1, -1, +1, +1, +1, -1, +1, +1, # reached the 5
            -1, +1, -1, -1, +1, -1, +1, +1, -1, -1,
            -1, +1, +1, -1, +1, -1, +1, -1, -1, +1,
            -1, -1, -1, +1, -1, -1, -1, +1, -1, -1,
            +1, -1, # wanders off
            +1, -1, +1, +1, -1, -1, +1, +1, -1, +1, +1,
            -1, -1, +1, -1, +1, +1, +1, -1, -1, +1, -1,
            +1, +1, -1, +1, +1, -1, -1, +1, +1, +1, -1,
            +1, -1, +1, +1, -1, +1, +1, +1 # finally reaches the 6
        ]

        for step in predetermined_steps:
            next_number = (curr_number + step) % (len(clock_points))
            path_arc = -step * TAU / 12
            arrow = Arrow(
                1.2 * clock_points[curr_number],
                1.2 * clock_points[next_number],
                buff=0,
                fill_color=YELLOW,
                path_arc=path_arc,
                thickness=5,
            )

            end_color = RED
            if len(covered_numbers) == 11 and next_number not in covered_numbers:
                end_color = TEAL
            self.play(
                VFadeInThenOut(arrow),
                bug.animate.move_to(clock_points[next_number]).set_anim_args(path_arc=path_arc, time_span=(0, 0.5)),
                clock.numbers[next_number].animate.set_color(end_color)
            )
            curr_number = next_number
            covered_numbers.add(curr_number)
        self.wait(3)

        numberLine = NumberLine(
            [-1, 10, 1],
            include_numbers = True
        )
        for num in numberLine.numbers:
            num.scale(2.3).shift(DOWN*0.08)
        numberLine.numbers[0].set_color(TEAL)
        numberLine.numbers[1].set_color(RED)
        for tick in numberLine.ticks:
            tick.set_stroke(width = 5)
        axis = Line(numberLine.get_start(), numberLine.get_end())
        self.play(
            AnimationGroup(
                AnimationGroup(
                    ShrinkToCenter(clock[1]),
                    ShrinkToCenter(clock[2]),
                    ShrinkToCenter(clock[3])
                ),
                AnimationGroup(
                    ReplacementTransform(VGroup(*[list(clock.ticks[6:]) + list(clock.ticks[:6])]), numberLine[0]),
                    ReplacementTransform(clock[0], axis),
                    ReplacementTransform(VGroup(*[list(clock.numbers[6:]) + list(clock.numbers[:6])]), numberLine[1]),
                    bug.animate.next_to(numberLine[0][1], UP)
                )
            , lag_ratio = 0.2, run_time = 3.5)
        )
        self.wait(2)


        curr_number = 1
        covered_numbers = {1}
        predetermined_steps = [
            +1, +1, -1, +1, -1, +1, +1, -1, -1, +1,
            +1, -1, +1, +1, +1, -1, +1, -1, +1, +1,
            -1, +1, +1, -1, +1, +1, +1, -1, +1, +1
        ]

        for step in predetermined_steps:
            next_number = (curr_number + step) % (len(clock_points))
            path_arc = -step*PI*0.2
            arrow = Arrow(
                numberLine[0][curr_number].get_top() + UP*0.1,
                numberLine[0][next_number].get_top() + UP*0.1,
                buff=0,
                fill_color=YELLOW,
                path_arc=path_arc,
                thickness=5,
            )

            end_color = RED
            if len(covered_numbers) == 11 and next_number not in covered_numbers:
                end_color = TEAL
            self.play(
                VFadeInThenOut(arrow),
                bug.animate.next_to(numberLine[0][next_number].get_top(), UP, buff = 0.3).set_anim_args(path_arc=path_arc, time_span=(0, 0.5)),
                numberLine.numbers[next_number].animate.set_color(end_color)
            )
            curr_number = next_number
            covered_numbers.add(curr_number)