from manim_imports_ext import *
from _2025.blocks_and_grover.qc_supplements import *
from _2025.blocks_and_grover.state_vectors import DisectAQuantumComputer


class Quiz(InteractiveScene):
    def construct(self):
        # Set up terms
        choices = VGroup(
            TexText(R"A) $\mathcal{O}\big(\sqrt{N}\big)$"),
            TexText(R"B) $\mathcal{O}\big(\log(N)\big)$"),
            TexText(R"C) $\mathcal{O}\big(\log(\log(N))\big)$"),
            TexText(R"D) $\mathcal{O}(1)$"),
        )
        choices.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        choices.to_edge(LEFT, buff=1.0)

        covers = VGroup(
            SurroundingRectangle(choice[2:], buff=SMALL_BUFF)
            for choice in choices
        )
        covers.set_fill(GREY_D, 1)
        covers.set_stroke(WHITE, 1)
        for cover, choice in zip(covers, choices):
            choice[2:].set_fill(opacity=0)
            cover.set_width(choices.get_width(), about_edge=LEFT, stretch=True)
            cover.align_to(covers, LEFT)
            cover.save_state()
            cover.stretch(0, 0, about_edge=LEFT)

        self.play(
            LaggedStartMap(Restore, covers, lag_ratio=0.25),
            LaggedStartMap(FadeIn, choices, lag_ratio=0.25),
        )
        self.wait()

        # Reference mostly wrong answers
        pis = Randolph().get_grid(6, 6, buff=2.0)
        pis.set_height(7)
        pis.to_edge(RIGHT)
        pis.sort(lambda p: np.dot(p, DR))

        symbol_height = 0.35
        symbols = [
            Exmark().set_color(RED).set_height(symbol_height),
            Checkmark().set_color(GREEN).set_height(symbol_height),
        ]
        all_symbols = VGroup()
        for pi in pis:
            pi.body.set_color(interpolate_color(BLUE_E, BLUE_C, random.random()))
            pi.change_mode("pondering")
            correct = random.random() < 0.2
            symbol = symbols[correct].copy()
            symbol.next_to(pi, UR, buff=0)
            all_symbols.add(symbol)
            pi.generate_target()
            pi.target.change_mode(["sad", "hooray"][correct])

        self.play(LaggedStartMap(FadeIn, pis, run_time=2))
        self.play(
            LaggedStartMap(MoveToTarget, pis, lag_ratio=0.01),
            LaggedStartMap(Write, all_symbols, lag_ratio=0.01),
        )
        pis.shuffle()
        self.play(LaggedStartMap(Blink, pis[::4]))
        self.wait()
        self.play(
            FadeOut(pis, lag_ratio=1e-3),
            FadeOut(all_symbols, lag_ratio=1e-3),
        )

        # Show question
        frame = self.frame

        question = VGroup(
            get_quantum_computer_symbol(),
            Clock(),
            Tex(R"?"),
        )
        for mob in question:
            mob.set_height(1.5)
        question.arrange(RIGHT, buff=0.5)
        question.set_width(4)
        question[2].scale(0.7)
        question.next_to(choices, UP, aligned_edge=LEFT)

        clock = question[1]
        cycle_animation(ClockPassesTime(clock, hours_passed=12, run_time=24))
        self.play(
            VFadeIn(question, suspend_mobject_updating=True, lag_ratio=0.01),
            VGroup(choices, covers).animate.shift(DOWN).set_anim_args(run_time=2),
            frame.animate.match_x(choices).set_anim_args(run_time=2),
        )
        self.add(choices, Point(), covers)
        choices.set_opacity(1)
        self.play(
            LaggedStart(
                (cover.animate.stretch(0, 0, about_edge=RIGHT).set_opacity(0)
                for cover in covers),
                lag_ratio=0.25,
            ),
        )
        self.wait(16)

        # Show distribution
        question.clear_updaters()
        dists = [
            np.array([18, 20, 8, 54], dtype=float),
            np.array([51, 55, 37, 65], dtype=float),  # Stanford
            np.array([17, 25, 5, 39], dtype=float),   # IMO, IIRC
        ]
        for dist in dists:
            dist[:] = dist / dist.sum()

        max_bar_width = 5.0
        prob_bar_group = VGroup()
        for dist in dists:
            prob_bars = VGroup()
            for choice, prob in zip(choices, dist):
                bar = Rectangle(width=prob * max_bar_width, height=0.35)
                bar.next_to(choice, LEFT)
                bar.set_fill(interpolate_color(BLUE_D, GREEN, prob * 1.5), 1)
                bar.set_stroke(WHITE, 1)
                prob_bars.add(bar)
            prob_bar_group.add(prob_bars)

        prob_bars = prob_bar_group[0].copy()

        prob_labels = VGroup()
        for bar in prob_bars:
            label = Integer(100, font_size=36, unit=R"\%")
            label.bar = bar
            label.add_updater(lambda m: m.set_value(np.round(100 * m.bar.get_width() / max_bar_width)))
            label.add_updater(lambda m: m.next_to(m.bar, LEFT))
            prob_labels.add(label)

        self.play(
            LaggedStart(
                (GrowFromPoint(bar, bar.get_right())
                for bar in prob_bars),
                lag_ratio=0.2,
            ),
            VFadeIn(prob_labels)
        )
        self.wait()
        for index in [1, 2, 0]:
            self.play(Transform(prob_bars, prob_bar_group[index]))
            self.wait()

        # Go through each answer
        covers = VGroup(
            SurroundingRectangle(VGroup(bar, label, choice))
            for bar, label, choice in zip(prob_bars, prob_labels, choices)
        )
        covers.set_stroke(width=0)
        covers.set_fill(BLACK, 0.8)

        self.add(Point())
        self.play(FadeIn(covers[:3]))
        self.wait()
        self.play(
            FadeOut(covers[1]),
            FadeIn(covers[3])
        )
        self.wait()
        self.play(
            FadeOut(covers[0]),
            FadeIn(covers[1])
        )
        self.wait()

        # Add two additional answers


class ShowOptionGraphs(InteractiveScene):
    def construct(self):
        # Axes
        x_max = 15
        axes = Axes((-1, x_max), (-1, x_max))
        axes.set_height(7)
        self.add(axes)

        # Add graphs
        graphs = VGroup(
            axes.get_graph(func, x_range=(0.01, x_max))
            for func in [
                lambda n: n,
                lambda n: math.sqrt(n),
                lambda n: 0.8 * math.log(n + 1),
                lambda n: math.log(math.log(n + 1) + 1),
                lambda n: 1,
            ]
        )
        graphs.set_submobject_colors_by_gradient(YELLOW, ORANGE, RED, RED_E, BLUE)
        labels = VGroup(
            Tex(sym, font_size=30).match_color(graph).next_to(graph.get_end(), RIGHT, SMALL_BUFF)
            for graph, sym in zip(graphs, [
                R"\mathcal{O}\left(N\right)",
                R"\mathcal{O}\left(\sqrt{N}\right)",
                R"\mathcal{O}\left(\log(N)\right)",
                R"\mathcal{O}\left(\log(\log(N))\right)",
                R"\mathcal{O}\left(1\right)",
            ])
        )
        labels[-1].shift(2 * SMALL_BUFF * DOWN)

        for graph, label in zip(graphs, labels):
            vect = label.get_center() - graph.get_end()
            self.play(
                ShowCreation(graph),
                VFadeIn(label),
                UpdateFromFunc(label, lambda m: m.move_to(graph.get_end() + vect)),
            )
        self.wait()


class NeedleInAHaystackProblem(InteractiveScene):
    def construct(self):
        # Set up terms
        shown_numbers = list(range(20))
        number_strs = list(map(str, shown_numbers))
        number_set = Tex("".join([
            R"\{",
            *[str(n) + "," for n in shown_numbers],
            R"\dots N - 1"
            R"\}",
        ]), isolate=number_strs)
        number_mobs = VGroup(number_set[n_str][0] for n_str in number_strs)
        number_set.set_width(FRAME_WIDTH - 1)
        number_set.to_edge(UP)

        machine = get_blackbox_machine()
        machine.set_z_index(2)

        self.play(FadeIn(number_set, lag_ratio=0.01))
        self.wait()

        # Show mystery machine
        q_marks = Tex(R"???", font_size=90)
        q_marks.space_out_submobjects(1.2)
        q_marks.next_to(machine, UP)

        self.play(
            FadeIn(machine, scale=2),
            Write(q_marks)
        )
        self.play(LaggedStartMap(FadeOut, q_marks, shift=0.25 * DOWN, lag_ratio=0.1))
        self.wait()

        # Plug in key value
        key_number = 12
        key_input = number_mobs[key_number]
        key_icon = SVGMobject("key").rotate(135 * DEG)
        key_icon.set_fill(YELLOW)
        key_icon.match_width(key_input)
        key_icon.next_to(key_input, DOWN, SMALL_BUFF)

        self.play(
            FlashAround(key_input),
            key_input.animate.set_color(YELLOW),
            FadeIn(key_icon, 0.25 * DOWN)
        )
        self.wait()

        in_mob = key_input.copy().set_color(YELLOW)
        self.play(in_mob.animate.scale(1.5).next_to(machine, LEFT, MED_LARGE_BUFF))
        self.play(self.evaluation_animation(in_mob, machine, True))
        self.wait()

        # Plug in other values
        other_inputs = number_mobs.copy()
        other_inputs.remove(other_inputs[key_number])
        other_inputs.add(number_set["N - 1"][0].copy())
        other_inputs.generate_target()
        other_inputs.target.arrange_in_grid(n_cols=3, buff=MED_SMALL_BUFF)
        other_inputs.target.next_to(machine, LEFT, LARGE_BUFF)

        self.play(
            FadeOut(in_mob, DOWN),
            FadeOut(machine.output_group, DOWN),
            MoveToTarget(other_inputs, lag_ratio=0.01),
        )
        machine.output_group.clear()
        self.play(LaggedStart(
            (self.evaluation_animation(mob, machine)
            for mob in other_inputs),
            lag_ratio=0.2,
        ))
        self.wait()
        self.play(
            FadeOut(other_inputs, shift=0.25 * DOWN, lag_ratio=0.01),
            FadeOut(machine.output_group, 0.25 * DOWN),
        )
        machine.output_group.clear()
        self.wait()

        # Show innards
        innards = Code("""
            def f(n):
                return (n == 12)
        """, font_size=16)
        innards[8:].shift(0.5 * RIGHT)
        innards.move_to(machine).shift(0.25 * LEFT)

        self.play(
            machine.animate.set_fill(opacity=0),
            FadeIn(innards)
        )
        self.wait()
        self.play(
            FadeOut(innards),
            machine.animate.set_fill(opacity=1),
            FadeIn(q_marks, shift=0.25 * UP, lag_ratio=0.25)
        )
        self.play(FadeOut(q_marks))
        self.wait()

        # Guess and check
        last_group = VGroup()
        for n, in_mob in enumerate(number_mobs[:key_number + 1].copy()):
            self.play(
                FadeOut(last_group),
                in_mob.animate.scale(1.5).next_to(machine, LEFT, MED_LARGE_BUFF)
            )
            output = (n == key_number)
            self.play(self.evaluation_animation(in_mob, machine, output))
            last_group = VGroup(in_mob, machine.output_group[0])
            machine.output_group.clear()

        self.wait()
        self.play(FadeOut(last_group))

        # Put into a superposition
        pile = number_mobs.copy()
        for mob in pile:
            mob.scale(0.5)
        superposition = Superposition(pile)
        superposition.set_offset_multiple(0)
        superposition.set_glow_opacity(0)
        superposition.update()

        superposition.generate_target()
        for piece in superposition.pieces:
            piece.scale(2)

        for point in superposition.target[2]:
            point.next_to(machine, LEFT, buff=2.0)
            point.scale(0.5)
            point.shift(np.random.normal(0, 0.5, 3))

        superposition.target[2].arrange(DOWN, buff=0.25).next_to(machine, LEFT, buff=1.5)

        superposition.target.set_offset_multiple(0.1)
        superposition.target.set_glow_opacity(0.1)

        self.play(
            MoveToTarget(superposition, run_time=2),
        )

        # Pass superposition through the function
        answers = VGroup(
            Text("True").set_color(GREEN) if n == key_number else Text("False").set_color(RED)
            for n, piece in enumerate(superposition.pieces)
        )
        answers.match_height(superposition.pieces[0])
        answers.arrange_to_fit_height(superposition.get_height())
        answers.next_to(machine, RIGHT, buff=1.5)
        answers.shuffle()
        answer_superposition = Superposition(answers, glow_color=RED)
        answer_superposition.set_offset_multiple(0)
        answer_superposition.set_glow_opacity(0)
        answer_superposition.update()

        superposition.set_z_index(2)
        self.play(LaggedStart(
            LaggedStart(
                (FadeOutToPoint(glow.copy(), machine.get_left() + 0.5 * RIGHT)
                for glow in superposition.glows),
                lag_ratio=0.1,
            ),
            LaggedStart(
                (FadeInFromPoint(answer, machine.get_right() + 0.5 * LEFT)
                for answer in answer_superposition.pieces),
                lag_ratio=0.05,
            ),
            lag_ratio=0.5
        ))
        self.play(answer_superposition.animate.set_offset_multiple(0.025).set_glow_opacity(1e-2))
        self.wait(10)

    def evaluation_animation(self, input_mob, machine, output=False, run_time=1.0):
        if output:
            out_mob = Text("True").set_color(GREEN)
        else:
            out_mob = Text("False").set_color(RED)
        out_mob.scale(1.25)
        out_mob.next_to(machine, RIGHT, MED_LARGE_BUFF)

        moving_input = input_mob.copy()
        input_mob.set_opacity(0.25)

        machine.output_group.add(out_mob)
        in_point = interpolate(machine.get_left(), machine.get_center(), 0.5)

        return AnimationGroup(
            FadeOutToPoint(moving_input, in_point, time_span=(0, 0.75 * run_time)),
            FadeInFromPoint(out_mob, machine.get_left(), time_span=(0.25 * run_time, run_time)),
        )


class LargeGuessAndCheck(InteractiveScene):
    key_value = 42
    wait_time_per_mob = 0.1
    row_size = 10

    def construct(self):
        # Create grid of values and machine
        N = self.row_size
        grid = VGroup(Integer(n) for n in range(int(self.row_size**2)))
        grid.arrange_in_grid(buff=0.75, fill_rows_first=False)
        grid.set_height(FRAME_HEIGHT - 1)

        output = Text("False").set_color(RED)
        output.match_height(grid[0])

        machine = get_blackbox_machine(height=1.5 * grid[0].get_height())
        machine.next_to(output, LEFT, SMALL_BUFF)
        machine_group = VGroup(machine, output)
        extra_width = machine_group.get_width()
        grid.shift(0.5 * extra_width * RIGHT)

        self.add(grid)

        # Sweep through
        self.add(machine_group)
        for n, mob in enumerate(grid):
            if n % self.row_size == 0:
                grid[n:n + self.row_size].shift(extra_width * LEFT)
            machine_group.next_to(mob, RIGHT, buff=0.5 * mob.get_width())
            if n != self.key_value:
                self.play(grid[n].animate.set_opacity(0.5), run_time=self.wait_time_per_mob)
                continue

            new_output = Text("True").set_color(GREEN)
            new_output.replace(machine_group[1])
            machine_group.replace_submobject(1, new_output)
            break

        rect = SurroundingRectangle(grid[n])
        self.play(ShowCreation(rect), grid[n].animate.set_color(YELLOW))
        self.wait()


class GuessAndCheckEarlyGet(LargeGuessAndCheck):
    key_value = 12


class GuessAndCheckLateGet(LargeGuessAndCheck):
    key_value = 92


class GuessAndCheckMidGet(LargeGuessAndCheck):
    key_value = 53


class BigGuessAndCheck(LargeGuessAndCheck):
    key_value = 573
    wait_time_per_mob = 0.01
    row_size = 30


class WriteClassicalBigO(InteractiveScene):
    def construct(self):
        # Background
        self.add(FullScreenRectangle().set_fill(GREY_E))

        # Terms
        avg = TexText(R"Avg: $\displaystyle \frac{1}{2} N$")
        arrow = Tex(R"\longrightarrow")
        big_o = Tex(R"\mathcal{O}(N)")
        group = VGroup(avg, arrow, big_o)
        group.arrange(RIGHT, SMALL_BUFF)
        group.scale(1.25)
        group.to_edge(UP, buff=MED_SMALL_BUFF)

        avg.save_state()
        avg.set_x(0)

        self.play(Write(avg))
        self.wait()
        self.play(Restore(avg))
        self.play(
            Write(arrow),
            TransformFromCopy(avg[-1], big_o[2], path_arc=45 * DEG),
            Write(big_o[:2]),
            Write(big_o[3:]),
        )
        self.wait()


class ReferenceNeedleInAHaystack(InteractiveScene):
    key = 61

    def construct(self):
        # Show that grid of 100 values, with arrows to exes or checks
        N = 10
        grid = VGroup(Integer(n) for n in range(int(N * N)))
        grid.arrange_in_grid(fill_rows_first=False, h_buff=1.35, v_buff=0.75)
        grid.set_height(FRAME_HEIGHT - 1)

        self.add(grid)

        # Show marks
        key = self.key
        symbols = VGroup()
        for n in range(N * N):
            if n == key:
                symbol = Checkmark().set_color(GREEN)
            else:
                symbol = Exmark().set_color(RED)
            symbol.set_height(grid[0].get_height())
            symbol.next_to(grid[n], RIGHT, SMALL_BUFF)
            symbols.add(symbol)

        key_group = VGroup(grid[key], symbols[key])
        symbols.shuffle()

        self.play(LaggedStartMap(FadeIn, symbols, shift=0.25 * RIGHT, lag_ratio=0.05))
        self.wait()

        # Show key
        rect = SurroundingRectangle(key_group)
        rect.set_stroke(GREEN, 2)
        fader = FullScreenFadeRectangle(fill_opacity=0.5)
        self.add(fader, key_group, rect)
        self.play(
            FadeIn(fader),
            ShowCreation(rect),
        )


class ReferenceNeedleInAHaystack2(ReferenceNeedleInAHaystack):
    key = 31


class SuperpositionAsParallelization(InteractiveScene):
    def construct(self):
        # Set up
        classical, quantum = symbols = VGroup(
            get_classical_computer_symbol(),
            get_quantum_computer_symbol(),
        )
        for symbol, vect in zip(symbols, [LEFT, RIGHT]):
            symbol.set_height(1)
            symbol.move_to(vect * FRAME_WIDTH / 4)
            symbol.to_edge(UP, buff=MED_SMALL_BUFF)

        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        v_line.set_stroke(WHITE, 1)

        self.add(symbols)
        self.add(v_line)

        # Bit string
        boxes = Square().get_grid(1, 4, buff=0)
        boxes.set_height(0.5)
        boxes.match_x(classical)
        boxes.set_stroke(WHITE, 2)

        bit_mobs = VGroup(Integer(0).move_to(box) for box in boxes)

        def match_value(bit_mobs, value):
            bit_str = bin(int(value))[2:].zfill(4)
            for bit, mob in zip(bit_str, bit_mobs):
                mob.set_value(int(bit))
            return bit_mobs

        self.play(
            Write(boxes),
            Write(bit_mobs)
        )
        value_tracker = ValueTracker(0)
        self.play(
            value_tracker.animate.set_value(12),
            UpdateFromFunc(bit_mobs, lambda m: match_value(m, value_tracker.get_value())),
            run_time=3,
            rate_func=linear
        )
        self.wait()

        # Superposition
        bit_strings = VGroup()
        for bits in it.product(* 4 * [[0, 1]]):
            bit_string = VGroup(Integer(b) for b in bits)
            bit_string.arrange(RIGHT, buff=SMALL_BUFF)
            bit_strings.add(bit_string)
        bit_strings.arrange(DOWN)
        bit_strings.set_height(5.5)
        bit_strings.next_to(quantum, DOWN, MED_LARGE_BUFF)
        bit_strings.set_fill(opacity=0.75)

        superposition = Superposition(bit_strings)
        superposition.set_offset_multiple(0)
        superposition.set_glow_opacity(0)
        superposition.update()

        superposition_name = TexText(R"``Superposition''")
        superposition_name.set_color(TEAL)
        superposition_name.next_to(superposition, RIGHT, aligned_edge=UP).shift(LEFT)

        self.play(
            LaggedStart(
                (TransformFromCopy(bit_mobs, bit_string)
                for bit_string in bit_strings),
                lag_ratio=0.05,
            )
        )
        self.play(
            superposition.animate.set_offset_multiple(0.1).set_glow_opacity(0.05).shift(1.5 * LEFT),
            Write(superposition_name, run_time=1)
        )
        self.wait(15)

        # Show parallelization lines
        mini_classical = VGroup(
            classical.copy().set_height(0.25).move_to(point).match_x(quantum)
            for point in superposition.center_points
        )
        lines = VGroup(
            VGroup(
                Line(mc.get_left() + 1.1 * LEFT, mc.get_left(), buff=0.1),
                Line(mc.get_right(), mc.get_right() + 1.1 * RIGHT, buff=0.1),
            )
            for mc in mini_classical
        )
        lines.set_stroke(GREY, 2)

        outputs = VGroup(
            Integer(int(n == 12), font_size=24).next_to(line, RIGHT)
            for n, line in enumerate(lines)
        )

        self.play(
            superposition.animate.set_offset_multiple(0.025),
            FadeOut(superposition_name),
            LaggedStart(
                (TransformFromCopy(classical, mc)
                for mc in mini_classical),
                lag_ratio=0.05,
            ),
            LaggedStartMap(ShowCreation, lines),
        )
        self.wait()
        self.play(LaggedStart(
            (TransformFromCopy(piece, output)
            for piece, output in zip(superposition.pieces, outputs)),
            lag_ratio=0.01,
            run_time=3
        ))
        self.wait(15)


class ListTwoMisconceptions(TeacherStudentsScene):
    def construct(self):
        # Add title and two misconceptions
        pass


class LogTable(InteractiveScene):
    def construct(self):
        # Set up table
        n_samples = 9
        line_width = 8
        line_buff = 0.75

        h_line = Line(LEFT, RIGHT).set_width(line_width)
        h_lines = h_line.get_grid(n_samples, 1, buff=line_buff)
        h_lines.set_stroke(WHITE, 1)
        h_lines.shift(0.25 * DOWN)

        v_line = Line(UP, DOWN).set_height(7)
        v_line.set_stroke(WHITE, 2)

        N_title, logN_title = titles = VGroup(
            Tex("N"),
            Tex(R"\log_2(N)"),
        )
        titles.scale(1.25)
        for sign, title in zip([-1, 1], titles):
            title.set_x(sign * 2)
            title.to_edge(UP)

        self.add(h_lines, v_line)
        self.add(titles)

        # Fill with numbers
        N_values = VGroup()
        logN_values = VGroup()

        for n, line in enumerate(h_lines[1:]):
            N = 10**(n + 1)
            N_value = Integer(N)
            logN_value = DecimalNumber(np.log2(N))
            N_value.next_to(line, UP, SMALL_BUFF).match_x(N_title)
            logN_value.next_to(line, UP, SMALL_BUFF).match_x(logN_title)
            N_value.align_to(logN_value, UP)
            N_values.add(N_value)
            logN_values.add(logN_value)

        self.add(N_values[0], logN_values[0])
        for index in range(len(N_values) - 1):
            self.play(
                TransformMatchingShapes(N_values[index].copy(), N_values[index + 1]),
                FadeIn(logN_values[index + 1], shift=0.5 * DOWN),
                run_time=1
            )
        self.add(N_values)
        self.add(logN_values)

        # Show addition
        all_arrows = VGroup()
        for line in h_lines[1:-1]:
            arrow = Arrow(
                logN_values[0].get_right(),
                logN_values[1].get_right(),
                path_arc=-PI,
                buff=0
            )
            arrow.scale(0.8)
            arrow.next_to(line, RIGHT, SMALL_BUFF)
            plus_label = Tex(R"+\log_2(10)", font_size=24)
            plus_label.set_color(BLUE)
            plus_label.next_to(arrow, RIGHT, SMALL_BUFF)

            all_arrows.add(VGroup(arrow, plus_label))

        self.play(LaggedStartMap(FadeIn, all_arrows, lag_ratio=0.25), run_time=3)


class SecondMisconception(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Misconception #2", font_size=90)
        title.to_edge(UP, buff=LARGE_BUFF)
        title.add(Underline(title))
        title.set_color(BLUE)

        words = Text(
            "Quantum Computers would make\n" + \
            "everything exponentially faster",
            font_size=60
        )
        words.next_to(title, DOWN, MED_LARGE_BUFF)
        red_cross = Cross(words["everything"])
        red_cross.set_stroke(RED, [0, 8, 8, 8, 0])
        new_words = Text("some very\nspecial problems", alignment="LEFT")
        new_words.set_color(RED)
        new_words.next_to(red_cross, DOWN, aligned_edge=LEFT)

        self.add(title)
        self.play(Write(words, run_time=2))
        self.wait()
        self.play(LaggedStart(
            ShowCreation(red_cross),
            FadeIn(new_words, lag_ratio=0.1),
            FadeOut(title),
            lag_ratio=0.35
        ))
        self.wait()

        # Show factoring number
        factors = VGroup(Integer(314159), Integer(271829))
        factors.arrange(RIGHT, buff=LARGE_BUFF)
        product = Integer(factors[0].get_value() * factors[1].get_value())
        product.next_to(factors, UP, LARGE_BUFF)
        lines = VGroup(
            Line(product.get_bottom(), factor.get_top(), buff=0.2)
            for factor in factors
        )
        lines.set_submobject_colors_by_gradient(BLUE, GREEN)
        product.set_color(TEAL)
        factors.set_submobject_colors_by_gradient(BLUE, GREEN)
        times = Tex(R"\times")
        times.move_to(factors)

        factor_group = VGroup(product, lines, factors, times)
        factor_group.next_to(words, DOWN, LARGE_BUFF, aligned_edge=RIGHT)

        self.play(FadeIn(product))
        self.play(
            LaggedStartMap(ShowCreation, lines, lag_ratio=0.25),
            LaggedStart(
                (TransformFromCopy(product, factor)
                for factor in factors),
                lag_ratio=0.25
            ),
            Write(times, time_span=(0.5, 1.5)),
        )
        self.wait()


class GroverTimeline(InteractiveScene):
    def construct(self):
        # Test
        timeline = NumberLine(
            (1990, 2025, 1),
            big_tick_spacing=5,
            width=FRAME_WIDTH - 1
        )
        timeline.set_y(-2)
        timeline.add_numbers(
            range(1990, 2030, 5),
            group_with_commas=False,
            font_size=24,
        )
        self.add(timeline)

        # BBBV
        bbbv_statement = VGroup(
            Text("Quantum Search", font_size=36),
            Tex(R"\ge", font_size=42),
            Tex(R"\mathcal{O}(\sqrt{N})", font_size=36),
        )
        bbbv_statement.arrange(RIGHT, SMALL_BUFF)
        bbbv_statement.to_corner(UL)
        bbbv_statement.set_color(RED)

        bbbv_attribution = TexText("BBBV$^*$ Theorem (1994)", font_size=36)
        bbbv_attribution.next_to(bbbv_statement, DOWN, aligned_edge=LEFT)
        bbbv_attribution.set_color(RED_B)

        bbbv_attribution.to_corner(UL)
        bbbv_statement.next_to(bbbv_attribution, DOWN, MED_SMALL_BUFF)
        bbbv_statement.set_x(-3.5)

        footnote = Text("*Bennett, Bernstein, Brassard, Vazirani", font_size=24)
        footnote.set_color(GREY_C)
        footnote.to_corner(DL, buff=MED_SMALL_BUFF)

        bbbv_dots = VGroup(
            Dot(timeline.n2p(1994)),
            Dot().next_to(bbbv_attribution, LEFT, SMALL_BUFF),
        )
        bbbv_dots.set_color(RED_B)
        arc = -45 * DEG
        bbbv_line = Line(
            bbbv_dots[0].get_center(),
            bbbv_dots[1].get_center(),
            path_arc=arc,
        )
        bbbv_line.set_stroke(RED_B)

        self.play(
            GrowFromCenter(bbbv_dots[0]),
            TransformFromCopy(*bbbv_dots, path_arc=arc),
            ShowCreation(bbbv_line),
            FadeIn(bbbv_attribution, UP),
            FadeIn(footnote),
        )
        self.play(
            FadeIn(bbbv_statement, 0.5 * DOWN)
        )
        self.wait()

        # Lov Grover
        grover_name = TexText("Grover's Algorithm (1996)", font_size=36)
        grover_name.next_to(bbbv_statement, DOWN, buff=0.75)
        grover_name.set_color(BLUE)
        grover_name.shift(0.5 * LEFT)

        grover_statement = bbbv_statement.copy()
        eq = Tex(R"=").replace(grover_statement[1], dim_to_match=0)
        grover_statement[1].become(eq)
        grover_statement.set_color(BLUE_D)
        grover_statement.next_to(grover_name, DOWN)
        grover_statement.match_x(bbbv_statement)

        grover_dots = VGroup(
            Dot(timeline.n2p(1996)),
            Dot().next_to(grover_name, LEFT, SMALL_BUFF),
        )
        arc = -35 * DEG
        grover_line = Line(
            grover_dots[0].get_center(),
            grover_dots[1].get_center(),
            path_arc=arc,
        )
        VGroup(grover_dots, grover_line).set_color(BLUE_B)

        self.play(TransformFromCopy(bbbv_dots[0], grover_dots[0], path_arc=-PI))
        self.play(
            TransformFromCopy(*grover_dots, path_arc=arc),
            ShowCreation(grover_line),
            FadeIn(grover_name, UP),
        )
        self.wait()
        self.play(
            TransformFromCopy(bbbv_statement, grover_statement),
        )
        self.wait()

        # Show examples
        examples = VGroup()
        for n in [6, 12]:
            n_eq = VGroup(Tex(R"N = "), Integer(10**n))
            n_eq.arrange(RIGHT, SMALL_BUFF)
            n_eq.to_corner(UR, buff=MED_LARGE_BUFF)
            steps = VGroup(Tex(R"\sim"), Integer(10**(n / 2)), Dot().set_fill(opacity=0), Text("Steps"))
            steps.arrange(RIGHT, buff=0.05)
            steps.next_to(n_eq, DOWN, LARGE_BUFF)

            arrow = Arrow(n_eq, steps, buff=0.15)

            examples.add(VGroup(n_eq, arrow, steps))

        sqrt_N = grover_statement[2][2:4]

        for n, example in enumerate(examples):
            if n == 0:
                self.play(
                    CountInFrom(example[0][1], 0),
                    VFadeIn(example[0]),
                )
            elif n == 1:
                self.play(
                    ReplacementTransform(examples[0][0], examples[1][0]),
                    FadeOut(examples[0][1:])
                )
            sqrt_rect = SurroundingRectangle(sqrt_N, buff=SMALL_BUFF)
            sqrt_rect.set_stroke(WHITE, 2)
            self.play(ShowCreation(sqrt_rect))
            self.play(
                GrowArrow(example[1]),
                sqrt_rect.animate.surround(example[2][1]).set_stroke(opacity=0),
                FadeTransform(sqrt_N.copy(), example[2][1]),
                FadeIn(example[2][0]),
                FadeIn(example[2][-1]),
            )
            self.add(example)
            self.wait()

        # Show the π / 4
        big_O = grover_statement[2]
        runtime = Tex(R"\left\lceil \frac{\pi}{4} \right\rceil \sqrt{N}", font_size=36)
        runtime.move_to(big_O, LEFT)
        runtime.set_color(BLUE_B)

        rect = SurroundingRectangle(big_O)
        rect.set_stroke(WHITE, 2)

        self.play(ShowCreation(rect))
        self.wait()
        self.play(
            VGroup(rect, big_O).animate.shift(DOWN),
            FadeIn(runtime, scale=2)
        )
        self.wait()
        self.play(rect.animate.surround(runtime[1], buff=0.05))


class NPProblemExamples(InteractiveScene):
    def construct(self):
        # Examples
        example_images = Group(
            self.get_sudoku(),
            ImageMobject("US_color_graph"),
            Square().set_opacity(0),
        )
        for img in example_images:
            img.set_height(2)
        example_images.arrange_in_grid(2, 2, buff=1.5)
        example_images[2].match_x(example_images[:2])

        example_names = VGroup(
            Text("Sudoku"),
            Text("Graph Coloring"),
            Text("Reversing Cryptographic\nHash Functions"),
        )
        examples = Group()
        for name, img in zip(example_names, example_images):
            name.scale(0.65)
            name.next_to(img, DOWN)
            examples.add(Group(img, name))
        examples.move_to(2 * LEFT)

        self.play(LaggedStartMap(FadeIn, examples, lag_ratio=0.5))

        # Name them
        big_rect = SurroundingRectangle(examples, buff=0.35)
        big_rect.set_stroke(BLUE, 3)
        big_rect.round_corners(radius=0.5)
        name = Text("NP Problems", font_size=60)
        name.next_to(big_rect, buff=MED_LARGE_BUFF)

        self.play(LaggedStart(
            ShowCreation(big_rect),
            Write(name),
            # self.frame.animate.set_x(2),
            lag_ratio=0.2
        ))
        self.wait()

    def get_sudoku(self):
        sudoku = SVGMobject("sudoku_example")
        small_width = sudoku[0].get_width()
        for part in sudoku.submobjects:
            if len(part.get_anchors()) == 5:
                part.set_fill(opacity=0)
                part.set_stroke(WHITE, 1, 0.5)
                if part.get_width() > 2 * small_width:
                    part.set_stroke(WHITE, 2, 1)
            else:
                part.set_fill(WHITE, 1)

        return sudoku


class ShowSha256(InteractiveScene):
    def construct(self):
        # Test
        import hashlib

        input_int = Integer(0, min_total_width=8, group_with_commas=False)
        output_text = Text("")
        lhs = VGroup(Text(R"SHA256("), input_int, Text(")"))
        lhs.arrange(RIGHT, buff=SMALL_BUFF),
        equation = VGroup(lhs, Tex("=").rotate(90 * DEG), output_text)
        equation.arrange(DOWN, buff=MED_SMALL_BUFF)

        def update_hash(text_mob):
            input_bytes = str(input_int.get_value()).encode()
            sha256_hash = hashlib.new('sha256')
            sha256_hash.update(input_bytes)
            hash_hex = sha256_hash.hexdigest()

            new_text = "\n".join(
                "".join(row)
                for row in np.array(list(hash_hex)).reshape((4, 16))
            )

            new_text = Text(new_text, font="Consolas")
            new_text.move_to(text_mob, UP)
            text_mob.set_submobjects(new_text.submobjects)

        output_text.add_updater(update_hash)

        self.add(equation)
        self.play(
            ChangeDecimalToValue(input_int, 2400, rate_func=linear, run_time=24)
        )


class ContrastTwoAlgorithmsFrame(DisectAQuantumComputer):
    def construct(self):
        # Set up screens
        background = FullScreenFadeRectangle()
        background.set_fill(GREY_E, 1)
        screens = Rectangle(6, 5).get_grid(1, 2, buff=LARGE_BUFF)
        screens.set_fill(BLACK, 1)
        screens.set_stroke(WHITE, 2)
        screens.to_edge(DOWN, buff=LARGE_BUFF)

        self.add(background)
        self.add(screens)

        # Titles
        titles = VGroup(
            VGroup(get_classical_computer_symbol(), Tex(R"\mathcal{O}(N)")),
            VGroup(get_quantum_computer_symbol(), Tex(R"\mathcal{O}(\sqrt{N})")),
        )
        for title, screen in zip(titles, screens):
            title[0].set_height(1.5)
            title[1].set_height(0.75)
            title.arrange(RIGHT, buff=0.5)
            title.next_to(screen, UP, aligned_edge=LEFT)

        self.add(titles)

        # Preview quantum search
        boxes = Square(0.1).get_grid(25, 4, fill_rows_first=False, v_buff=0.05, h_buff=1.1)
        boxes.set_height(screens[1].get_height() - MED_LARGE_BUFF)
        boxes.move_to(screens[1], UL).shift(MED_SMALL_BUFF * DR)

        values = VGroup(
            Integer(n, font_size=12).replace(box, dim_to_match=1)
            for n, box in enumerate(boxes)
        )

        dist = np.ones(100)
        width_ratio = 5
        bars = self.get_dist_bars(dist, boxes, width_ratio=width_ratio)

        q_dots = DotCloud().to_grid(4, 25).rotate(-90 * DEG)
        q_dots.replace(titles[1][0], dim_to_match=1)
        q_dots.stretch(0.5, 1)
        lines = VGroup(
            Line(point, box.get_center())
            for point, box in zip(q_dots.get_points(), boxes)
        )
        for line in lines:
            line.insert_n_curves(20)
            color = random_bright_color(hue_range=(0.3, 0.4))
            line.set_stroke(color, [0, 2, 2, 0], opacity=0.5)

        self.add(values)
        self.add(bars)

        for n in range(1, 10):
            dist[42] += n
            width_ratio *= 0.9
            lines.shuffle
            self.play(
                LaggedStartMap(VShowPassingFlash, lines, time_width=1.5, lag_ratio=2e-3),
                Transform(bars, self.get_dist_bars(dist, boxes, width_ratio=width_ratio), time_span=(0.5, 1))
            )


class QuantumCompilation(InteractiveScene):
    def construct(self):
        # Show circuitry
        machine = get_blackbox_machine()
        label = machine.submobjects[0]
        machine.remove(label)
        circuit = SVGMobject("BinaryFunctionCircuit")
        circuit.flip(RIGHT)
        circuit.set_stroke(width=0)
        circuit.set_fill(BLUE_B, 1)
        circuit.set_height(machine.get_height() * 0.8)
        circuit.move_to(machine).shift(0.25 * RIGHT)
        circuit.scale(2, about_point=ORIGIN)
        circuit.sort(lambda p: np.dot(p, DR))

        self.add(machine, label)

        self.wait()
        self.play(
            machine.animate.scale(2, about_point=ORIGIN).set_fill(GREY_E),
            FadeOut(label, scale=2),
        )
        self.play(Write(circuit, lag_ratio=0.05))
        self.wait()

        # Show binary input
        number = Integer(13, font_size=72, edge_to_fix=ORIGIN)
        bit_string = BitString(number.get_value())
        bit_string.next_to(machine, LEFT)
        number.next_to(machine, LEFT, MED_LARGE_BUFF)

        bit_string.set_z_index(-1)
        output = BitString(0, length=1).scale(1.5)
        output.set_z_index(-1)
        output.next_to(machine, RIGHT, MED_LARGE_BUFF)

        self.play(FadeIn(number, RIGHT))
        self.play(
            number.animate.next_to(bit_string, UP, MED_LARGE_BUFF),
            TransformFromCopy(number.replicate(5), bit_string, lag_ratio=0.01),
        )
        self.wait()

        self.play(
            FadeOut(bit_string.copy(), 2 * RIGHT, lag_ratio=0.05, path_arc=45 * DEG),
            FadeIn(output, RIGHT, time_span=(0.75, 1.5))
        )
        self.play(
            ChangeDecimalToValue(number, 5),
            UpdateFromFunc(bit_string, lambda m: m.set_value(number.get_value())),
            run_time=1
        )
        output.set_value(1)
        self.wait()

        # Show quantum case
        c_machine = VGroup(machine, circuit)
        c_machine.target = c_machine.generate_target()
        c_machine.target.scale(0.5).to_edge(UP)

        q_machine = Square().match_style(machine).set_height(0.5 * machine.get_height())
        lines = Line(ORIGIN, 0.75 * RIGHT).get_grid(4, 1, v_buff=0.25)
        lines.next_to(q_machine, LEFT, buff=0)
        q_machine.add(lines)
        q_machine.add(lines.copy().next_to(q_machine, RIGHT, buff=0))
        q_machine.to_edge(DOWN, buff=1.5)

        q_label = Text("Quantum\nGates")  # If I were ambitious, I'd show the proper quantum circuit here
        q_label.set_color(TEAL)
        q_label.set_height(q_machine.get_height() * 0.4)
        q_label.move_to(q_machine)

        arrow = Arrow(c_machine.target, q_machine, thickness=5)

        self.play(
            MoveToTarget(c_machine),
            bit_string.animate.next_to(c_machine.target, LEFT),
            output.animate.next_to(c_machine.target, RIGHT, MED_LARGE_BUFF),
            FadeOut(number, UP),
        )
        self.play(GrowArrow(arrow))
        self.play(
            FadeTransform(c_machine[0].copy(), q_machine),
            TransformFromCopy(c_machine[1], q_label, lag_ratio=0.01, run_time=2),
        )
        self.wait()

        # Map to quantum input
        q_input = KetGroup(bit_string.copy())
        q_input.next_to(q_machine, LEFT)
        q_output = q_input.copy()
        neg = Tex(R"-").next_to(q_output, LEFT, SMALL_BUFF)
        q_output.add(neg)
        q_output.next_to(q_machine, RIGHT)

        input_rect = SurroundingRectangle(bit_string)
        input_rect.set_stroke(YELLOW, 2)
        output_rect = SurroundingRectangle(output)
        output_rect.set_stroke(GREEN, 2)
        check = Checkmark()
        check.match_height(output)
        check.set_color(GREEN)
        check.next_to(output, RIGHT)

        self.play(ShowCreation(input_rect))
        self.play(TransformFromCopy(input_rect, output_rect, path_arc=-45 * DEG))
        self.play(
            FadeOut(output_rect),
            Write(check[0], run_time=1)
        )
        self.wait()
        self.play(
            input_rect.animate.surround(q_input),
            TransformFromCopy(VGroup(VectorizedPoint(bit_string.get_center()), bit_string), q_input)
        )
        self.play(
            FadeOut(input_rect),
            FadeOut(q_input.copy(), 3 * RIGHT),
            FadeIn(q_output, 3 * RIGHT, time_span=(0.5, 1.5))
        )
        self.wait()

        # Show False inputs
        flipped_input = q_input.copy()
        flipped_output = q_output.copy()

        input_value_tracker = ValueTracker(number.get_value())
        ex = Exmark()
        ex.set_color(RED)
        ex.replace(check, 1)

        self.remove(q_output, check)
        self.add(ex)
        output.set_value(0)

        input_value_tracker.increment_value(1)
        self.play(
            input_value_tracker.animate.set_value(13).set_anim_args(rate_func=linear),
            UpdateFromFunc(bit_string, lambda m: m.set_value(int(input_value_tracker.get_value()))),
            UpdateFromFunc(q_input[1], lambda m: m.set_value(int(input_value_tracker.get_value()))),
            run_time=2
        )
        self.wait()

        q_output2 = q_input.copy()
        q_output2.next_to(q_machine, RIGHT, MED_LARGE_BUFF)
        self.play(TransformFromCopy(q_input, q_output2, path_arc=45 * DEG))
        self.wait()

        # Show combination
        combined_input = VGroup(q_input.copy(), Tex(R"+"), flipped_input)
        combined_input.arrange(DOWN, buff=SMALL_BUFF)
        combined_input.next_to(q_machine, LEFT)
        key_icon = get_key_icon()
        key_icon.match_height(q_input)
        key_icon.next_to(combined_input[2], LEFT, SMALL_BUFF)

        combined_output = VGroup(q_output2.copy(), Tex(R"+"), flipped_output)
        combined_output.arrange(DOWN, buff=SMALL_BUFF)
        combined_output.next_to(q_machine, RIGHT)

        self.play(
            ReplacementTransform(q_input, combined_input[0]),
            Write(combined_input[1:]),
            ReplacementTransform(q_output2, combined_output[0]),
            Write(combined_output[1:]),
            FadeIn(key_icon)
        )
        self.play(
            input_value_tracker.animate.set_value(5).set_anim_args(rate_func=linear),
            UpdateFromFunc(bit_string, lambda m: m.set_value(int(input_value_tracker.get_value()))),
        )
        output.set_value(1)
        self.remove(ex)
        self.add(check)
        self.wait()
