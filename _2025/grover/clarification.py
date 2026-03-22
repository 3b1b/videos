from manim_imports_ext import *
from _2025.blocks_and_grover.qc_supplements import *


default_sudoku_values = [  # TODO
    [7, 4, 3, 6, 5, 1, 9, 8, 2],
    [1, 8, 9, 2, 3, 4, 7, 6, 5],
    [6, 2, 5, 9, 8, 7, 4, 3, 1],
    [3, 2, 5, 8, 6, 9, 4, 1, 7],
    [9, 7, 6, 3, 4, 1, 5, 2, 8],
    [1, 4, 8, 7, 5, 2, 3, 6, 9],
    [2, 3, 8, 5, 7, 6, 1, 9, 4],
    [4, 1, 7, 8, 9, 3, 6, 5, 2],
    [5, 9, 6, 2, 1, 4, 8, 7, 3],
]

default_sudoku_locked_cells = [
    [3, 7, 8],
    [1, 6, 7, 8],
    [3, 5],
    [1, 3, 6, 8],
    [5],
    [1, 3, 4, 8],
    [5, 6],
    [5],
    [4, 9],
]


class Sudoku(VGroup):
    def __init__(
        self,
        values=default_sudoku_values,
        locked_cells=default_sudoku_locked_cells,
        height=4,
        big_square_stroke_width=3,
        little_square_stroke_width=0.5,
        locked_number_color=BLUE_B,
        num_to_square_height_ratio=0.5
    ):
        self.big_grid = self.get_square_grid(height, big_square_stroke_width)
        self.little_grids = VGroup(
            self.get_square_grid(height / 3, little_square_stroke_width).move_to(square)
            for square in self.big_grid
        )
        self.numbers = VGroup(
            VGroup(
                Integer(num).replace(square, 1).scale(num_to_square_height_ratio)
                for square, num in zip(grid, arr)
            )
            for grid, arr in zip(self.little_grids, values)
        )
        self.locked_cells = locked_cells
        self.locked_numbers = VGroup()
        self.unlocked_numbers = VGroup()

        for coords, group in zip(locked_cells, self.numbers):
            for x, num in enumerate(group):
                if x in coords:
                    self.locked_numbers.add(num)
                else:
                    self.unlocked_numbers.add(num)
        self.locked_numbers.set_fill(locked_number_color, border_width=2)

        super().__init__(self.big_grid, self.little_grids, self.numbers)

    def get_rows(self):
        grids = self.numbers
        slices = [slice(0, 3), slice(3, 6), slice(6, 9)]
        rows = VGroup()
        for slc1 in slices:
            for slc2 in slices:
                row = VGroup()
                for grid in grids[slc1]:
                    for num in grid[slc2]:
                        row.add(num)
                rows.add(row)
        return rows

    def get_columns(self):
        grids = self.numbers
        slices = [slice(0, 9, 3), slice(1, 9, 3), slice(2, 9, 3)]
        cols = VGroup()
        for slc1 in slices:
            for slc2 in slices:
                col = VGroup()
                for grid in grids[slc1]:
                    for num in grid[slc2]:
                        col.add(num)
                cols.add(col)
        return cols

    def get_number_squares(self):
        return self.numbers

    def get_square_grid(
        self,
        height,
        stroke_width,
        stroke_color=WHITE,
        fill_color=GREY_E,
    ):
        square = Square()
        square.set_fill(fill_color, 1)
        square.set_stroke(stroke_color, stroke_width)
        grid = square.get_grid(3, 3, buff=0)
        grid.set_height(height)
        return grid


class Intro(InteractiveScene):
    random_seed = 1

    def construct(self):
        # Test
        background = FullScreenRectangle().set_fill(GREY_E)
        background.set_fill(GREY_E, 0.5)
        self.add(background)

        icon = get_quantum_computer_symbol(height=3)
        icon.center()
        self.play(Write(icon, run_time=3, lag_ratio=1e-2, stroke_color=TEAL))
        self.wait()
        self.play(icon.animate.to_edge(LEFT))
        self.wait()

        # Comments
        folder = Path('/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2025/blocks_and_grover/Comments')
        comments = Group(
            Group(ImageMobject(folder / name))
            for name in os.listdir(folder)
        )
        for comment in comments:
            comment.add_to_back(SurroundingRectangle(comment, buff=0).set_stroke(WHITE, 1))
            comment.set_width(4)
            comment.move_to(4 * LEFT)
            comment.shift(np.random.uniform(-1, 1) * RIGHT + np.random.uniform(-3, 3) * UP)

        self.play(
            FadeOut(icon, time_span=(0, 2)),
            LaggedStartMap(FadeIn, comments, lag_ratio=0.6, shift=0.25 * UP, run_time=8)
        )
        self.wait()


class HowDoYouKnowWhichAxis(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.play(
            morty.change("guilty"),
            stds[2].says("Doesn't this assumes\nknowing the key value?", mode="angry", bubble_direction=LEFT),
            stds[0].change("confused", look_at=self.screen),
            stds[1].change("pleading", look_at=self.screen),
        )
        self.wait()
        self.look_at(self.screen)
        self.wait(5)
        self.play(self.change_students("erm", "confused", "sassy"))
        self.wait(5)


class SudokuChecker(InteractiveScene):
    def construct(self):
        # Introduce, and add solution
        sudoku = Sudoku(height=6)
        sudoku.unlocked_numbers.set_opacity(0)

        self.play(Write(sudoku, run_time=3, lag_ratio=1e-2))
        sudoku.unlocked_numbers.shuffle()
        self.play(sudoku.unlocked_numbers.animate.set_opacity(1).set_anim_args(lag_ratio=0.1), run_time=3)
        self.wait()

        # Check rows, columns, squares
        row_rects, col_rects, square_rects = rect_groups = VGroup(
            VGroup(
                VGroup(SurroundingRectangle(num) for num in sg)
                for sg in group
            )
            for group in [sudoku.get_rows(), sudoku.get_columns(), sudoku.get_number_squares()]
        )
        rect_groups.set_stroke(YELLOW, 3)

        for group in rect_groups:
            self.play(
                LaggedStart(*(
                    VFadeInThenOut(sg, lag_ratio=0.025)
                    for sg in group
                ), lag_ratio=0.5, run_time=5)
            )

        # Replace with question marks
        unlocked_numbers = sudoku.unlocked_numbers
        q_marks = VGroup(
            Text("?").replace(num, 1)
            for num in sudoku.unlocked_numbers
        )
        q_marks.set_color(RED)
        unlocked_numbers.save_state()
        self.play(
            Transform(unlocked_numbers, q_marks, lag_ratio=1e-2, run_time=2)
        )
        self.wait()
        self.play(Restore(unlocked_numbers))

        # Show randomization
        def randomize_numbers(numbers):
            for number in numbers:
                number.set_value(random.uniform(1, 9))

        self.play(UpdateFromFunc(sudoku.unlocked_numbers, randomize_numbers, run_time=10))


class SudokuCheckingCode(InteractiveScene):
    def construct(self):
        # Thanks Claude!
        code = Code("""
            def is_valid_sudoku(board):
                \"\"\"
                Check if a completed Sudoku board is valid.

                Args:
                    board: A 9x9 list of lists where each
                    cell contains an integer from 1 to 9

                Returns:
                    bool: True if the solution is valid, False otherwise
                \"\"\"
                # Check rows
                for row in board:
                    if set(row) != set(range(1, 10)):
                        return False

                # Check columns
                for col in range(9):
                    column = [board[row][col] for row in range(9)]
                    if set(column) != set(range(1, 10)):
                        return False

                # Check 3x3 sub-boxes
                for box_row in range(0, 9, 3):
                    for box_col in range(0, 9, 3):
                        # Get all numbers in the current 3x3 box
                        box = []
                        for i in range(3):
                            for j in range(3):
                                box.append(board[box_row + i][box_col + j])
                        if set(box) != set(range(1, 10)):
                            return False

                # If all checks pass, the solution is valid
                return True
        """, alignment="LEFT")
        code.set_height(7)
        self.play(ShowIncreasingSubsets(code, run_time=8, rate_func=linear))
        self.wait()


class ArrowToQC(InteractiveScene):
    def construct(self):
        # Test
        icon = get_quantum_computer_symbol(height=2.5)
        icon.center().to_edge(DOWN)
        arrow = Vector(1.5 * DOWN, thickness=6)
        arrow.next_to(icon, UP)

        self.play(
            GrowArrow(arrow),
            FadeIn(icon, DOWN)
        )
        self.wait()


class CompiledSudokuVerifier(InteractiveScene):
    def construct(self):
        # Set up
        sudoku = Sudoku(height=5)
        sudoku.to_edge(LEFT)

        machine = get_blackbox_machine(height=3, label_tex=R"\text{Verifier}")
        machine.next_to(sudoku, RIGHT, LARGE_BUFF)
        machine[-1].scale(0.65).set_color(YELLOW)

        self.add(sudoku)
        self.add(machine)

        # Pile logic gates into the machine
        gates = VGroup(
            SVGMobject("and_gate"),
            SVGMobject("or_gate"),
            SVGMobject("not_gate"),
        )
        names = VGroup(Text(text) for text in ["AND", "OR", "NOT"])
        names.scale(0.5)
        names.set_fill(GREY_C)

        gates.set_height(0.65)
        gates.set_fill(GREY_B)
        gates.arrange(RIGHT, buff=MED_LARGE_BUFF)
        gates.next_to(machine, UP, MED_LARGE_BUFF)
        for name, gate in zip(names, gates):
            name.next_to(gate, UP)

        pile_of_gates = VGroup(*it.chain(*(g.replicate(200) for g in gates)))
        pile_of_gates.shuffle()
        pile_of_gates.set_fill(opacity=0.25)
        pile_of_gates.generate_target()
        for gate in pile_of_gates.target:
            gate.scale(0.25)
            shift = np.random.uniform(-1.5, 1.5, 3)
            shift[2] = 0
            gate.move_to(machine.get_center() + shift)
            gate.set_fill(opacity=0.1)

        self.add(gates, names)
        self.play(
            MoveToTarget(pile_of_gates, lag_ratio=3.0 / len(pile_of_gates), run_time=8)
        )

        # Turn sudoku into binary
        all_numbers = VGroup(num for grid in sudoku.numbers for num in grid)
        bit_groups = VGroup(
            BitString(num.get_value()).replace(num, 1).scale(0.35)
            for num in all_numbers
        )

        self.play(
            ReplacementTransform(all_numbers, bit_groups, lag_ratio=1e-3),
            sudoku.big_grid.animate.set_fill(opacity=0.1).set_stroke(opacity=0.25),
            sudoku.little_grids.animate.set_fill(opacity=0.1).set_stroke(opacity=0.25),
        )
        self.wait()

        target_bits = bit_groups.copy()
        target_bits.arrange(DOWN, buff=0.025)
        target_bits.set_height(machine.get_height() * 0.5)
        target_bits.move_to(machine, LEFT)
        target_bits.set_opacity(0.1)
        self.play(TransformFromCopy(bit_groups, target_bits, lag_ratio=1e-3, run_time=2))
        self.wait()

        # Show outputs
        outputs = VGroup(Integer(1), Integer(0))
        outputs.set_height(0.75)
        outputs.next_to(machine, RIGHT, LARGE_BUFF)
        marks = VGroup(Checkmark().set_color(GREEN), Exmark().set_color(RED))
        for mark, output in zip(marks, outputs):
            mark.match_height(output)
            mark.next_to(output, RIGHT)
            self.play(FadeIn(output, RIGHT))
            self.play(Write(mark))
            self.wait()
            self.play(FadeOut(mark), FadeOut(output))


class StateVectorsAsABasis(InteractiveScene):
    def construct(self):
        # Axes
        frame = self.frame
        x_range = y_range = z_range = (-2, 2, 1)
        axes = ThreeDAxes(x_range, y_range, z_range)
        axes.set_height(4)

        basis_vectors = VGroup(
            Vector(2 * vect, thickness=4)
            for vect in np.identity(3)
        )
        basis_vectors.set_submobject_colors_by_gradient(BLUE_D, BLUE_B)
        for vect in basis_vectors:
            vect.rotate(90 * DEG, axis=vect.get_vector())

        frame.reorient(-23, 81, 0, (-1.0, 0, 0.5), 4)
        frame.add_ambient_rotation()
        self.add(axes)

        # Bit strings
        two_qubits = VGroup(
            KetGroup(BitString(n, length=2))
            for n in range(4)
        )
        four_qubits = VGroup(
            KetGroup(BitString(n, length=4))
            for n in range(16)
        )
        for group in [two_qubits, four_qubits]:
            group.fix_in_frame()
            group.arrange(DOWN)
            group.set_max_height(7)
            group.to_edge(LEFT, buff=LARGE_BUFF)
        two_qubits.scale(1.5, about_edge=LEFT).space_out_submobjects(1.25)

        basis_labels = VGroup(
            two_qubits[0].copy().scale(0.3).rotate(90 * DEG, RIGHT).next_to(basis_vectors[0].get_end(), OUT, SMALL_BUFF),
            two_qubits[1].copy().scale(0.3).rotate(90 * DEG, RIGHT).next_to(basis_vectors[1].get_end(), OUT, SMALL_BUFF),
            two_qubits[2].copy().scale(0.3).rotate(90 * DEG, RIGHT).next_to(basis_vectors[2].get_end(), RIGHT, SMALL_BUFF),
        )
        basis_labels.unfix_from_frame()

        self.play(LaggedStartMap(FadeIn, two_qubits, shift=0.5 * UP, lag_ratio=0.5))

        for src, trg, vect in zip(two_qubits, basis_labels, basis_vectors):
            self.play(
                TransformFromCopy(src, trg),
                GrowArrow(vect)
            )
        self.wait(2)

        # Name basis vectors
        basis_name = TexText(R"``Basis vectors''")
        basis_name.fix_in_frame()
        basis_name.set_color(BLUE)
        basis_name.to_corner(UR, buff=MED_SMALL_BUFF)
        self.play(Write(basis_name))
        self.wait(10)
        self.play(FadeOut(basis_name))

        # Replace with larger vectors
        new_basis_labels = VGroup(
            four_qubits[0].copy().scale(0.3).rotate(90 * DEG, RIGHT).next_to(basis_vectors[0].get_end(), OUT, SMALL_BUFF),
            four_qubits[1].copy().scale(0.3).rotate(90 * DEG, RIGHT).next_to(basis_vectors[1].get_end(), OUT, SMALL_BUFF),
            four_qubits[2].copy().scale(0.3).rotate(90 * DEG, RIGHT).next_to(basis_vectors[2].get_end(), RIGHT, SMALL_BUFF),
        )
        new_basis_labels.unfix_from_frame()
        self.play(
            FadeOut(two_qubits),
            FadeOut(basis_labels),
            LaggedStartMap(FadeIn, four_qubits, lag_ratio=0.25)
        )

        for src, trg in zip(four_qubits, new_basis_labels):
            self.play(TransformFromCopy(src, trg))
        self.wait(20)


class OperationsOnQC(InteractiveScene):
    def construct(self):
        # Bad output
        icon = get_quantum_computer_symbol(height=3)
        icon.center()
        in_ket = KetGroup(BitString(12).scale(2))
        in_ket.next_to(icon, LEFT)
        arrows = Vector(RIGHT, thickness=4).replicate(2)
        arrows[0].next_to(icon, LEFT)
        arrows[1].next_to(icon, RIGHT)
        in_ket.next_to(arrows, LEFT)

        bad_output = VGroup(Text("True"), Text("or"), Text("False"))
        bad_output.scale(1.5)
        bad_output.arrange(DOWN)
        bad_output.next_to(arrows, RIGHT)
        big_cross = Cross(bad_output)
        big_cross.scale(1.25)
        big_cross.set_stroke(RED, [0, 12, 12, 12, 0])

        self.add(icon, arrows, in_ket)
        self.play(LaggedStart(
            FadeOut(in_ket.copy(), shift=2 * RIGHT, scale=0.5),
            FadeIn(bad_output, shift=2 * RIGHT, scale=2),
            lag_ratio=0.5
        ))
        self.play(ShowCreation(big_cross))
        self.wait()
        self.play(FadeOut(in_ket), FadeOut(bad_output), FadeOut(big_cross))
        self.wait()


class TwoByTwoGrid(InteractiveScene):
    random_seed = 1

    def construct(self):
        # Set up
        rects = ScreenRectangle().set_height(FRAME_HEIGHT / 2).get_grid(2, 2, buff=0)
        h_line = Line(LEFT, RIGHT).replace(rects, 0)
        v_line = Line(UP, DOWN).replace(rects, 1)
        lines = VGroup(h_line, v_line)
        lines.set_stroke(WHITE, 2)

        self.add(lines)

        # Add classical verifiers
        verifiers = get_blackbox_machine(label_tex="").replicate(4)
        for n, verifier, rect in zip(it.count(), verifiers, rects):
            verifier.set_height(1.0)
            verifier.move_to(rect)
            if n < 2:
                label = Text("Verifier", font_size=24)
                label.set_color(YELLOW)
            else:
                verifier.rotate(90 * DEG)
                label = get_quantum_computer_symbol(height=0.75)

            label.move_to(verifier)
            verifier.add(label)

        verifiers[:2].shift(0.75 * RIGHT)

        good_sudoku, bad_sudoku = sudokus = VGroup(
            Sudoku(big_square_stroke_width=2, little_square_stroke_width=0.25)
            for x in range(2)
        )
        sudokus.set_height(2.5)

        for number in bad_sudoku.unlocked_numbers:
            number.set_value(random.randint(1, 9))

        classical_outputs = VGroup(Integer(1), Integer(0))
        marks = VGroup(Checkmark().set_color(GREEN), Exmark().set_color(RED))

        for sudoku, verifier, output, mark in zip(sudokus, verifiers, classical_outputs, marks):
            sudoku.next_to(verifier, LEFT, MED_LARGE_BUFF)
            output.next_to(verifier, RIGHT, MED_LARGE_BUFF)
            mark.match_height(output)
            mark.next_to(output, RIGHT, SMALL_BUFF)

        self.add(verifiers[:2])
        self.add(sudokus)

        for sudoku, verifier, output, mark in zip(sudokus, verifiers, classical_outputs, marks):
            self.play(LaggedStart(
                FadeOutToPoint(sudoku.numbers.copy(), verifier.get_center(), lag_ratio=1e-3),
                FadeInFromPoint(output, verifier.get_center()),
                lag_ratio=0.5,
                run_time=2
            ))
            self.play(Write(mark))

        self.wait()

        # Map to quantum verifiers
        self.play(TransformFromCopy(verifiers[:2], verifiers[2:], run_time=2))
        self.wait()

        # Translate True behavior
        good_input = self.turn_into_bits(good_sudoku)
        good_ket = KetGroup(good_input.copy(), height_scale_factor=1.5)
        good_ket.next_to(verifiers[2], UP)
        good_ket_out = good_ket.copy()
        neg = Tex("-", font_size=24)
        neg.set_fill(GREEN, border_width=3)
        neg.next_to(good_ket_out, LEFT, SMALL_BUFF)
        good_ket_out.add_to_back(neg)
        good_ket_out.next_to(verifiers[2], DOWN, buff=0.2)

        mult_neg_1_words = TexText(R"Multiply\\by $-1$", font_size=36)
        mult_neg_1_words.set_fill(TEAL_A)
        mult_neg_1_words.next_to(verifiers[2], RIGHT, MED_LARGE_BUFF)

        self.play(FadeTransform(good_input.copy(), good_ket))
        self.wait()
        self.play(
            LaggedStart(
                FadeOutToPoint(good_ket.copy(), verifiers[2].get_center(), lag_ratio=0.01),
                FadeInFromPoint(good_ket_out, verifiers[2].get_center(), lag_ratio=0.01),
                lag_ratio=0.5
            ),
            FadeIn(mult_neg_1_words, RIGHT),
        )
        self.wait()

        # Translate False behavior (a lot of coying, but I'm in a rush)
        bad_input = self.turn_into_bits(bad_sudoku)
        bad_ket = KetGroup(bad_input.copy(), height_scale_factor=1.5)
        bad_ket.next_to(verifiers[3], UP)
        bad_ket_out = bad_ket.copy()
        bad_ket_out.next_to(verifiers[3], DOWN)

        mult_pos_1_words = TexText(R"Multiply\\by $+1$", font_size=36)
        mult_pos_1_words.set_fill(TEAL_A)
        mult_pos_1_words.next_to(verifiers[3], RIGHT, MED_LARGE_BUFF)

        self.play(FadeTransform(bad_input.copy(), bad_ket))
        self.wait()
        self.play(
            LaggedStart(
                FadeOutToPoint(bad_ket.copy(), verifiers[3].get_center(), lag_ratio=0.01),
                FadeInFromPoint(bad_ket_out, verifiers[3].get_center(), lag_ratio=0.01),
                lag_ratio=0.5
            ),
            FadeIn(mult_pos_1_words, RIGHT),
        )
        self.wait()

        # Show logic gates
        gate_groups = VGroup(
            SVGMobject("and_gate").replicate(2),
            SVGMobject("not_gate").replicate(2),
            SVGMobject("or_gate").replicate(2),
        )
        for group in gate_groups:
            group.set_fill(BLUE)
            group.target = group.generate_target()
            for mob, box in zip(group, verifiers):
                mob.match_height(box)
                mob.scale(0.8)
                mob.next_to(box, UP)
            for mob, box in zip(group.target, verifiers[2:]):
                mob.match_width(box)
                mob.scale(0.7)
                mob.move_to(box)
                mob.set_fill(TEAL, 0.5)

        for group in gate_groups:
            self.play(LaggedStartMap(FadeIn, group, shift=UP, lag_ratio=0.5))
            self.wait()
            self.play(TransformFromCopy(group, group.target))
            self.play(FadeOut(group.target))
            self.play(FadeOut(group))

    def turn_into_bits(self, sudoku):
        # Test
        bits = VGroup(
            BitString(num.get_value()).replace(num, 1).scale(0.35)
            for grid in sudoku.numbers for num in grid
        )
        sudoku.save_state()

        in_group = VGroup(
            bits[0].copy().set_width(0.25),
            Tex(R"\cdots", font_size=20),
            bits[-1].copy().set_width(0.25),
        )
        for piece in in_group:
            piece.space_out_submobjects(0.85)
        in_group[1].scale(0.7)
        in_group.arrange(RIGHT, buff=0.025)
        in_group.replace(sudoku, 0)
        in_group.set_fill(GREY_A)

        sudoku.saved_state.scale(0.5)
        sudoku.saved_state.fade(0.5)
        sudoku.saved_state.to_edge(UP, buff=MED_SMALL_BUFF)

        self.play(
            sudoku.animate.fade(0.9),
            FadeIn(bits, lag_ratio=1e-3)
        )
        self.play(LaggedStart(
            ReplacementTransform(bits[0], in_group[0]),
            *(ReplacementTransform(bs, in_group[1]) for bs in bits[1:-1]),
            ReplacementTransform(bits[-1], in_group[-1]),
            Restore(sudoku),
            lag_ratio=1e-2,
        ))
        return in_group


class AskWhyThatsTrue(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.play(
            stds[1].says("How does that work?", mode="maybe"),
            stds[0].change('erm', look_at=self.screen),
            stds[2].change("confused", look_at=self.screen),
            morty.change('hesitant'),
        )
        self.look_at(self.screen)
        self.wait(5)

        # Mapping
        mapping = VGroup(
            VGroup(Text("True").set_color(GREEN), Vector(DOWN), Tex(R"\times -1")).arrange(DOWN),
            VGroup(Text("False").set_color(RED), Vector(DOWN), Tex(R"\times +1")).arrange(DOWN),
        )
        for part in mapping:
            tex = part[2]
            tex[1].scale(0.75, about_point=tex[2].get_left())
            tex[0].scale(1.5)
        mapping.arrange(RIGHT, buff=2.0)
        mapping.move_to(self.hold_up_spot, DOWN).shift(LEFT)

        self.play(
            morty.change("raise_right_hand"),
            FadeIn(mapping, UP),
            stds[1].debubble(mode="pondering"),
            stds[0].change("sassy", mapping),
            stds[2].change("hesitant", mapping),
        )
        self.wait()
        self.play(morty.change("hesitant"))
        self.wait(3)
        self.play(morty.change("pondering", mapping))
        self.wait(3)


class ListOfConfusions(InteractiveScene):
    def construct(self):
        # Test
        items = BulletedList(
            "Insufficient detail",
            "Bad framing",
            "Glossing over linearity",
            buff=1.0
        )
        rects = VGroup(
            SurroundingRectangle(item[1:])
            for item in items
        )
        rects.set_stroke(width=0)
        rects.set_fill(GREY_D, 1)

        self.add(items)
        self.add(rects[1:])
        self.wait()
        self.play(
            items[0].animate.fade(0.5),
            rects[1].animate.stretch(0, 0, about_edge=RIGHT),
        )
        self.wait()
        self.play(
            items[1].animate.fade(0.5),
            rects[2].animate.stretch(0, 0, about_edge=RIGHT),
        )
        self.wait()


class SolveSHAWord(InteractiveScene):
    def construct(self):
        words = VGroup(
            TexText(R"Solve for ${x}$"),
            Tex(R"\text{SHA256}({x}) = 0"),
        )
        words.set_color(GREY_B)
        for word in words:
            word["{x}"].set_color(YELLOW)
        words.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        self.play(Write(words))
        self.wait()


class ThatsOnMe(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.play(
            morty.says("That's on me", mode="guilty"),
            self.change_students("pondering", "well", "hesitant", look_at=self.screen),
        )
        self.wait(3)


class ShowSuperposition(InteractiveScene):
    def construct(self):
        # Axes
        frame = self.frame
        x_range = y_range = z_range = (-2, 2, 1)
        axes = ThreeDAxes(x_range, y_range, z_range)
        unit_size = 3
        axes.set_height(2 * unit_size)

        basis_vectors = VGroup(
            Vector(unit_size * vect, thickness=5)
            for vect in np.identity(3)
        )
        basis_vectors.set_submobject_colors_by_gradient(BLUE_D, BLUE_B)
        for n, vect in enumerate(basis_vectors):
            if n < 2:
                vect.always.set_perpendicular_to_camera(frame)
            else:
                vect.rotate(90 * DEG, OUT)

        frame.reorient(-33, 74, 0)
        frame.clear_updaters()
        frame.add_ambient_rotation(0.5 * DEG)
        self.add(axes)
        self.add(basis_vectors)

        # Label bases
        basis_labels = VGroup(
            KetGroup(BitString(n, length=2)).set_height(0.35)
            for n in range(3)
        )
        basis_labels.rotate(90 * DEG, RIGHT)
        for label, vect, direction in zip(basis_labels, basis_vectors, [RIGHT, UP + OUT, RIGHT]):
            label.next_to(vect.get_end(), direction, MED_SMALL_BUFF)
            label.set_fill(vect.get_color(), 1)

        self.add(basis_labels, basis_vectors)
        self.wait(4)

        # Show a general vector
        vect_coords = normalize([np.sqrt(2) / 2, 0.5, 0.5])
        vector = Vector(unit_size * vect_coords, thickness=5, fill_color=TEAL)
        vector.set_perpendicular_to_camera(frame)
        dec_kw = dict(include_sign=True, font_size=36)
        vect_label = VGroup(
            DecimalNumber(vect_coords[0], **dec_kw),
            KetGroup(BitString(0, 2)),
            DecimalNumber(vect_coords[1], **dec_kw),
            KetGroup(BitString(1, 2)),
            DecimalNumber(vect_coords[2], **dec_kw),
            KetGroup(BitString(2, 2)),
            DecimalNumber(0, **dec_kw),
            KetGroup(BitString(3, 2)),
        )
        vect_label[1::2].set_submobject_colors_by_gradient(BLUE, BLUE_E)
        vect_label.arrange(RIGHT, buff=SMALL_BUFF)
        vect_label.rotate(90 * DEG, RIGHT)
        vect_label.next_to(vector.get_end(), RIGHT)

        self.play(
            GrowArrow(vector),
            frame.animate.set_x(1),
        )
        self.play(FadeIn(vect_label))
        self.wait(5)

        # Show column vector
        col = DecimalMatrix(np.array([*vect_coords, 0]).reshape(-1, 1), decimal_config=dec_kw)
        col.scale(0.75)
        col.rotate(90 * DEG, RIGHT)
        col.next_to(vector.get_end(), RIGHT)
        eq = Tex("=")
        eq.rotate(90 * DEG, RIGHT)
        eq.next_to(col, RIGHT, SMALL_BUFF)

        self.play(
            FadeIn(col.get_brackets()),
            TransformFromCopy(vect_label[0::2], col.get_entries(), run_time=2),
            FadeIn(eq),
            vect_label.animate.next_to(eq, RIGHT, SMALL_BUFF)
        )

        # Surrounding rectangles
        rects = VGroup(
            SurroundingRectangle(mob.copy().rotate(90 * DEG, LEFT)).rotate(90 * DEG, RIGHT)
            for mob in [col, vect_label]
        )
        rects.set_stroke(YELLOW, 2)
        self.play(ShowCreation(rects[0]))
        self.wait(5)
        self.play(Transform(*rects))
        self.wait(7)

        # Write superposition
        word = TexText("``Superposition''", font_size=72)
        word.rotate(90 * DEG, RIGHT)
        word.set_color(YELLOW)
        word.next_to(rects[0], OUT)

        self.play(Write(word), frame.animate.set_x(2), run_time=3)
        self.wait(8)


class NorthEastTraveler(InteractiveScene):
    def construct(self):
        # Add compass
        compass = self.get_compass()
        compass.move_to(5 * RIGHT + 2.5 * UP)
        self.add(compass)

        # Show travler
        randy = Randolph(height=1, mode="tease")
        vel_vect = Vector(2 * RIGHT, thickness=4, color=YELLOW)
        vel_vect.move_to(randy.get_bottom(), LEFT)
        travler = VGroup(randy, vel_vect)
        travler.rotate(45 * DEG) 
        travler.move_to(4 * LEFT + 2 * DOWN)

        self.add(travler)
        self.play(travler.animate.shift(3 * UR), run_time=5, rate_func=linear)

        # Show components
        components = VGroup(
            Vector(math.sqrt(2) * UP).shift(math.sqrt(2) * RIGHT).set_fill(GREEN),
            Vector(math.sqrt(2) * RIGHT).set_fill(RED)
        )
        components.shift(vel_vect.get_start() - components[1].get_start())
        labels = VGroup(
            Text("North").next_to(components[0], RIGHT, buff=-0.05),
            Text("East").next_to(components[1], DOWN, SMALL_BUFF),
        )

        for component, label in zip(components, labels):
            label.scale(0.75)
            label.match_color(component)
            self.play(
                GrowArrow(component),
                FadeIn(label)
            )
        self.wait()

        # Show sum
        sum_expr = VGroup(labels[0].copy(), Tex(R"+", font_size=24), labels[1].copy())
        sum_expr.arrange(RIGHT, buff=SMALL_BUFF)
        sum_expr.next_to(vel_vect.get_end(), UP)
        self.play(
            TransformFromCopy(labels, sum_expr[0::2]),
            Write(sum_expr[1])
        )
        self.wait()

        # Do a 90 degree rotation
        north_group = VGroup(components[0], labels[0])
        east_group = VGroup(components[1], labels[1])
        self.play(
            FadeOut(sum_expr),
            north_group.animate.shift(DR).fade(0.5),
            east_group.animate.shift(DR).fade(0.5),
        )
        self.wait()

        self.add(travler.copy().fade(0.75))
        t_rot_marks = self.show_90_degree_rotation(travler, 45 * DEG, about_point=vel_vect.get_start())
        self.wait()

        self.play(
            north_group.animate.set_fill(opacity=1).shift(UR),
            FadeOut(t_rot_marks)
        )
        n_rot_marks = self.show_90_degree_rotation(components[0], 90 * DEG, about_point=components[0].get_start())
        self.wait()
        self.play(
            east_group.animate.set_fill(opacity=1).shift(DR + DOWN)
        )
        e_rot_marks = self.show_90_degree_rotation(components[1], 0, about_point=components[1].get_start())

        self.add(components.copy().set_opacity(0.5))
        self.play(components[1].animate.shift(vel_vect.get_start() - components[1].get_start()))
        self.play(components[0].animate.shift(components[1].get_end() - components[0].get_start()))
        self.wait()
        self.play(LaggedStart(*(
            Rotate(mob, -90 * DEG, about_point=vel_vect.get_start(), run_time=6, rate_func=there_and_back_with_pause)
            for mob in [travler, *components]
        ), lag_ratio=0.05))

    def get_compass(self):
        spike = Triangle()
        spike.set_shape(0.25, 1)
        spike.move_to(ORIGIN, DOWN)
        spikes = VGroup(
            spike.copy().rotate(x * TAU / 4, about_point=ORIGIN)
            for x in range(4)
        )
        lil_spikes = spikes.copy().rotate(45 * DEG).scale(0.75)
        dot = Circle(radius=0.25)
        compass = VGroup(spikes, lil_spikes, dot)
        compass.set_stroke(width=0)
        compass.set_fill(GREY_D, 1)
        compass.set_shading(0.5, 0.5, 1)

        labels = VGroup(map(Text, "NWSE"))
        labels.scale(0.5)
        for label, spike, vect in zip(labels, spikes, compass_directions(4, start_vect=UP)):
            label.next_to(spike, np.round(vect), SMALL_BUFF)
        compass.add(labels)

        return compass

    def show_90_degree_rotation(self, mobject, start_angle, about_point, radius=1.5):
        arc = Arc(start_angle, 90 * DEG, arc_center=about_point, radius=radius)
        arc.add_tip(width=0.2, length=0.2)
        arc.set_color(YELLOW)
        label = Tex(R"90^\circ")
        midpoint = arc.pfp(0.5)
        label.next_to(midpoint, normalize(midpoint - about_point))

        self.play(LaggedStart(
            ShowCreation(arc),
            FadeIn(label),
            Rotate(mobject, 90 * DEG, about_point=about_point),
            lag_ratio=0.5
        ))

        return VGroup(arc, label)


class SimpleTwobitKet(InteractiveScene):
    def construct(self):
        group = KetGroup(BitString(2, 2))
        group.scale(2)
        self.add(group)


class ShowLinearityExample(InteractiveScene):
    def construct(self):
        # Add machine
        machine = get_blackbox_machine()
        machine[-1].set_color(TEAL)
        icon = get_quantum_computer_symbol(height=1)
        icon.next_to(machine, DOWN)

        self.add(machine, icon)
        self.wait()

        # Show a weighted sum
        kets = VGroup(
            KetGroup(BitString(n, 2))
            for n in range(4)
        )
        kets.arrange(DOWN, aligned_edge=LEFT, buff=MED_LARGE_BUFF)
        kets.set_submobject_colors_by_gradient(BLUE, BLUE_E)
        kets.next_to(machine, LEFT, MED_LARGE_BUFF)

        components = VGroup(
            DecimalNumber(n, include_sign=True, font_size=42)
            for n in normalize([1, -2, 3, -4])
        )
        for component, ket in zip(components, kets):
            component.next_to(ket, LEFT, SMALL_BUFF)

        in_group = VGroup(components, kets)
        self.play(FadeIn(in_group, RIGHT))
        self.wait()
        self.play(
            FadeOutToPoint(in_group.copy(), machine.get_left() + RIGHT, lag_ratio=0.01, run_time=2)
        )

        # Show output
        out_kets = VGroup()
        for ket in kets:
            f_group = Tex(R"f()", font_size=60)
            f_group.set_color(TEAL)
            f_group[:2].next_to(ket, LEFT, SMALL_BUFF)
            f_group[2].next_to(ket, RIGHT, SMALL_BUFF)
            out_ket = VGroup(f_group, ket.copy())
            out_kets.add(out_ket)

        out_components = components.copy()
        out_kets.next_to(machine, RIGHT, buff=1.5)
        for out_ket, component in zip(out_kets, out_components):
            component.next_to(out_ket, LEFT, SMALL_BUFF)

        out_group = VGroup(out_components, out_kets)

        self.play(
            FadeInFromPoint(out_group, machine.get_right() + LEFT, lag_ratio=0.01, run_time=2)
        )
        self.wait()

        # Highlight components
        in_rects = VGroup(map(SurroundingRectangle, components))
        out_rects = VGroup(map(SurroundingRectangle, out_components))

        self.play(
            LaggedStartMap(VFadeInThenOut, in_rects, lag_ratio=0.5),
            LaggedStartMap(VFadeInThenOut, out_rects, lag_ratio=0.5),
            run_time=5,
        )
        self.wait()


class ZGateExample(InteractiveScene):
    def construct(self):
        # Test
        z_gates = VGroup(get_blackbox_machine(label_tex="Z") for n in range(3))
        z_gates.scale(0.35)
        z_gates.arrange(DOWN, buff=LARGE_BUFF)
        z_gates.move_to(4 * LEFT + UP)
        for gate in z_gates:
            gate[-1].set_color(TEAL)
            gate[-1].scale(1.5)

        zero, one = kets = VGroup(
            KetGroup(Integer(0)),
            KetGroup(Integer(1)),
        )
        gen_input = VGroup(Tex(R"x"), zero.copy(), Tex(R"+"), Tex(R"y"), one.copy())
        gen_input.arrange(RIGHT, buff=SMALL_BUFF)
        inputs = VGroup(zero, one, gen_input)
        for in_group, gate in zip(inputs, z_gates):
            in_group.next_to(gate, LEFT)

        # Act on zero, then one
        outputs = VGroup(
            zero.copy().next_to(z_gates[0], RIGHT),
            VGroup(Tex("-"), one.copy()).arrange(RIGHT, buff=SMALL_BUFF).next_to(z_gates[1], RIGHT)
        )

        self.play(FadeIn(z_gates[0]))
        self.wait()
        self.play(FadeIn(zero, RIGHT))
        self.play(LaggedStart(
            FadeOutToPoint(zero.copy(), z_gates[0].get_center(), lag_ratio=0.05),
            FadeInFromPoint(outputs[0], z_gates[0].get_center(), lag_ratio=0.05),
            lag_ratio=0.5
        ))
        self.wait()
        self.play(
            TransformFromCopy(*z_gates[:2]),
            TransformFromCopy(*kets),
        )
        self.play(LaggedStart(
            FadeOutToPoint(one.copy(), z_gates[1].get_center(), lag_ratio=0.05),
            FadeInFromPoint(outputs[1], z_gates[1].get_center(), lag_ratio=0.05),
            lag_ratio=0.5
        ))
        self.wait()

        # General input
        self.play(
            TransformFromCopy(*z_gates[1:3]),
            TransformFromCopy(kets, gen_input[1::3]),
            *(FadeIn(gen_input[i]) for i in [0, 2, 3])
        )
        self.wait()

        rhss = VGroup(
            Tex(R"xZ|0\rangle + y Z|1\rangle ", t2c={"Z": TEAL}),
            Tex(R"x |0\rangle - y |1\rangle"),
        )
        rhss[0].next_to(z_gates[2], RIGHT)
        rhss[1].next_to(rhss[0], DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        self.play(LaggedStart(
            FadeOutToPoint(gen_input.copy(), z_gates[2].get_center(), lag_ratio=0.01),
            FadeInFromPoint(rhss[0], z_gates[2].get_center(), lag_ratio=0.01),
            lag_ratio=0.5
        ))
        self.wait()
        self.play(FadeIn(rhss[1], DOWN))
        self.wait()


class PassSuperpositionIntoVerifier(InteractiveScene):
    def construct(self):
        # Set up machine
        frame = self.frame
        frame.shift(1.5 * LEFT)
        machine = get_blackbox_machine(label_tex="V")
        machine[-1].set_color(TEAL)
        self.add(machine)

        # Bit string to general input
        bit_ket = KetGroup(Tex("0111...0011"))
        bit_ket.next_to(machine, LEFT, MED_LARGE_BUFF)

        basis_kets = VGroup(
            KetGroup(Tex(R"b_0")),
            KetGroup(Tex(R"b_1")),
            Tex(R"\vdots"),
            KetGroup(Tex(R"b_k")),
            Tex(R"\vdots"),
            KetGroup(Tex(R"b_{n-2}")),
            KetGroup(Tex(R"b_{n-1}")),
        )
        basis_kets.set_submobject_colors_by_gradient(BLUE, BLUE_E)
        basis_kets.arrange(DOWN)
        basis_kets.next_to(machine, LEFT, MED_LARGE_BUFF)
        components = VGroup(
            Tex(R"x_0"), Tex(R"+x_1"), VectorizedPoint(), Tex("+x_k"), VectorizedPoint(), Tex("+x_{n-2}"), Tex(R"+x_{n-1}")
        )
        for comp, ket in zip(components, basis_kets):
            comp.next_to(ket, LEFT, SMALL_BUFF)

        basis_label = Text("Basis")
        basis_label.next_to(bit_ket, LEFT)
        brace = Brace(components, LEFT)
        superposition_word = brace.get_text("Superposition")

        self.add(bit_ket)
        self.play(FadeIn(basis_label))
        self.wait()
        self.play(
            Transform(bit_ket, basis_kets[3]),
            basis_label.animate.next_to(basis_kets[3], LEFT),
        )
        self.wait()
        self.play(
            FadeTransform(basis_label, superposition_word),
            GrowFromCenter(brace),
            FadeIn(basis_kets, lag_ratio=0.1, run_time=2),
            FadeIn(components, lag_ratio=0.1, run_time=2),
        )
        self.remove(bit_ket)
        self.wait()

        # Show the output
        out_kets = basis_kets.copy()
        out_Vs = VGroup(
            Tex(R"V").set_color(TEAL).next_to(ket, LEFT, SMALL_BUFF)
            for ket in out_kets
        )
        VGroup(out_Vs[2], out_Vs[4]).set_opacity(0)
        out_comps = components.copy()
        out_comps.shift(out_Vs[0].get_width() * LEFT)
        out_group = VGroup(out_comps, out_Vs, out_kets)
        out_group.next_to(machine, RIGHT)

        in_group = VGroup(components, basis_kets)

        self.play(
            LaggedStart(
                FadeOutToPoint(in_group.copy(), machine.get_left() + RIGHT, lag_ratio=0.01),
                FadeInFromPoint(out_group, machine.get_right() + LEFT, lag_ratio=0.01),
                lag_ratio=0.5,
                run_time=2
            ),
            frame.animate.center(),
            FadeOut(brace),
            FadeOut(superposition_word),
        )
        self.wait()

        # Show unchanged parts
        last_annotation = VGroup()
        for n in [0, 1, -2, -1]:
            comp = out_comps[n]
            V = out_Vs[n]
            ket = out_kets[n]

            rect = SurroundingRectangle(VGroup(comp, V, ket))
            rect.set_stroke(YELLOW)
            label = Text("Unchanged", font_size=36)
            label.set_color(YELLOW)
            label.next_to(rect, RIGHT)
            annotation = VGroup(rect, label)
            self.play(LaggedStart(
                FadeOut(last_annotation),
                FadeIn(annotation),
                FadeOut(V),
                comp.animate.shift(V.get_width() * RIGHT),
            ))
            last_annotation = annotation
        self.play(FadeOut(annotation))

        # Show key solution
        sol_index = 3
        solution_label = Text("Sudoku\nSolution")
        solution_label.next_to(components[sol_index], LEFT, LARGE_BUFF)
        solution_arrow = Arrow(solution_label, components[sol_index])

        out_rect = SurroundingRectangle(
            VGroup(out_comps[sol_index], out_Vs[sol_index], out_kets[sol_index])
        )
        out_rect.set_stroke(YELLOW)
        flip_word = Text("Flip!")
        flip_word.set_color(YELLOW)
        flip_word.next_to(out_rect, RIGHT)
        new_out_comp = Tex(R"-x_k")
        new_out_comp.next_to(out_kets[sol_index], LEFT, SMALL_BUFF)

        self.play(
            FadeIn(out_rect),
            FadeIn(flip_word),
            Transform(out_comps[sol_index], new_out_comp),
            FadeOut(out_Vs[sol_index], scale=0.25),
        )
        self.play(
            Write(solution_label),
            GrowArrow(solution_arrow)
        )
        self.wait()

        # Parallelization
        lines = VGroup(
            Line(basis_kets[n].get_right(), out_comps[n].get_left(), buff=SMALL_BUFF)
            for n in range(7)
        )

        new_machines = VGroup(machine.copy().scale(0.25).move_to(line) for line in lines)

        pre_machine = VGroup(machine.copy())
        self.remove(machine)
        self.add(lines, Point(), pre_machine)
        self.play(
            Transform(pre_machine, new_machines),
            LaggedStartMap(ShowCreation, lines)
        )
        self.wait()
        self.play(
            FadeOut(lines),
            FadeOut(pre_machine),
            FadeIn(machine),
        )
        self.wait()

        # Column vector
        in_col = TexMatrix(np.array(["x_0", "x_1", R"\vdots", "x_k", R"\vdots", R"x_{n-2}", R"x_{n-1}"]).reshape(-1, 1))
        out_col = TexMatrix(np.array(["x_0", "x_1", R"\vdots", "-x_k", R"\vdots", R"x_{n-2}", R"x_{n-1}"]).reshape(-1, 1))

        for col, vect in zip([in_col, out_col], [LEFT, RIGHT]):
            col.match_height(basis_kets)
            col.next_to(machine, vect)

        self.play(
            FadeOut(solution_label),
            FadeOut(solution_arrow),
            FadeOut(basis_kets),
            FadeOut(out_kets),
            out_rect.animate.surround(out_col.get_entries()[3]).set_stroke(width=1),
            FadeOut(flip_word),
            ReplacementTransform(components, in_col.get_entries()),
            ReplacementTransform(out_comps, out_col.get_entries()),
            Write(in_col.get_brackets()),
            Write(out_col.get_brackets()),
        )
        self.wait()


class OverOrUnderExplain(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            morty.change("shruggie"),
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen),
        )
        self.wait(2)
        self.play(LaggedStart(
            stds[2].change("erm", look_at=morty.eyes),
            morty.change("hesitant", look_at=self.students),
        ))
        self.wait(2)
        self.play(
            morty.change("raise_right_hand"),
            self.change_students("hesitant", "well", "confused", look_at=3 * UR)
        )
        self.wait(5)


class IsThisUseful(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            morty.change("hesitant", self.screen),
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen)
        )
        self.wait(4)
        self.play(
            stds[1].says("Is this...useful?", mode="confused"),
            stds[0].change("hesitant", look_at=stds[1].eyes),
            stds[2].change("well", look_at=stds[1].eyes),
            morty.change("guilty", look_at=stds[1].eyes),
        )
        self.wait(4)


class SudokuBruteForce(InteractiveScene):
    def construct(self):
        # Test
        words = Text("Comically inefficient\nbrute force approach:")
        steps = TexText("$9^{60}$ Steps", isolate=["60"])
        steps.scale(1.5)
        group = VGroup(words, steps)
        group.arrange(RIGHT, buff=MED_LARGE_BUFF)
        group.to_edge(UP)

        brace = Brace(steps[:3], DOWN)
        number = Tex("{:,d}".format(9**60).replace(",", "{,}"))
        number.scale(0.75)
        number.set_color(RED)
        number.next_to(brace, DOWN).shift_onto_screen()

        self.add(group)
        self.play(
            GrowFromCenter(brace),
            FadeIn(number, lag_ratio=0.1)
        )
        self.wait()

        # Show Grover
        grover_words = TexText("Using Grover's Algorithm:")
        grover_words.move_to(words, RIGHT)
        grover_steps = TexText(R"$\displaystyle \left\lceil\frac{\pi}{4} 9^{30}\right\rceil$ Steps", isolate=["30"])
        grover_steps.move_to(steps, LEFT)
        new_brace = Brace(grover_steps[:8], DOWN)
        new_number = Tex("{:,d}".format(int(np.ceil(9**30 * PI / 4))).replace(",", "{,}"))
        new_number.scale(0.75)
        new_number.set_color(RED)
        new_number.next_to(new_brace, DOWN)

        self.play(
            FadeTransformPieces(words, grover_words),
            FadeTransform(steps, grover_steps),
            Transform(brace, new_brace),
            FadeTransformPieces(number, new_number),
        )
        self.wait()


class ShaInversionCounts(InteractiveScene):
    def construct(self):
        # Test
        classical = VGroup(
            get_classical_computer_symbol(height=1),
            TexText("$2^{256}$ Steps"),
        )
        quantum = VGroup(
            get_quantum_computer_symbol(height=1),
            TexText(R"$\displaystyle \left\lceil \frac{\pi}{4} 2^{128} \right\rceil$ Steps"),
        )
        group = VGroup(classical, quantum)
        for elem in group:
            elem.arrange(RIGHT)
        group.arrange(DOWN, buff=2.0, aligned_edge=LEFT)
        group.to_corner(UR)

        self.play(FadeIn(classical, UP))
        self.wait()
        self.play(FadeTransformPieces(classical.copy(), quantum, lag_ratio=1e-4))
        self.wait()


class SkepticalPiCreature(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph().to_edge(DOWN).shift(3 * LEFT)
        randy.body.set_color(MAROON_E)
        morty = Mortimer().to_edge(DOWN).shift(3 * RIGHT)
        morty.make_eye_contact(randy)
        for pi in [randy, morty]:
            pi.change_mode("tease")
            pi.body.insert_n_curves(100)

        self.play(LaggedStart(
            randy.says("Quantum Computing will\nchange everything!", mode="surprised"),
            morty.change("hesitant")
        ))
        self.play(Blink(morty))
        self.wait(2)
        self.play(Blink(randy))
        self.add(Point())
        self.play(
            morty.says("...will it?", mode="sassy"),
            randy.change("guilty"),
        )
        self.play(Blink(morty))
        self.wait()


class FactoringNumbers(InteractiveScene):
    def construct(self):
        # Show factoring number
        factor_values = [314159265359, 1618033988749]
        icon = get_quantum_computer_symbol(height=2).move_to(RIGHT)
        factors = VGroup(Integer(value) for value in factor_values)
        factors.arrange(RIGHT, buff=LARGE_BUFF)
        product_value = int(factor_values[0] * factor_values[1])
        product = Tex("{:,d}".format(product_value).replace(",", "{,}"))
        product.next_to(icon, LEFT)

        product.set_color(TEAL)
        factors.set_submobject_colors_by_gradient(BLUE, GREEN)

        times = Tex(R"\times")
        times.move_to(factors)

        factor_group = VGroup(factors[0], times, factors[1])
        factor_group.arrange(DOWN, SMALL_BUFF)
        factor_group.next_to(icon, RIGHT, MED_LARGE_BUFF)

        self.add(icon)
        self.play(FadeIn(product, lag_ratio=0.1))
        self.wait()
        self.play(LaggedStart(
            FadeOutToPoint(product.copy(), icon.get_center(), lag_ratio=0.02, path_arc=45 * DEG),
            FadeInFromPoint(factor_group, icon.get_center(), lag_ratio=0.02, path_arc=45 * DEG),
            lag_ratio=0.3
        ))
        self.wait()


class FourBitAdder(InteractiveScene):
    def construct(self):
        # Test
        circuit = SVGMobject("Four_bit_adder_with_carry_lookahead")
        circuit.set_height(7.0)
        circuit.set_stroke(WHITE, 1)
        circuit.set_fill(BLACK, 0)
        circuit.sort(lambda p: np.dot(p, DR))

        self.play(Write(circuit, lag_ratio=1e-2, run_time=3))
        self.wait()


class PatronScroll(PatreonEndScreen):
    pass
