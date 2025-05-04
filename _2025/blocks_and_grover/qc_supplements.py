from manim_imports_ext import *
from _2025.blocks_and_grover.state_vectors import *


def get_quantum_computer_symbol(height=2, color=GREY_B, symbol_tex=R"|Q\rangle", symbol_color=TEAL):
    chip = SVGMobject("computer_chip")
    chip.set_height(height)
    chip.to_edge(RIGHT)
    chip.set_fill(GREY_C)
    chip.set_shading(0.7, 0, 0)
    symbol = Tex(symbol_tex)
    symbol.set_fill(symbol_color)
    symbol.set_stroke(symbol_color, 1)
    symbol.set_height(0.4 * chip.get_height())
    symbol.move_to(chip)

    result = VGroup(chip, symbol)
    return result


def get_classical_computer_symbol(height=2, color=GREY_B, symbol_tex=R"\mathcal{C}", symbol_color=YELLOW):
    return get_quantum_computer_symbol(height, color, symbol_tex, symbol_color)


def get_blackbox_machine(height=2, color=GREY_D, label_tex="f(n)", label_height_ratio=0.25):
    square = Square(height)
    in_tri = ArrowTip().set_height(0.5 * height)
    out_tri = in_tri.copy().rotate(PI)
    in_tri.move_to(square.get_left())
    out_tri.move_to(square.get_right())
    machine = Union(square, in_tri, out_tri)
    machine.set_fill(color, 1)
    machine.set_stroke(WHITE, 2)

    label = Tex(label_tex)
    label.set_height(label_height_ratio * height)
    label.move_to(machine)
    machine.add(label)

    machine.output_group = VGroup()

    return machine


def get_bit_circuit(n_bits=4):
    circuit = SVGMobject("BitCircuit")
    circuit.set_stroke(WHITE, 2)
    circuit[-2:].set_fill(WHITE, 1)
    result = circuit.get_grid(1, n_bits, buff=0)
    return result


def get_magnifying_glass(height=3.0, color=GREY_D):
    glass = SVGMobject("magnifying_glass2")
    glass.set_fill(GREY_D)
    glass.set_shading(0.35, 0.15, 0.5)
    circle = VMobject().set_points(glass[0].get_subpaths()[1])
    circle.set_fill(GREEN_SCREEN, 1, border_width=3)
    circle.set_stroke(width=0)
    glass.add_to_back(circle)
    glass.set_height(height)
    return glass


def get_key_icon(height=0.5):
    key_icon = SVGMobject("key").rotate(135 * DEG)
    key_icon.set_fill(YELLOW)
    key_icon.set_height(height)
    return key_icon


class Superposition(Group):
    def __init__(
        self,
        pieces,
        offset_multiple=0.2,
        max_rot_vel=3,
        glow_color=TEAL,
        glow_stroke_range=(1, 22, 4),
        glow_stroke_opacity=0.05
    ):
        self.pieces = pieces
        self.center_points = Group(
            Point(piece.get_center())
            for piece in pieces
        )
        self.offset_multipler = ValueTracker(offset_multiple)

        for piece, point_mob in zip(pieces, self.center_points):
            piece.center_point = point_mob

            piece.offset_vect = rotate_vector(RIGHT, np.random.uniform(0, TAU))
            piece.offset_vect_rot_vel = np.random.uniform(-max_rot_vel, max_rot_vel)

        glow_strokes = np.arange(*glow_stroke_range)
        glows = pieces.replicate(len(glow_strokes))
        glows.set_fill(opacity=0)
        glows.set_joint_type('no_joint')
        for glow, sw in zip(glows, glow_strokes):
            glow.set_stroke(glow_color, width=float(sw), opacity=glow_stroke_opacity)

        self.glows = glows

        super().__init__(glows, pieces, self.center_points, self.offset_multipler)
        self.add_updater(lambda m, dt: m.update_piece_positions(dt))

    def update_piece_positions(self, dt):
        offset_multiple = self.offset_multipler.get_value()

        for piece in self.pieces:
            piece.offset_vect = rotate_vector(piece.offset_vect, dt * piece.offset_vect_rot_vel)
            piece.offset_radius = offset_multiple
            piece.move_to(piece.center_point.get_center() + piece.offset_radius * piece.offset_vect)

        for glow in self.glows:
            for sm1, sm2 in zip(glow.family_members_with_points(), self.pieces.family_members_with_points()):
                sm1.match_points(sm2)

    def set_offset_multiple(self, value):
        self.offset_multipler.set_value(value)

    def set_glow_opacity(self, opacity=0.1):
        self.glows.set_stroke(opacity=opacity)


###


class ReferenceSummary(InteractiveScene):
    def construct(self):
        # Laptop
        laptop = Laptop()
        self.frame.reorient(85, 79, 0, (-1.04, -1.42, 0.47), 6.97)
        self.add(laptop)

        # Randy
        randy = Randolph(mode="thinking", height=4)
        randy.body.insert_n_curves(100)
        randy.fix_in_frame()
        randy.next_to(ORIGIN, LEFT)
        randy.look_at(2 * RIGHT)

        self.add(randy)
        self.play(Blink(randy))
        self.wait()
        self.play(Blink(randy))
        self.play(randy.change("confused").fix_in_frame())
        self.wait(2)
        self.play(Blink(randy))
        self.wait(2)


class WriteQuantumComputingTitle(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Quantum Computing", font_size=120)
        title.move_to(UP)
        subtitle = Tex(R"\frac{1}{\sqrt{2}}\big(|0\rangle + |1\rangle \big)", font_size=90)
        subtitle.next_to(title, DOWN, LARGE_BUFF)
        title.set_color(TEAL)

        self.add(title, subtitle)
        self.play(LaggedStart(
            Flash(letter, color=TEAL, flash_radius=1.0, line_stroke_width=2)
            for letter in title
        ))


class MentionQuiz(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher

        to_confusion = VGroup(
            Vector(1.5 * RIGHT, thickness=6),
            Text("???", font_size=90).space_out_submobjects(1.5)
        )
        to_confusion.arrange(RIGHT)
        to_confusion.next_to(self.screen, RIGHT)
        to_confusion.shift(0.5 * UP + LEFT)

        for pi in self.pi_creatures:
            pi.body.insert_n_curves(1000)

        self.play(
            morty.change("angry", self.screen),
            self.change_students("thinking", "tease", "confused", look_at=self.screen),
        )
        self.wait(2)
        self.play(
            morty.change("raise_right_hand", to_confusion),
            self.change_students("erm", "guilty", "hesitant"),
            FadeInFromPoint(to_confusion, morty.get_corner(UL), lag_ratio=0.1),
        )
        self.wait()
        self.play(morty.says("Quiz time!"))
        self.wait(3)


class QuizMarks(InteractiveScene):
    def construct(self):
        marks = VGroup(
            Exmark().set_color(RED),
            Exmark().set_color(RED),
            Checkmark().set_color(GREEN),
            Exmark().set_color(RED),
        )
        marks.arrange(DOWN).scale(2)
        self.add(marks)


class SimpleScreenReference(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher

        self.play(
            morty.change("hesitant"),
            self.change_students("erm", "concentrating", "sassy", look_at=self.screen)
        )
        self.wait(5)


class StudentsCommentOnQuiz(TeacherStudentsScene):
    def construct(self):
        # Ask about quantum computer
        morty = self.teacher
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)

        qc = get_quantum_computer_symbol(height=1.5)
        qc.move_to(self.hold_up_spot, DOWN)
        qc_outline = qc.copy()
        qc_outline.set_fill(opacity=0).set_stroke(TEAL, 2)

        q_marks = VGroup(
            Text("?!?", font_size=72).next_to(std, UP)
            for std in self.students
        )

        self.play(
            morty.change("raise_right_hand"),
            FadeIn(qc, UP),
            self.change_students("confused", "horrified", "angry"),
            LaggedStartMap(FadeIn, q_marks, shift=0.25 * UP),
        )
        self.play(
            VShowPassingFlash(qc_outline, time_width=2, run_time=3),
            Blink(morty),
        )
        self.wait(4)

        # Reference alternate version of blackbox function
        std = self.students[2]
        machine = get_blackbox_machine(height=1.0)
        machine.next_to(std.get_corner(UR), UP)
        arrow = Tex(R"\updownarrow").set_height(1)
        arrow.next_to(machine, UP)

        self.play(LaggedStart(
            FadeOut(q_marks[:2], lag_ratio=0.1),
            self.change_students("pondering", "erm", "raise_right_hand"),
            morty.change("tease"),
            qc.animate.next_to(arrow, UP),
            FadeIn(machine),
            q_marks[2].animate.next_to(arrow, LEFT),
            Write(arrow),
        ))
        self.wait(5)

        # Thought bubble
        bubble = ThoughtBubble().pin_to(std)
        self.play(
            self.change_students("pondering", "pondering", "thinking", look_at=self.screen, lag_ratio=0.1),
            FadeIn(bubble, lag_ratio=0.2),
            LaggedStartMap(FadeOut, VGroup(machine, arrow, q_marks[2], qc), shift=LEFT, lag_ratio=0.1, run_time=1)
        )
        self.wait(3)


class Wrong(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer()
        morty.to_corner(DR)
        # morty.body.insert_n_curves(10000)
        self.play(morty.says("Wrong!", mode="surprised"))
        self.play(Blink(morty))
        self.wait()

        old_bubble = morty.bubble
        new_bubble = morty.get_bubble(Text("Also wrong!"), bubble_type=SpeechBubble)
        self.play(
            TransformMatchingStrings(
                old_bubble.content,
                new_bubble.content,
                key_map={"Wrong": "wrong"},
                run_time=1
            ),
            Transform(old_bubble[0], new_bubble[0]),
            morty.change("tease")
        )
        self.wait()


class WriteCheckMark(InteractiveScene):
    def construct(self):
        # Test
        check = Checkmark()
        check.set_color(GREEN)
        check.set_height(0.75)
        self.play(
            Write(check),
            Flash(check.get_left() + 1.0 * LEFT, flash_radius=0.8, line_length=0.2, color=GREEN)
        )
        self.wait()


class ExpressSkepticism(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph(height=3.5)
        randy.flip()
        randy.to_corner(DR)
        for mood in ["sassy", "pondering", "confused"]:
            self.play(randy.change(mood, ORIGIN))
            for _ in range(2):
                self.play(Blink(randy))
                self.wait(2)


class WriteGroversAlgorithm(InteractiveScene):
    def construct(self):
        text = TexText("Grover's Algorithm", font_size=60)
        self.play(Write(text))
        self.wait()


class TwoThirdsDivision(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle().set_fill(GREY_E, 0.5))

        rects = Rectangle(3, 0.5).replicate(2)
        rects.arrange(RIGHT, buff=0)
        rects.set_fill(BLUE_E, 1)
        rects.set_submobject_colors_by_gradient(BLUE_E, BLUE_D)
        rects.set_stroke(WHITE, 1)
        rects.stretch_to_fit_width(12)
        rects.move_to(2 * DOWN)

        self.add(rects)
        self.wait()
        self.play(
            rects[0].animate.stretch_to_fit_width(8, about_edge=LEFT),
            rects[1].animate.stretch_to_fit_width(4, about_edge=RIGHT),
            run_time=2
        )
        self.wait()


class ReactToStrangeness(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        randy.flip()
        randy.to_corner(UR)

        self.play(randy.change('confused', ORIGIN))
        self.play(Blink(randy))
        self.play(randy.change('erm', ORIGIN))
        for _ in range(2):
            self.play(Blink(randy))
            self.wait(2)


class DotsAndArrow(InteractiveScene):
    def construct(self):
        # Test
        dots = Tex(R"\cdots", font_size=160)
        dots.space_out_submobjects(0.8)
        arrow = Vector(2.0 * RIGHT, thickness=8)
        arrow.next_to(dots, RIGHT, MED_LARGE_BUFF)
        self.play(LaggedStart(
            FadeIn(dots, lag_ratio=0.5),
            GrowArrow(arrow),
            lag_ratio=0.7
        ))
        self.wait()


class BigCross(InteractiveScene):
    def construct(self):
        # Test
        cross = Cross(Rectangle(4, 7))
        max_width = 15
        cross.set_stroke(RED, width=[0, max_width, max_width, max_width, 1])
        self.play(ShowCreation(cross))
        self.wait()


class VectSize16(InteractiveScene):
    def construct(self):
        # Test
        brace = Brace(Line(3 * DOWN, 3 * UP), RIGHT)
        tex = brace.get_tex("2^{4} = 16")
        tex.shift(SMALL_BUFF * UR)
        self.play(GrowFromCenter(brace), Write(tex))
        self.wait()


class MoreDimensionsNote(InteractiveScene):
    def construct(self):
        # Test
        words = TexText(R"(Except $2^k$ dimensions instead of 3)")
        words.set_color(GREY_B)
        self.play(FadeIn(words, lag_ratio=1))
        self.wait()


class QuestionsOnTheStateVector(TeacherStudentsScene):
    def construct(self):
        # React
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)  # Make it 10k?
        morty = self.teacher
        stds = self.students
        self.play(
            morty.change("hesitant"),
            self.change_students("pleading", "confused", "erm", look_at=self.screen)
        )
        self.wait(3)

        # Questions
        self.play(
            stds[0].says("But...\nwhat is it?", mode="maybe", look_at=morty.eyes)
        )
        self.wait()
        self.add(Point())
        self.play(
            stds[1].says("Why square things?", mode="raise_right_hand", look_at=morty.eyes)
        )
        self.wait(3)

        # Reference Complex
        expr = Tex(R"\mathds{R}^n \text{ vs. } \mathds{C}^n")
        expr.move_to(self.hold_up_spot, DOWN)

        self.play(
            morty.change("raise_right_hand"),
            FadeIn(expr, UP),
            stds[0].debubble(mode="sassy"),
            stds[1].debubble(mode="erm"),
            stds[2].change("confused", look_at=expr)
        )
        self.wait(3)
        self.play(self.change_students("pondering", "pondering", "pondering", look_at=self.screen))
        self.wait(3)


class MagnifyingGlassOverComputer(InteractiveScene):
    def construct(self):
        # Test
        comp = get_quantum_computer_symbol(height=2.5)
        comp.center()
        glass = get_magnifying_glass()
        glass.next_to(comp, UL, buff=2).shift_onto_screen()

        self.add(comp, glass)
        self.play(
            glass.animate.shift(-glass[0].get_center()).set_anim_args(path_arc=-60 * DEG),
            rate_func=there_and_back_with_pause,
            run_time=6
        )
        self.wait()


class SimpleSampleValue(InteractiveScene):
    def construct(self):
        value = Tex(R"|0011\rangle", font_size=60)
        value.set_fill(border_width=2)
        self.add(value)


class BitVsQubitMatrix(InteractiveScene):
    def construct(self):
        # Set up frame
        boxes = Rectangle(5, 3).get_grid(2, 2, buff=0)
        boxes.to_edge(DOWN, buff=MED_LARGE_BUFF)
        boxes.shift(RIGHT)
        boxes.set_stroke(WHITE, 1)

        top_titles = VGroup(
            Text("Bit", font_size=72),
            Text("Qubit", font_size=72),
        )
        for title, box in zip(top_titles, boxes):
            title.next_to(box, UP)
            title.align_to(top_titles[0], UP)

        side_titles = VGroup(
            Text("State", font_size=48),
            Text("What you\nobserve", font_size=48),
        )
        for title, box in zip(side_titles, boxes[::2]):
            title.next_to(box, LEFT)
        side_titles[0].match_x(side_titles[1])

        for i in [0, 2, 3]:
            content = Text(R"0 or 1", font_size=72)
            content.move_to(boxes[i])
            boxes[i].add(content)

        self.add(boxes)
        self.add(top_titles)
        self.add(side_titles)


class ReferenceQubitComplexity(TeacherStudentsScene):
    def construct(self):
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(1000)
        expr = Tex(R"\mathds{R}^2 \text{ vs. } \mathds{C}^2", font_size=60)
        expr.move_to(self.hold_up_spot, DOWN)
        self.play(
            self.teacher.change("raise_right_hand"),
            self.change_students("erm", "confused", "tease", look_at=expr),
            FadeIn(expr, UP)
        )
        self.wait(2)
        self.play(
            self.teacher.change("tease"),
            FadeOut(expr, shift=3 * RIGHT, path_arc=-90 * DEG),
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen)
        )
        self.wait(4)


class ConfusionAtPresmises(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.play(
            morty.change('guilty'),
            stds[2].says("Okay...\nbut why?", mode="sassy", bubble_direction=LEFT),
            stds[1].change("confused", self.screen),
            stds[0].change("maybe", self.screen),
        )
        self.wait(2)
        self.play(self.change_students("confused", "erm", "maybe", look_at=self.screen))
        self.wait(3)


class BitExamples(InteractiveScene):
    def construct(self):
        # All bits
        circuit = get_bit_circuit(1)
        circuit.to_edge(LEFT, buff=0.75).to_edge(DOWN, buff=0.25)

        switch = SVGMobject("light_switch")
        switch.set_height(1.5)
        switch.to_corner(UL)
        switch.match_x(circuit)
        switch.set_fill(GREY_B)
        switch.flip(RIGHT)

        coins = Circle(radius=0.5).replicate(2)
        for coin, color, letter in zip(coins, [RED_E, BLUE_E], "TH"):
            coin.set_fill(color, 1)
            coin.set_stroke(WHITE, 2)
            coin.add(Text(letter).move_to(coin))
        coins.set_y(0).match_x(circuit)
        coins[1].flip(RIGHT)
        coins.apply_depth_test()
        op_tracker = ValueTracker(0)

        def update_coins(coins):
            index = int(op_tracker.get_value() > 0.5)
            coins[index].set_opacity(1)
            coins[1 - index].set_opacity(0)

        coins.add_updater(update_coins)

        self.play(
            Write(circuit, lag_ratio=1e-2),
            Write(switch, lag_ratio=1e-2),
            FadeIn(coins),
        )
        switch.flip(RIGHT)
        self.play(
            Rotate(coins, PI, axis=RIGHT),
            op_tracker.animate.set_value(1),
        )
        self.wait()


class KetDefinition(InteractiveScene):
    def construct(self):
        # Test
        square = Rectangle(1, 0.75)
        dots = Tex(R"\cdots")
        dots.space_out_submobjects(0.5)
        dots.replace(square, 0).scale(0.75)
        ket = VGroup(Ket(square), dots)
        ket.set_stroke(WHITE, 3)

        ket_name = TexText(R"``ket''", font_size=60)
        ket_name.next_to(ket, UP, buff=MED_LARGE_BUFF)

        self.add(ket)
        self.play(Write(ket_name))
        self.wait()

        # Show unit vector
        arrow = Tex(R"\longrightarrow", font_size=90)

        vector = Vector(2 * UP + RIGHT, thickness=6, fill_color=TEAL)
        vector.next_to(arrow, RIGHT, MED_LARGE_BUFF)
        brace = LineBrace(vector, DOWN, buff=SMALL_BUFF)
        brace_label = brace.get_tex("1")

        self.play(
            ket.animate.next_to(arrow, LEFT, MED_LARGE_BUFF),
            MaintainPositionRelativeTo(ket_name, ket),
            Write(arrow),
        )
        self.play(GrowArrow(vector))
        self.play(
            GrowFromCenter(brace),
            FadeIn(brace_label)
        )
        self.wait()

        # Show examples of what goes inside it
        examples = VGroup(
            Tex(tex)
            for tex in [
                R"0",
                R"1",
                "+z",
                "-z",
                R"\updownarrow",
                R"E",
                R"\text{Dead}",
                R"\text{Alive}",
                R"\psi",
            ]
        )
        last = dots
        for example in examples:
            example.scale(2)
            example.set_max_width(dots.get_width())
            example.move_to(last)
            self.play(
                FadeOut(last, 0.5 * UP),
                FadeIn(example, 0.5 * UP),
                rate_func=linear,
                run_time=0.5
            )
            last = example


class ClassicalGates(InteractiveScene):
    def construct(self):
        # Title
        title = Text("Logic Gates", font_size=60)
        title.to_edge(UP)
        self.add(title)

        # Basic gates
        gates = VGroup(
            SVGMobject("and_gate"),
            SVGMobject("or_gate"),
            SVGMobject("not_gate"),
        )
        gates.get_width()
        gates.set_width(2.0)
        gates.set_fill(GREY_B)
        gates.arrange(RIGHT, buff=2)
        gates.set_y(0)

        gate_names = VGroup(map(Text, ["AND", "OR", "NOT"]))
        for name, gate in zip(gate_names, gates):
            name.set_color(GREY_B)
            name.next_to(gate, DOWN)

        self.play(
            LaggedStartMap(Write, gates, lag_ratio=0.5),
            LaggedStartMap(Write, gate_names, lag_ratio=0.5),
        )

        # Bit examples
        inputs = VGroup(Text("01"), Text("01"), Text("0"))
        outputs = VGroup(Text("0"), Text("1"), Text("1"))
        for in_bits, out_bit, gate in zip(inputs, outputs, gates):
            in_bits.arrange(DOWN)
            in_bits.arrange_to_fit_height(1)
            in_bits.next_to(gate, LEFT, buff=SMALL_BUFF)
            out_bit.next_to(gate, RIGHT, buff=SMALL_BUFF)

        self.play(LaggedStartMap(FadeIn, inputs, shift=0.5 * RIGHT, lag_ratio=0.2, run_time=1.0)),
        self.play(LaggedStart(
            *(FadeOut(in_bit.copy(), RIGHT) for in_bit in inputs),
            *(FadeIn(out_bit, 0.5 * RIGHT) for out_bit in outputs),
            lag_ratio=0.2
        ))
        self.add(inputs, outputs)

        # Show full circuit
        gate_groups = VGroup(
            VGroup(gate, name, in_bits, out_bits)
            for gate, name, in_bits, out_bits in zip(gates, gate_names, inputs, outputs)
        )

        circuit = SVGMobject("Four_bit_adder_with_carry_lookahead")
        circuit.set_height(5)
        circuit.to_edge(DOWN, buff=0.25)
        circuit.set_fill(opacity=0)
        circuit.set_stroke(WHITE, 1)

        self.play(
            gate_groups.animate.scale(0.35).space_out_submobjects(0.9).next_to(title, DOWN, MED_LARGE_BUFF),
            Write(circuit, run_time=3)
        )
        self.wait()


class HLine(InteractiveScene):
    def construct(self):
        line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        line.set_stroke(GREY_B, 2)
        self.play(ShowCreation(line))
        self.wait()


class ShowQubitThroughTwoHadamardGates(InteractiveScene):
    def construct(self):
        # Add the line
        line = Line(4 * LEFT, 4 * RIGHT)
        h_box = Square(1)
        h_box.set_fill(BLACK, 1).set_stroke(WHITE)
        h_box.add(Text("H", font_size=60))
        h_boxes = h_box.replicate(2)
        h_boxes.arrange(RIGHT, buff=3)

        self.add(line, Point(), h_boxes)

        # Symbols
        symbols = VGroup(
            Tex(R"|0\rangle", font_size=60),
            Tex(R"\frac{1}{\sqrt{2}}\left(|0\rangle + |1\rangle\right)"),
            Tex(R"|0\rangle", font_size=60),
        )
        symbols[0].next_to(line, LEFT, MED_SMALL_BUFF)
        symbols[1].next_to(line, UP, LARGE_BUFF)
        symbols[2].next_to(line, RIGHT, MED_SMALL_BUFF)

        mid_sym_rect = SurroundingRectangle(symbols[1], buff=0.2)
        mid_sym_lines = VGroup(
            Line(line.pfp(0.5 + 0.01 * u), mid_sym_rect.get_corner(DOWN + u * RIGHT))
            for u in [-1, 1]
        )
        VGroup(mid_sym_lines, mid_sym_rect).set_stroke(BLUE, 2)

        # Show progression
        dot = GlowDot(color=BLUE, radius=0.5)
        dot.move_to(line.get_left())
        dot.set_opacity(0)

        cover = FullScreenFadeRectangle()
        cover.set_fill(BLACK, 1)
        cover.stretch(0.45, 0, about_edge=RIGHT)

        self.add(line, dot, h_boxes, cover)

        self.play(FadeIn(symbols[0], 0.5 * UR, run_time=1))
        self.play(dot.animate.set_opacity(1))
        self.wait()
        self.play(dot.animate.move_to(line.get_center()))
        self.play(
            ShowCreation(mid_sym_lines, lag_ratio=0),
            GrowFromPoint(mid_sym_rect, dot.get_center()),
            GrowFromPoint(symbols[1], dot.get_center()),
        )
        self.wait()
        self.play(FadeOut(cover))
        self.play(dot.animate.move_to(line.get_end()))
        self.play(FadeIn(symbols[2], 0.25 * RIGHT, rate_func=rush_from))
        self.wait()


class ReferencePreview(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(1000)

        self.play(
            morty.says("First, a preview"),
            self.change_students("happy", "tease", "happy", look_at=morty.eyes)
        )
        self.wait(4)
        self.play(
            morty.debubble(mode="raise_right_hand"),
            self.change_students("pondering", "thinking", "pondering", look_at=2 * UR)
        )
        self.look_at(3 * UR)
        self.wait(4)

        # Ask about sign
        stds = self.students
        self.play(
            stds[2].says("Why can you flip the\nkey sign like that?", mode="raise_left_hand", look_at=self.teacher.eyes),
            stds[1].change("sassy", look_at=3 * UR),
            stds[0].change("maybe", look_at=3 * UR),
            morty.change('tease')
        )
        self.wait(5)


class GroverPreviewBox(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        rect = ScreenRectangle(height=4.0)
        rect.next_to(morty, UL)
        rect.set_fill(BLACK, 1).set_stroke(WHITE, 2)

        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)
        for pi in self.students:
            pi.change_mode("confused").look_at(rect)

        self.add(rect)
        self.play(
            morty.change("raise_right_hand", rect),
            self.change_students("pondering", "pondering", "pondering", look_at=rect)
        )
        self.wait(4)
        self.play(self.change_students("tease", "thinking", "pondering", look_at=rect))
        self.wait(5)


class SimpleMagnifyingGlass(InteractiveScene):
    def construct(self):
        # Test
        glass = get_magnifying_glass(height=3)
        glass.to_corner(UL)
        glass.target = glass.generate_target()
        glass.target.scale(1.75)
        glass.target.shift(1.5 * LEFT - glass.target[0].get_center())

        self.add(glass)
        self.wait()
        self.play(
            MoveToTarget(glass, path_arc=-45 * DEG),
            run_time=6,
            rate_func=there_and_back_with_pause,
        )
        self.wait()


class ShowAbstractionArrows(InteractiveScene):
    def construct(self):
        arrows = VGroup(
            Arrow(point + 4 * LEFT, ORIGIN + 0.35 * point, buff=1.0, thickness=5)
            for point in np.linspace(2.5 * DOWN, 2.5 * UP, 3)
        )
        arrows.set_fill(border_width=2)
        self.play(LaggedStartMap(GrowArrow, arrows, lag_ratio=0.1))


class WriteGroversAlgorithm2(InteractiveScene):
    def construct(self):
        text = TexText("Grover's Algorithm", font_size=72)
        text.to_edge(UP)
        self.play(Write(text))
        self.wait()


class StareAtPicture(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)

        self.play(
            morty.change("raise_right_hand", self.screen),
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen),
        )
        self.wait()
        self.play(
            morty.change("speaking", look_at=self.students),
            self.change_students("tease", "well", "coin_flip_2")
        )
        self.wait(5)


class TryingToDescribeComputing(InteractiveScene):
    def construct(self):
        # Characters
        randy = Randolph()
        randy.move_to(2 * DOWN + 3 * LEFT)
        buddy = PiCreature(color=MAROON_E).flip()
        buddy.next_to(randy, RIGHT, buff=3)
        randy.make_eye_contact(buddy)

        for pi in [randy, buddy]:
            pi.body.insert_n_curves(100)

        self.add(randy, buddy)

        # Objects
        laptop = Laptop()
        laptop.scale(0.75)
        laptop.rotate(70 * DEG, LEFT)
        laptop.rotate(45 * DEG, UP)
        laptop.move_to(UP + 0.5 * LEFT)

        chip = get_classical_computer_symbol(height=1)
        chip.next_to(randy, UR, MED_LARGE_BUFF)

        # Show objects
        self.play(LaggedStart(
            randy.change("raise_right_hand", laptop),
            FadeIn(laptop, UP),
            buddy.change("erm", laptop),
            lag_ratio=0.5,
        ))
        self.play(Blink(buddy))
        self.wait()
        self.play(LaggedStart(
            randy.change("well", buddy.eyes),
            FadeIn(chip, UP),
            laptop.animate.shift(1.5 * UP),
            buddy.change("confused"),
            lag_ratio=0.25
        ))
        self.play(Blink(randy))

        # Show factoring numbers
        factors = Tex(R"91 = 7 \times 13")
        factors.next_to(randy, UL, MED_LARGE_BUFF)
        factors.shift_onto_screen(buff=LARGE_BUFF)
        self.play(
            randy.change("raise_left_hand", factors),
            FadeInFromPoint(factors, chip.get_center(), lag_ratio=0.05),
        )
        self.play(Blink(buddy))
        self.wait(2)

        # Put numbers in chip
        c_label = chip[1]
        chip.remove(c_label)

        seven = factors["7"][0].copy()
        product = factors[R"7 \times 13"][0].copy()

        self.play(
            randy.change('raise_right_hand', chip),
            chip.animate.scale(2, about_edge=DOWN),
            FadeOut(c_label, 0.5 * UP),
            FadeOut(laptop, UP),
            buddy.change("pondering", chip),
        )
        self.play(
            seven.animate.move_to(chip).scale(1.5),
        )
        self.play(Blink(buddy))
        self.wait()

        product.move_to(chip)
        self.play(
            ReplacementTransform(seven, product[0]),
            Write(product[1:]),
            buddy.change('hesitant', chip),
        )
        result = Tex(R"91")
        result.scale(1.5)
        result.move_to(chip)
        self.play(
            TransformFromCopy(product, result, lag_ratio=0.2),
            product.animate.set_opacity(0.25),
        )
        self.play(Blink(randy))
        self.wait()

        # Logic gates
        gates = SVGMobject("Four_bit_adder_with_carry_lookahead")
        gates.set_height(4)
        gates.to_edge(UP, MED_SMALL_BUFF)
        gates.to_edge(LEFT)
        gates.set_fill(opacity=0)
        gates.set_stroke(WHITE, 1)

        self.play(
            randy.change("dance_3", gates),
            Write(gates, run_time=2),
            FadeOut(factors, DOWN),
            buddy.change("awe", gates)
        )
        self.play(Blink(buddy))
        self.play(Blink(randy))
        self.wait()
        self.play(
            randy.change("tease"),
            FadeOut(gates, 3 * LEFT, rate_func=running_start, path_arc=30 * DEG)
        )
        self.wait()


class ProbForMillionDim(InteractiveScene):
    def construct(self):
        # Context
        context = VGroup(
            Tex(R"N = 2^{20}"),
            TexText(R"\# Reps: 804"),
        )
        context.arrange(DOWN, aligned_edge=LEFT)
        context.to_corner(UL)

        # Prob
        theta = math.asin(2**(-10))
        prob = math.sin((2 * 804 + 1) * theta)**2

        chance_lhs = Tex(R"P(k) = ", t2c={"k": YELLOW})
        chance_rhs = DecimalNumber(100 * prob, num_decimal_places=7, unit="%")
        chance_rhs.next_to(chance_lhs, RIGHT, SMALL_BUFF).shift(0.05 * UR)
        chance = VGroup(chance_lhs, chance_rhs)
        chance.scale(1.25)
        chance.to_edge(RIGHT)

        brace = Brace(chance_rhs, DOWN)
        equation = brace.get_tex(R"\sin((2 \cdot 804 + 1) \theta)^2")
        equation.set_color(GREY_C)
        equation.scale(0.7, about_edge=UP)

        self.play(Write(chance))
        self.play(
            GrowFromCenter(brace),
            FadeIn(equation),
        )
        self.wait()


class AskAreYouSure(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        randy.to_edge(LEFT).shift(2 * DOWN)

        number = Integer(random.randint(0, int(1e6)))
        answer = KetGroup(number)
        answer.set_width(2)
        answer.move_to(4 * RIGHT)

        self.add(answer)
        self.add(randy)
        self.play(randy.change('pondering', answer))
        self.wait()
        self.play(randy.says("Are we...sure?", "hesitant"))
        self.play(Blink(randy))
        self.wait()

        # Show the machine
        machine = get_blackbox_machine()
        machine.to_edge(UP)
        output = Text("True", font_size=60)
        output.set_color(GREEN)
        output.next_to(machine, RIGHT, MED_LARGE_BUFF)

        self.play(
            randy.debubble("pondering", machine),
            FadeOut(answer[0]),
            number.animate.scale(1.5).next_to(machine, LEFT).set_anim_args(path_arc=-45 * DEG),
            FadeIn(machine),
        )

        self.add(number.copy().set_opacity(0.5))
        self.play(
            FadeOutToPoint(number, machine.get_center(), path_arc=-45 * DEG, lag_ratio=0.01)
        )
        self.play(
            FadeIn(output, 2 * RIGHT),
            randy.change("well", output),
        )
        self.play(Blink(randy))
        self.wait(2)


class WrapUpList(InteractiveScene):
    def construct(self):
        # List
        morty = Mortimer()
        morty.to_corner(DR)

        items = BulletedList(
            "The Lie",
            R"Why $\sqrt{\quad}$",
            "A suprising analogy",
            buff=0.75
        )
        items[1][-2:].shift(SMALL_BUFF * UP)
        items[1][1:].shift(0.5 * SMALL_BUFF * DOWN)
        items.scale(1.25)
        items.to_edge(UP, LARGE_BUFF)

        morty.body.insert_n_curves(100)

        self.play(
            morty.change("raise_right_hand", items),
            FadeInFromPoint(items[0], morty.get_corner(UL), lag_ratio=0.01)
        )
        self.play(Blink(morty))
        self.wait()
        self.play(
            morty.change("pondering", items[1]),
            Write(items[1]),
        )
        self.wait()
        self.play(
            morty.change("tease", 5 * UR),
            FadeIn(items[2], DOWN),
        )
        self.play(Blink(morty))
        self.wait()

        for n, mode in enumerate(["guilty", "tease", "surprised"]):
            self.play(
                items.animate.fade_all_but(n, scale_factor=0.5),
                morty.change(mode, items[n]),
            )
            self.play(Blink(morty))
            self.wait()


class WriteShorName(InteractiveScene):
    def construct(self):
        name = Text("Peter Shor", font_size=72)
        name.to_corner(UR)
        self.play(Write(name, run_time=3))
        self.wait()


class OneWordSummary(InteractiveScene):
    def construct(self):
        # Replace "Parallelism" with "Pythagoras"
        prompt = Text("Source of the speed-up", font_size=72)
        prompt.move_to(2 * UP)

        words = VGroup(
            Text("Parallelism").set_color(TEAL),
            Text("Pythagoras").set_color(YELLOW),
        )
        words.scale(2)
        words.next_to(prompt, DOWN, LARGE_BUFF)
        strike = Line(words[0].get_left(), words[0].get_right())
        strike.set_stroke(RED, 10)

        self.add(prompt)
        self.add(words[0])
        self.wait()
        self.play(ShowCreation(strike))
        self.play(
            FadeIn(words[1], 0.5 * DOWN),
            words[0].animate.scale(0.75).shift(2 * DOWN).set_opacity(0.25),
            strike.animate.scale(0.75).shift(2 * DOWN),
        )
        self.wait()


class PythagoreanIntuition(InteractiveScene):
    def construct(self):
        # Set up axes
        x_range = y_range = z_range = (-2, 2)
        axes = ThreeDAxes(x_range, y_range, z_range)
        plane = NumberPlane(x_range, y_range)
        plane.fade(0.5)
        axes_group = VGroup(plane, axes)
        axes_group.scale(2)

        # Trace square
        frame = self.frame
        square = Square(2)
        square.move_to(ORIGIN, DL)
        square.set_stroke(WHITE, 2)

        side_lines = VGroup(
            Line(axes.get_origin(), axes.c2p(1, 0, 0)),
            Line(axes.c2p(1, 0, 0), axes.c2p(1, 1, 0)),
            Line(axes.c2p(1, 1, 0), axes.c2p(1, 1, 1)),
        )
        side_lines.set_stroke(RED, 4)
        ones = VGroup(
            Tex(R"1", font_size=36).next_to(line, vect, SMALL_BUFF)
            for line, vect in zip(side_lines, [DOWN, RIGHT, RIGHT])
        )
        ones[2].rotate(90 * DEG, RIGHT)

        dot = GlowDot(color=RED)
        dot.move_to(ORIGIN)

        frame.set_height(5).move_to(square)
        self.add(square, dot)
        for line, one in zip(side_lines[:2], ones):
            self.play(
                ShowCreation(line),
                FadeIn(one, 0.5 * line.get_vector()),
                dot.animate.move_to(line.get_end())
            )
        self.wait()
        self.play(
            MoveAlongPath(dot, square, rate_func=lambda t: 1 - 0.5 * smooth(t))
        )

        # Show diagonal
        diag = Line(square.get_corner(DL), square.get_corner(UR))
        diag.set_stroke(PINK, 3)

        sqrts = VGroup(
            Tex(R"\sqrt{1^2 + 1^2}", font_size=24),
            Tex(R"\sqrt{2}", font_size=36),
        )
        for sqrt in sqrts:
            sqrt.next_to(diag.pfp(0.5), UL, buff=0.05)

        self.play(
            ShowCreation(diag),
            dot.animate.move_to(square.get_corner(UR)),
            TransformFromCopy(ones[:2], sqrts[0]["1"], time_span=(1, 2)),
            *(
                Write(sqrts[0][tex], time_span=(1, 2))
                for tex in [R"\sqrt", "+", "2"]
            ),
            run_time=2,
        )
        self.wait()
        self.play(
            TransformMatchingTex(*sqrts, key_map={"1^2 + 1^2": "2"}, run_time=1)
        )
        self.wait()

        # Bring it up to a cube
        axes_group.set_z_index(-1)
        cube = VCube(2)
        cube.move_to(ORIGIN, DL + IN)
        cube.set_stroke(WHITE, 2)
        cube.set_fill(opacity=0)

        self.add(cube, side_lines[:2])
        self.play(
            FadeIn(axes_group),
            ShowCreation(cube, lag_ratio=0.1, time_span=(0.5, 2.0)),
            frame.animate.reorient(-16, 68, 0, (0.45, 0.98, 1.05), 4.36),
            run_time=2
        )
        frame.add_ambient_rotation(DEG)
        line = side_lines[2]
        self.play(
            ShowCreation(line),
            FadeIn(ones[2], 0.5 * line.get_vector()),
            dot.animate.move_to(line.get_end())
        )
        self.wait(2)

        # Show three dimensional diagonal
        diag3 = Line(axes.c2p(0, 0, 0), axes.c2p(1, 1, 1))
        diag3.set_stroke(YELLOW, 3)

        new_sqrts = VGroup(
            Tex(R"\sqrt{\sqrt{2}^2 + 1^2}", font_size=24),
            Tex(R"\sqrt{3}", font_size=36),
        )
        for sqrt in new_sqrts:
            sqrt.rotate(90 * DEG, RIGHT)
            sqrt.next_to(diag3.get_center(), LEFT + OUT, SMALL_BUFF)

        self.play(ShowCreation(diag3, run_time=2))
        self.play(
            TransformFromCopy(ones[2], new_sqrts[0]["1"][0], time_span=(1, 2)),
            TransformFromCopy(sqrts[1], new_sqrts[0][R"\sqrt{2}"][0], time_span=(1, 2)),
            *(
                Write(new_sqrts[0][tex], time_span=(1, 2))
                for tex in [R"\sqrt", "+", "2"]
            ),
            run_time=2,
        )
        self.wait()
        self.play(
            TransformMatchingTex(
                *new_sqrts,
                key_map={R"\sqrt{2}^2 + 1^2": "3"},
                match_animation=FadeTransform,
                run_time=1,
            )
        )
        self.wait(6)

        # Show observables
        symbols = VGroup(ones, sqrts[1], new_sqrts[1])
        wireframe = VGroup(cube, side_lines, diag, diag3)

        basis_vectors = VGroup(
            Vector(2 * v, thickness=4, fill_color=color)
            for v, color in zip(np.identity(3), [BLUE_E, BLUE_D, BLUE_C])
        )
        basis_vectors.set_z_index(1)
        for vector in basis_vectors:
            vector.always.set_perpendicular_to_camera(frame)

        obs_labels = VGroup(
            KetGroup(Text(f"Obs {n}", font_size=30), height_scale_factor=1.5, buff=0.05)
            for n in range(1, 4)
        )
        obs_labels[2].rotate(90 * DEG, RIGHT)
        for vector, label, nudge in zip(basis_vectors, obs_labels, [UP, RIGHT, RIGHT]):
            label.next_to(vector.get_end(), vector.get_vector() + nudge, buff=0.05)

        self.add(Point(), basis_vectors)
        self.play(
            LaggedStartMap(GrowArrow, basis_vectors, lag_ratio=0.25),
            FadeOut(symbols),
            FadeOut(dot),
            wireframe.animate.set_stroke(opacity=0.2),
            frame.animate.reorient(13, 67, 0, (-0.04, 0.76, 0.87), 4.84),
        )
        self.play(LaggedStartMap(FadeIn, obs_labels, lag_ratio=0.25))
        self.wait(4)

        new_cube = cube.copy()
        new_cube.deactivate_depth_test()
        new_cube.set_z_index(0)
        new_cube.set_stroke(WHITE, 3, 1)
        self.play(
            Write(new_cube, stroke_width=5, lag_ratio=0.1, run_time=3),
        )
        self.play(FadeOut(new_cube))
        self.wait(4)

        # Show many diagonal directions
        diag_vects = VGroup(
            Vector(2 * normalize(np.array(tup)))
            for tup in it.product(* 3 * [[-1, 0, 1]])
            if get_norm(tup) > 0
        )
        for vect in diag_vects:
            vect.set_perpendicular_to_camera(frame)
            color = random_bright_color(
                hue_range=(0.4, 0.5),
                saturation_range=(0.5, 0.7),
                luminance_range=(0.5, 0.6)
            )
            vect.set_color(color)

        self.play(
            FadeOut(obs_labels),
            LaggedStartMap(GrowArrow, diag_vects),
            frame.animate.reorient(-19, 61, 0, (-0.17, 0.22, -0.25), 6.96),
            run_time=3
        )
        self.wait(10)


class GrainOfSalt(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)

        self.play(
            morty.change("coin_flip_2"),
            self.change_students("hesitant", "sassy", "tease", look_at=self.screen),
        )
        self.wait(2)
        self.play(
            morty.says("Take this with\na grain of salt", mode="hesitant"),
            self.change_students("pondering", "pondering", "pondering", look_at=morty.eyes),
        )
        self.wait(3)
        self.play(
            morty.debubble(mode="raise_right_hand", look_at=3 * UP),
            self.change_students("hesitant", "erm", "hesitant", look_at=2 * UP)
        )
        self.wait(4)

        # More pondering
        self.play(
            morty.change("tease"),
            self.change_students("pondering", "thinking", "thinking", look_at=2 * UP)
        )
        self.wait(5)


class PatronScroll(PatreonEndScreen):
    def construct(self):
        # Title
        title = Text("Special thanks to\nthese supporters")
        title.to_corner(UR).shift(LEFT)
        title.set_color(BLUE)
        underline = Underline(title)
        rect = BackgroundRectangle(VGroup(title, underline))
        rect.set_fill(BLACK, 1)
        rect.scale(2, about_edge=DOWN)

        morty = Mortimer(height=1)
        morty.next_to(title, LEFT)
        morty.flip()

        for mob in rect, title, underline, morty:
            mob.fix_in_frame()
            self.add(mob)

        # Names
        names = self.get_names()
        name_mobs = VGroup(Text(name) for name in names)
        name_mobs.scale(0.5)
        for mob in name_mobs:
            mob.set_max_width(4)
        name_mobs.arrange(DOWN, aligned_edge=LEFT)
        name_mobs.next_to(underline.get_left(), DR).shift(0.5 * RIGHT)
        name_mobs.set_z_index(-1)

        self.add(name_mobs)

        # Scroll
        frame = self.frame
        dist = - name_mobs.get_y(DOWN) - 1
        total_time = 37
        velocity = dist / total_time
        frame.clear_updaters()
        frame.add_updater(lambda m, dt: m.shift(velocity * dt * DOWN))

        self.play(morty.change("gracious", name_mobs).fix_in_frame())
        for x in range(total_time):
            if random.random() < 0.15:
                self.play(Blink(morty))
            else:
                self.wait()


class ConstructQRCode2(InteractiveScene):
    def construct(self):
        # Test
        code = SVGMobject("channel_support_QR_code")
        code.set_fill(BLACK, 1)
        code.set_height(4)
        background = SurroundingRectangle(code, buff=0.25)
        background.set_fill(GREY_A, 1)
        background.set_stroke(width=0)
        background.set_z_index(-1)

        squares = code[:-6]
        corner_pieces = code[-6:]

        squares.shuffle()
        squares.sort(get_norm)
        squares.set_fill(interpolate_color(BLUE_E, BLACK, 0.5), 1)

        union = Union(*squares.copy().space_out_submobjects(0.99)).scale(1 / 0.99)
        union.set_stroke(WHITE, 2)
        union_pieces = VGroup(
            VMobject().set_points(path)
            for path in union.get_subpaths()
        )
        union_pieces.submobjects.sort(key=lambda m: -len(m.get_points()))
        union_pieces.note_changed_family()
        union_pieces.set_stroke(WHITE, 1)
        union_pieces.set_anti_alias_width(3)

        # New
        frame = self.frame
        frame.set_height(3)
        self.add(background, union_pieces, squares, corner_pieces)
        self.play(
            frame.animate.to_default_state(),
            ShowCreation(
                union_pieces,
                lag_ratio=0,
            ),
            # Write(squares, lag_ratio=0.1, time_span=(10, 20)),
            FadeIn(background, time_span=(20, 25)),
            Write(squares, stroke_color=BLUE, stroke_width=1, time_span=(12, 25)),
            Write(corner_pieces, time_span=(20, 25)),
            run_time=25,
        )
        self.play(
            FadeOut(union_pieces),
            squares.animate.set_fill(BLACK, 1),
        )
        squares.shuffle()
        self.play(LaggedStart(
            *(
                Rotate(square, 90 * DEG)
                for square in squares
            ),
            lag_ratio=0.02,
            run_time=10
        ))

        # Old
        return

        squares.save_state()
        squares.arrange_in_grid(buff=0)
        squares.move_to(background)
        squares.shuffle()
        for square in squares:
            dot = Dot()
            dot.set_fill(BLACK)
            dot.replace(square)
            square.set_points(dot.get_points())
        squares.set_stroke(WHITE, 1)
        squares.set_fill(opacity=0)

        background.save_state()
        background.set_fill(GREY_D)
        background.match_height(squares.saved_state)

        self.play(
            Restore(background),
            Restore(squares, lag_ratio=0.1),
            Write(corner_pieces, time_span=(18, 23), lag_ratio=0.25),
            run_time=35,
        )
