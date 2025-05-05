from manim_imports_ext import *
from _2025.blocks_and_grover.qc_supplements import *


class BitString(VGroup):
    def __init__(self, value, length=4, buff=SMALL_BUFF):
        self.length = length
        bit_mob = Integer(0)
        super().__init__(bit_mob.copy() for n in range(length))
        self.arrange(RIGHT, buff=buff)
        self.set_value(value)

    def set_value(self, value):
        bits = bin(value)[2:].zfill(self.length)
        for mob, bit in zip(self, bits):
            mob.set_value(int(bit))


class Ket(Tex):
    def __init__(self, mobject, height_scale_factor=1.25, buff=SMALL_BUFF):
        super().__init__(R"| \rangle")
        self.set_height(height_scale_factor * mobject.get_height())
        self[0].next_to(mobject, LEFT, buff)
        self[1].next_to(mobject, RIGHT, buff)


class KetGroup(VGroup):
    def __init__(self, mobject, **kwargs):
        ket = Ket(mobject, **kwargs)
        super().__init__(ket, mobject)


class RandomSampling(Animation):
    def __init__(
        self,
        mobject: Mobject,
        samples: list,
        weights: list[float] | None = None,
        **kwargs
    ):
        self.samples = samples
        self.weights = weights
        super().__init__(mobject, **kwargs)

    def interpolate(self, alpha: float) -> None:
        if self.weights is None:
            target = random.choice(self.samples)
        else:
            target = random.choices(self.samples, self.weights)[0]
        self.mobject.set_submobjects(target.submobjects)


class ContrstClassicalAndQuantum(InteractiveScene):
    def construct(self):
        # Titles
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

        # Bits
        frame = self.frame
        value = ord('C')
        short_boxed_bits = self.get_boxed_bits(12, 4)
        boxed_bits = self.get_boxed_bits(value, 8)
        for group in short_boxed_bits, boxed_bits:
            group.match_x(classical)
        boxes, bits = boxed_bits

        self.add(short_boxed_bits)
        self.wait()
        self.play(
            FadeOut(v_line, shift=2 * RIGHT),
            FadeOut(quantum, shift=RIGHT),
            ReplacementTransform(short_boxed_bits, boxed_bits),
            frame.animate.match_x(classical),
        )
        self.wait()

        # Draw layers of abstraction
        layers = Rectangle(8.0, 1.5).replicate(3)
        layers.arrange(UP, buff=0)
        layers.set_stroke(width=0)
        layers.set_fill(opacity=0.5)
        layers.set_submobject_colors_by_gradient(BLUE_E, BLUE_D, BLUE_C)
        layers.set_z_index(-1)
        layers.move_to(boxes)

        layers_name = Text("Layers\nof\nAbstraction", alignment="LEFT")
        layers_name.next_to(layers, RIGHT)

        layer_names = VGroup(
            Text("Hardware"),
            Text("Bits"),
            Text("Data types"),
        )
        layer_names.set_fill(GREY_B)
        layer_names.scale(0.6)
        for name, layer in zip(layer_names, layers):
            name.next_to(layer, LEFT, MED_SMALL_BUFF)

        num_mob = Integer(value)
        num_mob.move_to(layers[2])
        character = Text(f"'{chr(value)}'")
        character.move_to(layers[2]).shift(0.75 * RIGHT)

        circuitry = get_bit_circuit(4)
        circuitry.set_height(layers[2].get_height() * 0.7)
        circuitry.move_to(layers[0])

        self.play(
            LaggedStartMap(FadeIn, layers, lag_ratio=0.25, run_time=1),
            FadeIn(layers_name, lag_ratio=1e-2),
            Write(layer_names[1]),
        )
        self.play(
            LaggedStart(
                (TransformFromCopy(bit, num_mob)
                for bit in bits),
                lag_ratio=0.02,
            ),
            FadeIn(layer_names[2], UP),
        )
        self.wait()
        self.play(
            num_mob.animate.shift(0.75 * LEFT),
            FadeIn(character, 0.5 * RIGHT)
        )
        self.wait()
        self.play(
            Write(circuitry),
            FadeIn(layer_names[0], DOWN)
        )
        self.wait()

        # Extend layers to the quantum case
        new_layers_name = Text("Layers of Abstraction")
        new_layers_name.next_to(layers, DOWN)
        new_layers_name.match_x(quantum)

        layers.target = layers.generate_target()
        layers.target.set_width(FRAME_WIDTH, stretch=True)
        layers.target.set_x(0)

        layer_names.generate_target()
        for name, layer in zip(layer_names.target, layers.target):
            name.next_to(layer.get_left(), RIGHT, MED_SMALL_BUFF)

        self.play(
            LaggedStart(
                frame.animate.set_x(0),
                MoveToTarget(layer_names),
                FadeIn(quantum, RIGHT),
                ShowCreation(v_line),
                Transform(layers_name, new_layers_name),
                Group(num_mob, character, boxed_bits).animate.shift(RIGHT),
                circuitry.animate.scale(0.8).shift(RIGHT)
            ),
            MoveToTarget(layers, lag_ratio=0.01),
        )
        self.wait()

        # Show quantum material
        qubit_string = BitString(0, length=8)
        qubit_string.set_value(ord("Q"))
        qubit_ket = Ket(qubit_string)
        qubits = VGroup(qubit_ket, qubit_string)

        qunit_num = Integer(ord("Q"))
        qunit_ket = Ket(qunit_num)
        qunit = VGroup(qunit_ket, qunit_num)

        ion = Group(
            GlowDot(color=RED, radius=0.5),
            Dot(radius=0.1).set_fill(RED, 0.5),
            Tex(R"+", font_size=14).set_fill(border_width=1)
        )
        trapped_ions = Group(ion.copy().shift(x * RIGHT) for x in np.linspace(0, 4, 8))

        for mob, layer in zip([trapped_ions, qubits, qunit], layers):
            mob.move_to(layer).match_x(quantum)

        for ion, bit in zip(trapped_ions, qubit_string):
            if bit.get_value() == 1:
                ion[0].set_opacity(0)

        self.play(LaggedStartMap(FadeIn, trapped_ions))
        self.wait()
        self.play(FadeIn(qubits, UP))
        self.wait()
        self.play(
            TransformFromCopy(qubits, qunit)
        )
        self.wait()
        value_tracker = ValueTracker(ord("Q"))
        for value in [ord('C'), ord('Q')]:
            self.play(
                value_tracker.animate.set_value(value),
                UpdateFromFunc(qunit_num, lambda m: m.set_value(int(value_tracker.get_value()))),
                UpdateFromFunc(qubit_string, lambda m: m.set_value(int(value_tracker.get_value()))),
                rate_func=linear,
                run_time=1.0
            )
            self.wait(0.25)

        # Show some measurements
        lasers = VGroup()
        for ion in trapped_ions:
            point = ion.get_center()
            laser = Line(point + 0.5 * DL, point)
            laser.insert_n_curves(20)
            laser.set_stroke(RED, [1, 3, 3, 3, 1])
            lasers.add(laser)

        for value in [*np.random.randint(0, 2**8, 4), ord("Q")]:
            qubit_string.generate_target()
            qunit_num.generate_target()
            trapped_ions.generate_target()
            qunit_num.target.set_value(value)
            qubit_string.target.set_value(value)
            for ion, bit in zip(trapped_ions.target, qubit_string.target):
                ion[0].set_opacity(1.0 - bit.get_value())
            self.play(
                LaggedStartMap(VShowPassingFlash, lasers, lag_ratio=0.1, time_width=2.0, run_time=2),
                MoveToTarget(trapped_ions, lag_ratio=0.1, time_span=(0.5, 2.0)),
                MoveToTarget(qubit_string, lag_ratio=0.1, time_span=(0.5, 2.0)),
                MoveToTarget(qunit_num, time_span=(1.0, 1.25)),
                Transform(qunit_ket, Ket(qunit_num.target), time_span=(1.0, 1.5)),
            )

        # Describe a ket
        morty = Mortimer(height=5)
        morty.move_to(np.array([13., -6., 0.]))
        big_ket = Ket(Square(1))
        big_ket.set_fill(border_width=3)
        big_ket.next_to(morty.get_corner(UL), UP, MED_LARGE_BUFF)
        big_ket_name = TexText("``ket''", font_size=96)
        big_ket_name.next_to(big_ket, UP, MED_LARGE_BUFF)

        self.play(
            frame.animate.reorient(0, 0, 0, (4.66, -2.55, 0.0), 13.19),
            morty.change("raise_right_hand", big_ket),
            VFadeIn(morty),
            *(
                TransformFromCopy(src, big_ket)
                for src in [qubit_ket, qunit_ket]
            ),
        )
        self.play(
            Write(big_ket_name, time_span=(0.75, 2.0)),
            FlashAround(big_ket, time_width=1.5, run_time=2)
        )
        self.wait()

        # Refocus
        self.remove(big_ket)
        self.play(LaggedStart(
            FadeOut(VGroup(morty, big_ket, big_ket_name)),
            TransformFromCopy(big_ket, qubit_ket),
            TransformFromCopy(big_ket, qunit_ket),
            frame.animate.to_default_state(),
        ))

        # Expand mid layer
        mid_layer = layers[1]
        mid_layer.set_z_index(-2)
        mid_layer.generate_target()
        mid_layer.target.set_height(7, stretch=True)
        mid_layer.target.move_to(layers, UP)
        mid_layer.target.set_fill(opacity=0.25)

        target_y = -1.0

        self.play(
            FadeOut(
                VGroup(layers[2], num_mob, character, qunit, layer_names[2]),
                UP,
            ),
            FadeOut(
                Group(layers[0], circuitry, trapped_ions, layer_names[0]),
                DOWN,
            ),
            FadeOut(layers_name, DOWN),
            qubits.animate.set_y(target_y),
            boxed_bits.animate.match_x(classical).set_y(target_y),
            layer_names[1].animate.set_y(target_y),
            MoveToTarget(mid_layer, time_span=(0.5, 2.0)),
            run_time=2
        )
        self.play(
            FadeOut(layers[1]),
            FadeOut(layer_names[1]),
            run_time=3
        )

        # Show state vs. what you read, classical
        contrast = VGroup(
            Text("State"),
            Tex(R"=", font_size=72),
            Text("What you see"),
        )
        contrast.arrange(RIGHT)
        contrast[2].align_to(contrast[0], UP)
        contrast.match_x(classical)
        contrast.set_y(0.5)
        contrast.shift(0.5 * RIGHT)

        boxed_bits_copy = boxed_bits.copy()
        boxed_bits_copy.scale(0.7)
        boxed_bits_copy.stretch(0.8, 0)
        for bit in boxed_bits_copy[1]:
            bit.stretch(1 / 0.8, 0)
        boxed_bits_copy.next_to(contrast[2], DOWN, buff=0.75)
        boxed_bits_copy[0].set_stroke(WHITE, 1)

        boxed_bits.target = boxed_bits_copy.copy()
        boxed_bits.target.match_x(contrast[0])

        self.play(
            FadeIn(contrast[::2]),
            MoveToTarget(boxed_bits),
        )
        self.play(
            Write(contrast[1]),
            TransformFromCopy(boxed_bits, boxed_bits_copy, path_arc=30 * DEG),
        )
        self.wait()
        self.play(*(
            LaggedStart(
                (bit.animate.set_stroke(YELLOW, 3).set_anim_args(rate_func=there_and_back)
                for bit in group[1]),
                lag_ratio=0.25,
                run_time=4
            )
            for group in [boxed_bits, boxed_bits_copy]
        ))
        self.wait()

        # Show state vs. what you read, quantum
        q_contrast = contrast.copy()
        q_contrast.match_x(quantum)
        ne = Tex(R"\ne", font_size=72)
        ne.move_to(q_contrast[1])
        ne.set_color(RED)
        q_contrast[1].become(ne)

        state_vector = Vector(UR, thickness=4)
        state_vector.set_color(TEAL)
        state_vector.next_to(q_contrast[0], DOWN, MED_LARGE_BUFF)
        state_vector.set_opacity(0)  # Going to overlap something else instead

        state_vector_outline = state_vector.copy().set_fill(opacity=0)
        state_vector_outline.set_stroke(BLUE_A, 3)
        state_vector_outline.insert_n_curves(100)

        qubits.generate_target()
        qubits.target[1].space_out_submobjects(0.8)
        qubits.target[0].become(Ket(qubits.target[1]))
        qubits.target.match_x(q_contrast[2]).match_y(state_vector)

        moving_rect = SurroundingRectangle(state_vector)
        moving_rect.set_stroke(YELLOW, 3, 0)

        self.play(LaggedStart(
            TransformFromCopy(contrast, q_contrast, path_arc=-45 * DEG),
            MoveToTarget(qubits),
            GrowArrow(state_vector),
        ))
        self.wait()
        self.play(moving_rect.animate.surround(qubits).set_stroke(YELLOW, 3, 1))
        self.play(FadeOut(moving_rect))
        self.play(
            value_tracker.animate.set_value(0).set_anim_args(rate_func=there_and_back, run_time=4),
            UpdateFromFunc(qubit_string, lambda m: m.set_value(int(value_tracker.get_value()))),
        )
        self.wait()

        # Show randomness
        qubit_samples = list()
        for n in range(2**8):
            sample = qubits.copy()
            sample[1].set_value(n)
            sample.shift(np.random.uniform(-0.05, 0.05, 3))
            sample.set_stroke(TEAL, 1)
            qubit_samples.append(sample)

        labels = VGroup(Text("Random"), Text("Deterministic"))
        for label, mob, color in zip(labels, [qubits, boxed_bits_copy], [TEAL, YELLOW]):
            label.scale(0.75)
            label.next_to(mob, DOWN, buff=MED_LARGE_BUFF)
            label.set_color(color)

        self.play(
            FadeIn(labels),
            RandomSampling(qubits, qubit_samples),
        )
        self.wait()
        for _ in range(8):
            self.play(RandomSampling(qubits, qubit_samples))
            self.wait()

    def get_boxed_bits(self, value, length, height=0.5):
        boxes = Square().get_grid(1, length, buff=0)
        boxes.set_height(height)
        boxes.set_stroke(WHITE, 2)
        bits = BitString(value, length)
        for bit, box in zip(bits, boxes):
            bit.move_to(box)
        return VGroup(boxes, bits)


class AmbientStateVector(InteractiveScene):
    moving = False

    def construct(self):
        plane, axes = self.get_plane_and_axes()

        frame.reorient(14, 76, 0)
        frame.add_ambient_rotation(3 * DEG)
        self.add(plane, axes)

        # Vector
        vector = Vector(2 * normalize([1, 1, 2]), thickness=5)
        vector.set_fill(border_width=2)
        vector.set_color(TEAL)
        vector.always.set_perpendicular_to_camera(frame)

        self.play(GrowArrow(vector))

        if not self.moving:
            self.wait(36)
        else:
            for n in range(16):
                axis = normalize(np.random.uniform(-1, 1, 3))
                angle = np.random.uniform(0, PI)
                self.play(Rotate(vector, angle, axis=axis, about_point=ORIGIN))
                self.wait()

        # Show the sphere
        frame.reorient(16, 77, 0)

        sphere = Sphere(radius=2)
        sphere.always_sort_to_camera(self.camera)
        sphere.set_color(BLUE, 0.25)
        sphere_mesh = SurfaceMesh(sphere, resolution=(41, 21))
        sphere_mesh.set_stroke(WHITE, 0.5, 0.5)

        self.play(
            ShowCreation(sphere),
            Write(sphere_mesh, lag_ratio=1e-3),
        )
        self.wait(10)

    def get_plane_and_axes(self, scale_factor=2.0):
        # Add axes
        frame = self.frame
        axes = ThreeDAxes((-1, 1), (-1, 1), (-1, 1))
        plane = NumberPlane(
            (-1, 1 - 1e-5),
            (-1, 1 - 1e-5),
            faded_line_ratio=5
        )
        plane.background_lines.set_stroke(opacity=0.5)
        plane.faded_lines.set_stroke(opacity=0.25)
        plane.axes.set_stroke(opacity=0.25)
        result = VGroup(plane, axes)
        result.scale(scale_factor)

        return result


class RotatingStateVector(AmbientStateVector):
    moving = True


class FlipsToCertainDirection(AmbientStateVector):
    def construct(self):
        # Test
        frame = self.frame
        plane, axes = axes_group = self.get_plane_and_axes(scale_factor=3)

        vector = Vector(axes.c2p(1, 0, 0), fill_color=TEAL, thickness=5)
        vector.always.set_perpendicular_to_camera(frame)

        frame.reorient(-31, 72, 0)
        frame.add_ambient_rotation(DEG)
        self.add(frame)
        self.add(plane, axes)
        self.add(vector)

        # Show some flips
        theta = 15 * DEG
        h_plane = Square3D()
        h_plane.replace(plane)
        h_plane.set_color(WHITE, 0.15)
        diag_plane = h_plane.copy().rotate(theta, axis=DOWN)
        ghosts = VGroup()

        for n in range(1, 14):
            axis = [UP, DOWN][n % 2]
            shown_plane = [h_plane, diag_plane][n % 2]
            if n == 1:
                shown_plane = VectorizedPoint()
            ghosts.add(vector.copy())
            ghosts.generate_target()
            for n, vect in enumerate(ghosts.target[::-1]):
                vect.set_opacity(0.5 / (n + 1))
            self.play(
                MoveToTarget(ghosts),
                Rotate(vector, n * theta, axis=axis, about_point=ORIGIN),
            )
            shown_plane.set_opacity(0.15)

        self.wait(6)


class DisectAQuantumComputer(InteractiveScene):
    def construct(self):
        # Set up the machine with a random pile of gates
        wires = Line(4 * LEFT, 4 * RIGHT).replicate(4)
        wires.arrange(DOWN, buff=0.5)
        wires.shift(LEFT)

        gates = VGroup(
            self.get_labeled_box(wires[0], 0.1),
            self.get_cnot(wires[1], wires[0], 0.2),
            self.get_cnot(wires[3], wires[1], 0.3),
            self.get_cnot(wires[2], wires[3], 0.4),
            self.get_labeled_box(wires[3], 0.5, "X"),
            self.get_labeled_box(wires[1], 0.5, "Z"),
            self.get_cnot(wires[2], wires[1], 0.6),
            self.get_labeled_box(wires[1], 0.7),
            self.get_cnot(wires[3], wires[1], 0.8),
            self.get_cnot(wires[0], wires[1], 0.9),
        )
        mes_gates = VGroup(
            self.get_measurement(wires[0], 1),
            self.get_measurement(wires[1], 1),
            self.get_measurement(wires[2], 1),
            self.get_measurement(wires[3], 1),
        )

        circuit = Group(wires, Point(), gates, mes_gates)
        machine_rect = SurroundingRectangle(circuit)
        machine_rect.set_stroke(TEAL, 2)
        qc_label = get_quantum_computer_symbol(height=1)
        qc_label.next_to(machine_rect, UP)

        self.add(circuit, machine_rect, qc_label)

        # Show a program running through quantum wires (TODO< show the results)
        n_repetitions = 25

        wire_glows = Group(
            GlowDot(wire.get_start(), color=TEAL, radius=0.25)
            for wire in wires
        )
        wire_glows.set_z_index(-1)
        gate_glows = gates.copy()
        gate_glows.set_stroke(TEAL, 1)
        gate_glows.set_fill(opacity=0)
        for glow in gate_glows:
            glow.add_updater(lambda m: m.set_stroke(width=4 * np.exp(
                -(0.5 * (m.get_x() - wire_glows.get_x()))**2
            )))

        output = self.get_random_qubits()
        output.next_to(machine_rect, RIGHT, buff=LARGE_BUFF)

        for n in range(n_repetitions):
            self.add(gate_glows)
            wire_glows.set_x(wires.get_x(LEFT))
            wire_glows.set_opacity(1)
            self.play(
                wire_glows.animate.match_x(mes_gates),
                rate_func=linear,
                run_time=1
            )
            output_value = random.randint(0, 15)
            output[0].set_value(output_value)
            for mes_gate, bit in zip(mes_gates, output[0]):
                mes_gate[-1].set_stroke(opacity=0)
                mes_gate[-1][bit.get_value()].set_stroke(opacity=1)
            self.add(output)
            wire_glows.set_opacity(0)
            self.play(wire_glows.animate.shift(6 * RIGHT), rate_func=linear)

        self.remove(wire_glows, gate_glows)

        # Setup all 16 outputs
        machine = Group(circuit, machine_rect, qc_label)
        all_qubits = output.replicate(16)
        for n, qubits in enumerate(all_qubits):
            qubits[0].set_value(n)
        all_qubits.arrange(DOWN)
        all_qubits.set_height(FRAME_HEIGHT - 1)
        all_qubits.move_to(machine, RIGHT)
        all_qubits.set_y(0)

        all_qubit_rects = VGroup(
            SurroundingRectangle(qubits, buff=0.05).set_stroke(YELLOW, 1, 0.5)
            for qubits in all_qubits
        )
        output_rect = SurroundingRectangle(output)

        brace = Brace(all_qubit_rects, RIGHT, SMALL_BUFF)
        brace_label = brace.get_tex("2^4 = 16")
        brace_label.shift(0.2 * RIGHT + 0.05 * UP)

        self.play(ShowCreation(output_rect))
        self.wait()
        self.remove(output, output_rect)
        self.play(
            machine.animate.to_edge(LEFT),
            TransformFromCopy(VGroup(output), all_qubits),
            TransformFromCopy(VGroup(output_rect), all_qubit_rects),
        )
        self.play(
            GrowFromCenter(brace),
            Write(brace_label),
        )
        self.wait()

        # Show probability distribution
        qc_label.generate_target()
        qc_label.target.set_y(0).scale(1.5)
        comp_to_dist_arrow = Arrow(qc_label.target, all_qubits, buff=1.0, thickness=6)

        dists = [
            np.random.randint(1, 8, 16).astype(float),
            np.random.random(16) + 10 * np.eye(16)[10],
            np.random.random(16) + 5,
            np.random.randint(1, 8, 16).astype(float),
        ]
        for dist in dists:
            dist /= dist.sum()

        all_dist_rects = VGroup(
            self.get_dist_bars(dist, all_qubits, width_ratio=6)
            for dist in dists
        )
        dist_rects = all_dist_rects[0]

        dist_rect_labels = VGroup(
            Integer(100 * prob, unit=R"\%", font_size=24, num_decimal_places=1).next_to(rect, RIGHT, SMALL_BUFF)
            for prob, rect in zip(dists[0], dist_rects)
        )

        self.play(
            FadeOut(machine_rect, DOWN),
            FadeOut(circuit, DOWN),
            MoveToTarget(qc_label)
        )
        self.play(
            GrowArrow(comp_to_dist_arrow),
            ReplacementTransform(all_qubit_rects, dist_rects, lag_ratio=0.1),
            FadeOut(brace, RIGHT),
            FadeOut(brace_label, RIGHT),
        )
        self.wait()
        self.play(LaggedStartMap(FadeIn, dist_rect_labels, shift=0.25 * RIGHT, lag_ratio=0.1, run_time=2))
        self.wait(2)
        self.play(FadeOut(dist_rect_labels))
        self.wait()
        for new_rects in all_dist_rects[1:]:
            self.play(Transform(dist_rects, new_rects, run_time=2))
            self.wait()

        # Show magnifying glass
        if False:
            # This is just ment for an insertion
            dist = VGroup(all_qubits, dist_rects)
            dist.shift(0.25 * LEFT)
            qc_label.scale(1.5).next_to(comp_to_dist_arrow, LEFT, MED_LARGE_BUFF)
            glass = get_magnifying_glass()
            glass.next_to(qc_label, UL).to_edge(UP)
            glass.save_state()
            dist_rects.save_state()

            wigglers = Superposition(
                VGroup(VGroup(qb, bar) for qb, bar in zip(all_qubits, dist_rects)).copy(),
                max_rot_vel=8,
                glow_stroke_opacity=0
            )

            index = 7
            new_dist = np.zeros(16)
            new_dist[index] = 7
            choice_rect = SurroundingRectangle(all_qubits[index])

            new_bars = self.get_dist_bars(new_dist, all_qubits, width_ratio=3)
            new_bars.set_stroke(width=0)
            new_bars[index].set_stroke(WHITE, 1)

            self.play(FadeIn(glass))
            for _ in range(2):
                self.play(
                    glass.animate.shift(qc_label.get_center() - glass[0].get_center()).set_anim_args(path_arc=-45 * DEG),
                    Transform(dist_rects, new_bars, time_span=(1.25, 1.5)),
                    FadeIn(choice_rect, time_span=(1.25, 1.5)),
                    run_time=2
                )
                self.wait()
                self.play(
                    Restore(glass, path_arc=45 * DEG),
                    FadeOut(choice_rect, time_span=(0.25, 0.5)),
                    run_time=1.5
                )
                self.wait()

            # Delicate state
            wigglers.set_offset_multiple(0)
            self.play(FadeOut(dist), FadeIn(wigglers))
            self.play(wigglers.animate.set_offset_multiple(0.05))
            self.wait(8)

            # For the chroma key
            qubits = all_qubits[index].copy()
            qubits.set_width(0.7 * qc_label.get_width()).move_to(qc_label)
            qubits.set_fill(border_width=3)

            self.clear()
            self.add(qubits)

        # Label it as 4 qubit
        title = Text("4 qubit quantum computer")
        title.next_to(qc_label, UP, buff=LARGE_BUFF)
        title.set_backstroke(BLACK, 3)
        underline = Underline(title, buff=-0.05)

        self.play(
            ShowCreation(underline, time_span=(1, 2)),
            Write(title, run_time=2),
        )
        self.wait()

        # k qubit case
        last_sym = title["4"][0]
        last_qubits = all_qubits
        last_dist_rects = dist_rects
        for k in [5, 6, "k"]:
            k_sym = Tex(str(k))
            k_sym.move_to(last_sym)
            k_sym.set_color(YELLOW)

            n_bits = k if isinstance(k, int) else 7

            multi_bitstrings = VGroup(
                BitString(n, length=n_bits)
                for n in range(2**n_bits)
            )
            new_qubits = VGroup(
                VGroup(bits, Ket(bits))
                for bits in multi_bitstrings
            )
            new_qubits.arrange(DOWN)
            new_qubits.set_height(FRAME_HEIGHT - 1)
            new_qubits.move_to(all_qubits)
            big_dist_rects = self.get_dist_bars(np.random.random(2**n_bits)**2, new_qubits)

            self.play(
                FadeOut(last_sym, 0.25 * UP),
                FadeIn(k_sym, 0.25 * UP),
                FadeOut(last_qubits),
                FadeOut(last_dist_rects),
                FadeIn(new_qubits),
                FadeIn(big_dist_rects),
            )
            self.wait()

            last_sym = k_sym
            last_qubits = new_qubits
            last_dist_rects = big_dist_rects

        # Count all outputs
        brace = Brace(last_dist_rects, RIGHT, SMALL_BUFF)
        brace_label = Tex(R"2^k", t2c={"k": YELLOW})
        brace_label.next_to(brace, RIGHT)

        bits = BitString(70, length=7)
        longer_example = last_qubits[100].copy().set_width(2.0)
        longer_example.next_to(brace_label, DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)

        down_brace = Brace(longer_example[0], DOWN, SMALL_BUFF)
        down_brace_label = down_brace.get_tex("k").set_color(YELLOW)

        self.play(
            GrowFromCenter(brace),
            TransformFromCopy(last_sym, brace_label),
        )
        self.play(
            TransformFromCopy(last_qubits[70], longer_example),
        )
        self.play(
            GrowFromCenter(down_brace),
            TransformFromCopy(brace_label[1], down_brace_label)
        )
        self.wait()

        # Highlight the word qubit
        word_rect = SurroundingRectangle(title["qubit"], buff=0.1)
        word_rect.set_stroke(YELLOW, 2)
        self.play(ReplacementTransform(underline, word_rect))
        self.wait()
        title[0].set_opacity(0)
        self.play(
            FadeOut(title),
            FadeOut(word_rect),
            FadeOut(last_sym),
            LaggedStartMap(FadeOut, VGroup(brace, brace_label, longer_example, down_brace, down_brace_label)),
            FadeOut(last_qubits),
            FadeOut(last_dist_rects),
            FadeIn(all_qubits),
            FadeIn(dist_rects),
        )
        self.wait()

        # Emphasize you see only one
        frame = self.frame

        dist_rect = SurroundingRectangle(VGroup(comp_to_dist_arrow, all_qubits, dist_rects), buff=0.25)
        dist_rect.set_stroke(BLUE, 2)
        dist_rect.stretch(1.2, 0)
        output.set_width(2)
        output.next_to(dist_rect, RIGHT, buff=2.5)
        output.align_to(all_qubits, UP)

        dist_words = Text("Implicit", font_size=72)
        dist_words.next_to(dist_rect, UP)
        output_words = Text("What you see", font_size=72)
        output_words.next_to(output, UP).match_y(dist_words)

        sample_rects = VGroup(
            SurroundingRectangle(qb, buff=0.05)
            for qb, bar in zip(all_qubits, dist_rects)
        )
        sample_rects.set_stroke(YELLOW, 1)
        sample = 6
        output[0].set_value(sample)

        self.play(LaggedStart(
            frame.animate.reorient(0, 0, 0, (4.2, 0.7, 0.0), 9.86),
            FadeIn(dist_rect),
            FadeIn(dist_words, lag_ratio=0.1),
            FadeIn(output),
            FadeIn(output_words, lag_ratio=0.1),
            FadeIn(sample_rects[sample]),
            lag_ratio=0.2
        ))

        # Don't see multiple at once
        superposition_pieces = all_qubits.copy()
        superposition_pieces.set_width(1.5)
        superposition_pieces.space_out_submobjects(0.5)
        superposition_pieces.move_to(output, UP)
        superposition = Superposition(superposition_pieces)
        big_cross = Cross(superposition, stroke_width=[0, 15, 15, 15, 0])

        self.play(
            FadeIn(superposition),
            FadeOut(output),
            ShowCreation(big_cross)
        )
        self.wait(6)
        self.play(
            FadeOut(superposition),
            FadeOut(big_cross),
        )

        # Show some random samples
        n_samples = 8
        output_choices = output.replicate(16)
        for n, choice, sample_rect in zip(it.count(), output_choices, sample_rects):
            choice[0].set_value(n)
            choice.shift(np.random.uniform(-0.1, 0.1, 3))
            choice.set_backstroke(TEAL, 2)
            choice.add(sample_rect)

        self.remove(output, sample_rects)
        selection = VGroup()
        for _ in range(n_samples):
            self.play(RandomSampling(selection, list(output_choices), weights=dists[-1]))
            self.wait()

        # Concentrate probability to one value
        conc_dist = np.zeros(16)
        num = 7
        conc_dist[num] = 1
        new_rects = self.get_dist_bars(conc_dist, all_qubits, width_ratio=2.0)
        self.play(
            FadeOut(selection),
            Transform(dist_rects, new_rects),
            dist_rect.animate.stretch(1.2, 0, about_edge=LEFT),
            UpdateFromFunc(dist_words, lambda m: m.match_x(dist_rect)),
            output_words.animate.shift(RIGHT),
        )
        self.wait()
        self.play(ShowCreation(sample_rects[num]))
        output[0].set_value(num)
        output.match_x(output_words)
        self.play(TransformFromCopy(all_qubits[num], output))
        self.wait()

        # Back to another distribution
        point1_index = 2
        point5_index = 3
        new_dist = np.random.random(16)
        new_dist[point1_index] = new_dist[point5_index] = 0
        new_dist *= (1.0 - 0.25 - 0.01) / new_dist.sum()
        new_dist[point1_index] = 0.01
        new_dist[point5_index] = 0.25

        self.play(Transform(dist_rects, self.get_dist_bars(new_dist, all_qubits)))
        self.remove(output, selection, sample_rects)
        VGroup(choice[:-1] for choice in output_choices).match_x(output_words)
        self.play(RandomSampling(selection, list(output_choices), weights=new_dist))
        self.remove(selection)
        self.add(output, sample_rects[num])
        self.wait()

        # Ask where the distribution comes from
        q_marks = Tex(R"???", font_size=96)
        dist_group = VGroup(all_qubits, dist_rects)
        q_marks.next_to(dist_group, UP, MED_LARGE_BUFF)

        self.play(
            dist_rect.animate.surround(dist_group),
            FadeTransformPieces(dist_words, q_marks),
        )
        self.wait()

        dist_group.add(dist_rect)

        # Introduce the state vector
        state = np.array([random.choice([-1, 1]) * np.sqrt(p) for p in new_dist])
        state[point5_index] = 0.5
        state[point1_index] = -0.1
        state_vector = DecimalMatrix(np.zeros((16, 1)), decimal_config=dict(include_sign=True, edge_to_fix=LEFT))
        for value, elem in zip(state, state_vector.elements):
            elem.set_value(value)
        state_vector.match_height(all_qubits)
        state_vector.next_to(qc_label, RIGHT, LARGE_BUFF)

        vector_title = Text("State Vector", font_size=60)
        vector_title.set_color(TEAL)
        vector_title.next_to(state_vector, UP, MED_LARGE_BUFF)

        comp_to_dist_arrow.target = comp_to_dist_arrow.generate_target()
        comp_to_dist_arrow.target.next_to(state_vector, RIGHT)

        vect_lines = self.get_connecting_lines(qc_label, state_vector, from_buff=-0.1, to_buff=0.1)

        self.play(LaggedStart(
            ShowCreation(vect_lines, lag_ratio=0),
            FadeInFromPoint(state_vector, qc_label.get_right()),
            MoveToTarget(comp_to_dist_arrow),
            dist_group.animate.next_to(comp_to_dist_arrow.target, RIGHT),
            MaintainPositionRelativeTo(sample_rects[num], dist_group),
            MaintainPositionRelativeTo(q_marks, dist_group),
            VGroup(output, output_words).animate.shift(0.5 * RIGHT),
        ))
        self.play(Write(vector_title))
        self.wait()

        # Label the positions of the vector
        pre_indices = VGroup(qb[0] for qb in all_qubits)
        indices = pre_indices.copy()
        indices.scale(0.75)
        indices.set_color(GREY_C)
        for bits, entry in zip(indices, state_vector.get_entries()):
            bits.next_to(state_vector, LEFT, buff=0.2)
            bits.match_y(entry)

        self.play(
            VGroup(qc_label, vect_lines).animate.shift(0.75 * LEFT),
            LaggedStart(
                (TransformFromCopy(pre_bits, bits, path_arc=30 * DEG)
                for pre_bits, bits in zip(pre_indices, indices)),
                lag_ratio=0.25,
                run_time=6
            ),
        )
        self.wait()

        # Fundamental rule
        ne_sign = Tex(R"\ne", font_size=120)
        ne_sign.move_to(comp_to_dist_arrow)
        ne_sign.set_color(RED)

        rule = Tex(R"x \rightarrow |x|^2", font_size=42)
        rule.next_to(comp_to_dist_arrow, UP, buff=SMALL_BUFF)

        template_eq = Tex(R"(+0.50)^2 = 0.25", font_size=20)
        entry_template = template_eq.make_number_changeable("+0.50", include_sign=True)
        percent_template = template_eq.make_number_changeable("0.25")

        vector_entries = state_vector.get_entries()

        bar_values = VGroup()
        for bar, entry in zip(dist_rects, vector_entries):
            entry_template.set_value(entry.get_value())
            percent_template.set_value(entry.get_value()**2)
            template_eq.next_to(bar, RIGHT, SMALL_BUFF)
            bar_values.add(template_eq.copy())

        self.play(
            FadeOut(comp_to_dist_arrow),
            FadeIn(ne_sign),
        )
        self.wait()
        self.play(
            FadeIn(comp_to_dist_arrow),
            FadeOut(ne_sign),
        )
        self.play(
            ReplacementTransform(q_marks, rule, path_arc=90 * DEG, run_time=1.5),
        )
        self.wait()
        self.play(
            dist_rect.animate.stretch(1.5, 0, about_edge=LEFT).set_anim_args(time_span=(1, 2)),
            FadeOut(sample_rects[num]),
            LaggedStart(
                (TransformFromCopy(entry, value[1], path_arc=-30 * DEG)
                for entry, value in zip(vector_entries, bar_values)),
                lag_ratio=0.5,
            ),
            LaggedStart(
                (FadeIn(value[0])
                for value in bar_values),
                lag_ratio=0.5,
                time_span=(2, 6)
            ),
            LaggedStart(
                (FadeIn(value[2:])
                for value in bar_values),
                lag_ratio=0.5,
                time_span=(2, 6)
            ),
            frame.animate.reorient(0, 0, 0, (2.15, 0.04, 0.0), 7.64),
            run_time=6,
        )
        self.remove(vector_title, output_words, output)
        self.wait()

        # Process the vector
        alt_states = [normalize(np.random.uniform(-1, 1, 16)) for x in range(2)]
        label_line = Line(UP, DOWN).set_height(1.0).next_to(qc_label.get_right(), LEFT, SMALL_BUFF)
        entries = state_vector.get_entries()
        comp_lines = VGroup(
            Line(
                label_line.pfp(alpha),
                entry.get_center(),
            ).insert_n_curves(20).set_stroke(
                color=random_bright_color(hue_range=(0.3, 0.5)),
                width=(0, 3, 3, 3, 0),
            )
            for alpha, entry in zip(np.linspace(0, 1, len(entries)), entries)
        )
        comp_lines.shuffle()

        self.play(FlashAround(rule, time_width=1.5, run_time=2))
        self.play(FadeOut(bar_values))
        for new_state in [*alt_states, state]:
            comp_lines.shuffle()
            self.play(
                LaggedStartMap(VShowPassingFlash, comp_lines, time_width=2.0, lag_ratio=0.02),
                Transform(dist_rects, self.get_dist_bars(new_state**2, all_qubits), time_span=(0.75, 1.75)),
                *(
                    ChangeDecimalToValue(entry, value, time_span=(0.75, 1.75))
                    for entry, value in zip(entries, new_state)
                ),
                run_time=2
            )

        # Highlight a few examples
        highlight_rect = SurroundingRectangle(
            VGroup(indices[point5_index], bar_values[point5_index])
        )
        highlight_rect.set_stroke(YELLOW, 2, 0)
        groups = [vector_entries, all_qubits, dist_rects, bar_values]
        bar_values.set_opacity(0)

        for index in [point5_index, point1_index]:
            self.play(
                frame.animate.reorient(0, 0, 0, (3.36, 0.87, 0.0), 5.60),
                dist_rect.animate.set_stroke(opacity=0),
                highlight_rect.animate.set_stroke(opacity=1).match_y(all_qubits[index]),
                *(
                    group[slc].animate.set_opacity(0.25)
                    for group in groups
                    for slc in [slice(0, index), slice(index + 1, None)]
                ),
                *(
                    group[index].animate.set_opacity(1)
                    for group in groups
                ),
                run_time=2,
            )
            self.play(
                FlashAround(indices[index], buff=SMALL_BUFF, color=RED, time_width=1.5),
                WiggleOutThenIn(indices[index]),
                run_time=2
            )
            self.wait()
            self.play(TransformFromCopy(vector_entries[index], bar_values[index][1], run_time=2, path_arc=-45 * DEG))
            self.play(FlashAround(bar_values[index][-1], buff=SMALL_BUFF, color=RED, time_width=1.5))
            self.wait()

        self.play(
            frame.animate.reorient(0, 0, 0, (2.13, 1.33, 0.0), 10.48),
            FadeOut(highlight_rect, time_span=(0, 1)),
            FadeIn(vector_title),
            *(
                group.animate.set_opacity(1)
                for group in groups
            ),
            FadeOut(bar_values, lag_ratio=0.01),
            run_time=4
        )

        # Concentrate value
        if False:  # Only used for an insertion
            # Concentrate onto one value
            og_state = state
            frame.reorient(0, 0, 0, (0.94, 0.34, 0.0), 9.20)
            target_state = np.zeros(16)
            key = 5
            target_state[key] = 1
            for alpha in [0.1, 0.1, 0.2, 0.25, 0.25, 0.25, 0.25, 1.0]:
                new_state = normalize(interpolate(state, target_state, alpha))

                self.play(
                    Transform(dist_rects, self.get_dist_bars(new_state**2, all_qubits, width_ratio=4), time_span=(0.5, 1.5)),
                    LaggedStartMap(VShowPassingFlash, comp_lines, time_width=2.0, lag_ratio=0.01, run_time=2.0),
                    *(
                        ChangeDecimalToValue(entry, value, time_span=(0.5, 1.5))
                        for entry, value in zip(state_vector.elements, new_state)
                    ),
                )

                state = new_state

            # Comment on the value
            rect = SurroundingRectangle(all_qubits[key])
            self.play(ShowCreation(rect, run_time=2))

        # Show the squares of all the values
        sum_expr = Tex(
            R"x_{0}^2 + x_{1}^2 + x_{2}^2 + x_{3}^2 + \cdots + x_{14}^2 + x_{15}^2 = 1",
            font_size=60
        )
        sum_expr.next_to(vector_title, UP, LARGE_BUFF)
        sum_expr.match_x(frame)

        self.play(
            LaggedStart(
                (FadeTransform(vector_entries[n].copy(), sum_expr[fR"x_{{{n}}}^2"])
                for n in [0, 1, 2, 3, 14, 15]),
                lag_ratio=0.05,
                run_time=2
            ),
            FadeTransformPieces(vector_entries[4:14].copy(), sum_expr[R"\cdots"][0], time_span=(0.5, 2.0)),
            Write(sum_expr["+"], time_span=(0.5, 1.5)),
            Write(sum_expr["= 1"], time_span=(1, 2)),
        )
        self.wait()
        self.play(FadeOut(sum_expr))

        # Flip some signs
        top_eq = Tex(R"(+0.50)^2 = 0.25", font_size=60)
        top_eq.move_to(sum_expr)
        top_value = top_eq.make_number_changeable("+0.50", include_sign=True)

        entry = vector_entries[point5_index]
        entry_rect = SurroundingRectangle(entry, buff=0.1)
        entry_rect.set_stroke(YELLOW, 1)
        value_rect = SurroundingRectangle(top_value)
        value_rect.set_stroke(YELLOW, 2)

        lines = VGroup(
            Line(entry_rect.get_corner(UL), value_rect.get_corner(DL)),
            Line(entry_rect.get_corner(UR), value_rect.get_corner(DR)),
        )
        lines.set_stroke(YELLOW, 1)

        self.play(ShowCreation(entry_rect))
        self.play(
            TransformFromCopy(entry_rect, value_rect),
            TransformFromCopy(entry, top_value),
            ShowCreation(lines, lag_ratio=0),
            FadeOut(vector_title),
        )
        self.wait()
        self.play(
            entry.animate.set_value(-0.5),
            top_value.animate.set_value(-0.5),
        )
        self.wait()
        self.play(
            Write(top_eq[0]),
            Write(top_eq[2:]),
            FadeOut(value_rect)
        )
        self.wait()
        self.play(
            entry.animate.set_value(0.5),
            top_value.animate.set_value(0.5),
        )
        self.wait()
        self.play(
            Uncreate(lines, lag_ratio=0),
            ReplacementTransform(top_value, entry),
            FadeOut(top_eq[0]),
            FadeOut(top_eq[2:]),
            FadeOut(entry_rect),
            FadeIn(vector_title),
            frame.animate.reorient(0, 0, 0, (0.45, 0.46, 0.0), 8.70),
        )

        # Flip some more signs
        signs = VGroup(entry[0] for entry in vector_entries)
        vector_entries.save_state()
        sign_choice = VGroup(Tex("+").set_color(BLUE), Tex("-").set_color(RED))
        sign_choice.set_width(1.2 * signs[0].get_width())
        sign_choices = VGroup(
            sign_choice.copy().move_to(sign, RIGHT)
            for sign in signs
        )

        for n in range(3):
            self.play(LaggedStart(*(
                Transform(sign, random.choice(choice), path_arc=PI)
                for sign, choice in zip(signs, sign_choices)),
                lag_ratio=0.15,
                run_time=2
            ))
        self.play(
            Restore(vector_entries),
        )

        # Go down to two dimensions
        small_state = normalize([1, 2])
        small_state_vector = DecimalMatrix(small_state.reshape([2, 1]), v_buff=1.0)
        small_state_vector.set_height(2.0)
        small_state_vector.move_to(state_vector, RIGHT)

        bits = VGroup(Integer(0), Integer(0))
        bits.arrange(DOWN, MED_LARGE_BUFF)
        qubits = VGroup(
            VGroup(bit, Ket(bit))
            for bit in bits
        )
        qubits.scale(1.5)
        qubits.arrange_to_fit_height(small_state_vector.get_entries().get_height())
        qubits.next_to(comp_to_dist_arrow, RIGHT, MED_LARGE_BUFF)
        qubits.reverse_submobjects()
        qubits[1][0].set_value(1)

        small_dist_bars = self.get_dist_bars(small_state**2, qubits, width_ratio=2)

        small_vect_lines = self.get_connecting_lines(qc_label, small_state_vector, from_buff=-0.1, to_buff=0.1)

        self.remove(state_vector, all_qubits, dist_rects, vect_lines)
        self.play(LaggedStart(
            FadeOut(indices, scale=0.25),
            TransformFromCopy(vect_lines, small_vect_lines),
            TransformFromCopy(state_vector.get_brackets(), small_state_vector.get_brackets()),
            TransformFromCopy(state_vector.get_entries(), small_state_vector.get_entries()),
            vector_title.animate.next_to(small_state_vector, UP, MED_LARGE_BUFF),
            FadeTransformPieces(all_qubits.copy(), qubits),
            TransformFromCopy(dist_rects, small_dist_bars),
            lag_ratio=0.1,
            run_time=3
        ))
        self.add(small_state_vector)
        self.wait()

        self.play(
            FadeOut(VGroup(small_vect_lines, small_state_vector, qubits, small_dist_bars)),
            FadeIn(VGroup(vect_lines, indices, state_vector, all_qubits, dist_rects)),
            vector_title.animate.next_to(state_vector, UP, MED_LARGE_BUFF),
        )

        # Apply Grover's
        self.play(
            frame.animate.reorient(0, 0, 0, (2, 0.45, 0.0), 9),
            qc_label[1].animate.set_color(BLUE),
        )
        key = 12
        state0 = np.sqrt(1.0 / 16) * np.ones(16)
        new_states = [state0]
        for n in range(3):
            state = new_states[-1].copy()
            state[key] *= -1
            new_states.append(state)
            new_states.append(2 * np.dot(state0, state) * state0 - state)

        key_icon = get_key_icon()
        key_entry_rect = SurroundingRectangle(VGroup(indices[key], vector_entries[key]))
        key_entry_rect.stretch(1.25, 0, about_edge=RIGHT)
        key_entry_rect.set_stroke(YELLOW, 2)

        for n, state in enumerate(new_states):
            anims = [Transform(
                dist_rects, self.get_dist_bars(state**2, all_qubits, width_ratio=4.0),
                time_span=(1, 2)
            )]
            if n == 1:
                # Highlight the outcomes
                entry_rects = VGroup(SurroundingRectangle(e, buff=0.05) for e in vector_entries)
                qubit_rects = VGroup(
                    SurroundingRectangle(VGroup(*pair), buff=0.05)
                    for pair in zip(all_qubits, dist_rects)
                )
                VGroup(entry_rects, qubit_rects).set_stroke(YELLOW, 2)
                self.play(
                    LaggedStartMap(VFadeInThenOut, entry_rects, lag_ratio=0.1),
                    LaggedStartMap(VFadeInThenOut, qubit_rects, lag_ratio=0.1),
                    run_time=6
                )
                self.wait()

                # Show the key
                bits = indices[key]
                bits.save_state()
                self.play(bits.animate.scale(2).set_color(WHITE).move_to(8 * RIGHT))

                key_icon.set_height(1.5 * bits.get_height())
                key_icon.next_to(bits, LEFT, SMALL_BUFF)
                self.play(Write(key_icon))
                self.wait()

                self.play(
                    Restore(bits),
                    key_icon.animate.scale(0.5).next_to(bits.saved_state, LEFT, buff=SMALL_BUFF),
                )

            if n % 2 == 1:
                neg1 = Tex(R"\times -1")
                neg1.next_to(key_entry_rect, RIGHT)
                neg1.set_color(YELLOW)
                anims.append(VFadeInThenOut(neg1, run_time=2))
                anims.append(VFadeInThenOut(key_entry_rect, run_time=2))
                anims.append(ChangeDecimalToValue(vector_entries[key], state[key], time_span=(0.5, 1.5)))
            else:
                anims.append(
                    LaggedStartMap(VShowPassingFlash, comp_lines, time_width=2.0, lag_ratio=0.005, run_time=2)
                )
                anims.extend([
                    ChangeDecimalToValue(entry, value, time_span=(1, 2))
                    for entry, value in zip(vector_entries, state)
                ])
            self.play(*anims)
            self.wait()

        # Read out from memory
        glass = get_magnifying_glass()
        glass.next_to(qc_label, UP, MED_LARGE_BUFF)
        glass.scale(0.75)
        glass.to_edge(LEFT, buff=0).shift(frame.get_x() * RIGHT)

        self.play(FadeIn(glass))
        self.play(
            glass.animate.shift(qc_label.get_center() - glass[0].get_center()).set_anim_args(path_arc=-45 * DEG),
            rate_func=there_and_back_with_pause,
            run_time=6
        )
        self.wait()

        qubits = all_qubits[key].copy()
        qubits.move_to(qc_label)
        qubits.scale(1.25)
        black_rect = FullScreenRectangle().set_fill(BLACK, 1)
        black_rect.fix_in_frame()
        self.add(black_rect, qubits)
        self.wait()

    def get_labeled_box(self, wire, alpha, label="H", size=0.5):
        box = Square(size)
        box.set_stroke(WHITE, 1)
        box.set_fill(BLACK, 1)
        box.move_to(wire.pfp(alpha))

        vect = rotate_vector(UP, PI / 8)
        arrow = VGroup(
            Vector(size * vect, thickness=1).center(),
            Vector(-size * vect, thickness=1).center(),
        )
        arrow.move_to(box)

        label = Tex(label)
        label.set_height(0.5 * size)
        label.move_to(box)

        return VGroup(box, label)

    def get_cnot(self, wire1, wire2, alpha):
        oplus = Tex(R"\oplus")
        dot = Dot()
        oplus.move_to(wire1.pfp(alpha))
        dot.move_to(wire2.pfp(alpha))
        connector = Line(dot, oplus, buff=0)
        connector.set_stroke(WHITE, 1)
        return VGroup(oplus, connector, dot)

    def get_measurement(self, wire, alpha, size=0.5):
        box = Square(size)
        box.set_stroke(WHITE, 1)
        box.set_fill(BLACK, 1)
        box.move_to(wire.pfp(alpha))
        arc = Arc(PI / 4, PI / 2)
        arc.set_width(0.7 * box.get_width())
        arc.move_to(box)
        arc.set_stroke(WHITE, 1)
        lines = VGroup(Line(ORIGIN, 0.2 * vect) for vect in [UL, UR])
        lines.move_to(box)
        lines.set_stroke(WHITE, 1)
        lines[1].set_stroke(opacity=0)
        return VGroup(box, arc, lines)

    def get_random_qubits(self):
        bits = BitString(random.randint(0, 15))
        ket = Ket(bits)
        return VGroup(bits, ket)

    def get_connecting_lines(self, from_mob, to_mob, from_buff=0, to_buff=0, stroke_color=TEAL_A, stroke_width=2):
        l_ur = from_mob.get_corner(UR) + from_buff * UR
        l_dr = from_mob.get_corner(DR) + from_buff * DR
        v_ul = to_mob.get_corner(UL) + to_buff * UL
        v_dl = to_mob.get_corner(DL) + to_buff * DL

        lines = VGroup(
            CubicBezier(l_ur, l_ur + RIGHT, v_ul + LEFT, v_ul),
            CubicBezier(l_dr, l_dr + RIGHT, v_dl + LEFT, v_dl),
        )
        lines.set_stroke(TEAL_A, 2)
        return lines

    def get_dist_bars(
        self,
        dist,
        objs,
        height_ratio=0.8,
        width_ratio=8,
        fill_colors=(BLUE_D, GREEN),
        stroke_color=WHITE,
        stroke_width=1,
    ):
        normalized_dist = np.array(dist) / sum(dist)
        height = objs[0].get_height() * height_ratio
        rects = VGroup(
            Rectangle(width_ratio * p, height).next_to(obj)
            for p, obj in zip(normalized_dist, objs)
        )
        rects.set_fill(opacity=1)
        rects.set_submobject_colors_by_gradient(*fill_colors)
        rects.set_stroke(stroke_color, stroke_width)
        return rects


class Qubit(DisectAQuantumComputer):
    def construct(self):
        # Set up plane
        frame = self.frame
        plane = self.get_plane()
        zero_label, one_label = qubit_labels = self.get_qubit_labels(plane)

        frame.move_to(plane)
        self.add(plane)
        self.add(qubit_labels)

        # Add vector
        vector = self.get_vector(plane)
        vector_label = DecimalMatrix([[1.0], [0.0]], bracket_h_buff=0.1, decimal_config=dict(include_sign=True))
        vector_label.add_background_rectangle()
        vector_label.scale(0.5)
        vector_label.set_backstroke(BLACK, 5)
        theta_tracker = ValueTracker(0)
        vector.add_updater(lambda m: m.set_angle(theta_tracker.get_value()))
        vector.add_updater(lambda m: m.shift(plane.c2p(0, 0) - m.get_start()))

        def get_state():
            theta = theta_tracker.get_value()
            return np.array([math.cos(theta), math.sin(theta)])

        def position_label(vector_label):
            x, y = get_state()
            buff = SMALL_BUFF + 0.5 * interpolate(vector_label.get_width(), vector_label.get_height(), x**2)

            vect = normalize(vector.get_vector())
            vector_label.move_to(vector.get_end() + buff * vect)

        def update_coordinates(vector_label):
            for element, value in zip(vector_label.elements, get_state()):
                element.set_value(value)

        vector_label.add_updater(position_label)
        vector_label.add_updater(update_coordinates)

        self.add(vector, vector_label)
        self.play(theta_tracker.animate.set_value(240 * DEG), run_time=5)
        self.play(theta_tracker.animate.set_value(120 * DEG), run_time=4)
        self.wait()

        # Add rule
        var_vect = TexMatrix([["x"], ["y"]], bracket_h_buff=0.1)
        var_vect.next_to(plane, RIGHT, buff=1.0)
        var_vect.to_edge(UP)
        var_vect.add_background_rectangle()

        coord_vect = DecimalMatrix([[0], [1]], v_buff=0.5, bracket_h_buff=0.1, decimal_config=dict(include_sign=True))
        coord_vect.scale(0.75)
        coord_vect.next_to(var_vect, DOWN, buff=2.25, aligned_edge=RIGHT)
        coord_vect.clear_updaters()
        coord_vect.add_background_rectangle()
        coord_vect.add_updater(update_coordinates)

        prob_rule = VGroup(
            Tex(R"P(0) = x^2"),
            Tex(R"P(1) = y^2"),
        )
        prob_rule.arrange(DOWN)
        prob_rule.next_to(var_vect, RIGHT, buff=1.5)
        var_arrow = Arrow(var_vect, prob_rule, buff=0.25)

        qubits = qubit_labels.copy()
        qubits.scale(2)
        qubits.arrange(DOWN)
        qubits.match_y(coord_vect)
        qubits.align_to(prob_rule, LEFT)
        dist_bars = always_redraw(lambda: self.get_dist_bars(get_state()**2, qubits, width_ratio=1.5))
        dist_bars.suspend_updating()

        dist_arrow = Arrow(coord_vect, qubits, buff=0.25)
        dist_arrow_label = Tex(R"c \rightarrow c^2", font_size=24)
        dist_arrow_label.next_to(dist_arrow, UP, SMALL_BUFF)

        bar_labels = VGroup(
            Integer(25, unit=R"\%", font_size=24),
            Integer(75, unit=R"\%", font_size=24),
        )

        def update_bar_labels(bar_labels):
            for label, bar, value in zip(bar_labels, dist_bars, get_state()):
                label.set_value(np.round(100 * value**2, 0))
                label.next_to(bar, RIGHT, SMALL_BUFF)

        bar_labels.add_updater(update_bar_labels)

        top_rule_rect = SurroundingRectangle(prob_rule[0])
        bar_rect = SurroundingRectangle(VGroup(qubits[0], bar_labels[0]))
        bar_rect.get_width()
        bar_rect.set_width(3, stretch=True, about_edge=LEFT)
        VGroup(top_rule_rect, bar_rect).set_stroke(YELLOW, 1.5)

        dist_group = VGroup(coord_vect, dist_arrow, dist_arrow_label, qubits, dist_bars, bar_labels)

        self.play(LaggedStart(
            TransformFromCopy(vector_label.copy().clear_updaters(), var_vect),
            frame.animate.center(),
            GrowArrow(var_arrow),
            FadeIn(prob_rule),
            run_time=2,
            lag_ratio=0.1
        ))
        self.play(ShowCreation(top_rule_rect))
        self.wait()
        self.play(LaggedStart(
            Transform(var_vect.copy(), coord_vect.copy().clear_updaters(), remover=True),
            TransformFromCopy(var_arrow, dist_arrow),
            FadeIn(dist_arrow_label, DOWN),
            TransformFromCopy(VGroup(pr[1:4] for pr in prob_rule).copy(), qubits),
            FadeTransformPieces(VGroup(pr[5:] for pr in prob_rule).copy(), dist_bars),
            TransformFromCopy(top_rule_rect, bar_rect),
            lag_ratio=1e-2
        ))
        self.add(coord_vect)
        self.play(FadeIn(bar_labels))
        self.wait()
        dist_bars.resume_updating()
        self.play(theta_tracker.animate.set_value(10 * DEG), run_time=8)
        self.wait()
        self.play(
            top_rule_rect.animate.match_y(prob_rule[1]),
            bar_rect.animate.match_y(qubits[1]),
        )
        self.play(theta_tracker.animate.set_value(90 * DEG), run_time=8)
        self.wait()
        self.play(FadeOut(top_rule_rect), FadeOut(bar_rect))
        self.wait()
        self.play(theta_tracker.animate.set_value(60 * DEG), run_time=2)

        # Note x^2 + y^2
        var_group = VGroup(var_vect, var_arrow, prob_rule)
        pythag = Tex(R"x^2 + y^2 = 1")
        pythag.match_x(var_group)
        pythag.to_edge(UP)

        self.play(LaggedStart(
            var_group.animate.shift(1.5 * DOWN),
            TransformFromCopy(prob_rule[0]["x^2"][0], pythag["x^2"][0]),
            Write(pythag["+"][0]),
            TransformFromCopy(prob_rule[1]["y^2"][0], pythag["y^2"][0]),
        ))
        self.play(Write(pythag["= 1"][0]), run_time=1)
        self.add(pythag)
        self.wait()

        # Show vector length
        brace = LineBrace(vector, DOWN, buff=0)
        brace_label = brace.get_tex(R"\sqrt{x^2 + y^2} = 1", font_size=36)
        brace_label.shift(MED_SMALL_BUFF * UP)

        circle = Circle(radius=vector.get_length())
        circle.move_to(plane)
        circle.set_stroke(YELLOW, 2)
        circle.rotate(vector.get_angle(), about_point=plane.get_center())

        vector_ghost = vector.copy()
        vector_ghost.clear_updaters()
        vector_ghost.set_opacity(0.25)

        self.play(
            GrowFromCenter(brace),
            TransformMatchingTex(pythag.copy(), brace_label, run_time=1)
        )
        self.wait()
        self.add(vector_ghost)
        self.play(
            ShowCreation(circle),
            theta_tracker.animate.set_value(theta_tracker.get_value() + TAU),
            run_time=6,
        )
        self.remove(vector_ghost)
        self.wait()
        self.play(
            FadeOut(brace),
            FadeOut(brace_label),
            circle.animate.set_stroke(width=1, opacity=0.5)
        )

        # Name this as a qubit
        title = Text("Qubit", font_size=90)
        title.next_to(plane.get_corner(UL), DR, MED_SMALL_BUFF)
        title.set_backstroke(BLACK, 4)

        self.play(Write(title))
        self.wait()

        # Illustrate collpase
        if False:
            # Just for an insertion
            small_plane = self.small_plane(plane)
            small_plane.set_z_index(-1)
            self.remove(plane, vector_label, pythag)
            self.add(small_plane)
            self.remove(title)
            theta_tracker.set_value(45 * DEG)

            self.wait()
            self.play(theta_tracker.animate.set_value(90 * DEG), run_time=0.15)
            self.wait()
            self.play(theta_tracker.animate.set_value(45 * DEG))
            self.wait()
            self.play(theta_tracker.animate.set_value(0), run_time=0.15)
            self.wait()

            # Put in the qubits
            one = one_label.copy()
            zero = zero_label.copy()
            for mob in [zero, one]:
                mob.set_height(1)
                mob.set_fill(WHITE, 1, border_width=3)
                mob.move_to(plane)

            self.clear()
            self.add(zero)

        # Ambient change
        theta_tracker.set_value(60 * DEG)
        for value in [180 * DEG, 90 * DEG, 0, 120 * DEG]:
            self.play(theta_tracker.animate.set_value(value), run_time=6)

        # Show kets
        self.play(FadeOut(vector_label))
        self.play(theta_tracker.animate.set_value(0), run_time=2)

        zero_vect = vector.copy().clear_updaters().set_fill(BLUE, 0.5)
        self.play(zero_label.animate.scale(2).next_to(zero_vect, DOWN, buff=0))
        self.play(
            FlashAround(zero_label, run_time=2, time_width=1.5, color=BLUE),
            FadeIn(zero_vect),
        )
        self.wait()
        self.play(theta_tracker.animate.set_value(90 * DEG), run_time=2)

        one_vect = vector.copy().clear_updaters().set_fill(GREEN, 0.5)
        self.play(
            one_label.animate.scale(2).next_to(one_vect, LEFT, buff=0)
        )
        self.play(
            FlashAround(one_label, run_time=2, time_width=1.5, color=GREEN),
            FadeIn(one_vect),
        )
        self.wait()

        # General unit vector
        var_vect_copy = var_vect.copy()
        self.play(theta_tracker.animate.set_value(55 * DEG), run_time=2)
        self.play(var_vect_copy.animate.scale(0.75).next_to(vector.get_end(), UR, buff=0))

        weighted_sum = Tex(R"x|0\rangle + y|1\rangle")
        weighted_sum.next_to(vector.get_end(), RIGHT)
        weighted_sum.shift(SMALL_BUFF * UL)
        weighted_sum.set_backstroke(BLACK, 5)
        red_cross = Cross(var_vect_copy)
        red_cross.scale(1.5)

        self.play(ShowCreation(red_cross))
        self.wait()
        self.play(
            FadeOut(red_cross),
            FadeTransform(var_vect_copy.get_entries()[0].copy(), weighted_sum["x"]),
            FadeTransform(var_vect_copy.get_entries()[1].copy(), weighted_sum["y"]),
            FadeOut(var_vect_copy),
            FadeTransform(zero_label.copy(), weighted_sum[R"|0\rangle"]),
            FadeTransform(one_label.copy(), weighted_sum[R"|1\rangle"]),
            FadeIn(weighted_sum["+"]),
        )
        self.wait()

        # Prepare to show a gate
        faders = VGroup(
            pythag, var_group, dist_group
        )
        faders.clear_updaters()

        plane2 = plane.copy()
        plane2.next_to(plane, RIGHT, buff=5)
        planes = VGroup(plane, plane2)

        arrow = Arrow(plane, plane2, thickness=8)
        arrow_label = Text("Hadamard gate", font_size=60)
        arrow_label.next_to(arrow, UP, SMALL_BUFF)
        matrix_tex = Tex(R"\frac{1}{\sqrt{2}} \left[\begin{array}{cc} 1 & 1 \\ 1 & -1 \end{array}\right]", font_size=24)
        matrix_tex.set_fill(GREY_B)
        matrix_tex.next_to(arrow, DOWN, SMALL_BUFF)

        self.play(
            frame.animate.set_width(planes.get_width() + LARGE_BUFF).move_to(planes),
            LaggedStartMap(FadeOut, faders, shift=0.5 * DOWN, lag_ratio=0.25),
            zero_vect.animate.set_fill(opacity=1),
            one_vect.animate.set_fill(opacity=1),
            FadeIn(plane2, shift=RIGHT),
            GrowArrow(arrow),
            FadeIn(arrow_label, lag_ratio=0.1),
            FadeIn(matrix_tex),
            run_time=2
        )

        # Hadamard gate
        vector.clear_updaters()
        movers = VGroup(circle, zero_vect, one_vect, vector)
        movers_image = movers.copy()
        movers_image.flip(axis=rotate_vector(RIGHT, PI / 8), about_point=plane.get_center())
        movers_image.move_to(plane2)

        labels = VGroup(zero_label, one_label, weighted_sum)
        labels.set_backstroke(BLACK, 3)
        labels_image = VGroup(
            Tex(R"\text{H}|0\rangle").scale(0.75).rotate(45 * DEG).next_to(movers_image[1].get_center(), UL, buff=-0.05),
            Tex(R"\text{H}|1\rangle").scale(0.75).rotate(-45 * DEG).next_to(movers_image[2].get_center(), DL, buff=-0.05),
            Tex(R"\text{H}\big(x|0\rangle + y|1\rangle\big)").scale(0.75).next_to(movers_image[3].get_end(), RIGHT, SMALL_BUFF),
        )
        labels_image.set_backstroke(BLACK, 7)

        self.play(
            TransformFromCopy(movers, movers_image, path_arc=-30 * DEG),
            TransformFromCopy(labels, labels_image, path_arc=-30 * DEG),
            run_time=2
        )
        self.wait()

        # Go through each part
        faders = VGroup(movers[1:], movers_image[1:], labels, labels_image)
        for index in range(2):
            faders.generate_target()
            for mob in faders.target:
                for j, part in enumerate(mob):
                    if j == index:
                        part.set_opacity(1)
                    else:
                        part.set_opacity(0.25)

            rect = SurroundingRectangle(labels[index])
            rect.set_stroke(YELLOW, 2)
            self.play(
                MoveToTarget(faders),
                ShowCreation(rect),
            )
            self.play(
                TransformFromCopy(movers[index + 1], movers_image[index + 1], path_arc=-30 * DEG),
                TransformFromCopy(labels[index], labels_image[index], path_arc=-30 * DEG),
                rect.animate.surround(labels_image[index]).set_anim_args(path_arc=-30 * DEG),
                run_time=2,
            )
            self.play(FadeOut(rect))
            self.wait()
        self.play(faders.animate.set_fill(opacity=1))

    def get_plane(self):
        plane = NumberPlane((-2, 2), (-2, 2), faded_line_ratio=5)
        plane.set_height(7.5)
        plane.to_edge(LEFT, buff=MED_SMALL_BUFF)
        return plane

    def get_small_plane(self, plane):
        x_range = (-1, 1 - 1e-5)
        small_plane = NumberPlane(x_range, x_range, faded_line_ratio=5)
        small_plane.set_height(0.5 * plane.get_height())
        small_plane.move_to(plane)
        return small_plane

    def get_qubit_labels(self, plane):
        zero, one = bits = VGroup(Integer(0), Integer(1))
        zero_label, one_label = qubit_labels = VGroup(
            VGroup(Ket(bit), bit)
            for bit in bits
        )
        qubit_labels.scale(0.5)
        zero_label.next_to(plane.c2p(1, 0), DR, SMALL_BUFF)
        one_label.next_to(plane.c2p(0, 1), DR, SMALL_BUFF)
        return qubit_labels

    def get_vector(self, plane, x=1, y=0, fill_color=TEAL, thickness=6):
        return Arrow(
            plane.c2p(0, 0),
            plane.c2p(x, y),
            buff=0,
            thickness=thickness,
            fill_color=fill_color
        )

    def thumbnail_insert(self):
        # To be put above the "note x^2 + y^2" above
        self.remove(vector_label)
        self.remove(plane)
        small_plane = self.get_small_plane(plane)
        small_plane.set_z_index(-1)
        small_plane.axes.set_stroke(WHITE, 4)
        small_plane.background_lines.set_stroke(BLUE, 3)
        small_plane.faded_lines.set_stroke(BLUE, 2, 0.8)
        qubit_labels.set_fill(border_width=2)
        vector.set_color(YELLOW)
        theta_tracker.set_value(45 * DEG)
        self.add(small_plane)

        # Add glass
        glass = get_magnifying_glass()
        glass.set_height(5)
        glass[0].set_fill(BLACK)
        glass.shift(vector.get_center() - glass[0].get_center())
        one = KetGroup(Integer(1))
        one.set_height(1.5)
        one.move_to(glass[0])
        self.add(glass, one)

    def z_filp_insert(self):
        # For the clarification supplement
        angle = theta_tracker.get_value()
        self.play(theta_tracker.animate.set_value(angle + TAU), run_time=5)
        theta_tracker.set_value(angle)

        # Flips
        flipper = VGroup(vector, circle, zero_vect, one_vect)
        flipper.clear_updaters()
        self.play(
            Rotate(flipper, PI, axis=RIGHT, about_point=plane.c2p(0, 0), run_time=6, rate_func=there_and_back_with_pause)
        )


class ShowAFewFlips(Qubit):
    def construct(self):
        # Set up
        title = Text("Quantum Gates", font_size=60)
        title.to_edge(UP)

        plane = self.get_plane()
        plane.center()
        small_plane = self.get_small_plane(plane)
        qubit_labels = self.get_qubit_labels(plane)
        vector = self.get_vector(plane)

        self.add(title)
        self.add(small_plane, qubit_labels, vector)

        # Show H, Z and X gates
        lines = DashedLine(2 * LEFT, 2 * RIGHT).replicate(3)
        for line, angle in zip(lines, [0, PI / 8, PI / 4]):
            line.rotate(angle)
        lines.set_stroke(YELLOW, 2)

        gate_labels = VGroup(Text(c) for c in "ZHX")
        gate_labels.next_to(plane.c2p(1, 1), DR)

        for i in [1, 0, 2, 1, 2, 1, 0, 1]:
            self.play(LaggedStart(
                AnimationGroup(
                    FadeIn(lines[i]),
                    FadeIn(gate_labels[i])
                ),
                Rotate(vector, PI, axis=lines[i].get_vector(), about_point=ORIGIN),
                lag_ratio=0.5
            ))
            self.play(
                FadeOut(lines[i]),
                FadeOut(gate_labels[i]),
            )


class ExponentiallyGrowingState(InteractiveScene):
    def construct(self):
        # Initialize vector
        label = TexText(R"State of a\\ 1 qubit computer")
        n_label = label.make_number_changeable("1", edge_to_fix=RIGHT)
        n_label.set_color(YELLOW)
        label.move_to(3.5 * LEFT)
        vect = self.get_state_vector(1)
        vect.move_to(1.5 * RIGHT)

        brace = self.get_brace_group(vect, 1)
        brace.set_opacity(0)

        self.add(label)
        self.add(vect)

        # Grow the vector
        for n in range(2, 9):
            new_vect = self.get_state_vector(n)
            new_vect.move_to(vect)
            new_brace = self.get_brace_group(new_vect, n)

            n_label.set_value(n)
            self.play(
                ReplacementTransform(vect[0], new_vect[0]),
                FadeTransform(vect[1], new_vect[1]),
                FadeTransformPieces(vect[2], new_vect[2]),
                FadeTransform(brace[0], new_brace[0]),
                FadeTransform(brace[1], new_brace[1]),
            )
            self.wait()

            vect = new_vect
            brace = new_brace

        # Change to 100
        frame = self.frame

        new_brace_label = Tex(R"2^{100}")
        new_brace_label.set_color(YELLOW)
        new_brace_label.next_to(brace[0], RIGHT).shift(SMALL_BUFF * UR)

        top_eq = Tex(R"2^{100} = " + "{:,}".format(2**100))
        top_eq.next_to(vect, UP, MED_LARGE_BUFF).set_x(0)

        self.play(
            FadeTransform(brace[1], new_brace_label),
            ChangeDecimalToValue(n_label, 100),
            frame.animate.set_height(9, about_edge=DOWN),
            FadeIn(top_eq),
            FadeTransform(vect, self.get_state_vector(n + 2).move_to(vect, RIGHT)),
        )
        self.wait()

    def get_state_vector(self, n):
        # Actualy function
        values = normalize(np.random.uniform(-1, 1, 2**n))
        array = DecimalMatrix(
            values.reshape((2**n, 1)),
            decimal_config=dict(include_sign=True)
        )
        array.set_max_height(7)

        bit_strings = VGroup(
            BitString(k, length=n)
            for k in range(2**n)
        )
        bit_strings.set_color(GREY)
        for bits, entry in zip(bit_strings, array.get_entries()):
            bits.set_max_height(entry.get_height())
            bits.next_to(array, LEFT, buff=0.2)
            bits.match_y(entry)

        return VGroup(bit_strings, array.get_brackets(), array.get_entries())

    def get_brace_group(self, vect, n):
        brace = Brace(vect, RIGHT, buff=0.25)
        label = brace.get_tex(Rf"2^{{{n}}} = {2**n}")
        label.shift(SMALL_BUFF * UR)
        return VGroup(brace, label)


class InvisibleStateValues(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        n = 5
        values = np.random.uniform(-1, 1, 2**n)
        vect = DecimalMatrix(values.reshape(-1, 1), decimal_config=dict(include_sign=True))
        vect.set_height(FRAME_HEIGHT - 1)
        indices = VGroup(BitString(k, 5) for k in range(2**n))
        for index, elem in zip(indices, vect.elements):
            index.match_height(elem)
            index.set_color(GREY_C)
            index.next_to(vect, LEFT, SMALL_BUFF)
            index.match_y(elem)

        rects = VGroup(SurroundingRectangle(elem, buff=0.05) for elem in vect.elements)
        rects.set_stroke(YELLOW, 1)

        frame.set_height(2, about_edge=UP)
        self.add(vect, indices)
        self.play(
            LaggedStartMap(ShowCreation, rects, lag_ratio=0.25),
            frame.animate.to_default_state(),
            run_time=5,
        )

        rects.target = rects.generate_target()
        rects.target.set_stroke(WHITE, 1)
        rects.target.set_fill(GREY_D, 1)
        self.play(MoveToTarget(rects, lag_ratio=0.1, run_time=3))
        self.wait()

        # Shrink down
        group = Group(indices, vect, Point(), rects)
        self.play(group.animate.set_height(4).move_to(4 * RIGHT), run_time=2)
        self.wait()

        # Revealed value
        value = KetGroup(BitString(13, 5))
        value.move_to(group)

        self.clear()
        self.add(value)
        self.wait()


class ThreeDSample(InteractiveScene):
    def construct(self):
        # Set up axes
        frame = self.frame
        x_range = y_range = z_range = (-2, 2)
        axes = ThreeDAxes(x_range, y_range, z_range, axis_config=dict(tick_size=0.05))
        plane = NumberPlane(x_range, y_range, faded_line_ratio=5)
        plane.axes.set_opacity(0)
        plane.background_lines.set_stroke(BLUE, 1, 0.5)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)

        rot_vel_tracker = ValueTracker(DEG)

        frame.reorient(-31, 71, 0, (0.22, 0.17, 0.13), 2.88)
        frame.add_updater(lambda m, dt: m.increment_theta(dt * rot_vel_tracker.get_value()))
        self.add(axes, plane, Point())

        # Add vector
        vector = Vector(normalize([1, -1, 2]), thickness=2, fill_color=TEAL)
        vector.always.set_perpendicular_to_camera(self.frame)
        vector.set_z_index(2)

        coord_array = DecimalMatrix(
            np.zeros((3, 1)),
            decimal_config=dict(include_sign=True)
        )
        coord_array.set_height(1.25)
        coord_array.to_corner(UL)
        coord_array.fix_in_frame()

        def update_coord_array(coord_array):
            for elem, coord in zip(coord_array.elements, vector.get_end()):
                elem.set_value(coord)
            coord_array.set_fill(GREY_B, border_width=1)

        coord_array.add_updater(update_coord_array)

        self.play(
            GrowArrow(vector),
            VFadeIn(coord_array),
        )
        self.play(Rotate(vector, TAU, axis=OUT, about_point=ORIGIN, run_time=6))
        self.wait(5)

        # Show 0, 1, and 2 directions
        symbols = VGroup(KetGroup(Integer(n)) for n in range(3))
        symbols.scale(0.5)
        symbols.set_backstroke(BLACK, 5)
        symbols.rotate(90 * DEG, RIGHT)
        directions = [UP + 0.5 * OUT, LEFT, LEFT]

        for symbol, direction, trg_coords in zip(symbols, directions, np.identity(3)):
            symbol.next_to(0.5 * trg_coords, direction, SMALL_BUFF)
            self.play(
                self.set_vect_anim(vector, trg_coords),
                FadeIn(symbol, 0.25 * OUT)
            )
            self.play(symbol.animate.scale(0.5).next_to(trg_coords, direction, SMALL_BUFF))
            self.wait(0.5)
        self.wait(2)

        # Highlight a secret key
        key_icon = get_key_icon()
        key_icon.rotate(90 * DEG, RIGHT)
        key_icon.match_depth(symbols[2])
        key_icon.next_to(symbols[2], LEFT, buff=0.05)

        self.play(
            FadeIn(key_icon, 0.25 * LEFT),
            symbols[2].animate.set_fill(YELLOW),
        )
        self.wait(3)

        # Go to the balanced state
        coord_rects = VGroup(
            SurroundingRectangle(elem)
            for elem in coord_array.elements
        )
        coord_rects.set_stroke(TEAL, 3)
        coord_rects.fix_in_frame()

        balanced_state = normalize([1, 1, 1])
        balance_name = Text("balanced state", font_size=16)
        balance_ket = KetGroup(Text("b", font_size=16), buff=0.035)
        for mob in [balance_name, balance_ket]:
            mob.rotate(90 * DEG, RIGHT)
            mob.next_to(balanced_state, OUT + RIGHT, buff=0.025)

        self.play(
            self.set_vect_anim(vector, balanced_state, run_time=4),
            FadeIn(balance_name, lag_ratio=0.1, time_span=(3, 4)),
        )
        self.play(LaggedStartMap(ShowCreation, coord_rects, lag_ratio=0.25))
        self.play(LaggedStartMap(FadeOut, coord_rects, lag_ratio=0.25))
        self.wait()
        self.play(
            FadeTransformPieces(balance_name, balance_ket[1]),
            Write(balance_ket[0])
        )
        self.play(rot_vel_tracker.animate.set_value(-DEG), run_time=3)
        self.wait(7)

        # Show the goal
        z_vect = vector.copy()
        z_vect.set_fill(YELLOW)

        tail = TracingTail(z_vect.get_end, stroke_color=YELLOW, time_traced=3)
        self.add(tail)
        self.wait(2)
        self.play(FadeIn(z_vect))
        self.play(
            self.set_vect_anim(z_vect, OUT, run_time=2)
        )
        self.wait(3)
        self.remove(tail)

        # Show 2d slice
        v_slice = NumberPlane(x_range, y_range, faded_line_ratio=5)
        v_slice.rotate(90 * DEG, RIGHT)
        v_slice.rotate(45 * DEG, OUT)
        v_slice.axes.set_stroke(WHITE, 1, 0.5)
        v_slice.background_lines.set_stroke(BLUE, 1, 1)
        v_slice.faded_lines.set_stroke(BLUE, 0.5, 0.25)

        b_vect_ghost = vector.copy()
        b_vect_ghost.set_fill(opacity=0.5)
        tail = TracingTail(vector.get_end, stroke_color=TEAL, time_traced=8)
        symbols[2].set_z_index(1)

        self.add(tail)
        self.play(
            frame.animate.reorient(43, 79, 0, (-0.26, 0.37, 0.15), 4.97),
            rot_vel_tracker.animate.set_value(-2 * DEG),
            FadeIn(v_slice),
            plane.animate.fade(0.75),
            axes.animate.set_stroke(opacity=0.5),
            run_time=3
        )
        self.add(b_vect_ghost)
        self.play(
            Rotate(
                vector,
                TAU,
                axis=np.cross(vector.get_end(), OUT),
                run_time=8,
                about_point=ORIGIN,
            )
        )
        self.play(
            frame.animate.reorient(-1, 79, 0, (-0.13, 0.55, 1.08), 7.67),
            coord_array.animate.set_x(0),
            run_time=2,
        )
        tail.clear_updaters()
        self.play(FadeOut(tail))
        self.wait(20)

        # Show xy line
        xy_line = v_slice.x_axis.copy()
        xy_line.set_stroke(WHITE, 3, 1)
        self.play(
            frame.animate.reorient(67, 77, 0, (-0.57, 0.12, 0.86), 6.62),
            run_time=2
        )
        self.wait(4)
        self.play(
            GrowFromCenter(xy_line),
            self.set_vect_anim(vector, normalize(UR)),
        )
        self.wait(10)
        self.play(self.set_vect_anim(vector, balanced_state))
        self.wait(20)

    def set_vect_anim(self, vector, trg_coords, run_time=1, **kwargs):
        return Rotate(
            vector,
            angle_between_vectors(vector.get_end(), trg_coords),
            axis=np.cross(vector.get_end(), trg_coords),
            about_point=ORIGIN,
            run_time=run_time,
            **kwargs
        )

    def flip_along_key_axis(self):
        # To be inserted after highlighting the key state above
        sphere = Sphere(radius=vector.get_length())
        mesh = SurfaceMesh(sphere, resolution=(51, 101))
        mesh.set_stroke(WHITE, 2, 0.1)

        key_vect = vector.copy()
        key_vect.clear_updaters()
        key_vect.set_fill(YELLOW, 0.5)

        self.add(key_vect)
        self.play(
            self.set_vect_anim(vector, normalize([1, 1, 1])),
            FadeIn(mesh)
        )
        vector.clear_updaters()
        for _ in range(4):
            self.play(
                Group(mesh, vector, key_vect).animate.stretch(-1, 2, about_point=ORIGIN),
                run_time=2
            )
            self.wait()


class GroversAlgorithm(InteractiveScene):
    def construct(self):
        # Set up plane
        x_range = y_range = (-1, 1 - 1e-6)
        plane = NumberPlane(x_range, y_range, faded_line_ratio=5)
        plane.set_height(6)
        plane.background_lines.set_stroke(BLUE, 1, 1)
        plane.faded_lines.set_stroke(BLUE, 1, 0.25)

        self.add(plane)

        # Add key and balance directions
        key_vect = Vector(plane.c2p(0, 1), thickness=5, fill_color=YELLOW)
        b_vect = key_vect.copy().rotate(- np.arccos(1 / math.sqrt(3)), about_point=ORIGIN)
        b_vect.set_fill(TEAL)

        key_label, b_label = labels = VGroup(
            KetGroup(Tex(char))
            for char in "kb"
        )
        labels.set_submobject_colors_by_gradient(YELLOW, TEAL)
        labels.set_backstroke(BLACK, 3)
        for label in labels:
            label[1].shift(0.02 * UR)
        key_label.next_to(key_vect.get_end(), UP, SMALL_BUFF)
        b_label.next_to(b_vect.get_end(), UR, SMALL_BUFF)
        key_icon = get_key_icon()
        key_icon.set_height(0.75 * key_label.get_height())
        key_icon.next_to(key_label, LEFT, SMALL_BUFF)

        self.add(key_vect, b_vect)
        self.add(key_label, b_label, key_icon)
        self.wait()

        # Highlight key
        self.play(GrowArrow(key_vect))
        self.play(FlashAround(key_label, time_width=1.5, run_time=2))

        # Label the x-direction
        x_vect = Vector(plane.c2p(1, 0), thickness=5)
        x_vect.set_fill(WHITE)
        x_label_example, x_label_general = x_labels = VGroup(
            Tex(R"\frac{1}{\sqrt{2}}\big(|0\rangle + |1\rangle \big)", font_size=36),
            Tex(R"\frac{1}{\sqrt{N - 1}} \sum_{n \ne k} |n \rangle", font_size=24),
        )
        for label in x_labels:
            label.next_to(x_vect.get_end(), RIGHT)

        x_label_general.shift(SMALL_BUFF * DL)

        self.play(
            TransformFromCopy(key_vect, x_vect, path_arc=-90 * DEG),
            FadeIn(x_label_general, time_span=(0.5, 1.5)),
        )
        self.wait()

        x_label_general.save_state()
        self.play(
            FadeIn(x_label_example, DOWN),
            x_label_general.animate.fade(0.5).shift(1.25 * DOWN),
        )
        self.wait()

        # Show component of b in the direction of key
        rhs = Tex(R"= \frac{1}{\sqrt{3}}\big(|0\rangle + |1\rangle + |2\rangle \big)", font_size=36)
        rhs.next_to(b_label, RIGHT)
        rhs.set_color(TEAL)

        b_vect_proj = Vector(plane.c2p(0, plane.p2c(b_vect.get_end())[1]), thickness=5)
        b_vect_proj.set_fill(TEAL_E, 1)
        dashed_line = DashedLine(b_vect.get_end(), b_vect_proj.get_end())

        self.play(FadeIn(rhs, lag_ratio=0.1))
        self.wait()
        self.play(
            TransformFromCopy(b_vect, b_vect_proj),
            ShowCreation(dashed_line),
        )
        self.wait()
        self.play(
            FlashAround(rhs[R"|2\rangle"], time_width=1.5, run_time=3),
            rhs[R"|2\rangle"].animate.set_color(YELLOW),
        )
        self.wait()

        # Set up N
        N_eq = Tex(R"N = 3")
        dim = N_eq.make_number_changeable("3", edge_to_fix=LEFT)
        N_eq.next_to(plane, UR, LARGE_BUFF).shift_onto_screen()

        self.play(
            LaggedStart(
                FadeOut(rhs),
                Restore(x_label_general),
                FadeOut(x_label_example, 0.5 * UP),
            ),
            Write(N_eq)
        )

        # Increase N
        N_tracker = ValueTracker(3)
        get_N = N_tracker.get_value

        def update_b_vects(vects):
            N = int(get_N())
            x = math.sqrt(1 - 1.0 / N)
            y = 1 / math.sqrt(N)
            vects[0].put_start_and_end_on(plane.c2p(0, 0), plane.c2p(x, y))
            vects[1].put_start_and_end_on(plane.c2p(0, 0), plane.c2p(0, y))

        self.play(
            N_tracker.animate.set_value(100).set_anim_args(rate_func=rush_into),
            UpdateFromFunc(dim, lambda m: m.set_value(int(get_N()))),
            UpdateFromFunc(VGroup(b_vect, b_vect_proj), update_b_vects),
            UpdateFromFunc(dashed_line, lambda m: m.become(DashedLine(b_vect.get_end(), b_vect_proj.get_end()))),
            UpdateFromFunc(b_label, lambda m: m.next_to(b_vect.get_end(), UR, SMALL_BUFF)),
            run_time=12
        )
        self.wait()

        # Reference the angle
        alpha = math.acos(1 / math.sqrt(N_tracker.get_value()))
        arc = Arc(90 * DEG, -alpha, radius=0.5)
        alpha_label = Tex(R"\alpha")
        alpha_label.next_to(arc.pfp(0.5), UR, SMALL_BUFF)
        lt_90 = Tex(R"< 90^\circ")
        lt_90.next_to(alpha_label, RIGHT).shift(0.05 * UL)

        self.play(
            FadeOut(b_vect_proj),
            FadeOut(dashed_line),
            ShowCreation(arc),
            FadeIn(alpha_label),
        )
        self.play(Write(lt_90))
        self.wait()
        self.play(FadeIn(VGroup(b_vect_proj, dashed_line), run_time=3, rate_func=there_and_back_with_pause, remover=True))
        self.wait()

        # Show the dot product
        frame = self.frame

        b_comp_tex = R"1 / \sqrt{N}"
        b_array = TexMatrix(np.array([
            *2 * [b_comp_tex],
            R"\vdots",
            *3 * [b_comp_tex],
            R"\vdots",
            *2 * [b_comp_tex],
        ]).reshape(-1, 1))
        k_array = TexMatrix(np.array([
            *2 * ["0"],
            R"\vdots",
            "0", "1", "0",
            R"\vdots",
            *2 * ["0"],
        ]).reshape(-1, 1))
        arrays = VGroup(k_array, b_array)

        for array in arrays:
            array.set_height(6)

        arrays.arrange(LEFT, buff=MED_LARGE_BUFF)
        arrays.next_to(plane, RIGHT, buff=2.5).to_edge(DOWN, buff=MED_LARGE_BUFF)
        dot = Tex(R"\cdot", font_size=72)
        dot.move_to(midpoint(b_array.get_right(), k_array.get_left()))

        k_array_label, b_array_label = array_labels = labels.copy()

        for label, arr in zip(array_labels, arrays):
            arr.set_fill(interpolate_color(label.get_fill_color(), WHITE, 0.2), 1)
            label.next_to(arr, UP, MED_LARGE_BUFF)

        self.play(
            GrowFromPoint(b_array, b_label.get_center()),
            TransformFromCopy(b_label, b_array_label),
            N_eq.animate.to_edge(UP, MED_SMALL_BUFF).set_x(2),
            frame.animate.set_x(4),
            FadeOut(x_label_general),
            FadeOut(x_vect),
            run_time=2
        )
        self.play(
            GrowFromPoint(k_array, key_label.get_center()),
            TransformFromCopy(key_label, k_array_label),
            Write(dot),
            run_time=2
        )
        self.wait()

        # Evaluate the dot product
        elem_rects = VGroup(
            SurroundingRectangle(VGroup(*elems), buff=0.05).set_width(2.5, stretch=True)
            for elems in zip(b_array.elements, k_array.elements)
        )
        for rect in elem_rects:
            rect.set_stroke(WHITE, 2)
            rect.align_to(elem_rects[0], RIGHT)

        equals = Tex(R"=").next_to(arrays, RIGHT)
        rhs = Tex(R"1 / \sqrt{N}")
        rhs.next_to(equals, RIGHT)

        self.play(Write(equals))
        self.play(
            LaggedStartMap(VFadeInThenOut, elem_rects, lag_ratio=0.2, run_time=5),
            FadeIn(rhs, time_span=(1.75, 2.25)),
        )
        self.wait()

        # Show cosine expression
        cos_expr = Tex(R"\cos(\alpha) = 1 / \sqrt{N}", font_size=36)
        cos_expr.next_to(alpha_label, UP, MED_LARGE_BUFF, aligned_edge=LEFT)

        self.play(LaggedStart(
            FadeOut(lt_90),
            Write(cos_expr[R"\cos("]),
            Write(cos_expr[R") ="]),
            TransformFromCopy(alpha_label, cos_expr[R"\alpha"][0]),
            TransformFromCopy(rhs, cos_expr[R"1 / \sqrt{N}"][0]),
            lag_ratio=0.1,
        ))
        self.wait()

        # Show sine of smaller angle
        target_thickness = 3

        for vect in [b_vect, key_vect]:
            vect.target = Arrow(ORIGIN, vect.get_end(), thickness=target_thickness, buff=0)
            vect.target.match_style(vect)

        theta = 90 * DEG - alpha
        theta_arc = Arc(0, theta, radius=2.0)
        theta_label = Tex(R"\theta", font_size=24)
        theta_label.next_to(theta_arc, RIGHT, buff=0.1)
        alpha_label.set_backstroke(BLACK, 5)
        theta_label.set_backstroke(BLACK, 5)

        sin_expr = Tex(R"\sin(\theta) = 1 / \sqrt{N}", font_size=36)
        sin_expr.move_to(cos_expr).set_y(-0.5)

        theta_approx = Tex(R"\theta \approx 1 / \sqrt{N}", font_size=36)
        theta_approx.next_to(sin_expr, DOWN, aligned_edge=RIGHT)

        cos_group = VGroup(arc, alpha_label, cos_expr)

        self.play(
            MoveToTarget(b_vect),
            MoveToTarget(key_vect),
            LaggedStart(
                TransformFromCopy(arc, theta_arc),
                TransformFromCopy(alpha_label, theta_label),
                TransformFromCopy(cos_expr, sin_expr),
                lag_ratio=0.25
            ),
            cos_group.animate.fade(0.5)
        )
        self.wait()
        self.play(
            TransformMatchingTex(
                sin_expr.copy(),
                theta_approx,
                matched_keys=[R"1 / \sqrt{N}"],
                key_map={"=": R"\approx"}
            )
        )
        self.wait()

        # Put it in the corner
        self.play(
            LaggedStartMap(FadeOut, VGroup(array_labels, arrays, dot, equals, rhs, cos_group, sin_expr), lag_ratio=0.25),
            frame.animate.set_x(-2),
            N_eq.animate.set_x(-2),
            theta_approx.animate.to_corner(UR, buff=MED_SMALL_BUFF).shift(2 * LEFT),
            run_time=2
        )
        self.wait()

        # Add vector components
        vector = b_vect.copy()

        initial_coords = 0.1 * np.ones(11)
        vect_coords = DecimalMatrix(initial_coords.reshape(-1, 1), num_decimal_places=3, decimal_config=dict(include_sign=True))
        vect_coords.set_height(6)
        vect_coords.next_to(frame.get_left(), RIGHT, MED_LARGE_BUFF)
        mid_index = len(initial_coords) // 2
        dot_indices = [mid_index - 2, mid_index + 2]
        arr_dots = VGroup()
        for index in dot_indices:
            element = vect_coords.elements[index]
            dots = Tex(R"\vdots")
            dots.move_to(element)
            element.become(dots)
            arr_dots.add(element)

        vect_coords.set_fill(GREY_B)

        def update_vect_coords(vect_coords):
            x, y = plane.p2c(vector.get_end())
            x /= math.sqrt(99)
            for n, elem in enumerate(vect_coords.elements):
                if n in dot_indices:
                    continue
                elif n == mid_index:
                    elem.set_value(y)
                else:
                    elem.set_value(x)

        vect_coords.add_updater(update_vect_coords)

        key_icon2 = get_key_icon(height=0.25)
        key_icon2.next_to(vect_coords, LEFT, SMALL_BUFF)

        self.add(vect_coords)
        self.add(key_icon2)

        # Add bars
        min_bar_width = 0.125
        bar_height = 0.25

        dec_elements = VGroup(
            elem for n, elem in enumerate(vect_coords.elements) if n not in dot_indices
        )
        bars = Rectangle(min_bar_width, bar_height).replicate(len(dec_elements))
        bars.set_fill(opacity=1)
        bars.set_submobject_colors_by_gradient(BLUE, GREEN)
        bars.set_stroke(WHITE, 0.5)

        for bar, elem in zip(bars, dec_elements):
            bar.next_to(vect_coords, RIGHT)
            bar.match_y(elem)
            bar.elem = elem

        def update_bars(bars):
            x, y = plane.p2c(vector.get_end())
            x /= math.sqrt(99)
            mid_index = len(bars) // 2
            for n, bar in enumerate(bars):
                if n == mid_index:
                    prob = 1 - 99 * (x**2)
                else:
                    prob = x**2
                width = min_bar_width * np.sqrt(prob / 0.01)
                bar.set_width(width, about_edge=LEFT, stretch=True)

        self.add(bars)

        # Show the flips
        circle = Circle(radius=0.5 * plane.get_width())
        circle.set_stroke(WHITE, 1)
        b_vect.set_fill(opacity=0.5)

        frame.set_field_of_view(20 * DEG)

        diag_line = DashedLine(-b_vect.get_end(), b_vect.get_end())
        h_line = DashedLine(plane.get_left(), plane.get_right())
        VGroup(h_line, diag_line).set_stroke(WHITE, 2)

        flip_line = h_line.copy()

        flipper = VGroup(circle, vector)

        vect_ghosts = VGroup()

        def right_filp(run_time=2):
            self.play(
                Rotate(flipper, PI, RIGHT, about_point=ORIGIN),
                vect_ghosts.animate.set_fill(opacity=0.25),
                run_time=run_time
            )

        def diag_flip(run_time=2, draw_time=0.5):
            self.play(ShowCreation(diag_line, run_time=draw_time))
            self.play(
                flipper.animate.flip(axis=diag_line.get_vector(), about_point=ORIGIN),
                vect_ghosts.animate.set_fill(opacity=0.25),
                UpdateFromFunc(bars, update_bars),
                run_time=run_time
            )
            self.play(FadeOut(diag_line, run_time=2 * draw_time))

        self.add(vect_ghosts, vector)
        self.play(FadeIn(circle))
        for n in range(4):
            vect_ghosts.add(vector.copy())
            right_filp()
            self.wait()
            vect_ghosts.add(vector.copy())
            diag_flip(draw_time=(0.5 if n < 2 else 1 / 30))
            self.wait()

        vect_ghosts.add(vect.copy()).set_fill(opacity=0.25)

        # Show vertical component
        if False:  # For an insertion
            # Show vert component
            self.add(diag_line)

            x, y = plane.p2c(vector.get_end())
            v_part = Arrow(plane.get_origin(), plane.c2p(0, y), thickness=4, buff=0, max_width_to_length_ratio=0.25)
            v_part.set_fill(GREEN)
            h_line = DashedLine(vector.get_end(), v_part.get_end())
            brace = Brace(v_part, LEFT, buff=0.05)

            self.play(
                FadeOut(diag_line),
                TransformFromCopy(vector, v_part),
                ShowCreation(h_line),
            )
            self.wait()
            self.play(GrowFromCenter(brace))
            self.wait()

        # Show steps of 2 * theta
        bars.add_updater(update_bars)

        arcs = VGroup(
            Arc(n * theta, 2 * theta, radius=circle.get_radius())
            for n in range(1, len(vect_ghosts), 2)
        )
        for arc, color in zip(arcs, it.cycle([BLUE, RED])):
            arc.set_stroke(color, 3)

        arc_labels = VGroup(
            Tex(R"2\theta", font_size=24).next_to(arc.pfp(0.5), normalize(arc.pfp(0.5)), SMALL_BUFF)
            for arc in arcs
        )

        self.play(
            Rotate(
                vector,
                -angle_between_vectors(vector.get_vector(), b_vect.get_vector()),
                about_point=ORIGIN
            ),
            FadeOut(b_label)
        )
        self.wait()
        right_filp()
        diag_flip(draw_time=0.25)
        self.wait()
        self.play(
            TransformFromCopy(vect_ghosts[0], vector, path_arc=theta),
            ShowCreation(arcs[0])
        )
        self.play(TransformFromCopy(theta_label, arc_labels[0]))
        self.wait()
        for arc, label in zip(arcs[1:], arc_labels[1:]):
            self.play(
                Rotate(vector, 2 * theta, about_point=ORIGIN),
                ShowCreation(arc),
                FadeIn(label, shift=0.5 * (arc.get_end() - arc.get_start()))
            )
            self.wait()

        # Show full angle
        quarter_arc = Arc(0, 90 * DEG)
        ninety_label = Tex(R"90^\circ")
        pi_halves_label = Tex(R"\pi / 2")
        for label in [ninety_label, pi_halves_label]:
            label.set_backstroke(BLACK, 5)
            label.next_to(quarter_arc.pfp(0.5), UR, buff=0.05)

        self.play(
            ShowCreation(quarter_arc),
            Write(ninety_label),
        )
        self.wait()
        self.play(FadeTransform(ninety_label, pi_halves_label))
        self.wait()

        # Calculate n steps
        lhs = Text("# Repetitions")
        lhs.next_to(plane, RIGHT, buff=1.0).set_y(2)
        rhs_terms = VGroup(
            Tex(R"\approx {\pi / 2 \over 2 \theta}"),
            Tex(R"= {\pi \over 4} \cdot {1 \over \theta}"),
            Tex(R"= {\pi \over 4} \sqrt{N}"),
        )
        rhs_terms.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        rhs_terms.shift(lhs.get_right() + 0.2 * RIGHT + 0.05 * UP - rhs_terms[0].get_left())

        self.play(
            frame.animate.set_x(3),
            FadeOut(vect_coords),
            FadeOut(bars),
            FadeOut(key_icon2),
            Write(lhs),
            Write(rhs_terms[0]),
        )
        self.wait()
        self.play(LaggedStartMap(FadeIn, rhs_terms[1:], shift=0.5 * DOWN, lag_ratio=0.5))
        self.wait()

        # Collapse
        self.play(LaggedStart(
            Rotate(
                vector,
                -angle_between_vectors(vector.get_vector(), b_vect.get_vector()),
                about_point=ORIGIN
            ),
            FadeOut(arcs),
            FadeOut(arc_labels),
            FadeOut(vect_ghosts),
            FadeOut(b_vect),
            FadeOut(theta_label),
            FadeOut(theta_arc),
            frame.animate.set_y(0.5),
            N_eq.animate.shift(0.5 * UP),
            theta_approx.animate.align_to(plane, RIGHT),
            VGroup(lhs, rhs_terms).animate.shift(1.5 *  UP)
        ))

        # Increment to 2^20
        new_theta = math.asin(2**(-10))
        dim.set_value(100)
        step_count = Tex(R"\frac{\pi}{4}\sqrt{2^{20}} = 804.248...")
        step_count.next_to(rhs_terms, DOWN, LARGE_BUFF)
        step_count.to_edge(RIGHT).shift(frame.get_x() * RIGHT)

        eq_two_twenty = Tex(R"=2^{20}")
        eq_two_twenty.next_to(N_eq["="], DOWN, aligned_edge=LEFT)

        self.play(
            ChangeDecimalToValue(dim, int(2**20)),
            Rotate(vector, new_theta - theta, about_point=ORIGIN),
            FadeIn(eq_two_twenty, time_span=(0, 1)),
            run_time=3
        )
        self.wait()
        self.play(
            TransformFromCopy(rhs_terms[-1][1:6], step_count[:5]),
            TransformFromCopy(eq_two_twenty[1:], step_count["2^{20}"][0]),
            run_time=2
        )
        self.play(Write(step_count["= 804.248..."][0]))
        self.wait()

        # Change vector
        step_tracker = ValueTracker(0)
        radius = 0.5 * plane.get_width()

        step_label = Tex(R"\#\text{Reps} = 0", font_size=36)
        step_label.next_to(plane.c2p(-0.6, 0), UP, SMALL_BUFF)
        step_count = step_label.make_number_changeable(0, edge_to_fix=UL)
        step_count.f_always.set_value(lambda: 0.5 * step_tracker.get_value())

        shadows = VectorizedPoint().replicate(300)

        def update_vector(vector):
            steps = int(step_tracker.get_value())
            if steps % 2 == 0:
                angle = new_theta * (steps + 1)
            else:
                angle = -new_theta * steps
            point = rotate_vector(radius * RIGHT, angle)
            vector.put_start_and_end_on(ORIGIN, point)
            shadows.remove(shadows[0])
            shadows.add(vector.copy())
            shadows.clear_updaters()
            for n, shadow in enumerate(shadows[::-1]):
                shadow.set_fill(opacity=0.75 / (n + 1))

        self.play(FadeIn(step_label))
        self.add(shadows)
        self.add(vect_coords, bars)
        self.play(
            step_tracker.animate.set_value(2 * 804),
            UpdateFromFunc(vector, update_vector),
            frame.animate.scale(1.4, about_edge=RIGHT).set_anim_args(time_span=(0, 8)),
            run_time=20,
            rate_func=linear,
        )
        self.play(FadeOut(shadows, lag_ratio=0.1, run_time=1))
        self.wait()

    def key_flip_insertion(self):
        # Test
        self.clear()
        key_ghost = key_vect.copy().set_fill(opacity=0.5)
        self.add(key_ghost)
        self.play(
            key_vect.animate.flip(axis=RIGHT, about_edge=DOWN),
            run_time=2
        )
        self.wait()

    def thumbnail_insertion(self):
        # Test
        plane.background_lines.set_stroke(BLUE, 8, 1)
        plane.faded_lines.set_stroke(BLUE, 5, 0.25)
        circle.set_stroke(WHITE, 3)
        self.remove(N_eq)
        self.remove(theta_label)
        self.remove(theta_arc)
        self.remove(theta_approx)


class TwoFlipsEqualsRotation(InteractiveScene):
    def construct(self):
        # Set up planes
        plane = NumberPlane((-2, 2), (-2, 2))
        plane.background_lines.set_stroke(BLUE, 1, 1)
        plane.faded_lines.set_stroke(BLUE, 1, 0.25)
        plane.axes.set_opacity(0.5)

        randy = Randolph(mode="pondering", height=3)

        ghost_plane = plane.copy()
        ghost_plane.fade(0.5)
        self.add(ghost_plane)

        plane2 = plane.copy()
        ghost_plane2 = ghost_plane.copy()
        randy2 = randy.copy()
        VGroup(plane2, ghost_plane2, randy2).next_to(plane, RIGHT, buff=3)

        self.add(plane, randy)

        # Show flips
        theta = 15 * DEG
        h_flip_line = DashedLine(plane.get_left(), plane.get_right())
        diag_flip_line = h_flip_line.copy().rotate(theta)
        flip_lines = VGroup(h_flip_line, diag_flip_line)
        flip_lines.set_stroke(WHITE, 4)

        for line in flip_lines:
            self.play(ShowCreation(line, run_time=0.5))
            self.play(
                randy.animate.flip(axis=line.get_vector(), about_point=plane.get_origin()),
                plane.animate.flip(axis=line.get_vector(), about_point=plane.get_origin()),
            )
        self.wait()

        # Show rotation
        arcs = VGroup(
            Arc(0, 90 * DEG, radius=1),
            Arc(180 * DEG, 90 * DEG, radius=1),
        )
        for arc, vect in zip(arcs, [UR, DL]):
            arc.set_stroke(WHITE, 5)
            arc.move_to(plane2.get_corner(vect))
            arc.add_tip()

        self.play(
            self.frame.animate.move_to(midpoint(plane.get_center(), plane2.get_center())),
            FadeIn(ghost_plane2),
            FadeIn(plane2),
            FadeIn(randy2),
        )
        self.play(
            Rotate(Group(plane2, Point(), randy2), 2 * theta, about_point=plane2.get_center()),
            *map(FadeIn, arcs)
        )
        self.wait()

        # Show angle
        arc = Arc(0, theta, radius=1.0)
        theta_label = Tex(R"\theta", font_size=30)
        theta_label.set_backstroke(BLACK, 3)
        theta_label.next_to(arc, RIGHT, SMALL_BUFF)
        theta_label.shift(0.025 * UP)

        rot_arc = Arc(0, 2 * theta, arc_center=plane2.get_center())
        two_theta_label = Tex(R"2 \theta")
        two_theta_label.set_backstroke(BLACK, 3)
        two_theta_label.next_to(rot_arc, RIGHT, SMALL_BUFF)

        self.play(ShowCreation(arc), Write(theta_label), run_time=1)
        self.play(
            TransformFromCopy(arc, rot_arc),
            TransformFromCopy(theta_label, two_theta_label),
        )
        self.wait()


class ComplexComponents(InteractiveScene):
    def construct(self):
        # Set up vectors
        state = normalize(np.random.uniform(-1, 1, 4))
        dec_vect = DecimalMatrix(state.reshape(-1, 1), decimal_config=dict(include_sign=True))
        dec_vect.move_to(3 * LEFT)

        indices = VGroup(BitString(n, 2) for n in range(4))
        for index, elem in zip(indices, dec_vect.elements):
            index.set_color(GREY_B)
            index.match_height(elem)
            index.next_to(dec_vect, LEFT)
            index.match_y(elem)

        x_vect, z_vect = var_vects = [
            TexMatrix(np.array([f"{char}_{n}" for n in range(4)]).reshape(-1, 1))
            for char in "xz"
        ]
        for vect in var_vects:
            vect.match_height(dec_vect)
            vect.move_to(dec_vect, LEFT)

        self.add(dec_vect, indices)
        self.wait()

        # Real number lines and complex planes
        number_lines = VGroup(
            NumberLine(
                (-1, 1, 0.1),
                big_tick_spacing=1,
                width=3,
                tick_size=0.05,
                stroke_color=GREY_C,
            )
            for n in range(4)
        )
        number_lines.arrange(DOWN, buff=1.0)
        number_lines.next_to(dec_vect, RIGHT, buff=1.0)

        complex_planes = VGroup(
            ComplexPlane((-1, 1), (-1, 1)).replace(number_line, 0)
            for number_line in number_lines
        )

        for number_line in number_lines:
            number_line.add_numbers([-1, 0, 1], font_size=24, buff=0.15)
        for plane in complex_planes:
            plane.add_coordinate_labels(font_size=16)

        complex_planes.generate_target()
        complex_planes.target.arrange(DOWN, buff=1.0)
        complex_planes.target.set_height(7)
        complex_planes.target.next_to(x_vect, RIGHT, buff=2.0)
        complex_planes.set_opacity(0)

        R_labels, C_labels = [
            VGroup(
                Tex(Rf"\mathds{{{char}}}", font_size=36).next_to(mob.get_right(), RIGHT)
                for mob in group
            )
            for char, group in zip("RC", [number_lines, complex_planes.target])
        ]

        # Set up dots
        state_tracker = ComplexValueTracker(state)

        dots = GlowDot(color=YELLOW).replicate(len(state))

        def update_dots(dots):
            for dot, value, plane in zip(dots, state_tracker.get_value(), complex_planes):
                dot.move_to(plane.n2p(value))

        dots.add_updater(update_dots)

        dot_lines = VGroup(Line() for dot in dots)
        dot_lines.set_stroke(YELLOW, 2, 0.5)

        def update_dot_lines(lines):
            for line, x, dot in zip(lines, x_vect.elements, dots):
                line.put_start_and_end_on(
                    x.get_right() + 0.05 * RIGHT,
                    dot.get_center()
                )

        update_dot_lines(dot_lines)

        self.play(
            LaggedStartMap(FadeIn, number_lines, lag_ratio=0.25),
            LaggedStartMap(FadeIn, R_labels, lag_ratio=0.25),
            LaggedStart(
                # (FadeInFromPoint(dot, elem.get_center())
                (FadeTransform(elem, dot)
                for elem, dot in zip(dec_vect.elements, dots)),
                lag_ratio=0.05,
                group_type=Group
            ),
            LaggedStart(
                (FadeTransform(elem.copy(), x)
                for elem, x in zip(dec_vect.elements, x_vect.elements)),
                lag_ratio=0.05,
            ),
            LaggedStartMap(ShowCreation, dot_lines),
            ReplacementTransform(dec_vect.get_brackets(), x_vect.get_brackets(), time_span=(1, 2)),
            run_time=2
        )
        self.add(dots, dot_lines)
        dot_lines.add_updater(update_dot_lines)

        self.play(
            state_tracker.animate.set_value(normalize(np.random.uniform(-1, 1, 4))),
            run_time=4
        )

        # Transition to complex plane
        number_lines.generate_target()
        for line, plane in zip(number_lines.target, complex_planes.target):
            line.replace(plane, 0)
            line.set_opacity(0)

        self.play(
            MoveToTarget(complex_planes),
            MoveToTarget(number_lines, remover=True),
            *(FadeTransform(R, C) for R, C in zip(R_labels, C_labels)),
            ReplacementTransform(x_vect, z_vect)
        )
        self.wait()

        for n in range(4):
            self.random_state_change(state_tracker, pump_index=(0 if n == 3 else None))

        # Zoom in on one value
        frame = self.frame

        plane = complex_planes[0]
        c_dot = GlowDot(color=YELLOW, radius=0.05)
        c_dot.move_to(dots[0])

        self.play(
            frame.animate.set_height(1.6).move_to(plane),
            FadeIn(c_dot),
            FadeOut(dots),
            FadeOut(dot_lines),
            run_time=2
        )

        # Show magnitude
        c_line = Line(plane.c2p(0), c_dot.get_center())
        c_line.set_stroke(YELLOW, 2)
        big_brace_width = 3
        brace = Brace(Line(LEFT, RIGHT).set_width(big_brace_width), DOWN)
        brace.scale(c_line.get_length() / big_brace_width, about_point=ORIGIN)
        brace.rotate(c_line.get_angle() + PI, about_point=ORIGIN)
        brace.shift(c_line.get_center())

        mag_label = Tex(R"|z_0|", font_size=12)
        mag_label.next_to(brace.get_center(), DR, buff=0.05),

        self.play(
            ShowCreation(c_line),
            GrowFromCenter(brace),
            FadeIn(mag_label),
        )

        # Show phase
        arc = always_redraw(lambda: Arc(
            0, c_line.get_angle() % TAU, radius=0.1, arc_center=plane.c2p(0),
            stroke_color=MAROON_B
        ))
        arc.update()
        arc.suspend_updating()

        phi_label = Tex(R"\varphi", font_size=12)
        phi_label.set_color(MAROON_B)
        phi_label.add_updater(
            lambda m: m.move_to(arc.pfp(0.5)).shift(0.5 * (arc.pfp(0.5) - plane.c2p(0)))
        )

        self.play(ShowCreation(arc), FadeIn(phi_label))
        self.wait()

        # Show prob
        prob_eq = Tex(R"\text{Prob} = |z_0|^2", font_size=12)
        prob_eq.move_to(mag_label)
        prob_eq.align_to(C_labels[0], RIGHT).shift(0.2 * RIGHT)

        self.play(
            TransformFromCopy(mag_label, prob_eq["|z_0|"][0], path_arc=45 * DEG),
            Write(prob_eq[R"\text{Prob} ="][0]),
            Write(prob_eq["2"][0]),
        )
        self.wait()

        # Change phase
        c_dot.add_updater(lambda m: m.move_to(c_line.get_end()))
        ghost_line = c_line.copy().set_stroke(opacity=0.5)

        self.add(ghost_line)
        arc.resume_updating()
        phi = c_line.get_angle() % TAU
        for angle in [-(1 - 1e-5) * phi, PI, phi - PI]:
            self.play(
                Rotate(c_line, angle, about_point=plane.n2p(0), run_time=6),
            )
            self.wait()

        # Zoom back out
        fader = Group(prob_eq, mag_label, brace, c_line, c_dot, arc, phi_label, ghost_line)
        fader.clear_updaters()
        dot_lines.set_stroke(opacity=0.25)
        self.play(
            frame.animate.to_default_state(),
            FadeOut(fader),
            FadeIn(dots),
            FadeIn(dot_lines),
            run_time=4
        )

        # More random motion
        self.random_state_change(state_tracker)

        dots.suspend_updating()
        self.play(
            *(
                Rotate(
                    dot,
                    random.random() * 2 * TAU,
                    about_point=plane.c2p(0),
                    run_time=10,
                    rate_func=there_and_back,
                )
                for dot, plane in zip(dots, complex_planes)
            )
        )
        dots.resume_updating()

        for n in range(4):
            self.random_state_change(state_tracker)

    def random_state_change(self, state_tracker, pump_index=None, run_time=2):
        rands = np.random.uniform(-1, 1, 4)
        if pump_index is not None:
            rands[pump_index] = 4
        mags = normalize(rands)
        phases = np.exp(TAU * np.random.random(4) * 1j)
        new_state = mags * phases
        self.play(state_tracker.animate.set_value(new_state), run_time=2)
