from manim_imports_ext import *


class GroverPreview(InteractiveScene):
    def construct(self):
        # Setup blocks
        block_spacing = 1.0
        blocks = Rectangle(12, 1).get_grid(12, 1, v_buff=block_spacing)
        blocks.set_fill(GREY_D, 1)
        blocks.set_stroke(BLUE, 1)
        blocks.move_to(2.5 * UP, UP)

        for block, tex, color in zip(blocks, it.cycle(["U_k", "U_s"]), it.cycle([BLUE, GREEN])):
            label = Tex(tex)
            label.set_color(color)
            label.move_to(block)
            block.set_stroke(color, 1)
            block.add(label)

        self.add(blocks)

        # Wires
        d = 100
        wires = VGroup(
            *Line(UP, DOWN).set_height(1).replicate(3),
            Tex(R"\dots"),
            *Line(UP, DOWN).set_height(1).replicate(3),
            Tex(R"\dots"),
            *Line(UP, DOWN).set_height(1).replicate(3),
        )
        wires.arrange(RIGHT)
        wires.arrange_to_fit_width(11)
        wires.next_to(blocks, UP, buff=0)
        true_wires = VGroup(w for w in wires if isinstance(w, Line))
        syms = ["1", "2", "3", "k-1", "k", "k+1", "98", "99", "100"]
        kets = VGroup(Tex(Rf"|{sym}\rangle") for sym in syms)
        for ket, line in zip(kets, true_wires):
            ket.set_height(0.35)
            ket.next_to(line, UP)

        key_index = 4
        kets[key_index].set_color(YELLOW)

        for block in blocks:
            block.wires = wires.copy()
            for wire in block.wires:
                if isinstance(wire, Line):
                    wire.set_height(block_spacing)
            block.wires.next_to(block, DOWN, buff=0)
            block.add(block.wires)

        self.add(wires)
        self.add(kets)

        # Add numbers
        amplitudes = VGroup(
            DecimalNumber(0.1, font_size=24, include_sign=True, num_decimal_places=3).next_to(wire, LEFT, buff=0.1)
            for wire in true_wires
        )
        amplitudes[key_index].set_color(YELLOW)

        diag_axis = np.array([math.sqrt(0.99), 0.1, 0])
        state_point = Point(diag_axis)

        def get_new_amplitudes():
            result = amplitudes.copy()
            value = math.sqrt(state_point.get_x()**2 / (d - 1))
            for dec in result:
                dec.set_value(value)
            result[key_index].set_value(state_point.get_y())
            return result

        self.add(amplitudes, state_point)

        # Repeatedly cycle
        curr_amplitudes = amplitudes
        frame = self.frame
        frame.set_y(1)

        for block, axis in zip(blocks, it.cycle([RIGHT, diag_axis])):
            state_point.flip(axis=axis, about_point=ORIGIN)
            new_amplitudes = get_new_amplitudes()
            new_amplitudes.match_y(block.wires)

            self.play(
                FadeOutToPoint(curr_amplitudes.copy(), block[0].get_center(), lag_ratio=0.0025, time_span=(0, 2)),
                FadeInFromPoint(new_amplitudes, block[0].get_center(), lag_ratio=0.0025, time_span=(1, 3.0)),
                frame.animate.set_y(min(block.wires.get_y() + 1, 1)),
                run_time=3,
            )

            curr_amplitudes = new_amplitudes


class ClassicalSearch(InteractiveScene):
    def construct(self):
        # Test
        in_out = VGroup(ArrowTip(), ArrowTip(angle=PI))
        in_out.arrange(RIGHT, buff=0.5)
        in_out.set_shape(1.75, 0.75)
        box = Square().set_shape(1.5, 1)
        machine = Union(in_out, box)
        machine.set_fill(GREY_E, 1).set_stroke(GREY_B, 2)
        machine.set_z_index(1)
        self.add(machine)

        items = VGroup(Integer(n) for n in range(25))
        items.arrange(DOWN, buff=0.5)
        items.next_to(box, LEFT, LARGE_BUFF)
        items.shift((box.get_y() - items[0].get_y()) * UP)

        self.add(items)

        # Loop through
        key = 12
        last_sym = VMobject()
        for n in range(key + 1):
            item = items[n]
            self.play(
                items.animate.shift((box.get_y() - item.get_y()) * UP),
                FadeOut(last_sym)
            )
            sym = Checkmark().set_color(GREEN) if n == key else Exmark().set_color(RED)
            sym.set_height(0.75)
            sym.next_to(box, RIGHT, LARGE_BUFF)
            self.play(
                FadeOutToPoint(item.copy(), box.get_right(), time_span=(0, 0.7)),
                FadeInFromPoint(sym, box.get_left(), time_span=(0.3, 1)),
            )
            last_sym = sym

        rect = SurroundingRectangle(items[key])
        rect.set_stroke(GREEN, 3)
        self.play(
            items[:n].animate.set_opacity(0.5),
            ShowCreation(rect),
            items[n + 1:].animate.set_opacity(0.5),
        )
