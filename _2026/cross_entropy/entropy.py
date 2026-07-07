from __future__ import annotations

from manim_imports_ext import *
from _2026.cross_entropy.distribution import StackedProbDistribution
from _2026.cross_entropy.next_char import CHAR_ALPHABET
from _2026.cross_entropy.next_char import get_next_char_distribution
from _2026.cross_entropy.next_char import load_huffman_table
from scipy.special import gamma


def bit_string_mobject(
    bit_string: str,
    font: str = "Consolas",
    font_size: int = 48,
    colors: List[Colors] = [GREY_C, GREY_B]
):
    result = Text(bit_string, font=font, font_size=font_size)
    result["0"].set_color(colors[0])
    result["1"].set_color(colors[1])
    return result


def int_to_bit_string(n_bits, value=None, **kwargs):
    if value is None:
        value = np.random.randint(0, 2**n_bits)
    return bit_string_mobject(f"{value:0{n_bits}b}", **kwargs)


class InstructionArrow(SVGMobject):
    def __init__(self, direction=RIGHT, width=0.5, *args, **kwargs):
        super().__init__("arrow.svg", *args, **kwargs)
        self.set_width(width)
        self.rotate(angle_of_vector(direction))
        self.direction = direction


class ArrowGroup(VGroup):
    def __init__(self):
        super().__init__(InstructionArrow(v) for v in [UP, DOWN, LEFT, RIGHT])


class ProbIcon(VGroup):
    def __init__(
        self,
        prob,
        radius=0.5,
        color=GREEN,
        label_font_size=24,
    ):
        super().__init__()
        self.sector_color = color
        self.label_font_size = label_font_size

        border = Circle(radius=radius).set_stroke(WHITE, 1)
        self.border = border
        self.add(border)

        self.sector = self.get_sector(prob)
        self.label = self.get_label(prob)

        self.add(self.border, self.sector, self.label)

    def set_value(self, value):
        self.label.set_value(100 * value)
        self.sector.become(self.get_sector(value))
        return self

    def get_sector(self, prob):
        radius = self.border.get_width() / 2
        sector = Sector(angle=-prob * TAU, start_angle=TAU / 4, radius=radius)
        sector.set_fill(self.sector_color, 1, border_width=1)
        sector.shift(self.border.get_center())
        return sector

    def get_label(self, prob):
        ndp = 0
        if prob < 0.01:
            ndp = 2
        elif (100 * prob) % 1 == 0.5:
            ndp = 1
        label = DecimalNumber(
            100 * prob,
            unit="%",
            font_size=self.label_font_size,
            num_decimal_places=ndp
        )
        label.next_to(self.border, DOWN, SMALL_BUFF)
        return label


class CompressibilitOfText(InteractiveScene):
    def construct(self):
        # Set up
        rects = Rectangle().get_grid(6, 6, buff=0)
        rects.set_shape(FRAME_WIDTH - 1, 6)
        rects.to_edge(RIGHT, buff=0)
        rects.to_edge(DOWN, buff=MED_LARGE_BUFF)
        n_shown = len(rects)
        dots = Tex(R"\vdots")
        dots.to_edge(DOWN, buff=SMALL_BUFF)

        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
        huffman_table = load_huffman_table(alphabet)

        char_mobs = VGroup()
        ascii_mobs = VGroup()
        pre_encodings = VGroup()
        huffman_mobs = VGroup()
        for char, rect in zip(alphabet, rects):
            char_mob = Text(char.lower() + ":")
            char_mob.shift(rect.get_left() - char_mob[1].get_right())
            ascii_mob = bit_string_mobject(f"{ord(char):08b}")
            ascii_mob.set_height(0.2)
            ascii_mob.next_to(char_mob[1], RIGHT, SMALL_BUFF, DOWN)

            pre_encoding = VGroup(
                int_to_bit_string(int(a)).match_height(ascii_mob).move_to(ascii_mob, LEFT)
                for a in np.linspace(1, 8, 30)
            )
            pre_encoding.add(*ascii_mob.copy().replicate(3))

            huffman_mob = bit_string_mobject(huffman_table[char.lower()])
            huffman_mob.match_height(ascii_mob)
            huffman_mob.set_max_width(ascii_mob.get_width())
            huffman_mob.move_to(ascii_mob, LEFT)

            char_mobs.add(char_mob)
            ascii_mobs.add(ascii_mob)
            pre_encodings.add(pre_encoding)
            huffman_mobs.add(huffman_mob)

        # Set up titles and bpc labels
        right_arrow = Tex(R"\rightarrow")
        right_arrow.to_edge(UP, buff=MED_LARGE_BUFF)
        right_arrow.stretch(1.5, 0)

        bpc_label = Tex(R"\sim 8.0 \text{bits} / \text{ch}")
        squiggle = bpc_label[0]
        squiggle.set_opacity(0)
        bpc_dec = bpc_label.make_number_changeable("8.0")
        bpc_dec.shift(0.2 * LEFT)
        bpc_dec.set_color(BLUE)
        bpc_label.next_to(right_arrow, RIGHT, SMALL_BUFF)

        titles = VGroup(
            Text("ASCII"),
            Text("Huffman"),
            Text("Neural methods"),
        )
        for title in titles:
            title.next_to(right_arrow, LEFT)
            title.align_to(bpc_dec, DOWN)

        # Show ascii table
        self.add(char_mobs, right_arrow, dots, bpc_label)
        self.play(
            Write(titles[0]),
            LaggedStartMap(ShowSubmobjectsOneByOne, pre_encodings, run_time=2, lag_ratio=0.01),
        )
        self.remove(pre_encodings)
        self.add(ascii_mobs)
        self.wait()

        # Show Huffman table
        self.remove(ascii_mobs)
        self.play(
            LaggedStart(
                (TransformFromCopy(ascii_mob, huffman_mob)
                for ascii_mob, huffman_mob in zip(ascii_mobs, huffman_mobs)),
                lag_ratio=0.05,
                run_time=2,
            ),
            LaggedStart(
                FadeIn(titles[1], 0.35 * UP),
                FadeOut(titles[0], 0.35 * UP),
                lag_ratio=0.1,
                run_time=1,
            ),
            ChangeDecimalToValue(bpc_dec, 4.2),
            squiggle.animate.set_opacity(1).shift(SMALL_BUFF * LEFT),
            right_arrow.animate.shift(SMALL_BUFF * LEFT)
        )
        self.wait()

        # Collapse to an example phrase
        phrase = Text("""
            It was the best of times, it was the worst of times,
            it was the age of wisdom, it was the age of foolishness...
        """, alignment="LEFT")
        phrase.move_to(rects, UP).shift(DOWN)
        phrase.set_fill(GREY_A)
        arrow = Vector(1.25 * DOWN, fill_color=BLUE, thickness=5)
        arrow.next_to(phrase, DOWN, buff=SMALL_BUFF)
        bits = bit_string_mobject("".join([random.choice("01") for n in range(len(phrase))]))
        for n in range(30, len(phrase), 30):
            bits[n:].next_to(bits[n - 30], DOWN, SMALL_BUFF, LEFT)
        bits.refresh_bounding_box()
        bits.next_to(arrow, DOWN)
        bits.set_x(0)

        colons = VGroup()
        chars = VGroup()
        for char_mob in char_mobs:
            char, colon = char_mob
            colons.add(colon)
            chars.add(char)

            char.target = char.generate_target()
            ch = char_mob.get_text()[0]

            trg = phrase[ch]
            if len(trg) > 0:
                char.target.replace(trg[0])
                char.target.match_color(trg[0])
            else:
                char.target.set_opacity(0).move_to(phrase)

        self.play(
            FadeIn(phrase, lag_ratio=0.01, run_time=2),
            LaggedStart(
                FadeIn(titles[2], 0.35 * UP),
                FadeOut(titles[1], 0.35 * UP),
                lag_ratio=0.1,
                run_time=1,
            ),
            FadeOut(dots),
            FadeOut(huffman_mobs, scale=0.5, lag_ratio=0.01),
            FadeOut(colons, scale=0.5, lag_ratio=0.1),
            LaggedStartMap(MoveToTarget, chars, lag_ratio=0.001, run_time=2),
            ChangeDecimalToValue(bpc_dec, +1.1),
        )
        self.play(
            GrowArrow(arrow),
            LaggedStart(
                (FlashAround(c, buff=0.05, time_width=1.5, stroke_width=2)
                for c in phrase),
                lag_ratio=0.01,
                run_time=3
            ),
            ShowIncreasingSubsets(bits, run_time=3),
        )
        self.wait()


class SpaceOfCodewords(InteractiveScene):
    layer_colors = (BLUE_E, TEAL_E)

    def construct(self):
        # Introduce diagram
        diagram = self.get_core_diagram(n_layers=8)
        ellipsis = Tex(R"\vdots", font_size=72)
        ellipsis.next_to(diagram, UP)

        self.add(diagram[0])
        for l1, l2 in zip(diagram, diagram[1:]):
            layer_copy = l1.copy()
            for box in layer_copy:
                point = VectorizedPoint()
                point.next_to(box[0], RIGHT, SMALL_BUFF)
                box[0].add(point)
            self.play(LaggedStart(
                TransformFromCopy(layer_copy, l2[0::2]),
                TransformFromCopy(layer_copy, l2[1::2]),
                lag_ratio=0.5,
                run_time=1.5
            ))
            self.wait(0.5)
        self.play(Write(ellipsis))
        self.add(diagram)
        self.wait()

        # Just stare at it
        frame = self.frame
        randy, morty = pis = VGroup(Randolph(), Mortimer())
        pis.arrange(RIGHT, buff=2.0)
        pis.next_to(diagram, DOWN, LARGE_BUFF)

        self.play(
            frame.animate.set_height(11, about_edge=UP),
            VFadeIn(pis),
            randy.change("pondering", diagram),
            morty.change("raise_right_hand", diagram),
        )
        self.play(Blink(randy))
        self.wait()
        self.play(randy.change("thinking", diagram))
        self.play(Blink(randy))
        self.wait()

        # Highlight layers
        bit_labels = VGroup(
            TexText(Rf"{n + 1} bit{"s" if n > 0 else ""} $\rightarrow$").next_to(layer, LEFT)
            for n, layer in enumerate(diagram)
        )

        last_bit_label = Mobject()
        for n, layer in enumerate(diagram):
            bit_label = bit_labels[n]
            self.highlight_layer(diagram, n, added_anims=[
                frame.animate.move_to(LEFT).set_height(1.05 * FRAME_HEIGHT),
                FadeIn(bit_label),
                FadeOut(last_bit_label),
                pis.animate.set_opacity(0)
            ])
            last_bit_label = bit_label
        self.wait()
        self.play(
            FadeOut(pis),
            FadeOut(last_bit_label),
            frame.animate.to_default_state(),
            diagram.animate.set_fill(opacity=1),
        )

        # Recursive parts
        to_fade = VGroup()
        for bit_string in ["0", "1", "00", "01", "10", "11", "100"]:
            prefixes = VGroup(
                rect[0][:len(bit_string)]
                for rect in self.get_blocks_with_prefix(diagram, bit_string)
            ).copy()
            prefixes.set_fill(RED, 1)
            prefix_rect = SurroundingRectangle(prefixes[0], buff=SMALL_BUFF)
            prefix_rect.set_stroke(RED, 2)

            self.highlight_bit_string(
                diagram,
                bit_string,
                prefix=True,
                added_anims=[ShowCreation(prefix_rect), FadeOut(to_fade)]
            )
            self.play(FadeIn(prefixes, lag_ratio=0.5 / len(prefixes)))
            to_fade = VGroup(prefixes, prefix_rect)
            self.wait()
            self.add(diagram)
        self.play(diagram.animate.set_fill(opacity=1), FadeOut(to_fade))

        # Allocate four particular strings
        symbols = VGroup(Vector(0.5 * v, thickness=8) for v in [UP, DOWN, LEFT, RIGHT])
        symbols.set_fill(YELLOW)
        prefixes = ["0", "10", "110", "111"]

        key_blocks = VGroup()
        equation_labels = VGroup()
        blocked_groups = VGroup()
        for symbol, prefix in zip(symbols, prefixes):
            all_blocks = self.get_blocks_with_prefix(diagram, prefix)
            block = all_blocks[0]
            blocks_above = all_blocks[1:]
            blocks_below = VGroup(
                self.get_block_by_bit_string(diagram, prefix[:n])
                for n in range(1, len(prefix))
            )

            label = block[0]
            label.target = label.generate_target()
            new_label = VGroup(label.target, Tex(R"="), symbol)
            new_label.arrange(RIGHT, buff=0.2)
            new_label.set_max_width(0.8 * block.get_width())
            new_label.move_to(label)

            blocks_below.target = blocks_below.generate_target()
            for block_below in blocks_below.target:
                block_below.set_fill(opacity=0.25).set_stroke(opacity=0.5)
                block_below[0].set_fill(opacity=0).set_stroke(width=0)

            self.play(
                blocks_above.animate.set_fill(opacity=0.25).set_stroke(opacity=0.5),
                MoveToTarget(blocks_below),
                MoveToTarget(label),
                Write(new_label[1:])
            )
            self.wait()
            key_blocks.add(block)
            block.equation = VGroup(label, *new_label[1:])
            equation_labels.add(block.equation)
            blocked_groups.add(blocks_above)

        self.add(diagram, equation_labels)

        # Explicity show blocking
        braces = VGroup(Brace(block, DOWN, SMALL_BUFF) for block in key_blocks)
        for brace in braces[1:]:
            brace.stretch(0.5, 1, about_edge=UP)
        prop_labels = VGroup(
            VGroup(brace, brace.get_tex(f"1 / {n}", font_size=36))
            for brace, n in zip(braces, [2, 4, 8, 8])
        )

        for block, prop_label, eq_label, blocked_group in zip(key_blocks, prop_labels, equation_labels, blocked_groups):
            underline = Underline(eq_label, buff=0.05)
            underline.set_max_height(0.7, stretch=True)
            underline.set_stroke(YELLOW, (0, 3, 3, 3, 0))
            self.play(
                ShowCreation(underline),
                blocked_group.animate.set_fill(RED, 0.2),
            )
            self.play(
                FadeIn(prop_label, scale=2)
            )
            self.wait()

    def get_core_diagram(self, n_layers=8, layer_height=0.75, width=12):
        diagram = VGroup(
            self.get_layer(n, layer_height, width)
            for n in range(1, n_layers - 1)
        )
        diagram.arrange(UP, buff=0)
        return diagram

    def get_layer(self, n, height=0.75, total_width=12):
        rects = Rectangle(total_width / 2**n, height).replicate(2**n)
        rects.arrange(RIGHT, buff=0)
        rects.set_fill(BLUE, 1)
        rects.set_submobject_colors_by_gradient(*self.layer_colors)
        rects.set_stroke(WHITE, 1)

        labels = VGroup()
        for rect, bits in zip(rects, it.product(*["01"] * n)):
            bit_string = "".join(bits)
            label = Text(bit_string)
            if n > 4:
                label.arrange(DOWN, buff=SMALL_BUFF)
            label.move_to(rect)
            label.set_max_height(0.8 * rect.get_height())
            label.set_max_width(0.8 * rect.get_width())
            rect.add(label)
            rect.bit_string = bit_string

        return rects

    def highlight_layer(self, diagram, layer_num, alt_opacity=0.1, added_anims=[]):
        diagram.target = diagram.generate_target()
        for n, layer in enumerate(diagram.target):
            if n == layer_num:
                opacity = 1
            else:
                opacity = alt_opacity
            layer.set_fill(opacity=opacity)
        self.play(MoveToTarget(diagram), *added_anims)

    def highlight_bit_string(self, diagram, bit_string, prefix=False, alt_opacity=0.1, added_anims=[]):
        diagram.target = diagram.generate_target()
        for layer in diagram.target:
            for rect in layer:
                if rect.bit_string == bit_string:
                    opacity = 1
                elif prefix and rect.bit_string.startswith(bit_string):
                    opacity = 1
                else:
                    opacity = alt_opacity
                rect.set_fill(opacity=opacity)
        self.play(MoveToTarget(diagram), *added_anims)

    def get_blocks_with_prefix(self, diagram, prefix):
        return VGroup(
            rect
            for layer in diagram
            for rect in layer
            if rect.bit_string.startswith(prefix)
        )

    def get_block_by_bit_string(self, diagram, bit_string):
        for layer in diagram:
            for rect in layer:
                if rect.bit_string == bit_string:
                    return rect
        return None


class IntroduceRandomNoise(InteractiveScene):
    def construct(self):
        # Add robot
        robot_path = Path(
            self.file_writer.get_output_file_rootname().parent.parent,
            'Lunar Rover Assets',
            'Stationary_fade_right.png'
        )
        robot = ImageMobject(robot_path)
        robot.set_height(2).to_edge(LEFT, buff=0)
        robot.set_z_index(1)
        self.add(robot)

        # Show bit streams
        symbols = VGroup(
            Vector(0.5 * v, thickness=4, fill_color=PINK).center()
            for v in [UP, DOWN, LEFT, RIGHT]
        )
        n_terms = 100
        indices = np.random.choice(
            np.arange(4),
            n_terms,
            p=[0.5, 0.25, 0.125, 0.125]
        )
        codewords = ["0", "10", "110", "111"]
        segregated_encoding = [codewords[i] for i in indices]
        bit_stream = bit_string_mobject("".join(segregated_encoding))
        bit_stream.next_to(robot, RIGHT).shift(0.15 * DOWN)

        symbol_stream = VGroup(symbols[i].copy() for i in indices)
        symbol_stream.next_to(bit_stream, UP, SMALL_BUFF)
        curr_i = 0
        for sym, codeword in zip(symbol_stream, segregated_encoding):
            new_i = curr_i + len(codeword)
            sym.match_x(bit_stream[curr_i:new_i])
            curr_i = new_i

        streams = VGroup(bit_stream, symbol_stream)
        velocity = 1.0
        streams.add_updater(lambda m, dt: m.shift(velocity * dt * LEFT))

        self.add(streams)
        self.wait(6)

        # Noise word
        noise_word = Text("Random Noise", font_size=72)
        noise_word.next_to(bit_stream, DOWN, MED_LARGE_BUFF)
        noise_word.set_x(0)

        self.play(Write(noise_word))
        self.wait(1.5)

        # Noise box
        frame = self.frame
        box = Square(side_length=5.5)
        box.next_to(noise_word, DOWN, buff=3.0)
        noise_word.save_state()

        full_sentence = Text(
            "Random Noise is incompressible",
            t2s={"incompressible": ITALIC},
            t2c={"incompressible": RED}
        )
        full_sentence.next_to(box, UP)

        def get_box_contents(row_size=32):
            rows = VGroup(int_to_bit_string(row_size) for n in range(row_size // 2))
            rows.arrange(DOWN, buff=0.225)
            result = VGroup(*it.chain(*rows))
            result.set_max_width(0.9 * box.get_width())
            result.move_to(box)
            return result

        def update_contents(bits):
            centers = [bit.get_center().copy() for bit in bits]
            random.shuffle(centers)
            for bit, center in zip(bits, centers):
                bit.move_to(center)

        contents = get_box_contents()

        self.play(
            frame.animate.move_to(box).shift(MED_SMALL_BUFF * UP),
            TransformFromCopy(bit_stream, contents),
            Transform(noise_word, full_sentence["Random Noise"][0]),
            Write(full_sentence["is incompressible"], time_span=(0.5, 1.5)),
            FadeIn(box),
            run_time=1.5
        )
        contents.add_updater(update_contents)
        streams.suspend_updating()
        self.wait(0.5)

        # Try to squish
        box.insert_n_curves(120)
        squish_box = box.copy().center().apply_function(
            lambda p: p * (1 - 0.1 * abs(math.cos(2 * math.atan2(*p[:2]))))
        )
        squish_box.move_to(box)
        squish_box.set_stroke(RED, 3)

        squish_arrows = VGroup(
            Vector(vect, thickness=8, fill_color=RED).next_to(box, -vect, buff=-0.1)
            for vect in [LEFT, RIGHT]
        )

        self.play(
            Transform(box, squish_box),
            FadeIn(squish_arrows, scale=0.8),
            rate_func=bezier([0, 1.5, 1.5, 0.5, 0.5, 0.5, 1.1, 1.1, 1.1, -0.1, -0.1, 0, 0]),
            run_time=3,
        )
        self.wait(5)

        # Back to the stream
        streams.resume_updating()
        streams.next_to(robot, RIGHT, buff=0)

        self.play(
            frame.animate.to_default_state(),
            VFadeOut(contents),
            FadeOut(box),
            Restore(noise_word),
            FadeOut(full_sentence["is incompressible"]),
            run_time=2,
        )
        self.wait(15)
        streams.clear_updaters()

        # Boxes
        boxes = VGroup(
            SurroundingRectangle(bit, buff=0.025)
            for bit in bit_stream
        )
        boxes.stretch(1.2, 1)
        boxes.set_stroke(WHITE, 1)

        min_index = 18

        self.play(
            FadeOut(symbol_stream, lag_ratio=0.1,),
            FadeOut(bit_stream[min_index:], lag_ratio=0.1),
            FadeIn(boxes, lag_ratio=0.1),
            run_time=2
        )

        # 50/50 Question
        box = boxes[min_index]
        question = Text("0 or 1", font="consolas")
        question.next_to(box, UP, buff=0.75)
        choices = [question[c] for c in "01"]
        percentages = VGroup(
            Text(R"50%", font_size=30).set_fill(GREY_B).next_to(choice, UP, buff=0.15)
            for choice in choices
        )
        question.add(percentages)
        choice_rects = VGroup(SurroundingRectangle(choice) for choice in choices)
        choice_rects.set_stroke(YELLOW, 2)
        question.add(choice_rects)

        arrows = VGroup(
            Arrow(choice.get_bottom(), box.get_top(), buff=SMALL_BUFF, fill_color=TEAL)
            for choice in choices
        )
        question.add(arrows)

        self.add(question)

        # Make choices
        bit_tracker = ValueTracker(0)
        bit_tracker.add_updater(lambda m, dt: m.increment_value(PI * dt))
        self.add(bit_tracker)

        def get_bit():
            return int(bit_tracker.get_value() * 2**10) % 2

        def toggle_stroke(pair):
            bit = get_bit()
            pair[bit].set_stroke(opacity=1)
            pair[1 - bit].set_stroke(opacity=0)

        def toggle_fill(pair):
            bit = get_bit()
            pair[bit].set_fill(opacity=1)
            pair[1 - bit].set_fill(opacity=0)

        n_boxes_covered = 20

        for box in boxes[min_index:min_index + n_boxes_covered]:
            inner_bit = bit_string_mobject("01")
            for part in inner_bit:
                part.match_x(bit_stream[0])
                part.move_to(box)

            choice_rects.set_stroke(opacity=0)
            self.play(
                question.animate.match_x(box),
                run_time=0.2
            )
            self.play(
                UpdateFromFunc(choice_rects, toggle_stroke),
                UpdateFromFunc(inner_bit, toggle_fill),
                run_time=0.5
            )
            self.add(inner_bit)


class RandomNoiseIsIncompressible(SpaceOfCodewords):
    def construct(self):
        # Receive n bits
        frame = self.frame
        frame.reorient(0, 0, 0, (-2.75, 0, 0.0), 5)
        robot_path = Path(
            self.file_writer.get_output_file_rootname().parent.parent,
            'Lunar Rover Assets',
            'Stationary.png'
        )
        robot = ImageMobject(robot_path)
        robot.set_height(2)
        robot.to_edge(LEFT, buff=SMALL_BUFF)

        n_bits = 16
        bit_string = self.get_bit_string(n_bits)
        bit_string.next_to(robot, RIGHT)

        self.add(robot)
        self.play(
            FadeIn(bit_string, shift=3 * LEFT, lag_ratio=0.1, run_time=3)
        )
        self.wait()

        # Instruction-by-instruction
        syms = ArrowGroup()
        colors = color_gradient([YELLOW, BLUE], 4, interp_by_hsl=True)
        cw_lengths = [1, 1, 1, 1, 2, 2, 2, 2, 3, 1]
        cw_partial_sums = np.cumsum(cw_lengths)
        buff = 0.5 * get_norm(bit_string[0].get_right() - bit_string[1].get_left())
        rects = VGroup(
            SurroundingRectangle(bit_string[n1:n2], buff=buff)
            for n1, n2 in zip([0, *cw_partial_sums], cw_partial_sums)
        )
        intruction_indices = [0, 0, 0, 0, 1, 1, 1, 1, 2, 0]
        instructions = VGroup()
        for idx, rect in zip(intruction_indices, rects):
            rect.set_stroke(colors[idx], 2)
            arrow = syms[idx].copy()
            arrow.set_fill(colors[idx])
            arrow.match_x(rect)
            arrow.set_y(rects.get_y() + 0.5)
            instructions.add(arrow)

        self.play(
            ShowIncreasingSubsets(instructions),
            ShowIncreasingSubsets(rects),
            rate_func=linear,
            run_time=1
        )
        self.wait()

        # Full message
        brace = Brace(bit_string, UP, SMALL_BUFF)
        message_word = brace.get_text("message")
        instructions.target = instructions.generate_target()
        instructions.target.arrange(RIGHT, buff=SMALL_BUFF).scale(0.5)
        instructions.target.next_to(message_word, UP)

        full_rect = SurroundingRectangle(bit_string).set_stroke(YELLOW, 2)

        self.play(LaggedStart(
            ReplacementTransform(rects, VGroup(full_rect), path_arc=10 * DEG, lag_ratio=0.1),
            MoveToTarget(instructions),
            GrowFromCenter(brace),
            FadeIn(message_word, 0.5 * UP),
        ))
        self.wait()

        # n bits
        mid_message = Tex(R"\rightarrow m_i")
        mid_message.next_to(bit_string[-1:], RIGHT, MED_SMALL_BUFF, index_of_submobject_to_align=0)
        mid_message[-1].set_opacity(0)

        low_brace = Brace(bit_string, DOWN, SMALL_BUFF)
        n_bit_label = low_brace.get_tex(R"n \text{ bits}")

        self.play(LaggedStart(
            frame.animate.set_height(6, about_edge=LEFT),
            FadeTransform(message_word[0].copy(), mid_message[1:]),
            Write(mid_message[0]),
            FadeOut(message_word),
            instructions.animate.set_width(1).next_to(mid_message[1], DOWN, SMALL_BUFF),
            ReplacementTransform(brace, low_brace, path_arc=-20 * DEG),
            FadeIn(n_bit_label, shift=DOWN),
            run_time=2
        ))
        self.wait()

        # Show all 2^n possible strings
        top_bit_strings, low_bit_strings = [
            VGroup(
                self.get_bit_string(n_bits, n)
                for n in n_range
            ).arrange(DOWN)
            for n_range in [range(4), range(2**n_bits - 4, 2**n_bits)]
        ]
        ellipses = VGroup(
            Tex(R"\vdots").next_to(bit_string, vect)
            for vect in [UP, DOWN]
        )
        top_bit_strings.next_to(ellipses, UP)
        low_bit_strings.next_to(ellipses, DOWN)

        top_messages = VGroup(
            Tex(fR"\rightarrow m_{n}").next_to(bs[-1:], RIGHT, MED_SMALL_BUFF, index_of_submobject_to_align=0)
            for n, bs in enumerate(top_bit_strings)
        )
        low_messages = VGroup(
            Tex(fR"\rightarrow m_{{2^n - {4 - n}}}").next_to(bs[-1:], RIGHT, MED_SMALL_BUFF, index_of_submobject_to_align=0)
            for n, bs in enumerate(low_bit_strings)
        )
        for message in [*top_messages, mid_message, *low_messages]:
            message[2:].scale(0.5, about_edge=LEFT)

        all_bit_strings = VGroup(*top_bit_strings, bit_string, *low_bit_strings)
        all_messages = VGroup(*top_messages, mid_message, *low_messages)

        brace = Brace(VGroup(top_messages, low_messages), RIGHT)
        count = brace.get_tex("2^n", font_size=72, buff=0.2)

        kw = dict(lag_ratio=0.5, run_time=4)
        self.play(
            frame.animate.to_default_state().shift(0.25 * DOWN),
            VGroup(low_brace, n_bit_label).animate.next_to(low_bit_strings, DOWN, SMALL_BUFF),
            GrowFromCenter(brace),
            LaggedStartMap(FadeIn, VGroup(*top_bit_strings, *low_bit_strings), **kw),
            LaggedStartMap(FadeIn, VGroup(*top_messages, *low_messages), **kw),
            Write(ellipses),
            Write(count),
            FadeOut(full_rect),
            mid_message.animate.set_fill(opacity=1),
        )
        self.wait()

        # Flash through
        for n in range(10):
            yellow_bit_strings = all_bit_strings.copy().set_color(YELLOW)
            yellow_bit_strings.shuffle()
            self.play(ShowSubmobjectsOneByOne(yellow_bit_strings, rate_func=linear))
            self.remove(yellow_bit_strings)

        # Reference equally likely
        prob_eq = Tex(R"P(m_i) = {1 \over 2^n}")
        prob_eq.move_to(bit_string).to_edge(RIGHT)
        arrows = VGroup(
            Arrow(
                prob_eq.get_left() + 0.25 * f * UP,
                mess.get_right(),
                path_arc=f * 20 * DEG,
                thickness=3,
                fill_color=TEAL,
                buff=0.2
            )
            for mess, f in zip(all_messages, np.linspace(1, -1, len(all_messages)))
        )
        self.play(
            FadeOut(brace),
            FadeTransform(count, prob_eq["2^n"]),
            Write(prob_eq[:-2]),
            LaggedStartMap(GrowArrow, arrows)
        )
        self.add(prob_eq)

        # Fade robot
        self.play(instructions.animate.set_width(2).next_to(robot, DOWN, buff=-0.2))
        self.wait()
        self.play(Group(robot, instructions).animate.fade(0.5).scale(0.75, about_edge=LEFT))
        self.wait()

        # Alternate data types
        prob_pointer = VGroup(arrows, prob_eq)
        text_rhss = VGroup(
            Text(f"= “{text}”", font_size=30).set_fill(random_bright_color(
                hue_range=(0.53, 0.53),
                saturation_range=(0.8, 0.8),
                luminance_range=(0.7, 0.4),
            ))
            for text in self.get_example_strings()
        )
        image_rhss = Group(
            Group(Tex("="), ImageMobject(filename, height=3)).arrange(RIGHT, SMALL_BUFF)
            for filename in self.get_example_image_names()
        )
        for pair in zip(image_rhss, text_rhss):
            for mob in pair:
                mob.next_to(mid_message, RIGHT).align_to(mid_message[1])
        alt_strings = VGroup(
            self.get_bit_string(n_bits).replace(bit_string)
            for n in range(len(text_rhss))
        )
        alt_strings[0].become(bit_string.copy())

        index_tracker = ValueTracker(0)

        def get_submob(group):
            n = len(group)
            index, _ = integer_interpolate(0, n, index_tracker.get_value())
            return group[index]

        shown_text = VGroup(text_rhss[0])
        shown_image = Group(image_rhss[-1])

        self.play(
            FadeOut(prob_pointer, RIGHT),
            FadeIn(shown_text, RIGHT, suspend_mobject_updating=True),
        )
        self.play(
            index_tracker.animate.set_value(1),
            UpdateFromFunc(bit_string, lambda m: m.set_submobjects(list(get_submob(alt_strings)))),
            UpdateFromFunc(shown_text, lambda m: m.set_submobjects([get_submob(text_rhss)])),
            run_time=2,
            rate_func=linear
        )
        self.wait()
        self.play(
            FadeOut(shown_text, RIGHT),
            FadeIn(shown_image, RIGHT),
        )
        self.play(
            index_tracker.animate.set_value(0),
            UpdateFromFunc(bit_string, lambda m: m.set_submobjects(list(get_submob(alt_strings)))),
            UpdateFromFunc(shown_image, lambda m: m.set_submobjects([get_submob(image_rhss)])),
            run_time=2,
            rate_func=linear
        )
        self.wait()
        self.play(
            Transform(bit_string, alt_strings[0]),
            FadeOut(shown_image, RIGHT),
            FadeIn(text_rhss[7], RIGHT),
        )
        self.wait()
        self.play(
            Transform(bit_string, alt_strings[1]),
            FadeOut(text_rhss[7], RIGHT),
            FadeIn(text_rhss[6], RIGHT),
        )
        self.wait()
        self.play(
            FadeOut(text_rhss[6]),
            FadeIn(prob_pointer, lag_ratio=0.1, run_time=2),
        )
        self.wait()

        # Emphasize random noise
        rect = SurroundingRectangle(bit_string)
        rects = VGroup(SurroundingRectangle(mob) for mob in all_bit_strings)
        rect.set_stroke(YELLOW, 2)
        rects.set_stroke(YELLOW, 2)

        self.play(ShowCreation(rect))
        self.play(
            UpdateFromFunc(bit_string, lambda m: m.become(self.get_bit_string(n_bits).replace(bit_string)))
        )
        self.wait()
        self.play(TransformFromCopy(VGroup(rect), rects))
        self.wait()
        self.play(
            FadeOut(rects),
            rect.animate.surround(prob_eq[R"{1 \over 2^n}"])
        )
        self.wait()
        self.play(FadeOut(rect))

        # Put into the diagram
        frame = self.frame

        to_fade = Group(
            robot, instructions, all_bit_strings, ellipses, all_messages, prob_pointer,
            low_brace, n_bit_label
        )
        diagram = self.get_core_diagram(n_layers=7)
        diagram.next_to(to_fade, RIGHT, buff=2.0)
        top_layer = diagram[-1]
        diagram.remove(top_layer)

        m_terms = VGroup(Tex(Rf"m_{{{n}}}", font_size=36) for n in range(16))
        for m_term, message in zip(m_terms, it.cycle([*top_messages, *low_messages])):
            m_term.move_to(message[1], LEFT)
            m_term[1:].scale(0.5, about_edge=LEFT)

        m_terms.target = m_terms.generate_target()
        for term, box in zip(m_terms.target, diagram[3]):
            term.next_to(box, UP, SMALL_BUFF)
            term.set_fill(box.get_fill_color())

        self.play(
            frame.animate.set_height(2 * FRAME_HEIGHT, about_edge=LEFT),
            FadeIn(diagram),
            FadeIn(top_layer),
            run_time=2,
        )
        self.wait()
        self.highlight_layer(diagram, 3, alt_opacity=0.2, added_anims=[
            frame.animate.set_height(FRAME_HEIGHT).move_to(diagram).set_anim_args(run_time=3),
            MoveToTarget(m_terms, run_time=2, lag_ratio=0.01),
            FadeOut(to_fade),
            FadeOut(top_layer)
        ])
        self.wait()

        # Circle all the bits
        rects = VGroup(SurroundingRectangle(block[0], buff=0.05) for block in diagram[3])
        rects.set_stroke(YELLOW, 4)
        rects.insert_n_curves(100)

        self.play(LaggedStartMap(VShowPassingFlash, rects, time_width=2, lag_ratio=0.08, run_time=5))

        # Push and pop
        m_terms.save_state()
        diagram.save_state()
        residue7 = self.push_down_message(m_terms, diagram, top_layer, index=7)
        self.wait()
        residue13 = self.push_down_message(m_terms, diagram, top_layer, index=13)
        self.wait()
        self.play(
            Restore(m_terms),
            Restore(diagram),
            FadeOut(VGroup(residue7, residue13)),
        )
        self.wait()

    def get_bit_string(self, n_bits, value=None, **kwargs):
        return int_to_bit_string(n_bits, value, **kwargs)

    def push_down_message(self, m_terms, diagram, top_layer, index):
        # Move one message down
        moved_m_term = m_terms[index]
        rect = SurroundingRectangle(moved_m_term).set_stroke(YELLOW, 2)
        old_box = diagram[3][index]
        new_box = diagram[2][index // 2]
        brace = Brace(new_box, DOWN, SMALL_BUFF)
        brace.stretch(0.5, 1, about_edge=UP)

        self.play(ShowCreation(rect))
        self.play(
            VGroup(moved_m_term, rect).animate.next_to(brace, DOWN, SMALL_BUFF),
            GrowFromCenter(brace),
            new_box.animate.set_fill(opacity=1),
            diagram[1][index // 4][0].animate.set_opacity(0),
            old_box.animate.set_fill(opacity=0.2),
            rate_func=rush_into,
        )

        # Show overlapping
        blocked_box = diagram[3][index - 1]
        blocked_box.target = blocked_box.generate_target()
        blocked_box.target.set_fill(RED)
        blocked_box.target[0].set_fill(WHITE)
        blocked_m_term = m_terms[index - 1]
        bang = Tex(R"!", font_size=30)
        bang.set_fill(RED)
        bang.next_to(blocked_m_term, RIGHT, buff=0.05, aligned_edge=DOWN)

        self.play(
            MoveToTarget(blocked_box),
            blocked_m_term.animate.set_color(RED),
            FadeIn(bang, shift=0.1 * UP, scale=0.5),
            rate_func=rush_from,
        )
        self.wait()

        # Wander
        wanderer = VGroup(m_terms[index - 1], bang)

        self.play(
            wanderer.animate.next_to(m_terms[0], UP).set_anim_args(path_arc=10 * DEG),
            blocked_box.animate.set_fill(opacity=0.1),
            run_time=2,
        )
        self.play(wanderer.animate.next_to(m_terms[-1], UP), run_time=2)
        self.play(wanderer.animate.next_to(m_terms[index - 2], UP), run_time=2)

        # Pop up
        top_layer_index = 2 * (index - 1)
        top_boxes = top_layer[top_layer_index - 2:top_layer_index]
        moved_m_terms = m_terms[index - 2:index]
        moved_m_terms.target = moved_m_terms.generate_target()
        for term, box in zip(moved_m_terms.target, top_boxes):
            term.next_to(box, UP, buff=0.15)
            term.set_fill(box.get_fill_color())
        moved_m_terms.target.space_out_submobjects(1.5)

        self.play(
            FadeIn(top_boxes),
            MoveToTarget(moved_m_terms),
            FadeOut(bang),
            diagram[3][index - 2].animate.set_fill(opacity=0.1),
            rate_func=rush_into,
        )
        self.wait()

        return VGroup(rect, brace, top_boxes)

    def get_example_strings(self):
        return [
            "Text to compress...",
            "Claude Shannon",
            "3Blue1Brown",
            "Randy the pi creature",
            "(hey, psst, subscribe)",
            "happy birthday to you",
            "Q7m4!",
            "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11",
            "2, 3, 5, 7, 11,",
            "Random examples of text",
            "are very hard to think of",
            "but I refuse to use an LLM",
            "when there‘s joy in",
            "easter eggs for the",
            "attentive viewer‘ like you",
            "It was the best of times...",
            "Wouldn't it be cool if",
            "these texts could really be",
            "compressed to just 16 bits",
            "blah, blah, blah, yada, yada",
        ]

    def get_example_image_names(self):
        return [
            "Andrey_Kolmogorov",
            "Richard_Hamming",
            "ShrewMole",
            "ss_central_america",
            "Fork",
            "Turing",
            "Cow",
            "Puppy",
            "Principia_equal_area",
            "US_color_graph",
            "TerryTaoPaulErdos",
            "ZoeyInGrass",
            "sars_icon",
            "PixarCampus",
            "IMG-9874",
            "PiCreatureDalle3_13",
            "Tom_In_Bowtie",
            "EiffelTower1",
            "Dalle3_blue_1",
            "Claude_Shannon",
        ]

    def old_material(self):
        # Show 3-bit examples for robot
        bit_string_diagram = VGroup(all_bit_strings, ellipses, all_messages, arrows, prob_eq)
        bit_string_diagram.save_state()

        three_bit_messages = VGroup(
            self.get_bit_string(3, n)
            for n in range(8)
        )
        three_bit_messages.arrange(DOWN)
        three_bit_messages.next_to(robot, RIGHT)

        syms = [Vector(0.5 * v) for v in [UP, DOWN, LEFT, RIGHT]]
        examples = VGroup(three_bit_messages[i] for i in [0, 7, 4])

        self.play(
            bit_string_diagram.animate.set_height(1.5).to_corner(UR),
            *(
                TransformFromCopy(bs1[-3:], bs2)
                for bs1, bs2 in zip(top_bit_strings, three_bit_messages[:4])
            ),
            *(
                TransformFromCopy(bs1[-3:], bs2)
                for bs1, bs2 in zip(low_bit_strings, three_bit_messages[4:])
            ),
        )
        self.wait()


class InformationGraph(InteractiveScene):
    prob_color = TEAL_D
    info_color = BLUE
    input_var = "x"

    def construct(self):
        # Add split diagrams
        frame = self.frame
        diagrams = VGroup(
            Square().get_grid(*dims, buff=0).set_shape(2, 2).set_stroke(WHITE, 2)
            for dims in [(1, 2), (2, 2), (2, 4), (4, 4)]
        )
        for diagram in diagrams:
            diagram[0].set_fill(self.prob_color, 1)

        diagrams.arrange(RIGHT, buff=LARGE_BUFF)

        stacks = VGroup()
        for n, diagram in enumerate(diagrams, start=1):
            power_eq = Tex(
                Rf"\left({{1 \over 2}} \right)^{n} = {{1 \over {2**n}}}",
                t2c={Rf"{{1 \over {2**n}}}": self.prob_color},
                font_size=30
            )
            log_eq = Tex(
                Rf"-\log_2(1 / {2**n}) = {n}",
                t2c={Rf"1 / {2**n}": self.prob_color},
                font_size=30
            )
            equiv = Tex(R"\Leftrightarrow").rotate(90 * DEG)
            stack = VGroup(power_eq, equiv, log_eq)
            stack.arrange(DOWN)
            stack.next_to(diagram, UP)
            stacks.add(stack)

        self.add(diagrams[0])
        self.add(stacks[0])
        frame.match_x(stacks[0])
        self.wait()
        for n in range(1, len(stacks)):
            self.play(
                frame.animate.match_x(stacks[:n + 1]),
                FadeIn(diagrams[n]),
                FadeIn(stacks[n]),
            )
            self.wait()

        # Prepare axes
        max_y = 6
        axes = Axes(
            (0, 1, 1 / 16),
            (0, max_y),
            width=8,
            height=5
        )
        axes.x_axis.add_numbers([0, 1], font_size=36, num_decimal_places=1)
        axes.x_axis.numbers[0].next_to(axes.get_origin(), DL, SMALL_BUFF)
        bit_lines = VGroup(
            Line(axes.c2p(0, y), axes.c2p(1, y)).set_stroke(GREY_B, 1)
            for y in range(1, max_y + 1)
        )
        bit_labels = VGroup(
            Text(f"{n} bits", font_size=24).set_fill(GREY_C).next_to(line, LEFT)
            for n, line in zip(it.count(1), bit_lines)
        )
        axes.add(bit_lines, bit_labels)

        p_label = Tex(self.input_var).set_color(self.prob_color)
        p_label.next_to(axes.x_axis.get_end(), RIGHT)
        axes.add(p_label)

        info_label = Tex(
            Rf"-\log_2({self.input_var})",
            t2c={R"-\log_2": self.info_color, self.input_var: self.prob_color}
        )
        info_label.next_to(axes.c2p(0, max_y), UR)

        # Add bars to graph
        bars = VGroup(
            Rectangle(
                width=0.25,
                height=axes.y_axis.get_unit_size() * n
            ).move_to(axes.c2p(0.5**n, 0), DOWN)
            for n in range(1, 5)
        )
        bars.set_stroke(WHITE, 1).set_fill(self.info_color, 1)

        bar_labels = VGroup(
            Tex(Rf"1 \over {2**n}", font_size=30).next_to(bar, DOWN).set_color(self.prob_color)
            for n, bar in enumerate(bars, start=1)
        )

        self.play(
            LaggedStart(*(
                TransformFromCopy(stack[2][f"1 / {2**n}"][0], label, run_time=2)
                for stack, label, n in zip(stacks, bar_labels, it.count(1))
            )),
            LaggedStart(*(
                TransformFromCopy(diagram[0], bar)
                for diagram, bar in zip(diagrams, bars)
            )),
            Write(info_label),
            FadeOut(diagrams),
            FadeOut(stacks),
            FadeIn(axes, time_span=(1, 3)),
            FadeIn(bars, time_span=(1, 3)),
        )
        self.wait()

        # Add graph indicator
        graph = axes.get_graph(
            lambda p: -np.log2(p),
            x_range=(1e-5, 1, 1e-2)
        )
        graph.set_stroke(self.info_color, 3)

        p_tracker = ValueTracker(0.5)
        get_p = p_tracker.get_value

        graph_dot = Group(TrueDot().make_3d(), GlowDot())
        graph_dot.set_color(TEAL)
        graph_dot.add_updater(
            lambda m: m.move_to(axes.i2gp(get_p(), graph))
        )

        p_tip = ArrowTip(90 * DEG).set_height(0.15)
        p_tip.set_color(self.prob_color)
        p_tip.add_updater(lambda m: m.move_to(axes.x_axis.n2p(get_p()), UP))

        bit_tip = ArrowTip().set_width(0.15)
        bit_tip.set_color(self.info_color)
        bit_tip.add_updater(lambda m: m.move_to(axes.y_axis.n2p(-math.log2(get_p())), RIGHT))

        p_dec = DecimalNumber(font_size=36, fill_color=self.prob_color, num_decimal_places=2)
        p_dec.add_updater(lambda m: m.set_value(get_p()).next_to(p_tip, DOWN, SMALL_BUFF))

        equals = Tex(R"=")
        equals.always.next_to(info_label, RIGHT, SMALL_BUFF)
        log_p_dec = DecimalNumber(font_size=48, fill_color=self.info_color)
        log_p_dec.add_updater(lambda m: m.set_value(-math.log2(get_p())).next_to(equals, RIGHT, SMALL_BUFF))

        v_line = always_redraw(
            lambda: DashedLine(
                p_tip.get_top(),
                graph_dot.get_center(),
                dash_length=0.05,
                stroke_width=1,
                stroke_color=WHITE
            )
        )
        h_line = always_redraw(
            lambda: DashedLine(
                bit_tip.get_right(),
                graph_dot.get_center(),
                dash_length=0.05,
                stroke_width=1,
                stroke_color=WHITE
            )
        )

        self.play(ShowCreation(graph))
        self.wait()
        self.play(
            FadeIn(p_tip),
            FadeIn(bit_tip),
            FadeIn(graph_dot),
            FadeIn(v_line),
            FadeIn(h_line),
            FadeIn(p_dec),
            FadeIn(equals),
            FadeIn(log_p_dec),
            FadeOut(bars),
            FadeOut(bar_labels),
        )
        self.wait()

        # Play with values
        self.play_with_prob(p_tracker, [0.2, 0.9, 0.1])

        # Show as information definition
        # lhs = Tex(R"\text{Information} :=")
        lhs = Tex(R"\text{Loss} =")
        lhs["Loss"].set_color(RED)
        lhs.move_to(info_label, UL)
        lhs.shift((equals.get_y() - lhs["="].get_y()) * UP)

        self.play(FlashAround(info_label, run_time=2))
        self.play(
            Write(lhs),
            info_label.animate.next_to(lhs, RIGHT, SMALL_BUFF, UP)
        )
        self.wait()

        # Play some more
        self.play_with_prob(p_tracker, [0.5, 0.9, 0.02, 0.5], run_time_each=4)

        # Add prob and entropy indicators
        frame = self.frame

        info_bar = Rectangle(0.5, 3)
        info_bar.set_fill(self.info_color, 1)
        info_bar.set_stroke(WHITE, 1)
        info_bar.next_to(axes.c2p(1, 0), RIGHT, buff=1.5, aligned_edge=DOWN)
        info_bar.add_updater(lambda m: m.match_height(v_line, stretch=True).align_to(v_line, DOWN))

        brace = always_redraw(lambda: Brace(info_bar, RIGHT, SMALL_BUFF))
        log_label = Tex(Rf"-\log_2({self.input_var})")
        log_label.always.next_to(brace, RIGHT, SMALL_BUFF)

        prob_icon = ProbIcon(p_tracker.get_value(), radius=0.5)
        prob_icon.next_to(info_bar, DOWN)
        prob_icon.add_updater(lambda m: m.set_value(get_p()))

        self.play(
            frame.animate.reorient(0, 0, 0, (2.56, -0.52, 0.0), 9.20),
            FadeIn(prob_icon),
            VFadeIn(info_bar),
            VFadeIn(brace, suspend_mobject_updating=True),
            VFadeIn(log_label),
        )
        self.wait()

        # Play with values more
        self.play_with_prob(p_tracker, [0.25, 0.1, 0.01])
        self.wait()
        self.play(p_tracker.animate.set_value(0.9), run_time=5)
        self.wait()

        # Compare to other graphs
        log_graph = graph
        top_to_fade = VGroup(lhs, equals, log_p_dec)
        value_indicators = Group(p_tip, bit_tip, graph_dot, v_line, h_line, p_dec)
        side_to_fade = VGroup(info_bar, brace, log_label, prob_icon)
        side_to_fade.clear_updaters()

        x_range = (1e-2, 1, 1e-2)
        alt_graphs = VGroup(
            axes.get_graph(lambda x: (1.0 / (x)) - 1, x_range=x_range),
            axes.get_graph(lambda x: np.tan((PI / 2) * (1 - x)), x_range=x_range),
            axes.get_graph(lambda x: gamma(x) - 1, x_range=x_range),
        )
        alt_graph_labels = VGroup(
            Tex(R"{1 \over x} - 1"),
            Tex(R"\tan\left({\pi \over 2}(1 - x)\right)"),
            Tex(R"\Gamma(x) - 1"),
        )
        colors = [YELLOW, GREEN, RED]
        xs = [0.15, 0.2, 0.3]
        for label, alt_graph, color, x in zip(alt_graph_labels, alt_graphs, colors, xs):
            alt_graph.set_color(color)
            label.set_color(color)
            label.next_to(axes.i2gp(x, alt_graph), UR, buff=0.05)
            label.set_z_index(1)

        self.play(
            frame.animate.to_default_state(),
            FadeOut(top_to_fade),
            FadeOut(value_indicators),
            FadeOut(side_to_fade),
            info_label.animate.set_x(axes.y_axis.get_x(), RIGHT).shift(SMALL_BUFF * UP),
        )

        curr_graphs = VGroup(log_graph)
        curr_labels = VGroup(info_label)
        for label, alt_graph in zip(alt_graph_labels, alt_graphs):
            self.play(
                curr_graphs.animate.set_stroke(opacity=0.2),
                curr_labels.animate.set_fill(opacity=0.2),
                ShowCreation(alt_graph),
                FadeIn(label, 0.5 * UP)
            )
            self.wait()
            curr_graphs.add(alt_graph)
            curr_labels.add(label)
        self.wait()
        self.play(
            log_graph.animate.set_stroke(opacity=1),
            info_label.animate.set_fill(opacity=1),
            curr_graphs[1:].animate.set_stroke(opacity=0.2),
            curr_labels[1:].animate.set_fill(opacity=0.2),
        )
        self.wait()

        # More ambient motion to discuss generality
        self.remove(alt_graphs, alt_graph_labels, info_label)
        self.add(p_tracker, *value_indicators)
        p_tracker.set_value(0.5)

        for value in [0.02, 0.9, 0.5]:
            self.play(p_tracker.animate.set_value(value), run_time=6)
        self.wait()

    def play_with_prob(self, p_tracker, values, run_time_each=3):
        for value in values:
            self.play(p_tracker.animate.set_value(value), run_time=run_time_each)


class InformationOfLanguage(InteractiveScene):
    random_seed = 3
    # example_text = "information theory quiz"
    example_text = "compression is intelligence"
    final_shown_sample_index = -1

    def construct(self):
        # Set up ticker tape
        frame = self.frame
        arrow_dist = [0.5, 0.25, 0.125, 0.125]
        text = self.example_text
        n_syms = len(text)
        arrow_choice_indexes = random.choices(
            list(range(4)),
            weights=arrow_dist,
            k=n_syms
        )

        top_tape = Square().get_grid(1, n_syms, buff=0)
        top_tape.set_width(FRAME_WIDTH - 1)
        top_tape.to_edge(UP)
        top_tape.set_stroke(WHITE, 2)

        low_tape = top_tape.copy()
        low_tape.set_y(-1)

        top_probs = np.array([arrow_dist[i] for i in arrow_choice_indexes])
        top_prob_icons = self.get_prob_icons(top_tape, top_probs)

        low_probs = np.array([
            get_next_char_distribution(" " if n == 0 else text[:n])[CHAR_ALPHABET.index(char)]
            for n, char in enumerate(text.lower())
        ])
        low_probs[-1] = 0.3  # This just seems wrong to me

        low_prob_icons = self.get_prob_icons(low_tape, low_probs)

        # Show stream of symbols
        arrow_templates = VGroup(svg[0] for svg in ArrowGroup())
        arrow_templates.set_fill(opacity=1)
        arrow_templates.set_height(0.7 * top_tape[0].get_height())
        arrow_templates.set_submobject_colors_by_gradient(YELLOW_B, YELLOW_C, YELLOW_D)

        arrows = VGroup(
            arrow_templates[idx].copy().move_to(square)
            for square, idx in zip(top_tape, arrow_choice_indexes)
        )

        frame.set_height(2.0).move_to(top_tape[0], LEFT).shift(0.5 * DOWN)
        self.play(
            frame.animate.set_height(FRAME_HEIGHT).move_to(top_tape),
            ShowIncreasingSubsets(top_tape, rate_func=linear),
            ShowIncreasingSubsets(arrows, rate_func=linear),
            ShowIncreasingSubsets(top_prob_icons, rate_func=linear),
            run_time=5
        )
        self.wait()

        # Show scanning distribution
        next_arrow, arrow_distribution = self.get_prediction(top_tape[0], top_tape[1], arrow_dist, arrow_templates, dist_mob_width=5)
        dist_bar_labels = VGroup(
            Tex(Rf"1/{denom}", font_size=24).next_to(bar, UP, SMALL_BUFF)
            for bar, denom in zip(arrow_distribution.bars, [2, 4, 8, 8])
        )
        arrow_distribution.add(dist_bar_labels)
        arrow_distribution.center()
        arrow_distribution.set_width(8.0)
        arrow_distribution.next_to(frame.get_top(), DOWN, MED_LARGE_BUFF)

        top_groups = VGroup(
            VGroup(*mobs)
            for mobs in zip(top_tape, arrows, top_prob_icons)
        )
        top_groups.save_state()

        self.play(LaggedStart(
            FadeIn(arrow_distribution),
            lag_ratio=0.5
        ))
        self.wait(0.5)

        self.add(next_arrow)
        self.remove(arrows[1:], top_prob_icons[1:])
        shift_value = top_tape[1].get_center() - top_tape[0].get_center()
        next_arrow.shift(-shift_value)
        for n in range(1, len(arrows)):
            next_arrow.shift(shift_value)
            arrow_distribution.highlight(arrow_choice_indexes[n], fade_labels=False)
            self.add(arrows[n], top_prob_icons[n])
            self.wait(0.3)
        self.play(
            FadeOut(arrow_distribution),
            FadeOut(next_arrow),
        )
        self.wait()

        # Show bits
        codewords = VGroup(
            bit_string_mobject(s, font="CMU Serif")
            for s in ["0", "10", "110", "111"]
        )
        codewords.scale(0.85)
        all_bits = VGroup()
        for idx, square in zip(arrow_choice_indexes, top_tape):
            bits = codewords[idx].copy()
            bits.space_out_submobjects(0.85)
            bits.next_to(square, UP)
            all_bits.add(bits)

        self.play(LaggedStart(
            (TransformFromCopy(arrow, bits)
            for arrow, bits in zip(arrows, all_bits)),
            lag_ratio=0.1,
        ))
        self.wait()
        self.play(FadeOut(all_bits))

        # Show stream of words
        text_mob = self.get_text_in_tape(text, low_tape)

        self.play(
            frame.animate.set_y(1.0).set_anim_args(time_span=(0, 3)),
            TransformFromCopy(arrows, text_mob, lag_ratio=0.15),
            TransformFromCopy(top_tape, low_tape, lag_ratio=0.15),
            run_time=5
        )
        self.wait()
        self.play(LaggedStartMap(FadeIn, low_prob_icons, lag_ratio=0.5, run_time=4))
        self.wait()

        # Show generation of text by sampling
        char_pred = get_next_char_distribution(text[0])
        all_chars = Text(CHAR_ALPHABET.upper().replace(" ", "_"), font="Consolas")
        text_pred = self.get_prediction(low_tape[0], low_tape[1], char_pred, all_chars)

        self.play(
            frame.animate.move_to(4 * LEFT + DOWN).set_height(6).set_anim_args(run_time=2),
            Write(text_pred[0]),
            FadeIn(text_pred[1]),
            FadeOut(text_mob[1:]),
            FadeOut(low_prob_icons[1:]),
            FadeOut(next_arrow),
            FadeOut(arrow_distribution),
            FadeOut(top_groups),
        )
        self.wait()

        frame.clear_updaters()
        frame.add_updater(lambda m, dt: m.shift(0.3 * dt * RIGHT))
        end_offset = self.final_shown_sample_index

        for n in range(1, n_syms + end_offset):
            # Select
            text_pred[1].highlight(CHAR_ALPHABET.index(text[n].lower()), other_bar_opacity=0.1)
            self.add(text_mob[n])
            self.add(low_prob_icons[n])
            self.wait(0.5)

            # Next pred
            char_pred = get_next_char_distribution(text[:n + 1])
            next_text_pred = self.get_prediction(low_tape[n], low_tape[n + 1], char_pred, all_chars)

            self.play(
                ReplacementTransform(text_pred[1], next_text_pred[1]),
                FadeOut(text_pred[0]),
                GrowArrow(next_text_pred[0]),
                run_time=0.5
            )
            text_pred = next_text_pred
            self.wait(0.3)

        # Recenter
        frame.clear_updaters()
        self.play(
            frame.animate.to_default_state(),
            FadeOut(text_pred),
            FadeIn(text_mob[end_offset:]),
            FadeIn(low_prob_icons[end_offset:]),
            run_time=2,
        )
        self.add(text_mob, low_prob_icons)

        # Show information
        prob_to_info = Tex(R"p \rightarrow -\log_2(p)", font_size=60)
        prob_to_info["p"][0].set_color(GREEN)
        prob_to_info[R"-\log_2(p)"][0].set_color(BLUE)
        prob_to_info.next_to(low_tape, UP, buff=3.0)

        top_group = VGroup(top_tape, arrows, top_prob_icons)

        low_infos = -np.log2(low_probs)
        bit_height = 0.2
        low_info_bars = self.get_information_bars(low_tape, low_infos, bit_height=bit_height)
        low_info_bars.save_state()
        for bar in low_info_bars:
            bar.stretch(1e-6, 1, about_edge=DOWN)
            bar.set_stroke(width=0)

        self.play(
            FadeIn(prob_to_info, 2 * UP, run_time=2),
            Restore(low_info_bars, lag_ratio=0.1, run_time=3),
            frame.animate.center().set_anim_args(run_time=3),
        )
        self.wait()

        # Show bit lines
        n_bit_lines = 12
        bit_lines = Line(low_tape.get_left(), low_tape.get_right()).replicate(n_bit_lines)
        bit_lines.set_stroke(GREY, 1)
        bit_lines.arrange(UP, buff=bit_height)
        bit_lines.align_to(low_info_bars, DOWN)
        bit_line_labels = VGroup(
            Text(f"{n} bits", font_size=12).set_color(GREY).next_to(line, LEFT, SMALL_BUFF)
            for n, line in enumerate(bit_lines)
        )

        self.add(bit_lines, low_info_bars)
        self.play(
            ShowCreation(bit_lines, lag_ratio=0.03),
            LaggedStartMap(FadeIn, bit_line_labels),
            frame.animate.set_height(8.2, about_edge=RIGHT),
            run_time=2
        )
        self.wait()

        # Highlight a few examples
        fade_rects = FullScreenFadeRectangle().replicate(2)
        fade_rects.set_fill(opacity=0.7)
        fade_rects[0].align_to(low_tape[3], RIGHT)
        fade_rects[1].align_to(low_tape[12], LEFT)
        self.add(fade_rects, prob_to_info)

        self.play(FadeIn(fade_rects))
        self.wait()
        self.play(
            fade_rects[0].animate.align_to(low_tape[18], RIGHT),
            fade_rects[1].animate.align_to(low_tape[20], LEFT),
        )
        self.wait()
        self.play(FadeOut(fade_rects))
        self.wait()

        # Reference units of bits
        low_prob_icons.save_state()
        bit_names = VGroup(label["bits"][0] for label in bit_line_labels)
        bit_name_rects = VGroup(
            SurroundingRectangle(name, buff=0.025)
            for name in bit_names
        )
        bit_name_rects.set_stroke(YELLOW, 2)

        self.play(
            frame.animate.reorient(0, 0, 0, (-4.64, -0.65, 0.0), 3.57),
            FadeOut(prob_to_info),
            LaggedStart(
                (name.animate.set_color(YELLOW)
                for name in bit_names),
                lag_ratio=0.05,
            ),
            run_time=3,
        )
        self.play(
            LaggedStartMap(
                VShowPassingFlash,
                bit_name_rects,
                time_width=1.5,
                lag_ratio=0.05,
            ),
            bit_line_labels.animate.set_color(GREY_B),
            run_time=3
        )
        self.wait()

        # Highlight I
        def get_info_label(index):
            bar = low_info_bars[index]
            info_label = DecimalNumber(low_infos[index])
            info_label.set_backstroke(BLACK, 5)
            info_label.set_width(1.5 * bar.get_width())
            info_label.next_to(bar, UP, buff=SMALL_BUFF)
            return info_label

        I_info_label = get_info_label(0)

        fractional_bits = bit_string_mobject("01100")
        frac_rect = BackgroundRectangle(fractional_bits[-1], buff=0)
        frac_rect.stretch(1 - (low_infos[0] % 1), 1, about_edge=DOWN)
        frac_rect.set_fill(BLACK, 0.9)
        fractional_bits.add(frac_rect)
        fractional_bits.next_to(text_mob[0], LEFT, buff=1.0)
        fractional_bits.match_y(I_info_label)
        fractional_bits.set_fill(border_width=0)

        bits_arrow = Arrow(
            text_mob[0],
            fractional_bits,
            SMALL_BUFF,
            path_arc=-60 * DEG,
            fill_color=TEAL,
        )

        self.play(
            Write(I_info_label),
            *(
                group[1:].animate.set_opacity(0.2)
                for group in [low_info_bars, text_mob, low_prob_icons]
            )
        )
        self.play(
            GrowArrow(bits_arrow),
            GrowFromPoint(fractional_bits, I_info_label.get_center()),
            frame.animate.reorient(0, 0, 0, (-5.72, -0.44, 0.0), 3.95),
        )
        self.wait()

        # Highlight the "o"
        o_index = 9
        o_info_label = get_info_label(o_index)

        fractional_bit = fractional_bits[-2:].copy()
        fractional_bit[1].stretch(1.1, 1, about_edge=DOWN).set_opacity(0.9)
        fractional_bit.next_to(o_info_label, UL, LARGE_BUFF)
        fractional_bit.add_to_back(
            BackgroundRectangle(fractional_bit, buff=0.1).set_fill(BLACK, 1).set_stroke(WHITE, 1, 0.5)
        )

        new_bit_arrow = Arrow(
            text_mob[o_index].get_left(),
            fractional_bit.get_bottom(),
            path_arc=-60 * DEG,
            fill_color=TEAL,
            buff=SMALL_BUFF,
        )

        self.play(
            frame.animate.reorient(0, 0, 0, (-3.23, -0.34, 0.0), 4.58).set_anim_args(run_time=2),
            FadeOut(fractional_bits),
            FadeOut(bits_arrow),
            FadeTransform(I_info_label, o_info_label, run_time=2),
            *(
                AnimationGroup(
                    group[o_index].animate.set_opacity(1),
                    group[0].animate.set_opacity(0.2),
                )
                for group in [low_info_bars, text_mob, low_prob_icons]
            )
        )
        self.wait()
        self.play(
            GrowArrow(new_bit_arrow),
            GrowFromPoint(fractional_bit, o_info_label.get_center())
        )
        self.wait()

        # Zoom out to full message
        short_text = "information theory"
        n_short = len(short_text)
        short_text_mob = text_mob[:n_short]
        low_prob_icons.saved_state[n_short:].set_opacity(0)
        p_labels = VGroup(Tex(Rf"p_{{{n}}}") for n in range(n_short))
        p_labels.scale(0.5)
        for icon, p_label in zip(low_prob_icons.saved_state, p_labels):
            p_label.move_to(icon[2])
            icon[2].set_opacity(0)

        self.remove(low_tape[n_short:])
        bit_lines.match_width(low_tape[:n_short], stretch=True, about_edge=LEFT)
        self.add(bit_lines, Point())
        self.play(
            frame.animate.set_height(8).match_x(short_text_mob).set_y(1),
            FadeOut(o_info_label),
            FadeOut(new_bit_arrow),
            FadeOut(fractional_bit),
            low_info_bars[:n_short].animate.set_opacity(1),
            low_info_bars[n_short:].animate.set_opacity(0),
            short_text_mob.animate.set_opacity(1),
            text_mob[n_short:].animate.set_opacity(0),
            Restore(low_prob_icons),
            FadeIn(p_labels, lag_ratio=0.1),
            run_time=2
        )

        # Encoding a phrase
        phrase = Text(f"“{short_text.upper()}”", font="Consolas")
        phrase.next_to(frame.get_top(), DOWN, buff=MED_LARGE_BUFF)
        encoding = int_to_bit_string(30)
        encoding.next_to(phrase, DOWN, LARGE_BUFF)
        arrow = Arrow(phrase, encoding, buff=SMALL_BUFF)

        self.play(
            FadeTransformPieces(text_mob["INFORMATION"][0].copy(), phrase["INFORMATION"][0]),
            FadeTransformPieces(text_mob["THEORY"][0].copy(), phrase["THEORY"][0]),
            Write(phrase[0], time_span=(0.5, 1.5)),
            Write(phrase[-1], time_span=(0.5, 1.5)),
        )
        self.play(
            GrowArrow(arrow),
            TransformFromCopy(phrase, encoding, lag_ratio=0.05, run_time=2),
        )
        self.wait()

        # Show probability factored out
        p_of = Tex(R"P()")
        p_of[:2].next_to(phrase, LEFT, SMALL_BUFF)
        p_of[2:].next_to(phrase, RIGHT, SMALL_BUFF)

        equals = Tex(R"=", font_size=72)
        equals.rotate(90 * DEG)
        equals.next_to(phrase, DOWN)

        factored_prob = Tex(R"p_0 \cdot p_1 \cdot p_2 \cdot p_3 \cdots p_n")
        factored_prob.next_to(equals, DOWN)

        encoding.save_state()

        self.play(
            encoding.animate.set_width(2.5).set_opacity(0.5).next_to(frame.get_corner(UR), DL),
            FadeOut(arrow, scale=0.5),
            Write(p_of),
        )
        self.play(
            Write(equals),
            *(
                TransformFromCopy(p_labels[n], factored_prob[f"p_{n}"][0])
                for n in [0, 1, 2, 3]
            ),
            TransformFromCopy(p_labels[-1], factored_prob["p_n"][0]),
            TransformFromCopy(p_labels[3:-1], factored_prob[R"\cdots"][0]),
            Write(factored_prob[R"\cdot"][:-1], time_span=(0.5, 2)),
            run_time=2
        )
        self.wait()

        # Cycle through probabilities
        icon_ghosts = low_prob_icons.copy().fade(0.75)

        self.add(icon_ghosts)
        self.play(LaggedStartMap(VFadeInThenOut, low_prob_icons, lag_ratio=0.25, run_time=5))
        self.play(FadeIn(low_prob_icons), FadeOut(icon_ghosts))
        self.wait()

        # Take logs
        log_of = Tex(R"-\log_2()").set_color(BLUE)
        log_of["()"].scale(1.1, about_edge=LEFT)
        log_ofs = log_of.replicate(2)

        for log, term in zip(log_ofs, [p_of, factored_prob]):
            log[:-1].next_to(term, LEFT, buff=SMALL_BUFF)
            log[-1:].next_to(term, RIGHT, buff=SMALL_BUFF)

        self.play(
            *map(Write, log_ofs)
        )
        self.wait()

        # Break up the log
        log_sum = Tex(
            R"-\log_2(p_0) -\log_2(p_1) -\log_2(p_2) - \cdots -\log_2(p_n)",
            t2c={R"\log_2": BLUE, "(": BLUE, ")": BLUE}
        )
        log_sum.next_to(equals, DOWN)

        stacked_bars = low_info_bars[:n_short].copy()
        stacked_bars.rotate(-90 * DEG)
        stacked_bars.arrange(RIGHT, buff=SMALL_BUFF)
        stacked_bars.match_width(log_sum)
        stacked_bars.next_to(log_sum, DOWN)

        self.remove(log_ofs[1])
        self.play(
            *(
                ReplacementTransform(factored_prob[f"p_{n}"], log_sum[f"p_{n}"])
                for n in [0, 1, 2, "n"]
            ),
            *(
                FadeOut(factored_prob[s])
                for s in ["p_3", R"\cdot"]
            ),
            ReplacementTransform(factored_prob[R"\cdots"], log_sum[R"\cdots"]),
            TransformFromCopy(log_ofs[1][R"-\log_2("], log_sum[R"-\log_2("]),
            TransformFromCopy(log_ofs[1][R")"], log_sum[R")"]),
            FadeIn(log_sum["-"][3]),
            run_time=2
        )
        self.wait()
        self.play(
            TransformFromCopy(low_info_bars[:n_short], stacked_bars)
        )
        self.play(stacked_bars.animate.arrange(RIGHT, buff=0).move_to(stacked_bars.get_center()))
        self.wait()

        # Relate encoding to information
        phrase.target = phrase.generate_target()
        phrase.target.shift(UP)

        arrow.next_to(phrase.target, DOWN, SMALL_BUFF)

        encoding.target = encoding.generate_target()
        encoding.target.set_opacity(1)
        encoding.target.match_width(stacked_bars)
        encoding.target.next_to(arrow, DOWN, SMALL_BUFF)

        self.play(
            frame.animate.set_height(9, about_edge=DOWN),
            FadeOut(log_ofs[0], UP),
            FadeOut(p_of, UP),
            MoveToTarget(phrase),
            FadeOut(equals),
            MoveToTarget(encoding),
            GrowArrow(arrow, time_span=(1, 2)),
            log_sum.animate.match_width(stacked_bars).next_to(stacked_bars, UP, SMALL_BUFF),
            run_time=2
        )

        # Add labels
        encoding_length = Text("Optimal \n Encoding length")
        encoding_length.next_to(encoding, LEFT, LARGE_BUFF, aligned_edge=DOWN)
        total_information = Text("Total information")
        total_information.next_to(VGroup(log_sum, stacked_bars), LEFT, LARGE_BUFF)
        total_information.set_color(BLUE_C)

        approx = Tex(R"\approx", font_size=72)
        approx.rotate(90 * DEG)
        approx.move_to(midpoint(encoding_length.get_bottom(), total_information.get_top()))

        equiv = VGroup(encoding_length, approx, total_information)

        self.play(
            FadeIn(equiv, LEFT),
            frame.animate.shift(1.5 * LEFT)
        )
        self.wait()

    def get_text_in_tape(self, text, tape):
        text_mob = Text(text.replace(" ", ".").upper(), font="Consolas")
        for char, char_mob, square in zip(text, text_mob, tape):
            char_mob.move_to(square)
            if char == " ":
                char_mob.scale(0)
            if char == "'":
                char_mob.shift(tape[0].get_height() * 0.25 * UP)
        return text_mob

    def get_prob_icons(self, tape, probs, width_ratio=0.7, max_height=2.0, color=GREEN, labeled=True):
        radius = tape[0].get_width() * 0.5 * width_ratio
        icons = VGroup(
            ProbIcon(
                p,
                radius=radius,
                label_font_size=16,
            ).next_to(box, DOWN, SMALL_BUFF)
            for p, box in zip(probs, tape)
        )
        return icons

        if labeled:
            for p, icon in zip(probs, icons):
                ndp = 0
                if p < 0.01:
                    ndp = 2
                elif (100 * p) % 1 == 0.5:
                    ndp = 1
                label = DecimalNumber(
                    100 * p,
                    unit="%",
                    font_size=16,
                    num_decimal_places=ndp
                )
                label.next_to(icon, DOWN, SMALL_BUFF)
                icon.add(label)
        return icons

    def get_information_bars(self, tape, info_values, width=0.25, bit_height=0.2, color=BLUE_D):
        rects = VGroup(
            Rectangle(width=width, height=bit_height * bits).next_to(box, UP, SMALL_BUFF)
            for box, bits in zip(tape, info_values)
        )
        rects.set_fill(color, 1)
        rects.set_stroke(WHITE, 1)
        return rects

    def get_prediction(self, box1, box2, dist, symbols, dist_mob_width=5, dist_mob_height=0.3):
        arrow = Arrow(
            LEFT, RIGHT,
            path_arc=-170 * DEG,
            thickness=5,
            buff=0.1
        )
        p1 = box1.get_top()
        p2 = box2.get_top()
        arrow.scale(0.5 * get_norm(p1 - p2))
        arrow.next_to(0.5 * (p1 + p2), UP, SMALL_BUFF)

        dist_mob = StackedProbDistribution(
            dist,
            width=dist_mob_width,
            height=dist_mob_height,
            labels=symbols.copy()
        )
        dist_mob.next_to(arrow, UP, SMALL_BUFF)
        return VGroup(arrow, dist_mob)

    def old(self):
        # Show generation of arrows by sampling (much copied from above)
        arrow_pred = self.get_prediction(top_tape[0], top_tape[1], arrow_dist, arrow_templates)

        for n in range(1, n_syms):
            # Select
            arrow_pred[1].highlight(arrow_choice_indexes[n], other_bar_opacity=0.1)
            self.add(arrows[n])
            self.wait(0.5)

            # Next pred
            next_arrow_pred = self.get_prediction(top_tape[n], top_tape[n + 1], arrow_dist, arrow_templates.copy())

            self.play(
                ReplacementTransform(arrow_pred[1], next_arrow_pred[1]),
                FadeOut(text_pred[0]),
                GrowArrow(next_text_pred[0]),
                run_time=0.2
            )
            text_pred = next_text_pred
            self.wait(0.5)

    def cycle_through_phrases(self):
        # Insert all this at the point of "Highlight a few examples"
        # Show average line
        avg_line = bit_lines[0].copy()
        avg_line.set_stroke(YELLOW, 2)
        avg_line.shift(bit_height * low_infos.mean() * UP)
        avg_word = Text("Average", font_size=48)
        avg_word.match_color(avg_line)
        avg_word.set_backstroke(BLACK, 5)
        avg_word.next_to(avg_line, UP, buff=-0.05)
        avg_word.match_x(low_tape[8])

        self.remove(prob_to_info)
        self.play(
            ShowCreation(avg_line),
            Write(avg_word),
        )
        self.wait()
        self.play(
            avg_line.animate.set_stroke(opacity=0.5),
            avg_word.animate.set_opacity(0.5),
        )

        # Cycle through alternate phrases
        phrases = [
            "great ideas are unexpected ",
            "general laws are are succinct, ",
            "once upon a time, there was",
            "3blue1brown is a weird name",
            "al-Khwarizmi was a mathematician",
            "prediction is compression, as shown",
            "a solemn fact to reflect upon, that every",
            "it is not the critic who counts, it is the",
            "cold and timid souls who know",
            "randy the pi creature, the blue one,",
        ]

        for phrase in phrases:
            new_text = phrase[:len(text)]
            new_text_mob = self.get_text_in_tape(new_text, low_tape)
            probs = np.array([
                get_next_char_distribution(" " if n == 0 else new_text[:n])[CHAR_ALPHABET.index(char)]
                for n, char in enumerate(new_text.lower())
            ])
            try:
                idx = new_text.index(",")
                new_text_mob[idx].align_to(low_tape, DOWN)
                probs[idx] = 0.3
            except ValueError:
                pass
            infos = -np.log2(probs)
            prob_icons = self.get_prob_icons(low_tape, probs)
            info_bars = self.get_information_bars(low_tape, -np.log2(probs), bit_height=bit_height)

            kw = dict(lag_ratio=0.05, run_time=2)
            self.play(
                FadeOut(text_mob, **kw),
                FadeIn(new_text_mob, **kw),
                FadeTransformPieces(text_mob, new_text_mob, **kw),
                ReplacementTransform(low_prob_icons, prob_icons, **kw),
                ReplacementTransform(low_info_bars, info_bars, **kw),
            )

            text_mob = new_text_mob
            low_prob_icons = prob_icons
            low_info_bars = info_bars

    def old_for_entropy(self):
        # Highlight all terms
        char_rects, prob_rects = all_rects = VGroup(
            VGroup(SurroundingRectangle(c, buff=0.05) for c in group)
            for group in [text_mob, low_prob_icons]
        )
        all_rects.set_stroke(YELLOW, 2)
        prob_to_info.set_width(2)
        prob_to_info.next_to(bits_group, UP, MED_SMALL_BUFF, LEFT).shift(0.5 * RIGHT)

        low_info_bars.save_state()
        for bar in low_info_bars:
            bar.stretch(0, 1, about_edge=DOWN)

        self.play(LaggedStartMap(ShowCreation, char_rects, run_time=1, lag_ratio=0.1))
        self.wait()
        self.play(ReplacementTransform(char_rects, prob_rects, lag_ratio=0.05))
        self.wait()
        self.add(bits_group, low_info_bars)
        self.play(
            FadeIn(bits_group),
            FadeIn(prob_to_info),
            FadeOut(prob_rects),
            Restore(low_info_bars, lag_ratio=0.05),
        )
        self.wait()

        # Show average line
        avg_line = bit_lines[0].copy()
        avg_line.set_stroke(YELLOW, 2)
        avg_line.shift(bit_height * low_infos.mean() * UP)
        avg_word = Text("Average", font_size=48)
        avg_word.match_color(avg_line)
        avg_word.set_backstroke(BLACK, 5)
        avg_word.next_to(avg_line, UP, buff=-0.05)
        avg_word.match_x(low_tape[8])

        self.play(ShowCreation(avg_line), Write(avg_word))
        self.wait()

        # Emphasize this is not well-defined
        eng_underline = Underline(entropy_question, buff=-0.07, stretch_factor=1)
        eng_underline.set_stroke(RED)
        morty = Mortimer(height=1.5)
        morty.next_to(eng_underline.get_right(), DOWN)
        q_marks = Text("???", font_size=72)
        q_marks.match_color(eng_underline)
        q_marks.next_to(morty, LEFT, aligned_edge=UP)

        for mob in eng_underline, morty, q_marks:
            mob.fix_in_frame()

        self.play(
            VFadeIn(morty),
            morty.change('confused').fix_in_frame(),
            ShowCreation(eng_underline),
            frame.animate.shift(1.0 * UP),
        )
        self.play(Write(q_marks))
        self.play(Blink(morty))
        self.play(
            morty.change("pondering", low_info_bars).fix_in_frame(),
            FadeOut(avg_word),
            FadeOut(avg_line),
        )
        self.wait()

        # Cycle through alternate phrases
        phrases = [
            "Mathematics is an art of logical abstractions",
            "Come on fhqwhgads I said come on fhqwhgads",
            "Oi, that's bollocks and you know it bruv",
            "yo that party last night was straight fire no cap fr fr",
            "omg lol i cant even rn this is sooo cursed lmaooo",
            "G'day mate, chuck us a tinnie from the esky",
            "ATP hydrolysis drives conformational changes in myosin",
            "tfw u finally git push --force and nobody notices",
            "It was the best of times it was the worst",
        ]

        for phrase in phrases:
            new_text = phrase[:len(text)]
            new_text_mob = self.get_text_in_tape(new_text, low_tape)
            probs = np.array([
                get_next_char_distribution(" " if n == 0 else new_text[:n])[CHAR_ALPHABET.index(char)]
                for n, char in enumerate(new_text.lower())
            ])
            if phrase in phrases[-1]:
                # Fudge a little to make predictable
                qs = 1 - probs
                qs *= np.linspace(1, 0, len(probs))**0.5
                probs = 1 - qs
            infos = -np.log2(probs)
            prob_icons = self.get_prob_icons(low_tape, probs)
            info_bars = self.get_information_bars(low_tape, -np.log2(probs), bit_height=bit_height)

            kw = dict(lag_ratio=0.05, run_time=2)
            self.play(
                FadeOut(text_mob, **kw),
                FadeIn(new_text_mob, **kw),
                FadeTransformPieces(text_mob, new_text_mob, **kw),
                ReplacementTransform(low_prob_icons, prob_icons, **kw),
                ReplacementTransform(low_info_bars, info_bars, **kw),
            )
            self.wait()

            text_mob = new_text_mob
            low_prob_icons = prob_icons
            low_info_bars = info_bars

        # Reference 1 bit estimate
        H_English = Tex(R"H[\text{English}]", t2c={"H": BLUE}, font_size=72)
        H_English.move_to(entropy_question)
        H_English.fix_in_frame()

        rhs = Tex(R"\approx 1 \; {\text{bit} \over \text{char}}")
        rhs[R"{\text{bit} \over \text{char}}"].scale(0.8)
        rhs.fix_in_frame()
        rhs.next_to(H_English, RIGHT)

        self.play(
            TransformMatchingTex(
                entropy_question,
                H_English,
                matched_keys={R"\text{Language}": R"\text{English}"},
                match_animation=FadeTransform,
            ),
            FadeOut(eng_underline),
            FadeOut(q_marks),
            morty.change("hesitant", H_English).fix_in_frame(),
        )
        self.play(Blink(morty))
        self.wait()
        self.play(
            morty.change("pondering", rhs).fix_in_frame(),
            Write(rhs),
        )
        self.wait()

        # Show bits
        bit_tape = low_tape.copy()
        bit_tape.shift(1.5 * DOWN)
        bits_text = "".join(random.choice(["0", "1"]) for m in text_mob)
        bits_mob = self.get_text_in_tape(bits_text, bit_tape)
        for bit, part in zip(bits_text, bits_mob):
            part.set_color([GREY_C, GREY_A][int(bit)])

        self.play(
            frame.animate.shift(1.5 * DOWN),
            FadeOut(prob_to_info),
            FadeOut(bit_lines[6:]),
            FadeOut(bit_line_labels[6:]),
            FadeOut(low_prob_icons),
            morty.change("thinking", low_tape).fix_in_frame()
        )

        self.play(
            TransformFromCopy(low_tape, bit_tape, lag_ratio=0.1),
            TransformFromCopy(text_mob, bits_mob, lag_ratio=0.1),
            morty.change("maybe", bits_mob).fix_in_frame().set_anim_args(time_span=(1, 2)),
            run_time=3
        )
        self.wait()


class GenericSymbolInformation(InformationOfLanguage):
    def construct(self):
        # Show tape with symbols and probabilities
        sym = Tex(R"S_1")
        var = sym.make_number_changeable("1", edge_to_fix=LEFT)
        var.scale(0.75, about_edge=LEFT)
        n_syms = 16

        tape = Square().get_grid(1, n_syms, buff=0)
        tape.set_width(FRAME_WIDTH - 1)

        symbols = VGroup()
        for n, square in enumerate(tape):
            var.set_value(n)
            new_sym = sym.copy()
            new_sym.move_to(square)
            symbols.add(new_sym)

        symbols.set_submobject_colors_by_gradient(YELLOW_B, RED)

        probs = np.random.random(n_syms)
        prob_icons = self.get_prob_icons(tape, probs)

        self.add(tape)
        self.play(
            ShowIncreasingSubsets(symbols),
            ShowIncreasingSubsets(prob_icons),
            rate_func=linear,
            run_time=2
        )
        self.wait()

        # Show average information information bars
        frame = self.frame
        info_values = -np.log2(probs)
        bit_height = 0.5
        info_bars = self.get_information_bars(tape, info_values, bit_height=bit_height)
        for bar in info_bars:
            bar.save_state()
            bar.stretch(0, 1, about_edge=DOWN)
            bar.set_stroke(opacity=0)

        avg_line = DashedLine(info_bars.get_left(), info_bars.get_right())
        avg_line.scale(1.05).align_to(tape, LEFT)
        avg_line.set_stroke(YELLOW, 3)
        avg_line.set_y(info_bars.get_y(DOWN) + bit_height * info_values.mean())
        avg_word = Text("Average\nInformation")
        avg_word.match_color(avg_line)
        avg_word.next_to(avg_line, RIGHT, SMALL_BUFF)

        self.play(
            frame.animate.reorient(0, 0, 0, (1.15, 0.56, 0.0), 9.71),
            LaggedStartMap(Restore, info_bars),
            ShowCreation(avg_line),
            FadeIn(avg_word, shift=3 * RIGHT, time_span=(0.5, 2)),
            run_time=2
        )
        self.wait()

        # Stack bars, put bits under
        stacked_bars = info_bars.copy()
        stacked_bars.rotate(-90 * DEG)
        stacked_bars.arrange(RIGHT, buff=0)
        stacked_bars.stretch(2, 1)
        stacked_bars.stretch(0.8, 0)
        stacked_bars.next_to(prob_icons, DOWN, LARGE_BUFF)
        stacked_bars.set_fill(BLUE_E)

        self.play(
            TransformFromCopy(info_bars, stacked_bars, lag_ratio=0.1),
            run_time=2
        )
        self.wait()
