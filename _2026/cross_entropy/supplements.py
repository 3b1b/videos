from manim_imports_ext import *
from _2026.cross_entropy.next_char import total_information
from _2026.cross_entropy.entropy import ArrowGroup
from _2026.cross_entropy.entropy import int_to_bit_string
from _2026.cross_entropy.entropy import bit_string_mobject
from _2026.cross_entropy.distribution import StackedProbDistribution
from _2024.transformers.helpers import Dial


class SimpleRect(InteractiveScene):
    dimensions = (1, 1)

    def construct(self):
        # Test
        rect = Rectangle(*self.dimensions).set_stroke(YELLOW, 3)
        rect.add_line_to(rect.get_corner(UL))
        rect.insert_n_curves(100)
        self.play(VShowPassingFlash(rect, time_width=1.5, run_time=5))
        self.wait()


class SimpleRectWide(SimpleRect):
    dimensions = (12, 1.5)


class SimpleRect3x4(SimpleRect):
    dimensions = (3, 4)


class SimpleRect2x4(SimpleRect):
    dimensions = (2, 4)


class SimpleRect4x1(SimpleRect):
    dimensions = (2, 0.5)


class AmbientEncodingMachine3(InteractiveScene):
    def construct(self):
        # Set up machine, bit stream and text stream
        machine = self.get_machine()
        text_stream = self.get_text_stream()
        bit_stream = self.get_bit_stream()

        vel_tracker = bit_stream.vel_tracker

        dial = Dial(
            value_range=(0, 10, 1),
            arc_angle=180 * DEG,
            radius=1,
            set_anim_streak_width=0,
        )
        dial.set_value(8)
        dial.next_to(machine, UP, buff=-0.5)

        label = Tex(R"8.0 \text{bits} / \text{sym}", font_size=36)
        dec = label.make_number_changeable("8.0")
        dec.shift(0.1 * LEFT)
        dec.set_color(BLUE)
        label.next_to(dial, UP)

        self.add(text_stream, bit_stream)
        self.add(machine, dial, label)

        self.wait(2)

        # Decrease
        # values = [4, 2.5, 1.5]
        values = [8]
        for value in values:
            self.play(
                vel_tracker.animate.set_value(value),
                dial.animate_set_value(value),
                ChangeDecimalToValue(dec, value),
                run_time=3
            )
            self.wait()

        # Ask about a fundamental limit
        limit_line = Line(dial.needle.get_start(), dial.value_to_point(1))
        limit_line.scale(1.5, about_point=limit_line.get_start())
        limit_line.set_stroke(RED, 5)
        # limit_word = Text("Limit?", font_size=30)
        limit_word = Tex("H(P)", font_size=30)
        limit_word.set_color(RED)
        limit_word.next_to(limit_line.get_end(), LEFT, SMALL_BUFF)

        self.play(
            ShowCreation(limit_line),
            FadeIn(limit_word, 0.25 * LEFT, lag_ratio=0.1, time_span=(0.5, 1.5))
        )
        self.play(
            vel_tracker.animate.set_value(1.0),
            dial.animate_set_value(1.0),
            ChangeDecimalToValue(dec, 1.0),
            run_time=10,
        )
        self.wait(20)

    def get_machine(self, width=1.5, height=1, color=GREY_D):
        square = Rectangle(width, height)
        in_tri = ArrowTip().set_height(0.5 * height)
        in_tri.stretch(2, 1)
        out_tri = in_tri.copy().rotate(PI)
        in_tri.move_to(square.get_left())
        out_tri.move_to(square.get_right())
        machine = Union(square, in_tri, out_tri)
        machine.set_fill(color, 1)
        machine.set_stroke(WHITE, 2)
        machine.set_z_index(1)
        machine.set_fill(opacity=0.8)
        return machine

    def get_text_stream(self, velocity=1, symbols=True):
        # Test
        text_parts = [
            "It was the best of times, it was the worst of times,",
            "it was the age of wisdom, it was the age of foolishness,",
            "it was the epoch of belief, it was the epoch of incredulity,",
            "it was the season of Light, it was the season of Darkness,",
            "it was the spring of hope, it was the winter of despair,",
            "we had everything before us, we had nothing before us,",
            "we were all going direct to Heaven, we were all going",
            "direct the other way – in short, the period was so far",
            "like the present period, that some of its noisiest authorities",
            "insisted on its being received, for good or for evil, in the",
            "superlative degree of comparison only.",
        ]
        text_mobs = VGroup(Text(part, font="Consolas") for part in text_parts)
        text_mobs.arrange(RIGHT, MED_SMALL_BUFF)
        text_stream = VGroup(*it.chain(*text_mobs))
        if symbols:
            values = np.random.choice([1, 2, 3, 4, 5], size=100, p=[0.1, 0.4, 0.2, 0.1, 0.2])
            colors = color_gradient([BLUE, YELLOW], 5, interp_by_hsl=True)
            text_stream = VGroup(
                Tex(Rf"s_{{{n}}}").set_color(colors[n - 1])
                for n in values
            )
            text_stream.arrange(RIGHT, buff=SMALL_BUFF, aligned_edge=UP)
        text_stream.next_to(ORIGIN, RIGHT)

        def update_stream(stream, dt):
            stream.shift(velocity * dt * LEFT)
            for ch in stream:
                ch.set_opacity(clip(ch.get_x(), 0, 1))

        text_stream.add_updater(update_stream)

        return text_stream

    def get_bit_stream(self, n_bits=100, velocity=8):
        bits = bit_string_mobject("".join(random.choice("01") for n in range(n_bits)))
        bits.vel_tracker = ValueTracker(velocity)
        buff = get_norm(bits[0].get_right() - bits[1].get_left())

        def update_bit_stream(bits, dt):
            bits.shift(bits.vel_tracker.get_value() * dt * LEFT)
            for bit in bits:
                bit.set_opacity(clip(-bit.get_x(), 0, 1))
                if bit.get_x() < -10:
                    bit.next_to(bits, RIGHT, buff=buff)

        bits.move_to(ORIGIN, RIGHT)
        bits.add_updater(update_bit_stream)

        return bits


class AskHow(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        self.play(randy.thinks("How?", mode="maybe", look_at=3 * UR))
        self.play(Blink(randy))
        self.wait()


class Timeline(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame

        btn = list(range(1900, 2030, 10))
        timeline = NumberLine(
            (1900, 2030, 1),
            width=20,
            big_tick_numbers=btn,
            tick_size=0.05,
            longer_tick_multiple=2,
        )
        timeline.add_numbers(btn, group_with_commas=False)

        tip = ArrowTip(angle=-90 * DEG)
        tip.set_color(TEAL)
        tip.set_height(0.2)
        tip.move_to(timeline.n2p(2026), DOWN)

        frame.set_height(6).move_to(timeline, RIGHT)

        self.add(timeline, tip)
        self.play(
            tip.animate.move_to(timeline.n2p(1948), DOWN),
            frame.animate.match_x(timeline.n2p(1970)).set_anim_args(time_span=(1, 5)),
            run_time=6,
        )


class SimpleNetwork(InteractiveScene):
    random_seed = 2

    def construct(self):
        # Set up a basic network
        frame = self.frame
        layer_sizes = [4, 6, 6, 4]
        layer_spacing = 4

        layers = VGroup(self.get_layer(size) for size in layer_sizes)
        layers.arrange(RIGHT, buff=layer_spacing)
        layers.set_height(6)

        connections = VGroup()
        for l1, l2 in zip(layers, layers[1:]):
            lines = VGroup()
            for n2 in l2:
                n2.connections = self.get_connections(n2, l1)
                lines.add(n2.connections)
            connections.add(lines)

        network = VGroup(layers, connections)
        network.set_height(3).to_edge(DOWN)

        self.add(network)
        # self.play(network.animate.set_height(0.75).center(), run_time=2)
        network.set_height(0.75).center()
        self.play(Write(network, run_time=2, lag_ratio=0.01))
        self.wait()

    def get_layer(self, layer_size, radius=0.3, fill_opacity=0.5, stroke_width=3, buff=1):
        layer = VGroup(
            Circle(radius=radius)
            for n in range(layer_size)
        )
        layer.set_fill(GREY, fill_opacity)
        layer.set_stroke(WHITE, stroke_width)
        layer.arrange(DOWN, buff=buff)

        return layer

    def get_connections(self, target, sources, max_stroke_width=1.5, colors=[BLUE, RED]):
        target.connections = VGroup()
        for neuron in sources:
            line = Line(
                neuron.get_center(),
                target.get_center(),
                buff=neuron.get_radius(),
                stroke_width=max_stroke_width * random.random(),
                stroke_color=random.choice(colors)
            )
            target.connections.add(line)
        return target.connections


class WriteLLM(InteractiveScene):
    def construct(self):
        llm = Text("LLM", font_size=72)
        llm.arrange(DOWN)

        self.play(FadeIn(llm, lag_ratio=0.1))
        self.wait(0.5)
        self.play(FadeOut(llm))


class PartialBitsForArithmeticCoding(InteractiveScene):
    def construct(self):
        # Test
        word = "compress"
        bits = "0.00110101000011100000010001010101010100101101010010101110"

        partial_infos = [
            int(np.ceil(total_information(word[:n])))
            for n in range(len(word))
        ]
        bit_string = bit_string_mobject(bits[:partial_infos[-1] + 2])
        bit_string.scale(2)
        bit_string.to_edge(UP)

        self.add(bit_string[:2])
        for n1, n2 in zip(partial_infos, partial_infos[1:]):
            self.play(
                FadeIn(
                    bit_string[n1 + 2:n2 + 2],
                    shift=0.2 * UP,
                    lag_ratio=0.75
                )
            )
            self.wait()


class SimpleEquiv(InteractiveScene):
    def construct(self):
        self.play(Write(Tex(R"\Leftrightarrow", font_size=90)))
        self.wait()


class CrossEntropyWords(InteractiveScene):
    def construct(self):
        text = Text("“Cross-entropy” loss", font_size=72)
        text.to_edge(UP)
        ce = text["Cross-entropy"]
        underline = Underline(ce)
        underline.set_stroke(color=PINK)

        self.play(FadeIn(text, UP))
        self.wait()
        self.play(
            ShowCreation(underline),
            ce.animate.set_color(PINK),
        )
        self.wait()


class CompressionIsIntellgenceWords(InteractiveScene):
    def construct(self):
        # Intro
        words = ["Compression", "is", "Intelligence"]
        colors = ["#0BFCB7", WHITE, YELLOW]
        phrase_mob = Text(" ".join(words), font_size=90)
        compression, is_, intelligence = mobs = VGroup(
            phrase_mob[word][0]
            for word in words
        )
        for mob, color in zip(mobs, colors):
            mob.set_color(color)

        compression.shift(0.5 * LEFT)
        intelligence.shift(0.5 * RIGHT)
        phrase_mob.to_edge(UP)

        self.play(ShowIncreasingSubsets(mobs, run_time=3))
        self.wait()

        # Conservative phrasing
        phrase_mob.save_state()
        theory_of = Text("The mathematical theory of")
        theory_of.set_color(TEAL_D)
        theory_of.match_width(compression)
        compression.refresh_bounding_box()
        theory_of.move_to(compression, UP)

        compression.target = compression.generate_target()
        compression.target.next_to(theory_of, DOWN, SMALL_BUFF)

        rightarrow = Tex(R"\longrightarrow", font_size=60)
        rightarrow.match_x(is_)
        rightarrow.match_y(VGroup(compression.target, theory_of))
        rightarrow.scale(1.5, about_edge=LEFT)
        useful = Text("is useful for")
        useful.match_width(rightarrow)
        useful.next_to(rightarrow, UP, buff=SMALL_BUFF)

        ai = Text("Artificial\nIntelligence", font_size=72)
        ai.set_color(YELLOW_D)
        ai.match_x(intelligence).match_y(rightarrow)

        self.play(LaggedStart(
            AnimationGroup(
                FadeTransformPieces(compression.copy(), theory_of),
                MoveToTarget(compression),
            ),
            Transform(is_, rightarrow),
            AnimationGroup(
                Transform(intelligence, ai["Intelligence"][0]),
                FadeIn(ai["Artificial"][0], lag_ratio=0.2, shift=0.1 * UP),
            ),
            lag_ratio=0.7
        ))
        self.wait()
        self.play(
            Restore(phrase_mob),
            FadeOut(theory_of, 0.5 * UP),
            FadeOut(ai["Artificial"][0], 0.5 * UP),
        )
        self.wait()

        # Trio of videos
        background = FullScreenFadeRectangle()
        background.set_fill(GREY_E, 0.5)
        videos = ScreenRectangle().replicate(3)
        videos.arrange(RIGHT, buff=LARGE_BUFF)
        videos.set_width(FRAME_WIDTH - 1)
        videos.set_fill(BLACK, 1)
        videos.set_stroke(WHITE, 2)
        brace = Brace(videos, UP, SMALL_BUFF)

        clean_phrase = Text("Compression is Intelligence")
        clean_phrase.next_to(brace, UP)

        phrase_mob.set_z_index(1)
        self.play(LaggedStart(
            FadeIn(background),
            Transform(phrase_mob, clean_phrase),
            GrowFromCenter(brace),
            LaggedStartMap(FadeIn, videos),
            lag_ratio=0.25
        ))
        self.wait()
        self.play(
            FadeOut(phrase_mob, 0.5 * RIGHT, lag_ratio=0.1),
            FadeOut(brace, scale=0.5, shift=0.25 * brace.get_width() * RIGHT),
            FadeOut(videos),
        )
        self.wait()


class ThreeFoldRelationship(InteractiveScene):
    def construct(self):
        # Set up
        tri = Triangle(start_angle=0)
        tri.set_height(4)

        radius = 1.7
        circles = Circle(radius=1.7).replicate(3)
        circles.set_stroke(WHITE, 2)
        circles[:2].arrange(DOWN, buff=LARGE_BUFF)
        circles[2].next_to(circles[:2], RIGHT, buff=1.5)
        circles.center()

        lines = VGroup(
            Line(c1.get_center(), c2.get_center(), buff=radius + SMALL_BUFF)
            for c1, c2 in it.combinations(circles, 2)
        )
        arrows = VGroup(
            Tex(R"\Leftrightarrow"),
            Tex(R"\longleftrightarrow"),
            Tex(R"\longleftrightarrow"),
        )
        for arrow, line in zip(arrows, lines):
            arrow.scale(1.5)
            arrow.rotate(line.get_angle())
            arrow.move_to(line)

        labels = VGroup(
            Text("Prediction").set_color("#951FF5"),
            Text("Compression").set_color(TEAL),
            Text("Intelligence").set_color(YELLOW),
        )
        for label, circle, vect in zip(labels, circles, [LEFT, LEFT, RIGHT]):
            label.next_to(circle, vect, buff=0.2)

        self.add(circles)
        self.play(LaggedStartMap(FadeIn, arrows, scale=10, lag_ratio=0.5))
        self.wait()
        self.play(LaggedStartMap(Write, labels, lag_ratio=0.5, run_time=3))
        self.wait()

        # Transition to AI
        ai = Text("Artificial\nIntelligence")
        ai.set_color(YELLOW_D)
        ai.move_to(labels[2], LEFT)

        self.play(
            ReplacementTransform(labels[2], ai["Intelligence"][0]),
            FadeIn(ai["Artificial"][0], lag_ratio=0.2, shift=0.1 * UP),
        )
        self.wait()


class KeyDefinitions(InteractiveScene):
    def construct(self):
        # Write terms
        terms = VGroup(
            Text("Information"),
            Text("Entropy"),
        )
        terms.scale(1.5)
        terms.to_edge(UP)

        for v, term in zip([LEFT, RIGHT], terms):
            term.shift(0.25 * FRAME_WIDTH * v)

        self.play(LaggedStartMap(Write, terms, lag_ratio=0.5))
        self.wait()

        # From insight
        bulbs = VGroup()
        arrows = VGroup()
        for term in terms:
            bulb = ThoughtBubble()
            bulb.remove(bulb.content)
            bulb.set_stroke(WHITE, 1.5)
            bulb.match_height(term)
            bulb.next_to(term[0], LEFT)
            arrow = Vector(RIGHT, thickness=4)
            arrow.set_color(WHITE)
            arrow.next_to(bulb, RIGHT)
            term.target = term.generate_target()
            term.target.scale(0.8)
            term.target.next_to(VGroup(arrow), RIGHT, MED_SMALL_BUFF, index_of_submobject_to_align=0)

            bulbs.add(bulb)
            arrows.add(arrow)

        self.play(
            LaggedStartMap(FadeIn, bulbs, scale=4, lag_ratio=0.5),
            LaggedStartMap(GrowArrow, arrows, lag_ratio=0.5),
            LaggedStartMap(MoveToTarget, terms, lag_ratio=0.5),
        )
        self.wait()


class SendingInstructions(InteractiveScene):
    def construct(self):
        # Test
        arrow_group = ArrowGroup()
        arrows = VGroup(random.choice(arrow_group).copy() for n in range(25))
        radius = 3
        arrows.move_to(radius * RIGHT)

        def update_arrow(arrow):
            r = get_norm(arrow.get_center())
            op = clip(inverse_interpolate(radius, 0.8 * radius, r), 0, 1)
            arrow.set_fill(opacity=op)

        for arrow in arrows:
            arrow.add_updater(update_arrow)

        self.play(
            LaggedStart(
                (arrow.animate.shift(2 * radius * LEFT).set_anim_args(path_arc=120 * DEG)
                for arrow in arrows),
                lag_ratio=0.1,
                run_time=12
            )
        )
        self.wait()


class UnevenDistribution(InteractiveScene):
    def construct(self):
        # Test
        dists = [
            [0.25] * 4,
            [0.5, 0.25, 0.125, 0.125]
        ]
        full_width = 6
        stacks = VGroup(
            StackedProbDistribution(dist, width=full_width, height=0.5, labels=ArrowGroup())
            for dist in dists
        )
        decs = VGroup(
            self.get_bar_dec(bar, full_width)
            for bar in stacks[0].bars
        )

        self.add(stacks[0], decs)
        self.wait()
        self.play(Transform(*stacks), run_time=2)
        self.wait()

    def get_bar_dec(self, bar, full_width):
        dec = DecimalNumber(0, font_size=24, num_decimal_places=3)
        dec.add_updater(lambda m: m.set_value(bar.get_width() / full_width))
        dec.add_updater(lambda m: m.next_to(bar, DOWN, SMALL_BUFF))
        return dec


class StreamOfBits(InteractiveScene):
    def construct(self):
        # Test
        radius = 5
        bits = int_to_bit_string(n_bits=50)

        def update_bit(bit):
            r = get_norm(bit.get_center())
            op = inverse_interpolate(radius, 0.8 * radius, r)
            bit.set_fill(opacity=op)

        for bit in bits:
            bit.move_to(radius * RIGHT)
            bit.add_updater(update_bit)

        self.play(
            bits.animate.shift(2 * radius * LEFT).set_anim_args(lag_ratio=0.05, rate_func=bezier([0, 0, 1, 1])),
            run_time=20
        )
        self.wait()


class EncodingQuestion(InteractiveScene):
    def construct(self):
        # Question
        question = Text("Most Efficient Encoding?", font_size=90)
        question.to_edge(UP, buff=0.75)

        arrow_group = ArrowGroup()
        arrow_group.set_fill(YELLOW)
        n_instructions = 24
        indices = random.choices(list(range(4)), weights=[0.5, 0.25, 0.125, 0.125], k=n_instructions)
        arrows = VGroup(
            arrow_group[idx].copy().move_to(0.75 * n * RIGHT)
            for n, idx in enumerate(indices)
        )
        arrows.set_max_width(FRAME_WIDTH - 1)
        arrows.next_to(question, DOWN, MED_LARGE_BUFF)

        self.play(
            FadeIn(arrows, shift=2 * LEFT, path_arc=30 * DEG, lag_ratio=0.05),
            run_time=2
        )

        # Encode
        naive_codewords = ["00", "01", "10", "11"]
        optimal_codewords = ["0", "10", "110", "111"]

        encoding = VGroup(
            bit_string_mobject(naive_codewords[idx])
            for idx in indices
        )
        buff = get_norm(encoding[0][0].get_right() - encoding[0][1].get_left())
        encoding.arrange(RIGHT, buff=buff)
        encoding.match_width(arrows)
        encoding.next_to(arrows, DOWN, MED_LARGE_BUFF)

        self.play(
            Write(question),
            LaggedStart(
                (TransformFromCopy(arrow, bits)
                for arrow, bits in zip(arrows, encoding)),
                lag_ratio=0.1,
            ),
            run_time=2
        )
        self.wait()

        # Squish
        rect = SurroundingRectangle(encoding, buff=0)
        rect.set_opacity(0)
        full_width = rect.get_width()
        n_bits = 2 * n_instructions

        outer_arrows = VGroup(
            Vector(v, thickness=8)
            for v in [RIGHT, LEFT]
        )
        outer_arrows.set_fill(RED)
        outer_arrows[0].add_updater(lambda m: m.next_to(rect, LEFT, SMALL_BUFF))
        outer_arrows[1].add_updater(lambda m: m.next_to(rect, RIGHT, SMALL_BUFF))

        def get_new_bit_string():
            new_n_bits = int((rect.get_width() / full_width) * n_bits)
            np.random.seed(new_n_bits)
            new_bits = int_to_bit_string(new_n_bits)
            new_bits.match_height(encoding)
            new_bits.move_to(rect)
            return new_bits

        new_bits = always_redraw(get_new_bit_string)

        self.remove(encoding)
        self.add(new_bits)
        rect.shift(RIGHT)
        self.play(FadeIn(outer_arrows, run_time=0.5))
        self.play(
            rect.animate.stretch(0.6, 0, about_edge=LEFT),
            run_time=5
        )
        self.wait()
        self.play(FadeOut(outer_arrows))
        self.wait()


class DrawingDividingLines(InteractiveScene):
    def construct(self):
        # Test
        x_values = list(np.arange(-8, 8, 0.5))
        last_lines = VGroup()
        for n in range(5):
            lines = VGroup(Line(UP, DOWN).set_height(0.5).set_stroke(WHITE, 2))
            lines.set_x(-5)
            for k in range(20):
                lines.add(lines[-1].copy().shift(0.35 * random.choice([1, 2, 3]) * RIGHT))

            if n == 0:
                self.play(ShowCreation(lines, lag_ratio=0.05, run_time=1))
            else:
                self.play(ReplacementTransform(last_lines, lines))
            self.wait()
            last_lines = lines


class PrefixFreeCodeName(InteractiveScene):
    def construct(self):
        # Test
        phrase = Text("Prefix-free code", font_size=90)
        phrase.to_edge(UP, buff=LARGE_BUFF)
        words = VGroup(phrase["Prefix"], phrase["-free"], phrase["code"])

        self.play(ShowIncreasingSubsets(words), run_time=2)
        self.wait()
        self.play(
            words[1][0].animate.space_out_submobjects(0.2, about_edge=DOWN).set_opacity(0),
            words[0::2].animate.arrange(RIGHT, buff=MED_SMALL_BUFF).move_to(words)
        )
        self.wait()
        self.play(FlashAround(VGroup(words[0], words[2]), time_width=1.5, run_time=2))
        self.wait()


class WeightedSum(InteractiveScene):
    def construct(self):
        lines = VGroup(
            Tex(R"\frac{1}{2} \cdot 1+\frac{1}{4} \cdot 2+\frac{1}{8} \cdot 3+\frac{1}{8} \cdot 3"),
            Tex(R"=1.75 \text{ bits}")
        )
        lines.arrange(DOWN, aligned_edge=RIGHT)
        lines.to_edge(RIGHT)
        lines.set_backstroke(BLACK, 5)
        self.add(lines)
        self.wait()

        # Test
        self.play(
            lines.animate.arrange(RIGHT, SMALL_BUFF).center().to_edge(UP).set_anim_args(path_arc=45 * DEG),
            run_time=2
        )
        self.wait()


class UltraCleverEncoding(InteractiveScene):
    def construct(self):
        # Test
        arrow_group = ArrowGroup()
        arrows = VGroup(
            arrow_group[i].copy()[0]
            for i in [1, 0, 2, 0, 2, 0, 1, 3, 1, 1, 0, 0, 0, 3, 2, 0]
        )
        arrows.set_fill(TEAL)
        arrows.arrange(RIGHT)

        lower_jumble = arrows.copy()
        lower_jumble.set_fill(opacity=0.5)
        for arrow in lower_jumble:
            arrow.scale(1.5)
            arrow.next_to(arrows, DOWN, MED_SMALL_BUFF)
            arrow.shift(np.random.uniform(-0.2, 0.2, 3))

        bits = int_to_bit_string(n_bits=len(arrows), font="CMU Serif")
        bits.space_out_submobjects(1.2)
        bits.next_to(arrows, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT).shift(SMALL_BUFF * LEFT)
        bits.set_color(TEAL)

        self.add(arrows)
        self.wait()
        self.play(
            TransformFromCopy(arrows, lower_jumble, path_arc=30 * DEG, lag_ratio=0.01)
        )
        self.play(VShowPassingFlash(lower_jumble.copy().set_fill(opacity=0).set_stroke(YELLOW, 3), time_width=1.5, run_time=3))
        self.play(ReplacementTransform(lower_jumble, bits, lag_ratio=0.01, path_arc=10 * DEG, run_time=2))
        self.wait()


class ReactToKeyExpression(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        morty.change_mode("pondering")
        self.remove(self.background)

        self.play(
            morty.change("surprised", 3 * UR),
            self.change_students("hooray", "sassy", "awe", look_at=3 * UR),
        )
        self.wait(2)
        self.play(morty.change("hooray"))
        self.wait()
        self.play(LaggedStartMap(FadeOut, self.pi_creatures, shift=DOWN, lag_ratio=0.1, run_time=1))
        self.wait()


class FlashLogP(InteractiveScene):
    def construct(self):
        # Test
        tex = Tex(R"-\log_2(p)")
        tex.set_fill(opacity=0)
        tex.set_stroke(YELLOW, 5, 0.25)
        flash_parts = VGroup(
            tex.copy().scale(1.02**n)
            for n in range(1, 20)
        )
        self.play(LaggedStartMap(VFadeInThenOut, flash_parts))
        self.wait()


class AnnotateLogP(InteractiveScene):
    def construct(self):
        # Add equation
        equation = Tex(R"-\log_2(p) = n")
        info = equation[R"-\log_2(p)"][0]
        info_rect = SurroundingRectangle(info, buff=SMALL_BUFF)
        info_rect.set_stroke(BLUE, 2)
        eq = equation["="][0]
        info_word = Text("Information")
        info_word.set_color(BLUE)
        info_word.match_width(info_rect)
        info_word.next_to(info_rect, DOWN, SMALL_BUFF)

        n_rect = SurroundingRectangle(equation["n"], buff=SMALL_BUFF)
        n_rect.match_height(info_rect, stretch=True).match_y(info_rect)
        n_rect.set_stroke(YELLOW, 2)
        n_bits = Text("# bits")
        n_bits.next_to(n_rect, UP, SMALL_BUFF, LEFT)
        n_bits.match_color(n_rect)

        self.add(equation)
        self.wait()
        self.play(
            ShowCreation(info_rect),
            FadeIn(info_word, lag_ratio=0.1),
        )
        self.wait()

        # Replace eq with ineq
        bang = Tex(R"!!")
        bang.set_color(RED)
        bang.next_to(eq, UP)
        ineq = Tex(R"\le")
        ineq.move_to(eq)

        self.play(
            eq.animate.set_color(RED),
            FadeIn(bang, 0.2 * UP)
        )
        self.wait()
        self.play(
            Transform(eq, ineq),
            FadeOut(bang)
        )
        self.play(
            FadeIn(n_rect),
            FadeIn(n_bits, lag_ratio=0.1),
        )
        self.wait()

        # Show averages
        n_bits.target = n_bits.generate_target()
        n_bits.target.next_to(ineq, RIGHT)
        overbars = VGroup(
            Line(mob.get_left(), mob.get_right()).next_to(mob, UP, SMALL_BUFF)
            for mob in [info, n_bits.target]
        )
        overbars.set_stroke(WHITE, 4)

        self.play(
            FadeOut(info_rect),
            FadeOut(info_word),
            MoveToTarget(n_bits),
            FadeOut(n_rect),
            FadeOut(equation["n"]),
        )
        self.play(
            ShowCreation(overbars, lag_ratio=0)
        )
        self.wait()


class AlternateLogExpressions(InteractiveScene):
    def construct(self):
        kw = dict(
            t2c={
                R"-\log_2": BLUE,
                R"\log_2": BLUE,
                R"\log_{1/2}": BLUE,
                "p": TEAL
            }
        )
        terms = VGroup(
            Tex(R"-\log_2(p)", **kw),
            Tex(R"= \log_2(1 / p)", **kw),
            Tex(R"= \log_{1/2}(p)", **kw),
        )
        terms.arrange(RIGHT)

        # Test
        self.add(terms[0])
        self.wait()
        for idx in [0, 1]:
            self.play(TransformMatchingStrings(terms[idx].copy(), terms[idx + 1], run_time=1, path_arc=30 * DEG))
            self.wait()


class ReactToModel(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph().flip()

        self.play(
            VFadeIn(randy),
            randy.change("sassy")
        )
        self.play(Blink(randy))
        self.wait()
        self.play(randy.change("hesitant", 3 * UP))
        self.play(Blink(randy))
        self.wait()
        self.play(randy.change("confused", 3 * UL))
        self.play(Blink(randy))
        self.wait()


class TwoMeaningsOfBits(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        meanings = VGroup(
            VGroup(
                Text("“Bits” as"),
                Tex(R"-\log_2(p)", t2c={R"-\log_2": BLUE, "p": TEAL}),
            ).arrange(RIGHT, MED_LARGE_BUFF),
            VGroup(
                Text("“Bits” as"),
                Text("# 1s and 0s")
            ).arrange(RIGHT, MED_LARGE_BUFF),
        )
        rects = Rectangle().get_grid(2, 1, buff=0).replace(self.frame, stretch=True)
        for meaning, rect in zip(meanings, rects):
            meaning.next_to(rect.get_corner(UL), DR)

        h_line = Line(LEFT, RIGHT).replace(rects, 0)

        frame.move_to(rects[1], UP)
        self.add(meanings[1])
        self.wait()
        self.play(LaggedStart(
            frame.animate.to_default_state(),
            FadeIn(h_line),
            FadeTransformPieces(meanings[1].copy(), meanings[0]),
            run_time=2
        ))
        self.wait()


class SimpleP(InteractiveScene):
    def construct(self):
        pass


class ThoughtBubbleWithLanguageQuestions(InteractiveScene):
    def construct(self):
        # Test
        contents = VGroup(
            Text("British vs.\nAmerican?"),
            Text("Slang?"),
            Text("Australian?"),
            Text("Jargon?"),
        )
        bubble = ThoughtBubble(contents)
        bubble.remove(contents)
        bubble[0][:3].flip(UR).next_to(bubble[0][3], LEFT).shift(0.5 * DOWN) 

        self.play(
            Write(bubble, run_time=2),
            FadeIn(contents[0], lag_ratio=0.1, time_span=(0.5, 2))
        )
        self.wait()
        for old, new in zip(contents, contents[1:]):
            self.play(LaggedStart(
                FadeIn(new, shift=0.2 * UP),
                FadeOut(old, shift=0.2 * UP),
                lag_ratio=0.3
            ))
            self.wait()


class ShowKeyFormulas(InteractiveScene):
    def construct(self):
        # Add formulas
        kw = dict(
            font_size=36,
            t2c={"p": TEAL, "q": BLUE, "p_i": TEAL, "q_i": BLUE}
        )
        formulas = VGroup(
            Tex(R"I = -\log_2(p)", **kw),
            Tex(R"H(p) = \sum_i p_i\big(-\log_2(p_i)\big)", **kw),
            Tex(R"H(p, q) = \sum_i p_i\big(-\log_2(q_i)\big)", **kw),
            Tex(R"D_{\text{KL}}(p || q) = \sum_i p_i\big(\log_2(p_i / q_i)\big)", **kw),
        )
        formulas.arrange(DOWN, buff=LARGE_BUFF)
        formulas.to_edge(RIGHT)

        self.play(LaggedStartMap(Write, formulas, lag_ratio=0.25, run_time=3))
        self.wait()

        # Move to corners
        rects = FullScreenFadeRectangle().get_grid(2, 2, buff=0)
        rects.set_height(FRAME_HEIGHT).move_to(ORIGIN)
        div_lines = VGroup(Line(UP, DOWN), Line(LEFT, RIGHT))
        div_lines.replace(rects, stretch=True)
        div_lines.set_stroke(WHITE, 2)

        for formula, rect in zip(formulas, rects):
            formula.target = formula.generate_target()
            formula.target.next_to(rect.get_top(), DOWN)

        self.play(MoveToTarget(formulas[0]))
        self.reveal_name(formulas[0], "Information")
        self.wait()
        self.play(
            LaggedStartMap(MoveToTarget, formulas[1:], lag_ratio=0.25, run_time=3),
            Write(div_lines),
        )
        self.wait()
        names = ["Entropy", "Cross-entropy", "KL Divergence"]
        for formula, name in zip(formulas[1:], names):
            self.reveal_name(formula, name)
        self.wait()

    def reveal_name(self, formula, name):
        lhs_str, rhs_str = formula.get_string().split("=")
        lhs = formula[lhs_str][0]
        rhs = formula["=" + rhs_str][0]
        rhs.target = rhs.generate_target()

        name_mob = Text(name, font_size=formula.font_size)
        name_mob.move_to(lhs, RIGHT)

        VGroup(name_mob, rhs.target).move_to(formula)
        self.play(
            FadeTransformPieces(lhs, name_mob),
            MoveToTarget(rhs)
        )


class EntropyRate(InteractiveScene):
    def construct(self):
        # Test
        name = Text("Entropy Rate: ")
        form = Tex(R"\lim _{n \rightarrow \infty} \frac{1}{n} H\left(X_1, X_2, \ldots X_n\right)", t2c={"H": BLUE})
        group = VGroup(name, form)
        group.arrange(RIGHT)
        group.to_edge(UP)

        pre_form = Tex(R"H(X)", t2c={"H": BLUE})
        pre_form.move_to(form, LEFT)
        pre_name = Text("Entropy:")
        pre_name.move_to(name, RIGHT)

        self.add(pre_name, pre_form)
        self.wait()
        self.play(
            TransformMatchingStrings(pre_name, name),
            TransformMatchingTex(pre_form, form),
            run_time=1
        )
        self.wait()


class EndScreen(SideScrollEndScreen):
    pass