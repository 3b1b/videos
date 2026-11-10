from manim_imports_ext import *
from _2026.cross_entropy.entropy import bit_string_mobject
from _2026.cross_entropy.entropy import int_to_bit_string


class HutterPrizeWinners(InteractiveScene):
    def construct(self):
        # Add title
        title = Text("Hutter Prize", font_size=90)
        subtitle = Text("Compress 1GB of Wikipedia")
        subtitle.set_width(title.get_width() + 1)
        subtitle.next_to(title, DOWN).set_color(BLUE)
        title_group = VGroup(title, subtitle)
        title_group.to_edge(UP)
        self.add(title_group)

        # Show compression
        text = "Wikipedia"
        bits = VGroup(
            int_to_bit_string(8, ord(c)).set_color(color)
            for c, color in zip(text, it.cycle([GREY_C, GREY_B]))
        )
        bits.arrange(RIGHT, buff=SMALL_BUFF)
        bits.set_width(FRAME_WIDTH - 1)
        bits.next_to(title_group, DOWN, buff=0.5)
        len_tracker = ValueTracker(len(bits.family_members_with_points()))

        def get_len():
            return int(len_tracker.get_value())

        arrows = VGroup(
            Vector(0.5 * v, fill_color=RED, thickness=5)
            for v in [RIGHT, LEFT]
        )

        def update_arrows(arrows):
            arrows[0].next_to(bits, LEFT, SMALL_BUFF)
            arrows[1].next_to(bits, RIGHT, SMALL_BUFF)

        arrows.add_updater(update_arrows)

        def update_bits(bits):
            new_len = get_len()
            if new_len == 8 * len(text):
                return
            random.seed(int(new_len))
            new_bits = int_to_bit_string(new_len)
            new_bits.match_height(bits)
            new_bits.move_to(bits)
            bits.set_submobjects(new_bits)

        self.play(FadeIn(bits))
        self.play(FadeIn(arrows))
        self.play(
            len_tracker.animate.set_value(15),
            UpdateFromFunc(bits, update_bits),
            run_time=4,
            rate_func=rush_from,
        )
        self.wait()
        self.play(FadeOut(arrows), FadeOut(bits))

        # Winner grid
        winners = VGroup(map(Text, [
            "Winners",
            "V. Ivanov",
            "D. Freelan",
            "I. Marcouch & K. Orav",
            "K. Orav & B. Knoll",
            "K. Orav",
            "S. Kumar",
        ]))
        sizes = VGroup(map(Text, [
            "Size",
            "100Mb",
            "108Mb",
            "109Mb",
            "110Mb",
            "112Mb",
            "114Mb",
        ]))
        prizes = VGroup(map(Text, [
            "Prize",
            "€37,300",
            "€5,240",
            "€5,050",
            "€7,950",
            "€6,911",
            "€5,187",
        ]))
        row_height = 0.75
        row = VGroup(
            Rectangle(group.get_width(), 0.75)
            for group in [winners, sizes, prizes]
        )
        row.arrange(RIGHT, buff=0)
        row.set_stroke(GREY_A, 1)
        table = row.get_grid(len(winners), 1, buff=0)
        table.set_fill(BLACK, 1)
        table[0].set_stroke(WHITE, 3)
        entries = VGroup()

        for row, winner, size, prize in zip(table, winners, sizes, prizes):
            entries.add(VGroup())
            for box, text in zip(row, [winner, size, prize]):
                text.scale(0.8)
                text.move_to(box)
                entries[-1].add(text)

        filled_table = VGroup(table, entries)
        filled_table.set_width(FRAME_WIDTH / 2 - 1)
        filled_table.next_to(title_group, DOWN, buff=1.5).to_edge(RIGHT)
        self.add(table, entries[0])
        for entry in entries[:0:-1]:
            self.play(FadeIn(entry, shift=1e-1 * UP, lag_ratio=5e-2))
        self.wait()
        self.play(filled_table.animate.scale(0.8, about_edge=UR))
        self.wait()

        # Highlight
        rects = VGroup(SurroundingRectangle(entry, buff=0.05) for entry in entries[1:])
        rects.set_stroke(YELLOW, 2)
        for rect in rects:
            rect.match_width(rects, stretch=True)
            rect.match_x(rects, LEFT)
        self.play(LaggedStartMap(ShowCreation, rects))
        self.wait()


class CompressedMessage(InteractiveScene):
    def construct(self):
        # Show ascii encoding
        message = Text("message", font_size=72)
        message.move_to(2 * UP)
        bits = VGroup(
            int_to_bit_string(8, ord(c)).set_color(color)
            for c, color in zip(message.get_text(), it.cycle([GREY_C, GREY_B]))
        )
        bits.arrange(RIGHT, buff=SMALL_BUFF)
        bits.set_width(FRAME_WIDTH - 1)
        bits.next_to(message, DOWN, buff=1.5)

        message.save_state()
        for letter, byte in zip(message, bits):
            letter.next_to(byte, UP, SMALL_BUFF)

        self.play(
            ShowIncreasingSubsets(message),
            ShowIncreasingSubsets(bits),
            run_time=1,
            rate_func=linear,
        )
        self.wait()

        # Compress down
        arrows = VGroup(
            Vector(0.5 * v, fill_color=RED, thickness=5)
            for v in [RIGHT, LEFT]
        )
        len_tracker = ValueTracker(len(bits.family_members_with_points()))

        def get_len():
            return int(len_tracker.get_value())

        compressed_expr = Text("compressed(message)", font_size=72)
        compressed_expr["compressed"].set_fill(TEAL)
        compressed_expr.next_to(bits, UP, buff=1.5)
        equals = Tex(R"=", font_size=72).rotate(90 * DEG)
        equals.move_to(midpoint(bits.get_top(), compressed_expr.get_bottom()))

        def update_arrows(arrows):
            arrows[0].next_to(bits, LEFT, SMALL_BUFF)
            arrows[1].next_to(bits, RIGHT, SMALL_BUFF)

        arrows.add_updater(update_arrows)

        def update_bits(bits):
            new_len = get_len()
            if new_len == 8 * len(message.get_text()):
                return
            random.seed(int(new_len))
            new_bits = int_to_bit_string(new_len)
            new_bits.match_height(bits)
            new_bits.move_to(bits)
            bits.set_submobjects(new_bits)

        brace = always_redraw(lambda: Brace(bits, DOWN))
        brace_label = always_redraw(lambda: brace.get_text(f"{get_len()} bits").set_color(YELLOW))

        fade_rect = BackgroundRectangle(VGroup(brace, brace_label), buff=SMALL_BUFF)
        fade_rect.set_fill(BLACK, 1)

        self.add(brace, brace_label)
        self.play(
            LaggedStart(
                ReplacementTransform(message, compressed_expr["message"][0]),
                VFadeIn(compressed_expr["compressed"][0]),
                FadeIn(compressed_expr["("][0]),
                FadeIn(compressed_expr[")"][0]),
                Write(equals),
                time_span=(0, 1),
            ),
            VFadeIn(arrows),
            UpdateFromFunc(bits, update_bits),
            FadeOut(fade_rect, time_span=(1, 2)),
            len_tracker.animate(run_time=3).set_value(20),
        )
        self.play(FadeOut(arrows))
        self.wait()

        # Show the length
        frame = self.frame
        len_func = Text("len()", font_size=72).set_color(YELLOW)
        len_func.move_to(compressed_expr).to_edge(LEFT)
        compressed_expr.target = compressed_expr.generate_target()
        compressed_expr.target.next_to(len_func["("], RIGHT, SMALL_BUFF)
        len_func[")"][0].next_to(compressed_expr.target, RIGHT, SMALL_BUFF)
        len_func.refresh_bounding_box()

        info_expr = Tex(R"-\log_2\left(P(\text{message})\right)", font_size=72)
        approx = Tex(R"\approx", font_size=72)
        approx.next_to(len_func, RIGHT)
        info_expr.next_to(approx, RIGHT)

        bits_group = VGroup(equals, brace_label, brace, bits)
        bits_group.clear_updaters()
        bits_group.target = bits_group.generate_target()
        bits_group.target[2].rotate(PI)
        bits_group.target.arrange(DOWN, SMALL_BUFF)
        bits_group.target.next_to(len_func, DOWN, MED_SMALL_BUFF)

        self.play(
            LaggedStart(
                MoveToTarget(compressed_expr),
                Write(len_func),
                MoveToTarget(bits_group, path_arc=30 * DEG),
                frame.animate.set_height(10, about_edge=UP).match_x(VGroup(info_expr, len_func)),
                lag_ratio=0.1
            ),
            AnimationGroup(
                Write(approx),
                FadeIn(info_expr[R"-\log_2\left(P("]),
                FadeIn(info_expr[R")\right)"]),
                time_span=(0.5, 1.5)
            ),
            FadeTransform(
                compressed_expr["message"][0].copy(),
                info_expr[R"\text{message}"][0],
                path_arc=30 * DEG,
                run_time=1.5,
            )
        )
        self.wait()

        # Graph information
        info_rect = SurroundingRectangle(info_expr)
        info_rect.set_stroke(BLUE, 2)
        info_word = Text("Information", font_size=60)
        info_word.set_color(BLUE)
        info_word.next_to(info_rect, DOWN, MED_SMALL_BUFF)

        comp_group = VGroup(len_func, compressed_expr, approx)
        comp_group.save_state()

        axes = Axes((0, 1, 1 / 16), (0, 10), width=15, height=8)
        axes.shift(info_expr.get_corner(UL) + 0.5 * LEFT - axes.y_axis.get_top())
        h_lines = VGroup(
            Line(axes.c2p(0, y), axes.c2p(1, y)).set_stroke(GREY, 1, 0.5)
            for y in range(1, 10)
        )
        axes.add(h_lines)
        bit_labels = VGroup(
            Text(f"{n} bits" if n > 1 else f"{n} bit").next_to(h_lines[n - 1], LEFT, MED_SMALL_BUFF)
            for n in range(10)
        )
        bit_labels.set_color(GREY)

        graph = axes.get_graph(lambda x: -math.log2(x), x_range=(1e-3, 1, 1e-2))
        graph.set_stroke(BLUE, 3)

        x_tracker = ValueTracker(0.5)
        get_x = x_tracker.get_value
        x_tip = ArrowTip(angle=90 * DEG)
        x_tip.set_color(TEAL)
        x_tip.add_updater(lambda m: m.move_to(axes.x_axis.n2p(get_x()), UP))
        x_label = Tex(R"P = 1 / 2")
        x_label.make_number_changeable("2").f_always.set_value(lambda: 1 / get_x())
        x_label.always.next_to(x_tip, DOWN, SMALL_BUFF)

        v_line = Line().set_stroke(TEAL, 1)
        h_line = Line().set_stroke(TEAL, 1)
        v_line.add_updater(lambda m: m.put_start_and_end_on(
            axes.c2p(get_x(), 0),
            axes.i2gp(get_x(), graph),
        ))
        h_line.add_updater(lambda m: m.put_start_and_end_on(
            axes.i2gp(get_x(), graph),
            axes.c2p(0, graph.underlying_function(get_x())),
        ))
        graph_dot = GlowDot(color=BLUE)
        graph_dot.add_updater(lambda m: m.move_to(axes.i2gp(get_x(), graph)))

        alt_def = Tex(R"= \log_{1/2}\left(P(\text{message})\right)", font_size=72)
        alt_def.next_to(info_expr, RIGHT, buff=0.2)

        self.play(ShowCreation(info_rect))
        self.play(
            frame.animate.reorient(0, 0, 0, (9.68, -2.34, 0.00), 10.80),
            Write(axes, lag_ratio=1e-2, time_span=(0.5, 2)),
            comp_group.animate.fade(0.5).scale(0.5).to_corner(UL),
            FadeOut(bits_group, scale=0.5, shift=LEFT),
            ShowCreation(graph, time_span=(1, 2)),
            VFadeIn(x_tip),
            VFadeIn(x_label),
            FadeOut(info_rect),
            FadeIn(info_word, 0.25 * DOWN),
            run_time=2
        )
        self.play(
            ShowCreation(h_line, suspend_mobject_updating=True),
            ShowCreation(v_line, suspend_mobject_updating=True),
            FadeIn(graph_dot),
            FadeIn(bit_labels[1]),
        )
        self.wait()
        self.play(TransformMatchingTex(info_expr.copy(), alt_def, path_arc=20 * DEG))
        self.wait()
        line_ghosts = VGroup()
        self.add(line_ghosts)
        max_bits_shown = 8
        for n in range(1, max_bits_shown):
            line_ghosts.add(VGroup(v_line, h_line).copy().clear_updaters())
            self.play(
                x_tracker.animate.set_value(get_x() / 2),
                FadeOut(bit_labels[n]),
                FadeIn(bit_labels[n + 1]),
            )
            self.wait()

        # Example equation
        example = VGroup(
            Tex(
                R"P = \left(\frac{1}{2}\right)^{k}",
                t2c={"k": BLUE, "p": TEAL}
            ),
            Tex(R"\Leftrightarrow").rotate(90 * DEG),
            TexText("$k$ bits of information", t2c={"k": BLUE})
        )
        example.arrange(DOWN)
        example.scale(1.5)
        example.move_to(graph, RIGHT)
        x_label.suspend_updating()

        self.play(LaggedStart(
            TransformFromCopy(x_label[-3:], example[0][R"\frac{1}{2}"][0]),
            TransformFromCopy(bit_labels[max_bits_shown][0], example[0]["k"]),
            FadeIn(example[0][R"\left("]),
            FadeIn(example[0][R"\right)"]),
            TransformFromCopy(x_label[0], example[0]["P"]),
            TransformFromCopy(x_label[1], example[0]["="]),
        ))
        self.play(FadeIn(example[1:], lag_ratio=0.1))
        self.wait()
        x_label.resume_updating()
        self.add(x_label)

        # Bring back result
        graph_group = Group(
            axes, bit_labels[max_bits_shown], graph, graph_dot, line_ghosts,
            x_tip, x_label, h_line, v_line, example,
        )

        self.play(
            frame.animate.reorient(0, 0, 0, (1.66, -2.08, 0.00), 11.75),
            Restore(comp_group),
            FadeOut(alt_def, RIGHT),
            graph_group.animate.scale(0.5).move_to(info_expr.get_corner(DL), UL).shift(0.25 * LEFT + 0.75 * DOWN),
            run_time=2
        )
        self.wait()

        # Examples
        texts = VGroup(
            Text("“Hello”"),
            Text("“Quiescent frogs”"),
        )
        texts.scale(2)
        arrows = Vector(2 * RIGHT, thickness=8).replicate(2)
        encodings = VGroup(
            int_to_bit_string(12),
            int_to_bit_string(100).arrange_in_grid(n_cols=16),
        )
        encodings.scale(1.75)
        for text, arrow, encoding in zip(texts, arrows, encodings):
            group = VGroup(text, arrow, encoding)
            group.arrange(RIGHT, buff=MED_LARGE_BUFF)
            group.move_to(frame).shift(1.5 * DOWN)

        graph_group.clear_updaters()
        self.play(LaggedStart(
            FadeOut(graph_group),
            FadeIn(texts[0]),
            GrowArrow(arrows[0]),
            ShowIncreasingSubsets(encodings[0]),
        ))
        self.wait()
        self.play(LaggedStart(
            FadeOut(texts[0], 0.25 * UP),
            FadeIn(texts[1], 0.25 * UP),
            ReplacementTransform(*arrows),
            FadeOut(encodings[0]),
            ShowIncreasingSubsets(encodings[1]),
        ))
        self.wait()

        # Increment the probability
        encoding = encodings[1]

        p, lp, rp = p_wrapper = Tex(R"P()", font_size=144)
        for vect, paren in zip([LEFT, RIGHT], [lp, rp]):
            paren.scale(2)
            paren.match_height(encoding, stretch=True)
            paren.next_to(encoding, vect, buff=0.1)
        p.next_to(lp, LEFT)

        prob = Tex(R"\left(\frac{1}{2}\right)^{100} = ", font_size=96)
        prob.next_to(p, LEFT, LARGE_BUFF).shift(0.2 * UP)
        prob["="].scale(2, about_edge=LEFT)
        exp = prob.make_number_changeable("100")
        exp.set_value(len(encoding))
        exp.set_color(YELLOW)
        new_bits = int_to_bit_string(10)
        new_bits.match_height(encoding[0])
        for n, bit in enumerate(new_bits):
            bit.match_y(encoding[-1])
            bit.match_x(encoding[-16 + n])

        self.play(LaggedStart(
            self.frame.animate.shift(0.5 * RIGHT),
            FadeTransformPieces(info_expr.copy()[R"P"], prob),
            FadeOut(texts[1], scale=0.5, shift=DL),
            FadeOut(arrows[1], scale=0.5, shift=DL),
            FadeIn(p_wrapper),
            lag_ratio=0.2
        ))
        self.wait()
        last_new_bit = VGroup()
        last_new_bit.save_state()
        for n in range(5):
            new_bit = new_bits[n]
            new_bit.save_state()
            new_bit.set_color(YELLOW)
            new_exp = Integer(exp.get_value() + 1)
            new_exp.replace(exp)
            new_exp.match_style(exp)
            self.play(LaggedStart(
                FadeIn(new_bits[n], 0.5 * UP),
                ReplacementTransform(exp, new_exp, time_span=(0, 0.5)),
                run_time=1,
                lag_ratio=0.2,
            ))
            exp = new_exp
            last_new_bit = new_bit
            self.wait(0.5)

        # Center
        fader = VGroup(prob, p_wrapper, encoding, new_bits[:5], exp)
        prob[5].set_opacity(0)
        self.play(
            FadeOut(fader, DOWN),
            self.frame.animate.scale(0.8).move_to(2 * UP + RIGHT),
            run_time=2
        )
        self.wait()

    def old_parts(self):
        # Rearrange (old)
        kw = dict(t2c={"I": BLUE, "p": TEAL})
        eq_lines = VGroup(
            Tex(R"\left(\frac{1}{2}\right)^{I} = p", **kw),
            Tex(R"2^{-I} = p", **kw),
            Tex(R"-I = \log_2(p)", **kw),
            Tex(R"I = -\log_2(p)", **kw),
        )
        eq_lines.scale(1.5)
        eq_lines.arrange(DOWN, buff=MED_LARGE_BUFF)
        eq_lines.move_to(axes, RIGHT)
        x_label.suspend_updating()
        def_rect = SurroundingRectangle(eq_lines[-1])
        def_rect.set_stroke(TEAL, 3)

        self.play(
            TransformFromCopy(x_label[-3:], eq_lines[0][R"\frac{1}{2}"][0]),
            TransformFromCopy(bit_labels[5][0], eq_lines[0]["I"]),
            FadeIn(eq_lines[0][R"\left("]),
            FadeIn(eq_lines[0][R"\right)"]),
            TransformFromCopy(x_label[0], eq_lines[0]["p"]),
            TransformFromCopy(x_label[1], eq_lines[0]["="]),
        )
        self.wait()
        kw = dict(run_time=1, path_arc=20 * DEG)
        for n in range(3):
            self.play(TransformMatchingTex(eq_lines[n].copy(), eq_lines[n + 1], **kw))
            self.wait(0.5)
        self.wait()
        self.play(ShowCreation(def_rect))

        # More and less likely
        brace.add_updater(lambda m: m.become(Brace(bits, UP)))
        brace_label.add_updater(lambda m: m.become(
            brace.get_text(f"{get_len()} bits", buff=MED_SMALL_BUFF).set_color(YELLOW)
        ))
        likely_words = VGroup(
            Text("More likely", font_size=96),
            Text("Less likely", font_size=96),
        )
        likely_words.next_to(new_x_axis, DOWN, buff=0.75)
        likely_words[1].align_to(new_x_axis, LEFT)

        self.play(
            len_tracker.animate.set_value(8),
            UpdateFromFunc(bits, update_bits),
            FadeIn(likely_words[0], 3 * RIGHT),
            x_tracker.animate.set_value(0.5),
            run_time=2
        )
        self.wait()
        self.play(
            len_tracker.animate.set_value(36),
            UpdateFromFunc(bits, update_bits),
            FadeTransform(likely_words[0], likely_words[1]),
            x_tracker.animate.set_value(0.01),
            run_time=2
        )
        self.wait()


class TimelineSinceShannon(InteractiveScene):
    def construct(self):
        # Add number line
        frame = self.frame
        timeline = NumberLine(
            (1920, 2030),
            width=30,
            tick_size=0.05,
            longer_tick_multiple=2,
            big_tick_spacing=10,
        )
        timeline.add_numbers(
            range(1920, 2030, 10),
            buff=MED_SMALL_BUFF,
            font_size=36,
            group_with_commas=False
        )
        timeline.set_y(-2)

        # Zoom in to 40's
        frame.set_height(15, about_point=timeline.get_center())
        self.add(timeline)
        self.play(
            frame.animate.center().set_height(8).match_x(timeline.n2p(1960)),
            run_time=2
        )
        self.wait()

        # Show characters
        frame.set_y(1.0)
        root = self.file_writer.get_output_file_rootname().parent.parent
        shannon = VideoMobject(Path(
            root.parent,
            'cross_entropy/Mitchell Animations/Renders/Misc Assets/ShannonLoopv01-MZ.mp4'
        ))
        riss_pasc = VideoMobject(Path(
            root, "Mitchell Animations/Rissannen-Pasco-Puppets/ArithmeticCodingPuppet-RissannenPasco_v2.mp4",
        ))
        for video, year in [(shannon, 1948), (riss_pasc, 1976)]:
            video.set_height(4)
            video.set_z_index(-1)
            video.next_to(timeline.n2p(year), UP, buff=0)
        riss_pasc.shift(0.25 * DOWN)
        names = VGroup(Text("Jorma\nRissanen"), Text("Richard\nPasco"))
        point = riss_pasc.get_top() + DOWN
        names[0].next_to(point, LEFT, buff=1.25)
        names[1].next_to(point, RIGHT, buff=1.25)

        self.add(shannon, riss_pasc)
        self.play(
            frame.animate.match_x(riss_pasc),
            run_time=2
        )
        riss_pasc.play_from(0)
        self.play(LaggedStart(
            Write(names[0]),
            Write(names[1]),
            lag_ratio=0.5
        ))
        self.wait()
        self.play(
            frame.animate.set_height(15, about_edge=DOWN),
            run_time=2
        )
        self.wait(7)


class SameWordDifferentContext(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Context", font_size=120)
        title.to_edge(UP)

        # Two phrases
        phrases = VGroup(
            Text("“Have a holly jolly Christmas”", font_size=72),
            Text("“3.14159265358Christmas”", font_size=72),
        )
        xmas = VGroup(phrase["Christmas"][0] for phrase in phrases)
        xmas.set_color(GREEN_B)

        phrases.arrange(DOWN, buff=1)

        self.add(title)
        self.add(phrases)


class WriteArithmeticCoding(InteractiveScene):
    def construct(self):
        title = Text("Arithmetic Coding")
        title.set_width(FRAME_WIDTH - 1)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()


class LinkArrows(InteractiveScene):
    def construct(self):
        # Test
        arrows = Vector(DOWN, thickness=6).replicate(3).arrange(RIGHT, buff=2)
        arrows.set_color(BLUE)
        arrows.to_edge(DOWN)
        self.play(LaggedStartMap(GrowArrow, arrows, lag_ratio=0.25, run_time=1))
        self.wait()


class PredictorToCompressor(InteractiveScene):
    def construct(self):
        # Transform
        pred, comp = words = VGroup(
            Text("Predictor").set_color("#A201FF"),
            Text("Compressor").set_color("#0FFDB7"),
        )
        root = self.file_writer.output_directory.parent.parent
        pred_loop, comp_loop = loops = Group(
            VideoMobject(str(root / "cross_entropy/Paul Assets/Loops/Prediction.mp4")),
            VideoMobject(str(root / "cross_entropy/Paul Assets/Loops/Comrpession.mp4")),
        )
        words.scale(1.5)
        for word, loop, edge in zip(words, loops, [LEFT, RIGHT]):
            loop.set_height(3)
            loop.move_to(edge * FRAME_WIDTH / 4)
            loop.to_edge(UP, buff=0.25)
            loop.set_z_index(-1)
            loop.add_updater(lambda m, dt: m.set_time((m.get_time() + dt) % m.get_duration()))
            word.move_to(loop).to_edge(UP, buff=MED_SMALL_BUFF)

        top_bar = FullScreenFadeRectangle()
        top_bar.set_fill(BLACK, 1)
        top_bar.set_height(1, about_edge=UP, stretch=True)

        arrow = Arrow(2 * LEFT, 2 * RIGHT, thickness=7).move_to(loops).shift(0.25 * DOWN)

        self.add(top_bar)
        self.play(
            FadeIn(pred, lag_ratio=0.1),
            FadeIn(pred_loop, lag_ratio=0.1),
        )
        self.wait()
        self.play(
            TransformFromCopy(pred, comp, path_arc=30 * DEG),
            GrowArrow(arrow),
            FadeIn(comp_loop),
        )
        self.wait(12)


class CompressionIsIntelligenceTitle(InteractiveScene):
    def construct(self):
        # Show title
        font_size = 90
        full_phrase = Text(
            "Compression is Intelligence",
            t2c={"Compression": TEAL, "Intelligence": YELLOW},
            font_size=font_size,
        )
        words = VGroup(full_phrase[s][0] for s in full_phrase.get_text().split(" "))
        words.space_out_submobjects(1.2)
        words.to_edge(UP)
        self.play(LaggedStart(
            (FadeIn(word, shift=0.5 * UP, scale=1.5)
            for word in words),
            lag_ratio=0.5
        ))
        self.wait()

        # Proposed definition
        comp, _is, _int = words
        quote_int = Text("“Intelligence”", font_size=font_size)
        quote_int.set_color(YELLOW)
        eq = Tex(R":=", font_size=font_size)
        trg_group = VGroup(quote_int, eq, comp.generate_target())
        trg_group.arrange(RIGHT, buff=MED_LARGE_BUFF)

        self.play(LaggedStart(
            MoveToTarget(comp, path_arc=45 * DEG),
            AnimationGroup(
                Transform(_int, quote_int["Intelligence"][0], path_arc=45 * DEG),
                FadeInFromPoint(quote_int["“"], _int.get_corner(UL)),
                FadeInFromPoint(quote_int["”"], _int.get_corner(UR)),
            ),
            FadeOut(_is, 0.5 * UP),
            run_time=2,
            lag_ratio=0.15
        ))
        self.play(Write(eq))
        self.wait()
        self.play(comp.animate.set_y(3))




class ThreeAspectsOfIntelligence(InteractiveScene):
    def construct(self):
        # Enumerate all three, progressively reveal
        title = Text("Aspects of Intelligence")
        subtitle = Text("(or at least, what I think I mean when I use the word)")
        pass


class PredictorFromCompressor(InteractiveScene):
    def construct(self):
        # Use compared predictions sizes to estimate a probability
        pass


class StokesTheoremCompression(InteractiveScene):
    def construct(self):
        pass


class CompareTwoModels(InteractiveScene):
    def construct(self):
        pass


class BalanceOfSpaceOnHarddrive(InteractiveScene):
    def construct(self):
        pass


class TrainingOnTheFly(InteractiveScene):
    def construct(self):
        pass