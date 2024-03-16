import torch
from manim_imports_ext import *
from _2024.transformers.helpers import *
from _2024.transformers.embedding import *


class HighLevelNetworkFlow(InteractiveScene):
    example_text = "To date, the cleverest thinker of all time was blank"
    use_words = False
    possible_next_tokens = [
        (" the", 0.0882),
        (" probably", 0.0437),
        (" John", 0.0404),
        (" Sir", 0.0366),
        (" Albert", 0.0363),
        (" Ber", 0.0331),
        (" a", 0.029),
        (" Isaac", 0.0201),
        (" undoubtedly", 0.0158),
        (" arguably", 0.0133),
        (" Im", 0.0116),
        (" Einstein", 0.0113),
        (" Ludwig", 0.0104),
    ]
    hide_block_labels = False

    def setup(self):
        super().setup()
        self.set_floor_plane("xz")
        self.camera.light_source.move_to([-10, 10, 30])
        self.layers = VGroup()
        self.blocks = Group()
        self.mlps = Group()

    def construct(self):
        frame = self.frame

        # Embedding
        self.show_initial_text_embedding()

        # Passing through some layers
        self.progress_through_attention_block(target_frame_x=-2)
        self.progress_through_mlp_block(
            show_one_by_one=True,
            sideview_orientation=(-78, -10, 0),
        )
        orientation = frame.get_euler_angles() / DEGREES
        mlp_kw = dict(sideview_orientation=orientation, final_orientation=orientation)
        att_kw = dict(target_orientation=orientation)
        self.progress_through_attention_block(target_frame_x=-3, **att_kw)
        self.progress_through_mlp_block(**mlp_kw)
        self.progress_through_attention_block(target_frame_x=-4, **att_kw)
        self.progress_through_mlp_block(**mlp_kw)

        # Show how it ends
        self.remove_mlps()
        self.mention_repetitions()
        self.focus_on_last_layer()
        self.show_unembedding()

    def get_embedding_array(
        self,
        shape=(10, 9),
        height=4,
        dots_index=-4,
        buff_ratio=0.4,
        bracket_color=GREY_B,
        backstroke_width=3,
        add_background_rectangle=False,
    ):
        result = VGroup()
        embeddings = VGroup(*(
            NumericEmbedding(length=shape[1])
            for n in range(shape[0])
        ))
        embeddings.set_height(height)
        buff = buff_ratio * embeddings[0].get_width()
        embeddings.arrange(RIGHT, buff=buff)

        # Background rectangle
        if add_background_rectangle:
            for embedding in embeddings:
                if not isinstance(embedding, NumericEmbedding):
                    continue
                embedding.add_background_rectangle()

        # Add brackets
        brackets = Tex("".join((
            R"\left[\begin{array}{c}",
            *(shape[0] // 3) * [R"\quad \\"],
            R"\end{array}\right]",
        )))
        brackets.set_height(1.1 * embeddings.get_height())
        lb = brackets[:len(brackets) // 2]
        rb = brackets[len(brackets) // 2:]
        lb.next_to(embeddings, LEFT, buff=-0.0 * buff)
        rb.next_to(embeddings, RIGHT, buff=-0.0 * buff)
        brackets.set_fill(bracket_color)

        # Assemble result
        dots = VGroup()
        result = VGroup(embeddings, dots, brackets)
        result.embeddings = embeddings
        result.dots = dots
        result.brackets = brackets
        result.set_backstroke(BLACK, backstroke_width)

        if dots_index is not None:
            self.swap_embedding_for_dots(result, dots_index)

        return result

    def swap_embedding_for_dots(self, embedding_array, dots_index=-4):
        embeddings = embedding_array.embeddings
        to_replace = embeddings[dots_index]
        dots = Tex(R"\dots", font_size=60)
        dots.set_width(0.75 * to_replace.get_width())
        dots.move_to(to_replace)
        embeddings.remove(to_replace)
        embedding_array.dots.add(dots)
        return embedding_array

    def get_next_layer_array(self, embedding_array, z_buff=3.0):
        next_array = embedding_array.copy()
        embeddings = [part for part in next_array if isinstance(part, NumericEmbedding)]
        for embedding in embeddings:
            for entry in embedding.get_entries():
                entry.set_value(random.uniform(*embedding.value_range))
        next_array.shift(z_buff * OUT)
        return next_array

    def get_block(
        self,
        layer,
        depth=2.0,
        buff=1.0,
        size_buff=1.0,
        color=GREY_E,
        opacity=0.75,
        shading=(0.25, 0.1, 0.0),
        title="Attention",
        title_font_size=96,
        title_backstroke_width=5,
    ):
        # Block
        body = Cube(color=color, opacity=opacity)
        body.deactivate_depth_test()
        width, height = layer.get_shape()[:2]
        body.set_shape(width + size_buff, height + size_buff, depth)
        body.set_shading(0.5, 0.5, 0.0)
        body.next_to(layer, OUT, buff=buff)
        body.sort(lambda p: np.dot(p, [-1, 1, 1]))

        title = Text(title, font_size=title_font_size)
        title.set_backstroke(BLACK, title_backstroke_width)
        title.next_to(body, UP, buff=0.1)
        block = Group(body, title)
        block.body = body
        block.title = title
        if self.hide_block_labels:
            title.set_opacity(0)

        return block

    def show_initial_text_embedding(self, word_scale_factor=0.6, bump_first=False):
        # Mention next word prediction task
        phrase = Text(self.example_text)
        phrase.set_max_width(FRAME_WIDTH - 1)
        if self.use_words:
            words = break_into_words(phrase)
            rects = get_piece_rectangles(words)
        else:
            words = break_into_tokens(phrase)
            rects = get_piece_rectangles(
                words, leading_spaces=True, h_buff=0
            )

        words.remove(words[-1])
        q_marks = Text("???")
        rects[-1].set_color(YELLOW)
        q_marks.next_to(rects[-1], DOWN)

        big_rect = Rectangle()
        big_rect.replace(rects[:-1], stretch=True)
        big_rect.set_stroke(GREY_B, 2)
        arrow = Arrow(big_rect.get_top(), rects[-1].get_top(), path_arc=-120 * DEGREES)
        arrow.scale(0.5, about_edge=DR)

        self.play(ShowIncreasingSubsets(words, run_time=1))
        self.add(rects[-1])
        self.play(LaggedStart(
            FadeIn(big_rect),
            ShowCreation(arrow),
            Write(q_marks),
            lag_ratio=0.3,
        ))
        self.wait()
        self.play(
            FadeOut(big_rect),
            LaggedStart(*(
                DrawBorderThenFill(rect)
                for rect in rects[:-1]
            ), lag_ratio=0.02),
            LaggedStart(*(
                token.animate.match_color(rect)
                for token, rect in zip(words, rects)
            )),
            FadeOut(arrow)
        )
        self.wait()

        # Label the tokens
        token_label = Text("Tokens", font_size=72)
        token_label.to_edge(UP)
        arrows = VGroup(
            Arrow(token_label.get_bottom(), rect.get_top()).match_color(rect)
            for rect in rects[:-1]
        )

        self.play(FadeIn(token_label, UP))
        self.play(LaggedStartMap(VFadeInThenOut, arrows, lag_ratio=0.25, run_time=4))
        self.play(FadeOut(token_label, DOWN))


        # Show words into vectors
        layer = self.get_embedding_array(
            shape=(len(words), 10),
            dots_index=None,
        )
        vectors = layer.embeddings

        blocks = VGroup(*(VGroup(rect, token) for rect, token in zip(rects, words)))
        q_group = VGroup(rects[-1], q_marks)
        blocks.target = blocks.generate_target()
        for block, vector in zip(blocks.target, vectors):
            block.scale(word_scale_factor)
            block.next_to(layer, UP, buff=1.5)
            block.match_x(vector)

        arrows = VGroup(*(
            Arrow(block, vect, stroke_width=3)
            for block, vect in zip(blocks.target, vectors)
        ))
        word_to_index = dict(zip(self.example_text.split(" "), it.count()))
        self.swap_embedding_for_dots(layer, word_to_index.get("...", -4))

        if bump_first:
            blocks.target[0].next_to(blocks.target[1], LEFT, buff=0.1)

        self.play(
            self.frame.animate.move_to(1.0 * UP),
            MoveToTarget(blocks),
            q_group.animate.scale(word_scale_factor).next_to(blocks.target, RIGHT, aligned_edge=UP),
            LaggedStartMap(FadeIn, vectors, shift=0.5 * DOWN),
            LaggedStartMap(GrowFromCenter, arrows),
            Write(layer.dots)
        )
        self.play(Write(layer.brackets))
        self.wait()

        self.token_blocks = blocks
        self.token_arrows = arrows
        self.final_word_question = VGroup(rects[-1], q_marks)

        self.layers.add(layer)

    def progress_through_attention_block(
        self,
        target_orientation=(-40, -15, 0),
        target_frame_height=14,
        target_frame_x=-2,
        target_frame_y=0,
        attention_anim_run_time=5,
    ):
        layer = self.layers[-1]
        layer.save_state()
        block = self.get_block(layer, title="Attention")
        z_diff = block.get_z() - layer.get_z()
        block_opacity = block.body[0].get_opacity()
        block.body[0].set_opacity(0)
        new_layer = layer.copy()
        new_layer.match_z(block)

        self.frame.target = self.frame.generate_target()
        self.frame.target.reorient(*target_orientation)
        self.frame.target.set_height(target_frame_height)
        self.frame.target.set_x(target_frame_x)
        self.frame.target.set_y(target_frame_y)

        self.play(
            MoveToTarget(self.frame),
            LaggedStart(
                layer.animate.set_opacity(0.25),
                FadeIn(block),
                TransformFromCopy(layer, new_layer),
                lag_ratio=0.3
            ),
            run_time=2
        )
        self.play_simple_attention_animation(new_layer, run_time=attention_anim_run_time)

        # Take new layer out of block
        self.add(*block.body, block.title, new_layer)
        new_z = block.get_z() + z_diff
        self.play(
            block.body[0].animate.set_opacity(block_opacity),
            new_layer.animate.set_z(new_z),
            self.frame.animate.set_z(new_z),
            Restore(layer),
        )
        self.add(block, new_layer)

        self.blocks.add(block)
        self.layers.add(new_layer)

    def play_simple_attention_animation(self, layer, run_time=5):
        arc_groups = VGroup()
        for e1 in layer.embeddings:
            arc_group = VGroup()
            for e2 in layer.embeddings:
                sign = (-1)**int(e2.get_x() < e1.get_x())
                arc_group.add(Line(
                    e2.get_top(), e1.get_top(), 
                    path_arc=sign * PI / 3,
                    stroke_color=random_bright_color(hue_range=(0.1, 0.3)),
                    stroke_width=5 * random.random()**5,
                ))
            arc_group.shuffle()
            arc_groups.add(arc_group)

        self.play(
            LaggedStart(*(
                AnimationGroup(
                    LaggedStartMap(VShowPassingFlash, arc_group.copy(), time_width=2, lag_ratio=0.05),
                    LaggedStartMap(ShowCreationThenFadeOut, arc_group, lag_ratio=0.05),
                )
                for arc_group in arc_groups
            ), lag_ratio=0.25),
            LaggedStartMap(RandomizeMatrixEntries, layer.embeddings, lag_ratio=0.5),
            run_time=run_time
        )
        self.add(layer)

    def progress_through_mlp_block(
        self,
        n_neurons=20,
        depth=3.0,
        buff=1.0,
        dot_buff_ratio=0.2,
        neuron_color=GREY_C,
        neuron_shading=(0.25, 0.75, 0.2),
        sideview_orientation=(-60, -5, 0),
        final_orientation=(-51, -18, 0),
        show_one_by_one=False,
    ):
        # MLP Test
        layer = self.layers[-1]
        block = self.get_block(
            layer,
            depth=depth,
            buff=buff,
            size_buff=1.0,
            title="Multilayer\nPerceptron",
            title_font_size=72,
        )

        # New layer
        new_layer = self.get_next_layer_array(layer)
        new_layer.next_to(block.body, OUT, buff)

        # Neurons
        def get_neurons(points):
            neurons = DotCloud(points, radius=0.1)
            neurons.make_3d()
            neurons.set_glow_factor(0)
            neurons.set_shading(*neuron_shading)
            neurons.set_color(neuron_color)
            neurons.set_opacity(np.random.uniform(0.5, 1, len(points)))
            return neurons

        all_neuron_points = np.zeros((0, 3))
        neuron_clusters = Group()
        connections = VGroup()
        y_min = block.body.get_y(DOWN) + SMALL_BUFF
        y_max = block.body.get_y(UP) - SMALL_BUFF
        block_z = block.body.get_z()
        for embedding in layer.embeddings:
            l1_points = np.array([e.get_center() for e in embedding.get_columns()[0]])
            l3_points = l1_points.copy()
            l1_points[:, 2] = block_z - 0.4 * depth
            l3_points[:, 2] = block_z + 0.4 * depth
            x0, y0, z0 = embedding.get_center()
            globals().update(locals())
            l2_points = np.array([
                [x0, y, block_z]
                for y in np.linspace(y_min, y_max, n_neurons)
            ])
            new_points = np.vstack([l1_points, l2_points, l3_points])
            all_neuron_points = np.vstack([all_neuron_points, new_points])
            neuron_clusters.add(get_neurons(new_points))
            # Lines
            weights = VGroup(*(
                Line(
                    p1, p2,
                    buff=0.1,
                    stroke_width=3,
                    stroke_opacity=random.random(),
                    stroke_color=value_to_color(random.uniform(-10, 10))
                )
                for points1, points2 in [
                    [l1_points, l2_points],
                    [l2_points, l3_points],
                ]
                for p1, p2 in it.product(points1, points2)
                if random.random() < 0.2
            ))
            connections.add(weights)
        neurons = get_neurons(all_neuron_points)
        connections.apply_depth_test()
        connections.set_flat_stroke(False)

        # Flow through layer
        if show_one_by_one:
            networks = Group(*(Group(cluster, lines.copy()) for cluster, lines in zip(neuron_clusters, connections)))
            last_network = VectorizedPoint()
            emb_pairs = list(zip(layer.embeddings, new_layer.embeddings))
            index = 4
            block.title.rotate(PI / 2, DOWN)
            self.play(
                FadeIn(block.title, time_span=(0, 1)),
                self.frame.animate.reorient(*sideview_orientation),
                Write(networks[index][1]),
                FadeIn(networks[index][0]),
                FadeTransform(emb_pairs[index][0].copy(), emb_pairs[index][1]),
                run_time=3
            )
            lag_kw = dict(lag_ratio=0.5, run_time=9)
            self.play(
                LaggedStart(*(
                    FadeTransform(e1.copy(), e2)
                    for e1, e2 in [*emb_pairs[:index], *emb_pairs[index + 1:]]
                ), **lag_kw),
                LaggedStart(*(
                    Write(network[1])
                    for network in [*networks[:index], *networks[index + 1:]]
                ), **lag_kw),
                LaggedStart(*(
                    FadeIn(network[0])
                    for network in [*networks[:index], *networks[index + 1:]]
                ), **lag_kw),
                self.frame.animate.reorient(0, -37, 0, (-1.08, 2.29, 7.99), 12.27),
                block.title.animate.rotate(PI / 2, UP),
                run_time=9,
            )
            self.remove(networks)
            self.add(neurons, connections, block.body, block.title, new_layer)
            self.play(
                FadeIn(block.body),
                FadeIn(new_layer),
                FadeOut(new_layer.embeddings.copy()),
                self.frame.animate.reorient(*final_orientation).match_z(new_layer),
                run_time=2
            )
        else:
            self.play(
                FadeIn(block.title, time_span=(0, 1)),
                self.frame.animate.reorient(*sideview_orientation),
                FadeIn(neurons, time_span=(0, 1)),
                Write(connections, stroke_width=3),
                TransformFromCopy(layer, new_layer),
                run_time=3
            )
            self.add(block.body, block.title, new_layer)
            self.play(
                self.frame.animate.reorient(*final_orientation).match_z(new_layer),
                FadeIn(block.body),
                run_time=2,
            )

        # Aggregate
        self.mlps.add(Group(neurons, connections))
        self.blocks.add(block)
        self.layers.add(new_layer)

    def remove_mlps(self):
        self.remove(self.mlps)

    def mention_repetitions(self, depth=8):
        # Mention repetition
        frame = self.frame
        layer = self.layers[-1]
        block = self.blocks[-1].body

        thin_blocks = block.replicate(2)
        thin_blocks.set_depth(1.0, stretch=True)

        dots = Tex(".....", font_size=250)
        brace = Brace(dots, UP)
        brace_text = brace.get_text("Many\nrepetitions")
        rep_label = Group(dots, brace, brace_text)
        rep_label.set_width(depth)
        rep_label.rotate(PI / 2, DOWN)
        rep_label.next_to(layer, OUT, buff=3.0)
        rep_label.align_to(ORIGIN, DOWN)

        thin_blocks[0].align_to(brace, IN)
        thin_blocks[1].align_to(brace, OUT)
        dots.scale(0.5)
        VGroup(brace, brace_text).next_to(thin_blocks, UP)

        final_layer = self.get_next_layer_array(layer)
        final_layer.set_z(rep_label.get_z(OUT) + 1)
        final_layer.save_state()
        final_layer.become(layer)
        final_layer.set_opacity(0)

        self.play(
            frame.animate.reorient(-58, -12, 0, (-3.27, 0.98, 26.89), 25),
            FadeIn(thin_blocks, lag_ratio=0.1),
            GrowFromCenter(brace),
            Write(brace_text, time_span=(1, 2)),
            Write(dots),
            run_time=3
        )
        self.play(
            Restore(final_layer, run_time=2)
        )
        self.wait()

        self.rep_label = rep_label
        self.blocks.add(*thin_blocks)
        self.layers.add(final_layer)

    def focus_on_last_layer(self):
        # Last Layer
        layer = self.layers[-1]
        rect = BackgroundRectangle(layer)
        rect.set_fill(BLACK, 0.5)
        rect.scale(3)

        self.rep_label.target = self.rep_label.generate_target()
        self.rep_label.target[1:].set_opacity(0)
        self.rep_label.target[0].set_opacity(0.5)

        target_frame_center = layer.get_center() + 4 * RIGHT + UP

        self.add(self.rep_label, rect, layer)
        self.play(
            FadeIn(rect),
            MoveToTarget(self.rep_label),
            *map(FadeOut, [block.title for block in self.blocks if hasattr(block, "title")]),
            self.frame.animate.reorient(-3, -12, 0, target_frame_center, 12.00),
            run_time=3,
        )

    def show_unembedding(self):
        # Unembedding
        label_font_size = 30
        layer = self.layers[-1]
        last_embedding = layer.embeddings[-1]
        rect = SurroundingRectangle(last_embedding)
        rect.set_stroke(YELLOW, 3)

        words, dist = zip(*self.possible_next_tokens)
        bars = BarChart(dist).bars
        bars.rotate(-90 * DEGREES)
        bars.next_to(layer, RIGHT, buff=6.0)
        bars.set_y(2)
        for bar, word, value in zip(bars, words, dist):
            percentage = DecimalNumber(
                100 * value,
                num_decimal_places=2,
                unit="%",
                font_size=label_font_size,
            )
            percentage.next_to(bar, RIGHT)
            text = Text(word, font_size=label_font_size)
            text.next_to(bar, LEFT)
            bar.push_self_into_submobjects()
            bar.add(text, percentage)

        dots = Tex(R"\vdots")
        dots.next_to(bars[-1][1], DOWN)
        bars.add(dots)
        brace = Brace(bars, LEFT)

        arrow = Line()
        arrow.clear_points()
        arrow.start_new_path(rect.get_top())
        arrow.add_cubic_bezier_curve_to(
            rect.get_top() + UP,
            rect.get_top() + UR,
            rect.get_top() + 2 * RIGHT,
        )
        arrow.add_line_to(
            brace.get_left() + 0.1 * LEFT,
        )
        arrow.make_smooth()
        arrow.add_tip()
        arrow.set_color(YELLOW)


        self.play(ShowCreation(rect))
        self.play(
            ShowCreation(arrow),
            GrowFromCenter(brace),
            LaggedStartMap(FadeIn, bars, shift=DOWN)
        )
        self.wait()


class SimplifiedFlow(HighLevelNetworkFlow):
    example_text = "Word vectors will be updated to encode more than mere words"
    attention_anim_run_time = 1.0
    orientation = (-55, -19, 0)
    target_frame_x = -2

    def construct(self):
        self.show_initial_text_embedding(word_scale_factor=0.6)
        self.show_simple_flow(np.linspace(-2, -8, 5))

    def show_simple_flow(self, x_range, orientation=None):
        if orientation is None:
            orientation = self.orientation
        for x in x_range:
            self.progress_through_attention_block(
                target_orientation=orientation,
                target_frame_x=self.target_frame_x,
                attention_anim_run_time=self.attention_anim_run_time,
            )
            self.progress_through_mlp_block(
                sideview_orientation=orientation,
                final_orientation=orientation,
            )


class SimplifiedFlowAlternateAngle(SimplifiedFlow):
    example_text = "The goal of the network is to predict the next token"
    attention_anim_run_time = 1.0
    orientation = (-10, -20, 0)
    target_frame_x = 0
    hide_block_labels = True


class MentionContextSizeAndUnembedding(SimplifiedFlow):
    example_text = "Harry Potter was a highly unusual boy ... least favourite teacher, Professor Snape"
    attention_anim_run_time = 5.0
    use_words = True

    def construct(self):
        # Initial flow
        self.show_initial_text_embedding(word_scale_factor=0.5)
        self.cycle_through_embeddings()
        self.show_simple_flow(
            np.linspace(-2, -5, 2),
            orientation=(-50, -19, 0)
        )
        self.show_context_size()

        # Skip to the end
        self.remove_mlps()
        self.mention_repetitions()
        self.focus_on_last_layer()
        self.remove(*self.layers[:-1])

        # Discuss unembedding
        self.show_desired_output()
        self.show_unembedding_matrix()

    def cycle_through_embeddings(self):
        layer = self.layers[0]
        blocks = self.token_blocks
        arrows = self.token_arrows
        embeddings = VGroup(*layer.embeddings, *layer.dots)
        embeddings.sort(lambda p: p[0])
        rect_groups = VGroup(*(
            VGroup(*(
                BackgroundRectangle(mob, buff=0.2 if mob in arrows else 0.1)
                for mob in tup
            ))
            for tup in zip(blocks, arrows, embeddings)
        ))
        rect_groups.set_opacity(0)
        self.add(rect_groups)

        for index in range(len(rect_groups)):
            rect_groups.generate_target()
            rect_groups.target.set_opacity(0.75)
            rect_groups.target[index].set_opacity(0)
            self.play(MoveToTarget(rect_groups))
        self.play(FadeOut(rect_groups))

    def show_context_size(self):
        # Zoom in
        frame = self.frame
        frame.save_state()
        block_titles = VGroup(*(block.title for block in self.blocks))
        layer = self.layers[-1]

        self.play(
            frame.animate.reorient(-27, -18, 0, (-0.84, -0.36, 18.15), 9.00),
            FadeOut(block_titles, lag_ratio=0.01),
            run_time=2,
        )

        # Show size
        font_size = 72
        brace = Brace(layer.embeddings, DOWN)
        label = brace.get_text("Context size", font_size=font_size)
        label.generate_target()
        rhs = Tex("= 2{,}048", font_size=font_size)
        full_label = VGroup(label.target, rhs)
        full_label.arrange(RIGHT, aligned_edge=UP)
        full_label.next_to(brace, DOWN)

        dim_brace = Brace(layer.embeddings, RIGHT)
        dim_label = dim_brace.get_text("12,288", font_size=font_size)

        self.play(
            GrowFromCenter(brace),
            FadeIn(label, 0.5 * DOWN),
        )
        self.wait()
        self.play(
            MoveToTarget(label),
            Write(rhs)
        )
        self.wait()
        self.play(
            FadeOut(layer.brackets),
            GrowFromCenter(dim_brace),
            FadeIn(dim_label, 0.5 * LEFT),
        )
        self.wait(3)

        # Return
        self.play(
            Restore(frame, run_time=3),
            FadeIn(block_titles, time_span=(2, 3), lag_ratio=0.01),
            FadeIn(layer.brackets),
            LaggedStartMap(FadeOut, VGroup(
                brace, label, rhs, dim_brace, dim_label
            ), shift=DOWN)
        )
        for layer, block in zip(self.layers, self.blocks):
            self.add(layer, block)
        self.add(self.layers[-1])

    def show_desired_output(self):
        # Show phrase
        frame = self.frame
        phrase = Text(self.example_text)
        phrase.set_max_width(FRAME_WIDTH - 1)
        phrase.to_corner(UL)
        phrase.fix_in_frame()
        last_word = phrase[self.example_text.split(" ")[-1]][0]
        last_word.set_opacity(0)
        rect = SurroundingRectangle(last_word, buff=0.1)
        rect.set_stroke(YELLOW, 2)
        rect.set_fill(YELLOW, 0.5)
        rect.align_to(last_word, LEFT)
        q_marks = Text("???")
        q_marks.next_to(rect, DOWN)
        q_group = VGroup(rect, q_marks)
        q_group.fix_in_frame()

        self.play(FadeIn(phrase, lag_ratio=0.1, run_time=2))
        self.play(FadeIn(q_group, 0.2 * RIGHT))
        self.wait()

        # Get all possible next words
        layer = self.layers[-1]
        word_strs = [
            "...",
            "Snake",
            "Snape",
            "Snare",
            "...",
            "Treks",
            "Trelawney",
            "Trellis",
            "...",
            "Quirky",
            "Quirrell",
            "Quirt",
            "..."
        ]
        words = VGroup()
        dots = VGroup()
        for word_str in word_strs:
            word = Text(word_str)
            words.add(word)
            if word_str == "...":
                dots.add(word)
                word.rotate(PI / 2)
        words.arrange(DOWN, aligned_edge=LEFT)
        dots.shift(0.25 * RIGHT)
        words.set_max_height(5.0)
        words.to_edge(DOWN, buff=1.0)
        words.to_edge(RIGHT, buff=0.1)
        words.fix_in_frame()

        # Add probability bars
        values = [0, 0.78, 0, 0, 0.16, 0, 0, 0.06, 0]
        bars = VGroup()
        probs = VGroup()
        value_iter = iter(values)
        bar_height = 0.8 * words[1].get_height()
        for word in words:
            if word in dots:
                continue
            value = next(value_iter)
            prob = DecimalNumber(value, font_size=24)
            bar = Rectangle(10 * value * bar_height, bar_height)
            bar.next_to(word, LEFT)
            hsl = list(Color(BLUE).get_hsl())
            hsl[0] = interpolate(0.5, 0.6, value)
            bar.set_fill(Color(hsl=hsl), 1)
            bar.set_stroke(BLUE, 1)
            prob.next_to(bar, LEFT)
            probs.add(prob)
            bars.add(bar)

        probs.fix_in_frame()
        bars.fix_in_frame()
        brace = Brace(VGroup(probs, dots), LEFT).fix_in_frame()
        arrow = Vector(RIGHT, stroke_width=8).fix_in_frame()
        arrow.set_color(YELLOW)
        arrow.next_to(brace, LEFT)

        # Show creation
        self.play(
            LaggedStartMap(FadeIn, words, shift=0.1 * DOWN),
            LaggedStartMap(FadeIn, bars),
            LaggedStartMap(FadeIn, probs, shift=0.2 * LEFT),
            GrowFromCenter(brace),
            GrowArrow(arrow),
        )
        self.wait()

        self.prob_group = VGroup(arrow, brace, words, bars, probs)
        self.phrase = VGroup(phrase, q_group)

    def show_unembedding_matrix(self, vector_index=-1):
        # Clear frame
        frame = self.frame
        prob_group = self.prob_group
        prob_group.save_state()
        prob_group.generate_target()
        prob_group.target.scale(0.5, about_edge=DR)
        prob_group.target.to_corner(DR)
        prob_group.target[0].set_opacity(0)

        self.play(
            FadeOut(self.phrase, UP),
            MoveToTarget(prob_group),
            frame.animate.reorient(0, 1, 0, (4.98, 3.34, 30.08), 12.00),
            run_time=2,
        )

        # Show the weight matrix
        layer = self.layers[-1]
        vector = layer.embeddings[vector_index].copy()
        matrix = WeightMatrix(
            shape=(15, vector.shape[0]),
            ellipses_row=8,
        )
        matrix.set_height(6)
        matrix.next_to(vector, UP, aligned_edge=RIGHT, buff=1.0)
        matrix.shift(LEFT)
        last_vector_rect = rect = SurroundingRectangle(vector, buff=0.1)
        rect.set_stroke(YELLOW, 3)
        vector.generate_target()
        vector.target.next_to(matrix, RIGHT)
        last_vect_arrow = Arrow(
            rect.get_top(), vector.target.get_bottom(),
        )
        last_vect_arrow.set_color(YELLOW)

        self.play(
            FadeIn(matrix, scale=0.8, shift=DR),
            *map(FadeOut, [self.token_blocks, self.token_arrows, self.final_word_question]),
        )
        self.play(ShowCreation(rect))
        self.play(MoveToTarget(vector), ShowCreation(last_vect_arrow))

        # Show matrix vector product
        eq, rhs = show_matrix_vector_product(self, matrix, vector)

        # Count values
        brace = Brace(rhs, RIGHT)
        brace_label = brace.get_tex(R"\sim 50k \text{ values}", font_size=60, buff=0.25)

        self.play(
            GrowFromCenter(brace),
            FadeIn(brace_label, lag_ratio=0.1),
        )
        brace_group = VGroup(brace, brace_label)

        # Show words
        word_strs = [
            "aah",
            "aardvark",
            "aardwolf",
            "aargh",
            "ab",
            "aback",
            "abacterial",
            "abacus",
            "...",
            "zygote",
            "zygotic",
            "zyme",
            "zymogen",
            "zymosis",
            "zzz",
        ]
        words = VGroup(*map(Text, word_strs))
        words[word_strs.index("...")].rotate(PI / 2)
        for word, entry in zip(words, rhs):
            word.set_max_height(entry.get_height())
            word.next_to(rhs, RIGHT, buff=0.25)
            word.match_y(entry)

        self.play(
            LaggedStartMap(FadeIn, words, shift=0.5 * RIGHT),
            brace_group.animate.next_to(words, RIGHT),
        )
        self.wait()

        # Mention softmax
        big_rect = SurroundingRectangle(VGroup(rhs, words), buff=0.25)
        big_rect.set_fill(TEAL, 0.1)
        big_rect.set_stroke(TEAL, 3)
        softmax_arrow = Vector(2.2 * RIGHT, stroke_width=8)
        softmax_arrow.next_to(big_rect, RIGHT, buff=0.1)
        softmax_label = Text("softmax", font_size=48)
        softmax_label.next_to(softmax_arrow, UP, buff=0.35)

        prob_group.generate_target()
        prob_group.target[0].scale(0).move_to(prob_group)
        prob_group.target.set_height(4)
        prob_group.target.to_edge(UP, buff=0.25).to_edge(RIGHT, buff=0.1)

        self.play(
            LaggedStart(
                FadeOut(brace_group),
                DrawBorderThenFill(big_rect),
                ShowCreation(softmax_arrow),
                FadeIn(softmax_label, lag_ratio=0.1),
            ),
            MoveToTarget(prob_group, time_span=(1, 2))
        )
        prob_group.unfix_from_frame()
        prob_group.match_height(rhs)
        prob_group.next_to(softmax_arrow, RIGHT, buff=0.25)
        self.wait()

        # Ask about other vectors
        rects = VGroup(*(
            SurroundingRectangle(emb)
            for emb in layer.embeddings[:-1]
        ))
        rects.set_stroke(PINK, 3)
        rects.set_fill(PINK, 0.25)
        question = Text("What about these?", font_size=90)
        question.next_to(layer, DOWN, buff=2.0)
        question_arrows = VGroup()
        for rect in rects:
            question_arrows.add(Arrow(
                question.get_top(), rect.get_bottom(),
                stroke_color=rect.get_color()
            ))

        self.play(
            frame.animate.reorient(-1, 0, 0, (1.69, 1.57, 29.79), 15.88),
            Write(question),
            run_time=2,
        )

        last_rect = VGroup()
        for rect, arrow in zip(rects, question_arrows):
            self.play(
                FadeOut(last_rect),
                FadeIn(rect),
                FadeIn(arrow),
            )
            last_rect = rect
        self.play(FadeOut(last_rect))
        self.wait()

        # Move back
        self.play(
            FadeOut(question_arrows, lag_ratio=0.1),
            FadeOut(question, lag_ratio=0.1),
            frame.animate.reorient(0, 1, 0, (4.27, 3.46, 29.83), 12.86),
            run_time=2,
        )
        # self.play(
        #     LaggedStartMap(FadeOut, VGroup(
        #         *question_arrows, question,
        #         matrix, vector, eq, rhs,
        #         big_rect, softmax_arrow, prob_group, words, softmax_label,
        #         last_vector_rect, last_vect_arrow,
        #     )),
        #     FadeOut(question_arrows, lag_ratio=0.1),
        #     FadeOut(question, lag_ratio=0.1),
        #     frame.animate.reorient(-2, 0, 0, (2.72, 1.82, 29.44), 10.31),
        #     run_time=2,
        # )
        self.wait()

        # Name the unembedding matrix
        matrix_rect = SurroundingRectangle(matrix, buff=0.1)
        matrix_rect.set_stroke(BLUE, 3)
        name = Text("Unembedding\nmatrix", font_size=60)
        label = Tex("W_U", font_size=90)
        name.next_to(matrix_rect, LEFT)
        label.next_to(name, DOWN)

        self.play(
            Write(matrix_rect, stroke_width=5, stroke_color=BLUE),
            Write(name, run_time=2),
            frame.animate.reorient(0, 1, 0, (4.24, 3.15, 29.81), 13.23),
        )
        self.wait()
        self.play(
            FadeIn(label, DOWN),
            name.animate.next_to(label, UP, buff=1)
        )
        self.wait()

        # Data flying
        data_modifying_matrix(self, matrix)
        self.wait()

        # Count parameters
        label.set_backstroke(BLACK, 6)
        entries = VGroup(*matrix.elements, *matrix.ellipses)
        row_rects = VGroup(*map(SurroundingRectangle, matrix.get_rows()))
        col_rects = VGroup(*map(SurroundingRectangle, matrix.get_columns()))
        VGroup(row_rects, col_rects).set_stroke(GREY, 1).set_fill(GREY_B, 0.5)

        left_brace = Brace(matrix, LEFT)
        top_brace = Brace(matrix, UP)
        vocab_count = Integer(50257, font_size=90)
        vocab_count.next_to(left_brace, LEFT)
        dim_count = Integer(12288, font_size=90)
        dim_count.next_to(top_brace, UP)

        top_equation = VGroup(
            Text("Total parameters = "),
            Integer(vocab_count.get_value()),
            Tex(R"\times"),
            Integer(dim_count.get_value()),
            Tex("="),
            Integer(vocab_count.get_value() * dim_count.get_value()).set_color(YELLOW),
        )
        top_equation.arrange(RIGHT)
        top_equation.scale(2)
        top_equation.next_to(dim_count, UP, buff=1)
        top_equation.align_to(vocab_count, LEFT)

        self.play(
            label.animate.move_to(matrix),
            FadeOut(name, lag_ratio=0.1),
            entries.animate.set_fill(opacity=0.5, border_width=0),
        )
        self.add(row_rects, label)
        self.play(
            GrowFromCenter(left_brace, time_span=(0, 1)),
            CountInFrom(vocab_count),
            LaggedStartMap(VFadeInThenOut, row_rects, lag_ratio=0.2),
            run_time=2.5,
        )
        self.add(col_rects, label)
        self.play(
            GrowFromCenter(top_brace, time_span=(0, 1)),
            CountInFrom(dim_count),
            LaggedStartMap(VFadeInThenOut, col_rects, lag_ratio=0.2),
            frame.animate.reorient(0, 0, 0, (5.66, 4.08, 29.89), 14.32),
            run_time=2.5,
        )
        self.wait()
        self.play(LaggedStart(
            Write(top_equation[0:6:2]),
            TransformFromCopy(vocab_count, top_equation[1]),
            TransformFromCopy(dim_count, top_equation[3]),
            frame.animate.reorient(0, -1, 0, (4.15, 5.07, 29.75), 17.21),
            run_time=3
        ))
        self.play(
            FadeTransform(top_equation[1:4].copy(), top_equation[-1])
        )
        self.wait()


class TextPassageIntro(InteractiveScene):
    example_text = MentionContextSizeAndUnembedding.example_text

    def construct(self):
        # Read in passage
        passage_str = Path(DATA_DIR, "harry_potter_3.txt").read_text()
        passage_str = passage_str.replace("\n", "\n\\\\")
        passage = TexText(passage_str, alignment="", additional_preamble=R"\tiny")
        passage.set_height(FRAME_HEIGHT - 1)
        passage[-len("Snape"):].set_opacity(0)

        # Initial surroundings
        frame = self.frame
        lh, rh = (1176, 1540)
        section = passage[lh:rh].copy()
        section.save_state()
        section.set_width(FRAME_WIDTH - 1)
        section.center()

        word_lh, word_rh = (150, 155)
        word = section[word_lh:word_rh].copy()
        word.save_state()
        word.set_height(0.75).center()

        self.play(Write(word))
        self.wait()
        self.play(
            Restore(word, time_span=(0, 0.5), remover=True),
            ShowIncreasingSubsets(section),
            run_time=1
        )
        self.play(ContextAnimation(word, section))
        self.wait()
        self.play(
            Restore(section, remover=True),
            ShowIncreasingSubsets(VGroup(*passage[:lh], *passage[rh:]))
        )
        self.add(passage)
        self.play(ContextAnimation(
            passage[lh:rh][word_lh:word_rh],
            VGroup(
                *passage[0:11],
                *passage[211:217],
                # *passage[2366:2374],
            ),
            lag_ratio=0.01
        ))
        self.wait()

        # Compress
        start, end = self.example_text.split(" ... ")
        short_text = Text(self.example_text)
        short_text["Snape"].set_opacity(0)
        short_text.set_max_width(FRAME_WIDTH - 1)
        dots = short_text["..."][0]

        lh, rh = (31, 2474)
        self.play(
            FadeTransformPieces(
                passage[:lh],
                short_text[start][0],
            ),
            ReplacementTransform(passage[lh:rh], dots),
            FadeTransformPieces(
                passage[rh:],
                short_text[end][0],
            ),
            run_time=3
        )
        self.add(short_text)
        self.wait()