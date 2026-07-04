from manim_imports_ext import *
from _2026.cross_entropy.entropy import InformationOfLanguage
from _2026.cross_entropy.distribution import StackedProbDistribution
from _2026.cross_entropy.next_char import CHAR_ALPHABET
from _2026.cross_entropy.next_char import get_next_char_distribution
from _2026.cross_entropy.next_char import gpt2_predict_next_token
from _2026.cross_entropy.next_char import gpt2_token_probability
from _2024.transformers.helpers import MachineWithDials
from _2024.transformers.embedding import get_token_encoding
from _2024.transformers.embedding import break_into_tokens
from _2024.transformers.embedding import get_piece_rectangles


def get_training_text_examples():
    data_file = Path(__file__).parent / "text_data" / "training_data_examples.txt"
    return data_file.read_text().splitlines()


class LanguageModel(VGroup):
    def __init__(
        self,
        block_dims=(4, 2.5, 0.2),
        n_blocks=10,
        block_color=GREY_D,
        block_opacity=0.5,
        block_shading=(0.25, 0.5, 0.2),
        phi=0 * DEG,
        theta=25 * DEG,
        dial_config=dict(
            n_rows=8,
            n_cols=14,
        ),
    ):
        super().__init__()
        # Add blocks
        blocks = VGroup(
            VPrism(*block_dims)
            for n in range(n_blocks)
        )
        blocks.set_fill(block_color, block_opacity)
        blocks.set_stroke(WHITE, 3, 0.5)
        blocks.set_shading(*block_shading)
        blocks.arrange(OUT)
        blocks.move_to(ORIGIN, OUT)

        self.blocks = blocks
        self.add(blocks)

        # Add dials
        front_face = MachineWithDials(
            width=block_dims[0],
            height=block_dims[1],
            **dial_config,
        )
        front_face.box.set_opacity(0)
        front_face.move_to(blocks, OUT)

        self.front_face = front_face
        self.add(front_face)

        # Rotate
        self.rotate(phi, RIGHT, about_edge=OUT)
        self.rotate(theta, UP, about_edge=OUT)
        self.center()

    def forward_pass(self, lag_ratio=0.1, run_time=2):
        over_blocks = self.blocks.copy()
        over_blocks.set_fill(TEAL_E, 0.5)
        return self.blocks.animate.set_fill(TEAL_E).set_anim_args(
            rate_func=there_and_back, lag_ratio=lag_ratio, run_time=run_time
        )


class AmbientModelPredictions(InteractiveScene):
    def construct(self):
        # Add model
        frame = self.frame
        model = LanguageModel()
        model.set_height(2)
        self.add(model)

        # Test
        self.play(model.forward_pass(lag_ratio=0.01))


class NextTokenPredictions(InteractiveScene):
    def construct(self):
        # Set up tokens
        text = "Once upon a time, there was a tiny pi creature who lived between 3rd and 4th"
        text_mob = Text(text)
        text_mob.to_edge(LEFT, buff=MED_LARGE_BUFF)

        tokens = break_into_tokens(text_mob)
        rects = get_piece_rectangles(tokens, h_buff=0, leading_spaces=True)

        tokenizer = get_token_encoding()
        token_indices = tokenizer.encode(text)
        _, offsets = tokenizer.decode_with_offsets(token_indices)
        prefixes = [text[:offset] for offset in offsets[1:]]
        text_tokens = [text[o1:o2] for o1, o2 in zip(offsets, offsets[1:])]

        # Predictions
        frame = self.frame
        arrows = VGroup(
            Vector(2.5 * RIGHT, fill_color=TEAL, thickness=7).next_to(rect, RIGHT, SMALL_BUFF)
            for rect in rects
        )
        pred_mobs = VGroup(
            self.get_next_token_prediction_mob(prefix)
            for prefix in prefixes
        )
        for pred_mob, arrow in zip(pred_mobs, arrows):
            pred_mob.next_to(arrow, RIGHT)

        moving_arrow = arrows[0].copy()

        token_ghosts = VGroup(tokens.copy(), rects.copy()).fade(0.8)
        arrow_ghosts = arrows.copy().set_fill(opacity=0.25)

        frame.add_updater(lambda m: m.move_to(moving_arrow))

        last_pm = VGroup()
        for pm, arrow, token, rect, next_token in zip(pred_mobs, arrows, tokens, rects, tokens[1:]):
            pre_pm = pm.copy()
            for part in pre_pm:
                part.scale(1e-4).fade(1).move_to(arrow.get_top() + SMALL_BUFF * UP)
            self.play(
                FadeIn(token),
                FadeIn(rect),
                moving_arrow.animate.move_to(arrow),
                FadeOut(last_pm, shift=(token.get_width() + 1) * RIGHT),
            )
            self.play(
                TransformFromCopy(pre_pm, pm, lag_ratio=0.002, path_arc=-25 * DEG)
            )
            self.wait(0.5)
            last_pm = pm
        self.play(
            FadeOut(last_pm),
        )

        # Step back to highlight model
        frame.clear_updaters()
        model_placeholder = Square(side_length=0.4 * rects.get_width())
        model_placeholder.next_to(moving_arrow, UP, MED_LARGE_BUFF, RIGHT)
        token_rects_copy = VGroup(*tokens[:-2], *rects[:-2]).copy()
        token_rects_copy.sort(lambda p: p[0])
        token_rects_copy.target = token_rects_copy.generate_target()
        pred_mob = last_pm
        pred_mob.scale(2, about_edge=LEFT)
        pred_mob.save_state()

        for mob in [*token_rects_copy.target, *pred_mob]:
            mob.move_to(model_placeholder).fade(1)

        self.play(frame.animate.reorient(0, 0, 0, (5.53, 1.67, 0.0), 14.22))
        self.wait()
        self.play(LaggedStart(
            MoveToTarget(token_rects_copy, path_arc=-90 * DEG, lag_ratio=0.01),
            Restore(pred_mob, lag_ratio=0.005),
            lag_ratio=0.7,
            run_time=2
        ))
        self.wait()

    def get_next_token_prediction_mob(
        self,
        text,
        bar_buff=0.3,
        font_size=40,
        n_shown=8,
        prob_1_width=3.0,
        prob_bar_height=0.4,
        prob_bar_colors=(BLUE_D, TEAL_D),
    ):
        tokens, probs = gpt2_predict_next_token(text, n_shown=n_shown)
        prob_bars = VGroup(
            Rectangle(prob * prob_1_width, prob_bar_height)
            for prob in probs
        )
        prob_bars.arrange(DOWN, buff=bar_buff, aligned_edge=LEFT)
        prob_bars.set_fill(WHITE, 1)
        prob_bars.set_submobject_colors_by_gradient(*prob_bar_colors)
        prob_bars.set_stroke(WHITE, 1)

        token_mobs = VGroup(
            Text(token, font_size=font_size).next_to(bar, LEFT, SMALL_BUFF)
            for bar, token in zip(prob_bars, tokens)
        )
        dots = Tex(R"\vdots")
        dots.next_to(token_mobs[-1], DR)

        percentages = VGroup(
            DecimalNumber(
                100 * p,
                unit="%",
                font_size=font_size,
                num_decimal_places=(1 if p > 0.01 else 2),
                fill_color=GREY_B
            ).next_to(prob_bar, RIGHT, SMALL_BUFF)
            for p, prob_bar in zip(probs, prob_bars)
        )

        rows = VGroup(
            VGroup(*trio)
            for trio in zip(token_mobs, prob_bars, percentages)
        )
        rows.add(dots)

        return rows

    def get_pred_mob_row(self, prediction_mob, token):
        for row in prediction_mob[:-1]:
            if row[0].get_text() == token:
                return row
        return prediction_mob[-1]

    def old_animations(self):
        # Predictions
        frame = self.frame
        arrows = VGroup(
            Arrow(
                r1.get_top(),
                r2.get_top(),
                buff=0.2,
                path_arc=-180 * DEG,
                thickness=5,
                fill_color=YELLOW
            )
            for r1, r2 in zip(rects, rects[1:])
        )
        pred_mobs = VGroup(
            self.get_next_token_prediction_mob(prefix).next_to(VGroup(arrow), UP, index_of_submobject_to_align=-1)
            for prefix, arrow in zip(prefixes, arrows)
        )
        for pred_mob in pred_mobs:
            pred_mob.match_y(pred_mobs[0])

        token_ghosts = VGroup(tokens.copy(), rects.copy()).fade(0.8)
        arrow_ghosts = arrows.copy().set_fill(opacity=0.25)
        self.add(token_ghosts)
        self.add(tokens[0], rects[0])

        frame.add_updater(lambda m, dt: m.scale(1 + 0.02 * dt, about_edge=LEFT))
        self.play(
            frame.animate.set_y(1.5).set_anim_args(time_span=(0, 1)),
            ShowSubmobjectsOneByOne(VGroup(*pred_mobs), rate_func=linear, int_func=np.ceil),
            ShowSubmobjectsOneByOne(VGroup(*arrows), rate_func=linear, int_func=np.ceil),
            ShowIncreasingSubsets(arrow_ghosts, rate_func=linear, int_func=np.ceil),
            *(
                LaggedStartMap(FadeIn, group[1:], lag_ratio=0.4)
                for group in [tokens, rects]
            ),
            run_time=20
        )
        frame.clear_updaters()

        # Highlight one prediction
        example_index = 5
        arrow = arrows[example_index]
        pred_mob = pred_mobs[example_index]

        true_pred = pred_mob[0].copy().set_fill(opacity=1).set_stroke(opacity=1)
        true_pred_rect = SurroundingRectangle(true_pred[2])
        true_pred_rect.set_stroke(YELLOW, 2)
        next_rect = rects[example_index + 1]
        next_token = tokens[example_index + 1]

        prob_label = Tex(R"p = P(\text{true next token})")
        prob_label.next_to(true_pred_rect, RIGHT)

        self.play(
            frame.animate.to_default_state().set_y(1.5),
            VFadeIn(arrow),
            FadeIn(pred_mob),
            FadeOut(tokens[example_index + 1:]),
            FadeOut(rects[example_index + 1:]),
        )
        self.wait()
        self.play(
            pred_mob.animate.set_fill(opacity=0.25).set_stroke(opacity=0.25),
            FadeIn(true_pred),
            ShowCreation(true_pred_rect),
            next_rect.animate.set_fill(YELLOW, 0.25).set_stroke(YELLOW, 2, 1),
            next_token.animate.set_fill(WHITE, 1),
        )
        self.wait()


class TokensAndPredictions2(NextTokenPredictions):
    def construct(self):
        # Set up tokens (Not great that I keep copy pasting this)
        text = "Once upon a time, there was a tiny pi creature"
        text_mob = Text(text)
        text_mob.center()

        tokens = break_into_tokens(text_mob)
        rects = get_piece_rectangles(tokens, h_buff=0, leading_spaces=True)
        token_mobs = VGroup(VGroup(*pair) for pair in zip(rects, tokens))

        tokenizer = get_token_encoding()  # Not needed?
        token_indices = tokenizer.encode(text)
        _, offsets = tokenizer.decode_with_offsets(token_indices)
        prefixes = [text[:offset] for offset in offsets[1:]]
        text_tokens = [text[o1:o2] for o1, o2 in zip(offsets, offsets[1:])]

        # Division into tokens
        token_title = Text("Tokens", font_size=72)
        token_title.to_edge(UP, buff=LARGE_BUFF)
        arrows = VGroup(
            Arrow(token_title.get_bottom(), rect.get_top(), fill_color=rect.get_fill_color())
            for rect in rects
        )

        self.play(FadeIn(tokens, lag_ratio=0.1))
        self.add(rects, tokens)
        self.play(
            FadeIn(token_title, shift=0.2 * UP, scale=1.5),
            FadeIn(rects, lag_ratio=0.1),
            LaggedStartMap(VFadeInThenOut, arrows, lag_ratio=0.05, run_time=2),
        )
        self.add(token_mobs)
        self.wait()

        # Show arrow and output
        arrow = Vector(3 * RIGHT, thickness=6, fill_color=TEAL)
        pred_mob_kw = dict(n_shown=12, prob_1_width=8, font_size=30)
        pred_mob = self.get_next_token_prediction_mob(text, **pred_mob_kw)
        pred_mob.set_max_height(FRAME_HEIGHT - 1)
        pred_mob.next_to(arrow, RIGHT)
        model_placeholder = Square(side_length=3)
        model_placeholder.next_to(arrow, UP)

        pre_pred_mob = pred_mob.copy()
        post_token_mobs = token_mobs.copy()
        for part in (*pre_pred_mob, *post_token_mobs):
            part.scale(1e-4).fade(1).move_to(model_placeholder)

        self.play(
            FadeOut(token_title, 0.5 * UP),
            token_mobs.animate.set_width(6).next_to(arrow, LEFT),
            FadeIn(arrow, DOWN),
        )
        self.wait()
        self.play(TransformFromCopy(token_mobs, post_token_mobs, lag_ratio=0.01, path_arc=-45 * DEG), run_time=3)
        self.play(TransformFromCopy(pre_pred_mob, pred_mob, lag_ratio=0.01, path_arc=-45 * DEG), run_time=2)
        self.wait()

        # Show bad predictions
        bad_pred_mob = self.get_next_token_prediction_mob("Hello world", **pred_mob_kw)
        for row in bad_pred_mob:
            row[1].set_fill(RED_D)

        bad_pred_mob.replace(pred_mob, dim_to_match=1)

        self.play(
            Transform(pred_mob, bad_pred_mob, run_time=5, rate_func=there_and_back_with_pause)
        )
        self.wait()


class AbstractGraphOfLossFunction(InteractiveScene):
    xy_init = (1.5, -3.75)
    # xy_init = (1.9, 3.75)

    def construct(self):
        # Set up graph
        frame = self.frame
        x_range = (-4, 4)
        axes = ThreeDAxes(x_range, x_range, (0, 5))
        plane = NumberPlane(x_range, x_range)
        plane.move_to(axes.get_origin())
        plane.background_lines.set_stroke(BLUE_D, 1, 0.5)
        plane.faded_lines.set_stroke(BLUE_E, 1, 0.2)

        def func(x, y):
            return 0.1 * (x * x + y * y) + np.sin(x) + 1.5

        graph = axes.get_graph(func)
        graph.set_color(RED, 0.5)
        graph.always_sort_to_camera(self.camera)

        mesh = SurfaceMesh(graph)
        mesh.set_stroke(WHITE, 1, 0.1)
        mesh.deactivate_depth_test()

        # Add graph indicators
        network = self.get_network()
        network.set_width(1)
        network.move_to(plane.c2p(*self.xy_init))

        line_kw = dict(
            dash_length=0.025,
            stroke_color=RED,
            stroke_width=1
        )

        def point_to_graph(point):
            x, y = plane.p2c(point)
            return axes.c2p(x, y, func(x, y))

        def get_v_line():
            in_point = network.get_center()
            return DashedLine(
                in_point, point_to_graph(in_point),
                **line_kw
            )

        v_line = always_redraw(get_v_line)
        graph_dot = Group(TrueDot().make_3d(), GlowDot())
        graph_dot.set_color(RED)
        graph_dot.f_always.move_to(v_line.get_end)

        # Add loss indicator
        loss_label = Text("Loss")
        loss_label.set_color(RED)
        loss_label.rotate(90 * DEG, RIGHT)
        loss_label.next_to(axes.z_axis, OUT, SMALL_BUFF)
        h_line = always_redraw(lambda: DashedLine(
            v_line.get_end(), axes.z_axis.n2p(axes.z_axis.p2n(v_line.get_end())),
            **line_kw
        ))

        loss_tracker = ValueTracker(4.78)
        get_loss = loss_tracker.get_value

        def get_loss_dec():
            loss = get_loss()
            dec = DecimalNumber(loss, fill_color=RED)
            dec.rotate(90 * DEG, RIGHT)
            dec.next_to(axes.z_axis.n2p(loss), LEFT, buff=0.2)
            return dec

        loss_dec = always_redraw(get_loss_dec)

        # Just show z_axis
        loss_slider = ArrowTip()
        loss_slider.rotate(PI).set_width(0.25)
        loss_slider.rotate(90 * DEG, RIGHT)
        loss_slider.set_color(RED)
        loss_slider.add_updater(lambda m: m.move_to(axes.z_axis.n2p(get_loss()), LEFT))

        loss_func_label = Text("Loss(model)")
        loss_func_label["Loss"].set_color(RED)
        loss_func_label.rotate(90 * DEG, RIGHT)
        loss_func_label.always.next_to(loss_slider, RIGHT)

        brace_lines = DashedLine(ORIGIN, RIGHT).get_grid(2, 1, buff=LARGE_BUFF)
        brace = Brace(brace_lines, LEFT, buff=SMALL_BUFF)
        brace_group = VGroup(brace, brace_lines)
        brace_group.rotate(90 * DEG, RIGHT)
        brace_group.next_to(axes.z_axis, LEFT, buff=0, aligned_edge=IN)

        words = VGroup(Text("Good"), Text("Bad"))
        words.rotate(90 * DEG, RIGHT)
        for word in words:
            word.always.next_to(brace, LEFT)

        frame.reorient(0, 90, 0, (-1.69, 0.68, 1.97), 8.00).set_field_of_view(1 * DEG)
        self.add(axes.z_axis, loss_dec, loss_slider, loss_func_label)

        self.play(
            loss_tracker.animate.set_value(0.43),
            FadeIn(brace_group),
            FadeIn(words[0]),
            run_time=2
        )
        self.wait()
        self.play(LaggedStart(
            brace_group.animate.stretch(2, 2).next_to(axes.z_axis.n2p(5), IN + LEFT, buff=0),
            VFadeOut(words[0]),
            VFadeIn(words[1]),
            loss_tracker.animate.set_value(axes.z_axis.p2n(v_line.get_end())),
            run_time=2
        ))
        self.wait()

        # Transition
        self.play(
            frame.animate.reorient(-28, 70, 0, (0.39, 0.18, 1.18), 10.55).set_field_of_view(45 * DEG),
            FadeOut(loss_func_label[4:]),
            ReplacementTransform(loss_func_label[:4], loss_label),
            FadeOut(loss_slider),
            FadeOut(brace_group),
            FadeOut(words[1]),
            Write(plane),
            ShowCreation(graph),
            Write(mesh),
            run_time=2
        )
        frame.add_ambient_rotation(1 * DEG)
        self.play(
            FadeIn(network, scale=3),
            ShowCreation(v_line, suspend_mobject_updating=True),
            ShowCreation(h_line, suspend_mobject_updating=True),
            FadeIn(graph_dot),
        )
        loss_tracker.add_updater(lambda m: m.set_value(axes.z_axis.p2n(v_line.get_end())))
        self.add(loss_tracker)

        # Iterations of gradient descent
        n_steps = 15
        learning_rate = 0.5
        epsilon = 1e-2

        for n in range(n_steps):
            point = network.get_center()
            x, y = plane.p2c(point)
            del_x = (func(x + epsilon, y) - func(x, y)) / epsilon
            del_y = (func(x, y + epsilon) - func(x, y)) / epsilon
            new_point = plane.c2p(
                x - learning_rate * del_x,
                y - learning_rate * del_y,
            )
            arrow = Arrow(point, new_point, thickness=5, buff=0)
            arrow.set_fill(YELLOW)
            arrow.scale(2, about_point=arrow.get_start())
            anims = [
                network.animate.move_to(new_point), VFadeInThenOut(arrow, run_time=2)
            ]
            self.play(*anims)
        self.wait()

    def get_network(self, layer_sizes=[3, 5, 5, 3]):
        layers = VGroup(
            Dot().get_grid(size, 1, buff=0.1)
            for size in layer_sizes
        )
        layers.arrange(RIGHT, buff=0.3)
        for layer in layers:
            for dot in layer:
                dot.set_stroke(WHITE, 1)
                dot.set_fill(WHITE, random.random())

        lines = VGroup()
        for l1, l2 in zip(layers, layers[1:]):
            lines.add(VGroup(
                Line(
                    n1.get_center(), n2.get_center(),
                    buff=n1.get_width() / 2,
                    stroke_width=random.random(),
                    stroke_color=random.choice([RED, BLUE])
                )
                for n1, n2 in it.product(l1, l2)
            ))

        return VGroup(layers, lines)

    def insert(self):
        # To be placed before transition
        self.remove(loss_dec)
        frame.reorient(22, 80, 0, (0.39, 0.18, 1.18), 10.55)
        graph.save_state()
        mesh.save_state()
        graph.stretch(0, 2).move_to(plane).set_opacity(0),
        mesh.stretch(0, 2).move_to(plane).set_opacity(0),
        self.play(
            frame.animate.reorient(-28, 70, 0, (0.39, 0.18, 1.18), 10.55),
            Restore(graph),
            Restore(mesh),
            run_time=2
        )
        frame.add_ambient_rotation(2 * DEG)
        self.wait(3)


class PileOfTrainingExamples(InteractiveScene):
    def construct(self):
        # Load training examples
        examples = VGroup(
            Text(text, font_size=24)
            for text in get_training_text_examples()
        )
        examples.set_fill(GREY_A, 1)

        # Show training examples
        frame = self.frame

        examples.arrange(DOWN, aligned_edge=LEFT)
        examples.to_edge(DOWN)

        frame.set_height(16)
        frame.move_to(examples, UP).shift(0.25 * UP)
        self.add(examples)
        self.play(
            LaggedStartMap(Write, examples, lag_ratio=0.08),
            frame.animate.move_to(examples, DOWN).shift(DOWN).set_anim_args(time_span=(1, 25)),
            run_time=25
        )
        self.wait()

        # Shift perspective
        self.play(
            frame.animate.reorient(32, 63, 0, (-2.09, 2.53, 0.23), 10.39),
            run_time=3
        )
        self.wait()

        # Pull out one example and tokenize
        example = examples[-1]
        self.play(
            LaggedStartMap(FadeOut, examples[-2::-1], shift=RIGHT, lag_ratio=0.01),
            example.animate.center().fix_in_frame().scale(2).set_color(WHITE),
            run_time=2
        )
        self.wait()


class LossFunction(NextTokenPredictions, InformationOfLanguage):
    def construct(self):
        # Set up tokens
        text = get_training_text_examples()[-1]
        text_mob = Text(text)
        text_mob.center()

        tokens = break_into_tokens(text_mob)
        rects = get_piece_rectangles(tokens, h_buff=0, leading_spaces=True)
        token_mobs = VGroup(VGroup(rect, token) for rect, token in zip(rects, tokens))

        tokenizer = get_token_encoding()
        token_indices = tokenizer.encode(text)
        _, offsets = tokenizer.decode_with_offsets(token_indices)
        offsets.append(len(text))
        prefixes = [text[:offset] for offset in offsets[1:]]
        text_tokens = [text[o1:o2] for o1, o2 in zip(offsets, offsets[1:])]

        self.add(text_mob)
        self.add(rects)

        # Break up into tokens
        probs = [1e-4]  # Arbitrary probability for first token (leaving it blank called too much attention to itself)
        probs += [
            gpt2_token_probability(prefix, token)
            for prefix, token in zip(prefixes, text_tokens[1:])
        ]
        tape = Square(side_length=0.75).get_grid(1, len(probs), buff=0)
        prob_icons = self.get_prob_icons(tape, probs)
        prob_icons.arrange_to_fit_width(13).move_to(DOWN)

        token_mobs.target = token_mobs.generate_target()
        for mob, icon in zip(token_mobs.target, prob_icons):
            mob.scale(0.65)
            mob.next_to(icon, UP)
        rects.set_opacity(0)

        self.play(
            LaggedStartMap(Write, prob_icons),
            MoveToTarget(token_mobs),
            run_time=2
        )
        self.wait()

        # Highlight a subsequence
        idx = 5
        model_placeholder = Square(side_length=2.5)
        model_placeholder.next_to(token_mobs[idx], UP, LARGE_BUFF, RIGHT)
        subsequence = token_mobs[:idx].copy()
        subsequence_target = subsequence.copy()
        cover_rect = Group(
            Point(),
            BackgroundRectangle(VGroup(token_mobs[idx:], prob_icons[idx:]), buff=SMALL_BUFF),
            BackgroundRectangle(prob_icons[:idx], buff=SMALL_BUFF),
        )

        arrow = Vector(RIGHT, fill_color=TEAL)
        arrow.next_to(model_placeholder, RIGHT)

        pred_mob = self.get_next_token_prediction_mob(
            prefixes[idx - 1],
            n_shown=8,
            font_size=20,
            bar_buff=0.15,
            prob_bar_height=0.2,
            prob_bar_colors=(GREEN_C, GREEN_E)
        )
        pred_mob.next_to(arrow, RIGHT)
        pred_mob.to_edge(UP, MED_SMALL_BUFF)
        pre_pred_mob = pred_mob.copy()

        for group in [subsequence_target, pre_pred_mob]:
            for piece in group.family_members_with_points():
                piece.move_to(model_placeholder)
                piece.set_opacity(0)

        brace = always_redraw(lambda: Brace(subsequence, UP))

        self.play(
            FadeIn(cover_rect),
            GrowFromCenter(brace, suspend_mobject_updating=True),
        )
        self.play(
            subsequence.animate.scale(0.9).arrange(RIGHT, buff=0).next_to(model_placeholder, LEFT),
        )
        self.play(LaggedStart(
            FadeOut(brace, suspend_mobject_updating=True),
            TransformFromCopy(subsequence, subsequence_target, path_arc=-90 * DEG, lag_ratio=0.01),
            GrowArrow(arrow),
            TransformFromCopy(pre_pred_mob, pred_mob, path_arc=90 * DEG, lag_ratio=0.001),
            lag_ratio=0.5
        ))
        self.wait()

        # Note relevant next probability
        next_token_index = 0
        tm_copy = token_mobs[idx].copy()
        self.add(tm_copy, cover_rect, token_mobs[idx])
        highlight_rect = SurroundingRectangle(token_mobs[idx], buff=0.1).set_stroke(YELLOW, 2)
        self.play(
            FlashAround(token_mobs[idx], run_time=2),
            FadeIn(token_mobs[idx]),
            FadeOut(tm_copy),
            ShowCreation(highlight_rect),
        )
        self.wait()
        self.play(
            pred_mob[:next_token_index].animate.fade(0.7),
            highlight_rect.animate.surround(pred_mob[next_token_index]),
            pred_mob[next_token_index + 1:].animate.fade(0.7),
            Transform(token_mobs[idx][1].copy(), pred_mob[next_token_index][0], remover=True)
        )
        self.wait()
        icon_copy = prob_icons[idx].copy()
        self.add(icon_copy, cover_rect, token_mobs[idx], prob_icons[idx], highlight_rect)
        self.play(
            FadeIn(prob_icons[idx]),
            FadeOut(icon_copy),
            highlight_rect.animate.surround(prob_icons[idx]),
            FadeTransformPieces(
                pred_mob[next_token_index][2].copy(),
                prob_icons[idx][2],
            )
        )
        self.wait()

        # Clear the board
        self.play(
            FadeOut(subsequence, 0.1 * LEFT, lag_ratio=0.1),
            FadeOut(pred_mob, 0.1 * RIGHT, lag_ratio=0.1),
            FadeOut(arrow, 0.1 * RIGHT),
            FadeOut(highlight_rect),
            FadeOut(cover_rect),
            LaggedStart(
                (FlashAround(icon, time_width=1.5)
                for icon in prob_icons),
                lag_ratio=0.05,
                run_time=3
            )
        )
        self.wait()

        # Show information
        frame = self.frame
        info_values = -np.log2(probs)
        bit_height = 0.2
        info_bars = self.get_information_bars(token_mobs, info_values, bit_height=bit_height)
        info_bars.save_state()
        info_bars.stretch(0, 1, about_edge=DOWN).set_stroke(opacity=0)

        information_title = Text("Information", font_size=72)
        information_title.set_color(BLUE)
        information_title.to_edge(UP)
        information_subtitle = Text("from the model’s perspective")
        information_subtitle.match_width(information_title)
        information_subtitle.next_to(information_title, DOWN, SMALL_BUFF)
        information_subtitle.set_color(BLUE_E)

        top_equation = VGroup(
            Text("Loss", font_size=72).set_color(RED),
            Tex(R"=", font_size=72),
            VGroup(information_title, information_subtitle),
        )
        top_equation.arrange(RIGHT)
        top_equation.to_edge(UP)
        top_equation[0].align_to(information_title, DOWN)
        top_equation[1].match_y(top_equation[0][-1])
        for part in top_equation:
            part.fix_in_frame()

        avg_bar = Line(information_title.get_corner(UL), information_title.get_corner(UR))
        avg_bar.set_stroke(GREY_B, 3)
        avg_bar.shift(SMALL_BUFF * UP)
        avg_bar.fix_in_frame()

        information_title.fix_in_frame()
        information_subtitle.fix_in_frame()

        prob_percentages = VGroup(icon[2] for icon in prob_icons)
        prob_qs = VGroup(
            Tex(Rf"q_{{{n}}}", font_size=30, fill_color=GREEN).move_to(icon[2], UP)
            for n, icon in enumerate(prob_icons)
        )

        y_axis = NumberLine((0, 15), unit_size=bit_height, tick_size=0)
        y_axis.rotate(90 * DEG)
        y_axis.next_to(info_bars, LEFT, MED_LARGE_BUFF, aligned_edge=DOWN)
        width = VGroup(y_axis, info_bars).get_width()
        h_lines = VGroup(
            Line(ORIGIN, width * RIGHT).move_to(y_axis.n2p(n), LEFT)
            for n in range(y_axis.x_range[1] + 1)
        )
        h_lines.set_stroke(GREY_C, 1, 0.35)

        y_label = Tex(R"-\log(q)")
        y_label.set_color(BLUE)
        y_label.next_to(y_axis, UP)

        for icon in prob_icons:
            if len(icon) == 3:
                icon.remove(icon[2])
        self.add(prob_icons)

        self.play(
            frame.animate.reorient(0, 0, 0, (-0.67, 1.25, 0.0), 8.42),
            Write(VGroup(*top_equation[:2], information_title)),
            Restore(info_bars, lag_ratio=0.2, run_time=2),
            FadeTransformPieces(prob_percentages, prob_qs),
            ShowCreation(avg_bar, time_span=(1, 2)),
        )
        self.play(
            FadeIn(information_subtitle, 0.2 * DOWN)
        )
        top_equation.set_backstroke(BLACK, 5)
        self.wait()
        self.play(
            Write(y_label),
            FadeIn(h_lines, shift=0.5 * UP, lag_ratio=0.1),
            Write(y_axis),
        )
        self.wait()

        # Reference a smarter model
        smarter_probs = np.array(probs)
        alphas = np.linspace(0, 1, len(probs))**(0.5)
        alphas[:3] *= 0.5
        alphas[-1] = 0.9
        smarter_probs = interpolate(np.array(probs), np.ones(len(probs)), alphas)
        smarter_infos = -np.log2(smarter_probs)
        dumb_probs = np.array([0.02 * np.random.random(len(probs))**3, probs]).min(0)
        dumb_infos = -np.log2(dumb_probs)

        smarter_prob_icons = self.get_prob_icons(tape, smarter_probs)
        dumb_prob_icons = self.get_prob_icons(tape, dumb_probs)
        for icon1, icon2, icon3 in zip(prob_icons, smarter_prob_icons, dumb_prob_icons):
            for new_icon in [icon2, icon3]:
                new_icon.remove(new_icon[2])
                new_icon.move_to(icon1)

        smarter_info_bars = self.get_information_bars(token_mobs, smarter_infos, bit_height=bit_height)
        dumb_info_bars = self.get_information_bars(token_mobs, dumb_infos, bit_height=bit_height)

        info_bars.save_state()
        prob_icons.save_state()

        bar_arrows = VGroup()
        for bar, p1, p2 in zip(info_bars, probs, smarter_probs):
            arrow = Vector(0.75 * DOWN, thickness=5)
            arrow.set_fill(RED, opacity=clip(3 * (p2 - p1), 0, 1))
            arrow.bar = bar
            arrow.add_updater(lambda m: m.next_to(m.bar, UP, SMALL_BUFF))
            bar_arrows.add(arrow)

        info_bars.refresh_bounding_box()
        loss_line = NumberLine(y_axis.x_range, unit_size=bit_height)
        loss_line.rotate(90 * DEG)
        loss_line.set_stroke(width=2)
        loss_line.next_to(info_bars, RIGHT, MED_LARGE_BUFF)
        loss_line.match_y(y_axis)
        loss_slider = ArrowTip(angle=PI).set_width(0.2).set_fill(RED, 1)
        loss_slider.next_to(loss_line.get_center(), RIGHT, buff=0)
        loss_slider.add_updater(lambda m: m.set_y(np.mean([
            bar.get_y(UP) for bar in info_bars
        ])))
        loss_label = Tex(R"\text{Loss} = 0.00", font_size=36)
        loss_label.set_color(RED)
        loss_label.make_number_changeable("0.00").add_updater(
            lambda m: m.set_value(loss_line.p2n(loss_slider.get_center()))
        )
        loss_label.always.next_to(loss_slider, RIGHT, SMALL_BUFF)
        loss_h_line = Line(loss_line.get_center(), y_axis.get_center())
        loss_h_line.set_stroke(RED, 1, 0.5)
        loss_h_line.always.match_y(loss_slider)

        self.add(prob_qs)
        self.play(
            FadeIn(loss_line),
            FadeIn(loss_slider),
            FadeIn(loss_label),
            FadeIn(loss_h_line),
            frame.animate.reorient(0, 0, 0, (0.77, 1.63, 0.0), 9.86),
        )
        self.play(
            VFadeIn(bar_arrows),
            Transform(prob_icons, smarter_prob_icons),
            Transform(info_bars, smarter_info_bars),
            run_time=3
        )
        self.play(FadeOut(bar_arrows))
        self.wait()
        self.play(
            Transform(prob_icons, dumb_prob_icons),
            Transform(info_bars, dumb_info_bars),
            run_time=3
        )
        self.wait()
        self.play(
            Restore(prob_icons),
            Restore(info_bars),
            run_time=2
        )
        self.wait()

        # Sidebar on -log shape (separate scene)

        # Show tokens one at a time
        kw = dict(rate_func=linear, run_time=10, int_func=np.ceil)
        self.play(*(
            ShowIncreasingSubsets(group, **kw)
            for group in [prob_icons, prob_qs, info_bars]
        ))
        self.wait()

        # Show many examples
        n_examples = 200
        axes = VGroup(y_axis, h_lines)
        x_step = get_norm(token_mobs[1].get_center() - token_mobs[0].get_center())
        height = token_mobs.get_height()
        other_examples = VGroup(
            self.get_token_sequence(text, x_step, height)
            for text in get_training_text_examples()[-2:-n_examples - 2:-1]
        )
        other_examples.arrange(UP, buff=2.0, aligned_edge=LEFT)
        other_examples.next_to(token_mobs[0].get_center(), UP, buff=2.0, aligned_edge=LEFT)
        for example in other_examples:
            example.shift(0.5 * example[0].get_width() * LEFT)

        top_equation.set_backstroke(BLACK, 3).set_z_index(2)
        to_fade = VGroup(loss_line, loss_label, loss_slider, loss_h_line, y_label)
        to_fade.clear_updaters()
        self.play(
            frame.animate.reorient(-26, 70, 0, (-1.64, 5.27, 1.6), 11.98).set_field_of_view(35 * DEG),
            Rotate(axes, 90 * DEG, axis=RIGHT, about_edge=DOWN),
            Rotate(info_bars, 90 * DEG, axis=RIGHT, about_point=axes.get_bottom()),
            FadeOut(to_fade),
            FadeOut(prob_icons),
            FadeOut(prob_qs),
            FadeIn(other_examples, time_span=(1.5, 3), lag_ratio=1e-5, shift=0.25 * UP),
            run_time=3
        )

        other_axes = VGroup(axes.copy().align_to(example, UP) for example in other_examples)
        other_bars = VGroup()
        for example in other_examples:
            bars = VGroup()
            for n, mob in enumerate(example, start=1):
                bar = info_bars[0].copy()
                bar.set_depth(
                    14 * bit_height * (1 / math.sqrt(n)) * random.uniform(0.7, 1.3),
                    about_edge=IN,
                    stretch=True
                )
                bar.move_to(mob.get_top(), IN)
                bars.add(bar)
            other_bars.add(bars)
        other_bars.set_stroke(WHITE, 1, 0.5)
        other_bars.set_fill(BLUE, 0.75)

        self.play(
            info_bars.animate.set_opacity(0.75),
            LaggedStart(
                (FadeIn(bars, lag_ratio=0.2)
                for bars in other_bars),
                lag_ratio=0.05,
                run_time=10
            ),
            frame.animate.reorient(30, 70, 0, (2.5, 7.22, 1.73), 15.11).set_anim_args(run_time=20)
        )

    def get_token_sequence(self, text, x_step, height):
        text_mob = Text(text)
        tokens = break_into_tokens(text_mob)
        rects = get_piece_rectangles(tokens, h_buff=0, leading_spaces=True)
        token_mobs = VGroup(VGroup(rect, token) for rect, token in zip(rects, tokens))
        token_mobs.set_height(height)
        for n, mob in enumerate(token_mobs):
            mob.set_max_width(x_step)
            mob.move_to(n * x_step * RIGHT)
        return token_mobs


class MyNameIsExample(InteractiveScene):
    def construct(self):
        # Feed in "My name is"
        model_placeholder = Square(side_length=3)

        # Show statistical distribution
        pred = gpt2_predict_next_token("My name is")

