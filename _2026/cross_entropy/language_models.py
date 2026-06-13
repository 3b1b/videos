from manim_imports_ext import *
from _2026.cross_entropy.distribution import StackedProbDistribution
from _2026.cross_entropy.next_char import CHAR_ALPHABET
from _2026.cross_entropy.next_char import get_next_char_distribution
from _2026.cross_entropy.next_char import gpt2_predict_next_token
from _2024.transformers.helpers import MachineWithDials
from _2024.transformers.embedding import get_token_encoding
from _2024.transformers.embedding import break_into_tokens
from _2024.transformers.embedding import get_piece_rectangles


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


class AmbientModelPredictions(InteractiveScene):
    def construct(self):
        # Add model
        frame = self.frame
        model = LanguageModel()
        model.set_height(2)
        self.add(model)

        # Test


class ExplainLossFunction(InteractiveScene):
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

        self.play(FadeIn(text_mob, lag_ratio=0.1))
        self.wait()
        self.play(LaggedStartMap(DrawBorderThenFill, rects))
        self.wait()

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

        # Highlight correct next prediction

    def get_next_token_prediction_mob(
        self,
        text,
        n_shown=7,
        prob_1_width=3.0,
        prob_bar_height=0.2,
        prob_bar_colors=(BLUE_D, TEAL_D),
    ):
        tokens, probs = gpt2_predict_next_token(text, n_shown=n_shown)
        prob_bars = VGroup(
            Rectangle(prob * prob_1_width, prob_bar_height)
            for prob in probs
        )
        prob_bars.arrange(DOWN, buff=MED_SMALL_BUFF, aligned_edge=LEFT)
        prob_bars.set_fill(WHITE, 1)
        prob_bars.set_submobject_colors_by_gradient(*prob_bar_colors)
        prob_bars.set_stroke(WHITE, 1)

        token_mobs = VGroup(
            Text(token, font_size=36).next_to(bar, LEFT)
            for bar, token in zip(prob_bars, tokens)
        )
        dots = Tex(R"\vdots")
        dots.next_to(token_mobs[-1], DR)

        percentages = VGroup(
            DecimalNumber(
                100 * p,
                unit="%",
                font_size=20,
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
