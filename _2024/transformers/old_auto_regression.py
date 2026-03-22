from manim_imports_ext import *
from _2024.transformers.helpers import *

from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
from transformers import PreTrainedModel
import torch
import openai
import tiktoken


@lru_cache(maxsize=1)
def get_gpt2_tokenizer(model_name='gpt2'):
    return GPT2Tokenizer.from_pretrained(model_name)


@lru_cache(maxsize=1)
def get_gpt2_model(model_name='gpt2'):
    return GPT2LMHeadModel.from_pretrained(model_name)


def gpt2_predict_next_token(text, n_shown=7):
    tokenizer = get_gpt2_tokenizer()
    model = get_gpt2_model()
    # Encode the input text
    indexed_tokens = tokenizer.encode(
        text, add_special_tokens=False, return_tensors='pt'
    )

    # Predict all tokens
    with torch.no_grad():
        outputs = model(indexed_tokens)
        # Pull out the first batch, and the last token prediction
        predictions = outputs[0][0, -1, :]

    # Get the predicted next token
    indices = torch.argsort(predictions)
    top_indices = reversed(indices[-n_shown:])
    tokens = list(map(tokenizer.decode, top_indices))
    probs = softmax(predictions)[top_indices]

    return tokens, probs


def gpt3_predict_next_token(text, n_shown=10, random_seed=0):
    openai.api_key = os.getenv('OPENAI_KEY')
    response = openai.Completion.create(
        # Or another model version, adjust as necessary
        engine="gpt-3.5-turbo-instruct",
        prompt=text,
        max_tokens=1,
        n=1,
        temperature=1.0,
        user=str(random_seed),
        # Retrieve more than are shown
        logprobs=50
    )
    top_logprob_dict = response.choices[0]["logprobs"]["top_logprobs"][0]
    tokens, logprobs = zip(*top_logprob_dict.items())
    probs = np.exp(logprobs)
    indices = np.argsort(probs)
    top_indices = indices[-1:-n_shown:-1]
    top_tokens = [tokens[i] for i in top_indices]
    top_probs = [probs[i] for i in top_indices]
    return top_tokens, top_probs


class SimpleAutogregression(InteractiveScene):
    text_corner = 3.5 * UP + 0.75 * RIGHT
    line_len = 31
    font_size = 35
    n_shown_predictions = 12
    seed_text = "Behold, a wild pi creature, foraging in its native"
    seed_text_color = BLUE_B
    machine_name = "Transformer"
    machine_phi = 10 * DEGREES
    machine_theta = 12 * DEGREES
    n_predictions = 120
    skip_through = False
    random_seed = 0
    model = "gpt2"

    def construct(self):
        # Repeatedly generate
        text_mob, next_word_line, machine = self.init_text_and_machine()
        for n in range(self.n_predictions):
            text_mob = self.new_selection_cycle(
                text_mob, next_word_line, machine,
                quick=(n > 10),
                skip_anims=self.skip_through,
            )

    def init_text_and_machine(self):
        # Set up active text
        self.cur_str = self.seed_text
        text_mob = self.string_to_mob(self.cur_str)
        text_mob.set_color(self.seed_text_color)
        next_word_line = self.get_next_word_line(text_mob)

        # Set up Transformer as some sort of machine
        machine = self.get_transformer_drawing()
        machine.set_y(0).to_edge(LEFT, buff=-0.6)

        self.add(text_mob)
        self.add(next_word_line)
        self.add(machine)

        return text_mob, next_word_line, machine

    def string_to_mob(self, text):
        text += " l"  # Dumb hack for alignment
        result = get_paragraph(
            text.replace("\n", " ").split(" "),
            self.line_len,
            self.font_size
        )
        result.move_to(self.text_corner, UL)
        result[-1].set_fill(BLACK, 0)  # Continue dumb hack
        result[-1].stretch(0, 0, about_edge=LEFT)
        return result

    def get_next_word_line(self, text_mob, char_len=7):
        next_word_line = Underline(text_mob[:char_len])
        next_word_line.set_stroke(TEAL, 2)
        next_word_line.next_to(text_mob[-1], RIGHT, SMALL_BUFF, aligned_edge=DOWN)
        if self.skip_through:
            next_word_line.set_opacity(0)
        return next_word_line

    def get_transformer_drawing(self):
        self.camera.light_source.move_to([-5, 5, 10])
        self.frame.set_field_of_view(20 * DEGREES)
        blocks = VGroup(
            VPrism(3, 2, 0.2)
            for n in range(10)
        )
        blocks.set_fill(GREY_D, 1)
        blocks.set_stroke(width=0)
        blocks.set_shading(0.25, 0.5, 0.2)
        blocks.arrange(OUT)
        blocks.move_to(ORIGIN, OUT)
        blocks.rotate(self.machine_phi, RIGHT, about_edge=OUT)
        blocks.rotate(self.machine_theta, UP, about_edge=OUT)

        blocks.deactivate_depth_test()
        for block in blocks:
            block.sort(lambda p: p[2])

        word = Text(self.machine_name, alignment="LEFT")
        word.next_to(blocks[-1], UP)
        word.shift(0.1 * UP + 0.4 * LEFT)
        word.move_to(blocks[-1])
        word.set_backstroke(BLACK, 5)
        out_arrow = Vector(
            0.5 * RIGHT, stroke_width=10,
            max_tip_length_to_length_ratio=0.5,
            max_width_to_length_ratio=12
        )
        out_arrow.next_to(blocks[-1], RIGHT, buff=SMALL_BUFF)
        out_arrow.set_opacity(0)

        result = VGroup(blocks, word, out_arrow)
        return result

    def get_distribution(
        self, words, probs, machine,
        font_size=24,
        width_100p=1.8,
        bar_height=0.25,
        show_ellipses=True
    ):
        labels = VGroup(Text(word, font_size=font_size) for word in words)
        bars = VGroup(
            Rectangle(prob * width_100p, bar_height)
            for prob, label in zip(probs, labels)
        )
        bars.arrange(DOWN, aligned_edge=LEFT, buff=0.5 * bar_height)
        bars.set_fill(opacity=1)
        bars.set_submobject_colors_by_gradient(TEAL, YELLOW)
        bars.set_stroke(WHITE, 1)

        bar_groups = VGroup()
        for label, bar, prob in zip(labels, bars, probs):
            prob_label = Integer(int(100 * prob), unit="%", font_size=0.75 * font_size)
            prob_label.next_to(bar, RIGHT, buff=SMALL_BUFF)
            label.next_to(bar, LEFT)
            bar_groups.add(VGroup(label, bar, prob_label))

        if show_ellipses:
            ellipses = Tex(R"\vdots", font_size=font_size)
            ellipses.next_to(bar_groups[-1][0], DOWN)
            bar_groups.add(ellipses)

        arrow_point = machine[-1].get_right()
        bar_groups.shift(arrow_point - bars.get_left() + 1.5 * RIGHT)
        bar_groups.align_to(machine, UP)

        return bar_groups

    def animate_text_input(self, text_mob, machine, position_text_over_machine=True, added_anims=[], lag_ratio=0.02):
        blocks = machine[0]
        text_copy = text_mob.copy()
        if position_text_over_machine:
            text_copy.target = text_copy.generate_target()
            text_copy.target.set_max_width(4)
            text_copy.target.next_to(blocks[0], UP)
            text_copy.target.shift_onto_screen()
            self.play(MoveToTarget(text_copy, path_arc=-45 * DEGREES))
        self.play(LaggedStart(
            *added_anims,
            Transform(
                text_copy,
                VGroup(VectorizedPoint(machine.get_top())),
                lag_ratio=lag_ratio,
                run_time=1,
                path_arc=-45 * DEGREES,
                remover=True,
            ),
            LaggedStart(
                (
                    block.animate.set_color(
                        block.get_color() if block is blocks[-1] else TEAL
                    ).set_anim_args(rate_func=there_and_back)
                    for block in blocks
                ),
                lag_ratio=0.1,
                run_time=1
            ),
            Animation(machine[1:]),
            lag_ratio=0.5
        ))

    def animate_prediction_ouptut(self, machine, cur_str):
        words, probs = self.predict_next_token(cur_str)
        bar_groups = self.get_distribution(words, probs, machine)
        self.play(
            LaggedStart(
                (FadeInFromPoint(bar_group, machine[0][-1].get_right())
                for bar_group in bar_groups),
                lag_ratio=0.025,
                group=bar_groups,
                run_time=1
            )
        )
        return bar_groups

    def animate_random_sample(self, bar_groups):
        widths = np.array([group[1].get_width() for group in bar_groups[:-1]])
        dist = widths / widths.sum()
        seed = random.randint(0, 1000)
        buff = 0.025
        highlight_rect = SurroundingRectangle(bar_groups[0], buff=buff)
        highlight_rect.set_stroke(YELLOW, 2)
        highlight_rect.set_fill(YELLOW, 0.25)

        def highlight_randomly(rect, dist, alpha):
            np.random.seed(seed + int(10 * alpha))
            index = np.random.choice(np.arange(len(dist)), p=dist)
            rect.surround(bar_groups[index], buff=buff)
            rect.stretch(1.1, 0)

        self.play(
            UpdateFromAlphaFunc(highlight_rect, lambda rect, a: highlight_randomly(rect, dist, a)),
            Animation(bar_groups)
        )

        bar_groups.add_to_back(highlight_rect)

    def animate_word_addition(self, bar_groups, text_mob, next_word_line, force_unskip=False):
        # Choose the highlighted_group
        bar_group = None
        if isinstance(bar_groups[0], Rectangle):
            # Use the highlight rect to find the group element
            bars = bar_groups[1:-1]
            diffs = [abs(bg.get_y() - bar_groups[0].get_y()) for bg in bars]
            bar_group = bar_groups[1:][np.argmin(diffs)]
        if bar_group is None:
            bar_group = bar_groups[0]

        # Animate selection
        word = bar_group[0].get_text()
        new_str = self.cur_str + word
        new_text_mob = self.string_to_mob(new_str)
        new_text_mob[:len(self.seed_text.replace(" ", ""))].set_color(self.seed_text_color)

        word_targets = new_text_mob[word.strip()]
        if len(word_targets) > 0:
            target = word_targets[-1]
        else:
            target = new_text_mob[-len(word) - 1:-1]

        # target = new_text_mob[-len(word):]

        self.add(bar_groups)
        self.play(
            FadeTransform(bar_group[0].copy(), target),
            Transform(
                next_word_line,
                self.get_next_word_line(new_text_mob),
            ),
        )
        if force_unskip:
            self.skip_animations = False
            target.save_state()
            target.set_fill(YELLOW)
            self.wait(0.5)
            target.restore()
            self.skip_animations = True
        self.play(
            FadeOut(bar_groups),
        )

        self.remove(text_mob)
        self.add(new_text_mob)

        self.cur_str = new_str

        return new_text_mob

    def new_selection_cycle(self, text_mob, next_word_line, machine, quick=False, skip_anims=False):
        if skip_anims:
            self.skip_animations = True

        if quick:
            words, probs = self.predict_next_token(self.cur_str)
            bar_groups = self.get_distribution(words, probs, machine)
            self.add(bar_groups)
        else:
            self.animate_text_input(text_mob, machine)
            bar_groups = self.animate_prediction_ouptut(machine, self.cur_str)
        self.animate_random_sample(bar_groups)
        new_text_mob = self.animate_word_addition(
            bar_groups, text_mob, next_word_line,
            force_unskip=skip_anims
        )
        return new_text_mob

    #

    def predict_next_token(self, text):
        result = None
        n_shown = self.n_shown_predictions
        if self.model == "gpt3":
            try:
                result = gpt3_predict_next_token(
                    text, n_shown, random_seed=self.random_seed
                )
            except Exception as e:
                pass
        if result is None:
            result = gpt2_predict_next_token(text, n_shown)
        return result


class AnnotateNextWord(SimpleAutogregression):
    def construct(self):
        text_mob, next_word_line, machine = self.init_text_and_machine()
        self.add(machine, *machine[1:])
        words, probs = self.predict_next_token(self.cur_str)
        bar_groups = self.get_distribution(words, probs, machine)

        self.add(bar_groups)

        # Initial text
        from manimlib.mobject.boolean_ops import Union
        highlight = Union(
            SurroundingRectangle(text_mob["in its native"]),
            SurroundingRectangle(text_mob["Behold, a wild pi creature, foraging"]),
        )
        highlight.set_stroke(BLUE, 3)
        arrow = Vector(RIGHT, stroke_width=10)
        arrow.next_to(highlight, LEFT)

        dist_rect = SurroundingRectangle(bar_groups)
        dist_rect.set_stroke(YELLOW, 2)

        self.play(
            ShowCreation(highlight),
            GrowArrow(arrow)
        )
        self.wait()
        self.play(
            arrow.animate.rotate(-PI / 2).next_to(dist_rect, UP),
            ReplacementTransform(highlight, dist_rect),
        )
        self.wait()
        self.play(
            FadeOut(dist_rect),
            FadeOut(arrow),
        )


class QuickerRegression(SimpleAutogregression):
    skip_through = True


class AutoregressionGPT3(SimpleAutogregression):
    model = "gpt3"


class QuickRegressionGPT3(SimpleAutogregression):
    skip_through = True
    model = "gpt3"


class GPT3CleverestAutocomplete(QuickRegressionGPT3):
    seed_text = "To date, the cleverest thinker of all time was"
    n_predictions = 70

    def construct(self):
        # Test
        text_mob, next_word_line, machine = self.init_text_and_machine()
        for n in range(self.n_predictions):
            text_mob = self.new_selection_cycle(
                text_mob, next_word_line, machine,
                skip_anims=(n > 2),
            )


class GPT3OnLearningSimpler(QuickRegressionGPT3):
    seed_text = "The most effective way to learn computer science is"
    text_corner = 3.5 * UP + 3 * LEFT
    line_len = 35
    font_size = 35
    n_predictions = 300
    time_per_prediction = 0.2
    random_seed = 313

    def construct(self):
        # Test
        cur_str = self.seed_text
        text_mob = VGroup()
        for n in range(self.n_predictions):
            self.remove(text_mob)
            words, probs = self.predict_next_token(cur_str)
            probs = probs / probs.sum()
            index = np.random.choice(np.arange(len(words)), p=probs)
            new_word = words[index]
            cur_str += new_word
            text_mob = self.string_to_mob(cur_str)
            text_mob[:len(self.seed_text.replace(" ", ""))].set_color(BLUE)
            text_mob[new_word.strip()][-1].set_color(YELLOW)
            if text_mob.get_bottom()[1] < -3:
                text_mob.shift(5 * UP)
                self.text_corner += 5 * UP
            self.add(text_mob)
            self.wait(self.time_per_prediction)


class ModelTakingInTextWithSurroundingPieces(SimpleAutogregression):
    def construct(self):
        text_mob, next_word_line, machine = self.init_text_and_machine()


class AthleteCompletion(SimpleAutogregression):
    seed_text = "Michael Jordan plays the sport of"
    text_corner = 3.5 * UP + 3.0 * LEFT
    machine_phi = 5 * DEGREES
    machine_theta = 12 * DEGREES
    model = "gpt3"

    def construct(self):
        # Initialize machine
        self.set_floor_plane("xz")
        frame = self.frame
        in_text, next_word_line, machine = self.init_text_and_machine()
        self.clear()
        machine = VGroup(*machine[0])
        machine.set_height(4)
        machine.next_to(in_text, DOWN, buff=LARGE_BUFF)

        dials = MachineWithDials(n_rows=10, n_cols=15).dials
        dials.set_stroke(opacity=0.25)
        dials.set_height(machine[-1].get_height() * 0.9)

        llm_title = Text("Large\nLanguage\nModel", alignment="LEFT", font_size=72)
        llm_title.set_backstroke(width=8)

        for mob in [dials, llm_title]:
            mob.rotate(self.machine_phi, RIGHT).rotate(self.machine_theta, UP)
            mob.move_to(machine[-1], OUT)

        last_block_copy = machine[-1].copy()
        self.add(last_block_copy)

        frame.reorient(-13, -6, 0)
        self.play(
            LaggedStart(
                (TransformFromCopy(last_block_copy.copy().set_opacity(0), block)
                for block in machine),
                lag_ratio=0.05,
            ),
            Write(dials),
            Write(llm_title),
            frame.animate.reorient(0, 0, 0),
            run_time=3
        )
        self.remove(last_block_copy)
        self.add(machine, dials, llm_title)

        # Feed in many facts
        facts = Path(DATA_DIR, "facts.txt").read_text().split("\n")
        fact_mobs = VGroup(get_paragraph(fact.split(" "), line_len=20) for fact in facts)
        directions = compass_directions(12, start_vect=UR)
        for fact_mob, vect in zip(fact_mobs, it.cycle(directions)):
            fact_mob.set_max_width(2)
            fact_mob.move_to(5 * vect).shift_onto_screen(buff=0.25)

        self.play(
            LaggedStart(
                (Succession(
                    FadeIn(fact_mob),
                    fact_mob.animate.set_opacity(0).move_to(machine.get_center()),
                )
                for fact_mob in fact_mobs),
                lag_ratio=0.05,
                run_time=8
            )
        )
        self.remove(fact_mobs)
        self.wait()

        # Show MJ fact
        full_input = VGroup(in_text, next_word_line)
        full_input.set_height(0.4)
        full_input.to_edge(UP)

        in_arrow = Arrow(full_input, machine, buff=0.1)
        predictions, probs = self.predict_next_token(self.seed_text)

        bar_groups = self.get_distribution(predictions, probs, machine)
        bar_groups.next_to(machine[-1], RIGHT, buff=1.5)
        out_arrow = Arrow(machine[-1], bar_groups)

        top_rect = SurroundingRectangle(VGroup(bar_groups[0]))

        self.play(FadeIn(full_input, scale=2))
        self.play(
            GrowArrow(in_arrow),
            Transform(full_input.copy(), full_input.copy().scale(0.5).set_opacity(0).move_to(machine.get_top()))
        )
        self.play(
            frame.animate.reorient(-14, -2, 0, (1.83, 0.07, -0.38), 8.63),
            LaggedStart(
                (block.animate.set_color(TEAL).set_anim_args(rate_func=there_and_back)
                for block in machine[:-1]),
                lag_ratio=0.1,
                run_time=1
            ),
        )
        self.play(
            ShowCreation(out_arrow),
            FadeIn(bar_groups, lag_ratio=0.1)
        )
        self.wait()
        self.play(ShowCreation(top_rect))

        # Reshow parameters
        self.play(
            FadeOut(llm_title),
            dials.animate.set_stroke(opacity=1)
        )
        for _ in range(5):
            self.play(
                LaggedStart(
                    (dial.animate_set_value(dial.get_random_value())
                    for dial in dials),
                    lag_ratio=0.25 / len(dials),
                    run_time=1
                )
            )

        # Quetsions
        questions = VGroup(Text("How?"), Text("Where?"))
        questions.arrange(RIGHT, buff=1.0)
        questions.set_height(0.5)
        questions.next_to(machine[-1], DOWN)

        for question in questions:
            self.play(FadeIn(question, 0.5 * UP, scale=1.5))
        self.wait()


class ThatWhichDoesNotKillMe(SimpleAutogregression):
    text_corner = 3.5 * UP + 5.0 * LEFT
    line_len = 75
    # seed_text = "That which does not kill you only makes you"
    seed_text = "Down by the river bank"
    model = "gpt3"

    def construct(self):
        # Test
        text_mob, next_word_line, machine = self.init_text_and_machine()
        machine.set_x(0)
        text_mob = self.new_selection_cycle(
            text_mob, next_word_line, machine,
            quick=False,
            skip_anims=False,
        )
