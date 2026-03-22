from transformers.models.videomae import image_processing_videomae
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
        logprobs=50  # I think this is actually set to a max of 20?
    )
    top_logprob_dict = response.choices[0]["logprobs"]["top_logprobs"][0]
    tokens, logprobs = zip(*top_logprob_dict.items())
    probs = np.exp(logprobs)
    indices = np.argsort(-probs)
    shown_tokens = [tokens[i] for i in indices[:n_shown]]
    return shown_tokens, probs[indices[:n_shown]]


def clean_text(text):
    return " ".join(filter(lambda s: s.strip(), re.split(r"\s", text)))


def next_token_bar_chart(
    words, probs,
    reference_point=ORIGIN,
    font_size=24,
    width_100p=1.0,
    prob_exp=0.75,
    bar_height=0.25,
    bar_space_factor=0.5,
    buff=1.2,
    show_ellipses=True,
    use_percent=True,
):
    labels = VGroup(Text(word, font_size=font_size) for word in words)
    bars = VGroup(
        Rectangle(prob**(prob_exp) * width_100p, bar_height)
        for prob, label in zip(probs, labels)
    )
    bars.arrange(DOWN, aligned_edge=LEFT, buff=bar_space_factor * bar_height)
    bars.set_fill(opacity=1)
    bars.set_submobject_colors_by_gradient(TEAL, YELLOW)
    bars.set_stroke(WHITE, 1)

    bar_groups = VGroup()
    for label, bar, prob in zip(labels, bars, probs):
        if use_percent:
            prob_label = Integer(int(100 * prob), unit="%", font_size=0.75 * font_size)
        else:
            prob_label = DecimalNumber(prob, font_size=0.75 * font_size)
        prob_label.next_to(bar, RIGHT, buff=SMALL_BUFF)
        label.next_to(bar, LEFT)
        bar_groups.add(VGroup(label, bar, prob_label))

    if show_ellipses:
        ellipses = Tex(R"\vdots", font_size=font_size)
        ellipses.next_to(bar_groups[-1][0], DOWN)
        bar_groups.add(ellipses)

    bar_groups.shift(reference_point - bars.get_left() + buff * RIGHT)

    return bar_groups


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


class AltSimpleAutoRegression(SimpleAutogregression):
    n_predictions = 1
    line_len = 25

    def reposition_transformer_drawing(self, machine):
        machine.move_to(0.5 * RIGHT)
        in_arrow = machine[-1].copy()
        in_arrow.rotate(-45 * DEGREES)
        in_arrow.next_to(machine, UL)
        self.add(in_arrow)
        return machine


class AnnotateNextWord(SimpleAutogregression):
    def construct(self):
        text_mob, next_word_line, machine = self.init_text_and_machine()
        self.add(machine, *machine[1:])
        words, probs = self.predict_next_token(self.cur_str)
        bar_groups = self.get_distribution(words, probs, machine[-1].get_right())

        self.add(bar_groups)

        # Initial text
        from manimlib.mobject.boolean_ops import Union
        highlight = Union(
            SurroundingRectangle(text_mob["Behold, a wild pi creature,"]),
            SurroundingRectangle(text_mob["foraging in its native"]),
        )
        highlight.set_stroke(BLUE, 3)
        arrow = Vector(LEFT, stroke_width=10)
        arrow.next_to(highlight, RIGHT).match_y(text_mob[0])

        dist_rect = SurroundingRectangle(bar_groups)
        dist_rect.set_stroke(YELLOW, 2)

        self.play(
            ShowCreation(highlight),
            GrowArrow(arrow)
        )
        self.wait()
        self.play(
            arrow.animate.rotate(PI / 2).next_to(dist_rect, UP),
            ReplacementTransform(highlight, dist_rect),
        )
        self.wait()
        self.play(
            FadeOut(dist_rect),
            FadeOut(arrow),
        )

        # Flash through
        self.remove(bar_groups)
        text_mob = self.new_selection_cycle(
            text_mob, next_word_line, machine,
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
    model = "gpt3"
    min_y = -3
    up_shift = 5 * UP
    show_dist = False

    def construct(self):
        # Test
        cur_str = self.seed_text
        text_mob = VGroup()
        for n in range(self.n_predictions):
            self.clear()
            words, probs = self.predict_next_token(cur_str, n_shown=20)
            index = np.random.choice(np.arange(len(words)), p=(probs / probs.sum()))
            new_word = words[index]
            cur_str += new_word
            text_mob = self.string_to_mob(cur_str)

            # Color seed
            if self.color_seed:
                text_mob[:len(self.seed_text.replace(" ", ""))].set_color(BLUE)

            # Add to text, shift if necessary
            text_mob[new_word.strip()][-1].set_color(YELLOW)
            if text_mob.get_bottom()[1] < self.min_y:
                text_mob.shift(self.up_shift)
                self.text_corner += self.up_shift
            self.add(text_mob)

            # Add the distribution
            if self.show_dist:
                dist = self.get_distribution(
                    words[:self.n_shown_predictions],
                    probs[:self.n_shown_predictions],
                    buff=0
                )
                dist.set_height(4)
                dist.to_edge(DOWN)
                rect = SurroundingRectangle(dist[min(index, len(dist) - 1)])
                self.add(dist, rect)

            self.wait(self.time_per_prediction)


class GPT3OnLongPassages(GPT3OnLearningSimpler):
    seed_text = "Writing long passages seems to involve more foresight and planning than what single-word prediction"
    n_predictions = 100
    color_seed = False


class GPT3CreaturePrediction(GPT3CleverestAutocomplete):
    seed_text = "the fluffy blue creature"
    n_predictions = 1


class GPT3CreaturePrediction2(GPT3CleverestAutocomplete):
    seed_text = "the fluffy blue creature roamed the"
    n_predictions = 1


class LowTempExample(GPT3OnLearningSimpler):
    seed_text = "Once upon a time, there was a"
    model = "gpt3"
    min_y = 1
    up_shift = 2 * UP
    show_dist = True
    temp = 0
    n_predictions = 200
    time_per_prediction = 0.25

    def predict_next_token(self, text, n_shown=None):
        words, probs = super().predict_next_token(text, n_shown)
        if self.temp == 0:
            probs = np.zeros_like(probs)
            probs[0] = 1
        else:
            probs = probs**(1 / self.temp)
            probs /= probs.sum()
        return words, probs


class HighTempExample(LowTempExample):
    temp = 5
    model = "gpt3"


class MidTempExample(LowTempExample):
    seed_text = "If you could see the underlying probability distributions a large language model uses when generating text, then"
    temp = 1
    model = "gpt3"


class ChatBotPrompt(SimpleAutogregression):
    system_prompt = """
        What follows is a conversation between a user and a helpful,
        very knowledgeable AI assistant.
    """
    user_prompt = "User: Give me some ideas for what to do when visiting Paris."
    ai_seed = "AI Assistant: "
    machine_name = "Large\nLanguage\nModel"

    line_len = 28
    font_size = 36
    color_seed = False

    n_predictions = 60
    model = "gpt3"
    random_seed = 12

    def construct(self):
        # Test
        text_mob, next_word_line, machine = self.init_text_and_machine()

        all_strs = list(map(clean_text, [self.system_prompt, self.user_prompt, self.ai_seed]))

        system_prompt, user_prompt, ai_seed = all_text = VGroup(
            get_paragraph(
                s.split(" "),
                font_size=self.font_size,
                line_len=self.line_len
            )
            for s in all_strs
        )
        all_text.arrange(DOWN, aligned_edge=LEFT, buff=0.75)
        all_text.move_to(self.text_corner, UL)
        self.remove(text_mob)
        self.add(all_text)

        text_mob = ai_seed
        self.text_corner = text_mob.get_corner(UL)
        next_word_line.next_to(ai_seed, RIGHT, aligned_edge=DOWN)

        self.cur_str = "\n\n".join(all_strs)

        # Comment on system prompt
        sys_rect = SurroundingRectangle(system_prompt)
        sys_rect.set_stroke(GREEN, 2)

        self.play(
            ShowCreation(sys_rect),
            system_prompt.animate.set_color(GREEN_B)
        )
        self.wait()

        # Users prompt
        from manimlib.mobject.boolean_ops import Union

        top_line = user_prompt["Give me some ideas for what"]
        low_line = user_prompt["to do when visiting Santiago."]
        user_rect = Union(
            SurroundingRectangle(low_line),
            SurroundingRectangle(top_line),
        )
        user_rect.set_stroke(BLUE, 2)

        sys_rect.insert_n_curves(100)
        self.play(
            ReplacementTransform(sys_rect, user_rect),
            top_line.animate.set_color(BLUE_B),
            low_line.animate.set_color(BLUE_B),
        )
        self.wait()
        self.play(
            FadeOut(user_rect),
        )

        # Run predictions
        text_mob = all_text
        self.add(all_text.copy())
        for n in range(self.n_predictions):
            text_mob = self.new_selection_cycle(
                text_mob, next_word_line, machine,
                skip_anims=(n > 0),
            )

    def string_to_mob(self, text):
        seed = self.ai_seed.strip()
        if seed in text:
            text = text[text.index(seed):]
        return super().string_to_mob(text)


class ChatBotPrompt2(ChatBotPrompt):
    user_prompt = "User: Can you explain what temperature is, in the context of softmax?"


class ChatBotPrompt3(ChatBotPrompt):
    user_prompt = "User: Can you give me some ideas for what to do while visiting Munich?"


class VoiceToTextExample(SimpleAutogregression):
    model_name = "voice-to-text"

    def construct(self):
        # Add model
        box = Rectangle(4, 3)
        box.set_stroke(WHITE, 2)
        name = Text(self.model_name, font_size=60)
        name.set_max_width(box.get_width())
        name.next_to(box, UP)
        machine = self.get_transformer_drawing()
        machine.center()
        machine.set_max_width(0.75 * box.get_width())
        machine.move_to(box)
        arrows = Vector(0.75 * RIGHT, stroke_width=8).replicate(2)
        arrows[0].next_to(box, LEFT, SMALL_BUFF)
        arrows[1].next_to(box, RIGHT, SMALL_BUFF)
        model = Group(box, name, arrows, machine)

        self.add(*model)
        self.add(Point())

        # Process input
        max_width = 3.75
        in_mob = self.get_input().set_max_width(max_width)
        out_mob = self.get_output().set_max_width(max_width)
        in_mob.next_to(arrows, LEFT)
        out_mob.next_to(arrows, RIGHT)

        self.add(in_mob)
        self.play(LaggedStart(
            FadeOutToPoint(
                in_mob.copy(), machine.get_left(),
                path_arc=-45 * DEGREES,
                lag_ratio=0.01,
            ),
            LaggedStart(
                (block.animate.set_color(TEAL).set_anim_args(rate_func=there_and_back)
                for block in machine[0][:-1]),
                lag_ratio=0.1,
                run_time=1,
            ),
            FadeInFromPoint(
                out_mob.copy(), machine.get_right(),
                path_arc=45 * DEGREES,
                lag_ratio=0.02
            ),
            lag_ratio=0.7
        ))
        self.wait()

    def get_input(self) -> Mobject:
        result =ImageMobject("AudioSnippet").set_width(3.75)
        result.set_height(3, stretch=True)
        return result

    def get_output(self) -> Mobject:
        return Text("""
            Some models take
            in audio and
            produce a transcript
        """, alignment="LEFT")


class TextToVoiceExample(VoiceToTextExample):
    model_name = "text-to-voice"

    def get_input(self):
        return Text("""
            This sentence comes from
            a model going the other
            way around, producing
            synthetic speech just
            from text.
        """, alignment="LEFT")

    def get_output(self):
        return super().get_input()


class TextToImage(VoiceToTextExample):
    model_name = "text-to-image"
    prompt = """
        1960s photograph of a cute fluffy blue wild pi
        creature, a creature whose body is shaped like
        the symbol π, who is foraging in its native territory,
        staring back at the camera with an exotic scene
        in the background.
    """
    image_name = "PiCreatureDalle3_5"

    def get_clean_prompt(self):
        return clean_text(self.prompt)

    def get_input(self):
        return get_paragraph(self.get_clean_prompt().split(" "), line_len=25)

    def get_output(self):
        return ImageMobject(self.image_name)

    def generate_output(self):
        # Test
        self.prompt = """
            1960s photograph of a cute fluffy blue wild pi
            creature, a creature whose face bears a subtle resemblence
            to the shape of the symbol π, who is foraging in its native
            territory, staring back at the camera with an exotic scene
            in the background.
        """

        self.prompt = "abstract depiction of furry fluffiness"

        openai.api_key = os.getenv('OPENAI_KEY')
        prompt = self.get_clean_prompt()

        response = openai.Image.create(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )

        image_url = response.data[0].url
        print(prompt)
        print(image_url)

        response = openai.Image.create_variation(
          image=open("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/images/raster/PiCreatureDalle3_17.png", "rb"),
          n=1,
          size="1024x1024"
        )


class TranslationExample(VoiceToTextExample):
    model_name = "machine translation"

    def get_input(self):
        return Text("Attention is all\nyou need")

    def get_output(self):
        return Group(Point(), *Text("注意力就是你所需要的一切"))


class PredictionVsGeneration(SimpleAutogregression):
    model = "gpt2"

    def construct(self):
        # Setup
        self.add(FullScreenRectangle())
        morty = Mortimer()
        morty.to_edge(DOWN)
        morty.body.insert_n_curves(100)
        self.add(morty)

        # Words
        words = VGroup(Text("Prediction"), Text("Generation"))
        words.scale(1.5)
        for vect, word in zip([UL, UR], words):
            word.next_to(morty, vect)
            word.shift(0.5 * UP)

        # Create prediction object
        seed_text = "The goal of predicting the next"
        self.n_shown_predictions = 8
        tokens, probs = self.predict_next_token(seed_text)
        dist = self.get_distribution(tokens, probs)
        brace = Brace(dist, LEFT, SMALL_BUFF)
        words = Text(seed_text, font_size=36).next_to(brace, LEFT)
        prediction = VGroup(words, brace, dist)
        prediction.set_width(FRAME_WIDTH / 2 - 1)
        prediction.next_to(morty, UL)
        prediction.shift(0.5 * UP).shift_onto_screen()
        self.add(prediction)

        # Animations
        self.play(
            morty.change("raise_right_hand", prediction),
            FadeIn(prediction[0], UP),
            GrowFromCenter(prediction[1]),
            LaggedStart(
                (FadeInFromPoint(bar, prediction[1].get_center())
                for bar in prediction[2]),
                lag_ratio=0.05,
            )
        )
        self.play(Blink(morty))
        self.play(
            morty.change("raise_left_hand", 3 * UR),
        )
        self.wait()
        self.play(Blink(morty))
        self.wait()


class ManyParallelPredictions(SimpleAutogregression):
    line_len = 200
    n_shown_predictions = 8
    model = "gpt3"

    def construct(self):
        # Setup
        self.fake_machine = VectorizedPoint().replicate(3)
        full_string = "Harry Potter was a highly unusual boy"

        # Draw last layer vectors
        last_layer = VGroup(
            NumericEmbedding(length=10)
            for n in range(12)
        )
        last_layer.arrange(RIGHT, buff=0.35 * last_layer[0].get_width())
        last_layer.set_height(3)
        last_layer.to_edge(DOWN)
        # self.add(last_layer)

        rects = VGroup(map(SurroundingRectangle, last_layer))
        rects.set_stroke(YELLOW, 2)
        arrows = VGroup(Vector(0.5 * UP).next_to(rect, UP, buff=0.1) for rect in rects)
        arrows.set_stroke(YELLOW)

        # Show prediction groups
        words = full_string.split(" ")
        substrings = [
            " ".join(words[:n + 1])
            for n in range(len(words))
        ]

        predictions = VGroup(
            self.get_prediction_group(substring)
            for substring in substrings
        )
        predictions[0].to_edge(UP, buff=1.25).align_to(rects[1], LEFT)
        for prediction, arrow, rect in zip(predictions, arrows, rects):
            prediction.move_to(predictions[0], LEFT)
            arrow.become(Arrow(
                rect.get_top(),
                prediction[1].get_left(),
            ))
            arrow.set_stroke(YELLOW)

        last_group = VGroup(
            rects[0].copy().set_opacity(0),
            arrows[0].copy().set_opacity(0),
            predictions[0].copy().set_opacity(0),
        )
        for rect, arrow, prediction in zip(rects, arrows, predictions):
            self.remove(last_group)
            self.play(
                TransformFromCopy(last_group[0], rect),
                TransformFromCopy(last_group[1], arrow),
                TransformMatchingStrings(last_group[2][0].copy(), prediction[0], run_time=1),
                FadeTransform(last_group[2][1].copy(), prediction[1]),
                FadeTransform(last_group[2][2].copy(), prediction[2]),
            )
            self.wait()
            last_group = VGroup(rect, arrow, prediction)

    def get_prediction_group(self, text):
        words, probs = self.predict_next_token(text)
        dist = self.get_distribution(
            words, probs,
            width_100p=2.0
        )
        dist.set_max_height(2.5)
        brace = Brace(dist, LEFT)
        prefix = Text(text, font_size=30)
        prefix.next_to(brace, LEFT)

        result = VGroup(prefix, brace, dist)

        return result


class PeekUnderTheHood(SimpleAutogregression):
    def construct(self):
        # Add parts
        text_mob, next_word_line, machine = self.init_text_and_machine()
        blocks, label, arrow = machine
        self.remove(text_mob, next_word_line)

        # Zoom in
        self.camera.light_source.move_to([-15, 5, 10])
        self.set_floor_plane("xz")

        blocks.rotate(-5 * DEGREES, UP, about_edge=OUT)
        blocks.rotate(-10 * DEGREES, RIGHT, about_edge=OUT)
        blocks.target = blocks.generate_target()
        blocks.target.set_height(5)
        blocks.target.center()
        blocks.target[5:].set_opacity(0.3)

        self.play(
            self.frame.animate.reorient(-23, -12, 0, (1.79, -0.56, 1.27), 8.40).set_anim_args(run_time=3),
            MoveToTarget(blocks, run_time=3),
            FadeOut(arrow, RIGHT),
            FadeOut(label, 2 * OUT),
        )
        self.wait()

        blocks[5:].set_opacity(0.3)

        # Add matrices
        matrices = VGroup(WeightMatrix(shape=(8, 8)) for x in range(9))
        matrices.arrange_in_grid(h_buff_ratio=0.25, v_buff_ratio=0.4)
        matrices.match_width(blocks)
        index = 6
        matrices.move_to(blocks[index], OUT)
        self.add(matrices, blocks[index:])