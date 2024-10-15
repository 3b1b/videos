from manim_imports_ext import *
from _2024.transformers.auto_regression import *
from _2024.transformers.helpers import *
from _2024.transformers.embedding import *


class PredictTheNextWord(SimpleAutogregression):
    text_corner = 3.5 * UP + 6.5 * LEFT
    machine_name = "Large\nLanguage\nModel"
    seed_text = "Paris is a city in"
    model = "gpt3"
    n_shown_predictions = 12
    random_seed = 2

    def construct(self):
        # Setup machine
        text_mob, next_word_line, machine = self.init_text_and_machine()
        machine.move_to(ORIGIN)
        machine[1].set_backstroke(BLACK, 3)

        text_group = VGroup(text_mob, next_word_line)
        text_group.save_state()
        text_group.scale(1.5)
        text_group.match_x(machine[0]).to_edge(UP)

        # Introduce the machine
        in_arrow = Arrow(text_group, machine[0].get_top(), thickness=5)
        frame = self.frame
        self.set_floor_plane("xz")
        blocks = machine[0]
        llm_text = machine[1]
        block_outlines = blocks.copy()
        block_outlines.set_fill(opacity=0)
        block_outlines.set_stroke(GREY_B, 2)
        block_outlines.insert_n_curves(20)

        dials = VGroup(
            Dial().get_grid(8, 12).set_width(0.9 * block.get_width()).move_to(block)
            for block in blocks
        )
        dials.set_stroke(opacity=0.5)
        for group in dials:
            for dial in group:
                dial.set_value(dial.get_random_value())
        flat_dials = VGroup(*it.chain(*dials))
        last_dials = dials[-1].copy()
        last_dials.set_stroke(opacity=0.1)

        self.clear()
        frame.reorient(-31, -4, -5, (-0.24, -0.26, -0.06), 3)
        self.play(
            FadeIn(blocks, shift=0.0, lag_ratio=0.01),
            LaggedStartMap(VShowPassingFlash, block_outlines.family_members_with_points(), time_width=2.0, lag_ratio=0.01, remover=True),
            LaggedStartMap(VFadeInThenOut, flat_dials, lag_ratio=0.001, remover=True),
            Write(llm_text, time_span=(2, 4), stroke_color=WHITE),
            FadeIn(last_dials, time_span=(4, 5)),
            frame.animate.reorient(0, 0, 0, (-0.17, -0.12, 0.0), 4.50),
            run_time=6,
        )
        blocks[-1].add(last_dials)
        self.play(
            frame.animate.to_default_state(),
            FadeIn(text_group, UP),
            GrowFromCenter(in_arrow),
            run_time=3
        )

        # Single word prediction
        out_arrow = Vector(1.5 * RIGHT, thickness=5)
        out_arrow.next_to(machine[0][-1], RIGHT)
        prediction = Text("France", font_size=72)
        prediction.next_to(out_arrow, RIGHT)

        self.animate_text_input(
            text_mob, machine,
            position_text_over_machine=False,
        )
        self.play(
            LaggedStart(
                (TransformFromCopy(VectorizedPoint(machine.get_right()), letter)
                for letter in prediction),
                lag_ratio=0.05,
            ),
            GrowArrow(out_arrow)
        )
        self.wait()
        machine.replace_submobject(2, out_arrow)

        # Probability distribution
        self.play(FadeOut(prediction, DOWN))
        bar_groups = self.animate_prediction_ouptut(machine, self.cur_str)
        self.wait()

        # Show auto_regression
        self.play(
            Restore(text_group),
            FadeOut(in_arrow),
        )

        seed_label = Text("Seed text")
        seed_label.set_color(YELLOW)
        seed_label.next_to(text_mob, DOWN)

        self.play(
            FadeIn(seed_label, rate_func=there_and_back_with_pause),
            FlashAround(text_mob, time_width=2),
            frame.animate.reorient(0, 0, 0, (0.7, -0.01, 0.0), 8.52),
            run_time=2,
        )

        self.animate_random_sample(bar_groups)
        new_text_mob = self.animate_word_addition(
            bar_groups, text_mob, next_word_line,
        )

        # More!
        for n in range(20):
            text_mob = self.new_selection_cycle(
                text_mob, next_word_line, machine,
                quick=True,
                skip_anims=(n > 5),
            )
            self.wait(0.25)


class LotsOfTextIntoTheMachine(PredictTheNextWord):
    run_time = 25
    max_snippet_width = 3

    def construct(self):
        # Add machine
        text_mob, next_word_line, machine = self.init_text_and_machine()
        machine.scale(1.5)
        self.clear()
        self.add(machine)

        blocks, title = machine[:2]
        dials = Dial().get_grid(8, 12).set_width(0.9 * blocks[-1].get_width()).move_to(blocks[-1])
        dials.set_stroke(opacity=0.1)
        blocks[-1].add(dials)

        machine.center()
        machine[1].set_stroke(BLACK, 3)

        # Feed in lots of text
        snippets = self.get_text_snippets()
        text_mobs = VGroup(get_paragraph(snippet.split(" "), line_len=25) for snippet in snippets)
        directions = compass_directions(12, start_vect=UR)
        for text_mob, vect in zip(text_mobs, it.cycle(directions)):
            text_mob.set_max_width(self.max_snippet_width)
            text_mob.move_to(5 * vect).shift_onto_screen(buff=0.25)

        self.play(
            LaggedStart(
                (Succession(
                    FadeIn(text_mob),
                    text_mob.animate.set_opacity(0).move_to(machine.get_center()),
                )
                for text_mob in text_mobs),
                lag_ratio=0.05,
                run_time=self.run_time
            )
        )
        self.remove(text_mobs)
        self.wait()

    def get_text_snippets(self):
        facts = Path(DATA_DIR, "pile_of_text.txt").read_text().split("\n")
        random.shuffle(facts)
        return facts


class EvenMoreTextIntoMachine(LotsOfTextIntoTheMachine):
    run_time = 40
    max_snippet_width = 2.5
    n_examples = 300
    context_size = 25

    def get_text_snippets(self):
        book = Path(DATA_DIR, "tale_of_two_cities.txt").read_text()
        book = book.replace("\n", " ")
        words = list(filter(lambda m: m, book.split(" ")))
        context_size = self.context_size
        result = []
        for n in range(self.n_examples):
            index = random.randint(0, len(words) - context_size - 1)
            result.append(" ".join(words[index:index + context_size]))

        return result


class WriteTransformer(InteractiveScene):
    def construct(self):
        text = Text("Transformer", font_size=120)
        self.play(Write(text))
        self.wait()


class LabelVector(InteractiveScene):
    def construct(self):
        brace = Brace(Line(UP, DOWN).set_height(4), RIGHT)
        name = Text("Vector", font_size=72)
        name.next_to(brace, RIGHT)
        name.set_backstroke(BLACK, 5)

        self.play(
            GrowFromCenter(brace),
            Write(name),
        )
        self.wait()


class AdjustingTheMachine(InteractiveScene):
    def construct(self):
        # Add a machine and repeatedly tweak it
        frame = self.frame
        self.set_floor_plane("xz")
        frame.reorient(-28, -17, 0, ORIGIN, 8.91)
        self.camera.light_source.move_to([-10, 10, 10])

        machine = MachineWithDials(n_rows=10, n_cols=12)
        machine.set_height(6)
        blocks = VCube().replicate(10)
        blocks.set_shape(machine.get_width(), machine.get_height(), 1.0)
        blocks.deactivate_depth_test()
        cam_loc = self.frame.get_implied_camera_location() 
        for block in blocks:
            block.sort(lambda p: -get_norm(p - cam_loc))
        blocks.set_fill(GREY_D, 1)
        blocks.set_shading(0.2, 0.5, 0.25)
        blocks.arrange(OUT, buff=0.5)
        blocks.move_to(machine, OUT)

        self.add(blocks)
        self.add(machine)

        frame.clear_updaters()
        frame.add_updater(lambda f: f.set_theta(-30 * DEGREES * math.cos(0.1 * self.time)))
        self.add(frame)
        for x in range(6):
            self.play(machine.random_change_animation(lag_factor=0.1))


class FirthQuote(InteractiveScene):
    def construct(self):
        # Show Quote
        quote = TexText(R"``You shall know a word\\by the company it keeps!''", font_size=60)
        image = ImageMobject("JohnRFirth")  # From https://www.cambridge.org/core/journals/bulletin-of-the-school-of-oriental-and-african-studies/article/john-rupert-firth/D926AFCBF99AD17D5C7A7A9C0558DFDC
        image.set_height(6.5)
        image.to_corner(UL, buff=0.5)
        name = Text("John R. Firth")
        name.next_to(image, DOWN)
        quote.move_to(midpoint(image.get_right(), RIGHT_SIDE))
        quote.to_edge(UP)

        self.play(
            FadeIn(image, 0.25 * UP),
            FadeIn(name, lag_ratio=0.1)
        )
        self.play(Write(quote))
        self.wait()

        # Show two sentences
        phrases = VGroup(
            Text("Down by the river bank"),
            Text("Deposit a check at the bank"),
        )
        bank = Text("bank", font_size=90)
        bank.set_color(TEAL)
        bank.match_x(quote).match_y(image)
        for phrase in phrases:
            phrase["bank"].set_color(TEAL)

        phrases.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        phrases.next_to(quote, DOWN, buff=2.5)
        phrases[1].set_opacity(0.15)
        banks = VGroup(
            phrase["bank"][0]
            for phrase in phrases
        )

        self.play(
            FadeIn(bank, scale=2, lag_ratio=0.25),
            quote.animate.scale(0.7, about_edge=UP).set_opacity(0.75)
        )
        self.wait()
        self.remove(bank)
        self.play(
            FadeIn(phrases[0][:len("downbytheriver")], lag_ratio=0.1),
            FadeIn(phrases[1][:len("depositacheckatthe")], lag_ratio=0.1),
            *(TransformFromCopy(bank, bank2) for bank2 in banks)
        )
        self.wait()
        self.play(
            phrases[0].animate.set_opacity(0.5),
            phrases[1].animate.set_opacity(1),
        )
        self.wait()

        # Isolate both phrases
        self.play(LaggedStart(
            FadeOut(image, LEFT, scale=0.5),
            FadeOut(name, LEFT, scale=0.5),
            FadeOut(quote, LEFT, scale=0.5),
            phrases.animate.set_opacity(1).arrange(DOWN, buff=3.5, aligned_edge=LEFT).move_to(0.5 * UP),
        ))
        self.wait()

        # Show influence
        query_rects = VGroup(
            SurroundingRectangle(bank)
            for bank in banks
        )
        query_rects.set_stroke(TEAL, 2)
        query_rects.set_fill(TEAL, 0.25)
        key_rects = VGroup(
            SurroundingRectangle(phrases[0]["river"]),
            SurroundingRectangle(phrases[1]["Deposit"]),
            SurroundingRectangle(phrases[1]["check"]),
        )
        key_rects.set_stroke(BLUE, 2)
        key_rects.set_fill(BLUE, 0.5)
        key_rects[2].match_height(key_rects[1], about_edge=UP, stretch=True)
        arrows = VGroup(
            Arrow(key_rects[0].get_top(), banks[0].get_top(), path_arc=-180 * DEGREES, buff=0.1),
            Arrow(key_rects[1].get_top(), banks[1].get_top(), path_arc=-90 * DEGREES),
            Arrow(key_rects[2].get_top(), banks[1].get_top(), path_arc=-90 * DEGREES),
        )
        arrows.set_color(BLUE)

        key_rects.save_state()
        key_rects[0].become(query_rects[0])
        key_rects[1].become(query_rects[1])
        key_rects[2].become(query_rects[1])
        key_rects.set_opacity(0)

        self.add(query_rects, phrases)
        self.play(FadeIn(query_rects, lag_ratio=0.25))
        self.wait()

        self.add(key_rects, phrases)
        self.play(Restore(key_rects, lag_ratio=0.1, path_arc=PI / 4, run_time=2))
        self.play(LaggedStartMap(Write, arrows, stroke_width=5, run_time=3))
        self.wait()

        # Show images
        images = Group(
            ImageMobject("RiverBank"),
            ImageMobject("FederalReserve"),
        )
        for image, bank in zip(images, banks):
            image.set_height(2.0)
            image.next_to(bank, DOWN, MED_SMALL_BUFF, aligned_edge=LEFT)

        self.play(
            LaggedStart(
                (FadeTransform(Group(word).copy(), image)
                for word, image in zip(banks, images)),
                lag_ratio=0.5,
                group_type=Group,
            )
        )
        self.wait(2)


class DownByTheRiverHeader(InteractiveScene):
    def construct(self):
        words = Text("Down by the river bank ...")
        rect = SurroundingRectangle(words["bank"])
        rect.set_fill(BLUE, 0.5)
        rect.set_stroke(BLUE, 3)
        brace = Brace(rect, DOWN, buff=SMALL_BUFF)
        self.add(rect, words, brace)


class RiverBankProbParts(SimpleAutogregression):
    seed_text = "Down by the river bank, "
    model = "gpt3"

    def construct(self):
        # Test
        text_mob, next_word_line, machine = self.init_text_and_machine()
        machine.set_x(0)
        words = [
            ",",
            "there",
            "where",
            "a",
            "the",
            "we",
            "in",
            "the",
            "I",
            "through",
        ]
        probs = softmax([6, 5, 4, 4, 3.5, 3.25, 3, 3, 2.5, 2])
        bar_groups = self.get_distribution(words, probs, machine)

        self.clear()
        bar_groups.set_height(6).center()
        self.play(
            LaggedStartMap(FadeIn, bar_groups, shift=0.25 * DOWN, run_time=3)
        )
        self.wait()


class FourStepsWithParameters(InteractiveScene):
    def construct(self):
        # Add rectangles and titles
        self.add(FullScreenRectangle(fill_color=GREY_E))
        rects = Square().replicate(4)
        rects.arrange(RIGHT, buff=0.25 * rects[0].get_width())
        rects.set_width(FRAME_WIDTH - 1.0)
        rects.center().to_edge(UP, buff=0.5)
        rects.set_fill(BLACK, 1)
        rects.set_stroke(WHITE, 2)
        names = VGroup(*map(TexText, [
            R"Text snippets\\$\downarrow$\\Vectors",
            R"Attention",
            R"Feedforward",
            R"Final prediction",
        ]))
        for name, rect in zip(names, rects):
            name.scale(0.8)
            name.next_to(rect, DOWN)

        self.add(rects)
        self.play(LaggedStartMap(FadeIn, names, shift=0.25 * DOWN, lag_ratio=0.25))
        self.wait()

        # Show many dials
        machines = VGroup(
            MachineWithDials(
                width=rect.get_width(),
                height=3.0,
                n_rows=9,
                n_cols=6,
            )
            for rect in rects
        )
        for machine, rect in zip(machines, rects):
            machine.next_to(rect, DOWN, buff=0)
            machine[0].set_opacity(0)
            machine.scale(rect.get_width() / machine.dials.get_width(), about_edge=UP)
            machine.dials.shift(0.25 * UP)
            for dial in machine.dials:
                dial.set_value(0)

        self.play(
            LaggedStart((
                LaggedStart(
                    (GrowFromPoint(dial, machine.get_top())
                    for dial in machine.dials),
                    lag_ratio=0.025,
                )
                for machine in machines
            ), lag_ratio=0.25),
            LaggedStartMap(FadeOut, names)
        )
        for _ in range(2):
            self.play(
                LaggedStart(
                    (machine.random_change_animation()
                    for machine in machines),
                    lag_ratio=0.2,
                )
            )


class ChatbotFeedback(InteractiveScene):
    random_seed = 404

    def construct(self):
        # Test
        self.frame.set_height(10).move_to(DOWN)
        user_prompt = "User: How and when was the internet invented?"

        prompt_mob = Text(user_prompt)
        prompt_mob.to_edge(UP)
        prompt_mob["User:"].set_color(BLUE)

        self.answer_mob = Text("AI Assistant:")
        self.answer_mob.next_to(prompt_mob, DOWN, buff=1.0, aligned_edge=LEFT)
        self.answer_mob.set_color(YELLOW)
        self.og_answer_mob = self.answer_mob

        self.add(prompt_mob, self.answer_mob)

        # Show multiple answer
        for n in range(8):
            self.give_answer(prompt_mob)
            mark = self.judge_answer()
            self.add(self.og_answer_mob)
            self.play(FadeOut(self.answer_mob), FadeOut(mark))
            self.answer_mob = self.og_answer_mob

    def display_answer(self, text):
        new_answer_mob = get_paragraph(text.replace("\n", " ").split(" "))
        new_answer_mob[:len(self.og_answer_mob)].match_style(self.og_answer_mob)
        new_answer_mob.move_to(self.og_answer_mob, UL)
        self.remove(self.answer_mob)
        self.answer_mob = new_answer_mob
        self.add(self.answer_mob)

    def give_answer(self, prompt_mob, max_responses=100):
        answer = self.og_answer_mob.get_text()
        user_prompt = prompt_mob.get_text()
        for n in range(max_responses):
            answer, stop = self.add_to_answer(user_prompt, answer)
            if stop:
                break
            self.display_answer(answer)
            self.wait(2 / 30)

    def judge_answer(self):
        mark = random.choice([
            Checkmark().set_color(GREEN),
            Exmark().set_color(RED),
        ])
        mark.scale(5)
        mark.next_to(self.answer_mob, RIGHT, aligned_edge=UP)
        rect = SurroundingRectangle(self.answer_mob)
        rect.match_color(mark)
        self.play(FadeIn(mark, scale=2), FadeIn(rect, scale=1.05))
        self.wait()
        return VGroup(mark, rect)

    def add_to_answer(self, user_prompt: str, answer: str):
        try:
            tokens, probs = gpt3_predict_next_token("\n\n".join([user_prompt, answer]))
            token = random.choices(tokens, np.array(probs) / sum(probs))[0]
        except IndexError:
            return answer, True

        stop = False
        if token == '<|endoftext|>':
            stop = True
        else:
            answer += token
        return answer, stop


class ContrastWithEarlierFrame(InteractiveScene):
    def construct(self):
        # Test
        vline = Line(UP, DOWN)
        vline.set_height(FRAME_HEIGHT)
        self.add(vline)

        titles = VGroup(
            VGroup(
                Text("Most earlier models"),
                # Vector(0.75 * DOWN, thickness=4),
                # Text("One word at a time")
            ),
            VGroup(
                Text("Transformers"),
                # Vector(0.75 * DOWN, thickness=4),
                # Text("All words in parallel")
            ),
        )
        for title, vect in zip(titles, [LEFT, RIGHT]):
            title.arrange(DOWN, buff=0.2)
            title.scale(1.5)
            title.move_to(FRAME_WIDTH * vect / 4)
            title.to_edge(UP)

        self.add(titles)


class SequentialProcessing(InteractiveScene):
    def construct(self):
        # Add text
        text = Text("Down by the river bank, where I used to go fishing ...")
        text.move_to(1.0 * DOWN)
        words = break_into_words(text)
        rects = get_piece_rectangles(words)
        blocks = VGroup(VGroup(rect, word) for rect, word in zip(rects, words))
        blocks.save_state()
        self.add(blocks)

        # Vector wandering over
        vect = NumericEmbedding()
        vect.set_width(1.0)
        vect.next_to(rects[0], UP)

        for n in range(len(blocks) - 1):
            blocks.target = blocks.saved_state.copy()
            blocks.target[:n].fade(0.75)
            blocks.target[n + 1:].fade(0.75)
            self.play(
                vect.animate.next_to(blocks[n], UP),
                MoveToTarget(blocks)
            )
            self.play(
                LaggedStart(
                    (ContextAnimation(elem, blocks[n][1], lag_ratio=0.01)
                    for elem in vect.get_entries()),
                    lag_ratio=0.01,
                ),
                RandomizeMatrixEntries(vect),
                run_time=2
            )
