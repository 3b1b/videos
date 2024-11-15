from manim_imports_ext import *
from _2024.transformers.generation import *
from _2024.transformers.helpers import *
from _2024.transformers.embedding import *
from _2024.transformers.ml_basics import *


# Intro

class HoldUpThumbnail(TeacherStudentsScene):
    def construct(self):
        # Test
        im = ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2024/transformers/Thumbnails/Chapter5_TN3.png")
        im_group = Group(
            SurroundingRectangle(im, buff=0).set_stroke(WHITE, 3),
            im
        )
        im_group.set_height(3)
        im_group.move_to(self.hold_up_spot, DOWN)

        morty = self.teacher
        stds = self.students

        self.play(
            FadeIn(im_group, UP),
            morty.change("raise_right_hand", look_at=im_group),
            self.change_students("tease", "happy", "tease", look_at=im_group),
        )
        self.wait(4)


class IsThisUsefulToShare(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        self.play(
            morty.says("Do you find\nthis useful?"),
            self.change_students("pondering", "hesitant", "well", look_at=self.screen)
        )
        self.wait(3)
        self.play(self.change_students("thinking", "pondering", "tease"))
        self.wait(3)


class AskAboutAttention(TeacherStudentsScene):
    def construct(self):
        # Test
        stds = self.students
        morty = self.teacher
        self.play(
            morty.change("tease"),
            stds[2].says("Can you explain what\nAttention does?", mode="raise_left_hand", bubble_direction=LEFT),
            stds[1].change("pondering", self.screen),
            stds[0].change("pondering", self.screen),
        )
        self.wait(4)


# Version 1

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

        flat_dials, last_dials = self.get_machine_dials(blocks)

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

    def get_machine_dials(self, blocks):
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

        return flat_dials, last_dials


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

        # Recreate
        word = Text("bank", font_size=72)
        word.set_color(TEAL)
        self.clear()

        self.add(word)
        self.wait()
        self.remove(word)
        self.play(
            *(
                FadeIn(phrase[phrase.get_text().replace("bank", "")])
                for phrase in phrases
            ),
            *(
                TransformFromCopy(word, phrase["bank"][0])
                for phrase in phrases
            )
        )
        self.add(phrases)

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
            "water",
            "river",
            "lake",
            "grass",
            "waves",
            "shallows",
            "pool",
            "depths",
            "foam",
            "mist",
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


# Version 2


class PartialScript(SimpleAutogregression):
    machine_name = "Magic next\nword predictor"
    machine_phi = 5 * DEGREES
    machine_theta = 6 * DEGREES

    def construct(self):
        # Set frame
        frame = self.frame
        self.set_floor_plane("xz")

        # Unfurl script
        curled_script_img = ImageMobject("HumanAIScript")
        curled_script_img.set_height(7)

        curves = VGroup(SVGMobject("JaggedCurl1")[0], SVGMobject("JaggedCurl2")[0])
        for curve in curves:
            curve.make_smooth(approx=False)
            curve.insert_n_curves(100)
            curve.set_stroke(WHITE, 3)
            curve.set_fill(opacity=0)
            curve.set_height(5)
        curves[1].scale(curves[0].get_arc_length() / curves[1].get_arc_length())

        resolution = (2, 200)  # Change
        surface_kw = dict(u_range=(-6, 6), v_range=(0.05, 0.95), resolution=resolution)
        curled_script_templates = Group(
            ParametricSurface(
                lambda u, v: (*curve.pfp(v)[:2], u),
                **surface_kw
            )
            for curve in curves
        )
        curled_script_templates[1].rotate(PI / 2, UP)
        curled_script_templates[0].rotate(-PI / 2)
        flat_script_template = ParametricSurface(
            lambda u, v: (u, v, 0),
            **surface_kw
        )
        curled_script0 = TexturedSurface(curled_script_templates[0], "HumanAIScript")
        curled_script1 = TexturedSurface(curled_script_templates[1], "HumanAIScript")
        curled_script1_torn = TexturedSurface(curled_script_templates[1], "HumanAIScriptTorn")
        flat_script = TexturedSurface(flat_script_template, "HumanAIScriptTorn")
        flat_script.replace(curled_script_img, stretch=True)

        for script in [curled_script0, curled_script1]:
            script.set_shading(0.25, 0.25, 0.35)
        curled_script1_torn.set_shading(0, 0, 0)
        flat_script.set_shading(0, 0, 0)

        frame.reorient(0, -1, 0, (-0.28, 0.69, 0.0), 14.43)
        self.play(
            TransformFromCopy(curled_script0, curled_script1),
            frame.animate.reorient(56, -17, 0, (-0.2, -1.52, -2.39), 20.05),
            run_time=3
        )
        self.play(
            frame.animate.reorient(-6, -11, 0, (1.06, -1.22, -2.65), 20.05),
            run_time=8,
        )
        self.play(
            FadeOut(curled_script1, shift=1e-2 * IN),
            FadeIn(curled_script1_torn, shift=1e-2 * IN),
        )
        self.play(
            ReplacementTransform(curled_script1_torn, flat_script),
            frame.animate.to_default_state(),
            run_time=2
        )
        self.wait()

        # Show the machine
        machine = self.get_transformer_drawing()
        machine[1].set_height(0.7).set_stroke(width=2)
        machine[1].set_opacity(0)
        machine.remove(machine[-1])
        machine.set_height(3)
        machine.to_edge(RIGHT)

        self.play(
            flat_script.animate.set_height(5).to_edge(LEFT),
            FadeIn(machine, lag_ratio=0.01)
        )
        self.add(machine)
        self.wait()

        # Show example input and output
        out_arrow = Vector(DOWN, thickness=6)
        out_arrow.next_to(machine, DOWN)
        in_arrow = out_arrow.copy().next_to(machine, UP, SMALL_BUFF)
        in_text = Text("To be or not to _")
        in_text[-1].stretch(3, 0, about_edge=LEFT)
        in_text.next_to(in_arrow, UP)
        prediction = Text("be", font_size=72)
        prediction.next_to(out_arrow, DOWN)

        self.play(FadeIn(in_text), GrowArrow(in_arrow))
        self.animate_text_input(in_text, machine, position_text_over_machine=False)
        self.play(
            GrowArrow(out_arrow),
            FadeIn(prediction, DOWN),
        )
        self.wait()

        # Clear the board
        script_text = self.get_text()
        script_text.set_width(0.89 * flat_script.get_width())
        script_text.next_to(flat_script.get_top(), DOWN, buff=0.33)

        font_size = 48 * (script_text[0].get_height() / Text("H").get_height())
        completion = "A transistor is a semiconductor device used to amplify or switch electronic signals. It consists of three layers of semiconductor material, either p-type or n-type, forming a structure with terminals called the emitter, base, and collector."
        words = completion.split(" ")
        paragraph = get_paragraph(completion.split(" "), font_size=font_size)
        paragraph.next_to(script_text, DOWN, aligned_edge=LEFT)
        paragraph.set_color(YELLOW)

        self.play(
            FadeIn(script_text),
            FadeOut(flat_script),
            FadeOut(VGroup(in_text, in_arrow, prediction)),
        )

        # Repeatedly add predictions
        machine.scale(1.25, about_edge=RIGHT)
        out_arrow.next_to(machine, DOWN, buff=0.5)

        blocks = machine[0]
        dials = Dial().get_grid(11, 16)
        dials.set_width(blocks[-1].get_width() * 0.95)
        dials.rotate(5 * DEGREES, RIGHT).rotate(10 * DEGREES, UP)
        dials.move_to(blocks[-1])
        dials.set_stroke(opacity=0.5)
        for dial in dials:
            dial.set_value(dial.get_random_value())
        dials.set_z_index(2)
        self.add(dials)

        curr_answer = VGroup()
        curr_answer.next_to(script_text, DOWN)
        for n in range(6):
            word = words[n]
            prediction = Text(words[n], font_size=72)
            prediction.next_to(out_arrow, DOWN)
            word_in_answer = paragraph[len(curr_answer):len(curr_answer) + len(word)]
            word_in_answer.set_color(YELLOW)
            mover = VGroup(script_text, curr_answer).copy()

            if n > 2:
                self.skip_animations = True

            self.play(
                mover.animate.set_height(1.8).next_to(machine, UP, SMALL_BUFF).set_anim_args(path_arc=-30 * DEGREES),
            )
            self.animate_text_input(
                mover, machine,
                position_text_over_machine=False,
                lag_ratio=1e-3
            )
            self.play(FadeIn(prediction, DOWN, rate_func=rush_from, run_time=0.5))

            if n > 2:
                self.skip_animations = False
                self.wait(0.5)
                self.skip_animations = True

            self.play(
                curr_answer.animate.set_color(WHITE),
                Transform(prediction, word_in_answer),
                FadeOut(mover),
            )
            curr_answer.add(*word_in_answer)
            self.add(curr_answer)
            self.remove(prediction)

    def get_text(self):
        script_text = Text("""
            Human:
            Can you explain the history of
            transistors and how they're relevant
            to computers? What is a transistor,
            and how exactly is it used to
            perform computations?

            AI assistant:
        """, alignment="LEFT")
        script_text["Human"].set_color(BLUE)
        script_text["AI assistant"].set_color(TEAL)

        script_text.set_height(4).to_edge(UP)
        return script_text

    def create_image(self):
        # Create image
        script_text = self.get_text()
        script_text.set_fill(BLACK)
        script_text["Human"].set_fill(BLUE_D)
        script_text["AI assistant"].set_fill(TEAL_D)
        self.add(FullScreenRectangle(fill_color="#FCF5E5", fill_opacity=1))
        self.add(script_text)

        # Add off test
        tear_off = SVGMobject('TearOff')
        tear_off.set_stroke(width=0)
        tear_off.set_fill(BLACK, 1)
        tear_off.set_width(7.5)
        tear_off.next_to(script_text, DOWN, buff=-0.2)
        self.add(tear_off)


class ShowMachineWithDials(PredictTheNextWord):
    words = ['worst', 'age', 'worse', 'best', 'most', 'end', 'very', 'blur']
    logprobs = [4.0, 2.15, 1.89, 1.4, 0.1, -0.18, -0.23, -0.61]

    def construct(self):
        # Show machine (same position as in PredictTheNextWord)
        frame = self.frame
        self.set_floor_plane("xz")
        blocks, llm_text, flat_dials, last_dials = self.get_blocks_and_dials()

        self.clear()
        self.add(frame)
        frame.reorient(0, 0, 0, (-0.17, -0.12, 0.0), 4.50)
        self.add(blocks, llm_text, last_dials)

        # Prepare dial highlight
        last_dials.target = last_dials.generate_target()
        self.fix_dials(last_dials.target)

        small_rect = SurroundingRectangle(last_dials[0], buff=0.025)
        small_rect.set_stroke(BLUE, 2)
        big_rect = small_rect.copy().scale(4)
        big_rect.next_to(blocks, UP, buff=SMALL_BUFF, aligned_edge=LEFT + OUT)
        big_rect.shift(1.5 * RIGHT)
        big_dial = last_dials[0].copy().scale(4).set_stroke(opacity=1)
        big_dial.move_to(big_rect)
        rect_lines = VGroup(
            Line(small_rect.get_corner(UL), big_rect.get_corner(DL)),
            Line(small_rect.get_corner(UR), big_rect.get_corner(DR)),
        )
        rect_lines.set_stroke(WHITE, width=(1, 3))
        highlighed_parameter_group = VGroup(small_rect, rect_lines, big_rect, big_dial)

        last_dials.set_stroke(width=1, opacity=1)
        self.play(
            MoveToTarget(last_dials),
            FadeOut(llm_text),
            FadeIn(small_rect),
        )

        # Show an example input and output
        example = self.get_example(blocks)
        in_text, in_arrow, out_arrow, bar_groups = example
        logprobs = example.logprobs
        true_probs = 100 * softmax(logprobs)
        bar_groups = self.get_output_distribution(self.words, 0.1 * logprobs, out_arrow)

        self.play(
            LaggedStart(
                ShowCreation(rect_lines, lag_ratio=0),
                TransformFromCopy(small_rect, big_rect),
                TransformFromCopy(last_dials[0], big_dial),
                FadeIn(in_text),
                GrowArrow(in_arrow),
                FadeIn(bar_groups),
                GrowArrow(out_arrow),
            ),
            frame.animate.reorient(0, 0, 0, (-0.43, 0.38, 0.0), 7.05),
            run_time=2
        )
        self.play(
            last_dials[0].animate_set_value(0.8),
            big_dial.animate_set_value(0.8),
            LaggedStart(
                (dial.animate_set_value(dial.get_random_value())
                for dial in last_dials[1:]),
                lag_ratio=1.0 / len(last_dials),
            ),
            *(
                self.bar_group_change_animation(bg, value)
                for bg, value in zip(bar_groups[:-1], true_probs)
            ),
            run_time=3
        )
        self.wait()

        # Play around tweaking the parameters, and seeing the output change
        self.play(
            LaggedStart(
                (dial.animate_set_value(0)
                for dial in last_dials[:12]),
                lag_ratio=0.01,
            ),
            big_dial.animate_set_value(0),
            self.bar_group_change_animation(bar_groups[0], 50),
            self.bar_group_change_animation(bar_groups[1], 34),
            self.bar_group_change_animation(bar_groups[2], 5),
            run_time=4,
        )
        self.play(
            LaggedStart(
                (dial.animate_set_value(1)
                for dial in last_dials[:12]),
                lag_ratio=0.01,
            ),
            big_dial.animate_set_value(1),
            self.bar_group_change_animation(bar_groups[0], 80),
            self.bar_group_change_animation(bar_groups[1], 5),
            self.bar_group_change_animation(bar_groups[2], 15),
            run_time=4,
        )
        self.wait()

        # Mention randomness
        random_words = Text("Initially random")
        random_words.next_to(blocks, UP)
        random_words.set_color(RED)
        out_dots = Tex(R"...", font_size=120)
        out_dots.next_to(out_arrow, RIGHT)

        self.play(
            FadeOut(big_rect),
            Uncreate(rect_lines, lag_ratio=0),
            FadeOut(small_rect),
            Transform(big_dial, last_dials[0])
        )
        self.play(
            Write(random_words),
            LaggedStart(
                (dial.animate_set_value(dial.get_random_value())
                for dial in last_dials),
                lag_ratio=0.5 / len(last_dials),
                run_time=2
            ),
            FadeOut(bar_groups),
        )
        self.play(Write(out_dots))
        self.wait()
        self.play(
            FadeOut(dots),
            FadeOut(random_words),
            FadeIn(bar_groups),
        )

        # Show many many parameters
        example.save_state()
        blocks.save_state()
        last_dials.save_state()
        all_dials = VGroup(*flat_dials, *last_dials)
        all_dials.generate_target()
        all_dials.target.space_out_submobjects(3)
        new_dials = VGroup(
            all_dials.target.copy().shift(3 * 2 * x * (flat_dials.get_center() - last_dials.get_center()))
            for x in range(1, 9)
        )

        self.play(
            FadeOut(example),
            FadeOut(blocks),
            FadeIn(flat_dials),
            FadeOut(bar_groups),
            FadeOut(out_arrow),
        )
        self.play(
            FadeOut(highlighed_parameter_group),
            MoveToTarget(all_dials),
            LaggedStart(
                (TransformFromCopy(all_dials.copy().set_opacity(0), nd)
                for nd in new_dials),
                lag_ratio=0.05,
            ),
            frame.animate.reorient(-9, 0, 0, (-0.71, -0.07, -0.06), 9.64),
            run_time=4
        )
        self.wait()

    def get_blocks_and_dials(self):
        machine = self.get_transformer_drawing()
        machine.move_to(ORIGIN)
        self.machine = machine

        blocks = machine[0]
        llm_text = machine[1]
        llm_text.set_backstroke(BLACK, 2)
        flat_dials, last_dials = self.get_machine_dials(blocks)
        return blocks, llm_text, flat_dials, last_dials

    def get_example(self, blocks):
        in_text = Text("It was the best\nof times it was\nthe _", alignment="LEFT")
        in_text[-1].stretch(4, 0, about_edge=LEFT)
        in_text.next_to(blocks, LEFT, LARGE_BUFF)
        in_arrow = Arrow(in_text, blocks)

        out_arrow = Vector(RIGHT)
        out_arrow.next_to(blocks[-1], RIGHT, buff=0.1)
        logprobs = np.array(self.logprobs)
        bar_groups = self.get_output_distribution(self.words, logprobs, out_arrow)
        example = VGroup(in_text, in_arrow, out_arrow, bar_groups)
        example.logprobs = logprobs
        return example

    def fix_dials(self, dials):
        for dial in dials:
            dial.set_stroke(width=1, opacity=1)
            dial.needle.set_stroke(width=(2, 0))
        return dials

    def bar_group_change_animation(self, bar_group, new_value):
        text, rect, value_mob = bar_group
        buff = value_mob.get_left() - rect.get_right()
        factor = new_value / value_mob.get_value()

        return AnimationGroup(
            rect.animate.stretch(factor, 0, about_edge=LEFT),
            ChangeDecimalToValue(value_mob, new_value),
            UpdateFromFunc(text, lambda m: value_mob.move_to(rect.get_right() + buff, LEFT)),
        )

    def get_output_distribution(self, words, logprobs, out_arrow):
        probs = softmax(logprobs)
        bar_groups = self.get_distribution(words, probs, self.machine, width_100p=1.0)
        bar_groups.next_to(out_arrow, RIGHT)
        return bar_groups


class ShowSingleTrainingExample(ShowMachineWithDials):
    logprobs = [4.0, 6.15, 1.89, 1.4, 0.1, -0.18, -0.23, -0.61]

    def construct(self):
        # Add state from before
        frame = self.frame
        self.set_floor_plane("xz")

        blocks, llm_text, flat_dials, last_dials = self.get_blocks_and_dials()
        self.fix_dials(last_dials)
        example = self.get_example(blocks)
        in_text, in_arrow, out_arrow, bar_groups = example

        self.add(blocks, last_dials)

        # Show example up top
        parts = ("It was the best of times it was the", "worst")
        sentence = Text(" ".join(parts))
        start = sentence[parts[0]][0]
        end = sentence[parts[1]][0]
        sentence.set_width(10)
        sentence.next_to(blocks, UP, buff=1.5)

        start_rect = SurroundingRectangle(start)
        start_rect.set_stroke(BLUE, 2)
        start_rect.set_fill(BLUE, 0.2)
        end_rect = SurroundingRectangle(end)
        end_rect.match_height(start_rect, stretch=True).match_y(start_rect)
        end_rect.set_stroke(YELLOW, 2)
        end_rect.set_fill(YELLOW, 0.2)
        arrow = Arrow(start_rect.get_top(), end_rect.get_top(), path_arc=-90 * DEGREES, thickness=5)
        arrow.set_fill(border_width=1)

        frame.reorient(0, 0, 0, (-0.36, 0.97, 0.0), 7.52)
        self.play(FadeIn(sentence, UP))
        self.play(
            LaggedStartMap(DrawBorderThenFill, VGroup(start_rect, end_rect)),
            FadeIn(arrow),
        )
        self.remove(last_dials)
        self.play(LaggedStart(
            AnimationGroup(
                TransformFromCopy(start, in_text[:-1]),
                TransformFromCopy(end_rect, in_text[-1]),
                FadeIn(in_arrow)
            ),
            LaggedStart(
                (
                    block.animate.set_color(
                        block.get_color() if block is blocks[-1] else TEAL
                    ).set_anim_args(rate_func=there_and_back)
                    for block in blocks
                ),
                group=blocks,
                lag_ratio=0.1,
                run_time=1
            ),
            Animation(last_dials),
            GrowArrow(out_arrow),
            LaggedStartMap(GrowFromPoint, bar_groups, point=out_arrow.get_start()),
            lag_ratio=0.3
        ))
        self.wait()

        # Flag bad prediction
        out_rects = VGroup(
            SurroundingRectangle(bg)
            for bg in bar_groups[:2]
        )
        out_rects.set_stroke(RED, 3)
        annotations = VGroup(
            Tex(tex, font_size=60).next_to(rect, LEFT, buff=SMALL_BUFF)
            for rect, tex in zip(out_rects, [R"\uparrow", R"\downarrow"])
        )
        annotations.set_color(RED)

        self.play(
            FadeTransform(end_rect.copy(), out_rects[0]),
            Write(annotations[0]),
        )
        self.wait()
        self.play(
            FadeTransform(*out_rects),
            FadeTransform(*annotations),
        )
        self.wait()
        self.play(
            FadeOut(out_rects[1]),
            FadeOut(annotations[1]),
        )

        # Adjust
        self.play(
            LaggedStart(
                (dial.animate_set_value(dial.get_random_value())
                for dial in last_dials),
                lag_ratio=1.0 / len(last_dials),
            ),
            LaggedStart(
                (FlashAround(dial, stroke_width=2, color=YELLOW, time_width=1, buff=0.025) for dial in last_dials),
                lag_ratio=1.0 / len(last_dials),
            ),
            self.bar_group_change_animation(bar_groups[0], 70),
            self.bar_group_change_animation(bar_groups[1], 20),
            self.bar_group_change_animation(bar_groups[2], 8),
            run_time=6
        )


class ParameterWeight(InteractiveScene):
    def construct(self):
        # Test
        text = Text("Parameter / Weight", font_size=72)
        text.to_edge(UP)
        text.set_color(YELLOW)
        param = text["Parameter"][0]
        param.save_state()
        param.set_x(0)

        self.play(Write(param))
        self.wait()
        self.play(LaggedStart(
            Restore(param),
            FadeIn(text["/ Weight"]),
        ))
        self.wait()


class LargeInLargeLanguageModel(InteractiveScene):
    def construct(self):
        # Test
        text = Text("Large Language Model", font_size=72)
        text.to_edge(UP)
        large = text["Large"][0]
        large.save_state()
        large.set_x(0)

        self.add(large)
        self.play(FlashUnder(large), large.animate.set_color(YELLOW))
        self.play(
            Restore(large, path_arc=-30 * DEGREES),
            Write(text[len(large):], time_span=(0.5, 1.5))
        )
        self.wait()


class ThousandsOfWords(InteractiveScene):
    def construct(self):
        # Find passage
        file = Path("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2024/transformers/data/tale_of_two_cities.txt")
        novel = file.read_text()
        start_index = novel.index("It was the best of times")
        end_index = novel.index("There were a king with a large jaw")

        # Add text
        passage = novel[start_index:start_index + 5000].replace("\n", " ")
        text = get_paragraph(passage.split(" "), line_len=150)
        text.set_width(14)
        text.to_edge(UP)
        self.add(text)


class EnormousAmountOfTrainingText(PremiseOfMLWithText):
    def construct(self):
        # Setup
        self.init_data()
        # n_rows = n_cols = 41
        n_rows = n_cols = 9
        screens = VGroup()
        for row in range(n_rows):
            for col in range(n_cols):
                screen = self.get_screen()
                screen.move_to(FRAME_WIDTH * row * RIGHT + FRAME_HEIGHT * col * DOWN)
                screens.add(screen)
        screens.center()
        screens.submobjects.sort(key=lambda sm: get_norm(sm.machine.get_center()))

        self.add(screens)

        # Add frame growth
        frame = self.frame
        frame.clear_updaters()
        frame.add_updater(lambda m: m.set_height(FRAME_HEIGHT * np.exp(0.2 * self.time)))

        # Show lots of new data
        inner_screens = screens[:25]
        n_examples = 20
        for n in range(n_examples):
            self.play(LaggedStart(
                *(self.change_example_animation(screen, show_dial_change=True)
                for screen in inner_screens),
                lag_ratio=0.1,
                run_time=0.5,
            ))

    def change_example_animation(self, screen, show_dial_change=True):
        new_example = VGroup(*self.new_input_output_example(*screen.arrows))
        time_span = (0, 0.35)
        anims = [
            FadeOut(screen.training_example, time_span=time_span),
            FadeIn(new_example, time_span=time_span),
        ]
        if show_dial_change:
            anims.append(screen.machine.random_change_animation(run_time=0.5))
        screen.training_example = new_example
        return AnimationGroup(*anims)

    def get_screen(self):
        border = FullScreenRectangle()
        border.set_fill(opacity=0)
        border.set_stroke(WHITE, 2)

        machine = MachineWithDials(width=3.5, height=2.5, n_rows=5, n_cols=7)
        machine.move_to(1.0 * RIGHT)
        in_arrow, out_arrow = arrows = Vector(RIGHT).replicate(2)
        in_arrow.next_to(machine, LEFT)
        out_arrow.next_to(machine, RIGHT)
        in_data, out_data = training_example = VGroup(
            *self.new_input_output_example(in_arrow, out_arrow)
        )

        screen = VGroup(
            border, machine,
            arrows, training_example
        )
        screen.border = border
        screen.machine = machine
        screen.arrows = arrows
        screen.training_example = training_example

        return screen

    def new_input_output_example(self, in_arrow, out_arrow):
        in_data, out_data = super().new_input_output_example(in_arrow, out_arrow)
        in_data.scale(0.8, about_edge=RIGHT)
        out_data.scale(0.8, about_edge=LEFT)
        return in_data, out_data


class BadChatBot(InteractiveScene):
    def construct(self):
        # Add bot
        bot = self.get_bot()
        bot.set_height(3)

        lines = Line(LEFT, RIGHT).get_grid(4, 1, buff=0.25)
        lines.set_stroke(WHITE, 1)
        lines[-1].stretch(0.5, 0, about_edge=LEFT)
        lines.set_width(3)
        bubble = SpeechBubble(lines, buff=MED_LARGE_BUFF)
        bubble.set_stroke(width=5)
        bubble.pin_to(bot).shift(DOWN)

        self.add(bot)
        self.play(Write(bubble, run_time=3))
        self.blink(bot)
        self.wait()

        # Make lines bad
        self.play(
            LaggedStart(
                (Transform(line, self.get_scribble(line))
                for line in lines),
                lag_ratio=0.1,
                run_time=2
            )
        )
        for _ in range(2):
            self.blink(bot)
            self.wait(2)

    def get_scribble(self, line):
        freqs = np.random.random(5)
        graph = FunctionGraph(
            lambda x: 0.05 * sum(math.sin(freq * TAU * x) for freq in freqs),
            x_range=(0, 5, 0.1)
        )
        graph.put_start_and_end_on(*line.get_start_and_end())
        graph.match_style(line)
        graph.set_stroke(color=RED)
        return graph

    def get_bot(self):
        bot = SVGMobject("Bot")
        subpaths = bot[0].get_subpaths()
        bot[0].set_points([*subpaths[0], subpaths[0][-1], *subpaths[1]])
        eyes = VGroup(Dot().replace(VMobject().set_points(subpath)) for subpath in subpaths[2:])
        bot.eyes = eyes
        bot.add(eyes)
        bot.set_stroke(width=0)

        bot.set_height(4)
        bot.set_fill(GREY_B)
        bot.set_shading(0.5, 0.5, 1)

        return bot

    def blink(self, bot):
        self.play(
            bot.eyes.animate.stretch(0, 1).set_anim_args(rate_func=squish_rate_func(there_and_back))
        )


class WriteRLHF(InteractiveScene):
    def construct(self):
        text = Text("Step 2: RLHF")
        full_text = Text("Reinforcement Learning\nwith Human Feedback")
        full_text.next_to(text, UP, LARGE_BUFF)
        full_text.align_to(text, RIGHT).shift(RIGHT)
        initials = VGroup(full_text[letter[0]][0][0] for letter in "RLHF")
        full_text.remove(*initials)

        self.add(text)
        self.wait()
        self.play(
            TransformFromCopy(text["RLHF"][0], initials, lag_ratio=0.25),
            Write(full_text, time_span=(1.5, 3)),
            run_time=3
        )
        self.wait()


class RLHFWorker(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle().set_fill(GREY_E, 1))
        # worker = SVGMobject("computer_stall")
        worker = SVGMobject("comp_worker")
        worker.set_height(4)
        worker.move_to(4 * LEFT)
        worker.set_fill(GREY_C, 1)

        rect = Rectangle(7, 5)
        rect.to_edge(RIGHT)
        rect.set_stroke(WHITE, 2)
        rect.set_fill(BLACK, 1)

        self.add(worker)
        self.add(rect)


class RLHFWorkers(ShowMachineWithDials):
    def construct(self):
        # Add workers
        self.add(FullScreenRectangle().set_fill(GREY_E, 1))
        workers = SVGMobject("comp_worker").get_grid(3, 2, buff=0.5)
        workers.set_height(7)
        workers.to_edge(LEFT)
        workers.set_fill(GREY_C, 1)

        self.add(workers)

        # Machine
        blocks, llm_text, flat_dials, last_dials = self.get_blocks_and_dials()
        machine = VGroup(blocks, last_dials)
        machine.set_height(4)
        machine.center().to_edge(RIGHT, buff=LARGE_BUFF)
        last_dials.set_stroke(opacity=1)

        self.add(machine)

        for _ in range(8):
            self.play(LaggedStart(
                (dial.animate_set_value(dial.get_random_value())
                for dial in last_dials),
                lag_ratio=0.5 / len(last_dials),
                run_time=2
            ))
            self.wait()


class SerialProcessing(InteractiveScene):
    phrase = "It was the best of times it was the worst of times"
    phrase_center = 2 * UP

    def construct(self):
        # Set up words
        words = self.get_words()
        rects = get_piece_rectangles(words)

        self.add(rects)
        self.add(words)

        # Animate in the vectors
        vectors = VGroup(
            self.get_abstract_vector().next_to(word, DOWN, LARGE_BUFF)
            for word in words
        )
        last_vect = VGroup(VectorizedPoint(rects[0].get_bottom()))

        for word, vect in zip(words, vectors):
            self.play(
                FadeIn(vect, run_time=2),
                LaggedStart(
                    (ContextAnimation(
                        square, VGroup(*word, *last_vect),
                        direction=DOWN,
                        lag_ratio=0.01,
                        path_arc=30 * DEGREES
                    )
                    for square in vect),
                    lag_ratio=0.05,
                    run_time=2
                ),
                last_vect.animate.set_opacity(0.2)
            )
            last_vect = vect

    def get_words(self):
        result = break_into_words(Text(self.phrase))
        result.move_to(self.phrase_center)
        return result

    def get_abstract_vector(self, values=None, default_length=10, elem_size=0.2):
        if values is None:
            values = np.random.uniform(-1, 1, default_length)
        result = Square().get_grid(len(values), 1, buff=0)
        result.set_width(elem_size)
        result.set_stroke(WHITE, 1)
        for square, value in zip(result, values):
            color = value_to_color(value, min_value=0, max_value=1)
            square.set_fill(color, opacity=1)
        return result


class ParallelProcessing(SerialProcessing):
    def construct(self):
        # Set up words
        words = self.get_words()
        rects = get_piece_rectangles(words)

        self.add(rects)
        self.add(words)

        # Animate in the vectors
        vectors = VGroup(
            self.get_abstract_vector().next_to(word, DOWN, buff=1.5)
            for word in words
        )

        lines = VGroup(
            Line(
                rect.get_bottom(), vect.get_top(),
                buff=0.05,
                stroke_color=WHITE,
                stroke_width=2 * random.random()**3
            )
            for rect in rects
            for vect in vectors
        )
        lines.shuffle()

        for vect, word in zip(vectors, words):
            vect.save_state()
            for square in vect:
                square.move_to(word)
                square.set_opacity(0)

        self.play(
            LaggedStartMap(ShowCreation, lines, lag_ratio=0.01),
            LaggedStartMap(Restore, vectors, lag_ratio=0)
        )
        self.play(lines.animate.set_stroke(opacity=0.25))
        self.wait()


class ManyComputationsPerUnitTimeV2(InteractiveScene):
    def construct(self):
        # Add computations
        box = Rectangle(5, 5)
        label = Text("1 Billion computations per Second")
        label.next_to(box, UP)
        self.add(box)
        self.add(label)

        comps = self.get_computations(box)
        self.add(comps)
        self.wait(3)

        # Place box into minute interval
        width = FRAME_WIDTH - 1
        number_lines = VGroup(
            minute_line := NumberLine((0, 60, 1), width=width, big_tick_spacing=10),
            hour_line := NumberLine((0, 60, 1), width=width, big_tick_spacing=10),
            day_line := NumberLine((0, 24, 1), width=width, big_tick_spacing=6),
            month_line := NumberLine((0, 31, 1), width=width),
            year_line := NumberLine((0, 12, 1), width=width),
            y100_line := NumberLine((0, 100, 1), width=width),
            y10k_line := NumberLine((0, 100, 1), width=width),
            y1M_line := NumberLine((0, 100, 1), width=width),
            y100M_line := NumberLine((0, 100, 1), width=width),
        )
        number_lines.move_to(DOWN)

        first_ticks = minute_line.ticks[:2]
        sec_brace = Brace(first_ticks, DOWN, buff=0, tex_string=R"\underbrace{\qquad\qquad}")
        sec_label = Text("Second", font_size=30).next_to(sec_brace, DOWN, SMALL_BUFF)

        self.play(
            ShowCreation(minute_line, lag_ratio=0.01),
            box.animate.match_width(first_ticks).move_to(first_ticks.get_center(), DOWN).set_stroke(width=1),
            TransformFromCopy(label["Second"][0], sec_label),
            GrowFromCenter(sec_brace),
            run_time=2
        )

        # Add other boxes
        minute_label = self.get_timeline_full_label(number_lines[1], "Minute")
        new_boxes = VGroup(
            box.copy().move_to(tick.get_center(), DL)
            for tick in minute_line.ticks[1:-1]
        )
        for new_box in new_boxes:
            new_box.save_state()
            new_box.move_to(box)
        computations = VGroup(
            self.get_computations(new_box, n_iterations=1)
            for new_box in new_boxes
        )
        # computations = VGroup()  # If needed

        self.add(computations)
        self.play(
            FadeIn(minute_label, DOWN),
            LaggedStartMap(Restore, new_boxes, lag_ratio=0.1),
            run_time=2
        )
        self.wait(2)

        # Add labels
        minute_line.add(minute_label)
        names = ["Hour", "Day", "Month", "Year", "100 Years", "10,000 Years", "1,000,000 Years", "100,000,000 Years"]
        for line, name in zip(number_lines[1:], names):
            line.label = self.get_timeline_full_label(line, name)
            line.add(line.label)

        # Arrange all lines
        number_lines[1:].arrange(DOWN, buff=2.0)
        number_lines[1:].next_to(minute_line, DOWN, buff=2.0)

        scale_lines = VGroup()
        for nl1, nl2 in zip(number_lines, number_lines[1:]):
            n = len(nl2.ticks) // 2
            mini_line = Line(nl2.ticks[n - 1].get_center(), nl2.ticks[n].get_center())
            pair = VGroup(
                DashedLine(nl1.get_start(), mini_line.get_start()),
                DashedLine(nl1.get_end(), mini_line.get_end()),
            )
            pair.set_stroke(WHITE, 2)
            nl1.target = nl1.copy()
            nl1.target.replace(mini_line, dim_to_match=0)
            nl1.target.shift(mini_line.pfp(0.5) - nl1.target.pfp(0.5))
            scale_lines.add(pair)

        # Start panning down
        lag_ratio = 1.5
        self.play(
            LaggedStart(
                *(AnimationGroup(*(ShowCreation(sl) for sl in pair)) for pair in scale_lines),
                lag_ratio=lag_ratio,
            ),
            LaggedStart(
                *(FadeIn(nl) for nl in number_lines[1:]),
                lag_ratio=lag_ratio,
            ),
            LaggedStart(
                *(TransformFromCopy(nl, nl.target) for nl in number_lines[:-1]),
                lag_ratio=lag_ratio,
            ),
            self.frame.animate.set_y(number_lines[-1].get_y() + 2).set_width(18).set_anim_args(
                rate_func=lambda t: interpolate(smooth(t), linear(t), there_and_back_with_pause(t, pause_ratio=0.8))
            ),
            run_time=30
        )
        self.play(self.frame.animate.reorient(0, 0, 0, (-0.03, -11.55, 0.0), 31.76), run_time=4)
        self.wait(4)

    def fade_in_bigger_interval(self, new_interval, prev_interval, fader, scale_factor, added_anims=[]):
        pivot = prev_interval.n2p(0)
        new_interval.save_state()
        new_interval.scale(scale_factor, about_point=pivot)
        new_interval[:-1].set_opacity(0)
        new_interval[-1].set_fill(BLACK)

        self.play(
            Restore(new_interval),
            prev_interval.animate.scale(1.0 / scale_factor, about_point=pivot).set_fill(border_width=0),
            fader.animate.scale(1.0 / scale_factor, about_point=pivot).set_opacity(0),
            *added_anims,
            run_time=4,
            rate_func=rush_from
        )
        self.remove(fader)

    def get_timeline_full_label(self, timeline, name):
        brace = Brace(Line().set_width(7), UP, buff=MED_SMALL_BUFF)
        brace.set_fill(border_width=5)
        brace.match_width(timeline)
        brace.next_to(timeline, UP, buff=MED_SMALL_BUFF)
        label = Text(name, font_size=72)
        label.next_to(brace, UP, MED_SMALL_BUFF)

        label.next_to(timeline, DOWN)
        return label

        return VGroup(brace, label)

    def get_computations(self, box, n_lines=10, n_iterations=3, n_digits=4, cycle_time=0.5):
        # Try adding lines
        lines = VGroup()
        for iteration in range(n_iterations):
            cluster = VGroup()
            for n in range(n_lines):
                x = random.uniform(0, 10**(n_digits))
                y = random.uniform(0, 10**(n_digits))
                if random.choice([True, False]):
                    comb = x * y
                    sym = Tex(R"\times")
                else:
                    comb = x + y
                    sym = Tex(R"+")
                line = VGroup(
                    DecimalNumber(x, num_decimal_places=3), sym,
                    DecimalNumber(y, num_decimal_places=3), Tex("="),
                    DecimalNumber(comb, num_decimal_places=3)
                )
                line.arrange(RIGHT, buff=SMALL_BUFF)
                lines.add(line)
                cluster.add(line)
            cluster.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
            cluster.set_max_height(0.9 * box.get_height())
            cluster.set_max_width(0.9 * box.get_width())
            cluster.move_to(box)

        # Add updater
        def update_lines(lines):
            sigma = 0.12
            alpha = (self.time / (cycle_time * n_iterations)) % 1
            step = 1.0 / len(lines)
            for n, line in enumerate(lines):
                x = min((
                    abs(a - n * step)
                    for a in (alpha - 1, alpha, alpha + 1)
                ))
                y = np.exp(-x**2 / sigma**2)
                line.set_fill(opacity=y)

            lines.set_height(0.9 * box.get_height())
            lines.move_to(box)

        lines.clear_updaters()
        lines.add_updater(update_lines)

        return lines

    def old(self):
        # Repeatedly scale down
        to_fade = VGroup(sec_brace, sec_label, box, comps, new_boxes, computations)
        scale_factors = [60, 24, 365, 1000]
        for new_int, prev_int, scale_factor in zip(number_lines[1:], number_lines[0:], scale_factors):
            self.fade_in_bigger_interval(
                new_int, prev_int, to_fade, scale_factor,
                added_anims=[label.animate.set_opacity(0)],
            )
            self.wait(2)
            to_fade = prev_int

        # Multiply last line by 100
        self.fade_in_bigger_interval(
            y1M_line, millenium_line, year_line, 1000,
            added_anims=[self.frame.animate.reorient(0, 0, 0, (-3.51, -5.18, 0.0), 12.93)],
        )

        lines = Line(LEFT, RIGHT).replicate(100)
        lines.match_width(y1M_line)
        lines.arrange_to_fit_height(10)
        lines.sort(lambda p: -p[1])
        lines.set_stroke(WHITE, 1)
        lines.move_to(y1M_line[0].get_center(), UP)

        side_brace, label100M = self.get_timeline_full_label(y1M_line, "100,000,000 Years")
        side_brace.rotate(PI / 2)
        side_brace.match_height(lines)
        side_brace.next_to(lines, LEFT)
        label100M.next_to(side_brace, LEFT)

        self.play(
            LaggedStart(
                (TransformFromCopy(lines[0].copy().set_opacity(0), line)
                for line in lines),
                lag_ratio=0.03,
                run_time=2
            ),
            FadeIn(side_brace, scale=10, shift=2 * DOWN, time_span=(1, 2)),
            FadeIn(label100M, time_span=(1, 2)),
        )
        self.wait()


class VectorLabel(InteractiveScene):
    def construct(self):
        # Test
        brace = Brace(Line(4 * UP, ORIGIN), LEFT)
        brace.center()
        brace.set_stroke(WHITE, 3)
        text = Text("Vector", font_size=90)
        text.next_to(brace, LEFT, MED_SMALL_BUFF)
        text.shift(SMALL_BUFF * UP)

        self.play(
            GrowFromCenter(brace),
            Write(text)
        )
        self.play(
            FlashUnder(text, color=YELLOW)
        )
        self.wait()


class ParameterToVectorAnnotation(InteractiveScene):
    def construct(self):
        # Test
        dials = VGroup(Dial(value_range=(-10, 10, 1)) for _ in range(10))
        dials.arrange(DOWN)
        dials.set_height(5)

        values = [1, 4.3, 2, 0.9, -1.5, 2.9, -1.2, 7.8, 0, -2.3]
        arrows = VGroup(
            Vector(0.5 * RIGHT, thickness=2).next_to(dial, RIGHT, buff=SMALL_BUFF)
            for dial in dials
        )

        self.play(
            Write(dials, lag_ratio=0.01),
            LaggedStartMap(GrowArrow, arrows),
        )
        self.play(LaggedStart(
            (dial.animate_set_value(value)
            for dial, value in zip(dials, values)),
            lag_ratio=0.05,
        ))
        self.wait()


class ThreeWordsToOne(InteractiveScene):
    def construct(self):
        # Test
        image = ImageMobject("CHMTopText")
        image.set_height(FRAME_HEIGHT)
        # self.add(image)

        phrase = Text("Computer History Museum", font_size=61)
        words = VGroup(phrase[word][0] for word in phrase.get_text().split(" "))
        words.move_to([0, 2.627, 0])
        og_words = words.copy()
        og_words.shift(DOWN)
        words[0].shift(0.13 * LEFT)
        words[2].shift(0.4 * RIGHT)
        colors = ["#63DCF7", "#90C9FA", "#85D4FE"]
        for word, color in zip(words, colors):
            word.set_color(color)

        words.save_state()

        self.add(words)
        self.wait()

        # Back to unity
        rect = SurroundingRectangle(og_words)
        rect.set_color(RED)
        chm_image = ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2024/transformers/chm/images/CHM_Exterior.jpeg")
        chm_image.match_width(rect)
        chm_image.next_to(rect, DOWN)

        self.play(Transform(words, og_words))
        self.play(
            ShowCreation(rect),
            FadeIn(chm_image, DOWN)
        )
        self.wait()

        # Three pieces
        rects = VGroup(
            SurroundingRectangle(word).set_fill(color, 0.2).set_stroke(color, 2)
            for word, color in zip(words.saved_state, colors)
        )
        words.set_z_index(1)

        icons = VGroup(
            SVGMobject("GenericComputer.svg"),
            SVGMobject("History.svg"),
            SVGMobject("Museum.svg"),
        )
        for word, icon in zip(words.saved_state, icons):
            icon.set_fill(word.get_color(), 1, border_width=1)
            icon.set_height(1)
            icon.next_to(word, DOWN)

        self.remove(chm_image)
        self.play(
            ReplacementTransform(VGroup(rect), rects),
            Restore(words),
            *(
                FadeTransform(chm_image.copy(), icon)
                for icon in icons
            )
        )
        self.wait()


class ExamplePhraseHeader(InteractiveScene):
    def construct(self):
        # Test
        phrase = Text("The Computer History Museum\nis located in ?????")
        phrase.to_edge(UP)
        rect = SurroundingRectangle(phrase).set_stroke(WHITE, 2)

        q_marks = phrase["?????"][0]
        q_marks[::4].set_fill(opacity=0)
        q_rect = SurroundingRectangle(q_marks)
        q_rect.set_fill(YELLOW, 0.25)
        q_rect.set_stroke(YELLOW, 2)

        self.add(q_rect)
        self.add(phrase)


class TrainingDataCHM(InteractiveScene):
    def construct(self):
        # Test
        passages = [
            "The Computer History Museum (CHM) is a museum ... located in Mountain View...",
            "Computer History Museum ... 1401 N. Shoreline Blvd. Mountain View...",
            "Things to do in Mountain View ... the Computer History Museum ...",
            "While I was in Mountain View ... stopped by the Computer History Museum ...",
        ]
        items = VGroup(
            get_paragraph(passage.split(" "), line_len=35, font_size=30)
            for passage in passages
        )

        items.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        items.to_corner(DL)
        items.shift(0.5 * UP)
        dots = Tex(R"\vdots")
        dots.next_to(items, DOWN, MED_LARGE_BUFF)
        dots.shift_onto_screen(buff=MED_SMALL_BUFF)
        items.add(dots)

        title = Text("Training Data")
        title.next_to(items, UP, buff=LARGE_BUFF)
        title.shift_onto_screen(buff=MED_SMALL_BUFF)
        underline = Underline(title)

        chm_phrases = VGroup(item["Computer History Museum"] for item in items)
        mv_phrases = VGroup(item["Mountain View"] for item in items)

        self.play(
            FadeIn(title),
            ShowCreation(underline),
            LaggedStartMap(FadeIn, items, shift=DOWN, lag_ratio=0.15)
        )
        self.wait()
        self.play(chm_phrases.animate.set_color(RED).set_anim_args(lag_ratio=0.1))
        self.wait()
        self.play(mv_phrases.animate.set_color(PINK).set_anim_args(lag_ratio=0.1))
        self.wait()

        # Arrows to ffn
        ffn_point = 3 * RIGHT + DOWN
        arrows = VGroup(
            Arrow(
                item.get_right(),
                interpolate(item.get_right(), ffn_point, 0.6),
                path_arc=arc * DEGREES,
            )
            for item, arc in zip(items[:-1], range(-40, 40, 20))
        )
        arrows.set_fill(border_width=1)
        self.play(Write(arrows, lag_ratio=0.1), run_time=3)
        self.play(
            LaggedStart(
                *(
                    FadeOutToPoint(letter.copy(), ffn_point)
                    for letter in VGroup(chm_phrases, mv_phrases).family_members_with_points()
                ),
                lag_ratio=1e-2,
                run_time=3
            )
        )
        self.wait()


class DivyUpParameters(ShowMachineWithDials):
    def construct(self):
        # Show machine
        frame = self.frame
        self.set_floor_plane("xz")

        machine = VGroup(*self.get_blocks_and_dials())
        blocks, llm_text, flat_dials, last_dials = machine
        machine.set_height(3.0)
        machine.to_edge(DOWN, buff=LARGE_BUFF)

        block_outlines = blocks.copy()
        block_outlines.set_fill(opacity=0)
        block_outlines.set_stroke(WHITE, 2)
        block_outlines.insert_n_curves(20)

        # last_dials.set_submobjects(last_dials[:3])  # Remove
        last_dials.set_stroke(opacity=1)
        for dial in last_dials:
            dial[0].set_stroke(width=1)
            dial[1].set_stroke(width=1)
            dial[3].set_stroke(width=(3, 0))

        frame.reorient(-23, -13, 0, (-0.41, -1.71, -0.06), 4.95)
        self.play(
            FadeIn(blocks, shift=0.0, lag_ratio=0.01),
            LaggedStartMap(VShowPassingFlash, block_outlines.family_members_with_points(), time_width=2.0, lag_ratio=0.01, remover=True),
            LaggedStartMap(VFadeInThenOut, flat_dials, lag_ratio=0.001, remover=True),
            FadeIn(last_dials, time_span=(2, 3)),
            self.frame.animate.reorient(10, -2, 0, (-0.25, -1.58, -0.02), 4.61),
            run_time=3,
        )
        self.remove(flat_dials)

        # Show individual blocks
        top_blocks = blocks[:3].copy()
        all_dials = VGroup(*last_dials)
        for block in top_blocks:
            dials = last_dials.copy()
            dials.rotate(self.machine_phi, RIGHT)
            dials.rotate(self.machine_theta, UP)
            dials.move_to(block)
            dials.set_stroke(opacity=1)
            block.add(dials)
            block.target = block.generate_target()
            dials.set_opacity(0)
            all_dials.add(*dials)

        block_targets = Group(block.target for block in top_blocks)
        block_targets.rotate(-self.machine_theta, UP)
        block_targets.rotate(-self.machine_phi, RIGHT)
        block_targets.set_height(2)
        block_targets.arrange(RIGHT, buff=1.5)
        block_targets.to_edge(UP)
        block_targets.set_shading(0.1, 0.1, 0.1)

        labels = VGroup(
            TexText(R"Word $\to$ Vector"),
            Text("Attention"),
            Text("Feedforward"),
        )
        for label, block in zip(labels, block_targets):
            label.next_to(block, DOWN)

        self.add(
            blocks[0], top_blocks[0],
            blocks[1], top_blocks[1],
            blocks[2], top_blocks[2],
            blocks[3:], last_dials
        )
        self.play(
            MoveToTarget(top_blocks[1], time_span=(0, 2)),
            MoveToTarget(top_blocks[2], time_span=(1, 3)),
            MoveToTarget(top_blocks[0], time_span=(2, 4)),
            Write(labels[1], time_span=(1.5, 2)),
            Write(labels[2], time_span=(2.5, 3)),
            Write(labels[0], time_span=(3.5, 4)),
            frame.animate.to_default_state(),
            run_time=4
        )
        self.wait()

        # Change all the parameters
        self.play(
            LaggedStart(
                (dial.animate_set_value(dial.get_random_value())
                for dial in all_dials),
                lag_ratio=1 / len(all_dials),
                run_time=6
            ),
            LaggedStart(
                (FlashAround(dial, buff=0, color=YELLOW)
                for dial in all_dials),
                lag_ratio=1 / len(all_dials),
                run_time=6
            ),
        )
        self.wait()


# End clips


class ShowPreviousVideos(InteractiveScene):
    def construct(self):
        # Backdrop
        background = FullScreenRectangle()
        self.add(background)

        line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        line.set_stroke(WHITE, 2)

        series_name = Text("Deep Learning Series", font_size=68)
        series_name.to_edge(UP, buff=0.35)
        self.add(series_name)

        # Show thumbnails
        thumbnails = Group(
            Group(
                Rectangle(16, 9).set_height(1).set_stroke(WHITE, 2),
                ImageMobject(f"https://img.youtube.com/vi/{slug}/maxresdefault.jpg", height=1)
            )
            for slug in [
                "aircAruvnKk",
                "IHZwWFHWa-w",
                "Ilg3gGewQ5U",
                "tIeHLnjs5U8",
                "wjZofJX0v4M",
                "eMlx5fFNoYc",
                "9-Jl0dxWQs8",
            ]
        )

        thumbnails.arrange_in_grid(n_cols=4, buff=0.2)
        thumbnails.set_width(FRAME_WIDTH - 1)
        thumbnails.next_to(series_name, DOWN, buff=1.0)
        thumbnails[-3:].set_x(0)

        self.play(LaggedStartMap(FadeIn, thumbnails, shift=0.3 * UP, lag_ratio=0.35, run_time=4))
        self.wait()

        # Rearrange
        left_x = -FRAME_WIDTH / 4
        self.play(
            series_name.animate.set_x(left_x),
            thumbnails.animate.arrange_in_grid(n_cols=2, buff=0.25).set_height(6).set_x(left_x).to_edge(DOWN),
            ShowCreation(line, time_span=(1, 2)),
            run_time=2,
        )
        self.wait()


class EndScreen(PatreonEndScreen):
    title_text = "Where to dig deeper"
    thanks_words = """
        Special thanks to these Patreon supporters
    """
