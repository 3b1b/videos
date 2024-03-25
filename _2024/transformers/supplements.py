from manim_imports_ext import *
from _2024.transformers.helpers import *


class GPTInitials(InteractiveScene):
    def construct(self):
        # Write name
        frame = self.frame
        name_str = "Generative Pre-trained Transformer"
        name = Text(name_str, font_size=72)
        name.to_edge(UP)
        name.save_state()
        frame.move_to(name).shift(DOWN)
        words = VGroup(name[word][0] for word in name_str.split(" "))
        initials = Text("GPT")
        initials.replace(name, dim_to_match=1)
        t_target = initials["T"][0].generate_target()
        t_target.shift(3 * RIGHT)

        words[0].next_to(initials["P"], LEFT, aligned_edge=DOWN)
        words[1].next_to(t_target, LEFT, aligned_edge=DOWN)

        morty = Mortimer(mode='plain').flip()
        morty.next_to(initials, DL).shift(0.5 * DOWN)
        morty.body.insert_n_curves(100)
        self.add(morty)

        def letter_anim(letters, point):
            for letter in letters:
                letter.save_state()
                letter.set_opacity(0)
                letter.move_to(point)
            return LaggedStart(
                (Restore(letter) for letter in letters),
                lag_ratio=0.05,
                time_span=(0.25, 0.75)
            )

        self.play(
            LaggedStartMap(FadeIn, initials, scale=2, lag_ratio=0.25, run_time=1),
            morty.change("raise_right_hand", initials),
        )
        self.play(Blink(morty))
        self.play(
            ReplacementTransform(initials[0], words[0][0]),
            letter_anim(words[0][1:], initials[0].get_center()),
            morty.animate.look_at(words[0]),
            run_time=1
        )
        self.wait(0.5)
        self.play(
            words[0].animate.next_to(words[1], LEFT, aligned_edge=DOWN),
            Transform(initials[2], t_target),
            ReplacementTransform(initials[1], words[1][0]),
            letter_anim(words[1][1:], initials[1].get_center()),
            morty.change("well", words[1]),
            run_time=1
        )
        self.remove(initials)
        self.wait(0.5)
        self.play(
            Transform(name[:-len(words[2])], name.saved_state[:-len(words[2])]),
            ReplacementTransform(initials[2], words[2][0]),
            letter_anim(words[2][1:], initials[2].get_center()),
            morty.animate.look_at(words[2])
        )
        self.add(name)
        self.play(Blink(morty))
        self.wait()

        # Set up T structure
        h_line = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        h_line.next_to(words, DOWN).set_x(0)
        v_lines = Line(UP, DOWN).set_height(FRAME_HEIGHT).replicate(2)
        v_lines.arrange(RIGHT, buff=FRAME_WIDTH / 3)
        v_lines.next_to(h_line, DOWN, buff=0)
        t_lines = VGroup(h_line, *v_lines)
        t_lines.set_stroke(GREY_B, 1)

        # Go through each word
        words.target = words.generate_target()
        words.target[0].set_fill(YELLOW)
        words.target[1:].set_fill(WHITE, 0.5, border_width=0)
        offset = FRAME_WIDTH / 3
        words.target[0].set_x(-offset)
        words.target[1].set_x(0)
        words.target[2].set_x(offset)
        line = Underline(words.target[0])
        line.set_stroke(YELLOW)
        self.play(LaggedStart(
            MoveToTarget(words),
            ShowCreation(line),
            frame.animate.center(),
            morty.change("thinking", words.target[0]).set_opacity(0),
            Write(t_lines, stroke_width=2),
            FlashAround(words.target[0].copy())
        ))
        self.remove(morty)
        self.wait()
        for i in [1, 2]:
            words.target = words.generate_target()
            words.target.set_fill(WHITE, 0.5, border_width=0)
            words.target[i].set_fill(YELLOW, 1, border_width=0.5)
            self.play(
                line.animate.become(Underline(words[i])).set_stroke(YELLOW).set_anim_args(run_time=0.75),
                FlashAround(words[i]),
                MoveToTarget(words),
            )
            self.wait()

        # Isolate just Transformer
        self.play(
            words[2].animate.set_x(0).set_color(WHITE).shift(0.25 * UP),
            line.animate.set_x(0).set_color(WHITE).set_width(6).shift(0.25 * UP),
            FadeOut(words[0], LEFT),
            FadeOut(words[1], 3 * LEFT),
            Uncreate(t_lines, lag_ratio=0),
        )
        self.wait()


class DifferentUsesOfModel(InteractiveScene):
    def construct(self):
        # Set up sentences
        sentences = VGroup(
            Text("A machine learning model ..."),
            Text("A fashion model ..."),
        )
        images = Group(
            NeuralNetwork([8, 6, 6, 8]),
            ImageMobject("Zoolander"),
        )
        for sent, image, sign in zip(sentences, images, [-1, 1]):
            sent.set_y(-2)
            sent.set_x(sign * FRAME_WIDTH / 4)
            image.set_width(4)
            image.next_to(sent, UP, buff=0.5)
        images[0].match_y(images[1])
        sentences[0]["model"].set_color(BLUE)
        sentences[1]["model"].set_color(YELLOW)

        # Put word in context
        word = Text("model", font_size=72)
        word.to_edge(UP, buff=0.25)

        self.play(FadeIn(word, UP))
        self.wait()
        self.play(
            FadeTransform(word.copy(), sentences[0]["model"]),
            LaggedStart(
                Write(sentences[0]),
                Write(images[0], lag_ratio=0.01, stroke_width=0.5),
                lag_ratio=0.5,
                run_time=2
            )
        )
        self.wait()
        self.play(
            FadeTransform(word.copy(), sentences[1]["model"]),
            LaggedStart(
                Write(sentences[1]),
                FadeIn(images[1], shift=0.5 * UP, scale=1.25),
                lag_ratio=0.2,
                run_time=2
            )
        )
        self.wait()

        # Show relevance
        s0, s1 = sentences
        path_arc = -0.65 * PI
        left_arrows = VGroup(
            Arrow(
                s0[word].get_top(),
                s0["model"].get_top(),
                path_arc=path_arc
            )
            for word in ["machine", "learning"]
        )
        right_arrow = Arrow(
            s1["fashion"].get_top(),
            s1["model"].get_top(),
            path_arc=path_arc
        )
        left_arrows[0].set_stroke(TEAL, opacity=0.9)
        left_arrows[1].set_stroke(TEAL_D, opacity=0.75)
        right_arrow.set_stroke(TEAL_C, opacity=0.8)

        self.play(
            LaggedStartMap(ShowCreation, left_arrows, lag_ratio=0.5),
            self.frame.animate.move_to(DOWN),
            images.animate.shift(UP),
        )
        self.play(ShowCreation(right_arrow))
        self.wait()

        # Show word vectors
        words = VGroup(
            *(s0[word] for word in s0.get_text().split(" ")[:-1]),
            *(s1[word] for word in s1.get_text().split(" ")[:-1]),
        )
        vectors = VGroup(
            NumericEmbedding().set_height(2).next_to(word, DOWN, buff=0.2)
            for word in words
        )

        self.play(
            LaggedStartMap(FadeIn, vectors, shift=0.25 * DOWN, lag_ratio=0.25, run_time=3)
        )
        self.play(
            LaggedStartMap(RandomizeMatrixEntries, vectors)
        )
        self.wait()


class BigMatrixMultiplication(InteractiveScene):
    mat_dims = (12, 12)

    def construct(self):
        from datetime import datetime
        np.random.seed(datetime.now().toordinal())
        random.seed(datetime.now().toordinal())
        # Test
        matrix = WeightMatrix(shape=self.mat_dims)
        matrix.set_width(FRAME_WIDTH - 4)
        matrix.to_edge(LEFT, buff=0.5)
        vector = NumericEmbedding(length=self.mat_dims[0])
        vector.match_height(matrix)
        vector.next_to(matrix, RIGHT)

        self.add(matrix)
        self.add(vector)
        show_matrix_vector_product(self, matrix, vector)
        self.wait()


class LongListOFQuestions(InteractiveScene):
    def construct(self):
        # Add word and vector
        word = Text("Queen")
        arrow = Vector(0.75 * RIGHT)
        vector = NumericEmbedding(length=12)
        vector.set_height(5)
        word_group = VGroup(word, arrow, vector)
        word_group.arrange(RIGHT, buff=0.15)
        word_group.to_edge(LEFT, buff=2.0)

        self.add(word_group)

        # Add neurons and questions
        questions = VGroup(map(Text, [
            "Is it English?",
            "Is it a noun?",
            "Does it refer to a person?",
            "Is it an amount?",
            "Is the tone assertive",
            "Is it a piece of a bigger word?",
            "Is it part of a quote?",
            "Is it part of a lie?",
        ]))
        questions.scale(0.75)
        n_questions = len(questions)
        neurons = VGroup(Circle(radius=0.2) for n in range(n_questions))
        neurons.add(Tex(R"\vdots", font_size=72))
        neurons.arrange_in_grid(n_questions, 1, buff_ratio=0.5)
        neurons.set_height(6)
        neurons.set_stroke(WHITE, 1)
        neurons.next_to(vector, RIGHT, buff=3.0)
        values = [0.9, 0.8, 0.85, 0.1, 0.5, 0.05, 0.2, 0.02]
        for neuron, question, value in zip(neurons, questions, values):
            neuron.set_fill(WHITE, value)
            question.next_to(neuron, RIGHT)

        # Add connections
        connections = VGroup(
            VGroup(
                Line(
                    elem.get_right(), neuron.get_center(),
                    buff=neuron.get_width() / 2
                ).set_stroke(
                    color=value_to_color(random.uniform(-10, 10)),
                    width=2 * random.random()
                )
                for elem in vector.get_entries()
            )
            for neuron in neurons[:-1]
        )

        # Animate
        lag_ratio = 0.3
        self.play(
            LaggedStart(
                (ShowCreation(line_group, lag_ratio=0)
                for line_group in connections),
                lag_ratio=lag_ratio,
            ),
            LaggedStartMap(FadeIn, neurons, lag_ratio=lag_ratio),
            LaggedStartMap(FadeIn, questions, lag_ratio=lag_ratio),
            run_time=4
        )
        self.wait()


class ChatBotIcon(InteractiveScene):
    def construct(self):
        # Add bot
        bot = SVGMobject("ChatBot")
        bot.set_fill(GREY_B)
        bot[0].set_stroke(WHITE, 3)
        bot.set_height(3)
        bot.to_edge(RIGHT)

        arrow = Vector(
            1.5 * RIGHT,
            max_tip_length_to_length_ratio=0.4,
            max_width_to_length_ratio=9.0,
        )
        arrow.set_stroke(width=20)
        arrow.next_to(bot, LEFT).match_y(bot[0])

        self.play(
            ShowCreation(arrow),
            Write(bot),
        )
        self.wait()


class GamePlan(InteractiveScene):
    screen_opacity = 0.0

    def construct(self):
        # Setup up icons
        self.add(FullScreenRectangle())
        videos = VideoIcon().get_grid(7, 1, buff_ratio=0.3)
        videos.set_fill(BLUE_B)
        videos.set_height(6.5)
        videos.to_corner(UL)
        column_x = videos.get_x()

        nn_vids = videos[:4]
        tr_vids = videos[4:]
        tr_vids.save_state()
        tr_vids.scale(1.25)
        tr_vids.space_out_submobjects(1.25)
        tr_vids.set_y(0).to_edge(LEFT)

        def highlight_video(video, group=videos):
            for vid in group:
                vid.target = vid.generate_target()
                if vid is video:
                    vid.target.set_x(column_x + 0.5)
                    vid.target.set_opacity(1)
                else:
                    vid.target.set_x(column_x)
                    vid.target.set_opacity(0.5)
            return LaggedStartMap(MoveToTarget, group, lag_ratio=0.01, run_time=1)

        self.add(tr_vids)

        # Here now
        here_arrow = Vector(0.75 * LEFT, stroke_width=10)
        here_arrow.set_color(RED).next_to(tr_vids[0], RIGHT)
        here_words = Text("You are\nhere")
        here_words.next_to(here_arrow, RIGHT)
        here_words.set_color(RED)
        here_group = VGroup(here_arrow, here_words)

        self.play(
            highlight_video(tr_vids[0], tr_vids),
            MaintainPositionRelativeTo(here_group, tr_vids[0]),
            VFadeIn(here_group),
        )
        self.wait()

        # First chapter
        curly = self.get_curly_brace(tr_vids[0])

        topics = VGroup(
            Text("Beginning"),
            Text("Ending"),
            Text("Background material"),
            Text("Premise of deep learning"),
            Text("Word embeddings"),
            Text("Dot products"),
            Text("Softmax"),
        )
        topics.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        topics.next_to(curly.get_corner(UR), DR, buff=0.25)
        for topic in topics[-4:]:
            topic.scale(0.8, about_edge=LEFT)
            topic.shift(0.5 * RIGHT)
            dot = Dot(color=WHITE)
            dot.next_to(topic, LEFT)
            topic.add(dot)

        screen = ScreenRectangle()
        screen.set_fill(BLACK, 1)
        screen.set_stroke(WHITE, 2)
        screen.set_opacity(self.screen_opacity)
        screen.set_height(5)
        screen.next_to(topics[0], DOWN, aligned_edge=LEFT)

        self.play(
            FadeOut(here_group),
            ShowCreation(curly),
            FadeIn(screen, RIGHT),
            FadeInFromPoint(topics[0], here_group.get_center()),
        )
        self.wait()
        self.play(
            topics[0].animate.set_opacity(0.5),
            FadeIn(topics[1]),
            screen.animate.next_to(topics[1], DOWN, aligned_edge=LEFT)
        )
        self.wait()
        self.play(
            screen.animate.scale(0.5, about_edge=DR).to_edge(RIGHT),
            topics[1].animate.set_opacity(0.5),
            LaggedStartMap(FadeIn, topics[2:], shift=0.1 * DOWN, lag_ratio=0.5)
        )
        self.wait()

        # Second chapter
        new_curly = self.get_curly_brace(tr_vids[1].copy().shift(0.5 * RIGHT))
        screen.target = screen.generate_target()
        screen.target.set_height(5)
        screen.target.next_to(curly, RIGHT)
        att_title = Text("Attention")
        att_title.next_to(screen.target, UP, aligned_edge=LEFT)
        self.play(
            highlight_video(tr_vids[1], tr_vids),
            curly.animate.become(new_curly),
            FadeOut(topics),
            MoveToTarget(screen),
            FadeInFromPoint(att_title, tr_vids[1].get_center()),
        )
        self.wait()

        # Third chapter
        new_curly = self.get_curly_brace(tr_vids[2].copy().shift(0.5 * RIGHT))
        chapter3_topics = Text(
            "MLPs, Training, Positional encodings, ..."
        )
        chapter3_topics.next_to(screen, UP, aligned_edge=LEFT)

        self.play(
            highlight_video(tr_vids[2], tr_vids),
            curly.animate.become(new_curly),
            FadeOut(att_title),
            FadeIn(chapter3_topics, lag_ratio=0.1, time_span=(1, 3)),
        )
        self.wait()

        # Show earlier chapters
        prev_thumbnails = Group(
            ImageMobject(f"nn{k}_thumbnail.png")
            for k in range(1, 5)
        )
        prev_thumbnails.arrange(RIGHT, buff=1.0)
        prev_thumbnails.set_width(FRAME_WIDTH - 2)
        prev_thumbnails.move_to(2 * UP)

        tn_dir = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2024/transformers/Thumbnails/"
        new_thumbnails = Group(
            ImageMobject(os.path.join(tn_dir, f"Chapter{n}"))
            for n in range(5, 8)
        )
        for tn1, tn2 in zip(prev_thumbnails, new_thumbnails):
            tn2.replace(tn1, stretch=True)
            tn2.next_to(tn1, DOWN, buff=1.0)
        chapter_titles = VGroup(
            Text(f"Chapter {k}", font_size=30)
            for k in range(1, 8)
        )
        for title, rect in zip(chapter_titles, (*prev_thumbnails, *new_thumbnails)):
            title.next_to(rect, UP, buff=0.2, aligned_edge=LEFT)

        tr_rect = SurroundingRectangle(
            Group(new_thumbnails, chapter_titles[4:]),
            buff=0.25
        )
        tr_rect.set_stroke(BLUE, 2)
        tr_label = Text("Transformers")
        tr_label.next_to(tr_rect, DOWN)

        self.play(
            FadeOut(curly),
            FadeOut(screen),
            FadeOut(chapter3_topics),
            LaggedStartMap(FadeIn, chapter_titles,),
            FadeIn(prev_thumbnails, shift=0.5 * UP, lag_ratio=0.25),
            *(
                FadeTransform(vid, tn)
                for vid, tn in zip(tr_vids, new_thumbnails)
            ),
        )
        self.play(
            ShowCreation(tr_rect),
            FadeIn(tr_label),
        )
        self.wait()

    def get_curly_brace(self, video, width=2.0, height=6.5, buff=0.1):
        start = video.get_right() + buff * RIGHT
        top_point = np.array([start[0] + width, 0.5 * height, 0])
        low_point = np.array([start[0] + width, -0.5 * height, 0])
        result = VGroup(
            CubicBezier(
                start,
                start + width * RIGHT,
                point + width * LEFT,
                point,
            )
            for point in [top_point, low_point]
        )
        result.set_stroke(GREY_A, 2)
        return result


class SkipAhead(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        morty = self.teacher
        self.play(
            morty.change("hesitant", self.students),
            self.change_students("confused", "pondering", "pondering", look_at=self.screen),
        )
        self.wait(2)
        self.play(self.change_students("confused", "tease", "well", look_at=morty.eyes))
        self.wait(5)


class SeaOfNumbersUnderlay(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)
        self.play(
            morty.change("pleading"),
            self.change_students("surprised", "horrified", "droopy")
        )
        self.look_at(3 * LEFT + 2 *  UP)
        self.look_at(3 * RIGHT + 2 * UP)
        self.look_at(3 * LEFT + 2 * UP)

        self.play(
            morty.change("raise_right_hand", self.screen),
            self.change_students("hesitant", "pondering", "maybe", look_at=self.screen)
        )
        self.wait(2)
        self.play(self.change_students("erm", "pondering", "confused", look_at=self.screen))
        self.wait(2)

        self.look_at(5 * RIGHT + 2 * UP)
        self.play(self.change_students("hesitant", "pondering", "hesitant", look_at=5 * RIGHT + 2 * UP))
        self.wait(3)
        self.play(
            morty.change("well"),
            self.change_students("pondering", "pondering", "erm", look_at=self.screen)
        )
        self.wait(3)
        self.play(
            morty.change("raise_left_hand", look_at=5 * RIGHT + 3 * UP),
            self.change_students("tease", "thinking", "pondering", look_at=5 * RIGHT + 3 * UP)
        )
        self.wait(8)


class Outdated(TeacherStudentsScene):
    def construct(self):
        # Add label
        text = Text("GPT-3", font="Consolas", font_size=72)
        openai_logo = SVGMobject("OpenAI.svg")
        openai_logo.set_fill(WHITE)
        openai_logo.set_height(2.0 * text.get_height())
        gpt3_label = VGroup(openai_logo, text)
        gpt3_label.arrange(RIGHT)
        gpt3_label.scale(0.75)
        param_count = Text("175B Parameters")
        param_count.set_color(BLUE)
        param_count.next_to(gpt3_label, DOWN, aligned_edge=LEFT)
        gpt3_label.add(param_count)

        gpt3_label.move_to(self.hold_up_spot, DOWN)

        morty = self.teacher
        morty.body.insert_n_curves(100)

        self.play(
            morty.change("raise_right_hand"),
            FadeIn(gpt3_label, UP),
        )
        self.play(self.change_students("raise_left_hand", "hesitant", "sassy"))
        self.play(
            self.students[0].says(TexText("Isn't that outdated?"))
        )
        self.wait(3)


class ConfusionAtScreen(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.change("well"),
            self.change_students("maybe", "confused", "concentrating", look_at=self.screen)
        )
        self.wait(2)
        self.play(
            self.teacher.change("tease"),
            self.change_students("hesitant", "plain", "erm", look_at=self.teacher.eyes)
        )
        self.wait(3)


class HoldUpExample(TeacherStudentsScene):
    def construct(self):
        self.background.set_fill(opacity=0.0)
        self.teacher.body.insert_n_curves(100)
        self.play(
            self.teacher.change("raise_right_hand"),
            self.change_students("happy", "hooray", "well", look_at=4 * UR)
        )
        self.wait(5)


class ReactToWordVectors(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer().flip()
        randy = Randolph().flip()
        morty, randy = pis = VGroup(morty, randy)
        pis.arrange(RIGHT, buff=2.0)
        pis.to_edge(DOWN)
        randy.make_eye_contact(morty)

        self.add(pis)
        self.play(
            PiCreatureSays(
                morty, "This is how search\nworks you know!",
                target_mode="hooray",
                content_introduction_class=FadeIn,
                content_introduction_kwargs=dict(lag_ratio=0.1),
            ),
            randy.change("guilty"),
        )
        self.play(Blink(randy))
        self.wait()
        dots = Text(".....", font_size=120)
        dots[:1].set_opacity(0)
        dots[-1:].set_opacity(0)
        self.play(
            morty.debubble(),
            PiCreatureBubbleIntroduction(
                randy, dots, target_mode="confused",
                bubble_type=ThoughtBubble,
            ),
            morty.change("tease", look_at=6 * LEFT),
        )
        self.play(Blink(morty))
        self.wait()


class DimensionComparrison(InteractiveScene):
    def construct(self):
        titles = VGroup(
            Text("3d vectors"),
            Text("Word vectors"),
        )
        titles.scale(1.5)
        for title, vect in zip(titles, [LEFT, RIGHT]):
            title.move_to(vect * FRAME_WIDTH / 4)
            title.to_edge(UP, buff=MED_SMALL_BUFF)
        h_line = Line(LEFT, RIGHT)
        h_line.set_width(FRAME_WIDTH)
        h_line.next_to(titles, DOWN)
        h_line.set_x(0)
        v_line = Line(UP, DOWN)
        v_line.set_height(FRAME_HEIGHT)
        lines = VGroup(h_line, v_line)
        lines.set_stroke(GREY_B, 2)

        self.play(
            ShowCreation(lines, lag_ratio=0.5),
            LaggedStartMap(Write, titles, lag_ratio=0.5)
        )
        self.wait()


class AtLeastKindOf(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            morty.says("...kind of", mode="hesitant"),
            self.change_students("hesitant", "sassy", "erm", look_at=self.screen)
        )
        self.wait(3)
        self.play(
            self.change_students("sassy", "hesitant", "hesitant", look_at=morty.eyes),
            morty.change("guilty"),
        )
        self.wait(4)


class NetworkEndAnnotation(InteractiveScene):
    opacity = 0.5

    def construct(self):
        im = ImageMobject("NetworkEnd")
        im.set_height(FRAME_HEIGHT)
        self.add(im)

        # word by word
        prof = Text("Professor").set_height(0.25).move_to(np.array([4.77, 3.36, 0.]))
        hp = Text("Harry Potter").set_height(0.33).move_to(np.array([-5.58, 3.33, 0.]))
        lf = Text("least favourite").set_height(0.26).move_to(np.array([1.39, 3.35, 0]))
        snape = Rectangle(3.5, 0.3).move_to(np.array([5.0, 1.11, 0]))

        def get_inverse_rect(mob):
            big_rect = FullScreenFadeRectangle()
            big_rect.scale(1.1)
            lil_rect = SurroundingRectangle(mob)
            big_rect.start_new_path(lil_rect.get_points()[-1])
            big_rect.append_points(lil_rect.get_points()[-2::-1])
            big_rect.set_stroke(WHITE, 1)
            big_rect.set_fill(BLACK, self.opacity)
            return big_rect

        rects = VGroup(map(get_inverse_rect, [prof, hp, lf, snape]))

        rect = rects[0].copy()
        self.play(FadeIn(rect))
        self.wait()
        for rect2 in rects[1:]:
            self.play(Transform(rect, rect2))
            self.wait()
        self.play(FadeOut(rect))


class LowTempHighTempContrast(InteractiveScene):
    def construct(self):
        # Test
        titles = VGroup(
            Text("Temp = 0", font_size=72).set_x(-FRAME_WIDTH / 4),
            Text("Temp = 5", font_size=72).set_x(FRAME_WIDTH / 4),
        )
        titles.to_edge(UP, buff=0.25)
        h_line = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        h_line.next_to(titles, DOWN, buff=0.1)
        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        lines = VGroup(h_line, v_line)
        lines.set_stroke(GREY_B, 2)

        self.play(
            LaggedStartMap(FadeIn, titles, shift=0.25 * UP, lag_ratio=0.25),
            LaggedStartMap(Write, lines, lag_ratio=0.5),
            run_time=1
        )
        self.wait()
