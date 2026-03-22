from manim_imports_ext import *
from _2024.transformers.helpers import *
from _2024.transformers.embedding import *


# Intro chapter


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
    random_seed = 9

    def construct(self):
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
            "Is A tone assertive",
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


class ConvolutionComment(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer()
        morty.to_corner(DR)
        bubble = morty.get_bubble(Text("""
            In other models, the weighted
            sums can be grouped differently,
            e.g. as convolutions, but for
            Transformers it's always
            matrix-vector multiplication.
        """, font_size=36, alignment="LEFT"), bubble_type=SpeechBubble)

        self.add(bubble)
        self.play(morty.change("speaking"))
        for x in range(2):
            self.play(Blink(morty))
            self.wait()


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


class Intuitions(TeacherStudentsScene):
    def construct(self):
        # Add words
        words = VGroup(
            Text("Structure of Deep Learning"),
            Text("Word embeddings"),
            Text("Dot products"),
            Text("Softmax"),
        )
        words.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        words.move_to(self.hold_up_spot, DOWN)
        checks = VGroup(
            Checkmark(font_size=72).next_to(word, LEFT)
            for word in words
        )
        checks.set_color(GREEN)

        morty = self.teacher
        self.play(
            LaggedStartMap(FadeIn, words, shift=UP, lag_ratio=0.1),
            morty.change("raise_right_hand"),
            self.change_students("thinking", "pondering", "well", look_at=words),
            run_time=1,
        )
        self.play(
            LaggedStartMap(Write, checks, lag_ratio=0.25, stroke_color=GREEN),
        )
        for pi in self.students:
            pi.body.insert_n_curves(100)
        self.play(
            self.change_students("tease", "thinking", "well")
        )
        self.wait(4)


class PiGesturingAtEarlyView(PiCreatureScene):
    def construct(self):
        morty = self.pi_creature.flip()
        morty.to_corner(DR)
        morty.shift(0.5 * LEFT)
        morty.set_color(GREY_BROWN)
        morty.body.insert_n_curves(100)
        for mode in ["raise_right_hand", "well", "gracious", "well", "tease"]:
            self.play(morty.change(mode, ORIGIN + 2 * random.random() * UP))
            self.wait(3)


class EndScreen(PatreonEndScreen):
    pass


# Attention chapter


class HighlightAttentionTitle(TeacherStudentsScene):
    def construct(self):
        # Add image
        im = ImageMobject("AttentionPaper")
        im.set_height(FRAME_HEIGHT)
        title = Text("Attention is All You Need")
        title.set_height(0.219)
        title.move_to(np.array([-0.037, 3.28, 0.0]))
        title.set_fill(BLACK, 1)
        self.clear()
        self.background.set_opacity(0)
        self.add(self.background, im)
        
        self.wait()
        morty = self.teacher
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)
        self.play(
            im.animate.set_opacity(0.1),
            title.animate.set_fill(WHITE).scale(2).next_to(morty, UP, MED_LARGE_BUFF).to_edge(RIGHT),
            LaggedStartMap(VFadeIn, self.pi_creatures),
            morty.change("raise_right_hand"),
            self.change_students("pondering", "well", "thinking", look_at=self.hold_up_spot)
        )
        self.wait()

        # # Small transition
        # alt_title = Text("Attention is all\nyou need")
        # alt_title.move_to(4.68 * LEFT)

        # self.play(
        #     # TransformMatchingStrings(title, alt_title, run_time=1),
        #     FadeTransformPieces(title, alt_title, run_time=1),
        #     FadeOut(im, scale=2),
        #     self.change_students("pondering", "pondering", "pondering", look_at=alt_title),
        # )
        # self.wait()

        # Highlight attention
        att = title["Attention"][0]
        rest = title["is All You Need"][0]
        self.play(
            FlashAround(att, run_time=2),
            att.animate.set_color(YELLOW),
        )
        self.wait(2)
        self.play(
            att.animate.center().to_edge(UP),
            FadeOut(rest, DR),
            FadeOut(im, scale=1.5),
            self.background.animate.set_opacity(0.75),
            morty.change("tease", 3 * UP),
            self.change_students(None, None, "pondering", look_at=3 * UP)
        )
        self.look_at(3 * UL)
        self.wait()
        self.look_at(3 * UR)
        self.wait(2)

        # Key property
        sentence = Text("What makes Attention powerful is that it's parallelizable")
        sentence.move_to(UP)
        sent_att = sentence["Attention"]
        sent_par = sentence["parallelizable"]
        sent_att.set_opacity(0)
        sent_par.set_opacity(0)
        par_box = SurroundingRectangle(sent_par, buff=0)
        par_box.stretch(1.2, 1, about_edge=DOWN)
        par_box.set_stroke(width=0)
        par_box.set_fill(RED, 0.2)
        par_line = Underline(sent_par, stretch_factor=1)
        par_line.set_stroke(RED, 2)

        self.play(
            att.animate.replace(sentence["Attention"]),
            FadeIn(sentence, lag_ratio=0.1),
            morty.change("raise_right_hand", sentence),
            self.change_students("sassy", "confused", "pondering", look_at=sentence)
        )
        self.play(
            ShowCreation(par_line),
            morty.animate.look_at(par_line),
            FadeIn(par_box),
            self.change_students("pondering", look_at=par_line),
        )
        self.wait(5)


class ThinkOfMoreExamples(TeacherStudentsScene):
    def construct(self):
        # Show general confusion
        morty = self.teacher
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)

        self.play(
            morty.change("raise_right_hand"),
            self.change_students("confused", "maybe", "confused", look_at=3 *UP, run_time=2, lag_ratio=0.25),
        )
        self.wait(2)
        self.play(morty.change("guilty"))
        self.play(
            
            self.change_students("confused", "pleading", "concentrating", look_at=3 * UP, run_time=2, lag_ratio=0.25)
        )
        self.wait(3)
        self.play(
            self.change_students("maybe", "confused", "dejected", look_at=morty.eyes, lag_ratio=0),
            morty.change("well")
        )
        self.wait(2)

        # Ask about the goal
        self.wait()
        self.play(LaggedStart(
            self.students[2].says("What is attention\nsupposed to do?"),
            self.students[0].change("maybe"),
            self.students[1].change("pondering"),
            morty.change("tease"),
            lag_ratio=0.1
        ))
        self.wait(5)


class SimplerExample(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.change_students("pondering", "thinking", "pondering", look_at=self.screen)
        )
        self.play(
            self.teacher.says("Take a simpler\nexample"),
            self.change_students("pondering", look_at=self.teacher.eyes)
        )
        self.play(self.change_students("thinking", "well", "tease"))
        self.wait(6)


class NotQuiteTrue(InteractiveScene):
    def construct(self):
        morty = Mortimer()
        morty.to_corner(DR)
        self.play(
            morty.says("Actually, that's not\nquite true!"),
            run_time=1
        )
        for x in range(2):
            self.play(Blink(morty))
            self.wait()


class ThisIsMadeUp(TeacherStudentsScene):
    def construct(self):
        for pi in self.students:
            pi.change_mode("pondering").look_at(self.screen)
        self.play(
            self.teacher.says("This is a made-up\nmotivating example"),
            self.change_students("pondering", look_at=self.teacher.eyes)
        )
        self.play(self.change_students("well", "sassy", "guilty", look_at=self.teacher.eyes))
        self.wait(4)


class AskAboutOtherEmbeddings(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.students[1].says(
                TexText(R"What does $W_Q$ do \\ to the non-nouns?"),
                mode="raise_left_hand"
            ),
            self.teacher.change("guilty"),
        )
        self.play(
            self.change_students("confused", None, "pondering", look_at=self.screen)
        )
        self.wait()
        self.play(self.teacher.change("shruggie"))
        self.play(
            self.change_students("sassy", "maybe", "sassy"),
        )
        self.wait(3)


class ShoutSoftmax(TeacherStudentsScene):
    def construct(self):
        self.play(LaggedStart(
            self.students[0].change("happy"),
            self.students[1].change("hooray"),
            self.students[2].says("Softmax!", mode="surprised", bubble_config=dict(buff=0.5, direction=LEFT)),
            self.teacher.change("well")
        ))
        self.wait(5)


class LeftArcSmaller(InteractiveScene):
    def construct(self):
        # Test
        arrow = Arrow(RIGHT, LEFT, path_arc=1.0 * PI, stroke_color=RED, stroke_width=8)
        self.play(ShowCreation(arrow))
        self.wait()


class SetThemToZero(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.students[0].says("Set them to 0?", mode="maybe"),
            self.students[1].change("pondering", look_at=self.screen),
            self.students[1].change("pondering", look_at=self.screen),

        )
        self.wait()
        self.play(
            self.teacher.says("Then they wouldn't\nbe normalized", mode="tease"),
        )
        self.wait(3)


class CalledMasking(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.says(TexText(R"This is called\\``masking''")),
            self.change_students(
                "pondering", "confused", "erm", look_at=self.screen,
            )
        )
        self.wait(5)


class ReferenceLargerContextTechnologies(InteractiveScene):
    def construct(self):
        # Test
        words = VGroup(
            Text("Sparse Attention Mechanisms"),
            Text("Blockwise Attention"),
            Text("Linformer"),
            Text("Reformer"),
            Text("Ring attention"),
            Text("Longformer"),
            Text("Adaptive Attention Span"),
            Tex(R"\vdots")
        )
        words.arrange(DOWN, aligned_edge=LEFT, buff=MED_LARGE_BUFF)
        words[-1].shift(0.5 * RIGHT)

        self.play(
            LaggedStartMap(FadeIn, words, shift=0.5 * DOWN, lag_ratio=0.5, run_time=4)
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeOut, words, shift=RIGHT, lag_ratio=0.1)
        )
        self.wait()


class AskAboutCrossAttention(TeacherStudentsScene):
    def construct(self):
        stds = self.students
        self.play(
            stds[0].change("hesitant", look_at=stds[1].eyes),
            stds[1].says("What about\ncross-attention?", bubble_config=dict(buff=0.5), mode="raise_left_hand"),
            stds[2].change("pondering", look_at=stds[1].eyes),
            self.teacher.change("well", look_at=stds[1].eyes)
        )
        self.wait(5)


class SelfVsCrossFrames(InteractiveScene):
    def construct(self):
        # Add screens
        self.add(FullScreenRectangle())
        screens = ScreenRectangle().replicate(2)
        screens.set_fill(BLACK, 1)
        screens.set_stroke(WHITE, 2)
        screens.set_height(0.45 * FRAME_HEIGHT)
        screens.arrange(RIGHT, buff=0.5)
        self.add(screens)

        # Add titles
        titles = VGroup(
            Text("Self-attention", font_size=60),
            Text("Cross-attention", font_size=60),
        )
        for title, screen in zip(titles, screens):
            title.next_to(screen, UP, buff=MED_LARGE_BUFF)

        self.play(Write(titles[0]))
        self.wait()
        self.play(TransformMatchingStrings(titles[0].copy(), titles[1]))
        self.wait()


class OngoingTranscription(InteractiveScene):
    def construct(self):
        phrase = Text("or maybe audio input of speech, and an ongoing transcription")
        words = break_into_words(phrase)
        for word in words:
            self.add(word)
            self.wait(0.1 * len(word))
        self.wait()


class ReferenceStraightforwardValueMatrix(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        morty.body.insert_n_curves(100)
        self.play(
            morty.change("raise_right_hand"),
            self.change_students("happy", "well", "tease", look_at=3 * UR)
        )
        self.wait(3)
        self.play(
            morty.change("hesitant"),
            self.change_students("erm", "hesitant", "guilty", look_at=3 * UR)
        )
        self.wait(5)


class SeekingMatchedParameters(TeacherStudentsScene):
    def construct(self):
        # Test
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)
        equation = VGroup(
            Text("# Value params").set_color(RED),
            Tex("=", font_size=72).rotate(PI / 2),
            Text("(# Query params) + (# Key params)"),
        )
        equation[2].scale(0.75)
        equation[2]["# Query params"].set_color(YELLOW)
        equation[2]["# Key params"].set_color(TEAL)
        equation.arrange(DOWN, buff=MED_LARGE_BUFF)
        equation.move_to(self.hold_up_spot, DOWN)

        self.play(
            self.teacher.change("raise_right_hand", equation),
            FadeIn(equation, UP),
            self.change_students("erm", "confused", "sassy", look_at=equation),
        )
        self.wait(2)
        self.play(
            self.change_students("pondering", "confused", "hesitant", look_at=self.screen)
        )
        self.wait(4)
        self.play(
            self.change_students("erm", "confused", "sassy", look_at=equation)
        )
        self.wait(4)


class HeadName(InteractiveScene):
    def construct(self):
        # Test
        title = Text("One head of attention", font_size=72)
        title.to_edge(UP)
        head = title["head"][0]
        self.play(
            Write(title, run_time=1)
        )
        self.play(
            FlashAround(head, time_width=2, run_time=2),
            head.animate.set_color(YELLOW),
        )
        self.wait()


class DInputAndOutputOfValue(InteractiveScene):
    def construct(self):
        # Test
        d_embed = 12_288
        in_label, out_label = [
            VGroup(Text(text), Integer(d_embed))
            for text in ["d_input", "d_output"]
        ]
        for label, shift in [(in_label, LEFT), (out_label, RIGHT)]:
            label.arrange(DOWN)
            label.scale(0.65)
            label.next_to(ORIGIN, UP, buff=LARGE_BUFF)
            label.shift(1.0 * shift)
            arrow = Arrow(label, 0.5 * shift)
            label.add(arrow)

        self.play(FadeIn(in_label, lag_ratio=0.1))
        self.wait()
        self.play(FadeIn(out_label, lag_ratio=0.1))
        self.wait()


class NowRepeatManyTimes(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen),
        )
        self.wait()
        self.play(
            self.teacher.says("Now do that about\n10,000 times"),
            self.change_students("droopy", "erm", "well", look_at=self.teacher.eyes)
        )
        self.wait(5)


class ALotToHoldInYouHead(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.says("It's a lot to\nhold in your head!", mode="surprised"),
            self.change_students("confused", "erm", "dejected", look_at=self.screen),
        )
        self.wait(5)


class ReactToMHSA(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.change("hesitant"),
            self.change_students("sad", "confused", "dejected", look_at=self.screen)
        )
        self.wait(3)
        self.play(
            self.change_students("guilty", "maybe", "erm")
        )
        self.wait(3)


class AskAboutOutput(TeacherStudentsScene):
    random_seed = 3
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.play(
            stds[0].change("hesitant", look_at=stds[1].eyes),
            stds[1].says("What about the\nOutput matrix?", mode="raise_left_hand"),
            stds[2].change("hesitant", look_at=stds[1].eyes),
        )
        self.play(
            morty.change("concentrating")
        )
        self.play(Blink(morty))
        self.wait(5)


class OneThirdOfWhatYouNeed(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle().fix_in_frame())
        title = Text("Attention is All You Need", font_size=72)
        all_word = title["All"][0]
        cross = Line(all_word.get_left(), all_word.get_right())
        cross.set_stroke(RED, 8)
        correction = Text("About 1/3 of What", font_size=60)
        correction.set_color(RED)
        correction.next_to(all_word, UP, MED_LARGE_BUFF)
        lines = VGroup(
            CubicBezier(
                all_word.get_corner(UP + v),
                all_word.get_corner(UP + v) + 0.5 * UP,
                correction.get_corner(DOWN + v) + 0.5 * DOWN,
                correction.get_corner(DOWN + v),
            )
            for v in [LEFT, RIGHT]
        )
        lines.set_stroke(RED, 2)

        self.add(title)
        self.wait()
        self.add(all_word, cross)
        self.play(ShowCreation(cross), all_word.animate.set_fill(opacity=0.5))
        self.play(
            FadeTransform(all_word.copy(), correction),
            ShowCreation(lines, lag_ratio=0),
        )
        self.wait()
        self.play(self.frame.animate.set_y(-3.75).set_height(11), run_time=2)
        self.wait()


class MoreResourcesBelow(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle())
        words = Text("More resources below", font_size=72)
        words.move_to(UP)
        arrows = Vector(1.5 * DOWN, stroke_width=10).get_grid(1, 3, buff=1.5)
        arrows.next_to(words, DOWN, buff=MED_LARGE_BUFF)
        morty = Mortimer()
        morty.body.insert_n_curves(100)
        morty.to_corner(DR)

        self.add(words)
        self.play(
            LaggedStartMap(GrowArrow, arrows, lag_ratio=0.5),
            morty.change("thinking", look_at=4 * DOWN)
        )
        self.play(Blink(morty))
        self.wait()


class PatreonEndScreen(EndScreen):
    pass


# MLP Chapter


class HowAndWhere(TeacherStudentsScene):
    def construct(self):
        # Test
        self.remove(self.background)
        self.play(
            self.students[1].says("How?", mode="raise_left_hand", look_at=self.screen),
            self.teacher.change("tease"),
            self.students[2].change("pondering", look_at=self.screen),
        )
        self.play(
            self.students[0].says("Where?", mode="maybe", look_at=self.screen),
        )
        self.wait(3)


class IntroducingMLPs(TeacherStudentsScene):
    def construct(self):
        # Look at screen
        morty = self.teacher
        screen = self.screen
        self.play(
            morty.change("raise_right_hand", screen),
            self.change_students("pondering", "confused", "pondering", look_at=screen),
        )
        self.wait(2)

        # Computation vs. interpretation
        words = VGroup(Text("Computation"), Text("Interpretation"))
        words.scale(1.5)
        words.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        words.to_corner(UR).shift(LEFT)
        check = Checkmark()
        check.match_height(words[0])
        check.next_to(words[0], RIGHT)
        check.set_color(GREEN)
        warning = SVGMobject("warning")
        warning.set_color(RED)
        warning.match_height(words[1])
        warning.next_to(words[1], RIGHT)

        self.play(
            FadeIn(words[0], UP),
            self.change_students("tease", "happy", "thinking", look_at=words[0]),
            morty.change("raise_left_hand", words[0]),
        )
        self.play(Write(check, stroke_color=GREEN))
        self.wait(3)
        self.play(
            FadeIn(words[1], UP),
            self.change_students("erm", "confused", "pondering", words[1]),
            morty.change("maybe", words[1])
        )
        self.play(Write(warning, stroke_color=RED))
        self.wait(5)


class ReferenceFactStorage(TeacherStudentsScene):
    def construct(self):
        # Look at screen
        morty = self.teacher
        screen = self.screen
        self.play(
            morty.change("raise_right_hand", screen),
            self.change_students("pondering", "confused", "pondering", look_at=screen),
        )
        self.wait(4)

        # Hold up words
        words = Text("Store a fact", font_size=72)
        words.next_to(morty, UP, LARGE_BUFF).shift_onto_screen()

        self.play(
            morty.change("raise_left_hand"),
            FadeIn(words, UP),
            self.change_students("erm", "maybe", "sassy", look_at=morty.eyes),
        )
        self.look_at(self.screen)
        self.wait(3)
        self.play(morty.change("tease"))
        self.wait(2)

        # Relax
        self.play(
            FadeOut(words, DOWN),
            self.change_students("pondering", "tease", "happy", look_at=self.screen),
        )
        self.wait(3)


class LookingAtPreview(TeacherStudentsScene):
    def construct(self):
        # Test
        bubble = ThoughtBubble(filler_shape=(5, 2.5))
        bubble.flip()
        bubble.pin_to(self.students[2])
        bubble.to_edge(LEFT)
        self.play(
            self.teacher.change("raise_right_hand"),
            self.change_students("pondering", "confused", "pondering", look_at=bubble),
        )
        self.play(FadeIn(bubble, lag_ratio=0.1))
        self.play(self.teacher.change("tease"))
        self.wait(2)
        self.play(self.change_students("erm", "pondering", "thinking", look_at=bubble))
        self.wait(3)


class RefreshersNeverHurt(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.change_students("confused", "horrified", "sad", look_at=self.screen),
        )
        self.wait(2)
        self.play(
            self.teacher.says(TexText(R"Let's do a\\quick refresher"), mode="tease"),
            self.change_students("pondering", "hesitant", "erm", look_at=self.teacher.eyes)
        )
        self.wait(3)


class EmbeddingLabel(InteractiveScene):
    def construct(self):
        # Background
        bg = ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2024/transformers/images/EmbeddingStill.jpg")
        bg.set_height(FRAME_HEIGHT)
        # self.add(bg)

        # Label
        ghost_vect = Rectangle()
        ghost_vect.set_shape(0.8, 4)
        ghost_vect.move_to([4.25, -1.0, 0])

        brace = Brace(ghost_vect, LEFT)
        name = brace.get_text("Embedding")
        length = Integer(12288)
        length.next_to(brace, LEFT, buff=0.5).shift(0.25 * UP)
        numbers_label = Text("Numbers")
        numbers_label.next_to(length, DOWN)
        gpt3_label = Text("(Length in GPT-3)", font_size=24)
        gpt3_label.next_to(length, UP, buff=1.0),
        gpt3_label.set_color(YELLOW)
        arrow = Arrow(gpt3_label.get_bottom(), length.get_top(), buff=0.1)

        self.play(
            GrowFromCenter(brace),
            Write(name)
        )
        self.wait()
        self.play(
            FadeTransform(name, numbers_label),
            CountInFrom(length, 0, run_time=1.5),
        )
        self.play(
            FadeIn(gpt3_label, lag_ratio=0.1),
            GrowFromCenter(arrow),
        )
        self.wait(1.5)


class ThatWhichDoesntKillHeader(InteractiveScene):
    def construct(self):
        # Test
        words = Text("That which does not kill you only makes you")
        words.to_edge(UP)
        rect = SurroundingRectangle(words["you"][-1], buff=0.1)
        rect.set_stroke(BLUE, 3)
        rect.set_fill(BLUE, 0.5)
        arrow = Arrow(rect.get_bottom(), rect.get_bottom() + 2 * DL)
        arrow.match_color(rect)

        brace = Brace(rect, DOWN, buff=0.1)

        self.add(rect, words, brace)


class QuickAttentionDescription(InteractiveScene):
    def construct(self):
        # To be added standing on an Attention block
        morty = Mortimer(height=2)
        morty.move_to(DOWN + LEFT)
        morty.flip()
        self.play(morty.says("Incorporate context", look_at=4 * DOWN))
        for x in range(2):
            self.play(Blink(morty))
            self.wait(2)


class QuickMLPDescription(InteractiveScene):
    def construct(self):
        # To be added standing on an MLP block
        morty = Mortimer(height=2, color=GREY_C)
        morty.move_to(DOWN + RIGHT)
        self.play(morty.says("More\ncomputation", mode="maybe", look_at=4 * DOWN))
        for x in range(2):
            self.play(Blink(morty))
            self.wait(2)


class ContrastBetweenSimpleComputationDifficultInterpretation(InteractiveScene):
    def construct(self):
        morty = Mortimer(height=2).to_edge(DOWN, buff=1.0)
        morty.body.insert_n_curves(100)
        items = VGroup(
            VGroup(Text("Computation"), Checkmark().set_height(0.5).set_color(GREEN)),
            VGroup(Text("Interpretation"), SVGMobject("warning").set_color(RED).set_height(0.5)),
        )
        for item, vect in zip(items, [LEFT, RIGHT]):
            item.scale(0.75)
            item.arrange(RIGHT)
            item.next_to(morty, UP + vect, buff=0.5)
            item.shift(-1.0 * vect * RIGHT)


        self.play(
            morty.change("raise_right_hand", items[0]),
            FadeIn(items[0], UP)
        )
        self.play(Blink(morty))
        self.wait()
        self.play(
            morty.change("raise_left_hand", items[1]),
            FadeIn(items[1], UP),
            items[0].animate.fade(0.5),
        )
        self.play(Blink(morty))
        self.wait()


class AmbientChangingDots(InteractiveScene):
    def construct(self):
        # Test
        dots = Dot().get_grid(20, 30)
        dots.set_height(8)
        dots.set_fill(opacity=0.5)
        dots.phases = np.random.uniform(0, TAU, len(dots))
        dots.freqs = np.random.uniform(0.3, 0.8, len(dots))

        def update_dots(dots):
            for dot, phase, freq in zip(dots, dots.phases, dots.freqs):
                dot.set_fill(opacity=np.cos(phase + freq * self.time)**2)
            return dots

        dots.add_updater(update_dots)
        self.add(dots)
        self.wait(30)


class MakeSomeAssumptions(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.teacher.says("We need a\nfew assumptions"),
            self.change_students("pondering", "sassy", "tease", look_at=self.screen)
        )
        self.play(self.teacher.change("raise_right_hand"))
        self.wait(6)


class MovingToSecondToken(InteractiveScene):
    def construct(self):
        # Add phrase
        phrase = Text("Michael Jordan plays the sport of")
        phrase.to_edge(UP)
        tokens = break_into_tokens(phrase)
        rects = get_piece_rectangles(tokens, h_buff=0, leading_spaces=True)

        self.add(tokens, rects)

        # Add vectors
        embeddings = VGroup(
            NumericEmbedding().scale(0.5).next_to(rect, DOWN, LARGE_BUFF)
            for rect in rects
        )
        arrows = VGroup(Arrow(rect, emb, buff=0.1) for rect, emb in zip(rects, embeddings))

        self.add(arrows)
        self.add(embeddings)

        # Animate in
        self.play(
            Write(rects),
            LaggedStartMap(GrowArrow, arrows),
            LaggedStartMap(FadeIn, embeddings, shift=0.5 * DOWN),
        )
        self.wait()

        # Highlight two
        highlight_rect = SurroundingRectangle(VGroup(rects[:2], embeddings[:2]))

        self.play(
            ShowCreation(highlight_rect),
            tokens[2:].animate.fade(0.5),
            rects[2:].animate.fade(0.5),
            arrows[2:].animate.fade(0.5),
            embeddings[2:].animate.fade(0.5),
        )
        self.wait()

        # Attention
        self.play(
            LaggedStart(
                (ContextAnimation(e2, embeddings[0].get_entries(), path_arc=90 * DEGREES, lag_ratio=0.1, min_stroke_width=2)
                for e2 in embeddings[1].get_entries()),
                lag_ratio=0.1
            ),
            RandomizeMatrixEntries(embeddings[1]),
            run_time=4
        )
        self.play(
            highlight_rect.animate.surround(VGroup(rects[1], embeddings[1]), buff=0)
        )
        self.wait()


class WhatAboutBiggerThanOne(TeacherStudentsScene):
    def construct(self):
        # Test
        self.screen.set_x(0)
        self.play(
            self.students[0].change("pondering", self.screen),
            self.students[1].says("And if it's\nbigger than 1?", mode="sassy", bubble_direction=RIGHT),
            self.students[2].change("erm", self.screen),
            self.teacher.change("guilty"),
        )
        self.wait(2)
        self.play(
            # self.teacher.says("Don't worry\nabout it", mode="maybe")
            self.teacher.change("maybe")
        )
        self.play(
            self.change_students("hesitant", "sassy", "angry")
        )
        self.wait(3)


class HighlightRect(InteractiveScene):
    def construct(self):
        rect = Rectangle(1, 3)
        rect.scale(0.5)
        rect.set_stroke(MAROON_B, 3)
        self.play(ShowCreation(rect))
        self.wait()


class AskWhy(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)

        # Test
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)
        self.play(
            self.teacher.change("guilty"),
            self.students[2].says(Text("Um, Why?", font_size=72), mode="maybe", look_at=self.screen),
            self.students[0].change("confused", self.screen),
            self.students[1].change("erm", self.screen),
        )
        self.wait()
        self.play(self.teacher.change("tease"))
        self.wait(2)


class GPT3Logo(InteractiveScene):
    def construct(self):
        # Test
        gpt3_label = Text("GPT-3", font="Consolas", font_size=72)
        openai_logo = SVGMobject("OpenAI.svg")
        openai_logo.set_fill(WHITE)
        openai_logo.set_height(2.0 * gpt3_label.get_height())
        title = VGroup(openai_logo, gpt3_label)
        title.arrange(RIGHT)
        title.to_edge(UP)

        self.play(
            FadeIn(gpt3_label, lag_ratio=0.1),
            Write(openai_logo, stroke_color=BLUE, stroke_width=0.5),
        )
        self.wait()


class AndGate(InteractiveScene):
    def construct(self):
        # Test
        gate = SVGMobject("and_gate")
        gate.set_fill(WHITE).set_stroke(width=0)
        name = Text("AND\nGate", font_size=96, alignment="LEFT")
        name.next_to(gate, RIGHT, LARGE_BUFF)
        self.play(
            Write(gate),
            FadeIn(name, lag_ratio=0.1, time_span=(0, 2)),
            run_time=3
        )
        self.wait()


class MJFactsAsVectorSum(InteractiveScene):
    def construct(self):
        # Test
        facts = VGroup(
            Tex(Rf"\overrightarrow{{\text{{{fact}}}}}")
            for fact in [
                "Basketball",
                "Chicago Bulls",
                "Number 23",
                "Born 1963",
            ]
        )
        facts.add(Tex(R"\vdots"))
        facts.arrange(DOWN, buff=0.75)
        colors = ["#F88158", "#CE1141", YELLOW, GREY, WHITE]
        for fact, color in zip(facts, colors):
            fact.set_color(color)

        plusses = Tex(R"+").replicate(len(facts) - 1)
        for f1, f2, plus in zip(facts, facts[1:], plusses):
            plus.move_to(midpoint(f1.get_bottom(), f2.get_top()))

        self.add(facts[0])
        for fact, plus in zip(facts[1:], plusses):
            self.play(
                FadeIn(fact, shift=0.5 * DOWN),
                Write(plus),
                run_time=1,
            )
            self.wait()


class AskAboutBias(TeacherStudentsScene):
    def construct(self):
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)
        self.play(
            self.students[0].change("erm", look_at=self.screen),
            self.students[1].change("confused", look_at=self.screen),
            self.students[2].says("What's that\nbias doing?", look_at=self.screen, bubble_direction=LEFT),
        )
        self.play(
            self.teacher.change("maybe")
        )
        self.wait(4)
        self.play(
            self.change_students("sassy", "maybe", "pondering", look_at=self.screen)
        )
        self.wait(4)


class ThatsIt(TeacherStudentsScene):
    def construct(self):
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)
        self.play(
            self.teacher.says("That's it!", mode="hooray", look_at=self.students),
            self.change_students("happy", "thinking", "well", look_at=self.screen)
        )
        self.wait()
        self.play(self.teacher.debubble(mode="raise_right_hand", look_at=self.screen))
        self.wait()
        self.play(
            self.change_students("thinking", "tease", "happy", look_at=self.screen)
        )
        self.wait(3)


class AddTwoMatrixSizes(InteractiveScene):
    def construct(self):
        # Test
        rect = Rectangle(3.0, 1.0)
        rect.set_stroke(BLUE, 3)
        total = Integer(2 * 4 * (12288**2))
        total.set_color(BLUE)
        total.next_to(rect, UP)

        self.play(
            ShowCreation(rect),
            Write(total)
        )
        self.wait()


class ReflectOnTwoThings(TeacherStudentsScene):
    def construct(self):
        # Initial reactions
        morty = self.teacher
        screen = self.screen
        stds = self.students

        morty.change_mode("raise_right_hand").look_at(self.screen)
        for std in stds:
            std.change_mode("happy")
        self.play(
            self.change_students("pondering", "thinking", "happy", look_at=screen)
        )
        self.wait(2)

        # Reflection points
        points = VGroup(
            Text("Two points of reflection"),
            Text("1."),
            Text("2."),
        )

        points[0].add(Underline(points[0], buff=-0.05))
        points[0].scale(1.25)
        points[0].set_color(YELLOW)

        dials = VGroup(Dial(initial_value=random.random()) for n in range(10))
        dials.set_height(0.5)
        dials.arrange(RIGHT)
        dials.set_flat_stroke(True)
        dials[-2].become(Tex(R"\dots").replace(dials[-2], dim_to_match=0))
        dials.next_to(points[1], RIGHT)
        points[1].add(dials)

        vectors = self.get_vectors()
        points[2].add(vectors)

        points.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        points.to_edge(UP)
        points[1:].shift(MED_SMALL_BUFF * RIGHT + 0.5 * DOWN)

        self.play(
            morty.change("tease", points[0]),
            self.change_students("erm", "plain", "hesitant", look_at=points[0]),
            Write(points[0], stroke_color=YELLOW_B),
        )
        self.wait(2)
        self.play(
            Write(points[1][:2]),
            LaggedStartMap(FadeIn, dials, lag_ratio=0.25),
            self.change_students("tease", "plain", "erm", look_at=points[1]),
            morty.change("raise_right_hand", points[1]),
        )
        self.wait(2)

        # Show vector clump
        self.play(
            VFadeIn(points[2]),
            Rotate(vectors, PI, axis=UP, run_time=8),
            self.change_students("confused", "hesitant", "erm", look_at=points[2]),
            morty.change("surprised", points[2]),
        )
        self.wait(3)

    def get_vectors(self):
        dodec = Dodecahedron()
        vectors = VGroup()
        for face in dodec:
            # for vert in face.get_anchors():
            for vert in [face.get_center()]:
                if not any([np.isclose(vert, v.get_end()).all() for v in vectors]):
                    vect = Vector(vert)
                    vect.set_color(random_bright_color(hue_range=(0.5, 0.7)))
                    vect.always.set_perpendicular_to_camera(self.frame)
                    vectors.add(vect)
        vectors.rotate(25 * DEGREES, axis=UR)
        vectors.set_height(1.5)
        return vectors


class RotatingVectors(ReflectOnTwoThings):
    def construct(self):
        self.clear()
        # Test
        vectors = self.get_vectors()
        vectors.set_height(4)

        self.play(
            Rotate(vectors, TAU, axis=UP, run_time=25, rate_func=linear),
        )


class AskIfThisIsReal(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students

        # Test
        self.play(
            stds[0].says("Is this how ChatGPT stores facts?"),
            morty.change("well"),
        )
        self.look_at(self.screen)
        self.wait()
        self.play(
            self.change_students("speaking", "pondering", "skeptical", look_at=self.screen)
        )
        self.wait(2)
        self.play(
            morty.says("Almost certainly\nnot quite...", mode="guilty", bubble_direction=RIGHT),
            stds[1].change("angry"),
            stds[2].change("erm"),
        )
        self.wait(4)


class SingleNeuronVsMultiple(InteractiveScene):
    def construct(self):
        # Add network
        radius = 0.1
        layers = VGroup(
            Dot(radius=radius).get_grid(n, 1, buff=radius)
            for n in [8, 16, 8]
        )
        layers.arrange(RIGHT, buff=2.0)
        layers.set_stroke(WHITE, 1)
        for layer in layers:
            for dot in layer:
                dot.set_fill(opacity=random.random())

        connections = VGroup(
            get_network_connections(layers[i], layers[i + 1])
            for i in (0, 1)
        )

        network = VGroup(layers, connections)
        network.set_height(5)
        network.center()

        self.add(network)

        # Show first neuron light up
        rect = SurroundingRectangle(layers[1][0])
        name = Text("Michael Jordan")
        name.next_to(rect, UP, SMALL_BUFF)
        name.save_state()
        for letter, dot in zip(*make_even(name, layers[0])):
            letter.move_to(dot)
            letter.set_opacity(0)

        thick_connections = connections.copy()
        for group in thick_connections:
            for line in group:
                line.set_stroke(width=2 * line.get_stroke_width(), opacity=1)
                line.insert_n_curves(20)
        self.play(
            LaggedStartMap(
                VShowPassingFlash,
                thick_connections[0],
                lag_ratio=1 / len(thick_connections[0]),
                time_width=2.0,
            ),
            layers[1][0].animate.set_fill(opacity=1),
            layers[1][1:].animate.set_fill(opacity=0),
            Restore(name, lag_ratio=0.05),
            run_time=2
        )
        self.play(ShowCreation(rect))
        self.wait()
        network.add(rect, name)

        # Split the image
        network_copy = network.copy()
        for dot in network_copy[0][1]:
            dot.set_fill(opacity=random.random())
        network_copy.to_edge(RIGHT)
        network_copy[-2].become(SurroundingRectangle(network_copy[0][1]))

        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        v_line.set_stroke(WHITE, 2)
        check = Checkmark().set_fill(GREEN).scale(2)
        ex = Exmark().set_fill(RED).scale(2)
        check.next_to(network_copy[-1], RIGHT)
        ex.move_to(check).shift(0.5 * FRAME_WIDTH * LEFT)

        self.play(
            network.animate.to_edge(LEFT),
            TransformFromCopy(network, network_copy),
            ShowCreation(v_line),
        )
        self.play(LaggedStart(
            Write(ex, stroke_color=RED),
            Write(check, stroke_color=GREEN),
            lag_ratio=0.5
        ))
        self.wait()


class WriteSuperposition(InteractiveScene):
    def construct(self):
        # Test
        word = Text("Superposition", font_size=120)
        outline = word.copy()
        outline.set_stroke(TEAL, 3)
        outline.set_fill(opacity=0)

        self.play(
            FadeIn(word, lag_ratio=0.1),
            LaggedStartMap(
                VShowPassingFlash,
                outline,
                time_width=2,
                run_time=5,
                lag_ratio=0.01
            )
        )
        self.wait()


class JohnsonLindenstraussName(InteractiveScene):
    def construct(self):
        # Test
        text = VGroup(
            Text("Johnson–Lindenstrauss\nLemma"),
            Tex(R"\Rightarrow", font_size=120),
        )
        text[0].set_color(RED_B)
        text.arrange(RIGHT, buff=MED_LARGE_BUFF)
        self.play(
            FadeIn(text[0], lag_ratio=0.1),
            Write(text[1], run_time=1)
        )


class ContrastGPTDimensionSizes(InteractiveScene):
    def construct(self):
        # Setup
        openai_logo = SVGMobject("OpenAI.svg")
        openai_logo.set_fill(WHITE)
        openai_logo.set_height(1.0)

        model_names = VGroup(
            Text("GPT-2"),
            Text("GPT-3"),
            Text("GPT-4"),
        )
        model_names.scale(1.25)
        model_names.arrange(RIGHT, buff=2.0)
        model_names.set_color(GREY_A)
        arrows = VGroup(
            Arrow(n1, n2, buff=0.25)
            for n1, n2 in zip(model_names, model_names[1:])
        )
        dim_counts = VGroup(
            Text(f"Model dim: {dim}", font_size=36)
            for dim in ["768", "12,288", "???"]
        )
        for model, count in zip(model_names, dim_counts):
            count.next_to(model, DOWN)

        arrows.add_to_back(Arrow().set_opacity(0))
        for name, count, arrow in zip(model_names, dim_counts, arrows):
            self.play(
                FadeIn(name),
                FadeIn(count, 0.5 * DOWN),
                GrowArrow(arrow)
            )
        self.wait()


class ReferenceSAP(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.background.scale(2)
        self.frame.scale(1.25, about_edge=DR)

        # Test
        bubble = stds[0].get_bubble("How would you\ntest this?", bubble_type=SpeechBubble)
        bubble.shift(0.5 * LEFT)
        self.play(LaggedStart(
            FadeIn(bubble, lag_ratio=0.1),
            stds[0].change("raise_left_hand"),
            stds[1].change("confused"),
            stds[2].change("maybe"),
            morty.change("tease")
        ))
        self.wait(2)
        self.play(
            morty.says(TexText(R"There's nice\\research using\\Sparse Autoencoders"), mode="hooray")
        )
        self.play(
            self.change_students(None, "pondering", "erm", look_at=morty.bubble),
        )
        self.wait()
        self.look_at(self.screen)
        self.wait(5)


class DetailsNotDiscussed(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Details not discussed")
        title.add(Underline(title))
        title.scale(1.25)
        title.set_color(RED)
        title.to_edge(UP).to_edge(RIGHT, buff=0)

        details = VGroup(
            Text("Tokenization"),
            Text("Positional encoding"),
            Text("Layer normalization"),
            Text("Training"),
        )
        dots = Dot().get_grid(len(details), 1, buff=0.75)
        dots.next_to(title, DOWN, buff=MED_LARGE_BUFF).shift(2 * LEFT)
        for detail, dot in zip(details, dots):
            detail.next_to(dot, RIGHT)
            detail.add_to_back(dot)
        vdots = Tex(R"\vdots")
        vdots.next_to(details, DOWN, MED_LARGE_BUFF).shift(LEFT)
        details.add(vdots)
        details.set_color(GREY_A)

        self.add(title)
        self.play(
            LaggedStartMap(FadeIn, details, shift=0.25 * DOWN, lag_ratio=0.5),
            run_time=4
        )
        self.wait()

        # Highlight training
        self.play(
            details[-2][1:].animate.scale(2, about_edge=LEFT).set_color(WHITE),
            details[-1].animate.shift(0.1 * DOWN),
            details[:-2].animate.set_opacity(0.5).scale(0.9, about_edge=UL),
        )
        self.wait()


class TriPanelWithPi(InteractiveScene):
    def construct(self):
        vlines = Line(UP, DOWN).replicate(2)
        vlines.set_height(FRAME_HEIGHT / 2)
        vlines.arrange(RIGHT, buff=FRAME_WIDTH / 3)
        vlines.to_edge(UP, buff=0)
        hline = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        hline.move_to(vlines, DOWN)
        lines = VGroup(*vlines, hline)

        lines.set_stroke(WHITE, 2)
        self.add(lines)

        # Test
        morty = Mortimer(mode="happy")
        morty.to_corner(DR).shift(3 * LEFT)

        self.play(morty.change("tease", 4 * UL))
        self.play(Blink(morty))
        self.wait()
        self.play(morty.change("coin_flip_2", 3 * UP))
        self.play(Blink(morty))
        self.wait()
        self.play(morty.change("hooray", 5 * UR).set_anim_args(path_arc=10 * DEGREES))
        self.play(Blink(morty))
        self.wait()


class WriteRLHF(InteractiveScene):
    def construct(self):
        words = VGroup(
            Text("Reinforcement"),
            Text("Learning with"),
            Text("Human"),
            Text("Feedback"),
        )
        words.scale(1.5)
        words.arrange(DOWN, aligned_edge=LEFT)
        words.set_color(GREY_A)
        self.play(LaggedStartMap(FadeIn, words, shift=0.25 * DOWN, lag_ratio=0.25))
        self.play(*(
            word[1:].animate.set_opacity(0.75)
            for word in words
        ))
        self.wait()


class ListOfFacts(InteractiveScene):
    def construct(self):
        # Title
        title = Text("Facts", font_size=120)
        title.to_corner(UL, buff=0.75)
        underline = Underline(title)
        underline.scale(1.3)
        underline.set_stroke(width=(0, 5, 5, 5, 0))
        title.add(underline)
        self.add(title)

        # List of facts
        n_facts = 15
        facts = VGroup(
            Text(line)
            for line in Path(DATA_DIR, "facts.txt").read_text().split("\n")[:n_facts]
        )
        facts.set_color(GREY_A)
        facts.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        facts.set_height(5.5)
        facts.next_to(title, DOWN, buff=0.5).align_to(title[0], LEFT)
        self.add(facts)

        # Add line to LLM
        vline = Line(UP, DOWN).replace(facts, dim_to_match=1)
        vline.next_to(facts, RIGHT, buff=2.0)
        vline.scale(0.7)

        lines = VGroup(
            Line(
                fact.get_right(),
                vline.pfp(a),
                path_arc=interpolate(-20, 20, a) * DEGREES,
                color=random_bright_color(hue_range=(0.1, 0.2))
            ).insert_n_curves(20).set_stroke(width=(0, 5, 5, 0))
            for fact, a in zip(
                facts,
                np.linspace(0, 1, len(facts))
            )
        )
        self.add(lines)
