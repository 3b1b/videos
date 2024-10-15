import gensim
import tiktoken
from pathlib import Path

from manim_imports_ext import *
from _2024.transformers.helpers import *


def get_token_encoding():
    return tiktoken.encoding_for_model("davinci")


def get_principle_components(data, n_components=3):
    covariance_matrix = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    order_of_importance = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, order_of_importance]  # sort the columns
    return sorted_eigenvectors[:, :n_components]


def find_nearest_words(model, vector, n=20):
    data = model.vectors
    indices = np.argsort(((data - vector)**2).sum(1))
    return [model.index_to_key[i] for i in indices[:n]]


def break_into_pieces(phrase_mob: Text, offsets: list[int]):
    phrase = phrase_mob.get_string()
    lhs = offsets
    rhs = [*offsets[1:], len(phrase)]
    result = []
    for lh, rh in zip(lhs, rhs):
        substr = phrase[lh:rh]
        start = phrase_mob.substr_to_path_count(phrase[:lh])
        end = start + phrase_mob.substr_to_path_count(substr)
        result.append(phrase_mob[start:end])
    return VGroup(*result)


def break_into_words(phrase_mob):
    offsets = [m.start() for m in re.finditer(" ", phrase_mob.get_string())]
    return break_into_pieces(phrase_mob, [0, *offsets])


def break_into_tokens(phrase_mob):
    tokenizer = get_token_encoding()
    tokens = tokenizer.encode(phrase_mob.get_string())
    _, offsets = tokenizer.decode_with_offsets(tokens)
    return break_into_pieces(phrase_mob, offsets)


def get_piece_rectangles(
    phrase_pieces,
    h_buff=0.05,
    v_buff=0.1,
    fill_opacity=0.15,
    fill_color=None,
    stroke_width=1,
    stroke_color=None,
    hue_range=(0.5, 0.6),
    leading_spaces=False,
):
    rects = VGroup()
    height = phrase_pieces.get_height() + 2 * v_buff
    last_right_x = phrase_pieces.get_x(LEFT)
    for piece in phrase_pieces:
        left_x = last_right_x if leading_spaces else piece.get_x(LEFT)
        right_x = piece.get_x(RIGHT)
        fill = random_bright_color(hue_range) if fill_color is None else fill_color
        stroke = fill if stroke_color is None else stroke_color
        rect = Rectangle(
            width=right_x - left_x + 2 * h_buff,
            height=height,
            fill_color=fill,
            fill_opacity=fill_opacity,
            stroke_color=stroke,
            stroke_width=stroke_width
        )
        if leading_spaces:
            rect.set_x(left_x, LEFT)
        else:
            rect.move_to(piece)
        rect.set_y(0)
        rects.add(rect)

        last_right_x = right_x

    rects.match_y(phrase_pieces)
    return rects


def get_word_to_vec_model(model_name="glove-wiki-gigaword-50"):
    filename = str(Path(DATA_DIR, model_name))
    if os.path.exists(filename):
        return gensim.models.keyedvectors.KeyedVectors.load(filename)
    model = gensim.downloader.load(model_name)
    model.save(filename)
    return model


def get_direction_lines(axes, direction, n_lines=500, color=YELLOW, line_length=1.0, stroke_width=3):
    line = Line(ORIGIN, line_length * normalize(direction))
    line.insert_n_curves(20).set_stroke(width=(0, stroke_width, stroke_width, stroke_width, 0))
    lines = line.replicate(n_lines)
    lines.set_color(color)
    for line in lines:
        line.move_to(axes.c2p(
            random.uniform(*axes.x_range),
            random.uniform(*axes.y_range),
            random.uniform(*axes.z_range),
        ))
    return lines


# For chapter 5


class LyingAboutTokens2(InteractiveScene):
    def construct(self):
        # Mention next word prediction task
        phrase = Text("The goal of our model is to predict the next word")

        words = break_into_tokens(phrase)
        rects = get_piece_rectangles(words, leading_spaces=True, h_buff=0)

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
                word.animate.match_color(rect)
                for word, rect in zip(words, rects)
            )),
            FadeOut(arrow)
        )
        self.wait()

        # Show words into vectors
        vectors = VGroup(*(
            NumericEmbedding(length=8)
            for word in words
        ))
        vectors.arrange(RIGHT, buff=1.0 * vectors[0].get_width())
        vectors.set_width(12)
        vectors.to_edge(DOWN, buff=1.0)
        vectors.to_edge(LEFT, buff=0.5)
        for vector, word in zip(vectors, words):
            vector.get_brackets().match_color(word[0])

        blocks = VGroup(*(VGroup(rect, word) for rect, word in zip(rects, words)))
        q_group = VGroup(rects[-1], q_marks)
        blocks.target = blocks.generate_target()
        for block, vector in zip(blocks.target, vectors):
            block.next_to(vector, UP, buff=1.5)

        arrows = VGroup(*(
            Arrow(block, vect, stroke_width=3)
            for block, vect in zip(blocks.target, vectors)
        ))

        self.play(
            MoveToTarget(blocks),
            q_group.animate.next_to(blocks.target, RIGHT, aligned_edge=UP),
            LaggedStartMap(FadeIn, vectors, shift=0.5 * DOWN),
            LaggedStartMap(GrowFromCenter, arrows),
        )
        self.wait()

        # Setup titles
        h_line = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        title1, title2 = titles = VGroup(
            Text("The Truth", font_size=72).to_edge(UP),
            Text("A Convenient Lie", font_size=72).next_to(h_line, DOWN),
        )
        h_line.set_stroke(WHITE, 2)
        h_line.next_to(titles[1], UP)
        for title in titles:
            title.add(Underline(title))

        # Show the lie
        phrase1, phrase2 = phrases = VGroup(
            Text("This process (known fancifully as tokenization) frequently subdivides words"),
            # Text("It's nice to sometimes pretend tokens are words"),
            Text("Let's pretend that tokens are always simply words"),
        )
        for phrase, title in zip(phrases, titles):
            phrase.set_width(FRAME_WIDTH - 1)
            phrase.next_to(title, DOWN, buff=1.0)

        tokens = break_into_tokens(phrase1)
        words = break_into_words(phrase2)
        token_rects = get_piece_rectangles(tokens, hue_range=(0.1, 0.2), leading_spaces=True, h_buff=0.0)
        word_rects = get_piece_rectangles(words, hue_range=(0.5, 0.6))

        self.play(
            FadeOut(blocks),
            FadeOut(q_group),
            FadeOut(arrows),
            FadeOut(vectors),
            ShowCreation(h_line),
            FadeIn(title1, lag_ratio=0.1),
            FadeIn(tokens),
        )
        self.add(token_rects, tokens)
        self.play(
            LaggedStartMap(FadeIn, token_rects),
            LaggedStart(*(
                token.animate.set_color(rect.get_color())
                for token, rect in zip(tokens, token_rects)
            ))
        )
        self.wait()
        self.play(
            FadeIn(title2, lag_ratio=0.1),
            FadeIn(words),
        )
        self.add(word_rects, words)
        self.play(
            LaggedStartMap(FadeIn, word_rects),
            LaggedStart(*(
                token.animate.set_color(rect.get_color())
                for token, rect in zip(words, word_rects)
            ))
        )
        self.wait()

        # Analyze tokenization
        brace = Brace(token_rects[8], buff=0.05)

        self.play(GrowFromCenter(brace))
        self.wait()
        for index in [2, 4, 5, 7, 8, 9, 11, 12]:
            self.play(brace.animate.become(Brace(token_rects[index], buff=0.05)))
            self.wait()


class DiscussTokenization(InteractiveScene):
    def construct(self):
        pass


class ImageTokens(InteractiveScene):
    n_divisions = 52

    def construct(self):
        # Add image
        image = ImageMobject("SmallFluffCreature")  # Change
        image.set_height(5)
        self.add(image)

        # Add pixels
        pixels = create_pixels(image, pixel_width=image.get_width() / self.n_divisions)
        big_pixels = create_pixels(image, pixel_width=image.get_width() / (self.n_divisions / 4))

        patches = big_pixels.copy().set_fill(opacity=0)
        p_points = np.array([p.get_center() for p in pixels])
        bp_points = np.array([bp.get_center() for bp in big_pixels])

        for pixel in pixels:
            dists = np.linalg.norm(bp_points - pixel.get_center(), axis=1)
            patches[np.argmin(dists)].add(pixel)

        # Anim test
        self.play(FadeIn(patches))
        self.remove(image)
        self.play(patches.animate.space_out_submobjects(2.0).scale(0.75))
        self.wait()
        self.play(LaggedStart(
            (patch.animate.set_stroke(TEAL, 3).set_anim_args(rate_func=there_and_back)
            for patch in patches),
            lag_ratio=5.0 / len(patches),
        ))
        self.wait()


class SoundTokens(InteractiveScene):
    def construct(self):
        # Add wave form
        n_lines = 100
        wave_form = Line(UP, DOWN).replicate(n_lines)
        wave_form.arrange(RIGHT)
        wave_form.arrange_to_fit_width(5)
        wave_form.next_to(ORIGIN, RIGHT)

        def func(x):
            x *= 1.7
            return sum([
                math.sin(x),
                0.5 * math.sin(2 * x),
                0.3 * math.sin(3 * x),
                0.2 * math.sin(4 * x),
                0.1 * math.sin(5 * x),
                0.15 * math.sin(6 * x),
            ])

        for line in wave_form:
            line.set_height(abs(func(line.get_x())))

        wave_form.center()
        self.add(wave_form)

        # Subdivide
        step = 5
        chunks = VGroup(wave_form[i:i + step] for i in range(0, len(wave_form), step))

        self.add(chunks)
        self.wait()
        self.play(chunks.animate.space_out_submobjects(2.0).scale(0.75))
        self.play(LaggedStart(
            (chunk.animate.set_stroke(TEAL, 3).scale(1.5).set_anim_args(rate_func=there_and_back)
            for chunk in chunks),
            lag_ratio=2.0 / len(chunks),
            run_time=2
        ))
        self.wait()


class IntroduceEmbeddingMatrix(InteractiveScene):
    def construct(self):
        # Load words
        words = [
            'aah',
            'aardvark',
            'aardwolf',
            'aargh',
            'ab',
            'aback',
            'abacterial',
            'abacus',
            'abalone',
            'abandon',
            'zygoid',
            'zygomatic',
            'zygomorphic',
            'zygosis',
            'zygote',
            'zygotic',
            'zyme',
            'zymogen',
            'zymosis',
            'zzz'
        ]

        # Get all words
        dots = Tex(R"\vdots")
        shown_words = VGroup(
            *map(Text, words[:10]),
            dots,
            *map(Text, words[-10:]),
        )
        shown_words.arrange(DOWN, aligned_edge=LEFT)
        dots.match_x(shown_words[:5])
        shown_words.set_height(FRAME_HEIGHT - 1)
        shown_words.move_to(LEFT)
        shown_words.set_fill(border_width=0)

        brace = Brace(shown_words, RIGHT)
        brace_text = brace.get_tex(R"\text{All words, } \sim 50\text{k}")

        self.play(
            LaggedStartMap(FadeIn, shown_words, shift=0.5 * LEFT, lag_ratio=0.1, run_time=2),
            GrowFromCenter(brace, time_span=(0.5, 2.0)),
            FadeIn(brace_text, time_span=(0.5, 1.5)),
        )
        self.wait()

        # Show embedding matrix
        dots_index = shown_words.submobjects.index(dots)
        matrix = WeightMatrix(
            shape=(10, len(shown_words)),
            ellipses_col=dots_index
        )
        matrix.set_width(13.5)
        matrix.center()
        columns = matrix.get_columns()

        matrix_name = Text("Embedding matrix", font_size=90)
        matrix_name.next_to(matrix, DOWN, buff=0.5)

        shown_words.target = shown_words.generate_target()
        shown_words.target.rotate(PI / 2)
        shown_words.target.next_to(matrix, UP)
        for word, column in zip(shown_words.target, columns):
            word.match_x(column)
            word.rotate(-45 * DEGREES, about_edge=DOWN)
        shown_words.target[dots_index].rotate(45 * DEGREES).move_to(
            shown_words.target[dots_index - 1:dots_index + 2]
        )
        new_brace = Brace(shown_words.target, UP, buff=0.0)
        column_rects = VGroup(*(
            SurroundingRectangle(column, buff=0.05)
            for column in columns
        ))
        column_rects.set_stroke(WHITE, 1)

        self.play(
            MoveToTarget(shown_words),
            brace.animate.become(new_brace),
            brace_text.animate.next_to(new_brace, UP, buff=0.1),
            LaggedStart(*(
                Write(column, lag_ratio=0.01, stroke_width=1)
                for column in columns
            ), lag_ratio=0.2, run_time=2),
            LaggedStartMap(FadeIn, matrix.get_brackets(), scale=0.5, lag_ratio=0)
        )
        self.play(Write(matrix_name, run_time=1))
        self.wait()

        # Show a few columns
        last_rect = VMobject()
        # for index in [9, -7, 7, -5, -6]:
        for index in range(len(columns)):
            for group in shown_words, columns:
                group.target = group.generate_target()
                group.target.set_opacity(0.2)
                group.target[index].set_opacity(1)
            rect = column_rects[index]
            self.play(
                *map(MoveToTarget, [shown_words, columns]),
                FadeIn(rect),
                FadeOut(last_rect),
            )
            last_rect = rect
            self.wait(0.5)
        self.play(
            FadeOut(last_rect),
            shown_words.animate.set_opacity(1),
            columns.animate.set_opacity(1),
        )

        # Label as W_E
        frame = self.frame
        lhs = Tex("W_E = ", font_size=90)
        lhs.next_to(matrix, LEFT)

        self.play(
            frame.animate.set_width(FRAME_WIDTH + 3, about_edge=RIGHT),
            Write(lhs)
        )
        self.wait()

        # Randomize entries
        rects = VGroup(*(
            SurroundingRectangle(entry).insert_n_curves(20)
            for entry in matrix.get_entries()
            if entry not in matrix.ellipses
        ))
        rects.set_stroke(WHITE, 1)
        for x in range(1):
            self.play(
                RandomizeMatrixEntries(matrix, lag_ratio=0.01),
                LaggedStartMap(VShowPassingFlash, rects, lag_ratio=0.01, time_width=1.5),
                run_time=2,
            )
        data_modifying_matrix(self, matrix)
        self.wait()

        # Highlight just one word
        matrix_group = VGroup(lhs, matrix, shown_words, matrix_name)
        index = words.index("aardvark")
        vector = VGroup(
            matrix.get_brackets()[0],
            matrix.get_columns()[index],
            matrix.get_brackets()[1],
        ).copy()
        vector.target = vector.generate_target()
        vector.target.arrange(RIGHT, buff=0.1)
        vector.target.set_height(4.5)
        vector.target.move_to(frame, DOWN).shift(0.5 * UP)
        vector.target.set_x(-3)

        word = shown_words[index].copy()
        word.target = word.generate_target()
        word.target.rotate(-45 * DEGREES)
        word.target.scale(3)
        word.target.next_to(vector.target, LEFT, buff=1.5)
        arrow = Arrow(word.target, vector.target)

        self.play(LaggedStart(
            matrix_group.animate.scale(0.5).next_to(frame.get_top(), DOWN, 0.5),
            FadeOut(brace, UP),
            FadeOut(brace_text, 0.5 * UP),
            MoveToTarget(word),
            MoveToTarget(vector),
            GrowFromPoint(arrow, word.get_center()),
        ), lag_ratio=0.15)
        self.wait()

        word_group = VGroup(word, arrow, vector)

        # Pull the matrix back up
        self.play(
            FadeOut(word_group, DOWN),
            matrix_group.animate.scale(2.0).move_to(frame)
        )

        # Have data fly across
        data_modifying_matrix(self, matrix, word_shape=(3, 10), alpha_maxes=(0.5, 0.9))
        self.wait()

        # Prep tokens
        encoding = get_token_encoding()
        n_vocab = encoding.n_vocab
        kw = dict(font_size=24)
        shown_tokens = VGroup(
            *(Text(encoding.decode([i]), **kw) for i in range(10)),
            shown_words[dots_index].copy().rotate(-45 * DEGREES),
            *(Text(encoding.decode([i]), **kw) for i in range(n_vocab - 10, n_vocab)),
        )
        for token, word in zip(shown_tokens, shown_words):
            token.rotate(45 * DEGREES)
            token.move_to(word, DL)
        shown_tokens[dots_index].move_to(
            shown_tokens[dots_index-1:dots_index + 2:2]
        )

        # Show dimensions
        top_brace = Brace(shown_words, UP)
        left_brace = Brace(matrix, LEFT, buff=SMALL_BUFF)
        vocab_count = Integer(50257)
        vocab_label = VGroup(vocab_count, Text("words"))
        vocab_label.arrange(RIGHT, aligned_edge=UP)
        vocab_label.next_to(top_brace, UP, SMALL_BUFF)
        token_label = Text("tokens", fill_color=YELLOW)
        token_label.move_to(vocab_label[1], LEFT)

        dim_count = Integer(12288)
        dim_count.next_to(left_brace, LEFT, SMALL_BUFF)

        self.play(
            GrowFromCenter(top_brace),
            CountInFrom(vocab_count, 0),
            FadeIn(vocab_label[1]),
        )
        self.wait()
        self.play(
            FadeOut(vocab_label[1], 0.5 * UP),
            FadeIn(token_label, 0.5 * UP),
            LaggedStartMap(FadeOut, shown_words, shift=0.25 * UP, lag_ratio=0.1),
            LaggedStartMap(FadeIn, shown_tokens, shift=0.25 * UP, lag_ratio=0.1),
        )
        self.wait()

        matrix_name.target = matrix_name.generate_target()
        matrix_name.target.shift(RIGHT)
        self.play(
            MoveToTarget(matrix_name),
            lhs.animate.next_to(matrix_name.target, LEFT),
            GrowFromCenter(left_brace),
            CountInFrom(dim_count, 0),
        )
        self.play(FlashAround(dim_count))
        self.wait()

        # Count total parameters
        matrix_group = VGroup(
            top_brace, vocab_count,
            left_brace, dim_count,
            matrix, shown_words, matrix_name, lhs
        )

        top_equation = VGroup(
            Text("Total parameters = "),
            dim_count.copy(),
            Tex(R"\times"),
            vocab_count.copy(),
            Tex("="),
            Integer(vocab_count.get_value() * dim_count.get_value()).set_color(YELLOW),
        )
        top_equation.arrange(RIGHT)
        top_equation.set_height(0.5)
        top_equation.next_to(matrix_group, UP, buff=1.0)
        result_rect = SurroundingRectangle(top_equation[-1])
        result_rect.set_stroke(YELLOW, 2)

        self.play(
            LaggedStartMap(FadeIn, top_equation[::2], shift=0.25 * UP, lag_ratio=0.5),
            TransformFromCopy(dim_count, top_equation[1]),
            TransformFromCopy(vocab_count, top_equation[3]),
            frame.animate.set_height(11).move_to(matrix_group, DOWN).shift(DOWN),
        )
        self.play(FadeTransform(
            top_equation[1:5:2].copy(), top_equation[-1]
        ))
        self.play(ShowCreation(result_rect))
        self.wait()


class Word2VecScene(InteractiveScene):
    default_frame_orientation = (-30, 70)

    axes_config = dict(
        x_range=(-5, 5, 1),
        y_range=(-5, 5, 1),
        z_range=(-4, 4, 1),
        width=8,
        height=8,
        depth=6.4,
    )
    label_rotation = PI / 2
    # embedding_model = "word2vec-google-news-300"
    embedding_model = "glove-wiki-gigaword-50"

    def setup(self):
        super().setup()

        # Load model
        self.model = get_word_to_vec_model(self.embedding_model)

        # Decide on basis
        self.basis = self.get_basis(self.model)

        # Add axes
        self.axes = ThreeDAxes(**self.axes_config)
        self.add(self.axes)

    def get_basis(self, model):
        return get_principle_components(model.vectors, 3).T

    def add_plane(self, color=GREY, stroke_width=1.0):
        axes = self.axes
        plane = NumberPlane(
            axes.x_range, axes.y_range,
            width=axes.get_width(),
            height=axes.get_height(),
            background_line_style=dict(
                stroke_color=color,
                stroke_width=stroke_width,
            ),
            faded_line_style=dict(
                stroke_opacity=0.25,
                stroke_width=0.5 * stroke_width,
            ),
            faded_line_ratio=1,
        )
        self.plane = plane
        self.add(plane)
        return plane

    def get_labeled_vector(
        self,
        word,
        coords=None,
        thickness=5,
        color=YELLOW,
        func_name: str | None = "E",
        buff=0.05,
        direction=None,
        label_config: dict = dict()
    ):
        # Return an arrow with word label next to it
        axes = self.axes
        if coords is None:
            coords = self.basis @ self.model[word.lower()]
        point = axes.c2p(*coords)
        label_config.update(label_buff=buff)
        if "label_rotation" not in label_config:
            label_config.update(label_rotation=self.label_rotation)
        arrow = LabeledArrow(
            axes.get_origin(),
            point,
            thickness=thickness,
            fill_color=color,
            label_text=word if func_name is None else f"{func_name}({word})",
            buff=0,
            direction=direction,
            **label_config,
        )
        arrow.always.set_perpendicular_to_camera(self.frame)
        return arrow


class AmbientWordEmbedding(Word2VecScene):
    def construct(self):
        # Setup
        frame = self.frame
        frame.reorient(-30, 82, 0)
        frame.add_ambient_rotation(3 * DEGREES)

        axes = self.axes
        axes.set_stroke(width=2)
        axes.set_height(7)
        axes.move_to(0.2 * FRAME_WIDTH * RIGHT + 1.0 * IN)

        # Add titles
        titles = VGroup(Text("Words"), Text("Vectors"))
        colors = [YELLOW, BLUE]
        titles.set_height(0.5)
        xs = [-4.0, axes.get_x()]
        for title, x, color in zip(titles, xs, colors):
            title.move_to(x * RIGHT)
            title.to_edge(UP)
            title.add(Underline(title))
            title.fix_in_frame()
            title.set_color(color)

        arrow = Arrow(titles[0], titles[1], buff=0.5)
        arrow.fix_in_frame()

        arrow_label = TexText("``Embedding''")
        arrow_label.set_submobject_colors_by_gradient(YELLOW, BLUE)
        arrow_label.next_to(arrow, UP, SMALL_BUFF)
        arrow_label.fix_in_frame()

        self.add(titles)
        self.add(arrow)

        # Add words
        words = "All data in deep learning must be represented as vectors".split(" ")
        pre_labels = VGroup(*(Text(word) for word in words))
        pre_labels.fix_in_frame()
        pre_labels.arrange(DOWN, aligned_edge=LEFT)
        pre_labels.next_to(titles[0], DOWN, buff=0.5)
        pre_labels.align_to(titles[0][0], LEFT)
        pre_labels.set_backstroke()

        coords = np.array([
            self.basis @ self.model[word.lower()]
            for word in words
        ])
        coords -= coords.mean(0)
        max_coord = max(coords.max(), -coords.min())
        coords *= 4.0 / max_coord

        embeddings = VGroup(*(
            self.get_labeled_vector(
                word,
                coord,
                stroke_width=2,
                color=interpolate_color(BLUE_D, BLUE_A, random.random()),
                func_name=None,
                label_config=dict(font_size=24)
            )
            for word, coord in zip(words, coords)
        ))

        self.play(LaggedStartMap(FadeIn, pre_labels, shift=0.2 * UP, lag_ratio=0.1, run_time=1))

        # Transition
        self.add(turn_animation_into_updater(
            Write(arrow_label, time_span=(1, 3))
        ))
        for label, vect in zip(pre_labels, embeddings):
            self.add(turn_animation_into_updater(
                TransformFromCopy(label, vect.label, run_time=2)
            ))
            self.add(turn_animation_into_updater(
                FadeIn(vect, run_time=1)
            ))
            self.wait(0.5)
        self.play(FlashAround(arrow_label, time_width=1.5, run_time=3))
        self.wait(15)


class ThreeDSpaceExample(InteractiveScene):
    def construct(self):
        # Set up axes
        frame = self.frame
        frame.reorient(-15, 78, 0, (1.07, 1.71, 1.41), 6.72)
        frame.add_ambient_rotation(1 * DEGREES)
        axes = ThreeDAxes((-5, 5), (-5, 5), (-4, 4))
        plane = NumberPlane((-5, 5), (-5, 5))
        plane.fade(0.5)

        self.add(plane)
        self.add(axes)

        # Show coordiantes creating directions
        x, y, z = coordinates = np.array([3, 1, 2])
        colors = [RED, GREEN, BLUE]

        coords = DecimalMatrix(np.zeros((3, 1)), num_decimal_places=1)
        coords.fix_in_frame()
        coords.to_corner(UR)
        coords.shift(1.5 * LEFT)
        coords.get_entries().set_submobject_colors_by_gradient(*colors)

        lines = VGroup(
            Line(axes.c2p(0, 0, 0), axes.c2p(x, 0, 0)),
            Line(axes.c2p(x, 0, 0), axes.c2p(x, y, 0)),
            Line(axes.c2p(x, y, 0), axes.c2p(x, y, z)),
        )
        lines.set_flat_stroke(False)
        lines.set_submobject_colors_by_gradient(*colors)
        labels = VGroup(*map(Tex, "xyz"))
        labels.rotate(89 * DEGREES, RIGHT)
        directions = [OUT, OUT + RIGHT, RIGHT]
        for label, line, direction in zip(labels, lines, directions):
            label.next_to(line, direction, buff=SMALL_BUFF)
            label.match_color(line)

        dot = GlowDot(color=WHITE)
        dot.move_to(axes.get_origin())

        vect = Arrow(axes.get_origin(), axes.c2p(x, y, z), buff=0)
        vect.set_flat_stroke(False)

        self.add(coords)
        for entry, line, label, value in zip(coords.get_entries(), lines, labels, coordinates):
            rect = SurroundingRectangle(entry)
            rect.set_fill(line.get_color(), 0.3)
            rect.set_stroke(line.get_color(), width=2)
            self.play(
                ShowCreation(line),
                FadeInFromPoint(label, line.get_start()),
                VFadeInThenOut(rect),
                ChangeDecimalToValue(entry, value),
                dot.animate.move_to(line.get_end()),
            )
            self.wait(0.5)
        self.play(ShowCreation(vect))

        # Wait for a bit
        self.wait(15)

        # Show many points
        points = GlowDots(np.random.uniform(-3, 3, size=(50, 3)), radius=0.1)
        frame.clear_updaters()
        self.play(
            FadeOut(coords),
            FadeOut(dot),
            FadeOut(plane),
            LaggedStartMap(FadeOut, VGroup(*lines, vect, *labels)),
            frame.animate.reorient(-81, 61, 0, (-0.82, 0.6, 0.36), 8.95),
            ShowCreation(points),
            run_time=2,
        )
        frame.add_ambient_rotation(5 * DEGREES)
        self.wait(2)

        # Take a 2d slice
        plane = Square3D()
        plane.set_height(10)
        plane.set_color([GREY_E, GREY_C])
        plane.set_opacity(0.25)
        grid = NumberPlane(
            (-5, 5), (-5, 5),
            background_line_style=dict(stroke_color=GREY_B, stroke_width=1),
            faded_line_ratio=0,
        )
        grid.axes.match_style(grid.background_lines)
        grid.match_height(plane)
        plane_group = Group(plane, grid)
        plane_group.rotate(60 * DEGREES, UR)

        bases = [
            normalize(point)
            for point in plane.get_points()[:2]
        ]

        def project(points):
            return np.array([
                sum(np.dot(point, b) * b for b in bases)
                for point in points
            ])

        projected_points = points.copy()
        projected_points.apply_points_function(project)

        projection_lines = VGroup(*(
            Line(p1, p2)
            for p1, p2 in zip(points.get_points(), projected_points.get_points())
        ))
        projection_lines.set_stroke()

        self.play(ShowCreation(plane), Write(grid, lag_ratio=0.01, stroke_width=1))
        self.wait(2)
        self.play(
            axes.animate.set_stroke(opacity=0.25),
            points.animate.set_opacity(0.5),
            TransformFromCopy(points, projected_points),
            ShowCreation(projection_lines, lag_ratio=0.05),
            run_time=3
        )
        self.play(
            FadeOut(points),
            FadeOut(projection_lines),
            FadeOut(axes)
        )
        self.wait(15)


class HighDimensionalSpaceCompanion(InteractiveScene):
    def construct(self):
        # Vector example
        word = Text("bank")
        vect = WeightMatrix(shape=(8, 1))
        vect.next_to(word, RIGHT, buff=LARGE_BUFF)
        vect.set_height(3)
        arrow = Arrow(word, vect)
        group = VGroup(word, arrow, vect)
        group.move_to(RIGHT)
        group.to_edge(UP, buff=0.1)
        self.add(group)

        # Draw vague embedding space
        bubble_center = np.array([0.5, -2.25, 0])
        base_bubble: VMobject = OldThoughtBubble()[-2][-1]
        base_bubble.set_shape(8, 7)
        base_bubble.rotate(PI)
        base_bubble.set_fill(GREY_D, opacity=[0.25, 1, 0.25])
        base_bubble.move_to(bubble_center)
        bubble_label = Text("Word vector space", font_size=60)
        bubble_label.move_to(base_bubble)
        bubble_label.shift(2.0 * UP)
        # bubble_label = Text("Embedding space", font_size=72)
        q_marks = Tex("???", font_size=120)
        q_marks.next_to(bubble_label, DOWN, buff=0.5)
        base_bubble.add(bubble_label, q_marks)

        def get_bubble():
            result = base_bubble.copy()
            result.apply_complex_function(
                lambda z: z * (1 + 0.025 * np.cos(5 * np.log(z).imag + self.time))
            )
            result.move_to(bubble_center)
            return result

        bubble = always_redraw(get_bubble)
        self.add(bubble)
        self.wait(10)

        # Show dimension
        brace = Brace(vect, RIGHT)
        label = VGroup(
            # Integer(12288),
            Integer(10000),
            Text("coordinates")
        )
        label.arrange(DOWN, aligned_edge=LEFT)
        label.set_height(1)
        label.set_color(YELLOW)
        label.next_to(brace, RIGHT)

        dimension_label = VGroup(
            label[0].copy(),
            Text("-dimensional")
        )
        dimension_label.arrange(RIGHT, buff=0.05, aligned_edge=UP)
        dimension_label.match_height(bubble_label).scale(0.8)
        dimension_label.set_color(YELLOW)
        dimension_label.next_to(q_marks, DOWN, buff=0.5)

        self.play(
            GrowFromCenter(brace),
            CountInFrom(label[0], 0),
            FadeIn(label[1]),
        )
        self.wait()
        self.play(
            TransformFromCopy(label[0], dimension_label[0]),
            FadeInFromPoint(dimension_label[1], label[0].get_center()),
        )
        self.remove(dimension_label)
        bubble_label.add(*dimension_label)

        self.wait(10)

        # Show 3d slice
        axes = ThreeDAxes()
        axes.rotate(20 * DEGREES, OUT)
        axes.rotate(80 * DEGREES, LEFT)
        axes.set_height(3)
        axes.move_to(bubble)
        axes.shift(0.5 * RIGHT)
        axes_label = TexText("3d ``slice''")
        axes_label.next_to(axes, RIGHT)
        axes_label.shift(0.35 * DOWN + 1.5 * LEFT)

        self.play(
            bubble_label.animate.scale(0.5).shift(1.2 * UP + 1.5 * LEFT),
            FadeOut(q_marks),
            Write(axes, lag_ratio=0.01),
            Write(axes_label)
        )
        self.wait(2)

        # Show some vector projections
        vectors = VGroup(*(
            Arrow(
                axes.get_origin(),
                axes.c2p(*np.random.uniform(-3, 3, 3)),
                buff=0,
                stroke_color=random_bright_color(hue_range=(0.55, 0.65))
            )
            for x in range(5)
        ))
        vectors.set_flat_stroke(False)

        self.play(LaggedStartMap(FadeIn, vectors, scale=0.5, lag_ratio=0.3))
        z_direction = axes.z_axis.get_vector()
        axes.add(vectors)
        self.play(Rotate(axes, -200 * DEGREES, axis=z_direction, run_time=10))


class LearningEmbeddings(Word2VecScene):
    def construct(self):
        # Setup
        self.add_plane()
        axes = self.axes
        plane = self.plane
        frame = self.frame
        frame.reorient(0, 90, 0)

        # Get sample words
        # phrase = "The first big idea is that as a model tweaks and tunes its weights"
        # phrase = "The big idea as a model tweaks and tunes its weights"
        phrase = "Features can be encoded with directions in a big space"
        words = [word.lower() for word in phrase.split(" ")]

        # Get initial and final states
        colors = [random_bright_color(hue_range=(0.5, 0.6)) for word in words]
        true_embeddings = np.array([
            self.basis @ self.model[word]
            for word in words
        ])
        true_embeddings -= true_embeddings.mean(0)
        true_embeddings *= 5 / np.abs(true_embeddings).max(0)

        np.random.seed(2)
        thetas = np.arange(0, TAU, TAU / len(words))
        thetas += np.random.uniform(-0.5, 0.5, thetas.size)
        amps = np.random.uniform(3, 5, thetas.size)
        initial_coords = [
            rotate_vector(amp * OUT, theta, axis=UP)
            for theta, amp in zip(thetas, amps)
        ]

        # Create word vectors
        word_vects = VGroup(*(
            self.get_labeled_vector(
                word,
                coords=coords,
                color=color,
                buff=0.05,
                func_name=None,
                label_config=dict(font_size=36)
            )
            for word, color, coords in zip(words, colors, initial_coords)
        ))
        labels = VGroup()
        for vect in word_vects:
            label = vect.label
            label.set_backstroke(BLACK, 3)
            label.vect = vect
            label.add_updater(lambda m: m.move_to(
                m.vect.get_end() + 0.25 * normalize(m.vect.get_vector())
            ))
            labels.add(label)

        self.play(
            LaggedStartMap(GrowArrow, word_vects, lag_ratio=0.2),
            LaggedStartMap(FadeIn, labels, lag_ratio=0.2),
            run_time=4
        )
        self.wait()

        # Tweak and tune weights
        turn_animation_into_updater(
            ApplyMethod(frame.reorient, 4, 72, 0, (-0.04, -0.18, -0.5), 8.00),
            run_time=8
        )
        self.progressive_nudges(word_vects, true_embeddings, 8)
        frame.clear_updaters()
        turn_animation_into_updater(
            ApplyMethod(frame.reorient, 38, 69, 0, (-0.32, 0.02, -0.54), 7.68),
            run_time=12
        )
        self.progressive_nudges(word_vects, true_embeddings, 12)

    def progressive_nudges(self, word_vects, true_embeddings, n_nudges, step_size=0.2):
        for x in range(n_nudges):
            anims = [
                vect.animate.put_start_and_end_on(
                    self.axes.get_origin(),
                    interpolate(vect.get_end(), self.axes.c2p(*embedding), step_size)
                )
                for vect, embedding in zip(word_vects, true_embeddings)
            ]
            self.play(*anims, run_time=0.5)
            self.wait(0.5)


class KingQueenExample(Word2VecScene):
    default_frame_orientation = (20, 70)

    def get_basis(self, model):
        basis = super().get_basis(model)
        basis[1] *= 2
        return basis

    def construct(self):
        # Axes and frame
        axes = self.axes
        frame = self.frame
        self.add_plane()
        self.plane.rotate(90 * DEGREES, LEFT)
        frame.reorient(-178, 9, 178, (2.15, 1.12, 0.56), 6.84)

        # Initial word vectors
        words = ["man", "woman", "king", "queen"]
        colors = [BLUE_B, RED_B, BLUE_D, RED_D]
        directions = [UR, RIGHT, UR, LEFT]
        all_coords = np.array([self.basis @ self.model[word] for word in words])
        all_coords[:2] += DOWN
        all_coords[2:] += 4 * LEFT + 1 * DOWN + IN

        label_config = dict(
            font_size=30,
            label_rotation=0,
        )
        man, woman, king, queen = word_vects = [
            self.get_labeled_vector(
                word,
                coords=coords,
                color=color,
                buff=0.05,
                direction=direction,
                label_config=label_config,
            )
            for word, color, direction, coords in zip(words, colors, directions, all_coords)
        ]
        woman.label.shift(SMALL_BUFF * DOWN)

        fake_queen_coords = all_coords[2] - all_coords[0] + all_coords[1]  # Tweak queen for demo purposes
        fake_queen = self.get_labeled_vector(
            "queen", fake_queen_coords,
            color=colors[3],
            label_config=label_config,
        )
        fake_queen.label.shift(0.1 * LEFT + 0.2 * DOWN)

        # Equation
        equation = self.get_equation1("queen", "king", "woman", "man")
        equation.set_x(0)
        eq, minus1, ek, approx, ew, minus2, em = equation
        top_rect = FullScreenFadeRectangle().set_fill(BLACK, 1)
        top_rect.set_height(1.5, about_edge=UP, stretch=True)
        top_rect.fix_in_frame()

        for part, vect in zip([em, ew, ek, eq], word_vects):
            part.set_fill(vect.get_color())

        # Show man and woman vectors
        diff = Arrow(man.get_end(), woman.get_end(), buff=0, stroke_color=YELLOW)
        diff.set_fill(YELLOW, opacity=0.8)
        diff.set_backstroke(BLACK, 3)
        self.play(
            LaggedStart(*map(Write, [ew, minus2, em])),
            GrowArrow(woman),
            FadeInFromPoint(woman.label, man.get_center()),
            GrowArrow(man),
            FadeInFromPoint(man.label, man.get_center()),
            frame.animate.reorient(0, 0, 0, (2.04, 2.06, 0.38), 4.76).set_anim_args(run_time=6)
        )
        self.play(
            GrowArrow(diff, time_span=(1, 3)),
            frame.animate.reorient(-179, 19, 179, (2.49, 1.96, 0.4), 4.76),
            run_time=5
        )

        # Show king and fake queen
        self.add(top_rect, *equation)
        new_diff = diff.copy()
        new_diff.shift(king.get_end() - man.get_end())

        self.play(
            FadeIn(top_rect),
            *map(Write, [eq, minus1, ek, approx]),
            LaggedStart(
                TransformFromCopy(man, king),
                TransformFromCopy(man.label, king.label),
                TransformFromCopy(woman, fake_queen),
                TransformFromCopy(woman.label, fake_queen.label),
            ),
            TransformFromCopy(diff, new_diff, time_span=(2, 3)),
            frame.animate.reorient(0, 2, 0, (0.04, 1.96, -0.13), 5.51).set_anim_args(run_time=3)
        )
        self.play(
            frame.animate.reorient(-110, 10, 110, (0.22, 1.6, -0.07), 6.72),
            run_time=10
        )

        # Rearrange the equation
        for mob in [ek, approx]:
            mob.target = mob.generate_target()
        approx.target.move_to(minus1, LEFT)
        ek.target.next_to(approx.target, RIGHT)
        minus1.target = Tex("+").next_to(ek.target, RIGHT, SMALL_BUFF)
        minus1.target.move_to(midpoint(ek.target.get_right(), ew.get_left()))
        minus1.target.fix_in_frame()

        self.play(
            FadeOut(fake_queen),
            FadeOut(fake_queen.label),
            FadeOut(new_diff),
        )
        self.play(
            LaggedStartMap(MoveToTarget, [minus1, ek, approx], path_arc=PI / 2)
        )
        self.play(FlashAround(VGroup(ek, em), run_time=3, time_width=1.5))
        self.play(TransformFromCopy(diff, new_diff))

        # Search near tip
        n_circs = 5
        src_circles = Circle(radius=1e-2).set_stroke(width=5, opacity=1).replicate(n_circs)
        trg_circles = Circle(radius=1).set_stroke(width=0, opacity=1).replicate(n_circs)
        circs = VGroup(src_circles, trg_circles)
        circs.set_stroke(WHITE)
        circs.move_to(new_diff.get_end())
        self.play(
            LaggedStart(*(
                Transform(src, trg)
                for src, trg in zip(src_circles, trg_circles)
            ), lag_ratio=0.15, run_time=3)
        )
        self.play(
            FadeIn(fake_queen),
            FadeIn(fake_queen.label),
        )
        self.wait()

        # Correct it
        self.play(
            TransformFromCopy(fake_queen, queen),
            TransformFromCopy(fake_queen.label, queen.label),
            VGroup(fake_queen, fake_queen.label).animate.set_opacity(0.2),
        )
        self.play(
            FadeOut(fake_queen),
            FadeOut(fake_queen.label),
            frame.animate.reorient(103, 9, -101, (0.01, 1.45, 0.07), 6.72),
            run_time=10
        )

        # Show a few other examples
        word_pairs = [
            ("uncle", "aunt"),
            ("brother", "sister"),
            ("nephew", "niece"),
            ("father", "mother"),
            ("son", "daughter"),
        ]
        turn_animation_into_updater(
            ApplyMethod(frame.reorient, -116, 21, 114, (0.37, 1.45, 0.23), 7.59, run_time=12)
        )

        last_group = VGroup(king, queen, king.label, queen.label, new_diff)
        last_equation = equation
        for word1, word2 in word_pairs:
            new_coords = np.array([self.basis @ self.model[w] for w in [word1, word2]])
            adj_point = np.array([
                np.random.uniform(-5, 0),
                np.random.uniform(2, 4),
                np.random.uniform(-3, 3),
            ])
            new_coords += (adj_point - new_coords[0])
            vect1 = self.get_labeled_vector(word1, color=colors[2], label_config=label_config, coords=new_coords[0])
            vect2 = self.get_labeled_vector(word2, color=colors[3], label_config=label_config, coords=new_coords[1])
            vect2.put_start_and_end_on(ORIGIN, vect1.get_end() + diff.get_vector() + np.random.uniform(-0.1, 0.1, 3))
            vect2.label.next_to(vect2.get_end(), LEFT)

            new_equation = self.get_equation1(word2, word1, "woman", "man")
            new_equation.move_to(equation, RIGHT)
            new_equation.match_style(equation)
            new_equation.set_fill(opacity=1)
            new_equation.fix_in_frame()

            diff_copy = diff.copy()
            diff_copy.shift(vect1.get_end() - diff_copy.get_start())

            self.play(
                LaggedStart(
                    FadeOut(last_group),
                    GrowArrow(vect1),
                    FadeIn(vect1.label),
                    GrowArrow(vect2),
                    FadeIn(vect2.label),
                ),
                *(
                    FadeTransform(sm1, sm2)
                    for sm1, sm2 in zip(last_equation, new_equation)
                ),
            )
            self.play(TransformFromCopy(diff, diff_copy))
            self.wait(2)

            last_equation = new_equation
            last_group = VGroup(vect1, vect2, vect1.label, vect2.label, diff_copy)
        self.wait(4)

        # Flash in direction
        vect = diff.get_vector()
        color = YELLOW
        lines = Line(ORIGIN, 2 * normalize(vect)).replicate(200)
        lines.insert_n_curves(20)
        lines.set_stroke(color, 3)
        for line in lines:
            line.move_to(np.random.uniform(-3, 3, 3))
        self.play(
            LaggedStartMap(
                VShowPassingFlash, lines,
                lag_ratio=1 / len(lines),
                run_time=4
            )
        )

    def get_labeled_vector(self, *args, **kwargs):
        kwargs.update(func_name = None)
        kwargs.update(thickness=3)
        return super().get_labeled_vector(*args, **kwargs)

    def get_equation1(self, word1, word2, word3, word4, colors=None):
        equation = TexText(
            # Rf"E({word1}) - E({word2}) $\approx$ E({word3}) - E({word4})",
            Rf"{{{word1}}} - {{{word2}}} $\approx$ {{{word3}}} - {{{word4}}}",
            font_size=48
        )
        equation.fix_in_frame(True)
        equation.to_corner(UR)
        if colors:
            words = [word1, word2, word3, word4]
            for word, color in zip(words, colors):
                equation[word].set_fill(color)
        pieces = VGroup(
            equation[f"{{{word1}}}"][0],
            equation["-"][0],
            equation[f"{{{word2}}}"][0],
            equation[R"$\approx$"][0],
            equation[f"{{{word3}}}"][0],
            equation["-"][1],
            equation[f"{{{word4}}}"][0],
        )
        pieces.fix_in_frame(True)
        return pieces

    def get_equation2(self, word1, word2, word3, word4, colors=None):
        equation = TexText(
            # Rf"E({word1}) + E({word2}) - E({word3}) $\approx$ E({word4})",
            Rf"{{{word1}}} + {{{word2}}} - {{{word3}}} $\approx$ {{{word4}}}",
            font_size=48
        )
        equation.fix_in_frame(True)
        equation.to_corner(UR)
        if colors:
            words = [word1, word2, word3, word4]
            for word, color in zip(words, colors):
                equation[word].set_fill(color)
        pieces = VGroup(
            equation[f"{{{word1}}}"],
            equation["+"],
            equation[f"{{{word2}}}"],
            equation["-"],
            equation[f"{{{word3}}}"],
            equation[R"$\approx$ "],
            equation[f"{{{word4}}}"],
        )
        pieces.fix_in_frame(True)
        return pieces


class HitlerMussoliniExample(KingQueenExample):
    words = ["Hitler", "Italy", "Germany", "Mussolini"]
    colors = [GREY_C, "#008C45", "#FFCC00", GREY_B]
    default_frame_orientation = (-17, 75, 0)
    second_frame_orientation = (-24, 66, 0)
    interpolation_factor = 0.2
    diff_color = RED_B

    def get_basis(self, model):
        v1, v2, v3, v4 = [model[word.lower()] for word in self.words]
        b1 = normalize(v2 - v3)
        b2 = normalize(v1 - v3)
        b3 = normalize(get_principle_components(model.vectors)[:, 0])
        return np.array([b1, b2, b3])

    def construct(self):
        # Set up
        frame = self.frame
        frame.move_to(1.0 * UP)
        frame.add_updater(lambda f, dt: f.increment_theta(dt * 1 * DEGREES))
        axes = self.axes

        # Add equation
        equation = self.get_equation2(*self.words, colors=self.colors)
        equation.center().to_edge(UP)
        self.add(equation[:-1])

        # Initialize vectors
        v1, v2, v3, v4 = vects = [
            self.get_labeled_vector(word, color=color)
            for word, color in zip(self.words, self.colors)
        ]
        fudged_v4 = self.get_labeled_vector(
            self.words[3],
            axes.p2c(interpolate(
                v1.get_end() + v2.get_end() - v3.get_end(),
                v4.get_end(),
                self.interpolation_factor,
            )),
            color=self.colors[3]
        )
        vects[3] = fudged_v4
        for vect in vects:
            vect.apply_depth_test()

        # Show (v3 - v2) difference
        diff = Arrow(
            v3.get_end(), v2.get_end(),
            buff=0,
            stroke_color=self.diff_color,
            stroke_width=2,
            flat_stroke=False,
        )
        diff.apply_depth_test()
        rect = SurroundingRectangle(equation[2:5])
        rect.set_stroke(diff.get_stroke_color(), 2)
        self.play(
            GrowArrow(v2),
            GrowArrow(v3),
            FadeIn(v2.label),
            FadeIn(v3.label),
        )
        self.play(
            ShowCreation(rect),
            Transform(v3.copy(), v2, remover=True),
            ShowCreation(diff)
        )
        self.wait(2)

        # Add to v1
        diff_copy = diff.copy()
        diff_copy.shift(v1.get_end() - diff.get_start())

        self.play(
            GrowArrow(v1),
            FadeIn(v1.label),
            frame.animate.reorient(*self.second_frame_orientation),
        )
        self.play(
            TransformFromCopy(diff, diff_copy),
            rect.animate.surround(equation[:5])
        )
        self.wait(2)
        self.play(
            rect.animate.surround(equation[-1]),
            FadeIn(equation[-1]),
            GrowArrow(fudged_v4),
            FadeIn(fudged_v4.label),
        )
        self.play(FadeOut(rect))
        self.wait(6)

        # Emphasize directions
        italy_vect = diff.get_vector()
        axis_vect = v1.get_end() - v3.get_end()
        for vect, color in [(italy_vect, RED), (axis_vect, GREY)]:
            lines = Line(ORIGIN, 2 * normalize(vect)).replicate(200)
            lines.insert_n_curves(20)
            lines.set_stroke(color, 3)
            for line in lines:
                line.move_to(np.random.uniform(-3, 3, 3))
            self.play(
                LaggedStartMap(
                    VShowPassingFlash, lines,
                    lag_ratio=1 / len(lines),
                    run_time=4
                )
            )


class SushiBratwurstExample(HitlerMussoliniExample):
    words = ["Sushi", "Germany", "Japan", "Bratwurst"]
    colors = [WHITE, "#FFCC00", "#BC002D", interpolate_color(GREY_BROWN, WHITE, 0.25)]
    interpolation_factor = -0.1
    default_frame_orientation = (-17, 80, 0)
    second_frame_orientation = (-24, 75, 0)
    diff_color = GREY_B

    def get_basis(self, model):
        basis = super().get_basis(model)
        basis = basis[[1, 2, 0]]
        basis[1] /= -2
        basis[2] /= 3
        return basis


class SizeDirection(Word2VecScene):
    def construct(self):
        # To illustrate "You could imagine many other directions in this space corresponding to semantic meaning"

        # Set up axes
        axes = self.axes
        frame = self.frame
        self.basis *= 1.5

        # Add vectors
        frame.reorient(35, 80, 0)
        colors = [BLUE_B, BLUE_C, BLUE_D]
        word_lists = [
            ["micrometer", "millimeter", "meter"],
            ["microgram", "milligram", "gram"],
            ["microliter", "milliliter", "liter"],
        ]
        vect_groups = VGroup(
            VGroup(
                self.get_labeled_vector(word, color=color, func_name=None)
                for word, color in zip(word_list, colors)
            )
            for word_list in word_lists
        )

        over_arrow = Arrow(2 * LEFT, 2 * RIGHT).shift(UP)
        over_arrow.set_stroke(YELLOW, width=10)
        over_words = Text("Size", font_size=72)
        over_words.set_color(YELLOW)
        over_words.set_backstroke(BLACK, 5)
        over_words.next_to(over_arrow, UP)
        annotation = VGroup(over_arrow, over_words)
        annotation.shift(LEFT)
        annotation.fix_in_frame()

        for vect_group in vect_groups:
            vect_group.labels = VGroup()
            for vect in vect_group:
                vect.label.rotate(45 * DEGREES, OUT)
                vect.label.next_to(vect.get_end(), normalize(vect.get_vector()), SMALL_BUFF)
                vect_group.labels.add(vect.label)

        self.play(
            frame.animate.reorient(49, 87, 0),
            LaggedStartMap(FadeIn, vect_groups[0], lag_ratio=0.25),
            LaggedStartMap(FadeIn, vect_groups[0].labels, lag_ratio=0.25),
            FadeIn(annotation, lag_ratio=0.1, time_span=(2, 3)),
            run_time=3
        )
        self.wait()
        for i in [0, 1]:
            self.play(
                ReplacementTransform(vect_groups[i], vect_groups[i + 1]),
                ReplacementTransform(vect_groups[i].labels, vect_groups[i + 1].labels),
            )
            self.wait()


class PluralityDirection(Word2VecScene):
    def construct(self):
        self.add_plane()
        self.axes.x_axis.set_stroke(opacity=0)
        self.axes.y_axis.set_stroke(opacity=0)

        # Test
        self.frame.reorient(-21, 77, 0, (1.97, -0.73, 0.54), 3.67)
        self.frame.add_updater(lambda m, dt: m.increment_theta(dt * DEGREES))
        words = ["cat", "cats"]
        all_coords = 2 * np.array([self.basis @ self.model[word] for word in words])
        colors = [BLUE, RED]
        cat, cats = [
            self.get_labeled_vector(
                word,
                coords=coords,
                color=color,
                buff=0.05,
            )
            for word, color, coords in zip(words, colors, all_coords)
        ]
        diff = Arrow(cat.get_end(), cats.get_end(), buff=0)
        diff.set_color(YELLOW)

        self.add(cat, cats)
        self.add(cat.label, cats.label)

        self.wait(5)
        self.play(ShowCreation(diff))
        self.wait(10)


class ShowNearestNeighbors(Word2VecScene):
    seed_word = "tower"
    color = YELLOW
    n_shown = 10
    frame_height = 4
    frame_center = (2.18, 0.09, 0.72)
    frame_orientation = (-21, 87, 0)
    wait_time_per_example = 0.5

    def construct(self):
        # Setup
        frame = self.frame
        frame.reorient(*self.frame_orientation, self.frame_center, self.frame_height)
        frame.add_updater(lambda f, dt: f.increment_theta(dt * DEGREES))
        self.add_plane()

        # Add seed
        word = self.seed_word.lower()
        seed_vect = self.get_labeled_vector(word, color=self.color)
        seed_group = VGroup(seed_vect, seed_vect.label)
        self.add(seed_group)

        # Add neighbors
        nearest_words = self.get_nearest_words(word)
        neighbors = VGroup(*(
            self.get_labeled_vector(
                word,
                # coords=seed_vect.get_end() + np.random.uniform(-0.5, 0.5, 3),
                color=WHITE
            )
            for word in nearest_words
        ))
        for neighbor in neighbors:
            neighbor.label.scale(0.75, about_edge=LEFT)
            neighbor.label.set_fill(border_width=0)
            neighbor.add(neighbor.label)

        # Description
        title = Text(f"Embeddings closest to E({self.seed_word})")
        underline = Underline(title)
        items = VGroup(*(
            Text(f"E({word})", font_size=36)
            for word in nearest_words
        ))
        items.arrange(DOWN, aligned_edge=LEFT)
        items.next_to(underline, DOWN, buff=0.5)
        items.align_to(title["E"][-1], LEFT)
        items.set_backstroke(BLACK, 8)

        desc = VGroup(title, underline, items)
        desc.fix_in_frame()
        desc.to_corner(UR)

        self.add(title, underline)

        # Add them all
        last_neighbor = VectorizedPoint()
        for item, neighbor in zip(items, neighbors):
            faded_last_neighbor = last_neighbor.copy()
            faded_last_neighbor.set_opacity(0.2)
            self.add(faded_last_neighbor, seed_group, neighbor)
            self.play(
                FadeIn(item),
                FadeIn(neighbor),
                FadeOut(last_neighbor),
                FadeIn(faded_last_neighbor),
            )
            last_neighbor = neighbor
            self.wait(self.wait_time_per_example)
        self.play(last_neighbor.animate.set_opacity(0.2))

        self.wait(10)

    def get_nearest_words(self, word):
        return find_nearest_words(self.model, self.model[word], self.n_shown + 1)[1:]

    def animate_in_neighbors(self, neighbors):
        # Old
        to_fade = VGroup()
        for neighbor in neighbors:
            neighbor.label.set_fill(border_width=0)
            self.add(to_fade, neighbor.label, seed_vect, seed_vect.label)
            self.play(
                FadeIn(neighbor),
                FadeIn(neighbor.label),
                to_fade.animate.set_opacity(0.25),
            )
            to_fade = VGroup(neighbor, neighbor.label)
        self.add(to_fade, neighbor.label, seed_vect, seed_vect.label)
        self.play(to_fade.animate.set_opacity(0.2))
        self.wait(5)


class ShowNearestNeighborsToWikipedia(ShowNearestNeighbors):
    seed_word = "wikipedia"
    color = BLUE
    default_frame_orientation = (10, 70)


class ShowNearestNeighborsToCat(ShowNearestNeighbors):
    seed_word = "cat"
    color = YELLOW


class ShowNearestNeighborsToNavy(ShowNearestNeighbors):
    seed_word = "navy"
    color = RED


class ShowNearestNeighborsToJump(ShowNearestNeighbors):
    seed_word = "jump"
    color = BLUE
    wait_time_per_example = 1.0
    frame_center = (2.18, -2.0, 0.0)
    random_seed = 1

    def add_plane(self):
        return VGroup()

    def get_nearest_words(self, word):
        return ["hop", "skip", "leap", "bound", "bounce", "drop", "vault"]


class DotProducts(InteractiveScene):
    def construct(self):
        # Add vectors
        plane = NumberPlane(
            (-4, 4), (-4, 4),
            background_line_style=dict(
                stroke_width=2,
                stroke_opacity=0.5,
                stroke_color=BLUE,
            ),
            faded_line_ratio=1
        )
        plane.set_height(6)
        plane.to_edge(LEFT, buff=0)
        vects = VGroup(
            Vector(0.5 * RIGHT + 2 * UP).set_stroke(MAROON_B, 6),
            Vector(1.0 * RIGHT + 0.5 * UP).set_stroke(YELLOW, 6),
        )
        vects.shift(plane.get_center())

        def get_dot_product():
            coords = np.array([plane.p2c(v.get_end()) for v in vects])
            return np.dot(coords[0], coords[1])

        self.add(plane)
        self.add(vects)

        # Vector labels
        vect_labels = VGroup(*(
            Tex(Rf"\vec{{\textbf{{ {char} }} }}")
            for char in "vw"
        ))
        for label, vect in zip(vect_labels, vects):
            label.vect = vect
            label.match_color(vect)
            label.add_updater(lambda l: l.move_to(
                l.vect.get_end() + 0.25 * normalize(l.vect.get_vector())
            ))

        self.add(vect_labels)

        # Add coordinate expressions
        vect_coords = VGroup(*(
            TexMatrix(
                [
                    [char + f"_{{{str(n)}}}"]
                    for n in [1, 2, 3, 4, "n"]
                ],
                bracket_h_buff=0.1,
                ellipses_row=-2,
            )
            for char in "vw"
        ))
        vect_coords.arrange(RIGHT, buff=0.75)
        vect_coords.next_to(plane, RIGHT, buff=1)
        vect_coords.set_y(1)
        for coords, vect in zip(vect_coords, vects):
            coords.get_entries().match_color(vect)
        dot = Tex(R"\cdot", font_size=72)
        dot.move_to(vect_coords)

        self.add(vect_coords, dot)

        # Add right hand side
        rhs = Tex("= +0.00", font_size=60)
        rhs.next_to(vect_coords, RIGHT)
        result = rhs.make_number_changeable("+0.00", include_sign=True)
        result.add_updater(lambda m: m.set_value(get_dot_product()))

        self.add(rhs)

        # Add dot product label
        brace = Brace(vect_coords, DOWN, buff=0.25)
        dp_label = brace.get_text("Dot product", buff=0.25)

        self.add(brace, dp_label)

        # Play around
        def dual_rotate(angle1, angle2, run_time=2):
            self.play(
                Rotate(vects[0], angle1 * DEGREES, about_point=plane.get_origin()),
                Rotate(vects[1], angle2 * DEGREES, about_point=plane.get_origin()),
                run_time=run_time
            )

        dual_rotate(-20, 20)
        dual_rotate(50, -60)
        dual_rotate(0, 80)
        dual_rotate(20, -80)

        # Show computation
        equals = rhs[0].copy()
        entry_pairs = VGroup(*(
            VGroup(*pair)
            for pair in zip(*[vc.get_columns()[0] for vc in vect_coords])
        ))
        prod_terms = entry_pairs.copy()
        for src_pair, trg_pair in zip(entry_pairs, prod_terms):
            trg_pair.arrange(RIGHT, buff=0.1)
            trg_pair.next_to(equals, RIGHT, buff=0.5)
            trg_pair.match_y(src_pair)
        prod_terms[-2].space_out_submobjects(1e-3)
        prod_terms[-2].match_x(prod_terms)
        prod_terms.target = prod_terms.generate_target()
        prod_terms.target.space_out_submobjects(1.5).match_y(vect_coords)
        plusses = VGroup(*(
            Tex("+", font_size=48).move_to(midpoint(m1.get_bottom(), m2.get_top()))
            for m1, m2 in zip(prod_terms.target, prod_terms.target[1:])
        ))

        rhs.target = rhs.generate_target()
        rhs.target[0].rotate(PI / 2)
        rhs.target.arrange(DOWN)
        rhs.target.next_to(prod_terms, DOWN)

        self.add(equals)
        self.play(
            LaggedStart(*(
                TransformFromCopy(m1, m2)
                for m1, m2 in zip(entry_pairs, prod_terms)
            ), lag_ratio=0.1, run_time=2),
            MoveToTarget(rhs)
        )
        self.wait()
        self.play(
            MoveToTarget(prod_terms),
            rhs.animate.next_to(prod_terms.target, DOWN),
            LaggedStartMap(Write, plusses),
        )
        self.wait()

        # Positive value
        dual_rotate(-65, 65)
        self.play(FlashAround(result, time_width=1.5, run_time=3))
        self.wait()

        # Orthogonal
        elbow = Elbow(width=0.25, angle=vects[0].get_angle())
        elbow.shift(plane.get_origin())
        zero = DecimalNumber(0)
        zero.replace(result, 1)
        dual_rotate(
            (vects[1].get_angle() + PI / 2 - vects[0].get_angle()) / DEGREES,
            0,
        )
        self.remove(result)
        self.add(zero)
        self.play(ShowCreation(elbow))
        self.wait()
        self.remove(elbow, zero)
        self.add(result)

        # Negative
        dual_rotate(20, -60)
        self.play(FlashAround(result, time_width=1.5, run_time=3))
        self.wait()

        # Play again
        dual_rotate(75, -95, run_time=8)


class DotProductWithPluralDirection(InteractiveScene):
    vec_tex = R"\vec{\text{plur}}"
    ref_words = ["cat", "cats"]
    word_groups = [
        ["puppy", "puppies"],
        ["octopus", "octopi", "octopuses", "octopodes"],
        ["student", "students"],
        ["one", "two", "three", "four"],
    ]
    x_range = (-4, 4 + 1e-4, 0.25)
    colors = [BLUE, RED]
    threshold = -1.0

    def construct(self):
        # Initialize equation
        self.model = get_word_to_vec_model()
        word_groups = self.word_groups
        words = list(it.chain(*word_groups))

        # Write plurality equation
        gen_lhs = self.get_equation_lhs(words[0])[0].copy()
        equals = Tex(":=")
        rf1, rf2 = self.ref_words
        rhs = Tex(
            Rf"E(\text{{{rf2}}}) - E(\text{{{rf1}}})",
            tex_to_color_map={
                Rf"\text{{{ref_word}}}": color
                for ref_word, color in zip(self.ref_words, self.colors)
            }
        )
        top_eq = VGroup(gen_lhs, equals, rhs)
        top_eq.arrange(RIGHT)
        gen_lhs.align_to(rhs, DOWN)
        top_eq.center().to_edge(UP, buff=0.5)

        self.play(FadeIn(rhs, UP))
        self.play(LaggedStart(
            FadeIn(equals, 0.5 * LEFT),
            FadeIn(gen_lhs, 1.0 * LEFT),
        ))
        self.wait()

        # Show on number line
        x_range = self.x_range
        number_line = NumberLine(
            x_range,
            big_tick_numbers=list(np.arange(*x_range[:2])),
            tick_size=0.05,
            longer_tick_multiple=3.0,
            width=12
        )
        number_line.rotate(PI / 2)
        number_line.add_numbers(
            np.arange(*x_range[:2]),
            num_decimal_places=1,
            font_size=40,
            direction=LEFT,
        )
        number_line.numbers.shift(SMALL_BUFF * LEFT)
        number_line.set_max_height(FRAME_HEIGHT - 1)
        number_line.to_edge(LEFT, buff=1.0)

        eq_lhs = self.get_equation_lhs(words[0])
        eq_rhs = self.get_equation_rhs(eq_lhs, words[0])
        equation = VGroup(eq_lhs, eq_rhs)
        brace = Brace(eq_lhs[2], LEFT, buff=0.1)
        brace.next_to(equation, LEFT, SMALL_BUFF, DOWN)
        equation_group = VGroup(brace, equation)
        dp = eq_rhs.get_value()

        word = eq_lhs[2][2:-1]
        lil_word = word.copy().scale(0.25)
        dot = GlowDot(color=word[0].get_color())
        dot.move_to(number_line.n2p(dp))

        lil_word.next_to(dot, RIGHT, buff=0)
        equation_group.next_to(dot, RIGHT, buff=0, submobject_to_align=brace)

        self.play(
            top_eq.animate.scale(0.75).to_corner(UR),
            TransformFromCopy(gen_lhs, eq_lhs[0]),
            FadeIn(eq_lhs[1:], shift=DOWN),
            FadeIn(brace, shift=DOWN),
            UpdateFromAlphaFunc(
                eq_rhs,
                lambda m, a: m.set_value(a * dp).next_to(eq_lhs[-1], RIGHT),
                run_time=1,
            ),
            Write(number_line, run_time=1)
        )
        self.add_dot(word, dot)
        self.wait()

        # Show some alternate
        new_rhs = eq_rhs.copy()
        eq_rhs.set_opacity(0)
        new_rhs.f_always.set_value(lambda: number_line.p2n(brace.get_center()))
        new_rhs.always.next_to(eq_lhs[-1], RIGHT)
        self.add(new_rhs)

        to_fade = Group(dot)
        for word_group in self.word_groups:
            for new_word in word_group:
                new_dp = self.get_dot_with_key_word(new_word)
                nl_point = number_line.n2p(new_dp)
                color = self.colors[int(new_dp > self.threshold)]
                new_dot = GlowDot(number_line.n2p(new_dp), color=color)
                new_lhs = self.get_equation_lhs(new_word)
                new_rhs = self.get_equation_rhs(new_lhs, new_word)
                new_rhs.set_opacity(0)
                new_equation = VGroup(new_lhs, new_rhs)
                new_equation.move_to(equation, LEFT)
                new_brace = brace.copy()
                new_equation_group = VGroup(new_brace, new_equation)
                y_shift = new_dot.get_y() - brace.get_y()
                new_equation_group.shift(y_shift * UP)

                if new_word == word_group[0]:
                    added_anim = FadeOut(to_fade)
                    to_fade = Group()
                else:
                    ghost = equation_group.copy()
                    ghost.target = ghost.generate_target()
                    ghost.target.set_fill(opacity=0.75)
                    ghost.target.scale(0.5, about_point=ghost[0].get_left())
                    added_anim = MoveToTarget(ghost)
                    to_fade.add(ghost)
                self.play(
                    Transform(equation_group, new_equation_group),
                    added_anim,
                )
                self.add_dot(new_lhs[2][2:-1], new_dot)
                to_fade.add(new_dot)

    def add_dot(self, word, dot):
        self.play(
            FadeInFromPoint(dot, word.get_center()),
            LaggedStart(
                (FadeTransform(char.copy(), dot.copy().set_opacity(0))
                for char in word),
                lag_ratio=2e-2,
                group_type=Group
            ),
            run_time=1
        )

    def get_equation_lhs(self, word):
        tex_pieces = [
            self.vec_tex, R"\cdot", Rf"E(\text{{{word}}})", "="
        ]
        expression = Tex(
            " ".join(tex_pieces),
            tex_to_color_map={self.vec_tex: YELLOW}
        )
        parts = [
            expression[tex_piece][0]
            for tex_piece in tex_pieces
        ]
        gen_part = parts[0]
        gen_part[0].set_width(0.75 * gen_part.get_width(), about_edge=DOWN)
        gen_part[0].shift(SMALL_BUFF * DOWN)
        value = self.get_dot_with_key_word(word)
        parts[2][2:-1].set_color(self.colors[int(value > self.threshold)])
        return VGroup(*parts)

    def get_equation_rhs(self, equation_lhs, word):
        rhs = DecimalNumber(self.get_dot_with_key_word(word))
        rhs.next_to(equation_lhs[-1], RIGHT)
        return rhs

    def get_dot_with_key_word(self, word):
        if word == "octopodes":
            return 2.3  # Hack
        elif word == "four":
            return 1.80  # To make the spacing nicer
        rf1, rf2 = self.ref_words
        return np.dot(
            (self.model[rf2] - self.model[rf1]).flatten(),
            self.model[word].flatten(),
        )


class DotProductWithGenderDirection(DotProductWithPluralDirection):
    vec_tex = R"\vec{\text{gen}}"
    ref_words = ["man", "woman"]
    words = [
        "mother", "father",
        "aunt", "uncle",
        "sister", "brother",
        "mama", "papa",
    ]
    x_range = (-5, 7 + 1e-4, 0.25)
    colors = [BLUE, RED]
    threshold = 1.0


class RicherEmbedding(InteractiveScene):
    def construct(self):
        # Add phrase
        phrase = Text("The King doth wake tonight and takes his rouse ...")
        phrase.to_edge(UP)
        words = break_into_words(phrase)
        rects = get_piece_rectangles(words)
        king_index = 1

        words.fix_in_frame()
        rects.fix_in_frame()

        self.add(words)

        # Setup axes
        self.set_floor_plane("xz")
        frame = self.frame
        frame.reorient(9, -6, 0)
        axes = ThreeDAxes((-5, 5), (-2, 2), (-5, 5))
        axes.shift(DOWN)
        plane = NumberPlane(
            (-5, 5), (-5, 5),
            background_line_style=dict(stroke_width=1, stroke_color=BLUE_E),
            faded_line_ratio=1,
        )
        plane.axes.set_stroke(GREY)
        plane.set_flat_stroke(False)
        plane.rotate(PI / 2, RIGHT)
        plane.move_to(axes)

        self.add(axes)
        self.add(plane)

        # Embed the word
        king_rect = rects[king_index]
        vector = Vector([-1, 1, 1])
        vector.shift(axes.get_origin())
        vector.match_color(king_rect)
        vector.set_flat_stroke(False)
        label = Text("King", font_size=24)
        label.next_to(vector.get_end(), normalize(vector.get_vector()), buff=0.1)

        self.play(DrawBorderThenFill(king_rect))
        self.play(
            TransformFromCopy(words[king_index], label),
            GrowArrow(vector),
        )
        self.wait(3)

        # Mention position
        index_labels = VGroup(*(
            Integer(n + 1, font_size=36).next_to(rect, DOWN, buff=0.2)
            for n, rect in enumerate(rects)
        ))
        index_labels.fix_in_frame()
        idx_vect, idx_label = self.get_added_vector(
            vector.get_end(), 0.5 * (RIGHT + OUT), "Pos. 2", TEAL,
            next_to_direction=UP,
            font_size=16
        )
        idx_label.rotate(45 * DEGREES, DOWN)
        idx_label.set_backstroke(BLACK, 1)

        self.play(
            LaggedStartMap(FadeIn, index_labels, shift=0.5 * DOWN),
        )
        self.play(
            TransformFromCopy(index_labels[king_index].set_backstroke(), idx_label),
            GrowArrow(idx_vect),
            frame.animate.reorient(-28, -22, 0).set_anim_args(run_time=3)
        )
        self.play(
            frame.animate.reorient(-11, -4, 0),
            LaggedStartMap(FadeOut, index_labels, lag_ratio=0.05, shift=0.5 * DOWN, time_span=(6, 7)),
            run_time=7
        )

        # Show king ingesting context
        self.play(
            LaggedStart(*(
                ContextAnimation(
                    words[king_index],
                    [*words[:king_index], *words[king_index + 1:]],
                    direction=DOWN,
                    fix_in_frame=True,
                    time_width=3,
                    min_stroke_width=3,
                    lag_ratio=0.05,
                    path_arc=PI / 3,
                )
                for n in range(3)
            ), lag_ratio=0.5),
            frame.animate.reorient(-5, -12, 0),
            run_time=5,
        )

        # Knock in many directions
        new_labeled_vector_args = [
            ([2, 1, 0], "lived in Scotland", None, DR),
            ([0, -1, -1], "murdered predecessor", None, RIGHT),
            ([-1.5, 1, -2], "in Shakespearean language", None, RIGHT),
        ]
        new_labeled_vects = VGroup()
        last_vect = idx_vect
        for args in new_labeled_vector_args:
            new_labeled_vects.add(self.get_added_vector(
                last_vect.get_end(), *args
            ))
            last_vect = new_labeled_vects[-1][0]
            last_vect.apply_depth_test()


        (vect1, label1), (vect2, label2), (vect3, label3) = new_labeled_vects
        self.play(
            GrowArrow(vect1),
            FadeIn(label1, 0.5 * DOWN),
            frame.animate.reorient(2, -16, 0, (0.6, -0.04, 0.02), 6.01).set_anim_args(run_time=8),
        )
        self.play(
            GrowArrow(vect2),
            FadeIn(label2, 0.5 * DOWN),
            frame.animate.reorient(35, -23, 0, (0.6, -0.04, 0.02), 6.01).set_anim_args(run_time=5),
        )
        self.play(
            GrowArrow(vect3),
            FadeIn(label3, 0.5 * DOWN),
            frame.animate.reorient(20, -29, 0, (0.61, 0.01, 0.0), 6.10).set_anim_args(run_time=5),
        )
        self.play(
            frame.animate.reorient(-19, -25, 0, (0.61, 0.01, 0.0), 6.10),
            run_time=5
        )

    def get_added_vector(self, curr_tip, direction, label, color=None, next_to_direction=UP, buff=0.1, font_size=24):
        if color is None:
            color = random_bright_color(hue_range=(0.45, 0.65))
        vect = Vector(direction)
        vect.set_color(color)
        vect.set_flat_stroke(False)
        vect.shift(curr_tip)
        text = Text(label, font_size=font_size)
        text.set_backstroke(BLACK, 4)
        text.next_to(vect.get_center(), next_to_direction, buff=buff)
        text.set_fill(border_width=0)

        result = VGroup(vect, text)
        return result


# For chapter 6

class MultipleMoleEmbeddings(Word2VecScene):
    default_frame_orientation = (0, 0)
    label_rotation = 0

    def setup(self):
        super().setup()
        self.set_floor_plane("xz")
        self.frame.add_ambient_rotation()
        self.add_plane()
        for mob in [self.plane, self.axes]:
            mob.rotate(-90 * DEGREES, RIGHT)

    def construct(self):
        # Show generic mole embedding
        frame = self.frame
        frame.reorient(-6, -6, 0, (-0.73, 1.29, -0.57), 5.27)
        phrases = VGroup(map(Text, [
            "American shrew mole",
            "One mole of carbon dioxide",
            "Take a biopsy of the mole",
        ]))
        for phrase in phrases:
            phrases.fix_in_frame()
            phrases.to_corner(UL)
            phrase["mole"][0].set_color(YELLOW)

        gen_vector = self.get_labeled_vector("mole", coords=(-2, 1.0, 1.5))
        curr_phrase = phrases[1]
        mover = curr_phrase["mole"][0]
        mover.set_backstroke(BLACK, 4)

        self.add(curr_phrase)
        self.wait()
        self.play(
            GrowArrow(gen_vector),
            TransformFromCopy(mover, gen_vector.label),
        )
        self.wait(10)

        # Show three refined meanings
        images = Group(
            ImageMobject("ShrewMole"),
            Tex(R"6.02 \times 10^{23}", font_size=24).set_color(BLUE),
            ImageMobject("LipMole"),
        )
        for image in images[::2]:
            image.set_height(0.5)
            image.set_opacity(0.75)

        colors = [GREY_BROWN, BLUE, ORANGE]
        ref_vects = VGroup(
            self.get_labeled_vector("", coords=coords)
            for coords in [
                (-1.0, -1.5, 1.5),
                (-4.0, 0.5, 1.0),
                (-0.5, 1.0, 2.5),
            ]
        )
        for vect, image, color in zip(ref_vects, images, colors):
            vect.set_color(color)
            image.next_to(vect.get_end(), UP, SMALL_BUFF)

        gen_vect_group = VGroup(gen_vector, gen_vector.label)

        self.play(
            frame.animate.reorient(-30, -5, 0, (-1.11, 1.35, -0.72), 5.27),
            LaggedStart(
                (TransformFromCopy(gen_vector, ref_vect)
                for ref_vect in ref_vects),
                lag_ratio=0.25,
                run_time=2,
            ),
            LaggedStart(
                (FadeInFromPoint(image, gen_vector.label.get_center())
                for image in images),
                lag_ratio=0.25,
                run_time=2,
                group_type=Group,
            ),
            gen_vect_group.animate.set_opacity(0.25).set_anim_args(run_time=2),
            run_time=2,
        )
        self.wait(3)

        ref_vect_groups = Group(
            Group(*pair) for pair in zip(ref_vects, images)
        )

        # Oscillate between meanings based on context
        diff_vects = VGroup(
            Arrow(gen_vector.get_end(), ref_vect.get_end(), buff=0)
            for ref_vect in ref_vects
        )
        diff_vects.set_color(GREY_B)

        last_phrase = curr_phrase
        last_diff = VGroup()
        for n, diff in enumerate(diff_vects):
            ref_vect_groups.target = ref_vect_groups.generate_target()
            ref_vect_groups.target.set_opacity(0.2)
            ref_vect_groups.target[n].set_opacity(1)
            if n != 2:
                ref_vect_groups.target[2][1].set_opacity(0.1)
            phrase = phrases[n]
            self.play(
                gen_vect_group.animate.set_opacity(1),
                MoveToTarget(ref_vect_groups),
                FadeOut(last_phrase, UP),
                FadeIn(phrase, UP),
                FadeOut(last_diff)
            )
            self.play(
                ShowCreation(diff, time_span=(1, 2)),
                TransformFromCopy(gen_vector, ref_vects[n], time_span=(1, 2)),
                ContextAnimation(
                    phrase["mole"][0], phrase,
                    direction=DOWN,
                    fix_in_frame=True,
                ),
            )
            self.wait(3)

            last_phrase = phrase
            last_diff = diff

        self.wait(5)

    def get_basis(self, model):
        basis = super().get_basis(model) * 2
        basis[2] *= -1
        return basis


class RefineTowerMeaning(MultipleMoleEmbeddings):
    def construct(self):
        # Set up vectors and images
        frame = self.frame
        frame.reorient(-26, -4, 0, (3.27, 1.57, 0.59), 5.28)
        frame.add_ambient_rotation(0.5 * DEGREES)

        words = VGroup(Text(word) for word in "Miniature Eiffel Tower".split(" "))
        words.scale(1.25)
        words.to_edge(UP)
        words.fix_in_frame()

        tower_images = Group(
            ImageMobject(f"Tower{n}")
            for n in range(1, 5)
        )
        eiffel_tower_images = Group(
            ImageMobject(f"EiffelTower{n}")
            for n in range(1, 4)
        )
        mini_eiffel_tower_images = Group(
            ImageMobject("MiniEiffelTower1")
        )
        image_groups = Group(
            tower_images,
            eiffel_tower_images,
            mini_eiffel_tower_images
        )

        vectors = VGroup(
            self.get_labeled_vector("", coords=coords)
            for coords in [
                (4, -1, 3.0),
                (5, -2, 1.5),
                (-3, -1, 2.5),
            ]
        )
        colors = [BLUE_D, GREY_B, GREY_C]
        for vector, color, image_group in zip(vectors, colors, image_groups):
            vector.set_color(color)
            for image in image_group:
                image.set_height(1.5)
                image.next_to(vector.get_end(), RIGHT * np.sign(vector.get_end()[0]))

        # Show tower
        tower = words[-1]
        tower.set_x(0)
        pre_tower_image = tower_images[0].copy()
        pre_tower_image.fix_in_frame()
        pre_tower_image.replace(tower, stretch=True)
        pre_tower_image.set_opacity(0)

        self.add(tower)
        self.wait()
        self.play(
            GrowArrow(vectors[0]),
            ReplacementTransform(pre_tower_image, tower_images[0]),
            run_time=2,
        )
        for ti1, ti2 in zip(tower_images, tower_images[1:]):
            self.play(
                FadeTransform(ti1, ti2),
                run_time=2
            )
        self.wait(2)

        # Eiffel tower
        words[:-1].set_opacity(0)
        eiffel_tower = words[-2:]

        self.play(
            frame.animate.reorient(-4, -7, 0, (2.95, 1.82, 0.49), 6.59),
            eiffel_tower.animate.set_opacity(1).arrange(RIGHT, aligned_edge=DOWN).to_edge(UP),
        )
        self.play(
            vectors[0].animate.set_opacity(0.25),
            tower_images[-1].animate.set_opacity(0.2),
            TransformFromCopy(vectors[0], vectors[1]),
            FadeTransform(tower_images[-1].copy(), eiffel_tower_images[0]),
            ContextAnimation(words[2], words[1], direction=DOWN, fix_in_frame=True),
            run_time=2,
        )
        for ti1, ti2 in zip(eiffel_tower_images, eiffel_tower_images[1:]):
            self.play(
                FadeTransform(ti1, ti2),
                run_time=2
            )
        self.wait(6)

        # Miniature eiffel tower
        self.play(
            frame.animate.reorient(-14, -2, 0, (-0.12, 2.21, 0.72), 7.05).set_anim_args(run_time=2),
            words.animate.set_opacity(1).arrange(RIGHT, aligned_edge=DOWN).to_edge(UP),
        )
        self.play(
            vectors[1].animate.set_opacity(0.25),
            eiffel_tower_images[-1].animate.set_opacity(0.2),
            TransformFromCopy(vectors[1], vectors[2]),
            FadeTransform(eiffel_tower_images[-1].copy(), mini_eiffel_tower_images[0]),
            ContextAnimation(words[2], words[0], direction=DOWN, fix_in_frame=True),
            run_time=2,
        )
        self.wait(10)


class UpdatingPoetryEmbedding(RicherEmbedding):
    def construct(self):
        # (Largely copied from RicherEmbedding, could factor better later)
        # Add phrase
        poem_str = "...\nTwo roads diverged in a wood, and I—\nI took the one less traveled by,"
        phrase = Text(poem_str, alignment="LEFT")
        phrase[:3].rotate(PI / 2).shift(SMALL_BUFF * UP)
        phrase.refresh_bounding_box()
        phrase.to_edge(UP, buff=SMALL_BUFF)
        words = break_into_words(phrase)
        rects = get_piece_rectangles(words)

        words.fix_in_frame()
        rects.fix_in_frame()

        self.add(words)

        # Setup axes
        self.set_floor_plane("xz")
        frame = self.frame
        frame.reorient(9, -6, 0)
        frame.reorient(9, -1, 0, 0.75 * UP)
        axes = ThreeDAxes((-5, 5), (-2, 2), (-5, 5))
        axes.shift(DOWN)
        plane = NumberPlane(
            (-5, 5), (-5, 5),
            background_line_style=dict(stroke_width=1, stroke_color=BLUE_E),
            faded_line_ratio=1,
        )
        plane.axes.set_stroke(GREY)
        plane.set_flat_stroke(False)
        plane.rotate(PI / 2, RIGHT)
        plane.move_to(axes)

        self.add(axes)
        self.add(plane)

        # Embed the word
        one_index = len(words) - 4
        one_rect = SurroundingRectangle(words[one_index])
        one_rect.set_fill(GREEN, 0.2)
        one_rect.set_stroke(GREEN, 2)
        one_rect.fix_in_frame()
        vector = Vector([-3, 1, 2])
        vector.shift(axes.get_origin())
        vector.match_color(one_rect)
        vector.set_flat_stroke(False)
        label = Text("one", font_size=36)
        label.next_to(vector.get_end(), normalize(vector.get_vector()), buff=0.1)

        self.play(DrawBorderThenFill(one_rect))
        self.play(
            TransformFromCopy(words[one_index], label),
            GrowArrow(vector),
        )
        self.wait(3)

        # Knock in many directions
        new_labeled_vector_args = [
            ([2, 1, 0], "of two roads", None, UL),
            ([2, -1, -1], "symbolizing choice", None, UR),
            ([0.5, 1, -3], "contrasting the original\nwith the familiar", None, DR),
        ]
        new_labeled_vects = VGroup()
        last_vect = vector
        for args in new_labeled_vector_args:
            new_labeled_vects.add(self.get_added_vector(
                last_vect.get_end(), *args
            ))
            last_vect = new_labeled_vects[-1][0]
            last_vect.apply_depth_test()
        orientation_args = [
            (-4, -12, 0, (-0.89, 0.03, -0.41), 8.10),
            (3, -9, 0, (-0.34, 0.49, -0.63), 8.60),
            (34, -14, 0, (-0.59, 0.49, -0.62), 9.20),
            (20, -29, 0, (0.61, 0.01, 0.0), 6.10),
        ]


        for (vect, label), orientation in zip(new_labeled_vects, orientation_args):
            self.play(
                GrowArrow(vect, time_span=(2, 3)),
                FadeIn(label, 0.5 * DOWN, time_span=(2, 3)),
                frame.animate.reorient(*orientation).set_anim_args(run_time=6),
                ContextAnimation(
                    one_rect, phrase[:-16],
                    run_time=4,
                    fix_in_frame=True,
                    path_arc=60 * DEGREES,
                    lag_ratio=1e-3,
                    direction=UP,
                ),
            )
        self.play(
            frame.animate.reorient(22, -23, 0, (-0.86, 0.4, -0.35), 7.15),
            run_time=5
        )


# For chapter 7

class SimpleSpaceExample(InteractiveScene):
    def construct(self):
        # Setup axes
        frame = self.frame
        plane, axes = self.add_plane_and_axes()
        frame.reorient(14, 77, 0, (2.23, 0.25, 1.13), 4.46)

        # Show an initial vector in the space
        frame.add_ambient_rotation()
        vect = Arrow(axes.c2p(0, 0, 0), axes.c2p(2, -1, 1), buff=0)
        vect.set_color(BLUE)
        vect.always.set_perpendicular_to_camera(self.frame)
        label = Text("you", font_size=24)
        # label = Text("bank", font_size=24).set_backstroke(BLACK, 5)
        label.rotate(PI / 2, RIGHT)
        label.next_to(vect.get_center(), OUT + LEFT, buff=0)

        self.play(
            ShowCreation(vect),
            FadeIn(label, vect.get_vector())
        )
        self.wait(5)

        # Many directions -> Different kinds of meaning
        ideas = VGroup(
            Text("Part of a command"),
            Text("Affectionate"),
            Text("Sadness"),
        )
        ideas.set_backstroke(BLACK, 3)
        ideas.scale(0.35)
        ideas.rotate(PI / 2, RIGHT)

        last_idea = VGroup()
        last_direction = 1.0 * normalize(cross(RIGHT, vect.get_vector()))
        for idea in ideas:
            direction = rotate_vector(last_direction, PI / 3, vect.get_vector())
            new_vect = self.get_added_vector(vect, direction)
            new_vect.set_perpendicular_to_camera(self.frame)
            idea.next_to(new_vect.get_center(), buff=0.1)
            lines = get_direction_lines(axes, new_vect.get_vector(), color=new_vect.get_color())
            self.play(
                FadeOut(last_idea),
                ShowCreation(new_vect),
                FadeIn(idea, new_vect.get_vector()),
                LaggedStartMap(ShowCreationThenFadeOut, lines, lag_ratio=2 / len(lines), run_time=2)
            )
            self.wait(1)
            last_idea = VGroup(new_vect, idea)
            last_direction = direction
        self.play(FadeOut(last_idea))
        self.wait(5)

        # Specific ideas added onto "you"
        ideas = VGroup(
            Text("needs an adjective next"),
            Text("preceded by \"that which does not kill\""),
            Text("related to growth and strength"),
            # Text("River bank"),
            # Text("Beginning of a story"),
            # Text("Establishing a setting"),
        )
        ideas.scale(0.4)
        ideas.rotate(PI / 2, RIGHT)
        directions = [
            (-0.25, -1, 0.75),
            (-0.5, -0.25, 0.5),
            (1.0, -0.5, 1.0),
        ]
        orientations = [
            (11, 92, 0, (2.69, 0.55, 1.12), 6.25),
            (-8, 83, 0, (2.73, 0.56, 1.24), 6.80),
            (-14, 79, 0, (2.49, 0.61, 1.41), 7.64),
        ]

        vects = VGroup(vect)
        concepts = VGroup(label)
        for idea, direction, orientation in zip(ideas, directions, orientations):
            point = vects[-1].get_end()
            new_vect = self.get_added_vector(vects[-1], direction)
            new_vect.always.set_perpendicular_to_camera(self.frame)
            idea.next_to(new_vect.get_center())
            self.play(
                frame.animate.reorient(*orientation),
                GrowArrow(new_vect),
                FadeIn(idea, 0.5 * new_vect.get_vector())
            )
            self.wait(2)
            vects.add(new_vect)
        self.wait(15)

    def add_plane_and_axes(
        self,
        x_range=(-4, 4),
        y_range=(-4, 4),
        z_range=(-3, 3),
    ):
        axes = ThreeDAxes(x_range, y_range, z_range)
        plane = NumberPlane(
            x_range, y_range,
            background_line_style=dict(
                stroke_color=GREY_D,
                stroke_width=1
            ),
            faded_line_ratio=1,
        )
        plane.axes.set_stroke(GREY_D, 0)

        self.add(plane, axes)
        return plane, axes

    def get_added_vector(self, last_vect, direction):
        point = last_vect.get_end()
        new_vect = Arrow(point, point + direction, buff=0)
        new_vect.set_color(random_bright_color())
        new_vect.set_flat_stroke(False)
        return new_vect


class ManyIdeasManyDirections(SimpleSpaceExample):
    random_seed = 2

    def construct(self):
        # Axes
        frame = self.frame
        plane, axes = self.add_plane_and_axes()
        frame.reorient(-17, 73, 0, (-0.06, 0.11, 0.31), 6.03)
        frame.add_ambient_rotation()

        # Many directions -> Different kinds of meaning
        ideas = VGroup(
            Text(word)
            for word in [
                "Typewriter",
                "Paradigm",
                "Whimsical",
                "Gelatinous",
                "Rainbow",
                "Serendipitous",
                "Algorithm",
                "Nebulous",
                "Spatula",
                "Lethargic",
                "Effervescent",
                "Asteroid",
                "Pungent",
                "Daydream",
                "Mercurial",
                "Cactus",
                "Diaphanous",
                "Hiccup",
                "Viscous",
                "Thunderclap",
            ]
        )
        ideas.set_backstroke(BLACK, 3)
        ideas.scale(0.5)
        ideas.rotate(PI / 2, RIGHT)

        last_idea = VGroup()
        last_direction = RIGHT + OUT
        for idea in ideas:
            direction = normalize(cross(last_direction, np.random.uniform(-1, 1, 3)))
            new_vect = Vector(direction)
            new_vect.set_perpendicular_to_camera(self.frame)
            new_vect.set_color(random_bright_color())
            idea.next_to(new_vect.get_end(), direction, buff=0.1)
            lines = get_direction_lines(axes, direction, color=new_vect.get_color(), n_lines=250, stroke_width=2)
            idea.set_fill(interpolate_color(new_vect.get_color(), WHITE, 0.5))
            self.play(
                FadeOut(last_idea),
                GrowArrow(new_vect),
                FadeIn(idea, new_vect.get_vector()),
                LaggedStartMap(ShowCreationThenFadeOut, lines, lag_ratio=1 / len(lines), run_time=1.5)
            )
            self.wait()
            last_idea = VGroup(new_vect, idea)
            last_direction = direction
        self.play(FadeOut(last_idea))
        self.wait(5)


class MJSpace(SimpleSpaceExample):
    def construct(self):
        # Set up axes
        frame = self.frame
        plane, axes = self.add_plane_and_axes()
        axes.set_stroke(width=1)
        frame.add_ambient_rotation()

        # Show vectors landing in the space
        sentence = Text("Michael Jordan plays the sport of basketball", font_size=36)
        sentence.to_edge(UP)
        tokens = break_into_tokens(sentence)
        token_rects = get_piece_rectangles(tokens, leading_spaces=True, h_buff=0)
        arrs = VGroup(
            NumericEmbedding().scale(0.25).next_to(rect, DOWN, buff=1.0)
            for rect in token_rects
        )
        arrows = VGroup(Arrow(rect, arr, buff=0.1) for rect, arr in zip(token_rects, arrs))
        vects = VGroup(
            Vector(np.random.uniform(-3, 3, 3))
            for arr in arrs
        )
        vects.set_stroke(GREY_B)
        vects.fix_in_frame()

        VGroup(token_rects, tokens, arrows, arrs).fix_in_frame()

        frame.reorient(-18, 86, 0, (0.21, 0.12, 3.56), 11.65)
        self.add(token_rects, tokens)
        self.play(
            LaggedStartMap(FadeIn, arrs, shift=DOWN, lag_ratio=0.1),
            LaggedStartMap(GrowArrow, arrows, lag_ratio=0.1),
        )
        self.wait()
        self.play(
            frame.animate.reorient(11, 76, 0, ORIGIN, FRAME_HEIGHT),
            FadeOut(VGroup(token_rects, tokens, arrows), UP, time_span=(1, 2)),
            LaggedStart(
                (Transform(arrow, vect)
                for arrow, vect in zip(arrs, vects)),
                lag_ratio=0.05,
            ),
            run_time=3
        )
        self.remove(arrs)
        self.add(vects)
        self.wait()
        self.play(LaggedStart(
            (vect.animate.scale(0, about_point=vect.get_start())
            for vect in vects),
            lag_ratio=0.05,
            remover=True
        ))

        # Show three directions
        colors = [YELLOW, RED, "#F88158"]
        all_coords = [normalize([-1, -1, 1])]
        all_coords.append(normalize(cross(all_coords[0], IN)))
        all_coords.append(-normalize(cross(all_coords[0], all_coords[1])))
        all_coords = np.array(all_coords)[[0, 2, 1]]
        labels = VGroup(*map(Text, ["First Name Michael", "Last Name Jordan", "Basketball"]))
        label_directions = [LEFT + OUT, IN, RIGHT + OUT]

        vect_groups = VGroup()
        vects = VGroup()
        for coords, label, color, direction in zip(all_coords, labels, colors, label_directions):
            vect = Vector(2.0 * coords)
            vect.set_color(color)
            vect.always.set_perpendicular_to_camera(self.frame)
            label.scale(0.5)
            label.rotate(PI / 2, RIGHT)
            label.set_color(color)
            label.next_to(vect.get_end(), direction, buff=0.1)
            label.set_fill(border_width=0.5)
            label.set_backstroke(BLACK, 4)
            vects.add(vect)
            vect_groups.add(VGroup(vect, label))

        orientations = [
            (17, 76, 0),
            (17, 80, 0),
            (-16, 77, 0),
        ]

        for vect, label, orientation in zip(vects, labels, orientations):
            lines = get_direction_lines(axes, vect.get_vector(), color=vect.get_color())
            self.play(
                GrowArrow(vect),
                FadeIn(label, vect.get_vector()),
                frame.animate.reorient(*orientation),
            )
            self.play(
                LaggedStartMap(ShowCreationThenFadeOut, lines, lag_ratio=2 / len(lines))
            )
            self.wait(2)

        # Bring in "plucked out" vector
        emb_coords = 2.0 * all_coords[:2].sum(0)
        emb = Vector(emb_coords)
        emb.always.set_perpendicular_to_camera(self.frame)
        emb.set_flat_stroke(False)
        emb_label = Tex(R"\vec{\textbf{E}}", font_size=30)
        emb_label.rotate(89 * DEGREES, RIGHT)
        emb_label.add_updater(lambda m: m.move_to(1.1 * emb.get_end()))
        emb_label.suspend_updating()

        self.play(
            frame.animate.reorient(7, 66, 0).set_anim_args(run_time=2),
            FadeIn(emb, shift=2 * (IN + LEFT)),
            FadeIn(emb_label, shift=2 * (IN + LEFT)),
        )
        self.wait()

        # Set up dot product display
        def get_proj_point(vect1, vect2):
            v1 = vect1.get_end()
            v2 = vect2.get_end()
            return v2 * np.dot(v1, v2) / np.dot(v2, v2)

        def get_dot_product_lines(vect, proj_line_color=GREY_A):
            dashed_line = always_redraw(
                lambda: Line(emb.get_end(), get_proj_point(emb, vect)).set_stroke(WHITE, 2).set_anti_alias_width(10)
            )
            proj_line = always_redraw(
                lambda: Line(ORIGIN, get_proj_point(emb, vect)).set_stroke(proj_line_color, width=4, opacity=0.75)
            )
            return dashed_line, proj_line

        m_dashed_line, m_proj_line = get_dot_product_lines(vects[0])

        formula = Tex(R"\vec{\textbf{E}} \cdot \big(\overrightarrow{\text{First Name Michael}}\big) = ", font_size=36)
        formula[3:-1].set_color(YELLOW)
        formula.to_corner(UL)
        formula.fix_in_frame()
        rhs = DecimalNumber(font_size=42)
        rhs.fix_in_frame()
        rhs.next_to(formula[-1], RIGHT, buff=0.15)
        rhs.target_vect = vects[0]
        rhs.add_updater(lambda m: m.set_value(np.dot(m.target_vect.get_end(), emb.get_end()) / 4.0))

        m_proj_line.suspend_updating()
        self.play(
            ShowCreation(m_dashed_line),
            TransformFromCopy(Line(ORIGIN, emb.get_end(), flat_stroke=False), m_proj_line),
            FadeIn(formula, UP),
            vect_groups[1:].animate.set_opacity(0.25),
        )
        m_proj_line.resume_updating()
        self.play(
            TransformFromCopy(rhs.copy().unfix_from_frame().set_opacity(0).move_to(m_proj_line), rhs),
        )
        emb_label.resume_updating()
        for _ in range(2):
            self.play(
                emb.animate.put_start_and_end_on(ORIGIN, [-2.5, -2.0, -0.5]),
                rate_func=wiggle,
                run_time=5
            )
        self.wait(2)
        self.play(emb.animate.put_start_and_end_on(axes.get_origin(), 1.5 * all_coords[1:3].sum(0)), run_time=3)
        self.play(frame.animate.reorient(26, 68, 0), run_time=2 )
        self.play(emb.animate.put_start_and_end_on(ORIGIN, [1.0, -1.5, -1.0]), run_time=3)
        self.wait(2)
        self.play(
            frame.animate.reorient(-4, 73, 0),
            emb.animate.put_start_and_end_on(ORIGIN, emb_coords),
            run_time=3
        )
        self.wait(5)

        # Dotting against L.N. Jordan
        j_dashed_line, j_proj_line = get_dot_product_lines(vects[1])
        j_paren = Tex(R"\big(\overrightarrow{\text{Last Name Jordan}}\big) = ", font_size=36)
        j_paren[:-1].set_color(RED)
        m_paren = formula[3:]
        m_paren.fix_in_frame()
        j_paren.move_to(m_paren, LEFT)
        j_paren.fix_in_frame()
        rhs.target_vect = vects[1]

        self.play(
            frame.animate.reorient(15, 97, 0),
            FadeOut(m_paren, UP, time_span=(1, 2)),
            FadeIn(j_paren, UP, time_span=(1, 2)),
            rhs.animate.next_to(j_paren, RIGHT, buff=0.15).set_anim_args(time_span=(1, 2)),
            LaggedStart(
                vect_groups[0].animate.set_opacity(0.25),
                vect_groups[1].animate.set_opacity(1),
                FadeOut(m_dashed_line),
                FadeOut(m_proj_line),
                lag_ratio=0.25,
                run_time=2
            )
        )
        j_proj_line.suspend_updating()
        self.play(
            ShowCreation(j_dashed_line),
            TransformFromCopy(Line(ORIGIN, emb.get_end(), flat_stroke=False), j_proj_line),
        )
        j_proj_line.resume_updating()
        self.play(
            emb.animate.put_start_and_end_on(ORIGIN, [-1.5, -1.5, 0]).set_anim_args(run_time=3, rate_func=there_and_back)
        )
        self.wait()

        # Dotting against basketball
        b_dashed_line, b_proj_line = get_dot_product_lines(vects[2])
        b_paren = Tex(R"\big(\overrightarrow{\text{Basketball}}\big) = ", font_size=36)
        b_paren[:-1].set_color(vects[2].get_color())
        b_paren.move_to(m_paren, LEFT)
        b_paren.fix_in_frame()
        rhs.suspend_updating()

        self.play(
            frame.animate.reorient(2, 65, 0),
            FadeOut(j_paren, UP),
            FadeIn(b_paren, UP),
            rhs.animate.next_to(b_paren[-1], RIGHT, buff=0.2).set_value(0),
            FadeOut(j_dashed_line),
            FadeOut(j_proj_line),
            vect_groups[1].animate.set_opacity(0.25),
            vect_groups[2].animate.set_opacity(1.0),
        )
        self.wait()

        rhs.target_vect = vects[2]
        rhs.resume_updating()
        self.add(b_dashed_line, b_proj_line)
        self.play(
            emb.animate.put_start_and_end_on(ORIGIN, [0.6, -2.2, 0]),
            rate_func=there_and_back,
            run_time=6,
        )
        self.wait(3)

        # Emphasize dot products with first two names
        self.play(
            frame.animate.reorient(5, 85, 0).set_anim_args(run_time=2),
            FadeOut(formula[:3]),
            FadeOut(b_paren),
            FadeOut(rhs),
            FadeOut(b_dashed_line),
            FadeOut(b_proj_line),
            vect_groups[:2].animate.set_opacity(1),
            vect_groups[2].animate.set_opacity(0.25),
        )
        self.wait()
        self.play(
            ShowCreation(m_dashed_line),
            ShowCreation(m_proj_line),
        )
        self.wait()
        self.play(
            ShowCreation(j_dashed_line),
            ShowCreation(j_proj_line),
        )
        self.wait(20)
        self.play(
            *map(FadeOut, [j_dashed_line, j_proj_line, m_dashed_line, m_proj_line, emb, emb_label]),
        )

        # Show sum of the first two names
        j_vect_copy, m_vect_copy = vect_copies = vects[:2].copy()
        vect_copies.clear_updaters()
        vect_copies.set_stroke(opacity=0.5)
        j_vect_copy.shift(vects[1].get_vector())
        m_vect_copy.shift(vects[0].get_vector())
        emb.put_start_and_end_on(axes.get_origin(), m_vect_copy.get_end())

        self.play(frame.animate.reorient(-6, 78, 0), run_time=2)
        self.play(LaggedStart(
            TransformFromCopy(vects[1], m_vect_copy),
            TransformFromCopy(vects[0], j_vect_copy),
            lag_ratio=0.5
        ))
        self.play(GrowArrow(emb))
        self.wait(4)

        # Show the basketball direction
        self.play(
            *map(FadeOut, [m_vect_copy, j_vect_copy, emb])
        )
        self.play(
            frame.animate.reorient(-19, 77, 0, (1.32, -0.22, -0.12), 3.75),
            vect_groups[:2].animate.set_opacity(0.25),
            vect_groups[2][0].animate.set_opacity(1.0),
            vect_groups[2][1].animate.set_opacity(1.0),
            run_time=2
        )
        self.wait(20)
