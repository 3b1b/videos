import gensim.downloader
import tiktoken
from pathlib import Path

from manim_imports_ext import *
from _2024.transformers.objects import *


DATA_DIR = Path(get_output_dir(), "2024/transformers/data/")
WORD_FILE = Path(DATA_DIR, "OWL3_Dictionary.txt")


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


class LyingAboutTokens(InteractiveScene):
    def construct(self):
        # Mention next word prediction task
        phrase = Text("The goal of our model is to predict the next word")

        words = break_into_words(phrase)
        rects = get_piece_rectangles(words, leading_spaces=False, h_buff=0.05)

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
            NumericEmbedding(length=5)
            for word in words
        ))
        vectors.arrange(RIGHT, buff=0.5 * vectors[0].get_width())
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
            Text("A Small Lie", font_size=72).to_edge(UP),
            Text("True Truth", font_size=72).next_to(h_line, DOWN),
        )
        h_line.set_stroke(WHITE, 2)
        h_line.next_to(titles[1], UP)
        for title in titles:
            title.add(Underline(title))

        # Show the lie
        phrase1, phrase2 = phrases = VGroup(
            Text("We'll pretend that text is cleanly broken into distinct words"),
            Text("The true subdivision (known fancifully as tokenization) looks different"),
        )
        for phrase, title in zip(phrases, titles):
            phrase.set_width(FRAME_WIDTH - 1)
            phrase.next_to(title, DOWN, buff=1.0)

        words = break_into_words(phrase1)
        tokens = break_into_tokens(phrase2)
        word_rects = get_piece_rectangles(words, hue_range=(0.5, 0.6))
        token_rects = get_piece_rectangles(tokens, hue_range=(0.1, 0.2), leading_spaces=True, h_buff=0.0)

        self.play(
            FadeOut(blocks),
            FadeOut(q_group),
            FadeOut(arrows),
            FadeOut(vectors),
            ShowCreation(h_line),
            FadeIn(title1, lag_ratio=0.1),
            FadeIn(words),
        )
        self.add(word_rects, words)
        self.play(
            LaggedStartMap(FadeIn, word_rects),
            LaggedStart(*(
                word.animate.set_color(rect.get_color())
                for word, rect in zip(words, word_rects)
            ))
        )
        self.wait()
        self.play(
            FadeIn(title2, lag_ratio=0.1),
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

        # Analyze tokenization
        brace = Brace(token_rects[10], buff=0.05)

        self.play(GrowFromCenter(brace))
        self.wait()
        for index in [6, 7, 5, 3, 9, 11]:
            self.play(brace.animate.become(Brace(token_rects[index], buff=0.05)))
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
        for index in [9, -7, 7, -5, -6]:
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
        for x in range(3):
            self.play(
                RandomizeMatrixEntries(matrix, lag_ratio=0.01),
                LaggedStartMap(VShowPassingFlash, rects, lag_ratio=0.01, time_width=1.5),
                run_time=2,
            )
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
        stroke_width=5,
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
        label_config["label_buff"] = buff
        return LabeledArrow(
            axes.get_origin(),
            point,
            stroke_width=stroke_width,
            stroke_color=color,
            flat_stroke=False,
            label_text=word if func_name is None else f"{func_name}({word})",
            buff=0,
            direction=direction,
            **label_config,
        )


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
        word = Text("aardvark")
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
        base_bubble: VMobject = ThoughtBubble()[-1]
        base_bubble.set_shape(8, 7)
        base_bubble.rotate(PI)
        base_bubble.set_fill(GREY_D, opacity=[0.5, 1, 0.5])
        base_bubble.move_to(bubble_center)

        def get_bubble():
            result = base_bubble.copy()
            result.apply_complex_function(
                lambda z: z * (1 + 0.025 * np.cos(5 * np.log(z).imag + self.time))
            )
            result.move_to(bubble_center)
            return result

        bubble = always_redraw(get_bubble)
        bubble_label = Text("Embedding space", font_size=72)
        bubble_label.move_to(bubble)
        bubble_label.shift(1.0 * UP)
        q_marks = Tex("???", font_size=120)
        q_marks.next_to(bubble_label, DOWN, buff=0.5)
        self.add(bubble)
        self.add(bubble_label, q_marks)
        self.wait(10)

        # Show dimension
        brace = Brace(vect, RIGHT)
        label = VGroup(
            Integer(12288),
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
        dimension_label.next_to(bubble_label, DOWN)

        self.play(
            GrowFromCenter(brace),
            CountInFrom(label[0], 0),
            FadeIn(label[1]),
        )
        self.wait()
        self.play(
            TransformFromCopy(label[0], dimension_label[0]),
            FadeInFromPoint(dimension_label[1], label[0].get_center()),
            q_marks.animate.next_to(dimension_label, DOWN, buff=0.5)
        )
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
        phrase = "The big idea as a model tweaks and tunes its weights"
        words = [word.lower() for word in phrase.split(" ")]

        # Get initial and final states
        colors = [random_bright_color(hue_range=(0.5, 0.6)) for word in words]
        true_embeddings = np.array([
            self.basis @ self.model[word]
            for word in words
        ])
        true_embeddings -= true_embeddings.mean(0)
        true_embeddings *= 3 / np.abs(true_embeddings).max(0)

        np.random.seed(2)
        thetas = np.arange(0, TAU, TAU / len(words))
        thetas += np.random.uniform(-0.5, 0.5, thetas.size)
        amps = np.random.uniform(2, 4, thetas.size)
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
            label.set_backstroke(BLACK, 8)
            label.vect = vect
            label.add_updater(lambda m: m.move_to(
                m.vect.get_end() + 0.25 * normalize(m.vect.get_vector())
            ))
            labels.add(label)

        for vect, label in zip(word_vects, labels):
            self.add(vect)
            self.play(FadeIn(label, scale=1.2, run_time=0.2))
            self.wait(0.1 * len(label) - 0.2)

        # Tweak and tune weights
        turn_animation_into_updater(
            ApplyMethod(frame.reorient, -29, 70, 0, (-0.04, -0.18, -0.5), 8.00),
            run_time=8
        )
        self.progressive_nudges(word_vects, true_embeddings, 8)
        frame.clear_updaters()
        turn_animation_into_updater(
            ApplyMethod(frame.reorient, 29, 70, 0, (-0.32, 0.02, -0.54), 7.68),
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
        frame.reorient(-178, 9, 178, (1.72, 0.91, 0.73), 6.80)

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
        diff = FillArrow(man.get_end(), woman.get_end(), buff=0, stroke_color=YELLOW)
        diff.set_fill(YELLOW, opacity=0.8)
        diff.set_backstroke(BLACK, 3)
        self.play(
            LaggedStart(*map(Write, [ew, minus2, em])),
            GrowArrow(woman),
            FadeInFromPoint(woman.label, man.get_center()),
            GrowArrow(man),
            FadeInFromPoint(man.label, man.get_center()),
            GrowArrow(diff, time_span=(2, 3)),
            frame.animate.reorient(0, 0, 0, (2.04, 2.06, 0.38), 4.76).set_anim_args(run_time=3)
        )
        self.play(frame.animate.reorient(-179, 19, 179, (2.49, 1.96, 0.4), 4.76), run_time=5)

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

    def get_equation1(self, word1, word2, word3, word4, colors=None):
        equation = TexText(
            Rf"E({word1}) - E({word2}) $\approx$ E({word3}) - E({word4})",
            font_size=48
        )
        equation.fix_in_frame(True)
        equation.to_corner(UR)
        if colors:
            words = [word1, word2, word3, word4]
            for word, color in zip(words, colors):
                equation[word].set_fill(color)
        pieces = VGroup(
            equation[f"E({word1})"][0],
            equation["-"][0],
            equation[f"E({word2})"][0],
            equation[R"$\approx$"][0],
            equation[f"E({word3})"][0],
            equation["-"][1],
            equation[f"E({word4})"][0],
        )
        pieces.fix_in_frame(True)
        return pieces

    def get_equation2(self, word1, word2, word3, word4, colors=None):
        equation = TexText(
            Rf"E({word1}) + E({word2}) - E({word3}) $\approx$ E({word4})",
            font_size=48
        )
        equation.fix_in_frame(True)
        equation.to_corner(UR)
        if colors:
            words = [word1, word2, word3, word4]
            for word, color in zip(words, colors):
                equation[word].set_fill(color)
        pieces = VGroup(
            equation[f"E({word1})"],
            equation["+"],
            equation[f"E({word2})"],
            equation["-"],
            equation[f"E({word3})"],
            equation[R"$\approx$ "],
            equation[f"E({word4})"],
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
        frame.move_to(1.5 * OUT)
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
    words = ["Sushi", "Japan", "Germany", "Bratwurst"]
    colors = [WHITE, "#BC002D", "#FFCC00", interpolate_color(GREY_BROWN, WHITE, 0.25)]
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


class ShowNearestNeighbors(Word2VecScene):
    seed_word = "tower"
    color = YELLOW
    n_shown = 10
    frame_height = 4
    frame_center = (2.18, 0.09, 0.72)
    frame_orientation = (-21, 87, 0)

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
        nearest_words = find_nearest_words(self.model, self.model[word], self.n_shown + 1)[1:]
        neighbors = VGroup(*(
            self.get_labeled_vector(word, color=WHITE)
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
            self.wait(0.5)
        self.play(last_neighbor.animate.set_opacity(0.2))

        self.wait(10)

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
        result = rhs.make_number_changable("+0.00", include_sign=True)
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


class DotProductWithGenderDirection(InteractiveScene):
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
    number_line_y = -1.5
    threshold = 1.0

    def construct(self):
        # Initialize equation
        self.model = get_word_to_vec_model()
        words = self.words
        eq_lhs = self.get_equation_lhs(words[0])

        # Write gender equation
        gen_lhs = eq_lhs[0].copy()
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
            numbers_with_elongated_ticks=list(np.arange(*x_range[:2])),
            tick_size=0.05,
            longer_tick_multiple=2.5,
            width=12
        )
        number_line.add_numbers(
            np.arange(*x_range[:2]),
            num_decimal_places=1,
            font_size=30,
        )
        number_line.move_to(self.number_line_y * UP)
        eq_rhs = self.get_equation_rhs(eq_lhs, words[0])
        equation = VGroup(eq_lhs, eq_rhs)
        low_brace = Brace(equation, DOWN)
        arrow = Vector(DOWN).next_to(low_brace, DOWN)
        equation_group = VGroup(equation, low_brace, arrow)
        dp = eq_rhs.get_value()

        word = eq_lhs[2][2:-1]
        lil_word = word.copy().scale(0.25)
        dot = GlowDot(color=word[0].get_color())
        dot.move_to(number_line.n2p(dp))

        lil_word.next_to(dot, UP, buff=0)
        equation_group.next_to(dot, UP)

        self.play(
            TransformFromCopy(gen_lhs, eq_lhs[0]),
            FadeIn(eq_lhs[1:], shift=RIGHT),
            FadeIn(low_brace, shift=RIGHT),
            FadeIn(arrow, shift=RIGHT),
            UpdateFromAlphaFunc(
                eq_rhs,
                lambda m, a: m.set_value(a * dp).next_to(eq_lhs[-1], RIGHT),
                run_time=1,
            ),
            Write(number_line, run_time=1)
        )
        self.play(
            FadeIn(dot, DOWN),
            # TransformFromCopy(word, lil_word)
        )
        self.wait()

        # Show some alternate
        new_rhs = eq_rhs.copy()
        eq_rhs.set_opacity(0)
        new_rhs.add_updater(lambda m: m.set_value(number_line.p2n(arrow.get_center())))
        new_rhs.add_updater(lambda m: m.next_to(eq_lhs[-1], RIGHT))
        self.add(new_rhs)

        low_brace.add_updater(lambda m: m.become(Brace(equation, DOWN)))

        for new_word in words[1:]:
            new_dp = self.get_gender_value(new_word)
            nl_point = number_line.n2p(new_dp)
            color = self.colors[int(new_dp > self.threshold)]
            new_dot = GlowDot(number_line.n2p(new_dp), color=color)
            new_lhs = self.get_equation_lhs(new_word)
            new_rhs = self.get_equation_rhs(new_lhs, new_word)
            new_rhs.set_opacity(0)
            new_equation = VGroup(new_lhs, new_rhs)
            new_equation.match_y(equation)
            new_equation.match_x(nl_point)

            word = new_lhs[2][2:-1]
            lil_word = word.copy().scale(0.25)
            lil_word.next_to(new_dot, UP, buff=0)

            self.play(
                Transform(equation, new_equation),
                arrow.animate.match_x(nl_point)
            )

            self.play(
                FadeIn(new_dot, DOWN),
                # TransformFromCopy(word, lil_word),
            )
            self.wait()

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
        value = self.get_gender_value(word)
        parts[2][2:-1].set_color(self.colors[int(value > self.threshold)])
        return VGroup(*parts)

    def get_equation_rhs(self, equation_lhs, word):
        rhs = DecimalNumber(self.get_gender_value(word))
        rhs.next_to(equation_lhs, RIGHT)
        return rhs

    def get_gender_value(self, word):
        rf1, rf2 = self.ref_words
        return np.dot(
            (self.model[rf2] - self.model[rf1]).flatten(),
            self.model[word].flatten(),
        )


class DotProductWithPluralityDirection(DotProductWithGenderDirection):
    vec_tex = R"\vec{\text{plur}}"
    ref_words = ["cat", "cats"]
    words = [
        "octopus", "octopi",
        "puppy", "puppies",
        "one", "two", "three", "four",
        "single", "multiple",
    ]
    x_range = (-8, 5 + 1e-4, 0.25)
    colors = [BLUE, RED]
    threshold = -1.0


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
            frame.animate.reorient(2, -16, 0, (0.6, -0.04, 0.02), 6.01).set_anim_args(run_time=3),
        )
        self.play(
            GrowArrow(vect2),
            FadeIn(label2, 0.5 * DOWN),
            frame.animate.reorient(35, -23, 0, (0.6, -0.04, 0.02), 6.01).set_anim_args(run_time=2),
        )
        self.play(
            GrowArrow(vect3),
            FadeIn(label3, 0.5 * DOWN),
            frame.animate.reorient(20, -29, 0, (0.61, 0.01, 0.0), 6.10).set_anim_args(run_time=3),
        )
        self.play(
            frame.animate.reorient(-19, -25, 0, (0.61, 0.01, 0.0), 6.10),
            run_time=5
        )

    def get_added_vector(self, curr_tip, direction, label, color=None, next_to_direction=UP, font_size=24):
        if color is None:
            color = random_bright_color(hue_range=(0.45, 0.65))
        vect = Vector(direction)
        vect.set_color(color)
        vect.set_flat_stroke(False)
        vect.shift(curr_tip)
        text = Text(label, font_size=font_size)
        text.set_backstroke(BLACK, 4)
        text.next_to(vect.get_center(), next_to_direction, buff=0.1)
        text.set_fill(border_width=0)

        result = VGroup(vect, text)
        return result


class SoftmaxBreakdown(InteractiveScene):
    def construct(self):
        # Show example probability distribution
        word_strs = ['Dumbledore', 'Flitwick', 'Mcgonagall', 'Quirrell', 'Snape', 'Sprout', 'Trelawney']
        words = VGroup(*map(Text, word_strs))
        values = np.array([0.3, -1, 0.5, 1.5, 3.4, -1, 2.5])
        prob_values = softmax(values)
        chart = BarChart(prob_values, width=10)

        probs = VGroup(*(DecimalNumber(pv) for pv in prob_values))
        probs.arrange(DOWN, buff=0.25)
        probs.generate_target()
        for prob, bar in zip(probs.target, chart.bars):
            prob.scale(0.5)
            prob.next_to(bar, UP)

        for word, bar in zip(words, chart.bars):
            word.scale(0.75)
            height = word.get_height()
            word.move_to(bar.get_bottom(), LEFT)
            word.rotate(-45 * DEGREES, about_point=bar.get_bottom())
            word.shift(height * DOWN)

        chart.save_state()
        for bar in chart.bars:
            bar.stretch(0, 1, about_edge=DOWN)
        chart.set_opacity(0)

        self.play(LaggedStartMap(FadeIn, probs, shift=0.25 * DOWN, lag_ratio=0.3))
        self.wait()
        self.play(
            Restore(chart, lag_ratio=0.1),
            MoveToTarget(probs),
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeIn, words),
        )
        self.wait()

        # Show constraint between 0 and 1
        bar = chart.bars[0]
        bar.save_state()
        prob = probs[0]
        prob.bar = bar
        max_height = chart.y_axis.get_y(UP) - chart.x_axis.get_y()
        prob.add_updater(lambda p: p.set_value(p.bar.get_height() / max_height))
        prob.add_updater(lambda p: p.match_height(probs[1]))
        prob.add_updater(lambda p: p.next_to(p.bar, UP))

        one_line = DashedLine(*chart.x_axis.get_start_and_end())
        one_line.set_stroke(RED, 2)
        one_line.align_to(chart.y_axis, UP)

        self.play(
            FadeIn(one_line, time_span=(0, 1)),
            bar.animate.set_height(max_height, about_edge=DOWN, stretch=True),
            run_time=2,
        )
        self.play(
            bar.animate.set_height(1e-4, about_edge=DOWN, stretch=True),
            run_time=2,
        )
        self.play(Restore(bar))
        self.wait()
        prob.clear_updaters()

        # Show sum
        prob_copies = probs.copy()
        prob_copies.scale(1.5)
        prob_copies.arrange(RIGHT, buff=1.0)
        prob_copies.to_edge(UP)
        prob_copies.shift(LEFT)
        plusses = VGroup(*(
            Tex("+").move_to(VGroup(p1, p2))
            for p1, p2 in zip(prob_copies, prob_copies[1:])
        ))
        equals = Tex("=").next_to(prob_copies, RIGHT)
        rhs = DecimalNumber(1.00)
        rhs.next_to(equals, RIGHT)

        self.play(
            TransformFromCopy(probs, prob_copies),
            Write(plusses),
            Write(equals),
            FadeOut(one_line),
        )
        globals().update(locals())
        self.play(
            LaggedStart(*(
                FadeTransform(pc.copy(), rhs)
                for pc in prob_copies
            ), lag_ratio=0.07)
        )
        self.wait()

        sum_group = VGroup(*prob_copies, *plusses, equals, rhs)
        chart_group = VGroup(chart, probs, words)

        # Show example matrix vector output
        n = len(words)
        vector = NumericEmbedding(length=n, ellipses_index=None)
        in_values = np.array([e.get_value() for e in vector.elements])
        rows = []
        for value in values:
            row = np.random.uniform(-1, 1, len(in_values))
            row *= value / np.dot(row, in_values)
            rows.append(row)
        matrix_values = np.array(rows)

        matrix = WeightMatrix(
            values=matrix_values,
            ellipses_row=None,
            ellipses_col=None,
            num_decimal_places=2,
        )
        for mob in matrix, vector:
            mob.set_height(4)
        vector.to_edge(UP).set_x(2.5)
        matrix.next_to(vector, LEFT)

        self.play(
            chart_group.animate.scale(0.35).to_corner(DL),
            FadeOut(sum_group),
            FadeIn(matrix, lag_ratio=0.01),
            FadeIn(vector, lag_ratio=0.01),
        )
        eq, rhs = show_matrix_vector_product(self, matrix, vector, x_max=9)
        self.wait()

        # Comment on output
        rhs_rect = SurroundingRectangle(rhs)
        rhs_words = Text("Not at all a\nprobability distribution!")
        rhs_words.next_to(rhs_rect, DOWN)

        neg_rects = VGroup(*(
            SurroundingRectangle(entry)
            for entry in rhs.get_entries()
            if entry.get_value() < 0
        ))
        gt1_rects = VGroup(*(
            SurroundingRectangle(entry)
            for entry in rhs.get_entries()
            if entry.get_value() > 1
        ))
        VGroup(rhs_rect, neg_rects).set_stroke(RED, 4)
        gt1_rects.set_stroke(BLUE, 4)

        for rect in (*neg_rects, *gt1_rects):
            neg = rect in neg_rects
            rect.word = Text("Negative" if neg else "> 1", font_size=36)
            rect.word.match_color(rect)
            rect.word.next_to(rhs, RIGHT)
            rect.word.match_y(rect)
        neg_words = VGroup(*(r.word for r in neg_rects))
        gt1_words = VGroup(*(r.word for r in gt1_rects))

        sum_arrow = Vector(DOWN).next_to(rhs, DOWN)
        sum_sym = Tex(R"\sum", font_size=36).next_to(sum_arrow, LEFT)
        sum_num = DecimalNumber(sum(e.get_value() for e in rhs.get_entries()))
        sum_num.next_to(sum_arrow, DOWN)

        self.play(
            ShowCreation(rhs_rect),
            FadeIn(rhs_words),
        )
        self.wait()
        self.play(
            ReplacementTransform(VGroup(rhs_rect), neg_rects),
            LaggedStart(*(FadeIn(rect.word, 0.5 * RIGHT) for rect in neg_rects)),
        )
        self.wait()
        self.play(
            ReplacementTransform(neg_rects, gt1_rects),
            FadeTransformPieces(neg_words, gt1_words),
        )
        self.wait()
        self.play(
            LaggedStart(
                FadeOut(rhs_words),
                FadeOut(gt1_rects),
                FadeOut(gt1_words),
            ),
            GrowArrow(sum_arrow),
            FadeIn(sum_num, DOWN),
            FadeIn(sum_sym),
        )
        self.wait()
        self.play(*map(FadeOut, [sum_arrow, sum_sym, sum_num]))

        # Preview softmax application
        rhs.generate_target()
        rhs.target.to_edge(LEFT, buff=1.5)
        rhs.target.set_y(0)

        softmax_box = Rectangle(
            width=5,
            height=rhs.get_height() + 1,
        )
        softmax_box.set_stroke(BLUE, 2)
        softmax_box.set_fill(BLUE_E, 0.5)
        in_arrow, out_arrow = Vector(RIGHT).replicate(2)
        in_arrow.next_to(rhs.target, RIGHT)
        softmax_box.next_to(in_arrow, RIGHT)
        out_arrow.next_to(softmax_box, RIGHT)

        softmax_label = Text("softmax", font_size=60)
        softmax_label.move_to(softmax_box)

        rhs_values = np.array([e.get_value() for e in rhs.get_entries()])
        dist = softmax(rhs_values)
        output = DecimalMatrix(dist.reshape((dist.shape[0], 1)))
        output.match_height(rhs)
        output.next_to(out_arrow, RIGHT)

        bars = chart.bars.copy()
        for bar, entry in zip(bars, output.get_entries()):
            bar.rotate(-PI / 2)
            bar.stretch(2, 0)
            bar.next_to(output)
            bar.match_y(entry)

        self.play(LaggedStart(
            FadeOut(matrix, 2 * LEFT),
            FadeOut(vector, 3 * LEFT),
            FadeOut(eq, 3.5 * LEFT),
            FadeOut(chart_group, DL),
            TransformFromCopy(chart.bars, bars),
            GrowArrow(in_arrow),
            FadeIn(softmax_box, RIGHT),
            FadeIn(softmax_label, RIGHT),
            MoveToTarget(rhs),
            GrowArrow(out_arrow),
            FadeIn(output, RIGHT),
        ), lag_ratio=0.2, run_time=2)
        self.wait()

        # Highlight larger and smaller parts

        # Exponentiate each part

        # Compute the sum

        # Divide each part by the sum

        # Comment on largest values
