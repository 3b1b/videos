import gensim.downloader

from manim_imports_ext import *
from _2024.transformers.objects import *


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
        self.model = gensim.downloader.load(self.embedding_model)

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

    def get_labeled_vector(
        self,
        word,
        coords=None,
        arrow=True,
        stroke_width=5,
        color=YELLOW,
        func_name: str | None = "E",
        label_config: dict = dict()
    ):
        # Return an arrow (or dot?) with word label next to it
        axes = self.axes
        if coords is None:
            coords = self.basis @ self.model[word.lower()]
        point = axes.c2p(*coords)
        arrow = LabeledArrow(
            axes.get_origin(),
            point,
            buff=0,
            stroke_width=stroke_width,
            stroke_color=color,
            flat_stroke=False,
            label_text=word if func_name is None else f"{func_name}({word})",
            **label_config,
        )
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

        self.add(titles)
        self.add(arrow)

        # Add words
        words = "All data in deep learning must be represented as vectors".split(" ")
        pre_labels = VGroup(*(Text(word) for word in words))
        pre_labels.fix_in_frame()
        pre_labels.arrange(DOWN)
        pre_labels.next_to(titles[0], DOWN)
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

        self.play(LaggedStartMap(FadeIn, pre_labels, shift=UP))

        # Transition
        for label, vect in zip(pre_labels, embeddings):
            self.add(turn_animation_into_updater(
                TransformFromCopy(label, vect.label, run_time=2)
            ))
            self.add(turn_animation_into_updater(
                FadeIn(vect, run_time=1)
            ))
            self.wait(0.5)

        self.wait(15)

#


class KingQueenExample(Word2VecScene):
    default_frame_orientation = (20, 70)

    def get_basis(self, model):
        basis = super().get_basis(model)
        basis[1] *= 2
        return basis

    def construct(self):
        # Setup
        axes = self.axes
        frame = self.frame

        words = ["man", "woman", "king", "queen"]
        colors = [BLUE_B, RED_B, BLUE_D, RED_D]
        directions = [OUT + UP, RIGHT, OUT + UP, RIGHT]

        # Prepare vector groups
        man, woman, king, queen = word_vects = [
            self.get_labeled_vector(word, color=color, buff=0.05, direction=direction)
            for word, color, direction in zip(words, colors, directions)
        ]

        fake_queen_coords = self.basis @ (self.model["king"] - self.model["man"] + self.model["woman"])  # Tweak queen for demo purposes
        fake_queen_coords += np.array([0.05, -0.02, -0.01])
        fake_queen = self.get_labeled_vector(
            "queen", fake_queen_coords,
            color=colors[3]
        )

        # Show man and woman vectors
        for vect in [man, woman]:
            self.play(
                GrowArrow(vect),
                FadeInFromPoint(vect.label, axes.get_center()),
                frame.animate.reorient(5, 60, 0).move_to([1.0, 0.5, 0.5]).set_height(3.00),
                run_time=2
            )

        # Equation
        equation = self.get_equation("king", "woman", "man", "queen")
        equation.shift(0.5 * LEFT)
        ek, plus, ew, minus, em, approx, eq = equation

        for part, vect in zip([em, ew, ek, eq], word_vects):
            part.set_fill(vect.label.get_fill_color())

        # Show difference
        diff = Arrow(man.get_end(), woman.get_end(), buff=0, stroke_color=YELLOW)
        self.play(
            ReplacementTransform(ew.copy().scale(0.5).shift(2.7 * DOWN + 2.5 * RIGHT).fade(1.0), ew),
            ReplacementTransform(em.copy().scale(0.5).shift(1.75 * DOWN + 0.0 * LEFT).fade(1.0), em),
            FadeIn(minus, UP),
            run_time=2
        )
        self.play(
            ShowCreation(diff),
            Transform(man.copy(), woman, remover=True)
        )
        self.wait()

        # Add it to king
        diff_copy = diff.copy()
        diff_copy.shift(king.get_end() - diff_copy.get_start())

        self.play(
            GrowArrow(king),
            FadeInFromPoint(king.label, axes.get_center()),
            frame.animate.reorient(-30, 71, 0).move_to([1.33, 0.68, 0.57]).set_height(2.83),
            run_time=2
        )
        self.play(TransformFromCopy(diff, diff_copy))
        self.play(
            LaggedStartMap(FadeIn, VGroup(ek, plus), shift=UP),
        )
        self.wait(2)
        self.play(
            FadeIn(fake_queen),
            FadeIn(fake_queen.label),
            Write(VGroup(approx, eq)),
            self.frame.animate.reorient(-15, 69, 0).move_to([1.33, 0.68, 0.57]).set_height(2.83).set_anim_args(run_time=6)
        )
        self.wait()

        # Correct it
        self.play(
            GrowArrow(queen),
            FadeInFromPoint(queen.label, queen.get_start()),
            VGroup(fake_queen, fake_queen.label).animate.set_opacity(0.2),
            frame.animate.reorient(-60, 72, 0).move_to([1.35, 0.7, 0.67]).set_height(3.29).set_anim_args(run_time=4)
        )
        self.play(
            frame.animate.reorient(-11, 70, 0).move_to([1.35, 0.7, 0.67]).set_height(3.29),
            FadeOut(fake_queen),
            FadeOut(fake_queen.label),
            run_time=10
        )

        # Show a few other examples
        word_pairs = [
            ("uncle", "aunt"),
            ("nephew", "niece"),
            ("father", "mother"),
            ("son", "daughter"),
        ]

        self.play(
            frame.animate.reorient(-28, 67, 0).move_to([1.57, 0.61, 0.54]).set_height(4),
            run_time=2
        )
        frame.clear_updaters()
        frame.add_updater(lambda f, dt: f.increment_theta(dt * 1 * DEGREES))
        last_group = VGroup(king, queen, king.label, queen.label, diff_copy)
        last_labels = VGroup(ek, eq)
        for word1, word2 in word_pairs:
            vect1 = self.get_labeled_vector(word1, color=colors[2])
            vect2 = self.get_labeled_vector(word2, color=colors[3])

            top_label1 = Text(f"E({word1})")
            top_label2 = Text(f"E({word2})")
            for label, prev_label, direction in [(top_label1, ek, RIGHT), (top_label2, eq, LEFT)]:
                label.set_fill(prev_label.get_fill_color())
                label.match_height(prev_label)
                label.move_to(prev_label, direction)
                label.fix_in_frame(True)

            new_diff = diff.copy()
            new_diff.shift(vect1.get_end() - new_diff.get_start())

            self.play(
                LaggedStart(
                    FadeOut(last_group),
                    GrowArrow(vect1),
                    FadeIn(vect1.label),
                    GrowArrow(vect2),
                    FadeIn(vect2.label),
                ),
                FadeOut(last_labels, UP),
                FadeIn(top_label1, UP),
                FadeIn(top_label2, UP),                
            )
            self.play(TransformFromCopy(diff, new_diff))
            self.wait(2)

            last_labels = VGroup(top_label1, top_label2)
            last_group = VGroup(vect1, vect2, vect1.label, vect2.label, new_diff)
        self.wait(4)

    def get_equation(self, word1, word2, word3, word4, colors=None):
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
        equation = self.get_equation(*self.words, colors=self.colors)
        equation.center().to_edge(UP)
        self.add(equation)

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
            GrowArrow(fudged_v4),
            FadeIn(fudged_v4.label),
        )
        self.play(FadeOut(rect))
        self.wait(6)


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

    def construct(self):
        # Add seed
        word = self.seed_word.lower()
        model = self.model
        seed_vect = self.get_labeled_vector(word, color=self.color)

        frame = self.frame
        frame.add_updater(lambda f, dt: f.increment_theta(dt * DEGREES))
        frame.move_to(seed_vect.get_end())
        frame.set_height(4)

        self.add(seed_vect)
        self.add(seed_vect.label)

        # Show neighbors
        neighbors = VGroup(*(
            self.get_labeled_vector(word, color=WHITE)
            for word in find_nearest_words(model, model[word], self.n_shown + 1)[1:]
        ))

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


class ShowNearestNeighborsToWikipediaCat(ShowNearestNeighbors):
    seed_word = "cat"
    color = YELLOW


class ShowNearestNeighborsToWikipediaNavy(ShowNearestNeighbors):
    seed_word = "navy"
    color = RED


class ShowNearestNeighborsToWikipediaJump(ShowNearestNeighbors):
    seed_word = "jump"
    color = BLUE
