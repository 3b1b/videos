from __future__ import annotations

from manim_imports_ext import *
from _2024.transformers.helpers import *
from _2024.transformers.embedding import break_into_words
from _2024.transformers.embedding import break_into_tokens
from _2024.transformers.embedding import get_piece_rectangles


class AttentionPatterns(InteractiveScene):
    def construct(self):
        # Add sentence
        phrase = " a fluffy blue creature roamed the verdant forest"
        phrase_mob = Text(phrase)
        phrase_mob.move_to(2 * UP)
        words = list(filter(lambda s: s.strip(), phrase.split(" ")))
        word2mob: Dict[str, VMobject] = {
            word: phrase_mob[" " + word][0]
            for word in words
        }
        word_mobs = VGroup(*word2mob.values())

        self.play(
            LaggedStartMap(FadeIn, word_mobs, shift=0.5 * UP, lag_ratio=0.25)
        )
        self.wait()

        # Create word rects
        word2rect: Dict[str, VMobject] = dict()
        for word in words:
            rect = SurroundingRectangle(word2mob[word])
            rect.set_height(phrase_mob.get_height() + SMALL_BUFF, stretch=True)
            rect.match_y(phrase_mob)
            rect.set_stroke(GREY, 2)
            rect.set_fill(GREY, 0.2)
            word2rect[word] = rect

        # Adjectives updating noun
        adjs = ["fluffy", "blue", "verdant"]
        nouns = ["creature", "forest"]
        others = ["a", "roamed", "the"]
        adj_mobs, noun_mobs, other_mobs = [
            VGroup(word2mob[substr] for substr in group)
            for group in [adjs, nouns, others]
        ]
        adj_rects, noun_rects, other_rects = [
            VGroup(word2rect[substr] for substr in group)
            for group in [adjs, nouns, others]
        ]
        adj_rects.set_submobject_colors_by_gradient(BLUE_C, BLUE_D, GREEN)
        noun_rects.set_color(GREY_BROWN).set_stroke(width=3)
        kw = dict()
        adj_arrows = VGroup(
            Arrow(
                adj_mobs[i].get_top(), noun_mobs[j].get_top(),
                path_arc=-150 * DEGREES, buff=0.1, stroke_color=GREY_B
            )
            for i, j in [(0, 0), (1, 0), (2, 1)]
        )

        self.play(
            LaggedStartMap(DrawBorderThenFill, adj_rects),
            Animation(adj_mobs),
        )
        self.wait()
        self.play(
            LaggedStartMap(DrawBorderThenFill, noun_rects),
            Animation(noun_mobs),
            LaggedStartMap(ShowCreation, adj_arrows, lag_ratio=0.2, run_time=1.5),
        )
        kw = dict(time_width=2, max_stroke_width=10, lag_ratio=0.2, path_arc=150 * DEGREES)
        self.play(
            ContextAnimation(noun_mobs[0], adj_mobs[:2], strengths=[1, 1], **kw),
            ContextAnimation(noun_mobs[1], adj_mobs[2:], strengths=[1], **kw),
        )
        self.wait()

        # Show embeddings
        all_rects = VGroup(*adj_rects, *noun_rects, *other_rects)
        all_rects.sort(lambda p: p[0])
        embeddings = VGroup(
            NumericEmbedding(length=10).set_width(0.5).next_to(rect, DOWN, buff=1.5)
            for rect in all_rects
        )
        emb_arrows = VGroup(
            Arrow(all_rects[0].get_bottom(), embeddings[0].get_top()).match_x(rect)
            for rect in all_rects
        )
        for index, vect in [(5, LEFT), (6, RIGHT)]:
            embeddings[index].shift(0.1 * vect)
            emb_arrows[index].shift(0.05 * vect)

        self.play(
            FadeIn(other_rects),
            Animation(word_mobs),
            LaggedStartMap(GrowArrow, emb_arrows),
            LaggedStartMap(FadeIn, embeddings, shift=0.5 * DOWN),
            FadeOut(adj_arrows)
        )
        self.wait()

        # Mention dimension of embedding
        frame = self.frame
        brace = Brace(embeddings[0], LEFT, buff=SMALL_BUFF)
        dim_value = Integer(12288)
        dim_value.next_to(brace, LEFT)
        dim_value.set_color(YELLOW)

        self.play(
            GrowFromCenter(brace),
            CountInFrom(dim_value, 0),
            frame.animate.move_to(LEFT)
        )
        self.wait()

        # Ingest meaning and and position
        images = Group(
            ImageMobject(f"Dalle3_{word}").set_height(1.1).next_to(word2rect[word], UP)
            for word in ["fluffy", "blue", "creature", "verdant", "forest"]
        )
        image_vects = VGroup(embeddings[i] for i in [1, 2, 3, 6, 7])

        self.play(
            LaggedStartMap(FadeIn, images, scale=2, lag_ratio=0.05)
        )
        self.play(
            LaggedStart(
                (self.bake_mobject_into_vector_entries(image, vect, group_type=Group)
                for image, vect in zip(images, image_vects)),
                group_type=Group,
                lag_ratio=0.2,
                run_time=4,
                remover=True
            ),
        )
        self.wait()
        self.add(embeddings, images)

        # Show positions
        pos_labels = VGroup(
            Integer(n, font_size=36).next_to(rect, DOWN, buff=0.1)
            for n, rect in enumerate(all_rects, start=1)
        )
        pos_labels.set_color(TEAL)

        self.play(
            LaggedStart(
                (arrow.animate.scale(0.7, about_edge=DOWN)
                for arrow in emb_arrows),
                lag_ratio=0.1,
            ),
            LaggedStartMap(FadeIn, pos_labels, shift=0.25 * DOWN, lag_ratio=0.1)
        )
        self.play(
            LaggedStart(
                (self.bake_mobject_into_vector_entries(pos, vect)
                for pos, vect in zip(pos_labels, embeddings)),
                lag_ratio=0.2,
                run_time=4,
                remover=True
            ),
        )
        self.wait()

        # Collapse vectors
        template = Tex(R"\vec{\textbf{E}}_{0}")
        template[0].scale(1.5, about_edge=DOWN)
        dec = template.make_number_changeable(0)
        emb_syms = VGroup()
        for n, rect in enumerate(all_rects, start=1):
            dec.set_value(n)
            sym = template.copy()
            sym.next_to(rect, DOWN, buff=0.75)
            sym.set_color(GREY_A)
            emb_syms.add(sym)
        for subgroup in [emb_syms[:4], emb_syms[4:]]:
            subgroup.arrange_to_fit_width(subgroup.get_width())

        emb_arrows.target = emb_arrows.generate_target()

        for rect, arrow, sym in zip(all_rects, emb_arrows.target, emb_syms):
            x_min = rect.get_x(LEFT)
            x_max = rect.get_x(RIGHT)
            low_point = sym[0].get_top()
            if x_min < low_point[0] < x_max:
                top_point = np.array([low_point[0], rect.get_y(DOWN), 0])
            else:
                top_point = rect.get_bottom()
            arrow.become(Arrow(top_point, low_point, buff=SMALL_BUFF))

        all_brackets = VGroup(emb.get_brackets() for emb in embeddings)
        for brackets in all_brackets:
            brackets.target = brackets.generate_target()
            brackets.target.stretch(0, 1, about_edge=UP)
            brackets.target.set_fill(opacity=0)

        ghost_syms = emb_syms.copy()
        ghost_syms.set_opacity(0)

        self.play(
            frame.animate.set_x(0).set_anim_args(run_time=2),
            LaggedStart(
                (AnimationGroup(
                    LaggedStart(
                        (FadeTransform(entry, sym)
                        for entry in embedding.get_columns()[0]),
                        lag_ratio=0.01,
                        group_type=Group
                    ),
                    MoveToTarget(brackets),
                    group_type=Group,
                )
                for sym, embedding, brackets in zip(ghost_syms, embeddings, all_brackets)),
                group_type=Group
            ),
            LaggedStartMap(FadeIn, emb_syms, shift=UP),
            brace.animate.stretch(0.25, 1, about_edge=UP).set_opacity(0),
            FadeOut(dim_value, 0.25 * UP),
            MoveToTarget(emb_arrows, lag_ratio=0.1, run_time=2),
            LaggedStartMap(FadeOut, pos_labels, shift=UP),
        )
        emb_arrows.refresh_bounding_box(recurse_down=True)  # Why?
        self.clear()
        self.add(emb_arrows, all_rects, word_mobs, images, emb_syms)
        self.wait()

        # Preview desired updates
        emb_sym_primes = VGroup(
            sym.copy().add(Tex("'").move_to(sym.get_corner(UR) + 0.05 * DL))
            for sym in emb_syms
        )
        emb_sym_primes.shift(2 * DOWN)
        emb_sym_primes.set_color(TEAL)

        full_connections = VGroup()
        for i, sym1 in enumerate(emb_syms, start=1):
            for j, sym2 in enumerate(emb_sym_primes, start=1):
                line = Line(sym1.get_bottom(), sym2.get_top(), buff=SMALL_BUFF)
                line.set_stroke(GREY_B, width=random.random()**2, opacity=random.random()**0.25)
                if (i, j) in [(2, 4), (3, 4), (4, 4), (7, 8), (8, 8)]:
                    line.set_stroke(WHITE, width=2 + random.random(), opacity=1)
                full_connections.add(line)

        blue_fluff = ImageMobject("BlueFluff")
        verdant_forest = ImageMobject("VerdantForest")
        for n, image in [(3, blue_fluff), (7, verdant_forest)]:
            image.match_height(images)
            image.scale(1.2)
            image.next_to(emb_sym_primes[n], DOWN, buff=MED_SMALL_BUFF)

        self.play(
            ShowCreation(full_connections, lag_ratio=0.01, run_time=2),
            LaggedStart(
                (TransformFromCopy(sym1, sym2)
                for sym1, sym2 in zip(emb_syms, emb_sym_primes)),
                lag_ratio=0.05,
            ),
        )
        self.wait()
        self.play(LaggedStart(
            LaggedStart(
                (FadeTransform(im.copy(), blue_fluff, remover=True)
                for im in images[:3]),
                lag_ratio=0.02,
                group_type=Group
            ),
            LaggedStart(
                (FadeTransform(im.copy(), verdant_forest, remover=True)
                for im in images[3:]),
                lag_ratio=0.02,
                group_type=Group
            ),
            lag_ratio=0.5,
            run_time=2
        ))
        self.add(blue_fluff, verdant_forest)
        self.wait()

        # Show black box that matrix multiples can be added to
        in_arrows = VGroup(
            Vector(0.25 * DOWN, max_width_to_length_ratio=12.0).next_to(sym, DOWN, SMALL_BUFF)
            for sym in emb_syms
        )
        box = Rectangle(15.0, 3.0)
        box.set_fill(GREY_E, 1)
        box.set_stroke(WHITE, 1)
        box.next_to(in_arrows, DOWN, SMALL_BUFF)
        out_arrows = in_arrows.copy()
        out_arrows.next_to(box, DOWN)

        self.play(
            FadeIn(box, 0.25 * DOWN),
            LaggedStartMap(FadeIn, in_arrows, shift=0.25 * DOWN, lag_ratio=0.025),
            LaggedStartMap(FadeIn, out_arrows, shift=0.25 * DOWN, lag_ratio=0.025),
            FadeOut(full_connections),
            emb_sym_primes.animate.next_to(out_arrows, DOWN, SMALL_BUFF),
            MaintainPositionRelativeTo(blue_fluff, emb_sym_primes),
            MaintainPositionRelativeTo(verdant_forest, emb_sym_primes),
            frame.animate.set_height(10).move_to(4 * UP, UP),
        )
        self.wait()

        # Clear the board
        self.play(
            frame.animate.set_height(8).move_to(2 * UP).set_anim_args(run_time=1.5),
            LaggedStartMap(FadeOut, Group(
                *images, in_arrows, box, out_arrows, emb_sym_primes,
                blue_fluff, verdant_forest,
            ), lag_ratio=0.1)
        )

        # Ask questions
        word_groups = VGroup(VGroup(*pair) for pair in zip(all_rects, word_mobs))
        for group in word_groups:
            group.save_state()
        q_bubble = SpeechBubble("Any adjectives\nin front of me?")
        q_bubble.move_tip_to(word2rect["creature"].get_top())

        a_bubbles = SpeechBubble("I am!", direction=RIGHT).replicate(2)
        a_bubbles[1].flip()
        a_bubbles[0].move_tip_to(word2rect["fluffy"].get_top())
        a_bubbles[1].move_tip_to(word2rect["blue"].get_top())

        self.play(
            FadeIn(q_bubble),
            word_groups[:3].animate.fade(0.75),
            word_groups[4:].animate.fade(0.75),
        )
        self.wait()
        self.play(LaggedStart(
            Restore(word_groups[1]),
            Restore(word_groups[2]),
            *map(Write, a_bubbles),
            lag_ratio=0.5
        ))
        self.wait()

        # Associate questions with vectors
        a_bubbles.save_state()
        q_arrows = VGroup(
            Vector(0.75 * DOWN).next_to(sym, DOWN, SMALL_BUFF)
            for sym in emb_syms
        )
        q_vects = VGroup(
            NumericEmbedding(length=7).set_height(2).next_to(arrow, DOWN)
            for arrow in q_arrows
        )
        question = q_bubble.content


        index = words.index("creature")
        q_vect = q_vects[index]
        q_arrow = q_arrows[index]
        self.play(LaggedStart(
            FadeOut(q_bubble.body, DOWN),
            question.animate.scale(0.75).next_to(q_vect, RIGHT),
            FadeIn(q_vect, DOWN),
            GrowArrow(q_arrow),
            frame.animate.move_to(ORIGIN),
            a_bubbles.animate.fade(0.5),
        ))
        self.play(
            self.bake_mobject_into_vector_entries(question, q_vect)
        )
        self.wait()

        # Label query vector
        brace = Brace(q_vect, LEFT, SMALL_BUFF)
        query_word = Text("Query")
        query_word.set_color(YELLOW)
        query_word.next_to(brace, LEFT, SMALL_BUFF)
        dim_text = Text("128-dimensional", font_size=36)
        dim_text.set_color(YELLOW)
        dim_text.next_to(brace, LEFT, SMALL_BUFF)
        dim_text.set_y(query_word.get_y(DOWN))

        self.play(
            GrowFromCenter(brace),
            FadeIn(query_word, 0.25 * LEFT),
        )
        self.wait()
        self.play(
            query_word.animate.next_to(dim_text, UP, SMALL_BUFF),
            FadeIn(dim_text, 0.1 * DOWN),
        )
        self.wait()

        # Show individual matrix product
        e_vect = NumericEmbedding(length=12)
        e_vect.match_width(q_vect)
        e_vect.next_to(q_vect, DR, buff=1.5)
        matrix = WeightMatrix(shape=(7, 12))
        matrix.match_height(q_vect)
        matrix.next_to(e_vect, LEFT)
        e_label_copy = emb_syms[index].copy()
        e_label_copy.next_to(e_vect, UP)
        q_vect.save_state()
        ghost_q_vect = NumericEmbedding(length=7).match_height(q_vect)
        ghost_q_vect.get_columns().set_opacity(0)
        ghost_q_vect.get_brackets().space_out_submobjects(1.75)
        ghost_q_vect.next_to(e_vect, RIGHT, buff=0.7)

        mat_brace = Brace(matrix, UP)
        mat_label = Tex("W_Q")
        mat_label.next_to(mat_brace, UP, SMALL_BUFF)
        mat_label.set_color(YELLOW)

        self.play(
            frame.animate.set_height(11).move_to(all_rects, UP).shift(0.35 * UP),
            FadeOut(a_bubbles),
            FadeInFromPoint(e_vect, emb_syms[index].get_center()),
            FadeInFromPoint(matrix, q_arrow.get_center()),
            TransformFromCopy(emb_syms[index], e_label_copy),
            FadeOut(q_vect),
            TransformFromCopy(q_vect, ghost_q_vect),
            MaintainPositionRelativeTo(question, q_vect),
        )
        self.play(
            GrowFromCenter(mat_brace),
            FadeIn(mat_label, 0.1 * UP),
        )
        self.remove(ghost_q_vect)
        eq, rhs = show_matrix_vector_product(self, matrix, e_vect)

        new_q_vect = rhs.deepcopy()
        new_q_vect.move_to(q_vect, LEFT)

        self.play(
            TransformFromCopy(rhs, new_q_vect, path_arc=PI / 2),
            question.animate.next_to(new_q_vect, RIGHT)
        )
        self.wait()

        # Collapse query vector
        q_sym_template = Tex(R"\vec{\textbf{Q}}_0", font_size=48)
        q_sym_template[0].scale(1.5, about_edge=DOWN)
        q_sym_template.set_color(YELLOW)
        subscript = q_sym_template.make_number_changeable(0)
        q_syms = VGroup()
        for n, arrow in enumerate(q_arrows, start=1):
            subscript.set_value(n)
            sym = q_sym_template.copy()
            sym.next_to(arrow, DOWN, SMALL_BUFF)
            q_syms.add(sym)

        mat_label2 = mat_label.copy()

        q_sym = q_syms[index]
        low_q_sym = q_sym.copy()
        low_q_sym.next_to(rhs, UP)

        self.play(LaggedStart(
            LaggedStart(
                (FadeTransform(entry, q_sym, remover=True)
                for entry in new_q_vect.get_columns()[0]),
                lag_ratio=0.01,
                group_type=Group,
            ),
            new_q_vect.get_brackets().animate.stretch(0, 1, about_edge=UP).set_opacity(0),
            FadeOutToPoint(query_word, q_sym.get_center()),
            FadeOutToPoint(dim_text, q_sym.get_center()),
            FadeOut(brace),
            question.animate.next_to(q_sym, DOWN),
            FadeIn(low_q_sym, UP),
            lag_ratio=0.1,
        ))
        self.remove(new_q_vect)
        self.add(q_sym)
        self.play(
            mat_label2.animate.scale(0.9).next_to(q_arrow, RIGHT, buff=0.15),
        )
        self.wait()

        # E to Q rects
        e_rects = VGroup(map(SurroundingRectangle, [emb_syms[index], e_vect]))
        q_rects = VGroup(map(SurroundingRectangle, [q_sym, rhs]))
        e_rects.set_stroke(TEAL, 3)
        q_rects.set_stroke(YELLOW, 3)
        self.play(ShowCreation(e_rects, lag_ratio=0.2))
        self.wait()
        self.play(Transform(e_rects, q_rects))
        self.wait()
        self.play(FadeOut(e_rects))

        # Add other query vectors
        remaining_q_arrows = VGroup(*q_arrows[:index], *q_arrows[index + 1:])
        remaining_q_syms = VGroup(*q_syms[:index], *q_syms[index + 1:])
        wq_syms = VGroup(
            Tex(R"W_Q", font_size=30).next_to(arrow, RIGHT, buff=0.1)
            for arrow in q_arrows
        )
        wq_syms.set_color(YELLOW)
        subscripts = VGroup(e_label_copy[-1], low_q_sym[-1][0])
        for subscript in subscripts:
            i_sym = Tex("i")
            i_sym.replace(subscript)
            i_sym.scale(0.75)
            i_sym.match_style(subscript)
            subscript.target = i_sym

        self.play(
            LaggedStartMap(GrowArrow, remaining_q_arrows),
            LaggedStartMap(FadeIn, remaining_q_syms, shift=0.1 * DOWN),
            ReplacementTransform(VGroup(mat_label2), wq_syms, lag_ratio=0.01, run_time=2),
            question.animate.shift(0.25 * DOWN),
            *map(Restore, word_groups),
            *map(MoveToTarget, subscripts),
        )
        self.wait()

        # Emphasize model weights
        self.play(
            LaggedStartMap(FlashAround, matrix.get_entries(), lag_ratio=1e-2),
            RandomizeMatrixEntries(matrix),
        )
        data_modifying_matrix(self, matrix, word_shape=(3, 8))
        self.wait()
        self.play(
            LaggedStartMap(FadeOut, VGroup(
                matrix, mat_brace, mat_label,
                e_vect, e_label_copy, eq, rhs,
                low_q_sym
            ), shift=0.2 * DR)
        )
        self.wait()

        # Move question
        noun_q_syms = VGroup(q_syms[words.index(word)] for word in ["creature", "forest"])

        self.play(
            question.animate.shift(0.25 * DOWN).match_x(noun_q_syms)
        )

        noun_q_lines = VGroup(
            Line(question.get_corner(v), sym.get_corner(-v) + 0.25 * v)
            for sym, v in zip(noun_q_syms, [UL, UR])
        )
        noun_q_lines.set_stroke(GREY, 1)
        self.play(ShowCreation(noun_q_lines, lag_ratio=0))
        self.wait()

        # Set up keys
        key_word_groups = word_groups.copy()
        key_word_groups.arrange(DOWN, buff=0.75, aligned_edge=RIGHT)
        key_word_groups.next_to(q_syms, DL, buff=LARGE_BUFF)
        key_word_groups.shift(3.0 * LEFT)
        key_emb_syms = emb_syms.copy()

        k_sym_template = Tex(R"\vec{\textbf{K}}_0", font_size=48)
        k_sym_template[0].scale(1.5, about_edge=DOWN)
        k_sym_template.set_color(TEAL)
        subscript = k_sym_template.make_number_changeable(0)

        k_syms = VGroup()
        key_emb_arrows = VGroup()
        wk_arrows = VGroup()
        wk_syms = VGroup()
        for group, emb_sym, n in zip(key_word_groups, key_emb_syms, it.count(1)):
            emb_arrow = Vector(0.5 * RIGHT)
            emb_arrow.next_to(group, RIGHT, SMALL_BUFF)
            emb_sym.next_to(emb_arrow, RIGHT, SMALL_BUFF)
            wk_arrow = Vector(0.75 * RIGHT)
            wk_arrow.next_to(emb_sym, RIGHT)
            wk_sym = Tex("W_k", font_size=30)
            wk_sym.set_fill(TEAL, border_width=1)
            wk_sym.next_to(wk_arrow, UP)
            subscript.set_value(n)
            k_sym = k_sym_template.copy()
            k_sym.next_to(wk_arrow, RIGHT, buff=MED_SMALL_BUFF)

            key_emb_arrows.add(emb_arrow)
            wk_arrows.add(wk_arrow)
            wk_syms.add(wk_sym)
            k_syms.add(k_sym)

        self.remove(question, noun_q_lines)
        self.play(
            frame.animate.move_to(2.5 * LEFT + 2.75 * DOWN),
            TransformFromCopy(word_groups, key_word_groups),
            TransformFromCopy(emb_arrows, key_emb_arrows),
            TransformFromCopy(emb_syms, key_emb_syms),
            run_time=2,
        )
        self.play(
            LaggedStartMap(GrowArrow, wk_arrows),
            LaggedStartMap(FadeIn, wk_syms, shift=0.1 * UP),
        )
        self.play(LaggedStart(
            (TransformFromCopy(e_sym, k_sym)
            for e_sym, k_sym in zip(key_emb_syms, k_syms)),
            lag_ratio=0.05,
        ))
        self.wait()

        # Isolate examples
        fade_rects = VGroup(
            BackgroundRectangle(VGroup(key_word_groups[0], wk_syms[0], k_syms[0])),
            BackgroundRectangle(VGroup(key_word_groups[3:], wk_syms[3:], k_syms[3:])),
            BackgroundRectangle(wq_syms[2]),
            BackgroundRectangle(VGroup(word_groups[:3], q_syms[:3])),
            BackgroundRectangle(VGroup(word_groups[4:], q_syms[4:])),
        )
        fade_rects.set_fill(BLACK, 0.75)
        fade_rects.set_stroke(BLACK, 3, 1)
        q_bubble = SpeechBubble("Any adjectives\nin front of me?")
        q_bubble.flip(RIGHT)
        q_bubble.next_to(q_syms[3][-1], DOWN, SMALL_BUFF, LEFT)
        a_bubbles = SpeechBubble("I'm an adjective!\nI'm there!").replicate(2)
        a_bubbles[0].pin_to(k_syms[1])
        a_bubbles[1].pin_to(k_syms[2])
        a_bubbles[1].flip(RIGHT, about_edge=DOWN)
        a_bubbles[1].shift(0.5 * DOWN)

        self.add(fade_rects, word_groups[3])
        self.play(FadeIn(fade_rects))
        self.play(FadeIn(q_bubble, lag_ratio=0.1))
        self.play(FadeIn(a_bubbles, lag_ratio=0.05))
        self.wait()

        # Show example key matrix
        matrix = WeightMatrix(shape=(7, 12))
        matrix.set_width(5)
        matrix.next_to(k_syms, UP, buff=2.0, aligned_edge=RIGHT)
        mat_rect = SurroundingRectangle(matrix, buff=MED_SMALL_BUFF)
        lil_rect = SurroundingRectangle(wk_syms[1])
        lines = VGroup(
            Line(lil_rect.get_corner(v + UP), mat_rect.get_corner(v + DOWN))
            for v in [LEFT, RIGHT]
        )
        VGroup(mat_rect, lil_rect, *lines).set_stroke(GREY_A, 1)

        self.play(ShowCreation(lil_rect))
        self.play(
            ShowCreation(lines, lag_ratio=0),
            TransformFromCopy(lil_rect, mat_rect),
            FadeInFromPoint(matrix, lil_rect.get_center()),
        )
        self.wait()
        data_modifying_matrix(self, matrix, word_shape=(3, 8))
        self.play(
            LaggedStartMap(FadeOut, VGroup(matrix, mat_rect, lines, lil_rect), run_time=1)
        )
        self.play(
            LaggedStartMap(FadeOut, VGroup(q_bubble, *a_bubbles), lag_ratio=0.25)
        )
        self.wait()

        # Draw grid
        emb_arrows.refresh_bounding_box(recurse_down=True)
        q_groups = VGroup(
            VGroup(group[i] for group in [
                emb_arrows, emb_syms, wq_syms, q_arrows, q_syms
            ])
            for i in range(len(emb_arrows))
        )
        q_groups.target = q_groups.generate_target()
        q_groups.target.arrange_to_fit_width(12, about_edge=LEFT)
        q_groups.target.shift(0.25 * DOWN)

        word_groups.target = word_groups.generate_target()
        for word_group, q_group in zip(word_groups.target, q_groups.target):
            word_group.scale(0.7)
            word_group.next_to(q_group[0], UP, SMALL_BUFF)

        h_lines = VGroup()
        v_buff = 0.5 * (key_word_groups[0].get_y(DOWN) - key_word_groups[1].get_y(UP))
        for kwg in key_word_groups:
            h_line = Line(LEFT, RIGHT).set_width(20)
            h_line.next_to(kwg, UP, buff=v_buff)
            h_line.align_to(key_word_groups, LEFT)
            h_lines.add(h_line)

        v_lines = VGroup()
        h_buff = 0.5
        for q_group in q_groups.target:
            v_line = Line(UP, DOWN).set_height(14)
            v_line.next_to(q_group, LEFT, buff=h_buff, aligned_edge=UP)
            v_lines.add(v_line)
        v_lines.add(v_lines[-1].copy().next_to(q_groups.target, RIGHT, 0.5, UP))

        grid_lines = VGroup(*h_lines, *v_lines)
        grid_lines.set_stroke(GREY_A, 1)

        self.play(
            frame.animate.set_height(15, about_edge=UP).set_x(-2).set_anim_args(run_time=3),
            MoveToTarget(q_groups),
            MoveToTarget(word_groups),
            ShowCreation(h_lines, lag_ratio=0.2),
            ShowCreation(v_lines, lag_ratio=0.2),
            FadeOut(fade_rects),
        )

        # Take all dot products
        dot_prods = VGroup()
        for k_sym in k_syms:
            for q_sym in q_syms:
                square_center = np.array([q_sym.get_x(), k_sym.get_y(), 0])
                dot = Tex(R".", font_size=72)
                dot.move_to(square_center)
                dot.set_fill(opacity=0)
                dot_prod = VGroup(k_sym.copy(), dot, q_sym.copy())
                dot_prod.target = dot_prod.generate_target()
                dot_prod.target.arrange(RIGHT, buff=0.15)
                dot_prod.target.scale(0.65)
                dot_prod.target.move_to(square_center)
                dot_prod.target.set_fill(opacity=1)
                dot_prods.add(dot_prod)

        self.play(
            LaggedStartMap(MoveToTarget, dot_prods, lag_ratio=0.025, run_time=4)
        )
        self.wait()

        # Show grid of dots
        dots = VGroup(
            VGroup(Dot().match_x(q_sym).match_y(k_sym) for q_sym in q_syms)
            for k_sym in k_syms
        )
        for n, row in enumerate(dots, start=1):
            for k, dot in enumerate(row, start=1):
                dot.set_fill(GREY_C, 0.8)
                dot.set_width(random.random())
                dot.target = dot.generate_target()
                dot.target.set_width(0.1 + 0.2 * random.random())
                if (n, k) in [(2, 4), (3, 4), (7, 8)]:
                    dot.target.set_width(0.8 + 0.2 * random.random())
        flat_dots = VGroup(*it.chain(*dots))

        self.play(
            dot_prods.animate.set_fill(opacity=0.75),
            LaggedStartMap(GrowFromCenter, flat_dots)
        )
        self.wait()
        self.play(LaggedStartMap(MoveToTarget, flat_dots, lag_ratio=0.01))
        self.wait()

        # Resize to reflect true pattern
        k_groups = VGroup(
            VGroup(group[i] for group in [
                key_word_groups, key_emb_arrows,
                key_emb_syms, wk_syms, wk_arrows, k_syms
            ])
            for i in range(len(emb_arrows))
        )
        for q_group, word_group in zip(q_groups, word_groups):
            q_group.add_to_back(word_group)
        self.add(k_groups, q_groups, Point())

        k_fade_rects = VGroup(map(BackgroundRectangle, k_groups))
        q_fade_rects = VGroup(map(BackgroundRectangle, q_groups))
        for rect in (*k_fade_rects, *q_fade_rects):
            rect.scale(1.05)
            rect.set_fill(BLACK, 0.8)

        self.play(
            frame.animate.move_to([-4.33, -2.4, 0.0]).set_height(9.52),
            FadeIn(k_fade_rects[:1]),
            FadeIn(k_fade_rects[3:]),
            FadeIn(q_fade_rects[:3]),
            FadeIn(q_fade_rects[4:]),
            run_time=2
        )
        self.wait()

        k_rects = VGroup(map(SurroundingRectangle, k_groups[1:3]))
        k_rects.set_stroke(TEAL, 2)
        q_rects = VGroup(SurroundingRectangle(q_groups[3]))
        q_rects.set_stroke(YELLOW, 2)

        self.play(
            ShowCreation(k_rects, lag_ratio=0.5, run_time=2),
            LaggedStartMap(
                FlashAround, k_groups[1:3],
                color=TEAL,
                time_width=2,
                lag_ratio=0.25,
                run_time=3
            ),
        )
        self.wait()
        self.play(TransformFromCopy(k_rects, q_rects))
        self.wait()

        # Show numerical dot product
        high_dot_prods = VGroup(dot_prods[8 + 3], dot_prods[2 * 8 + 3])
        dots_to_grow = VGroup(dots[1][3], dots[2][3])
        numerical_dot_prods = VGroup(
            VGroup(
                DecimalNumber(
                    np.random.uniform(-100, 10),
                    include_sign=True,
                    font_size=42,
                    num_decimal_places=1,
                    edge_to_fix=ORIGIN,
                ).move_to(dot)
                for dot in row
            )
            for row in dots
        )
        for n, row in enumerate(numerical_dot_prods):
            row[n].set_value(5 * random.random())  # Add some self relevance
        flat_numerical_dot_prods = VGroup(*it.chain(*numerical_dot_prods))
        for ndp in flat_numerical_dot_prods:
            ndp.set_fill(interpolate_color(RED_E, GREY_C, random.random()))
        high_numerical_dot_prods = VGroup(
            numerical_dot_prods[1][3],
            numerical_dot_prods[2][3],
            numerical_dot_prods[6][7],
        )
        for hdp in high_numerical_dot_prods:
            hdp.set_value(92 + 2 * random.random())
            hdp.set_color(WHITE)
        low_numerical_dot_prod = numerical_dot_prods[5][3]
        low_numerical_dot_prod.set_value(-31.4)
        low_numerical_dot_prod.set_fill(RED_D)

        self.play(
            *(dtg.animate.scale(1.25) for dtg in dots_to_grow),
            *(CountInFrom(ndp, run_time=1) for ndp in high_numerical_dot_prods[:2]),
            *(VFadeIn(ndp) for ndp in high_numerical_dot_prods[:2]),
            *(FadeOut(dot_prod, run_time=0.5) for dot_prod in dot_prods),
        )
        self.wait()

        # Show "attends to"
        att_arrow = Arrow(k_rects.get_top(), q_rects.get_left(), path_arc=-90 * DEGREES)
        att_words = TexText("``Attends to''", font_size=72)
        att_words.next_to(att_arrow.pfp(0.4), UL)

        self.play(
            ShowCreation(att_arrow),
            Write(att_words),
        )
        self.wait()
        self.play(FadeOut(att_words), FadeOut(att_arrow))

        # Contrast with "the" and "creature"
        self.play(
            frame.animate.move_to([-2.79, -3.66, 0.0]).set_height(12.29),
            *(k_rect.animate.surround(k_groups[5]) for k_rect in k_rects),
            FadeIn(k_fade_rects[1:3]),
            FadeOut(k_fade_rects[5]),
            run_time=2,
        )
        self.play(
            CountInFrom(low_numerical_dot_prod),
            VFadeIn(low_numerical_dot_prod),
            FadeOut(dots[5][3]),
        )
        self.wait()

        # Zoom out on full grid
        self.play(
            frame.animate.move_to([-1.5, -4.8, 0.0]).set_height(15).set_anim_args(run_time=3),
            LaggedStart(
                FadeOut(k_rects),
                FadeOut(q_rects),
                FadeOut(k_fade_rects[:5]),
                FadeOut(k_fade_rects[6:]),
                FadeOut(q_fade_rects[:3]),
                FadeOut(q_fade_rects[4:]),
                FadeOut(dots),
                LaggedStartMap(FadeIn, numerical_dot_prods),
                Animation(high_numerical_dot_prods.copy(), remover=True),
                Animation(low_numerical_dot_prod.copy(), remover=True),
            )
        )
        self.wait()

        # Focus on one column
        ndp_columns = VGroup(
            VGroup(row[i] for row in numerical_dot_prods)
            for i in range(len(numerical_dot_prods[0]))
        )
        col_rect = SurroundingRectangle(ndp_columns[3], buff=0.25)
        col_rect.set_stroke(YELLOW, 2)
        weight_words = Text("We want these to\nact like weights", font_size=96)
        weight_words.set_backstroke(BLACK, 8)
        weight_words.next_to(col_rect, RIGHT, buff=MED_LARGE_BUFF)
        weight_words.match_y(h_lines[2])

        index = words.index("creature")
        self.play(
            ShowCreation(col_rect),
            grid_lines.animate.set_stroke(opacity=0.5),
            ndp_columns[:index].animate.set_opacity(0.35),
            ndp_columns[index + 1:].animate.set_opacity(0.35),
            FadeIn(weight_words, lag_ratio=0.1)
        )
        self.wait()

        # Show softmax of each columns
        self.set_floor_plane("xz")
        col_arrays = [np.array([num.get_value() for num in col]) for col in ndp_columns]
        softmax_arrays = list(map(softmax, col_arrays))
        softmax_cols = VGroup(
            VGroup(DecimalNumber(v) for v in softmax_array)
            for softmax_array in softmax_arrays
        )
        sm_arrows = VGroup()
        sm_labels = VGroup()
        sm_rects = VGroup()
        for sm_col, col in zip(softmax_cols, ndp_columns):
            for sm_val, val in zip(sm_col, col):
                sm_val.move_to(val)
            sm_col.save_state()
            sm_col.shift(6 * OUT)
            sm_rect = SurroundingRectangle(sm_col)
            sm_rect.match_style(col_rect)
            VGroup(sm_col, sm_rect).rotate(30 * DEGREES, DOWN)
            arrow = Arrow(col, sm_col.get_center() + SMALL_BUFF * RIGHT + IN)
            label = Text("softmax", font_size=72)
            label.set_backstroke(BLACK, 5)
            label.rotate(90 * DEGREES, DOWN)
            label.next_to(arrow, UP)
            sm_arrows.add(arrow)
            sm_labels.add(label)
            sm_rects.add(sm_rect)

        index = words.index("creature")
        self.play(
            frame.animate.reorient(-47, -7, 0, (-2.48, -5.84, -1.09), 20),
            GrowArrow(sm_arrows[index], time_span=(1, 2)),
            FadeIn(sm_labels[index], lag_ratio=0.1, time_span=(1, 2)),
            TransformFromCopy(ndp_columns[index], softmax_cols[index], time_span=(1.5, 3)),
            TransformFromCopy(col_rect, sm_rects[index], time_span=(1.5, 3)),
            FadeOut(weight_words),
            run_time=3
        )
        self.wait()

        remaining_indices = [*range(index), *range(index + 1, len(ndp_columns))]
        last_index = index
        for index in remaining_indices:
            self.play(
                ndp_columns[last_index].animate.set_opacity(0.35),
                ndp_columns[index].animate.set_opacity(1),
                col_rect.animate.move_to(ndp_columns[index]),
                softmax_cols[last_index].animate.set_opacity(0.25),
                *map(FadeOut, [sm_rects[last_index], sm_arrows[last_index], sm_labels[last_index]]),
            )
            self.play(
                GrowArrow(sm_arrows[index]),
                FadeIn(sm_labels[index], lag_ratio=0.1),
                TransformFromCopy(ndp_columns[index], softmax_cols[index]),
                TransformFromCopy(col_rect, sm_rects[index]),
            )
            last_index = index

        self.play(
            FadeOut(col_rect),
            *map(FadeOut, [sm_rects[last_index], sm_arrows[last_index], sm_labels[last_index]]),
        )
        self.wait()
        self.play(
            frame.animate.reorient(0, 0, 0, (-2.64, -4.8, 0.0), 14.54),
            LaggedStartMap(Restore, softmax_cols, lag_ratio=0.1),
            FadeOut(ndp_columns, time_span=(0, 1.5)),
            run_time=3,
        )
        self.wait()

        # Label attention pattern
        for n, row in enumerate(dots):
            if n not in [3, 7]:
                row[n].set_width(0.7 + 0.2 * random.random())
        dots[1][3].set_width(0.6 + 0.1 * random.random())
        dots[2][3].set_width(0.6 + 0.1 * random.random())
        dots[6][7].set_width(0.9 + 0.1 * random.random())

        pattern_words = Text("Attention\nPattern", font_size=120)
        pattern_words.move_to(grid_lines, UL).shift(LEFT)

        self.play(
            FadeOut(softmax_cols, lag_ratio=0.001),
            FadeIn(dots, lag_ratio=0.001),
            Write(pattern_words),
            run_time=2
        )
        self.wait()

        # Preview masking
        masked_dots = VGroup()
        for n, row in enumerate(dots):
            masked_dots.add(*row[:n])
        mask_rects = VGroup()
        for dot in masked_dots:
            mask_rect = Square(0.5)
            mask_rect.set_stroke(RED, 2)
            mask_rect.move_to(dot)
            mask_rects.add(mask_rect)

        lag_ratio=1.0 / len(mask_rects)
        self.play(ShowCreation(mask_rects, lag_ratio=lag_ratio))
        self.play(
            LaggedStart(
                (dot.animate.scale(0) for dot in masked_dots),
                lag_ratio=lag_ratio
            )
        )
        self.play(
            FadeOut(mask_rects, lag_ratio=lag_ratio)
        )
        self.wait()

        # Set aside keys and queries
        pattern = VGroup(grid_lines, dots)
        for group in q_groups:
            group.sort(lambda p: -p[1])
            group.target = group.generate_target()
            m3 = len(group) - 3
            group.target[m3:].scale(0, about_edge=DOWN)
            group.target[:m3].move_to(group, DOWN)

        self.play(
            frame.animate.move_to((-2.09, -5.59, 0.0)).set_height(12.95).set_anim_args(run_time=3),
            LaggedStartMap(MoveToTarget, q_groups),
            FadeOut(pattern_words),
            v_lines.animate.stretch(0.95, 1, about_edge=DOWN),
        )
        self.play(
            LaggedStartMap(FadeOut, k_syms, shift=0.5 * DOWN, lag_ratio=0.1),
            LaggedStartMap(FadeOut, wk_syms, shift=0.5 * DOWN, lag_ratio=0.1),
        )
        self.wait()

        # Add values
        value_color = RED
        big_wv_sym = Tex(R"W_V", font_size=90)
        big_wv_sym.set_color(value_color)
        big_wv_sym.next_to(h_lines, UP, MED_LARGE_BUFF, LEFT)
        wv_word = Text("Value matrix", font_size=90)
        wv_word.next_to(big_wv_sym, UP, MED_LARGE_BUFF)
        wv_word.set_color(value_color)

        wv_arrows = wk_arrows
        v_sym_template = Tex(R"\vec{\textbf{V}}_{0}")
        v_sym_template[0].scale(1.5, about_edge=DOWN)
        v_sym_template.set_fill(value_color, border_width=1)
        subscript = v_sym_template.make_number_changeable("0")

        wv_syms = VGroup()
        v_syms = VGroup()
        for n, arrow in enumerate(wv_arrows, start=1):
            wv_sym = Tex("W_V", font_size=36)
            wv_sym.set_fill(value_color, border_width=1)
            wv_sym.next_to(arrow, UP, buff=0.2, aligned_edge=LEFT)
            subscript.set_value(n)
            v_sym = v_sym_template.copy()
            v_sym.next_to(arrow, RIGHT, MED_SMALL_BUFF)

            v_syms.add(v_sym)
            wv_syms.add(wv_sym)

        self.play(
            FadeIn(big_wv_sym, 0.5 * DOWN),
            FadeIn(wv_word, lag_ratio=0.1),
        )
        self.play(
            LaggedStart(
                (TransformFromCopy(big_wv_sym, wv_sym)
                for wv_sym in wv_syms),
                lag_ratio=0.15,
            ),
            run_time=3
        )
        self.play(
            LaggedStart(
                (TransformFromCopy(e_sym, v_sym)
                for e_sym, v_sym in zip(key_emb_syms, v_syms)),
                lag_ratio=0.15,
            ),
        )
        self.wait()
        self.play(
            FadeTransform(v_syms, k_syms),
            FadeTransform(wv_syms, wk_syms),
            rate_func=there_and_back_with_pause,
            run_time=3,
        )
        self.remove(k_syms, wk_syms)
        self.add(v_syms, wv_syms)
        self.wait()

        # Show column of weights
        index = words.index("creature")
        weighted_sum_cols = VGroup()
        for sm_col in softmax_cols:
            weighted_sum_col = VGroup()
            for weight, v_sym in zip(sm_col, v_syms):
                product = VGroup(weight, v_sym.copy())
                product.target = product.generate_target()
                product.target.arrange(RIGHT)
                product.target[1].shift(UP * (
                    product.target[0].get_y(DOWN) -
                    product.target[1][1].get_y(DOWN)
                ))
                product.target.scale(0.75)
                product.target.move_to(weight)
                product.target.set_fill(
                    opacity=clip(0.6 + weight.get_value(), 0, 1)
                )
                weighted_sum_col.add(product)
            weighted_sum_cols.add(weighted_sum_col)

        self.play(
            FadeOut(dots, lag_ratio=0.1),
            FadeIn(q_fade_rects[:index]),
            FadeIn(q_fade_rects[index + 1:]),
            FadeIn(softmax_cols[index]),
        )
        self.wait()
        self.play(
            LaggedStartMap(MoveToTarget, weighted_sum_cols[index])
        )
        self.wait()

        # Emphasize fluffy and blue weights
        rects = VGroup(
            key_word_groups[i][0].copy()
            for i in [1, 2]
        )
        alt_rects = VGroup(
            SurroundingRectangle(value, buff=SMALL_BUFF)
            for value in (* softmax_cols[index][:1], *softmax_cols[index][3:])
        )
        alt_rects.set_stroke(RED, 1)
        self.play(
            LaggedStart(
                (rect.animate.surround(value)
                for rect, value in zip(rects, softmax_cols[index][1:3])),
                lag_ratio=0.2,
            )
        )
        self.wait()
        self.play(Transform(rects, alt_rects))
        self.wait()
        self.play(FadeOut(rects, lag_ratio=0.1))

        # Show sum (Start re-rendering here, 151)
        emb_sym = emb_syms[index]
        ws_col = weighted_sum_cols[index]
        creature = images[2]
        creature.set_height(1.5)
        creature.next_to(word_groups[index], UP)

        emb_sym.target = emb_sym.generate_target()
        emb_sym.target.scale(1.25, about_edge=UP)
        sum_rect = SurroundingRectangle(emb_sym.target)
        sum_rect.set_stroke(YELLOW, 2)
        sum_rect.target = sum_rect.generate_target()
        sum_rect.target.surround(ws_col, buff=MED_SMALL_BUFF)
        plusses = VGroup()
        for m1, m2 in zip(ws_col, ws_col[1:]):
            plus = Tex(R"+", font_size=72)
            plus.move_to(midpoint(m1.get_bottom(), m2.get_top()))
            plusses.add(plus)

        self.play(
            frame.animate.reorient(0, 0, 0, (-2.6, -4.79, 0.0), 15.07).set_anim_args(run_time=2),
            MoveToTarget(emb_sym),
            ShowCreation(sum_rect),
            FadeIn(creature, UP),
            FadeOut(wv_word),
            FadeOut(big_wv_sym),
        )
        self.add(Point(), q_fade_rects[index + 1:])  # Hack
        self.wait()
        self.play(
            frame.animate.reorient(0, 0, 0, (-2.9, -6.5, 0.0), 19).set_anim_args(run_time=2),
            MoveToTarget(sum_rect, run_time=2),
            Write(plusses),
        )
        self.wait()

        # Show Delta E
        low_eqs = VGroup(
            Tex("=", font_size=72).rotate(PI / 2).next_to(wsc[-1].target, DOWN, buff=0.5)
            for wsc in weighted_sum_cols
        )
        low_eqs.set_color(YELLOW)
        delta_Es = VGroup()
        for emb_sym, eq in zip(emb_syms, low_eqs):
            delta = Tex(R"\Delta")
            delta.match_height(emb_sym[1])
            delta.next_to(emb_sym[1], LEFT, buff=0, aligned_edge=DOWN)
            delta_E = VGroup(delta, emb_sym.copy())
            delta_E.set_color(YELLOW)
            delta_E.set_height(0.8)
            delta_E.next_to(eq, DOWN)
            delta_Es.add(delta_E)

        self.play(
            LaggedStart(
                (FadeTransform(term.copy(), delta_Es[index])
                for term in weighted_sum_cols[index]),
                lag_ratio=0.05,
                group_type=Group
            ),
            Write(low_eqs[index])
        )
        self.wait()

        # Add Delta E
        creature_group = Group(creature, q_groups[index]).copy()
        creature_group.target = creature_group.generate_target()
        creature_group.target.scale(1.5)
        creature_group.target.next_to(h_lines, RIGHT, buff=4.0)
        creature_group.target.align_to(creature, UP)
        right_plus = Tex("+", font_size=96)
        right_eq = Tex("=", font_size=120).rotate(PI / 2)
        right_plus.next_to(creature_group.target, DOWN)
        creature_delta_E = delta_Es[index].copy()
        creature_delta_E.target = creature_delta_E.generate_target()
        creature_delta_E.target.set_height(1.0)
        creature_delta_E.target.next_to(right_plus, DOWN)
        right_eq.next_to(creature_delta_E.target, DOWN, MED_LARGE_BUFF)
        E_prime = emb_sym_primes[index].copy()
        E_prime.set_height(1.25)
        E_prime.next_to(right_eq, DOWN, MED_LARGE_BUFF)
        blue_fluff.set_height(2.5)
        blue_fluff.next_to(E_prime, DOWN, MED_LARGE_BUFF)

        self.play(LaggedStart(
            frame.animate.reorient(0, 0, 0, (4.96, -5.61, 0.0), 19.00),
            MoveToTarget(creature_group),
            FadeTransform(sum_rect.copy(), right_plus),
            MoveToTarget(creature_delta_E),
            run_time=2,
            lag_ratio=0.1
        ))
        self.wait()
        self.play(
            FadeTransform(creature_group[1][-4].copy(), E_prime),
            FadeTransform(creature_delta_E.copy(), E_prime),
            Write(right_eq),
            FadeTransform(creature_group[0].copy(), blue_fluff, path_arc=-PI / 2, run_time=2)
        )
        self.wait()

        right_sum_group = Group(
            creature_group, right_plus, creature_delta_E,
            right_eq, E_prime, blue_fluff
        )

        # Show all column sums
        plus_groups = VGroup(
            plusses.copy().match_x(col[0].target)
            for col in weighted_sum_cols
        )
        plus_groups.set_fill(GREY_C, 1)

        for col in softmax_cols:
            for value in col:
                value.set_fill(
                    opacity=clip(0.6 + value.get_value(), 0, 1)
                )

        self.play(LaggedStart(
            right_sum_group.animate.fade(0.75),
            FadeOut(sum_rect),
            FadeOut(creature),
            FadeOut(q_fade_rects[:index]),
            FadeOut(q_fade_rects[index + 1:]),
            FadeIn(softmax_cols[:index]),
            FadeIn(softmax_cols[index + 1:]),
            plusses.animate.set_fill(GREY_C, 1),
            run_time=2,
        ))
        self.play(
            LaggedStart(
                (LaggedStartMap(MoveToTarget, col)
                for col in weighted_sum_cols),
                lag_ratio=0.1
            ),
            v_lines.animate.set_stroke(GREY_A, 4, 1),
            *(
                e_sym.animate.scale(1.25, about_edge=UP)
                for e_sym in (*emb_syms[:index], *emb_syms[index + 1:])
            ),
        )

        other_indices = [*range(index), *range(index + 1, len(plus_groups))]
        self.play(LaggedStart(
            (LaggedStart(
                FadeIn(plus_groups[j], lag_ratio=0.1),
                Write(low_eqs[j]),
                LaggedStart(
                    (FadeTransform(ws.copy(), delta_Es[j])
                    for ws in weighted_sum_cols[j]),
                    lag_ratio=0.05,
                    group_type=Group
                ),
                lag_ratio=0.05,
            )
            for j in other_indices),
            lag_ratio=0.1,
            group_type=Group
        ))
        self.wait()

        # Add all deltas to embeddings
        equations = VGroup()
        equation_targets = VGroup()
        for E, dE, Ep in zip(emb_syms.copy(), delta_Es.copy(), emb_sym_primes):
            Ep.match_height(E)
            plus = Tex("+", font_size=96)
            eq = Tex("=", font_size=96).rotate(PI / 2)
            equation = VGroup(E, plus, dE, eq, Ep)
            equation.target = equation.generate_target()
            for mob in equation.target[::2]:
                mob.set_height(0.8)
            equation.target.arrange(DOWN)
            for mob in [Ep, plus, eq]:
                mob.set_opacity(0)
                mob.move_to(dE)
            equations.add(equation)
            equation_targets.add(equation.target)

        equation_targets.scale(1.25)
        equation_targets.arrange(RIGHT, buff=0.75)
        equation_targets.next_to(h_lines, RIGHT, buff=1.5)
        equation_targets.match_y(h_lines)

        self.play(
            frame.animate.reorient(0, 0, 0, (9.5, -7.17, 0.0), 20.33),
            LaggedStartMap(MoveToTarget, equations, lag_ratio=0.05),
            FadeTransform(right_sum_group, equation_targets[index]),
            run_time=2.0
        )
        self.wait()

        result_rect = SurroundingRectangle(
            VGroup(eq[-1] for eq in equations),
            buff=0.25
        )
        result_rect.set_stroke(TEAL, 3)
        self.play(
            ShowCreation(result_rect)
        )
        self.wait()

    def bake_mobject_into_vector_entries(self, mob, vector, path_arc=30 * DEGREES, group_type=None):
        entries = vector.get_entries()
        mob_copies = mob.replicate(len(entries))
        return AnimationGroup(
            LaggedStart(
                (FadeOutToPoint(mc, entry.get_center(), path_arc=path_arc)
                for mc, entry in zip(mob_copies, entries)),
                lag_ratio=0.05,
                group_type=group_type,
                run_time=2,
                remover=True
            ),
            RandomizeMatrixEntries(
                vector,
                rate_func=lambda t: clip(smooth(2 * t - 1), 0, 1),
                run_time=2
            ),
        )

    def scrap():
        # To be inserted after Show grid of dots sections
        self.remove(dot_prods)
        np.random.seed(time.gmtime().tm_sec)
        pattern = np.random.normal(0, 1, (8, 8))
        for n in range(len(pattern[0])):
            pattern[:, n][n + 1:] = -np.inf
            pattern[:, n] = softmax(pattern[:, n])
        for row, arr in zip(dots, pattern):
            for dot, value in zip(row, arr):
                dot.set_width(value**0.5)
        dots.set_fill(GREY_B, 1)
        return

        ### To be inserted in "Show softmax" section
        np.random.seed(time.gmtime().tm_sec)
        softmax_arrays = np.random.normal(0, 1, (8, 8))
        for n in range(len(softmax_arrays[0])):
            softmax_arrays[:, n][n + 1:] = -np.inf
            softmax_arrays[:, n] = softmax(softmax_arrays[:, n])
        softmax_arrays = softmax_arrays.T
        ###

    def thumbnail():
        ### Thumbnail design, insert in the middle of softmax show columns ###
        self.remove(q_groups)
        self.add(q_syms)
        out_dots = VGroup()
        for col in softmax_cols:
            for value in col:
                dot = Dot(radius=0.35)
                dot.move_to(value)
                dot.set_fill(WHITE, opacity=interpolate(0.1, 0.9, value.get_value()))
                out_dots.add(dot)
        out_dots.shift(2 * OUT)
        out_dots.set_stroke(WHITE, 2, 0.25)
        self.remove(softmax_cols)
        self.remove(sm_rects[last_index])
        self.add(out_dots)
        index = 3
        ndp_columns[-1].set_opacity(0.25)
        ndp_columns[index].set_opacity(1)
        sm_label_group = VGroup(sm_arrows[last_index], sm_labels[last_index])
        sm_label_group.match_x(ndp_columns[index])
        sm_label_group[1].scale(1.5, about_edge=DOWN)
        sm_label_group[1].set_fill(border_width=0)
        col_rect.match_x(ndp_columns[index])
        col_rect.set_flat_stroke(False)
        sm_col = col_rect.copy()
        # sm_col.set_width(out_dots[0].get_width() + 0.2)
        sm_col.match_z(out_dots)
        sm_col.set_flat_stroke(False)
        self.add(sm_col)
        self.remove(sm_labels[last_index])
        sm_arrows[last_index].set_stroke(width=10)
        sm_arrows[last_index].shift(OUT)

        grid_lines.set_stroke(WHITE, 2)
        v_lines.set_height(12, about_edge=DOWN, stretch=True)

        frame.set_field_of_view(35 * DEGREES)
        frame.reorient(-52, -2, 0, (-1.74, -7.1, -0.03), 14.72)
        ###

        ### To be inserted before Set aside keys and queries
        frame.move_to([-4.62, -5.04, 0.0]).set_height(14.5)
        self.remove(pattern_words)

        for dot in dots.family_members_with_points():
            value = dot.get_radius() / 0.5
            dot.set_fill(WHITE, opacity=value**0.75)
            dot.set_width(1)

        title = Text("Attention", font_size=250)
        title.set_fill(border_width=2)
        title.next_to(q_syms, LEFT, LARGE_BUFF, DOWN)
        title.shift(0.5 * UP)
        # self.add(title)

        q_syms.set_fill(border_width=1.5)
        k_syms.set_fill(border_width=1.5)
        for q in q_syms:
            q.scale(1.5, about_edge=DOWN)
        for k in k_syms:
            k.scale(1.5, about_edge=RIGHT)

        self.remove(word_groups, q_arrows, emb_arrows, emb_syms, wq_syms)
        VGroup(key_word_groups, key_emb_syms, key_emb_arrows, wk_arrows, wk_syms).shift(0.25 * LEFT)
        ###


class MyseteryNovel(InteractiveScene):
    def construct(self):
        # Create paragraphs
        text = Path(DATA_DIR, "murder_story.txt").read_text()
        paragraphs = VGroup(
            get_paragraph(para.split(" "), line_len=40)
            for para in text.split("\n\n")
        )
        dots = Tex(R"\vdots", font_size=200)
        paragraphs.replace_submobject(4, dots)
        paragraphs.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        dots.match_x(paragraphs)
        self.add(paragraphs)

        # Mark last word
        last_word = paragraphs[-1]["Derek!\""][0]
        rect = SurroundingRectangle(last_word)
        rect.set_stroke(YELLOW, 2)
        rect.set_fill(YELLOW, 0.25)
        q_marks = Tex("???")
        q_marks.move_to(rect)
        rect.add(q_marks)
        rect.shift(0.05 * DR)

        last_word.scale(0).set_fill(BLACK)
        self.add(rect)

        # Show the first line
        frame = self.frame
        frame.set_y(15)
        paragraphs.set_fill(opacity=0.25)
        opening = paragraphs[0]["It was a dark and stormy night."][0]
        self.play(opening.animate.set_fill(opacity=1).set_anim_args(lag_ratio=0.1))
        self.wait()

        # Scroll down
        penultimate_words = paragraphs[-1]["therefore, the murderer was"][0]
        self.play(
            frame.animate.set_y(-15.4),
            paragraphs.animate.set_fill(opacity=1).set_anim_args(lag_ratio=0.01),
            run_time=5,
        )
        self.wait()
        self.add(penultimate_words.copy())
        self.play(paragraphs.animate.set_opacity(0.25))
        self.wait()

        # Show the final vector
        was = penultimate_words[-3:]
        arrow = FillArrow(ORIGIN, DOWN, buff=0, thickness=0.07)
        arrow.next_to(was, DOWN, MED_SMALL_BUFF)
        vect = NumericEmbedding(length=12)
        vect.set_height(5)
        vect.next_to(arrow, DOWN)

        self.play(LaggedStart(
            frame.animate.set_y(-17.5).set_height(12.5),
            FadeIn(arrow, scale=3, shift=DOWN),
            FadeIn(vect, DOWN),
            run_time=2
        ))
        self.context_anim(paragraphs[-1], vect)
        self.wait()

        # Zoom out more
        vect_group = VGroup(arrow, vect)
        vect_group.target = vect_group.generate_target()
        vect_group.target.scale(2.35, about_edge=UP)
        self.play(
            paragraphs.animate.set_fill(opacity=0.8),
            frame.animate.set_height(37).set_y(-14),
            MoveToTarget(vect_group),
            run_time=2
        )
        self.context_anim(paragraphs[-4:], vect)

    def context_anim(self, source, vect):
        flat_source = VGroup(*source.family_members_with_points())
        vect_len = len(vect.get_entries())
        self.play(
            LaggedStart(
                (ContextAnimation(
                    entry, flat_source[n::vect_len],
                    path_arc=-PI / 2,
                    run_time=5,
                    lag_ratio=1e-3,
                    max_stroke_width=2
                )
                for n, entry in enumerate(vect.get_entries())),
                lag_ratio=0.1,
            ),
            RandomizeMatrixEntries(vect, run_time=5),
        )


class RoadNotTaken(InteractiveScene):
    def construct(self):
        # Add poem
        stanza_strs = [
            """
                Two roads diverged in a yellow wood,
                And sorry I could not travel both
                And be one traveler, long I stood
                And looked down one as far as I could
                To where it bent in the undergrowth;
            """,
            """
                Then took the other, as just as fair,
                And having perhaps the better claim,
                Because it was grassy and wanted wear;
                Though as for that the passing there
                Had worn them really about the same,
            """,
            """
                And both that morning equally lay
                In leaves no step had trodden black.
                Oh, I kept the first for another day!
                Yet knowing how way leads on to way,
                I doubted if I should ever come back.
            """,
            """
                I shall be telling this with a sigh
                Somewhere ages and ages hence:
                Two roads diverged in a wood, and I—
                I took the one less traveled by,
                And that has made all the difference.
            """,
        ]
        poem = Text("\n\n".join(stanza_strs), alignment="LEFT")
        stanzas = VGroup(poem[stanza_str][0] for stanza_str in stanza_strs)
        stanzas.arrange_in_grid(h_buff=1.5, v_buff=1.0, fill_rows_first=False)
        stanzas.set_width(FRAME_WIDTH - 1)
        stanzas.move_to(0.5 * UP)
        poem.refresh_bounding_box(recurse_down=True)

        self.play(FadeIn(poem, lag_ratio=0.01, run_time=4))
        self.wait()

        # Note all text until "one"
        rect = SurroundingRectangle(poem)
        less = poem["less"][-1]
        one = poem["one"][-1]
        diff_rects = VGroup(
            SurroundingRectangle(mob).scale(10, about_edge=UL)
            for mob in [less, poem["And"][-1]]
        )
        for diff_rect in diff_rects:
            rect = Difference(rect, diff_rect)
        rect.set_stroke(TEAL, 3)

        less_index = poem.submobjects.index(less[0])
        faded_portion = poem[less_index:]
        active_portion = poem[:less_index]
        less_rect = SurroundingRectangle(less)
        less_rect.set_stroke(YELLOW, 3)
        one_rect = SurroundingRectangle(one)
        one_rect.become(Difference(one_rect, less_rect))
        one_rect.match_height(less_rect, about_edge=DOWN, stretch=True)
        one_rect.set_stroke(BLUE, 3)
        arrow = Vector(0.75 * UP)
        arrow.next_to(one, DOWN, SMALL_BUFF)
        arrow.set_stroke(YELLOW)
        active_portion_copy = active_portion.copy()
        active_portion_copy.set_color(TEAL_B)

        self.play(
            FadeIn(rect),
            Write(active_portion_copy, run_time=2, stroke_color=TEAL, lag_ratio=0.01),
            faded_portion.animate.set_fill(opacity=0.5),
        )
        self.play(FadeOut(active_portion_copy))
        self.wait()
        self.play(GrowArrow(arrow))
        self.wait()
        self.play(
            ShowCreation(less_rect),
            less.animate.set_fill(opacity=1),
            arrow.animate.match_x(less),
        )
        self.wait()
        self.remove(less_rect)
        self.play(
            arrow.animate.match_x(one),
            TransformFromCopy(less_rect, one_rect),
        )
        self.wait()

        # Highlight "two roads"
        one = one.copy()
        less = less.copy()
        two_roads = poem["Two roads"][-1].copy()
        took_the = poem["I took the"][-1].copy()

        self.play(
            FadeIn(two_roads, lag_ratio=0.1),
            FadeIn(took_the, lag_ratio=0.1),
            FadeIn(one),
            arrow.animate.rotate(-PI / 2).next_to(two_roads, LEFT, SMALL_BUFF),
            poem.animate.set_fill(opacity=0.5),
            run_time=1.5
        )
        self.wait()

        # Highlight "took the other" and "grassy and wanted wear"
        top_two_roads = poem["Two roads diverged"][0].copy()
        took_other = poem["Then took the other"][0].copy()
        wanted_wear = poem["it was grassy and wanted wear"][0].copy()
        for phrase in [top_two_roads, took_other, wanted_wear]:
            phrase.set_fill(WHITE, 1)

        self.play(
            arrow.animate.rotate(PI / 2).next_to(top_two_roads, DOWN, SMALL_BUFF),
            FadeIn(top_two_roads),
        )
        self.wait()
        self.play(
            arrow.animate.rotate(3 * PI / 4).next_to(took_other, UP, SMALL_BUFF),
            FadeIn(took_other)
        )
        self.wait()
        self.play(
            arrow.animate.rotate(-PI / 2).next_to(wanted_wear, DOWN, SMALL_BUFF),
            FadeIn(wanted_wear)
        )
        self.wait()

        # Higlight words throughout
        active_portion_copy.set_fill(YELLOW_A, 1)

        self.play(
            LaggedStart(
                (FadeIn(char, rate_func=there_and_back_with_pause)
                for char in active_portion_copy),
                lag_ratio=0.005,
                run_time=6
            )
        )
        self.wait()

        # Show less again
        self.play(
            arrow.animate.rotate(-PI / 4).next_to(less, DOWN, SMALL_BUFF),
            ShowCreation(less_rect),
            less.animate.set_fill(WHITE, 1)
        )
        self.wait()

        # Show final embedding
        frame = self.frame
        embedding = NumericEmbedding(length=10)
        embedding.set_height(3)
        embedding.next_to(one, DOWN, buff=arrow.get_length() + 2 * SMALL_BUFF)

        self.play(
            arrow.animate.rotate(PI).next_to(one, DOWN, SMALL_BUFF).set_anim_args(path_arc=PI),
            frame.animate.set_height(9).move_to(DOWN)
        )
        self.play(TransformFromCopy(one, embedding))
        self.play(RandomizeMatrixEntries(embedding))
        self.wait()


class QueryMap(InteractiveScene):
    map_tex = "W_Q"
    map_color = YELLOW
    src_name = "Creature"
    pos_word = "position 4"
    trg_name = "Any adjectives\nbefore position 4?"
    in_vect_color = BLUE_B
    in_vect_coords = (3, 2, -2)
    out_vect_coords = (-2, -1)

    def construct(self):
        # Setup 3d axes
        axes_3d = ThreeDAxes((-4, 4), (-3, 3), (-4, 4))
        xz_plane = NumberPlane(
            (-4, 4), (-4, 4),
            background_line_style=dict(
                stroke_color=GREY,
                stroke_width=1,
            ),
            faded_line_ratio=0
        )
        xz_plane.rotate(90 * DEGREES, RIGHT)
        xz_plane.move_to(axes_3d)
        xz_plane.axes.set_opacity(0)
        axes_3d.add(xz_plane)
        axes_3d.set_height(2.0)

        self.set_floor_plane("xz")
        frame = self.frame
        frame.set_field_of_view(30 * DEGREES)
        frame.reorient(-32, 0, 0, (2.13, 1.11, 0.27), 4.50)
        frame.add_ambient_rotation(1 * DEGREES)

        self.add(axes_3d)

        # Set up target plane
        plane = NumberPlane(
            (-3, 3), (-3, 3),
            faded_line_ratio=1,
            background_line_style=dict(
                stroke_color=BLUE,
                stroke_width=1,
                stroke_opacity=0.75
            ),
            faded_line_style=dict(
                stroke_color=BLUE,
                stroke_width=1,
                stroke_opacity=0.25,
            )
        )
        plane.set_height(3.5)
        plane.to_corner(DR)

        arrow = Tex(R"\longrightarrow")
        arrow.set_width(2)
        arrow.stretch(0.75, 1)
        arrow.next_to(plane, LEFT, buff=1.0)
        arrow.set_color(self.map_color)

        map_name = Tex(self.map_tex, font_size=72)
        map_name.set_color(self.map_color)
        map_name.next_to(arrow.get_left(), UR, SMALL_BUFF).shift(0.25 * RIGHT)

        for mob in [plane, arrow, map_name]:
            mob.fix_in_frame()

        self.add(plane)
        self.add(arrow)
        self.add(map_name)

        # Add titles
        titles = VGroup(
            Text("Embedding space"),
            Text("Query/Key space"),
        )
        subtitles = VGroup(
            Text("12,288-dimensional"),
            Text("128-dimensional"),
        )
        subtitles.scale(0.75)
        subtitles.set_fill(GREY_B)
        x_values = [-frame.get_x() * FRAME_HEIGHT / frame.get_height(), plane.get_x()]
        for title, subtitle, x_value in zip(titles, subtitles, x_values):
            subtitle.next_to(title, DOWN, SMALL_BUFF)
            title.add(subtitle)
            title.next_to(plane, UP, MED_LARGE_BUFF)
            title.set_x(x_value)
            title.fix_in_frame()

        self.add(titles)

        # Show vector transformation
        in_vect = Arrow(axes_3d.get_origin(), axes_3d.c2p(*self.in_vect_coords), buff=0)
        in_vect.set_stroke(self.in_vect_color)
        in_vect_label = TexText("``" + self.src_name + "''", font_size=24)
        pos_label = Text(self.pos_word, font_size=16)
        pos_label.next_to(in_vect_label, DOWN, SMALL_BUFF)
        pos_label.set_opacity(0.75)
        in_vect_label.add(pos_label)
        in_vect_label.set_color(self.in_vect_color)
        in_vect_label.next_to(in_vect.get_end(), UP, SMALL_BUFF)

        out_vect = Arrow(plane.get_origin(), plane.c2p(*self.out_vect_coords), buff=0)
        out_vect.set_stroke(self.map_color)
        out_vect_label = Text(self.trg_name, font_size=30)
        out_vect_label.next_to(out_vect.get_end(), DOWN, buff=0.2)
        out_vect_label.set_backstroke(BLACK, 5)
        VGroup(out_vect, out_vect_label).fix_in_frame()

        self.play(
            GrowArrow(in_vect),
            FadeInFromPoint(in_vect_label, axes_3d.get_origin()),
        )
        self.wait(2)
        self.play(
            TransformFromCopy(in_vect, out_vect),
            FadeTransform(in_vect_label.copy(), out_vect_label),
            run_time=2,
        )
        self.wait(20)
        self.play(FadeOut(out_vect_label))
        self.wait(5)


class KeyMap(QueryMap):
    map_tex = "W_K"
    map_color = TEAL
    src_name = "Fluffy"
    pos_word = "position 2"
    trg_name = "Adjective at\nposition 2"
    in_vect_color = BLUE_B
    in_vect_coords = (-3, 1, 2)
    out_vect_coords = (-1.75, -1)


class DescribeAttentionEquation(InteractiveScene):
    def construct(self):
        # Stage image
        image = ImageMobject("AttentionPaperStill")
        image.set_height(FRAME_HEIGHT)
        self.add(image)

        # Add equation
        equation = Tex(R"\text{Attention}(Q, K, V) = \text{softmax}\left({K^T Q \over \sqrt{d_k}}\right) V")
        equation.set_height(1.06929)
        equation.move_to([-0.41406, 1.177, 0])

        self.play(
            FadeIn(equation),
            FadeOut(image),
        )
        self.wait()

        # Show Q and K arrays
        syms = ["Q", "K"]
        colors = [YELLOW, TEAL]
        q_array, k_array = arrays = VGroup(
            self.get_array_representation(sym, color)
            for sym, color in zip(syms, colors)
        )
        arrays.arrange(RIGHT, buff=1.5)
        arrays.next_to(equation, DOWN, buff=1.0)

        lil_rects = VGroup()
        rect_lines = VGroup()
        big_rects = VGroup()
        for arr, sym, color in zip(arrays, syms, colors):
            lil_rect = SurroundingRectangle(equation["Q"][0])
            lil_rect.match_x(equation[sym][0])
            big_rect = SurroundingRectangle(arr)
            lines = VGroup(
                Line(lil_rect.get_corner(DOWN + v), big_rect.get_corner(UP + v))
                for v in [LEFT, RIGHT]
            )
            VGroup(lil_rect, big_rect, lines).set_stroke(color, 2)
            lil_rects.add(lil_rect)
            rect_lines.add(lines)
            big_rects.add(big_rect)

            self.play(
                ShowCreation(lil_rect),
                equation[sym].animate.set_color(color),
            )
            self.play(
                TransformFromCopy(lil_rect, big_rect),
                FadeInFromPoint(arr, lil_rect.get_center()),
                ShowCreation(lines, lag_ratio=0)
            )
        self.wait()

        # Highlight numerator
        num_rect = SurroundingRectangle(equation["K^T Q"])
        num_rect.set_stroke(BLUE, 2)

        self.play(
            ReplacementTransform(lil_rects[0], num_rect),
            ReplacementTransform(lil_rects[1], num_rect),
            FadeOut(rect_lines)
        )
        self.wait()

        # Arrange for grid
        frame = self.frame
        qs = q_array[1]
        ks = k_array[1]
        q_array.remove(qs)
        k_array.remove(ks)

        h_buff = 0.8
        v_buff = 0.6

        qs.target = qs.generate_target()
        qs.target.scale(0.75)
        qs.target.arrange(RIGHT, buff=h_buff)
        qs.target.next_to(equation, DOWN, buff=0.75)

        ks.target = ks.generate_target()
        ks.target.scale(0.75)
        ks.target.arrange(DOWN, buff=MED_LARGE_BUFF)
        ks.target[-2].rotate(PI / 2)
        ks.target.next_to(qs.target, DL, buff=v_buff)

        self.play(
            frame.animate.move_to(1.5 * DOWN),
            FadeOut(q_array),
            FadeOut(k_array),
            MoveToTarget(qs),
            MoveToTarget(ks),
            big_rects[0].animate.surround(qs.target).set_stroke(opacity=0),
            big_rects[1].animate.surround(ks.target).set_stroke(opacity=0),
            run_time=2
        )

        # Add grid lines
        grid = VGroup(qs, ks)

        v_lines = Line(UP, DOWN).match_height(grid).scale(1.1).replicate(len(qs) + 1)
        for v_line, mob in zip(v_lines, (ks, *qs)):
            v_line.next_to(mob, RIGHT, buff=h_buff / 2)
            v_line.align_to(qs, UP)

        h_lines = Line(LEFT, RIGHT).match_width(grid).scale(1.1).replicate(len(ks) + 1)
        for h_line, mob in zip(h_lines, (qs, *ks)):
            h_line.next_to(mob, DOWN, buff=v_buff / 2)
            h_line.align_to(ks, LEFT)

        VGroup(v_lines, h_lines).set_stroke(GREY_B, 1)

        grid.add(v_lines, h_lines)

        self.play(
            FadeIn(h_lines, lag_ratio=0.1),
            FadeIn(v_lines, lag_ratio=0.1),
            ks[-2].animate.match_y(h_lines[-3:-1]),
        )

        # Dot products
        dot_prods = VGroup()
        for q in qs:
            for k in ks:
                dot = Tex(".")
                dot.match_x(q)
                dot.match_y(k)
                dot_prod = VGroup(q.copy(), dot, k.copy())
                dot_prod.target = dot_prod.generate_target()
                dot_prod.target.arrange(RIGHT, buff=SMALL_BUFF)
                dot_prod.target.scale(0.7)
                dot_prod.target.move_to(dot)
                if len(q) == 3:
                    dot_prod.target[1:].scale(0)
                    for mob in dot_prod.target:
                        mob.move_to(dot)
                elif len(k) == 3:
                    dot_prod.target[:-1].scale(0)
                    for mob in dot_prod.target:
                        mob.move_to(dot)
                dot.set_opacity(0)
                dot_prods.add(dot_prod)

        self.play(
            LaggedStartMap(MoveToTarget, dot_prods, lag_ratio=0.01),
            run_time=3
        )
        self.wait()

        # Add sqrt to denominator
        sqrt_part = equation[R"\over \sqrt{d_k}"][0]

        denoms = VGroup()
        for dot_prod in dot_prods:
            dot_prod.target = dot_prod.generate_target()
            if 3 in [len(dot_prod[0]), len(dot_prod[2])]:
                continue
            denom = sqrt_part.copy()
            denom.set_fill(opacity=0.9)
            denom.match_width(dot_prod)
            denom.move_to(dot_prod.get_center(), UP)
            dot_prod.target.next_to(denom, UP, buff=SMALL_BUFF)
            VGroup(dot_prod.target, denom).scale(0.75)
            denoms.add(denom)

        self.play(num_rect.animate.surround(equation[R"K^T Q \over \sqrt{d_k}"]))
        self.play(
            LaggedStartMap(MoveToTarget, dot_prods, lag_ratio=0.05, time_span=(1, 3)),
            LaggedStart(
                (TransformFromCopy(sqrt_part, denom)
                for denom in denoms),
                lag_ratio=0.01,
            ),
            run_time=3
        )
        self.wait()

        # Highlight softmax
        self.play(
            num_rect.animate.surround(equation[R"\text{softmax}\left({K^T Q \over \sqrt{d_k}}\right)"])
        )
        self.wait()

        # Mention V
        v_parts = equation["V"]
        v_rects = VGroup(map(SurroundingRectangle, v_parts))
        v_rects.set_stroke(RED, 3)

        self.play(
            ReplacementTransform(VGroup(num_rect), v_rects),
            v_parts.animate.set_color(RED),
        )
        self.wait()

    def get_array_representation(self, sym, color=WHITE, length=7):
        template = Tex(f"{sym}_0")
        template.set_fill(color)
        substr = template.make_number_changeable(0)
        terms = VGroup()
        term_lines = VGroup()
        term_groups = VGroup()
        for n in range(1, length + 1):
            if n == length:
                substr.become(Tex("n").replace(substr))
            else:
                substr.set_value(n)
            substr.set_color(color)
            term = template.copy()
            lines = Line(ORIGIN, 0.5 * UP).replicate(2)
            lines.arrange(DOWN, buff=term.get_height() + 2 * SMALL_BUFF)
            lines.move_to(term)
            term_lines.add(lines)
            terms.add(term)
            term_groups.add(VGroup(term, lines))
        term_groups.arrange(RIGHT, buff=MED_SMALL_BUFF)

        dots = Tex(R"\dots")
        dots.replace(terms[-2], dim_to_match=0)
        terms.replace_submobject(length - 2, dots)
        term_groups.remove(term_groups[-2])

        brackets = Tex("[]")
        brackets.stretch(1.5, 1)
        brackets.set_height(term_groups.get_height() + MED_SMALL_BUFF)
        for bracket, vect in zip(brackets, [LEFT, RIGHT]):
            bracket.next_to(terms, vect, SMALL_BUFF)

        result = VGroup(brackets, terms, term_lines)

        return result


class ShowAllPossibleNextTokenPredictions(InteractiveScene):
    def construct(self):
        # Add phrase
        phrase = Text("the fluffy blue creature roamed the verdant forest despite")
        plain_words = break_into_words(phrase)
        rects = get_piece_rectangles(plain_words)
        words = VGroup(VGroup(*pair) for pair in zip(rects, plain_words))
        words = words[:-1]
        words.to_edge(LEFT, buff=MED_LARGE_BUFF)

        next_token_box = rects[-1].copy()
        next_token_box.set_color(YELLOW)
        next_token_box.set_stroke(YELLOW, 3)
        next_token_box.next_to(words, RIGHT, buff=LARGE_BUFF)
        q_marks = Tex("???")
        q_marks.move_to(next_token_box)
        next_token_box.add(q_marks)

        arrow = Arrow(words, next_token_box, buff=SMALL_BUFF)

        self.add(words)
        self.play(
            GrowArrow(arrow),
            FadeIn(next_token_box, RIGHT)
        )
        self.wait()

        # Set up subphrases
        scale_factor = 0.75
        v_buff = 0.4
        subphrases = VGroup(
            words[:n].copy().scale(scale_factor)
            for n in range(1, len(words) + 1)
        )
        subphrases.arrange(DOWN, buff=v_buff, aligned_edge=LEFT)
        subphrases.to_corner(UL)

        rhs = VGroup(arrow, next_token_box)
        alt_rhss = VGroup(
            rhs.copy().scale(scale_factor).next_to(subphrase, RIGHT, SMALL_BUFF)
            for subphrase in subphrases
        )

        self.play(
            Transform(words, subphrases[-1]),
            Transform(rhs, alt_rhss[-1]),
        )
        for n in range(len(subphrases) - 1, 0, -1):
            sp1 = subphrases[n]
            sp2 = subphrases[n - 1]
            rhs1 = alt_rhss[n]
            rhs2 = alt_rhss[n - 1]
            self.play(
                TransformFromCopy(sp1[:len(sp2)], sp2),
                TransformFromCopy(rhs1, rhs2),
                rate_func=linear,
                run_time=0.5
            )
        self.wait()

        # Highlight two examples
        for phrase, alt_rhs in zip(subphrases, alt_rhss):
            arrow = alt_rhs[0]
            alt_rhs.remove(arrow)
            phrase.add(arrow)
            phrase.save_state()
        index = 3
        self.play(LaggedStart(
            FadeOut(alt_rhss),
            FadeOut(rhs),
            words.animate.fade(0.75),
            subphrases[:index].animate.fade(0.75),
            subphrases[index + 1:].animate.fade(0.75),
            subphrases[index].animate.align_to(3 * RIGHT, RIGHT),
        ))
        self.wait()
        self.play(
            subphrases[index].animate.align_to(subphrases, LEFT).fade(0.75),
            subphrases[5].animate.restore().align_to(3 * RIGHT, RIGHT),
        )
        self.wait()

    def get_next_word_distribution():
        pass


class ShowMasking(InteractiveScene):
    def construct(self):
        # Set up two patterns
        shape = (6, 6)
        left_grid = Square().get_grid(*shape, buff=0)
        left_grid.set_shape(5.5, 5)
        left_grid.to_edge(LEFT)
        left_grid.set_y(-0.5)
        left_grid.set_stroke(GREY_B, 1)

        right_grid = left_grid.copy()
        right_grid.to_edge(RIGHT)

        grids = VGroup(left_grid, right_grid)
        arrow = Arrow(left_grid, right_grid)
        sm_label = Text("softmax")
        sm_label.next_to(arrow, UP)

        titles = VGroup(
            Text("Unnormalized\nAttention Pattern"),
            Text("Normalized\nAttention Pattern"),
        )
        for title, grid in zip(titles, grids):
            title.next_to(grid, UP, buff=MED_LARGE_BUFF)

        values_array = np.random.normal(0, 2, shape)
        font_size = 30
        raw_values = VGroup(
            DecimalNumber(
                value,
                include_sign=True,
                font_size=font_size,
            ).move_to(square)
            for square, value in zip(left_grid, values_array.flatten())
        )

        self.add(left_grid)
        self.add(right_grid)
        self.add(titles)
        self.add(arrow)
        self.add(sm_label)
        self.add(raw_values)

        # Highlight lower lefts
        changers = VGroup()
        for n, dec in enumerate(raw_values):
            i = n // shape[1]
            j = n % shape[1]
            if i > j:
                changers.add(dec)
                neg_inf = Tex(R"-\infty", font_size=36)
                neg_inf.move_to(dec)
                neg_inf.set_fill(RED, border_width=1.5)
                dec.target = neg_inf
                values_array[i, j] = -np.inf
        rects = VGroup(map(SurroundingRectangle, changers))
        rects.set_stroke(RED, 3)

        self.play(LaggedStartMap(ShowCreation, rects))
        self.play(
            LaggedStartMap(FadeOut, rects),
            LaggedStartMap(MoveToTarget, changers)
        )
        self.wait()

        # Normalized values
        normalized_array = np.array([
            softmax(col)
            for col in values_array.T
        ]).T
        normalized_values = VGroup(
            DecimalNumber(value, font_size=font_size).move_to(square)
            for square, value in zip(right_grid, normalized_array.flatten())
        )
        for n, value in enumerate(normalized_values):
            value.set_fill(opacity=interpolate(0.5, 1, rush_from(value.get_value())))
            if (n // shape[1]) > (n % shape[1]):
                value.set_fill(RED, 0.75)

        self.play(
            LaggedStart(
                (FadeTransform(v1.copy(), v2)
                for v1, v2 in zip(raw_values, normalized_values)),
                lag_ratio=0.05,
                group_type=Group
            )
        )
        self.wait()


class ScalingAPattern(InteractiveScene):
    def construct(self):
        # Position grid
        N = 50
        grid = Square(side_length=1.0).get_grid(N, N, buff=0)
        grid.set_stroke(GREY_A, 1)
        grid.stretch(0.89, 0)
        grid.stretch(0.70, 1)
        # grid.move_to(1.67 * LEFT + 1.596 * UP, UL)
        grid.move_to(5.0 * LEFT + 2.5 * UP, UL)
        self.add(grid)

        # Dots
        values = np.random.normal(0, 1, (N, N))
        dots = VGroup()
        for n, row in enumerate(values):
            row[:n] = -np.inf
        for k, col in enumerate(values.T):
            for n, value in enumerate(softmax(col)):
                dot = Dot(radius=0.3 * value**0.75)
                dot.move_to(grid[n * N + k])
                dots.add(dot)
        dots.set_fill(GREY_C, 1)
        self.add(dots)

        # Add symbols
        q_template = Tex(R"\vec{\textbf{Q}}_0").set_color(YELLOW)
        k_template = Tex(R"\vec{\textbf{K}}_0").set_color(TEAL)
        for template in [q_template, k_template]:
            template.scale(0.75)
            template.substr = template.make_number_changeable("0")

        qs = VGroup()
        ks = VGroup()
        for n, square in enumerate(grid[:N], start=1):
            q_template.substr.set_value(n)
            q_template.next_to(square, UP, buff=SMALL_BUFF)
            qs.add(q_template.copy())
        for k, square in enumerate(grid[::N], start=1):
            k_template.substr.set_value(k)
            k_template.next_to(square, LEFT, buff=2 * SMALL_BUFF)
            ks.add(k_template.copy())
        self.add(qs, ks)

        # Slowly zoom out
        self.play(
            self.frame.animate.reorient(0, 0, 0, (14.72, -14.71, 0.0), 38.06),
            grid.animate.set_stroke(width=1, opacity=0.25),
            dots.animate.set_fill(GREY_B, 1).set_stroke(GREY_B, 1),
            run_time=20,
        )
        self.wait()


class IntroduceValueMatrix(InteractiveScene):
    def setup(self):
        self.fix_new_entries_in_frame = False
        super().setup()

    def construct(self):
        # Initialized axes
        frame = self.frame
        self.set_floor_plane("xz")
        axes = ThreeDAxes((-4, 4), (-4, 4), (-4, 4))
        plane = NumberPlane(
            (-4, 4), (-4, 4),
            background_line_style=dict(
                stroke_color=GREY,
                stroke_width=1,
                stroke_opacity=0.5,
            )
        )
        plane.axes.set_opacity(0)
        plane.rotate(PI / 2, RIGHT)
        axes.add(plane)

        frame.reorient(5, -4, 0, (-4.66, 2.07, 0.04), 12.48)
        # frame.add_ambient_rotation()
        self.add(axes)

        # Add word pair
        words = VGroup(Text("blue"), Text("fluffy"), Text("creature"))
        words.scale(1.5)
        words.arrange(RIGHT, aligned_edge=UP)
        words.to_edge(UP)
        words.to_edge(LEFT, buff=0)
        rects = get_piece_rectangles(words, h_buff=0.1)
        rects[0].set_color(BLUE)
        rects[1].set_color(TEAL)
        rects[2].set_color(ORANGE)
        arrows = VGroup(Vector(DOWN).next_to(rect, DOWN) for rect in rects)
        embs = VGroup(
            NumericEmbedding(length=8).set_height(4.0).next_to(arrow, DOWN)
            for arrow in arrows
        )

        blue_group = VGroup(rects[0], words[0], arrows[0], embs[0])
        blue_group.set_opacity(0)

        self.fix_new_entries_in_frame = True
        self.add(rects)
        self.add(words)
        self.add(arrows)
        self.add(embs)

        # Add word vectors
        creature_vect = self.get_labeled_vector(axes, (-2, 3, 1), ORANGE, "Dalle3_creature")
        with_fluffy_vect = self.get_labeled_vector(axes, (2, 3, 1), GREY_BROWN, "Dalle3_creature_2")
        with_blue_vect = self.get_labeled_vector(axes, (1, 2, 4), BLUE, "BlueFluff")

        self.wait()
        self.fix_new_entries_in_frame = False
        self.play(
            FadeTransform(words[1].copy(), creature_vect[1]),
            TransformFromCopy(
                Arrow(embs[1].get_bottom(), embs[1].get_top(), buff=0).fix_in_frame().set_stroke(width=10, opacity=0.25),
                creature_vect[0],
            )
        )
        self.add(creature_vect)

        # Show influence
        diff_vect = Arrow(
            creature_vect[0].get_end(),
            with_fluffy_vect[0].get_end(),
            buff=0
        )
        diff_vect.scale(0.95)
        self.fix_new_entries_in_frame = False
        self.play(
            FadeTransform(creature_vect[1].copy(), with_fluffy_vect[1]),
            TransformFromCopy(creature_vect[0], with_fluffy_vect[0]),
            run_time=3,
        )
        self.add(with_fluffy_vect)
        self.play(GrowArrow(diff_vect, run_time=2))

        self.fix_new_entries_in_frame = True
        self.play(
            RandomizeMatrixEntries(embs[2], time_span=(1, 5)),
            LaggedStart(
                (ContextAnimation(entry, embs[1].get_entries(), path_arc=10 * DEGREES, lag_ratio=0.1)
                for entry in embs[2].get_entries()),
                lag_ratio=0.01,
                run_time=5,
            ),
        )
        self.wait()

        # Make room
        corner_group = VGroup(rects, words, arrows, embs)
        self.play(
            frame.animate.reorient(10, -7, 0, (-8.33, -0.79, 0.37), 16.82),
            corner_group.animate.set_height(3).to_edge(UP, buff=0.25).set_x(-2),
            run_time=2
        )

        # Show value matrix
        matrix = WeightMatrix(shape=(8, 8))
        matrix.set_height(2.75)
        matrix.to_corner(DL)
        matrix_brace = Brace(matrix, UP)
        matrix_label = Tex("W_V")
        matrix_label.next_to(matrix_brace, UP)
        matrix_label.set_color(RED)

        fluff_emb = embs[1]
        in_vect_rect = SurroundingRectangle(fluff_emb)
        in_vect_rect.set_stroke(TEAL, 2)
        in_vect = fluff_emb.copy()
        in_vect.match_height(matrix)
        in_vect.next_to(matrix, RIGHT, SMALL_BUFF)
        in_vect_path = self.get_top_vect_to_low_vect_path(fluff_emb, in_vect, TEAL)

        self.fix_new_entries_in_frame = True
        self.play(
            FadeIn(matrix, lag_ratio=1e-3),
            GrowFromCenter(matrix_brace),
            FadeIn(matrix_label, shift=0.25 * UP)
        )
        self.play(ShowCreation(in_vect_rect))
        self.play(
            ShowCreation(in_vect_path),
            TransformFromCopy(fluff_emb, in_vect, path_arc=-20 * DEGREES),
            run_time=2
        )

        # Show matrix product
        eq, rhs = show_matrix_vector_product(self, matrix, in_vect)
        self.wait()

        # Position value vect
        value_rect = SurroundingRectangle(rhs)
        value_rect.set_stroke(RED, 2)
        value_label = Text("Value")
        value_label.next_to(value_rect, RIGHT)
        value_label.set_color(RED)
        value_label.set_backstroke()
        self.fix_new_entries_in_frame = True
        self.play(
            ShowCreation(value_rect),
            FadeIn(value_label, lag_ratio=0.1)
        )
        self.wait()

        value_label2 = value_label.copy()
        value_label2.set_backstroke(BLACK, 5)
        value_label2.scale(1.5)
        value_label2.next_to(diff_vect, UP, MED_SMALL_BUFF)
        value_label2.unfix_from_frame()

        self.fix_new_entries_in_frame = False
        self.play(
            frame.animate.reorient(29, -2, 0, (-7.48, 1.91, 1.21), 11.89),
            FadeInFromPoint(value_label2, np.array([-4, -5, 0])),
            TransformFromCopy(value_rect, diff_vect),
            run_time=2
        )
        self.wait()

        # Show blue
        blue_group.target = blue_group.generate_target()
        blue_group.target[0].set_stroke(opacity=1)
        blue_group.target[0].set_fill(opacity=0.2)
        blue_group.target[1:].set_opacity(1)
        blue_group.target.shift(0.2 * LEFT)

        blue_path = self.get_top_vect_to_low_vect_path(blue_group.target, in_vect, BLUE)
        blue_emb = blue_group[3]
        blue_in_vect = blue_emb.copy().set_opacity(1)
        blue_in_vect.replace(in_vect)

        self.fix_new_entries_in_frame = True
        self.play(
            MoveToTarget(blue_group),
            LaggedStartMap(FadeOut, VGroup(
                in_vect_path, in_vect_rect,
                rhs, value_rect, value_label,
                value_label2,
            )),
            run_time=1
        )
        self.play(
            TransformFromCopy(blue_emb, blue_in_vect),
            ShowCreation(blue_path),
            FadeOut(in_vect, 3 * DOWN),
            run_time=1.5
        )
        eq, rhs2 = show_matrix_vector_product(self, matrix, blue_in_vect)

        # Show in diagram
        diff2 = Arrow(
            with_fluffy_vect[0].get_end(),
            with_blue_vect[0].get_end(),
            buff=0.05
        )
        diff2.set_flat_stroke(False)
        rhs_rect = SurroundingRectangle(rhs2)
        rhs_rect.set_stroke(RED, 2)

        self.fix_new_entries_in_frame = True
        self.play(ShowCreation(rhs_rect))
        self.fix_new_entries_in_frame = False
        self.add(diff2)
        self.play(
            TransformFromCopy(rhs_rect, diff2),
            FadeIn(diff2),
            frame.animate.reorient(-16, -3, 0, (-6.41, 2.78, 1.37), 13.21),
            TransformFromCopy(with_fluffy_vect[0], with_blue_vect[0]),
            FadeTransform(with_fluffy_vect[1].copy(), with_blue_vect[1]),
            run_time=2,
        )
        frame.add_ambient_rotation(2 * DEGREES)
        self.wait(8)


    def get_top_vect_to_low_vect_path(self, top_vect, low_vect, color, top_buff=0.1, low_buff=0.2, bezier_factor=1.5):
        result = CubicBezier(
            top_vect.get_bottom() + top_buff * DOWN,
            top_vect.get_bottom() + bezier_factor * DOWN,
            low_vect.get_top() + bezier_factor * UP,
            low_vect.get_top() + low_buff * UP,
        )
        result.set_stroke(color, 3)
        return result

    def get_labeled_vector(self, axes, coords, color, image_name, image_height=1.0):
        vect = Arrow(axes.get_origin(), axes.c2p(*coords), buff=0)
        vect.set_color(color)
        image = ImageMobject(image_name)
        image.set_height(image_height)
        image.next_to(vect.get_end(), UP, MED_SMALL_BUFF)

        return Group(vect, image)

    def add(self, *mobjects):
        if self.fix_new_entries_in_frame:
            for mob in mobjects:
                mob.fix_in_frame()
        super().add(*mobjects)


class CountMatrixParameters(InteractiveScene):
    count_font_size = 36

    def construct(self):
        # Add three matrices
        d_embed = 12_288
        d_key = 128
        key_mat_shape = (5, 10)

        que_mat = WeightMatrix(shape=key_mat_shape)
        key_mat = WeightMatrix(shape=key_mat_shape)
        val_mat = WeightMatrix(shape=(key_mat_shape[1], key_mat_shape[1]))
        matrices = VGroup(que_mat, key_mat, val_mat)
        for matrix in matrices:
            matrix.set_max_width(4)

        matrices.arrange(DOWN, buff=0.75)

        colors = [YELLOW, TEAL, RED]

        titles = VGroup(Text("Query"), Text("Key"), Text("Value"))
        que_title, key_title, val_title = titles
        titles.arrange(DOWN, aligned_edge=LEFT)
        titles.next_to(matrices, LEFT, LARGE_BUFF)
        for title, matrix, color in zip(titles, matrices, colors):
            title.match_y(matrix)
            title.set_color(color)

        self.play(
            LaggedStartMap(FadeIn, titles, shift=0.25 * LEFT, lag_ratio=0.5),
            LaggedStart(
                (FadeIn(matrix, lag_ratio=1e-2)
                for matrix in matrices),
                lag_ratio=0.5,
            )
        )
        self.wait()

        # Data animations
        change_anims = [RandomizeMatrixEntries(mat) for mat in matrices]
        highlight_anims = [
            LaggedStartMap(FlashUnder, mat.get_entries(), lag_ratio=5e-3, stroke_width=1)
            for mat in matrices
        ]

        self.play(
            LaggedStart(highlight_anims, lag_ratio=0.2),
            LaggedStart(change_anims, lag_ratio=0.2),
            run_time=3
        )

        # Ask about total number of parameters
        rects = VGroup(
            SurroundingRectangle(entry, buff=0.025)
            for matrix in matrices
            for entry in matrix.get_entries()
        )
        rects.set_stroke(WHITE, 1)
        question = Text("How many\nparameters?")
        question.next_to(matrices, RIGHT, LARGE_BUFF)

        self.play(
            ShowCreation(rects, lag_ratio=5e-3, run_time=2),
            Write(question)
        )
        self.play(FadeOut(rects))
        self.wait()

        # Make room to count query/key
        value_group = VGroup(val_title, val_mat)
        value_group.save_state()
        qk_mats = matrices[:2]
        qk_mats.target = qk_mats.generate_target()
        qk_mats.target.arrange(RIGHT, buff=3.0)
        qk_mats.target.move_to(DR)

        self.play(
            FadeOut(question, DR),
            value_group.animate.scale(0.25).to_corner(DR).fade(0.25),
            MoveToTarget(qk_mats),
            que_title.animate.next_to(qk_mats.target[0], UP, buff=2.0),
            key_title.animate.next_to(qk_mats.target[1], UP, buff=2.0),
        )

        # Count up query and key
        que_col_count = self.show_column_count(que_mat, d_embed)
        key_col_count = self.show_column_count(key_mat, d_embed)
        self.wait()
        que_row_count = self.show_row_count(que_mat, d_key)
        key_row_count = self.show_row_count(key_mat, d_key)
        self.wait()

        que_product = self.show_product(
            que_col_count, que_row_count,
            added_anims=[que_title.animate.shift(UP)]
        )
        key_product = self.show_product(
            key_col_count, key_row_count,
            added_anims=[key_title.animate.shift(UP)]
        )
        self.wait()

        # Pull up the value matrix
        qk_titles = titles[:2]
        qk_titles.target = qk_titles.generate_target()
        qk_titles.target.arrange(DOWN, buff=2.0, aligned_edge=LEFT)
        qk_titles.target.to_corner(UL)
        qk_titles.target.scale(0.5, about_edge=UL)

        qk_mats.target = qk_mats.generate_target()

        qk_rhss = VGroup(que_product[-1], key_product[-1]).copy()
        qk_rhss.target = qk_rhss.generate_target()

        for mat, title, rhs in zip(qk_mats.target, qk_titles.target, qk_rhss.target):
            rhs.scale(0.5)
            mat.scale(0.5)
            rhs.next_to(title, DOWN, SMALL_BUFF, aligned_edge=LEFT)
            mat.next_to(VGroup(title, rhs), RIGHT, buff=MED_LARGE_BUFF)

        self.play(
            MoveToTarget(qk_titles),
            MoveToTarget(qk_mats),
            MoveToTarget(qk_rhss),
            FadeOut(VGroup(
                que_product, key_product,
                que_col_count, que_row_count,
                key_col_count, key_row_count,
            ), shift=0.5 * UL, lag_ratio=1e-3, time_span=(0, 1.0)),
            value_group.animate.restore().arrange(DOWN, buff=1.0).move_to(2.0 * RIGHT + 0.5 * DOWN),
            run_time=2,
        )
        self.wait()

        # Count up current value
        in_vect = NumericEmbedding(length=key_mat_shape[1])
        in_vect.match_height(val_mat)
        in_vect.next_to(val_mat, RIGHT, SMALL_BUFF)

        val_col_count = self.show_column_count(
            val_mat, d_embed,
            added_anims=[val_title.animate.shift(UP)]
        )
        self.play(FadeIn(in_vect))
        eq, rhs = show_matrix_vector_product(self, val_mat, in_vect)
        val_row_count = self.show_row_count(val_mat, d_embed)
        self.wait()
        val_product = self.show_product(
            val_col_count, val_row_count,
            added_anims=[val_title.animate.shift(UP)]
        )
        self.wait()

        # Compare the two
        frame = self.frame
        q_group, k_group = qk_groups = VGroup(
            VGroup(*trip)
            for trip in zip(qk_mats, qk_titles, qk_rhss)
        )
        for group, y in zip(qk_groups, [+1.25, -1.25]):
            group.save_state()
            group.target = group.generate_target()
            group.target.scale(2)
            group.target.next_to(val_mat, LEFT, buff=2.5)
            group.target.set_y(y)

        self.play(
            frame.animate.reorient(0, 0, 0, (-1.58, 0.02, 0.0), 9.22),
            LaggedStartMap(MoveToTarget, qk_groups),
        )
        self.wait()

        # Circle both
        val_rhs_rect = SurroundingRectangle(val_product[-1])
        val_rhs_rect.set_stroke(RED_B, 3)
        qk_rhs_rects = VGroup(
            SurroundingRectangle(rhs) for rhs in qk_rhss
        )
        qk_rhs_rects[0].set_stroke(YELLOW, 3)
        qk_rhs_rects[1].set_stroke(TEAL, 3)

        big_rect = FullScreenFadeRectangle()
        big_rect.scale(2)
        big_rect.set_fill(opacity=0.5)
        val_rhs_copy = val_product[-1].copy()
        qk_rhs_copies = qk_rhss.copy()

        self.add(big_rect, val_rhs_copy)
        self.play(
            FadeIn(big_rect),
            ShowCreation(val_rhs_rect)
        )
        self.wait()
        self.play(
            TransformFromCopy(VGroup(val_rhs_rect), qk_rhs_rects),
            FadeIn(qk_rhs_copies)
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeOut, VGroup(
                big_rect, qk_rhs_copies, val_rhs_copy,
                qk_rhs_rects, val_rhs_rect
            ))
        )

        # Cross out
        cross = Cross(val_product, stroke_width=[0, 12, 0]).scale(1.1)
        self.play(LaggedStart(
            FadeOut(qk_groups, 2 * UR, scale=0.5),
            ShowCreation(cross),
            frame.animate.set_height(FRAME_HEIGHT).move_to(RIGHT),
            run_time=2,
            lag_ratio=0.1
        ))
        self.wait()
        self.play(FadeOut(val_product), FadeOut(cross))

        # Factor out
        val_down_mat = WeightMatrix(shape=key_mat_shape)
        val_up_mat = WeightMatrix(shape=(key_mat_shape[1], 4))
        val_down_mat.match_width(val_mat)
        val_up_mat.match_height(in_vect)

        val_down_mat.move_to(val_mat, RIGHT)
        val_up_mat.next_to(val_down_mat, LEFT, SMALL_BUFF)

        self.remove(val_mat)
        self.play(
            TransformFromCopy(val_mat.get_brackets(), val_down_mat.get_brackets()),
            TransformFromCopy(val_mat.get_columns(), val_down_mat.get_columns()),
            TransformFromCopy(val_mat.get_brackets(), val_up_mat.get_brackets()),
            TransformFromCopy(val_mat.get_rows(), val_up_mat.get_rows()),
            val_col_count.animate.next_to(val_down_mat, UP, SMALL_BUFF),
            val_row_count.animate.next_to(val_up_mat, LEFT, SMALL_BUFF),
        )
        self.add(val_down_mat)
        self.wait()

        # Circle the full linear map
        big_rect = SurroundingRectangle(VGroup(val_row_count, val_col_count))
        big_rect.round_corners(radius=0.25)
        big_rect.set_stroke(RED_B, 2)
        linear_map_words = Text("Linear map")
        linear_map_words.next_to(big_rect, UP)
        linear_map_words.set_color(RED_B)

        in_label, out_label = [
            VGroup(Text(text), Integer(d_embed))
            for text in ["d_input", "d_output"]
        ]
        for label, array, shift in [(in_label, in_vect, LEFT), (out_label, rhs, RIGHT)]:
            label.arrange(DOWN)
            label.scale(0.65)
            label.next_to(array, UP, buff=LARGE_BUFF)
            label.shift(0.25 * shift)
            arrow = Arrow(label, array)
            label.add(arrow)

        self.play(
            FadeIn(big_rect),
            FadeTransform(val_title, linear_map_words),
        )
        self.wait()
        self.play(FadeIn(in_label, lag_ratio=0.1))
        self.play(FadeIn(out_label, lag_ratio=0.1))
        self.wait(2)

        # Show the value_down map
        val_down_group = VGroup(val_down_mat, val_col_count)
        val_up_group = VGroup(val_up_mat, val_row_count)
        val_down_group.save_state()
        val_up_group.save_state()

        small_row_count = self.show_row_count(
            val_down_mat, d_key,
            added_anims=[val_up_group.animate.scale(0.5).to_edge(LEFT, buff=1.25).fade(0.5)]
        )
        self.wait()
        self.play(frame.animate.set_y(0.5))
        self.wait()

        value_down_rect = SurroundingRectangle(
            VGroup(small_row_count, val_down_mat, val_col_count)
        )
        value_down_rect.round_corners(radius=0.25)
        value_down_rect.set_stroke(RED_B, 2)
        value_down_title = TexText(R"Value$_\downarrow$")
        value_down_title.set_fill(RED_B)
        value_down_title.next_to(val_down_mat, DOWN)

        self.remove(big_rect)
        self.play(
            TransformFromCopy(big_rect, value_down_rect),
            FadeOut(linear_map_words),
            FadeIn(value_down_title, DOWN)
        )
        self.wait()

        # Show value_up map
        small_row_count.target = small_row_count.generate_target()
        small_row_count.target.rotate(-PI / 2)
        small_row_count.target[1].rotate(PI / 2)
        small_row_count.target[0].stretch_to_fit_width(val_up_group.saved_state[0].get_width())
        small_row_count.target[1].next_to(small_row_count.target[0], UP, SMALL_BUFF)
        small_row_count.target.next_to(val_up_group.saved_state[0], UP, SMALL_BUFF)
        big_rect.set_height(3.9, stretch=True)
        big_rect.align_to(VGroup(val_down_mat, val_up_group.saved_state), DR)
        big_rect.shift(0.8 * DOWN + 0.05 * RIGHT)
        linear_map_words.next_to(big_rect, UP)

        value_up_title = TexText(R"Value$_\uparrow$")
        value_up_title.set_fill(RED_B)
        value_up_title.next_to(val_up_group.saved_state[0], DOWN)

        self.play(LaggedStart(
            val_down_group.animate.fade(0.5),
            value_down_title.animate.fade(0.5),
            ReplacementTransform(value_down_rect, big_rect),
            Restore(val_up_group),
            MoveToTarget(small_row_count),
            FadeIn(linear_map_words, shift=0.5 * UP),
            run_time=2,
        ))
        val_up_group.add(small_row_count)
        self.wait()
        self.play(TransformFromCopy(value_down_title, value_up_title))
        self.wait()

        # Low rank label
        low_rank_words = TexText("``Low rank'' transformation")
        low_rank_words.next_to(big_rect, UP)
        low_rank_words.shift(0.5 * LEFT)
        self.play(
            val_down_group.animate.set_fill(opacity=1),
            value_down_title.animate.set_fill(opacity=1),
            FadeTransform(linear_map_words, low_rank_words)
        )
        self.wait()

    def scrap(self):
        # Label the value matrix
        tiny_buff = 0.025
        value_rect = SurroundingRectangle(val_down_group, buff=tiny_buff)
        value_rect.stretch(1.2, 1)
        value_rect.round_corners(0.1)
        value_rect.set_stroke(RED, 3)
        value_arrow = Vector(DOWN)
        value_arrow.match_color(value_rect)
        value_arrow.next_to(value_rect, UP, SMALL_BUFF)

        val_up_group.save_state()
        out_rect = SurroundingRectangle(val_up_group, buff=tiny_buff)
        out_rect.set_height(big_rect.get_height() - SMALL_BUFF, stretch=True)
        out_rect.match_y(big_rect)
        out_rect.round_corners(0.1)
        out_rect.set_stroke(PINK, 3)
        out_arrow = Vector(0.5 * DOWN)
        out_arrow.next_to(out_rect, UP, SMALL_BUFF)
        out_arrow.match_color(out_rect)
        output_title = TexText("Output$^{*}$")
        output_title.match_color(out_rect)
        output_title.next_to(out_arrow, UP, SMALL_BUFF)


        self.play(LaggedStart(
            Restore(val_down_group),
            LaggedStartMap(FadeOut, VGroup(in_label, out_label)),
            TransformFromCopy(big_rect, value_rect),
            FadeOut(linear_map_words),
            val_title.animate.next_to(value_arrow, UP, SMALL_BUFF),
            FadeIn(value_arrow, shift=DOWN),
            val_up_group.animate.fade(0.5),
        ))
        self.wait()
        self.play(LaggedStart(
            TransformFromCopy(big_rect, out_rect),
            TransformFromCopy(value_arrow, out_arrow),
            FadeTransform(val_title.copy(), output_title),
            Restore(val_up_group),
        ))
        self.wait()

    def show_column_count(self, matrix, count, added_anims=[]):
        cols = matrix.get_columns()
        col_rects = VGroup(SurroundingRectangle(cols[0], buff=0).match_x(col) for col in cols)
        col_rects.set_stroke(WHITE, 1, 0.5)
        col_rects.set_fill(GREY_D, 0.5)
        top_brace = Brace(col_rects, UP, buff=SMALL_BUFF)
        count_mob = Integer(count, font_size=self.count_font_size)
        count_mob.next_to(top_brace, UP)

        self.play(
            GrowFromCenter(top_brace),
            CountInFrom(count_mob, 0),
            FadeIn(col_rects, lag_ratio=0.25),
            *added_anims,
        )
        self.play(FadeOut(col_rects))
        return VGroup(top_brace, count_mob)

    def show_row_count(self, matrix, count, added_anims=[]):
        rows = matrix.get_rows()
        row_rects = VGroup(SurroundingRectangle(rows[0], buff=0).match_y(row) for row in rows)
        row_rects.set_stroke(WHITE, 1, 0.5)
        row_rects.set_fill(GREY_D, 0.5)
        left_brace = Brace(matrix, LEFT, buff=SMALL_BUFF)
        count_mob = Integer(count, font_size=self.count_font_size)
        count_mob.next_to(left_brace, LEFT)

        self.play(
            GrowFromCenter(left_brace),
            CountInFrom(count_mob, 0),
            FadeIn(row_rects, lag_ratio=0.25),
            *added_anims,
        )
        self.play(FadeOut(row_rects))
        return VGroup(left_brace, count_mob)

    def show_product(self, col_count, row_count, added_anims=[]):
        col_dec = col_count[1]
        row_dec = row_count[1]
        prod_dec = Integer(
            col_dec.get_value() * row_dec.get_value(),
            font_size=self.count_font_size
        )

        equation = VGroup(
            row_dec.copy(),
            Tex(R"\times", font_size=self.count_font_size),
            col_dec.copy(),
            Tex(R"=", font_size=self.count_font_size),
            prod_dec
        )
        equation.arrange(RIGHT,buff=SMALL_BUFF)
        for index in [0, 2]:
            equation[index].align_to(equation[4], UP)
        equation.next_to(col_dec, UP, buff=1.0)

        self.play(
            TransformFromCopy(row_dec, equation[0]),
            FadeIn(equation[1]),
            TransformFromCopy(col_dec, equation[2]),
            FadeIn(equation[3]),
            *added_anims
        )
        self.play(
            FadeTransform(equation[0].copy(), equation[4]),
            FadeTransform(equation[2].copy(), equation[4]),
        )
        self.add(equation)
        return equation


class LowRankTransformation(InteractiveScene):
    def construct(self):
        # Add three sets of axes
        frame = self.frame
        frame.set_field_of_view(10 * DEGREES)

        all_axes = VGroup(
            self.get_3d_axes(),
            self.get_2d_axes(),
            self.get_3d_axes(),
        )
        all_axes.arrange(RIGHT, buff=2.0)
        all_axes.set_width(FRAME_WIDTH - 2)
        all_axes.move_to(0.5 * DOWN)
        dim_labels = VGroup(
            Text("12,288 dims"),
            Text("128 dims"),
            Text("12,288 dims"),
        )
        dim_labels.scale(0.75)
        dim_labels.set_fill(GREY_A)
        for label, axes in zip(dim_labels, all_axes):
            label.next_to(axes, UP, buff=MED_LARGE_BUFF)

        map_arrows = Tex(R"\rightarrow", font_size=96).replicate(2)
        map_arrows.set_color(YELLOW)
        for arrow, vect in zip(map_arrows, [LEFT, RIGHT]):
            arrow.next_to(all_axes[1], vect, buff=0.5)

        axes_group = VGroup(all_axes, dim_labels)
        self.add(axes_group)
        self.add(map_arrows)

        # Add vectors
        all_coords = [
            (4, 2, 1),
            (2, 3),
            (-3, 3, -2),
        ]
        colors = [BLUE, RED_B, RED_C]
        vects = VGroup(
            Arrow(axes.get_origin(), axes.c2p(*coords), buff=0, stroke_color=color)
            for axes, coords, color in zip(all_axes, all_coords, colors)
        )

        self.add(vects[0])
        for v1, v2 in zip(vects, vects[1:]):
            self.play(TransformFromCopy(v1, v2))

        for axes, vect in zip(all_axes, vects):
            axes.add(vect)
        for axes in all_axes[0::2]:
            axes.add_updater(lambda m, dt: m.rotate(2 * dt * DEGREES, axis=m.y_axis.get_vector()))
        self.wait(3)

        # Add title
        big_rect = SurroundingRectangle(axes_group, buff=0.5)
        big_rect.round_corners(radius=0.5)
        big_rect.set_stroke(RED_B, 2)
        title = Text("Low-rank transformation", font_size=72)
        title.next_to(big_rect, UP, buff=MED_LARGE_BUFF)

        self.play(
            ShowCreation(big_rect),
            FadeIn(title, shift=0.25 * UP)
        )
        self.wait(5)


    def get_3d_axes(self, height=3):
        result = ThreeDAxes((-4, 4), (-4, 4), (-4, 4))
        result.set_height(height)
        result.rotate(20 * DEGREES, DOWN)
        result.rotate(5 * DEGREES, RIGHT)
        return result

    def get_2d_axes(self, height=2):
        plane = NumberPlane(
            (-4, 4), (-4, 4),
            faded_line_ratio=0,
            background_line_style=dict(
                stroke_color=GREY_B,
                stroke_width=1,
                stroke_opacity=0.5
            )
        )
        plane.set_height(height)
        return plane


class ThinkAboutOverallMap(InteractiveScene):
    def construct(self):
        # Test
        rect = Rectangle(6.5, 2.75)
        rect.round_corners(radius=0.5)
        rect.set_stroke(RED_B, 2)
        label = Text("Think about the\noverall map")
        label.next_to(rect, UP, aligned_edge=LEFT)
        label.shift(0.5 * RIGHT)
        self.play(
            ShowCreation(rect),
            FadeIn(label, UP),
        )
        self.wait()


class CrossAttention(InteractiveScene):
    def construct(self):
        # Show both
        en_tokens = self.get_words("I do not want to pet it")
        fr_tokens = self.get_words("Je ne veux pas le caresser", hue_range=(0.2, 0.3))
        phrases = VGroup(en_tokens, fr_tokens)
        phrases.arrange(DOWN, buff=2.0)
        self.play(LaggedStartMap(FadeIn, en_tokens, scale=2, lag_ratio=0.25))
        self.wait()
        self.play(LaggedStartMap(FadeIn, fr_tokens, scale=2, lag_ratio=0.25))
        self.wait()

        # Create attention pattern
        unnormalized_pattern = [
            [3, 0, 0, 0, 0, 0],
            [0, 1, 1.3, 1, 0, 0],
            [0, 3, 0, 3, 0, 0],
            [0, 0, 3, 0, 0, 0],
            [0, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 3, 0],
        ]
        attention_pattern = np.array([
            softmax(col) for col in unnormalized_pattern
        ]).T

        # Show connections
        lines = VGroup()
        for n, row in enumerate(attention_pattern.T):
            for k, value in enumerate(row):
                line = Line(en_tokens[n].get_bottom(), fr_tokens[k].get_top(), buff=0)
                line.set_stroke(
                    color=[
                        en_tokens[n][0].get_color(),
                        fr_tokens[k][0].get_color(),
                    ],
                    width=3,
                    opacity=value,
                )
                lines.add(line)

        self.play(ShowCreation(lines, lag_ratio=0.01, run_time=2))
        self.wait(2)
        self.play(FadeOut(lines))

        # Create grid
        grid = Square().get_grid(len(fr_tokens), len(en_tokens), buff=0)
        grid.stretch(1.2, 0)
        grid.set_stroke(GREY_B, 1)
        grid.set_height(5.0)
        grid.to_edge(DOWN, buff=SMALL_BUFF)
        grid.set_x(1)

        # Create qk symbols
        q_sym_generator = self.get_symbol_generator(R"\vec{\textbf{Q}}_0", color=YELLOW)
        k_sym_generator = self.get_symbol_generator(R"\vec{\textbf{K}}_0", color=TEAL)
        e_sym_generator = self.get_symbol_generator(R"\vec{\textbf{E}}_0", color=GREY_B)
        f_sym_generator = self.get_symbol_generator(R"\vec{\textbf{F}}_0", color=BLUE)

        q_syms = VGroup(q_sym_generator(n + 1) for n in range(len(en_tokens)))
        k_syms = VGroup(k_sym_generator(n + 1) for n in range(len(fr_tokens)))
        e_syms = VGroup(e_sym_generator(n + 1) for n in range(len(en_tokens)))
        f_syms = VGroup(f_sym_generator(n + 1) for n in range(len(fr_tokens)))
        VGroup(q_syms, k_syms, e_syms, f_syms).scale(0.65)

        for q_sym, e_sym, square in zip(q_syms, e_syms, grid):
            q_sym.next_to(square, UP, SMALL_BUFF)
            e_sym.next_to(q_sym, UP, buff=0.65)

        for k_sym, f_sym, square in zip(k_syms, f_syms, grid[::len(en_tokens)]):
            k_sym.next_to(square, LEFT, SMALL_BUFF)
            f_sym.next_to(k_sym, LEFT, buff=0.75)

        q_arrows = VGroup(Arrow(*pair, buff=0.1) for pair in zip(e_syms, q_syms))
        k_arrows = VGroup(Arrow(*pair, buff=0.1) for pair in zip(f_syms, k_syms))
        e_arrows = VGroup(Vector(0.4 * DOWN).next_to(e_sym, UP, SMALL_BUFF) for e_sym in e_syms)
        f_arrows = VGroup(Vector(0.5 * RIGHT).next_to(f_sym, LEFT, SMALL_BUFF) for f_sym in f_syms)
        arrows = VGroup(q_arrows, k_arrows, e_arrows, f_arrows)
        arrows.set_color(GREY_B)

        wq_syms = VGroup(
            Tex("W_Q", font_size=20, fill_color=YELLOW).next_to(arrow, RIGHT, buff=0.1)
            for arrow in q_arrows
        )
        wk_syms = VGroup(
            Tex("W_K", font_size=20, fill_color=TEAL).next_to(arrow, UP, buff=0.1)
            for arrow in k_arrows
        )

        # Move tokens into place
        en_tokens.target = en_tokens.generate_target()
        fr_tokens.target = fr_tokens.generate_target()
        for token, arrow in zip(en_tokens.target, e_arrows):
            token.next_to(arrow, UP, SMALL_BUFF)
        for token, arrow in zip(fr_tokens.target, f_arrows):
            token.next_to(arrow, LEFT, SMALL_BUFF)
        self.play(
            MoveToTarget(en_tokens),
            MoveToTarget(fr_tokens),
        )
        self.play(
            LaggedStartMap(GrowArrow, e_arrows),
            LaggedStartMap(GrowArrow, f_arrows),
            LaggedStartMap(FadeIn, e_syms, shift=0.25 * DOWN),
            LaggedStartMap(FadeIn, f_syms, shift=0.25 * RIGHT),
            lag_ratio=0.25,
            run_time=1.5,
        )
        self.play(
            LaggedStartMap(GrowArrow, q_arrows),
            LaggedStartMap(GrowArrow, k_arrows),
            LaggedStartMap(FadeIn, wq_syms, shift=0.25 * DOWN),
            LaggedStartMap(FadeIn, wk_syms, shift=0.25 * RIGHT),
            LaggedStartMap(FadeIn, q_syms, shift=0.5 * DOWN),
            LaggedStartMap(FadeIn, k_syms, shift=0.5 * RIGHT),
            lag_ratio=0.25,
            run_time=1.5,
        )
        self.play(FadeIn(grid, lag_ratio=1e-2), run_time=2)
        self.wait()

        # Show dot products
        dot_prods = VGroup()
        for q_sym in q_syms:
            for k_sym in k_syms:
                dot = Tex(".")
                dot.match_x(q_sym)
                dot.match_y(k_sym)
                dot_prod = VGroup(q_sym.copy(), dot, k_sym.copy())
                dot_prod.target = dot_prod.generate_target()
                dot_prod.target.arrange(RIGHT, buff=SMALL_BUFF)
                dot_prod.target.scale(0.7)
                dot_prod.target.move_to(dot)
                dot.set_opacity(0)
                dot_prods.add(dot_prod)

        self.play(
            LaggedStartMap(MoveToTarget, dot_prods, lag_ratio=0.01),
            run_time=3
        )
        self.wait()

        # Show dots
        dots = VGroup()
        for square, value in zip(grid, attention_pattern.flatten()):
            dot = Dot(radius=value * 0.4)
            dot.set_fill(GREY_B, 1)
            dot.move_to(square)
            dots.add(dot)


        self.play(
            LaggedStartMap(GrowFromCenter, dots, lag_ratio=1e-2),
            dot_prods.animate.set_fill(opacity=0.2).set_anim_args(lag_ratio=1e-3),
            run_time=4
        )
        self.wait()




        pass

    def get_words(self, text, hue_range=(0.5, 0.6)):
        sent = Text(text)
        tokens = break_into_words(sent)
        rects = get_piece_rectangles(
            tokens, hue_range=hue_range,
            # h_buff=0, leading_spaces=True
        )
        return VGroup(VGroup(*pair) for pair in zip(rects, tokens))

    def get_symbol_generator(self, raw_tex, subsrc="0", color=WHITE):
        template = Tex(raw_tex)
        template.set_color(color)
        subscr = template.make_number_changeable(subsrc)

        def get_sym(number):
            subscr.set_value(number)
            return template.copy()

        return get_sym


class CarCrashedExample(InteractiveScene):
    def construct(self):
        # Add sentence
        sentence = Text("... when suddenly they crashed the car into a tree ...")
        words = break_into_words(sentence)
        rects = get_piece_rectangles(words)
        word_groups = VGroup(VGroup(*pair) for pair in zip(rects, words))

        car = word_groups[6]
        crashed = VGroup(*it.chain(*(wg[1] for wg in word_groups[3:6])))
        arrow = Vector(UP).next_to(car, UP, SMALL_BUFF)

        self.play(LaggedStartMap(FadeIn, word_groups, shift=0.25 * UP, lag_ratio=0.25))
        self.play(
            word_groups[:3].animate.fade(0.5),
            word_groups[7:].animate.fade(0.5),
            FadeIn(arrow),
        )
        self.wait()

        # Influence
        self.play(ContextAnimation(car, crashed, direction=DOWN, run_time=5))
        self.wait()


class TwoHarrysExample(InteractiveScene):
    def construct(self):
        # Test
        s1, s2 = sentences = VGroup(
            break_into_words(Text("... " + " ... ".join(words)))
            for words in [
                ("wizard", "Hogwarts", "Hermione", "Harry"),
                ("Queen", "Sussex", "William", "Harry"),
            ]
        )
        sentences.arrange(DOWN, buff=2.0, aligned_edge=RIGHT)
        sentences.to_edge(LEFT)

        def context_anim(group):
            self.play(
                ContextAnimation(
                    group[-1],
                    VGroup(*it.chain(*group[1:-1:2])),
                    direction=DOWN,
                    path_arc=PI / 4,
                    run_time=5,
                    lag_ratio=0.025,
                )
            )

        self.add(s1)
        context_anim(s1)
        self.wait()
        self.play(FadeTransformPieces(s1.copy(), s2))
        context_anim(s2)


class ManyTypesOfUpdates(InteractiveScene):
    def construct(self):
        # Add matrices
        shapes = [(4, 8), (4, 8), (8, 4), (4, 8)]
        names = ["W_Q", "W_K", R"\uparrow W_V", R"\downarrow W_V"]
        colors = [YELLOW, TEAL, RED_B, RED_C]

        matrices = VGroup(
            WeightMatrix(shape=shape)
            for shape in shapes
        )
        buff_ratio = 0.35
        matrices.arrange(RIGHT, buff=matrices[0].get_width() * buff_ratio)
        matrices[-1].next_to(matrices[-2], RIGHT, buff=matrices[-2].get_width() * 0.1)
        matrices.center()
        matrices.set_width(FRAME_WIDTH - 2)
        matrices.to_edge(UP, buff=1.0)
        titles = VGroup(
            Tex(name).set_color(color).match_x(mat).to_edge(UP, buff=MED_SMALL_BUFF)
            for name, color, mat in zip(names, colors, matrices)
        )
        for title in titles[2:]:
            title[0].next_to(title[1], LEFT, buff=0.5 * SMALL_BUFF)

        self.add(matrices, titles)

        # Add phrase
        phrase = Text("John hit the brakes sharply, they screeched loudly, and he jolted forward.")
        raw_words = break_into_words(phrase)
        rects = get_piece_rectangles(raw_words)
        rects.fade(0.5)
        words = VGroup(VGroup(*pair) for pair in zip(rects, raw_words))
        words.set_width(FRAME_WIDTH - 1)
        words.center().set_y(-2)

        self.add(words)

        labels = index_labels(words)
        labels.shift(0.5 * DOWN)

        # Set up association types
        attention_types = [
            (
                "Adverb to verb",
                [
                    (1, 4, 1.0), 
                    (6, 7, 1.0), 
                ]
            ),
            (
                "Subject to verb",
                [
                    (0, 1, 1.0),
                    (3, 6, 0.5),
                    (5, 6, 0.5),
                    (0, 10, 0.5),
                    (9, 10, 0.5),
                ],
            ),
            (
                "Antecedent to pronoun",
                [
                    (0, 9, 1.0),
                    (3, 5, 1.0),
                ]
            ),
            (
                "Related to the subject",
                [
                    (0, 1, 0.25),
                    (0, 3, 0.25),
                    (0, 9, 0.2),
                    (0, 10, 0.2),
                    (0, 11, 0.2),
                ]
            ),
            (
                "Related to the object",
                [
                    (3, 4, 0.2),
                    (3, 5, 0.5),
                    (3, 6, 0.35),
                    (3, 7, 0.2),
                ]
            ),
        ]

        # Animate
        last_group = VGroup()
        for description, connections in attention_types:
            desc = Text(description)
            desc.center()
            connections = VGroup(
                Line(
                    words[i].get_top(),
                    words[j].get_top(),
                    path_arc=-PI / 2,
                    stroke_color=random_bright_color(
                        hue_range=(0.3, 0.5),
                        luminance_range=(0.5, 0.7),
                    ),
                    stroke_opacity=strength**0.5,
                )
                for (i, j, strength) in connections
            )
            connections.set_stroke(width=(0, 5, 5, 5, 0))
            connections.shuffle()
            self.play(
                FadeOut(last_group),
                # FadeIn(desc, shift=0.25 * UP),
                ShowCreation(connections, lag_ratio=0.25, run_time=0.5 * len(connections)),
                LaggedStart(
                    (self.get_matrix_update_anim(mat)
                    for mat in matrices),
                    lag_ratio=0.15,
                ),
                LaggedStart(
                    (VShowPassingFlash(
                        line.copy().insert_n_curves(100).set_stroke(width=10),
                        time_width=2.0,
                        run_time=2,
                    )
                    for line in connections),
                    lag_ratio=0.1,
                )
            )
            self.wait(2)
            # last_group = VGroup(desc, connections)
            last_group = VGroup(connections)

    def get_matrix_update_anim(self, matrix):
        rects = VGroup(
            Underline(entry, buff=0.05)
            for entry in matrix.get_entries()
        )
        rects.set_stroke(WHITE, 1)
        return AnimationGroup(
            LaggedStartMap(ShowCreationThenFadeOut, rects, lag_ratio=1e-2),
            RandomizeMatrixEntries(matrix)
        )


class MultiHeadedAttention(InteractiveScene):
    def construct(self):
        # Mention head
        background_rect = FullScreenRectangle()
        single_title = Text("Single head of attention")
        multiple_title = Text("Multi-headed attention")
        titles = VGroup(single_title, multiple_title)
        for title in titles:
            title.scale(1.25)
            title.to_edge(UP)

        screen_rect = ScreenRectangle(height=6)
        screen_rect.set_fill(BLACK, 1)
        screen_rect.set_stroke(WHITE, 3)
        screen_rect.next_to(titles, DOWN, buff=0.5)

        head = single_title["head"][0]

        self.add(background_rect)
        self.add(single_title)
        self.add(screen_rect)
        self.wait()
        self.play(
            FlashAround(head, run_time=2),
            head.animate.set_color(YELLOW),
        )
        self.wait()

        # Change title
        kw = dict(path_arc=45 * DEGREES)
        self.play(
            FadeTransform(single_title["Single"], multiple_title["Multi-"], **kw),
            FadeTransform(single_title["head"], multiple_title["head"], **kw),
            FadeIn(multiple_title["ed"], 0.25 * RIGHT),
            FadeTransform(single_title["attention"], multiple_title["attention"], **kw),
            FadeOut(single_title["of"])
        )
        self.add(multiple_title)

        # Set up images
        n_heads = 15
        directory = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2024/transformers/attention/images/"
        heads = Group()
        for n in range(n_heads):
            im = ImageMobject(os.path.join(directory, f"AttentionPattern{n % 4 + 1}"))
            im.set_opacity(1)
            im.shift(0.01 * OUT)
            rect = SurroundingRectangle(im, buff=0)
            rect.set_fill(BLACK, 0.75)
            rect.set_stroke(WHITE, 1, 1)
            heads.add(Group(rect, im))

        # Show many parallel layers
        self.set_floor_plane("xz")
        frame = self.frame
        multiple_title.fix_in_frame()
        background_rect.fix_in_frame()

        heads.set_height(4)
        heads.arrange(OUT, buff=1.0)
        heads.move_to(DOWN)
        pre_head = ImageMobject(os.path.join(directory, f"AttentionPattern0"))

        pre_head.replace(screen_rect)
        pre_head = Group(screen_rect, pre_head)

        self.add(pre_head)
        self.wait()
        self.play(
            frame.animate.reorient(41, -12, 0, (-1.0, -1.42, 1.09), 12.90).set_anim_args(run_time=2),
            background_rect.animate.set_fill(opacity=0.75),
            FadeTransform(pre_head, heads[-1], time_span=(1, 2)),
        )
        self.play(
            frame.animate.reorient(48, -11, 0, (-1.0, -1.42, 1.09), 12.90),
            LaggedStart(
                (FadeTransform(heads[-1].copy(), image)
                for image in heads),
                lag_ratio=0.1,
                group_type=Group,
            ),
            run_time=4,
        )
        self.add(heads)
        self.wait()

        # Show matrices
        colors = [YELLOW, TEAL, RED, PINK]
        texs = ["W_Q", "W_K", R"\downarrow W_V", R"\uparrow W_V"]
        n_shown = 9
        wq_syms, wk_syms, wv_down_syms, wv_up_syms = sym_groups = VGroup(
            VGroup(
                Tex(tex + f"^{{({n})}}", font_size=36).next_to(image, UP, MED_SMALL_BUFF)
                for n, image in enumerate(heads[:-n_shown - 1:-1], start=1)
            ).set_color(color).set_backstroke(BLACK, 5)
            for tex, color in zip(texs, colors)
        )
        for group in wv_down_syms, wv_up_syms:
            for sym in group:
                sym[0].next_to(sym[1], LEFT, buff=0.025)
        dots = Tex(R"\dots", font_size=90)
        dots.rotate(PI / 2, UP)
        sym_rot_angle = 70 * DEGREES
        for syms in sym_groups:
            syms.align_to(heads, LEFT)
            for sym in syms:
                sym.rotate(sym_rot_angle, UP)
            dots.next_to(syms, IN, buff=0.5)
            dots.match_style(syms[0])
            syms.add(dots.copy())

        up_shift = 0.75 * UP
        self.play(
            LaggedStartMap(FadeIn, wq_syms, shift=0.2 * UP, lag_ratio=0.25),
            frame.animate.reorient(59, -7, 0, (-1.62, 0.25, 1.29), 14.18),
            run_time=2,
        )
        for n in range(1, len(sym_groups)):
            self.play(
                LaggedStartMap(FadeIn, sym_groups[n], shift=0.2 * UP, lag_ratio=0.1),
                sym_groups[:n].animate.shift(up_shift),
                run_time=1,
            )
        self.wait()

        # Count up 96 heads
        depth = heads.get_depth()
        brace = Brace(Line(LEFT, RIGHT).set_width(0.5 * depth), UP).scale(2)
        brace_label = brace.get_text("96", font_size=96, buff=MED_SMALL_BUFF)
        brace_group = VGroup(brace, brace_label)
        brace_group.rotate(PI / 2, UP)
        brace_group.next_to(heads, UP, buff=MED_LARGE_BUFF)

        self.add(brace, brace_label, sym_groups)
        self.play(
            frame.animate.reorient(62, -6, 0, (-0.92, -0.08, -0.51), 14.18).set_anim_args(run_time=5),
            GrowFromCenter(brace),
            sym_groups.animate.set_fill(opacity=0.5).set_stroke(width=0),
            FadeIn(brace_label, 0.5 * UP, time_span=(0.5, 1.5)),
        )

        # Set up pure attention patterns, flattened
        for head in heads:
            n_rows = 8
            grid = Square().get_grid(n_rows, 1, buff=0).get_grid(1, n_rows, buff=0)
            grid.set_stroke(WHITE, 1, 0.5)
            grid.set_height(0.9 * head.get_height())
            grid.move_to(head)

            pattern = np.random.normal(0, 1, (n_rows, n_rows))
            for n in range(len(pattern[0])):
                pattern[:, n][n + 1:] = -np.inf
                pattern[:, n] = softmax(pattern[:, n])
            pattern = pattern.T

            dots = VGroup()
            for col, values in zip(grid, pattern):
                for square, value in zip(col, values):
                    if value < 1e-3:
                        continue
                    dot = Dot(radius=0.4 * square.get_height() * value)
                    dot.move_to(square)
                    dots.add(dot)
            dots.set_fill(GREY_B, 1)
            grid.add(dots)

            head.add(grid)
            head.target = head.generate_target()
            grid.set_opacity(0)
            head.target[1].set_opacity(0)
            head.target[0].set_opacity(1)

        n_shown = 4
        heads_target = Group(h.target for h in heads)
        heads_target.arrange(LEFT, buff=MED_LARGE_BUFF)
        heads_target.set_height(1.5)
        heads_target.to_edge(LEFT)
        heads_target.shift(2 * UP)
        heads_target[:-n_shown].set_opacity(0)

        # Set up key/query targets
        for group in sym_groups:
            group.generate_target()
        group_targets = [group.target for group in sym_groups]

        for head, wq, wk, wv_down, wv_up in zip(heads_target[::-1], *group_targets):
            for sym in [wq, wk, wv_down, wv_up]:
                sym.set_fill(opacity=1)
                sym.set_height(0.35)
                sym.rotate(-sym_rot_angle, UP)
            wk.next_to(head, UP, aligned_edge=LEFT)
            wq.next_to(wk, RIGHT, buff=0.35)
            wv_up.next_to(head, UP, aligned_edge=LEFT)
            wv_down.next_to(wv_up, RIGHT, buff=0.35)

        for group in group_targets:
            group[n_shown:].set_opacity(0)

        # Animate the flattening
        right_dots = Tex(R"\dots", font_size=96)
        right_dots.move_to(heads_target[-n_shown - 1], LEFT).shift(MED_SMALL_BUFF * RIGHT)

        brace_group.target = brace_group.generate_target()
        brace_group.target.shift(UP)
        brace_group.target.set_opacity(0)

        self.play(
            frame.animate.reorient(0, 0, 0, ORIGIN, FRAME_HEIGHT).set_anim_args(run_time=2),
            FadeOut(multiple_title, UP),
            MoveToTarget(brace_group, remover=True),
            MoveToTarget(wq_syms, time_span=(0.5, 2)),
            MoveToTarget(wk_syms, time_span=(0.5, 2)),
            FadeOut(wv_down_syms),
            FadeOut(wv_up_syms),
            LaggedStartMap(MoveToTarget, heads, lag_ratio=0.01),
            Write(right_dots, time_span=(1.5, 2.0)),
        )

        att_patterns = VGroup(
            VGroup(head[0], head[2])
            for head in heads[:len(heads) - n_shown - 1:-1]
        )
        self.remove(heads)
        self.add(att_patterns)

        # Show value maps
        for group in [wv_up_syms, wv_down_syms]:
            group.become(group.target)

        value_diagrams = VGroup()
        arrows = VGroup()
        all_v_stacks = VGroup()
        for pattern, wv_up, wv_down, idx in zip(att_patterns, wv_up_syms, wv_down_syms, it.count(1)):
            rect = pattern[0].copy()

            v_stack = VGroup(Tex(Rf"\vec{{\textbf{{v}}}}_{n}") for n in range(1, 4))
            v_stack.arrange(DOWN, buff=LARGE_BUFF)
            v_stack.set_color(RED)
            plusses = VGroup()
            coefs = VGroup()
            for n, v_term in enumerate(v_stack):
                coef = Tex(f"w_{n + 1}")
                coef.next_to(v_term, LEFT, SMALL_BUFF)
                coef.set_fill(GREY_B)
                plus = Tex("+")
                plus.next_to(VGroup(coef, v_term), DOWN)
                plusses.add(plus)
                coefs.add(coef)
            dots = Tex(R"\vdots")
            dots.next_to(plusses, DOWN)
            v_stack.add(coefs, plusses, dots)

            v_stacks = v_stack.replicate(4)
            v_stacks.arrange(RIGHT, buff=LARGE_BUFF)
            v_stacks.set_height(rect.get_height() * 0.85)
            v_stacks.set_fill(border_width=1)

            v_terms = VGroup(
                *(Tex(Rf"\vec{{\textbf{{v}}}}_{n}^{{({idx})}}") for n in range(1, 4)),
                Tex(R"\dots")
            )
            v_terms[:3].set_color(RED)
            v_terms.arrange(RIGHT)
            v_terms.set_width(0.8 * rect.get_width())
            v_terms.move_to(rect)

            diagram = VGroup(rect, v_terms)
            diagram.to_edge(DOWN, buff=1.5)

            v_stacks.move_to(rect)
            all_v_stacks.add(v_stacks)

            VGroup(wv_up, wv_down).next_to(diagram, UP, buff=SMALL_BUFF, aligned_edge=LEFT)

            arrow = Arrow(pattern, diagram, buff=0.5)
            arrow.shift(0.25 * UP)

            value_diagrams.add(diagram)
            arrows.add(arrow)

        right_dots2 = right_dots.copy()

        self.play(
            LaggedStart(
                (FadeTransform(m1.copy(), m2)
                for m1, m2 in zip(att_patterns, value_diagrams)),
                lag_ratio=0.25,
                group_type=Group,
            ),
            LaggedStartMap(FadeIn, wv_up_syms, shift=DOWN, lag_ratio=0.25),
            LaggedStartMap(FadeIn, wv_down_syms, shift=DOWN, lag_ratio=0.25),
            LaggedStartMap(GrowArrow, arrows, lag_ratio=0.25),
            right_dots2.animate.match_y(value_diagrams).set_anim_args(time_span=(1.0, 1.75)),
        )
        self.wait()

        self.play(
            LaggedStart(
                (Transform(VGroup(diagram[1]), v_stacks)
                for diagram, v_stacks in zip(value_diagrams, all_v_stacks)),
                lag_ratio=0.25,
                run_time=2
            )
        )
        self.remove(value_diagrams)
        new_diagrams = VGroup(
            VGroup(vd[0], stacks)
            for vd, stacks in zip(value_diagrams, all_v_stacks)
        )
        value_diagrams = new_diagrams
        self.add(value_diagrams)

        # Show sums
        index = 2
        rects = VGroup()
        delta_Es = VGroup()
        arrows = VGroup()
        for n, diagram in enumerate(value_diagrams, start=1):
            diagram.target = diagram.generate_target()
            stacks = diagram.target[1]
            stacks.set_opacity(0.5)
            stacks[index].set_opacity(1, border_width=1)
            rect = SurroundingRectangle(stacks[index], buff=0.05)

            arrow = Vector(0.5 * DOWN)
            arrow.set_color(BLUE)
            arrow.next_to(rect, DOWN, SMALL_BUFF)

            delta_E = Tex(Rf"\Delta \vec{{\textbf{{E}}}}^{{({n})}}_i", font_size=36)
            delta_E.set_color(BLUE)
            delta_E.next_to(arrow, DOWN, SMALL_BUFF)

            rects.add(rect)
            arrows.add(arrow)
            delta_Es.add(delta_E)

        rects.set_stroke(BLUE, 2)

        self.play(
            LaggedStartMap(MoveToTarget, value_diagrams),
            LaggedStartMap(ShowCreation, rects),
            LaggedStartMap(GrowArrow, arrows),
            LaggedStartMap(FadeIn, delta_Es, shift=0.5 * DOWN),
        )
        self.wait()

        # Add together all changes
        low_delta_Es = delta_Es.copy()
        low_delta_Es.scale(1.5)
        low_delta_Es.arrange(RIGHT, buff=0.75)
        low_delta_Es.next_to(delta_Es, DOWN, buff=1.0)
        plusses = VGroup(
            Tex("+", font_size=72).next_to(ldE, buff=0.1).shift(0.1 * DOWN)
            for ldE in low_delta_Es
        )
        dots = Tex(R"\dots", font_size=72).next_to(plusses, RIGHT)

        self.play(
            TransformFromCopy(delta_Es, low_delta_Es),
            Write(plusses),
            Write(dots),
            frame.animate.reorient(0, 0, 0, (-0.99, -1.51, 0.0), 10.71),
        )
        self.wait()

        # Include original embedding
        og_emb = Tex(R"\vec{\textbf{E}}_i", font_size=72)
        og_emb_plus = Tex("+", font_size=72)
        og_emb_plus.next_to(low_delta_Es, LEFT, SMALL_BUFF)
        og_emb.next_to(og_emb_plus, LEFT, 2 * SMALL_BUFF)
        lil_rect = SurroundingRectangle(og_emb)
        big_rect = SurroundingRectangle(VGroup(og_emb, low_delta_Es, dots), buff=0.25)
        lil_rect.set_stroke(WHITE, 2)
        big_rect.set_stroke(TEAL, 3)
        og_label = Text("Original\nembedding")
        new_label = Text("New\nembedding")
        new_label.set_color(TEAL)
        for label in [og_label, new_label]:
            label.next_to(lil_rect, LEFT, buff=MED_LARGE_BUFF)

        self.play(
            FadeIn(og_emb, shift=RIGHT, scale=0.5),
            Write(og_emb_plus),
            FadeIn(og_label, shift=RIGHT),
        )
        self.play(ShowCreation(lil_rect))
        self.wait()
        self.play(
            ReplacementTransform(lil_rect, big_rect),
            FadeTransform(og_label, new_label)
        )
        self.wait()


class OutputMatrix(InteractiveScene):
    def construct(self):
        # Set up all heads
        matrix_pairs = VGroup(self.get_factored_value_map() for x in range(3))
        matrix_pairs.arrange(RIGHT, buff=LARGE_BUFF)
        matrix_pairs.to_edge(LEFT)
        matrix_pairs.set_y(1)
        dots = Tex(R"\dots", font_size=120)
        dots.next_to(matrix_pairs, RIGHT, LARGE_BUFF)

        rects = VGroup(SurroundingRectangle(pair, buff=0.25) for pair in matrix_pairs)
        rects.set_stroke(RED, 2)
        labels = VGroup()
        for n, rect in enumerate(rects, start=1):
            rect.set_height(2.5, stretch=True, about_edge=UP)
            rect.round_corners(radius=0.1)
            label = Text(f"Head {n}\nValue map", font_size=36)
            label.next_to(rect, UP)
            labels.add(label)

        up_labels = VGroup()
        down_labels = VGroup()
        for n, pair in enumerate(matrix_pairs, start=1):
            up_mat, down_mat = pair
            down_label = TexText(Rf"Value$^{{({n})}}_{{\downarrow}}$", font_size=30)
            up_label = TexText(Rf"Value$^{{({n})}}_{{\uparrow}}$", font_size=30)
            for label, mat, v in zip([up_label, down_label], pair, [ORIGIN, 0.25 * RIGHT]):
                label.next_to(pair, DOWN, buff=0.5)
                label[-1].scale(1.5, about_edge=UL)
                label.match_x(mat)
                label.shift(v)
                arrow = FillArrow(label[2], mat, thickness=0.025)
                arrow.scale(0.6)
                label.add(arrow)

            up_labels.add(up_label)
            down_labels.add(down_label)

        up_labels.set_fill(RED_B)
        down_labels.set_fill(RED_C)

        # Animate
        for pair, rect, label, up_label, down_label in zip(matrix_pairs, rects, labels, up_labels, down_labels):
            mat_labels =VGroup(up_label, down_label)
            self.play(
                FadeIn(label, 0.25 * UP),
                LaggedStartMap(FadeIn, pair, scale=1.25, lag_ratio=0.5),
                LaggedStartMap(FadeIn, mat_labels, lag_ratio=0.5),
                ShowCreation(rect),
            )
        self.play(Write(dots))
        self.wait()

        # Aggregate into the output matrix
        up_matrices = VGroup(pair[0] for pair in matrix_pairs)
        stapled_up_matrices = up_matrices.copy()
        for mat in stapled_up_matrices:
            brackets = mat[-2:]
            brackets[0].stretch(0, 0, about_edge=RIGHT)
            brackets[1].stretch(0, 0, about_edge=LEFT)
            brackets.set_opacity(0)
        stapled_up_matrices.arrange(RIGHT, buff=SMALL_BUFF)
        stapled_up_matrices.scale(2)
        stapled_up_matrices.next_to(rects, DOWN, buff=1.5)

        up_labels.target = up_labels.generate_target()
        lines = VGroup()
        for stum, up_label in zip(stapled_up_matrices, up_labels.target):
            line = Line(UP, DOWN).match_height(stum)
            line.set_stroke(WHITE, 1)
            line.next_to(stum, RIGHT, buff=SMALL_BUFF / 2)
            lines.add(line)
            up_label[-1].set_opacity(0)
            up_label[-1].scale(0, about_edge=DOWN)
            up_label.scale(0.75)
            up_label.next_to(stum, UP, buff=SMALL_BUFF)

        out_dots = dots.copy()
        out_dots.scale(0.5)
        out_dots.next_to(lines, RIGHT)
        out_brackets = up_matrices[0].get_brackets().copy()
        out_brackets.match_height(stapled_up_matrices)
        out_brackets[0].next_to(stapled_up_matrices, LEFT, SMALL_BUFF)
        out_brackets[1].next_to(out_dots, RIGHT, SMALL_BUFF)

        out_matrix = VGroup(stapled_up_matrices, lines, out_dots, out_brackets)

        self.play(
            self.frame.animate.reorient(0, 0, 0, (-0.88, -0.87, 0.0), 8.00),
            up_matrices.animate.set_opacity(0.5),
            TransformFromCopy(up_matrices, stapled_up_matrices, lag_ratio=1e-4),
            MoveToTarget(up_labels),
            TransformFromCopy(dots, out_dots),
            FadeIn(lines, lag_ratio=0.5),
            FadeIn(out_brackets, scale=1.25),
            run_time=2
        )
        self.wait()

        # Circle and label output
        out_rect = SurroundingRectangle(VGroup(out_matrix, up_labels), buff=MED_SMALL_BUFF)
        out_rect.round_corners(radius=0.1)
        out_rect.set_stroke(PINK, 3)
        out_label = Text("Output\nmatrix")
        out_label.set_color(PINK)
        out_label.next_to(out_rect, LEFT)

        self.play(
            ShowCreation(out_rect),
            FadeIn(out_label, shift=0.25 * LEFT, scale=1.25),
        )
        self.wait()

        # Center the down matrices
        self.play(
            LaggedStart(
                (pair[1].animate.shift(0.5 * LEFT)
                for pair in matrix_pairs),
                lag_ratio=0.05,
            ),
            LaggedStart(
                (label.animate.shift(0.5 * LEFT)
                for label in down_labels),
                lag_ratio=0.05,
            ),
            LaggedStartMap(FadeOut, up_matrices)
        )
        self.wait()

    def get_factored_value_map(self, big_d=7, lil_d=4, height=1.0):
        matrices = VGroup(
            WeightMatrix(shape=(big_d, lil_d)),
            WeightMatrix(shape=(lil_d, big_d)),
        )
        matrices.arrange(RIGHT, buff=matrices[0].get_width() * 0.1)
        matrices.set_height(height)
        return matrices


class Parallelizability(InteractiveScene):
    def construct(self):
        # Set up curves
        n_instances = 20
        comp_syms = Tex(R"+\,\times").replicate(n_instances)
        comp_syms.arrange(DOWN)
        comp_syms.set_height(5.5)
        comp_syms.to_edge(DOWN)
        left_point = comp_syms.get_left() + 2 * LEFT
        right_point = comp_syms.get_right() + 2 * RIGHT
        curves = VGroup()
        for sym in comp_syms:
            curve = VMobject()
            curve.start_new_path(left_point)
            curve.add_cubic_bezier_curve_to(
                left_point + RIGHT,
                sym.get_left() + LEFT,
                sym.get_left()
            )
            curve.add_line_to(sym.get_right())
            curve.add_cubic_bezier_curve_to(
                sym.get_right() + RIGHT,
                right_point + LEFT,
                right_point,
            )
            curve.insert_n_curves(10)
            curves.add(curve)
        curves.set_stroke(width=(0, 2, 2, 2, 0))
        curves.set_submobject_colors_by_gradient(TEAL, BLUE)

        # Setup words
        in_word = Text("Input")
        out_word = Text("output")
        in_word.next_to(left_point, LEFT, SMALL_BUFF)
        out_word.next_to(right_point, RIGHT, SMALL_BUFF)
        self.add(comp_syms, in_word, out_word)

        # GPU symbol
        gpu = SVGMobject("gpu_large.svg")
        gpu.set_fill(GREY_B)
        gpu.set_width(1.5)
        gpu.next_to(comp_syms, UP)
        gpu_name = Text("GPU")
        gpu_name.next_to(gpu, UP)
        gpu_name.set_fill(GREY_B)
        self.add(gpu, gpu_name)

        # Animation
        for n in range(4):
            curves.shuffle()
            self.play(
                LaggedStartMap(
                    ShowPassingFlash, curves,
                    lag_ratio=5e-3,
                    time_width=1.5,
                    run_time=4
                )
            )
