from manim_imports_ext import *
from _2024.transformers.helpers import *


class AttentionPatterns(InteractiveScene):
    def construct(self):
        # Add sentence
        phrase = " the fluffy blue creature foraged in a verdant forest"
        phrase_mob = Text(phrase)
        phrase_mob.move_to(2 * UP)
        words = phrase.split(" ")
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
        others = ["the", "foraged", "in", "a"]
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
        image_vects = VGroup(embeddings[i] for i in [1, 2, 3, 7, 8])

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
        dec = template.make_number_changeable(0)
        vect_syms = VGroup()
        for n, rect in enumerate(all_rects, start=1):
            dec.set_value(n)
            sym = template.copy()
            sym.next_to(rect, DOWN, buff=LARGE_BUFF)
            sym.set_color(GREY_A)
            vect_syms.add(sym)
        prev_center = vect_syms.get_center()
        vect_syms.arrange_to_fit_width(vect_syms.get_width())
        vect_syms.move_to(prev_center)

        emb_arrows.target = emb_arrows.generate_target()
        for arrow, rect, sym in zip(emb_arrows.target, all_rects, vect_syms):
            arrow.become(Arrow(
                rect.get_bottom(),
                sym[0].get_top(),
                buff=SMALL_BUFF
            ))

        all_brackets = VGroup(emb.get_brackets() for emb in embeddings)
        for brackets in all_brackets:
            brackets.target = brackets.generate_target()
            brackets.target.stretch(0, 1, about_edge=UP)
            brackets.target.set_fill(opacity=0)

        ghost_syms = vect_syms.copy()
        ghost_syms.set_opacity(0)

        self.play(
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
            LaggedStartMap(FadeIn, vect_syms, shift=UP),
            brace.animate.stretch(0.25, 1, about_edge=UP).set_opacity(0),
            FadeOut(dim_value, 0.25 * UP),
            MoveToTarget(emb_arrows, lag_ratio=0.1, run_time=2),
            LaggedStartMap(FadeOut, pos_labels, shift=UP)
        )
        self.clear()
        self.add(emb_arrows, all_rects, word_mobs, images, vect_syms)
        self.wait()

        # Preview desired updates
        vect_sym_primes = VGroup(
            sym.copy().add(Tex("'").move_to(sym.get_corner(UR) + 0.05 * DL))
            for sym in vect_syms
        )
        vect_sym_primes.shift(2 * DOWN)
        vect_sym_primes.set_color(TEAL)

        full_connections = VGroup(
            Line(sym1.get_bottom(), sym2.get_top(), buff=SMALL_BUFF)
            for sym2 in vect_sym_primes
            for sym1 in vect_syms
        )
        full_connections.set_stroke(GREY_B, 1)

        blue_fluff = ImageMobject("BlueFluff")
        verdant_forest = ImageMobject("VerdantForest")
        for n, image in [(3, blue_fluff), (8, verdant_forest)]:
            image.match_height(images)
            image.scale(1.2)
            image.next_to(vect_sym_primes[n], DOWN, buff=MED_SMALL_BUFF)

        self.play(
            ShowCreation(full_connections, lag_ratio=0.01, run_time=2),
            LaggedStart(
                (TransformFromCopy(sym1, sym2)
                for sym1, sym2 in zip(vect_syms, vect_sym_primes)),
                lag_ratio=0.05,
            ),
        )
        self.wait()
        self.play(LaggedStart(
            LaggedStart(
                (FadeTransform(im.copy(), blue_fluff)
                for im in images[:3]),
                lag_ratio=0.02,
                group_type=Group
            ),
            LaggedStart(
                (FadeTransform(im.copy(), verdant_forest)
                for im in images[3:]),
                lag_ratio=0.02,
                group_type=Group
            ),
            lag_ratio=0.5,
            run_time=2
        ))

        # Show black box that matrix multiples can be added to
        in_arrows = VGroup(Vector(0.5 * DOWN).next_to(sym, DOWN) for sym in vect_syms)
        dots = VGroup(Tex(R"\vdots").next_to(arrow, DOWN) for arrow in in_arrows)
        box = Rectangle(vect_syms.get_width() + 1, 1.5)
        box.set_fill(GREY_E, 1)
        box.set_stroke(WHITE, 1)
        box.next_to(in_arrows, DOWN)
        out_arrows = in_arrows.copy()
        out_arrows.next_to(box, DOWN)

        self.add(box)
        self.add(out_arrows)
        self.add(vect_sym_primes)

        # Ask questions

        # Associate questions with vectors

        # Show matrices

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
