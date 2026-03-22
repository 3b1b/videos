from manim_imports_ext import *
import sympy
from _2023.clt.main import ChartBars


class TwinPrimScrolling(InteractiveScene):
    def construct(self):
        # Create list of primes
        n_max = 1000
        primes = list(sympy.primerange(2, n_max))
        prime_mobs = VGroup(*map(Integer, primes))
        prime_mobs.arrange(RIGHT, buff=MED_LARGE_BUFF)
        prime_mobs.move_to(1 * RIGHT, LEFT)
        prime_mobs.shift(2.5 * DOWN)
        prime_mobs.set_fill(border_width=1)

        twin_prime_color = BLUE
        tp_groups = VGroup()
        twin_primes = set()
        for i in range(len(primes) - 1):
            if primes[i] + 2 == primes[i + 1]:
                twin_primes.add(primes[i])
                twin_primes.add(primes[i + 1])
                arc = Line(
                    prime_mobs[i].get_top(),
                    prime_mobs[i + 1].get_top(),
                    path_arc=-PI,
                    stroke_color=twin_prime_color,
                    stroke_width=2,
                )
                arc.scale(0.9, about_edge=UP)
                plus_2 = Tex("+2", font_size=24)
                plus_2.next_to(arc, UP, SMALL_BUFF)
                plus_2.set_color(twin_prime_color)
                tp_groups.add(VGroup(
                    *prime_mobs[i:i + 2].copy(),
                    arc, plus_2
                ))

        non_twin_primes = VGroup(*(
            pm
            for pm, prime in zip(prime_mobs, primes)
            if prime not in twin_primes
        ))

        highlight_point = 5

        def update_tp_groups(tp_groups):
            for tp_group in tp_groups:
                for mob in tp_group[2:]:
                    mob.match_x(tp_group[:2])
                x_coord = tp_group.get_x()
                if x_coord < -10:
                    tp_groups.remove(tp_group)
                    continue
                pre_alpha = inverse_interpolate(
                    highlight_point + 0.25,
                    highlight_point - 0.25,
                    x_coord
                )
                alpha = clip(pre_alpha, 0, 1)
                color = interpolate_color(WHITE, twin_prime_color, alpha)
                tp_group[:2].set_color(color)
                tp_group[2].set_stroke(width=2 * alpha)
                tp_group[3].set_opacity(alpha)

        tp_groups.add_updater(update_tp_groups)

        self.add(non_twin_primes)
        self.add(tp_groups)

        # Animation
        velocity = 1.5
        run_time = non_twin_primes.get_width()

        def shift_updater(mobject, dt):
            mobject.shift(velocity * dt * LEFT)

        tp_groups.add_updater(shift_updater)
        non_twin_primes.add_updater(shift_updater)

        self.wait(run_time)


class Timeline(InteractiveScene):
    def construct(self):
        # Timeline
        timeline = NumberLine(
            (-500, 2050, 100),
            width=FRAME_WIDTH - 1
        )
        timeline.move_to(2.5 * DOWN)
        timeline.add_numbers(
            list(range(0, 2050, 500)),
            direction=DOWN,
            group_with_commas=False,
            font_size=36,
        )
        label = Tex(R"\sim 300 BC", font_size=36)
        v_line = Line(DOWN, UP)
        v_line.move_to(timeline.n2p(-300), DOWN)
        v_line.shift(SMALL_BUFF * DOWN)
        v_line.set_stroke(WHITE, 1)
        label.next_to(v_line, DOWN, buff=0.15)

        self.add(timeline)
        self.add(v_line)
        self.add(label)

        # Label
        question = TexText(
            R"``Are there infinitely many twin primes?''\\ \quad -Euclid",
            font_size=48,
            alignment="",
        )
        question[-7:].shift(RIGHT + 0.1 * DOWN)
        question.next_to(v_line, UP)
        question.shift_onto_screen(buff=0.2)

        arrows = VGroup(
            Vector(3 * LEFT),
            Text("No one can\nanswer", font_size=60).set_color(YELLOW),
            Vector(3 * RIGHT)
        )
        arrows.arrange(RIGHT)
        arrows.set_width(get_norm(timeline.n2p(2000) - timeline.n2p(-300)))
        arrows.next_to(v_line, RIGHT)
        arrows.shift(0.5 * UP)

        # self.add(question)
        self.add(arrows)


class InfinitePrimes(InteractiveScene):
    def construct(self):
        # Test
        n_max = 150
        primes = list(sympy.primerange(2, n_max))
        prime_mobs = VGroup(*map(Integer, primes))
        prime_mobs.set_height(1.5)
        prime_mobs.use_winding_fill(False)

        def update_opacity(prime):
            # alpha = inverse_interpolate(10, 0, prime.get_y())
            alpha = inverse_interpolate(-20, 0, prime.get_z())
            prime.set_fill(opacity=alpha)

        for n, prime in enumerate(prime_mobs):
            # prime.rotate(88 * DEGREES, RIGHT)
            prime.set_height(1.5 * 0.95**n)
            prime.move_to(3 * n * IN)
            prime.add_updater(update_opacity)

        rect = FullScreenRectangle()
        rect.set_fill(BLACK, 1)
        rect.fix_in_frame()

        self.frame.reorient(-70, -30, 70)
        self.frame.set_focal_distance(5)

        self.add(prime_mobs)
        self.play(
            prime_mobs.animate.shift(70 * OUT),
            FadeIn(rect, time_span=(10, 12)),
            run_time=12,
            rate_func=rush_into,
        )

    def old_attempt(self):
        # Old attempt
        point = UL
        height = 2
        shift = 0.75 * np.array([1, -1.2, 0])
        scale = 1.0
        opacity = 1.0

        scale_factor = 0.65
        shift_scale_factor = 0.7
        opacity_scale_factor = 0.9

        for n, mob in enumerate(prime_mobs):
            mob.set_height(height * scale_factor**n)
            mob.move_to(point + sum(
                shift * shift_scale_factor**k
                for k in range(n)
            ))
            mob.set_fill(opacity=opacity * opacity_scale_factor**n)

        prime_mobs.submobjects.reverse()
        self.add(prime_mobs)


class ThoughtBubble(InteractiveScene):
    def construct(self):
        randy = Randolph()
        randy.to_edge(LEFT)
        bubble = randy.get_bubble("Suppose\nnot...")
        self.add(bubble, bubble.content)


class SimpleBubble(InteractiveScene):
    def construct(self):
        self.add(OldThoughtBubble())


class EuclidProof(InteractiveScene):
    def construct(self):
        # Suppose finite
        prime_sequence = Tex(R"2, 3, 5, \dots , p_n", font_size=72)
        prime_sequence.move_to(UP + LEFT)
        last = prime_sequence["p_n"]
        finite_words = Text("All primes (suppose finite)", font_size=60)
        finite_words.next_to(last, UR).shift(0.2 * UP)
        sequence_rect = SurroundingRectangle(prime_sequence)
        sequence_rect.set_stroke(YELLOW, 2)
        sequence_rect.set_stroke(YELLOW, 2)
        finite_words.next_to(sequence_rect, UP)
        finite_words.shift(RIGHT * (sequence_rect.get_x() - finite_words["All primes"].get_x()))

        last_arrow = Arrow(
            finite_words["prime"].get_corner(DL),
            last,
            path_arc=-PI / 3,
            buff=0.1
        )
        VGroup(finite_words, sequence_rect).set_color(YELLOW)

        self.add(prime_sequence)
        self.add(finite_words)
        self.add(sequence_rect)

        # Multiply, add 1, factor
        product = Tex(R"N = 2 \cdot 3 \cdot 5 \cdots p_n", font_size=72)
        product.next_to(prime_sequence, DOWN, LARGE_BUFF)

        plus_one = Tex("+1", font_size=72)
        plus_one.next_to(product, RIGHT, 0.2)
        plus_one.shift(0.05 * UP)
        N_mob = VGroup(product, plus_one)
        N_mob.match_x(prime_sequence)

        psc = prime_sequence.copy()
        self.play(
            TransformMatchingTex(
                psc,
                product,
                matched_pairs=[
                    (psc[","], product[R"\cdot"]),
                    (psc[R"\dots"], product[R"\cdots"]),
                ],
                run_time=1
            )
        )
        self.play(Write(plus_one))
        self.wait()

        # Factor
        N_rect = SurroundingRectangle(
            VGroup(product[2:], plus_one)
        )
        factor_arrow = Vector(DL)
        factor_arrow.next_to(N_rect, DOWN)
        factor_word = Text("Prime factors", font_size=60)
        factor_word.next_to(factor_arrow, RIGHT, buff=0)

        VGroup(N_rect, factor_arrow, factor_word).set_color(TEAL)

        factor_eq = Tex(R"N = q_1 \cdots q_k", font_size=72)
        factor_eq[R"q_1 \cdots q_k"].set_color(RED)
        factor_eq.next_to(product, DOWN, buff=1.5, aligned_edge=LEFT)

        self.play(
            FadeTransformPieces(product.copy(), factor_eq),
            FadeIn(N_rect),
            GrowArrow(factor_arrow),
            FadeIn(factor_word, 0.5 * DOWN),
        )
        self.wait(3)

        # Contradiction
        q_rect = SurroundingRectangle(factor_eq["q_1"], buff=0.1)
        q_rect.set_stroke(WHITE, 3)
        q_words = TexText(R"Cannot be in $\{2, 3, 5, \dots, p_n\}$", font_size=60)
        q_words.next_to(q_rect, UP, aligned_edge=LEFT)
        q_words.match_color(factor_eq[2])
        rect = SurroundingRectangle(Group(*(
            mob
            for mob in self.mobjects
            if isinstance(mob, StringMobject)
        )), buff=0.5)
        rect.set_stroke(WHITE, 3)
        rect.shift(0.35 * DOWN)
        rect.set_fill(RED, 0.1)
        cont_word = Text("Contradiction!", font_size=90)
        cont_word.next_to(rect, UP, buff=0.5, aligned_edge=RIGHT)
        cont_word.set_color(WHITE)

        self.play(
            Transform(N_rect, q_rect),
            FadeTransformPieces(factor_word, q_words),
            FadeOut(factor_arrow),
        )
        self.wait(4)
        self.play(
            FadeIn(rect),
            FadeIn(cont_word, 0.5 * UP),
        )
        self.wait()

    def infinite(self):
        # Interlude to show infinite
        inf_sequence = Tex(R"2, 3, 5, 7, 11, \dots", font_size=72)
        inf_sequence.move_to(prime_sequence, LEFT)
        inf_arrow = Vector(RIGHT)
        inf_arrow.next_to(inf_sequence, RIGHT, SMALL_BUFF)
        inf_words = Text("Infinite", font_size=60)
        inf_words.next_to(inf_arrow, DOWN, aligned_edge=LEFT)

        self.add(inf_sequence, inf_arrow, inf_words)


class PrimeDensityHistogram(InteractiveScene):
    def construct(self):
        # Axes
        max_x = 10000
        step = 100
        labeled_xs = list(range(1000, max_x + 1000, 1000))
        axes = Axes(
            (0, max_x, step),
            (0, 0.5, 0.1),
            width=FRAME_WIDTH - 2,
            height=6,
            x_axis_config=dict(
                tick_size=0.03,
                big_tick_numbers=labeled_xs,
                longer_tick_multiple=3,
            )
        )
        axes.x_axis.add_numbers(
            labeled_xs,
            font_size=16,
            buff=0.25,
        )
        axes.y_axis.add_numbers(
            np.arange(0, 0.6, 0.1),
            num_decimal_places=2,
        )

        y_label = Text("""
            Proportion of primes
            in ranges of length 1,000
        """, font_size=36)
        y_label.next_to(axes.y_axis.get_top(), RIGHT, buff=0.5)
        y_label.to_edge(UP)
        axes.add(y_label)

        self.add(axes)

        # Bars
        proportions = []
        for n in range(0, max_x, step):
            n_primes = len(list(sympy.primerange(n, n + step)))
            proportions.append(n_primes / step)

        bars = ChartBars(axes, proportions)

        self.add(bars)


class PrimesNearMillion(InteractiveScene):
    def construct(self):
        # Add line
        T = int(1e6)
        radius = 800
        spacing = 50
        labeled_numbers = list(range(T - radius, T + radius, spacing))
        number_line = NumberLine(
            (T - radius, T + radius),
            width=250,
            tick_size=0.075,
        )
        number_line.ticks[::spacing // 5].stretch(2, 1)
        # number_line.stretch(0.2, 0)
        number_line.add_numbers(labeled_numbers, font_size=48, buff=0.5)
        self.add(number_line)

        # Primes
        primes = np.array(list(sympy.primerange(T - radius, T + radius)))
        dots = GlowDots(number_line.n2p(primes))
        dots.set_glow_factor(2)
        dots.set_radius(0.35)
        self.add(dots)

        # Highlight twins
        arcs = VGroup()
        for p1, p2 in zip(primes, primes[1:]):
            if p1 + 2 == p2:
                arc = Line(
                    number_line.n2p(p1),
                    number_line.n2p(p2),
                    path_arc=-PI
                )
                arc.set_stroke(YELLOW, 3)
                plus_2 = Tex("+2", font_size=24)
                plus_2.set_fill(YELLOW)
                plus_2.next_to(arc, UP, SMALL_BUFF)
                arcs.add(arc, plus_2)

        # Pan
        line_group = Group(number_line, dots, arcs)
        line_group.shift(1.5 * DOWN + -number_line.n2p(1e6))
        line_group.add_updater(lambda m, dt: m.shift(2 * dt * LEFT))
        self.add(line_group)

        # Words
        t2c = {"T": BLUE}
        kw = dict(font_size=90, t2c=t2c)
        words = TexText("How dense are primes?", **kw)
        lhs = TexText("Prime density near $T$", **kw)
        approx = Tex(R"\approx", **kw)
        approx.rotate(PI / 2)
        rhs = Tex(R"1 / \ln(T)", **kw)
        group = VGroup(lhs, approx, rhs)
        group.arrange(DOWN, buff=0.5)
        group.to_edge(UP)
        words.move_to(lhs)

        example = TexText("(e.g. $T = 1{,}000{,}000$)", font_size=60, t2c=t2c)
        example.next_to(lhs, DOWN, LARGE_BUFF)
        arrow = Arrow(
            lhs["T"].get_bottom(),
            example.get_right(),
            stroke_width=8,
            stroke_color=BLUE,
            path_arc=-PI / 2,
            buff=0.2,
        )

        self.add(words)
        self.wait(13)
        self.play(
            FadeTransform(words, lhs, run_time=1),
            ShowCreation(arrow),
            FadeIn(example, time_span=(1, 2)),
        )
        self.wait(5)
        self.play(
            FadeOut(arrow),
            FadeOut(example),
            Write(approx),
            FadeIn(rhs[:-2], DOWN),
            FadeIn(rhs[-1], DOWN),
            TransformFromCopy(lhs["T"], rhs["T"]),
        )
        self.wait(45)

    def old_zooming(self):
        sf = 0.1
        self.play(
            number_line.animate.scale(sf, about_point=ORIGIN),
            dots.animate.scale(sf, about_point=ORIGIN).set_radius(0.1),
            rate_func=rush_from,
            run_time=17
        )
        self.wait()

        # Zoom in to twin prime
        zoom_point = number_line.n2p(1000210)
        frame = self.frame

        self.play(
            frame.animate.move_to(zoom_point).set_height(0.60),
            dots.animate.set_radius(0.03),
            run_time=5
        )


class PrimePanning(InteractiveScene):
    def construct(self):
        # (A bit too much copy paste from above)
        N_max = 500
        number_line = NumberLine(
            (0, N_max),
            unit_size=0.5,
            tick_size=0.075,
        )
        number_line.ticks[::10].stretch(2, 1)
        number_line.add_numbers(range(0, N_max), font_size=20, buff=0.2)
        number_line.move_to(2 * LEFT, LEFT)
        self.add(number_line)

        # Primes
        primes = np.array(list(sympy.primerange(0, N_max)))
        dots = GlowDots(number_line.n2p(primes))
        dots.set_glow_factor(2)
        dots.set_radius(0.35)
        self.add(dots)

        # Pan
        frame = self.frame
        frame.set_height(4)
        frame.add_updater(lambda m, dt: m.shift(1.5 * dt * RIGHT))
        self.wait(90)


class SieveWithMod(InteractiveScene):
    def construct(self):
        # Setup prime list title
        prime_title = Text("Primes: ", font_size=72)
        prime_title.to_edge(UP)
        prime_title.set_x(-5)

        self.add(prime_title)

        # Setup grid
        row_size = 12
        grid = Square().get_grid(20, row_size, buff=0)
        grid.set_stroke(GREY_B, 1)
        grid.set_width(FRAME_WIDTH - 5)
        grid.move_to(1.65 * UP, UP)
        labels = VGroup()
        labeled_boxes = VGroup()
        for n, square in enumerate(grid, start=1):
            label = Integer(n)
            label.set_max_width(0.5 * square.get_width())
            label.move_to(square)
            square.label = label
            labels.add(label)
            labeled_boxes.add(VGroup(square, label))

        self.add(grid)
        self.add(labels)

        # Do the initial sift
        non_primes = VGroup(
            label for label in labels
            if not sympy.isprime(label.get_value())
        )
        primes = VGroup(
            label for label in labels
            if sympy.isprime(label.get_value())
        )
        prime_list = primes.copy()
        prime_list.scale(72 / 48)
        prime_list.arrange(RIGHT, buff=MED_LARGE_BUFF, aligned_edge=DOWN)
        prime_list.next_to(prime_title, RIGHT, MED_LARGE_BUFF, aligned_edge=DOWN)
        self.play(
            non_primes.animate.set_fill(opacity=0.25),
            lag_ratio=0.025,
            run_time=5
        )
        self.wait()
        self.play(
            TransformFromCopy(primes, prime_list)
        )
        self.play(
            labels.animate.set_fill(opacity=1),
            FadeOut(prime_list)
        )

        # Reduction title
        reduction_label = Text("Reduce all mod")
        reduction_label.next_to(grid, UP, buff=MED_SMALL_BUFF, aligned_edge=LEFT)
        reduction_label.set_fill(GREY_A)
        reduction_label.set_fill(opacity=0)

        self.add(reduction_label)

        # Reduction game (this will be looped)
        arrows = VGroup()
        reductions = VGroup()
        colors = color_gradient([BLUE_E, BLUE_A, RED_E], 8)

        for _ in range(5):
            # Pull out the next prime
            prime_label = labels[1]
            prime_value = int(prime_label.get_value())
            highlight = SurroundingRectangle(prime_label)
            list_prime = prime_label.copy()
            list_prime.scale(72 / 48)
            list_prime.next_to(prime_title, RIGHT, buff=0.35, aligned_edge=DOWN)
            list_prime_highlight = SurroundingRectangle(list_prime)
            comma = Text(",")
            comma.next_to(list_prime.get_corner(DR), RIGHT, SMALL_BUFF)
            reduction_prime = prime_label.copy()
            reduction_prime.next_to(reduction_label, RIGHT, MED_SMALL_BUFF, DOWN)

            rows = VGroup(
                labeled_boxes[n:n + row_size]
                for n in range(0, len(grid), row_size)
            )
            rows.target = rows.generate_target()
            for row in rows.target:
                row.arrange(RIGHT, buff=0)
                row.set_width(grid.get_width())
                row.set_max_height(rows.target[0].get_height())
                row.align_to(rows.target[0], LEFT)
            rows.target.arrange(DOWN, buff=1.2)
            rows.target.move_to(grid, UP)

            self.play(
                ShowCreation(highlight),
                FadeOut(arrows),
                FadeOut(reductions),
            )
            self.play(
                TransformFromCopy(prime_label, list_prime),
                FadeIn(comma, UP),
                TransformFromCopy(highlight, list_prime_highlight),
            )
            self.play(
                FadeOut(list_prime_highlight),
                FadeOut(highlight),
                TransformFromCopy(list_prime, reduction_prime),
                reduction_label.animate.set_opacity(1),
                MoveToTarget(rows),
            )
            prime_title.add(list_prime)

            # Add reductions
            arrows = VGroup()
            reductions = VGroup()
            for label, box in zip(labels, grid):
                reduction = Integer(label.get_value() % prime_value)
                reduction.next_to(box, DOWN, buff=MED_LARGE_BUFF)
                reductions.add(reduction)
                color = YELLOW if label.get_value() % prime_value == 0 else BLUE
                arrow = Arrow(
                    label, reduction,
                    buff=0.15,
                    max_tip_length_to_length_ratio=0.4,
                    stroke_width=3,
                    stroke_color=GREY_A,
                )
                arrows.add(arrow)
                rect = SurroundingRectangle(reduction)
                rect.set_stroke(color, width=1)
                reduction.add(rect)

            for n, arrow, reduction in zip(it.count(), arrows, reductions):
                self.add(arrow, reduction)
                if n < 36:
                    self.wait(0.2 * (1 + (n % 2)))

            # Kill the zeros
            killed_indices = [
                n
                for n, label in enumerate(labels)
                if label.get_value() % prime_value == 0
            ]
            self.play(
                *(
                    LaggedStartMap(FadeOut, VGroup(group[n] for n in killed_indices), shift=0.1 * DOWN, lag_ratio=0.1)
                    for group in [labeled_boxes, arrows, reductions]
                ),
                FadeOut(reduction_prime)
            )

            for group in [labeled_boxes, grid, labels, arrows, reductions]:
                to_remove = [group[n] for n in killed_indices]
                group.remove(*to_remove)


class DensityFormula(InteractiveScene):
    def construct(self):
        # Formula
        t2c = {
            "T": BLUE,
        }
        kw = dict(t2c=t2c, font_size=90)
        lhs = TexText("Prime density near $T$", **kw)
        approx = Tex(R"\approx", **kw)
        approx.rotate(PI / 2)
        rhs = Tex(R"1 / \ln(T)", **kw)
        group = VGroup(lhs, approx, rhs)
        group.arrange(DOWN, buff=0.5)
        group.to_edge(UP)

        example = TexText("(e.g. $T = 1{,}000{,}000$)", font_size=60, t2c=t2c)
        example.next_to(lhs, DOWN, LARGE_BUFF)
        arrow = Arrow(
            lhs["T"].get_bottom(),
            example.get_right(),
            stroke_width=8,
            stroke_color=BLUE,
            path_arc=-PI / 2,
            buff=0.2,
        )

        self.add(lhs, example, arrow)
        self.wait()

        self.remove(example, arrow)
        self.add(approx, rhs)

    def old_mess(self):
        # Formula
        kw = dict(t2c={
            "T": BLUE,
            "1{,}000{,}000": BLUE,
        })
        lhs = TexText("What's the density of primes near $T$", **kw)
        rhs = Tex(R"\approx \frac{1}{\ln(T)}", **kw)
        group = VGroup(lhs, rhs)
        group.arrange(RIGHT)
        group.to_edge(UP)

        q_mark = Text("?")
        q_mark.next_to(lhs, RIGHT, buff=0.1)
        q_mark.align_to(lhs[0], DOWN)
        lhs_note = TexText(
            "(Some big number, e.g. $T = 1{,}000{,}000$)",
            font_size=36,
            **kw
        )
        lhs_note.next_to(lhs, DR, buff=LARGE_BUFF)
        lhs_note.shift_onto_screen()
        arrow = Arrow(lhs_note.get_top(), lhs["T"], buff=0.1)
        arrow.set_color(BLUE)

        self.add(lhs)
        self.add(q_mark, lhs_note, arrow)
        self.wait()

        self.remove(q_mark, lhs_note, arrow)
        self.remove(lhs["What's the"])
        self.add(rhs)


class OldGapsInPrimes(InteractiveScene):
    def construct(self):
        pass


class NewGapsInPrimes(InteractiveScene):
    def construct(self):
        # Show number line
        x_min = 99980
        x_max = 100100
        line = NumberLine(
            x_range=(x_min, x_max),
            width=0.5 * (x_max - x_min)
        )
        line.to_edge(LEFT).shift(2 * LEFT)
        line.set_y(1)
        
        primes = [n for n in range(x_min, x_max) if sympy.isprime(n)]
        labels = line.add_numbers(primes, font_size=48)
        lc = labels[:2].get_center()
        labels[:2].arrange(RIGHT, buff=0.5).move_to(lc)
        dots = GlowDots([line.n2p(p) for p in primes])

        self.add(line)
        self.add(dots)

        # Pan over
        center_label = labels[-2]
        self.play(
            self.frame.animate.match_x(center_label).set_height(11),
            run_time=8
        )

        # Label it
        prime_label = TexText(R"Big prime, $p$", font_size=96)
        prime_label.set_color(YELLOW)
        prime_label.next_to(center_label.get_corner(DL), DOWN, 1.5, aligned_edge=RIGHT)
        arrow = Arrow(
            prime_label[-1].get_top(), center_label.get_bottom(),
            buff=0.25,
        )
        arrow.set_stroke(YELLOW)

        self.play(
            FadeIn(prime_label),
            GrowArrow(arrow),
        )
        self.wait()

        # Show the gap
        brace = Brace(Line(*dots.get_points()[-2:]), UP)
        gap_label = Text("gap", font_size=72)
        gap_label.next_to(brace, UP)
        self.play(
            GrowFromPoint(brace, brace.get_left()),
            FadeIn(gap_label, 2 * RIGHT)
        )
        self.wait()

        # Expected gap size
        eq = TexText(R"E[gap] = $\ln(p)$", font_size=96)
        eq["p"][-1].set_color(YELLOW)
        eq.next_to(self.frame.get_top(), DOWN, buff=MED_LARGE_BUFF)

        self.play(LaggedStart(
            FadeIn(eq["E["]),
            FadeIn(eq["] = "]),
            FadeIn(eq[R"\ln("]),
            FadeIn(eq[R")"]),
            FadeTransform(gap_label.copy(), eq["gap"][0]),
            FadeTransform(prime_label[-1].copy(), eq["p"][-1]),
            lag_ratio=0.1,
        ))
        self.wait()


class CrankEmail(InteractiveScene):
    def construct(self):
        # Background
        rect = FullScreenRectangle(fill_color=WHITE, fill_opacity=1)
        rect.scale(2)
        rect.stretch(10, 1, about_edge=UP)
        self.add(rect)

        frame = self.frame
        frame.set_height(9)
        frame.to_edge(UP, buff=0)

        # Rows of numbers
        numbers1 = list(range(1, 100))
        numbers2 = list(filter(lambda m: m % 2 != 0, numbers1))
        numbers3 = list(filter(lambda m: m % 3 != 0, numbers2))

        arrays = VGroup(
            self.create_array(numbers1[:30], 2, BLUE_D),
            self.create_array(numbers2[:30], 3, GREEN_D),
            self.create_array(numbers3[:20], 5, RED_D),
        )

        arrays.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        arrays.to_corner(UL)

        # Paragraphs
        kw = dict(alignment="LEFT", font_size=48, fill_color=BLACK, font="Roboto")
        paragraphs = VGroup(
            Text("""
                Dear sir,

                I have proven the twin prime conjecture.

                I study an elegant proof of prime generation:
                List all natural numbers, and start by
                reducing each of them modulo 2:
            """, **kw),
            arrays[0],
            Text("""
                Remove numbers which have reduced
                to 0, reduce what remains modulo 3:
            """, **kw),
            arrays[1],
            Text("""
                Again, remove numbers which have reduced
                to 0, reduce what remains modulo 5:
            """, **kw),
            arrays[2],
        )

        paragraphs.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        paragraphs.shift(paragraphs[0].get_x() * LEFT)
        paragraphs.to_edge(UP, buff=0.2)

        self.add(paragraphs)
        for array in arrays:
            for grid in array:
                self.remove(grid[1])

        # Fill in grid, slide frame
        self.anticipate_frame_to_y(-5, run_time=3)
        for grid in arrays[0]:
            self.animate_reduction(grid)

        self.cross_out_the_zeros(arrays[0])
        self.wait(0.5)
        self.play(frame.animate.set_y(-11))

        for n, grid in enumerate(arrays[1]):
            self.animate_reduction(grid)
            if n == 0:
                self.anticipate_frame_to_y(-15, run_time=2)

        self.anticipate_frame_to_y(-22, run_time=2)
        self.cross_out_the_zeros(arrays[1])
        self.wait(2)

        for grid in arrays[2]:
            self.animate_reduction(grid)
        self.cross_out_the_zeros(arrays[2])

    def create_array(
        self,
        numbers,
        modulus,
        color=BLUE,
        height=0.85,
        spacing=10,
        buff=0.75,
        width=10,
    ):
        result = VGroup()
        row1_content = numbers
        row2_content = [n % modulus for n in numbers]
        row1_title = "Numbers: "
        row2_title = f"Mod {modulus}:"
        n = 0
        while n < len(row1_content):
            result.add(self.create_table(
                row1_title, row2_title,
                row1_content[n:n + spacing],
                row2_content[n:n + spacing],
                color=color,
                height=height,
            ))
            n += spacing
            row1_title = ""
            row2_title = ""
        result.arrange(DOWN, buff=buff, aligned_edge=RIGHT)
        result.set_width(width)
        return result

    def create_table(
        self,
        row1_title,
        row2_title,
        row1_content,
        row2_content,
        x_spacing=0.6,
        font_size=36,
        color=BLUE,
        height=0.85
    ):
        # Numbers
        row1_mobs, row2_mobs = (
            VGroup(*(
                Integer(n, font_size=font_size)
                for n in content
            ))
            for content in [row1_content, row2_content]
        )
        for x, mob in enumerate(row1_mobs):
            mob.set_x(x_spacing * x)
        row1_mobs.to_edge(LEFT)
        row1_mobs.set_fill(BLACK)

        row2_mobs.set_color(color)
        for m1, m2 in zip(row1_mobs, row2_mobs):
            m2.set_max_width(0.5 * x_spacing)
            m2.next_to(m1, DOWN, 0.5)

        grid = VGroup(row1_mobs, row2_mobs)

        # Titles
        row1_label = Text(row1_title, font_size=font_size)
        row2_label = Text(row2_title, font_size=font_size)
        row1_label.set_color(BLACK)
        row1_label.next_to(row1_mobs, LEFT, buff=0.5)
        row2_label.next_to(row2_mobs, LEFT, buff=0.5)
        row2_label.set_color(color)
        grid.add(row1_label, row2_label)

        # Grid lines
        h_line = Line(ORIGIN, grid.get_width() * RIGHT)
        h_line.move_to(grid, LEFT)
        v_lines = VGroup(*(
            Line(grid.get_height() * UP, ORIGIN).set_x(
                0.5 * (row1_mobs[i].get_right()[0] + row1_mobs[i + 1].get_left()[0])
            ).align_to(grid, UP)
            for i in range(len(row1_mobs) - 1)
        ))
        v_lines.set_stroke(BLACK, width=1)
        h_line.set_stroke(BLACK, width=1)

        grid.add(h_line, v_lines)
        grid.set_height(height, about_edge=LEFT)

        return grid

    def animate_reduction(self, array, beat_time=0.17):
        reductions = array[1]
        self.remove(reductions)
        self.wait(0.1)
        for i, term in enumerate(reductions):
            self.add(term)
            self.wait(beat_time)
            m10 = i % 10
            if m10 % 2 == 1:
                self.wait(beat_time)
            if m10 == 9:
                self.wait(beat_time)
        self.add(array)

    def cross_out_the_zeros(self, array):
        crosses = VGroup()
        rects = VGroup()
        for grid in array:
            for m1, m2 in zip(grid[0], grid[1]):
                if m2.get_value() == 0:
                    crosses.add(Cross(m1).scale(1.5))
                    rect = SurroundingRectangle(VGroup(m1, m2))
                    rect.set_fill(RED, 0.2)
                    rect.set_stroke(width=0)
                    rects.add(rect)
        self.play(
            ShowCreation(crosses, lag_ratio=0),
            FadeIn(rects, lag_ratio=0.1)
        )

    def anticipate_frame_to_y(self, y, run_time=3):
        turn_animation_into_updater(
            ApplyMethod(self.frame.set_y, y, run_time=run_time)
        )


class SieveOfEratosthenes(InteractiveScene):
    grid_shape = (10, 10)
    n_iterations = 10
    rect_buff = 0.1

    def construct(self):
        # Initialize grid
        grid = Square().get_grid(*self.grid_shape, buff=0)
        grid.set_height(FRAME_HEIGHT - 1)
        grid.set_stroke(width=1)
        number_mobs = self.get_number_mobs(grid)
        number_mobs[0].set_opacity(0)

        self.add(grid, number_mobs)

        # Run the sieve
        modulus = 2
        numbers = list(range(2, len(grid) + 1))
        for n in range(self.n_iterations):
            numbers = list(filter(lambda n: n % modulus != 0, numbers))
            to_remove = VGroup(*(
                mob
                for mob in number_mobs
                if mob.get_value() % modulus == 0
            ))
            rects = VGroup(*(
                SurroundingRectangle(tr, buff=self.rect_buff)
                for tr in to_remove
                if tr.get_fill_opacity() > 0.5
            ))
            rects.set_stroke(RED, 1)

            self.play(
                to_remove.animate.set_color(RED),
                Write(rects, stroke_color=RED, stroke_width=2),
                lag_ratio=0.1,
                run_time=2
            )
            self.wait()
            self.play(
                to_remove[0].animate.set_color(WHITE),
                to_remove[1:].animate.set_opacity(0),
                number_mobs[0].animate.set_opacity(0),
                FadeOut(rects)
            )
            modulus = numbers[0]

    def get_number_mobs(self, grid):
        return VGroup(*(
            Integer(i).set_height(0.3 * box.get_height()).move_to(box)
            for i, box in zip(it.count(1), grid)
        ))


class GiantSieve(SieveOfEratosthenes):
    grid_shape = (25, 25)
    n_iterations = 30
    rect_buff = 0.02

    # def get_number_mobs(self, grid):
    #     radius = grid[0].get_width() / 4
    #     return VGroup(*(
    #         Dot(radius=radius).move_to(box)
    #         for box in grid
    #     ))


class WannaProve(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        morty = Mortimer()
        VGroup(randy, morty).to_edge(DOWN)
        randy.set_x(-3)
        morty.set_x(3)
        morty.make_eye_contact(randy)

        self.play(
            randy.says("Twin primes\nare infinite!", mode="hooray"),
            morty.change("concentrating")
        )
        self.wait()
        self.play(
            FadeOut(randy.bubble),
            randy.change("pondering", look_at=3 * UP),
            morty.change("raise_right_hand", look_at=3 * UP)
        )

        # Hold up picture
