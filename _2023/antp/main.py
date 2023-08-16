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
            Text("No one can\nanswer", font_size=36).set_color(YELLOW),
            Vector(3 * RIGHT)
        )
        arrows.arrange(RIGHT)
        arrows.set_width(get_norm(timeline.n2p(2000) - timeline.n2p(-300)))
        arrows.next_to(v_line, RIGHT)

        # self.add(question)
        self.add(arrows)


class InfinitePrimes(InteractiveScene):
    def construct(self):
        n_max = 44
        primes = list(sympy.primerange(2, n_max))
        prime_mobs = VGroup(*map(Integer, primes))
        prime_mobs.arrange(RIGHT, buff=MED_LARGE_BUFF)
        prime_mobs.to_edge(LEFT)
        dots = Tex(R"\dots")
        dots.next_to(prime_mobs, RIGHT)
        arrow = Vector(RIGHT)
        arrow.next_to(dots, RIGHT)
        infinite = Text("Infinite", font_size=72)
        infinite.next_to(arrow, DOWN, buff=0.75).shift_onto_screen()
        infinite.set_color(YELLOW)

        self.add(prime_mobs, dots, arrow)
        # self.add(infinite)


class ThoughtBubble(InteractiveScene):
    def construct(self):
        randy = Randolph()
        randy.to_edge(LEFT)
        bubble = randy.get_bubble("Suppose\nnot...")
        self.add(bubble, bubble.content)


class EuclidProof(InteractiveScene):
    def construct(self):
        # Suppose finite
        prime_sequence = Tex(R"p_1, p_2, \dots , p_n")
        prime_sequence.move_to(3 * LEFT + 2 * UP)
        brace = Brace(prime_sequence, UP)
        brace_text = brace.get_text("All primes, suppose finite")
        brace_text.set_color(YELLOW)

        self.add(prime_sequence)
        self.add(brace)
        self.add(brace_text)

        # Multiply, add 1, factor
        product = VGroup(*(
            prime_sequence[tex][0].copy()
            for tex in ["p_1", "p_2", R"\dots", "p_n"]
        ))
        product.target = product.generate_target()
        product.target.arrange(RIGHT, buff=SMALL_BUFF)
        product.target.next_to(prime_sequence, DOWN, buff=LARGE_BUFF)
        plus_one = Tex("+1")
        plus_one.next_to(product.target, RIGHT, SMALL_BUFF)
        plus_one.shift(0.05 * UP)

        factor = Text("primeFactors()")
        factor.set_color(GREY_A)
        factor[:-1].next_to(product.target, LEFT, buff=SMALL_BUFF)
        factor[-1:].next_to(plus_one, RIGHT, buff=SMALL_BUFF)
        factor_rhs = Tex(R" = q_1 \cdots q_k")
        factor_rhs.next_to(factor[-1], RIGHT)
        factor_rhs[1:].set_color(RED)

        self.play(MoveToTarget(product))
        self.play(Write(plus_one))
        self.wait()
        self.play(FadeIn(factor, UP))
        self.play(FadeIn(factor_rhs))
        self.wait(2)

        # Contradiction
        rect = SurroundingRectangle(factor_rhs[1:3], buff=0.05)
        rect.set_stroke(WHITE, 2)
        words = TexText(R"Cannot be any of $p_i$", font_size=42)
        words.next_to(rect, DOWN, aligned_edge=LEFT)
        words.set_color(RED)

        self.play(
            ShowCreation(rect),
            FadeIn(words, 0.25 * DOWN)
        )
        self.wait()


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
                numbers_with_elongated_ticks=labeled_xs,
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
        radius = 500
        labeled_numbers = list(range(T - radius, T + radius, 10))
        number_line = NumberLine(
            (T - radius, T + radius),
            tick_size=0.03,
            numbers_with_elongated_ticks=labeled_numbers,
        )
        number_line.ticks[::10].stretch(1.5, 1)
        number_line.stretch(0.2, 0)
        number_line.add_numbers(labeled_numbers)
        self.add(number_line)

        # Primes
        primes = np.array(list(sympy.primerange(T - radius, T + radius)))
        dots = GlowDots(number_line.n2p(primes))
        dots.set_radius(0.3)
        self.add(dots)

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


class DensityFormula(InteractiveScene):
    def construct(self):
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


class GapsInPrimes(InteractiveScene):
    def construct(self):
        pass


class CrankEmail(InteractiveScene):
    def construct(self):
        # Rows of numbers
        numbers1 = list(range(1, 60))
        numbers2 = list(filter(lambda m: m % 2 != 0, numbers1))
        numbers3 = list(filter(lambda m: m % 3 != 0, numbers2))

        arrays = VGroup(
            self.create_array(numbers1, 2, BLUE_D),
            self.create_array(numbers2, 3, GREEN_D),
            self.create_array(numbers3, 5, RED_D),
        )

        arrays.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        arrays.to_corner(UL)

        # Paragraphs
        self.add(
            FullScreenRectangle(fill_color=WHITE, fill_opacity=1).scale(3)
        )

        kw = dict(alignment="LEFT", font_size=36, fill_color=BLACK, font="Roboto")
        paragraphs = VGroup(
            Text("""
                Dear sir,

                I have found a most marvelous proof for the infinitude of twin primes,
                and I was hoping you could help to review it and offer me advice on
                next steps for publication.

                We begin our study of primes with a deceptively simple procedure. List
                all natural numbers, and start by reducing each of them modulo 2
            """, **kw),
            arrays[0],
            Text("""
                From here, remove all numbers which have reduced to 0 in this step,
                and reduce what remains by  3.
            """, **kw),
            arrays[1],
            Text("""
                Likewise, remove all numbers which have reduced to 0 in the last step,
                and reduce what remains by 5.
            """, **kw),
            arrays[2],
        )

        paragraphs.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        paragraphs.to_corner(UL)

        self.add(paragraphs[:2])
        self.animate_reduction(arrays[0])

        # Next steps
        frame = self.frame
        for i, modulus in [(1, 2), (2, 3)]:
            arrays[i][1].set_opacity(0)
            crosses = VGroup(*(
                Cross(mob).scale(1.5)
                for mob in arrays[i - 1][0][1::modulus]
            ))

            self.play(
                frame.animate.align_to(arrays[i], DOWN).shift(DOWN),
                FadeIn(crosses, lag_ratio=0.2),
                FadeIn(paragraphs[2 * i]),
                FadeIn(arrays[i]),
                run_time=2,
            )
            arrays[i][1].set_opacity(1)
            self.animate_reduction(arrays[i])

        # End

    def create_array(self, numbers, modulus, color=BLUE, height=0.85):
        number_mobs = VGroup(*map(Integer, numbers))
        number_mobs.arrange(RIGHT, buff=0.5)
        number_mobs.to_edge(LEFT)
        number_mobs.set_fill(BLACK),
        reductions = VGroup(*(
            Integer(number % modulus).next_to(mob, DOWN, 0.5)
            for number, mob in zip(numbers, number_mobs)
        ))
        reductions.set_color(color)
        grid = VGroup(number_mobs, reductions)

        top_label = Text("Numbers: ", fill_color=BLACK)
        low_label = Text(f"Mod {modulus}: ")
        top_label.next_to(number_mobs, LEFT, buff=0.5)
        low_label.next_to(reductions, LEFT, buff=0.5)
        low_label.set_color(color)

        grid.add(top_label, low_label)

        h_line = Line(ORIGIN, grid.get_width() * RIGHT)

        h_line.move_to(grid, LEFT)
        v_lines = VGroup(*(
            Line(grid.get_height() * UP, ORIGIN).set_x(
                0.5 * (number_mobs[i].get_right()[0] + number_mobs[i + 1].get_left()[0])
            ).align_to(grid, UP)
            for i in range(len(number_mobs) - 1)
        ))
        v_lines.set_stroke(BLACK, width=1)
        h_line.set_stroke(BLACK, width=1)

        grid.add(h_line, v_lines)
        grid.set_height(height, about_edge=LEFT)

        return grid

    def animate_reduction(self, array, beat_time=0.2):
        # Test
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


class SieveOfEratosthenes(InteractiveScene):
    def construct(self):
        # Initialize grid
        grid = Square().get_grid(10, 10, buff=0)
        grid.set_height(FRAME_HEIGHT - 1)
        grid.set_stroke(width=1)
        number_mobs = VGroup(*(
            Integer(i).set_height(0.3 * box.get_height()).move_to(box)
            for i, box in zip(it.count(1), grid)
        ))

        self.add(grid, number_mobs)

        # Run the sieve
        modulus = 2
        numbers = list(range(2, len(grid) + 1))
        for n in range(10):
            globals().update(locals())
            numbers = list(filter(lambda n: n % modulus != 0, numbers))
            to_remove = VGroup(*(
                mob
                for mob in number_mobs
                if mob.get_value() % modulus == 0
            ))
            rects = VGroup(*(
                SurroundingRectangle(tr, buff=0.1)
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
