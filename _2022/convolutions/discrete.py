from __future__ import annotations

from manim_imports_ext import *
from _2022.borwein.main import *
import scipy.signal


def get_die_faces(**kwargs):
    result = VGroup(*(DieFace(n, **kwargs) for n in range(1, 7)))
    result.arrange(RIGHT, buff=MED_LARGE_BUFF)
    return result


def get_aligned_pairs(group1, group2, n):
    return VGroup(*(
        VGroup(m1, m2)
        for m1 in group1
        for m2 in group2
        if m1.index + m2.index == n
    ))


def get_pair_rects(pairs, together=True, buff=SMALL_BUFF, corner_radius=0.1):
    if together:
        return VGroup(*(
            SurroundingRectangle(pair, buff=buff).round_corners(corner_radius)
            for pair in pairs
        ))
    else:
        return VGroup(*(
            VGroup(*(
                SurroundingRectangle(m, buff=buff).round_corners(corner_radius)
                for m in pair
            ))
            for pair in pairs
        ))


def get_row_shift(top_row, low_row, n):
    min_index = low_row[0].index
    max_index = top_row[-1].index
    max_sum = min_index + max_index
    if n <= max_sum:
        x_shift = top_row[n - 2 * min_index].get_x() - low_row[0].get_x()
    else:
        x_shift = top_row[-1].get_x() - low_row[n - max_sum].get_x()
    return low_row.animate.shift(x_shift * RIGHT)


def dist_to_bars(dist, bar_width=0.5, height=2.0, bar_colors=(BLUE_D, GREEN_D)):
    bars = Rectangle(width=bar_width).get_grid(1, len(dist), buff=0)
    bars.set_color_by_gradient(*bar_colors)
    bars.set_fill(opacity=1)
    bars.set_stroke(WHITE, 1)
    for bar, value, index in zip(bars, dist, it.count()):
        bar.set_height(value, stretch=True, about_edge=DOWN)
        bar.index = index
    bars.set_height(height, stretch=True)
    bars.center()
    return bars


def get_bar_dividing_lines(bars):
    v_lines = VGroup()
    for bar in bars:
        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        v_line.set_stroke(GREY_C, 1, 0.75)
        v_line.set_x(bar.get_left()[0])
        v_line.set_y(0)
        v_lines.add(v_line)
    v_lines.add(v_lines[-1].copy().set_x(bars.get_right()[0]))
    return v_lines


def add_labels_to_bars(bars, dist, width_ratio=0.7, **number_config):
    labels = VGroup(*(DecimalNumber(x, **number_config) for x in dist))
    labels.set_max_width(width_ratio * bars[0].get_width())
    for label, bar in zip(labels, bars):
        label.next_to(bar, UP, SMALL_BUFF)
        bar.value_label = label
        bar.push_self_into_submobjects()
        bar.add(label)
    return bars


def prod(values):
    return reduce(op.mul, values, 1)


def get_lagrange_polynomial(data):
    def poly(x):
        return sum(
            y0 * prod(
                (x - x1) for x1, y1 in data if x1 != x0
            ) / prod(
                (x0 - x1) for x1, y1 in data if x1 != x0
            )
            for x0, y0 in data
        )

    return poly


def kinked_function(x):
    if x < -2:
        return 0
    elif x < -1:
        return -x - 2
    elif x < 1:
        return x
    elif x < 2:
        return -x + 2
    else:
        return 0


# Introduction

class WaysToCombine(InteractiveScene):
    def construct(self):
        # Functions
        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        v_line.set_stroke(WHITE, 2)

        axes1, axes2, axes3 = all_axes = VGroup(*(
            Axes((-3, 3), (-1, 1), height=2.0, width=FRAME_WIDTH * 0.5 - 1)
            for x in range(3)
        ))
        all_axes.arrange(DOWN, buff=LARGE_BUFF)
        all_axes.set_height(FRAME_HEIGHT - 1)
        all_axes.move_to(midpoint(ORIGIN, RIGHT_SIDE))

        def f(x):
            return kinked_function(x)

        def g(x):
            return np.exp(-0.5 * x**2)

        f_graph = axes1.get_graph(f).set_stroke(BLUE, 2)
        g_graph = axes2.get_graph(g).set_stroke(YELLOW, 2)
        sum_graph = axes3.get_graph(lambda x: f(x) + g(x)).set_stroke(TEAL, 2)
        prod_graph = axes3.get_graph(lambda x: f(x) * g(x)).set_stroke(TEAL, 2)
        x_samples = np.linspace(*axes1.x_range[:2], 100)
        f_samples = list(map(f, x_samples))
        g_samples = list(map(g, x_samples))
        conv_samples = np.convolve(f_samples, g_samples, mode='same')
        conv_samples *= 0.1  # Artificially scale down
        conv_points = axes3.c2p(x_samples, conv_samples)
        conv_graph = VMobject().set_points_smoothly(conv_points)
        conv_graph.set_stroke(TEAL, 2)

        kw = dict(font_size=30, tex_to_color_map={"f": BLUE, "g": YELLOW})
        f_label = Tex("f(x)", **kw).move_to(axes1.get_corner(UL), UL)
        g_label = Tex("g(x)", **kw).move_to(axes2.get_corner(UL), UL)
        sum_label = Tex("[f + g](x)", **kw).move_to(axes3.get_corner(UL), UL)
        prod_label = Tex(R"[f \cdot g](x)", **kw).move_to(axes3.get_corner(UL), UL)
        conv_label = Tex(R"[f * g](x)", **kw).move_to(axes3.get_corner(UL), UL)
        graph_labels = VGroup(f_label, g_label, sum_label, prod_label, conv_label)

        # Sequences
        seq1 = np.array([1, 2, 3, 4])
        seq2 = np.array([5, 6, 7, 8])
        kw = dict(
            font_size=48,
            tex_to_color_map={"a": BLUE, "b": YELLOW},
            isolate=[",", "[", "]"]
        )
        seq1_tex, seq2_tex, sum_seq, prod_seq, conv_seq = seqs = VGroup(
            OldTex(f"a = {list(seq1)}", **kw),
            OldTex(f"b = {list(seq2)}", **kw),
            OldTex(f"a + b = {list(seq1 + seq2)}", **kw),
            OldTex(Rf"a \cdot b = {list(seq1 * seq2)}", **kw),
            OldTex(Rf"a * b = {list(np.convolve(seq1, seq2))}", **kw),
        )
        seqs.move_to(midpoint(ORIGIN, LEFT_SIDE))
        seq1_tex.match_y(axes1)
        seq2_tex.match_y(axes2)
        for seq in seqs[2:]:
            seq.match_y(axes3)

        # Operation labels
        op_labels = VGroup()
        left_op_labels = VGroup(*map(Text, ["Addition", "Multiplication", "Convolution"]))
        left_op_labels.set_color(TEAL)
        for left_op_label, seq, graph_label in zip(left_op_labels, seqs[2:], graph_labels[2:]):
            left_op_label.next_to(seq, UP, MED_LARGE_BUFF, aligned_edge=LEFT)
            right_op_label = left_op_label.copy().scale(0.7)
            right_op_label.next_to(graph_label, UP, aligned_edge=LEFT)
            op_labels.add(VGroup(left_op_label, right_op_label))

        # Introduce
        kw = dict(lag_ratio=0.7)
        self.play(
            LaggedStartMap(Write, seqs[:2], **kw),
            run_time=2,
        )
        self.play(
            FadeIn(v_line),
            LaggedStartMap(FadeIn, all_axes[:2], **kw),
            LaggedStartMap(ShowCreation, VGroup(f_graph, g_graph), **kw),
            LaggedStartMap(FadeIn, graph_labels[:2], **kw),
            run_time=2
        )
        self.wait()

        # Ways to combine?
        all_boxes = VGroup(*(SurroundingRectangle(m[2:]) for m in seqs[:3]))
        all_boxes.set_stroke(width=2)
        boxes = all_boxes[:2]
        boxes[0].set_color(BLUE)
        boxes[1].set_color(YELLOW)
        mystery_box = all_boxes[2]
        mystery_box.set_color(GREEN)
        q_marks = Text("????")
        q_marks.space_out_submobjects(1.5)
        q_marks.move_to(mystery_box)
        box_arrows = VGroup(*(
            Arrow(box.get_right(), mystery_box.get_corner(UR), path_arc=-PI / 3)
            for box in boxes
        ))

        self.play(
            *map(Write, box_arrows),
            FadeIn(boxes),
            *(
                Transform(
                    box.copy().set_stroke(width=0, opacity=0),
                    mystery_box.copy(),
                    path_arc=-PI / 3,
                    remover=True
                )
                for box in boxes
            )
        )
        self.add(mystery_box)
        comb_graph = sum_graph.copy()
        self.play(
            Write(q_marks),
            ReplacementTransform(axes1.copy(), axes3),
            Transform(axes2.copy(), axes3.copy(), remover=True),
            ReplacementTransform(f_graph.copy(), comb_graph),
            Transform(g_graph.copy(), comb_graph.copy(), remover=True),
        )
        self.play(Transform(comb_graph, prod_graph))
        self.play(Transform(comb_graph, conv_graph))
        self.play(LaggedStartMap(FadeOut, VGroup(
            *boxes, *box_arrows, mystery_box, q_marks, comb_graph
        )))

        # Sums and products
        tuples = [
            (sum_seq, sum_label, axes3, sum_graph, op_labels[0]),
            (prod_seq, prod_label, axes3, prod_graph, op_labels[1])
        ]
        for seq, label, axes, graph, op_label in tuples:
            self.play(LaggedStart(
                TransformMatchingShapes(
                    VGroup(*seq1_tex[:2], *seq2_tex[:2]).copy(),
                    seq[:4]
                ),
                TransformMatchingShapes(graph_labels[:2].copy(), label),
                FadeIn(op_label, DOWN)
            ))
            self.add(axes)
            # Go point by point
            value_rects = VGroup(*(
                VGroup(*map(SurroundingRectangle, s[-8::2]))
                for s in [seq1_tex, seq2_tex, seq]
            ))
            dots = Group(*(GlowDot(color=WHITE) for x in range(3)))
            self.play(
                *map(FadeIn, [seq[4], seq[-1]]),
                *(
                    VFadeInThenOut(rects, lag_ratio=0.5)
                    for rects in value_rects
                ),
                LaggedStart(*(
                    FadeIn(seq[n:n + 2 if n < 11 else n + 1])
                    for n in range(5, 12, 2)
                ), lag_ratio=0.5),
                run_time=2
            )
            self.play(
                ShowCreation(graph, rate_func=linear),
                UpdateFromFunc(dots[2], lambda m: m.move_to(graph.get_end())),
                *(
                    UpdateFromAlphaFunc(dot, lambda d, a: d.set_opacity(min(10 * a * (1 - a), 1)))
                    for dot in dots
                ),
                *(
                    MoveAlongPath(dot, graph)
                    for dot, graph in zip(dots, [f_graph, g_graph])
                ),
                run_time=4
            )
            self.wait()
            self.play(*map(FadeOut, [seq, label, graph, op_label]))

        # Convolutions
        self.play(LaggedStart(
            TransformMatchingShapes(
                VGroup(*seq1_tex[:2], *seq2_tex[:2]).copy(),
                conv_seq[:4],
            ),
            FadeIn(op_labels[2][0], DOWN),
            TransformMatchingShapes(graph_labels[:2].copy(), conv_label),
            FadeIn(op_labels[2][1], DOWN),
        ))
        self.play(FadeIn(conv_seq[4:], lag_ratio=0.2, run_time=2))

        # Hint at computation
        nums1 = seq1_tex[3::2]
        nums2 = seq2_tex[3::2]
        for mobs in nums1, nums2:
            for i, mob in enumerate(mobs):
                mob.index = i
        nums3 = conv_seq[5::2]

        nums1.set_color(BLUE)
        nums2.set_color(YELLOW)

        last_group = VGroup()
        for n, num3 in enumerate(nums3):
            rect = SurroundingRectangle(num3, buff=SMALL_BUFF)
            rect.set_stroke(TEAL, 2)
            rect.round_corners()
            pairs = get_aligned_pairs(nums1, nums2, n)
            lines = VGroup(*(Line(m1, m2) for m1, m2 in pairs))
            lines.set_stroke(TEAL, 2)

            group = VGroup(rect, lines)
            self.play(FadeIn(group), FadeOut(last_group), run_time=0.25)
            self.wait(0.25)
            last_group = group
        self.play(FadeOut(last_group, run_time=0.5))
        self.wait()

        # Conv graph
        self.play(ShowCreation(conv_graph, run_time=3))
        self.wait()
        self.play(MoveAlongPath(GlowDot(color=WHITE), conv_graph, run_time=5, remover=True))
        self.wait()


# Probabilities


class DiceExample(InteractiveScene):
    def construct(self):
        # Start showing grid
        blue_dice = get_die_faces(fill_color=BLUE_E, dot_color=WHITE)
        red_dice = get_die_faces(fill_color=RED_E, dot_color=WHITE)
        VGroup(blue_dice, red_dice).arrange(DOWN, buff=LARGE_BUFF)
        grid = Square().get_grid(6, 6, buff=0)
        grid.set_height(6)
        grid.to_edge(LEFT, buff=2.0)
        grid.shift(0.5 * DOWN)
        grid.set_stroke(WHITE, 1)

        blue_dice.save_state()
        red_dice.save_state()

        self.play(LaggedStart(
            FadeIn(blue_dice, lag_ratio=0.1, shift=0.25 * UP, run_time=2, rate_func=overshoot),
            FadeIn(red_dice, lag_ratio=0.1, shift=0.25 * UP, run_time=2, rate_func=overshoot),
            lag_ratio=0.2
        ))
        self.wait()
        self.play(
            Write(grid, run_time=2, stroke_color=YELLOW, stroke_width=4),
            *(
                die.animate.set_width(0.6 * square.get_width()).next_to(square, UP, SMALL_BUFF)
                for die, square in zip(blue_dice, grid[:6])
            ),
            *(
                die.animate.set_width(0.6 * square.get_width()).next_to(square, LEFT, SMALL_BUFF)
                for die, square in zip(red_dice, grid[0::6])
            ),
        )

        # Add all mini dice
        mini_dice = VGroup()
        for n, square in enumerate(grid):
            j, i = n // 6, n % 6
            blue = blue_dice[i].copy()
            red = red_dice[j].copy()
            blue.sum = i + j + 2
            red.sum = i + j + 2
            blue.generate_target()
            red.generate_target()
            group = VGroup(blue.target, red.target)
            group.set_stroke(width=1)
            group.arrange(RIGHT, buff=SMALL_BUFF)
            group.set_width(square.get_width() * 0.8)
            group.move_to(square)
            mini_dice.add(blue, red)

        combinations_label = VGroup(
            OldTex("6^2 = "), Integer(36), Text("Combinations")
        )
        combinations_label.arrange(RIGHT, aligned_edge=DOWN)
        combinations_label.to_edge(RIGHT)

        self.play(
            LaggedStartMap(MoveToTarget, mini_dice, lag_ratio=0.02, run_time=2),
            FadeIn(combinations_label[0]),
            CountInFrom(combinations_label[1], 1, run_time=2),
            VFadeIn(combinations_label[1]),
            FadeIn(combinations_label[2], 0),
        )
        self.wait()

        # Go Through diagonals
        last_prob_label = VMobject()
        for n in range(2, 13):
            to_fade = VGroup()
            to_highlight = VGroup()
            for die in mini_dice:
                if die.sum == n:
                    to_highlight.add(die)
                else:
                    to_fade.add(die)
            pairs = VGroup(*(VGroup(m1, m2) for m1, m2 in zip(to_highlight[::2], to_highlight[1::2])))
            num = len(pairs)
            prob_label = self.get_p_sum_expr(n, Rf"\frac{{{num}}}{{36}}")
            prob_label.next_to(combinations_label, UP, buff=1.5)

            self.play(
                FadeIn(prob_label, UP),
                FadeOut(last_prob_label, UP),
                to_highlight.animate.set_opacity(1),
                to_fade.animate.set_opacity(0.2),
                blue_dice.animate.set_opacity(0.5),
                red_dice.animate.set_opacity(0.5),
            )
            if n <= 4:
                self.play(
                    LaggedStart(*(
                        FlashAround(pair.copy(), remover=True, time_width=1, run_time=1.5)
                        for pair in pairs
                    ), lag_ratio=0.2),
                )
                self.wait()
            last_prob_label = prob_label
            self.wait()

        # Reset
        self.play(
            FadeOut(grid),
            FadeOut(mini_dice, lag_ratio=0.01),
            FadeOut(combinations_label, RIGHT),
            FadeOut(prob_label, RIGHT),
            blue_dice.animate.restore(),
            red_dice.animate.restore(),
        )
        self.wait()

        # Slide rows across
        self.play(
            Rotate(red_dice, PI),
        )
        self.wait()

        last_prob_label = VMobject()
        last_rects = VMobject()
        for n in range(2, 13):
            pairs = self.get_aligned_pairs(blue_dice, red_dice, n)
            prob_label = self.get_p_sum_expr(n, Rf"\frac{{{len(pairs)}}}{{36}}")
            prob_label.to_edge(UP)

            self.play(
                self.get_dice_shift(blue_dice, red_dice, n),
                FadeOut(last_rects, run_time=0.5)
            )
            rects = get_pair_rects(pairs)
            self.play(
                FadeOut(last_prob_label, UP),
                FadeIn(prob_label, UP),
                LaggedStartMap(ShowCreation, rects, lag_ratio=0.2),
            )
            self.wait()

            last_prob_label = prob_label
            last_rects = rects

        # Realign
        self.play(
            FadeOut(last_rects),
            FadeOut(last_prob_label, UP),
            red_dice.animate.next_to(blue_dice, DOWN, buff=1.5),
        )

        # Show implicit probabilities, and alternates
        all_dice = VGroup(*blue_dice, *red_dice)
        sixths = VGroup(*(
            OldTex("1 / 6", font_size=36).next_to(die, UP, SMALL_BUFF)
            for die in all_dice
        ))
        sixths.set_stroke(WHITE, 0)
        blue_probs, red_probs = [
            np.random.uniform(0, 1, 6)
            for x in range(2)
        ]
        for probs in blue_probs, red_probs:
            probs[:] = (probs / probs.sum()).round(2)
            probs[-1] = 1.0 - probs[:-1].sum()  # Ensure it's a valid distribution

        new_prob_labels = VGroup()
        all_dice.generate_target()
        for die, prob in zip(all_dice.target, (*blue_probs, *red_probs)):
            label = DecimalNumber(prob, font_size=36)
            label.next_to(die, UP, SMALL_BUFF)
            die.set_opacity(prob / (2 / 6))
            new_prob_labels.add(label)

        question = Text("Non-uniform probabilities?")
        question.to_edge(UP)

        self.play(LaggedStartMap(Write, sixths, lag_ratio=0.1))
        self.wait()
        self.play(
            Write(question, run_time=1),
            FadeOut(sixths, 0.25 * UP, lag_ratio=0.03, run_time=4),
            FadeIn(new_prob_labels, 0.25 * UP, lag_ratio=0.03, run_time=4),
            MoveToTarget(all_dice, run_time=3)
        )
        self.wait()

        for die, prob_label in zip(all_dice, new_prob_labels):
            die.prob_label = prob_label
            die.prob = prob_label.get_value()
            die.add(prob_label)

        # March!
        last_rects = VMobject()
        last_prob_label = question
        n = 2
        while n < 13:
            pairs = self.get_aligned_pairs(blue_dice, red_dice, n)
            prob_label = self.get_p_sum_expr(n, rhs=" ")
            rhs = self.get_conv_rhs(pairs, prob_label)
            VGroup(prob_label, rhs).center().to_edge(UP)

            self.play(
                self.get_dice_shift(blue_dice, red_dice, n),
                FadeOut(last_rects, run_time=0.5),
                FadeOut(last_prob_label, 0.5 * UP),
                FadeIn(prob_label, 0.5 * UP),
            )
            rects = self.get_pair_rects(pairs)
            self.play(FadeIn(rects, lag_ratio=0.2))
            self.play(TransformMatchingShapes(
                VGroup(*(
                    VGroup(blue.prob_label, red.prob_label)
                    for blue, red in pairs
                )).copy(),
                rhs
            ))
            self.wait()

            if n == 4 and isinstance(blue_dice[0].prob_label, DecimalNumber):
                # Make more general
                blue_labels = VGroup(*(OldTex(f"a_{{{i}}}", font_size=42) for i in range(1, 7)))
                red_labels = VGroup(*(OldTex(f"b_{{{i}}}", font_size=42) for i in range(1, 7)))
                blue_labels.set_color(BLUE)
                red_labels.set_color(RED)
                old_prob_labels = VGroup()
                for die, label in zip(all_dice, (*blue_labels, *red_labels)):
                    label.next_to(die[0], UP, SMALL_BUFF)
                    die.remove(die.prob_label)
                    old_prob_labels.add(die.prob_label)
                    die.prob_label = label
                    die.add(label)
                self.play(
                    FadeIn(VGroup(*blue_labels, *red_labels), shift=0.25 * UP, lag_ratio=0.04),
                    FadeOut(old_prob_labels, shift=0.25 * UP, lag_ratio=0.04),
                    FadeOut(rhs, time_span=(0, 1)),
                    run_time=4
                )
            else:
                n += 1

            last_prob_label = VGroup(prob_label, rhs)
            last_rects = rects

        # Show all formulas
        n_range = list(range(2, 13))
        lhss = VGroup(*(self.get_p_sum_expr(n) for n in n_range))
        pairss = VGroup(*(self.get_aligned_pairs(blue_dice, red_dice, n) for n in n_range))
        rhss = VGroup(*(self.get_conv_rhs(pairs, lhs) for pairs, lhs in zip(pairss, lhss)))
        prob_labels = VGroup()
        for lhs, rhs in zip(lhss, rhss):
            prob_labels.add(VGroup(lhs, rhs))
        prob_labels.arrange(DOWN, aligned_edge=LEFT)
        prob_labels.set_height(FRAME_HEIGHT - 1)
        prob_labels.to_edge(LEFT)

        dice = VGroup(blue_dice, red_dice)
        dice.generate_target()
        dice.target[1].rotate(PI)
        for m in dice.target[1]:
            m.prob_label.rotate(PI)
            m.prob_label.next_to(m, UP, SMALL_BUFF)
        dice.target.arrange(DOWN, buff=MED_LARGE_BUFF)
        dice.target.set_width(5)
        dice.target.to_edge(RIGHT)
        dice.target.to_corner(DR)
        self.play(
            LaggedStartMap(FadeIn, prob_labels[:-1], lag_ratio=0.2, run_time=2),
            FadeTransform(last_prob_label, prob_labels[-1], time_span=(0.5, 2.0)),
            MoveToTarget(dice, path_arc=PI / 2),
            FadeOut(last_rects),
        )
        self.wait()

        # Name convolution
        a_rect = SurroundingRectangle(VGroup(*(die.prob_label for die in blue_dice)))
        b_rect = SurroundingRectangle(VGroup(*(die.prob_label for die in red_dice)))
        rhs_rects = VGroup(*(SurroundingRectangle(rhs) for rhs in rhss))
        VGroup(a_rect, b_rect, *rhs_rects).set_stroke(YELLOW, 2)

        conv_name = OldTexText(
            R"Convolution of ", "$(a_i)$", " and ", "$(b_i)$",
            tex_to_color_map={"$(a_i)$": BLUE, "$(b_i)$": RED}
        )
        conv_eq = Tex(
            R"(a * b)_n = \sum_{\substack{i, j \\ i + j = n}} a_i \cdot b_j",
            isolate=["a_i", "*", "b_j", "(a * b)_n"]
        )
        conv_label = VGroup(conv_name, conv_eq)
        conv_label.arrange(DOWN, buff=LARGE_BUFF)
        conv_label.scale(0.9)
        conv_label.to_corner(UR)

        self.play(
            LaggedStartMap(
                ShowCreationThenFadeOut,
                VGroup(a_rect, b_rect),
                lag_ratio=0.5,
                run_time=3,
            ),
            LaggedStartMap(
                FadeIn,
                conv_name[1:5:2],
                shift=UP,
                lag_ratio=0.5,
                run_time=2,
            )
        )
        self.wait()
        self.play(FadeIn(conv_name[0:4:2], lag_ratio=0.1))
        self.play(LaggedStartMap(
            ShowCreationThenFadeOut,
            rhs_rects,
            lag_ratio=0.1,
            run_time=5
        ))
        self.wait()

        # Show grid again
        diagonals = VGroup()
        plusses = VGroup()
        rhss.save_state()
        for n, rhs in zip(it.count(2), rhss):
            diag = VGroup()
            for i in range(0, len(rhs), 4):
                diag.add(rhs[i:i + 3])
                if i > 0:
                    plusses.add(rhs[i - 1])
            diagonals.add(diag)

        diagonals.generate_target()
        for k, square in enumerate(grid):
            i = k // 6
            j = k % 6
            i2 = j if (i + j <= 5) else 5 - i
            diagonals.target[i + j][i2].move_to(square)

        blue_dice.save_state()
        blue_dice.generate_target()
        red_dice.save_state()
        red_dice.generate_target()
        for dice, squares, vect in (blue_dice, grid[:6], UP), (red_dice, grid[::6], LEFT):
            dice.save_state()
            dice.generate_target()
            for die, square in zip(dice.target, squares):
                die.scale(0.75)
                die.next_to(square, vect, SMALL_BUFF)

        self.play(
            plusses.animate.set_opacity(0),
            FadeOut(lhss, LEFT, run_time=0.5, lag_ratio=0.01),
            MoveToTarget(diagonals),
            MoveToTarget(blue_dice),
            MoveToTarget(red_dice),
            run_time=2
        )
        self.play(FadeIn(grid, lag_ratio=0.01))
        self.add(diagonals)
        self.wait()
        for n in range(len(diagonals)):
            diagonals.generate_target()
            diagonals.target.set_opacity(0.2)
            diagonals.target[n].set_opacity(1)
            self.play(MoveToTarget(diagonals))
        self.wait()
        self.play(
            Restore(blue_dice),
            Restore(red_dice),
            diagonals.animate.set_opacity(1)
        )
        self.remove(diagonals)
        self.play(
            FadeOut(grid),
            Restore(rhss),
            FadeIn(lhss, time_span=(1, 2), lag_ratio=0.01),
            run_time=2
        )
        self.wait()

        # Write equation
        self.wait(2)
        self.play(FadeIn(conv_eq, DOWN))
        self.wait()

        # Highlight example
        n = 4
        pairs = self.get_aligned_pairs(blue_dice, red_dice, n + 2)
        pair_rects = get_pair_rects(pairs, together=False)

        self.play(
            prob_labels[:n].animate.set_opacity(0.35),
            prob_labels[n + 1:].animate.set_opacity(0.35),
        )
        self.wait()
        self.play(ShowSubmobjectsOneByOne(pair_rects, remover=True, rate_func=linear, run_time=3))
        self.wait()

        # Alternate formula notation
        alt_rhs = OldTex(R"\sum_{i = 1}^6 a_i \cdot b_{n - i}")
        alt_rhs.scale(0.9)
        alt_rhs.move_to(conv_eq[7], LEFT)

        self.play(
            FadeIn(alt_rhs, 0.5 * DOWN),
            conv_eq[7:].animate.shift(1.5 * DOWN).set_opacity(0.5),
            dice.animate.to_edge(DOWN)
        )
        self.wait()

    def get_p_sum_expr(self, n, rhs=" "):
        raw_expr = Tex(
            fR"P\big(O + O = {n}\big) = \, {rhs}",
            isolate=("O", str(n), "=", rhs)
        )
        for index, color in zip([2, 4], [BLUE_E, RED_E]):
            square = DieFace(1, fill_color=color, stroke_width=1)[0]
            square.replace(raw_expr[index])
            square.match_y(raw_expr[3])
            raw_expr.replace_submobject(index, square)
        return raw_expr

    def get_dice_shift(self, top_dice, low_dice, n):
        return get_row_shift(top_dice, low_dice, n)

    def get_aligned_pairs(self, top_dice, low_dice, n):
        return get_aligned_pairs(top_dice, low_dice, n)

    def get_pair_rects(self, pairs, together=True):
        return get_pair_rects(pairs, together)

    def get_conv_rhs(self, pairs, prob_label):
        rhs = VGroup()
        for (blue, red) in pairs:
            rhs.add(blue.prob_label.copy().set_color(BLUE))
            rhs.add(OldTex(R"\cdot"))
            rhs.add(red.prob_label.copy().set_color(RED))
            rhs.add(OldTex("+"))
        rhs.remove(rhs[-1])
        rhs.arrange(RIGHT, buff=SMALL_BUFF)
        rhs.next_to(prob_label, RIGHT, buff=0.2)
        return rhs


class SimpleExample(InteractiveScene):
    def construct(self):
        # Question
        question = Text("What is")
        conv = Tex("(1, 2, 3) * (4, 5, 6)")
        group = VGroup(question, conv)
        group.arrange(DOWN)
        group.to_edge(UP)

        self.play(Write(question, run_time=1.5), FadeIn(conv, 0.5 * DOWN, time_span=(0.5, 1.5)))
        self.wait()

        # Blocks
        top_row = Square(side_length=0.75).get_grid(1, 3, buff=0)
        top_row.set_stroke(GREY_B, 2)
        top_row.set_fill(GREY_E, 1)
        low_row = top_row.copy()
        for row, values in (top_row, range(1, 4)), (low_row, range(4, 7)):
            for index, value, square in zip(it.count(), values, row):
                value_label = Integer(value)
                value_label.move_to(square)
                square.value_label = value_label
                square.add(value_label)
                square.value = value
                square.index = index

        VGroup(top_row, low_row).arrange(RIGHT, buff=LARGE_BUFF)

        self.play(
            TransformMatchingShapes(conv[1:6:2].copy(), top_row),
            TransformMatchingShapes(conv[9:14:2].copy(), low_row),
        )
        self.wait()

        # Labels
        self.add_block_labels(top_row, "a", BLUE)
        self.add_block_labels(low_row, "b", RED)

        # Set up position
        top_row.generate_target()
        low_row.generate_target()
        low_row.target.rotate(PI)
        for square in low_row.target:
            square.value_label.rotate(PI)
            square.label.rotate(PI)
        top_row.target.center()
        low_row.target.next_to(top_row.target, DOWN, MED_LARGE_BUFF)

        conv_result = np.convolve([1, 2, 3], [4, 5, 6])
        rhs_args = ["=", R"\big("]
        for k in conv_result:
            rhs_args.append(str(k))
            rhs_args.append(",")
        rhs_args[-1] = R"\big)"
        rhs = OldTex(*rhs_args)
        rhs[1:].set_color(YELLOW)
        conv.generate_target()
        group = VGroup(conv.target, rhs)
        group.arrange(RIGHT, buff=0.2)
        group.next_to(top_row, UP, buff=2),

        self.play(LaggedStart(
            MoveToTarget(top_row),
            MoveToTarget(low_row, path_arc=PI),
            MoveToTarget(conv),
            Write(VGroup(*rhs[:2], rhs[-1])),
            FadeOut(question, UP),
        ))
        self.wait()

        # March!
        c_labels = VGroup()
        for n in range(len(conv_result)):
            self.play(get_row_shift(top_row, low_row, n))

            pairs = get_aligned_pairs(top_row, low_row, n)
            label_pairs = VGroup(*(
                VGroup(m1.value_label, m2.value_label)
                for m1, m2 in pairs
            ))
            new_label_pairs = label_pairs.copy()
            expr = VGroup()
            symbols = VGroup()
            for label_pair in new_label_pairs:
                label_pair.arrange(RIGHT, buff=MED_SMALL_BUFF)
                label_pair.next_to(expr, RIGHT, SMALL_BUFF)
                dot = OldTex(R"\dot")
                dot.move_to(label_pair)
                plus = OldTex("+")
                plus.next_to(label_pair, RIGHT, SMALL_BUFF)
                expr.add(*label_pair, dot, plus)
                symbols.add(dot, plus)
            symbols[-1].scale(0, about_point=symbols[-2].get_right())
            expr.next_to(label_pairs, UP, LARGE_BUFF)
            c_label = OldTex(f"c_{n}", font_size=30, color=YELLOW).next_to(rhs[2 * n + 2], UP)

            rects = VGroup(*(
                SurroundingRectangle(lp, buff=0.2).set_stroke(YELLOW, 1).round_corners()
                for lp in label_pairs
            ))
            self.play(FadeIn(rects, lag_ratio=0.5))
            self.play(
                LaggedStart(*(
                    TransformFromCopy(lp, nlp)
                    for lp, nlp in zip(label_pairs, new_label_pairs)
                ), lag_ratio=0.5),
                Write(symbols),
            )
            self.wait()
            anims = [
                FadeTransform(expr.copy(), rhs[2 * n + 2]),
                c_labels.animate.set_opacity(0.35),
                FadeIn(c_label)
            ]
            if n < 4:
                anims.append(Write(rhs[2 * n + 3]))
            self.play(*anims)
            self.wait()
            self.play(FadeOut(expr), FadeOut(rects))

            c_labels.add(c_label)
        self.play(FadeOut(c_labels))

        # Grid of values
        equation = VGroup(conv, rhs)
        values1 = VGroup(*(block.value_label for block in top_row)).copy()
        values2 = VGroup(*(block.value_label for block in low_row)).copy()

        grid = Square(side_length=1.0).get_grid(3, 3, buff=0)
        grid.set_stroke(WHITE, 2)
        grid.set_fill(GREY_E, 1.0)
        grid.move_to(DL)

        self.play(
            Write(grid, time_span=(0.5, 2.0)),
            LaggedStart(
                *(
                    value.animate.next_to(square, UP, buff=0.2)
                    for value, square in zip(values1, grid[:3])
                ),
                *(
                    value.animate.next_to(square, LEFT, buff=0.2)
                    for value, square in zip(values2, grid[0::3])
                ),
                run_time=2
            ),
            *(
                MaintainPositionRelativeTo(block, value)
                for row, values in [(top_row, values1), (low_row, values2)]
                for block, value in zip(row, values)
            ),
            VFadeOut(top_row),
            VFadeOut(low_row),
            equation.animate.center().to_edge(UP)
        )

        # Products
        products = VGroup()
        diag_groups = VGroup().replicate(5)
        for n, square in enumerate(grid):
            i, j = n // 3, n % 3
            v1 = values1[j]
            v2 = values2[i]
            product = Integer(v1.get_value() * v2.get_value())
            product.match_height(v1)
            product.move_to(square)
            product.factors = (v1, v2)
            square.product = product
            products.add(product)
            diag_groups[i + j].add(product)

        products.set_color(GREEN)

        self.play(LaggedStart(*(
            ReplacementTransform(factor.copy(), product)
            for product in products
            for factor in product.factors
        ), lag_ratio=0.1))
        self.wait()

        # Circle diagonals
        products.rotate(PI / 4)
        ovals = VGroup()
        radius = 0.3
        for diag in diag_groups:
            oval = SurroundingRectangle(diag, buff=0.19)
            oval.set_width(2 * radius, stretch=True)
            oval.set_stroke(YELLOW, 2)
            oval.round_corners(radius=radius)
            ovals.add(oval)
        VGroup(products, ovals).rotate(-PI / 4)
        ovals[0].become(Circle(radius=radius).match_style(ovals[0]).move_to(products[0]))

        arrows = VGroup(*(
            Vector(0.5 * UP).next_to(part, DOWN)
            for part in rhs[2::2]
        ))
        arrows.set_color(YELLOW)

        curr_arrow = arrows[0].copy()
        curr_arrow.shift(0.5 * DOWN).set_opacity(0)
        for n, oval, arrow in zip(it.count(), ovals, arrows):
            self.play(
                ShowCreation(oval),
                ovals[:n].animate.set_stroke(opacity=0.25),
                Transform(curr_arrow, arrow)
            )
            self.wait(0.5)
        self.play(ovals.animate.set_stroke(opacity=0.25), FadeOut(curr_arrow))
        self.wait()

        grid_group = VGroup(grid, values1, values2, products, ovals)

        # Show polynomial
        polynomial_eq = Tex(
            R"\left(1 + 2x + 3x^2\right)\left(4 + 5x + 6x^2\right)"
            R"={4} + {13}x + {28}x^2 + {27}x^3 + {18}x^4",
            tex_to_color_map=dict(
                (f"{{{n}}}", YELLOW)
                for n in conv_result
            )
        )
        polynomial_eq.next_to(equation, DOWN, MED_LARGE_BUFF)

        self.play(
            FadeIn(polynomial_eq, DOWN),
            grid_group.animate.center().to_edge(DOWN)
        )
        self.wait()

        # Replace terms
        self.play(grid_group.animate.set_height(4.5, about_edge=DOWN))

        for i, value in zip((*range(3), *range(3)), (*values1, *values2)):
            tex = ["", "x", "x^2"][i]
            value.target = Tex(f"{value.get_value()}{tex}")
            value.target.scale(value.get_height() / value.target[0].get_height())
            value.target.move_to(value, DOWN)
        values2[1].target.align_to(values2[0].target, RIGHT)
        values2[2].target.align_to(values2[0].target, RIGHT)
        for n, diag_group in enumerate(diag_groups):
            tex = ["", "x", "x^2", "x^3", "x^4"][n]
            for product in diag_group:
                product.target = Tex(f"{product.get_value()}{tex}")
                product.target.match_style(product)
                product.target.scale(0.9)
                product.target.move_to(product)

        eq_values1 = VGroup(polynomial_eq[1], polynomial_eq[3:5], polynomial_eq[6:9])
        eq_values2 = VGroup(polynomial_eq[11], polynomial_eq[13:15], polynomial_eq[16:19])

        for values, eq_values in [(values1, eq_values1), (values2, eq_values2)]:
            self.play(
                LaggedStart(*(TransformMatchingShapes(ev.copy(), v.target) for ev, v in zip(eq_values, values))),
                LaggedStart(*(FadeTransform(v, v.target[0]) for v in values)),
            )
        self.wait()
        old_rects = VGroup()
        for n, prod in enumerate(products):
            new_rects = VGroup(
                SurroundingRectangle(values1[n % 3].target),
                SurroundingRectangle(values2[n // 3].target),
            )
            new_rects.set_stroke(GREEN, 2)
            self.play(
                FadeIn(new_rects),
                FadeOut(old_rects),
                FadeTransform(prod, prod.target[:len(prod)]),
                FadeIn(prod.target[len(prod):], scale=2),
                FlashAround(prod.target, time_width=1),
                run_time=1.0
            )
            old_rects = new_rects
        self.play(FadeOut(old_rects))

        # Show diagonals again
        arrows = VGroup(*(
            Vector(0.5 * UP).next_to(polynomial_eq.select_part(f"{{{n}}}"), DOWN, buff=SMALL_BUFF)
            for n in conv_result
        ))
        arrows.set_color(YELLOW)

        curr_arrow = arrows[0].copy().shift(DOWN).set_opacity(0)
        for n, oval, arrow in zip(it.count(), ovals, arrows):
            self.play(
                oval.animate.set_stroke(opacity=1),
                Transform(curr_arrow, arrow),
                ovals[:n].animate.set_stroke(opacity=0.25),
            )
            self.wait(0.5)
        self.play(
            FadeOut(curr_arrow),
            ovals.animate.set_stroke(opacity=0.5)
        )

    def add_block_labels(self, blocks, letter, color=BLUE, font_size=30):
        labels = VGroup()
        for n, square in enumerate(blocks):
            label = OldTex(f"{letter}_{{{n}}}", font_size=font_size)
            label.set_color(color)
            label.next_to(square, UP, SMALL_BUFF)
            square.label = label
            square.add(label)
            labels.add(label)
        return labels


class MovingAverageExample(InteractiveScene):
    dist1 = [*5 * [0.1], *5 * [1], *5 * [0.1], *5 * [1], *5 * [0.1]]
    dist2 = 5 * [0.2]
    march_anim_run_time = 1.0
    always_preview_result = True

    def construct(self):
        # All bars
        self.camera.frame.scale(1.01)
        dist1 = np.array(self.dist1)
        dist2 = np.array(self.dist2)
        conv_dist = np.convolve(dist1, dist2)

        top_bars = dist_to_bars(dist1, height=1.5, bar_colors=(BLUE_D, TEAL_D))
        low_bars = dist_to_bars(dist2, height=1.5, bar_colors=(RED_D, GOLD_E))
        conv_bars = dist_to_bars(conv_dist, height=1.5, bar_colors=(GREEN_D, YELLOW_E))

        top_bars.center().to_edge(UP)
        low_bars.stretch(max(dist2), 1, about_edge=DOWN)
        low_bars.arrange(LEFT, aligned_edge=DOWN, buff=0)
        low_bars.next_to(top_bars, DOWN, buff=1.2, aligned_edge=LEFT)
        conv_bars.match_x(top_bars)
        conv_bars.to_edge(DOWN, LARGE_BUFF)
        v_lines = get_bar_dividing_lines(conv_bars)

        add_labels_to_bars(top_bars, dist1, num_decimal_places=1, width_ratio=0.4)
        add_labels_to_bars(low_bars, dist2, num_decimal_places=1, width_ratio=0.4)
        add_labels_to_bars(conv_bars, conv_dist, num_decimal_places=2)

        self.add(v_lines)
        self.play(FadeIn(top_bars, lag_ratio=0.1, run_time=2))
        self.play(FadeIn(low_bars))

        lb_rect = SurroundingRectangle(low_bars)
        lb_rect.round_corners().set_stroke(YELLOW, 2)
        sum_label = Tex(R"\sum_i y_i = 1")
        sum_label.set_color(YELLOW)
        sum_label.next_to(lb_rect)
        self.play(ShowCreation(lb_rect))
        self.play(Write(sum_label, run_time=1))
        self.wait()
        self.play(FadeOut(lb_rect), FadeOut(sum_label))

        # March!
        last_rects = VGroup()
        for n, conv_bar in enumerate(conv_bars):
            rect = conv_bar[0]
            value = conv_bar[1]

            rect.save_state()
            rect.stretch(0, 1, about_edge=DOWN)
            rect.set_opacity(0)

            self.play(
                get_row_shift(top_bars, low_bars, n),
                FadeOut(last_rects),
                run_time=self.march_anim_run_time,
            )

            pairs = get_aligned_pairs(top_bars, low_bars, n)
            label_pairs = VGroup(*(VGroup(m1.value_label, m2.value_label) for m1, m2 in pairs))
            rects = VGroup(*(
                SurroundingRectangle(lp, buff=0.05).set_stroke(YELLOW, 2).round_corners()
                for lp in label_pairs
            ))
            rects.set_stroke(YELLOW, 2)

            self.play(
                FadeIn(rects, lag_ratio=0.5),
                conv_bars[:n].animate.set_opacity(0.5),
                run_time=self.march_anim_run_time,
            )

            self.play(
                *(
                    FadeTransform(label.copy(), value)
                    for lp in label_pairs
                    for label in lp
                ),
                Restore(rect),
                run_time=self.march_anim_run_time,
            )

            if self.always_preview_result:
                self.add(conv_bars)
                conv_bars.set_opacity(0.5)
                conv_bar.set_opacity(1)

            self.wait(0.5)

            last_rects = rects

        self.play(
            FadeOut(last_rects),
            conv_bars.animate.set_opacity(1),
        )


class MovingAverageFast(MovingAverageExample):
    march_anim_run_time = 0


class AltMovingAverage(MovingAverageExample):
    dist2 = [0.1, 0.2, 0.4, 0.2, 0.1]


class AltMovingAverageFast(AltMovingAverage):
    march_anim_run_time = 0


class MovingAverageFast2(AltMovingAverageFast):
    always_preview_result = True


class CompareSizes(InteractiveScene):
    def construct(self):
        # Show them all!
        int_arr1 = [3, 1, 4, 1, 5, 9]
        int_arr2 = [5, 7, 7]
        conv_arr = np.convolve(int_arr1, int_arr2)

        arrays = VGroup()
        for arr in (int_arr1, int_arr2, conv_arr):
            squares = Square().get_grid(1, len(arr), buff=0)
            squares.set_height(0.7)
            squares.set_stroke(WHITE, 1)
            squares.set_fill(GREY_E, 1)
            for square, elem in zip(squares, arr):
                int_mob = Integer(elem).move_to(square)
                square.add(int_mob)
            arrays.add(squares)

        top_arr, low_arr, conv_arr = arrays

        arrays.arrange(DOWN, buff=1.0)
        arrays[:2].shift(UP)
        VGroup(*(square[0] for square in arrays[2])).set_opacity(0)

        self.add(*arrays)

        # Length labels
        braces = VGroup(*(Brace(arr, vect, buff=SMALL_BUFF) for arr, vect in zip(arrays, [UP, DOWN, DOWN])))
        brace_labels = VGroup(*(brace.get_tex(tex, buff=SMALL_BUFF) for brace, tex in zip(braces, ["n", "m", "n + m - 1"])))
        braces[1].add_updater(lambda m: m.match_x(arrays[1]))
        brace_labels[1].add_updater(lambda m: m.match_x(arrays[1]))

        self.add(braces, brace_labels)

        # Flip
        self.remove(low_arr)
        fake_arr = low_arr.deepcopy()
        fake_arr.generate_target(use_deepcopy=True)
        fake_arr.target.rotate(PI)
        for square in fake_arr.target:
            square[0].rotate(PI)
        self.play(MoveToTarget(fake_arr, path_arc=PI, lag_ratio=0.01))
        self.remove(fake_arr)
        low_arr.rotate(PI)
        for square in low_arr:
            square[0].rotate(PI)

        # March!
        for arr in (top_arr, low_arr):
            for index, square in enumerate(arr):
                square.index = index

        for n in range(len(conv_arr)):
            self.play(
                get_row_shift(top_arr, low_arr, n),
                run_time=0.5
            )
            pairs = get_aligned_pairs(top_arr, low_arr, n)
            rects = VGroup(*(
                SurroundingRectangle(pair, buff=-0.05)
                for pair in pairs
            ))
            self.add(rects)
            conv_arr[n].set_opacity(1)
            self.play(ShowIncreasingSubsets(rects, int_func=np.ceil), run_time=0.5)
            self.wait(0.25)
            self.remove(rects)

        # Truncate
        brace = braces[2]
        brace_label = brace_labels[2]

        small_brace = Brace(conv_arr[2:-2], DOWN, buff=SMALL_BUFF)
        small_brace_label = small_brace.get_text("Only consider full overlaps")

        mid_brace = Brace(conv_arr[1:-1], DOWN, buff=SMALL_BUFF)
        mid_brace_label = mid_brace.get_text("Match biggest input size")

        self.play(
            Transform(brace, small_brace),
            FadeTransform(brace_label, small_brace_label),
            conv_arr[:2].animate.set_opacity(0.25),
            conv_arr[-2:].animate.set_opacity(0.25),
        )
        self.wait()
        self.play(
            Transform(brace, mid_brace),
            FadeTransform(small_brace_label, mid_brace_label),
            conv_arr[1].animate.set_opacity(1),
            conv_arr[-2].animate.set_opacity(1),
        )
        self.wait()


# Image processing


class ImageConvolution(InteractiveScene):
    image_name = "MarioSmall"
    image_height = 6.0
    kernel_tex = None
    scalar_conv = False
    pixel_stroke_width = 1.0
    pixel_stroke_opacity = 1.0
    kernel_decimal_places = 2
    kernel_color = BLUE
    grayscale = False

    def setup(self):
        super().setup()
        # Set up the pixel grids
        pixels = self.get_pixel_value_array() / 255.0
        kernel = self.get_kernel()
        if self.scalar_conv:
            conv = scipy.signal.convolve(pixels.mean(2), kernel, mode='same')
        else:
            conv = scipy.signal.convolve(pixels, np.expand_dims(kernel, 2), mode='same')

        conv = np.clip(conv, -1, 1)

        pixel_array = self.get_pixel_array(pixels)
        kernel_array = self.get_kernel_array(kernel, pixel_array, tex=self.kernel_tex)
        conv_array = self.get_pixel_array(conv)
        conv_array.set_fill(opacity=0)

        VGroup(pixel_array, conv_array).arrange(RIGHT, buff=2.0)
        kernel_array.move_to(pixel_array[0])

        self.add(pixel_array)
        self.add(conv_array)
        self.add(kernel_array)

        # Set up index tracker
        index_tracker = ValueTracker(0)

        def get_index():
            return int(clip(index_tracker.get_value(), 0, len(pixel_array) - 1))

        kernel_array.add_updater(lambda m: m.move_to(pixel_array[get_index()]))
        conv_array.add_updater(lambda m: m.set_fill(opacity=0))
        conv_array.add_updater(lambda m: m[:get_index() + 1].set_fill(opacity=1))

        right_rect = conv_array[0].copy()
        right_rect.set_fill(opacity=0)
        right_rect.set_stroke(self.kernel_color, 4, opacity=1)
        right_rect.add_updater(lambda m: m.move_to(conv_array[get_index()]))
        self.add(right_rect)

        self.index_tracker = index_tracker
        self.pixel_array = pixel_array
        self.kernel_array = kernel_array
        self.conv_array = conv_array
        self.right_rect = right_rect

    def get_pixel_value_array(self):
        im_path = get_full_raster_image_path(self.image_name)
        image = Image.open(im_path)
        return np.array(image)[:, :, :3]

    def get_pixel_array(self, array: np.ndarray):
        height, width = array.shape[:2]

        pixel_array = Square().get_grid(height, width, buff=0)
        for pixel, value in zip(pixel_array, it.chain(*array)):
            if value.size == 3:
                # Value is rgb valued
                rgb = np.abs(value).clip(0, 1)
                if self.grayscale:
                    rgb[:] = rgb.mean()
            else:
                # Treat as scalar, color red for negative green for positive
                rgb = [max(-value, 0), max(value, 0), max(value, 0)]
            pixel.set_fill(rgb_to_color(rgb), 1.0)
        pixel_array.set_height(self.image_height)
        pixel_array.set_max_width(5.75)
        pixel_array.set_stroke(WHITE, self.pixel_stroke_width, self.pixel_stroke_opacity)

        return pixel_array

    def get_kernel_array(self, kernel: np.ndarray, pixel_array: VGroup, tex=None):
        kernel_array = VGroup()
        values = VGroup()
        for row in kernel:
            for x in row:
                square = pixel_array[0].copy()
                square.set_fill(BLACK, 0)
                square.set_stroke(self.kernel_color, 2, opacity=1)
                if tex:
                    value = OldTex(tex)
                else:
                    value = DecimalNumber(x, num_decimal_places=self.kernel_decimal_places)
                value.set_width(square.get_width() * 0.7)
                value.set_backstroke(BLACK, 3)
                value.move_to(square)
                values.add(value)
                square.add(value)
                kernel_array.add(square)
        for value in values:
            value.set_height(values[0].get_height())
        kernel_array.reverse_submobjects()
        kernel_array.arrange_in_grid(*kernel.shape, buff=0)
        kernel_array.move_to(pixel_array[0])
        return kernel_array

    def get_kernel(self):
        return np.ones((3, 3)) / 9

    # Setup combing and zooming
    def set_index(self, value, run_time=8, rate_func=linear):
        self.play(
            self.index_tracker.animate.set_value(value),
            run_time=run_time,
            rate_func=rate_func
        )

    def zoom_to_kernel(self, run_time=2):
        ka = self.kernel_array
        self.play(
            self.camera.frame.animate.set_height(1.5 * ka.get_height()).move_to(ka),
            run_time=run_time
        )

    def zoom_to_new_pixel(self, run_time=4):
        ka = self.kernel_array
        ca = self.conv_array
        frame = self.camera.frame
        curr_center = frame.get_center().copy()
        index = int(self.index_tracker.get_value())
        new_center = ca[index].get_center()
        center_func = bezier([curr_center, curr_center, new_center, new_center])

        target_height = 1.5 * ka.get_height()
        height_func = bezier([
            frame.get_height(), frame.get_height(), FRAME_HEIGHT,
            target_height, target_height,
        ])
        self.play(
            UpdateFromAlphaFunc(frame, lambda m, a: m.set_height(height_func(a)).move_to(center_func(a))),
            run_time=run_time,
            rate_func=linear,
        )

    def reset_frame(self, run_time=2):
        self.play(
            self.camera.frame.animate.to_default_state(),
            run_time=run_time
        )

    def show_pixel_sum(self, tex=None, convert_to_vect=True, row_len=9):
        # Setup sum
        ka = self.kernel_array
        pa = self.pixel_array
        frame = self.camera.frame

        rgb_vects = VGroup()
        lil_pixels = VGroup()
        expr = VGroup()

        ka_copy = VGroup()
        stroke_width = 2 * FRAME_HEIGHT / frame.get_height()

        lil_height = 1.0
        for square in ka:
            ka_copy.add(square.copy().set_stroke(TEAL, stroke_width))
            sc = square.get_center()
            pixel = pa[np.argmin([get_norm(p.get_center() - sc) for p in pa])]
            color = pixel.get_fill_color()
            rgb = color_to_rgb(color)
            rgb_vect = DecimalMatrix(rgb.reshape((3, 1)), num_decimal_places=2)
            rgb_vect.set_height(lil_height)
            rgb_vect.set_color(color)
            if get_norm(rgb) < 0.1:
                rgb_vect.set_color(WHITE)
            rgb_vects.add(rgb_vect)

            lil_pixel = pixel.copy()
            lil_pixel.match_width(rgb_vect)
            lil_pixel.set_stroke(WHITE, stroke_width)
            lil_pixels.add(lil_pixel)

            if tex:
                lil_coef = OldTex(tex, font_size=36)
            else:
                lil_coef = square[0].copy()
                lil_coef.set_height(lil_height * 0.5)
            expr.add(lil_coef, lil_pixel, OldTex("+", font_size=48))

        expr[-1].scale(0, about_edge=LEFT)  # Stray plus
        rows = VGroup(*(
            expr[n:n + 3 * row_len]
            for n in range(0, len(expr), 3 * row_len)
        ))
        for row in rows:
            row.arrange(RIGHT, buff=0.2)
        rows.arrange(DOWN, buff=0.4, aligned_edge=LEFT)

        expr.set_max_width(FRAME_WIDTH - 1)
        expr.to_edge(UP)
        expr.fix_in_frame()

        for vect, pixel in zip(rgb_vects, lil_pixels):
            vect.move_to(pixel)
            vect.set_max_width(pixel.get_width())
        rgb_vects.fix_in_frame()

        # Reveal top
        top_bar = FullScreenRectangle().set_fill(BLACK, 1)
        top_bar.set_height(rgb_vects.get_height() + 0.5, stretch=True, about_edge=UP)
        top_bar.fix_in_frame()

        self.play(
            frame.animate.scale(1.2, about_edge=DOWN),
            FadeIn(top_bar, 2 * DOWN),
        )

        # Show sum
        for n in range(len(ka_copy)):
            self.remove(*ka_copy)
            self.add(ka_copy[n])
            self.add(expr[:3 * n + 2])
            self.wait(0.25)
        self.remove(*ka_copy)
        if convert_to_vect:
            self.play(LaggedStart(*(
                Transform(lil_pixel, rgb_vect)
                for lil_pixel, rgb_vect in zip(lil_pixels, rgb_vects)
            )))
        self.wait()
        result = VGroup(top_bar, expr)
        return result


class BoxBlurMario(ImageConvolution):
    kernel_tex = "1 / 9"
    image_name = "MarioSmall"
    pixel_stroke_opacity = 0.5
    stops = (131, 360)
    final_run_time = 8

    def construct(self):
        # March
        for index in self.stops:
            self.set_index(index)
            self.zoom_to_kernel()
            if index == self.stops[0]:
                top_bar = self.show_pixel_sum(tex=R"\frac{1}{9}")
            self.wait()
            self.zoom_to_new_pixel(run_time=8)
            self.wait()
            if index == self.stops[0]:
                self.play(FadeOut(top_bar))
            self.reset_frame()
        self.set_index(len(self.pixel_array) - 1, run_time=self.final_run_time)
        self.wait()


class BoxBlurCat(BoxBlurMario):
    image_name = "PixelArtCat"
    stops = ()
    final_run_time = 20


class GaussianBluMario(ImageConvolution):
    kernel_decimal_places = 3
    focus_index = 256
    final_run_time = 10

    def construct(self):
        # March!
        self.set_index(self.focus_index)
        self.wait()
        self.zoom_to_kernel()
        self.wait()

        # Gauss surface
        kernel_array = self.kernel_array
        frame = self.camera.frame

        gaussian = ParametricSurface(
            lambda u, v: [u, v, np.exp(-(u**2) - v**2)],
            u_range=(-3, 3),
            v_range=(-3, 3),
            resolution=(101, 101),
        )
        gaussian.set_color(BLUE, 0.8)
        gaussian.match_width(kernel_array)
        gaussian.stretch(2, 2)
        gaussian.add_updater(lambda m: m.move_to(kernel_array, IN))

        self.play(
            FadeIn(gaussian),
            frame.animate.reorient(10, 70),
            run_time=3
        )
        self.wait()
        top_bar = self.show_pixel_sum(convert_to_vect=False)
        self.wait()
        self.zoom_to_new_pixel()
        self.wait()
        self.play(
            frame.animate.set_height(8).reorient(0, 60).move_to(ORIGIN),
            FadeOut(top_bar, time_span=(0, 1)),
            run_time=3,
        )

        # More walking
        self.set_index(len(self.pixel_array), run_time=self.final_run_time)
        self.wait()

    def get_kernel(self):
        # Oh good, hard coded, I hope you feel happy with yourself.
        return np.array([
            [0.00296902, 0.0133062, 0.0219382, 0.0133062, .00296902],
            [0.0133062, 0.0596343, 0.0983203, 0.0596343, 0.0133062],
            [0.0219382, 0.0983203, 0.162103, 0.0983203, 0.0219382],
            [0.0133062, 0.0596343, 0.0983203, 0.0596343, 0.0133062],
            [0.00296902, 0.0133062, 0.0219382, 0.0133062, 0.00296902],
        ])


class GaussianBlurCat(GaussianBluMario):
    image_name = "PixelArtCat"
    focus_index = 254

    def construct(self):
        for arr in self.pixel_array, self.conv_array:
            arr.set_stroke(width=0.5, opacity=0.5)
        super().construct()


class GaussianBlurCatNoPause(GaussianBlurCat):
    stops = ()
    focus_index = 0
    final_run_time = 30


class SobelFilter1(ImageConvolution):
    scalar_conv = True
    image_name = "BitRandy"
    pixel_stroke_width = 1
    pixel_stroke_opacity = 0.2
    kernel_color = YELLOW
    stops = (194, 400, 801)
    grayscale = True

    def construct(self):
        self.zoom_to_kernel()
        # Show kernel
        kernel = self.kernel_array
        kernel.generate_target()
        for square in kernel.target:
            v = square[0].get_value()
            square.set_fill(
                rgb_to_color([2 * max(-v, 0), 2 * max(v, 0), 2 * max(v, 0)]),
                opacity=0.5,
                recurse=False
            )
            square.set_stroke(WHITE, 1, recurse=False)
        self.play(MoveToTarget(kernel))
        self.wait()
        self.reset_frame()

        # Example walking
        for index in self.stops:
            self.set_index(index)
            self.zoom_to_kernel()
            self.play(*(
                square.animate.set_fill(opacity=0, recurse=False)
                for square in kernel
            ), rate_func=there_and_back_with_pause, run_time=3)
            self.add(kernel)
            self.wait()
            self.zoom_to_new_pixel()
            self.wait()
            self.reset_frame()
        self.set_index(len(self.pixel_array) - 1, run_time=20)

    def get_kernel(self):
        return np.array([
            [-0.25, 0, 0.25],
            [-0.5, 0, 0.5],
            [-0.25, 0, 0.25],
        ])


class SobelFilter2(SobelFilter1):
    stops = ()

    def get_kernel(self):
        return super().get_kernel().T


class SobelFilterCat(SobelFilter1):
    scalar_conv = True
    image_name = "PixelArtCat"
    pixel_stroke_width = 1
    pixel_stroke_opacity = 0.2
    kernel_color = WHITE
    stops = ()
    grayscale = False


class SobelFilterKirby(SobelFilter1):
    image_name = "KirbySmall"
    grayscale = False


class SharpenFilter(ImageConvolution):
    image_name = "KirbySmall"
    kernel_decimal_places = 1
    grayscale = False

    def construct(self):
        for arr in self.pixel_array, self.conv_array:
            arr.set_stroke(WHITE, 0.25, 0.5)
        for square in self.kernel_array:
            square[0].scale(0.6)
        self.set_index(len(self.pixel_array) - 1, run_time=20)

    def get_kernel(self):
        return np.array([
            [0.0, 0.0, -0.25, 0.0, 0.0],
            [0.0, -0.25, -0.5, -0.25, 0.0],
            [-0.25, -0.5, 5.0, -0.5, -0.25],
            [0.0, -0.25, -0.5, -0.25, 0.0],
            [0.0, 0.0, -0.25, 0.0, 0.0],
        ])


# Convolution theorem


class ContrastConvolutionToMultiplication(InteractiveScene):
    def construct(self):
        # Set up divide
        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        v_line.set_stroke(GREY, 2)

        kw = dict(font_size=60)
        conv_name = Text("Convolution", **kw)
        mult_name = Text("Multiplication", **kw)
        conv_name.set_x(-FRAME_WIDTH / 4).to_edge(UP)
        mult_name.set_x(FRAME_WIDTH / 4).to_edge(UP)

        self.add(v_line)
        self.add(conv_name)
        self.add(mult_name)

        # Set up arrays
        arr1 = np.arange(1, 6)
        arr2 = np.arange(6, 11)
        conv = np.convolve(arr1, arr2)
        prod = arr1 * arr2
        quart = FRAME_WIDTH / 4

        left_arrays, right_arrays = (
            VGroup(*(
                self.get_array_mobject(arr, color)
                for arr, color in [(arr1, BLUE), (arr2, YELLOW)]
            )).arrange(DOWN, buff=1.0).move_to(vect * quart + UP)
            for vect in [LEFT, RIGHT]
        )

        conv_array = self.get_array_mobject(conv, color=TEAL)
        prod_array = self.get_array_mobject(prod, color=TEAL)
        conv_array.next_to(left_arrays, DOWN, buff=1.5)
        prod_array.next_to(right_arrays, DOWN, buff=1.5)

        self.add(left_arrays)
        self.add(right_arrays)

        # Show convolution
        top_arr = left_arrays[0]
        low_arr = left_arrays[1]
        low_arr.generate_target()
        low_arr.target.rotate(PI, about_point=low_arr.elements[0].get_center())
        for elem in low_arr.target[1:-1]:
            elem.rotate(PI)
        low_arr.target[2:-2:2].set_y(low_arr[1].get_y(DOWN))
        self.play(
            FadeIn(VGroup(conv_array[0], conv_array[-1])),
            MoveToTarget(low_arr, path_arc=PI),
        )
        for n, elem in enumerate(conv_array.elements):
            pairs = get_aligned_pairs(top_arr.elements, low_arr.elements, n)

            self.play(
                get_row_shift(top_arr.elements, low_arr.elements, n),
                MaintainPositionRelativeTo(low_arr[::2], low_arr.elements),
                run_time=0.25
            )

            lines = VGroup(*(Line(m1, m2, buff=0.1) for m1, m2 in pairs))
            lines.set_stroke(TEAL, 1)

            self.add(lines, elem)

            tally = 0
            for (m1, m2), line in zip(pairs, lines):
                tally += m1.get_value() * m2.get_value()
                lines.set_stroke(width=1)
                line.set_stroke(width=4)
                elem.set_value(tally)
                self.wait(0.25)
            self.wait(0.25)
            self.remove(lines)
            self.add(conv_array[2 * n + 2])
        self.wait()

        # Show multiplication
        low_arr = right_arrays[0]
        top_arr = right_arrays[1]
        lines = VGroup(*(
            Line(e1, e2, buff=0.1)
            for e1, e2 in zip(top_arr.elements, low_arr.elements)
        ))
        lines.set_stroke(TEAL, 1)

        self.play(
            FadeIn(VGroup(prod_array[0], prod_array[-1])),
            FadeIn(lines),
        )
        for n, elem in enumerate(prod_array.elements):
            lines.set_stroke(width=1)
            lines[n].set_stroke(width=4)
            self.add(elem)
            self.add(prod_array[2 * n + 2])
            self.wait(0.5)
        self.play(FadeOut(lines))

    def get_array_mobject(self, array, color=WHITE, font_size=48):
        kw = dict(font_size=font_size)
        result = VGroup(OldTex("[", **kw))
        commas = VGroup()
        elements = VGroup()
        for index, elem in enumerate(array):
            int_mob = Integer(elem, **kw)
            int_mob.index = index
            elements.add(int_mob)
            result.add(int_mob)
            comma = OldTex(",", **kw)
            commas.add(comma)
            result.add(comma)
        result.remove(commas[-1])
        commas.remove(commas[-1])
        result.add(OldTex("]", **kw))
        result.arrange(RIGHT, buff=0.1)
        commas.set_y(result[1].get_y(DOWN))

        elements.set_color(color)
        result.elements = elements
        result.commas = commas

        return result


class BigPolynomials(InteractiveScene):
    def construct(self):
        # Initialize grid
        N = 8
        height = 6.5
        grid = Square().get_grid(N, N, height=height, buff=0, group_by_rows=True)
        grid.set_stroke(WHITE, 1)
        grid.to_edge(DOWN, buff=MED_SMALL_BUFF)
        grid.shift(2 * LEFT)
        for i, row in enumerate(grid):
            if i == N - 2:
                for j, square in enumerate(row):
                    if j == N - 2:
                        self.replace_square(square, OldTex(R"\ddots"))
                    else:
                        self.replace_square(square, OldTex(R"\vdots"))
            else:
                self.replace_square(row[N - 2], OldTex(R"\cdots"))

        self.add(grid)

        # Polynomial terms
        a_terms, b_terms = all_terms = [
            VGroup(*(
                Tex(RF"{letter}_{{{n}}} x^{{{n}}}", font_size=24)
                for n in (*range(N - 1), 99)
            ))
            for letter in ["a", "b"]
        ]
        for terms in all_terms:
            terms[0].remove(*terms[0][-2:])
            terms[1].remove(*terms[1][-1:])

        for terms, group, vect in [(a_terms, grid[0], UP), (b_terms, grid, LEFT)]:
            for term, square in zip(terms, group):
                term.next_to(square, vect, SMALL_BUFF)

        a_terms[-2].become(OldTex(R"\cdots", font_size=24).move_to(a_terms[-2]).shift(0.02 * DOWN))
        b_terms[-2].become(OldTex(R"\vdots", font_size=24).move_to(b_terms[-2]))

        a_terms.set_color(BLUE_C)
        b_terms.set_color(TEAL_C)

        self.add(a_terms)
        self.add(b_terms)

        # Plusses
        for terms, vect in (a_terms, RIGHT), (b_terms, DOWN):
            terms.plusses = VGroup()
            for t1, t2 in zip(terms, terms[1:]):
                plus = OldTex("+", font_size=24).match_color(terms)
                plus.move_to(midpoint(t1.get_corner(vect), t2.get_corner(-vect)))
                terms.plusses.add(plus)
            self.add(terms.plusses)

        # Product terms
        prod_terms = VGroup()
        diags = VGroup(*(VGroup() for n in range(11)))

        for i, row in enumerate(grid):
            pre_b = b_terms[i][:2]
            if i == N - 2:
                continue
            if i == N - 1:
                i = 99
            for j, square in enumerate(row):
                pre_a = a_terms[j][:2]
                if j == N - 2:
                    continue
                if j == N - 1:
                    j = 99
                term = OldTex(f"a_{{{j}}}", f"b_{{{i}}}", f"x^{{{i + j}}}", font_size=20)
                if i + j == 0:
                    term[2].remove(*term[2][:-2])
                elif i + j == 1:
                    term[2].remove(term[2][:-2])
                term[0].match_color(a_terms)
                term[1].match_color(b_terms)
                term.set_max_width(0.9 * square.get_width())
                term.pre_a = pre_a
                term.pre_b = pre_b
                term.move_to(square)
                prod_terms.add(term)

                if i + j < len(diags):
                    diags[i + j].add(term)

        # Animate
        a_label = Text("100 terms", font_size=30)
        a_label.next_to(a_terms, UP)
        b_label = Text("100 terms", font_size=30)
        b_label.next_to(b_terms, UP, MED_LARGE_BUFF, aligned_edge=RIGHT)
        product_count = OldTexText(R"$100 \times 100 = 10{,}000$ \\ products", font_size=60)
        product_count.move_to(midpoint(grid.get_right(), RIGHT_SIDE))

        self.play(
            FlashAround(a_terms, run_time=2, time_width=2),
            FadeIn(a_label)
        )
        self.play(
            FlashAround(b_terms, run_time=2, time_width=2),
            FadeIn(b_label),
            FadeOut(a_label)
        )
        self.play(FadeOut(b_label))
        self.wait()
        self.play(
            LaggedStart(*(
                AnimationGroup(
                    TransformFromCopy(term.pre_a, term[0]),
                    TransformFromCopy(term.pre_b, term[1]),
                    FadeIn(term[2], rate_func=squish_rate_func(smooth, 0.5, 1)),
                )
                for term in prod_terms
            ), lag_ratio=0.1, run_time=5),
            Write(product_count),
        )
        self.add(prod_terms)

        # Group along diagonals
        self.play(prod_terms.animate.set_opacity(0.2))
        for n in range(len(diags)):
            diags.generate_target()
            diags.target.set_opacity(0.2)
            diags.target[n].set_opacity(1.0)
            self.play(MoveToTarget(diags))
        self.wait()

    def replace_square(self, square, mob):
        mob.move_to(square)
        mob.set_max_width(square.get_width() / 2)
        mob.set_max_height(square.get_width() / 2)
        square.set_opacity(0)
        square.add(mob)


class FunctionToCoefficientCommutativeDiagram(InteractiveScene):
    def construct(self):
        # Axes
        axes1, axes2, axes3 = all_axes = VGroup(*(
            Axes((-3, 3), (-2, 4), width=6, height=4)
            for x in range(3)
        ))

        all_axes.arrange(RIGHT, buff=1.0)
        axes3.shift(4 * RIGHT)
        all_axes.set_width(FRAME_WIDTH - 1)
        all_axes.move_to(UP)

        # Graphs
        def p1(x):
            return 0.2 * (x + 3) * x * (x - 2)

        def p2(x):
            return -0.1 * (x + 2) * (x - 2) * (x - 3)

        graphs = VGroup(
            axes1.get_graph(p1).set_stroke(BLUE),
            axes2.get_graph(p2).set_stroke(TEAL),
            axes3.get_graph(lambda x: p1(x) * p2(x)).set_stroke(YELLOW),
        )
        graphs.set_stroke(width=2)

        kw = dict(font_size=30)
        graph_labels = VGroup(*(
            Tex(tex, **kw).next_to(axes.get_top(), RIGHT, aligned_edge=UP)
            for tex, axes in zip(["f(x)", "g(x)", R"f(x) \cdot g(x)"], all_axes)
        ))

        # Coefficients
        a_labels, b_labels, conv_label = coef_labels = VGroup(
            Tex(R"(a_0, a_1, \dots, a_n)", **kw),
            Tex(R"(b_0, b_1, \dots, b_m)", **kw),
            VGroup(
                Tex("c_0 = a_0 b_0", **kw),
                Tex("c_1 = a_0 b_1 + a_1 b_0", **kw),
                Tex("c_2 = a_0 b_2 + a_1 b_1 + a_2 b_0", **kw),
                Tex("c_3 = a_0 b_3 + a_1 b_2 + a_2 b_1 + a_3 b_0", **kw),
                Tex(R"\vdots", **kw),
            ).arrange(DOWN, aligned_edge=LEFT).scale(0.85)
        )
        v_arrows = VGroup()
        for labels, graph, axes in zip(coef_labels, graphs, all_axes):
            arrow = Vector(0.8 * DOWN)
            arrow.next_to(axes, DOWN)
            v_arrows.add(arrow)
            labels.next_to(arrow, DOWN)
            labels.match_color(graph)

            arrow_label = Text("Coefficients", font_size=24)
            arrow_label.next_to(arrow, RIGHT, buff=0.2)
            arrow.add(arrow_label)

        conv_label.to_edge(RIGHT, buff=SMALL_BUFF)

        # Operations
        mult_arrow = Arrow(axes2, axes3, buff=0.1, stroke_width=6)
        mult_arrow_label = Text("Multiplication", font_size=30)
        mult_arrow_label.next_to(mult_arrow, UP, SMALL_BUFF)
        mult_arrow = VGroup(mult_arrow, mult_arrow_label)
        mult_arrow.set_color(RED)

        coef_rect = SurroundingRectangle(coef_labels[:2])
        coef_rect.set_stroke(GREY, 2)
        conv_arrow = Arrow(coef_rect, conv_label[0], buff=0.3, stroke_width=6)
        conv_arrow_label = Text("Convolution", font_size=30)
        conv_arrow_label.next_to(conv_arrow, DOWN, SMALL_BUFF)
        conv_arrow = VGroup(conv_arrow, conv_arrow_label)
        conv_arrow.set_color(RED)

        # Animations
        self.play(
            LaggedStartMap(ShowCreation, graphs[:2]),
            LaggedStartMap(FadeIn, graph_labels[:2]),
            LaggedStartMap(FadeIn, all_axes[:2]),
        )
        self.play(LaggedStart(
            Write(mult_arrow, run_time=1),
            TransformFromCopy(graphs[0], graphs[2].copy(), remover=True),
            TransformFromCopy(graphs[1], graphs[2]),
            TransformFromCopy(all_axes[0], all_axes[2].copy(), remover=True),
            TransformFromCopy(all_axes[1], all_axes[2]),
            TransformFromCopy(graph_labels[0], graph_labels[2][:5]),
            TransformFromCopy(graph_labels[1], graph_labels[2][5:]),
        ), lag_ratio=0.2)
        self.wait()
        self.play(
            LaggedStartMap(Write, v_arrows[:2], lag_ratio=0.7),
            LaggedStartMap(FadeIn, coef_labels[:2], shift=DOWN, lag_ratio=0.7),
        )
        self.wait()
        self.play(ShowCreation(coef_rect), FadeIn(conv_arrow))
        self.play(
            FadeTransformPieces(a_labels.copy(), conv_label),
            FadeTransformPieces(b_labels.copy(), conv_label),
        )

        # Pointwise product
        all_dots = Group(*(
            Group(*(
                GlowDot(axes.i2gp(x, graph), radius=0.1, glow_factor=0.8, color=WHITE)
                for x in range(-3, 4)
            ))
            for axes, graph in zip(all_axes, graphs)
        ))
        all_circles = VGroup(*(
            VGroup(*(
                Circle(radius=0.1).set_stroke(YELLOW, 2).move_to(dot)
                for dot in dots
            ))
            for dots in all_dots
        ))
        self.play(
            LaggedStartMap(FadeIn, all_dots[0], scale=0.5, lag_ratio=0.5),
            LaggedStartMap(FadeIn, all_dots[1], scale=0.5, lag_ratio=0.5),
        )
        self.wait()
        self.play(
            ShowSubmobjectsOneByOne(all_circles[0]),
            ShowSubmobjectsOneByOne(all_circles[1]),
            ShowSubmobjectsOneByOne(all_circles[2]),
            ShowIncreasingSubsets(all_dots[2]),
            run_time=4,
            rate_func=linear,
        )
        self.remove(all_circles)
        self.wait()
        self.play(Write(v_arrows[2]))
        self.wait()


class DataPointsToPolynomial(InteractiveScene):
    def construct(self):
        # Axes
        axes = Axes((-1, 10), (-3, 3), width=FRAME_WIDTH - 2, height=4)
        kw = dict(font_size=30)
        axes.add(OldTex("x", **kw).next_to(axes.x_axis.get_end(), DR, buff=0.2))
        axes.add(OldTex("y", **kw).next_to(axes.y_axis.get_end(), LEFT, MED_SMALL_BUFF))

        self.add(axes)

        # Graphs and data points
        y_values = [3, 1, 2, -3, -1, 2, 0, 0, 1]
        data = list(enumerate(y_values))
        dots = Group(*(
            GlowDot(axes.c2p(x, y), glow_factor=0.8, radius=0.1)
            for x, y in data
        ))
        dots.set_color(TEAL)
        circles = VGroup(*(
            Circle(radius=0.075).set_stroke(YELLOW, 2).move_to(dot)
            for dot in dots
        ))

        graphs = VGroup(*(
            axes.get_graph(get_lagrange_polynomial(data[:n]))
            for n in range(1, len(data) + 1)
        ))
        graphs.set_stroke(BLUE, 2)

        # Increasing polynomials
        poly_tex = OldTex(
            "a_0", "+", "a_1 x", "+", *it.chain(*(
                (f"a_{n} x^{n}", "+")
                for n in range(2, len(data) - 1)
            ))
        )
        poly_tex.to_corner(UR)

        graph = graphs[1].copy()
        self.play(
            LaggedStartMap(FadeIn, dots[:2], lag_ratio=0.7, run_time=1),
            LaggedStartMap(ShowCreationThenFadeOut, circles[:2], lag_ratio=0.2, run_time=1),
        )
        self.play(
            ShowCreation(graph),
            FadeIn(poly_tex[:3])
        )
        self.wait()
        for n in range(2, len(data) - 1):
            self.play(
                ShowCreationThenFadeOut(circles[n], run_time=1.5),
                FadeIn(dots[n]),
                Transform(graph, graphs[n]),
                FadeIn(poly_tex[2 * n - 1:2 * n + 1], time_span=(0, 1), lag_ratio=0.1),
                run_time=2
            )
            self.wait()


class PolynomialSystem(InteractiveScene):
    N = 8

    def construct(self):
        # Setup polynomials
        frame = self.camera.frame
        N = self.N
        coefs = VGroup(*(Tex(Rf"c_{{{n}}}") for n in range(N)))
        coefs.set_submobject_colors_by_gradient(BLUE, TEAL)
        x_powers = VGroup(*(Tex(Rf"x^{{{n}}}") for n in range(N)))
        poly_x = self.get_polynomial(coefs, x_powers)

        top_lhs = Tex("h(x) = ")
        top_lhs.next_to(poly_x[0][0], LEFT, buff=0.1)
        top_eq = VGroup(top_lhs, poly_x)
        top_eq.center()

        fg = Tex(R"f(x) \cdot g(x)")
        eq = Tex("=")
        eq.rotate(PI / 2)
        eq.next_to(top_lhs[:4], UP)
        fg.next_to(eq, UP).shift_onto_screen()

        self.add(top_eq)

        # Suppose you don't know coefficients
        words = Text("Suppose these are a mystery")
        words.next_to(poly_x, UP, buff=2.0)
        arrows = VGroup(*(
            Arrow(
                interpolate(
                    words.get_corner(DL) + 0.5 * RIGHT,
                    words.get_corner(DR) + 0.5 * LEFT,
                    n / (N - 1)),
                coef,
                color=coef.get_color(),
            )
            for n, coef in enumerate(poly_x.coefs)
        ))

        self.play(
            FadeIn(words, lag_ratio=0.1),
            LaggedStartMap(GrowArrow, arrows),
            LaggedStart(*(
                FlashAround(coef, time_width=1)
                for coef in poly_x.coefs
            ), lag_ratio=0.1, run_time=3)
        )
        self.wait()
        self.play(
            Write(eq),
            FadeIn(fg, 0.25 * UP)
        )
        self.wait()

        # Sweep away
        self.play(
            LaggedStart(
                FadeOut(words, UP),
                FadeOut(arrows, 2 * UP, lag_ratio=0.05),
                FadeOut(fg, UP),
                FadeOut(eq, 1.5 * UP),
                top_eq.animate.to_edge(UP)
            ),
            frame.animate.set_height(11, about_edge=UR),
            run_time=2
        )

        # Set up the large system
        lhss = VGroup(*(Tex(f"h({n})=") for n in range(N)))
        rhss = VGroup(*(
            self.get_polynomial(
                coefs.copy(),
                # VGroup(*(Integer(x**n) for n in range(N)))
                VGroup(*(Tex(f"{x}^{{{n}}}") for n in range(N)))
            )
            for x in range(N)
        ))
        equations = VGroup()
        for lhs, rhs in zip(lhss, rhss):
            lhs.next_to(rhs[0][0], LEFT, 0.1)
            equations.add(VGroup(lhs, rhs))
        equations.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        equations.next_to(top_eq, DOWN, aligned_edge=LEFT, buff=1.5)

        self.play(
            LaggedStart(*(
                FadeTransform(top_eq.copy(), eq)
                for eq in equations
            ), lag_ratio=0.02, run_time=3),
        )

        # Suppose you _do_ know h(0), h(1), h(2), etc.
        words2 = Text("But suppose\nyou do know\nthese", t2s={"do": ITALIC})
        words2.set_color(YELLOW)
        words2.move_to(midpoint(equations.get_left(), frame.get_left()))
        words2.align_to(equations, UP)

        self.play(
            Write(words2),
            LaggedStart(*(FlashAround(lhs[:4], time_width=1.5) for lhs in lhss), run_time=5, lag_ratio=0.1)
        )
        self.wait()

        # Trow out the inputs
        inputs = VGroup(*(lhs[2] for lhs in lhss))
        inputs.save_state()
        consts = VGroup(*(
            power
            for eq in rhss
            for power in eq.powers
        ))
        boxes = VGroup(*(VGroup(SurroundingRectangle(const, buff=0)) for const in consts))
        boxes.set_stroke(RED, 1)

        self.play(
            FadeOut(words2, LEFT, rate_func=running_start),
            inputs.animate.set_color(RED).shift(2 * LEFT).set_anim_args(run_time=1.5, lag_ratio=0.2, path_arc=PI / 2),
            Transform(consts, boxes, lag_ratio=0.01, run_time=3),
        )
        self.play(FadeOut(inputs, LEFT, lag_ratio=0.2, rate_func=running_start))
        inputs.restore()
        inputs.set_opacity(0)
        self.add(lhss)

        # Add roots of unity
        kw = dict(tex_to_color_map={R"\omega": YELLOW})
        omega_def = OldTex(
            Rf"""
            \text{{Let }} \omega = e^{{2\pi i / {N}}} \qquad
            \text{{Notice }} \omega^{{{N}}} = 1
            """,
            font_size=60,
            **kw
        )
        omega_def.next_to(lhss, UP, buff=1.5, aligned_edge=LEFT)

        all_omega_powers = [
            VGroup(*(
                Tex(Rf"\omega^{{{(k * n) % N}}}", **kw)
                for n in range(N)
            ))
            for k in range(0, N)
        ]
        new_rhss = VGroup(*(
            self.get_polynomial(coefs.copy(), omega_powers)
            for omega_powers in all_omega_powers
        ))

        new_lhss = VGroup(
            *(Tex(Rf"h(\omega^{n}) = ") for n in range(N))
        )

        self.play(
            frame.animate.set_height(13, about_edge=DR),
            FadeIn(omega_def),
            top_eq.animate.shift(1.75 * UP),
        )
        for old_lhs, old_rhs, new_lhs, new_rhs in zip(lhss, rhss, new_lhss, new_rhss):
            new_lhs.move_to(old_lhs, LEFT)
            new_rhs.next_to(new_lhs, RIGHT, buff=0.1)
            new_lhs[2].set_color(YELLOW)
            self.play(
                FadeTransformPieces(old_lhs, new_lhs),
                FadeTransformPieces(old_rhs, new_rhs),
            )
        self.wait()

        # Label as DFT
        brace = Brace(new_lhss, LEFT)
        kw = dict(font_size=60)
        label = OldTexText(R"Discrete\\Fourier\\Transform", alignment=R"\raggedright", isolate=list("DFT"), **kw)
        label.next_to(brace, LEFT)

        sub_label = OldTexText("of $(c_i)$", **kw)[0]
        sub_label[2:].set_color(BLUE)
        sub_label.next_to(label, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        self.play(GrowFromCenter(brace), FadeIn(label))
        self.play(Write(sub_label))
        self.wait()

        # Show redundancy
        last_rects = VGroup()
        for n in range(N):
            tex = Rf"\omega^{{{n}}}"
            rects = VGroup()
            for rhs in new_rhss:
                for power in rhs.powers[1:]:
                    if power.get_tex() == tex:
                        rects.add(SurroundingRectangle(power))
            rects.set_stroke(RED, 3)
            self.play(FadeIn(rects), FadeOut(last_rects))
            last_rects = rects
        self.play(FadeOut(last_rects))

        # Set up two arrays
        pre_h_list = VGroup(*(lhs[:5] for lhs in new_lhss))
        h_list = pre_h_list.copy()
        h_list.next_to(new_lhss, LEFT, buff=2.0)
        h_rect = SurroundingRectangle(h_list, buff=0.25)
        h_rect.set_stroke(WHITE, 1)
        h_rect.set_fill(GREY_E, 1)

        c_list = poly_x.coefs.copy()
        for c, h in zip(c_list, h_list):
            c.scale(1.25)
            c.move_to(h)

        c_rect = h_rect.copy()
        c_rect.shift(5 * LEFT)
        c_list.move_to(c_rect)

        short_label = OldTexText("DFT", isolate=list("DFT"))
        short_label.next_to(h_rect, UP, buff=0.5).shift(sub_label.get_width() * LEFT / 2)

        self.play(
            FadeIn(c_rect),
            TransformFromCopy(new_rhss[0].coefs, c_list, path_arc=-PI / 3)
        )
        self.play(
            TransformMatchingTex(label, short_label),
            sub_label.animate.scale(48 / 60).next_to(short_label, RIGHT),
            FadeInFromPoint(h_rect, pre_h_list.get_center()),
            TransformFromCopy(pre_h_list, h_list),
        )

        # Indicate fast back and forth
        top_arrow = Arrow(c_rect, h_rect).shift(2 * UP)
        low_arrow = Arrow(h_rect, c_rect).shift(2 * DOWN)

        for arrow in (top_arrow, low_arrow):
            fft_label = Text("FFT")
            fft_label.next_to(arrow, UP)
            run_time = OldTex(R"\mathcal{O}\big(N\log(N)\big)")
            run_time.next_to(arrow, DOWN)
            arrow.fft_label = fft_label
            arrow.run_time = run_time

            self.play(
                GrowArrow(arrow),
                FadeIn(fft_label, arrow.get_vector())
            )
            self.play(FadeIn(run_time, 0.5 * DOWN))
            self.wait()

    def get_polynomial(self, coefs, powers, buff=0.1):
        result = VGroup()
        result.plusses = VGroup()
        result.dots = VGroup()
        for coef, power in zip(coefs, powers):
            if power is powers[0]:
                power.scale(0, about_edge=LEFT)
            plus = OldTex("+")  # Font size?
            result.add(coef)
            if isinstance(power, Integer):
                dot = OldTex(R"\cdot")
            else:
                dot = VGroup(VectorizedPoint())
            result.dots.add(dot)
            result.add(dot)
            result.add(power, plus)
            result.plusses.add(plus)
        result.remove(result[-1])  # Stray plus
        result.arrange(RIGHT, buff=buff)
        for mob in result:
            mob.shift(mob[0].get_y(DOWN) * DOWN)
        for coef, dot in zip(coefs, result.dots):
            if not isinstance(dot, Tex):
                coef.shift(buff * RIGHT)
        result.dots.set_y(result.get_y())
        result.coefs = coefs
        result.powers = powers
        return result


class RootsOfUnity(InteractiveScene):
    def construct(self):
        # Add plane
        plane = ComplexPlane((-2, 2), (-2, 2))
        plane.scale(2.0 / 1.5)
        plane.to_corner(UL)
        plane.add_coordinate_labels([1, 1j], font_size=24)

        circle = Circle(radius=plane.x_axis.get_unit_size())
        circle.move_to(plane.n2p(0))
        circle.set_stroke(YELLOW, 2)

        N = 8
        roots = [np.exp(TAU * 1j * k / N) for k in range(N)]
        root_dots = Group(*(GlowDot(plane.n2p(root)) for root in roots))
        root_lines = VGroup(*(Line(plane.n2p(0), d.get_center()) for d in root_dots))
        root_lines.set_stroke(YELLOW, 1)

        self.add(plane)
        self.add(root_lines[0], root_dots[0])
        self.play(
            ShowCreation(circle),
            Rotate(Group(root_lines[0], root_dots[0]), TAU, about_point=plane.n2p(0)),
            run_time=2,
        )
        self.wait()

        # Show powers
        kw = dict(tex_to_color_map={R"\omega": YELLOW}, font_size=36)
        max_power = 3 * N
        powers = VGroup(*(
            OldTex(Rf"\omega^{{{k}}}", **kw)
            for k in range(max_power)
        ))
        powers.set_backstroke(BLACK, 3)
        for power, line in zip(powers, it.cycle(root_lines)):
            vect = line.get_vector()
            vect += UP if line.get_vector()[1] > -0.1 else DOWN
            vect += RIGHT if line.get_vector()[0] > -0.1 else LEFT
            power.next_to(line.get_end(), normalize(vect), buff=SMALL_BUFF)

        shown_powers = VGroup(powers[0])
        moving_power = powers[0].copy()
        for k in range(max_power - 1):
            shown_powers.generate_target()
            shown_powers.target.set_opacity(0.8)
            if k > N - 1:
                shown_powers.target[(k + 1) % N].set_opacity(0)

            kw = dict(path_arc=TAU / N)
            self.play(
                Transform(moving_power, powers[k + 1], **kw),
                Transform(root_lines[k % N].copy(), root_lines[(k + 1) % N], remover=True, **kw),
                Transform(root_dots[k % N].copy(), root_dots[(k + 1) % N], remover=True, **kw),
                MoveToTarget(shown_powers, **kw),
            )
            self.add(root_lines[:k + 2], root_dots[:k + 2])
            if k < N - 1:
                shown_powers.add(powers[k + 1])
        self.play(
            FadeOut(moving_power),
            powers[:N].animate.set_opacity(1)
        )
        self.wait()


class AlgorithmOutline(InteractiveScene):
    def construct(self):
        # Title
        title = Text("Fast(?) convolution algorithm")
        title.to_edge(UP, buff=0.35)
        title.set_backstroke(BLACK, 3)
        underline = Underline(title, buff=-0.05).scale(1.5)
        underline.insert_n_curves(20)
        underline.set_stroke(GREY, (0, 3, 3, 3, 0))
        self.add(underline, title)

        # Arrays
        t2c = {
            tex: BLUE
            for tex in [R"\textbf{a}", "a_0", "a_1", "a_2", "a_{n - 1}"]
        }
        t2c.update({
            tex: TEAL
            for tex in [R"\textbf{b}", "b_0", "b_1", "b_2", "b_{m - 1}"]
        })
        tex_kw = dict(tex_to_color_map=t2c, font_size=36)
        lists = VGroup(
            Tex(R"\textbf{a} = [a_0, a_1, a_2, \dots, a_{n - 1}]", **tex_kw),
            Tex(R"\textbf{b} = [b_0, b_1, b_2, \dots, b_{m - 1}]", **tex_kw),
            Tex(R"""\textbf{a} * \textbf{b} = \left[\begin{array}{l}
                    a_0 b_0, \\
                    a_0 b_1 + a_1 b_0, \\
                    a_0 b_2 + a_1 b_1  + a_2 b_0, \\
                    \quad \vdots \\
                    a_{n - 1} b_{m - 1}
            \end{array}\right]""", **tex_kw),
        )
        lists.arrange(DOWN, buff=1.7, aligned_edge=LEFT)
        lists.next_to(underline, DOWN, LARGE_BUFF)
        lists.to_edge(LEFT)
        lists[2][4:].scale(0.7, about_edge=LEFT)
        lists[2].refresh_bounding_box()
        lists[2].to_edge(DOWN, buff=MED_SMALL_BUFF)
        conv_rect = SurroundingRectangle(lists[2], buff=SMALL_BUFF)
        conv_rect.set_stroke(YELLOW, 2)
        q_marks = Text("???", color=YELLOW)
        q_marks.next_to(conv_rect, RIGHT, MED_SMALL_BUFF)

        self.add(lists[:2])
        self.play(ShowCreation(underline))
        self.play(
            TransformMatchingShapes(VGroup(*lists[0], *lists[1]).copy(), lists[2]),
            FadeIn(conv_rect),
            FadeIn(q_marks),
        )
        self.wait()

        # Show polynomials
        polys = VGroup(
            Tex(R"f(x) = a_0 + a_1 x + a_2 x^2 + \cdots + a_{n - 1}x^{n - 1}", **tex_kw),
            Tex(R"g(x) = b_0 + b_1 x + b_2 x^2 + \cdots + b_{m - 1}x^{m - 1}", **tex_kw),
        )
        for poly, listy in zip(polys, lists):
            poly.next_to(listy, DOWN, aligned_edge=LEFT)

        axes = VGroup(*(
            Axes((-3, 3), (-2, 2), width=5, height=1.5)
            for x in range(3)
        ))
        axes.to_edge(RIGHT)
        axes[0].match_y(polys[0])
        axes[1].match_y(polys[1])
        axes[2].move_to(2 * axes[1].get_origin() - axes[0].get_origin())

        def f(x):
            return 0.2 * (x + 3) * (x + 1) * (x - 2)

        def g(x):
            return 0.1 * (x + 2) * (x + 0) * (x - 3.25)

        graphs = VGroup(
            axes[0].get_graph(f, color=BLUE),
            axes[1].get_graph(g, color=TEAL),
            axes[2].get_graph(lambda x: f(x) * g(x), color=YELLOW),
        )
        graphs.set_stroke(width=2)

        label_kw = dict(font_size=24)
        graph_labels = VGroup(
            Tex("f(x)", **label_kw).move_to(axes[0], UL),
            Tex("g(x)", **label_kw).move_to(axes[1], UL),
            Tex(R"f(x) \cdot g(x)", **label_kw).move_to(axes[2], UL),
        )

        self.play(
            LaggedStart(*(
                TransformFromCopy(listy.copy(), poly)
                for listy, poly in zip(lists, polys)
            )),
            LaggedStartMap(FadeIn, axes[:2]),
            LaggedStartMap(ShowCreation, graphs[:2]),
            LaggedStartMap(FadeIn, graph_labels[:2]),
            lag_ratio=0.5,
        )
        self.wait()

        # Show samples
        x_samples = np.arange(-3, 3.5, 0.5)
        f_points, g_points, fg_points = all_points = [
            [ax.i2gp(x, graph) for x in x_samples]
            for ax, graph in zip(axes, graphs)
        ]

        f_dots, g_dots, fg_dots = all_dots = Group(*(
            Group(*(GlowDot(point, color=WHITE, glow_factor=0.8, radius=0.07) for point in points))
            for points in all_points
        ))

        self.play(
            FadeIn(f_dots, lag_ratio=0.7),
            FadeIn(g_dots, lag_ratio=0.7),
            run_time=5
        )
        self.wait()
        self.play(
            TransformFromCopy(axes[1], axes[2]),
            TransformMatchingShapes(graph_labels[:2].copy(), graph_labels[2]),
            LaggedStart(*(
                Transform(Group(fd, gd).copy(), Group(fgd), remover=True)
                for fd, gd, fgd in zip(*all_dots)
            ), lag_ratio=0.5, run_time=5)
        )
        self.add(fg_dots)
        self.play(ShowCreation(graphs[2], run_time=2))
        self.wait()

        # Show arrow
        final_arrow = Arrow(graph_labels[2], conv_rect.get_corner(UR), path_arc=PI / 5)
        final_arrow.set_color(RED)

        self.play(Write(final_arrow))
        self.wait()

        # Erase graphs
        crosses = VGroup(*map(Cross, axes[:2]))
        for cross in crosses:
            cross.insert_n_curves(20)
            cross.set_stroke(RED, (0, 10, 10, 10, 0))

        self.play(
            LaggedStartMap(ShowCreation, crosses, lag_ratio=0.5, run_time=2),
            polys[0].animate.scale(0.5, about_edge=DL),
            polys[1].animate.scale(0.5, about_edge=DL),
            LaggedStartMap(FadeOut, Group(
                final_arrow, axes[2], graph_labels[2], fg_dots, graphs[2], q_marks,
            ))
        )
        self.play(
            LaggedStartMap(FadeOut, Group(
                axes[0], graphs[0], graph_labels[0], f_dots, crosses[0],
                axes[1], graphs[1], graph_labels[1], g_dots, crosses[1],
            ), shift=0.2 * RIGHT)
        )

        # Show FFTs
        t2c = {
            tex: RED
            for tex in [R"\hat{\textbf{a}}", R"\hat{a}_0", R"\hat{a}_1", R"\hat{a}_2", R"\hat{a}_{m + n - 1}"]
        }
        t2c.update({
            tex: MAROON_C
            for tex in [R"\hat{\textbf{b}}", R"\hat{b}_0", R"\hat{b}_1", R"\hat{b}_2", R"\hat{b}_{m + n - 1}"]
        })
        fft_kw = dict(tex_to_color_map=t2c, font_size=36, isolate=["="])
        fft_lists = VGroup(
            Tex(R"\hat{\textbf{a}} = [\hat{a}_0, \hat{a}_1, \hat{a}_2, \dots, \hat{a}_{m + n - 1}]", **fft_kw),
            Tex(R"\hat{\textbf{b}} = [\hat{b}_0, \hat{b}_1, \hat{b}_2, \dots, \hat{b}_{m + n - 1}]", **fft_kw),
            Tex(R"""\hat{\textbf{a}} \cdot \hat{\textbf{b}} = [
                    \hat{a}_0 \hat{b}_0,
                    \hat{a}_1 \hat{b}_1,
                    \hat{a}_2 \hat{b}_2,
                    \dots,
            ]""", **fft_kw),
        )
        for fft_list in fft_lists:
            fft_list.shift(-fft_list.select_part("=").get_center())
        fft_lists.to_edge(RIGHT)

        arrows = VGroup()
        arrow_labels = VGroup()
        for orig_list, fft_list, n in zip(lists, fft_lists, it.count()):
            fft_list.match_y(orig_list)
            arrow = Arrow(orig_list, fft_list, buff=0.3)
            arrow.label = Text("Inverse FFT" if n == 2 else "FFT", font_size=36)
            arrow.label.next_to(arrow, UP)
            arrow_labels.add(arrow.label)
            arrows.add(arrow)
        arrows[2].rotate(PI)

        mult_arrow = Vector(2 * DOWN).move_to(VGroup(fft_lists[1], fft_lists[2]).get_center())
        mult_arrow.label = Text("Multiply\n(pointwise)", font_size=36)
        mult_arrow.label.next_to(mult_arrow, RIGHT)

        kw = dict(lag_ratio=0.75, run_time=2)
        self.play(
            title[:4].animate.match_x(title[4:7], RIGHT),
            FadeOut(title[4:7], 0.1 * DOWN, lag_ratio=0.1),
        )
        self.play(
            LaggedStartMap(FadeIn, fft_lists[:2], shift=1.5 * RIGHT, **kw),
            LaggedStartMap(GrowArrow, arrows[:2], **kw),
            LaggedStartMap(FadeIn, arrow_labels[:2], shift=0.5 * RIGHT, **kw),
        )
        self.wait()
        self.play(LaggedStart(
            polys[0].animate.scale(1.5, about_edge=DL),
            polys[1].animate.scale(1.5, about_edge=DL),
            lag_ratio=0.5,
        ))
        self.wait()
        self.play(
            FadeIn(fft_lists[2], DOWN),
            GrowArrow(mult_arrow),
            FadeIn(mult_arrow.label, 0.5 * DOWN)
        )
        self.wait()
        self.play(
            GrowArrow(arrows[2]),
            FadeIn(arrow_labels[2], shift=LEFT),
        )
        self.wait()


# TODO
class FourierCoefficients(InteractiveScene):
    def construct(self):
        # Axes
        axes = Axes((0, 1, 0.1), (0, 2), width=8, height=4)
        axes.add_coordinate_labels(font_size=24, num_decimal_places=1)
        axes.to_edge(DOWN)
        self.add(axes)

        def f(x):
            return max(np.exp(-x**2) + math.cos(2 * PI * x) - 0.2 * math.sin(4 * PI * x), 0.25)

        graph = axes.get_graph(f)
        graph.set_stroke(BLUE, 2)

        self.add(graph)

        # Coefficients
        exp = R"{\left(e^{2\pi i \cdot t}\right)}"
        kw = dict(tex_to_color_map={
            "{x}": BLUE,
            exp: TEAL,
        })
        equations = VGroup(
            Tex(R"f(x) = c_0 + c_1 {x} + c_2 {x}^2 + c_3 {x}^3 + \dots", **kw),
            Tex(Rf"f(t) = c_0 + c_1 {exp} + c_2 {exp}^2 + c_3 {exp}^3 + \dots", **kw),
            Tex(Rf"f(t) = \cdots + c_{{-1}} {exp}^{{-1}} + c_0 + c_1 {exp} + c_2 {exp}^2 + \dots", **kw),
            # Tex(R"f(t) = \cdots + \hat f(-1) e^{-2\pi i t} + \hat f(0) + \hat f(1) e^{2\pi i t} + \hat f(2) e^{2\pi i \cdot 2t} + \dots"),
        )

        equations = VGroup

        last = VMobject()
        last.move_to(FRAME_HEIGHT * UP / 2)
        for equation in equations:
            equation.next_to(last, DOWN, MED_LARGE_BUFF)
            self.play(FadeIn(equation, DOWN))
            self.wait()
            self.play(
                FadeOut(last, UP),
                equation.animate.to_edge(UP)
            )
            last = equation


class EndScreen(PatreonEndScreen):
    pass
