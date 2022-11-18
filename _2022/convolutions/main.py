from manim_imports_ext import *
from _2022.borwein.main import *
from typing import Union
import scipy.signal

ManimColor = Union[str, Color]


class DieFace(VGroup):
    def __init__(self,
                 value: int,
                 side_length: float = 1.0,
                 corner_radius: float = 0.15,
                 stroke_color: ManimColor = WHITE,
                 stroke_width: float = 2.0,
                 fill_color: ManimColor = GREY_E,
                 dot_radius: float = 0.08,
                 dot_color: ManimColor = BLUE_B,
                 dot_coalesce_factor: float = 0.5):
        dot = Dot(radius=dot_radius, fill_color=dot_color)
        square = Square(
            side_length=side_length,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            fill_color=fill_color,
            fill_opacity=1.0,
        )
        square.round_corners(corner_radius)

        if not (1 <= value <= 6):
            raise Exception("DieFace only accepts integer inputs between 1 and 6")

        edge_group = [
            (ORIGIN,),
            (UL, DR),
            (UL, ORIGIN, DR),
            (UL, UR, DL, DR),
            (UL, UR, ORIGIN, DL, DR),
            (UL, UR, LEFT, RIGHT, DL, DR),
        ][value - 1]

        arrangement = VGroup(*(
            dot.copy().move_to(square.get_bounding_box_point(vect))
            for vect in edge_group
        ))
        arrangement.space_out_submobjects(dot_coalesce_factor)

        super().__init__(square, arrangement)
        self.value = value
        self.index = value


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
        f_label = MTex("f(x)", **kw).move_to(axes1.get_corner(UL), UL)
        g_label = MTex("g(x)", **kw).move_to(axes2.get_corner(UL), UL)
        sum_label = MTex("[f + g](x)", **kw).move_to(axes3.get_corner(UL), UL)
        prod_label = MTex(R"[f \cdot g](x)", **kw).move_to(axes3.get_corner(UL), UL)
        conv_label = MTex(R"[f * g](x)", **kw).move_to(axes3.get_corner(UL), UL)
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
            Tex(f"a = {list(seq1)}", **kw),
            Tex(f"b = {list(seq2)}", **kw),
            Tex(f"a + b = {list(seq1 + seq2)}", **kw),
            Tex(Rf"a \cdot b = {list(seq1 * seq2)}", **kw),
            Tex(Rf"a * b = {list(np.convolve(seq1, seq2))}", **kw),
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
            Tex("6^2 = "), Integer(36), Text("Combinations")
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
            Tex("1 / 6", font_size=36).next_to(die, UP, SMALL_BUFF)
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
                blue_labels = VGroup(*(Tex(f"a_{{{i}}}", font_size=42) for i in range(1, 7)))
                red_labels = VGroup(*(Tex(f"b_{{{i}}}", font_size=42) for i in range(1, 7)))
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

        conv_name = TexText(
            R"Convolution of ", "$(a_i)$", " and ", "$(b_i)$",
            tex_to_color_map={"$(a_i)$": BLUE, "$(b_i)$": RED}
        )
        conv_eq = MTex(
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
        alt_rhs = Tex(R"\sum_{i = 1}^6 a_i \cdot b_{n - i}")
        alt_rhs.scale(0.9)
        alt_rhs.move_to(conv_eq[7], LEFT)

        self.play(
            FadeIn(alt_rhs, 0.5 * DOWN),
            conv_eq[7:].animate.shift(1.5 * DOWN).set_opacity(0.5),
            dice.animate.to_edge(DOWN)
        )
        self.wait()

    def get_p_sum_expr(self, n, rhs=" "):
        raw_expr = MTex(
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
            rhs.add(Tex(R"\cdot"))
            rhs.add(red.prob_label.copy().set_color(RED))
            rhs.add(Tex("+"))
        rhs.remove(rhs[-1])
        rhs.arrange(RIGHT, buff=SMALL_BUFF)
        rhs.next_to(prob_label, RIGHT, buff=0.2)
        return rhs


class SimpleExample(InteractiveScene):
    def construct(self):
        # Question
        question = Text("What is")
        conv = MTex("(1, 2, 3) * (4, 5, 6)")
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
        rhs = Tex(*rhs_args)
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
                dot = Tex(R"\dot")
                dot.move_to(label_pair)
                plus = Tex("+")
                plus.next_to(label_pair, RIGHT, SMALL_BUFF)
                expr.add(*label_pair, dot, plus)
                symbols.add(dot, plus)
            symbols[-1].scale(0, about_point=symbols[-2].get_right())
            expr.next_to(label_pairs, UP, LARGE_BUFF)
            c_label = Tex(f"c_{n}", font_size=30, color=YELLOW).next_to(rhs[2 * n + 2], UP)

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
        polynomial_eq = MTex(
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
            value.target = MTex(f"{value.get_value()}{tex}")
            value.target.scale(value.get_height() / value.target[0].get_height())
            value.target.move_to(value, DOWN)
        values2[1].target.align_to(values2[0].target, RIGHT)
        values2[2].target.align_to(values2[0].target, RIGHT)
        for n, diag_group in enumerate(diag_groups):
            tex = ["", "x", "x^2", "x^3", "x^4"][n]
            for product in diag_group:
                product.target = MTex(f"{product.get_value()}{tex}")
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
            label = Tex(f"{letter}_{{{n}}}", font_size=font_size)
            label.set_color(color)
            label.next_to(square, UP, SMALL_BUFF)
            square.label = label
            square.add(label)
            labels.add(label)
        return labels


class ConvolveDiscreteDistributions(InteractiveScene):
    def construct(self):
        # Set up two distributions
        dist1 = np.array([np.exp(-0.25 * (x - 3)**2) for x in range(6)])
        dist2 = np.array([1.0 / (x + 1)**1.2 for x in range(6)])
        for dist in dist1, dist2:
            dist /= dist.sum()

        top_bars = dist_to_bars(dist1, bar_colors=(BLUE_D, TEAL_D))
        low_bars = dist_to_bars(dist2, bar_colors=(RED_D, GOLD_E))
        all_bars = VGroup(top_bars, low_bars)
        all_bars.arrange(DOWN, buff=1.5)
        all_bars.move_to(4.5 * LEFT)

        add_labels_to_bars(top_bars, dist1)
        add_labels_to_bars(low_bars, dist2)

        for bars, color in (top_bars, BLUE_E), (low_bars, RED_E):
            for i, bar in zip(it.count(1), bars):
                die = DieFace(i, fill_color=color, stroke_width=1, dot_color=WHITE)
                die.set_width(bar.get_width() * 0.7)
                die.next_to(bar, DOWN, SMALL_BUFF)
                bar.die = die
                bar.add(die)
                bar.index = i

        # V lines
        v_lines = get_bar_dividing_lines(top_bars)
        VGroup()
        for bar in top_bars:
            v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
            v_line.set_stroke(GREY_C, 1, 0.75)
            v_line.set_x(bar.get_left()[0])
            v_line.set_y(0)
            v_lines.add(v_line)
        v_lines.add(v_lines[-1].copy().set_x(top_bars.get_right()[0]))
        # v_lines.set_stroke(opacity=0)

        # Set up new distribution
        conv_dist = np.convolve(dist1, dist2)
        conv_bars = dist_to_bars(conv_dist, bar_colors=(GREEN_E, YELLOW_E))
        conv_bars.to_edge(RIGHT)

        add_labels_to_bars(conv_bars, conv_dist)

        for n, bar in zip(it.count(2), conv_bars):
            sum_sym = VGroup(
                top_bars[0].die.copy().scale(0.7),
                Tex("+", font_size=16),
                low_bars[0].die.copy().scale(0.7),
                Tex("=", font_size=24).rotate(PI / 2),
                Tex(str(n), font_size=24),
            )
            sum_sym[0].remove(sum_sym[0][1])
            sum_sym[2].remove(sum_sym[2][1])
            sum_sym.arrange(DOWN, buff=SMALL_BUFF)
            sum_sym[:2].shift(0.05 * DOWN)
            sum_sym[:1].shift(0.05 * DOWN)
            sum_sym.next_to(bar, DOWN, buff=SMALL_BUFF)
            bar.add(sum_sym)

        # Dist labels
        plabel_kw = dict(tex_to_color_map={"X": BLUE, "Y": RED})
        PX = MTex("P_X", **plabel_kw)
        PY = MTex("P_Y", **plabel_kw)
        PXY = MTex("P_{X + Y}", **plabel_kw)

        PX.next_to(top_bars.get_corner(UR), DR)
        PY.next_to(low_bars.get_corner(UR), DR)
        PXY.next_to(conv_bars, UP, LARGE_BUFF)

        # Add distributions
        self.play(
            FadeIn(top_bars, lag_ratio=0.1),
            FadeIn(v_lines, lag_ratio=0.2),
            Write(PX),
        )
        self.wait()
        self.play(
            FadeIn(low_bars, lag_ratio=0.1),
            Write(PY),
        )
        self.wait()

        self.play(
            FadeIn(conv_bars),
            FadeTransform(PX.copy(), PXY),
            FadeTransform(PY.copy(), PXY),
        )
        self.wait()

        # March!
        self.play(low_bars.animate.arrange(LEFT, aligned_edge=DOWN, buff=0).move_to(low_bars))

        last_rects = VGroup()
        for n in range(2, 13):
            conv_bars.generate_target()
            conv_bars.target.set_opacity(0.35)
            conv_bars.target[n - 2].set_opacity(1.0)

            self.play(
                get_row_shift(top_bars, low_bars, n),
                MaintainPositionRelativeTo(PY, low_bars),
                FadeOut(last_rects),
                MoveToTarget(conv_bars),
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
                # Restore(bar[0], time_span=(0.5, 1.0)),
                # Write(bar[2], time_span=(0.5, 1.0)),
            )

            self.play(*(
                FadeTransform(label.copy(), conv_bars[n - 2].value_label)
                for lp in label_pairs
                for label in lp
            ))
            self.wait(0.5)

            last_rects = rects

        conv_bars.target.set_opacity(1.0)
        self.play(
            FadeOut(last_rects),
            get_row_shift(top_bars, low_bars, 7),
            MaintainPositionRelativeTo(PY, low_bars),
            MoveToTarget(conv_bars),
        )

        # Emphasize that these are also functions
        func_label = Text("Function", font_size=36)
        func_label.next_to(PX, UP, LARGE_BUFF, aligned_edge=LEFT)
        func_label.shift_onto_screen(buff=SMALL_BUFF)
        arrow = Arrow(func_label, PX.get_top(), buff=0.2)
        VGroup(func_label, arrow).set_color(YELLOW)
        x_args = VGroup(*(
            MTex(
                f"({x}) = {np.round(dist1[x - 1], 2)}"
            ).next_to(PX, RIGHT, SMALL_BUFF)
            for x in range(1, 7)
        ))
        die_rects = VGroup()
        value_rects = VGroup()
        for index, x_arg in enumerate(x_args):
            x_die = top_bars[index].die
            value_label = top_bars[index].value_label
            die_rect = SurroundingRectangle(x_die, buff=SMALL_BUFF)
            value_rect = SurroundingRectangle(value_label, buff=SMALL_BUFF)
            for rect in die_rect, value_rect:
                rect.set_stroke(YELLOW, 2).round_corners()
            die_rects.add(die_rect)
            value_rects.add(value_rect)

        index = 2
        x_arg = x_args[index]
        die_rect = die_rects[index]
        value_rect = value_rects[index]
        x_die = top_bars[index].die
        value_label = top_bars[index].value_label

        self.play(Write(func_label), ShowCreation(arrow))
        self.wait()
        self.play(ShowCreation(die_rect))
        self.play(FadeTransform(x_die.copy(), x_arg[:3]))
        self.play(TransformFromCopy(die_rect, value_rect))
        self.play(FadeTransform(value_label.copy(), x_arg[3:]))
        self.wait()
        for i in range(6):
            self.remove(*die_rects, *value_rects, *x_args)
            self.add(die_rects[i], value_rects[i], x_args[i])
            self.wait(0.5)

        func_group = VGroup(func_label, arrow)
        func_group_copies = VGroup(
            func_group.copy().shift(PXY.get_center() - PX.get_center()),
            func_group.copy().shift(PY.get_center() - PX.get_center()),
        )
        self.play(*(
            TransformFromCopy(func_group, func_group_copy)
            for func_group_copy in func_group_copies
        ))
        self.wait()
        self.play(LaggedStartMap(FadeOut, VGroup(
            func_group, *func_group_copies, die_rects[-1], value_rects[-1], *x_args[-1]
        )))

        # State definition again
        conv_def = MTex(
            R"\big[P_X * P_Y\big](s) = \sum_{x = 1}^6 P_X(x) \cdot P_Y(s - x)",
            font_size=36,
            **plabel_kw,
        )
        conv_def.next_to(conv_bars, UP, buff=MED_LARGE_BUFF)

        PXY.generate_target()
        lhs = conv_def[:10]
        PXY.target.next_to(lhs, UP, LARGE_BUFF).shift_onto_screen(buff=SMALL_BUFF)
        eq = Tex("=").rotate(90 * DEGREES)
        eq.move_to(midpoint(PXY.target.get_bottom(), lhs.get_top()))

        self.play(LaggedStart(
            MoveToTarget(PXY),
            Write(eq),
            TransformFromCopy(PX, lhs[1:3]),
            TransformFromCopy(PY, lhs[4:6]),
            Write(VGroup(lhs[0], lhs[3], *lhs[6:])),
        ))
        self.wait()
        self.play(Write(conv_def[10:]))
        self.wait()


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
        sum_label = MTex(R"\sum_i y_i = 1")
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

            globals().update(locals())
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
        conv *= 2  # Delete

        if not self.scalar_conv:
            conv = np.clip(conv, 0, 1)

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
                    value = Tex(tex)
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
                lil_coef = Tex(tex, font_size=36)
            else:
                lil_coef = square[0].copy()
                lil_coef.set_height(lil_height * 0.5)
            expr.add(lil_coef, lil_pixel, Tex("+", font_size=48))

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
        # March!
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


class SobelFilterKirby(SobelFilter2):
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
        result = VGroup(Tex("[", **kw))
        commas = VGroup()
        elements = VGroup()
        for index, elem in enumerate(array):
            int_mob = Integer(elem, **kw)
            int_mob.index = index
            elements.add(int_mob)
            result.add(int_mob)
            comma = Tex(",", **kw)
            commas.add(comma)
            result.add(comma)
        result.remove(commas[-1])
        commas.remove(commas[-1])
        result.add(Tex("]", **kw))
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
                        self.replace_square(square, Tex(R"\ddots"))
                    else:
                        self.replace_square(square, Tex(R"\vdots"))
            else:
                self.replace_square(row[N - 2], Tex(R"\cdots"))

        self.add(grid)

        # Polynomial terms
        a_terms, b_terms = all_terms = [
            VGroup(*(
                MTex(RF"{letter}_{{{n}}} x^{{{n}}}", font_size=24)
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

        a_terms[-2].become(Tex(R"\cdots", font_size=24).move_to(a_terms[-2]).shift(0.02 * DOWN))
        b_terms[-2].become(Tex(R"\vdots", font_size=24).move_to(b_terms[-2]))

        a_terms.set_color(BLUE_C)
        b_terms.set_color(TEAL_C)

        self.add(a_terms)
        self.add(b_terms)

        # Plusses
        for terms, vect in (a_terms, RIGHT), (b_terms, DOWN):
            terms.plusses = VGroup()
            for t1, t2 in zip(terms, terms[1:]):
                plus = Tex("+", font_size=24).match_color(terms)
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
                term = Tex(f"a_{{{j}}}", f"b_{{{i}}}", f"x^{{{i + j}}}", font_size=20)
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
        product_count = TexText(R"$100 \times 100 = 10{,}000$ \\ products", font_size=60)
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
            MTex(tex, **kw).next_to(axes.get_top(), RIGHT, aligned_edge=UP)
            for tex, axes in zip(["f(x)", "g(x)", R"f(x) \cdot g(x)"], all_axes)
        ))

        # Coefficients
        a_labels, b_labels, conv_label = coef_labels = VGroup(
            MTex(R"(a_0, a_1, \dots, a_n)", **kw),
            MTex(R"(b_0, b_1, \dots, b_m)", **kw),
            VGroup(
                MTex("c_0 = a_0 b_0", **kw),
                MTex("c_1 = a_0 b_1 + a_1 b_0", **kw),
                MTex("c_2 = a_0 b_2 + a_1 b_1 + a_2 b_0", **kw),
                MTex("c_3 = a_0 b_3 + a_1 b_2 + a_2 b_1 + a_3 b_0", **kw),
                MTex(R"\vdots", **kw),
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
        axes.add(Tex("x", **kw).next_to(axes.x_axis.get_end(), DR, buff=0.2))
        axes.add(Tex("y", **kw).next_to(axes.y_axis.get_end(), LEFT, MED_SMALL_BUFF))

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
        poly_tex = Tex(
            "a_0", "+", "a_1 x", "+", *it.chain(*(
                (f"a_{n} x^{n}", "+")
                for n in range(2, len(data) - 1)
            ))
        )
        poly_tex.to_corner(UR)

        graph = graphs[1].copy()
        self.play(
            LaggedStartMap(FadeIn, dots[:2], lag_ratio=0.7, run_time=1),
            LaggedStartMap(ShowCreationThenFadeOut, circles[:2], scale=0.25, lag_ratio=0.2, run_time=1),
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
        coefs = VGroup(*(MTex(Rf"c_{{{n}}}") for n in range(N)))
        coefs.set_submobject_colors_by_gradient(BLUE, TEAL)
        x_powers = VGroup(*(MTex(Rf"x^{{{n}}}") for n in range(N)))
        poly_x = self.get_polynomial(coefs, x_powers)

        top_lhs = MTex("h(x) = ")
        top_lhs.next_to(poly_x[0][0], LEFT, buff=0.1)
        top_eq = VGroup(top_lhs, poly_x)
        top_eq.center()

        fg = MTex(R"f(x) \cdot g(x)")
        eq = MTex("=")
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
        lhss = VGroup(*(MTex(f"h({n})=") for n in range(N)))
        rhss = VGroup(*(
            self.get_polynomial(
                coefs.copy(),
                # VGroup(*(Integer(x**n) for n in range(N)))
                VGroup(*(MTex(f"{x}^{{{n}}}") for n in range(N)))
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
        omega_def = Tex(
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
                MTex(Rf"\omega^{{{(k * n) % N}}}", **kw)
                for n in range(N)
            ))
            for k in range(0, N)
        ]
        new_rhss = VGroup(*(
            self.get_polynomial(coefs.copy(), omega_powers)
            for omega_powers in all_omega_powers
        ))

        new_lhss = VGroup(
            *(MTex(Rf"h(\omega^{n}) = ") for n in range(N))
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
        label = TexText(R"Discrete\\Fourier\\Transform", alignment=R"\raggedright", isolate=list("DFT"), **kw)
        label.next_to(brace, LEFT)

        sub_label = TexText("of $(c_i)$", **kw)[0]
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

        short_label = TexText("DFT", isolate=list("DFT"))
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
            run_time = Tex(R"\mathcal{O}\big(N\log(N)\big)")
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
            plus = Tex("+")  # Font size?
            result.add(coef)
            if isinstance(power, Integer):
                dot = Tex(R"\cdot")
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
            Tex(Rf"\omega^{{{k}}}", **kw)
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

            self.play(
                Transform(moving_power, powers[k + 1]),
                Transform(root_lines[k % N].copy(), root_lines[(k + 1) % N], remover=True),
                Transform(root_dots[k % N].copy(), root_dots[(k + 1) % N], remover=True),
                MoveToTarget(shown_powers),
                path_arc=TAU / N
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
            MTex(R"\textbf{a} = [a_0, a_1, a_2, \dots, a_{n - 1}]", **tex_kw),
            MTex(R"\textbf{b} = [b_0, b_1, b_2, \dots, b_{m - 1}]", **tex_kw),
            MTex(R"""\textbf{a} * \textbf{b} = \left[\begin{array}{l}
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
            MTex(R"f(x) = a_0 + a_1 x + a_2 x^2 + \cdots + a_{n - 1}x^{n - 1}", **tex_kw),
            MTex(R"g(x) = b_0 + b_1 x + b_2 x^2 + \cdots + b_{m - 1}x^{m - 1}", **tex_kw),
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
            MTex("f(x)", **label_kw).move_to(axes[0], UL),
            MTex("g(x)", **label_kw).move_to(axes[1], UL),
            MTex(R"f(x) \cdot g(x)", **label_kw).move_to(axes[2], UL),
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
            MTex(R"\hat{\textbf{a}} = [\hat{a}_0, \hat{a}_1, \hat{a}_2, \dots, \hat{a}_{m + n - 1}]", **fft_kw),
            MTex(R"\hat{\textbf{b}} = [\hat{b}_0, \hat{b}_1, \hat{b}_2, \dots, \hat{b}_{m + n - 1}]", **fft_kw),
            MTex(R"""\hat{\textbf{a}} \cdot \hat{\textbf{b}} = [
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
            MTex(R"f(x) = c_0 + c_1 {x} + c_2 {x}^2 + c_3 {x}^3 + \dots", **kw),
            MTex(Rf"f(t) = c_0 + c_1 {exp} + c_2 {exp}^2 + c_3 {exp}^3 + \dots", **kw),
            MTex(Rf"f(t) = \cdots + c_{{-1}} {exp}^{{-1}} + c_0 + c_1 {exp} + c_2 {exp}^2 + \dots", **kw),
            # MTex(R"f(t) = \cdots + \hat f(-1) e^{-2\pi i t} + \hat f(0) + \hat f(1) e^{2\pi i t} + \hat f(2) e^{2\pi i \cdot 2t} + \dots"),
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


# Continuous case


class TransitionToContinuousProbability(InteractiveScene):
    def construct(self):
        # Setup axes and initial graph
        axes = Axes((0, 12), (0, 1, 0.2), width=14, height=5)
        axes.to_edge(LEFT, LARGE_BUFF)

        def pd(x):
            return (x**4) * np.exp(-x) / 8.0

        graph = axes.get_graph(pd)
        graph.set_stroke(WHITE, 2)
        bars = axes.get_riemann_rectangles(graph, dx=1, x_range=(0, 6), input_sample_type="right")
        bars.set_stroke(WHITE, 3)

        y_label = Text("Probability", font_size=24)
        y_label.next_to(axes.y_axis, UP, SMALL_BUFF)

        self.add(axes)
        self.add(y_label)
        self.add(*bars)

        # Label as die probabilities
        dice = get_die_faces(fill_color=BLUE_E, dot_color=WHITE, stroke_width=1)
        dice.set_height(0.5)
        for bar, die in zip(bars, dice):
            die.next_to(bar, DOWN)

        self.play(FadeIn(dice, 0.1 * UP, lag_ratio=0.05, rate_func=overshoot))
        self.wait()
        self.play(FadeOut(dice, RIGHT, rate_func=running_start, run_time=1, path_arc=-PI / 5, lag_ratio=0.01))

        # Make continuous
        all_rects = VGroup(*(
            axes.get_riemann_rectangles(
                graph,
                x_range=(0, min(6 + n, 12)),
                dx=(1 / n),
                input_sample_type="right",
            ).set_stroke(WHITE, width=(2.0 / n), opacity=(2.0 / n), background=False)
            for n in (*range(1, 10), *range(10, 20, 2), *range(20, 100, 5))
        ))
        area = all_rects[-1]
        area.set_stroke(width=0)

        self.remove(bars)
        self.play(ShowSubmobjectsOneByOne(all_rects, rate_func=bezier([0, 0, 0, 0, 1, 1]), run_time=5))
        self.play(ShowCreation(graph))
        self.wait()

        # Show continuous value
        x_tracker = ValueTracker(0)
        get_x = x_tracker.get_value
        tip = ArrowTip(angle=PI / 2)
        tip.set_height(0.25)
        tip.add_updater(lambda m: m.move_to(axes.c2p(get_x(), 0), UP))
        x_label = DecimalNumber(font_size=36)
        x_label.add_updater(lambda m: m.set_value(get_x()))
        x_label.add_updater(lambda m: m.next_to(tip, DOWN, buff=0.2, aligned_edge=LEFT))

        self.play(FadeIn(tip), FadeIn(x_label))
        self.play(x_tracker.animate.set_value(12), run_time=6)
        self.remove(tip, x_label)

        # Labels
        x_label = Text("Value of XYZ next year")
        x_label.next_to(axes.c2p(4, 0), DOWN, buff=0.45)

        density = Text("Probability density")
        density.match_height(y_label)
        density.move_to(y_label, LEFT)
        cross = Cross(y_label)

        self.play(Write(x_label))
        self.wait()
        self.play(ShowCreation(cross))
        self.play(
            VGroup(y_label, cross).animate.shift(0.5 * UP),
            FadeIn(density)
        )
        self.wait()

        # Interpretation
        range_tracker = ValueTracker([0, 12])

        def update_area(area):
            values = range_tracker.get_value()
            x1, x2 = axes.x_axis.n2p(values)[:, 0]
            for bar in area:
                if x1 < bar.get_x() < x2:
                    bar.set_opacity(1)
                else:
                    bar.set_opacity(0.25)

        area.add_updater(update_area)

        v_lines = Line(DOWN, UP).replicate(2)
        v_lines.set_stroke(GREY_A, 1)
        v_lines.set_height(FRAME_HEIGHT)

        def update_v_lines(v_lines):
            values = range_tracker.get_value()
            for value, line in zip(values, v_lines):
                line.move_to(axes.c2p(value, 0), DOWN)

        v_lines.add_updater(update_v_lines)

        self.play(
            range_tracker.animate.set_value([3, 5]),
            VFadeIn(v_lines),
            run_time=2,
        )
        self.wait()
        for pair in [(5, 6), (1, 3), (2.5, 3), (2, 7), (4, 5), (0, 12)]:
            self.play(range_tracker.animate.set_value(pair), run_time=2)
            self.wait()


class Convolutions(InteractiveScene):
    axes_config = dict(
        x_range=(-3, 3, 1),
        y_range=(-1, 1, 1.0),
        width=6,
        height=2,
    )
    f_graph_style = dict(stroke_color=BLUE, stroke_width=2)
    g_graph_style = dict(stroke_color=YELLOW, stroke_width=2)
    fg_graph_style = dict(stroke_color=GREEN, stroke_width=4)
    conv_graph_style = dict(stroke_color=TEAL, stroke_width=2)
    f_graph_x_step = 0.1
    g_graph_x_step = 0.1
    f_label_tex = "f(x)"
    g_label_tex = "g(t - x)"
    fg_label_tex = R"f(x) \cdot g(t - x)"
    t_color = TEAL
    area_line_dx = 0.05
    jagged_product = True
    g_is_rect = False

    def setup(self):
        super().setup()
        if self.g_is_rect:
            k_tracker = self.k_tracker = ValueTracker(1)

        # Add axes
        all_axes = self.all_axes = self.get_all_axes()
        f_axes, g_axes, fg_axes, conv_axes = all_axes
        x_min, x_max = self.axes_config["x_range"][:2]

        self.disable_interaction(*all_axes)
        self.add(*all_axes)

        # Add f(x)
        f_graph = self.f_graph = f_axes.get_graph(self.f, x_range=(x_min, x_max, self.f_graph_x_step))
        f_graph.set_style(**self.f_graph_style)
        f_label = self.get_label(self.f_label_tex, f_axes)
        if self.jagged_product:
            f_graph.make_jagged()

        self.add(f_graph)
        self.add(f_label)

        # Add g(t - x)
        self.toggle_selection_mode()  # So triangle is highlighted
        t_indicator = self.t_indicator = ArrowTip().rotate(90 * DEGREES)
        t_indicator.set_height(0.15)
        t_indicator.set_fill(self.t_color, 0.8)
        t_indicator.move_to(g_axes.get_origin(), UP)
        t_indicator.add_updater(lambda m: m.align_to(g_axes.get_origin(), UP))

        def get_t():
            return g_axes.x_axis.p2n(t_indicator.get_center())

        g_graph = self.g_graph = g_axes.get_graph(lambda x: 0, x_range=(x_min, x_max, self.g_graph_x_step))
        g_graph.set_style(**self.g_graph_style)
        if self.g_is_rect:
            x_min = g_axes.x_axis.x_min
            x_max = g_axes.x_axis.x_max
            g_graph.add_updater(lambda m: m.set_points_as_corners([
                g_axes.c2p(x, y)
                for t in [get_t()]
                for k in [k_tracker.get_value()]
                for x, y in [
                    (x_min, 0), (-0.5 / k + t, 0), (-0.5 / k + t, k), (0.5 / k + t, k), (0.5 / k + t, 0), (x_max, 0)
                ]
            ]))
        else:
            g_axes.bind_graph_to_func(g_graph, lambda x: self.g(get_t() - x), jagged=self.jagged_product)

        g_label = self.g_label = self.get_label(self.g_label_tex, g_axes)

        t_label = VGroup(*Tex("t = ")[0], DecimalNumber())
        t_label.arrange(RIGHT, buff=SMALL_BUFF)
        t_label.scale(0.5)
        t_label.set_backstroke(width=8)
        t_label.add_updater(lambda m: m.next_to(t_indicator, DOWN, submobject_to_align=m[0], buff=0.15))
        t_label.add_updater(lambda m: m.shift(m.get_width() * LEFT / 2))
        t_label.add_updater(lambda m: m[-1].set_value(get_t()))

        self.add(g_graph)
        self.add(g_label)
        self.add(t_indicator)
        self.add(t_label)

        # Show integral of f(x) * g(t - x)
        def prod_func(x):
            k = self.k_tracker.get_value() if self.g_is_rect else 1
            return self.f(x) * self.g((get_t() - x) * k) * k

        fg_graph, pos_graph, neg_graph = (
            fg_axes.get_graph(lambda x: 0, x_range=(x_min, x_max, self.g_graph_x_step))
            for x in range(3)
        )
        fg_graph.set_style(**self.fg_graph_style)
        VGroup(pos_graph, neg_graph).set_stroke(width=0)
        pos_graph.set_fill(BLUE, 0.5)
        neg_graph.set_fill(RED, 0.5)

        get_discontinuities = None
        if self.g_is_rect:
            def get_discontinuities():
                k = self.k_tracker.get_value()
                return [get_t() - 0.5 / k, get_t() + 0.5 / k]

        kw = dict(
            jagged=self.jagged_product,
            get_discontinuities=get_discontinuities,
        )
        fg_axes.bind_graph_to_func(fg_graph, prod_func, **kw)
        fg_axes.bind_graph_to_func(pos_graph, lambda x: max(prod_func(x), 0), **kw)
        fg_axes.bind_graph_to_func(neg_graph, lambda x: min(prod_func(x), 0), **kw)

        self.prod_graphs = VGroup(fg_graph, pos_graph, neg_graph)

        fg_label = self.fg_label = self.get_label(self.fg_label_tex, fg_axes)

        self.add(pos_graph, neg_graph, fg_axes, fg_graph)
        self.add(fg_label)

        # Show convolution
        conv_graph = self.conv_graph = self.get_conv_graph(conv_axes, self.f, self.g)

        graph_dot = GlowDot(color=WHITE)
        graph_dot.add_updater(lambda d: d.move_to(conv_graph.quick_point_from_proportion(
            inverse_interpolate(x_min, x_max, get_t())
        )))
        graph_line = Line(stroke_color=WHITE, stroke_width=1)
        graph_line.add_updater(lambda l: l.put_start_and_end_on(
            graph_dot.get_center(),
            [graph_dot.get_x(), conv_axes.get_y(), 0],
        ))
        self.conv_graph_dot = graph_dot
        self.conv_graph_line = graph_line

        conv_label = Tex(
            R"(f * g)(t) := \int_{-\infty}^\infty f(x) \cdot g(t - x) dx",
            font_size=36
        )
        conv_label.next_to(conv_axes, UP)

        self.add(conv_graph)
        self.add(graph_dot)
        self.add(graph_line)
        self.add(conv_label)

        # Now play!

    def get_all_axes(self):
        all_axes = VGroup(*(Axes(**self.axes_config) for x in range(4)))
        all_axes[:3].arrange(DOWN, buff=0.75)
        all_axes[3].next_to(all_axes[:3], RIGHT, buff=1.5)
        all_axes[3].y_axis.stretch(2, 1)
        all_axes.to_edge(LEFT)
        all_axes.to_edge(DOWN, buff=0.1)

        for i, axes in enumerate(all_axes):
            x_label = Tex("x" if i < 3 else "t", font_size=24)
            x_label.next_to(axes.x_axis.get_right(), UP, MED_SMALL_BUFF)
            axes.x_label = x_label
            axes.x_axis.add(x_label)
            axes.y_axis.ticks.set_opacity(0)
            axes.x_axis.ticks.stretch(0.5, 1)

        return all_axes

    def get_label(self, tex, axes):
        label = Tex(tex, font_size=36)
        label.move_to(midpoint(axes.get_origin(), axes.get_right()))
        label.match_y(axes.get_top())
        return label

    def get_conv_graph(self, axes, f, g, dx=0.1):
        dx = 0.1
        x_min, x_max = axes.x_range[:2]
        x_samples = np.arange(x_min, x_max + dx, dx)
        f_samples = np.array([f(x) for x in x_samples])
        g_samples = np.array([g(x) for x in x_samples])
        conv_samples = np.convolve(f_samples, g_samples, mode='same')
        conv_graph = VMobject().set_style(**self.conv_graph_style)
        conv_graph.set_points_smoothly(axes.c2p(x_samples, conv_samples * dx))
        return conv_graph

    def f(self, x):
        return 0.5 * np.exp(-0.8 * x**2) * (0.5 * x**3 - 3 * x + 1)

    def g(self, x):
        return np.exp(-x**2) * np.sin(2 * x)


class ProbConvolutions(Convolutions):
    jagged_product = True

    def f(self, x):
        return max(-abs(x) + 1, 0)

    def g(self, x):
        return 0.5 * np.exp(-6 * (x - 0.5)**2) + np.exp(-6 * (x + 0.5)**2)


class ProbConvolutionControlled(ProbConvolutions):
    t_time_pairs = [(-2.5, 4), (2.5, 10), (-1, 6)]
    initial_t = 0

    def construct(self):
        t_indicator = self.t_indicator
        g_axes = self.all_axes[1]

        def set_t(t):
            return t_indicator.animate.set_x(g_axes.c2p(t, 0)[0])

        t_indicator.set_x(g_axes.c2p(self.initial_t, 0)[0])
        for t, time in self.t_time_pairs:
            self.play(set_t(t), run_time=time)
            self.wait()


class ProbConvolutionControlledToMatch3D(ProbConvolutionControlled):
    t_time_pairs = [(1.5, 4), (-0.5, 8), (1.0, 8)]
    initial_t = 0.5


class AltConvolutions(Convolutions):
    jagged_product = True

    def construct(self):
        t_indicator = self.t_indicator
        g_axes = self.all_axes[1]

        # Sample values
        for t in [3, -3, -1.0]:
            self.play(t_indicator.animate.set_x(g_axes.c2p(t, 0)[0]), run_time=3)
            self.wait()

    def f(self, x):
        if x < -2:
            return -0.5
        elif x < -1:
            return x + 1.5
        elif x < 1:
            return -0.5 * x
        else:
            return 0.5 * x - 1

    def g(self, x):
        return np.exp(-3 * x**2)


class MovingAverageAsConvolution(Convolutions):
    g_graph_x_step = 0.1
    jagged_product = True
    g_is_rect = True

    def construct(self):
        # Setup
        super().construct()
        t_indicator = self.t_indicator
        g_axes = self.all_axes[1]
        self.g_label.shift(0.25 * UP)

        y_axes = VGroup(*(axes.y_axis for axes in self.all_axes[1:3]))
        fake_ys = y_axes.copy()
        for fake_y in fake_ys:
            fake_y.stretch(1.2, 1)
        self.add(*fake_ys, *self.mobjects)

        # Sample values
        def set_t(t):
            return t_indicator.animate.set_x(g_axes.c2p(t, 0)[0])

        self.play(set_t(-2.5), run_time=2)
        self.play(set_t(2.5), run_time=8)
        self.wait()
        self.play(set_t(-1), run_time=3)
        self.wait()

        # Isolate to slice
        top_line, side_line = Line().replicate(2)
        top_line.add_updater(lambda l: l.put_start_and_end_on(*self.g_graph.get_anchors()[4:6]))
        side_line.add_updater(lambda l: l.put_start_and_end_on(*self.g_graph.get_anchors()[2:4]))

        top_line.set_stroke(width=0)
        self.add(top_line)

        left_rect, right_rect = fade_rects = FullScreenFadeRectangle().replicate(2)
        left_rect.add_updater(lambda m: m.set_x(top_line.get_left()[0], RIGHT))
        right_rect.add_updater(lambda m: m.set_x(top_line.get_right()[0], LEFT))

        self.play(FadeIn(fade_rects))
        self.play(set_t(-2), run_time=3)
        self.play(set_t(-0.5), run_time=3)
        self.wait()
        self.play(FadeOut(fade_rects))

        # Show rect dimensions
        get_k = self.k_tracker.get_value
        top_label = DecimalNumber(1, font_size=24)
        top_label.add_updater(lambda m: m.set_value(1 / get_k()))
        top_label.add_updater(lambda m: m.next_to(top_line, UP, SMALL_BUFF))
        side_label = DecimalNumber(1, font_size=24)
        side_label.add_updater(lambda m: m.set_value(get_k()))
        side_label.add_updater(lambda m: m.next_to(side_line, LEFT, SMALL_BUFF))

        def change_k(k, run_time=3):
            new_conv_graph = self.get_conv_graph(
                self.all_axes[3], self.f, lambda x: self.g(k * x) * k,
            )
            self.play(
                self.k_tracker.animate.set_value(k),
                Transform(self.conv_graph, new_conv_graph),
                run_time=run_time
            )

        top_line.set_stroke(WHITE, 3)
        side_line.set_stroke(RED, 3)
        self.play(
            ShowCreation(side_line),
            VFadeIn(side_label)
        )
        self.wait()
        self.play(
            ShowCreation(top_line),
            VFadeIn(top_label),
        )
        self.wait()
        change_k(0.5)
        self.wait()
        self.play(set_t(-1.5), run_time=3)
        self.wait()
        change_k(2)
        self.wait()
        change_k(1)
        self.play(*map(FadeOut, [top_label, top_line, side_label, side_line]))

        # Show area
        rect = Rectangle()
        rect.set_fill(YELLOW, 0.5)
        rect.set_stroke(width=0)
        rect.set_gloss(1)
        rect.add_updater(lambda m: m.set_width(g_axes.x_axis.unit_size / get_k(), stretch=True))
        rect.add_updater(lambda m: m.set_height(g_axes.y_axis.unit_size * get_k(), stretch=True))
        rect.add_updater(lambda m: m.set_x(t_indicator.get_x()))
        rect.add_updater(lambda m: m.set_y(g_axes.get_origin()[1], DOWN))

        area_label = Tex(R"\text{Area } = 1", font_size=36)
        area_label.next_to(rect, UP, MED_LARGE_BUFF)
        area_label.to_edge(LEFT)
        arrow = Arrow(area_label.get_bottom(), rect.get_center())

        avg_label = TexText(R"Average value of\\$f(x)$ in the window", font_size=24)
        avg_label.move_to(area_label, DL)
        shift_value = self.all_axes[2].get_origin() - g_axes.get_origin() + 0.5 * DOWN
        avg_label.shift(shift_value)
        arrow2 = arrow.copy().shift(shift_value)

        self.play(
            Write(area_label, stroke_color=WHITE),
            ShowCreation(arrow),
            FadeIn(rect)
        )
        self.wait()
        self.play(
            FadeIn(avg_label, lag_ratio=0.1),
            ShowCreation(arrow2)
        )
        self.wait()
        for k in [1.4, 0.8, 1.0]:
            change_k(k)
        self.play(*map(FadeOut, [area_label, arrow, avg_label, arrow2]))

        # Slide once more
        self.play(set_t(-2.5), run_time=3)
        self.play(set_t(2.5), run_time=8)

    def f(self, x):
        return kinked_function(x)

    def g(self, x):
        return rect_func(x)


class GaussianConvolution(Convolutions):
    jagged_product = True

    def f(self, x):
        return np.exp(-x**2) / np.sqrt(PI)

    def g(self, x):
        return np.exp(-x**2) / np.sqrt(PI)


class DiagonalSlices(ProbConvolutions):
    def setup(self):
        InteractiveScene.setup(self)

    def construct(self):
        # Add axes
        frame = self.camera.frame
        axes = self.axes = ThreeDAxes(
            (-2, 2), (-2, 2), (0, 1),
            height=7, width=7, depth=2
        )
        axes.z_axis.apply_depth_test()
        axes.add_axis_labels(z_tex="", font_size=36)
        plane = NumberPlane(
            (-2, 2), (-2, 2), height=7, width=7,
            axis_config=dict(
                stroke_width=1,
                stroke_opacity=0.5,
            ),
            background_line_style=dict(
                stroke_color=GREY_B, stroke_opacity=0.5,
                stroke_width=1,
            )
        )

        self.add(axes, axes.z_axis)
        self.add(plane)

        # Graph
        surface = axes.get_graph(lambda x, y: self.f(x) * self.g(y))
        surface.always_sort_to_camera(self.camera)

        surface_mesh = SurfaceMesh(surface, resolution=(21, 21))
        surface_mesh.set_stroke(WHITE, 0.5, 0.5)

        func_name = Tex(R"f(x) \cdot g(y)")
        func_name.to_corner(UL)
        func_name.fix_in_frame()

        self.add(surface)
        self.add(surface_mesh)
        self.add(func_name)

        # Slicer
        t_tracker = ValueTracker(0.5)
        slice_shadow = self.get_slice_shadow(t_tracker)
        slice_graph = self.get_slice_graph(t_tracker)

        equation = VGroup(MTex("x + y = "), DecimalNumber(color=YELLOW))
        equation[1].next_to(equation[0][-1], RIGHT, buff=0.2)
        equation.to_corner(UR)
        equation.fix_in_frame()
        equation[1].add_updater(lambda m: m.set_value(t_tracker.get_value()))

        set_label = MTex(R"\{(x, t - x): x \in \mathds{R}\}", tex_to_color_map={"t": YELLOW}, font_size=30)
        set_label.next_to(equation, DOWN, MED_LARGE_BUFF, aligned_edge=RIGHT)
        set_label.fix_in_frame()

        self.play(frame.animate.reorient(20, 70), run_time=5)
        self.wait()
        self.play(frame.animate.reorient(0, 0))
        self.wait()

        self.add(slice_shadow, slice_graph, axes.z_axis, axes.axis_labels, plane)
        self.play(
            FadeIn(slice_shadow),
            ShowCreation(slice_graph),
            Write(equation),
            FadeOut(surface_mesh),
            FadeOut(axes.z_axis),
        )
        self.wait()
        self.play(
            FadeIn(set_label, 0.5 * DOWN),
            MoveAlongPath(GlowDot(), slice_graph, run_time=5, remover=True)
        )
        self.wait()
        self.play(frame.animate.reorient(114, 75), run_time=3)
        self.wait()

        # Change t  (Fade out surface mesh?)
        def change_t_anims(t):
            return [
                t_tracker.animate.set_value(t),
                UpdateFromFunc(slice_shadow, lambda m: m.become(self.get_slice_shadow(t_tracker))),
                UpdateFromFunc(slice_graph, lambda m: m.become(self.get_slice_graph(t_tracker))),
            ]

        self.play(*change_t_anims(1.5), run_time=4)
        self.wait()
        self.play(
            *change_t_anims(-0.5),
            frame.animate.reorient(140, 50).set_anim_args(time_span=(0, 4)),
            run_time=8
        )
        self.wait()
        self.play(*change_t_anims(1.0), frame.animate.reorient(99, 77), run_time=8)
        self.wait()

    def get_slice_shadow(self, t_tracker, u_max=5.0, v_range=(-4.0, 4.0)):
        xu = self.axes.x_axis.unit_size
        yu = self.axes.y_axis.unit_size
        zu = self.axes.z_axis.unit_size
        x0, y0, z0 = self.axes.get_origin()
        t = t_tracker.get_value()

        return Surface(
            uv_func=lambda u, v: [
                xu * (u - v) / 2 + x0,
                yu * (u + v) / 2 + y0,
                zu * self.f((u - v) / 2) * self.g((u + v) / 2) + z0 + 2e-2
            ],
            u_range=(t, t + u_max),
            v_range=v_range,
            resolution=(201, 201),
            color=BLACK,
            opacity=1,
            gloss=0,
            reflectiveness=0,
            shadow=0,
        )

    def get_slice_graph(self, t_tracker, color=WHITE, stroke_width=4):
        t = t_tracker.get_value()
        x_min, x_max = self.axes.x_range[:2]
        y_min, y_max = self.axes.y_range[:2]

        if t > 0:
            x_range = (t - y_max, x_max)
        else:
            x_range = (x_min, t - y_min)

        return ParametricCurve(
            lambda x: self.axes.c2p(x, t - x, self.f(x) * self.g(t - x)),
            x_range,
            stroke_color=color,
            stroke_width=stroke_width,
            fill_color=TEAL_D,
            fill_opacity=0.5,
        )


class RepeatedConvolution(MovingAverageAsConvolution):
    resolution = 0.01
    n_iterations = 12

    def construct(self):
        # Clean the board
        dx = self.resolution
        axes1, axes2, axes3, conv_axes = self.all_axes
        conv_axes.y_axis.stretch(1.5 / 2.0, 1)
        g_graph = self.g_graph

        x_min, x_max = axes1.x_range[:2]
        x_samples = np.arange(x_min, x_max + dx, dx)
        f_samples = np.array([self.f(x) for x in x_samples])
        g_samples = np.array([self.g(x) for x in x_samples])

        self.remove(self.f_graph)
        self.remove(self.prod_graphs)
        self.remove(self.conv_graph)
        self.remove(self.conv_graph_dot)
        self.remove(self.conv_graph_line)
        for axes in self.all_axes[:3]:
            axes.x_label.set_opacity(0)

        # New f graph
        f_graph = g_graph.deepcopy()
        f_graph.clear_updaters()
        f_graph.set_stroke(BLUE)
        f_graph.shift(axes1.get_origin() - axes2.get_origin())

        self.add(f_graph)

        # New prod graph
        t_indicator = self.t_indicator

        def get_t():
            return axes2.x_axis.p2n(t_indicator.get_center())

        def set_t(t):
            return t_indicator.animate.set_x(axes2.c2p(t)[0])

        def update_prod_graph(prod_graph):
            prod_samples = f_samples.copy()
            t = get_t()
            prod_samples[x_samples < t - 0.5] = 0
            prod_samples[x_samples > t + 0.5] = 0
            prod_graph.set_points_as_corners(
                axes3.c2p(x_samples, prod_samples)
            )

        prod_graph = VMobject()
        prod_graph.set_stroke(GREEN, 2)
        prod_graph.set_fill(BLUE_E, 1)
        prod_graph.add_updater(update_prod_graph)

        self.add(prod_graph)
        self.add(self.fg_label)

        # Convolution
        conv_samples, conv_graph = self.get_conv(
            x_samples, f_samples, g_samples, conv_axes
        )
        endpoint_dot = GlowDot(color=WHITE)
        endpoint_dot.add_updater(lambda m: m.move_to(conv_graph.get_points()[-1]))

        self.add(conv_graph)

        # Show new convolutions
        for n in range(self.n_iterations):
            t_indicator.set_x(axes2.c2p(-3, 0)[0])
            self.play(
                set_t(3),
                ShowCreation(conv_graph),
                UpdateFromAlphaFunc(
                    endpoint_dot, lambda m, a: m.set_opacity(a),
                    time_span=(0, 0.5),
                ),
                run_time=5,
                rate_func=bezier([0, 0, 1, 1])
            )
            self.play(FadeOut(endpoint_dot))
            shift_value = axes1.get_origin() - conv_axes.get_origin()
            cg_anim = conv_graph.animate.stretch(1 / 1.5, 1, about_point=conv_axes.get_origin())
            cg_anim.shift(shift_value)
            cg_anim.match_style(f_graph)
            self.play(
                cg_anim,
                FadeOut(f_graph, shift_value),
                FadeOut(axes1, shift_value),
                Transform(conv_axes.deepcopy(), axes1, remover=True)
            )
            self.add(axes1, conv_graph)

            f_samples[:] = conv_samples
            f_graph = conv_graph
            conv_samples, conv_graph = self.get_conv(
                x_samples, f_samples, g_samples, conv_axes
            )

    def get_conv(self, x_samples, f_samples, g_samples, axes):
        """
        Returns array of samples and graph
        """
        conv_samples = self.resolution * scipy.signal.fftconvolve(
            f_samples, g_samples, mode='same'
        )
        conv_graph = VMobject().set_points_as_corners(
            axes.c2p(x_samples, conv_samples)
        )
        conv_graph.set_stroke(TEAL, 2)
        return conv_samples, conv_graph

    def f(self, x):
        return rect_func(x)


# Final
class FunctionAverage(InteractiveScene):
    def construct(self):
        # Axes and graph
        def f(x):
            return 0.5 * np.exp(-0.8 * x**2) * (0.5 * x**3 - 3 * x + 1)


# Old rect material


class MovingAverageOfRectFuncs(Convolutions):
    f_graph_x_step = 0.01
    g_graph_x_step = 0.01
    jagged_product = True

    def construct(self):
        super().construct()
        t_indicator = self.t_indicator
        g_axes = self.all_axes[1]
        self.all_axes[3].y_axis.match_height(g_axes.y_axis)
        self.conv_graph.set_height(0.5 * g_axes.y_axis.get_height(), about_edge=DOWN, stretch=True)

        for t in [3, -3, 0]:
            self.play(t_indicator.animate.set_x(g_axes.c2p(t, 0)[0]), run_time=5)
        self.wait()

    def f(self, x):
        return rect_func(x / 2)

    def g(self, x):
        return 1.5 * rect_func(1.5 * x)


class RectConvolutionsNewNotation(MovingAverages):
    def construct(self):
        # Setup axes
        x_min, x_max = -1.0, 1.0
        all_axes = axes1, axes2, axes3 = VGroup(*(
            Axes(
                (x_min, x_max, 0.5), (0, 5),
                width=3.75, height=4
            )
            for x in range(3)
        ))
        all_axes.arrange(RIGHT, buff=LARGE_BUFF, aligned_edge=DOWN)
        for axes in all_axes:
            axes.x_axis.add_numbers(font_size=12, num_decimal_places=1)
        axes2.y_axis.add_numbers(font_size=12, num_decimal_places=0, direction=DL, buff=0.05)
        all_axes.move_to(DOWN)

        self.add(all_axes)

        # Prepare convolution graphs
        dx = 0.01
        xs = np.arange(x_min, x_max + dx, dx)
        k_range = list(range(3, 9, 2))
        conv_graphs = self.get_all_convolution_graphs(xs, rect_func(xs), axes3, k_range)
        VGroup(*conv_graphs).set_stroke(TEAL, 3)

        rect_defs = VGroup(
            self.get_rect_func_def(),
            *(self.get_rect_k_def(k) for k in k_range)
        )
        rect_defs.scale(0.75)
        rect_defs.next_to(axes2, UP)
        rect_defs[0][9:].scale(0.7, about_edge=LEFT)
        rect_defs[0].next_to(axes1, UP).shift_onto_screen()

        conv_labels = VGroup(
            Tex(R"\big[\text{rect} * \text{rect}_3\big](x)"),
            Tex(R"\big[\text{rect} * \text{rect}_3 * \text{rect}_5\big](x)"),
            Tex(R"\big[\text{rect} * \text{rect}_3 * \text{rect}_5 * \text{rect}_7 \big](x)"),
        )
        conv_labels.scale(0.75)
        conv_labels.match_x(axes3).match_y(rect_defs)

        # Show rect_1 * rect_3
        rect_graphs = VGroup(*(
            self.get_rect_k_graph(axes2, k)
            for k in [1, *k_range]
        ))
        rect_graphs[0].set_color(BLUE)
        rect_graphs[0].match_x(axes1)

        rect = Rectangle(axes2.x_axis.unit_size / 3, axes2.y_axis.unit_size * 3)
        rect.set_stroke(width=0)
        rect.set_fill(YELLOW, 0.5)
        rect.move_to(axes2.get_origin(), DOWN)

        self.add(*rect_graphs[:2])
        self.add(*rect_defs[:2])
        self.add(conv_graphs[0])

        self.play(FadeIn(rect))
        self.wait()

        self.play(
            Transform(rect_defs[0][:4].copy(), conv_labels[0][0][1:5], remover=True, path_arc=-PI / 3),
            Transform(rect_defs[1][:5].copy(), conv_labels[0][0][6:11], remover=True, path_arc=-PI / 3),
            FadeIn(conv_labels[0][0], lag_ratio=0.1, time_span=(1.5, 2.5)),
            FadeOut(rect),
            run_time=2
        )
        self.wait()

        # Show the rest
        for n in range(2):
            left_graph = rect_graphs[n] if n == 0 else conv_graphs[n - 1]
            left_label = rect_defs[n] if n == 0 else conv_labels[n - 1]
            k = 2 * n + 5
            new_rect = Rectangle(axes2.x_axis.unit_size / k, axes2.y_axis.unit_size * k)
            new_rect.set_stroke(width=0)
            new_rect.set_fill(YELLOW, 0.5)
            new_rect.move_to(axes2.get_origin(), DOWN)
            self.play(
                FadeOut(left_graph, 1.5 * LEFT),
                FadeOut(left_label, 1.5 * LEFT),
                FadeOut(rect_defs[n + 1]),
                FadeOut(rect_graphs[n + 1]),
                conv_labels[n].animate.match_x(axes1),
                conv_graphs[n].animate.match_x(axes1),
            )
            self.play(
                Write(rect_defs[n + 2], stroke_color=WHITE),
                ShowCreation(rect_graphs[n + 2]),
                FadeIn(new_rect),
                run_time=1,
            )
            self.wait()
            left_conv = conv_labels[n][0][1:-4]
            r = len(left_conv) + 1
            self.play(
                Transform(left_conv.copy(), conv_labels[n + 1][0][1:r], remover=True, path_arc=-PI / 3),
                Transform(rect_defs[2][:5].copy(), conv_labels[n + 1][0][r + 1:r + 6], remover=True, path_arc=-PI / 3),
                FadeIn(conv_labels[n + 1][0], lag_ratio=0.1, time_span=(0.5, 1.5)),
                ShowCreation(conv_graphs[n + 1]),
            )
            self.play(FadeOut(new_rect))
            self.wait()

    def get_rect_k_graph(self, axes, k):
        x_range = axes.x_axis.x_range
        x_range[2] = 1 / k
        return axes.get_graph(
            lambda x: k * rect_func(k * x),
            discontinuities=(-1 / (2 * k), 1 / (2 * k)),
            stroke_color=YELLOW,
            stroke_width=3,
        )

    def get_rect_k_def(self, k):
        return Tex(Rf"\text{{rect}}_{{{k}}}(x) := {k} \cdot \text{{rect}}({k}x)")[0]


class RectConvolutionFacts(InteractiveScene):
    def construct(self):
        # Equations
        equations = VGroup(
            Tex(R"\text{rect}", "(0)", "=", "1.0"),
            Tex(
                R"\big[",
                R"\text{rect}", "*",
                R"\text{rect}_3",
                R"\big]", "(0)", "=", "1.0"
            ),
            Tex(
                R"\big[",
                R"\text{rect}", "*",
                R"\text{rect}_3", "*",
                R"\text{rect}_5",
                R"\big]", "(0)", "=", "1.0"
            ),
            Tex(R"\vdots"),
            Tex(
                R"\big[",
                R"\text{rect}", "*",
                R"\text{rect}_3", "*", R"\cdots", "*",
                R"\text{rect}_{13}",
                R"\big]", "(0)", "=", "1.0"
            ),
            Tex(
                R"\big[",
                R"\text{rect}", "*",
                R"\text{rect}_3", "*", R"\cdots", "*",
                R"\text{rect}_{13}", "*",
                R"\text{rect}_{15}",
                R"\big]", "(0)", "=", SUB_ONE_FACTOR + R"\dots"
            ),
        )

        for eq in equations:
            eq.set_color_by_tex(R"\text{rect}", BLUE)
            eq.set_color_by_tex("_3", TEAL)
            eq.set_color_by_tex("_5", GREEN)
            eq.set_color_by_tex("_{13}", YELLOW)
            eq.set_color_by_tex("_{15}", RED_B)

        equations.arrange(DOWN, buff=0.75, aligned_edge=RIGHT)
        equations[3].match_x(equations[2][-1])
        equations[-1][:-1].align_to(equations[-2][-2], RIGHT)
        equations[-1][-1].next_to(equations[-1][:-1], RIGHT)
        equations.set_width(FRAME_WIDTH - 4)
        equations.center()

        # Show all (largely copy pasted...)
        self.add(equations[0])
        for i in range(4):
            if i < 3:
                src = equations[i].copy()
            else:
                src = equations[i + 1].copy()

            if i < 2:
                target = equations[i + 1]
            elif i == 2:
                target = VGroup(*equations[i + 1], *equations[i + 2])
            else:
                target = equations[i + 2]
            self.play(TransformMatchingTex(src, target))
            self.wait(0.5)

        self.wait()
