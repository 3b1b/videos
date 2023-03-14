from manim_imports_ext import *
from _2022.convolutions.continuous import *
from _2023.clt.main import *

# Convolutions

class ConvolveDiscreteDistributions(InteractiveScene):
    long_form = True
    dist1 = EXP_DISTRIBUTION
    dist2 = np.array([1.0 / (x + 1)**1.2 for x in range(6)])

    def construct(self):
        # Set up two distributions
        dist1 = np.array(self.dist1)
        dist2 = np.array(self.dist2)
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
        PX = Tex("P_X", **plabel_kw)
        PY = Tex("P_Y", **plabel_kw)
        PXY = Tex("P_{X + Y}", **plabel_kw)

        PX.next_to(top_bars.get_corner(UR), DR)
        PY.next_to(low_bars.get_corner(UR), DR)
        PXY.next_to(conv_bars, UP, LARGE_BUFF)

        # Add distributions
        frame = self.frame
        frame.set_height(6).move_to(top_bars)

        self.play(
            self.show_bars_creation(top_bars, lag_ratio=0.05),
            Write(PX),
        )
        self.wait()
        self.play(
            self.show_bars_creation(low_bars, lag_ratio=0.1),
            Write(PY),
            FadeIn(v_lines, lag_ratio=0.2),
            frame.animate.set_height(FRAME_HEIGHT).set_y(0).set_anim_args(run_time=1),
        )
        self.wait()

        self.play(
            self.show_bars_creation(conv_bars),
            FadeTransform(PX.copy(), PXY),
            FadeTransform(PY.copy(), PXY),
            frame.animate.center().set_anim_args(run_time=1)
        )
        self.wait()

        # Flip
        low_bars.target = low_bars.generate_target()
        low_bars.target.arrange(LEFT, aligned_edge=DOWN, buff=0).move_to(low_bars)
        low_bars.target.move_to(low_bars)
        self.play(
            MoveToTarget(low_bars, path_arc=PI / 3, lag_ratio=0.005)
        )
        self.wait()

        # March!
        last_pair_rects = VGroup()
        for n in [7, *range(2, 13)]:
            conv_bars.generate_target()
            conv_bars.target.set_opacity(0.35)
            conv_bars.target[n - 2].set_opacity(1.0)

            self.play(
                get_row_shift(top_bars, low_bars, n),
                MaintainPositionRelativeTo(PY, low_bars),
                FadeOut(last_pair_rects),
                MoveToTarget(conv_bars),
            )
            pairs = get_aligned_pairs(top_bars, low_bars, n)

            label_pairs = VGroup(*(VGroup(m1.value_label, m2.value_label) for m1, m2 in pairs))
            die_pairs = VGroup(*(VGroup(m1.die, m2.die) for m1, m2 in pairs))
            pair_rects = VGroup(*(
                SurroundingRectangle(pair, buff=0.05).set_stroke(YELLOW, 2).round_corners()
                for pair in pairs
            ))
            pair_rects.set_stroke(YELLOW, 2)
            for rect in pair_rects:
                rect.set_width(label_pairs[0].get_width() + 0.125, stretch=True)

            self.play(FadeIn(pair_rects, lag_ratio=0.5))

            fade_anims = []
            if self.long_form:
                # Spell out the full dot product
                products = Tex(R"P(O) \cdot P(O)", isolate="O", font_size=36).replicate(len(pairs))
                products.arrange(DOWN, buff=MED_LARGE_BUFF)
                products.next_to(conv_bars, LEFT, MED_LARGE_BUFF)
                products.to_edge(UP, buff=LARGE_BUFF)
                plusses = Tex("+", font_size=36).replicate(len(pairs) - 1)
                for plus, lp1, lp2 in zip(plusses, products, products[1:]):
                    plus.move_to(VGroup(lp1, lp2))

                die_targets = die_pairs.copy()
                for dp, dt_pair, product in zip(die_pairs, die_targets, products):
                    for die, O in zip(dt_pair, product.select_parts("O")):
                        die.match_width(O)
                        die.move_to(O)
                        O.set_opacity(0)
                    product.save_state()
                    product[:len(product) // 2].replace(dp[0])
                    product[len(product) // 2:].replace(dp[1])
                    product.set_opacity(0)

                self.play(
                    LaggedStart(*(
                        TransformFromCopy(dp, dt)
                        for dp, dt in zip(die_pairs, die_targets)
                    ), lag_ratio=0.5),
                    LaggedStartMap(Restore, products, lag_ratio=0.5),
                    LaggedStartMap(Write, plusses, lag_ratio=0.5),
                    run_time=3
                )
                self.wait()
                prod_group = VGroup(*products, *die_targets, *plusses)
                prod_group.generate_target()
                prod_group.target.set_opacity(0)
                for mob in prod_group.target:
                    mob.replace(conv_bars[n - 2].value_label, stretch=True)
                self.play(MoveToTarget(prod_group, remover=True))
                self.wait()
            else:
                self.play(
                    *(
                        FadeTransform(label.copy(), conv_bars[n - 2].value_label)
                        for lp in label_pairs
                        for label in lp
                    ),
                    *fade_anims,
                )
                self.wait(0.5)

            last_pair_rects = pair_rects

        conv_bars.target.set_opacity(1.0)
        self.play(
            FadeOut(last_pair_rects),
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
            Tex(
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
        conv_def = Tex(
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

    def show_bars_creation(self, bars, lag_ratio=0.05, run_time=3):
        anims = []
        for bar in bars:
            rect, num, face = bar
            num.rect = rect
            rect.save_state()
            rect.stretch(0, 1, about_edge=DOWN)
            rect.set_opacity(0)

            anims.extend([
                FadeIn(face),
                rect.animate.restore(),
                CountInFrom(num, 0),
                UpdateFromAlphaFunc(num, lambda m, a: m.next_to(m.rect, UP, SMALL_BUFF).set_opacity(a)),
            ])

        return LaggedStart(*anims, lag_ratio=lag_ratio, run_time=run_time)


class ConvolveMatchingDiscreteDistributions(ConvolveDiscreteDistributions):
    dist1 = EXP_DISTRIBUTION
    dist2 = EXP_DISTRIBUTION


class RepeatedDiscreteConvolutions(InteractiveScene):
    distribution = EXP_DISTRIBUTION

    def construct(self):
        # Divide up space
        h_lines = Line(LEFT, RIGHT).set_width(FRAME_WIDTH).replicate(4)
        h_lines.arrange(DOWN, buff=FRAME_HEIGHT / 3).center()
        h_lines.set_stroke(WHITE, 1)
        self.add(h_lines[1:3])

        # Initial distributions
        dist = self.distribution
        top_bars = self.get_bar_group(dist, colors=(BLUE, TEAL))
        top_bars.next_to(h_lines[1], UP, SMALL_BUFF)

        low_bars = top_bars.copy()
        low_bars.set_y(-top_bars.get_y())
        low_bars.next_to(h_lines[2], UP, SMALL_BUFF)

        VGroup(top_bars, low_bars).shift(2 * LEFT)

        self.add(top_bars)
        self.add(low_bars)

        # Add labels

        # Repeated convolution
        self.flip_bar_group(low_bars)
        low_bars.save_state()
        for n in range(5):
            new_bars = self.show_convolution(top_bars, low_bars)
            self.wait()
            self.play(
                new_bars.animate.move_to(top_bars, DL).set_anim_args(path_arc=-120 * DEGREES),
                FadeOut(top_bars, UP),
                Restore(low_bars),
            )
            # TODO, things with labels

            top_bars = new_bars

    def get_bar_group(
        self, 
        dist, 
        colors=(BLUE, TEAL), 
        y_unit=4,
        bar_width=0.35,
        num_decimal_places=2,
        min_value=1,
    ):
        bars = self.get_bars(dist, colors, y_unit, bar_width)
        result = VGroup(
            bars,
            self.get_bar_value_labels(bars, min_value),
            self.get_bar_prob_labels(bars, dist, num_decimal_places),
        )
        result.dist = dist
        return result

    def get_bars(self, dist, colors=(BLUE, TEAL), y_unit=4, bar_width=0.35):
        axes = Axes(
            (0, len(dist)), (0, 1),
            height=y_unit,
            width=bar_width * len(dist)
        )
        bars = ChartBars(axes, dist, fill_opacity=0.75)
        bars.set_submobject_colors_by_gradient(*colors)
        return bars

    def get_bar_value_labels(self, bars, min_value=1):
        values = VGroup(*(
            Integer(x + min_value, font_size=16)
            for x in range(len(bars))
        ))
        for bar, value in zip(bars, values):
            value.next_to(bar, DOWN, SMALL_BUFF)

        return values

    def get_bar_prob_labels(self, bars, dist, num_decimal_places=2):
        probs = VGroup(*(
            DecimalNumber(p, font_size=16, num_decimal_places=num_decimal_places)
            for p in dist
        ))
        for bar, prob in zip(bars, probs):
            prob.set_max_width(0.75 * bar.get_width())
            prob.next_to(bar, UP, SMALL_BUFF)

        return probs

    def get_dist_label(self, indices):
        index_strs = [f"X_{{{i}}}" for i in indices]
        if len(indices) > 3:
            index_strs = [index_strs[0], R"\cdots", index_strs[-1]]
        sub_tex = "+".join(index_strs)
        return Tex(f"P_{{{sub_tex}}}")

    def flip_bar_group(self, bar_group):
        bars = bar_group[0]
        bars.target = bars.generate_target()
        bars.target.arrange(LEFT, buff=0, aligned_edge=DOWN)
        bars.target.align_to(bars[0], DR)
        self.play(
            MoveToTarget(bars, lag_ratio=0.05, path_arc=0.5),
            *(
                MaintainPositionRelativeTo(
                    VGroup(value, prob), bar
                )
                for bar, value, prob in zip(*bar_group)
            ),
        )
        self.add(bar_group)

    def show_convolution(self, top_bars, low_bars):
        # New bars
        new_dist = np.convolve(top_bars.dist, low_bars.dist)
        new_bars = self.get_bar_group(
            new_dist,
            y_unit=8,
            num_decimal_places=3,
            min_value=top_bars[1][0].get_value() + low_bars[1][0].get_value(),
        )
        new_bars.next_to(BOTTOM, UP)
        new_bars.align_to(top_bars, LEFT)

        # March!
        for n in range(len(new_bars[0])):
            x_diff = top_bars[0][0].get_x() - low_bars[0][0].get_x()
            x_diff += low_bars[0][0].get_width() * n
            self.play(
                low_bars.animate.shift(x_diff * RIGHT),
                run_time=0.5
            )
            index_pairs = [
                (k, n - k) for k in range(n + 1)
                if 0 <= n - k < len(low_bars[0])
                if 0 <= k < len(top_bars[0])
            ]
            highlights = VGroup(*(
                VGroup(top_bars[0][i].copy(), low_bars[0][j].copy())
                for i, j in index_pairs
            ))
            highlights.set_color(YELLOW)

            conv_rect, value_label, prob_label = (group[n] for group in new_bars)
            partial_rects = VGroup()
            partial_labels = VGroup()

            products = [top_bars.dist[i] * low_bars.dist[j] for i, j in index_pairs]
            for partial_value in np.cumsum(products):
                rect = conv_rect.copy()
                rect.stretch(
                    partial_value / new_bars.dist[n],
                    dim=1,
                    about_edge=DOWN,
                )
                label = prob_label.copy()
                label.set_value(partial_value)
                label.next_to(rect, UP, SMALL_BUFF)
                partial_rects.add(rect)
                partial_labels.add(label)

            self.add(value_label)
            self.play(
                ShowSubmobjectsOneByOne(highlights, remover=True),
                ShowSubmobjectsOneByOne(partial_rects, remover=True),
                ShowSubmobjectsOneByOne(partial_labels, remover=True),
                run_time=0.15 * len(products)
            )
            self.add(*(group[:n + 1] for group in new_bars))
            self.wait(0.5)

        return new_bars


class TransitionToContinuousProbability(InteractiveScene):
    def construct(self):
        # Setup axes and initial graph
        axes = Axes((0, 12), (0, 1, 0.2), width=14, height=5)
        axes.to_edge(LEFT, LARGE_BUFF)
        axes.to_edge(DOWN, buff=1.25)

        def pd(x):
            return (x**4) * np.exp(-x) / 8.0

        graph = axes.get_graph(pd)
        graph.set_stroke(WHITE, 2)
        bars = axes.get_riemann_rectangles(graph, dx=1, x_range=(0, 6), input_sample_type="right")
        bars.set_stroke(WHITE, 3)

        y_label = Text("Probability", font_size=48)
        y_label.next_to(axes.y_axis, UP, SMALL_BUFF)
        y_label.shift_onto_screen()

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
        x_labels = VGroup(*(
            Text(text, font_size=36)
            for text in [
                "Height of a randomly chosen tree",
                "Value of XYZ next year",
                "Wavelength of a randomly chosen solar photon",
            ]
        ))
        for x_label in x_labels:
            x_label.next_to(axes.c2p(4, 0), DOWN, buff=0.45)

        density = Text("Probability density")
        density.match_height(y_label)
        density.move_to(y_label, LEFT)
        cross = Cross(y_label)
        cross.set_stroke(RED, width=(0, 8, 8, 8, 0))

        self.play(Write(x_labels[0]))
        for xl1, xl2 in zip(x_labels, x_labels[1:]):
            self.wait()
            self.play(
                FadeOut(xl1, 0.5 * UP),
                FadeIn(xl2, 0.5 * UP),
            )
        self.wait()
        self.play(ShowCreation(cross))
        self.play(
            VGroup(y_label, cross).animate.shift(0.75 * UP),
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

        # Total area label
        label = Text("Total area = 1")
        label.next_to(graph.get_top(), UL).shift(RIGHT)
        arrow = Arrow(label.get_bottom(), axes.c2p(4, 0.2))

        self.play(Write(label), ShowCreation(arrow))
        self.wait()


class GaussConvolutions(Convolutions):
    conv_y_stretch_factor = 1.0

    def construct(self):
        super().construct()

    def f(self, x):
        return np.exp(-x**2)

    def g(self, x):
        return np.exp(-x**2)
