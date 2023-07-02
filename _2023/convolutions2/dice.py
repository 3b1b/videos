from manim_imports_ext import *
from _2022.convolutions.discrete import *
from _2023.clt.main import *


SKEW_DISTRIBUTION = [0.12, 0.23, 0.31, 0.18, 0.12, 0.04]

# Helpers


def get_bar_group(
    dist,
    bar_colors=(BLUE_D, TEAL_D),
    value_labels=None,
    width_ratio=0.7,
    height=2.0,
    number_config=dict(),
    label_buff=SMALL_BUFF,
):
    bars = dist_to_bars(dist, bar_colors=bar_colors, height=height)
    p_labels = VGroup(*(DecimalNumber(x, **number_config) for x in dist))
    p_labels.set_max_width(width_ratio * bars[0].get_width())
    for p_label, bar in zip(p_labels, bars):
        p_label.next_to(bar, UP, SMALL_BUFF)

    if value_labels is None:
        value_labels = VectorizedPoint().replicate(len(dist))

    for value_label, bar in zip(value_labels, bars):
        value_label.set_width(width_ratio * bars[0].get_width())
        value_label.next_to(bar, DOWN, buff=label_buff)

    labeled_bars = VGroup(*(
        VGroup(bar, value_label, p_label)
        for bar, value_label, p_label in zip(bars, value_labels, p_labels)
    ))
    for group in labeled_bars:
        group.bar, group.die, group.value_label = group

    return labeled_bars


def die_sum_labels(color1=BLUE_E, color2=RED_E, height=1.0):
    die1, die2 = dice = [
        DieFace(1, fill_color=color)
        for color in [color1, color2]
    ]
    for die in dice:
        die.remove(die[1])
        die.set_height(height / 3)

    result = VGroup()
    for n in range(2, 13):
        sum_sym = VGroup(
            die1.copy(),
            Tex("+", font_size=24),
            die2.copy(),
            Tex("=", font_size=24).rotate(90 * DEGREES),
            Tex(str(n), font_size=30),
        )
        sum_sym.arrange(DOWN, buff=SMALL_BUFF)
        sum_sym[:2].shift(0.05 * DOWN)
        sum_sym[:1].shift(0.05 * DOWN)
        sum_sym.set_height(height)
        result.add(sum_sym)
    result.arrange(RIGHT)
    return result


def rotate_sum_label(sum_label):
    sum_label.arrange(RIGHT, buff=SMALL_BUFF)
    sum_label[-2].rotate(90 * DEGREES)
    sum_label[-2:].set_height(sum_label.get_height(), about_edge=LEFT)
    sum_label[-2:].shift(SMALL_BUFF * RIGHT)
    sum_label[-1].shift(SMALL_BUFF * RIGHT)


def p_mob(mob, scale_factor=1.0):
    used_mob = mob.copy()
    aspect_ratio = mob.get_width() / mob.get_height()
    Os = "O" * int(np.round(aspect_ratio))
    tex = Tex(f"P({Os})")
    used_mob.replace(tex[Os], dim_to_match=0)
    used_mob.scale(scale_factor)
    result = VGroup(*tex[:2], used_mob, tex[-1])
    result.arg = used_mob
    return result


# Scenes


class SumAlongDiagonal(InteractiveScene):
    samples = 4

    dist1 = EXP_DISTRIBUTION
    dist2 = SKEW_DISTRIBUTION

    dist1_colors = (BLUE_D, TEAL_D)
    dist2_colors = (RED_D, GOLD_E)
    sum_colors = (GREEN_E, YELLOW_E)

    def construct(self):
        # Setup distributions
        dist1 = self.dist1
        dist2 = self.dist2
        blue_dice = get_die_faces(fill_color=BLUE_E, dot_color=WHITE)
        red_dice = get_die_faces(fill_color=RED_E, dot_color=WHITE)
        bar_groups = VGroup(
            get_bar_group(dist1, self.dist1_colors, blue_dice),
            get_bar_group(dist2, self.dist2_colors, red_dice),
        )

        bar_groups.arrange(DOWN, buff=LARGE_BUFF)
        bar_groups.to_edge(LEFT)

        self.add(bar_groups)

        # Setup the sum distribution
        conv_dist = np.convolve(dist1, dist2)
        sum_labels = die_sum_labels()
        sum_bar_group = get_bar_group(
            conv_dist, self.sum_colors, sum_labels,
            number_config=dict(num_decimal_places=3, font_size=30),
            label_buff=MED_SMALL_BUFF,
        )
        sum_bar_group.to_edge(RIGHT, buff=LARGE_BUFF)
        sum_bar_group.set_y(0)

        buckets = VGroup()
        for bar in sum_bar_group:
            base = Line(LEFT, RIGHT)
            base.match_width(bar)
            base.move_to(bar[0], DOWN)
            v_lines = Line(DOWN, UP).replicate(2)
            v_lines.set_height(6)
            v_lines[0].move_to(base.get_left(), DOWN)
            v_lines[1].move_to(base.get_right(), DOWN)
            bucket = VGroup(base, *v_lines)
            bucket.set_stroke(GREY_C, 2)
            buckets.add(bucket)

        self.add(sum_labels)
        self.add(buckets)

        # Repeatedly sample from these two (for a while)
        self.show_repeated_samples(
            dist1, dist2, *bar_groups, buckets, sum_labels,
            n_animated_runs=1, n_total_runs=2,
        )

        # Ask about sum values
        rects = VGroup(*(
            SurroundingRectangle(sum_label)
            for sum_label in sum_labels
        ))
        words1 = Text("What's the probability\nof this?", font_size=36)
        words2 = Text("Or this?", font_size=36)

        words1.next_to(rects[0], DOWN, MED_SMALL_BUFF)
        words2.next_to(rects[1], DOWN, MED_SMALL_BUFF)

        self.play(ShowCreation(rects[0]), FadeIn(words1))
        self.wait()
        self.play(
            TransformMatchingStrings(words1, words2, run_time=1),
            FadeOut(rects[0]),
            FadeIn(rects[1]),
        )
        self.wait()
        for i in range(1, len(rects) - 1):
            self.play(
                FadeOut(rects[i]), FadeIn(rects[i + 1]),
                words2.animate.match_x(rects[i + 1]),
                run_time=0.5
            )
        self.wait()
        self.play(FadeOut(words2), FadeOut(rects[-1]))
        self.play(FadeOut(sum_labels), FadeOut(buckets))

        # Draw grid of dice values
        grid = Square().get_grid(6, 6, buff=0, fill_rows_first=False, )
        grid.flip(RIGHT)
        grid.set_stroke(WHITE, 1)
        grid.set_height(5.5)
        grid.to_edge(RIGHT, buff=LARGE_BUFF)
        grid.to_edge(UP)

        blue_row = blue_dice.copy()
        red_col = red_dice.copy()
        for square, die in zip(grid[::6], blue_row):
            die.set_width(0.5 * square.get_width())
            die.next_to(square, DOWN)
        for square, die in zip(grid, red_col):
            die.set_width(0.5 * square.get_width())
            die.next_to(square, LEFT)

        self.play(
            ShowCreation(grid, lag_ratio=0.5),
            TransformFromCopy(blue_dice, blue_row),
            TransformFromCopy(red_dice, red_col),
        )

        dice_pairs = VGroup()
        anims = []
        for n, square in enumerate(grid):
            templates = VGroup(
                blue_row[n // 6],
                red_col[n % 6],
            )
            pair = templates.copy()
            pair.arrange(RIGHT, buff=SMALL_BUFF)
            pair.set_width(square.get_width() * 0.7)
            pair.move_to(square)
            dice_pairs.add(pair)
            anims.extend([
                TransformFromCopy(templates, pair),
            ])

        self.play(LaggedStart(*anims, lag_ratio=0.1))
        self.add(dice_pairs)
        self.wait()

        full_table = VGroup(blue_row, red_col, grid, dice_pairs)

        # Highlight (4, 2) pair
        pairs = [
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 5),
            (2, 4),
            (2, 3),
            (3, 3),
            (3, 2),
            (4, 2),
        ]

        for pair in pairs:
            self.isolate_pairs(bar_groups, full_table, pair)
            self.wait(0.5)
        self.wait()

        # Show the probability of (4, 2)
        def get_p_label(dice):
            p_label = VGroup(
                p_mob(dice), Tex("="),
                p_mob(dice[0]),
                p_mob(dice[1]),
            )
            p_label.arrange(RIGHT, buff=SMALL_BUFF)
            return p_label

        i0, j0 = pairs[-1]
        p_label = get_p_label(dice_pairs[(i0 - 1) * 6 + (j0 - 1)])
        p_label.to_edge(UP)
        p_label.shift(2.5 * LEFT)

        movers = VGroup(
            dice_pairs[(i0 - 1) * 6 + j0 - 1],
            blue_row[(i0 - 1)],
            red_col[(j0 - 1)],
        ).copy()
        self.play(
            bar_groups.animate.set_width(2.5, about_edge=DL),
            full_table.animate.set_width(5.5, about_edge=DR),
            LaggedStart(
                movers[0].animate.replace(p_label[0][2]),
                movers[1].animate.replace(p_label[2][2]),
                movers[2].animate.replace(p_label[3][2]),
                lag_ratio=0.25,
            ),
        )
        self.play(FadeIn(p_label))
        self.remove(movers)
        self.wait()

        # Show numerical product
        prod_rhs = Tex("= (0.00)(0.00)")
        num_rhs = Tex("= 0.000")

        value1, value2 = prod_rhs.make_number_changable("0.00", replace_all=True)
        value1.set_value(dist1[i0 - 1]).set_color(BLUE)
        value2.set_value(dist2[j0 - 1]).set_color(RED)
        prod_rhs.next_to(p_label[1], DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        pair_prob = num_rhs.make_number_changable("0.000")
        pair_prob.set_value(dist1[i0 - 1] * dist2[j0 - 1])
        num_rhs.next_to(prod_rhs, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        self.play(LaggedStart(
            Transform(bar_groups[0][i0 - 1][2].copy(), value1.copy(), remover=True),
            Transform(bar_groups[1][j0 - 1][2].copy(), value2.copy(), remover=True),
            Write(prod_rhs, lag_ratio=0.1),
            lag_ratio=0.5
        ))
        self.wait()
        self.play(FadeIn(num_rhs, DOWN))
        self.wait()

        example = VGroup(p_label, prod_rhs, num_rhs)

        # Assumption
        morty = Mortimer(height=2.0).flip()
        morty.next_to(example, DOWN, buff=2.0)
        morty.shift(0.5 * LEFT)

        self.play(
            morty.says(
                "Assuming rolls\nare independent!",
                mode="surprised",
                max_bubble_width=3.5
            ),
            VFadeIn(morty),
        )
        self.play(Blink(morty))
        self.wait()
        self.play(LaggedStartMap(FadeOut, VGroup(morty, morty.bubble, morty.bubble.content)))

        # Set up the multiplication table
        full_table.generate_target()
        full_table.target.set_width(6.5)
        full_table.target.set_opacity(1)
        full_table.target[2].set_fill(opacity=0)
        full_table.target.to_edge(RIGHT, buff=1.5)
        full_table.target.set_y(0)

        self.play(
            MoveToTarget(full_table),
            example.animate.scale(0.75, about_edge=UL).shift(0.5 * LEFT)
        )

        marginal_labels = VGroup(*(
            VGroup(*(bar[2] for bar in group)).copy()
            for group in bar_groups
        ))
        margin_dice = VGroup(*full_table[:2], full_table[3])
        for margins, dice in zip(marginal_labels, margin_dice):
            margins.generate_target()
            dice.generate_target()
            for die, prob in zip(dice.target, margins.target):
                center = die.get_center()
                die.scale(0.7)
                die.set_opacity(0.85)
                prob.next_to(die, DOWN, buff=0.125)
                prob.scale(1.5)
                VGroup(prob, die).move_to(center)

        marginal_labels[0].target.set_fill(BLUE, 1)
        marginal_labels[1].target.set_fill(RED, 1)

        margin_dice[2].generate_target()
        for dice in margin_dice[2].target:
            dice.scale(0.75, about_edge=UP)
            dice.set_opacity(0.85)
            dice.set_stroke(width=1)
            dice.shift(0.1 * UP)

        for groups in zip(marginal_labels, margin_dice):
            self.play(*map(MoveToTarget, groups), lag_ratio=0.001)
        self.play(MoveToTarget(margin_dice[2], lag_ratio=0.001))
        self.wait()

        full_table.add(marginal_labels)

        # Fill in multiplication table
        grid_probs = VGroup()
        self.add(grid_probs)
        for n, square in enumerate(grid):
            i, j = n // 6, n % 6
            dice = margin_dice[2][n]
            margin1 = marginal_labels[0][i]
            margin2 = marginal_labels[1][j]
            prob = DecimalNumber(
                margin1.get_value() * margin2.get_value(),
                num_decimal_places=3
            )
            prob.set_height(margin1.get_height())
            prob.next_to(dice, DOWN, SMALL_BUFF)

            rects = VGroup(*(
                SurroundingRectangle(mob, buff=SMALL_BUFF)
                for mob in [margin1, margin2, prob]
            ))
            rects.set_stroke(YELLOW, 1)

            grid_probs.add(prob)

            value1.set_value(dist1[i])
            value2.set_value(dist2[j])
            pair_prob.set_value(dist1[i] * dist2[j])
            new_p_label = get_p_label(dice.copy().set_opacity(1))
            new_p_label.replace(p_label, dim_to_match=1)
            p_label.become(new_p_label)

            bar_groups.set_opacity(0.35)
            bar_groups[0][i].set_opacity(1)
            bar_groups[1][j].set_opacity(1)

            self.add(rects)
            self.wait(0.25)
            self.remove(rects)
        self.wait()

        full_table.add(grid_probs)

        # Fade out example
        self.play(
            FadeOut(example),
            bar_groups.animate.set_opacity(1),
        )
        self.wait()

        # Show it as a 3d plot
        bar_groups.fix_in_frame()
        sum_bar_group.fix_in_frame()

        bars_3d = VGroup()
        scale_factor = 30
        for square, prob in zip(grid, grid_probs):
            prism = VCube()
            prism.set_fill(GREY_D, 0.85)
            prism.set_stroke(WHITE, 1, 0.5)
            prism.match_width(square)
            prism.set_depth(scale_factor * prob.get_value(), stretch=True)
            prism.move_to(square, IN)
            prism.save_state()
            prism.stretch(0.001, 2, about_edge=IN)
            prism.set_opacity(0)
            bars_3d.add(prism)

        self.play(
            LaggedStartMap(Restore, bars_3d),
            self.frame.animate.reorient(12, 65, 0).move_to([-0.48, 0.37, 0.77]).set_height(9.43),
            run_time=3,
        )
        self.add(full_table, *bars_3d)
        self.play(
            self.frame.animate.reorient(43, 66, 0).move_to([-0.06, 0.21, -0.29]).set_height(10.59),
            run_time=7,
        )
        self.wait()

        # Show sum distribution
        self.play(
            LaggedStartMap(FadeOut, bar_groups, shift=2 * LEFT),
            FadeIn(sum_bar_group),
            full_table.animate.to_edge(LEFT),
            self.frame.animate.reorient(4, 65, 0).move_to([0.64, 0.8, 0.69]).set_height(10.59),
            *(
                MaintainPositionRelativeTo(bar, full_table)
                for bar in bars_3d
            ),
            run_time=2,
        )
        self.wait()

        # Reposition
        self.play(
            self.frame.animate.to_default_state(),
            FadeOut(bars_3d),
            run_time=2,
        )
        self.wait()

        # Add up along all diagonals
        low_group = VGroup(blue_row, marginal_labels[0])
        left_group = VGroup(red_col, marginal_labels[1])
        diagonals = VGroup(*(
            VGroup(*(
                VGroup(dice_pairs[n], grid_probs[n])
                for n in range(36)
                if (n // 6) + (n % 6) == s
            ))
            for s in range(11)
        ))

        diagonals.save_state()
        rects = VGroup()
        diagonals.rotate(45 * DEGREES)
        for diagonal in diagonals:
            rect = SurroundingRectangle(diagonal)
            rect.stretch(0.9, 1)
            rect.round_corners()
            rects.add(rect)
        VGroup(rects, diagonals).rotate(-45 * DEGREES, about_point=diagonals.get_center())

        rects.set_fill(YELLOW, 0.25)
        rects.set_stroke(YELLOW, 2)

        last_rect = VGroup(low_group, left_group)
        for n, rect in zip(it.count(), rects):
            self.add(rect, diagonals)
            diagonals.generate_target()
            diagonals.target.set_opacity(0.5)
            diagonals.target[n].set_opacity(1)

            sum_bar_group.generate_target()
            sum_bar_group.target.set_opacity(0.4)
            sum_bar_group.target[n].set_opacity(1)

            self.play(
                FadeOut(last_rect),
                FadeIn(rect),
                MoveToTarget(diagonals),
                MoveToTarget(sum_bar_group),
            )
            last_rect = rect
        self.wait()
        self.play(
            diagonals.animate.set_opacity(1),
            sum_bar_group.animate.set_opacity(1),
            FadeOut(last_rect),
            FadeIn(low_group),
            FadeIn(left_group),
        )

        # Show 3d grid again
        self.play(
            self.frame.animate.reorient(27, 66, 0).move_to([-0.16, 1.41, 0.77]).set_height(9.36),
            FadeIn(bars_3d),
            sum_bar_group.animate.to_edge(RIGHT),
            run_time=2,
        )
        self.wait()

        # Go through diagonals of the plot
        sorted_bars = VGroup(*bars_3d)
        camera_pos = self.frame.get_implied_camera_location()
        sorted_bars.sort(lambda p: -get_norm(p - camera_pos))
        self.add(*sorted_bars)

        diagonal_bar_groups = VGroup().replicate(11)

        for s in range(11):
            sum_bar_group.generate_target()
            sum_bar_group.target.set_opacity(0.2)
            sum_bar_group.target[s].set_opacity(1)

            for n, bar in enumerate(bars_3d):
                bar.generate_target()
                bar.target.set_opacity(0.1)
                bar.target.set_stroke(width=0)
                if (n // 6) + (n % 6) == s:
                    bar.target.set_opacity(1)
                    diagonal_bar_groups[s].add(bar)

            self.play(
                MoveToTarget(sum_bar_group),
                *map(MoveToTarget, bars_3d),
            )
            self.wait()
        self.play(
            bars_3d.animate.set_opacity(0.8).set_stroke(width=0.5),
            sum_bar_group.animate.set_opacity(1),
        )
        self.wait()

        # Highlight bars
        bars_3d.save_state()
        bar_highlights = bars_3d.copy()
        bar_highlights.set_fill(opacity=0)
        bar_highlights.set_stroke(TEAL, 3)
        self.play(ShowCreationThenFadeOut(bar_highlights, lag_ratio=0.001, run_time=2))

        # Collapase diagonals
        bars_3d.generate_target()
        for bar in bars_3d.target:
            bar.stretch(0.5, 0)
            bar.stretch(0.5, 1)
        bars_3d.target.set_fill(opacity=1)
        bars_3d.target.set_submobject_colors_by_gradient(GREEN_D, YELLOW_D)
        bars_3d.target.set_stroke(WHITE, 1)
        self.play(
            self.frame.animate.reorient(36, 46, 0).move_to([-0.56, 0.55, 1.22]).set_height(7.71),
            MoveToTarget(bars_3d),
            # FadeOut(full_table, IN),
            sum_bar_group.animate.set_width(4.0).to_edge(RIGHT),
            run_time=2,
        )
        self.wait()

        diagonal_bar_groups.apply_depth_test()
        new_diagonals = diagonal_bar_groups.copy()
        for group in new_diagonals:
            group.arrange(IN, buff=0)
        new_diagonals.arrange(UR, buff=MED_SMALL_BUFF, aligned_edge=IN)
        new_diagonals.move_to(bars_3d.get_corner(DR))
        new_diagonals.shift(DR)
        
        self.play(
            self.frame.animate.reorient(40, 61, 0).move_to([1.69, 0.33, -0.73]).set_height(12.96),
            ReplacementTransform(
                diagonal_bar_groups, new_diagonals,
                lag_ratio=0.001,
            ),
            run_time=5,
        )
        self.add(full_table, new_diagonals)
        self.play(
            self.frame.animate.reorient(40, 85, 0).move_to([3.05, 1.93, 0.77]).set_height(14.93),
            sum_bar_group.animate.set_width(5.5, about_edge=RIGHT),
            FadeOut(full_table),
            run_time=3
        )
        self.wait()

    def show_repeated_samples(
        self,
        dist1,
        dist2,
        bar_group1,
        bar_group2,
        buckets,
        sum_labels,
        n_animated_runs=20,
        n_total_runs=150,
        marker_height=0.1,
        marker_color=YELLOW_D,
    ):
        marker_template = Rectangle(
            height=marker_height,
            width=buckets[0].get_width() * 0.8,
            fill_color=marker_color,
            fill_opacity=1,
            stroke_color=WHITE,
            stroke_width=1,
        )
        markers = VGroup(*(
            VGroup(VectorizedPoint(bucket.get_bottom()))
            for bucket in buckets
        ))

        var1 = scipy.stats.rv_discrete(values=(range(6), dist1))
        var2 = scipy.stats.rv_discrete(values=(range(6), dist2))

        for n in range(n_total_runs):
            x = var1.rvs()
            y = var2.rvs()

            animate = n < n_animated_runs

            # Show dice
            dice = VGroup()
            for group, value in [(bar_group1, x), (bar_group2, y)]:
                die = group[value][1].copy()
                die.set_opacity(1)
                die.scale(2)
                die.next_to(group, RIGHT, LARGE_BUFF)
                dice.add(die)
                group.set_opacity(0.5)
                group[value].set_opacity(1)
                self.add(die)
                self.wait(0.25 if animate else 0.0)

            # Highlight sum
            sum_labels.set_opacity(0.25)
            sum_labels[x + y].set_opacity(1)

            # Drop marker in the appropriate sum bucket
            marker = marker_template.copy()
            marker.move_to(markers[x + y].get_top(), DOWN)
            if animate:
                self.play(FadeIn(marker, DOWN, rate_func=rush_into, run_time=0.5))
                self.wait(0.5)

            markers[x + y].add(marker)
            self.add(markers)

            if animate:
                self.play(LaggedStart(
                    FadeOut(dice[0]),
                    bar_group1.animate.set_opacity(0.5),
                    FadeOut(dice[1]),
                    bar_group2.animate.set_opacity(0.5),
                    sum_labels.animate.set_opacity(0.25),
                    run_time=0.5
                ))
            else:
                self.wait(0.1)
                self.remove(dice)
                VGroup(bar_group1, bar_group2).set_opacity(0.5)
                sum_labels.set_opacity(0.25)

        self.wait()
        self.play(
            FadeOut(markers, lag_ratio=0.01),
            bar_group1.animate.set_opacity(1),
            bar_group2.animate.set_opacity(1),
            sum_labels.animate.set_opacity(1),
        )

    def isolate_pairs(self, bar_groups, full_table, *ij_tuples):
        full_table.set_opacity(0.25)
        full_table[2].set_fill(opacity=0)
        bar_groups.set_opacity(0.35)

        for i, j in ij_tuples:
            im1 = i - 1
            jm1 = j - 1
            n = im1 * 6 + jm1

            bar_groups[0][im1].set_opacity(1)
            bar_groups[1][jm1].set_opacity(1)

            full_table[0][im1].set_opacity(1)
            full_table[1][jm1].set_opacity(1)
            full_table[2][n].set_stroke(opacity=1)
            full_table[3][n].set_opacity(1)


class ConvolveDiscreteDistributions(SumAlongDiagonal):
    long_form = True

    def construct(self):
        # Set up two distributions
        dist1 = self.dist1
        dist2 = self.dist2
        blue_dice = get_die_faces(fill_color=BLUE_E, dot_color=WHITE)
        red_dice = get_die_faces(fill_color=RED_E, dot_color=WHITE)
        top_bars, low_bars = bar_groups = VGroup(
            get_bar_group(dist1, self.dist1_colors, blue_dice),
            get_bar_group(dist2, self.dist2_colors, red_dice),
        )

        bar_groups.arrange(DOWN, buff=LARGE_BUFF)
        bar_groups.to_edge(LEFT)

        self.add(bar_groups)

        # Setup the sum distribution
        conv_dist = np.convolve(dist1, dist2)
        sum_labels = die_sum_labels()
        sum_bar_group = get_bar_group(conv_dist, self.sum_colors, sum_labels)
        sum_bar_group.to_edge(RIGHT, buff=LARGE_BUFF)
        sum_bar_group.set_y(0)

        self.add(sum_bar_group)

        # V lines
        v_lines = get_bar_dividing_lines(top_bars)
        for bar in top_bars:
            v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
            v_line.set_stroke(GREY_C, 1, 0.75)
            v_line.set_x(bar.get_left()[0])
            v_line.set_y(0)
            v_lines.add(v_line)
        v_lines.add(v_lines[-1].copy().set_x(top_bars.get_right()[0]))

        # Flip
        low_bars.target = low_bars.generate_target()
        low_bars.target.arrange(LEFT, aligned_edge=DOWN, buff=0).move_to(low_bars)
        low_bars.target.move_to(low_bars)

        rect = SurroundingRectangle(low_bars)
        label = Text("Flip this")
        label.next_to(rect, RIGHT)

        low_arrow = Arrow(low_bars.get_right(), low_bars.get_left())
        low_arrow.set_stroke(color=YELLOW)
        low_arrow.next_to(low_bars, DOWN)

        self.play(
            ShowCreation(rect),
            FadeIn(label)
        )
        self.wait()
        self.play(
            MoveToTarget(low_bars, path_arc=PI / 3, lag_ratio=0.005)
        )
        self.play(ShowCreation(low_arrow), FadeOut(rect))
        self.wait()
        self.play(
            FadeOut(label), FadeOut(low_arrow),
            ShowCreation(v_lines, lag_ratio=0.1),
        )
        self.wait()

        # Show corresponding pairs
        rows = VGroup(blue_dice, VGroup(*reversed(red_dice))).copy()
        pairs = VGroup(*(VGroup(*pair) for pair in zip(*rows)))
        self.play(rows.animate.arrange(UP, SMALL_BUFF).next_to(bar_groups, RIGHT))

        rows.generate_target()
        rows.target.rotate(-90 * DEGREES)
        for row in rows.target:
            for die in row:
                die.rotate(90 * DEGREES)
        rows.target.arrange(RIGHT, buff=MED_SMALL_BUFF)
        rows.target.set_height(5)
        rows.target.next_to(bar_groups, RIGHT, buff=1)

        sum_bar_group.generate_target()
        sum_bar_group.target.set_opacity(0.25)
        sum_bar_group.target[7 - 2].set_opacity(1)

        self.play(
            MoveToTarget(rows, run_time=2),
            MoveToTarget(sum_bar_group),
        )
        self.wait()

        # Go through all pairs
        last_rect = VMobject()
        for n in [*range(6), *range(4, -1, -1)]:
            pair = pairs[n]
            rect = SurroundingRectangle(pair)
            rect.round_corners()
            bar_groups.generate_target()
            bar_groups.target.set_opacity(0.35)
            bar_groups.target[0][n].set_opacity(1)
            bar_groups.target[1][5 - n].set_opacity(1)
            self.play(
                FadeIn(rect),
                FadeOut(last_rect),
                MoveToTarget(bar_groups),
                run_time=0.5
            )
            self.wait(0.5)
            last_rect = rect
        self.play(FadeOut(last_rect), bar_groups.animate.set_opacity(1))
        self.wait()
        self.play(FadeOut(pairs), sum_bar_group.animate.set_opacity(1))

        # March!
        for bars in bar_groups:
            for i, bar in zip(it.count(1), bars):
                bar.index = i

        for n in [7, 5, *range(2, 13)]:
            sum_bar_group.generate_target()
            sum_bar_group.target.set_opacity(0.25)
            sum_bar_group.target[n - 2].set_opacity(1.0)

            self.play(
                get_row_shift(top_bars, low_bars, n),
                MoveToTarget(sum_bar_group),
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

            fade_anims = []

            # Spell out the full dot product
            products = VGroup()
            die_pair_targets = VGroup()
            for die_pair in die_pairs:
                product = VGroup(
                    p_mob(die_pair[0]),
                    p_mob(die_pair[1]),
                )
                product.arrange(RIGHT, buff=SMALL_BUFF)
                die_pair_targets.add(VGroup(
                    product[0].arg,
                    product[1].arg,
                ))
                products.add(product)

            products.arrange(DOWN, buff=0.75)
            products.move_to(midpoint(sum_bar_group.get_left(), bar_groups.get_right()))
            products.shift(2 * UP).shift_onto_screen()
            plusses = Tex("+", font_size=48).replicate(len(pairs))
            plusses[-1].scale(0).set_opacity(0)
            for plus, lp1, lp2 in zip(plusses, products, products[1:]):
                plus.move_to(VGroup(lp1, lp2))

            self.play(
                ShowIncreasingSubsets(products),
                ShowIncreasingSubsets(plusses),
                ShowIncreasingSubsets(pair_rects),
                run_time=0.35 * len(products)
            )
            self.wait(0.5)

            prod_group = VGroup(*products, *plusses)
            mover = prod_group.copy()
            mover.sort(lambda p: -p[1])
            mover.generate_target()
            mover.target.set_opacity(0)
            for mob in mover.target:
                mob.replace(sum_bar_group[n - 2].value_label, stretch=True)
            self.play(MoveToTarget(mover, remover=True, lag_ratio=0.002))
            self.wait(0.5)
            self.play(
                FadeOut(prod_group),
                FadeOut(pair_rects),
            )

        self.play(
            get_row_shift(top_bars, low_bars, 7),
            sum_bar_group.animate.set_opacity(1.0),
            run_time=0.5
        )

        # Distribution labels
        plabel_kw = dict(tex_to_color_map={"X": BLUE, "Y": RED})
        PX = Tex("P_X", **plabel_kw)
        PY = Tex("P_Y", **plabel_kw)
        PXY = Tex("P_{X + Y}", **plabel_kw)

        PX.next_to(top_bars.get_corner(UR), DR)
        PY.next_to(low_bars.get_corner(UR), DR)
        PXY.next_to(sum_bar_group, UP, MED_LARGE_BUFF)

        # Function label
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

        # Die rectangles
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

        # Describe the distribution as a function
        top_rect = SurroundingRectangle(top_bars)
        top_rect.set_stroke(BLUE, 3)
        top_rect.round_corners(radius=0.25)

        self.play(ShowCreation(top_rect))
        self.play(
            Write(PX),
            Write(func_label),
            ShowCreation(arrow),
        )
        self.wait()

        self.play(ShowCreation(die_rect), FadeOut(top_rect))
        self.play(FadeTransform(x_die.copy(), x_arg[:3]))
        self.play(TransformFromCopy(die_rect, value_rect))
        self.play(FadeTransform(value_label.copy(), x_arg[3:]))
        self.wait()
        for i in range(6):
            self.remove(*die_rects, *value_rects, *x_args)
            self.add(die_rects[i], value_rects[i], x_args[i])
            self.wait(0.5)

        # Label other distribution functions
        func_group = VGroup(func_label, arrow)
        func_group_Y = func_group.copy().shift(PY.get_center() - PX.get_center())
        func_group_XY = func_group.copy().shift(PXY.get_center() - PX.get_center())

        self.play(
            TransformFromCopy(func_group, func_group_Y),
            Write(PY),
            FadeOut(die_rects[-1]),
            FadeOut(value_rects[-1])
        )
        self.play(
            TransformFromCopy(func_group, func_group_XY),
            Write(PXY),
        )
        self.wait()
        self.play(LaggedStartMap(FadeOut, VGroup(
            func_group, func_group_Y, func_group_XY, x_args[-1]
        )))

        # Label convolution
        sum_bar_group.generate_target()
        sum_bar_group.target.shift(DOWN)
        conv_def = Tex(
            R"\big[P_X * P_Y\big](s) = \sum_{x = 1}^6 P_X(x) \cdot P_Y(s - x)",
            font_size=36,
            isolate=["x = 1", "6"],
            **plabel_kw,
        )
        conv_def.next_to(sum_bar_group.target, UP, buff=MED_LARGE_BUFF)

        PXY_arg = Tex("(s)", font_size=36)
        PXY.generate_target()
        lhs = conv_def[:10]
        PXY.target.next_to(lhs, UP, LARGE_BUFF).shift_onto_screen(buff=SMALL_BUFF)
        PXY_arg.next_to(PXY.target, RIGHT, buff=SMALL_BUFF)
        eq = Tex("=").rotate(90 * DEGREES)
        eq.move_to(midpoint(PXY.target.get_bottom(), lhs.get_top()))

        conv_rect = SurroundingRectangle(conv_def["P_X * P_Y"], buff=0.05)
        conv_rect.set_stroke(YELLOW, 2)
        conv_word = Text("Convolution")
        conv_word.match_color(conv_rect)
        conv_word.next_to(conv_rect, DOWN, buff=SMALL_BUFF)

        self.play(LaggedStart(
            MoveToTarget(sum_bar_group),
            MoveToTarget(PXY),
            FadeIn(PXY_arg, UP),
            Write(eq),
            TransformFromCopy(PX, lhs[1:3]),
            TransformFromCopy(PY, lhs[4:6]),
            Write(VGroup(lhs[0], lhs[3], *lhs[6:])),
        ))
        self.wait()
        self.play(
            Write(conv_word),
            ShowCreation(conv_rect),
        )
        self.wait()
        self.play(
            conv_rect.animate.become(SurroundingRectangle(conv_def["*"], buff=0.05, stroke_width=1))
        )
        self.wait()
        self.play(FadeOut(conv_word), FadeOut(conv_rect))

        self.add(conv_def)
        conv_def[10:].set_opacity(0)

        # Question right hand side
        question_rhs = Text("= (What formula goes here?)", font_size=30)
        question_rhs.next_to(conv_def[:10], RIGHT)
        self.play(Write(question_rhs))
        self.wait()

        # Show example input of 4
        ex_rhs = Tex(R"(4) = P_{-}(1)P_{-}(3) + P_{-}(2)P_{-}(2) + P_{-}(3)P_{-}(1)")
        ex_rhs.scale(0.9)
        ex_rhs.next_to(PXY, RIGHT, buff=0.1)
        for n, dot in enumerate(ex_rhs["-"]):
            even = n % 2 == 0
            substr = Tex("X" if even else "Y", font_size=24)
            substr.set_color(BLUE if even else RED)
            substr.move_to(dot)
            dot[0].become(substr)
            ex_rhs[ex_rhs.submobjects.index(dot[0]) + 2].match_color(substr)

        PXY_copy = PXY.copy()
        PXY_copy.generate_target()
        VGroup(PXY_copy.target, ex_rhs).to_edge(RIGHT, buff=-0.5)

        eq.generate_target()
        eq.target.rotate(-90 * DEGREES)
        eq.target.next_to(conv_def, LEFT)

        PXY.generate_target()
        PXY.target.next_to(eq.target, LEFT)
        VGroup(PXY.target, eq.target).align_to(PXY_copy.target, LEFT)

        example_box = SurroundingRectangle(VGroup(PXY_copy.target, ex_rhs))
        example_box.set_stroke(TEAL, 1)
        example_words = Text("For example")
        example_words.match_color(example_box)
        example_words.next_to(example_box, UP)

        self.play(LaggedStart(
            MoveToTarget(PXY_copy),
            MoveToTarget(eq),
            MoveToTarget(PXY),
            Transform(PXY_arg, ex_rhs["(4)"], remover=True),
            conv_def.animate.next_to(eq.target, RIGHT),
            MaintainPositionRelativeTo(question_rhs, conv_def),
            FadeIn(ex_rhs, LEFT),
            PX.animate.next_to(bar_groups[0], LEFT),
            PY.animate.next_to(bar_groups[1], LEFT),
            self.frame.animate.set_height(9).move_to(0.5 * UP),
            sum_bar_group.animate.shift(0.5 * DOWN),
            ShowCreation(example_box),
            FadeIn(example_words, UP),
            run_time=2
        ))
        self.wait()

        example = VGroup(PXY_copy, ex_rhs)
        self.add(example)

        # Cycle through cases
        example_box.save_state()
        for part in ex_rhs[re.compile(R"P[^+]*P[^+]*\)")]:
            self.play(
                example_box.animate.replace(part, stretch=True).scale(1.1).set_stroke(width=2),
                run_time=0.5
            )
            self.wait()
        self.play(FadeOut(example_box))
        self.wait()

        # Show full definition
        general_words = Text("In general")
        general_words.next_to(conv_def, UP)
        general_words.match_x(example_words)
        general_words.set_color(TEAL)

        example_words.generate_target()
        example_words.target.scale(0.75)
        example_words.target.set_y(4.5)
        example_words.target.set_color(GREY_B)

        conv_def.set_opacity(1)
        self.play(
            Write(conv_def[10:]),
            FadeOut(question_rhs, DOWN),
            FadeIn(general_words, DOWN),
            MoveToTarget(example_words),
            example.animate.scale(0.75).next_to(example_words.target, DOWN),
        )
        self.wait()

        # Talk through formula
        s_arrow = Vector(0.5 * DOWN, stroke_color=YELLOW)
        s_arrow.next_to(conv_def["s"][0], UP, SMALL_BUFF)
        x_arrow, y_arrow = s_arrow.replicate(2)
        x_arrow.next_to(conv_def["x"][1], UP, SMALL_BUFF)
        y_arrow.next_to(conv_def["s - x"], UP, SMALL_BUFF)

        self.play(GrowArrow(s_arrow))
        self.wait()
        self.play(LaggedStart(
            TransformFromCopy(s_arrow, x_arrow),
            TransformFromCopy(s_arrow, y_arrow),
        ))
        self.wait()

        xeq1 = conv_def["x = 1"]
        y_arrow.save_state()
        self.play(
            s_arrow.animate.scale(0.75).rotate(90 * DEGREES).next_to(xeq1, LEFT, SMALL_BUFF),
            y_arrow.animate.scale(0.75).rotate(-90 * DEGREES).next_to(xeq1, RIGHT, SMALL_BUFF),
        )
        self.wait()
        self.play(
            s_arrow.animate.next_to(conv_def["6"], LEFT, SMALL_BUFF),
            y_arrow.animate.next_to(conv_def["6"], RIGHT, SMALL_BUFF),
        )
        self.wait()
        self.play(
            Restore(y_arrow),
            FadeOut(s_arrow),
        )
        self.wait()
        self.play(LaggedStart(FadeOut(x_arrow), FadeOut(y_arrow)))

        # Show zero'd example
        bar_groups.generate_target()
        bar_groups.target.scale(2 / 3, about_point=top_bars.get_top() + UP)

        example_words = TexText("Plugging in $s = 4$")
        example_words.next_to(conv_def, DOWN, buff=1.0)

        example = Tex(
            R"""
            [P_X * P_Y](4) =
            &P_X(1) \cdot P_Y(3) \; + \\
            &P_X(2) \cdot P_Y(2) \; + \\
            &P_X(3) \cdot P_Y(1) \; + \\
            &P_X(4) \cdot P_Y(0) \; + \\
            &P_X(5) \cdot P_Y(-1) \; + \\
            &P_X(6) \cdot P_Y(-2)
            """, 
            t2c={
                "X": BLUE,
                "Y": RED,
            },
            font_size=36
        )
        example.next_to(example_words, DOWN, buff=0.5)

        summands = VGroup(*(
            example[Rf"P_X({x}) \cdot P_Y({4 - x})"]
            for x in range(1, 7)
        ))
        plusses = example["+"]
        plusses.shift(SMALL_BUFF * RIGHT)
        plusses.add(VectorizedPoint(summands[-1].get_center()))

        self.play(
            FadeIn(example[R"[P_X * P_Y](4) ="]),
            FadeOut(v_lines),
            MoveToTarget(bar_groups),
            MaintainPositionRelativeTo(PX, bar_groups[0]),
            Write(example_words),
            PY.animate.next_to(bar_groups.target[1], RIGHT).match_x(PX),
            sum_bar_group.animate.scale(0.5).next_to(bar_groups.target, DOWN, buff=0.75, aligned_edge=LEFT),
        )
        last_rect = VectorizedPoint(summands[0].get_center())
        for x, summand, plus in zip(it.count(1), summands, plusses):
            rect = SurroundingRectangle(VGroup(summand, plus))
            rect.set_stroke(BLUE, 2)
            rect.round_corners()
            rect.add(Tex(Rf"x = {x}").next_to(rect, DOWN, SMALL_BUFF))
            self.play(
                FadeTransform(
                    conv_def[R"P_X(x) \cdot P_Y(s - x)"].copy(),
                    summand,
                ),
                FadeTransform(last_rect, rect),
                FadeIn(plus),
                run_time=0.5
            )
            self.wait()
            last_rect = rect
        self.play(FadeOut(last_rect))

        # Highlight zeros
        zeroed_terms = VGroup(*(
            example[Rf"P_Y({n})"]
            for n in range(0, -4, -1)
        ))
        zeroed_rect = SurroundingRectangle(zeroed_terms)
        zeroed_rect.set_stroke(RED, 3)
        zeroed_rect.stretch(1.3, 0, about_edge=LEFT)
        zeroed_rect.round_corners()

        eq_zero = Tex("= 0")
        eq_zero.next_to(zeroed_rect, RIGHT)
        eq_zero.set_color(RED)

        self.play(ShowCreation(zeroed_rect))
        self.wait()
        self.play(Write(eq_zero))
        self.play(
            summands[3:].animate.set_opacity(0.35),
            plusses[3:].animate.set_opacity(0.35),
        )
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


class ShowConvolutionOfLists(SumAlongDiagonal):
    def construct(self):
        # Set up two distributions
        dist1 = self.dist1
        dist2 = self.dist2
        conv_dist = np.convolve(dist1, dist2)
        kw = dict(height=1.5)
        blue_bars, red_bars, sum_bars = bar_groups = VGroup(
            get_bar_group(dist1, self.dist1_colors, **kw),
            get_bar_group(dist2, self.dist2_colors, **kw),
            get_bar_group(conv_dist, self.sum_colors, **kw),
        )

        # Create equation
        parens = Tex("()()")
        parens.stretch(2, 1)
        parens.match_height(bar_groups)
        asterisk = Tex("*", font_size=96)
        equation = VGroup(
            parens[0], blue_bars, parens[1],
            asterisk,
            parens[2], red_bars, parens[3],
            Tex("=", font_size=96),
            sum_bars,
        )
        equation.arrange(RIGHT)
        equation.set_width(FRAME_WIDTH - 1)
        equation.to_edge(UP, buff=1.0)

        self.add(equation)
        self.remove(sum_bars)
        self.play(
            TransformFromCopy(blue_bars, sum_bars, lag_ratio=0.003),
            TransformFromCopy(red_bars, sum_bars, lag_ratio=0.003),
            run_time=1.5
        )
        self.wait()

        # Name operation
        arrow = Vector(0.5 * DOWN)
        arrow.next_to(asterisk, UP)
        name = Text("Convolution", font_size=60)
        name.next_to(arrow, UP)
        VGroup(arrow, name).set_color(YELLOW)

        self.play(
            Write(name),
            GrowArrow(arrow)
        )
        self.play(FlashAround(asterisk))
        self.wait()

        # Lists of numbers vs functions
        list_words = Text("List of numbers", font_size=36).replicate(3)
        func_words = Text("Function", font_size=36).replicate(3)
        crosses = VGroup()
        for list_word, func_word, bar_group in zip(list_words, func_words, bar_groups):
            list_word.next_to(bar_group, DOWN)
            func_word.next_to(bar_group, DOWN)
            crosses.add(Cross(list_word))

        for list_word, bar_group in zip(list_words, bar_groups):
            self.play(
                FadeIn(list_word, DOWN),
                LaggedStart(*(
                    FlashAround(bar[2], time_width=1.5, buff=0.05)
                    for bar in bar_group
                ), lag_ratio=0.03, run_time=2)
            )
            self.wait()
        self.play(LaggedStartMap(ShowCreation, crosses, lag_ratio=0.1, run_time=1))
        self.play(
            LaggedStartMap(FadeIn, func_words, shift=0.5 * DOWN, scale=0.5, lag_ratio=0.5),
            list_words.animate.shift(0.5 * DOWN),
            crosses.animate.shift(0.5 * DOWN),n
        )
        self.wait()

        # Cycle through appropriate pairs
        for s in range(len(sum_bars)):
            bar_groups.set_opacity(0.75)
            sum_bars[s].set_opacity(1)
            for x in range(len(blue_bars)):
                bar_groups[:2].set_opacity(0.75)
                y = s - x
                if 0 <= y < len(red_bars):
                    blue_bars[x].set_opacity(1)
                    red_bars[y].set_opacity(1)
                    self.wait(0.25)
            self.wait(0.5)
        self.play(bar_groups.animate.set_opacity(1))
        self.wait()


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
