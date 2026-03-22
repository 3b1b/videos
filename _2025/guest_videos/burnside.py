from manim_imports_ext import *


class IncompleteSquares(InteractiveScene):
    def construct(self):
        # Set up top row
        squares = VGroup(self.get_square(n) for n in range(16))
        squares.arrange(RIGHT, buff=0.5)
        squares.center()
        squares.set_width(FRAME_WIDTH - 1)
        self.add(squares)

        # Reorder by groups
        groups = VGroup(
            VGroup(squares[i] for i in index_groups)
            for index_groups in [
                [0],
                [1, 2, 4, 8],
                [3, 6, 9, 12],
                [5, 10],
                [7, 11, 13, 14],
                [15],
            ]
        )
        groups.target = groups.generate_target()
        for group in groups.target:
            group.arrange(RIGHT, buff=0.25)
        groups.target.arrange(RIGHT, buff=0.5)
        groups.target.set_width(FRAME_WIDTH - 1)

        rects = VGroup(
            SurroundingRectangle(group, buff=0.15).set_stroke(width=3).round_corners()
            for group in groups.target
        )
        rects.set_submobject_colors_by_gradient(RED, YELLOW)

        self.play(
            MoveToTarget(groups, path_arc=45 * DEG, run_time=3),
        )
        self.play(LaggedStartMap(ShowCreation, rects, lag_ratio=0.5))
        self.wait()

        # Count the groups
        ones = VGroup(
            Integer(1).next_to(rect, UP)
            for rect in rects
        )
        plusses = VGroup(
            Tex(R"+").move_to(VGroup(pair))
            for pair in zip(ones, ones[1:])
        )
        brace = Brace(ones, UP)
        six = Integer(6).next_to(brace, UP)
        brace_group = VGroup(brace, six)

        self.play(
            LaggedStartMap(FadeIn, ones, shift=0.25 * UP),
            LaggedStartMap(FadeIn, plusses, shift=0.25 * UP),
        )
        self.wait()
        self.play(
            GrowFromCenter(brace),
            Write(six)
        )
        self.wait()

        # Show the fractions
        all_fractions = VGroup(ones[0])
        fraction_groups = VGroup(ones[:1])
        ones_shift = 0.15 * UP
        for group in groups[1:-1]:
            n = len(group)
            new_fracs = VGroup(
                Tex(Rf"1 \over {{{n}}}", font_size=36).next_to(square, UP).match_y(ones).shift(ones_shift)
                for square in group
            )
            all_fractions.add(*new_fracs)
            fraction_groups.add(new_fracs)

        all_fractions.add(ones[-1])
        fraction_groups.add(ones[-1:])

        new_plusses = VGroup(
            Tex(R"+", font_size=36).move_to(VGroup(*pair))
            for pair in zip(all_fractions, all_fractions[1:])
        )

        self.play(
            FadeOut(plusses),
            brace_group.animate.next_to(all_fractions, UP, SMALL_BUFF),
            ones.animate.shift(ones_shift)
        )
        for one, frac_group in zip(ones, fraction_groups):
            self.play(ReplacementTransform(one, frac_group, lag_ratio=0.01, run_time=1))

        self.play(FadeIn(new_plusses, lag_ratio=0.1))

        top_sum = VGroup(all_fractions, new_plusses)

        # Show the rotations
        frame = self.frame
        v_line = Line(rects.get_left(), rects.get_right()).next_to(rects, DOWN)
        v_line.set_stroke(WHITE, 2)
        v_line.next_to(rects, DOWN)
        v_line.scale(1.2, about_edge=RIGHT)

        def get_rot_sym(angle, label_tex):
            arcs = VGroup(
                Arc(0, angle),
                Arc(PI, angle)
            )
            for arc in arcs:
                arc.set_stroke(TEAL, 3)
                arc.add_tip()
            arcs.scale(0.45)
            label = Tex(label_tex, font_size=24)
            return VGroup(arcs, label)

        rot_symbols = VGroup(
            Text("Id"),
            get_rot_sym(90 * DEG, R"90^\circ"),
            get_rot_sym(165 * DEG, R"180^\circ"),
            get_rot_sym(-90 * DEG, R"-90^\circ"),
        )
        rot_symbols.arrange(DOWN, buff=0.75)
        rot_symbols.next_to(v_line, DOWN, 0.75)
        rot_symbols.set_x(rects.get_x(LEFT) - rot_symbols.get_width() - MED_LARGE_BUFF)

        self.play(
            ShowCreation(v_line),
            frame.animate.reorient(0, 0, 0, (-1.55, -2.67, 0.0), 10.59),
            FadeIn(rot_symbols)
        )
        self.wait()

        # Create columns
        columns = VGroup()
        squares.sort(lambda p: p[0])  # Sort from left to right
        for square in squares:
            col = VGroup(
                square.copy().rotate(angle)
                for angle in np.arange(0, TAU, TAU / 4)
            )
            col.match_x(square)
            for part, sym in zip(col, rot_symbols):
                part.match_y(sym)

            columns.add(col)

        fixed_points = [col[0] for col in columns]
        fixed_points.extend(columns[0][1:])
        fixed_points.extend(columns[-1][1:])
        fixed_points.extend([
            columns[9][2],
            columns[10][2],
        ])
        fixed_point_dots = Group(GlowDot(radius=0.5).move_to(point) for point in fixed_points)

        # Show example columns
        low_opacity = 0.2
        ex_index = 4
        self.play(
            rects.animate.set_stroke(opacity=low_opacity),
            squares[:ex_index].animate.set_stroke(opacity=low_opacity),
            squares[ex_index + 1:].animate.set_stroke(opacity=low_opacity),
            all_fractions.animate.set_fill(opacity=low_opacity),
            top_sum.animate.set_opacity(low_opacity),
            brace_group.animate.set_opacity(low_opacity)
        )
        for piece in columns[ex_index]:
            self.play(TransformFromCopy(squares[ex_index], piece, path_arc=30 * DEG))
        self.wait()
        self.play(
            squares[1:ex_index].animate.set_stroke(opacity=1),
            rects[1].animate.set_stroke(opacity=1)
        )
        self.wait()

        ex_index = 9
        self.play(squares[ex_index].animate.set_stroke(opacity=1))
        for piece in columns[ex_index]:
            self.play(TransformFromCopy(squares[ex_index], piece, path_arc=30 * DEG))
        self.wait()
        self.play(
            rects[3].animate.set_stroke(opacity=1),
            squares[10].animate.set_stroke(opacity=1),
        )
        self.wait()

        ex_index = 15
        self.play(squares[ex_index].animate.set_stroke(opacity=1))
        for piece in columns[ex_index]:
            self.play(TransformFromCopy(squares[ex_index], piece, path_arc=30 * DEG))
        self.wait()
        self.play(rects[5].animate.set_stroke(opacity=1))

        # Show fixed point dots
        dot = GlowDot(radius=0.5)
        ex_dots1 = Group(dot.copy().move_to(part) for part in columns[9][0::2])
        ex_dots2 = Group(dot.copy().move_to(part) for part in columns[15])
        ex_dots3 = Group(dot.copy().move_to(columns[4][0]))

        for dots in [ex_dots1, ex_dots2, ex_dots3]:
            self.play(FadeIn(dots))
            self.wait()

        # Show fractions
        ex_fractions = VGroup(all_fractions[i] for i in [4, 9, 15])
        ex_dot_groups = Group(ex_dots3, ex_dots1, ex_dots2)

        def get_fourth_exprs(dot_group):
            return VGroup(
                Tex(R"1 / 4", font_size=24).next_to(dot, DOWN, buff=0)
                for dot in dot_group
            )

        ex_fourths = VGroup(
            get_fourth_exprs(dot_group)
            for dot_group in ex_dot_groups
        )

        self.play(ex_fractions.animate.set_opacity(1))
        self.wait()
        self.play(
            LaggedStart(
                (TransformFromCopy(frac, fourth_group, path_arc=30 * DEG)
                for frac, fourth_group in zip(ex_fractions, ex_fourths)),
                lag_ratio=0.75,
            )
        )
        self.wait()

        # Light everything back up
        self.play(
            brace_group.animate.set_opacity(1),
            top_sum.animate.set_opacity(1),
            rects.animate.set_stroke(opacity=1),
            squares.animate.set_stroke(opacity=1),
        )
        self.wait()

        # Show all transform/shape pairs
        all_fourths = get_fourth_exprs(fixed_point_dots)

        self.play(LaggedStart(
            (ReplacementTransform(square.replicate(4), col, path_arc=30 * DEG)
            for square, col in zip(squares, columns)),
            lag_ratio=0.5,
            run_time=5
        ))
        self.play(
            FadeIn(fixed_point_dots),
            FadeOut(ex_dot_groups),
        )
        self.play(
            FadeIn(all_fourths),
            FadeOut(ex_fourths),
        )
        self.wait()

    def get_square(self, edge_pattern: int = 0, edge_color=BLUE, edge_stroke_width=5):
        outline = VGroup(
            DashedLine(p1, p2, dash_length=0.1).set_stroke(GREY_C, 1)
            for p1, p2 in adjacent_pairs([UL, UR, DR, DL])
        )
        pattern = [(edge_pattern >> (3 - i)) & 1 == 1 for i in range(4)]  # Pattern of bools
        bold_edges = VGroup()

        for edge, include in zip(outline, pattern):
            if include:
                line = Line(edge.get_start(), edge.get_end())
                line.set_stroke(edge_color, edge_stroke_width)
                line.scale(1.075)  # Bad hack for bevel
                bold_edges.add(line)

        return VGroup(outline, bold_edges)
