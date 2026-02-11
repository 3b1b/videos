from manim_imports_ext import *


def get_boundary_volume_texs():
    return [
        R"0",
        R"2",
        R"2\pi r",
        R"4\pi r^2",
        R"2\pi^2 r^3",
        R"{8 \over 3}\pi^2 r^4",
        R"\pi^3 r^5",
        R"{16 \pi^3 \over 15} r^6",
        R"{\pi^4 \over 3} r^7",
        R"{32 \pi^4 \over 105} r^8",
    ]


def get_volume_texs():
    return [
        R"1",
        R"2 r",
        R"\pi r^2",
        R"{4 \over 3} \pi r^3",
        R"{\pi^2 \over 2} r^4",
        R"{8 \over 15} \pi^2 r^5",
        R"{\pi^3 \over 6} r^6",
        R"{16 \pi^3 \over 105} r^7",
        R"{\pi^4 \over 24} r^8",
        R"{32 \pi^4 \over 945} r^9",
        R"{\pi^5 \over 120} r^10",
    ]


class CircumferenceToArea(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"{r}": BLUE}
        circum_formula = Tex(R"\text{Circumference} = 2 \pi {r}", t2c=t2c)

        radius = 3
        circle = Circle(radius=radius)
        circle.set_stroke(YELLOW, 5)
        radius_line = Line(ORIGIN, radius * RIGHT)
        radius_line.set_stroke(WHITE, 3)
        r_label = Tex(R"r", font_size=72)
        r_label.set_color(BLUE)
        r_label.next_to(radius_line, UP, SMALL_BUFF)
        r_group = VGroup(radius_line, r_label)
        r_group.set_z_index(1)

        self.add(circle, r_group)
        self.play(
            ShowCreation(circle),
            Rotate(r_group, TAU, about_point=ORIGIN),
            run_time=3,
        )
        self.wait()

        # Inner circles
        circles = VGroup(
            circle.copy().set_width(a * circle.get_width())
            for a in np.linspace(1, 0, 100)
        )
        circles.set_stroke(YELLOW, 3, 0.25)
        self.play(
            ReplacementTransform(
                circle.replicate(len(circles)).set_stroke(width=0, opacity=0),
                circles,
                lag_ratio=0.1
            ),
            run_time=3
        )
        self.wait()


class SurfaceAreaToVolume(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        radius = 3
        sphere = Sphere(radius=radius)
        sphere.set_color(BLUE, 0.5)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere, resolution=(51, 26))
        mesh.set_stroke(WHITE, 2, 0.2)

        frame.reorient(44, 56, 0)
        self.play(
            ShowCreation(sphere),
            Write(mesh),
            run_time=2
        )
        self.wait()

        # Inner spheres
        inner_spheres = Group(
            sphere.copy().set_width(a * sphere.get_width())
            for a in np.linspace(0, 1, 50)
        )
        inner_spheres.set_color(BLUE, 0.2)
        inner_spheres.set_clip_plane(IN, 0)

        self.remove(sphere)
        self.add(inner_spheres, mesh)
        self.play(
            ShowCreation(inner_spheres, lag_ratio=0.5, run_time=3),
        )
        self.play(inner_spheres.animate.set_clip_plane(IN, radius), run_time=2)
        self.wait()


class VolumeGrid(InteractiveScene):
    tex_to_color = {"r": BLUE}

    def construct(self):
        # Write the grid
        frame = self.frame
        n_cols = 10
        grid = self.get_grid(n_cols)
        boundary_labels, volume_labels = self.get_volume_labels()
        labels = VGroup(
            VGroup(*pair)
            for pair in zip(boundary_labels, volume_labels)
        )

        for label_pair, col in zip(labels, grid):
            for label, square in zip(label_pair, col):
                label.move_to(square)
                square.label = label

        self.add(grid[1:4])

        # Row labels
        row_labels = VGroup(
            Tex(R"\partial B^n"),
            Tex(R"B^n"),
        )
        for square, label in zip(grid[1], row_labels):
            label.next_to(square, LEFT)

        col_labels = VGroup(Integer(n, font_size=40) for n in range(n_cols))
        for col, label in zip(grid, col_labels):
            label.next_to(col, UP)

        cols_label = Text("Dimension")
        cols_label.next_to(col_labels[1:4], UP)
        cols_label.to_edge(UP, buff=MED_SMALL_BUFF)

        self.add(col_labels[1:4])
        self.add(cols_label)

        # Add for d=2 and d=3
        def highlight_cell(row, col, run_time=3, fill_color=TEAL_E, fill_opacity=0.5):
            kw = dict(rate_func=there_and_back_with_pause, run_time=run_time)
            cell = grid[col][row]
            return cell.animate.set_fill(fill_color, fill_opacity).set_anim_args(**kw)

        for d, n in it.product([2, 3], [0, 1]):
            self.play(
                highlight_cell(n, d),
                Write(labels[d][n]),
            )

        # Show derivatives
        self.show_derivative_and_integral(grid, 2)
        self.show_derivative_and_integral(grid, 3)

        # Add row labels
        rect = SurroundingRectangle(row_labels[1])
        rect.set_stroke(TEAL, 3)
        boundary_word = Text("“Boundary”")
        boundary_word.set_color(TEAL)
        boundary_word.next_to(row_labels[0], LEFT)

        self.play(ShowCreation(rect), Write(row_labels[1]))
        self.wait()
        self.play(
            rect.animate.surround(row_labels[0]),
            TransformMatchingTex(row_labels[1].copy(), row_labels[0]),
            run_time=1
        )
        self.wait()
        self.play(
            frame.animate.set_x(-3),
            FadeIn(boundary_word, lag_ratio=0.1),
            rect.animate.surround(row_labels[0][0])
        )
        self.wait()
        self.play(
            frame.animate.set_x(0),
            FadeOut(rect),
            FadeOut(boundary_word),
        )
        self.wait()

        # Add for d=1
        self.play(
            highlight_cell(1, 1),
            FadeIn(volume_labels[1])
        )
        self.play(
            highlight_cell(0, 1),
            FadeIn(boundary_labels[1])
        )

        # Ask about the rest
        q_marks = VGroup(
            VGroup(
                Tex(R"?", font_size=72).move_to(cell)
                for cell in col
            )
            for col in grid[4:]
        )
        q_marks.set_fill(YELLOW)
        self.play(
            Write(q_marks, lag_ratio=0.05),
            Write(grid[4:], lag_ratio=0.05),
            Write(col_labels[4:], lag_ratio=0.05),
            cols_label.animate.match_x(col_labels),
        )

        # Show knights move
        knight_group = self.get_knights_move_group(grid, 3)
        circle_cell = grid[2][0]

        self.play(FadeIn(knight_group))
        self.wait()
        self.play(circle_cell.animate.set_fill(RED, 0.35))
        self.wait()

        # New knights moves
        for d in range(4, 10):
            self.play(
                knight_group.animate.move_to(grid[d - 2][1], DL),
                FadeOut(q_marks[d - 4][0])
            )
            label_copy = boundary_labels[d].copy()
            self.play(
                TransformMatchingTex(boundary_labels[2].copy(), label_copy),
                TransformMatchingTex(volume_labels[d - 2].copy(), boundary_labels[d]),
                run_time=1
            )
            self.remove(label_copy)
            self.wait()
            self.show_derivative_and_integral(
                grid, d,
                int_added_anims=[
                    FadeOut(q_marks[d - 4][1]),
                    TransformMatchingTex(boundary_labels[d].copy(), volume_labels[d]),
                ],
                skip_derivative=True
            )
            self.wait()

        # Clean up
        self.play(
            FadeOut(knight_group),
            circle_cell.animate.set_fill(opacity=0)
        )

        # Show volume constants
        t2c = {"b_n": YELLOW, "b_{n - 2}": YELLOW}
        gen_formula = Tex(R"V(B^n) = b_n r^n", t2c=t2c, font_size=72)
        gen_formula["r"].set_color(BLUE)
        gen_formula.to_edge(DOWN)
        gen_b_part = gen_formula["b_n"][0]

        kw = dict(font_size=48)
        c_formulas = VGroup(
            Tex(R"b_0 = 1", **kw),
            Tex(R"b_1 = 2", **kw),
            Tex(R"b_2 = \pi", **kw),
            Tex(R"b_3 = {4 \over 3} \pi", **kw),
            Tex(R"b_4 = {\pi^2 \over 2}", **kw),
            Tex(R"b_5 = {8 \over 15} \pi^2", **kw),
            Tex(R"b_6 = {\pi^3 \over 6}", **kw),
            Tex(R"b_7 = {16 \over 105} \pi^3", **kw),
            Tex(R"b_8 = {\pi^4 \over 24}", **kw),
        )

        self.play(Write(gen_formula))
        self.wait()

        last_highlight = VGroup()
        last_b_formula = VGroup()
        for col, b_formula, label in zip(grid[1:], c_formulas[1:], volume_labels[1:]):
            highlight = col[1].copy()
            highlight.set_fill(TEAL_E, 0.5)

            b_part = b_formula[re.compile(r"b_.")][0]
            b_part.set_color(YELLOW)
            b_formula.move_to(highlight)
            b_formula.shift(1.0 * highlight.get_height() * DOWN)

            group = VGroup(highlight, b_formula)
            self.play(
                FadeOut(last_highlight),
                FadeOut(last_b_formula),
                FadeIn(highlight),
                TransformFromCopy(gen_b_part, b_part),
                FadeTransform(
                    label[:len(b_formula) - 3].copy(),
                    b_formula[3:],
                    time_span=(0.25, 1)
                ),
                Write(b_formula[2], time_span=(0.25, 1.0)),
            )
            self.wait()

            last_highlight = highlight
            last_b_formula = b_formula
        self.play(
            FadeOut(last_highlight),
            FadeOut(last_b_formula),
        )

        # Show recursion formula
        recursion_formula = Tex(R"b_n = {2\pi \over n} b_{n - 2}", t2c=t2c, font_size=72)
        alt_recursion_formula = Tex(R"b_n = {\pi \over n / 2} b_{n - 2}", t2c=t2c, font_size=72)
        recursion_formula.to_corner(DR)
        alt_recursion_formula.move_to(recursion_formula)

        self.play(
            gen_formula.animate.match_y(recursion_formula).to_edge(LEFT, buff=LARGE_BUFF),
            TransformFromCopy(gen_formula["b_n"], recursion_formula["b_n"]),
        )
        self.play(Write(recursion_formula[2:]))
        self.wait()
        self.play(TransformMatchingTex(recursion_formula, alt_recursion_formula))
        self.wait()

        # Shrink
        zero_group = VGroup(grid[0], col_labels[0], labels[0])
        zero_group.set_fill(opacity=0)
        zero_group.set_stroke(opacity=0)
        grid_group = VGroup(grid, row_labels, col_labels, cols_label, labels)
        formula_group = VGroup(gen_formula, alt_recursion_formula)
        self.play(
            grid_group.animate.set_height(3.0, about_edge=UP),
            formula_group.animate.arrange(DOWN, buff=0.5, aligned_edge=LEFT).set_max_height(2).to_corner(DL)
        )
        self.wait()

        # Show recursion example
        stages = VGroup(
            Tex(R"b_8 = {\pi \over 4} b_6"),
            Tex(R"b_8 = {\pi \over 4} {\pi \over 3} b_4"),
            Tex(R"b_8 = {\pi \over 4} {\pi \over 3} {\pi \over 2} b_2"),
            Tex(R"b_8 = {\pi \over 4} {\pi \over 3} {\pi \over 2} {\pi \over 1} b_0"),
        )
        stages.set_height(1.2)
        stages.next_to(formula_group, RIGHT, buff=2.5)
        for stage in stages:
            stage.align_to(stages[-1], LEFT)
            stage[re.compile(r"b_.")].set_color(YELLOW)

        mult_arrows = VGroup(
            Arrow(
                col1.get_bottom(),
                col2.get_bottom(),
                path_arc=120 * DEG
            )
            for col1, col2 in zip(grid[0::2], grid[2::2])
        )
        arrow_label_texs = [
            R"\times \pi / 1",
            R"\times \pi / 2",
            R"\times \pi / 3",
            R"\times \pi / 4",
        ]
        for arrow, tex in zip(mult_arrows, arrow_label_texs):
            label = Tex(tex)
            label.next_to(arrow, DOWN, SMALL_BUFF)
            arrow.push_self_into_submobjects()
            arrow.add(label)

        zero_cells = grid[0]

        highlights = VGroup(col[1].copy() for col in grid[0::2])
        highlights.set_fill(TEAL, 0.5)

        self.play(
            FadeOut(boundary_labels[4:]),
            FadeOut(volume_labels[4:]),
        )
        self.play(
            Write(stages[0]),
        )
        self.wait()
        self.play(
            Write(mult_arrows[-1]),
            FadeIn(highlights[-2:]),
        )
        self.wait()
        self.play(
            TransformMatchingTex(stages[0], stages[1], key_map={"b_6": "b_4"}, run_time=1),
            Write(mult_arrows[-2]),
            FadeIn(highlights[-3]),
        )
        self.wait()
        self.play(
            TransformMatchingTex(stages[1], stages[2], key_map={"b_4": "b_2"}, run_time=1),
            Write(mult_arrows[-3]),
            FadeIn(highlights[-4]),
        )
        self.wait()
        self.play(
            TransformMatchingTex(stages[2], stages[3], key_map={"b_2": "b_0"}, run_time=1),
            Write(mult_arrows[-4]),
            FadeIn(highlights[-5]),
            grid[0].animate.set_stroke(opacity=1),
            Write(col_labels[0]),
            row_labels.animate.next_to(grid, LEFT, MED_SMALL_BUFF)
        )
        self.wait()

        # Fill in zero terms
        self.play(volume_labels[0].animate.set_fill(opacity=1))
        self.wait()
        self.play(boundary_labels[0].animate.set_fill(opacity=1))
        self.wait()

        # General formula
        b8_form = stages[-1]

        gen_b_form = Tex(R"b_n = {\pi^{n / 2} \over (n / 2)!}", t2c=t2c, font_size=48)
        gen_b_form.move_to(b8_form)
        gen_b_form.to_edge(RIGHT, buff=LARGE_BUFF)

        small_b8_form = b8_form.copy()
        small_b8_form.generate_target()
        small_b8_form.target[-2:].set_opacity(0)
        small_b8_form.target.shift(0.75 * LEFT)
        small_b8_form.target.scale(gen_b_form[0].get_height() / small_b8_form[0].get_height())

        pis = b8_form[3:-2]
        pis_target = gen_b_form[R"{\pi^{n / 2} \over (n / 2)!}"][0]
        pis_rect = SurroundingRectangle(pis, buff=SMALL_BUFF)
        pis_rect.set_stroke(TEAL, 3)

        self.play(ShowCreation(pis_rect))
        self.wait()
        self.play(
            TransformMatchingTex(
                b8_form, gen_b_form,
                key_map={"b_8": "b_n"},
                matched_keys=[R"\pi", R"\over"]
            ),
            pis_rect.animate.surround(pis_target),
            MoveToTarget(small_b8_form),
            run_time=1
        )
        self.play(FadeOut(pis_rect))
        self.wait()

        # Substitute in
        final_formula = Tex(
            R"V(B^n) = {\pi^{n / 2} \over (n / 2)!} {r}^n",
            t2c={"{r}": BLUE},
            font_size=72
        )
        final_formula.next_to(grid, DOWN, buff=2.25)
        final_formula.to_edge(LEFT)

        bn_parts = VGroup(
            formula[R"{\pi^{n / 2} \over (n / 2)!}"]
            for formula in [gen_b_form, final_formula]
        )
        bn_rect = SurroundingRectangle(bn_parts[0])
        bn_rect.set_stroke(YELLOW, 1)

        self.play(
            FadeOut(small_b8_form),
            FadeOut(alt_recursion_formula),
        )
        self.play(ShowCreation(bn_rect))
        self.play(
            TransformFromCopy(*bn_parts),
            TransformFromCopy(
                gen_formula[R"V(B^n) = "].copy(),
                final_formula[R"V(B^n) = "],
            ),
            TransformFromCopy(
                gen_formula[R"r^n"],
                final_formula[R"{r}^n"],
            ),
            FadeOut(gen_b_form),
            FadeOut(gen_formula),
            bn_rect.animate.surround(bn_parts[1], buff=0.05),
        )
        self.wait()
        self.play(
            bn_rect.animate.surround(final_formula, buff=0.25).set_stroke(width=2),
        )
        self.add(final_formula)
        self.wait()

        # Fill in even volume labels
        def fill_every_other_label_from(n=2):
            for vl1, vl2 in zip(volume_labels[n::2], volume_labels[n + 2::2]):
                self.play(
                    TransformMatchingTex(vl1.copy(), vl2, path_arc=60 * DEG),
                    run_time=1
                )

            self.wait()
            self.play(LaggedStart(
                *(
                    TransformMatchingTex(vl.copy(), bl)
                    for vl, bl in zip(volume_labels[n + 2::2], boundary_labels[n + 2::2])
                ),
                run_time=1.5,
                lag_ratio=0.2
            ))

        fill_every_other_label_from(2)

        # Shift multiplication arrows
        pure_mult_arrows = VGroup(ma[0] for ma in mult_arrows)
        mult_arrow_labels = VGroup(ma[1] for ma in mult_arrows)

        shift_vect = grid[1].get_center() - grid[0].get_center()
        pure_mult_arrows.generate_target()
        pure_mult_arrows.target.shift(shift_vect)

        new_arrow_texs = [
            R"\times {\pi \over 3 / 2}",
            R"\times {\pi \over 5 / 2}",
            R"\times {\pi \over 7 / 2}",
            R"\times {\pi \over 9 / 2}",
        ]
        new_arrow_labels = VGroup(map(Tex, new_arrow_texs))
        for label, arrow in zip(new_arrow_labels, pure_mult_arrows.target):
            label.next_to(arrow, DOWN, SMALL_BUFF)

        self.play(
            MoveToTarget(pure_mult_arrows, lag_ratio=0.2),
            LaggedStart(
                *(
                    TransformMatchingTex(l1, l2)
                    for l1, l2 in zip(mult_arrow_labels, new_arrow_labels)
                ),
                lag_ratio=0.2,
            ),
            run_time=1.5
        )
        self.wait()

        # Fill in odd volume labels
        fill_every_other_label_from(3)

        # Plug it in for n = 1
        d1_form = Tex(R"V(B^1) = {\pi^{1/2} \over (1/2)!} {r} = 2{r}", t2c={"{r}": BLUE})
        alt_d1_form = Tex(R"V(B^1) = {\sqrt{\pi} \over (1/2)!} {r} = 2{r}", t2c={"{r}": BLUE})
        for form in d1_form, alt_d1_form:
            form.next_to(final_formula, RIGHT, buff=1.0, aligned_edge=DOWN)

        self.play(
            VGroup(final_formula, bn_rect).animate.scale(0.7, about_edge=DL),
            FadeTransform(final_formula.copy(), d1_form),
        )
        self.wait()
        self.play(
            TransformMatchingTex(
                d1_form,
                alt_d1_form,
                key_map={R"^{1/2}": R"\sqrt"},
                matched_keys=[R"\pi"],
                run_time=1,
            )
        )
        self.wait()

        # Half factoril fact
        half_fact = Tex(R"(1/2)! = {\sqrt{\pi} \over 2}")
        half_fact.move_to(d1_form)

        self.play(TransformMatchingTex(alt_d1_form, half_fact, path_arc=-PI / 2))
        self.wait()

    def get_grid(self, n_cols=10, width=FRAME_WIDTH - 1):
        cell = Square()
        cell.set_stroke(WHITE, 2)
        col = cell.get_grid(2, 1, buff=0)
        grid = col.get_grid(1, n_cols, buff=0)
        grid.set_width(width)
        grid.to_edge(UP, buff=1.5)
        grid.set_z_index(-1)
        return grid

    def get_volume_labels(self):
        config = dict(
            t2c=self.tex_to_color,
            font_size=36,
        )
        return VGroup(
            VGroup(
                Tex(tex, **config)
                for tex in texs
            )
            for texs in [
                get_boundary_volume_texs(),
                get_volume_texs(),
            ]
        )

    def show_derivative_and_integral(
        self,
        grid,
        dim,
        upper_buff=1.25,
        deriv_added_anims=[],
        int_added_anims=[],
        skip_derivative=False
    ):
        top_cell = grid[dim][0]
        low_cell = grid[dim][1]
        right_point = VGroup(top_cell, low_cell).get_right()

        down_arrow = Arrow(
            top_cell.get_right(),
            low_cell.get_right(),
            buff=SMALL_BUFF,
            thickness=5,
            path_arc=-180 * DEG
        )
        down_arrow.scale(0.8, about_point=right_point)

        up_arrow = down_arrow.copy().flip(RIGHT)

        deriv_label = Tex(R"{d / dr}", t2c=self.tex_to_color)
        deriv_label.next_to(up_arrow, RIGHT, SMALL_BUFF)
        int_label = Tex(R"\int \dots dr", t2c=self.tex_to_color)
        int_label.next_to(down_arrow, RIGHT, SMALL_BUFF)

        cover_rect = Rectangle(width=grid.get_width(), height=grid.get_height() + upper_buff)
        cover_rect.set_fill(BLACK, 0.85)
        cover_rect.set_stroke(width=0)
        cover_rect.next_to(right_point, RIGHT, buff=5e-3)
        cover_rect.shift(1e-2 * RIGHT)

        if skip_derivative:
            self.play(
                FadeIn(cover_rect),
                Write(down_arrow),
                Write(int_label),
                *int_added_anims
            )
            self.wait()
        else:
            self.play(LaggedStart(
                FadeIn(cover_rect),
                Write(up_arrow),
                Write(deriv_label),
                *deriv_added_anims,
                lag_ratio=0.5,
            ))
            self.wait()
            self.play(
                TransformMatchingTex(deriv_label, int_label, run_time=1),
                ReplacementTransform(up_arrow, down_arrow),
                *int_added_anims,
            )
            self.wait()
        self.play(
            FadeOut(down_arrow),
            FadeOut(int_label),
            FadeOut(cover_rect),
        )

    def get_knights_move_group(self, grid, d, colors=[GREEN, YELLOW], opacity=0.4):
        # Test
        cells = VGroup(grid[d - 2][1], grid[d][0]).copy()
        for cell, color in zip(cells, colors):
            cell.set_fill(color, opacity)

        arrow = Arrow(cells[0], cells[1], thickness=5, buff=-0.25)
        arrow.set_backstroke(BLACK, 5)

        return VGroup(cells, arrow)


class ShowCircleAreaDerivative(InteractiveScene):
    def construct(self):
        # Shrinking difference
        r = 2
        dr_tracker = ValueTracker(0.5)
        get_dr = dr_tracker.get_value

        circle = self.get_circle(r)
        dA_group = always_redraw(lambda: self.get_dA_group(r, get_dr()))

        self.add(circle)
        self.add(dA_group)

        # Shrink
        dr_tracker.set_value(0)
        self.play(dr_tracker.animate.set_value(0.5), run_time=2)
        self.wait()
        self.play(dr_tracker.animate.set_value(0.1), run_time=3)
        self.wait()

    def get_circle(self, r, fill_color=TEAL_E, fill_opacity=0.75, label="A"):
        result = VGroup()

        circle = Circle(radius=r)
        circle.set_fill(fill_color, fill_opacity)
        circle.set_stroke(WHITE, 0)
        result.add(circle)

        circle_label = Tex(label, font_size=72)
        circle_label.shift(0.5 * r * UP)
        result.add(circle_label)

        rad_line = Line(ORIGIN, r * RIGHT)
        rad_line.rotate(-45 * DEG, about_point=ORIGIN)
        r_label = Tex(R"r")
        r_label.next_to(rad_line.get_center(), UR, SMALL_BUFF)
        result.add(rad_line, r_label)

        return result

    def get_dA_group(self, r, dr, fill_color=RED_E, fill_opacity=0.5, label_color=WHITE):
        annulus = Annulus(r, r + dr)
        annulus.set_fill(fill_color, fill_opacity)
        annulus.set_stroke(width=0)
        line = Line(r * RIGHT, (r + dr) * RIGHT)
        dr_label = Tex(R"dr")
        dr_label.set_fill(label_color)
        dr_label.set_max_width(0.5 * line.get_width())
        dr_label.next_to(line, UP, buff=SMALL_BUFF)

        return VGroup(annulus, line, dr_label)


class CircleDerivativeFormula(InteractiveScene):
    def construct(self):
        # Test
        formulas = VGroup(
            Tex(tex, t2c={"dA": RED_D, "dr": BLUE})
            for tex in [
                R"dA = (2 \pi r) dr",
                R"{dA \over dr} = 2 \pi r",
            ]
        )
        formulas.scale(3)

        self.add(formulas[0])
        self.play(TransformMatchingTex(*formulas, path_arc=-90 * DEG))
        self.wait()


class BuildCircleWithCombinedAnnulusses(ShowCircleAreaDerivative):
    def construct(self):
        # Test
        dr = 0.1
        radius = 3.9
        rings = VGroup(
            Annulus(r, r + dr)
            for r in np.arange(0, radius, dr)
        )
        rings.set_submobject_colors_by_gradient(TEAL_E, BLUE_E)
        rings.set_stroke(BLACK, 0.5, 1)
        for ring in rings:
            ring.insert_n_curves(100)

        self.play(FadeIn(rings, lag_ratio=0.5, run_time=3))


class ShowSphereVolumeDerivative(ShowCircleAreaDerivative):
    def construct(self):
        # Set up
        frame = self.frame
        self.set_floor_plane("xz")

        r = 3
        dr_tracker = ValueTracker(0)
        get_dr = dr_tracker.get_value

        circle = self.get_circle(r, label="V", fill_opacity=1)
        dV_group = always_redraw(lambda: self.get_dA_group(r, get_dr()))

        inner_sphere = Sphere(radius=r)
        inner_sphere.set_color(TEAL_E, 1)
        inner_sphere.set_clip_plane(IN, r)
        sphere_mesh = SurfaceMesh(inner_sphere, resolution=(51, 26))
        sphere_mesh.set_stroke(WHITE, 1, 0.2)
        sphere_mesh.rotate(90 * DEG, RIGHT)

        def get_outer_sphere():
            sphere = Sphere(radius=r + get_dr())
            sphere.set_color(RED_E, 0.5)
            sphere.set_clip_plane(IN, 0)
            sphere.sort_faces_back_to_front(LEFT)
            return sphere

        outer_sphere = always_redraw(get_outer_sphere)

        self.add(circle)
        self.add(inner_sphere, sphere_mesh)

        frame.reorient(-75, -21, 0, ORIGIN, 8.73)
        self.play(
            frame.animate.reorient(42, -15, 0, ORIGIN, 8.73),
            inner_sphere.animate.set_clip_plane(IN, 0),
            run_time=3,
        )
        self.add(inner_sphere, circle, sphere_mesh)
        self.play(FadeIn(circle))
        self.wait()

        # Show dV
        self.add(outer_sphere)
        self.add(dV_group)
        sphere_mesh.add_updater(lambda m: m.set_width(2 * (r + get_dr())).move_to(ORIGIN))
        self.play(dr_tracker.animate.set_value(0.5), run_time=2)
        self.wait()
        self.play(dr_tracker.animate.set_value(0.1), run_time=3)
        self.wait()

        # Clean shrinking
        self.clear()
        self.add(outer_sphere, sphere_mesh, dV_group)
        dV_group.add_updater(lambda m: m[1:].set_opacity(0))
        dr_tracker.set_value(0.5)
        self.play(dr_tracker.animate.set_value(0.0), run_time=5)



class SphereDerivativeFormula(InteractiveScene):
    def construct(self):
        # Test
        formulas = VGroup(
            Tex(tex, t2c={"dV": RED_D, "{r}": BLUE})
            for tex in [
                R"dV = (4 \pi {r}^2) d{r}",
                R"{dV \over d{r}} = 4 \pi {r}^2",
            ]
        )
        formulas.scale(3)

        self.add(formulas[0])
        self.play(TransformMatchingTex(*formulas, path_arc=-90 * DEG))
        self.wait()


class SimpleLineWithEndPoints(InteractiveScene):
    def construct(self):
        # Test
        line = Line(LEFT, RIGHT)
        line.set_width(6)
        line.set_stroke(TEAL, 3)
        center_dot = Dot(ORIGIN, radius=0.05)

        brace = Brace(line, UP, buff=MED_SMALL_BUFF)
        brace.stretch(0.5, 0, about_edge=RIGHT)
        brace_label = brace.get_tex("r")

        end_points = Group(
            Group(Dot(), GlowDot()).move_to(point)
            for point in line.get_start_and_end()
        )
        end_points.set_color(YELLOW)

        self.add(line, center_dot)
        self.play(GrowFromCenter(brace), Write(brace_label))
        self.wait()
        self.play(FadeIn(end_points, lag_ratio=0.75))
        self.wait()


class ZAxisWithCircle(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        axes = ThreeDAxes((-2, 2), (-2, 2), (-2, 2))
        axes.set_width(4)
        sphere = Sphere(radius=1)
        sphere.always_sort_to_camera(self.camera)
        sphere.set_color(BLUE, 0.2)
        mesh = SurfaceMesh(sphere, resolution=(51, 26))
        mesh.set_stroke(WHITE, 2, 0.25)

        z_tracker = ValueTracker(0.6)
        get_z = z_tracker.get_value

        z_line = Line(axes.c2p(0, 0, -1), axes.c2p(0, 0, 1))
        z_line.set_stroke(GREEN, 10)
        z_line.apply_depth_test()
        z_dot = TrueDot(color=GREEN, radius=0.05)
        z_dot.make_3d()
        z_dot.add_updater(lambda m: m.move_to(axes.z_axis.n2p(get_z())))

        circle = Circle(radius=0.8)
        circle.apply_depth_test()
        circle.set_stroke(RED, 10)
        circle.add_updater(lambda m: m.set_width(2.01 * math.sqrt(1 - get_z()**2)))
        circle.add_updater(lambda m: m.move_to(axes.z_axis.n2p(get_z())))

        circle_shadow = VGroup()

        def update_shadow(shadow):
            if len(shadow) > 0 and abs(shadow[-1].get_z() - circle.get_z()) < 5e-3:
                return
            shadow.add(circle.copy().clear_updaters().set_stroke(opacity=0.15, width=2).set_width(2))
            return shadow

        circle_shadow.add_updater(update_shadow)

        frame.reorient(23, 70, 0, (-0.06, 0.05, -0.19), 3.02)
        self.add(axes, z_line, z_dot, circle, sphere, mesh)
        # self.add(circle_shadow)
        self.play(z_tracker.animate.set_value(0.9), run_time=4)
        self.play(z_tracker.animate.set_value(-0.9), run_time=8)
        self.play(z_tracker.animate.set_value(0.6), run_time=6)


class SeparateRingsOfLatitude(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        sphere = Sphere(radius=1)
        sphere.set_color(BLUE_E)
        sphere.always_sort_to_camera(self.camera)

        n_rings = 50
        rings = VGroup(
            Circle(radius=math.sqrt(1 - z**2)).move_to(z * OUT)
            for z in np.linspace(-1, 1, n_rings)
        )
        rings.set_stroke(BLUE, 2, 0.5)

        frame.reorient(4, 78, 0, (-0.03, 0.01, 0.03), 2.88)
        self.add(sphere)
        self.play(
            sphere.animate.set_opacity(0.2),
            LaggedStartMap(FadeIn, rings),
            run_time=3
        )
        self.wait()
        self.play(
            rings[n_rings // 2].animate.set_stroke(YELLOW, 3, 1),
            rings[:n_rings // 2].animate.set_stroke(opacity=0.25),
            rings[n_rings // 2 + 1:].animate.set_stroke(opacity=0.25),
        )
        self.wait()


# TODO, too much code redundancy below?
class CrossLineWithCircle(InteractiveScene):
    def construct(self):
        # Equation
        line = Line(DOWN, UP)
        line.set_stroke(GREEN_E, 8)
        circle = Circle()
        circle.set_stroke(RED, 3)

        group = Group(
            Tex(R"\partial B^3 = ", font_size=120),
            circle,
            Tex(R"\times", font_size=120),
            line,
        )
        group[0].shift(MED_SMALL_BUFF * UL)
        group.arrange(RIGHT)
        group[-1].shift(0.5 * RIGHT)

        self.add(group)

        # Formulas
        sphere_3d_form = Tex(R"x^2 + y^2 + z^2 = 1")
        sphere_3d_form.to_corner(UL)
        sphere_3d_form.next_to(group[0], UP, buff=MED_LARGE_BUFF)
        sphere_3d_form.shift(LEFT)

        circle_form = Tex(R"x^2 + y^2 = 1")
        circle_form.next_to(circle, DOWN)
        line_form = Tex(R"z^2 \le 1")
        line_form.next_to(line, DOWN)

        self.add(sphere_3d_form)
        self.add(circle_form)
        self.add(line_form)


class CrossDiskWithCircle(InteractiveScene):
    def construct(self):
        # Test
        disk = Circle()
        disk.set_fill(BLUE_E, 1)
        disk.set_stroke(WHITE, 1)
        circle = Circle()
        circle.set_stroke(RED, 3)

        group = VGroup(
            Tex(R"\partial B^4 = ", font_size=120),
            circle,
            Tex(R"\times", font_size=120),
            disk,
        )
        group[0].shift(SMALL_BUFF * UL)
        group.arrange(RIGHT)

        self.add(group)

        # Formulas
        sphere_4d_form = Tex(R"x^2 + y^2 + z^2 + w^2 = 1")
        sphere_4d_form.to_corner(UL)
        sphere_4d_form.next_to(group[0], UP, buff=LARGE_BUFF)
        sphere_4d_form.shift(LEFT)

        circle_form = Tex(R"x^2 + y^2 = 1")
        circle_form.next_to(circle, DOWN)
        disk_form = Tex(R"z^2 + w^2 \le 1")
        disk_form.next_to(disk, DOWN)

        self.add(sphere_4d_form)
        self.add(circle_form)
        self.add(disk_form)


class CrossBallWithCircle(InteractiveScene):
    def construct(self):
        # Equation
        ball = Sphere()
        ball.set_color(BLUE_E, 1)
        circle = Circle()
        circle.set_stroke(RED, 3)

        group = Group(
            Tex(R"\partial B^5 = ", font_size=120),
            circle,
            Tex(R"\times", font_size=120),
            ball,
        )
        group[0].shift(SMALL_BUFF * UL)
        group.arrange(RIGHT)

        self.add(group)

        # Formulas
        sphere_4d_form = Tex(R"x^2 + y^2 + z^2 + w^2 + v^2 = 1")
        sphere_4d_form.to_corner(UL)
        sphere_4d_form.next_to(group[0], UP, buff=LARGE_BUFF)
        sphere_4d_form.shift(LEFT)

        circle_form = Tex(R"x^2 + y^2 = 1")
        circle_form.next_to(circle, DOWN)
        ball_form = Tex(R"z^2 + w^2 + v^2 \le 1")
        ball_form.next_to(ball, DOWN)
        ball_form.shift(RIGHT)

        self.add(sphere_4d_form)
        self.add(circle_form)
        self.add(ball_form)


class ShowNumericalValues(InteractiveScene):
    def construct(self):
        # Set up
        axes = Axes((0, 25), (0, 6))
        axes.to_edge(UP, buff=LARGE_BUFF)
        axes.to_edge(LEFT, buff=MED_LARGE_BUFF)
        axes.x_axis.add_numbers()
        y_label = TexText("Volume of a\nunit ball")
        y_label.next_to(axes.y_axis.get_top(), RIGHT)
        x_label = Text("Dimension")
        x_label.next_to(axes.x_axis.get_end(), UP)
        x_label.shift_onto_screen()
        axes.add(x_label)
        axes.add(y_label)

        def func(n):
            return math.pi**(n / 2) / math.gamma(n/2 + 1)

        graph = axes.get_graph(func)
        graph.set_stroke(BLUE, 2)

        self.add(axes)

        # Add terms
        formulas = VGroup(
            Tex(s.split(" r")[0])
            for s in get_volume_texs()
        )
        v_lines = VGroup(
            axes.get_v_line_to_graph(x, graph, line_func=Line)
            for x in range(len(formulas))
        )
        v_lines.set_stroke(BLUE, 5)
        dots = VGroup(Dot(line.get_end()) for line in v_lines)
        dots.set_fill(BLUE_E)

        expressions = VGroup()
        for n, formula, dot in zip(it.count(), formulas, dots):
            formula.next_to(dot, RIGHT)
            approx = VGroup(
                Tex(R"\approx"),
                DecimalNumber(func(n))
            )
            approx.arrange(RIGHT)
            approx.next_to(formula, RIGHT)
            expressions.add(VGroup(formula, *approx))
            if n < 2:
                approx.set_fill(opacity=0)

        last_expression = VGroup()
        for v_line, dot, expression in zip(v_lines, dots, expressions):
            self.remove(last_expression)
            self.add(v_line, dot, expression)
            self.wait()
            last_expression = expression

        # Show general graph
        gen_formula = Tex(R"\pi^{n/2} \over (n/2)!")
        gen_formula.next_to(axes.i2gp(9, graph), UR)

        self.play(
            ShowCreation(graph),
            v_lines.animate.set_stroke(opacity=0.25),
            dots.animate.set_fill(opacity=0.25),
            FadeOut(last_expression[1:]),
            FadeTransform(last_expression[0], gen_formula),
            run_time=2
        )
        self.wait()
        self.play(
            self.frame.animate.reorient(0, 0, 0, (5.6, 0.15, 0.0), 14.77),
            run_time=3
        )
        self.wait()


class WriteB100Volume(InteractiveScene):
    def construct(self):
        # Test
        formula = Tex(R"B^{100} \rightarrow {\pi^{50} \over 50!} \approx 2.37 \times 10^{-40}")
        self.add(formula)


class SphereEquator(InteractiveScene):
    def construct(self):
        frame = self.frame
        radius = 3
        sphere = Sphere(radius=radius)
        equator = Circle(radius=radius)
        sphere.set_color(BLUE_E, 0.7)
        sphere.always_sort_to_camera(self.camera)
        equator.set_stroke(YELLOW, 3)
        equator.apply_depth_test()

        self.add(equator, sphere)


class Distributions(InteractiveScene):
    def construct(self):
        # Test
        import matplotlib.pyplot as plt
        import torch

        # List of vectors in some dimension, with many
        # more vectors than there are dimensions
        num_vectors = 100000
        vector_len = 10000

        big_matrix = np.random.normal(size=(num_vectors, vector_len))
        norms = np.linalg.norm(big_matrix, axis=1)
        big_matrix /= norms[:, np.newaxis]

        plt.style.use('dark_background')
        plt.hist(big_matrix[:, -1], bins=1000, range=(-1, 1))
        plt.show()


class UnitCircleAndSquare(InteractiveScene):
    def construct(self):
        radius = 2
        circle = Circle(radius=radius)
        circle.set_fill(BLUE, 0.2)
        circle.set_stroke(BLUE, 3)
        square = Square(side_length=radius)
        square.set_stroke(RED, 3)
        square.set_fill(RED, 0.25)
        square.move_to(circle.get_center(), DL)
        one_label = Integer(1)
        one_label.next_to(square.get_bottom(), UP, SMALL_BUFF)

        self.add(circle, square, one_label)


class UniSphereAndSquare(InteractiveScene):
    def construct(self):
        radius = 2
        sphere = Sphere(radius=radius)
        sphere.set_color(BLUE, 0.2)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere, resolution=(51, 26))
        mesh.set_stroke(WHITE, 1, 0.25)
        cube = VCube(side_length=radius)
        cube.move_to(ORIGIN, DL + IN)
        cube.set_color(RED, 0.25)
        cube.set_stroke(RED, 3)
        cube.deactivate_depth_test()
        one_label = Integer(1)
        one_label.rotate(90 * DEG, RIGHT)
        one_label.next_to(RIGHT, OUT, SMALL_BUFF)

        self.frame.reorient(25, 65, 0, (0.22, 0.13, 0.09), 4.66)
        self.add(sphere, mesh, cube, one_label)
