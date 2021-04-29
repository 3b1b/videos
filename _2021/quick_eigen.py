from manim_imports_ext import *

# Colors

COL1_COLOR = MAROON_B
COL2_COLOR = MAROON_C
LAMBDA1_COLOR = TEAL_A
LAMBDA2_COLOR = TEAL_D
MEAN_COLOR = BLUE_B
PROD_COLOR = BLUE_D


# Scenes

class Assumptions(TeacherStudentsScene):
    def construct(self):
        self.play(
            PiCreatureSays(self.teacher, TexText("I'm assuming you know\\\\ what eigenvalues are.")),
            self.get_student_changes(
                "erm", "happy", "tease",
                look_at_arg=ORIGIN,
            )
        )
        self.play(self.students[0].animate.change("guilty").look(LEFT))
        self.wait()

        eigen_expression = Tex("""
            \\text{det}\\left( \\left[ \\begin{array}{cc}
                3 - \\lambda & 1 \\\\
                4 & 1 - \\lambda
            \\end{array} \\right] \\right)
        """)
        eigen_expression.move_to(self.hold_up_spot, DOWN)
        eigen_expression.to_edge(RIGHT, buff=2)
        VGroup(eigen_expression[0][7], eigen_expression[0][12]).set_color(TEAL)
        cross = Cross(eigen_expression)
        cross.set_stroke(RED, width=(1, 5, 5, 1))

        self.play(
            RemovePiCreatureBubble(self.teacher, target_mode="raise_right_hand"),
            FadeIn(eigen_expression, UP),
            self.students[1].animate.change("hesitant"),
            self.students[2].animate.change("sassy"),
        )
        self.wait()
        self.play(
            ShowCreation(cross),
            self.teacher.animate.change("tease", cross),
            self.students[1].animate.change("pondering", cross),
            self.students[2].animate.change("confused", cross),
        )
        self.wait(3)

        self.embed()


class PreviousVideoWrapper(Scene):
    def construct(self):
        self.add(FullScreenRectangle())

        title = Text("Video introducing eigenvalues", font_size=72)
        screen = ScreenRectangle(height=6)
        screen.set_fill(BLACK, 1)
        screen.set_stroke(BLUE, 1)
        title.match_width(screen)
        title.to_edge(UP)
        screen.next_to(title, DOWN)

        self.add(screen)
        self.play(FadeIn(title, UP))
        self.wait(2)
        self.play(
            FadeOut(title, UP),
            screen.animate.set_height(7).center(),
        )
        self.wait()


class SneakierEigenVector(ExternallyAnimatedScene):
    pass


class TypicalComputation(Scene):
    def construct(self):
        # Task
        words, mat = task = VGroup(
            Text("Find the eigenvalues of ", t2c={"eigenvalues": TEAL}),
            IntegerMatrix([[3, 1], [4, 1]]).set_height(1),
        )
        task.arrange(RIGHT, buff=MED_LARGE_BUFF)
        task.to_edge(UP)
        self.add(task)

        # Top line of the computation
        det_expression = Tex("""
            \\text{det}\\left( \\left[ \\begin{array}{cc}
                3 - \\lambda & 1 \\\\
                4 & 1 - \\lambda
            \\end{array} \\right] \\right)
        """)[0]
        lambdas = VGroup(det_expression[7], det_expression[12])
        lambdas.set_color(TEAL)
        t0, t1, t2, t3 = terms = VGroup(
            det_expression[5:8],
            det_expression[8:9],
            det_expression[9:10],
            det_expression[10:13],
        ).copy()
        p_height = terms[0].get_height() * 1.5
        for term in terms:
            parens = Tex("(", ")")
            parens.set_height(p_height)
            parens[0].next_to(term, LEFT, 0.5 * SMALL_BUFF)
            parens[1].next_to(term, RIGHT, 0.5 * SMALL_BUFF)
            term.parens = parens
            term.parens.set_opacity(0)
            term.add(term.parens)

        eq = Tex("=")
        eq.next_to(det_expression, RIGHT)
        rhs = VGroup(
            t0.copy(), t3.copy(), Tex("-"), t1.copy(), t2.copy()
        )
        rhs.arrange(RIGHT)
        rhs.next_to(eq, RIGHT)
        rhs.set_opacity(1)
        VGroup(det_expression, terms, eq, rhs).next_to(task, DOWN, LARGE_BUFF)

        movers = VGroup(det_expression[5], det_expression[10])
        movers.save_state()
        movers[0].move_to(det_expression[6])
        movers[1].move_to(det_expression[11])

        self.play(
            TransformFromCopy(mat.get_brackets(), VGroup(*(det_expression[i] for i in [4, 13]))),
            TransformFromCopy(mat.get_entries(), VGroup(*(det_expression[i] for i in [5, 8, 9, 10]))),
            FadeTransform(
                mat.get_brackets().copy().set_opacity(0),
                VGroup(*(det_expression[i] for i in [0, 1, 2, 3, 14]))
            ),
            run_time=1
        )
        self.play(
            Write(VGroup(*(det_expression[i] for i in [6, 7, 11, 12]))),
            Restore(movers),
        )
        self.wait()
        self.add(det_expression)

        self.play(
            FadeIn(eq),
            TransformFromCopy(VGroup(t0, t3), rhs[:2]),
        )
        self.play(
            FadeIn(rhs[2]),
            TransformFromCopy(VGroup(t1, t2), rhs[3:])
        )
        self.wait()

        # Line 2
        eq2 = eq.copy()
        eq2.shift(DOWN)
        self.add(eq2)
        rhs2 = Tex("\\left( 3 - 4\\lambda + \\lambda^2 \\right) - 4")[0]
        rhs2.next_to(eq2, RIGHT)
        VGroup(rhs2[4], rhs2[6]).set_color(TEAL)

        top_terms = VGroup(
            VGroup(rhs[0][0], rhs[0][2]),
            VGroup(rhs[1][0], rhs[1][2]),
        )
        alt_mid = Tex("-3\\lambda", tex_to_color_map={"\\lambda": TEAL})
        alt_mid.move_to(rhs2[2:5], DL)
        bottom_terms = VGroup(rhs2[1], alt_mid, rhs2[2:5], rhs2[5:8])
        for pair, bt in zip(it.product(*top_terms), bottom_terms):
            rects = VGroup(*(SurroundingRectangle(t, buff=SMALL_BUFF) for t in pair))
            self.add(rects)
            self.add(bt)
            self.wait(0.5)
            if bt is alt_mid:
                self.remove(bt)
            self.remove(rects)
        self.play(
            FadeIn(VGroup(rhs2[0], rhs2[8])),
            FadeTransform(rhs[2:].copy(), rhs2[9:])
        )
        self.wait()

        # Line 3
        eq3 = eq2.copy().shift(DOWN)
        rhs3 = Tex("\\lambda^2 - 4 \\lambda - 1")[0]
        rhs3.next_to(eq3, RIGHT)
        VGroup(rhs3[0], rhs3[4]).set_color(TEAL)

        kw = {"path_arc": 45 * DEGREES}
        self.play(LaggedStart(
            TransformFromCopy(eq2, eq3, **kw),
            Transform(rhs2[1].copy(), rhs3[6].copy(), remover=True, **kw),
            TransformFromCopy(rhs2[9:11], rhs3[5:], **kw),
            TransformFromCopy(rhs2[2:5], rhs3[2:5], **kw),
            TransformFromCopy(rhs2[6:8], rhs3[0:2], **kw),
            run_time=1.5, lag_ratio=0.02,
        ))
        self.wait()

        # Characteristic polynomial
        brace = Brace(rhs3, DOWN)
        char_poly = VGroup(
            Text("Characteristic polynomial of", font_size=30, fill_color=BLUE),
            mat.copy()
        )
        char_poly.arrange(RIGHT)
        char_poly.next_to(brace, DOWN)
        char_poly.shift_onto_screen()

        self.play(
            GrowFromCenter(brace),
            FadeIn(char_poly, DOWN),
        )
        self.wait()

        # Roots
        equals_zero = Tex("= 0")
        equals_zero.next_to(rhs3, RIGHT)
        root_words = Tex(
            "\\lambda_1, \\lambda_2 \\,=\\, \\text{roots}",
            tex_to_color_map={
                "\\lambda_1": TEAL_C,
                "\\lambda_2": TEAL_B,
                "=": WHITE,
            }
        )
        new_rhs3 = VGroup(rhs3, equals_zero)
        root_words.next_to(brace, DOWN)
        root_words.match_x(new_rhs3)
        self.play(
            FadeIn(root_words, DOWN),
            FadeOut(char_poly, DOWN),
            brace.animate.become(Brace(new_rhs3)),
            Write(equals_zero),
        )
        self.wait()

        # Quadratic formula
        formula = Tex("\\frac{4 \\pm \\sqrt{4^2 - 4(1)(-1)}}{2}")
        formula2 = Tex("=\\frac{4 \\pm \\sqrt{20}}{2}")
        formula3 = Tex("= 2 \\pm \\sqrt{5}")
        formula.move_to(root_words[-1], LEFT)
        formula.shift(0.5 * DL)

        self.play(
            TransformMatchingShapes(rhs3.copy(), formula),
            FadeOut(root_words[-1]),
            root_words[:-1].animate.shift(0.5 * DL),
        )
        self.wait()
        solution = VGroup(root_words[:-1], formula)
        self.play(
            solution.animate.shift(formula2.get_width() * LEFT),
        )
        formula2.next_to(formula, RIGHT)
        self.play(FadeIn(formula2))
        self.wait()
        solution.add(formula2)
        self.play(
            solution.animate.shift(formula3.get_width() * LEFT),
        )
        formula3.next_to(formula2, RIGHT)
        self.play(FadeIn(formula3))
        self.wait()

        # Straight line
        full_rect = FullScreenFadeRectangle()
        arrow = Arrow(mat, formula3, thickness=0.05)
        arrow.set_fill(YELLOW)

        self.add(full_rect, task, formula3)
        self.play(FadeIn(full_rect))
        self.play(GrowArrow(arrow))
        self.wait()


class TweakDiagonalValue(ExternallyAnimatedScene):
    pass


class DetEquationLineOfReasoning(ExternallyAnimatedScene):
    pass


class OutlineThreeFacts(Scene):
    def construct(self):
        # Matrix to lambdas
        mat = Matrix([["a", "b"], ["c", "d"]], v_buff=0.8, h_buff=0.8)
        mat.set_column_colors(COL1_COLOR, COL2_COLOR)

        lambdas = Tex("\\lambda_1", "\\,,\\,", "\\lambda_2")
        lambdas[0].set_color(LAMBDA1_COLOR)
        lambdas[2].set_color(LAMBDA2_COLOR)
        arrow = Vector(1.5 * RIGHT)
        group = VGroup(mat, arrow, lambdas)
        group.arrange(RIGHT)
        arrow_label = Text("Quick?", font_size=24)
        arrow_label.next_to(arrow, UP, buff=0)

        self.add(mat)
        self.play(
            GrowArrow(arrow),
            Write(arrow_label, run_time=1),
            LaggedStart(*(
                AnimationGroup(*(
                    Transform(entry, lambdas[i])
                    for entry in mat.get_entries().deepcopy()
                ))
                for i in [0, 2]
            ), lag_ratio=0.3),
            FadeIn(lambdas[1]),
        )
        self.clear()
        self.add(group, arrow_label)
        self.wait()

        # Three steps
        indices = VGroup(*(Text(str(i) + ")", font_size=48) for i in range(1, 4)))
        indices.set_color(GREY_B)
        indices.arrange(DOWN, aligned_edge=LEFT, buff=2)
        indices.to_edge(LEFT)

        group.generate_target()
        group.target[1].rotate(-90 * DEGREES)
        group.target[1].scale(0.5)
        group.target.arrange(DOWN)
        group.target.to_corner(DR)

        self.play(
            LaggedStartMap(FadeIn, indices, shift=0.25 * UP, lag_ratio=0.3),
            FadeOut(arrow_label),
            MoveToTarget(group),
        )
        self.wait()

        # Trace
        tr_mat = mat.deepcopy()
        tr = Tex("\\text{tr}", "\\Big(", "\\Big)", font_size=60)
        tr[1:].match_height(tr_mat, stretch=True)
        tr.set_submobjects([*tr[:-1], tr_mat, tr[-1]])
        tr.arrange(RIGHT, buff=SMALL_BUFF)
        tr.next_to(indices[0], RIGHT, MED_LARGE_BUFF)

        tr_rects = VGroup(
            SurroundingRectangle(tr_mat.get_entries()[0]),
            SurroundingRectangle(tr_mat.get_entries()[3]),
        )
        tr_rects.set_color(BLUE_C)
        moving_tr_rects = tr_rects.copy()
        moving_tr_rects.generate_target()

        tex_kw = {
            "tex_to_color_map": {
                "a": COL1_COLOR,
                "b": COL2_COLOR,
                "c": COL1_COLOR,
                "d": COL2_COLOR,
                "=": WHITE,
                "\\lambda_1": LAMBDA1_COLOR,
                "\\lambda_2": LAMBDA2_COLOR,
            }
        }
        tr_rhs = Tex("= a + d = \\lambda_1 + \\lambda_2", **tex_kw)
        tr_rhs.next_to(tr, RIGHT)

        for term, rect in zip(tr_rhs[1:4:2], moving_tr_rects.target):
            rect.move_to(term)

        self.play(
            TransformFromCopy(mat, tr_mat),
            Write(VGroup(*tr[:2], tr[-1])),
        )
        self.play(LaggedStart(*map(ShowCreation, tr_rects)))
        tr.add(tr_rects)
        self.play(
            MoveToTarget(moving_tr_rects),
            TransformFromCopy(tr_mat.get_entries()[0], tr_rhs.get_part_by_tex("a")),
            TransformFromCopy(tr_mat.get_entries()[3], tr_rhs.get_part_by_tex("d")),
            FadeIn(tr_rhs[0:4:2]),
        )
        self.play(FadeOut(moving_tr_rects))
        self.wait()
        self.play(
            TransformMatchingShapes(lambdas.copy(), tr_rhs[4:]),
        )
        self.wait()

        # Mean of eigenvalues
        half = Tex("1 \\over 2")
        half.move_to(tr, LEFT)
        tr.generate_target()
        tr.target.next_to(half, RIGHT, SMALL_BUFF)
        new_tr_rhs = Tex("= {a + d \\over 2} = {\\lambda_1 + \\lambda_2 \\over 2}", **tex_kw)
        new_tr_rhs.next_to(tr.target, RIGHT)

        self.play(
            GrowFromCenter(half),
            MoveToTarget(tr),
            TransformMatchingShapes(tr_rhs, new_tr_rhs),
        )
        self.wait()

        # Determinant
        det_mat = mat.deepcopy()
        det = Tex("\\text{det}", "\\Big(", "\\Big)", font_size=60)
        det[1:].match_height(det_mat, stretch=True)
        det.set_submobjects([*det[:-1], det_mat, det[-1]])
        det.arrange(RIGHT, buff=SMALL_BUFF)
        det.next_to(indices[1], RIGHT, MED_LARGE_BUFF)

        det_rhs = Tex("= ad - bc = \\lambda_1 \\lambda_2", **tex_kw)
        det_rhs.next_to(det, RIGHT)
        self.play(
            TransformFromCopy(mat, det_mat),
            Write(VGroup(*det[:2], det[-1])),
        )

        path = VMobject()
        path.set_points_smoothly([
            det_mat.get_corner(UL),
            *[
                det_mat.get_entries()[i].get_center()
                for i in [0, 3, 1, 2]
            ],
            det_mat.get_corner(DL),
        ])
        path.set_stroke(BLUE, 3)

        self.add(path)
        self.play(
            VShowPassingFlash(path, time_width=1, run_time=3, rate_function=linear),
            LaggedStart(
                Animation(Mobject(), remover=True),
                FadeIn(det_rhs[:3]),
                FadeIn(det_rhs[3:6]),
                lag_ratio=0.7,
            )
        )
        self.wait()
        self.play(
            TransformMatchingShapes(lambdas.copy(), det_rhs[6:])
        )
        self.wait()

        # Mean and product
        eq_m = TexText("=", " $m$", "\\quad (mean)")
        eq_m[1].set_color(MEAN_COLOR)
        eq_m.next_to(new_tr_rhs, RIGHT)
        eq_p = TexText("=", " $p$", "\\quad (product)")
        eq_p[1].set_color(PROD_COLOR)
        eq_p.next_to(det_rhs, RIGHT)

        form_lhs = lambdas.copy()
        form_rhs = Tex("= {m} \\pm \\sqrt{\\,{m}^2 - {p}}", tex_to_color_map={"{m}": MEAN_COLOR, "{p}": PROD_COLOR})
        form_lhs.next_to(indices[2], RIGHT)
        form_rhs.next_to(form_lhs, RIGHT)

        third_point_placeholder = Text("(We'll get to this...)", font_size=30)
        third_point_placeholder.set_fill(GREY_C)
        third_point_placeholder.next_to(indices[2], RIGHT, MED_LARGE_BUFF)

        form = VGroup(indices[2], form_lhs, form_rhs)
        rect = SurroundingRectangle(VGroup(form), buff=MED_SMALL_BUFF)

        randy = Randolph(height=2)
        randy.next_to(rect, RIGHT)
        randy.to_edge(DOWN)

        self.play(
            FadeIn(third_point_placeholder),
            VFadeIn(randy),
            randy.animate.change("erm", third_point_placeholder)
        )
        self.play(Blink(randy))
        self.wait()
        self.play(
            randy.animate.change("thinking", eq_m),
            Write(eq_m)
        )
        self.play(
            randy.animate.look_at(eq_p),
            Write(eq_p),
            mat.animate.set_height(1, about_edge=DOWN)
        )
        self.play(Blink(randy))
        self.wait()

        # Example matrix
        ex_mat = IntegerMatrix([[8, 4], [2, 6]])
        ex_mat.set_height(1.25)
        ex_mat.set_column_colors(COL1_COLOR, COL2_COLOR)
        ex_mat.next_to(randy, RIGHT, aligned_edge=UP)

        kw = {"tex_to_color_map": {"m": MEAN_COLOR, "p": PROD_COLOR, "=": WHITE, "-": WHITE}}
        m_eq = Tex("m = 7", **kw)
        p_eq1 = Tex("p = 48 - 8", **kw)
        p_eq2 = Tex("p = 40", **kw)

        for mob in (m_eq, p_eq1, p_eq2):
            mob.next_to(ex_mat, RIGHT, buff=MED_LARGE_BUFF)
        m_eq.shift(0.5 * UP)
        VGroup(p_eq1, p_eq2).shift(0.5 * DOWN)

        diag_rects = VGroup(
            SurroundingRectangle(ex_mat.get_entries()[0]),
            SurroundingRectangle(ex_mat.get_entries()[3]),
        )
        off_diag_rects = VGroup(
            SurroundingRectangle(ex_mat.get_entries()[1]),
            SurroundingRectangle(ex_mat.get_entries()[2]),
        )
        diag_rects.set_color(PROD_COLOR)
        off_diag_rects.set_color(PROD_COLOR)
        mean_rect = SurroundingRectangle(m_eq[2])
        mean_rect.set_color(MEAN_COLOR)

        tr_rect = SurroundingRectangle(VGroup(indices[0], tr, eq_m)).set_stroke(MEAN_COLOR)
        det_rect = SurroundingRectangle(VGroup(indices[1], det, eq_p)).set_stroke(PROD_COLOR)

        self.play(
            randy.animate.change("pondering", ex_mat),
            FadeIn(ex_mat, RIGHT),
            FadeOut(group, RIGHT),
        )
        self.play(Blink(randy))
        self.wait()

        self.play(FadeIn(tr_rect))
        self.play(
            Write(m_eq[:2]),
            LaggedStartMap(ShowCreation, diag_rects, lag_ratio=0.5, run_time=1),
            randy.animate.look_at(m_eq),
        )
        self.wait()
        self.remove(diag_rects)
        self.play(
            TransformFromCopy(diag_rects, mean_rect),
            FadeTransform(ex_mat.get_entries()[0].copy(), m_eq[2]),
            FadeTransform(ex_mat.get_entries()[3].copy(), m_eq[2]),
        )
        self.play(Blink(randy))
        self.wait()

        self.play(FadeOut(tr_rect), FadeIn(det_rect))
        self.play(
            Write(p_eq1[:2]),
            randy.animate.change("hesitant", p_eq1),
            FadeOut(mean_rect),
        )

        path.replace(ex_mat.get_entries(), dim_to_match=1)
        self.play(VShowPassingFlash(path, time_width=1, run_time=2))
        self.wait()
        self.play(
            FadeIn(diag_rects),
            FadeIn(p_eq1[2]),
        )
        self.play(
            FadeOut(diag_rects),
            FadeIn(off_diag_rects),
            FadeIn(p_eq1[3:]),
        )
        self.play(
            FadeOut(off_diag_rects),
        )
        self.play(
            randy.animate.change("tease", p_eq2),
            FadeOut(p_eq1),
            FadeIn(p_eq2),
        )
        self.play(FadeOut(det_rect))
        self.wait()

        # Let other stuff happen up top
        full_rect = FullScreenFadeRectangle()
        full_rect.set_fill(BLACK, 1)

        ex = VGroup(ex_mat, m_eq, p_eq2)

        self.add(full_rect, randy, ex)
        self.play(
            FadeIn(full_rect),
            randy.animate.change("pondering", ORIGIN)
        )
        for x in range(10):
            if random.random() < 0.5:
                self.play(Blink(randy))
            else:
                self.wait()

        # Show final formula
        ex_rect = SurroundingRectangle(ex, buff=0.35)
        ex_rect.set_stroke(GREY_A)
        ex_rect.set_fill(interpolate_color(GREY_E, BLACK, 0.25))
        ex_rect.set_opacity(0)
        ex_group = VGroup(ex_rect, ex)
        ex_group.generate_target()
        ex_group.target.set_height(1.5)
        ex_group.target.to_corner(DR)
        ex_group.target[0].set_opacity(1)

        self.play(
            FadeOut(full_rect),
            FadeOut(randy),
            MoveToTarget(ex_group)
        )

        self.play(
            FadeIn(form_lhs, 0.25 * UP),
            FadeOut(third_point_placeholder, 0.25 * UP)
        )
        self.play(TransformMatchingShapes(
            VGroup(m_eq[0], p_eq2[0]).copy(),
            form_rhs,
        ))
        self.play(ShowCreation(rect))
        self.wait()


class ShowSquishingAndStretching(Scene):
    def construct(self):
        plane = NumberPlane()

        self.add(plane)

        self.embed()


class MeanProductExample(Scene):
    def construct(self):
        # Number line and midpoint
        number_line = NumberLine((0, 14))
        number_line.add_numbers()
        number_line.set_width(FRAME_WIDTH - 1)
        number_line.to_edge(UP, buff=1.5)
        nl = number_line

        m = 7
        m_dot = Dot(nl.n2p(m))
        m_dot.set_color(MEAN_COLOR)
        m_label = Tex("m", color=MEAN_COLOR)
        m_label.next_to(m_dot, UP, buff=MED_SMALL_BUFF)
        label7 = Tex("7", color=MEAN_COLOR)

        # Distance tracking
        d_tracker = ValueTracker(4)

        def get_l1_point():
            return nl.n2p(m - d_tracker.get_value())

        def get_l2_point():
            return nl.n2p(m + d_tracker.get_value())

        l1_dot, l2_dot = (Dot(color=TEAL) for x in range(2))
        l1_label = Tex("\\lambda_1", color=LAMBDA1_COLOR)
        l2_label = Tex("\\lambda_2", color=LAMBDA2_COLOR)
        l1_arrow, l2_arrow = (Arrow(color=WHITE) for x in range(2))

        l1_dot.add_updater(lambda m: m.move_to(get_l1_point()))
        l2_dot.add_updater(lambda m: m.move_to(get_l2_point()))
        always(l1_label.next_to, l1_dot, UP, buff=MED_SMALL_BUFF)
        always(l2_label.next_to, l2_dot, UP, buff=MED_SMALL_BUFF)
        m_label.match_y(l1_label)
        l1_arrow.add_updater(lambda m: m.set_points_by_ends(
            m_label.get_left() + SMALL_BUFF * LEFT, l1_label.get_right() + SMALL_BUFF * RIGHT,
        ))
        l2_arrow.add_updater(lambda m: m.set_points_by_ends(
            m_label.get_right() + SMALL_BUFF * RIGHT, l2_label.get_left() + SMALL_BUFF * LEFT,
        ))

        minus_d = Tex("-d")
        plus_d = Tex("+d")
        always(minus_d.next_to, l1_arrow, UP, SMALL_BUFF)
        always(plus_d.next_to, l2_arrow, UP, SMALL_BUFF)
        plus_qm = Tex("+??")
        minus_qm = Tex("-??")
        always(plus_qm.move_to, plus_d)
        always(minus_qm.move_to, minus_d)

        VGroup(plus_d, minus_d).set_opacity(0)

        label7.move_to(m_label)
        self.add(number_line)
        self.add(m_dot)
        self.add(label7)
        self.add(l1_dot)
        self.add(l2_dot)
        self.add(l1_label)
        self.add(l2_label)
        self.add(l1_arrow)
        self.add(l2_arrow)
        self.add(minus_d)
        self.add(plus_d)
        self.add(minus_qm)
        self.add(plus_qm)

        d_tracker.add_updater(lambda m: m.set_value(4 - 2.5 * np.sin(0.25 * self.time)))
        self.add(d_tracker)
        self.wait(10)
        self.play(
            UpdateFromAlphaFunc(
                Mobject(),
                lambda m, a: VGroup(plus_d, minus_d).set_opacity(a),
                remover=True
            ),
            UpdateFromAlphaFunc(
                Mobject(),
                lambda m, a: VGroup(plus_qm, minus_qm).set_opacity(1 - a),
                remover=True
            ),
        )
        self.remove(plus_qm, minus_qm)
        self.wait(5)

        # Write the product
        kw = {"tex_to_color_map": {"\\,7": MEAN_COLOR, "d": GREY_A, "=": WHITE, "-": WHITE}}
        texs = VGroup(*(Tex(tex, **kw) for tex in [
            "(\\,7 + d\\,)(\\,7 - d\\,)",
            "\\,7^2 - d^2 = ",
            "40 =",
            "d^2 = \\,7^2 - 40",
            "d^2 = 9",
            "d = 3",
        ]))
        texs[:3].arrange(LEFT)
        texs[1].align_to(texs[2], DOWN)
        texs[:3].next_to(nl, DOWN, MED_LARGE_BUFF).to_edge(LEFT, buff=LARGE_BUFF)
        for t1, t2 in zip(texs[2:], texs[3:]):
            t2.next_to(t1, DOWN, MED_LARGE_BUFF)
            t2.shift((t1.get_part_by_tex("=").get_x() - t2.get_part_by_tex("=").get_x()) * RIGHT)

        texs[0].save_state()
        texs[0].move_to(texs[1], UL)

        self.play(Write(VGroup(texs[2], texs[0])))
        self.wait(3)
        self.play(Restore(texs[0]))
        self.play(TransformMatchingShapes(texs[0].copy(), texs[1], path_arc=45 * DEGREES))
        self.wait(3)

        self.play(LaggedStart(
            TransformFromCopy(texs[2][0], texs[3][6], path_arc=-45 * DEGREES),
            TransformFromCopy(texs[2][1], texs[3][2]),
            TransformFromCopy(texs[1][:3], texs[3][3:6]),
            TransformFromCopy(texs[1][3:5], texs[3][0:2], path_arc=45 * DEGREES),
            lag_ratio=0.1
        ))
        self.wait(2)
        self.play(FadeIn(texs[4], DOWN))
        self.wait()
        self.play(FadeIn(texs[5], DOWN))

        # Show final d geometrically
        d_tracker.clear_updaters()
        self.play(d_tracker.animate.set_value(3), rate_func=rush_into)
        minus_d.clear_updaters()
        plus_d.clear_updaters()

        plus_3 = Tex("+3").move_to(plus_d)
        minus_3 = Tex("-3").move_to(minus_d)
        self.play(
            LaggedStart(
                FadeOut(minus_d, 0.25 * UP),
                FadeOut(plus_d, 0.25 * UP),
                lag_ratio=0.25
            ),
            LaggedStart(
                FadeIn(minus_3, 0.25 * UP),
                FadeIn(plus_3, 0.25 * UP),
                lag_ratio=0.25
            )
        )
        self.wait()

        # Highlight solutions
        rects = VGroup(
            SurroundingRectangle(VGroup(l1_label, nl.numbers[4])),
            SurroundingRectangle(VGroup(l2_label, nl.numbers[10])),
        )
        self.play(LaggedStartMap(ShowCreation, rects), lag_ratio=0.3)
        self.wait(2)
        self.play(LaggedStartMap(FadeOut, rects), lag_ratio=0.3)

        # Replace with general variables
        ms = VGroup(m_label.copy())
        sevens = VGroup(label7)
        ps = VGroup()
        fourties = VGroup()
        for tex in texs[:-2]:
            for fourty in tex.get_parts_by_tex("40"):
                fourties.add(fourty)
                ps.add(Tex("p").set_color(PROD_COLOR).move_to(fourty).shift(0.1 * DOWN))
            for seven in tex.get_parts_by_tex("7"):
                sevens.add(seven)
                m = Tex("m").set_color(MEAN_COLOR)
                m.scale(0.95)
                m.move_to(seven, DR)
                m.shift(0.04 * RIGHT)
                ms.add(m)
        ps[0].shift(0.05 * RIGHT)
        ps[1].shift(0.1 * LEFT)

        for g1, g2 in ((sevens, ms), (fourties, ps)):
            self.play(
                LaggedStartMap(FadeOut, g1, shift=0.25 * UP, lag_ratio=0.3),
                LaggedStartMap(FadeIn, g2, shift=0.25 * UP, lag_ratio=0.3),
                texs[-2:].animate.set_opacity(0)
            )
        self.remove(texs[-2:])
        self.wait()
        self.play(FlashUnder(texs[3]))
        self.wait()

        plus_form, minus_form = [
            Tex(
                c + "\\sqrt{\\,m^2 - p}",
                tex_to_color_map={"m": MEAN_COLOR, "p": PROD_COLOR},
                font_size=24,
            ).move_to(d, DOWN)
            for c, d in [("+", plus_d), ("-", minus_d)]
        ]
        pre_group = VGroup(*texs[3][4:6], ms[-1], ps[-1])
        self.play(
            TransformMatchingShapes(pre_group.copy(), minus_form),
            TransformMatchingShapes(pre_group.copy(), plus_form),
            FadeOut(VGroup(minus_3, plus_3), shift=0.25 * UP),
        )
        self.wait()


class JingleAnimation(Scene):
    def construct(self):
        form = Tex("m \\pm \\sqrt{m^2 - p}")[0]
        form.set_height(2)
        m, old_pm, sqrt, root, m2, squared, minus, p = form
        VGroup(m, m2).set_color(MEAN_COLOR)
        VGroup(p).set_color(PROD_COLOR)
        m.refresh_bounding_box()  # Why?
        p.refresh_bounding_box()  # Why?

        pm = VGroup(Tex("+"), Tex("-"))
        pm.arrange(DOWN, buff=0)
        pm.replace(old_pm)
        pm.save_state()
        pm.scale(1.5)
        pm.arrange(DOWN, buff=2)

        mean = Text("mean", color=WHITE).next_to(m, DOWN)
        product = Text("product", color=WHITE).next_to(p, DOWN)
        plus_word = Text("plus", color=YELLOW).next_to(pm, UP)
        minus_word = Text("minus", color=YELLOW).next_to(pm, DOWN)

        def snap(t):
            return t**5

        self.play(FadeIn(m, scale=0.5, rate_func=snap))
        self.add(mean)
        self.wait(0.4)
        self.add(pm[0], plus_word)
        self.wait(0.2)
        self.add(pm[1], minus_word)
        self.wait(0.2)
        self.play(
            Restore(pm, rate_func=smooth, run_time=0.5),
            FadeOut(VGroup(plus_word, minus_word, mean), run_time=0.5),
        )

        # new_sqrt = VMobject()
        # new_sqrt.set_points(
        #     sqrt.get_points()[::-3],
        #     root.get_points()[::-3],
        # )

        sqrt_outline = sqrt.copy()
        sqrt_outline.set_stroke(YELLOW, 5)
        sqrt_outline.set_fill(opacity=0)
        sqrt_outline.insert_n_curves(100)
        root.save_state()
        root.stretch(0, 0, about_edge=LEFT)
        self.play(
            FadeIn(sqrt, rate_func=squish_rate_func(smooth, 0.25, 0.75)),
            VShowPassingFlash(sqrt_outline, time_width=2),
            Restore(root, rate_func=squish_rate_func(snap, 0.25, 1)),
            run_time=1,
        )
        self.wait(0.5)

        self.play(
            TransformFromCopy(m, m2, path_arc=-120 * DEGREES, rate_func=smooth, run_time=0.5),
        )
        # squared.save_state()
        # squared.rotate(90 * DEGREES).scale(0.5).set_opacity(0)
        # self.play(Restore(squared, run_time=0.25))
        self.play(FadeTransform(m2.copy(), squared, run_time=0.25))
        minus.save_state()
        minus.stretch(0, 0, about_edge=LEFT)
        self.play(Restore(minus), run_time=0.7)
        self.wait(0.25)
        self.play(FadeIn(p, scale=0.8, rate_func=snap, run_time=0.2))
        self.add(product)
        self.play(Flash(p, flash_radius=0.8 * p.get_height(), run_time=0.5))
        self.remove(product)
        self.wait()
