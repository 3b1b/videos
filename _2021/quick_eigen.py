from manim_imports_ext import *

# Colors

COL_COLORS = [MAROON_B, MAROON_C]
EIGEN_COLORS = [TEAL_A, TEAL_D]
MEAN_COLOR = BLUE_B
PROD_COLOR = BLUE_D


def det_path_anim(matrix, run_time=2):
    path = VMobject()
    path.set_points_smoothly([
        matrix.get_corner(UL),
        *[
            matrix.get_entries()[i].get_center()
            for i in [0, 3, 1, 2]
        ],
        matrix.get_corner(DL),
    ])
    path.set_stroke(BLUE, 3)

    return VShowPassingFlash(path, time_width=1, run_time=run_time, rate_function=linear)


def get_diag_rects(matrix, color=MEAN_COLOR, off_diagonal=False):
    if off_diagonal:
        entries = matrix.get_entries()[1:3]
    else:
        entries = matrix.get_entries()[0::3]
    return VGroup(*(
        SurroundingRectangle(entry, buff=SMALL_BUFF, color=color)
        for entry in entries
    ))


def get_prism(verts, depth=2):
    result = VGroup()
    result.add(Polygon(*verts))
    zv = depth * OUT
    for v1, v2 in zip(verts, [*verts[1:], verts[0]]):
        result.add(Polygon(v1, v2, v2 + zv, v1 + zv))
    result.add(Polygon(*verts).shift(zv))
    result.set_stroke(width=0)
    result.set_fill(GREY, 1)
    result.set_gloss(1)
    for mob in list(result):
        m2 = mob.copy()
        m2.reverse_points()
        result.add(m2)
    result.apply_depth_test()
    return result


def get_mod_mat(matrix, h_buff=1.3, v_buff=0.8, t2c={}):
    t2c["\\lambda"] = EIGEN_COLORS[1]
    mod_mat = Matrix(
        [
            [matrix[0][0] + " - \\lambda", matrix[0][1]],
            [matrix[1][0], matrix[1][1] + " - \\lambda"],
        ],
        element_to_mobject_config={"tex_to_color_map": t2c},
        h_buff=h_buff,
        v_buff=v_buff,
    )
    return mod_mat


def get_det_mod_mat(mod_mat):
    parens = OldTex("(", ")")
    parens.stretch(2, 1)
    parens.match_height(mod_mat)
    parens[0].next_to(mod_mat, LEFT, SMALL_BUFF)
    parens[1].next_to(mod_mat, RIGHT, SMALL_BUFF)
    det = Text("det")
    det.next_to(parens, LEFT, SMALL_BUFF)
    return VGroup(det, parens, mod_mat)


def get_shadow(vmobject, width=50, n_copies=25):
    shadow = VGroup()
    for w in np.linspace(width, 0, n_copies):
        part = vmobject.copy()
        part.set_fill(opacity=0)
        part.set_stroke(BLACK, width=w, opacity=1.0 / width)
        shadow.add(part)
    return shadow


# Scenes


class Thumbnail(Scene):
    def construct(self):
        grid = NumberPlane(faded_line_ratio=0)
        grid.apply_matrix([[3, 1], [0, 2]])
        grid.set_opacity(0.5)
        self.add(grid)

        mat_mob = IntegerMatrix([[3, 1], [4, 3]], h_buff=0.8)
        mat_mob.set_height(3)
        mat_mob.add_to_back(BackgroundRectangle(mat_mob))
        mat_mob.to_edge(UP, buff=MED_SMALL_BUFF)
        self.add(mat_mob)

        a, b, c, d = mat_mob.get_entries()
        a_to_d = d.get_center() - a.get_center()
        rect = Rectangle(height=1, width=get_norm(a_to_d) + 1)
        rect.round_corners()
        rect.set_stroke(MEAN_COLOR, 3)
        rect.rotate(angle_of_vector(a_to_d))
        rect.move_to(VGroup(a, d))
        rect2 = rect.copy()
        rect2.set_color(PROD_COLOR)
        rect2.rotate(-2 * angle_of_vector(a_to_d))
        rect2.move_to(rect)

        dashed_rects = VGroup(
            DashedVMobject(rect.insert_n_curves(100), num_dashes=50),
            DashedVMobject(rect2.insert_n_curves(100), num_dashes=50),
        )

        self.add(dashed_rects)

        answer = OldTex(
            "\\lambda_1, \\lambda_2 = {3} \\pm \\sqrt{\\,{3}^2 - {5}} = 5, 1",
            tex_to_color_map={"{3}": MEAN_COLOR, "{p}": PROD_COLOR}
        )
        answer.add_background_rectangle()
        answer.set_width(12)
        answer.to_edge(DOWN)
        # answer.shift(SMALL_BUFF * UP)
        self.add(answer)

        arrow = Arrow(mat_mob, answer, fill_color=YELLOW, thickness=0.15, buff=0.3)
        arrow.set_stroke(BLACK, 30, background=True)
        arrow.shift(0.2 * UP)
        self.add(arrow)

        return
        #old
        backdrop = ImageMobject("QuickEigenThumbnailBackdrop")
        backdrop.set_height(FRAME_HEIGHT)
        backdrop.set_opacity(0.5)
        self.add(backdrop)

        buff = 0.75
        det = get_det_mod_mat(get_mod_mat([["3", "1"], ["4", "1"]]))
        det.set_height(2.25)
        det.to_corner(UR, buff=buff)
        self.add(get_shadow(det), det)

        not_this = OldTexText("Not\\\\this")
        not_this.match_height(det)
        not_this.to_corner(UL, buff=buff)
        not_this.set_color(RED)
        self.add(not_this)

        kw = {
            "tex_to_color_map": {
                "\\lambda_1": EIGEN_COLORS[0],
                "\\lambda_2": EIGEN_COLORS[1],
                "M": YELLOW,
            }
        }
        facts = VGroup(
            OldTex("\\text{det}(M) = \\,\\, \\lambda_1 \\cdot \\lambda_2", **kw),
            OldTex("\\text{tr}(M) = \\lambda_1 + \\lambda_2", **kw),
        )
        facts.arrange(DOWN, buff=MED_SMALL_BUFF, index_of_submobject_to_align=2)
        facts.match_height(det)
        facts.match_x(det)
        facts.set_y(-det.get_y())
        self.add(get_shadow(facts), facts)

        use_these = OldTexText("Use\\\\these")
        use_these.match_height(not_this)
        use_these.to_corner(DL, buff=buff)
        use_these.set_color(BLUE)
        not_this.match_x(use_these)
        self.add(use_these)

        not_arrow = Arrow(not_this, det, fill_color=RED, thickness=0.1, buff=0.5)
        use_arrow = Arrow(use_these, facts, fill_color=BLUE, thickness=0.1, buff=0.5)
        self.add(get_shadow(not_arrow), not_arrow)
        self.add(get_shadow(use_arrow), use_arrow)

        ## DELETE
        # mp = OldTex(
        #     "{m} \\pm \\sqrt{\\,{m}^2 - {p}}",
        #     tex_to_color_map={
        #         "{m}": MEAN_COLOR,
        #         "{p}": MEAN_COLOR,
        #     }
        # )
        # for mob, u in [(det, -1), (mp, 1)]:
        #     mob.set_width(FRAME_WIDTH / 2 - 1)
        #     mob.set_x(u * FRAME_WIDTH / 4)
        #     mob.set_y(-1)

        # v_line = DashedLine(4 * UP, 4 * DOWN)

        # ex = Exmark()
        # check = Checkmark()
        # VGroup(ex, check).scale(5)
        # ex.move_to(det).to_edge(UP, LARGE_BUFF)
        # check.move_to(mp).to_edge(UP, LARGE_BUFF)

        # self.add(v_line)
        # self.add(det)
        # self.add(mp)
        # self.add(ex)
        # self.add(check)

        self.embed()


class Assumptions(TeacherStudentsScene):
    def construct(self):
        self.play(
            PiCreatureSays(self.teacher, OldTexText("I'm assuming you know\\\\ what eigenvalues are.")),
            self.change_students(
                "erm", "happy", "tease",
                look_at=ORIGIN,
            ),
            run_time=2,
        )
        self.play(self.students[0].change("guilty").look(LEFT))
        self.wait()

        eigen_expression = OldTex("""
            \\text{det}\\left( \\left[ \\begin{array}{cc}
                3 - \\lambda & 1 \\\\
                4 & 1 - \\lambda
            \\end{array} \\right] \\right)
        """)
        eigen_expression.move_to(self.hold_up_spot, DOWN)
        eigen_expression.to_edge(RIGHT, buff=2)
        VGroup(eigen_expression[0][7], eigen_expression[0][12]).set_color(TEAL)
        cross = Cross(eigen_expression)
        cross.set_stroke(RED, width=(2, 5, 5, 2))
        words = Text("Not this!")
        words.set_color(RED)
        words.next_to(cross, UP, MED_LARGE_BUFF)

        self.play(
            RemovePiCreatureBubble(self.teacher, target_mode="raise_right_hand"),
            FadeIn(eigen_expression, UP),
            self.students[1].change("hesitant"),
            self.students[2].change("sassy"),
        )
        self.play(
            ShowCreation(cross),
            FadeIn(words, 0.25 * UP),
            self.teacher.change("tease", cross),
            self.students[1].change("pondering", cross),
            self.students[2].change("hesitant", cross),
        )
        self.wait(3)
        fade_rect = FullScreenFadeRectangle()
        fade_rect.set_fill(BLACK, opacity=0.7)
        self.add(fade_rect, self.students[0])
        self.play(FadeIn(fade_rect))
        self.play(self.students[0].change("maybe", cross))
        self.play(Blink(self.students[0]))


class ExamplesStart(Scene):
    def construct(self):
        words = OldTexText("Examples start\\\\at", " 4:53")
        words.set_width(8)
        words[-1].set_color(YELLOW)
        self.play(Write(words, run_time=1))
        self.play(FlashAround(words[1], stroke_width=8))
        self.wait()


class PreviousVideoWrapper(Scene):
    def construct(self):
        self.add(FullScreenRectangle())

        screen = ScreenRectangle(height=6)
        screen.set_fill(BLACK, 1)
        screen.set_stroke(BLUE, 3)
        screen.to_edge(DOWN)
        im = ImageMobject("eigen_thumbnail")
        im.replace(screen)
        screen = Group(screen, im)
        title = Text("Introduction", font_size=48)
        # title.match_width(screen)
        title.next_to(screen, UP, MED_LARGE_BUFF)
        # screen.next_to(title, DOWN)

        self.add(screen)
        self.play(Write(title))
        self.wait(2)
        self.play(
            FadeOut(title, UP),
            screen.animate.set_height(7).center(),
        )
        self.wait()


class GoalOfRediscovery(TeacherStudentsScene):
    def construct(self):
        self.play(
            PiCreatureSays(self.teacher, OldTexText("The goal is\\\\rediscovery")),
            self.change_students(
                "happy", "tease", "hooray",
                run_time=1.5,
            )
        )
        self.wait(2)

        recap_words = Text("Quick recap")
        recap_words.move_to(self.screen, UP)
        self.play(
            RemovePiCreatureBubble(self.teacher, target_mode="raise_right_hand", look_at=recap_words),
            self.change_students("pondering", "hesitant", "pondering", look_at=recap_words),
            GrowFromPoint(recap_words, self.teacher.get_corner(UL)),
        )
        self.wait(4)


class RecapWrapper(Scene):
    def construct(self):
        self.add(FullScreenRectangle())
        screen = ScreenRectangle(height=6.75)
        screen.set_stroke(BLUE_B, 2)
        screen.set_fill(BLACK, 1)
        screen.to_edge(DOWN, buff=0.25)
        title = Text("Quick review")
        title.to_edge(UP, buff=0.25)
        self.add(title, screen)


class VisualizeEigenvector(Scene):
    def construct(self):
        plane = NumberPlane(faded_line_ratio=0)
        plane.set_stroke(width=3)
        coords = [-1, 1]
        vector = Vector(plane.c2p(*coords), fill_color=YELLOW)
        array = IntegerMatrix([[-1], [1]], v_buff=0.9)
        array.scale(0.7)
        array.set_color(vector.get_color())
        array.add_to_back(BackgroundRectangle(array))
        array.generate_target()
        array.next_to(vector.get_end(), LEFT)
        array.target.next_to(2 * vector.get_end(), LEFT)
        two_times = OldTex("2 \\cdot")
        two_times.set_stroke(BLACK, 8, background=True)
        two_times.next_to(array.target, LEFT)
        span_line = Line(-4 * vector.get_end(), 4 * vector.get_end())
        span_line.set_stroke(YELLOW_E, 1)

        matrix = [[3, 1], [0, 2]]
        mat_mob = IntegerMatrix(matrix)
        mat_mob.set_x(4).to_edge(UP)
        mat_mob.set_column_colors(GREEN, RED)
        mat_mob.add_to_back(BackgroundRectangle(mat_mob))
        plane.set_stroke(background=True)

        bases = VGroup(
            Vector(RIGHT, fill_color=GREEN_E),
            Vector(UP, fill_color=RED_E),
        )
        faint_plane = plane.copy()
        faint_plane.set_stroke(GREY, width=1, opacity=0.5)

        self.add(faint_plane, plane, bases)
        self.add(vector)
        self.add(mat_mob)

        self.play(Write(array))
        self.add(span_line, vector, array)
        self.play(ShowCreation(span_line))
        self.wait()
        self.play(
            plane.animate.apply_matrix(matrix),
            bases[0].animate.put_start_and_end_on(ORIGIN, plane.c2p(3, 0)),
            bases[1].animate.put_start_and_end_on(ORIGIN, plane.c2p(1, 2)),
            vector.animate.scale(2, about_point=ORIGIN),
            MoveToTarget(array),
            GrowFromPoint(two_times, array.get_left() + SMALL_BUFF * LEFT),
            run_time=3,
            path_arc=0,
        )
        self.wait()

        self.remove(mat_mob)
        self.remove(array)
        self.remove(two_times)


class EigenvalueEquationRearranging(Scene):
    def construct(self):
        v_tex = "\\vec{\\textbf{v}}"
        zero_tex = "\\vec{\\textbf{0}}"
        kw = {
            "tex_to_color_map": {
                "A": BLUE,
                "\\lambda": EIGEN_COLORS[1],
                v_tex: YELLOW,
                "=": WHITE,
            }
        }
        lines = VGroup(
            OldTex("A", v_tex, "=", "\\lambda ", v_tex, **kw),
            OldTex("A", v_tex, "=", "\\lambda ", " I ", v_tex, **kw),
            OldTex("A", v_tex, "-", "\\lambda ", " I ", v_tex, "=", zero_tex, **kw),
            OldTex("(A", "-", "\\lambda ", "I)", v_tex, "=", zero_tex, **kw),
            OldTex("\\text{det}", "(A", "-", "\\lambda ", "I)", "=", "0", **kw),
        )
        for line in lines:
            line.shift(-line.get_part_by_tex("=").get_center())

        mat_prod_brace = Brace(lines[0][:2])
        mat_prod_label = Text("Matrix product", color=BLUE, font_size=24)
        mat_prod_label.next_to(mat_prod_brace, DOWN, SMALL_BUFF, aligned_edge=RIGHT)
        scalar_prod_brace = Brace(lines[0][3:5])
        scalar_prod_label = Text("Scalar product", color=EIGEN_COLORS[1], font_size=24)
        scalar_prod_label.next_to(scalar_prod_brace, DOWN, SMALL_BUFF, aligned_edge=LEFT)

        self.add(lines[0])
        self.play(
            LaggedStart(
                GrowFromCenter(mat_prod_brace),
                GrowFromCenter(scalar_prod_brace),
                lag_ratio=0.3
            ),
            LaggedStart(
                FadeIn(mat_prod_label, 0.25 * DOWN),
                FadeIn(scalar_prod_label, 0.25 * DOWN),
                lag_ratio=0.3
            )
        )
        self.wait()

        id_brace = Brace(lines[1].get_part_by_tex("I"))
        id_label = Text("Identity matrix", font_size=24)
        id_label.next_to(id_brace, DOWN, SMALL_BUFF, aligned_edge=LEFT)

        self.play(
            TransformMatchingTex(lines[0], lines[1]),
            ReplacementTransform(scalar_prod_brace, id_brace),
            FadeTransform(scalar_prod_label, id_label),
            VGroup(mat_prod_label, mat_prod_brace).animate.shift(lines[1].get_left() - lines[0].get_left()),
        )
        self.wait()

        for shift_value, line in zip(it.count(1), lines[2:5]):
            line.shift(shift_value * 0.75 * DOWN)

        self.play(
            TransformMatchingTex(lines[1].copy(), lines[2], path_arc=-45 * DEGREES),
            FadeOut(VGroup(mat_prod_label, mat_prod_brace, id_brace, id_label), 0.5 * DOWN)
        )
        self.wait()
        self.play(
            TransformMatchingShapes(lines[2].copy(), lines[3]),
        )
        self.wait()

        v_part = lines[3].get_part_by_tex(v_tex)
        nz_label = Text("Non-zero vector", color=YELLOW, font_size=24)
        nz_label.next_to(v_part, DR, MED_LARGE_BUFF)
        arrow = Arrow(nz_label.get_corner(UL), v_part, fill_color=YELLOW, buff=0.1, thickness=0.025)

        self.play(
            Write(nz_label, run_time=2),
            GrowArrow(arrow)
        )
        self.wait()
        self.play(
            TransformMatchingShapes(lines[3].copy(), lines[4]),
            FadeOut(nz_label),
            FadeOut(arrow),
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
        det_expression = OldTex("""
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
            parens = OldTex("(", ")")
            parens.set_height(p_height)
            parens[0].next_to(term, LEFT, 0.5 * SMALL_BUFF)
            parens[1].next_to(term, RIGHT, 0.5 * SMALL_BUFF)
            term.parens = parens
            term.parens.set_opacity(0)
            term.add(term.parens)

        eq = OldTex("=")
        eq.next_to(det_expression, RIGHT)
        rhs = VGroup(
            t0.copy(), t3.copy(), OldTex("-"), t1.copy(), t2.copy()
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
        rhs2 = OldTex("\\left( 3 - 4\\lambda + \\lambda^2 \\right) - 4")[0]
        rhs2.next_to(eq2, RIGHT)
        VGroup(rhs2[4], rhs2[6]).set_color(TEAL)

        top_terms = VGroup(
            VGroup(rhs[0][0], rhs[0][2]),
            VGroup(rhs[1][0], rhs[1][2]),
        )
        alt_mid = OldTex("-3\\lambda", tex_to_color_map={"\\lambda": TEAL})
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
        rhs3 = OldTex("\\lambda^2 - 4 \\lambda - 1")[0]
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
        equals_zero = OldTex("= 0")
        equals_zero.next_to(rhs3, RIGHT)
        root_words = OldTex(
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
        formula = OldTex("\\frac{4 \\pm \\sqrt{4^2 - 4(1)(-1)}}{2}")
        formula2 = OldTex("=\\frac{4 \\pm \\sqrt{20}}{2}")
        formula3 = OldTex("= 2 \\pm \\sqrt{5}")
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
        mat.set_column_colors(COL_COLORS[0], COL_COLORS[1])

        lambdas = OldTex("\\lambda_1", "\\,,\\,", "\\lambda_2")
        lambdas[0].set_color(EIGEN_COLORS[0])
        lambdas[2].set_color(EIGEN_COLORS[1])
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
        tr = OldTex("\\text{tr}", "\\Big(", "\\Big)", font_size=60)
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
                "a": COL_COLORS[0],
                "b": COL_COLORS[1],
                "c": COL_COLORS[0],
                "d": COL_COLORS[1],
                "=": WHITE,
                "\\lambda_1": EIGEN_COLORS[0],
                "\\lambda_2": EIGEN_COLORS[1],
            }
        }
        tr_rhs = OldTex("= a + d = \\lambda_1 + \\lambda_2", **tex_kw)
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
        half = OldTex("1 \\over 2")
        half.move_to(tr, LEFT)
        tr.generate_target()
        tr.target.next_to(half, RIGHT, SMALL_BUFF)
        new_tr_rhs = OldTex("= {a + d \\over 2} = {\\lambda_1 + \\lambda_2 \\over 2}", **tex_kw)
        new_tr_rhs.next_to(tr.target, RIGHT)

        self.play(
            GrowFromCenter(half),
            MoveToTarget(tr),
            TransformMatchingShapes(tr_rhs, new_tr_rhs),
        )
        self.wait()

        # Determinant
        det_mat = mat.deepcopy()
        det = OldTex("\\text{det}", "\\Big(", "\\Big)", font_size=60)
        det[1:].match_height(det_mat, stretch=True)
        det.set_submobjects([*det[:-1], det_mat, det[-1]])
        det.arrange(RIGHT, buff=SMALL_BUFF)
        det.next_to(indices[1], RIGHT, MED_LARGE_BUFF)

        det_rhs = OldTex("= ad - bc = \\lambda_1 \\lambda_2", **tex_kw)
        det_rhs.next_to(det, RIGHT)
        self.play(
            TransformFromCopy(mat, det_mat),
            Write(VGroup(*det[:2], det[-1])),
        )

        self.play(
            det_path_anim(det_mat),
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
        eq_m = OldTexText("=", " $m$", "\\quad (mean)")
        eq_m[1].set_color(MEAN_COLOR)
        eq_m.next_to(new_tr_rhs, RIGHT)
        eq_p = OldTexText("=", " $p$", "\\quad (product)")
        eq_p[1].set_color(PROD_COLOR)
        eq_p.next_to(det_rhs, RIGHT)

        form_lhs = lambdas.copy()
        form_rhs = OldTex("= {m} \\pm \\sqrt{\\,{m}^2 - {p}}", tex_to_color_map={"{m}": MEAN_COLOR, "{p}": PROD_COLOR})
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
            randy.change("erm", third_point_placeholder)
        )
        self.play(Blink(randy))
        self.wait()
        self.play(
            randy.change("thinking", eq_m),
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
        ex_mat.set_column_colors(COL_COLORS[0], COL_COLORS[1])
        ex_mat.next_to(randy, RIGHT, aligned_edge=UP)

        kw = {"tex_to_color_map": {"m": MEAN_COLOR, "p": PROD_COLOR, "=": WHITE, "-": WHITE}}
        m_eq = OldTex("m = 7", **kw)
        p_eq1 = OldTex("p = 48 - 8", **kw)
        p_eq2 = OldTex("p = 40", **kw)

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
            randy.change("raise_right_hand", ex_mat),
            FadeIn(ex_mat, RIGHT),
            FadeOut(group, RIGHT),
        )
        self.play(Blink(randy))
        self.wait()

        self.play(FadeIn(tr_rect), randy.change("pondering", tr_rhs))
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

        self.play(FadeOut(tr_rect), FadeIn(det_rect), randy.animate.look_at(tr_rhs))
        self.play(
            Write(p_eq1[:2]),
            randy.change("hesitant", p_eq1),
            FadeOut(mean_rect),
        )

        self.play(det_path_anim(ex_mat))
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
            randy.change("tease", p_eq2),
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
            randy.change("pondering", ORIGIN)
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


class MeansMatch(Scene):
    def construct(self):
        t2c = {
            "{a}": COL_COLORS[0],
            "{d}": COL_COLORS[1],
            "\\lambda_1": EIGEN_COLORS[0],
            "\\lambda_2": EIGEN_COLORS[1],
            "=": WHITE,
        }
        equation = OldTex(
            "{{a} + {d} \\over 2} = {\\lambda_1 + \\lambda_2 \\over 2}",
            tex_to_color_map=t2c
        )
        mean_eq = OldTex(
            "\\text{mean}({a}, {d}) = \\text{mean}(\\lambda_1, \\lambda_2)",
            tex_to_color_map=t2c
        )

        mean_eq.next_to(equation, DOWN, LARGE_BUFF)

        self.add(equation)

        self.play(TransformMatchingShapes(equation[5:].copy(), mean_eq[6:]))
        self.wait()
        self.play(TransformMatchingShapes(equation[:5].copy(), mean_eq[:6]))
        self.wait()


class ShowSquishingAndStretching(Scene):
    def construct(self):
        self.camera.frame.set_height((3 / 4) * FRAME_HEIGHT)

        # Transform
        plane = NumberPlane(
            (-20, 20), (-20, 20),
            background_line_style={"stroke_width": 3},
            faded_line_ratio=0,
        )
        plane.axes.set_stroke(BLUE, 3)
        back_plane = NumberPlane(
            faded_line_ratio=0,
        )
        back_plane.set_stroke(GREY_B, 1, opacity=0.5)
        mat = [[2, 2], [1, 2]]
        eigenvalues, eigenvectors = np.linalg.eig(mat)
        eigenvectors = [normalize(v) for v in eigenvectors.T]
        eigenlines = VGroup(*(
            Line(10 * ev, -10 * ev, color=color)
            for ev, color in zip(eigenvectors, EIGEN_COLORS)
        ))
        eigenlines.set_stroke(GREY_B, 2)

        eigenvect_mobs = VGroup(*(
            Vector(ev, fill_color=color)
            for ev, color in zip(eigenvectors, EIGEN_COLORS)
        ))
        eigenvect_mobs[0].add_updater(lambda m: m.put_start_and_end_on(plane.c2p(0, 0), plane.c2p(*eigenvectors[0][:2])))
        eigenvect_mobs[1].add_updater(lambda m: m.put_start_and_end_on(plane.c2p(0, 0), plane.c2p(*eigenvectors[1][:2])))

        # Basis vectors
        bases = VGroup(
            Vector(RIGHT, fill_color=GREEN, thickness=0.05),
            Vector(UP, fill_color=RED, thickness=0.05),
        )
        bases[0].add_updater(lambda m: m.put_start_and_end_on(plane.c2p(0, 0), plane.c2p(1, 0)))
        bases[1].add_updater(lambda m: m.put_start_and_end_on(plane.c2p(0, 0), plane.c2p(0, 1)))

        disk = Circle(radius=1)
        disk.set_fill(YELLOW, 0.25)
        disk.set_stroke(YELLOW, 3)

        morpher = VGroup(plane, disk)

        self.add(back_plane, morpher, eigenlines, *eigenvect_mobs)

        self.play(
            morpher.animate.apply_matrix(mat), run_time=4
        )
        self.wait()

        # Labels
        labels = VGroup(*(
            OldTex("\\text{Stretch by }", f"\\lambda_{i}", color=color, font_size=30)
            for i, color in zip((1, 2), EIGEN_COLORS)
        ))
        for label, vect in zip(labels, eigenvect_mobs):
            label.next_to(
                vect.get_center(),
                rotate_vector(normalize(vect.get_vector()), 90 * DEGREES),
                buff=0.1,
                index_of_submobject_to_align=1,
            )
        labels.set_stroke(BLACK, 5, background=True)

        self.play(LaggedStartMap(FadeIn, labels, lag_ratio=0.7))
        self.wait()


class MeanProductExample(Scene):
    def construct(self):
        # Number line and midpoint
        number_line = NumberLine((0, 14))
        number_line.add_numbers()
        number_line.set_width(FRAME_WIDTH - 1)
        number_line.to_edge(UP, buff=1.5)
        nl = number_line

        mean = 7
        m_dot = Dot(nl.n2p(mean))
        m_dot.set_color(MEAN_COLOR)
        m_label = OldTex("m", color=MEAN_COLOR)
        m_label.next_to(m_dot, UP, buff=MED_SMALL_BUFF)
        label7 = OldTex("7", color=MEAN_COLOR)

        # Distance tracking
        d_tracker = ValueTracker(4)

        def get_l1_point():
            return nl.n2p(mean - d_tracker.get_value())

        def get_l2_point():
            return nl.n2p(mean + d_tracker.get_value())

        l1_dot, l2_dot = (Dot(color=TEAL) for x in range(2))
        l1_label = OldTex("\\lambda_1", color=EIGEN_COLORS[0])
        l2_label = OldTex("\\lambda_2", color=EIGEN_COLORS[1])
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

        minus_d = OldTex("-d")
        plus_d = OldTex("+d")
        always(minus_d.next_to, l1_arrow, UP, SMALL_BUFF)
        always(plus_d.next_to, l2_arrow, UP, SMALL_BUFF)
        plus_qm = OldTex("+??")
        minus_qm = OldTex("-??")
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
        self.wait(20)
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
        texs = VGroup(*(OldTex(tex, **kw) for tex in [
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

        plus_3 = OldTex("+3").move_to(plus_d)
        minus_3 = OldTex("-3").move_to(minus_d)
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
                ps.add(OldTex("p").set_color(PROD_COLOR).move_to(fourty).shift(0.1 * DOWN))
            for seven in tex.get_parts_by_tex("7"):
                sevens.add(seven)
                m = OldTex("m").set_color(MEAN_COLOR)
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
            OldTex(
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


class AskIfThatsBetter(Scene):
    def construct(self):
        randy = Randolph(height=2)
        randy.to_edge(DOWN)
        randy.change("pondering", UL)

        self.add(BackgroundRectangle(randy, fill_opacity=1), randy)
        self.play(
            PiCreatureSays(
                randy, "Is that better?", target_mode="sassy",
                bubble_config={"direction": LEFT, "width": 4, "height": 2},
                look_at=UL
            )
        )
        self.play(Blink(randy))
        self.wait()
        self.play(
            RemovePiCreatureBubble(randy, target_mode="thinking")
        )
        self.wait(2)
        self.play(randy.change("pondering", UL))
        self.play(Blink(randy))
        self.wait()


class OutstandingChannel(Scene):
    def construct(self):
        morty = Mortimer()
        morty.to_corner(DR)
        self.play(PiCreatureSays(
            morty, OldTexText("Outstanding\\\\channel"), target_mode="hooray",
            bubble_config={"height": 3, "width": 4}
        ))
        self.play(Blink(morty))
        self.wait()


class JingleAnimation(Scene):
    def construct(self):
        form = OldTex("m \\pm \\sqrt{m^2 - p}")[0]
        form.set_height(2)
        m, old_pm, sqrt, root, m2, squared, minus, p = form
        VGroup(m, m2).set_color(MEAN_COLOR)
        VGroup(p).set_color(PROD_COLOR)
        m.refresh_bounding_box()  # Why?
        p.refresh_bounding_box()  # Why?

        pm = VGroup(OldTex("+"), OldTex("-"))
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

        self.play(FadeIn(m, scale=0.5, rate_func=snap, run_time=0.5))
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

        sqrt_outline = sqrt.copy()
        sqrt_outline.set_stroke(YELLOW, 10)
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
        self.play(
            TransformFromCopy(m, m2, path_arc=-120 * DEGREES, rate_func=smooth, run_time=0.6),
        )
        self.play(FadeTransform(m2.copy(), squared, run_time=0.5))
        minus.save_state()
        minus.stretch(0, 0, about_edge=LEFT)
        self.play(Restore(minus), run_time=0.5, rate_func=smooth)
        self.wait(0.2)
        self.play(
            FadeIn(p, scale=0.5, rate_func=snap, run_time=0.5),
        )
        self.add(product)
        self.wait(0.4)
        self.play(
            Flash(p, flash_radius=0.8 * p.get_height(), run_time=0.5),
        )
        self.wait(0.3)
        self.remove(product)
        self.wait()


class Example1(TeacherStudentsScene):
    def construct(self):
        # Show matrix
        mat = IntegerMatrix([[3, 1], [4, 1]])
        mat.set_column_colors(*COL_COLORS)
        mat.move_to(self.hold_up_spot, DOWN)

        self.play(
            self.teacher.change("raise_right_hand", mat),
            FadeIn(mat.get_brackets(), UP)
        )
        self.play(
            Write(mat.get_entries(), lag_ratio=0.1, run_time=2),
            self.change_students("pondering", "pondering", "thinking", look_at=mat)
        )
        self.wait()

        bubble = ThoughtBubble(height=4, width=7)
        bubble.pin_to(self.students[2])
        self.play(
            self.students[2].change("tease", bubble),
            Write(bubble)
        )
        self.wait(3)

        # Write formula
        shift_vect = 5 * LEFT
        arrow = Vector(0.75 * RIGHT)
        arrow.next_to(mat, RIGHT)
        formula = OldTex(
            "\\lambda_1, \\, \\lambda_2 = {2} \\pm \\sqrt{\\,{2}^2 - (-1)}",
            tex_to_color_map={
                "\\lambda_1": EIGEN_COLORS[0],
                "\\lambda_2": EIGEN_COLORS[1],
                "(-1)": PROD_COLOR,
                "{2}": MEAN_COLOR,
            }
        )
        formula.next_to(arrow, RIGHT)
        VGroup(formula, arrow).shift(shift_vect)

        twos = formula.get_parts_by_tex("{2}")
        min1 = formula.get_part_by_tex("(-1)")
        m_rects = VGroup(*(SurroundingRectangle(two, buff=0) for two in twos))
        m_rects.set_stroke(MEAN_COLOR)
        p_rect = SurroundingRectangle(min1, buff=0)
        p_rect.set_stroke(PROD_COLOR)
        form_rects = VGroup(m_rects, p_rect)
        VGroup(twos, min1).set_opacity(0)

        self.play(
            FadeOut(bubble),
            mat.animate.shift(shift_vect),
            FadeIn(arrow, shift_vect),
            FadeIn(formula),
            FadeIn(form_rects),
            self.change_students("pondering", "pondering", "pondering", look_at=ORIGIN)
        )
        self.play(
            self.teacher.change("tease", formula),
            *(pi.animate.look_at(formula) for pi in self.students),
        )
        self.wait()

        # Show mean
        m_eq = OldTex("m", "=", "2", tex_to_color_map={"m": MEAN_COLOR, "2": MEAN_COLOR})
        m_eq.next_to(mat, UP, LARGE_BUFF)
        t2c = {
            "\\lambda_1": EIGEN_COLORS[0],
            "\\lambda_2": EIGEN_COLORS[1],
            "{m}": MEAN_COLOR,
            "{p}": PROD_COLOR,
            "=": WHITE,
        }
        diag_rects = VGroup(*(SurroundingRectangle(mat.get_entries()[i]) for i in [0, 3]))
        diag_rects.set_stroke(MEAN_COLOR)
        two_rect = SurroundingRectangle(m_eq[2])
        two_rect.set_stroke(MEAN_COLOR)

        self.play(
            Write(m_eq[:2]),
            *(pi.animate.look_at(m_eq) for pi in self.pi_creatures)
        )
        self.play(
            ShowCreation(diag_rects),
            self.teacher.change("tease", diag_rects),
            *(pi.animate.look_at(diag_rects) for pi in self.students),
        )
        self.wait()
        self.play(
            FadeTransform(diag_rects[0].copy(), m_eq[2], remover=True),
            FadeTransform(diag_rects[1].copy(), m_eq[2], remover=True),
            *(pi.animate.look_at(two_rect) for pi in self.pi_creatures),
        )
        self.wait(2)

        self.remove(twos, min1)
        VGroup(twos, min1).set_opacity(1)
        for i in (0, 1):
            self.play(
                TransformFromCopy(m_eq[2], twos[i]),
                FadeOut(m_rects[i]),
                self.teacher.change("raise_right_hand", twos),
                *(pi.change("thinking", twos) for pi in self.students)
            )
        self.wait()

        # Show product
        prod_eq = OldTex(
            # "\\lambda_1 \\lambda_2 =",
            "{p} = {3} - {4} = -1",
            tex_to_color_map={
                "{3}": COL_COLORS[0],
                "{4}": COL_COLORS[0],
                **t2c
            }
        )
        prod_eq.next_to(mat, UP, LARGE_BUFF)
        prod_eq.set_x(mat.get_center()[0], LEFT)

        lhs = prod_eq[:1]

        self.play(
            m_eq.animate.scale(0.7).next_to(mat, LEFT).to_edge(LEFT),
            FadeIn(lhs),
            FadeOut(diag_rects),
            self.teacher.change("happy"),
            self.change_students("erm", "hesitant", "thinking", look_at=prod_eq)
        )
        self.play(det_path_anim(mat, run_time=1))
        self.play(
            FadeIn(prod_eq[1]),
            FadeTransform(mat.get_entries()[0].copy(), prod_eq[2]),
            FadeTransform(mat.get_entries()[3].copy(), prod_eq[2]),
            VFadeInThenOut(VGroup(*(SurroundingRectangle(e) for e in mat.get_entries()[0::3]))),
            self.teacher.change("coin_flip_1"),
        )
        self.play(
            FadeIn(prod_eq[3]),
            FadeTransform(mat.get_entries()[1].copy(), prod_eq[4]),
            FadeTransform(mat.get_entries()[2].copy(), prod_eq[4]),
            VFadeInThenOut(VGroup(*(SurroundingRectangle(e) for e in mat.get_entries()[1:3]))),
        )
        self.wait()
        self.play(
            Write(prod_eq[-2:]),
            self.change_students("tease", "tease", "happy", look_at=prod_eq.get_right()),
        )
        self.play(
            FadeOut(p_rect),
            FadeTransform(prod_eq[-1].copy(), min1),
            self.teacher.change("raise_left_hand", min1),
            *(pi.animate.look_at(min1) for pi in self.students)
        )
        self.wait()
        self.play(
            prod_eq.animate.scale(0.7).next_to(m_eq, UP).to_edge(LEFT)
        )
        self.wait(2)

        # Final answer
        rhs = OldTex("2\\pm\\sqrt{5}")
        rhs.move_to(formula[4], LEFT)

        self.play(
            self.teacher.change("tease", rhs),
            TransformMatchingShapes(formula[4:6].copy(), rhs),
            formula[4:].animate.shift(UP),
        )
        self.wait()


class SameExampleWrapper(Scene):
    def construct(self):
        self.add(FullScreenRectangle())
        screen = ScreenRectangle()
        screen.set_fill(BLACK, 1)
        screen.set_stroke(BLUE, 3)
        screen.set_height(6)
        screen.to_edge(DOWN)
        title = Text("Same example from before")
        title.next_to(screen, UP)
        self.add(screen, title)


class GeneralExample(Scene):
    matrix = [[3, 1], [4, 1]]

    def construct(self):
        mat = IntegerMatrix(self.matrix)
        mat.move_to(2 * LEFT)
        mat.set_column_colors(*COL_COLORS)

        matrix = np.array(self.matrix)
        a, b, c, d = matrix.flatten()

        mean = (a + d) / 2
        if (a + d) % 2 == 0:
            mean = int(mean)
        prod = a * d - b * c
        mean_str = "{" + str(mean) + "}"
        prod_str = "{" + str(prod) + "}"

        self.add(mat)
        self.wait()

        mean_eq = OldTex(f"m = {mean_str}", tex_to_color_map={"m": MEAN_COLOR, mean_str: MEAN_COLOR})
        mean_eq.next_to(mat, UP, LARGE_BUFF)
        diag_rects = get_diag_rects(mat)

        self.play(
            FadeIn(mean_eq[:2], 0.25 * UP)
        )
        self.play(LaggedStartMap(ShowCreation, diag_rects, lag_ratio=0.3, run_time=1))
        self.play(
            FadeTransform(diag_rects[0].copy(), mean_eq[2]),
            FadeTransform(diag_rects[1].copy(), mean_eq[2]),
        )
        self.play(FadeOut(diag_rects))

        last_part = "(" + prod_str + ")}" if prod < 0 else prod_str + "}"
        formula = OldTex(
            mean_str, "\\pm \\sqrt{\\,", mean_str, "^2 - ", last_part,
            font_size=60,
            tex_to_color_map={mean_str: MEAN_COLOR, prod_str: PROD_COLOR},
        )
        formula.next_to(mat, RIGHT, buff=1.5)

        self.play(
            Write(formula[1]),
            Write(formula[3]),
            FadeTransform(mean_eq[2].copy(), formula[0]),
            FadeTransform(mean_eq[2].copy(), formula[2]),
        )
        self.wait()

        # Product
        prod_eq = OldTex(
            f"p = {a * d} - {b * c} = {prod}",
            tex_to_color_map={
                str(prod): PROD_COLOR,
                str(a * d): COL_COLORS[0],
                str(b * c): COL_COLORS[0],
                "=": WHITE,
            }
        )
        prod_eq.move_to(mean_eq)

        self.play(
            FadeIn(prod_eq[:2]),
            mean_eq.animate.shift(UP),
        )
        self.play(
            FadeIn(prod_eq[2]),
            VFadeInThenOut(get_diag_rects(mat, color=YELLOW), run_time=1.5),
        )
        self.play(
            FadeIn(prod_eq[3:5]),
            VFadeInThenOut(get_diag_rects(mat, color=YELLOW, off_diagonal=True), run_time=1.5),
        )
        self.wait()
        self.play(Write(prod_eq[5:]))
        self.play(FadeTransform(prod_eq[-1].copy(), formula[4:]))
        self.wait()

        # Simplify
        line2 = OldTex(mean_str, "\\pm", "\\sqrt{\\,", str(mean**2 - prod), "}", font_size=60)
        if mean == 0:
            line2[0].scale(0, about_point=line2[1].get_left())
        line2[0].set_color(MEAN_COLOR)
        line2.next_to(formula, DOWN, buff=0.7, aligned_edge=LEFT)

        self.play(FadeTransform(formula.copy()[:4], line2))
        self.wait()

        line3 = self.get_final_simplification()
        if line3:
            line3.next_to(line2, DOWN, buff=0.7, aligned_edge=LEFT)
            self.play(FadeIn(line3, DOWN))
            self.wait()
            answer = line3[-1]
        else:
            answer = line2
        self.play(ShowCreation(SurroundingRectangle(answer)))
        self.wait()

    def get_final_simplification(self):
        return None


class Example2(GeneralExample):
    matrix = [[2, 7], [1, 8]]

    def get_final_simplification(self):
        return OldTex("5 \\pm 4", "=", "9,\\,1", font_size=60)


class Example3(GeneralExample):
    matrix = [[3, 11], [1, 11]]


class Example4(GeneralExample):
    matrix = [[2, -1], [2, 0]]


class Example5(GeneralExample):
    matrix = [[2, 3], [5, 7]]


class PauliMatrices(Scene):
    def construct(self):
        self.camera.frame.focal_distance = 20

        # Matrices
        colors = [RED, GREEN, BLUE]
        lhss, matrices = self.get_lhss_and_matrices(colors)

        self.add(lhss)
        self.play(LaggedStartMap(FadeIn, matrices, shift=0.25 * RIGHT, lag_ratio=0.3, run_time=2))
        self.wait()

        # Title
        title = Text("Pauli spin matrices")
        title.next_to(group, RIGHT, buff=1.25)
        self.play(Write(title, run_time=2))
        self.wait()

        # Axes
        axes = ThreeDAxes(
            (-2, 2), (-2, 2), (-2, 2),
            axis_config={"include_tip": False}
        )
        axes.set_flat_stroke(False)
        axes.rotate(20 * DEGREES, OUT)
        axes.rotate(70 * DEGREES, LEFT)
        axes.set_height(1.7)
        axes.labels = VGroup(*(
            OldTex(ch, font_size=24).next_to(axis.get_end(), vect, buff=SMALL_BUFF)
            for ch, axis, vect in zip("xyz", axes.get_axes(), [RIGHT, UP, UP])
        ))
        axes.add(axes.labels)
        axes.set_stroke(background=True)

        axes_copies = VGroup()
        for i, matrix, color, (v1, v2) in zip(it.count(), matrices, colors, [(UP, DOWN), (LEFT, RIGHT), (RIGHT, RIGHT)]):
            ac = axes.deepcopy()
            ac.labels.set_fill(GREY_C, 1)
            ac.labels[i].set_fill(color, 1)
            axis = ac.get_axes()[i]
            vp1, vm1 = (
                Arrow(axis.n2p(0), axis.n2p(n), buff=0.05, fill_color=color, thickness=0.05)
                for n in (2, -2)
            )
            for vector, tex, v in [(vp1, "+1", v1), (vm1, "-1", v2)]:
                vector.add(OldTex(tex, font_size=24).next_to(vector, v, buff=0.05))
            ac.add(vp1, vm1)
            ac.next_to(matrix, RIGHT, buff=LARGE_BUFF)
            axes_copies.add(ac)

        acx, acy, acz = axes_copies

        self.play(
            FadeOut(title),
            Write(acx, stroke_width=0.5),
        )
        self.wait()
        self.play(TransformFromCopy(acx, acy))
        self.wait()
        self.play(TransformFromCopy(acy, acz))
        self.wait()
        self.play(LaggedStartMap(FadeOut, axes_copies, shift=0.25 * DOWN, run_time=1, lag_ratio=0.3))

        # Compute means
        means_rects = VGroup(*(
            get_diag_rects(matrix)
            for matrix in matrices
        ))
        mean_eqs = VGroup(*(
            OldTex("m = 0", tex_to_color_map={"m": MEAN_COLOR, "0": WHITE}).next_to(matrix, RIGHT, buff=MED_LARGE_BUFF)
            for matrix in matrices
        ))

        kw = {
            "tex_to_color_map": {
                "0": MEAN_COLOR,
                "{p}": PROD_COLOR,
            }
        }
        forms = VGroup(*(
            OldTex("0 \\pm \\sqrt{\\,0^2 - {p}}", **kw).next_to(matrix, RIGHT).to_edge(RIGHT)
            for matrix in matrices
        ))
        simple_forms = VGroup(*(
            OldTex("\\pm \\sqrt{-{p}}", **kw).move_to(form, LEFT)
            for form in forms
        ))

        for me, mr, form in zip(mean_eqs, means_rects, forms):
            self.play(
                FadeIn(mr),
                Write(me[:2], run_time=1),
            )
            self.play(
                TransformFromCopy(mr, me[2], run_time=0.7)
            )
        self.wait()
        self.play(
            LaggedStart(*(
                FadeIn(VGroup(form[1], form[3], form[4]))
                for form in forms
            ), lag_ratio=0.2),
            LaggedStart(*(
                TransformFromCopy(me.get_parts_by_tex("0"), form.get_parts_by_tex("0"))
                for me, form in zip(mean_eqs, forms)
            ), lag_ratio=0.2),
        )
        self.wait()
        self.play(LaggedStart(*(
            TransformMatchingShapes(form, simple_form)
            for form, simple_form in zip(forms, simple_forms)
        )))
        self.wait()

        # Products
        prod_eqs = VGroup(*(
            OldTex("p = -1", tex_to_color_map={"p": PROD_COLOR, "-1": WHITE}).next_to(
                matrix, RIGHT, buff=MED_LARGE_BUFF
            ).shift(0.5 * DOWN)
            for matrix in matrices
        ))

        self.play(
            mean_eqs.animate.shift(0.5 * UP),
            LaggedStart(*(FadeIn(pe[:2]) for pe in prod_eqs)),
            FadeOut(means_rects)
        )
        self.play(LaggedStart(*(det_path_anim(matrix) for matrix in matrices), lag_ratio=0.2))

        for matrix, pe in zip(matrices, prod_eqs):
            r1 = get_diag_rects(matrix)
            r2 = get_diag_rects(matrix, off_diagonal=True)
            VGroup(r1, r2).set_color(YELLOW)
            self.play(FadeIn(r1))
            self.play(FadeOut(r1), FadeIn(r2), FadeIn(pe[2]))
            self.play(FadeOut(r2))
            self.wait()

        final_forms = VGroup(*(
            OldTex("= \\pm 1").move_to(sf.get_center(), LEFT)
            for sf in simple_forms
        ))

        self.play(
            LaggedStart(*(
                FadeTransform(pe[2].copy(), ff)
                for pe, ff in zip(prod_eqs, final_forms)
            ), lag_ratio=0.3),
            LaggedStart(*(
                sf.animate.next_to(ff, LEFT)
                for sf, ff in zip(simple_forms, final_forms)
            ))
        )
        self.wait()

        # Bring back axes
        axes_copies.shift(2 * RIGHT)
        self.play(
            FadeOut(simple_forms), FadeOut(final_forms),
            FadeIn(acx),
        )
        self.play(FadeIn(acy))
        self.play(FadeIn(acz))
        self.wait()

    def get_lhss_and_matrices(self, colors):
        lhss = VGroup(*(OldTex(f"\\sigma_{c} = ") for c in "xyz"))
        kw = {"h_buff": 0.7, "v_buff": 0.7}
        matrices = VGroup(
            Matrix([["0", "1"], ["1", "0"]], **kw),
            Matrix([["0", "-i"], ["i", "0"]], **kw),
            Matrix([["1", "0"], ["0", "-1"]], **kw),
        )
        lhss.set_submobject_colors_by_gradient(*colors)
        lhss.arrange(DOWN, buff=2.5)
        for lhs, matrix in zip(lhss, matrices):
            matrix.set_height(1.5)
            matrix.next_to(lhs, RIGHT)

        group = VGroup(lhss, matrices)
        group.move_to(2 * LEFT)
        return group


class SpinMeasurements(ThreeDScene):
    def construct(self):
        # Reorient
        frame = self.camera.frame
        frame.reorient(-70, 70)
        frame.add_updater(lambda m, dt: m.increment_theta(0.01 * dt))
        self.add(frame)

        # Create machine
        north_verts = [UL + 0.5 * DOWN, UR + 0.5 * DOWN, DR, 2 * DOWN, DL]
        south_verts = [DOWN + 2 * LEFT, UP + 2 * LEFT, UL, LEFT, RIGHT, UR, UP + 2 * RIGHT, DOWN + 2 * RIGHT]

        north_bar = get_prism(north_verts, depth=2)
        south_bar = get_prism(south_verts, depth=2)
        N_text = Text("N").move_to(north_bar, OUT).shift(0.025 * OUT)
        S_text = Text("S").move_to(south_bar, OUT).shift(0.025 * OUT)
        S_text.set_y(-0.5)
        north_bar.add(N_text)
        south_bar.add(S_text)
        south_bar.move_to(ORIGIN, UP).shift(0.5 * UP)
        north_bar.move_to(south_bar.get_top(), DOWN)

        machine = VGroup(north_bar, south_bar)
        machine.set_stroke(width=0)
        # for bar in machine:
        #     bar.space_out_submobjects(1.1)

        # Screen
        screen = FullScreenRectangle()
        axes = Axes((-3, 3), (-3, 3))
        for sm in axes.get_family():
            sm.flat_stroke = False
            sm.insert_n_curves(10)
        axes.add(OldTex("+z").next_to(axes.c2p(0, 2), RIGHT))
        axes.add(OldTex("-z").next_to(axes.c2p(0, -2), RIGHT))
        axes.shift(0.1 * OUT)
        screen.set_height(5)
        screen.set_fill(GREY_D, 1)
        screen.set_stroke(WHITE, 2)
        screen.shift(5 * IN)
        axes.shift(5 * IN)

        # Rotate all
        everything = VGroup(screen, axes, machine)
        everything.apply_depth_test()
        self.add(everything)
        self.update_frame()  # Why?
        everything.rotate(89.5 * DEGREES, RIGHT, about_point=ORIGIN)

        self.play(
            FadeIn(screen),
            FadeIn(axes),
            FadeIn(machine, lag_ratio=0.05),
        )
        self.wait()

        # Paths
        def path(t):
            if t < -1:
                z = 0
            elif t < 1:
                z = (1 / 4) * (t + 1)**2
            else:
                z = t

            return [0, t, z]

        high_beam = ParametricCurve(path, (-5, 5))
        high_beam.set_stroke(BLUE, 10)
        high_beam.set_depth(1, stretch=True, about_point=ORIGIN)
        high_beam.apply_depth_test()
        low_beam = high_beam.copy()
        low_beam.stretch(-1, 2, about_point=ORIGIN)

        def hit_anim(spin=1):
            circ = Circle(radius=0.5)
            circ.set_stroke(BLUE, 0)
            circ.rotate(90 * DEGREES, RIGHT)
            circ.move_to(5 * UP + spin * OUT)
            tex = OldTex("+1" if spin == 1 else "-1")
            tex.shift(20 * OUT)
            self.add(tex)
            self.update_frame()
            tex.rotate(90 * DEGREES, RIGHT)
            tex.move_to(circ.get_left())
            pre_circ = circ.copy()
            pre_circ.scale(0)
            pre_circ.set_stroke(BLUE, 10)
            return AnimationGroup(
                Transform(pre_circ, circ, remover=True),
                VFadeInThenOut(tex),
            )

        for x in range(20):
            if random.random() < 0.5:
                beam = high_beam
                spin = +1
            else:
                beam = low_beam
                spin = -1
            self.play(LaggedStart(
                VShowPassingFlash(beam, time_width=0.5),
                hit_anim(spin),
                lag_ratio=0.6
            ))


class HeresTheThing(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            "Wait, why?",
            target_mode="raise_right_hand",
            look_at=self.screen,
            index=2,
        )
        self.play(
            self.change_students("maybe", "confused", "raise_right_hand", look_at=self.screen),
            self.teacher.change("happy"),
        )
        self.wait(3)
        self.teacher_says(
            OldTexText("Here's the thing..."),
            added_anims=[self.change_students("sassy", "plain", "hesitant")],
            target_mode="hesitant",
        )
        self.wait(2)
        self.play(RemovePiCreatureBubble(self.teacher, target_mode="raise_right_hand"))
        for pi in self.pi_creatures:
            for eye in pi.eyes:
                eye.refresh_bounding_box()
        self.look_at(self.screen)
        self.play_student_changes("tease", "thinking", "happy", look_at=self.screen)
        self.wait(3)
        words = OldTexText("Somewhat\\\\self-defeating")
        words.next_to(self.screen, RIGHT)
        words.shift(RIGHT)
        self.play(
            self.teacher.change("guilty"),
            self.change_students("sassy", "hesitant", "sassy"),
            Write(words),
        )
        self.wait(4)


class NotAllHopeIsLost(TeacherStudentsScene):
    def construct(self):
        self.teacher_says(
            OldTexText("There's still a\\\\good example here"),
            target_mode="speaking",
            bubble_config={"height": 3, "width": 4},
            added_anims=[self.change_students("erm", "sassy", "hesitant")],
        )
        self.wait(2)


class PauliMatricesWithCharacteristicPolynomial(PauliMatrices):
    def construct(self):
        # Set up determinants
        colors = [RED, GREEN, BLUE]
        lhss, matrices = group = self.get_lhss_and_matrices(colors)
        group.to_edge(LEFT)

        kw = {
            "element_to_mobject_config": {
                "tex_to_color_map": {
                    "-\\lambda": EIGEN_COLORS[1],
                }
            },
            "v_buff": 0.7,
            "h_buff": 1.3,
        }
        new_matrices = VGroup(
            Matrix([["-\\lambda", "1"], ["1", "-\\lambda"]], **kw),
            Matrix([["-\\lambda", "-i"], ["i", "-\\lambda"]], **kw),
            Matrix([["1 -\\lambda", "0"], ["0", "-1 -\\lambda"]], **kw),
        )
        arrows = VGroup()
        det_terms = VGroup()
        for mat, new_mat in zip(matrices, new_matrices):
            mat.set_height(1.0, about_edge=LEFT)
            new_mat.replace(mat, dim_to_match=1)
            new_mat.shift(3.75 * RIGHT)

            parens = OldTex("()", font_size=60)[0]
            parens.match_height(new_mat, stretch=True)
            parens[0].next_to(new_mat, LEFT, buff=0.1)
            parens[1].next_to(new_mat, RIGHT, buff=0.1)
            det = Text("det", font_size=30)
            det.next_to(parens, LEFT, SMALL_BUFF)

            det_terms.add(VGroup(det, parens, new_mat))
            arrows.add(Arrow(mat, det, buff=0.2))

        self.add(lhss, matrices)
        self.wait()
        self.play(
            LaggedStart(*(
                TransformMatchingShapes(mat.copy(), det_term)
                for mat, det_term in zip(matrices, det_terms)
            ), lag_ratio=0.8),
            LaggedStartMap(GrowArrow, arrows, lag_ratio=0.8),
        )
        self.wait()

        # Polynomials
        kw = {"tex_to_color_map": {
            "\\lambda": EIGEN_COLORS[1],
            "=": WHITE,
        }}
        rhss = VGroup(
            OldTex("= \\lambda^2 - 1 = 0", **kw),
            OldTex("= \\lambda^2 - 1 = 0", **kw),
            OldTex("= (1 - \\lambda)(-1 - \\lambda) = 0", **kw),
        )

        lpm1s = VGroup()
        for rhs, det_term in zip(rhss, det_terms):
            rhs.next_to(det_term, RIGHT, SMALL_BUFF, submobject_to_align=rhs[0])
            self.play(LaggedStart(
                det_path_anim(det_term[-1]),
                TransformMatchingShapes(det_term[-1].get_entries().copy(), rhs),
                lag_ratio=0.5
            ))
            if rhs is rhss[-1]:
                continue

            self.play(FlashUnder(rhs))
            self.wait()
            lpm1 = OldTex("\\lambda = \\pm 1", **kw)
            lpm1.next_to(rhs, DOWN, buff=0.75)
            lpm1s.add(lpm1)
            self.play(TransformMatchingShapes(rhs[1:4].copy(), lpm1))
            self.wait()

        # Comment on last
        rect = SurroundingRectangle(matrices[2])
        words = Text("Already diagonal!", font_size=24)
        words.next_to(rect, UP)
        words.set_color(YELLOW)
        diag_rects = get_diag_rects(matrices[2], color=EIGEN_COLORS[1])

        self.play(FadeInFromLarge(rect))
        self.play(Write(words, run_time=1))
        self.wait()
        self.play(ShowCreation(diag_rects))
        self.wait()

        self.play(LaggedStartMap(FadeOut, VGroup(
            *arrows, *det_terms, *rhss, *lpm1s,
            rect, words, diag_rects,
        )))

        # Show combinations
        colors = [YELLOW_B, YELLOW_C, YELLOW_D]

        def get_combo(triplet):
            group = VGroup(
                OldTex("a"), triplet[0].copy(), OldTex("+"),
                OldTex("b"), triplet[1].copy(), OldTex("+"),
                OldTex("c"), triplet[2].copy()
            )
            group.arrange(RIGHT, buff=0.15)
            group[3].align_to(group[0], DOWN)
            group[6].align_to(group[0], DOWN)
            group.set_color(WHITE)
            group[0::3].set_submobject_colors_by_gradient(*colors)
            return group

        true_lhss = VGroup(*(lhs[0][:2] for lhs in lhss))
        row1 = get_combo(true_lhss)
        row1[1::3].shift(0.06 * DOWN)
        row2 = get_combo(matrices)
        rows = VGroup(row1, row2)
        rows.arrange(DOWN, buff=LARGE_BUFF)
        rows.to_corner(UR, buff=LARGE_BUFF)

        kw = {"tex_to_color_map": {
            "a": colors[0],
            "b": colors[1],
            "c": colors[2],
            "\\lambda": EIGEN_COLORS[1],
        }}
        normalized_eq = OldTex("\\left(a^2 + b^2 + c^2 = 1\\right)", **kw)
        normalized_eq.next_to(row2, DOWN, LARGE_BUFF)
        normalized_eq.to_edge(DOWN)

        full_mat = Matrix(
            [
                ["c", "a - bi"],
                ["a + bi", "-c"],
            ],
            element_to_mobject_config=kw,
            h_buff=1.5
        )
        full_mat.next_to(row2, DOWN, LARGE_BUFF)

        self.play(TransformMatchingShapes(true_lhss.copy(), row1))
        self.wait()
        self.play(FadeTransformPieces(row1.copy(), row2))
        self.wait()
        self.play(TransformMatchingShapes(row2.copy(), full_mat))
        self.wait()  # Let some physics happen
        self.play(TransformMatchingShapes(row2[0::3].copy(), normalized_eq))
        self.wait()

        # Talk about eigen computations
        self.play(FadeOut(lhss), FadeOut(matrices))

        randy = Randolph()
        randy.set_height(2)
        randy.next_to(full_mat, LEFT).to_edge(DOWN)
        randy.set_opacity(0)
        diag_rects = get_diag_rects(full_mat)
        bubble = ThoughtBubble(height=2, width=2)
        bubble.pin_to(randy)
        m_eq = OldTex("m = 0", tex_to_color_map={"m": MEAN_COLOR})
        p_eq = OldTex("p = ??", tex_to_color_map={"p": PROD_COLOR})
        bubble.position_mobject_inside(m_eq)
        bubble.position_mobject_inside(p_eq)

        p_eq2 = OldTex("p = -1", tex_to_color_map={"p": PROD_COLOR})
        p_eq2.next_to(full_mat, RIGHT, aligned_edge=DOWN)

        self.play(
            VFadeIn(randy),
            randy.animate.set_opacity(1).change("hesitant", full_mat)
        )
        self.play(ShowCreation(diag_rects))
        self.play(
            Write(bubble),
            Write(m_eq),
            randy.change("pondering", bubble)
        )
        self.play(randy.change("thinking", bubble))
        self.play(Blink(randy))
        self.wait()
        self.play(
            FadeIn(p_eq, 0.25 * UP),
            m_eq.animate.next_to(full_mat, RIGHT, aligned_edge=UP),
            randy.change("hesitant", full_mat),
            FadeOut(diag_rects),
        )
        self.play(det_path_anim(full_mat))
        self.play(Blink(randy))
        self.wait()
        self.play(
            FadeOut(bubble),
            randy.change("tease", p_eq2),
            Transform(p_eq, p_eq2),
        )
        self.play(Blink(randy))
        self.wait()

        # Characteristic polynomial
        mod_mat = Matrix(
            [
                ["c - \\lambda", "a - bi"],
                ["a + bi", "-c - \\lambda"],
            ],
            element_to_mobject_config=kw,
            h_buff=2.0
        )
        parens = OldTex("(", ")")
        parens.stretch(2, 1)
        parens.match_height(mod_mat)
        parens[0].next_to(mod_mat, LEFT, SMALL_BUFF)
        parens[1].next_to(mod_mat, RIGHT, SMALL_BUFF)
        det = Text("det")
        det.next_to(parens, LEFT, SMALL_BUFF)
        char_poly = VGroup(det, parens, mod_mat)
        char_poly.next_to(randy, UL)

        self.play(
            TransformMatchingShapes(full_mat.copy(), char_poly, run_time=2),
            randy.change("raise_left_hand", char_poly),
        )
        self.play(randy.change("horrified", char_poly))
        self.play(Blink(randy))
        self.wait()
        self.play(randy.change("tired"))
        self.wait()


class ThreeSpinExamples(Scene):
    def construct(self):
        frame = self.camera.frame
        frame.focal_distance = 20
        frame.reorient(-20, 70)

        axes = ThreeDAxes(
            (-1, 1), (-1, 1), (-1, 1),
            axis_config={"tick_size": 0, "include_tip": False}
        )
        axes.insert_n_curves(10)
        axes.flat_stroke = False
        axes.set_stroke(WHITE, 3)
        axes.apply_depth_test()

        all_axes = axes.get_grid(3, 1)
        all_axes.arrange(IN, buff=0.8)
        self.add(all_axes)

        for axes, vect, color in zip(all_axes, [RIGHT, UP, OUT], [RED, GREEN, BLUE]):
            sphere = Sphere()
            sphere.scale(0.9)
            mesh = SurfaceMesh(sphere, resolution=(21, 11))
            mesh.set_stroke(color, width=1, opacity=0.8)
            mesh.apply_matrix(z_to_vector(vect))
            sphere = Group(sphere, mesh)
            sphere.scale(0.5)
            sphere.move_to(axes)
            axes.sphere = sphere
            sphere.vect = vect
            sphere.add_updater(lambda m, dt: m.rotate(dt, m.vect))
            self.add(sphere)

        self.wait(2 * TAU)


class GeneralDirection(Scene):
    def construct(self):
        frame = self.camera.frame
        frame.reorient(-30, 70)
        frame.add_updater(lambda m, dt: m.increment_theta(0.01 * dt))
        self.add(frame)

        axes = ThreeDAxes(
            (-2, 2), (-2, 2), (-4, 4),
            axis_config={'include_tip': False},
            depth=6,
        )
        axes.set_stroke(width=1)
        self.add(axes)

        vect = axes.c2p(2, 2, 2)
        vector = Vector(vect)
        vector.set_fill(YELLOW)

        label = Matrix([["a"], ["b"], ["c"]], v_buff=0.6)
        # label.get_entries().set_submobject_colors_by_gradient(
        #     YELLOW_B, YELLOW_C, YELLOW_D
        # )
        label.get_entries().set_color(YELLOW)
        label.rotate(89 * DEGREES, RIGHT)
        label.next_to(vector.get_end(), RIGHT)

        sphere = Sphere()
        sphere.scale(0.5)
        mesh = SurfaceMesh(sphere, resolution=(21, 11))
        mesh.set_stroke(BLUE, width=0.5, opacity=0.5)
        for mob in sphere, mesh:
            mob.apply_matrix(z_to_vector(vect))
            mob.add_updater(lambda m, dt: m.rotate(0.75 * dt, axis=vect))

        axes.apply_depth_test()
        for sm in axes.get_family():
            sm.insert_n_curves(10)

        self.add(vector, sphere, mesh)
        self.play(
            GrowArrow(vector),
            UpdateFromAlphaFunc(sphere, lambda m, a: m.set_opacity(a)),
            Write(mesh),
        )
        self.play(Write(label))
        self.wait(6)
        self.play(
            vector.animate.scale(0.5, about_point=ORIGIN),
            label.animate.shift(-0.5 * vect)
        )
        self.wait(24)


class TwoValuesEvenlySpaceAroundZero(Scene):
    def construct(self):
        nl = NumberLine((-2, 2, 0.25), width=6)
        nl.add_numbers(np.arange(-2, 3, 1.0), num_decimal_places=1, font_size=24)
        d_tracker = ValueTracker(2)
        get_d = d_tracker.get_value
        dots = VGroup(*(Dot() for x in range(2)))
        labels = VGroup(OldTex("\\lambda_1"), OldTex("\\lambda_2"))
        for group in dots, labels:
            group.set_submobject_colors_by_gradient(*EIGEN_COLORS)

        def update_dots(dots):
            dots[0].move_to(nl.n2p(-get_d()))
            dots[1].move_to(nl.n2p(get_d()))

        def update_labels(labels):
            for label, dot in zip(labels, dots):
                label.match_color(dot)
                label.next_to(dot, UP, SMALL_BUFF)

        dots.add_updater(update_dots)
        labels.add_updater(update_labels)

        prod_label = VGroup(
            labels.copy().clear_updaters().arrange(RIGHT, buff=SMALL_BUFF),
            OldTex("="),
            DecimalNumber(-4),
        )
        prod_label[-1].match_height(prod_label[0])
        prod_label.arrange(RIGHT)
        prod_label.next_to(labels, UP, LARGE_BUFF)
        prod_label[-1].add_updater(lambda m: m.set_value(-get_d()**2))

        self.add(nl, dots, labels, prod_label)

        self.play(d_tracker.animate.set_value(0.5), run_time=3)
        self.play(d_tracker.animate.set_value(1.5), run_time=3)
        self.play(d_tracker.animate.set_value(1.0), run_time=2)
        self.wait()


class MPIsSolvingCharPoly(TeacherStudentsScene):
    def construct(self):
        formula = OldTex(
            "{m} \\pm \\sqrt{\\,{m}^2 - {p}}",
            tex_to_color_map={"{m}": MEAN_COLOR, "{p}": PROD_COLOR},
            font_size=72
        )
        char_poly = get_det_mod_mat(get_mod_mat([["a", "b"], ["c", "d"]]))
        eq0 = OldTex("=0")
        eq0.next_to(char_poly, RIGHT, SMALL_BUFF)
        char_poly.add(eq0)
        char_poly.move_to(self.hold_up_spot, DOWN)
        arrow = Arrow(RIGHT, LEFT)
        arrow.next_to(char_poly, LEFT)

        self.teacher_holds_up(formula)
        self.play_student_changes(
            "happy", "thinking", "tease",
            look_at=formula
        )
        self.wait()
        self.play(
            self.teacher.change("hooray", char_poly),
            formula.animate.next_to(arrow, LEFT),
            FadeIn(char_poly, 0.5 * UP),
        )
        self.play_student_changes(
            "erm", "pondering", "pondering",
            look_at=char_poly,
            added_anims=[GrowArrow(arrow)]
        )
        self.wait(3)


class QuadraticPolynomials(Scene):
    def construct(self):
        # Set up equation
        kw = {
            "tex_to_color_map": {
                "\\lambda_1": EIGEN_COLORS[0],
                "\\lambda_2": EIGEN_COLORS[1],
                "=": WHITE,
            }
        }
        equation = VGroup(
            OldTex("x^2 - 10x + 9", **kw)[0],
            OldTex("=").rotate(PI / 2),
            OldTex("x^2 - (\\lambda_1 + \\lambda_2)x + \\lambda_1 \\lambda_2", **kw),
            OldTex("=").rotate(PI / 2),
            OldTex("(x - \\lambda_1)(x - \\lambda_2)", **kw),
        )
        equation.arrange(DOWN)
        equation.to_edge(LEFT)
        line1, eq1, line2, eq2, line3 = equation
        for line in line2, line3:
            line.set_submobjects(line.family_members_with_points())

        line3.save_state()
        line3.move_to(line2)

        # Graph
        axes = Axes(
            (-5, 10),
            (-20, 20),
            height=7,
            width=FRAME_WIDTH / 2,
        )
        axes.y_axis.ticks.stretch(0.75, 0)
        axes.to_edge(RIGHT)
        graph = axes.get_graph(lambda x: x**2 - 10 * x + 9)
        graph.set_color(BLUE)
        graph_label = line1.copy()
        graph_label.set_color(BLUE)
        graph_label.next_to(graph.get_end(), UP)
        graph_label.to_edge(RIGHT)

        root_dots = VGroup()
        root_labels = VGroup()
        for i, n, vect in zip((0, 1), (1, 9), (RIGHT, LEFT)):
            dot = Dot(axes.c2p(n, 0), color=EIGEN_COLORS[i])
            label = OldTex(f"\\lambda_{i + 1}")
            label.match_color(dot)
            label.next_to(dot, UP + vect, buff=0.05)
            root_dots.add(dot)
            root_labels.add(label)
        root_dots.set_stroke(BLACK, 5, background=True)

        self.play(Write(axes))
        self.play(
            ShowCreation(graph, run_time=3),
            FadeIn(graph_label, UP)
        )
        self.play(
            LaggedStartMap(GrowFromCenter, root_dots),
            LaggedStart(
                GrowFromPoint(root_labels[0], root_dots[0].get_center()),
                GrowFromPoint(root_labels[1], root_dots[1].get_center()),
            )
        )
        self.wait()

        # Animate to equation
        self.play(TransformFromCopy(graph_label, line1))
        self.play(
            Write(eq1),
            TransformMatchingShapes(root_labels.copy(), line3)
        )
        self.wait()
        self.play(Restore(line3))
        xs = VGroup(line3[1], line3[7])
        self.play(TransformFromCopy(xs, line2[:2]))  # x^2
        self.play(LaggedStart(  # x
            AnimationGroup(
                TransformFromCopy(line3[1], line2[10].copy()),
                TransformFromCopy(line3[9:11], line2[7:9]),
                FadeIn(line2[2])
            ),
            AnimationGroup(
                TransformFromCopy(line3[7], line2[10].copy(), remover=True),
                TransformFromCopy(line3[3:5], line2[4:6]),
                FadeIn(VGroup(line2[3], line2[6], line2[9]))
            ),
        ))
        self.play(
            TransformFromCopy(
                VGroup(*line3[3:5], *line3[9:11]),
                VGroup(*line2[12:16]),
            ),
            FadeIn(line2[11]),
            FadeIn(eq2),
        )
        self.wait()

        # Highlight sum and product
        for i, j, k, l in [(3, 5, 3, 10), (7, 8, 12, 16)]:
            self.play(
                FadeIn(SurroundingRectangle(line1[i:j])),
                FadeIn(SurroundingRectangle(line2[k:l])),
                run_time=2,
                rate_func=there_and_back_with_pause,
                remover=True
            )

        # Show mean and product
        line2 = VGroup(*line2.family_members_with_points())
        line3 = VGroup(*line3.family_members_with_points())
        quad_terms = VGroup(line1[0:2], line2[0:2])
        lin_terms = VGroup(line1[2:5], line2[2:10])
        const_terms = VGroup(line1[6:], line2[11:])

        mean_arrow = Vector(UP)
        mean_arrow.next_to(lin_terms[0], UP, MED_SMALL_BUFF)
        times_half = OldTex("\\times -\\frac{1}{2}", font_size=24)
        times_half.next_to(mean_arrow, RIGHT, buff=0)
        m_eq = OldTex(
            "{\\lambda_1 + \\lambda_2 \\over 2}", "=", "5",
            tex_to_color_map={"\\lambda_1": EIGEN_COLORS[0], "\\lambda_2": EIGEN_COLORS[1]}
        )
        m_eq.next_to(mean_arrow, UP, SMALL_BUFF, submobject_to_align=m_eq[-1])
        m_eq[:-2].scale(0.7, about_edge=RIGHT)
        m_dot = Dot(axes.c2p(5, 0))
        m_dot.scale(0.5)
        m_label = Integer(5, font_size=30)
        m_label.next_to(m_dot, UP, SMALL_BUFF)

        p_rects = VGroup(*map(SurroundingRectangle, const_terms))
        p_rects.set_stroke(PROD_COLOR)

        fade_rect = FullScreenFadeRectangle()
        fade_rect.set_fill(BLACK, 0.75)
        fade_rect.replace(equation, stretch=True)

        self.add(fade_rect, *quad_terms)
        self.play(FadeIn(fade_rect))
        self.wait()
        self.add(fade_rect, *lin_terms)
        self.wait()
        self.play(
            GrowArrow(mean_arrow),
            FadeIn(times_half, shift=0.25 * UP, scale=2),
            FadeIn(m_eq[:-1])
        )
        self.play(Write(m_eq[-1]))
        self.play(
            FadeTransform(m_eq[-1].copy(), m_dot),
            FadeTransform(m_eq[-1].copy(), m_label),
        )
        self.wait()
        self.add(fade_rect, *const_terms)
        VGroup(mean_arrow, times_half).set_opacity(0.5)
        self.play(ShowCreation(p_rects))
        self.wait()

        # Show final roots
        d_terms = VGroup(*(
            OldTex(u, "\\sqrt{25 - 9}", font_size=24)
            for u in ["+", "-"]
        ))
        my = m_label.get_y()
        arrows = VGroup(
            Arrow(m_label.get_left(), [root_dots[0].get_left()[0], my, 0], buff=0.1),
            Arrow(m_label.get_right(), [root_dots[1].get_right()[0], my, 0], buff=0.1),
        )
        for d_term, arrow in zip(d_terms, arrows):
            d_term.next_to(arrow, UP, buff=0)

        self.play(FadeOut(p_rects))
        self.play(
            TransformMatchingShapes(equation[0][-2:].copy(), d_terms),
            FadeOut(root_labels),
            *map(GrowArrow, arrows)
        )
        self.wait()
        self.play(
            FadeOut(fade_rect),
        )
        self.wait()


class SimplerQuadraticFormula(Scene):
    def construct(self):
        # Show comparison
        form1 = OldTex(
            "{m} \\pm \\sqrt{\\,{m}^2 - {p}}",
            tex_to_color_map={"{m}": MEAN_COLOR, "{p}": PROD_COLOR},
            font_size=72
        )
        words = OldTexText("takes less to\\\\remember than")
        form2 = OldTex(
            "{-b \\pm \\sqrt{\\,b^2 - 4ac} \\over 2a}",
            tex_to_color_map={
                "a": MAROON_A,
                "b": MAROON_B,
                "c": MAROON_C,
            },
            font_size=72
        )
        group = VGroup(form1, words, form2)
        group.arrange(DOWN, buff=1.5)

        self.add(form1)
        self.play(Write(words, run_time=1))
        self.play(FadeIn(form2, DOWN))
        self.wait()


class SkipTheMiddleStep(Scene):
    def construct(self):
        matrix = IntegerMatrix([[3, 1], [4, 1]])
        matrix.set_column_colors(*COL_COLORS)
        char_poly = OldTex(
            "\\lambda^2 - 4\\lambda - 1",
            tex_to_color_map={"\\lambda": EIGEN_COLORS[0]}
        )
        mp_formula = OldTex(
            "{2} \\pm \\sqrt{\\,{2}^2 - (-1)}",
            tex_to_color_map={
                "{2}": MEAN_COLOR,
                "(-1)": PROD_COLOR,
            }
        )

        group = VGroup(matrix, char_poly, mp_formula)
        group.arrange(RIGHT, buff=2)

        mat_label, char_poly_label, eigen_label = labels = VGroup(
            OldTexText("Matrix"),
            OldTexText("Characteristic\\\\polynomial"),
            OldTexText("Eigenvalues"),
        )
        for label, mob, color in zip(labels, group, [COL_COLORS[0], EIGEN_COLORS[0], MEAN_COLOR]):
            label.set_color(color)
            label.next_to(mob, DOWN, MED_LARGE_BUFF)
            label.align_to(labels[0], UP)

        arc = -90 * DEGREES
        arrows = VGroup(
            Arrow(matrix.get_corner(UR), char_poly.get_top(), path_arc=arc),
            Arrow(char_poly.get_top(), mp_formula.get_top(), path_arc=arc),
            Arrow(matrix.get_corner(UR), mp_formula.get_top(), path_arc=0.8 * arc),
        )
        arrows.shift(0.5 * UP)

        self.add(matrix, mat_label)
        self.play(
            DrawBorderThenFill(arrows[0]),
            GrowFromPoint(char_poly, matrix.get_top(), path_arc=arc),
            FadeTransform(mat_label.copy(), char_poly_label),
        )
        self.wait()
        self.play(
            DrawBorderThenFill(arrows[1]),
            TransformFromCopy(char_poly, mp_formula),
            FadeTransform(char_poly_label.copy(), eigen_label),
        )
        self.wait()
        self.play(
            VFadeInThenOut(get_diag_rects(matrix), run_time=2),
            LaggedStart(*(
                ShowCreationThenFadeOut(SurroundingRectangle(sm, buff=0.1, color=WHITE), run_time=2)
                for sm in mp_formula.get_parts_by_tex("{2}")
            ))
        )
        self.play(
            det_path_anim(matrix),
            ShowCreationThenFadeOut(SurroundingRectangle(
                mp_formula.get_parts_by_tex("(-1)"), buff=0.1, color=WHITE,
            ), run_time=2)
        )
        self.wait()
        self.play(
            Transform(arrows[0], arrows[2]),
            Transform(arrows[1], arrows[2]),
            VGroup(char_poly, char_poly_label).animate.set_opacity(0.15),
        )
        self.wait()

        #
        frame = self.camera.frame
        quad_form = OldTex(
            "{-b \\pm \\sqrt{\\,b^2 - 4ac} \\over 2a}",
            tex_to_color_map={
                "a": BLUE_B,
                "b": BLUE_C,
                "c": BLUE_D,
            }
        )
        words = OldTexText("Never could\\\\have worked!")
        group = VGroup(quad_form, words)
        group.arrange(RIGHT, buff=LARGE_BUFF)
        group.next_to(arrows, UP, LARGE_BUFF)
        cross = Cross(quad_form)
        cross.set_stroke(width=(1, 3, 3, 1))

        self.play(
            frame.animate.set_y(1),
            FadeIn(quad_form, DOWN),
        )
        self.wait()
        self.play(
            Write(words),
            ShowCreation(cross)
        )
        self.wait()


class GeneralCharPoly(Scene):
    def construct(self):
        t2c = {
            "a": COL_COLORS[0],
            "b": COL_COLORS[1],
            "c": COL_COLORS[0],
            "d": COL_COLORS[1],
            "\\lambda": EIGEN_COLORS[1],
            "-": WHITE,
            "+": WHITE,
            "=": WHITE,
        }
        det = get_det_mod_mat(get_mod_mat([["a", "b"], ["c", "d"]], t2c=t2c))
        rhs1 = OldTex("= (a - \\lambda)(d - \\lambda) - bc", tex_to_color_map=t2c)
        rhs2 = OldTex("= \\lambda^2 - (a + d)\\lambda + (ad - bc)", tex_to_color_map=t2c)

        rhs1.next_to(det, RIGHT)
        rhs2.next_to(rhs1, DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        VGroup(det, rhs1, rhs2).center()

        self.add(det)
        self.wait()
        self.play(TransformMatchingShapes(
            det[-1].get_entries()[0::3].copy(),
            rhs1[:10],
        ))
        self.play(TransformMatchingShapes(
            det[-1].get_entries()[1:3].copy(),
            rhs1[10:],
            path_arc=45 * DEGREES,
        ))
        self.wait()
        self.play(TransformMatchingShapes(rhs1.copy(), rhs2, run_time=1.5))
        self.wait()

        self.play(ShowCreation(SurroundingRectangle(rhs2[3:9], buff=0.05, color=MEAN_COLOR)))
        self.play(ShowCreation(SurroundingRectangle(rhs2[11:], buff=0.05, color=PROD_COLOR)))
        self.wait()

        self.embed()


class EndScreen(PatreonEndScreen):
    pass
