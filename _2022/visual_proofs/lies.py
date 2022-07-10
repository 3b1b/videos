from manim_imports_ext import *


EQUATOR_STYLE = dict(stroke_color=TEAL, stroke_width=2)


def get_sphere_slices(radius=1.0, n_slices=20):
    delta_theta = TAU / n_slices
    north_slices = Group(*(
        ParametricSurface(
            uv_func=lambda u, v: [
                radius * math.sin(v) * math.cos(u),
                radius * math.sin(v) * math.sin(u),
                radius * math.cos(v),
            ],
            u_range=[theta, theta + delta_theta],
            v_range=[0, PI / 2],
            resolution=(4, 25),
        )
        for theta in np.arange(0, TAU, delta_theta)
    ))
    north_slices.set_x(0)
    color_slices(north_slices)

    equator = Circle(**EQUATOR_STYLE)
    equator.insert_n_curves(100)
    equator.match_width(north_slices)
    equator.move_to(ORIGIN)
    equator.apply_depth_test()

    return Group(north_slices, get_south_slices(north_slices, dim=2), equator)


def get_flattened_slices(radius=1.0, n_slices=20, straightened=True):
    slc = ParametricSurface(
        # lambda u, v: [u * v, 1 - v, 0],
        lambda u, v: [u * math.sin(v * PI / 2), 1 - v, 0],
        u_range=[-1, 1],
        v_range=[0, 1],
        resolution=(4, 25),
    )
    slc.set_width(TAU / n_slices, stretch=True)
    slc.set_height(radius * PI / 2)
    north_slices = slc.get_grid(1, n_slices, buff=0)
    north_slices.move_to(ORIGIN, DOWN)
    color_slices(north_slices)
    equator = Line(
        north_slices.get_corner(DL), north_slices.get_corner(DR),
        **EQUATOR_STYLE,
    )

    return Group(north_slices, get_south_slices(north_slices, dim=1), equator)


def color_slices(slices, colors=(BLUE_D, BLUE_E)):
    for slc, color in zip(slices, it.cycle([BLUE_D, BLUE_E])):
        slc.set_color(color)
    return slices


def get_south_slices(north_slices, dim):
    ss = north_slices.copy().stretch(-1, dim, about_point=ORIGIN)
    for slc in ss:
        slc.reverse_points()
    return ss


# Scenes


class PreviewThreeExamples(InteractiveScene):
    def construct(self):
        # Setup
        self.add(FullScreenRectangle())
        rects = Rectangle(3.5, 4.5).replicate(3)
        rects.set_stroke(WHITE, 2)
        rects.set_fill(BLACK, 1)
        rects.arrange(RIGHT, buff=0.75)
        rects.to_edge(DOWN, buff=0.25)

        # Titles
        titles = VGroup(
            TexText("S.A. = $\\pi^2 R^2$"),
            Tex("\\pi = 4").scale(1.5),
            Text("All triangles\nare isosceles"),
        )
        titles.set_width(rects[0].get_width())
        for title, rect in zip(titles, rects):
            title.next_to(rect, UP)

        self.play(
            LaggedStartMap(DrawBorderThenFill, rects, lag_ratio=0.3),
            LaggedStartMap(FadeIn, titles, shift=0.25 * UP, lag_ratio=0.3),
        )
        self.wait()

        # Increasing subtlety
        arrow = Arrow(LEFT, RIGHT, stroke_width=10)
        arrow.set_width(FRAME_WIDTH - 1)
        arrow.next_to(titles, UP)
        arrow.set_stroke(opacity=(0.5, 0.9, 1))
        words = Text("Increasingly subtle", font_size=72, color=YELLOW)
        words.next_to(arrow, UP)
        VGroup(arrow, words).to_edge(UP)
        arrow.set_x(0)
        arrow.set_color(YELLOW)

        self.play(
            ShowCreation(arrow),
            FadeIn(words, lag_ratio=0.1),
        )
        self.wait()


class Intro(TeacherStudentsScene):
    CONFIG = {"background_color": BLACK}

    def construct(self):
        morty = self.teacher
        ss = self.students

        self.play(morty.change("raise_right_hand"))
        self.play_student_changes(
            "pondering", "angry", "confused",
            look_at=self.screen
        )
        self.wait()
        self.play(ss[0].change("worried", self.screen))
        self.wait()
        self.play(ss[1].change("pondering", self.screen))
        self.wait(3)


class SimpleSphereQuestion(InteractiveScene):
    def construct(self):
        # Intro statement
        text = TexText("A ``proof'' of a formula for\\\\the surface area of a sphere")
        text.set_stroke(WHITE, 0)

        rect = Rectangle(width=TAU, height=TAU / 4)
        rect.set_stroke(width=0)
        rect.set_fill(GREY_E, 1)
        rect.set_width(FRAME_WIDTH)
        self.add(rect)

        morty = Mortimer(height=1.5)
        randy = Randolph(height=1.5)
        morty.next_to(text, RIGHT)
        randy.next_to(text, LEFT)
        VGroup(morty, randy).align_to(rect, DOWN).shift(0.1 * UP)
        self.add(morty, randy)

        self.play(
            Write(text),
            morty.change("raise_right_hand", text),
            randy.change("thinking", text)
        )
        self.play(Blink(morty))
        self.wait()


class SphereExample(InteractiveScene):
    radius = 2.0
    n_slices = 20
    slice_stroke_width = 1.0
    # show_true_slices = False
    show_true_slices = True

    def construct(self):
        # Setup
        frame = self.camera.frame
        frame.set_focal_distance(100)
        light = self.camera.light_source
        light.move_to([-10, 2, 5])

        # Create the sphere
        img_path = "/Users/grant/Dropbox/3Blue1Brown/videos/2022/visual_proofs/lies/images/SimpleSphereQuestion.png"
        radius = 2.5
        sphere = TexturedSurface(Sphere(radius=radius), img_path)
        sphere.set_opacity(1.0)
        sphere.rotate(91 * DEGREES, OUT).rotate(80 * DEGREES, LEFT)
        mesh = SurfaceMesh(sphere)
        mesh.set_stroke(BLUE_B, 1, 0.5)
        banner = TexturedSurface(Surface(resolution=sphere.resolution), img_path)
        banner.set_width(FRAME_WIDTH)
        banner.set_height(FRAME_WIDTH / 4, stretch=True)
        banner.center()
        banner.set_gloss(0)
        banner.set_reflectiveness(0)
        banner.set_shadow(0)

        self.add(banner)
        self.play(ReplacementTransform(banner, sphere, run_time=2))
        self.play(Write(mesh, run_time=1))
        self.wait()

        # Slice sphere
        slices = get_sphere_slices(n_slices=self.n_slices)
        slices.rotate(90 * DEGREES, OUT).rotate(80 * DEGREES, LEFT)
        slices.scale(radius)
        slice_highlights = slices[0][len(slices[0]) // 4:3 * len(slices[0]) // 4].copy().set_color(YELLOW)
        slice_highlights.scale(1.01, about_point=ORIGIN)

        flat_slices = get_flattened_slices(n_slices=self.n_slices)
        flat_slices.to_edge(RIGHT, buff=1.0)

        self.play(
            FadeIn(slices),
            FadeOut(sphere, lag_ratio=0, scale=0.95),
            FadeOut(mesh, lag_ratio=0, scale=0.95),
        )
        self.play(LaggedStart(*(
            FadeIn(sh, rate_func=there_and_back)
            for sh in slice_highlights
        ), lag_ratio=0.35, run_time=1.5))
        self.remove(slice_highlights)
        self.wait()

        # Unfold sphere
        self.play(slices.animate.scale(1 / radius).to_corner(UL).shift(IN))
        pre_slices = slices.copy()
        self.add(pre_slices, slices)
        for slcs in pre_slices:
            for slc in slcs:
                slc.set_color(interpolate_color(slc.get_color(), BLACK, 0.0))
        flat_slices[2].shift(0.01 * OUT)
        self.play(
            Transform(slices[0], flat_slices[0]),
            Transform(slices[2], flat_slices[2]),
            run_time=2,
        )
        self.wait()
        self.play(
            Transform(
                slices[1], flat_slices[1],
                run_time=2,
            ),
        )
        self.wait()

        # Show width line
        slc = flat_slices[0][0]
        v_tracker = ValueTracker(0)
        width_line = Line(LEFT, RIGHT)
        width_line.set_stroke(RED, 3)

        def update_width_line(width_line, slc=slc, v_tracker=v_tracker):
            v = v_tracker.get_value()
            width_line.set_width(1.2 * slc.get_width() * math.sin(v) + 1e-2)
            width_line.move_to(interpolate(slc.get_top(), slc.get_bottom(), v))

        width_line.add_updater(update_width_line)
        self.add(width_line)
        self.play(v_tracker.animate.set_value(1), run_time=3)
        self.play(v_tracker.animate.set_value(0), run_time=3)
        self.remove(width_line)

        # Interlink
        tri_template = Triangle(start_angle=90 * DEGREES)
        tri_template.set_width(2).set_height(1, stretch=True)
        tri_template.move_to(ORIGIN, DOWN)

        if self.show_true_slices:
            tri_template = VMobject()
            dtheta = TAU / self.n_slices
            curve = ParametricCurve(lambda phi: [-math.sin(phi) * dtheta / 2, PI / 2 - phi, 0], t_range=(0, PI / 2))
            curve2 = curve.copy().stretch(-1, 0, about_point=ORIGIN)
            curve2.reverse_points()
            tri_template.append_vectorized_mobject(curve)
            tri_template.add_line_to(curve2.get_start())
            tri_template.append_vectorized_mobject(curve2)

        vslices = VGroup(*(
            VGroup(*(
                tri_template.copy().rotate(rot).replace(slc, stretch=True)
                for slc in hemi
            ))
            for rot, hemi in zip([0, PI], slices)
        ))
        for hemi, vhemi in zip(slices, vslices):
            for slc, vslc in zip(hemi, vhemi):
                vslc.set_fill(slc.get_color(), 1)
                vslc.set_stroke(WHITE, 0)
        slices[2].deactivate_depth_test()
        vslices.add(slices[2].copy())

        vslices[1].move_to(vslices[0][0].get_top(), UL)
        vslices[1].set_stroke(WHITE, self.slice_stroke_width)
        vslices.center()

        self.play(FadeTransformPieces(slices, vslices))
        self.wait()

        if self.show_true_slices:
            self.play(vslices.animate.set_opacity(0.5))

        # Show equator
        circ_label = Text("Circumference")
        circ_label.next_to(vslices[2], DOWN)
        circ_formula = Tex("2\\pi R")
        circ_formula.next_to(vslices[2], DOWN)
        circ_formula.set_stroke(WHITE, 0)
        equator = pre_slices[2]

        vslices[2].set_stroke()
        self.play(
            Write(circ_label),
            VShowPassingFlash(
                vslices[2].copy().set_stroke(YELLOW, 5).insert_n_curves(20),
                time_width=1.5,
                run_time=1.5,
            ),
            vslices[2].animate.set_color(YELLOW),
        )
        self.play(equator.animate.shift(1.5 * DOWN).set_color(YELLOW))
        self.wait()
        self.play(equator.animate.shift(1.5 * UP))
        self.wait()
        self.play(
            Write(circ_formula),
            circ_label.animate.next_to(circ_formula, DOWN)
        )
        self.wait()

        # Arc height
        edge = Line(vslices.get_corner(DL), vslices[0][0].get_top())
        edge.set_stroke(PINK, 2)
        q_marks = Tex("???")
        q_marks.next_to(edge.get_center(), LEFT, SMALL_BUFF)
        arc = Arc(0, 90 * DEGREES)
        arc.match_style(edge)
        arc.set_height(pre_slices.get_height() / 2)
        arc.rotate(-10 * DEGREES, LEFT)
        arc.shift(pre_slices[0][0].get_points()[0] - arc.get_end())

        arc_form = Tex("{\\pi \\over 2} R")
        arc_form.scale(0.5)
        arc_form.next_to(arc.pfp(0.5), RIGHT)
        arc_form2 = arc_form.copy()
        arc_form2.scale(1.5)
        arc_form2.move_to(q_marks, RIGHT)

        self.play(
            ShowCreation(edge),
            Write(q_marks),
        )
        self.play(WiggleOutThenIn(edge, run_time=1))
        self.wait()
        self.play(TransformFromCopy(edge, arc))
        self.play(Write(arc_form))
        self.wait()
        self.play(
            TransformFromCopy(arc_form, arc_form2),
            FadeOut(q_marks, DOWN)
        )
        self.wait()

        # Area
        arc_tex = "{\\pi \\over 2} R"
        circ_tex = "2\\pi R"
        eq_parts = ["\\text{Area}", "=", arc_tex, "\\times", circ_tex, "=", "\\pi^2 R^2"]
        equation = MTex(" ".join(eq_parts), isolate=eq_parts)
        equation.center().to_edge(UP, buff=LARGE_BUFF)
        rect = SurroundingRectangle(equation.select_parts(eq_parts[-1]))
        rect.set_stroke(YELLOW, 2)

        self.play(
            Write(equation.select_parts("\\text{Area}")),
            Write(equation.select_parts("=")[0]),
            Write(equation.select_parts("\\times")),
            TransformFromCopy(arc_form2, equation.select_parts(arc_tex)),
            TransformFromCopy(circ_formula, equation.select_parts(circ_tex)),
        )
        self.wait()
        self.play(
            Write(equation.select_parts("=")[1]),
            Write(equation.select_parts("\\pi^2 R^2")),
        )
        self.play(ShowCreation(rect))
        self.wait()


class SphereExample50(SphereExample):
    n_slices = 50
    slice_stroke_width = 0.5


class SphereExample100(SphereExample):
    n_slices = 100
    slice_stroke_width = 0.1


class CallOutSphereExampleAsWrong(InteractiveScene):
    def construct(self):
        pass


class SomethingSomethingLimits(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.says("Something,\nsomething\nlimits!", mode="shruggie"),
            self.change_students("hesitant", "angry", "erm")
        )
        self.wait()


class PiEqualsFourOverlay(InteractiveScene):
    def construct(self):
        words = TexText("Fine, as long as\\\\$\\pi = 4$")
        words.scale(2)
        words.set_color(RED)
        self.play(Write(words, stroke_color=WHITE))
        self.wait()


class Proof2Slide(InteractiveScene):
    def construct(self):
        # Setup
        self.add(FullScreenRectangle())
        title = TexText("``Proof'' \\#2", font_size=60)
        title.to_edge(UP)
        subtitle = Tex("\\pi = 4", font_size=60)
        subtitle.next_to(title, DOWN)
        subtitle.set_color(RED)
        self.add(title, subtitle)

        # Number line
        t_tracker = ValueTracker(0)
        get_t = t_tracker.get_value

        radius = 1.25

        circle = Circle(radius=radius)
        circle.to_edge(DOWN)
        circle.set_fill(BLUE_E, 1)
        circle.set_stroke(width=0)
        circle.rotate(-TAU / 4)

        nl = NumberLine((0, 4), width=4 * circle.get_width())
        nl.add_numbers()
        nl.move_to(2 * DOWN)

        v_lines = VGroup(*(
            DashedLine(ORIGIN, circle.get_height() * UP).move_to(nl.n2p(x), DOWN)
            for x in range(5)
        ))
        v_lines.set_stroke(GREY_B, 1)

        circum = circle.copy()
        circum.set_fill(opacity=0)
        circum.set_stroke(YELLOW, 2)

        def update_circum(circum):
            line = Line(nl.n2p(0), nl.n2p(get_t() * 4))
            arc = Arc(start_angle=-TAU / 4, angle=TAU * (1 - get_t()), radius=radius)
            arc.shift(circle.get_bottom() - arc.get_start())
            circum.set_points(np.vstack([line.get_points(), arc.get_points()]))
            return circum

        circum.add_updater(update_circum)

        radial_line = Line(circle.get_bottom(), circle.get_center(), stroke_width=1)
        radial_line.add_updater(lambda m: m.set_angle(TAU * (0.75 - get_t())).shift(circle.get_center() - m.get_start()))

        circle.move_to(nl.n2p(0.5), DOWN)

        self.add(nl, circle, radial_line, circum)
        self.play(LaggedStartMap(ShowCreation, v_lines, run_time=1))

        # Roll
        self.play(circle.animate.set_x(nl.n2p(0)[0]))
        self.play(
            circle.animate.set_x(nl.n2p(4)[0]),
            t_tracker.animate.set_value(1),
            rate_func=linear,
            run_time=4,
        )
        self.wait()


class CircleExample(InteractiveScene):
    n_slices = 20
    sector_stroke_width = 1.0

    def construct(self):
        radius = 2.0

        # Slice up circle
        circle = Circle(radius=radius)
        circle.set_stroke(WHITE, 1)
        circle.set_fill(BLUE_E, 1)

        question = Text("Area?")
        question.next_to(circle, UP)

        sectors = self.get_sectors(circle, n_slices=self.n_slices)

        self.play(
            DrawBorderThenFill(circle),
            Write(question, stroke_color=WHITE)
        )
        self.wait()
        self.play(Write(sectors))
        self.remove(circle)

        # Lay out sectors
        laid_sectors = sectors.copy()
        N = len(sectors)
        dtheta = TAU / N
        angles = np.arange(0, TAU, dtheta)
        for sector, angle in zip(laid_sectors, angles):
            sector.rotate(-90 * DEGREES - angle - dtheta / 2)

        laid_sectors.arrange(RIGHT, buff=0, aligned_edge=DOWN)
        laid_sectors.move_to(1.5 * DOWN)

        self.play(
            sectors.animate.scale(0.7).to_corner(UL),
            question.animate.to_corner(UR),
        )
        self.play(TransformFromCopy(sectors, laid_sectors, run_time=2))
        self.wait()

        # Interslice
        lh, rh = laid_sectors[:N // 2], laid_sectors[N // 2:]
        lh.generate_target()
        rh.generate_target()
        rh.target.rotate(PI)
        rh.target.move_to(lh[0].get_top(), UL)
        VGroup(lh.target, rh.target).set_x(0)
        rh.target.shift(UP)
        lh.target.shift(DOWN)

        self.play(
            MoveToTarget(lh, run_time=1.5),
            MoveToTarget(rh, run_time=1.5, path_arc=PI),
        )
        self.play(
            lh.animate.shift(UP),
            rh.animate.shift(DOWN),
        )
        self.wait()
        self.play(*(
            LaggedStart(*(
                VShowPassingFlash(piece, time_width=2)
                for piece in group.copy().set_fill(opacity=0).set_stroke(RED, 5)
            ), lag_ratio=0.02, run_time=4)
            for group in [laid_sectors, sectors]
        ))

        # Side lengths
        ulp = lh[0].get_top()
        width_line = Line(ulp, rh.get_corner(UR))
        width_line.set_stroke(YELLOW, 3)
        width_form = Tex("\\pi R")
        width_form.next_to(width_line, UP)

        semi_circ = Arc(angle=PI)
        semi_circ.set_stroke(YELLOW, 3)
        semi_circ.replace(sectors)
        semi_circ.move_to(sectors, UP)

        height_line = Line(lh.get_corner(DL), ulp)
        height_line.set_stroke(PINK, 3)
        height_form = Tex("R")
        height_form.next_to(height_line, LEFT)

        radial_line = Line(sectors.get_center(), sectors.get_right())
        radial_line.match_style(height_line)
        pre_R_label = Tex("R").next_to(radial_line, UP, SMALL_BUFF)

        self.play(ShowCreation(width_line))
        self.play(TransformFromCopy(width_line, semi_circ, path_arc=-PI / 2, run_time=2))
        self.wait()
        self.play(Write(width_form, stroke_color=WHITE))
        self.wait()

        self.play(ShowCreation(height_line))
        self.play(TransformFromCopy(height_line, radial_line))
        self.play(Write(pre_R_label))
        self.play(ReplacementTransform(pre_R_label, height_form))
        self.wait()

        # Area
        rhs = Tex("=\\pi R^2")
        question.generate_target()
        question.target.match_y(sectors).match_x(lh)
        question.target[-1].scale(0, about_edge=LEFT)
        rhs.next_to(question.target, RIGHT)

        rect = SurroundingRectangle(VGroup(question.target, rhs))
        rect.set_stroke(YELLOW, 2)

        self.play(MoveToTarget(question))
        self.play(
            TransformMatchingShapes(VGroup(height_form, width_form).copy(), rhs)
        )
        self.wait()
        self.play(ShowCreation(rect))

    def get_sectors(self, circle, n_slices=20, fill_colors=[BLUE_D, BLUE_E]):
        angle = TAU / n_slices
        sectors = VGroup(*(
            Sector(angle=angle, start_angle=i * angle, fill_color=color, fill_opacity=1)
            for i, color in zip(range(n_slices), it.cycle(fill_colors))
        ))
        sectors.set_stroke(WHITE, self.sector_stroke_width)
        sectors.replace(circle, stretch=True)
        return sectors


class CircleExample50(CircleExample):
    n_slices = 50
    sector_stroke_width = 0.5


class CircleExample100(CircleExample):
    n_slices = 100
    sector_stroke_width = 0.2


class SideBySide(InteractiveScene):
    def construct(self):
        pass


class SphereSectorAnalysis(Scene):
    n_slices = 20

    def construct(self):
        # Setup
        radius = 2.0
        frame = self.camera.frame
        frame.reorient(10, 60)

        axes = ThreeDAxes()
        axes.insert_n_curves(20)
        axes.set_stroke(GREY_B, 1)
        axes.apply_depth_test()

        sphere = Sphere(radius=radius)
        mesh = SurfaceMesh(sphere, resolution=(self.n_slices + 1, 3))
        mesh.set_stroke(BLUE_E, 1)
        slices = get_sphere_slices(radius=radius, n_slices=self.n_slices)
        slices[2].scale(1.007)
        slices.rotate(-90 * DEGREES, OUT)

        self.add(axes)
        self.add(slices, mesh)
        self.play(
            ShowCreation(slices, lag_ratio=0.05),
            frame.animate.reorient(-10, 70),
            run_time=2,
        )
        self.add(slices, mesh)
        self.wait()

        # Isolate one slice
        self.play(
            slices[0][1:].animate.set_opacity(0.1),
            slices[1].animate.set_opacity(0.1),
        )
        self.wait()

        # Preview varying width
        dtheta = TAU / self.n_slices
        phi_tracker = ValueTracker(45 * DEGREES)
        get_phi = phi_tracker.get_value

        def get_width_line():
            return ParametricCurve(
                lambda t: radius * np.array([
                    math.sin(t) * math.sin(get_phi()),
                    -math.cos(t) * math.sin(get_phi()),
                    math.cos(get_phi()),
                ]),
                t_range=(0, dtheta),
                stroke_color=RED,
                stroke_width=2,
            )

        width_line = get_width_line()
        width_label = Text("Width", color=RED)
        width_label.rotate(90 * DEGREES, RIGHT)
        width_label.set_stroke(BLACK, 1, background=True)
        width_label.add_updater(lambda m: m.next_to(width_line, OUT, buff=SMALL_BUFF).set_width(1.5 * width_line.get_width() + 1e-2))

        self.play(ShowCreation(width_line), Write(width_label))
        self.wait()
        width_line.add_updater(lambda m: m.match_points(get_width_line()))
        for angle in 0, PI / 2, 0:
            self.play(phi_tracker.animate.set_value(angle), run_time=2)
        self.play(FadeOut(width_label))
        self.wait()

        # Reorient
        slices.generate_target()
        mesh.generate_target()
        for mob in slices, mesh:
            for submob in mob.target.family_members_with_points():
                if submob.get_x() < -0.1:
                    submob.set_opacity(0)

        self.add(slices, width_line)
        self.play(
            frame.animate.reorient(-65, 65),
            MoveToTarget(slices),
            MoveToTarget(mesh),
            slices[2].animate.set_stroke(width=0),
            run_time=2,
        )
        frame.add_updater(lambda f, dt: f.increment_theta(0.01 * dt))

        # Show phi angle
        def get_sphere_point():
            return radius * (math.cos(get_phi()) * OUT + math.sin(get_phi()) * DOWN)

        def get_radial_line():
            return Line(
                ORIGIN, get_sphere_point(),
                stroke_color=YELLOW,
                stroke_width=2,
            )

        phi_label = Tex("\\phi", font_size=30)

        def get_angle_label():
            arc = Arc(start_angle=90 * DEGREES, angle=-get_phi(), radius=0.5, n_components=8)
            arc.set_stroke(WHITE, 2)
            arc.set_fill(opacity=0)
            label = phi_label.copy()
            label.next_to(arc.pfp(0.5), UP, buff=SMALL_BUFF)
            result = VGroup(arc, label)
            result.rotate(90 * DEGREES, RIGHT, about_point=ORIGIN)
            result.rotate(90 * DEGREES, IN, about_point=ORIGIN)
            return result

        def get_lat_line_radius():
            point = get_sphere_point()
            return Line(point[2] * OUT, point, stroke_color=PINK, stroke_width=2)

        def get_lat_line():
            result = Circle(radius=radius * math.sin(get_phi()))
            result.set_stroke(RED, 1, opacity=0.5)
            result.set_z(get_sphere_point()[2])
            return result

        radial_line = get_radial_line()
        radial_line.add_updater(lambda m: m.match_points(get_radial_line()))
        angle_label = get_angle_label()
        angle_label[0].add_updater(lambda m: m.match_points(get_angle_label()[0]).set_stroke(WHITE, 2).set_fill(opacity=0))
        angle_label[1].add_updater(lambda m: m.move_to(get_angle_label()[1]))
        lat_line_radius = get_lat_line_radius()
        lat_line_radius.add_updater(lambda m: m.match_points(get_lat_line_radius()))
        lat_line_label = Tex("R\\sin(\\phi)", font_size=24)
        lat_line_label.rotate(90 * DEGREES, RIGHT).rotate(90 * DEGREES, IN)
        lat_line_label.next_to(lat_line_radius, OUT, SMALL_BUFF)

        self.play(ShowCreation(radial_line))
        self.play(
            phi_tracker.animate.set_value(30 * DEGREES),
            UpdateFromAlphaFunc(angle_label, lambda m, a: m.update().set_opacity(a)),
        )
        self.wait()
        self.play(
            phi_tracker.animate.set_value(75 * DEGREES),
            run_time=3,
        )
        self.play(phi_tracker.animate.set_value(45 * DEGREES), run_time=2)
        self.wait()

        lat_line_label.add_updater(lambda m: m.next_to(lat_line_radius, OUT, SMALL_BUFF))
        self.play(
            ShowCreation(lat_line_radius),
            Write(lat_line_label)
        )
        self.wait()

        lat_line = get_lat_line()
        lat_line.add_updater(lambda m: m.match_points(get_lat_line()))
        self.play(ShowCreation(lat_line))
        self.wait()

        # Show delta theta
        brace = Brace(Line(ORIGIN, radius * dtheta * RIGHT), UP, buff=SMALL_BUFF)
        delta_theta = Tex("\\Delta \\theta", font_size=36)
        delta_theta.next_to(brace, UP, buff=SMALL_BUFF)
        dt_label = VGroup(brace, delta_theta)
        dt_label.rotate(90 * DEGREES, RIGHT)
        dt_label.next_to(radius * DOWN, OUT, buff=0)
        dt_label.rotate(0.5 * dtheta, OUT, about_point=ORIGIN)

        self.play(Write(dt_label), frame.animate.set_theta(-30 * DEGREES))
        self.wait()

        # Exact formula
        formula1 = Tex("2\\pi", " R \\sin(\\phi)", "\\cdot", "{\\Delta \\theta", " \\over 2\\pi}")
        formula2 = Tex("R \\sin(\\phi)", "\\cdot", "\\Delta \\theta")
        for formula in formula1, formula2:
            formula.to_corner(UR)
            formula.fix_in_frame()

        self.play(Write(formula1))
        self.wait()
        self.play(
            FadeTransform(formula1[1], formula2[0]),
            FadeTransform(formula1[2], formula2[1]),
            FadeTransform(formula1[3], formula2[2]),
            FadeOut(formula1[0]),
            FadeOut(formula1[4:]),
        )
        self.wait()

        # Graph
        axes = Axes(
            (0, PI, PI / 4),
            (0, 1, 1 / 2),
            height=1,
            width=PI,
        )
        axes.y_axis.add(Text("Wedge width", font_size=24, color=RED).next_to(axes.c2p(0, 1), LEFT))
        axes.x_axis.add(Tex("\\pi / 2", font_size=24).next_to(axes.c2p(PI / 2, 0), DOWN))
        axes.x_axis.add(Tex("\\pi", font_size=24).next_to(axes.c2p(PI, 0), DOWN))
        axes.x_axis.add(Tex("\\phi", font_size=24).next_to(axes.c2p(PI, 0), UR, buff=SMALL_BUFF))
        axes.to_corner(UL)
        axes.fix_in_frame()
        graph = axes.get_graph(lambda x: math.sin(x), x_range=[0, PI / 2])
        graph.set_stroke(RED, 2)
        graph.fix_in_frame()
        dot = Dot()
        dot.scale(0.25)
        dot.fix_in_frame()

        self.play(FadeIn(axes), frame.animate.set_theta(-50 * DEGREES))
        self.wait()
        self.play(phi_tracker.animate.set_value(0))
        self.wait()
        graph_copy = graph.copy()
        self.play(
            phi_tracker.animate.set_value(90 * DEGREES),
            ShowCreation(graph),
            UpdateFromAlphaFunc(dot, lambda d, a: d.move_to(graph_copy.pfp(a))),
            run_time=6,
        )
        self.wait()
        self.play(
            phi_tracker.animate.set_value(45 * DEGREES),
            UpdateFromAlphaFunc(dot, lambda d, a: d.move_to(graph.pfp(1 - 0.5 * a))),
            run_time=3,
        )
        self.wait()


class FalseVsTrueSurfaceAreaOverlay(InteractiveScene):
    def construct(self):
        false_answer = VGroup(
            Text("False answer", color=RED),
            Tex("\\pi^2 R^2"),
        )
        true_answer = VGroup(
            Text("True answer", color=GREEN),
            Tex("4\\pi R^2"),
        )
        answers = VGroup(false_answer, true_answer)
        for answer in answers:
            answer.arrange(DOWN, buff=MED_LARGE_BUFF)
        answers.arrange(RIGHT, buff=LARGE_BUFF, aligned_edge=UP)

        self.play(FadeIn(false_answer))
        self.play(FadeIn(true_answer))
        self.wait()


class FakeAreaManipulation(InteractiveScene):
    CONFIG = {
        "unit": 0.5
    }

    def construct(self):
        # Setup
        unit = self.unit
        group1, group2 = groups = self.get_diagrams()
        for group in groups:
            group.set_width(10 * unit, stretch=True)
            group.set_height(12 * unit, stretch=True)
            group.move_to(3 * DOWN, DOWN)
            group[2].append_points(3 * [group[2].get_left() + LEFT])
            group[3].append_points(3 * [group[3].get_right() + RIGHT])

        grid = NumberPlane(
            x_range=(-30, 30),
            y_range=(-30, 30),
            faded_line_ratio=0,
        )
        grid.set_stroke(width=1)
        grid.scale(unit)
        grid.shift(3 * DOWN - grid.c2p(0, 0))

        vertex_dots = VGroup(
            Dot(group1.get_top()),
            Dot(group1.get_corner(DR)),
            Dot(group1.get_corner(DL)),
        )

        self.add(grid)
        self.add(*group1)
        self.add(vertex_dots)

        self.disable_interaction(grid, vertex_dots)
        targets = [group1.copy(), group2.copy()]

        self.wait(note="Manually manipulate")

        # Animate swap
        kw = {
            "lag_ratio": 0.1,
            "run_time": 2,
            "rate_func": bezier([0, 0, 1, 1]),
        }
        path_arc_factors = [-1, 1, 0, 0, -1, 1]
        for target in targets:
            self.play(group1.animate.space_out_submobjects(1.2))
            self.play(*[
                Transform(
                    sm1, sm2,
                    path_arc=path_arc_factors[i] * 60 * DEGREES,
                    **kw
                )
                for i, sm1, sm2 in zip(it.count(), group1, target)
            ])
            self.wait(2)

        # Zoom
        lines = VGroup(
            Line(grid.c2p(0, 12), grid.c2p(-5, 0)),
            Line(grid.c2p(0, 12), grid.c2p(5, 0)),
        )
        lines.set_stroke(YELLOW, 2)
        self.disable_interaction(lines)

        frame = self.camera.frame
        frame.save_state()

        self.play(ShowCreation(lines, lag_ratio=0))
        self.play(
            frame.animate.scale(0.15).move_to(group1[0].get_corner(UR)),
            run_time=4,
        )
        self.wait(3)
        self.play(frame.animate.restore(), run_time=2)

        # Another switch
        self.wait(note="Hold for next swap")
        self.play(*(
            Transform(sm1, sm2, **kw)
            for sm1, sm2 in zip(group1, targets[0])
        ))
        self.wait()

        # Another zooming
        self.play(
            frame.animate.scale(0.15).move_to(group1[4].get_corner(UR)),
            run_time=4,
        )
        self.wait(2)
        self.play(frame.animate.restore(), run_time=2)

        # Show slopes
        tris = VGroup(group1[0], group1[4])
        lil_lines = VGroup(*(Line(tri.get_corner(DL), tri.get_corner(UR)) for tri in tris))
        lil_lines[0].set_stroke(PINK, 3)
        lil_lines[1].set_stroke(WHITE, 3)

        slope_labels = VGroup(
            TexText("Slope =", " $5 / 2$"),
            TexText("Slope =", " $7 / 3$"),
        )
        for line, label in zip(lil_lines, slope_labels):
            label.next_to(line.pfp(0.5), UL, buff=0.7)
            arrow = Arrow(label.get_bottom(), line.pfp(0.5))
            label.add(arrow)

        self.play(
            FadeOut(lines[0]),
            ShowCreation(lil_lines),
        )
        for line, label in zip(lil_lines, slope_labels):
            p1, p2 = line.get_start_and_end()
            corner = [p2[0], p1[1], 0]
            x_line = Line(p1, corner).set_stroke(line.get_color(), 2)
            y_line = Line(corner, p2).set_stroke(line.get_color(), 2)
            self.play(
                FadeIn(label[:2]),
                ShowCreation(label[2]),
            )
            self.play(
                TransformFromCopy(line, y_line),
                FlashAround(label[1][0]),
            )
            self.play(
                TransformFromCopy(line, x_line),
                FlashAround(label[1][2]),
            )
            self.wait()

    def get_diagrams(self):
        unit = self.unit

        tri1 = Polygon(2 * LEFT, ORIGIN, 5 * UP)
        tri2 = tri1.copy()
        tri2.flip()
        tri2.next_to(tri1, RIGHT, buff=0)
        tris = VGroup(tri1, tri2)
        tris.scale(unit)
        tris.move_to(3 * UP, UP)
        tris.set_stroke(width=0)
        tris.set_fill(BLUE_D)
        tris[1].set_color(BLUE_C)

        ell = Polygon(
            ORIGIN,
            4 * RIGHT,
            4 * RIGHT + 2 * UP,
            2 * RIGHT + 2 * UP,
            2 * RIGHT + 5 * UP,
            5 * UP,
        )
        ell.scale(unit)
        ells = VGroup(ell, ell.copy().rotate(PI).shift(2 * unit * UP))
        ells.next_to(tris, DOWN, buff=0)

        ells.set_stroke(width=0)
        ells.set_fill(GREY)
        ells[1].set_fill(GREY_BROWN)

        big_tri = Polygon(ORIGIN, 3 * LEFT, 7 * UP)
        big_tri.set_stroke(width=0)
        big_tri.scale(unit)

        big_tri.move_to(ells.get_corner(DL), DR)
        big_tris = VGroup(big_tri, big_tri.copy().rotate(PI, UP, about_point=ORIGIN))

        big_tris[0].set_fill(RED_E, 1)
        big_tris[1].set_fill(RED_C, 1)
        full_group = VGroup(*tris, *ells, *big_tris)
        full_group.set_height(5, about_edge=UP)

        alt_group = full_group.copy()

        alt_group[0].move_to(alt_group, DL)
        alt_group[1].move_to(alt_group, DR)
        alt_group[4].move_to(alt_group[0].get_corner(UR), DL)
        alt_group[5].move_to(alt_group[1].get_corner(UL), DR)
        alt_group[2].rotate(90 * DEGREES)
        alt_group[2].move_to(alt_group[1].get_corner(DL), DR)
        alt_group[2].rotate(-90 * DEGREES)
        alt_group[2].move_to(alt_group[0].get_corner(DR), DL)
        alt_group[3].move_to(alt_group[1].get_corner(DL), DR)

        full_group.set_opacity(0.75)
        alt_group.set_opacity(0.75)

        return full_group, alt_group


class ContrastSphericalGeometry(InteractiveScene):
    def construct(self):
        # Titles
        titles = VGroup(
            Text("Spherical geometry"),
            Text("Euclidean geometry"),
        )
        for title, v in zip(titles, [LEFT, RIGHT]):
            title.move_to(FRAME_WIDTH * v / 4)
            title.to_edge(UP)

        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        v_line.set_stroke(WHITE, 1)
        self.add(v_line, titles)

        # Setup sphere
        sphere = TexturedSurface(Sphere(radius=1.5), "EarthTextureMap", "NightEarthTextureMap")
        mesh = SurfaceMesh(sphere)
        mesh.set_stroke(GREY_C, 1, opacity=0.5)
        sphere_group = Group(sphere, mesh)
        sphere_group.rotate(80 * DEGREES, LEFT)
        sphere_group.rotate(-3 * DEGREES, OUT)
        sphere_axis = mesh[0].get_end() - sphere.get_center()
        sphere_group.move_to(FRAME_WIDTH * LEFT / 4)
        sphere_group.to_edge(DOWN, buff=0.75)
        self.add(sphere_group)

        # Flat shapes
        shapes = VGroup(
            Polygon(RIGHT, UP, DL).set_fill(BLACK, 1),
            Square().set_fill(BLUE_C, 1),
            Circle().set_fill(BLUE_E, 1),
            RegularPolygon(5).set_fill(GREY_BROWN, 1)
        )
        shapes.set_stroke(WHITE, 2)
        for shape in shapes:
            shape.set_height(1)
        shapes.arrange_in_grid()
        shapes.match_x(titles[1])
        shapes.to_edge(DOWN)

        pre_tri = shapes[0].copy()
        pre_tri.scale(2.5)
        pre_tri.move_to(shapes).align_to(sphere, DOWN)
        pre_tri_arcs, pre_tri_angle_labels = self.get_flat_angle_labels(pre_tri)
        shift_line, new_arcs, new_labels = self.get_aligned_angle_labels(
            pre_tri, pre_tri_arcs, pre_tri_angle_labels
        )

        # Setup spherical triangle
        t2c = {
            "\\alpha": BLUE_B,
            "\\beta": BLUE_C,
            "\\gamma": BLUE_D,
        }
        kw = dict(tex_to_color_map=t2c)
        sph_tri = VGroup(
            VMobject().pointwise_become_partial(mesh[16], 0.5, 1.0),
            VMobject().pointwise_become_partial(mesh[19], 0.5, 1.0).reverse_points(),
            VMobject().pointwise_become_partial(mesh[26], 16 / 20, 19 / 20).reverse_points(),
        )
        sph_tri.set_stroke(BLUE, 3)
        sph_tri_angle_labels = MTex("\\alpha\\beta\\gamma", font_size=30, **kw)
        sph_tri_angle_labels.set_backstroke()
        for label, curve in zip(sph_tri_angle_labels, sph_tri):
            label.curve = curve
            label.add_updater(lambda l: l.move_to(l.curve.get_end()))
            label.add_updater(lambda l: l.shift(0.2 * normalize(sph_tri.get_center() - l.curve.get_end())))

        sph_tri.deactivate_depth_test()

        # Equations
        angle_equations = VGroup(
            MTex("\\alpha + \\beta + \\gamma > 180^\\circ", **kw),
            MTex("\\alpha + \\beta + \\gamma = 180^\\circ", **kw),
        )
        for eq, title in zip(angle_equations, titles):
            eq.next_to(title, DOWN, buff=1.5)

        # Write triangles
        sph_tri_angle_labels.suspend_updating()
        self.play(
            LaggedStartMap(Write, VGroup(
                pre_tri, pre_tri_arcs, pre_tri_angle_labels,
                sph_tri, sph_tri_angle_labels,
            )),
            *(FadeIn(eq, 0.25 * DOWN) for eq in angle_equations)
        )
        sph_tri_angle_labels.resume_updating()
        sphere_group.add(sph_tri, sph_tri_angle_labels)
        sph_tri.deactivate_depth_test()
        self.add(sphere_group)
        self.play()

        # Justify euclidean case
        self.play(
            FadeIn(shift_line, shift=shift_line.shift_vect),
            Rotate(sphere_group, -30 * DEGREES, axis=sphere_axis, run_time=2)
        )
        for i in (0, 1):
            self.play(
                TransformFromCopy(pre_tri_arcs[i + 1], new_arcs[i]),
                TransformFromCopy(pre_tri_angle_labels[i + 1], new_labels[i]),
            )
        self.wait()

        pre_tri_group = VGroup(
            pre_tri, pre_tri_arcs, pre_tri_angle_labels,
            shift_line, new_arcs, new_labels
        )

        # Write area formulas
        area_formulas = VGroup(
            MTex("\\text{Area}(\\Delta) = (\\alpha + \\beta + \\gamma - \\pi) R^2", **kw),
            MTex("\\text{Area}(\\Delta) = \\frac{1}{2} bh")
        )
        for title, formula in zip(titles, area_formulas):
            formula.move_to(title).shift(1.5 * DOWN)

        self.play(
            FadeIn(area_formulas, 0.5 * DOWN),
            angle_equations.animate.shift(0.75 * DOWN)
        )

        # Mention Gaussian Curvature
        curvature_words = VGroup(
            TexText("Gaussian curvature > 0", color=GREEN_B),
            TexText("Gaussian curvature = 0", color=YELLOW),
        )
        for words, eq in zip(curvature_words, angle_equations):
            words.next_to(eq, DOWN, MED_LARGE_BUFF)

        self.play(
            FadeIn(curvature_words, 0.7 * DOWN),
            sphere_group.animate.scale(0.7, about_edge=DOWN),
            pre_tri_group.animate.scale(0.7, about_edge=DOWN),
        )
        self.wait()

        # Try unraveling sphere
        unwrapped = TexturedSurface(Surface(), "EarthTextureMap", "NightEarthTextureMap")
        unwrapped.set_height(2)
        unwrapped.set_width(TAU, stretch=True)
        unwrapped.set_width(FRAME_WIDTH / 2 - 1)
        unwrapped.next_to(titles[1], DOWN, LARGE_BUFF)

        loss_words = Text("Some geometric information\nmust be lost")
        loss_words.set_color(RED)
        loss_words.set_max_width(unwrapped.get_width())
        loss_words.next_to(unwrapped, DOWN, MED_LARGE_BUFF)

        self.play(
            LaggedStart(*(
                FadeOut(mob, RIGHT)
                for mob in [area_formulas[1], angle_equations[1], curvature_words[1], pre_tri_group]
            )),
            TransformFromCopy(sphere, unwrapped, run_time=2)
        )
        self.play(Write(loss_words))
        self.wait(2)

    def get_flat_angle_labels(self, tri):
        arcs = VGroup()
        for v1, v2, v3 in adjacent_n_tuples(tri.get_vertices(), 3):
            a1 = angle_of_vector(v2 - v1)
            a2 = angle_of_vector(v3 - v2)
            arc = Arc(
                start_angle=a2,
                angle=PI - (a2 - a1) % PI,
                radius=0.2
            )
            arc.shift(v2)
            arcs.add(arc)

        labels = MTex("\\alpha \\beta \\gamma", font_size=30)
        for label, arc in zip(labels, arcs):
            vect = normalize(arc.pfp(0.5) - midpoint(arc.get_start(), arc.get_end()))
            label.move_to(arc.pfp(0.5))
            label.shift(0.25 * vect)

        result = VGroup(arcs, labels)
        for group in result:
            group.set_submobject_colors_by_gradient(BLUE_B, BLUE_C, BLUE_D)
        return result

    def get_aligned_angle_labels(self, tri, arcs, labels):
        verts = tri.get_vertices()
        line = Line(verts[0], verts[2])
        line.set_stroke(GREY_B, 2)

        vect1 = verts[1] - verts[2]
        vect2 = verts[1] - verts[0]
        line.shift_vect = (vect1 + vect2) / 2
        line.shift(line.shift_vect)

        new_arcs = arcs[1:3].copy()
        new_labels = labels[1:3].copy()
        VGroup(new_arcs[0], new_labels[0]).rotate(PI, about_point=midpoint(verts[1], verts[2]))
        VGroup(new_arcs[1], new_labels[1]).rotate(PI, about_point=midpoint(verts[1], verts[0]))
        for label in new_labels:
            label.rotate(PI)

        return VGroup(line, new_arcs, new_labels)


class GiveItAGo(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        ss = self.students

        self.play(
            morty.says("Can you see\nwhat's happening?"),
            ss[0].change("pondering", self.screen),
            ss[1].change("angry", self.screen),
            ss[2].change("confused", self.screen),
        )
        self.wait()
        self.play(ss[1].change("maybe"))
        self.play(morty.change("tease"), ss[0].change("thinking"))
        self.wait(4)


class SquareCircleExample(InteractiveScene):
    def construct(self):
        # Setup
        radius = 2.0
        circle = Circle(radius=radius, n_components=32)
        rich_circle = Circle(radius=radius, n_components=2**14)
        circle.set_fill(BLUE_E, 1)
        circle.set_stroke(WHITE, 1)
        approx_curves = [
            self.get_square_approx(rich_circle, 4 * 2**n)
            for n in range(10)
        ]
        square = approx_curves[0].copy()

        self.add(circle)

        # Ask about circumference
        radial_line = Line(ORIGIN, circle.get_right())
        radial_line.set_stroke(WHITE, 1)
        radius_label = Tex("1")
        radius_label.next_to(radial_line, UP, SMALL_BUFF)

        circum = circle.copy()
        circum.set_stroke(YELLOW, 3).set_fill(opacity=0)
        question = Text("What is the circumference?")
        question.next_to(circle, UP, MED_LARGE_BUFF)

        unwrapped_circum = Line(LEFT, RIGHT)
        unwrapped_circum.set_width(PI * circle.get_width())
        unwrapped_circum.match_style(circum)
        unwrapped_circum.next_to(circle, UP)

        diameter = Line(circle.get_left(), circle.get_right())
        diameter.set_stroke(RED, 2)

        self.play(
            ShowCreation(radial_line),
            Write(radius_label, stroke_color=WHITE)
        )
        self.wait()

        self.play(
            Write(question),
            ShowCreation(circum)
        )
        self.wait()
        self.play(
            question.animate.to_edge(UP),
            Transform(circum, unwrapped_circum),
        )
        self.play(ShowCreation(diameter))
        self.wait()
        self.play(*map(FadeOut, [question, circum, diameter]))

        # Show perimeter length
        points = [square.get_edge_center(np.round(vect)) for vect in compass_directions(8)]
        new_radii = VGroup(*(
            Line(p1, p2).match_style(radial_line)
            for p1, p2 in adjacent_pairs(points)
        ))
        new_radii.save_state()
        new_radii.space_out_submobjects(1.1)

        perimeter_label = TexText("Perimeter = $8$")
        perimeter_label.to_edge(UP)

        self.play(ShowCreation(square), run_time=3)
        self.play(square.animate.scale(1.2), rate_func=there_and_back)
        self.wait()
        self.play(
            TransformFromCopy(VGroup(radial_line), new_radii, lag_ratio=0.2, run_time=2),
            FadeIn(perimeter_label),
        )
        self.wait()
        self.play(new_radii.animate.restore())
        self.play(FadeOut(new_radii))
        self.wait()

        # Finer approximations
        for i, curve in enumerate(approx_curves[1:]):
            curve.set_color(YELLOW)
            self.play(
                square.animate.set_stroke(width=1),
                TransformFromCopy(square, curve)
            )
            self.wait()
            if i == 0:
                dots = GlowDot().replicate(2)
                dots.set_color(BLUE)
                sc = square.copy().insert_n_curves(200)
                cc = curve.copy().insert_n_curves(200)
                self.play(VGroup(square, curve).animate.set_stroke(opacity=0.2))
                self.play(
                    MoveAlongPath(dots[0], square),
                    MoveAlongPath(dots[1], curve),
                    ShowCreation(sc),
                    ShowCreation(cc),
                    rate_func=linear,
                    run_time=6,
                )
                self.play(FadeOut(dots), FadeOut(sc), FadeOut(cc), VGroup(square, curve).animate.set_stroke(opacity=1))

            if i == 1:
                curve.set_stroke(width=5)
                print(self.num_plays)
            self.play(FadeOut(square), curve.animate.set_color(RED))
            square = curve

        # Zoom in
        frame = self.camera.frame
        self.wait(note="Prepare for zoom")
        self.play(frame.animate.set_height(0.05).move_to(circle.pfp(1 / 8)), run_time=4)
        self.wait()
        self.play(frame.animate.to_default_state(), run_time=3)

        # Define parametric curve
        frame = self.camera.frame
        t_tracker = ValueTracker(0)
        get_t = t_tracker.get_value
        dot = GlowDot()
        t_axis = UnitInterval()
        t_axis.set_width(6)
        t_axis.next_to(circle, RIGHT, buff=1.5)
        t_axis.add_numbers()
        t_indicator = Triangle(start_angle=-90 * DEGREES)
        t_indicator.set_height(0.1)
        t_indicator.set_fill(RED, 1)
        t_indicator.set_stroke(WHITE, 0)
        t_label = VGroup(Tex("t = "), DecimalNumber())
        t_label.arrange(RIGHT)
        t_label.next_to(t_axis, UP, buff=LARGE_BUFF)
        VGroup(t_axis, t_label).to_edge(UP)

        globals().update(locals())
        t_label[1].add_updater(lambda d: d.set_value(get_t()))
        dot.add_updater(lambda d: d.move_to(square.pfp(get_t())))
        t_indicator.add_updater(lambda m: m.move_to(t_axis.n2p(get_t()), DOWN))

        c_labels = VGroup(*(Tex(f"c_{n}(t)") for n in range(len(approx_curves))))
        c_labels.add(Tex("c_\\infty (t)"))
        for label in c_labels:
            label.scale(0.75)
            label.add_updater(lambda m: m.next_to(dot, UR, buff=-SMALL_BUFF))

        self.play(
            Transform(square, approx_curves[0]),
            frame.animate.move_to(4 * RIGHT),
            FadeIn(dot),
            FadeIn(t_label),
            Write(c_labels[0]),
            Write(t_axis),
            Write(t_indicator),
            run_time=1
        )
        square.match_points(approx_curves[0])
        self.wait()
        self.play(t_tracker.animate.set_value(1), run_time=7)
        self.wait()
        t_tracker.set_value(0)

        self.play(
            Transform(square, approx_curves[1]),
            FadeTransform(c_labels[0], c_labels[1]),
        )
        self.play(t_tracker.animate.set_value(1), run_time=7)
        self.wait()
        t_tracker.set_value(0)

        # Show limits
        self.play(t_tracker.animate.set_value(0.2), run_time=3)
        dot_shadows = VGroup()
        self.add(dot_shadows)
        for i in range(2, len(approx_curves)):
            dot_shadow = Dot(radius=0.01, color=YELLOW, opacity=0.5)
            dot_shadow.move_to(dot)
            dot_shadows.add(dot_shadow)
            self.play(
                Transform(square, approx_curves[i]),
                FadeTransform(c_labels[i - 1], c_labels[i]),
                run_time=0.5,
            )
            self.wait(0.5)

        # Write limits
        lim_tex_ex = MTex("\\lim_{n \\to \\infty} c_{n}(" + "{:.1f}".format(get_t()) + ")")
        lim_tex = Tex("c_\\infty(t)", ":=", "\\lim_{n \\to \\infty} c_{n}(t)")
        for lt in lim_tex_ex, lim_tex:
            lt.next_to(t_axis, DOWN, aligned_edge=LEFT, buff=2.0)
        lim_arrow = Arrow(lim_tex_ex.get_corner(UL), dot.get_center(), buff=0.1, stroke_width=2, color=YELLOW)
        self.play(Write(lim_tex_ex))
        self.play(ShowCreation(lim_arrow))
        self.wait()
        self.play(
            FadeTransform(lim_tex_ex, lim_tex[2]),
            Write(lim_tex[:2]),
        )
        self.wait()
        self.play(FadeOut(dot_shadows), FadeOut(lim_arrow), FadeTransform(c_labels[9], c_labels[-1]))
        self.play(t_tracker.animate.set_value(0), run_time=2)
        self.play(t_tracker.animate.set_value(1), run_time=8)
        self.wait()

        # This is a circle
        text = Text("This is, precisely, a circle", t2s={"precisely": ITALIC})
        text.next_to(lim_tex, DOWN, LARGE_BUFF, aligned_edge=LEFT)
        arrow = Arrow(text, lim_tex[0])
        VGroup(text, arrow).set_color(GREEN)

        self.play(Write(text), ShowCreation(arrow))
        self.wait()
        self.play(FadeOut(text), FadeOut(arrow), lim_tex.animate.shift(UP))

        # Mismatched limits
        t2c = {
            "\\lim_{n \\to \\infty}": YELLOW,
            "\\text{len}": RED,
            "c_n(t)": WHITE,
            "\\Big(": WHITE,
            "\\Big)": WHITE,
        }
        lim_len = MTex("\\lim_{n \\to \\infty}\\Big(\\text{len}\\big(c_n(t)\\big) \\Big) = 8", tex_to_color_map=t2c)
        len_lim = MTex("\\text{len} \\Big( \\lim_{n \\to \\infty} c_n(t) \\Big) = 2\\pi", tex_to_color_map=t2c)
        lims = VGroup(lim_len, len_lim)
        lim_len.next_to(lim_tex, DOWN, LARGE_BUFF, aligned_edge=LEFT)

        top_group = VGroup(t_axis, t_indicator, t_label)

        self.play(Write(lim_len))
        self.wait()
        self.play(
            VGroup(lim_tex, lim_len).animate.next_to(frame.get_corner(UR), DL, MED_LARGE_BUFF),
            FadeOut(top_group, 2 * UR),
        )

        len_lim.next_to(lim_len, DOWN, buff=2.0, aligned_edge=LEFT)
        not_eq = Tex("\\ne", font_size=96)
        not_eq.rotate(90 * DEGREES)
        not_eq.move_to(VGroup(len_lim, lim_len))
        not_eq.match_x(len_lim)

        self.play(*(
            TransformFromCopy(lim_len.select_parts(tex), len_lim.select_parts(tex))
            for tex in t2c.keys()
        ))
        self.play(Write(not_eq))
        self.wait()
        self.play(Write(len_lim[3:]))
        self.wait()

        # Commentary
        morty = Mortimer()

    def get_square_approx(self, circle, n_samples):
        radius = circle.radius
        points = [
            radius * np.array([math.cos(a), math.sin(a), 0])
            for a in np.linspace(0, TAU, n_samples + 1)
        ]
        result = VMobject()
        result.start_new_path(points[0])
        for p1, p2 in zip(points, points[1:]):
            corners = np.array([
                [p2[0], p1[1], 0],
                [p1[0], p2[1], 0]
            ])
            corner = corners[np.argmax(np.apply_along_axis(np.linalg.norm, 1, corners))]
            result.add_line_to(corner)
            result.add_line_to(p2)

        result.set_stroke(RED, 2)
        return result


class ObviouslyWrong(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        ss = self.students
        self.play(LaggedStart(
            ss[2].says("Obviously those\nperimeters aren't\nthe circle...", mode="sassy", look_at=self.screen, bubble_direction=LEFT),
            ss[0].change("erm", self.screen),
            ss[1].change("angry", self.screen),
            morty.change("guilty")
        ))
        self.wait(5)


class UpshotOfLimitExample(InteractiveScene):
    def construct(self):
        # Title
        title = Text("The takeaway", font_size=60)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        underline = Underline(title, buff=-0.05)
        underline.scale(1.2)
        underline.insert_n_curves(20)
        underline.set_stroke(WHITE, (1, 3, 3, 1))

        subtitle = Text(
            "What's true of a sequence may not be true of the limit",
            t2c={"sequence": BLUE, "limit": YELLOW},
        )
        subtitle.set_max_width(FRAME_WIDTH - 1)
        subtitle.next_to(title, DOWN, buff=MED_LARGE_BUFF)

        self.add(title, underline, subtitle)

        # Various limits
        n = 4
        numbers = VGroup(
            Tex("1.4"),
            Tex("1.41"),
            Tex("1.414"),
            Tex("1.4142"),
            Tex("\\sqrt{2}")
        )
        numbers.scale(1.25)
        properties = VGroup(
            *Text("Rational", font_size=30).replicate(n),
            Text("Irrational", font_size=30)
        )

        folder = "/Users/grant/Dropbox/3Blue1Brown/videos/2022/visual_proofs/lies/images/"
        rational_example = VGroup(numbers, properties)
        circle_example = Group(
            Group(*(
                ImageMobject(os.path.join(folder, "SquareApprox" + end)).set_width(1.25)
                for end in ["1", "2", "3", "4", "Inf"]
            )),
            VGroup(
                *TexText("Len = 8").replicate(n),
                TexText("Len = $2\\pi$")
            )
        )
        fourier_example = Group(
            Group(*(
                ImageMobject(os.path.join(folder, "Fourier" + end)).set_width(1.25)
                for end in ["1", "2", "3", "4", "Inf"]
            )),
            VGroup(
                *TexText("Continuous").replicate(n),
                TexText("Discontinuous")
            )
        )
        examples = Group(rational_example, circle_example, fourier_example)

        for objs, descs in examples:
            arrow = Tex("\\cdots \\, \\rightarrow", font_size=72)
            group = Group(*objs[:n], arrow, objs[n])
            for x, mob in enumerate(group):
                mob.move_to(2.75 * x * RIGHT)
            group.set_x(0)
            for desc, obj in zip(descs, objs):
                desc.set_max_width(1.25 * obj.get_width())
                desc.next_to(obj, DOWN)
            descs[:n].set_color(BLUE)
            descs[n].set_color(YELLOW)
            objs.insert_submobject(n, arrow)

        examples.arrange(DOWN, buff=LARGE_BUFF)
        examples.set_height(5.0)
        examples.to_edge(DOWN)

        for i in range(3):
            objs, descs = examples[i]
            self.play(
                LaggedStartMap(FadeIn, objs, lag_ratio=0.5, scale=2),
                LaggedStartMap(FadeIn, descs, lag_ratio=0.5, scale=2),
                run_time=2
            )


class WhyWeNeedProofs(TeacherStudentsScene):
    def construct(self):
        phrases = VGroup(
            Text("Looks can be deceiving"),
            Text("We need rigor!"),
            Text("We need proofs!"),
        )
        phrases.move_to(self.hold_up_spot, DOWN)
        for phrase in phrases:
            phrase.shift_onto_screen()
            phrase.align_to(phrases[0], LEFT)

        image = ImageMobject("Euclid").set_height(3)
        rect = SurroundingRectangle(image, buff=0)
        rect.set_stroke(WHITE, 2)
        name = Text("Euclid")
        name.next_to(image, DOWN, )
        euclid = Group(image, rect, name)
        euclid.to_corner(UL)

        morty = self.teacher
        ss = self.students

        # Show phrases
        self.play(
            morty.change("raise_right_hand"),
            FadeIn(phrases[0], UP),
            ss[0].change("happy"),
            ss[1].change("tease"),
            ss[2].change("happy"),
        )
        self.wait()
        self.play(
            morty.change("hooray"),
            phrases[0].animate.shift(UP),
            FadeIn(phrases[1], UP),
        )
        self.play(
            phrases[0].animate.shift(UP),
            phrases[1].animate.shift(UP),
            FadeIn(phrases[2], UP),
        )
        self.wait()
        self.play(
            morty.change("tease"),
            FadeIn(euclid, lag_ratio=0.3),
            self.change_students("pondering", "pondering", "thinking", look_at=euclid)
        )
        self.wait(4)


class Proof3Slide(InteractiveScene):
    def construct(self):
        self.add(FullScreenRectangle())
        title = TexText("``Proof'' \\#3", font_size=60)
        subtitle = Text("All triangles are isosceles", font_size=60, color=BLUE)
        title.to_edge(UP)
        subtitle.next_to(title, DOWN)
        self.add(title, subtitle)

        # Triangle
        tri = Triangle()
        tri.set_fill(BLUE_E)
        tri.set_stroke(WHITE, 1)
        tri.stretch(2, 1)
        tri.to_corner(DL, buff=LARGE_BUFF).shift(RIGHT)
        A, B, C = tri.get_top(), tri.get_corner(DL), tri.get_corner(DR)
        AB = Line(A, B, color=YELLOW)
        AC = Line(A, C, color=TEAL)
        eq = Tex("\\overline{AB}", " = ", "\\overline{AC}")
        eq.next_to(tri, UP)
        eq[0].match_color(AB)
        eq[2].match_color(AC)

        self.add(tri)
        self.play(ShowCreation(AB), Write(eq[0]))
        self.play(
            TransformFromCopy(AB, AC),
            FadeTransform(eq[0].copy(), eq[2]),
            Write(eq[1]),
        )
        self.wait()


class SpeakingOfLimits(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.says("Speaking of\nlimits..."),
            self.change_students("happy", "tease", "thinking"),
        )
        self.wait(4)


class IntegralExample(InteractiveScene):
    def construct(self):
        # Setup
        axes = Axes((-1, 4), (-1, 10), width=12, height=6)
        graph = axes.get_graph(lambda x: x**2)
        graph.set_stroke(TEAL, 2)
        all_rects = VGroup(*(
            axes.get_riemann_rectangles(graph, (0, 3), dx).set_stroke(BLACK, np.round(4 * dx, 1), background=False)
            for dx in [2**(-n) for n in range(2, 8)]
        ))
        rects = all_rects[0]
        last_rects = all_rects[-1].copy()
        last_rects.set_stroke(width=0)

        self.add(axes)
        self.play(ShowCreation(graph))
        self.play(FadeIn(last_rects, lag_ratio=0.1))
        self.wait()
        self.play(FadeIn(rects, lag_ratio=0.1), FadeOut(last_rects))
        self.wait()
        self.play(LaggedStart(*(
            ApplyMethod(rect.set_color, YELLOW, rate_func=there_and_back)
            for rect in rects
        ), lag_ratio=0.5, run_time=4))

        # Iterations
        for new_rects in all_rects[1:]:
            self.play(Transform(rects, new_rects))
            self.wait(0.5)

        # Zoom
        frame = self.camera.frame

        self.play(frame.animate.move_to(graph.pfp(0.5)).set_height(0.1), run_time=3)
        self.wait()
        self.play(frame.animate.to_default_state(), run_time=3)
        self.wait()


class IntegralError(InteractiveScene):
    def construct(self):
        # (Coped from above)
        axes = Axes((-1, 4), (-1, 10), width=12, height=6)
        graph = axes.get_graph(lambda x: x**2)
        graph.set_stroke(TEAL, 2)
        all_rects = VGroup(*(
            axes.get_riemann_rectangles(graph, (0, 3), dx).set_stroke(BLACK, np.round(4 * dx, 1), background=False)
            for dx in [2**(-n) for n in range(2, 9)]
        ))
        rects = all_rects[0]
        last_rects = all_rects[-1].copy()
        last_rects.set_stroke(width=0)

        self.add(axes, graph)
        self.add(rects)

        # Mention error
        error = last_rects.copy()
        error.set_fill(RED, 1)

        error_label = Text("Error", font_size=72, color=RED)
        error_label.move_to(rects, UP).shift(UP)
        error_arrows = VGroup(*(
            Arrow(error_label, rect.get_top() + 0.2 * UP, stroke_width=2, tip_width_ratio=8)
            for rect in rects
        ))
        error_arrows.set_color(RED)

        self.add(error, rects)
        self.play(
            FadeIn(error, lag_ratio=0.5, run_time=2),
            Write(error_label, stroke_color=RED),
            LaggedStart(*(
                ShowCreation(arrow)
                for arrow in error_arrows
            ), lag_ratio=0.2, run_time=3)
        )
        self.wait()

        # Error boxes
        all_error_boxes = VGroup()
        for rg in all_rects:
            boxes = VGroup()
            for rect in rg:
                box = Rectangle(stroke_width=0, fill_opacity=0.5, fill_color=RED)
                box.replace(Line(
                    axes.i2gp(axes.x_axis.p2n(rect.get_left()), graph),
                    axes.i2gp(axes.x_axis.p2n(rect.get_right()), graph),
                ), stretch=True)
                boxes.add(box)
            all_error_boxes.add(boxes)
        self.play(
            FadeIn(all_error_boxes[0], lag_ratio=0.1),
            error.animate.set_opacity(0.25),
            FadeOut(error_arrows),
        )
        for rg, ebg in zip(all_rects, all_error_boxes):
            for rect, box in zip(rg, ebg):
                rect.add(box)
        self.remove(all_error_boxes)
        self.add(rects)

        # Inequality
        error_label.generate_target()
        error_label.target.scale(0.7)
        sum_squares = MTex("< \\sum \\Delta", isolate="\\Delta")
        rhs = VGroup(Tex("="), DecimalNumber(num_decimal_places=4))
        rhs.arrange(RIGHT)
        rhs[1].set_value(0.3417)

        group = VGroup(error_label.target, sum_squares, rhs)
        group.arrange(RIGHT)
        group.to_edge(UP)

        pre_box_sym = rects[-1].submobjects[0]
        box_sym = pre_box_sym.copy()
        box_sym.replace(sum_squares[2], dim_to_match=1)
        sum_squares.replace_submobject(2, box_sym)

        self.play(
            MoveToTarget(error_label),
            Write(sum_squares[:2]),
            TransformFromCopy(pre_box_sym, box_sym)
        )
        self.add(rhs)
        self.play(
            CountInFrom(rhs[1], 0),
            LaggedStart(*(
                VFadeInThenOut(r.copy().set_fill(opacity=1), buff=0, color=RED)
                for r in all_error_boxes[0]
            ), lag_ratio=0.2),
            run_time=2
        )
        self.wait()

        # Iterations
        for new_rects in all_rects[1:]:
            self.play(
                Transform(rects, new_rects),
                ChangeDecimalToValue(rhs[1], rhs[1].get_value() / 4),
                run_time=3,
            )
            self.wait(0.5)


class IntegralExampleWithErrorBoxes(IntegralExample):
    show_error_boxes = True


class DefiningTheLengthOfACurve(InteractiveScene):
    def construct(self):
        pass


class FalseEuclidProofAnnotation(InteractiveScene):
    def construct(self):
        # path = "/Users/grant/Dropbox/3Blue1Brown/videos/2022/visual_proofs/lies/images/FalseEuclidProof.jpg"
        # self.add(ImageMobject(path).set_width(FRAME_WIDTH))

        # Points
        A = np.array([-1.94444444, 1.44444444, 0.])
        B = np.array([-4.44444444, -0.02777778, 0.])
        C = np.array([-1.09722222, -0.48611111, 0.])
        D = np.array([-2.63888889, -0.27777778, 0.])
        E = np.array([-1.56944444, 0.55555556, 0.])
        F = np.array([-3.01388889, 0.83555556, 0.])
        P = np.array([-2.58333333, 0.122222, 0.])
        # dots = Group(*(GlowDot(point, color=RED) for point in [A, B, C, D, E, F, P]))

        AFP = Polygon(A, F, P)
        AEP = Polygon(A, E, P)
        BPD = Polygon(B, P, D)
        CPD = Polygon(C, P, D)
        BFP = Polygon(B, F, P)
        CEP = Polygon(C, E, P)

        tris = VGroup(AFP, AEP, BPD, CPD, BFP, CEP)
        tris.set_stroke(BLACK, 1)
        tris[:2].set_fill(BLUE)
        tris[2:4].set_fill(GREEN)
        tris[4:].set_fill(RED)
        tris.set_fill(opacity=0.8)

        # Final sum
        AF = Line(A, F)
        FB = Line(F, B)
        AB = Line(A, B)
        AE = Line(A, E)
        EC = Line(E, C)
        AC = Line(A, C)
        lines = VGroup(AF, FB, AB, AE, EC, AC)
        for line in lines:
            brace = Brace(Line(ORIGIN, line.get_length() * RIGHT), UP)
            brace.next_to(ORIGIN, UP, buff=0.1)
            angle = line.get_angle()
            angle = (angle + PI / 2) % PI - PI / 2
            brace.rotate(angle, about_point=ORIGIN)
            brace.shift(line.get_center())
            brace.set_fill(BLACK, 1)
            line.brace = brace

        self.play(GrowFromCenter(AF.brace), run_time=1)
        self.play(GrowFromCenter(FB.brace), run_time=1)
        self.wait()
        self.play(
            Transform(AF.brace, AB.brace, path_arc=45 * DEGREES),
            Transform(FB.brace, AB.brace, path_arc=45 * DEGREES),
        )
        self.wait()
        self.play(GrowFromCenter(AE.brace), run_time=1)
        self.play(GrowFromCenter(EC.brace), run_time=1)
        self.wait()
        self.play(
            Transform(AE.brace, AC.brace, path_arc=45 * DEGREES),
            Transform(EC.brace, AC.brace, path_arc=45 * DEGREES),
        )
        self.wait()
        return

        # Lines for final triangles
        BP = Line(B, P)
        CP = Line(C, P)
        PF = Line(P, F)
        PE = Line(P, E)
        BF = Line(B, F)
        CE = Line(C, E)

        VGroup(BP, CP).set_stroke(BLUE_E, 5)
        VGroup(PF, PE).set_stroke(TEAL, 5)
        VGroup(BF, CE).set_stroke(RED, 5)

        self.play(*map(ShowCreation, [BP, CP]))
        self.play(*map(ShowCreation, [PF, PE]))
        self.wait()
        self.play(
            TransformFromCopy(PF, BF, path_arc=90 * DEGREES),
            TransformFromCopy(PE, CE, path_arc=-90 * DEGREES),
        )
        self.wait()

        # Compare AB to BC
        AB = Line(A, B).set_stroke(RED, 3)
        AC = Line(A, C).set_stroke(BLUE, 3)

        self.play(ShowCreation(AB))
        self.play(ShowCreation(AC))
        self.wait()
        self.add(AB.copy(), AC.copy())
        self.play(
            AB.animate.set_angle(-90 * DEGREES).next_to(A, RIGHT, aligned_edge=UP, buff=2),
            AC.animate.set_angle(-90 * DEGREES).next_to(A, RIGHT, aligned_edge=UP, buff=2.5),
        )
        self.wait()
        self.play(FadeOut(AB), FadeOut(AC))

        # Bisector labels
        perp = Text("Perpendicular\nbisector", font_size=30, color=BLACK, stroke_width=0)
        perp.next_to(F, UL, buff=0.5)
        perp.set_color(BLUE_E)
        perp_arrow = Arrow(perp, midpoint(D, P), buff=0.1, stroke_width=2)
        perp_arrow.match_color(perp)

        ang_b = Text("Angle\nbisector", font_size=30, color=BLACK, stroke_width=0)
        ang_b.next_to(F, UL, buff=0.5)
        ang_b.set_color(RED_E)
        ang_b_arrow = Arrow(ang_b, midpoint(A, P), buff=0.1, stroke_width=2)
        ang_b_arrow.match_color(ang_b)

        self.play(Write(perp), ShowCreation(perp_arrow), run_time=1)
        self.wait()
        self.play(FadeOut(perp), FadeOut(perp_arrow))
        self.wait()
        self.play(Write(ang_b), ShowCreation(ang_b_arrow), run_time=1)
        self.wait()
        self.play(FadeOut(ang_b), FadeOut(ang_b_arrow))
        self.wait()

        # Similar triangles
        for pair in [(AFP, AEP), (BPD, CPD), (BFP, CEP)]:
            self.play(DrawBorderThenFill(pair[0]))
            self.play(TransformFromCopy(pair[0], pair[1]))
            self.wait()
            self.play(LaggedStartMap(FadeOut, VGroup(*pair)))
            self.wait()


class FalseEuclidFollowup(InteractiveScene):
    def construct(self):
        # All triangles are equilateral
        tri1 = Polygon(3 * UP, DL, RIGHT)
        tri2 = Polygon(3 * UP, LEFT, RIGHT)
        tri3 = tri2.copy().set_height(math.sqrt(3) * tri2.get_width() / 2, stretch=True, about_edge=DOWN)
        tris = VGroup(tri1, tri2, tri3)
        tris.set_fill(BLUE_D, 1)
        tris.set_stroke(WHITE, 2)
        tris.scale(1.5)
        tris.to_edge(DOWN, buff=1.0)
        tri = tri1

        def get_side(i, j, tri=tri):
            return Line(tri.get_vertices()[i], tri.get_vertices()[j]).set_stroke(YELLOW, 4)

        labels = Text("ABC")
        for letter, vert in zip(labels, tri.get_vertices()):
            letter.next_to(vert, normalize(vert - tri.get_center_of_mass()), SMALL_BUFF)

        equation = MTex("\\overline{AB} = \\overline{AC}")
        equation2 = MTex("\\overline{AB} = \\overline{AC} = \\overline{BC}")
        equation.next_to(labels, LEFT, aligned_edge=UP)
        equation2.move_to(equation).to_edge(LEFT)
        words = Text("All triangles are\nisosceles")
        words.next_to(equation, DOWN, LARGE_BUFF)
        VGroup(equation, words).to_edge(LEFT)
        iso = words.select_parts("isosceles")
        iso_cross = Cross(iso)
        equi = Text("equilateral")
        equi.move_to(iso, UP)
        equi.set_color(YELLOW)

        self.add(tri, labels)
        self.wait()
        self.play(
            FadeIn(equation), FadeIn(words),
            ShowCreationThenDestruction(get_side(0, 1), run_time=2),
            ShowCreationThenDestruction(get_side(0, 2), run_time=2),
        )
        self.play(Transform(tri, tri2), labels[1].animate.shift(1.5 * UP))
        self.wait()
        self.play(CyclicReplace(*labels))
        self.play(
            ShowCreationThenDestruction(get_side(1, 0), run_time=2),
            ShowCreationThenDestruction(get_side(1, 2), run_time=2),
            ShowCreation(iso_cross)
        )
        cross_group = VGroup(iso, iso_cross)
        self.play(
            cross_group.animate.shift(DOWN),
            FadeIn(equi, 0.5 * DOWN),
            Transform(tri, tri3),
            labels[2].animate.next_to(tri3, UP, SMALL_BUFF)
        )
        self.play(
            Transform(equation, equation2[:len(equation)]),
            FadeIn(equation2[len(equation):], RIGHT),
        )
        self.wait()

        words = VGroup(words.select_parts("All triangles are"), equi)

        # Three possibilities
        possibilities = VGroup(
            Text("1. This is true"),
            TexText("2. Euclid's axioms $\\Rightarrow$ falsehoods")[0],
            Text("3. This proof has a flaw"),
        )
        possibilities.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        possibilities.to_edge(RIGHT, buff=LARGE_BUFF)
        poss_title = Text("Possibilities", font_size=60)
        poss_title.next_to(possibilities, UP, buff=1.5, aligned_edge=LEFT)
        poss_title.add(Underline(poss_title))

        tri.add(labels)
        self.play(
            FadeOut(cross_group, DL),
            tri.animate.match_width(words).next_to(words, DOWN, LARGE_BUFF),
            LaggedStart(*(
                FadeIn(poss[:2], shift=LEFT)
                for poss in possibilities
            )),
            Write(poss_title)
        )
        self.wait()
        for poss in possibilities:
            self.play(Write(poss[2:], stroke_color=WHITE))
            self.wait()


class TryToFindFault(TeacherStudentsScene):
    def construct(self):
        # Setup
        morty = self.teacher
        ss = self.students

        points = compass_directions(5, start_vect=UP)
        star = Polygon(*(points[i] for i in [0, 2, 4, 1, 3]))
        star.set_fill(YELLOW, 1)
        star.set_stroke(width=0)
        star.set_gloss(1)
        star.set_height(0.5)
        stars = star.replicate(3)
        stars.arrange(RIGHT)
        stars.move_to(self.hold_up_spot, DOWN)
        stars.insert_n_curves(10)
        stars.refresh_triangulation()

        self.play(
            morty.says("I dare you to\nfind a fault", mode="tease"),
        )
        self.play_student_changes("pondering", "pondering", "sassy")
        self.wait()
        self.play(
            morty.debubble(mode="raise_right_hand"),
            self.change_students("thinking", "happy", "tease"),
            Write(stars)
        )
        self.wait(3)


class SideSumTruthiness(InteractiveScene):
    def construct(self):
        # Lines
        A, E, C = Dot().replicate(3)
        A.move_to(2 * UL)
        C.move_to(DR)
        VGroup(A, C).to_corner(DR, buff=1.5)

        alpha_tracker = ValueTracker(0.75)
        E.add_updater(lambda m: m.move_to(interpolate(
            A.get_center(), C.get_center(), alpha_tracker.get_value()
        )))

        AE = Line().set_stroke(RED, 3, opacity=0.75)
        EC = Line().set_stroke(BLUE, 3, opacity=0.75)
        AE.add_updater(lambda l: l.put_start_and_end_on(A.get_center(), E.get_center()))
        EC.add_updater(lambda l: l.put_start_and_end_on(E.get_center(), C.get_center()))

        labels = Text("AEC")
        labels[0].next_to(A, UP, SMALL_BUFF)
        labels[1].add_updater(lambda m: m.next_to(E, DL, SMALL_BUFF))
        labels[2].next_to(C, RIGHT, SMALL_BUFF)

        self.add(AE, EC, A, E, C, *labels)

        # Equation
        eq = Tex("\\overline{AE} + \\overline{EC} = \\overline{AC}", font_size=60)
        eq.to_corner(UR)

        labels = VGroup(
            VGroup(Text("True"), Checkmark()),
            VGroup(Text("False"), Exmark()),
        )
        labels.scale(60 / 48)
        for label in labels:
            label[0].match_color(label[1])
            label.arrange(RIGHT, buff=MED_LARGE_BUFF)
            label.next_to(eq, DOWN, MED_LARGE_BUFF)

        labels[0].add_updater(lambda m: m.set_opacity(1 if 0 < alpha_tracker.get_value() < 1 else 0))
        labels[1].add_updater(lambda m: m.set_opacity(0 if 0 < alpha_tracker.get_value() < 1 else 1))

        self.add(eq)
        self.add(labels)

        # Move around
        for alpha in [0.3, 0.7, 0.5, 0.7, 0.3, -0.3, 0.7, 1.4, 1.2, 0.5]:
            self.play(alpha_tracker.animate.set_value(alpha), run_time=2)
        self.wait()


class PythagoreanProofSketch(InteractiveScene):
    def construct(self):
        # First orientation
        tri = Polygon(ORIGIN, UP, 2 * RIGHT)
        tri.set_stroke(WHITE, 1)
        tri.set_fill(BLUE_E, 1)

        verts = tri.get_vertices()
        tris = VGroup(tri, tri.copy().rotate(PI, about_point=midpoint(*verts[1:])))
        tris.add(*tris.copy().rotate(PI, axis=UL, about_point=tris.get_corner(DL)))
        tris[2:4].flip(UP)
        big_square = SurroundingRectangle(tris, buff=0)
        big_square.set_stroke(WHITE, 2)
        VGroup(tris, big_square).center().set_height(5)

        A, B = tri.get_height(), tri.get_width()
        a_square = Square(A)
        a_square.move_to(big_square, UL)
        a_square.set_fill(RED_C, 0.75)
        b_square = Square(B)
        b_square.move_to(big_square, DR)
        b_square.set_fill(RED_D, 0.75)

        a_square_label = MTex("a^2").move_to(a_square)
        b_square_label = MTex("b^2").move_to(b_square)

        # Pre triangle
        pre_tri = tris[0].copy()
        pre_tri.scale(1.5)
        pre_tri.shift(-pre_tri.get_center_of_mass())
        side_labels = MTex("abc")
        side_labels[0].next_to(pre_tri, LEFT)
        side_labels[1].next_to(pre_tri, DOWN)
        side_labels[2].next_to(pre_tri.get_center(), UR)

        self.add(pre_tri)
        self.add(side_labels)

        # Equation
        equation = MTex("a^2 + b^2 = c^2", isolate=["a^2", "+", "b^2", "=", "c^2"])
        equation.next_to(big_square, UP).to_edge(UP)

        # A^2 and B^2
        scale_factor = tris[0].get_height() / pre_tri.get_height()
        shift_vect = tris[0].get_center() - pre_tri.get_center()
        self.play(
            ReplacementTransform(pre_tri, tris[0]),
            side_labels.animate.scale(scale_factor, about_point=pre_tri.get_center()).shift(shift_vect)
        )
        self.add(tris[1], side_labels)
        self.play(LaggedStart(
            TransformFromCopy(tris[0], tris[1], path_arc=PI),
            TransformFromCopy(tris[0], tris[2]),
            TransformFromCopy(tris[0], tris[3]),
            FadeIn(big_square),
            Animation(side_labels),
            lag_ratio=0.3
        ))
        self.play(
            FadeIn(a_square),
            FadeTransform(side_labels[0], a_square_label),
            FadeIn(b_square),
            FadeTransform(side_labels[1], b_square_label),
        )
        self.play(
            FadeOut(a_square),
            FadeOut(b_square),
            TransformFromCopy(a_square_label, equation.select_parts("a^2")),
            TransformFromCopy(b_square_label, equation.select_parts("b^2")),
            Write(equation.select_parts("+")),
        )
        self.wait()

        # Second orientation
        tris.save_state()
        tris2 = tris.copy()

        tris2[0].rotate(PI / 2, OUT, about_point=tris2[0].get_corner(DR))
        tris2[3].rotate(-PI / 2).move_to(big_square, DL)
        tris2[2].move_to(big_square, UL)

        c_square = Square(math.sqrt(A**2 + B**2))
        c_square.rotate(math.atan(B / A))
        c_square.move_to(big_square)
        c_square.set_fill(RED_D, 0.7)
        c_square_label = MTex("c^2").move_to(c_square)

        self.play(LaggedStart(
            FadeOut(a_square_label),
            FadeOut(b_square_label),
            *(Transform(tris[i], tris2[i]) for i in (0, 2, 3)),
            lag_ratio=0.5
        ))
        self.wait()

        self.play(
            FadeIn(c_square),
            FadeIn(c_square_label),
        )
        self.play(
            FadeTransform(c_square_label.copy(), equation.select_parts("c^2")),
            Write(equation.select_parts("=")),
            c_square.animate.set_fill(opacity=0.5)
        )
        self.wait()
        self.play(FadeOut(c_square))

        # Final gif
        self.play(
            Transform(tris, tris.saved_state, lag_ratio=0.2, run_time=3, path_arc=45 * DEGREES),
            FadeOut(c_square_label),
            FadeIn(a_square_label, time_span=(1.5, 2.5)),
            FadeIn(b_square_label, time_span=(2, 3)),
        )
        self.wait()
        self.play(
            Transform(tris, tris2, lag_ratio=0.2, run_time=3, path_arc=45 * DEGREES),
            FadeIn(c_square_label, time_span=(2, 3)),
            FadeOut(a_square_label, time_span=(0, 1)),
            FadeOut(b_square_label, time_span=(0.5, 1.5)),
        )
        self.wait()


class LastSideBySide(InteractiveScene):
    def construct(self):
        self.add(FullScreenRectangle())
        squares = Square().replicate(2)
        squares.set_stroke(WHITE, 2)
        squares.set_fill(BLACK, 2)
        squares.set_height(5)
        squares.set_width(6, stretch=True)
        for square, vect in zip(squares, [LEFT, RIGHT]):
            square.move_to(FRAME_WIDTH * vect / 4)
        squares.to_edge(DOWN, buff=0.7)
        self.add(squares)

        titles = VGroup(
            TexText("Given examples\\\\like this..."),
            TexText("What's needed to\\\\make this rigorous?"),
        )
        for title, square in zip(titles, squares):
            title.next_to(square, UP, buff=MED_LARGE_BUFF)

        self.play(Write(titles[0]))
        self.wait()
        self.play(Write(titles[1]))
        self.wait()


class ByTheWay(InteractiveScene):
    def construct(self):
        self.play(Write(Text("By the way...")))
        self.wait()


class EndScreen(PatreonEndScreen):
    CONFIG = {
        "thanks_words": "Special thanks to the following patrons",
    }
