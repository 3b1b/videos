from manim_imports_ext import *


class PhotographVsHologram(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Single piece of film")
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        photo = VGroup(
            Text("Photograph"),
            Text("One point of view"),
        )
        holo = VGroup(
            Text("Our Recording"),
            Text("Many points of view"),
        )
        for part, x in zip([photo, holo], [-1, 1]):
            part[1].scale(0.7)
            part[1].set_color(GREY_A)
            part.arrange(DOWN)
            part.set_x(x * FRAME_WIDTH / 4)
            part.set_y(2)
            part.add_to_back(Arrow(title.get_edge_center(x * RIGHT), part.get_top(), path_arc=-x * 60 * DEGREES))

        self.add(title)

        for part in [photo, holo]:
            self.play(
                GrowArrow(part[0]),
                FadeInFromPoint(part[1], part[0].get_start()),
            )
            self.wait()
            self.play(Write(part[2]))
            self.wait()


class GlintArrow(InteractiveScene):
    def construct(self):
        # Test
        circle = Circle(radius=0.25)
        circle.set_stroke(RED, 4)
        circle.insert_n_curves(20)
        arrow = Vector(DR, thickness=5, fill_color=RED)
        arrow.next_to(circle, UL, buff=0)
        self.play(
            GrowArrow(arrow),
            VShowPassingFlash(circle, time_width=1.5, time_span=(0.5, 1.5)),
        )
        self.wait()


class WriteHologram(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Hologram", font_size=96)
        title.to_edge(UP)
        self.play(Write(title, stroke_color=GREEN_SCREEN, stroke_width=3, lag_ratio=0.1, run_time=3))
        self.wait()


class WriteTransmissionHologram(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Transmission Hologram", font_size=96)
        title.to_edge(UP)
        self.play(Write(title, stroke_color=GREEN_SCREEN, stroke_width=3, lag_ratio=0.1, run_time=3))
        self.wait()


class WriteWhiteLightReflectionHologram(InteractiveScene):
    def construct(self):
        # Test
        title = Text("White Light Reflection Hologram", font_size=72)
        title.to_edge(UP)
        self.play(Write(title, stroke_color=WHITE, stroke_width=3, lag_ratio=0.1, run_time=3))
        self.wait()


class IndicationOnPhotograph(InteractiveScene):
    def construct(self):
        # Image
        image = ImageMobject("ContrastWithPhotography")
        image.set_height(FRAME_HEIGHT)
        # self.add(image)

        # Dots
        p0 = (-3.82, 2.11, 0.0)
        p1 = (-3.14, 2.17, 0.0)
        p2 = (2.61, 2.82, 0.0)
        p3 = (3.53, 2.24, 0.0)

        dot1 = GlowDot(p0, color=TEAL)
        dot2 = GlowDot(p2, color=TEAL)

        arrow = StrokeArrow(p0, p2, buff=SMALL_BUFF, path_arc=-30 * DEGREES)
        arrow.set_stroke(TEAL)

        self.play(FadeIn(dot1), FlashAround(dot1, stroke_width=5, time_width=2, color=TEAL))
        self.play(
            TransformFromCopy(dot1, dot2, path_arc=-30 * DEGREES),
            ShowCreation(arrow, run_time=2),
        )
        self.play(
            dot1.animate.move_to(p1),
            dot2.animate.move_to(p3),
            arrow.animate.match_points(StrokeArrow(p1, p3, buff=0.1, path_arc=-30 * DEGREES)),
            rate_func=there_and_back,
            run_time=3
        )
        self.play(
            FadeOut(dot1),
            FadeOut(dot2),
            FadeOut(arrow),
        )


class ContrastPhotographyAndHolography(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Information lost in ordinary photography", font_size=60)
        title.to_edge(UP)
        underline = Underline(title, buff=-0.05)

        points = BulletedList(
            "All but one viewing angle",
            "Phase",
            buff=LARGE_BUFF
        )

        points.scale(1.25)
        points.next_to(underline, DOWN, buff=1.5)
        points.shift(2 * LEFT)
        self.add(*(p[0] for p in points))

        self.play(
            FadeIn(title),
            ShowCreation(underline)
        )

        for point in points:
            self.wait()
            self.play(Write(point[1:]))

        self.play(
            FlashAround(points[1], color=TEAL, time_width=1.5, run_time=2),
            points[1].animate.set_color(TEAL)
        )
        self.wait()


class Outline(InteractiveScene):
    def construct(self):
        # Add top part
        frame = self.frame
        frame.set_y(2)

        background = FullScreenRectangle()
        background.fix_in_frame()
        self.add(background)

        top_outlines = ScreenRectangle().replicate(3)
        top_outlines.arrange(RIGHT, buff=1.5)
        top_outlines.set_width(13)
        top_outlines.set_stroke(WHITE, 2)
        top_outlines.set_fill(BLACK, 1)
        top_outlines.to_edge(UP, buff=LARGE_BUFF)

        top_images = Group(
            ImageMobject("HologramProcess.jpg"),
            ImageMobject("SimplestHologram.jpg"),
            ImageMobject("HologramEquation.png"),
        )
        top_rects = Group(
            Group(outline, image.replace(outline))
            for outline, image in zip(top_outlines, top_images)
        )

        top_titles = VGroup(
            Text("The process"),
            Text("The simplest hologram"),
            Text("The general derivation"),
        )
        top_titles.scale(0.75)

        for rect, title in zip(top_rects, top_titles):
            title.next_to(rect, UP, SMALL_BUFF)
            self.play(
                FadeIn(rect),
                FadeIn(title, 0.25 * UP)
            )
            self.wait()

        # Highlight process
        self.play(
            FadeOut(top_rects[1:]),
            FadeOut(top_titles[1:]),
            frame.animate.set_height(4).move_to(top_rects[0]),
            run_time=2
        )
        self.wait()
        self.play(
            FadeIn(top_rects[1:]),
            FadeIn(top_titles[1:]),
            frame.animate.to_default_state().set_y(2),
            run_time=2
        )

        # Isolation animation
        def isolate(rects, titles, index, faded_opacity=0.15):
            return AnimationGroup(
                *(
                    title.animate.set_opacity(1 if i == index else faded_opacity)
                    for i, title in enumerate(titles)
                ),
                *(
                    rect.animate.set_opacity(1 if i == index else faded_opacity)
                    for i, rect in enumerate(rects)
                ),
            )

        self.play(isolate(top_rects, top_titles, 0))
        self.wait()
        self.play(isolate(top_rects, top_titles, 1))
        self.wait()

        # Break down middle step
        mid_outlines = top_outlines.copy()
        mid_outlines.set_opacity(1)
        mid_outlines.next_to(top_rects, DOWN, buff=1.5)
        outer_lines = VGroup(
            CubicBezier(
                top_rects[1].get_corner(DL),
                top_rects[1].get_corner(DL) + DOWN,
                mid_outlines[0].get_corner(UL) + 2 * UP,
                mid_outlines[0].get_corner(UL),
            ),
            CubicBezier(
                top_rects[1].get_corner(DR),
                top_rects[1].get_corner(DR) + DOWN,
                mid_outlines[2].get_corner(UR) + 2 * UP,
                mid_outlines[2].get_corner(UR),
            ),
        )
        mid_images = Group(
            ImageMobject("ZonePlateExposure"),
            ImageMobject("DotReconstruction"),
            ImageMobject("MultipleDotHologram"),
        )

        mid_rects = Group(
            Group(outline, image.replace(outline))
            for outline, image in zip(mid_outlines, mid_images)
        )

        mid_titles = VGroup(
            Text("Exposure pattern"),
            Text("Reconstruction"),
            Text("Added complexity"),
        )
        mid_titles.scale(0.75)
        for rect, title in zip(mid_rects, mid_titles):
            title.next_to(rect, UP, SMALL_BUFF)
            rect.save_state()
            rect.replace(top_rects[1])
            rect.set_opacity(0)

        self.play(
            LaggedStartMap(Restore, mid_rects),
            ShowCreation(outer_lines, lag_ratio=0),
            frame.animate.set_y(0),
            FadeIn(mid_titles, time_span=(1.5, 2.0), lag_ratio=0.025),
            run_time=2
        )
        self.wait()
        for index in range(3):
            self.play(isolate(mid_rects, mid_titles, index))
            self.wait()
        self.play(isolate(mid_rects, mid_titles, 0))
        self.wait()

        # Minilesson
        low_outline = mid_outlines[1].copy()
        low_outline.next_to(mid_rects[1], DOWN, buff=1.5)

        low_image = ImageMobject("DiffractionGrating")
        low_image.replace(low_outline)
        low_rect = Group(low_outline, low_image)
        low_rect.shift(1.5 * LEFT)

        in_arrow = Arrow(mid_rects[0].get_bottom() + LEFT, low_rect.get_left(), path_arc=PI / 2, thickness=5, buff=0.15)
        up_arrow = Arrow(low_rect, mid_rects[1], thickness=5, buff=0.15)

        low_title = Text("Mini-lesson on\nDiffraction Gratings")
        low_title.next_to(low_rect, DOWN)

        self.play(
            frame.animate.set_y(-4),
            FadeIn(low_rect, DOWN),
            FadeIn(low_title, DOWN),
            GrowArrow(in_arrow),
            run_time=2
        )
        self.play(GrowArrow(up_arrow))
        self.wait()

        # Zoom in on mini-lesson
        frame.save_state()
        self.play(
            frame.animate.set_height(4).move_to(Group(low_rect, low_title)),
            FadeOut(VGroup(in_arrow, up_arrow)),
            run_time=2,
        )
        self.wait()
        self.play(
            Restore(frame),
            FadeIn(VGroup(in_arrow, up_arrow)),
            run_time=2
        )
        self.wait()

        # Back to the middle
        self.play(
            LaggedStartMap(FadeOut, Group(low_title, low_rect, arrow), lag_ratio=0.1, run_time=1),
            frame.animate.set_y(0).set_anim_args(run_time=2),
        )
        self.wait()
        self.play(isolate(mid_rects, mid_titles, 2))
        self.wait()

        # Back to the top
        self.play(
            LaggedStart(
                (mid_rect.animate.replace(top_rects[1]).set_opacity(0)
                for mid_rect in mid_rects),
                lag_ratio=0.05,
                group_type=Group
            ),
            FadeOut(mid_titles),
            Uncreate(outer_lines, lag_ratio=0),
            frame.animate.set_y(2),
            run_time=2
        )
        self.wait()
        self.play(isolate(top_rects, top_titles, 2))
        self.wait()


class GoalOfRediscovery(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        self.play(
            self.change_students("pondering", "confused", "maybe", look_at=self.screen),
            morty.change("guilty"),
        )
        self.wait(2)
        self.play(
            morty.says("I want this to\nfeel rediscovered"),
            self.change_students("tease", "erm", "pondering", look_at=morty.eyes)
        )
        self.wait(2)
        self.look_at(self.screen)
        self.play(self.change_students("thinking", "pondering", "thinking"))
        self.wait(4)


class NameCraigAndSally(InteractiveScene):
    def construct(self):
        # Test
        image = ImageMobject("CraigSallyPaulGrant")
        image.set_height(FRAME_HEIGHT)

        craig_point = (1.2, 2.0, 0)
        sally_point = (3.9, 1.3, 0)

        craig_arrow = Vector(DR, thickness=5)
        craig_arrow.next_to(craig_point, UL)
        craig_name = Text("Craig Newswanger")
        craig_name.next_to(craig_arrow.get_start(), UP, buff=SMALL_BUFF)

        sally_arrow = Vector(DL, thickness=5)
        sally_arrow.next_to(sally_point, UR)
        sally_name = Text("Sally Weber")
        sally_name.next_to(sally_arrow.get_start(), UP, buff=SMALL_BUFF)

        VGroup(craig_arrow, craig_name, sally_arrow, sally_name).set_fill(WHITE).set_backstroke(BLACK, 5)

        self.play(
            GrowArrow(craig_arrow),
            FadeIn(craig_name),
        )
        self.play(
            GrowArrow(sally_arrow),
            FadeIn(sally_name),
        )


class ShowALens(InteractiveScene):
    def construct(self):
        # Add lens
        arc = Arc(-30 * DEGREES, 60 * DEGREES)
        flipped = arc.copy().rotate(PI).next_to(arc, LEFT, buff=0)
        arc.append_vectorized_mobject(flipped)
        lens = arc
        lens.set_stroke(WHITE, 2)
        lens.set_fill(GREY_E, 1)
        lens.set_shading(0.1, 0.5, 0)
        lens.set_height(3)

        # Add lines
        n_lines = 20
        points = np.linspace(lens.get_top(), lens.get_bottom(), n_lines)
        focal_point = lens.get_left() + 2 * LEFT
        in_lines = VGroup(Line(point + 8 * RIGHT, point) for point in points)
        out_lines = VGroup(Line(point, focal_point) for point in points)
        for in_line, out_line in zip(in_lines, out_lines):
            out_line.scale(1.25, about_point=out_line.get_start())
            in_line.append_vectorized_mobject(out_line)

        in_lines.set_stroke(WHITE, 1)
        in_lines.insert_n_curves(100)

        self.add(in_lines, lens)
        self.play(ShowCreation(in_lines, lag_ratio=0.01, run_time=2, rate_func=linear))
        self.wait()

        # Label viewing angle
        text = Text("Single viewing angle")
        text.next_to(lens.get_top(), UR)
        arrow = Vector(RIGHT).next_to(text, RIGHT)

        self.play(LaggedStart(Write(text, run_time=1), GrowArrow(arrow), lag_ratio=0.5))
        self.wait()


class AskAboutRecordingPhase(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer()
        morty.to_corner(DR)
        randy = Randolph(height=2.5)
        randy.next_to(morty, LEFT, buff=1.5, aligned_edge=DOWN)
        for pi in [morty, randy]:
            pi.body.insert_n_curves(100)

        self.add(morty, randy)
        self.play(
            morty.says(
                Text("""
                    How could you
                    detect this phase
                    difference?
                """),
                look_at=randy.eyes,
            ),
            randy.change("pondering", morty.eyes)
        )
        self.play(Blink(randy))
        self.play(
            randy.animate.look_at(3 * UL),
            morty.change("tease", 3 * UL),
        )
        self.play(randy.change("erm", 3 * UL))
        self.play(Blink(morty))
        self.play(Blink(randy))
        self.wait(2)
        self.play(randy.change("confused", 3 * UL))
        self.play(morty.change("pondering", 3 * UL))
        self.play(Blink(randy))
        self.wait(3)
        self.play(randy.change("thinking", 3 * LEFT))
        self.play(Blink(randy))
        self.play(Blink(morty))
        self.wait(2)


class HowIsThisHelpful(TeacherStudentsScene):
    def construct(self):
        # Test
        stds = self.students
        morty = self.teacher
        self.play(LaggedStart(
            stds[1].says("What does this have\nto do with recording 3d?", mode="maybe", look_at=self.screen),
            stds[0].change("confused", self.screen),
            stds[2].change("erm", self.screen),
            lag_ratio=0.25,
            run_time=2
        ))
        self.play(morty.change('well'))
        self.wait(2)
        self.play(self.change_students("maybe", "confused", "pondering", look_at=self.screen))
        self.wait(5)


class NoWhiteLight(InteractiveScene):
    def construct(self):
        # Test
        white_label = VGroup(
            Text("White light", font_size=60),
            Text("(many frequencies)").set_color(GREY_B)
        )
        white_label.arrange(DOWN)
        white_label.to_edge(UP).set_x(-3)
        laser_label = VGroup(
            Text("Laser", font_size=60).set_color(GREEN_SCREEN),
            Text("(single frequency)").set_color(GREY_B),
        )
        laser_label.arrange(DOWN)
        laser_label.move_to(white_label, UP)

        cross = Cross(white_label, stroke_width=(0, 7, 7, 7, 0))

        self.add(white_label)
        self.wait()
        self.play(ShowCreation(cross))
        self.wait()
        self.play(
            FadeOut(VGroup(white_label, cross), DR),
            FadeIn(laser_label, DR),
        )
        self.wait()


class AnnotateSetup(InteractiveScene):
    def construct(self):
        # Add iamge
        image = ImageMobject("CraigSallyHologramSetup")
        image.set_height(FRAME_HEIGHT)
        # self.add(image)

        # Reference polarization
        pol_words = Text("Correct the polarization")
        pol_words.set_backstroke(BLACK, 5)
        point = (-0.5, 1.2, 0)
        pol_words.next_to(point, UR, buff=1.5).shift(LEFT)
        pol_arrow = Arrow(pol_words["Correct"].get_bottom(), point)

        self.play(
            Write(pol_words),
            GrowArrow(pol_arrow)
        )
        self.wait()
        self.play(
            FadeOut(pol_words),
            FadeOut(pol_arrow),
        )

        # Show object beam
        obj_beam_points = [
            (-1.625, 2.167, 0),
            (-1.625, 1.264, 0),
            (0.375, 1.264, 0),
            (-0.625, -3.625, 0),
            (-0.111, -3.167, 0),
        ]

        obj_beam = VMobject().set_points_as_corners(obj_beam_points)
        obj_beam.set_stroke(GREEN_SCREEN, 4)
        obj_beam_name = Text("Object beam")
        obj_beam_name.set_color(GREEN_SCREEN)
        obj_beam_name.next_to(obj_beam.pfp(0.5), RIGHT)

        scene_point = (1.278, -1.972, 0)
        scene_circle = Circle(radius=0.75).shift(scene_point)
        obj_beam_spread = VGroup(
            Line(obj_beam_points[-1], scene_circle.pfp(a) + np.random.uniform(-0.025, 0.025, 3))
            for a in np.arange(0, 1, 0.001)
        )
        obj_beam_spread.set_stroke(GREEN_SCREEN, 1, 0.25)

        self.play(
            ShowCreation(obj_beam, rate_func=linear),
            FadeIn(obj_beam_name, lag_ratio=0.1),
        )
        self.play(ShowCreation(obj_beam_spread, lag_ratio=0, rate_func=rush_from))
        self.wait()

        # Show reference beam
        ref_beam_points = [
            obj_beam_points[0],
            (-1.61, -1.39, 0),
            (-1.056, -1.809, 0),
        ]
        ref_beam = obj_beam.copy()
        ref_beam.set_points_as_corners(ref_beam_points)
        ref_beam_name = Text("Reference beam")
        ref_beam_name.match_style(obj_beam_name)
        ref_beam_name.next_to(ref_beam.pfp(0.5), LEFT)

        ref_circle = Circle(radius=0.5)
        ref_circle.rotate(PI / 2, RIGHT)
        ref_circle.move_to((1.23, -3.39, 0))
        ref_beam_spread = VGroup(
            Line(ref_beam_points[-1], ref_circle.pfp(a) + np.random.uniform(-0.025, 0.025, 3))
            for a in np.arange(0, 1, 0.001)
        )
        ref_beam_spread.match_style(obj_beam_spread)

        self.play(
            FadeTransform(obj_beam_name, ref_beam_name),
            ReplacementTransform(obj_beam, ref_beam),
            ReplacementTransform(obj_beam_spread, ref_beam_spread),
        )
        self.wait()


class ArrowWithQMark(InteractiveScene):
    def construct(self):
        arrow = Arrow(3 * LEFT, 3 * RIGHT, path_arc=-PI / 2)
        q_marks = Tex(R"???", font_size=72)
        q_marks.next_to(arrow, UP)

        self.play(
            Write(arrow),
            Write(q_marks),
            run_time=2
        )
        self.wait()


class ProblemSolvingTipNumber1(InteractiveScene):
    def construct(self):
        # Title
        title = Text("Universal problem-solving tip #1", font_size=72)
        title.to_edge(UP, buff=LARGE_BUFF)
        underline = Underline(title, buff=-SMALL_BUFF)
        parts = VGroup(title[text][0] for text in re.split(" |-", title.text))

        words = Text("Start with the simplest\nversion of your problem", font_size=72)
        words.set_color(YELLOW)

        self.play(LaggedStartMap(FadeIn, parts, shift=0.25 * UP, lag_ratio=0.25, run_time=2))
        self.play(ShowCreation(underline))
        self.wait()
        self.play(Write(words), run_time=3)
        self.wait()


class NameElectromagneticField(InteractiveScene):
    def construct(self):
        # Test
        name1 = Text("Electromagetic Field")
        name2 = Text("Electric Field")
        for name in [name1, name2]:
            name.set_backstroke(BLACK, 5)
            name.scale(1.5)
            name.to_edge(UP)

        expl = Text("3d Vector\nat each point")
        expl.set_x(FRAME_WIDTH / 4)
        expl.to_edge(UP)
        arrow = Vector(RIGHT)
        arrow = Vector(2 * RIGHT)
        arrow.next_to(expl, LEFT)

        self.play(Write(name1), run_time=1)
        self.wait()
        self.play(TransformMatchingStrings(name1, name2), run_time=1)
        self.wait()
        self.play(
            name2.animate.next_to(arrow, LEFT),
            GrowArrow(arrow),
            FadeIn(expl, RIGHT),
        )
        self.wait()


class HoldUpDiffractionEquationInAnticipation(TeacherStudentsScene):
    def construct(self):
        # Test
        equation = Tex(R"d \cdot \sin(\theta) = n \lambda", font_size=60)
        equation.move_to(self.hold_up_spot, DOWN)
        name = TexText("Diffraction equation")
        name.next_to(equation, UP, buff=1.5)
        arrow = Arrow(name, equation)

        morty = self.teacher
        morty.body.insert_n_curves(100)

        self.play(
            morty.change("raise_right_hand", equation),
            FadeIn(equation, UP),
            self.change_students("confused", "confused", "erm", equation)
        )
        self.play(
            Write(name),
            GrowArrow(arrow),
        )
        self.wait()
        self.play(
            self.change_students("maybe", "confused", "sassy", look_at=self.screen)
        )
        self.wait(3)
        self.play(
            morty.says("You can discover\nthis yourself", mode="hooray"),
            LaggedStartMap(FadeOut, VGroup(equation, arrow, name), shift=UR, lag_ratio=0.2, run_time=2),
            self.change_students("pondering", "pondering", "thinking", look_at=morty.eyes)
        )
        self.wait(3)


class HowLongIsThatSnippet(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.says("How long is\nthat snippet?"),
            self.change_students("happy", "tease", "thinking", look_at=self.screen)
        )
        self.wait(3)


class HoldUpDiffractionEquation(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher

        equation = Tex(R"d \cdot \sin(\theta) = \lambda", t2c={R"\lambda": TEAL}, font_size=60)
        equation.move_to(self.hold_up_spot, DOWN)

        words = Text("Remember this")
        words.next_to(equation, UP, 1.5)
        equation.to_edge(RIGHT)
        arrow = Arrow(words, equation)

        self.play(
            morty.change("raise_left_hand", look_at=equation),
            FadeIn(equation, UP),
            self.change_students("pondering", "pondering", "pondering", equation)
        )
        self.play(
            FadeIn(words, lag_ratio=0.1),
            GrowArrow(arrow),
        )
        self.play(morty.change("tease"))
        self.wait(3)

        morty.body.insert_n_curves(100)
        self.play(
            self.change_students("thinking", "erm", "pondering", look_at=self.screen),
            morty.change("raise_right_hand", look_at=self.screen),
        )
        self.wait(4)


class AskAboutCenterBeam(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        randy.to_edge(DOWN)
        self.play(
            randy.thinks("Why is there a\ncentral beam?", mode="confused")
        )
        self.play(Blink(randy))
        self.play(randy.animate.look_at(3 * UR))
        self.play(randy.animate.look_at(3 * DR))


class DistanceApproximation(InteractiveScene):
    def construct(self):
        # Test
        expr = Tex(R"\sqrt{L^2 + x^2} \approx L + \frac{x^2}{2L}")
        expr.to_corner(UL)

        circle = SurroundingRectangle(expr[R"\frac{x^2}{2L}"])
        circle.round_corners()
        circle.set_stroke(RED)
        words = Text("First order\nTaylor approx.")
        words.next_to(circle, DOWN, LARGE_BUFF)
        words.set_color(RED)
        words.set_backstroke(BLACK, 5)
        arrow = Arrow(words.get_top(), circle.get_bottom())
        arrow.set_color(RED)

        self.play(Write(expr))
        self.wait()
        self.play(
            ShowCreation(circle),
            GrowArrow(arrow),
            FadeIn(words)
        )
        self.wait()


class CircleDiffractionEquation(InteractiveScene):
    def construct(self):
        # Add equation
        self.add(FullScreenRectangle())
        equation = Tex(
            R"{d} \cdot \sin(\theta) = n \lambda",
            font_size=72,
            t2c={R"{d}": BLUE, R"\theta": YELLOW, R"\lambda": TEAL}
        )
        equation.set_backstroke(BLACK)
        title = Text("The Diffraction Equation", font_size=60)
        title.set_color(GREY_A)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        equation.next_to(title, DOWN, MED_LARGE_BUFF)

        self.add(title)
        self.add(equation)
        self.play(
            FlashAround(equation, run_time=4, time_width=1.5, buff=MED_SMALL_BUFF),
        )
        self.wait()


class DiffractionGratingGreenLaserExampleNumbers(InteractiveScene):
    def construct(self):
        # Test
        lines = VGroup(
            VGroup(
                Text("Grating: 500 lines / mm"),
                Tex(R"\Rightarrow d = 2 \times 10^{-6} \text{m}"),
            ).arrange(RIGHT),
            VGroup(
                Text("Green light wavelength: 500 nm "),
                Tex(R"\Rightarrow \lambda = 5 \times 10^{-7} \text{m}"),
            ).arrange(RIGHT),
            Tex(R"\theta = \sin^{-1}(\lambda / d) = \sin^{-1}\left(5 \times 10^{-7} / 2 \times 10^{-6} \right) \approx 14.5^\circ")
        )
        lines.arrange(DOWN, aligned_edge=LEFT, buff=LARGE_BUFF)
        lines.to_edge(LEFT)
        lines[0][1].align_to(lines[1][1], LEFT)
        self.add(lines)


class AnnotateZerothOrderBeam2(InteractiveScene):
    def construct(self):
        image = ImageMobject("DiffractionGratingGreenLaser.jpg")
        # self.add(image.set_height(FRAME_HEIGHT))

        # Names
        points = [
            (-0.67, 1.64, 0.0),
            (3.64, 1.9, 0.0),
            (-4.9, 1.85, 0.0),
            (-5.4, 1.01),
            (2.42, 0.028),
        ]

        names = VGroup(
            Text("Zeroth-order beam"),
            Text("First-order beams"),
            Text("Second-order beams"),
        )
        names.next_to(points[0], UP, buff=2.5)
        names.shift_onto_screen()
        arrow0 = Arrow(names[0], points[0])
        arrows1 = VGroup(
            Arrow(names[1].get_bottom(), points[1]),
            Arrow(names[1].get_bottom(), points[2]),
        )
        arrows2 = VGroup(
            Arrow(names[2].get_bottom(), points[3]),
            Arrow(names[2].get_bottom(), points[4]),
        )
        VGroup(names, arrow0, arrows1, arrows2).set_fill(GREY_E)

        self.play(
            Write(names[0]),
            GrowArrow(arrow0),
            run_time=1
        )
        self.wait()
        self.remove(arrow0)
        self.play(
            TransformMatchingStrings(names[0], names[1], key_map={"Zeroth": "First", "m": "ms"}),
            TransformFromCopy(arrow0.replicate(2), arrows1),
            run_time=1
        )
        self.wait()
        self.play(
            TransformMatchingStrings(names[1], names[2], key_map={"First": "Second"}),
            ReplacementTransform(arrows1, arrows2),
            run_time=1
        )
        self.wait()


class ExactSpacingQuestion(InteractiveScene):
    def construct(self):
        # Test
        question = Text("What is the exact spacing?", font_size=72)
        question.set_backstroke()
        question.to_edge(UP)
        subq = TexText(R"As a function of $\theta'$")
        subq.set_color(GREY_A)

        self.play(Write(question, stroke_color=WHITE))
        self.wait()
        self.play(question.animate.scale(48 / 72).set_x(-0.25 * FRAME_WIDTH))
        self.wait()
        subq.next_to(question, DOWN)
        self.play(FadeIn(subq, 0.5 * DOWN))
        self.wait()


class DejaVu(TeacherStudentsScene):
    def construct(self):
        # Test
        stds = self.students
        self.play(
            stds[0].change("pondering", self.screen),
            stds[1].change("pondering", self.screen),
            stds[2].says("That looks\nfamiliar", bubble_config=dict(direction=LEFT), look_at=self.screen),
            self.teacher.change("tease"),
        )
        self.wait(2)

        diff_eq = Tex(R"d \cdot \sin(\theta) = \lambda")
        diff_eq.next_to(stds[0], UR)
        self.play(
            stds[1].change("raise_left_hand", look_at=diff_eq),
            FadeIn(diff_eq, UP)
        )
        self.look_at(diff_eq)
        self.play(stds[2].change("pondering", diff_eq))
        self.wait(3)


class WhatAHologramReconstructs(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle())
        title = Text("What a Hologram reproduces", font_size=60)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        underline = Underline(title, buff=-0.05)

        items = VGroup(
            Text("1) Copy of the reference wave"),
            Text("2) Copy of the object wave"),
            TexText(R"3) Reflection$^*$ of the object wave"),
        )
        items.scale(0.8)
        items.arrange(DOWN, buff=1.75, aligned_edge=LEFT)
        items.next_to(underline, DOWN, LARGE_BUFF)
        items.to_edge(LEFT)

        self.play(
            FadeIn(title),
            ShowCreation(underline)
        )

        self.play(
            LaggedStart(
                (FadeIn(item[:2])
                for item in items),
                lag_ratio=0.05,
            )
        )
        self.wait()
        for n, item in enumerate(items):
            self.play(
                items[:n].animate.set_opacity(0.25),
                FadeIn(item[2:], lag_ratio=0.1)
            )
            self.wait()


class AskAboutHigherOrderBeams(TeacherStudentsScene):
    def construct(self):
        stds = self.students
        morty = self.teacher
        self.play(LaggedStart(
            stds[0].change('pondering', self.screen),
            stds[1].says("What about higher\norder beams?", look_at=self.screen),
            stds[2].change("confused", self.screen),
            morty.change("tease")
        ))
        self.wait()
        self.play(stds[1].change("sassy", morty.eyes))
        self.wait(6)


class BinaryVsSinusoidalDiffraction(InteractiveScene):
    def construct(self):
        # Add axes
        top_axes, low_axes = two_axes = VGroup(
            Axes((0, 4, 0.5), (0, 1, 0.5), width=6, height=1.5)
            for _ in range(2)
        )
        two_axes.arrange(DOWN, buff=2.0)
        two_axes.to_edge(LEFT, buff=1.5)

        for axes in two_axes:
            y_label = Text("Opacity", font_size=30)
            y_label.next_to(axes.y_axis, UP, buff=MED_SMALL_BUFF)
            axes.add(y_label)
            axes.y_axis.add_numbers(font_size=16, num_decimal_places=1)

        self.add(two_axes)

        # Add graphs
        discontinuities = [*np.arange(5), *(np.arange(5) + 0.2)]
        discontinuities.sort()
        top_graph = top_axes.get_graph(self.func1, x_range=(0, 3.99), discontinuities=discontinuities)
        top_graph.set_stroke(TEAL, 2)

        low_axes.num_sampled_graph_points_per_tick = 20
        low_graph = low_axes.get_graph(self.func2)
        low_graph.set_stroke(BLUE, 2)

        graphs = VGroup(top_graph, low_graph)

        # Add gratings
        top_grating = self.get_grating(top_axes, self.func1, 1000)
        low_grating = self.get_grating(low_axes, self.func2, 1000)
        gratings = VGroup(top_grating, low_grating)

        # Animate in
        graphs.set_stroke(width=1)
        self.add(graphs)
        for graph, grating in [(top_graph, top_grating), (low_graph, low_grating)]:
            self.add(grating, graph)
            self.play(
                graph.animate.set_stroke(width=2),
                VShowPassingFlash(graph.copy().set_stroke(width=5), time_width=3, rate_func=linear),
                # ShowCreation(graph, rate_func=linear),
                FadeIn(grating, lag_ratio=1e-3),
                run_time=2
            )
            self.play(
                grating.animate.next_to(grating, RIGHT, MED_LARGE_BUFF),
                run_time=2
            )
            self.wait()

        # Labels
        high_beam_labels = VGroup(
            Text("Higher order beams", font_size=36).next_to(grating, UP)
            for grating in gratings
        )
        check = Checkmark().set_color(GREEN).scale(1.5)
        check.next_to(high_beam_labels[0])
        ex = Exmark().set_color(RED).scale(1.5)
        ex.next_to(high_beam_labels[1])

        self.play(LaggedStart(
            FadeIn(high_beam_labels),
            Write(check),
            Write(ex),
            lag_ratio=0.5
        ))
        self.wait()

    def get_grating(self, axes, func, n_slits):
        grating = Rectangle().replicate(n_slits)
        grating.set_stroke(width=0)
        grating.set_fill(GREY_D)
        grating.set_shading(0.25, 0.25, 0)
        grating.arrange(RIGHT, buff=0)
        grating.set_shape(axes.x_axis.get_width(), axes.y_axis.get_height())
        grating.move_to(axes.c2p(0, 0), DL)

        for mob in grating:
            mob.set_opacity(func(axes.x_axis.p2n(mob.get_center())))

        # grating.next_to(grating, RIGHT, MED_LARGE_BUFF)

        return grating

    def func1(self, x):
        return 0 if x % 1 < 0.2 else 1

    def func2(self, x):
        return math.sin(TAU * x)**2


class AskWhyForSinusoidalGratingFact(TeacherStudentsScene):
    def construct(self):
        # Test
        stds = self.students
        morty = self.teacher

        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)

        self.play(LaggedStart(
            stds[1].says("Wait, why?", mode="confused", look_at=self.screen),
            stds[0].change("pondering", self.screen),
            stds[2].change("maybe", self.screen),
        ))
        self.wait()

        implication = VGroup(
            Tex(R"\Leftarrow").scale(2),
            Text("More formal\nholography explanation")
        )
        implication.arrange(RIGHT, buff=0.25)
        implication.move_to(2.5 * UP + 2.5 * RIGHT)

        self.play(
            morty.change("raise_right_hand"),
            GrowFromPoint(implication, morty.get_corner(UL))
        )
        self.play(self.change_students("thinking", "sassy", "erm"))
        self.look_at(self.screen)
        self.play(stds[1].change("pondering", self.screen))
        self.wait(4)


class NeedsMoreRigor(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher

        arrow = Vector(LEFT, thickness=4)
        arrow.move_to(2.5 * UP)
        words = Text("Sketchy and non-rigorous")
        words.next_to(arrow)

        self.play(
            morty.change("guilty"),
            self.change_students("erm", "hesitant", "sassy", look_at=self.screen)
        )
        self.play(
            GrowArrow(arrow),
            Write(words)
        )
        self.wait(4)


class CompareFilmTypes(InteractiveScene):
    def construct(self):
        # Test
        categories = VGroup(
            Text("Polaroid instant film"),
            Text("Microfilm"),
            Text("What we used (Bayfol HX200)"),
        )
        categories[2]["(Bayfol HX200)"].set_color(GREY_B)

        resolutions = VGroup(
            Text("10-20 lines per mm"),
            Text("100-200 lines per mm"),
            Text("> 5,000 lines per mm"),
        )

        arrows = Vector(RIGHT, thickness=4).replicate(3)

        last_group = VGroup()
        for text, res, arrow in zip(categories, resolutions, arrows):
            arrow = Vector(RIGHT)
            group = VGroup(text, arrow, res)
            group.arrange(RIGHT)
            res.align_to(text, UP)
            group.to_edge(UP)

            text.save_state()
            text.set_x(0)

            self.play(
                FadeOut(last_group, 0.5 * UP),
                FadeIn(text, 0.5 * UP),
            )
            self.wait()
            self.play(
                Restore(text),
                GrowArrow(arrow),
                FadeIn(res, RIGHT)
            )
            self.wait()

            last_group = group


class TheresADeeperIssue(TeacherStudentsScene):
    def construct(self):
        for std in self.students:
            std.change_mode("pondering")
            std.look_at(self.screen)
        self.play(
            self.teacher.says("There's a\ndeeper issue"),
            self.change_students("erm", "guilty", "hesitant", look_at=self.teacher.eyes)
        )
        self.wait(2)
        self.play(
            self.change_students("pondering", "pondering", 'pondering', look_at=self.screen)
        )
        self.wait(2)


class SharpMindedViewer(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle())
        randy = Randolph()
        randy.to_edge(LEFT)
        randy.shift(DOWN)

        self.play(randy.change("sassy", RIGHT))
        self.play(Blink(randy))
        self.wait(2)
        self.play(randy.change("confused", RIGHT))
        self.wait()
        self.play(Blink(randy))
        self.wait()


class WhatThisExplanationLacks(InteractiveScene):
    def construct(self):
        # Title
        title = Text("Gaps in the single-point explanation", font_size=60)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        underline = Underline(title, buff=-0.05)

        self.play(
            FadeIn(title, lag_ratio=0.1),
            ShowCreation(underline),
        )
        self.wait()

        # Add points
        points = BulletedList(
            "How do errors in the various approximations accumulate?",
            "Why don't higher order beams exist?",
            "What happens when you change the reference beam angle?",
            buff=0.5,
            font_size=48,
        )
        points.next_to(underline, DOWN, buff=0.5)
        points.to_edge(LEFT, buff=LARGE_BUFF)

        self.play(
            LaggedStartMap(FadeIn, points, shift=UP, lag_ratio=0.5)
        )
        self.wait()
        self.play(points[:2].animate.set_opacity(0.5))
        self.wait()


class SimplicityToGenerality(InteractiveScene):
    def construct(self):
        # Initialize little triangles
        frame = self.frame
        surface = Torus(resolution=(49, 49))
        surface.set_width(4)
        verts = surface.get_shader_data()['point']
        triangles = VGroup(
            Polygon(*tri)
            for tri in zip(verts[0::3], verts[1::3], verts[2::3])
        )
        triangles.set_stroke(WHITE, 0.5, 0.5)

        tri1 = triangles[3500]
        dists = [get_norm(tri.get_center() - tri1.get_center()) for tri in triangles]
        neighbors = VGroup(triangles[idx] for idx in np.argsort(dists))
        ng1 = neighbors[1:10]
        ng2 = neighbors[10:200]
        ng3 = neighbors[200:]
        for group in [ng1, ng2, ng3]:
            group.save_state()
            for tri in group:
                tri.become(tri1)
                tri.set_stroke(width=0.1, opacity=0.25)
        ng3.set_stroke(opacity=0.05)

        frame.reorient(4, 83, 0, (-0.11, -0.05, -0.34), 1.91)

        self.add(tri1)
        self.play(
            LaggedStart(
                Restore(ng1, lag_ratio=0.3),
                Restore(ng2, lag_ratio=0.02),
                Restore(ng3, lag_ratio=3e-4),
                lag_ratio=0.4
            ),
            frame.animate.reorient(29, 64, 0, (-0.15, 0.02, -0.43), 3.21),
            run_time=9,
        )
        self.add(triangles)

        # True surface
        equation = Tex(R"\left(x^2+y^2+z^2+R^2-r^2\right)^2=4 R^2\left(x^2+y^2\right)")
        equation.fix_in_frame()
        equation.to_edge(UP, buff=0.35)

        new_surface = ParametricSurface(
            lambda u, v: surface.uv_func(v, u),
            u_range=(0, TAU),
            v_range=(0, TAU),
            resolution=(250, 250)
        )
        new_surface.set_width(4)
        new_surface.set_opacity(0.5)
        new_surface.always_sort_to_camera(self.camera)

        triangles.shuffle()
        self.play(
            FadeOut(triangles, scale=0.8, lag_ratio=3e-4),
            frame.animate.reorient(77, 62, 0, (-0.27, -0.05, -0.23), 4.29),
            run_time=2
        )
        self.play(
            ShowCreation(new_surface, run_time=3),
            frame.animate.reorient(58, 66, 0, (-0.27, -0.05, -0.23), 4.29).set_anim_args(run_time=4),
        )
        self.play(frame.animate.reorient(50, 79, 0, (-0.27, -0.05, -0.23), 4.29), run_time=3)


class ModeledAs2D(InteractiveScene):
    def construct(self):
        # Test
        words = Text("Modeled as 2D")
        words.to_edge(UP)
        point = 1.5 * RIGHT
        arrow = Arrow(words.get_right(), point, path_arc=-180 * DEGREES, thickness=4)

        self.play(
            FadeIn(words, 0.25 * UP),
            DrawBorderThenFill(arrow, run_time=2)
        )
        self.wait()


class ThinkingAboutRediscovery(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        randy.body.insert_n_curves(100)
        randy.to_corner(DL)
        bubble = ThoughtBubble(filler_shape=(5, 2))
        bubble.pin_to(randy)
        bubble = bubble[0]
        bubble.set_fill(opacity=0)

        rect = FullScreenRectangle().set_fill(BLACK, 1)
        rect.append_points([rect.get_points()[-1], *bubble[-1].get_points()])
        self.add(rect, randy)

        self.play(
            randy.change('pondering'),
            Write(bubble, stroke_color=WHITE, lag_ratio=0.2)
        )
        self.play(Blink(randy))
        self.wait(2)
        self.play(randy.change('thinking', bubble.get_top()))
        self.play(Blink(randy))
        self.play(randy.change('tease', bubble.get_top()))
        for _ in range(2):
            self.play(Blink(randy))
            self.wait(2)


class IntroduceFormalSection(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(LaggedStart(
            stds[0].change("sassy", morty.eyes),
            stds[1].says("Okay but really, why\ndoes it work?", look_at=morty.eyes),
            stds[2].change("angry", look_at=morty.eyes),
            morty.change("guilty"),
        ))
        self.wait(2)
        self.play(LaggedStart(
            morty.change("tease"),
            stds[0].change("well"),
            stds[1].debubble(),
            stds[2].change("hesitant"),
            lag_ratio=0.5
        ))
        self.wait(2)

        # Show complex plane
        plane = ComplexPlane((-3, 3), (-3, 3))
        plane.faded_lines.set_stroke(BLUE_E, 1, 0.5)
        plane.axes.set_stroke(width=1)
        plane.background_lines.set_stroke(width=1)
        plane.set_width(3)
        plane.next_to(self.hold_up_spot, UP)

        a, b = (2, 1.5)
        z = complex(a, b)
        z_dot = GlowDot(plane.n2p(0), color=WHITE)
        z_dot.set_z_index(1)
        h_line = Line(plane.n2p(0), plane.n2p(a))
        v_line = Line(plane.n2p(a), plane.n2p(z))
        h_line.set_stroke(YELLOW, 3)
        v_line.set_stroke(RED, 3)
        a_label = Tex(R"a", font_size=24)
        a_label.match_color(h_line)
        a_label.next_to(h_line, DOWN, buff=0.1)
        b_label = Tex(R"bi", font_size=24)
        b_label.match_color(v_line)
        b_label.next_to(v_line, RIGHT, buff=0.1)
        z_label = Tex(R"a + bi", font_size=24)
        z_label.next_to(v_line.get_end(), UR, SMALL_BUFF)
        VGroup(a_label, b_label, z_label).set_backstroke(BLACK, 3)

        self.play(
            morty.change("raise_right_hand", plane),
            self.change_students(*3 * ["pondering"], look_at=plane),
            FadeIn(plane, UP),
        )
        self.play(
            ShowCreation(h_line),
            FadeIn(a_label, 0.5 * RIGHT),
            z_dot.animate.move_to(plane.n2p(a)),
        )
        self.play(
            ShowCreation(v_line),
            FadeIn(b_label, 0.5 * UP),
            z_dot.animate.move_to(plane.n2p(z)),
        )
        self.play(
            TransformMatchingShapes(VGroup(a_label, b_label).copy(), z_label)
        )
        self.play(self.change_students("erm", "thinking", "well", look_at=plane))
        self.wait(2)


class HoldUpEquation(InteractiveScene):
    def construct(self):
        pass


class GaborQuote(InteractiveScene):
    def construct(self):
        # Test
        quote = TexText(R"""
            ``This was an exercise\\
            in serendipity, the art of\\
            looking for something and\\
            finding something else''
        """, font_size=60, alignment="")
        quote.to_edge(RIGHT)

        self.play(Write(quote), run_time=2, lag_ratio=0.01)
        self.play(quote["looking for something"].animate.set_color(BLUE), lag_ratio=0.1)
        self.play(quote["finding something else"].animate.set_color(TEAL), lag_ratio=0.1)
        self.wait()


class GaborQuote2(InteractiveScene):
    def construct(self):
        # Test
        quote = TexText(R"""
            ``In holography, nature\\
            is on the inventor's side''
        """, font_size=60, alignment="")
        quote.to_edge(RIGHT)

        self.play(Write(quote), run_time=2, lag_ratio=0.1)
        self.wait()


class ComplexConjugateFact(InteractiveScene):
    def construct(self):
        # Show conjugate
        z = complex(2, 1)
        plane = ComplexPlane(
            (-6, 6), (-4, 4),
            faded_line_ratio=4,
            background_line_style=dict(
                stroke_color=BLUE_D,
                stroke_width=1,
            ),
            faded_line_style=dict(
                stroke_color=BLUE_E,
                stroke_width=1,
                stroke_opacity=0.5,
            ),
        )
        plane.set_height(7)
        plane.to_edge(DOWN, buff=0.25)
        plane.add_coordinate_labels(font_size=16)
        title = Text("Complex plane")
        title.next_to(plane, UP, buff=SMALL_BUFF)

        z_arrow = Arrow(plane.get_origin(), plane.n2p(z), buff=0)
        z_arrow.set_color(YELLOW)
        z_label = Tex(R"z = a + bi")
        z_label.next_to(z_arrow.get_end(), UP)

        z_conj = complex(z.real, -z.imag)
        z_conj_arrow = Arrow(plane.get_origin(), plane.n2p(z_conj), buff=0)
        z_conj_arrow.set_color(TEAL)
        z_conj_label = Tex("z^* = a - bi")
        z_conj_label.next_to(z_conj_arrow.get_end(), DOWN)

        conj_label = Text("Complex conjugate")
        conj_label.next_to(z_conj_label, DOWN, aligned_edge=LEFT)
        conj_label.set_backstroke(BLACK, 5)

        self.add(plane, title)
        self.add(z_arrow, z_label)
        self.wait()
        self.play(Write(conj_label, stroke_color=WHITE))
        self.play(
            TransformFromCopy(z_arrow, z_conj_arrow, path_arc=-PI / 2),
            TransformMatchingStrings(z_label.copy(), z_conj_label, path_arc=-PI / 2),
            run_time=2
        )
        self.wait()

        # Show product
        product_arrow = Arrow(plane.get_origin(), plane.n2p(z * z_conj), buff=0)
        product_arrow.set_color(WHITE)
        product_label = Tex(R"z \cdot z^* = |z|^2")
        product_label.next_to(product_arrow.get_end(), UP)
        product_label.shift(0.75 * RIGHT)

        self.play(
            TransformFromCopy(z_label[0], product_label[R"z \cdot"][0]),
            TransformFromCopy(z_conj_label[:2], product_label["z^*"][0]),
            TransformFromCopy(z_arrow, product_arrow),
            TransformFromCopy(z_conj_arrow, product_arrow),
        )
        self.play(Write(product_label["= |z|^2"][0]))
        self.wait()


class PrepareComplexAlgebra(InteractiveScene):
    def construct(self):
        # Test
        lines = VGroup(
            Tex(R"R \cdot (1 - \text{Opacity})"),
            Tex(R"R \cdot (1 - c|R + O|^2)"),
            Tex(R"R - c R \cdot |R + O|^2"),
        )
        lines.arrange(DOWN, buff=0.75)
        lines.to_edge(UP)

        label = Text("Wave beyond\nthe film")
        arrow = Vector(LEFT)
        arrow.next_to(lines[0], RIGHT)
        label.next_to(arrow, RIGHT)

        rect = SurroundingRectangle(lines[2][R"R \cdot |R + O|^2"], buff=0.05)
        rect.set_stroke(TEAL, 2)

        self.play(
            FadeIn(label, lag_ratio=0.1),
            GrowArrow(arrow),
            FadeIn(lines[0], LEFT),
        )
        self.wait()
        self.play(
            TransformMatchingStrings(lines[0].copy(), lines[1]),
            run_time=2
        )
        self.wait()
        self.play(
            TransformMatchingStrings(lines[1].copy(), lines[2]),
            run_time=2
        )
        self.wait()
        self.play(
            ShowCreation(rect),
            lines[:2].animate.set_opacity(0.5),
            lines[2][R"R - c"].animate.set_opacity(0.5),
        )
        self.wait()


class ComplexAlgebra(InteractiveScene):
    def construct(self):
        # Add first lines
        lines = VGroup(
            Tex(R"R \cdot |R + O|^2"),
            Tex(R"R \cdot (R + O)(R^* + O^*)"),
            Tex(R"R \cdot R \cdot R^* + R \cdot R^* \cdot O  + R \cdot R \cdot O^*+ R \cdot O \cdot O^*"),
            Tex(R"\left(|R|^2 + |O|^2 \right) \cdot R + |R|^2 \cdot O + R^2 \cdot O^*"),
        )
        lines[2].scale(0.75)
        lines.arrange(DOWN, buff=LARGE_BUFF)
        lines.to_edge(UP)

        label = Text("Part of the wave\nbeyond the film", font_size=36)
        arrow = Vector(LEFT)
        arrow.next_to(lines[0], RIGHT)
        label.next_to(arrow, RIGHT)
        label.shift_onto_screen()

        self.add(lines[0])
        self.play(
            Write(label),
            GrowArrow(arrow)
        )
        self.wait()
        self.play(LaggedStart(
            TransformFromCopy(lines[0][:-1], lines[1][R"R \cdot (R + O)"][0]),
            TransformFromCopy(lines[0]["|R + O|"][0].copy(), lines[1]["(R^* + O^*)"][0]),
            lag_ratio=0.1,
            run_time=2
        ))
        self.wait()

        # FOIL expansion
        R0 = lines[1]["R"][0]
        R = lines[1]["R"][1]
        O = lines[1]["O"][0]
        Rc = lines[1]["R^*"][0]
        Oc = lines[1]["O^*"][0]
        l1_groups = [
            VGroup(R0, R, Rc),
            VGroup(R0, O, Rc),
            VGroup(R0, R, Oc),
            VGroup(R0, O, Oc),
        ]
        l2_groups = [
            lines[2][substr]
            for substr in [
                R"R \cdot R \cdot R^*",
                R"R \cdot R^* \cdot O",
                R"R \cdot R \cdot O^*",
                R"R \cdot O \cdot O^*",
            ]
        ]
        plusses = lines[2]["+"]
        plusses.add_to_back(VectorizedPoint())

        pre_rects = VGroup(map(self.get_term_rect, l1_groups[0]))
        post_rect = self.get_term_rect(l2_groups[0])
        VGroup(pre_rects, post_rect).set_stroke(width=0, opacity=0)
        self.add(pre_rects, post_rect)

        for l1_group, l2_group, plus in zip(l1_groups, l2_groups, plusses):
            self.play(
                Transform(pre_rects, VGroup(map(self.get_term_rect, l1_group))),
                Transform(post_rect, self.get_term_rect(l2_group)),
                FadeIn(l2_group),
                FadeIn(plus),
            )
            self.wait(0.5)

        self.add(lines[2])
        self.play(
            FadeOut(pre_rects),
            FadeOut(post_rect),
        )
        self.wait()

        # Highlight conjugate pairs
        pair_rects = VGroup(
            self.get_term_rect(lines[2][R"R \cdot R^*"][0]),
            self.get_term_rect(lines[2][R"R \cdot R^*"][1]),
            self.get_term_rect(lines[2][R"O \cdot O^*"]),
        )
        pair_rects.set_stroke(RED, 2)

        real_label = Text("Real numbers")
        real_label.next_to(lines[2], DOWN, buff=1.5)
        real_label.set_color(RED)
        arrows = VGroup(
            Arrow(real_label, rect.get_bottom())
            for rect in pair_rects
        )
        arrows.set_color(RED)

        self.play(ShowCreation(pair_rects, lag_ratio=0.1))
        self.play(
            FadeIn(real_label, lag_ratio=0.1),
            LaggedStartMap(GrowArrow, arrows)
        )
        self.wait()
        self.play(
            FadeOut(real_label),
            FadeOut(arrows),
            FadeOut(pair_rects),
        )
        self.wait()

        # Organize into the last line
        lines[3].next_to(lines[2], DOWN, buff=1.5)
        lines[3].set_opacity(1)
        l3_groups = [
            lines[3][R"\left(|R|^2 + |O|^2 \right) \cdot R"],
            lines[3][R"|R|^2 \cdot O"],
            lines[3][R"R^2 \cdot O^*"],
        ]
        l2_plusses = lines[2]["+"]
        l3_plusses = lines[3]["+"]

        l2_rects = VGroup(map(self.get_term_rect, l2_groups))
        l3_rects = VGroup(map(self.get_term_rect, l3_groups))
        l3_rects[2].match_height(l3_rects, stretch=True, about_edge=UP)

        box1_lines = VGroup(
            self.connecting_line(l2_rects[0], l3_rects[0]),
            self.connecting_line(l2_rects[3], l3_rects[0]),
        )
        box2_line = self.connecting_line(l2_rects[1], l3_rects[1])
        box3_line = self.connecting_line(l2_rects[2], l3_rects[2])

        real_brace1 = Brace(lines[3][R"\left(|R|^2 + |O|^2 \right)"], DOWN)
        real_brace2 = Brace(lines[3][R"|R|^2"][1], DOWN)
        real_label = real_brace1.get_text("Some real number", font_size=36)

        fade_opacity = 0.5

        self.play(
            ShowCreation(box1_lines, lag_ratio=0.5),
            FadeIn(l2_rects[0]),
            FadeIn(l2_rects[3]),
            FadeTransform(l2_groups[0].copy(), l3_groups[0], time_span=(0.5, 2)),
            FadeTransform(l2_groups[3].copy(), l3_groups[0], time_span=(1, 2)),
            FadeIn(l3_rects[0], time_span=(1, 2)),
            l2_groups[1].animate.set_opacity(fade_opacity),
            l2_groups[2].animate.set_opacity(fade_opacity),
            l2_plusses.animate.set_opacity(fade_opacity),
            run_time=2
        )
        self.wait()
        self.play(
            GrowFromCenter(real_brace1),
            FadeIn(real_label, shift=0.25 * DOWN)
        )
        self.wait()
        self.play(
            box1_lines.animate.set_stroke(WHITE, 1, 0.5),
            VGroup(l2_rects[0], l2_rects[3], l3_rects[0]).animate.set_stroke(GREY, 1, 0.5),
            l2_groups[1].animate.set_opacity(1),
            l2_groups[0].animate.set_opacity(fade_opacity),
            l2_groups[3].animate.set_opacity(fade_opacity),
            l3_groups[0].animate.set_opacity(fade_opacity),
            FadeOut(real_brace1),
            real_label.animate.set_opacity(fade_opacity),
        )
        self.play(
            FadeIn(l2_rects[1]),
            ShowCreation(box2_line),
            FadeIn(l3_rects[1], time_span=(0.5, 1.5)),
            TransformMatchingShapes(l2_groups[1].copy(), l3_groups[1], run_time=1),
            FadeIn(l3_plusses[1]),
        )
        self.wait()
        self.play(
            GrowFromCenter(real_brace2),
            real_label.animate.next_to(real_brace2, DOWN).set_opacity(1)
        )
        self.wait()
        self.play(
            box2_line.animate.set_stroke(WHITE, 1, 0.5),
            VGroup(l2_rects[1], l3_rects[1]).animate.set_stroke(GREY, 1, 0.5),
            l2_groups[2].animate.set_opacity(1),
            l2_groups[1].animate.set_opacity(fade_opacity),
            l3_groups[1].animate.set_opacity(fade_opacity),
            l3_plusses[1].animate.set_opacity(fade_opacity),
            FadeOut(real_brace2),
            FadeOut(real_label),
        )
        self.play(
            FadeTransform(l2_groups[2].copy(), l3_groups[2], run_time=1.5),
            ShowCreation(box3_line, run_time=1.5),
            FadeIn(l2_rects[1]),
            FadeIn(l3_rects[2], time_span=(0.5, 1.5)),
            FadeIn(l3_plusses[2])
        )
        self.wait()

        # Bring back
        self.play(
            FadeOut(VGroup(box1_lines, box2_line, box3_line)),
            FadeOut(l2_rects),
            l3_rects.animate.set_stroke(TEAL, 1, 1),
            lines[2].animate.set_opacity(0.5),
            lines[3].animate.set_opacity(1),
        )
        self.wait()

    def get_term_rect(self, term):
        rect = SurroundingRectangle(term)
        rect.round_corners()
        rect.set_stroke(TEAL, 2)
        return rect

    def connecting_line(self, high_box, low_box):
        return CubicBezier(
            high_box.get_bottom(),
            high_box.get_bottom() + 1.0 * DOWN,
            low_box.get_top() + 1.0 * UP,
            low_box.get_top(),
        ).set_stroke(WHITE, 2)


class PlaneAfterFilm(InteractiveScene):
    def construct(self):
        # Test
        image = ImageMobject("FilmFromBehind")
        image.set_height(FRAME_HEIGHT)
        image.set_height(FRAME_HEIGHT)
        image.fix_in_frame()
        # self.add(image)

        rect = Polygon(
            (2.56, -0.12, 0.0),
            (5.69, -2.51, 0.0),
            (6.25, 1.31, 0.0),
            (2.72, 2.62, 0.0),
        )
        rect.set_fill(BLACK, 0.75)
        rect.set_stroke(BLUE, 10)
        rect.fix_in_frame()
        words = Text("True on this\n2D Plane")

        self.set_floor_plane("xz")
        self.frame.reorient(73, -24, 0, (-1.53, -0.68, 4.14), 9.19)

        self.play(
            DrawBorderThenFill(rect),
            FadeIn(words),
        )
        self.wait()


class LookingAtScreen(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.change_students("confused", "pondering", "maybe", look_at=self.screen),
            self.teacher.change("well")
        )
        self.wait(5)


class EndScreen(PatreonEndScreen):
    pass


# Old stubs

class DoubleSlitSupplementaryGraphs(InteractiveScene):
    def construct(self):
        # Setup all three axes, with labels

        # Show constructive interference

        # Show destructive interference
        ...


class DistApproximations(InteractiveScene):
    def construct(self):
        # Show sqrt(L^2 + x^2) approx L + x/(2L) approx L
        pass


class DiffractionEquation(InteractiveScene):
    def construct(self):
        # Add equation
        equation = Tex(R"{d} \cdot \sin(\theta) = \lambda", font_size=60)
        equation.set_backstroke(BLACK)
        arrow = Vector(DOWN, thickness=4)
        arrow.set_color(BLUE)

        d, theta, lam = syms = [equation[s][0] for s in [R"{d}", R"\theta", R"\lambda"]]
        colors = [BLUE, YELLOW, TEAL]

        arrow.next_to(d, UP, LARGE_BUFF)
        arrow.set_fill(opacity=0)

        self.add(equation)

        for sym, color in zip(syms, colors):
            self.play(
                FlashAround(sym, color=color, time_span=(0.25, 1.25)),
                sym.animate.set_fill(color),
                arrow.animate.next_to(sym, UP).set_fill(color, 1),
            )
            self.wait()
        self.play(FadeOut(arrow))
