from manim_imports_ext import *
from _2023.clt.main import *


# class MysteryConstant(InteractiveScene):
#     def construct(self):
#         eq_C = Tex("= C", t2c={"C": RED}, font_size=24)
#         eq_C.to_edge(UP).shift(3 * LEFT)

#         words = Text("Mystery Constant", font_size=56)
#         words.set_color(RED)
#         words.next_to(ORIGIN, RIGHT)
#         words.to_edge(UP)

#         arrow = Arrow(words, eq_C, stroke_color=RED)

#         self.play(
#             Write(words),
#             FadeIn(eq_C, LEFT),
#             GrowArrow(arrow)
#         )
#         self.wait()


class WignerZoom(InteractiveScene):
    def construct(self):
        im1 = ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/clt/artwork/wigner-speech/wigner-speech-transition-3.jpg")
        im1.set_height(FRAME_HEIGHT)
        self.add(im1)

        # Test
        self.play(
            self.frame.animate.scale(0.4419, about_edge=UL),
            run_time=4,
        )
        self.wait()


class WignerReverseZoom(InteractiveScene):
    def construct(self):
        # Test
        im1 = ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/clt/artwork/wigner-speech/wigner-speech-transition-4.jpg")
        im1.set_height(FRAME_HEIGHT)
        self.add(im1)

        frame = self.frame
        frame.save_state()
        frame.scale(0.4419, about_edge=UL)

        words = TexText(R"Now you're pushing \\ the joke too far...", font_size=20)
        words.set_backstroke(BLACK, 6)
        words.next_to(frame.get_corner(UR), DL, buff=0.2)
        words.shift(0.7 * LEFT)

        self.play(FadeIn(words, lag_ratio=0.1, run_time=2))
        self.play(
            Restore(frame),
            words.animate.set_stroke(width=3),
            run_time=7,
        )
        self.wait()


class PaperTitle(InteractiveScene):
    def construct(self):
        # Test
        title = Text(R"""
            The Unreasonable Effectiveness
            of Mathematics in the
            Natural Sciences
        """, alignment="LEFT", font_size=60)
        title.to_corner(UL)
        self.play(Write(title, run_time=3, lag_ratio=0.1))
        self.wait()
        part = title["Unreasonable Effectiveness"]
        self.play(
            part.animate.set_color(BLUE),
            FlashUnder(part, time_width=1.5, run_time=2, color=BLUE)
        )
        self.wait()


class StoryWords(InteractiveScene):
    quote = R"""
        ``There is a story about \\
        two friends who were \\
        classmates in high school...
    """

    def construct(self):
        # Test
        quote = TexText(self.quote, font_size=75, alignment=R"")
        quote.to_corner(DL)
        quote.set_backstroke(BLACK, 8)

        self.play(FadeIn(quote, lag_ratio=0.1, run_time=2))
        self.wait()


class ClosingStoryWords(StoryWords):
    quote = R"""
        ...surely the population has \\
        nothing to do with the \\
        circumference of the circle!''
    """


class ClassicProofFrame(VideoWrapper):
    title = "A classic proof, due to Poisson"
    wait_time = 4


class OtherVideos(InteractiveScene):
    def construct(self):
        # Test
        folder = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/clt/why_pi_supplements/images/"
        images = Group(*(
            ImageMobject(os.path.join(folder, name))
            for name in ["VCubingxThumbnail", "BrimathThumbnail", "DrAlters", "KeithConrad"]
        ))
        images.set_height(0.35 * FRAME_HEIGHT)
        images.arrange_in_grid(h_buff=2.0, v_buff=1.0)
        images.to_edge(DOWN)
        names = VGroup(
            Text("vcubingx"),
            Text("BriTheMathGuy"),
            Text("idan-alter.github.io"),
            Text("kconrad.math.uconn.edu"),
        )

        rects = VGroup()
        groups = Group()
        for name, image in zip(names, images):
            name.scale(0.75)
            name.move_to(image.get_top())
            name.shift(0.25 * UP)
            rect = SurroundingRectangle(image, buff=0)
            rects.add(rect)
            groups.add(Group(rect, image, name))

        rects.set_stroke(WHITE, 4)

        self.play(LaggedStartMap(FadeIn, groups, lag_ratio=0.5))
        self.wait()


class CirclesToPopulation(InteractiveScene):
    def construct(self):
        # Show arrow
        circle = Circle(radius=1.5)
        circle.set_stroke(RED, 3)
        circle.set_fill(RED, 0.25)
        circle.to_corner(UL)

        screen = ScreenRectangle(height=5)
        screen.to_corner(DR)
        title = Text("Population statistics")
        title.next_to(screen, UP, SMALL_BUFF)

        arrow = Arrow(circle.get_right() + UP, title, path_arc=-0.5 * PI)
        q_marks = Text("???")
        q_marks.move_to(arrow)

        self.add(circle)
        self.play(
            ShowCreation(arrow),
            FadeIn(title, DR)
        )
        self.wait()
        self.play(Write(q_marks))
        self.wait()


class LastVideoFrame(InteractiveScene):
    def construct(self):
        self.add(FullScreenRectangle().set_fill(GREY_E, 1))
        screen = ScreenRectangle(height=6.5)
        screen.to_edge(DOWN, buff=MED_SMALL_BUFF)
        screen.set_fill(BLACK, 1)
        screen.set_stroke(BLACK, 0)
        screen_outline = screen.copy().set_stroke(BLUE_D, 2)
        self.add(screen)

        # Test
        titles = VGroup(
            Text("Central Limit Theorem"),
            Text("Normal Distribution"),
            Text("Normal Distribution (aka a Gaussian)"),
        )
        for title in titles:
            title.scale(56 / 48)
            title.move_to(midpoint(screen.get_top(), TOP))

        self.add(titles[0])
        self.play(ShowCreation(screen_outline, run_time=3))
        self.wait(2)
        self.play(
            FadeOut(titles[0], UP),
            FadeIn(titles[1], UP),
        )
        self.wait()
        self.play(
            ReplacementTransform(titles[1], titles[2]["Normal Distribution"][0]),
            Write(titles[2]["(aka a Gaussian)"])
        )
        self.wait(3)


class SatisfyThisDisbelief(InteractiveScene):
    def construct(self):
        # Words
        words = Text("""
            What explanation would
            satisfy their disbelief?
        """, alignment="LEFT")
        words.set_backstroke(width=3)
        words.to_corner(UR)
        arrow = Arrow(words.get_left(), 2 * LEFT + 2 * UP)

        pis = VGroup(
            Mortimer(),
            Randolph(color=BLUE_D).flip(),
            Randolph(color=BLUE_E).flip(),
            Randolph(color=BLUE_C).flip(),
        )
        pis.arrange_in_grid()
        pis.set_height(4)
        pis.to_corner(DR)

        self.add(pis)
        pis[0].change_mode("happy")
        self.play(LaggedStart(
            pis[0].change("raise_right_hand"),
            pis[1].change("pondering", look_at=arrow.get_end()),
            pis[2].change("pondering", look_at=arrow.get_end()),
            pis[3].change("pondering", look_at=arrow.get_end()),
            lag_ratio=0.5
        ))
        for x in range(2):
            self.play(Blink(random.choice(pis)))
            self.wait()

        self.play(
            FadeIn(words, lag_ratio=0.1),
            GrowArrow(arrow),
            pis[0].change("raise_left_hand", look_at=words),
            *(pi.animate.look_at(words) for pi in pis[1:])
        )
        for x in range(3):
            self.play(Blink(random.choice(pis)))
            self.wait()
        self.wait()


class ThreeStepPlan(InteractiveScene):
    def construct(self):
        # Steps
        step_titles = VGroup(*(
            Text(f"Step {n}", font_size=48) for n in [1, 2, 3]
        ))
        kw = dict(t2c={"e^{-{x}^2}": BLUE, R"\sqrt{\pi}": TEAL}, color=GREY_A, font_size=36, alignment="")
        step_contents = VGroup(
            TexText(R"Explain why the area under \\$e^{-{x}^2}$ is $\sqrt{\pi}$", **kw),
            TexText(R"Explain where $e^{-{x}^2}$ comes from \\ in the first place", **kw),
            TexText(R"Connect it all to the \\ Central Limit Theorem", **kw),
        )

        steps = VGroup()
        for title, content in zip(step_titles, step_contents):
            content.next_to(title, DOWN, aligned_edge=LEFT)
            steps.add(VGroup(title, content))
        steps.arrange(DOWN, buff=0.75, aligned_edge=LEFT)
        steps.to_corner(UL)

        step_rects = VGroup(*(
            SurroundingRectangle(content)
            for content in step_contents
        ))
        step_rects.set_fill(GREY_E, 1)
        step_rects.set_stroke(RED, 1, 0.5)

        self.add(steps)
        self.add(step_rects)
        self.remove(step_contents)
        self.wait()

        for rect, content in zip(step_rects, step_contents):
            self.add(content, rect)
            self.play(rect.animate.stretch(0, 0, about_edge=RIGHT).set_stroke(opacity=0))
            self.remove(rect)
            self.wait()

        # Step by step
        for step in steps:
            step.save_state()

        for step in steps:
            anims = [Restore(step)]
            for other_step in steps:
                if other_step is not step:
                    anims.append(other_step.animate.restore().scale(0.75, about_edge=LEFT).set_opacity(0.5))
            self.play(*anims)
            self.wait()


class ManyBulgingFunctions(InteractiveScene):
    def construct(self):
        # Test
        axes = Axes((-3, 3), (0, 1), width=3, height=1, axis_config=dict(tick_size=0.05))
        all_axes = axes.get_grid(3, 2, h_buff=LARGE_BUFF, v_buff=1.5)
        all_axes.to_edge(RIGHT)

        ie = math.exp(-1)
        eem1 = math.exp(-math.exp(-1))
        functions = [
            lambda x: (1 / (1 + x**2)),
            lambda x: (1 / (1 + x**4)),
            lambda x: np.exp(-x**2),
            lambda x: np.exp(-x**4),
            lambda x: eem1 * (np.abs(x) + ie)**(-np.abs(x) - ie),
            lambda x: np.sinc(x)**2,
        ]

        func_labels = VGroup(
            Tex(R"\frac{1}{1 + x^2}"),
            Tex(R"\frac{1}{1 + x^4}"),
            Tex(R"e^{-x^2}", font_size=60),
            Tex(R"e^{-x^4}", font_size=60),
            Tex(R"e^{-1/e} \left(\left|x\right|+\frac{1}{e}\right)^{-\left(\left|x\right|+\frac{1}{e}\right)}", font_size=36),
            Tex(R"\left(\frac{\sin(\pi x)}{\pi x} \right)^2"),
        )

        colors = [TEAL_C, TEAL_D, BLUE_C, BLUE_D, RED_C, RED_D]

        plots = VGroup()
        for axes, func, label, color in zip(all_axes, functions, func_labels, colors):
            label.scale(0.5)
            label.move_to(axes.get_corner(UR))
            label.shift_onto_screen()
            graph = axes.get_graph(func)
            graph.set_stroke(color, 3)
            graph.move_to(axes.get_top(), UP)

            plots.add(VGroup(axes, graph, label))

        key_plot = plots[2]
        plots.remove(key_plot)

        self.play(
            FadeIn(key_plot[0]),
            ShowCreation(key_plot[1]),
            Write(key_plot[2]),
        )
        self.wait()
        self.play(LaggedStartMap(FadeIn, plots, lag_ratio=0.5))
        self.wait()
        self.play(plots.animate.fade(0.5))
        self.wait()


class WarningCalculus(InteractiveScene):
    def construct(self):
        # Add warning sign
        warning_sign = Triangle().set_height(2)
        warning_sign.set_stroke(RED, 5)
        warning_sign.set_fill(RED, 0.25)
        warning_sign.round_corners(radius=0.1)
        bang = Text("!")
        bang.set_height(warning_sign.get_height() * 0.7)
        bang.next_to(warning_sign.get_bottom(), UP, SMALL_BUFF)
        bang.match_color(warning_sign)
        warning_sign.add(bang)
        signs = warning_sign.replicate(2)
        signs.arrange(RIGHT)

        warning = VGroup(
            Text("Warning", font_size=120),
            Text("Calculus ahead", font_size=90)
        )
        warning.arrange(DOWN, MED_LARGE_BUFF)
        warning.set_color(RED)
        signs.match_height(warning)
        signs[0].next_to(warning, LEFT)
        signs[1].next_to(warning, RIGHT)
        warning.add(*signs)
        warning.to_edge(UP)

        self.add(warning)

        # Explanation
        text = Text("""
            I mean, it's not that bad, but what follows does assume some familiarity
            with integrals. If you haven't yet learned calculus, feel free to give
            it a go and watch anyway, the visuals hopefully help to make the main idea
            clear, but know that in that case you shouldn't feel bad at all if it
            doesn't make sense!

            If that is you, perhaps I could interest you in watching the series about
            calculus on this channel? I mean, you do you, learn calculus from wherever
            you want. I'm just saying, it's right there if you want it.
        """, alignment="LEFT", font_size=36)
        text.next_to(warning, DOWN, LARGE_BUFF)

        self.play(FadeIn(text, lag_ratio=1 / len(text)))


class IntegralTitleCard(InteractiveScene):
    def construct(self):
        # Test
        curve = FunctionGraph(lambda x: np.exp(-x**2), x_range=(-2.5, 2.5, 0.1))
        curve.set_width(8)
        curve.set_height(3, stretch=True)
        curve.move_to(DOWN)
        curve.set_stroke(BLUE, 3)
        curve.set_fill(BLUE, 0.5)

        # Make it into an equation
        curve.set_height(2)
        parens = Tex("()")
        parens.match_height(curve)
        parens[0].next_to(curve, LEFT, buff=0.2)
        parens[1].next_to(curve, RIGHT, buff=0.2)
        lhs = VGroup(curve, parens)
        two = Integer(2).move_to(lhs.get_corner(UR))
        lhs.add(two)

        equals = Tex("=", font_size=90)
        equals.next_to(lhs, RIGHT, MED_LARGE_BUFF)

        circle = Circle()
        circle.match_style(curve)
        circle.set_height(3)
        circle.next_to(equals, RIGHT, MED_LARGE_BUFF)
        radius = Line(circle.get_center(), circle.get_right())
        radius.set_stroke(WHITE, 2)
        r_label = Integer(1).next_to(radius, UP, SMALL_BUFF)
        rhs = VGroup(circle, radius, r_label)

        equation = VGroup(lhs, equals, rhs)
        equation.move_to(DOWN)

        # Animate
        self.add(curve)
        circle.set_fill(opacity=0)
        self.play(LaggedStart(
            Write(VGroup(parens, two, equals), run_time=1),
            TransformFromCopy(curve.copy().set_fill(opacity=0), circle),
            FadeIn(VGroup(radius, r_label)),
            FadeIn(circle.copy().set_fill(opacity=0.5)),
            lag_ratio=0.5,
            run_time=3
        ))
        # circle.set_fill(opacity=0.5)
        self.add(rhs)
        # self.play(circle.animate.set_fill(opacity=0.5))
        self.wait(2)

        return

        # Alternatively...
        func_label = Tex("e^{-x^2}", font_size=60)
        func_label.next_to(curve.pfp(0.6), UR)
        area_label = Tex(R"\sqrt{\pi}", font_size=120)
        area_label.move_to(curve, DOWN).shift(0.5 * UP)

        self.play(ShowCreation(curve), FadeIn(func_label, lag_ratio=0.1))
        self.play(curve.animate.set_fill(BLUE, 0.5), Write(area_label))
        self.wait()


class Usually(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        self.play(self.change_students("pondering", "pondering", "pondering", look_at=self.screen))
        self.play(self.teacher.says("At least, usually"))
        self.play(self.change_students("erm", "confused", "sassy"))
        self.wait(3)


class SorryWhat(TeacherStudentsScene):
    def construct(self):
        stds = self.students
        morty = self.teacher
        self.remove(self.background)
        self.play(LaggedStart(
            stds[1].says("Sorry, what?!", mode="angry"),
            stds[0].change("sassy", look_at=morty),
            stds[2].change("hesitant", look_at=morty),
            morty.change("tease"),
            lag_ratio=0.2,
        ))
        self.wait(3)


class WhoOrderedMoreDimensions(TeacherStudentsScene):
    def construct(self):
        # Ask
        stds = self.students
        morty = self.teacher
        self.remove(self.background)
        self.play(LaggedStart(
            stds[0].says("Why", mode="raise_left_hand"),
            stds[2].change("hesitant", look_at=morty),
            morty.change("tease"),
            stds[1].says(
                "Who ordered \n another dimension?",
                mode="confused",
                bubble_direction=LEFT,
            ),
            lag_ratio=0.2,
        ))
        self.wait(3)
        self.play(LaggedStart(
            morty.says("Well, watch\nwhat happens", mode="shruggie"),
            stds[0].debubble(mode="hesitant"),
            stds[1].debubble(mode="maybe"),
            stds[2].change("sassy"),
        ))
        self.wait(3)
        self.play(
            FadeOut(VGroup(morty.bubble, morty.bubble.content)),
            morty.says("It never hurts to\ntry similar problems"),
            self.change_students("pondering", "erm", "pondering", look_at=self.screen)
        )
        self.wait(3)


class MysteryConstant(InteractiveScene):
    def construct(self):
        eq_C = Tex("= C", t2c={"C": RED}, font_size=24)
        eq_C.to_edge(UP).shift(3 * LEFT)

        words = Text("Mystery Constant", font_size=56)
        words.set_color(RED)
        words.next_to(ORIGIN, RIGHT)
        words.to_edge(UP)

        arrow = Arrow(words, eq_C, stroke_color=RED)

        self.play(
            Write(words),
            FadeIn(eq_C, LEFT),
            GrowArrow(arrow)
        )
        self.wait()


class YLineFlash(InteractiveScene):
    def construct(self):
        line = Line(7 * RIGHT, 7 * LEFT)
        line.insert_n_curves(50)
        line.set_stroke(YELLOW, 5)
        self.play(VShowPassingFlash(line, run_time=3, time_width=0.7))


class VolumeEqualsPi(InteractiveScene):
    def construct(self):
        text = TexText(R"Volume = $\pi$")
        text[R"\pi"].scale(2, about_edge=LEFT)
        self.add(text)


class OurKind(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        self.play(self.change_students("pondering", "thinking", "pondering", look_at=self.screen))
        self.play(
            self.teacher.says("Where there is\ncircular symmetry,\nour kind thrives", mode="hooray")
        )
        self.play(self.change_students("happy", "thinking", "tease"))
        self.play(self.teacher.change("tease"))
        self.look_at(self.screen)
        self.wait(6)


class DirectlyUseful(InteractiveScene):
    def construct(self):
        self.add(FullScreenRectangle().set_fill(GREY_E, 1))
        rects = ScreenRectangle().replicate(2)
        rects.set_stroke(WHITE, 2)
        rects.set_fill(BLACK, 1)
        rects.arrange(RIGHT, buff=LARGE_BUFF)
        rects.set_width(FRAME_WIDTH - 1)
        rects.to_edge(DOWN, buff=1.5)
        self.add(rects)

        arrow = Arrow(rects[1].get_top(), rects[0].get_top(), path_arc=0.5 * PI)
        words = Text("Directly useful")
        words.next_to(arrow, UP)

        self.play(
            ShowCreation(arrow),
            FadeIn(words, lag_ratio=0.1),
        )
        self.wait()


class CompareThreeIntegrals(InteractiveScene):
    def construct(self):
        # Frames
        self.add(FullScreenRectangle().set_fill(GREY_E, 0.5))
        buff = MED_LARGE_BUFF
        rects = Rectangle(height=5.0, width=(FRAME_WIDTH - 4 * buff) / 3).replicate(3)
        rects.arrange(RIGHT, buff=buff)
        rects.to_edge(DOWN, buff=1.0)
        rects.set_fill(BLACK, 1)
        rects.set_stroke(TEAL, 1)

        # Titles
        kw = dict(
            font_size=48,
            t2c={"C": PINK, "{x}": BLUE, "{y}": YELLOW, "{r}": RED}
        )
        titles = VGroup(
            TexText("Area = $C$", **kw),
            TexText("Volume = $C^2$", **kw),
            TexText(R"Volume = $\pi$", **kw),
        )
        titles[2][R"\pi"].scale(1.5, about_edge=DL)
        integrals = VGroup(
            Tex(R"\int_{-\infty}^\infty e^{-{x}^2}\,d{x} = C", **kw),
            Tex(R"\int_{-\infty}^\infty \int_{-\infty}^\infty e^{-{y}^2} e^{-{x}^2}\,d{x}\,d{y} = C^2", **kw),
            Tex(R"\int_0^\infty 2\pi r \cdot e^{-{r}^2} \,d{r} = \pi", **kw),
        )
        integrals.scale(0.5)

        for title, integral, rect in zip(titles, integrals, rects):
            title.next_to(rect, UP, buff=MED_LARGE_BUFF)
            integral.next_to(rect.get_top(), DOWN)

        # Show first two
        self.add(*rects[:2])
        self.wait()

        for title, integral in zip(titles[:2], integrals[:2]):
            self.play(
                FadeIn(title, 0.5 * UP),
                FadeIn(integral, 0.5 * DOWN)
            )
        self.wait(2)

        # Show third
        self.play(
            FadeIn(rects[2]),
            FadeIn(titles[2], 0.5 * UP),
            FadeIn(integrals[2], 0.5 * DOWN)
        )
        self.wait()

        # Equations
        eq_pi = titles[2][R"= $\pi$"][0].copy()
        eq_pi.generate_target()
        eq_sqrt_pi = Tex(R"= \sqrt{\pi}", **kw)
        for title, rhs in zip(titles[:2], (eq_sqrt_pi, eq_pi.target)):
            title.generate_target()
            rhs.next_to(title, RIGHT)
            rhs.shift((title["="].get_y() - rhs[0].get_y()) * UP)
            VGroup(title.target, rhs).match_x(title)

        self.play(LaggedStart(
            MoveToTarget(eq_pi),
            MoveToTarget(titles[1]),
            lag_ratio=0.35,
        ))
        self.play(LaggedStart(
            TransformMatchingShapes(eq_pi.copy(), eq_sqrt_pi, run_time=1),
            MoveToTarget(titles[0]),
            lag_ratio=0.35,
        ))
        self.wait()
        self.play(FlashAround(eq_sqrt_pi[1:], run_time=2, time_width=1))
        self.wait()


class FeelsLikeATrick(InteractiveScene):
    def construct(self):
        # Test
        words = VGroup(
            Text("This feels \n like a trick"),
            Text("How could you have \n discovered this?"),
        )
        words.to_corner(UL)
        screen = ScreenRectangle(height=5)
        screen.to_corner(DR)

        arrows = VGroup(*(
            Arrow(word.get_right(), screen.get_top(), path_arc=-0.3 * PI)
            for word in words
        ))

        self.play(Write(words[0], run_time=1), ShowCreation(arrows[0]))
        self.wait(2)
        self.play(
            FadeOut(words[0], UP),
            FadeIn(words[1], UP),
            Transform(*arrows),
        )
        self.wait(2)


class HerschelMaxwellTitle(InteractiveScene):
    def construct(self):
        title = Text("""
            The Herschel-Maxwell
            derivation for a Gaussian
        """, font_size=72)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        title.set_backstroke(width=5)
        self.play(Write(title))
        self.wait(3)

        # Transition to Herschel name
        name = Text("John Herschel", font_size=60)
        years = Text("1792 - 1871")
        bio = VGroup(name, years)
        bio.arrange(DOWN)
        bio.to_corner(UL)
        bio.shift(1.0 * RIGHT)

        pre_name = title["Herschel"]
        title.remove(*pre_name)
        self.play(
            FadeOut(title, lag_ratio=0.1),
            ReplacementTransform(pre_name, name["Herschel"]),
            FadeInFromPoint(name["John"], pre_name.get_left()),
        )
        self.play(Write(years))
        self.wait()


class HerschelDescription(InteractiveScene):
    def construct(self):
        # Positions
        positions = VGroup(
            Text("Mathematician"),
            Text("Scientist"),
            Text("Inventor"),
        )
        positions.scale(1.5)
        positions.to_edge(RIGHT, buff=1.5)

        last = VGroup()
        for position in positions:
            self.play(
                FadeIn(position, 0.5 * UP),
                FadeOut(last, 0.5 * UP),
            )
            self.wait(0.5)
            last = position
        self.wait()
        self.play(FadeOut(positions[-1]))

         # Contributions
        contributions = VGroup(
            Text("Chemistry"),
            Text("Astronomy"),
            Text("Photography"),
            Text("Botany"),
        )
        contributions.scale(1.25)
        contributions.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        contributions.move_to(3 * RIGHT)
        self.play(ShowIncreasingSubsets(contributions, rate_func=linear, run_time=3))
        self.play(LaggedStartMap(FadeOut, contributions, lag_ratio=0.5, run_time=3))
        self.wait()

        # Moons
        desc = TexText(R"He named many of \\ Saturn's moons")
        desc.move_to(3.5 * RIGHT).to_edge(UP)
        moon_names = VGroup(
            Text("Mimas"),
            Text("Enceladus"),
            Text("Tethys"),
            Text("Dione"),
            Text("Rhea"),
            Text("Titan"),
            Text("Iapetus"),
        )
        moon_names.set_color(GREY_B)
        moon_names.scale(0.75)
        moon_names.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        moons = Group()
        for name in moon_names:
            image = ImageMobject(name.get_string())
            image.set_height(1.25 * name.get_height())
            image.next_to(name, LEFT, buff=0.2)
            moons.add(Group(image, name))

        moons.next_to(desc, DOWN, MED_LARGE_BUFF)

        self.play(
            FadeIn(desc),
            LaggedStartMap(FadeIn, moons, shift=0.5 * DOWN, lag_ratio=0.5, run_time=3)
        )
        self.wait()


class GaussianQuestion(InteractiveScene):
    def construct(self):
        # Test
        curve = FunctionGraph(lambda x: math.exp(-x**2), x_range=(-3, 3, 0.1))
        curve.set_width(6)
        curve.set_height(2, stretch=True)
        curve.to_edge(RIGHT).shift(UP)
        curve.set_stroke(BLUE, 3)

        func = Tex(R"{1 \over \sqrt{2 \pi} } e^{-x^2 / 2}", font_size=40)
        func.move_to(curve.get_bottom())

        question = Text("Where does this\ncome from?")
        question.next_to(func, DOWN, MED_LARGE_BUFF)

        year = Text("(1850)", font_size=40)
        year.next_to(question, DOWN, MED_LARGE_BUFF)
        year.set_fill(GREY_B)

        self.play(
            ShowCreation(curve),
            FadeIn(func),
            Write(question)
        )
        self.play(FadeIn(year, DOWN))
        self.wait(2)


class WhatWouldBeNice(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.says("You know what\nwould be nice?"),
            self.change_students("pondering", "happy", "tease", look_at=self.screen)
        )
        self.wait(4)


class HowDoYouApproachThat(TeacherStudentsScene):
    def construct(self):
        stds = self.students
        self.remove(self.background)
        self.play(
            stds[0].says("How do you \n approach that?", mode="raise_left_hand"),
            stds[1].change("maybe", look_at=3 * UP),
            stds[2].change("erm", look_at=3 * UP),
            self.teacher.change("tease")
        )
        self.wait()
        self.play(
            self.teacher.change("raise_right_hand"),
            self.change_students("confused", "confused", "pondering", look_at=3 * UP)
        )
        self.wait(3)


class ImplicationPIP(InteractiveScene):
    def construct(self):
        self.add(FullScreenRectangle().set_fill(GREY_E, 1))
        screens = ScreenRectangle().replicate(2)
        screens.set_height(0.45 * FRAME_HEIGHT)
        screens.arrange(RIGHT, buff=0.75)
        screens.set_stroke(WHITE, 2)
        screens.set_fill(BLACK, 1)
        screens.move_to(0.5 * DOWN)

        arrow = Arrow(screens[0].get_top(), screens[1].get_top(), path_arc=-0.4 * PI)
        words = Text("These necessarily imply")
        words.next_to(arrow, UP)

        self.add(screens)
        self.wait()
        self.play(
            ShowCreation(arrow),
            FadeIn(words, lag_ratio=0.1)
        )
        self.wait()


class MaxwellsEquations(InteractiveScene):
    def construct(self):
        kw = dict(
            t2c={R"\mathbf{E}": RED, R"\mathbf{B}": TEAL}
        )
        equations = VGroup(
            Tex(R"\nabla \cdot \mathbf{E}=4 \pi \rho", **kw),
            Tex(R"\nabla \cdot \mathbf{B}=0", **kw),
            Tex(R"\nabla \times \mathbf{E}=-\frac{1}{c} \frac{\partial \mathbf{B}}{\partial t}", **kw),
            Tex(R"\nabla \times \mathbf{B}=\frac{1}{c}\left(4 \pi \mathbf{J}+\frac{\partial \mathbf{E}}{\partial t}\right)", **kw),
        )
        equations.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        equations.set_height(6.5)
        equations.to_edge(LEFT)
        equations.set_backstroke(BLACK, 3)

        self.play(LaggedStartMap(FadeIn, equations, shift=0.5 * DOWN, lag_ratio=0.5))
        self.wait()


class StatisticalMechanicsIn3d(InteractiveScene):
    def construct(self):
        # Test
        n_balls = 200
        self.camera.light_source.move_to([0, -5, 5])
        frame = self.frame
        frame.reorient(20, 70)
        frame.set_height(2)
        frame.move_to([0.5, 0.5, 0.5])

        balls = DotCloud(np.random.random((n_balls, 3)))
        balls.set_radius(0.015)
        balls.make_3d()
        balls.set_shading(0.25, 0.5, 1)

        cube = VCube(side_length=1.0)
        cube.move_to(ORIGIN, [-1, -1, -1])
        cube.set_stroke(BLUE, 2)
        cube.set_fill(opacity=0)
        self.add(cube)

        velocities = np.random.normal(0, 3, (n_balls, 3))

        def update(balls, dt):
            points = balls.get_points()
            new_points = np.clip(points + 0.1 * velocities * dt, 0, 1)
            velocities[new_points <= 0] *= -1
            velocities[new_points >= 1] *= -1
            balls.set_points(new_points)

            return balls

        balls.add_updater(update)

        self.add(balls)
        self.play(frame.animate.reorient(-20), run_time=20)


class ThreeDExpression(InteractiveScene):
    def construct(self):
        expr = Tex(
            R"{1 \over \sigma^3 (2\pi)^{3/2}} e^{-(x^2 + y^2 + z^2) / 2 \sigma^2}",
            t2c={R"\sigma": RED},
            font_size=48
        )
        expr[R"1 \over \sigma^3 (2\pi)^{3/2}}"].scale(0.7, about_edge=RIGHT)
        expr.to_corner(UL)
        self.add(expr)


class DefiningProperty(InteractiveScene):
    def construct(self):
        # Setup
        prop = VGroup(
            Text("1. Radial symmetry", font_size=36),
            Text("2. Independence of\neach coordinate", font_size=36),
        )
        prop.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        implies = Tex(R"\Downarrow", font_size=72)
        form = Tex(R"e^{-c r^2}", t2c={"c": RED}, font_size=72)

        derivation = VGroup(prop, implies, form)
        derivation.arrange(DOWN, buff=MED_LARGE_BUFF)
        form.shift(0.25 * UP)
        derivation.to_edge(LEFT, buff=1.0).shift(0.5 * DOWN)
        derivation_box = SurroundingRectangle(derivation, buff=0.5)
        derivation_box.set_stroke(YELLOW, 2)
        derivation_box.set_fill(YELLOW, 0.05)

        title = Text("Herschel-Maxwell\nDerivation")
        title.next_to(derivation_box, UP, MED_LARGE_BUFF)

        self.add(derivation_box)
        self.add(derivation)
        self.add(title)
        self.wait()

        # If we view this
        words = Text(
            "Suppose we take this\nto define a Gaussian...",
            t2s={"define": ITALIC},
            t2c={"define": TEAL},
            alignment="LEFT"
        )
        words.match_width(derivation_box)
        words.next_to(derivation_box, UP)

        self.play(LaggedStart(
            FadeIn(words, 0.5 * UP),
            FadeOut(title, UP),
            lag_ratio=0.2,
        ))
        self.wait()
        self.play(
            FlashAround(prop[0], color=TEAL, time_width=1, run_time=2)
        )
        self.wait()


class ReactToDefiningProperty(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.remove(self.background)
        self.play(
            morty.change("raise_right_hand"),
            self.change_students("sassy", "hesitant", "erm", look_at=self.screen),
        )
        self.play(
            self.change_students("erm", "sassy", "hesitant", look_at=self.screen),
        )
        self.wait(2)

        # Ask
        self.play(LaggedStart(
            stds[1].says("But that assumes \n we're in 2D", mode="angry"),
            stds[0].change("sassy"),
            stds[2].animate.look_at(morty),
            morty.change("guilty"),
            lag_ratio=0.2
        ))
        self.wait(3)
        self.play(LaggedStart(
            stds[1].debubble(mode="sassy"),
            stds[2].says("What about\nthe Central Limit Theorem?"),
            stds[0].change("hesitant"),
            morty.change("hesitant"),
        ))
        self.wait(3)
        self.play(morty.change("tease", self.screen))
        self.wait()
        self.play(
            stds[0].change("pondering", self.screen),
            stds[1].change("pondering", self.screen),
        )
        self.wait(3)


class NextVideo(InteractiveScene):
    def construct(self):
        self.add(FullScreenRectangle().set_fill(GREY_E, 0.75))

        # Thumbnails
        images = Group(
            ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2022/convolutions/discrete/images/EquationThumbnail.png"),
            ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/clt/main/images/Thumbnail.png"),
            ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/clt/integral/images/ProofSummary.jpg"),
        )
        images.set_height(3)
        titles = VGroup(
            Text("Convolutions"),
            Text("Central Limit Theorem"),
            TexText(R"Why $\sqrt{\pi}$?"),
        )
        thumbnails = Group()
        for image, title in zip(images, titles):
            rect = SurroundingRectangle(image, buff=0).set_stroke(WHITE, 3)
            title.next_to(image, UP)
            title.set_max_width(image.get_width())
            thumbnails.add(Group(rect, image, title))

        thumbnails.arrange(DOWN, buff=LARGE_BUFF)
        thumbnails.set_height(7)
        thumbnails.to_corner(UL)
        thumbnails.to_edge(LEFT, buff=1.0)

        # Video 1
        morty = Mortimer(height=1).flip()
        morty.change_mode("tease")
        morty.next_to(images[0], RIGHT, MED_LARGE_BUFF, aligned_edge=DOWN)

        bubble = SpeechBubble(width=3, height=1)
        bubble.set_stroke(width=1)
        bubble.move_to(morty.get_corner(UR), DL).shift(SMALL_BUFF * LEFT)
        bubble.add_content(Text("In the next video..."))

        self.add(thumbnails[0])
        self.play(
            VFadeIn(morty),
            morty.change("speaking"),
            FadeIn(bubble),
            Write(bubble.content, run_time=1)
        )
        self.play(
            morty.animate.look_at(thumbnails[1]),
            FadeIn(thumbnails[1], DOWN)
        )

        # Video 2
        morty.generate_target()
        morty.target.align_to(thumbnails[1], DOWN)
        morty.target.change_mode("maybe").look_at(thumbnails[2])
        words = bubble.content
        words2 = Text("Or rather, the next one...", t2s={"next": ITALIC})
        bubble2 = morty.target.get_bubble(words2, bubble_type=SpeechBubble, width=4, height=1)
        bubble2.set_stroke(WHITE, 1)

        ghost1 = VGroup(morty, bubble, words).copy()
        to_fade = VGroup(ghost1)

        self.play(
            MoveToTarget(morty),
            Transform(bubble, bubble2),
            Transform(words, words2),
            ghost1.animate.set_opacity(0.25),
        )
        words.become(words2)
        self.play(Blink(morty), FadeIn(thumbnails[2], DOWN))

        # Video 3
        morty.generate_target()
        morty.target.align_to(thumbnails[2], DOWN)
        morty.target.change_mode("guilty")
        words3 = Text("But for real this time")
        bubble3 = morty.target.get_bubble(words3, bubble_type=SpeechBubble, width=4, height=1)
        bubble3.set_stroke(WHITE, 1)

        ghost2 = VGroup(morty, bubble, words).copy()
        to_fade.add(ghost2)

        self.play(
            MoveToTarget(morty),
            Transform(bubble, bubble3),
            Transform(words, words3),
            ghost2.animate.set_opacity(0.25)
        )
        words.become(words3)
        self.play(Blink(morty))
        self.wait(2)

        # Right screen
        screen = ScreenRectangle()
        screen.set_height(4)
        screen.set_stroke(WHITE, 2)
        screen.set_fill(BLACK, 1)
        screen.to_edge(RIGHT)

        arrows = VGroup(*(
            Arrow(thumbnail.get_right(), screen.get_left() + vect, buff=0.3)
            for thumbnail, vect in zip(thumbnails, [UP, ORIGIN, DOWN])
        ))

        self.play(
            LaggedStartMap(GrowArrow, arrows),
            FadeIn(screen),
            LaggedStartMap(FadeOut, VGroup(*to_fade, morty, bubble, words))
        )
        self.wait()


class NDimensionalBallGrid(InteractiveScene):
    def construct(self):
        # Setup grid
        N_terms = 7
        grid = Square().get_grid(3, N_terms + 1, buff=0, group_by_rows=True)
        grid.set_stroke(WHITE, 1)
        for row in grid:
            row[0].stretch(1.5, 0, about_edge=RIGHT)
        grid.set_width(FRAME_WIDTH - 1)
        grid.center().to_edge(DOWN, buff=1.0)
        self.add(grid)

        # Content
        kw = dict(t2c={"r": BLUE})
        numbers = VGroup(
            *(Integer(n) for n in range(1, 6)),
            Tex(R"\cdots"),
            Text(R"N"),
        )
        interiors = VGroup(
            Tex(R"2r", **kw),
            Tex(R"\pi r^2", **kw),
            Tex(R"\frac{4}{3} \pi r^3", **kw),
            Tex(R"\frac{1}{2} \pi^2 r^4", **kw),
            Tex(R"\frac{8}{15} \pi^2 r^5", **kw),
            Tex(R"\cdots"),
            Tex(R"\frac{\pi^{n / 2}}{(n / 2)!} r^n", **kw),
        )
        exteriors = VGroup(
            Tex(R"2", **kw),
            Tex(R"2 \pi r", **kw),
            Tex(R"4 \pi r^2", **kw),
            Tex(R"2 \pi^2 r^3", **kw),
            Tex(R"\frac{8}{3} \pi^2 r^4", **kw),
            Tex(R"\cdots"),
            Tex(R"n \frac{\pi^{n / 2}}{(n / 2)!} r^{n - 1}", **kw),
        )
        for group, row in zip([numbers, interiors, exteriors], grid):
            for elem, square in zip(group, row[1:]):
                elem.set_max_width(0.8 * square.get_width())
                elem.move_to(square)

        row_titles = VGroup(
            Text("Dimension"),
            TexText(R"Volume of \\ $n$-ball", font_size=30),
            TexText(R"Volume of \\ $n$-ball boundary", font_size=30),
        )
        for title, row in zip(row_titles, grid):
            title.set_max_width(0.8 * row[0].get_width())
            title.move_to(row[0])

        self.add(row_titles, numbers)

        # Shapes
        line = Line()
        line.set_stroke(BLUE_A, 3)
        one_ball = VGroup(
            line,
            Dot(line.get_start()),
            Dot(line.get_end())
        )
        one_ball[1:].set_color(BLUE)

        circle = Circle()
        circle.set_stroke(BLUE, 2)
        circle.set_fill(BLUE, 0.5)

        sphere = Sphere(color=BLUE)
        mesh = SurfaceMesh(sphere)
        mesh.set_stroke(WHITE, 0.5, 0.5)
        ball = Group(sphere, mesh)
        ball.rotate(70 * DEGREES, LEFT)

        shapes = Group(
            one_ball, circle, ball,
            Randolph("shruggie"),
            Tex(R"\dots"),
            *VectorizedPoint().replicate(2),
        )
        for shape, square in zip(shapes, grid[0][1:]):
            shape.set_max_width(0.7 * square.get_width())
            shape.move_to(square.get_top())
            shape.shift(0.6 * shape.get_width() * UP)
        shapes[4].match_y(shapes[0])

        # Animate in
        for shape, interior, exterior in zip(shapes, interiors, exteriors):
            self.play(FadeIn(shape), FadeIn(interior))
            self.play(TransformMatchingTex(interior.copy(), exterior, run_time=1))
        self.wait()


class FinalExercise(InteractiveScene):
    def construct(self):
        # Test
        kw = dict(
            t2c={
                "I_n": YELLOW,
                "I_1": YELLOW,
                "I_2": YELLOW,
                "I_{n - 2}": YELLOW,
                "C_2": TEAL,
                "C_3": TEAL,
                "C_4": TEAL,
                "C_5": TEAL,
                "C_6": TEAL,
                "C_n": TEAL,
                "C_{n - 2}": TEAL,
                "{r}": BLUE,
            },
            font_size=30,
            alignment=""
        )
        parts = VGroup(
            TexText(R"""
                \textbf{Part 1} \\

                Define $I_n$ to be the integral below. For example, we showed in this \\
                video that $I_1 = \sqrt{\pi}$ and $I_2 = \pi$.
                $$
                I_n =
                \underbrace{\int_{-\infty}^\infty \cdots \int_{-\infty}^\infty}_{n \text{ times}}
                e^{-(x_1^2 + \cdots + x_n^2)} dx_1 \cdots dx_n
                $$
                Explain why $I_n = \pi^{n / 2}$. Specifically, factor the expression then integrate \\
                along each of the $n$ coordinate axes.
            """, **kw),
            TexText(R"""
                \textbf{Part 2} \\

                We all learn that the circumference of a circle is $2\pi {r}$, and the surface area \\
                of a sphere is $4\pi {r}^2$. In general, the boundary of an $n$-dimensional ball has \\
                an $(n-1)$-dimensional volume of $C_n {r}^{n - 1}$ for some constant $C_n$. For\\
                example, $C_2 = 2\pi$ and $C_3 = 4 \pi$. Our goal is to figure out this value $C_n$ \\
                for each dimension $n$.\\

                Explain why the integral from the previous problem can also be \\
                interpreted as
                $$
                I_n = \int_0^\infty C_n {r}^{n - 1} e^{-{r}^2} dr
                $$
            """, **kw),
            TexText(R"""
                \textbf{Part 3} \\

                Using integration by parts, show that
                \begin{align*}
                I_n = \int_0^\infty C_n {r}^{n - 1} e^{-{r}^2} dr 
                &=\frac{n - 2}{2} C_n \int_0^\infty {r}^{n - 3} e^{-{r}^2} dr \\
                &= \frac{n - 2}{2} C_n \cdot { I_{n - 2} \over C_{n - 2}}
                \end{align*}
                \quad \\
                (Hint, let $u = {r}^{n - 2}$ and $dv = {r} e^{-{r}^2} dr$)
            """, **kw),
            TexText(R"""
                \textbf{Part 4} \\

                Combine parts 1 and 3 to deduce $C_n = \frac{2\pi}{n - 2} C_{n - 2}$ for $n > 2$. Use this \\
                to write the values of $C_3$, $C_4$, $C_5$, and as many more as you'd like. \\

                Considering the recurrence relation above, together with the assumption \\
                that $(-1/2)! = \sqrt{\pi}$ (a story for another day), can you explain why \\
                the formula below for the $(n-1)$-dimensional boundary of a $n$-dimensional \\
                ball makes sense? \\
                $$ n {\pi^{n/2} \over (n/2)!} {r}^{n - 1}$$
            """, **kw),
        )

        rects = ScreenRectangle().get_grid(2, 2, buff=0)
        rects.set_height(7)
        rects.set_width(FRAME_WIDTH, stretch=True)
        rects.to_edge(DOWN, buff=0)
        rects.set_stroke(WHITE, 2)
        self.add(rects)

        title = Text("Challenge for the ambitious viewer", font_size=60)
        title.set_color(RED)
        title.to_edge(UP, MED_SMALL_BUFF)
        self.add(title)

        parts.scale(0.7)
        for part, rect in zip(parts, rects):
            part.move_to(rect, UL).shift(DR * SMALL_BUFF)

        self.add(parts)

        self.play(LaggedStartMap(FadeIn, parts, lag_ratio=0.7))
        self.wait()


class EndScreen(PatreonEndScreen):
    scroll_time = 30


class Thanks(InteractiveScene):
    def construct(self):
        message = Text("Thank you \n patrons!", font_size=72)
        morty = Mortimer().flip()
        morty.next_to(ORIGIN, LEFT)
        message.next_to(morty, RIGHT)

        self.play(
            morty.change("gracious"),
            Write(message)
        )
        self.play(Blink(morty))
        self.wait()
        self.play(Blink(morty))
        self.wait(2)
