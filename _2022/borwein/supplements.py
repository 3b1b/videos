from manim_imports_ext import *
from _2022.borwein.main import *


# Pi creature scenes and quick supplements

class UniverseIsMessingWithYou(TeacherStudentsScene):
    def construct(self):
        pass


class ExpressAnger(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        self.students[1].change_mode("maybe")
        self.play(
            self.teacher.change("raise_left_hand", look_at=3 * RIGHT + UP),
            self.change_students("pondering", "confused", "thinking", look_at=3 * RIGHT + UP)
        )
        self.wait(2)
        self.play(
            self.teacher.change("tease", look_at=3 * RIGHT + UP),
            self.change_students("angry", "erm", "pleading", look_at=3 * RIGHT + UP)
        )
        self.wait(5)


class LooksCanBeDeceiving(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.says("Looks can be\ndeceiving", mode="tease"),
            self.change_students("angry", "erm", "pleading", look_at=self.screen)
        )
        self.wait(3)


class AstuteAmongYou(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.play(
            stds[2].says(
                TexText("What about\\\\$x=0$?"),
                bubble_direction=LEFT,
                mode="raise_right_hand"
            ),
            stds[0].change("hesitant"),
            stds[1].change("pondering"),
            morty.change("happy", stds[2].eyes),
        )
        self.wait(2)
        self.play(
            stds[0].says("The squeeze theorem!", mode="hooray"),
            RemovePiCreatureBubble(stds[2])
        )
        self.look_at(stds[0].get_corner(UR))
        self.wait(3)

        # Hold Up
        point = stds.get_top() + 2 * UP
        self.play(
            stds[0].debubble(mode="raise_right_hand", look_at=point),
            stds[1].change("thinking", point),
            stds[2].change("erm", point),
            morty.change("tease", point)
        )
        self.wait(5)


class AreaToSignedArea(InteractiveScene):
    def construct(self):
        # Words
        equals = Tex("=")
        equals.to_edge(UP, buff=1.0).shift(2.5 * LEFT)
        area = Text("Area")
        area.next_to(equals, RIGHT)
        arrow = Arrow(area.get_bottom(), UP)

        signed_area = Text("Signed area", t2s={"Signed": ITALIC})
        signed_area.next_to(equals, RIGHT)

        blue_minus_red = TexText(
            "= (Blue area) - (Red area)",
            tex_to_color_map={"(Blue area)": BLUE, "(Red area)": RED}
        )
        blue_minus_red.next_to(signed_area, RIGHT)

        self.play(FadeIn(equals), Write(area), ShowCreation(arrow))
        self.wait()
        self.play(TransformMatchingShapes(area, signed_area))
        self.wait()
        self.play(FadeIn(blue_minus_red[:2]))
        self.wait()
        self.play(FadeIn(blue_minus_red[2:]))
        self.wait()


class HoldOffUntilEnd(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.says("I'll mention the\ntrick at the end"),
            self.change_students("confused", "hesitant", "maybe", look_at=self.screen)
        )
        self.wait()
        self.teacher_says("But first...", target_mode="tease")
        self.wait()


class WhatsGoingOn(InteractiveScene):
    def construct(self):
        randy = Randolph(height=2)
        randy.to_corner(DL).look(RIGHT)
        self.play(randy.says("What on earth\nis going on?", mode="angry"))
        self.play(randy.animate.look(DR))
        self.play(Blink(randy))
        self.wait(2)
        self.play(randy.change("confused"))
        self.play(Blink(randy))
        self.wait(2)


class SeemsUnrelated(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer(height=2.0)
        morty.to_edge(DOWN)
        self.play(morty.says("That should modify\nthe area, right?", mode="maybe"))
        self.play(Blink(morty))
        self.wait()


class WhatsThePoint(TeacherStudentsScene):
    def construct(self):
        stds = self.students
        self.play(
            stds[2].says(
                TexText(R"So it computes $\pi$ \\ what's the point?"),
                bubble_direction=LEFT,
                mode="angry",
            ),
            stds[1].change("sassy", self.screen),
            stds[0].change("pondering", self.screen),
            self.teacher.change("guilty")
        )
        self.wait(2)
        self.play(
            self.teacher.says("Look what \n happens next", mode="tease", bubble_direction=RIGHT),
            stds[2].debubble(),
            stds[1].change("erm"),
            stds[0].change("thinking"),
        )
        self.wait(2)


class BillionBillionBillion(InteractiveScene):
    def construct(self):
        fraction = Tex(get_fifteenth_frac_tex())[0]
        # self.add(fraction)

        for x in (4, 16, 28):
            part = VGroup(*fraction[x:x + 11], *fraction[x + 40:x + 40 + 11])
            vh = VHighlight(part, color_bounds=(YELLOW, BLACK), max_stroke_addition=8)
            self.add(vh, part)
            self.wait(0.5)
            self.remove(vh, part)
        self.wait()


class MovingAverageFrames(InteractiveScene):
    def construct(self):
        # Frames
        self.add(FullScreenRectangle())
        screens = ScreenRectangle(height=4).replicate(2)
        screens.arrange(RIGHT, buff=1.0)
        screens.set_width(FRAME_WIDTH - 1)
        screens.set_stroke(WHITE, 1)
        screens.set_fill(BLACK, 1)
        screens.move_to(0.5 * DOWN)
        self.add(screens)

        # Titles
        titles = VGroup(
            Text("Step 1:\nAn easier analogy", alignment="LEFT"),
            Text("Step 2:\nReveal it's more than an analogy", alignment="LEFT"),
        )
        titles.match_width(screens[0])
        for title, screen in zip(titles, screens):
            title.next_to(screen, UP, aligned_edge=LEFT)

        # Animations
        n = len("Step1:")
        for i in range(2):
            self.play(Write(titles[i][:n]))
            self.play(FadeIn(titles[i][n:], 0.5 * DOWN))
            self.wait()


class FirstInASequence(InteractiveScene):
    def construct(self):
        morty = Mortimer(height=2).flip()
        morty.to_corner(DR)
        self.play(morty.says("First in a\nsequence"))
        self.play(Blink(morty))
        self.wait(2)


class XInMovingAverageGraphs(MovingAverages):
    def construct(self):
        axes1, axes2 = self.get_axes(), self.get_axes()
        axes1.to_edge(UP)
        axes2.to_edge(DOWN).shift(0.15 * DOWN)

        # self.add(axes1, axes2)  # Remove

        # Labels
        x = -0.61
        y = 0.1
        xp1 = axes1.c2p(x, 0)
        x_label = Tex(f"x = {x}", tex_to_color_map={"x": YELLOW})
        x_label.next_to(xp1, DOWN, buff=LARGE_BUFF)
        arrow = Arrow(x_label[0].get_top(), xp1, buff=0.2).set_color(YELLOW)

        h_line = DashedLine(axes2.c2p(0, y), axes2.c2p(x, y))
        h_line.set_stroke(RED, 2)

        us = axes1.x_axis.unit_size
        brace = Brace(Line(xp1 + us * LEFT / 6, xp1 + us * RIGHT / 6), DOWN)
        brace_text = brace.get_text("Average value\nin this window", font_size=36)

        self.play(
            Write(x_label),
            ShowCreation(arrow)
        )
        self.wait()
        self.play(ShowCreation(h_line))
        self.wait()
        self.play(
            FadeOut(x_label),
            FadeOut(arrow),
            GrowFromCenter(brace),
            FadeIn(brace_text, 0.5 * DOWN)
        )
        self.wait()


class ThisConstant(InteractiveScene):
    def construct(self):
        words1 = Text("This constant\nhere")
        words1.to_edge(RIGHT)
        arrow = Arrow(words1.get_bottom() + 0.1 * DOWN, words1.get_bottom() + DOWN + 1.5 * LEFT)
        VGroup(words1, arrow).set_color(YELLOW)

        same = Text("Same").match_style(words1)
        same.move_to(words1.select_part("This"), DR)

        self.play(Write(words1), ShowCreation(arrow), run_time=1)
        self.wait()
        self.play(
            FadeIn(same, 0.5 * UP),
            FadeOut(words1.select_part("This"), 0.5 * UP),
        )


class HeavyMachinery(TeacherStudentsScene):
    def construct(self):
        stds = self.students
        morty = self.teacher
        self.play(
            stds[2].says("Why on earth are\nthose two related?", bubble_direction=LEFT, look_at=self.screen),
            morty.change("tease"),
            stds[0].change("pleading", self.screen),
            stds[1].change("confused", self.screen),
        )
        self.wait(2)
        self.play(self.change_students("maybe", "confused", "hesitant"))
        self.wait(2)

        # Heavy machinery
        kw = dict(font_size=60)
        ft = Text("Fourier Transforms", **kw)
        conv = TexText("Convolution", "s", **kw)
        conv_thm = TexText("The ", "Convolution", R"\\Theorem", **kw)
        for text in ft, conv, conv_thm:
            text.move_to(self.hold_up_spot, DOWN)

        self.play(
            morty.change("raise_right_hand"),
            FadeIn(ft, UP),
            stds[2].debubble(mode="guilty", look_at=ft),
            stds[0].change("erm", look_at=ft),
            stds[1].change("sad", look_at=ft),
        )
        self.wait()
        self.play(
            FadeIn(conv, UP),
            ft.animate.next_to(conv, UP, MED_LARGE_BUFF)
        )
        self.wait(2)
        self.play(morty.change("tease"))
        self.wait()

        # Don't want to assume
        ft.save_state()
        conv.save_state()
        self.play(
            morty.says("First, a high\nlevel overview", bubble_direction=RIGHT),
            self.change_students("skeptical", "erm", "pondering"),
            VGroup(ft, conv).animate.scale(0.55).to_corner(UR),
        )
        self.wait(3)

        self.play(
            morty.debubble(mode="raise_right_hand"),
            conv.animate.restore(),
            self.change_students("pondering", "thinking", "thinking", look_at=self.screen),
        )
        self.wait(4)

        # Mention theorem
        self.play(
            TransformMatchingTex(conv, conv_thm),
            morty.change("hooray", conv_thm),
        )
        self.play(self.change_students("thinking", "hesitant", "pondering", look_at=conv_thm))
        self.wait(3)


class EngineersSinc(InteractiveScene):
    def construct(self):
        # Blah
        sinc = Tex(R"{\text{sin}(\pi x) \over \pi x}", tex_to_color_map={R"\pi": TEAL})
        sinc.shift(UP)
        rect = SurroundingRectangle(sinc, buff=SMALL_BUFF)
        rect.set_stroke(YELLOW, 2)
        rect.round_corners()
        rect.insert_n_curves(20)

        label = Text("sinc\n(to an engineer)")
        label.select_part("sinc").scale(1.5, about_edge=DOWN)
        label.next_to(rect, DOWN, MED_LARGE_BUFF)
        label.set_color(GREY_A)

        # self.add(sinc)
        self.play(FadeIn(label, DOWN))
        self.wait()
        self.play(ShowCreation(rect))
        self.wait()


class ThinkAboutMovingAverages(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        self.play(
            morty.change("raise_right_hand", self.screen),
            self.change_students("pondering", "thinking", "happy", look_at=self.screen)
        )
        self.wait(3)
        self.play(
            morty.says("Think about\nmoving averages"),
        )
        self.play(self.change_students(*3 * ["pondering"]))
        self.wait(3)


class ConceptAndNotationFrames(InteractiveScene):
    def construct(self):
        # Screens
        self.add(FullScreenRectangle())
        screens = Rectangle(height=3, width=4).replicate(2)
        screens.arrange(RIGHT, buff=0.25)
        screens.set_width(FRAME_WIDTH - 0.5)
        screens.set_stroke(WHITE, 1)
        screens.set_fill(BLACK, 1)
        screens.move_to(0.5 * DOWN)
        self.add(screens)

        # Titles
        titles = VGroup(
            Text("Moving average fact"),
            Text("In our new notation"),
            Text("How does this..."),
            Text("...relate to this"),
            # Text("These computations..."),
            # Text("...are (in a sense)\nthe same as these"),
        )
        for i, title in enumerate(titles):
            title.next_to(screens[i % 2], UP)
        titles[1].align_to(titles[0], UP)

        # Animations
        for i in range(2):
            self.play(FadeIn(titles[i]))
            self.wait()
        for i in range(2, 4):
            self.play(FadeIn(titles[i], 0.5 * UP), FadeOut(titles[i - 2], 0.5 * UP))
            self.wait()


class WhatsThat(InteractiveScene):
    def construct(self):
        randy = Randolph(height=2)
        self.play(randy.says("...meaning?", mode="maybe", look_at=DR))
        self.play(Blink(randy))
        self.wait()


class SubPiComment(InteractiveScene):
    def construct(self):
        morty = Mortimer(height=2).flip()
        morty.to_corner(DL)

        self.play(
            morty.says(Tex(R"x \rightarrow \pi x", tex_to_color_map={R"\pi": TEAL}))
        )
        self.play(morty.animte.look(RIGHT))
        for x in range(2):
            self.play(Blink(morty))
            self.play(morty.animate.look(DR + x * DOWN))
            self.wait(2)


class HowDoYouCompute(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.student_says(
            "But how do you\ncompute this\nFourier Transform?",
            mode="angry",
            run_time=2
        )
        self.play(
            morty.change("guilty"),
            stds[0].change("confused", self.screen),
            stds[1].change("maybe", self.screen),
        )
        self.wait(2)
        self.play(
            stds[2].debubble(),
            morty.says("We'll get there"),
        )
        self.play(self.change_students("erm", "hesitant", "hesitant"))
        self.wait(2)


class TranslateToFourierLand(InteractiveScene):
    def construct(self):
        # Setup right side
        right_rect = FullScreenRectangle()
        right_rect.set_fill(TEAL, 0.2)
        right_rect.set_stroke(width=0)
        right_rect.stretch(0.5, 0, about_edge=RIGHT)
        ft_label = Text("Fourier Land", font_size=60)
        ft_label.next_to(right_rect.get_top(), DOWN, MED_SMALL_BUFF)
        ft_label.set_backstroke(width=5)
        ft_label.set_fill(TEAL)
        self.add(right_rect, ft_label)

        # Break up a wave?
        left_rect = right_rect.copy().to_edge(LEFT, buff=0)
        axes_kw = dict(width=FRAME_WIDTH - 2, height=3)
        left_axes = VGroup(*(
            Axes((0, 12, 1), (-2, 2), **axes_kw).scale(0.5)
            for x in range(3)
        ))
        left_axes.arrange(DOWN, buff=0.7)
        left_axes.move_to(left_rect).to_edge(DOWN, buff=SMALL_BUFF)

        left_graphs = VGroup(
            left_axes[0].get_graph(lambda x: math.sin(3 * x)),
            left_axes[1].get_graph(lambda x: math.sin(4 * x)),
            left_axes[2].get_graph(lambda x: math.sin(3 * x) + math.sin(4 * x)),
        )
        left_graphs.set_submobject_colors_by_gradient(BLUE, YELLOW, GREEN)

        left_symbols = VGroup(Tex("+"), Tex("=").rotate(PI / 2)).scale(1.5)
        left_symbols[0].move_to(left_axes[0:2])
        left_symbols[1].move_to(left_axes[1:3])

        # Peaks in Fourier land
        right_axes = VGroup(*(
            Axes((0, 5), (0, 3), **axes_kw).scale(0.5)
            for x in range(3)
        ))
        for a1, a2 in zip(left_axes, right_axes):
            a2.match_y(a1).match_x(right_rect)

        def bell(x, n):
            return 3 * np.exp(-40 * (x - n)**2)

        right_graphs = VGroup(
            right_axes[0].get_graph(lambda x: bell(x, 3)),
            right_axes[1].get_graph(lambda x: bell(x, 4)),
            right_axes[2].get_graph(lambda x: bell(x, 3) + bell(x, 4)),
        )
        right_graphs.match_style(left_graphs)
        right_symbols = left_symbols.copy().match_x(right_rect).shift(0.5 * DOWN)

        self.play(
            LaggedStartMap(FadeIn, left_axes, lag_ratio=0.7),
            LaggedStartMap(ShowCreation, left_graphs, lag_ratio=0.7),
            LaggedStartMap(Write, left_symbols, lag_ratio=0.7),
            run_time=3
        )
        self.wait()
        self.play(LaggedStart(
            TransformFromCopy(left_axes, right_axes),
            TransformFromCopy(left_graphs, right_graphs),
            TransformFromCopy(left_symbols, right_symbols),
        ))
        self.wait(3)

        # Other properties
        to_fade = VGroup(
            left_axes, left_graphs, left_symbols,
            right_axes, right_graphs, right_symbols,
        )

        property_pairs = [
            VGroup(
                Tex(R"\int_{-\infty}^\infty f(t) dt"),
                Tex(R"\hat f(0)"),
            ),
            VGroup(
                Tex(R"f(t) \cdot g(t)"),
                # Tex(R"\hat f(\omega) * \hat g(\omega)", tex_to_color_map={R"\omega": YELLOW}),
                Tex(R"\int_{-\infty}^\infty \hat f(\xi) \hat g(\omega - \xi) d\xi", tex_to_color_map={R"\omega": YELLOW}),
            ),
            VGroup(
                Tex(R"\sum_{n = -\infty}^\infty f(n)"),
                Tex(R"\sum_{k = -\infty}^\infty \hat f(k)"),
            ),
        ]

        for pair in property_pairs:
            pair.scale(1.5)
            m1, m2 = pair
            m1.move_to(left_rect)
            m2.move_to(right_rect)

            self.play(FadeOut(to_fade), FadeIn(pair))
            self.wait(3)
            to_fade = pair


class KeyFactFrame(VideoWrapper):
    title = "Key fact"


class TranslatedByFourier(InteractiveScene):
    def construct(self):
        # Two expressions
        # left_expr = MTex(R"\int_{-\infty}^\infty f(t) \cdot g(t) dt")
        # right_expr = MTex(
        #     R"\Big[ \mathcal{F}[f(t)] * \mathcal{F}[g(t)] \Big](0)",
        #     tex_to_color_map={R"\mathcal{F}": TEAL}
        # )
        left_expr = Text("Multiplication\nand\nIntegration")
        right_expr = Text("Moving averages\nand\nEvaluating at 0")
        left_expr.move_to(2 * UP).set_x(-FRAME_WIDTH / 4)
        right_expr.match_y(left_expr).set_x(FRAME_WIDTH / 4)

        equals = Tex("=").move_to(midpoint(left_expr.get_right(), right_expr.get_left()))
        arrow = Vector(2 * RIGHT, color=TEAL)
        arrow.move_to(equals)
        arrow_label = Text("Translate to\nFourier land", font_size=24)
        arrow_label.next_to(arrow, UP)
        arrow_label.match_color(arrow)

        # brace = Brace(right_expr[:-3], UP, buff=SMALL_BUFF)
        # brace_text = TexText("To be explained,\\\\think ``moving average''", font_size=36)
        # brace_text.next_to(brace, UP, SMALL_BUFF)

        self.play(Write(left_expr))
        self.wait(2)
        self.play(LaggedStart(
            # Write(equals),
            GrowArrow(arrow),
            FadeIn(arrow_label, lag_ratio=0.1),
            FadeTransform(left_expr.copy(), right_expr, path_arc=-PI / 5),
            lag_ratio=0.5
        ))
        self.wait(2)
        # self.play(GrowFromCenter(brace), FadeIn(brace_text, 0.5 * UP))
        self.wait()


class UnsatisfyingEnd(TeacherStudentsScene):
    def construct(self):
        # Ask
        self.remove(self.background)
        morty = self.teacher
        stds = self.students

        self.play(
            stds[0].says("But you haven't explained\nwhy any of that is true?", mode="angry"),
            stds[1].change("hesitant"),
            stds[2].change("pleading"),
            morty.change("guilty"),
        )
        stds[0].bubble.insert_n_curves(20)
        self.wait(3)
        self.play(
            morty.says("Patience.", mode="tease"),
            stds[0].debubble(),
            stds[2].change("hesitant")
        )
        self.wait(2)


class EndScreen(PatreonEndScreen):
    pass
