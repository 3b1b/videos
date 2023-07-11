from manim_imports_ext import *
from _2023.convolutions2.continuous import *


class IntroWords(InteractiveScene):
    def construct(self):
        # Test
        title1 = Text("Last video", font_size=72)
        title1.to_edge(UP)
        title2 = Text("Today: An important example", font_size=72)
        title2.move_to(title1)
        VGroup(title1, title2).set_backstroke(width=3)

        self.play(Write(title1, run_time=1))
        self.wait()
        self.play(
            FadeOut(title1, 0.5 * UP),
            FadeIn(title2, 0.5 * UP),
        )
        self.wait()


class NewIntroWords(InteractiveScene):
    def construct(self):
        kw = dict(font_size=66)
        words = VGroup(
            Text("Last chapter", **kw),
            Text("Convolution between two Gaussians", **kw),
            Text("Central Limit Theorem", **kw),
            Text("Today: A satisfying visual argument", **kw),
        )
        words.to_edge(UP, buff=MED_SMALL_BUFF)

        last_word = VMobject()
        for word in words:
            self.play(
                FadeOut(last_word, 0.5 * UP),
                FadeIn(word, 0.5 * UP),
            )
            self.wait()
            last_word = word


class MultipleBellishCurves(InteractiveScene):
    def construct(self):
        # Axes
        line_style = dict(stroke_color=GREY_B, stroke_width=1)
        faded_line_style = dict(stroke_opacity=0.25, **line_style)
        all_axes = VGroup(*(
            NumberPlane(
                (-3, 3), (0, 1, 0.5),
                height=1.5, width=4,
                background_line_style=line_style,
                faded_line_style=faded_line_style,
            )
            for _ in range(3)
        ))
        all_axes.arrange(DOWN, buff=LARGE_BUFF)
        all_axes.to_edge(LEFT)

        # Graphs
        def pseudo_bell(x):
            A = np.abs(x) + np.exp(-1)
            return np.exp(-np.exp(-1)) * A**(-A)

        graphs = VGroup(
            all_axes[0].get_graph(lambda x: np.exp(-x**2)),
            all_axes[1].get_graph(lambda x: 1 / (1 + x**2)),
            all_axes[2].get_graph(pseudo_bell),
        )
        labels = VGroup(
            Tex("e^{-x^2}"),
            Tex(R"\frac{1}{1 + x^2}"),
            Tex(
                R"e^{-1 / e}\left(|x|+\frac{1}{e}\right)^{-\left(|x|+\frac{1}{e}\right)}",
                font_size=40
            ),
        )

        plots = VGroup()
        colors = color_gradient([YELLOW, RED], 3)
        for axes, label, graph, color in zip(all_axes, labels, graphs, colors):
            label.next_to(axes, RIGHT)
            graph.set_stroke(color, 3)
            plots.add(VGroup(axes, graph))

        # Show initial graph
        plot = plots[0]
        plot.save_state()
        plot.center()
        plot.set_height(4)
        label = labels[0]
        label.save_state()
        label.set_height(1.25)
        label.next_to(plot.get_corner(UR), DL)
        words = Text("Normal Distribution (aka Gaussian)", font_size=60)
        words.next_to(plot, UP, MED_LARGE_BUFF)
        words.save_state()
        normal = words["Normal Distribution"]
        gaussian = words["(aka Gaussian)"]
        gaussian.set_opacity(0)
        normal.set_x(0)
        gaussian.set_x(0)

        self.add(plot)
        self.add(words)
        graph_copy = plot[1].copy()
        self.play(UpdateFromAlphaFunc(
            plot[1], lambda m, a: m.pointwise_become_partial(
                graph_copy, 0.5 - 0.5 * a, 0.5 + 0.5 * a
            ),
            run_time=2,
        ))
        self.play(words.animate.restore())
        self.wait()
        self.play(
            Write(label),
            VShowPassingFlash(
                graph_copy.set_stroke(YELLOW, width=10),
                time_width=2,
                run_time=3,
            ),
        )
        self.wait()

        # Ask why
        question = Text("Why this function?", font_size=60)
        arrow = Vector(LEFT)
        arrow.next_to(label.saved_state, RIGHT)
        question.next_to(arrow, RIGHT)
        self.play(
            plot.animate.restore(),
            label.animate.restore(),
            words.animate.match_width(plot.saved_state).next_to(plot.saved_state, UP, SMALL_BUFF),
            GrowArrow(arrow),
            FadeIn(question, lag_ratio=0.1, shift=0.2 * LEFT),
        )
        self.wait()

        last_plot = plot
        last_label = label
        for plot, label in zip(plots[1:], labels[1:]):
            self.play(
                TransformFromCopy(last_plot, plot),
                TransformMatchingTex(last_label.copy(), label, run_time=1),
            )
            self.wait()
            last_plot = plot
            last_label = label
        self.wait()

        # Highlight first function
        l0 = labels[0]
        l0.generate_target()
        l0.target.set_height(1.0, about_edge=DL)
        l0.target.shift(0.1 * RIGHT)
        self.play(LaggedStart(
            MoveToTarget(l0),
            VGroup(arrow, question).animate.next_to(l0.target, RIGHT, aligned_edge=DOWN),
            labels[1:].animate.fade(0.5),
            plots[1:].animate.fade(0.5),
            FlashAround(l0.target, time_width=1.5),
        ))
        self.wait()


class LastFewVideos(InteractiveScene):
    def construct(self):
        self.add(FullScreenRectangle())

        # Images
        root = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/"
        images = Group(*(
            ImageMobject(os.path.join(root, ext))
            for ext in [
                "2022/convolutions/discrete/images/EquationThumbnail.png",
                "2023/clt/main/images/Thumbnail.png",
                "2023/clt/Thumbnail.jpg",
                "2023/convolutions2/Thumbnail/Thumbnail2.png",
            ]
        ))
        titles = VGroup(*(
            Text("Convolutions (discrete)"),
            Text("Central limit theorem"),
            TexText(R"Why $\pi$ is in a Gaussian"),
            Text("Convolutions (continuous)"),
        ))
        titles.scale(1.4)
        thumbnails = Group()
        for image, title in zip(images, titles):
            rect = SurroundingRectangle(image, buff=0)
            rect.set_stroke(WHITE, 3)
            title.next_to(image, UP, buff=MED_LARGE_BUFF)
            thumbnails.add(Group(rect, image, title))

        thumbnails.arrange_in_grid(buff=thumbnails[0].get_width() * 0.2)
        thumbnails.set_height(FRAME_HEIGHT - 1)

        self.play(LaggedStartMap(FadeIn, thumbnails, shift=UP, lag_ratio=0.7, run_time=4))
        self.wait()


class AddingCopiesOfAVariable(InteractiveScene):
    def construct(self):
        # Test
        expr = Tex(R"X_1 + X_2 + \cdots + X_N \text{ is approximately Gaussian}")
        expr.set_fill(GREY_A)
        expr_lhs = expr[R"X_1 + X_2 + \cdots + X_N"][0]
        expr.to_edge(UP)
        expr_lhs.save_state()
        expr_lhs.set_x(0)

        self.play(FadeIn(expr_lhs, lag_ratio=0.5, run_time=2))
        self.wait()
        self.play(
            Restore(expr_lhs),
            Write(expr[len(expr_lhs):], run_time=1)
        )
        self.wait(3)

        # Limit
        expr2 = Tex(R"X_1 + X_2 + \cdots + X_N \longrightarrow \text{Gaussian}")
        expr2.match_style(expr)
        expr2.move_to(expr)
        lim = Tex(R"N \to \infty", font_size=24)
        lim.next_to(expr2[R"\longrightarrow"], UP, buff=0.1)
        lim.set_color(YELLOW)

        self.play(
            TransformMatchingTex(expr, expr2),
            FadeIn(lim, 0.25 * UP, time_span=(1, 2))
        )
        self.wait()


class WhyGaussian(InteractiveScene):
    def construct(self):
        question = TexText("What makes $e^{-x^2}$ special?", font_size=60)
        question.to_edge(UP, buff=LARGE_BUFF)
        self.play(Write(question))
        self.wait()


class AskAboutConvolution(InteractiveScene):
    def construct(self):
        text = TexText(
            "Convolution between $e^{-x^2}$ and $e^{-y^2}$",
            t2c={"x": BLUE, "y": YELLOW},
            font_size=60
        )
        text.to_edge(UP, buff=MED_LARGE_BUFF)
        text.set_backstroke(width=2)
        self.add(text)


class PreviewExplicitCalculation(InteractiveScene):
    def construct(self):
        # Title
        goal = Text("Goal: Compute a convolution between two Gaussian functions")
        goal.set_width(FRAME_WIDTH - 1)
        goal.to_edge(UP)
        conv_word = goal["convolution"]
        gauss_word = goal["Gaussian"]

        self.add(goal)

        # Convolution
        conv_color = BLUE
        tex_kw = dict(
            t2c={"{f}": BLUE, "{g}": TEAL, R"\sigma_1": RED, R"\sigma_2": RED_B},
            font_size=42,
        )
        conv_eq = Tex(R"[{f} * {g}](s) = \int_{-\infty}^\infty {f}(x){g}(s - x)dx", **tex_kw)
        conv_eq.next_to(goal, DOWN, buff=1.5)
        conv_eq.to_edge(LEFT)
        conv_arrow = Arrow(conv_word.get_bottom(), conv_eq.get_top() + SMALL_BUFF * UP)
        conv_arrow.set_color(conv_color)

        self.play(LaggedStart(
            FlashAround(conv_word, time_width=1.5, run_time=2.0, color=conv_color),
            conv_word.animate.set_color(conv_color),
            GrowArrow(conv_arrow),
            FadeTransform(conv_word.copy(), conv_eq),
            lag_ratio=0.2,
        ))
        self.wait()

        # Gaussian
        gauss_color = RED
        gaussian1 = Tex(
            R"f(x) = {1 \over \sigma_1 \sqrt{2 \pi}} e^{-x^2 / 2 \sigma_1^2}",
            **tex_kw
        )
        gaussian2 = Tex(
            R"g(y) = {1 \over \sigma_2 \sqrt{2 \pi}} e^{-y^2 / 2 \sigma_2^2}",
            **tex_kw
        )
        gaussian1.match_y(conv_eq)
        gaussian1.to_edge(RIGHT)
        gaussian2.next_to(gaussian1, DOWN, LARGE_BUFF)
        gauss_arrow = Arrow(gauss_word.get_bottom(), gaussian1.get_top() + SMALL_BUFF * UP)
        gauss_arrow.set_color(gauss_color)
        f_rect = SurroundingRectangle(conv_eq["{f}(x)"], buff=0.05)
        g_rect = SurroundingRectangle(conv_eq["{g}(s - x)"], buff=0.05)
        f_rect.set_stroke(RED, 5)
        g_rect.set_stroke(RED, 5)

        self.play(
            ShowCreation(f_rect),
            FadeTransform(gauss_word.copy(), gaussian1),
            GrowArrow(gauss_arrow),
            gauss_word.animate.set_color(gauss_color),
        )
        self.wait()
        self.play(
            ReplacementTransform(f_rect, g_rect),
            FadeIn(gaussian2, DOWN),
        )
        self.wait()
        self.play(FadeOut(g_rect))
        self.wait()

        # Combine
        full_expr = Tex(R"""
            \int_{-\infty}^\infty
            \frac{1}{2\pi \sigma_1 \sigma_2}
            e^{-x^2 / 2\sigma_1^2} e^{-(s-x)^2 / 2\sigma_2^2} \,dx
        """, **tex_kw)
        full_expr.next_to(conv_eq, DOWN, buff=2.0, aligned_edge=LEFT)
        full_expr_rect = SurroundingRectangle(full_expr)
        full_expr_rect.set_stroke(RED_E, 2)
        arrow_kw = dict(stroke_width=2, stroke_color=RED_E)
        arrows = VGroup(*(
            Arrow(conv_eq, full_expr_rect, **arrow_kw),
            Arrow(gaussian1.get_left(), full_expr_rect, **arrow_kw),
            Arrow(gaussian2.get_left(), full_expr_rect.get_right(), **arrow_kw),
        ))

        self.play(
            LaggedStart(*(
                TransformMatchingShapes(conv_eq[9:13].copy(), full_expr[R"\int_{-\infty}^\infty"][0]),
                TransformMatchingShapes(
                    VGroup(*gaussian1[5:13], gaussian2[5:13]).copy(),
                    full_expr[R"\frac{1}{2\pi \sigma_1 \sigma_2}"][0]
                ),
                TransformMatchingShapes(gaussian1[13:].copy(), full_expr[R"e^{-x^2 / 2\sigma_1^2}"][0]),
                TransformMatchingShapes(gaussian2[13:].copy(), full_expr[R"e^{-(s-x)^2 / 2\sigma_2^2}"][0]),
                TransformMatchingShapes(conv_eq[-2:].copy(), full_expr[R"dx"]),
            ), run_time=3, lag_ratio=0.1),
            LaggedStartMap(GrowArrow, arrows),
        )
        self.play(ShowCreation(full_expr_rect))
        self.add(full_expr)
        self.wait(3)


class NothingWrongWithThat(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        self.remove(self.background)

        self.play(
            morty.says("There's nothing\nwrong with that!", mode="hooray"),
            self.change_students("awe", "horrified", "confused", look_at=self.screen),
        )
        self.wait(3)
        self.play(
            morty.debubble(mode="tease"),
            self.change_students("pondering", "plain", "well")
        )
        self.wait()
        self.play(
            morty.change("raise_right_hand", look_at=3 * UP),
            self.change_students("well", "pondering", "tease", look_at=3 * UR)
        )
        self.wait(10)


class SimpleBellRHS(InteractiveScene):
    def construct(self):
        self.add(Tex("= e^{-x^2}"))


class SimpleBellRHS2(InteractiveScene):
    def construct(self):
        self.add(Tex("= e^{-(s - x)^2}"))


class ConvolutionMeaning(InteractiveScene):
    def construct(self):
        # x + y = s
        kw = dict(t2c={"{s}": YELLOW})
        words = VGroup(
            Tex(R"[f * g]({s})", font_size=60, **kw),
            Tex(R"\longrightarrow", **kw),
            Tex(R"\text{How likely is it that } x + y = {s} \, ?", **kw),
        )
        words.arrange(RIGHT, buff=0.5)
        words.to_edge(UP)

        words[0].save_state()
        words[0].set_x(0)
        self.play(FadeIn(words[0], 0.5 * UP))
        self.wait()
        self.play(
            Restore(words[0]),
            Write(words[1]),
            FadeIn(words[2], RIGHT),
        )
        self.wait()

        # Mention sqrt(2)
        new_rhs = Tex(R"(\text{This area}) / \sqrt{2}", font_size=60)
        new_rhs[R"\text{This area}"].set_color(TEAL)
        new_rhs.next_to(words[1], RIGHT)
        new_rhs.set_opacity(0)

        self.play(
            VGroup(*words[:2], new_rhs).animate.set_x(0).set_opacity(1),
            FadeOut(words[2], RIGHT),
        )
        self.wait()


class RotationalSymmetryAnnotations(InteractiveScene):
    def construct(self):
        # Add equation
        kw = dict(t2c={"x": BLUE, "y": YELLOW, "r": RED})
        top_eq = Tex("f(x)g(y) = e^{-x^2} e^{-y^2}", **kw)
        top_eq_lhs = top_eq["f(x)g(y)"][0]
        top_eq_rhs = top_eq["= e^{-x^2} e^{-y^2}"][0]
        top_eq.to_corner(UL)

        self.add(top_eq)

        # Expand equation
        rhs2 = Tex("= e^{-(x^2 + y^2)}", **kw)
        rhs3 = Tex("= e^{-r^2}", **kw)

        rhs2.next_to(top_eq_rhs, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
        rhs3.next_to(rhs2, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        xy_rect = SurroundingRectangle(rhs2["x^2 + y^2"], buff=0.05)
        r_rect = SurroundingRectangle(rhs3["r^2"], buff=0.05)
        rects = VGroup(xy_rect, r_rect)
        rects.set_stroke(RED, 1)

        self.play(
            TransformMatchingShapes(
                top_eq_rhs.copy(), rhs2,
                path_arc=30 * DEGREES,
            )
        )
        self.wait()
        self.play(ShowCreation(xy_rect))
        self.play(
            TransformMatchingTex(
                rhs2.copy(), rhs3,
                key_map={"x^2 + y^2": "r^2"},
                run_time=1
            ),
            TransformFromCopy(xy_rect, r_rect),
        )
        self.play(FadeOut(rects))
        self.wait()


class UniqueCharacterization(InteractiveScene):
    def construct(self):
        # Test
        kw = dict(t2c={"{x}": BLUE, "{y}": YELLOW})
        group = VGroup(
            TexText("Rotational symmetry of $f({x})f({y})$", font_size=36, **kw),
            Tex(R"\Downarrow"),
            Tex(R"f({x}) = Ae^{-c {x}^2}", **kw),
        )
        group.arrange(DOWN)
        group.to_corner(UR)

        self.add(group[0])
        self.wait()
        self.play(
            Write(group[1]),
            FadeInFromPoint(group[2], group[0]["f({x})"].get_center()),
            run_time=1
        )
        self.wait()


class SliceLineAnnotations(InteractiveScene):
    def construct(self):
        # Line equation
        kw = dict(t2c={"s": RED})
        line_eq = Tex("x + y = s", **kw)
        line_eq.to_edge(UP)
        s_term = line_eq["s"][0]
        decimal_rhs = DecimalNumber(-6, edge_to_fix=LEFT)
        decimal_rhs.move_to(s_term, LEFT)
        decimal_rhs.shift(0.025 * UP)
        s_eq = Tex("=").rotate(90 * DEGREES)
        s_eq.next_to(s_term, DOWN, SMALL_BUFF)

        self.add(line_eq, decimal_rhs)
        self.remove(s_term)
        self.play(ChangeDecimalToValue(decimal_rhs, 1), run_time=5 / 1.5)
        self.wait()
        self.play(
            decimal_rhs.animate.next_to(s_eq, DOWN, SMALL_BUFF),
            Write(s_eq),
            FadeIn(s_term, 0.5 * DOWN),
        )
        self.wait()

        # What we want
        words = Text("What we want:")
        words.to_corner(UR)
        conv = Tex("[f * g](s)", **kw)
        conv.next_to(words, DOWN, MED_LARGE_BUFF)
        conv_eq = Tex("=").rotate(90 * DEGREES)
        conv_eq.next_to(conv, DOWN)
        area = Tex(R"(\text{This area}) / \sqrt{2}")
        area[R"\text{This area}"].set_color(TEAL)
        area.next_to(conv_eq, DOWN)

        self.play(
            FadeIn(words, 0.5 * UP),
            FadeIn(conv, 0.5 * DOWN),
        )
        self.play(FlashAround(conv, run_time=2, time_width=1.5))
        self.wait()
        self.play(
            Write(conv_eq),
            FadeIn(area, 0.5 * DOWN),
        )
        for s in [1.5, 0.5, 1.0]:
            self.play(ChangeDecimalToValue(decimal_rhs, s))
            self.wait()

        self.play(
            LaggedStartMap(FadeOut, VGroup(
                words, conv, conv_eq, area, s_eq, decimal_rhs
            )),
            line_eq.animate.to_corner(UR),
        )
        self.wait()


class YIntegralAnnotations(InteractiveScene):
    def construct(self):
        # Setup
        kw = dict(
            t2c={
                "{s}": RED,
                "{y}": YELLOW,
                "{x}": BLUE,
                R"\text{Area}": TEAL
            }
        )
        integrals = VGroup(
            Tex(R"\text{Area} = \int_{\text{-}\infty}^\infty e^{-{x}^2} \cdot e^{-{y}^2} \, d{y}", **kw),
            Tex(R"\text{Area} = \int_{\text{-}\infty}^\infty e^{-({s} / \sqrt{2})^2} \cdot e^{-{y}^2} \, d{y}", **kw),
            Tex(R"\text{Area} = e^{-({s} / \sqrt{2})^2} \int_{\text{-}\infty}^\infty e^{-{y}^2} \, d{y}", **kw),
            Tex(R"\text{Area} = e^{-({s} / \sqrt{2})^2} \sqrt{\pi}", **kw),
        )
        for integral in integrals:
            integral.to_edge(UP)

        x_rect = SurroundingRectangle(integrals[0]["e^{-{x}^2}"], buff=0.05)
        s_rect1 = SurroundingRectangle(integrals[1][R"e^{-({s} / \sqrt{2})^2}"], buff=0.05)
        s_rect2 = SurroundingRectangle(integrals[2][R"e^{-({s} / \sqrt{2})^2}"], buff=0.05)
        y_int = integrals[2][R"\int_{\text{-}\infty}^\infty e^{-{y}^2} \, d{y}"]
        y_int_rect = SurroundingRectangle(y_int, buff=0.05)

        x_rect.set_stroke(BLUE, 2)
        s_rect1.set_stroke(RED, 2)
        s_rect2.set_stroke(RED, 2)
        y_int_rect.set_stroke(YELLOW, 2)
        const_word = Text("Constant!", font_size=36)
        const_word.next_to(x_rect, DOWN, buff=MED_SMALL_BUFF, aligned_edge=LEFT)
        const_word.shift(SMALL_BUFF * RIGHT)

        # Replace x
        self.play(Write(integrals[0]))
        self.wait()
        self.play(
            ShowCreation(x_rect),
            FadeIn(const_word, scale=0.7),
        )
        self.wait()

        self.play(
            TransformMatchingTex(*integrals[:2]),
            ReplacementTransform(x_rect, s_rect1),
            const_word.animate.match_x(s_rect1).set_anim_args(run_time=2),
        )
        self.wait()

        # Factor out
        self.play(
            FadeOut(const_word, 0.5 * DOWN),
            VShowPassingFlash(
                s_rect1.copy().insert_n_curves(100).set_stroke(width=5),
                run_time=2,
            ),
            ShowCreation(s_rect1, run_time=2)
        )
        self.wait()
        s_rect2.set_stroke(opacity=0)
        self.play(
            TransformMatchingTex(*integrals[1:3], path_arc=45 * DEGREES),
            ReplacementTransform(s_rect1, s_rect2, path_arc=45 * DEGREES),
            run_time=2
        )
        self.wait()

        # Emphasize separation
        s_rect2.set_stroke(RED, 2, 1)
        self.play(ShowCreation(s_rect2))
        self.wait()
        self.play(ReplacementTransform(s_rect2, y_int_rect))
        self.wait()

        y_int_group = VGroup(y_int_rect, y_int.copy())
        y_int.set_opacity(0)
        int_eq = Tex("=").rotate(90 * DEGREES)
        pi_term = integrals[3][R"\sqrt{\pi}"]
        int_eq.next_to(pi_term, DOWN, SMALL_BUFF)

        self.play(
            TransformMatchingTex(*integrals[2:4]),
            y_int_group.animate.scale(0.5).next_to(int_eq, DOWN, SMALL_BUFF),
            Write(int_eq),
        )
        self.wait()
        self.play(LaggedStartMap(FadeOut, VGroup(int_eq, *y_int_group)))
        self.wait()

        # Area expression
        area_eq = Tex(R"\text{Area} = e^{-{s}^2 / 2} \sqrt{\pi}", **kw)
        area_eq.to_edge(UP)
        area_eq.save_state()
        twos = area_eq["2"]
        twos_copies = twos.copy()
        twos[0].become(twos_copies[1])
        twos[1].become(twos_copies[0])

        self.play(
            TransformMatchingTex(integrals[3], area_eq)
        )
        area_eq.restore()
        self.wait()

        # From area to convolution
        conv_eqs = VGroup(
            Tex(R"[f * g]({s}) = \text{Area} / \sqrt{2}", **kw),
            Tex(R"[f * g]({s}) = \text{Area} / \sqrt{2} = e^{-{s}^2 / 2} \sqrt{\pi} / \sqrt{2}", **kw),
            Tex(R"[f * g]({s}) = \text{Area} / \sqrt{2} = e^{-{s}^2 / 2} \sqrt{\pi \over 2}", **kw),
        )
        conv_eqs[0].next_to(ORIGIN, LEFT, LARGE_BUFF).to_edge(UP)
        conv_eqs[1].move_to(area_eq)
        conv_eqs[2].move_to(area_eq)

        self.play(
            Write(conv_eqs[0]),
            Transform(area_eq["Area"].copy(), conv_eqs[0]["Area"].copy(), remover=True),
            area_eq.animate.next_to(ORIGIN, RIGHT, LARGE_BUFF).to_edge(UP),
        )
        self.wait()
        self.play(
            LaggedStart(*(
                FadeTransform(src[tex][-1], conv_eqs[1][tex][-1])
                for src, tex in [
                    (conv_eqs[0], conv_eqs[0].get_tex()),
                    (area_eq, R"e^{-{s}^2 / 2} \sqrt{\pi}"),
                    (conv_eqs[0], R"/ \sqrt{2}"),
                ]
            )),
            Write(conv_eqs[1]["="][1]),
            FadeOut(area_eq[R"\text{Area} ="][0]),
        )
        self.clear()
        self.add(conv_eqs[1])
        self.wait()
        self.play(TransformMatchingTex(
            *conv_eqs[1:3],
            key_map={R"\sqrt{\pi} / \sqrt{2}": R"\sqrt{\pi \over 2}"},
            run_time=1.5,
        ))
        self.wait()


class OscillatingGraphValue(InteractiveScene):
    def construct(self):
        # Add graph
        axes = Axes((-3, 3), (0, 2), width=8, height=1.5)
        axes.move_to(1.5 * UP)
        graph = axes.get_graph(lambda x: np.exp(-x**2 / 2) * math.sqrt(PI))
        graph.set_stroke(TEAL, 2)
        axes.add(Tex("s").set_color(RED).next_to(axes.x_axis.get_end(), UR, SMALL_BUFF))

        self.add(axes, graph)

        # Add s tracker
        s_tracker = ValueTracker(-3)
        get_s = s_tracker.get_value
        globals().update(locals())
        v_line = always_redraw(lambda: axes.get_v_line_to_graph(get_s(), graph, line_func=Line).set_stroke(WHITE, 1))
        dot = GlowDot(radius=0.2, color=WHITE)
        dot.add_updater(lambda m: m.move_to(axes.i2gp(get_s(), graph)))
        tri = Triangle(start_angle=PI / 2)
        tri.set_height(0.05)
        tri.set_fill(TEAL, 1)
        tri.set_stroke(width=0)
        tri.add_updater(lambda m: m.move_to(axes.c2p(get_s(), 0), UP))
        globals().update(locals())
        label = DecimalNumber(0, font_size=24)
        label.add_updater(lambda m: m.set_value(get_s()))
        label.add_updater(lambda m: m.next_to(tri, DOWN, SMALL_BUFF))

        self.add(v_line, dot, label, tri)

        for _ in range(2):
            self.play(
                s_tracker.animate.set_value(3),
                rate_func=there_and_back,
                run_time=24
            )


class ShowGaussianConvolutionsAsEquations(InteractiveScene):
    def construct(self):
        # Add axes
        axes = VGroup(*(
            NumberPlane(
                (-3, 3), (0, 2),
                width=3, height=1,
                background_line_style=dict(stroke_width=1, stroke_color=GREY_D),
                faded_line_style=dict(stroke_width=1, stroke_opacity=0.2, stroke_color=GREY_D)
            )
            for x in range(6)
        ))
        axes.set_height(1.25)
        axes.arrange_in_grid(2, 3, h_buff=0.75, v_buff=3.0)
        for ax in axes[2::3]:
            ax.shift(0.75 * RIGHT)
        axes.center()

        stars = Tex("*", font_size=72).replicate(2)
        eqs = Tex("=", font_size=72).replicate(2)
        stars[0].move_to(axes[0:2])
        eqs[0].move_to(axes[1:3])
        stars[1].move_to(axes[3:5])
        eqs[1].move_to(axes[4:6])

        self.add(axes)
        self.add(stars, eqs)

        # Equations
        kw = dict(
            font_size=30,
            t2c={"x": BLUE, "y": YELLOW, "{s}": TEAL, R"\sigma": RED}
        )
        equations = VGroup(
            Tex(R"e^{-x^2}", **kw),
            Tex(R"e^{-y^2}", **kw),
            Tex(R"\sqrt{\frac{\pi}{2}} e^{-{s}^2 / 2}", **kw),
            Tex(R"\frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} x^2 / \sigma^2}", **kw),
            Tex(R"\frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} y^2 / \sigma^2}", **kw),
            Tex(R"\frac{1}{\sqrt{2}\sigma \sqrt{2\pi}} e^{-\frac{1}{2} {s}^2 / 2 \sigma^2}", **kw),
        )
        normal_annotations = VGroup(
            Tex(R"\mathcal{N}\left(0, \sigma^2\right)", **kw),
            Tex(R"\mathcal{N}\left(0, \sigma^2\right)", **kw),
            Tex(R"\mathcal{N}\left(0, 2\sigma^2\right)", **kw),
        )
        normal_annotations.scale(1.25)

        for eq, ax in zip(equations, axes):
            eq.next_to(ax, UP, MED_SMALL_BUFF)
        for ann, eq in zip(normal_annotations, equations[3:]):
            ann.next_to(eq, UP, MED_LARGE_BUFF)

        # Graphs
        sigma = 0.35
        graphs = VGroup(
            axes[0].get_graph(lambda x: np.exp(-x**2)),
            axes[1].get_graph(lambda y: np.exp(-y**2)),
            axes[2].get_graph(lambda s: np.exp(-s**2 / 2) * math.sqrt(PI / 2)),
            axes[3].get_graph(lambda x: gauss_func(x, 0, sigma)),
            axes[4].get_graph(lambda y: gauss_func(y, 0, sigma)),
            axes[5].get_graph(lambda s: gauss_func(s, 0, math.sqrt(2) * sigma)),
        )
        colors = [BLUE, YELLOW, TEAL]
        for graph, color in zip(graphs, it.cycle(colors)):
            graph.set_stroke(color, 2)

        # Animations
        self.add(equations[:3])
        self.add(graphs[:3])
        self.wait()
        self.play(
            LaggedStart(*(
                FadeTransform(eq1.copy(), eq2)
                for eq1, eq2 in zip(equations[:3], equations[3:])
            ), lag_ratio=0.15),
            LaggedStart(*(
                TransformFromCopy(graph1, graph2)
                for graph1, graph2 in zip(graphs[:3], graphs[3:])
            ), lag_ratio=0.15),
            LaggedStartMap(
                FadeIn, normal_annotations[:2],
                shift=0.5 * DOWN,
                lag_ratio=0.25,
                run_time=1
            )
        )
        self.wait()
        self.play(
            FadeTransform(normal_annotations[0].copy(), normal_annotations[2]),
            FadeTransform(normal_annotations[1].copy(), normal_annotations[2]),
            run_time=1
        )
        self.wait()


class Exercise(InteractiveScene):
    def construct(self):
        # Tex key word args
        tex_kw = dict(
            t2c={
                R"\sigma_1": RED,
                R"\sigma_2": RED_B,
            }
        )

        # Set up axes
        planes = VGroup(*(
            NumberPlane(
                (-1, 3), (-1, 3),
                background_line_style=dict(stroke_color=GREY_A, stroke_width=1, stroke_opacity=0.5),
                faded_line_style=dict(stroke_color=GREY_A, stroke_width=1, stroke_opacity=0.25),
                width=5, height=5
            )
            for x in range(2)
        ))
        labels = VGroup(*map(Tex, ["x", "y", "x'", "y'"]))
        labels.scale(0.75)
        label_iter = iter(labels)
        for plane in planes:
            for axis, vect in zip(plane.axes, [RIGHT, UP]):
                label = next(label_iter)
                label.next_to(axis.get_end(), vect, SMALL_BUFF)
                plane.add(label)

        planes.arrange(RIGHT, buff=4.0)
        planes.set_width(10)
        planes.to_edge(LEFT)
        planes.set_y(-1.5)

        arrow = Arrow(*planes, stroke_width=6, stroke_color=RED, buff=0.5)

        self.add(planes)
        self.add(arrow)

        # Show lines and intersections
        lines = VGroup(
            Line(planes[0].c2p(-1, 3), planes[0].c2p(3, -1)),
            Line(planes[1].c2p(-0.25, 3), planes[1].c2p(2.25, -1)),
        )
        lines.set_stroke(YELLOW, 2)
        intersection_labels = VGroup(*(
            Tex(tex, **tex_kw)
            for tex in [
                "(s, 0)",
                "(0, s)",
                R"(s / \sigma_1)",
                R"(s / \sigma_2)",
            ]
        ))
        intersection_labels.scale(0.5)
        intersection_labels_iter = iter(intersection_labels)
        dots = Group()

        for plane, line in zip(planes, lines):
            for axis in plane.axes:
                point = find_intersection(
                    line.get_start(), line.get_vector(),
                    axis.get_start(), axis.get_vector(),
                )
                dot = GlowDot(point, color=WHITE)
                dots.add(dot)
                label = next(intersection_labels_iter)
                label.set_backstroke(width=2)
                label.next_to(point, DL, buff=SMALL_BUFF)

        l2_perp = lines[1].copy().rotate(90 * DEGREES)
        l2_perp.shift(planes[1].get_origin() - l2_perp.get_start())
        mid_point = find_intersection(
            l2_perp.get_start(), l2_perp.get_vector(),
            lines[1].get_start(), lines[1].get_vector()
        )
        d_line = Line(planes[1].get_origin(), mid_point)
        d_line.set_stroke(GREEN, 2)
        d_label = Tex("d", font_size=36)
        d_label.next_to(d_line.get_center(), UL, buff=0.05)
        elbow = Elbow()
        elbow.rotate(l2_perp.get_angle() + 90 * DEGREES, about_point=ORIGIN)
        elbow.set_stroke(width=1)
        elbow.shift(mid_point)

        self.add(lines)
        self.add(dots)
        self.add(intersection_labels)
        self.add(d_line)
        self.add(d_label)
        self.add(elbow)

        # Equations
        d_eq = Tex(R"d = \frac{s}{\sqrt{\sigma_1^2 + \sigma_2^2}}", **tex_kw)
        d_eq.scale(0.85)
        d_eq.next_to(planes[1], RIGHT, buff=0.35)

        xy_eq = Tex("x + y = s")
        xy_eq.scale(0.7)
        xy_eq.set_backstroke(width=3)
        xy_eq.next_to(lines[0].get_center(), UR, SMALL_BUFF)

        change_of_coord_eqs = VGroup(
            Tex(R"x' = x / \sigma_1", **tex_kw),
            Tex(R"y' = y / \sigma_2", **tex_kw),
        )
        change_of_coord_eqs.arrange(DOWN)
        change_of_coord_eqs.scale(0.7)
        change_of_coord_eqs.next_to(arrow, UP)

        self.add(d_eq)
        self.add(xy_eq)
        self.add(change_of_coord_eqs)

        # Words
        tex_kw["alignment"] = ""
        words = VGroup(
            TexText(R"""
                Consider the diagonal slice method for two Gaussians with\\
                different standard deviations, $\sigma_1$ and $\sigma_2$:\\
                $$
                f(x) = \frac{1}{\sigma_1 \sqrt{2\pi}} e^{-\frac{1}{2}(x / \sigma_1)^2}
                \quad \text{ and } \quad
                g(y) = \frac{1}{\sigma_2 \sqrt{2\pi}} e^{-\frac{1}{2}(y / \sigma_2)^2}
                \qquad\qquad\qquad\qquad\qquad\qquad
                $$
                The graph of $f(x)g(y)$ is no longer rotationally symmetric.\\
                However, it will be if you pass to a new set of coordinates\\
                $(x', y')$ as illustrated below. Why?
            """, **tex_kw),
            TexText(R"""
                The transformatoin of the line $x + y = s$ is illustrated below.\\
                After the transformation, the area of a the slice of the graph\\
                over this line is changed by some factor which depends on\\
                $\sigma_1$ and $\sigma_2$, but importantly, not on $s$.
                \\ \\
                Explain how to find the distance $d$ in the digram below, and\\
                how this shows that the area of a slice of $f(x)g(y)$ over the line\\
                $x + y = s$ is proportional to $e^{-\frac{1}{2} s^2 / (\sigma_1^2 + \sigma_2^2)}$.
            """, **tex_kw),
        )
        words.arrange(RIGHT, buff=LARGE_BUFF, aligned_edge=UP)
        words.set_width(FRAME_WIDTH - 1)
        words.to_edge(UP)
        v_line = Line(words.get_top(), words.get_bottom())
        v_line.set_stroke(GREY_A, 1)
        v_line.move_to(midpoint(words[0].get_right(), words[1].get_left()))
        words.add(v_line)

        self.add(words)


class WhatsTheBigdeal(TeacherStudentsScene):
    def construct(self):
        # What's the big deal
        morty = self.teacher
        stds = self.students
        self.remove(self.background)

        self.play(
            stds[0].change("pondering", look_at=self.screen),
            stds[1].change("maybe", look_at=self.screen),
            stds[2].says("What's the\nbig deal?", mode="dance_3"),
            morty.change("guilty"),
        )
        self.wait(2)
        self.play(
            self.change_students("pondering", "sassy", "hesitant", look_at=self.screen),
            morty.change("plain"),
            run_time=2
        )
        self.wait(6)

        # Aren't they common
        self.play(
            stds[2].debubble(mode="heistant"),
            stds[1].says("What else\nwould it be?", mode="maybe", run_time=1),
            stds[0].change("well"),
        )
        self.play(
            stds[2].says("Normal distributions\nare very common\nright?", mode="speaking", look_at=morty.eyes),
            run_time=2,
        )
        self.add(stds[2].bubble, stds[2].bubble.content)
        self.wait(3)

        # But are they?
        common_words = stds[2].bubble.content["Normal distributions\nare very common"][0].copy()
        are_they = Text("But are they?")
        are_they.set_color(RED)
        are_they.move_to(self.hold_up_spot, DOWN)

        self.play(
            stds[0].change("pondering", look_at=self.screen),
            stds[1].debubble(),
            stds[2].debubble(),
            morty.change("sassy"),
            FadeIn(are_they, UP),
            common_words.animate.next_to(are_they, UP)
        )
        self.wait()
        self.play(morty.change("hesitant", are_they))
        self.wait(2)

        # Central limit theorem
        clt = Text("Central Limit Theorem")
        clt.set_color(YELLOW)
        clt.move_to(common_words).to_edge(UP, buff=LARGE_BUFF)
        implies = Tex(R"\Downarrow")
        implies.next_to(clt, DOWN)

        n = len("Normaldistributions")
        self.play(
            morty.change("raise_right_hand"),
            self.change_students("pondering", "erm", "pondering", look_at=clt),
            common_words[:n].animate.next_to(implies, DOWN),
            FadeOut(common_words[n:], DOWN),
            FadeOut(are_they, DOWN),
            Write(clt),
            Write(implies)
        )
        self.wait(5)


class AddedBubble(InteractiveScene):
    def construct(self):
        randy = Randolph()
        randy.to_corner(DL)
        self.play(randy.says("It follows from \n the CLT"))
        self.remove(randy)


class StepsToProof(InteractiveScene):
    def construct(self):
        # Title
        title = Text("Steps to proving the CLT", font_size=60)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        underline = Underline(title, buff=-0.05, stretch_factor=1.5)
        self.add(title, underline)

        # Steps
        steps = VGroup(
            Text("""
                Step 1: Show that for all (finite variance) distribution, there exists
                some universal shape that this process will approach.
            """, t2s={"some": ITALIC}, alignment="LEFT"),
            Text("""
                Step 2: Show that the convolution of two Gaussians is another Gaussian.
            """, alignment="LEFT")
        )
        steps.set_width(0.9 * FRAME_WIDTH)
        steps.set_fill(WHITE)
        steps.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        steps.next_to(underline, DOWN, buff=0.75)

        steps[0]["(finite variance)"].set_opacity(0.7)

        # Two steps
        self.play(LaggedStartMap(
            FadeIn, VGroup(steps[0]["Step 1:"], steps[1]["Step 2:"]),
            shift=0.5 * UP,
            lag_ratio=0.5
        ))
        self.wait()

        # Step 1
        self.play(
            FadeIn(steps[0][len("Step1:"):], lag_ratio=0.01, run_time=3),
            FadeOut(steps[1]["Step 2:"]),
        )
        self.play(
            steps[0]["some universal shape"].animate.set_color(TEAL),
            lag_ratio=0.1,
        )
        self.wait()

        # Step 2
        self.play(
            FadeIn(steps[1], lag_ratio=0.01, run_time=2),
            self.frame.animate.move_to(steps, UP).shift(0.25 * UP)
        )
        texts = ["two Gaussians", "another Gaussian"]
        for text, color in zip(texts, [YELLOW, TEAL]):
            self.play(
                steps[1][text].animate.set_color(color),
                lag_ratio=0.1,
            )
            self.wait(0.5)
        self.wait()


class HerschelMaxwellWords(InteractiveScene):
    def construct(self):
        # Test
        ideas = VGroup(
            VGroup(
                Text("Herschel-Maxwell derivation", font_size=60),
                TexText(
                    R"Rotational symmetry of $f({x})f({y}) \Rightarrow f({x}) = Ae^{-cx^2}$",
                    t2c={"{x}": BLUE, "{y}": YELLOW},
                    font_size=36,
                ),
            ).arrange(DOWN),
            TexText(
                R"Why $\pi$ is in this formula",
                font_size=60,
                t2c={R"\pi": YELLOW}
            )
        )
        for idea in ideas:
            idea.move_to(3.15 * UP)

        self.play(FadeIn(ideas[0], 0.5 * UP))
        self.wait()
        self.play(
            FadeIn(ideas[1], 0.5 * UP),
            FadeOut(ideas[0], 0.5 * UP),
        )
        self.wait()


class LinksInDescription(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.teacher.says("""
                Links for the
                theoretically curious
                in the description
            """),
            self.change_students(
                "pondering", "hooray", "well",
                look_at=self.teacher.eyes
            )
        )
        self.play(self.change_students(
            "pondering", "well", "tease",
            look_at=4 * DOWN,
        ))
        self.wait(5)


class DrawQRCode(InteractiveScene):
    def construct(self):
        # Test
        # self.add(FullScreenRectangle(fill_color=GREY_A))
        code = SVGMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/convolutions2/gauss_example_supplements/images/SubstackQR2.svg")
        code.remove(*code[:2])
        code.set_fill(BLACK, 1)
        code.set_height(7)
        back_rect = SurroundingRectangle(code)
        back_rect.set_fill(GREY_A, 1)
        back_rect.set_stroke(WHITE, 1)

        code.shuffle()
        self.play(
            FadeIn(back_rect, time_span=(3, 5)),
            Write(code, stroke_color=RED_E, stroke_width=2, lag_ratio=0.005, run_time=5),
        )
        self.wait()


class Thumbnail1(InteractiveScene):
    def construct(self):
        # Test
        line_style = dict(stroke_color=GREY_B, stroke_width=1)
        faded_line_style = dict(stroke_opacity=0.25, **line_style)
        plane = NumberPlane(
            (-3, 3), (0, 1, 0.5),
            height=1.5, width=4,
            background_line_style=line_style,
            faded_line_style=faded_line_style,
        )
        plane.set_width(FRAME_WIDTH)
        plane.to_edge(DOWN)

        graph = plane.get_graph(lambda x: np.exp(-x**2))
        graph.set_stroke(TEAL, 5)

        expr = Tex("e^{-x^2}", font_size=120)
        expr.next_to(plane.c2p(1.5, 1), DOWN)

        question = Text("Why this function?", font_size=90)
        question.to_edge(UP)

        arrow = Arrow(
            question["this"], expr["e"],
            stroke_width=10,
            stroke_color=YELLOW
        )

        self.add(plane)
        self.add(graph)
        self.add(question)
        self.add(arrow)
        self.add(expr)


class EndScreen(PatreonEndScreen):
    pass