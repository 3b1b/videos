from manim_imports_ext import *
import scipy.integrate as integrate


class SplitScreen(InteractiveScene):
    def construct(self):
        h_lines = Line(DOWN, UP).set_height(FRAME_HEIGHT).replicate(2)
        h_lines.arrange(RIGHT, buff=FRAME_WIDTH / 3)
        h_lines.center()
        h_lines.set_stroke(GREY_B, 2)
        self.add(h_lines)


class ImagineProving(InteractiveScene):
    def construct(self):
        self.add(FullScreenRectangle())
        screen = ScreenRectangle()
        screen.set_height(0.70 * FRAME_HEIGHT)
        screen.to_edge(DOWN)
        screen.set_stroke(GREY_A, 1)
        self.add(screen)

        # Randy
        randy = Randolph(height=1.5)
        randy.next_to(screen, UP, aligned_edge=LEFT)
        words = Text("""
            Imagine yourself proving
            the central limit theorem
        """)
        yourself = words["Imagine yourself"][0]
        words["central limit theorem"].set_color(YELLOW)
        words.next_to(randy, RIGHT, buff=1.5)
        arrow = Arrow(words, randy)

        yourself.save_state()
        yourself.match_y(words)

        self.play(
            Write(yourself),
            ShowCreation(arrow),
            randy.change("pondering", screen)
        )
        self.play(Blink(randy))
        self.wait()
        self.play(
            Restore(yourself),
            Write(words[len(yourself):]),
            randy.change("thinking"),
        )
        for x in range(2):
            self.play(Blink(randy))
            self.wait(2)


class WhyGaussian(InteractiveScene):
    def construct(self):
        self.add(TexText("Why $e^{-x^2}$?", font_size=72))


class AskAboutTheProof(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        self.pi_creatures.scale(0.5, about_edge=DOWN)
        self.pi_creatures.space_out_submobjects(1.5)

        # Ask
        morty = self.teacher
        stds = self.students
        self.play(
            stds[0].change("pondering", 5 * DOWN),
            stds[1].says("Proof?"),
            stds[2].change("sassy"),
            morty.change("tease"),
        )
        self.wait(6)

        words = VGroup(
            Text("Moment generating functions"),
            Text("Culumants"),
            Text("Characteristic functions"),
        )
        words.move_to(morty.get_corner(UL), DOWN)
        words.shift(0.2 * UP)
        words.scale(0.65)
        self.play(
            morty.change("raise_right_hand"),
            FadeIn(words[0], UP),
            stds[1].debubble(mode="well"),
            stds[0].change("confused", morty.eyes),
            stds[2].change("pondering", morty.eyes),
        )
        self.wait()
        for i in [1, 2]:
            self.play(
                words[:i].animate.next_to(words[i], LEFT, buff=0.5),
                FadeIn(words[i], LEFT)
            )
            self.wait()
        self.wait(3)


class RefinedHypothesis(InteractiveScene):
    def construct(self):
        # Setup
        tex_kw = dict(
            t2c={"X": BLUE_B}
        )
        # Given X, sum depends only on mu and sigma

        # ...
        # Rescale the sum, ask about the limit
        # Specify the sense of the limit
        # Ask about the moments
        # Rephrase by asking for the MGF of the scaled sum
        # Rephrase by asking for the CGF of the scaled sum
        # Show explicity CGF
        # Turn into explicit MGF
        pass


class LookingBeyondExpectationAndVariance(InteractiveScene):
    def construct(self):
        # Only E[X] and Var(X) matter
        kw = dict(t2c={
            R"\mu": PINK,
            R"\sigma": RED,
            R"\mathds{E}[X]": PINK,
            R"\text{Var}(X)": RED,
        })
        top_words1 = TexText(R"Only $\mu$ and $\sigma$ matter", **kw)
        top_words2 = TexText(R"Only $\mathds{E}[X]$ and $\text{Var}(X)$ matter", **kw)
        for words in [top_words1, top_words2]:
            words.to_edge(UP)
            words.set_backstroke(width=2)

        self.play(Write(top_words1, run_time=1))
        self.wait()
        self.remove(top_words1)
        self.play(LaggedStart(*(
            ReplacementTransform(
                top_words1[t1][0], top_words2[t2][0],
            )
            for t1, t2 in [
                ("Only", "Only"),
                (R"$\mu$", R"$\mathds{E}[X]$"),
                ("and", "and"),
                (R"$\sigma$", R"$\text{Var}(X)$"),
                ("matter", "matter"),
            ]
        )))
        self.wait()

        # Add distribution
        axes = Axes((-4, 4), (0, 0.75, 0.25), width=10, height=3)
        axes.to_edge(DOWN)

        def func(x):
            return (np.exp(-(x - 1)**2) + 0.5 * np.exp(-(x + 1)**2)) / 2.66

        graph = axes.get_graph(func)
        graph.set_stroke(TEAL, 3)
        mu = integrate.quad(lambda x: x * func(x), -4, 4)[0]
        mu_line = Line(axes.c2p(mu, 0), axes.c2p(mu, 0.5))
        mu_line.set_stroke(PINK, 2)
        exp_label = Tex(R"\mathds{E}[X]", font_size=30)
        exp_label.next_to(mu_line, UP, SMALL_BUFF)
        exp_label.set_color(PINK)
        sigma_arrows = VGroup(Vector(LEFT), Vector(RIGHT))
        sigma_arrows.arrange(RIGHT)
        sigma_arrows.move_to(mu_line.pfp(0.2))
        sigma_arrows.set_color(RED)

        plot = VGroup(axes, graph, mu_line, exp_label, sigma_arrows)

        self.play(LaggedStartMap(Write, plot))
        self.wait()

        # What else is there
        moment_list = Tex(R"""
            \mathds{E}\left[X\right],\quad
            \mathds{E}\left[X^2\right],\quad
            \mathds{E}\left[X^3\right],\quad
            \mathds{E}\left[X^4\right],\quad
            \mathds{E}\left[X^5\right],\quad
            \dots
        """)
        moment_list.next_to(top_words2, DOWN, LARGE_BUFF)
        moments = moment_list[re.compile(r"\\mathds{E}\\left\[X.*\\right\]")]
        commas = moment_list[","]

        commas.shift(SMALL_BUFF * RIGHT)
        moments[1:].shift(SMALL_BUFF * LEFT)
        for moment in moments:
            y = moment[0].get_y()
            moment[0].next_to(moment[1], LEFT, SMALL_BUFF)
            moment[0].set_y(y)

        moment_boxes = VGroup(*(
            SurroundingRectangle(moment, buff=0.05)
            for moment in moments
        ))
        moment_boxes.set_stroke(TEAL, 2)
        moment_boxes.set_fill(GREY_E, 1)
        var_sub = Tex(R"\text{Var}(X)")
        var_sub.move_to(commas[:2]).align_to(moments[0], DOWN)

        question = Text("What else is there?")
        question.next_to(moment_boxes[2:], DOWN)

        self.play(LaggedStart(
            TransformFromCopy(top_words2[R"\mathds{E}[X]"][0], moments[0]),
            Write(commas[0]),
            TransformFromCopy(top_words2[R"\text{Var}(X)"][0], var_sub),
            Write(commas[1]),
            lag_ratio=0.5,
        ))
        self.play(
            LaggedStartMap(FadeIn, moment_boxes[2:]),
            LaggedStartMap(FadeIn, commas[2:]),
            Write(question)
        )
        self.play(Write(moment_list[R"\dots"][0]))
        self.wait()

        # Expand variance
        var_equation = Tex(R"\text{Var}(X) = \mathds{E}[X^2] - \mathds{E}[X]^2")
        full_var_equation = Tex(R"""
            \text{Var}(X)
            &= \mathds{E}[(X - \mu)^2] \\
            &= \mathds{E}[X^2 - 2\mu X + \mu^2] \\
            &= \mathds{E}[X^2] - 2 \mu \mathds{E}[X] + \mu^2] \\
            &= \mathds{E}[X^2] - \mathds{E}[X]^2 \\
        """, font_size=32, t2c={R"\mu": PINK})
        var_equation.to_edge(UP)
        full_var_equation.to_corner(DL)
        full_var_equation.shift(UP)
        full_var_equation.set_fill(GREY_A)
        second_moment = var_equation[R"\mathds{E}[X^2]"][0]
        second_moment_rect = SurroundingRectangle(second_moment)

        self.play(
            TransformFromCopy(var_sub, var_equation[:len(var_sub)]),
            FadeIn(var_equation[len(var_sub):], UP),
            FadeOut(top_words2, UP),
        )
        self.wait()
        self.play(FadeIn(full_var_equation))
        self.wait()
        self.play(
            ShowCreation(second_moment_rect),
            FlashAround(second_moment, run_time=1.5)
        )
        self.play(
            TransformFromCopy(second_moment, moments[1]),
            FadeOut(var_sub, DOWN),
        )
        self.wait()
        self.play(
            FadeOut(second_moment_rect),
            var_equation.animate.scale(0.7).to_corner(UL),
            FadeOut(full_var_equation, DOWN)
        )

        # Show the moments
        self.add(moment_list, moment_boxes[2:])
        for box in moment_boxes[2:]:
            self.play(
                FadeOut(box, UP),
                question.animate.set_opacity(0),
            )
        self.remove(question)
        self.wait()

        # Name the moments
        moment_name = TexText("Moments of $X$")
        moment_name.to_edge(UP, buff=MED_SMALL_BUFF)
        moment_name.match_x(moments)
        moment_name.set_color(BLUE)
        points = np.linspace(moment_name.get_corner(DL), moment_name.get_corner(DR), len(moments))
        arrows = VGroup(*(
            Arrow(point, moment.get_top() + 0.1 * UP)
            for point, moment in zip(points, moments)
        ))
        arrows.set_color(BLUE)

        self.play(
            FadeIn(moment_name, UP),
            FadeOut(var_equation, UL),
        )
        self.play(LaggedStartMap(GrowArrow, arrows))
        self.wait()

        # Hypothesis
        rect = SurroundingRectangle(VGroup(moments[2], moment_list[R"\dots"]))
        rect.set_stroke(RED, 3)
        hyp = TexText(R"""
            Hypothesis: These have no \\
            influence on $X_1 + \cdots + X_n$
        """, font_size=42)
        hyp["Hypothesis"].set_color(RED)
        hyp.next_to(rect, DOWN).shift(LEFT)
        correction = Text("diminishing", font_size=42)
        correction.set_color(YELLOW)
        correction.next_to(hyp["no"], RIGHT)
        cross = Cross(hyp["no"])
        cross.scale(1.5)

        self.play(
            ShowCreation(rect),
            Write(hyp),
        )
        self.wait()
        self.play(ShowCreation(cross))
        self.play(Write(correction))
        self.wait()


class DefineMGF(InteractiveScene):
    def construct(self):
        # Equations
        mgf_name = Text("Moment Generating Function")
        mgf_name.to_corner(UL)

        kw = dict(t2c={
            "t": YELLOW,
            R"\frac{t^2}{2}": YELLOW,
            R"\frac{t^k}{k!}": YELLOW,
        })
        top_eq = Tex(R"M_X(t) = \mathds{E}[e^{tX}]", **kw)
        top_eq.next_to(mgf_name, DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        kw["font_size"] = 42
        lines = VGroup(
            Tex(R"\mathds{E}\left[1 + tX + \frac{t^2}{2} X^2 + \cdots + \frac{t^k}{k!} X^k + \cdots\right]", **kw),
            Tex(R"1 + \mathds{E}[X]t + \mathds{E}[X^2] \frac{t^2}{2} + \cdots + \mathds{E}[X^k] \frac{t^k}{k!} + \cdots", **kw),
        )
        stack_kw = dict(buff=1.75, aligned_edge=LEFT)
        lines.arrange(DOWN, **stack_kw)
        lines.next_to(top_eq, DOWN, **stack_kw)

        # Introduce the expression
        x_rect = SurroundingRectangle(top_eq["X"][0], buff=0.05)
        x_rect.set_stroke(BLUE, 2)
        indic_arrow = Vector(0.5 * UP)
        indic_arrow.next_to(x_rect, DOWN, SMALL_BUFF)
        indic_arrow.target = indic_arrow.generate_target()
        indic_arrow.target.next_to(top_eq["t"][0], DOWN, SMALL_BUFF)

        x_words, t_words = [
            Text(text, font_size=36)
            for text in ["Random variable", "Some real number"]
        ]
        x_words.next_to(indic_arrow, DOWN, SMALL_BUFF)
        x_words.shift_onto_screen()
        t_words.next_to(indic_arrow.target, DOWN, SMALL_BUFF)
        t_words.shift_onto_screen()

        self.play(Write(mgf_name, run_time=1))
        self.wait()
        self.play(
            FadeTransform(mgf_name[0].copy(), top_eq[0]),
            FadeIn(top_eq[1:5], lag_ratio=0.1)
        )
        self.play(
            GrowArrow(indic_arrow),
            FadeIn(x_words, 0.5 * DOWN),
            ShowCreation(x_rect),
        )
        self.wait()
        self.play(
            MoveToTarget(indic_arrow),
            FadeTransform(x_words, t_words),
            FadeOut(x_rect),
        )
        self.wait()
        self.play(
            TransformFromCopy(*top_eq["X"]),
            TransformFromCopy(*top_eq["t"]),
            Write(top_eq[R"= \mathds{E}[e"][0]),
            Write(top_eq[R"]"][0]),
        )
        self.play(FadeOut(indic_arrow), FadeOut(t_words))
        self.wait()

        # Expand series
        exp_term = top_eq[R"e^{tX}"]
        exp_series = lines[0][re.compile(r"1 +.*\\cdots")][0]
        exp_rects = VGroup(*(
            SurroundingRectangle(m, buff=0.05)
            for m in [exp_term, exp_series]
        ))
        exp_rects.set_stroke(BLUE, 2)

        arrow1 = Arrow(exp_term.get_bottom(), exp_series, stroke_color=BLUE, buff=0.3)
        arrow1_label = Text("Taylor series", font_size=32)
        arrow1_label.next_to(arrow1.get_center(), LEFT)

        gf_words = mgf_name["Generating Function"]
        gf_rect = SurroundingRectangle(gf_words, buff=0.05)
        gf_rect.set_stroke(TEAL, 2)
        gf_expl = Text("Focus on coefficients\nin a series expansion", font_size=36)
        gf_expl.next_to(gf_rect, DOWN)
        gf_expl.align_to(mgf_name["Function"], LEFT)

        self.play(
            ShowCreation(gf_rect),
            gf_words.animate.set_color(TEAL),
        )
        self.play(
            FadeIn(gf_expl, lag_ratio=0.1)
        )
        self.wait()
        self.play(ShowCreation(exp_rects[0]))
        self.play(
            ReplacementTransform(*exp_rects),
            TransformMatchingShapes(exp_term.copy(), exp_series, run_time=1),
            TransformFromCopy(top_eq[R"\mathds{E}["], lines[0][R"\mathds{E}\left["]),
            TransformFromCopy(top_eq[R"]"], lines[0][R"]"]),
            GrowArrow(arrow1),
            FadeIn(arrow1_label),
        )
        self.play(FadeOut(exp_rects[1]))
        self.wait()

        # Linearity of expectations
        frame = self.frame
        arrow2 = Arrow(*lines)
        arrow2.match_style(arrow1)
        arrow2.rotate(arrow1.get_angle() - arrow2.get_angle())
        arrow2_label = Text("Linearity of Expectation", font_size=32)
        arrow2_label.next_to(arrow2.get_center(), LEFT)

        self.play(
            frame.animate.align_to(top_eq, UP).shift(0.5 * UP),
            mgf_name.animate.next_to(top_eq, RIGHT, buff=1.0).set_color(WHITE),
            FadeOut(gf_expl, DOWN),
            FadeOut(gf_rect),
        )
        self.play(
            TransformMatchingTex(lines[0].copy(), lines[1]),
            GrowArrow(arrow2),
            FadeIn(arrow2_label),
        )
        self.wait()

        # Highlight term by term
        terms = VGroup(lines[1]["1"][0])
        terms.add(*lines[1][re.compile(r'(?<=\+)(.*?)(?=\+)')])
        terms.remove(terms[-2])
        rects = VGroup(*(SurroundingRectangle(term) for term in terms))
        rects.set_stroke(PINK, 2)

        last_rect = VMobject()
        for rect in rects:
            self.play(ShowCreation(rect), FadeOut(last_rect))
            self.wait()
            last_rect = rect
        self.play(FadeOut(last_rect))


class DirectMGFInterpretation(InteractiveScene):
    random_seed = 4
    n_samples = 30

    def construct(self):
        # Graph
        t_tracker = ValueTracker(1)
        get_t = t_tracker.get_value
        axes = Axes((-2, 2), (0, 15), width=8, height=7)
        axes.x_axis.add(Tex("x").next_to(
            axes.x_axis.get_right(), UR, SMALL_BUFF)
        )

        def get_exp_graph():
            t = get_t()
            graph = axes.get_graph(lambda x: np.exp(t * x))
            graph.set_stroke(BLUE, 2)
            return graph

        exp_graph = get_exp_graph()
        kw = dict(t2c={"t": YELLOW})
        graph_label = VGroup(
            Tex("e^{tX}", **kw).scale(1.3),
            Tex("t = 1.00", **kw),
        )
        graph_label.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        graph_label.to_corner(UR).shift(3 * LEFT)
        value = graph_label[1].make_number_changeable("1.00")
        value.add_updater(lambda m: m.set_value(get_t()))

        self.add(axes)
        self.add(exp_graph)
        self.add(graph_label)

        # Prepare samples
        samples = np.random.uniform(-2, 2, self.n_samples)

        def get_sample_group(samples):
            ys = np.exp(get_t() * samples)
            y_points = axes.y_axis.n2p(ys)
            x_points = axes.x_axis.n2p(samples)
            graph_points = axes.c2p(samples, ys)
            v_lines = VGroup(*(
                Line(xp, gp)
                for xp, gp in zip(x_points, graph_points)
            ))
            h_lines = VGroup(*(
                Line(gp, yp)
                for gp, yp in zip(graph_points, y_points)
            ))
            lines = VGroup(v_lines, h_lines)
            lines.set_stroke(WHITE, 1, 0.7)

            kw = dict(radius=0.075, glow_factor=1)
            x_dots = Group(*(GlowDot(point, color=YELLOW, **kw) for point in x_points))
            y_dots = Group(*(GlowDot(point, color=RED, **kw) for point in y_points))

            sample_group = Group(x_dots, v_lines, h_lines, y_dots)
            return sample_group

        def get_Ey():
            return np.exp(get_t() * samples).mean()

        x_dots, v_lines, h_lines, y_dots = s_group = get_sample_group(samples)

        # Show samples
        sample_arrows = VGroup(*(
            Vector(0.5 * DOWN).next_to(dot, UP, buff=0)
            for dot in x_dots
        ))
        sample_arrows.set_color(YELLOW)
        sample_words = Text("Samples of X")
        sample_words.next_to(sample_arrows, UP, aligned_edge=LEFT)

        self.play(
            # FadeIn(sample_words, time_span=(0, 1)),
            LaggedStartMap(VFadeInThenOut, sample_arrows, lag_ratio=0.1),
            LaggedStartMap(FadeIn, x_dots, lag_ratio=0.1),
            run_time=1,
        )
        # self.play(FadeOut(sample_words))

        # Show outputs
        Ey_tip = ArrowTip()
        Ey_tip.scale(0.5)
        Ey_tip.add_updater(lambda m: m.move_to(axes.y_axis.n2p(get_Ey()), RIGHT))
        Ey_label = Tex(R"\mathds{E}[e^{tX}]", **kw)
        Ey_label.add_updater(lambda m: m.next_to(Ey_tip, LEFT))

        y_dots.save_state()
        y_dots.set_points([l.get_end() for l in v_lines])
        for dot in y_dots:
            dot.set_radius(0)

        self.play(ShowCreation(v_lines, lag_ratio=0.3, run_time=2))
        self.wait()
        self.play(
            Restore(y_dots, lag_ratio=0),
            ShowCreation(h_lines, lag_ratio=0),
            run_time=2,
        )

        yd_mover = y_dots.copy()
        yd_mover.generate_target()
        points = yd_mover.get_points().copy()
        points[:] = Ey_tip.get_start()
        yd_mover.target.set_points(points)
        yd_mover.target.set_opacity(0)

        self.play(
            MoveToTarget(yd_mover),
            FadeIn(Ey_tip),
            FadeIn(Ey_label, scale=0.5),
        )
        self.wait()

        # Move around
        # self.frame.set_height(10)
        s_group.add_updater(lambda m: m.become(get_sample_group(samples)))
        exp_graph.add_updater(lambda g: g.become(get_exp_graph()))
        self.remove(*s_group)
        self.add(s_group)

        t_tracker.add_updater(lambda m: m.set_value(1.5 * math.sin(self.time / 6)))
        self.add(t_tracker)
        self.wait(48)

        for _ in range(15):
            target_t = random.uniform(0, 2.5)
            self.play(
                t_tracker.animate.set_value(target_t),
                run_time=7
            )

        for target_t in [1.5, 0.5, 1.2, 0.2, 2.0, 1.0]:
            self.play(
                t_tracker.animate.set_value(target_t),
                run_time=5
            )


class AddingVariablesWithMGF(InteractiveScene):
    def construct(self):
        # What we want is M_{X1+...+Xn}(t)

        # Addition rule
        pass


class ScalingProperty(InteractiveScene):
    def construct(self):
        # Test
        kw = dict(t2c={"c": YELLOW, "t": PINK, "X": WHITE, "Y": WHITE})
        equation = VGroup(
            Tex(R"M_{cX}(t) = ", **kw),
            Tex(R"\mathds{E}[e^{tcX}] = ", **kw),
            Tex(R"M_X(ct)", **kw),
        )
        equation.arrange(RIGHT)

        self.add(equation[0])
        self.wait()
        for i in [0, 1]:
            self.play(TransformMatchingTex(
                equation[i].copy(), equation[i + 1],
                path_arc=-45 * DEGREES
            ))
            self.wait()

        # Equation 2
        equation.generate_target()
        equation.target[1].scale(0).set_opacity(0)
        equation.target.arrange(RIGHT, buff=0.05)

        equation2 = Tex(R"M_{X+Y}(t) = M_X(t)M_Y(t)", **kw)
        equation2.next_to(equation.target, UP, buff=LARGE_BUFF)
        VGroup(equation.target, equation2).center()

        self.play(
            MoveToTarget(equation),
            FadeIn(equation2, UP),
        )
        self.wait()
        self.play(
            FadeOut(equation),
            equation2.animate.to_edge(UP),
        )
        self.wait()


class DefineCGF(InteractiveScene):
    def construct(self):
        # Take logs
        kw = dict(t2c={"t": PINK, R"\log": RED_B})
        mgf_eq = Tex(R"M_{X+Y}(t) = M_X(t)M_Y(t)", **kw)
        mgf_eq.to_edge(UP)
        log_eq = Tex(R"""
            \log\left( M_{X + Y}(t) \right) =
            \log\left( M_{X}(t) \right) +
            \log\left( M_{Y}(t) \right)
        """, font_size=36, **kw)
        log_eq.next_to(mgf_eq, DOWN, LARGE_BUFF)

        self.add(mgf_eq)
        self.wait()
        self.play(
            TransformMatchingTex(mgf_eq.copy(), log_eq)
        )
        self.wait()

        # Name CGF
        parts = log_eq[re.compile(r"\\log\\left(.* \\right)")]
        braces = VGroup()
        for part in parts:
            braces.add(Brace(part, DOWN, buff=SMALL_BUFF))

        cgf_funcs = VGroup(
            Tex("K_{X + Y}(t)", font_size=36, **kw),
            Tex("K_{X}(t)", font_size=36, **kw),
            Tex("K_{Y}(t)", font_size=36, **kw),
        )
        for name, brace in zip(cgf_funcs, braces):
            name.next_to(brace, DOWN, SMALL_BUFF)

        cgf_name = TexText("``Cumulant Generating Function''")
        cgf_name.next_to(cgf_funcs, DOWN, buff=1.5)
        arrows = VGroup(*(Arrow(cgf_name, func) for func in cgf_funcs))

        self.play(
            LaggedStartMap(GrowFromCenter, braces, lag_ratio=0.2),
            LaggedStartMap(FadeIn, cgf_funcs, shift=0.5 * DOWN, lag_ratio=0.2),
        )
        self.wait()
        self.play(
            Write(cgf_name),
            LaggedStartMap(GrowArrow, arrows),
        )
        self.wait()

        # Highlight generating function
        gen_func = cgf_name["Generating Function"]
        rect = SurroundingRectangle(gen_func, buff=0.1)
        rect.set_stroke(YELLOW, 1)

        series = Tex(
            R"K_X(t) = k_0 + k_1 t + k_2 \frac{t^2}{2} + k_3 \frac{t^3}{6} + \cdots",
            font_size=42,
            t2c={
                R"t": PINK,
                R"\frac{t^2}{2}": PINK,
                R"\frac{t^3}{6}": PINK,
            },
        )
        series.next_to(cgf_name, DOWN, LARGE_BUFF)

        self.play(
            ShowCreation(rect),
            gen_func.animate.set_color(YELLOW)
        )
        self.wait()
        self.play(FadeIn(series, lag_ratio=0.1, run_time=2))
        self.wait()


class ExpandCGF(InteractiveScene):
    def construct(self):
        # Taylor series for log
        t2c = {
            "x": YELLOW,
            "{t}": MAROON_B,
            R"\frac{t^2}{2}": MAROON_B,
            R"\frac{t^3}{3!}": MAROON_B,
            R"\frac{t^m}{m!}": MAROON_B,
        }
        kw = dict(t2c=t2c)
        log_eq = VGroup(
            Tex(R"\log(1 + x)", **kw),
            Tex(R"= x -\frac{1}{2}x^{2} +\frac{1}{3}x^{3} -\frac{1}{4}x^{4} +\cdots", **kw)
        )
        log_eq.arrange(RIGHT, buff=0.15)
        log_eq.to_edge(UP)
        equals = log_eq[1][0]
        approx = Tex(R"\approx")
        approx.move_to(equals)
        taylor_label = Text("Taylor Series expansion")
        taylor_label.next_to(log_eq, DOWN, buff=MED_LARGE_BUFF)

        self.add(log_eq)
        self.add(taylor_label)

        # Graph of log
        bls = dict(
            stroke_color=GREY_B,
            stroke_width=1,
            stroke_opacity=0.5,
        )
        axes = NumberPlane(
            (-1, 10), (-3, 3),
            background_line_style=bls,
            faded_line_ratio=0,
            num_sampled_graph_points_per_tick=10,
        )
        axes.set_height(5)
        axes.to_edge(DOWN, buff=0.25)
        graph = axes.get_graph(lambda x: np.log(1.01 + x))
        graph.make_smooth()
        graph.set_stroke(BLUE, 3)

        self.add(axes, graph)

        # Approximations
        rects = VGroup(*(
            SurroundingRectangle(log_eq[1][s][0])
            for s in [
                R"x",
                R"x -\frac{1}{2}x^{2}",
                R"x -\frac{1}{2}x^{2} +\frac{1}{3}x^{3}",
                R"x -\frac{1}{2}x^{2} +\frac{1}{3}x^{3} -\frac{1}{4}x^{4}",
                R"x -\frac{1}{2}x^{2} +\frac{1}{3}x^{3} -\frac{1}{4}x^{4} +\cdots",
            ]
        ))
        rects.set_stroke(TEAL, 2)

        coefs = [0, *[(-1)**(n + 1) / n for n in range(1, 14)]]

        def poly(x, coefs):
            return sum(c * x**n for n, c in enumerate(coefs))

        approxs = VGroup(*(
            axes.get_graph(lambda x: poly(x, coefs[:N]))
            for N in range(2, len(coefs))
        ))
        approxs.set_stroke(YELLOW, 3, opacity=0.75)

        top_rect = FullScreenRectangle().set_fill(BLACK, 1)
        top_rect.set_height(
            FRAME_HEIGHT / 2 - axes.get_top()[1] - SMALL_BUFF,
            about_edge=UP,
            stretch=True
        )

        rect = rects[0].copy()
        rect.scale(1e-6).move_to(rects[0].get_corner(DL))
        approx = VectorizedPoint(axes.c2p(0, 0))

        self.add(approx, top_rect, log_eq, taylor_label)

        for target_rect, target_approx in zip(rects, approxs):
            self.play(
                Transform(rect, target_rect),
                FadeOut(approx),
                FadeIn(target_approx),
            )
            approx = target_approx
            self.wait()
        for target_approx in approxs[len(rects):]:
            self.play(
                FadeOut(approx),
                FadeIn(target_approx),
                rect.animate.stretch(1.02, 0, about_edge=LEFT),
            )
            approx = target_approx

        self.wait()

        self.remove(top_rect)
        self.play(LaggedStartMap(
            FadeOut,
            VGroup(axes, graph, approx, rect, taylor_label)
        ))

        # Set up expansion
        mgf_term = R"\mathds{E}[X]{t} + \mathds{E}[X^2]\frac{t^2}{2} + \mathds{E}[X^3]\frac{t^3}{3!} + \cdots"
        log_mgf = Tex(R"\log(M_X({t}))", **kw)
        log_mgf.next_to(log_eq[0], DOWN, buff=1.75, aligned_edge=RIGHT)
        kt = Tex("K_X({t}) = ", **kw)

        full_rhs = Tex(Rf"""
            &=\left({mgf_term}\right)\\
            &-\frac{{1}}{{2}} \left({mgf_term}\right)^{{2}}\\
            &+\frac{{1}}{{3}} \left({mgf_term}\right)^{{3}}\\
            &-\frac{{1}}{{4}} \left({mgf_term}\right)^{{4}}\\
            &+\cdots
        """, font_size=36, **kw)

        full_rhs.next_to(log_mgf, RIGHT, aligned_edge=UP)
        log_mgf.next_to(full_rhs["="], LEFT)
        kt.next_to(log_mgf, LEFT, buff=0.2)

        # Setup log anim
        self.play(
            TransformMatchingTex(log_eq[0].copy(), log_mgf),
            FadeIn(kt, DOWN),
            run_time=1
        )
        self.wait()
        self.play(LaggedStart(
            *(
                TransformFromCopy(log_eq[1][tex][0], full_rhs[tex][0])
                for tex in [
                    "=",
                    R"-\frac{1}{2}",
                    R"+\frac{1}{3}",
                    R"-\frac{1}{4}",
                    R"+\cdots",
                    "^{2}", "^{3}", "^{4}",
                ]
            ),
            *(
                AnimationGroup(
                    TransformFromCopy(x, lp),
                    TransformFromCopy(x, rp),
                )
                for x, lp, rp in zip(
                    log_eq[1]["x"],
                    full_rhs[R"\left("],
                    full_rhs[R"\right)"],
                )
            ),
            lag_ratio=0.1,
            run_time=4
        ))
        self.wait()

        # Interim equation
        frame = self.frame
        log_eq.generate_target()
        log_eq.target.shift(UP)
        mid_eq = Tex(f"x = M_X({{t}}) - 1 = {mgf_term}", font_size=42, **kw)
        mid_eq.set_y(midpoint(log_eq.target.get_y(DOWN), full_rhs.get_y(UP)))

        self.play(LaggedStart(
            frame.animate.set_height(10),
            MoveToTarget(log_eq),
            TransformFromCopy(log_eq[0]["x"][0], mid_eq["x"][0]),
            TransformFromCopy(log_mgf["M_X({t})"][0], mid_eq["M_X({t})"][0]),
            Write(mid_eq["="]),
            Write(mid_eq["- 1"]),
        ))
        self.wait()
        self.play(FadeIn(mid_eq[mgf_term][0], lag_ratio=0.1, run_time=2))
        self.wait()

        # Fill in parens
        self.play(LaggedStart(
            *(
                TransformFromCopy(mid_eq[mgf_term][0], term, path_arc=-60 * DEGREES)
                for term in full_rhs[f"{mgf_term}"]
            ),
            lag_ratio=0.2,
            run_time=4
        ))
        self.wait()

        # Reposition
        k_eq = VGroup(kt, log_mgf, full_rhs)
        self.play(
            k_eq.animate.align_to(mid_eq, UP),
            FadeOut(mid_eq, UP)
        )
        self.wait()

        # Expand out first order term
        term11, term21, term31 = full_rhs[R"\mathds{E}[X]{t}"][:3].copy()
        term12, term22 = full_rhs[R"\mathds{E}[X^2]\frac{t^2}{2}"][:2].copy()
        term13 = full_rhs[R"\mathds{E}[X^3]\frac{t^3}{3!}"][0].copy()

        parens2 = VGroup(
            full_rhs[R"-\frac{1}{2} \left("][0],
            full_rhs[R"\right)^{2}"][0],
        ).copy()
        parens3 = VGroup(
            full_rhs[R"+\frac{1}{3} \left("][0],
            full_rhs[R"\right)^{3}"][0],
        ).copy()

        expansion = Tex(R"""
            K_X({t}) =
            \mathds{E}[X]{t} + \text{Var}(X)\frac{t^2}{2}
            + K_3[X]\frac{t^3}{3!} + \cdots + K_m[X]\frac{t^m}{m!} + \cdots
        """, **kw)
        expansion[R"\text{Var}"].set_color(BLUE)

        expansion.next_to(k_eq, DOWN, buff=0.75, aligned_edge=LEFT)

        self.add(term11)
        self.play(full_rhs.animate.set_opacity(0.3))
        self.wait()
        self.play(
            TransformFromCopy(kt, expansion[R"K_X({t}) ="][0]),
            TransformFromCopy(term11, expansion[R"\mathds{E}[X]{t}"][0]),
        )
        self.wait()

        # Second order term
        full_second = Tex(
            R"\left(\mathds{E}[X^2] - \mathds{E}[X]^2 \right) \frac{t^2}{2}",
            **kw
        )
        plus = expansion["+"][0]
        full_second.next_to(plus, RIGHT, index_of_submobject_to_align=0)

        self.play(
            FadeOut(term11),
            FadeIn(term12),
            FadeIn(parens2),
            FadeIn(term21),
        )
        self.wait()
        self.play(FlashAround(term12, run_time=1.5))
        self.play(
            LaggedStart(*(
                TransformFromCopy(m1, full_second[tex][0])
                for m1, tex in [
                    (term12[:4], R"\mathds{E}[X^2]"),
                    (term12[4:], R"\frac{t^2}{2}"),
                ]
            )),
            Write(plus),
            Write(full_second[R"\left("][0]),
            Write(full_second[R"\right)"][0]),
        )
        self.wait()
        self.play(
            TransformFromCopy(term21, full_second[R"- \mathds{E}[X]^2"][0])
        )
        self.wait()

        # Highlight variance
        expanded_var = full_second[R"\left(\mathds{E}[X^2] - \mathds{E}[X]^2 \right)"]
        var_term = expansion[R"\text{Var}(X)"]
        rect = SurroundingRectangle(expanded_var, buff=0.05)
        rect.set_stroke(BLUE, 2)
        arrow = Vector(DR)
        arrow.next_to(var_term, UL, SMALL_BUFF)
        arrow.set_color(BLUE)

        self.play(
            VGroup(expanded_var, rect).animate.next_to(arrow.get_start(), UL, SMALL_BUFF),
            FadeIn(var_term),
            GrowArrow(arrow),
            ReplacementTransform(
                full_second[R"\frac{t^2}{2}"][0],
                expansion[R"\frac{t^2}{2}"][0],
            ),
        )
        self.wait()
        self.play(
            FadeOut(expanded_var),
            FadeOut(rect),
            FadeOut(arrow),
        )

        # Third term preview
        self.play(LaggedStart(
            FadeOut(term12),
            FadeIn(term13),
            FadeIn(term22),
            FadeIn(term31),
        ))
        self.play(Write(expansion[R"+ K_3[X]\frac{t^3}{3!}"]))
        self.wait()
        self.play(
            full_rhs.animate.set_opacity(1),
            Write(expansion[R"+ \cdots + K_m[X]\frac{t^m}{m!} + \cdots"])
        )
        self.wait()


class FirstFewTermsOfCGF(InteractiveScene):
    def construct(self):
        # Test
        kw = dict(t2c={
            "t": PINK,
            R"\frac{t^2}{2}": PINK,
            R"\frac{t^3}{3!}": PINK,
            R"\frac{t^4}{4!}": PINK,
        })
        rhs = Tex(R"""
            = \mathds{E}[X]t
            + \text{Var}(X)\frac{t^2}{2}
            + K_3[X]\frac{t^3}{3!}
            + K_4[X]\frac{t^4}{4!}
            + \cdots
            """,
            **kw
        )

        self.add(rhs)


class ConfusionAboutCGF(TeacherStudentsScene):
    def construct(self):
        # Setup
        morty = self.teacher
        stds = self.students
        kw = dict(t2c={
            "t": PINK,
            R"\frac{t^2}{2}": PINK,
            R"\frac{t^3}{6}": PINK,
            R"\frac{t^m}{m!}": PINK,
            R"\log": BLUE_B
        })
        equation = Tex(
            R"""
            K_X(t)
            = \log\left(\mathds{E}[e^{tX}]\right)
            = \sum_{m=1}^\infty K_m[X] \frac{t^m}{m!}
            """,
            **kw
        )
        equation.next_to(morty.get_corner(UL), UP)
        equation.shift_onto_screen()

        # React
        self.play(
            morty.change("raise_right_hand"),
            FadeIn(equation, UP),
            self.change_students("confused", "maybe", "confused", look_at=equation),
        )
        self.wait(3)

        # Another way...
        equation.generate_target()
        equation.target.scale(0.7)
        equation.target.to_corner(UR)
        eq_rect = SurroundingRectangle(equation.target)

        words = Text("Another language to\ndescribe distributions")
        words.move_to(self.hold_up_spot, DOWN)
        arrow = Arrow(words, eq_rect, stroke_color=YELLOW)

        self.play(
            morty.change("tease"),
            MoveToTarget(equation),
            self.change_students("hesitant", "sassy", "erm", look_at=words)
        )
        self.play(
            Write(words),
            ShowCreation(arrow),
        )
        self.wait(3)

        # Back and forth
        words.generate_target()
        words.target.scale(0.7)
        words.target.next_to(equation, LEFT, buff=1.5)
        words.target.shift_onto_screen()
        arrow.target = Arrow(words.target, equation, stroke_color=YELLOW)

        self.play(
            morty.says("I'm thinking of\na distribution", run_time=1),
            MoveToTarget(words),
            MoveToTarget(arrow),
        )
        self.wait()
        self.play(
            stds[0].says("Oh cool, what\nis it?", mode="coin_flip_2", run_time=1),
            stds[1].change("happy"),
            stds[1].change("tease"),
        )
        self.wait()

        # Show pdf
        axes = Axes((0, 10), (0, 0.3), width=4, height=1.25)
        axes.move_to(self.hold_up_spot, DOWN)
        graph = axes.get_graph(lambda x: np.exp(-x) * x**3 / 6)
        graph.set_stroke(YELLOW, 2)

        label = Tex(R"p_X(x) = e^{-x} x^3 / 6", font_size=36)
        label.move_to(axes, UL)
        label.shift(0.5 * UP)

        plot = VGroup(axes, graph, label)

        self.play(
            morty.debubble(mode="raise_right_hand"),
            FadeIn(plot, UP),
        )
        self.wait()

        # Show CGF
        cgf = Tex(R"""
            K_{X}(t) = 4t + 4 \frac{t^2}{2} + 8 \frac{t^3}{6} + \cdots
            """,
            **kw
        )
        cgf.move_to(self.hold_up_spot, DOWN)

        self.play(
            morty.change("tease"),
            stds[0].change("sassy"),
            stds[1].change("hesitant"),
            plot.animate.set_width(2).to_corner(UL),
            FadeIn(cgf, UP),
        )
        self.wait(2)

        # Term by term
        words = VGroup(Text("Expected value"), Text("Variance"))
        coefs = cgf["4"]
        arrows = VGroup(*(Vector(0.5 * DOWN).next_to(coef, UP) for coef in coefs))
        arrows.set_color(YELLOW)
        words.set_color(YELLOW)

        for word, arrow in zip(words, arrows):
            word.next_to(arrow, UP, SMALL_BUFF)

        self.play(
            FadeIn(words[0]),
            FadeIn(arrows[0]),
        )
        self.wait(2)
        self.play(
            ReplacementTransform(*words),
            ReplacementTransform(*arrows),
        )
        self.wait(5)


class TermByTermSum(InteractiveScene):
    def construct(self):
        # One line equation
        top_eq = Tex("K_{X+Y}(t) = K_X(t) + K_Y(t)", t2c={"t": MAROON_B})
        top_eq.to_edge(UP)
        self.add(top_eq)
        self.wait()

        # Set up equations
        sum_eq, x_eq, y_eq = all_eqs = VGroup(*(
            VGroup(
                Tex(Rf"K_{{{sym}}}({{t}})"),
                Tex("="),
                Tex(Rf"\mathds{{E}}[{sym}] {{t}}"),
                Tex("+"),
                Tex(Rf"\text{{Var}}({sym}) \frac{{t^2}}{{2}}"),
                Tex("+"),
                Tex(Rf"K_3[{sym}] \frac{{t^3}}{{3!}}"),
                Tex(R"+ \cdots")
            )
            for sym in ["X + Y", "X", "Y"]
        ))
        for eq in all_eqs:
            eq.arrange(RIGHT, buff=SMALL_BUFF)

        all_eqs.arrange(DOWN, buff=LARGE_BUFF)
        for p1, p2, p3 in zip(x_eq, y_eq, sum_eq):
            for part in p1, p2, p3:
                part["{t}"].set_color(MAROON_B)
                part[R"\frac{t^2}{2}"].set_color(MAROON_B)
                part[R"\frac{t^3}{3!}"].set_color(MAROON_B)
            p1.match_x(p3)
            p2.match_x(p3)

        plus = Tex("+").scale(1.25)
        equals = Tex("=").scale(1.25).rotate(90 * DEGREES)
        plus.move_to(VGroup(x_eq[0], y_eq[0]))
        equals.move_to(VGroup(x_eq[0], sum_eq[0]))

        sum_eq[4].shift(0.05 * UP)

        anim_kw = dict(path_arc=-45 * DEGREES)
        self.play(LaggedStart(
            ReplacementTransform(top_eq["K_{X+Y}(t)"][0], sum_eq[0], **anim_kw),
            ReplacementTransform(top_eq["+"][1], plus, **anim_kw),
            ReplacementTransform(top_eq["K_X(t)"][0], x_eq[0], **anim_kw),
            ReplacementTransform(top_eq["="][0], equals, **anim_kw),
            ReplacementTransform(top_eq["K_Y(t)"][0], y_eq[0], **anim_kw),
            run_time=3,
        ))
        self.play(LaggedStartMap(FadeIn, sum_eq[1:]))
        self.play(FadeIn(x_eq[1:], DOWN))
        self.play(FadeIn(y_eq[1:], DOWN))
        self.wait()

        # Highlight variance
        pe_copy = VGroup(plus, equals).copy()
        rects = VGroup(*(
            SurroundingRectangle(VGroup(*(eq[i] for eq in all_eqs)))
            for i in [2, 4, 6]
        ))
        rects.set_stroke(YELLOW, 2)
        self.play(
            ShowCreation(rects[0]),
            pe_copy.animate.match_x(rects[0]),
        )
        self.wait()
        for r1, r2 in zip(rects, rects[1:]):
            self.play(
                ReplacementTransform(r1, r2),
                pe_copy.animate.match_x(r2),
            )
            self.wait()
        self.play(
            rects[2].animate.become(SurroundingRectangle(all_eqs).set_stroke(width=0)),
            FadeOut(pe_copy),
        )
        self.wait()


class CumulantsOfScaledSum(InteractiveScene):
    def construct(self):
        # Setup
        kw = dict(
            t2c={
                "{t}": MAROON_B,
                R"\frac{t^2}{2}": MAROON_B,
                R"\frac{t^3}{6}": MAROON_B,
                R"\frac{t^m}{m!}": MAROON_B,
            }
        )
        clt_var_tex = R"{X_1 + \cdots + X_n \over \sigma \sqrt{n}}"
        clt_var_flat = R"(X_1 + \cdots + X_n) / \sigma \sqrt{n}"
        cgf_def = Tex(R"""
            K_X({t})
            = \log\left(\mathds{E}\left[e^{{t}X}\right]\right)
            = \sum_{m=1}^\infty K_m[X] \frac{t^m}{m!}
        """, **kw)
        cgf_of_clt_var = Tex(
            Rf"""
                \mathds{{E}}\left[{clt_var_tex}\right] {{t}} 
                + \text{{Var}}\left({clt_var_tex}\right) \frac{{t^2}}{{2}}
                + K_3\left[{clt_var_tex}\right] \frac{{t^3}}{{6}}
                + \cdots 
            """,
        )
        cgf_of_clt_var.set_max_width(FRAME_WIDTH - 1)
        cgf_of_clt_var.to_edge(UP)

        clt_var = Tex(clt_var_tex, **kw)

        # Highlight scaled sum
        clt_var.set_height(1.5)
        clt_var.move_to(ORIGIN, DOWN)
        rect = SurroundingRectangle(clt_var)
        rect.set_stroke(YELLOW, 2)
        limit_words = TexText(R"What happens to \\ this as $n \to \infty$?")
        limit_words.next_to(rect, DOWN)
        limit_words.next_to(rect, DOWN)

        self.add(clt_var)
        self.play(
            FlashAround(clt_var, run_time=1.5),
            ShowCreation(rect),
            FadeIn(limit_words, lag_ratio=0.1),
        )
        self.wait()

        # Bring in CGF
        cgf_def.to_edge(UP)
        cgf_of_clt_sum = Tex(Rf"""
            K_{{{clt_var_flat}}}({{t}})
            = \sum_{{m=1}}^\infty
            K_m\left[{clt_var_tex}\right]
            \frac{{t^m}}{{m!}}
        """, **kw)
        cumulant_of_clt_var = cgf_of_clt_sum[Rf"K_m\left[{clt_var_tex}\right]"][0]
        cgf_of_clt_sum.save_state()
        cgf_of_clt_sum.set_opacity(0)
        cumulant_of_clt_var.set_opacity(1)
        cumulant_of_clt_var.center()
        cumulant_of_clt_var.set_height(1.5)

        self.play(
            clt_var.animate.center(),
            FadeIn(cgf_def, lag_ratio=0.1),
            FadeOut(limit_words),
            FadeOut(rect),
        )
        self.wait()
        self.play(
            ReplacementTransform(
                clt_var,
                cgf_of_clt_sum[clt_var_tex][0],
                path_arc=45 * DEGREES,
            ),
            Write(cumulant_of_clt_var[:3]),
            Write(cumulant_of_clt_var[-1:]),
        )
        self.wait()
        self.play(Restore(cgf_of_clt_sum))
        self.wait()

        # Discuss scaling
        sigma_sqrt_n = cgf_of_clt_sum[R"\sigma \sqrt{n}"][0]
        sigma_sqrt_n_rect = SurroundingRectangle(sigma_sqrt_n, buff=0.05)
        sigma_sqrt_n_rect.set_stroke(YELLOW, 2)

        scaling_equation = Tex(
            R"""
                K_{cX}(t)
                = \log\left(\mathds{E}\left[e^{tcX} \right]\right)
                = \sum_{m=1}^\infty K_m[X] c^m \frac{t^m}{m!}
            """,
            t2c={
                "t": MAROON_B,
                R"\frac{t^m}{m!}": MAROON_B,
                "c": YELLOW,
            }
        )
        part1 = scaling_equation["K_{cX}(t)"][0]
        part2 = scaling_equation[re.compile(r"=.*\\right\)")][0]
        part3 = scaling_equation[re.compile(r"= \\sum.*m\!\}")][0]
        arrow = Vector(LEFT).next_to(part1, RIGHT)
        question = Text("How does this expand?")
        question.next_to(arrow)

        cX_rect = SurroundingRectangle(scaling_equation["cX"][1], buff=0.05)
        tc_rect = SurroundingRectangle(scaling_equation["tc"][0], buff=0.05)
        new_K_rect = SurroundingRectangle(scaling_equation["K_m[X] c^m"], buff=0.1)
        VGroup(cX_rect, tc_rect).set_stroke(BLUE, 2)
        new_K_rect.set_stroke(YELLOW, 2)

        self.play(ShowCreation(sigma_sqrt_n_rect))
        self.wait()
        self.play(
            TransformFromCopy(
                cgf_of_clt_sum[f"K_{{{clt_var_flat}}}({{t}})"][0],
                part1,
            ),
            Write(question),
            GrowArrow(arrow),
            cgf_of_clt_sum.animate.scale(0.75).to_edge(DOWN, buff=LARGE_BUFF),
            FadeOut(sigma_sqrt_n_rect, 2 * DOWN + 0.2 * RIGHT),
        )
        self.wait()
        self.play(
            FadeOut(question, lag_ratio=0.1),
            arrow.animate.next_to(part2, RIGHT),
            FadeIn(part2, lag_ratio=0.1)
        )
        self.wait()
        self.play(ShowCreation(cX_rect))
        self.wait()
        self.play(ReplacementTransform(cX_rect, tc_rect))
        self.wait()
        self.play(
            FadeOut(tc_rect),
            arrow.animate.next_to(part3, RIGHT),
            FadeIn(part3, lag_ratio=0.1)
        )
        self.wait()
        self.play(ShowCreation(new_K_rect))
        self.wait()
        self.play(
            FadeOut(cgf_def, UP),
            FadeOut(arrow, UP),
            scaling_equation.animate.to_edge(UP),
            MaintainPositionRelativeTo(new_K_rect, scaling_equation),
        )

        # Scaling rule for cumulant
        scaling_rule = Tex(R"K_m[cX] = c^m K_m[X]", t2c={"c": YELLOW})
        var_example = TexText(R"""
            For example, $\text{Var}(cX) = c^2 \text{Var}(X)$
        """, t2c={"c": YELLOW})
        var_example.move_to(scaling_rule.get_bottom())

        self.play(Write(scaling_rule["K_m[cX] = "][0]))
        self.wait()
        self.play(
            new_K_rect.animate.move_to(scaling_rule["c^m K_m[X]"]),
            LaggedStart(*(
                TransformFromCopy(
                    scaling_equation[tex][0],
                    scaling_rule[tex][0],
                    path_arc=45 * DEGREES,
                )
                for tex in ["K_m[X]", "c^m"]
            ))
        )
        self.play(FadeOut(new_K_rect))
        self.wait()
        self.play(
            scaling_rule.animate.next_to(var_example, UP, buff=0.5),
            FadeIn(var_example, 0.5 * DOWN)
        )
        self.wait()

        # Apply to the cummulant of the clt variable
        lhs = cumulant_of_clt_var.copy()
        scaling_rule.generate_target()
        scaling_rule.target.center().to_edge(UP)
        implies = Tex(R"\Downarrow", font_size=72)
        implies.next_to(scaling_rule.target, DOWN, buff=0.5)
        rhs = Tex(R"""
            = \left({1 \over \sigma \sqrt{n}} \right)^m
            K_m[X_1 + \cdots + X_n]
        """, font_size=42)
        rhs[R"\left({1 \over \sigma \sqrt{n}} \right)"].set_color(YELLOW)
        lhs.next_to(implies, DOWN, buff=0.5)

        self.play(
            FadeOut(var_example, UP),
            FadeOut(scaling_equation, UP),
            MoveToTarget(scaling_rule),
        )
        self.play(
            Write(implies),
            TransformFromCopy(cumulant_of_clt_var, lhs),
        )
        denom = lhs[-5:-1]
        self.play(
            FlashAround(denom),
            denom.animate.set_color(YELLOW)
        )
        self.wait()

        center_point = lhs.get_center() + LEFT
        rhs.next_to(center_point, RIGHT, buff=0.1)
        self.play(
            lhs.animate.next_to(center_point, LEFT, 0.1),
            TransformMatchingShapes(lhs.copy(), rhs),
        )
        self.wait()

        # Expand sum
        K_of_sum = rhs[R"K_m[X_1 + \cdots + X_n]"]
        K_of_sum_rect = SurroundingRectangle(K_of_sum, buff=0.1)
        K_of_sum_rect.set_stroke(BLUE, 2)

        sum_of_K = Tex(R"K_m[X_1] + \cdots + K_m[X_n]", font_size=42)
        nK = Tex(R"n \cdot K_m[X]", font_size=42)
        equals = Tex("=").rotate(90 * DEGREES).replicate(2)
        stack = VGroup(equals[0], sum_of_K, equals[1], nK)
        stack.arrange(DOWN)
        stack.next_to(K_of_sum_rect, DOWN)

        self.play(ShowCreation(K_of_sum_rect))
        self.wait()
        self.play(
            Write(equals[0]),
            TransformMatchingShapes(K_of_sum.copy(), sum_of_K, run_time=1.5)
        )
        self.wait()
        self.play(
            Write(equals[1]),
            TransformMatchingTex(sum_of_K.copy(), nK, run_time=1.5),
            cgf_of_clt_sum.animate.set_height(0.75).shift(3 * LEFT)
        )
        self.wait()
        nK.target = nK.generate_target()
        nK.target.move_to(K_of_sum, LEFT)
        self.play(
            MoveToTarget(nK),
            FadeOut(equals, UP),
            FadeOut(sum_of_K, UP),
            FadeOut(K_of_sum, UP),
            K_of_sum_rect.animate.match_points(SurroundingRectangle(nK.target))
        )
        self.play(
            FadeOut(K_of_sum_rect),
        )
        self.wait()

        # Emphasize expression
        cumulant_of_clt_var_equation = VGroup(
            lhs, rhs[re.compile(r"=.*\^m")], nK
        )
        full_rect = SurroundingRectangle(cumulant_of_clt_var_equation, buff=0.5)
        full_rect.set_stroke(TEAL, 3)
        words = TexText(f"Excellent! This is a complete(?) description of ${clt_var_tex}$")
        words.next_to(full_rect, UP, buff=MED_LARGE_BUFF)

        self.play(
            ShowCreation(full_rect),
            FadeOut(scaling_rule, UP),
            FadeOut(implies, 2 * UP),
        )
        self.play(Write(words))
        self.wait()

        # Clean up
        clean_rhs = Tex(R"""
            \frac{1}{\sigma^m}
            \frac{n}{n^{m / 2}}
            K_m[X]
        """)

        self.play(
            FadeOut(words, UP),
            FadeOut(full_rect, 2 * UP),
            FadeOut(cgf_of_clt_sum),
            cumulant_of_clt_var_equation.animate.to_edge(UP),
        )
        self.wait()


class ConnectingArrow(InteractiveScene):
    def construct(self):
        # Test
        arrow = CubicBezier(
            np.array([-0.5, -3.33, 0]),
            np.array([3, -3.33, 0]),
            np.array([-1, 1.0, 0]),
            np.array([0.7, 1.85, 0]),
        )
        arrow.set_stroke(RED_E, 8)
        line = Line(*arrow.get_points()[-2:])
        line.add_tip()
        tip = line.tip
        tip.shift(0.25 * line.get_vector())
        tip.set_fill(RED_E, 1)

        label = TexText(R"$m^{\text{th}}$ cumulant of $\frac{X_1 + \cdots + X_n}{\sigma \sqrt{n}}$")
        label.set_fill(RED_E)
        label.set_stroke(BLACK, 2, background=True)
        label.next_to(arrow.pfp(0.5), RIGHT, SMALL_BUFF)

        self.play(
            ShowCreation(arrow),
            FadeIn(tip, time_span=(0.8, 1.0))
        )
        self.play(Write(label, stroke_color=BLACK, lag_ratio=0.2))
        self.wait()


class PictureCharacteristicFunction(InteractiveScene):
    def construct(self):
        pass
