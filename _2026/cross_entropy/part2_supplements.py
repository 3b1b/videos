from manim_imports_ext import *


class CrossEntropyKLDivergenceTitles(InteractiveScene):
    def construct(self):
        # Test
        titles = VGroup(
            Text("Cross-Entropy", font_size=60),
            Text("KL Divergence", font_size=60),
        )
        titles.to_edge(UP)
        kw = dict(t2c={"p_i": GREEN, "q_i": PINK})
        formulas = VGroup(
            Tex(R"\sum_i p_i \big(-\log(q_i)\big)", **kw),
            Tex(R"\sum_i p_i \log\left({p_i / q_i}\right)", **kw),
        )

        for x, title, formula in zip([-1, 1], titles, formulas):
            title.set_x(x * FRAME_WIDTH / 4)
            formula.next_to(title, DOWN)

            self.play(LaggedStart(
                Write(title),
                FadeIn(formula, 0.5 * DOWN),
                lag_ratio=0.25
            ))
            self.wait()


class MovingBrace(InteractiveScene):
    def construct(self):
        # Test
        brace = Brace(ScreenRectangle(height=FRAME_HEIGHT / 2), UP)
        brace.set_y(2.5).to_edge(LEFT, buff=0.75)

        self.play(GrowFromCenter(brace))
        self.wait()
        self.play(brace.animate.to_edge(RIGHT, buff=0.75))
        self.wait()


class AcknowledgeContrived(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer().flip()
        self.play(morty.says("Contrived? Yes. \n But bear with me..."))
        self.play(morty.change("shruggie"))
        self.wait()


class AskAboutGreenGraph(InteractiveScene):
    def construct(self):
        morty = Mortimer().flip()
        morty.to_edge(LEFT).shift(DOWN)
        morty.body.insert_n_curves(1000)
        self.play(
            morty.says("What is this\nnew curve?", "speaking", look_at=3 * RIGHT),
        )
        self.play(Blink(morty))
        self.play(morty.change("tease"))
        self.play(Blink(morty))
        self.wait(5)


class AskWhyLogs(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        randy.flip()
        randy.to_edge(RIGHT)

        self.play(randy.says("Why logs?", mode="maybe", look_at=ORIGIN))
        self.play(Blink(randy))
        self.wait()
        self.play(randy.change("confused"))
        self.wait()


class AssumingFamiliarity(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        morty.body.insert_n_curves(500)

        self.play(
            morty.change("raise_right_hand", 3 * UP),
            self.change_students("tease", "well", "happy", look_at=3 * UP)
        )
        self.wait()
        self.play(self.change_students("tease", "well", "thinking", look_at=3 * UP), run_time=2)
        self.wait(2)
        self.play(
            morty.says("Quick recaps\nnever hurt"),
            self.change_students("hesitant", "tease", "well")
        )
        self.wait(3)


class PreviousVideos(InteractiveScene):
    def construct(self):
        # Test
        images = Group(
            ImageMobject("https://img.youtube.com/vi/IHZwWFHWa-w/maxresdefault.jpg"),
            ImageMobject("https://img.youtube.com/vi/Ilg3gGewQ5U/maxresdefault.jpg"),
        )
        thumbnails = Group(
            Group(
                SurroundingRectangle(image, buff=0).set_stroke(WHITE, 3),
                image
            )
            for image in images
        )
        thumbnails.set_width(4)
        thumbnails.arrange(DOWN, buff=1.5)
        thumbnails.to_corner(UL).to_edge(LEFT, buff=LARGE_BUFF)
        titles = VGroup(Text("Gradient Descent"), Text("Backpropagation"))

        for title, tn in zip(titles, thumbnails):
            title.next_to(tn, DOWN)
            self.play(LaggedStart(
                FadeIn(tn, 0.5 * DOWN),
                FadeIn(title, 1.0 * DOWN),
                lag_ratio=0.2
            ))
        self.wait()


class YouTheEngineer(InteractiveScene):
    def construct(self):
        # test
        randy = Randolph()
        randy.to_edge(LEFT)

        self.play(randy.change("erm", RIGHT))
        self.play(Blink(randy))
        self.wait(2)
        self.play(randy.change("hesitant", RIGHT))
        self.play(Blink(randy))
        self.wait()
        self.play(Blink(randy))
        self.wait(2)
        self.play(randy.change("pondering", self.frame.get_corner(UL)))
        self.play(Blink(randy))

        # Away
        self.play(
            randy.change("horrified"),
            VFadeOut(randy),
            self.frame.animate.shift(2 * UP)
        )
        self.wait()


class BillionsOfInputs(InteractiveScene):
    def construct(self):
        # Simple function
        func = Tex(R"L(x, y)", font_size=90, t2c={"L": RED})
        self.play(Write(func))
        self.wait()

        # Many labels
        lp = func["L("][0]
        rp = func[")"][0]
        xy = func["x, y"][0]

        lp.target = lp.generate_target()
        rp.target = rp.generate_target()
        lp.target.shift(2 * LEFT)
        rp.target.shift(2 * RIGHT)
        min_x = lp.target.get_x(RIGHT)
        max_x = rp.target.get_x(LEFT)

        w_i_template = Tex(R"w_{0}", font_size=48)
        subscr = w_i_template.make_number_changeable("0", edge_to_fix=LEFT)
        all_ws = VGroup()
        n_terms = 200
        for n in range(n_terms):
            subscr.set_value(n)
            all_ws.add(w_i_template.copy())
        all_ws[-1].remove(all_ws[-1][-1])
        all_ws.arrange(RIGHT)
        for mob in all_ws:
            mob.add(Text(",").next_to(mob, RIGHT, buff=0.05).match_y(mob.get_corner(DL)))

        def update_opacities(xs):
            for mob in xs:
                x = mob.get_x()
                alpha = min(
                    clip(inverse_interpolate(min_x, min_x + LARGE_BUFF, x), 0, 1),
                    clip(inverse_interpolate(max_x, max_x - LARGE_BUFF, x), 0, 1),
                )
                mob.set_fill(opacity=alpha)

        all_ws.add_updater(update_opacities)
        all_ws.next_to(lp.target, RIGHT, SMALL_BUFF)

        self.play(
            MoveToTarget(lp),
            MoveToTarget(rp),
            FadeOut(xy),
            VFadeIn(all_ws, suspend_mobject_updating=True),
        )
        self.play(
            all_ws.animate.next_to(rp, LEFT),
            run_time=20,
            rate_func=lambda t: t**3,
        )
        self.wait()


class EntropyOfPTitle(InteractiveScene):
    def construct(self):
        title = Text("Entropy of P", font_size=72, t2c={"P": GREEN})
        equation = Tex(R"\sum_i p_i (-\log_2 p_i)", t2c={"p_i": GREEN})
        title.to_edge(UP)
        equation.next_to(title, DOWN)

        self.add(title, equation)


class TeacherStudentGesture(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        morty.body.insert_n_curves(100)

        self.play(
            morty.change("raise_right_hand", 3 * UP),
            self.change_students("pondering", "thinking", "pondering", look_at=3 * UP)
        )
        self.wait()
        self.play(morty.change("tease"))
        self.wait(5)


class KeyFact(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"Q": PINK, "P": GREEN}
        kw = dict(font_size=60, t2c=t2c)
        title = Text("Key fact about Cross-Entropy\nof Q relative to P", **kw)
        title.to_corner(UL)

        key_fact = title["Key fact"]
        key_fact_underline = Underline(key_fact, buff=-0.05)
        key_fact_underline.set_stroke(YELLOW)

        pin = SVGMobject("push_pin")
        pin.set_fill((WHITE, GREY_E))
        pin.set_height(0.5)
        pin.rotate(-45 * DEG)
        pin.next_to(title["P"], RIGHT, SMALL_BUFF)
        P = title["P"][0]
        P_rect = SurroundingRectangle(P, buff=SMALL_BUFF)
        P_rect.set_stroke(WHITE, 1)

        facts = VGroup(
            Text("• Minimal when Q = P", **kw),
            Text("• Minimal value = H(P)", **kw),
        )
        facts.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)

        facts.next_to(title, DOWN, buff=2.0)
        facts.to_edge(LEFT)

        self.add(title)
        self.play(
            ShowCreation(key_fact_underline),
            key_fact.animate.set_fill(YELLOW),
        )
        self.wait()
        self.play(FadeIn(P_rect, scale=0.5, rate_func=rush_into, run_time=0.5))
        self.wait()

        for fact in facts:
            self.play(FadeIn(fact, DOWN))
            self.wait()


class ThreeDEntropyGraph(InteractiveScene):
    graph_resolution = (101, 101)

    def construct(self):
        # Test
        frame = self.frame
        epsilon = 1e-8
        axes = ThreeDAxes(
            (0, 1, 0.1), (0, 1, 0.1), (0, 5, 1),
            width=10,
            height=10,
            depth=10,
        )
        plane = NumberPlane((0, 1), (0, 1), width=10, height=10)
        plane.shift(axes.get_origin() - plane.get_origin())

        p_tracker = ValueTracker(np.array([0.1, 0.2, 0.7]))

        p_labels = VGroup(Tex(R"p_1"), Tex(R"p_2"))
        p_labels.rotate(90 * DEG, RIGHT)
        p_labels.scale(2)
        p_labels.set_color(GREEN)
        p_labels[0].next_to(axes.x_axis, RIGHT, SMALL_BUFF)
        p_labels[1].next_to(axes.y_axis.get_top(), LEFT, SMALL_BUFF)

        self.add(axes, plane, p_labels)

        # Add graph
        def ce_func(q1, q2):
            ps = p_tracker.get_value()
            qs = (q1, q2, 1 - q1 - q2)
            return sum([-p * np.log2(abs(q)) for p, q in zip(ps, qs)])

        def ent_func(p1, p2):
            # Do a check on domain?
            probs = (p1, p2, 1 - p1 - p2)
            return sum([-p * np.log2(abs(p)) for p in probs])

        ce_graph = always_redraw(lambda: self.get_simplex_graph(ce_func, axes, PINK))
        ent_graph = self.get_simplex_graph(ent_func, axes, GREEN)

        frame.reorient(-30, 85, 0, (-3.28, 0.09, 2.49), 15.68)
        self.add(ent_graph)
        self.add(ce_graph)

        # Add line
        v_line = Line(IN, OUT)
        v_line.set_stroke(GREEN, 2)

        def update_line(line):
            x, y = p_tracker.get_value()[:2]
            z = ent_func(x, y)
            v_line.put_start_and_end_on(
                axes.c2p(x, y, 0),
                axes.c2p(x, y, z),
            )
            return v_line

        v_line.add_updater(update_line)
        dot = GlowDot().set_color(GREEN)
        dot.f_always.move_to(v_line.get_end)

        self.add(v_line)
        self.add(dot)

        # Animate
        self.play(
            frame.animate.reorient(34, 90, 0, (-3.28, 0.09, 2.49), 15.68),
            p_tracker.animate.set_value([0.7, 0.1, 0.2]),
            run_time=12
        )
        ce_graph.clear_updaters()

    def get_simplex_graph(self, func, axes, color, opacity=0.5, epsilon=1e-8):
        x_range = (epsilon, 1 - epsilon)
        matrix = np.array([
            axes.x_axis.n2p(1) - axes.x_axis.n2p(0),
            axes.y_axis.n2p(1) - axes.y_axis.n2p(0),
            axes.z_axis.n2p(1) - axes.z_axis.n2p(0),
        ]).T
        graph = ParametricSurface(
            lambda u, v: [u, (1 - u) * v, func(u, (1 - u) * v)],
            u_range=x_range,
            v_range=x_range,
            color=color,
            opacity=opacity,
            resolution=self.graph_resolution
        )
        graph.apply_matrix(matrix, about_point=ORIGIN)
        graph.shift(axes.c2p(0, 0, 0))
        graph.sort_faces_back_to_front(DOWN)
        return graph


class NeglectedFacts(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        neglected_facts = VGroup(
            Text("Batch size scheduling"),
            Text("Optimizer choice and tuning"),
            Text("Learning rate schedules"),
            Text("Gradient clipping"),
            Text("Compute-optimal scaling laws"),
            Tex(R"\vdots")
        )
        neglected_facts.arrange(DOWN, aligned_edge=LEFT)
        neglected_facts[-1].set_x(neglected_facts.get_x() - 1)
        neglected_facts.to_corner(UR)

        self.add(neglected_facts)

        self.play(
            morty.change("raise_left_hand", look_at=5 * UR),
            self.change_students("sassy", "hesitant", "guilty", look_at=5 * UR),
            LaggedStartMap(FadeIn, neglected_facts, shift=0.5 * DOWN, lag_ratio=0.7, run_time=5)
        )
        self.wait(3)
        self.play(
            morty.change("tease", stds),
            self.change_students("pondering", "sassy", "pondering"),
            LaggedStartMap(FadeOut, neglected_facts, shift=RIGHT, lag_ratio=0.1),
        )
        self.wait()


class Lame(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        self.play(
            morty.says("Lame!", mode="angry"),
            self.change_students("guilty", "confused", "erm", look_at=self.screen)
        )
        self.wait(5)


class DistillationArrow(InteractiveScene):
    def construct(self):
        # Test
        arrow = Line(3 * UP, 4 * LEFT, path_arc=90 * DEG, buff=0.1, stroke_width=10)
        arrow.add_tip()
        arrow.set_color(TEAL)
        word = Text("Distillation", font_size=72)
        word.set_backstroke(BLACK, 5)
        word.next_to(arrow.pfp(0.5), UL)

        self.play(
            ShowCreation(arrow),
            FadeIn(word, 0.5 * LEFT, lag_ratio=0.05),
        )
        self.wait()


class AutoregressionArrow2(InteractiveScene):
    dims = (1, 1)

    def construct(self):
        # Test
        curve = Rectangle(*self.dims).round_corners().rotate(-90 * DEG)
        line = curve.copy().pointwise_become_partial(curve, 0.05, 0.675)
        line.set_stroke(TEAL, 5)
        tip = ArrowTip(-90 * DEG)
        tip.move_to(line.get_end(), UP).shift(1e-2 * UP)
        tip.match_color(line)
        line.add(tip)

        self.add(line)


class AskWhy(TeacherStudentsScene):
    def construct(self):
        # Test
        self.remove(self.background)
        stds = self.students
        morty = self.teacher

        self.play(
            stds[1].says("Wait, why?", mode="confused", look_at=self.screen),
            stds[0].change("pondering", self.screen),
            stds[2].change("pondering", self.screen),
            morty.change("tease")
        )
        self.wait(1)
        self.play(
            morty.change("raise_left_hand", 6 * UR),
            FadeOut(stds[1].bubble),
            self.change_students("pondering", "erm", "thinking", look_at=6 * UR)
        )
        self.wait(5)
        self.play(LaggedStartMap(FadeOut, self.pi_creatures, shift=DOWN, lag_ratio=0.1))
        self.wait()


class ShowGradientEquation(InteractiveScene):
    def construct(self):
        # Test
        kw = dict(t2c={"q": PINK, "q_i": PINK, "p_i": GREEN, R"\lambda": BLUE})
        equations = VGroup(
            Tex(R"\nabla_q \left(\sum_i p_i f(q_i) \right) = \lambda \nabla_q \left(\sum_i q_i \right)", **kw),
            Tex(R"p_i f'(q_i) = \lambda", **kw),
            Tex(R"q_i f'(q_i) = \lambda", **kw),
            Tex(R"f'(q) = {\lambda \over q}", **kw),
        )
        equations.arrange(DOWN, buff=MED_LARGE_BUFF)
        equations[1:].shift(LEFT)
        annotations = VGroup(
            TexText("for all $i$"),
            TexText(R"Assuming minimum $\Leftrightarrow q_i = p_i$", **kw),
            Tex(R"f(q) = \lambda \log(q)", **kw),
        )
        for annotation, equation in zip(annotations, equations[1:]):
            arrow = Vector(0.75 * LEFT).next_to(equation, RIGHT)
            if equation is equations[-1]:
                arrow.flip()
            annotation.scale(0.8)
            annotation.next_to(arrow, RIGHT)
            annotation.add_to_back(arrow)
            equation.add(annotation)

        self.play(FadeIn(equations[0], DOWN))
        self.wait()
        self.play(LaggedStartMap(FadeIn, equations[1:], shift=DOWN, lag_ratio=0.25))
        self.wait(4)


class FootnoteDifferneceMeasure(InteractiveScene):
    def construct(self):
        # Test
        footnote = TexText(R"""
            $^*$ The difference metric the authors defined is slightly \\
            more complicated. Define the expression above to be $\Delta_{Ab}$, \\
            for text $A$ and a small sample $b$ from text $B$. The full \\
            difference between documents was defined as
            $$S_{AB} := (\Delta_{Ab} - \Delta_{Bb}) / |b|$$
        """, alignment="")
        footnote[R"S_{AB} := (\Delta_{Ab} - \Delta_{Bb}) / |b|"].match_x(footnote).scale(1.5, about_edge=UP)
        footnote.scale(0.5)
        self.add(footnote)


class DifferenceLine(InteractiveScene):
    def construct(self):
        # Test
        line = Line(UP, DOWN).set_height(1).set_stroke(RED, 6)
        diff_word = Text("Difference")
        diff_word.next_to(line, RIGHT, SMALL_BUFF).shift(0.1 * DOWN)
        diff_word.set_color(RED)

        self.play(
            ShowCreation(line),
            FadeIn(diff_word, lag_ratio=0.1)
        )
        self.wait()


class ArrowsToFormula(InteractiveScene):
    def construct(self):
        # Test
        formula = Tex(R"\sum_i p_i (-\log q_i)", t2c={"p_i": GREEN, "q_i": PINK})
        formula.scale(1.5).to_edge(RIGHT, LARGE_BUFF)

        screens = ScreenRectangle().get_grid(2, 1, buff=0.5)
        screens.set_height(FRAME_HEIGHT - 1).to_edge(LEFT)
        screens.stretch(0.9, 0, about_edge=LEFT)

        path_arc = 90 * DEG
        arrows = VGroup(
            Arrow(screens[0].get_right() + UP, formula.get_top() + LEFT, path_arc=-path_arc, thickness=5),
            Arrow(screens[1].get_right() + DOWN, formula.get_bottom() + LEFT, path_arc=path_arc, thickness=5),
        )
        arrows.set_fill(WHITE)

        # Animate
        formula.save_state()
        for part in formula:
            part.set_opacity(0).move_to(arrows[1].get_start())
        self.play(
            Write(arrows[1]),
            Restore(formula, lag_ratio=0.01),
            run_time=2
        )
        self.wait()
        self.play(Write(arrows[0]))
        self.wait()


class PiAndCircleAnalogy2(InteractiveScene):
    def construct(self):
        # Show sum
        sum_expr = Tex(R"\sqrt{6 \sum_{n = 1}^{100} \frac{1}{n^2}}", isolate=["100"])
        rhs = Tex(R"= 3.1415926\dots")
        VGroup(sum_expr, rhs).arrange(RIGHT, buff=SMALL_BUFF).center()
        top_bound = sum_expr.make_number_changeable("100", group_with_commas=True, edge_to_fix=ORIGIN)
        top_bound.set_color(YELLOW)
        rhs_value = rhs.make_number_changeable("3.1415926", edge_to_fix=LEFT)
        sigma = sum_expr[7]
        sigma_width = sigma.get_width()

        n_tracker = ValueTracker(1)
        get_n = n_tracker.get_value
        top_bound.f_always.set_value(n_tracker.get_value)
        top_bound.always.set_max_width(sigma_width)
        rhs_value.add_updater(lambda m: m.set_value(
            math.sqrt(6 * sum(1 / n**2 for n in range(1, int(get_n()))))
        ))
        inf = Tex(R"\infty", font_size=36)
        inf.move_to(top_bound).set_color(YELLOW)

        circle = Circle(radius=0.49 * FRAME_HEIGHT)
        circle.rotate(45 * DEG)
        circle.set_stroke(WHITE, 3, 0.5)
        lights = Group(
            Group(
                GlowDot(circle.pfp(a), glow_factor=1, radius=radius, opacity=0.8 / (n + 2)**2)
                for n, radius in enumerate(np.arange(0.25, 5.25, 0.25))
            )
            for a in np.arange(0, 1, 1 / 16)
        )

        self.add(sum_expr, rhs)
        self.play(
            n_tracker.animate.set_value(10000).set_anim_args(run_time=6),
            FadeIn(circle, time_span=(3, 5)),
            FadeIn(lights, time_span=(4, 6)),
        )
        rhs_value.clear_updaters()
        self.play(
            n_tracker.animate.set_value(20000),
            VFadeOut(top_bound),
            FadeIn(inf),
            ChangeDecimalToValue(rhs_value, PI),
        )
        self.wait()

        # Transition to cross-entropy
        ce_formula = Tex(R"\sum_i p_i (-\log q_i)", t2c={"p_i": GREEN, "q_i": PINK}, font_size=60)
        compression_rects = Rectangle().get_grid(15, 1, buff=0)
        compression_rects.set_shape(3, 8)
        compression_rects.next_to(ce_formula, RIGHT, LARGE_BUFF)
        compression_rects.set_stroke(WHITE, 1, 0.25)
        compression_rects.set_z_index(-1)
        for rect in compression_rects:
            rect.set_fill(random_bright_color(hue_range=(0.1, 0.2)), 0.2)
        big_rect = SurroundingRectangle(compression_rects, buff=0)
        big_rect.set_stroke(WHITE, 2, 0)
        arrows = VGroup(Vector(0.5 * DOWN, thickness=5), Vector(0.5 * UP, thickness=5))
        arrows.set_fill(YELLOW)
        arrows[0].always.next_to(compression_rects, UP, SMALL_BUFF)
        arrows[1].always.next_to(compression_rects, DOWN, SMALL_BUFF)

        sum_expr.remove(sigma)
        sum_expr.remove(top_bound)
        sum_expr.add(inf)
        self.play(
            FadeOut(sum_expr, 0.2 * LEFT, lag_ratio=0.05),
            FadeOut(rhs, 0.2 * RIGHT, lag_ratio=0.05),
            ReplacementTransform(sigma, ce_formula[0]),
            Write(ce_formula[1:], time_span=(0.5, 2.0), stroke_color=WHITE),
            ReplacementTransform(circle, big_rect),
            FadeIn(compression_rects),
            FadeOut(lights),
        )
        self.wait()
        self.add(arrows)
        self.play(compression_rects.animate.stretch(0.3, 1), run_time=3)
        self.play(FadeOut(arrows))


class PredictorCompressorTitle(InteractiveScene):
    def construct(self):
        group = VGroup(
            Text("Predictor", font_size=72, fill_color="#951FF4").set_x(-FRAME_WIDTH / 4),
            Tex(R"\longleftrightarrow", font_size=120),
            Text("Compressor", font_size=72, fill_color="#74F9BB").set_x(FRAME_WIDTH / 4),
        )
        group.to_edge(UP)
        self.add(group[::2])
        self.wait()
        self.play(GrowFromCenter(group[1]))
        self.wait()


class AskAboutKLDivergence(TeacherStudentsScene):
    def construct(self):
        # Test
        self.remove(self.background)
        morty = self.teacher
        morty.body.insert_n_curves(1000)
        stds = self.students

        self.play(
            stds[2].says("What about\nKL Divergence?", bubble_direction=LEFT),
            stds[0].change("erm", look_at=stds[2].eyes),
            stds[1].change("confused", look_at=stds[2].eyes),
            morty.change("tease")
        )
        self.wait(3)
        self.play(
            morty.change('raise_right_hand', self.screen),
            self.change_students("pondering", "thinking", "pondering", look_at=self.screen),
            FadeOut(stds[2].bubble)
        )
        self.wait(3)

        # Transition elsewhere
        frame = self.frame
        self.play(
            self.change_students("bump", "hesitant", "well"),
            frame.animate.align_to(morty, LEFT).shift(MED_LARGE_BUFF * LEFT),
            morty.change("raise_left_hand", 8 * RIGHT),
            run_time=2
        )
        self.wait()


class EndScreen(SideScrollEndScreen):
    pass