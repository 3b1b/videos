from manim_imports_ext import *
from _2025.laplace.integration import get_complex_graph


class WriteLaplace(InteractiveScene):
    def construct(self):
        title = Text("Laplace Transform", font_size=72)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()


class TwoLevels(InteractiveScene):
    def construct(self):
        # Test
        words = VGroup(
            Text("Solve equations"),
            Text("See what it’s doing"),
        )
        words.scale(1.5)
        for word, y in zip(words, (2, -2)):
            word.set_y(y)
            word.to_edge(LEFT, buff=LARGE_BUFF)
            self.play(Write(word))
            self.wait()


class TwoKeyIdeas(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"{s}": YELLOW, "{s_n}": YELLOW, "{t}": BLUE}
        title = Text("Key ideas from the last chapter", font_size=72)
        title.to_edge(UP)
        underline = Underline(title, buff=-0.05)

        self.add(title, underline)

        ideas = VGroup(
            TexText(R"1) How to think about $e^{{s}{t}}$", t2c=t2c),
            TexText(R"2) Often, for functions in physics, $\displaystyle f({t}) = \sum_{n=1}^N c_n e^{{s_n} {t}}$", t2c=t2c),
        )
        for idea in ideas:
            idea[:2].scale(1.25, about_edge=RIGHT)
        ideas[0][R"$e^{{s}{t}}$"].scale(1.5, about_edge=DL)
        ideas.arrange(DOWN, aligned_edge=LEFT, buff=LARGE_BUFF)
        ideas.to_edge(LEFT)

        numbers = VGroup(idea[:2] for idea in ideas)
        words = VGroup(idea[2:] for idea in ideas)

        self.play(LaggedStartMap(FadeIn, numbers, shift=LEFT, lag_ratio=0.5))
        self.wait()
        for word in words:
            self.play(Write(word))
            self.wait()


class LevelsOfUnderstanding(InteractiveScene):
    def construct(self):
        items = BulletedList(
            "1) Use",
            "2) Dissect",
            "3) Reinvent",
            font_size=72,
            buff=2
        )
        numbers = VGroup()
        words = VGroup()
        for item in items:
            item[0].scale(0).set_opacity(0)
            numbers.add(item[1:3])
            words.add(item[3:])
        items.to_edge(LEFT)

        self.play(LaggedStartMap(FadeIn, numbers, shift=DOWN, lag_ratio=0.35))
        self.wait()
        for word in words:
            self.play(Write(word))
            self.wait()


class DrivingACar(InteractiveScene):
    def construct(self):
        # Test
        car = Car()
        car.move_to(4 * LEFT)
        car[0][2].set_fill(BLUE).insert_n_curves(100)
        self.add(car)
        self.play(MoveCar(4 * RIGHT), run_time=10)
        self.wait()


class WhatIsItTryingToDo2(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(1000)

        self.play(
            morty.change("raise_right_hand", look_at=3 * UP),
            self.change_students("thinking", "confused", "pondering", look_at=3 * UR)
        )
        self.wait(2)
        self.play(LaggedStart(
            stds[0].change("pondering", morty.eyes),
            stds[1].says("What does it\nactually do?", mode="raise_left_hand", look_at=morty.eyes),
            stds[2].change("hesitant", morty.eyes),
            morty.change("tease"),
            lag_ratio=0.3
        ))
        self.wait(4)

        # Test2
        self.play(
            stds[0].change("pondering", 3 * UP),
            stds[1].debubble(look_at=3 * UP),
            stds[2].change("pondering", 3 * UP),
            morty.change("raise_right_hand"),
        )
        self.wait(3)


class ReferenceComplexExponents(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        for pi in self.pi_creatures:
            pi.body.insert_n_curves(500)

        self.play(
            morty.change("raise_right_hand"),
            self.change_students('pondering', 'thinking', 'tease', look_at=self.screen)
        )
        self.wait(3)
        self.play(self.change_students('tease', 'tease', 'confused', look_at=self.screen))
        self.wait(2)
        self.play(self.change_students('thinking', 'tease', 'erm', look_at=self.screen))
        self.wait(2)
        self.play(
            morty.change('hesitant', stds[2].eyes),
            stds[2].says("Why?", mode='maybe', bubble_direction=LEFT),
        )
        self.wait(3)
        self.play(
            morty.change("raise_right_hand"),
            stds[2].debubble(mode="pondering", look_at=3 * UR),
            stds[0].change("pondering", look_at=3 * UR),
            stds[1].change("hesitant", look_at=3 * UR),
        )
        self.wait(3)


class ReferenceWorkedExample(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        rect = Rectangle(8, 1.5)
        rect.next_to(morty, UP).to_edge(RIGHT).shift(UP)
        rect.set_stroke(YELLOW, 2)
        self.play(
            morty.change("raise_left_hand", rect),
            self.change_students("pondering", "thinking", "erm", look_at=rect),
        )
        self.play(ShowCreation(rect))
        self.wait()
        self.play(
            morty.change("raise_right_hand", self.screen),
            FadeOut(rect),
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen)
        )
        self.wait(7)


class ButWhatIsIt(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            stds[0].change("erm", self.screen),
            stds[1].says("But, what is it?", mode="maybe", bubble_direction=LEFT),
            stds[2].change("sassy", self.screen),
            morty.change("guilty"),
        )
        self.wait(2)
        self.play(
            morty.change("raise_right_hand", 2 * UR),
            stds[0].change("pondering", 2 * UR),
            stds[1].debubble(mode="pondering", look_at=2 * UR),
            stds[2].change("pondering", 2 * UR),
        )
        self.wait(3)


class MoreInAMoment(InteractiveScene):
    def construct(self):
        morty = Mortimer()
        morty.body.insert_n_curves(1000)
        self.play(morty.says("Much, much more on\nthis in a moment"))
        self.play(morty.change("tease"))
        self.play(Blink(morty))
        self.wait()


class YouAsAMathematician(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph(height=2.5)
        randy.flip()
        randy.to_edge(DOWN, buff=MED_SMALL_BUFF)

        you = Text("You")
        you.next_to(randy, UR, LARGE_BUFF)
        mathematician = Text("(Mathematician)")
        mathematician.next_to(you, DOWN, aligned_edge=LEFT)
        arrow = Arrow(you.get_corner(DL), randy.body.get_corner(UR) + 0.7 * LEFT, buff=0.1, thickness=5)

        self.play(
            randy.change('pondering', 3 * UL),
            GrowArrow(arrow),
            Write(you)
        )
        self.play(FadeIn(mathematician, 0.25 * DOWN))
        self.play(Blink(randy))
        self.wait()

        # Reference machine
        label = VGroup(you, mathematician)
        self.play(
            randy.change("raise_left_hand", 3 * UR),
            label.animate.next_to(randy, RIGHT).set_opacity(0.5),
            FadeOut(arrow, scale=0.5, shift=DL),
        )
        self.play(Blink(randy))
        self.wait(2)
        self.play(
            randy.change("hesitant", 3 * UL)
        )
        for _ in range(2):
            self.play(Blink(randy))
            self.wait(2)
        self.play(randy.change("pondering", 3 * UL))
        for _ in range(2):
            self.play(Blink(randy))


class FullCosInsideSum(InteractiveScene):
    def construct(self):
        tex = Tex(R"\frac{1}{2} \left( e^{(i - {s}){t}} + e^{(\minus i - {s}){t}} \right)", t2c={"{t}": BLUE, "{s}": YELLOW})
        self.add(tex)


class ReferenceTheIntegral(TeacherStudentsScene):
    def construct(self):
        # React to the preview
        morty = self.teacher
        stds = self.students

        self.play(
            morty.change("guilty"),
            self.change_students("confused", "concentrating", "pleading", look_at=self.screen)
        )
        self.wait(4)

        # Show expression
        t2c = {"{s}": YELLOW, "{t}": BLUE}
        lt_def = Tex(R"F({s}) = \int^\infty_0 f({t}) e^{\minus {s} {t}} d{t}", t2c=t2c)
        lt_def.move_to(self.hold_up_spot, DOWN)
        lt_def.shift(0.5 * DOWN)

        self.play(
            morty.change("raise_right_hand", look_at=lt_def),
            FadeIn(lt_def, UP),
            self.change_students("confused", "tease", "happy", look_at=lt_def)
        )
        self.wait(2)

        # Highlight integral
        int_rect = SurroundingRectangle(lt_def[R"\int^\infty_0"])
        body_rect = SurroundingRectangle(lt_def[R"f({t}) e^{\minus {s} {t}}"])
        VGroup(int_rect, body_rect).set_stroke(TEAL, 2)
        arrow = Vector(0.75 * DOWN, thickness=6)
        arrow.set_fill(TEAL)
        arrow.next_to(int_rect, UP)

        self.play(
            morty.change("hesitant", int_rect),
            ShowCreation(int_rect),
            GrowArrow(arrow),
            self.change_students("pondering", "happy", "tease"),
        )
        self.wait(2)

        # Show complex plane
        s_plane = ComplexPlane((-2, 2), (-2, 2))
        s_plane.add_coordinate_labels(font_size=16)
        s_plane.set_width(2.5)
        s_plane.next_to(body_rect, UP, LARGE_BUFF)
        s_plane.shift_onto_screen()
        s_plane.to_edge(UP, SMALL_BUFF)

        s = complex(-0.1, 1.5)
        t_max = 20
        func_path = ParametricCurve(
            lambda t: s_plane.n2p(2 * math.cos(t) * np.exp(s * t)),
            t_range=(1e-3, t_max, 0.1),
        )
        func_path.set_stroke(YELLOW, 2)
        dot = GlowDot()
        dot.move_to(func_path.get_start())
        tail = TracedPath(dot.get_center, stroke_color=YELLOW, time_traced=5)
        tail.add_updater(lambda m: m.set_stroke(YELLOW, width=(0, 3)))
        for t in range(30):
            tail.update(1 / 30)

        self.play(
            morty.change("raise_left_hand", s_plane),
            self.change_students("erm", "confused", "sassy", s_plane),
            arrow.animate.rotate(PI).scale(0.75).next_to(body_rect, UP).set_anim_args(path_arc=90 * DEG),
            ReplacementTransform(int_rect, body_rect),
        )
        self.add(tail)
        self.play(
            FadeIn(s_plane),
            MoveAlongPath(dot, func_path, run_time=10, rate_func=linear),
        )

        # Clear the board
        self.play(
            VFadeOut(tail),
            FadeOut(body_rect),
            FadeOut(arrow),
            FadeOut(s_plane),
            FadeOut(dot),
            lt_def.animate.move_to(2 * UP),
            morty.change("tease", look_at=2 * UP),
            self.change_students("pondering", "pondering", "pondering", look_at=2 * UP)
        )
        self.wait()

        # Remove f(t)
        lt_def.save_state()
        ft = lt_def[R"f({t})"][0]
        ft_rect = SurroundingRectangle(ft, buff=0.05)
        ft_rect.set_stroke(RED, 2)
        self.play(
            ShowCreation(ft_rect)
        )
        self.play(
            VGroup(ft_rect, ft).animate.to_corner(UL).set_stroke(width=0).fade(0.5),
            lt_def[R"e^{\minus {s} {t}} d{t}"].animate.align_to(ft, LEFT).shift(SMALL_BUFF * LEFT),
            lt_def["F({s}) = "].animate.set_fill(opacity=0),
        )
        self.wait(5)

        # Bring back the definition
        exp_int = VGroup(*lt_def[R"\int^\infty_0"][0], *lt_def[R"e^{\minus {s} {t}} d{t}"]).copy()
        exp_int_rhs = Tex(R"= \frac{1}{{s}}", t2c=t2c)
        exp_int_rhs.next_to(exp_int, RIGHT)
        exp_int_rhs.save_state()
        exp_int_rhs.move_to(morty.get_corner(UL)).set_fill(opacity=0)
        morty.body.insert_n_curves(500)
        exp_int.add(*exp_int_rhs)

        ft2, arrow, fancy_L = new_def_lhs = VGroup(
            Tex(R"f({t})", t2c=t2c),
            Vector(1.5 * RIGHT),
            Tex(R"\mathcal{L}"),
        )
        arrow.next_to(lt_def[R"\int"], LEFT)
        fancy_L.next_to(arrow, UP, SMALL_BUFF)
        ft2.next_to(arrow, LEFT)

        lt_def.saved_state[:5].set_opacity(0)

        self.play(
            morty.change("raise_right_hand", exp_int_rhs),
            Restore(exp_int_rhs)
        )
        self.wait(2)
        self.play(
            exp_int.animate.scale(0.5).fade(0.5).to_corner(UR),
            Restore(lt_def),
            self.change_students("confused", "erm", "sassy", look_at=exp_int),
            morty.change("guilty", stds[2].eyes),
            Write(new_def_lhs),
        )
        self.wait()
        self.play(
            TransformFromCopy(ft2, ft, path_arc=PI / 2)
        )
        self.wait(3)

        # Substitute in an exponential
        exp1, exp2 = exp_examples = Tex(R"e^{1.5{t}}", t2c=t2c).replicate(2)
        ft.refresh_bounding_box()
        exp1.move_to(ft2)
        exp2.move_to(ft)
        exp_examples.align_to(lt_def["e"], DOWN)

        self.play(
            LaggedStart(
                FadeOut(ft2, 0.5 * UP),
                FadeIn(exp1, 0.5 * UP),
                FadeOut(ft, 0.5 * UP),
                FadeIn(exp2, 0.5 * UP),
                lag_ratio=0.25
            ),
            morty.change("raise_right_hand"),
            self.change_students("pondering", "thinking", "tease")
        )
        self.wait(4)

        # Collapse the inside
        lt_def["f({t})"].set_fill(opacity=0)
        new_int = Tex(R"\int^\infty_0 e^{(1.5 - {s}){t}} d{t}", t2c=t2c)
        new_int.move_to(lt_def[R"\int"], LEFT).shift(1.0 * DOWN)
        new_int2 = Tex(R"\int^\infty_0 e^{\minus({s} - 1.5){t}} d{t}", t2c=t2c)
        new_int2.move_to(new_int)

        self.play(
            VGroup(exp_examples, arrow, fancy_L, lt_def).animate.to_edge(UP),
            FadeIn(new_int, DOWN),
            self.change_students("pondering", "pondering", "pondering", new_int),
            morty.change("tease", new_int),
        )
        self.wait()
        self.play(TransformMatchingTex(new_int, new_int2, path_arc=90 * DEG))
        self.wait()

        # Show the answer
        exp_int.target = exp_int.generate_target()
        exp_int.target.scale(1.5, about_edge=UR)
        exp_int.target.set_opacity(1)
        exp_int_rect = SurroundingRectangle(exp_int.target, buff=MED_SMALL_BUFF)
        exp_int_rect.set_stroke(YELLOW, 2)

        example_rhs = Tex(R"= {1 \over {s} - 1.5}", t2c=t2c)
        example_rhs.next_to(new_int2[-1], RIGHT)

        self.play(
            morty.change('raise_left_hand', exp_int),
            MoveToTarget(exp_int),
            ShowCreation(exp_int_rect),
            self.change_students("pondering", "thinking", "happy", look_at=exp_int),
        )
        self.wait()
        self.play(
            TransformMatchingShapes(exp_int[-4:], example_rhs),
            FadeOut(exp_int[:-4]),
            exp_int_rect.animate.surround(example_rhs, buff=SMALL_BUFF),
            morty.change("raise_right_hand"),
            run_time=1.5,
        )
        self.wait()

        # Label it as pole
        words = Text("Pole at 1.5")
        words.next_to(exp_int_rect, UP)
        words.match_x(example_rhs[1:])
        words.set_color(YELLOW)

        self.play(
            Write(words),
            exp_int_rect.animate.surround(example_rhs[1:], buff=SMALL_BUFF),
            morty.change("tease", words),
        )
        self.wait(3)
        self.play(
            FadeOut(exp_int_rect),
            FadeOut(words),
        )

        # Make it general
        srcs = VGroup(exp1, exp2, new_int2, example_rhs)
        trgs = VGroup(
            Tex(R"e^{a{t}}", t2c=t2c),
            Tex(R"e^{a{t}}", t2c=t2c),
            Tex(R"\int^\infty_0 e^{\minus(s - a){t}} d{t}", t2c=t2c),
            Tex(R"= {1 \over {s} - a}", t2c=t2c),
        )
        example_rects = VGroup(
            SurroundingRectangle(src["1.5"], buff=0.05)
            for src in srcs
        )
        example_rects.set_stroke(TEAL, 2)
        for src, trg in zip(srcs, trgs):
            trg.move_to(src)

        self.play(Write(example_rects))
        self.wait()
        self.play(
            FadeOut(example_rects),
            LaggedStart(
                (TransformMatchingTex(src, trg, key_map={"1.5": "a"})
                for src, trg in zip(srcs, trgs)),
                lag_ratio=0.2,
                run_time=1.5,
            )
        )
        self.wait()

        # Write conclusion
        new_rhs = trgs[3]

        self.play(
            FadeOut(trgs[2]),
            new_rhs.animate.next_to(lt_def, RIGHT),
            self.change_students("thinking", "happy", "tease", look_at=new_rhs),
            morty.change("raise_right_hand", new_rhs),
        )

        big_arrow = Arrow(trgs[0].get_bottom(), new_rhs.get_bottom(), path_arc=90 * DEG, thickness=5)
        words = TexText(R"Exponentials $\longrightarrow$ Poles")
        words.next_to(big_arrow, DOWN)

        self.play(
            Write(big_arrow),
            FadeIn(words, lag_ratio=0.1),
        )
        self.look_at(words)
        self.wait(3)

        # Clear
        self.remove(self.background)
        self.remove(self.pi_creatures)


class TryADifferentInterpretation(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        self.play(
            morty.says("Let’s try an\nalternate\ninterpretation"),
            self.change_students("pondering", "thinking", "pondering", look_at=self.screen)
        )
        self.wait(2)
        self.play(self.change_students("maybe", "pondering", "thinking", look_at=self.screen))
        self.wait(4)


class AskAboutZero(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph(height=2)
        self.play(randy.says(Tex(R"s = 0 ?"), mode="erm", look_at=3 * UP + 0.5 * RIGHT))
        for _ in range(2):
            self.play(Blink(randy))
            self.wait(2)


class SeemsDumbAndPointless(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students

        self.play(LaggedStart(
            stds[0].says("Seems dumb", mode="sassy", look_at=self.screen),
            Animation(Point()),
            stds[2].animate.look_at(self.screen),
            morty.change('guilty'),
            stds[1].says(
                "And pointless!",
                mode="maybe",
                look_at=self.screen,
                bubble_direction=LEFT,
                bubble_creation_class=FadeIn
            ),
            # stds[1].animate.look_at(self.screen),
            lag_ratio=0.25
        ))
        self.wait(7)

        # Show Analytic Continuation
        words = Text("Analytic\nContinuation", font_size=60)
        words.next_to(morty, UP, LARGE_BUFF)
        words.match_x(morty.get_left())

        self.wait(2)
        self.play(
            stds[0].debubble("pondering", look_at=words),
            stds[1].debubble("pondering", look_at=words),
            stds[2].change("pondering", words),
            morty.change("raise_right_hand"),
            Write(words),
        )
        self.wait(7)


class ReferenceAnalyticContinuation(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(1000)
        words = Text("Analytic\nContinuation", font_size=60)
        words.next_to(morty, UP, LARGE_BUFF)
        words.match_x(morty.get_left())

        self.play(
            self.change_students("pondering", "pondering", 'pondering', look_at=self.screen),
            morty.change("raise_right_hand"),
            Write(words),
        )
        self.wait(4)
        self.play(self.change_students("erm", "thinking", "pondering", look_at=self.screen))
        self.wait(4)
        self.play(self.change_students("pondering", "tease", "thinking", look_at=self.screen))
        self.wait(4)


class PreviewAnalyticContinuation(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        plane = ComplexPlane((-3, 3), (-3, 3))
        plane.add_coordinate_labels(font_size=16)

        pole_values = [1 - 2.0j, -1 - 2.0j, 2 + 1j, -2 + 1j, 2.5j]
        graph = get_complex_graph(
            plane,
            lambda s: -1 * sum((np.divide(1, (s - p)) for p in pole_values))
        )
        graph.rotate(90 * DEG, about_point=plane.n2p(0))
        graph.sort_faces_back_to_front(DOWN)
        graph.set_opacity(0.8)

        frame.reorient(-1, 75, 0, (0.35, 0.49, 1.27), 7.85)
        self.add(plane, graph)
        self.play(
            ShowCreation(graph),
            frame.animate.reorient(-24, 72, 0, (0.97, 0.24, 0.97), 9.65),
            run_time=12
        )


class SimpleRect(InteractiveScene):
    def construct(self):
        rect = Rectangle(4.5, 2)
        rect.set_stroke(YELLOW, 2)

        self.play(ShowCreation(rect))
        self.wait()


class ReactingToCosineMachine(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        randy.flip()
        randy.to_edge(RIGHT)
        randy.shift(DOWN)

        for mode in ["erm", "confused"]:
            self.play(randy.change(mode))
            self.play(Blink(randy))
            self.wait(2)

        # Consider expression
        randy.body.insert_n_curves(500)
        self.play(randy.change("thinking", look_at=3 * UR))
        self.play(Blink(randy))
        self.wait()
        self.play(randy.change("tease", look_at=randy.get_top() + 2 * UP))
        self.play(Blink(randy))
        self.wait(5)


class PonderingCosineMachine(InteractiveScene):
    def construct(self):
        morty = Mortimer(height=1.5)
        morty.flip()
        morty.to_corner(UL)

        for mode in ["tease", "pondering"]:
            self.play(morty.change(mode, look_at=ORIGIN))
            self.play(Blink(morty))
            self.wait(2)


class LaplaceFourierContrast(InteractiveScene):
    def construct(self):
        # Titles and definitions
        self.add(FullScreenRectangle())

        titles = VGroup(
            Text("Laplace Transform"),
            Text("Fourier Transform"),
        )
        t2c = {"{t}": BLUE, "{s}": YELLOW, R"\xi": RED}
        formulas = VGroup(
            Tex(R"F({s}) = \int^\infty_0 f({t}) e^{\minus {s}{t}} d{t}", t2c=t2c),
            Tex(R"\hat f(\xi) = \int^\infty_{\minus \infty} f({t}) e^{\minus 2\pi i \xi {t}} d{t}", t2c=t2c),
        )
        formulas.scale(0.75)

        for title, formula, sign in zip(titles, formulas, [-1, 1]):
            for mob in [title, formula]:
                mob.next_to(1.5 * UP, UP)
                mob.set_x(sign * FRAME_WIDTH / 4)
        titles[0].align_to(titles[1], UP)

        self.play(
            LaggedStartMap(FadeIn, titles, shift=0.5 * UP, lag_ratio=0.5)
        )
        self.wait()
        self.play(
            titles.animate.to_edge(UP),
            FadeIn(formulas, 0.5 * UP)
        )
        self.wait()


class NotWhatYouWouldSee(InteractiveScene):
    def construct(self):
        morty = Mortimer().flip()
        morty.to_corner(DL)
        self.play(
            morty.says("Not the\nstandard form", mode="hesitant")
        )
        self.play(Blink(morty))
        self.wait()
        self.play(morty.debubble(mode="pondering", look_at=5 * UL))
        self.play(Blink(morty))
        self.wait()


class CosLTLogicReversal(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle().set_fill(GREY_E, 0.5))
        arrow = Tex(R"\rightarrow", font_size=120)
        cos_lt = Tex(
            R"\frac{s}{s^2 + \omega^2}",
            font_size=90,
            t2c={"s": YELLOW, R"\omega": PINK}
        )
        cos_lt.next_to(arrow, RIGHT, LARGE_BUFF)

        self.add(arrow, cos_lt)
        self.wait()
        self.play(cos_lt.animate.next_to(arrow, LEFT, LARGE_BUFF).set_anim_args(path_arc=120 * DEG), run_time=3)
        self.wait()


class CosineEqualsWhat(InteractiveScene):
    def construct(self):
        self.add(FullScreenRectangle().set_fill(GREY_E, 0.5))
        cos = Tex(R"\cos(t) = ???", font_size=90, t2c={"t": BLUE})
        q_marks = cos["???"][0]
        q_marks.scale(1.2, about_edge=UL)
        q_marks.space_out_submobjects(1.1)
        q_marks.shift(MED_SMALL_BUFF * RIGHT)
        self.add(cos)
        self.play(Write(q_marks))
        self.wait()


class OhLookAtTheTime(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students

        morty.body.insert_n_curves(500)
        # Test
        self.play(
            morty.change("raise_right_hand"),
            self.change_students("dejected", "tired", "pondering", look_at=3 * UP)
        )
        self.wait(3)
        self.play(
            morty.change("hesitant", 5 * UR),
        )
        self.play(self.change_students("tease", "well", "happy"))
        self.wait(3)
        self.play(morty.change("raise_left_hand", 5 * UR))
        self.wait(3)


class TimePassing(InteractiveScene):
    def construct(self):
        clock = Clock()
        self.add(clock)
        self.play(ClockPassesTime(clock, hours_passed=2, run_time=10))


class DerivativeRule(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"t": BLUE, "s": YELLOW}
        kw = dict(t2c=t2c, font_size=96)
        in_texs = VGroup(
            Tex(R"x(t)", **kw),
            Tex(R"x'(t)", **kw),
        )
        out_texs = VGroup(
            Tex(R"X(s)", **kw),
            Tex(R"s X(s) - x(0)", **kw),
        )
        arrows = Vector(3 * RIGHT, thickness=7).replicate(2)

        in_texs.arrange(DOWN, buff=2)
        in_texs.to_edge(LEFT, buff=2)

        for in_tex, arrow, output in zip(in_texs, arrows, out_texs):
            arrow.next_to(in_tex, RIGHT)
            output.next_to(arrow, RIGHT)
            fancy_L = Tex(R"\mathcal{L}")
            fancy_L.next_to(arrow, UP, SMALL_BUFF)
            arrow.add(fancy_L)

        self.add(in_texs[0], arrows[0], out_texs[0])
        self.wait()
        self.play(
            TransformMatchingTex(in_texs[0].copy(), in_texs[1]),
            TransformMatchingTex(out_texs[0].copy(), out_texs[1]),
            TransformFromCopy(*arrows),
            run_time=1.5
        )
        self.wait()
        SpeechBubble


class EndScreen(SideScrollEndScreen):
    pass