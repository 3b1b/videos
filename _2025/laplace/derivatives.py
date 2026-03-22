from manim_imports_ext import *
from _2025.laplace.integration import get_complex_graph


tex_to_color = {
    "{t}": BLUE,
    "{s}": YELLOW,
}


class ForcedOscillatorEquation(InteractiveScene):
    def construct(self):
        x_colors = color_gradient([TEAL, RED], 3, interp_by_hsl=True)
        t2c = {
            "x(t)": x_colors[0],
            "x'(t)": x_colors[1],
            "x''(t)": x_colors[2],
            "x_0": x_colors[0],
            "v_0": x_colors[1],
            R"\omega": PINK,
            "{s}": YELLOW,
        }
        ode = Tex(R"m x''(t) + \mu x'(t) + k x(t) = F_0 \cos(\omega t)", t2c=t2c)
        ode.to_edge(UP)

        self.add(ode)

        # Comment on third force
        rect = SurroundingRectangle(ode[R"F_0 \cos(\omega t)"], buff=0.2)
        rect.set_stroke(YELLOW, 2)

        self.play(ShowCreation(rect))
        self.wait()
        self.play(rect.animate.surround(ode[R"\omega"], buff=0.1))
        self.wait()
        self.play(FadeOut(rect))
        self.wait()


class MoveAroundPolesSeeDynamics(InteractiveScene):
    def construct(self):
        # Add s_plane (and graph)
        frame = self.frame
        s_plane = ComplexPlane((-3, 3), (-3, 3))
        s_plane.add_coordinate_labels(font_size=20)

        s_trackers = Group(
            ComplexValueTracker(+2j),
            ComplexValueTracker(-2j),
            ComplexValueTracker(+1.0j - 0.2),
            ComplexValueTracker(-1.0j - 0.2),
            ComplexValueTracker(-1.0),
        )

        t_tracker = ValueTracker(0)
        t_tracker.add_updater(lambda m, dt: m.increment_value(dt))
        get_t = t_tracker.get_value

        s_points = Group(TrueDot(), GlowDot()).replicate(len(s_trackers))
        s_points.set_color(YELLOW)
        for point, s_tracker in zip(s_points, s_trackers):
            point.s_tracker = s_tracker
            point.add_updater(lambda m: m.move_to(s_plane.n2p(m.s_tracker.get_value())))

        frame.reorient(-14, 82, 0, (-4.89, 2.68, 3.02), 10.32)
        self.add(t_tracker)

        # Add title
        arrow = Vector(3 * RIGHT, thickness=6)
        kw = dict(
            t2c={"t": BLUE, "s": YELLOW},
            font_size=72
        )
        title = VGroup(Tex(R"f(t)", **kw), arrow, Tex(R"F(s)", **kw))
        title.arrange(RIGHT)
        title.to_edge(UP)
        lt_label = Tex(R"\mathcal{L}")
        lt_label.next_to(arrow, UP, buff=0)
        arrow.add(lt_label)
        title.fix_in_frame()

        self.add(title[0])

        # Add time graph
        axes = Axes((0, 15), (-2, 2), width=6, height=3)
        axes.to_edge(LEFT)
        axes.shift(2 * DOWN)
        time_graph = axes.get_graph(self.get_time_func(s_trackers, amp=0.4))
        time_graph.set_stroke(BLUE, 2)
        axes.fix_in_frame()
        time_graph.fix_in_frame()

        self.add(axes, time_graph)

        # Add mass
        amp_factor_tracker = ValueTracker(2.0)

        number_line = NumberLine((-2, 2), width=6)
        number_line.next_to(axes, UP, LARGE_BUFF)
        number_line.add_numbers(font_size=16)
        number_line.fix_in_frame()

        tip = ArrowTip(angle=-90 * DEG)
        tip.stretch(0.25, 0)
        tip.func = self.get_time_func(s_trackers, amp=0.4)
        tip.add_updater(lambda m: m.move_to(number_line.n2p(m.func(get_t())), DOWN))
        tip.fix_in_frame()

        mass = Square(side_length=0.5)
        mass.set_fill(BLUE_E, 1)
        mass.set_stroke(WHITE, 2)
        mass.always.next_to(tip, UP)
        mass.fix_in_frame()
        mass_ghost = mass.copy()
        mass_ghost.set_opacity(0)

        v_vect = always_redraw(lambda: Arrow(
            mass.get_center(),
            mass.get_center() + np.clip(5 * (mass.get_center() - mass_ghost.get_center()), -1, 1),
            buff=0,
            thickness=4,
            fill_color=RED
        ).fix_in_frame())

        self.add(number_line, tip, mass, v_vect, mass_ghost)
        self.wait(5)

        # Show graph
        epsilon = 1e-5

        def get_graph():
            graph = get_complex_graph(
                s_plane,
                self.get_s_func(s_trackers),
                # resolution=(101, 101)
                # resolution=(31, 31)
            )
            for line in graph[1]:
                if line.get_z(OUT) > 1e3:
                    line.set_stroke(opacity=0)
            return graph

        def update_graph(graph):
            graph.become(get_graph())

        graph = get_graph()

        self.play(
            FadeIn(s_plane, lag_ratio=0.01),
            FadeIn(s_points),
            FadeIn(graph[0]),
            Write(graph[1]),
            Write(title[1]),
            FadeIn(title[2], RIGHT)
        )
        self.play(
            frame.animate.reorient(0, 0, 0, (-5.0, 1.08, 0.0), 9.62).set_field_of_view(10 * DEG),
            run_time=3
        )
        self.wait(8)

        # Move s
        t_tracker.set_value(0)
        s_trackers.remove(*s_trackers[2:])
        s_trackers[1].add_updater(lambda m: m.set_value(s_trackers[0].get_value().conjugate()))
        s_tracker = s_trackers[0]
        s_tracker.set_value(2j)

        amp_tracker = ValueTracker(1)

        time_graph.add_updater(lambda m: m.match_points(axes.get_graph(self.get_time_func(s_trackers))))
        time_graph.set_clip_plane(DOWN, 0)

        def new_func(t):
            return amp_tracker.get_value() * math.cos(2 * t)

        tip.func = new_func

        update_graph(graph)
        self.add(*s_trackers)
        self.remove(s_points[2:])
        self.wait(10)

        amp_tracker.add_updater(lambda m: m.set_value(0.992 * m.get_value()))
        self.add(amp_tracker)
        self.play(
            s_tracker.animate.set_value(-0.5 + 2j),
            UpdateFromFunc(graph, update_graph),
            run_time=2
        )
        self.wait(4)

        amp_tracker.clear_updaters()
        amp_tracker.add_updater(lambda m: m.set_value(1.005 * m.get_value()))
        self.play(
            s_tracker.animate.set_value(0.2 + 2j),
            UpdateFromFunc(graph, update_graph),
            run_time=3
        )
        self.wait(15)

    def get_time_func(self, s_trackers, amp=0.5):
        def time_func(t):
            return amp * np.sum([np.exp(st.get_value() * t) for st in s_trackers]).real

        return time_func

    def get_s_func(self, s_trackers, epsilon=1e-5, amp=0.5):
        def s_func(s):
            s += epsilon
            return amp * np.sum([1.0 / (s - st.get_value() + epsilon) for st in s_trackers])

        return s_func


class DerivativeFormula(InteractiveScene):
    tex_config = dict(t2c=tex_to_color, font_size=72)

    def construct(self):
        # Set up commutative diagram
        kw = self.tex_config
        ft, Fs, dft, sFs = terms = VGroup(
            Tex(R"f({t})", **kw),
            Tex(R"F({s})", **kw),
            Tex(R"f'({t})", **kw),
            Tex(R"{s}F({s}) - f(0)", **kw),
        )
        terms.arrange_in_grid(
            h_buff=2.0,
            v_buff=3.0,
            aligned_edge=LEFT,
            fill_rows_first=False
        )
        terms.to_edge(UP, buff=LARGE_BUFF)
        terms.shift(RIGHT)

        dist = get_norm(ft.get_bottom() - Fs.get_top()) - MED_LARGE_BUFF
        down_arrow = Vector(dist * DOWN, thickness=6)
        arrow_kw = dict(thickness=6)
        lt_arrows = VGroup(
            down_arrow.copy().next_to(term, DOWN)
            for term in [ft, dft]
        )
        lt_arrows.set_fill(GREY_A)
        for arrow in lt_arrows:
            self.add_arrow_label(arrow, R"\mathcal{L}", RIGHT)

        deriv_arrow = Arrow(ft, dft, thickness=6)
        s_mult_arrow = Arrow(Fs, sFs, thickness=6)

        self.add_arrow_label(deriv_arrow, R"d / d{t}", UP)
        self.add_arrow_label(s_mult_arrow, R"\times {s}", UP)

        # Add terms
        L_df = Tex(R"\mathcal{L}\big\{f'({t})\big\}", **kw)
        L_df.move_to(sFs, LEFT)
        equals = Tex(R"=", font_size=72).rotate(90 * DEG)
        equals.next_to(sFs["{s}F({s})"], DOWN)

        self.play(Write(ft), run_time=1)
        self.wait()
        self.play(LaggedStart(
            self.grow_arrow(deriv_arrow),
            TransformMatchingTex(ft.copy(), dft, run_time=1, path_arc=45 * DEG),
            lag_ratio=0.25
        ))
        self.wait()
        self.play(LaggedStart(
            self.grow_arrow(lt_arrows[1]),
            TransformMatchingTex(
                dft.copy(),
                L_df,
                run_time=1.5,
                path_arc=45 * DEG,
                matched_keys=[R"f'({t})"]
            ),
            lag_ratio=0.25
        ))
        self.wait()
        self.play(LaggedStart(
            self.grow_arrow(lt_arrows[0]),
            TransformFromCopy(ft.copy(), Fs, run_time=1.5, path_arc=45 * DEG),
            lag_ratio=0.25
        ))
        self.wait()
        self.play(LaggedStart(
            self.grow_arrow(s_mult_arrow),
            FadeTransform(Fs.copy(), sFs[1:5]),
            TransformFromCopy(Fs[2], sFs[0]),
            L_df.animate.scale(0.7).next_to(equals, DOWN),
            Write(equals),
            lag_ratio=0.1
        ))
        self.wait()

        # Correction
        almost = Text("Almost...")
        almost.set_color(RED)
        almost.next_to(equals, RIGHT, MED_SMALL_BUFF)

        self.play(FadeIn(almost, lag_ratio=0.1))
        self.wait()
        self.play(LaggedStart(
            TransformFromCopy(ft, sFs["f(0)"][0], path_arc=-45 * DEG, run_time=2),
            Write(sFs["-"]),
        ))
        self.play(
            FadeOut(almost),
            VGroup(equals, L_df).animate.space_out_submobjects(1.2).next_to(sFs, DOWN)
        )
        self.wait()

        # Highlight parts
        rect = SurroundingRectangle(deriv_arrow.label)
        rect.set_stroke(TEAL, 4)
        mid_lt_arrow = lt_arrows[0].copy().match_x(deriv_arrow)
        self.add_arrow_label(mid_lt_arrow, R"\mathcal{L}", RIGHT)
        mid_arrow_group = VGroup(mid_lt_arrow, mid_lt_arrow.label)
        mid_arrow_group.shift(0.25 * UP)
        mid_arrow_group.set_fill(opacity=0.5)

        self.play(ShowCreation(rect))
        self.wait()
        self.play(
            rect.animate.surround(s_mult_arrow.label),
            self.grow_arrow(mid_lt_arrow),
        )
        self.play(
            FadeOut(mid_arrow_group),
            FadeOut(rect),
        )
        self.wait()

        # Comment on -f(0)
        frame = self.frame
        randy = Randolph().flip()
        randy.next_to(sFs, DR, LARGE_BUFF)
        randy.shift(0.5 * LEFT)
        morty = Mortimer().flip()
        morty.next_to(randy, LEFT, buff=2.0)
        morty.body.insert_n_curves(500)
        quirk = sFs["- f(0)"][0]
        quirk_rect = SurroundingRectangle(quirk)
        quirk_rect.set_stroke(RED, 5)
        ic_words = Text("Initial\nCondition", font_size=72)
        ic_words.next_to(quirk, UR, LARGE_BUFF)
        ic_arrow = Arrow(ic_words.get_bottom(), quirk.get_right(), path_arc=-90 * DEG, thickness=6)

        self.play(LaggedStart(
            frame.animate.reorient(0, 0, 0, (2.0, -0.79, 0.0), 9.49),
            FadeOut(equals),
            FadeOut(L_df),
            ShowCreation(quirk_rect),
            VFadeIn(randy),
            randy.change("angry", quirk),
            lag_ratio=0.1
        ))
        self.play(Blink(randy))
        self.wait()
        self.play(
            VFadeIn(morty),
            morty.change("tease", randy.eyes),
            randy.change('hesitant', morty.eyes),
        )
        self.play(Blink(morty))
        self.wait()
        self.play(
            morty.change("raise_right_hand", ic_words),
            randy.change("pondering", ic_words),
            FadeIn(ic_words, lag_ratio=0.1, run_time=1),
            Write(ic_arrow)
        )
        self.wait()

        # Ask why
        bubble = randy.get_bubble("Why?", SpeechBubble, direction=LEFT)
        bubble.shift(0.5 * LEFT)

        exp_propety_frame = Rectangle(16, 9).replace(terms, 1)
        exp_propety_frame.next_to(terms, RIGHT, buff=1.0)
        exp_propety_frame.set_stroke(BLUE, 0)

        self.play(LaggedStart(
            randy.change("maybe"),
            Write(bubble),
            morty.change("thinking", lt_arrows),
            FadeOut(ic_words),
            FadeOut(ic_arrow),
            FadeOut(quirk_rect),
            lag_ratio=0.1
        ))
        self.play(Blink(randy))
        self.play(
            Group(frame, randy, morty, bubble).animate.scale(1.25, about_edge=UL),
        )
        self.play(
            randy.change("raise_left_hand", exp_propety_frame),
            morty.change('pondering', exp_propety_frame),
            FadeIn(exp_propety_frame),
            FadeOut(bubble),
        )
        self.wait()
        self.play(
            ShowCreationThenFadeOut(quirk_rect),
            morty.animate.look_at(quirk),
            randy.change('sassy', quirk),
        )
        self.play(Blink(randy))
        self.play(Blink(morty))
        self.wait()

        # Reset
        self.play(FadeOut(exp_propety_frame))
        self.play(
            frame.animate.reorient(0, 0, 0, (-1.0, -0.61, 0.0), 9.39),
            LaggedStartMap(FadeOut, VGroup(morty, randy), shift=DOWN),
            run_time=2
        )

        # Substitute in e^(at)
        fade_group = VGroup(
            deriv_arrow, deriv_arrow.label,
            s_mult_arrow, s_mult_arrow.label,
            lt_arrows[1], lt_arrows[1].label,
            dft,
            sFs
        )

        eat_terms = VGroup(
            Tex(R"f({t}) = e^{a{t}}", **kw),
            Tex(R"F({s}) = {1 \over {s} - a}", **kw),
            Tex(R"f'({t}) = a \cdot e^{a{t}}", **kw),
            Tex(R"{a \over {s} - a}", **kw),
        )
        for term, eat_term, corner in zip(terms, eat_terms, [DR, UR, DL, UL]):
            eat_term.move_to(term, corner)
        eat_terms[3].align_to(eat_terms[1], DOWN)

        self.remove(ft)
        self.play(
            TransformFromCopy(ft, eat_terms[0][:len(ft)]),
            Write(eat_terms[0][len(ft):]),
            s_mult_arrow.animate.scale(0.8, about_edge=UP),
            fade_group.animate.set_fill(opacity=0.2),
        )
        self.wait()
        self.play(
            TransformMatchingShapes(
                eat_terms[0][len(ft):].copy(),
                eat_terms[1][len(Fs):],
                path_arc=-45 * DEG,
                run_time=1.5,
            ),
            FadeTransform(Fs, eat_terms[1][:len(Fs)]),
            VGroup(s_mult_arrow, s_mult_arrow.label).animate.shift(0.4 * DOWN)
        )
        self.add(eat_terms[1])
        self.wait()
        self.play(
            VGroup(deriv_arrow, deriv_arrow.label, dft).animate.set_fill(opacity=1)
        )
        self.wait()
        self.remove(dft)
        self.play(
            TransformFromCopy(eat_terms[0][-3:], eat_terms[2][-3:], path_arc=-45 * DEG, run_time=1.5),
            TransformFromCopy(eat_terms[0][-4], eat_terms[2][-6], path_arc=-45 * DEG, run_time=1.5),
            TransformFromCopy(dft, eat_terms[2][:len(dft)])
        )
        self.play(
            Write(eat_terms[2][-4]),
            TransformFromCopy(eat_terms[2][-2], eat_terms[2][-5], path_arc=45 * DEG),
        )
        self.wait()

        # Show transform of right hand side
        left_group_copy = VGroup(
            eat_terms[0]["e^{a{t}}"][0],
            lt_arrows[0],
            lt_arrows[0].label,
            eat_terms[1][R"{1 \over {s} - a}"][0]
        ).copy()
        a_dot_copy = eat_terms[2][R"a \cdot"][0].copy()
        a_dot_rect = SurroundingRectangle(a_dot_copy).set_stroke(TEAL, 2)

        shift_value = eat_terms[2]["e^{a{t}}"].get_center() - left_group_copy[0].get_center()
        self.play(
            left_group_copy.animate.shift(shift_value).set_anim_args(path_arc=30 * DEG),
            sFs.animate.shift(3.0 * DOWN),
            run_time=1.5
        )
        self.play(ShowCreation(a_dot_rect))
        self.play(
            a_dot_copy.animate.next_to(left_group_copy[-1][1], LEFT),
            MaintainPositionRelativeTo(a_dot_rect, a_dot_copy),
        )
        self.wait()
        self.play(
            FadeOut(left_group_copy[:3]),
            ReplacementTransform(left_group_copy[3][1:], eat_terms[3][1:]),
            ReplacementTransform(a_dot_copy, eat_terms[3][:1]),
            FadeOut(left_group_copy[3][0], LEFT),
            FadeOut(a_dot_rect, LEFT),
            VGroup(lt_arrows[1], lt_arrows[1].label).animate.set_fill(opacity=1),
        )
        self.wait()

        # Differentiation to multiplication
        randy = Randolph(height=3)
        randy.next_to(eat_terms[1], DOWN, buff=LARGE_BUFF)
        randy.shift(2 * LEFT)
        mult_a_arrow = Arrow(
            eat_terms[1][-5:].get_bottom(),
            eat_terms[3].get_bottom(),
            path_arc=120 * DEG,
            thickness=6
        )
        rect = SurroundingRectangle(deriv_arrow.label)
        rect.set_stroke(TEAL, 4)
        mult_word = Text("Multiplication")
        times_a = Tex(R"\times a", **kw)
        for mob in [mult_word, times_a]:
            mob.next_to(mult_a_arrow, DOWN)

        self.play(
            frame.animate.reorient(0, 0, 0, (-1.1, -1.43, 0.0), 11.38),
            VFadeIn(randy),
            randy.change("shruggie", eat_terms[3]),
        )
        self.play(Blink(randy))
        self.wait()
        self.play(ShowCreation(rect))
        self.play(LaggedStart(
            randy.change("pondering", mult_word),
            rect.animate.surround(mult_word),
            Write(mult_word),
            Write(mult_a_arrow),
            lag_ratio=0.2
        ))
        self.play(FadeOut(rect))
        self.wait()
        self.play(randy.change("erm", mult_word))
        self.play(FadeTransformPieces(mult_word, times_a))
        self.play(Blink(randy))
        self.wait()

        # Contrast against multiplication by s
        equals.next_to(eat_terms[3], DOWN, MED_LARGE_BUFF)
        q_marks = Tex(R"???", font_size=60).replicate(2)
        q_marks.set_color(RED)
        q_marks[0].next_to(s_mult_arrow.label, UP)
        q_marks[1].next_to(equals, RIGHT)

        self.play(LaggedStart(
            randy.animate.change_mode("horrified").shift(0.25 * DL).set_opacity(0),
            # FadeOut(mult_a_arrow),
            ReplacementTransform(mult_a_arrow, s_mult_arrow),
            # FadeOut(times_a),
            ReplacementTransform(times_a, s_mult_arrow.label),
            Animation(Point()),
            VGroup(s_mult_arrow, s_mult_arrow.label).animate.set_fill(opacity=1),
            Write(equals),
            sFs.animate.set_fill(opacity=1).next_to(equals, DOWN, MED_LARGE_BUFF),
            Write(q_marks),
        ))
        self.remove(randy)
        self.wait()

        # Show algebra
        added_frac = Tex(R"+ {{s} - a \over {s} - a}", **kw)
        minus_one = Tex(R"-1", **kw)
        added_frac.next_to(eat_terms[3], RIGHT, SMALL_BUFF)
        minus_one.next_to(added_frac[R"\over"], RIGHT, MED_SMALL_BUFF)

        added_frac_rect = SurroundingRectangle(added_frac, SMALL_BUFF)
        added_frac_rect.set_stroke(BLUE, 1)

        plus_one = Tex(R"+ 1", font_size=60)
        plus_one.set_color(BLUE)
        plus_one.next_to(added_frac_rect, DOWN)

        cover_rect = BackgroundRectangle(VGroup(equals, sFs), buff=MED_SMALL_BUFF)
        cover_rect.set_fill(BLACK, 0.8)

        combined_fraction = Tex(R"{a + {s} - a \over {s} - a}", **kw)
        clean_combined_fraction = Tex(R"{{s} \over {s} - a}", **kw)
        for mob in [combined_fraction, clean_combined_fraction]:
            mob.move_to(eat_terms[3], LEFT)

        self.play(
            FadeIn(cover_rect),
            FadeIn(added_frac_rect),
            Write(added_frac["+"][0]),
            *(
                TransformFromCopy(eat_terms[3][tex], added_frac[tex])
                for tex in [R"{s} - a", R"\over"]
            ),
        )
        self.play(Write(plus_one))
        self.wait()
        self.play(Write(minus_one))
        self.wait()
        self.remove(eat_terms[3], added_frac)
        self.play(
            TransformFromCopy(eat_terms[3]["a"][0], combined_fraction["a"][0]),
            TransformFromCopy(added_frac["+"][0], combined_fraction["+"][0]),
            TransformFromCopy(added_frac["{s} - a"][0], combined_fraction["{s} - a"][0]),
            TransformFromCopy(eat_terms[3][R"\over"][0], combined_fraction[R"\over"][0]),
            TransformFromCopy(added_frac[R"\over"][0], combined_fraction[R"\over"][0]),
            TransformFromCopy(eat_terms[3][R"{s} - a"][0], combined_fraction[R"{s} - a"][1]),
            TransformFromCopy(added_frac[R"{s} - a"][1], combined_fraction[R"{s} - a"][1]),
            added_frac_rect.animate.surround(combined_fraction),
            minus_one.animate.next_to(combined_fraction, RIGHT),
            FadeOut(plus_one),
            run_time=1.5
        )
        self.wait()
        self.play(
            TransformMatchingTex(combined_fraction, clean_combined_fraction, run_time=1),
            added_frac_rect.animate.surround(clean_combined_fraction),
            minus_one.animate.next_to(clean_combined_fraction, RIGHT)
        )
        self.wait()

        # Emphasize how this matches the rule
        pole = eat_terms[1][-5:]
        pole_rect = SurroundingRectangle(pole)
        pole_rect.match_style(added_frac_rect)

        self.play(
            FadeOut(added_frac_rect),
            FadeOut(q_marks[0], shift=0.25 * RIGHT, lag_ratio=0.1),
            ShowCreation(pole_rect),
        )
        self.play(
            pole_rect.animate.surround(clean_combined_fraction, SMALL_BUFF),
            TransformFromCopy(pole[1:], clean_combined_fraction[1:], path_arc=10 * DEG),
            TransformFromCopy(s_mult_arrow.label[1], clean_combined_fraction[0], path_arc=-10 * DEG),
            run_time=2
        )
        self.play(FadeOut(pole_rect, run_time=0.5))
        self.wait(0.5)

        # Emphasize minus f(0)
        low_eq_group = VGroup(equals, sFs)

        self.add(low_eq_group, cover_rect)
        self.play(
            FadeOut(cover_rect),
            low_eq_group.animate.match_x(VGroup(clean_combined_fraction, minus_one)),
            FadeOut(q_marks[1]),
        )

        quirk_rects = VGroup(
            SurroundingRectangle(minus_one, buff=SMALL_BUFF),
            SurroundingRectangle(sFs["- f(0)"][0], buff=SMALL_BUFF),
        )
        quirk_rects.set_stroke(RED, 2)
        minus_e_zero = Tex(R" - e^{a0}", **kw)
        minus_e_zero["0"].set_color(BLUE)
        minus_e_zero.next_to(quirk_rects[1], DOWN)

        self.play(ShowCreation(quirk_rects, lag_ratio=0))
        self.wait()
        self.play(
            TransformFromCopy(eat_terms[0]["= e^{a{t}}"][0], minus_e_zero, run_time=2)
        )
        self.wait()
        self.play(LaggedStartMap(FadeOut, VGroup(*quirk_rects, minus_e_zero), lag_ratio=0.5, run_time=1))

        # Maybe show more generally.
        kw["t2c"].update({"c_n": TEAL})
        exp_sum_terms = VGroup(
            Tex(R"f({t}) = \sum_{n=1}^N c_n e^{a_n {t}}", **kw),
            Tex(R"F({s}) = \sum_{n=1}^N {c_n \over {s} - a_n}", **kw),
            Tex(R"f'({t}) = \sum_{n=1}^N c_n \cdot a_n e^{a_n {t}}", **kw),
            Tex(R"\sum_{n=1}^N \left( c_n {{s} \over {s} - a_n} - c_n \right)", **kw),
        )
        for exp_sum_term, eat_term, corner in zip(exp_sum_terms, eat_terms, [DR, UR, DL, UL]):
            exp_sum_term.move_to(eat_term, corner)
        eat_terms[3].align_to(eat_terms[1], DOWN)

        self.play(
            LaggedStart(
                (TransformMatchingTex(eat_term, exp_sum_term)
                for eat_term, exp_sum_term in zip(eat_terms[:3], exp_sum_terms)),
                lag_ratio=0.2,
            ),
            FadeTransform(VGroup(clean_combined_fraction, minus_one), exp_sum_terms[-1], time_span=(1, 2)),
            LaggedStart(
                *(
                    VGroup(arrow, arrow.label).animate.shift(vect)
                    for arrow, vect in [
                        (lt_arrows[0], LEFT),
                        (deriv_arrow, 0.5 * UP),
                        (lt_arrows[1], RIGHT),
                        (s_mult_arrow, 0.5 * DOWN),
                    ]
                ),
                lag_ratio=0.5,
            ),
            self.frame.animate.reorient(0, 0, 0, (-0.15, -0.39, 0.0), 13.58),
            low_eq_group.animate.shift(DR),
            run_time=3
        )
        self.wait()

    def add_arrow_label(self, arrow, label_tex, direction, buff=SMALL_BUFF):
        arrow.label = Tex(label_tex, **self.tex_config)
        arrow.label.next_to(arrow, direction, buff=buff)

    def grow_arrow(self, arrow, run_time=1):
        """
        Assumes the arrow has a .label attribute
        """
        return AnimationGroup(
            GrowArrow(arrow),
            FadeIn(arrow.label, shift=0.25 * arrow.get_vector()),
            run_time=run_time
        )


class PreviewStrategy(InteractiveScene):
    def construct(self):
        # Set up terms
        rect = Rectangle(6, 4)
        ode, lt_ode, lt_ans, ans = terms = VGroup(
            Text("Differential\nEquation"),
            Text("Transformed\nEquation"),
            Text("Transformed\nSolution"),
            Text("Solution"),
        )
        VGroup(lt_ode, lt_ans).set_color(YELLOW)
        for term, corner in zip(terms, [UL, DL, DR, UR]):
            term.move_to(rect.get_corner(corner))

        # Set up arrows
        lt_arrow = self.get_lt_arrow(ode, lt_ode, buff=MED_SMALL_BUFF)
        solve_arrow = self.get_lt_arrow(lt_ode, lt_ans, label_tex=R"\substack{\text{Solve} \\ \text{(Algebraically)}}")
        solve_arrow[1][:5].scale(1.5, about_edge=DOWN).shift(0.1 * UP)
        inv_lt_arrow = self.get_lt_arrow(lt_ans, ans, label_tex=R"\mathcal{L}^{-1}", buff=MED_SMALL_BUFF)

        arrows = VGroup(lt_arrow, solve_arrow, inv_lt_arrow)

        # Show creation of terms
        self.add(ode)
        self.wait()
        self.play(
            self.grow_lt_arrow(lt_arrow),
            TransformMatchingStrings(ode.copy(), lt_ode, key_map={"Differential": "Transformed"}),
            run_time=1.5
        )
        self.wait()
        self.play(
            self.grow_lt_arrow(solve_arrow),
            FadeTransform(lt_ode.copy(), lt_ans),
            run_time=1.5
        )
        self.wait()
        self.play(
            self.grow_lt_arrow(inv_lt_arrow),
            FadeTransform(lt_ans["Solution"][0].copy(), ans),
            run_time=1.5
        )
        self.wait()

        # Add domain backgrounds
        time_domain = FullScreenRectangle()
        time_domain.stretch(0.55, 1, about_edge=UP)
        time_domain.set_stroke(BLUE, 3)
        time_domain.set_fill(opacity=0)
        s_domain = FullScreenRectangle()
        s_domain.stretch(0.45, 1, about_edge=DOWN)
        s_domain.set_stroke(YELLOW, 3)
        s_domain.set_fill(opacity=0)

        time_label = Text("Time domain")
        s_label = Text("s domain")
        s_label.set_fill(YELLOW)

        for label, domain in [(time_label, time_domain), (s_label, s_domain)]:
            label.next_to(domain.get_corner(UL), DR)

        self.play(LaggedStart(
            FadeIn(time_domain),
            FadeIn(time_label),
            FadeIn(s_domain),
            FadeIn(s_label),
        ))
        self.wait()

    def get_lt_arrow(self, m1, m2, thickness=4, label_font_size=36, buff=0.15, label_tex=R"\mathcal{L}"):
        arrow = Arrow(m1, m2, buff=buff, thickness=thickness)
        arrow.set_fill(border_width=2)
        label = Tex(label_tex, font_size=label_font_size)
        label.move_to(arrow.get_center())
        shift_dir = rotate_vector(normalize(arrow.get_vector()), 90 * DEG)
        label.shift(1.25 * label.get_height() * shift_dir)
        return VGroup(arrow, label)

    def grow_lt_arrow(self, lt_arrow):
        return AnimationGroup(
            GrowArrow(lt_arrow[0]),
            FadeIn(lt_arrow[1], shift=0.25 * lt_arrow[0].get_vector())
        )


class WalkThroughEquationSolution(PreviewStrategy):
    def construct(self):
        # Add ode
        x_colors, t2c = self.get_x_colors_and_t2c()
        ode = Tex(R"m x''(t) + \mu x'(t) + k x(t) = F_0 \cos(\omega t)", t2c=t2c)
        ode.to_edge(UP)

        xt = ode["x(t)"][0]
        dxt = ode["x'(t)"][0]
        ddxt = ode["x''(t)"][0]
        xt_group = VGroup(xt, dxt, ddxt)

        self.add(ode)

        # Transform of the full equation
        ode_lt_lhs = Tex(R"""
            m\Big({s}^2 X({s}) - {s} x_0 - v_0 \Big)
            + \mu \Big( {s} X({s}) - x_0 \Big)
            + k X({s})
        """, t2c=t2c)
        factored_ode_lt_lhs = Tex(R"""
            X({s})\big(m{s}^2 + \mu{s} + k\big)
            - m v_0 - (m{s} + \mu)x_0
        """, t2c=t2c)
        ode_lt_lhs.next_to(xt_group, DOWN, buff=1.5)
        factored_ode_lt_lhs.next_to(ode_lt_lhs, DOWN, buff=1.5)

        x_lt = ode_lt_lhs[R"X({s})"][-1]
        dx_lt = ode_lt_lhs[R"{s} X({s}) - x_0"][-1]
        ddx_lt = ode_lt_lhs[R"{s}^2 X({s}) - {s} x_0 - v_0"][0]
        x_lt_parts = VGroup(x_lt, dx_lt, ddx_lt)

        # Show each transform
        for part in x_lt_parts:
            part.save_state()
        x_lt_parts[:2].match_x(xt_group)

        xt_rect = SurroundingRectangle(xt, buff=0.05)
        dxt_rect = SurroundingRectangle(dxt, buff=0.05)
        ddxt_rect = SurroundingRectangle(ddxt, buff=0.05)
        xt_rects = VGroup(xt_rect, dxt_rect, ddxt_rect)
        for rect, color in zip(xt_rects, x_colors):
            rect.set_stroke(color, width=2)

        xt_arrow = self.get_lt_arrow(xt_rect, x_lt)
        dxt_arrow = self.get_lt_arrow(dxt_rect, dx_lt)
        ddx_arrow = self.get_lt_arrow(ddxt_rect, ddx_lt)

        self.play(ShowCreation(xt_rect))
        self.play(
            self.grow_lt_arrow(xt_arrow),
            FadeTransform(xt.copy(), x_lt)
        )
        self.wait()
        self.play(ShowCreation(dxt_rect))
        self.play(
            self.grow_lt_arrow(dxt_arrow),
            FadeTransform(dxt.copy(), dx_lt)
        )
        self.wait()

        # Ask about L{x''(t)}
        ddx_lt_rect = SurroundingRectangle(ddx_lt, buff=SMALL_BUFF)
        ddx_lt_rect.set_stroke(RED, 1)

        self.play(ShowCreation(ddxt_rect)),
        self.play(LaggedStart(
            self.grow_lt_arrow(ddx_arrow),
            TransformFromCopy(ddxt_rect, ddx_lt_rect),
            Restore(x_lt),
            Transform(xt_arrow, self.get_lt_arrow(xt_rect, x_lt.saved_state)),
            Restore(dx_lt),
            Transform(dxt_arrow, self.get_lt_arrow(dxt_rect, dx_lt.saved_state)),
        ))
        self.wait()

        # Show second derivative rule
        ddx_lt_lhs = Tex(R"\mathcal{L}\Big\{x''(t)\Big\}", t2c=t2c)
        ddx_lt_rhss = VGroup(
            Tex(R"= {s} \mathcal{L}\Big\{x'(t)\Big\} - x'(0)", t2c=t2c),
            Tex(R"= {s} \mathcal{L}\Big\{x'(t)\Big\} - v_0", t2c=t2c),
            Tex(R"= {s} \Big({s}X({s}) - x_0 \Big) - v_0", t2c=t2c),
            Tex(R"= {s}^2 X({s}) - {s} x_0 - v_0", t2c=t2c),
        )
        ddx_lt_lhs.to_edge(DOWN, buff=1.5)
        ddx_lt_lhs.to_edge(LEFT, buff=2.0)
        for rhs in ddx_lt_rhss:
            rhs.next_to(ddx_lt_lhs, RIGHT)
        for rhs in ddx_lt_rhss[2:]:
            rhs.next_to(ddx_lt_rhss[1], RIGHT)

        v0_rect = SurroundingRectangle(ddx_lt_rhss[0][R"x'(0)"])
        v0_rect.set_stroke(x_colors[1], 2)

        self.play(
            TransformFromCopy(ddx_arrow[1], ddx_lt_lhs[0]),
            TransformFromCopy(ddxt, ddx_lt_lhs[R"x''(t)"][0]),
            Write(ddx_lt_lhs[R"\Big\{"]),
            Write(ddx_lt_lhs[R"\Big\}"]),
        )
        self.wait()
        self.play(
            TransformMatchingTex(ddx_lt_lhs.copy(), ddx_lt_rhss[0], path_arc=30 * DEG, run_time=1)
        )
        self.wait()
        self.play(ShowCreation(v0_rect))
        self.play(
            TransformMatchingTex(
                ddx_lt_rhss[0],
                ddx_lt_rhss[1],
                matched_keys=[R"= {s} \mathcal{L}\Big\{x'(t)\Big\} - "],
                key_map={R"x'(0)": R"v_0"},
                run_time=1,
            ),
            v0_rect.animate.surround(ddx_lt_rhss[1][R"v_0"])
        )
        self.play(FadeOut(v0_rect))
        self.wait()
        self.play(
            TransformMatchingTex(
                ddx_lt_rhss[1].copy(),
                ddx_lt_rhss[2],
                key_map={R"\mathcal{L}\Big\{x'(t)\Big\}": R"\Big({s}X({s}) - x_0 \Big)"},
                matched_keys=[R"- v_0"],
                run_time=1.5,
                path_arc=30 * DEG,
            ),
        )
        self.wait()
        self.play(
            TransformMatchingTex(
                ddx_lt_rhss[2],
                ddx_lt_rhss[3],
                matched_keys=[R"- v_0", "x_0", "X({s})"],
                run_time=1.5,
                path_arc=30 * DEG,
            ),
        )
        self.wait()
        self.play(
            FadeOut(ddx_lt_lhs),
            FadeOut(ddx_lt_rhss[1]),
            FadeOut(ddx_lt_rhss[3][0]),
            FadeTransform(ddx_lt_rhss[3][1:], ddx_lt),
            run_time=2
        )
        self.play(FadeOut(ddx_lt_rect))
        self.wait()

        # Bring along constants
        eq_index = ode.submobjects.index(ode["="][0][0])
        ode_lhs_rect = SurroundingRectangle(ode[:eq_index])
        ode_lhs_rect.set_stroke(BLUE, 2)
        ode_lt_lhs_rect = SurroundingRectangle(ode_lt_lhs)
        ode_lt_lhs_rect.set_stroke(YELLOW, 2)
        lhs_arrow = self.get_lt_arrow(ode_lhs_rect, ode_lt_lhs_rect.copy().shift(0.5 * RIGHT))

        self.play(
            LaggedStart(
                *(
                    TransformFromCopy(ode[tex][0], ode_lt_lhs[tex][0])
                    for tex in ["m", R"\mu", "k"]
                ),
                lag_ratio=0.5
            ),
            AnimationGroup(*(
                FadeIn(ode_lt_lhs[tex])
                for tex in [R"\Big(", R"\Big)", "+"]
            ))
        )
        self.wait()
        self.play(LaggedStart(
            FadeOut(xt_arrow),
            FadeOut(ddx_arrow),
            FadeOut(xt_rect),
            FadeOut(ddxt_rect),
            ReplacementTransform(dxt_rect, ode_lhs_rect),
            ReplacementTransform(dxt_arrow, lhs_arrow),
            FadeIn(ode_lt_lhs_rect),
        ))
        self.wait()

        # Factor out X(s)
        Xs_parts = VGroup(
            ode_lt_lhs[R"m\Big({s}^2 X({s})"][0],
            ode_lt_lhs[R"\mu \Big( {s} X({s})"][0],
            ode_lt_lhs[R"k X({s})"][0],
        )
        Xs_part_rects = VGroup(
            SurroundingRectangle(part, buff=0.1)
            for part in Xs_parts
        )
        Xs_part_rects[2].match_height(Xs_part_rects, stretch=True)
        Xs_part_rects.set_stroke(YELLOW, 2)

        ode_lt_lhs.set_fill(opacity=0.35)
        Xs_parts.set_fill(opacity=1)
        Xs_parts[0][1].set_fill(opacity=0.25)
        Xs_parts[1][1].set_fill(opacity=0.25)
        ode_lt_lhs.save_state()
        ode_lt_lhs.set_fill(opacity=1)

        self.remove(ode_lt_lhs_rect)
        self.play(
            FadeOut(ode_lt_lhs_rect),
            FadeIn(Xs_part_rects),
            Restore(ode_lt_lhs),
            lhs_arrow.animate.scale(0.75, about_edge=UL)
        )
        self.wait()
        self.play(LaggedStart(
            *[
                TransformFromCopy(ode_lt_lhs[tex][index0], factored_ode_lt_lhs[tex][index1])
                for tex in [R"X({s})", "m", R"{s}^2", R"\mu", R"{s}", "k"]
                for index0 in [3 if tex == R"{s}" else 0]
                for index1 in [2 if tex == R"{s}" else 0]
            ] + [
                Write(factored_ode_lt_lhs["+"][:2]),
                Write(factored_ode_lt_lhs[R"\big("]),
                Write(factored_ode_lt_lhs[R"\big)"]),
            ],
            lag_ratio=0.1,
            run_time=3
        ))
        self.wait()

        # Show initial conditions
        ic_parts = VGroup(
            ode_lt_lhs[R"- {s} x_0 - v_0"][0],
            ode_lt_lhs[R"- x_0"][0],
        )
        ic_part_consts = VGroup(
            ode_lt_lhs[R"m\Big("][0],
            ode_lt_lhs[R"\Big)"][0],
            ode_lt_lhs[R"\mu \Big("][0],
            ode_lt_lhs[R"\Big)"][1],
        ).copy().set_fill(opacity=1)
        ic_part_rects = VGroup(SurroundingRectangle(part, buff=SMALL_BUFF) for part in ic_parts)
        ic_part_rects.set_stroke(TEAL, 2)
        factored_ic_part = factored_ode_lt_lhs[R"- m v_0 - (m{s} + \mu)x_0"]

        self.play(
            FadeOut(Xs_part_rects),
            FadeIn(ic_part_rects),
            Xs_parts.animate.set_fill(opacity=0.25),
            ic_parts.animate.set_fill(opacity=1),
            FadeIn(ic_part_consts),
        )
        self.play(
            Write(factored_ic_part),
        )
        self.wait()

        # Comment on initial conditions
        ic_rect = SurroundingRectangle(factored_ic_part, buff=0.15)
        ic_rect.set_stroke(TEAL, 3)

        ic_words = Text("Initial conditions")
        ic_words.next_to(ic_rect, DOWN)
        zero_ic = Tex(R"\text{Let’s assume } x_0 = v_0 = 0", t2c=t2c)
        zero_ic.next_to(ic_words, DOWN, aligned_edge=LEFT)

        poly_part = Tex(R"X({s})\big(m{s}^2 + \mu{s} + k\big)", t2c=t2c)
        poly_part.move_to(factored_ode_lt_lhs[poly_part.get_tex()][0])

        lt_equals = Tex(R"=")
        lt_equals.match_y(ode_lt_lhs)
        lt_equals.match_x(ode["="])

        self.play(
            ShowCreation(ic_rect),
            Write(ic_words),
            run_time=1
        )
        self.wait()
        self.play(FadeIn(zero_ic, 0.5 * DOWN))
        self.wait()

        self.play(
            FadeOut(ode_lt_lhs, UP),
            FadeOut(ic_part_rects, UP),
            FadeOut(ic_part_consts, UP),
            poly_part.animate.next_to(lt_equals, LEFT),
            lhs_arrow.animate.scale(1 / 0.75, about_edge=UL),
        )
        self.play(
            LaggedStartMap(FadeOut, VGroup(factored_ode_lt_lhs, ic_rect, ic_words, zero_ic), lag_ratio=0.25),
            run_time=2,
        )
        self.wait()

        # Mirror image
        ode_rhs = ode[R"F_0 \cos(\omega t)"][0]

        part_pairs = [
            # [ode[tex1][index].copy(), poly_part[tex2][index].copy()]
            [ode[tex1][index].copy(), poly_part_copy[tex2][index].copy()]
            for tex1, tex2, index in [
                ("m", "m", 0),
                ("x''(t)", R"{s}^2", 0),
                ("+", "+", 0),
                (R"\mu", R"\mu", 0),
                (R"x'(t)", R"{s}", -1),
                ("+", "+", 1),
                ("k", "k", 0),
            ]
        ]

        self.play(LaggedStart(
            ode_rhs.animate.set_fill(opacity=0.25),
            FadeOut(ode_lhs_rect.copy()),
            ShowCreation(ode_lhs_rect),
            lag_ratio=0.5
        ))
        self.wait()
        self.play(LaggedStart(
            *(TransformFromCopy(*pair) for pair in part_pairs),
            lag_ratio=0.025
        ))
        self.wait()
        self.play(LaggedStart(
            *(TransformFromCopy(*reversed(pair)) for pair in part_pairs),
            lag_ratio=0.025
        ))
        self.wait()

        for pair in part_pairs:
            self.remove(*pair)
        self.add(poly_part)

        # Transform cosine
        ode_lt_rhs = Tex(R"{F_0 {s} \over {s}^2 + \omega^2}", t2c=t2c)
        ode_lt_rhs.next_to(lt_equals, RIGHT)

        rhs_arrow = self.get_lt_arrow(ode_rhs, ode_lt_rhs)

        self.play(
            Write(lt_equals),
            TransformFromCopy(lhs_arrow, rhs_arrow),
            ode_rhs.animate.set_fill(opacity=1),
            ode_lhs_rect.animate.surround(ode_rhs),
        )
        self.wait()
        self.play(LaggedStart(
            *(
                TransformFromCopy(ode[tex][0], ode_lt_rhs[tex][0])
                for tex in ["F_0", R"\omega"]
            ),
            FadeIn(ode_lt_rhs[R"{s} \over {s}^2 +"][0]),
            FadeIn(ode_lt_rhs[R"^2"][1]),
        ))
        self.add(ode_lt_rhs)
        self.wait()

        # Walk through cosine transform
        cos_transform_parts = VGroup(
            Tex(R"\mathcal{L}\big\{\cos(\omega t)\big\}", t2c=t2c),
            Tex(R"= \mathcal{L}\left\{\frac{1}{2}e^{i \omega t} + \frac{1}{2} e^{\minus i \omega t} \right\}", t2c=t2c),
            Tex(R"= \frac{1}{2} \mathcal{L}\big\{e^{i \omega t}\big\} + \frac{1}{2} \mathcal{L}\big\{e^{\minus i \omega t}\big\}", t2c=t2c),
            Tex(R"= \frac{1}{2} {1 \over {s} - \omega i} + \frac{1}{2} {1 \over {s} + \omega i}", t2c=t2c),
            Tex(R"= {{s} \over {s}^2 + \omega^2}", t2c=t2c),
        )
        cos_transform_parts.arrange(RIGHT)
        cos_transform_parts.to_edge(DOWN, buff=1.5)
        cos_transform_parts.to_edge(LEFT, buff=0.5)
        for part in cos_transform_parts[1:]:
            part.next_to(cos_transform_parts[0], RIGHT)
        cos_transform_parts[-1].next_to(cos_transform_parts[-2], RIGHT, aligned_edge=DOWN)

        self.play(LaggedStart(
            TransformFromCopy(ode[R"\cos(\omega t)"], cos_transform_parts[0][R"\cos(\omega t)"]),
            FadeTransform(rhs_arrow[1].copy(), cos_transform_parts[0][R"\mathcal{L}"]),
            Write(cos_transform_parts[0][R"\big\{"]),
            Write(cos_transform_parts[0][R"\big\}"]),
        ))
        self.wait()
        self.play(
            TransformMatchingTex(
                cos_transform_parts[0].copy(),
                cos_transform_parts[1],
                matched_keys=[R"\omega"],
                run_time=1
            )
        )
        self.wait()
        self.play(
            TransformMatchingTex(
                cos_transform_parts[1],
                cos_transform_parts[2],
                matched_keys=[R"\mathcal{L}", R"e^{i \omega t}", R"e^{\minus i \omega t}"],
                key_map={R"\left\{": R"\big\{", R"\right\}": R"\big\}", },
                run_time=1
            )
        )
        self.wait()
        self.play(
            TransformMatchingTex(
                cos_transform_parts[2],
                cos_transform_parts[3],
                matched_keys=[R"\frac{1}{2}", "+"],
                key_map={
                    R"\mathcal{L}\big\{e^{i \omega t}\big\}": R"{1 \over {s} - \omega i}",
                    R"\mathcal{L}\big\{e^{\minus i \omega t}\big\}": R"{1 \over {s} + \omega i}",
                },
                run_time=1,
                path_arc=30 * DEG,
            )
        )
        self.wait()
        self.play(Write(cos_transform_parts[4], run_time=1))
        self.wait()
        self.play(
            *(FadeOut(cos_transform_parts[i]) for i in [0, 3, 4]),
            FadeOut(ode_lhs_rect)
        )
        self.wait()

        # Divide out
        poly_tex = R"\big(m{s}^2 + \mu{s} + k\big)"
        cos_lt_denom_tex = R"\left({s}^2 + \omega^2 \right)"
        true_poly_part = poly_part[poly_tex][0]
        final_answer = Tex(R"X({s}) = {F_0 {s} \over " + cos_lt_denom_tex + poly_tex + "}", t2c=t2c)
        final_answer.next_to(lt_equals, DOWN, buff=1.5)

        poly_rect = SurroundingRectangle(true_poly_part, buff=SMALL_BUFF)
        poly_rect.set_stroke(TEAL, 2)

        self.play(ShowCreation(poly_rect))
        self.play(LaggedStart(
            AnimationGroup(
                TransformFromCopy(true_poly_part, final_answer[poly_tex][0]),
                poly_rect.animate.surround(final_answer[poly_tex][0])
            ),
            TransformFromCopy(poly_part["X({s})"], final_answer["X({s})"]),
            AnimationGroup(
                TransformFromCopy(ode_lt_rhs[R"F_0 {s} \over"], final_answer[R"F_0 {s} \over"]),
                TransformFromCopy(ode_lt_rhs[R"{s}^2 + \omega^2"], final_answer[R"{s}^2 + \omega^2"]),
            ),
            Write(final_answer["="]),
            Write(final_answer[R"\left("]),
            Write(final_answer[R"\right)"]),
            lag_ratio=0.1,
            run_time=3
        ))
        self.play(FadeOut(poly_rect))
        self.wait()

        # Pull up final answer
        self.play(
            LaggedStartMap(FadeOut, VGroup(lhs_arrow, rhs_arrow, poly_part, lt_equals, ode_lt_rhs), shift=0.2 * UP, lag_ratio=0.2, run_time=1),
            LaggedStart(
                final_answer.animate.next_to(ode, DOWN, MED_LARGE_BUFF).shift(RIGHT),
                ode.animate.scale(0.75, about_edge=UP),
                run_time=2,
                lag_ratio=0.25,
            )
        )
        self.wait()

        # Write L{x(t)}
        final_lhs = Tex(R"\mathcal{L}\left\{x(t)\right\} = ", t2c=t2c)
        final_lhs.next_to(final_answer, LEFT)
        final_lhs.shift(SMALL_BUFF * UP)
        xt_copy = ode["x(t)"][0].copy()

        self.play(xt_copy.animate.replace(final_lhs["x(t)"]))
        self.play(Write(final_lhs))
        self.remove(xt_copy)
        self.wait()

        # Reference inversion
        rhs_tex = final_answer.get_tex().split( "= ")[1]
        inverse_equation = Tex(
            R"x(t) = \mathcal{L}^{-1}\left\{" + rhs_tex + R"\right\}",
            t2c=t2c
        )
        inverse_equation.next_to(final_answer, DOWN, LARGE_BUFF)
        inverse_equation.set_x(0)

        self.play(LaggedStart(
            *(
                TransformFromCopy(final_lhs[tex], inverse_equation[tex])
                for tex in ["x(t)", R"\mathcal{L}", R"\left\{", R"\right\}", "="]
            ),
            FadeInFromPoint(inverse_equation["-1"], final_lhs.get_center()),
            TransformFromCopy(final_answer[rhs_tex], inverse_equation[rhs_tex]),
            lag_ratio=0.025,
            run_time=1.5
        ))
        self.wait()
        self.play(FadeOut(inverse_equation, DOWN))

        # Ask about denominator
        denom_tex = cos_lt_denom_tex + poly_tex
        sub_texs = [rhs_tex, denom_tex, poly_tex, cos_lt_denom_tex]
        rhs, denom, poly_part, cos_lt_denom = answer_parts = VGroup(
            final_answer[tex][0]
            for tex in sub_texs
        )
        rect = SurroundingRectangle(rhs, buff=0.05)
        rect.set_stroke(YELLOW, 2)

        zero_question = Text("When is this 0?")
        zero_question.next_to(rect, DOWN)

        self.play(ShowCreation(rect))
        self.wait()
        self.play(
            final_lhs.animate.set_fill(opacity=0.5),
            final_answer[:final_answer.submobjects.index(cos_lt_denom[0])].animate.set_fill(opacity=0.5),
            rect.animate.surround(denom),
            FadeIn(zero_question, lag_ratio=0.1)
        )
        self.wait()

        # Show quadratic formula
        implies = Tex(R"\Longrightarrow", font_size=72)
        implies.rotate(-90 * DEG)
        implies.next_to(poly_part, DOWN)
        eq_0 = Tex(R"=0", font_size=36)
        eq_0.next_to(implies, RIGHT, buff=0)
        implies.add(eq_0)
        implies.add(eq_0.copy().fade(1).next_to(implies, LEFT, buff=0))

        quadratic_form = Tex(R"{s} = {-\mu \pm \sqrt{\mu^2 - 4mk} \over 2m}", t2c=t2c, font_size=36)
        quadratic_form.next_to(implies, DOWN)

        poly_part_copy = Tex(poly_tex, t2c=t2c)
        poly_part_copy.replace(poly_part)

        self.play(
            FadeTransformPieces(zero_question, eq_0),
            Write(implies),
            rect.animate.surround(poly_part),
            cos_lt_denom.animate.set_fill(opacity=0.5)
        )
        self.wait()
        self.play(
            TransformMatchingTex(poly_part_copy, quadratic_form, lag_ratio=0.01)
        )
        self.wait()

        # Show omega i and -omega i roots
        cos_poles = Tex(R"{s} = \pm \omega i", t2c=t2c)
        cos_poles.next_to(implies, DOWN)
        cos_poles.match_x(cos_lt_denom)

        self.play(
            cos_lt_denom.animate.set_fill(opacity=1),
            poly_part.animate.set_fill(opacity=0.5),
            rect.animate.surround(cos_lt_denom),
            implies.animate.match_x(cos_lt_denom),
            FadeTransformPieces(quadratic_form, cos_poles)
        )
        self.wait()

    def get_x_colors_and_t2c(self):
        x_colors = color_gradient([TEAL, RED], 3, interp_by_hsl=True)
        t2c = {
            "x(t)": x_colors[0],
            "x'(t)": x_colors[1],
            "x''(t)": x_colors[2],
            "x_0": x_colors[0],
            "v_0": x_colors[1],
            R"\omega": PINK,
            "{s}": YELLOW,
        }
        return x_colors, t2c


class ShowSolutionPoles(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        F_0 = 1
        m = 1
        k = 2
        mu = 2
        omega = 2

        def func(s):
            denom = (s**2 + omega**2) * (m * s**2 + mu * s + k)
            return np.divide(F_0 * s, denom)

        plane = ComplexPlane((-3, 3), (-3, 3), faded_line_ratio=0)
        plane.add_coordinate_labels(font_size=16)
        plot = get_complex_graph(plane, func)

        self.add(plane, plot)

        # Pan around
        self.play(
            frame.animate.reorient(45, 82, 0, (-0.54, 0.3, 1.06), 5.71),
            run_time=3
        )
        self.play(
            frame.animate.reorient(-31, 83, 0, (-0.13, 0.5, 1.02), 5.71),
            run_time=13,
        )
        self.wait()

        # Emphasize the four roots
        roots = [-1 + 1j, -1 - 1j, 2j, -2j]
        s_points = [plane.n2p(s) for s in roots]
        s_dots = Group(Group(TrueDot(p), GlowDot(p)) for p in s_points)
        s_dots.set_color(YELLOW)
        s_labels = VGroup(
            Tex(Rf"s_{n}").next_to(point, RIGHT)
            for n, point in enumerate(s_points, 1)
        )
        s_labels.set_color(YELLOW)
        s_labels.set_backstroke(BLACK, 5)

        self.play(
            frame.animate.to_default_state().set_field_of_view(1 * DEG),
            plot[0].animate.set_opacity(0.25),
            plot[1].animate.set_stroke(opacity=0.01),
            run_time=2
        )
        self.play(
            FadeIn(s_labels, lag_ratio=0.25),
            FadeIn(s_dots, lag_ratio=0.25),
        )
        self.wait()

        # Highlight certain roots
        rect = SurroundingRectangle(Group(s_labels[:2], s_dots[:2]))
        self.play(ShowCreation(rect))
        self.wait()
        self.play(rect.animate.surround(Group(s_labels[2:], s_dots[2:])))
        self.wait()
        self.play(FadeOut(rect))


class PartialFractionDecomposition(WalkThroughEquationSolution):
    def construct(self):
        # Set up equation
        x_colors, t2c = self.get_x_colors_and_t2c()
        poly_tex = R"\big(m{s}^2 + \mu{s} + k\big)"
        cos_lt_denom_tex = R"\left({s}^2 + \omega^2 \right)"
        denom_tex = cos_lt_denom_tex + poly_tex
        tex_kw = dict(t2c=t2c, font_size=42)

        final_answer = Tex(R"X({s}) = {F_0 {s} \over " + denom_tex + "}", **tex_kw)
        final_answer.to_corner(UL)
        final_answer.save_state()
        final_answer.center().scale(1.5)
        self.add(final_answer)

        # Show the roots
        t2c.update({f"r_{n}": TEAL for n in range(1, 5)})
        denom_rect = SurroundingRectangle(final_answer[denom_tex], buff=0.1)
        denom_rect.target = SurroundingRectangle(final_answer.saved_state[denom_tex], buff=0.1)
        VGroup(denom_rect, denom_rect.target).set_stroke(TEAL, 3)

        denom_roots_title = Text("Roots of denominator", font_size=36)
        denom_roots_title.next_to(final_answer.saved_state, DOWN, buff=1.25, aligned_edge=LEFT)
        denom_roots_title.add(Underline(denom_roots_title))

        roots = VGroup(
            Tex(R"r_1 = +\omega i", **tex_kw),
            Tex(R"r_2 = -\omega i", **tex_kw),
            Tex(R"r_3 = \left(-\mu + \sqrt{\mu^2 - 4mk} \right) / 2m", **tex_kw),
            Tex(R"r_4 = \left(-\mu - \sqrt{\mu^2 - 4mk} \right) / 2m", **tex_kw),
        )
        for root in roots[2:]:
            root[3:].scale(0.8, about_edge=LEFT)
        roots.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        roots.next_to(denom_roots_title[0], DOWN, aligned_edge=LEFT, buff=0.75)

        self.play(ShowCreation(denom_rect))
        self.play(
            Restore(final_answer),
            MoveToTarget(denom_rect),
            FadeIn(denom_roots_title),
            LaggedStart(*(
                FadeIn(root[:3])
                for root in roots
            ), lag_ratio=0.15),
            LaggedStart(
                FadeTransform(final_answer[cos_lt_denom_tex].copy(), roots[0][3:]),
                FadeTransform(final_answer[cos_lt_denom_tex].copy(), roots[1][3:]),
                FadeTransform(final_answer[poly_tex].copy(), roots[2][3:]),
                FadeTransform(final_answer[poly_tex].copy(), roots[3][3:]),
                lag_ratio=0.15,
            ),
        )
        self.play(FadeOut(denom_rect))
        self.wait()

        # Break apart fraction
        rhs_str = " = " + " + ".join([
            fR"{{c_{n} \over {{s}} - r_{n}}}"
            for n in range(1, 5)
        ])
        rhs = Tex(rhs_str, **tex_kw)
        rhs.next_to(final_answer, RIGHT)

        self.play(
            LaggedStart(
                *(
                    Transform(root[:2].copy(), rn, remover=True)
                    for root, rn in zip(roots, rhs[re.compile(r"r_.")])
                ),
                lag_ratio=0.2
            ),
            FadeIn(rhs, lag_ratio=0.1, time_span=(0.5, 1.5))
        )
        self.add(rhs)
        self.wait()

        # Highlight c terms
        c_terms = rhs[re.compile(r"c_.")]
        c_rects = VGroup(SurroundingRectangle(c_term, buff=0.1) for c_term in c_terms)
        c_rects.set_stroke(BLUE, 2)
        q_marks = VGroup(Tex(R"???", font_size=36).next_to(rect, UP, SMALL_BUFF) for rect in c_rects)

        self.play(
            LaggedStartMap(ShowCreation, c_rects, lag_ratio=0.2),
            LaggedStartMap(FadeIn, q_marks, lag_ratio=0.2),
        )
        self.wait()

        # Write up the partial fraction decomposition idea
        clean_fraction = Tex(
            R"F_0 {s} / m \over ({s} - r_1)({s} - r_2)({s} - r_3)({s} - r_4)",
            **tex_kw
        )
        example = Tex(R"c_1 = {F_0 r_1 / m \over (r_1 - r_2)(r_1 - r_3)(r_1 - r_4)}", **tex_kw)
        pfd_group = VGroup(
            Text("It’s easiest to rewrite our fraction as", **tex_kw),
            clean_fraction,
            TexText(R"""
                To calculate each term $c_i$, remove the $({s} - r_i)$ term \\
                from our fraction and plug in ${s} = r_i$. For example,
            """, alignment="", **tex_kw),
            example,
            TexText(R"""
                Why? This essentially amounts to multiplying both sides\\
                of our equation by $({s} - r_1)$ and taking the limit as ${s} \to r_1$
            """, alignment="", **tex_kw),
        )
        pfd_group.arrange(DOWN, buff=0.5)
        for part in pfd_group[0::2]:
            part.align_to(pfd_group, LEFT)

        pfd_box = SurroundingRectangle(pfd_group, buff=0.5)
        pfd_box.set_fill(GREY_E, 1)
        pfd_box.set_stroke(WHITE, 2)
        pfd_group.add_to_back(pfd_box)
        pfd_group.set_width(7.0)
        pfd_group.to_corner(DR)

        pfd_title = Text("Partial Fraction Decomposition")
        pfd_title.next_to(pfd_box, UP)
        pfd_group.add_to_back(pfd_title)

        self.play(
            FadeIn(pfd_group, lag_ratio=5e-3, run_time=2),
        )
        self.wait()
        self.play(
            FadeOut(pfd_group),
            FadeOut(c_rects),
            FadeOut(q_marks),
        )

        # Show inverse laplace process
        t2c["{t}"] = BLUE
        exp_sum = Tex("+".join([
            Rf"c_{i} e^{{r_{i} {{t}} }}"
            for i in range(1, 5)
        ]), t2c=t2c, font_size=46)

        exp_sum.next_to(rhs[1:], DOWN, buff=1.5)
        frac_parts = rhs[re.compile(r"{c_. \\over {s} - r_.}")]
        exp_parts = exp_sum[re.compile(r"c_. e\^{r_. {t} }")]

        inv_lt_arrows = VGroup(
            self.get_lt_arrow(frac_part, exp_part, label_tex=R"\mathcal{L}^{-1}")
            for frac_part, exp_part in zip(frac_parts, exp_parts)
        )
        frac_rects = VGroup(SurroundingRectangle(part) for part in frac_parts)
        frac_rects.set_stroke(YELLOW, 2)

        self.play(ShowCreation(frac_rects, lag_ratio=0.2))
        self.play(
            LaggedStartMap(self.grow_lt_arrow, inv_lt_arrows),
            LaggedStart(*(
                TransformFromCopy(rhs[re.compile(p1)], exp_sum[re.compile(p2)], lag_ratio=0.05)
                for p1, p2 in [
                    (r"c_.", r"c_."),
                    (r"\\over", r"e"),
                    (r"{s}", r"{t}"),
                    (r"r_.", r"r_."),
                    (r"\+", r"\+"),
                ]
            ), lag_ratio=0.1, run_time=2),
            # TransformMatchingTex(rhs.copy(), exp_sum),
        )
        self.add(exp_sum)
        self.wait()

        # Highlight rs then cs
        upper_r_rects, lower_r_rects, upper_c_rects, lower_c_rects = [
            VGroup(SurroundingRectangle(term, buff=0.025) for term in group[pattern]).set_stroke(YELLOW, 2)
            for pattern in [re.compile(R"r_."), re.compile(R"c_.")]
            for group in [rhs, exp_sum]
        ]

        self.play(ReplacementTransform(frac_rects, upper_r_rects))
        self.wait()
        self.play(TransformFromCopy(upper_r_rects, lower_r_rects))
        self.wait()
        self.play(
            ReplacementTransform(upper_r_rects, upper_c_rects),
            ReplacementTransform(lower_r_rects, lower_c_rects),
        )
        self.wait()
        self.play(FadeOut(upper_c_rects), FadeOut(lower_c_rects))

        # Highlight cosine part
        index = exp_sum.submobjects.index(exp_sum["+"][1][0])
        cos_part_rect = SurroundingRectangle(exp_sum[:index], buff=0.1)
        cos_part_rect.set_stroke(BLUE, 2)
        c1_tex = R"{F_0 \over 2m(k/m - \omega^2) + 2\mu \omega i}"
        c2_tex = R"{F_0 \over 2m(k/m - \omega^2) - 2\mu \omega i}"
        const_tex = R"{F_0 \over 2m(k/m - \omega^2)}"
        const_tex2 = R"{F_0 \over m(k/m - \omega^2)}"
        two_exp_tex = R" \left(e^{+\omega i t} + e^{-\omega i t} \right)"
        cos_exprs = VGroup(
            Tex(R"c_1 e^{+\omega i t} + c_2 e^{-\omega i t}", t2c=t2c),
            Tex(const_tex + two_exp_tex, t2c=t2c),
            Tex(const_tex + R"2\cos(\omega t)", t2c=t2c),
            Tex(const_tex2 + R"\cos(\omega t)", t2c=t2c),
        )
        cos_exprs.next_to(cos_part_rect, DOWN, LARGE_BUFF)
        const_part = cos_exprs[1][const_tex][0]
        const_rect = SurroundingRectangle(const_part)
        two_exp_rect = SurroundingRectangle(cos_exprs[1][two_exp_tex])
        cos_rect = SurroundingRectangle(cos_exprs[2][R"2\cos(\omega t)"])
        amp_rect = SurroundingRectangle(cos_exprs[3][const_tex2])
        diff_rect = SurroundingRectangle(cos_exprs[3][R"k/m - \omega^2"])
        VGroup(c_rects, const_rect, two_exp_rect, cos_rect, amp_rect, diff_rect).set_stroke(YELLOW, 2)

        const_values = VGroup(
            Tex(c1_tex, t2c=t2c, font_size=24),
            Tex(c2_tex, t2c=t2c, font_size=24),
        )
        const_values.arrange(RIGHT, buff=LARGE_BUFF)
        const_values.next_to(cos_exprs[0], DOWN, LARGE_BUFF, aligned_edge=LEFT)
        const_value_rects = VGroup(SurroundingRectangle(value, buff=0.05) for value in const_values)
        c_terms = cos_exprs[0][re.compile("c_.")]
        c_rects = VGroup(SurroundingRectangle(c_term, buff=0.05) for c_term in c_terms)
        rect_lines = VGroup(
            Line(r2.get_bottom(), r1.get_top())
            for r1, r2 in zip(const_value_rects, c_rects)
        )
        VGroup(const_value_rects, c_rects, rect_lines).set_stroke(RED, 1)

        mu_assumption = Tex(R"\text{Assume } \mu \approx 0", font_size=36)
        mu_assumption.next_to(const_value_rects, DOWN)

        self.play(
            ShowCreation(cos_part_rect),
            exp_sum[index:].animate.set_opacity(0.25),
            inv_lt_arrows.animate.set_fill(opacity=0.25),
            roots[2:].animate.set_opacity(0.25),
            denom_roots_title.animate.set_opacity(0.25),
        )
        self.wait()
        self.play(FadeIn(cos_exprs[0], DOWN))
        self.wait()
        self.play(
            LaggedStartMap(ShowCreation, c_rects, lag_ratio=0.35),
            LaggedStartMap(ShowCreation, const_value_rects, lag_ratio=0.35),
            LaggedStartMap(ShowCreation, rect_lines, lag_ratio=0.35),
            LaggedStart(
                *(FadeTransform(c_term.copy(), const_value)
                for c_term, const_value in zip(c_terms, const_values)),
                lag_ratio=0.35,
            )
        )
        self.wait()
        self.play(Write(mu_assumption))
        self.play(
            const_values[0][R"+ 2\mu \omega i}"].animate.set_fill(opacity=0.25),
            const_values[1][R"- 2\mu \omega i}"].animate.set_fill(opacity=0.25),
        )
        self.wait()
        self.play(LaggedStart(
            TransformFromCopy(const_values[0][:len(const_part)], const_part),
            TransformFromCopy(const_values[1][:len(const_part)], const_part),
            FadeOut(VGroup(const_values, rect_lines, c_rects)),
            FadeOut(c_terms),
            ReplacementTransform(cos_exprs[0][R"e^{+\omega i t}"], cos_exprs[1][R"e^{+\omega i t}"]),
            ReplacementTransform(const_value_rects[0], const_rect),
            ReplacementTransform(cos_exprs[0][R"e^{-\omega i t}"], cos_exprs[1][R"e^{-\omega i t}"]),
            ReplacementTransform(const_value_rects[1], const_rect),
            ReplacementTransform(cos_exprs[0][R"+"][1], cos_exprs[1][R"+"][1]),
            FadeIn(cos_exprs[1][R"\left("]),
            FadeIn(cos_exprs[1][R"\right)"]),
            mu_assumption.animate.next_to(cos_exprs[1], DOWN, MED_LARGE_BUFF),
            lag_ratio=0.01
        ))
        self.add(cos_exprs[1])
        self.wait()
        self.play(ReplacementTransform(const_rect, two_exp_rect))
        self.wait()
        self.play(
            ReplacementTransform(two_exp_rect, cos_rect),
            TransformMatchingTex(*cos_exprs[1:3], key_map={two_exp_tex: R"2\cos(\omega t)"}),
            run_time=1
        )
        self.wait()
        self.play(
            ReplacementTransform(cos_rect, amp_rect),
            TransformMatchingTex(*cos_exprs[2:4]),
            run_time=1
        )
        self.wait()
        self.play(ReplacementTransform(amp_rect, diff_rect))
        self.wait()


class IntegrateByParts(InteractiveScene):
    def construct(self):
        # Show formulas
        t2c = {"{t}": BLUE, "{s}": YELLOW}
        ibp_rule = VGroup(
            Text("Integration by parts: "),
            Tex(R"\int u dv = uv - \int v du", t2c={"u": PINK, "v": RED})
        )
        ibp_rule.arrange(RIGHT, buff=MED_LARGE_BUFF)
        ibp_rule.to_edge(UP, buff=MED_LARGE_BUFF)

        lt_deriv_lines = VGroup(
            Tex(R"\mathcal{L}\big\{f'({t})\big\} =\int_0^\infty f'({t}) e^{\minus {s}{t}} d{t}", t2c=t2c),
            Tex(R"=\Big[f({t}) e^{\minus {s}{t}} \Big]_0^\infty - \int_0^\infty f({t}) (-{s}) e^{\minus {s}{t}} d{t}", t2c=t2c),
            Tex(R"=-f(0) + {s} \int_0^\infty f({t}) e^{\minus {s}{t}} d{t}", t2c=t2c),
            Tex(R"={s} \mathcal{L}\big\{f({t})\big\} -f(0)", t2c=t2c),
        )
        lt_deriv_lines.arrange(DOWN, aligned_edge=LEFT)
        lt_deriv_lines[1:].align_to(lt_deriv_lines[0]["="], LEFT)
        lt_deriv_lines.next_to(ibp_rule, DOWN, LARGE_BUFF, aligned_edge=LEFT)

        # Show Definition
        definition = lt_deriv_lines[0]
        definition.save_state()
        definition.set_height(1.6).center()
        eq_index = definition.submobjects.index(definition["="][0][0])

        def_brace = Brace(definition[eq_index:], UP)
        def_words = Text("Definition", font_size=60)
        def_words.next_to(def_brace, UP)

        self.add(definition[:eq_index])
        self.wait()
        self.play(LaggedStart(
            Transform(*definition["f'({t})"].copy(), path_arc=60 * DEG, remover=True),
            GrowFromCenter(def_brace),
            FadeIn(def_words, 0.5 * UP),
            Write(definition[eq_index:], lag_ratio=0.1),
            lag_ratio=0.2
        ))
        self.wait()
        self.play(LaggedStart(
            Transform(
                definition[R"\int"][0].copy(),
                ibp_rule[1][R"\int"][0].copy(),
                remover=True,
                path_arc=-45 * DEG,
            ),
            FadeOut(def_brace),
            FadeOut(def_words),
            FadeIn(ibp_rule, lag_ratio=0.1),
            lag_ratio=0.2
        ))
        self.wait()

        # Show remaining lines
        self.play(Restore(definition))
        self.play(
            FadeTransform(ibp_rule[1][4:].copy(), lt_deriv_lines[1])
        )
        self.wait()
        for line in lt_deriv_lines[2:]:
            self.play(FadeIn(line, 0.5 * DOWN))
            self.wait()

        # Ask about times s and minus f(0)
        final_line = lt_deriv_lines[-1]
        s_rect = SurroundingRectangle(final_line["{s}"][0], buff=0.05)
        s_rect.set_stroke(YELLOW, 2)
        minus_f0_rect = SurroundingRectangle(final_line["-f(0)"][0], buff=0.05)
        minus_f0_rect.set_stroke(BLUE, 2)

        why = Text("Why?")
        why.next_to(s_rect, LEFT, buff=1.5).shift(0.5 * UP)
        why_arrow = Arrow(why.get_right() + SMALL_BUFF * RIGHT, s_rect.get_top(), path_arc=-90 * DEG, buff=0.05)
        VGroup(why, why_arrow).set_color(YELLOW)

        why2 = Text("Why?")
        why2.next_to(minus_f0_rect, RIGHT)
        why2.set_color(BLUE)

        self.play(
            Write(why),
            Write(why_arrow),
            ShowCreation(s_rect),
            lt_deriv_lines[1:3].animate.set_fill(opacity=0.3),
        )
        self.wait()
        self.play(
            ShowCreation(minus_f0_rect),
            Write(why2),
        )
        self.wait()


class FromPropertyToLaplaceTransform(InteractiveScene):
    def construct(self):
        # Add desired property
        t2c = {"{t}": BLUE, "{s}": YELLOW}
        prop = VGroup(
            Tex(R"d / d{t}", font_size=72, t2c=t2c),
            Vector(DOWN, thickness=5),
            Tex(R"\times {s}", font_size=72, t2c=t2c),
        )
        prop.arrange(DOWN)
        prop.to_edge(LEFT, buff=LARGE_BUFF)
        prop_rect = SurroundingRectangle(prop, buff=MED_LARGE_BUFF)
        prop_rect.set_stroke(GREEN, 2)
        goal_word = Text("Goal", font_size=72)
        goal_word.set_fill(GREEN)
        goal_word.next_to(prop_rect, UP)

        self.add(prop)
        self.play(
            ShowCreation(prop_rect),
            Write(goal_word)
        )

        # Show threads
        n_threads = 10
        threads = VGroup(
            self.get_thread(prop_rect.get_right() + shift)
            for shift in np.linspace(0.5 * UP, 0.5 * DOWN, n_threads)
        )
        kw = dict(
            lag_ratio=1.0 / n_threads,
            run_time=4
        )
        self.play(
            ShowCreation(threads, **kw),
            LaggedStartMap(VShowPassingFlash, threads.copy().set_stroke(WHITE, 2), time_width=2.0, **kw)
        )
        self.wait()

        # Transforms
        lt, inv_lt = transforms = VGroup(
            Tex(R"F({s}) = \int_0^\infty f({t}) e^{\minus {s}{t}} d{t}", t2c=t2c),
            Tex(R"f({t}) = \frac{1}{2\pi i} \int_{a - i\infty}^{a + i\infty} F({s}) e^{{s}{t}} d{s}", t2c=t2c),
        )

        transforms.arrange(DOWN, buff=2.5)
        transforms.to_edge(RIGHT)

        transform_rects = VGroup(SurroundingRectangle(term) for term in transforms)

        v_line = Line(*transform_rects, buff=0)
        h_line = Line(prop_rect.get_right(), v_line.get_center())
        VGroup(v_line, h_line, transform_rects).set_stroke(GREY, 1)

        self.play(
            Transform(threads, h_line.replicate(len(threads)), remover=True, lag_ratio=0.1, run_time=3)
        )
        self.add(h_line)
        self.play(GrowFromCenter(v_line, run_time=0.5))
        self.play(
            FadeIn(transform_rects),
            FadeIn(transforms),
        )
        self.wait()

        # Unify
        transforms.target = transforms.generate_target()
        transforms.target.space_out_submobjects(0.7)
        new_rect = SurroundingRectangle(transforms.target)
        new_rect.set_stroke(GREY, 2)

        self.play(
            MoveToTarget(transforms),
            ReplacementTransform(transform_rects[0], new_rect),
            ReplacementTransform(transform_rects[1], new_rect),
            v_line.animate.scale(0),
            h_line.animate.put_start_and_end_on(prop_rect.get_right(), new_rect.get_left()),
        )
        self.wait()

    def get_thread(self, start_point, step_size=0.15, angle_range=(-45 * DEG, 45 * DEG), n_steps=20):
        points = [start_point]
        for n in range(n_steps):
            step = rotate_vector(RIGHT, random.uniform(*angle_range))
            points.append(points[-1] + step)

        path = VMobject().set_points_smoothly(points, approx=False)
        path.set_stroke(GREY, 1)
        return path


class EndScreen(SideScrollEndScreen):
    pass
