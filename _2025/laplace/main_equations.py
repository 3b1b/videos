from manim_imports_ext import *
from _2025.laplace.integration import get_complex_graph
from _2025.laplace.exponentials import SPlane
from _2025.laplace.exponentials import get_exp_graph_icon


class DrivenHarmonicOscillatorEquation(InteractiveScene):
    def construct(self):
        colors = color_gradient([TEAL, RED], 3, interp_by_hsl=True)
        equation = Tex(
            R"m x''(t) + \mu x'(t) + k x(t) = F_0 \cos(\omega t)",
            t2c={
                "x(t)": colors[0],
                "x'(t)": colors[1],
                "x''(t)": colors[2],
                R"\omega": PINK,
            }
        )
        self.add(equation)


class SimpleCosGraph(InteractiveScene):
    rotation_frequency = TAU / 4

    def construct(self):
        # Show graph
        def get_t():
            return self.time

        t_max = 20
        axes = Axes((0, t_max), (-1, 1), x_axis_config=dict(unit_size=0.6))
        axes.scale(1.25).to_edge(LEFT)

        x_axis = axes.x_axis

        x_axis.add(VGroup(
            Tex(tex, font_size=20).next_to(x_axis.n2p(n), DOWN)
            for n, tex in zip(it.count(1, 2), self.get_pi_frac_texs())
        ))

        def cos_func(t):
            return np.cos(self.rotation_frequency * t)

        graph = axes.get_graph(cos_func)
        graph.set_stroke(TEAL, 3)
        graph_ghost = graph.copy()
        graph_ghost.set_stroke(opacity=0.5)
        shown_graph = graph.copy()
        shown_graph.add_updater(lambda m: m.pointwise_become_partial(graph, 0, get_t() / t_max))

        output_dot = Group(TrueDot(), GlowDot()).set_color(YELLOW)
        output_dot.add_updater(lambda m: m.move_to(axes.y_axis.n2p(cos_func(get_t()))))

        h_line = Line()
        h_line.set_stroke(WHITE, 1)
        h_line.f_always.put_start_and_end_on(output_dot.get_center, shown_graph.get_end)

        cos_label = Tex(R"\cos(t)", font_size=60)
        cos_label.to_edge(UP)

        self.add(axes, graph_ghost, output_dot, shown_graph, h_line)
        self.wait(10)
        self.play(FadeOut(x_axis))
        self.wait(14)

    def get_pi_frac_texs(self):
        return [
            R"\pi / 2", R"\pi", R"3 \pi / 2", R"2\pi",
            R"5\pi / 2", R"3\pi", R"7 \pi / 2", R"4\pi",
            R"9\pi / 2", R"5\pi",
        ]


class BreakUpCosineTex(InteractiveScene):
    def construct(self):
        # Show sum
        pure_cos = Tex(R"\cos(t)", font_size=72)
        pure_cos.to_edge(UP)
        tex_pieces = [R"\cos(t)", "=", R"\frac{1}{2}", "e^{+it}", "{+}", R"{1 \over 2}", "e^{-it}"]
        sum_tex = Tex(" ".join(tex_pieces), t2c={"+i": YELLOW, "-i": YELLOW})
        sum_tex.to_edge(UP, buff=MED_SMALL_BUFF).shift(0.5 * RIGHT)
        cos, equals, half1, eit, plus, half2, enit = pieces = VGroup(
            sum_tex[tex][0] for tex in remove_list_redundancies(tex_pieces)
        )

        self.add(pure_cos)
        self.wait()
        self.play(
            FadeTransform(pure_cos, cos),
            Write(pieces[1:])
        )
        self.wait()

        # Fade parts
        pieces.generate_target()
        pieces.target.set_fill(opacity=0.35)
        pieces.target[3].set_fill(opacity=1)
        pieces.target.space_out_submobjects(1.2)
        self.play(MoveToTarget(pieces))
        self.wait()
        self.play(VGroup(plus, enit).animate.set_fill(opacity=1))
        self.wait()
        self.play(pieces.animate.set_fill(opacity=1))
        self.play(pieces.animate.space_out_submobjects(1 / 1.2))
        self.wait()


class TranslateToNewLanguage(InteractiveScene):
    graph_resolution = (301, 301)
    show_integral = True
    label_config = dict(
        font_size=72,
        t2c={"{t}": BLUE, "{s}": YELLOW}
    )

    def construct(self):
        # Set up a functions
        full_s_samples = self.get_s_samples()
        func_s_samples = [
            complex(-2, 2),
            complex(-2, -2),
            complex(0, 1),  # Changed
            complex(-1, 0),
            complex(0, -1),  # Changed
        ]
        func_weights = [-1, -1, 1j, 2, -1j]

        def func(t):
            return sum([
                (weight * np.exp(complex(0.1 * s.real, s.imag) * t)).real
                for s, weight in zip(func_s_samples, func_weights)
            ])

        # Graph
        axes, graph, graph_label = self.get_graph_group(func)

        # Show the S-plane pieces
        frame = self.frame
        frame.set_y(0.5)
        s_plane, exp_pieces, s_plane_name = self.get_s_plane_and_exp_pieces(full_s_samples)

        self.play(LaggedStart(
            FadeIn(axes),
            ShowCreation(graph),
            FadeIn(graph_label),
            FadeIn(s_plane_name, lag_ratio=0.1),
            LaggedStartMap(FadeIn, exp_pieces, lag_ratio=0.1),
        ))
        self.wait()

        # Narrow down specific pieces
        exp_pieces.save_state()
        exp_pieces.generate_target()
        key_pieces = VGroup()
        for piece, s_sample in zip(exp_pieces.target, full_s_samples):
            if s_sample not in func_s_samples:
                piece.fade(0.7)
            else:
                key_pieces.add(piece)

        weight_labels = VGroup(
            Tex(Rf"\times {w}", font_size=24).next_to(piece.get_top(), DOWN, SMALL_BUFF)
            for w, piece in zip([R"\minus 1", R"\minus 1", R"\minus i", "2", "i"], key_pieces)
        )
        self.play(
            MoveToTarget(exp_pieces),
            LaggedStartMap(FadeIn, weight_labels),
        )
        self.play(LaggedStart(
            (Transform(graph.copy(), piece[-1].copy().insert_n_curves(100), remover=True)
            for piece in key_pieces),
            lag_ratio=0.1,
            group_type=Group,
            run_time=2
        ))
        self.wait()

        # Reveal plane
        frame = self.frame
        arrow, fancy_L, Fs_label = self.get_arrow_to_Fs(graph_label)
        Fs_label.save_state()
        Fs_label.become(graph_label)

        def Func(s):
            result = sum([
                np.divide(w, (s - s0))
                for s0, w in zip(func_s_samples, func_weights)
            ])
            return min(100, result)

        lt_graph = get_complex_graph(s_plane, Func, resolution=self.graph_resolution, face_sort_direction=DOWN)
        lt_graph.stretch(0.25, 2, about_point=s_plane.n2p(0))
        lt_graph.save_state()
        lt_graph.stretch(0, 2, about_point=s_plane.n2p(0))
        lt_graph.set_opacity(0)

        exp_pieces.target = exp_pieces.saved_state.copy()
        for piece in exp_pieces.target:
            piece.scale(0.35)

        self.add(exp_pieces, lt_graph, graph_label, arrow, Fs_label, Point(), weight_labels)
        self.play(
            FadeOut(s_plane_name),
            GrowArrow(arrow),
            Write(fancy_L),
            Restore(Fs_label, time_span=(1, 2), path_arc=-10 * DEG),
            FadeIn(s_plane),
            MoveToTarget(exp_pieces, lag_ratio=1e-3),
            FadeOut(weight_labels),
            Restore(lt_graph, time_span=(1.5, 3)),
            frame.animate.reorient(70, 86, 0, (-4.94, -2.45, 3.51), 19.43),
            run_time=3
        )

        # Show interal and continuation
        if self.show_integral:
            # For an insertion
            integral = Tex(R"= \int^\infty_0 f({t})e^{\minus{s}{t}}d{t}", t2c=self.label_config["t2c"])
            integral.fix_in_frame()
            integral.next_to(Fs_label, RIGHT)
            integral.set_backstroke(BLACK, 5)
            rect = BackgroundRectangle(VGroup(Fs_label, integral))
            rect.set_fill(BLACK, 0.8)
            rect.scale(2, about_edge=DL)
            rect.shift(0.25 * DOWN)

            graph_copy = lt_graph[0].copy()
            graph_copy.set_clip_plane(RIGHT, -s_plane.get_left()[0])
            graph_copy.fade(0.5)

            self.add(rect, Fs_label, integral)
            self.play(
                FadeIn(rect),
                Write(integral, run_time=1)
            )
            self.wait()
            lt_graph.set_clip_plane(RIGHT, s_plane.get_left()[0])
            self.play(
                frame.animate.reorient(-10, 85, 0, (-2.18, 0.45, 3.12), 11.52),
                lt_graph.animate.set_clip_plane(RIGHT, -s_plane.n2p(0)[0]),
                run_time=3
            )
            self.play(
                frame.animate.reorient(33, 85, 0, (-1.81, 0.13, 2.44), 12.25),
                run_time=6
            )
            self.wait()
            self.add(graph_copy, rect, Fs_label, integral)
            self.play(
                FadeOut(rect),
                ShowCreation(graph_copy),
                frame.animate.reorient(-2, 67, 0, (-0.95, -0.06, 1.91), 9.45),
                run_time=8
            )

            # Show key exponentials below poles
            for piece in key_pieces:
                piece.set_height(1)
            self.add(key_pieces, graph_copy, graph)
            self.play(
                frame.animate.reorient(0, 0, 0, (-1.23, 1.49, 1.9), 9.36),
                FadeIn(key_pieces),
                exp_pieces.animate.fade(0.5),
                run_time=3,
            )

        # Reorient
        self.play(frame.animate.reorient(-39, 90, 0, (-1.37, 1.1, 4.12), 10.93), run_time=15)
        self.play(frame.animate.reorient(0, 0, 0, (-2.13, 0.07, 2.2), 9.79), run_time=10)
        self.play(frame.animate.reorient(84, 87, 0, (-4.22, -3.48, 5.26), 22.43), run_time=10)

        # Poles as lines
        pole_lines = VGroup(
            Line(s_plane.n2p(s), s_plane.n2p(s) + 20 * OUT)
            for s in func_s_samples
        )
        pole_lines.set_stroke(WHITE, 3)

        key_pieces.target = key_pieces.generate_target()
        target_rects = VGroup()
        for piece in key_pieces.target:
            piece.set_height(1.2)
            target_rect = piece[0].copy()
            target_rect.set_fill(opacity=0)
            target_rects.add(target_rect)

        self.add(pole_lines, key_pieces, lt_graph)
        self.play(
            ShowCreation(pole_lines, lag_ratio=0.1),
        )
        self.play(
            frame.animate.reorient(0, 0, 0, (-0.98, 0.82, 0.0), 10.00),
            lt_graph[0].animate.set_opacity(0.2),
            lt_graph[1].animate.set_opacity(0.05),
            pole_lines.animate.stretch(0, 2, about_edge=IN),
            MoveToTarget(key_pieces),
            FadeIn(weight_labels),
            run_time=3,
        )
        self.wait()

        # Shift things down
        top_rect = FullScreenRectangle()
        top_rect.set_fill(BLACK, 1).set_stroke(width=0)
        top_rect.set_height(2.5, about_edge=UP, stretch=True)
        top_rect.fix_in_frame()

        h_line = DashedLine(top_rect.get_corner(DL), top_rect.get_corner(DR))
        h_line.set_stroke(WHITE, 1)
        h_line.fix_in_frame()

        top_rect.save_state()
        top_rect.stretch(0, 1, about_edge=UP)

        self.play(
            Restore(top_rect),
            ShowCreation(h_line, time_span=(1, 2)),
            VGroup(axes, graph).animate.shift(2 * DOWN),
            VGroup(graph_label, arrow, fancy_L, Fs_label).animate.shift(3 * DOWN),
            frame.animate.reorient(-17, 90, 0, (-2.86, 1.57, 3.14), 10.95),
            run_time=2
        )
        self.play(frame.animate.reorient(39, 92, 0, (-4.35, 0.64, 3.03), 14.99), run_time=20)

    def get_graph_group(self, func, func_tex=R"f({t})"):
        # axes = Axes((0, 7), (-4, 4), width=0.5 * FRAME_WIDTH - 1, height=5)
        axes = Axes((0, 8), (-1, 6, 0.5), width=0.3 * FRAME_WIDTH - 1, height=7)
        axes.to_edge(LEFT).shift(0.5 * DOWN)
        graph = axes.get_graph(func)
        graph.set_stroke(BLUE, 5)
        graph.set_scale_stroke_with_zoom(True)
        axes.set_scale_stroke_with_zoom(True)
        graph_label = Tex(func_tex, **self.label_config)
        graph_label.move_to(axes).to_edge(UP, buff=LARGE_BUFF)

        graph_group = VGroup(axes, graph, graph_label)
        graph_group.fix_in_frame()
        return graph_group

    def get_s_samples(self):
        return [complex(a, b) for a in range(-2, 3) for b in range(-2, 3)]

    def get_s_plane_and_exp_pieces(self, s_samples):
        s_plane = ComplexPlane((-3, 3), (-3, 3))
        s_plane.set_height(7.5)
        s_plane.move_to(3.75 * RIGHT)
        s_plane.set_z_index(-1)

        exp_pieces = VGroup(
            self.get_exp_graph(s).move_to(s_plane.n2p(s))
            for s in s_samples
        )
        s_plane_name = Text("S-plane", font_size=72)
        s_plane_name.next_to(exp_pieces, UP, MED_SMALL_BUFF)
        return s_plane, exp_pieces, s_plane_name

    def get_arrow_to_Fs(self, graph_label):
        arrow = Vector(2 * RIGHT, thickness=5, fill_color=WHITE)
        arrow.fix_in_frame()
        arrow.next_to(graph_label, RIGHT, buff=MED_LARGE_BUFF)

        fancy_L = Tex(R"\mathcal{L}", font_size=60)
        fancy_L.next_to(arrow, UP, buff=0)
        fancy_L.fix_in_frame()

        Fs_label = Tex(R"F({s})", **self.label_config)
        Fs_label.next_to(arrow, RIGHT, MED_LARGE_BUFF)
        Fs_label.fix_in_frame()
        Fs_label.set_z_index(1)
        Fs_label.set_backstroke(BLACK, 5)

        return VGroup(arrow, fancy_L, Fs_label)

    def get_exp_graph(self, s, **kwargs):
        return get_exp_graph_icon(s, **kwargs)


class TranslateDifferentialEquationAndInvert(InteractiveScene):
    def construct(self):
        # Translate the equations
        colors = color_gradient([TEAL, RED], 3, interp_by_hsl=True)
        t2c = {
            R"{s}": YELLOW,
            R"x(t)": colors[0],
            R"x'(t)": colors[1],
            R"x''(t)": colors[2],
            R"\omega": PINK,
        }
        kw = dict(t2c=t2c, font_size=30)
        lhs = Tex(R"m x''(t) + \mu x'(t) + k x(t) = F_0 \cos(\omega t)", **kw)
        rhs1 = Tex(R"m {s}^2 X({s}) + \mu {s} X({s}) + k X({s}) = \frac{F_0 {s}}{({s}^2 + \omega^2)}", **kw)
        rhs2 = Tex(R"X({s}) \left( m {s}^2 + \mu {s} + k \right) = \frac{F_0 {s}}{({s}^2 + \omega^2)}", **kw)
        rhs3 = Tex(R"X({s}) = \frac{F_0 {s}}{\left({s}^2 + \omega^2\right) \left( m {s}^2 + \mu {s} + k \right)}", **kw)

        for sign, term in zip([-1, 1, 1, 1], [lhs, rhs1, rhs2, rhs3]):
            term.set_x(sign * FRAME_WIDTH / 4)
            term.set_y(3.25)
            term.scale(2)
            term.set_x(0)
            term.set_y(-sign * 2)

        arrow = Arrow(lhs, rhs1, thickness=6, buff=0.5)
        arrow_label = Tex(R"\mathcal{L}", font_size=72)
        arrow_label.next_to(arrow, RIGHT, SMALL_BUFF)

        ode_word = Text("Differential Equation")
        algebra_word = Text("Algebra")
        ode_word.next_to(lhs, DOWN)
        algebra_word.next_to(rhs1, DOWN)

        VGroup(ode_word, algebra_word).set_opacity(0)

        # Add domain backgrounds
        time_domain = FullScreenRectangle()
        time_domain.set_stroke(BLUE, 2)
        time_domain.set_fill(BLACK, 1)
        time_domain.stretch(0.5, 1, about_edge=UP)
        time_label = Text("Time domain")

        s_domain = time_domain.copy()
        s_domain.to_edge(DOWN, buff=0)
        s_domain.set_fill(GREY_E, 1)
        s_domain.set_stroke(YELLOW, 2)
        s_label = Text("s domain")

        for label, domain in [(time_label, time_domain), (s_label, s_domain)]:
            label.next_to(domain.get_corner(UL), DR)

        self.add(time_domain, time_label)
        self.add(s_domain, s_label)

        # Do the algebra
        self.play(
            FadeIn(lhs, lag_ratio=0.1),
            FadeIn(ode_word, lag_ratio=0.1),
        )
        self.wait()
        self.play(
            GrowArrow(arrow),
            FadeIn(arrow_label, DOWN),
        )
        self.play(
            TransformMatchingTex(
                lhs.copy(),
                rhs1,
                path_arc=-10 * DEG,
                lag_ratio=3e-2,
                key_map={
                    R"F_0 \cos(\omega t)": R"\frac{F_0 {s}}{({s}^2 + \omega^2)}",
                    "x(t) = ": "X({s}) = ",
                    "x'(t)": R"{s} X({s})",
                    "x''(t)": "{s}^2 X({s})",
                    "(t)": "({s})",
                }
            )
        )
        self.wait()
        self.play(
            FadeIn(algebra_word, 0.5 * DOWN),
            TransformMatchingTex(rhs1, rhs2, path_arc=-30 * DEG),
        )
        self.play(
            TransformMatchingTex(
                rhs2,
                rhs3,
                path_arc=-10 * DEG,
                matched_keys=[R"\left( m {s}^2 \mu s + k \right)", R"X({s})"]
            )
        )
        self.wait()

        # Show inversion
        inv_L = Tex(R"\mathcal{L}^{-1}", font_size=72)
        inv_L.next_to(arrow, LEFT, buff=0)

        xt = Text(R"Solution", font_size=72)
        xt.next_to(arrow, UP, MED_LARGE_BUFF)

        self.play(LaggedStart(
            Rotate(arrow, PI),
            ReplacementTransform(arrow_label, inv_L),
            lhs.animate.scale(0.5).to_corner(UR),
        ))
        self.play(FadeIn(xt, 2 * UP))
        self.wait()


class DesiredMachine(InteractiveScene):
    show_ode = True

    def construct(self):
        # Add machine
        machine = self.get_machine()
        machine.rotate(90 * DEG)
        machine.center()
        fancy_L = Tex(R"\mathcal{L}", font_size=120)
        fancy_L.move_to(machine)
        machine.set_z_index(1)
        fancy_L.set_z_index(1)

        self.add(machine, fancy_L)

        # Pump in a function
        t2c = {"{t}": BLUE, "{s}": YELLOW}
        in_func = Tex(R"x({t})", t2c=t2c, font_size=90)
        in_func[re.compile("s_.")].set_color(YELLOW)
        in_func.next_to(machine, UP, MED_LARGE_BUFF)
        in_func_ghost = in_func.copy().set_fill(opacity=0.5)

        self.play(Write(in_func))
        self.wait()
        self.add(in_func_ghost)
        self.play(
            FadeOutToPoint(in_func, machine.get_bottom(), lag_ratio=0.025)
        )
        self.wait()

        # Pump in a differential equation
        if show_ode:
            ode = Tex(R"m x''({t}) + \mu x'({t}) + k x(t) = F_0 \cos(\omega{t})", t2c=t2c, font_size=60)
            ode.next_to(machine, UP, MED_LARGE_BUFF)
            ode_ghost = ode.copy().set_fill(opacity=0.5)

            self.play(
                in_func_ghost.animate.to_edge(UP, buff=MED_SMALL_BUFF),
                FadeIn(ode, lag_ratio=0.1),
            )
            self.wait()
            self.add(ode_ghost)
            self.play(LaggedStart(
                (FadeOutToPoint(piece, machine.get_top() + 0.5 * DOWN, path_arc=arc)
                for piece, arc in zip(ode, np.linspace(-70 * DEG, 70 * DEG, len(ode)))),
                lag_ratio=0.05,
                run_time=2
            ))
            self.wait()

        # Result
        out_func = Tex(R"x({t}) = c_1 e^{s_1 {t}} + c_2 e^{s_2 {t}} + c_3 e^{s_3 {t}} + c_4 e^{s_4 {t}}", t2c=t2c, font_size=72)
        # out_func = Tex(R"x({t}) = \sum_{n=1}^N c_n e^{s_n {t}}", t2c=t2c, font_size=72)
        s_parts = out_func[re.compile("s_.")]
        c_parts = out_func[re.compile("c_.")]
        s_parts.set_color(YELLOW)
        c_parts.set_color(GREY_A)
        out_func.next_to(machine, DOWN, MED_LARGE_BUFF)

        self.play(LaggedStart(
            (FadeInFromPoint(piece, machine.get_bottom() + 0.5 * UP, path_arc=arc)
            for piece, arc in zip(out_func, np.linspace(-70 * DEG, 70 * DEG, len(out_func)))),
            lag_ratio=0.05,
            run_time=2
        ))

        # Make way for Laplace Transform words
        if False:
            # For an insert
            text = Text("Laplace Transform", font_size=72)
            machine.set_z_index(0)
            self.play(
                LaggedStart(
                    FadeOut(in_func_ghost, UP),
                    FadeOut(ode_ghost, 2 * UP),
                    FadeOut(out_func, DOWN),
                    FadeTransform(fancy_L[0], text[0]),
                    FadeIn(text[1:], lag_ratio=0.1),
                    run_time=2,
                    lag_ratio=0.2
                ),
                FadeOut(machine, scale=3, run_time=2)
            )
            self.wait()

        # Highlight s and c
        s_rects = VGroup(SurroundingRectangle(part, buff=0.05) for part in s_parts)
        c_rects = VGroup(SurroundingRectangle(part, buff=0.05) for part in c_parts)
        s_rects.set_stroke(YELLOW, 2)
        c_rects.set_stroke(WHITE, 2)

        s_part_copies = s_parts.copy()
        c_part_copies = c_parts.copy()

        self.add(s_part_copies)
        self.play(
            Write(s_rects),
            out_func.animate.set_opacity(0.75),
        )
        self.wait()
        self.play(
            ReplacementTransform(s_rects, c_rects, lag_ratio=0.1),
            FadeOut(s_part_copies),
            FadeIn(c_part_copies),
        )
        self.wait()
        self.play(FadeOut(c_rects), out_func.animate.set_fill(opacity=1))
        self.remove(c_part_copies)

        # Ask about exponential pieces
        mobs = Group(*self.mobjects)
        randy = Randolph()
        randy.move_to(5 * LEFT + 3 * DOWN, DL)
        randy.look_at(out_func),
        exp_piece = Tex(R"e^{{s}{t}}", t2c=t2c, font_size=90)
        exp_piece.next_to(randy, UR, LARGE_BUFF).shift(0.5 * DOWN)
        exp_piece.insert_submobject(2, VectorizedPoint(exp_piece[2].get_right()))

        self.play(
            LaggedStartMap(FadeOut, mobs, run_time=2),
            TransformFromCopy(out_func["e^{s_1 {t}}"][0], exp_piece, run_time=2),
            VFadeIn(randy, time_span=(0.5, 2.0)),
            randy.change("confused", exp_piece).set_anim_args(run_time=2),
        )
        self.play(Blink(randy))
        self.wait()
        for mode in ["pondering", "thinking", "tease"]:
            self.play(randy.change(mode, exp_piece))
            self.play(Blink(randy))
            self.wait(2)

    def get_machine(self, width=1.5, height=2, color=GREY_D):
        square = Rectangle(width, height)
        in_tri = ArrowTip().set_height(0.5 * height)
        in_tri.stretch(2, 1)
        out_tri = in_tri.copy().rotate(PI)
        in_tri.move_to(square.get_left())
        out_tri.move_to(square.get_right())
        machine = Union(square, in_tri, out_tri)
        machine.set_fill(color, 1)
        machine.set_stroke(WHITE, 2)
        return machine


class ExpDeriv(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"{t}": BLUE, "{s}": YELLOW}
        lhs, rhs = terms = VGroup(
            Tex(tex, t2c=t2c, font_size=90)
            for tex in [R"e^{{s}{t}}", R"{s} e^{{s}{t}}"]
        )
        terms.arrange(RIGHT, buff=4)

        arrows = VGroup(
            Arrow(
                lhs.get_corner(sign * UP + RIGHT),
                rhs.get_corner(sign * UP + LEFT),
                path_arc=-sign * 75 * DEG,
                thickness=4
            )
            for sign in [1, -1]
        )
        arrows.set_fill(border_width=2)
        arrows[1].shift(0.25 * DOWN)
        arrow_labels = VGroup(
            Tex(tex, t2c=t2c, font_size=60).next_to(arrow, vect)
            for arrow, tex, vect in zip(arrows, ["d / d{t}", R"\times {s}"], [UP, DOWN])
        )
        self.add(terms[0])
        self.play(
            TransformFromCopy(terms[0].copy(), terms[1][1:], path_arc=-75 * DEG),
            Write(arrows[0], time_span=(0.5, 1.5)),
            FadeIn(arrow_labels[0], lag_ratio=0.1),
            run_time=1.5,
        )
        self.play(FadeTransform(terms[1][2].copy(), terms[1][0], path_arc=90 * DEG, run_time=0.75))
        self.play(LaggedStart(
            TransformFromCopy(*arrows),
            TransformFromCopy(*arrow_labels),
        ))
        self.wait(2)


class IntroduceTransform(SPlane):
    long_ambient_graph_display = False

    def construct(self):
        # Write "Laplace Transform"
        text = Text("Laplace Transform", font_size=72)
        laplace_word = text["Laplace"][0]
        transform_word = text["Transform"][0]
        laplace_transform_word = VGroup(laplace_word, transform_word)
        transform_word_rect = SurroundingRectangle(transform_word, buff=SMALL_BUFF)
        randy = Randolph().flip()
        randy.next_to(laplace_transform_word, DR)
        q_marks = Tex(R"???", font_size=72).space_out_submobjects(1.5)
        q_marks.next_to(transform_word_rect, UP)
        q_marks.set_color(YELLOW)

        self.add(laplace_transform_word)
        self.wait()
        self.play(LaggedStart(
            VFadeIn(randy),
            randy.change("confused", transform_word),
            ShowCreation(transform_word_rect),
            laplace_word.animate.set_opacity(0.5),
            LaggedStartMap(FadeIn, q_marks, shift=0.5 * UP, lag_ratio=0.25),
            lag_ratio=0.1
        ))
        self.wait()

        # Show a function with an arrow
        t2c = {"{t}": BLUE, "{s}": YELLOW}
        name_group = VGroup(laplace_transform_word, transform_word_rect, q_marks)
        name_group.target = name_group.generate_target(use_deepcopy=True)
        name_group.target[2].set_fill(opacity=0).move_to(transform_word_rect)
        name_group.target.scale(0.5).to_edge(UP)
        name_group.target[1].set_stroke(width=0)

        mapsto = Tex(R"\xmapsto{\qquad}", additional_preamble=R"\usepackage{mathtools}")
        mapsto.rotate(-90 * DEG)
        mapsto.set_fill(border_width=2)

        t_mob, _, ft_mob = func_group = VGroup(Tex(R"{t}", t2c=t2c), mapsto, Tex(R"f({t})", t2c=t2c))
        func_group.arrange(DOWN)
        func_name = Text("function")
        func_name.next_to(mapsto, LEFT, SMALL_BUFF)

        mapsto.save_state()
        mapsto.stretch(0, 1, about_edge=UP).set_fill(opacity=0)

        self.play(
            MoveToTarget(name_group),
            randy.change("pondering", func_group),
            FadeIn(func_name, lag_ratio=0.1),
        )
        self.play(
            Restore(mapsto),
            FadeIn(t_mob, 0.1 * UP)
        )
        self.play(LaggedStart(
            TransformFromCopy(t_mob, ft_mob[2], path_arc=45 * DEG, run_time=1),
            FadeTransform(func_name[0].copy(), ft_mob[0], path_arc=45 * DEG, run_time=1),
            Write(ft_mob[1::2]),
            lag_ratio=0.25,
        ))
        self.wait()

        # Show the meta-notion
        func_group.generate_target()
        func_group.target.arrange(DOWN)
        func_group.target.move_to(4 * LEFT)

        braces = VGroup(
            Brace(func_group.target, direction, SMALL_BUFF).scale(1.25, about_point=func_group.target.get_center())
            for direction in [LEFT, RIGHT]
        )
        braces.save_state()
        braces.set_opacity(0)
        for brace in braces:
            brace.replace(func_group, dim_to_match=1)

        short_func_name = Tex(R"f")
        short_func_name.next_to(func_group.target[1], LEFT, buff=0)

        right_arrow = Vector(3 * RIGHT, thickness=6)
        right_arrow.next_to(braces.saved_state, RIGHT, MED_LARGE_BUFF)

        self.play(LaggedStart(
            MoveToTarget(func_group),
            Restore(braces),
            LaggedStart(
                [FadeTransform(func_name[0], short_func_name)] + [
                    char.animate.set_opacity(0).replace(short_func_name)
                    for char in func_name[1:]
                ],
                lag_ratio=0.02,
                remover=True
            ),
            GrowArrow(right_arrow),
            lag_ratio=0.05
        ))
        self.play(transform_word.animate.scale(1.5).next_to(right_arrow, UP))
        self.play(Blink(randy))
        self.wait()

        # Show output function
        f_group = VGroup(braces, t_mob, short_func_name, mapsto, ft_mob)
        F_braces, s_mob, F_mob, right_mapsto, Fs_mob = F_group = VGroup(
            braces.copy(),
            Tex(R"{s}", t2c=t2c),
            Tex(R"F"),
            mapsto.copy(),
            Tex(R"F({s})", t2c=t2c),
        )
        for mob1, mob2 in zip(f_group, F_group):
            mob2.move_to(mob1)
        F_group[2].shift(SMALL_BUFF * LEFT)

        F_group.next_to(right_arrow, RIGHT, MED_LARGE_BUFF)

        self.play(
            LaggedStart(
                (TransformFromCopy(*pair, path_arc=-60 * DEG)
                for pair in zip(f_group, F_group)),
                lag_ratio=0.025,
                # run_time=1.5
                run_time=8
            ),
            randy.change("tease").scale(0.75).to_corner(DR),
        )
        self.wait()

        # Talk through parts
        rect = SurroundingRectangle(F_group[-1]["F"], buff=0.05)
        rect.set_stroke(RED, 3)
        vect = Vector(0.5 * UP, thickness=4)
        vect.set_color(RED)
        vect.next_to(rect, DOWN, SMALL_BUFF)

        self.play(laplace_word.animate.set_fill(opacity=1).scale(1.5).next_to(transform_word, UP, MED_SMALL_BUFF, LEFT))
        self.play(
            ShowCreation(rect),
            GrowArrow(vect),
            randy.change("pondering", rect),
        )
        self.wait()
        self.play(
            rect.animate.surround(f_group[-1][0], buff=0.05),
            MaintainPositionRelativeTo(vect, rect),
        )
        self.wait()
        self.play(Blink(randy))
        self.play(
            rect.animate.surround(f_group[1], buff=0.05).set_anim_args(path_arc=-90 * DEG),
            vect.animate.rotate(PI).next_to(f_group[1], UP, SMALL_BUFF).set_anim_args(path_arc=-90 * DEG),
        )
        self.wait()
        self.play(
            rect.animate.surround(F_group[1], buff=0.05).set_anim_args(path_arc=-90 * DEG),
            MaintainPositionRelativeTo(vect, rect),
            randy.animate.look_at(F_group),
        )
        self.play(Blink(randy))
        self.wait()
        self.play(FadeOut(vect), FadeOut(rect))

        # Show Laplace Transform expression
        lt_def = Tex(R"\int^\infty_0 f({t}) e^{\minus {s}{t}} d{t}", t2c=t2c)
        lt_def.move_to(Fs_mob, LEFT)
        new_interior = VGroup(s_mob, F_mob, right_mapsto, lt_def)
        F_braces.generate_target()
        F_braces.target.set_height(new_interior.get_height() + MED_LARGE_BUFF, stretch=True)
        F_braces.target.match_y(right_arrow)
        F_braces.target[1].next_to(new_interior, RIGHT).match_y(F_braces.target[0])

        q_marks = Tex(R"???").set_color(YELLOW)
        q_marks.next_to(randy, UP, SMALL_BUFF).shift(0.25 * RIGHT + 0.5 * DOWN)

        self.play(
            FadeOut(Fs_mob),
            Write(lt_def),
            randy.change("pleading").scale(0.75, about_edge=DR),
            FadeIn(q_marks, 0.1 * UP, lag_ratio=0.1),
            MoveToTarget(F_braces),
            new_interior[:3].animate.match_x(lt_def).shift(0.1 * UP),
        )
        self.play(Blink(randy))
        self.wait()

        # Discuss expression
        rect = SurroundingRectangle(lt_def)
        rect.set_stroke(YELLOW, 2)

        s_vect = Vector(0.25 * DOWN)
        s_vect.set_fill(YELLOW)
        s_vect.next_to(lt_def["{s}"][0], UP, SMALL_BUFF)

        self.play(
            ShowCreation(rect),
            randy.change('confused')
        )
        self.wait()
        self.play(
            Transform(rect, s_vect.copy().match_style(rect)),
            FadeIn(s_vect, time_span=(0.75, 1)),
        )
        self.play(
            randy.change("pondering", s_vect),
            FadeOut(q_marks),
            FadeOut(rect),
        )
        self.play(Blink(randy))
        self.wait()

        # Show F(s) left hand side
        F_lhs = Tex(R"F({s}) = ", t2c=t2c)
        F_lhs.move_to(lt_def, LEFT)

        right_shift = F_lhs.get_width() * RIGHT

        self.play(
            TransformFromCopy(s_mob, F_lhs["{s}"][0]),
            TransformFromCopy(F_mob, F_lhs["F"][0]),
            Write(F_lhs[re.compile(r"\(|\)|=")]),
            lt_def.animate.shift(right_shift),
            MaintainPositionRelativeTo(s_vect, lt_def),
            F_braces[1].animate.shift(right_shift),
            new_interior[:3].animate.shift(right_shift + 0.1 * UP),
            randy.animate.shift(0.5 * DOWN),
            self.frame.animate.shift(0.5 * DOWN),
        )

        # Reference the naming convention
        F_rect = SurroundingRectangle(F_lhs[0], buff=0.05)
        F_rect.set_stroke(RED, 2)

        self.play(
            ShowCreation(F_rect),
            s_vect.animate.next_to(F_rect, UP, SMALL_BUFF).set_color(RED)
        )
        self.wait()
        self.play(
            F_rect.animate.surround(ft_mob[0], buff=0.05),
            s_vect.animate.next_to(ft_mob[0], UP, SMALL_BUFF),
            randy.animate.look_at(ft_mob)
        )
        self.play(Blink(randy))
        self.wait()
        self.play(FadeOut(F_rect), FadeOut(s_vect))

        # Clear the board
        lt_group = VGroup(F_lhs, lt_def)

        self.remove(lt_group)
        everything = Group(*self.get_mobjects())
        everything.sort(lambda p: np.dot(p, UL))

        self.play(
            LaggedStartMap(FadeOut, everything, shift=0.1 * DOWN, lag_ratio=0.1),
            lt_group.animate.center().set_height(2),
            self.frame.animate.to_default_state(),
            run_time=2,
        )

        # Highlight inner part
        outer_part = VGroup(lt_def[R"\int^\infty_0"][0], lt_def[R"d{t}"][0])
        inner_part = lt_def[R"f({t}) e^{\minus {s}{t}}"][0]

        rect = SurroundingRectangle(inner_part, buff=SMALL_BUFF)
        outer_rects = VGroup(SurroundingRectangle(piece, buff=SMALL_BUFF) for piece in outer_part)
        VGroup(rect, outer_rects).set_stroke(RED, 2)

        self.play(
            ShowCreation(rect),
            outer_part.animate.set_fill(opacity=0.2),
            F_lhs.animate.set_fill(opacity=0.2),
        )
        self.wait()
        self.play(
            # TransformFromCopy(rect.replicate(2), outer_rects),
            rect.animate.surround(outer_part, buff=SMALL_BUFF),
            outer_part.animate.set_fill(opacity=1),
            inner_part.animate.set_fill(opacity=0.5),
        )
        self.wait()

        # Put at the top
        lt_group = VGroup(F_lhs, outer_part, inner_part)
        lt_group.target = lt_group.generate_target()
        lt_group.target.set_height(1.2).to_edge(UP, buff=MED_SMALL_BUFF)
        lt_group.target[:2].set_fill(opacity=0.2)
        lt_group.target[2].set_fill(opacity=1)

        lt_group.target.set_fill(opacity=1)

        self.play(
            MoveToTarget(lt_group),
            # rect.animate.surround(lt_group.target[2]).set_stroke(width=1),
            rect.animate.surround(lt_group.target).set_stroke(width=0),
        )
        self.wait()
        self.play(
            rect.animate.surround(lt_def["s"][0], buff=0.05).set_stroke(YELLOW)
        )
        self.wait()

        # Show s as a complex number
        plane = ComplexPlane((-3, 3), (-3, 3))
        plane.next_to(lt_group, DOWN)
        plane.add_coordinate_labels(font_size=16)

        s_dot = Group(TrueDot(), GlowDot()).set_color(YELLOW)
        s_dot.move_to(plane.n2p(2 + 1j))
        s_label = Tex(R"s").set_color(YELLOW)
        s_label.always.next_to(s_dot[0], UR, SMALL_BUFF)

        self.play(
            Write(plane, lag_ratio=1e-2),
            TransformFromCopy(lt_def["s"][0], s_label),
            FadeInFromPoint(s_dot, rect.get_center()),
            FadeOut(rect),
        )
        self.wait()

        # Wander
        final_z = -2 + 1j
        n_iterations = 12
        for n in range(12):
            z = complex(*np.random.uniform(-3, 3, 2))
            if n == n_iterations - 1:
                z = final_z
            self.play(s_dot.animate.move_to(plane.n2p(z)).set_anim_args(path_arc=45 * DEG))
            self.wait()

        # Plug in cos(t)
        lt_def.refresh_bounding_box()
        lt_of_cos = Tex(lt_def.get_tex().replace("f({t})", R"\cos({t})"), t2c=t2c)
        lt_of_cos.move_to(lt_def, LEFT)
        cos_lt_group = VGroup(
            F_lhs,
            VGroup(lt_of_cos[R"\int^\infty_0"][0], lt_of_cos[R"d{t}"][0]),
            lt_of_cos[R"\cos({t}) e^{\minus {s}{t}}"][0]
        )
        cos_lt_group[1].set_fill(opacity=0.2)

        imag_circles = VGroup(
            Dot(plane.n2p(z)).set_stroke(YELLOW, 1).set_fill(opacity=0.25)
            for z in [1j, -1j]
        )

        f_term = lt_def["f"][-1]
        cos_term = lt_of_cos[R"\cos"][0]
        f_rect = SurroundingRectangle(f_term, buff=0.05)
        f_rect.set_stroke(PINK, 2)

        self.play(ShowCreation(f_rect))
        self.play(
            TransformMatchingTex(
                lt_def,
                lt_of_cos,
                matched_pairs=[(f_term, cos_term)],
                run_time=1
            ),
            f_rect.animate.surround(cos_term, buff=0.05),
            F_lhs.animate.shift((lt_of_cos.get_width() - lt_def.get_width()) * 0.5 * LEFT),
            *map(ShowCreation, imag_circles),
        )
        self.play(FadeOut(f_rect))
        self.wait()

        # Create the graph
        frame = self.frame
        frame.to_default_state()
        cos_lt_group.fix_in_frame()
        self.add(cos_lt_group)

        graph = get_complex_graph(
            plane,
            lambda s: s / (s**2 + 1),
            resolution=(501, 501),
            mesh_resolution=(31, 31)
        )
        graph.stretch(0.25, 2, about_point=plane.n2p(0))

        # Show the graph
        self.play(
            cos_lt_group.animate.to_edge(LEFT, buff=MED_SMALL_BUFF).set_fill(opacity=1),
            ShowCreation(graph[0]),
            Write(graph[1], stroke_width=1, time_span=(2, 4)),
            frame.animate.reorient(-28, 74, 0, OUT, 8.37),
            run_time=4,
        )
        frame.clear_updaters()
        frame.add_ambient_rotation()
        for z in [1j, -1j]:
            self.play(s_dot.animate.move_to(plane.n2p(z)), run_time=2)
            self.wait(2)

        # Long ambient graph rotation (one branch here)
        if self.long_ambient_graph_display:
            self.remove(cos_lt_group)
            self.wait(19)
            self.add(imag_circles, graph)
            self.play(FadeOut(graph), FadeOut(imag_circles))
            frame.clear_updaters()
            s_plane = plane
            s_plane.save_state()
            s_plane.set_height(3.5)
            s_plane.to_corner(UR, buff=MED_SMALL_BUFF).shift(LEFT)
            s_plane.background_lines.set_stroke(BLUE, 1, 1)
            s_plane.target = s_plane.copy()

            s_dot.target = s_dot.generate_target()
            s_dot.target.move_to(s_plane.n2p(0.2 + 2j))

            s_plane.restore()
            self.play(
                frame.animate.to_default_state(),
                MoveToTarget(s_plane),
                MoveToTarget(s_dot),
                run_time=2
            )

        # Remove graphs
        self.remove(plane, graph, s_dot, s_label, imag_circles)
        frame.to_default_state()
        frame.clear_updaters()

        # Ignore the integral
        interior = cos_lt_group[2]
        interior_rect = SurroundingRectangle(interior, buff=0.05)
        interior_rect.set_stroke(TEAL, 2)

        self.add(interior)
        self.play(
            ShowCreation(interior_rect),
            cos_lt_group[:2].animate.set_fill(opacity=0.2),
        )
        self.wait()

        # Replace break down cos(t) with exponentials
        arrow = Vector(DOWN)
        arrow.match_color(interior_rect)
        arrow.next_to(interior_rect, DOWN, SMALL_BUFF)
        expanded = Tex(
            R"\frac{1}{2} \left(e^{i{t}} + e^{\minus i{t}} \right) e^{\minus{s}{t}}",
            t2c={"i": WHITE, "-i": WHITE, **t2c},
        )
        expanded.next_to(arrow, DOWN, MED_LARGE_BUFF)
        expanded_brace = Brace(expanded, UP, SMALL_BUFF)
        expanded_brace.set_color(TEAL)

        index = -4
        self.play(
            FadeTransform(interior[:index].copy(), expanded[:index]),
            TransformFromCopy(interior[index:], expanded[index:]),
            GrowArrow(arrow),
            FadeInFromPoint(expanded_brace, arrow.get_start()),
        )
        self.wait()

        # Focus on just part
        eit, est = pair = VGroup(expanded["e^{i{t}}"][0], expanded[R"e^{\minus{s}{t}}"][0])
        pair_rects = VGroup(SurroundingRectangle(p, buff=0.05) for p in pair)
        pair_rects.set_stroke(BLUE, 2)
        pair_copy = pair.copy()

        self.add(pair_copy),
        self.play(
            expanded.animate.set_fill(opacity=0.5),
            ShowCreation(pair_rects, lag_ratio=0.5)
        )
        self.wait()

        # Combine
        combined_term = Tex(R"e^{(i - {s}){t}}", t2c=t2c)
        combined_term.next_to(pair, DOWN, LARGE_BUFF)
        comb_arrows = VGroup(Arrow(p, combined_term, buff=0.25, thickness=2) for p in pair)
        comb_arrows.set_fill(BLUE)

        self.play(
            TransformMatchingShapes(pair.copy(), combined_term),
            *map(GrowArrow, comb_arrows),
            run_time=1
        )
        self.wait()

        # Pull up s-plane and output plane
        s_plane = plane
        s_plane.set_height(3.5)
        s_plane.to_corner(UR, buff=MED_SMALL_BUFF).shift(LEFT)
        s_plane.background_lines.set_stroke(BLUE, 1, 1)

        s_tracker = ComplexValueTracker(0.2 + 2j)
        t_tracker = ValueTracker(0)
        get_s = s_tracker.get_value
        get_t = t_tracker.get_value

        def get_ims():
            return 1j - get_s()

        s_dot.add_updater(lambda m: m.move_to(s_plane.n2p(get_s())))

        exp_plane = self.get_exp_plane()
        exp_plane.set_height(3.5)
        exp_plane.next_to(s_plane, DOWN, MED_LARGE_BUFF)
        output_label = self.get_output_dot_and_label(
            exp_plane,
            get_s=get_ims,
            get_t=get_t,
            s_tex="(i - {s})"
        )
        output_path = self.get_output_path(exp_plane, get_t, get_ims)

        self.play(FadeIn(s_plane), FadeIn(s_dot), FadeIn(s_label))
        self.play(
            FadeIn(exp_plane),
            FadeTransform(combined_term.copy(), output_label[1]),
            FadeIn(output_label[0]),
        )
        self.add(output_path)
        self.play(t_tracker.animate.set_value(2 * TAU), rate_func=linear, run_time=10)
        self.play(s_tracker.animate.increment_value(-0.3), rate_func=there_and_back, run_time=4)
        self.play(s_tracker.animate.set_value(0.2), run_time=4)
        self.play(s_tracker.animate.set_value(1j), run_time=4)
        self.wait()

        # Let t play
        eq_i_rhs = Tex(R"= i")
        eq_i_rhs.next_to(s_label, RIGHT, SMALL_BUFF).shift(0.04 * UP)
        self.play(Write(eq_i_rhs), run_time=1)

        t_tracker.set_value(0)
        self.play(t_tracker.animate.set_value(20), run_time=20, rate_func=linear)

        # Show s = i
        down_arrow = Vector(0.75 * DOWN)
        down_arrow.next_to(combined_term, DOWN, SMALL_BUFF)
        arrow_label = Tex(R"{s} = i", font_size=24, t2c=t2c)
        arrow_label.next_to(down_arrow, RIGHT, buff=0)
        const = Tex(R"e^{0{t}}", t2c=t2c)
        const.next_to(down_arrow, DOWN, SMALL_BUFF)
        eq_1 = Tex(R"= 1")
        eq_1.next_to(const, RIGHT, SMALL_BUFF, aligned_edge=DOWN)

        self.play(
            GrowArrow(down_arrow),
            FadeIn(arrow_label),
            FadeIn(const, DOWN),
        )
        self.wait()
        self.play(Write(eq_1, run_time=1))
        self.wait()

        # Move around s
        s_tracker.clear_updaters()
        t_tracker.set_value(100)
        output_label.clear_updaters()

        self.play(
            FadeOut(eq_i_rhs),
            FadeOut(output_label),
        )

        self.play()
        for _ in range(6):
            self.play(
                s_tracker.animate.set_value(complex(random.uniform(-0.1, 0.5), random.uniform(-3, 3))),
                run_time=2
            )
            self.wait()
        self.play(s_tracker.animate.set_value(1j), run_time=2)
        self.remove(output_path)

        self.wait()

        # State the goal
        big_rect = SurroundingRectangle(expanded[:-4])
        big_rect.set_stroke(BLUE, 2)
        goal = Text("Goal:\nReveal these terms", alignment="LEFT")
        goal.next_to(big_rect, DOWN, aligned_edge=LEFT)

        self.play(
            ReplacementTransform(pair_rects, VGroup(big_rect)),
            LaggedStartMap(FadeOut, VGroup(comb_arrows, combined_term, down_arrow, arrow_label, const, eq_1)),
            FadeOut(Group(exp_plane, s_plane, s_label, s_dot)),
            expanded.animate.set_fill(opacity=1),
        )
        self.play(FadeIn(goal, lag_ratio=0.1))
        self.wait()
        self.play(
            FadeOut(goal),
            big_rect.animate.surround(expanded),
        )
        self.wait()

        # Highlight the integral again
        self.add(expanded)
        self.remove(pair_copy)
        self.play(
            LaggedStartMap(FadeOut, VGroup(interior_rect, arrow, expanded_brace, expanded, big_rect)),
            cos_lt_group.animate.set_fill(opacity=1),
        )
        self.wait()


class SimpleToComplex(InteractiveScene):
    def construct(self):
        # Add key expressions
        t2c = {"{t}": BLUE, "{s}": YELLOW}
        terms = VGroup(
            Tex(R"f({t}) = 1", t2c=t2c),
            Tex(R"f({t}) = e^{at}", t2c=t2c),
            Tex(R"f({t}) = \sum_{n=1}^N c_n e^{s_n {t}}", t2c=t2c),
        )
        terms.scale(1.5)
        terms.arrange(DOWN, buff=1.5)
        arrows = VGroup(
            Arrow(*pair, buff=0.25, thickness=5)
            for pair in zip(terms, terms[1:])
        )
        terms[2].shift(SMALL_BUFF * DOWN)

        self.add(terms[0])
        for arrow, term1, term2 in zip(arrows, terms, terms[1:]):
            self.play(
                TransformMatchingTex(
                    term1.copy(),
                    term2,
                    key_map={"1": "e^{at}", "a": "s_n"}
                ),
                GrowArrow(arrow),
                run_time=1
            )
            self.wait()

        # Set up Laplace Transforms
        lt_left_x = terms.get_x(RIGHT) + 3
        lt_arrows = VGroup(
            Arrow(
                term.get_right(),
                lt_left_x * RIGHT + term.get_y() * UP,
                buff=0.5,
                thickness=6
            )
            for term in terms
        )
        lt_arrows.set_color(GREY_A)
        lt_arrow_labels = VGroup(
            Tex(R"\mathcal{L}").next_to(arrow, UP, buff=0)
            for arrow in lt_arrows
        )
        lt_integrals = VGroup(
            Tex(
                R"\int^\infty_0 " + tex + R"\cdot e^{\minus{s}{t}}d{t}",
                t2c=t2c
            )
            for tex in ["1", "e^{a{t}}", R"\sum_{n=1}^N c_n e^{s_n {t}}"]
        )
        lt_rhss = VGroup(
            Tex(R"= \frac{1}{{s}}", t2c=t2c),
            Tex(R"= \frac{1}{{s} - a}", t2c=t2c),
            Tex(R"= \sum_{n=1}^N \frac{c_n}{{s} - s_n}", t2c=t2c),
        )
        for integral, arrow, rhs in zip(lt_integrals, lt_arrows, lt_rhss):
            integral.next_to(arrow, RIGHT)
            rhs.scale(1.25)
            rhs.next_to(integral, RIGHT)

        # Show laplace transforms
        frame = self.frame

        self.play(
            LaggedStart(
                GrowArrow(lt_arrows[0]),
                Write(lt_arrow_labels[0]),
                FadeIn(lt_integrals[0], RIGHT),
                lag_ratio=0.3
            ),
            terms[1:].animate.set_fill(opacity=0.1),
            arrows.animate.set_fill(opacity=0.1),
            frame.animate.set_x(4),
            run_time=2
        )
        self.wait()
        self.play(Write(lt_rhss[0]))
        self.wait()
        for index in [1, 2]:
            self.play(
                arrows[index - 1].animate.set_fill(opacity=1),
                terms[index].animate.set_fill(opacity=1),
            )
            self.play(
                LaggedStart(
                    GrowArrow(lt_arrows[index]),
                    Write(lt_arrow_labels[index]),
                    FadeIn(lt_integrals[index], RIGHT),
                    lag_ratio=0.3
                ),
            )
            self.wait()
            self.play(Write(lt_rhss[index]))
            self.wait()
        self.play(frame.reorient(0, 0, 0, (5.5, 0.04, 0.0), 10), run_time=2)


class SetSToMinus1(InteractiveScene):
    def construct(self):
        eq = Tex(R"\frac{1}{\minus 1} = \minus 1")
        eq[R"\minus 1"][0].set_color(YELLOW)
        self.play(Write(eq))
        self.wait()


class RealExtension(InteractiveScene):
    def construct(self):
        # Show limited domain
        axes = Axes((-1, 10), (-1, 5), width=FRAME_WIDTH - 1, height=6)
        self.add(axes)

        def func(x):
            decay = math.exp(-0.05 * (x + 1))
            poly = -0.003 * x**3 - 0.2 * (0.15 * x)**2 + 0.2 * x
            return (decay + 0.5) * (math.cos(1.0 * x) + 1.5) + poly

        limited_domain = (2, 6)

        partial_graph = axes.get_graph(func, x_range=limited_domain)
        partial_graph.set_stroke(BLUE, 5)
        f_label = Tex(R"f(x)")
        f_label.next_to(partial_graph.get_end(), UL)

        limited_domain_line = Line(
            axes.c2p(limited_domain[0], 0),
            axes.c2p(limited_domain[1], 0),
        )
        limited_domain_line.set_stroke(BLUE, 5)
        limited_domain_words = Text("Limited Domain")
        limited_domain_words.next_to(limited_domain_line, UP, SMALL_BUFF)

        self.add(axes)
        self.play(
            ShowCreation(partial_graph),
            Write(f_label)
        )
        self.play(
            ShowCreation(limited_domain_line),
            FadeIn(limited_domain_words, lag_ratio=0.1)
        )
        self.wait()
        self.play(TransformFromCopy(limited_domain_line, partial_graph))
        self.wait()

        # Extend the graph
        points = partial_graph.get_anchors()

        def get_extension(nudge_size=0):
            pre_xs = np.arange(1, -2, -1)
            post_xs = np.arange(7, 11)
            result = VGroup(
                self.get_extension(axes, points[3::-1], pre_xs, func, nudge_size=nudge_size),
                self.get_extension(axes, points[-4:], post_xs, func, nudge_size=nudge_size),
            )
            result[0].set_clip_plane(LEFT, axes.c2p(limited_domain[0], 0)[0])
            result[1].set_clip_plane(RIGHT, -axes.c2p(limited_domain[1], 0)[0])
            return result

        extension = get_extension()
        self.play(ShowCreation(extension, lag_ratio=0, run_time=4))

        # Change around
        extension.save_state()
        for n in range(5):
            new_extension = get_extension(nudge_size=3)
            self.play(extension.animate.become(new_extension), run_time=1)
        self.play(Restore(extension))

        # Show a derivative
        x_tracker = ValueTracker(limited_domain[0])
        tan_line = always_redraw(lambda : axes.get_tangent_line(
            x_tracker.get_value(), partial_graph, length=2
        ).set_stroke(WHITE, 3))

        self.play(GrowFromCenter(tan_line, suspend_mobject_updating=True))
        self.play(x_tracker.animate.set_value(limited_domain[1]), run_time=5)
        self.play(FadeOut(tan_line, suspend_mobject_updating=True))

        # Wiggly spaghetti
        def tweaked_func(x):
            x0, x1 = limited_domain
            if x < x0:
                return func(x) + 0.5 * (x - x0)**2
            elif x < x1:
                return func(x)
            else:
                return func(x) - 0.5 * (x - x1)**2

        full_graph = axes.get_graph(func)
        modifed_graph = axes.get_graph(tweaked_func)

        group = VGroup(full_graph, modifed_graph)
        group.set_stroke(RED, 5)
        group.set_z_index(-1)

        self.play(
            FadeOut(extension),
            FadeIn(full_graph),
        )
        self.play(
            full_graph.animate.become(modifed_graph),
            rate_func=lambda t: wiggle(t, 5),
            run_time=8
        )

    def get_extension(self, axes, pre_start, xs, func, nudge_size=0, stroke_width=5, stroke_color=RED):
        ys = np.array([func(x) + (nudge_size * (random.random() - 0.5)) for x in xs])
        new_points = axes.c2p(xs, ys)

        result = VMobject()
        result.set_points_smoothly([*pre_start, *new_points], approx=False)
        result.insert_n_curves(100)

        result.set_stroke(stroke_color, stroke_width)
        result.set_z_index(-1)
        return result


class ComplexExtension(InteractiveScene):
    def construct(self):
        # Set up input and output planes
        input_plane, output_plane = planes = VGroup(
            ComplexPlane((-3, 3), (-3, 3))
            for n in range(2)
        )
        planes.set_height(5)
        planes.arrange(RIGHT, buff=1.5)
        for plane in planes:
            plane.axes.set_stroke(WHITE, 1, 1)
            plane.background_lines.set_stroke(BLUE, 1, 0.5)
            plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)

        self.add(planes)

        # Set up limited domain
        domain = self.get_rect_group(2, 1, input_plane, 0.2 + 0.2j)
        domain.set_z_index(2)
        self.add(domain)

        # Show a mapping
        def func(z):
            return -0.05j * z**3

        def point_func(points):
            return np.array([
                output_plane.n2p(func(input_plane.p2n(p)))
                for p in points
            ])

        mapped_domain = domain.copy().apply_points_function(point_func, about_edge=None)
        mapped_domain.set_z_index(2)

        arrow = Arrow(
            domain.get_top(),
            mapped_domain.get_top(),
            path_arc=-90 * DEG,
            fill_color=TEAL,
            thickness=5
        )
        arrow.set_z_index(1)
        func_label = Tex(R"f(z)")
        func_label.next_to(arrow, UP, SMALL_BUFF)
        func_label.set_backstroke()

        self.play(
            Write(arrow, time_span=(0, 1)),
            Write(func_label, time_span=(0.5, 1.5)),
            ReplacementTransform(
                domain.copy().set_fill(opacity=0),
                mapped_domain,
                path_arc=-90 * DEG,
                run_time=3
            )
        )

        # Show the extension
        dark_red = interpolate_color(RED_E, BLACK, 0.5)
        extension = self.get_extended_domain(input_plane, domain, 4, 2, corner_value=-1, color=dark_red)
        mapped_extension = extension.copy().apply_points_function(point_func, about_edge=None)

        self.play(Write(extension, run_time=1, lag_ratio=1e-2))
        self.wait()
        self.play(
            TransformFromCopy(extension.copy().set_fill(opacity=0), mapped_extension, path_arc=-90 * DEG),
            run_time=3
        )
        self.wait()

        # Two possibilities
        frame = self.frame
        possibilities = VGroup(
            Text("1) There is no such extension", font_size=72),
            Text("2) There is only one extension", font_size=72),
        )
        possibilities.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        possibilities.next_to(planes, UP, buff=LARGE_BUFF)
        only_one = possibilities[1]["only one"][0]
        underline = Underline(only_one, buff=-SMALL_BUFF).set_stroke(YELLOW)

        self.play(
            Write(possibilities[0][:2]),
            Write(possibilities[1][:2]),
            frame.animate.reorient(0, 0, 0, (0, 1.5, 0.0), 9).set_anim_args(time_span=(0, 1)),
            run_time=2
        )
        self.wait()
        self.play(FadeIn(possibilities[0][2:], lag_ratio=0.1))
        self.wait()
        self.play(FadeIn(possibilities[1][2:], lag_ratio=0.1))
        self.play(
            ShowCreation(underline),
            only_one.animate.set_fill(YELLOW)
        )
        self.wait()

        # Analytic continuation
        ac_words = Text("Analytic Continuation", font_size=72)
        ac_words.next_to(output_plane, UP, MED_LARGE_BUFF)

        darkest_red = interpolate_color(RED_E, BLACK, 0.9)
        big_domain = self.get_extended_domain(input_plane, extension, 6, 6, corner_value=-3 - 3j, color=darkest_red)
        big_domain.set_stroke(WHITE, 0.5, 0.1)
        big_domain.set_fill(dark_red, 0.5)
        mapped_big_domain = big_domain.copy().apply_points_function(point_func, about_edge=None)

        self.play(
            FadeOut(possibilities, lag_ratio=0.05),
            FadeOut(underline),
            Write(ac_words, time_span=(0.5, 2.)),
        )
        self.play(
            Write(mapped_big_domain, lag_ratio=5e-2, stroke_color=RED, stroke_width=0.5),
            Write(big_domain, lag_ratio=5e-2, stroke_color=RED, stroke_width=0.5),
            run_time=8
        )
        self.wait()

    def get_extended_domain(self, plane, domain, width, height, corner_value, color=RED_E):
        extension = self.get_rect_group(width, height, plane, corner_value=corner_value, color=color)

        min_x = domain.get_x(LEFT)
        max_x = domain.get_x(RIGHT)
        min_y = domain.get_y(DOWN)
        max_y = domain.get_y(UP)
        to_remove = list()
        for square in extension:
            if (min_x < square.get_x() < max_x) and (min_y < square.get_y() < max_y):
                to_remove.append(square)
        extension.remove(*to_remove)
        extension.sort(lambda p: get_norm(p - domain.get_center()))
        return extension

    def get_rect_group(self, width, height, plane, corner_value=0, square_density=5, color=BLUE_E):
        square = Square(side_length=plane.x_axis.get_unit_size() / square_density)
        square.set_stroke(WHITE, 0.5)
        square.set_fill(color, 1)
        square.insert_n_curves(20)
        grid = square.get_grid(square_density * height, square_density * width, buff=0)
        grid.move_to(plane.n2p(corner_value), DL)
        return grid

        rect = Rectangle(width, height)
        rect.set_width(width * plane.x_axis.get_unit_size())
        rect.move_to(plane.n2p(corner_value), DL)
        rect.set_stroke(color, 2)
        rect.set_fill(color, 1)
        rect.insert_n_curves(200)

        rect_lines = VGroup(
            Line(DOWN, UP).get_grid(1, width * line_density + 1, buff=SMALL_BUFF),
            Line(LEFT, RIGHT).get_grid(height * line_density + 1, 1, buff=SMALL_BUFF),
        )
        for group in rect_lines:
            group.replace(rect, stretch=True)
            group.set_stroke(WHITE, 1, 0.5)
            for line in group:
                line.insert_n_curves(20)

        return VGroup(rect, rect_lines)


class WriteFPrimeExists(InteractiveScene):
    def construct(self):
        words = TexText("$f'(z)$ Exists")
        self.play(Write(words))
        self.wait()


class ZetaFunctionPlot(InteractiveScene):
    # resolution = (51, 51)
    resolution = (1001, 1001)  # Probably takes like an hour to compute

    def construct(self):
        # Planes
        x_max = 25
        s_plane = ComplexPlane((-x_max, x_max), (-x_max, x_max), faded_line_ratio=5)
        s_plane.set_height(40)
        s_plane.add_coordinate_labels(font_size=16)

        partial_plane = ComplexPlane((1, x_max), (-x_max, x_max))
        partial_plane.shift(s_plane.n2p(0) - partial_plane.n2p(0))

        self.add(s_plane)

        # True function
        import mpmath as mp

        def zeta_log_deriv(s):
            epsilon = 1e-4
            if s == 1:
                return 1 / epsilon
            out = mp.zeta(s)
            if abs(out) < 1e-3:
                return 1 / epsilon
            out_prime = (mp.zeta(s + epsilon) - out) / epsilon
            # return mp.zeta(s, derivative=1) / out
            return out_prime / out

        graph = get_complex_graph(
            s_plane,
            zeta_log_deriv,
            resolution=self.resolution,
        )
        graph.set_clip_plane(RIGHT, -1)
        graph.set_opacity(0.6)

        self.add(graph)

        # Panning
        frame = self.frame
        frame.reorient(24, 82, 0, (0.39, 0.58, 0.49), 4.02)
        self.play(
            frame.animate.reorient(-14, 79, 0, (-0.62, -0.08, 1.41), 8.12),
            run_time=8
        )
        self.play(
            graph.animate.set_clip_plane(RIGHT, x_max),
            frame.animate.reorient(-13, 80, 0, (-0.28, -0.01, 2.17), 12.57),
            run_time=5
        )
        self.play(
            frame.animate.reorient(32, 81, 0, (-0.98, -2.23, 5.19), 32.31),
            run_time=20
        )
        self.play(
            frame.animate.reorient(91, 84, 0, (-0.25, -1.54, 7.01), 32.31),
            run_time=15,
        )


class WriteZetaPrimeFact(InteractiveScene):
    def construct(self):
        # Test
        formula = Tex(
            R"\frac{\zeta'({s})}{\zeta({s})} = \sum_{spacer} \sum_{k=1}^\infty \frac{1}{k} \frac{1}{p^{s}}",
            t2c={"{s}": YELLOW}
        )
        spacer = formula["spacer"][0]
        p_prime = TexText("$p$ prime")
        p_prime.replace(spacer)
        p_prime.scale(0.8, about_edge=UP)
        formula.remove(*spacer)
        formula.add(*p_prime)
        formula.sort()

        formula.to_edge(UP, buff=MED_SMALL_BUFF)
        self.play(Write(formula))
        self.wait()


class SimpleExpToPole(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"{t}": BLUE, "{s}": YELLOW}
        kw = dict(t2c=t2c, font_size=72)
        lhs, arrow, rhs = group = VGroup(
            Tex(R"e^{a{t}}", **kw),
            Vector(1.5 * RIGHT, thickness=5),
            Tex(R"{1 \over {s} - a}", **kw)
        )
        group.arrange(RIGHT, buff=MED_LARGE_BUFF)
        group.to_edge(UP, MED_LARGE_BUFF)
        fancy_L = Tex(R"\mathcal{L}")
        fancy_L.next_to(arrow, UP, SMALL_BUFF)

        self.play(FadeIn(lhs))
        self.play(
            GrowArrow(arrow),
            Write(fancy_L),
            run_time=1,
        )
        self.play(
            Write(rhs[:-1]),
            TransformFromCopy(lhs["a"], rhs["a"])
        )
        self.wait()


class Linearity(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"f": GREEN, "g": BLUE, R"\mathcal{L}": GREY_A, R"\big\{": WHITE, R"\big\}": WHITE, "a": GREEN_A, "b": BLUE_A}
        lhs = Tex(R"\mathcal{L}\big\{a \cdot f(t) + b \cdot g(t) \big\}", t2c=t2c)
        rhs = Tex(R"a \cdot \mathcal{L} \big\{ f(t) \big\} + b \cdot \mathcal{L} \big\{ g(t) \big\}", t2c=t2c)
        arrow = Vector(RIGHT, thickness=4)
        group = VGroup(lhs, arrow, rhs)
        group.arrange(RIGHT)

        self.add(lhs)
        self.wait()
        self.play(
            GrowArrow(arrow),
            LaggedStart(
                AnimationGroup(*(
                    TransformFromCopy(lhs[tex][0], rhs[tex][0], path_arc=45 * DEG)
                    for tex in [R"\mathcal{L}", R"\big\{", R"a \cdot", "f(t)", R"\big\}"]
                )),
                AnimationGroup(*(
                    TransformFromCopy(lhs[tex][0], rhs[tex][-1], path_arc=-45 * DEG)
                    for tex in ["+", R"\mathcal{L}", R"\big\{", R"b \cdot", "g(t)", R"\big\}"]
                )),
                lag_ratio=0.75,
                run_time=3
            ),
        )
        self.wait()


class LaplaceTransformOfCosineSymbolically(InteractiveScene):
    def construct(self):
        # Add defining integral
        frame = self.frame
        t2c = {
            "{t}": BLUE,
            "{s}": YELLOW,
            R"\omega": PINK,
            R"int^\infty_0": WHITE,
        }
        key_strings = [
            R"int^\infty_0",
            R"e^{\minus{s}{t}} d{t}",
            "+",
            R"\frac{1}{2}",
            R"e^{i{t}}",
            R"e^{\minus i{t}}",
        ]
        kw = dict(isolate=key_strings, t2c=t2c)

        cos_t = Tex(R"\cos({t})", **kw)
        cos_t.to_corner(UL, buff=LARGE_BUFF)
        arrow = Vector(1.5 * RIGHT)
        arrow.next_to(cos_t)
        fancy_L = Tex(R"\mathcal{L}")
        fancy_L.next_to(arrow, UP, SMALL_BUFF)

        def lt_string(interior):
            return Rf"\int^\infty_0 " + interior + R"e^{\minus{s}{t}} d{t}"

        lt_def = Tex(lt_string(R"\cos({t})"), **kw)
        lt_def.next_to(arrow, RIGHT)

        self.add(cos_t)
        self.play(LaggedStart(
            GrowArrow(arrow),
            Write(fancy_L),
            Write(lt_def[R"\int^\infty_0"]),
            TransformFromCopy(cos_t, lt_def[R"\cos({t})"][0], path_arc=45 * DEG),
            Write(lt_def[R"e^{\minus{s}{t}} d{t}"]),
            lag_ratio=0.2,
        ))

        # Split up into exponential parts
        spilt_cos_str = R"\left( \frac{1}{2} e^{i{t}} + \frac{1}{2} e^{\minus i{t}} \right)"
        split_inside = Tex("=" + lt_string(spilt_cos_str), **kw)
        split_inside.next_to(lt_def, RIGHT)

        cos_rect = SurroundingRectangle(lt_def[R"\cos({t})"])
        cos_rect.set_stroke(TEAL, 2)

        self.play(ShowCreation(cos_rect))
        self.play(
            TransformMatchingTex(
                lt_def.copy(),
                split_inside,
                key_map={R"\cos({t})": spilt_cos_str},
                path_arc=30 * DEG,
                mismatch_animation=FadeTransform,
            ),
            cos_rect.animate.surround(split_inside[spilt_cos_str]).set_anim_args(path_arc=30 * DEG),
            run_time=1.5
        )
        self.play(FadeOut(cos_rect))
        self.wait()
        self.add(split_inside)

        # Rect growth
        self.play(cos_rect.animate.surround(split_inside[1:]).set_stroke(width=5))
        self.wait()
        self.play(FadeOut(cos_rect))

        # Linearity
        split_tex = " ".join([
            R"\frac{1}{2}", lt_string(R"e^{i{t}}"), R"\, + \,",
            R"\frac{1}{2}", lt_string(R"e^{\minus i{t}}"),
        ])
        split_outside = Tex(split_tex, **kw)
        side_eq = Tex(R"=", font_size=72).rotate(90 * DEG)
        side_eq.next_to(split_inside, DOWN, MED_LARGE_BUFF)
        split_outside.next_to(side_eq, DOWN, MED_LARGE_BUFF)
        split_outside.shift_onto_screen()

        srcs = VGroup()
        trgs = VGroup()
        for tex in key_strings:
            src = split_inside[tex]
            trg = split_outside[tex]
            if tex is key_strings[0]:
                src = VGroup(part[:3] for part in src)
                trg = VGroup(part[:3] for part in trg)
            srcs.add(src)
            trgs.add(trg)

        self.play(
            Write(side_eq),
            LaggedStart(
                (TransformFromCopy(*pair)
                for pair in zip(srcs[:3], trgs[:3])),
                lag_ratio=0.01,
                run_time=2
            ),
        )
        self.wait()
        self.play(
            TransformFromCopy(srcs[3][0], trgs[3][0]),
            TransformFromCopy(srcs[4][0], trgs[4][0])
        )
        self.wait()
        self.play(
            TransformFromCopy(srcs[3][1], trgs[3][1]),
            TransformFromCopy(srcs[5][0], trgs[5][0])
        )
        self.wait()

        # Collapse to poles
        exp_transform_parts = VGroup(
            split_outside[lt_string(R"e^{i{t}}")],
            split_outside[lt_string(R"e^{\minus i{t}}")],
        )
        pole_strings = [R"\frac{1}{{s} - i}", R"\frac{1}{{s} \, + \, i}"]
        half_string = R"\frac{1}{2}"
        pole_sum = Tex(
            R" \, ".join([half_string, pole_strings[0], "+", half_string, pole_strings[1]]),
            **kw
        )
        pole_sum.scale(1.25)
        pole_sum.move_to(split_outside).shift(0.2 * LEFT)

        split_inside_rect = SurroundingRectangle(split_inside[spilt_cos_str])
        exp_transform_rects = VGroup(
            SurroundingRectangle(part, buff=SMALL_BUFF)
            for part in exp_transform_parts
        )
        pole_rects = VGroup(
            SurroundingRectangle(pole_sum[tex], buff=SMALL_BUFF)
            for tex in pole_strings
        )

        VGroup(split_inside_rect, exp_transform_rects, pole_rects).set_stroke(TEAL, 2)

        self.play(ShowCreation(split_inside_rect))
        self.wait()
        self.play(LaggedStart(*(
            TransformFromCopy(split_inside_rect, rect)
            for rect in exp_transform_rects
        )))
        self.play(FadeOut(split_inside_rect))
        self.wait()
        for i, tex in enumerate([R"e^{i{t}}", R"e^{\minus i{t}}"]):
            self.play(
                ReplacementTransform(exp_transform_rects[i], pole_rects[i]),
                ReplacementTransform(split_outside[half_string][i], pole_sum[half_string][i]),
                FadeTransform(split_outside[lt_string(tex)], pole_sum[pole_strings[i]]),
                Transform(split_outside["+"][0], pole_sum["+"][0])
            )
            self.play(FadeOut(pole_rects[i]))
        self.remove(split_outside)
        self.add(pole_sum)
        self.play(pole_sum.animate.match_x(side_eq))

        # Read it as "pole at i", etc.
        pole_rects = VGroup(
            SurroundingRectangle(pole_sum[tex], buff=SMALL_BUFF)
            for tex in pole_strings
        )
        pole_rects.set_stroke(YELLOW, 2)
        pole_words = VGroup(
            TexText(Rf"Pole at \\ $s = {value}$", font_size=60, t2c={"Pole at": YELLOW, "s": YELLOW})
            for value in ["i", "-i"]
        )

        last_group = VGroup()
        for word, rect in zip(pole_words, pole_rects):
            word.next_to(rect, DOWN, MED_LARGE_BUFF)
            self.play(
                FadeIn(word, lag_ratio=0.1),
                ShowCreation(rect),
                FadeOut(last_group)
            )
            self.wait()
            last_group = VGroup(word, rect)

        self.play(FadeOut(last_group))

        # Add an omega
        old_group = VGroup(cos_t, lt_def, split_inside, pole_sum)
        new_group = VGroup(
            Tex(R"\cos(\omega{t})", **kw),
            Tex(lt_string(R"\cos(\omega{t})"), **kw),
            Tex("=" + lt_string(R"\left(\frac{1}{2} e^{i\omega{t}} + \frac{1}{2}e^{\minus i \omega {t}} \right)"), **kw),
            Tex(R" \, ".join([
                half_string, R"\frac{1}{{s} - \omega i}", "+",
                half_string, R"\frac{1}{{s} \, + \, \omega i}",
            ]), **kw)
        )
        for new, old in zip(new_group, old_group):
            new.match_width(old)
            new.move_to(old)

        omegas = VGroup()
        for new in new_group:
            omegas.add(*new[R"\omega"])

        omega_copies = omegas.copy()
        omegas.set_fill(opacity=0)
        omegas[0].set_fill(opacity=1)

        cos_omega = new_group[0]
        cos_omega.scale(1.25, about_edge=RIGHT)
        cos_omega_rect = SurroundingRectangle(cos_omega)
        cos_omega_rect.set_stroke(PINK, 2)

        self.play(
            ShowCreation(cos_omega_rect),
            TransformMatchingTex(cos_t, cos_omega),
            run_time=1
        )
        self.wait()
        self.play(
            LaggedStart(
                (TransformMatchingTex(old, new)
                for new, old in zip(new_group[1:], old_group[1:])),
                lag_ratio=0.05,
                run_time=1
            ),
            TransformFromCopy(
                omegas[0].replicate(len(omega_copies) - 1),
                omega_copies[1:],
                path_arc=30 * DEG,
                lag_ratio=0.1,
                run_time=2
            ),
        )
        self.remove(omega_copies)
        omegas.set_fill(opacity=1)
        self.add(new_group)
        self.play(FadeOut(cos_omega_rect))
        self.wait()

        # Simplify fraction
        lower_arrow = Tex(R"\longleftarrow", font_size=60)
        lower_arrow.next_to(pole_sum, LEFT)

        transform_kw = dict(
            matched_keys=[
                R"{s}^2 \,+\, \omega^2",
                R"{s} \,+\, \omega i",
                R"{s} - \omega i",
                R"\over",
            ],
            key_map={
                R"({s} - \omega i)({s} + \omega i)": R"{s}^2 \,+\, \omega^2"
            }
        )

        steps = VGroup(
            Tex(R"""
                \frac{1}{2}\left(
                {{s} \,+\, \omega i \over ({s} - \omega i)({s} + \omega i)} +
                {{s} - \omega i \over ({s} - \omega i)({s} + \omega i)}
                \right)
            """, **kw),
            Tex(R"""
                \frac{1}{2}\left(
                {{s} \,+\, \omega i \over {s}^2 \,+\, \omega^2} +
                {{s} - \omega i \over {s}^2 \,+\, \omega^2}
                \right)
            """, **kw),
            Tex(R"""
                \frac{1}{2} {{s} \,+\, \omega i \,+\, {s} - \omega i \over {s}^2 \,+\, \omega^2}
            """, **kw),
            Tex(R"""
                \frac{1}{2} {2{s} \over {s}^2 \,+\, \omega^2}
            """, **kw),
            Tex(R"{{s} \over {s}^2 \,+\, \omega^2}", **kw),
        )
        for step in steps:
            step.next_to(lower_arrow, LEFT)

        self.play(
            Write(lower_arrow),
            FadeTransform(pole_sum.copy(), steps[0]),
            frame.animate.set_height(8.5, about_edge=DR),
            run_time=2
        )
        for step1, step2 in zip(steps, steps[1:]):
            self.play(
                TransformMatchingTex(step1, step2, **transform_kw)
            )
            self.wait()

        # Circle answer
        answer = steps[-1]
        answer.target = answer.generate_target()
        answer.target.scale(1.5, about_edge=RIGHT)
        answer_rect = SurroundingRectangle(answer.target)
        answer_rect.set_stroke(TEAL, 3)
        self.play(
            ShowCreation(answer_rect),
            MoveToTarget(answer)
        )
        self.wait()

        # Highlight direct equality
        lt_def, int_of_expanded, imag_result = new_group[-3:]

        direct_equals = Tex(R"=", font_size=90)
        direct_equals.rotate(90 * DEG)
        direct_equals.next_to(lt_def, DOWN, MED_LARGE_BUFF)

        to_fade = VGroup(int_of_expanded, side_eq, imag_result)

        self.play(
            lower_arrow.animate.set_fill(opacity=0),
            to_fade.animate.set_fill(opacity=0.2),
            answer_rect.animate.surround(VGroup(lt_def, answer)),
            Write(direct_equals),
        )
        self.wait()
        self.play(
            lower_arrow.animate.set_fill(opacity=1).rotate(PI).set_anim_args(path_arc=PI),
            answer_rect.animate.surround(VGroup(answer, imag_result)),
            imag_result.animate.set_fill(opacity=1),
        )
        self.wait()


class SimplePolesOverImaginaryLine(InteractiveScene):
    def construct(self):
        s_plane = ComplexPlane((-5, 5), (-5, 5))
        s_plane.add_coordinate_labels()
        omega = 2
        graph = get_complex_graph(
            s_plane,
            lambda s: (s**2) / (s**2 + omega**2 + 1e-6),
            resolution=(301, 301)
        )
        graph[0].sort_faces_back_to_front(DOWN)
        graph[1].set_clip_plane(OUT, 0)
        self.add(s_plane, graph)

        # Pan
        frame = self.frame
        frame.reorient(-57, 78, 0, (-1.28, 0.48, 0.92), 9.65)
        self.play(frame.animate.reorient(59, 78, 0, (-0.31, -0.1, 0.87), 10.53), run_time=12)


class IntegrationByParts(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"{s}": YELLOW, "{t}": BLUE, R"\omega": PINK}
        steps = VGroup(
            Tex(R"X = \int^\infty_0 \cos(\omega{t}) e^{\minus {s}{t}}d{t}", t2c=t2c),
            Tex(R"X = \left[\frac{1}{\omega} \sin(\omega{t}) e^{\minus {s}{t}} \right]_0^\infty - \int^\infty_0 \frac{1}{\omega} \sin(\omega {t}) \left(\minus {s} e^{\minus{s}{t}} \right) d{t}", t2c=t2c),
            Tex(R"X = \frac{s}{\omega} \int^\infty_0 \sin(\omega{t}) e^{\minus{s}{t}} d{t}", t2c=t2c),
            Tex(R"X = \frac{s}{\omega} \left(\left[\frac{\minus 1}{\omega} \cos(\omega{t}) e^{\minus {s}{t}} \right]_0^\infty - \int^\infty_0 \frac{\minus 1}{\omega} \cos(\omega{t}) \left(\minus {s} e^{\minus{s}{t}} \right) d{t} \right)", t2c=t2c),
            Tex(R"X = \frac{s}{\omega} \left(\frac{1}{\omega} - \frac{s}{\omega} \int^\infty_0 \cos(\omega{t}) e^{\minus{s}{t}} d{t} \right)", t2c=t2c),
            Tex(R"X = \frac{s}{\omega^2} \left(1 - {s} X \right)", t2c=t2c),
            Tex(R"X\left(\omega^2 + {s}^2 \right) = {s}", t2c=t2c),
            Tex(R"X = \frac{s}{\omega^2 + {s}^2}", t2c=t2c),
        )
        steps.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        steps.to_edge(LEFT, buff=LARGE_BUFF)

        randy = Randolph(mode="raise_left_hand")
        randy.to_edge(DOWN)
        steps[0].save_state()
        steps[0].next_to(randy, UL, MED_LARGE_BUFF)
        randy.look_at(steps[0])

        ibp = Tex(R"\int u \, dv = uv - \int v \, du", t2c={"u": RED, "v": PINK})
        ibp.next_to(randy, UR, MED_LARGE_BUFF)

        self.add(randy, steps[0])
        self.play(Blink(randy))
        self.play(
            randy.change("raise_right_hand", ibp),
            FadeIn(ibp, UP)
        )
        self.wait()
        self.play(
            self.frame.animate.set_height(steps.get_height() + 3, about_edge=LEFT),
            Restore(steps[0]),
            Write(steps[1:], run_time=2),
            randy.change("pondering", 5 * UL).shift(6 * RIGHT + 2 * DOWN),
            ibp.animate.shift(6 * RIGHT + 3 * DOWN),
        )
        self.play(randy.animate.look_at(steps[-1]))
        self.wait()


class AlternateBreakDown(TranslateToNewLanguage):
    def construct(self):
        # Set up
        axes, graph, graph_label = self.get_graph_group(
            lambda t: 0.35 * t**2,
            func_tex=R"f({t}) = {t}^2"
        )
        graph_label.set_backstroke(BLACK, 8)
        s_samples = self.get_s_samples()
        s_plane, exp_pieces, s_plane_name = self.get_s_plane_and_exp_pieces(s_samples)
        arrow, fancy_L, Fs_label = self.get_arrow_to_Fs(graph_label)

        self.add(axes, graph, graph_label)
        self.add(exp_pieces)

        # Note equal to a sum
        ne = Tex(R"\ne", font_size=96)
        ne.next_to(graph_label, RIGHT, MED_LARGE_BUFF)
        ne.set_color(RED)
        sum_tex = Tex(
            R"\sum_{n=1}^N c_n e^{s_n t}",
            t2c={"s_n": YELLOW, "c_n": GREY_A},
            font_size=72,
        )
        sum_tex.next_to(ne, RIGHT)
        ne_rhs = VGroup(ne, sum_tex)

        self.play(LaggedStart(
            FadeIn(ne, scale=2),
            Write(sum_tex),
            exp_pieces.animate.scale(0.7, about_edge=DR),
            lag_ratio=0.2
        ))
        self.wait()

        # Show transform
        self.play(
            LaggedStart(
                FadeOut(graph_label[4:]),
                graph_label[:4].animate.next_to(arrow, LEFT),
                GrowArrow(arrow),
                Write(fancy_L),
                TransformFromCopy(graph_label, Fs_label, path_arc=20 * DEG),
                lag_ratio=0.05
            ),
            FadeOut(ne_rhs, DOWN, scale=0.5),
        )
        self.wait()

        # Show integral
        inv_lt = Tex(
            R"f({t}) = \frac{1}{2\pi i} \int_\gamma F({s}) e^{{s}{t}} d{s}",
            t2c=self.label_config["t2c"]
        )
        inv_lt.next_to(arrow, DOWN, LARGE_BUFF)

        s_plane.replace(exp_pieces)
        s_plane.add_coordinate_labels(font_size=12)
        line = Line(s_plane.get_bottom(), s_plane.get_top())
        line.shift(RIGHT)
        line.set_stroke(YELLOW, 2)
        line.insert_n_curves(20)

        self.play(
            TransformFromCopy(graph_label[:4], inv_lt[:4]),
            TransformFromCopy(Fs_label, inv_lt["F({s})"][0]),
            Write(inv_lt[4:]),
        )
        self.play(
            FadeIn(s_plane),
            LaggedStartMap(FadeOut, exp_pieces, scale=0.1),
        )
        self.play(
            VShowPassingFlash(line.copy()),
            ShowCreation(line),
            run_time=5
        )
        self.wait()