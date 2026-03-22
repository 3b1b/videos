from manim_imports_ext import *
from _2025.laplace.exponentials import *


def z_to_color(z, sat=0.5, lum=0.5):
    angle = math.atan2(z.imag, z.real)
    return Color(hsl=(angle / TAU, sat, lum))


def get_complex_graph(
    s_plane,
    func,
    min_real=None,
    pole_buff=1e-3,
    color_by_phase=True,
    opacity=0.7,
    shading=(0.1, 0.1, 0.1),
    resolution=(301, 301),
    saturation=0.5,
    luminance=0.5,
    face_sort_direction=UP,
    mesh_resolution=(61, 61),
    mesh_stroke_style=dict(
        color=WHITE,
        width=1,
        opacity=0.15
    )
):
    u_range = list(s_plane.x_range[:2])
    v_range = list(s_plane.y_range[:2])

    if min_real is not None:
        u_range[0] = min_real + pole_buff

    unit_size = s_plane.x_axis.get_unit_size()
    graph = ParametricSurface(
        lambda u, v: [
            *s_plane.c2p(u, v)[:2],
            unit_size * abs(func(complex(u, v)))
        ],
        u_range=u_range,
        v_range=v_range,
        resolution=resolution
    )
    graph.set_shading(*shading)

    if color_by_phase:
        graph.color_by_uv_function(
            lambda u, v: z_to_color(func(complex(u, v)), sat=saturation, lum=luminance)
        )

    graph.set_opacity(opacity)
    graph.sort_faces_back_to_front(face_sort_direction)

    # Add mesh
    mesh = SurfaceMesh(graph, resolution=mesh_resolution)
    mesh.set_stroke(**mesh_stroke_style)

    return Group(graph, mesh)


class IntegrateConstant(InteractiveScene):
    def construct(self):
        # Axes and graph
        axes = Axes((0, 100), (0, 2, 0.25), width=100, height=3)
        axes.add_coordinate_labels(num_decimal_places=2, font_size=20)
        axes.to_corner(DL)

        graph = axes.get_graph(lambda t: 1)
        graph.set_stroke(BLUE, 3)
        graph_label = Tex(R"f(t) = 1")
        graph_label.next_to(graph, UP)
        graph_label.to_edge(RIGHT)

        self.add(axes)
        self.add(graph, graph_label)

        # Integral
        t_tracker = ValueTracker(0.01)
        get_t = t_tracker.get_value

        v_tracker = ValueTracker(1)
        t_tracker.add_updater(lambda m, dt: m.increment_value(v_tracker.get_value() * dt))

        integral = Tex(R"\int^{0.01}_0 1 \, dt = 0.01", font_size=72)
        integral.to_edge(UP, buff=3.0)
        integral["1"].set_color(BLUE)
        decimals = integral.make_number_changeable("0.01", replace_all=True)
        decimals[0].scale(0.5, about_edge=LEFT)
        integral[2].scale(0.5, about_edge=LEFT)
        integral[:3].shift(0.5 * RIGHT)
        integral.set_scale_stroke_with_zoom(True)
        for dec in decimals:
            dec.add_updater(lambda m: m.set_value(get_t()))

        rect = Rectangle()
        rect.set_fill(BLUE, 0.5)
        rect.set_stroke(WHITE, 2)
        rect.set_z_index(-1)
        x_unit = axes.x_axis.get_unit_size()
        y_unit = axes.y_axis.get_unit_size()
        origin = axes.get_origin()
        rect.add_updater(
            lambda m: m.set_shape(get_t() * x_unit, y_unit).move_to(origin, DL)
        )

        integral.fix_in_frame()

        self.add(rect, integral, t_tracker)

        # Grow
        v_tracker.set_value(1)
        self.play(
            v_tracker.animate.set_value(12),
            self.frame.animate.reorient(0, 0, 0, (42.64, 14.41, 0.0), 58.35).set_anim_args(time_span=(6, 15)),
            run_time=20
        )


class IntegrateRealExponential(InteractiveScene):
    def construct(self):
        # Add integral expression
        t2c = {R"{s}": YELLOW}
        integral = Tex(R"\int^\infty_0 e^{\minus {s}t} dt", t2c=t2c)
        integral.set_x(1)
        integral.to_edge(UP)
        integral.save_state()
        integral.scale(1.5, about_edge=UP)
        self.add(integral)

        # Add the graph of e^{-st}
        max_x = 15
        unit_size = 4
        axes = Axes((0, max_x, 0.25), (0, 1, 0.25), unit_size=unit_size)
        axes.to_edge(DL, buff=1.0)
        axes.add_coordinate_labels(num_decimal_places=2, font_size=20)

        def exp_func(t):
            return np.exp(-get_s() * t)

        s_tracker = ValueTracker(1)
        get_s = s_tracker.get_value
        graph = axes.get_graph(np.exp)
        graph.set_stroke(BLUE, 3)
        axes.bind_graph_to_func(graph, exp_func)

        graph_label = Tex(R"e^{\minus {s}t}", t2c=t2c, font_size=72)
        graph_label.next_to(axes.y_axis.get_top(), UR).shift(0.5 * RIGHT)

        self.play(
            FadeIn(axes),
            TransformFromCopy(integral[R"e^{\minus {s}t}"], graph_label),
            ShowCreation(graph, suspend_mobject_updating=True, run_time=3),
            Restore(integral),
        )

        # Add a slider for s
        s_slider = Slider(s_tracker, x_range=(0, 5), var_name="s")
        s_slider.scale(1.5)
        s_slider.to_edge(UP, buff=MED_LARGE_BUFF)
        s_slider.align_to(axes.c2p(0, 0), LEFT)

        self.play(VFadeIn(s_slider))
        for value in [5, 0.25, 1]:
            self.play(
                s_tracker.animate.set_value(value),
                run_time=4
            )
            self.wait()

        # Show integral as area
        equals = Tex(R"=", font_size=72).rotate(90 * DEG)
        equals.next_to(integral, DOWN)
        area_word = Text("Area", font_size=60)
        area_word.next_to(equals, DOWN)

        area = axes.get_area_under_graph(graph)

        def update_area(area):
            area.become(axes.get_area_under_graph(graph))

        arrow = Arrow(area_word.get_corner(DL), axes.c2p(0.75, 0.5), thickness=4)

        self.play(
            LaggedStart(
                Animation(graph.copy(), remover=True),
                Write(equals),
                FadeIn(area_word, DOWN),
                GrowArrow(arrow),
                UpdateFromFunc(area, update_area),
                lag_ratio=0.25
            ),
            ShowCreation(graph, suspend_mobject_updating=True, run_time=3),
        )
        self.wait()

        # Try altenrate s value
        frame = self.frame

        area.clear_updaters()
        area.add_updater(update_area)
        self.play(
            s_tracker.animate.set_value(-0.2),
            frame.animate.reorient(0, 0, 0, (16.79, 7.82, 0.0), 28.01).set_anim_args(time_span=(2, 5)),
            run_time=5
        )

        # Show area = 1 from s = 1
        simple_integral = Tex(R"\int^\infty_0 e^{\minus t} dt")
        simple_integral.move_to(integral)
        simple_exp = Tex(R"e^{\minus t}")
        simple_exp.move_to(graph_label)

        anti_deriv = Tex(R"=\big[\minus e^{\minus t} \big]^\infty_0")
        simple_rhs = Tex(R"=0 - (\minus 1)")
        anti_deriv.next_to(simple_integral, RIGHT)
        simple_rhs.next_to(anti_deriv, RIGHT)

        equals_one = Tex(R"= 1", font_size=60)
        equals_one.next_to(area_word)

        area_one_label = Tex(R"1", font_size=60)
        area_one_label.move_to(axes.c2p(0.35, 0.35))
        area_one_label.set_z_index(1)

        self.remove(integral, graph_label)
        self.play(
            TransformMatchingTex(integral.copy(), simple_integral),
            TransformMatchingTex(graph_label.copy(), simple_exp),
            run_time=1
        )
        self.play(
            TransformMatchingTex(simple_integral.copy(), anti_deriv, run_time=1.5, path_arc=30 * DEG),
        )
        rect_kw = dict(buff=0.05, stroke_width=1.5)
        self.play(
            FadeIn(simple_rhs[:2], time_span=(0, 1)),
            VFadeInThenOut(SurroundingRectangle(anti_deriv[R"\minus e^{\minus t}"], **rect_kw)),
            VFadeInThenOut(SurroundingRectangle(anti_deriv[R"\infty"], **rect_kw)),
            VFadeInThenOut(SurroundingRectangle(simple_rhs[:2], **rect_kw)),
            run_time=1.5
        )
        self.play(
            FadeIn(simple_rhs[2:], time_span=(0, 1)),
            VFadeInThenOut(SurroundingRectangle(anti_deriv[R"\minus e^{\minus t}"], **rect_kw)),
            VFadeInThenOut(SurroundingRectangle(anti_deriv[R"0"], **rect_kw)),
            VFadeInThenOut(SurroundingRectangle(simple_rhs[2:], **rect_kw)),
            run_time=1.5
        )
        self.wait()
        self.play(TransformMatchingTex(simple_rhs.copy(), equals_one, run_time=1))
        self.play(TransformFromCopy(equals_one["1"], area_one_label))
        self.wait()

        # Comapre to unit square
        square = Polygon(
            axes.c2p(0, 1),
            axes.c2p(1, 1),
            axes.c2p(1, 0),
            axes.c2p(0, 0),
        )
        square.set_stroke(WHITE, 3)
        square.set_fill(BLUE, 0.0)

        area_s_tracker = ValueTracker(get_s())
        area_x_max_tracker = ValueTracker(graph.x_range[1])
        squishy_area = always_redraw(  # Currently unused
            lambda: axes.get_area_under_graph(axes.get_graph(
                lambda t: np.exp(-area_s_tracker.get_value() * t),
                x_range=(0, area_x_max_tracker.get_value())
            ))
        )

        tail_area = axes.get_area_under_graph(graph, x_range=(1, 6))
        tail_area.set_fill(RED_E, 0.75)
        corner_area = axes.get_graph(lambda t: np.exp(-get_s() * t), (0, 1))
        corner_area.add_line_to(axes.c2p(1, 1))
        corner_area.add_line_to(axes.c2p(0, 1))
        corner_area.match_style(tail_area)
        corner_area.set_z_index(-2)

        area_one_label.save_state()

        self.play(FadeIn(tail_area))
        self.wait()
        self.play(
            ShowCreation(square),
            FadeIn(corner_area),
            area_one_label.animate.move_to(square),
        )
        self.wait()
        self.play(
            FadeOut(square),
            FadeOut(corner_area),
            FadeOut(tail_area),
            Restore(area_one_label),
            FadeOut(anti_deriv),
            FadeOut(simple_rhs)
        )
        self.wait()

        # Show squishing the area
        stretch_label = VGroup(
            TexText(R"Squish by $\frac{1}{s}$", t2c=t2c),
            Vector(2 * LEFT, thickness=5, fill_color=YELLOW)
        )
        stretch_label.arrange(DOWN, buff=MED_SMALL_BUFF)
        stretch_label.move_to(axes.c2p(0.5, 0.5))

        area_word.set_z_index(1)
        area_word.target = area_word.generate_target()
        area_word.target.move_to(axes.c2p(0.6, 0.33))

        rhs = Tex(R"= \frac{1}{s}", t2c=t2c, font_size=60)
        rhs.next_to(area_word.target, RIGHT)

        area.set_z_index(-1)
        area.add_updater(update_area)

        self.play(LaggedStart(
            TransformMatchingTex(simple_integral, integral, run_time=1),
            FadeIn(graph_label, 0.5 * DOWN),
            FadeOut(simple_exp, 0.5 * DOWN),
            FadeOut(equals_one, 0.5 * DOWN),
            FadeOut(area_one_label),
            lag_ratio=0.1
        ))
        self.play(
            s_tracker.animate.set_value(5).set_anim_args(run_time=8),
            FadeIn(stretch_label, 1.5 * LEFT, time_span=(1, 5)),
            FadeOut(arrow),
        )
        self.wait()
        self.play(
            LaggedStart(
                FadeOut(stretch_label),
                MoveToTarget(area_word),
                FadeTransform(stretch_label[0][-3:].copy(), rhs[1:]),
                FadeTransform(stretch_label[1].copy(), rhs[0]),
                FadeOut(equals),
                run_time=2,
                lag_ratio=0.1
            )
        )

        # Show area value
        dec_rhs = Tex(R"= 1.00", font_size=60)
        dec_rhs.make_number_changeable("1.00").add_updater(lambda m: m.set_value(1 / get_s()))
        dec_rhs.always.next_to(rhs, RIGHT)

        self.play(
            VFadeIn(dec_rhs),
            s_tracker.animate.set_value(0.01).set_anim_args(run_time=12, rate_func=bezier([0, 1, 1, 1])),
            self.frame.animate.reorient(0, 0, 0, (15.36, 0, 0.0), 30).set_anim_args(time_span=(6, 11)),
        )
        self.wait()
        self.play(
            VGroup(area_word, rhs).animate.next_to(axes.c2p(0, 0), UR, MED_LARGE_BUFF).set_anim_args(time_span=(1, 3)),
            s_tracker.animate.set_value(0.75),
            VFadeOut(dec_rhs, time_span=(2, 4)),
            self.frame.animate.to_default_state(),
            run_time=4,
        )
        self.wait()

        # Averages over intervals
        v_lines = VGroup(
            DashedLine(axes.c2p(0, 0), axes.c2p(0, 1.25)),
            DashedLine(axes.c2p(1, 0), axes.c2p(1, 1.25)),
        )
        v_lines.set_stroke(WHITE, 1)

        unit_int = Tex(R"\int^1_0 e^{\minus {s}t} dt", t2c=t2c, font_size=60)
        unit_int.move_to(v_lines, UP)

        graph_for_unit_area = axes.get_graph(exp_func)
        graph_for_unit_area.set_stroke(width=0)
        unit_int_area = always_redraw(
            lambda: axes.get_area_under_graph(graph_for_unit_area, (0, 1))
        )
        avg_value = exp_func(np.linspace(0, 1, 1000)).mean()

        def slosh_rate_func(t, cycles=3):
            return min(smooth(2 * cycles * t), 1) + 0.7 * math.sin(cycles * TAU * t) * (t - 1)**2

        self.remove(integral)
        area.clear_updaters()
        self.play(
            *map(FadeOut, [area_word, rhs, graph_label, s_slider]),
            *map(ShowCreation, v_lines),
            TransformMatchingTex(integral.copy(), unit_int),
            FadeOut(area),
            FadeIn(unit_int_area, time_span=(0.5, 1), suspend_mobject_updating=True),
        )
        self.wait()
        self.play(
            Transform(
                graph_for_unit_area,
                axes.get_graph(lambda t: avg_value).set_stroke(width=0),
                run_time=3,
                rate_func=slosh_rate_func,
            ),
        )
        unit_int_area.suspend_updating()

        # Show average
        avg_label1 = self.get_avg_label(unit_int_area, 0, 1)
        avg_label1.save_state()
        avg_label1.space_out_submobjects(0.5)
        avg_label1.set_opacity(0)
        self.play(Restore(avg_label1))
        self.wait()

        # Emphasize that it's a unit interval
        frame = self.frame
        unit_line = Line(axes.c2p(0, 0), axes.c2p(1, 0))
        unit_line.set_stroke(YELLOW, 5)
        brace = Brace(unit_line, DOWN, buff=MED_LARGE_BUFF)
        unit_label = brace.get_tex("1", buff=MED_SMALL_BUFF, font_size=72)
        unit_group = VGroup(brace, unit_line, unit_label)

        self.play(LaggedStart(
            GrowFromCenter(brace),
            ShowCreation(unit_line),
            FadeIn(unit_label, 0.25 * DOWN),
            frame.animate.set_y(-1.5),
            run_time=1.5
        ))
        self.wait()

        # Area = height
        height_line = Line(unit_int_area.get_corner(DL), unit_int_area.get_corner(UL))
        height_line.set_stroke(RED, 5)

        area_eq_height = TexText(R"= Area = Width $\times$ Height", t2c={"Area": BLUE, "Height": RED, "Width": YELLOW})
        area_eq_height.next_to(unit_int, RIGHT)
        fade_rect = BackgroundRectangle(area_eq_height, buff=0.25, fill_opacity=1)
        fade_rect.stretch(2, 1, about_edge=DOWN)

        sample_dots = DotCloud([axes.c2p(a, exp_func(a)) for a in np.linspace(0, 1, 30)])
        sample_dots.set_color(WHITE)
        sample_dots.set_glow_factor(2)
        sample_dots.set_radius(0.15)

        self.play(
            FadeIn(fade_rect),
            FadeIn(area_eq_height),
            ShowCreation(height_line),
        )
        self.wait()
        self.play(
            area_eq_height[R"Width $\times$"].animate.set_opacity(0),
            area_eq_height[R"Height"].animate.move_to(area_eq_height["Width"], UL),
        )
        self.wait()
        self.play(FlashAround(unit_int, run_time=2, time_width=1.5))
        self.wait()

        self.play(ShowCreation(sample_dots))
        self.play(sample_dots.animate.stretch(0, 1).match_y(axes.c2p(0, avg_value)), rate_func=lambda t: slosh_rate_func(t, 2), run_time=3)
        self.play(FadeOut(sample_dots), FadeOut(height_line), FadeOut(area_eq_height))

        # Average over next interval
        v_lines2 = v_lines.copy()
        v_lines2.move_to(axes.c2p(1, 0), DL)
        unit_int2 = Tex(R"\int^2_1 e^{\minus {s}t} dt", t2c=t2c, font_size=60)
        unit_int2.move_to(v_lines2, UP)

        area2_graph = graph.copy().clear_updaters().set_stroke(width=0)
        pile2 = always_redraw(lambda: axes.get_area_under_graph(area2_graph, (1, 2)))
        avg_value2 = get_norm(pile2.get_area_vector()) / (axes.x_axis.get_unit_size()**2)
        avg_label2 = self.get_avg_label(
            pile2.copy().set_height(axes.y_axis.get_unit_size() * avg_value2, stretch=True, about_edge=DOWN),
            1, 2
        )

        self.add(v_lines[0])
        self.play(
            TransformFromCopy(v_lines, v_lines2, path_arc=-20 * DEG),
            TransformMatchingTex(
                unit_int.copy(),
                unit_int2,
                path_arc=-20 * DEG,
                run_time=1,
                key_map={"0": "1", "1": 2},
            ),
            FadeIn(pile2, suspend_mobject_updating=True),
            unit_group.animate.match_x(pile2),
        )
        self.play(
            area2_graph.animate.stretch(0, 1).match_y(axes.y_axis.n2p(avg_value2)).set_anim_args(
                rate_func=slosh_rate_func,
                run_time=3,
            ),
            FadeIn(avg_label2)
        )
        pile2.clear_updaters()
        self.wait()

        # Show all integrals
        new_groups = VGroup()
        piles = VGroup(unit_int_area, pile2)
        avg_labels = VGroup(avg_label1, avg_label2)

        for n in range(2, 6):
            new_v_lines = v_lines.copy()
            new_v_lines.move_to(axes.c2p(n, 0), DL)
            new_unit_int = Tex(Rf"\int^{n + 1}_{n} " + R"e^{\minus {s}t} dt", t2c=t2c, font_size=60)
            if n == 5:
                new_unit_int = Tex(R"\cdots", font_size=90)
            new_unit_int.match_y(unit_int)
            new_unit_int.match_x(new_v_lines)
            s = get_s()
            avg_y = np.mean([np.exp(-s * t) for t in np.arange(n, n + 1, 1e-3)])
            new_pile = pile2.copy().clear_updaters()
            new_pile.set_height(avg_y * axes.y_axis.get_unit_size(), stretch=True)
            new_pile.move_to(axes.c2p(n, 0), DL)
            new_avg_label = self.get_avg_label(new_pile, n, n + 1)

            new_group = VGroup(new_v_lines, new_unit_int, new_pile, new_avg_label)
            new_groups.add(new_group)
            piles.add(new_pile)
            avg_labels.add(new_avg_label)

        big_brace = Brace(VGroup(unit_int, new_groups[-1][1]), UP, font_size=90, buff=LARGE_BUFF)
        integral.set_height(3)
        integral.next_to(big_brace, UP, buff=LARGE_BUFF)

        self.play(
            self.frame.animate.reorient(0, 0, 0, (5.34, 2.63, 0.0), 13.89),
            LaggedStartMap(FadeIn, new_groups, lag_ratio=0.75),
            FadeOut(unit_group, time_span=(0, 2)),
            run_time=4
        )
        self.play(
            GrowFromCenter(big_brace),
            FadeIn(integral),
        )
        self.wait()

        # Show adding all the heights
        height_vects = VGroup(
            Arrow(pile.get_bottom(), pile.get_top(), buff=0, thickness=6, fill_color=RED)
            for pile in piles
        )

        equals = Tex(R"=", font_size=120)
        equals.move_to(integral).shift(RIGHT)

        stacked_vects = height_vects.copy()
        stacked_vects.arrange(UP, buff=SMALL_BUFF)
        stacked_vects.scale(0.9)
        stacked_vects.next_to(equals, RIGHT, LARGE_BUFF)

        self.play(
            LaggedStartMap(FadeOut, avg_labels, lag_ratio=0.2),
            LaggedStartMap(FadeIn, height_vects, lag_ratio=0.2),
            run_time=1,
        )
        self.wait()
        self.play(
            TransformFromCopy(height_vects, stacked_vects, lag_ratio=0.05, run_time=2),
            integral.animate.next_to(equals, LEFT, LARGE_BUFF),
            Write(equals),
        )
        self.wait()

    def get_avg_label(self, pile, start=0, end=1, font_size=30):
        word = Text(f"Average over [{start}, {end}]", font_size=font_size)
        word.move_to(pile)
        word.set_max_height(pile.get_height() / 4)
        buff = min(SMALL_BUFF, 0.05 * pile.get_height())
        arrows = VGroup(
            Arrow(word.get_top() + buff * UP, pile.get_top(), buff=0),
            Arrow(word.get_bottom() + buff * DOWN, pile.get_bottom(), buff=0),
        )
        return VGroup(word, arrows)


class IntegrateComplexExponential(SPlane):
    staggered_path_colors = [MAROON_B, MAROON_C]
    t_max = 100  # For dynamic path and vector sum.  Change to 100 for final render
    initial_s = 0.2 + 1j
    s_label_config = dict(
        hide_zero_components_on_complex=False,
        include_sign=True,
        num_decimal_places=1,
    )

    def setup(self):
        super().setup()
        # Trackers
        self.s_tracker = ComplexValueTracker(self.initial_s)
        self.t_tracker = ValueTracker(0)
        get_s = self.s_tracker.get_value
        get_t = self.t_tracker.get_value

        def exp_func(t):
            return np.exp(-get_s() * t)

        self.exp_func = exp_func

    def construct(self):
        # Trackers
        s_tracker = self.s_tracker
        t_tracker = self.t_tracker
        exp_func = self.exp_func
        t2c = self.tex_to_color_map

        get_s = self.s_tracker.get_value
        get_t = self.t_tracker.get_value

        # Add s plane
        s_plane, s_dot, s_label, s_plane_label = s_group = self.get_s_group()
        self.add(*s_group)

        # Add exp plane
        exp_plane = self.get_exp_plane()
        exp_plane.set_width(6)
        exp_plane.next_to(s_plane, RIGHT, buff=1.0)
        exp_plane.add_coordinate_labels(font_size=16)
        exp_plane.axes.set_stroke(width=1)
        exp_plane_label = Tex(R"e^{\minus st}", font_size=72, t2c=t2c)
        exp_plane_label.next_to(exp_plane, UP)

        output_dot, output_label = self.get_output_dot_and_label(exp_plane, lambda: -get_s(), get_t, label_direction=DR, s_tex=R"\minus s")
        output_label.set_z_index(1)
        output_label_added_shift = Point(ORIGIN)
        output_label.add_updater(lambda m: m.shift(output_label_added_shift.get_center()))

        self.add(exp_plane, exp_plane_label, output_dot, output_label)

        # Show inital path
        path = always_redraw(lambda: self.get_output_path(exp_plane, exp_func, 0, 20))

        self.play(
            ShowCreation(path, suspend_mobject_updating=True),
            t_tracker.animate.set_value(path.t_range[1]),
            rate_func=linear,
            run_time=10,
        )
        self.play(
            s_tracker.animate.set_value(0.2 - 1j),
            rate_func=there_and_back,
            run_time=6,
        )
        self.play(
            s_tracker.animate.set_value(-0.2 + 1j),
            rate_func=there_and_back,
            run_time=6,
        )

        path.suspend_updating()

        # Fadeable transition to the start
        path_ghost = path.copy().set_stroke(opacity=0.4)
        self.wait()
        t_tracker.set_value(0)
        self.remove(path)
        self.add(path_ghost)
        self.wait()

        # Show [0, 1]
        subpath_0 = self.get_output_path(exp_plane, exp_func, 0, 1, stroke_color=self.staggered_path_colors[0])
        avg_vect0 = self.get_mean_vector(exp_plane, exp_func, 0, 1)
        many_points = self.get_sample_dots(subpath_0)
        avg_dot0 = TrueDot(avg_vect0.get_end(), color=RED)

        int_tex0 = Tex(R"\int^1_0 e^{\minus st} dt", t2c=t2c, font_size=48)
        int_tex0.next_to(exp_plane.n2p(1), UP, MED_SMALL_BUFF)
        int_tex0.set_backstroke(BLACK, 5)

        self.play(FadeIn(int_tex0, 0.25 * UP))
        self.play(
            t_tracker.animate.set_value(1),
            ShowCreation(subpath_0),
            rate_func=linear,
            run_time=2
        )
        self.wait()
        self.play(ShowCreation(many_points))
        self.wait()
        self.play(ReplacementTransform(many_points, avg_dot0))
        self.wait()
        self.play(
            GrowArrow(avg_vect0),
            int_tex0.animate.set_height(0.75).next_to(avg_vect0.get_center(), UR, buff=SMALL_BUFF)
        )
        self.play(FadeOut(avg_dot0))
        self.wait()

        # Bring in integral plane
        lil_exp_plane = self.get_exp_plane(x_range=(-1, 1))
        lil_exp_plane.set_width(3)
        lil_exp_plane.move_to(exp_plane, UR).shift(0.5 * UP + 0.75 * LEFT)
        lil_exp_plane.save_state()
        lil_exp_plane.axes.set_stroke(width=1)

        int_plane = self.get_exp_plane()
        int_plane.set_width(3)
        int_plane.next_to(lil_exp_plane, DOWN, MED_LARGE_BUFF)
        int_plane.axes.set_stroke(width=1)
        int_plane.add_coordinate_labels(font_size=12)

        to_upper = lil_exp_plane.get_center() - exp_plane.get_center()
        to_lower = int_plane.get_center() - exp_plane.get_center()

        int_plane_label = Tex(R"\int^\infty_0 e^{\minus st} dt", t2c=t2c, font_size=48)
        int_plane_label.next_to(int_plane, LEFT, aligned_edge=UP)

        lower_avg_vect0 = self.get_mean_vector(int_plane, exp_func, 0, 1, thickness=1, fill_color=self.staggered_path_colors[0])

        self.play(
            exp_plane.animate.move_to(lil_exp_plane.saved_state).set_opacity(0),
            FadeIn(lil_exp_plane, to_upper),
            VGroup(path_ghost, subpath_0, avg_vect0).animate.shift(to_upper),
            int_tex0.animate.scale(0.5, about_point=exp_plane.n2p(0)).shift(to_lower),
            TransformFromCopy(avg_vect0, lower_avg_vect0),
            FadeIn(int_plane),
            exp_plane_label.animate.next_to(lil_exp_plane, LEFT, aligned_edge=UP),
        )
        self.wait()

        # Add the next few arrows
        avg_vects = VGroup(avg_vect0)
        lower_avg_vects = VGroup(lower_avg_vect0)
        subpaths = VGroup(subpath_0)
        int_texs = VGroup(int_tex0)
        shifts = [
            0.8 * LEFT + 0.1 * DOWN,
            LEFT,
            0.2 * LEFT + 0.4 * UP,
            0.4 * UP,
        ]

        def get_subpath(n):
            result = self.get_output_path(exp_plane, exp_func, n, n + 1, stroke_width=3)
            result.set_stroke(self.staggered_path_colors[n % 2])
            return result

        for n in range(1, 5):
            subpath_n = get_subpath(n)
            sample_points = self.get_sample_dots(subpath_n)
            avg_vect, lower_avg_vect = [
                self.get_mean_vector(plane, exp_func, n, n + 1, thickness=thickness)
                for plane, thickness in [(exp_plane, 3), (int_plane, 1)]
            ]
            lower_avg_vect.set_fill(self.staggered_path_colors[n % 2], border_width=0.5)
            lower_avg_vect.set_stroke(width=0)
            lower_avg_vect.put_start_on(lower_avg_vects[-1].get_end())
            avg_dot = sample_points.copy().set_points([avg_vect.get_end()])

            int_tex = Tex(Rf"\int_{n}^{n + 1} e^{{\minus s t}} dt", t2c=t2c)
            int_tex.match_height(int_tex0)
            int_tex.scale(1.0 - 0.15 * n)
            int_tex.next_to(lower_avg_vect.get_center(), rotate_vector(lower_avg_vect.get_vector(), 90 * DEG), buff=SMALL_BUFF)

            self.play(
                t_tracker.animate.set_value(n + 1),
                ShowCreation(subpath_n),
                output_label_added_shift.animate.move_to(shifts[n - 1]),
                ShowCreation(sample_points),
                run_time=2,
                rate_func=linear,
            )
            self.play(
                avg_vects[-1].animate.set_fill(opacity=0.5).set_stroke(width=0),
                GrowArrow(avg_vect),
                Transform(sample_points, avg_dot),
            )
            self.play(
                FadeOut(sample_points),
                TransformFromCopy(avg_vect, lower_avg_vect),
                FadeIn(int_tex),
            )

            avg_vects.add(avg_vect)
            lower_avg_vects.add(lower_avg_vect)
            subpaths.add(subpath_n)
            int_texs.add(int_tex)

        # Show numerous more
        for n in range(5, 20):
            subpath = get_subpath(n)
            avg_vect = self.get_mean_vector(exp_plane, exp_func, n, n + 1, thickness=3)
            lower_avg_vect = self.get_mean_vector(int_plane, exp_func, n, n + 1, thickness=1)
            lower_avg_vect.set_fill(self.staggered_path_colors[n % 2], border_width=0.5)
            lower_avg_vect.set_stroke(width=0)
            lower_avg_vect.put_start_on(lower_avg_vects[-1].get_end())

            anims = [
                ShowCreation(subpath, rate_func=linear),
                t_tracker.animate.set_value(n + 1).set_anim_args(rate_func=linear),
                FadeIn(avg_vect),
            ]
            if n == 5:
                anims.append(avg_vects.animate.set_opacity(0.25).set_stroke(width=0))
            else:
                anims.append(avg_vects[-1].animate.set_opacity(0.25).set_stroke(width=0))
                anims.append(TransformFromCopy(avg_vects[-1], lower_avg_vects[-1]))
            self.play(*anims)

            avg_vects.add(avg_vect)
            lower_avg_vects.add(lower_avg_vect)
            subpaths.add(subpath)

            self.add(avg_vects, subpaths, *lower_avg_vects[:-1])

        # Show the full integral
        int_dot = Group(TrueDot(radius=0.025), GlowDot())
        int_dot.set_color(MAROON_B)
        int_dot.move_to(int_plane.n2p(1 / get_s()))
        int_rect = SurroundingRectangle(int_plane_label)
        int_rect.set_stroke(YELLOW, 2)

        self.play(
            LaggedStart(
                (FadeTransform(int_tex, int_plane_label)
                for int_tex in int_texs),
                lag_ratio=0.025,
                group_type=Group,
            ),
            *map(FadeOut, [output_label, output_dot, avg_vects]),
        )
        self.play(ShowCreation(int_rect))
        self.wait()
        self.play(
            int_rect.animate.surround(int_dot[0], buff=0),
            FadeIn(int_dot),
        )
        self.play(FadeOut(int_rect))
        self.wait()

        # Make the diagram dynamic
        staggered_path = self.get_dynamic_output_path(exp_plane, exp_func)
        int_pieces = self.get_dynamic_vector_sum(int_plane, exp_func)
        int_dot.add_updater(lambda m: m.move_to(int_plane.n2p(1.0 / get_s())))

        dynamic_pieces = Group(staggered_path, int_pieces)

        self.play(FlashAround(Group(s_label, s_dot), time_width=1.5, run_time=2))
        self.remove(path_ghost, subpaths, lower_avg_vects)
        self.add(staggered_path, int_pieces, int_dot)

        # Move around the path
        s_rect = int_rect.copy()

        self.play(s_tracker.animate.set_value(0.2 - 1j), run_time=6)
        self.wait()
        self.play(s_tracker.animate.set_value(1), run_time=4)
        s_rect.surround(Group(s_label))
        self.play(ShowCreation(s_rect))
        self.play(s_rect.animate.surround(int_dot[0], buff=0))
        self.play(FadeOut(s_rect))
        self.wait()

        # Approach zero, then go vertical
        frame = self.frame
        self.play(
            s_tracker.animate.set_value(0.1).set_anim_args(rate_func=bezier([0, 1, 1, 1])),
            frame.animate.reorient(0, 0, 0, (2.75, 0, 0.0), 11).set_anim_args(time_span=(2, 8)),
            run_time=8,
        )
        self.wait()
        self.play(
            s_tracker.animate.set_value(0.1 + 2j),
            frame.animate.to_default_state(),
            run_time=10
        )
        self.play(s_tracker.animate.set_value(0.1 - 2j), run_time=5)
        self.wait()
        self.play(s_tracker.animate.set_value(0.2 + 1j), run_time=5)
        self.wait()

        # Emphasize output
        int_vect = self.get_vector(int_plane, 1 / get_s(), thickness=2, fill_color=WHITE)
        int_vect.set_stroke(width=0).set_fill(border_width=0.5)
        rect = SurroundingRectangle(int_plane_label)
        rect.set_stroke(YELLOW, 2)
        int_vect_outline = int_vect.copy().set_fill(opacity=0).set_stroke(WHITE, 1)

        self.play(ShowCreation(rect))
        self.wait()
        self.play(
            rect.animate.surround(int_dot[0], buff=0),
            int_pieces.animate.set_fill(opacity=0.5),
        )
        self.wait()
        self.play(
            GrowArrow(int_vect),
            FadeOut(rect),
            FadeOut(int_dot),
        )
        self.wait()

        # Plot it
        frame = self.frame
        s_plane_group = Group(s_plane, s_dot, s_label, s_plane_label)
        s_label.set_flat_stroke(True)
        s_plane.set_flat_stroke(False)
        s_plane_group.save_state()
        s_plane_group.center()
        self.remove(s_plane_group)

        for mob in self.get_mobjects():
            mob.fix_in_frame()
            mob.set_z_index(2)

        right_rect = FullScreenRectangle()
        right_rect.set_fill(BLACK, 1).set_stroke(width=0)
        right_rect.set_width(lil_exp_plane.get_width() + 2, about_edge=RIGHT, stretch=True)
        v_line = DashedLine(right_rect.get_corner(DL), right_rect.get_corner(UL))
        v_line.set_stroke(WHITE, 1)
        right_pannel = VGroup(right_rect, v_line)
        right_pannel.fix_in_frame()
        right_pannel.set_z_index(1)
        self.add(right_pannel)

        def get_output_magnitude(s):
            return abs(1.0 / s)

        z_line = Line(ORIGIN, OUT)
        z_line.set_stroke(WHITE, 2)

        unit_size = s_plane.x_axis.get_unit_size()
        z_line.add_updater(lambda m: m.put_start_and_end_on(
            s_plane.n2p(get_s()),
            s_plane.n2p(get_s()) + unit_size * get_output_magnitude(get_s()) * OUT,
        ))

        out_dot = Group(TrueDot(), GlowDot())
        out_dot.set_color(TEAL)
        out_dot.f_always.move_to(z_line.get_end)

        traced_graph = TracedPath(z_line.get_end, stroke_color=TEAL)
        traced_graph.update()

        int_vect.add_updater(
            lambda m: m.put_start_and_end_on(
                int_plane.n2p(0),
                int_plane.n2p(1.0 / get_s()),
            )
        )

        pre_z_line = Line(int_vect.get_start(), int_vect.get_end())
        pre_z_line.set_stroke(WHITE, 5)
        pre_z_line.set_z_index(2)

        new_exp_plane_label = exp_plane_label.copy().scale(0.5).next_to(lil_exp_plane, RIGHT, aligned_edge=UP)
        new_int_plane_label = int_plane_label.copy().scale(0.5).next_to(int_plane, RIGHT, SMALL_BUFF, aligned_edge=UP)

        s_plane_group.restore()
        self.play(
            frame.animate.reorient(-3, 68, 0, (1.43, 0.51, -0.28), 6.50),
            s_plane_group.animate.center(),
            ReplacementTransform(pre_z_line, z_line),
            FadeOut(exp_plane_label, time_span=(1, 2)),
            FadeIn(new_exp_plane_label, time_span=(1, 2)),
            FadeOut(int_plane_label, time_span=(1, 2)),
            FadeIn(new_int_plane_label, time_span=(1, 2)),
            VFadeIn(v_line, time_span=(2, 3)),
            run_time=3,
        )
        self.play(
            FadeIn(out_dot, time_span=(0, 1)),
            frame.animate.reorient(8, 64, 0, (1.43, 0.51, -0.28), 6.50),
            run_time=3
        )

        self.add(traced_graph)
        self.play(
            s_tracker.animate.set_value(0.2 - 2j),
            frame.animate.reorient(25, 87, 0, (3.18, 1.73, 2.63), 11.08).set_anim_args(time_span=(1, 7.5)),
            run_time=10
        )
        self.play(s_tracker.animate.set_value(0.4 - 2j))
        self.play(
            s_tracker.animate.set_value(0.4 + 2j),
            frame.animate.reorient(44, 81, 0, (3.16, 1.71, 2.63), 11.08),
            run_time=5,
        )
        self.play(s_tracker.animate.set_value(0.6 + 2j))
        self.play(
            s_tracker.animate.set_value(0.6 - 2j),
            frame.animate.reorient(66, 89, 0, (3.31, 2.07, 2.96), 9.69),
            run_time=5,
        )
        self.play(s_tracker.animate.set_value(0.8 - 2j))
        self.play(
            s_tracker.animate.set_value(0.8 + 2j),
            frame.animate.reorient(22, 84, 0, (2.56, 1.53, 1.72), 10.14),
            run_time=5
        )
        self.play(s_tracker.animate.set_value(1.0 + 2j))
        self.play(
            s_tracker.animate.set_value(1.0 - 2j),
            frame.animate.reorient(22, 84, 0, (2.56, 1.53, 1.72), 10.14),
            run_time=5
        )
        self.wait()

        traced_graph.suspend_updating()

        # Add the graph
        graph = get_complex_graph(s_plane, lambda s: 1.0 / s, min_real=0)[0]
        graph.set_shading(0.1, 0.1, 0)
        graph.save_state()
        graph.set_color(GREY_B)
        mesh = SurfaceMesh(graph, resolution=(11, 21))
        mesh.set_stroke(WHITE, 0.5, 0.5)
        for line in mesh:
            if line.get_z(IN) < 0:
                mesh.remove(line)

        self.play(
            ShowCreation(graph),
            ShowCreation(mesh, lag_ratio=1e-2, time_span=(1, 4)),
            VFadeOut(traced_graph),
            frame.animate.reorient(34, 74, 0, (2.02, 1.44, 1.88), 9.87),
            out_dot.animate.set_color(WHITE),
            run_time=4
        )

        self.play(
            s_tracker.animate.set_value(0.1 - 0.1j),
            frame.animate.reorient(38, 82, 0, (2.84, 2.62, 4.72), 14.21).set_anim_args(time_span=(0, 6)),
            run_time=10
        )
        self.wait()
        self.play(s_tracker.animate.set_value(0.5 - 0.1j), run_time=5)
        self.play(frame.animate.reorient(26, 72, 0, (2.33, 1.59, 1.91), 10.76), run_time=3)
        self.wait()
        self.play(
            s_tracker.animate.set_value(0.5 - 1j),
            run_time=3
        )
        self.wait()

        # Explain color
        hue_circle = Circle(radius=1.5 * int_plane.x_axis.get_unit_size())
        hue_circle.set_color_by_proportion(lambda h: Color(hsl=(h, 0.5, 0.5)))
        hue_circle.set_stroke(width=5)
        hue_circle.move_to(int_plane)
        hue_circle.fix_in_frame().set_z_index(2)

        brace = LineBrace(int_vect, direction=UP, buff=0)
        brace.set_fill(border_width=1)
        brace.set_z_index(2).fix_in_frame()

        self.play(GrowFromCenter(brace))
        self.wait()
        self.play(FadeOut(brace))

        self.play(ShowCreation(hue_circle, run_time=2))
        self.wait()
        self.play(
            Restore(graph, time_span=(0, 1)),
            frame.animate.reorient(29, 72, 0, (1.79, 1.9, 1.65), 11.21),
            run_time=4,
        )
        self.play(s_tracker.animate.set_value(0.5 + 1j), run_time=5)
        self.wait()

        # Show negative real part
        left_plane = Rectangle()
        left_plane.set_stroke(width=0).set_fill(RED, 0.5)
        left_plane.replace(s_plane, stretch=True)
        left_plane.stretch(0.5, 0, about_edge=LEFT)
        left_plane.save_state()
        left_plane.stretch(0, 0, about_edge=RIGHT)

        staggered_path.set_clip_plane(RIGHT, -v_line.get_x())
        int_pieces.set_clip_plane(RIGHT, -v_line.get_x())

        self.play(
            frame.animate.reorient(-38, 70, 5, (3.17, 1.96, -0.36), 9.16),
            FadeOut(hue_circle),
            Restore(left_plane, time_span=(3, 5)),
            run_time=5
        )
        self.wait()

        # Move to negative real
        self.play(
            s_tracker.animate.set_value(-0.5),
            VFadeOut(z_line, time_span=(0, 1.5)),
            FadeOut(out_dot, time_span=(0, 1.5)),
            VFadeOut(int_vect, time_span=(0, 1.0)),
            run_time=3
        )
        self.play(s_tracker.animate.set_value(-0.1 - 1j).set_anim_args(run_time=3))
        self.play(
            *(
                plane.animate.scale(0.1, about_point=plane.n2p(0))
                for plane in [exp_plane, lil_exp_plane, int_plane]
            ),
        )
        self.play(
            ShowCreation(staggered_path, suspend_mobject_updating=True),
            ShowIncreasingSubsets(int_pieces, suspend_mobject_updating=True),
            run_time=4
        )
        self.play(
            UpdateFromAlphaFunc(
                s_tracker,
                lambda m, a: m.set_value(complex(
                    -0.1 - math.sin(PI * a),
                    interpolate(-1, 1, a)
                ))
            ),
            run_time=8
        )
        self.play(
            *(
                plane.animate.scale(10, about_point=plane.n2p(0)).set_anim_args(time_span=(2, 4))
                for plane in [exp_plane, lil_exp_plane, int_plane]
            ),
            s_tracker.animate.set_value(1j),
            run_time=4
        )
        self.play(FadeIn(z_line), FadeIn(out_dot))
        self.wait()

        # Show imaginary input
        self.remove(staggered_path, int_pieces)
        int_pieces.set_fill(opacity=1)
        t_tracker.set_value(0)
        output_dot.fix_in_frame().set_z_index(2)
        output_label.add_updater(lambda m: m.fix_in_frame().set_z_index(2))
        self.play(
            FadeIn(output_dot),
            VFadeIn(output_label),
            left_plane.animate.set_fill(opacity=0.2),
        )
        max_n = 25
        self.play(
            ShowCreation(staggered_path[:max_n].set_z_index(2)),
            ShowIncreasingSubsets(int_pieces[:max_n].set_z_index(2), int_func=np.floor),
            t_tracker.animate.set_value(max_n).set_anim_args(time_span=(max_n / (max_n + 1), max_n)),
            rate_func=linear,
            run_time=max_n
        )
        self.play(
            VFadeOut(output_label),
            FadeOut(output_dot),
            t_tracker.animate.set_value(max_n + 1).set_anim_args(rate_func=linear),
            ShowIncreasingSubsets(int_pieces[max_n:].set_z_index(2)),
            FadeIn(staggered_path[max_n:].set_z_index(2))
        )

        # Show the limiting value
        self.add(staggered_path, int_pieces)
        self.add(int_vect)
        self.play(
            s_tracker.animate.set_value(0.1 + 1j),
            int_pieces.animate.set_fill(opacity=0.75),
            VFadeIn(int_vect),
            FadeOut(left_plane),
            run_time=2
        )
        rect = SurroundingRectangle(int_vect).set_z_index(2)
        self.play(ShowCreation(rect))
        self.play(FadeOut(rect))
        self.wait()
        self.play(
            s_tracker.animate.set_value(1j),
            run_time=5
        )
        self.wait()

        # Move along imaginary line
        boundary = TracedPath(z_line.get_end, stroke_color=RED, stroke_width=3)

        self.add(boundary, out_dot)
        self.play(s_tracker.animate.set_value(2j), run_time=3)
        self.play(s_tracker.animate.set_value(0.1j), run_time=3)
        self.play(s_tracker.animate.set_value(1j), run_time=3)
        boundary.clear_updaters()
        self.play(FadeOut(boundary))

        # Show this as 1 / s
        int_equation = Tex(R"\int^\infty_0 e^{\minus st} dt = \frac{1}{s}", t2c=t2c)
        int_equation.to_edge(UP).shift(LEFT)
        int_equation.fix_in_frame()
        lhs_mover = new_int_plane_label.copy()

        self.play(
            frame.animate.reorient(0, 85, 1, (5.32, 3.46, 3.69), 13.49),
            s_tracker.animate.set_value(0.2 + 1j),
            run_time=3,
        )
        self.play(
            lhs_mover.animate.replace(int_equation[:-4]),
            Write(int_equation[-4:], time_span=(0.75, 1.75)),
        )
        self.remove(lhs_mover)
        self.add(int_equation)
        self.wait()

        # Look closely at s = i again
        int_plane_rect = SurroundingRectangle(int_plane)
        int_plane_rect.fix_in_frame()
        int_plane_rect.set_stroke(YELLOW, 5).insert_n_curves(100)
        int_plane_rect.set_z_index(3)
        int_plane_rect.scale(0.5, about_edge=DOWN).scale(1.1)

        i_eq = Tex(R"\frac{1}{i} = -i")
        i_eq.next_to(int_equation[-3:], DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
        i_eq.fix_in_frame()

        left_arrow = Vector(LEFT).set_color(YELLOW)
        left_arrow.next_to(s_plane.n2p(0.4 + 1j), DOWN)

        self.play(s_tracker.animate.set_value(0.2 - 1j), run_time=6)
        self.play(
            s_tracker.animate.set_value(0.5 + 1j),
            frame.animate.reorient(-49, 79, 0, (6.24, 2.45, -0.13), 10.06),
            run_time=4
        )
        self.play(
            GrowArrow(left_arrow, time_span=(0, 1)),
            VShowPassingFlash(int_plane_rect, time_width=1.5, time_span=(4, 7)),
            Write(i_eq, time_span=(10, 12)),
            s_tracker.animate.set_value(1j),
            run_time=15
        )
        self.wait()
        self.play(
            frame.animate.reorient(0, 85, 1, (5.32, 3.46, 3.69), 13.49),
            FadeOut(i_eq),
            FadeOut(left_arrow),
            run_time=3
        )

        # Ambient s movement
        for z in [0.2 - 1j, 1 - 1j, 1 + 1j, 0.2 + 1j]:
            self.play(s_tracker.animate.set_value(z), run_time=6)

        # Talk again about the left plane
        full_plane = Rectangle().set_stroke(width=0).set_fill(RED, 0.5)
        full_plane.replace(s_plane, stretch=True)
        full_plane.set_fill(GREEN, 0.5)

        left_plane = full_plane.copy()
        left_plane.set_fill(RED, 0.5)
        left_plane.stretch(0.5, 0, about_edge=LEFT)
        left_plane.save_state()
        left_plane.stretch(0, 0, about_edge=RIGHT)

        equation_rect = SurroundingRectangle(int_equation)
        equation_rect.set_stroke(YELLOW, 2)
        equation_rect.fix_in_frame()

        self.play(
            Restore(left_plane),
            frame.animate.reorient(-36, 79, 4, (7.71, 2.4, 1.86), 15.36),
            run_time=3
        )
        self.wait()
        self.play(ShowCreation(equation_rect))
        self.wait()
        self.play(equation_rect.animate.surround(int_equation[R"\frac{1}{s}"], buff=SMALL_BUFF))
        self.wait()
        self.play(
            ReplacementTransform(left_plane, full_plane)
        )
        self.wait()

        # Describe continuation
        lhs_rect = SurroundingRectangle(int_equation[:-4])
        rhs_rect = SurroundingRectangle(int_equation[-3:])
        VGroup(lhs_rect, rhs_rect).fix_in_frame().set_stroke(YELLOW, 2)

        def pole_func(s):
            if s != 0:
                return 1.0 / s
            return 100

        extended_graph = get_complex_graph(s_plane, pole_func)
        extended_mesh = SurfaceMesh(extended_graph, resolution=(21, 21))
        extended_mesh.remove(*(line for line in extended_mesh if line.get_z(IN) < 0))
        extended_mesh.set_stroke(WHITE, 0.5, 0.15)
        extended_mesh.shift(1e-2 * OUT)

        self.add(extended_graph, graph, extended_mesh, mesh)
        self.play(
            FadeOut(graph, 1e-2 * IN),
            FadeIn(extended_graph),
            FadeOut(mesh),
            FadeIn(extended_mesh),
            frame.animate.reorient(0, 85, 0, (5.32, 3.46, 3.69), 13.49).set_anim_args(run_time=5),
            FadeOut(full_plane),
        )
        self.play(
            frame.animate.reorient(-17, 62, 9, (6.29, -0.4, 3.75), 13.87),
            FadeOut(equation_rect),
            run_time=5
        )
        curr_s = complex(get_s())
        self.play(
            UpdateFromAlphaFunc(s_tracker, lambda m, a: m.set_value(curr_s * np.exp(a * TAU * 1j))),
            run_time=30,
        )
        self.wait()

        # Discuss pole (Mostly with overlays)
        self.play(
            frame.animate.reorient(-36, 83, 1, (5.84, -1.06, 3.19), 10.63),
            s_tracker.animate.set_value(0.1 + 0.1j),
            run_time=20
        )
        self.play(
            frame.animate.reorient(0, 0, 0, (1.95, 0.95, -0.0), 15.34),
            s_tracker.animate.set_value(1e-3).set_anim_args(time_span=(0, 5)),
            extended_mesh.animate.set_stroke(opacity=0),
            VFadeOut(int_vect, time_span=(0, 7)),
            run_time=10,
        )
        self.play(frame.animate.reorient(0, 83, 0, (5.18, 4.04, 3.15), 14.50), run_time=10)

    def get_s_group(self):
        s_plane = self.get_s_plane()
        s_plane.set_width(6)
        s_plane.to_corner(DL)
        s_plane.axes.set_stroke(width=1)

        s_dot, s_label = self.get_s_dot_and_label(s_plane, get_s=self.s_tracker.get_value)
        s_plane_label = Tex(R"s", t2c=self.tex_to_color_map, font_size=72)
        s_plane_label.next_to(s_plane, UP)

        s_group = Group(s_plane, s_dot, s_label, s_plane_label)
        return s_group

    def get_output_path(self, plane, func, t_min=0, t_max=20, stroke_color=TEAL, stroke_width=2, step_size=1e-1):
        return ParametricCurve(
            lambda t: plane.n2p(func(t)),
            t_range=[t_min, t_max, step_size],
            stroke_color=stroke_color,
            stroke_width=stroke_width,
        )

    def get_vector(self, plane, value, backstroke_width=2, **kwargs):
        vect = Arrow(plane.n2p(0), plane.n2p(value), buff=0, **kwargs)
        vect.set_backstroke(BLACK, width=backstroke_width)
        return vect

    def get_mean_vector(self, plane, func, t_min, t_max, thickness=3, fill_color=RED, n_samples=1000, **kwargs):
        x_range = np.linspace(t_min, t_max, n_samples)
        x_mean = np.mean(func(x_range))
        return self.get_vector(plane, x_mean, thickness=thickness, fill_color=fill_color, **kwargs)
        return vect

    def get_sample_dots(self, subpath, n_samples=20, radius=0.05, glow_factor=0.5, color=RED):
        return DotCloud(
            [subpath.pfp(a) for a in np.linspace(0, 1, n_samples)],
            radius=radius,
            glow_factor=glow_factor,
            color=color
        )

    def get_dynamic_output_path(self, plane, func, stroke_width=3, step_size=1e-2):
        t_range = list(range(0, self.t_max))
        t_samples = [np.arange(t, t + 1 + step_size, step_size) for t in t_range]
        path = VGroup(VMobject() for t in t_range)

        def update_path(path):
            for piece, samples in zip(path, t_samples):
                piece.set_points_as_corners(plane.n2p(func(samples)))
            return path

        path.add_updater(update_path)
        for piece, color in zip(path, it.cycle(self.staggered_path_colors)):
            piece.set_stroke(color, stroke_width)

        return path

    def get_dynamic_vector_sum(self, plane, func, thickness=1, backstroke_width=0, n_samples=100, border_width=0.5):
        t_range = list(range(0, self.t_max))
        vects = VGroup(
            self.get_vector(
                plane, 1,
                thickness=thickness,
                backstroke_width=backstroke_width,
                fill_color=color,
            )
            for t, color in zip(t_range, it.cycle(self.staggered_path_colors))
        )
        vects.set_fill(border_width=border_width)
        t_samples = [np.linspace(t, t + 1, n_samples) for t in t_range]

        def update_vects(vects):
            avg_values = [0] + [func(samples).mean() for samples in t_samples]
            end_values = np.cumsum(avg_values)
            end_points = plane.n2p(end_values)
            for vect, p0, p1 in zip(vects, end_points, end_points[1:]):
                vect.put_start_and_end_on(p0, p1)
            return vect

        vects.add_updater(update_vects)

        return vects


class BreakDownLaplaceTransform(IntegrateComplexExponential):
    func_tex = R"e^{1.5 {t}}"
    initial_s = 2 + 1j
    s_plane_x_range = (-3, 3)
    s_label_font_size = 24
    t_max = 100  # For dynamic path and vector sum.  Change to 100 for final render
    # pole_value = 1.5
    pole_value = -0.2 + 1.5j

    def construct(self):
        self.add_core_pieces()

        frame = self.frame
        s_tracker = self.s_tracker
        exp_plane, int_plane = self.output_planes

        # Talk through the parts
        frame.reorient(0, 52, 0, (-0.69, 0.41, 0.56), 10.93)
        int_rect = SurroundingRectangle(int_plane.label)
        int_rect.set_stroke(YELLOW, 2)
        int_rect.fix_in_frame()

        self.play(
            frame.animate.reorient(20, 68, 0, (-0.45, 0.24, 0.17), 6.28),
            s_tracker.animate.set_value(1.6 - 1j),
            ShowCreation(int_rect, time_span=(4, 5)),
            run_time=8
        )
        self.play(int_rect.animate.surround(exp_plane.label))
        self.draw_upper_plot(draw_time=8)
        self.play(
            s_tracker.animate.set_value(1.6),
            frame.animate.reorient(24, 65, 0, (-0.35, 1.85, 1.61), 11.20),
            run_time=5
        )
        self.play(
            s_tracker.animate.set_value(1.501),
            run_time=5
        )

        # Alternate pan to pole
        self.clear()
        to_add = Group(self.s_group, self.graph_tracer, self.graph)
        to_add.shift(-self.s_group[0].n2p(0))
        self.add(to_add)

        frame.reorient(-35, 84, 0, (-1.14, 1.42, 4.73), 14.55)
        self.s_tracker.set_value(1 + 1j)
        self.play(
            frame.animate.reorient(0, 43, 0, (-0.42, 0.6, 2.1), 9.91),
            self.s_tracker.animate.set_value(1.5001),
            run_time=10
        )

    def add_core_pieces(self):
        # Planes and their contenst
        s_group = self.get_s_group()
        s_plane = s_group[0]
        s_plane.faded_lines.set_opacity(0)

        output_planes = self.get_output_planes()
        exp_plane, int_plane = output_planes

        right_rect = FullScreenRectangle()
        right_rect.set_fill(BLACK, 1)
        right_rect.set_width(RIGHT_SIDE[0] - output_planes.get_left()[0] + SMALL_BUFF, stretch=True, about_edge=RIGHT)

        # Dynamic pieces
        output_path = self.get_dynamic_output_path(exp_plane, self.inner_func)
        vect_sum = self.get_dynamic_vector_sum(int_plane, self.inner_func)
        integral_vect = self.get_integral_vect(int_plane)

        for mob in [output_path, vect_sum, integral_vect]:
            mob.set_clip_plane(RIGHT, -exp_plane.get_x(LEFT))

        graph_tracer = self.get_graph_tracer(s_plane)

        graph = get_complex_graph(s_plane, self.transformed_func)

        # Add everything
        right_group = VGroup(right_rect, output_planes, output_path, vect_sum)
        right_group.fix_in_frame()
        right_group.set_scale_stroke_with_zoom(True)

        self.add(
            s_group,
            graph_tracer,
            graph,
            right_rect,
            output_planes,
            output_path,
            vect_sum,
            integral_vect,
        )

        self.s_group = s_group
        self.output_planes = output_planes
        self.output_path = output_path
        self.vect_sum = vect_sum
        self.integral_vect = integral_vect
        self.graph_tracer = graph_tracer
        self.graph = graph

    def get_integral_vect(self, int_plane):
        vect = Vector(RIGHT, fill_color=WHITE, thickness=2)
        dot = GlowDot(color=PINK)
        origin = int_plane.n2p(0)

        vect.add_updater(lambda m: m.put_start_and_end_on(
            origin, int_plane.n2p(self.transformed_func(self.s_tracker.get_value()))
        ))
        dot.f_always.move_to(vect.get_end)

        group = Group(vect, dot)
        group.fix_in_frame()
        return group

    def get_graph_tracer(self, s_plane):
        v_line = Line(ORIGIN, OUT).set_stroke(WHITE, 3)
        graph_dot = Group(TrueDot(), GlowDot())
        graph_dot.set_color(TEAL)

        group = Group(v_line, graph_dot)
        s_plane_unit = s_plane.x_axis.get_unit_size()

        def update_tracer(group):
            s = self.s_tracker.get_value()
            s_point = s_plane.n2p(s)
            int_value = self.transformed_func(s)

            line, dot = group
            line.put_start_and_end_on(s_point, s_point + s_plane_unit * abs(int_value) * OUT)
            dot.move_to(line.get_end())

        group.add_updater(update_tracer)
        return group

    def get_output_planes(self, width=3, edge_buff=2.5):
        exp_plane = self.get_exp_plane(x_range=(-1, 1))
        exp_plane.set_width(width)
        exp_plane.axes.set_stroke(width=1)
        exp_plane.to_corner(UP, MED_LARGE_BUFF)
        exp_plane.to_edge(RIGHT, buff=edge_buff)

        int_plane = self.get_exp_plane()
        int_plane.set_width(width)
        int_plane.move_to(exp_plane).to_edge(DOWN)
        int_plane.axes.set_stroke(width=1)
        int_plane.add_coordinate_labels(font_size=12)

        kw = dict(t2c=self.tex_to_color_map)
        exp_plane.label = Tex(self.func_tex, R"e^{\minus {s} {t}}", **kw)
        exp_plane.label.next_to(exp_plane, RIGHT, aligned_edge=UP)

        int_plane.label = Tex(R"\int^\infty_0 " + self.func_tex + R"e^{\minus {s} {t}} d{t}", **kw)
        int_plane.label.scale(0.6)
        int_plane.label.next_to(int_plane, RIGHT, aligned_edge=UP)

        planes = VGroup(exp_plane, int_plane)
        for plane in planes:
            plane.add(plane.label)

        return planes

    def func(self, t):
        return np.exp(1.5 * t)

    def inner_func(self, t):
        return self.func(t) * np.exp(-self.s_tracker.get_value() * t)

    def transformed_func(self, s):
        return np.divide(1.0, (s - self.pole_value))

    def draw_upper_plot(self, draw_time=12, rate_multiple=2):
        exp_plane = self.output_planes[0]
        n_pieces = int(rate_multiple * draw_time)
        path_copy = VMobject()
        path_copy.fix_in_frame()
        path_copy.start_new_path(self.output_path[0].get_start())
        for part in self.output_path[:n_pieces]:
            path_copy.append_vectorized_mobject(part)

        tracing_vect = Vector(fill_color=YELLOW)
        tracing_vect.add_updater(lambda m: m.put_start_and_end_on(
            exp_plane.n2p(0),
            path_copy.get_end(),
        ))
        tracing_vect.fix_in_frame()

        vect_sum_copy = self.vect_sum.copy()[:n_pieces]
        vect_sum_copy.clear_updaters()
        self.remove(self.output_path, self.vect_sum, self.integral_vect)

        self.add(tracing_vect)
        self.play(
            ShowCreation(path_copy),
            ShowIncreasingSubsets(self.vect_sum[:n_pieces], int_func=np.ceil),
            run_time=draw_time,
            rate_func=linear
        )
        self.remove(path_copy, tracing_vect)
        self.add(self.output_path, self.vect_sum, self.integral_vect)
        self.wait()

    def show_simple_pole(self):
        # Clean the board
        frame = self.frame
        for line in self.graph[1]:
            if line.get_z(OUT) > 100 or line.get_z(IN) < -100:
                line.set_stroke(opacity=0)
        self.clear()
        self.add(self.s_group, self.graph, self.graph_tracer)
        self.graph_tracer[0].set_stroke(width=1)

        # Move
        frame.reorient(-30, 83, 0, (-4.35, 1.2, 2.52), 11.47)
        self.play(
            frame.animate.reorient(0, 0, 0, (-5.17, -0.19, 0.0), 9.88).set_field_of_view(20 * DEG),
            self.s_tracker.animate.set_value(self.pole_value + 1e-5),
            run_time=10
        )
        self.wait()




class LaplaceTransformOfCos(BreakDownLaplaceTransform):
    func_tex = R"\cos(t)"
    s_label_config = dict(
        hide_zero_components_on_complex=True,
        include_sign=True,
        num_decimal_places=2,
    )
    t_max = 200  # For dynamic path and vector sum.

    def construct(self):
        self.add_core_pieces()

        frame = self.frame
        s_tracker = self.s_tracker

        # Pan around, and highlight the two poles
        frame.reorient(0, 0, 0, (-0.44, -0.47, 0.0), 8.00)
        self.play(
            frame.animate.reorient(28, 69, 0, (-0.42, 0.29, 1.36), 7.45),
            s_tracker.animate.set_value(0.02 + 1j),
            run_time=10
        )
        self.play(
            frame.animate.reorient(-38, 81, 0, (-0.24, 0.38, 1.05), 8.77),
            s_tracker.animate.set_value(0.02 - 1j),
            run_time=10
        )
        self.play(
            frame.animate.reorient(13, 72, 0, (-1.41, 0.26, 1.11), 7.37),
            s_tracker.animate.set_value(0.1 + 0.57j),
            run_time=8
        )

        # Further panning
        self.play(
            frame.animate.reorient(0, 31, 0, (-1.08, -0.52, 0.3), 7.37),
            s_tracker.animate.set_value(0.01 + 1j),
            run_time=4
        )
        self.play(
            s_tracker.animate.set_value(0.2 - 1j),
            run_time=5,
        )
        self.play(
            frame.animate.reorient(21, 71, 0, (-1.48, -0.28, 0.8), 5.07),
            s_tracker.animate.set_value(0.5 + 1j),
            run_time=20
        )

        # Highlight the integral (Start here)
        exp_plane, int_plane = self.output_planes
        int_rect = SurroundingRectangle(int_plane.label, buff=SMALL_BUFF)
        int_rect.set_stroke(YELLOW, 2)
        int_rect.fix_in_frame()

        self.play(ShowCreation(int_rect))
        self.wait()
        self.play(int_rect.animate.surround(exp_plane.label, buff=SMALL_BUFF))
        self.wait()

        # Draw the upper plot
        self.draw_upper_plot()

        # Move s to 0.2i
        self.play(
            s_tracker.animate.set_value(0.2j),
            self.graph[0].animate.set_opacity(0.5),
            self.graph[1].animate.set_stroke(opacity=0.05),
            frame.animate.reorient(0, 15, 0, (-1.05, -1.03, 0.59), 4.66),
            run_time=4,
        )

        # Trace the path
        t_tracker = self.t_tracker
        t_tracker.set_value(0)
        get_t = t_tracker.get_value
        get_s = s_tracker.get_value

        output_path = self.output_path
        vect_sum = self.vect_sum
        int_vect = self.integral_vect
        exp_vect = Arrow(exp_plane.n2p(0), exp_plane.n2p(1), buff=0)
        exp_vect.set_fill(YELLOW)
        exp_vect.fix_in_frame()
        exp_vect.add_updater(lambda m: m.put_start_and_end_on(
            exp_plane.n2p(0),
            exp_plane.n2p(np.exp(-get_s() * get_t())),
        ))
        full_vect = exp_vect.copy().clear_updaters()
        full_vect.add_updater(lambda m: m.put_start_and_end_on(
            exp_plane.n2p(0),
            exp_plane.n2p(self.func(get_t()) * np.exp(-get_s() * get_t())),
        ))

        t_label = Tex(R"{t} = 0.0", t2c=self.tex_to_color_map)
        t_label.next_to(exp_plane.label, DOWN, LARGE_BUFF, aligned_edge=LEFT)
        t_label.make_number_changeable("0.0").f_always.set_value(t_tracker.get_value)
        t_label.fix_in_frame()

        output_path_copy = VMobject()
        output_path_copy.start_new_path(self.output_path[0].get_start())
        output_path_copy.fix_in_frame()
        for part in output_path:
            output_path_copy.append_vectorized_mobject(part)
        partial_path = output_path_copy.copy()
        partial_path.fix_in_frame()
        partial_path.add_updater(lambda m: m.pointwise_become_partial(
            output_path_copy, 0, get_t() / self.t_max
        ))

        vect_sum_copy = vect_sum.copy()
        vect_sum_copy.clear_updaters()
        growing_vect_sum = VGroup(*vect_sum_copy)
        growing_vect_sum.fix_in_frame()

        def update_growing_sum(group):
            group.set_submobjects(vect_sum_copy.submobjects[:int(get_t())])
            if len(group) == 0:
                return
            group[:-1].set_fill(opacity=0.35, border_width=0.5)
            group[-1].set_fill(opacity=1, border_width=2)

        growing_vect_sum.add_updater(update_growing_sum)

        t_tracker.clear_updaters()
        t_tracker.add_updater(lambda m, dt: m.increment_value(dt))

        self.play(
            int_rect.animate.surround(exp_plane.label[-4:], SMALL_BUFF),
            FadeOut(output_path, suspend_mobject_updating=True),
            FadeOut(vect_sum, suspend_mobject_updating=True),
            FadeOut(int_vect, suspend_mobject_updating=True),
            FadeIn(exp_vect),
            FadeIn(t_label),
        )
        self.add(t_tracker)
        self.wait(12)
        self.add(partial_path)
        self.play(
            int_rect.animate.surround(exp_plane.label, SMALL_BUFF),
            exp_vect.animate.set_fill(opacity=0.25),
            VFadeIn(full_vect),
        )
        self.wait(12)
        self.add(growing_vect_sum)
        self.play(int_rect.animate.surround(int_plane.label, buff=SMALL_BUFF))
        self.wait(20)
        self.play(
            frame.animate.reorient(25, 55, 0, (-0.91, -0.76, 0.83), 5.93),
            self.graph[0].animate.set_opacity(0.7),
            self.graph[1].animate.set_stroke(opacity=0.1),
            run_time=5
        )
        self.wait(5)
        growing_vect_sum.clear_updaters()
        self.play(
            VFadeOut(growing_vect_sum),
            VFadeOut(partial_path),
            VFadeOut(t_label),
            VFadeOut(exp_vect),
            VFadeOut(full_vect),
            VFadeOut(int_rect),
            VFadeIn(output_path),
            VFadeIn(vect_sum),
            FadeIn(int_vect)
        )

        # Add a small real part, increase imaginary
        self.play(
            s_tracker.animate.increment_value(0.05),
            run_time=3
        )
        self.draw_upper_plot(draw_time=12, rate_multiple=6)
        self.play(
            s_tracker.animate.increment_value(0.8j),
            run_time=20,
        )
        self.draw_upper_plot(draw_time=12, rate_multiple=4)
        self.play(
            s_tracker.animate.increment_value(-0.03),
            frame.animate.reorient(29, 78, 0, (-0.48, 0.16, 2.64), 10.00),
            run_time=7,
        )

        # Walk the imaginary line
        self.play(s_tracker.animate.increment_value(0.05), run_time=3)
        self.wait()
        self.play(s_tracker.animate.increment_value(1.3j), run_time=20)
        self.play(s_tracker.animate.increment_value(-0.5j), run_time=10)
        self.play(s_tracker.animate.increment_value(1), run_time=6)
        self.play(s_tracker.animate.increment_value(-0.8), run_time=6)
        self.play(s_tracker.animate.increment_value(-0.2j), run_time=6)
        self.play(s_tracker.animate.set_value(0.2j), run_time=3)

        # Pan
        self.play(frame.animate.reorient(-50, 67, 5, (0.47, -1.1, 1.08), 11.24), run_time=10)

    def func(self, t):
        return np.cos(t)

    def transformed_func(self, s):
        return np.divide(s, s**2 + 1**2)


class SimplePole(InteractiveScene):
    def construct(self):
        frame = self.frame
        plane = ComplexPlane((-3, 3), (-3, 3))
        a = complex(-0.5 + 1.5j)
        graph = get_complex_graph(plane, lambda s: np.divide(1.0, (s - a)))

        self.add(plane, graph)
        frame.reorient(-22, 89, 0, (0.31, 0.85, 3.73), 12.49)
        self.play(frame.animate.reorient(23, 88, 0, (0.31, 0.85, 3.73), 12.49), run_time=10)
