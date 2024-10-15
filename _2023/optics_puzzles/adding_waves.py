from manim_imports_ext import *
from _2023.optics_puzzles.objects import *


class WhiteLightAsASum(InteractiveScene):
    def construct(self):
        # Initial orientation
        frame = self.frame
        frame.reorient(0, 90)

        # Create axes
        n_colors = 15
        axes_width = 9.0
        x_max = 8
        spectral_axes = VGroup(*(
            ThreeDAxes((0, x_max), (-1, 1), (-1, 1))
            for _ in range(n_colors)
        ))
        spectral_axes.set_width(axes_width)
        spectral_axes.arrange(UP, buff=0.05)

        white_axes = ThreeDAxes((0, x_max), (-1, 1), (-1, 1))
        white_axes.set_width(axes_width)
        white_axes.shift(2 * DOWN)
        spectral_axes.next_to(white_axes, UP)

        all_axes = [white_axes, *spectral_axes]
        for axes in all_axes:
            axes.y_axis.set_opacity(0)

        for axes in spectral_axes:
            axes.z_axis.stretch(0.5, 2)

        white_axes.z_axis.stretch(1.5, 2)

        # Create individual waves
        white_parts = VGroup()
        spectral_waves = VGroup()
        alphas = np.linspace(0.0, 1.0, n_colors)
        for alpha, s_axes in zip(alphas, spectral_axes):
            wave_len = interpolate(1.0, 2.0, alpha)
            for axes, group in [(s_axes, spectral_waves), (white_axes, white_parts)]:
                wave = OscillatingWave(
                    axes,
                    y_amplitude=0,
                    z_amplitude=1,
                    wave_len=wave_len,
                    color=get_spectral_color(1 - alpha)
                )
                group.add(wave)

        # Show white wave
        white_wave = MeanWave(white_parts)
        white_wave.set_stroke(width=1)
        white_vects = OscillatingFieldWave(white_axes, white_wave, tip_width_ratio=3)
    
        self.add(white_axes, white_wave, white_vects)
        self.wait(5)

        # Show spectral parts
        symbols = Tex("=" + "+" * (n_colors - 1))
        symbols.scale(2)
        symbols.rotate(90 * DEGREES)
        for symbol, a1, a2 in zip(symbols, all_axes, all_axes[1:]):
            symbol.move_to(VGroup(a1, a2))

        spectral_vect_waves = VGroup()
        for sa, sw in zip(spectral_axes, spectral_waves):
            vwave = OscillatingFieldWave(sa, sw, tip_width_ratio=3)
            sw.set_stroke(width=1)
            spectral_vect_waves.add(vwave)

        self.play(
            frame.animate.reorient(50, 65, 0).move_to([0.6, 2.9, -0.16]).set_height(11).set_anim_args(run_time=3),
            LaggedStartMap(FadeIn, spectral_axes),
            LaggedStartMap(VFadeIn, spectral_waves),
            LaggedStartMap(VFadeIn, spectral_vect_waves),
            LaggedStartMap(FadeIn, symbols),
        )
        self.play(
            self.frame.animate.reorient(19, 65, 0).move_to([0.6, 2.9, -0.16]).set_height(11.00),
            run_time=8
        )
        return

        # Scrap
        arrows = VGroup(*(
            Arrow(
                sa.get_left(),
                interpolate(white_axes.get_corner(UR), white_axes.get_corner(DR), alpha),
                buff=0.3,
                stroke_color=wave.get_color()
            )
            for sa, wave, alpha in zip(
                spectral_axes,
                spectral_waves,
                np.linspace(0, 1, n_colors),
            )
        ))

        pass


class AddTwoSineWaves(InteractiveScene):
    def construct(self):
        # Setup axes
        axes1, axes2, axes3 = all_axes = [self.get_axes() for _ in range(3)]
        for axes, y in zip(all_axes, [2.5, 0, -3.0]):
            axes.set_y(y)
            axes.to_edge(RIGHT)

        # First two waves
        wave1 = self.get_variable_wave(axes1, color=BLUE, func_name="f(t)", index="1")
        wave2 = self.get_variable_wave(axes2, color=YELLOW, func_name="g(t)", index="2", omega=PI)

        self.play(
            FadeIn(axes1),
            ShowCreation(wave1[0]),
        )
        self.play(LaggedStartMap(FadeIn, wave1[2]),)
        self.change_all_parameters(wave1, 0.5, PI, PI / 4)
        self.wait()

        self.play(
            FadeIn(axes2),
            ShowCreation(wave2[0]),
        )
        self.play(LaggedStartMap(FadeIn, wave2[2]),)
        self.change_all_parameters(wave2, 0.75, 2 * PI, PI / 2)
        self.wait()

        self.add(axes2)
        self.add(wave2)

        # Sum wave
        wave3 = axes3.get_graph(
            lambda x: wave1.func(x) + wave2.func(x),
            color=GREEN,
            bind=True
        )
        wave3_label = Tex("f(t) + g(t)", font_size=36)
        wave3_label.move_to(axes3.c2p(0.5, 1.5), DL)

        self.play(LaggedStart(
            TransformFromCopy(axes1, axes3),
            TransformFromCopy(axes2, axes3),
            TransformFromCopy(wave1.labels[3]["f(t)"], wave3_label["f(t)"]),
            TransformFromCopy(wave2.labels[3]["g(t)"], wave3_label["+ g(t)"]),
            lag_ratio=0.05
        ))
        self.play(
            TransformFromCopy(wave1[0], wave3),
            TransformFromCopy(wave2[0], wave3),
        )
        self.wait()

        # Some example changes
        self.change_wave_parameter(wave1, "phi", TAU)
        self.change_wave_parameter(wave2, "A", 0.5)
        self.change_wave_parameter(wave1, "omega", 4.5)
        self.change_wave_parameter(wave1, "phi", 3 * PI / 4)
        self.change_wave_parameter(wave2, "omega", 3 * PI)
        self.change_wave_parameter(wave2, "phi", 5)
        self.wait()

        # Lock both frequencies
        frame = self.frame
        t2c = {R"\omega": GOLD}
        top_words = TexText(
            R"If both have matching \\ frequencies, $\omega$...",
            font_size=36,
            t2c=t2c
        )
        top_words.next_to(VGroup(wave1.labels, wave2.labels), LEFT, buff=2.0)

        omega_locks = VGroup()
        for wave in [wave1, wave2]:
            lock = SVGMobject("lock")
            lock.match_height(wave.labels[1])
            lock.next_to(wave.labels[1], LEFT, SMALL_BUFF)
            omega_locks.add(lock)

        omega_locks.set_color(GOLD)

        top_arrows = VGroup(*(
            Arrow(top_words.get_right(), lock.get_left())
            for lock in omega_locks
        ))

        self.play(
            frame.animate.set_x(-5),
            FadeIn(top_words, 0.5 * LEFT),
            LaggedStartMap(ShowCreation, top_arrows),
            run_time=2,
        )
        self.play(
            LaggedStartMap(FadeIn, omega_locks),
            *self.get_wave_change_animations(wave1, "omega", PI),
            *self.get_wave_change_animations(wave2, "omega", PI),
            run_time=2
        )
        self.wait()

        # Graph is also sine
        low_words = TexText(
            R"Then this is also a sine \\ wave with frequency $\omega$",
            font_size=36,
            t2c=t2c,
        )
        low_words.align_to(top_words, LEFT)
        low_words.match_y(axes3).shift(UP)
        low_arrow = Arrow(low_words.get_right(), axes3.get_left())

        self.play(
            FadeTransform(top_words.copy(), low_words),
            FadeIn(low_arrow, 2 * DOWN),
        )
        self.wait()

        # Show lower function label
        sum_sine = Tex(R"= 1.00\sin(\omega t +2.00)", t2c={R"\omega": GOLD}, font_size=36)
        sum_sine.next_to(wave3_label, RIGHT, SMALL_BUFF)
        sum_A = sum_sine.make_number_changeable("1.00")
        sum_phi = sum_sine.make_number_changeable("+2.00", include_sign=True)
        sum_A.set_color(RED)
        sum_phi.set_color(PINK)
        sum_parameters = VGroup(sum_A, sum_phi)

        def update_sum_parameters(params):
            A1 = wave1.trackers[0].get_value()
            A2 = wave2.trackers[0].get_value()
            phi1 = wave1.trackers[2].get_value()
            phi2 = wave2.trackers[2].get_value()
            z3 = A1 * np.exp(phi1 * 1j) + A2 * np.exp(phi2 * 1j)
            params[0].set_value(abs(z3))
            params[1].set_value(np.log(z3).imag)

        sum_parameters.add_updater(update_sum_parameters)

        self.play(Write(sum_sine))
        self.add(sum_parameters)

        # Ask about sum parameters
        param_arrows = VGroup()
        param_qmarks = VGroup()
        for param in sum_parameters:
            arrow = Vector(0.5 * DOWN)
            arrow.next_to(param, UP, buff=SMALL_BUFF)
            arrow.match_color(param)
            q_mark = Text("?")
            q_mark.match_color(param)
            q_mark.next_to(arrow, UP, SMALL_BUFF)
            param_arrows.add(arrow)
            param_qmarks.add(q_mark)
            self.play(ShowCreation(arrow), FadeIn(q_mark))
        self.wait()

        # Change the other parameters a bunch
        self.change_wave_parameter(wave1, "phi", 0)
        self.change_wave_parameter(wave1, "A", 1.0)
        self.change_wave_parameter(wave2, "phi", -0.8, run_time=4)
        self.change_wave_parameter(wave2, "A", 0.75)
        self.change_wave_parameter(wave1, "phi", PI / 3)

        # Clear the board
        low_fade_rect = FullScreenFadeRectangle()
        low_fade_rect.set_height(5.25, about_edge=DOWN, stretch=True)
        low_fade_rect.set_fill(BLACK, 0.85)
        self.play(LaggedStart(
            FadeOut(top_words),
            FadeOut(top_arrows),
            FadeOut(low_words),
            FadeOut(low_arrow),
            FadeOut(param_arrows),
            FadeOut(param_qmarks),
            FadeIn(low_fade_rect),
        ))

        # Show first and second phasors
        phasor1 = self.get_phasor(axes1, wave1)
        phasor2 = self.get_phasor(axes2, wave2)

        for wave, phasor in [(wave1, phasor1), (wave2, phasor2)]:
            # Setup and add phasor
            A_label = wave.labels[0][:2]
            phi_label = wave.labels[2][:2]
            phasor.A_label = A_label.copy()
            phasor.A_label.scale(0.75)
            max_A_height = phasor.A_label.get_height()
            phasor.A_label.vector = phasor.vector
            phasor.A_label.add_updater(lambda m: m.set_height(min(max_A_height, m.vector.get_length())))
            phasor.A_label.add_updater(lambda m: m.next_to(
                m.vector.pfp(0.5),
                normalize(rotate_vector(
                    m.vector.get_vector(),
                    (1 if m.vector.get_angle() > 0 else -1) * 90 * DEGREES
                )),
                buff=0.05,
            ))
            phasor.phi_label = phi_label.copy()
            phasor.phi_label.scale(0.65)
            max_phi_height = phasor.phi_label.get_height()
            phasor.phi_label.plane = phasor.plane
            phasor.phi_label.arc = phasor.arc
            phasor.phi_label.add_updater(lambda m: m.set_height(min(max_phi_height, m.arc.get_height())))
            phasor.phi_label.add_updater(lambda m: m.move_to(m.plane.n2p(
                1.5 * m.plane.p2n(m.arc.pfp(0.5))
            )))

            self.play(
                FadeIn(phasor.plane, LEFT),
                FadeIn(phasor.rot_vect, LEFT),
            )
            self.wait()

            # Show y coordinate
            wave.output_indicator = self.get_output_indicator(
                wave.axes, wave.wave, phasor.t_tracker
            )
            self.play(
                FadeIn(phasor.y_line),
                FadeIn(phasor.y_dot),
                FadeIn(wave.output_indicator),
            )
            self.play(
                phasor.t_tracker.animate.set_value(4),
                run_time=8,
                rate_func=linear,
            )
            self.wait()
            self.play(
                FadeOut(wave.output_indicator),
                FadeOut(phasor.y_line),
                FadeOut(phasor.y_dot),
            )

            self.add(
                phasor.vector,
                phasor.rot_vect,
            )

            # Show amplitude and phase
            self.play(TransformFromCopy(A_label, phasor.A_label))
            self.play(
                wave.trackers[0].animate.set_value(1.5 * wave.trackers[0].get_value()),
                run_time=2,
                rate_func=there_and_back,
            )
            self.wait()
            self.play(
                TransformFromCopy(phi_label, phasor.phi_label),
                FadeIn(phasor.arc),
            )
            self.play(
                wave.trackers[2].animate.set_value(wave.trackers[2].get_value() + PI / 2),
                run_time=4,
                rate_func=there_and_back,
            )

            # Shrink rect
            self.play(
                low_fade_rect.animate.set_height(3, about_edge=DOWN, stretch=True)
            )

        # Show the sum
        sum_plane = ComplexPlane(
            (-1, 1), (-1, 1),
            background_line_style=dict(stroke_color=GREY_B, stroke_width=1),
            faded_line_style=dict(stroke_color=GREY_B, stroke_width=0.5, stroke_opacity=0.25),
            faded_line_ratio=4,
        )
        sum_plane.set_height(2.0)
        sum_plane.match_x(phasor1)
        sum_plane.match_y(axes3)
        sum_plane.to_edge(DOWN, buff=MED_SMALL_BUFF)
        low_vects = VGroup(*(
            Vector(
                stroke_width=3,
                stroke_color=wave.wave.get_color(),
            )
            for wave in [wave1, wave2]
        ))
        def update_low_vects(vects):
            z1 = phasor1.get_z()
            z2 = phasor2.get_z()
            vects[0].put_start_and_end_on(
                sum_plane.n2p(0),
                sum_plane.n2p(z1),
            )
            vects[1].put_start_and_end_on(
                sum_plane.n2p(z1),
                sum_plane.n2p(z1 + z2),
            )

        low_vects.add_updater(update_low_vects)

        self.play(
            FadeIn(sum_plane),
            FadeOut(low_fade_rect),
        )
        self.wait()
        self.play(
            TransformFromCopy(phasor1.rot_vect, low_vects[0]),
            TransformFromCopy(phasor2.rot_vect, low_vects[1]),
        )
        self.add(low_vects)
        self.wait()

        # Show the sum of the two vectors
        sum_vect = Vector(stroke_color=wave3.get_color(), stroke_width=3)
        sum_vect.add_updater(lambda m: m.put_start_and_end_on(
            sum_plane.n2p(0),
            sum_plane.n2p(phasor1.get_z() + phasor2.get_z()),
        ))

        sum_output_indicator = self.get_output_indicator(axes3, wave3, phasor1.t_tracker)

        def show_rotation(max_t=4):
            for phasor in [phasor1, phasor2]:
                phasor.t_tracker.set_value(0)
            self.add(
                wave1.output_indicator,
                wave2.output_indicator,
                sum_output_indicator,
            )
            self.play(
                phasor1.t_tracker.animate.set_value(max_t),
                phasor2.t_tracker.animate.set_value(max_t),
                run_time=4 * max_t,
                rate_func=linear,
            )
            self.play(*map(FadeOut, (
                wave1.output_indicator,
                wave2.output_indicator,
                sum_output_indicator,
            )))

        self.play(ShowCreation(sum_vect))
        show_rotation()
        self.wait()

        # Emphasize amplitude and phase of the new wave
        param_rects = VGroup(*(
            SurroundingRectangle(param).match_color(param)
            for param in sum_parameters
        ))
        param_rects.set_stroke(width=2)

        sum_arc = Arc(angle=sum_parameters[1].get_value(), radius=0.5, arc_center=sum_plane.n2p(0))
        sum_arc.set_stroke(WHITE, 2)

        sum_A_copy, sum_phi_copy = sum_parameters.copy()
        sum_A_copy.scale(0.5)
        sum_A_copy.rotate(sum_vect.get_angle())
        sum_A_copy.next_to(sum_vect.get_center(), UP, buff=0.05)
        sum_phi_copy.scale(0.25)
        sum_phi_copy.next_to(sum_arc, RIGHT, buff=0.05)

        self.play(
            ShowCreation(param_arrows[0]),
            FadeIn(param_qmarks[0]),
            ShowCreation(param_rects[0]),
        )
        self.wait()
        self.play(
            TransformFromCopy(sum_parameters[0], sum_A_copy),
        )
        self.wait()
        self.play(
            ReplacementTransform(*param_arrows),
            ReplacementTransform(*param_qmarks),
            ReplacementTransform(*param_rects),
        )
        self.wait()
        self.play(
            TransformFromCopy(sum_parameters[1], sum_phi_copy),
            ShowCreation(sum_arc)
        )
        self.wait()
        self.play(*map(FadeOut, (
            sum_A_copy, sum_phi_copy, sum_arc,
            param_arrows[1], param_qmarks[1], param_rects[1]
        )))

        self.wait()

        # Play around with some alternate values, again
        phi0 = PI / 6
        self.play(
            *self.get_wave_change_animations(wave1, "phi", phi0),
            *self.get_wave_change_animations(wave2, "phi", phi0),
            run_time=3
        )
        self.wait()
        self.play(
            *self.get_wave_change_animations(wave2, "phi", phi0 - PI),
            run_time=3
        )
        self.wait()
        self.change_wave_parameter(wave2, "phi", phi0 - PI / 2, run_time=3)
        self.change_wave_parameter(wave2, "A", 0.2)
        self.wait()

        # Highlight sum offset
        self.play(FlashAround(low_vects, run_time=2))
        self.wait()
        show_rotation(4)

        # Bigger and smaller shift
        for value in [0.1, 0.3]:
            self.change_wave_parameter(wave2, "A", value)
            self.wait()


    def get_axes(
        self,
        x_range=(0, 8),
        y_range=(-1, 1),
        width=11,
        height=1.25
    ):
        result = Axes(
            x_range, y_range,
            width=width,
            height=height
        )
        result.add_coordinate_labels(font_size=16, buff=0.15)
        return result

    def get_variable_wave(
        self,
        axes,
        color=YELLOW,
        func_name="f(t)",
        A=1.0,
        omega=TAU,
        phi=0.0,
        index="",
        label_font_size=36,
        parameter_font_size=24,
        A_color=RED,
        k_color=GOLD,
        phi_color=PINK,
        func_name_coords=(0.5, 1.25)
    ):
        A_tracker = ValueTracker(A)
        k_tracker = ValueTracker(omega)
        phi_tracker = ValueTracker(phi)
        trackers = Group(A_tracker, k_tracker, phi_tracker)

        A_tex = f"A_{{{index}}}"
        omega_tex = Rf"\omega_{{{index}}}"
        phi_tex = Rf"\phi_{{{index}}}"
        t2c = {
            A_tex: A_color,
            omega_tex: k_color,
            phi_tex: phi_color,
        }

        get_A = A_tracker.get_value
        get_k = k_tracker.get_value
        get_phi = phi_tracker.get_value

        def func(x):
            return get_A() * np.sin(get_k() * x + get_phi())

        wave = axes.get_graph(
            func,
            stroke_color=color,
            bind=True
        )
        labels = VGroup(*(
            Tex(tex, font_size=parameter_font_size, t2c=t2c)
            for tex in [
                f"{A_tex} = 1.00",
                f"{omega_tex} = 1.00",
                f"{phi_tex} = 1.00",
            ]
        ))
        labels.arrange(DOWN, aligned_edge=RIGHT)
        labels.next_to(axes, LEFT, MED_LARGE_BUFF)
        for label, tracker in zip(labels, trackers):
            label.tracker = tracker
            label.value = label.make_number_changeable("1.00")
            label.add_updater(lambda m: m.value.set_value(m.tracker.get_value()))

        name_label = Tex(
            fR"{func_name} = {A_tex} \sin({omega_tex} t + {phi_tex})",
            tex_to_color_map=t2c,
            font_size=label_font_size,
        )
        name_label.move_to(axes.c2p(*func_name_coords), DL)
        labels.add(name_label)

        result = Group(wave, trackers, labels)
        result.axes = axes
        result.func = func
        result.wave = wave
        result.trackers = trackers
        result.labels = labels

        return result

    def get_phasor(self, axes, wave_group, plane_height=2.0):
        # Plane
        plane = ComplexPlane(
            (-1, 1), (-1, 1),
            background_line_style=dict(stroke_color=GREY_B, stroke_width=1),
            faded_line_style=dict(stroke_color=GREY_B, stroke_width=0.5, stroke_opacity=0.5),
            faded_line_ratio=4,
        )
        plane.set_height(plane_height)
        plane.next_to(axes, LEFT, buff=4.0)

        # Initial arrow
        graph, trackers, labels = wave_group
        get_A, get_omega, get_phi = (t.get_value for t in trackers)

        def get_z0():
            return get_A() * np.exp(get_phi() * 1j)

        vector = Arrow(
            plane.n2p(0), plane.n2p(1),
            buff=0,
            stroke_color=GREY_B,
            stroke_width=3,
            stroke_opacity=0.5
        )
        vector.add_updater(lambda m: m.put_start_and_end_on(
            plane.n2p(0), plane.n2p(get_z0())
        ))

        # Arc
        arc = always_redraw(lambda: Arc(
            angle=get_phi(),
            radius=min(plane_height / 6, 0.5 * vector.get_length()),
            arc_center=plane.n2p(0)
        ).set_stroke(WHITE, 1))

        # t tracker
        t_tracker = ValueTracker(0)
        get_t = t_tracker.get_value
        rot_vect = vector.copy()
        rot_vect.set_stroke(graph.get_color(), opacity=1)

        def get_z():
            return np.exp(get_omega() * get_t() * 1j) * get_z0()

        rot_vect.add_updater(lambda m: m.put_start_and_end_on(
            plane.n2p(0), plane.n2p(get_z())
        ))

        # Y glow dot
        y_dot = GlowDot(color=graph.get_color())
        y_dot.add_updater(lambda m: m.match_x(plane).match_y(rot_vect.get_end()))
        y_line = Line()
        y_line.set_stroke(WHITE, 1, 0.5)
        y_line.add_updater(lambda m: m.put_start_and_end_on(
            rot_vect.get_end(), y_dot.get_center()
        ))

        # Result
        result = Group(plane, vector, arc, t_tracker, rot_vect, y_dot, y_line)
        result.plane = plane
        result.vector = vector
        result.arc = arc
        result.t_tracker = t_tracker
        result.rot_vect = rot_vect
        result.y_dot = y_dot
        result.y_line = y_line
        result.get_z = get_z

        return result

    def get_output_indicator(self, axes, graph, t_tracker):
        # Triangle
        func = graph.underlying_function
        get_t = t_tracker.get_value
        triangle = ArrowTip(angle=PI / 2)
        triangle.set_height(0.1)
        triangle.set_fill(GREY_C)
        triangle.add_updater(lambda m: m.move_to(axes.x_axis.n2p(get_t()), UP))

        # Glow dot
        dot = GlowDot(color=graph.get_color())
        dot.add_updater(lambda m: m.move_to(axes.c2p(get_t(), func(get_t()))))

        # Vertical line
        v_line = Line()
        v_line.set_stroke(WHITE, 1, 0.5)
        v_line.add_updater(lambda m: m.put_start_and_end_on(
            axes.x_axis.n2p(get_t()),
            dot.get_center()
        ))

        result = Group(dot, v_line)
        return result

    def get_wave_change_animations(self, wave_group, parameter, value):
        index = ["A", "omega", "phi"].index(parameter)
        if len(parameter) > 1:
            parameter = Rf"\{parameter}"
        wave, trackers, labels = wave_group

        rect1 = SurroundingRectangle(labels[index])
        rect2 = SurroundingRectangle(labels[3][parameter])
        rect1.stretch(1.1, 0, about_edge=LEFT)
        rect2.stretch(1.25, 0, about_edge=LEFT)
        rects = VGroup(rect1, rect2)
        rects.match_color(labels[index][0])

        return [
            VFadeInThenOut(rects, rate_func=lambda t: there_and_back(t)**0.5),
            trackers[index].animate.set_value(value),
        ]

    def change_wave_parameter(self, wave_group, parameter, value, run_time=2):
        self.play(
            *self.get_wave_change_animations(wave_group, parameter, value, ),
            run_time=run_time
        )

    def change_all_parameters(self, wave_group, *values, **kwargs):
        parameters = ["A", "omega", "phi"]
        for parameter, value in zip(parameters, values):
            if value is not None:
                self.change_wave_parameter(wave_group, parameter, value)


class AddTwoRotatingVectors(InteractiveScene):
    def construct(self):
        pass


class WavePlusLayerInfluence(InteractiveScene):
    default_frame_orientation = (-90, 0, 90)

    def construct(self):
        # Initialize axes
        frame = self.frame
        boxes = FullScreenRectangle().replicate(3)
        boxes.set_height(FRAME_HEIGHT / 3, stretch=True)
        boxes.arrange(DOWN, buff=0)

        x_range = (-12, 12)
        y_range = (-1, 1)
        z_range = (-1, 1)
        top_axes, mid_axes, low_axes = (
            ThreeDAxes(x_range, y_range, z_range),
            Axes(x_range, y_range),
            Axes(x_range, y_range),
        )
        all_axes = VGroup(top_axes, mid_axes, low_axes)
        all_axes.move_to(boxes[1])
        low_axes.move_to(boxes[2])
        for axes in all_axes:
            axes.set_stroke(opacity=0)
            axes.x_axis.set_stroke(opacity=0.5)

        # Initialize labels
        text_kw = dict(font_size=36)
        labels = VGroup(
            Text("Incoming light", **text_kw),
            Text("Wave from layer oscillations", **text_kw),
            Text("Net effect", **text_kw),
        )
        for label, box in zip(labels, boxes):
            label.next_to(box.get_corner(UL), DR, buff=MED_SMALL_BUFF)
        labels[2].shift(0.25 * UP)

        # Initialize waves
        wave1 = OscillatingWave(
            top_axes,
            y_amplitude=0.75,
            z_amplitude=0.0,
            wave_len=4.0,
            speed=1.5,
        )
        wave2_scale_tracker = ValueTracker(0.2)

        def wave2_func(x):
            offset_x = np.abs(x) + wave1.wave_len / 4
            y, z = wave1.xt_to_yz(offset_x, wave1.time)
            return wave2_scale_tracker.get_value() * y

        def sum_func(x):
            return wave1.xt_to_yz(x, wave1.time)[0] + wave2_func(x)


        wave2 = mid_axes.get_graph(wave2_func, bind=True)
        wave2.set_stroke(BLUE, 2)
        wave3 = low_axes.get_graph(sum_func, bind=True)
        wave3.set_stroke(TEAL, 2)

        # Vect waves
        field_kw = dict(stroke_width=2, stroke_opacity=0.5, tip_width_ratio=3)
        vect_wave1 = OscillatingFieldWave(top_axes, wave1, **field_kw)
        vect_wave2 = GraphAsVectorField(
            mid_axes, wave2_func,
            stroke_color=wave2.get_color(),
            **field_kw
        )
        vect_wave3 = GraphAsVectorField(
            low_axes, sum_func,
            stroke_color=wave3.get_color(),
            **field_kw
        )
        for vect_wave in [vect_wave1, vect_wave2, vect_wave3]:
            vect_wave.insert_updater(lambda m: m.reset_sample_points(), index=0)

        wave1_group = VGroup(wave1, vect_wave1)
        wave2_group = VGroup(wave2, vect_wave2)
        wave3_group = VGroup(wave3, vect_wave3)

        # Layers of charges
        layer_xs = np.arange(0, 6, 0.25)
        layers = Group()
        for x in layer_xs:
            charges = DotCloud(color=BLUE)
            charges.to_grid(31, 15)
            charges.make_3d()
            charges.set_shape(3, 3)
            charges.set_radius(0.035)
            charges.set_opacity(0.5)
            charges.rotate(90 * DEGREES, UP)
            charges.sort_points(lambda p: np.dot(p, OUT + UP))
            charges.x = x
            charges.amplitude_tracker = ValueTracker(0)
            charges.add_updater(lambda m: m.move_to(mid_axes.c2p(
                m.x, m.amplitude_tracker.get_value() * wave1.xt_to_yz(m.x, wave1.time)[0], 0,
            )))
            layers.add(charges)

        # Glass
        glass = VCube()
        glass.set_fill(opacity=0.25)
        glass.deactivate_depth_test()                                                              
        glass.set_shading(0.5, 0.5, 0)
        buff = 0.1
        glass.set_shape(*(
            dim + buff
            for dim in layers.get_shape()
        ))
        glass.move_to(layers, LEFT)
        glass.sort(lambda p: -p[0])
        glass.set_stroke(WHITE, 0.5, 0.5)

        # Show initial wave, then add layers
        frame = self.frame
        frame.reorient(-135, 20, 135)

        self.add(top_axes)
        self.add(wave1, vect_wave1)
        self.wait(2)
        self.play(
            Write(glass, time_span=(0, 2)),
            frame.animate.reorient(-110, 10, 110).set_anim_args(run_time=4),
        )
        self.play(
            LaggedStart(*(
                layer.amplitude_tracker.animate.set_value(0.1)
                for layer in layers
            ), lag_ratio=0, time_span=(0, 2)),
            LaggedStart(*(
                FadeIn(layer, suspend_mobject_updating=False)
                for layer in layers
            ), lag_ratio=0, time_span=(0, 2)),
        )
        self.play(
            self.frame.animate.reorient(-90, -40, 90),
            FadeOut(glass),
            run_time=9
        )
        self.wait(3)

        # Go back to just one layer
        self.play(
            LaggedStart(*(
                FadeOut(charges, suspend_mobject_updating=False)
                for charges in layers[1:]
            ), run_time=5, lag_ratio=0.25),
            layers[0].amplitude_tracker.animate.set_value(0.2),
            self.frame.animate.reorient(-90, -20, 90),
            run_time=5
        )
        self.wait(3)

        # Add the second order wave, then separate
        self.play(
            VFadeIn(wave2_group),
            FadeIn(mid_axes),
            run_time=1
        )
        self.wait(5)
        self.play(
            top_axes.animate.move_to(boxes[0], DOWN),
            frame.animate.reorient(-90, 0, 90).set_focal_distance(100),
            LaggedStartMap(FadeIn, labels[:2], shift=UP),
            run_time=2
        )
        self.wait(6)

        # Show sum (todo, add + and =)
        plus, eq = plus_eq = Tex("+=", font_size=72) 
        eq.rotate(90 * DEGREES)
        plus.move_to(all_axes[0:2])
        eq.move_to(all_axes[1:3])
        plus_eq.set_x(FRAME_WIDTH / 4)

        self.play(
            LaggedStartMap(FadeIn, plus_eq),
            ShowCreation(low_axes),
        )
        self.play(
            VFadeIn(wave3_group),
            Write(labels[2]),
        )
        self.wait(12)

        # Comment on reflected light
        reflection_label = VGroup(
            Vector(2 * LEFT),
            Text("Reflected light", font_size=36),
        )
        reflection_label.arrange(RIGHT)
        reflection_label.next_to(mid_axes.get_origin(), LEFT, buff=0.75)
        reflection_label.shift(0.5 * DOWN)
        reflection_label.set_color(YELLOW)

        self.play(
            GrowArrow(reflection_label[0]),
            Write(reflection_label[1]),
        )
        self.wait(8)

        # Cover left half
        cover = FullScreenFadeRectangle()
        cover.set_fill(BLACK, 0.9)
        cover.stretch(0.5, 0, about_edge=LEFT)

        self.add(cover, charges, labels)
        self.play(
            FadeOut(reflection_label),
            FadeIn(cover),
            plus_eq.animate.set_x(1.5),
            frame.animate.set_x(0.5 * FRAME_WIDTH - 2),
            *(
                label.animate.set_x(FRAME_WIDTH - 2.5, RIGHT)
                for label in labels
            ),
            run_time=2
        )
        self.wait(9)

        # Compare
        wave1.stop_clock()
        wave1_copy = wave1.copy()
        wave1_copy.clear_updaters()
        wave1_copy.set_stroke(width=3)

        self.wait()
        self.add(wave1_copy, cover, charges)
        self.play(
            wave1_copy.animate.match_y(wave3),
            run_time=2
        )
        self.wait()

        # Indicate tiny change
        def find_peak(wave, threshold=1e-2):
            points = wave.get_points()
            sub_points = points[int(0.6 * len(points)):int(0.8 * len(points))]
            top_y = wave.get_top()[1]
            index = np.argmax(sub_points[:, 1])
            return sub_points[index]

        line = Line(find_peak(wave3), find_peak(wave1_copy))
        shift_arrow = Vector(
            line.get_length() * LEFT * 2,
            stroke_width=3,
            max_tip_length_to_length_ratio=10,
        )
        shift_arrow.next_to(line, UP, buff=0.2)
        shift_label = Text("shift", font_size=24)
        always(shift_label.next_to, shift_arrow, UP)

        self.play(
            ShowCreation(shift_arrow),
            Write(shift_label),
        )
        self.wait()

        # Play with different strengths
        net_stretch = 1.0
        for stretch_factor in [3.0, 0.5, 2.0, 0.5, 1.0]:
            stretch = stretch_factor / net_stretch
            net_stretch = stretch_factor

            scale_arrows = VGroup(Vector(0.5 * UP), Vector(0.5 * DOWN))
            scale_arrows.arrange(DOWN if stretch > 1 else UP, buff=1.0)
            scale_arrows.move_to(mid_axes.c2p(2, 0))
            scale_arrows.set_stroke(opacity=1)
            scale_arrows.save_state()
            scale_arrows.stretch(0.5 if stretch > 1 else 1.5, 1)
            scale_arrows.set_stroke(opacity=0)

            self.play(
                Restore(scale_arrows),
                wave2_scale_tracker.animate.set_value(stretch * wave2_scale_tracker.get_value()),
                shift_arrow.animate.stretch(stretch, 0, about_edge=RIGHT),
            )
            self.play(FadeOut(scale_arrows))
            self.wait()

        self.play(
            wave2_scale_tracker.animate.set_value(0.2),
            FadeOut(wave1_copy),
            FadeOut(shift_arrow),
            FadeOut(shift_label),
        )

        # Restart
        wave1.start_clock()
        self.wait(8)


class Pulse(InteractiveScene):
    def construct(self):
        # Setup axes
        n_parts = 10
        x_max = 4
        left_axes_group = VGroup(*(
            Axes(
                (0, x_max), (-1, 1),
                width=0.5 * FRAME_WIDTH - 2,
                height=1,
                num_sampled_graph_points_per_tick=20,
            )
            for _ in range(n_parts)
        ))
        left_axes_group.arrange(DOWN, buff=0.75)
        left_axes_group.set_height(FRAME_HEIGHT - 1)
        left_axes_group.to_edge(LEFT)

        right_axes = Axes((0, x_max), (-1, 1), width=0.5 * FRAME_WIDTH - 1, height=3)
        right_axes.to_edge(RIGHT)

        brace = Brace(left_axes_group, RIGHT, buff=0.5)
        arrow = Arrow(brace, right_axes)
        sigma = Tex(R"\Sigma", font_size=72).next_to(arrow, UP)

        self.add(left_axes_group)
        self.add(right_axes)
        self.add(brace, arrow, sigma)

        # Graphs
        f0 = 6
        sigma = 2
        frequencies = np.linspace(0, 2 * f0, n_parts + 1)[1:]
        weights = 0.25 * np.exp(-0.5 * ((frequencies - f0) / sigma) ** 2)
        speeds = np.linspace(8, 10, n_parts)
        time_tracker = ValueTracker(0)
        colors = color_gradient([BLUE, YELLOW], n_parts)

        left_graphs = VGroup(*(
            self.create_graph(axes, freq, speed, color, time_tracker)
            for axes, freq, speed, color in zip(
                left_axes_group,
                frequencies,
                speeds,
                colors,
            )
        ))

        def sum_func(x):
            time = time_tracker.get_value()
            return sum(
                weight * self.wave_func(x, freq, speed, time)
                for weight, freq, speed in zip(weights, frequencies, speeds)
            )

        sum_graph = right_axes.get_graph(sum_func, bind=True)
        sum_graph.set_stroke(TEAL, 3)

        self.add(left_graphs)
        self.add(sum_graph)

        final_time = 20.0
        time_tracker.set_value(0)
        self.play(
            time_tracker.animate.increment_value(final_time),
            rate_func=linear,
            run_time=final_time,
        )


    def wave_func(self, x, freq, speed, time):
        return np.cos(freq * x - speed * time)

    def create_graph(self, axes, freq, speed, color, time_tracker):
        return axes.get_graph(
            lambda x: self.wave_func(x, freq, speed, time_tracker.get_value()),
            stroke_color=color,
            stroke_width=2,
            bind=True
        )

