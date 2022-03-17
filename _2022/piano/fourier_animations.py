from manim_imports_ext import *

from _2022.piano.wav_to_midi import DATA_DIR
from scipy.io import wavfile


def get_wave_sum(axes, freqs, amplitudes=None, phases=None):
    if amplitudes is None:
        amplitudes = np.ones(len(freqs))
    if phases is None:
        phases = np.zeros(len(freqs))
    return axes.get_graph(lambda t: sum(
        amp * math.sin(TAU * freq * (t - phase))
        for freq, amp, phase in zip(freqs, amplitudes, phases)
    ))


class SumOfWaves(Scene):
    def construct(self):
        # Show single pure wave
        axes = Axes(
            (0, 12), (-1, 1),
            height=2,
        )
        base_freq = 0.5
        wave = get_wave_sum(axes, [base_freq])
        wave.set_stroke(BLUE, 2)

        x = 4.5
        brace = Brace(
            Line(axes.i2gp(x, wave), axes.i2gp(x + 1 / base_freq, wave)),
            UP, buff=SMALL_BUFF
        )
        brace_label = brace.get_text(
            "220 cycles / sec.",
            buff=SMALL_BUFF,
            font_size=36,
        )

        axes_labels = VGroup(*(
            Text(word, font_size=30)
            for word in ["Air pressure", "Time"]
        ))
        axes_labels[0].next_to(axes.y_axis, UP).to_edge(LEFT)
        axes_labels[1].next_to(axes.x_axis, UP).to_edge(RIGHT)

        self.add(axes)
        brace_rf = squish_rate_func(smooth, 0.25, 0.5)
        label_rf = squish_rate_func(smooth, 0.25, 1)
        self.play(
            ShowCreation(wave, rate_func=linear),
            GrowFromCenter(brace, rate_func=brace_rf),
            Write(brace_label, rate_func=label_rf),
            run_time=3,
        )
        self.play(LaggedStartMap(
            Write, axes_labels,
            lag_ratio=0.8
        ))
        self.wait()

        # Show multiple waves
        freq_multiples = [1, 6 / 5, 3 / 2, 21 / 12]
        freqs = [base_freq * r for r in freq_multiples]

        low_axes_group = VGroup(*(
            Axes((0, 12), (-1, 1), height=0.65)
            for freq in freqs
        ))
        low_axes_group.arrange(UP, buff=0.4)
        low_axes_group.to_edge(DOWN)
        low_axes_group.to_edge(RIGHT)

        waves = VGroup(*(
            get_wave_sum(la, [freq])
            for la, freq in zip(low_axes_group, freqs)
        ))
        waves.set_submobject_colors_by_gradient(BLUE, YELLOW)
        waves.set_stroke(width=2)

        axes_labels = VGroup(*(
            Text(f"{int(mult * 220)} Hz", font_size=24)
            for mult in freq_multiples
        ))
        for low_axes, label in zip(low_axes_group, axes_labels):
            label.next_to(low_axes, LEFT)

        self.play(
            FadeOut(VGroup(axes_labels[1], brace), DOWN),
            ReplacementTransform(brace_label, axes_labels[0]),
            ReplacementTransform(axes, low_axes_group[0]),
            ReplacementTransform(wave, waves[0]),
            *(
                TransformFromCopy(axes, low_axes)
                for low_axes in low_axes_group[1:]
            )
        )
        self.play(
            LaggedStartMap(
                ShowCreation, waves[1:],
                lag_ratio=0.5,
                rate_func=linear,
            ),
            LaggedStartMap(
                FadeIn, axes_labels[1:],
                lag_ratio=0.5,
            ),
            run_time=2,
        )
        self.wait()

        # Show sum
        top_axes = Axes((0, 12), (-4, 4), height=2.25)
        top_axes.to_edge(UP, buff=MED_SMALL_BUFF)
        top_axes.align_to(low_axes_group, RIGHT)
        top_rect = Rectangle(FRAME_WIDTH, top_axes.get_height() + 0.5)
        top_rect.move_to(top_axes)
        top_rect.set_x(0)
        top_rect.set_stroke(WHITE, 0)
        top_rect.set_fill(GREY_E, 1.0)
        sum_label = Text("Sum")
        sum_label.to_edge(UP, buff=0.25)

        amp_tracker = ValueTracker(np.ones(len(freqs)))
        comp_wave = always_redraw(lambda: get_wave_sum(
            top_axes, freqs, amplitudes=amp_tracker.get_value(),
        ).set_stroke(TEAL, 2))

        self.play(
            FadeIn(top_rect),
            FadeIn(top_axes),
            FadeIn(sum_label),
            *(
                Transform(wave.deepcopy(), comp_wave, remover=True)
                for wave in waves
            )
        )
        self.add(comp_wave)
        self.wait()

        # Tweak magnitudes
        for index in range(len(waves)):
            wave = waves[index]
            wave.index = index
            wave.max_height = wave.get_height()
            wave.add_updater(lambda w: w.set_height(
                amp_tracker.get_value()[w.index] * w.max_height,
                stretch=True
            ))

        self.add(*waves)

        changes = [
            # (index, d_value)
            (3, -0.8),
            (2, -0.9),
            (1, 0.6),
            (0, 0.5),
            (3, 0.8),
            (0, -1.1),
            (1, 0.5),
        ]
        for index, d_value in changes:
            values = amp_tracker.get_value().copy()
            values[index] += d_value
            arrows = VGroup(
                Vector(0.5 * UP),
                Vector(0.5 * DOWN),
            )
            arrows.arrange(DOWN if d_value > 0 else UP)
            axes = low_axes_group[index]
            arrows.match_height(axes)
            arrows.next_to(axes, LEFT)

            self.play(
                amp_tracker.animate.set_value(values),
                FadeIn(arrows[0], 0.25 * UP),
                FadeIn(arrows[1], 0.25 * DOWN),
            )
            self.play(FadeOut(arrows, run_time=0.75))
        self.wait()


class DecomposeAudioSegment(Scene):
    audio_file = os.path.join(DATA_DIR, "audio_clips", "SignalFromSpeech.wav")
    sample_density = 1 / 5
    n_sine_waves = 5

    def construct(self):
        self.add_full_waveform()
        self.zoom_in_on_segment()
        self.prepare_for_3d()
        self.break_down_into_fourier_components()
        self.back_to_full_signal()

    def add_full_waveform(self):
        sample_rate, signal = wavfile.read(self.audio_file)
        signal = signal[:, 0] / np.abs(signal).max()
        signal = signal[::int(1 / self.sample_density)]

        axes = Axes(
            (0, len(signal), sample_rate * self.sample_density), (-1, 1, 0.25),
            height=6,
            width=15,
        )
        axes.to_edge(LEFT)

        xs = np.arange(len(signal))
        points = axes.c2p(xs, signal)
        graph = VMobject()
        graph.set_points_as_corners(points)
        graph.set_stroke(BLUE, 1.0)

        self.add(axes)
        self.play(
            ShowCreation(
                graph,
                rate_func=squish_rate_func(linear, 0.05, 1),
            ),
            VShowPassingFlash(
                graph.copy().set_stroke(BLUE_B, 3),
                time_width=0.1,
                rate_func=linear,
            ),
            run_time=5,
        )

        self.axes = axes
        self.graph = graph

    def zoom_in_on_segment(self):
        axes = self.axes
        graph = self.graph

        point = graph.pfp(0.428)[0] * RIGHT
        zoom_rect = Rectangle(0.4, 4.0)
        zoom_rect.move_to(point)
        zoom_rect.set_stroke(WHITE, 2)

        graph_snippet = VMobject()
        graph_points = graph.get_anchors()
        lx = zoom_rect.get_left()[0]
        rx = zoom_rect.get_right()[0]
        xs = graph_points[:, 0]
        snippet_points = graph_points[(xs > lx) * (xs < rx)]
        graph_snippet.set_points_as_corners(snippet_points)
        graph_snippet.match_style(graph)
        point = graph_snippet.get_center().copy()
        zoom_rect.move_to(point)

        movers = [axes, graph, graph_snippet, zoom_rect]

        frame = self.camera.frame
        for mover in movers:
            mover.save_state()
            mover.generate_target()
            mover.target.stretch(frame.get_width() / zoom_rect.get_width(), 0, about_point=point)
            mover.target.stretch(frame.get_height() / zoom_rect.get_height(), 1, about_point=point)
            mover.target.shift(-point)
        graph_snippet.target.set_stroke(width=3)
        zoom_rect.target.set_stroke(width=0)
        axes.target.set_stroke(opacity=0)

        self.play(Write(zoom_rect))
        self.play(
            *map(MoveToTarget, movers),
            run_time=4
        )
        self.remove(graph, axes)
        self.wait()

        # Swap axes
        new_axes = Axes((-2, 12), (-1, 1, 0.25), width=FRAME_WIDTH + 1)
        new_axes.shift(LEFT_SIDE + RIGHT - new_axes.get_origin())

        self.play(FadeIn(new_axes))

        self.original_graph = graph
        self.original_axes = axes
        self.axes = new_axes
        self.graph = graph_snippet

    def prepare_for_3d(self):
        frame = self.camera.frame
        for mob in self.mobjects:
            mob.rotate(PI / 2, RIGHT)
        frame.reorient(0, 90)
        self.add(frame)

    def break_down_into_fourier_components(self):
        t_axes = self.axes
        graph = self.graph

        # Take the fourier transform
        t_max = t_axes.x_range[1]
        ts, values = t_axes.p2c(graph.get_points()[::6])
        signal = values[(ts > 0) * (ts < t_max)]
        signal_fft = np.fft.fft(signal)
        signal_fft /= len(signal)
        signal_fft_abs = np.abs(signal_fft)
        signal_fft_phase = np.log(signal_fft).imag

        # Prepare the graph
        max_freq = signal.size / t_max
        f_axes = Axes(
            (0, max_freq / 2, max_freq / len(signal) / 2),
            (0, 1, 1 / 8),
            height=t_axes.get_depth(),
            width=150,
        )
        f_axes.rotate(PI / 2, RIGHT)
        f_axes.rotate(PI / 2, OUT)
        f_axes.shift(t_axes.get_origin() - f_axes.get_origin())
        freqs = np.fft.fftfreq(signal.size, 1 / max_freq) % max_freq

        fft_graph = VMobject()
        fft_graph.set_points_as_corners([
            f_axes.c2p(freq, 2 * value)
            for freq, value in zip(freqs, signal_fft_abs)
        ])
        fft_graph.set_stroke(GREEN, 3)
        freq_label = Text("Frequency", font_size=60)
        freq_label.rotate(PI / 2, RIGHT)
        freq_label.rotate(PI / 2, OUT)
        freq_label.next_to(f_axes.c2p(1.3, 0), OUT + UP)

        # Express the most dominant signals as sine waves
        sine_waves = VGroup()
        amps = []
        for index in range(1, 50):
            freq = freqs[index]
            amp = signal_fft_abs[index]
            phase = signal_fft_phase[index]
            wave = t_axes.get_graph(
                lambda t: 2 * amp * np.cos(TAU * freq * (t + phase)),
                x_range=(0, t_max),
            )
            wave.match_y(f_axes.c2p(freq, 0))
            wave.set_stroke(opacity=clip(15 * amp, 0.35, 1))
            wave.amp = amp
            wave.freq = freq
            wave.phase = phase
            amps.append(amp)
            sine_waves.add(wave)

        sine_waves.set_submobject_colors_by_gradient(YELLOW, GREEN, RED, ORANGE)
        sine_waves.set_stroke(width=3)
        top_waves = VGroup(*[sine_waves[i] for i in [4, 9, 13, 14]]).copy()

        # Break down
        frame = self.camera.frame
        frame.generate_target()
        frame.target.set_euler_angles(1.2, 1.35)
        frame.target.set_height(10.5)
        frame.target.move_to([1.5, 5.0, 0.7])

        self.play(
            FadeIn(f_axes),
            MoveToTarget(frame, run_time=8),
            LaggedStart(
                *(TransformFromCopy(graph, wave) for wave in top_waves),
                lag_ratio=0.8,
                run_time=3,
            )
        )
        frame.add_updater(lambda f, dt: f.increment_theta(0.25 * dt * DEGREES))
        self.play(Write(freq_label))
        self.wait(3)
        self.play(
            FadeIn(sine_waves, lag_ratio=0.1, run_time=3),
        )
        self.wait(3)

        # Collapse into FFT graph
        lines = VGroup(*(
            Line(f_axes.c2p(freqs[i], 0), f_axes.i2gp(freqs[i], fft_graph))
            for i in range(1, len(sine_waves))
        ))
        lines.set_stroke(GREEN, 2)
        lines.set_flat_stroke(False)

        frame.clear_updaters()
        frame.generate_target()
        frame.target.set_euler_angles(1.22, 1.54)
        frame.target.move_to([1.92, 7.29, 1.05])

        fft_label = TexText("|Fourier Transform|", font_size=60)
        fft_label.rotate(PI / 2, RIGHT).rotate(PI / 2, OUT)
        fft_label.next_to(f_axes.i2gp(freqs[5], fft_graph), OUT)
        fft_label.set_color(GREEN)

        piano = Piano()
        f_step = f_axes.x_range[2]
        piano.set_width(get_norm(f_axes.c2p(88 * f_step) - f_axes.get_origin()))
        piano.rotate(PI / 2, OUT)
        piano.move_to(f_axes.get_origin(), DR)
        piano.set_opacity(0.5)

        wave_shadows = sine_waves.copy().set_stroke(opacity=0.1)
        self.remove(top_waves, sine_waves)
        self.add(wave_shadows)
        self.play(
            LaggedStart(
                *(
                    TransformFromCopy(wave, line)
                    for wave, line in zip(sine_waves, lines)
                ),
                lag_ratio=0.1,
                run_time=8,
            ),
            graph.animate.set_stroke(width=1, opacity=0.5),
            ShowCreation(fft_graph, run_time=5),
            Write(fft_label),
            MoveToTarget(frame, run_time=5),
        )
        self.wait(2)
        self.add(piano, freq_label, fft_graph, lines)
        self.play(
            Write(piano),
            frame.animate.set_phi(1.25),
            run_time=3,
        )
        self.wait()

        # Pull out dominant signals
        glow_keys = VGroup(*(
            piano[np.argmin([
                get_norm(k.get_center() - wave.get_left())
                for k in piano
            ])]
            for wave in top_waves
        ))
        peak_dots = GlowDots([
            lines[np.argmin([
                get_norm(line.get_start() - wave.get_left())
                for line in lines
            ])].get_end()
            for wave in top_waves
        ])

        self.play(
            ShowCreation(peak_dots),
            LaggedStartMap(ShowCreation, top_waves),
            frame.animate.set_euler_angles(0.72, 1.15).move_to([2., 4., 1.]),
            ApplyMethod(glow_keys.set_fill, RED, 1, rate_func=squish_rate_func(smooth, 0, 0.2)),
            run_time=6,
        )
        self.wait()

        # Reconstruct
        approx_wave = graph.copy()  # Cheating
        approx_wave.set_points_smoothly(graph.get_points()[::150], true_smooth=True)
        approx_wave.set_stroke(TEAL, 3, 1.0)

        self.play(
            frame.animate.reorient(0, 90).move_to(ORIGIN).set_height(10),
            graph.animate.set_stroke(width=2, opacity=0.5),
            *(ReplacementTransform(wave, approx_wave) for wave in top_waves),
            LaggedStartMap(FadeOut, VGroup(fft_graph, lines, fft_label, freq_label, f_axes)),
            FadeOut(peak_dots),
            FadeOut(wave_shadows),
            FadeOut(piano),
            run_time=3,
        )
        self.wait()

        self.approx_wave = approx_wave

    def back_to_full_signal(self):
        # Back to original graph
        self.play(
            FadeOut(self.axes),
            FadeOut(self.approx_wave),
            self.graph.animate.set_stroke(opacity=1),
        )
        self.camera.frame.reorient(0, 0)
        self.graph.rotate(-PI / 2, RIGHT)
        self.play(
            Restore(self.original_axes),
            Restore(self.original_graph),
            Restore(self.graph),
            run_time=3,
        )

        # Show windows
        axes = self.original_axes
        graph = self.original_graph

        windows = Rectangle().get_grid(1, 75, buff=0)
        windows.replace(graph, stretch=True)
        windows.stretch(1.1, 1)
        windows.set_stroke(WHITE, 1)

        piano = Piano()
        piano.set_width(12)
        piano.next_to(axes, UP).set_x(0)
        piano.save_state()
        self.add(piano)

        for window in windows[:40]:
            fade_rect = BackgroundRectangle(axes)
            fade_rect.scale(1.01)
            fade_rect = Difference(fade_rect, window)
            fade_rect.set_fill(BLACK, 0.6)
            fade_rect.set_stroke(width=0)

            piano.restore()
            VGroup(*random.sample(list(piano), random.randint(1, 4))).set_color(RED)

            self.add(fade_rect, window)
            self.wait(0.25)
            self.remove(fade_rect, window)


class WaveformDescription(DecomposeAudioSegment):
    def construct(self):
        self.add_full_waveform()

        # Line passing over waveform
        axes = self.axes
        graph = self.graph

        line = Line(DOWN, UP)
        line.set_stroke(WHITE, 1)
        line.match_height(axes)
        line.move_to(axes.get_origin())
        line.add_updater(lambda l, dt: l.shift(0.1 * dt * RIGHT))

        dot = GlowDot()
        dot.add_updater(lambda d: d.move_to(axes.i2gp(
            axes.x_axis.p2n(line.get_x()),
            graph
        )))
        self.add(line, dot)

        # Words
        waveform = Text("Waveform", font_size=72)
        waveform.to_edge(UP)

        y_label = Text("Intensity", font_size=36)
        y_label.next_to(axes.y_axis, UP).shift_onto_screen()
        x_label = Text("Time", font_size=36)
        x_label.next_to(axes.x_axis, UP).to_edge(RIGHT, buff=SMALL_BUFF)

        self.wait(4)
        self.play(Write(waveform))
        self.wait(2)
        self.play(Write(y_label), run_time=1)
        self.play(Write(x_label), run_time=1)
        self.wait(10)
