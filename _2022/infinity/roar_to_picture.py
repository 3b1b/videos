from manim_imports_ext import *
from _2022.piano.fourier_animations import DecomposeAudioSegment
from _2019.diffyq.part2.fourier_series import FourierCirclesScene


class DecomposeRoar(DecomposeAudioSegment):
    audio_file = "/Users/grant/Dropbox/3Blue1Brown/videos/2022/infinity/Roar.wav"
    graph_point = 0.428
    zoom_rect_dims = (0.2, 6.0)

    def construct(self):
        self.add_full_waveform()
        self.zoom_in_on_segment(
            self.axes, self.graph,
            self.graph_point, self.zoom_rect_dims
        )
        self.prepare_for_3d()
        self.show_all_waves()

    def show_all_waves(self):
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
            wave.set_stroke(opacity=0.75)
            wave.amp = amp
            wave.freq = freq
            wave.phase = phase
            amps.append(amp)
            sine_waves.add(wave)

        sine_waves.set_submobject_colors_by_gradient(YELLOW, GREEN, RED, ORANGE)
        sine_waves.set_stroke(width=3)

        # Break down
        frame = self.camera.frame
        frame.generate_target()
        frame.target.reorient(67, 71)
        frame.target.set_height(10.5)
        frame.target.move_to([2.6, 6.15, 0.79])

        self.play(
            FadeIn(f_axes),
            MoveToTarget(frame),
            LaggedStart(
                *(
                    ReplacementTransform(
                        graph.copy().set_opacity(0),
                        wave
                    )
                    for wave in sine_waves
                ),
                lag_ratio=1 / len(sine_waves),
            ),
            run_time=8
        )
        self.wait(3)


class CombineWavesToImage(FourierCirclesScene):
    CONFIG = {
        "peace_sign_height": 6.0,
        "n_shown_waves": 5,
        "n_vectors": 500,
        # "n_vectors": 50,
        "slow_factor": 0.1,
        "max_circle_stroke_width": 2.0,
        "parametric_function_step_size": 0.001,
        # "parametric_function_step_size": 0.01,
        "drawn_path_color": YELLOW,
        "remove_background_waves": False,
    }

    def construct(self):
        self.generate_coefs()
        self.show_sine_waves()
        self.piece_together_circles()

    def generate_coefs(self):
        drawing = self.get_drawing()
        coefs = self.get_coefficients_of_path(drawing)
        vectors = self.get_rotating_vectors(coefficients=coefs)
        circles = self.get_circles(vectors)

        self.style_vectors(vectors)
        self.style_circles(circles)

        self.path = drawing
        self.coefs = coefs
        self.vectors = vectors
        self.circles = circles

    def show_sine_waves(self):
        # Initialized copycat vectors and circles
        s = slice(1, 1 + self.n_shown_waves)
        shown_vectors = self.vectors[s].copy().clear_updaters()
        shown_circles = self.circles[s].copy().clear_updaters()
        scale_factor = 1.0 / 3.0
        for vect, circ in zip(shown_vectors, shown_circles):
            vect.scale(scale_factor)
            circ.scale(scale_factor)
            vect.center_point = VectorizedPoint()
            vect.center_func = vect.center_point.get_center
            circ.center_func = vect.center_point.get_center
            vect.add_updater(lambda v: v.set_angle(
                np.log(v.coefficient).imag + self.get_vector_time() * v.freq * TAU
            ))
            vect.add_updater(lambda v: v.shift(v.center_func() - v.get_start()))
            circ.add_updater(lambda c: c.move_to(c.center_func()))
        shown_circles.set_stroke(width=2)
        shown_vectors.set_stroke(width=2)
        graphs = VGroup()

        # Add axes
        max_y = int(np.ceil(max(v.get_length() for v in shown_vectors)))
        all_axes = VGroup(*(
            Axes(
                x_range=(0, 8 * PI, PI),
                y_range=(-max_y, max_y),
                height=1.0,
                width=18,
            )
            for n in range(self.n_shown_waves)
        ))
        all_axes.arrange(DOWN, buff=1.0)

        dots = VGroup(Tex("\\vdots", font_size=100), Tex(""))
        dots.arrange(RIGHT)
        dots.next_to(all_axes, DOWN, buff=0.5)
        dots.match_x(all_axes[0].get_origin())

        frame = self.camera.frame
        group = Group(all_axes, dots)
        group.move_to(0.5 * UP)
        group.set_height(7)
        frame.set_height(group.get_height() + 1)
        frame.move_to(group)
        frame.shift(0.5 * LEFT)

        # Add graphs
        colors = color_gradient([RED, YELLOW], self.n_shown_waves)
        v2g_lines = VGroup()
        for vect, axes, color in zip(shown_vectors, all_axes, colors):
            vect.center_point.next_to(axes, LEFT, buff=0.75)
            axes.amp = vect.get_length() / axes.y_axis.get_unit_size()
            axes.vect = vect
            axes.color = color

            axes.graph = always_redraw(
                lambda a=axes: a.get_graph(
                    lambda x: a.amp * math.sin(a.vect.freq * x + a.vect.get_angle()),
                    stroke_width=2,
                    stroke_color=a.color
                )
            )
            graphs.add(axes.graph)

            line = Line()
            line.set_stroke(WHITE, 1.0)
            line.vect = vect
            line.graph = axes.graph
            line.add_updater(lambda l: l.put_start_and_end_on(
                l.vect.get_end(), l.graph.get_start()
            ))
            v2g_lines.add(line)

        # Animate the addition of all these?
        self.wait(1)
        for mobs in zip(all_axes, graphs, shown_vectors, shown_circles, v2g_lines):
            self.add(*mobs)
            self.wait(0.25)
        self.add(dots)

        self.all_axes = all_axes
        self.shown_vectors = shown_vectors
        self.shown_circles = shown_circles
        self.graphs = graphs
        self.v2g_lines = v2g_lines
        self.dots = dots

    def piece_together_circles(self):
        # Setup path and labels
        true_path = self.path
        fade_region = self.get_fade_region(true_path)

        sf = self.get_slow_factor()
        rt = 5
        future_vt = self.get_vector_time() + sf * rt
        self.wait((1.0 - (future_vt % 1.0)) / sf)

        label = VGroup(Integer(100, edge_to_fix=DR), Text("terms"))
        label.arrange(RIGHT, buff=0.15)
        label.next_to(true_path, UP, buff=0.85)
        label[0].set_value(self.n_shown_waves)

        # Animate entry
        drawn_path = self.get_drawn_path(self.vectors[:self.n_shown_waves + 1])
        terms = self.get_terms(self.n_shown_waves)
        self.play(
            FadeIn(fade_region, lag_ratio=0.1),
            LaggedStart(*(
                TransformFromCopy(v1, v2)
                for v1, v2 in zip(self.shown_vectors, self.vectors[1:])
            ), lag_ratio=0.1),
            LaggedStart(*(
                TransformFromCopy(c1, c2)
                for c1, c2 in zip(self.shown_circles, self.circles[1:])
            ), lag_ratio=0.1),
            FadeIn(label),
            Write(terms),
            # self.slow_factor_tracker.animate.set_value(0.1),
            run_time=rt,
        )
        self.vector_clock.set_value(0)

        # Set up drawn paths
        self.add(drawn_path, terms)
        if self.remove_background_waves:
            self.play(
                fade_region.animate.set_height(2 * FRAME_WIDTH).set_opacity(0.05),
                run_time=2
            )
            self.remove(
                self.all_axes, self.graphs,
                self.shown_vectors, self.shown_circles,
                self.v2g_lines, self.dots,
                fade_region,
            )
        else:
            self.wait(2)

        # Now add on new vectors one at a time
        drawn_paths = dict()
        all_terms = dict()

        drawing_group = VGroup(
            drawn_path,
            terms,
            label,
            VGroup(),  # Vectors
            VGroup(),  # Circles
        )

        def update_drawing_group(group, alpha):
            n, _ = integer_interpolate(self.n_shown_waves, self.n_vectors, alpha)
            if n not in drawn_paths:
                drawn_paths[n] = self.get_drawn_path(self.vectors[:n])
            if n not in all_terms:
                all_terms[n] = self.get_terms(n)
            group[0].become(drawn_paths[n])
            group[1].become(all_terms[n])
            group[2][0].set_value(n)
            group[3].set_submobjects(self.vectors[:n])
            group[4].set_submobjects(self.circles[:n])

            self.style_vectors(group[3])
            self.style_circles(group[4])

            group.update()
            return group

        self.remove(self.vectors, self.circles, drawn_path, terms)

        self.play(UpdateFromAlphaFunc(
            drawing_group, update_drawing_group,
            run_time=15,
            rate_func=lambda a: a**2,
        ))

        inf = Tex("\\infty")
        inf.match_height(label[0])
        inf.move_to(label[0], RIGHT)

        self.play(
            FadeIn(true_path),
            VFadeOut(drawn_path),
            ChangeDecimalToValue(label[0], 1000),
            VFadeOut(label[0]),
            FadeIn(inf),
            run_time=3,
        )
        self.wait(12)

    ##

    def get_drawing(self):
        # drawing = self.get_peace_sign()
        svg = SVGMobject("/Users/grant/Dropbox/3Blue1Brown/videos/2022/infinity/monster-holding-heart.svg")
        drawing = svg[1]
        drawing.set_height(self.peace_sign_height)
        drawing.set_stroke(self.drawn_path_color, 2)
        return drawing

    def get_peace_sign(self):
        theta = 40 * DEGREES
        arc1 = Arc(270 * DEGREES, -theta, n_components=1)
        arc2 = Arc(270 * DEGREES + theta, -(theta + PI), n_components=7)
        circ = Circle(start_angle=90 * DEGREES, n_components=12)
        parts = [
            circ,
            Line(UP, ORIGIN),
            Line(ORIGIN, DOWN),
            arc1,
            Line(arc1.get_end(), ORIGIN),
            Line(ORIGIN, arc2.get_start()),
            arc2,
        ]
        path = VMobject()
        for part in parts:
            path.append_points(part.get_points())

        return path

    def style_vectors(self, vectors):
        for k, vector in enumerate(vectors):
            vector.set_stroke(width=2.5 / (1 + k / 10), opacity=1)
        return vectors

    def style_circles(self, circles):
        mcsw = self.max_circle_stroke_width
        circles.set_stroke(Color('green'))
        for k, circle in zip(it.count(0), circles):
            circle.set_stroke(width=mcsw / (1 + k / 4), opacity=1)

        if len(circles) > 25:
            circles.set_stroke(opacity=(1.0 / (len(circles) - 25))**0.2)

        return circles

    def get_fade_region(self, path, n_disks=100):
        disk = Circle(radius=path.get_width() / 2)
        disks = VGroup(*(
            disk.copy().scale(sf)
            for sf in np.linspace(1.0, 2.25, n_disks)
        ))
        disks.set_stroke(width=0)
        disks.set_fill(BLACK, opacity=3.0 / n_disks)
        return disks

    def get_drawn_path(self, vectors):
        if len(vectors) == 0:
            return VGroup()
        return super().get_drawn_path(vectors, stroke_width=4).set_stroke(self.drawn_path_color)

    def get_terms(self, n_terms, max_terms=40, max_width=12):
        tex_str = ""
        ks = list(range(-int(np.floor(n_terms / 2)), int(np.ceil(n_terms / 2))))
        ks.sort(key=abs)
        ks = ks[:max_terms]
        ks.sort()
        for k in ks:
            tex_str += "c_{k} e^{k \\cdot 2 \\pi i \\cdot t} +".replace("k", str(k))
        if n_terms <= max_terms:
            tex_str = tex_str[:-1]  # Remove trailing plus
        else:
            tex_str = "\\cdots + " + tex_str + "\\cdots"

        font_size = 180 / n_terms
        result = Tex(tex_str, font_size=font_size)
        result.set_max_width(max_width)
        result.next_to(self.path, UP, MED_SMALL_BUFF)

        max_light_terms = 15
        if n_terms > max_light_terms:
            result.set_fill(opacity=max_light_terms / (n_terms - max_light_terms))
            result.set_stroke(width=0)

        return result
