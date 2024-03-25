from manim_imports_ext import *
from _2023.optics_puzzles.objects import *


class WaveIntoMedium(TimeVaryingVectorField):
    def __init__(
        self,
        interface_origin=ORIGIN,
        interface_normal=DR,
        prop_direction=RIGHT,
        index=1.5,
        c=2.0,
        frequency=0.25,
        amplitude=1.0,
        x_density=5.0,
        y_density=5.0,
        width=15.0,
        height=15.0,
        norm_to_opacity_func=lambda n: np.tanh(n),
        **kwargs
    ):
        def time_func(points, time):
            k = frequency / c
            phase = TAU * (k * np.dot(points, prop_direction.T) - frequency * time)
            kickback = np.dot(points - interface_origin, interface_normal.T)
            kickback[kickback < 0] = 0
            phase += kickback * index * c
            return amplitude * np.outer(np.cos(phase), OUT)

        super().__init__(
            time_func,
            x_density=x_density,
            y_density=y_density,
            width=width,
            height=height,
            norm_to_opacity_func=norm_to_opacity_func,
            **kwargs
        )


class ScalarFieldByOpacity(DotCloud):
    def __init__(
        self,
        # Takes (n, 3) array of points to n-array of values between 0 and 1
        opacity_func,
        width=15,
        height=8,
        density=10,
        color=WHITE,
    ):
        step = 1.0 / density
        radius = step / 2.0
        points = np.array([
            [x, y, 0]
            for x in np.arange(-width / 2, width / 2 + step, step)
            for y in np.arange(-height / 2, height / 2 + step, step)
        ])

        super().__init__(points, color=color, radius=radius)
        self.opacity_func = opacity_func

        def update_opacity(dots):
            dots.set_opacity(opacity_func(dots.get_points()))

        self.add_updater(update_opacity)


class WavesByOpacity(ScalarFieldByOpacity):
    def __init__(
        self,
        wave: VectorField,
        vects_to_opacities=lambda v: np.tanh(v[:, 2]),
        **kwargs
    ):
        super().__init__(
            opacity_func=lambda p: vects_to_opacities(wave.func(p)),
            **kwargs
        )


# Scenes


class SnellsLaw(InteractiveScene):
    # index = 1.5
    index = 1.0

    def construct(self):
        # Setup key objects
        medium = FullScreenRectangle()
        medium.set_fill(BLUE, 0.3).set_stroke(width=0)
        medium.stretch(0.5, 1, about_edge=DOWN)

        hit_point = medium.get_top()
        angle_tracker = ValueTracker(40 * DEGREES)
        beam = self.get_beam(angle_tracker, hit_point)

        glow = GlowDot(color=YELLOW)

        self.add(medium)

        # Shine in
        anim_kw = dict(run_time=3, rate_func=linear)
        self.play(
            ShowCreation(beam, **anim_kw),
            UpdateFromFunc(glow, lambda m: m.move_to(beam.get_end()))
        )
        self.wait()

        # Add angle labels
        normal_line = Line(2 * UP, 2 * DOWN)
        normal_line.move_to(hit_point)
        normal_line.set_stroke(WHITE, 1)

        def get_theta1():
            return angle_tracker.get_value()

        def get_theta2():
            return np.arcsin(np.sin(get_theta1()) / self.index)

        arc_kw = dict(radius=0.8, stroke_width=1)
        arc1 = always_redraw(lambda: Arc(90 * DEGREES, get_theta1(), **arc_kw))
        arc2 = always_redraw(lambda: Arc(-90 * DEGREES, get_theta2(), **arc_kw))
        theta1_label = Tex(R"\theta_1")
        theta1_label.arc = arc1
        theta2_label = Tex(R"\theta_2")
        theta2_label.arc = arc2

        def update_theta_label(label):
            point = label.arc.pfp(0.2)
            vect = normalize(point - hit_point)
            label.set_height(min(0.4, max(1.0 * label.arc.get_width(), 1e-2)))
            label.next_to(point, vect, buff=SMALL_BUFF)

        theta1_label.add_updater(update_theta_label)
        theta2_label.add_updater(update_theta_label)

        ineq = Tex(R"\theta_1 > \theta_2")
        ineq.move_to(FRAME_WIDTH * RIGHT / 4 + 3 * UP)

        self.play(
            LaggedStart(
                ShowCreation(normal_line),
                ShowCreation(arc1),
                ShowCreation(arc2),
                Write(theta1_label),
                Write(theta2_label),
                lag_ratio=0.1,
                run_time=1
            ),
            FadeIn(ineq, UP)
        )
        self.wait()

        angle_labels = VGroup(normal_line, arc1, arc2, theta1_label, theta2_label)

        # Vary the angle for a bit
        self.remove(arc2, theta2_label, ineq)

        beam.set_stroke(opacity=1, width=2)
        for width in np.linspace(2, 10.0, 40):
            beam_copy = beam.copy()
            beam_copy.set_stroke(width=width, opacity=0.2 / (width)**1.4)
            self.add(beam_copy)

        for angle in [70 * DEGREES, -50 * DEGREES, 40 * DEGREES]:
            self.play(angle_tracker.animate.set_value(angle), run_time=5)

        # Tank analogy
        tank = SVGMobject("tank")
        tank.set_fill(WHITE)
        tank.rotate(-16 * DEGREES)
        tank.set_height(1)
        tank.move_to(beam.pfp(0.25))

        self.play(FadeOut(angle_labels))
        self.frame.set_height(4)
        self.play(
            tank.animate.move_to(beam.pfp(0.49)).set_anim_args(rate_func=linear, run_time=6),
            VFadeIn(tank),
            self.frame.animate.set_height(4).set_anim_args(time_span=(4, 6))
        )
        self.play(
            tank.animate.rotate(get_theta2() - get_theta1()).move_to(hit_point + 0.3 * UP + 0.1 * LEFT),
            rate_func=linear,
            run_time=1.5,
        )
        self.play(
            tank.animate.move_to(beam.pfp(0.65)).set_anim_args(rate_func=linear, run_time=8),
            VFadeOut(tank, time_span=(7, 8)),
            self.frame.animate.set_height(8).set_anim_args(time_span=(6, 8))
        )
        self.wait()

        # Write Snell's law
        law = Tex(R"{\sin(\theta_1) \over v_1} = {\sin(\theta_2) \over v_2}", font_size=52)
        title = TexText("Snell's Law", font_size=60)
        group = VGroup(title, law)
        group.arrange(DOWN, buff=0.75)
        group.to_corner(UR)

        self.play(FadeIn(angle_labels))
        self.play(
            Write(title),
            FlashUnder(title, run_time=2),
            FadeOut(ineq[">"]),
            Transform(ineq[R"\theta_1"], law[R"\theta_1"]),
            Transform(ineq[R"\theta_2"], law[R"\theta_2"]),
        )
        self.play(FadeIn(law))
        self.wait()
        tl1_copy = theta1_label.copy()
        tl2_copy = theta2_label.copy()
        self.play(LaggedStart(
            Transform(tl1_copy, law[R"\theta_1"]),
            Transform(tl2_copy, law[R"\theta_2"]),
            lag_ratio=0.5
        ))
        self.play(FadeIn(law, lag_ratio=0.1, run_time=2))
        self.remove(tl1_copy, tl2_copy)
        self.wait()

        # Vary the angle again
        for angle in [70 * DEGREES, 20 * DEGREES, 40 * DEGREES]:
            self.play(angle_tracker.animate.set_value(angle), run_time=3)

        # Show speeds
        in_beam = Line(beam.get_start(), hit_point)
        out_beam = Line(hit_point, beam.get_end())
        in_beam.set_length(8, about_point=hit_point)

        def get_dot_anim():
            dot = GlowDot()
            return Succession(
                MoveAlongPath(dot, in_beam, rate_func=linear, run_time=3),
                MoveAlongPath(dot, out_beam, rate_func=linear, run_time=3),
                remover=True,
            )

        self.play(LaggedStart(
            *(
                get_dot_anim()
                for x in range(50)
            ),
            lag_ratio=0.02,
            run_time=20,
        ))

    def get_beam(self, angle_tracker, hit_point, stroke_color=YELLOW, stroke_width=3):
        result = Line()
        result.set_stroke(stroke_color, stroke_width)

        def update_beam(beam):
            theta1 = angle_tracker.get_value()
            theta2 = np.arcsin(np.sin(theta1) / self.index)
            beam.set_points_as_corners([
                hit_point + 0.6 * FRAME_HEIGHT * rotate_vector(UP, theta1) / math.cos(theta1),
                hit_point,
                hit_point + 0.6 * FRAME_HEIGHT * rotate_vector(DOWN, theta2) / math.cos(theta2),
            ])

        result.add_updater(update_beam)
        return result


class WavesIntoAngledMedium(InteractiveScene):
    default_frame_orientation = (0, 0)
    interface_origin = ORIGIN
    interface_normal = DR
    prop_direction = RIGHT
    frequency = 0.5
    c = 1.0
    index = 2.0
    amplitude = 0.5

    def get_medium(
        self,
        width=10,
        height=30,
        depth=5,
        color=BLUE,
        opacity=0.2,
    ):
        medium = VCube(side_length=1.0)
        medium.set_shape(width, height, depth)
        medium.set_fill(color, opacity)
        medium.move_to(self.interface_origin, LEFT)
        medium.rotate(
            angle_of_vector(self.interface_normal),
            about_point=self.interface_origin
        )
        return medium

    def get_wave(self, **kwargs):
        config = dict(
            interface_origin=self.interface_origin,
            interface_normal=self.interface_normal,
            prop_direction=self.prop_direction,
            frequency=self.frequency,
            c=self.c,
            index=self.index,
            max_vect_len=np.inf,
            amplitude=self.amplitude,
            norm_to_opacity_func=lambda n: sigmoid(n),
        )
        config.update(kwargs)
        return WaveIntoMedium(**config)

    def get_wave_dots(self, wave, density=20, offset=0.2, color=WHITE, max_opacity=1.0, **kwargs):
        return WavesByOpacity(
            wave,
            density=density,
            vects_to_opacities=lambda v: max_opacity * np.tanh(v[:, 2] - offset * self.amplitude),
            color=color,
            **kwargs
        )


class TransitionToOverheadView(WavesIntoAngledMedium):
    interface_normal = RIGHT
    index = 2.0
    amplitude = 0.5

    def construct(self):
        # 1D case
        frame = self.frame
        medium = self.get_medium(opacity=0.3, height=8.0, depth=2.0)
        medium.remove(medium[-1])
        medium.sort(lambda p: -p[1] - p[2])
        medium.set_stroke(WHITE, 0.5, 0.5)
        medium.set_flat_stroke(False)
        wave_1d = self.get_wave(x_density=10, height=0.0)
        wave_1d.set_stroke(YELLOW)
        plane = NumberPlane(
            background_line_style=dict(
                stroke_color=BLUE_D,
                stroke_width=1,
                stroke_opacity=1,
            ),
        )
        plane.axes.set_stroke(width=1)
        plane.fade(0.5)

        self.add(plane, medium, wave_1d)
        frame.reorient(-37, 61, 0)

        self.play(
            frame.animate.reorient(0, 90).set_height(6),
            run_time=5,
        )
        self.wait(10)

        # Highlight wave length
        past_time = wave_1d.time
        wave_1d.suspend_updating()
        wave_len = 1.0
        mult = 2.0
        brace1 = Brace(Line(ORIGIN, mult * wave_len * RIGHT), UP)
        brace1.add(brace1.get_tex(Rf"\lambda = {wave_len}"))
        brace2 = Brace(Line(ORIGIN, 0.6 * mult * wave_len * RIGHT), UP)
        brace2.add(brace2.get_tex(Rf"\lambda = {0.6 * wave_len}"))

        for brace in [brace1, brace2]:
            brace.rotate(90 * DEGREES, RIGHT)
            brace.set_fill(border_width=0)

        brace1.next_to(3 * LEFT + 0.35 * OUT, OUT)
        brace2.next_to(1.85 * RIGHT + 0.35 * OUT, OUT)

        self.play(FadeIn(brace1, lag_ratio=0.1))
        self.wait()
        self.play(FadeTransform(brace1.copy(), brace2))
        self.wait()
        wave_1d.time = past_time
        wave_1d.resume_updating()
        self.play(
            FadeOut(brace1, RIGHT),
            FadeOut(brace2, 0.6 * RIGHT),
        )
        self.wait(3)

        # Transition to 2d
        wave_2d = self.get_wave(
            width=20.0, height=8.0,
            norm_to_opacity_func=lambda n: 0.5 * sigmoid(2 * n),
        )
        wave_2d.set_stroke(YELLOW)
        invisible_wave_2d = self.get_wave(
            width=20.0, height=12.0,
            norm_to_opacity_func=None,
            stroke_opacity=0,
        )
        wave_dots = self.get_wave_dots(
            invisible_wave_2d,
        )

        self.remove(wave_1d)
        wave_2d.time = wave_1d.time
        self.add(wave_2d)
        self.play(
            frame.animate.reorient(-10, 45).set_height(8).set_anim_args(
                run_time=7,
                time_span=(1, 7),
            ),
        )
        self.wait(2)
        self.wait(8)

        self.remove(wave_2d)
        invisible_wave_2d.time = wave_2d.time
        self.add(invisible_wave_2d, wave_dots)
        self.wait(2)
        self.play(
            frame.animate.reorient(0, 0),
            plane.animate.fade(0.5),
            medium.animate.set_opacity(0.2),
            run_time=4,

        )
        self.wait(8)

        # Show wave lengths again
        invisible_wave_2d.suspend_updating()
        VGroup(brace1, brace2).rotate(90 * DEGREES, LEFT)
        VGroup(brace1, brace2).set_backstroke(BLACK, 3)
        brace1.next_to(4 * LEFT, UP)
        brace2.next_to(2.45 * RIGHT, UP)
        braces = VGroup(brace1, brace2)

        self.play(
            FadeIn(braces),
        )
        self.wait()

        # Pan back down to 1d wave
        wave_1d.time = invisible_wave_2d.time
        wave_1d.update()
        wave_1d.clear_updaters()
        self.play(
            # FadeOut(wave_dots),
            FadeIn(wave_1d),
            frame.animate.reorient(0, 70),
            braces.animate.rotate(90 * DEGREES, RIGHT, about_point=ORIGIN).shift(0.3 * OUT),
            run_time=5,
        )
        self.wait()


class AngledMedium(WavesIntoAngledMedium):
    amplitude = 1.2
    frequency = 2.0 / 3.0

    def construct(self):
        # Add setup
        plane = NumberPlane()
        plane.fade(0.75)
        medium = self.get_medium()
        self.add(plane)
        self.add(medium)

        # Wave
        wave = self.get_wave()
        wave.set_opacity(0)
        wave_dots = self.get_wave_dots(wave, offset=0.5)
        self.add(wave, wave_dots)
        self.wait(30)


class AngledMediumSingleFront(AngledMedium):
    def get_wave_dots(self, wave, density=20, offset=0.2, color=RED, max_opacity=1.0, **kwargs):
        return super().get_wave_dots(wave, density, offset, color, max_opacity, **kwargs)


class AngledMediumAnnotations(InteractiveScene):
    def construct(self):
        # Set up the lane
        h_lines = Line(LEFT, RIGHT).replicate(3)
        h_lines.set_width(0.7 * FRAME_WIDTH)
        h_lines.arrange(DOWN, buff=0.5)
        h_lines.move_to(ORIGIN, RIGHT)

        index = WavesIntoAngledMedium.index
        angle = math.asin(math.sin(45 * DEGREES) / index)
        diag_lines = h_lines.copy()
        for line1, line2 in zip(h_lines, diag_lines):
            line2.move_to(line1.get_end(), LEFT)
            line2.rotate(-angle, about_point=line1.get_end())

        lane_points = [
            h_lines[0].get_left(),
            h_lines[0].get_right(),
            diag_lines[0].get_end(),
            diag_lines[2].get_end(),
            h_lines[2].get_right(),
            h_lines[2].get_left(),
        ]

        fade_rect = FullScreenFadeRectangle()
        fade_rect.scale(1.5)
        fade_rect.start_new_path(lane_points[-1])
        for point in lane_points:
            fade_rect.add_line_to(point)

        fade_rect.set_fill(BLACK, 0.8)

        beam = h_lines[1].copy()
        beam.add_line_to(diag_lines[1].get_end())
        beam.set_stroke(YELLOW, 1.5)
        beam.insert_n_curves(100)

        self.play(
            FadeIn(fade_rect),
        )
        self.play(
            ShowCreation(beam, run_time=3),
            VShowPassingFlash(beam.copy().set_stroke(width=5), run_time=3)
        )
        self.wait()


class LineGame(InteractiveScene):
    def construct(self):
        # Add line and medium
        interface = Line(6 * DL, 6 * UR)
        medium = Square().set_fill(BLUE, 0.2).set_stroke(width=0)
        medium.move_to(ORIGIN, LEFT)
        medium.rotate(-45 * DEGREES, about_point=ORIGIN)
        medium.scale(20, about_point=ORIGIN)
        self.add(medium, interface)

        # Prepare lines, with control
        large_spacing = 1.5
        small_spacing = 0.6 * large_spacing

        circ = Circle(radius=0.5)
        circ.set_fill(interpolate_color(BLUE_E, BLACK, 0.5), 1)
        circ.set_stroke(WHITE, 2)
        circ.next_to(ORIGIN, RIGHT, buff=1.0)
        dial = Vector(0.8 * RIGHT)
        dial.move_to(circ)

        top_lines = self.get_line_group(interface, UP, large_spacing)
        low_lines = always_redraw(lambda: self.get_line_group(
            interface, rotate_vector(dial.get_vector(), -PI / 2), small_spacing,
            line_color=GREEN,
            dot_color=YELLOW
        ))

        top_spacing_label = self.get_spacing_label(top_lines[1], R"\lambda_1")
        low_spacing_label = self.get_spacing_label(low_lines[1], R"\lambda_2")
        low_spacing_label.shift(UP)

        # Add top lines, then lower lines
        self.play(FadeIn(top_lines, lag_ratio=0.1))
        self.play(Write(top_spacing_label))
        self.wait()
        self.highlight_intersection_points(top_lines, PINK, LEFT)
        self.wait()

        # Reposition lower lines
        key_angle = -19.9 * DEGREES
        dial.rotate(key_angle)
        low_lines.update()
        low_spacing_label.rotate(key_angle, about_point=ORIGIN)
        top_lines.save_state()
        self.play(
            top_lines.animate.fade(0.8),
            FadeIn(low_lines, lag_ratio=0.1),
        )
        self.play(Write(low_spacing_label))
        self.wait()
        dial.set_stroke(opacity=0)
        self.play(
            Rotate(dial, -key_angle, run_time=2, remover=True),
            Rotate(low_spacing_label, -key_angle, run_time=2, about_point=ORIGIN),
        )
        dial.set_stroke(opacity=1)
        self.wait()
        self.highlight_intersection_points(low_lines, YELLOW, RIGHT)
        self.play(Restore(top_lines))

        # Rotate lower lines
        self.add(low_lines)
        self.play(
            FadeIn(circ),
            FadeIn(dial)
        )
        low_lines.resume_updating()
        for angle in [-15, -15, 10.1]:
            self.play(
                Rotate(dial, angle * DEGREES),
                Rotate(low_spacing_label, angle * DEGREES, about_point=ORIGIN),
                run_time=3
            )
            self.wait()
        low_lines.suspend_updating()
        self.highlight_intersection_points(low_lines, YELLOW)

        # Ask about angles
        theta1 = top_lines[1][0].get_angle() - interface.get_angle()
        theta2 = low_lines[1][0].get_angle() + TAU - (interface.get_angle() + PI)
        radius = 0.75
        arc1 = Arc(interface.get_angle(), theta1, radius=radius)
        arc2 = Arc(interface.get_angle() + PI, theta2, radius=radius)
        theta1_label = Tex(R"\theta_1", font_size=36)
        theta2_label = Tex(R"\theta_2", font_size=36)
        theta1_label.next_to(arc1.pfp(0.5), UP, SMALL_BUFF)
        theta2_label.next_to(arc2.pfp(0.75), DL, SMALL_BUFF)

        self.play(
            ShowCreation(arc1),
            Write(theta1_label),
            self.frame.animate.set_height(6.5).set_anim_args(run_time=2),
            FadeOut(circ),
            FadeOut(dial),
        )
        self.wait()
        self.play(
            ShowCreation(arc2),
            Write(theta2_label)
        )
        self.wait()

    def get_line_group(
        self,
        interface,
        line_direction=UP,
        spacing=1.0,
        n_lines=17,
        length=30,
        line_color=WHITE,
        line_stroke_width=2,
        dot_radius=0.05,
        dot_color=RED,
        dot_opacity=1,
    ):
        # Calculate dot spacing
        interface_vect = normalize(interface.get_vector())
        line_vect = normalize(line_direction)
        interface_center = interface.get_center()
        angle = angle_between_vectors(interface_vect, line_vect)
        dot_spacing = spacing / math.sin(angle)

        points = [
            interface_center + i * dot_spacing * interface_vect
            for i in range(-n_lines // 2, n_lines // 2 + 1)
        ]
        dots = VGroup(*(
            Dot(point, radius=dot_radius).set_fill(dot_color, dot_opacity)
            for point in points
        ))
        lines = VGroup(*(
            Line(
                point,
                point + length * line_vect,
                stroke_color=line_color,
                stroke_width=line_stroke_width,
            )
            for point in points
        ))
        result = VGroup(dots, lines)
        result.set_stroke(background=True)
        return result

    def get_spacing_label(self, lines, label_tex):
        n = len(lines)
        line1, line2 = lines[n // 2: (n // 2) + 2]
        x1 = line1.get_x()
        x2 = line2.get_x()
        dist = x2 - x1
        arrows = VGroup(
            Arrow(dist * LEFT / 4, dist * RIGHT / 2, buff=SMALL_BUFF),
            Arrow(dist * RIGHT / 4, dist * LEFT / 2, buff=SMALL_BUFF),
        )
        label = Tex(label_tex)
        label.set_max_width(0.9 * arrows.get_width())
        label.next_to(arrows, UP, MED_SMALL_BUFF)
        result = VGroup(arrows, label)
        result.move_to(VGroup(line1, line2))
        result.shift_onto_screen(buff=LARGE_BUFF)
        label.set_fill(border_width=0.5)
        return result

    def highlight_intersection_points(self, line_group, color, arrow_direction=LEFT):
        points = VGroup(*(p for p in line_group[0] if -4 < p.get_y() < 4))
        arrows = VGroup(*(
            Vector(arrow_direction).next_to(point, -arrow_direction, SMALL_BUFF)
            for point in points
        ))
        arrows.set_color(color)
        self.play(LaggedStartMap(GrowArrow, arrows, lag_ratio=0.25))
        self.play(
            LaggedStartMap(Indicate, line_group[0], scale_factor=2, color=color, lag_ratio=0.2),
            LaggedStartMap(FlashAround, line_group[0], buff=0.2, color=color, lag_ratio=0.2),
            run_time=4,
        )
        self.play(FadeOut(arrows))


class Prism(InteractiveScene):
    def construct(self):
        # Add flat
        flat_prism = Triangle()
        flat_prism.set_height(4)
        flat_prism.set_stroke(WHITE, 1)
        flat_prism.set_fill(BLUE, 0.2)

        prism = Prismify(flat_prism, depth=5)
        prism.set_fill(BLUE_D, 0.25, border_width=0)
        prism.set_stroke(WHITE, 0)
        prism.sort(lambda p: -p[2])
        prism.apply_depth_test()
        prism.deactivate_depth_test()
        prism.set_shading(0.5, 0.5, 0)

        verts = flat_prism.get_vertices()
        in_edge = Line(verts[0], verts[1])
        out_edge = Line(verts[0], verts[2])

        self.add(flat_prism)

        # Beams of light
        frame = self.frame
        self.camera.light_source.move_to((-10, -10, 10))

        left_side = 10 * LEFT
        in_beam = Line(left_side, in_edge.get_center())
        in_beam.set_stroke(WHITE, 3)

        def get_beams(light_in):
            return self.get_beams(
                min_index=1.3,
                # max_index=1.4,
                max_index=1.45,
                n_beams=200,
                in_beam=light_in,
                in_edge=in_edge,
                out_edge=out_edge,
            )

        beams = always_redraw(lambda: get_beams(in_beam))
        self.play(
            ShowCreation(beams, time_span=(0.5, 1.5), lag_ratio=0),
            ShowCreation(in_beam, time_span=(0, 1.0)),
            rate_func=linear
        )

        # Show x-ray
        x_ray = self.get_beams(0.8, 0.8, 1, in_beam, in_edge, out_edge)
        x_ray.set_stroke("#FF00D5", 8)

        self.add(x_ray, in_beam)
        self.play(ShowCreation(x_ray, run_time=2))
        self.wait()
        self.remove(x_ray)

        # Transition to 3d
        self.play(
            FadeOut(flat_prism),
            FadeIn(prism),
            frame.animate.reorient(-90, 35, 90).set_height(15).move_to(RIGHT + 4 * DOWN),
            run_time=4,
        )
        turn_animation_into_updater(
            ApplyMethod(frame.reorient, -90, 10, 90, run_time=15)
        )

        # for vect, alpha in zip([DOWN, 2 * UP, ORIGIN], [0.3, 0.3, 0.5]):
        for vect, alpha in zip([5 * DOWN, 3 * DOWN, 4 * DOWN], [0.3, 0.3, 0.5]):
            self.play(
                in_beam.animate.put_start_and_end_on(left_side + vect, in_edge.pfp(alpha)),
                # run_time=5
            )

        # Back to 2d
        frame.clear_updaters()
        self.play(
            frame.animate.reorient(0, 0, 0).set_height(8),
            in_beam.animate.put_start_and_end_on(left_side + 2 * DOWN, in_edge.get_center()),
            FadeOut(prism, time_span=(1, 2)),
            FadeIn(flat_prism, time_span=(1, 2)),
            run_time=2,
        )

        self.wait()

    def get_beams(self, min_index, max_index, n_beams, in_beam, in_edge: Line, out_edge: Line):
        alphas = np.linspace(0, 1, n_beams)**1.5
        indices = interpolate(min_index, max_index, alphas)

        normal1 = rotate_vector(normalize(in_edge.get_vector()), PI / 2)
        normal2 = rotate_vector(normalize(out_edge.get_vector()), PI / 2)
        in_point = in_beam.get_end()
        vect1 = normalize(in_beam.get_vector())

        theta1 = angle_between_vectors(normal1, vect1)
        theta2s = np.arcsin(np.sin(theta1) / indices)
        vect2s = np.array([
            rotate_vector(normal1, theta2)
            for theta2 in theta2s
        ])
        out_points = np.array([
            find_intersection(in_point, vect2, out_edge.get_start(), out_edge.get_vector())
            for vect2 in vect2s
        ])
        theta3s = np.array([
            angle_between_vectors(normal2, vect2)
            for vect2 in vect2s
        ])
        theta4s = np.arcsin(np.sin(theta3s) * indices)
        vect3s = np.array([
            rotate_vector(normal2, -theta4)
            for theta4 in theta4s
        ])

        beams = VGroup(*(
            VMobject().set_points_as_corners([
                in_point - 1.0 * FRAME_WIDTH * vect1,
                in_point,
                out_point,
                out_point + 3.0 * FRAME_WIDTH * vect3
            ])
            for out_point, vect3 in zip(out_points, vect3s)
        ))

        for alpha, beam in zip(np.linspace(0, 1, n_beams), beams):
            beam.set_stroke(get_spectral_color(alpha), 1, opacity=0.8)

        return beams
