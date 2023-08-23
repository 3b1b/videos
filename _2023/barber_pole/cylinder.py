from manim_imports_ext import *
from matplotlib import colormaps
from _2023.barber_pole.e_field import VectorField

spectral_cmap = colormaps.get_cmap("Spectral")


def get_spectral_color(alpha):
    return Color(rgb=spectral_cmap(alpha)[:3])


def get_spectral_colors(n_colors, lower_bound=0, upper_bound=1):
    return [
        get_spectral_color(alpha)
        for alpha in np.linspace(lower_bound, upper_bound, n_colors)
    ]


def get_axes_and_plane(
    x_range=(0, 24),
    y_range=(-1, 1),
    z_range=(-1, 1),
    x_unit=1,
    y_unit=2,
    z_unit=2,
    origin_point=5 * LEFT,
    axes_opacity=0.5,
    plane_line_style=dict(
        stroke_color=GREY_C,
        stroke_width=1,
        stroke_opacity=0.5
    ),
):
    axes = ThreeDAxes(
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        width=x_unit * (x_range[1] - x_range[0]),
        height=y_unit * (y_range[1] - y_range[0]),
        depth=z_unit * (z_range[1] - z_range[0]),
    )
    axes.shift(origin_point - axes.get_origin())
    axes.set_opacity(axes_opacity)
    axes.set_flat_stroke(False)
    plane = NumberPlane(
        axes.x_range, axes.y_range,
        width=axes.x_axis.get_length(),
        height=axes.y_axis.get_length(),
        background_line_style=plane_line_style,
        axis_config=dict(stroke_width=0),
    )
    plane.shift(axes.get_origin() - plane.get_origin())
    plane.set_flat_stroke(False)

    return axes, plane


def get_twist(wave_length, distance):
    # 350 is arbitrary. Change
    return distance / (wave_length / 350)**2


class OscillatingWave(VMobject):
    def __init__(
        self,
        axes,
        amplitude=0.75,
        wave_len=0.5,
        twist_rate=0.0,  # In rotations per unit distance
        speed=1.0,
        sample_resolution=0.005,
        stroke_width=2,
        offset=ORIGIN,
        color=None,
        **kwargs,
    ):
        self.axes = axes
        self.amplitude = amplitude
        self.wave_len = wave_len
        self.twist_rate = twist_rate
        self.speed = speed
        self.sample_resolution = sample_resolution
        self.offset = offset

        super().__init__(**kwargs)

        color = color or self.get_default_color(wave_len)
        self.set_stroke(color, stroke_width)
        self.set_flat_stroke(False)

        self.time = 0

        self.add_updater(lambda m, dt: m.update_points(dt))

    def update_points(self, dt):
        self.time += dt
        xs = np.arange(
            self.axes.x_axis.x_min,
            self.axes.x_axis.x_max,
            self.sample_resolution
        )
        self.set_points_as_corners(
            self.offset + self.wave_func(xs, self.time)
        )

    def wave_func(self, x, t):
        phase = TAU * t * self.speed / self.wave_len
        outs = self.amplitude * np.sin(TAU * x / self.wave_len - phase)
        twist_angles = x * self.twist_rate * TAU
        y = np.sin(twist_angles) * outs
        z = np.cos(twist_angles) * outs

        return self.axes.c2p(x, y, z)

    def get_default_color(self, wave_len):
        return get_spectral_color(inverse_interpolate(
            1.5, 0.5, wave_len
        ))


class OscillatingFieldWave(VectorField, OscillatingWave):
    def __init__(self, axes, *args, **kwargs):
        pass


class MeanWave(VMobject):
    def __init__(self, waves, **kwargs):
        self.waves = waves
        super().__init__(**kwargs)
        self.set_flat_stroke(False)
        self.add_updater(lambda m, dt: m.update_points(dt))

    def update_points(self, dt):
        for wave in self.waves:
            wave.update_points(dt)

        points = sum(wave.get_points() for wave in self.waves) / len(self.waves)
        self.set_points(points)


class SugarCylinder(Cylinder):
    def __init__(
        self, axes, camera,
        radius=0.5,
        color=BLUE_A,
        opacity=0.2,
        shading=(0.5, 0.5, 0.5),
        resolution=(51, 101),
    ):
        super().__init__(
            color=color,
            opacity=opacity,
            resolution=resolution,
            shading=shading,
        )
        self.set_width(2 * axes.z_axis.get_unit_size() * radius)
        self.set_depth(axes.x_axis.get_length(), stretch=True)
        self.rotate(PI / 2, UP)
        self.move_to(axes.get_origin(), LEFT)
        # self.set_shading(*shading)
        self.always_sort_to_camera(camera)


class Polarizer(VGroup):
    def __init__(
        self, axes,
        radius=1.0,
        angle=0,
        stroke_color=GREY_C,
        stroke_width=2,
        fill_color=GREY_C,
        fill_opacity=0.25,
        n_lines=14,
        line_opacity=0.2,
        arrow_stroke_color=WHITE,
        arrow_stroke_width=5,

    ):
        true_radius = radius * axes.z_axis.get_unit_size()
        circle = Circle(
            radius=true_radius,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
        )

        lines = VGroup(*(
            Line(circle.pfp(a), circle.pfp(1 - a))
            for a in np.arccos(np.linspace(1, -1, n_lines + 2)[1:-1]) / TAU
        ))
        lines.set_stroke(WHITE, 1, opacity=line_opacity)

        arrow = Vector(
            0.5 * true_radius * UP,
            stroke_color=arrow_stroke_color,
            stroke_width=arrow_stroke_width,
        )
        arrow.move_to(circle.get_top(), DOWN)

        super().__init__(
            circle, lines, arrow,
            # So the center works correctly
            VectorizedPoint(circle.get_bottom() + arrow.get_height() * DOWN),
        )
        self.set_flat_stroke(True)
        self.rotate(PI / 2, RIGHT)
        self.rotate(PI / 2, IN)
        self.rotate(angle, RIGHT)
        self.rotate(1 * DEGREES, UP)


class ProbagatingRings(VGroup):
    def __init__(
        self, line,
        n_rings=5,
        start_width=3,
        width_decay_rate=0.1,
        stroke_color=WHITE,
        growth_rate=2.0,
        spacing=0.2,
    ):
        ring = Circle(radius=1e-3, n_components=101)
        ring.set_stroke(stroke_color, start_width)
        ring.apply_matrix(z_to_vector(line.get_vector()))
        ring.move_to(line)
        ring.set_flat_stroke(False)

        super().__init__(*ring.replicate(n_rings))

        self.growth_rate = growth_rate
        self.spacing = spacing
        self.width_decay_rate = width_decay_rate
        self.start_width = start_width
        self.time = 0

        self.add_updater(lambda m, dt: self.update_rings(dt))

    def update_rings(self, dt):
        if dt == 0:
            return
        self.time += dt
        space = 0
        for ring in self.submobjects:
            effective_time = max(self.time - space, 0)
            target_radius = max(effective_time * self.growth_rate, 1e-3)
            ring.scale(target_radius / ring.get_radius())
            space += self.spacing
            ring.set_stroke(width=np.exp(-self.width_decay_rate * effective_time))
        return self


class TwistedRibbon(ParametricSurface):
    def __init__(
        self,
        axes,
        amplitude,
        twist_rate,
        start_point=(0, 0, 0),
        color=WHITE,
        opacity=0.4,
        resolution=(101, 11),
    ):
        super().__init__(
            lambda u, v: axes.c2p(
                u,
                v * amplitude * np.sin(TAU * twist_rate * u),
                v * amplitude * np.cos(TAU * twist_rate * u)
            ),
            u_range=axes.x_range[:2],
            v_range=(-1, 1),
            color=color,
            opacity=opacity,
            resolution=resolution,
            prefered_creation_axis=0,
        )
        self.shift(axes.c2p(*start_point) - axes.get_origin())


# Scenes


class SimpleLightBeam(InteractiveScene):
    default_frame_orientation = (-33, 85)
    axes_config = dict()
    amplitude = 0.5
    wave_len = 1.0
    speed = 1.0

    def construct(self):
        axes, plane = get_axes_and_plane(**self.axes_config)
        self.add(axes, plane)

        # Introduce wave
        wave = OscillatingWave(
            axes,
            amplitude=self.amplitude,
            wave_len=self.wave_len,
            speed=self.speed
        )

        def update_wave(wave):
            st = self.time * self.speed  # Suppressor threshold
            points = wave.get_points().copy()
            xs = axes.x_axis.p2n(points)
            suppressors = np.clip(smooth(st - xs), 0, 1)
            points[:, 1] *= suppressors
            points[:, 2] *= suppressors
            wave.set_points(points)
            return wave

        wave.add_updater(update_wave)

        self.add(wave)
        self.play(
            self.frame.animate.reorient(-98, 77, 0).move_to([-0.87, 0.9, -0.43]),
            run_time=8,
        )
        self.play(
            self.frame.animate.reorient(-10, 77, 0).move_to([-0.87, 0.9, -0.43]),
            run_time=4
        )
        self.wait(3)

        # Label directions
        z_label = Tex("z")
        z_label.rotate(PI / 2, RIGHT)
        z_label.next_to(axes.z_axis, OUT)

        y_label = Tex("y")
        y_label.rotate(PI / 2, RIGHT)
        y_label.next_to(axes.y_axis, UP + OUT)

        x_label = VGroup(
            TexText("$x$-direction"),
            Vector(RIGHT, stroke_color=WHITE),
        )
        x_label.arrange(RIGHT)
        x_label.set_flat_stroke(False)
        x_label.rotate(PI / 2, RIGHT)
        x_label.next_to(z_label, RIGHT, buff=2.0)
        x_label.match_z(axes.c2p(0, 0, 0.75))

        self.play(
            FadeIn(z_label, 0.5 * OUT),
            FadeIn(y_label, 0.5 * UP),
        )
        self.wait(3)
        self.play(
            Write(x_label[0]),
            GrowArrow(x_label[1]),
        )
        self.play(
            self.frame.animate.reorient(-41, 77, 0).move_to([-0.87, 0.9, -0.43]),
            run_time=12,
        )
        self.wait(6)


class TwistingLightBeam(SimpleLightBeam):
    amplitude = 0.5
    wave_len = 1.0
    twist_rate = 1 / 36
    speed = 1.0

    def construct(self):
        # Axes
        axes, plane = get_axes_and_plane(**self.axes_config)
        self.add(axes, plane)

        # Add wave
        wave = OscillatingWave(
            axes,
            amplitude=self.amplitude,
            wave_len=self.wave_len,
            speed=self.speed
        )

        twist_rate_tracker = ValueTracker(0)

        def update_twist_rate(wave):
            wave.twist_rate = twist_rate_tracker.get_value()
            return wave

        wave.add_updater(update_twist_rate)

        cylinder = SugarCylinder(axes, self.camera, radius=self.amplitude)

        self.add(wave)
        self.frame.reorient(-41, 77, 0).move_to([-0.87, 0.9, -0.43])
        self.wait(4)
        cylinder.save_state()
        cylinder.stretch(0, 0, about_edge=RIGHT)
        self.play(
            Restore(cylinder, time_span=(0, 3)),
            twist_rate_tracker.animate.set_value(self.twist_rate).set_anim_args(time_span=(0, 3)),
            self.frame.animate.reorient(-47, 80, 0).move_to([0.06, -0.05, 0.05]).set_height(8.84),
            run_time=6,
        )
        self.wait(2)
        self.play(
            self.frame.animate.reorient(-130, 77, 0).move_to([0.35, -0.36, 0.05]),
            run_time=10,
        )
        self.wait()
        self.play(
            self.frame.animate.reorient(-57, 77, 0).move_to([0.35, -0.36, 0.05]),
            run_time=10,
        )

        # Add rod with oscillating ball
        x_tracker, plane, rod, ball, x_label = self.get_slice_group(axes, wave)
        plane.save_state()
        plane.stretch(0, 2, about_edge=OUT)

        frame_anim = self.frame.animate.reorient(-45, 79, 0)
        frame_anim.move_to([0.63, 0.47, -0.25])
        frame_anim.set_height(10.51)
        frame_anim.set_anim_args(run_time=3)

        self.add(rod, ball, plane, cylinder)
        self.play(
            frame_anim,
            FadeIn(rod),
            Restore(plane),
            FadeIn(x_label),
            UpdateFromAlphaFunc(wave,
                lambda m, a: m.set_stroke(
                    width=interpolate(2, 1, a),
                    opacity=interpolate(1, 0.5, a),
                ),
                run_time=3,
                time_span=(0, 2),
            ),
            UpdateFromAlphaFunc(ball, lambda m, a: m.set_opacity(a)),
        )
        self.wait(9)

        # Show twist down the line of the cylinder
        x_tracker.set_value(0)
        x_tracker.clear_updaters()
        x_tracker.add_updater(lambda m, dt: m.increment_value(0.5 * dt))
        self.add(x_tracker)
        self.wait(5)
        self.play(
            self.frame.animate.reorient(-87, 88, 0).move_to([0.63, 0.47, -0.25]).set_height(10.51),
            run_time=5,
        )
        self.wait(3)
        self.play(
            self.frame.animate.reorient(-43, 78, 0).move_to([0.63, 0.47, -0.25]).set_height(10.51),
            run_time=5
        )
        self.play(
            self.frame.animate.reorient(-34, 80, 0).move_to([1.61, -0.05, 0.3]).set_height(10.30),
            run_time=15,
        )
        self.wait(10)

    def get_slice_group(self, axes, wave):
        x_tracker = ValueTracker(0)
        get_x = x_tracker.get_value

        rod = self.get_polarization_rod(axes, wave, get_x)
        ball = self.get_wave_ball(wave, get_x)
        plane = self.get_slice_plane(axes, get_x)
        x_label = self.get_plane_label(axes, plane)

        return Group(x_tracker, plane, rod, ball, x_label)

    def get_polarization_rod(self, axes, wave, get_x, stroke_color=None, length_mult=2.0, stroke_width=3):
        rod = Line(IN, OUT)
        rod.set_stroke(
            color=stroke_color or wave.get_stroke_color(),
            width=stroke_width,
        )
        rod.set_flat_stroke(False)
        wave_z = axes.z_axis.p2n(wave.get_center())
        wave_y = axes.y_axis.p2n(wave.get_center())

        def update_rod(rod):
            x = get_x()
            rod.put_start_and_end_on(
                axes.c2p(x, wave_y, wave_z - length_mult * wave.amplitude),
                axes.c2p(x, wave_y, wave_z + length_mult * wave.amplitude),
            )
            rod.rotate(-TAU * wave.twist_rate * x, RIGHT)
            return rod

        rod.add_updater(update_rod)
        return rod

    def get_wave_ball(self, wave, get_x, radius=0.075):
        ball = TrueDot(radius=radius)
        ball.make_3d()
        ball.set_color(wave.get_color())

        def update_ball(ball):
            ball.move_to(wave.offset + wave.wave_func(get_x(), wave.time))
            return ball

        ball.add_updater(update_ball)
        return ball

    def get_slice_plane(self, axes, get_x):
        plane = Square(side_length=axes.z_axis.get_length())
        plane.set_fill(BLUE, 0.25)
        plane.set_stroke(width=0)
        circle = Circle(
            radius=axes.z_axis.get_unit_size() * self.amplitude,
            n_components=100,
        )
        circle.set_flat_stroke(False)
        circle.set_stroke(BLACK, 1)
        plane.add(circle)
        plane.rotate(PI / 2, UP)
        plane.add_updater(lambda m: m.move_to(axes.c2p(get_x(), 0, 0)))
        return plane

    def get_plane_label(self, axes, plane, font_size=24, color=GREY_B):
        x_label = Tex("x = 0.00", font_size=font_size)
        x_label.set_fill(color)
        x_label.value_mob = x_label.make_number_changable("0.00")
        x_label.rotate(PI / 2, RIGHT)
        x_label.rotate(PI / 2, IN)

        def update_x_label(x_label):
            x_value = x_label.value_mob
            x_value.set_value(axes.x_axis.p2n(plane.get_center()))
            x_value.rotate(PI / 2, RIGHT)
            x_value.rotate(PI / 2, IN)
            x_value.next_to(x_label[1], DOWN, SMALL_BUFF)
            x_label.next_to(plane, OUT)
            return x_label

        x_label.add_updater(update_x_label)
        return x_label


class TwistingBlueLightBeam(TwistingLightBeam):
    wave_len = 0.5
    twist_rate = 1 / 24


class TwistingRedLightBeam(TwistingLightBeam):
    wave_len = 1.5
    twist_rate = 1 / 48


class TwistingWithinCylinder(InteractiveScene):
    default_frame_orientation = (-40, 80)
    n_lines = 11
    pause_down_the_tube = True

    def construct(self):
        # Reference objects
        frame = self.frame
        axes, plane = get_axes_and_plane(
            x_range=(0, 8),
            y_range=(-2, 2),
            z_range=(-2, 2),
            y_unit=1,
            z_unit=1,
            origin_point=3 * LEFT
        )
        cylinder = SugarCylinder(axes, self.camera, radius=0.5)

        self.add(plane, axes)
        self.add(cylinder)

        # Light lines
        lines = VGroup()
        colors = get_spectral_colors(self.n_lines)
        for color in colors:
            line = Line(ORIGIN, 0.95 * OUT)
            line.set_flat_stroke(False)
            line.set_stroke(color, 2)
            lines.add(line)

        lines.arrange(DOWN, buff=0.1)
        lines.move_to(cylinder.get_left())

        # Add polarizer to the start
        light = GlowDot(color=WHITE, radius=3)
        light.move_to(axes.c2p(-3, 0, 0))
        polarizer = Polarizer(axes, radius=0.6)
        polarizer.move_to(axes.c2p(-1, 0, 0))
        polarizer_label = Text("Linear polarizer", font_size=36)
        polarizer_label.rotate(PI / 2, RIGHT)
        polarizer_label.rotate(PI / 2, IN)
        polarizer_label.next_to(polarizer, OUT)
        frame.reorient(-153, 79, 0)
        frame.shift(1.0 * IN)

        self.play(GrowFromCenter(light))
        self.play(
            Write(polarizer_label),
            FadeIn(polarizer, IN),
            light.animate.shift(LEFT).set_anim_args(time_span=(1, 3)),
            self.frame.animate.reorient(-104, 77, 0).center().set_anim_args(run_time=3),
        )

        # Many waves
        waves = VGroup(*(
            OscillatingWave(
                axes,
                amplitude=0.3,
                wave_len=wave_len,
                color=line.get_color(),
                offset=LEFT + line.get_y() * UP
            )
            for line, wave_len in zip(
                lines,
                np.linspace(2.0, 0.5, len(lines))
            )
        ))
        waves.set_stroke(width=1)
        superposition = MeanWave(waves)
        superposition.set_stroke(WHITE, 2)
        superposition.add_updater(lambda m: m.stretch(4, 2, about_point=ORIGIN))

        self.play(
            VFadeIn(superposition),
            FadeOut(cylinder),
        )
        self.play(
            self.frame.animate.reorient(-66, 76, 0),
            light.animate.scale(0.25),
            run_time=10,
        )
        self.remove(superposition)
        superposition.suspend_updating()
        self.play(*(
            TransformFromCopy(superposition, wave, run_time=2)
            for wave in waves
        ))

        # Go through individual waves
        self.add(waves)
        for wave1 in waves:
            anims = []
            for wave2 in waves:
                wave2.current_opacity = wave2.get_stroke_opacity()
                if wave1 is wave2:
                    wave2.target_opacity = 1
                else:
                    wave2.target_opacity = 0.1
                anims.append(UpdateFromAlphaFunc(wave2, lambda m, a: m.set_stroke(
                    opacity=interpolate(m.current_opacity, m.target_opacity, a)
                )))
            self.play(*anims, run_time=0.5)
            self.wait()

        for wave in waves:
            wave.current_opacity = wave.get_stroke_opacity()
            wave.target_opacity = 1

        self.play(
            *(
                UpdateFromAlphaFunc(wave, lambda m, a: m.set_stroke(
                    opacity=interpolate(m.current_opacity, m.target_opacity, a)
                ))
                for wave in waves
            ),
            frame.animate.reorient(-55, 76, 0).move_to([-0.09, 0.13, -0.17]).set_height(7.5),
            run_time=3
        )

        # Introduce lines
        white_lines = lines.copy()
        white_lines.set_stroke(WHITE)
        white_lines.arrange(UP, buff=0)
        white_lines.move_to(axes.get_origin())

        plane = Square(side_length=2 * axes.z_axis.get_unit_size())
        plane.set_fill(WHITE, 0.25)
        plane.set_stroke(width=0)
        plane.add(
            Circle(radius=0.5 * cylinder.get_depth(), n_components=100).set_stroke(BLACK, 1)
        )
        plane.rotate(PI / 2, UP)
        plane.move_to(axes.get_origin())
        plane.save_state()
        plane.stretch(0, 2, about_edge=UP)

        self.play(
            ReplacementTransform(waves, lines, lag_ratio=0.1, run_time=3),
            frame.animate.reorient(-61, 83, 0).move_to([0.03, -0.16, -0.28]).set_height(7).set_anim_args(run_time=2),
            Restore(plane),
            FadeIn(cylinder),
        )
        self.add(axes, lines)
        self.wait()
        self.play(
            lines.animate.arrange(UP, buff=0).move_to(axes.get_origin()),
            FadeIn(white_lines),
            FadeOut(polarizer),
            FadeOut(polarizer_label),
            FadeOut(light),
        )
        self.wait()

        # Enable lines to twist through the tube
        line_start, line_end = white_lines[0].get_start_and_end()

        distance_tracker = ValueTracker(0)

        wave_lengths = np.linspace(700, 400, self.n_lines)  # Is this right?
        for line, wave_length in zip(lines, wave_lengths):
            line.wave_length = wave_length

        def update_lines(lines):
            dist = distance_tracker.get_value()
            for line in lines:
                line.set_points_as_corners([line_start, line_end])
                line.rotate(get_twist(line.wave_length, dist), RIGHT)
                line.move_to(axes.c2p(dist, 0, 0))
                line.set_gloss(3 * np.exp(-3 * dist))

        lines.add_updater(update_lines)

        # Add wave trails
        trails = VGroup(*(
            self.get_wave_trail(line)
            for line in lines
        ))
        continuous_trails = Group(*(
            self.get_continuous_wave_trail(axes, line)
            for line in lines
        ))
        for trail in continuous_trails:
            x_unit = axes.x_axis.get_unit_size()
            x0 = axes.get_origin()[0]
            trail.add_updater(
                lambda t: t.set_clip_plane(LEFT, distance_tracker.get_value() + x0)
            )
        self.add(trails, lines, white_lines)

        # Move light beams down the pole
        self.add(distance_tracker)
        distance_tracker.set_value(0)
        plane.add_updater(lambda m: m.match_x(lines))
        self.remove(white_lines)

        if self.pause_down_the_tube:
            # Test
            self.play(
                self.frame.animate.reorient(-42, 76, 0).move_to([0.03, -0.16, -0.28]).set_height(7.00),
                distance_tracker.animate.set_value(4),
                run_time=6,
                rate_func=linear,
            )
            trails.suspend_updating()
            self.play(
                self.frame.animate.reorient(67, 77, 0).move_to([-0.31, 0.48, -0.33]).set_height(4.05),
                run_time=3,
            )
            self.wait(2)
            trails.resume_updating()
            self.play(
                distance_tracker.animate.set_value(axes.x_axis.x_max),
                self.frame.animate.reorient(-36, 79, 0).move_to([-0.07, 0.06, 0.06]).set_height(7.42),
                run_time=6,
                rate_func=linear,
            )
            trails.clear_updaters()
            self.play(
                self.frame.animate.reorient(-10, 77, 0).move_to([0.42, -0.16, -0.03]).set_height(5.20),
                trails.animate.set_stroke(width=3, opacity=0.25).set_anim_args(time_span=(0, 3)),
                run_time=10,
            )
        else:
            self.play(
                self.frame.animate.reorient(-63, 84, 0).move_to([1.04, -1.86, 0.55]).set_height(1.39),
                distance_tracker.animate.set_value(axes.x_axis.x_max),
                run_time=15,
                rate_func=linear,
            )
            trails.clear_updaters()
            lines.clear_updaters()

            self.play(
                self.frame.animate.reorient(64, 81, 0).move_to([3.15, 0.46, -0.03]).set_height(5),
                run_time=3,
            )
            self.wait()

        # Add polarizer at the end
        end_polarizer = Polarizer(axes, radius=0.6)
        end_polarizer.next_to(lines, RIGHT, buff=0.5)

        self.play(
            FadeIn(end_polarizer, OUT),
            FadeOut(plane),
            self.frame.animate.reorient(54, 78, 0).move_to([3.15, 0.46, -0.03]).set_height(5.00).set_anim_args(run_time=4)
        )
        end_polarizer.save_state()
        self.play(end_polarizer.animate.fade(0.8))

        # Show a few different frequencies
        vertical_components = VGroup()
        for index in range(len(lines)):
            lines.generate_target()
            trails.generate_target()
            lines.target.set_opacity(0)
            trails.target.set_opacity(0)
            lines.target[index].set_opacity(1)
            trails.target[index].set_opacity(0.2)

            line = lines[index]
            x = float(axes.x_axis.p2n(cylinder.get_right()))
            vcomp = line.copy().set_opacity(1)
            vcomp.stretch(0, 1)
            vcomp.move_to(axes.c2p(x, -2 + index / len(lines), 0))
            z = float(axes.z_axis.p2n(vcomp.get_zenith()))
            y_min, y_max = axes.y_range[:2]
            globals().update(locals())
            dashed_lines = VGroup(*(
                DashedLine(axes.c2p(x, y_min, u * z), axes.c2p(x, y_max, u * z), dash_length=0.02)
                for u in [1, -1]
            ))
            dashed_lines.set_stroke(WHITE, 0.5)
            dashed_lines.set_flat_stroke(False)

            self.play(
                MoveToTarget(lines),
                MoveToTarget(trails),
                FadeIn(dashed_lines),
                FadeIn(vcomp),
                self.frame.animate.reorient(77, 87, 0).move_to([3.1, 0.4, 0]).set_height(5),
            )
            self.play(
                FadeOut(dashed_lines),
            )

            vertical_components.add(vcomp)

        self.play(
            lines.animate.set_opacity(1),
            trails.animate.set_opacity(0.05),
        )

        # Final color
        def get_final_color():
            rgbs = np.array([
                line.data["stroke_rgba"][0, :3]
                for line in lines
            ])
            depths = np.array([v_line.get_depth() for v_line in vertical_components])
            alphas = depths / depths.sum()
            rgb = ((rgbs**0.5) * alphas[:, np.newaxis]).sum(0)**2.0
            return rgb_to_color(rgb)

        new_color = get_final_color()
        new_lines = vertical_components.copy()
        for line in new_lines:
            line.set_depth(cylinder.get_depth())
            line.set_stroke(new_color, 4)
            line.next_to(end_polarizer, RIGHT, buff=0.5)

        self.play(
            Restore(end_polarizer),
            TransformFromCopy(vertical_components, new_lines),
            self.frame.animate.reorient(43, 73, 0).move_to([3.3, 0.66, -0.38]).set_height(5.68),
            run_time=4,
        )
        self.play(
            self.frame.animate.reorient(45, 72, 0).move_to([3.17, 0.4, -0.56]),
            run_time=8,
        )

        # Twist the tube
        result_line = new_lines[0]
        self.remove(new_lines)
        self.add(result_line)
        result_line.add_updater(lambda l: l.set_stroke(get_final_color()))

        line_group = VGroup(trails, lines)

        p1, p2 = axes.c2p(0, 1, 0), axes.c2p(0, -1, 0)
        twist_arrows = VGroup(
            Arrow(p1, p2, path_arc=PI),
            Arrow(p2, p1, path_arc=PI),
        )
        twist_arrows.rotate(PI / 2, UP, about_point=axes.get_origin())
        twist_arrows.apply_depth_test()
        self.add(twist_arrows, cylinder, line_group, vertical_components)

        for v_comp, line in zip(vertical_components, lines):
            v_comp.line = line
            v_comp.add_updater(lambda m: m.match_depth(m.line))

        self.play(
            ShowCreation(twist_arrows, lag_ratio=0),
            Rotate(line_group, PI, axis=RIGHT, run_time=12, rate_func=linear)
        )

    def get_wave_trail(self, line, spacing=0.05, opacity=0.05):
        trail = VGroup()
        trail.time = 1

        def update_trail(trail, dt):
            trail.time += dt
            if trail.time > spacing:
                trail.time = 0
                trail.add(line.copy().set_opacity(opacity).set_shading(0, 0, 0))

        trail.add_updater(update_trail)
        return trail

    def get_continuous_wave_trail(self, axes, line, opacity=0.4):
        return TwistedRibbon(
            axes,
            amplitude=0.5 * line.get_length(),
            twist_rate=get_twist(line.wave_length, TAU),
            color=line.get_color(),
            opacity=opacity,
        )


class InducedWiggleInCylinder(TwistingLightBeam):
    random_seed = 2
    cylinder_radius = 0.5
    wave_config = dict(
        amplitude=0.15,
        wave_len=0.5,
        color=get_spectral_color(0.1),
        speed=1.0,
        twist_rate=1 / 24
    )

    def construct(self):
        # Setup
        frame = self.frame
        frame.reorient(-51, 80, 0).move_to(0.5 * IN).set_height(9)

        axes, plane = get_axes_and_plane(**self.axes_config)
        cylinder = SugarCylinder(axes, self.camera, radius=self.cylinder_radius)
        wave = OscillatingWave(axes, **self.wave_config)
        x_tracker, plane, rod, ball, x_label = slice_group = self.get_slice_group(axes, wave)
        rod = self.get_polarization_rod(axes, wave, x_tracker.get_value, length_mult=5.0)

        axes_labels = Tex("yz", font_size=30)
        axes_labels.rotate(89 * DEGREES, RIGHT)
        axes_labels[0].next_to(axes.y_axis.get_top(), OUT, SMALL_BUFF)
        axes_labels[1].next_to(axes.z_axis.get_zenith(), OUT, SMALL_BUFF)
        axes.add(axes_labels)

        light = GlowDot(radius=4, color=RED)
        light.move_to(axes.c2p(-3, 0, 0))

        polarizer = Polarizer(axes, radius=0.5)
        polarizer.move_to(axes.c2p(-1, 0, 0))

        self.add(axes, cylinder, polarizer, light)

        # Bounces of various points
        randy = self.get_observer(axes.c2p(8, -3, -0.5))
        self.play(
            self.frame.animate.reorient(-86, 70, 0).move_to([1.01, -2.98, -0.79]).set_height(11.33),
            FadeIn(randy, time_span=(0, 1)),
            run_time=2,
        )
        max_y = 0.5 * self.cylinder_radius
        for _ in range(10):
            point = axes.c2p(
                random.uniform(axes.x_axis.x_min, axes.x_axis.x_max),
                random.uniform(-max_y, -max_y),
                random.uniform(-max_y, -max_y),
            )
            dot = TrueDot(point, radius=0.05)
            dot.make_3d()
            line = VMobject().set_points_as_corners(
                [light.get_center(), point, randy.eyes.get_top()]
            )
            line.set_stroke(RED, 2)
            line.set_flat_stroke(False)
            self.add(dot, cylinder)
            self.play(ShowCreation(line))
            self.play(FadeOut(line), FadeOut(dot))

        # Show slice such that wiggling is in z direction
        x_tracker.set_value(0)
        self.add(wave, cylinder)
        self.play(
            self.frame.animate.reorient(-73, 78, 0).move_to([0.8, -2.22, -0.83]).set_height(10.64),
            light.animate.scale(0.5),
            polarizer.animate.fade(0.5),
            VFadeIn(wave),
        )
        self.wait(4)
        self.add(wave, cylinder)
        self.play(
            FadeIn(plane),
            FadeIn(x_label),
            FadeIn(rod),
        )
        self.play(
            x_tracker.animate.set_value(12),
            run_time=12,
            rate_func=linear,
        )
        self.add(rod, ball, wave, cylinder)

        # Show observer
        line_of_sight = DashedLine(randy.eyes.get_top(), rod.get_center())
        line_of_sight.set_stroke(WHITE, 2)
        line_of_sight.set_flat_stroke(False)

        self.play(
            self.frame.animate.reorient(-60, 79, 0).move_to([0.73, -0.59, -0.39]).set_height(9.63),
            ShowCreation(line_of_sight, time_span=(3, 4)),
            run_time=8,
        )
        self.wait(2)

        # Show propagating rings
        self.show_propagation(rod)

        # Move to a less favorable spot
        new_line_of_sight = DashedLine(randy.eyes.get_top(), axes.c2p(6, 0, 0))
        new_line_of_sight.match_style(line_of_sight)
        new_line_of_sight.set_flat_stroke(False)

        self.remove(ball)
        self.play(
            x_tracker.animate.set_value(6),
            FadeOut(line_of_sight, time_span=(0, 0.5)),
            run_time=4,
        )
        self.add(ball, wave, cylinder, plane)
        self.play(ShowCreation(new_line_of_sight))
        self.wait(4)

        # New propagations
        self.show_propagation(rod)

        # Show ribbon
        ribbon = TwistedRibbon(
            axes,
            amplitude=wave.amplitude,
            twist_rate=wave.twist_rate,
            color=wave.get_color(),
        )

        self.add(ribbon, cylinder)
        self.play(ShowCreation(ribbon, run_time=5))
        self.wait()
        self.play(
            self.frame.animate.reorient(8, 77, 0).move_to([2.01, -0.91, -0.58]).set_height(5.55),
            FadeOut(randy),
            run_time=2,
        )
        self.wait(4)
        self.play(
            self.frame.animate.reorient(-25, 76, 0).move_to([4.22, -1.19, -0.5]),
            x_tracker.animate.set_value(12),
            FadeOut(new_line_of_sight, time_span=(0, 0.5)),
            run_time=3,
        )
        self.wait(4)
        self.play(
            self.frame.animate.reorient(-61, 78, 0).move_to([0.7, 0.05, -0.69]).set_height(9.68),
            FadeIn(randy),
            run_time=3,
        )
        self.play(
            LaggedStartMap(FadeOut, Group(
                line_of_sight, plane, rod, ball, x_label
            ))
        )

        # Show multiple waves
        n_waves = 11
        amp = 0.03
        zs = np.linspace(0.5 - amp, -0.5 + amp, n_waves)
        small_wave_config = dict(self.wave_config)
        small_wave_config["amplitude"] = amp

        waves = VGroup(*(
            OscillatingWave(
                axes,
                offset=axes.c2p(0, 0, z)[2] * OUT,
                **small_wave_config
            )
            for z in zs
        ))

        self.remove(ribbon)
        self.play(
            FadeOut(wave),
            VFadeIn(waves),
        )
        self.wait(4)

        # Focus on various x_slices
        x_tracker.set_value(0)
        rods = VGroup(*(
            self.get_polarization_rod(
                axes, lil_wave, x_tracker.get_value,
                length_mult=1,
                stroke_width=2,
            )
            for lil_wave in waves
        ))
        balls = Group(*(
            self.get_wave_ball(lil_wave, x_tracker.get_value, radius=0.025)
            for lil_wave in waves
        ))
        sf = 1.2 * axes.z_axis.get_unit_size() / plane.get_height()
        plane.scale(sf)
        plane[0].scale(1.0 / sf)

        plane.update()
        x_label.update()
        self.add(plane, rods, balls, cylinder, x_label)
        self.play(
            self.frame.animate.reorient(-90, 83, 0).move_to([0.17, -0.37, -0.63]).set_height(7.35).set_anim_args(run_time=3),
            FadeOut(light),
            FadeOut(polarizer),
            FadeIn(plane),
            FadeIn(rods),
            FadeIn(x_label),
            waves.animate.set_stroke(width=0.5, opacity=0.5).set_anim_args(time_span=(1, 2), suspend_mobject_updating=False),
            cylinder.animate.set_opacity(0.05).set_anim_args(time_span=(1, 2))
        )
        self.wait(4)
        self.play(
            self.frame.animate.reorient(-91, 90, 0).move_to([-0.01, -1.39, 0.21]).set_height(3.70),
            x_tracker.animate.set_value(5).set_anim_args(rate_func=linear),
            run_time=12,
        )
        self.wait(4)

        # Show lines of sight
        lines_of_sight = VGroup(*(
            self.get_line_of_sign(rod, randy, stroke_width=0.5)
            for rod in rods
        ))

        self.play(ShowCreation(lines_of_sight[0]))
        self.show_propagation(rods[0])
        for line1, line2 in zip(lines_of_sight, lines_of_sight[1:]):
            self.play(FadeOut(line1), FadeIn(line2), run_time=0.25)
            self.wait(0.25)
        self.wait(4)
        self.play(FadeIn(lines_of_sight[:-1]))
        self.add(lines_of_sight)

        # Move closer and farther
        self.play(
            randy.animate.shift(3.5 * UP + 0.5 * IN),
            run_time=2,
        )
        self.wait(8)
        self.play(
            self.frame.animate.reorient(-91, 89, 0).move_to([-0.05, -3.75, 0.07]).set_height(8.92),
            randy.animate.shift(10 * DOWN),
            run_time=2,
        )
        self.wait(8)

    def show_propagation(self, rod, run_time=10):
        rings = ProbagatingRings(rod, start_width=5)
        self.add(rings)
        self.wait(run_time)
        self.play(VFadeOut(rings))

    def get_observer(self, location=ORIGIN):
        randy = Randolph(mode="pondering")
        randy.look(RIGHT)
        randy.rotate(PI / 2, RIGHT)
        randy.rotate(PI / 2, OUT)
        randy.move_to(location)
        return randy

    def get_line_of_sign(self, rod, observer, stroke_color=WHITE, stroke_width=1):
        line = Line(ORIGIN, 5 * RIGHT)
        line.set_stroke(stroke_color, stroke_width)
        line.add_updater(lambda l: l.put_start_and_end_on(
            observer.eyes.get_top(), rod.get_center()
        ))
        line.set_flat_stroke(False)
        return line
