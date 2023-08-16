from manim_imports_ext import *
from matplotlib import colormaps

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
        start_point=(0, 0, 0),
        color=None,
        **kwargs,
    ):
        self.axes = axes
        self.amplitude = amplitude
        self.wave_len = wave_len
        self.twist_rate = twist_rate
        self.speed = speed
        self.sample_resolution = sample_resolution
        self.start_point = start_point

        super().__init__(**kwargs)

        color = color or self.get_default_color(wave_len)
        self.set_stroke(color, stroke_width)
        self.set_flat_stroke(False)

        self.time = 0

        self.add_updater(lambda m, dt: m.update_points(dt))

    def update_points(self, dt):
        self.time += dt

        axes = self.axes
        x_min = axes.x_axis.x_min
        x_max = axes.x_axis.x_max
        xs = np.arange(x_min, x_max, self.sample_resolution)
        phase = TAU * self.time * self.speed / self.wave_len
        outs = self.amplitude * np.sin(TAU * xs / self.wave_len - phase)
        twist_angles = xs * self.twist_rate * TAU
        ys = np.sin(twist_angles) * outs
        zs = np.cos(twist_angles) * outs
        x0, y0, z0 = self.start_point

        self.set_points_as_corners(
            axes.c2p(x0 + xs, y0 + ys, z0 + zs)
        )

    def get_default_color(self, wave_len):
        return get_spectral_color(inverse_interpolate(
            2.0, 0.5, wave_len
        ))


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
    speed = 0.5

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

        # Prepare objects to show change in oscillation direction
        x_tracker = ValueTracker(0)
        get_x = lambda: float(x_tracker.get_value())

        rod = Line(IN, OUT)
        rod.set_stroke(wave.get_stroke_color(), width=3)
        rod.set_flat_stroke(False)

        plane = Square(side_length=axes.z_axis.get_length())
        plane.set_fill(BLUE, 0.25)
        plane.set_stroke(width=0)
        plane.rotate(PI / 2, UP)
        plane.add_updater(lambda m: m.move_to(axes.c2p(get_x(), 0, 0)))

        ball = TrueDot(radius=0.075)
        ball.make_3d()
        ball.move_to(rod.get_center())
        ball.set_color(wave.get_color())

        def update_rod(rod):
            x = get_x()
            twist = twist_rate_tracker.get_value()
            rod.put_start_and_end_on(
                axes.c2p(x, 0, -2.0 * self.amplitude),
                axes.c2p(x, 0, 2.0 * self.amplitude),
            )
            rod.rotate(-TAU * twist * x, RIGHT)
            return rod

        def update_ball(ball):
            alpha = inverse_interpolate(
                axes.x_axis.x_min,
                axes.x_axis.x_max,
                get_x()
            )
            ball.move_to(wave.pfp(alpha))

        rod.add_updater(update_rod)
        ball.add_updater(update_ball)

        # Add rod with oscillating ball
        x_tracker.set_value(0)
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
        frame_configs = [
            [(-42, 75, 0), [0.54, 0.55, -0.31], 9.40],
            [(-48, 69, 0), [1.15, -0.38, -0.16], 5.52],
            [(-36, 77, 0), [2.19, 1.56, -0.34], 10.80],
            [(-45, 80, 0), [4.84, -0.18, 0.25], 8.04],
        ]

        for x, f_conf in zip(range(3, 15, 3), frame_configs):
            angles, center, height = f_conf
            b_rad = ball.radius
            globals().update(locals())
            self.play(
                self.frame.animate.reorient(*angles).move_to(center).set_height(height),
                x_tracker.animate.set_value(x).set_anim_args(time_span=(0, 1), rate_func=linear),
                UpdateFromAlphaFunc(
                    ball,
                    lambda m, a: m.set_radius(0 if a < 1 / 6 else b_rad),
                ),
                run_time=6,
                rate_func=linear
            )
            self.wait(3)

        self.play(
            self.frame.animate.reorient(-34, 80, 0).move_to([2.49, 1.68, -0.64]).set_height(11.95),
            run_time=8,
        )
        self.wait(8)


class TwistingBlueLightBeam(TwistingLightBeam):
    wave_len = 0.5
    twist_rate = 1 / 24


class TwistingRedLightBeam(TwistingLightBeam):
    wave_len = 2.0
    twist_rate = 1 / 48


class TwistingWithinCylinder(InteractiveScene):
    default_frame_orientation = (-40, 80)
    n_lines = 11

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

        self.play(GrowFromCenter(light))
        self.play(
            Write(polarizer_label),
            FadeIn(polarizer, IN),
            self.frame.animate.reorient(-63, 78, 0).move_to([-0.09, 0.13, -0.17]).set_height(7.36).set_anim_args(run_time=3),
        )

        self.wait()

        # Many waves
        waves = VGroup(*(
            OscillatingWave(
                axes,
                amplitude=0.3,
                wave_len=wave_len,
                color=line.get_color(),
                start_point=(-1, axes.y_axis.p2n(line.get_center()), 0),
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
            frame.animate.reorient(-102, 77, 0).move_to([-0.09, 0.13, -0.17]).set_height(7.36),
            light.animate.scale(0.25).shift(1 * LEFT),
            run_time=6,
        )
        self.play(
            frame.animate.reorient(-57, 77, 0).move_to([-0.09, 0.13, -0.17]).set_height(7.36),
            run_time=6,
        )
        self.remove(superposition)
        superposition.clear_updaters()
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
            self.play(*anims)

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
                rho = 1 / (line.wave_length / 350)**2  # Change this
                line.rotate(dist * rho, RIGHT)
                line.move_to(axes.c2p(dist, 0, 0))
                line.set_gloss(3 * np.exp(-3 * dist))

        lines.add_updater(update_lines)

        # Add wave trails
        trails = VGroup(*(
            self.get_wave_trail(line)
            for line in lines
        ))
        self.add(trails, lines, white_lines)

        # Move light beams down the pole
        distance_tracker.set_value(0)
        plane.add_updater(lambda m: m.match_x(lines))
        self.remove(white_lines)
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
        new_color = interpolate_color(colors[-1], colors[4], 0.2)
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
            self.frame.animate.reorient(60, 72, 0).move_to([3.17, 0.4, -0.56]),
            run_time=8,
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


class ShowInteractionsWithPolarizer(InteractiveScene):
    def construct(self):
        pass


