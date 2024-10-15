from manim_imports_ext import *
from _2023.optics_puzzles.objects import *


# Scenes


class SimpleLightBeam(InteractiveScene):
    default_frame_orientation = (-33, 85)
    axes_config = dict()
    z_amplitude = 0.5
    wave_len = 2.0
    speed = 1.0
    color = YELLOW
    oscillating_field_config = dict(
        stroke_opacity=0.5,
        stroke_width=2,
        tip_width_ratio=1
    )

    def construct(self):
        axes, plane = get_axes_and_plane(**self.axes_config)
        self.add(axes, plane)

        # Introduce wave
        wave = OscillatingWave(
            axes,
            z_amplitude=self.z_amplitude,
            wave_len=self.wave_len,
            speed=self.speed,
            color=self.color
        )
        vect_wave = OscillatingFieldWave(axes, wave, **self.oscillating_field_config)

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
        vect_wave.add_updater(update_wave)

        self.add(wave)
        self.play(
            self.frame.animate.reorient(-98, 77, 0).move_to([-0.87, 0.9, -0.43]),
            run_time=8,
        )
        self.add(vect_wave, wave)
        self.play(
            VFadeIn(vect_wave),
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
    z_amplitude = 0.5
    wave_len = 2.0
    twist_rate = 1 / 72
    speed = 1.0
    color = YELLOW

    def construct(self):
        # Axes
        axes, plane = get_axes_and_plane(**self.axes_config)
        self.add(axes, plane)

        # Add wave
        wave = OscillatingWave(
            axes,
            z_amplitude=self.z_amplitude,
            wave_len=self.wave_len,
            speed=self.speed,
            color=self.color
        )
        vect_wave = OscillatingFieldWave(axes, wave, **self.oscillating_field_config)

        twist_rate_tracker = ValueTracker(0)

        def update_twist_rate(wave):
            wave.twist_rate = twist_rate_tracker.get_value()
            return wave

        wave.add_updater(update_twist_rate)

        cylinder = SugarCylinder(axes, self.camera, radius=self.z_amplitude)

        self.add(vect_wave, wave)
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
                axes.c2p(x, wave_y, wave_z - length_mult * wave.z_amplitude),
                axes.c2p(x, wave_y, wave_z + length_mult * wave.z_amplitude),
            )
            rod.rotate(TAU * wave.twist_rate * x, RIGHT)
            return rod

        rod.add_updater(update_rod)
        return rod

    def get_wave_ball(self, wave, get_x, radius=0.075):
        ball = TrueDot(radius=radius)
        ball.make_3d()
        ball.set_color(wave.get_color())

        def update_ball(ball):
            ball.move_to(wave.offset + wave.xt_to_point(get_x(), wave.time))
            return ball

        ball.add_updater(update_ball)
        return ball

    def get_slice_plane(self, axes, get_x):
        plane = Square(side_length=axes.z_axis.get_length())
        plane.set_fill(BLUE, 0.25)
        plane.set_stroke(width=0)
        circle = Circle(
            radius=axes.z_axis.get_unit_size() * self.z_amplitude,
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
        x_label.value_mob = x_label.make_number_changeable("0.00")
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
    wave_len = 1.0
    twist_rate = 1 / 48
    color = PURPLE


class TwistingRedLightBeam(TwistingLightBeam):
    wave_len = 3.0
    twist_rate = 1 / 96
    color = RED


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
                z_amplitude=0.3,
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
    random_seed = 3
    cylinder_radius = 0.5
    wave_config = dict(
        z_amplitude=0.15,
        wave_len=0.5,
        color=get_spectral_color(0.1),
        speed=1.0,
        twist_rate=-1 / 24
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
        line = VMobject()
        line.set_stroke(RED, 2)
        line.set_flat_stroke(False)
        dot = TrueDot(radius=0.05)
        dot.make_3d()
        for x in range(10):
            point = axes.c2p(
                random.uniform(axes.x_axis.x_min, axes.x_axis.x_max),
                random.uniform(-max_y, -max_y),
                random.uniform(-max_y, -max_y),
            )
            line_points = [light.get_center(), point, randy.eyes.get_top()]
            self.add(dot, cylinder)
            if x == 0:
                dot.move_to(point)
                line.set_points_as_corners(line_points)
                self.play(ShowCreation(line))
            else:
                self.play(
                    line.animate.set_points_as_corners(line_points),
                    dot.animate.move_to(point),
                )
                self.wait()
        self.play(
            FadeOut(line),
            FadeOut(dot),
        )

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
            Write(line_of_sight, time_span=(3, 4), lag_ratio=0),
            run_time=5,
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
            amplitude=wave.z_amplitude,
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
        small_wave_config["z_amplitude"] = amp

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


class VectorFieldWigglingNew(InteractiveScene):
    default_frame_orientation = (-33, 85)

    def construct(self):
        # Waves
        axes, plane = get_axes_and_plane()
        self.add(axes, plane)

        wave = OscillatingWave(
            axes,
            wave_len=3.0,
            speed=1.5,
            color=BLUE,
            z_amplitude=0.5,
        )
        vector_wave = OscillatingFieldWave(axes, wave)
        wave_opacity_tracker = ValueTracker(0)
        vector_opacity_tracker = ValueTracker(1)
        wave.add_updater(lambda m: m.set_stroke(opacity=wave_opacity_tracker.get_value()))
        vector_wave.add_updater(lambda m: m.set_stroke(opacity=vector_opacity_tracker.get_value()))

        self.add(wave, vector_wave)

        # Charges
        charges = DotCloud(color=RED)
        charges.to_grid(50, 50)
        charges.set_radius(0.04)
        charges.set_height(2 * axes.z_axis.get_length())
        charges.rotate(PI / 2, RIGHT).rotate(PI / 2, IN)
        charges.move_to(axes.c2p(-10, 0, 0))
        charges.make_3d()

        charge_opacity_tracker = ValueTracker(1)
        charges.add_updater(lambda m: m.set_opacity(charge_opacity_tracker.get_value()))
        charges.add_updater(lambda m: m.set_z(0.3 * wave.xt_to_point(0, self.time)[2]))

        self.add(charges, wave, vector_wave)

        # Pan camera
        self.frame.reorient(47, 69, 0).move_to([-8.68, -7.06, 2.29]).set_height(5.44)
        self.play(
            self.frame.animate.reorient(-33, 83, 0).move_to([-0.75, -1.84, 0.38]).set_height(8.00),
            run_time=10,
        )
        self.play(
            self.frame.animate.reorient(-27, 80, 0).move_to([-0.09, -0.42, -0.1]).set_height(9.03),
            wave_opacity_tracker.animate.set_value(1).set_anim_args(time_span=(1, 2)),
            vector_opacity_tracker.animate.set_value(0.5).set_anim_args(time_span=(1, 2)),
            run_time=4,
        )

        # Highlight x_axis
        x_line = Line(*axes.x_axis.get_start_and_end())
        x_line.set_stroke(BLUE, 10)

        self.play(
            wave_opacity_tracker.animate.set_value(0.25),
            vector_opacity_tracker.animate.set_value(0.25),
            charge_opacity_tracker.animate.set_value(0.25),
        )
        self.play(
            ShowCreation(x_line, run_time=2),
        )
        self.wait(5)

        # Show 3d wave
        wave_3d = VGroup()
        origin = axes.get_origin()
        for y in np.linspace(-1, 1, 5):
            for z in np.linspace(-1, 1, 5):
                vects = OscillatingFieldWave(
                    axes, wave,
                    max_vect_len=0.5,
                    norm_to_opacity_func=lambda n: 0.75 * np.arctan(n),
                )
                vects.y = y
                vects.z = z
                vects.add_updater(lambda m: m.shift(axes.c2p(0, m.y, m.z) - origin))
                wave_3d.add(vects)

        self.wait(2)
        wave_opacity_tracker.set_value(0)
        self.remove(vector_wave)
        self.remove(x_line)
        self.add(wave_3d)
        self.wait(2)

        self.play(
            self.frame.animate.reorient(22, 69, 0).move_to([0.41, -0.67, -0.1]).set_height(10.31),
            run_time=8
        )
        self.play(
            self.frame.animate.reorient(-48, 68, 0).move_to([0.41, -0.67, -0.1]),
            run_time=10
        )


class ClockwiseCircularLight(InteractiveScene):
    clockwise = True
    default_frame_orientation = (-20, 70)
    color = YELLOW
    x_range = (0, 24)
    amplitude = 0.5

    def setup(self):
        super().setup()
        # Axes
        axes, plane = get_axes_and_plane(x_range=self.x_range)
        self.add(axes, plane)

        # Wave
        wave = OscillatingWave(
            axes,
            wave_len=3,
            speed=0.5,
            z_amplitude=self.amplitude,
            y_amplitude=self.amplitude,
            y_phase=-PI / 2 if self.clockwise else PI / 2,
            color=self.color,
        )
        vect_wave = OscillatingFieldWave(axes, wave)
        vect_wave.set_stroke(opacity=0.7)

        self.add(wave, vect_wave)

    def construct(self):
        self.play(
            self.frame.animate.reorient(73, 82, 0),
            run_time=5
        )
        for pair in [(100, 70), (59, 72), (110, 65), (60, 80)]:
            self.play(
                self.frame.animate.reorient(*pair),
                run_time=12,
            )


class CounterclockwiseCircularLight(ClockwiseCircularLight):
    clockwise = False
    color = RED


class AltClockwiseCircularLight(ClockwiseCircularLight):
    x_range = (0, 8)
    amplitude = 0.4

    def construct(self):
        self.frame.reorient(69, 81, 0)
        self.wait(12)


class AltCounterclockwiseCircularLight(CounterclockwiseCircularLight):
    x_range = (0, 8)
    amplitude = 0.4

    def construct(self):
        self.frame.reorient(69, 81, 0)
        self.wait(12)


class TransitionTo2D(InteractiveScene):
    default_frame_orientation = (-20, 70)
    wave_config = dict(
        color=BLUE,
        wave_len=3,
    )

    def construct(self):
        # Axes
        axes, plane = get_axes_and_plane(x_range=(0, 8.01))
        self.add(axes, plane)

        # Waves
        wave = OscillatingWave(axes, **self.wave_config)
        vect_wave = OscillatingFieldWave(axes, wave)
        wave_opacity_tracker = ValueTracker(1)
        wave.add_updater(lambda m: m.set_stroke(opacity=wave_opacity_tracker.get_value()))
        vect_wave.add_updater(lambda m: m.set_stroke(opacity=wave_opacity_tracker.get_value()))

        single_vect = Vector(OUT, stroke_color=wave.get_color(), stroke_width=4)
        single_vect.set_flat_stroke(False)

        def update_vect(vect):
            x_max = axes.x_axis.x_max
            base = axes.c2p(x_max, 0, 0)
            vect.put_start_and_end_on(base, base + wave.xt_to_point(x_max, wave.time) * [0, 1, 1])

        single_vect.add_updater(update_vect)

        self.add(wave, vect_wave)

        # Shift perspective
        self.frame.reorient(69, 81, 0)
        wave_opacity_tracker.set_value(0.8)
        self.wait(6)
        self.play(
            self.frame.animate.reorient(73, 84, 0),
            VFadeIn(single_vect, time_span=(2, 3)),
            wave_opacity_tracker.animate.set_value(0.35).set_anim_args(time_span=(2, 3)),
            run_time=5,
        )
        self.wait()
        self.play(
            self.frame.animate.reorient(90, 90, 0),
            wave_opacity_tracker.animate.set_value(0),
            run_time=4,
        )
        self.wait(4)


class TransitionTo2DRightHanded(TransitionTo2D):
    wave_config = dict(
        wave_len=3,
        y_amplitude=0.5,
        z_amplitude=0.5,
        y_phase=PI / 2,
        color=RED,
    )


class TransitionTo2DLeftHanded(TransitionTo2D):
    wave_config = dict(
        wave_len=3,
        y_amplitude=0.5,
        z_amplitude=0.5,
        y_phase=-PI / 2,
        color=YELLOW,
    )


class LinearAsASuperpositionOfCircular(InteractiveScene):
    rotation_rate = 0.25
    amplitude = 2.0

    def construct(self):
        # Set up planes
        plane_config = dict(
            background_line_style=dict(stroke_color=GREY, stroke_width=1),
            faded_line_style=dict(stroke_color=GREY, stroke_width=0.5, stroke_opacity=0.5),
        )
        planes = VGroup(
            ComplexPlane((-1, 1), (-1, 1), **plane_config),
            ComplexPlane((-1, 1), (-1, 1), **plane_config),
            ComplexPlane((-2, 2), (-2, 2), **plane_config),
        )
        planes[:2].arrange(DOWN, buff=2.0).set_height(FRAME_HEIGHT - 1.5).next_to(ORIGIN, LEFT, 1.0)
        planes[2].set_height(6).next_to(ORIGIN, RIGHT, 1.0)
        # planes.arrange(RIGHT, buff=1.5)
        self.add(planes)

        # Set up trackers
        phase_trackers = ValueTracker(0).replicate(2)
        phase1_tracker, phase2_tracker = phase_trackers

        def update_phase(m, dt):
            m.increment_value(TAU * self.rotation_rate * dt)

        def slow_changer(m, dt):
            m.increment_value(-0.5 * TAU * self.rotation_rate * dt)

        for tracker in phase_trackers:
            tracker.add_updater(update_phase)

        self.add(*phase_trackers)

        def get_z1():
            return 0.5 * self.amplitude * np.exp((PI / 2 + phase1_tracker.get_value()) * 1j)

        def get_z2():
            return 0.5 * self.amplitude * np.exp((PI / 2 - phase2_tracker.get_value()) * 1j)

        def get_sum():
            return get_z1() + get_z2()

        # Vectors
        vects = VGroup(
            self.get_vector(planes[0], get_z1, color=RED),
            self.get_vector(planes[1], get_z2, color=YELLOW),
            self.get_vector(planes[2], get_sum, color=BLUE),
            self.get_vector(planes[2], get_z1, color=RED),
            self.get_vector(planes[2], get_sum, get_base=get_z1, color=YELLOW),
        )

        self.add(*vects)

        # Polarization line
        pol_line = Line(UP, DOWN)
        pol_line.set_stroke(YELLOW, 1)
        pol_line.match_height(planes[2])
        pol_line.move_to(planes[2])

        def update_pol_line(line):
            if abs(vects[2].get_length()) > 1e-3:
                line.set_angle(vects[2].get_angle())
                line.move_to(planes[2].n2p(0))
            return line

        pol_line.add_updater(update_pol_line)

        self.add(pol_line, *planes, *vects)

        # Write it as an equation
        plus = Tex("+", font_size=72)
        equals = Tex("=", font_size=72)
        plus.move_to(planes[0:2])
        # equals.move_to(planes[1:3])
        equals.move_to(ORIGIN)

        self.add(plus, equals)

        # Slow down annotation
        arcs = VGroup(
            Arrow(LEFT, RIGHT, path_arc=-PI, stroke_width=2),
            Arrow(RIGHT, LEFT, path_arc=-PI, stroke_width=2),
        )
        arcs.move_to(planes[0])
        slow_word = Text("Slow down!")
        slow_word.next_to(planes[0], DOWN)
        sucrose = Sucrose(height=1)
        sucrose.balls.scale_radii(0.25)
        sucrose.fade(0.5)
        sucrose.move_to(planes[0])
        slow_group = Group(slow_word, arcs, sucrose)

        def slow_down():
            self.play(FadeIn(slow_group, run_time=0.25))
            phase1_tracker.add_updater(slow_changer)
            self.wait(0.75)
            phase1_tracker.remove_updater(slow_changer)
            self.play(FadeOut(slow_group))

        # Highlight constituent parts
        back_rects = VGroup(*(BackgroundRectangle(plane) for plane in planes))
        back_rects.set_fill(opacity=0.5)

        self.wait(8)

        self.add(back_rects[1])
        VGroup(vects[1], vects[2], vects[4]).set_stroke(opacity=0.25)
        self.wait(8)
        self.remove(back_rects[1])

        self.add(back_rects[0])
        vects.set_stroke(opacity=1)
        VGroup(vects[0], vects[2]).set_stroke(opacity=0.25)
        self.wait(8)
        vects.set_stroke(opacity=1)
        self.remove(back_rects)
        self.wait(4)

        # Rotation labels
        for tracker in phase_trackers:
            tracker.set_value(0)

        rot_labels = VGroup(*(
            TexText("Total rotation: 0.00")
            for _ in range(2)
        ))
        for rot_label, plane, tracker in zip(rot_labels, planes, phase_trackers):
            rot_label.set_height(0.2)
            rot_label.set_color(GREY_B)
            rot_label.next_to(plane, UP)
            dec = rot_label.make_number_changeable("0.00", edge_to_fix=LEFT)
            dec.phase_tracker = tracker
            dec.add_updater(lambda m: m.set_value(m.phase_tracker.get_value() / TAU))

        self.add(rot_labels)

        # Let it play, occasionally kink
        self.wait(9)
        for _ in range(20):
            slow_down()
            self.wait(3 * random.random())

    def get_vector(self, plane, get_z, get_base=lambda: 0, color=BLUE):
        vect = Vector(UP, stroke_color=color)
        vect.add_updater(lambda m: m.put_start_and_end_on(
            plane.n2p(get_base()),
            plane.n2p(get_z())
        ))
        return vect
