from manim_imports_ext import *


class BeamSplitter(InteractiveScene):
    def construct(self):
        # Add laser device
        frame = self.frame
        light_source = self.camera.light_source
        light_source.move_to([-8, 5, 1])
        pointer = self.get_laser_pointer()
        pointer.to_edge(LEFT)
        beam = self.get_beam(pointer.get_right(), pointer.get_right() + 1000 * RIGHT)
        pointer.set_z_index(2)

        theta_tracker = ValueTracker(90 * DEG)
        pointer.curr_angle = 90 * DEG

        def set_theta(target_angle, run_time=2):
            curr_angle = theta_tracker.get_value()
            return AnimationGroup(
                theta_tracker.animate.set_value(target_angle),
                Rotate(pointer, curr_angle - target_angle, axis=RIGHT),
                run_time=run_time
            )

        frame.reorient(-121, 76, 0, (-3.29, -0.25, -0.34), 4.80)
        self.add(pointer)
        self.play(ShowCreation(beam, rate_func=lambda t: t**10))

        # Set up linear vector field
        wave = self.get_wave(theta_tracker, start_point=pointer.get_right(), max_x=200)

        self.add(theta_tracker)
        self.play(VFadeIn(wave))
        self.play(
            frame.animate.reorient(-57, 74, 0, (-3.29, -0.18, -0.18), 4.80),
            run_time=8
        )

        # Add sample vector
        amplitude = wave.amplitude
        sample_point = pointer.get_right() + 3 * RIGHT

        corner_plane, h_plane, v_plane, plane_in_situ = planes = VGroup(
            NumberPlane((-1, 1), (-1, 1))
            for _ in range(4)
        )
        for plane in planes:
            plane.axes.set_stroke(WHITE, 2, 0.5)
            plane.background_lines.set_stroke(opacity=0.5)
            plane.faded_lines.set_stroke(opacity=0.25)

        plane_in_situ.scale(amplitude)
        plane_in_situ.rotate(90 * DEG, RIGHT).rotate(90 * DEG, IN)
        plane_in_situ.move_to(sample_point)

        fixed_planes = planes[:3]
        fixed_planes.fix_in_frame()
        fixed_planes.scale(1.25)
        fixed_planes.arrange(RIGHT, buff=1.0)
        fixed_planes.to_corner(UL)

        corner_vector = Vector(RIGHT, thickness=3, fill_color=BLUE)
        corner_vector.plane = corner_plane
        corner_vector.wave = wave
        corner_vector.force_unit = False
        corner_vector.fix_in_frame()

        def update_corner_vect(vect, vertical=False, horizontal=False):
            coords = vect.wave.axes.p2c(sample_point)
            output = vect.wave.func(np.array([coords]))[0]
            x = np.dot(output, DOWN) / amplitude if not vertical else 0
            y = np.dot(output, OUT) / amplitude if not horizontal else 0
            if vect.force_unit:
                theta = theta_tracker.get_value()
                x /= math.cos(theta) or 1
                y /= math.sin(theta) or 1
            vect.put_start_and_end_on(vect.plane.c2p(0, 0), vect.plane.c2p(x, y))
            return vect

        corner_vector.add_updater(update_corner_vect)

        self.play(
            FadeIn(plane_in_situ),
            TransformFromCopy(plane_in_situ, corner_plane, run_time=2)
        )
        self.play(VFadeIn(corner_vector))
        self.play(frame.animate.reorient(-80, 79, 0, (-3.36, 0.1, -0.46), 4.80), run_time=12)

        # Add beam splitter
        split_point_dist = 6
        split_point = pointer.get_right() + split_point_dist * RIGHT
        splitter = Cube()
        splitter.set_color(WHITE)
        splitter.set_opacity(0.25)
        splitter.rotate(45 * DEG)
        splitter.set_height(0.5)
        splitter.move_to(split_point)

        top_axes, low_axies = split_axes = VGroup(ThreeDAxes(), ThreeDAxes())
        split_axes.move_to(split_point)
        for axes, sgn in zip(split_axes, [1, -1]):
            axes.rotate(sgn * 45 * DEG)

        short_wave = self.get_wave(theta_tracker, pointer.get_right(), max_x=split_point_dist, stroke_opacity=0.5)
        short_wave.time = wave.time
        corner_vector.wave = short_wave
        top_wave = self.get_wave(theta_tracker, split_point, refraction_angle=45 * DEG, project_horizontal=True, stroke_opacity=0.25)
        low_wave = self.get_wave(theta_tracker, split_point, refraction_angle=-45 * DEG, project_vertical=True, stroke_opacity=0.25)

        short_beam = self.get_beam(pointer.get_right(), split_point)
        top_beam = self.get_beam(split_point, split_point + 20 * UR / math.sqrt(2))
        low_beam = self.get_beam(split_point, split_point + 20 * DR / math.sqrt(2))
        top_beam.f_always.set_stroke(opacity=lambda: math.cos(theta_tracker.get_value()))
        low_beam.f_always.set_stroke(opacity=lambda: math.sin(theta_tracker.get_value()))
        top_beam.suspend_updating()
        low_beam.suspend_updating()

        self.play(
            FadeIn(splitter),
            FadeOut(wave),
            FadeIn(short_wave),
            FadeIn(top_wave),
            FadeIn(low_wave),
            FadeOut(beam),
            FadeIn(short_beam),
            FadeIn(top_beam),
            FadeIn(low_beam),
            plane_in_situ.animate.fade(0.5),
            frame.animate.reorient(-76, 62, 0, (-2.7, 0.08, -0.8), 6.22).set_anim_args(run_time=4)
        )
        self.wait(3)

        # Show rotation of the beam
        top_beam.resume_updating()
        low_beam.resume_updating()

        polarization_line = DashedLine(LEFT, RIGHT)
        polarization_line.set_stroke(WHITE, 1)
        polarization_line.fix_in_frame()
        polarization_line.add_updater(lambda m: m.set_angle(theta_tracker.get_value()))
        polarization_line.add_updater(lambda m: m.move_to(corner_plane))

        rot_arrows = VGroup(
            Arrow(RIGHT, LEFT, path_arc=PI),
            Arrow(LEFT, RIGHT, path_arc=PI),
        )
        rot_arrows.scale(0.5)
        rot_arrows.rotate(90 * DEG, RIGHT)
        rot_arrows.rotate(90 * DEG, OUT)
        rot_arrows.move_to(pointer.get_right() + 0.5 * RIGHT)

        self.add(polarization_line, corner_vector)
        self.play(Write(rot_arrows, lag_ratio=0, run_time=1))
        self.play(set_theta(0, run_time=3))
        self.play(FadeOut(rot_arrows))
        self.play(frame.animate.reorient(-90, 82, 0, (-1.73, 0.07, 1.0), 8.00), run_time=6)

        self.play(set_theta(60 * DEG, run_time=4))
        self.wait(4)

        # Express sample vector as a sum
        eq, plus = signs = VGroup(Tex(R"="), Tex(R"+"))
        signs.scale(1.5)
        signs.fix_in_frame()
        for sign, plane1, plane2 in zip(signs, fixed_planes, fixed_planes[1:]):
            sign.move_to(midpoint(plane1.get_right(), plane2.get_left()))

        coords = VGroup(DecimalNumber(0, unit=R"\times"), DecimalNumber(0, unit=R"\times"))  # Stopped using these, maybe later?
        coords.fix_in_frame()
        coords.scale(0.75)
        for coord, plane in zip(coords, fixed_planes[1:]):
            coord.next_to(plane, LEFT, SMALL_BUFF)
        coords[0].add_updater(lambda m: m.set_value(math.sin(theta_tracker.get_value())))
        coords[1].add_updater(lambda m: m.set_value(math.cos(theta_tracker.get_value())))

        h_vect, soft_h_vect, v_vect, soft_v_vect = corner_vector.replicate(4).clear_updaters()

        h_vect.plane = h_plane
        v_vect.plane = v_plane
        soft_h_vect.plane = corner_plane
        soft_v_vect.plane = corner_plane

        VGroup(soft_h_vect, soft_v_vect).set_fill(opacity=0.5)
        h_vect.add_updater(lambda m: update_corner_vect(m, horizontal=True))
        soft_h_vect.add_updater(lambda m: update_corner_vect(m, horizontal=True))
        v_vect.add_updater(lambda m: update_corner_vect(m, vertical=True))
        soft_v_vect.add_updater(lambda m: update_corner_vect(m, vertical=True))

        for plane in h_plane, v_plane:
            plane.save_state()
            plane.move_to(corner_plane)
            plane.set_stroke(opacity=0)

        self.play(
            VFadeIn(h_plane),
            VFadeIn(v_plane),
            VFadeIn(soft_h_vect),
            VFadeIn(soft_v_vect),
        )
        self.wait()
        self.play(
            Restore(h_plane, path_arc=30 * DEG),
            VFadeIn(h_vect),
            Write(eq),
        )
        self.play(
            Restore(v_plane, path_arc=30 * DEG),
            VFadeIn(v_vect),
            Write(plus),
        )
        self.wait(5)

        # Do some rotations
        curr_angle = 60 * DEG
        for target_angle in [90 * DEG, 0, 60 * DEG]:
            self.play(set_theta(target_angle))
            self.wait(3)

        # Add the angle, and sine/cosine terms
        arc = always_redraw(lambda: Arc(
            0, theta_tracker.get_value(), radius=0.5, arc_center=corner_plane.get_center(),
        ).fix_in_frame())
        theta_label = Tex(R"\theta")
        theta_label.fix_in_frame()
        theta_label_height = theta_label.get_height()

        def update_theta_label(theta_label):
            point = arc.pfp(0.25)
            direction = rotate_vector(RIGHT, 0.5 * theta_tracker.get_value())
            height = min(arc.get_height(), theta_label_height)
            theta_label.set_height(height)
            theta_label.next_to(point, direction, SMALL_BUFF)

        theta_label.add_updater(update_theta_label)

        movers = VGroup(h_plane, plus, v_plane)
        for mob in movers:
            mob.generate_target()

        cos_term = Tex(R"\cos(\theta) \, \cdot ").fix_in_frame()
        sin_term = Tex(R"\sin(\theta) \, \cdot ").fix_in_frame()
        rhs = VGroup(cos_term, h_plane.target, plus.target, sin_term, v_plane.target)
        rhs.arrange(RIGHT, buff=0.25)
        rhs.next_to(eq, RIGHT, 0.25)

        self.play(
            VFadeIn(arc),
            Write(theta_label),
        )
        h_vect.force_unit = True
        v_vect.force_unit = True
        self.play(
            LaggedStartMap(MoveToTarget, movers),
            LaggedStartMap(FadeIn, VGroup(cos_term, sin_term)),
            FadeTransform(theta_label.copy().clear_updaters(), cos_term[R"\theta"][0], time_span=(0.25, 1.25)),
            FadeTransform(theta_label.copy().clear_updaters(), sin_term[R"\theta"][0], time_span=(0.5, 1.5)),
            run_time=1.5
        )
        self.wait(4)

        # Put each part in context
        sin_part = VGroup(sin_term, v_plane)
        cos_part = VGroup(cos_term, h_plane)

        self.play(
            sin_part.animate.scale(0.5).rotate(5 * DEG).rotate(45 * DEGREES, UP).shift(3 * DOWN),
            rate_func=there_and_back_with_pause,
            run_time=4
        )
        self.play(
            cos_part.animate.scale(0.5).rotate(-5 * DEG).rotate(45 * DEGREES, DOWN).shift(3 * DOWN + 2 * LEFT),
            rate_func=there_and_back_with_pause,
            run_time=4
        )
        self.wait(3)

        # More rotation!
        for target_angle in [80 * DEG, 10 * DEG, 60 * DEG]:
            self.play(set_theta(target_angle))
            self.wait()

        # Energy is proportional to amlpitude squared
        e_expr = Tex(R"E = k \cdot (\text{Amplitude})^2", font_size=36)
        e_expr.fix_in_frame()
        e_expr.next_to(corner_plane, DOWN, aligned_edge=LEFT)
        amp_brace = LineBrace(polarization_line.copy().scale(0.5, about_edge=UR), buff=SMALL_BUFF)
        amp_brace.fix_in_frame()
        one_label = amp_brace.get_tex("1").fix_in_frame()
        VGroup(amp_brace, one_label).set_fill(WHITE)

        self.play(
            Write(e_expr),
            frame.animate.reorient(-77, 80, 0, (-1.83, 2.76, 0.89), 8.00),
        )
        self.wait(5)
        self.play(
            GrowFromCenter(amp_brace),
            Write(one_label),
            VFadeOut(soft_h_vect),
            VFadeOut(soft_v_vect),
        )
        self.wait()

        # Numbers of the 60 degree example
        eq_60 = Tex(R"= 60^\circ")

        # A lot of lingering

        # Add photo sensors

        # Turn down power

    def get_laser_pointer(self):
        box = Prism(0.75, 0.25, 0.25)
        box.set_color(GREY_D)
        box.set_shading(0.5, 0.5, 0)

        cone = ParametricSurface(
            lambda u, v: np.array([
                u,
                u * math.cos(-TAU * v),
                u * math.sin(TAU * v),
            ])
        )
        cone.stretch(5, 0)
        cone.set_width(0.25)
        cone.move_to(box.get_right())
        cone.set_color(GREY)

        return Group(box, cone)

    def get_beam(
        self,
        start,
        end,
        color=GREEN_SCREEN,
        stroke_width=2,
        opacity=1.0,
        anti_alias_width=25,
    ):
        beam = Line(start, end)
        beam.set_stroke(color, stroke_width, opacity)
        beam.set_anti_alias_width(anti_alias_width)
        return beam

    def get_wave(
        self,
        theta_tracker,
        start_point=ORIGIN,
        max_x=20,
        refraction_angle=0 * DEG,
        wave_number=4.0,
        freq=0.5,
        amplitude=0.25,  # Maybe replace with an amplitude tracker?
        color=BLUE,
        stroke_opacity=0.5,
        vector_density=0.1,
        max_vect_len=1.0,
        project_vertical=False,
        project_horizontal=False,
    ):
        axes = ThreeDAxes()
        axes.rotate(refraction_angle)
        axes.move_to(start_point)

        def field_func(points, time):
            theta = theta_tracker.get_value()
            magnitudes = amplitude * np.cos(wave_number * points[:, 0] - TAU * freq * time)
            result = np.zeros_like(points)
            if not project_vertical:
                result[:, 1] = -math.cos(theta) * magnitudes
            if not project_horizontal:
                result[:, 2] = math.sin(theta) * magnitudes
            return result

        density = 0.1
        sample_coords = np.arange(0, max_x, density)[:, np.newaxis] * RIGHT
        wave = TimeVaryingVectorField(
            field_func,
            axes,
            sample_coords=sample_coords,
            max_vect_len=max_vect_len,
            color=color,
            stroke_opacity=stroke_opacity
        )
        wave.amplitude = amplitude
        wave.axes = axes
        return wave
