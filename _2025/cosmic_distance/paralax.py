from manim_imports_ext import *
from _2025.cosmic_distance.planets import *


class SimpleDotsParalax(InteractiveScene):
    show_pi_perspective = False

    def construct(self):
        # Add stars
        frame = self.frame
        self.set_floor_plane("xz")

        height = 4
        cube = VCube(height)
        cube.set_fill(opacity=0)
        cube.set_stroke(BLUE, 2)

        n_stars = 200
        stars = GlowDots(np.random.uniform(-1, 1, (n_stars, 3)))
        stars.scale(height / 2)
        stars.set_color(WHITE)
        stars.set_glow_factor(2)
        stars.set_radii(np.random.uniform(0, 0.075, n_stars))

        self.add(cube)
        self.add(stars)

        self.play(ShowCreation(stars, run_time=4))

        # Add randy
        randy = Randolph(height=1)
        randy.next_to(cube, LEFT, buff=1)

        self.play(
            VFadeIn(randy),
            randy.change("pondering", look_at=RIGHT)
        )

        if self.show_pi_perspective:
            self.play(
                frame.animate.reorient(-89, -4, 0, (0.01, 0.21, 0.0), 3.05),
                randy.animate.set_opacity(0),
                cube.animate.set_stroke(width=5).set_anti_alias_width(10),
                run_time=3,
            )
            frame.always.match_z(randy)
        else:
            self.play(frame.animate.reorient(-40, -26, 0), run_time=3)

        # Move up and down
        for dy in [1, -2, 2, -2, 1]:
            self.play(randy.animate.shift(dy * 1.5 * IN), run_time=5)

        return

        # Show some triangle
        star_points = np.array(list(filter(lambda p: get_norm(p) < 1, stars.get_points())))
        star_points[0] *= 10
        random.seed(1)
        verts = random.sample(list(star_points), 3)
        triangle = Polygon(*verts)
        triangle.set_color(RED)
        red_stars = DotCloud(verts)
        red_stars.set_radius(0.02)
        red_stars.set_color(RED)

        self.play(
            ShowCreation(triangle),
            FadeIn(red_stars),
        )


class SimpleDotsFromPerspective(SimpleDotsParalax):
    show_pi_perspective = True


class ParalxInSolarSystem(InteractiveScene):
    show_celestial_sphere = False

    def construct(self):
        # Add sun and earth
        frame = self.frame
        light_source = self.camera.light_source

        earth = get_earth(radius=0.01)
        earth.rotate(EARTH_TILT_ANGLE, UP)
        earth_axis = rotate_vector(OUT, EARTH_TILT_ANGLE, UP)
        sun = get_sun(radius=0.07, big_glow_ratio=10)

        orbit_radius = 5
        sun.move_to(ORIGIN)
        light_source.move_to(sun)

        orbit = Circle(radius=orbit_radius, n_components=100)
        orbit.set_stroke(BLUE, width=(0, 3))
        orbit.rotate(-30 * DEG)
        orbit.add_updater(lambda m, dt: m.rotate(10 * dt * DEG))
        orbit.set_stroke(flat=False)
        orbit.set_anti_alias_width(5)
        orbit.apply_depth_test()

        earth.add_updater(lambda m: m.move_to(orbit.get_end()))
        # earth.add_updater(lambda m, dt: m.rotate(2 * TAU * dt, axis=earth_axis))

        self.add(orbit, earth, sun)
        self.wait(3)

        # Add stars
        if self.show_celestial_sphere:
            n_stars = 0
            celestial_sphere = get_celestial_sphere()
            self.add(celestial_sphere)
        else:
            n_stars = 3000
        points = np.random.uniform(-1, 1, (n_stars, 3))
        points = normalize_along_axis(points, 1)
        distances = np.random.uniform(10, 50, n_stars)
        radii = np.random.uniform(0, 0.2, n_stars)
        points = points * distances[:, np.newaxis]
        stars = GlowDots(points)
        stars.set_color(WHITE)
        stars.set_radii(radii)

        self.add(stars, orbit, earth)
        self.play(
            ShowCreation(stars),
            frame.animate.reorient(0, 71, 0, (2.29, -0.66, 0.52), 43.56),
            run_time=4,
        )
        self.wait(3)

        self.play(
            frame.animate.set_euler_angles(-10 * DEG, 86 * DEG, 0).set_height(0.25),
            # frame.animate.set_height(0.1),
            UpdateFromAlphaFunc(sun, lambda m, a: frame.move_to(interpolate(frame.get_center(), earth.get_center(), min(2 * a, 1)))),
            run_time=5
        )
        frame.clear_updaters()
        frame.add_updater(lambda m: m.move_to(earth))

        self.wait(30)

        # Zoom back out
        self.play(
            frame.animate.reorient(7, 74, 0, ORIGIN, 120),
            stars.animate.set_radii(2 * radii),
            run_time=3,
        )
        new_points = 1000 * normalize_along_axis(stars.get_points(), 1)
        self.play(
            stars.animate.set_points(new_points).set_radii(20 * radii),
            run_time=4,
        )
        self.play(
            frame.animate.reorient(-89, 74, 0),
            run_time=12
        )


class ShowConstellationsDuringOrbit(ParalxInSolarSystem):
    show_celestial_sphere = True


class ParalaxMeasurmentFromEarth(InteractiveScene):
    def construct(self):
        # Add earth
        self.camera.light_source.move_to(500 * RIGHT)

        radius = 3
        earth = Circle(radius=radius)
        earth.set_fill(BLUE_B, 0.5)
        earth.set_stroke(WHITE, 3)
        earth.to_edge(LEFT)
        earth_back = earth.copy()
        earth_back.set_fill(BLACK, 1).set_stroke(width=0)

        earth_pattern = SVGMobject("earth")
        earth_pattern.replace(earth)
        earth_pattern.set_fill(Color(hsl=(0.23, 0.5, 0.2)), 1)

        self.add(earth_back, earth, earth_pattern)

        # Add two observers
        pi_height = 0.25
        randy, morty = pis = VGroup(
            Randolph(height=2, mode="hesitant").look_at(10 * RIGHT),
            Mortimer(height=2, mode="pondering").look_at(10 * RIGHT),
        )
        angles = [45 * DEG, -55 * DEG]
        labels = VGroup(
            Text("Observer 1", font_size=36),
            Text("Observer 2", font_size=36),
        )
        pis.arrange(DOWN, buff=1.0)
        pis.move_to(3 * RIGHT)

        obs_points = []
        obs_dots = Group()

        for pi, label, angle in zip(pis, labels, angles):
            label.next_to(pi, DOWN, SMALL_BUFF)
            target_point = earth.pfp((angle / TAU) % 1)

            pi.target = pi.generate_target()
            pi.target.set_height(pi_height)
            pi.target.next_to(target_point, UP, buff=0)
            pi.target.rotate(angle - 90 * DEG, about_point=target_point)

            label.target = label.generate_target()
            label.target.scale(0.5)
            # label.target.next_to(pi.target, rotate_vector(RIGHT, angle), buff=SMALL_BUFF)
            label.target.next_to(pi.target, UP * np.sign(angle), buff=SMALL_BUFF, aligned_edge=LEFT)

            obs_dots.add(TrueDot(target_point, color=pi.get_color()).make_3d())
            obs_points.append(target_point)

        self.play(
            LaggedStartMap(FadeIn, pis, shift=0.5 * UP, lag_ratio=0.5),
            LaggedStartMap(FadeIn, labels, shift=0.25 * UP, lag_ratio=0.5),

        )
        self.play(LaggedStartMap(Blink, pis, lag_ratio=0.25))
        self.play(
            LaggedStartMap(MoveToTarget, pis, lag_ratio=0.7),
            LaggedStartMap(MoveToTarget, labels, lag_ratio=0.7),
            FadeIn(obs_dots, time_span=(0.75, 1.25)),
        )
        self.wait()

        # Show lines to object
        frame = self.frame
        obj = GlowDot(12 * RIGHT, color=TEAL)

        obs_lines = VGroup(
            DashedLine(obs_points[0], obj.get_center()),
            DashedLine(obs_points[1], obj.get_center()),
        )
        obs_lines.set_stroke(WHITE, 2)

        self.play(
            frame.animate.set_width(20, about_edge=LEFT),
            *map(ShowCreation, obs_lines),
            FadeIn(obj),
            run_time=2
        )
        self.wait()

        # Analogy with eyeballs
        eyes = Randolph().eyes
        eyes.set_height(1)
        eyes.set_z_index(-1)

        def look_at(eye, object, midpoint):
            direction = normalize(object.get_center() - midpoint)
            eye.pupil.move_to(midpoint + 0.8 * eye.pupil.get_width() * direction)

        for eye, point, angle in zip(eyes, obs_points, angles):
            eye.next_to(ORIGIN, UP, buff=-0.35)
            eye.rotate(angle - 90 * DEG, about_point=ORIGIN)
            eye.shift(point)
            eye.point = point
            eye.add_updater(lambda m: look_at(m, obj, m.point))

        for line, dot in zip(obs_lines, obs_dots):
            line.dot = dot
            line.add_updater(lambda m: m.put_start_and_end_on(m.dot.get_center(), obj.get_center()))

        self.play(
            FadeIn(eyes),
            FadeOut(pis),
            FadeOut(labels),
        )

        obj.save_state()
        for vect in [6 * LEFT, 4 * UP, 4 * DOWN + 20 * RIGHT]:
            self.play(obj.animate.shift(vect), run_time=3)
        self.play(Restore(obj, run_time=3))
        self.play(
            FadeOut(eyes),
            FadeIn(pis),
            FadeIn(labels),
        )

        # Add stars
        conversion_factor = radius / EARTH_RADIUS
        celestial_sphere = get_celestial_sphere(radius=JUPITER_ORBIT_RADIUS * conversion_factor, constellation_opacity=0.0)
        celestial_sphere.set_z_index(-2)
        low_obs_group = VGroup(obs_lines[1], pis[1], labels[1])
        low_obs_group.save_state()
        frame.save_state()
        self.play(
            FadeIn(celestial_sphere),
            low_obs_group.animate.fade(0.75),
            frame.animate.set_height(20, about_edge=LEFT).shift(2 * RIGHT),
        )

        # Show moving observer
        self.play(
            Rotate(Group(pis[0], obs_dots[0]), angles[1] - angles[0], about_point=earth.get_center()),
            MaintainPositionRelativeTo(labels[0], pis[0]),
            run_time=8,
            rate_func=there_and_back,
        )
        self.play(
            Restore(low_obs_group),
            Restore(frame),
        )

        # Show line between
        line_between = Line(*obs_points)
        line_between.set_stroke(YELLOW, 3)
        brace_between = LineBrace(line_between, DOWN)

        self.play(
            ShowCreation(line_between),
            earth.animate.set_fill(opacity=0.35).set_stroke(width=2, opacity=1),
            earth_pattern.animate.set_fill(opacity=0.75),
        )
        self.wait()
        self.play(GrowFromCenter(brace_between))
        self.wait()
        self.play(FadeOut(brace_between))
        self.wait()

        # Move dot around
        self.play(low_obs_group.animate.fade(0.9))
        self.play(obj.animate.shift(3 * UP), rate_func=wiggle, run_time=5)
        self.play(Restore(low_obs_group))

        # Add angle labels
        colors = [BLUE, RED]
        angle_labels = self.get_angle_labels(obs_lines, obs_points, line_between, arc_props=[0.75, 0.5], colors=colors)

        for angle_label in angle_labels:
            self.play(Write(angle_label))
            self.wait()

        # Show remaining angle
        tip_arc = Arc(
            obs_lines[0].get_angle() + PI,
            obs_lines[1].get_angle() - obs_lines[0].get_angle(),
            arc_center=obj.get_center(),
            radius=1
        )
        tip_arc_label = Tex(
            R"180^\circ - \alpha - \beta",
            t2c={R"\alpha": colors[0], R"\beta": colors[1]}
        )
        tip_arc_label.next_to(tip_arc, LEFT, MED_SMALL_BUFF)

        self.play(LaggedStart(
            ShowCreation(tip_arc),
            FadeTransform(angle_labels[0][1].copy(), tip_arc_label[R"\alpha"][0]),
            FadeTransform(angle_labels[1][1].copy(), tip_arc_label[R"\beta"][0]),
            Write(tip_arc_label[R"180^\circ"]),
            Write(tip_arc_label[R"-"]),
            run_time=2
        ))
        self.wait()

        # Emphasize one distance
        obs1_brace = LineBrace(obs_lines[0])

        self.play(GrowFromCenter(brace_between))
        self.wait()
        self.play(Transform(brace_between, obs1_brace))
        self.wait()
        self.play(FadeOut(brace_between))

        # Replace with true earth
        frame.set_field_of_view(20 * DEG)
        true_earth = get_earth(radius=radius)
        true_earth.move_to(earth)
        true_earth.set_z_index(-1)
        true_earth.rotate(90 * DEG, LEFT)
        true_earth.rotate(140 * DEG, UP)
        true_earth.rotate(-EARTH_TILT_ANGLE, OUT)

        new_obs_lines = VGroup(
            Line(ol.get_start(), ol.get_end())
            for ol in obs_lines
        )
        new_obs_lines.match_style(obs_lines)

        self.play(
            FadeIn(true_earth),
            FadeOut(earth_back),
            FadeOut(earth),
            FadeOut(earth_pattern),
            FadeOut(tip_arc_label),
            FadeOut(tip_arc),
            FadeOut(obs_lines),
            FadeIn(new_obs_lines),
        )
        self.wait()

        obs_lines = new_obs_lines

        # Drag point very far away, show orbitss
        self.set_floor_plane("xz")

        for line, dot in zip(obs_lines, obs_dots):
            line.dot = dot
            line.add_updater(lambda m: m.put_start_and_end_on(m.dot.get_center(), obj.get_center()))

        angle_labels.add_updater(
            lambda m: m.become(
                self.get_angle_labels(
                    obs_lines,
                    obs_points=[obs_dots[0].get_center(), obs_dots[1].get_center()],
                    line_between=line_between,
                    arc_props=[0.75, 0.5]
                )
            )
        )

        moon_orbit = Circle(radius=MOON_ORBIT_RADIUS * conversion_factor)
        moon_orbit.set_stroke(GREY_B, width=(0, 3))
        moon_orbit.move_to(earth)
        moon_orbit.rotate(90 * DEG, LEFT)
        moon = get_moon(radius=conversion_factor * MOON_RADIUS)
        moon.move_to(moon_orbit.get_right())

        frame.add_updater(lambda m, dt: m.set_phi(interpolate(m.get_phi(), -90 * DEG, 0.025 * dt)))

        self.add(moon_orbit, moon)
        self.play(
            obj.animate.move_to(moon),
            frame.animate.set_height(1.5 * moon_orbit.get_width()).move_to(moon_orbit.get_right()).set_field_of_view(35 * DEG),
            FadeIn(moon_orbit),
            run_time=5
        )
        self.wait(5)

        # Show Venus
        sun = get_sun(SUN_RADIUS * conversion_factor, big_glow_ratio=20)
        sun.move_to(earth.get_center() + EARTH_ORBIT_RADIUS * conversion_factor * RIGHT)

        earth_orbit = Circle(radius=EARTH_ORBIT_RADIUS * conversion_factor)
        venus_orbit = Circle(radius=VENUS_ORBIT_RADIUS * conversion_factor)
        for orbit, color in zip([earth_orbit, venus_orbit], [BLUE, TEAL]):
            orbit.rotate(PI)
            orbit.set_stroke(color, width=(0, 3))
            orbit.move_to(sun)
            orbit.rotate(90 * DEG, LEFT)

        self.add(sun)
        self.play(
            frame.animate.set_height(0.4 * earth_orbit.get_width()).move_to(interpolate(venus_orbit.get_left(), sun.get_center(), 0.25)),
            FadeIn(earth_orbit, time_span=(2, 4)),
            FadeIn(venus_orbit, time_span=(2, 4)),
            obj.animate.move_to(venus_orbit.get_left()),
            run_time=8,
        )
        self.wait(4)
        frame.save_state()

        # Zoom back in
        frame.clear_updaters()
        if False:
            # This is for the transition to transit of Venus scene
            frame.clear_updaters()
            obs_lines.apply_depth_test()
            self.remove(line_between, angle_labels, pis, labels)
            # self.remove(obs_lines[1])
            self.play(
                frame.animate.reorient(-62, -2, 0, (4.64, 1.98, 2.86), 15.80),
                FadeOut(moon_orbit, time_span=(3, 4)),
                FadeOut(moon, time_span=(3, 4)),
                FadeOut(earth_orbit, time_span=(3, 4)),
                run_time=5,
                rate_func=lambda t: smooth(rush_from(t)),
            )
            self.play(frame.animate.reorient(0, 0, 0, (5.93, 0.25, 0.0), 15.86), run_time=5)
            self.wait()

            self.play(obs_dots[0].animate.move_to(obs_dots[1]), run_time=2)
            self.wait()

        self.play(
            frame.animate.reorient(0, 1, 0, (3.02, 0.82, -0.03), 15.80),
            FadeOut(moon_orbit, time_span=(3, 4)),
            FadeOut(moon, time_span=(3, 4)),
            FadeOut(earth_orbit, time_span=(3, 4)),
            run_time=4,
        )
        self.wait()
        self.add(angle_labels, obs_lines)
        self.play(
            Rotate(Group(pis[0], obs_dots[0]), 90 * DEG - angles[0], about_point=earth.get_center()),
            Rotate(Group(pis[1], obs_dots[1]), -90 * DEG - angles[1], about_point=earth.get_center()),
            MaintainPositionRelativeTo(labels[0], pis[0]),
            MaintainPositionRelativeTo(labels[1], pis[1]),
            UpdateFromFunc(line_between, lambda m: m.put_start_and_end_on(obs_dots[0].get_center(), obs_dots[1].get_center())),
            run_time=3
        )
        self.wait()

        # Slow zoom out
        self.play(
            frame.animate.reorient(-33, -9, 0, (20739.48, 3596.8, 5435.71), 33171.78),
            FadeIn(earth_orbit, time_span=(10, 12)),
            run_time=30,
            rate_func=lambda t: smooth(smooth(t))
        )

        # Zoom out to more of the solar system
        # frame.restore()

        new_orbits = VGroup(
            Circle(radius=r * conversion_factor)
            for r in [MERCURY_ORBIT_RADIUS, MARS_ORBIT_RADIUS, JUPITER_ORBIT_RADIUS, SATURN_ORBIT_RADIUS]
        )

        for orbit, color in zip(new_orbits, [GREY_B, RED, ORANGE, GREY_BROWN]):
            orbit.set_stroke(color, (0, 3))
            orbit.rotate(random.random() * TAU)
            orbit.rotate(90 * DEG, LEFT)
            orbit.move_to(sun)

        all_orbits = VGroup(new_orbits[0], venus_orbit, earth_orbit, *new_orbits[1:])
        periods = [
            MERCURY_ORBIT_PERIOD,
            VENUS_ORBIT_PERIOD,
            EARTH_ORBIT_PERIOD,
            MARS_ORBIT_PERIOD,
            JUPITER_ORBIT_PERIOD,
            SATURN_ORBIT_PERIOD,
        ]
        for orbit, period in zip(all_orbits, periods):
            orbit.period = period
            orbit.clear_updaters()
            orbit.add_updater(lambda m, dt: m.rotate(20 * dt / m.period, axis=UP))

        self.play(
            FadeIn(new_orbits, time_span=(0, 3)),
            frame.animate.reorient(-29, -41, 0, ORIGIN, 753807.38),
            celestial_sphere.animate.set_width(20 * JUPITER_ORBIT_RADIUS * conversion_factor),
            run_time=20
        )

    def get_angle_labels(
        self,
        obs_lines,
        obs_points,
        line_between,
        arc_props=[0.5, 0.5],
        arc_radius=0.5,
        colors=[BLUE, RED],
        backstroke_width=4,
    ):
        arc_radius = 0.5
        angle_syms = Tex(R"\alpha \beta")
        angle_syms.set_backstroke(BLACK, backstroke_width)
        colors = [BLUE, RED]
        angle_labels = VGroup()
        for obs_line, obs_point, angle_sym, arc_prop, color in zip(obs_lines, obs_points, angle_syms, arc_props, colors):
            obs_angle = obs_line.get_angle()
            line_angle = line_between.get_angle() + (PI if obs_angle > 0 else 0)
            arc = Arc(obs_angle, line_angle - obs_angle, arc_center=obs_point, radius=arc_radius)
            arc.set_stroke(color, 3)

            angle_sym.next_to(arc.pfp(arc_prop), arc.pfp(arc_prop) - obs_point)
            angle_sym.set_fill(color, border_width=1)

            angle_labels.add(VGroup(arc, angle_sym))

        angle_labels.set_stroke(behind=True)

        return angle_labels


class TransitOfVenus(InteractiveScene):
    path_y = -1
    include_image = False

    def construct(self):
        # Add image (just for development)
        if self.include_image:
            path = self.file_writer.get_output_file_rootname()
            im_path = Path(path.parent.parent, "Paul Animations/6. Transit Of Venus/New Transit of Venus scenes/JustSun.tif")
            image.set_height(FRAME_HEIGHT)
            self.add(image)

        # Add venus
        path = Line(3 * LEFT, 3 * RIGHT)
        path.set_y(self.path_y)
        path.set_stroke(BLACK, 2)

        venus = Dot(radius=0.05).set_fill(BLACK)
        venus.move_to(path.get_start())
        venus.set_fill(border_width=1)
        venus.set_anti_alias_width(5)
        self.add(venus)

        # Show transit
        venus.move_to(path.get_start())
        velocity = 0.25
        venus.clear_updaters()
        venus.add_updater(lambda m, dt: m.shift(dt * velocity * RIGHT))
        wait_time = 1.0
        copies = VGroup()
        self.add(copies)
        for _ in range(int(path.get_length() / velocity / wait_time)):
            self.wait(wait_time)
            copies.add(venus.copy().clear_updaters())

        self.remove(venus)
        self.play(Transform(copies, VGroup(path)))
        self.wait()


class TransitOfVenusHigher(TransitOfVenus):
    path_y = +0.5


class TransitOfVenusSlightlyHigher(TransitOfVenus):
    path_y = -0.9


class TransitOfVenusMiddle(TransitOfVenus):
    path_y = 0


class NearbyStars(InteractiveScene):
    def construct(self):
        # Add sun and earth
        orbit_radius = 3.5
        conversion_factor = orbit_radius / EARTH_ORBIT_RADIUS

        sun = get_sun(radius=conversion_factor * SUN_RADIUS, big_glow_ratio=20)
        sun.center()
        orbit = Circle(radius=orbit_radius)
        orbit.set_stroke(BLUE, (0, 4))
        earth_glow = GlowDot(color=BLUE)
        earth_glow.f_always.move_to(orbit.get_start)

        celestial_sphere = get_celestial_sphere(constellation_opacity=0)
        celestial_sphere[0].set_opacity(1)

        self.add(celestial_sphere, sun, orbit, earth_glow)

        # Show the astronomical unit
        dist_line = Line()
        dist_line.set_stroke(WHITE, 1)
        dist_line.f_always.put_start_and_end_on(sun.get_center, orbit.get_start)

        dist_label = Text("Astronomical\nUnit", font_size=36)
        dist_label.f_always.move_to(
            lambda: dist_line.get_center() + 0.5 * normalize(rotate_vector(dist_line.get_vector(), 90 * DEG))
        )

        self.play(
            FadeIn(dist_line, time_span=(0, 1)),
            FadeIn(dist_label, time_span=(0, 1)),
            Rotate(orbit, TAU, about_point=ORIGIN, rate_func=linear, run_time=10),
        )
        self.wait()

        # Transition to initials
        dist_label.clear_updaters()
        au_label = Text("A.U.", font_size=36)

        def update_au_label(label):
            point = dist_line.get_center()
            direction = normalize(rotate_vector(point, 90 * DEG))
            step = 0.65 * interpolate(label.get_width(), label.get_height(), abs(direction[1]))
            label.move_to(point + step * direction)

        au_label.add_updater(update_au_label)

        self.play(LaggedStart(
            *(
                ReplacementTransform(dist_label[t2][0], au_label[t1][i])
                for t1, t2, i in zip("A.U.", ["A", "stronomical", "U", "nit"], [0, 0, 0, 1])
            ),
            lag_ratio=0.2
        ))
        self.add(au_label)

        # Position to the side
        frame = self.frame
        self.play(
            Rotate(orbit, 90 * DEG),
            frame.animate.reorient(0, 0, 0, 7 * RIGHT, 14),
            run_time=2
        )

        # Zoom into and out of earth real quick
        frame.save_state()
        earth = get_earth(radius=orbit_radius * (EARTH_RADIUS / EARTH_ORBIT_RADIUS))
        earth.move_to(earth_glow)
        earth.rotate(EARTH_TILT_ANGLE, RIGHT)
        frame.move_to(earth)
        frame.set_height(2 * earth.get_height())
        frame.reorient(-74, 79, 0)
        self.camera.light_source.move_to(sun)

        self.remove(earth_glow, orbit, dist_line)
        self.add(earth)
        self.wait()
        srf = squish_rate_func(smooth, 0.7, 1)
        self.play(
            UpdateFromAlphaFunc(frame, lambda m, a: m.reorient(
                *interpolate(np.array([-74, 79, 0]), np.zeros(3), a),
                interpolate(earth.get_center(), 7 * RIGHT, srf(a)),
                np.exp(interpolate(np.log(2 * earth.get_height()), np.log(14), smooth(a))),
            ), run_time=5),
            FadeIn(earth_glow, time_span=(2.5, 4.5)),
            FadeIn(orbit, time_span=(1, 4)),
            FadeIn(dist_line, time_span=(1, 4)),
            FadeIn(au_label, time_span=(4, 5)),
            FadeOut(earth),
            run_time=5,
        )

        # Show observations
        star = Group(
            ImageMobject('StarFourPoints').set_height(0.8).center(),
            GlowDot(color=WHITE).center()
        )
        star[1].add_updater(lambda m: m.set_width(0.4 * ((1 + math.sin(1.5 * self.time)))))
        star.move_to(50 * RIGHT)
        obs_points = Group(
            TrueDot(point, radius=0.1).set_color(GREEN).make_3d()
            for point in [orbit.get_top(), orbit.get_bottom()]
        )
        obs_lines = VGroup(
            self.get_obs_line(obs_point, star)
            for obs_point in obs_points
        )
        obs_lines.set_stroke(WHITE, 2)
        for line, point in zip(obs_lines, obs_points):
            line.start_point = point
            line.star = star
            line.add_updater(lambda m: m.put_start_and_end_on(m.start_point.get_center(), m.star.get_center()))

        obs_labels = VGroup(Text(f"Observation {n}") for n in [1, 2])
        for label, point, vect in zip(obs_labels, obs_points, [UP, DOWN]):
            label.next_to(point, vect, MED_SMALL_BUFF)

        self.add(star)

        self.play(
            ShowCreation(obs_lines[0], suspend_mobject_updating=True),
            FadeIn(obs_labels[0], 0.25 * UP),
            FadeIn(obs_points[0]),
        )
        self.wait()
        self.play(Rotate(orbit, PI), run_time=2)
        self.play(
            ShowCreation(obs_lines[1], suspend_mobject_updating=True),
            FadeIn(obs_labels[1], DOWN),
            FadeIn(obs_points[1]),
        )
        self.wait()

        # Show the angle vary during the orbit
        self.play(
            star.animate.move_to(15 * RIGHT),
            run_time=2
        )
        self.wait()

        obs_lines.suspend_updating()
        sample_obs_line = self.get_obs_line(earth_glow, star)
        self.play(
            FadeIn(sample_obs_line),
            obs_lines.animate.set_stroke(opacity=0.1)
        )
        self.play(Rotate(orbit, PI, run_time=10))
        self.wait()
        self.play(
            FadeOut(sample_obs_line),
            obs_lines.animate.set_stroke(opacity=1),
        )

        # Pull it far away, then back
        curr_center = star.get_center()
        curr_angle = obs_lines[1].get_angle() - obs_lines[0].get_angle()
        orbit_radius / math.tan(curr_angle / 2)

        obs_lines.resume_updating()
        self.play(
            UpdateFromAlphaFunc(star, lambda m, a: m.move_to(
                RIGHT * orbit_radius / math.tan(interpolate(curr_angle, 1e-5, there_and_back_with_pause(a)) / 2)
            )),
            run_time=6,
        )

        # Label the distance and angle
        line_to_star = Line(sun.get_center(), star.get_center())
        line_to_star.set_stroke(RED, 3)
        dist_label = Tex("D", font_size=60)
        dist_label.next_to(line_to_star, UP, buff=2 * SMALL_BUFF)
        dist_label.match_color(line_to_star)

        arc = Arc(PI, -curr_angle / 2, arc_center=star.get_center(), radius=3)
        arc_label = Tex(R"\theta / 2", font_size=60)
        arc_label.next_to(arc, LEFT, buff=SMALL_BUFF)

        self.play(
            ShowCreation(line_to_star),
            obs_lines.animate.set_stroke(width=1),
            FadeIn(dist_label, RIGHT),
        )
        self.wait()
        self.play(
            ShowCreation(arc),
            Write(arc_label),
        )
        self.play(FlashAround(arc_label, run_time=2))
        self.wait()
        self.play(
            Transform(obs_lines[0].copy().clear_updaters(), obs_lines[1].copy(), remover=True),
            run_time=2
        )
        self.wait()

        # Write the tangent equation
        kw = dict(
            t2c={R"\text{A.U.}": BLUE, "D": RED},
            font_size=72
        )
        eq1, eq2 = equations = VGroup(
            Tex(R"\tan\left(\theta / 2\right) = {\text{A.U.} \over D}", **kw),
            Tex(R"\theta = 2 \cdot \tan^{-1}\left({\text{A.U.} \over D}\right)", **kw),
        )
        equations.arrange(DOWN, buff=LARGE_BUFF)
        equations.next_to(frame.get_top(), DOWN, buff=-0.5)
        equations.align_to(dist_label, LEFT)

        self.play(LaggedStart(
            frame.animate.shift(UP),
            Write(eq1[R"\tan\left("]),
            FadeTransform(arc_label.copy(), eq1[R"\theta / 2"][0]),
            Write(eq1[R"\right) = "]),
            FadeTransform(au_label.copy().clear_updaters(), eq1["A.U."][0]),
            Write(eq1[R"\over"]),
            FadeTransform(dist_label.copy(), eq1["D"][0]),
            lag_ratio=0.25,
            run_time=3
        ))
        self.wait()
        self.play(TransformMatchingTex(eq1.copy(), eq2, path_arc=90 * DEG, run_time=2))
        self.wait()

        # Throw in Proxima Centauri numbers
        ac_labels = VGroup(
            Text(text, font_size=60, t2c={"D": RED, "A.U.": BLUE})
            for text in ["Proxima Centauri", "D = 40.17 trillion km", "D = 268,553 A.U."]
        )
        for label in ac_labels:
            label.add_background_rectangle()
        ac_labels.arrange(DOWN, aligned_edge=LEFT, buff=MED_LARGE_BUFF)
        ac_labels.next_to(star, DOWN, aligned_edge=LEFT, buff=0).shift(0.5 * LEFT)
        ac_labels[2][0].set_opacity(0)

        for label in ac_labels:
            self.play(Write(label), frame.animate.set_x(8.5), run_time=2)
            self.wait()

        # Plug it in
        shift_value = 2 * LEFT + 2 * UP
        rhs = Tex(R"= 2 \cdot \tan^{-1}\left(1 \over 268{,}553 \right)", font_size=72)
        rhs.next_to(eq2, RIGHT)
        rhs.shift(shift_value)

        answer = Tex(R"=0.000413^\circ", font_size=72)
        answer.next_to(rhs, RIGHT)

        answer_in_arc_seconds = Tex(R"\approx 1.5 \text{ arc-seconds}", font_size=72)
        answer_in_arc_seconds.next_to(answer, DOWN, LARGE_BUFF, aligned_edge=LEFT)

        for tex in [answer, answer_in_arc_seconds]:
            tex.add_background_rectangle()

        self.play(LaggedStart(
            equations.animate.shift(shift_value),
            frame.animate.move_to(11 * RIGHT + 3 * UP).set_height(16),
            *(
                TransformFromCopy(eq2[tex][0], rhs[tex][0])
                for tex in [R"2 \cdot \tan^{-1}\left(", R"\right)"]
            ),
            FadeIn(rhs[R"1 \over"]),
            FadeIn(rhs[R"="]),
            FadeTransform(ac_labels[2]["268,553"].copy(), rhs["268{,}553"].copy()),
            run_time=2,
            lag_ratio=0.1,
        ))
        self.wait()
        self.play(Write(answer))
        self.wait()
        self.play(FadeIn(answer_in_arc_seconds, DOWN))
        self.wait()

        # Fade out and push star away
        self.play(LaggedStartMap(
            FadeOut,
            VGroup(line_to_star, dist_label, arc, arc_label, *ac_labels),
            shift=0.1 * DOWN,
            lag_ratio=0.25
        ))

        obs_lines.resume_updating()
        self.play(
            star.animate.move_to(1000 * RIGHT),
            rate_func=lambda t: t**4,
            run_time=5
        )

    def get_obs_line(self, obj1, obj2, dash_length=0.1, stroke_color=WHITE, stroke_width=2):
        # line = DashedLine(obj1.get_center(), obj2.get_center())
        line = Line(obj1.get_center(), obj2.get_center())
        line.set_stroke(stroke_color, stroke_width)
        line.f_always.put_start_and_end_on(obj1.get_center, obj2.get_center)
        return line
