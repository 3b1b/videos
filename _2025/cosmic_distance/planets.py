from manim_imports_ext import *


EARTH_TILT_ANGLE = 23.3 * DEG

# All in Kilometers
EARTH_RADIUS = 6_371
MOON_RADIUS = 1_737.4
MOON_ORBIT_RADIUS = 384_400
SUN_RADIUS = 695_700

MERCURY_ORBIT_RADIUS = 6.805e7
VENUS_ORBIT_RADIUS = 1.082e8
EARTH_ORBIT_RADIUS = 1.473e8
MARS_ORBIT_RADIUS = 2.280e8
CERES_ORBIT_RADIUS = 4.130e8
JUPITER_ORBIT_RADIUS = 7.613e8
SATURN_ORBIT_RADIUS = 1.439e9

# In days
MERCURY_ORBIT_PERIOD = 87.97
VENUS_ORBIT_PERIOD = 224.7
EARTH_ORBIT_PERIOD = 365.25
MARS_ORBIT_PERIOD = 686.98
JUPITER_ORBIT_PERIOD = 4332.82
SATURN_ORBIT_PERIOD = 10755.7

# In km / s
SPEED_OF_LIGHT = 299792


def get_earth(radius=1.0, day_texture="EarthTextureMap", night_texture="NightEarthTextureMap"):
    sphere = Sphere(radius=radius)
    earth = TexturedSurface(sphere, day_texture, night_texture)
    return earth


def get_sphere_mesh(radius=1.0):
    sphere = Sphere(radius=radius)
    mesh = SurfaceMesh(sphere)
    mesh.set_stroke(WHITE, 0.5, 0.5)
    return mesh


def get_moon(radius=1.0, resolution=(101, 51)):
    moon = TexturedSurface(Sphere(radius=radius, resolution=resolution), "MoonTexture", "DarkMoonTexture")
    moon.set_shading(0.25, 0.25, 1)
    return moon


def get_sun(
    radius=1.0,
    near_glow_ratio=2.0,
    near_glow_factor=2,
    big_glow_ratio=4,
    big_glow_factor=1,
    big_glow_opacity=0.35,
):
    sun = TexturedSurface(Sphere(radius=radius), "SunTexture")
    sun.set_shading(0, 0, 0)
    sun.to_edge(LEFT)

    # Glows
    near_glow = GlowDot(radius=near_glow_ratio * radius, glow_factor=near_glow_factor)
    near_glow.move_to(sun)

    big_glow = GlowDot(radius=big_glow_ratio * radius, glow_factor=big_glow_factor, opacity=big_glow_opacity)
    big_glow.move_to(sun)

    return Group(sun, near_glow, big_glow)


def get_planet(name, radius=1.0):
    planet = TexturedSurface(Sphere(radius=radius), f"{name}Texture", f"Dark{name}Texture")
    planet.set_shading(0.25, 0.25, 1)
    return planet


def get_celestial_sphere(radius=1000, constellation_opacity=0.1):
    sphere = Group(
        TexturedSurface(Sphere(radius=radius, clockwise=True), "hiptyc_2020_8k"),
        TexturedSurface(Sphere(radius=0.99 * radius, clockwise=True), "constellation_figures"),
    )
    sphere.set_shading(0, 0, 0)
    sphere[1].set_opacity(constellation_opacity)

    sphere.rotate(EARTH_TILT_ANGLE, RIGHT)

    return sphere


def get_planet_symbols(text, font_size=48):
    return Tex(
        Rf"\{text}",
        additional_preamble=R"\usepackage{wasysym}",
        font_size=font_size,
    )


class PerspectivesOnEarth(InteractiveScene):
    def construct(self):
        # Ask about size
        light = self.camera.light_source
        light.move_to(50 * LEFT)
        frame = self.frame
        frame.set_field_of_view(25 * DEG)

        conversion_factor = 1.0 / EARTH_RADIUS

        earth = get_earth(radius=EARTH_RADIUS * conversion_factor)
        earth.rotate(-EARTH_TILT_ANGLE, UP)
        earth_axis = rotate_vector(OUT, -EARTH_TILT_ANGLE, UP)

        earth.add_updater(lambda m, dt: m.rotate(dt * 10 * DEG, axis=earth_axis))

        self.add(earth)

        # Clearly show the size of the earth
        brace = Brace(earth, LEFT)
        brace.stretch(0.5, 1, about_edge=UP)
        label = brace.get_tex(Rf"R_E", font_size=24)
        VGroup(brace, label).rotate(90 * DEG, RIGHT, about_point=brace.get_bottom())

        dashed_lines = VGroup(
            DashedLine(brace.get_corner(OUT + RIGHT), earth.get_zenith(), dash_length=0.02),
            DashedLine(brace.get_corner(IN + RIGHT), earth.get_center(), dash_length=0.02),
        )
        dashed_lines.set_stroke(WHITE, 2)

        frame.reorient(0, 90, 0, ORIGIN, 3.42)

        self.play(
            GrowFromCenter(brace),
            Write(label),
            *map(ShowCreation, dashed_lines),
        )
        self.wait(4)
        self.play(LaggedStartMap(FadeOut, VGroup(label, brace, *dashed_lines), shift=IN))

        # Make it wobble a bit
        earth.suspend_updating()

        def wave(x):
            return math.sin(2 * x) * np.exp(-x * x)

        def homotopy(x, y, z, t):
            t = 3 * (2 * t - 1)
            return (x + 0.1 * wave(t - z), y, z + 0.1 * wave(t - x))

        self.play(Homotopy(homotopy, earth, run_time=5))

        # Turn it into a disk
        earth.save_state()
        flat_earth = TexturedSurface(
            ParametricSurface(
                lambda u, v: ((1 - v) * math.cos(u), (1 - v) * math.sin(u), 0),
                u_range=(0, TAU),
                v_range=(0, 1),
                resolution=earth.resolution
            ),
            "EarthTextureMap",
            "NightEarthTextureMap",
        )

        rot = rotation_matrix_transpose(40 * DEG, (-1, 0, -0.5)).T
        flat_earth = earth.copy().apply_matrix(rot).stretch(1e-2, 2)
        flat_earth.data["d_normal_point"] = flat_earth.get_points() + 1e-3 * OUT

        self.play(
            Transform(earth, flat_earth, run_time=4),
            frame.animate.reorient(0, 74, 0, (-0.03, -0.14, 0.04), 3.42),
            light.animate.shift(10 * OUT + 10 * LEFT),
            run_time=2
        )
        self.play(Rotate(earth, PI / 3, axis=RIGHT, run_time=3, rate_func=wiggle))
        self.wait()

        # Zoom out to the moon
        orbit = Circle(radius=MOON_ORBIT_RADIUS * conversion_factor)
        orbit.set_stroke(GREY_C, width=(0, 3))
        orbit.rotate(-45 * DEG)
        orbit.add_updater(lambda m, dt: m.rotate(5 * dt * DEG))

        moon = get_moon(radius=MOON_RADIUS * conversion_factor)
        moon.to_edge(RIGHT)
        moon.add_updater(lambda m: m.move_to(orbit.get_start()))

        self.add(orbit, moon)
        self.play(
            Restore(earth, time_span=(0, 3)),
            frame.animate.reorient(0, 0, 0, ORIGIN, 1.1 * orbit.get_height()),
            run_time=4,
        )

        interp_factor = ValueTracker(0)
        frame.add_updater(lambda m: m.move_to(interpolate(m.get_center(), moon.get_center(), interp_factor.get_value())))
        self.play(
            frame.animate.reorient(-67, 60, 0,).set_height(3),
            interp_factor.animate.set_value(0.2),
            run_time=8
        )
        self.wait(4)


class SphericalEarthVsFlat(InteractiveScene):
    def construct(self):
        # Add earth
        light = self.camera.light_source
        light.move_to(50 * LEFT)
        frame = self.frame
        frame.set_field_of_view(25 * DEG)
        earth = get_earth(radius=3)
        earth.rotate(-EARTH_TILT_ANGLE, UP)
        earth_axis = rotate_vector(OUT, -EARTH_TILT_ANGLE, UP)
        self.add(earth)

        frame.reorient(160, 75, 0, (-0.53, -0.33, 1.91), 1.84)
        self.play(
            frame.animate.reorient(166, 63, 0, ORIGIN, FRAME_HEIGHT),
            run_time=2
        )
        self.wait()
        self.play(frame.animate.to_default_state(), run_time=2)
        self.wait()

        # Look around
        self.look_from_various_angles()

        # Squish
        day_earth = get_earth(radius=earth.get_height() / 2, night_texture="EarthTextureMap")
        flat_earth = day_earth.copy().set_depth(1e-2, stretch=True)
        flat_earth.data["d_normal_point"] = flat_earth.data["point"] + 1e-3 * OUT
        self.play(FadeOut(earth, scale=0.99), FadeIn(day_earth, scale=0.99))
        self.play(
            Transform(day_earth, flat_earth),
            light.animate.set_z(10),
            run_time=2
        )

        # Ellipse
        circle = Circle(radius=3)
        circle.set_stroke(TEAL, 3)
        self.play(frame.animate.reorient(10, 80, 0), run_time=2)
        self.play(ShowCreation(circle))
        self.wait()

    def look_from_various_angles(self, run_time=1.5):
        frame = self.frame
        orientations = [
            (10, 80, 0),
            (-150, 129, 0),
            (97, 55, 0),
            (0, 0, 0)
        ]
        for orientation in orientations:
            self.play(frame.animate.reorient(*orientation), run_time=run_time)
            self.wait()


class SizeOfEarthRenewed(InteractiveScene):
    radius = 3.0

    def construct(self):
        # Setup
        self.set_floor_plane("xz")
        frame = self.frame
        frame.set_field_of_view(15 * DEG)

        light = self.camera.light_source
        light.move_to(20 * RIGHT)

        # Add earth
        sphere = Sphere(radius=self.radius)
        earth = get_earth(radius=self.radius)
        mesh = get_sphere_mesh(radius=self.radius)
        mesh.rotate(-2 * DEG)
        transparent_earth = earth.copy()
        transparent_earth.set_opacity(0.25)

        inner_shell = sphere.copy()
        inner_shell.set_color(GREY_E)
        inner_shell.set_height(0.99 * 2 * self.radius)

        earth_group = Group(inner_shell, earth, mesh, transparent_earth)
        earth_group.rotate(90 * DEG, LEFT)

        slice_tracker = ValueTracker(self.radius + SMALL_BUFF)
        earth.add_updater(lambda m: m.set_clip_plane(OUT, slice_tracker.get_value()))
        inner_shell.add_updater(lambda m: m.set_clip_plane(OUT, slice_tracker.get_value()))
        mesh.add_updater(lambda m: m.set_clip_plane(IN, -slice_tracker.get_value()))
        transparent_earth.add_updater(lambda m: m.set_clip_plane(IN, -slice_tracker.get_value()))

        circle = Circle(radius=self.radius)

        earth_axis = rotate_vector(UP, -EARTH_TILT_ANGLE)

        axis_line = Line(DOWN, UP).set_height(8)
        axis_line.set_stroke(WHITE, 1)

        earth_group.rotate(147 * DEG, UP)
        earth_group.rotate(-EARTH_TILT_ANGLE, OUT)

        self.add(earth)

        # Unflatten earth
        earth.save_state()
        earth.stretch(1e-3, 0)
        earth.data["d_normal_point"] = earth.get_points() + 1e-3 * RIGHT
        earth.note_changed_data()

        frame.reorient(5, 0, -90, 2 * RIGHT)

        self.play(
            frame.animate.reorient(0, 0, -90, RIGHT),
            Restore(earth),
            run_time=3,
        )

        # Add rays from the sun
        sun = GlowDot(100 * RIGHT, radius=1)
        n_rays = 25
        rays = Line(LEFT, RIGHT).replicate(n_rays)
        rays.set_stroke(YELLOW, 1)

        def update_rays(rays):
            ys = np.linspace(earth.get_y(UP), earth.get_y(DOWN), len(rays))
            for ray, y in zip(rays, ys):
                ray.put_start_and_end_on(
                    sun.get_center(),
                    [math.sqrt(abs(self.radius**2 - y**2)), y, 0],
                )

        rays.add_updater(update_rays)
        rays.set_z_index(-1)

        self.play(
            FadeIn(rays, shift=0.5 * LEFT, lag_ratio=0.02),
            run_time=2
        )
        self.play(
            frame.animate.reorient(-67, -7, 0, (0.92, 2.13, 3.94), 20.25),
            sun.animate.move_to(50 * RIGHT),
            run_time=3,
        )
        self.play(
            sun.animate.move_to(1000 * RIGHT).set_anim_args(time_span=(0, 3)),
            frame.animate.reorient(-130, -14, 0, (2.66, 0.14, -0.69), 8.18).set_anim_args(time_span=(2, 7)),
            run_time=7,
        )

        # Show line through Syene
        center_dot = GlowDot(color=RED)
        slice_tracker.set_value(self.radius + SMALL_BUFF)
        angle = 7 * DEG

        h_line = Line(ORIGIN, 20 * RIGHT)
        h_line.set_stroke(YELLOW, 2)

        self.add(earth_group)
        self.play(
            slice_tracker.animate.set_value(0),
            FadeIn(center_dot, scale=0.25),
            run_time=2
        )
        self.play(
            ShowCreation(h_line, rate_func=rush_into, run_time=2),
            FadeOut(rays, run_time=2),
            frame.animate.reorient(-205, -11, 0, (3.0, -0.0, 0.0), 6.50).set_anim_args(run_time=10),
        )

        # Rotate the earth about a bit
        earth_group.save_state()
        self.play(Rotate(earth_group, 40 * DEG, axis=OUT, run_time=3))
        self.play(Rotate(earth_group, 40 * DEG, axis=UP, run_time=3, rate_func=there_and_back))
        self.play(Rotate(earth_group, -40 * DEG, axis=OUT, run_time=3))

        # Emphasize the tilt of the earth's axis
        axis_line = Line(-1.25 * self.radius * earth_axis, 1.25 * self.radius * earth_axis)
        axis_line.set_stroke(WHITE, 2)
        self.play(
            Rotate(earth_group, 2 * TAU, axis=earth_axis),
            FadeIn(axis_line, time_span=(0, 1)),
            FadeIn(rays, remover=True, time_span=(4, 15), rate_func=lambda t: there_and_back_with_pause(t, 9 / 11)),
            frame.animate.reorient(-91, -36, 0, (4.91, -1.44, 0.39), 11.01).set_anim_args(time_span=(4, 15), rate_func=lambda t: there_and_back_with_pause(t, 0.2)),
            run_time=15,
        )

        # Show tropic of cancer
        def get_lat_line(angle):
            result = Circle(radius=self.radius)
            result.rotate(90 * DEG, LEFT).rotate(EARTH_TILT_ANGLE, axis=IN)
            result.set_stroke(TEAL, 2)
            result.apply_depth_test()
            result.scale(math.cos(angle))
            result.shift(math.sin(angle) * earth_axis * self.radius)
            return result

        equator_label = Text("Equator")
        equator_label.flip().rotate(EARTH_TILT_ANGLE, IN)
        equator_label.move_to(self.radius * IN + 0.15 * earth_axis)
        equator_label.set_backstroke()

        cancer_label = Text("Tropic of Cancer")
        cancer_label.flip().rotate(EARTH_TILT_ANGLE, IN)
        cancer_label.move_to(op.add(
            math.cos(EARTH_TILT_ANGLE) * self.radius * IN,
            (math.sin(EARTH_TILT_ANGLE) * self.radius + 0.15) * earth_axis,
        ))

        tropic_of_cancer = get_lat_line(EARTH_TILT_ANGLE)

        d_line = h_line.copy().rotate(angle, about_point=ORIGIN)
        d_line.set_stroke(PINK)

        alex_point = earth.get_center() + self.radius * normalize(d_line.get_vector())
        alex_name = Text("Alexandria", font_size=36).flip()
        alex_name.next_to(alex_point, UR, buff=SMALL_BUFF).shift(0.45 * UP)
        alex_arrow = Arrow(alex_name.get_bottom() + 0.35 * LEFT, alex_point, buff=SMALL_BUFF, thickness=2)

        syene_point = earth.get_right()
        syene_name = Text("Syene", font_size=36).flip()
        syene_name.next_to(syene_point, DR, buff=MED_SMALL_BUFF).shift(0.25 * DOWN)
        syene_arrow = Arrow(syene_name.get_top(), syene_point, buff=SMALL_BUFF, thickness=2)

        self.play(
            Rotate(earth_group, TAU, axis=earth_axis),
            ShowCreation(tropic_of_cancer),
            axis_line.animate.set_stroke(opacity=0.5).set_anim_args(time_span=(0, 1)),
            run_time=12,
        )

        self.play(
            frame.animate.reorient(-172, 0, 0, (3.05, -0.01, -0.01), 6.69),
            Write(cancer_label, time_span=(1, 3)),
            run_time=3
        )
        self.wait()
        self.play(
            Write(syene_name),
            GrowArrow(syene_arrow),
        )
        self.play(
            frame.animate.reorient(-225, -8, 0, (3.05, -0.01, -0.01), 6.69),
            FadeOut(cancer_label, time_span=(0, 2)),
            run_time=4
        )
        self.wait()

        # Show line through Alexandria
        self.play(
            ShowCreation(d_line),
            frame.animate.reorient(-189, 6, 0, (3.05, -0.01, -0.01), 6.69),
            run_time=3,
        )
        self.play(
            Write(alex_name),
            GrowArrow(alex_arrow),
        )
        self.wait()

        # Show the angle
        earth_point = circle.pfp(angle / TAU)
        upper_ray = Line(earth_point, earth_point + 20 * RIGHT)
        upper_ray.match_style(h_line)

        arc = Arc(0, 2 * angle, radius=3, arc_center=earth_point)
        arc.scale(1 / 2, about_edge=DOWN)
        arc_labels = VGroup()
        for tex in [R"\theta", R"7^{\circ}"]:
            arc_label = Tex(tex, font_size=36)
            arc_label.flip()
            arc_label.next_to(arc, RIGHT, SMALL_BUFF)
            arc_labels.add(arc_label)

        arc_label = arc_labels[0]

        self.play(
            FadeIn(upper_ray),
            ShowCreation(arc),
            FadeIn(arc_label),
        )
        self.wait()
        self.play(Transform(arc_label, arc_labels[1]))
        self.wait()
        self.play(
            arc.animate.shift(earth_point - arc.get_end()).shift(0.01 * LEFT).set_stroke(RED),
            arc_label.animate.next_to(earth_point, DR, buff=0.05),
            upper_ray.animate.set_stroke(width=1, opacity=0.5),
            run_time=2
        )
        self.wait()
        self.play(LaggedStartMap(FadeOut, VGroup(alex_name, alex_arrow, syene_name, syene_arrow)))

        # Show full circumference
        self.play(
            frame.animate.reorient(-173, 0, 0, (2.91, 0.1, -0.01), 7.20),
            ShowCreation(circle),
            run_time=4,
        )

        # Ambient panning
        circle.apply_depth_test()
        self.play(
            FadeOut(arc_label, time_span=(3, 5)),
            frame.animate.reorient(110 - 360, -10, 0, (-0.41, -0.16, 3.6), 7.86),
            run_time=24,
        )

    def old_material(self):
        # Tilt earth
        v_line = axis_line.copy()
        d_line = axis_line.copy()

        arc = Arc(90 * DEG, -EARTH_TILT_ANGLE, radius=1.75)
        arc.set_stroke(WHITE, 2)
        arc_label = Tex(R"23.5^\circ", font_size=36)
        arc_label.next_to(arc.pfp(0.65), UP, buff=0.15)

        self.play(
            Rotate(earth_group, -EARTH_TILT_ANGLE, axis=OUT),
            Rotate(axis_line, -EARTH_TILT_ANGLE, axis=OUT),
            Rotate(d_line, -EARTH_TILT_ANGLE, axis=OUT),
            FadeIn(v_line),
            VFadeIn(d_line),
            ShowCreation(arc),
            Write(arc_label),
            run_time=2
        )
        self.wait()
        self.add(rays, axis_line, earth_group)
        self.play(
            FadeOut(VGroup(v_line, d_line, arc, arc_label), time_span=(0, 1)),
            frame.animate.reorient(-138, -25, 0, (4.72, -0.66, -0.11), 10.77),
            run_time=12
        )


class AlBiruniEarthMeasurement(InteractiveScene):
    def construct(self):
        # Add earth and mountain
        radius = 2
        height = 0.3

        earth = Circle(radius=radius)
        earth.set_fill(BLUE_B, 0.5)
        earth.set_stroke(WHITE, 3)
        earth.rotate(90 * DEG)

        earth_pattern = SVGMobject("earth")
        earth_pattern.rotate(90 * DEG)
        earth_pattern.replace(earth)
        earth_pattern.set_fill(Color(hsl=(0.23, 0.5, 0.2)), 1)

        mountain_tip = earth.get_top() + height * UP
        mountain = Polyline(
            earth.pfp(0.02), mountain_tip, earth.pfp(0.98)
        )
        mountain.set_stroke(GREY_B, 4)

        self.add(earth)
        self.add(earth_pattern)
        self.add(mountain)

        # Show line of sight
        theta = math.acos(radius / (radius + height))
        line_length = 3

        line_of_sight = DashedLine(ORIGIN, line_length * RIGHT, dash_length=DEFAULT_DASH_LENGTH / 2)
        line_of_sight.set_stroke(WHITE, 2)
        line_of_sight.rotate(-theta, about_point=ORIGIN)
        line_of_sight.shift(mountain_tip)

        horizontal = Line(mountain_tip, mountain_tip + line_length * RIGHT)
        horizontal.set_stroke(WHITE, 2)
        horizontal_copy = horizontal.copy()

        arc = Arc(0, -theta, arc_center=mountain_tip, radius=0.5)
        theta_label = Tex(R"\theta", font_size=24)
        theta_label.next_to(arc, RIGHT, SMALL_BUFF)

        self.play(ShowCreation(line_of_sight))
        self.wait()
        self.play(ShowCreation(horizontal))
        self.play(
            Rotate(horizontal_copy, -theta, about_point=mountain_tip),
            ShowCreation(arc),
            Write(theta_label)
        )
        self.play(FadeOut(horizontal_copy))
        self.wait()

        # Show radius of the earth
        radius_line = Line(earth.get_center(), earth.get_top())
        radius_line.rotate(-theta, about_point=earth.get_center())
        radius_line.set_stroke(WHITE, 4)
        R_label = Tex(R"R", font_size=36)
        R_label.next_to(radius_line.get_center(), DR, buff=0.05)

        self.play(
            earth_pattern.animate.set_fill(opacity=0.5),
            earth.animate.set_fill(opacity=0.2),
            ShowCreation(radius_line),
            Write(R_label),
        )
        self.wait()

        # Show triangle
        elbow = Elbow(angle=180 * DEG - theta)
        elbow.shift(radius_line.get_end())
        hyp = Line(earth.get_center(), mountain_tip)
        hyp.set_stroke(RED, 3)
        hyp_brace = Brace(hyp, LEFT, buff=0.1)
        hyp_label = hyp_brace.get_tex("R + h", font_size=36)

        self.play(
            ShowCreation(elbow),
            ShowCreation(hyp),
        )
        self.wait()
        self.play(
            GrowFromCenter(hyp_brace),
            Write(hyp_label),
        )
        self.wait()

        # Show angle
        low_arc = Arc(90 * DEG, -theta, arc_center=earth.get_center(), radius=0.5)
        low_theta_label = theta_label.copy()
        low_theta_label.next_to(low_arc.pfp(0.6), UP, buff=0.075)

        self.play(FlashAround(theta_label))
        self.play(
            TransformFromCopy(arc, low_arc),
            TransformFromCopy(theta_label, low_theta_label),
        )
        self.wait()

        # Show the equation
        frame = self.frame
        equation = Tex(R"R = (R + h)\cos(\theta)", font_size=36)
        equation.to_edge(UP, buff=0)

        self.play(
            frame.animate.shift(UP),
            LaggedStart(
                TransformFromCopy(R_label, equation["R"][0]),
                Write(equation["= ("][0]),
                TransformFromCopy(hyp_label, equation["R + h"][0]),
                Write(equation[R")\cos("][0]),
                TransformFromCopy(low_theta_label, equation[R"\theta"][0]),
                Write(equation[")"][-1]),
                lag_ratio=0.25
            )
        )
        self.wait()

        # Simplify
        eq2 = Tex(R"R - R \cos(\theta) = h \cos(\theta)", font_size=36)
        eq3 = Tex(R"R = {h\cos(\theta) \over 1 - \cos(\theta)}", font_size=36)
        eq2.move_to(equation).shift(0.5 * UP)
        eq3.next_to(eq2, DOWN)
        rect = SurroundingRectangle(eq3)
        rect.set_fill(BLACK, 0.8)

        self.play(equation.animate.next_to(eq2, UP), frame.animate.shift(0.5 * UP))
        self.play(TransformMatchingTex(equation.copy(), eq2, path_arc=45 * DEG, lag_ratio=0.01))
        self.wait()
        self.play(TransformMatchingTex(eq2.copy(), eq3, path_arc=45 * DEG, lag_ratio=0.01))
        self.add(rect, eq3)
        self.play(DrawBorderThenFill(rect))
        self.wait()


class LuarEclipse(InteractiveScene):
    def construct(self):
        # Setup earth and moon
        light = self.camera.light_source
        light.move_to(50 * LEFT)
        frame = self.frame
        frame.set_field_of_view(25 * DEG)

        conversion_factor = 1.0 / EARTH_RADIUS

        earth = get_earth(radius=EARTH_RADIUS * conversion_factor)
        earth.rotate(-EARTH_TILT_ANGLE, UP)
        earth_axis = rotate_vector(OUT, -EARTH_TILT_ANGLE, UP)

        earth.add_updater(lambda m, dt: m.rotate(dt * 10 * DEG, axis=earth_axis))

        self.add(earth)

        # Clearly show the size of the earth
        brace = Brace(earth, LEFT)
        label = brace.get_tex(Rf"2 R_E")
        re_label = Integer(2 * EARTH_RADIUS, unit=" km", font_size=36)
        re_label.next_to(label, DOWN, aligned_edge=RIGHT)
        dashed_lines = VGroup(
            DashedLine(brace.get_corner(UR), earth.get_top() + RIGHT, dash_length=0.03),
            DashedLine(brace.get_corner(DR), earth.get_bottom() + RIGHT, dash_length=0.03),
        )
        dashed_lines.set_stroke(GREY, 2)

        frame.reorient(-2, 59, 0, (-0.08, 0.07, -0.08), 2.76)
        self.play(
            frame.animate.to_default_state(),
            LaggedStart(
                Animation(Mobject(), remover=True),
                Animation(Mobject(), remover=True),
                Animation(Mobject(), remover=True),
                FadeIn(brace, scale=100),
                FadeIn(label),
                FadeIn(dashed_lines),
                lag_ratio=0.5,
            ),
            run_time=3
        )
        self.wait()
        self.play(FadeIn(re_label))
        self.wait()

        # Zoom out from earth to moon
        orbit = Circle(radius=MOON_ORBIT_RADIUS * conversion_factor)
        orbit.set_stroke(GREY_C, width=(0, 3))
        # orbit.rotate(-45 * DEG)
        orbit.rotate(-135 * DEG)
        orbit.add_updater(lambda m, dt: m.rotate(3 * dt * DEG))

        moon = get_moon(radius=MOON_RADIUS * conversion_factor)
        moon.to_edge(RIGHT)
        moon.add_updater(lambda m: m.move_to(orbit.get_start()))

        radial_line = Line()
        radial_line.set_stroke(WHITE, 0.5)
        radial_line.add_updater(lambda m: m.put_start_and_end_on(earth.get_center(), moon.get_center()))

        # dist_question = Text("Distance to the moon?", font_size=400)
        # dist_question.add_updater(lambda m: m.next_to(radial_line.get_center(), UR, buff=1))
        dist_question = TexText(R"Distance to the moon \\ $\approx$ 385,000km $\pm$ 21,000km", font_size=600)
        dist_question.add_updater(lambda m: m.next_to(radial_line.get_center(), DR, buff=1))

        trg_height = 1.2 * orbit.get_height()
        self.add(orbit, moon)
        self.play(
            frame.animate.set_height(trg_height),
            FadeIn(radial_line, time_span=(2, 3)),
            FadeIn(dist_question, time_span=(2, 3)),
            run_time=4,
        )
        self.wait(20)

        # Show earth's shadow
        sunlight = GlowDot(radius=3 * FRAME_WIDTH, glow_factor=1, opacity=0.5)
        sunlight.move_to(1.5 * FRAME_WIDTH * LEFT)
        sunlight.fix_in_frame()

        shadow = Rectangle(
            width=3 * MOON_ORBIT_RADIUS * conversion_factor,
            height=2 * EARTH_RADIUS * conversion_factor,
        )
        shadow.set_stroke(width=0)
        shadow.set_fill(BLACK, 1)
        shadow.move_to(ORIGIN, LEFT)

        sunlight.set_z_index(-1)
        shadow.set_z_index(-1)

        self.add(sunlight, shadow)
        self.play(GrowFromCenter(sunlight, run_time=3))
        self.wait()

        # Zoom in to the moon
        orbit.clear_updaters()

        self.play(
            Rotate(orbit, -angle_of_vector(orbit.get_start()) - 1.25 * DEG),
            frame.animate.set_height(6 * EARTH_RADIUS * conversion_factor).move_to(MOON_ORBIT_RADIUS * conversion_factor * RIGHT),
            FadeOut(radial_line, time_span=(0, 2)),
            FadeOut(dist_question, time_span=(0, 2)),
            sunlight.animate.set_opacity(0.35),
            run_time=5
        )

        # Show the moon passing through
        self.play(
            Rotate(orbit, 2.5 * DEG, about_point=ORIGIN),
            Rotate(moon, 2.5 * DEG, about_point=ORIGIN),
            run_time=12,
            rate_func=linear,
        )
        self.wait()

        # Show the width
        brace_copy = brace.copy().flip()
        brace_copy.next_to(orbit, RIGHT, LARGE_BUFF)
        label_copy = VGroup(label, re_label).copy()
        label_copy.arrange(DOWN, aligned_edge=LEFT)
        label_copy.next_to(brace_copy, RIGHT, SMALL_BUFF)

        self.play(
            GrowFromCenter(brace_copy),
            FadeIn(label_copy, lag_ratio=0.1)
        )
        self.wait()

        # Show the time
        time_label = Tex(R"~4 \text{ hours}", font_size=60)
        time_label.next_to(brace_copy, RIGHT)

        self.play(
            FadeOut(label_copy, lag_ratio=0.1),
            FadeIn(time_label, lag_ratio=0.1),
        )
        self.wait()

        # Show the full lunar month
        orbit_rate = ValueTracker(0)
        orbit.clear_updaters()
        orbit.add_updater(lambda m, dt: m.rotate(orbit_rate.get_value() * dt * DEG))

        large_text_height = 2 * earth.get_height()

        month_arrow = Arc(
            5 * DEG,
            350 * DEG,
            arc_center=2.0 * earth.get_right(),
            radius=0.9 * MOON_ORBIT_RADIUS * conversion_factor
        )
        month_arrow.set_stroke(TEAL, 3)
        month_arrow.add_tip()
        month_arrow.tip.set_height(4 * EARTH_RADIUS * conversion_factor, about_point=month_arrow.get_end())

        month_label = Text("28 days")
        month_label.set_height(2 * large_text_height)
        month_label.match_color(month_arrow)
        month_label.next_to(month_arrow.pfp(3 / 8), DR, buff=earth.get_height())

        self.play(
            frame.animate.move_to(MOON_ORBIT_RADIUS * conversion_factor * LEFT).set_height(1.35 * orbit.get_height()).set_anim_args(run_time=4),
            orbit_rate.animate.set_value(15).set_anim_args(run_time=4),
            time_label.animate.set_height(large_text_height, about_point=brace_copy.get_right()).set_anim_args(run_time=4),
            ShowCreation(month_arrow, time_span=(1, 5)),
            Write(month_label, time_span=(2, 3)),
        )
        self.wait(32)

    def show_circles(self):
        radii = [1e4, 1e5]
        circles = VGroup()
        for n, radius in enumerate(radii):
            circle = Circle(radius=radius * conversion_factor)
            circle.set_stroke(WHITE, 2)
            radial_line = Line(ORIGIN, circle.get_right())
            radial_line.set_stroke(GREY, 1)
            label = Integer(radius, unit=" km")
            label.set_width(0.25 * radial_line.get_width())
            label.move_to(radial_line.pfp(0.6), DOWN).shift(0.001 * radius * UP)
            group = VGroup(radial_line, label, circle)
            circles.add(group)

    def get_red_moon(self):
        red_moon = TexturedSurface(
            Sphere(radius=MOON_RADIUS * conversion_factor),
            "RedMoonTexture", "DarkMoonTexture",
        ).set_opacity(0.8)
        return red_moon


class PenumbraAndUmbra(InteractiveScene):
    def construct(self):
        # Add earth and sun
        frame = self.frame
        frame.set_field_of_view(10 * DEG)
        light_source = self.camera.light_source

        sun = get_sun(radius=2, big_glow_ratio=10, big_glow_factor=2)
        sun.move_to(7 * LEFT)

        earth = get_earth(radius=0.7)
        earth.rotate(90 * DEG, LEFT).rotate(EARTH_TILT_ANGLE, OUT)
        earth.move_to(2 * RIGHT)

        light_source.move_to(sun)
        self.add(sun, earth)

        # Add shadows
        umbra, penumbra, umbral_lines, penumbral_lines = shadow_group = self.get_umbral_lines(sun[0], earth)

        umbrum_word = Text("Umbra", font_size=24)
        umbrum_word.move_to(umbra)
        penumbrum_word = Text("Penumbra", font_size=24)
        penumbrum_word.move_to(umbrum_word)
        penumbrum_word.shift(0.55 * earth.get_height() * UP)

        self.add(penumbra, penumbral_lines, earth, penumbrum_word)
        self.play(
            FadeIn(penumbra),
            FadeIn(penumbral_lines),
            FadeIn(penumbrum_word),
        )
        self.add(*shadow_group, earth, umbrum_word, penumbrum_word)
        self.play(
            FadeIn(umbral_lines, lag_ratio=0),
            FadeIn(umbra, time_span=(0.5, 1)),
            FadeIn(umbrum_word, time_span=(0.5, 1)),
        )
        self.wait()

        # Pull away
        self.add(shadow_group, earth, umbrum_word, penumbrum_word)

        shift_vect = 100 * LEFT
        self.play(
            sun[:2].animate.shift(shift_vect),
            sun[2].animate.shift(5 * LEFT).scale(2),
            frame.animate.reorient(0, 0, 0, (4.21, 0.56, 0.0), 13.04),
            umbrum_word.animate.shift(2 * RIGHT),
            penumbrum_word.animate.scale(0.5, about_edge=DOWN).shift(2 * RIGHT),
            UpdateFromFunc(shadow_group, lambda m: m.become(self.get_umbral_lines(sun[0], earth))),
            run_time=8,
        )

    def get_umbral_lines(self, circle1, circle2):
        r1 = circle1.get_width() / 2
        r2 = circle2.get_width() / 2
        c1 = circle1.get_center()
        c2 = circle2.get_center()

        vect = c2 - c1
        dist = get_norm(vect)

        cs1 = c1 + vect / (1.0 - r2 / r1)
        cs2 = c1 + vect / (1.0 + r2 / r1)

        angle = math.asin(r1 / get_norm(cs1 - c1))

        v1 = rotate_vector(UP, -angle)
        v2 = rotate_vector(DOWN, angle)
        v3 = rotate_vector(UP, angle)
        v4 = rotate_vector(DOWN, -angle)

        umbral_lines = VGroup(
            Line(c1 + r1 * v1, cs1),
            Line(c1 + r1 * v2, cs1),
        )
        umbral_lines.set_stroke(WHITE, 0.5)
        umbra = Polygon(cs1, c2 + r2 * v1, c2, c2 + r2 * v2)
        umbra.set_fill(BLACK, 1).set_stroke(width=0)

        penumbral_lines = VGroup(
            Line(cs2, c2 + r2 * v3).scale(10),
            Line(cs2, c2 + r2 * v4).scale(10),
        )
        penumbral_lines.set_stroke(WHITE, 0.5)
        penumbra = Polygon(
            c2 + r2 * v3, penumbral_lines[0].get_end(),
            penumbral_lines[1].get_end(), c2 + r2 * v4,
        )
        penumbra.set_fill(BLACK, 0.5)
        penumbra.set_stroke(width=0)

        return VGroup(umbra, penumbra, umbral_lines, penumbral_lines)


class LineOfSight(InteractiveScene):
    def construct(self):
        # Add earth
        light = self.camera.light_source
        light.move_to(20 * RIGHT)
        frame = self.frame
        frame.set_field_of_view(25 * DEG)

        conversion_factor = 1 / EARTH_RADIUS

        earth = get_earth(radius=EARTH_RADIUS * conversion_factor)
        earth.rotate(90 * DEG)
        earth.rotate(-EARTH_TILT_ANGLE, UP)
        earth_axis = rotate_vector(OUT, -EARTH_TILT_ANGLE, UP)

        frame.set_height(2.25)
        self.add(earth)

        # Add moon
        orbit = Circle(radius=MOON_ORBIT_RADIUS * conversion_factor)
        orbit.set_stroke(GREY_C, width=(0, 3))
        orbit.rotate(PI)

        moon = get_moon(radius=MOON_RADIUS * conversion_factor)
        moon.rotate(PI)
        moon.move_to(orbit.get_start())

        self.add(orbit, moon)

        # Show a line of sight
        line = Line(earth.get_center() + 0.75 * UP, moon.get_center())
        line.set_stroke(BLUE, 1)

        words = Text("Line of sight")
        words.set_width(line.get_width() * 0.5)
        words.set_color(BLUE)
        words.next_to(line, UP, buff=-0.5 * earth.get_height())

        angle = (MOON_RADIUS / (TAU * MOON_ORBIT_RADIUS)) * TAU
        line.rotate(angle, about_point=line.get_start())

        self.play(
            frame.animate.set_height(MOON_ORBIT_RADIUS * conversion_factor * 2.25),
            ShowCreation(line),
            FadeIn(words, lag_ratio=0.1, time_span=(2, 4)),
            run_time=5
        )

        # Zoom into the moon
        self.play(
            frame.animate.reorient(0, 0, 0, (-59.4, 0.01, 0.0), 2.83),
            line.animate.set_stroke(width=2),
            run_time=3
        )
        self.play(
            Rotate(line, -2 * angle, about_point=line.get_start()),
            run_time=2,
        )
        self.wait()
        self.play(
            frame.animate.set_height(MOON_ORBIT_RADIUS * conversion_factor * 2.25).center(),
            line.animate.set_stroke(width=1),
            run_time=3
        )

        # Rotate over 24 hours
        timer = DecimalNumber(0.0, num_decimal_places=1, edge_to_fix=RIGHT)
        units = Text("hours")
        timer[-1].shift(SMALL_BUFF * RIGHT)
        timer.fix_in_frame()
        timer.move_to(UR)
        timer.add_updater(lambda m: m.set_stroke(width=0).set_fill(border_width=0))
        units.next_to(timer, RIGHT, buff=0.2, aligned_edge=DOWN)
        units.fix_in_frame()
        units.set_stroke(width=0).set_fill(border_width=0)
        self.play(
            VFadeIn(timer, time_span=(0, 1)),
            VFadeIn(units, time_span=(0, 1)),
            ChangeDecimalToValue(timer, 24),
            Rotate(earth, -TAU, about_point=ORIGIN),
            Rotate(line, -TAU, about_point=ORIGIN),
            Rotate(words, -TAU, about_point=ORIGIN),
            Rotate(moon, TAU / 28, about_point=ORIGIN),
            Rotate(orbit, TAU / 28, about_point=ORIGIN),
            run_time=5
        )
        self.wait()

        self.play(*map(FadeOut, [timer, units, words, line]))

        # Ambient rotation
        self.play(
            Rotate(orbit, 90 * DEG, about_point=ORIGIN, rate_func=linear),
            Rotate(moon, 90 * DEG, about_point=ORIGIN, rate_func=linear),
            run_time=12
        )

        # Show elliptical orbit
        moon.f_always.move_to(orbit.get_end)

        orbit_ghost = orbit.copy().set_stroke(opacity=0.5)

        self.add(orbit_ghost)
        self.play(
            orbit.animate.stretch(0.9, 1).stretch(1.1, 0).shift(3 * earth.get_width() * LEFT),
            rate_func=there_and_back,
            run_time=5
        )

        # Zoom in then out
        self.play(
            frame.animate.reorient(80, 82, 0, moon.get_center(), 0.80),
            run_time=5
        )
        self.wait()
        self.play(
            frame.animate.reorient(0, 0, 0, ORIGIN, MOON_ORBIT_RADIUS * conversion_factor * 2.25),
            run_time=5
        )
        self.wait()

        # Show many moons
        ratio = int(2 * PI * MOON_ORBIT_RADIUS / MOON_RADIUS / 2)
        moons = Group()
        for a in np.arange(0, 1, 1 / ratio):
            lil_moon = get_moon(resolution=(21, 11))
            lil_moon.match_height(moon)
            lil_moon.move_to(orbit.pfp(a))
            moons.add(lil_moon)

        self.play(
            FadeOut(orbit, run_time=3),
            FadeIn(moons, lag_ratio=0.05, run_time=5),
        )
        self.wait()

        # Shrink down its orbit
        n_moons = 32
        small_orbit = orbit.copy()
        small_orbit.set_height(4.5 * n_moons * MOON_RADIUS * conversion_factor / TAU)
        small_orbit.set_stroke(WHITE, 1)

        inner_moons = Group()
        for a in np.arange(0, 1, 1.0 / n_moons):
            lil_moon = get_moon(resolution=(51, 25))
            lil_moon.match_height(moon)
            lil_moon.move_to(small_orbit.pfp(a))
            lil_moon.save_state()
            lil_moon.move_to(orbit.pfp(a))
            inner_moons.add(lil_moon)

        self.remove(moon)
        self.play(
            LaggedStartMap(Restore, inner_moons, lag_ratio=0),
            FadeOut(moons),
            frame.animate.set_height(1.5 * small_orbit.get_height()).center().set_anim_args(time_span=(1, 5)),
            run_time=5
        )


class DistanceToSun(InteractiveScene):
    def construct(self):
        # Show sun and the earth
        frame = self.frame
        frame.set_field_of_view(15 * DEG)
        light_source = self.camera.light_source

        earth = get_earth(radius=0.1)
        earth.rotate(90 * DEG)
        earth.rotate(-EARTH_TILT_ANGLE, UP)
        earth.to_edge(LEFT, buff=1.0)

        sun = get_sun(radius=1.0)
        sun.move_to((0.5 * FRAME_WIDTH - 2) * RIGHT)
        sun.rotate(90 * DEG, LEFT)

        light_source.f_always.move_to(sun[0].get_center)
        self.add(light_source)

        self.add(earth, sun)
        frame.reorient(0, 0, 0, earth.get_center(), 2 * earth.get_height())
        self.play(frame.animate.to_default_state(), run_time=5)

        # Show distance and radius
        sun_brace = Brace(sun[0], RIGHT)
        sun_brace.stretch(0.5, 1, about_edge=UP)
        sun_brace.move_to(sun.get_center(), DL).shift(SMALL_BUFF * RIGHT)
        sun_radius_label = sun_brace.get_tex("R_S", buff=0.05)
        VGroup(sun_brace, sun_radius_label).set_color(BLACK)

        dist_line = Line(earth.get_center(), sun.get_center())
        dist_line.insert_n_curves(10)
        dist_line.set_stroke(WHITE, width=(0, *5 * [2], 0))
        dist_label = Tex(R"D_S")
        dist_label.next_to(dist_line, UP, SMALL_BUFF)

        self.play(
            GrowFromCenter(sun_brace),
            Write(sun_radius_label),
        )
        self.wait()
        self.play(
            ShowCreation(dist_line),
            Write(dist_label),
        )
        self.wait()

        # Show ratio
        ratio = Tex(R"{R_S \over D_S}")
        ratio.to_corner(UL)
        ratio.set_color(YELLOW)

        self.play(
            TransformFromCopy(sun_radius_label, ratio["R_S"][0]),
            TransformFromCopy(dist_label, ratio["D_S"][0]),
            Write(ratio[R"\over"][0])
        )
        self.wait()

        # Zoom into the moon
        orbit_radius = 2
        orbit = Circle(radius=orbit_radius, n_components=100)
        orbit.move_to(earth)
        orbit.set_stroke(GREY, (0, 2))

        scaled_moon_radius = 0.5 * (sun[0].get_height() / get_dist(earth.get_center(), sun.get_center())) * orbit_radius
        small_moon_radius = (1.0 / 4) * earth.get_height()
        moon = get_moon(radius=small_moon_radius)
        moon.move_to(orbit.get_end())

        self.add(orbit, moon)
        self.play(
            FadeIn(orbit),
            FadeIn(moon),
            dist_line.animate.set_stroke(opacity=0),
            frame.animate.reorient(0, 0, 0, (-4.94, 0.05, 0.0), 1.83),
            run_time=4,
        )
        self.wait()

        # Show the moon sizes and ratio
        moon_brace = sun_brace.copy().set_color(WHITE)
        moon_brace.set_height(moon.get_height())
        moon_brace.next_to(moon, RIGHT, buff=0, aligned_edge=UP)
        moon_brace.stretch(0.5, 1, about_edge=UP)
        moon_radius_label = moon_brace.get_tex("R_M", font_size=8, buff=0.01)
        moon_radius_label.align_to(moon_brace, DOWN)

        moon_dist_line = Line(earth.get_center(), moon.get_center())
        moon_dist_line.insert_n_curves(20)
        moon_dist_line.set_stroke(TEAL_A, (0, 2, 2, 2, 0))
        moon_dist_label = Tex("D_M", font_size=8)
        moon_dist_label.next_to(moon_dist_line, UP, buff=0.01)

        moon_ratio = Tex(R"{R_M \over D_M}", font_size=10)
        moon_ratio.set_color(GREY_B)
        moon_ratio.next_to(frame.get_corner(UL), DR, buff=SMALL_BUFF)

        self.play(FadeIn(moon_ratio))
        self.play(
            GrowFromCenter(moon_brace),
            TransformFromCopy(moon_ratio["R_M"][0], moon_radius_label),
        )
        self.play(
            ShowCreation(moon_dist_line),
            TransformFromCopy(moon_ratio["D_M"][0], moon_dist_label),
        )
        self.wait()

        # Compare ratios
        equals = Tex("=")
        equals.next_to(ratio, RIGHT, 2 * SMALL_BUFF)

        sun_point = Point(sun.get_center())
        size_ratio = moon.get_height() / get_dist(moon.get_center(), earth.get_center())
        sun[0].add_updater(lambda m: m.move_to(sun_point).set_height(size_ratio * get_dist(sun_point.get_center(), earth.get_center())))
        sun[1].add_updater(lambda m: m.move_to(sun[0]).set_height(1.2 * sun[0].get_height()))
        sun[2].add_updater(lambda m: m.move_to(sun[0]))

        VGroup(sun_brace, sun_radius_label).set_color(WHITE)
        sun_brace.f_always.set_height(lambda: 0.5 * sun[0].get_height(), stretch=lambda: True)
        sun_brace.always.next_to(sun[0], RIGHT, buff=0, aligned_edge=UP)
        sun_radius_label.always.next_to(sun_brace, buff=0.05)

        self.remove(sun_brace, sun_radius_label, dist_line, dist_label, ratio)

        self.play(
            frame.animate.reorient(-87.85, 89.126, 0, [-5.6279674 ,-0.0808167, 0.00590339], 0.03),
            sun[2].animate.set_opacity(0.15),
            *map(FadeOut, [moon_ratio, moon_dist_line, moon_dist_label, moon_brace, moon_radius_label, orbit]),
            run_time=5,
        )
        self.wait()
        self.play(
            frame.animate.to_default_state(),
            FadeIn(orbit),
            run_time=3
        )

        VGroup(sun_radius_label, dist_label).set_color(YELLOW)
        self.play(
            GrowFromCenter(sun_brace),
            GrowFromCenter(sun_radius_label),
        )
        dist_line.set_stroke(opacity=1),
        self.play(
            FadeIn(dist_line),
            FadeIn(dist_label, shift=0.25 * UP),
        )
        self.wait()

        # Wiggle the orbit
        self.play(
            orbit.animate.stretch(0.8, 0).stretch(1.2, 1),
            UpdateFromFunc(Group(moon), lambda m: m.move_to(orbit.get_end())),
            rate_func=wiggle,
            run_time=4
        )
        self.wait()

        # Fade Out labels
        self.play(
            LaggedStartMap(
                FadeOut,
                VGroup(dist_line, dist_label, sun_brace, sun_radius_label),
                scale=0.5,
            ),
        )

        # Shift sun scale around
        sun.clear_updaters()
        to_moon = moon.get_center() - earth.get_center()
        true_sun_center = earth.get_center() + to_moon * (EARTH_ORBIT_RADIUS / MOON_ORBIT_RADIUS)
        true_sun_height = earth.get_height() * (SUN_RADIUS / EARTH_RADIUS)
        sun.save_state()
        sun.target = sun.generate_target()
        sun.target.scale(true_sun_height / sun[0].get_height())
        sun.target.move_to(true_sun_center)
        sun.target[1].set_radius(1.2 * true_sun_height)
        sun.target[2].set_radius(10 * true_sun_height)

        self.play(
            MoveToTarget(sun),
            frame.animate.reorient(0, 0, 0, (365.67, 14.24, 0.0), 485.35),
            run_time=5
        )

        sun.target[0].set_height(2 * moon.get_height())
        sun.target[1].set_height(1.2 * 2 * moon.get_height())
        sun.target[2].set_height(50 * 2 * moon.get_height())
        sun.target[2].set_opacity(0.35)
        sun.target.move_to(earth.get_center() + 2 * to_moon)

        self.play(
            MoveToTarget(sun, time_span=(0, 2)),
            frame.animate.reorient(0, 0, 0, (-3.89, 0.03, 0.0), 4.73),
            run_time=4
        )

        # Show being twice as big
        dist_brace = Brace(Line(earth.get_top(), moon.get_top()), UP, buff=0.1)
        dist_brace2 = dist_brace.copy().shift(dist_brace.get_width() * RIGHT)

        side_brace = Brace(moon, RIGHT, buff=0)
        side_brace.stretch(0.25, 0, about_edge=LEFT)
        side_brace_pair = side_brace.get_grid(2, 1, buff=0)
        side_brace_pair.next_to(sun[0], RIGHT, buff=0)

        self.play(GrowFromPoint(dist_brace, dist_brace.get_left()))
        self.play(TransformFromCopy(dist_brace, dist_brace2))

        self.play(GrowFromCenter(side_brace))
        self.play(TransformFromCopy(VGroup(side_brace), side_brace_pair))
        self.wait()

        self.play(LaggedStartMap(FadeOut, VGroup(dist_brace, dist_brace2, side_brace, side_brace_pair)))
        self.wait()

        # Show one moon orbit (with the intention of showing phases in the corner)
        orbit_group = Group(orbit, moon)

        self.play(Rotate(orbit_group, TAU, about_point=earth.get_center(), rate_func=linear, run_time=10))
        self.play(Rotate(orbit_group, TAU / 8, about_point=earth.get_center(), rate_func=linear, run_time=1.25))

        # Sun highlights half, we see half
        words1 = Text("Sun illuminates half")
        words2 = Text("We see half")
        sub_words = Text("(a different half)", font_size=36)

        words1.move_to(4 * RIGHT + 2 * UP)
        words2.move_to(4 * LEFT + 2 * DOWN)
        sub_words.next_to(words2, DOWN)

        arrow1 = Arrow(words1.get_bottom(), 0.75 * RIGHT, path_arc=-60 * DEG)
        arrow2 = Arrow(words2.get_top(), 0.75 * LEFT, path_arc=-60 * DEG)

        VGroup(words1, words2, sub_words, arrow1, arrow2).scale(0.6 / FRAME_HEIGHT, about_point=ORIGIN).shift(moon.get_center())
        VGroup(words2, sub_words, arrow2).set_color(BLUE_D)

        our_half = Sphere(radius=0.51 * moon.get_height())
        our_half.set_color(BLUE, 0.35)
        our_half.always_sort_to_camera(self.camera)
        our_half.rotate(90 * DEG, UP)
        our_half.rotate(45 * DEG)
        our_half.move_to(moon)
        our_half.pointwise_become_partial(our_half, 0, 0.5)

        self.play(
            frame.animate.reorient(0, 0, 0, moon.get_center(), 0.6),
            run_time=2
        )
        self.play(
            FadeIn(words1, lag_ratio=0.1),
            Write(arrow1),
        )
        self.wait()
        self.play(
            FadeIn(words2, lag_ratio=0.1),
            Write(arrow2),
            ShowCreation(our_half)
        )
        self.wait()
        self.play(FadeIn(sub_words, 0.03 * DOWN))
        self.play(
            FadeOut(VGroup(words1, arrow1, words2, arrow2, sub_words)),
            frame.animate.reorient(0, 0, 0, (-6.23, 0.02, 0.0), 4.73).set_anim_args(run_time=2),
        )

        # Transition to a different phase
        orbit_group.add(our_half)
        self.play(Rotate(orbit_group, TAU / 4, about_point=earth.get_center(), run_time=5))
        self.play(our_half.animate.set_opacity(0))

        # Flat moon
        moon.save_state()
        moon.target = moon.generate_target()
        moon.target.rotate(45 * DEG)
        moon.target.stretch(0.01, 0)
        moon.target.rotate(-45 * DEG)
        moon.target.data["d_normal_point"] = moon.target.data["point"] + 1e-3 * DR

        self.play(MoveToTarget(moon))
        self.play(
            frame.animate.reorient(3, 63, 0, (-6.08, 1.44, -0.78), 2.88),
            run_time=3
        )
        self.play(
            Rotate(orbit_group, -TAU / 4, about_point=earth.get_center(), run_time=8, rate_func=there_and_back),
        )
        self.play(
            Restore(moon, time_span=(0, 1)),
            frame.animate.reorient(0, 0, 0, (-4.8, 0, 0.0), 4.22),
            run_time=2
        )

        # Show full and new moons
        self.play(Rotate(orbit_group, TAU / 8, about_point=earth.get_center(), run_time=2))
        full_moon = moon.copy()
        full_moon_label = Text("Full moon", font_size=15)
        full_moon_label.next_to(full_moon, UR, buff=0.025)
        self.play(Write(full_moon_label))
        self.wait()

        self.add(full_moon)
        self.play(Rotate(orbit_group, TAU / 2, about_point=earth.get_center(), run_time=2))

        new_moon = moon.copy()
        new_moon_label = Text("New moon", font_size=15)
        new_moon_label.next_to(new_moon, UL, buff=0.025)
        self.play(Write(new_moon_label))
        self.wait()
        self.add(new_moon)

        # Ask about half moon
        question = Text("When is the\nhalf moon?", font_size=15)
        question.set_color(RED)
        question.always.next_to(moon, DL, buff=0.025)

        self.play(
            Rotate(orbit_group, 3 * TAU / 8, about_point=earth.get_center(), run_time=8),
            VFadeIn(question, time_span=(1, 3))
        )
        self.play(Rotate(orbit_group, -TAU / 8, about_point=earth.get_center(), run_time=8))
        self.wait()

        # Show incorrect right angle
        not_here = Text("Not here!", font_size=15)
        not_here.next_to(moon, DL, buff=0.05)
        not_here.set_color(RED)
        half_moon_label = Text("Half moon", font_size=15)

        def get_half_moon_angle():
            orbit_radius = orbit.get_width() / 2
            sun_dist = get_dist(sun[0].get_center(), earth.get_center())
            return math.acos(orbit_radius / sun_dist)

        def get_half_moon_point():
            theta = get_half_moon_angle()
            return earth.get_center() + rotate_vector(RIGHT, theta) * orbit.get_width() / 2

        half_moon_point = get_half_moon_point()

        lines1 = VGroup(
            Line(sun[0].get_center(), earth.get_center()),
            Line(earth.get_center(), orbit.get_top()),
        )
        lines2 = VGroup(
            Line(sun[0].get_center(), half_moon_point),
            Line(half_moon_point, earth.get_center())
        )
        VGroup(lines1, lines2).set_stroke(WHITE, 2)

        elbow1 = Elbow(width=0.15).shift(earth.get_center())
        elbow2 = Elbow(width=0.15, angle=get_half_moon_angle() - PI).shift(half_moon_point)

        self.play(
            FadeOut(question),
            FadeIn(not_here, scale=0.75),
            FadeIn(lines1),
            FadeIn(elbow1),
        )
        self.wait()
        self.play(
            ReplacementTransform(lines1, lines2, time_span=(2, 3)),
            ReplacementTransform(elbow1, elbow2, time_span=(2, 3)),
            Rotate(orbit_group, get_half_moon_angle() - 90 * DEG, about_point=earth.get_center()),
            FadeOut(not_here, time_span=(0, 1)),
            run_time=3
        )

        half_moon_label.always.next_to(moon, UR, buff=0.025)
        self.play(FadeIn(half_moon_label))
        self.wait()

        # Zoom in
        self.play(
            frame.animate.reorient(-30, 87, 0, (-5.02, 1.67, 0.0), 0.35),
            # elbow2.animate.scale(0.1, about_point=half_moon_point),
            FadeOut(lines2),
            FadeOut(elbow2),
            FadeOut(half_moon_label),
            run_time=3,
        )
        self.wait()
        self.play(
            frame.animate.reorient(3, 0, 0, (-5.02, 1.71, 0.01), 0.36),
            our_half.animate.set_opacity(0.35),
            run_time=2
        )
        self.wait()

        lit_half = our_half.copy().set_opacity(0)
        lit_half.shift(1e-2 * OUT)
        self.play(
            Transform(
                lit_half,
                lit_half.copy().rotate(90 * DEG, about_point=half_moon_point).set_color(YELLOW, 0.35),
                path_arc=90 * DEG,
            ),
            run_time=1
        )
        self.wait()
        self.play(
            FadeOut(lit_half),
            our_half.animate.set_opacity(0),
            FadeIn(half_moon_label),
            frame.animate.reorient(0, 0, 0, (-2.89, 0.13, 0.0), 6.83),
            FadeIn(elbow2),
            FadeIn(lines2),
            run_time=5,
        )

        # Move sun away
        def get_lunar_angle():
            return angle_of_vector(moon.get_center() - earth.get_center())

        def get_implied_sun_location():
            return earth.get_center() + RIGHT * (orbit.get_width() / 2) / math.cos(get_lunar_angle())

        sun.f_always.move_to(get_implied_sun_location)

        sun_line, moon_line = lines2
        orbit_group.add(moon_line, elbow2)

        sun_line.add_updater(lambda m: m.put_start_and_end_on(moon.get_center(), sun.get_center()))

        self.remove(lines2)
        self.add(orbit_group, sun_line)
        self.play(Rotate(orbit_group, 25 * DEG, about_point=earth.get_center()), run_time=5)
        self.play(Rotate(orbit_group, -35 * DEG, about_point=earth.get_center()), run_time=5)
        self.play(Rotate(orbit_group, 25 * DEG, about_point=earth.get_center()), run_time=10)

        # Add angle label
        v_line = DashedLine(earth.get_center(), earth.get_center() + orbit.get_width() * UP)
        v_line.set_stroke(WHITE, 1)

        def get_diff_arc(radius=0.75, color=WHITE, stroke_width=3):
            return Arc(
                90 * DEG,
                get_half_moon_angle() - 90 * DEG,
                radius=radius,
                arc_center=earth.get_center()
            ).set_stroke(color, stroke_width)

        arc = always_redraw(get_diff_arc)
        theta_label = Tex(R"\theta", font_size=24)
        theta_label.add_updater(lambda m: m.set_width(min(0.15, 0.5 * arc.get_width())))
        theta_label.add_updater(lambda m: m.next_to(arc.pfp(0.6), UP, SMALL_BUFF))

        self.play(
            ShowCreation(v_line),
            FadeIn(arc),
            FadeIn(theta_label),
        )
        self.wait()
        self.play(
            Rotate(orbit_group, 10 * DEG, about_point=earth.get_center()),
            run_time=8
        )

        # Equation
        dist_eq = Tex(R"D_S = {D_M \over \sin(\theta)}", t2c={"D_S": YELLOW, "D_M": GREY_B})
        dist_eq.to_edge(UP)
        dist_eq.fix_in_frame()

        dist_line = Line(earth.get_center(), sun[0].get_center())
        dist_line.set_stroke(YELLOW, 2)
        dist_label = Tex(R"D_S", font_size=72)
        dist_label.next_to(dist_line, DOWN)

        self.play(
            frame.animate.reorient(0, 0, 0, (4.7, 0.17, 0.0), 14.93),
            ShowCreation(dist_line, time_span=(3, 5)),
            FadeIn(dist_label, RIGHT, time_span=(3, 5)),
            FadeIn(dist_eq, time_span=(3, 5)),
            run_time=5
        )
        self.wait()

        # Zoom in on discrpency
        self.play(
            FadeOut(dist_eq, time_span=(0, 2)),
            frame.animate.reorient(0, 0, 0, (-6.0, 1.05, 0.0), 2.52),
            run_time=5,
        )

        # Comment on discrepency
        disc_arc = get_diff_arc(radius=orbit_radius, color=BLUE, stroke_width=5)
        est_words = Text("Aristarchus estimated\n6 hours", font_size=15)
        est_words.next_to(frame.get_corner(UL), DR, SMALL_BUFF)
        est_words.set_color(BLUE)

        question = Text("How much time\nis this?", font_size=15)
        question.move_to(est_words, UP)
        question.set_color(BLUE)

        arrow = Arrow(
            est_words["6 hours"].get_bottom() + 0.05 * DOWN,
            disc_arc.get_center(),
            buff=0.025,
            path_arc=90 * DEG,
            thickness=1.5
        )
        arrow.set_fill(BLUE)

        self.play(
            FadeIn(question),
            FadeIn(arrow),
            ShowCreation(disc_arc)
        )
        self.wait()
        self.play(
            FadeOut(question, 0.25 * UP),
            FadeIn(est_words, 0.25 * UP),
        )
        self.wait()

        # Show true answer
        true_answer = TexText(R"True answer: $\sim30$ minutes", font_size=15)
        true_answer.set_color(TEAL)
        true_answer.move_to(est_words, DL)

        self.play(
            FadeOut(est_words, 0.25 * LEFT, time_span=(1, 2)),
            FadeIn(true_answer, 0.25 * LEFT, time_span=(1.25, 2.25)),
            Rotate(orbit_group, 4 * DEG, about_point=earth.get_center()),
            UpdateFromAlphaFunc(disc_arc, lambda m, a: m.become(
                get_diff_arc(
                    radius=orbit_radius,
                    color=interpolate_color(BLUE, TEAL, a),
                    stroke_width=5
                )
            )),
            arrow.animate.set_color(TEAL).shift(0.05 * LEFT),
            run_time=3,
        )
        self.wait()

        # Show false distance to the Sun
        brace = Brace(Line(earth.get_bottom(), new_moon.get_bottom()), buff=0)
        brace_copies = brace.get_grid(1, 20, buff=0)
        brace_copies.move_to(brace, LEFT)

        v_line_copies = VGroup(
            v_line.copy().align_to(brace_copy, RIGHT)
            for brace_copy in brace_copies
        )
        v_line_copies.stretch(0.5, 1, about_edge=DOWN)

        self.play(
            FadeOut(true_answer),
            FadeOut(arrow),
            FadeOut(disc_arc),
            FadeOut(dist_line),
            FadeOut(dist_label),
            Rotate(orbit_group, 1 * DEG - math.asin(1 / 20), about_point=earth.get_center())
        )
        self.play(
            frame.animate.reorient(0, 0, 0, (13.31, -0.69, 0.0), 26.92),
            FadeIn(brace, time_span=(0, 1)),
            LaggedStart(
                (TransformFromCopy(brace, brace2, path_arc=45 * DEG)
                for brace2 in brace_copies),
                lag_ratio=0.1,
                time_span=(1, 5)
            ),
            LaggedStartMap(FadeIn, v_line_copies, lag_ratio=0.1, time_span=(1.5, 5)),
            sun[2].animate.set_radius(50).set_glow_factor(2),
            run_time=5,
        )
        self.wait()

        # True distance
        self.play(
            Rotate(orbit_group, math.asin(1 / 20) - math.asin(1 / 383), about_point=earth.get_center()),
            frame.animate.reorient(0, 0, 0, (364.45, -1.75, 0.0), 480.47),
            run_time=6
        )
        self.wait()


class PhasesOfTheMoon(InteractiveScene):
    def construct(self):
        # Position ourselves
        moon = get_moon(radius=2)
        moon.set_shading(0, 0, 2)

        frame = self.frame
        frame.set_field_of_view(10 * DEG)
        light_source = self.camera.light_source
        light_source.move_to(10 * LEFT)

        self.add(moon)

        frame.reorient(-90, 90, 0)

        theta_tracker = ValueTracker(0)
        dist_to_earth = 20 * moon.get_width()
        earth_point = 10 * moon.get_width() * LEFT

        def get_sun_location():
            theta = theta_tracker.get_value()
            return earth_point + 2 * dist_to_earth * rotate_vector(RIGHT, -theta)

        light_source.f_always.move_to(get_sun_location)

        dot = GlowDot()
        dot.set_radius(5)
        dot.always.move_to(light_source)

        self.add(moon, light_source)
        self.play(
            theta_tracker.animate.set_value(TAU),
            run_time=10,
            rate_func=linear
        )


class HowManyEarthsInsideSun(InteractiveScene):
    def construct(self):
        # Add earth and sun
        frame = self.frame
        frame.set_field_of_view(10 * DEG)

        earth_radius = 0.25
        earth = get_earth(radius=earth_radius)
        earth.move_to(0.25 * FRAME_WIDTH * LEFT)

        sun = get_sun(radius=7 * earth_radius)
        sun.move_to(0.25 * FRAME_WIDTH * RIGHT)

        self.add(earth)
        self.add(sun)

        # Show seven circles
        circle = Circle(radius=earth_radius)
        circle.move_to(earth)
        circle.set_stroke(BLUE, 5)

        stack = circle.get_grid(7, 1, buff=0)
        stack.move_to(sun)

        self.play(ShowCreation(circle))
        self.play(LaggedStart(
            (TransformFromCopy(circle, circle2)
            for circle2 in stack),
            lag_ratio=0.05,
        ))
        self.wait()

        # Show large sun
        big_sun = get_sun(radius=109 * earth_radius)
        big_sun.move_to(200 * earth_radius * RIGHT)

        big_stack = circle.get_grid(109, 1, buff=0)
        big_stack.move_to(big_sun[0])
        big_stack.set_stroke(width=1)

        self.play(
            Transform(sun, big_sun),
            frame.animate.reorient(0, 0, 0, (8.71, 0.01, 0.0), 89.70),
            circle.animate.set_stroke(width=1),
            # FadeOut(stack),
            Transform(stack, big_stack, lag_ratio=0.001),
            run_time=3
        )


class EarthAroundSun(InteractiveScene):
    def construct(self):
        # Add earth and sun
        frame = self.frame
        frame.set_field_of_view(10 * DEG)
        frame.set_height(10)

        earth_radius = 0.1
        orbit_radius = 6.0

        earth = get_earth(radius=earth_radius)
        earth.move_to(3 * LEFT)

        sun = get_sun(radius=7 * earth_radius)
        sun.move_to(3 * RIGHT)

        # Orbits
        sun_orbit = TracingTail(sun, stroke_color=YELLOW, stroke_width=(0, 10), time_traced=5)
        earth_orbit = TracingTail(earth, stroke_color=BLUE, stroke_width=(0, 5), time_traced=5)

        self.add(sun_orbit, sun, earth_orbit, earth)
        self.wait(2)
        self.play(
            Rotate(sun, TAU, about_point=earth.get_center()),
            # ShowCreation(sun_orbit),
            frame.animate.move_to(earth).set_height(16).set_anim_args(time_span=(0, 4)),
            run_time=10
        )
        self.play(FadeOut(sun_orbit))
        self.wait()
        self.play(
            frame.animate.move_to(sun).set_anim_args(time_span=(0, 4)),
            Rotate(earth, TAU, about_point=sun.get_center()),
            run_time=10
        )
        self.wait(3)
        self.play(sun.animate.scale(109 / 7, about_edge=LEFT).move_to(10 * RIGHT), run_time=4)
        self.wait()


class NearestPlanets(InteractiveScene):
    random_seed = 2
    highlighted_orbit = None
    linger = False

    def construct(self):
        # Frame
        frame = self.frame

        # Add sun
        sun = get_sun(radius=0.01, big_glow_ratio=20)
        sun.center()

        # Add celestial sphere
        celestial_sphere = TexturedSurface(Sphere(radius=200), "hiptyc_2020_8k")
        celestial_sphere.set_shading(0, 0, 0)
        celestial_sphere.set_opacity(0.75)
        self.add(celestial_sphere)
        self.add(sun)

        # Add orbits
        radius_conversion = 1.0 / EARTH_ORBIT_RADIUS
        seconds_per_day = MERCURY_ORBIT_PERIOD

        radii = radius_conversion * np.array([
            MERCURY_ORBIT_RADIUS,
            VENUS_ORBIT_RADIUS,
            EARTH_ORBIT_RADIUS,
            MARS_ORBIT_RADIUS,
            JUPITER_ORBIT_RADIUS,
            SATURN_ORBIT_RADIUS,
        ])
        periods = [
            MERCURY_ORBIT_PERIOD,
            VENUS_ORBIT_PERIOD,
            EARTH_ORBIT_PERIOD,
            MARS_ORBIT_PERIOD,
            JUPITER_ORBIT_PERIOD,
            SATURN_ORBIT_PERIOD,
        ]
        colors = [GREY_C, TEAL, BLUE, RED, ORANGE, GREY_BROWN]

        orbits = VGroup()
        for radius, period, color in zip(radii, periods, colors):
            orbit = Circle(radius=radius)
            orbit.set_stroke(color, width=(0, 3 * radius**0.25))
            orbit.rotate(random.random() * TAU, about_point=ORIGIN)
            orbit.set_anti_alias_width(5)
            orbit.period = period
            orbit.add_updater(lambda m, dt: m.rotate(0.5 * (seconds_per_day / m.period) * TAU * dt))
            orbits.add(orbit)

        self.add(*orbits)

        # Add symbols
        symbol_texs = [R"\mercury", R"\venus", R"\earth", R"\mars", R"\jupiter", R"\saturn"]
        symbols = Tex(
            "".join(symbol_texs),
            additional_preamble=R"\usepackage{wasysym}",
            font_size=16
        )
        for symbol, orbit in zip(symbols, orbits):
            radius = orbit.get_width() / 2
            symbol.orbit = orbit
            symbol.scale(radius**0.5)
            buff = symbol.get_height()
            symbol.factor = (radius - buff) / radius
            symbol.add_updater(lambda m: m.move_to(m.orbit.get_start() * m.factor))

        symbols.update()
        self.add(*symbols)

        # Highlight
        if self.highlighted_orbit is not None:
            orbits.set_stroke(opacity=0.25)
            symbols.set_fill(opacity=0.25)
            orbits[self.highlighted_orbit].set_stroke(opacity=1)
            symbols[self.highlighted_orbit].set_fill(opacity=1)

        # Zoom out
        frame = self.frame

        frame.reorient(0, 56, 0, (-0.11, 0.08, -0.37), 2.76)
        self.play(
            frame.animate.reorient(0, 0, 0, ORIGIN, 15),
            run_time=30,
        )

        if self.linger:
            self.wait(30)

        # Ask about relative sizes
        braces = VGroup()
        for orbit, symbol_tex, angle in zip(orbits, symbol_texs, np.linspace(90 * DEG, 0, len(symbol_texs))):
            brace = Brace(Line(ORIGIN, orbit.get_width()**0.5 * RIGHT), UP, buff=0)
            brace.set_width(orbit.get_width() / 2, about_edge=LEFT)
            sym = Tex(Rf"R{symbol_tex}", additional_preamble=R"\usepackage{wasysym}", font_size=24)
            sym[1].scale(0.5)
            sym[1].next_to(sym[0].get_corner(DR), RIGHT, buff=0)
            sym.match_height(brace)
            sym.set_backstroke(BLACK, 5)
            sym.next_to(brace, UP, buff=0.05)
            brace.add(sym)
            brace.rotate(angle, RIGHT, about_edge=DOWN)
            braces.add(brace)
        braces.reverse_submobjects()

        main_brace = braces[0].copy()
        self.play(
            Succession(
                GrowFromCenter(main_brace),
                *(Transform(main_brace, b, rate_func=lambda t: smooth(clip(1.5 * t, 0, 1))) for b in braces[1:]),
            ),
            frame.animate.reorient(0, 61, 0, ORIGIN, 2.5).set_field_of_view(20 * DEG),
            run_time=6
        )
        self.wait()
        self.play(FadeOut(main_brace))

        # Prepare nested spheres
        spheres = Group(
            self.get_open_sphere(orbit.get_radius())
            for orbit in orbits
        )
        for sphere in spheres:
            sphere.clip_tracker.set_value(1.3)
            sphere[0].set_opacity(0.25)

        # Show platonic solids
        solids = self.get_platonic_solids()
        for solid in solids:
            solid.add_updater(lambda m, dt: m.rotate(10 * DEG * dt * math.cos(self.time), axis=OUT))

        box = SurroundingRectangle(solids)
        solids.next_to(
            rotate_vector(frame.get_corner(UL), frame.get_phi(), RIGHT),
            IN + RIGHT,
            buff=SMALL_BUFF
        )

        self.play(
            LaggedStartMap(FadeIn, solids, shift=0.25 * OUT, run_time=2, lag_ratio=0.25),
            FadeOut(symbols),
        )
        self.wait(3)

        # Show the nesting
        orbits.apply_depth_test()
        spheres.set_color(GREY_D, 1)
        spheres.set_shading(0.25, 0.25, 0.2)
        self.camera.light_source.move_to(20 * LEFT)

        octo, icos, dodec, tetra, cube = solids
        target_solids = self.get_platonic_solids()
        factors = [1.5, 1.15, 1.2, 2.0, 1.0]
        for target_solid, sphere, factor in zip(target_solids, spheres, factors):
            target_solid.set_width(factor * sphere.get_width())
            target_solid.shift(-target_solid.get_all_points().mean(0))
        target_solids[4].center()

        def drop_sphere(index, run_time=1.5):
            self.add(spheres[index])
            self.play(
                FadeIn(spheres[index], scale=0.7),
                run_time=run_time
            )
            self.add(solids[index:])

        self.add(spheres[0], solids[0])
        self.play(  # Octohedron
            FadeIn(spheres[0]),
            ReplacementTransform(solids[0], target_solids[0]),
            frame.animate.reorient(-8, 72, 0, (0.0, 0.0, 0.0), 2.50),
            run_time=2
        )
        drop_sphere(1, run_time=2)
        self.wait(2)
        self.play(  # Dodecahedron
            target_solids[0].animate.set_stroke(opacity=0.25),
            ReplacementTransform(solids[1], target_solids[1]),
            frame.animate.reorient(-2, 69, 0, ORIGIN, 2.92),
            solids[2:].animate.shift(LEFT + 0.1 * OUT),
            run_time=2
        )
        drop_sphere(2)
        self.play(  # Dodec
            target_solids[1].animate.set_stroke(opacity=0.25),
            ReplacementTransform(solids[2], target_solids[2]),
            frame.animate.reorient(-16, 69, 0, ORIGIN, 4),
            solids[3:].animate.shift(LEFT + 0.5 * OUT),
            run_time=2
        )
        drop_sphere(3)
        self.play(  # Tetra
            target_solids[2].animate.set_stroke(opacity=0.25),
            ReplacementTransform(solids[3], target_solids[3]),
            frame.animate.reorient(-31, 68, 0, ORIGIN, 9),
            solids[4:].animate.scale(5).shift(3 * LEFT + 2 * OUT),
            run_time=2
        )
        drop_sphere(4)
        self.play(  # Cube
            target_solids[3].animate.set_stroke(opacity=0.25),
            ReplacementTransform(solids[4], target_solids[4]),
            frame.animate.reorient(-9, 69, 0, ORIGIN, 15),
            run_time=2
        )
        drop_sphere(5)
        self.play(frame.animate.reorient(-19, 72, 0, ORIGIN, 20), run_time=3)

        # Ambient panning
        frame.clear_updaters()
        frame.add_ambient_rotation(3 * DEG)
        self.play(
            LaggedStart(*(sphere.clip_tracker.animate.set_value(0) for sphere in spheres[::-1]), lag_ratio=0.25, run_time=6)
        )
        last_solid = target_solids[-1]
        for solid in (*target_solids, *target_solids, *target_solids):
            self.play(
                solid.animate.set_stroke(WHITE, 2, 1),
                last_solid.animate.set_stroke(WHITE, 1, 0.25),
            )
            last_solid = solid

        # Show difficulty in making the theory fit
        frame.clear_updaters()
        orbits.suspend_updating()

        self.add(*spheres)
        self.play(*(
            mob.animate.scale(1.1).set_anim_args(rate_func=lambda t: wiggle(t, 5), time_span=(random.random(), 4 + random.random()))
            for mob in (*spheres, *orbits, *target_solids)
        ))

        # Trouble with orbits
        self.play(
            frame.animate.reorient(0, 0, 0, ORIGIN, 12),
            target_solids.animate.set_stroke(opacity=0.2),
            FadeOut(spheres),
            run_time=2
        )

        factors = np.random.uniform(0.8, 1.2, len(orbits))
        angles = np.random.uniform(0, TAU, len(orbits))
        self.play(
            *(
                orbit.animate.rotate(angle).stretch(factor, 0).stretch(1 / factor, 1).rotate(-angle).set_anim_args(
                    rate_func=lambda t: wiggle(t, 7),
                    time_span=(random.random(), 6 + random.random())
                )
                for orbit, factor, angle in zip(orbits, factors, angles)
            ),
            FadeOut(target_solids),
        )

        # Get rid of circles, add planets
        self.camera.light_source.move_to(ORIGIN)

        small_radius = 0.01
        planets = Group(
            Sphere(radius=small_radius * math.sqrt(orbit.get_radius()), color=orbit.get_color()).set_shading(0.25, 0.25, 1)
            for orbit in orbits
        )
        planets.replace_submobject(2, get_earth(radius=small_radius))
        for planet, orbit, symbol in zip(planets, orbits, symbols):
            planet.f_always.move_to(orbit.get_start)
            symbol.clear_updaters()
            symbol.always.next_to(planet, UR, buff=0.025)

        fading_orbits = orbits.copy()
        orbits.set_stroke(opacity=0)

        self.play(
            FadeOut(fading_orbits, shift=2 * LEFT, lag_ratio=0.2),
            FadeIn(planets),
            FadeIn(symbols),
            celestial_sphere.animate.set_opacity(0.25),
            frame.animate.set_height(4),
            run_time=3,
        )
        orbits.resume_updating()

        # Add observation lines
        lines = VGroup()
        non_earth_planets = [*planets[:2], *planets[3:]]
        for planet in non_earth_planets:
            # line = Line().set_stroke(colors[list(planets).index(planet)], 1)
            line = Line().set_stroke(planet.get_color(), 2, opacity=0.75)
            line.f_always.put_start_and_end_on(planets[2].get_center, planet.get_center)
            lines.add(line)
        lines.apply_depth_test()

        self.play(
            frame.animate.set_height(6).move_to(DOWN),
            FadeIn(lines, time_span=(0, 1)),
            run_time=12
        )

        # Change perspective
        orbits.suspend_updating()
        earth = planets[2]
        self.play(
            frame.animate.reorient(82, 87, 0, earth.get_center(), 0.05),
            FadeOut(symbols),
            *(planet.animate.scale(0.1) for planet in non_earth_planets),
            celestial_sphere.animate.set_opacity(1),
            run_time=3
        )
        self.wait()

    def get_open_sphere(self, radius, color=GREY_C, opacity=0.5):
        sphere = Sphere(radius=radius)
        sphere.set_color(color, opacity)
        sphere.always_sort_to_camera(self.camera)

        mesh = SurfaceMesh(sphere, normal_nudge=0, resolution=(51, 25))
        mesh.set_stroke(WHITE, 0.5, 0.1)
        mesh.set_anti_alias_width(1)
        mesh.deactivate_depth_test()

        result = Group(sphere, mesh)
        result.clip_tracker = ValueTracker(0)

        sphere.add_updater(lambda m: m.set_clip_plane(IN, result.clip_tracker.get_value() * radius))
        mesh.add_updater(lambda m: m.set_clip_plane(OUT, result.clip_tracker.get_value() * radius))

        return result

    def get_platonic_solids(self):
        # Platonic solid test
        dodec = Dodecahedron()
        cube = VCube()

        icos_verts = np.array([pent.get_vertices().mean(0) for pent in dodec])
        octo_verts = np.array([square.get_vertices().mean(0) for square in cube])
        tetra_verts = vertices = np.array([
            [0, 0, 1],
            [np.sqrt(8 / 9), 0, -1 / 3],
            [-np.sqrt(2 / 9), np.sqrt(2 / 3), -1 / 3],
            [-np.sqrt(2 / 9), -np.sqrt(2 / 3), -1 / 3],
        ])

        octo = self.wireframe_from_points(octo_verts, 4)
        icos = self.wireframe_from_points(icos_verts, 5)
        tetra = self.wireframe_from_points(tetra_verts, 3)

        solids = VGroup(octo, icos, dodec, tetra, cube)
        for solid in solids:
            solid.set_height(0.25)
            solid.set_stroke(WHITE, 1)
            solid.set_fill(opacity=0)
            solid.set_stroke(flat=False)
            solid.apply_depth_test()
        solids.arrange(RIGHT, buff=0.25)

        return solids

    def wireframe_from_points(self, points, n_neighbors):
        lines = VGroup()
        for point in points:
            norms = np.linalg.norm(points - point, axis=1)
            indices = np.argsort(norms)
            lines.add(*(
                Line(point, points[index])
                for index in indices[1:1 + n_neighbors]
            ))

        return lines


class HighlightEarthOrbit(NearestPlanets):
    highlighted_orbit = 2


class HighlightMarsOrbit(NearestPlanets):
    highlighted_orbit = 3


class HighlightJupiterOrbit(NearestPlanets):
    highlighted_orbit = 4


class KeplersMethod(InteractiveScene):
    def construct(self):
        # Add earth, mars
        frame = self.frame
        frame.set_field_of_view(45 * DEG)
        light_source = self.camera.light_source
        light_source.move_to(ORIGIN)

        orbit_scale_factor = 2.5 / EARTH_ORBIT_RADIUS

        sun = get_sun(radius=0.025, big_glow_ratio=20)
        sun.center()
        earth_orbit = Circle(radius=EARTH_ORBIT_RADIUS * orbit_scale_factor)
        earth_orbit.set_stroke(BLUE, 1)
        earth_orbit.stretch(1.03, 0)

        mars_orbit = Circle(radius=MARS_ORBIT_RADIUS * orbit_scale_factor)
        mars_orbit.set_stroke(RED, 1)
        mars_orbit.stretch(1.03, 0).rotate(50 * DEG)

        earth = get_earth(radius=0.01)
        earth.move_to(earth_orbit.get_start())
        earth_glow = GlowDot(color=BLUE, radius=0.25)
        earth_glow.always.move_to(earth)

        mars = get_planet("Mars", radius=0.01)
        mars_glow = GlowDot(color=RED, radius=0.1)
        mars_glow.always.move_to(mars)
        mars.move_to(mars_orbit.get_start())

        self.add(sun)
        self.add(earth)
        self.add(mars)
        self.add(mars_glow)

        # Add celestial sphere
        celestial_sphere = get_celestial_sphere()
        celestial_sphere.set_z_index(-2)
        celestial_sphere.rotate(170 * DEG)
        self.add(celestial_sphere)

        # Zoom out from earth observing mars, adding an arrow to it
        frame.move_to(earth)
        frame.set_height(6 * earth.get_height())
        frame.reorient(-9, 86, 0)

        line_to_mars = self.get_line_between_bodies(earth, mars, RED)
        line_to_mars.set_z_index(-1)

        symbols = Tex(R"\earth \mars", additional_preamble=R"\usepackage{wasysym}",)
        symbols.scale(0.75)
        earth_symbol, mars_symbol = symbols
        earth_symbol.always.next_to(earth, RIGHT, SMALL_BUFF, aligned_edge=DOWN)
        mars_symbol.always.next_to(mars, UR, buff=0.05)

        self.play(
            frame.animate.to_default_state().set_anim_args(time_span=(1.5, 6)),
            ShowCreation(line_to_mars, suspend_mobject_updating=True),
            mars.animate.scale(5),
            mars_glow.animate.scale(3),
            earth.animate.scale(5).set_anim_args(time_span=(4, 6)),
            FadeIn(earth_glow, time_span=(4, 5)),
            FadeIn(symbols, time_span=(4, 5)),
            run_time=6
        )
        self.wait()

        # Move around the earth and mars
        self.play(
            Rotate(earth, TAU, about_point=earth.get_center() + 0.4 * DOWN),
            MaintainPositionRelativeTo(mars, earth),
            frame.animate.set_height(9),
            run_time=4
        )
        self.play(
            mars.animate.move_to(interpolate(mars.get_center(), earth.get_center(), 0.5)),
            run_time=4,
            rate_func=wiggle,
        )

        # Show the direction the sun
        line_to_sun = self.get_line_between_bodies(earth, sun, YELLOW)
        line_to_sun.set_stroke(width=1)

        self.play(
            ShowCreation(line_to_sun, suspend_mobject_updating=True, run_time=2),
            line_to_mars.animate.set_stroke(width=1)
        )
        self.wait()
        self.play(
            Rotate(earth, TAU, about_point=sun.get_center()),
            Rotate(mars, (EARTH_ORBIT_PERIOD / MARS_ORBIT_PERIOD) * TAU, about_point=sun.get_center()),
            run_time=10
        )

        # Unsure of the distance to the sun
        sun_earth_group = Group(sun[0], earth)
        brace = Brace(Line(sun.get_center(), earth.get_center()), UP)
        q_marks = brace.get_tex("???", buff=SMALL_BUFF)

        brace.f_always.set_width(lambda: get_dist(sun.get_center(), earth.get_center()))
        brace.always.next_to(sun.get_center(), UP, SMALL_BUFF, aligned_edge=LEFT)
        q_marks.always.next_to(brace, UP, SMALL_BUFF)

        self.play(LaggedStart(
            GrowFromPoint(brace, sun.get_center(), suspend_mobject_updating=True),
            FadeIn(q_marks, shift=0.25 * DOWN, suspend_mobject_updating=True),
        ))
        self.play(
            earth.animate.shift(0.5 * LEFT),
            MaintainPositionRelativeTo(mars, earth),
            rate_func=lambda t: wiggle(t, 6),
            run_time=8
        )
        brace.set_fill(border_width=0)
        self.play(FadeOut(brace, suspend_mobject_updating=True), FadeOut(q_marks))

        # Reference both orbits we want to know
        angle = 1.6 * TAU

        earth_annulus = self.get_mystery_orbit(earth_orbit)
        mars_annulus = self.get_mystery_orbit(mars_orbit)

        self.play(
            Rotate(earth, angle, about_point=sun.get_center()),
            Rotate(mars, (EARTH_ORBIT_PERIOD / MARS_ORBIT_PERIOD) * angle, about_point=sun.get_center()),
            FadeIn(earth_annulus, time_span=(3, 5)),
            FadeIn(mars_annulus, time_span=(4, 6)),
            run_time=12
        )

        # Clear the board
        self.play(
            FadeOut(earth_annulus),
            FadeOut(mars_annulus),
            celestial_sphere[0].animate.set_opacity(0.5),
            celestial_sphere[1].animate.set_opacity(0),
        )

        # Fix mars in space
        pin = SVGMobject("push_pin")
        pin.set_height(0.75)
        pin.rotate(40 * DEG, UP)
        pin.set_fill(GREY_E, 1)
        pin.set_shading(0.5, 0.25, 0)
        pin.rotate(10 * DEG)
        pin.move_to(mars.get_center(), DR)

        earth_orbit.set_shape(5.2, 5.0)
        earth_orbit.move_to(0.5 * math.sqrt(earth_orbit.get_width()**2 - earth_orbit.get_height()**2) * LEFT)
        earth_orbit.rotate(angle_of_vector(earth.get_center()) - angle_of_vector(earth_orbit.get_start()), about_point=ORIGIN)
        dens_func = bezier([0, 0.2, 0.8, 1])
        n_samples = 24
        sample_points = [earth_orbit.pfp(dens_func(a)) for a in np.arange(0, 1, 1 / n_samples)]
        earth.move_to(sample_points[0])

        self.play(
            FadeIn(pin, shift=0.5 * DR, rate_func=rush_into),
        )
        self.wait()

        # Show several lines deducing where earth is
        self.remove(line_to_sun, line_to_mars)

        ghost_lines = VGroup()
        ghost_earths = Group()
        self.add(ghost_earths)
        for sample_point in sample_points:
            self.play(
                earth.animate.move_to(sample_point),
                ghost_lines.animate.set_stroke(opacity=0.2),
                run_time=0.5
            )
            new_lines = self.show_earth_location_with_intersection(earth, mars, sun)
            ghost_lines.add(new_lines)
            ghost_earths.add(earth_glow.copy().clear_updaters())

        # Move around fixed mars, see implied ghost locations
        self.play(ghost_lines.animate.set_stroke(opacity=0.2))
        earth.add_updater(lambda m: m.move_to(ghost_earths[-1]))
        pin.add_updater(lambda m: m.move_to(mars.get_center(), DR))
        self.bind_ghost_lines_to_mars(ghost_lines, ghost_earths, mars)

        self.play(
            # Rotate(mars, TAU, about_point=mars.get_center() + 0.2 * LEFT, run_time=5),
            mars.animate.scale(1.2, about_point=ORIGIN),
            rate_func=wiggle,
            run_time=5
        )
        earth.clear_updaters()

        # Unfix mars
        ghost_earths.clear_updaters()
        self.play(
            FadeOut(pin, UL),
            FadeOut(ghost_earths, lag_ratio=0.1, run_time=1),
            FadeOut(ghost_lines, lag_ratio=0.1, run_time=1),
            Rotate(mars, TAU, about_point=sun.get_center(), run_time=8)
        )
        self.wait()

        # Show samples after 5 martian years
        ghost_lines, ghost_earths = self.show_time_series_measurments(earth, mars, sun, earth_glow, 5, 4)
        self.bind_ghost_lines_to_mars(ghost_lines, ghost_earths, mars)

        self.play(*map(FadeOut, [earth, earth_glow, earth_symbol]))
        self.play(Rotate(mars, 5 * DEG, about_point=ORIGIN, run_time=5, rate_func=wiggle))
        self.play(mars.animate.shift(LEFT), run_time=5, rate_func=wiggle)

        earth.move_to(ghost_earths[0])
        mars_copy = mars_glow.copy()
        mars_copy.clear_updaters()
        mars_copy.set_color(PINK)
        self.play(
            FadeOut(ghost_lines),
            ghost_earths.animate.set_color(GREEN),
            FadeIn(mars_copy)
        )

        self.play(
            Rotate(mars, 5 * DEG, about_point=sun.get_center()),
            Rotate(earth, (MARS_ORBIT_PERIOD / EARTH_ORBIT_PERIOD) * 5 * DEG, about_point=sun.get_center()),
            FadeIn(earth_glow),
        )
        self.wait()
        og_ghost_earths = ghost_earths

        # Find a new sample
        ghost_lines, ghost_earths = self.show_time_series_measurments(earth, mars, sun, earth_glow, 5, 2)
        self.bind_ghost_lines_to_mars(ghost_lines, ghost_earths, mars)
        self.play(FadeOut(earth), FadeOut(earth_glow))

        self.play(Rotate(mars, 5 * DEG, about_point=ORIGIN, run_time=5, rate_func=wiggle))
        self.play(mars.animate.shift(LEFT), run_time=5, rate_func=wiggle)

        self.play(*map(FadeOut, [
            ghost_lines, ghost_earths, og_ghost_earths,
            earth_glow, mars_copy,
        ]))

        # Show clusters of five
        step = (MARS_ORBIT_PERIOD / EARTH_ORBIT_PERIOD)
        clusters = Group()
        mars_line_groups = Group()

        last_cluster = Group()
        last_mars_lines = VGroup()
        for n in range(30):
            initial_mars_spot = mars.get_center().copy()
            mars.rotate(5 * DEG, about_point=ORIGIN)

            start = n * (5 / 360)
            alphas = (start + np.arange(5) * step) % 1
            points = [earth_orbit.pfp(dens_func(a)) for a in alphas]
            cluster = Group(GlowDot(point).set_color(BLUE) for point in points)
            mars_lines = VGroup(Line(mars.get_center(), dot.get_center(), buff=0) for dot in cluster)
            mars_lines.set_stroke(RED, 1)
            clusters.add(cluster)

            # Animation
            cluster.set_color(WHITE)
            rigid_group = Group(mars, mars_glow, mars_lines, cluster)
            rigid_group.save_state()
            rigid_group.scale(np.random.uniform(0.9, 1.1), about_point=ORIGIN)
            rigid_group.rotate(np.random.uniform(0, 2 * DEG), about_point=ORIGIN)

            new_mars_spot = mars.get_center().copy()
            mars.move_to(initial_mars_spot)

            self.play(
                FadeOut(last_mars_lines),
                mars.animate.move_to(new_mars_spot)
            )
            self.play(
                ShowIncreasingSubsets(cluster),
                ShowIncreasingSubsets(mars_lines),
                *(dot.animate.set_color(BLUE).set_radius(0.15) for dot in last_cluster)
            )
            self.play(Restore(rigid_group))
            last_cluster = cluster
            last_mars_lines = mars_lines

        earth.move_to(earth_orbit.get_start())
        self.play(
            FadeOut(last_mars_lines),
            FadeOut(last_cluster),
            FadeOut(clusters),
            FadeIn(earth),
            FadeIn(earth_glow),
            FadeIn(earth_symbol),
            ShowCreation(earth_orbit),
        )

        # Show the deduction of Mars' orbit
        mars.set_x(4.8)
        earth_location_tracker = ValueTracker(0)
        earth.add_updater(lambda m: m.move_to(earth_orbit.pfp(dens_func(earth_location_tracker.get_value() % 1))))
        earth_glow.update()
        mars_lines = VGroup()
        earth_copies = Group()
        self.add(mars_lines, earth_copies)
        for n in range(5):
            earth_copy = earth_glow.copy()
            earth_copy.clear_updaters()
            earth_copy.orbit_offset = earth_location_tracker.get_value()
            mars_line = self.get_line_between_bodies(earth_copy, mars, RED, 1, 2)
            mars_line.set_stroke(opacity=0.7)
            mars_line.suspend_updating()

            self.play(
                ShowCreation(mars_line),
                FadeIn(earth_copy)
            )
            mars_lines.add(mars_line)
            earth_copies.add(earth_copy)
            if n == 0:
                self.play(
                    mars.animate.move_to(mars_line.pfp(0.75)).set_anim_args(rate_func=wiggle),
                    frame.animate.set_height(10),
                    run_time=4,
                )
                self.play(
                    mars.animate.set_opacity(0),
                    mars_glow.animate.set_opacity(0),
                    mars_symbol.animate.set_opacity(0),
                )
            if n < 4:
                self.play(
                    Rotate(mars, TAU, about_point=sun.get_center()),
                    earth_location_tracker.animate.increment_value(MARS_ORBIT_PERIOD / EARTH_ORBIT_PERIOD),
                    run_time=2,
                )
            else:
                self.play(
                    mars.animate.set_opacity(1),
                    mars_glow.animate.set_opacity(1),
                    mars_symbol.animate.set_opacity(1),
                )

        # Shift forward a bunch
        self.play(*map(FadeOut, [earth, earth_symbol, earth_glow]))
        earth_location_tracker.set_value(0)
        for dot, mars_line in zip(earth_copies, mars_lines):
            dot.add_updater(lambda m: m.move_to(earth_orbit.pfp(dens_func((earth_location_tracker.get_value() + m.orbit_offset) % 1))))
            mars_line.resume_updating()

        mars_copies = Group()
        self.add(earth_copies, mars_copies)
        for n in range(100):
            d_alpha = 0.01
            mars_copy = mars_glow.copy()
            mars_copy.clear_updaters()
            mars_copy.scale(0.25)
            mars_copies.add(mars_copy)
            mars.rotate(d_alpha * (EARTH_ORBIT_PERIOD / MARS_ORBIT_PERIOD) * TAU, about_point=0.5 * UR)
            earth_location_tracker.increment_value(d_alpha)
            self.wait(0.2)

    def get_line_between_bodies(self, body1, body2, color, stroke_width=2, scale_factor=10):
        line = Line()
        line.set_stroke(color, stroke_width)
        line.f_always.put_start_and_end_on(body1.get_center, body2.get_center)
        line.add_updater(lambda m: m.scale(scale_factor, about_point=m.get_start()))
        return line

    def get_mystery_orbit(self, orbit, opacity=0.25, n_q_marks=12):
        annulus = Annulus(0.9 * orbit.get_radius(), 1.1 * orbit.get_radius())
        annulus.set_fill(orbit.get_color(), opacity=opacity)
        q_marks = VGroup(Tex("?").move_to(orbit.pfp(a)) for a in np.arange(0, 1, 1 / n_q_marks))
        q_marks.set_fill(opacity=0.25)
        return VGroup(annulus, q_marks)

    def show_time_series_measurments(self, earth, mars, sun, earth_glow, n_measurements=5, rotation_time=2):
        ghost_lines = VGroup()
        ghost_earths = Group()
        corner_counter = VGroup(
            Integer(687, include_sign=True, edge_to_fix=RIGHT),
            Text("Days")
        )
        corner_counter.arrange(RIGHT, aligned_edge=UP)
        corner_counter.to_corner(UL, buff=0)

        self.add(ghost_lines)
        self.add(ghost_earths)
        for n in range(n_measurements):
            new_lines = self.show_earth_location_with_intersection(earth, mars, sun)
            ghost_lines.add(new_lines)
            ghost_earths.add(earth_glow.copy().clear_updaters())
            if n < n_measurements - 1:
                corner_counter[0].set_value(0)
                self.add(corner_counter)
                self.play(
                    Rotate(mars, TAU, about_point=sun.get_center()),
                    Rotate(earth, (MARS_ORBIT_PERIOD / EARTH_ORBIT_PERIOD) * TAU, about_point=sun.get_center()),
                    ChangeDecimalToValue(corner_counter[0], 687),
                    ghost_lines.animate.set_stroke(opacity=0.2).set_anim_args(time_span=(0, 1)),
                    run_time=rotation_time
                )
            else:
                self.play(
                    ghost_lines.animate.set_stroke(opacity=0.2),
                    FadeOut(corner_counter)
                )

        return ghost_lines, ghost_earths

    def show_earth_location_with_intersection(self, earth, mars, sun):
        lines = VGroup(
            self.get_line_between_bodies(mars, earth, RED, 1, 2),
            self.get_line_between_bodies(sun, earth, YELLOW, 1, 2),
        )
        lines.clear_updaters()
        lines.set_stroke(width=1)
        self.play(ShowCreation(lines, lag_ratio=0))
        return lines

    def bind_ghost_lines_to_mars(self, ghost_lines, ghost_earths, mars):
        sun_point = ghost_lines[0][1].get_start()
        for pair, dot in zip(ghost_lines, ghost_earths):
            mars_line, sun_line = pair
            mars_line.add_updater(lambda m: m.shift(mars.get_center() - m.get_start()))

            dot.sun_line = sun_line
            dot.mars_line = mars_line
            dot.add_updater(lambda m: m.move_to(find_intersection(
                m.sun_line.get_start(), m.sun_line.get_vector(),
                m.mars_line.get_start(), m.mars_line.get_vector(),
            )))

    def get_rigid_lines_from_mars(self, mars, ghost_earths):
        mars_lines = VGroup(
            self.get_line_between_bodies(mars, dot, RED, 1, 1)
            for dot in ghost_earths
        )
        mars_lines.clear_updaters()
        mars_lines.set_stroke(opacity=0.5)
        return mars_lines


class ShowCreationOfAllOrbits(KeplersMethod):
    def construct(self):
        # Add all orbits (Copied largely from NearestPlanets)
        frame = self.frame
        sun = get_sun(radius=0.02, big_glow_ratio=20).center()
        celestial_sphere = get_celestial_sphere(constellation_opacity=0.01)

        self.add(celestial_sphere)
        self.add(sun)

        # Add orbits
        radius_conversion = 2.0 / EARTH_ORBIT_RADIUS
        seconds_per_day = MERCURY_ORBIT_PERIOD

        radii = radius_conversion * np.array([
            MERCURY_ORBIT_RADIUS,
            VENUS_ORBIT_RADIUS,
            EARTH_ORBIT_RADIUS,
            MARS_ORBIT_RADIUS,
            JUPITER_ORBIT_RADIUS,
            SATURN_ORBIT_RADIUS,
        ])
        periods = [
            MERCURY_ORBIT_PERIOD,
            VENUS_ORBIT_PERIOD,
            EARTH_ORBIT_PERIOD,
            MARS_ORBIT_PERIOD,
            JUPITER_ORBIT_PERIOD,
            SATURN_ORBIT_PERIOD,
        ]
        eccentricities = [
            0.206,
            0,
            0.017,
            0.093,
            0.048,
            0.056
        ]
        colors = [GREY_C, TEAL, BLUE, RED, ORANGE, GREY_BROWN]

        orbits = VGroup()
        for radius, period, color, ecc in zip(radii, periods, colors, eccentricities):
            orbit = Circle(radius=radius)
            orbit.set_stroke(color, width=2)
            orbit.stretch(math.sqrt(1 - ecc**2), 1)
            orbit.shift(0.5 * radius * ecc * RIGHT)
            orbit.rotate(random.random() * TAU, about_point=ORIGIN)
            orbit.set_anti_alias_width(5)
            orbit.period = period
            orbits.add(orbit)

        # Add planet dots
        symbol_texs = [R"\mercury", R"\venus", R"\earth", R"\mars", R"\jupiter", R"\saturn"]
        symbols = Tex(
            "".join(symbol_texs),
            additional_preamble=R"\usepackage{wasysym}",
            font_size=32
        )
        planet_dots = Group()
        symbols[-2:].scale(3)
        for orbit, symbol in zip(orbits, symbols):
            dot = GlowDot(color=orbit.get_color())
            dot.move_to(orbit.get_start())
            orbit.dot = dot
            orbit.symbol = symbol
            symbol.buff_factor = (orbit.get_radius() + 0.6 * symbol.get_height()) / orbit.get_radius()
            symbol.dot = dot
            symbol.add_updater(lambda m: m.move_to(m.buff_factor * m.dot.get_center()))
            planet_dots.add(dot)

        # Draw the orbits
        earth_orbit = orbits[2]
        earth_orbit.set_stroke(width=1)

        frame.set_phi(70 * DEG)
        frame.add_updater(lambda m, dt: m.set_phi(m.get_phi() * (1 - 0.03 * dt)))
        frame.set_height(3)
        growth_rate_tracker = ValueTracker(0.1)
        frame.add_updater(lambda m, dt: m.set_height(m.get_height() + growth_rate_tracker.get_value() * dt))

        self.add(earth_orbit)
        self.draw_orbit(earth_orbit, orbits[3], MARS_ORBIT_PERIOD, 5)
        self.draw_orbit(earth_orbit, orbits[0], MERCURY_ORBIT_PERIOD, 7)
        self.draw_orbit(earth_orbit, orbits[1], VENUS_ORBIT_PERIOD, 6)
        self.draw_orbit(
            earth_orbit, orbits[4], JUPITER_ORBIT_PERIOD, 2, run_time=10,
            added_anims=[frame.animate.set_height(18)],
        )
        growth_rate_tracker.set_value(0.4)
        self.play(FadeIn(orbits[5]))

        # Just linger
        self.wait(9)

    def draw_orbit(self, earth_orbit, orbit, period, n_samples=3, run_time=4, added_anims=[]):
        orbit.set_stroke(width=2)
        earth_year_tracker = ValueTracker()
        get_earth_time = earth_year_tracker.get_value

        def get_earth_orbit_point(offset):
            return earth_orbit.pfp((offset + earth_year_tracker.get_value()) % 1)

        earth_dots = Group()
        lines = VGroup()
        for n in range(n_samples):
            dot = GlowDot(color=earth_orbit.get_color(), radius=0.1)
            dot.offset = n * (period / EARTH_ORBIT_PERIOD)
            dot.add_updater(lambda m: m.move_to(get_earth_orbit_point(m.offset)))
            earth_dots.add(dot)

            line = self.get_line_between_bodies(dot, orbit.dot, orbit.get_color(), 1, 1.1)
            lines.add(line),

        self.play(
            FadeIn(earth_dots),
            ShowCreation(lines, lag_ratio=0.1, suspend_mobject_updating=True),
            FadeIn(orbit.dot),
            FadeIn(orbit.symbol),
        )
        self.play(
            ShowCreation(orbit),
            UpdateFromFunc(orbit.dot, lambda m: m.move_to(orbit.get_end())),
            earth_year_tracker.animate.set_value(period / EARTH_ORBIT_PERIOD),
            *added_anims,
            run_time=run_time,
        )
        self.play(
            FadeOut(lines),
            FadeOut(earth_dots),
            FadeOut(orbit.dot),
            FadeOut(orbit.symbol),
            orbit.animate.set_stroke(width=1)
        )


class LightFromEarthToMoon(InteractiveScene):
    def construct(self):
        # Earth and moon
        self.camera.light_source.move_to(20 * LEFT)
        self.frame.set_field_of_view(15 * DEG)

        conversion_factor = 12 / MOON_ORBIT_RADIUS
        earth = get_earth(radius=conversion_factor * EARTH_RADIUS)
        earth.rotate(EARTH_TILT_ANGLE, axis=DOWN)
        moon = get_moon(radius=conversion_factor * MOON_RADIUS)
        earth.to_edge(LEFT, buff=0.75)
        moon.move_to(earth.get_center() + conversion_factor * MOON_ORBIT_RADIUS * RIGHT)

        labels = VGroup(Text("Earth"), Text("Moon"))
        for label, body in zip(labels, [earth, moon]):
            label.scale(0.75).next_to(body, UP, buff=0.25)

        self.add(earth, moon)
        self.add(labels)

        # Light beam
        low_x = earth.get_x(RIGHT)
        high_x = moon.get_left()[0] - 0.1
        pulse = self.get_pulse(SPEED_OF_LIGHT * conversion_factor * LEFT, low_x, high_x)
        pulse.next_to(moon, LEFT, buff=0)

        self.add(pulse)
        self.wait(20)

    def get_pulse(
        self,
        velocity,
        low_x=-np.inf,
        high_x=np.inf,
        radius=0.2,
        max_stroke_width=6,
        tail_time=0.25,
        label_text=""
    ):
        pulse = GlowDot(radius=radius).set_color(WHITE)

        pulse.velocities = np.zeros_like(pulse.get_points())
        pulse.velocities[:] = velocity

        def update_pulse(pulse, dt):
            points = pulse.get_points()
            new_points = points.copy()
            too_low = points[:, 0] < low_x
            too_high = points[:, 0] > high_x

            new_points[too_low][:, 0] = low_x
            new_points[too_high][:, 0] = high_x

            pulse.velocities[too_low] *= -1
            pulse.velocities[too_high] *= -1
            new_points += pulse.velocities * dt
            pulse.set_points(new_points)

        pulse.add_updater(update_pulse)
        tail = TracingTail(pulse, stroke_width=(0, max_stroke_width), time_traced=tail_time)

        result = Group(pulse, tail)

        if label_text:
            label = Text(label_text, font_size=24)
            label.always.next_to(pulse, DOWN, buff=0)
            result.add(label)

        return result


class LightFromSunToEarth(LightFromEarthToMoon):
    def construct(self):
        # Sun and earth
        self.frame.set_field_of_view(15 * DEG)

        conversion_factor = 12 / EARTH_ORBIT_RADIUS
        earth = get_earth(radius=conversion_factor * EARTH_RADIUS)
        earth.rotate(EARTH_TILT_ANGLE, axis=DOWN)

        sun = get_sun(radius=SUN_RADIUS * conversion_factor, big_glow_ratio=20)
        sun.move_to(6.5 * LEFT)
        earth.move_to(sun.get_center() + conversion_factor * EARTH_ORBIT_RADIUS * RIGHT)
        earth_glow = GlowDot(earth.get_center(), color=BLUE)

        colors = [GREY_C, TEAL, BLUE]
        radii = [MERCURY_ORBIT_RADIUS, VENUS_ORBIT_RADIUS, EARTH_ORBIT_RADIUS]
        orbits = VGroup(
            Circle(radius=radius * conversion_factor)
            for radius, color in zip(radii, colors)
        )
        glows = Group()
        np.random.seed(1)
        angles = [-7 * DEG, 8 * DEG, 0]
        for orbit, angle, color in zip(orbits, angles, colors):
            orbit.rotate(angle)
            orbit.move_to(sun)
            orbit.set_stroke(color, (0, 3))
            glow = GlowDot(color=orbit.get_color())
            glow.move_to(orbit.get_start())
            glows.add(glow)

        symbol_texs = [R"\sun", R"\mercury", R"\venus", R"\earth"]
        labels = Tex(
            "".join(symbol_texs),
            additional_preamble=R"\usepackage{wasysym}",
            font_size=40
        )
        labels[0].set_opacity(0)
        for label, body in zip(labels, [sun[0], *glows]):
            label.next_to(body.get_center(), UR, buff=0.15)

        self.camera.light_source.move_to(sun)

        self.add(sun, earth, glows, orbits, labels)

        # Add pulse
        pulse = self.get_pulse(
            SPEED_OF_LIGHT * conversion_factor * RIGHT,
            low_x=sun.get_x(RIGHT),
            high_x=earth.get_x(LEFT),
            tail_time=20,
            label_text="Light"
        )
        pulse.next_to(sun, RIGHT)

        self.add(pulse)
        self.wait(40)


class LightAcrossEarth(LightFromEarthToMoon):
    def construct(self):
        # Test
        self.frame.set_field_of_view(10 * DEG)
        self.camera.light_source.move_to([-5, 0, 2])
        radius = 2.5
        earth = get_earth(radius=radius)
        earth.rotate(90 * DEG - EARTH_TILT_ANGLE, LEFT)
        conversion_factor = radius / EARTH_RADIUS
        pulse = self.get_pulse(
            0.1 * SPEED_OF_LIGHT * conversion_factor * RIGHT,
            low_x=earth.get_x(LEFT) + 0.025,
            high_x=earth.get_x(RIGHT),
            tail_time=0.4,
        )
        pulse.move_to(earth.get_corner(DL) + 0.5 * DOWN)

        v_lines = Group(
            DashedLine(
                earth.get_corner(UP + vect) + UP,
                earth.get_corner(DOWN + vect) + DOWN,
            ).set_stroke(WHITE, 1)
            for vect in [LEFT, RIGHT]
        )
        v_lines[1].shift(0.1 * RIGHT)

        self.add(earth)
        self.add(v_lines)
        self.add(pulse)
        self.wait(50)


class LightAcrossEarthsOrbit(LightFromEarthToMoon):
    def construct(self):
        # Add orbit
        orbit = Circle(radius=3)
        orbit.set_stroke(BLUE, width=(0, 3))
        orbit.rotate(190 * DEG)

        sun = get_sun(radius=0.05, big_glow_ratio=20)
        sun.center()

        self.add(sun, orbit)

        # Show light
        pulse = self.get_pulse(velocity=0.25 * LEFT, tail_time=5)
        pulse.next_to(orbit.get_right(), RIGHT, buff=LARGE_BUFF)
        pulse.match_y(orbit.get_start())
        label = Text("Light from Io", font_size=24)
        label.always.next_to(pulse, DOWN, buff=-0.05)

        self.add(pulse)
        self.wait(4)
        self.play(FadeIn(label))
        self.wait(20)

    def get_light_pulse(self, width=2, n_points=25, dropoff=2):
        pulse = GlowDots(np.linspace(ORIGIN, width * RIGHT, n_points), radius=0.1)
        pulse.set_opacity(np.linspace(1, 0, pulse.get_num_points())**dropoff)
        pulse.set_color(WHITE)
        return pulse


class EarthAndVenus(InteractiveScene):
    def construct(self):
        # Add sun, stars, and orbits (copied from below)
        frame = self.frame
        conversion_factor = 3.5 / EARTH_ORBIT_RADIUS
        celestial_sphere = get_celestial_sphere(radius=100, constellation_opacity=0.05)

        sun = get_sun(radius=SUN_RADIUS * conversion_factor, big_glow_ratio=20)
        sun.center()

        colors = [TEAL, BLUE]
        radii = np.array([VENUS_ORBIT_RADIUS, EARTH_ORBIT_RADIUS]) * conversion_factor
        orbits = VGroup(
            Circle(radius=radius).set_stroke(color, (0, 5))
            for radius, color in zip(radii, colors)
        )
        glows = Group(GlowDot(color=orbit.get_color()) for orbit in orbits)
        symbols = VGroup(*map(get_planet_symbols, ["venus", "earth"]))

        for symbol, glow, orbit in zip(symbols, glows, orbits):
            orbit.rotate(PI)
            glow.f_always.move_to(orbit.get_start)
            symbol.scale(0.5)
            symbol.glow = glow
            # symbol.add_updater(lambda m: m.move_to(1.075 * m.glow.get_center()))
            symbol.always.next_to(glow, UL, buff=-SMALL_BUFF)

        self.add(celestial_sphere, sun, orbits, glows, symbols)

        # Add line
        venus, earth = glows
        angles = [(180 / period) * TAU for period in [VENUS_ORBIT_PERIOD, EARTH_ORBIT_PERIOD]]
        for orbit, angle in zip(orbits, angles):
            orbit.rotate(-angle)

        line_of_sight = Line()
        line_of_sight.set_stroke(WHITE, 1)
        line_of_sight.f_always.put_start_and_end_on(earth.get_center, venus.get_center)
        line_of_sight.add_updater(lambda m: m.scale(5, about_point=m.get_start())),

        self.play(
            *(Rotate(orbit, angle) for orbit, angle in zip(orbits, angles)),
            VFadeIn(line_of_sight, time_span=(8, 10)),
            run_time=20,
            # rate_func=lambda t: 0.999 * bezier([0, 1, 1])(t),
            rate_func=lambda t: 0.999 * t,
        )
        line_of_sight.clear_updaters()
        self.play(
            frame.animate.reorient(-89, 82, 0, (-0.0, 0.01, -0.01), 0.49),
            run_time=5,
        )
        self.play(
            line_of_sight.animate.rotate(-0.6 * DEG, about_point=earth.get_center()),
            run_time=6,
        )
        self.wait()


class RadarToVenus(LightFromEarthToMoon):
    def construct(self):
        # Add sun, stars, and orbits
        frame = self.frame
        conversion_factor = 7 / EARTH_ORBIT_RADIUS
        celestial_sphere = get_celestial_sphere(radius=100, constellation_opacity=0.05)

        sun = get_sun(radius=SUN_RADIUS * conversion_factor, big_glow_ratio=20)
        sun.center()

        colors = [TEAL, BLUE]
        radii = np.array([VENUS_ORBIT_RADIUS, EARTH_ORBIT_RADIUS]) * conversion_factor
        angles = [160 * DEG, 180 * DEG]
        orbits = VGroup(
            Circle(radius=radius).rotate(angle).set_stroke(color, (0, 5))
            for radius, color, angle in zip(radii, colors, angles)
        )
        glows = Group(
            GlowDot(orbit.get_start(), color=orbit.get_color())
            for orbit in orbits
        )
        symbols = VGroup(*map(get_planet_symbols, ["venus", "earth"]))
        for symbol, glow in zip(symbols, glows):
            symbol.scale(0.5)
            symbol.rotate(110 * DEG, LEFT)
            symbol.rotate(-54 * DEG, OUT)
            symbol.next_to(glow, IN, buff=0)

        frame.reorient(-51, 60, 0, (-2.11, 1.24, -2.0), 7.66)
        frame.add_ambient_rotation(-1 * DEG)
        self.add(celestial_sphere, sun, orbits, glows, symbols)

        # Show pulses
        venus_point = orbits[0].get_start()
        earth_point = orbits[1].get_start()

        for _ in range(4):
            pulse = self.get_pulse(
                velocity=0.15 * (venus_point - earth_point),
                radius=0.05,
                max_stroke_width=2,
                low_x=earth_point[0],
                high_x=venus_point[0],
                tail_time=0.5,
            )
            pulse.move_to(earth_point)
            self.add(pulse)
            self.wait(0.2)
        self.wait(20)
