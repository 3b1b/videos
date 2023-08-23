from manim_imports_ext import *


def acceleration_from_position(pos_func, time, dt=1e-3):
    p0 = pos_func(time - dt)
    p1 = pos_func(time)
    p2 = pos_func(time + dt)
    return (p0 + p2 - 2 * p1) / dt**2


def points_to_particle_info(origin, points, radius):
    """
    Given an origin, a set of points, and a radius, this returns:

    1) The unit vectors directed from the origin to each point

    2) The distances from the origin to each point

    3) An adjusted version of those distances where points
    within a given radius of the origin are considered to
    be farther away, approaching infinity at the origin.
    The intent is that when this is used for colomb/lorenz
    forces, field vectors within a radius of a particle don't
    blow up
    """
    diffs = points - origin
    norms = np.linalg.norm(diffs, axis=1)[:, np.newaxis]
    unit_diffs = np.zeros_like(diffs)
    np.true_divide(diffs, norms, out=unit_diffs, where=(norms > 0))

    adjusted_norms = norms.copy()
    mask = (0 < norms) & (norms < radius)
    adjusted_norms[mask] = radius * radius / norms[mask]
    adjusted_norms[norms == 0] = np.inf

    return unit_diffs, norms, adjusted_norms


def colomb_force(points, particle, radius=None):
    if radius is None:
        radius = particle.get_radius()
    unit_diffs, norms, adjusted_norms = points_to_particle_info(particle.get_center(), points, radius)
    return particle.get_charge() * unit_diffs / adjusted_norms**2


def lorentz_force(
    points,
    particle,
    # Takes in time, returns acceleration vector
    # for the charge at that time. Defaults to
    # particle.get_past_acceleration
    acceleration_func=None,
    radius=None,
    c=2.0,
    epsilon0=0.025,
):
    if radius is None:
        radius = particle.get_radius()
    unit_diffs, norms, adjusted_norms = points_to_particle_info(particle.get_center(), points, radius)

    if acceleration_func is None:
        acceleration_func = particle.get_past_acceleration

    delays = norms[:, 0] / c
    if particle.track_position_history:
        # past_positions = np.array([
        #     particle.get_past_position(delay)
        #     for delay in delays
        # ])
        past_positions = particle.get_past_position(delays)
        unit_diffs = normalize_along_axis(points - past_positions, 1)
    acceleration = acceleration_func(delays)
    dot_prods = (unit_diffs * acceleration).sum(1)[:, np.newaxis]
    a_perp = acceleration - dot_prods * unit_diffs

    denom = 4 * PI * epsilon0 * c**2 * adjusted_norms
    return -particle.get_charge() * a_perp / denom


class ChargedParticle(Group):
    def __init__(
        self,
        point=ORIGIN,
        charge=1.0,
        color=RED,
        show_sign=True,
        sign="+",
        radius=0.2,
        rotation=0,
        sign_stroke_width=2,
        track_position_history=False,
        history_size=7200,
    ):
        self.charge = charge

        sphere = TrueDot(radius=radius, color=color)
        sphere.make_3d()
        sphere.move_to(point)
        super().__init__(sphere)
        self.sphere = sphere

        if show_sign:
            sign = Tex(sign)
            sign.set_height(radius)
            sign.rotate(rotation, RIGHT)
            sign.set_stroke(WHITE, sign_stroke_width)
            sign.move_to(sphere)
            self.add(sign)
            self.sign = sign

        self.track_position_history = track_position_history
        self.history_size = history_size

        self.init_clock()
        self.add_updater(lambda m, dt: m.increment_clock(dt))

    def init_clock(self):
        self.clock = 0
        self.time_step = 1 / 30  # This will be updated
        self.recent_positions = np.tile(self.get_center(), 3).reshape((3, 3))
        if self.track_position_history:
            self.position_history = np.zeros((self.history_size, 3))
            self.acceleration_history = np.zeros((self.history_size, 3))
            self.history_index = -1
            # self.n_history_changes = 0
            # self.position_history = []
            # self.acceleration_history = []

    def increment_clock(self, dt):
        if dt == 0:
            return self
        self.clock += dt
        self.time_step = dt
        self.recent_positions[0:2] = self.recent_positions[1:3]
        self.recent_positions[2] = self.get_center()
        if self.track_position_history:
            self.add_to_position_history()

    def add_to_position_history(self):
        self.history_index += 1
        hist_size = self.history_size
        # If overflowing, copy second half of history
        # lists to the first half, and reset index
        if self.history_index >= hist_size:
            for arr in [self.position_history, self.acceleration_history]:
                arr[:hist_size // 2, :] = arr[hist_size // 2:, :]
            self.history_index = (hist_size // 2) + 1

        self.position_history[self.history_index] = self.get_center()
        self.acceleration_history[self.history_index] = self.get_acceleration()
        return self

    def ignore_last_motion(self):
        self.recent_positions[:] = self.get_center()
        return self

    def get_charge(self):
        return self.charge

    def get_radius(self):
        return self.sphere.get_radius()

    def get_internal_time(self):
        return self.clock

    def scale(self, factor, *args, **kwargs):
        super().scale(factor, *args, **kwargs)
        self.sphere.set_radius(factor * self.sphere.get_radius())
        return self

    def get_acceleration(self):
        p0, p1, p2 = self.recent_positions
        if (p0 == p1).all() or (p1 == p2).all():
            # Otherwise, starts and stops have artificially
            # high acceleration
            return np.zeros(3)
        return (p0 + p2 - 2 * p1) / self.time_step**2

    def get_info_from_delays(self, info_arr, delays):
        if not hasattr(self, "acceleration_history"):
            raise Exception("track_position_history is not turned on")

        if len(info_arr) == 0:
            return np.zeros((len(delays), 3))

        pre_indices = self.history_index - delays / self.time_step
        indices = np.clip(pre_indices, 0, self.history_index).astype(int)

        return info_arr[indices]

    def get_past_acceleration(self, delays):
        return self.get_info_from_delays(self.acceleration_history, delays)

    def get_past_position(self, delays):
        return self.get_info_from_delays(self.position_history, delays)


class AccelerationVector(Vector):
    def __init__(
        self,
        particle,
        stroke_color=PINK,
        stroke_width=4,
        flat_stroke=False,
        norm_func=lambda n: np.tanh(n),
        **kwargs
    ):
        self.norm_func = norm_func

        super().__init__(
            RIGHT,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            flat_stroke=flat_stroke,
            **kwargs
        )
        self.add_updater(lambda m: m.pin_to_particle(particle))

    def pin_to_particle(self, particle):
        a_vect = particle.get_acceleration()
        norm = get_norm(a_vect)
        if self.norm_func is not None and norm > 0:
            a_vect = self.norm_func(norm) * a_vect / norm
        center = particle.get_center()
        self.put_start_and_end_on(center, center + a_vect)


class VectorField(VMobject):
    def __init__(
        self,
        func,
        color=BLUE,
        center=ORIGIN,
        x_density=2.0,
        y_density=2.0,
        z_density=2.0,
        width=14,
        height=8,
        depth=0,
        stroke_width: float = 2,
        tip_width_ratio: float = 4,
        tip_len_to_width: float = 0.01,
        max_vect_len: float | None = None,
        min_drawn_norm: float = 1e-2,
        flat_stroke=False,
        norm_to_opacity_func=None,
        norm_to_rgb_func=None,
        **kwargs
    ):
        self.func = func
        self.stroke_width = stroke_width
        self.tip_width_ratio = tip_width_ratio
        self.tip_len_to_width = tip_len_to_width
        self.min_drawn_norm = min_drawn_norm
        self.norm_to_opacity_func = norm_to_opacity_func
        self.norm_to_rgb_func = norm_to_rgb_func

        if max_vect_len is not None:
            self.max_vect_len = max_vect_len
        else:
            densities = np.array([x_density, y_density, z_density])
            dims = np.array([width, height, depth])
            self.max_vect_len = 1.0 / densities[dims > 0].mean()

        self.init_sample_points(
            center, width, height, depth,
            x_density, y_density, z_density
        )
        self.init_base_stroke_width_array(len(self.sample_points))

        super().__init__(
            stroke_color=color,
            flat_stroke=flat_stroke,
            **kwargs
        )

        n_samples = len(self.sample_points)
        self.set_points(np.zeros((8 * n_samples - 1, 3)))
        self.set_stroke(width=stroke_width)
        self.update_vectors()

    def init_sample_points(
        self,
        center: np.ndarray,
        width: float,
        height: float,
        depth: float,
        x_density: float,
        y_density: float,
        z_density: float
    ):
        to_corner = np.array([width / 2, height / 2, depth / 2])
        spacings = 1.0 / np.array([x_density, y_density, z_density])
        to_corner = spacings * (to_corner / spacings).astype(int)
        lower_corner = center - to_corner
        upper_corner = center + to_corner + spacings
        self.sample_points = cartesian_product(*(
            np.arange(low, high, space)
            for low, high, space in zip(lower_corner, upper_corner, spacings)
        ))

    def init_base_stroke_width_array(self, n_sample_points):
        arr = np.ones(8 * n_sample_points - 1)
        arr[4::8] = self.tip_width_ratio
        arr[5::8] = self.tip_width_ratio * 0.5
        arr[6::8] = 0
        arr[7::8] = 0
        self.base_stroke_width_array = arr

    def set_stroke(self, color=None, width=None, opacity=None, background=None, recurse=True):
        super().set_stroke(color, None, opacity, background, recurse)
        if width is not None:
            self.set_stroke_width(float(width))
        return self

    def set_stroke_width(self, width: float):
        if self.get_num_points() > 0:
            self.get_stroke_widths()[:] = width * self.base_stroke_width_array
            self.stroke_width = width
        return self

    def update_vectors(self):
        tip_width = self.tip_width_ratio * self.stroke_width
        tip_len = self.tip_len_to_width * tip_width
        samples = self.sample_points

        # Get raw outputs and lengths
        outputs = self.func(samples)
        norms = np.linalg.norm(outputs, axis=1)[:, np.newaxis]

        # How long should the arrows be drawn?
        max_len = self.max_vect_len
        if max_len < np.inf:
            drawn_norms = max_len * np.tanh(norms / max_len)
        else:
            drawn_norms = norms

        # What's the distance from the base of an arrow to
        # the base of its head?
        dist_to_head_base = np.clip(drawn_norms - tip_len, 0, np.inf)

        # Set all points
        unit_outputs = np.zeros_like(outputs)
        np.true_divide(outputs, norms, out=unit_outputs, where=(norms > self.min_drawn_norm))

        points = self.get_points()
        points[0::8] = samples
        points[2::8] = samples + dist_to_head_base * unit_outputs
        points[4::8] = points[2::8]
        points[6::8] = samples + drawn_norms * unit_outputs
        for i in (1, 3, 5):
            points[i::8] = 0.5 * (points[i - 1::8] + points[i + 1::8])
        points[7::8] = points[6:-1:8]

        # Adjust stroke widths
        width_arr = self.stroke_width * self.base_stroke_width_array
        width_scalars = np.clip(drawn_norms / tip_len, 0, 1)
        width_scalars = np.repeat(width_scalars, 8)[:-1]
        self.get_stroke_widths()[:] = width_scalars * width_arr

        # Potentially adjust opacity and color
        if self.norm_to_opacity_func is not None:
            self.get_stroke_opacities()[:] = self.norm_to_opacity_func(
                np.repeat(norms, 8)[:-1]
            )
        if self.norm_to_rgb_func is not None:
            self.get_stroke_colors()
            self.data['stroke_rgba'][:, :3] = self.norm_to_rgb_func(
                np.repeat(norms, 8)[:-1]
            )

        self.note_changed_data()
        return self


class ChargeBasedVectorField(VectorField):
    default_color = BLUE

    def __init__(self, *charges, **kwargs):
        self.charges = charges
        super().__init__(
            self.get_forces,
            color=kwargs.pop("color", self.default_color),
            **kwargs
        )
        self.add_updater(lambda m: m.update_vectors())

    def get_forces(self, points):
        # To be implemented in subclasses
        return np.zeros_like(points)


class ColombField(ChargeBasedVectorField):
    default_color = YELLOW

    def get_forces(self, points):
        return sum(
            colomb_force(points, charge)
            for charge in self.charges
        )


class LorentzField(ChargeBasedVectorField):
    def __init__(
        self, *charges,
        radius_of_suppression=None,
        c=2.0,
        **kwargs
    ):
        self.radius_of_suppression = radius_of_suppression
        self.c = c
        super().__init__(*charges, **kwargs)

    def get_forces(self, points):
        return sum(
            lorentz_force(
                points, charge,
                radius=self.radius_of_suppression,
                c=self.c
            )
            for charge in self.charges
        )


# Scenes


class TestFields(InteractiveScene):
    def construct(self):
        # Test colomb field
        particles = ChargedParticle(rotation=0).replicate(1)
        particles.arrange(DOWN)
        particles.move_to(6 * LEFT)

        field = ColombField(*particles)

        self.add(field, particles)
        self.play(particles.animate.move_to(0.2 * UP), run_time=3)

        self.clear()

        # Test Lorenz field
        def pos_func(time):
            return 0.1 * np.sin(5 * time) * OUT

        particle = ChargedParticle(
            rotation=0,
            radius=0.1,
            track_position_history=True
        )
        particles = particle.get_grid(20, 1, buff=0.25)
        particles.add_updater(lambda m: m.move_to(pos_func(self.time)))

        field = LorentzField(
            *particles,
            radius_of_suppression=1.0,
            x_density=4,
            y_density=4,
            max_vect_len=1,
            height=10,
        )
        field.set_stroke(opacity=0.7)

        self.frame.reorient(-20, 70)
        self.add(field, particles)
        self.wait(10)


class IntroduceEField(InteractiveScene):
    def construct(self):
        # Show two neighboring particles
        frame = self.frame
        frame.set_field_of_view(1 * DEGREES)

        charges = ChargedParticle(rotation=0).replicate(2)
        charges.arrange(RIGHT, buff=4)

        question = VGroup(
            Text("""
                How does the position
                and motion of this...
            """),
            Text("influence this?"),
        )
        for q, charge, vect in zip(question, charges, [LEFT, RIGHT]):
            q.next_to(charge, UP + vect, buff=1.0).shift(-2 * vect)

        question[1].align_to(question[0], DOWN)
        q0_bottom = question[0].get_bottom()
        arrow0 = always_redraw(lambda: Arrow(q0_bottom, charges[0]))
        arrow1 = Arrow(question[1].get_bottom(), charges[1])
        arrows = VGroup(arrow0, arrow1)

        self.play(LaggedStartMap(FadeIn, charges, shift=UP, lag_ratio=0.5))
        self.add(arrow0)
        self.play(
            Write(question[0]),
            charges[0].animate.shift(UR).set_anim_args(
                rate_func=wiggle,
                time_span=(1, 3),
            )
        )
        self.play(
            Write(question[1]),
            ShowCreation(arrow1),
        )
        self.wait()

        # Show force arrows
        def show_colomb_force(arrow, charge1, charge2):
            root = charge2.get_center()
            vect = 4 * colomb_force(
                charge2.get_center()[np.newaxis, :],
                charge1
            )[0]
            arrow.put_start_and_end_on(root, root + vect)

        colomb_vects = Vector(RIGHT, stroke_width=5, stroke_color=YELLOW).replicate(2)
        colomb_vects[0].add_updater(lambda a: show_colomb_force(a, *charges))
        colomb_vects[1].add_updater(lambda a: show_colomb_force(a, *charges[::-1]))

        self.add(*colomb_vects, *charges)
        self.play(
            FadeOut(question, time_span=(0, 1)),
            FadeOut(arrows, time_span=(0, 1)),
            charges.animate.arrange(RIGHT, buff=1.25),
            run_time=2
        )

        # Show force word
        force_words = Text("Force", font_size=48).replicate(2)
        force_words.set_fill(border_width=1)
        fw_width = force_words.get_width()

        def place_force_word_on_arrow(word, arrow):
            word.set_width(min(0.5 * arrow.get_width(), fw_width))
            word.next_to(arrow, UP, buff=0.2)

        force_words[0].add_updater(lambda w: place_force_word_on_arrow(w, colomb_vects[0]))
        force_words[1].add_updater(lambda w: place_force_word_on_arrow(w, colomb_vects[1]))

        self.play(LaggedStartMap(FadeIn, force_words, run_time=1, lag_ratio=0.5))
        self.add(force_words, charges)
        self.wait()

        # Add distance label
        d_line = always_redraw(lambda: DashedLine(
            charges[0].get_right(), charges[1].get_left(),
            dash_length=0.025
        ))
        d_label = Tex("r = 0.00", font_size=36)
        d_label.next_to(d_line, DOWN, buff=0.35)
        d_label.add_updater(lambda m: m.match_x(d_line))
        dist_decimal = d_label.make_number_changable("0.00")

        def get_d():
            return get_norm(charges[0].get_center() - charges[1].get_center())

        dist_decimal.add_updater(lambda m: m.set_value(get_d()))

        # Show graph
        axes = Axes((0, 10), (0, 1, 0.25), width=10, height=5)
        axes.shift(charges[0].get_center() + 1 * UP - axes.get_origin())
        axes.add(
            Text("Distance", font_size=36).next_to(axes.c2p(10, 0), UP),
            Text("Force", font_size=36).next_to(axes.c2p(0, 0.8), LEFT),
        )
        graph = axes.get_graph(lambda x: 0.5 / x**2, x_range=(0.01, 10, 0.05))
        graph.make_jagged()
        graph.set_stroke(YELLOW, 2)

        graph_dot = GlowDot(color=WHITE)
        graph_dot.add_updater(lambda d: d.move_to(axes.i2gp(get_d(), graph)))

        d_label.update()
        self.play(
            frame.animate.move_to([3.5, 2.5, 0.0]),
            LaggedStart(
                FadeIn(axes),
                ShowCreation(graph),
                FadeIn(graph_dot),
                ShowCreation(d_line),
                FadeIn(d_label, 0.25 * UP),
            ),
            run_time=2,
        )
        self.wait()

        for buff in (0.4, 8, 1.25):
            self.play(
                charges[1].animate.next_to(charges[0], RIGHT, buff=buff),
                run_time=4
            )
            self.wait()

        # Write Colomb's law
        colombs_law = Tex(R"""
            F = {q_1 q_2 \over 4 \pi \epsilon_0} \cdot \frac{1}{r^2}
        """)
        colombs_law_title = TexText("Colomb's law")
        colombs_law_title.move_to(axes, UP)
        colombs_law.next_to(colombs_law_title, DOWN, buff=0.75)

        rect = SurroundingRectangle(colombs_law["q_1 q_2"])
        rect.set_stroke(YELLOW, 2)
        rect.set_fill(YELLOW, 0.25)

        self.play(
            FadeIn(colombs_law_title),
            FadeIn(colombs_law, UP),
        )
        self.wait()
        self.add(rect, colombs_law)
        self.play(FadeIn(rect))
        self.wait()
        self.play(rect.animate.surround(colombs_law[R"4 \pi \epsilon_0"]))
        self.wait()
        self.play(rect.animate.surround(colombs_law[R"\frac{1}{r^2}"]))
        self.wait()
        self.play(charges[1].animate.next_to(charges[0], RIGHT, buff=3.0), run_time=3)
        self.play(FadeOut(rect))
        self.wait()

        # Remove graph
        d_line.clear_updaters()
        self.play(
            frame.animate.center(),
            VGroup(colombs_law, colombs_law_title).animate.to_corner(UL),
            LaggedStartMap(FadeOut, Group(
                axes, graph, graph_dot, d_line, d_label,
                force_words, colomb_vects
            )),
            charges[0].animate.center(),
            FadeOut(charges[1]),
            run_time=2,
        )
        self.wait()

        # Show Colomb's law vector field
        colombs_law.add_background_rectangle()
        colombs_law_title.add_background_rectangle()
        field = ColombField(charges[0], x_density=3.0, y_density=3.0)
        dots = DotCloud(field.sample_points, radius=0.025, color=RED)
        dots.make_3d()

        self.add(dots, colombs_law_title, colombs_law)
        self.play(ShowCreation(dots))
        self.wait()
        self.add(field, colombs_law_title, colombs_law)
        self.play(FadeIn(field))
        for vect in [2 * RIGHT, 4 * LEFT, 2 * RIGHT]:
            self.play(charges[0].animate.shift(vect).set_anim_args(path_arc=PI, run_time=3))
        self.wait()

        # Electric field
        e_colombs_law = Tex(R"""
            \vec{E}(\vec{r}) = {q \over 4 \pi \epsilon_0}
            \cdot \frac{1}{||\vec{r}||^2}
            \cdot \frac{\vec{r}}{||\vec{r}||}
        """)
        e_colombs_law.move_to(colombs_law, LEFT)
        ebr = BackgroundRectangle(e_colombs_law)
        r_vect = Vector(2 * RIGHT + UP)
        r_vect.set_stroke(GREEN)
        r_label = e_colombs_law[R"\vec{r}"][0].copy()
        r_label.next_to(r_vect.get_center(), UP, buff=0.1)
        r_label.set_backstroke(BLACK, 20)

        e_words = VGroup(
            Text("Electric Field:"),
            Text(
                """
                What force would be
                applied to a unit charge
                at a given point
                """,
                t2s={"would": ITALIC},
                t2c={"unit charge": RED},
                alignment="LEFT",
                font_size=36
            ),
        )
        e_words.set_backstroke(BLACK, 20)
        e_words.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        e_words.next_to(e_colombs_law, DOWN, buff=0.5)
        e_words.to_edge(LEFT, buff=MED_SMALL_BUFF)

        rect.surround(e_colombs_law[R"\vec{E}"])
        rect.scale(0.9, about_edge=DR)

        self.play(
            FadeOut(colombs_law, UP),
            FadeIn(ebr, UP),
            FadeIn(e_colombs_law, UP),
        )
        self.wait()
        self.add(ebr, rect, e_colombs_law)
        self.play(FadeIn(rect))
        self.play(Write(e_words, stroke_color=BLACK))
        self.wait()
        self.play(
            FadeOut(e_words),
            rect.animate.surround(e_colombs_law[R"(\vec{r})"][0], buff=0)
        )
        self.add(r_vect, charges[0])
        self.play(
            field.animate.set_stroke(opacity=0.4),
            FadeTransform(e_colombs_law[R"\vec{r}"][0].copy(), r_label),
            ShowCreation(r_vect),
        )
        self.wait()
        self.play(
            rect.animate.surround(e_colombs_law[R"\frac{\vec{r}}{||\vec{r}||}"])
        )
        self.wait()

        # Not the full story!
        words = Text("Not the full story!", font_size=60)
        arrow = Vector(LEFT)
        arrow.next_to(colombs_law_title, RIGHT)
        arrow.set_color(RED)
        words.set_color(RED)
        words.set_backstroke(BLACK, 20)
        words.next_to(arrow, RIGHT)
        charges[1].move_to(20 * RIGHT)

        self.remove(field)
        new_field = ColombField(*charges, x_density=3.0, y_density=3.0)
        new_field.set_stroke(opacity=float(field.get_stroke_opacity()))
        self.add(new_field)

        self.play(
            FadeIn(words, lag_ratio=0.1),
            ShowCreation(arrow),
            FadeOut(rect),
            FadeOut(r_vect),
            FadeOut(r_label),
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeOut, Group(
                ebr, dots, colombs_law_title, e_colombs_law,
                words, arrow,
            )),
            charges[0].animate.to_edge(LEFT, buff=1.0),
            charges[1].animate.to_edge(RIGHT, buff=1.0),
            run_time=3,
        )

        # Wiggle here -> wiggle there
        tmp_charges = Group(*(ChargedParticle(track_position_history=True, charge=0.3) for x in range(2)))
        tmp_charges[0].add_updater(lambda m: m.move_to(charges[0]))
        tmp_charges[1].add_updater(lambda m: m.move_to(charges[1]))
        for charge in tmp_charges:
            charge.ignore_last_motion()
        lorentz_field = LorentzField(
            *tmp_charges,
            x_density=6.0,
            y_density=6.0,
            norm_to_opacity_func=lambda n: np.clip(0.5 * n, 0, 0.75)
        )
        self.add(lorentz_field, *tmp_charges)

        influence_ring = self.get_influence_ring(charges[0].get_center())

        self.add(influence_ring, charges)
        wiggle_kwargs = dict(
            rate_func=lambda t: wiggle(t, 3),
            run_time=1.5,
            suspend_mobject_updating=False,
        )
        self.play(charges[0].animate.shift(UP).set_anim_args(**wiggle_kwargs))
        dist = get_norm(charges[1].get_center() - charges[0].get_center())
        self.wait_until(lambda: influence_ring.get_radius() > dist, max_time=dist / 2.0)
        self.play(charges[1].animate.shift(0.25 * DOWN).set_anim_args(**wiggle_kwargs))
        self.wait(4)
        self.play(
            FadeOut(influence_ring),
            FadeOut(new_field),
            FadeOut(lorentz_field)
        )
        self.remove(tmp_charges)

        # Show this the force
        ring = self.get_influence_ring(charges[0].get_center())

        ghost_charge = charges[0].copy().set_opacity(0.25)
        ghost_charge.shift(0.1 * IN)
        a_vect = Vector(UP).shift(charges[0].get_center())
        a_vect.set_stroke(PINK)
        a_label = Tex(R"\vec{a}(t_0)", font_size=48)
        a_label.set_color(PINK)
        a_label.next_to(a_vect, RIGHT, SMALL_BUFF)

        f_vect = Vector(1.0 * DOWN).shift(charges[1].get_center())
        f_vect.set_stroke(BLUE)
        f_label = Tex(R"\vec{F}(t)")
        f_label.set_color(BLUE)
        f_label.next_to(f_vect, LEFT, buff=0.15)

        time_label = Tex("t = 0.00")
        time_label.to_corner(UL)
        time_decimal = time_label.make_number_changable("0.00")
        time_decimal.add_updater(lambda m: m.set_value(ring.time))

        start_point = charges[0].get_center().copy()
        speed = 2.0

        def field_func(points):
            time = ring.time
            diffs = (points - start_point)
            norms = np.linalg.norm(diffs, axis=1)
            past_times = time - (norms / speed)
            mags = np.exp(-3 * past_times)
            mags[past_times < 0] = 0
            return mags[:, np.newaxis] * DOWN

        field = VectorField(
            field_func,
            height=0,
            x_density=4.0,
            max_vect_len=1.0,
        )
        field.add_updater(lambda f: f.update_vectors())

        self.add(time_label, a_vect, a_label, charges)
        self.wait()
        self.add(ring, ghost_charge, field, charges)

        target = charges[0].get_center() + 2 * UP
        charges[0].add_updater(lambda m, dt: m.shift(3 * dt * (target - m.get_center())))
        self.wait_until(lambda: ring.get_radius() > dist)

        self.add(f_vect, f_label, charges)
        ring.suspend_updating()
        charges[0].suspend_updating()
        self.add(f_vect, charges[1])
        self.play(
            FadeIn(f_vect),
            FadeIn(f_label),
            FadeOut(field),
        )

        # Write the Lorentz force
        lorentz_law = Tex(R"""
            \vec{F}(t) = 
            {-q_1 q_2 \over 4\pi \epsilon_0 c^2}
            {1 \over r}
            \vec{a}_\perp(t - r / c)
        """)
        lorentz_law.to_edge(UP)
        lorentz_law[R"\vec{F}(t)"][0].match_style(f_label)

        a_hat_perp = lorentz_law[R"\vec{a}_\perp"][0]
        a_hat_perp.match_style(a_label)
        a_hat_perp.save_state()
        a_hat_perp[2].set_opacity(0)
        a_hat_perp[:2].move_to(a_hat_perp, RIGHT)
        a_hat_perp[:2].scale(1.25, about_edge=DR)

        lorentz_law["("][1].match_style(a_label)
        lorentz_law[")"][1].match_style(a_label)

        self.play(
            Transform(
                f_label.copy(),
                lorentz_law[R"\vec{F}(t)"][0].copy(),
                remover=True,
                run_time=1.5,
            ),
            FadeIn(lorentz_law, time_span=(1, 2))
        )
        self.wait()

        # Go through parts of the equation
        rect = SurroundingRectangle(lorentz_law["-q_1 q_2"])
        rect.set_stroke(YELLOW, 2)
        rect.set_fill(YELLOW, 0.2)

        r_line = DashedLine(ghost_charge.get_right(), charges[1].get_left())
        r_label = Tex("r").next_to(r_line, UP)

        self.add(rect, lorentz_law)
        self.play(FadeIn(rect))
        self.wait()
        self.play(rect.animate.surround(lorentz_law[R"4\pi \epsilon_0 c^2"]))
        self.wait()
        self.play(
            rect.animate.surround(lorentz_law[R"{1 \over r}"]),
            ShowCreation(r_line),
        )
        self.play(TransformFromCopy(lorentz_law[R"r"][1], r_label))
        self.wait()
        self.play(rect.animate.surround(lorentz_law[R"\vec{a}_\perp(t - r / c)"]))
        self.wait()
        self.play(rect.animate.surround(lorentz_law[R"t - r / c"], buff=0.05))
        self.wait()

        # Indicate back in time
        new_a_label = Tex(R"\vec{a}(t - r / c)")
        new_a_label.match_style(a_label)
        new_a_label.move_to(a_label, LEFT)

        ring.clear_updaters()
        time_decimal.clear_updaters()
        charges[0].clear_updaters()
        self.add(charges[0])
        self.play(
            ring.animate.scale(1e-3),
            UpdateFromFunc(time_decimal, lambda m: m.set_value(
                ring.get_radius() / 2
            )),
            charges[0].animate.shift(2 * DOWN).set_anim_args(
                time_span=(1, 4),
                rate_func=lambda t: smooth(t)**0.5,
            ),
            run_time=4,
        )
        time_decimal.set_value(0)
        self.play(
            TransformMatchingStrings(a_label, new_a_label),
            FadeOut(rect),
        )
        self.remove(rect)
        self.remove(ring)

        # Do another wiggle
        ring = self.get_influence_ring(charges[0].get_center())
        time_decimal.add_updater(lambda m: m.set_value(ring.time))

        self.add(ring)
        self.play(charges[0].animate.shift(UP).set_anim_args(**wiggle_kwargs))
        self.wait_until(lambda: ring.get_radius() > dist)
        self.play(charges[1].animate.shift(0.5 * DOWN).set_anim_args(**wiggle_kwargs))
        self.remove(ring)
        self.play(FadeOut(time_label))

        # Add back perpenducular part
        charges.target = charges.generate_target()
        charges.target.arrange(UR, buff=3).center()
        r_line.target = r_line.generate_target()
        r_line.target.become(DashedLine(
            charges.target[0].get_center(),
            charges.target[1].get_center(),
        ))
        f_vect.target = f_vect.generate_target()
        f_vect.target.rotate(45 * DEGREES)
        f_vect.target.shift(charges.target[1].get_center() - f_vect.target.get_start())
        rect = SurroundingRectangle(a_hat_perp.saved_state, buff=0.1)
        rect.set_stroke(YELLOW, 2)
        rect.set_fill(YELLOW, 0.25)

        self.add(rect, lorentz_law)
        self.play(FadeIn(rect, scale=0.5))
        self.play(Restore(a_hat_perp))
        self.wait()

        self.remove(ghost_charge)
        self.play(
            MoveToTarget(charges),
            MoveToTarget(r_line),
            MoveToTarget(f_vect),
            r_label.animate.next_to(r_line.target.get_center(), UL, SMALL_BUFF),
            f_label.animate.next_to(f_vect.target.get_center(), UR, buff=0),
            new_a_label.animate.next_to(charges.target[0], UL, buff=0),
            MaintainPositionRelativeTo(a_vect, charges[0]),
            run_time=2
        )
        self.wait()

        r_unit = normalize(charges[1].get_center() - charges[0].get_center())
        a_perp_vect = Vector(
            a_vect.get_vector() - np.dot(a_vect.get_vector(), r_unit) * r_unit,
        )
        a_perp_vect.match_style(a_vect)
        a_perp_vect.set_stroke(interpolate_color(PINK, WHITE, 0.5))
        a_perp_vect.shift(a_vect.get_end() - a_perp_vect.get_end())

        a_hat_perp2 = a_hat_perp.copy()
        a_hat_perp2.scale(0.9)
        a_hat_perp2.next_to(a_perp_vect.get_center(), UR, buff=0.1)
        a_hat_perp2.match_color(a_perp_vect)

        self.play(TransformFromCopy(a_vect, a_perp_vect))
        self.play(TransformFromCopy(a_hat_perp, a_hat_perp2))
        self.wait()
        rings = VGroup()
        for x in range(2):
            wiggle_kwargs = dict(
                run_time=2,
                rate_func=lambda t: wiggle(t, 5)
            )
            ring = self.get_influence_ring(charges[0].get_center())
            rings.add(ring)
            dist = get_norm(charges[0].get_center() - charges[1].get_center())

            self.add(ring)
            self.play(charges[0].animate.shift(0.5 * UP).set_anim_args(**wiggle_kwargs))
            self.wait_until(lambda: ring.get_radius() > dist)
            self.play(charges[1].animate.shift(0.25 * DR).set_anim_args(**wiggle_kwargs))
        self.play(FadeOut(rings))

        # Clear the canvas
        plane = NumberPlane(
            background_line_style=dict(stroke_color=GREY_D, stroke_opacity=0.75, stroke_width=1),
            axis_config=dict(stroke_opacity=(0.25))
        )
        new_lorentz = Tex(R"""
            \vec{E}_{\text{rad}}(\vec{r}, t) = 
            {-q \over 4\pi \epsilon_0 c^2}
            {1 \over ||\vec{r}||}
            \vec{a}_\perp(t - ||\vec{r}|| / c)
        """, font_size=36)
        new_lorentz.to_corner(UL)
        lhs = new_lorentz[R"\vec{E}_{\text{rad}}(\vec{r}, t)"]
        lhs.set_color(BLUE)
        new_lorentz[R"\vec{a}_\perp("].set_color(PINK)
        new_lorentz[R")"][1].set_color(PINK)

        lhs_rect = SurroundingRectangle(lhs)
        arrow = Vector(UP).next_to(lhs_rect, DOWN)

        self.add(plane, lorentz_law, *charges)
        self.remove(rect)
        self.play(
            LaggedStartMap(FadeOut, Group(
                r_line, r_label,
                a_hat_perp2, a_perp_vect,
                a_vect, new_a_label, new_a_label,
                f_vect, f_label, charges[1],
            )),
            FadeIn(plane, time_span=(1, 2)),
            charges[0].animate.center().set_anim_args(time_span=(1, 2)),
            FadeTransform(lorentz_law, new_lorentz),
        )
        self.play(
            ShowCreation(lhs_rect),
            GrowArrow(arrow),
        )
        self.wait()
        self.play(FadeOut(lhs_rect), FadeOut(arrow))

        # Show vector field
        charge = ChargedParticle(
            track_position_history=True
        )
        field = LorentzField(
            charge,
            stroke_width=3,
            x_density=4.0,
            y_density=4.0,
            max_vect_len=0.25,
            norm_to_opacity_func=lambda n: np.clip(1.5 * n, 0, 1),
        )
        a_vect = AccelerationVector(charge)
        small_charges = DotCloud(field.sample_points, radius=0.02)
        small_charges.match_color(charges[1][0])
        small_charges.make_3d()
        new_lorentz.set_backstroke(BLACK, 20)

        self.add(small_charges, new_lorentz)
        self.play(ShowCreation(small_charges))
        self.wait()

        self.remove(charges[0])
        self.add(field, a_vect, charge, new_lorentz)
        charge.ignore_last_motion()
        wiggle_kwargs = dict(
            rate_func=lambda t: wiggle(t, 3),
            run_time=3.0,
            suspend_mobject_updating=False,
        )
        self.play(
            charge.animate.shift(0.4 * UP).set_anim_args(**wiggle_kwargs),
        )
        self.wait(8)

        charge.init_clock()
        charge.ignore_last_motion()
        charge.add_updater(lambda m: m.move_to(
            0.25 * np.sin(0.5 * TAU * m.get_internal_time()) * UP
        ))
        self.wait(30)

    def get_influence_ring(self, center_point, color=WHITE, speed=2.0, max_width=3.0, width_decay_exp=0.5):
        ring = Circle()
        ring.set_stroke(color)
        ring.move_to(center_point)
        ring.time = 0

        def update_ring(ring, dt):
            ring.time += dt
            radius = ring.time * speed
            ring.set_width(max(2 * radius, 1e-3))
            ring.set_stroke(width=max_width / (1 + radius)**width_decay_exp)
            return ring

        ring.add_updater(update_ring)
        return ring


class ShowTheEffectsOfOscillatingCharge(InteractiveScene):
    amplitude = 0.25
    frequency = 0.5
    direction = UP

    show_acceleration_vector = True
    origin = None

    axes_config = dict(
        axis_config=dict(stroke_opacity=0.7),
        x_range=(-10, 10),
        y_range=(-5, 5),
        z_range=(-3, 3),
    )
    particle_config = dict(
        track_position_history=True,
        radius=0.15,
    )
    acceleration_vector_config = dict()
    field_config = dict(
        max_vect_len=0.35,
        stroke_opacity=0.75,
        radius_of_suppression=1.0,
        height=10,
        x_density=4.0,
        y_density=4.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(2 * n, 0, 0.8)
    )

    def setup(self):
        super().setup()
        self.add_axes()
        self.add_axis_labels(self.axes)
        self.add_particles(self.axes)
        self.add_field(self.particles)
        if self.show_acceleration_vector:
            self.add_acceleration_vectors(self.particles)

    def add_axes(self):
        self.axes = ThreeDAxes(**self.axes_config)
        if self.origin is not None:
            self.axes.shift(self.origin - self.axes.get_origin())
        self.add(self.axes)

    def add_axis_labels(self, axes):
        axis_labels = label = Tex("xyz")
        if axes.z_axis.get_stroke_opacity() > 0:
            axis_labels.rotate(PI / 2, RIGHT)
            axis_labels[0].next_to(axes.x_axis.get_right(), OUT)
            axis_labels[1].next_to(axes.y_axis.get_top(), OUT)
            axis_labels[2].next_to(axes.z_axis.get_zenith(), RIGHT)
        else:
            axis_labels[1].clear_points()
            axis_labels[0].next_to(axes.x_axis.get_right(), UP)
            axis_labels[2].next_to(axes.y_axis.get_top(), RIGHT)

        self.axis_labels = axis_labels
        self.add(self.axis_labels)

    def add_particles(self, axes):
        self.particles = self.get_particles()
        self.particles.add_updater(lambda m: m.move_to(
            axes.c2p(*self.oscillation_function(self.time))
        ))
        for particle in self.particles:
            particle.ignore_last_motion()
        self.add(self.particles)

    def get_particles(self):
        return Group(ChargedParticle(**self.particle_config))

    def add_field(self, particles):
        self.field = LorentzField(*particles, **self.field_config)
        self.add(self.field, particles)

    def add_acceleration_vectors(self, particles):
        self.acceleration_vectors = VGroup(*(
            AccelerationVector(particle)
            for particle in particles
        ))
        self.add(self.acceleration_vectors, self.particles)

    def oscillation_function(self, time):
        return self.amplitude * np.sin(TAU * self.frequency * time) * self.direction

    def construct(self):
        # Test
        self.wait(20)


class OscillateOnYOneDField(ShowTheEffectsOfOscillatingCharge):
    origin = 5 * LEFT
    axes_config = dict(
        axis_config=dict(stroke_opacity=0.7),
        z_axis_config=dict(stroke_opacity=0),
        x_range=(-3, 12),
        y_range=(-3, 3)
    )
    field_config = dict(
        max_vect_len=1,
        stroke_opacity=1.0,
        radius_of_suppression=0.25,
        height=0,
        x_density=4.0,
        c=2.0,
        norm_to_opacity_func=None
    )

    def construct(self):
        # Start wiggling
        axes = self.axes
        field = self.field
        particles = self.particles

        points = DotCloud(field.sample_points, color=BLUE)
        points.make_3d()
        points.set_radius(0.03)
        field.suspend_updating()
        particles.suspend_updating()

        self.add(points, particles)
        self.play(ShowCreation(points))
        self.wait()
        self.time = 0
        particles.resume_updating()
        for particle in particles:
            particle.ignore_last_motion()
        field.resume_updating()
        self.wait(24.5)
        paused_time = float(self.time)

        # Zoom in
        field.suspend_updating()
        particles.suspend_updating()
        self.remove(particles)
        self.remove(field)
        field_copy = field.copy()
        field_copy.clear_updaters()
        particle = particles[0].copy()
        particle.clear_updaters()
        self.add(field_copy, particle)

        frame = self.frame
        particle.save_state()
        particle.target = particle.generate_target()
        particle.target[0].set_radius(0.075)
        particle.target[1].scale(0.5)
        particle.target[1].set_stroke(width=1)

        self.play(
            frame.animate.set_height(3, about_point=axes.get_origin()),
            MoveToTarget(particle),
            self.acceleration_vectors.animate.set_stroke(opacity=0.2),
            run_time=2
        )

        # Go through points
        last_line = VMobject()
        last_ghost = Group()
        step = get_norm(field.sample_points[0] - field.sample_points[1])
        for x in np.arange(1, 9):
            ghost = particle.copy()
            ghost.fade(0.5)
            dist = get_norm(axes.c2p(x * step, 0) - particle.get_center())
            ghost.move_to(particle.get_past_position(dist / field.c))
            line = Line(
                ghost.get_center(),
                axes.c2p(x * step, 0)
            )
            line.set_stroke(WHITE, 1)
            elbow = Elbow(width=0.1)
            angle = line.get_angle() + 90 * DEGREES
            if x > 3:
                angle += 90 * DEGREES
            elbow.rotate(angle, about_point=ORIGIN)
            elbow.shift(line.get_end())
            elbow.set_stroke(WHITE, 1)
            self.play(
                ShowCreation(line),
                FadeOut(last_line),
                FadeOut(last_ghost, scale=0),
                GrowFromCenter(ghost),
                FadeIn(elbow, time_span=(0.5, 1)),
            )
            self.wait(0.5)
            last_line = Group(line, elbow)
            last_ghost = ghost
        self.play(FadeOut(last_line))

        self.time = paused_time
        self.play(
            Restore(particle),
            frame.animate.to_default_state().set_anim_args(run_time=3)
        )
        self.remove(field_copy, particle)
        self.add(particles, field)
        self.wait(5)


class OscillateOnYTwoDField(ShowTheEffectsOfOscillatingCharge):
    particle_config = dict(
        track_position_history=True,
        radius=0.15,
    )
    field_config = dict(
        max_vect_len=0.25,
        stroke_opacity=0.75,
        radius_of_suppression=0.25,
        height=10,
        x_density=4.0,
        y_density=4.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(1.5 * n, 0, 1.0)
    )

    def construct(self):
        # Start wiggling
        axes = self.axes
        field = self.field
        particles = self.particles

        self.wait(60)


class DiscussDecay(OscillateOnYOneDField):
    def construct(self):
        # Start wiggling
        axes = self.axes
        particles = self.particles
        self.wait(8)

        # Show graph
        axes_config = dict(self.axes_config)
        axes_config.pop("z_axis_config")
        axes2d = Axes(**axes_config)
        axes2d.shift(axes.get_origin() - axes2d.get_origin())
        graph = axes2d.get_graph(lambda x: 2 / x, x_range=(0.5, 12))
        graph.set_stroke(TEAL, 2)

        words = TexText(R"Decays proportionally to $\frac{1}{r}$")
        words[R"$\frac{1}{r}$"].scale(1.5, about_edge=LEFT).set_color(TEAL)
        words.move_to(2 * UP)

        particles[0].ignore_last_motion()
        self.play(
            ShowCreation(graph),
            Write(words),
        )
        self.wait(20)


class ChargeOnZAxis(ShowTheEffectsOfOscillatingCharge):
    default_frame_orientation = (-20, 70)
    direction = OUT

    origin = ORIGIN

    axes_config = dict(
        axis_config=dict(stroke_opacity=0.7),
        x_range=(-8, 8),
        y_range=(-6, 6),
        z_range=(-3, 3),
    )
    particle_config = dict(
        show_sign=False,
        rotation=PI / 2,
        track_position_history=True,
        radius=0.2,
    )
    field_config = dict(
        max_vect_len=0.5,
        stroke_opacity=0.7,
        radius_of_suppression=1.0,
        width=40,
        height=40,
        depth=0,
        x_density=4.0,
        y_density=4.0,
        z_density=1.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(n, 0, 0.8)
    )

    def construct(self):
        # Test
        self.play(self.frame.animate.reorient(16, 71, 0), run_time=12)
        self.play(self.frame.animate.reorient(-15, 84, 0), run_time=6)
        self.play(self.frame.animate.reorient(-38, 64, 0), run_time=10)
        self.play(self.frame.animate.reorient(24, 66, 0), run_time=10)


class RowOfCharges(ChargeOnZAxis):
    n_charges = 17
    particle_buff = 0.25
    particle_config = dict(
        rotation=PI / 2,
        track_position_history=True,
        radius=0.1,
        show_sign=False,
        charge=0.15
    )
    field_config = dict(
        max_vect_len=0.5,
        stroke_opacity=0.7,
        radius_of_suppression=1.0,
        width=30,
        height=30,
        depth=0,
        x_density=4.0,
        y_density=4.0,
        z_density=1.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(1.5 * n, 0, 0.8)
    )
    show_acceleration_vector = False
    def construct(self):
        # Test
        self.play(self.frame.animate.reorient(-7, 62, 0).set_height(16), run_time=12)
        self.play(self.frame.animate.reorient(26, 70, 0), run_time=12)
        self.play(self.frame.animate.reorient(-26, 70, 0), run_time=12)

    def get_particles(self):
        return Group(*(
            ChargedParticle(**self.particle_config)
            for n in range(self.n_charges)
        )).arrange(UP, buff=self.particle_buff)


class RowOfChargesMoreCharges(RowOfCharges):
    n_charges = 100
    particle_buff = 0.01

    particle_config = dict(
        rotation=PI / 2,
        track_position_history=True,
        radius=0.05,
        show_sign=False,
        charge=0.0,
    )
    field_config = dict(
        max_vect_len=0.5,
        stroke_opacity=0.7,
        radius_of_suppression=1.0,
        width=30,
        height=30,
        depth=0,
        x_density=4.0,
        y_density=4.0,
        z_density=1.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(1.5 * n, 0, 0.8)
    )


class RowOfChargesXAxis(RowOfCharges):
    field_config = dict(
        max_vect_len=1.0,
        stroke_opacity=0.7,
        radius_of_suppression=0.25,
        width=40,
        height=0,
        depth=0,
        x_density=8.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(1.5 * n, 0, 0.8)
    )
    axes_config = dict(
        axis_config=dict(stroke_opacity=0.7),
        x_range=(-20, 20),
        y_range=(-6, 6),
        z_range=(-3, 3),
    )

    def setup(self):
        super().setup()
        self.frame.reorient(-26, 70, 0).set_height(16)
        self.axis_labels[0].set_x(8)

    def construct(self):
        # Form the field
        self.wait(20)

        # Zoom in
        self.play(
            self.frame.animate.reorient(-15, 84, 0).move_to([4.36, -1.83, 0.37]).set_height(5.59),
            run_time=3
        )

        # Show graph
        axes_kw = dict(self.axes_config)
        axes_kw.pop("z_range")
        axes = Axes(**axes_kw)
        graph1 = axes.get_graph(lambda r: 2.0 / r, x_range=(0.01, 20, 0.1))
        graph2 = axes.get_graph(lambda r: 1.0 / r**0.3, x_range=(0.01, 20, 0.1))
        graphs = VGroup(graph1, graph2)
        graphs.rotate(PI / 2, RIGHT, about_point=axes.get_origin())
        graphs.set_flat_stroke(False)
        graphs.set_stroke(TEAL, 2)

        words = VGroup(
            TexText(R"Instead of decaying like $\frac{1}{r}$"),
            TexText(R"It decays much more gently"),
        )
        words.fix_in_frame()
        words.to_edge(UP, buff=1.5)

        self.play(
            ShowCreation(graph1, run_time=2),
            FadeIn(words[0], 0.5 * UP)
        )
        self.wait()
        self.play(
            FadeOut(words[0], 0.5 * UP),
            FadeIn(words[1], 0.5 * UP),
            Transform(*graphs)
        )
        self.wait(6)


class RowOfChargesXAxisMoreCharges(RowOfChargesXAxis):
    n_charges = 100
    particle_buff = 0.1
    particle_config = dict(
        rotation=PI / 2,
        track_position_history=True,
        radius=0.05,
        show_sign=False,
        charge=3.0 / 50,
    )

    def construct(self):
        # Test
        self.wait(12)  # Let the field form


class WavesIn3D(ChargeOnZAxis):
    field_config = dict(
        max_vect_len=0.5,
        stroke_opacity=0.25,
        radius_of_suppression=1.0,
        height=10,
        depth=10,
        x_density=4.0,
        y_density=4.0,
        z_density=1.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(0.5 * n, 0, 0.8)
    )


class WiggleHereWiggleThere(IntroduceEField):
    def construct(self):
        # Setup
        charges = Group(*(
            ChargedParticle(track_position_history=True)
            for _ in range(2)
        ))
        charges[0].to_edge(LEFT)
        charges[1].to_edge(RIGHT)
        dist = get_norm(charges[0].get_center() - charges[1].get_center())
        for charge in charges:
            charge.ignore_last_motion()

        field = LorentzField(
            *charges,
            x_density=6.0,
            y_density=6.0,
            norm_to_opacity_func=lambda n: np.clip(0.5 * n, 0, 0.75)
        )
        self.add(field)
        self.add(*charges)

        # Wiggles
        ring1 = self.get_influence_ring(charges[0].get_center())
        self.add(ring1)
        wiggle_kwargs = dict(
            rate_func=lambda t: wiggle(t, 3),
            run_time=1.5,
            suspend_mobject_updating=False,
        )
        self.play(charges[0].animate.shift(UP).set_anim_args(**wiggle_kwargs))
        self.wait_until(lambda: ring1.get_radius() > dist, max_time=dist / 2.0)

        ring2 = self.get_influence_ring(charges[1].get_center())
        self.add(ring2)
        self.play(charges[1].animate.shift(0.5 * DOWN).set_anim_args(**wiggle_kwargs))
        self.wait_until(lambda: ring2.get_radius() > dist, max_time=dist / 2.0)

        self.play(charges[0].animate.shift(0.25 * UP).set_anim_args(**wiggle_kwargs))
        self.wait(6)
