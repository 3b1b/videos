from manim_imports_ext import *


def example_pos_func(time):
    return np.array([-5, np.sin(time), 0])


def acceleration_from_position(pos_func, time, dt=0.01):
    p0 = pos_func(time - dt)
    p1 = pos_func(time)
    p2 = pos_func(time + dt)
    return (p0 + p2 - 2 * p1) / dt**2


def points_to_particle_info(points, particle, radius=None):
    """
    Given a set of points, and a particle, this returns:

    1) The unit vectors directed from the particle to each point

    2) The distances from the particle to each point

    3) An adjusted version of those distances where points
    within a given radius of the particle are considered to
    be farther away, approaching infinity at the particle's
    center. The intent is that when this is used for colomb/lorenz
    forces, field vectors within a radius of a particle don't
    blow up
    """
    diffs = points - particle.get_center()
    norms = np.linalg.norm(diffs, axis=1)[:, np.newaxis]
    unit_diffs = np.zeros_like(diffs)
    np.true_divide(diffs, norms, out=unit_diffs, where=(norms > 0))

    if radius is None:
        radius = particle.get_radius()
    adjusted_norms = norms.copy()
    mask = (0 < norms) & (norms < radius)
    adjusted_norms[mask] = radius * radius / norms[mask]
    adjusted_norms[norms == 0] = np.inf

    return unit_diffs, norms, adjusted_norms


def colomb_force(points, particle):
    unit_diffs, norms, adjusted_norms = points_to_particle_info(points, particle)
    return particle.get_charge() * unit_diffs / adjusted_norms**2


def lorentz_force(
    points,
    particle,
    # Takes in time, returns acceleration vector
    # for the charge at that time. Defaults to
    # particle.get_acceleration_at_time
    acceleration_func=None,
    radius=None,
    c=2.0,
    epsilon0=0.025,
):
    unit_diffs, norms, adjusted_norms = points_to_particle_info(
        points, particle, radius=radius
    )
    time = particle.get_internal_time()

    if acceleration_func is None:
        acceleration_func = particle.get_acceleration_at_time

    delays = norms[:, 0] / c
    acceleration = np.array([
        acceleration_func(time - delay)
        for delay in delays
    ])
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
        track_acceleration_history=False,
    ):
        self.charge = charge

        sphere = TrueDot(radius=radius, color=color)
        sphere.make_3d()
        sphere.move_to(point)
        super().__init__(sphere)
        self.sphere = sphere

        self.init_clock()
        self.add_updater(lambda m, dt: m.increment_clock(dt))

        if show_sign:
            sign = Tex(sign)
            sign.set_height(radius)
            sign.rotate(rotation, RIGHT)
            sign.set_stroke(WHITE, sign_stroke_width)
            sign.move_to(sphere)
            self.add(sign)
            self.sign = sign

        if track_acceleration_history:
            self.acceleration_history = []
            self.add_updater(lambda m, dt: m.add_to_acceleration_history(dt))

    def init_clock(self):
        self.clock = 0
        self.time_step = 1 / 30  # This will be updated
        self.recent_positions = np.tile(self.get_center(), 3).reshape((3, 3))

    def increment_clock(self, dt):
        if dt == 0:
            return self
        self.clock += dt
        self.time_step = dt
        self.recent_positions[0:2] = self.recent_positions[1:3]
        self.recent_positions[2] = self.get_center()

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

    def get_acceleration_at_time(self, time):
        if not hasattr(self, "acceleration_history"):
            raise Exception("track_acceleration_history is not turned on")

        steps = time / self.time_step
        index = int(steps)
        frac = steps % 1
        if 0 < index < len(self.acceleration_history) - 1:
            return interpolate(
                self.acceleration_history[index],
                self.acceleration_history[index + 1],
                frac
            )
        # return np.zeros(3)
        return np.array([0, 0.01, 0])

    def add_to_acceleration_history(self, dt):
        # TODO, we probably don't want to let this
        # list grow without bound. We should cap it some way?
        if dt > 0:
            self.acceleration_history.append(self.get_acceleration())
        return self


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
        **kwargs
    ):
        self.func = func
        self.stroke_width = stroke_width
        self.tip_width_ratio = tip_width_ratio
        self.tip_len_to_width = tip_len_to_width
        self.min_drawn_norm = min_drawn_norm

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
        tip_width = self.tip_width_ratio * self.get_stroke_width()
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

        # # Adjust stroke widths
        # width_arr = self.stroke_width * self.base_stroke_width_array
        # width_scalars = np.clip(drawn_norms / tip_len, 0, 1)
        # width_scalars = np.repeat(width_scalars, 8)[:-1]
        # self.get_stroke_widths()[:] = width_scalars * width_arr

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
        particles = ChargedParticle(rotation=0).replicate(5)
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
            track_acceleration_history=True
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
        # Show nearby neighboring particles
        charge = ChargedParticle(rotation=0)

        radius = 2
        neighbors = Group(*(
            ChargedParticle(radius * point, rotation=0)
            for point in compass_directions(8)
        ))

        force_arrows = VGroup()
        force_words = VGroup()
        for neighbor in neighbors:
            vect = normalize(neighbor.get_center() - charge.get_center())
            arrow = Vector(vect, stroke_width=5, stroke_color=BLUE)
            arrow.shift(neighbor.get_center() + neighbor.get_radius() * vect)
            force_arrows.add(arrow)

            force_word = Text("Force", font_size=24)
            force_word.next_to(ORIGIN, UP, SMALL_BUFF)
            angle = arrow.get_angle()
            if PI / 2 < angle < 3 * PI / 2:
                angle -= PI
            force_word.rotate(angle, about_point=ORIGIN)
            force_word.shift(arrow.get_center())
            force_words.add(force_word)

        self.play(
            FadeIn(charge, 0.25 * UP, lag_ratio=0.01),
        )
        self.play(LaggedStart(*(
            FadeIn(neighbor, shift=0.1 * neighbor.get_center())
            for neighbor in neighbors
        )))
        self.wait()

        self.play(
            LaggedStartMap(GrowArrow, force_arrows),
            LaggedStartMap(FadeIn, force_words),
        )

        # Draw electric field


class ShowTheEffectsOfOscillatingCharge(InteractiveScene):
    amplitude = 0.25
    frequency = 0.5
    direction = UP

    axes_config = dict(
        axis_config=dict(stroke_opacity=0.2),
    )
    particle_config = dict(
        track_acceleration_history=True,
        radius=0.2,
    )
    field_config = dict(
        max_vect_len=0.5,
        stroke_opacity=0.75,
        radius_of_suppression=1.0,
        height=10,
        x_density=4.0,
        y_density=4.0,
        c=2.0
    )

    def setup(self):
        super().setup()
        # Axes
        axes = ThreeDAxes(**self.axes_config)
        self.add(axes)

        # Oscillating charge
        particle = ChargedParticle(**self.particle_config)
        particle.add_updater(lambda m: m.move_to(self.oscillation_function(self.time)))

        field = LorentzField(particle, **self.field_config)

        self.add(field, particle)

    def construct(self):
        # Test
        self.wait(10)

    def oscillation_function(self, time):
        return self.amplitude * np.sin(TAU * self.frequency * time) * self.direction


class ChargeOnZAxis(ShowTheEffectsOfOscillatingCharge):
    default_frame_orientation = (-20, 70)
    direction = OUT
    particle_config = dict(
        rotation=PI / 2,
        track_acceleration_history=True,
        radius=0.2,
    )

    field_config = dict(
        max_vect_len=0.5,
        stroke_opacity=0.7,
        radius_of_suppression=1.0,
        height=14,
        depth=0,
        x_density=4.0,
        y_density=4.0,
        z_density=1.0,
        c=2.0
    )


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
        c=2.0
    )
