from __future__ import annotations

from manim_imports_ext import *
from matplotlib import colormaps


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from manimlib.typing import Vect3


spectral_cmap = colormaps.get_cmap("Spectral")

# Helper functions


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


def acceleration_from_position(pos_func, time, dt=1e-3):
    p0 = pos_func(time - dt)
    p1 = pos_func(time)
    p2 = pos_func(time + dt)
    return (p0 + p2 - 2 * p1) / dt**2


def points_to_particle_info(particle, points, radius=None, c=2.0):
    """
    Given an origin, a set of points, and a radius, this returns:

    1) The unit vectors directed from the origin to each point

    2) The distances from the origin to each point

    3) An adjusted version of those distances where points
    within a given radius of the origin are considered to
    be farther away, approaching infinity at the origin.
    The intent is that when this is used for coulomb/lorenz
    forces, field vectors within a radius of a particle don't
    blow up
    """
    if radius is None:
        radius = particle.get_radius()

    if particle.track_position_history:
        approx_delays = np.linalg.norm(points - particle.get_center(), axis=1) / c
        centers = particle.get_past_position(approx_delays)
    else:
        centers = particle.get_center()

    diffs = points - centers
    norms = np.linalg.norm(diffs, axis=1)[:, np.newaxis]
    unit_diffs = np.zeros_like(diffs)
    np.true_divide(diffs, norms, out=unit_diffs, where=(norms > 0))

    adjusted_norms = norms.copy()
    mask = (0 < norms) & (norms < radius)
    adjusted_norms[mask] = radius * radius / norms[mask]
    adjusted_norms[norms == 0] = np.inf

    return unit_diffs, norms, adjusted_norms


def coulomb_force(points, particle, radius=None):
    unit_diffs, norms, adjusted_norms = points_to_particle_info(particle, points, radius)
    return particle.get_charge() * unit_diffs / adjusted_norms**2


def lorentz_force(
    points,
    particle,
    radius=None,
    c=2.0,
    epsilon0=0.025,
):
    unit_diffs, norms, adjusted_norms = points_to_particle_info(particle, points, radius, c)
    delays = norms[:, 0] / c

    acceleration = particle.get_past_acceleration(delays)
    dot_prods = (unit_diffs * acceleration).sum(1)[:, np.newaxis]
    a_perp = acceleration - dot_prods * unit_diffs

    denom = 4 * PI * epsilon0 * c**2 * adjusted_norms
    return -particle.get_charge() * a_perp / denom


# For the cylinder


class OscillatingWave(VMobject):
    def __init__(
        self,
        axes,
        y_amplitude=0.0,
        z_amplitude=0.75,
        z_phase=0.0,
        y_phase=0.0,
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
        self.y_amplitude = y_amplitude
        self.z_amplitude = z_amplitude
        self.z_phase = z_phase
        self.y_phase = y_phase
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
        self.clock_is_stopped = False

        self.add_updater(lambda m, dt: m.update_points(dt))

    def update_points(self, dt):
        if not self.clock_is_stopped:
            self.time += dt
        xs = np.arange(
            self.axes.x_axis.x_min,
            self.axes.x_axis.x_max,
            self.sample_resolution
        )
        self.set_points_as_corners(
            self.offset + self.xt_to_point(xs, self.time)
        )

    def stop_clock(self):
        self.clock_is_stopped = True

    def start_clock(self):
        self.clock_is_stopped = False

    def xt_to_yz(self, x, t):
        phase = TAU * t * self.speed / self.wave_len
        y_outs = self.y_amplitude * np.sin(TAU * x / self.wave_len - phase - self.y_phase)
        z_outs = self.z_amplitude * np.sin(TAU * x / self.wave_len - phase - self.z_phase)
        twist_angles = x * self.twist_rate * TAU
        y = np.cos(twist_angles) * y_outs - np.sin(twist_angles) * z_outs
        z = np.sin(twist_angles) * y_outs + np.cos(twist_angles) * z_outs

        return y, z

    def xt_to_point(self, x, t):
        y, z = self.xt_to_yz(x, t)
        return self.axes.c2p(x, y, z)

    def get_default_color(self, wave_len):
        return get_spectral_color(inverse_interpolate(
            1.5, 0.5, wave_len
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


# For fields


class ChargedParticle(Group):
    def __init__(
        self,
        point=ORIGIN,
        charge=1.0,
        mass=1.0,
        color=RED,
        show_sign=True,
        sign="+",
        radius=0.2,
        rotation=0,
        sign_stroke_width=2,
        track_position_history=True,
        history_size=7200,
        euler_steps_per_frame=10,
    ):
        self.charge = charge
        self.mass = mass

        sphere = TrueDot(radius=radius, color=color)
        sphere.make_3d()
        sphere.move_to(point)
        self.sphere = sphere

        self.track_position_history = track_position_history
        self.history_size = history_size
        self.velocity = np.zeros(3)  # Only used if force are added
        self.euler_steps_per_frame = euler_steps_per_frame
        self.init_clock(point)

        super().__init__(sphere)

        if show_sign:
            sign = Tex(sign)
            sign.set_height(radius)
            sign.rotate(rotation, RIGHT)
            sign.set_stroke(WHITE, sign_stroke_width)
            sign.move_to(sphere)
            self.add(sign)
            self.sign = sign

    # Related to updaters

    def update(self, dt: float = 0, recurse: bool = True):
        super().update(dt, recurse)
        # Do this instead of adding an updater, because
        # otherwise all animations require the
        # suspend_mobject_updating=false flag
        self.increment_clock(dt)

    def init_clock(self, start_point):
        self.time = 0
        self.time_step = 1 / 30  # This will be updated
        self.recent_positions = np.tile(start_point, 3).reshape((3, 3))
        if self.track_position_history:
            self.position_history = np.zeros((self.history_size, 3))
            self.acceleration_history = np.zeros((self.history_size, 3))
            self.history_index = -1

    def increment_clock(self, dt):
        if dt == 0:
            return self
        self.time += dt
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

    def add_force(self, force_func: Callable[[Vect3], Vect3]):
        espf = self.euler_steps_per_frame

        def update_from_force(particle, dt):
            if dt == 0:
                return
            for _ in range(espf):
                acc = force_func(particle.get_center()) / self.mass
                self.velocity += acc * dt / espf
                self.shift(self.velocity * dt / espf)

        self.add_updater(update_from_force)
        return self

    def add_spring_force(self, k=1.0, center=None):
        center = center if center is not None else self.get_center().copy()
        self.add_force(lambda p: k * (center - p))
        return self

    def add_field_force(self, field):
        charge = self.get_charge()
        self.add_force(lambda p: charge * field.get_forces([p])[0])
        return self

    def fix_x(self):
        x = self.get_x()
        self.add_updater(lambda m: m.set_x(x))

    # Getters

    def get_charge(self):
        return self.charge

    def get_radius(self):
        return self.sphere.get_radius()

    def get_internal_time(self):
        return self.time

    def scale(self, factor, *args, **kwargs):
        super().scale(factor, *args, **kwargs)
        self.sphere.set_radius(factor * self.sphere.get_radius())
        return self

    def get_acceleration(self):
        p0, p1, p2 = self.recent_positions
        # if (p0 == p1).all() or (p1 == p2).all():
        if np.isclose(p0, p1).all() or np.isclose(p1, p2).all():
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
        stroke_color=BLUE,
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

        self.sample_points = self.get_sample_points(
            center, width, height, depth,
            x_density, y_density, z_density
        )
        self.init_base_stroke_width_array(len(self.sample_points))

        super().__init__(
            stroke_color=stroke_color,
            flat_stroke=flat_stroke,
            **kwargs
        )

        n_samples = len(self.sample_points)
        self.set_points(np.zeros((8 * n_samples - 1, 3)))
        self.set_stroke(width=stroke_width)
        self.update_vectors()

    def get_sample_points(
        self,
        center: np.ndarray,
        width: float,
        height: float,
        depth: float,
        x_density: float,
        y_density: float,
        z_density: float
    ) -> np.ndarray:
        to_corner = np.array([width / 2, height / 2, depth / 2])
        spacings = 1.0 / np.array([x_density, y_density, z_density])
        to_corner = spacings * (to_corner / spacings).astype(int)
        lower_corner = center - to_corner
        upper_corner = center + to_corner + spacings
        return cartesian_product(*(
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


class TimeVaryingVectorField(VectorField):
    def __init__(
        self,
        # Takes in an array of points and a float for time
        time_func,
        **kwargs
    ):
        self.time = 0
        super().__init__(func=lambda p: time_func(p, self.time), **kwargs)
        self.add_updater(lambda m, dt: m.increment_time(dt))
        always(self.update_vectors)

    def increment_time(self, dt):
        self.time += dt


class ChargeBasedVectorField(VectorField):
    default_color = BLUE

    def __init__(self, *charges, **kwargs):
        self.charges = list(charges)
        super().__init__(
            self.get_forces,
            color=kwargs.pop("color", self.default_color),
            **kwargs
        )
        self.add_updater(lambda m: m.update_vectors())

    def get_forces(self, points):
        # To be implemented in subclasses
        return np.zeros_like(points)


class CoulombField(ChargeBasedVectorField):
    default_color = YELLOW

    def get_forces(self, points):
        return sum(
            coulomb_force(points, charge)
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


class ColoumbPlusLorentzField(LorentzField):
    def get_forces(self, points):
        return sum(
            lorentz_force(
                points, charge,
                radius=self.radius_of_suppression,
                c=self.c
            ) + sum(
                coulomb_force(points, charge)
                for charge in self.charges
            )
            for charge in self.charges
        )


class GraphAsVectorField(VectorField):
    def __init__(
        self,
        axes: Axes | ThreeDAxes,
        # Maps x to y, or x to (y, z)
        graph_func: Callable[[VectN], VectN] | Callable[[VectN], Tuple[VectN, VectN]],
        x_density=10.0,
        max_vect_len=np.inf,
        **kwargs,
    ):
        self.sample_xs = np.arange(axes.x_axis.x_min, axes.x_axis.x_max, 1.0 / x_density)
        self.axes = axes

        def vector_func(points):
            output = graph_func(self.sample_xs)
            if isinstance(axes, ThreeDAxes):
                graph_points = axes.c2p(self.sample_xs, *output)
            else:
                graph_points = axes.c2p(self.sample_xs, output)
            base_points = axes.x_axis.n2p(self.sample_xs)
            return graph_points - base_points

        super().__init__(
            func=vector_func,
            max_vect_len=max_vect_len,
            **kwargs
        )
        always(self.update_vectors)

    def reset_sample_points(self):
        self.sample_points = self.get_sample_points()

    def get_sample_points(self, *args, **kwargs):
        # Override super class and ignore all length/density information
        return self.axes.x_axis.n2p(self.sample_xs)


class OscillatingFieldWave(GraphAsVectorField):
    def __init__(self, axes, wave, **kwargs):
        self.wave = wave
        if "stroke_color" not in kwargs:
            kwargs["stroke_color"] = wave.get_color()
        super().__init__(
            axes=axes,
            graph_func=lambda x: wave.xt_to_yz(x, wave.time),
            **kwargs
        )

    def get_sample_points(self, *args, **kwargs):
        # Override super class and ignore all length/density information
        return self.wave.offset + self.axes.x_axis.n2p(self.sample_xs)


# Structure


class Molecule(Group):
    # List of characters
    atoms = []

    # List of 3d coordinates
    coordinates = np.zeros((0, 3))

    # List of pairs of indices
    bonds = []

    atom_to_color = {
        "H": RED,
        "O": BLUE,
        "C": GREEN,
    }
    atom_to_radius = {
        "H": 0.1,
        "O": 0.2,
        "C": 0.15,
    }
    ball_config = dict(shading=(0.25, 0.5, 0.5), glow_factor=0.25)
    stick_config = dict(stroke_width=1, stroke_color=GREY_A, flat_stroke=False)

    def __init__(self, height=2.0, **kwargs):
        coords = np.array(self.coordinates)
        radii = np.array([self.atom_to_radius[atom] for atom in self.atoms])
        rgbas = np.array([color_to_rgba(self.atom_to_color[atom]) for atom in self.atoms])

        balls = DotCloud(coords, **self.ball_config)
        balls.set_radii(radii)
        balls.set_rgba_array(rgbas)

        sticks = VGroup()
        for i, j in self.bonds:
            c1, c2 = coords[[i, j], :]
            r1, r2 = radii[[i, j]]
            unit_vect = normalize(c2 - c1)

            sticks.add(Line(
                c1 + r1 * unit_vect, c2 - r2 * unit_vect,
                **self.stick_config
            ))

        super().__init__(balls, sticks, **kwargs)

        self.apply_depth_test()
        self.balls = balls
        self.sticks = sticks
        self.set_height(height)


class Sucrose(Molecule):
    atoms = [
        "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O",
        "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C",
        "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H",
    ]
    coordinates = np.array([
        [-1.468 ,  0.4385, -0.9184],
        [-0.6033, -0.8919,  0.8122],
        [ 0.9285,  0.4834, -0.3053],
        [-3.0702, -2.0054,  1.1933],
        [-4.62  ,  0.6319,  0.7326],
        [ 1.2231,  0.2156,  2.5658],
        [ 3.6108, -1.7286,  0.6379],
        [ 3.15  ,  1.8347,  1.1537],
        [-1.9582, -1.848 , -2.43  ],
        [-1.3845,  3.245 , -0.8933],
        [ 3.8369,  0.2057, -2.5044],
        [-1.4947, -0.8632, -0.3037],
        [-2.9301, -1.0229,  0.1866],
        [-3.229 ,  0.3737,  0.6887],
        [-2.5505,  1.2243, -0.3791],
        [ 0.7534, -0.7453,  0.3971],
        [ 1.6462, -0.7853,  1.639 ],
        [ 3.1147, -0.5553,  1.2746],
        [ 3.2915,  0.6577,  0.3521],
        [ 2.2579,  0.7203, -0.7858],
        [-1.0903, -1.9271, -1.3122],
        [-2.0027,  2.5323,  0.1653],
        [ 2.5886, -0.1903, -1.9666],
        [-3.6217, -1.2732, -0.6273],
        [-2.8148,  0.5301,  1.6917],
        [-3.2289,  1.4361, -1.215 ],
        [ 1.0588, -1.5992, -0.2109],
        [ 1.5257, -1.753 ,  2.1409],
        [ 3.6908, -0.4029,  2.1956],
        [ 4.31  ,  0.675 , -0.0511],
        [ 2.2441,  1.7505, -1.1644],
        [-1.1311, -2.9324, -0.8803],
        [-0.0995, -1.7686, -1.74  ],
        [-1.2448,  2.3605,  0.9369],
        [-2.799 ,  3.1543,  0.5841],
        [ 1.821 , -0.1132, -2.7443],
        [ 2.6532, -1.2446, -1.6891],
        [-3.98  , -1.9485,  1.5318],
        [-4.7364,  1.5664,  0.9746],
        [ 0.2787,  0.0666,  2.7433],
        [ 4.549 , -1.5769,  0.4327],
        [ 3.3427,  2.6011,  0.5871],
        [-1.6962, -2.5508, -3.0488],
        [-0.679 ,  2.6806, -1.2535],
        [ 3.7489,  1.1234, -2.8135],
    ])
    bonds = [
        (0, 11),
        (0, 14),
        (1, 11),
        (1, 15),
        (2, 15),
        (2, 19),
        (3, 12),
        (3, 37),
        (4, 13),
        (4, 38),
        (5, 16),
        (5, 39),
        (6, 17),
        (6, 40),
        (7, 18),
        (7, 41),
        (8, 20),
        (8, 42),
        (9, 21),
        (9, 43),
        (10, 22),
        (10, 44),
        (11, 12),
        (11, 20),
        (12, 13),
        (12, 23),
        (13, 14),
        (13, 24),
        (14, 21),
        (14, 25),
        (15, 16),
        (15, 26),
        (16, 17),
        (16, 27),
        (17, 18),
        (17, 28),
        (18, 19),
        (18, 29),
        (19, 22),
        (19, 30),
        (20, 31),
        (20, 32),
        (21, 33),
        (21, 34),
        (22, 35),
        (22, 36),
    ]
