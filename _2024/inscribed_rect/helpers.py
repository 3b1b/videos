from __future__ import annotations

from manim_imports_ext import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from manimlib.typing import Vect3, Vect4


def get_loop(anchors, stroke_color=WHITE, stroke_width=3):
    result = VMobject()
    result.set_points_smoothly(anchors, approx=False)
    result.set_stroke(stroke_color, stroke_width)
    return result


def get_example_loop(index=1, stroke_color=WHITE, stroke_width=3, width=5):
    result = SVGMobject(f"example_loop{index}").family_members_with_points()[0]
    result.set_width(width)
    result.set_stroke(stroke_color, stroke_width)
    return result


def get_special_dot(
    color=YELLOW,
    radius=0.05,
    glow_radius_multiple=3,
    glow_factor=1.5
):
    return Group(
        TrueDot(radius=radius).make_3d(),
        GlowDot(radius=radius * glow_radius_multiple, glow_factor=glow_factor)
    ).set_color(color)


def smooth_index(lst: list, real_index: float):
    N = len(lst)
    scaled_index = real_index * (N - 1)
    int_index = int(scaled_index)
    residue = scaled_index % 1
    if int_index >= N - 1:
        return lst[-1]
    return interpolate(lst[int_index], lst[int_index + 1], residue)


def get_quick_loop_func(loop: VMobject, n_samples=501):
    samples = np.array(list(map(loop.pfp, np.linspace(0, 1, n_samples))))

    def func(x):
        return smooth_index(samples, x)

    return func


def get_surface_func(loop_func: Callable[[float], Vect3]):
    def func(u, v):
        point1 = loop_func(u)
        point2 = loop_func(v)
        midpoint = mid(point1, point2)
        dist = get_norm(point1 - point2)
        return (*midpoint[:2], dist)
    return func


def get_half_parametric_func(func):
    def half_func(u, v):
        return func(u * v, v)
    return half_func


def find_rectangle(
    loop_func: Callable[[float], Vect3],
    initial_condition: Vect4 = np.arange(0, 1, 0.25),
    target_angle: float = 60 * DEGREES,
    initial_param_range: float = 1.0,
    n_samples_per_range: int = 10,
    n_refinements: int = 4,
    return_cost = False
) -> Vect4:
    """
    Returns an numpy array of 4 elements, between 0 and 1, such that 
    entering them into loop_func approximately gives a rectangle.
    """
    params = initial_condition.copy()
    param_range = initial_param_range
    min_cost = np.inf

    for _ in range(n_refinements):
        param_groups = [
            np.linspace(x - param_range / 2, x + param_range / 2, n_samples_per_range) % 1
            for x in params
        ]
        sample_groups = [
            np.array([loop_func(x) for x in param_group])
            for param_group in param_groups
        ]

        min_cost = np.inf
        best_idx_group = None
        for idx_group in it.product(*4 * [range(n_samples_per_range)]):
            a, b, c, d = [sg[i] for sg, i in zip(sample_groups, idx_group)]
            ac_dist = get_dist(a, c)
            mid_dist_ratio = get_dist(midpoint(a, c), midpoint(b, d)) / ac_dist
            dist_dist_ratio = abs(ac_dist - get_dist(b, d)) / ac_dist
            angle = abs(angle_between_vectors(c - a, d - b))
            if angle > PI / 2:
                angle = PI - angle
            cost = mid_dist_ratio + dist_dist_ratio + abs(angle - target_angle) / TAU
            if cost < min_cost:
                best_idx_group = idx_group
                min_cost = cost
        params = [pg[i] for pg, i in zip(param_groups, best_idx_group)]

        param_range /= n_samples_per_range

    if return_cost:
        return params, min_cost
    else:
        return params


# Surface functions
def square_func(u, v):
    return (u, v, 0)


def tube_func(u, v):
    return (-math.sin(TAU * u), v, math.cos(TAU * u))


def torus_func(u, v, outer_radius=1.5, inner_radius=0.5):
    theta = TAU * v
    phi = TAU * u
    p = math.cos(theta) * RIGHT + math.sin(theta) * UP
    q = -math.sin(phi) * p + math.cos(phi) * OUT
    return outer_radius * p + inner_radius * q


def mobius_strip_func(u, v, outer_radius=1.5, inner_radius=0.5):
    theta = TAU * v
    phi = theta / 2
    p = math.cos(theta) * RIGHT + math.sin(theta) * UP
    q = math.cos(phi) * p + math.sin(phi) * OUT
    return outer_radius * p + inner_radius * q * (2 * u - 1)


def alt_mobius_strip_func(u, v):
    phi = TAU * v
    vect = rotate_vector(RIGHT, phi / 2, axis=UP)
    vect = rotate_vector(vect, phi, axis=OUT)
    ref_point = np.array([np.cos(phi), np.sin(phi), 0])
    return ref_point + 0.7 * (u - 0.5) * vect


def torus_uv_to_mobius_uv(u, v):
    v2 = (u + v) % 1
    u2 = abs(u - v)
    if u + v >= 1.0:
        u2 = 1.0 - u2
    return u2, v2


def stereo_project_point(point, axis=0, r=1, max_norm=10000):
    point = fdiv(point * r, point[axis] + r)
    point[axis] = 0
    norm = get_norm(point)
    if norm > max_norm:
        point *= max_norm / norm
    return point


def sudanese_band_func(u, v):
    eta = PI * u
    phi = TAU * v
    z1 = math.sin(eta) * np.exp(complex(0, phi))
    z2 = math.cos(eta) * np.exp(complex(0, phi / 2))
    r4_point = np.array([z1.real, z1.imag, z2.real, z2.imag])
    r4_point[:3] = rotate_vector(r4_point[:3], PI / 3, axis=[1, 1, 1])
    result = stereo_project_point(r4_point, axis=0)[1:]
    result = rotate_vector(result, 60 * DEG, OUT)
    result = rotate_vector(result, 90 * DEG, RIGHT)
    return result
