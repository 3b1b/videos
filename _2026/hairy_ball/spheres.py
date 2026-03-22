from manim_imports_ext import *
import numpy as np


if TYPE_CHECKING:
    from typing import Callable, Iterable, Sequence, TypeVar, Tuple, Optional
    from manimlib.typing import Vect2, Vect3, VectN, VectArray, Vect2Array, Vect3Array, Vect4Array


def fibonacci_sphere(samples=1000):
    """
    Create uniform-ish points on a sphere

    Parameters
    ----------
    samples : int
        Number of points to create. The default is 1000.

    Returns
    -------
    points : NumPy array
        Points on the unit sphere.

    """

    # Define the golden angle
    phi = np.pi * (np.sqrt(5) - 1)

    # Define y-values of points
    pos = np.array(range(samples), ndmin=2)
    y = 1 - (pos / (samples - 1)) * 2

    # Define radius of cross-section at y
    radius = np.sqrt(1 - y * y)

    # Define the golden angle increment
    theta = phi * pos

    # Define x- and z- values of poitns
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    # Merge together x,y,z
    points = np.concatenate((x, y, z))

    # Transpose to get coordinates in right place
    points = np.transpose(points)

    return points


def stereographic_proj(points3d, epsilon=1e-10):
    x, y, z = points3d.T

    denom = 1 - z
    denom[np.abs(denom) < epsilon] = np.inf
    return np.array([x / denom, y / denom, 0 * z]).T


def inv_streographic_proj(points2d):
    u, v = points2d.T
    norm_squared = u * u + v * v
    denom = 1 + norm_squared
    return np.array([
        2 * u / denom,
        2 * v / denom,
        (norm_squared - 1) / denom,
    ]).T


def right_func(points):
    return np.repeat([[1, 0]], len(points), axis=0)


def stereographic_vector_field(points3d, vector_field_2d):
    points2d = stereographic_proj(points3d)[:, :2]
    vects2d = vector_field_2d(points2d)
    u, v = points2d.T
    vect_u, vect_v = vects2d.T

    # Compute Jacobian
    r_squared = u**2 + v**2
    denom = 1 + r_squared
    denom_squared = denom**2

    # For x = 2u / (1 + u² + v²):
    dx_du = 2 * (1 + v**2 - u**2) / denom_squared
    dx_dv = -4 * u * v / denom_squared

    # For y = 2v / (1 + u² + v²):
    dy_du = -4 * u * v / denom_squared
    dy_dv = 2 * (1 + u**2 - v**2) / denom_squared

    # For z = (u² + v² - 1) / (1 + u² + v²):
    dz_du = 4 * u / denom_squared
    dz_dv = 4 * v / denom_squared

    # Apply the Jacobian: [v_x, v_y, v_z]^T = J × [v_u, v_v]^T
    vect_x = dx_du * vect_u + dx_dv * vect_v
    vect_y = dy_du * vect_u + dy_dv * vect_v
    vect_z = dz_du * vect_u + dz_dv * vect_v

    return np.array([vect_x, vect_y, vect_z]).T


def rotation_field(points3d, axis=IN):
    return np.cross(points3d, axis)


def flatten_field(points3d, vector_field_3d):
    vects = vector_field_3d(points3d)
    norms = normalize_along_axis(points3d, 1)
    return np.cross(vects, norms)


def get_sphereical_vector_field(
    v_func,
    axes,
    points,
    color=BLUE,
    stroke_width=1,
    mvltss=1.0,
    tip_width_ratio=4,
    tip_len_to_width=0.01,
):

    field = VectorField(
        v_func, axes,
        sample_coords=1.01 * points,
        max_vect_len_to_step_size=mvltss,
        density=1,
        stroke_width=stroke_width,
        tip_width_ratio=tip_width_ratio,
        tip_len_to_width=tip_len_to_width,
    )
    field.apply_depth_test()
    field.set_stroke(color, opacity=0.8)
    field.set_scale_stroke_with_zoom(True)
    return field


class SphereStreamLines(StreamLines):
    def __init__(self, func, coordinate_system, density=50, sample_coords=None, **kwargs):
        self.sample_coords = sample_coords
        super().__init__(func, coordinate_system, density=density, **kwargs)

    def get_sample_coords(self):
        if self.sample_coords is None:
            coords = fibonacci_sphere(int(4 * PI * self.density))
        else:
            coords = self.sample_coords
        return coords

    def draw_lines(self):
        super().draw_lines()
        for submob in self.submobjects:
            submob.set_points(normalize_along_axis(submob.get_points(), 1))
        return self


# Scenes


class TeddyHeadSwirl(InteractiveScene):
    def construct(self):
        img = ImageMobject("TeddyHead")
        img.set_height(FRAME_HEIGHT)
        img.fix_in_frame()
        # self.add(img)

        # Add sphere
        frame = self.frame
        axes = ThreeDAxes()

        def v_func(points):
            perp = np.cross(points, OUT)
            alt_perp = np.cross(points, perp)
            return 0.4 * (perp + alt_perp)

        top_sample_points = np.array([
            point for point in fibonacci_sphere(10_000) if point[2] > 0.95
        ])
        full_sample_points = fibonacci_sphere(1000)

        top_lines, full_lines = lines = VGroup(*(
            SphereStreamLines(
                v_func,
                axes,
                sample_coords=samples,
                arc_len=0.25,
                dt=0.05,
                max_time_steps=10,
            )
            for samples in [top_sample_points, full_sample_points]
        ))
        lines.set_stroke(WHITE, 2, 0.7)
        full_lines.set_stroke(WHITE, 2, 0.7)
        top_animated_lines = AnimatedStreamLines(top_lines)
        full_animated_lines = AnimatedStreamLines(full_lines)

        frame.reorient(-50, 12, 0, (0.15, -0.04, 0.0), 3.09)
        self.add(top_animated_lines)
        self.wait(4)
        self.add(full_animated_lines)
        self.play(
            frame.animate.reorient(-39, 75, 0),
            run_time=6
        )


class IntroduceVectorField(InteractiveScene):
    def construct(self):
        # Set up sphere
        frame = self.frame
        self.camera.light_source.move_to([0, -10, 10])
        radius = 3
        axes = ThreeDAxes((-2, 2), (-2, 2), (-2, 2))
        axes.scale(radius)
        sphere = Sphere(radius=radius)
        sphere.set_color(BLUE_B, 0.3)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere, (51, 25))
        mesh.set_stroke(WHITE, 1, 0.15)

        frame.reorient(0, 90, 0)
        self.play(
            frame.animate.reorient(30, 65, 0),
            ShowCreation(sphere),
            Write(mesh, time_span=(1, 3), lag_ratio=1e-2),
            run_time=3
        )

        # Tangent plane
        v_tracker = Point()
        v_tracker.move_to(radius * OUT)

        def place_on_vect(mobject):
            matrix = z_to_vector(v_tracker.get_center())
            mobject.set_points(np.dot(mobject.points_at_zenith, matrix.T))

        v_dot = Dot(radius=0.05)
        v_dot.set_fill(YELLOW)

        plane = Square(side_length=2 * radius)
        plane.set_stroke(WHITE, 1)
        plane.set_fill(GREY, 0.5)

        for mob in [plane, v_dot]:
            mob.move_to(radius * OUT)
            mob.points_at_zenith = mob.get_points().copy()
            mob.add_updater(place_on_vect)

        self.play(
            VFadeIn(v_dot),
            v_tracker.animate.move_to(radius * normalize(RIGHT + OUT)),
            run_time=2
        )
        self.wait()
        plane.update()
        plane.suspend_updating()
        self.play(
            FadeIn(plane),
            frame.animate.reorient(13, 78, 0),
            run_time=2
        )
        self.play(frame.animate.reorient(53, 60, 0, (-0.29, 0.09, 0.31), 8.97), run_time=2)
        self.wait()

        # Show one vector
        def v_func(points3d):
            v1 = stereographic_vector_field(points3d, right_func)
            v2 = normalize_along_axis(rotation_field(points3d, RIGHT), 1)
            v3 = normalize_along_axis(rotation_field(points3d, RIGHT + IN), 1)
            return (3 * v1 + v2 + v3) / 6

        def vector_field_3d(points3d):
            x, y, z = points3d.T
            return np.array([
                np.cos(3 * x) * np.sin(y),
                np.cos(5 * z) * np.sin(x),
                -z**2 + x,
            ]).T

        def alt_v_func(points3d):
            return normalize_along_axis(flatten_field(points3d, vector_field_3d), 1)

        def get_tangent_vect():
            origin = v_tracker.get_center()
            vector = Arrow(origin, origin + radius * v_func(normalize(origin).reshape(1, -1)).flatten(), buff=0, thickness=3)
            vector.set_fill(BLUE)
            vector.set_perpendicular_to_camera(frame)
            return vector

        tangent_vect = get_tangent_vect()
        self.add(plane, tangent_vect, v_dot)
        self.play(GrowArrow(tangent_vect))
        self.play(Rotate(tangent_vect, TAU, axis=v_tracker.get_center(), about_point=tangent_vect.get_start(), run_time=2))
        self.wait()

        # Show more vectors
        plane.resume_updating()
        tangent_vect.add_updater(lambda m: m.match_points(get_tangent_vect()))
        og_vect = rotate_vector(v_tracker.get_center(), 10 * DEG, axis=DOWN)
        v_tracker.clear_updaters()
        v_tracker.add_updater(lambda m, dt: m.rotate(60 * DEG * dt, axis=og_vect, about_point=ORIGIN))
        v_tracker.add_updater(lambda m, dt: m.rotate(1 * DEG * dt, axis=RIGHT, about_point=ORIGIN))

        frame.clear_updaters()
        frame.add_ambient_rotation(-2.5 * DEG)

        self.wait(2)

        field = self.get_vector_field(axes, v_func, 4000, start_point=v_tracker.get_center())
        self.add(field, plane, v_tracker, tangent_vect, v_dot)
        self.play(
            ShowCreation(field),
            run_time=5
        )
        self.wait(4)
        self.play(
            FadeOut(plane),
            FadeOut(tangent_vect),
            FadeOut(v_dot),
            frame.animate.reorient(0, 59, 0, (-0.03, 0.15, -0.08), 6.80),
            run_time=4
        )
        self.wait(5)

        # Show denser field
        dots, dense_dots = [
            DotCloud(fibonacci_sphere(num), radius=0.01)
            for num in [4000, 400_000]
        ]
        for mob in [dots, dense_dots]:
            mob.make_3d()
            mob.set_color(WHITE)
            mob.scale(radius * 1.01)

        dense_field = self.get_vector_field(axes, v_func, 50_000, mvltss=5.0)
        dense_field.set_stroke(opacity=0.35)

        dots.set_radius(0.02)
        dense_dots.set_radius(0.01)

        self.play(ShowCreation(dots, run_time=3))
        self.wait()
        self.play(
            FadeOut(dots, time_span=(1.5, 2.5)),
            FadeOut(field, time_span=(2.5, 3)),
            ShowCreation(dense_field),
            run_time=3
        )
        frame.clear_updaters()
        self.play(frame.animate.reorient(-66, 52, 0, (-0.51, 0.27, 0.2), 2.81), run_time=4)
        self.wait()

        # Show plane and tangent vector again
        field.save_state()
        field.set_stroke(width=1e-6)
        self.play(
            dense_field.animate.set_stroke(width=1e-6),
            Restore(field, time_span=(1, 3)),
            frame.animate.reorient(48, 54, 0, (-2.74, -2.49, -0.1), 11.93),
            VFadeIn(plane),
            VFadeIn(tangent_vect),
            run_time=5
        )
        self.wait(5)

        v_tracker.clear_updaters()
        null_point = 3 * normalize(np.array([-1, -0.25, 0.8]))
        self.play(
            v_tracker.animate.move_to(null_point),
            frame.animate.reorient(-43, 55, 0, (-2.42, 2.76, -0.55), 9.22),
            run_time=3
        )
        self.remove(tangent_vect)
        v_dot.update()
        self.play(
            FadeIn(v_dot, scale=0.25),
            FadeOut(plane, scale=0.25),
        )
        self.wait()

        # Define streamlines
        static_stream_lines = SphereStreamLines(
            lambda p: v_func(np.array(p).reshape(-1, 3)).flatten(),
            axes,
            density=400,
            stroke_width=2,
            magnitude_range=(0, 10),
            solution_time=1,
        )
        static_stream_lines.scale(radius)
        static_stream_lines.set_stroke(BLUE_B, 3, 0.8)
        stream_lines = AnimatedStreamLines(static_stream_lines, lag_range=10, rate_multiple=0.5)
        stream_lines.apply_depth_test()

        # Show the earth
        earth = TexturedSurface(sphere, "EarthTextureMap", "NightEarthTextureMap")
        earth.rotate(-90 * DEG, IN)
        earth.scale(1.001)
        frame.add_ambient_rotation(2 * DEG)

        self.add(sphere, earth, stream_lines, mesh, field)
        self.play(
            FadeOut(v_dot),
            FadeIn(earth, time_span=(0, 3)),
            field.animate.set_stroke(opacity=0.25).set_anim_args(time_span=(0, 2)),
            frame.animate.reorient(-90, 74, 0, (-1.37, 0.04, 0.37), 5.68),
        )
        self.wait(10)

        # Zoom in to null point
        frame.clear_updaters()
        dense_field = self.get_vector_field(axes, v_func, 10_000)
        dense_field.set_stroke(opacity=0.35)
        self.play(
            FadeOut(field, run_time=2),
            FadeIn(dense_field, run_time=2),
            frame.animate.reorient(-69, 54, 0, (-0.01, 0.19, -0.04), 4.81),
            run_time=5
        )
        self.wait(3)
        self.play(
            frame.animate.reorient(129, 70, 0, (-0.18, 0.18, -0.1), 8.22),
            run_time=20
        )
        self.wait(10)

    def get_vector_field(self, axes, v_func, n_points, start_point=None, mvltss=1.0, random_order=False):
        points = fibonacci_sphere(n_points)

        if start_point is not None:
            points = points[np.argsort(np.linalg.norm(points - start_point.reshape(-1, 3), axis=1))]
        if random_order:
            indices = list(range(len(points)))
            random.shuffle(indices)
            points = points[indices]

        alpha = clip(inverse_interpolate(10_000, 1000, n_points), 0, 1)
        stroke_width = interpolate(1, 3, alpha**2)

        return get_sphereical_vector_field(v_func, axes, points, mvltss=mvltss, stroke_width=stroke_width)

    def old(self):
        # Vary the density
        density_tracker = ValueTracker(2000)
        field.add_updater(
            lambda m: m.become(self.get_vector_field(int(density_tracker.get_value()), axes, v_func))
        )
        self.add(field)
        field.resume_updating()
        frame.suspend_updating()
        self.play(
            density_tracker.animate.set_value(50_000),
            frame.animate.reorient(-30, 50, 0, (-0.07, 0.02, 0.36), 3.01),
            run_time=5,
        )
        field.suspend_updating()
        self.wait(3)


class StereographicProjection(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        x_max = 20
        axes = ThreeDAxes((-x_max, x_max), (-x_max, x_max), (-2, 2))
        plane = NumberPlane((-x_max, x_max), (-x_max, x_max))
        plane.background_lines.set_stroke(BLUE, 1, 0.5)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)
        axes.apply_depth_test()
        plane.apply_depth_test()

        sphere = Sphere(radius=1)
        sphere.set_opacity(0.5)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere)
        mesh.set_stroke(WHITE, 1, 0.25)

        self.add(sphere, mesh, axes, plane)
        frame.reorient(-15, 64, 0, (0.0, 0.1, -0.09), 4.0)

        # Show the 2d cross section
        frame.clear_updaters()
        sphere.set_clip_plane(UP, 1)
        n_dots = 20
        sample_points = np.array([
            math.cos(theta) * OUT + math.sin(theta) * RIGHT
            for theta in np.linspace(0, TAU, n_dots + 2)[1:-1]
        ])
        sphere_dots, plane_dots, proj_lines = self.get_dots_and_lines(sample_points)

        self.play(
            sphere.animate.set_clip_plane(UP, 0),
            frame.animate.reorient(-43, 74, 0, (0.0, 0.0, -0.0), 3.50),
            FadeIn(sphere_dots, time_span=(1, 2)),
            ShowCreation(proj_lines, lag_ratio=0, time_span=(1, 2)),
            run_time=2
        )
        frame.add_ambient_rotation(2 * DEG)

        sphere_dot_ghosts = sphere_dots.copy().set_opacity(0.5)
        self.remove(sphere_dots)
        self.add(sphere_dot_ghosts)
        self.play(
            TransformFromCopy(sphere_dots, plane_dots, lag_ratio=0.5, run_time=10),
        )
        self.wait(3)

        planar_group = Group(sphere_dot_ghosts, plane_dots, proj_lines)

        # Show more points on the sphere
        sample_points = fibonacci_sphere(200)
        sphere_dots, plane_dots, proj_lines = self.get_dots_and_lines(sample_points)

        self.play(
            sphere.animate.set_clip_plane(UP, -1),
            frame.animate.reorient(-65, 73, 0, (-0.09, -0.01, -0.15), 5.08),
            ShowCreation(proj_lines, lag_ratio=0),
            FadeOut(planar_group),
            run_time=2,
        )
        self.wait(4)
        self.play(FadeIn(sphere_dots))
        self.wait(2)

        sphere_dot_ghosts = sphere_dots.copy().set_opacity(0.5)
        self.remove(sphere_dots)
        self.add(sphere_dot_ghosts)

        self.play(
            TransformFromCopy(sphere_dots, plane_dots, run_time=3),
        )
        self.wait(3)

        # Inverse projection
        plane.insert_n_curves(100)
        plane.save_state()
        proj_plane = plane.copy()
        proj_plane.apply_points_function(lambda p: inv_streographic_proj(p[:, :2]))
        proj_plane.make_smooth()
        proj_plane.background_lines.set_stroke(BLUE, 2, 1)
        proj_plane.faded_lines.set_stroke(BLUE, 1, 0.5)

        self.play(
            Transform(plane_dots, sphere_dot_ghosts),
            FadeOut(sphere_dot_ghosts, scale=0.9),
            Transform(plane, proj_plane),
            proj_lines.animate.set_stroke(opacity=0.2),
            run_time=4,
        )
        self.play(
            frame.animate.reorient(-20, 38, 0, (-0.04, -0.03, 0.13), 3.54),
            run_time=5
        )
        self.wait(5)
        self.play(
            frame.animate.reorient(-27, 73, 0, (-0.03, 0.03, 0.04), 5.27),
            Restore(plane),
            FadeOut(plane_dots),
            run_time=5
        )
        self.wait(2)

        # Show a vector field
        xy_field = VectorField(lambda ps: np.array([RIGHT for p in ps]), plane)
        xy_field.set_stroke(BLUE)
        xy_field.save_state()
        xy_field.set_stroke(width=1e-6)

        self.play(Restore(xy_field))
        self.wait(5)

        # Project the vector field up
        proj_field = xy_field.copy()
        proj_field.apply_points_function(lambda p: inv_streographic_proj(p[:, :2]), about_point=ORIGIN)
        proj_field.replace(sphere)
        proj_plane.background_lines.set_stroke(BLUE, 1, 0.5)
        proj_plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)
        proj_plane.axes.set_stroke(WHITE, 0)

        self.play(
            Transform(plane, proj_plane),
            Transform(xy_field, proj_field),
            run_time=5,
        )
        self.play(
            frame.animate.reorient(-35, 31, 0, (0.05, 0.22, 0.22), 1.59),
            FadeOut(proj_lines),
            run_time=10
        )
        self.wait(8)

        # Show the flow (Maybe edit as a simple split-screen)
        proto_stream_lines = VGroup(
            Line([x, y, 0], [x + 20, y, 0]).insert_n_curves(25)
            for x in range(-100, 100, 10)
            for y in np.arange(-100, 100, 0.25)
        )
        for line in proto_stream_lines:
            line.virtual_time = 1
        proto_stream_lines.set_stroke(WHITE, 2, 0.8)
        proto_stream_lines.apply_points_function(lambda p: inv_streographic_proj(p[:, :2]), about_point=ORIGIN)
        proto_stream_lines.scale(1.01)
        proto_stream_lines.make_smooth()
        animated_lines = AnimatedStreamLines(proto_stream_lines, rate_multiple=0.2)

        sphere.set_color(GREY_E, 1)
        sphere.set_clip_plane(UP, 1)
        sphere.set_height(1.98).center()
        xy_field.apply_depth_test()
        animated_lines.apply_depth_test()
        self.add(sphere, mesh, plane, animated_lines, xy_field)
        self.play(
            FadeIn(sphere),
            FadeOut(xy_field),
            plane.animate.fade(0.25),
            xy_field.animate.set_stroke(opacity=0.5),
            frame.animate.reorient(-30, 29, 0, ORIGIN, 3.0),
            run_time=3
        )
        self.wait(30)
        return

    def get_dots_and_lines(self, sample_points, color=YELLOW, radius=0.025, stroke_opacity=0.35):
        sphere_dots = Group(TrueDot(point) for point in sample_points)
        for dot in sphere_dots:
            dot.make_3d()
            dot.set_color(color)
            dot.set_radius(radius)

        plane_dots = sphere_dots.copy().apply_points_function(stereographic_proj)
        proj_lines = VGroup(
            VGroup(
                Line(OUT, dot.get_center())
                for dot in dots
            )
            for dots in [plane_dots, sphere_dots]
        )
        proj_lines.set_stroke(color, 1, stroke_opacity)

        return sphere_dots, plane_dots, proj_lines

    def flow_with_projection_insertion(self):
        # For an insertion
        frame.clear_updaters()
        frame.reorient(-18, 77, 0, (-0.04, 0.04, 0.09), 5.43)
        frame.clear_updaters()
        frame.add_ambient_rotation(1 * DEG)
        sphere.set_clip_plane(UP, 2)
        sphere.set_color(GREY_D, 0.5)
        xy_field.apply_depth_test()
        xy_field.set_stroke(opacity=0.)
        proj_lines.set_stroke(opacity=0)
        proj_plane.axes.set_stroke(width=1, opacity=0.5)
        self.clear()
        sphere.scale(0.99)
        self.add(axes, xy_field, proj_lines, sphere, plane, frame)

        # Particles
        n_samples = 50_000
        x_max = axes.x_range[1]
        sample_points = np.random.uniform(-x_max, x_max, (n_samples, 3))
        sample_points[:, 2] = 0
        particles = DotCloud(sample_points)
        particles.set_radius(0.015)
        particles.set_color(BLUE)
        particles.make_3d()

        particle_opacity_tracker = ValueTracker(1)
        proj_particle_radius_tracker = ValueTracker(0.01)

        proj_particles = particles.copy()
        proj_particles.set_opacity(1)

        x_vel = 0.5

        def update_particles(particles, dt):
            particles.shift(dt * x_vel * RIGHT)
            points = particles.get_points()
            points[points[:, 0] > x_max, 0] -= 2 * x_max
            particles.set_points(points)
            particles.set_opacity(particle_opacity_tracker.get_value())

            sphere_points = inv_streographic_proj(points[:, :2])
            zs = sphere_points[:, 2]
            proj_particles.set_points(sphere_points)
            proj_particles.set_radius((proj_particle_radius_tracker.get_value() * (1.5 - zs)).reshape(-1, 1))

        particles.add_updater(update_particles)
        self.add(particles, sphere)
        self.wait(7)

        # Project
        moving_particles = particles.copy()
        moving_particles_opacity_tracker = ValueTracker(0)
        moving_particles.add_updater(lambda m: m.set_opacity(moving_particles_opacity_tracker.get_value()))

        field = get_sphereical_vector_field(
            lambda p3d: stereographic_vector_field(p3d, right_func),
            axes,
            fibonacci_sphere(1000),
            stroke_width=0.5
        )
        field.set_stroke(WHITE, opacity=0.5)

        self.play(proj_lines.animate.set_opacity(0.4))
        self.play(
            particle_opacity_tracker.animate.set_value(0.15),
            Transform(moving_particles, proj_particles),
            moving_particles_opacity_tracker.animate.set_value(1),
            run_time=5
        )
        self.remove(moving_particles)
        self.add(sphere, particles, proj_particles)
        self.play(
            proj_lines.animate.set_opacity(0.1),
            frame.animate.reorient(-30, 29, 0, ORIGIN, 3.0),
            proj_particle_radius_tracker.animate.set_value(0.0075),
            FadeIn(field),
            run_time=3
        )
        self.wait(15)


    def old(self):
        earth = TexturedSurface(sphere, "EarthTextureMap")
        earth.set_opacity(1)

        earth_group = Group(earth)
        earth_group.save_state()
        proj_earth = earth_group.copy()
        proj_earth.apply_points_function(stereographic_proj)
        proj_earth.interpolate(proj_earth, earth_group, 0.01)

        self.remove(earth_group)
        self.play(TransformFromCopy(earth_group, proj_earth), run_time=3)
        self.wait()
        self.remove(proj_earth)
        self.play(TransformFromCopy(proj_earth, earth_group), run_time=3)


class SimpleRightwardFlow(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        x_max = 20
        axes = ThreeDAxes((-x_max, x_max), (-x_max, x_max), (-2, 2))
        plane = NumberPlane((-x_max, x_max), (-x_max, x_max))
        plane.background_lines.set_stroke(BLUE, 1, 0.5)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)
        axes.apply_depth_test()
        plane.apply_depth_test()

        # Simple flow
        frame.set_height(4)

        xy_field = VectorField(lambda ps: np.array([RIGHT for p in ps]), plane)
        xy_field.set_stroke(BLUE)
        self.add(xy_field)

        proto_stream_lines = VGroup(
            Line([x, y, 0], [x + 1, y, 0]).insert_n_curves(20)
            for x in np.arange(-10, 10, 0.5)
            for y in np.arange(-10, 10, 0.1)
        )
        for line in proto_stream_lines:
            line.virtual_time = 1
        proto_stream_lines.set_stroke(WHITE, 2, 0.8)

        animated_plane_lines = AnimatedStreamLines(proto_stream_lines, rate_multiple=0.2)

        self.add(animated_plane_lines)
        self.wait(30)


class SingleNullPointHairyBall(InteractiveScene):
    hide_top = True

    def construct(self):
        # Set up
        frame = self.frame
        radius = 3
        sphere = Sphere(radius=radius)
        sphere.set_color(GREY_E, 1)
        sphere.set_shading(0.1, 0.1, 0.3)
        sphere.always_sort_to_camera(self.camera)
        axes = ThreeDAxes((-2, 2), (-2, 2), (-2, 2))
        axes.scale(radius)

        self.camera.light_source.move_to(3 * LEFT + 12 * UP + 3 * OUT)

        frame.reorient(-3, 161, 0)
        self.add(sphere)

        # Add vector field
        def v_func(points3d):
            new_points = stereographic_vector_field(points3d, right_func)
            norms = np.linalg.norm(new_points, axis=1)
            new_points *= 0.2 / norms[:, np.newaxis]
            return new_points

        def out_func(points3d):
            return 0.2 * points3d

        n_points = 50_000
        v_range = (-0.95, 1) if self.hide_top else (-1, 1)
        points = np.array([
            normalize(sphere.uv_func(TAU * random.random(), math.acos(pre_v)))
            for pre_v in np.random.uniform(*v_range, n_points)
        ])
        pre_field, field = fields = [
            get_sphereical_vector_field(
                func, axes, points,
                stroke_width=3,
                mvltss=3,
            )
            for func in [out_func, v_func]
        ]
        for lines in fields:
            lines.set_stroke(BLUE_E, opacity=0.75)
            lines.data['stroke_width'] = 0.5
            lines.note_changed_data()

        q_marks = Tex(R"???", font_size=72)
        q_marks.rotate(-90 * DEG)
        q_marks.move_to(sphere.get_zenith())
        disk = Circle(radius=1)
        disk = Sphere(radius=radius, v_range=(0.9 * PI, PI))
        disk.set_color(BLACK)
        disk.deactivate_depth_test()
        top_q = Group(disk, q_marks)
        top_q.set_z_index(1)

        if not self.hide_top:
            top_q.set_opacity(0)

        self.add(pre_field, sphere)
        self.play(
            ReplacementTransform(pre_field, field, run_time=2),
            frame.animate.reorient(-92, 11, 0).set_anim_args(run_time=7),
            FadeIn(top_q, time_span=(3.5, 5)),
        )
        self.play(
            frame.animate.reorient(-168, 127, 0),
            FadeOut(top_q, time_span=(4.2, 5)),
            run_time=10
        )


class SingleNullPointHairyBallRevealed(SingleNullPointHairyBall):
    hide_top = False


class PairsOfNullPoints(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        radius = 3
        sphere_scale = 0.99
        axis_range = (-4, 4)

        sphere = Sphere(radius=sphere_scale * radius)
        sphere.set_color(GREY_D, 0.5)
        sphere.always_sort_to_camera(self.camera)
        axes = ThreeDAxes(axis_range, axis_range, axis_range)
        axes.scale(radius)

        frame.reorient(-155, 74, 0)
        self.add(axes, sphere)

        # Vector field
        def source_and_sink(points2d, offset=-3):
            x, y = points2d.T
            return np.array([x - offset, y]).T

        def twirl_func(points2d, offset=-3):
            x, y = points2d.T
            return np.array([-y, x - offset]).T

        rotation = np.identity(3)[:, [0, 2, 1]]

        field = self.get_spherical_field(axes, twirl_func, rotation=rotation)
        offset_tracker = ValueTracker(-1)

        self.add(sphere, field)
        frame.reorient(-136, 77, 0, (0.91, 0.31, 0.79), 12.58)
        self.play(
            frame.animate.reorient(-177, 80, 0, (0.0, 0.0, 0.0), 8),
            run_time=4
        )
        frame.add_ambient_rotation(1 * DEG)

        # Change the field around
        new_params = [
            (rotation_matrix_transpose(90 * DEG, UP), -5),
            (rotation_matrix_transpose(30 * DEG, DOWN), -2),
            (rotation_matrix_transpose(120 * DEG, LEFT), 3),
            (rotation, -1),
        ]
        for new_rot, offset in new_params:
            new_field = self.get_spherical_field(
                axes, lambda ps: twirl_func(ps, offset), rotation=new_rot
            )
            self.play(Transform(field, new_field, run_time=2))

        self.play(
            offset_tracker.animate.set_value(-3),
            UpdateFromFunc(
                field,
                lambda m: m.become(
                    self.get_spherical_field(axes, lambda ps: twirl_func(ps, offset_tracker.get_value()), rotation=rotation)
                )
            ),
            run_time=4
        )

        # Show some flow
        streamlines = self.get_streamlines(field, axes, radius)
        self.add(streamlines)
        self.wait(8)

        # New field
        field2 = self.get_spherical_field(axes, source_and_sink)
        streamlines2 = self.get_streamlines(field2, axes, radius, density=100)

        self.play(FadeOut(streamlines))
        self.play(Transform(field, field2))
        self.play(FadeIn(streamlines2))
        self.wait(8)

    def get_streamlines(self, field, axes, radius, density=100):
        streamlines = SphereStreamLines(
            lambda p: field.func(np.array(p).reshape(-1, 3)).flatten(), axes,
            density=density,
            solution_time=1.0,
            dt=0.05,
        )
        streamlines.scale(radius, about_point=ORIGIN)
        streamlines.set_stroke(WHITE, 1, 0.5)
        animated_lines = AnimatedStreamLines(streamlines, rate_multiple=0.2)
        animated_lines.apply_depth_test()
        return animated_lines

    def get_spherical_field(
        self,
        axes,
        plane_func,
        n_sample_points=2000,
        stroke_width=2,
        rotation=np.identity(3),
    ):
        def v_func(points3d):
            rot_points = np.dot(points3d, rotation)
            new_points = stereographic_vector_field(rot_points, plane_func)
            norms = np.linalg.norm(new_points, axis=1)
            new_points *= 1.0 / norms[:, np.newaxis]
            new_points = np.dot(new_points, rotation.T)
            return new_points

        sample_points = fibonacci_sphere(n_sample_points)
        field = get_sphereical_vector_field(
            v_func, axes, sample_points,
            stroke_width=stroke_width
        )
        field.set_flat_stroke(False)

        return field


class AskAboutOutside(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        radius = 2
        axes = ThreeDAxes().scale(radius)
        axes.set_stroke(WHITE, 1, 0.5)
        axes.set_flat_stroke(False)
        axes.set_z_index(1)

        def sphere_uv(u, v):
            return [math.cos(u) * math.sin(v), math.sin(u) * math.sin(v), -math.cos(v)]

        sphere_group = self.get_warped_sphere_group(sphere_uv)
        sphere, mesh = sphere_group

        frame.reorient(-6, 70, 0)
        self.add(sphere, mesh, axes)

        # Show a character on the surface
        morty = Mortimer(mode="confused", height=0.1).flip()
        mirror_morty = morty.copy().flip(axis=RIGHT, about_edge=DOWN)
        mirror_morty.fade(0.5)
        out_arrow = Vector(0.15 * UP, thickness=0.5)
        out_arrow.next_to(morty, LEFT, buff=0.025, aligned_edge=DOWN)
        in_arrow = out_arrow.copy().flip(axis=RIGHT, about_edge=DOWN)
        out_arrow.set_color(BLUE)
        in_arrow.set_color(RED)
        morty_group = VGroup(morty, mirror_morty, out_arrow, in_arrow)

        morty_group.rotate(90 * DEG, RIGHT)
        morty_group.move_to(sphere.get_zenith())
        morty_group.rotate(30 * DEG, axis=UP, about_point=ORIGIN)

        sphere.set_clip_plane(UP, 2)

        self.play(
            FadeIn(morty, time_span=(0, 1)),
            sphere.animate.set_opacity(0.25),
            frame.animate.reorient(-5, 83, 0, (1.09, 0.51, 1.62), 1.15),
            run_time=3
        )
        self.play(GrowArrow(out_arrow))
        self.wait()
        self.play(
            TransformFromCopy(morty, mirror_morty),
            GrowArrow(in_arrow)
        )
        self.wait()

        # Show obvious outside and inside
        out_arrows, in_arrows = all_arrows = VGroup(
            VGroup(
                Arrow(radius * a1 * point, radius * a2 * point, fill_color=color, thickness=4, buff=0)
                for point in compass_directions(24)
            )
            for a1, a2, color in [(1.1, 1.5, BLUE), (0.9, 0.7, RED)]
        )
        all_arrows.rotate(90 * DEG, RIGHT, about_point=ORIGIN)

        self.play(
            frame.animate.reorient(-4, 82, 0, (0.15, -0.01, 0.02), 8.08),
            LaggedStartMap(GrowArrow, out_arrows, lag_ratio=0.01, time_span=(3, 6)),
            run_time=6
        )
        self.wait()
        self.play(
            FadeOut(out_arrows),
            FadeOut(morty_group),
            sphere.animate.set_opacity(1)
        )

        # Warp the sphere
        def twist_sphere_uv(u, v):
            return [math.cos(u + v - 0.5) * math.sin(v), math.sin(u + v - 0.5) * math.sin(v), -math.cos(v)]

        def squish_sphere_uv(u, v):
            x, y, z = twist_sphere_uv(u, v)
            dist = math.sqrt(x**2 + y**2)
            z *= -35 * (dist - 0.25) * (dist - 0.75) * (dist - 0) * (dist - 1.1)
            return (x, y, z)

        twisted_sphere = self.get_warped_sphere_group(twist_sphere_uv)
        squish_sphere = self.get_warped_sphere_group(squish_sphere_uv)

        self.play(
            Transform(sphere_group, twisted_sphere),
            frame.animate.reorient(-1, 73, 0, (-0.23, 0.06, -0.09), 6.00),
            run_time=5
        )
        self.play(
            frame.animate.reorient(12, 73, 0, (-0.09, 0.1, -0.06), 4.71),
            Transform(sphere_group, squish_sphere),
            run_time=5
        )

        # Example point
        index = 9200
        point = sphere.get_points()[index]
        normal = normalize(sphere.data["d_normal_point"][index] - point)
        dot = TrueDot(point, color=YELLOW)
        dot.make_3d()
        dot.deactivate_depth_test()
        dot.set_radius(0.01)
        dot.set_z_index(1)

        in_vect, out_vect = vects = VGroup(
            Vector(sign * 0.2 * normal, thickness=0.5, fill_color=color).shift(point)
            for sign, color in zip([1, -1], [RED, BLUE])
        )
        for vect in vects:
            vect.set_perpendicular_to_camera(frame)

        self.add(dot)
        self.play(
            FadeIn(dot),
            frame.animate.reorient(-47, 81, 0, (-0.55, 0.17, 0.13), 0.65),
            run_time=3
        )
        self.play(GrowArrow(in_vect))
        self.play(GrowArrow(out_vect))
        self.wait()

        # Show a homotopy
        def homotopy(x, y, z, t):
            alpha = clip((x + 3) / 6 - 1 + 2 * t, 0, 1)
            shift = wiggle(alpha, 3)
            return (x, y, z + 0.35 * shift)

        self.play(
            FadeOut(vects, time_span=(0, 1)),
            FadeOut(dot, time_span=(0, 1)),
            Homotopy(homotopy, sphere_group),
            frame.animate.reorient(-44, 78, 0, (-0.16, 0.02, 0.31), 4.18),
            run_time=6,
        )


    def get_warped_sphere_group(self, uv_func, radius=2, mesh_resolution=(61, 31), u_range=(0, TAU), v_range=(0, PI)):
        surface = ParametricSurface(uv_func, u_range=u_range, v_range=v_range, resolution=(201, 101))
        surface.always_sort_to_camera(self.camera)
        surface.set_color(GREY_D)
        mesh = SurfaceMesh(surface, resolution=mesh_resolution, normal_nudge=0)
        mesh.set_stroke(WHITE, 0.5, 0.25)
        mesh.deactivate_depth_test()
        result = Group(surface, mesh)
        result.scale(radius, about_point=ORIGIN)
        return result


class InsideOut(InteractiveScene):
    def construct(self):
        # Show sphere
        frame = self.frame
        self.camera.light_source.move_to([-3, 3, 3])
        radius = 3
        inner_scale = 0.999
        axes = ThreeDAxes((-5, 5), (-5, 5), (-5, 5))
        axes.set_stroke(WHITE, 1, 0.5)
        axes.apply_depth_test()
        axes.z_axis.rotate(0.1 * DEG, RIGHT)

        sphere = self.get_colored_sphere(radius, inner_scale)
        sphere.set_clip_plane(UP, radius)

        mesh = SurfaceMesh(sphere[0], resolution=(61, 31))
        mesh.set_stroke(WHITE, 1, 0.25)

        frame.reorient(-68, 70, 0)
        self.add(axes, sphere, mesh)
        self.play(sphere.animate.set_clip_plane(UP, 0), run_time=2)

        # Show point go to antipoode
        point = radius * normalize(LEFT + OUT)
        p_dot = TrueDot(point, color=YELLOW, radius=0.05).make_3d()
        p_label = Tex(R"p")
        p_label.rotate(90 * DEG, RIGHT).rotate(90 * DEG, IN)
        p_label.next_to(p_dot, OUT, SMALL_BUFF)

        neg_p_dot = p_dot.copy().move_to(-point)
        neg_p_label = Tex(R"-p")
        neg_p_label.rotate(90 * DEG, RIGHT)
        neg_p_label.next_to(neg_p_dot, IN, SMALL_BUFF)

        neg_p_dot.move_to(p_dot)

        dashed_line = DashedLine(point, -point, buff=0)
        dashed_line.set_stroke(YELLOW, 2)

        semi_circle = Arc(135 * DEG, 180 * DEG, radius=radius)
        semi_circle.set_stroke(YELLOW, 3)
        semi_circle.rotate(90 * DEG, RIGHT, about_point=ORIGIN)
        dashed_semi_circle = DashedVMobject(semi_circle, num_dashes=len(dashed_line))

        self.play(
            FadeIn(p_dot, scale=0.5),
            Write(p_label),
        )
        self.play(
            ShowCreation(dashed_semi_circle),
            MoveAlongPath(neg_p_dot, semi_circle, rate_func=linear),
            TransformFromCopy(p_label, neg_p_label, time_span=(2, 3)),
            Rotate(p_label, 90 * DEG, OUT),
            frame.animate.reorient(59, 87, 0),
            run_time=3,
        )
        frame.add_ambient_rotation(-2 * DEG)
        self.wait(2)
        self.play(ReplacementTransform(dashed_semi_circle, dashed_line))
        self.wait(3)

        # Show more antipodes
        angles = np.linspace(0, 90 * DEG, 15)[1:]
        top_dots, low_dots, lines = groups = [
            Group(
                template.copy().rotate(angle, axis=UP, about_point=ORIGIN)
                for angle in angles
            )
            for template in [p_dot, neg_p_dot, dashed_line]
        ]
        for group in groups:
            group.set_submobject_colors_by_gradient(YELLOW, BLUE, interp_by_hsl=True)

        self.play(FadeIn(top_dots, lag_ratio=0.1))
        frame.clear_updaters()
        self.play(
            LaggedStartMap(ShowCreation, lines, lag_ratio=0.25),
            LaggedStart(
                (TransformFromCopy(top_dot, low_dot, rate_func=linear)
                for top_dot, low_dot in zip(top_dots, low_dots)),
                lag_ratio=0.25,
                group_type=Group,
            ),
            frame.animate.reorient(-49, 60, 0),
            run_time=7
        )

        # Just show the cap
        cap = self.get_colored_sphere(radius, inner_scale, v_range=(0.75 * PI, PI))
        frame.clear_updaters()

        self.add(cap, sphere, mesh)
        self.play(
            FadeOut(p_label),
            FadeOut(neg_p_label),
            FadeOut(top_dots),
            FadeOut(lines),
            FadeOut(low_dots),
            FadeOut(p_dot),
            FadeOut(neg_p_dot),
            FadeOut(dashed_line),
            FadeOut(sphere, 0.1 * UP),
            FadeIn(cap),
        )
        self.wait()
        self.play(frame.animate.reorient(-48, 93, 0), run_time=3)
        self.wait()

        # Add normal vectors
        uv_samples = np.array([
            [(u + v) % TAU, v]
            for v in np.linspace(0.75 * PI, 0.95 * PI, 10)
            for u in np.linspace(0, TAU, 20)
        ])
        normal_vectors = VGroup(
            VMobject().set_points_as_corners([ORIGIN, RIGHT, RIGHT, 2 * RIGHT])
            for sample in uv_samples
        )
        for vect in normal_vectors:
            vect.set_stroke(BLUE_B, width=[1, 1, 1, 6, 3, 0], opacity=0.5)
        normal_vectors.apply_depth_test()

        def update_normal_vectors(normal_vectors):
            points = np.array([cap[0].uv_to_point(u, v) for u, v in uv_samples])
            du_points = np.array([cap[0].uv_to_point(u + 0.1, v) for u, v in uv_samples])
            dv_points = np.array([cap[0].uv_to_point(u, v + 0.1) for u, v in uv_samples])
            normals = normalize_along_axis(np.cross(du_points - points, dv_points - points), 1)
            for point, normal, vector in zip(points, normals, normal_vectors):
                vector.put_start_and_end_on(point, point + 0.3 * normal)

        update_normal_vectors(normal_vectors)
        self.play(FadeIn(normal_vectors))

        # Show transition to antipode
        anti_cap = cap.copy().rotate(PI, axis=OUT, about_point=ORIGIN).stretch(-1, 2, about_point=ORIGIN)
        anti_cap[0].shift(1e-2 * OUT)
        all_points = cap[0].get_points()
        indices = random.sample(list(range(len(all_points))), 200)
        pre_points = all_points[indices]
        post_points = -1 * pre_points

        antipode_lines = VGroup(
            Line(point, -point)
            for point in pre_points
        )
        antipode_lines.set_stroke(YELLOW, 1, 0.25)
        antipode_lines.apply_depth_test()

        def update_lines(lines):
            points1 = cap[0].get_points()[indices]
            points2 = anti_cap[0].get_points()[indices]
            for line, p1, p2 in zip(lines, points1, points2):
                line.put_start_and_end_on(p1, p2)

        antipode_lines.add_updater(update_lines)

        rot_arcs = VGroup(
            Arrow(4 * RIGHT, 4 * LEFT, path_arc=180 * DEG, thickness=5),
            Arrow(4 * LEFT, 4 * RIGHT, path_arc=180 * DEG, thickness=5),
        )
        flip_arrows = VGroup(
            Arrow(3 * IN, 3.2 * OUT, thickness=5),
            Arrow(3 * OUT, 3.2 * IN, thickness=5),
        ).rotate(90 * DEG).shift(4 * RIGHT)

        self.play(
            ShowCreation(antipode_lines, lag_ratio=0, suspend_mobject_updating=True),
            frame.animate.reorient(8, 79, 0, (0.0, 0.02, 0.0)),
            run_time=4,
        )
        self.wait()
        normal_vectors.add_updater(update_normal_vectors)
        self.play(
            Write(rot_arcs, lag_ratio=0, run_time=1),
            Rotate(cap, PI, axis=OUT, run_time=3, about_point=ORIGIN),
            Rotate(mesh, PI, axis=OUT, run_time=3, about_point=ORIGIN),
        )
        self.play(FadeOut(rot_arcs))
        self.play(
            FadeIn(flip_arrows, time_span=(0, 1)),
            Transform(cap, anti_cap),
            mesh.animate.stretch(-1, 2, about_point=ORIGIN),
            VFadeOut(antipode_lines),
            run_time=3
        )
        self.play(FadeOut(flip_arrows))
        self.play(frame.animate.reorient(-9, 96, 0, (0.0, 0.02, 0.0)), run_time=5)
        self.wait()

        # Antipode homotopy

    def get_colored_sphere(
        self,
        radius=3,
        inner_scale=0.999,
        outer_color=BLUE_E,
        inner_color=GREY_BROWN,
        u_range=(0, TAU),
        v_range=(0, PI),
    ):
        outer_sphere = Sphere(radius=radius, u_range=u_range, v_range=v_range)
        inner_sphere = outer_sphere.copy()
        outer_sphere.set_color(outer_color, 1)
        inner_sphere.set_color(inner_color, 1)
        inner_sphere.scale(inner_scale)
        return Group(outer_sphere, inner_sphere)

    def old_homotopy(self):
        def homotopy(x, y, z, t, scale=-1):
            p = np.array([x, y, z])
            power = 1 + 0.2 * (x / radius)
            return interpolate(p, scale * p, t**power)

        antipode_lines = VGroup(
            Line(point, -point)
            for point in random.sample(list(cap[0].get_points()), 100)
        )
        antipode_lines.set_stroke(YELLOW, 1, 0.25)

        self.play(
            Homotopy(lambda x, y, z, t: homotopy(x, y, z, t, -inner_scale), cap[0]),
            Homotopy(lambda x, y, z, t: homotopy(x, y, z, t, -1.0 / inner_scale), cap[1]),
            ShowCreation(antipode_lines, lag_ratio=0),
            frame.animate.reorient(-65, 68, 0),
            run_time=3
        )
        self.play(FadeOut(antipode_lines))
        self.play(frame.animate.reorient(-48, 148, 0), run_time=3)
        self.wait()
        self.play(frame.animate.reorient(-70, 83, 0), run_time=3)
        self.play(frame.animate.reorient(-124, 77, 0), run_time=10)


class UnitNormals(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        surface = Torus()
        surface.set_color(GREY_D)

        uv_samples = np.array([
            [u, v]
            for v in np.linspace(0, TAU, 25)
            for u in np.linspace(0, TAU, 50)
        ])
        points = np.array([surface.uv_to_point(u, v) for u, v in uv_samples])
        uv_samples = uv_samples[np.argsort(points[:, 0])]

        normal_vectors = VGroup(
            VMobject().set_points_as_corners([ORIGIN, RIGHT, RIGHT, 2 * RIGHT])
            for sample in uv_samples
        )
        for vect in normal_vectors:
            vect.set_stroke(BLUE_D, width=[2, 2, 2, 12, 6, 0], opacity=0.5)
        normal_vectors.set_stroke(WHITE)
        normal_vectors.apply_depth_test()
        normal_vectors.set_flat_stroke(False)

        def update_normal_vectors(normal_vectors):
            points = np.array([surface.uv_to_point(u, v) for u, v in uv_samples])
            du_points = np.array([surface.uv_to_point(u + 0.1, v) for u, v in uv_samples])
            dv_points = np.array([surface.uv_to_point(u, v + 0.1) for u, v in uv_samples])
            normals = normalize_along_axis(np.cross(du_points - points, dv_points - points), 1)
            for point, normal, vector in zip(points, normals, normal_vectors):
                vector.put_start_and_end_on(point, point + 0.5 * normal)

        update_normal_vectors(normal_vectors)

        frame.reorient(51, 57, 0, (0.39, 0.01, -0.5), 8.00)
        frame.add_ambient_rotation(5 * DEG)
        self.add(surface)
        self.add(normal_vectors)
        self.play(ShowCreation(normal_vectors, run_time=3))
        self.wait(6)


class DefineOrientation(InsideOut):
    def construct(self):
        # Latitude and Longitude
        frame = self.frame
        radius = 3
        sphere = Sphere(radius=radius)
        sphere.set_color(GREY_E, 1)
        earth = TexturedSurface(sphere, "EarthTextureMap", "NightEarthTextureMap")
        earth.set_opacity(0.5)
        mesh = SurfaceMesh(sphere, resolution=(73, 37))
        mesh.set_stroke(WHITE, 1, 0.25)

        uv_tracker = ValueTracker(np.array([180 * DEG, 90 * DEG]))

        dot = TrueDot()
        dot.set_color(YELLOW)
        dot.add_updater(lambda m: m.move_to(sphere.uv_func(*uv_tracker.get_value())))
        dot.set_z_index(2)

        lat_label, lon_label = lat_lon_labels = VGroup(
            Tex(R"\text{Lat: }\, 10^\circ"),
            Tex(R"\text{Lon: }\, 10^\circ"),
        )
        lat_lon_labels.arrange(DOWN, aligned_edge=LEFT)
        lat_lon_labels.fix_in_frame()
        lat_lon_labels.to_corner(UL)
        lat_label.make_number_changeable("10", edge_to_fix=RIGHT).add_updater(
            lambda m: m.set_value(np.round(self.get_lat_lon(*uv_tracker.get_value())[0]))
        )
        lon_label.make_number_changeable("10", edge_to_fix=RIGHT).add_updater(
            lambda m: m.set_value(np.round(self.get_lat_lon(*uv_tracker.get_value())[1]))
        )
        lat_lon_labels.add_updater(lambda m: m.fix_in_frame())

        self.add(sphere, mesh)
        self.add(lat_lon_labels)
        frame.reorient(-66, 85, 0, (-0.06, 0.18, 0.06), 6.78)
        self.play(FadeIn(dot))

        lon_line = TracedPath(dot.get_center, stroke_color=RED)
        self.add(lon_line, dot)
        self.play(uv_tracker.animate.increment_value([0, 30 * DEG]), run_time=4)
        lon_line.suspend_updating()

        lat_line = TracedPath(dot.get_center, stroke_color=TEAL)
        self.add(lat_line, dot)
        self.play(uv_tracker.animate.increment_value([45 * DEG, 0]), run_time=4)
        lat_line.suspend_updating()

        # Add labels to all the points
        u, v = uv_tracker.get_value()

        label_template = Tex(R"(10^\circ, 10^\circ)", isolate=["10"])
        label_template.set_backstroke(BLACK, 1)
        lon_num_template, lat_num_template = label_template.make_number_changeable("10", replace_all=True)

        def get_lat_lon_label(u, v, font_size=5):
            lat, lon = self.get_lat_lon(u, v)
            lat_num_template.set_value(np.round(lat))
            lon_num_template.set_value(np.round(lon))
            label = label_template.copy()
            label.scale(font_size / 48)
            label.move_to(sphere.get_zenith())
            label.rotate(PI - v, axis=RIGHT, about_point=ORIGIN)
            label.rotate(-270 * DEG + u, axis=OUT, about_point=ORIGIN)
            return label

        u_radius = 50 * DEG
        v_radius = 30 * DEG
        all_labels = VGroup(
            get_lat_lon_label(sub_u, sub_v)
            for sub_u in np.arange(u - u_radius, u + u_radius, 10 * DEG)
            for sub_v in np.arange(v - v_radius, v + v_radius, 5 * DEG)
        )
        all_labels.sort(lambda p: get_norm(p - dot.get_center()))

        self.play(
            frame.animate.reorient(-39, 60, 0, (-0.64, 0.45, 0.06), 4.33),
            run_time=3
        )
        self.play(
            FadeIn(all_labels, lag_ratio=0.001, run_time=3),
            dot.animate.set_opacity(0.25),
        )
        self.wait()

        # Show some kind of warping
        def homotopy(x, y, z, t):
            alpha = clip((x + 3) / 6 - 1 + 2 * t, 0, 1)
            shift = wiggle(alpha, 3)
            return (x, y, z + 0.35 * shift)

        dot.suspend_updating()
        group = Group(sphere, mesh, lat_line, lon_line, dot, all_labels)

        self.play(
            Homotopy(homotopy, group, run_time=10)
        )
        dot.resume_updating()
        self.wait()

        # Show tangent vectors
        u, v = uv_tracker.get_value()
        epsilon = 1e-4
        point = sphere.uv_func(u, v)
        u_step = normalize(sphere.uv_func(u + epsilon, v) - point)
        v_step = normalize(sphere.uv_func(u, v + epsilon) - point)

        u_vect = Arrow(point, point + 0.5 * u_step, buff=0, thickness=2).set_color(TEAL)
        v_vect = Arrow(point, point + 0.5 * v_step, buff=0, thickness=2).set_color(RED)
        tangent_vects = VGroup(u_vect, v_vect)
        tangent_vects.set_z_index(1)
        tangent_vects.set_fill(opacity=0.8)
        for vect in tangent_vects:
            vect.set_perpendicular_to_camera(frame)

        self.play(
            dot.animate.set_opacity(1).scale(0.5),
            all_labels.animate.set_stroke(width=0).set_fill(opacity=0.25),
        )
        self.wait()

        lat_line2 = TracedPath(dot.get_center, stroke_color=TEAL, stroke_width=1)
        lat_line2.set_scale_stroke_with_zoom(True)
        self.add(lat_line2)
        self.play(
            uv_tracker.animate.increment_value([30 * DEG, 0]).set_anim_args(rate_func=wiggle),
            FadeOut(lon_line),
            run_time=3
        )
        lat_line2.clear_updaters()
        self.play(GrowArrow(tangent_vects[0]))
        self.wait()

        lon_line2 = TracedPath(dot.get_center, stroke_color=RED, stroke_width=1)
        lon_line2.set_scale_stroke_with_zoom(True)
        self.add(lon_line2)
        self.play(
            tangent_vects[0].animate.set_fill(opacity=0.5).set_anim_args(time_span=(0, 1)),
            uv_tracker.animate.increment_value([0, 30 * DEG]).set_anim_args(rate_func=wiggle),
            FadeOut(lat_line),
            run_time=3
        )
        lon_line2.clear_updaters()
        self.play(GrowArrow(tangent_vects[1]))
        self.play(tangent_vects.animate.set_fill(opacity=1))
        self.wait()

        # Show normal vector
        normal_vect = Arrow(
            point, point + 0.5 * np.cross(u_step, v_step),
            thickness=2,
            buff=0
        )
        normal_vect.set_fill(BLUE, 0.8)
        normal_vect.rotate(90 * DEG, axis=normal_vect.get_vector())

        self.play(
            GrowArrow(normal_vect, time_span=(2, 4)),
            FadeOut(lat_lon_labels, time_span=(0, 2)),
            frame.animate.reorient(-96, 54, 0, (-0.23, -1.08, 0.25), 3.67),
            run_time=8
        )

        # Show a full vector field
        normal_field = get_sphereical_vector_field(
            lambda p: p,
            ThreeDAxes(),
            points=np.array([
                sphere.uv_func(u, v)
                for u in np.arange(0, TAU, TAU / 60)
                for v in np.arange(0, PI, PI / 30)
            ])
        )

        normal_field.save_state()
        normal_field.set_stroke(width=1e-6)

        self.play(
            Restore(normal_field, time_span=(2, 5)),
            frame.animate.reorient(-103, 62, 0, (-0.09, 0.25, 0.23), 7.26),
            run_time=10
        )
        self.wait()
        self.add(self.camera.light_source)

        # Move around light
        light = GlowDot(radius=0.5, color=WHITE)
        light.move_to(self.camera.light_source)
        light.save_state()
        self.camera.light_source.always.move_to(light)

        self.add(self.camera.light_source)
        self.play(
            light.animate.move_to(4 * normalize(light.get_center())),
            run_time=3
        )
        self.play(Rotate(light, TAU, axis=OUT, about_point=ORIGIN, run_time=6))
        self.play(
            FadeOut(normal_field),
            FadeOut(normal_vect),
            Restore(light),
            all_labels.animate.set_fill(opacity=0.5),
            frame.animate.reorient(-63, 66, 0, (-0.09, 0.25, 0.23), 7.26),
            run_time=3
        )

        # Warp sphere
        dot.suspend_updating()
        group = Group(sphere, mesh, lat_line2, lon_line2, all_labels, tangent_vects, dot)
        group.save_state()
        group.target = group.generate_target()
        group.target.rotate(-45 * DEG)
        group.target.scale(0.5)
        group.target.stretch(0.25, 0)
        group.target.shift(RIGHT)
        group.target.apply_complex_function(lambda z: z**3)
        group.target.center()
        group.target.set_height(6)
        group.target.rotate(45 * DEG)

        self.play(Homotopy(homotopy, group, run_time=3))
        self.play(
            MoveToTarget(group),
            frame.animate.reorient(-94, 60, 0, (0.5, 0.84, 0.91), 1.06),
            run_time=8
        )
        self.wait()
        self.play(MoveAlongPath(dot, lat_line2, run_time=5))
        self.play(MoveAlongPath(dot, lon_line2, run_time=5))

        new_normal_vector = Vector(
            0.35 * normalize(np.cross(tangent_vects[0].get_vector(), tangent_vects[1].get_vector())),
            fill_color=BLUE,
            thickness=1
        )
        new_normal_vector.shift(tangent_vects[0].get_start())
        new_normal_vector.set_perpendicular_to_camera(self.frame)
        self.play(
            GrowArrow(new_normal_vector),
            frame.animate.reorient(-125, 52, 0, (0.5, 0.84, 0.91), 1.06),
            run_time=2
        )
        self.wait()
        self.play(
            FadeOut(new_normal_vector, time_span=(0, 1)),
            Restore(group),
            frame.animate.reorient(-78, 66, 0, (-0.13, 0.12, 0.13), 6.54),
            run_time=5
        )

        # Show antipode map
        group = Group(lat_line2, lon_line2, tangent_vects, dot)
        anti_group = group.copy().scale(-1, min_scale_factor=-np.inf, about_point=ORIGIN)

        antipode_lines = VGroup(
            Line(p1, p2)
            for index in [0, 1]
            for p1, p2 in zip(group[index].get_points(), anti_group[index].get_points())
        )
        antipode_lines.set_stroke(YELLOW, 1, 0.1)

        self.play(
            FadeOut(sphere, scale=0.9),
            FadeOut(all_labels),
            VGroup(lat_line2, lon_line2).animate.set_stroke(width=1),
            FadeIn(normal_vect),
            frame.animate.reorient(-93, 55, 0, (-0.26, -0.85, 0.24), 4.27),
            run_time=2
        )

        self.play(
            TransformFromCopy(group, anti_group),
            frame.animate.reorient(-194, 105, 0, (0.7, -0.22, -0.61), 4.64),
            ShowCreation(antipode_lines, lag_ratio=0),
            run_time=5
        )
        self.play(antipode_lines.animate.set_stroke(opacity=0.02))
        self.wait()

        # Map over labels
        all_labels.set_fill(opacity=0.5)
        anti_labels = all_labels.copy().scale(-1, min_scale_factor=-np.inf, about_point=ORIGIN)

        for label in anti_labels:
            label.rotate(PI, axis=np.cross(label.get_center(), IN))

        self.play(FadeIn(all_labels, lag_ratio=0.001, run_time=3))
        self.remove(all_labels)
        self.play(TransformFromCopy(all_labels, anti_labels, lag_ratio=0.0002, run_time=3))
        self.wait()
        self.play(
            MoveAlongPath(dot, anti_group[0], run_time=5),
            anti_group[1].animate.set_stroke(opacity=0.25),
            anti_group[2][1].animate.set_fill(opacity=0.25),
        )
        self.play(
            MoveAlongPath(dot, anti_group[1], run_time=5),
            anti_group[1].animate.set_stroke(opacity=1),
            anti_group[2][1].animate.set_fill(opacity=1),
            anti_group[0].animate.set_stroke(opacity=0.25),
            anti_group[2][0].animate.set_fill(opacity=0.25),
        )
        self.wait()
        self.play(
            anti_group[0].animate.set_stroke(opacity=1),
            anti_group[2][0].animate.set_fill(opacity=1),
        )

        # New normal
        new_normal = normal_vect.copy()
        new_normal.shift(-2 * point)

        self.play(
            GrowArrow(new_normal),
            frame.animate.reorient(-197, 90, 0, (1.16, -0.3, -0.99), 4.64),
            run_time=3
        )
        self.wait()

        # Show reverserd vector field
        anti_normal_field = get_sphereical_vector_field(
            lambda p: -p,
            ThreeDAxes(),
            points=np.array([
                sphere.uv_func(u, v)
                for u in np.arange(0, TAU, TAU / 60)
                for v in np.arange(0, PI, PI / 30)
            ])
        )

        frame.reorient(162, 89, 0, (1.16, -0.3, -0.99), 4.64)
        self.play(
            FadeOut(group),
            FadeOut(antipode_lines),
            FadeIn(anti_normal_field, time_span=(0, 2)),
            frame.animate.reorient(360 - 177, 82, 0, (-0.03, -0.11, 0.86), 9.51),
            run_time=10
        )

        # Show outward vectors again
        self.remove(anti_normal_field)
        self.remove(anti_group)
        self.remove(new_normal)
        self.add(group)
        self.add(normal_field)
        antipode_lines.set_stroke(YELLOW, 1, 0.1)
        self.play(ShowCreation(antipode_lines, lag_ratio=0, run_time=2))
        self.wait()

    def get_lat_lon(self, u, v):
        return np.array([
            v / DEG - 90,
            u / DEG - 180,
        ])


class FlowingWater(InteractiveScene):
    def construct(self):
        # Set up axes
        radius = 2
        frame = self.frame
        axes = ThreeDAxes((-3, 3), (-3, 3), (-3, 3))
        axes.scale(radius)
        frame.reorient(-89, 77, 0)
        frame.add_ambient_rotation(2 * DEG)
        self.add(axes)

        # Add water
        water = self.get_water(sigma0=0.2, n_droplets=1_000_000, opacity=0.1, refresh_ratio=0.015)
        water.scale(0.01, about_point=ORIGIN)
        source_dot = GlowDot(ORIGIN, color=BLUE)

        self.add(source_dot, water)
        water.refresh_sigma_tracker.set_value(0.2)
        water.opacity_tracker.set_value(0.05)
        water.radius_tracker.set_value(0.015)
        frame.reorient(-108, 73, 0, (0.01, 0.06, -0.03), 4.0),
        self.play(
            water.radius_tracker.animate.set_value(0.02),
            water.opacity_tracker.animate.set_value(0.1),
            water.refresh_sigma_tracker.animate.set_value(2.5),
            frame.animate.reorient(-120, 80, 0, ORIGIN, 8.00),
            run_time=10
        )
        self.wait(20)

        # Show full sphere
        sphere = Sphere(radius=radius)
        sphere.set_color(GREY, 0.25)
        sphere.set_shading(0.5, 0.5, 0.5)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere, resolution=(61, 31))
        mesh.set_stroke(WHITE, 1, 0.5)
        mesh.set_z_index(2)

        def get_unit_normal_field(u_range, v_range):
            return get_sphereical_vector_field(
                lambda p: p,
                ThreeDAxes(),
                points=np.array([
                    sphere.uv_func(u, v)
                    for u in u_range
                    for v in v_range
                ])
            )

        normal_field = get_unit_normal_field(
            np.arange(0, TAU, TAU / 60),
            np.arange(0, PI, PI / 30),
        )

        self.play(
            ShowCreation(sphere),
            Write(mesh, lag_ratio=1e-3),
            run_time=2
        )
        self.wait(2)
        self.play(FadeIn(normal_field))
        self.wait(7.2)

        # Show single patch
        u_range_params = (0 * DEG, 15 * DEG, 5 * DEG)
        v_range_params = (120 * DEG, 130 * DEG, 5 * DEG)
        patch = Sphere(
            radius=radius,
            u_range=u_range_params[:2],
            v_range=v_range_params[:2],
        )
        patch.set_color(WHITE, 0.6)

        patch_normals = get_unit_normal_field(np.arange(*u_range_params), np.arange(*v_range_params))

        self.play(
            FadeOut(sphere, time_span=(0, 1)),
            FadeOut(normal_field, time_span=(0, 1)),
            FadeIn(patch, time_span=(0, 1)),
            FadeIn(patch_normals, time_span=(0, 1)),
            mesh.animate.set_stroke(opacity=0.1),
            frame.animate.reorient(19, 55, 0, (1.65, 0.28, 0.98), 2.32),
            water.opacity_tracker.animate.set_value(0.125),
            water.radius_tracker.animate.set_value(0.01),
            run_time=5
        )
        frame.clear_updaters()
        self.wait(8)

        patch_group = Group(patch, patch_normals)

        # Original patch behavior
        self.play(
            Rotate(patch_group, PI, axis=UP),
            run_time=2
        )
        self.wait(2.5)
        self.play(
            Rotate(patch_group, PI, axis=UP, time_span=(0, 1)),
            FadeIn(sphere),
            FadeIn(normal_field),
            frame.animate.reorient(30, 78, 0, (0.19, -0.03, 0.09), 5.67),
            water.opacity_tracker.animate.set_value(0.15),
            water.radius_tracker.animate.set_value(0.015),
            run_time=5
        )
        self.play(FadeOut(patch_group))
        frame.add_ambient_rotation(-1 * DEG)
        self.wait(3)
        self.play(
            FadeOut(normal_field),
            sphere.animate.set_shading(1, 1, 1),
        )
        self.wait(5)

        # Show deformations
        sphere_group = Group(sphere, mesh)

        def alt_uv_func(u, v, params, wiggle_size=1.0, max_freq=4):
            x, y, z = sphere.uv_func(u, v)
            return (
                x + wiggle_size * params[0] * np.cos(max_freq * params[1] * y),
                y + wiggle_size * params[4] * np.cos(max_freq * params[5] * z),
                z + wiggle_size * params[2] * np.cos(max_freq * params[3] * x),
            )

        np.random.seed(3)
        for n in range(20):
            params = np.random.random(6)
            new_sphere = ParametricSurface(
                lambda u, v: alt_uv_func(u, v, params),
                u_range=sphere.u_range,
                v_range=sphere.v_range,
                resolution=sphere.resolution
            )
            new_sphere.match_style(sphere)
            new_mesh = SurfaceMesh(new_sphere, resolution=mesh.resolution)
            new_mesh.match_style(mesh)
            new_group = Group(new_sphere, new_mesh)

            if 10 < n < 13:
                new_group.shift(1.75 * radius * RIGHT)

            self.play(Transform(sphere_group, new_group, run_time=2))
            self.wait(2)

    def get_water(
        self,
        n_droplets=500_000,
        radius=0.02,
        opacity=0.2,
        sigma0=3,
        refresh_sigma=2.5,
        velocity=10,
        refresh_ratio=0.01,
    ):
        points = np.random.normal(0, sigma0, (n_droplets, 3))
        water = DotCloud(points)
        water.set_radius(radius)
        water.set_color(BLUE)
        water.opacity_tracker = ValueTracker(opacity)
        water.radius_tracker = ValueTracker(radius)
        water.refresh_sigma_tracker = ValueTracker(refresh_sigma)
        water.velocity = velocity

        def flow_out(water, dt):
            if dt == 0:
                pass
            points = water.get_points()
            radii = np.linalg.norm(points, axis=1)
            denom = 4 * PI * radii**2
            denom[denom == 0] = 1
            vels = points / denom[:, np.newaxis]
            new_points = points + water.velocity * vels * dt

            n_refreshes = int(refresh_ratio * len(points))
            indices = np.random.randint(0, len(points), n_refreshes)
            new_points[indices] = np.random.normal(0, water.refresh_sigma_tracker.get_value(), (n_refreshes, 3))
            water.set_points(new_points)
            water.set_opacity(water.opacity_tracker.get_value() / np.clip(radii, 1, np.inf))
            water.set_radius(water.radius_tracker.get_value())
            return water

        water.add_updater(flow_out)

        return water

    def rotate_patch(self):
        # to be inserted in "show single patch" above
        patch_group.clear_updaters()
        point = patch.get_center()
        path = Arc(0, PI)
        path.rotate(20 * DEG, RIGHT)
        path.put_start_and_end_on(point, -point)

        patch_group.move_to(path.get_start())
        self.wait(5)
        self.play(
            MoveAlongPath(patch_group, path),
            frame.animate.reorient(-25, 92, 0, (-0.75, -0.96, -0.16), 3.16),
            run_time=5
        )
        self.wait(5)


class SurfaceFoldedOverSelf(FlowingWater):
    def construct(self):
        # Set up axes
        radius = 2
        frame = self.frame
        axes = ThreeDAxes((-3, 3), (-3, 3), (-3, 3))
        axes.scale(radius)
        frame.reorient(-20, 77, 0)
        self.add(axes)

        water = self.get_water()
        self.add(water)

        # Deform sphere
        sphere = Sphere(radius=radius)
        sphere.set_color(GREY, 0.25)
        sphere.set_shading(0.5, 0.5, 0.5)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere, resolution=(61, 31))
        mesh.set_stroke(WHITE, 1, 0.25)
        mesh.set_z_index(2)

        sphere_group = Group(sphere, mesh)
        self.add(sphere_group)
        self.wait(10)

        # Morph
        def inflate(points):
            radii = np.linalg.norm(points, axis=1)
            new_radii = 2.0 * (1 - 1 / (radii + 1))
            return points * (new_radii / radii)[:, np.newaxis]

        sphere_target = sphere_group.copy()
        sphere_target.rotate(90 * DEG, LEFT)
        sphere_target.stretch(0.2, 0)
        sphere_target.stretch(0.5, 1)
        sphere_target.move_to(RIGHT)
        sphere_target.apply_complex_function(lambda z: z**3)
        sphere_target.rotate(90 * DEG, RIGHT)
        sphere_target.shift(radius * OUT)
        sphere_target.apply_points_function(inflate, about_point=ORIGIN)
        sphere_target.replace(sphere)

        sphere_group.save_state()
        self.play(
            frame.animate.reorient(-25, 76, 0, (-0.13, 0.29, 0.85), 3.38).set_anim_args(time_span=(2, 5)),
            Transform(sphere_group, sphere_target),
            water.radius_tracker.animate.set_value(0.01),
            water.opacity_tracker.animate.set_value(0.1),
            run_time=5
        )
        frame.add_ambient_rotation(0.5 * DEG)

        # Show vectors
        direction = OUT + 0.2 * LEFT

        vect = Vector(0.35 * direction, fill_color=GREEN)
        vect.set_perpendicular_to_camera(frame)
        vect.shift(1.1 * direction)
        vect.set_fill(opacity=0.75)

        vect1 = vect.copy()
        vect2 = vect.copy().shift(0.4 * direction).set_color(RED)
        vect3 = vect.copy().shift(0.8 * direction)

        dashed_line = DashedLine(ORIGIN, 5 * direction, dash_length=0.02)
        dashed_line.set_stroke(WHITE, 4)
        dashed_line_ghost = dashed_line.copy()
        dashed_line_ghost.set_stroke(opacity=0.25)
        dashed_line.apply_depth_test()

        self.play(
            ShowCreation(dashed_line, run_time=5),
            ShowCreation(dashed_line_ghost, run_time=5),
        )
        self.wait()
        self.play(
            GrowArrow(vect1),
            GrowArrow(vect3),
        )
        self.wait(3)
        self.play(GrowArrow(vect2))
        self.wait(4)
        self.play(
            FadeOut(VGroup(vect1, vect2, vect3)),
            FadeOut(dashed_line),
            FadeOut(dashed_line_ghost),
        )

        # Shift back
        frame.clear_updaters()
        self.play(
            Restore(sphere_group),
            frame.animate.reorient(10, 78, 0, ORIGIN, 7),
            run_time=7
        )


class InsideOutWithNormalField(InteractiveScene):
    def construct(self):
        # Axes and plane
        radius = 3
        frame = self.frame
        axes = ThreeDAxes()
        plane = NumberPlane((-8, 8), (-8, 8))
        plane.background_lines.set_stroke(GREY_C, 1)
        plane.faded_lines.set_stroke(GREY_C, 0.5, 0.5)
        plane.apply_depth_test()
        axes.apply_depth_test()
        self.add(axes, plane)

        # Add sphere
        outer_sphere = Sphere(radius=radius)
        inner_sphere = Sphere(radius=0.99 * radius)
        outer_sphere.set_color(BLUE_E, 1)
        inner_sphere.set_color(interpolate_color(GREY_BROWN, BLACK, 0.5), 1)
        for sphere in [inner_sphere, outer_sphere]:
            sphere.always_sort_to_camera(self.camera)
            sphere.set_shading(0.2, 0.2, 0.1)
        mesh = SurfaceMesh(outer_sphere, resolution=(61, 31))
        mesh.set_stroke(WHITE, 1, 0.5)
        mesh.set_z_index(2)

        def get_unit_normal_field(u_range, v_range, sign=1):
            return get_sphereical_vector_field(
                lambda p: sign * p,
                axes,
                points=np.array([
                    sphere.uv_func(u, v)
                    for u in u_range
                    for v in v_range
                ]),
                color=BLUE_D,
                stroke_width=3
            )

        u_range = np.arange(0, TAU, TAU / 30)
        v_range = np.arange(0, PI, PI / 15)
        outer_normals = get_unit_normal_field(u_range, v_range)
        inner_normals = get_unit_normal_field(u_range, v_range, -1)

        sign_tracker = ValueTracker(1)
        outer_normals.add_updater(lambda m: m.set_stroke(opacity=float(sign_tracker.get_value() > 0)))
        inner_normals.add_updater(lambda m: m.set_stroke(opacity=float(sign_tracker.get_value() < 0)))

        sphere_group = Group(inner_sphere, outer_sphere, mesh, outer_normals, inner_normals)
        sphere_group.set_clip_plane(UP, 0)
        frame.reorient(-23, 72, 0)
        self.add(sphere_group, axes)

        # Inversion homotopy
        def homotopy(x, y, z, t, scale=-1):
            p = np.array([x, y, z])
            power = 1 + 0.2 * (x / radius)
            return interpolate(p, scale * p, t**power)

        frame.add_ambient_rotation(3 * DEG)
        self.play(
            Homotopy(homotopy, sphere_group),
            sign_tracker.animate.set_value(-1),
            run_time=5
        )
        self.wait(3)


class ProjectedCombedHypersphere(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        axes = ThreeDAxes((-5, 5), (-5, 5), (-5, 5))
        plane = NumberPlane((-5, 5), (-5, 5))
        plane.background_lines.set_stroke(BLUE, 1)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.5)
        frame.reorient(86, 78, 0, (-0.13, 0.04, 0.63))
        self.add(axes)

        # Add lines
        flow_lines = VGroup(
            self.get_flow_line_from_point(normalize(np.random.normal(0, 1, 4)))
            for n in range(2000)
        )
        for line in flow_lines:
            line.virtual_time = TAU
            line.get_center()
            stroke_width = clip(get_norm(line.get_center()), 0, 3)
            color = random_bright_color(hue_range=(0.45, 0.55))
            line.set_stroke(color, stroke_width)

        self.add(flow_lines)

        frame.add_ambient_rotation(4 * DEG)
        self.play(
            LaggedStartMap(
                VShowPassingFlash,
                flow_lines,
                lag_ratio=3 / len(flow_lines),
                run_time=45,
                time_width=0.7,
                rate_func=linear
            )
        )

    def get_flow_line_from_point(self, point4d, stroke_color=WHITE, stroke_width=2):
        points4d = self.get_hypersphere_circle_points(point4d)
        points3d = self.streo_4d_to_3d(points4d)
        line = VMobject().set_points_smoothly(points3d)
        line.set_stroke(stroke_color, stroke_width)
        return line

    def get_hypersphere_circle_points(self, point4d, n_samples=100):
        x, y, z, w = point4d
        perp = np.array([-y, x, -w, z])
        return np.array([
            math.cos(a) * point4d + math.sin(a) * perp
            for a in np.linspace(0, TAU, n_samples)
        ])

    def streo_4d_to_3d(self, points4d):
        xyz = points4d[:, :3]
        w = points4d[:, 3]

        # Stereographic projection formula: (x, y, z) / (1 - w)
        # Reshape w for broadcasting
        denominator = (1 - w).reshape(-1, 1)

        # Handle potential division by zero (north pole)
        # Add small epsilon to avoid exact division by zero
        epsilon = 1e-10
        denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)

        projected = xyz / denominator

        return projected
