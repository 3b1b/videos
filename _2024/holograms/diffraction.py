from manim_imports_ext import *


def hsl_to_rgb(hsl):
    """
    Convert an array of HSL values to RGB.

    Args:
    hsl (np.ndarray): A numpy array of shape (n, 3), where each row represents an HSL value
                      (Hue [0, 1), Saturation [0, 1], Lightness [0, 1]).

    Returns:
    np.ndarray: An array of shape (n, 3), containing RGB values in the range [0, 1].
    """
    h = hsl[:, 0]
    s = hsl[:, 1]
    l = hsl[:, 2]

    def hue_to_rgb(p, q, t):
        t = np.where(t < 0, t + 1, np.where(t > 1, t - 1, t))
        return np.where(t < 1/6, p + (q - p) * 6 * t,
               np.where(t < 1/2, q,
               np.where(t < 2/3, p + (q - p) * (2/3 - t) * 6, p
        )))

    q = np.where(l < 0.5, l * (1 + s), l + s - l * s)
    p = 2 * l - q

    r = hue_to_rgb(p, q, h + 1/3)
    g = hue_to_rgb(p, q, h)
    b = hue_to_rgb(p, q, h - 1/3)

    rgb = np.stack([r, g, b], axis=1)
    return rgb


class PointSourceDiffractionPattern(InteractiveScene):
    def construct(self):
        # Set up axes
        self.set_floor_plane("xz")
        axes = ThreeDAxes()
        axes.set_stroke(opacity=0.5)
        self.add(axes)

        # Set up light sources
        radius = 2.0
        n_sources = 5
        ring = Circle(radius=radius)
        ring.set_stroke(WHITE, 3)
        points = DotCloud(np.array([ring.pfp(a) for a in np.arange(0, 1, 1 / n_sources)]))
        points.set_glow_factor(2)
        points.set_radius(0.2)
        points.set_color(WHITE)
        self.add(points)

        # Add dots
        dots = DotCloud()
        dots.to_grid(80, 80)
        dots.set_radius(0.05)
        dots2 = dots.copy()
        dots2.rotate(PI / 2, RIGHT)
        dots.set_points(np.vstack([dots.get_points(), dots2.get_points()]))

        dots.set_z(5)
        self.add(dots)

        # Color the dots
        freq_tracker = ValueTracker(2.0)
        center_tracker = Point()
        point_sources = points.get_points()
        dots.add_updater(lambda d: self.color_dot_cloud_by_diffraction(
            point_sources, d, freq_tracker.get_value(),
            max_mag=3.0,
        ))
        dots.add_updater(lambda m: m.move_to(center_tracker.get_center()))
        dots.add_updater(lambda m: m.set_color(WHITE))

        # Animate
        self.play(
            self.frame.animate.reorient(-53, -8, 0, (0.41, -0.3, 0.12), 15.29),
            run_time=3
        )

        self.play(
            freq_tracker.animate.set_value(2.5),
            run_time=8,
        )

        self.play(
            center_tracker.animate.move_to(4 * OUT),
            run_time=5
        )


    def color_dot_cloud_by_diffraction(self, point_sources, dot_cloud, frequency=1.0, max_mag=5.0):
        centers = dot_cloud.get_points()
        diffs = centers[:, np.newaxis, :] - point_sources[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        amplitudes = np.exp(distances * TAU * 1j * frequency).sum(1)
        hues = (np.log(amplitudes).imag / TAU) % 1
        mags = abs(amplitudes)
        opacities = 0.5 * np.clip(mags / max_mag, 0, 1)

        n = len(centers)
        hsl = 0.5 * np.ones((n, 3))
        hsl[:, 0] = hues
        rgbas = np.ones((n, 4))
        rgbas[:, :3] = hsl_to_rgb(hsl)
        rgbas[:, 3] = opacities

        dot_cloud.set_rgba_array(rgbas)
        return dot_cloud

    def color_group_by_diffraction(self, point_sources, mobjects, frequency=1.0, max_mag=5.0):
        # Old implementation
        for mob in mobjects:
            center = mob.get_center()
            diffs = point_sources - center
            distances = np.linalg.norm(diffs, axis=1) 
            amplitude = np.exp(distances * TAU * 1j * frequency).sum()
            hue = np.log(amplitude).imag / TAU
            mag = abs(amplitude)
            opacity = clip(mag / max_mag, 0, 1)
            mob.set_fill(
                Color(hsl=(hue, 0.5, 0.5)),
                opacity=opacity
            )
        return mobjects


    def old_color_dots(self, dots, ring, frequency = 1.0):
        for dot in dots:
            dist = get_norm(dot.get_center() - ring.get_top())
            phase = frequency * dist % 1
            amp = 1.0 / dist**2
            color = Color(hsl=(phase, 0.5, 0.5))
            dot.set_width(
                10 * amp,
                stretch=1
            )
            dot.set_color(color)
