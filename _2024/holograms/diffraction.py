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

    r = hue_to_rgb(p, q, h + 1 / 3)
    g = hue_to_rgb(p, q, h)
    b = hue_to_rgb(p, q, h - 1 / 3)

    rgb = np.stack([r, g, b], axis=1)
    return rgb


class PointSourceDiffractionPattern(InteractiveScene):
    # default_frame_orientation = (-35, -10)
    include_axes = True
    axes_config = dict(
        x_range=(-6, 6),
        y_range=(-6, 6),
        z_range=(-6, 6),
    )
    light_frequency = 2.0
    wave_length = 1.0
    use_hue = False
    max_mag = 3.0

    def setup(self):
        super().setup()

        self.set_floor_plane("xz")
        self.frame.reorient(-35, -10)

        if self.include_axes:
            axes = self.axes = ThreeDAxes(**self.axes_config)
            axes.set_stroke(opacity=0.5)
            self.add(axes)

        # Set up light sources
        points = self.point_sources = DotCloud(self.get_point_source_locations())
        points.set_glow_factor(2)
        points.set_radius(0.2)
        points.set_color(WHITE)
        self.add(points)

        # Add frequency trackerg
        self.frequency_tracker = ValueTracker(self.light_frequency)
        self.wave_length_tracker = ValueTracker(self.wave_length)
        self.light_time_tracker = ValueTracker(0)
        self.max_mag_tracker = ValueTracker(self.max_mag)

    def get_point_source_locations(self):
        radius = 2.0
        n_sources = 5
        ring = Circle(radius=radius)
        ring.set_stroke(WHITE, 3)
        return np.array([ring.pfp(a) for a in np.arange(0, 1, 1 / n_sources)])

    def get_light_time(self):
        return self.light_time_tracker.get_value()

    def get_frequency(self):
        return self.frequency_tracker.get_value()

    def color_dot_cloud_by_diffraction(self, dot_cloud):
        frequency = self.get_frequency()
        point_sources = self.point_sources.get_points()
        max_mag = self.max_mag_tracker.get_value()

        centers = dot_cloud.get_points()
        diffs = centers[:, np.newaxis, :] - point_sources[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        amplitudes = np.exp(distances * TAU * 1j * frequency).sum(1)
        mags = abs(amplitudes)
        opacities = 0.5 * np.clip(mags / max_mag, 0, 1)

        n = len(centers)
        rgbas = dot_cloud.data["rgba"]
        rgbas[:, 3] = opacities

        if self.use_hue:
            hues = (np.log(amplitudes).imag / TAU) % 1
            hsl = 0.5 * np.ones((n, 3))
            hsl[:, 0] = hues
            rgbas[:, :3] = hsl_to_rgb(hsl)

        dot_cloud.set_rgba_array(rgbas)
        return dot_cloud

    def create_dot_sheet(self, width=4, height=4, radius=0.05, z=0, make_3d=False):
        # Add dots
        dots = DotCloud()
        dots.set_color(WHITE)
        dots.to_grid(int(height / radius / 2), int(width / radius / 2))
        dots.set_shape(width, height)
        dots.set_radius(radius)
        dots.add_updater(self.color_dot_cloud_by_diffraction)
        dots.suspend_updating = lambda: None  # Never suspend!
        dots.set_z(z)

        if make_3d:
            dots.make_3d()

        return dots

    # TODO, have a picture-in-picture graph showing the sine waves for a given source
    # TODO, have a picture in picture phasor


class DoubleSlit(PointSourceDiffractionPattern):
    max_mag = 2.0

    def construct(self):
        # General setup
        points = self.point_sources.get_points()
        light = self.camera.light_source
        light.move_to(5 * RIGHT + 5 * UP + 4 * OUT)
        frame = self.frame
        # Nix axes?

        # Set up pair of slits
        buff = 0.1
        mid_block_width = 2 - 2 * buff
        blocks = Group(
            Cube().set_shape(3, 1, 0.5),
            Cube().set_shape(mid_block_width, 1, 0.5),
            Cube().set_shape(3, 1, 0.5),
        )
        blocks.set_color(GREY_D)
        blocks[0].next_to(points[0], LEFT, buff=buff)
        blocks[1].move_to(midpoint(*points))
        blocks[2].next_to(points[1], RIGHT, buff=buff)

        floor_sheet = Square3D(resolution=(21, 21))
        floor_sheet.set_shape(10, 10)
        floor_sheet.rotate(PI / 2, RIGHT)
        floor_sheet.set_opacity(0.5)
        floor_sheet.next_to(blocks, DOWN, buff=0)

        self.add(floor_sheet)
        self.add(blocks)

        # Set up a back screen
        sheet = self.create_dot_sheet(10, 2, radius=0.025, z=-2)
        sheet.align_to(blocks, DOWN)
        sheet.align_to(floor_sheet, IN)
        sheet.update()
        # sheet.clear_updaters()

        self.add(sheet)

        sheet.rotate(PI / 2, LEFT)
        sheet.move_to(self.point_sources, OUT)

        # Testing
        sheets = Group(
            sheet.copy().set_z(z)
            for z in np.arange(-5, 0, 0.5)
        )

        for sheet in sheets:
            sheet.update()
            sheet.clear_updaters()

        for sheet in sheets:
            sheet.data["rgba"][:, 3] *= 0.5

        for sheet in sheets:
            points = sheet.get_points()
            sheet.set_points(points + np.random.uniform(-0.001, 0.001, size=points.shape))

        self.add(sheets)

        # Show intensity over time for one beam of light at a time
        sine_wave = FunctionGraph(math.sin, x_range=(-10 * TAU, 10 * TAU, 0.01 * TAU))
        sine_wave.set_color(GREEN)
        sine_wave.set_width(5)

        end_point = Point(sheet.get_center() + 0.5 * DOWN + 0.25 * RIGHT)

        sine_wave.put_start_and_end_on(points[0], )
        sine_wave.apply_depth_test()

        sine_wave2 = sine_wave.copy()
        sine_wave2.put_start_and_end_on(points[1], sheet.get_center() + 0.5 * DOWN + 0.25 * RIGHT)

        self.add(sine_wave)
        self.add(sine_wave2)


        # Show constructive interference

        # Show deconstructive interference

        # Graph

        # Add dot screen


    def get_point_source_locations(self):
        return [LEFT, RIGHT]
