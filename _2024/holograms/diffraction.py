from __future__ import annotations
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


class DiffractionPattern(Mobject):
    shader_folder: str = str(Path(Path(__file__).parent, "diffraction_shader"))
    data_dtype: Sequence[Tuple[str, type, Tuple[int]]] = [
        ('point', np.float32, (3,)),
    ]
    render_primitive: int = moderngl.TRIANGLES

    def __init__(
        self,
        point_sources: Vect3Array = np.zeros((1, 3)),
        shape: tuple[float, float] = (4.0, 4.0),
        color: ManimColor = WHITE,
        opacity: float = 1.0,
        frequency: float = 2.0,
        wave_number: float = 2.0,
        max_amp: Optional[float] = None,
        **kwargs
    ):
        self.shape = shape
        super().__init__(**kwargs)

        if max_amp is None:
            max_amp = len(point_sources)
        self.set_uniforms(dict(
            frequency=frequency,
            wave_number=wave_number,
            max_amp=max_amp,
        ))
        self.set_color(color, opacity)
        self.set_point_sources(point_sources)

    def init_data(self) -> None:
        super().init_data(length=6)
        self.data["point"][:] = [UL, DL, UR, DR, UR, DL]

    def init_points(self) -> None:
        self.set_shape(*self.shape)

    def init_uniforms(self):
        super().init_uniforms()

    def set_color(
        self,
        color: ManimColor | Iterable[ManimColor] | None,
        opacity: float | Iterable[float] | None = None,
    ) -> Self:
        if color is not None:
            self.set_uniform(color=color_to_rgb(color))
        if opacity is not None:
            self.set_uniform(opacity=opacity)
        return self

    def set_point_sources(self, sources: Vect3Array):
        full_point_sources = np.zeros((16, 3))
        full_point_sources[:len(sources)] = sources
        for n, source in enumerate(sources):
            self.set_uniform(**{f"point_source{n}": source})
        self.set_uniform(n_sources=len(sources))


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


class DiffractionTest(InteractiveScene):
    def construct(self):
        # Test
        plate = DiffractionPattern(
            color=RED,
            opacity=0.5,
        )
        plate.set_height(FRAME_HEIGHT)
        plate.set_uniform(wave_number=10)
        sources = GlowDots(np.array([UR, UL]) + OUT)
        sources.set_color(WHITE)
        plate.f_always.set_point_sources(sources.get_points)

        self.add(plate)
        self.add(sources)

        # Move sources
        self.play(sources.animate.move_to(2 * OUT), run_time=3)
        self.play(Rotate(sources, PI, axis=UP, run_time=4))

        sources.add_point(UP)

        circle = Circle()
        circle.rotate(PI / 2, RIGHT)
        sources.set_points([
            circle.pfp(a)
            for a in np.arange(0, 1, 1 / 8)
        ])

        plates = Group(plate.copy() for x in range(5))
        plates.arrange(OUT, buff=0.1)
        plates.move_to(ORIGIN, OUT)
        for plate in plates:
            plate.set_color(RED, opacity=0.25)
        self.add(plates)

        self.play(Rotate(plates, 90 * DEGREES, axis=RIGHT, run_time=5))
        self.play(plates.animate.move_to(3 * UP))


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
