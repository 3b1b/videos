from manim_imports_ext import *
from _2023.barber_pole.objects import *


class WaveIntoMedium(TimeVaryingVectorField):
    def __init__(
        self,
        interface_origin=ORIGIN,
        interface_normal=DR,
        prop_direction=RIGHT,
        index=1.5,
        c=2.0,
        frequency=0.25,
        amplitude=1.0,
        x_density=5.0,
        y_density=5.0,
        width=15.0,
        height=15.0,
        norm_to_opacity_func=lambda n: np.tanh(n),
        **kwargs
    ):
        def time_func(points, time):
            k = frequency / c
            phase = TAU * (k * np.dot(points, prop_direction.T) - frequency * time)
            kickback = np.dot(points - interface_origin, interface_normal.T)
            kickback[kickback < 0] = 0
            phase += kickback * (index - 1) * c
            return amplitude * np.outer(np.cos(phase)**2, OUT)

        super().__init__(
            time_func,
            x_density=x_density,
            y_density=y_density,
            width=width,
            height=height,
            norm_to_opacity_func=norm_to_opacity_func,
            **kwargs
        )


# Scenes

class SpeedInMediumFastPart(InteractiveScene):
    wave_config = dict(
        z_amplitude=0,
        y_amplitude=1.0,
        color=YELLOW,
        wave_len=3.0,
        speed=1.5,
    )

    def construct(self):
        # Basic wave
        axes = ThreeDAxes((-12, 12), (-4, 4))
        axes.z_axis.set_stroke(opacity=0)
        wave = OscillatingWave(axes, **self.wave_config)
        vect_wave = OscillatingFieldWave(axes, wave)
        vect_wave.set_stroke(opacity=0.5)

        self.add(axes, wave, vect_wave)

        # Water label
        rect = FullScreenRectangle()
        rect.stretch(0.5, 0, about_edge=RIGHT)
        rect.set_stroke(width=0)
        rect.set_fill(BLUE, 0.35)
        label = Text("Water", font_size=60)
        label.next_to(rect.get_top(), DOWN)
        self.add(rect, label)

        # Propagate
        self.wait(30)


class SpeedInMediumSlower(SpeedInMediumFastPart):
    wave_config = dict(
        z_amplitude=0,
        y_amplitude=1.0,
        color=YELLOW,
        wave_len=2.0,
        speed=1.0,
    )


class VectorOverMedia(InteractiveScene):
    def construct(self):
        vect = Vector(DOWN)
        vect.next_to(UP, UP)
        vect.to_edge(LEFT, buff=0)

        def update_vect(v, dt):
            speed = 1.5 if v.get_x() < 0 else 1.0
            v.shift(dt * RIGHT * speed)

        vect.add_updater(update_vect)
        self.add(vect)
        self.wait(13)


class PhaseKickBacks(SpeedInMediumFastPart):
    n_layers = 10
    kick_back_value = -0.5
    line_style = dict(
        stroke_width=2.0,
        stroke_color=BLUE_B,
    )
    layer_add_on_run_time = 5

    def construct(self):
        # Axes
        axes = ThreeDAxes((-12, 12), (-4, 4))
        axes.z_axis.set_stroke(opacity=0)

        # Layers and trackers
        lines = Line(DOWN, UP, **self.line_style).replicate(self.n_layers)
        lines.set_height(5)
        lines.arrange(RIGHT)
        lines.arrange_to_fit_width(FRAME_WIDTH / 2)
        lines.move_to(ORIGIN, LEFT)

        layer_xs = np.array([axes.x_axis.p2n(line.get_center()) for line in lines])
        phase_kick_trackers = Group(*(ValueTracker(0) for line in lines))

        # Set up wave
        wave = OscillatingWave(axes, **self.wave_config)

        def wave_func(x, t):
            phase = np.ones_like(x)
            phase *= TAU * t * wave.speed / wave.wave_len
            for layer_x, pkt in zip(layer_xs, phase_kick_trackers):
                phase[x > layer_x] += pkt.get_value()

            y = wave.y_amplitude * np.sin(TAU * x / wave.wave_len - phase)
            return axes.c2p(x, y, 0 * x)

        wave.wave_func = wave_func

        vect_wave = OscillatingFieldWave(axes, wave)
        vect_wave.set_stroke(opacity=0.5)

        self.add(wave, vect_wave)
        lag_kw = dict(run_time=self.layer_add_on_run_time, lag_ratio=0.5)
        self.play(
            LaggedStart(*(
                FadeIn(line, 0.5 * DOWN)
                for line in lines
            ), **lag_kw),
            LaggedStart(*(
                pkt.animate.set_value(self.kick_back_value)
                for pkt in phase_kick_trackers
            ), **lag_kw),
        )
        self.wait(8)


class DensePhaseKickBacks25(PhaseKickBacks):
    n_layers = 25
    kick_back_value = -0.4
    line_style = dict(
        stroke_width=1.0,
        stroke_color=BLUE_B,
    )


class DensePhaseKickBacks50(PhaseKickBacks):
    n_layers = 50
    kick_back_value = -0.2
    line_style = dict(
        stroke_width=1.0,
        stroke_color=BLUE_B,
    )


class DensePhaseKickBacks100(PhaseKickBacks):
    n_layers = 100
    kick_back_value = -0.15
    line_style = dict(
        stroke_width=1.5,
        stroke_color=BLUE_C,
        stroke_opacity=0.7,
    )
    layer_add_on_run_time = 10


class WavesIntoAngledMedium(InteractiveScene):
    default_frame_orientation = (0, 62)

    def construct(self):
        # Add setup
        interface_origin = ORIGIN
        interface_normal = np.array([1, -0.2, 0.0])
        prop_direction = RIGHT
        index = 3.0
        substance = VCube(side_length=1.0)
        substance.stretch(10, 0)
        substance.stretch(15, 1)
        substance.stretch(5, 2)
        substance.set_fill(BLUE, 0.2)
        substance.move_to(interface_origin, LEFT)
        substance.rotate(angle_of_vector(interface_normal), about_point=interface_origin)

        self.add(substance)

        # Wave
        wave = WaveIntoMedium(
            interface_origin=interface_origin,
            interface_normal=interface_normal,
            prop_direction=prop_direction,
            index=index,
            amplitude=0.5,
            c=1.0,
            # norm_to_opacity_func=lambda n: np.clip(0.5 * n, 0, 1)**10,
            max_vect_len=np.inf,
            norm_to_opacity_func=lambda n: np.abs(2 * n)**2,
        )
        self.add(wave)
        self.wait(20)


class ResponsiveCharge(InteractiveScene):
    def construct(self):
        # Driving chrage
        charge1 = ChargedParticle(charge=0.25)
        charge1.add_spring_force(k=10)
        charge1.move_to(0.3 * DOWN)

        # Responsive charge
        k = 20
        charge2 = ChargedParticle(charge=1.0, radius=0.1, show_sign=False)
        charge2.move_to(2.5 * RIGHT)
        # charge2.add_field_force(field)
        charge2.add_spring_force(k=k)
        charge2.add_force(lambda p: wave.wave_func(p[0], wave.time) * [0, 1, 1])
        # charge2.fix_x()
        self.add(charge2)

        # E field
        # field_type = ColoumbPlusLorentzField
        field_type = LorentzField
        field = field_type(
            charge1, charge2,
            x_density=4.0,
            y_density=4.0,
            norm_to_opacity_func=lambda n: np.clip(0.5 * n, 0, 1),
            c=1.0,
        )
        self.add(field)

        # Pure wave
        axes = ThreeDAxes()
        wave = OscillatingWave(axes, y_amplitude=1.0, z_amplitude=0.0, wave_len=2.0)
        field_wave = OscillatingFieldWave(axes, wave)
        wave.set_stroke(opacity=0.5)


        self.add(axes, wave, field_wave)

        # omega = (wave.speed / wave.wave_len) * TAU
        # omega_0 = math.sqrt(k / charge2.mass)
        # v0 = omega / (omega_0**2 - omega**2)
        # charge2.velocity = v0 * UP

        self.wait(20)

        # Plane
        plane = NumberPlane()
        plane.fade(0.5)
        self.add(plane)

        # Test wiggle
        self.play(
            charge1.animate.shift(UP).set_anim_args(
                rate_func=wiggle,
                run_time=3,
            )
        )
        self.wait(4)
