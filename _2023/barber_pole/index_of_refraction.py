from manim_imports_ext import *
from _2023.barber_pole.objects import *


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
