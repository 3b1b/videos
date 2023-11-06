from manim_imports_ext import *
from _2023.barber_pole.objects import *


class SlicedWave(Group):
    default_wave_config = dict(
        z_amplitude=0,
        y_amplitude=1,
        wave_len=2.0,
        color=BLUE,
    )
    default_layer_style = dict(
        stroke_width=2.0,
        stroke_color=WHITE,
    )
    default_vect_wave_style = dict(
        stroke_opacity=0.5
    )

    def __init__(
        self,
        axes,
        layer_xs,
        phase_kick_back=0,
        layer_height=4.0,
        wave_config = dict(),
        vect_wave_style=dict(),
        layer_style=dict(),
    ):
        self.layer_xs = layer_xs
        self.axes = axes
        wave_kw = merge_dicts_recursively(self.default_wave_config, wave_config)
        vwave_kw = merge_dicts_recursively(self.default_vect_wave_style, vect_wave_style)
        line_kw = merge_dicts_recursively(self.default_layer_style, layer_style)

        self.wave = OscillatingWave(axes, **wave_kw)
        self.vect_wave = OscillatingFieldWave(axes, self.wave, **vwave_kw)
        self.phase_kick_trackers = [
            ValueTracker(phase_kick_back)
            for x in layer_xs
        ]
        self.layers = VGroup()
        for x in layer_xs:
            line = Line(DOWN, UP, **line_kw)
            line.set_height(layer_height)
            line.move_to(axes.c2p(x, 0))
            self.layers.add(line)

        self.wave.xt_to_yz = self.xt_to_yz

        super().__init__(
            self.wave,
            self.vect_wave,
            self.layers,
            *self.phase_kick_trackers
        )

    def xt_to_yz(self, x, t):
        phase = np.ones_like(x)
        phase *= TAU * t * self.wave.speed / self.wave.wave_len
        for layer_x, pkt in zip(self.layer_xs, self.phase_kick_trackers):
            phase[x > layer_x] += pkt.get_value()

        y = self.wave.y_amplitude * np.sin(TAU * x / self.wave.wave_len - phase)
        return y, np.zeros_like(x)


# Scenes

class SpeedInMediumFastPart(InteractiveScene):
    z_amplitude = 0
    y_amplitude = 1.0
    color = YELLOW
    wave_len = 3.0
    speed = 1.5
    medium_color = BLUE
    medium_opacity = 0.35
    add_label = True
    run_time = 30

    def construct(self):
        # Basic wave
        axes = ThreeDAxes((-12, 12), (-4, 4))
        axes.z_axis.set_stroke(opacity=0)
        axes.y_axis.set_stroke(opacity=0)
        wave = OscillatingWave(
            axes,
            z_amplitude=self.z_amplitude,
            y_amplitude=self.y_amplitude,
            color=self.color,
            wave_len=self.wave_len,
            speed=self.speed,
        )
        vect_wave = OscillatingFieldWave(axes, wave)
        vect_wave.set_stroke(opacity=0.5)

        self.add(axes, wave, vect_wave)

        # Water label
        rect = FullScreenRectangle()
        rect.stretch(0.5, 0, about_edge=RIGHT)
        rect.set_stroke(width=0)
        rect.set_fill(self.medium_color, self.medium_opacity)
        self.add(rect)

        if self.add_label:
            label = Text("Water", font_size=60)
            label.next_to(rect.get_top(), DOWN)
            self.add(label)

        # Propagate
        self.wait(self.run_time)


class SpeedInMediumSlower(SpeedInMediumFastPart):
    wave_len = 2.0
    speed = 1.0


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


class VioletWaveFast(SpeedInMediumFastPart):
    color = get_spectral_color(0.95)
    wave_len = 1.2
    y_amplitude = 0.5
    speed = 1.5
    medium_opacity = 0.25
    add_label = False
    run_time = 15


class VioletWaveSlow(VioletWaveFast):
    wave_len = 1.2 * 0.5
    speed = 1.5 * 0.5


class GreenWaveFast(VioletWaveFast):
    color = get_spectral_color(0.65)
    wave_len = 1.5


class GreenWaveSlow(GreenWaveFast):
    wave_len = 1.5 * 0.6
    speed = 1.5 * 0.6


class OrangeWaveFast(VioletWaveFast):
    color = get_spectral_color(0.3)
    wave_len = 2.0


class OrangeWaveSlow(OrangeWaveFast):
    wave_len = 2.0 * 0.7
    speed = 1.5 * 0.7


class RedWaveFast(VioletWaveFast):
    color = get_spectral_color(0.05)
    wave_len = 2.5


class RedWaveSlow(RedWaveFast):
    wave_len = 2.5 * 0.8
    speed = 1.5 * 0.8


class PhaseKickBacks(SpeedInMediumFastPart):
    layer_xs = np.arange(0, 8, 1)
    kick_back_value = 0
    axes_config = dict(
        x_range=(-8, 8),
        y_range=(-4, 4),
    )
    line_style = dict()
    wave_config = dict()
    vect_wave_style = dict()
    layer_add_on_run_time = 5

    def get_axes(self):
        axes = ThreeDAxes(**self.axes_config)
        axes.z_axis.set_stroke(opacity=0)
        return axes

    def get_layer_xs(self):
        return self.layer_xs

    def get_sliced_wave(self):
        return SlicedWave(
            self.get_axes(),
            self.get_layer_xs(),
            wave_config=self.wave_config,
            vect_wave_style=self.vect_wave_style,
            layer_style=self.line_style,
            phase_kick_back=self.kick_back_value,
        )

    def setup(self):
        super().setup()
        self.sliced_wave = self.get_sliced_wave()
        self.add(self.sliced_wave)


class RevertToOneLayerAtATime(PhaseKickBacks):
    # layer_xs = np.arange(-0, 7.5, 0.11)
    # kick_back_value = -0.25
    # n_layers_skipped = 8
    layer_xs = np.arange(0, FRAME_WIDTH / 2, FRAME_WIDTH / 2**(11))
    kick_back_value = -0.025
    n_layers_skipped = 128

    exagerated_phase_kick = -0.8

    line_style = dict(
        stroke_width=1,
        stroke_opacity=0.25,
        stroke_color=BLUE_B
    )
    axes_config = dict(
        x_range=(-12, 12),
        y_range=(-4, 4),
    )
    wave_config = dict(
        color=YELLOW,
        sample_resolution=0.001
    )
    vect_wave_style = dict(tip_width_ratio=3, stroke_opacity=0.5)

    def construct(self):
        # Objects
        sliced_wave = self.sliced_wave
        wave = sliced_wave.wave
        layers = sliced_wave.layers
        pkts = sliced_wave.phase_kick_trackers

        # Revert to one single layer
        block_label = Text("Material (e.g. glass)")
        block_label.next_to(layers, UP, aligned_edge=LEFT)
        layer_label = Text("Thin layer of material")
        layer_label.next_to(layers[0], UP, buff=0.75)

        self.add(block_label)
        self.wait(5)

        kw = dict(run_time=5, lag_ratio=0.01)
        self.play(
            LaggedStart(*(
                layer.animate().set_stroke(width=5, opacity=0).set_height(5)
                for layer in layers[:0:-1]
            ), **kw),
            LaggedStart(*(
                pkt.animate.set_value(0)
                for pkt in pkts[:0:-1]
            ), **kw),
            layers[0].animate.set_stroke(width=2, opacity=1).set_height(5).set_anim_args(time_span=(4, 5)),
            FadeTransform(
                block_label, layer_label,
                time_span=(4, 5)
            ),
        )
        self.wait(4)

        # Pause, shift the phase a bit
        arrow = Vector(1.0 * LEFT, stroke_width=6)
        arrow.next_to(wave, UP, buff=0.75)
        arrow.set_x(0, LEFT).shift(0.1 * RIGHT)
        phase_kick = self.exagerated_phase_kick
        kick_words = Text("Kick back\nthe phase", font_size=36)
        kick_words.next_to(arrow, RIGHT)
        kick_label = VGroup(arrow, kick_words)
        kick_label.set_color(RED)

        self.wait(1 - (wave.time % 1))
        wave.stop_clock()
        self.wait()
        self.play(
            pkts[0].animate.set_value(phase_kick),
            FadeIn(kick_label, LEFT),
        )
        self.wait()

        # Show the phase kick
        form1 = Tex(R"A\sin(kx)")
        form2 = Tex(R"A\sin(kx + 1.00)")
        pk_decimal: DecimalNumber = form2.make_number_changable("1.00")
        pk_decimal.set_value(-phase_kick)
        pk_decimal.set_color(RED)

        VGroup(form1, form2).next_to(wave, DOWN)
        form1.set_x(-FRAME_WIDTH / 4)
        form2.set_x(+FRAME_WIDTH / 4)


        self.play(FadeIn(form1, 0.5 * DOWN))
        self.wait()
        self.play(TransformMatchingStrings(form1.copy(), form2, path_arc=20 * DEGREES))
        self.wait()
        for value in [-0.01, phase_kick]:
            self.play(
                pkts[0].animate.set_value(value),
                ChangeDecimalToValue(pk_decimal, -value),
                run_time=2,
            )
            self.wait()

        # Add phase kick label
        pk_label = Text("Phase kick = ", font_size=36, fill_color=GREY_B)
        pk_label.next_to(wave, UP, LARGE_BUFF)
        pk_label.to_edge(LEFT)

        form2.remove(pk_decimal)
        form2.add(pk_decimal.copy())
        self.add(form2)
        self.play(
            pk_decimal.animate.match_height(pk_label).next_to(pk_label, RIGHT, 0.2, DOWN),
            ReplacementTransform(kick_words["Kick"].copy(), pk_label["kick"]),
            ReplacementTransform(kick_words["phase"].copy(), pk_label["Phase"]),
            FadeIn(pk_label["="]),
            run_time=2
        )
        pk_label.add(pk_decimal)
        self.add(pk_label)
        self.play(
            FadeOut(form1),
            FadeOut(form2),
        )

        # Show following layers
        for layer in layers[1:]:
            layer.match_height(layers[0])
            layer.match_style(layers[0])
            layer.set_stroke(opacity=0)

        nls = self.n_layers_skipped
        shown_layers = VGroup(layers[0])
        layer_arrows = VGroup(VectorizedPoint(layer_label.get_center()))
        for layer, pkt in zip(layers[nls::nls], pkts[nls::nls]):
            layer.set_stroke(opacity=1)
            shown_layers.add(layer)

            layer_label.target = layer_label.generate_target()
            layer_label.target.set_height(0.35)
            layer_label.target.match_x(shown_layers)
            layer_label.target.to_edge(UP, buff=0.25)

            layer_arrows.target = VGroup(*(
                Arrow(
                    layer_label.target.get_bottom(),
                    sl.get_top(),
                    buff=0.1,
                    stroke_width=2,
                    stroke_opacity=0.5,
                )
                for sl in shown_layers
            ))

            self.play(
                FadeOut(kick_label),
                GrowFromCenter(layer),
                MoveToTarget(layer_label),
                MoveToTarget(layer_arrows),
            )
            kick_label.align_to(layer, LEFT).shift(0.1 * RIGHT)
            self.play(
                FadeIn(kick_label, 0.5 * LEFT),
                pkt.animate.set_value(phase_kick)
            )
        self.play(FadeOut(kick_label))

        # Restart clock
        wave.start_clock()
        self.play(FadeOut(layer_label), FadeOut(layer_arrows))
        self.wait(6)

        # Change phase kick
        self.play(FlashAround(pk_label))
        for value in [-0.01, 2 * self.exagerated_phase_kick]:
            phase_kick = value
            globals().update(locals())
            self.play(
                ChangeDecimalToValue(pk_decimal, -phase_kick),
                *(
                    pkt.animate.set_value(phase_kick)
                    for pkt in pkts[::nls]
                ),
                run_time=2
            )
            self.wait(4)

        # Number of layers label
        n_shown_layers = len(layers[::nls])
        n_layers_label = TexText(f"Num. layers = {n_shown_layers}", font_size=36)
        n_layers_label.set_color(GREY_B)
        n_layers_label.next_to(pk_label, UP, MED_LARGE_BUFF, LEFT)
        nl_decimal = n_layers_label.make_number_changable(n_shown_layers)
        nl_decimal.set_color(BLUE)

        self.play(FadeIn(n_layers_label, UP))
        self.wait(8)

        # Fill in
        opacity = 1.0
        stroke_width = 2.0
        kw = dict(lag_ratio=0.1, run_time=3)
        while nls > 1:
            # Update parameters
            nls //= 2
            opacity = 0.25 + 0.5 * (opacity - 0.25)
            stroke_width = 1.0 + 0.5 * (stroke_width - 1.0)
            phase_kick /= 2

            new_layers = layers[nls::2 * nls]
            old_layers = layers[0::2 * nls]

            new_nl_decimal = Integer(len(layers[::nls]))
            new_nl_decimal.match_height(nl_decimal)
            new_nl_decimal.match_style(nl_decimal)
            new_nl_decimal.move_to(nl_decimal, LEFT)

            new_pk_decimal = DecimalNumber(
                -phase_kick,
                num_decimal_places=(2 if -phase_kick > 0.05 else 3)
            )
            new_pk_decimal.match_height(pk_decimal)
            new_pk_decimal.match_style(pk_decimal)
            new_pk_decimal.move_to(pk_decimal, LEFT)

            new_layers.set_stroke(width=stroke_width, opacity=opacity)

            globals().update(locals())  # Only necessary for embedded runs
            self.play(
                LaggedStart(*(
                    GrowFromCenter(layer)
                    for layer in new_layers
                ), **kw),
                old_layers.animate.set_stroke(width=stroke_width, opacity=opacity),
                LaggedStart(*(
                    pkt.animate.set_value(phase_kick)
                    for pkt in pkts[::nls]
                ), **kw),
                LaggedStart(
                    FadeOut(nl_decimal, 0.2 * UP),
                    FadeIn(new_nl_decimal, 0.2 * UP),
                    FadeOut(pk_decimal, 0.2 * UP),
                    FadeIn(new_pk_decimal, 0.2 * UP),
                    lag_ratio=0.1,
                    run_time=1
                )
            )

            nl_decimal = new_nl_decimal
            pk_decimal = new_pk_decimal

            globals().update(locals())  # Only necessary for embedded runs
            self.play(*(
                pkt.animate.set_value(phase_kick)
                for pkt in pkts[::nls]
            ))
            self.wait(3 if nls >= 16 else 1)

        # Wait
        self.wait(10)


class DissolveLayers(PhaseKickBacks):
    def construct(self):
        # Test
        sliced_wave = self.sliced_wave
        layers = sliced_wave.layers
        pkts = sliced_wave.phase_kick_trackers

        # Zoom in on layers
        frame = self.frame
        layers_copy = layers.copy()
        layers_copy.set_stroke(width=1)
        fade_rect = FullScreenRectangle().set_fill(BLACK, 1)

        self.wait(4)
        self.add(fade_rect, layers_copy)
        self.play(
            frame.animate.scale(0.2).move_to(layers).set_anim_args(run_time=4),
            FadeIn(fade_rect, time_span=(1, 3)),
            FadeIn(layers_copy, time_span=(1, 3)),
        )
        self.wait()
        self.play(
            frame.animate.to_default_state().set_anim_args(run_time=2),
            FadeOut(fade_rect),
            FadeOut(layers_copy),
        )
        self.wait(2)

        # Dissolve
        kw = dict(run_time=8, lag_ratio=0.1)
        self.play(
            LaggedStart(*(
                layer.animate().set_stroke(WHITE, 2, 0).set_height(5)
                for layer in layers[:0:-1]
            ), **kw),
            LaggedStart(*(
                pkt.animate.set_value(0)
                for pkt in pkts[:0:-1]
            ), **kw),
            layers[0].animate.set_stroke(WHITE, 2).set_height(5).set_anim_args(time_span=(7, 8))
        )
        self.wait(3)


class IntroducePhaseKickBack(PhaseKickBacks):
    layer_xs = np.arange(0, 8, PI / 4)
    vect_wave_style = dict(stroke_width=0)

    def construct(self):
        # Set up sine wave
        sliced_wave = self.sliced_wave
        axes = sliced_wave.axes
        layers = sliced_wave.layers
        pkts = sliced_wave.phase_kick_trackers
        wave = sliced_wave.wave

        self.remove(sliced_wave)
        wave.stop_clock()
        self.add(wave)
        self.add(*pkts)

        # # Pair of braces
        # brace1 = Brace(Line(LEFT_SIDE, ORIGIN, buff=0.25), UP)
        # brace2 = brace1.copy().set_x(FRAME_WIDTH / 4)
        # braces = VGroup(brace1, brace2)
        # braces.set_y(2)
        # self.add(braces)

        # b1_tex = brace1.get_tex(R"\sin(\omega t - kx)")
        # b2_tex = brace2.get_tex(R"\sin(\omega t - kx - \Delta \phi)")

        # self.add(b1_tex)
        # self.add(b2_tex)

        # Add one layer of material
        self.play(GrowFromCenter(layers[0])) # TODO: Some kind of labels here?

        # Show small kick back
        arrow = Vector(2 * LEFT, stroke_width=8)
        arrow.next_to(wave, UP, buff=0.75)
        arrow.set_x(0, LEFT).shift(0.5 * RIGHT)
        phase_kick = -0.5

        self.play(
            pkts[0].animate.set_value(phase_kick),
            FadeIn(arrow, LEFT),
        )
        self.play(FadeOut(arrow))

        # Add more layers of material
        for layer, pkt in zip(layers[1:], pkts[1:]):
            arrow.align_to(layer, LEFT).shift(0.25 * RIGHT)
            self.play(
                GrowFromCenter(layer),
                FadeIn(arrow, LEFT),
                pkt.animate.set_value(phase_kick),
            )
            self.play(FadeOut(arrow), run_time=0.5)

        # Make it all more dense
        pass


class PhaseKickBackAddInLayers(PhaseKickBacks):
    n_layers = 10
    kick_back_value = 0
    target_kick_back = -0.5

    def get_layer_xs(self):
        return np.linspace(0, 8, self.n_layers)

    def construct(self):
        sliced_wave = self.sliced_wave
        lag_kw = dict(run_time=self.layer_add_on_run_time, lag_ratio=0.5)
        self.play(
            LaggedStart(*(
                FadeIn(line, 0.5 * DOWN)
                for line in sliced_wave.layers
            ), **lag_kw),
            LaggedStart(*(
                pkt.animate.set_value(self.target_kick_back)
                for pkt in sliced_wave.phase_kick_trackers
            ), **lag_kw),
        )
        self.wait(8)


class DensePhaseKickBacks25(PhaseKickBackAddInLayers):
    n_layers = 25
    target_kick_back = -0.4
    line_style = dict(
        stroke_width=1.0,
        stroke_color=BLUE_B,
    )


class DensePhaseKickBacks50(PhaseKickBackAddInLayers):
    n_layers = 50
    target_kick_back = -0.2
    line_style = dict(
        stroke_width=1.0,
        stroke_color=BLUE_B,
    )


class DensePhaseKickBacks100(PhaseKickBackAddInLayers):
    n_layers = 100
    target_kick_back = -0.15
    line_style = dict(
        stroke_width=1.5,
        stroke_color=BLUE_C,
        stroke_opacity=0.7,
    )
    layer_add_on_run_time = 10


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
        charge2.add_force(lambda p: wave.xt_to_point(p[0], wave.time) * [0, 1, 1])
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
