from manim_imports_ext import *
from _2023.optics_puzzles.objects import *


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
        damping_per_layer=1.0,
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
        self.absorbtion_trackers = [
            ValueTracker(damping_per_layer)
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

    def set_layer_xs(self, xs):
        self.layer_xs = xs

    def xt_to_yz(self, x, t):
        phase = np.ones_like(x)
        phase *= TAU * t * self.wave.speed / self.wave.wave_len
        amplitudes = self.wave.y_amplitude * np.ones_like(x)
        for layer_x, pkt, at in zip(self.layer_xs, self.phase_kick_trackers, self.absorbtion_trackers):
            phase[x > layer_x] += pkt.get_value()
            amplitudes[x > layer_x] *= at.get_value()

        y = amplitudes * np.sin(TAU * x / self.wave.wave_len - phase)
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
    material_label = "Glass"

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
            label = Text(self.material_label, font_size=60)
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
            # speed = 1.5 if v.get_x() < 0 else 1.0
            speed = 1.33 if v.get_x() < 0 else 1.33 / 1.5
            v.shift(dt * RIGHT * speed)

        vect.add_updater(update_vect)
        word = Text("Phase velocity")
        word.add_updater(lambda m: m.next_to(vect, UP))
        self.add(vect)
        self.add(word)
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
    damping_per_layer = 1.0

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
            damping_per_layer=self.damping_per_layer,
        )

    def setup(self):
        super().setup()
        self.sliced_wave = self.get_sliced_wave()
        self.add(self.sliced_wave)


class RevertToOneLayerAtATime(PhaseKickBacks):
    layer_xs = np.arange(0, FRAME_WIDTH / 2, FRAME_WIDTH / 2**(11))
    kick_back_value = -0.025
    n_layers_skipped = 64

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

        # Add label
        block_label = Text("Material (e.g. glass)")
        block_label.next_to(layers, UP, aligned_edge=LEFT)

        for pkt in pkts:
            pkt.set_value(-0.015)

        self.wait(5)
        self.play(Write(block_label["Material"], run_time=1))
        self.play(FadeIn(block_label["(e.g. glass)"], 0.25 * UP))
        self.add(block_label)
        self.wait(2)

        # Show layers
        layers.save_state()

        rect = BackgroundRectangle(sliced_wave)
        rect.set_fill(BLACK, 0.9)
        self.add(sliced_wave, rect, layers)
        self.play(
            layers.animate.arrange(RIGHT, buff=0.3).move_to(ORIGIN, LEFT).set_stroke(width=2, opacity=1),
            FadeIn(rect),
            *(pkt.animate.set_value(0) for pkt in pkts),
            run_time=2,
        )
        self.wait(3)

        # Revert to one single layer
        left_layers = VGroup()
        for layer in layers:
            if layer.get_x() < FRAME_WIDTH / 2:
                left_layers.add(layer)
        self.remove(layers)
        self.add(left_layers)

        layer_label = Text("Thin layer of material")
        layer_label.next_to(layers[0], UP, buff=0.75)

        kw = dict(run_time=5, lag_ratio=0.1)
        self.play(
            LaggedStart(*(
                layer.animate().set_stroke(opacity=0).shift(DOWN)
                for layer in left_layers[:0:-1]
            ), **kw),
            layers[0].animate.set_stroke(width=2, opacity=1).set_height(5).set_anim_args(time_span=(4, 5)),
            FadeTransform(
                block_label, layer_label,
                time_span=(4, 5)
            ),
            FadeOut(rect, time_span=(3, 5)),
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
        pk_decimal: DecimalNumber = form2.make_number_changeable("1.00")
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
        for value in [-0.01, self.exagerated_phase_kick]:
            phase_kick = value
            self.play(
                ChangeDecimalToValue(pk_decimal, -phase_kick),
                *(
                    pkt.animate.set_value(phase_kick)
                    for pkt in pkts[::nls]
                ),
                run_time=4
            )
            self.wait(2)
        self.wait(2)

        # Number of layers label
        n_shown_layers = len(layers[::nls])
        n_layers_label = TexText(f"Num. layers = {n_shown_layers}", font_size=36)
        n_layers_label.set_color(GREY_B)
        n_layers_label.next_to(pk_label, UP, MED_LARGE_BUFF, LEFT)
        nl_decimal = n_layers_label.make_number_changeable(n_shown_layers)
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

            self.play(*(
                pkt.animate.set_value(phase_kick)
                for pkt in pkts[::nls]
            ))
            self.wait(3 if nls >= 32 else 1)

        # Wait
        self.wait(10)


class SimplerRevertToOneLayerAtATime(RevertToOneLayerAtATime):
    layer_xs = np.arange(-0, 7.5, 0.11)
    kick_back_value = -0.25
    n_layers_skipped = 8


class SlowedAndAbsorbed(RevertToOneLayerAtATime):
    layer_xs = np.arange(-FRAME_WIDTH / 3, FRAME_WIDTH / 3, FRAME_WIDTH / 2**(8))
    kick_back_value = -0.2
    damping_per_layer = 1 - 2e-2
    wave_config = dict(
        color=YELLOW,
        sample_resolution=0.001,
    )
    line_style = dict(
        stroke_color=BLUE_B,
        stroke_width=1.5,
        stroke_opacity=0.5,
    )

    def construct(self):
        # Objects
        sliced_wave = self.sliced_wave
        wave = sliced_wave.wave
        layers = sliced_wave.layers
        pkts = sliced_wave.phase_kick_trackers
        ats = sliced_wave.absorbtion_trackers

        # Show labels
        label = Text("Wave gets slowed and absorbed")
        label.next_to(layers, UP)
        abs_words = label["and absorbed"]

        mu_label = Tex(R"\mu").set_color(PINK)
        mu_line = NumberLine((0, 1, 0.2), width=1.0, tick_size=0.05)
        mu_line.rotate(PI / 2)
        mu_line.to_corner(UR)
        mu_indicator = Triangle(start_angle=0)
        mu_indicator.set_width(0.15)
        mu_indicator.set_fill(PINK, 1)
        mu_indicator.set_stroke(width=0)
        mu_tracker = ValueTracker(1)
        mu_indicator.add_updater(lambda m: m.move_to(mu_line.n2p(mu_tracker.get_value()), RIGHT))
        mu_label.add_updater(lambda m: m.next_to(mu_indicator, LEFT, buff=0.1))

        for pkt in pkts:
            pkt.set_value(0)
        for at in ats:
            at.set_value(1)

        self.play(
            FadeIn(label, time_span=(0, 2)),
            FlashAround(abs_words, color=PINK, time_span=(2, 4)),
            abs_words.animate.set_color(PINK).set_anim_args(time_span=(2, 4)),
            FadeIn(mu_line),
            FadeIn(mu_indicator),
            FadeIn(mu_label),
            LaggedStart(*(
                pkt.animate.set_value(self.kick_back_value)
                for pkt in pkts
            )),
            LaggedStart(*(
                at.animate.set_value(self.damping_per_layer)
                for at in ats
            )),
            LaggedStart(*(
                FadeIn(layer, scale=0.8)
                for layer in layers
            )),
            run_time=5,
        )

        # Play with mu
        for mu in [0.1, 1.0]:
            self.play(
                mu_tracker.animate.set_value(mu),
                *(at.animate.set_value(1 - mu * 2e-2) for at in ats),
                run_time=3,
            )

        self.wait(17)


class PlayWithIndex(RevertToOneLayerAtATime):
    layer_xs = np.arange(-FRAME_WIDTH / 4, FRAME_WIDTH / 4, FRAME_WIDTH / 2**(10))

    def construct(self):
        # Objects
        sliced_wave = self.sliced_wave
        wave = sliced_wave.wave
        layers = sliced_wave.layers
        layers.set_stroke(BLUE, 1, 0.5)
        pkts = sliced_wave.phase_kick_trackers

        global_pk = ValueTracker(self.kick_back_value)
        for pkt in pkts:
            pkt.add_updater(lambda m: m.set_value(global_pk.get_value()))

        def get_index():
            return 1 - 20 * global_pk.get_value()

        # equation label
        equation = Tex(
            R"""
                \text{Index of refraction } = 
                {\small \text{Speed in a vacuum} \over \text{Speed in medium}}
                = 1.00
            """,
            t2c={
                 R"\text{Speed in a vacuum}": YELLOW,
                 R"\text{Speed in medium}": BLUE,
            }
        )
        equation.next_to(layers, UP)
        rhs = equation.make_number_changeable("1.00")

        rhs.add_updater(lambda m: m.set_value(get_index()))

        arrow = Arrow(rhs.get_bottom(), layers.get_corner(UR) + 1.0 * DL)

        self.add(equation)
        self.add(arrow)

        # Speed label
        speed_label = Tex(R"\text{Speed} = c / 1.00")
        speed_factor = speed_label.make_number_changeable("1.00")
        speed_factor.add_updater(lambda m: m.set_value(get_index()))
        speed_label.next_to(layers.get_bottom(), UP)

        self.add(speed_label)

        # Change value
        self.wait(3)
        for value in [0, 0.02]:
            self.play(global_pk.animate.set_value(value), run_time=2)
            self.wait(3)
        self.wait(10)


class RedLight(RevertToOneLayerAtATime):
    wave_config = dict(
        color=RED,
        stroke_width=6,
        sample_resolution=0.001,
        wave_len=4.0,
    )
    kick_back_value = -0.005

    def construct(self):
        # Test
        sliced_wave = self.sliced_wave
        wave = sliced_wave.wave
        layers = sliced_wave.layers
        pkts = sliced_wave.phase_kick_trackers
        sliced_wave.update()
        sliced_wave.clear_updaters()
        sliced_wave.vect_wave.set_stroke(opacity=1)
        sliced_wave.wave.make_jagged()

        new_wave = VMobject()
        new_wave.set_points_smoothly(sliced_wave.wave.get_anchors()[0::5])
        new_wave.match_style(sliced_wave.wave)

        self.add(self.sliced_wave)
        self.add(new_wave)
        self.remove(sliced_wave.layers)


class XRay(RedLight):
    wave_config = dict(
        color=YELLOW,
        stroke_width=6,
        sample_resolution=0.001,
        wave_len=1.0,
    )
    kick_back_value = 0.02


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


class FastWave(PhaseKickBacks):
    layer_xs = np.arange(-3, 3, 0.01)
    kick_back_value = 0.01
    line_style = dict(
        stroke_width=1,
        stroke_opacity=0.35,
        stroke_color=BLUE_D
    )
    wave_config = dict(
        color=YELLOW,
        sample_resolution=0.001
    )

    def construct(self):
        # Test
        self.wait(20)


class KickForward(PhaseKickBacks):
    layer_xs = np.arange(0, 8, 0.25)
    kick_back_value = 0.25
    line_style = dict(
        stroke_color=BLUE,
        stroke_width=2,
        stroke_opacity=0.75,
    )
    wave_config = dict(
        color=YELLOW,
        sample_resolution=0.001
    )
    time_per_layer = 0.25

    def construct(self):
        # Objects
        sliced_wave = self.sliced_wave
        wave = sliced_wave.wave
        layers = sliced_wave.layers
        layers.stretch(1.2, 1)
        pkts = sliced_wave.phase_kick_trackers

        # Just show one layer
        layer_label = Text("Thin layer of material")
        layer_label.next_to(layers[0], UP, buff=0.75)

        for pkt in pkts:
            pkt.set_value(0)

        self.wait(1 - (wave.time % 1))
        wave.stop_clock()
        self.remove(layers)
        self.play(
            ShowCreation(layers[0]),
            FadeIn(layer_label, lag_ratio=0.1),
        )
        self.wait()

        # Kick back then forth
        kick_back = VGroup(
            Vector(LEFT, stroke_width=6),
            Text("Kick back\nthe phase", font_size=36),
        )
        kick_back.set_color(RED)
        kick_forward = VGroup(
            Vector(RIGHT, stroke_width=6),
            Text(
                "Kick forward\nthe phase", font_size=36,
                t2s={"forward": ITALIC}
            ),
        )
        kick_forward.set_color(GREEN)

        for label in [kick_back, kick_forward]:
            label.arrange(RIGHT)
            label.next_to(wave, UP)
            label.align_to(layers[0], LEFT).shift(MED_SMALL_BUFF * RIGHT)

        self.play(
            pkts[0].animate.set_value(-0.75),
            FadeIn(kick_back, LEFT),
        )
        self.wait()
        self.play(FadeOut(kick_back))
        self.play(
            pkts[0].animate.set_value(self.kick_back_value),
            FadeIn(kick_forward, RIGHT),
        )
        self.wait()
        self.play(FadeOut(layer_label))

        # Add other layers
        wave.start_clock()
        remaining_layers = layers[1:]
        self.play(
            ShowIncreasingSubsets(remaining_layers, rate_func=linear),
            UpdateFromFunc(
                kick_forward,
                lambda m: m.align_to(remaining_layers.get_right(), LEFT).shift(MED_SMALL_BUFF * RIGHT),
            ),
            LaggedStart(*(
                pkt.animate.set_value(self.kick_back_value)
                for pkt in pkts[1:]
            ), lag_ratio=1),
            run_time = len(remaining_layers) * self.time_per_layer
        )

        # Wait
        self.wait(20)


class DenserKickForward(KickForward):  # Run at 9
    layer_xs = np.arange(0, 8, 0.1)
    kick_back_value = 0.1
    line_style = dict(
        stroke_color=BLUE,
        stroke_width=1.5,
        stroke_opacity=0.75,
    )
    time_per_layer = 0.1


class DensestKickForward(DenserKickForward):
    layer_xs = np.arange(0, 8, 0.025)
    kick_back_value = 0.025
    time_per_layer = 0.025
    line_style = dict(
        stroke_color=BLUE,
        stroke_width=1.0,
        stroke_opacity=0.65,
    )
