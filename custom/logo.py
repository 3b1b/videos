from __future__ import annotations 

import numpy as np
import itertools as it
import random

from manimlib.constants import *
from manimlib.scene.scene import Scene
from manimlib.mobject.geometry import AnnularSector
from manimlib.mobject.geometry import Circle
from manimlib.mobject.geometry import Polygon
from manimlib.mobject.svg.text_mobject import Text
from manimlib.mobject.svg.tex_mobject import TexText
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.utils.bezier import interpolate
from manimlib.utils.space_ops import angle_of_vector
from manimlib.utils.rate_functions import squish_rate_func
from manimlib.utils.rate_functions import smooth
from manimlib.animation.animation import Animation
from manimlib.animation.transform import Restore
from manimlib.animation.transform import Transform
from manimlib.animation.composition import AnimationGroup
from manimlib.animation.composition import LaggedStartMap
from manimlib.animation.creation import Write


class Logo(VMobject):
    pupil_radius: float = 1.0
    outer_radius: float = 2.0
    iris_background_blue: ManimColor = "#74C0E3"
    iris_background_brown: ManimColor = "#8C6239"
    blue_spike_colors: list[ManimColor] = [
        "#528EA3",
        "#3E6576",
        "#224C5B",
        BLACK,
    ]
    brown_spike_colors: list[ManimColor] = [
        "#754C24",
        "#603813",
        "#42210b",
        BLACK,
    ]
    n_spike_layers: int = 4
    n_spikes: int = 28
    spike_angle: float = TAU / 28

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_iris_back()
        self.add_spikes()
        self.add_pupil()

    def add_iris_back(self):
        blue_iris_back = AnnularSector(
            inner_radius=self.pupil_radius,
            outer_radius=self.outer_radius,
            angle=270 * DEGREES,
            start_angle=180 * DEGREES,
            fill_color=self.iris_background_blue,
            fill_opacity=1,
            stroke_width=0,
        )
        brown_iris_back = AnnularSector(
            inner_radius=self.pupil_radius,
            outer_radius=self.outer_radius,
            angle=90 * DEGREES,
            start_angle=90 * DEGREES,
            fill_color=self.iris_background_brown,
            fill_opacity=1,
            stroke_width=0,
        )
        self.iris_background = VGroup(
            blue_iris_back,
            brown_iris_back,
        )
        self.add(self.iris_background)

    def add_spikes(self):
        layers = VGroup()
        radii = np.linspace(
            self.outer_radius,
            self.pupil_radius,
            self.n_spike_layers,
            endpoint=False,
        )
        radii[:2] = radii[1::-1]  # Swap first two
        if self.n_spike_layers > 2:
            radii[-1] = interpolate(radii[-1], self.pupil_radius, 0.25)

        for radius in radii:
            tip_angle = self.spike_angle
            half_base = radius * np.tan(tip_angle)
            triangle, right_half_triangle = [
                Polygon(
                    radius * UP,
                    half_base * RIGHT,
                    vertex3,
                    fill_opacity=1,
                    stroke_width=0,
                )
                for vertex3 in (half_base * LEFT, ORIGIN,)
            ]
            left_half_triangle = right_half_triangle.copy()
            left_half_triangle.flip(UP, about_point=ORIGIN)

            n_spikes = self.n_spikes
            full_spikes = [
                triangle.copy().rotate(
                    -angle,
                    about_point=ORIGIN
                )
                for angle in np.linspace(
                    0, TAU, n_spikes, endpoint=False
                )
            ]
            index = (3 * n_spikes) // 4
            if radius == radii[0]:
                layer = VGroup(*full_spikes)
                layer.rotate(
                    -TAU / n_spikes / 2,
                    about_point=ORIGIN
                )
                layer.brown_index = index
            else:
                half_spikes = [
                    right_half_triangle.copy(),
                    left_half_triangle.copy().rotate(
                        90 * DEGREES, about_point=ORIGIN,
                    ),
                    right_half_triangle.copy().rotate(
                        90 * DEGREES, about_point=ORIGIN,
                    ),
                    left_half_triangle.copy()
                ]
                layer = VGroup(*it.chain(
                    half_spikes[:1],
                    full_spikes[1:index],
                    half_spikes[1:3],
                    full_spikes[index + 1:],
                    half_spikes[3:],
                ))
                layer.brown_index = index + 1

            layers.add(layer)

        # Color spikes
        blues = self.blue_spike_colors
        browns = self.brown_spike_colors
        for layer, blue, brown in zip(layers, blues, browns):
            index = layer.brown_index
            layer[:index].set_color(blue)
            layer[index:].set_color(brown)

        self.spike_layers = layers
        self.add(layers)

    def add_pupil(self):
        self.pupil = Circle(
            radius=self.pupil_radius,
            fill_color=BLACK,
            fill_opacity=1,
            stroke_width=0,
            stroke_color=BLACK,
            sheen=0.0,
        )
        self.pupil.rotate(90 * DEGREES)
        self.add(self.pupil)

    def cut_pupil(self):
        pupil = self.pupil
        center = pupil.get_center()
        new_pupil = VGroup(*[
            pupil.copy().pointwise_become_partial(pupil, a, b)
            for (a, b) in [(0.25, 1), (0, 0.25)]
        ])
        for sector in new_pupil:
            sector.add_cubic_bezier_curve_to([
                sector.get_points()[-1],
                *[center] * 3,
                *[sector.get_points()[0]] * 2
            ])
        self.remove(pupil)
        self.add(new_pupil)
        self.pupil = new_pupil

    def get_blue_part_and_brown_part(self):
        if len(self.pupil) == 1:
            self.cut_pupil()
        blue_part = VGroup(
            self.iris_background[0],
            *[
                layer[:layer.brown_index]
                for layer in self.spike_layers
            ],
            self.pupil[0],
        )
        brown_part = VGroup(
            self.iris_background[1],
            *[
                layer[layer.brown_index:]
                for layer in self.spike_layers
            ],
            self.pupil[1],
        )
        return blue_part, brown_part


class LogoGenerationTemplate(Scene):
    def setup(self):
        super().setup()
        frame = self.camera.frame
        frame.shift(DOWN)

        self.logo = Logo()
        name = Text("3Blue1Brown")
        name.scale(2.5)
        name.next_to(self.logo, DOWN, buff=MED_LARGE_BUFF)
        name.set_gloss(0.2)
        self.channel_name = name

    def construct(self):
        logo = self.logo
        name = self.channel_name

        self.play(
            Write(name, run_time=3),
            *self.get_logo_animations(logo)
        )
        self.wait()

    def get_logo_animations(self, logo):
        return []  # For subclasses


class LogoGeneration(LogoGenerationTemplate):
    def construct(self):
        logo = self.logo
        name = self.channel_name

        layers = logo.spike_layers

        logo.save_state()

        for layer in layers:
            for spike in layer:
                spike.save_state()
                point = np.array(spike.get_points()[0])
                angle = angle_of_vector(point)
                spike.rotate(-angle + 90 * DEGREES)
                spike.stretch_to_fit_width(0.2)
                spike.stretch_to_fit_height(0.5)
                spike.point = point
            for spike in layer[::2]:
                spike.rotate(180 * DEGREES)
            layer.arrange(LEFT, buff=0.1)
        layers.arrange(UP)
        layers.to_edge(DOWN)

        wrong_spike = layers[1][-5]
        wrong_spike.real_saved_state = wrong_spike.saved_state.copy()
        wrong_spike.saved_state.scale(0.25, about_point=wrong_spike.point)
        wrong_spike.saved_state.rotate(90 * DEGREES)
        self.wrong_spike = wrong_spike

        def get_spike_animation(spike, **kwargs):
            return Restore(spike, **kwargs)

        logo.iris_background.save_state()
        logo.iris_background.set_fill(opacity=0.0)
        logo.iris_background.scale(0.8)

        alt_name = name.copy()
        alt_name.set_stroke(BLACK, 5)

        self.play(
            Restore(
                logo.iris_background,
                rate_func=squish_rate_func(smooth, 1.0 / 3, 1),
                run_time=2,
            ),
            AnimationGroup(*(
                LaggedStartMap(
                    get_spike_animation, layer,
                    run_time=2,
                    # rate_func=squish_rate_func(smooth, a / 3.0, (a + 0.9) / 3.0),
                    lag_ratio=2 / len(layer),
                    path_arc=-90 * DEGREES
                )
                for layer, a in zip(layers, [0, 2, 1, 0])
            )),
            Animation(logo.pupil),
            Write(alt_name),
            Write(name),
            run_time=3
        )

        self.wait(0.25)
        self.play(
            Transform(
                wrong_spike,
                wrong_spike.real_saved_state,
            ),
            Animation(self.logo),
            run_time=0.75
        )


class SortingLogoGeneration(LogoGenerationTemplate):
    def get_logo_animations(self, logo):
        layers = logo.spike_layers

        for j, layer in enumerate(layers):
            for i, spike in enumerate(layer):
                spike.angle = (13 * (i + 1) * (j + 1) * TAU / 28) % TAU
                if spike.angle > PI:
                    spike.angle -= TAU
                spike.save_state()
                spike.rotate(
                    spike.angle,
                    about_point=ORIGIN
                )
                # spike.get_points()[1] = rotate_vector(
                #     spike.get_points()[1], TAU/28,
                # )
                # spike.get_points()[-1] = rotate_vector(
                #     spike.get_points()[-1], -TAU/28,
                # )

        def get_spike_animation(spike, **kwargs):
            return Restore(
                spike, path_arc=-spike.angle,
                **kwargs
            )

        logo.iris_background.save_state()
        # logo.iris_background.scale(0.49)
        logo.iris_background.set_fill(GREY_D, 0.5)

        return [
            Restore(
                logo.iris_background,
                rate_func=squish_rate_func(smooth, 2.0 / 3, 1),
                run_time=3,
            ),
            AnimationGroup(*[
                LaggedStartMap(
                    get_spike_animation, layer,
                    run_time=2,
                    # rate_func=squish_rate_func(smooth, a / 3.0, (a + 0.9) / 3.0),
                    lag_ratio=0.2,
                )
                for layer, a in zip(layers, [0, 2, 1, 0])
            ]),
            Animation(logo.pupil),
        ]


class LogoTest(Scene):
    def construct(self):
        n_range = list(range(4, 40, 4))
        for n, denom in zip(n_range, np.linspace(14, 28, len(n_range))):
            logo = Logo(**{
                "iris_background_blue": "#78C0E3",
                "iris_background_brown": "#8C6239",
                "blue_spike_colors": [
                    "#528EA3",
                    "#3E6576",
                    "#224C5B",
                    BLACK,
                ],
                "brown_spike_colors": [
                    "#754C24",
                    "#603813",
                    "#42210b",
                    BLACK,
                ],
                "n_spike_layers": 4,
                "n_spikes": n,
                "spike_angle": TAU / denom,
            })
            self.add(logo)
            self.wait()
            self.clear()
        self.add(logo)


class LogoGenerationFlurry(LogoGenerationTemplate):
    random_seed: int = 2

    def get_logo_animations(self, logo):
        layers = logo.spike_layers
        for i, layer in enumerate(layers):
            random.shuffle(layer.submobjects)
            for spike in layer:
                spike.save_state()
                spike.scale(0.5)
                spike.apply_complex_function(np.log)
                spike.rotate(-90 * DEGREES, about_point=ORIGIN)
                spike.set_fill(opacity=0)
            layer.rotate(i * PI / 5)

        logo.iris_background.save_state()
        logo.iris_background.scale(0.25)
        logo.iris_background.fade(1)

        return [
            Restore(
                logo.iris_background,
                run_time=3,
            ),
            AnimationGroup(*[
                LaggedStartMap(
                    Restore, layer,
                    run_time=3,
                    path_arc=180 * DEGREES,
                    # rate_func=squish_rate_func(smooth, a / 3.0, (a + 1.9) / 3.0),
                    lag_ratio=0.8,
                )
                for layer, a in zip(layers, [0, 0.2, 0.1, 0])
            ]),
            Animation(logo.pupil),
        ]


class WrittenLogo(LogoGenerationTemplate):
    def get_logo_animations(self, logo):
        return [Write(logo, stroke_color=None, stroke_width=2, run_time=3, lag_ratio=5e-3)]



class LogoGenerationFivefold(LogoGenerationTemplate):
    def construct(self):
        logo = self.logo
        iris, spike_layers, pupil = logo

        name = OldTexText("3Blue1Brown")
        name.scale(2.5)
        name.next_to(logo, DOWN, buff=MED_LARGE_BUFF)
        name.set_gloss(0.2)

        self.add(iris)
        anims = []
        for layer in spike_layers:
            for n, spike in enumerate(layer):
                angle = (5 * (n + 1) * TAU / len(layer)) % TAU
                spike.rotate(angle, about_point=ORIGIN)
                anims.append(Rotate(
                    spike, -angle,
                    about_point=ORIGIN,
                    run_time=5,
                    # rate_func=rush_into,
                ))
                self.add(spike)
        self.add(pupil)

        def update(alpha):
            spike_layers.set_opacity(alpha)
            mid_alpha = 4.0 * (1.0 - alpha) * alpha
            spike_layers.set_stroke(WHITE, 1, opacity=mid_alpha)
            pupil.set_stroke(WHITE, 1, opacity=mid_alpha)
            iris.set_stroke(WHITE, 1, opacity=mid_alpha)

        name.flip().flip()
        self.play(
            *anims,
            VFadeIn(iris, run_time=3),
            UpdateFromAlphaFunc(
                Mobject(),
                lambda m, a: update(a),
                run_time=3,
            ),
            # FadeIn(name, run_time=3, lag_ratio=0.5, rate_func=linear),
            Write(name, run_time=3)
        )
        self.wait(2)


class Vertical3B1B(Scene):
    def construct(self):
        words = OldTexText(
            "3", "Blue", "1", "Brown",
        )
        words.scale(2)
        words[::2].scale(1.2)
        buff = 0.2
        words.arrange(
            DOWN,
            buff=buff,
            aligned_edge=LEFT,
        )
        words[0].match_x(words[1][0])
        words[2].match_x(words[3][0])
        self.add(words)

        logo = Logo()
        logo.next_to(words, LEFT)
        self.add(logo)

        VGroup(logo, words).center()
