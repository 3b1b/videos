from big_ol_pile_of_manim_imports import *


class LogoGeneration(LogoGenerationTemplate):
    def construct(self):
        logo = self.logo
        name = self.channel_name
    # def get_logo_animations(self, logo):
        layers = logo.spike_layers

        logo.save_state()

        for layer in layers:
            for spike in layer:
                spike.save_state()
                point = np.array(spike.points[0])
                angle = angle_of_vector(point)
                spike.rotate(-angle + 90 * DEGREES)
                spike.stretch_to_fit_width(0.2)
                spike.stretch_to_fit_height(0.5)
                spike.point = point
            for spike in layer[::2]:
                spike.rotate(180 * DEGREES)

                # spike.rotate(90 * DEGREES, about_point=ORIGIN)
                # spike.shift(point)
                # spike.make_smooth()
                # # spike.scale(0.5, about_point=point)
                # spike.set_fill(opacity=0.2)
                # # spike.scale(1.5)
                # spike.point = point
            layer.arrange(LEFT, buff=0.1)
            # random.shuffle(layer.submobjects)
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
        # logo.iris_background.scale(0.49)
        logo.iris_background.set_fill(opacity=0.0)
        logo.iris_background.scale(0.8)

        alt_name = name.copy()
        alt_name.set_stroke(BLACK, 5)

        self.play(*[
            Restore(
                logo.iris_background,
                rate_func=squish_rate_func(smooth, 1.0 / 3, 1),
                run_time=2,
            ),
            AnimationGroup(*[
                LaggedStartMap(
                    get_spike_animation, layer,
                    run_time=2,
                    # rate_func=squish_rate_func(smooth, a / 3.0, (a + 0.9) / 3.0),
                    lag_ratio=0.5,
                    path_arc=-90 * DEGREES
                )
                for layer, a in zip(layers, [0, 2, 1, 0])
            ]),
            Animation(logo.pupil),
            Write(alt_name),
            Write(name),
        ], run_time=2)

        self.wait(0.25)
        self.play(
            Transform(
                wrong_spike,
                wrong_spike.real_saved_state,
            ),
            Animation(self.logo),
            UpdateFromAlphaFunc(
                logo.pupil,
                lambda p, a: p.set_sheen(
                    0.1 * there_and_back(a),
                    UL + 2 * a * RIGHT
                )
            ),
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
                # spike.points[1] = rotate_vector(
                #     spike.points[1], TAU/28,
                # )
                # spike.points[-1] = rotate_vector(
                #     spike.points[-1], -TAU/28,
                # )


        def get_spike_animation(spike, **kwargs):
            return Restore(
                spike, path_arc=-spike.angle,
                **kwargs
            )

        logo.iris_background.save_state()
        # logo.iris_background.scale(0.49)
        logo.iris_background.set_fill(DARK_GREY, 0.5)

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