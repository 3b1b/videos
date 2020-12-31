from big_ol_pile_of_manim_imports import *


class LogoGenerationFlurry(LogoGenerationTemplate):
    CONFIG = {
        "random_seed": 2,
    }

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
