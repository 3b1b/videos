from manim_imports_ext import *


PERF_GRAY = "#BFBFC7"
PERF_BLUE = "#052E4D"
PERF_GREEN = "#38761D"
PERF_RED = "#CC0000"
PERF_ORANGE = "#E69138"
PERF_PINKISH = "#F03546"


class NeuralNetworkTest(InteractiveScene):
    def construct(self):
        # Set up a basic network
        frame = self.frame
        layer_sizes = [2, 4, 4, 2]
        layer_spacing = 2.75

        layers = VGroup(self.get_layer(size) for size in layer_sizes)
        layers.arrange(RIGHT, buff=layer_spacing)
        layers.set_height(6)

        connections = VGroup()
        for l1, l2 in zip(layers, layers[1:]):
            lines = VGroup()
            for n2 in l2:
                n2.connections = self.get_connections(n2, l1)
                lines.add(n2.connections)
            connections.add(lines)

        self.add(layers)
        self.add(connections)

        # Prepare our focus neuron
        neuron = layers[2][0]
        prev_connections = neuron.connections
        prev_connections[0].set_stroke(BLUE, 3)
        prev_connections[1].set_stroke(RED, 2)
        prev_connections[2].set_stroke(BLUE, 5)
        prev_connections[3].set_stroke(BLUE, 1)

        # Animate the connections and backrop
        self.animate_forward_pass(connections)
        self.wait()
        self.animate_back_prop(connections)
        self.wait()

        # Isolate
        neuron_copy = neuron.copy()
        prev_connections_copy = prev_connections.copy()
        prev_layer_copy = layers[1].copy()

        self.play(
            FadeIn(neuron_copy),
            FadeIn(prev_connections_copy),
            FadeIn(prev_layer_copy),
            layers.animate.set_fill(opacity=0.1).set_stroke(opacity=0.1),
            connections.animate.set_stroke(opacity=0.1),
        )

        # Show dentrite
        dentrite = self.get_dentrites(neuron_copy, prev_layer_copy)[0]
        dentrite.set_z_index(2)
        den_connections = self.get_connections(dentrite, prev_layer_copy)
        den_connections[0].set_stroke(RED, 2)
        den_connections[1].set_stroke(BLUE, 3)
        den_connections[2].set_stroke(RED, 3)
        den_connections[3].set_stroke(BLUE, 2)

        self.play(
            FadeIn(dentrite, 2 * DOWN)
        )
        self.play(
            ShowCreation(den_connections, lag_ratio=0.25),
            prev_connections_copy.animate.set_stroke(width=0.5),
        )
        self.wait()

        # Add connection between dentrite and neuron
        den_to_neuron_connection = self.get_den_connection(dentrite, neuron)
        self.play(ShowCreation(den_to_neuron_connection))
        self.wait()
        self.play(FadeOut(den_to_neuron_connection, shift=DOWN))

        # Show a forward pass
        self.play(
            layers.animate.set_fill(opacity=0.8).set_stroke(opacity=0.8),
            connections.animate.set_stroke(opacity=0.8),
            FadeOut(prev_connections_copy),
        )
        all_connections = VGroup(*connections, *den_connections)
        all_connections.sort(lambda p: p[0])
        self.animate_forward_pass(all_connections)

        # Show backpropagation
        self.animate_back_prop(
            connections,
            to_fade=[den_connections, den_to_neuron_connection],
        )

    def get_layer(self, layer_size, radius=0.3, fill_opacity=0.5, stroke_width=3, buff=1):
        layer = VGroup(
            Circle(radius=radius)
            for n in range(layer_size)
        )
        layer.set_fill(GREY, fill_opacity)
        layer.set_stroke(WHITE, stroke_width)
        layer.arrange(DOWN, buff=buff)

        return layer

    def get_connections(self, target, sources, max_stroke_width=4, colors=[BLUE, RED]):
        target.connections = VGroup()
        for neuron in sources:
            line = Line(
                neuron.get_center(),
                target.get_center(),
                buff=neuron.get_radius(),
                stroke_width=max_stroke_width * random.random(),
                stroke_color=random.choice(colors)
            )
            target.connections.add(line)
        return target.connections

    def get_den_connection(self, dentrite, neuron):
        den_connection = DashedLine(
            dentrite.get_center(),
            neuron.get_center(),
            buff=neuron.get_radius(),
            dash_length=0.1,
            positive_space_ratio=0.7,
        )
        den_connection.set_stroke(PERF_ORANGE, 6)
        return den_connection

    def get_dentrites(self, neuron, prev_layer, n_dentrites=1, fill_color=PERF_BLUE, stroke_color=PERF_GRAY, stroke_width=3, radius=0.3):
        dentrite = Circle(radius=radius)
        dentrite.set_fill(fill_color, 1)
        dentrite.set_stroke(stroke_color, stroke_width)

        dentrites = dentrite.replicate(n_dentrites)
        alphas = np.linspace(0, 1, n_dentrites + 2)[1:-1]
        for den, alpha in zip(dentrites, alphas):
            den.set_x(interpolate(prev_layer.get_x(), neuron.get_x(), alpha))
            den.set_y(neuron.get_y() + 1 * neuron.get_height())

        return dentrites

    def animate_computational_pass(self, connections, run_time=3, lag_ratio=0.05, to_fade=[], fade_amount=0.75, reverse=False):
        animated_connections = VGroup(line.copy() for line in connections.family_members_with_points())
        if reverse:
            animated_connections.sort(lambda p: -p[0])
            for line in animated_connections:
                line.rotate(PI)
        else:
            animated_connections.sort(lambda p: p[0])

        mobs_to_fade = VGroup(*to_fade)
        self.play(
            connections.animate.fade(fade_amount).set_anim_args(rate_func=lambda t: there_and_back_with_pause(t, 0.6)),
            mobs_to_fade.animate.fade(0.5).set_anim_args(rate_func=lambda t: there_and_back_with_pause(t, 0.6)),
            LaggedStartMap(ShowCreation, animated_connections, lag_ratio=lag_ratio),
            run_time=run_time
        )
        self.play(FadeOut(animated_connections))

    def animate_forward_pass(self, connections, **kwargs):
        self.animate_computational_pass(connections, **kwargs)

    def animate_back_prop(self, connections, to_fade=[], **kwargs):
        self.animate_computational_pass(connections, reverse=True, **kwargs)
