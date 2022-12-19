from manim_imports_ext import *

# Should this actually subclass Axes?


class HistogramBars(VGroup):
    def __init__(self, axes,
                 data=None,
                 x_range=None,
                 fill_color=BLUE,
                 fill_opacity=0.5,
                 stroke_color=WHITE,
                 stroke_width=1.0):
        if x_range is None:
            x_range = axes.x_range
        xs = np.arange(*x_range)
        self.x_to_index = dict(zip(xs, it.count()))
        x_step = xs[1] - xs[0]
        width = axes.x_axis.unit_size * x_step

        # Create a list of rectangles arranged side by side, one for each x value
        super().__init__(*(
            Rectangle(width=width, height=0).move_to(axes.c2p(x, 0), DL)
            for x in xs
        ))
        self.set_fill(fill_color, fill_opacity)
        self.set_stroke(stroke_color, stroke_width)

        if data is not None:
            self.set_data(data)

    def set_data(self, data):
        pass


# Scenes

class HistogramTest(InteractiveScene):
    def construct(self):
        # Test
        axes = Axes((0, 10), (0, 1, 0.25), width=12, height=4)
        hist = HistogramBars(axes, [5, 5, 3, 4, 2, 3, 1])

        axes.y_axis.add_numbers(num_decimal_places=2)
        axes.x_axis.add_numbers(num_decimal_places=0)

        graph = axes.get_graph(lambda x: 0.5 * math.sin(x) + 0.5)
        graph.set_stroke(YELLOW)

        self.add(axes)
        self.add(hist)

        self.play(ShowCreation(graph, run_time=2))
        self.wait()

        # Another part
        self.play(Transform(graph, axes.get_graph(lambda x: 0.2 * math.cos(x))))
        self.wait()
