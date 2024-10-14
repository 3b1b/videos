from manim_imports_ext import *
from _2024.transformers.helpers import *
from _2024.transformers.embedding import *
from _2024.transformers.generation import *


class DialTest(InteractiveScene):
    def construct(self):
        # Test
        dial = Dial(radius=0.5)
        self.add(dial)
        self.play(dial.animate_set_value(0.5, run_time=1))

        # Test
        machine = MachineWithDials()
        self.add(machine)
        self.play(machine.random_change_animation())


class MLWithinDeepL(InteractiveScene):
    def construct(self):
        # Organize boxes
        kw = dict(font_size=36, opacity=0.25)
        model_boxes = VGroup(
            self.get_titled_box("Multilayer Perceptrons", BLUE_D, **kw),
            self.get_titled_box("Convolutional Neural Networks", BLUE_D, **kw),
            self.get_titled_box("Transformers", BLUE, **kw),
        )
        for box in model_boxes:
            box.box.set_width(model_boxes.get_width(), stretch=True)
        dots = Tex(R"\vdots", font_size=72)
        model_boxes.add(dots)
        model_boxes.arrange(DOWN, buff=0.1)
        dots.shift(0.2 * DOWN)
        transformer_box = model_boxes[2]

        dl_box = self.get_titled_box(
            "Deep Learning", TEAL,
            font_size=60,
            y_space=model_boxes.get_height() + 1.0,
            x_space=2.75,
            opacity=0.05
        )

        model_boxes.next_to(dl_box.title, DOWN)

        # Animate in word
        transformer_box.save_state()
        transformer_box.box.set_opacity(0)
        transformer_box.set_height(1)
        transformer_box.move_to(np.array([-1.58, -2.01, 0]))

        self.add(transformer_box)
        self.wait()
        self.add(dl_box, transformer_box)
        self.play(LaggedStart(
            FadeIn(dl_box, scale=1.2),
            Restore(transformer_box),
            *(FadeIn(model_boxes[i]) for i in [0, 1, 3]),
        ), lag_ratio=0.75, run_time=2)
        self.wait()

        dl_box.add(model_boxes)
        self.add(dl_box)

        # Place within ML box
        ml_box = self.get_titled_box(
            "Machine Learning",
            GREEN,
            opacity=0.1,
            font_size=72,
            x_space=6.0,
            y_space=5.0
        )
        dl_box.target = dl_box.generate_target()
        blank_boxes = dl_box.box.replicate(2)
        inner_boxes = VGroup(*blank_boxes, dl_box.target)
        reg_drawing = self.get_regression_drawing()
        bayes_net = self.get_bayes_net_drawing()
        for drawing, box in zip([reg_drawing, bayes_net], blank_boxes):
            drawing.set_height(0.8 * box.get_height())
            drawing.move_to(box)
            box.add(drawing)
        inner_boxes.set_height(3.5)
        inner_boxes.arrange(RIGHT)
        inner_boxes.set_max_width(ml_box.get_width() - 0.5)
        inner_boxes.next_to(ml_box.title, DOWN, buff=1.0)

        self.add(ml_box, dl_box, blank_boxes)
        self.play(
            FadeIn(ml_box),
            MoveToTarget(dl_box),
            LaggedStartMap(FadeIn, blank_boxes, scale=2.0, lag_ratio=0.5)
        )
        self.wait()

        ml_box.add(dl_box, blank_boxes)

        # Learn from data
        words = Text("Learn from data", font_size=72)
        words.to_edge(UP, buff=MED_SMALL_BUFF)
        learn = words["Learn"][0]
        learn.save_state()
        learn.set_x(0)
        words["data"].set_color(YELLOW)
        ml_box.target = ml_box.generate_target()
        ml_box.target.scale(0.75)
        ml_box.target.to_edge(DOWN)
        arrow = Arrow(ml_box.target, words)

        self.play(
            MoveToTarget(ml_box),
            GrowFromCenter(arrow),
            TransformFromCopy(ml_box.title["Learn"][0], learn),
        )
        self.play(
            Restore(learn),
            FadeIn(words["from data"][0], lag_ratio=0.1, shift=0.2 * RIGHT),
        )
        self.wait()
        self.play(
            FadeOut(ml_box),
            FadeOut(arrow),
        )
        self.wait()

        # Go back to the box
        self.clear()
        ml_box.center()
        self.add(ml_box)

        # Pop out
        ml_box.remove(dl_box)
        ml_box.add(dl_box.copy())
        ml_box.target = ml_box.generate_target()
        ml_box.target.scale(0.25).to_edge(LEFT)
        dl_box.target = dl_box.generate_target()
        dl_box.target.scale(2.0)
        dl_box.target.next_to(ml_box.target, RIGHT, buff=0.75),
        lines = VGroup(*(
            Line(
                ml_box.target[-1].get_corner(RIGHT + v),
                dl_box.target.get_corner(LEFT + v)
            )
            for v in [UP, DOWN]
        ))
        lines.set_stroke(TEAL, 2)

        self.play(
            MoveToTarget(ml_box),
            MoveToTarget(dl_box),
            GrowFromPoint(lines[0], dl_box.get_corner(UR)),
            GrowFromPoint(lines[1], dl_box.get_corner(DR)),
            run_time=1.5,
        )
        self.wait()

        # Show a neural network
        network = NeuralNetwork([5, 10, 5])
        network.next_to(dl_box, RIGHT, buff=1.0)

        self.play(
            FadeIn(network.layers[0]),
            ShowCreation(network.lines[0], lag_ratio=0.01),
            FadeIn(network.layers[1], lag_ratio=0.5),
            run_time=2
        )
        self.play(
            ShowCreation(network.lines[1], lag_ratio=0.01),
            FadeIn(network.layers[2], lag_ratio=0.5),
            run_time=2
        )

        # Ambiently change the network
        for _ in range(6):
            self.play(
                network.animate.randomize_line_style().randomize_layer_values(),
                run_time=3,
                lag_ratio=1e-4
            )

        # Pile of matrices
        pile_words = Text("Pile of matrices")
        pile_words.next_to(network, UP)
        path_arc = -60 * DEGREES
        arrow = Arrow(dl_box.get_top(), pile_words.get_corner(UL), path_arc=path_arc)
        matrices = VGroup(*(
            WeightMatrix(shape=(8, 6), ellipses_row=None, ellipses_col=None)
            for x in range(10)
        ))
        matrices.match_width(network)
        matrices.move_to(network, UP)
        matrices.shift(0.5 * DOWN)
        matrix_shift = 0.5 * (IN + RIGHT)

        matrices.arrange(OUT, buff=0.25)
        matrices.move_to(network)

        for matrix in matrices[:-1]:
            matrix.target = matrix.generate_target()
            for entry in matrix.target.get_entries():
                dot = Dot(radius=0.05)
                dot.set_fill(entry.get_fill_color(), opacity=0.25)
                dot.move_to(entry)
                entry.become(dot)
            matrix.target[-1].set_opacity(0.25)
        matrices[-1].get_entries().set_backstroke(BLACK, 8)

        self.play(
            FadeOut(network, 2 * DOWN),
            ShowCreation(arrow),
            FadeInFromPoint(pile_words, dl_box.title.get_center(), path_arc=path_arc),
            FadeOut(network, DOWN)
        )
        mat_shift = 0.5 * IN + 0.25 * DOWN
        self.play(
            LaggedStart(*(
                Succession(
                    FadeIn(matrix, shift=mat_shift),
                    MoveToTarget(matrix)
                )
                for matrix in matrices[:-1]
            ), lag_ratio=0.25, run_time=5),
            Animation(Point()),
            FadeIn(matrices[-1], shift=mat_shift, time_span=(3.75, 4.75))
        )
        self.wait()

    def get_titled_box(self, text, color, font_size=48, y_space=0.5, x_space=0.5, opacity=0.1):
        title = Text(text, font_size=font_size)
        box = Rectangle(
            title.get_width() + x_space,
            title.get_height() + y_space
        )
        box.set_fill(interpolate_color(BLACK, color, opacity), 1)
        box.set_stroke(color, 2)
        title.next_to(box.get_top(), DOWN, buff=MED_SMALL_BUFF)
        result = VGroup(box, title)
        result.box = box
        result.title = title
        return result

    def get_regression_drawing(self):
        axes = Axes((-1, 10), (-1, 10))
        m = 0.5
        y0 = 2
        line = axes.get_graph(lambda x: y0 + m * x)
        line.set_stroke(YELLOW, 2)
        dots = VGroup(
            Dot(axes.c2p(x, y0 + m * x + np.random.normal()))
            for x in np.random.uniform(0, 10, 15)
        )

        reg_drawing = VGroup(axes, dots, line)
        return reg_drawing

    def get_bayes_net_drawing(self):
        radius = MED_SMALL_BUFF
        node = Circle(radius=radius)
        node.set_stroke(GREY_B, 2)
        node.shift(2 * DOWN)
        nodes = VGroup(
            node.copy().shift(x * RIGHT + y * UP)
            for x, y in [
                (-1, 0),  
                (1, 0),
                (-2, 2),
                (0, 2),
                (2, 2),
                (-2, 4),
                (0, 4),
            ]
        )
        edge_index_pairs = [
            (2, 0),
            (3, 0),
            (3, 1),
            (4, 1),
            (5, 2),
            (6, 3),
        ]
        edges = VGroup()
        for i1, i2 in edge_index_pairs:
            n1, n2 = nodes[i1], nodes[i2]
            edge = Arrow(
                n1.get_center(), 
                n2.get_center(),
                buff=radius,
                color=WHITE,
                stroke_width=3
            )
            edges.add(edge)

        network = VGroup(nodes, edges)
        return network


class ShowCross(InteractiveScene):
    def construct(self):
        # Test
        cross = Cross(Square(side_length=5))
        cross.set_stroke(width=[0, 30, 0])
        self.play(ShowCreation(cross))
        self.wait()


class FlashThroughImageData(InteractiveScene):
    time_per_example = 0.1

    def construct(self):
        # Images
        image_data = load_image_net_data()
        arrow = Vector(RIGHT)

        for path, text in ProgressDisplay(image_data):
            image = ImageMobject(str(path))
            label = Text(text.split(",")[0])
            label.use_winding_fill(False)
            image.next_to(arrow, LEFT)
            label.next_to(arrow, RIGHT)
            self.add(image, arrow, label)
            self.wait(self.time_per_example)
            self.remove(image, label)

            if hasattr(image, "shader_wrapper"):
                for tid in image.shader_wrapper.texture_names_to_ids.values():
                    release_texture(tid)


class FlashThroughTextData2(InteractiveScene):
    n_examples = 200
    time_per_example = 0.1
    window_size = 50
    line_len = 35
    ul_point = 5 * LEFT + 3 * UP

    def construct(self):
        # Test
        totc = read_in_book(name="tale_of_two_cities")
        words = re.split(r"\s", totc)
        words = list(filter(lambda s: s, words))

        for n in range(self.n_examples):
            index = random.randint(0, len(words) - self.window_size)
            window = words[index:index + self.window_size]
            phrase = get_paragraph(window, line_len=self.line_len)
            phrase.move_to(self.ul_point, UL)

            word = phrase[window[-1]][-1]
            rect = SurroundingRectangle(word, buff=0.1)
            rect.set_stroke(YELLOW, 2)
            rect.set_fill(YELLOW, 0.5)

            self.add(phrase)
            self.wait(self.time_per_example)
            self.remove(phrase)


class TweakedMachine(InteractiveScene):
    n_tweaks = 200
    time_per_example = 0.1

    def construct(self):
        # Test
        machine = MachineWithDials(
            dial_config=dict(
                value_to_color_config=dict(
                    low_negative_color=BLUE_E,
                    high_negative_color=BLUE_B,
                )
            )
        )
        machine.move_to(2 * DOWN)
        machine.set_width(4)
        arrow = Vector(DOWN, stroke_width=10)
        arrow.next_to(machine, UP)

        self.add(machine, arrow)

        values = np.array([d.get_random_value() for d in machine.dials])

        for n in range(self.n_tweaks):
            nudges = np.random.uniform(-1, 1, values.shape)
            values += 0.1 * nudges
            values[values > 1.0] = 0.9
            values[values < 0.0] = 0.1
            for dial, value in zip(machine.dials, values):
                dial.set_value(value)
            self.wait(self.time_per_example)


class PremiseOfML(InteractiveScene):
    box_center = RIGHT
    n_examples = 50
    random_seed = 316
    show_matrices = False

    def construct(self):
        self.init_data()

        # Set up input and output
        machine = self.get_machine()
        machine.set_width(4)
        machine.move_to(self.box_center)
        model_label = Text("Model", font_size=72)
        model_label.move_to(machine.box)
        in_arrow = Vector(RIGHT).next_to(machine, LEFT)
        out_arrow = Vector(RIGHT).next_to(machine, RIGHT)

        self.add(machine.box)
        self.add(in_arrow, out_arrow)
        self.add(model_label)

        # Show initial input and output
        in_data, out_data = self.new_input_output_example(in_arrow, out_arrow)

        in_word, out_word = [
            Text(word).next_to(machine, UP).match_x(mob).shift_onto_screen()
            for word, mob in [("Input", in_data), ("Output", out_data)]
        ]

        self.play(
            FadeIn(in_data, lag_ratio=0.001),
            FadeIn(in_word, 0.5 * UP),
        )
        self.play(FadeOutToPoint(in_data.copy(), machine.get_left(), lag_ratio=0.005, path_arc=-60 * DEGREES))
        self.play(
            FadeInFromPoint(out_data, machine.get_right(), lag_ratio=0.1, path_arc=60 * DEGREES),
            FadeIn(out_word, 0.5 * UP)
        )
        self.wait()

        # Show code
        model_label.target = model_label.generate_target()
        model_label.target.scale(in_word[0].get_height() / model_label[0].get_height())
        model_label.target.align_to(in_word, UP)
        code = self.get_code()
        code.set_height(machine.get_height() - MED_SMALL_BUFF)
        code.set_max_width(machine.get_width() - MED_SMALL_BUFF)
        code.move_to(machine, UP).shift(SMALL_BUFF * DOWN)

        self.play(
            MoveToTarget(model_label),
            ShowIncreasingSubsets(code, run_time=3),
        )
        self.wait()

        # Show tunable parameters
        param_label = Text("Tunable parameters")
        param_label.next_to(machine, UP)
        param_label.set_color(BLUE)

        self.play(
            FadeOut(code, 0.25 * DOWN, lag_ratio=0.01),
            Write(machine.dials, lag_ratio=0.001),
            FadeOut(model_label, 0.5 * UP),
            FadeIn(param_label, 0.5 * UP),
        )
        self.play(machine.rotate_all_dials())
        self.wait()

        # Show lots of new data
        for n in range(self.n_examples):
            new_in_data, new_out_data = self.new_input_output_example(in_arrow, out_arrow)
            self.add(in_data, out_data)
            time_span = (0, 0.35)
            self.play(
                machine.random_change_animation(run_time=0.5),
                FadeOut(in_data, time_span=time_span),
                FadeOut(out_data, time_span=time_span),
                FadeIn(new_in_data, time_span=time_span),
                FadeIn(new_out_data, time_span=time_span),
            )
            in_data, out_data = new_in_data, new_out_data

        if not self.show_matrices:
            return

        # Make room
        up_shift = 1.5 * UP
        down_shift = 1.75 * DOWN

        down_group = Group(in_arrow, machine, param_label, out_arrow, out_data, out_word)
        self.play(
            in_data.animate.scale(0.75).shift(up_shift + 0.5 * UP),
            UpdateFromFunc(out_data, lambda m: m.match_y(in_data)),
            in_word.animate.shift(up_shift),
            down_group.animate.shift(down_shift),
        )

        # Create pixels
        image = in_data
        pixels = create_pixels(in_data)

        # Show input array
        in_array = NumericEmbedding(shape=(10, 10), ellipses_col=-2)
        in_array.match_height(machine)
        in_array.next_to(in_arrow, LEFT)
        image.set_opacity(0.8)

        self.play(
            TransformFromCopy(
                pixels,
                VGroup(*(in_array.get_entries().family_members_with_points())),
                run_time=2,
                lag_ratio=1e-3
            ),
            FadeInFromPoint(in_array.get_brackets(), image.get_bottom()),
            Write(in_array.get_ellipses(), time_span=(1, 2))
        )
        self.play(image.animate.set_opacity(1))
        self.wait()

        # Show one dimensional array
        vector = NumericEmbedding(length=10)
        vector.replace(in_array, dim_to_match=1)
        vector.move_to(in_array, RIGHT)

        self.remove(in_array)
        self.play(
            TransformFromCopy(in_array.get_brackets(), vector.get_brackets()),
            TransformFromCopy(in_array.get_columns()[5], vector.get_columns()[0]),
            *map(FadeOut, in_array.get_columns()),
        )
        self.wait()
        self.remove(vector)
        self.play(LaggedStart(
            TransformFromCopy(vector.get_brackets(), in_array.get_brackets()),
            TransformFromCopy(vector.get_columns()[0], in_array.get_columns()[5]),
            *(
                FadeIn(col, shift=col.get_center() - vector.get_center())
                for col in in_array.get_columns()
            )
        ))
        self.wait()

        # Show 3d tensor
        self.frame.set_field_of_view(30 * DEGREES)
        dot_array = in_array.copy()
        for entry in (*dot_array.get_entries(), *dot_array.get_ellipses()):
            dot = Dot(entry.get_center(), radius=0.06)
            entry.set_submobjects([dot])

        tensor = VGroup(*(
            dot_array.copy()
            for n in range(5)
        ))
        for layer in tensor:
            for dot in (*layer.get_entries(), *layer.get_ellipses()):
                dot.set_fill(
                    interpolate_color(GREY_C, GREY_B, random.random()),
                    opacity=0.5,
                )
                dot.set_backstroke(BLACK, 2)
        tensor.arrange(OUT, buff=0.25)
        tensor.move_to(in_array, RIGHT)
        tensor.rotate(5 * DEGREES, RIGHT)
        tensor.rotate(5 * DEGREES, UP)

        self.remove(in_array)
        self.play(TransformFromCopy(VGroup(in_array), tensor))
        self.play(Rotate(tensor, 20 * DEGREES, axis=UP, run_time=4))
        self.play(Transform(tensor, VGroup(in_array), remover=True))
        self.add(in_array)

        # Express output as an array of numbers
        values = np.random.uniform(0, 1, (10, 1))
        values[5] = 9.7
        out_array = DecimalMatrix(values, ellipses_row=-2)
        out_array.match_height(machine)
        out_array.match_y(out_arrow)
        out_array.match_x(out_word)

        self.play(
            FadeInFromPoint(out_array, machine.get_right(), lag_ratio=1e-3),
            out_data.animate.scale(0.75).fade(0.5).rotate(-PI / 2).next_to(out_array, RIGHT, buff=0.25),
        )
        self.wait()

        # Describe parameters as weights
        weights_label = Text("Weights")
        weights_label.next_to(machine, UP, buff=0.5)
        weights_label.match_color(param_label)
        equiv = Tex(R"\Updownarrow")
        equiv.next_to(weights_label, UP)

        top_dials = machine.dials[:8]
        dial_rects = VGroup(*map(SurroundingRectangle, top_dials))
        dial_rects.set_stroke(TEAL, 2)
        dial_arrows = VGroup(*(
            Arrow(weights_label.get_bottom(), rect.get_top(), buff=0.05)
            for rect in dial_rects
        ))
        dial_arrows.set_stroke(TEAL)

        self.play(
            FadeIn(weights_label, scale=2),
            param_label.animate.next_to(equiv, UP),
            Write(equiv),
        )
        self.play(
            LaggedStart(*(
                VFadeInThenOut(VGroup(arrow, rect))
                for arrow, rect in zip(dial_arrows, dial_rects)
            ), lag_ratio=0.25, run_time=3)
        )
        self.wait()

        # Show weighted sum
        machine.dials.save_state()
        weights_label.set_backstroke(BLACK, 5)
        weights_label.target = weights_label.generate_target()
        weights_label.target.next_to(top_dials, DOWN, buff=0.25)
        weighted_sum = Tex(
            R"w_1 x_1 + w_2 x_2 + w_3 x_3 + \cdots + w_n x_n",
            font_size=42,
        )
        weighted_sum.next_to(machine, UP, buff=1.0)
        weight_parts = weighted_sum[re.compile(r"w_\d|w_n")]
        weight_parts.set_color(BLUE)
        data_parts = weighted_sum[re.compile(r"x_\d|x_n")]
        data_parts.set_color(GREY_A)

        indices = [0, 1, 2, -1]
        dial_lines = VGroup(*(
            Line(top_dials[n].get_top(), weight_parts[n].get_bottom(), buff=0.1)
            for n in indices
        ))
        ellipses = weighted_sum[R"\cdots"]
        dial_lines.set_stroke(BLUE_B, 1)

        column = in_array.get_columns()[-1]
        col_rect = SurroundingRectangle(column)
        col_rect.set_stroke(YELLOW, 2)

        self.play(ShowCreation(col_rect))
        self.play(
            FadeOut(VGroup(param_label, equiv), UP),
            MoveToTarget(weights_label),
            machine.dials[8:].animate.fade(0.75),
            LaggedStart(*(
                TransformFromCopy(column[n], data_parts[n])
                for n in indices
            )),
            Group(in_data, in_word).animate.to_edge(LEFT, buff=0.25)
        )
        self.play(
            Write(weighted_sum["+"]),
            Write(weighted_sum[R"\cdots"]),
            LaggedStart(*(
                FadeTransform(top_dials[n].copy(), weight_parts[n])
                for n in indices
            )),
            LaggedStartMap(ShowCreation, dial_lines),
            run_time=1
        )
        self.wait()
        for x in range(3):
            self.play(*(
                dial.animate_set_value(dial.get_random_value())
                for dial in top_dials
            ))

        # Wrap a function around it
        func_wrapper = Tex(R"f()")
        func_wrapper[:2].next_to(weighted_sum, LEFT, buff=SMALL_BUFF)
        func_wrapper[2].next_to(weighted_sum, RIGHT, buff=SMALL_BUFF)
        func_wrapper.set_color(PINK)

        nl_words = Text("Simple nonlinear\nfunction", font_size=42, alignment="LEFT")
        nl_words.next_to(func_wrapper, UP, buff=1.5, aligned_edge=LEFT)
        nl_words.match_color(func_wrapper)
        nl_arrow = Arrow(nl_words, func_wrapper[0].get_top())
        nl_arrow.match_color(nl_words)

        self.play(
            FadeIn(func_wrapper),
            FadeIn(nl_words, lag_ratio=0.1),
            ShowCreation(nl_arrow),
        )
        self.wait()

        # Show next layer
        weights_label.target = weights_label.generate_target()
        weights_label.target.next_to(weighted_sum, UP, buff=1.0)
        dial_lines.target = VGroup(*(
            Line(
                weights_label.target, weight_parts[index].get_top(),
                buff=SMALL_BUFF
            )
            for index in indices
        ))
        dial_lines.target.match_style(dial_lines)

        layer1 = NumericEmbedding(shape=(10, 5), ellipses_col=-2)
        layer1.match_height(in_array)
        layer1.next_to(in_arrow, RIGHT)
        mid_arrow = in_arrow.copy()
        mid_arrow.next_to(layer1, RIGHT)
        dots = Tex(R"\dots").next_to(mid_arrow, RIGHT)

        expr_rect = SurroundingRectangle(func_wrapper)
        expr_rect.set_stroke(PINK, 2)
        x01_rect = SurroundingRectangle(layer1.elements[0])
        x01_rect.match_style(expr_rect)
        rect_lines = VGroup(*(
            Line(expr_rect.get_corner(DOWN + v), x01_rect.get_corner(UP + v))
            for v in [LEFT, RIGHT]
        ))
        rect_lines.match_style(expr_rect)

        self.play(LaggedStart(
            FadeOut(weights_label),
            FadeOut(dial_lines),
            FadeOut(nl_words),
            FadeOut(nl_arrow),
            FadeOut(col_rect),
            FadeOut(machine),
            FadeIn(expr_rect),
        ))
        self.play(
            TransformFromCopy(in_array.get_brackets(), layer1.get_brackets()),
            TransformFromCopy(in_arrow, mid_arrow),
            out_arrow.animate.next_to(dots, RIGHT),
            Write(dots),
        )
        self.play(
            TransformFromCopy(expr_rect, x01_rect),
            ShowCreation(rect_lines, lag_ratio=0),
            FadeInFromPoint(layer1.elements[0], expr_rect.get_center()),
        )
        self.play(ShowIncreasingSubsets(layer1[1:-1]))
        self.add(layer1)
        self.wait()

        # Highlight a subset of the data
        in_subset = VGroup(*(
            elem
            for row in in_array.get_rows()[:3]
            for elem in row[:3]
        ))
        in_subset_rects = VGroup(*map(SurroundingRectangle, in_subset))
        data_part_rects = VGroup(*map(SurroundingRectangle, data_parts))
        self.play(
            LaggedStartMap(ShowCreationThenFadeOut, in_subset_rects, lag_ratio=0.02),
            LaggedStartMap(ShowCreationThenFadeOut, data_part_rects, lag_ratio=0.04),
            run_time=3
        )
        self.wait()

        # Show added layers
        to_fade = VGroup(
            func_wrapper, expr_rect, rect_lines, x01_rect,
            weighted_sum
        )

        self.play(
            LaggedStartMap(FadeOut, to_fade, run_time=1),
            in_arrow.animate.scale(0.5, about_edge=LEFT),
            layer1.animate.rotate(70 * DEGREES, UP).next_to(in_arrow, RIGHT, buff=-0.25),
            mid_arrow.animate.scale(0.5).next_to(in_arrow, RIGHT, buff=0.75),
        )

        layer1_group = VGroup(layer1, mid_arrow)
        layer2_group, layer3_group = layer1_group.replicate(2)
        layer2_group.next_to(layer1_group, RIGHT, buff=SMALL_BUFF)
        layer3_group.next_to(layer2_group, RIGHT, buff=SMALL_BUFF)
        self.play(TransformFromCopy(layer1_group, layer2_group))
        self.play(
            TransformFromCopy(layer2_group, layer3_group),
            VGroup(dots, out_arrow).animate.next_to(layer3_group, RIGHT),
        )
        self.play(
            LaggedStart(*(
                dot.animate.shift(0.1 * UP).set_anim_args(rate_func=there_and_back)
                for dot in dots
            ), lag_ratio=0.25)
        )
        self.wait()

        # Bring back machine
        layers = VGroup(layer1_group, layer2_group, layer3_group, dots)

        self.play(
            FadeIn(machine, scale=0.8),
            FadeIn(weights_label, shift=DOWN),
            ShowCreation(dial_lines, lag_ratio=0.1),
            FadeIn(weighted_sum, shift=UP),
            FadeOut(layers, scale=0.8),
        )
        self.wait()
        self.play(
            machine.random_change_animation()
        )
        self.wait()

        # Show a matrix
        frame = self.frame
        matrix, vector, equals, rhs = get_full_matrix_vector_product()
        mat_prod_group = VGroup(matrix, vector, equals, rhs)
        mat_prod_group.next_to(machine, UP, buff=2.0)
        mat_prod_group.shift(0.5 * LEFT)

        p0 = machine.get_corner(UL)
        p1 = matrix.get_corner(DL)
        p2 = machine.get_corner(UR)
        p3 = rhs.get_corner(DR)
        brace = VGroup(
            CubicBezier(p0, p0 + 2 * UP, p1 + 2 * DOWN, p1 + 0.1 * DOWN),
            CubicBezier(p2, p2 + 2 * UP, p3 + 2 * DOWN, p3 + 0.1 * DOWN),
        )
        brace.set_stroke(WHITE, 5)

        self.play(LaggedStart(
            TransformFromCopy(data_parts, vector.get_columns()[0]),
            TransformFromCopy(weight_parts, matrix.get_rows()[0]),
            FadeTransform(weighted_sum, rhs.get_rows()[0]),
            frame.animate.set_height(10, about_edge=DOWN),
            FadeOut(in_data, DOWN),
            FadeOut(out_data, DOWN),
            in_word.animate.next_to(in_array, UP),
            FadeIn(matrix, lag_ratio=0.1),
            ShowCreation(brace, lag_ratio=0),
            weights_label.animate.set_height(0.5).next_to(matrix, UP, buff=MED_SMALL_BUFF),
            Uncreate(dial_lines, lag_ratio=0.1),
            FadeOut(col_rect),
            machine.dials.animate.restore(),
            FadeIn(vector.get_brackets()),
            FadeIn(rhs.get_brackets()),
            FadeIn(equals),
            run_time=3,
            lag_ratio=0.1,
        ))
        self.wait()

        # Animate matrix vector product
        ghost_row = rhs.get_rows()[0].copy()
        ghost_row.set_opacity(0.25)
        self.add(ghost_row)
        show_symbolic_matrix_vector_product(
            self, matrix, vector, rhs,
            run_time_per_row=1.5
        )
        self.remove(ghost_row)
        self.wait()

        # Associate weights with dials
        w_elems = matrix.get_entries()
        moving_dials = machine.dials[:len(w_elems)].copy()
        moving_dials.target = moving_dials.generate_target()
        for dial, w_elem in zip(moving_dials.target, w_elems):
            dial.move_to(w_elem)
            dial.scale(2)

        self.play(
            w_elems.animate.set_opacity(0.25),
            MoveToTarget(moving_dials, run_time=2),
        )
        self.play(
            LaggedStart(*(
                dial.animate_set_value(dial.get_random_value())
                for dial in moving_dials
            ), lag_ratio=0.02, run_time=3)
        )
        self.wait()
        self.play(
            FadeOut(moving_dials),
            w_elems.animate.set_opacity(1),
        )

        # Vector an data slice
        v_rect = SurroundingRectangle(vector.get_entries())
        self.play(
            ShowCreation(v_rect),
            ShowCreation(col_rect),
        )
        self.wait()
        self.play(
            FadeOut(v_rect),
            FadeOut(col_rect),
        )
        self.wait()

        # Show many matrices
        lhs = VGroup(matrix, vector)
        small_mat_product = Tex(R"W_{10} v_{11}")
        small_mat_product[R"W_{10}"].set_color(BLUE)
        w_index = small_mat_product.make_number_changeable("10")
        v_index = small_mat_product.make_number_changeable("11")
        small_mat_products = VGroup()
        n_rows, n_cols = 16, 8
        for n in range(n_rows * n_cols):
            w_index.set_value(n + 1)
            v_index.set_value(n + 1)
            new_prod = small_mat_product.copy()
            new_prod.arrange(RIGHT, buff=SMALL_BUFF, aligned_edge=DOWN)
            small_mat_products.add(new_prod)
        small_mat_products.arrange_in_grid(n_rows, n_cols, v_buff_ratio=2.0)
        small_mat_products.replace(machine.dials)

        mv_label = Text("matrix-vector products")
        mv_label.next_to(machine, UP, buff=1.0)
        mv_label[-1].set_opacity(0)
        mv_top_label = Text("Many, many")
        mv_top_label.next_to(mv_label, UP)
        mv_arrows = VGroup(*(
            Arrow(mv_label.get_bottom(), smp.get_top(), buff=0.1)
            for smp in small_mat_products
        ))

        self.play(
            FadeTransform(mat_prod_group, small_mat_products[0]),
            Uncreate(brace, lag_ratio=0),
            FadeOut(machine.dials, run_time=0.5),
            FadeTransform(weights_label, mv_label),
            GrowFromPoint(mv_arrows[0], weights_label.get_bottom()),
            frame.animate.set_height(FRAME_HEIGHT).move_to(DOWN).set_anim_args(time_span=(1, 2)),
            run_time=2,
        )
        self.wait()
        self.remove(mv_arrows)
        self.play(
            FadeIn(mv_top_label, UP),
            mv_label[-1].animate.set_opacity(1),
            ShowIncreasingSubsets(small_mat_products, rate_func=linear, run_time=12, int_func=np.ceil),
            ShowSubmobjectsOneByOne(mv_arrows, rate_func=linear, run_time=12, int_func=np.ceil),
        )
        self.remove(mv_arrows)
        self.play(FadeOut(mv_arrows[-1]))
        self.wait()

    def init_data(self):
        self.image_data = load_image_net_data()

    def new_input_output_example(self, in_arrow, out_arrow) -> tuple[Mobject, Mobject]:
        path, label_text = random.choice(self.image_data)
        image = ImageMobject(str(path))
        image.set_width(4)
        image.next_to(in_arrow, LEFT)
        label = Text(label_text.split(",")[0])
        label.set_max_width(2.5)
        label.next_to(out_arrow, RIGHT)
        return image, label

    def get_machine(self):
        return MachineWithDials()

    def get_code(self):
        # Test
        src = """
            #include <opencv2/opencv.hpp>
            #include <iostream>

            using namespace cv;
            using namespace std;

            int main(int argc, char** argv) {
                Mat image = imread(argv[1], IMREAD_GRAYSCALE);
                if (image.empty()) {
                    cout << "Could not open image" << endl;
                    return -1;
                }

                // Blur the image to reduce noise
                Mat blurredImage;
                GaussianBlur(image, blurredImage, Size(5, 5), 0);

                // Detect edges with Canny
                Mat edges;
                Canny(blurredImage, edges, 100, 200);
        """
        return Code(src, language="C++", alignment="LEFT")


class PremiseOfMLWithText(PremiseOfML):
    random_seed = 316

    def init_data(self):
        totc = read_in_book(name="tale_of_two_cities")
        words = re.split(r"\s", totc)
        words = list(filter(lambda s: s, words))
        self.all_words = words

    def new_input_output_example(self, in_arrow, out_arrow):
        words = self.all_words
        window_size = 25
        index = random.randint(0, len(words) - window_size)
        window = words[index:index + window_size]
        in_text = get_paragraph(window[:-1], line_len=25)
        in_text.set_max_width(4)
        in_text.next_to(in_arrow, LEFT)
        out_text = Text(window[-1])
        out_text.next_to(out_arrow, RIGHT)
        return in_text, out_text

    def get_machine(self):
        machine = super().get_machine()
        machine.add(VectorizedPoint().next_to(machine, DOWN, buff=0.5))
        return machine

    def get_code(self):
        # Test
        src = """
            using namespace std;

            vector<string> findCapitalizedWords(const string& text) {
                vector<string> capitalizedWords;
                stringstream ss(text);
                string word;

                while (ss >> word) {
                    // Check for uppercase
                    if (!word.empty() && isupper(word[0])) {
                        capitalizedWords.push_back(word);
                    }
                }

                return capitalizedWords;
            }

            int main() {
                string text;
                cout << "Enter text: ";
                getline(cin, text); // Using getline to read spaces
        """
        return Code(src, language="C++", alignment="LEFT")


class PremiseOfMLWithMatrices(PremiseOfML):
    # Skip to animation 9
    show_matrices = True
    n_examples = 0
    random_seed = 6


class LinearRegression(InteractiveScene):
    radom_seed = 1

    def construct(self):
        # Set up axes
        x_min, x_max = (-1, 12)
        y_min, y_max = (-1, 10)
        axes = Axes((x_min, x_max), (y_min, y_max), width=12, height=6)
        axes.to_edge(DOWN)
        self.add(axes)

        # Add data
        n_data_points = 30
        m = 0.75
        y0 = 1

        data = np.array([
            (x, y0 + m * x + 0.75 * np.random.normal(0, 1))
            for x in np.random.uniform(2, x_max, n_data_points)
        ])
        points = axes.c2p(data[:, 0], data[:, 1])
        dots = DotCloud(points)

        dots.set_color(YELLOW)
        dots.set_glow_factor(1)
        dots.set_radius(0.075)

        self.add(dots)

        # Make title
        title = Text("Linear Regression", font_size=72)
        title.to_edge(UP)

        # Show line
        m_tracker = ValueTracker(m)
        y0_tracker = ValueTracker(y0)
        line = Line()
        line.set_stroke(TEAL, 2)

        def update_line(line):
            curr_y0 = y0_tracker.get_value()
            curr_m = m_tracker.get_value()
            line.put_start_and_end_on(
                axes.c2p(0, curr_y0),
                axes.c2p(x_max, curr_y0 + curr_m * x_max),
            )

        line.add_updater(update_line)

        self.play(
            FadeIn(title, UP),
            ShowCreation(line),
        )
        self.wait()

        # Label inputs and outputs
        in_labels = VGroup(Text("Input"), Text("Square footage"))
        out_labels = VGroup(Text("Output"), Text("Price"))
        for in_label in in_labels:
            in_label.next_to(axes.x_axis, DOWN, buff=0.1, aligned_edge=RIGHT)
        for out_label in out_labels:
            out_label.rotate(90 * DEGREES)
            out_label.next_to(axes.y_axis, LEFT, aligned_edge=UP)

        self.play(LaggedStart(
            FadeIn(in_labels[0], lag_ratio=0.1),
            FadeIn(out_labels[0], lag_ratio=0.1),
            lag_ratio=0.5,
        ))
        self.wait()
        self.play(LaggedStart(
            FadeTransform(*in_labels),
            FadeTransform(*out_labels),
            lag_ratio=0.8,
        ))
        self.wait()

        # Emphasize line
        self.play(
            VShowPassingFlash(
                line.copy().set_stroke(BLUE, 8).scale(1.1).insert_n_curves(100),
                time_width=1.5,
                run_time=2
            ),
        )
        self.wait()

        # Add line parameter updaters
        words = ["slope", "y-intercept"]
        value_ranges = [(0, 2, 0.2), (-2, 3, 0.5)]
        m_label, y0_label = labels = VGroup(
            VGroup(
                Dial(value_range=value_range),
                Text(f"{text} = "),
                DecimalNumber(),
            )
            for text, value_range in zip(words, value_ranges)
        )
        for label, tracker in zip(labels, [m_tracker, y0_tracker]):
            label[0].set_height(2 * label[2].get_height())
            label.arrange(RIGHT)
            label[0].f_always.set_value(tracker.get_value)
            label[2].f_always.set_value(tracker.get_value)
        labels.arrange(DOWN, aligned_edge=LEFT)
        labels.next_to(axes.y_axis, RIGHT, buff=1.0)
        labels.to_edge(UP)

        self.play(
            FadeOut(title, UP),
            FadeIn(m_label, UP),
        )
        self.play(
            m_tracker.animate.set_value(1.5),
            run_time=2,
        )
        self.play(FadeIn(y0_label, UP))
        self.play(
            y0_tracker.animate.set_value(-2),
            run_time=2
        )
        self.wait()

        # Tweak line parameters
        for n in range(10):
            alpha = random.random()
            if alpha > 0.5:
                alpha += 1
            new_m = interpolate(m_tracker.get_value(), m, alpha)
            new_y0 = interpolate(y0_tracker.get_value(), y0, alpha)
            self.play(LaggedStart(
                m_tracker.animate.set_value(new_m),
                y0_tracker.animate.set_value(new_y0),
                run_time=1.5,
                lag_ratio=0.25,
            ))
            self.wait(0.5)


class ShowGPT3Numbers(InteractiveScene):
    def construct(self):
        # Title
        gpt3_label = Text("GPT-3", font="Consolas", font_size=72)
        openai_logo = SVGMobject("OpenAI.svg")
        openai_logo.set_fill(WHITE)
        openai_logo.set_height(2.0 * gpt3_label.get_height())
        title = VGroup(openai_logo, gpt3_label)
        title.arrange(RIGHT)
        title.to_edge(UP)

        self.add(title)

        # 175b weights
        n_param = 175_181_291_520
        weights_count = Integer(n_param, color=BLUE)
        weights_text = VGroup(Text("Total parameters:"), weights_count)
        weights_text.arrange(RIGHT, buff=MED_SMALL_BUFF)
        weights_text.next_to(title, DOWN, buff=1.0)
        weights_arrow = Arrow(weights_count, gpt3_label, stroke_width=6, buff=0.2)

        param_shape = (8, 24)
        pre_dials = Dial().get_grid(*param_shape)
        dial_matrix = MobjectMatrix(
            pre_dials, *param_shape,
            ellipses_row=-2,
            ellipses_col=-2,
        )
        dial_matrix.set_width(FRAME_WIDTH)
        dial_matrix.next_to(weights_text, DOWN, buff=MED_SMALL_BUFF)

        dials = dial_matrix.get_entries()
        dots = dial_matrix.get_ellipses()

        self.play(
            FadeIn(weights_text[:-1], time_span=(0, 3)),
            CountInFrom(weights_count, 0),
            GrowArrow(weights_arrow, time_span=(0, 3)),
            LaggedStartMap(FadeIn, pre_dials, scale=3, lag_ratio=0.1),
            run_time=10,
        )
        self.play(
            LaggedStart(
                (dial.animate_set_value(dial.get_random_value())
                for dial in dials),
                lag_ratio=1.0 / len(dials),
                run_time=5
            )
        )
        self.wait()

        # Change name to weights
        new_name = Text("Total weights: ")
        new_name.move_to(weights_text[0], RIGHT)

        self.play(
            Transform(weights_text[0]["Total"][0], new_name["Total"][0]),
            Transform(weights_text[0]["parameters:"][0], new_name["weights:"][0]),
        )
        self.wait()

        # Organize dials into matrices
        mat_text = Text("Organized into 27,938 matrices")
        mat_text["27,938"].set_color(TEAL)
        mat_text.next_to(weights_text, DOWN, buff=MED_SMALL_BUFF)
        mat_text.shift((weights_count.get_x(LEFT) - mat_text["27,938"].get_x(LEFT)) * RIGHT)

        mat_grid_shape = n, m = (3, 7)
        matrices = VGroup(
            WeightMatrix(shape=(5, 5))
            for n in range(np.product(mat_grid_shape))
        )
        matrices.arrange_in_grid(
            *mat_grid_shape,
            v_buff_ratio=0.3,
            h_buff_ratio=0.2,
        )
        matrices.set_width(FRAME_WIDTH - 1)
        mat_dots = VGroup(
            *(
                Tex(R"\dots").next_to(mat, RIGHT)
                for mat in matrices[m - 1::m]
            ),
            *(
                Tex(R"\vdots").next_to(mat, DOWN)
                for mat in matrices[-m:]
            )
        )
        matrices_group = VGroup(matrices, mat_dots)
        matrices_group.set_width(FRAME_WIDTH - 1)
        matrices_group.next_to(mat_text, DOWN, buff=0.5)
        matrices_group.set_x(0)
        all_entries = VGroup(
            entry
            for mat in matrices
            for row in mat.get_rows()
            for entry in row
        )

        pre_entries = []
        height = all_entries[0].get_height()
        for n, entry in enumerate(all_entries):
            index = n * len(dials) // len(all_entries)
            dial = dials[min(index, len(dials) - 1)].copy()
            dial.target = dial.generate_target()
            dial.target.set_height(height)
            dial.target.move_to(entry)
            pre_entries.append(dial)
        pre_entries = VGroup(*pre_entries)

        self.remove(dial_matrix)
        lag_ratio = 1 / len(all_entries)
        self.play(
            Write(mat_text),
            LaggedStartMap(MoveToTarget, pre_entries, lag_ratio=lag_ratio),
            TransformFromCopy(dots, mat_dots),
            *(FadeIn(mat.get_brackets()) for mat in matrices)
        )
        self.play(
            FadeOut(pre_entries, lag_ratio=0.2 * lag_ratio),
            FadeIn(all_entries, lag_ratio=0.2 * lag_ratio),
            run_time=2
        )
        self.add(matrices)
        self.wait()

        # Show 8 different categories
        count_text = VGroup(weights_text, mat_text)
        title_scale_factor = 0.75
        count_text.target = count_text.generate_target()
        count_text.target.scale(title_scale_factor)
        count_text.target.to_edge(UP, MED_SMALL_BUFF).to_edge(LEFT)
        h_line = Line(LEFT, RIGHT)
        h_line.set_width(FRAME_WIDTH)
        h_line.next_to(count_text.target, DOWN).set_x(0)
        h_line.insert_n_curves(10)
        h_line.set_stroke(width=[0, 3, 3, 3, 0])

        category_names = VGroup(*map(TexText, [
            "Embedding",
            "Key",
            "Query",
            # "Value",  # Dumb alignment hack
            # "Output",
            R"Value$_\downarrow$",
            R"Value$_\uparrow$",
            "Up-projection",
            "Down-projection",
            "Unembedding",
        ]))
        # category_names[3][-1].set_fill(BLACK)  # Dumb alignment hack
        category_names.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        category_names.set_height(5.5)
        category_names.next_to(h_line, DOWN, buff=MED_LARGE_BUFF)
        category_names.to_edge(LEFT, buff=0.5)
        category_names.set_fill(border_width=0.2)

        mat_index = 0
        counts = [1, * 6 * [3], 1]
        mat_groups = VGroup()
        for name, count, dots in zip(category_names, counts, mat_dots):
            new_mat_index = mat_index + count
            mat_group = matrices[mat_index:new_mat_index]
            mat_index = new_mat_index

            mat_group.target = mat_group.generate_target()
            if len(mat_group) > 1:
                mat_group.target.add(*mat_group.copy())
            mat_group.target.arrange(RIGHT, buff=LARGE_BUFF)
            mat_group.target.set_height(0.25)
            mat_group.target.next_to(category_names, RIGHT)
            mat_group.target.match_y(name)

            dots.target = dots.generate_target()
            if dots.get_width() < dots.get_height():
                dots.target.rotate(90 * DEGREES)
            dots.target.next_to(mat_group.target, RIGHT)
            mat_groups.add(mat_group)
        mat_dots[0].target.set_opacity(0)
        mat_dots[7].target.set_opacity(0)

        n_groups = len(category_names)
        self.play(LaggedStart(
            MoveToTarget(count_text),
            title.animate.scale(title_scale_factor).next_to(count_text.target, RIGHT, LARGE_BUFF),
            FadeOut(weights_arrow),
            GrowFromCenter(h_line),
            FadeIn(category_names),
            LaggedStart(map(MoveToTarget, mat_groups), lag_ratio=0.05),
            LaggedStart(map(MoveToTarget, mat_dots[:n_groups]), lag_ratio=0.05),
            LaggedStart(map(FadeOut, mat_dots[n_groups:]), lag_ratio=0.05),
            FadeOut(matrices[sum(counts):]),
        ))

        # Add lines
        h_lines = Line(LEFT, RIGHT).set_width(13).replicate(n_groups)
        h_lines.set_stroke(WHITE, 1, 0.5)
        for name, line in zip(category_names, h_lines):
            line.next_to(name, DOWN, buff=0.1, aligned_edge=LEFT)
            name.line = line
        v_line = Line(
            mat_groups.get_corner(DL) + 0.5 * DOWN,
            mat_groups.get_corner(UL) + 0.25 * UP,
        )
        v_line.shift(SMALL_BUFF * LEFT)
        v_line.match_style(h_lines)

        self.play(
            Write(h_lines),
            Write(v_line),
        )
        self.wait()

        # Prepare expressions for parameter counts
        const_to_value = {
            "n_vocab": 50_257,
            "d_embed": 12_288,
            "d_query": 128,
            "d_value": 128,
            "n_heads": 96,
            "n_layers": 96,
            "n_neurons": 4 * 12_288,
        }
        const_lists = [
            ["d_embed", "n_vocab"],
            ["d_query", "d_embed", "n_heads", "n_layers",],
            ["d_query", "d_embed", "n_heads", "n_layers",],
            ["d_value", "d_embed", "n_heads", "n_layers",],
            ["d_embed", "d_value", "n_heads", "n_layers"],
            ["n_neurons", "d_embed", "n_layers"],
            ["d_embed", "n_neurons", "n_layers"],
            ["n_vocab", "d_embed"],
        ]

        def get_product_expression(category, consts, font_size=30, suffix=None):
            values = [const_to_value[const] for const in consts]
            result = np.product(values)
            result_str = "{:,}".format(result)
            expr = VGroup()
            expr = Text(
                " * ".join(consts) + " = " + result_str,
                font_size=font_size,
            )
            expr.next_to(v_line, RIGHT)
            expr.align_to(category.line, DOWN)
            expr.shift(0.25 * expr.get_height() * UP)
            expr.rhs = expr[result_str]
            expr.rhs.set_color(BLUE)

            counts = VGroup(
                Integer(
                    const_to_value[const],
                    font_size=0.8 * font_size,
                )
                for const in consts
            )
            counts.next_to(expr, UP, buff=0.05)
            for count, const in zip(counts, consts):
                count.match_x(expr[const])
            counts.set_fill(GREY_B)

            result = VGroup(expr, counts)

            if suffix is not None:
                label = Text(suffix)
                label.match_height(expr)
                label.next_to(expr, RIGHT, buff=MED_SMALL_BUFF)
                result.add(label)

            return result

        product_expressions = VGroup(
            get_product_expression(category, consts)
            for category, consts in zip(category_names, const_lists)
        )
        exprs = [pe[0] for pe in product_expressions]
        counts = [pe[1] for pe in product_expressions]

        # Embedding
        def highlight_category(*indices):
            category_names.target = category_names.generate_target()
            category_names.target.set_fill(opacity=0.15, border_width=0)
            for index in indices:
                category_names.target[index].set_fill(opacity=1, border_width=0.5)
            return MoveToTarget(category_names)

        self.play(
            FadeOut(mat_groups),
            FadeOut(mat_dots[1:7]),
            highlight_category(0)
        )
        self.play(
            FadeIn(exprs[0]),
            FadeIn(counts[0], 0.25 * UP),
        )
        self.wait()

        # Unembedding
        total = Integer(2 * 12_288 * 50_257)
        total.to_edge(RIGHT, buff=1.0)
        total.set_color(BLUE)
        total_box = SurroundingRectangle(total, buff=0.25)
        total_box.set_fill(BLACK, 1)
        total_box.set_stroke(WHITE, 2)
        lines = VGroup(*(Line(exprs[i].get_right(), total_box) for i in [0, 7]))
        lines.set_stroke(BLUE, 2)

        self.play(
            highlight_category(0, 7),
            TransformMatchingStrings(exprs[0].copy(), exprs[7]),
            TransformFromCopy(counts[0][0].copy(), counts[7][1]),
            TransformFromCopy(counts[0][1].copy(), counts[7][0]),
            run_time=2
        )
        self.wait()
        self.play(
            ShowCreation(lines, lag_ratio=0),
            FadeIn(total_box),
            FadeTransform(exprs[0][-11:].copy(), total),
            FadeTransform(exprs[7][-11:].copy(), total),
        )
        self.wait()
        self.play(FlashAround(weights_count, time_width=1.5, run_time=2))
        self.wait()
        self.play(
            FadeOut(lines),
            FadeOut(total_box),
            FadeOut(total),
        )
        self.wait()

        # Attention matrices
        covered_categories = [0, 7]
        att_categories = [1, 2, 3, 4]
        per_head_factors = [
            ["d_query", "d_embed"],
            ["d_query", "d_embed"],
            ["d_value", "d_embed"],
            ["d_embed", "d_value"],
        ]
        per_head_exprs = VGroup(
            get_product_expression(name, factors, suffix="per head")
            for name, factors in zip(category_names[1:5], per_head_factors)
        )
        per_layer_exprs = VGroup(
            get_product_expression(name, factors + ["n_heads"], suffix="per layer")
            for name, factors in zip(category_names[1:5], per_head_factors)
        )
        full_att_exprs = product_expressions[1:5]
        for group in [per_head_exprs, per_layer_exprs, full_att_exprs]:
            sum_box = SurroundingRectangle(
                VGroup(expr[0].rhs for expr in group)
            )
            sum_box.set_stroke(BLUE, 2)
            sum_label = Integer(sum(
                np.product(list(count.get_value() for count in expr[1]))
                for expr in group
            ))
            sum_label.set_color(BLUE)
            sum_label.next_to(sum_box, DOWN)
            sum_box.add(sum_label)
            group.sum_box = sum_box

        self.play(
            *(
                product_expressions[i].animate.set_fill(opacity=0.25, border_width=0)
                for i in covered_categories
            ),
            highlight_category(att_categories[0]),
            FadeIn(per_head_exprs[0], shift=0.5 * RIGHT)
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeIn, per_head_exprs[1:], shift=0.5 *DOWN, lag_ratio=0.5),
            highlight_category(*att_categories),
        )
        self.wait()
        self.play(FadeIn(per_head_exprs.sum_box, run_time=3, rate_func=there_and_back_with_pause))
        self.wait()
        self.play(
            FadeOut(per_head_exprs),
            FadeIn(per_layer_exprs),
        )
        self.wait()
        self.play(FadeIn(per_layer_exprs.sum_box, run_time=3, rate_func=there_and_back_with_pause))
        self.wait()
        self.play(
            FadeOut(per_layer_exprs),
            FadeIn(full_att_exprs),
        )
        self.wait()
        self.play(FadeIn(full_att_exprs.sum_box))
        self.wait()

        # Compare with total weights
        total_weights_rect = SurroundingRectangle(weights_count)
        total_weights_rect.set_stroke(BLUE_B, 2)
        box = full_att_exprs.sum_box.copy()
        box.remove(box.submobjects[0])
        self.play(Transform(box, total_weights_rect))
        self.wait()
        self.play(
            FadeOut(box),
            FadeOut(full_att_exprs.sum_box),
        )
        self.wait()

        # MLP matrices
        mlp_categories = [5, 6]
        mlp_exprs = product_expressions[5:7]
        per_layer_exprs = VGroup(
            get_product_expression(category_names[i], const_lists[i][:2], suffix="per layer")
            for i in mlp_categories
        )

        self.play(
            full_att_exprs.animate.set_fill(opacity=0.25, border_width=0),
            highlight_category(*mlp_categories),
        )
        self.wait()
        self.play(FadeIn(per_layer_exprs[0]))
        self.wait()
        self.play(
            TransformMatchingStrings(per_layer_exprs[0][0].copy(), per_layer_exprs[1][0]),
            TransformFromCopy(per_layer_exprs[0][1][0], per_layer_exprs[1][1][1]),
            TransformFromCopy(per_layer_exprs[0][1][1], per_layer_exprs[1][1][0]),
            TransformFromCopy(per_layer_exprs[0][2], per_layer_exprs[1][2]),
            run_time=1
        )
        self.wait()
        self.play(
            FadeOut(per_layer_exprs),
            FadeIn(mlp_exprs),
        )
        self.wait()

        # Sum up MLP right hand sides
        rhs_rect = SurroundingRectangle(VGroup(expr[0].rhs for expr in mlp_exprs))
        rhs_rect.set_stroke(BLUE, 2)
        rhs_rect.stretch(1.2, 1, about_edge=DOWN)
        c2v = const_to_value
        mlp_total = Integer(2 * c2v["n_neurons"] * c2v["d_embed"] * c2v["n_layers"])
        mlp_total.next_to(rhs_rect)
        mlp_total.set_color(BLUE)
        mlp_total_rect = BackgroundRectangle(mlp_total)
        mlp_total_rect.set_fill(BLACK, 1)

        self.play(
            FadeIn(rhs_rect),
            FadeIn(mlp_total_rect),
            FadeTransform(mlp_exprs[0][0].rhs.copy(), mlp_total),
            FadeTransform(mlp_exprs[1][0].rhs.copy(), mlp_total),
        )
        self.wait()

        # Align all right hand sides
        self.play(
            category_names.animate.set_fill(opacity=1, border_width=0.5),
            product_expressions.animate.set_fill(opacity=1, border_width=0.5),
        )

        all_rhss = VGroup(
            VGroup(expr[0]["="][0], expr[0].rhs)
            for expr in product_expressions
        )
        all_rhss.target = all_rhss.generate_target()
        for mob in all_rhss.target:
            mob.align_to(product_expressions, RIGHT)
            mob.shift(0.5 * RIGHT)
        all_rhss_rect = SurroundingRectangle(all_rhss.target)
        all_rhss_rect.match_style(rhs_rect)

        self.play(
            FadeOut(mlp_total_rect, RIGHT),
            FadeOut(mlp_total, RIGHT),
            ReplacementTransform(rhs_rect, all_rhss_rect),
            MoveToTarget(all_rhss)
        )
        self.wait()

        # Move weights count
        self.play(LaggedStart(
            h_line.animate.scale(0.5, about_edge=LEFT),
            weights_text.animate.arrange(DOWN).scale(1.5).next_to(all_rhss_rect, UP),
            FadeOut(mat_text, LEFT),
            title.animate.to_edge(LEFT, buff=2.5),
            lag_ratio=0.2,
            run_time=2
        ))
        self.wait()


class DistinguishWeightsAndData(InteractiveScene):
    def construct(self):
        # Set up titles
        weights_title, data_title = titles = VGroup(
            Text(word, font_size=60)
            for word in ["Weights", "Data"]
        )
        weights_title.set_color(BLUE)
        data_title.set_color(GREY_B)

        for title, sign in zip(titles, [-1, 1]):
            title.set_x(sign * FRAME_WIDTH / 4)
            title.to_edge(UP, buff=0.25)
            underline = Underline(title, stretch_factor=1.5)
            underline.match_color(title)
            underline.set_y(title[0].get_y(DOWN) - 0.1)
            title.add(underline)

        v_line = Line(UP, DOWN).set_height(4.5)
        v_line.to_edge(UP, buff=0)
        v_line.set_stroke(GREY_A, 2)

        # Set up matrices
        matrices = VGroup(
            WeightMatrix(
                shape=(6, 8),
                ellipses_row=None,
                ellipses_col=None,
            )
            for n in range(4)
        )
        matrices.arrange_in_grid(v_buff=1, h_buff=1)
        vectors = VGroup(
            NumericEmbedding(length=8, ellipses_row=None)
            for n in range(8)
        )
        vectors.arrange(RIGHT)

        tensors = VGroup(matrices, vectors)
        for group, title in zip(tensors, titles):
            group.set_height(2.5)
            group.next_to(title, DOWN, buff=0.5)

        # Mix up all the numbers
        mat_nums = VGroup(
            elem
            for matrix in matrices
            for elem in matrix.get_entries()
        )
        mat_braces = VGroup(
            brace
            for matrix in matrices
            for brace in matrix.get_brackets()
        )
        vec_nums = VGroup(
            elem
            for vector in vectors
            for elem in vector.get_entries()
        )
        vec_braces = VGroup(
            brace
            for vector in vectors
            for brace in vector.get_brackets()
        )

        def random_point(x_min, x_max, y_min, y_max):
            return np.array([
                random.uniform(x_min, x_max),
                random.uniform(y_min, y_max),
                0
            ])

        all_nums = VGroup(*mat_nums, *vec_nums)
        all_nums.shuffle()
        for num in all_nums:
            states = num.replicate(4)
            for state in states[1:]:
                state.set_height(0.15)
            sign = 1 if num in vec_nums else -1
            states[1].move_to(random_point(6.5 * sign, 1 * sign, 0, 3.5))
            states[2].move_to(random_point(-8, 8, -4, 4))
            states[3].move_to(random_point(-8, 8, -4, 4))
            states[3].set_opacity(0)
            num.states = states
            num.become(states[3])

        self.add(all_nums)

        # Animations
        lag_ratio = 1 / len(all_nums)
        self.play(
            LaggedStart(
                (Transform(num, num.states[2], path_arc=PI)
                for num in all_nums),
                lag_ratio=lag_ratio,
                run_time=3
            ),
        )
        self.wait()
        self.play(
            LaggedStart(
                (LaggedStart(
                    (Transform(num, num.states[1])
                    for num in group),
                    lag_ratio=lag_ratio,
                    run_time=2
                )
                for group in [mat_nums, vec_nums]),
                lag_ratio=0.5
            ),
            ShowCreation(v_line),
        )
        self.play(
            Write(weights_title),
            LaggedStart(
                (Transform(num, num.states[0])
                for num in mat_nums),
                lag_ratio=lag_ratio,
                run_time=2
            ),
            FadeIn(mat_braces, lag_ratio=0.1, time_span=(1, 2)),
        )
        self.play(
            Write(data_title),
            LaggedStart(
                (Transform(num, num.states[0])
                for num in vec_nums),
                lag_ratio=lag_ratio,
                run_time=2
            ),
            FadeIn(vec_braces, lag_ratio=0.1, time_span=(1, 2)),
        )
        self.wait()

        # Add subtitles
        subtitles = VGroup(
            Text("What defines the model", font_size=40),
            Text("What the model processes", font_size=40),
        )
        for subtitle, title, group in zip(subtitles, titles, tensors):
            subtitle.next_to(title, DOWN)
            self.play(
                FadeIn(subtitle, lag_ratio=0.1),
                group.animate.next_to(subtitle, DOWN, buff=0.5),
            )
            self.wait()


class SoftmaxBreakdown(InteractiveScene):
    def construct(self):
        # Show example probability distribution
        word_strs = ['Dumbledore', 'Flitwick', 'Mcgonagall', 'Quirrell', 'Snape', 'Sprout', 'Trelawney']
        words = VGroup(*(Text(word_str, font_size=30) for word_str in word_strs))
        values = np.array([-0.8, -5.0, 0.5, 1.5, 3.4, -2.3, 2.5])
        prob_values = softmax(values)
        chart = BarChart(prob_values, width=10)
        chart.bars.set_stroke(width=1)

        probs = VGroup(*(DecimalNumber(pv) for pv in prob_values))
        probs.arrange(DOWN, buff=0.25)
        probs.generate_target()
        for prob, bar in zip(probs.target, chart.bars):
            prob.scale(0.5)
            prob.next_to(bar, UP)

        for word, bar in zip(words, chart.bars):
            word.scale(0.75)
            height = word.get_height()
            word.move_to(bar.get_bottom(), LEFT)
            word.rotate(-45 * DEGREES, about_point=bar.get_bottom())
            word.shift(height * DOWN)

        chart.save_state()
        for bar in chart.bars:
            bar.stretch(0, 1, about_edge=DOWN)
        chart.set_opacity(0)

        seq_title = Text("Sequence of numbers", font_size=60)
        seq_title.next_to(probs, LEFT, buff=0.75)
        seq_title.set_color(YELLOW)
        prob_title = Text("Probability distribution", font_size=60)
        prob_title.set_color(chart.bars[3].get_color())
        prob_title.center().to_edge(UP)

        self.play(
            LaggedStartMap(FadeIn, probs, shift=0.25 * DOWN, lag_ratio=0.3),
            FadeIn(seq_title),
            run_time=1
        )
        self.wait()
        self.play(
            Restore(chart, lag_ratio=0.1),
            MoveToTarget(probs),
            FadeTransform(seq_title, prob_title),
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeIn, words),
        )
        self.wait()

        # Show constraint between 0 and 1
        index = 3
        bar = chart.bars[index]
        bar.save_state()
        prob = probs[index]
        prob.bar = bar
        max_height = chart.y_axis.get_y(UP) - chart.x_axis.get_y()
        prob.f_always.set_value(lambda: prob.bar.get_height() / max_height)
        prob.always.match_height(probs[1])
        prob.always.next_to(prob.bar, UP)

        one_line = DashedLine(*chart.x_axis.get_start_and_end())
        one_line.set_stroke(RED, 2)
        one_line.align_to(chart.y_axis, UP)

        low_line = one_line.copy()
        low_line.set_stroke(PINK, 5)
        low_line.match_y(chart.x_axis)

        self.play(FadeIn(low_line), FadeIn(one_line), FadeOut(prob_title))
        self.play(low_line.animate.match_y(one_line))
        self.play(FadeOut(low_line))
        self.wait()

        self.play(
            FadeIn(one_line, time_span=(0, 1)),
            bar.animate.set_height(max_height, about_edge=DOWN, stretch=True),
            run_time=2,
        )
        self.play(
            bar.animate.set_height(1e-4, about_edge=DOWN, stretch=True),
            run_time=2,
        )
        self.play(Restore(bar))
        self.wait()
        prob.clear_updaters()

        # Show sum
        prob_copies = probs.copy()
        prob_copies.scale(1.5)
        prob_copies.arrange(RIGHT, buff=1.0)
        prob_copies.to_edge(UP)
        prob_copies.shift(LEFT)
        plusses = VGroup(*(
            Tex("+").move_to(VGroup(p1, p2))
            for p1, p2 in zip(prob_copies, prob_copies[1:])
        ))
        equals = Tex("=").next_to(prob_copies, RIGHT)
        rhs = DecimalNumber(1.00)
        rhs.next_to(equals, RIGHT)

        self.play(
            TransformFromCopy(probs, prob_copies),
            Write(plusses),
            Write(equals),
            FadeOut(one_line),
        )
        self.play(
            LaggedStart(*(
                FadeTransform(pc.copy(), rhs)
                for pc in prob_copies
            ), lag_ratio=0.07)
        )
        self.wait()

        sum_group = VGroup(*prob_copies, *plusses, equals, rhs)
        chart_group = VGroup(chart, probs, words)

        # Show example matrix vector output
        n = len(words)
        vector = NumericEmbedding(length=n, ellipses_row=None)
        in_values = np.array([e.get_value() for e in vector.elements])
        rows = []
        for value in values:
            row = np.random.uniform(-1, 1, len(in_values))
            row *= value / np.dot(row, in_values)
            rows.append(row)
        matrix_values = np.array(rows)

        matrix = WeightMatrix(
            values=matrix_values,
            ellipses_row=None,
            ellipses_col=None,
            num_decimal_places=2,
        )
        for mob in matrix, vector:
            mob.set_height(4)
        vector.to_edge(UP).set_x(2.5)
        matrix.next_to(vector, LEFT)

        self.play(LaggedStart(
            chart_group.animate.scale(0.35).to_corner(DL),
            FadeOut(sum_group, UP),
            FadeIn(matrix, UP),
            FadeIn(vector, UP),
        ))
        eq, rhs = show_matrix_vector_product(self, matrix, vector, x_max=9)
        self.wait()

        # Comment on output
        rhs_rect = SurroundingRectangle(rhs)
        rhs_words = Text("Not at all a\nprobability distribution!")
        rhs_words.next_to(rhs_rect, DOWN)

        neg_rects = VGroup(*(
            SurroundingRectangle(entry)
            for entry in rhs.get_entries()
            if entry.get_value() < 0
        ))
        gt1_rects = VGroup(*(
            SurroundingRectangle(entry)
            for entry in rhs.get_entries()
            if entry.get_value() > 1
        ))
        VGroup(rhs_rect, neg_rects).set_stroke(RED, 4)
        gt1_rects.set_stroke(BLUE, 4)

        for rect in (*neg_rects, *gt1_rects):
            neg = rect in neg_rects
            rect.word = Text("Negative" if neg else "> 1", font_size=36)
            rect.word.match_color(rect)
            rect.word.next_to(rhs, RIGHT)
            rect.word.match_y(rect)
        neg_words = VGroup(*(r.word for r in neg_rects))
        gt1_words = VGroup(*(r.word for r in gt1_rects))

        sum_arrow = Vector(DOWN).next_to(rhs, DOWN)
        sum_sym = Tex(R"\sum", font_size=36).next_to(sum_arrow, LEFT)
        sum_num = DecimalNumber(sum(e.get_value() for e in rhs.get_entries()))
        sum_num.next_to(sum_arrow, DOWN)

        self.play(
            ShowCreation(rhs_rect),
            FadeIn(rhs_words),
        )
        self.wait()
        self.play(
            ReplacementTransform(VGroup(rhs_rect), neg_rects),
            LaggedStart(*(FadeIn(rect.word, 0.5 * RIGHT) for rect in neg_rects)),
        )
        self.wait()
        self.play(
            ReplacementTransform(neg_rects, gt1_rects),
            FadeTransformPieces(neg_words, gt1_words),
        )
        self.wait()
        self.play(
            LaggedStart(
                FadeOut(rhs_words),
                FadeOut(gt1_rects),
                FadeOut(gt1_words),
            ),
            GrowArrow(sum_arrow),
            FadeIn(sum_num, DOWN),
            FadeIn(sum_sym),
        )
        self.wait()
        self.play(*map(FadeOut, [sum_arrow, sum_sym, sum_num]))

        # Preview softmax application
        rhs.generate_target()
        rhs.target.to_edge(LEFT, buff=1.5)
        rhs.target.set_y(0)

        softmax_box = Rectangle(width=5, height=6.5)
        softmax_box.set_stroke(BLUE, 2)
        softmax_box.set_fill(BLUE_E, 0.5)
        in_arrow, out_arrow = Vector(RIGHT).replicate(2)
        in_arrow.next_to(rhs.target, RIGHT)
        softmax_box.next_to(in_arrow, RIGHT)
        out_arrow.next_to(softmax_box, RIGHT)

        softmax_label = Text("softmax", font_size=60)
        softmax_label.move_to(softmax_box)

        rhs_values = np.array([e.get_value() for e in rhs.get_entries()])
        dist = softmax(rhs_values)
        output = DecimalMatrix(dist.reshape((dist.shape[0], 1)))
        output.match_height(rhs)
        output.next_to(out_arrow, RIGHT)

        bars = chart.bars.copy()
        for bar, entry in zip(bars, output.get_entries()):
            bar.rotate(-PI / 2)
            bar.stretch(2, 0)
            bar.next_to(output)
            bar.match_y(entry)

        self.play(LaggedStart(
            FadeOut(matrix, 2 * LEFT),
            FadeOut(vector, 3 * LEFT),
            FadeOut(eq, 3.5 * LEFT),
            FadeOut(chart_group, DL),
            GrowArrow(in_arrow),
            FadeIn(softmax_box, RIGHT),
            FadeIn(softmax_label, RIGHT),
            MoveToTarget(rhs),
            GrowArrow(out_arrow),
            FadeIn(output, RIGHT),
            TransformFromCopy(chart.bars, bars),
        ), lag_ratio=0.2, run_time=2)
        self.wait()

        # Highlight larger and smaller parts
        rhs_entries = rhs.get_entries()
        changer = VGroup(rhs_entries, output.get_entries(), bars)
        changer.save_state()
        for index in range(4, 0, -1):
            changer.target = changer.saved_state.copy()
            changer.target.set_fill(border_width=0)
            for group in changer.target:
                for j, elem in enumerate(group):
                    if j != index:
                        elem.fade(0.8)
            self.play(MoveToTarget(changer))
            self.wait()
        self.play(Restore(changer))
        self.remove(changer)
        self.add(rhs, output, bars)
        self.wait()

        # Swap out for variables
        variables = VGroup(*(
            Tex(f"x_{{{n}}}", font_size=48).move_to(elem)
            for n, elem in enumerate(rhs_entries, start=1)
        ))

        self.remove(rhs_entries)
        self.play(
            LaggedStart(*(
                TransformFromCopy(entry, variable, path_arc=PI / 2)
                for entry, variable in zip(rhs_entries, variables)
            ), lag_ratio=0.1, run_time=1.0)
        )
        self.wait()

        # Exponentiate each part
        exp_parts = VGroup(*(
            Tex(f"e^{{{var.get_tex()}}}", font_size=48).move_to(var)
            for var in variables
        ))
        exp_parts.align_to(softmax_box, LEFT)
        exp_parts.shift(0.75 * RIGHT)
        exp_parts.space_out_submobjects(1.5)
        gt0s = VGroup(
            Tex(R"> 0").next_to(exp_part, aligned_edge=DOWN)
            for exp_part in exp_parts
        )

        self.play(
            softmax_label.animate.next_to(softmax_box, UP, buff=0.15),
            LaggedStart(*(
                TransformMatchingStrings(var.copy(), exp_part)
                for var, exp_part in zip(variables, exp_parts)
            ), run_time=1, lag_ratio=0.01)
        )
        self.play(LaggedStartMap(FadeIn, gt0s, shift=0.5 * RIGHT, lag_ratio=0.25, run_time=1))
        self.wait()
        self.play(FadeOut(gt0s))

        # Compute the sum
        exp_sum = Tex(R"\sum_{n=0}^{N-1} e^{x_{n}}", font_size=42)
        exp_sum[R"e^{x_{n}}"].scale(1.5, about_edge=LEFT)
        exp_sum.next_to(softmax_box.get_right(), LEFT, buff=0.75)

        lines = VGroup(*(Line(exp_part.get_right(), exp_sum.get_left(), buff=0.1) for exp_part in exp_parts))
        lines.set_stroke(TEAL, 2)

        self.play(
            LaggedStart(*(
                FadeTransform(exp_part.copy(), exp_sum)
                for exp_part in exp_parts
            ), lag_ratio=0.01),
            LaggedStartMap(ShowCreation, lines, lag_ratio=0.01),
            run_time=1
        )
        self.wait()
        self.play(FadeOut(lines))

        # Divide each part by the sum
        lil_denoms = VGroup()
        for exp_part in exp_parts:
            slash = Tex("/").match_height(exp_sum)
            slash.next_to(exp_sum, LEFT, buff=0)
            denom = VGroup(slash, exp_sum).copy()
            denom.set_height(exp_part.get_height() * 1.5)
            denom.next_to(exp_part, RIGHT, buff=0)
            lil_denoms.add(denom)
        lil_denoms.align_to(softmax_box.get_center(), LEFT)

        lines = VGroup(*(Line(exp_sum.get_left(), denom.get_center()) for denom in lil_denoms))
        lines.set_stroke(TEAL, 1)

        self.remove(exp_sum)
        self.play(
            exp_parts.animate.next_to(lil_denoms, LEFT, buff=0),
            LaggedStart(*(
                FadeTransform(exp_sum.copy(), denom)
                for denom in lil_denoms
            ), lag_ratio=0.01),
        )
        self.wait()

        # Resize box
        sm_terms = VGroup(*(
            VGroup(exp_part, denom)
            for exp_part, denom in zip(exp_parts, lil_denoms)
        ))
        sm_terms.generate_target()

        target_height = 5.0
        full_output = Group(output, bars)
        full_output.generate_target()
        full_output.target.set_height(target_height, about_edge=RIGHT)
        full_output.target.shift(1.5 * LEFT)
        equals = Tex("=")
        equals.next_to(full_output.target, LEFT)

        softmax_box.generate_target()
        softmax_box.target.set_width(3.0, stretch=True)
        VGroup(softmax_box.target, sm_terms.target).set_height(target_height + 0.5).next_to(equals, LEFT)

        rhs.generate_target()
        rhs_entries.become(variables)
        self.remove(variables)
        rhs.target.set_height(target_height)
        rhs.target.next_to(softmax_box.target, LEFT, buff=1.5)

        self.play(
            softmax_label.animate.next_to(softmax_box.target, UP),
            MoveToTarget(softmax_box),
            MoveToTarget(sm_terms),
            MoveToTarget(full_output),
            MoveToTarget(rhs),
            FadeTransform(out_arrow, equals),
            in_arrow.animate.become(
                Arrow(rhs.target, softmax_box.target).match_style(in_arrow)
            ),
        )
        self.wait()

        # Set up updaters
        output_entries = output.get_entries()
        bar_width_ratio = bars.get_width() / max(o.get_value() for o in output_entries)
        temp_tracker = ValueTracker(1)

        def update_outs(output_entries):
            inputs = [entry.get_value() for entry in rhs_entries]
            outputs = softmax(inputs, temp_tracker.get_value())
            for entry, output in zip(output_entries, outputs):
                entry.set_value(output)

        def update_bars(bars):
            for bar, entry in zip(bars, output_entries):
                width = max(bar_width_ratio * entry.get_value(), 1e-3)
                bar.set_width(width, about_edge=LEFT, stretch=True)

        output_entries.clear_updaters().save_state()
        bars.clear_updaters().save_state()
        output_entries.add_updater(update_outs)
        bars.add_updater(update_bars)

        self.add(bars, output_entries)

        # Tweak values
        index_value_pairs = [
            (6, 4.0),
            (4, 4.2),
            (2, 4.0),
            (0, 6.0),
            (4, 9.9)
        ]
        # index_value_pairs = [  # For emphasizing a max
        #     (3, 8.5),
        #     (6, 8.0),
        #     (2, 8.1),
        #     (0, 9.0),
        # ]
        for index, value in index_value_pairs:
            entry = rhs_entries[index]
            rect = SurroundingRectangle(entry)
            rect.set_stroke(BLUE if value > entry.get_value() else RED, 3)
            self.play(
                ChangeDecimalToValue(entry, value),
                FadeIn(rect, time_span=(0, 1)),
                run_time=4
            )
            self.play(FadeOut(rect))

        # Add temperature
        frame = self.frame
        temp_color = RED
        new_title = Text("softmax with temperature")
        new_title["temperature"].set_color(temp_color)
        get_t = temp_tracker.get_value
        t_line = NumberLine(
            (0, 10, 0.2),
            tick_size=0.025,
            big_tick_spacing=1,
            longer_tick_multiple=2.0,
            width=4
        )
        t_line.set_stroke(width=1.5)
        t_line.next_to(softmax_box, UP)
        t_tri = ArrowTip(angle=-90 * DEGREES)
        t_tri.set_color(temp_color)
        t_tri.set_height(0.2)
        t_label = Tex("T = 0.00", font_size=36)
        t_label.rhs = t_label.make_number_changeable("0.00")
        t_label["T"].set_color(temp_color)
        t_tri.add_updater(lambda m: m.move_to(t_line.n2p(get_t()), DOWN))
        t_label.add_updater(lambda m: m.rhs.set_value(get_t()))
        t_label.add_updater(lambda m: m.next_to(t_tri, UP, buff=0.1, aligned_edge=LEFT))
        t_label.update()

        new_title.next_to(t_label, UP, buff=0.5).match_x(softmax_box)

        self.play(
            frame.animate.move_to(0.75 * UP),
            TransformMatchingStrings(softmax_label, new_title),
            FadeIn(t_line),
            FadeIn(t_tri),
            FadeIn(t_label),
            run_time=1
        )

        # Change formula
        template = Tex(R"e^{x_{0} / T} / \sum_{n=0}^{N - 1} e^{x_n / T}")
        template["T"].set_color(temp_color)
        template["/"][1].scale(1.9, about_edge=LEFT)
        template[R"\sum_{n=0}^{N - 1}"][0].scale(0.7, about_edge=RIGHT)
        index_part = template.make_number_changeable("0")

        new_sm_terms = VGroup()
        all_Ts = VGroup()
        for n, term in enumerate(sm_terms, start=1):
            template.replace(term, dim_to_match=1)
            index_part.set_value(n)
            new_term = template.copy()
            all_Ts.add(*new_term["T"])
            new_sm_terms.add(new_term)

        self.play(
            LaggedStart(*(
                FadeTransform(old_term, new_term)
                for old_term, new_term in zip(sm_terms, new_sm_terms)
            )),
            LaggedStart(*(
                TransformFromCopy(t_label[0], t_mob[0])
                for t_mob in all_Ts
            )),
        )
        self.wait()

        # Oscilate between values
        for value in [4, 10, 2]:
            self.play(temp_tracker.animate.set_value(value), run_time=8)
            self.wait()
        self.play(temp_tracker.animate.set_value(0), run_time=3)
        max_rects = VGroup(
            SurroundingRectangle(rhs.get_entries()[4]),
            SurroundingRectangle(VGroup(output.get_entries()[4], bars[4])),
        )
        self.play(LaggedStartMap(ShowCreationThenFadeOut, max_rects))
        self.wait()
        for value in [5, 1, 7]:
            self.play(temp_tracker.animate.set_value(value), run_time=4)
            self.wait()

        # Describe logits
        prob_arrows, logit_arrows = (
            VGroup(*(
                Vector(-vect).next_to(entry, vect, buff=0.25)
                for entry in matrix.get_entries()
            ))
            for matrix, vect in [(output, RIGHT), (rhs, LEFT)]
        )
        prob_arrows.next_to(bars, RIGHT)
        prob_rects = VGroup(*map(SurroundingRectangle, output.get_entries()))
        logit_rects = VGroup(*map(SurroundingRectangle, rhs.get_entries()))
        VGroup(prob_rects, logit_rects).set_stroke(width=1)

        prob_words = Text("Probabilities")
        prob_words.next_to(output, UP, buff=0.25)
        logit_words = Text("Logits")
        logit_words.next_to(rhs, UP, buff=0.25)

        logit_group = VGroup(logit_arrows, logit_words, logit_rects)
        logit_group.set_color(TEAL)
        prob_group = VGroup(prob_arrows, prob_words, prob_rects)
        prob_group.set_color(YELLOW)

        for arrows, word, rects in [prob_group, logit_group]:
            self.play(
                t_line.animate.set_y(3.35),
                Write(word),
                Write(rects, stroke_width=5, stroke_color=rects[0].get_stroke_color(), lag_ratio=0.3, run_time=3),
            )
            self.wait()


class CostFunction(InteractiveScene):
    def construct(self):
        # Add graph
        axes = Axes((0, 1, 0.1), (0, 5, 1), width=10, height=6)
        axes.center().to_edge(LEFT)
        axes.x_axis.add_numbers(num_decimal_places=1)
        axes.y_axis.add_numbers(num_decimal_places=0, direction=LEFT)
        x_label = Tex("p")
        x_label.next_to(axes.x_axis.get_right(), UR)
        axes.add(x_label)

        graph = axes.get_graph(lambda x: -np.log(x), x_range=(0.001, 10, 0.01))
        graph.set_color(RED)

        expr = Tex(R"\text{Cost} = -\log(p)", font_size=60)
        expr.next_to(axes.i2gp(0.1, graph), UR, buff=0.1)

        self.add(axes, graph, expr)

        # Add sample phrase
        phrase = Text("Watching 3Blue1Brown makes you smarter")
        phrase.scale(0.75)
        phrase.to_edge(UP)
        phrase.align_to(axes.c2p(0.1, 0), LEFT)
        pieces = break_into_tokens(phrase)
        pieces[-1].set_opacity(0.0)
        rects = get_piece_rectangles(pieces, leading_spaces=True, h_buff=0)

        self.add(rects, pieces)

        # Add predictions
        arrow = Vector(0.5 * DOWN)
        arrow.next_to(rects[-1], DOWN, SMALL_BUFF)
        index = 0

        tokens, probs = gpt3_predict_next_token(phrase.get_text()[:-len(" smarter")])
        bar_chart = next_token_bar_chart(
            tokens[:8], probs[:8],
            width_100p=7.0,
            bar_space_factor=1.0,
            use_percent=False,
        )
        bar_chart.next_to(arrow, DOWN)
        bar_chart.shift(1.25 * RIGHT)
        bar_chart.set_opacity(0.5)
        bar_chart[index].set_opacity(1.0)
        rect = SurroundingRectangle(bar_chart[index])

        self.add(arrow, bar_chart, rect)

        # Animate in graph
        self.play(
            ShowCreation(graph, run_time=3),
            Write(expr, run_time=2),
        )
        self.wait()

        # Show point on the graph
        line = axes.get_line_from_axis_to_point(0, axes.i2gp(probs[index], graph), line_func=Line)
        line.set_stroke(YELLOW)

        self.play(FadeTransform(rect.copy(), line))
        self.wait()
