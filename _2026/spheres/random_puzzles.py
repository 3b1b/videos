from manim_imports_ext import *


class IntervalWithSample(Group):
    def __init__(
        self,
        width=4,
        numbers_font_size=36,
        x_range=(-1, 1, 0.25),
        label_step=0.5,
        marker_length=0.25,
        marker_width=(0, 15),
        marker_color=BLUE_B,
        dec_label_font_size=30,
    ):
        self.number_line = NumberLine(x_range, width=width)
        self.number_line.add_numbers(
            (-1, 0, 1),
            font_size=numbers_font_size,
            num_decimal_places=0
        )
        self.x_tracker = ValueTracker()
        get_x = self.x_tracker.get_value

        self.marker = Line(ORIGIN, marker_length * UP)
        self.marker.set_stroke(marker_color, width=marker_width, flat=True)
        self.marker.add_updater(lambda m: m.put_start_on(self.number_line.n2p(get_x())))

        self.dec_label = DecimalNumber(font_size=dec_label_font_size)
        self.dec_label.add_updater(lambda m: m.set_value(get_x()))
        self.dec_label.add_updater(
            lambda m: m.next_to(self.marker.get_end(), normalize(self.marker.get_vector()), SMALL_BUFF)
        )

        super().__init__(
            self.x_tracker,
            self.number_line,
            self.marker,
            self.dec_label
        )


class DotHistory(GlowDots):
    def __init__(self, pos_func, color=GREEN, fade_rate=0.95):
        super().__init__(points=[pos_func()], color=color)
        self.pos_func = pos_func
        self.fade_rate = fade_rate
        self.add_updater(lambda m: m.update_dots())

    def update_dots(self):
        curr_points = self.get_points()
        point = self.pos_func()
        opacities = self.get_opacities().copy()
        if not np.isclose(point, curr_points[-1]).all():
            self.append_points([point])
            opacities = np.append(opacities, [1])
        opacities *= self.fade_rate
        self.set_opacity(opacities)

    def reset(self):
        self.set_points([self.pos_func()])


class RandomSumsOfSquares(InteractiveScene):
    tex_to_color = {"x": BLUE, "y": YELLOW, "z": RED, "w": PINK}

    def get_intervals(self, n_intervals, buff=MED_LARGE_BUFF, **kwargs):
        result = Group(IntervalWithSample(**kwargs) for n in range(n_intervals))
        result.arrange(DOWN, buff=buff)
        return result

    def get_labeled_intervals(self, label_texs, font_size=48, buff=MED_LARGE_BUFF):
        intervals = self.get_intervals(len(label_texs), buff=buff)
        labels = VGroup(
            Tex(tex, font_size=font_size, t2c=self.tex_to_color)
            for tex in label_texs
        )
        for label, interval in zip(labels, intervals):
            label.next_to(interval.number_line.get_start(), LEFT, MED_LARGE_BUFF)
            interval.label = label
            interval.add(label)
        return intervals

    def get_probability_question(self, sum_tex="x^2 + y^2"):
        question = Tex(Rf"P\left({sum_tex} \le 1 \right)", t2c=self.tex_to_color)
        question.to_corner(UR, buff=LARGE_BUFF)
        return question

    def get_evaluation_object(self, intervals):
        equation = Tex(" + ".join(["(+0.00)^2"] * len(intervals)) + "= 1.00")
        x_terms = equation.make_number_changeable("+0.00", replace_all=True, include_sign=True)
        for x_term, interval in zip(x_terms, intervals):
            if hasattr(interval, "label"):
                x_term.match_color(interval.label[0])
            x_term.f_always.set_value(interval.x_tracker.get_value)
        rhs = equation.make_number_changeable("1.00")
        rhs.add_updater(lambda m: m.set_value(sum([
            interval.x_tracker.get_value()**2
            for interval in intervals
        ])))

        marks = VGroup(
            Checkmark().set_color(GREEN),
            Exmark().set_color(RED),
        )
        marks.set_width(0.75 * rhs.get_width())
        marks[0].add_updater(lambda m: m.set_opacity(float(rhs.get_value() <= 1)))
        marks[1].add_updater(lambda m: m.set_opacity(float(rhs.get_value() > 1)))
        marks.next_to(rhs, DOWN)
        equation.add(marks)

        return equation

    def set_interval_values_randomly(self, intervals):
        for interval in intervals:
            value = random.uniform(*interval.number_line.x_range[:2])
            interval.x_tracker.set_value(value)

    def randomize_intervals(self, intervals, run_time=3, frequency=0.1):
        intervals.value_at_last_change = -1

        def update_intervals(intervals, alpha):
            curr_value = int(alpha * run_time / frequency)
            if curr_value != intervals.value_at_last_change:
                self.set_interval_values_randomly(intervals)
                intervals.value_at_last_change = curr_value

        return UpdateFromAlphaFunc(intervals, update_intervals, rate_func=linear, run_time=run_time)


class SumOfTwoSquares(RandomSumsOfSquares):
    def construct(self):
        # Set up
        intervals = self.get_labeled_intervals("xy", buff=LARGE_BUFF)
        intervals.to_corner(UL)
        question = self.get_probability_question()
        question.set_width(5)
        question.set_x(3.5)
        evaluation = self.get_evaluation_object(intervals)
        evaluation.next_to(question, DOWN, LARGE_BUFF)

        # Show initial randomization
        frame = self.frame
        frame.set_height(6).move_to(intervals)

        self.add(intervals[0])
        self.play(self.randomize_intervals(intervals[:1], frequency=0.25, run_time=10))
        self.add(intervals)
        self.play(self.randomize_intervals(intervals, frequency=0.25, run_time=10))
        self.play(
            frame.animate.to_default_state(),
            FadeIn(question),
            run_time=2,
        )
        self.wait()
        evaluation.update()
        evaluation.suspend_updating()
        self.play(Write(evaluation))
        self.wait()
        evaluation.resume_updating()
        intervals[0].x_tracker.set_value(0.25)
        intervals[1].x_tracker.set_value(-0.15)
        self.wait()

        self.play(self.randomize_intervals(intervals, frequency=0.25, run_time=10))

        # Change to axes
        axes = Axes((-1, 1, 0.25), (-1, 1, 0.25), width=6, height=6)
        axes.set_height(6)
        axes.move_to(3 * LEFT)
        for axis in axes:
            axis.add_numbers([-1, 0, 1])
            axis.numbers[1].set_opacity(0)

        get_x = intervals[0].x_tracker.get_value
        get_y = intervals[1].x_tracker.get_value
        x_dot = GlowDot(color=BLUE)
        x_dot.add_updater(lambda m: m.move_to(intervals[0].number_line.n2p(get_x())))
        y_dot = GlowDot(color=YELLOW)
        y_dot.add_updater(lambda m: m.move_to(intervals[1].number_line.n2p(get_y())))

        xy_dot = GlowDot(color=WHITE)
        xy_dot.add_updater(lambda m: m.move_to(axes.c2p(get_x(), get_y())))
        h_line, v_line = Line().set_stroke(WHITE, 1).replicate(2)
        h_line.f_always.put_start_and_end_on(y_dot.get_center, xy_dot.get_center)
        v_line.f_always.put_start_and_end_on(x_dot.get_center, xy_dot.get_center)

        xy_coord_label = Tex(R"(x, y)", t2c={"x": BLUE, "y": YELLOW}, font_size=24)
        xy_coord_label.add_updater(lambda m: m.next_to(xy_dot.get_center(), UR, buff=SMALL_BUFF))

        self.play(
            question.animate.set_height(0.7).to_corner(UR),
            FadeOut(evaluation.clear_updaters()),
            Transform(intervals[0].number_line, axes.x_axis),
            Transform(intervals[1].number_line, axes.y_axis),
            intervals[0].dec_label.animate.set_opacity(0),
            intervals[1].dec_label.animate.set_opacity(0),
            intervals[0].marker.animate.set_opacity(0),
            intervals[1].marker.animate.set_opacity(0),
            FadeIn(x_dot),
            FadeIn(y_dot),
            intervals[0].label.animate.next_to(axes.x_axis.get_end(), RIGHT, SMALL_BUFF),
            intervals[1].label.animate.next_to(axes.y_axis.get_end(), UP, SMALL_BUFF),
            run_time=2
        )
        VGroup(h_line, v_line).update()
        self.play(
            ShowCreation(h_line, suspend_mobject_updating=True),
            ShowCreation(v_line, suspend_mobject_updating=True),
            TransformFromCopy(x_dot, xy_dot, suspend_mobject_updating=True),
            TransformFromCopy(y_dot, xy_dot, suspend_mobject_updating=True),
            FadeIn(xy_coord_label[0::2]),
            TransformFromCopy(intervals[0].label, xy_coord_label[1]),
            TransformFromCopy(intervals[1].label, xy_coord_label[3]),
        )
        self.add(xy_coord_label, xy_dot, h_line, v_line)
        self.wait()

        # Show the random points within a square
        dot_history = DotHistory(xy_dot.get_center)
        dot_history.set_z_index(-1)

        square = Square(side_length=axes.x_axis.get_length())
        square.move_to(axes.get_origin())
        square.set_fill(GREEN, 0.1)
        square.set_stroke(GREEN, 1)

        self.add(dot_history)
        self.play(
            self.randomize_intervals(intervals, frequency=0.2, run_time=10),
            FadeIn(square),
        )
        self.wait(3)
        self.remove(dot_history)

        # Show circle
        circle = Circle()
        circle.replace(square)
        circle.set_stroke(TEAL, 3)
        circle.set_fill(TEAL, 0.1)
        underline = Underline(question[R"x^2 + y^2 \le 1"], buff=0, stretch_factor=1)
        underline.match_style(circle)

        self.play(ShowCreation(underline))
        self.wait()
        self.play(ReplacementTransform(underline, circle, run_time=1.5))
        self.wait()

        # Pythagorean Theorem
        self.play(
            intervals[0].x_tracker.animate.set_value(0.6),
            intervals[1].x_tracker.animate.set_value(0.8),
        )

        r_line = Line(axes.get_origin(), xy_dot.get_center())
        r_line.set_stroke(RED, 4)
        r_label = Tex(R"r")
        r_label.set_color(RED)
        r_label.next_to(r_line.get_center(), UL, SMALL_BUFF)

        h_line.save_state()
        v_line.save_state()
        h_line.suspend_updating()
        v_line.suspend_updating()
        h_line.match_y(axes.get_origin())
        h_line.set_stroke(BLUE, 4)
        v_line.set_stroke(YELLOW, 4)

        self.tex_to_color["{r}"] = RED
        pythag = Tex(R"x^2 + y^2 = {r}^2", t2c=self.tex_to_color, font_size=72)
        pythag.to_edge(RIGHT, buff=LARGE_BUFF)

        self.play(
            ShowCreation(r_line),
            FadeIn(r_label, 0.25 * r_line.get_vector())
        )
        self.play(Write(pythag))
        self.wait()
        self.play(
            FadeOut(r_line),
            FadeOut(r_label),
            FadeOut(pythag),
            Restore(h_line),
            Restore(v_line),
        )

        h_line.resume_updating()
        v_line.resume_updating()

        # More randomness
        dot_history.reset()
        self.add(dot_history)
        self.play(self.randomize_intervals(intervals, frequency=0.2, run_time=30))
        self.wait(2)
        self.remove(dot_history)

        # Go three d
        q3d_tex = "x^2 + y^2 + z^2"
        question_3d = self.get_probability_question(q3d_tex)
        question_3d.match_height(question)
        question_3d.move_to(question, RIGHT)
        question_3d.fix_in_frame()
        frame = self.frame

        length = axes.x_axis.get_length()
        axes3d = ThreeDAxes((-1, 1), (-1, 1), (-1, 1), width=length, height=length, depth=length)
        axes3d.move_to(axes.get_origin())
        z_interval = self.get_intervals(1, width=length)[0]
        z_interval.number_line.rotate(90 * DEG, OUT).rotate(90 * DEG, RIGHT)
        z_interval.number_line.remove(z_interval.number_line.numbers)
        z_interval.number_line.ticks.stretch(0.5, 0)
        z_interval.shift(axes.get_origin() - z_interval.number_line.n2p(0))
        z_interval.marker.set_opacity(0)
        z_interval.dec_label.set_opacity(0)
        z_axis_label = Tex(R"z")
        z_axis_label.set_color(RED)
        z_axis_label.rotate(90 * DEG, RIGHT)
        z_axis_label.next_to(axes3d, OUT, MED_SMALL_BUFF)

        get_z = z_interval.x_tracker.get_value
        z_interval.x_tracker.set_value(random.random())
        xyz_dot = GlowDot(color=WHITE)
        xyz_dot.add_updater(lambda m: m.move_to(axes3d.c2p(get_x(), get_y(), get_z())))
        z_line = Line().set_stroke(WHITE, 1)
        z_line.f_always.put_start_and_end_on(xy_dot.get_center, xyz_dot.get_center)
        xyz_coord_label = Tex(R"(x, y, z)", t2c=self.tex_to_color, font_size=30)
        xyz_coord_label.rotate(90 * DEG, RIGHT)
        xyz_coord_label.add_updater(lambda m: m.next_to(xyz_dot.get_center(), RIGHT, SMALL_BUFF))

        self.play(TransformMatchingTex(question, question_3d, run_time=1))
        self.play(FlashAround(question_3d[q3d_tex], time_width=1.5, run_time=2))
        self.play(
            frame.animate.reorient(-26, 79, 0, (-0.13, 0.08, 0.47), 10.09),
            Write(z_interval.number_line),
            Write(z_axis_label),
            run_time=2
        )
        self.play(
            TransformFromCopy(xy_dot, xyz_dot, suspend_mobject_updating=True),
            xy_dot.animate.set_opacity(0),
            TransformMatchingTex(xy_coord_label, xyz_coord_label),
            ShowCreation(z_line, suspend_mobject_updating=True),
            run_time=1
        )
        self.add(xyz_dot, z_line)
        self.wait()

        # Show cube and sphere
        cube = VCube()
        cube.match_width(square)
        cube.match_style(square)
        cube.set_fill(opacity=0.05)
        cube.move_to(square)
        cube.deactivate_depth_test()
        cube.save_state()
        cube.stretch(0, 2)

        sphere = Sphere()
        sphere.replace(cube)
        sphere_mesh = SurfaceMesh(sphere, resolution=(101, 51))
        sphere_mesh.set_stroke(WHITE, 2, 0.1)
        sphere_mesh.deactivate_depth_test()

        self.remove(square)
        self.play(
            Restore(cube),
            Write(sphere_mesh, lag_ratio=1e-3, time_span=(1, 3)),
            FadeOut(circle, time_span=(1, 3)),
            frame.animate.reorient(-41, 69, 0, (-0.34, -0.78, -0.59), 12.47).set_anim_args(run_time=4),
        )
        self.wait()

        # Even more randomness
        dot_history = DotHistory(xyz_dot.get_center)
        self.add(dot_history, xyz_dot, z_line)
        intervals.add(z_interval)
        self.play(
            self.randomize_intervals(intervals, frequency=0.2, run_time=20),
            frame.animate.reorient(14, 62, 0, (1.36, 0.96, 0.63), 12.10),
            run_time=20
        )
        self.play(self.randomize_intervals(intervals, frequency=0.2, run_time=20))


class AskAboutThreeSumsOfSquares(RandomSumsOfSquares):
    label_texs = "xyz"
    square_sum_tex = "x^2 + y^2 + z^2"

    def construct(self):
        intervals = self.get_labeled_intervals(self.label_texs, buff=LARGE_BUFF)
        intervals.to_edge(LEFT)
        question = self.get_probability_question(self.square_sum_tex)
        question.to_corner(UR)
        evaluation = self.get_evaluation_object(intervals)
        evaluation.set_width(6.5)
        evaluation.next_to(question, DOWN, LARGE_BUFF)
        evaluation.to_edge(RIGHT)

        self.add(intervals)
        self.add(question)
        self.add(evaluation)

        self.play(self.randomize_intervals(intervals, frequency=0.2, run_time=30))


class AskAboutFourSumsOfSquares(AskAboutThreeSumsOfSquares):
    label_texs = "xyzw"
    square_sum_tex = "x^2 + y^2 + z^2 + w^2"


class AskAboutLargeSumOfSquares(RandomSumsOfSquares):
    def construct(self):
        # Test
        intervals = self.get_intervals(
            8,
            width=6,
            numbers_font_size=24,
            dec_label_font_size=24,
            marker_length=0.15
        )
        intervals.set_height(7)
        intervals.to_edge(LEFT, buff=LARGE_BUFF)
        to_remove = intervals[-3:-1]
        dots = Tex(R"\vdots", font_size=90)
        dots.move_to(to_remove)
        intervals.remove(*to_remove)

        labels = VGroup(
            Tex(Rf"x_{{{int(n)}}}", font_size=36)
            for n in [*range(1, 6), 100]
        )
        labels.set_submobject_colors_by_gradient(BLUE, YELLOW)
        for label, interval in zip(labels, intervals):
            label.next_to(interval.number_line.get_start(), LEFT, MED_SMALL_BUFF)

        question = Tex(
            R"P\left(x_1^2 + x_2^2 + \cdots + x_{100}^2 \le 1 \right)",
            t2c={"x_1": labels[0].get_color(), "x_2": labels[1].get_color(), "x_{100}": labels[-1].get_color()}
        )
        question.to_corner(UR)

        self.add(intervals)
        self.add(labels)
        self.add(dots)
        self.add(question)

        self.play(self.randomize_intervals(intervals, run_time=20, frequency=0.2))


class DotProductOfUnitVectors(InteractiveScene):
    random_seed = 2

    def construct(self):
        # Set up
        plane = NumberPlane((-4, 4), (-2, 2))
        plane.set_height(12)
        plane.background_lines.set_stroke(BLUE, 1, 0.5)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)
        unit_size = plane.x_axis.get_unit_size()

        circle = Circle(radius=unit_size)
        circle.set_stroke(GREY_C, 2)

        self.add(plane, circle)

        # Vectors
        v1, v2 = vects = Vector(unit_size * RIGHT, thickness=6).replicate(2)
        colors = [PINK, YELLOW]
        for vect, color, char in zip(vects, colors, "vw"):
            vect.set_fill(color)
            vect.label = Tex(R"\vec{\textbf{" + char + R"}}")
            vect.label.match_color(vect)
            vect.label.set_backstroke(BLACK, 5)
            vect.label.scale(2)
            vect.angle_tracker = ValueTracker(0)

        v1.angle_tracker.set_value(0.0)
        v2.angle_tracker.set_value(0.6)

        def update_vector(vector):
            vector.put_start_and_end_on(ORIGIN, circle.pfp(vector.angle_tracker.get_value() % 1))
            vector.label.move_to(vector.get_end() + 0.2 * vector.get_vector())

        v1.add_updater(update_vector)
        v2.add_updater(update_vector)

        self.add(v1, v2, v1.label, v2.label)

        self.play(v1.angle_tracker.animate.set_value(0.4), run_time=3)
        self.wait()

        # Show projection
        proj_group = always_redraw(lambda: self.get_projection_group(v1, v2))
        proj_group.suspend_updating()
        dashed_line, proj_line, proj_brace, label = proj_group

        self.play(LaggedStart(
            ShowCreation(dashed_line),
            FadeIn(proj_line),
            GrowFromCenter(proj_brace),
            AnimationGroup(
                TransformFromCopy(v1.label, label[0]),
                TransformFromCopy(v2.label, label[1]),
                Write(label[2])
            ),
            lag_ratio=0.5,
        ))
        self.wait()

        # Wander
        proj_group.resume_updating()
        self.add(proj_group)

        self.play(
            v1.angle_tracker.animate.set_value(0.2),
            run_time=3
        )
        self.wait()


        self.play(
            v2.angle_tracker.animate.set_value(0.25),
            run_time=3
        )
        self.wait()
        for value in [-0.23, 0.2]:
            self.play(
                v1.angle_tracker.animate.set_value(value),
                run_time=5
            )
            self.wait()

        # Randomness
        self.remove(proj_group)
        for n in range(50):
            v1.angle_tracker.set_value(random.random())
            v2.angle_tracker.set_value(random.random())
            self.wait(0.2)

    def get_projection_group(self, v1, v2):
        # Test
        dp = np.dot(v1.get_vector(), v2.get_vector()) / (v1.get_length() * v2.get_length())
        proj_line = Line(
            v2.get_start(),
            interpolate(v2.get_start(), v2.get_end(), dp)
        )
        proj_line.set_stroke(WHITE, 8)
        orientation = cross(v1.get_vector(), v2.get_vector())[2]
        proj_brace = LineBrace(
            proj_line,
            buff=SMALL_BUFF,
            # direction=UP if orientation > 0 else DOWN
            direction=UP if dp > 0 else DOWN
        )
        label = VGroup(v1.label, v2.label).copy()
        label.scale(0.5)
        label.arrange(RIGHT, buff=0.25)
        label.add(Tex(R"\cdot").move_to(label))
        angle = (proj_line.get_angle() + (PI if orientation > 0 else 0))
        angle = (angle + PI / 2) % PI - PI / 2
        label.rotate(angle)
        label.move_to(proj_brace.get_center() + 3 * (proj_brace.get_center() - proj_line.get_center()))

        dashed_line = DashedLine(v1.get_end(), proj_line.get_end())
        dashed_line.set_stroke(WHITE, 1)

        result = VGroup(dashed_line, proj_line, proj_brace, label)

        return result


class Random3DVectors(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        radius = 2
        axes = ThreeDAxes(unit_size=radius)
        sphere = Sphere(radius=radius)
        mesh = SurfaceMesh(sphere, resolution=(51, 26))
        mesh.set_stroke(WHITE, 2, 0.1)

        v1, v2 = vects = Vector(RIGHT, thickness=4).replicate(2)
        colors = [PINK, YELLOW]
        for vect, color in zip(vects, colors):
            vect.set_fill(color)
            vect.always.set_perpendicular_to_camera(frame)
            self.set_vector_randomly(vect, radius)

        frame.reorient(-38, 74, 0, (0.04, -0.01, -0.06), 4.82)
        self.add(axes, mesh)
        self.add(vects)

        # Add dot product label
        def get_dp():
            vect1 = v1.get_vector()
            vect2 = v2.get_vector()
            return np.dot(vect1, vect2) / (get_norm(vect1) * get_norm(vect2))

        dp_label = Tex(R"\vec{\textbf{v}} \cdot \vec{\textbf{w}} = 0.00")
        dp_label[R"\vec{\textbf{v}}"].set_color(PINK)
        dp_label[R"\vec{\textbf{w}}"].set_color(YELLOW)
        num = dp_label.make_number_changeable("0.00")
        num.f_always.set_value(get_dp)

        dp_label.fix_in_frame()
        dp_label.to_corner(UL)
        self.add(dp_label)

        # Randomize
        frame.add_ambient_rotation(2 * DEG)
        for n in range(100):
            for vect in vects:
                self.set_vector_randomly(vect, radius)
            self.wait(0.2)

        # Only one random vector
        v2.put_start_and_end_on(ORIGIN, radius * OUT)
        v2.set_fill(opacity=0.5)

        dot = GlowDot()
        dot.set_color(v1.get_color())
        dot_ghosts = Group()

        def project(points):
            points[:, :2] = 0
            return points

        z_line = Line(ORIGIN, OUT)
        z_line.set_stroke(WHITE, 8)
        self.add(dot, dot_ghosts, z_line)

        for n in range(100):
            self.set_vector_randomly(v1, radius)
            z_line.set_points_as_corners([ORIGIN, v1.get_end()[2] * OUT])
            dot.move_to(v1.get_end())
            dot_ghosts.add(dot.copy())
            for ghost in dot_ghosts:
                ghost.set_opacity(ghost.get_opacity() * 0.9)
            self.wait(0.2)

    def set_vector_randomly(self, vect, radius):
        point = radius * normalize(np.random.normal(0, 1, 3))
        vect.put_start_and_end_on(ORIGIN, point)
        return vect


class Distributions(InteractiveScene):
    def construct(self):
        # Test
        axes = Axes(
            (-1, 1, 0.2),
            (0, 1),
            width=6,
            height=4
        )
        axes.add_coordinate_labels(num_decimal_places=1, font_size=16)
        axes.y_axis.set_opacity(0)

        # graph = axes.get_graph(lambda x: np.exp(-5 * x**2))
        # graph = axes.get_graph(lambda x: np.sqrt(1 - x**2))
        # graph = axes.get_graph(lambda x: 0.3 * x**2 + 0.5)
        graph = axes.get_graph(lambda x: 0.5)

        rects = axes.get_riemann_rectangles(graph, dx=0.1)
        rects.set_fill(opacity=0.5)
        rects.set_stroke(WHITE, 1, 0.5)

        self.add(axes, rects)
