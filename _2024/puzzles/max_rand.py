from manim_imports_ext import *


class Randomize(Animation):
    def __init__(self, value_tracker, frequency=8, rand_func=random.random, final_value=None, **kwargs):
        self.value_tracker = value_tracker
        self.rand_func = rand_func
        self.frequency = frequency
        self.final_value = final_value if final_value is not None else rand_func()

        self.last_alpha = 0
        self.running_tally = 0
        super().__init__(value_tracker, **kwargs)

    def interpolate_mobject(self, alpha):
        if not self.new_step(alpha):
            return

        value = self.rand_func() if alpha < 1 else self.final_value
        self.value_tracker.set_value(value)

    def new_step(self, alpha):
        d_alpha = alpha - self.last_alpha
        self.last_alpha = alpha
        self.running_tally += self.frequency * d_alpha * self.run_time
        if self.running_tally > 1:
            self.running_tally = self.running_tally % 1
            return True
        return False


class TrackingDots(Animation):
    def __init__(self, point_func, fade_factor=0.95, radius=0.25, color=YELLOW, **kwargs):
        self.point_func = point_func
        self.fade_factor = fade_factor
        self.dots = GlowDot(point_func(), color=color, radius=radius)
        kwargs.update(remover=True)
        super().__init__(self.dots, **kwargs)

    def interpolate_mobject(self, alpha):
        opacities = self.dots.get_opacities()
        point = self.point_func()
        if not np.isclose(self.dots.get_end(), point).all():
            self.dots.add_point(point)
            opacities = np.hstack([opacities, [1]])
        opacities *= self.fade_factor
        self.dots.set_opacity(opacities)


def get_random_var_label_group(axis, label_name, color=GREY, initial_value=None, font_size=36, direction=None):
    if initial_value is None:
        initial_value = random.uniform(*axis.x_range[:2])
    tracker = ValueTracker(initial_value)
    tip = ArrowTip(angle=90 * DEGREES)
    tip.set_height(0.15)
    tip.set_fill(color)
    tip.rotate(-axis.get_angle())
    if direction is None:
        direction = np.round(rotate_vector(UP, -axis.get_angle()), 1)
    tip.add_updater(lambda m: m.move_to(axis.n2p(tracker.get_value()), direction))
    label = Tex(label_name, font_size=font_size)
    label.set_color(color)
    label.set_backstroke(BLACK, 5)
    label.always.next_to(tip, -direction, buff=0.1)

    return Group(tracker, tip, label)


class MaxProcess(InteractiveScene):
    def construct(self):
        # Set up intervals
        intervals = VGroup(UnitInterval() for _ in range(3))
        intervals.set_width(3)
        intervals.arrange(DOWN, buff=2.5)
        intervals.shift(2 * LEFT)
        intervals[1].shift(0.5 * UP)
        for interval in intervals:
            interval.add_numbers(np.arange(0, 1.1, 0.2), font_size=16, buff=0.1, direction=UP)
            interval.numbers.set_opacity(0.75)

        colors = [BLUE, YELLOW, GREEN]
        x1_group, x2_group, max_group = groups = Group(
            get_random_var_label_group(interval, "", color=color)
            for interval, color in zip(intervals, colors)
        )
        x1_tracker, x1_tip, x1_label = x1_group
        x2_tracker, x2_tip, x2_label = x2_group
        max_tracker, max_tip, max_label = max_group
        max_tracker.add_updater(lambda m: m.set_value(max(x1_tracker.get_value(), x2_tracker.get_value())))

        self.add(intervals)
        self.add(groups)

        # Add labels
        tex_to_color = {"x_1": BLUE, "x_2": YELLOW}
        labels = VGroup(
            Tex(tex + R"\rightarrow 0.00", t2c=tex_to_color)
            for tex in [
                R"x_1 = \text{rand}()",
                R"x_2 = \text{rand}()",
                R"\max(x_1, x_2)",
            ]
        )
        for label, group, interval in zip(labels, groups, intervals):
            label.next_to(interval, RIGHT, buff=0.5)
            num = label.make_number_changeable("0.00")
            num.tracker = group[0]
            num.add_updater(lambda m: m.set_value(m.tracker.get_value()))

        self.add(labels)

        # Add rectangles
        top_rect = SurroundingRectangle(intervals[:2], buff=0.25)
        top_rect.stretch(1.1, 1)
        top_rect.set_stroke(WHITE, 2)
        top_rect.set_fill(GREY_E, 1)

        arrow = Vector(1.5 * DOWN, thickness=5)
        arrow.next_to(top_rect, DOWN)
        arrow_label = Text("max", font_size=60)
        arrow_label.next_to(arrow, RIGHT)

        self.add(top_rect, intervals, groups)
        self.add(arrow, arrow_label)

        # Line
        def get_line():
            x1 = x1_tracker.get_value()
            x2 = x2_tracker.get_value()
            tip = x1_tip if x1 > x2 else x2_tip
            line = DashedLine(max_tip.get_top(), tip.get_top())
            line.set_stroke(GREY, 2, opacity=0.5)
            return line

        line = always_redraw(get_line)
        self.add(line)

        # Animate
        self.play(
            Randomize(x1_tracker, frequency=4, run_time=30),
            Randomize(x2_tracker, frequency=4, run_time=30),
            TrackingDots(x1_tip.get_top, color=BLUE),
            TrackingDots(x2_tip.get_top, color=YELLOW),
            TrackingDots(max_tip.get_top, color=GREEN),
        )


class SqrtProcess(InteractiveScene):
    def construct(self):
        # A fair bit of copy pasting from above
        # Set up intervals
        intervals = VGroup(UnitInterval() for _ in range(2))
        intervals.set_width(3)
        intervals.arrange(DOWN, buff=3.5)
        intervals.shift(2 * LEFT)
        for interval in intervals:
            interval.add_numbers(np.arange(0, 1.1, 0.2), font_size=16, buff=0.1, direction=UP)
            interval.numbers.set_opacity(0.75)

        colors = [BLUE, TEAL]
        x_group, sqrt_group = groups = Group(
            get_random_var_label_group(interval, "", color=color)
            for interval, color in zip(intervals, colors)
        )
        x_tracker, x_tip, x_label = x_group
        sqrt_tracker, sqrt_tip, sqrt_label = sqrt_group
        sqrt_tracker.add_updater(lambda m: m.set_value(math.sqrt(x_tracker.get_value())))

        self.add(intervals)
        self.add(groups)

        # Add labels
        tex_to_color = {"x": BLUE}
        labels = VGroup(
            Tex(tex + R"\rightarrow 0.00", t2c=tex_to_color)
            for tex in [
                R"x = \text{rand}()",
                R"\sqrt{x}",
            ]
        )
        for label, group, interval in zip(labels, groups, intervals):
            label.next_to(interval, RIGHT, buff=0.5)
            num = label.make_number_changeable("0.00")
            num.tracker = group[0]
            num.add_updater(lambda m: m.set_value(m.tracker.get_value()))

        self.add(labels)

        # Big arrow
        arrow = Arrow(*intervals, buff=0.5, thickness=5)
        label = Text(R"sqrt", font_size=60)
        label.next_to(arrow, RIGHT)

        self.add(arrow, label)

        # Animate
        self.play(
            Randomize(x_tracker, frequency=4, run_time=30),
            TrackingDots(x_tip.get_top, color=colors[0]),
            TrackingDots(sqrt_tip.get_top, color=colors[1]),
        )


class SquareAndSquareRoot(InteractiveScene):
    def construct(self):
        # Test
        lines = VGroup(
            Tex(R"\left(\frac{1}{2}\right)^2 = \frac{1}{4}"),
            Tex(R"\sqrt{\frac{1}{4}} = \frac{1}{2}"),
        )
        lines.scale(1.5)

        self.play(FadeIn(lines[0]))
        self.wait()
        self.play(TransformMatchingStrings(
            *lines,
            matched_pairs=[(lines[0]["2"][1], lines[1][R"\sqrt"][0])],
            path_arc=PI / 2,
        ))


class GawkAtEquivalence(InteractiveScene):
    def construct(self):
        # Test
        expr = VGroup(
            Text("max(rand(), rand())"),
            Tex(R"\updownarrow", font_size=90),
            Text("sqrt(rand())"),
        )
        expr.arrange(DOWN)
        expr.to_edge(UP)
        randy = Randolph()
        randy.to_edge(DOWN)

        self.add(expr)
        self.play(randy.change("confused", expr))
        self.play(Blink(randy))
        self.wait()
        self.play(randy.change("maybe", expr))
        self.play(Blink(randy))
        self.wait()


class VisualizeMaxOfPairCDF(InteractiveScene):
    def construct(self):
        # Setup axes and trackers
        axes = Axes((0, 1, 0.1), (0, 1, 0.1), width=6, height=6)
        axes.add_coordinate_labels(
            (0, 0.5, 1.0), (0.5, 1.0),
            excluding=[],
            num_decimal_places=1,
            buff=0.15,
            font_size=16
        )
        x1_tracker, x1_tip, x1_label = get_random_var_label_group(axes.x_axis, "x_1", BLUE, 0.7)
        x2_tracker, x2_tip, x2_label = get_random_var_label_group(axes.y_axis, "x_2", YELLOW, 0.3)
        tex_to_color = {"x_1": BLUE, "x_2": YELLOW}

        # Show x1
        self.add(axes.x_axis)
        self.play(
            FadeIn(x1_tip),
            FadeIn(x1_label),
        )
        self.wait()

        self.play(
            Randomize(x1_tracker, run_time=8),
            TrackingDots(x1_tip.get_top, run_time=10, color=BLUE),
        )
        self.wait()

        # Highlight a given range
        eq = Tex(R"P(a < x_1 < b) = b - a")
        rect = SurroundingRectangle(axes.x_axis.ticks)
        rect.set_fill(BLUE, 0.5)
        rect.set_stroke(WHITE, 0)
        rect.stretch(0.25, 0)
        rect.move_to(x1_tip.get_top())
        eq.next_to(rect, UP)
        eq.shift((rect.get_x() - eq["x_1"].get_x()) * RIGHT)

        self.play(
            FadeIn(rect),
            FadeIn(eq, 0.25 * UP)
        )
        self.wait(2)
        self.play(
            FadeOut(rect),
            FadeOut(eq),
        )

        # Show x2
        self.play(
            FadeIn(axes.y_axis),
            Write(x2_tip),
            Write(x2_label),
        )
        self.wait()
        self.play(
            Randomize(x2_tracker, run_time=8, final_value=0.48),
            TrackingDots(x2_tip.get_right, run_time=10, color=YELLOW),
        )

        # Show the pair inside the square
        def get_xy_point():
            return axes.c2p(x1_tracker.get_value(), x2_tracker.get_value())

        v_line = Line().set_stroke(WHITE, 1, 0.5)
        h_line = Line().set_stroke(WHITE, 1, 0.5)
        v_line.f_always.put_start_and_end_on(x1_tip.get_top, get_xy_point)
        h_line.f_always.put_start_and_end_on(x2_tip.get_right, get_xy_point)

        xy_dot = Dot(radius=0.05)
        xy_dot.f_always.move_to(get_xy_point)
        xy_dot.update()

        coord_label = Tex("(x_1, x_2)", t2c=tex_to_color, font_size=36)
        coord_label.always.next_to(xy_dot, UR, SMALL_BUFF)
        coord_label.update()

        self.play(
            ShowCreation(v_line),
            ShowCreation(h_line),
            FadeIn(xy_dot),
            FadeIn(coord_label[0::3]),
            FadeTransform(x1_label.copy(), coord_label["x_1"], remover=True),
            FadeTransform(x2_label.copy(), coord_label["x_2"], remover=True),
        )
        self.add(coord_label)
        self.wait()

        # Randomize it
        big_square = Square()
        big_square.set_fill(GREY_E, 1)
        big_square.set_stroke(GREY_D, 1)
        big_square.replace(Line(axes.c2p(0, 0), axes.c2p(1, 1)))

        self.add(big_square, *self.mobjects)
        self.play(
            FadeIn(big_square, run_time=2),
            Randomize(x2_tracker, run_time=6, frequency=8, final_value=0.69),
            Randomize(x1_tracker, run_time=6, frequency=8, final_value=0.42),
            TrackingDots(get_xy_point, run_time=9, fade_factor=0.97, color=GREEN),
        )

        # Bring up max(x1, x2), show where its true
        max_expr = Tex(R"\max(x_1, x_2) = 0.7", t2c=tex_to_color)
        max_expr.to_corner(UR)

        max_lines = VGroup(
            Line(axes.c2p(0.7, 0.7), axes.c2p(0.7, 0)),
            Line(axes.c2p(0.7, 0.7), axes.c2p(0, 0.7)),
        )
        max_lines.set_stroke(GREEN, 3)

        self.play(FadeIn(max_expr, UP))
        self.play(
            x1_tracker.animate.set_value(0.7),
            x2_tracker.animate.set_value(0.7),
        )
        self.wait()
        for tracker, line in zip([x2_tracker, x1_tracker], max_lines):
            self.add(line, xy_dot)
            self.play(
                tracker.animate.set_value(0),
                ShowCreation(line),
                run_time=3
            )
            self.play(tracker.animate.set_value(0.7), run_time=3)
            self.wait()

        # Ask about probability
        prob_eq = Tex(R"P(\max(x_1, x_2) = 0.7)", t2c=tex_to_color)
        prob_ineq = Tex(R"P(\max(x_1, x_2) \le 0.7)", t2c=tex_to_color)
        gen_prob_ineq = Tex(R"P(\max(x_1, x_2) \le R)", t2c=tex_to_color)
        cdf_expr = Tex(R"P(\max(x_1, x_2) \le R) = R^2", t2c=tex_to_color)
        for tex in [prob_eq, prob_ineq, gen_prob_ineq, cdf_expr]:
            tex.to_corner(UR)
            tex.shift(0.5 * UP)

        words = Text("Not helpful")
        words.set_color(RED)
        words.next_to(prob_eq, DOWN)

        self.play(
            TransformMatchingStrings(max_expr, prob_eq),
            self.frame.animate.move_to(0.5 * UP),
            run_time=1,
        )
        self.wait()
        self.play(Write(words))
        self.wait()
        self.play(
            max_lines.animate.set_stroke(width=0.1).set_anim_args(rate_func=there_and_back),
            run_time=4
        )

        # Go from P(x = r) to P(x <= r)
        eq_rect = SurroundingRectangle(prob_eq["="])
        eq_rect.set_color(RED)

        inner_lines = VGroup(
            max_lines.copy().scale(sf, about_point=axes.get_origin()).set_stroke(width=2, opacity=0.5)
            for sf in np.linspace(1, 0, 100)
        )
        inner_square = Square()
        inner_square.set_fill(GREEN, 0.35)
        inner_square.set_stroke(GREEN, 0)
        inner_square.replace(max_lines)

        self.play(FadeTransform(words, eq_rect))
        self.wait()
        self.play(
            TransformMatchingStrings(prob_eq, prob_ineq, key_map={"=": R"\le"}),
            eq_rect.animate.surround(prob_ineq[R"\le"]),
            run_time=1
        )
        self.play(
            LaggedStart(
                (ReplacementTransform(max_lines.copy().set_stroke(opacity=0), mlc)
                for mlc in inner_lines),
                lag_ratio=0.01,
                run_time=8
            ),
            Animation(xy_dot),
            FadeOut(eq_rect),
        )
        self.add(inner_lines, inner_square, h_line, v_line, xy_dot, coord_label)
        self.play(
            FadeIn(inner_square),
            FadeOut(inner_lines),
        )
        self.wait()

        self.play(
            Randomize(x1_tracker, run_time=4, frequency=8, final_value=0.38),
            Randomize(x2_tracker, run_time=4, frequency=8, final_value=0.42),
            TrackingDots(get_xy_point, run_time=6, fade_factor=0.97, color=GREEN),
        )

        # Describe CDF
        self.play(TransformMatchingStrings(prob_ineq, gen_prob_ineq, key_map={"0.7": "R"}, run_time=1))
        self.play(TransformMatchingStrings(gen_prob_ineq, cdf_expr, run_time=1))
        self.play(
            VGroup(inner_square, max_lines).animate.scale(0.5, about_edge=DL).set_anim_args(rate_func=there_and_back),
            run_time=3
        )
        self.wait()

        # Make way
        shift_value = 3 * RIGHT
        self.play(
            cdf_expr.animate.shift(5 * RIGHT + UP),
            self.frame.animate.shift(4.5 * RIGHT).scale(1.2),
        )

        # Comapare to the sqrt case
        sqrt_lines = VGroup(
            Tex(R"P(\sqrt{x_1} \le R)", t2c=tex_to_color),
            Tex(R"P(x_1 \le R^2)", t2c=tex_to_color),
            Tex(R"= R^2", t2c=tex_to_color),
        )
        sqrt_lines.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        sqrt_lines.next_to(cdf_expr, DOWN, buff=2.0, aligned_edge=LEFT)
        sqrt_lines[2].next_to(sqrt_lines[1], RIGHT)
        mid_eq = Tex("=").rotate(90 * DEGREES).move_to(sqrt_lines[:2])

        self.play(Write(sqrt_lines[0]))
        self.wait()
        self.play(
            TransformMatchingStrings(sqrt_lines[0].copy(), sqrt_lines[1]),
            Write(mid_eq),
        )
        self.wait()
        self.play(Write(sqrt_lines[2]))
        self.wait()

        # Go to three dimensions
        frame = self.frame
        axes3d = ThreeDAxes((0, 1, 0.1), (0, 1, 0.1), (0, 1, 0.1))
        axes3d.set_width(axes.x_axis.get_length())
        axes3d.shift(axes.get_origin() - axes3d.get_origin())
        z_axis = axes3d.z_axis
        z_axis.set_width(axes.x_axis.ticks.get_height(), stretch=True)
        frame.clear_updaters()
        frame.add_ambient_rotation(-1 * DEGREES)

        # self.add(cdf_expr, Point())
        self.remove(cdf_expr, sqrt_lines, mid_eq)
        frame.center()
        self.play(
            # FadeOut(sqrt_lines),
            # FadeOut(mid_eq),
            # cdf_expr.animate.fix_in_frame().center().to_edge(UP),
            # FadeOut(cdf_expr),
            Write(axes3d.z_axis),
            frame.animate.reorient(26, 74, 0, (-0.42, 0.83, 1.66), 10.11),
            run_time=3
        )

        # Add cube
        tex_to_color["x_3"] = RED
        new_cdf_expr = Tex(R"P(\max(x_1, x_2, x_3) \le R) = R^3", t2c=tex_to_color)
        new_cdf_expr.fix_in_frame()
        new_cdf_expr.move_to(cdf_expr, RIGHT)

        width = inner_square.get_width()
        cube = VGroup(
            inner_square.copy().shift(width * OUT),
            inner_square.copy().rotate(PI / 2, LEFT, about_edge=UP),
            inner_square.copy().rotate(PI / 2, UP, about_edge=RIGHT),
            inner_square.copy().rotate(PI / 2, DOWN, about_edge=LEFT),
            inner_square.copy().rotate(PI / 2, RIGHT, about_edge=DOWN),
        )
        cube.set_stroke(GREEN, 3)
        cube.set_fill(GREEN, 0.2)
        cube.save_state()
        cube.stretch(0, 2, about_edge=IN)
        cube.set_fill(opacity=0)
        cube[0].set_fill(opacity=0.35)

        x3_tracker, x3_tip, x3_label = get_random_var_label_group(z_axis, "x_3", color=RED, direction=RIGHT)
        x3_tracker.set_value(0.2)
        VGroup(x3_tip, x3_label).rotate(PI / 2, RIGHT)
        x3_tip.rotate(PI / 2, UP)

        def get_xzy_point():
            return axes3d.c2p(
                x1_tracker.get_value(),
                x2_tracker.get_value(),
                x3_tracker.get_value(),
            )

        xyz_dot = TrueDot().make_3d()
        xyz_dot.add_updater(lambda m: m.move_to(get_xzy_point()))

        self.remove(max_lines)
        self.remove(inner_square)
        self.play(
            FadeIn(x3_tip),
            FadeIn(x3_label),
            Restore(cube),
            # FadeTransform(cdf_expr, new_cdf_expr, time_span=(1, 2)),
            FadeOut(h_line),
            FadeOut(v_line),
            FadeOut(coord_label),
            FadeTransform(Group(xy_dot), xyz_dot),
            run_time=3
        )

        self.play(
            Randomize(x1_tracker, run_time=16),
            Randomize(x2_tracker, run_time=16),
            Randomize(x3_tracker, run_time=16),
            TrackingDots(get_xzy_point, run_time=20, color=GREEN, radius=0.15),
        )


class MaxOfThreeTex(InteractiveScene):
    def construct(self):
        # Test
        expr = TexText(R"max(rand(), rand(), rand()) $\leftrightarrow$ rand()$^{1 / 3}$")
        expr["rand()"].set_submobject_colors_by_gradient(BLUE, YELLOW, RED, BLUE)
        self.play(Write(expr))
        self.wait()


class Arrows(InteractiveScene):
    def construct(self):
        arrows = Vector(DOWN, thickness=5).replicate(3)
        arrows.arrange(RIGHT, buff=1.0)
        arrows.set_fill(YELLOW)
        self.play(LaggedStartMap(GrowArrow, arrows))
        self.wait()
