from manim_imports_ext import *


def get_coef_colors(n_coefs=3):
    return [
        interpolate_color_by_hsl(TEAL, RED, a)
        for a in np.linspace(0, 1, n_coefs)
    ]


class SrpingMassSystem(VGroup):
    def __init__(
        self,
        x0=0,
        v0=0,
        k=3,
        mu=0.1,
        equilibrium_length=7,
        equilibrium_position=ORIGIN,
        direction=RIGHT,
        spring_stroke_color=GREY_B,
        spring_stroke_width=2,
        spring_radius=0.25,
        n_spring_curls=8,
        mass_width=1.0,
        mass_color=BLUE_E,
        mass_label="m",
        external_force=None,
    ):
        super().__init__()
        self.equilibrium_position = equilibrium_position
        self.fixed_spring_point = equilibrium_position - (equilibrium_length - 0.5 * mass_width) * direction
        self.direction = direction
        self.rot_off_horizontal = angle_between_vectors(RIGHT, direction)
        self.mass = self.get_mass(mass_width, mass_color, mass_label)
        self.spring = self.get_spring(spring_stroke_color, spring_stroke_width, n_spring_curls, spring_radius)
        self.add(self.spring, self.mass)

        self.k = k
        self.mu = mu
        self.set_x(x0)
        self.velocity = v0

        self.external_force = external_force

        self._is_running = True
        self.add_updater(lambda m, dt: m.time_step(dt))

    def get_spring(self, stroke_color, stroke_width, n_curls, radius):
        spring = ParametricCurve(
            lambda t: [t, -radius * math.sin(TAU * t), radius * math.cos(TAU * t)],
            t_range=(0, n_curls, 1e-2),
            stroke_color=stroke_color,
            stroke_width=stroke_width,
        )
        spring.rotate(self.rot_off_horizontal)
        return spring

    def get_mass(self, mass_width, mass_color, mass_label):
        mass = Square(mass_width)
        mass.set_fill(mass_color, 1)
        mass.set_stroke(WHITE, 1)
        mass.set_shading(0.1, 0.1, 0.1)
        label = Tex(mass_label)
        label.set_max_width(0.5 * mass.get_width())
        label.move_to(mass)
        mass.add(label)
        mass.label = label
        return mass

    def set_x(self, x):
        self.mass.move_to(self.equilibrium_position + x * self.direction)
        spring_width = SMALL_BUFF + get_norm(self.mass.get_left() - self.fixed_spring_point)
        self.spring.rotate(-self.rot_off_horizontal)
        self.spring.set_width(spring_width, stretch=True)
        self.spring.rotate(self.rot_off_horizontal)
        self.spring.move_to(self.fixed_spring_point, -self.direction)

    def get_x(self):
        return (self.mass.get_center() - self.equilibrium_position)[0]

    def time_step(self, delta_t, dt_size=1e-3):
        if not self._is_running:
            return
        if delta_t == 0:
            return

        state = [self.get_x(), self.velocity]
        sub_steps = max(int(delta_t / dt_size), 1)
        true_dt = delta_t / sub_steps
        for _ in range(sub_steps):
            # ODE
            x, v = state
            state += np.array([v, self.get_force(x, v)]) * true_dt

        self.set_x(state[0])
        self.velocity = state[1]

    def pause(self):
        self._is_running = False

    def unpause(self):
        self._is_running = True

    def set_k(self, k):
        self.k = k
        return self

    def set_mu(self, mu):
        self.mu = mu
        return self

    def get_velocity(self):
        return self.velocity

    def set_velocity(self, velocity):
        self.velocity = velocity
        return self

    def get_velocity_vector(self, scale_factor=0.5, thickness=3.0, v_offset=-0.25, color=GREEN):
        """Get a vector showing the mass's velocity"""
        vector = Vector(RIGHT, fill_color=color)
        v_shift = v_offset * UP
        vector.add_updater(lambda m: m.put_start_and_end_on(
            self.mass.get_center() + v_shift,
            self.mass.get_center() + v_shift + scale_factor * self.velocity * RIGHT
        ))
        return vector

    def get_force_vector(self, scale_factor=0.5, thickness=3.0, v_offset=-0.25, color=RED):
        """Get a vector showing the mass's velocity"""
        vector = Vector(RIGHT, fill_color=color)
        v_shift = v_offset * UP
        vector.add_updater(lambda m: m.put_start_and_end_on(
            self.mass.get_center() + v_shift,
            self.mass.get_center() + v_shift + scale_factor * self.get_force(self.get_x(), self.velocity) * RIGHT
        ))
        return vector

    def add_external_force(self, func):
        self.external_force = func

    def get_force(self, x, v):
        force = -self.k * x - self.mu * v
        if self.external_force is not None:
            force += self.external_force()
        return force


class BasicSpringScene(InteractiveScene):
    def construct(self):
        # Add spring, give some initial oscillation
        spring = SrpingMassSystem(
            x0=2,
            mu=0.1,
            k=3,
            equilibrium_position=2 * LEFT,
            equilibrium_length=5,
        )
        self.add(spring)

        # Label on a number line
        number_line = NumberLine(x_range=(-4, 4, 1))
        number_line.next_to(spring.equilibrium_position, DOWN, buff=2.0)
        number_line.add_numbers(font_size=24)

        # Dashed line from mass to number line
        dashed_line = DashedLine(
            spring.mass.get_bottom(),
            number_line.n2p(spring.get_x()),
            stroke_color=GREY,
            stroke_width=2
        )
        dashed_line.always.match_x(spring.mass)

        # Arrow tip on number line
        arrow_tip = ArrowTip(length=0.2, width=0.1)
        arrow_tip.rotate(-90 * DEG)  # Point downward
        arrow_tip.set_fill(TEAL)
        arrow_tip.add_updater(lambda m: m.move_to(number_line.n2p(spring.get_x()), DOWN))

        x_label = Tex("x = 0.00", font_size=24)
        x_number = x_label.make_number_changeable("0.00")
        x_number.add_updater(lambda m: m.set_value(spring.get_x()))
        x_label.add_updater(lambda m: m.next_to(arrow_tip, UR, buff=0.1))

        # Ambient playing, fade in labels
        self.wait(2)
        self.play(
            VFadeIn(number_line),
            VFadeIn(dashed_line),
            VFadeIn(arrow_tip),
            VFadeIn(x_label),
        )
        self.wait(7)

        # (For an insertion)
        if False:
            x_label_arrow = Vector(1.5 * DL, thickness=8)
            x_label_arrow.set_fill(YELLOW)
            x_label_arrow.always.move_to(arrow_tip, DL).shift(2 * RIGHT + 0.75 * UP)
            self.play(
                VFadeIn(x_label_arrow, time_span=(1, 2)),
                x_label.animate.scale(2),
                run_time=2
            )
            self.wait(8)

        # Show velocity
        x_color, v_color, a_color = [interpolate_color_by_hsl(TEAL, RED, a) for a in np.linspace(0, 1, 3)]
        v_vect = spring.get_velocity_vector(color=v_color, scale_factor=0.25)
        a_vect = spring.get_force_vector(color=a_color, scale_factor=0.25)
        a_vect.add_updater(lambda m: m.shift(v_vect.get_end() - m.get_start()))

        self.play(VFadeIn(v_vect))
        self.wait(5)
        self.play(VFadeIn(a_vect))
        self.wait(8)
        self.wait_until(lambda: spring.velocity <= 0)

        # Show the force law
        self.remove(v_vect)
        a_vect.remove_updater(a_vect.get_updaters()[-1])
        spring.pause()

        self.wait()
        for x in range(2, 5):
            self.play(spring.animate.set_x(x))
            self.wait()

        # Back and forth
        t_tracker = ValueTracker(0)
        self.play(
            UpdateFromAlphaFunc(
                spring,
                lambda m, a: m.set_x(4 * math.cos(2 * TAU * a)),
                rate_func=linear,
                run_time=8,
            )
        )

        # Ambient springing
        spring.unpause()
        spring.set_mu(0.25)
        self.wait(15)

        # Show the solution graph
        frame = self.frame

        time_tracker = ValueTracker(0)
        time_tracker.add_updater(lambda m, dt: m.increment_value(dt))

        axes = Axes(
            x_range=(0, 20, 1),
            y_range=(-2, 2, 1),
            width=12,
            height=3,
            axis_config={"stroke_color": GREY}
        )
        axes.next_to(spring, UP, LARGE_BUFF)
        axes.align_to(number_line, LEFT)

        x_axis_label = Text("Time", font_size=24).next_to(axes.x_axis, RIGHT, buff=0.1)
        y_axis_label = Tex("x(t)", font_size=24).next_to(axes.y_axis.get_top(), RIGHT, buff=0.1)
        axes.add(x_axis_label)
        axes.add(y_axis_label)

        tracking_point = Point()
        tracking_point.add_updater(lambda p: p.move_to(
            axes.c2p(time_tracker.get_value(), spring.get_x())
        ))

        position_graph = TracedPath(
            tracking_point.get_center,
            stroke_color=BLUE,
            stroke_width=3,
        )

        spring.pause()
        spring.set_velocity(0)
        self.play(
            frame.animate.reorient(0, 0, 0, (2.88, 1.88, 0.0), 12.48),
            FadeIn(axes),
            VFadeOut(a_vect),
            spring.animate.set_x(2),
        )
        self.add(tracking_point, position_graph, time_tracker)
        spring.unpause()
        self.wait(20)
        position_graph.clear_updaters()
        self.wait(20)


class DampingForceDemo(InteractiveScene):
    def construct(self):
        # Create spring-mass system with invisible spring and damping only
        spring_system = SrpingMassSystem(
            x0=-4,
            v0=2,
            k=0,
            mu=0.3,
            equilibrium_position=ORIGIN,
            equilibrium_length=6,
            mass_width=0.8,
            mass_color=BLUE_E,
            mass_label="m",
        )
        spring_system.spring.set_opacity(0)
        self.add(spring_system)

        # Create velocity vector
        v_color = interpolate_color_by_hsl(TEAL, RED, 0.5)
        velocity_vector = spring_system.get_velocity_vector(color=v_color, scale_factor=0.8)

        velocity_label = Tex(R'\vec{\textbf{v}}', font_size=24)
        velocity_label.set_color(v_color)
        velocity_label.always.next_to(velocity_vector, RIGHT, buff=SMALL_BUFF)
        velocity_label.add_updater(lambda m: m.set_max_width(0.5 * velocity_vector.get_width()))

        # Create damping force vector
        damping_vector = spring_system.get_velocity_vector(scale_factor=-0.5, color=RED, v_offset=-0.5)
        damping_label = Tex(R"-\mu v", fill_color=RED, font_size=24)
        damping_label.always.next_to(damping_vector, DOWN, SMALL_BUFF)

        # Add vectors and labels
        self.add(velocity_vector, velocity_label)
        self.add(damping_vector, damping_label)

        # Let the system evolve
        self.wait(15)


class SolveDampedSpringEquation(InteractiveScene):
    def construct(self):
        # Show x and its derivatives
        pos, vel, acc = funcs = VGroup(
            Tex(R"x(t)"),
            Tex(R"x'(t)"),
            Tex(R"x''(t)"),
        )
        funcs.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)

        labels = VGroup(
            Text("Position").set_color(BLUE),
            Text("Velocity").set_color(RED),
            Text("Acceleration").set_color(YELLOW),
        )
        colors = get_coef_colors()
        for line, label, color in zip(funcs, labels, colors):
            label.set_color(color)
            label.next_to(line, RIGHT, MED_LARGE_BUFF)
            label.align_to(labels[0], LEFT)

        VGroup(funcs, labels).to_corner(UR)

        arrows = VGroup()
        for l1, l2 in zip(funcs, funcs[1:]):
            arrow = Line(l1.get_left(), l2.get_left(), path_arc=150 * DEG, buff=0.2)
            arrow.add_tip(width=0.2, length=0.2)
            arrow.set_color(GREY_B)
            ddt = Tex(R"\frac{d}{dt}", font_size=30)
            ddt.set_color(GREY_B)
            ddt.next_to(arrow, LEFT, SMALL_BUFF)
            arrow.add(ddt)
            arrows.add(arrow)

        self.play(Write(funcs[0]), Write(labels[0]))
        self.wait()
        for func1, func2, label1, label2, arrow in zip(funcs, funcs[1:], labels, labels[1:], arrows):
            self.play(LaggedStart(
                GrowFromPoint(arrow, arrow.get_corner(UR), path_arc=30 * DEG),
                TransformFromCopy(func1, func2, path_arc=30 * DEG),
                FadeTransform(label1.copy(), label2),
                lag_ratio=0.1
            ))
            self.wait()

        deriv_group = VGroup(funcs, labels, arrows)

        # Show F=ma
        t2c = {
            "x(t)": colors[0],
            "x'(t)": colors[1],
            "x''(t)": colors[2],
        }
        equation1 = Tex(R"{m} x''(t) = -k x(t) - \mu x'(t)", t2c=t2c)
        equation1.to_corner(UL)

        ma = equation1["{m} x''(t)"][0]
        kx = equation1["-k x(t)"][0]
        mu_v = equation1[R"- \mu x'(t)"][0]
        rhs = VGroup(kx, mu_v)

        ma_brace, kx_brace, mu_v_brace = braces = VGroup(
            Brace(part, DOWN, buff=SMALL_BUFF)
            for part in [ma, kx, mu_v]
        )
        label_texs = [R"\textbf{F}", R"\text{Spring force}", R"\text{Damping}"]
        for brace, label_tex in zip(braces, label_texs):
            brace.add(brace.get_tex(label_tex))

        self.play(TransformFromCopy(acc, ma[1:], path_arc=-45 * DEG))
        self.play(LaggedStart(
            GrowFromCenter(ma_brace),
            Write(ma[0]),
            run_time=1,
            lag_ratio=0.1
        ))
        self.wait()
        self.play(LaggedStart(
            Write(equation1["= -k"][0]),
            FadeTransformPieces(ma_brace, kx_brace),
            TransformFromCopy(pos, equation1["x(t)"][0], path_arc=-45 * DEG),
        ))
        self.wait()
        self.play(LaggedStart(
            FadeTransformPieces(kx_brace, mu_v_brace),
            Write(equation1[R"- \mu"][0]),
            TransformFromCopy(vel, equation1["x'(t)"][0], path_arc=-45 * DEG),
        ))
        self.wait()
        self.play(FadeOut(mu_v_brace))

        # Rearrange
        equation2 = Tex(R"{m} x''(t) + \mu x'(t) + k x(t) = 0", t2c=t2c)
        equation2.move_to(equation1, UL)

        self.play(TransformMatchingTex(equation1, equation2, path_arc=45 * DEG))
        self.wait()

        # Hypothesis of e^st
        t2c = {"s": YELLOW, "x(t)": TEAL}
        hyp_word, hyp_tex = hypothesis = VGroup(
            Text("Hypothesis: "),
            Tex("x(t) = e^{st}", t2c=t2c),
        )
        hypothesis.arrange(RIGHT)
        hypothesis.to_corner(UR)
        sub_hyp_word = TexText(R"(For some $s$)", t2c={"$s$": YELLOW}, font_size=36, fill_color=GREY_B)
        sub_hyp_word.next_to(hyp_tex, DOWN)

        self.play(LaggedStart(
            FadeTransform(pos.copy(), hyp_tex[:4], path_arc=45 * DEG, remover=True),
            FadeOut(deriv_group),
            Write(hyp_word, run_time=1),
            Write(hyp_tex[4:], time_span=(0.5, 1.5)),
        ))
        self.add(hypothesis)
        self.wait()
        self.play(FadeIn(sub_hyp_word, 0.25 * DOWN))
        self.wait()

        # Plug it in
        t2c["s"] = YELLOW
        equation3 = Tex(R"{m} s^2 e^{st} + \mu s e^{st} + k e^{st} = 0", t2c=t2c)
        equation3.next_to(equation2, DOWN, LARGE_BUFF)
        pos_parts = VGroup(equation2["x(t)"][0], equation3["e^{st}"][-1])
        vel_parts = VGroup(equation2["x'(t)"][0], equation3["s e^{st}"][0])
        acc_parts = VGroup(equation2["x''(t)"][0], equation3["s^2 e^{st}"][0])
        matched_parts = VGroup(pos_parts, vel_parts, acc_parts)

        pos_rect, vel_rect, acc_rect = rects = VGroup(
            SurroundingRectangle(group[0], buff=0.05).set_stroke(group[0][0].get_color(), 1)
            for group in matched_parts
        )

        pos_arrow, vel_arrow, acc_arrow = arrows = VGroup(
            Arrow(*pair, buff=0.1)
            for pair in matched_parts
        )

        for rect, arrow, pair in zip(rects, arrows, matched_parts):
            self.play(ShowCreation(rect))
            self.play(
                GrowArrow(arrow),
                FadeTransform(pair[0].copy(), pair[1]),
                rect.animate.surround(pair[1]),
            )
            self.wait()
        self.play(
            LaggedStart(
                (TransformFromCopy(equation2[tex], equation3[tex])
                for tex in ["{m}", "+", "k", R"\mu", "=", "0"]),
                lag_ratio=0.05,
            ),
        )
        self.wait()
        self.play(FadeOut(arrows, lag_ratio=0.1), FadeOut(rects, lag_ratio=0.1))

        # Solve for s
        key_syms = ["s", "m", R"\mu", "k"]
        equation4, equation5, equation6 = new_equations = VGroup(
            Tex(R"e^{st} \left( ms^2 + \mu s + k \right) = 0", t2c=t2c),
            Tex(R"ms^2 + \mu s + k = 0", t2c=t2c),
            Tex(R"{s} = {{-\mu \pm \sqrt{\mu^2 - 4mk}} \over 2m}", isolate=key_syms)
        )
        rhs = equation6[2:]
        rhs.set_width(equation5.get_width() - equation6[:2].get_width(), about_edge=LEFT)
        equation6.refresh_bounding_box()
        equation6["{s}"].set_color(YELLOW)
        equation6.scale(1.25, about_edge=LEFT)

        new_equations.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        new_equations.move_to(equation3, UL)
        equation4 = new_equations[0]

        exp_rect = SurroundingRectangle(equation4[R"e^{st}"])
        exp_rect.set_stroke(YELLOW, 2)
        ne_0 = VGroup(Tex(R"\ne").rotate(90 * DEG), Integer(0))
        ne_0.arrange(DOWN).next_to(exp_rect, DOWN)

        self.play(
            TransformMatchingTex(
                equation3,
                equation4,
                matched_keys=[R"e^{st}"],
                run_time=1.5,
                path_arc=30 * DEG
            )
        )
        self.wait(0.5)
        self.play(ShowCreation(exp_rect))
        self.wait()
        self.play(Write(ne_0))
        self.wait()
        self.play(FadeOut(ne_0))
        self.play(
            *(
                TransformFromCopy(equation4[key], equation5[key])
                for key in [R"ms^2 + \mu s + k", "= 0"]
            ),
            FadeOut(exp_rect),
        )
        self.wait()

        # Show mirror image
        self.play(
            TransformMatchingTex(
                equation5.copy(), equation2.copy(),
                key_map={
                    "s^2": "x''(t)",
                    R"\mu s": R"\mu x(t)",
                    R"k": R"k x(t)",
                },
                # match_animation=FadeTransform,
                # mismatch_animation=FadeTransform,
                remover=True,
                rate_func=there_and_back_with_pause,
                run_time=6
            ),
            equation4.animate.set_fill(opacity=0.25),
        )
        self.play(equation4.animate.set_fill(opacity=1))
        self.wait()

        # Cover up mu terms
        boxes = VGroup(
            SurroundingRectangle(mob)
            for mob in [
                equation2[R"+ \mu x'(t)"],
                equation4[R"+ \mu s"],
                equation5[R"+ \mu s"],
            ]
        )
        boxes.set_fill(BLACK, 0)
        boxes.set_stroke(colors[1], 2)

        self.add(Point())
        self.play(FadeIn(boxes, lag_ratio=0.1))
        self.play(boxes.animate.set_fill(BLACK, 0.8).set_stroke(width=1, opacity=0.5))
        self.wait()

        # Add simple answer
        simple_answer = Tex(R"s = \pm i \sqrt{k / m}", t2c=t2c)
        simple_answer.next_to(equation5, DOWN, LARGE_BUFF, aligned_edge=RIGHT)

        omega_brace = Brace(simple_answer[R"\sqrt{k / m}"], DOWN, SMALL_BUFF)
        omega = omega_brace.get_tex(R"\omega")
        omega.set_color(PINK)

        self.play(FadeIn(simple_answer))
        self.wait()
        self.play(GrowFromCenter(omega_brace), Write(omega))
        self.wait()

        simple_answer.add(omega_brace, omega)

        # Reminder of what s represents
        s_copy = simple_answer[0].copy()
        s_rect = SurroundingRectangle(s_copy)

        self.play(ShowCreation(s_rect))
        self.wait()
        self.play(
            s_rect.animate.surround(hyp_tex["e^{st}"]).set_anim_args(path_arc=-60 * DEG),
            FadeTransform(s_copy, hyp_tex["s"], path_arc=-60 * DEG),
            run_time=2
        )
        self.wait()
        self.play(FadeOut(s_rect))

        # Move hypothesis
        frame = self.frame
        self.play(
            frame.animate.scale(1.5, about_edge=LEFT),
            hypothesis.animate.next_to(equation2, UP, LARGE_BUFF, aligned_edge=LEFT),
            FadeOut(sub_hyp_word),
            run_time=1.5
        )
        self.wait()

        # Show quadratic formula
        qf_arrow = Arrow(
            equation5.get_right(),
            equation6.get_corner(UR) + 0.5 * LEFT,
            path_arc=-150 * DEG
        )
        qf_words = Text("Quadratic\nFormula", font_size=30, fill_color=GREY_B)
        qf_words.next_to(qf_arrow.get_center(), UR)

        naked_equation = equation6.copy()
        for sym in key_syms:
            naked_equation[sym].scale(0).set_fill(opacity=0).move_to(10 * LEFT)

        qf_rect = SurroundingRectangle(equation6[2:])
        qf_rect.set_stroke(YELLOW, 1.5)

        self.play(
            FadeOut(simple_answer, DOWN),
            boxes.animate.set_fill(opacity=0).set_stroke(width=2, opacity=1)
        )
        self.play(FadeOut(boxes))
        self.wait()
        self.play(
            TransformFromCopy(equation5["s"], equation6["s"]),
            Write(equation6["="]),
            GrowFromPoint(qf_arrow, qf_arrow.get_corner(UL)),
            FadeIn(qf_words, shift=0.5 * DOWN),
        )
        self.play(
            LaggedStart(*(
                TransformFromCopy(equation5[sym], equation6[sym], time_span=(0.5, 1.5))
                for sym in key_syms[1:]
            ), lag_ratio=0.1),
            Write(naked_equation),
        )
        self.wait()
        self.remove(naked_equation)
        self.add(equation6)
        self.play(ShowCreation(qf_rect))
        self.wait()

    def old_material(self):
        # Show implied exponentials
        final_equation = new_equations[-1]
        consolidated_lines = VGroup(
            hypothesis,
            equation2,
            equation4,
            final_equation,
        )
        consolidated_lines.target = consolidated_lines.generate_target()
        consolidated_lines.target.scale(0.7)
        consolidated_lines.target.arrange(DOWN, buff=MED_LARGE_BUFF)
        consolidated_lines.target.to_corner(UL)

        implies = Tex(R"\Longrightarrow", font_size=60)
        implies.next_to(consolidated_lines.target[0], RIGHT, buff=0.75)

        t2c = {"x(t)": TEAL, R"\omega": PINK}
        imag_exps = VGroup(
            Tex(R"x(t) = e^{+i \omega t}", t2c=t2c),
            Tex(R"x(t) = e^{-i \omega t}", t2c=t2c),
        )
        imag_exps.arrange(RIGHT, buff=2.0)
        imag_exps.next_to(implies, RIGHT, buff=0.75)

        self.remove(final_equation)
        self.play(LaggedStart(
            FadeOut(arrows),
            FadeOut(equation3, 0.5 * UP),
            FadeOut(sub_hyp_word),
            MoveToTarget(consolidated_lines),
            Write(implies),
        ))
        for imag_exp, sgn in zip(imag_exps, "+-"):
            self.play(
                TransformFromCopy(hyp_tex["x(t) ="][0], imag_exp["x(t) ="][0]),
                TransformFromCopy(hyp_tex["e"][0], imag_exp["e"][0]),
                TransformFromCopy(hyp_tex["t"][-1], imag_exp["t"][-1]),
                FadeTransform(final_equation[R"\pm i"][0].copy(), imag_exp[Rf"{sgn}i"][0]),
                FadeTransform(final_equation[R"\sqrt{k/m}"][0].copy(), imag_exp[R"\omega"][0]),
            )

        omega_brace = Brace(final_equation[R"\sqrt{k/m}"], DOWN, SMALL_BUFF)
        omega_label = omega_brace.get_tex(R"\omega").set_color(PINK)
        self.play(GrowFromCenter(omega_brace), Write(omega_label))
        self.wait()

        # Combine two solutions
        cos_equation = Tex(R"e^{+i \omega t} + e^{-i \omega t} = 2\cos(\omega t)", t2c={R"\omega": PINK})
        cos_equation.move_to(imag_exps)
        omega_brace2 = omega_brace.copy()
        omega_brace2.stretch(0.5, 0).match_width(cos_equation[R"\omega"][-1])
        omega_brace2.next_to(cos_equation[R"\omega"][-1], DOWN, SMALL_BUFF)
        omega_brace2_tex = omega_brace2.get_tex(R"\sqrt{k / m}", buff=SMALL_BUFF, font_size=24)

        self.remove(imag_exps)
        self.play(
            TransformFromCopy(imag_exps[0][R"e^{+i \omega t}"], cos_equation[R"e^{+i \omega t}"]),
            TransformFromCopy(imag_exps[1][R"e^{-i \omega t}"], cos_equation[R"e^{-i \omega t}"]),
            FadeOut(imag_exps[0][R"x(t) ="]),
            FadeOut(imag_exps[1][R"x(t) ="]),
            Write(cos_equation["+"][1]),
        )
        self.wait()
        self.play(Write(cos_equation[R"= 2\cos(\omega t)"]))
        self.wait()
        self.play(GrowFromCenter(omega_brace2), Write(omega_brace2_tex))

        # Clear the board
        self.play(LaggedStart(
            FadeOut(implies),
            FadeOut(cos_equation),
            FadeOut(omega_brace2),
            FadeOut(omega_brace2_tex),
            FadeOut(consolidated_lines[2:]),
            FadeOut(omega_brace),
            FadeOut(omega_label),
            lag_ratio=0.1
        ))

        # Add damping term
        t2c = {"x''(t)": colors[2], "x'(t)": colors[1], "x(t)": colors[0], "{s}": YELLOW}
        new_lines = VGroup(
            Tex(R"m x''(t) + \mu x'(t) + k x(t) = 0", t2c=t2c),
            Tex(R"m ({s}^2 e^{{s}t}) + \mu ({s} e^{{s}t}) + k (e^{{s}t}) = 0", t2c=t2c),
            Tex(R"e^{{s}t}\left(m {s}^2 + \mu {s} + k \right) = 0", t2c=t2c),
            Tex(R"m {s}^2 + \mu {s} + k = 0", t2c=t2c),
            Tex(R"{s} = {{-\mu \pm \sqrt{\mu^2 - 4mk}} \over 2m}", t2c=t2c),
        )
        new_lines.scale(0.7)
        new_lines.arrange(DOWN, aligned_edge=LEFT, buff=MED_LARGE_BUFF)
        new_lines.move_to(equation2, UL)

        self.play(
            TransformMatchingTex(
                equation2,
                new_lines[0],
                matched_keys=t2c.keys(),
                run_time=1
            )
        )
        self.wait()
        for line1, line2 in zip(new_lines, new_lines[1:]):
            if line1 is new_lines[0]:
                key_map = {
                    "x''(t)": R"({s}^2 e^{{s}t})",
                    "x'(t)": R"({s} e^{{s}t})",
                    "x(t)": R"(e^{{s}t})",
                }
            else:
                key_map = dict()
            self.play(TransformMatchingTex(line1.copy(), line2, key_map=key_map, run_time=1, lag_ratio=0.01))
            self.wait()


class DampedSpringSolutionsOnSPlane(InteractiveScene):
    def construct(self):
        # Add the plane
        plane = ComplexPlane((-3, 2), (-2, 2))
        plane.set_height(5)
        plane.background_lines.set_stroke(BLUE, 1)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)
        plane.add_coordinate_labels(font_size=24)
        plane.move_to(DOWN)
        plane.to_edge(RIGHT, buff=1.0)
        self.add(plane)

        # Add the sliders
        colors = [interpolate_color_by_hsl(RED, TEAL, a) for a in np.linspace(0, 1, 3)]
        chars = ["m", R"\mu", "k"]
        m_slider, mu_slider, k_slider = sliders = VGroup(
            self.get_slider(char, color)
            for char, color in zip(chars, colors)
        )
        m_tracker, mu_tracker, k_tracker = trackers = Group(
            slider.value_tracker for slider in sliders
        )
        sliders.arrange(RIGHT, buff=MED_LARGE_BUFF)
        sliders.next_to(plane, UP, aligned_edge=LEFT)

        for tracker, value in zip(trackers, [1, 0, 3]):
            tracker.set_value(value)

        self.add(trackers)
        self.add(sliders[0], sliders[2])

        # Add the dots
        def get_roots():
            a, b, c = [t.get_value() for t in trackers]
            m = -b / 2
            p = c / a
            disc = m**2 - p
            radical = math.sqrt(disc) if disc >= 0 else 1j * math.sqrt(-disc)
            return (m + radical, m - radical)

        def update_dots(dots):
            roots = get_roots()
            for dot, root in zip(dots, roots):
                dot.move_to(plane.n2p(root))

        root_dots = GlowDot().replicate(2)
        root_dots.add_updater(update_dots)

        s_rhs_point = Point((-4.09, -1.0, 0.0))
        rect_edge_point = (-3.33, -1.18, 0.0)

        def update_lines(lines):
            for line, dot in zip(lines, root_dots):
                line.put_start_and_end_on(
                    s_rhs_point.get_center(),
                    dot.get_center(),
                )

        lines = Line().replicate(2)
        lines.set_stroke(YELLOW, 2, 0.35)
        lines.add_updater(update_lines)

        self.add(root_dots)

        # Play with k
        self.play(ShowCreation(lines, lag_ratio=0, suspend_mobject_updating=True))
        self.play(k_tracker.animate.set_value(1), run_time=2)
        self.play(m_tracker.animate.set_value(4), run_time=2)
        self.wait()
        self.play(k_tracker.animate.set_value(3), run_time=2)
        self.play(m_tracker.animate.set_value(1), run_time=2)
        self.wait()

        # Play with mu
        self.play(
            s_rhs_point.animate.move_to(rect_edge_point),
            VFadeOut(lines),
            VFadeIn(sliders[1])
        )
        self.wait()
        self.play(mu_tracker.animate.set_value(3), run_time=5)
        self.wait()
        self.play(mu_tracker.animate.set_value(0.5), run_time=3)
        self.play(ShowCreation(lines, lag_ratio=0, suspend_mobject_updating=True))

        # Background
        self.add_background_image()

        # Zoom out and show graph
        frame = self.frame

        axes = Axes((0, 10, 1), (-1, 1, 1), width=10, height=3.5)
        axes.next_to(plane, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        def func(t):
            roots = get_roots()
            return 0.5 * (np.exp(roots[0] * t) + np.exp(roots[1] * t)).real

        graph = axes.get_graph(func)
        graph.set_stroke(TEAL, 3)
        axes.bind_graph_to_func(graph, func)

        graph_label = Tex(R"\text{Re}[e^{st}]", t2c={"s": YELLOW}, font_size=72)
        graph_label.next_to(axes.get_corner(UL), DL)

        self.play(
            frame.animate.set_height(12, about_point=4 * UP + 2 * LEFT),
            FadeIn(axes, time_span=(1.5, 3)),
            ShowCreation(graph, suspend_mobject_updating=True, time_span=(1.5, 3)),
            Write(graph_label),
            run_time=3
        )
        self.wait()

        # Show exponential decay
        exp_graph = axes.get_graph(lambda t: np.exp(get_roots()[0].real * t))
        exp_graph.set_stroke(WHITE, 1)

        self.play(ShowCreation(exp_graph))
        self.wait()

        # More play
        self.play(k_tracker.animate.set_value(1), run_time=2)
        self.play(k_tracker.animate.set_value(4), run_time=2)
        self.play(FadeOut(exp_graph))
        self.wait()
        self.play(mu_tracker.animate.set_value(2), run_time=3)
        self.play(k_tracker.animate.set_value(2), run_time=2)
        self.wait()
        self.play(mu_tracker.animate.set_value(3.5), run_time=3)
        self.play(k_tracker.animate.set_value(5), run_time=2)
        self.wait()
        self.play(
            mu_tracker.animate.set_value(0.5),
            m_tracker.animate.set_value(3),
            run_time=3
        )
        self.wait()

        # Smooth all the way to end
        self.play(mu_tracker.animate.set_value(4.2), run_time=12)

    def add_background_image(self):
        image = ImageMobject('/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2025/laplace/shm/images/LaplaceFormulaStill.png')
        image.replace(self.frame)
        image.set_z_index(-1)
        self.background_image = image
        self.add(image)

    def get_slider(self, char_name, color=WHITE, x_range=(0, 5), height=1.5, font_size=36):
        tracker = ValueTracker(0)
        number_line = NumberLine(x_range, width=height, tick_size=0.05)
        number_line.rotate(90 * DEG)

        indicator = ArrowTip(width=0.1, length=0.2)
        indicator.rotate(PI)
        indicator.add_updater(lambda m: m.move_to(number_line.n2p(tracker.get_value()), LEFT))
        indicator.set_color(color)

        label = Tex(Rf"{char_name} = 0.00", font_size=font_size)
        label[char_name].set_color(color)
        label.rhs = label.make_number_changeable("0.00")
        label.always.next_to(indicator, RIGHT, SMALL_BUFF)
        label.rhs.f_always.set_value(tracker.get_value)

        slider = VGroup(number_line, indicator, label)
        slider.value_tracker = tracker
        return slider

    def insertion(self):
        # Insertion after "play with mu" above
        self.wait()
        self.play(mu_tracker.animate.set_value(3), run_time=3)
        self.play(k_tracker.animate.set_value(1), run_time=2)
        self.play(k_tracker.animate.set_value(4), run_time=2)
        self.wait()
        self.play(mu_tracker.animate.set_value(4), run_time=2)
        self.play(k_tracker.animate.set_value(0.5), run_time=2)
        self.wait()
        self.play(mu_tracker.animate.set_value(0.5), run_time=4)
        self.play(k_tracker.animate.set_value(2), run_time=4)


class RotatingExponentials(InteractiveScene):
    def construct(self):
        # Create time tracker
        t_tracker = ValueTracker(0)
        t_tracker.add_updater(lambda m, dt: m.increment_value(dt))
        get_t = t_tracker.get_value
        omega = PI / 2

        def get_x():
            return math.cos(omega * get_t())

        self.add(t_tracker)

        # Create two complex planes side by side
        left_plane, right_plane = planes = VGroup(
            ComplexPlane(
                (-2, 2), (-2, 2),
                background_line_style=dict(stroke_color=BLUE, stroke_width=1),
            )
            for _ in range(2)
        )
        for plane in planes:
            plane.axes.set_stroke(width=1)
            plane.set_height(3.5)
            plane.add_coordinate_labels(font_size=16)
        planes.arrange(RIGHT, buff=1.0)
        planes.to_edge(RIGHT)
        planes.to_edge(UP, buff=1.5)

        self.add(planes)

        # Add titles
        t2c = {R"\omega": PINK}
        left_title, right_title = titles = VGroup(
            Tex(tex, t2c=t2c, font_size=48)
            for tex in [
                R"e^{+i \omega t}",
                R"e^{-i \omega t}",
            ]
        )
        for title, plane in zip(titles, planes):
            title.next_to(plane, UP)

        self.add(titles)

        # Create rotating vectors
        left_vector = self.get_rotating_vector(left_plane, 1j * omega, t_tracker, color=TEAL)
        right_vector = self.get_rotating_vector(right_plane, -1j * omega, t_tracker, color=RED)
        vectors = VGroup(left_vector, right_vector)

        left_tail, right_tail = tails = VGroup(
            TracingTail(vect.get_end, stroke_color=vect.get_color(), time_traced=2)
            for vect in vectors
        )

        self.add(Point())
        self.add(vectors, tails)

        # Add time display
        time_display = Tex("t = 0.00", font_size=36).to_corner(UR)
        time_label = time_display.make_number_changeable("0.00")
        time_label.add_updater(lambda m: m.set_value(t_tracker.get_value()))

        # Animate rotation
        self.wait(12)

        # Add spring
        spring = SrpingMassSystem(
            equilibrium_position=planes[0].get_bottom() + DOWN,
            equilibrium_length=3,
            n_spring_curls=8,
            mass_width=0.5,
            spring_radius=0.2,
        )
        spring.pause()
        unit_size = planes[0].x_axis.get_unit_size()
        spring.add_updater(lambda m: m.set_x(unit_size * get_x()))

        v_line = Line()
        v_line.set_stroke(BLUE_A, 2)
        v_line.f_always.put_start_and_end_on(spring.mass.get_top, left_vector.get_end)

        self.play(VFadeIn(spring), VFadeIn(v_line))
        self.wait(20)
        self.play(
            VFadeOut(spring),
            VFadeOut(v_line),
            VFadeOut(tails),
        )

        # Add them up
        new_plane_center = planes.get_center()
        shift_factor = ValueTracker(0)
        right_vector.add_updater(lambda m: m.shift(shift_factor.get_value() * left_vector.get_vector()))

        sum_expr = VGroup(titles[0], Tex(R"+"), titles[1])
        sum_expr.target = sum_expr.generate_target()
        sum_expr.target.arrange(RIGHT, buff=MED_SMALL_BUFF, aligned_edge=DOWN)
        sum_expr.target.next_to(planes, UP, MED_SMALL_BUFF)
        sum_expr[1].set_opacity(0).next_to(planes, UP)

        result_dot = GlowDot()
        result_dot.f_always.move_to(right_vector.get_end)

        self.play(
            planes[0].animate.move_to(new_plane_center),
            planes[1].animate.move_to(new_plane_center).set_opacity(0),
            MoveToTarget(sum_expr),
            run_time=2,
        )
        self.play(shift_factor.animate.set_value(1))
        self.play(FadeIn(result_dot))
        self.wait(4)

        # Add another spring
        spring = SrpingMassSystem(
            equilibrium_position=planes[0].get_bottom() + DOWN,
            equilibrium_length=5,
            n_spring_curls=8,
            mass_width=0.5,
            spring_radius=0.2,
        )
        spring.pause()
        unit_size = planes[0].x_axis.get_unit_size()
        spring.add_updater(lambda m: m.set_x(2 * unit_size * get_x()))

        v_line = Line()
        v_line.set_stroke(BLUE_A, 2)
        v_line.f_always.put_start_and_end_on(spring.mass.get_top, result_dot.get_center)

        self.play(VFadeIn(spring), VFadeIn(v_line))
        self.wait(2)

        # Right hand side
        rhs = Tex(R"= 2 \cos(\omega t)", t2c={R"\omega": PINK})
        rhs.next_to(sum_expr, RIGHT, buff=MED_SMALL_BUFF).shift(SMALL_BUFF * DOWN)

        self.play(Write(rhs))
        self.wait(20)

    def get_rotating_vector(self, plane, s, t_tracker, color=TEAL, thickness=3):
        """Create a rotating vector for e^(st) on the given plane"""
        def update_vector(vector):
            t = t_tracker.get_value()
            c = vector.coef_tracker.get_value()
            z = c * np.exp(s * t)
            vector.put_start_and_end_on(plane.n2p(0), plane.n2p(z))

        vector = Arrow(LEFT, RIGHT, fill_color=color, thickness=thickness)
        vector.coef_tracker = ComplexValueTracker(1)
        vector.add_updater(update_vector)

        return vector


class SimpleSolutionSummary(InteractiveScene):
    def construct(self):
        # Summary of the "strategy" up top
        t2c = {"m": RED, "k": TEAL, "{s}": YELLOW, R"\omega": PINK}
        kw = dict(t2c=t2c, font_size=36)
        arrow = Vector(1.5 * RIGHT)
        top_eq = VGroup(
            Tex(R"m x''(t) + k x(t) = 0", **kw),
            arrow,
            Tex(R"e^{{s}t}\left(m{s}^2 + k\right) = 0", **kw),
            Tex(R"\Longrightarrow", **kw),
            Tex(R"{s} = \pm i \underbrace{\sqrt{k / m}}_{\omega}", **kw),
        )
        top_eq.arrange(RIGHT)
        top_eq[-1].align_to(top_eq[2], UP)
        guess = Tex(R"\text{Guess } e^{{s}t}", **kw)
        guess.scale(0.75)
        guess.next_to(arrow, UP, buff=0)
        arrow.add(guess)

        top_eq.set_width(FRAME_WIDTH - 1)

        top_eq.center().to_edge(UP)
        self.add(top_eq)


class ShowFamilyOfComplexSolutions(RotatingExponentials):
    tex_to_color_map = {R"\omega": PINK}
    plane_config = dict(
        background_line_style=dict(stroke_color=BLUE, stroke_width=1),
        faded_line_style=dict(stroke_color=BLUE, stroke_width=0.5, stroke_opacity=0.25),
    )
    vect_colors = [TEAL, RED]
    rotation_frequency = TAU / 4

    def construct(self):
        # Show the equation
        frame = self.frame
        frame.set_x(-10)

        colors = get_coef_colors()
        t2c = {"x''(t)": colors[2], "x'(t)": colors[1], "x(t)": colors[0]}
        equation = Tex(R"m x''(t) + k x(t) = 0", t2c=t2c, font_size=42)
        equation.next_to(frame.get_left(), RIGHT, buff=1.0)

        arrow = Vector(3.0 * RIGHT, thickness=6, fill_color=GREY_B)
        arrow.next_to(equation, RIGHT, MED_LARGE_BUFF)

        strategy_words = VGroup(
            Text("“Strategy”"),
            TexText(R"Guess $e^{{s}t}$", t2c={R"{s}": YELLOW}, font_size=36, fill_color=GREY_A)
        )
        strategy_words.arrange(DOWN)
        strategy_words.next_to(arrow, UP, MED_SMALL_BUFF)

        self.add(equation)
        self.play(
            GrowArrow(arrow),
            FadeIn(strategy_words, lag_ratio=0.1)
        )
        self.wait()

        # Show two basis solutions on the left
        t2c = self.tex_to_color_map
        left_planes, left_plane_labels = self.get_left_planes(label_texs=[R"e^{+i\omega t}", R"e^{-i\omega t}"])
        rot_vects, tails, t_tracker = self.get_rot_vects(left_planes)
        left_planes_brace = Brace(left_planes, LEFT, MED_SMALL_BUFF)

        self.add(rot_vects, tails)
        self.add(t_tracker)
        self.play(
            GrowFromCenter(left_planes_brace),
            FadeIn(left_planes),
            FadeTransform(strategy_words[1][R"e^{{s}t}"].copy(), left_plane_labels[0]),
            FadeTransform(strategy_words[1][R"e^{{s}t}"].copy(), left_plane_labels[1]),
            VFadeIn(rot_vects),
        )
        self.wait(3)

        self.wait(8)

        # Show combination with tunable parameters
        right_plane = self.get_right_plane()
        right_plane.next_to(left_planes, RIGHT, buff=1.5)

        scaled_solution = Tex(
            R"c_1 e^{+i\omega t} + c_2 e^{-i\omega t}",
            t2c={R"\omega": PINK, "c_1": BLUE, "c_2": BLUE}
        )
        scaled_solution.next_to(right_plane, UP)

        vect1, vect2 = right_rot_vects = self.get_rot_vect_sum(right_plane, t_tracker)
        c1_eq, c2_eq = coef_eqs = VGroup(
            VGroup(Tex(fR"c_{n} = "), DecimalNumber(1))
            for n in [1, 2]
        )
        coef_eqs.scale(0.85)
        for coef_eq in coef_eqs:
            coef_eq.arrange(RIGHT, buff=SMALL_BUFF)
            coef_eq[1].align_to(coef_eq[0][0], DOWN)
            coef_eq[0][:2].set_fill(BLUE)
        coef_eqs.arrange(DOWN, MED_LARGE_BUFF)
        coef_eqs.to_corner(UR)
        coef_eqs.shift(LEFT)

        c1_eq[1].add_updater(lambda m: m.set_value(vect1.coef_tracker.get_value()))
        c2_eq[1].add_updater(lambda m: m.set_value(vect2.coef_tracker.get_value()))

        self.play(
            FadeIn(right_plane),
            FadeOut(left_planes_brace),
            frame.animate.center(),
            run_time=2
        )
        self.play(LaggedStart(
            FadeTransform(left_plane_labels[0].copy(), scaled_solution[R"e^{+i\omega t}"]),
            FadeIn(scaled_solution[R"c_1"]),
            TransformFromCopy(rot_vects[0], right_rot_vects[0], suspend_mobject_updating=True),
            FadeTransform(left_plane_labels[1].copy(), scaled_solution[R"e^{-i\omega t}"]),
            FadeIn(scaled_solution[R"+"][1]),
            FadeIn(scaled_solution[R"c_2"]),
            TransformFromCopy(rot_vects[1], right_rot_vects[1], suspend_mobject_updating=True)
        ))
        self.play(LaggedStart(
            FadeTransformPieces(scaled_solution[R"c_1"].copy(), c1_eq),
            FadeTransformPieces(scaled_solution[R"c_2"].copy(), c2_eq)
        ))
        self.play(LaggedStart(
            vect1.coef_tracker.animate.set_value(2),
            vect2.coef_tracker.animate.set_value(0.5),
            lag_ratio=0.5
        ))

        comb_tail = TracingTail(vect2.get_end, stroke_color=YELLOW, time_traced=2)
        glow_dot = GlowDot()
        glow_dot.f_always.move_to(vect2.get_end)
        self.add(comb_tail)
        self.play(FadeIn(glow_dot))

        self.wait(6)
        self.play(LaggedStart(
            vect1.coef_tracker.animate.set_value(complex(1.5, 1)),
            vect2.coef_tracker.animate.set_value(complex(0.5, -1.25)),
        ))
        self.wait(7)

        # Change the coefficients
        t_tracker.suspend_updating()
        self.play(
            FadeOut(comb_tail, suspend_mobject_updating=True),
            LaggedStart(
                vect1.coef_tracker.animate.set_value(complex(0.31, -0.41)),
                vect2.coef_tracker.animate.set_value(complex(2.71, -0.82)),
            ),
        )
        self.wait()
        self.play(
            LaggedStart(
                vect1.coef_tracker.animate.set_value(complex(-1.03, 0.5)),
                vect2.coef_tracker.animate.set_value(complex(1.5, 0.35)),
            ),
        )
        self.add(comb_tail)
        self.wait(2)
        t_tracker.resume_updating()

        # Zoom out
        self.play(frame.animate.set_height(13.75, about_edge=RIGHT), run_time=2)
        self.wait(4)
        self.play(frame.animate.to_default_state(), run_time=2)

        # Go to real valued
        self.play(
            LaggedStart(
                vect1.coef_tracker.animate.set_value(1),
                vect2.coef_tracker.animate.set_value(1),
            ),
        )
        self.wait(6)

        # Show initial conditions
        initial_conditions = VGroup(
            Tex(R"x_0 = 0.00"),
            Tex(R"v_0 = 0.00"),
        )
        x0_value = initial_conditions[0].make_number_changeable("0.00")
        v0_value = initial_conditions[1].make_number_changeable("0.00")
        x0_value.set_value(2)
        initial_conditions.scale(0.85)
        initial_conditions.arrange(DOWN)
        initial_conditions.move_to(coef_eqs, LEFT)
        initial_conditions.to_edge(UP)
        implies = Tex(R"\Downarrow", font_size=72)
        implies.next_to(initial_conditions, DOWN)

        t_tracker.suspend_updating()
        t_tracker.set_value((t_tracker.get_value() + 2) % 4 - 2)
        self.play(
            FadeIn(initial_conditions),
            Write(implies),
            coef_eqs.animate.next_to(implies, DOWN).align_to(initial_conditions, LEFT),
        )
        self.remove(comb_tail)
        self.play(
            vect1.coef_tracker.animate.set_value(1),
            vect2.coef_tracker.animate.set_value(1),
            t_tracker.animate.set_value(0),
        )
        self.wait()
        self.remove(comb_tail)

        # Highlight values, rise
        t_tracker.resume_updating()

        highlight_rect = SurroundingRectangle(initial_conditions[0])
        highlight_rect.set_stroke(YELLOW, 2)

        self.play(ShowCreation(highlight_rect))
        self.wait()
        self.play(highlight_rect.animate.surround(initial_conditions[1]))
        self.wait(2)
        self.play(highlight_rect.animate.surround(coef_eqs))
        self.wait(4)

        self.play(
            vect1.coef_tracker.animate.set_value(1.5),
            vect2.coef_tracker.animate.set_value(1.5),
            ChangeDecimalToValue(x0_value, 3)
        )
        self.wait(12)

    def get_left_planes(self, label_texs: list[str]):
        planes = VGroup(
            ComplexPlane((-1, 1), (-1, 1), **self.plane_config)
            for _ in range(2)
        )
        planes.arrange(DOWN, buff=1.0)
        planes.set_height(6.5)
        planes.to_corner(DL)
        planes.set_z_index(-1)

        labels = VGroup(Tex(tex, t2c=self.tex_to_color_map) for tex in label_texs)
        for label, plane in zip(labels, planes):
            label.next_to(plane, UP, SMALL_BUFF)

        return planes, labels

    def get_rot_vects(self, planes):
        t_tracker = ValueTracker(0)
        t_tracker.add_updater(lambda m, dt: m.increment_value(dt))

        rot_vects = VGroup(
            self.get_rotating_vector(plane, u * 1j * self.rotation_frequency, t_tracker, color)
            for plane, u, color in zip(planes, [+1, -1], self.vect_colors)
        )
        tails = VGroup(
            TracingTail(vect.get_end, stroke_color=vect.get_color(), time_traced=2)
            for vect in rot_vects
        )

        return Group(rot_vects, tails, t_tracker)

    def get_rot_vect_sum(self, plane, t_tracker):
        vect1, vect2 = vect_sum = VGroup(
            self.get_rotating_vector(
                plane,
                u * 1j * self.rotation_frequency,
                t_tracker,
                color,
            )
            for u, color in zip([+1, -1], self.vect_colors)
        )
        vect2.add_updater(lambda m: m.put_start_on(vect1.get_end()))
        return vect_sum

    def get_right_plane(self, x_range=(-3, 3), height=5.5):
        right_plane = ComplexPlane(x_range, x_range, **self.plane_config)
        right_plane.set_height(height)
        return right_plane

    def add_scale_tracker(vector, initial_value=1):
        """
        Assumes the vector has another updater constantly setting a location in the plane
        """
        vector.c_tracker = ComplexValueTracker(initial_value)

        def update_vector(vect):
            c = vect.c_tracker.get_value()
            vect.scale()
            pass


class GuessSine(InteractiveScene):
    func_name = R"\sin"

    def construct(self):
        # Set up
        self.frame.set_height(9, about_edge=LEFT)
        func_name = self.func_name
        func_tex = Rf"{func_name}(\omega t)"

        t2c = {R"\omega": PINK, "x(t)": TEAL, "x''(t)": RED}
        equation = Tex(R"m x''(t) + k x(t) = 0", t2c=t2c)
        equation.to_edge(LEFT)
        guess_words = TexText(Rf"Try $x(t) = {func_tex}$", t2c=t2c, font_size=36)

        arrow = Arrow(guess_words.get_left(), guess_words.get_right(), buff=-0.1, thickness=6)
        arrow.next_to(equation, RIGHT)
        guess_words.next_to(arrow, UP, SMALL_BUFF)

        self.add(equation)
        self.add(arrow)
        self.add(guess_words)

        # Sub
        sub = Tex(fR"-m \omega^2 {func_tex}  + k {func_tex} = 0", t2c=t2c)
        sub.next_to(arrow, RIGHT)
        simple_sub = Tex(Rf"\left(-m \omega^2 + k\right) {func_tex} = 0", t2c=t2c)
        simple_sub.next_to(arrow, RIGHT)
        implies = Tex(R"\Rightarrow", font_size=72)
        implies.next_to(simple_sub, RIGHT)

        t2c[R"\ding{51}"] = GREEN
        t2c["Valid"] = GREEN
        result = TexText(R"\ding{51} Valid if $\omega = \sqrt{k / m}$", t2c=t2c, font_size=36)
        result.next_to(simple_sub, DOWN)

        simple_sub.shift(0.05 * UP)
        blank_sub = sub.copy()
        blank_sub[func_tex].set_opacity(0)
        func_parts = sub[func_tex]

        self.play(LaggedStart(
            TransformMatchingTex(equation.copy(), blank_sub, path_arc=30 * DEG, run_time=2),
            FadeTransform(guess_words[func_tex][0].copy(), func_parts[0]),
            FadeTransform(guess_words[func_tex][0].copy(), func_parts[1]),
            lag_ratio=0.25
        ))
        self.wait()
        self.play(
            TransformMatchingTex(blank_sub, simple_sub),
            Transform(func_parts[0], simple_sub[func_tex][0], path_arc=-30 * DEG),
            Transform(func_parts[1], simple_sub[func_tex][0], path_arc=-30 * DEG),
            run_time=1
        )
        self.wait()
        self.play(FadeIn(result, 0.5 * UP))
        self.wait()


class GuessCosine(GuessSine):
    func_name = R"\cos"


class ShowFamilyOfRealSolutions(InteractiveScene):
    t2c = {R"\omega": PINK}
    omega = PI
    x_max = 10

    def construct(self):
        # Add cosine and sine graphs up top
        cos_graph, sin_graph = small_graphs = VGroup(
            self.get_small_graph(math.cos, R"\cos(\omega t)", BLUE),
            self.get_small_graph(math.sin, R"\sin(\omega t)", RED),
        )
        small_graphs.arrange(RIGHT, buff=LARGE_BUFF)
        small_graphs.to_edge(UP, buff=MED_SMALL_BUFF)

        self.add(small_graphs)

        # Add master graph
        coef_trackers = ValueTracker(1).replicate(2)

        def func(t):
            c1 = coef_trackers[0].get_value()
            c2 = coef_trackers[1].get_value()
            return c1 * np.cos(self.omega * t) + c2 * np.sin(self.omega * t)

        axes = Axes((0, self.x_max), (-3, 3), width=self.x_max, height=4)
        axes.to_edge(DOWN)
        graph_label = Tex(R"+1.00 \cos(\omega t) +1.00 \sin(\omega t)", t2c=self.t2c)
        graph_label.next_to(axes.y_axis.get_top(), UR)
        coef_labels = graph_label.make_number_changeable("+1.00", replace_all=True, edge_to_fix=RIGHT, include_sign=True)
        coef_labels.set_color(YELLOW)
        coef_labels[0].add_updater(lambda m: m.set_value(coef_trackers[0].get_value()))
        coef_labels[1].add_updater(lambda m: m.set_value(coef_trackers[1].get_value()))

        graph = axes.get_graph(func)
        graph.set_stroke(TEAL, 5)
        axes.bind_graph_to_func(graph, func)

        self.add(axes)
        self.add(graph_label)
        self.add(graph)

        # Tweak the parameters
        for c1, c2 in [(-0.5, 2), (1.5, 0), (0.25, -2)]:
            self.play(LaggedStart(
                coef_trackers[0].animate.set_value(c1),
                coef_trackers[1].animate.set_value(c2),
                lag_ratio=0.5
            ))
            self.wait()

        # Tweak with spring
        for c1, c2 in [(1.6, 1.8), (-2.7, 0.18), (0.5, -2)]:
            self.play(LaggedStart(
                coef_trackers[0].animate.set_value(c1),
                coef_trackers[1].animate.set_value(c2),
                lag_ratio=0.5
            ))
            self.show_spring(axes, graph, func)

    def get_small_graph(self, func, func_name, color):
        axes = Axes((0, 6), (-2, 2), height=2, width=6)
        graph = axes.get_graph(lambda t: func(self.omega * t))
        graph.set_stroke(color, 3)
        label = Tex(func_name, t2c=self.t2c, font_size=36)
        label.move_to(axes, UP)

        return VGroup(axes, graph, label)

    def show_spring(self, axes, graph, func):
        graph_copy = graph.copy()
        graph_copy.clear_updaters()

        spring = SrpingMassSystem(
            equilibrium_length=4,
            equilibrium_position=axes.get_right() + RIGHT,
            direction=UP,
            mass_width=0.75,
            spring_radius=0.2,
        )
        spring.add_updater(
            lambda m: m.set_x(axes.y_axis.get_unit_size() * axes.y_axis.p2n(graph_copy.get_end()))
        )

        h_line = Line()
        h_line.set_stroke(WHITE, 1)
        h_line.f_always.put_start_and_end_on(
            spring.mass.get_left,
            graph_copy.get_end,
        )

        self.play(
            graph.animate.set_stroke(opacity=0.2),
            ShowCreation(graph_copy, rate_func=linear, run_time=self.x_max),
            VFadeIn(spring, run_time=1),
            VFadeIn(h_line, run_time=1),
        )
        self.play(FadeOut(spring), FadeOut(h_line))
        self.wait()

        self.remove(graph_copy, spring, h_line)
        graph.set_stroke(opacity=1)


class SetOfInitialConditions(InteractiveScene):
    graph_time = 8

    def construct(self):
        # Set up all boxes
        frame = self.frame
        box = Rectangle(width=3.5, height=2.0)
        box.set_stroke(WHITE, 1)
        v_line = DashedLine(box.get_top(), box.get_bottom())
        v_line.set_stroke(GREY_C, 1, 0.5)
        v_line.scale(0.9)
        box.add(v_line)

        n_rows = 5
        n_cols = 5
        box_row = box.get_grid(1, n_cols, buff=0)
        box_grid = box_row.get_grid(n_rows, 1, buff=0)
        for row, v0 in zip(box_grid, np.linspace(1, -1, n_rows)):
            for box, x0 in zip(row, np.linspace(-1, 1, n_cols)):
                box.spring = self.get_spring_in_a_box(box, x0=x0, v0=v0)

        # Show the first example
        mid_row = box_grid[n_rows // 2]
        x0_labels = VGroup(
            Tex(Rf"x_0 = {x0}", font_size=48).next_to(box, UP, SMALL_BUFF)
            for x0, box in zip(range(-2, 3), mid_row)
        )
        mid_row_springs = VGroup(box.spring for box in mid_row)

        mid_row_solutions = VGroup(
            self.get_solution_graph(box, x0=x0)
            for box, x0 in zip(mid_row, range(-2, 3))
        )
        last_solution = mid_row_solutions[-1]

        last_box = mid_row[-1]
        last_spring = last_box.spring
        top_line = Line(last_box.get_center(), last_spring.mass.get_center())
        top_line.set_y(last_spring.mass.get_y(UP))
        brace = LineBrace(Line(ORIGIN, 1.5 * RIGHT))
        brace.match_width(top_line)
        brace.next_to(top_line, UP, SMALL_BUFF)
        brace.stretch(0.5, 1, about_edge=DOWN)
        last_x = last_spring.get_x()
        last_spring.set_x(0)

        self.add(last_box, last_spring)
        frame.move_to(mid_row[-1]).set_height(5)

        self.play(
            GrowFromPoint(brace, brace.get_left()),
            last_spring.animate.set_x(last_x),
            Write(x0_labels[-1]),
        )
        self.wait()
        last_spring.unpause()
        last_spring.v_vect.set_opacity(0)
        self.play(
            frame.animate.reorient(0, 0, 0, (6.7, -0.61, 0.0), 6.16).set_anim_args(run_time=2),
            FadeIn(last_solution[0]),
            ShowCreation(last_solution[1], rate_func=linear, run_time=self.graph_time)
        )
        last_spring.pause()

        # Show the full middle row
        self.play(
            frame.animate.center().set_width(box_grid.get_width() + 2).set_anim_args(time_span=(0, 1.5)),
            last_spring.animate.set_x(1).set_anim_args(time_span=(0, 1)),
            LaggedStartMap(FadeIn, VGroup(*reversed(mid_row[:-1])), lag_ratio=0.75),
            LaggedStartMap(FadeIn, VGroup(*reversed(mid_row_springs[:-1])), lag_ratio=0.75),
            LaggedStartMap(FadeIn, VGroup(*reversed(x0_labels[:-1])), lag_ratio=0.75),
            LaggedStartMap(FadeIn, VGroup(*reversed(mid_row_solutions[:-1])), lag_ratio=0.75),
            run_time=3
        )
        self.wait()

        graphs = VGroup(solution[1] for solution in mid_row_solutions)
        faint_graphs = graphs.copy().set_stroke(width=1, opacity=0.25)
        for spring in mid_row_springs:
            spring.unpause()
            spring.v_vect.set_opacity(0)
            spring.x0 = spring.get_x()
        self.add(faint_graphs)
        self.play(
            ShowCreation(graphs, lag_ratio=0, run_time=self.graph_time, rate_func=linear),
        )
        self.wait()
        self.remove(faint_graphs)
        for spring in mid_row_springs:
            spring.pause()
            spring.v_vect.set_opacity(0)
            spring.set_velocity(0)
        self.play(
            FadeOut(mid_row_solutions),
            *(spring.animate.set_x(spring.x0) for spring in mid_row_springs)
        )

        # Show initial velocities
        v0_labels = VGroup(
            Tex(Rf"v_0 = {v0}", font_size=48).next_to(row, LEFT).set_fill(TEAL)
            for v0, row in zip(range(2, -3, -1), box_grid)
        )
        other_indices = [0, 1, 3, 4]
        row_springs = VGroup(VGroup(box.spring for box in row) for row in box_grid)

        self.play(
            frame.animate.set_width(box_grid.get_width() + 3, about_edge=DR),
            Write(v0_labels[2])
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeIn, VGroup(box_grid[i] for i in other_indices), lag_ratio=0.5, run_time=3),
            LaggedStartMap(FadeIn, VGroup(v0_labels[i] for i in other_indices), lag_ratio=0.5, run_time=3),
            LaggedStartMap(FadeIn, VGroup(row_springs[i] for i in other_indices), lag_ratio=0.5, run_time=3),
            x0_labels.animate.next_to(box_grid, UP, SMALL_BUFF).set_anim_args(run_time=1),
            FadeOut(brace),
        )
        self.wait()

        # Let it play
        for row in box_grid:
            for box in row:
                box.spring.unpause()

        self.wait(20)

        # Add graphs
        all_solutions = VGroup(
            self.get_solution_graph(box, x0=x0, v0=v0, graph_color=YELLOW).move_to(box).scale(0.8)
            for row, v0 in zip(box_grid, range(2, -3, -1))
            for box, x0 in zip(row, range(-2, 3))
        )
        all_axes = VGroup(s[0] for s in all_solutions)
        all_graphs = VGroup(s[1] for s in all_solutions)

        self.remove(*(box.spring for row in box_grid for box in row))
        self.add(all_axes)
        self.play(ShowCreation(all_graphs, lag_ratio=1e-1, run_time=3))
        self.wait()

        # Highlight one
        highlight = all_solutions[14].copy()
        self.add(highlight)
        self.play(all_solutions.animate.fade(0.75))
        self.wait()

    def get_spring_in_a_box(self, box, x0=0, v0=0, k=9, mu=0.5):
        box_width = box.get_width()
        spring = SrpingMassSystem(
            x0=x0,
            v0=v0,
            k=k,
            mu=mu,
            mass_width=0.1 * box_width,
            equilibrium_length=0.5 * box_width,
            equilibrium_position=box.get_center(),
            spring_radius=0.035 * box_width,
        )
        v_vect = spring.get_velocity_vector(v_offset=-box_width * 0.1, scale_factor=0.5)
        spring.add(v_vect)
        spring.v_vect = v_vect
        spring.pause()
        return spring

    def get_solution_graph(self, box, x0=2, v0=0, k=9, mu=0.5, width_factor=0.8, graph_color=TEAL):
        axes = Axes(
            x_range=(0, self.graph_time, 1),
            y_range=(-2, 2),
            width=width_factor * box.get_width(),
            height=box.get_height(),
        )
        axes.set_stroke(GREY, 1)
        axes.next_to(box, DOWN)

        s = 0.5 * (-mu + 1j * math.sqrt(4 * k - mu**2))
        z0 = complex(
            x0,
            (s.real * x0 - v0) / s.imag
        )

        graph = axes.get_graph(
            lambda t: (z0 * np.exp(s * t)).real,
        )
        graph.set_stroke(graph_color, 2)
        return VGroup(axes, graph)


# For ODE video


class SpringInTheWind(InteractiveScene):
    F_0 = 1.0
    omega = 2
    k = 3
    mu = 0.1

    def setup(self):
        super().setup()
        # Set up wind

        plane = NumberPlane()
        wind = TimeVaryingVectorField(
            lambda p, t: np.tile(self.external_force(t) * RIGHT, (len(p), 1)),
            plane,
            density=1,
            color=WHITE,
            stroke_width=6,
            stroke_opacity=0.5,
        )

        # Set up spring and ODE
        spring = SrpingMassSystem(
            k=self.k,
            mu=self.mu,
            external_force=lambda: self.external_force(wind.time)
        )
        spring.add_external_force(lambda: self.F_0 * math.cos(self.omega * wind.time))

        self.add(wind)
        self.add(spring)

        self.spring = spring
        self.wind = wind

    def external_force(self, time):
        return self.F_0 * math.cos(self.omega * time)

    def construct(self):
        spring = self.spring
        wind = self.wind
        self.play(VFadeIn(wind))
        self.wait(60)


class ShowSpringInWindGraph(SpringInTheWind):
    mu = 0.25
    k = 3
    omega = 2.5

    def construct(self):
        spring = self.spring
        wind = self.wind

        # Add graph
        t_max = 40
        frame = self.frame
        frame.set_y(1)
        graph_block = Rectangle(width=FRAME_WIDTH, height=2.5)
        graph_block.move_to(frame, UP)
        graph_block.set_stroke(width=0)
        graph_block.set_fill(BLACK, 1)

        axes = Axes((0, t_max), (-0.5, 0.5, 0.25), width=FRAME_WIDTH - 1, height=2.0)
        axes.x_axis.ticks.stretch(0.5, 1)
        axes.move_to(graph_block)
        axis_label = Text("Time", font_size=24)
        axis_label.next_to(axes.x_axis.get_right(), DOWN)
        axes.add(axis_label)

        graph = TracedPath(
            lambda: axes.c2p(self.time, spring.get_x()),
            stroke_color=BLUE,
            stroke_width=3,
        )

        self.add(graph_block)
        self.add(axes)
        self.add(graph)

        # Play it out
        self.wait(t_max)
        graph.clear_updaters()
        self.play(
            VFadeOut(spring),
            VFadeOut(wind),
        )

        # Comment on the graph
        left_highlight = graph_block.copy()
        left_highlight.set_width(6.5, stretch=True, about_edge=LEFT)
        left_highlight.set_fill(YELLOW, 0.2)
        left_highlight.set_height(2, stretch=True)
        right_highlight = left_highlight.copy()
        right_highlight.set_width(10, stretch=True, about_edge=LEFT)
        right_highlight.next_to(left_highlight, RIGHT, buff=0)
        right_highlight.set_fill(GREEN, 0.2)

        self.play(FadeIn(left_highlight))
        self.wait()
        self.play(FadeIn(right_highlight))
        self.wait()

        # Show solution components
        axes1, axes2 = axes_copies = VGroup(axes.deepcopy() for _ in range(2))
        buff = 0.75
        axes_copies.arrange(DOWN, buff=buff)
        axes_copies.next_to(axes, DOWN, buff=buff)

        s_root = (-self.mu + 1j * math.sqrt(-(self.mu**2 - 4 * self.k))) / 2.0
        amp = 0.35
        shm_graph = axes1.get_graph(lambda t: amp * np.exp(s_root * t).real)
        shm_graph.set_stroke(GREEN, 4)
        cos_graph = axes2.get_graph(lambda t: -amp * math.cos(self.omega * t))
        cos_graph.set_stroke(YELLOW, 4)

        equals = Tex(R"=", font_size=72).rotate(90 * DEG)
        equals.move_to(VGroup(axes, axes1)).set_x(-5)
        equals.shift(0.25 * DOWN)
        plus = Tex(R"+", font_size=72)
        plus.move_to(VGroup(axes_copies)).match_x(equals)

        self.play(LaggedStart(
            Write(equals),
            FadeIn(axes1),
            FadeIn(shm_graph),
            Write(plus),
            FadeIn(axes2),
            FadeIn(cos_graph),
            lag_ratio=0.15
        ))
        self.wait()

        # First graph label
        colors = color_gradient([TEAL, RED], 3, interp_by_hsl=True)
        shm_eq = VGroup(
            Text("Solution to", font_size=36),
            Tex(
                R"mx''(t) + \mu x'(t) + k x(t) = 0",
                t2c={
                    "x(t)": colors[0],
                    "x'(t)": colors[1],
                    "x''(t)": colors[2],
                },
                font_size=36,
                alignment="",
            ),
        )
        shm_eq.arrange(DOWN)
        shm_eq.next_to(axes1.x_axis, UP, buff=0.35)
        shm_eq.set_x(-2.5)

        self.add(shm_graph.copy().set_stroke(opacity=0.25))
        self.play(
            FadeIn(shm_eq, lag_ratio=0.1),
            ShowCreation(shm_graph, run_time=3, rate_func=linear),
        )
        self.wait()

        # Grow left highlight
        self.play(
            left_highlight.animate.set_height(8, stretch=True, about_edge=UP).shift(0.15 * UP),
            shm_eq.animate.shift(5 * RIGHT),
            run_time=2,
        )
        self.wait()

        # Draw all graphs
        self.add(graph.copy().set_stroke(opacity=0.25))
        self.add(cos_graph.copy().set_stroke(opacity=0.25))
        self.play(
            left_highlight.animate.set_fill(opacity=0.1),
            *(
                ShowCreation(mob, run_time=15, rate_func=linear)
                for mob in [graph, shm_graph, cos_graph]
            ),
        )
