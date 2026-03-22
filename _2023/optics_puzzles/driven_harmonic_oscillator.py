from _2023.optics_puzzles.objects import TimeVaryingVectorField
from _2023.optics_puzzles.objects import Calcite
from _2023.optics_puzzles.objects import Sucrose
from manim_imports_ext import *


class HarmonicOscillator(TrueDot):
    def __init__(
        self,
        center=np.zeros(3),
        initial_velocity=np.zeros(3),
        k=20.0,
        damping=0.1,
        mass=1.0,
        radius=0.5,
        color=BLUE,
        three_d=True,
        **kwargs
    ):
        self.k = k
        self.mass = mass
        self.damping = damping
        self.velocity = initial_velocity
        self.center_of_attraction = center
        self.external_forces = []

        super().__init__(
            radius=radius,
            color=color,
            **kwargs
        )
        if three_d:
            self.make_3d()
        self.move_to(center)
        self.add_updater(lambda m, dt: self.update_position(dt))

    def update_position(self, dt):
        time_step = 1 / 300
        n_divisions = max(int(dt / time_step), 1)
        true_step = dt / n_divisions
        for _ in range(n_divisions):
            self.velocity += self.get_acceleration() * true_step
            self.shift(self.velocity * true_step)

    def get_displacement(self):
        return self.get_center() - self.center_of_attraction

    def get_acceleration(self):
        disp = self.get_displacement()
        result = -self.k * disp / self.mass - self.damping * self.velocity
        for force in self.external_forces:
            result += force() / self.mass
        return result

    def reset_velocity(self):
        self.velocity = 0

    def set_damping(self, damping):
        self.damping = damping

    def set_k(self, k):
        self.k = k
        return self

    def suspend_updating(self):
        super().suspend_updating()
        self.reset_velocity()

    def set_external_forces(self, funcs):
        self.external_forces = list(funcs)
        return self

    def add_external_force(self, func):
        self.external_forces.append(func)
        return self


class Spring(VMobject):
    def __init__(
        self, mobject, base_point,
        edge=ORIGIN,
        stroke_color=GREY,
        stroke_width=2,
        twist_rate=8.0,
        n_twists=8,
        radius=0.1,
        lead_length=0.25,
        **kwargs
    ):
        super().__init__(**kwargs)

        helix = ParametricCurve(
            lambda t: [
                radius * math.cos(TAU * t),
                radius * math.sin(TAU * t),
                t / twist_rate
            ],
            t_range=(0, n_twists, 0.01)
        )
        helix.rotate(PI / 2, UP)
        # helix.make_jagged()

        self.start_new_path(helix.get_start() + lead_length * LEFT)
        self.add_line_to(helix.get_start())
        self.append_vectorized_mobject(helix)
        self.add_line_to(helix.get_end() + lead_length * RIGHT)

        self.set_stroke(color=stroke_color, width=stroke_width)
        self.set_flat_stroke(False)

        reference_points = self.get_points().copy()
        width = self.get_width()
        self.add_updater(lambda m: m.set_points(reference_points))
        self.add_updater(lambda m: m.stretch(
            get_norm(base_point - mobject.get_edge_center(edge)) / width, 0
        ))
        self.add_updater(lambda m: m.put_start_and_end_on(
            base_point, mobject.get_edge_center(edge)
        ))

    def get_length(self):
        return get_norm(self.get_end() - self.get_start())


class DynamicPlot(VMobject):
    def __init__(
        self,
        axes,
        func,
        stroke_color=RED,
        stroke_width=3,

    ):
        self.axes = axes
        self.func = func
        self.time = 0
        super().__init__(
            stroke_color=stroke_color,
            stroke_width=stroke_width,
        )
        self.reset()
        self.add_updater(lambda m, dt: m.add_point(dt))

    def add_point(self, dt):
        self.time += dt
        if self.time < self.axes.x_axis.x_max:
            self.add_line_to(self.axes.c2p(self.time, self.func()))
        return self

    def reset(self):
        self.clear_points()
        self.time = 0
        self.start_new_path(self.axes.c2p(0, self.func()))
        return self


class DrivenHarmonicOscillator(InteractiveScene):
    def construct(self):
        # Zoom in on a plane of charges
        frame = self.frame
        frame.reorient(-90, 80, 90)

        zoom_out_radius = 0.035
        radius = 0.2
        charges = DotCloud(color=BLUE_D, radius=zoom_out_radius)
        charges.to_grid(31, 31)
        charges.set_height(3)
        charges.set_radius(zoom_out_radius)
        charges.make_3d()

        self.add(charges)
        in_shift = 0.01 * IN
        charges_target_height = 59
        self.play(
            frame.animate.reorient(-90, 0, 90).center().set_focal_distance(100).set_anim_args(time_span=(0, 4)),
            charges.animate.center().set_height(charges_target_height).set_radius(radius).shift(in_shift),
            run_time=6,
        )
        frame.reorient(0, 0, 0)

        # Add springs
        sho = HarmonicOscillator(
            center=charges.get_center() - in_shift,
            radius=radius,
            color=charges.get_color(),
        )
        cover = BackgroundRectangle(sho, buff=0.1)
        cover.set_fill(BLACK, 1)
        spacing = get_norm(charges.get_points()[1] - charges.get_points()[0])
        small_radius = 0.35 * radius
        springs = VGroup(*(
            Spring(sho, sho.get_center() + (spacing - small_radius) * vect)
            for vect in compass_directions(4)
        ))
        sho.add_updater(lambda m: springs.update())

        self.add(cover, springs, sho)
        self.play(
            charges.animate.set_opacity(0.5).set_radius(small_radius),
            FadeIn(springs),
        )
        self.wait()

        # Show example oscillating
        self.play(sho.animate.shift(0.5 * UL))
        sho.suspend_updating()
        self.wait()
        sho.resume_updating()
        sho.set_damping(0.25)
        self.wait(8)

        sho.move_to(sho.center_of_attraction)
        sho.reset_velocity()

        # Add coordiante plane
        axes = Axes(axis_config=dict(tick_size=0.05))
        axes.set_stroke(width=1, opacity=1)
        axes.set_flat_stroke(False)
        axes.scale(spacing)
        self.add(axes, springs, sho)
        self.play(FadeIn(axes))
        sho.suspend_updating()
        self.play(sho.animate.move_to(0.75 * UP + 0.5 * LEFT))

        # Set up Hooke's law
        t2c = {
            R"\vec{\textbf{x}}(t)": RED,
            R"\vec{\textbf{v}}(t)": PINK,
            R"\vec{\textbf{F}}(t)": YELLOW,
            R"\vec{\textbf{a}}(t)": YELLOW,
            R"\frac{d^2 \vec{\textbf{x}}}{dt^2}(t) ": YELLOW,
            R"\omega_r": PINK,
            R"\omega_l": TEAL,
        }

        x_vect = always_redraw(lambda: Arrow(
            axes.c2p(0, 0), sho.get_center(),
            stroke_width=5, stroke_color=RED, buff=0
        ))

        equation = Tex(R"\vec{\textbf{F}}(t) = -k \vec{\textbf{x}}(t)", t2c=t2c)
        equation.move_to(axes.c2p(0.5, 0.5), LEFT)

        x_label = equation[R"\vec{\textbf{x}}(t)"]
        x_label.set_backstroke(BLACK, 5)
        x_label.save_state()
        x_label.next_to(x_vect, RIGHT, 0.05, DOWN)

        self.play(
            GrowArrow(x_vect),
            springs.animate.set_stroke(opacity=0.35)
        )
        self.play(Write(x_label))
        self.wait()
        self.play(sho.animate.move_to(DR), rate_func=there_and_back, run_time=3)
        self.wait()
        self.play(
            Write(equation[R"\vec{\textbf{F}}(t) = -k "]),
            Restore(x_label),
        )
        self.add(equation)

        # Show force vector
        sho.set_damping(0.005)
        F_vect = Vector(stroke_color=YELLOW)

        def update_F_vect(F_vect, vect_scale=0.04):
            center = sho.get_center()
            acc = sho.get_acceleration()
            F_vect.put_start_and_end_on(center, center + vect_scale * acc)

        F_vect.add_updater(update_F_vect)

        initial_position = 0.75 * UL

        self.play(
            FlashAround(equation[R"\vec{\textbf{F}}"]),
            ReplacementTransform(x_vect, F_vect, path_arc=PI),
        )
        self.wait()
        self.play(sho.animate.move_to(0.25 * UL))
        self.wait()
        self.play(sho.animate.move_to(initial_position))
        self.wait()
        sho.resume_updating()
        self.wait(6)

        # Show graphical solution
        up_shift = 1.5 * UP
        plot_rect, plot_axes, plot = self.get_plot_group(
            lambda: np.sign(sho.get_center()[1]) * get_norm(sho.get_center()),
        )
        plot_group1 = VGroup(plot_rect, plot_axes, plot)
        plot_group1.to_corner(UR, buff=0.1)
        plot_group1.shift(up_shift)

        sho.move_to(initial_position)
        sho.reset_velocity()
        plot.reset()
        self.add(*plot_group1)
        self.play(
            frame.animate.shift(up_shift),
            equation.animate.to_corner(UL).match_y(plot_rect),
            FadeIn(plot_rect),
            FadeIn(plot_axes),
            FadeOut(charges),
            run_time=2
        )
        self.wait(15)
        plot.suspend_updating()
        sho.suspend_updating()
        self.play(sho.animate.center(), run_time=2)

        # Show the equation for the solution
        tex_kw = dict(t2c=t2c)
        equations = VGroup(equation)
        equations.add(
            Tex(R"m \vec{\textbf{a}}(t) = -k \vec{\textbf{x}}(t)", **tex_kw),
            Tex(R"\frac{d^2 \vec{\textbf{x}}}{dt^2}(t) = -{k \over m} \vec{\textbf{x}}(t)", **tex_kw),
            Tex(R"\vec{\textbf{x}}(t) = \vec{\textbf{x}}_0 \cos( \sqrt{k \over m} \cdot t)", **tex_kw),
            Tex(R"\vec{\textbf{x}}(t) = \vec{\textbf{x}}_0 \cos(\omega_r t)", **tex_kw),
        )
        eq1, eq2, eq3, eq4, eq5 = equations

        eq2.next_to(eq1, DOWN, LARGE_BUFF)
        eq3.next_to(eq2, DOWN, LARGE_BUFF, aligned_edge=LEFT)
        eq4.next_to(eq2, DOWN, buff=0.75, aligned_edge=LEFT)
        eq5.move_to(eq4, LEFT)

        implies = Tex(R"\Downarrow").replicate(2)
        implies[0].move_to(VGroup(eq1, eq2))
        implies[1].move_to(VGroup(eq2, eq3))
        implies[1].match_x(implies[0])

        eq1_copy = eq1.copy()
        self.play(
            TransformMatchingTex(
                eq1_copy, eq2,
                matched_pairs=[
                    (eq1_copy[R"\vec{\textbf{F}}(t)"], eq2[R"\vec{\textbf{a}}(t)"]),
                    (eq1_copy[R"= -k \vec{\textbf{x}}(t)"], eq2[R"= -k \vec{\textbf{x}}(t)"]),
                ],
                run_time=1
            ),
            FadeIn(implies[0], 0.5 * DOWN)
        )
        self.wait()
        eq2_copy = eq2.copy()
        self.play(
            TransformMatchingTex(
                eq2_copy, eq3,
                matched_pairs=[
                    (eq2_copy[R"\vec{\textbf{a}}(t)"], eq3[R"\frac{d^2 \vec{\textbf{x}}}{dt^2}(t)"]),
                    (eq2_copy[R"k"], eq3["k"]),
                ],
                path_arc=PI / 4,
            ),
            FadeIn(implies[1])
        )
        self.wait()
        self.play(
            eq3.animate.move_to(eq2).align_to(eq1, LEFT),
            FadeOut(eq2),
            FadeOut(implies[1]),
        )

        # Show solution for a given initial condition
        initial_position = 0.5 * UR
        x0 = eq4[R"\vec{\textbf{x}}_0"]
        cos_part = eq4[R"\cos( \sqrt{k \over m} \cdot t)"]
        x0_copy = x0.copy()
        x0_copy.next_to(0.5 * initial_position, UL, buff=0.05)
        x0_copy.set_color(RED)
        x0_copy.set_backstroke(BLACK, 4)
        ic_label = Text("Initial condition")
        ic_label.next_to(initial_position, UR, MED_LARGE_BUFF)

        x0_rect = SurroundingRectangle(x0, buff=0.05)
        x0_rect.set_stroke(TEAL, 2)

        self.remove(F_vect)
        self.add(x_vect)
        self.play(
            FadeIn(ic_label),
            sho.animate.move_to(initial_position)
        )
        self.play(FadeIn(x0_copy, DOWN))
        self.wait()

        self.play(
            TransformFromCopy(x0_copy, x0),
            Write(implies[1]),
            *(
                TransformFromCopy(eq3[tex], eq4[tex])
                for tex in ["=", R"\vec{\textbf{x}}(t)"]
            )
        )
        self.play(
            FadeIn(cos_part, lag_ratio=0.1),
        )
        self.play(
            FlashAround(cos_part, color=RED, run_time=3, time_width=1.5),
        )
        self.wait()
        self.play(
            FadeTransform(ic_label, x0_rect)
        )
        self.play(
            plot.animate.stretch(0.5, 1),
            sho.animate.move_to(0.5 * initial_position),
            x0_copy.animate.next_to(0.25 * initial_position, UL, SMALL_BUFF),
            rate_func=there_and_back_with_pause,
            run_time=6,
        )

        # Reset
        self.play(
            FadeOut(plot),
            FadeOut(x0_copy),
            FadeOut(x_vect),
            run_time=2
        )
        self.wait()

        sho.resume_updating()
        plot.reset()
        plot.resume_updating()
        self.add(plot)

        # Describe frequency terms
        sqrt_km = eq4[R"\sqrt{k \over m}"]
        sqrt_km_rect = SurroundingRectangle(sqrt_km, buff=0.05)
        k_rect = SurroundingRectangle(eq4["k"])
        m_rect = SurroundingRectangle(eq4["m"])
        VGroup(sqrt_km_rect, k_rect, m_rect).set_stroke(TEAL, 2)

        for rect in k_rect, m_rect:
            rect.arrow = Vector(0.5 * UP)
            rect.arrow.match_color(rect)
            rect.arrow.next_to(rect, RIGHT, buff=SMALL_BUFF)

        self.play(ReplacementTransform(x0_rect, sqrt_km_rect))
        self.wait(10)
        plot.reset()
        original_k = sho.k
        sho.set_k(4 * original_k)
        sho.move_to(initial_position)
        sho.reset_velocity()
        self.play(
            ReplacementTransform(sqrt_km_rect, k_rect),
            GrowArrow(k_rect.arrow)
        )
        initial_position = sho.get_center()
        self.wait(5)
        self.wait_until(lambda: get_norm(sho.get_center() - initial_position) < 0.05)
        sho.set_k(0.5 * original_k)
        sho.move_to(initial_position)
        sho.reset_velocity()
        self.play(
            ReplacementTransform(k_rect, m_rect),
            FadeOut(k_rect.arrow),
            GrowArrow(m_rect.arrow),
        )
        self.wait(8)
        sho.set_k(original_k)

        # Define omega_0
        omega0_eq = Tex(R"\omega_r = \sqrt{k / m}")
        omega0_eq[R"\omega_r"].set_color(PINK)
        omega0_eq.next_to(eq5, DOWN, buff=1.25)
        omega0_name = TexText("``Resonant frequency''", font_size=36)
        omega0_name.next_to(omega0_eq, DOWN)

        plot.reset()
        self.play(
            FadeOut(m_rect),
            FadeOut(m_rect.arrow),
        )
        self.play(
            TransformFromCopy(sqrt_km, omega0_eq[R"\sqrt{k / m}"]),
            Write(omega0_eq[R"\omega_r = "]),
            TransformMatchingTex(eq4, eq5),
        )
        self.wait(2)
        self.play(FadeIn(omega0_name))
        self.wait(12)

        # Clean up solution
        corner_box = Rectangle(width=3, height=plot_rect.get_height())
        corner_box.set_fill(interpolate_color(BLUE_E, BLACK, 0.75), 1.0)
        corner_box.set_stroke(WHITE, 1)
        corner_box.match_y(plot_rect)
        corner_box.to_edge(LEFT, buff=SMALL_BUFF)

        free_solution = VGroup(eq1, implies[0], eq5, omega0_eq)
        free_solution.target = free_solution.generate_target()
        free_solution.target.arrange(DOWN)
        free_solution.target.set_height(0.8 * corner_box.get_height())
        free_solution.target.move_to(corner_box)

        sho.suspend_updating()
        self.play(
            FadeIn(corner_box),
            MoveToTarget(free_solution),
            *map(FadeOut, [eq3, implies[1], omega0_name]),
            sho.animate.center(),
        )
        corner_box.push_self_into_submobjects()
        corner_box.add(free_solution)
        self.wait()
        self.play(
            FadeOut(plot_group1),
            frame.animate.shift(-up_shift),
            corner_box.animate.shift(-up_shift),
            run_time=2
        )
        corner_box.fix_in_frame()

        # Write new equation with driving force
        free_label = Text("No external\nforces", font_size=36)
        free_label.next_to(corner_box, DOWN)
        t2c[R"\vec{\textbf{E}}_0"] = BLUE_D

        driven_eq = Tex(
            R"""
                \vec{\textbf{F}}(t) =
                - k \vec{\textbf{x}}(t)
                + \vec{\textbf{E}}_0 q \cos(\omega_l t)
            """,
            t2c=t2c
        )
        driven_eq.to_edge(UP)
        driven_eq.set_x(FRAME_WIDTH / 4)

        external_force = driven_eq[R"\vec{\textbf{E}}_0 q \cos(\omega_l t)"]
        external_force_rect = SurroundingRectangle(external_force, buff=SMALL_BUFF)
        external_force_rect.set_stroke(TEAL, 2)
        external_force_label = Text("Force from a\nlight wave", font_size=36)
        external_force_label.next_to(external_force_rect, DOWN)
        external_force_label.set_backstroke(BLACK, 7)

        self.play(FadeIn(free_label, lag_ratio=0.1))
        self.wait()
        self.play(TransformMatchingTex(eq1.copy(), driven_eq))

        driven_eq_group = VGroup(
            BackgroundRectangle(driven_eq, buff=0.5).set_fill(BLACK, 0.9),
            BackgroundRectangle(external_force_label, buff=0.5).set_fill(BLACK, 0.9),
            driven_eq, external_force_rect,
            external_force_label
        )
        driven_eq_group.fix_in_frame()
        self.add(driven_eq_group)

        # Add a oscillating E field
        omega_tracker = ValueTracker(2.0)
        F_max = 0.5
        wave_number = 2.0
        axes.set_flat_stroke(False)

        def time_func(points, time):
            omega = omega_tracker.get_value()
            result = np.zeros(points.shape)
            result[:, 1] = F_max * np.cos(wave_number * points[:, 2] - omega * time)
            return result

        field_config = dict(
            stroke_color=TEAL,
            stroke_width=3,
            stroke_opacity=0.5,
            max_vect_len=1.0,
            x_density=1.0,
            y_density=1.0,
        )
        planar_field = TimeVaryingVectorField(
            time_func,
            **field_config
        )
        z_axis_field = TimeVaryingVectorField(
            time_func,
            height=0, width=0, depth=16,
            z_density=5,
            **field_config
        )
        full_field = TimeVaryingVectorField(
            time_func,
            depth=16,
            z_density=5,
            height=5,
            width=5,
            norm_to_opacity_func=lambda n: n,
            **field_config,
        )

        z_axis_field.set_stroke(opacity=1)
        full_field_opacity_mult = ValueTracker(0)

        def udpate_full_field_opacity(ff):
            ff.data["stroke_rgba"][:, 3] *= full_field_opacity_mult.get_value()

        full_field.add_updater(udpate_full_field_opacity)

        sho.set_k(8)
        sho.set_damping(1)
        sho.set_external_forces([
            lambda: 3 * planar_field.func(np.array([ORIGIN]))[0]
        ])
        sho.center()
        sho.reset_velocity()
        sho.resume_updating()
        self.add(planar_field, corner_box, driven_eq_group)
        self.play(
            VFadeIn(planar_field),
            FadeOut(corner_box, time_span=(0, 1)),
            FadeOut(free_label, time_span=(0, 1)),
        )
        self.wait(12)

        # Gather clean oscillations for B roll later
        self.remove(driven_eq_group)
        self.wait(30)
        self.add(driven_eq_group)

        self.play(
            frame.animate.reorient(-100, 100, 90),
            run_time=6
        )
        z_axis_field.time = full_field.time
        self.play(
            full_field_opacity_mult.animate.set_value(0),
            VFadeIn(z_axis_field),
            run_time=2
        )
        self.remove(full_field)
        self.play(
            frame.animate.reorient(-100, 80, 90),
            run_time=6
        )
        planar_field.set_stroke(opacity=0.5)
        self.play(
            frame.animate.reorient(-90, 0, 90).set_focal_distance(100).set_height(8),
            VFadeOut(z_axis_field),
            VFadeIn(planar_field),
            run_time=4
        )
        self.wait(4)

        # Change perspective a bunch
        full_field.time = planar_field.time
        frame.set_focal_distance(10)
        self.add(full_field)
        self.play(
            frame.animate.reorient(-100, 80, 90).set_height(10),
            full_field_opacity_mult.animate.set_value(1).set_anim_args(time_span=(2, 4)),
            VFadeOut(planar_field, time_span=(2, 4), remover=False),
            run_time=4,
        )
        planar_field.set_stroke(opacity=0)

        self.remove(driven_eq_group)
        full_field_opacity_mult.set_value(0)
        frame.to_default_state().reorient(-90, 0, 90)
        self.play(
            full_field_opacity_mult.animate.set_value(1).set_anim_args(time_span=(1, 4)),
            frame.animate.reorient(-95, 60, 90).set_height(10),
            run_time=4,
        )
        self.play(
            frame.animate.reorient(-100, 110, 90),
            run_time=12
        )
        self.play(
            frame.animate.reorient(-120, 80, 95),
            run_time=10
        )
        planar_field.time = full_field.time
        planar_field.set_stroke(opacity=1)
        self.play(
            frame.animate.reorient(-90, 0, 90).set_height(8),
            full_field_opacity_mult.animate.set_value(0).set_anim_args(time_span=(8, 10)),
            VFadeIn(planar_field, time_span=(8, 10)),
            run_time=12
        )
        self.add(driven_eq_group)
        self.wait(5)

        # Show graphical solution
        up_shift = UP
        driven_eq_group.unfix_from_frame()
        plot_rect, plot_axes, plot = self.get_plot_group(
            lambda: 2 * sho.get_y(),
            width=FRAME_WIDTH - 1,
            max_t=20
        )
        plot_group2 = VGroup(plot_rect, plot_axes, plot)
        plot_group2.to_edge(UP, buff=SMALL_BUFF).shift(up_shift)

        plot_box1, plot_box2 = plot_boxes = Rectangle().replicate(2)
        plot_boxes.match_height(plot_rect)
        plot_boxes.set_stroke(width=0)
        plot_boxes.set_fill(opacity=0.25)
        plot_boxes.set_submobject_colors_by_gradient(GREY_BROWN, TEAL)
        for box, width, x in zip(plot_boxes, (5, 15.5), (0, 5)):
            box.set_width(width * plot_axes.x_axis.get_unit_size(), stretch=True)
            box.move_to(plot_axes.c2p(x, 0), LEFT)

        sho.suspend_updating()
        self.play(
            frame.animate.shift(up_shift),
            FadeIn(plot_rect),
            FadeIn(plot_axes),
            driven_eq_group.animate.shift(1.0 * DOWN),
            sho.animate.center(),
            VFadeOut(planar_field, time_span=(0, 1)),
            run_time=2,
        )
        self.wait()

        planar_field.time = 0
        sho.resume_updating()
        plot.reset()
        self.add(planar_field, driven_eq_group, plot_rect, plot_axes, plot)
        self.play(VFadeIn(planar_field))
        self.wait(5)
        self.play(FadeIn(plot_box1))
        self.wait(4)
        self.play(FadeIn(plot_box2))
        self.wait(9)
        plot.suspend_updating()
        self.wait(3)

        # Compare with the previous plot
        self.play(
            VFadeOut(axes),
            VFadeOut(planar_field),
            FadeOut(sho),
            VFadeOut(springs),
        )

        down_shift = 2.5 * DOWN
        plot_group2.add(*plot_boxes)
        plot_group1.next_to(plot_group2, UP, aligned_edge=LEFT).shift(down_shift)
        top_axes = plot_group1[1]
        VGroup(top_axes.x_axis, plot_group1[2]).stretch(
            plot_axes.x_axis.get_unit_size() / top_axes.x_axis.get_unit_size(),
            0, about_edge=LEFT
        )
        top_axes[-2].match_x(top_axes.x_axis.get_right())

        corner_box.unfix_from_frame()
        corner_box.next_to(plot_group1, RIGHT)
        self.remove(*driven_eq_group[:2])
        self.play(
            FadeIn(plot_group1),
            FadeIn(corner_box),
            plot_group2.animate.shift(down_shift),
            driven_eq.animate.shift(down_shift),
            FadeOut(driven_eq_group[-2:], down_shift),
            run_time=3
        )
        self.wait()

        # Emphasize different frequencies
        omega0_eq_copy = omega0_eq.copy()
        omega_copy = driven_eq[R"\omega_l"].copy()

        self.play(
            omega0_eq_copy.animate.set_height(0.4).move_to(plot_group1, UR).shift(SMALL_BUFF * DL)
        )
        self.wait()
        self.play(
            omega_copy.animate.move_to(plot_axes.c2p(14, 0.8))
        )
        self.wait()

        # Show equation for the solution
        driven_eq.target = driven_eq.generate_target()
        driven_eq.target.to_edge(LEFT, buff=MED_SMALL_BUFF)
        driven_eq.target.shift(0.5 * DOWN)
        implies = Tex(R"\Rightarrow", font_size=72)
        implies.next_to(driven_eq.target, RIGHT, MED_LARGE_BUFF)

        solution = Tex(
            R"""
                \vec{\textbf{x}}(t) = 
                \frac{q \vec{\textbf{E}}_0}{m\left(\omega_r^2-\omega_l^2\right)}
                \cos(\omega_l t)
            """,
            t2c=t2c
        )
        solution.next_to(implies, RIGHT, MED_LARGE_BUFF)
        implies.match_y(solution)

        self.play(
            MoveToTarget(driven_eq),
            FadeIn(implies, LEFT),
            FadeIn(solution, RIGHT),
        )
        self.wait()

        # Comment on the equation
        full_rect = SurroundingRectangle(solution)
        amp_rect = SurroundingRectangle(solution[R"\frac{q \vec{\textbf{E}}_0}{m\left(\omega_r^2-\omega_l^2\right)}"])
        E_rect = SurroundingRectangle(solution[R"\vec{\textbf{E}}_0"])
        q_rect = SurroundingRectangle(solution[6])
        freq_diff_rect = SurroundingRectangle(
            solution[R"\omega_r^2-\omega_l^2"],
            buff=0.05
        )
        lil_rects = VGroup(amp_rect, E_rect, q_rect, freq_diff_rect)
        steady_state_rect = plot_box2.copy().set_fill(opacity=0)
        VGroup(
            full_rect, amp_rect, E_rect,
            freq_diff_rect, steady_state_rect
        ).set_stroke(YELLOW, 2)

        self.play(ShowCreation(full_rect))
        self.wait()
        self.play(TransformFromCopy(full_rect, steady_state_rect))
        self.wait()
        self.play(
            ReplacementTransform(full_rect, amp_rect),
            FadeOut(steady_state_rect),
        )
        self.wait()
        for r1, r2 in zip(lil_rects, lil_rects[1:]):
            self.play(ReplacementTransform(r1, r2))
            self.wait()

        # Reintroduce oscillator
        plot_group2.add(omega_copy)
        plot_group2.target = plot_group2.generate_target()
        plot_group2.target.shift(1.75 * UP)

        top_rect = plot_rect.copy()
        top_rect.set_fill(BLACK, 1).set_stroke(width=0)
        top_rect.next_to(plot_group2.target, UP, buff=0)

        to_fade = VGroup(
            plot_group1, omega0_eq_copy, corner_box,
            driven_eq, implies,
        )

        self.add(planar_field, springs, top_rect, plot_group2, solution, freq_diff_rect)
        self.add(to_fade)
        planar_field.set_stroke(opacity=0)
        planar_field.suspend_updating()
        sho.center()
        sho.suspend_updating()

        self.play(
            frame.animate.shift(UP),
            solution.animate.move_to(top_rect),
            MaintainPositionRelativeTo(freq_diff_rect, solution),
            MoveToTarget(plot_group2),
            FadeOut(to_fade, UP),
            FadeIn(sho),
            VFadeIn(springs),
            VFadeIn(planar_field),
            run_time=2,
        )

        # Show strong resonance
        close_freq_words = Tex(R"\text{If } \omega_l \approx \omega_r", t2c=t2c)
        close_freq_words.next_to(plot_rect, DOWN, aligned_edge=LEFT)

        self.add(*plot_group2)
        self.play(LaggedStart(
            FadeOut(plot_boxes),
            FadeOut(plot),
            FadeOut(omega_copy),
            FadeIn(close_freq_words),
            Transform(
                freq_diff_rect,
                SurroundingRectangle(close_freq_words).set_stroke(width=0),
                remover=True
            ),
            *(
                TransformFromCopy(solution[tex][0], close_freq_words[tex][0])
                for tex in [R"\omega_r", R"\omega_l"]
            )
        ))
        self.wait()
        self.play(
            plot_axes.y_axis.animate.stretch(0.75, 1),
            plot_axes[-1].animate.shift(0.1 * DOWN + 0.4 * LEFT)
        )

        sho.set_k(16)
        sho.set_damping(0.25)
        sho.center()
        sho.reset_velocity()
        omega_tracker.set_value(4)
        planar_field.set_stroke(opacity=1)
        plot.reset()
        plot.resume_updating()
        sho.resume_updating()
        planar_field.resume_updating()
        self.add(plot)
        self.play(VFadeIn(planar_field))
        self.wait(30)

        # Out of sync frequencies
        half = Tex(R"0.5")
        omega_r = close_freq_words[R"\omega_r"][0]
        half.move_to(omega_r, LEFT)
        half.align_to(omega_r[0], DOWN)

        sho.suspend_updating()
        plot.suspend_updating()
        self.play(
            sho.animate.center(),
            planar_field.animate.set_opacity(0),
            FadeOut(plot),
        )

        self.play(
            Write(half),
            omega_r.animate.shift((half.get_width() + 0.05) * RIGHT)
        )
        close_freq_words.add(half)
        self.add(BackgroundRectangle(close_freq_words).set_fill(BLACK, 1), close_freq_words)
        self.play(FlashAround(VGroup(close_freq_words, half)))
        self.wait()

        plot.reset()
        plot.resume_updating()
        omega_tracker.set_value(2)
        sho.set_damping(0.25)
        sho.resume_updating()
        planar_field.set_stroke(opacity=1)
        self.add(plot)
        self.play(VFadeIn(planar_field))
        self.wait(19)
        plot.reset()
        self.wait(20)

    def scrap(self):
        ## Line 281, used for thumbnail ##
        self.remove(axes)
        self.remove(equation)
        self.add(sho)
        springs.set_stroke(GREY_B, 4, 1) 
        sho.shift(0.5 * LEFT)
        F_vect.set_stroke(width=10)
        self.frame.set_height(5)
        ############

        # For clean driven_eq
        self.clear()
        self.add(driven_eq)
        self.play(
            ShowCreation(external_force_rect),
            FadeIn(external_force_label, lag_ratio=0.1),
        )
        ###

        # Show damping (Right after show force vector)
        damp_term = Tex(R"- \mu \vec{\textbf{v}}(t)", t2c=t2c)
        damp_term.next_to(equation, RIGHT, SMALL_BUFF)
        damp_rect = SurroundingRectangle(damp_term)
        damp_rect.set_stroke(PINK, 2)
        damp_arrow = Vector(DOWN).next_to(damp_rect, UP)
        damp_arrow.match_color(damp_rect)

        up_shift = 1.5 * UP
        plot_rect, plot_axes, plot = self.get_plot_group(
            lambda: np.sign(sho.get_center()[1]) * get_norm(sho.get_center()),
            width=14,
            max_t=20,
        )
        plot_group1 = VGroup(plot_rect, plot_axes, plot)
        plot_group1.to_corner(UR, buff=0.1)
        plot_group1.shift(up_shift)

        sho.move_to(initial_position)
        sho.reset_velocity()
        plot.reset()
        self.add(*plot_group1)
        frame.shift(up_shift)
        self.add(
            plot_rect, plot_axes, plot
        )

        self.wait(3)
        sho.set_damping(0.5)
        self.play(Write(damp_term))
        self.play(ShowCreation(damp_rect), GrowArrow(damp_arrow))
        self.wait(17)
        ###

    def get_plot_group(
        self,
        func,
        width=10.0,
        height=2.0,
        max_t=12.0,
    ):
        plot_rect = Rectangle(width, height)
        plot_rect.set_fill(GREY_E, 1)
        plot_rect.set_stroke(WHITE, 1)

        plot_axes = Axes((0, max_t), (-1, 1), width=width - 1, height=height - 0.25)
        plot_axes.move_to(plot_rect)
        y_axis_label = Tex(R"x(t)", font_size=20)
        y_axis_label.set_color(RED)
        y_axis_label.next_to(plot_axes.y_axis.get_top(), RIGHT)
        t_axis_label = Tex("t", font_size=24)
        t_axis_label.next_to(plot_axes.x_axis.get_right(), DOWN)

        plot_axes.add(t_axis_label, y_axis_label)
        plot = DynamicPlot(plot_axes, func)

        return plot_rect, plot_axes, plot


class JigglesInCalcite(InteractiveScene):
    polarization_direction = 1

    def construct(self):
        # Set up crystal
        calcite = Calcite(height=8)
        calcite.center()

        index = 118
        calcium_center = calcite.balls.get_points()[index]
        radii = calcite.balls.get_radii()
        radii[index] = 0
        calcite.balls.set_radii(radii)

        calcium = HarmonicOscillator(center=calcium_center)
        calcium.set_radius(np.max(radii))
        calcium.set_color(GREEN)
        calcium.set_glow_factor(calcite.balls.get_glow_factor())
        calcium.move_to(calcium_center)

        self.add(calcite, calcium)

        # Initial panning
        frame = self.frame
        frame.reorient(12, 64, 0).move_to([0.21, -0.18, -0.77]).set_height(9)
        self.play(
            frame.animate.reorient(1, 84, 0).move_to([-0.08, -0.16, -0.53]).set_height(9),
            run_time=3
        )
        self.wait()

        # Add springs
        spring_length = 2.5
        springs = VGroup(
            *(
                Spring(
                    calcium,
                    calcium.get_center() + spring_length * (v_vect + 0.2 * h_vect),
                    edge=v_vect
                )
                for v_vect in [UP, DOWN]
                for h_vect in [LEFT, ORIGIN, RIGHT]
            ),
            *(
                Spring(
                    calcium,
                    calcium.get_center() + spring_length * h_vect,
                    edge=h_vect
                )
                for h_vect in [LEFT, RIGHT]
            )
        )
        springs.set_stroke(opacity=0.7)
        self.play(
            VFadeIn(springs),
            calcium.animate.shift(RIGHT),
            calcite.balls.animate.set_opacity(0.1),
            frame.animate.reorient(-2, 25, 0).move_to(calcium).set_height(6),
            run_time=2,
        )

        # Show two resonant frequencies
        def wait_until_centered():
            disp = calcium.get_displacement()
            self.wait_until(lambda: np.dot(calcium.get_displacement(), disp) <= 0)
            calcium.move_to(calcium_center)
            calcium.reset_velocity()

        for vect, k in [(RIGHT, 5), (UP, 30)]:
            self.play(calcium.animate.move_to(calcium_center + vect), run_time=0.5)
            calcium.reset_velocity()
            calcium.set_k(k)
            self.wait(6)
            wait_until_centered()

        # Shine in light
        omega = -4.0
        F_max = 1.0
        wave_number = 2.0

        def time_func(points, time):
            result = np.zeros(points.shape)
            result[:, self.polarization_direction] = F_max * np.cos(wave_number * points[:, 2] - omega * time)
            return result

        field_config = dict(
            stroke_color=TEAL,
            stroke_width=3,
            stroke_opacity=0.5,
            max_vect_len=1.0,
            x_density=1.0,
            y_density=1.0,
            center=calcium_center,
        )
        z_axis_field = TimeVaryingVectorField(
            time_func,
            height=0, width=0, depth=16,
            z_density=5,
            **field_config
        )

        z_axis_field.set_stroke(opacity=0.5)

        calcium.set_k([5, 30][self.polarization_direction])
        calcium.set_damping(1)
        calcium.set_external_forces([
            lambda: 3 * z_axis_field.func(np.array([calcium_center]))[0]
        ])

        self.play(
            VFadeIn(z_axis_field),
            frame.animate.reorient(108, 46, -102).move_to(calcium).set_height(12),
            run_time=3
        )
        self.wait(25)


class JigglesInCalciteY(JigglesInCalcite):
    polarization_direction = 0


class SpiralPaths(InteractiveScene):
    default_frame_orientation = (-30, 70)
    color = RED
    sign = 1

    def construct(self):
        # Sucrose
        sucrose = Sucrose()
        sucrose.rotate(PI / 2)
        sucrose.set_height(7)
        sucrose.set_opacity(0.2)
        self.add(sucrose)

        # Frame motion
        frame = self.frame
        frame.add_updater(lambda t: t.reorient(-30 * math.sin(0.1 * self.time)))

        # Spiral
        helix = ParametricCurve(
            lambda t: [
                math.cos(self.sign * t),
                math.sin(self.sign * t),
                0.25 * t
            ],
            t_range=(-TAU, TAU, 0.01)
        )
        line = Line(helix.get_end(), helix.get_start())
        spiral = VGroup(helix, line)
        spiral.rotate(PI / 2, LEFT, about_point=ORIGIN)
        spiral.set_height(5, stretch=True)
        spiral.center()
        spiral.set_stroke(self.color, 1)
        spiral.set_flat_stroke(False)

        self.add(spiral)

        charge = Group(
            GlowDot(color=self.color),
            TrueDot(color=self.color, radius=0.15),
        )
        self.add(charge)
        for _ in range(5):
            self.play(MoveAlongPath(charge, helix, run_time=3))
            self.play(MoveAlongPath(charge, line, run_time=2))


class SpiralPathsLeftHanded(SpiralPaths):
    sign = -1
    color = YELLOW
