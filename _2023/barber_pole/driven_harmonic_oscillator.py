from operator import eq
from _2023.barber_pole.objects import TimeVaryingVectorField
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
        self.k = k / mass
        self.damping = damping
        self.velocity = initial_velocity
        self.center_of_attraction = center
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
        time_step = 0.01
        n_divisions = max(int(dt / time_step), 1)
        true_step = dt / n_divisions
        for _ in range(n_divisions):
            self.velocity += self.get_acceleration() * true_step
            self.shift(self.velocity * true_step)

    def get_acceleration(self):
        rel_x = self.get_center() - self.center_of_attraction
        return -self.k * rel_x - self.damping * self.velocity

    def reset_velocity(self):
        self.velocity = 0

    def set_damping(self, damping):
        self.damping = damping

    def set_k(self, k):
        self.k = k

    def suspend_updating(self):
        super().suspend_updating()
        self.reset_velocity()


class Spring(VMobject):
    def __init__(
        self, mobject, base_point,
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
            t_range=(0, n_twists, 0.05)
        )
        helix.rotate(PI / 2, UP)
        helix.make_jagged()

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
            get_norm(base_point - mobject.get_center()) / width, 0
        ))
        self.add_updater(lambda m: m.put_start_and_end_on(
            base_point, mobject.get_center()
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
        frame.reorient(-90, 70, 90)

        radius = 0.2
        charges = DotCloud(color=BLUE_D, radius=radius)
        charges.to_grid(11, 11)
        charges.make_3d()

        self.add(charges)
        in_shift = 0.01 * IN
        self.play(
            frame.animate.reorient(-90, 0, 90).set_focal_distance(100),
            charges.animate.scale(4).set_radius(radius).shift(in_shift),
            run_time=4,
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
        globals().update(locals())
        small_radius = 0.35 * radius
        springs = VGroup(*(
            Spring(sho, sho.get_center() + (spacing - small_radius) * vect)
            for vect in compass_directions(4)
        ))

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
        axes.scale(spacing)
        self.add(axes, springs, sho)
        self.play(FadeIn(axes))
        sho.suspend_updating()
        self.play(sho.animate.move_to(0.75 * UP + 0.5 * LEFT))

        # Set up Hooke's law
        sho.set_damping(0.005)
        sho.set_k(4.0)

        x_vect = Arrow(
            axes.c2p(0, 0), sho.get_center(),
            stroke_width=5, stroke_color=RED, buff=0
        )

        equation = Tex(
            R"\vec{\textbf{F}}(t) = -k \vec{\textbf{x}}(t)",
            t2c={R"\vec{\textbf{F}}(t)": YELLOW, R"\vec{\textbf{x}}(t)": RED}
        )
        equation.to_corner(UL)

        x_label = equation[R"\vec{\textbf{x}}(t)"]
        x_label.save_state()
        x_label.next_to(x_vect.pfp(0.45), UR, SMALL_BUFF)

        self.play(
            GrowArrow(x_vect),
            springs.animate.set_stroke(opacity=0.35)
        )
        self.play(Write(x_label))
        self.wait()
        self.play(
            Write(equation[R"\vec{\textbf{F}}(t) = -k "]),
            Restore(x_label),
        )
        self.add(equation)

        # Show force vector
        F_vect = Vector(stroke_color=YELLOW)

        def update_F_vect(F_vect, vect_scale=0.2):
            center = sho.get_center()
            acc = sho.get_acceleration()
            F_vect.put_start_and_end_on(center, center + vect_scale * acc)

        F_vect.add_updater(update_F_vect)

        self.play(
            FlashAround(equation[R"\vec{\textbf{F}}"]),
            ReplacementTransform(x_vect, F_vect, path_arc=PI),
        )
        self.wait()
        sho.resume_updating()
        self.wait(8)

        # Show graphical solution
        up_shift = 1.5 * UP
        plot_rect = Rectangle(10, 2)
        plot_rect.set_fill(GREY_E, 1)
        plot_rect.set_stroke(WHITE, 1)
        plot_rect.to_corner(UR, buff=0.1)
        plot_rect.shift(up_shift)

        plot_axes = Axes((0, 12), (-1, 1), width=9, height=1.75)
        plot_axes.move_to(plot_rect)
        y_axis_label = Tex(R"x(t)", font_size=20)
        y_axis_label.match_color(x_label.family_members_with_points()[0])
        y_axis_label.next_to(plot_axes.y_axis.get_top(), RIGHT)
        t_axis_label = Tex("t", font_size=24)
        t_axis_label.next_to(plot_axes.x_axis.get_right(), DOWN)

        plot_axes.add(t_axis_label, y_axis_label)
        plot = DynamicPlot(plot_axes, lambda: np.sign(sho.get_center()[1]) * get_norm(sho.get_center()))

        plot_group = VGroup(plot_rect, plot_axes, plot)

        self.add(*plot_group)
        self.play(
            frame.animate.shift(up_shift),
            equation.animate.match_y(plot_rect),
            FadeIn(plot_rect),
            FadeIn(plot_axes),
            FadeOut(charges),
            run_time=2
        )
        self.wait(10)
        plot.suspend_updating()
        sho.suspend_updating()
        self.play(sho.animate.center(), run_time=2)

        # Show the equation for the solution
        tex_kw = dict(t2c={
            R"\vec{\textbf{x}}(t)": RED,
            R"\vec{\textbf{a}}(t)": YELLOW,
            R"\frac{d^2 \vec{\textbf{x}}}{dt^2}(t) ": YELLOW,
            R"\omega_0": PINK,
        })
        equations = VGroup(equation)
        equations.add(
            Tex(R"m \vec{\textbf{a}}(t) = -k \vec{\textbf{x}}(t)", **tex_kw),
            Tex(R"\frac{d^2 \vec{\textbf{x}}}{dt^2}(t) = -{k \over m} \vec{\textbf{x}}(t)", **tex_kw),
            Tex(R"\vec{\textbf{x}}(t) = \vec{\textbf{x}}_0 \cos( \sqrt{k \over m} \cdot t)", **tex_kw),
            Tex(R"\vec{\textbf{x}}(t) = \vec{\textbf{x}}_0 \cos(\omega_0 t)", **tex_kw),
        )
        eq1, eq2, eq3, eq4, eq5 = equations

        eq2.next_to(eq1, DOWN, LARGE_BUFF)
        eq3.move_to(eq2).align_to(eq1, LEFT)
        eq4.next_to(eq2, DOWN, buff=0.75, aligned_edge=LEFT)
        eq5.move_to(eq4, LEFT)

        implies = Tex(R"\Downarrow").replicate(2)
        implies[0].move_to(VGroup(eq1, eq2))
        implies[1].move_to(VGroup(eq3, eq4))

        eq1_copy = eq1.copy()
        self.play(
            TransformMatchingTex(
                eq1_copy, eq2,
                matched_pairs=[
                    (eq1_copy[R"\vec{\textbf{F}}(t)"], eq2[R"\vec{\textbf{a}}(t)"])
                ],
                run_time=1
            ),
            FadeIn(implies[0], 0.5 * DOWN)
        )
        self.wait()
        self.play(
            TransformMatchingTex(
                eq2, eq3,
                matched_pairs=[(
                    eq2[R"\vec{\textbf{a}}(t)"],
                    eq3[R"\frac{d^2 \vec{\textbf{x}}}{dt^2}(t)"],
                )],
                path_arc=PI / 4,
            )
        )
        self.wait()
        self.play(
            FadeIn(eq4, DOWN),
            FadeIn(implies[1], 0.5 * DOWN),
        )
        self.wait()

        # Highlight initial position
        x0_rect = SurroundingRectangle(eq4[R"\vec{\textbf{x}}_0"], buff=0.05)
        x0_rect.set_stroke(TEAL, 2)

        self.remove(F_vect)
        self.play(ShowCreation(x0_rect))
        self.play(
            sho.animate.shift(0.5 * UR),
            FadeOut(plot),
            run_time=2
        )
        self.wait()

        sho.resume_updating()
        plot.reset()
        plot.resume_updating()
        self.add(plot)
        self.wait(6)

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
        self.wait(6)
        plot.reset()
        sho.set_k(16)
        self.play(
            ReplacementTransform(sqrt_km_rect, k_rect),
            GrowArrow(k_rect.arrow)
        )
        self.wait(5)
        sho.set_k(4)
        self.play(
            ReplacementTransform(k_rect, m_rect),
            FadeOut(k_rect.arrow),
            GrowArrow(m_rect.arrow),
        )
        self.wait(5)

        # Define omega_0
        omega0_eq = Tex(R"\omega_0 = \sqrt{k / m}")
        omega0_eq[R"\omega_0"].set_color(PINK)
        omega0_eq.next_to(eq5, DOWN, buff=1.25)

        plot.reset()
        self.play(
            FadeOut(m_rect),
            FadeOut(m_rect.arrow),
        )
        self.play(
            TransformFromCopy(sqrt_km, omega0_eq[R"\sqrt{k / m}"]),
            Write(omega0_eq[R"\omega_0 = "]),
            TransformMatchingTex(eq4, eq5),
        )
        self.wait(10)

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
            *map(FadeOut, [eq3, implies[1]]),
            sho.animate.center(),
        )
        corner_box.push_self_into_submobjects()
        corner_box.add(free_solution)
        self.wait()
        self.play(
            FadeOut(plot_group),
            frame.animate.shift(-up_shift),
            corner_box.animate.shift(-up_shift),
            run_time=2
        )
        corner_box.fix_in_frame()

        # Add a driving force (E field)
        omega = 2.0
        F_max = 0.5
        wave_number = 2.0

        def time_func(points, time):
            result = np.zeros(points.shape)
            result[:, 1] = F_max * np.cos(wave_number * points[:, 2] - omega * time)
            return result

        field_config = dict(
            stroke_color=TEAL,
            stroke_width=2,
            stroke_opacity=0.5,
        )
        planar_field = TimeVaryingVectorField(time_func, **field_config)
        z_axis_field = TimeVaryingVectorField(
            time_func,
            height=0, width=0, depth=16,
            z_density=5,
            **field_config
        )

        self.add(z_axis_field, corner_box)
        self.play(
            frame.animate.reorient(-90, -80, 90).set_focal_distance(10),
            VFadeIn(z_axis_field, time_span=(0, 1)),
            run_time=3,
        )
        self.wait(3)

        self.play(VFadeIn(z_axis_field))
        self.wait(6)




        # Show graphical solution
        # Show equation for the solution (Work it out on paper?)
        pass
