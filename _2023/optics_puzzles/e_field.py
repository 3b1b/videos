from manim_imports_ext import *
from _2023.optics_puzzles.objects import *



def get_influence_ring(center_point, color=WHITE, speed=2.0, max_width=3.0, width_decay_exp=0.5):
    ring = Circle()
    ring.set_stroke(color)
    ring.move_to(center_point)
    ring.time = 0

    def update_ring(ring, dt):
        ring.time += dt
        radius = ring.time * speed
        ring.set_width(max(2 * radius, 1e-3))
        ring.set_stroke(width=max_width / (1 + radius)**width_decay_exp)
        return ring

    ring.add_updater(update_ring)
    return ring


# Scenes


class TestFields(InteractiveScene):
    def construct(self):
        # Test coulomb field
        particles = ChargedParticle(rotation=0).replicate(1)
        particles.arrange(DOWN)
        particles.move_to(6 * LEFT)

        field = CoulombField(*particles)

        self.add(field, particles)
        self.play(particles.animate.move_to(0.2 * UP), run_time=3)

        self.clear()

        # Test Lorenz field
        def pos_func(time):
            return 0.1 * np.sin(5 * time) * OUT

        particle = ChargedParticle(
            rotation=0,
            radius=0.1,
            track_position_history=True
        )
        particles = particle.get_grid(20, 1, buff=0.25)
        particles.add_updater(lambda m: m.move_to(pos_func(self.time)))

        field = LorentzField(
            *particles,
            radius_of_suppression=1.0,
            x_density=4,
            y_density=4,
            max_vect_len=1,
            height=10,
        )
        field.set_stroke(opacity=0.7)

        self.frame.reorient(-20, 70)
        self.add(field, particles)
        self.wait(10)


class IntroduceEField(InteractiveScene):
    def construct(self):
        # Show two neighboring particles
        frame = self.frame
        frame.set_field_of_view(1 * DEGREES)

        charges = ChargedParticle(rotation=0).replicate(2)
        charges.arrange(RIGHT, buff=4)

        question = VGroup(
            Text("""
                How does the position
                and motion of this...
            """),
            Text("influence this?"),
        )
        for q, charge, vect in zip(question, charges, [LEFT, RIGHT]):
            q.next_to(charge, UP + vect, buff=1.0).shift(-2 * vect)

        question[1].align_to(question[0], DOWN)
        q0_bottom = question[0].get_bottom()
        arrow0 = always_redraw(lambda: Arrow(q0_bottom, charges[0]))
        arrow1 = Arrow(question[1].get_bottom(), charges[1])
        arrows = VGroup(arrow0, arrow1)

        self.play(LaggedStartMap(FadeIn, charges, shift=UP, lag_ratio=0.5))
        self.add(arrow0)
        self.play(
            Write(question[0]),
            charges[0].animate.shift(UR).set_anim_args(
                rate_func=wiggle,
                time_span=(1, 3),
            )
        )
        self.play(
            Write(question[1]),
            ShowCreation(arrow1),
        )
        self.wait()

        # Show force arrows
        def show_coulomb_force(arrow, charge1, charge2):
            root = charge2.get_center()
            vect = 4 * coulomb_force(
                charge2.get_center()[np.newaxis, :],
                charge1
            )[0]
            arrow.put_start_and_end_on(root, root + vect)

        coulomb_vects = Vector(RIGHT, stroke_width=5, stroke_color=YELLOW).replicate(2)
        coulomb_vects[0].add_updater(lambda a: show_coulomb_force(a, *charges))
        coulomb_vects[1].add_updater(lambda a: show_coulomb_force(a, *charges[::-1]))

        self.add(*coulomb_vects, *charges)
        self.play(
            FadeOut(question, time_span=(0, 1)),
            FadeOut(arrows, time_span=(0, 1)),
            charges.animate.arrange(RIGHT, buff=1.25),
            run_time=2
        )

        # Show force word
        force_words = Text("Force", font_size=48).replicate(2)
        force_words.set_fill(border_width=1)
        fw_width = force_words.get_width()

        def place_force_word_on_arrow(word, arrow):
            word.set_width(min(0.5 * arrow.get_width(), fw_width))
            word.next_to(arrow, UP, buff=0.2)

        force_words[0].add_updater(lambda w: place_force_word_on_arrow(w, coulomb_vects[0]))
        force_words[1].add_updater(lambda w: place_force_word_on_arrow(w, coulomb_vects[1]))

        self.play(LaggedStartMap(FadeIn, force_words, run_time=1, lag_ratio=0.5))
        self.add(force_words, charges)
        self.wait()

        # Add distance label
        d_line = always_redraw(lambda: DashedLine(
            charges[0].get_right(), charges[1].get_left(),
            dash_length=0.025
        ))
        d_label = Tex("r = 0.00", font_size=36)
        d_label.next_to(d_line, DOWN, buff=0.35)
        d_label.add_updater(lambda m: m.match_x(d_line))
        dist_decimal = d_label.make_number_changeable("0.00")

        def get_d():
            return get_norm(charges[0].get_center() - charges[1].get_center())

        dist_decimal.add_updater(lambda m: m.set_value(get_d()))

        # Show graph
        axes = Axes((0, 10), (0, 1, 0.25), width=10, height=5)
        axes.shift(charges[0].get_center() + 1 * UP - axes.get_origin())
        axes.add(
            Text("Distance", font_size=36).next_to(axes.c2p(10, 0), UP),
            Text("Force", font_size=36).next_to(axes.c2p(0, 0.8), LEFT),
        )
        graph = axes.get_graph(lambda x: 0.5 / x**2, x_range=(0.01, 10, 0.05))
        graph.make_jagged()
        graph.set_stroke(YELLOW, 2)

        graph_dot = GlowDot(color=WHITE)
        graph_dot.add_updater(lambda d: d.move_to(axes.i2gp(get_d(), graph)))

        d_label.update()
        self.play(
            frame.animate.move_to([3.5, 2.5, 0.0]),
            LaggedStart(
                FadeIn(axes),
                ShowCreation(graph),
                FadeIn(graph_dot),
                ShowCreation(d_line),
                FadeIn(d_label, 0.25 * UP),
            ),
            run_time=2,
        )
        self.wait()

        for buff in (0.4, 8, 1.25):
            self.play(
                charges[1].animate.next_to(charges[0], RIGHT, buff=buff),
                run_time=4
            )
            self.wait()

        # Write Coulomb's law
        coulombs_law = Tex(R"""
            F = {q_1 q_2 \over 4 \pi \epsilon_0} \cdot \frac{1}{r^2}
        """)
        coulombs_law_title = TexText("Coulomb's law")
        coulombs_law_title.move_to(axes, UP)
        coulombs_law.next_to(coulombs_law_title, DOWN, buff=0.75)

        rect = SurroundingRectangle(coulombs_law["q_1 q_2"])
        rect.set_stroke(YELLOW, 2)
        rect.set_fill(YELLOW, 0.25)

        self.play(
            FadeIn(coulombs_law_title),
            FadeIn(coulombs_law, UP),
        )
        self.wait()
        self.add(rect, coulombs_law)
        self.play(FadeIn(rect))
        self.wait()
        self.play(rect.animate.surround(coulombs_law[R"4 \pi \epsilon_0"]))
        self.wait()
        self.play(rect.animate.surround(coulombs_law[R"\frac{1}{r^2}"]))
        self.wait()
        self.play(charges[1].animate.next_to(charges[0], RIGHT, buff=3.0), run_time=3)
        self.play(FadeOut(rect))
        self.wait()

        # Remove graph
        d_line.clear_updaters()
        self.play(
            frame.animate.center(),
            VGroup(coulombs_law, coulombs_law_title).animate.to_corner(UL),
            LaggedStartMap(FadeOut, Group(
                axes, graph, graph_dot, d_line, d_label,
                force_words, coulomb_vects
            )),
            charges[0].animate.center(),
            FadeOut(charges[1]),
            run_time=2,
        )
        self.wait()

        # Show Coulomb's law vector field
        coulombs_law.add_background_rectangle()
        coulombs_law_title.add_background_rectangle()
        field = CoulombField(charges[0], x_density=3.0, y_density=3.0)
        dots = DotCloud(field.sample_points, radius=0.025, color=RED)
        dots.make_3d()

        self.add(dots, coulombs_law_title, coulombs_law)
        self.play(ShowCreation(dots))
        self.wait()
        self.add(field, coulombs_law_title, coulombs_law)
        self.play(FadeIn(field))
        for vect in [2 * RIGHT, 4 * LEFT, 2 * RIGHT]:
            self.play(charges[0].animate.shift(vect).set_anim_args(path_arc=PI, run_time=3))
        self.wait()

        # Electric field
        e_coulombs_law = Tex(R"""
            \vec{E}(\vec{r}) = {q \over 4 \pi \epsilon_0}
            \cdot \frac{1}{||\vec{r}||^2}
            \cdot \frac{\vec{r}}{||\vec{r}||}
        """)
        e_coulombs_law.move_to(coulombs_law, LEFT)
        ebr = BackgroundRectangle(e_coulombs_law)
        r_vect = Vector(2 * RIGHT + UP)
        r_vect.set_stroke(GREEN)
        r_label = e_coulombs_law[R"\vec{r}"][0].copy()
        r_label.next_to(r_vect.get_center(), UP, buff=0.1)
        r_label.set_backstroke(BLACK, 20)

        e_words = VGroup(
            Text("Electric Field:"),
            Text(
                """
                What force would be
                applied to a unit charge
                at a given point
                """,
                t2s={"would": ITALIC},
                t2c={"unit charge": RED},
                alignment="LEFT",
                font_size=36
            ),
        )
        e_words.set_backstroke(BLACK, 20)
        e_words.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        e_words.next_to(e_coulombs_law, DOWN, buff=0.5)
        e_words.to_edge(LEFT, buff=MED_SMALL_BUFF)

        rect.surround(e_coulombs_law[R"\vec{E}"])
        rect.scale(0.9, about_edge=DR)

        self.play(
            FadeOut(coulombs_law, UP),
            FadeIn(ebr, UP),
            FadeIn(e_coulombs_law, UP),
        )
        self.wait()
        self.add(ebr, rect, e_coulombs_law)
        self.play(FadeIn(rect))
        self.play(Write(e_words, stroke_color=BLACK))
        self.wait()
        self.play(
            FadeOut(e_words),
            rect.animate.surround(e_coulombs_law[R"(\vec{r})"][0], buff=0)
        )
        self.add(r_vect, charges[0])
        self.play(
            field.animate.set_stroke(opacity=0.4),
            FadeTransform(e_coulombs_law[R"\vec{r}"][0].copy(), r_label),
            ShowCreation(r_vect),
        )
        self.wait()
        self.play(
            rect.animate.surround(e_coulombs_law[R"\frac{\vec{r}}{||\vec{r}||}"])
        )
        self.wait()

        # Example E vect
        e_vect = r_vect.copy()
        e_vect.scale(0.25)
        e_vect.set_stroke(BLUE)
        e_vect.shift(r_vect.get_end() - e_vect.get_start())
        e_vect_label = Tex(R"\vec{E}", font_size=36)
        e_vect_label.set_backstroke(BLACK, 5)
        e_vect_label.next_to(e_vect.get_center(), UL, buff=0.1).shift(0.05 * UR)

        self.play(
            TransformFromCopy(r_vect, e_vect, path_arc=PI / 2),
            FadeTransform(e_coulombs_law[:2].copy(), e_vect_label),
            run_time=2
        )
        self.wait()

        # Not the full story!
        words = Text("Not the full story!", font_size=60)
        arrow = Vector(LEFT)
        arrow.next_to(coulombs_law_title, RIGHT)
        arrow.set_color(RED)
        words.set_color(RED)
        words.set_backstroke(BLACK, 20)
        words.next_to(arrow, RIGHT)
        charges[1].move_to(20 * RIGHT)

        self.remove(field)
        field = CoulombField(*charges, x_density=3.0, y_density=3.0)
        field.set_stroke(opacity=float(field.get_stroke_opacity()))
        self.add(field)

        self.play(
            FadeIn(words, lag_ratio=0.1),
            ShowCreation(arrow),
            FadeOut(rect),
            FadeOut(r_vect),
            FadeOut(r_label),
            FadeOut(e_vect),
            FadeOut(e_vect_label),
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeOut, Group(
                dots, coulombs_law_title, e_coulombs_law,
                words, arrow,
            )),
            FadeOut(ebr),
            charges[0].animate.to_edge(LEFT, buff=1.0),
            charges[1].animate.to_edge(RIGHT, buff=1.0),
            run_time=3,
        )

        # Wiggle here -> wiggle there
        tmp_charges = Group(*(ChargedParticle(track_position_history=True, charge=0.3) for x in range(2)))
        tmp_charges[0].add_updater(lambda m: m.move_to(charges[0]))
        tmp_charges[1].add_updater(lambda m: m.move_to(charges[1]))
        for charge in tmp_charges:
            charge.ignore_last_motion()
        lorentz_field = ColoumbPlusLorentzField(
            *tmp_charges,
            x_density=6.0,
            y_density=6.0,
            norm_to_opacity_func=lambda n: np.clip(0.5 * n, 0, 0.75)
        )
        self.remove(field)
        self.add(lorentz_field, *tmp_charges)

        influence_ring0 = self.get_influence_ring(charges[0].get_center()).set_stroke(opacity=0)
        influence_ring1 = self.get_influence_ring(charges[1].get_center()).set_stroke(opacity=0)
        dist = get_norm(charges[1].get_center() - charges[0].get_center())
        wiggle_kwargs = dict(
            rate_func=lambda t: wiggle(t, 3),
            run_time=1.5,
            suspend_mobject_updating=False,
        )

        self.add(influence_ring0, charges)
        self.play(charges[0].animate.shift(UP).set_anim_args(**wiggle_kwargs))
        self.wait_until(lambda: influence_ring0.get_radius() > dist, max_time=dist / 2.0)

        self.add(influence_ring1)
        self.play(charges[1].animate.shift(0.5 * DOWN).set_anim_args(**wiggle_kwargs))
        self.wait_until(lambda: influence_ring1.get_radius() > dist, max_time=dist / 2.0)
        self.play(charges[0].animate.shift(0.25 * UP).set_anim_args(**wiggle_kwargs))
        self.wait(6)
        self.play(
            FadeOut(influence_ring0),
            FadeOut(influence_ring1),
            FadeOut(lorentz_field)
        )
        self.remove(tmp_charges)

        # Show this the force
        ring = self.get_influence_ring(charges[0].get_center())

        ghost_charge = charges[0].copy().set_opacity(0.25)
        ghost_charge.shift(0.1 * IN)
        a_vect = Vector(UP).shift(charges[0].get_center())
        a_vect.set_stroke(PINK)
        a_label = Tex(R"\vec{a}(t_0)", font_size=48)
        a_label.set_color(PINK)
        a_label.next_to(a_vect, RIGHT, SMALL_BUFF)

        f_vect = Vector(1.0 * DOWN).shift(charges[1].get_center())
        f_vect.set_stroke(BLUE)
        f_label = Tex(R"\vec{F}(t)")
        f_label.set_color(BLUE)
        f_label.next_to(f_vect, LEFT, buff=0.15)

        time_label = Tex("t = 0.00")
        time_label.to_corner(UL)
        time_decimal = time_label.make_number_changeable("0.00")
        time_decimal.add_updater(lambda m: m.set_value(ring.time))

        start_point = charges[0].get_center().copy()
        speed = 2.0

        def field_func(points):
            time = ring.time
            diffs = (points - start_point)
            norms = np.linalg.norm(diffs, axis=1)
            past_times = time - (norms / speed)
            mags = np.exp(-3 * past_times)
            mags[past_times < 0] = 0
            return mags[:, np.newaxis] * DOWN

        field = VectorField(
            field_func,
            height=0,
            x_density=4.0,
            max_vect_len=1.0,
        )
        field.add_updater(lambda f: f.update_vectors())

        self.add(time_label, a_vect, a_label, charges)
        self.wait()
        self.add(ring, ghost_charge, field, charges)

        target = charges[0].get_center() + 2 * UP
        charges[0].add_updater(lambda m, dt: m.shift(3 * dt * (target - m.get_center())))
        self.wait_until(lambda: ring.get_radius() > dist)

        self.add(f_vect, f_label, charges)
        ring.suspend_updating()
        charges[0].suspend_updating()
        self.add(f_vect, charges[1])
        self.play(
            FadeIn(f_vect),
            FadeIn(f_label),
            FadeOut(field),
        )

        # Write the Lorentz force
        lorentz_law = Tex(R"""
            \vec{F}(t) = 
            {-q_1 q_2 \over 4\pi \epsilon_0 c^2}
            {1 \over r}
            \vec{a}_\perp(t - r / c)
        """)
        lorentz_law.to_edge(UP)
        lorentz_law[R"\vec{F}(t)"][0].match_style(f_label)

        a_hat_perp = lorentz_law[R"\vec{a}_\perp"][0]
        a_hat_perp.match_style(a_label)
        a_hat_perp.save_state()
        a_hat_perp[2].set_opacity(0)
        a_hat_perp[:2].move_to(a_hat_perp, RIGHT)
        a_hat_perp[:2].scale(1.25, about_edge=DR)

        lorentz_law["("][1].match_style(a_label)
        lorentz_law[")"][1].match_style(a_label)

        self.play(
            Transform(
                f_label.copy(),
                lorentz_law[R"\vec{F}(t)"][0].copy(),
                remover=True,
                run_time=1.5,
            ),
            FadeIn(lorentz_law, time_span=(1, 2))
        )
        self.wait()

        # Go through parts of the equation
        rect = SurroundingRectangle(lorentz_law["-q_1 q_2"])
        rect.set_stroke(YELLOW, 2)
        rect.set_fill(YELLOW, 0.2)

        r_line = DashedLine(ghost_charge.get_right(), charges[1].get_left())
        r_label = Tex("r").next_to(r_line, UP)

        self.add(rect, lorentz_law)
        self.play(FadeIn(rect))
        self.wait()
        self.play(rect.animate.surround(lorentz_law[R"4\pi \epsilon_0 c^2"]))
        self.wait()
        self.play(
            rect.animate.surround(lorentz_law[R"{1 \over r}"]),
            ShowCreation(r_line),
        )
        self.play(TransformFromCopy(lorentz_law[R"r"][1], r_label))
        self.wait()
        self.play(rect.animate.surround(lorentz_law[R"\vec{a}_\perp(t - r / c)"]))
        self.wait()
        self.play(rect.animate.surround(lorentz_law[R"t - r / c"], buff=0.05))
        self.wait()

        # Indicate back in time
        new_a_label = Tex(R"\vec{a}(t - r / c)")
        new_a_label.match_style(a_label)
        new_a_label.move_to(a_label, LEFT)

        ring.clear_updaters()
        time_decimal.clear_updaters()
        charges[0].clear_updaters()
        self.add(charges[0])
        self.play(
            ring.animate.scale(1e-3),
            UpdateFromFunc(time_decimal, lambda m: m.set_value(
                ring.get_radius() / 2
            )),
            charges[0].animate.shift(2 * DOWN).set_anim_args(
                time_span=(1, 4),
                rate_func=lambda t: smooth(t)**0.5,
            ),
            run_time=4,
        )
        time_decimal.set_value(0)
        self.play(
            TransformMatchingStrings(a_label, new_a_label),
            FadeOut(rect),
        )
        self.remove(rect)
        self.remove(ring)

        # Do another wiggle
        ring = self.get_influence_ring(charges[0].get_center())
        time_decimal.add_updater(lambda m: m.set_value(ring.time))

        self.add(ring)
        self.play(charges[0].animate.shift(UP).set_anim_args(**wiggle_kwargs))
        self.wait_until(lambda: ring.get_radius() > dist)
        self.play(charges[1].animate.shift(0.5 * DOWN).set_anim_args(**wiggle_kwargs))
        self.remove(ring)
        self.play(FadeOut(time_label))

        # Add back perpenducular part
        charges.target = charges.generate_target()
        charges.target.arrange(UR, buff=3).center()
        r_line.target = r_line.generate_target()
        r_line.target.become(DashedLine(
            charges.target[0].get_center(),
            charges.target[1].get_center(),
        ))
        f_vect.target = f_vect.generate_target()
        f_vect.target.rotate(45 * DEGREES)
        f_vect.target.shift(charges.target[1].get_center() - f_vect.target.get_start())
        rect = SurroundingRectangle(a_hat_perp.saved_state, buff=0.1)
        rect.set_stroke(YELLOW, 2)
        rect.set_fill(YELLOW, 0.25)

        self.add(rect, lorentz_law)
        self.play(FadeIn(rect, scale=0.5))
        self.play(Restore(a_hat_perp))
        self.wait()

        self.remove(ghost_charge)
        self.play(
            MoveToTarget(charges),
            MoveToTarget(r_line),
            MoveToTarget(f_vect),
            r_label.animate.next_to(r_line.target.get_center(), UL, SMALL_BUFF),
            f_label.animate.next_to(f_vect.target.get_center(), UR, buff=0),
            new_a_label.animate.next_to(charges.target[0], UL, buff=0),
            MaintainPositionRelativeTo(a_vect, charges[0]),
            run_time=2
        )
        self.wait()

        r_unit = normalize(charges[1].get_center() - charges[0].get_center())
        a_perp_vect = Vector(
            a_vect.get_vector() - np.dot(a_vect.get_vector(), r_unit) * r_unit,
        )
        a_perp_vect.match_style(a_vect)
        a_perp_vect.set_stroke(interpolate_color(PINK, WHITE, 0.5))
        a_perp_vect.shift(a_vect.get_end() - a_perp_vect.get_end())

        a_hat_perp2 = a_hat_perp.copy()
        a_hat_perp2.scale(0.9)
        a_hat_perp2.next_to(a_perp_vect.get_center(), UR, buff=0.1)
        a_hat_perp2.match_color(a_perp_vect)

        self.play(TransformFromCopy(a_vect, a_perp_vect))
        self.play(TransformFromCopy(a_hat_perp, a_hat_perp2))
        self.wait()
        rings = VGroup()
        for x in range(2):
            wiggle_kwargs = dict(
                run_time=2,
                rate_func=lambda t: wiggle(t, 5)
            )
            ring = self.get_influence_ring(charges[0].get_center())
            rings.add(ring)
            dist = get_norm(charges[0].get_center() - charges[1].get_center())

            self.add(ring)
            self.play(charges[0].animate.shift(0.5 * UP).set_anim_args(**wiggle_kwargs))
            self.wait_until(lambda: ring.get_radius() > dist)
            self.play(charges[1].animate.shift(0.25 * DR).set_anim_args(**wiggle_kwargs))
        self.play(FadeOut(rings))

        # Clear the canvas
        plane = NumberPlane(
            background_line_style=dict(stroke_color=GREY_D, stroke_opacity=0.75, stroke_width=1),
            axis_config=dict(stroke_opacity=(0.25))
        )
        new_lorentz = Tex(R"""
            \vec{E}_{\text{rad}}(\vec{r}, t) = 
            {-q \over 4\pi \epsilon_0 c^2}
            {1 \over ||\vec{r}||}
            \vec{a}_\perp(t - ||\vec{r}|| / c)
        """, font_size=36)
        new_lorentz.to_corner(UL)
        lhs = new_lorentz[R"\vec{E}_{\text{rad}}(\vec{r}, t)"]
        lhs.set_color(BLUE)
        new_lorentz[R"\vec{a}_\perp("].set_color(PINK)
        new_lorentz[R")"][1].set_color(PINK)

        lhs_rect = SurroundingRectangle(lhs)
        arrow = Vector(UP).next_to(lhs_rect, DOWN)

        self.add(plane, lorentz_law, *charges)
        self.remove(rect)
        self.play(
            LaggedStartMap(FadeOut, Group(
                r_line, r_label,
                a_hat_perp2, a_perp_vect,
                a_vect, new_a_label, new_a_label,
                f_vect, f_label, charges[1],
            )),
            FadeIn(plane, time_span=(1, 2)),
            charges[0].animate.center().set_anim_args(time_span=(1, 2)),
            FadeTransform(lorentz_law, new_lorentz),
        )
        self.play(
            ShowCreation(lhs_rect),
            GrowArrow(arrow),
        )
        self.wait()
        self.play(FadeOut(lhs_rect), FadeOut(arrow))

        # Show vector field
        charge = ChargedParticle(
            track_position_history=True
        )
        field = LorentzField(
            charge,
            stroke_width=3,
            x_density=4.0,
            y_density=4.0,
            max_vect_len=0.25,
            norm_to_opacity_func=lambda n: np.clip(1.5 * n, 0, 1),
        )
        a_vect = AccelerationVector(charge)
        small_charges = DotCloud(field.sample_points, radius=0.02)
        small_charges.match_color(charges[1][0])
        small_charges.make_3d()
        new_lorentz.set_backstroke(BLACK, 20)

        self.add(small_charges, new_lorentz)
        self.play(ShowCreation(small_charges))
        self.wait()

        self.remove(charges[0])
        self.add(field, a_vect, charge, new_lorentz)
        charge.ignore_last_motion()

        # Have some fun with the charge
        wiggle_kwargs = dict(
            rate_func=lambda t: wiggle(t, 3),
            run_time=3.0,
            suspend_mobject_updating=False,
        )
        lemniscate = ParametricCurve(
            lambda t: np.sin(t)**2 * (np.cos(t) * RIGHT + np.sin(t) * UP),
            t_range=(0, TAU, TAU / 200)
        )

        self.play(
            charge.animate.shift(0.4 * UP).set_anim_args(**wiggle_kwargs),
        )
        self.wait(3)
        self.play(
            MoveAlongPath(charge, lemniscate, run_time=6)
        )
        self.wait(3)
        for point in [2 * RIGHT, ORIGIN]:
            self.play(charge.animate.move_to(point).set_anim_args(path_arc=PI, run_time=5, suspend_mobject_updating=False))
        self.wait(5)

        # Set it oscillating
        charge.init_clock()
        charge.ignore_last_motion()
        charge.add_updater(lambda m: m.move_to(
            0.25 * np.sin(0.5 * TAU * m.get_internal_time()) * UP
        ))
        self.wait(30)

    def get_influence_ring(self, center_point, color=WHITE, speed=2.0, max_width=3.0, width_decay_exp=0.5):
        return get_influence_ring(center_point, color, speed, max_width, width_decay_exp)


class AltEFieldIntroduction(IntroduceEField):
    def construct(self):
        # Title
        title1 = TexText(R"Light $\rightarrow$ Wave in the Electromagnetic field")
        title2 = TexText(R"Electric field")
        title2.scale(1.5)
        VGroup(title1, title2).to_edge(UP)

        self.add(title1)
        self.wait()
        self.play(
            TransformMatchingStrings(title1, title2)
        )
        title2.add_to_back(BackgroundRectangle(title2))

        # Add field
        density = 4.0
        charges = Group(
            ChargedParticle(ORIGIN, 1),
            ChargedParticle(3 * UL, -1, color=BLUE, sign="-"),
            ChargedParticle([3, -2, 0], 2),
            ChargedParticle([5, 1, 0], -2, color=BLUE, sign="-"),
        )
        field_config = dict(
            x_density=density,
            y_density=density,
            width=2 * FRAME_WIDTH,
        )        
        c_field = CoulombField(*charges, **field_config)
        c_field_opacity_tracker = ValueTracker(0)
        c_field.add_updater(lambda m: m.set_stroke(opacity=c_field_opacity_tracker.get_value()))
        points = DotCloud(c_field.sample_points, radius=0.02)
        points.make_3d()
        points.set_color(BLUE)
        points.move_to(0.01 * IN)

        self.add(points, charges, title2)
        self.play(
            ShowCreation(points),
            FadeIn(charges),
        )
        self.add(points, c_field, charges, title2)
        self.play(
            charges[0].animate.shift(0.0001 * UP).set_anim_args(rate_func=there_and_back),
            c_field_opacity_tracker.animate.set_value(1),
        )
        self.wait()

        # Show example charge
        for charge in charges:
            charge.save_state()
            charge.target = charge.generate_target()
            charge.target.fade(0.5)
            charge.target.scale(1e-3)

        hyp_charge = ChargedParticle(sign="+1")
        hyp_charge.move_to(UR)
        vect = Vector(stroke_color=YELLOW)
        vect.charge = hyp_charge
        vect.field = c_field
        glow = GlowDot(hyp_charge.get_center(), color=RED, radius=0.5)
        hyp_charge.add_to_back(glow)

        hyp_words = Text("Hypothetical\nunit charge", font_size=24)
        hyp_words.set_backstroke(BLACK, 5)
        hyp_words.charge = hyp_charge
        hyp_words.add_updater(lambda m: m.next_to(m.charge, buff=-SMALL_BUFF))

        def update_vect(vect):
            p = vect.charge.get_center()
            vect.put_start_and_end_on(p, p + vect.field.func([p])[0])
            return vect

        vect.add_updater(update_vect)

        self.play(
            *map(MoveToTarget, charges),
            c_field_opacity_tracker.animate.set_value(0.5),
            VFadeIn(vect),
            FadeIn(hyp_charge),
            FadeIn(hyp_words),
        )
        for point in [UL, (-3, 2, 0), (2, -3, 0)]:
            self.play(hyp_charge.animate.move_to(point + 0.05 * OUT), run_time=5)

        # Emphasize coulombenss
        c1 = charges[0]
        c2 = ChargedParticle((FRAME_WIDTH - 2) * RIGHT)
        c1.charge = c2.charge = 0.5
        l_field = ColoumbPlusLorentzField(
            c1, c2,
            norm_to_opacity_func=lambda n: np.arctan(n),
            **field_config
        )
        self.play(
            FadeOut(points),
            ReplacementTransform(c_field, l_field),
            Restore(c1),
            FadeOut(hyp_charge),
            FadeOut(hyp_words),
            FadeOut(vect),
            *(
                c.animate.move_to(10 * c.get_center())
                for c in charges[1:]
            ),
        )
        self.add(l_field, charges[0], title2)
        self.wait()

        hyp_charge.next_to(charges[0], RIGHT, buff=0.5)
        vect.field = l_field
        self.play(
            VFadeIn(vect),
            FadeIn(hyp_charge),
        )
        self.play(
            hyp_charge.animate.next_to(charges[0], LEFT, buff=0.5).set_anim_args(path_arc=PI),
            run_time=8,
        )
        self.play(
            FadeOut(hyp_charge),
            FadeOut(vect),
            FadeOut(title2),
        )

        # Shake the charge
        def wiggle_charge(charge, direction, run_time=2):
            return charge.animate.shift(direction).set_anim_args(
                rate_func=lambda t: wiggle(t, 3),
                run_time=run_time,
                suspend_mobject_updating=False,
            )

        self.play(wiggle_charge(c1, 0.5 * UP))
        self.wait(5)

        # Indicate speed
        ring = self.get_influence_ring(c1.get_center())
        speed_words = Text("Speed = c", font_size=36)
        speed_words.set_backstroke(BLACK, 5)
        speed_words.ring = ring
        speed_words.add_updater(lambda m: m.next_to(m.ring.get_right(), LEFT, SMALL_BUFF))
        self.add(ring, speed_words)
        self.play(
            wiggle_charge(c1, 0.5 * UP),
            VFadeIn(speed_words)
        )
        self.wait(5)

        # Show second charge
        self.add(c2)
        self.play(
            self.frame.animate.move_to(Group(c1, c2)),
            run_time=3
        )

        amp = 0.5
        for sign in [1, -1, 1, -1]:
            q1, q2 = (c1, c2) if sign > 0 else (c2, c1)
            ring = self.get_influence_ring(q1.get_center())
            ring.set_stroke(opacity=0)
            dist = get_norm(q1.get_center() - q2.get_center())
            self.add(ring)
            self.play(wiggle_charge(q1, sign * amp * UP))
            self.wait_until(lambda: ring.get_radius() > dist)
            amp *= 0.4
        self.wait(4)


class TestForMithuna(InteractiveScene):
    def construct(self):
        # Setup
        n_rows = 10
        n_cols = 16
        n_charges = n_rows * n_cols
        charges = Group(*(
            ChargedParticle(
                track_position_history=True,
                charge=10 / n_charges,
                radius=0.1,
                show_sign=False,
            )
            for _ in range(n_charges)
        ))
        charges.arrange_in_grid(n_rows, n_cols, buff=0.5)
        charges.center()
        self.add(charges)

        columns = Group(*(charges[i::n_cols] for i in range(n_cols)))

        field = LorentzField(
            *charges,
            stroke_width=3,
            x_density=4.0,
            y_density=4.0,
            radius_of_suppression=0.1,
            # max_vect_len=np.inf,
            max_vect_len=None,
            norm_to_opacity_func=lambda n: np.clip(1.5 * n, 0, 0.7),
        )
        self.add(field)

        c_dot = GlowDot().get_grid(1, 100, buff=0.5)
        c_dot.move_to(charges)
        c_dot.add_updater(lambda m, dt: m.shift(field.c * dt * RIGHT))

        self.wait(0.1)
        # self.add(c_dot)
        self.play(LaggedStart(
            *(
                col.animate.shift(0.5 * UP).set_anim_args(
                    rate_func=lambda t: wiggle(t, 6),
                    suspend_mobject_updating=False,
                    run_time=8,
                )
                for col in columns
            ),
            lag_ratio=1 / n_cols
        ))
        self.wait(5)

        # # Rotate
        # self.play(
        #     Rotate(
        #         charges,
        #         TAU,
        #         # rate_func=wiggle,
        #         suspend_mobject_updating=False,
        #         run_time=5,
        #     )
        # )


class ShowTheEffectsOfOscillatingCharge(InteractiveScene):
    amplitude = 0.25
    frequency = 0.5
    direction = UP

    show_acceleration_vector = True
    origin = None

    axes_config = dict(
        axis_config=dict(stroke_opacity=0.7),
        x_range=(-10, 10),
        y_range=(-5, 5),
        z_range=(-3, 3),
    )
    particle_config = dict(
        track_position_history=True,
        radius=0.15,
    )
    acceleration_vector_config = dict()
    field_config = dict(
        max_vect_len=0.35,
        stroke_opacity=0.75,
        radius_of_suppression=1.0,
        height=10,
        x_density=4.0,
        y_density=4.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(2 * n, 0, 0.8)
    )
    field_class = LorentzField

    def setup(self):
        super().setup()
        self.add_axes()
        self.add_axis_labels(self.axes)
        self.add_particles(self.axes)
        self.add_field(self.particles)
        if self.show_acceleration_vector:
            self.add_acceleration_vectors(self.particles)

    def add_axes(self):
        self.axes = ThreeDAxes(**self.axes_config)
        if self.origin is not None:
            self.axes.shift(self.origin - self.axes.get_origin())
        self.add(self.axes)

    def add_axis_labels(self, axes):
        axis_labels = label = Tex("xyz")
        if axes.z_axis.get_stroke_opacity() > 0:
            axis_labels.rotate(PI / 2, RIGHT)
            axis_labels[0].next_to(axes.x_axis.get_right(), OUT)
            axis_labels[1].next_to(axes.y_axis.get_top(), OUT)
            axis_labels[2].next_to(axes.z_axis.get_zenith(), RIGHT)
        else:
            axis_labels[1].clear_points()
            axis_labels[0].next_to(axes.x_axis.get_right(), UP)
            axis_labels[2].next_to(axes.y_axis.get_top(), RIGHT)

        self.axis_labels = axis_labels
        self.add(self.axis_labels)

    def add_particles(self, axes):
        self.particles = self.get_particles()
        self.particles.add_updater(lambda m: m.move_to(
            axes.c2p(*self.oscillation_function(self.time))
        ))
        for particle in self.particles:
            particle.ignore_last_motion()
        self.add(self.particles)

    def get_particles(self):
        return Group(ChargedParticle(**self.particle_config))

    def add_field(self, particles):
        self.field = self.field_class(*particles, **self.field_config)
        self.add(self.field, particles)

    def add_acceleration_vectors(self, particles):
        self.acceleration_vectors = VGroup(*(
            AccelerationVector(particle)
            for particle in particles
        ))
        self.add(self.acceleration_vectors, self.particles)

    def oscillation_function(self, time):
        return self.amplitude * np.sin(TAU * self.frequency * time) * self.direction

    def construct(self):
        # Test
        self.wait(20)


class SendingLittlePulses(ShowTheEffectsOfOscillatingCharge):
    axes_config = dict(
        axis_config=dict(stroke_opacity=0.7),
        x_range=(0, 10),
        y_range=(-3, 3),
        z_range=(-1, 1),
    )
    field_config = dict(
        max_vect_len=0.25,
        stroke_opacity=0.75,
        radius_of_suppression=1.0,
        height=10,
        x_density=4.0,
        y_density=4.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(1.5 * n, 0, 0.8)
    )

    def add_axis_labels(self, axes):
        pass

    def construct(self):
        # Setup
        self.axes.z_axis.set_stroke(opacity=0)
        self.axes.y_axis[-1]
        particle = self.particles[0]
        particle.clear_updaters()

        # Test
        for _ in range(10):
            shake_size = 0.5 + random.random()
            self.play(
                particle.animate.shift(0.06 * shake_size * UP),
                rate_func=lambda t: wiggle(t, 2), run_time=(shake_size),
            )
            self.wait(random.choice([1, 2]))
        self.wait(4)


class OscillateOnYOneDField(ShowTheEffectsOfOscillatingCharge):
    origin = 5 * LEFT
    axes_config = dict(
        axis_config=dict(stroke_opacity=0.7),
        z_axis_config=dict(stroke_opacity=0),
        x_range=(-3, 12),
        y_range=(-3, 3)
    )
    field_config = dict(
        max_vect_len=1,
        stroke_opacity=1.0,
        radius_of_suppression=0.25,
        height=0,
        x_density=4.0,
        c=2.0,
        norm_to_opacity_func=None
    )

    def construct(self):
        # Start wiggling
        axes = self.axes
        field = self.field
        particles = self.particles

        points = DotCloud(field.sample_points, color=BLUE)
        points.make_3d()
        points.set_radius(0.03)
        field.suspend_updating()
        particles.suspend_updating()

        self.add(points, particles)
        self.play(ShowCreation(points))
        self.wait()
        self.time = 0
        particles.resume_updating()
        for particle in particles:
            particle.ignore_last_motion()
        field.resume_updating()
        self.wait(24.5)
        paused_time = float(self.time)

        # Zoom in
        field.suspend_updating()
        particles.suspend_updating()
        self.remove(particles)
        self.remove(field)
        field_copy = field.copy()
        field_copy.clear_updaters()
        particle = particles[0].copy()
        particle.clear_updaters()
        self.add(field_copy, particle)

        frame = self.frame
        particle.save_state()
        particle.target = particle.generate_target()
        particle.target[0].set_radius(0.075)
        particle.target[1].scale(0.5)
        particle.target[1].set_stroke(width=1)

        self.play(
            frame.animate.set_height(3, about_point=axes.get_origin()),
            MoveToTarget(particle),
            self.acceleration_vectors.animate.set_stroke(opacity=0.2),
            run_time=2
        )

        # Go through points
        last_line = VMobject()
        last_ghost = Group()
        step = get_norm(field.sample_points[0] - field.sample_points[1])
        for x in np.arange(1, 9):
            ghost = particle.copy()
            ghost.fade(0.5)
            dist = get_norm(axes.c2p(x * step, 0) - particle.get_center())
            ghost.move_to(particle.get_past_position(dist / field.c))
            line = Line(
                ghost.get_center(),
                axes.c2p(x * step, 0)
            )
            line.set_stroke(WHITE, 1)
            elbow = Elbow(width=0.1)
            angle = line.get_angle() + 90 * DEGREES
            if x > 3:
                angle += 90 * DEGREES
            elbow.rotate(angle, about_point=ORIGIN)
            elbow.shift(line.get_end())
            elbow.set_stroke(WHITE, 1)
            self.play(
                ShowCreation(line),
                FadeOut(last_line),
                FadeOut(last_ghost, scale=0),
                GrowFromCenter(ghost),
                FadeIn(elbow, time_span=(0.5, 1)),
            )
            self.wait(0.5)
            last_line = Group(line, elbow)
            last_ghost = ghost
        self.play(FadeOut(last_line))

        self.time = paused_time
        self.play(
            Restore(particle),
            frame.animate.to_default_state().set_anim_args(run_time=3)
        )
        self.remove(field_copy, particle)
        self.add(particles, field)
        self.wait(5)


class OscillateOnYTwoDField(ShowTheEffectsOfOscillatingCharge):
    particle_config = dict(
        track_position_history=True,
        radius=0.15,
    )
    field_config = dict(
        max_vect_len=0.25,
        stroke_opacity=0.75,
        radius_of_suppression=0.25,
        height=10,
        x_density=4.0,
        y_density=4.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(1.5 * n, 0, 1.0)
    )

    def construct(self):
        # Start wiggling
        axes = self.axes
        field = self.field
        particles = self.particles

        self.wait(60)


class DiscussDecay(OscillateOnYOneDField):
    def construct(self):
        # Start wiggling
        axes = self.axes
        particles = self.particles
        self.wait(8)

        # Show graph
        axes_config = dict(self.axes_config)
        axes_config.pop("z_axis_config")
        axes2d = Axes(**axes_config)
        axes2d.shift(axes.get_origin() - axes2d.get_origin())
        graph = axes2d.get_graph(lambda x: 2 / x, x_range=(0.5, 12))
        graph.set_stroke(TEAL, 2)

        words = TexText(R"Decays proportionally to $\frac{1}{r}$")
        words[R"$\frac{1}{r}$"].scale(1.5, about_edge=LEFT).set_color(TEAL)
        words.move_to(2 * UP)

        particles[0].ignore_last_motion()
        self.play(
            ShowCreation(graph),
            Write(words),
        )
        self.wait(20)


class ChargeOnZAxis(ShowTheEffectsOfOscillatingCharge):
    default_frame_orientation = (-20, 70)
    direction = OUT

    origin = ORIGIN

    axes_config = dict(
        axis_config=dict(stroke_opacity=0.7),
        x_range=(-8, 8),
        y_range=(-6, 6),
        z_range=(-3, 3),
    )
    particle_config = dict(
        show_sign=False,
        rotation=PI / 2,
        track_position_history=True,
        radius=0.2,
    )
    field_config = dict(
        max_vect_len=0.5,
        stroke_opacity=0.7,
        radius_of_suppression=1.0,
        width=40,
        height=40,
        depth=0,
        x_density=4.0,
        y_density=4.0,
        z_density=1.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(n, 0, 0.8)
    )

    def construct(self):
        # Test
        self.play(self.frame.animate.reorient(16, 71, 0), run_time=12)
        self.play(self.frame.animate.reorient(-15, 84, 0), run_time=6)
        self.play(self.frame.animate.reorient(-38, 64, 0), run_time=10)
        self.play(self.frame.animate.reorient(24, 66, 0), run_time=10)


class ThreeCharges(ChargeOnZAxis):
    def get_particles(self):
        return Group(*(
            ChargedParticle(**self.particle_config)
            for n in range(3)
        )).arrange(UP, buff=2)


class Introduce3dMovements(ChargeOnZAxis):
    axes_config = dict(
        axis_config=dict(stroke_opacity=0.7),
        x_range=(-5, 5),
        y_range=(-5, 5),
        z_range=(-3, 3),
    )

    def construct(self):
        charge = self.particles[0]
        self.remove(self.particles)
        self.add(charge)

        # Test
        kw = dict(suspend_mobject_updating=False)
        frame = self.frame
        frame.reorient(-13, 71, 0).move_to([-0.24, 0.12, 0.04]).set_height(4.88)

        self.play(
            Rotate(charge, TAU, axis=UP, about_point=RIGHT, **kw),
            self.frame.animate.reorient(-17, 71, 0).move_to([-0.12, 0.17, 0.27]).set_height(7.16),
            run_time=6,
        )
        self.play(
            self.frame.animate.reorient(-20, 69, 0).set_height(8).center(),
            Rotate(charge, TAU, axis=OUT, about_point=RIGHT, **kw),
            run_time=6,
        )
        # self.wait(4)
        self.play(
            charge.animate.shift(0.8 * (UP + OUT)).set_anim_args(rate_func=lambda t: wiggle(t, 6), **kw),
            self.frame.animate.reorient(20, 71, 0).move_to([0.31, 0.54, -0.3]).set_height(8.22),
            run_time=8,
        )
        self.play(
            charge.animate.shift(4 * DOWN).set_anim_args(rate_func=there_and_back, **kw),
            self.frame.animate.reorient(-21, 64, 0).move_to([0.31, 0.54, -0.3]).set_height(8.22),
            run_time=9,
        )
        self.wait(4)


class Introduce3dMovements3DVects(Introduce3dMovements):
    field_config = dict(
        max_vect_len=0.25,
        stroke_opacity=0.7,
        radius_of_suppression=1.0,
        width=20,
        height=20,
        depth=8,
        x_density=4.0,
        y_density=4.0,
        z_density=1.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(n, 0, 0.6)
    )


class CoulombLorentzExample(Introduce3dMovements):
    field_class = ColoumbPlusLorentzField
    field_config = dict(
        max_vect_len=0.3,
        stroke_opacity=0.7,
        radius_of_suppression=1.0,
        width=40,
        height=40,
        depth=0,
        x_density=4.0,
        y_density=4.0,
        z_density=1.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(n, 0, 0.8)
    )


class CoulombLorentzExample3D(Introduce3dMovements3DVects):
    field_class = ColoumbPlusLorentzField
    field_config = dict(
        max_vect_len=0.35,
        stroke_opacity=0.7,
        radius_of_suppression=1.0,
        width=10,
        height=10,
        depth=8,
        x_density=2.0,
        y_density=2.0,
        z_density=2.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(n, 0, 0.75)
    )


class RowOfCharges(ChargeOnZAxis):
    n_charges = 17
    particle_buff = 0.25
    particle_config = dict(
        rotation=PI / 2,
        track_position_history=True,
        radius=0.1,
        show_sign=False,
        charge=0.15
    )
    field_config = dict(
        max_vect_len=0.5,
        stroke_opacity=0.7,
        radius_of_suppression=1.0,
        width=30,
        height=30,
        depth=0,
        x_density=4.0,
        y_density=4.0,
        z_density=1.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(1.5 * n, 0, 0.8)
    )
    show_acceleration_vector = False

    def construct(self):
        # Test
        self.play(self.frame.animate.reorient(-7, 62, 0).set_height(16), run_time=12)
        self.play(self.frame.animate.reorient(26, 70, 0), run_time=12)
        self.play(self.frame.animate.reorient(-26, 70, 0), run_time=12)

    def get_particles(self):
        return Group(*(
            ChargedParticle(**self.particle_config)
            for n in range(self.n_charges)
        )).arrange(UP, buff=self.particle_buff)


class PlaneOfCharges(RowOfCharges):
    n_rows = 20
    n_cols = 20
    particle_buff = 0.1
    grid_height = 6

    particle_config = dict(
        rotation=PI / 2,
        track_position_history=True,
        radius=0.05,
        show_sign=False,
        charge=2.0 / 400.0,
    )

    def get_particles(self):
        result = Group(*(
            ChargedParticle(**self.particle_config)
            for _ in range(self.n_rows * self.n_cols)
        ))
        result.arrange_in_grid(
            self.n_rows, self.n_cols,
            buff=self.particle_buff
        )
        result.set_width(self.grid_height)
        result.rotate(PI / 2, UP)
        return result


class RowOfChargesMoreCharges(RowOfCharges):
    n_charges = 100
    particle_buff = 0.01

    particle_config = dict(
        rotation=PI / 2,
        track_position_history=True,
        radius=0.05,
        show_sign=False,
        charge=0.0,
    )
    field_config = dict(
        max_vect_len=0.5,
        stroke_opacity=0.7,
        radius_of_suppression=1.0,
        width=30,
        height=30,
        depth=0,
        x_density=4.0,
        y_density=4.0,
        z_density=1.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(1.5 * n, 0, 0.8)
    )


class AltRowOfCharges(RowOfCharges):
    def construct(self):
        # Test
        self.play(self.frame.animate.reorient(-4, 82, 0).move_to([3.12, -0.06, 1.0]).set_height(5.23), run_time=12)
        self.play(self.frame.animate.reorient(-20, 69, 0).set_height(8.00), run_time=12)
        self.play(self.frame.animate.reorient(-13, 78, 0).move_to([4.32, -0.91, 0.42]).set_height(5.27), run_time=12)


class RowOfChargesXAxis(RowOfCharges):
    field_config = dict(
        max_vect_len=1.0,
        stroke_opacity=0.7,
        radius_of_suppression=0.25,
        width=40,
        height=0,
        depth=0,
        x_density=8.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(1.5 * n, 0, 0.8)
    )
    axes_config = dict(
        axis_config=dict(stroke_opacity=0.7),
        x_range=(-20, 20),
        y_range=(-6, 6),
        z_range=(-3, 3),
    )

    def setup(self):
        super().setup()
        self.frame.reorient(-26, 70, 0).set_height(16)
        self.axis_labels[0].set_x(8)

    def construct(self):
        # Form the field
        self.wait(20)

        # Zoom in
        self.play(
            self.frame.animate.reorient(-15, 84, 0).move_to([4.36, -1.83, 0.37]).set_height(5.59),
            run_time=3
        )

        # Show graph
        axes_kw = dict(self.axes_config)
        axes_kw.pop("z_range")
        axes = Axes(**axes_kw)
        graph1 = axes.get_graph(lambda r: 2.0 / r, x_range=(0.01, 20, 0.1))
        graph2 = axes.get_graph(lambda r: 1.0 / r**0.3, x_range=(0.01, 20, 0.1))
        graphs = VGroup(graph1, graph2)
        graphs.rotate(PI / 2, RIGHT, about_point=axes.get_origin())
        graphs.set_flat_stroke(False)
        graphs.set_stroke(TEAL, 2)

        words = VGroup(
            TexText(R"Instead of decaying like $\frac{1}{r}$"),
            TexText(R"It decays much more gently"),
        )
        words.fix_in_frame()
        words.to_edge(UP, buff=1.5)

        self.play(
            ShowCreation(graph1, run_time=2),
            FadeIn(words[0], 0.5 * UP)
        )
        self.wait()
        self.play(
            FadeOut(words[0], 0.5 * UP),
            FadeIn(words[1], 0.5 * UP),
            Transform(*graphs)
        )
        self.wait(6)


class RowOfChargesXAxisMoreCharges(RowOfChargesXAxis):
    n_charges = 100
    particle_buff = 0.1
    particle_config = dict(
        rotation=PI / 2,
        track_position_history=True,
        radius=0.05,
        show_sign=False,
        charge=3.0 / 50,
    )

    def construct(self):
        # Test
        self.wait(12)  # Let the field form


class RowOfChargesWiggleOnY(RowOfCharges):
    direction = UP


class WavesIn3D(ChargeOnZAxis):
    field_config = dict(
        max_vect_len=0.5,
        stroke_opacity=0.25,
        radius_of_suppression=1.0,
        height=10,
        depth=10,
        x_density=4.0,
        y_density=4.0,
        z_density=1.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(0.5 * n, 0, 0.8)
    )


class WiggleHereWiggleThere(IntroduceEField):
    def construct(self):
        # Setup
        charges = Group(*(
            ChargedParticle(track_position_history=True)
            for _ in range(2)
        ))
        charges[0].to_edge(LEFT, buff=2.0)
        charges[1].to_edge(RIGHT, buff=2.0)
        dist = get_norm(charges[0].get_center() - charges[1].get_center())
        for charge in charges:
            charge.ignore_last_motion()

        field = LorentzField(
            *charges,
            x_density=6.0,
            y_density=6.0,
            norm_to_opacity_func=lambda n: np.clip(0.5 * n, 0, 0.75)
        )
        self.add(field)
        self.add(*charges)

        # Wiggles
        wiggle_kwargs = dict(
            rate_func=lambda t: wiggle(t, 3),
            run_time=1.5,
            suspend_mobject_updating=False,
        )

        def wiggle_charge(charge, vect):
            ring = self.get_influence_ring(charge.get_center())
            ring.set_stroke(opacity=2 * get_norm(vect))
            self.add(ring)
            self.play(charge.animate.shift(vect).set_anim_args(**wiggle_kwargs))
            self.wait_until(lambda: ring.get_radius() > dist, max_time=dist / 2.0)

        for n, charge in zip(range(6), it.cycle((charges))):
            wiggle_charge(charge, UP * (-1)**n / 2**n)


class ScatteringOfPolarizedBeam(InteractiveScene):
    def construct(self):
        pass


class CircularPolarization1D(ShowTheEffectsOfOscillatingCharge):
    default_frame_orientation = (-20, 70)
    amplitude = 0.2
    field_config = dict(
        max_vect_len=1.0,
        stroke_opacity=0.85,
        radius_of_suppression=0.4,
        height=0,
        width=30,
        x_density=5.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(2 * n, 0, 0.8)
    )
    particle_config = dict(
        track_position_history=True,
        radius=0.15,
        show_sign=False,
    )

    def construct(self):
        # Setup
        frame = Square()
        frame.set_stroke(WHITE, 2)
        frame.set_fill(WHITE, 0.25)
        frame.rotate(PI / 2, UP)
        frame.move_to(self.axes.c2p(3, 0, 0))
        frame.set_flat_stroke(False)

        field = self.field
        field_opacity_tracker = ValueTracker(1)
        field.add_updater(lambda m: m.set_stroke(opacity=field_opacity_tracker.get_value()))

        lone_vect_config = dict(self.field_config)
        lone_vect_config["width"] = 0
        lone_vect_config["stroke_width"] = 3
        lone_vect = LorentzField(self.particles[0], **lone_vect_config)
        lone_vect.sample_points = np.array([frame.get_center()])

        # Pan
        self.play(
            self.frame.animate.reorient(81, 86, 0),
            run_time=12,
        )
        self.add(lone_vect)
        self.play(
            FadeIn(frame),
            field_opacity_tracker.animate.set_value(0.25),
            self.frame.animate.reorient(41, 76, 0),
            run_time=5,
        )
        self.play(
            self.frame.animate.reorient(79, 70, 0),
            run_time=12,
        )
        self.play(
            field_opacity_tracker.animate.set_value(0.15),
            self.frame.animate.reorient(97, 84, 0),
            run_time=12
        )
        self.play(
            field_opacity_tracker.animate.set_value(0.35),
            self.frame.animate.reorient(59, 73, 0),
            run_time=12,
        )

    def oscillation_function(self, time):
        angle = TAU * self.frequency * time
        return self.amplitude * np.array([-0, np.sin(angle), np.cos(angle)])


class PIPHelper(InteractiveScene):
    def construct(self):
        frame = Square(side_length=3)
        frame.set_stroke(WHITE, 4)
        frame.set_fill(WHITE, 0.25)
        frame.to_corner(UR)

        vect = Vector(RIGHT, stroke_color=BLUE)
        vect.shift(frame.get_center())
        self.add(frame)
        self.play(
            Rotating(vect, -15 * TAU, about_point=frame.get_center()),
            run_time=30,
            rate_func=linear,
        )


class CircularPolarization2D(CircularPolarization1D):
    field_config = dict(
        max_vect_len=0.35,
        stroke_opacity=0.85,
        radius_of_suppression=0.5,
        height=30,
        width=30,
        x_density=4.0,
        y_density=4.0,
        c=2.0,
        norm_to_opacity_func=lambda n: np.clip(2 * n, 0, 0.8)
    )

    def construct(self):
        # Test
        self.play(
            self.frame.animate.reorient(79, 78, 0),
            run_time=12,
        )
        self.play(
            self.frame.animate.reorient(-83, 80, 0),
            run_time=24,
        )
        self.play(
            self.frame.animate.reorient(17, 62, 0),
            run_time=18
        )


class RandomRicochet(InteractiveScene):
    default_frame_orientation = (-36, 70)

    def construct(self):
        # Setup
        frame = self.frame
        plane, axes = self.get_plane_and_axes()
        self.add(plane, axes)

        # Light and Ball
        ball = TrueDot(radius=0.1)
        ball.make_3d()
        ball.set_color(RED)

        light = GlowDot(radius=2)
        light.move_to(5 * DOWN)
        self.add(light, ball)

        # Beams
        n_beams = 15
        beams = VGroup()
        for _ in range(n_beams):
            point = 5 * normalize(np.random.uniform(-5, 5, 3))
            beam = VMobject().set_points_as_corners([
                light.get_center(), ball.get_center(), point
            ])
            beam.set_stroke(YELLOW, 5)
            beam.set_flat_stroke(False)
            beam.insert_n_curves(100)
            beams.add(beam)

        frame.reorient(-36, 70, 0) 
        frame.clear_updaters()
        frame.add_updater(lambda m, dt: m.increment_theta(dt * 3 * DEGREES))
        for beam in beams:
            self.play(
                VShowPassingFlash(
                    beam,
                    time_width=0.3,
                    run_time=1.5
                )
            )

    def get_plane_and_axes(self):
        axes = ThreeDAxes(axis_config=dict(tick_size=0))
        axes.set_stroke(opacity=0.5)
        plane = NumberPlane(
            x_range=axes.x_range,
            y_range=axes.y_range,
            background_line_style=dict(
                stroke_color=GREY_B,
                stroke_width=1.0,
                stroke_opacity=0.5,
            )
        )
        plane.axes.set_opacity(0)
        return plane, axes


class PolarizedScattering(RandomRicochet):
    field_config = dict(
        width=20,
        height=20,
        x_density=5.0,
        y_density=5.0,
        stroke_color=BLUE,
        norm_to_opacity_func=lambda n: np.clip(n, 0, 0.75)
    )

    def construct(self):
        # Setup
        frame = self.frame
        plane, axes = self.get_plane_and_axes()
        self.add(plane, axes)

        # Wave
        wave = OscillatingWave(
            axes,
            wave_len=2.0,
            color=YELLOW
        )
        vects = OscillatingFieldWave(axes, wave)
        wave.set_stroke(opacity=0)
        vects_opacity_tracker = ValueTracker(0.75)
        vects.add_updater(lambda m: m.set_stroke(opacity=vects_opacity_tracker.get_value()))

        self.add(wave, vects)

        # Charge
        charge = ChargedParticle(show_sign=False, radius=0.1, track_position_history=True)
        charge.add_updater(lambda m: m.move_to(
            -0.5 * wave.xt_to_point(0, self.time)
        ))
        a_vect = AccelerationVector(
            charge,
            norm_func=lambda n: 0.5 * np.tanh(n)
        )

        self.add(a_vect, charge)
        self.wait(6)

        # Result field
        charge.charge = 1.0
        field = LorentzField(charge, **self.field_config)
        field_opacity_multiple = ValueTracker(1)

        def update_field(f):
            mult = field_opacity_multiple.get_value()
            f.set_stroke(opacity=mult * f.get_stroke_opacities())
            return f

        field.add_updater(update_field)

        charge.ignore_last_motion()
        self.add(field, vects)
        self.play(
            self.frame.animate.reorient(15, 70, 0),
            run_time=12
        )

        # Show propagation
        rings = ProbagatingRings(axes.z_axis, n_rings=5, spacing=0.4)
        direction_vectors = VGroup(*(
            Arrow(v, 3 * v) for v in compass_directions(8)
        ))

        self.add(rings)
        self.play(
            LaggedStartMap(GrowArrow, direction_vectors, lag_ratio=0.1),
            field_opacity_multiple.animate.set_value(0.5),
            vects_opacity_tracker.animate.set_value(0.25),
        )
        self.play(
            self.frame.animate.reorient(-32, 73, 0),
            run_time=8,
        )
        self.play(VFadeOut(rings))

        # More vertical direction
        direction_vectors.generate_target()
        for dv in direction_vectors.target:
            dv.scale(0.5)
            dv.shift(-0.5 * dv.get_start())
            dv.rotate(
                70 * DEGREES, axis=cross(dv.get_vector(), OUT),
                about_point=ORIGIN
            )

        self.play(MoveToTarget(direction_vectors, run_time=2))
        self.play(FadeOut(direction_vectors))
        self.wait(6)


class PolarizedScatteringYZ(PolarizedScattering):
    field_config = dict(
        width=0,
        height=20,
        depth=20,
        x_density=5.0,
        y_density=5.0,
        stroke_color=BLUE,
        norm_to_opacity_func=lambda n: np.clip(n, 0, 1.0)
    )


class OneOfManyCharges(InteractiveScene):
    speed = 1.5
    wave_len = 3
    default_frame_orientation=(0, 70, 0)
    dot_amplitude_factor = 0.2
    plane_wave_field_config = dict(
        x_density=4.0,
        y_density=0.5,
        z_density=1.0,
        width=28,
        height=0,
        depth=12,
        tip_width_ratio=3,
        stroke_color=TEAL,
        stroke_width=3,
        stroke_opacity=0.5,
        norm_to_opacity_func=(lambda n: 2 * n),
        max_vect_len=1.0
    )
    plane_wave_amlpitude = 0.5
    charge_field_config = dict(
        x_density=4.0,
        y_density=4.0,
        z_density=4.0,
        width=28,
        height=0,
        depth=16,
        tip_width_ratio=3,
        stroke_color=BLUE,
        stroke_width=3,
        norm_to_opacity_func=(lambda n: np.tanh(n)),
    )
    charge_index = 3155

    dots_dims = (11, 20, 20)
    dots_shape = (6, 4, 4)

    def setup(self):
        super().setup()

        # Axes
        self.axes = ThreeDAxes()
        self.add(self.axes)

        # Add incoming wave
        omega = TAU * self.speed / self.wave_len
        k = TAU / self.wave_len
        amplitude = 0.5

        def plane_wave_func(points, t):
            return self.plane_wave_amlpitude * np.outer(
                np.cos(k * np.dot(points, RIGHT) - omega * t),
                OUT
            )

        plane_wave = TimeVaryingVectorField(
            plane_wave_func,
            **self.plane_wave_field_config
        )

        plane_wave.opacity_multiplier = ValueTracker(1)
        plane_wave.add_updater(lambda m: m.set_stroke(opacity=m.opacity_multiplier.get_value() * m.get_stroke_opacities()))

        self.plane_wave_func = plane_wave_func
        self.plane_wave = plane_wave
        self.add(plane_wave)

        # Add 3d grid of charges
        dots = DotCloud(color=BLUE)
        dots.to_grid(*self.dots_dims, height=None)
        dots.set_radius(0)
        dots.set_shape(*self.dots_shape)
        dots.set_radius(0.05)
        dots.make_3d(0.1)

        self.dot_center_refs = dots.copy()
        self.dot_center_refs.set_opacity(0)
        self.dot_center_refs.set_radius(0)

        def update_dots(dots):
            centers = self.dot_center_refs.get_points()
            offsets =  plane_wave.func(centers)
            offsets *= self.dot_amplitude_factor
            dots.set_points(centers + offsets)

        dots.add_updater(update_dots)

        dots.opacity_multiplier = ValueTracker(0.75)
        dots.add_updater(lambda m: m.set_opacity(m.opacity_multiplier.get_value()))

        self.dots = dots
        self.add(dots)


    def construct(self):
        # Pan a little to start
        frame = self.frame
        self.play(
            frame.animate.reorient(25, 72, 0),
            run_time=4
        )

        # Create field based on the charges
        if self.charge_index > 0:
            charge = ChargedParticle(
                charge=-5,
                color=BLUE,
                show_sign=False,
                radius=0.1
            )
            charge.add_updater(lambda m: m.move_to(self.dots.get_points()[self.charge_index]))
            charge.update()
            charge.ignore_last_motion()
            charge_field = LorentzField(
                charge,
                center=charge.get_y() * UP,
                radius_of_suppression=0.2,
                **self.charge_field_config
            )
            # acc_vect = AccelerationVector(charge)
            acc_vect = VectorizedPoint()

            self.add(charge_field, acc_vect, charge, self.dots)
            self.play(
                self.plane_wave.opacity_multiplier.animate.set_value(0),
                self.dots.opacity_multiplier.animate.set_value(0.1),
                FadeIn(charge, suspend_mobject_updating=False),
                VFadeIn(acc_vect),
                run_time=2,
            )
        else:
            self.play(
                self.dots.opacity_multiplier.animate.set_value(0.2),
                run_time=2
            )
        self.play(
            frame.animate.reorient(25, 72, 0),
            run_time=8
        )
        self.play(
            frame.animate.reorient(-25, 72, 0),
            run_time=8
        )


class AlternateCompositeChargesInPlane(OneOfManyCharges):
    dots_dims = (11, 10, 10)
    dots_shape = (6, 4, 4)
    random_seed = 2

    def construct(self):
        # Objects
        frame = self.frame
        plane_wave = self.plane_wave
        dots = self.dots
        dots.set_radius(0.075)
        dots.opacity_multiplier.set_value(1)

        # Some panning
        frame.reorient(-17, 76, 0)
        self.play(
            frame.animate.reorient(16, 77, 0),
            run_time=12
        )

        # Reorient
        self.play(
            dots.animate.set_height(0, stretch=True),
            self.dot_center_refs.animate.set_height(0, stretch=True),
            frame.animate.reorient(0, 90).set_height(6),
            self.axes.y_axis.animate.set_stroke(opacity=0),
            plane_wave.opacity_multiplier.animate.set_value(0.5),
            run_time=2,
        )
        self.wait(2)
        self.play(
            dots.opacity_multiplier.animate.set_value(0.25),
            plane_wave.opacity_multiplier.animate.set_value(0.0),
        )
        self.wait()

        # Add charges
        w, h, d = self.dots_dims
        indices = np.arange(0, w * h * d, w)
        charge = ChargedParticle(
            color=BLUE,
            radius=dots.get_radius(),
            show_sign=False,
        )
        charges = charge.replicate(len(indices))
        charges.apply_depth_test(False)
        for charge, index in zip(charges, indices):
            charge.index = index
            charge.add_updater(lambda m: m.move_to(dots.get_points()[m.index] + 0.01 * DOWN))
        
        charges.shuffle()

        charges[0].update()
        charges[0].ignore_last_motion()
        charge_field = LorentzField(
            charges[0],
            radius_of_suppression=0.2,
            **self.charge_field_config
        )
        self.add(charges[0])
        self.add(charge_field)
        self.wait(3)

        start_time = float(self.time)
        for n, charge in enumerate(charges[1:]):
            charge.update()
            charge.ignore_last_motion()

            if n < 2:
                time = 3
            if n < 5:
                time = 2
            elif n < 15:
                time = 1
            else:
                time = 0.2

            charge_field.charges.append(charge)
            n_charges = len(charge_field.charges)
            alpha = inverse_interpolate(start_time, start_time + 30, self.time)
            q_per_particle = interpolate(1, 0.5, alpha)
            for c2 in charge_field.charges:
                c2.charge = q_per_particle

            self.add(charge)
            self.wait(time)

        self.wait(30)


class FullCompositeEffect(OneOfManyCharges):
    default_frame_orientation = (-25, 72, 0)
    dot_amplitude_factor = 0.1
    index = 1.75

    def construct(self):
        frame = self.frame
        dots = self.dots
        plane_wave = self.plane_wave

        min_x = dots.get_x(LEFT)
        max_x = dots.get_x(RIGHT)

        def new_pw_func(points, time):
            adj_points = points.copy()
            to_left = np.clip(adj_points[:, 0] - min_x, 0, np.inf)
            to_right = np.clip(adj_points[:, 0] - max_x, 0, np.inf)
            adj_points[:, 0] += (self.index - 1) * to_left - (self.index - 1) * to_right
            return self.plane_wave_func(adj_points, time)

        new_wave = TimeVaryingVectorField(
            new_pw_func,
            stroke_color=YELLOW,
            x_density=10,
            z_density=1,
            width=14,
            height=0,
            depth=0,
            max_vect_len=np.inf,
            norm_to_opacity_func=lambda n: 2 * n,
        )

        new_wave.opacity_multiplier = ValueTracker(0)
        new_wave.add_updater(lambda m: m.set_stroke(opacity=m.get_stroke_opacities() * m.opacity_multiplier.get_value()))

        # Test
        self.add(new_wave)
        self.play(
            frame.animate.reorient(0, 90, 0).set_height(8).set_focal_distance(100),
            dots.opacity_multiplier.animate.set_value(0.75),
            plane_wave.opacity_multiplier.animate.set_value(0),
            new_wave.opacity_multiplier.animate.set_value(1),
            VFadeIn(new_wave),
            run_time=5
        )
        self.wait(15)


class FullCompositeEffectIndexLessThanOne(FullCompositeEffect):
    index = 0.7


class ManyParallelPropagations(OneOfManyCharges):
    dots_dims = (11, 20, 20)
    dots_shape = (6, 0, 4)

    def construct(self):
        # Test
        dots = self.dots
        plane_wave = self.plane_wave
        self.axes.y_axis.set_opacity(0)

        frame = self.frame
        frame.reorient(0, 90, 0)
        frame.set_focal_distance(10)

        dots.set_radius(0.05)
        dots.opacity_multiplier.set_value(1)
        plane_wave.opacity_multiplier.set_value(0)

        rings = []
        for _ in range(10):
            min_x, min_y, min_z = self.dot_centers[0]
            max_x, max_y, max_z = self.dot_centers[-1]
            for x in np.linspace(min_x, max_x, 6):
                for z in np.linspace(min_z, max_z, 10):
                    ring = get_influence_ring([x, 0, z], speed=self.speed, max_width=1, width_decay_exp=2)
                    ring.rotate(PI / 2, RIGHT)
                    rings.append(ring)
                    self.add(ring)
            self.wait()
            for ring in rings:
                if ring.get_stroke_width() < 0.01:
                    self.remove(ring)
        self.wait(5)


class ResponsiveCharge(InteractiveScene):
    def construct(self):
        # Driving chrage
        charge1 = ChargedParticle(charge=0.25)
        charge1.add_spring_force(k=10)
        charge1.move_to(0.3 * DOWN)

        # Responsive charge
        k = 20
        charge2 = ChargedParticle(charge=1.0, radius=0.1, show_sign=False)
        charge2.move_to(2.5 * RIGHT)
        # charge2.add_field_force(field)
        charge2.add_spring_force(k=k)
        charge2.add_force(lambda p: wave.xt_to_point(p[0], wave.time) * [0, 1, 1])
        # charge2.fix_x()
        self.add(charge2)

        # E field
        # field_type = ColoumbPlusLorentzField
        field_type = LorentzField
        field = field_type(
            charge1, charge2,
            x_density=4.0,
            y_density=4.0,
            norm_to_opacity_func=lambda n: np.clip(0.5 * n, 0, 1),
            c=1.0,
        )
        self.add(field)

        # Pure wave
        axes = ThreeDAxes()
        wave = OscillatingWave(axes, y_amplitude=1.0, z_amplitude=0.0, wave_len=2.0)
        field_wave = OscillatingFieldWave(axes, wave)
        wave.set_stroke(opacity=0.5)


        self.add(axes, wave, field_wave)

        # omega = (wave.speed / wave.wave_len) * TAU
        # omega_0 = math.sqrt(k / charge2.mass)
        # v0 = omega / (omega_0**2 - omega**2)
        # charge2.velocity = v0 * UP

        self.wait(20)

        # Plane
        plane = NumberPlane()
        plane.fade(0.5)
        self.add(plane)

        # Test wiggle
        self.play(
            charge1.animate.shift(UP).set_anim_args(
                rate_func=wiggle,
                run_time=3,
            )
        )
        self.wait(4)
