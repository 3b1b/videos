from manim_imports_ext import *


class EllipticFunctions(InteractiveScene):
    def construct(self):
        # Create two planes
        x_max = 15
        in_plane, out_plane = planes = VGroup(
            ComplexPlane((-x_max, x_max), (-x_max, x_max)),
            ComplexPlane((-x_max, x_max), (-x_max, x_max)),
        )
        in_plane.set_width(5.5)
        out_plane.set_width(5.5)
        in_plane.add_coordinate_labels(
            list(range(-x_max, x_max + 1, 5)) + [x * 1j for x in list(range(-x_max, x_max + 1, 5)) if x != 0],
            font_size=10
        )
        out_plane.add_coordinate_labels(
            list(range(-x_max, x_max + 1, 5)) + [x * 1j for x in list(range(-x_max, x_max + 1, 5)) if x != 0],
            font_size=10
        )
        in_plane.coordinate_labels.shift(UR * 0.06)
        out_plane.coordinate_labels.shift(UR * 0.06)
        in_plane.set_stroke(GREY_D, 1)
        out_plane.set_stroke(GREY_D, 1)
        VGroup(in_plane, out_plane).arrange(buff=0.75).to_edge(DOWN, buff=0.5)
        self.add(in_plane, out_plane)
        self.add(Dot(radius=0.04).set_color(WHITE).set_stroke(width=4, color=BLACK).move_to(in_plane.n2p(0)))

        z = Tex("z", font_size=70).set_color(YELLOW).next_to(in_plane, UP, buff=0.2).set_x(in_plane.c2p(0)[0])
        f_of_z = Tex(
            r"""
                \wp(z)=\frac{1}{z^2}+\sum_{\lambda \in \left\{k_1 \omega_1+k_2 \omega_2\right\}}
                \left(\frac{1}{(z-\lambda)^2}-\frac{1}{\lambda^2}\right)
            """,
            font_size=28,
            tex_to_color_map={r"\wp": PINK, "z": YELLOW, r"\lambda": BLUE, r"\omega_1": GREEN, r"\omega_2": RED}
        ).next_to(
            out_plane, UP, buff=0.2
        ).set_x(
            out_plane.c2p(0)[0]
        )
        self.add(z, f_of_z)

        # Add lattice points
        omega1 = 1 + 1j
        omega2 = 2 - 3j
        omega1_dot = GlowDot().set_color(GREEN).move_to(in_plane.n2p(omega1))
        omega2_dot = omega1_dot.copy().set_color(RED).move_to(in_plane.n2p(omega2))
        omega1_label = Tex(
            r"\omega_1", font_size=30
        ).set_color(
            GREEN
        ).set_stroke(
            width=7, color=BLACK, behind=True
        ).next_to(
            omega1_dot, UR, buff=-0.1
        )
        omega2_label = Tex(
            r"\omega_2", font_size=30
        ).set_color(
            RED
        ).set_stroke(
            width=7, color=BLACK, behind=True
        ).next_to(
            omega2_dot, UR, buff=-0.1
        )

        lattice = VGroup()
        lambdas = []
        for k1 in range(-20, 20):
            for k2 in range(-10, 10):
                lamb = k1 * omega1 + k2 * omega2
                if lamb != 0 and lamb.real >= -x_max and lamb.real <= x_max and lamb.imag >= -x_max and lamb.imag <= x_max:
                    lambdas.append(lamb)
                    dot = Dot(radius=0.04).set_color(BLUE).set_opacity(0.9).move_to(in_plane.n2p(lamb))
                    lattice.add(dot)
        self.add(lattice, omega1_dot, omega1_label, omega2_dot, omega2_label)

        # Add z dot
        offset = -0.3 * omega1 + 0.9 * omega2
        z_init = offset
        z_dot = GlowDot().set_color(YELLOW).move_to(in_plane.n2p(z_init))
        z_label = Tex("z", font_size=35).set_color(YELLOW).set_stroke(width=7, color=BLACK, behind=True)
        z_label.add_updater(lambda m: m.next_to(z_dot, UL, buff=-0.1))
        # z_tail = TracingTail(z_dot, stroke_color = YELLOW, time_traced = 6)
        self.add(z_dot, z_label)

        # Add p(z) dot
        def p(z):
            return 1 / z**2 + sum([1 / (z - lamb)**2 - 1 / lamb**2 for lamb in lambdas])

        p_z_dot = GlowDot().set_color(PINK)
        p_z_dot.add_updater(lambda m: m.move_to(out_plane.n2p(p(in_plane.p2n(z_dot.get_center())))))
        p_z_label = Tex(
            r"\wp(z)", font_size=35, tex_to_color_map={r"\wp": PINK, "z": YELLOW}
        ).set_stroke(width=7, color=BLACK, behind=True)
        p_z_label.add_updater(lambda m: m.next_to(p_z_dot, UR, buff=-0.1))
        p_z_tail = TracingTail(p_z_dot, stroke_color=PINK, time_traced=6)
        self.add(p_z_dot, p_z_label, p_z_tail)

        # Move z around
        # choices = [omega1, omega2, -omega1, -omega2, omega1 + omega2, omega1 - omega2, -omega1 + omega2, -omega1 -omega2]
        choices = [omega1, omega2, -omega1, -omega2]
        for _ in range(10):
            next_point = in_plane.p2n(z_dot.get_center()) + random.choice(choices)
            self.play(z_dot.animate.move_to(in_plane.n2p(next_point)), run_time=3)
            self.wait(0.6)


class MobiusTransformation1(InteractiveScene):
    def construct(self):
        # Create planes
        x_max = 3
        a_plane, b_plane, c_plane, d_plane, out_plane = planes = VGroup(
            ComplexPlane((-x_max, x_max), (-x_max, x_max)),
            ComplexPlane((-x_max, x_max), (-x_max, x_max)),
            ComplexPlane((-x_max, x_max), (-x_max, x_max)),
            ComplexPlane((-x_max, x_max), (-x_max, x_max)),
            ComplexPlane((-1, 1), (-1, 1)),
        )
        for plane in [a_plane, b_plane, c_plane, d_plane]:
            plane.set_width(2.5)
            plane.add_coordinate_labels(
                list(range(-x_max, x_max + 1)) + [x * 1j for x in list(range(-x_max, x_max + 1)) if x != 0],
                font_size=10
            )
            plane.coordinate_labels.shift(UR * 0.04)
            plane.set_stroke(GREY_D, 1)
        out_plane.set_width(5.5)
        out_plane.add_coordinate_labels([-1, 0, 1, -1j, 1j])
        out_plane.set_stroke(GREY_D, 1)
        VGroup(
            VGroup(a_plane, b_plane, c_plane, d_plane).arrange_in_grid(buff=0.2),
            out_plane
        ).arrange(buff=0.75).to_edge(DOWN, buff=0.5)
        VGroup(a_plane, b_plane, c_plane, d_plane).scale(1.02).shift(out_plane[0].get_bottom()[1] - c_plane[0].get_bottom()[1])
        self.add(planes)
        var_colors = [RED, GREEN, BLUE, ORANGE]
        f_of_z = Tex(
            r"f(z) = \displaystyle\frac{az + b}{cz + d}",
            font_size=32,
            tex_to_color_map={"f": PINK, "z": YELLOW, "a": var_colors[0], "b": var_colors[1], "c": var_colors[2], "d": var_colors[3]}
        ).set_stroke(
            width=7, color=BLACK, behind=True
        ).next_to(
            out_plane, UP, buff=0.2
        ).set_x(
            out_plane.c2p(0)[0]
        )
        self.add(f_of_z)

        # Add dots for a, b, c, and d
        # random.uniform(-x_max, x_max) + random.uniform(-x_max, x_max)*1j for _ in range(4)
        var_trackers = [ValueTracker([1, 0]), ValueTracker([0, 0]), ValueTracker([0, 0]), ValueTracker([0, 1])]
        dots = Group(*[Group(TrueDot(), GlowDot()).set_color(var_colors[i]) for i in range(4)])
        for i in range(len(dots)):
            dots[i].add_updater(lambda m, i=i: m.move_to(planes[i].n2p(complex(*var_trackers[i].get_value()))))
        labels = VGroup(*[
            Tex("abcd"[i], font_size=35).set_color(var_colors[i]).set_stroke(width=7, color=BLACK, behind=True)
            for i in range(4)
        ])
        for i in range(4):
            labels[i].add_updater(lambda m, i=i: m.next_to(dots[i], UR, buff=-0.1))
        self.add(dots, labels)

        # Add result_plane
        grid_size = 20
        squares = Square().get_grid(grid_size, grid_size, buff=0)
        squares.match_width(out_plane[0]).move_to(out_plane[0])
        squares.set_stroke(PINK, 1, 0.5)
        squares.insert_n_curves(10)
        squares.save_state()

        def update_squares(m):
            m.restore()
            m.apply_function(
                lambda z: out_plane.n2p(
                    (
                        complex(*var_trackers[0].get_value()) * out_plane.p2n(z) + complex(*var_trackers[1].get_value())
                    ) / (
                        complex(*var_trackers[2].get_value()) * out_plane.p2n(z) + complex(*var_trackers[3].get_value())
                    )
                )
            )
        squares.add_updater(update_squares)
        self.add(squares, f_of_z, planes[:4], dots, labels)

        # Move around a, b, c, and d
        for _ in range(6):
            new_variables_values = [[random.uniform(-x_max, x_max) for _ in range(2)] for _ in range(4)]
            self.play(
                AnimationGroup(*[
                    var_trackers[i].animate.set_value(new_variables_values[i])
                    for i in range(4)
                ]), run_time=3)
            self.wait(1)


class MobiusTransformation2BiggerGridTest(InteractiveScene):
    def construct(self):
        # Create a unit plane
        unit_plane = ComplexPlane((-1, 1), (-1, 1))
        unit_plane.set_height(FRAME_HEIGHT * 0.85)
        unit_plane.add_coordinate_labels([-1, 0, 1, -1j, 1j])
        unit_plane.set_stroke(GREY_D, 1)
        self.add(unit_plane)

        # Set up updaters and dots
        angle_trackers = [
            ValueTracker(a) for a in sorted([random.uniform(0, TAU) for _ in range(3)])
        ]

        def get_mobius_func():
            z_ang, o_ang, i_ang = [t.get_value() for t in angle_trackers]

            w0 = np.exp(1j * z_ang)
            w1 = np.exp(1j * o_ang)
            winf = np.exp(1j * i_ang)

            a = winf
            c = 1
            d = (w1 - winf) / (w0 - w1)
            b = w0 * d

            return lambda z: (a * z + b) / (c * z + d)

        grid_size_x = 50
        grid_size_y = 50
        squares_template = Square().get_grid(grid_size_y, grid_size_x, buff=0)
        squares_template.match_width(unit_plane[0]).move_to(unit_plane[0]).align_to(unit_plane.n2p(0), DOWN)
        squares_template.scale(15, about_point=squares_template.get_bottom())
        squares_template.insert_n_curves(80)
        for square in squares_template:
            square.set_stroke(width=max(1, 3 / unit_plane.p2n(square.get_center()).imag**0.4))

        mobius_grid = always_redraw(lambda:
                                    squares_template.copy().set_submobject_colors_by_gradient(
                                        BLUE, YELLOW, interp_by_hsl=True
                                    ).apply_function(
                                        lambda p: unit_plane.n2p(get_mobius_func()(unit_plane.p2n(p)))
                                    )
                                    )
        self.add(mobius_grid)

        dots = always_redraw(lambda:
                             Group(
                                 Group(TrueDot(), GlowDot()).set_color(RED).move_to(unit_plane.n2p(get_mobius_func()(0))),
                                 Group(TrueDot(), GlowDot()).set_color(GREEN).move_to(unit_plane.n2p(get_mobius_func()(1))),
                                 Group(TrueDot(), GlowDot()).set_color(ORANGE).move_to(unit_plane.n2p(get_mobius_func()(1e8)))
                             )
                             )
        self.add(dots)

        # Move the dots around
        for i in range(6):
            min_gap = TAU / 8

            remaining_tau = TAU - (3 * min_gap)
            breaks = sorted([random.uniform(0, remaining_tau) for _ in range(2)])

            gaps = [
                breaks[0],
                breaks[1] - breaks[0],
                remaining_tau - breaks[1]
            ]

            new_angles = []
            current_angle = random.uniform(0, TAU)  # Random starting rotation
            for gap in gaps:
                current_angle += gap + min_gap
                new_angles.append(current_angle % TAU)

            new_angles.sort()

            animations = [
                angle_trackers[j].animate.set_value(new_angles[j])
                for j in range(3)
            ]

            self.play(
                *animations,
                run_time=3
            )
            self.wait(1)


class NaiveLinearScaling(InteractiveScene):
    def construct(self):
        # Add the canvas
        canvas = Square(
            side_length=6,
            fill_opacity=0,
            stroke_width=5,
            stroke_color=WHITE
        )
        self.add(canvas)
        corners = [DR, DL, UL, UR]

        # Draw a little square in the bottom left
        square1b = Square(
            side_length=0.25,
            fill_opacity=0.7,
            fill_color=BLUE,
            stroke_width=0
        ).align_to(canvas, DR)
        square1b.shift(-corners[0] * square1b.get_width() * 1.5)
        self.play(DrawBorderThenFill(square1b))
        self.wait(2)

        # Scale a copy of the square up around the 4 corners
        square2a = square1b.copy().set_color(GREEN)
        self.play(FadeIn(square2a))
        lines_per_side = 4
        corner = 1
        connectingLines1 = always_redraw(
            lambda corner=corner: VGroup(*[
                Line(
                    square1b.get_corner(corners[corner]) + (corners[(corner + 1) % 4] - corners[corner]) * 0.5 * square1b.get_width() * i / (lines_per_side - 1),
                    square2a.get_corner(corners[corner]) + (corners[(corner + 1) % 4] - corners[corner]) * 0.5 * square2a.get_width() * i / (lines_per_side - 1), stroke_width=2
                ).set_color(YELLOW)
                for i in range(lines_per_side)
            ])
        )
        self.add(connectingLines1)
        self.bring_to_back(connectingLines1)
        self.bring_to_back(canvas)
        self.play(
            square2a.animate.scale(4).align_to(canvas, corners[1])
        )
        square2b = square1b.copy().align_to(square2a, corners[1]).shift(-corners[1] * square1b.get_width() * 1.5)

        square3a = square2b.copy().set_color(GREEN)
        self.play(FadeIn(square2b), FadeIn(square3a))
        corner = 2
        connectingLines2 = always_redraw(
            lambda corner=corner: VGroup(*[
                Line(
                    square2b.get_corner(corners[corner]) + (corners[(corner + 1) % 4] - corners[corner]) * 0.5 * square2b.get_width() * i / (lines_per_side - 1),
                    square3a.get_corner(corners[corner]) + (corners[(corner + 1) % 4] - corners[corner]) * 0.5 * square3a.get_width() * i / (lines_per_side - 1), stroke_width=2
                ).set_color(YELLOW)
                for i in range(lines_per_side)
            ])
        )
        self.add(connectingLines2)
        self.bring_to_back(connectingLines2)
        self.bring_to_back(canvas)
        self.play(
            square3a.animate.scale(4).align_to(canvas, UL)
        )
        square3b = square1b.copy().align_to(square3a, UL).shift(-corners[2] * square1b.get_width() * 1.5)

        square4a = square3b.copy().set_color(GREEN)
        self.play(FadeIn(square3b), FadeIn(square4a))
        corner = 3
        connectingLines3 = always_redraw(
            lambda corner=corner: VGroup(*[
                Line(
                    square3b.get_corner(corners[corner]) + (corners[(corner + 1) % 4] - corners[corner]) * 0.5 * square3b.get_width() * i / (lines_per_side - 1),
                    square4a.get_corner(corners[corner]) + (corners[(corner + 1) % 4] - corners[corner]) * 0.5 * square4a.get_width() * i / (lines_per_side - 1), stroke_width=2
                ).set_color(YELLOW)
                for i in range(lines_per_side)
            ])
        )
        self.add(connectingLines3)
        self.bring_to_back(connectingLines3)
        self.bring_to_back(canvas)
        self.play(
            square4a.animate.scale(4).align_to(canvas, UR)
        )
        square4b = square1b.copy().align_to(square4a, UR).shift(-corners[3] * square1b.get_width() * 1.5)

        square1a = square4b.copy().set_color(GREEN)
        self.play(FadeIn(square4b), FadeIn(square1a))
        corner = 0
        connectingLines4 = always_redraw(
            lambda corner=corner: VGroup(*[
                Line(
                    square4b.get_corner(corners[corner]) + (corners[(corner + 1) % 4] - corners[corner]) * 0.5 * square4b.get_width() * i / (lines_per_side - 1),
                    square1a.get_corner(corners[corner]) + (corners[(corner + 1) % 4] - corners[corner]) * 0.5 * square1a.get_width() * i / (lines_per_side - 1), stroke_width=2
                ).set_color(YELLOW)
                for i in range(lines_per_side)
            ])
        )
        self.add(connectingLines4)
        self.bring_to_back(connectingLines4)
        self.bring_to_back(canvas)
        self.play(
            square1a.animate.scale(4).align_to(canvas, DR)
        )

        # Zoom in on one of the corners
        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.reorient(0, 0, 0, (np.float32(-2.06), np.float32(-2.52), np.float32(0.0)), 1.20),
            run_time=2.5
        )

        # Show the pressure from the right and the bottom
        moving_square_from_right = square2b.copy().scale(
            0.7
        ).set_stroke(
            color=RED, width=5
        ).set_fill(
            opacity=0
        ).align_to(
            self.camera.frame.get_right(), LEFT
        ).shift(RIGHT * 0.2)
        self.add(moving_square_from_right)
        moving_square_from_right.generate_target()
        trapezoid1 = Polygon(
            [1.9, 1, 0],
            [0, 1.15, 0],
            [0, -1.15, 0],
            [1.9, -1, 0],
            fill_opacity=0,
            stroke_width=5,
            stroke_color=RED,
        )
        trapezoid1.match_height(moving_square_from_right).scale(1.2).scale(1 / 0.7).move_to(square2b)
        moving_square_from_right.target.become(trapezoid1)
        self.bring_to_front(square2b)
        # effectExaggerated = TexText(r"$^*$Effect exaggerated for demonstration", font_size = 9).set_color(WHITE).align_to(self.camera.frame.get_corner(UR), UR).shift(DL*0.1 + DOWN*0.06)

        moving_square_from_bottom = square2b.copy().scale(
            0.7
        ).set_stroke(
            color=ORANGE, width=5
        ).set_fill(
            opacity=0
        ).align_to(
            self.camera.frame.get_bottom(), UP
        ).shift(DOWN * 0.2)
        self.add(moving_square_from_bottom)
        moving_square_from_bottom.generate_target()

        v1 = trapezoid1.get_vertices()
        v2 = [rotate_vector(v, -PI * 0.5) for v in v1]
        v2_cycled = v2[1:] + [v2[0]]

        trapezoid2 = Polygon(*v2_cycled)
        trapezoid2.set_color(ORANGE).set_stroke(width=5).move_to(square2b)

        self.add(VGroup(moving_square_from_bottom, moving_square_from_right), Point(), square2b)
        moving_square_from_bottom.target.become(trapezoid2)
        self.play(
            MoveToTarget(moving_square_from_bottom),
            MoveToTarget(moving_square_from_right),
            square2b.animate.set_opacity(0.6), run_time=5)
        self.wait(2)
        self.add(VGroup(moving_square_from_bottom, moving_square_from_right), Point(), square2b)
        self.play(FadeOut(moving_square_from_bottom), FadeOut(moving_square_from_right), square2b.animate.set_opacity(1))

        # Wiggle the square to try to fit the two different shapes
        square2b.save_state()
        connectingLines1.clear_updaters()
        connectingLines2.clear_updaters()
        connectingLines3.clear_updaters()
        connectingLines4.clear_updaters()
        # self.play(self.camera.frame.animate.scale(0.8, about_point = square2b.get_center()).shift(LEFT*0.25), run_time = 2)
        shift_amt = 0.01
        for i in range(3):
            self.add(connectingLines2, Point(), square2b)
            self.play(
                AnimationGroup(
                    connectingLines1[0].animate.put_start_and_end_on(
                        connectingLines1[0].get_start() + LEFT * 3.3, connectingLines1[0].get_end()
                    ),
                    connectingLines1[1].animate.put_start_and_end_on(
                        connectingLines1[1].get_start() + LEFT * 3.3, connectingLines1[1].get_end()),
                    connectingLines1[2].animate.put_start_and_end_on(
                        connectingLines1[2].get_start() + LEFT * 3.3, connectingLines1[2].get_end()),
                    connectingLines1[3].animate.put_start_and_end_on(
                        connectingLines1[3].get_start() + LEFT * 3.3, connectingLines1[3].get_end()),
                    square2b.animate.shift(0)
                ) if i == 0 else
                AnimationGroup(
                    connectingLines1[0].animate.put_start_and_end_on(
                        connectingLines1[0].get_start() + RIGHT * 3.3, connectingLines1[0].get_end()
                    ),
                    connectingLines1[1].animate.put_start_and_end_on(
                        connectingLines1[1].get_start() + RIGHT * 3.3, connectingLines1[1].get_end()
                    ),
                    connectingLines1[2].animate.put_start_and_end_on(
                        connectingLines1[2].get_start() + RIGHT * 3.3, connectingLines1[2].get_end()
                    ),
                    connectingLines1[3].animate.put_start_and_end_on(
                        connectingLines1[3].get_start() + RIGHT * 3.3, connectingLines1[3].get_end()
                    ),
                    connectingLines2[0].animate.put_start_and_end_on(
                        connectingLines2[0].get_start() + LEFT * 2 * shift_amt, connectingLines2[0].get_end()
                    ),
                    connectingLines2[1].animate.put_start_and_end_on(
                        connectingLines2[1].get_start() + LEFT * shift_amt, connectingLines2[1].get_end()
                    ),
                    connectingLines2[2].animate.put_start_and_end_on(
                        connectingLines2[2].get_start() + RIGHT * shift_amt, connectingLines2[2].get_end()
                    ),
                    connectingLines2[3].animate.put_start_and_end_on(
                        connectingLines2[3].get_start() + RIGHT * 2 * shift_amt, connectingLines2[3].get_end()
                    ),
                    square2b.animate.shift(0)
                ) if i % 2 == 1 else
                AnimationGroup(
                    connectingLines2[0].animate.put_start_and_end_on(
                        connectingLines2[0].get_start() + RIGHT * 2 * shift_amt, connectingLines2[0].get_end()
                    ),
                    connectingLines2[1].animate.put_start_and_end_on(
                        connectingLines2[1].get_start() + RIGHT * shift_amt, connectingLines2[1].get_end()
                    ),
                    connectingLines2[2].animate.put_start_and_end_on(
                        connectingLines2[2].get_start() + LEFT * shift_amt, connectingLines2[2].get_end()
                    ),
                    connectingLines2[3].animate.put_start_and_end_on(
                        connectingLines2[3].get_start() + LEFT * 2 * shift_amt, connectingLines2[3].get_end()
                    ),
                    square2b.animate.shift(0)
                ),
                square2b.animate.match_points(
                    trapezoid1.copy() if i % 2 == 0 else trapezoid2.copy()
                ) if i < 2 else square2b.animate.restore(),
                run_time=1.5
            )
            self.wait(1.5)

        # Turn the lines into curves
        self.bring_to_front(VGroup(square1b, square2b, square3b, square4b).set_opacity(1))
        self.play(self.camera.frame.animate.restore(), run_time=2)
        offset = 3
        self.play(
            *[
                line.animate.become(
                    CubicBezier(
                        line.get_start(),
                        line.get_start() + LEFT * offset,
                        line.get_end() + RIGHT * offset,
                        line.get_end()
                    ).set_color(YELLOW).set_stroke(width=2)
                )
                for line in connectingLines1
            ],
            *[
                line.animate.become(
                    CubicBezier(
                        line.get_start(),
                        line.get_start() + UP * offset,
                        line.get_end() + DOWN * offset,
                        line.get_end()
                    ).set_color(YELLOW).set_stroke(width=2)
                )
                for line in connectingLines2
            ],
            *[
                line.animate.become(
                    CubicBezier(
                        line.get_start(),
                        line.get_start() + RIGHT * offset,
                        line.get_end() + LEFT * offset,
                        line.get_end()
                    ).set_color(YELLOW).set_stroke(width=2)
                )
                for line in connectingLines3
            ],
            *[
                line.animate.become(
                    CubicBezier(
                        line.get_start(),
                        line.get_start() + DOWN * offset,
                        line.get_end() + UP * offset,
                        line.get_end()
                    ).set_color(YELLOW).set_stroke(width=2)
                )
                for line in connectingLines4
            ],
            run_time=1
        )
        self.wait(2)


class RightAngles(InteractiveScene):
    def construct(self):
        # Create a right angle symbol
        line1 = Line(ORIGIN, RIGHT)
        line2 = Line(RIGHT, UR).shift(DL * 0.04)
        VGroup(line1, line2).center().set_stroke(width=15, color=PURE_RED)
        self.play(AnimationGroup(ShowCreation(line1), ShowCreation(line2), lag_ratio=0.8))


class WriteSquares(InteractiveScene):
    def construct(self):
        squaresText = TexText(
            "Squares!",
            font_size=150
        ).set_color(WHITE).set_stroke(width=10, color=BLACK, behind=True)
        self.play(Write(squaresText, stroke_color=WHITE), run_time=2)
