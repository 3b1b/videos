from manim_imports_ext import *


class TalkFrame(InteractiveScene):
    def construct(self):
        # Test
        background = FullScreenRectangle()
        background.set_fill(interpolate_color(BLUE_E, BLACK, 0.9), 1)
        background.set_shading(0.1, 0.1, 0)

        slides = ScreenRectangle()
        slides.set_fill(BLACK, 1)
        slides.set_stroke(WHITE, 1)
        slides.set_height(6)
        slides.to_edge(LEFT, buff=0.1)

        speaker = ScreenRectangle()
        speaker.set_width(3)
        speaker.set_fill(BLACK, 1)
        speaker.set_stroke(WHITE, 1)
        speaker.next_to(slides, RIGHT, aligned_edge=UP)

        self.add(background)
        self.add(slides)
        self.add(speaker)


class IntroSphereAnimation(InteractiveScene):
    def construct(self):
        # Circles to sphere
        frame = self.frame
        frame.set_height(4)

        circle = Circle(radius = 1)
        circle.set_stroke(TEAL, 2)

        lattitude_lines = VGroup(
            Circle(radius=math.sqrt(1 - z**2)).set_z(z)
            for z in np.linspace(-0.99, 0.99, 50)
        )
        lattitude_lines.set_stroke(TEAL, 1, 0.5)

        self.play(ShowCreation(circle))
        self.remove(circle)
        self.play(
            LaggedStart(
                (TransformFromCopy(circle, lat_line)
                for lat_line in lattitude_lines),
                lag_ratio=0.05,
            ),
            frame.animate.reorient(-2, 56, 0, (0.02, -0.09, -0.07), 2.83),
            run_time=6,
        )
        frame.add_ambient_rotation(4 * DEG)
        self.wait()

        # Spheres
        mesh = SurfaceMesh(Sphere(radius=1), resolution=(101, 201))
        sphere = VMobject()
        for part in mesh:
            sphere.append_vectorized_mobject(part)
        sphere.set_stroke(BLUE, 0.1)

        lattitude_spheres = VGroup()
        pre_points = sphere.get_points().copy()
        for w in np.linspace(-0.99, 0.99, 20):
            points_4d = np.zeros((len(pre_points), 4))
            points_4d[:, :3] = pre_points * math.sqrt(1 - w**2)
            points_4d[:, 3] = w
            lat_sphere = sphere.copy()
            lat_sphere.points_4d = points_4d
            lattitude_spheres.add(lat_sphere)

        lattitude_spheres.set_stroke(opacity=0.1)

        angle_tracker = ValueTracker(0)

        def update_lat_sphere(lat_sphere):
            rot_matrix_T = rotation_matrix_transpose(angle_tracker.get_value(), axis=UR)
            pre_proj = lat_sphere.points_4d.copy()
            pre_proj[:, 1:] = np.dot(pre_proj[:, 1:], rot_matrix_T)
            lat_sphere.set_points(pre_proj[:, :3])

        for lat_sphere in lattitude_spheres:
            lat_sphere.add_updater(update_lat_sphere)

        self.play(FadeIn(sphere), FadeOut(lattitude_lines))
        self.wait()
        self.play(
            LaggedStart(
                (ReplacementTransform(sphere.copy().scale(0.99).set_opacity(0), lat_sphere)
                for lat_sphere in lattitude_spheres),
                lag_ratio=0.05,
            ),
            FadeOut(sphere),
            angle_tracker.animate.set_value(80 * DEG),
            run_time=6
        )
        self.wait()
        self.play(
            angle_tracker.animate.set_value(360 * DEG),
            run_time=12,
        )
        self.wait()


class IntroText(InteractiveScene):
    def construct(self):
        # Test
        group = VGroup(
            Text("Exploring high-dimensional spheres", font_size=90),
            Text("Delivered at UC Santa Cruz on February 17, 2026"),
        )
        group[1].set_color(GREY_A)
        group.arrange(DOWN, buff=0.5)
        group.set_width(FRAME_WIDTH - 3)
        self.play(LaggedStart(
            Write(group[0], lag_ratio=0.1),
            FadeIn(group[1], 0.25 * DOWN),
            lag_ratio=0.6
        ))
        self.wait()


class CornerDistance(InteractiveScene):
    def construct(self):
        words = Text("Distance to corner")
        pythag = Tex(R"\sqrt{1^2 + 1^2 + \cdots + 1^2 + 1^2}")
        sqrt3 = Tex(R"= \sqrt{N}")
        r_eq = Tex(R"r = \sqrt{N} - 1")

        words.move_to(2 * UP)
        equals = Tex(R"=", font_size=72).rotate(90 * DEG)
        equals.next_to(words, DOWN)
        pythag.next_to(equals, DOWN)
        sqrt3.next_to(pythag, RIGHT)
        r_eq.next_to(pythag, DOWN, buff=LARGE_BUFF)

        self.add(words, equals, pythag, sqrt3, r_eq)


class Hypercube(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        bases = [RIGHT, UP, OUT, np.array([1.5, 1.0, 0.5])]
        pre_cube = self.get_cube([*bases[:3], ORIGIN])
        hypercube = self.get_cube(bases)

        frame.reorient(-21, 79, 0, (1.13, 0.35, 0.88), 3.81)
        frame.add_ambient_rotation(2 * DEG)
        self.add(pre_cube)
        self.wait()
        self.play(ReplacementTransform(pre_cube, hypercube, run_time=2))
        self.wait(8)

        # Flatten
        flat_bases = [RIGHT, UP, np.array([0.2, 0.5, 0]), 1 * OUT]
        flat_cube = self.get_cube(flat_bases)

        leg1 = Line(bases[0], bases[1] + bases[2])
        leg2 = Line(bases[0], bases[0] + bases[3])
        leg1.set_stroke(RED, 3)
        leg2.set_stroke(TEAL, 3)

        root3_label = Tex(R"\sqrt{3}", font_size=24)
        root3_label.rotate(90 * DEG, RIGHT)
        root3_label.next_to(leg1.get_center(), LEFT)
        one_label = Tex(R"1", font_size=24)
        one_label.rotate(90 * DEG, RIGHT)
        one_label.next_to(leg2.get_center(), IN)

        hyp = Line(leg1.get_end(), leg2.get_end())
        hyp.set_stroke(YELLOW, 3)
        hyp_label = Tex(R"\sqrt{3 + 1}", font_size=24)
        hyp_label.rotate(90 * DEG, RIGHT)
        hyp_label.next_to(hyp.get_center(), OUT, buff=SMALL_BUFF)

        self.play(
            hypercube.animate.set_stroke(WHITE, 1, 0.5),
            ShowCreation(leg1),
            FadeIn(root3_label)
        )
        self.play(
            ShowCreation(leg2),
            FadeIn(one_label),
        )
        self.wait()
        self.play(
            FadeTransformPieces(VGroup(root3_label, one_label).copy(), hyp_label),
            ShowCreation(hyp)
        )
        self.wait(12)

    def get_cube(self, bases, stroke_color=WHITE, stroke_width=2):
        n = len(bases)
        lines = VGroup()
        for bits in it.product(*n * [[0, 1]]):
            base_point = sum([
                bit * basis
                for bit, basis in zip(bits, bases)
            ])
            for bit, basis in zip(bits, bases):
                if bit == 0:
                    lines.add(Line(base_point, base_point + basis))
        lines.set_stroke(stroke_color, stroke_width)
        return lines


class REqExample(InteractiveScene):
    def construct(self):
        # Test
        tex = Tex(R"r = \sqrt{10} - 1 \approx 2.162")
        self.add(tex)


class HyperbolaSquare(InteractiveScene):
    def construct(self):
        # Label distances
        square = VGroup(
            Line(UL, UR),
            Line(UR, DR),
            Line(DR, DL),
            Line(DL, UL),
        )
        square.set_height(4)
        square_shadows = VGroup(
            square.copy().scale(0.95**n).shift(0.05 * DR * n).set_stroke(opacity=0.5 / n)
            for n in range(1, 20)
        )

        center_dot = Dot()
        center_dot.move_to(square)
        side_line = Line(square.get_center(), square.get_right())
        diag_line = Line(square.get_center(), square.get_corner(UR))
        side_line.set_stroke(BLUE, 3)
        diag_line.set_stroke(YELLOW, 3)

        side_label = Tex(R"1")
        side_label.next_to(side_line, DOWN, SMALL_BUFF)
        diag_label = Tex(R"\sqrt{N}")
        diag_label.next_to(diag_line.get_center(), UL, SMALL_BUFF)
        diag_label.set_backstroke(BLACK, 5)

        self.add(square)
        self.add(square_shadows)

        self.play(
            GrowFromCenter(center_dot),
            ShowCreation(side_line),
            FadeIn(side_label, 0.5 * RIGHT)
        )
        self.wait()
        self.play(
            ShowCreation(diag_line),
            FadeIn(diag_label, 0.5 * UR),
        )
        self.wait()

        # Warp it
        warp_square = self.get_warp_square(square)
        warp_square_shadows = VGroup(self.get_warp_square(ss) for ss in square_shadows)

        side_line.target = side_line.generate_target()
        side_line.target.put_start_and_end_on(center_dot.get_center(), warp_square[1].pfp(0.5))
        diag_line.target = diag_line.generate_target()
        diag_line.target.put_start_and_end_on(center_dot.get_center(), warp_square.get_corner(UR))

        self.wait()
        self.play(
            Transform(square, warp_square),
            Transform(square_shadows, warp_square_shadows),
            MoveToTarget(side_line),
            MoveToTarget(diag_line),
            side_label.animate.next_to(side_line.target, DOWN, SMALL_BUFF),
            diag_label.animate.shift(0.4 * DOWN + 0.2 * LEFT),
            run_time=3
        )
        self.wait()

        # Corner spheres
        frame = self.frame
        lil_radius = 0.6
        big_radius = diag_line.get_length() - lil_radius
        circles = Circle(radius=lil_radius).replicate(4)
        circles.set_stroke(BLUE, 3)
        circles.set_fill(BLUE, 0.25)
        for circle, side in zip(circles, warp_square):
            circle.move_to(side.get_start())

        big_circle = Circle(radius=big_radius)
        big_circle.set_stroke(GREEN, 3).set_fill(GREEN, 0.25)

        self.play(
            LaggedStartMap(GrowFromCenter, circles),
            frame.animate.set_height(9),
        )
        self.wait()
        self.play(GrowFromCenter(big_circle))
        self.wait()

        # Show many more corners
        group = VGroup(warp_square, circles)
        n_new_groups = 12
        angles = np.linspace(0, 90 * DEG, n_new_groups + 2)[1:-1]
        alt_groups = VGroup(
            group.copy().rotate(theta, about_point=ORIGIN)
            for theta in angles
        )
        alt_groups.fade(0.75)

        corner_label = Tex(R"2^N \text{ corners}", font_size=60)
        corner_label.to_corner(UR)
        corner_label.fix_in_frame()

        self.play(
            FadeIn(corner_label),
            LaggedStart(
                (TransformFromCopy(group.copy().set_fill(opacity=0), alt_group, path_arc=angle)
                for angle, alt_group in zip(angles, alt_groups)),
                lag_ratio=0.01,
                run_time=3
            ),
            frame.animate.set_height(11),
            run_time=3
        )
        self.wait()

    def get_warp_square(self, square, scale_factor=1.7):
        hyper = FunctionGraph(lambda x: math.sqrt(1 + x**2), x_range=(-2, 2, 0.02))
        warp_square = VGroup(
            hyper.copy().put_start_and_end_on(*side.get_start_and_end())
            for side in square
        )
        warp_square.scale(scale_factor)
        warp_square.match_style(square)
        return warp_square


class AreaCircleOverAreaSquareThenVolume(InteractiveScene):
    def construct(self):
        # Left hand side
        circle_color = TEAL
        square_color = GREEN
        circle_color = BLUE
        square_color = RED

        font_size = 72
        c_tex = R"{CC \over CC}"
        s_tex = R"{SS \over SS}"
        frac = Tex(
            fR"{{\text{{Area}}\left( {c_tex} \right) \over \text{{Area}}\left( {s_tex} \right)}}",
            font_size=font_size
        )
        area_words = frac[R"\text{Area}"]
        for part in area_words:
            part.scale(0.75, about_edge=RIGHT).shift(SMALL_BUFF * RIGHT)
        frac.next_to(ORIGIN, LEFT)
        frac.to_edge(UP, LARGE_BUFF)
        circle = Circle()
        circle.set_stroke(circle_color, 3)
        circle.set_fill(circle_color, 0.25)
        square = Square()
        square.set_stroke(square_color, 3)
        square.set_fill(square_color, 0.25)

        circle.replace(frac[c_tex])
        circle.shift(0.05 * UP)
        square.replace(frac[s_tex])
        frac[c_tex].scale(0).set_opacity(0)
        frac[s_tex].scale(0).set_opacity(0)

        self.wait(0.1)
        self.play(
            Write(frac, lag_ratio=1e-1),
            LaggedStart(
                Write(circle),
                Write(square),
                lag_ratio=0.5
            )
        )
        self.wait()

        # Right hand side
        equals = Tex(R"=", font_size=90)
        equals.rotate(90 * DEG)
        equals.next_to(frac, DOWN)
        approx = Tex(R"\approx", font_size=font_size)
        value = DecimalNumber(PI / 4, num_decimal_places=3, font_size=font_size)
        rhs = Tex(
            R"{\pi (1)^2 \over 2 \times 2}",
            font_size=font_size,
            t2c={R"\pi (1)^2": circle_color, R"2 \times 2": square_color}
        )
        rhs.next_to(equals, DOWN)
        approx.next_to(rhs, RIGHT)
        value.next_to(approx, RIGHT)

        self.play(Write(equals), Write(rhs))
        self.wait()
        self.play(Write(approx), FadeIn(value, RIGHT))
        self.wait()

        # Make it three d
        sphere = Sphere()
        sphere.set_color(circle_color, 1)
        sphere.rotate(90 * DEG, RIGHT)
        sphere.replace(circle)
        sphere_mesh = SurfaceMesh(sphere)
        sphere_mesh.set_stroke(WHITE, 1, 0.25)
        sphere_mesh.deactivate_depth_test()
        cube = VCube()
        cube.set_fill(square_color, 0.2)
        cube.set_stroke(square_color, 3)
        cube.deactivate_depth_test()
        cube.rotate(20 * DEG, RIGHT).rotate(0 * DEG, UP)
        cube.replace(square).scale(0.9)

        volume_words = Text("Volume").replicate(2)
        for v_word, a_word in zip(volume_words, area_words):
            v_word.move_to(a_word, RIGHT)

        new_frac = Tex(
            R"{(4/3) \pi (1)^3 \over 2 \times 2 \times 2}",
            t2c={R"(4/3) \pi (1)^3": circle_color, R"2 \times 2 \times 2": square_color},
            font_size=font_size
        )
        new_frac.next_to(equals, DOWN)

        self.play(
            Write(sphere_mesh, lag_ratio=1e-2),
            FadeOut(VGroup(equals, rhs, approx, value)),
            FadeTransform(circle, sphere),
            FadeTransform(square, cube),
            *(
                FadeTransformPieces(a_word, v_word)
                for v_word, a_word in zip(volume_words, area_words)
            )
        )
        self.play(
            Write(equals),
            FadeIn(new_frac, DOWN),
        )
        self.wait()

        # New approx
        value = DecimalNumber((4 / 3) * PI / 8, font_size=font_size, num_decimal_places=3)
        approx = Tex(R"\approx", font_size=font_size)
        approx.next_to(new_frac, RIGHT)
        value.next_to(approx, RIGHT)

        self.play(
            Write(approx),
            FadeIn(value, RIGHT),
        )


class BigVolumeRatio(InteractiveScene):
    def construct(self):
        # Test
        fraction = Tex(R"\text{Vol}\Big(\text{100D Unit Ball}\Big) \over 2^{100}")
        fraction["2^{100}"].scale(1.5, about_edge=UP).shift(SMALL_BUFF * DOWN)

        numerator = fraction[R"\text{Vol}\Big(\text{100D Unit Ball}\Big)"]
        numerator_rect = SurroundingRectangle(numerator, buff=0)
        numerator_rect.set_stroke(BLUE, 3)
        randy = Randolph(height=2)
        randy.next_to(fraction, LEFT, MED_LARGE_BUFF)
        randy.shift(0.5 * DOWN)

        self.wait(0.1)
        self.play(FadeIn(fraction))
        self.wait(3)
        self.play(
            VFadeIn(randy),
            randy.change("confused"),
            ShowCreation(numerator_rect)
        )
        self.play(Blink(randy))
        self.wait()

        # Reveal answer
        new_numerator = Tex(R"\pi^{50} / 50!", font_size=72)
        new_numerator.move_to(numerator, DOWN)

        self.play(
            FadeOut(numerator_rect),
            fraction[R"\over"].animate.match_width(new_numerator, stretch=True),
            FadeTransform(numerator, new_numerator),
            randy.change("tease"),
        )
        self.play(Blink(randy))
        self.wait()


class Derivatives(InteractiveScene):
    def construct(self):
        kw = dict(t2c={"r": BLUE})
        power_rules = VGroup(
            Tex(R"\frac{d}{dr} r = 1", **kw),
            Tex(R"\frac{d}{dr} r^2 = 2r", **kw),
            Tex(R"\frac{d}{dr} r^3 = 3r^2", **kw),
            Tex(R"\frac{d}{dr} r^4 = 4r^3", **kw),
            Tex(R"\vdots", **kw),
        )
        power_rules.arrange(DOWN, aligned_edge=LEFT, buff=0.6)
        power_rules[-1].match_x(power_rules[-2]["="])
        self.add(power_rules)


class Integrals(InteractiveScene):
    def construct(self):
        kw = dict(t2c={"r": BLUE, "R": TEAL})
        inv_power_rules = VGroup(
            Tex(R"\int_0^R 1 \, dr = R", **kw),
            Tex(R"\int_0^R r \, dr = \frac{1}{2} R^2", **kw),
            Tex(R"\int_0^R r^2 \, dr = \frac{1}{3} R^3", **kw),
            Tex(R"\int_0^R r^3 \, dr = \frac{1}{4} R^4", **kw),
            Tex(R"\vdots", **kw),
        )
        inv_power_rules.arrange(DOWN, aligned_edge=LEFT, buff=0.6)
        inv_power_rules[-1].match_x(inv_power_rules[-2]["="])
        self.add(inv_power_rules)


class DrawTorus(InteractiveScene):
    def construct(self):
        # Add torus
        frame = self.frame
        r1 = 3
        r2 = 1

        circle1 = Circle().set_fill(GREEN_E, 0.5).set_stroke(GREEN, 2)
        circle2 = Circle().set_stroke(RED, 2)
        circles = VGroup(circle1, circle2)
        circles.arrange(RIGHT, buff=LARGE_BUFF)

        self.add(circles)
        self.wait()
        self.play(
            frame.animate.reorient(19, 63, 0, (0.05, -0.15, -0.75), 9.57),
            circle1.animate.rotate(90 * DEG, LEFT).move_to(r1 * RIGHT),
            circle2.animate.set_width(2 * r1).move_to(ORIGIN),
            run_time=2
        )
        
        # Add torus
        torus = Torus(r1=r1, r2=r2)
        partial_torus = Torus()
        partial_torus.set_color(GREY, 0.5)
        partial_torus.always_sort_to_camera(self.camera)
        self.play(
            UpdateFromAlphaFunc(
                partial_torus,
                lambda m, a: m.pointwise_become_partial(torus, 0, a, axis=0),
            ),
            Rotate(circle1, TAU, about_point=ORIGIN),
            run_time=5
        )
        self.wait()


class GammaGraph(InteractiveScene):
    def construct(self):
        # Test
        axes = Axes((-1, 5), (0, 20), width=FRAME_WIDTH - 1, height=FRAME_HEIGHT - 1.5)
        axes.x_axis.add_numbers()
        self.add(axes)

        import scipy
        graph = axes.get_graph(lambda x: scipy.special.gamma(x + 1), x_range=(-0.99, 5, 0.01))
        graph.set_stroke(TEAL, 5)
        self.add(graph)


class CommentOnGeneralFormula(InteractiveScene):
    def construct(self):
        morty = Mortimer()
        morty.to_corner(DR)
        randy = Randolph()
        randy.next_to(morty, LEFT, buff=1.5)

        self.play(
            morty.says("Beautiful!", mode="hooray"),
            randy.change("hesitant", morty.eyes)
        )
        self.play(Blink(morty))
        self.wait()
        self.play(
            morty.debubble(mode="guilty"),
            randy.says("What about\nodd n?", mode="confused", look_at=3 * LEFT + DOWN)
        )
        self.play(Blink(randy))
        self.wait()


class SquarePyramid(InteractiveScene):
    def construct(self):
        # Show pyramid
        frame = self.frame

        n_squares = 40
        max_side_length = 2
        squares = VGroup(
            Square(side_length)
            for side_length in np.linspace(max_side_length, 0.01, n_squares)
        )
        squares.arrange(OUT, buff=max_side_length / n_squares)
        squares.set_fill(RED, 0.2)
        squares.set_stroke(RED, 2, 0.5)

        frame.reorient(21, 67, 0, (-0.2, -0.12, -0.92), 4.41)
        self.play(
            LaggedStartMap(FadeIn, squares, shift=RIGHT, lag_ratio=0.35, run_time=8),
        )

        # Volume
        label =Tex(R"\text{Vol.} = \frac{1}{3} \text{B} \times \text{H}", font_size=72)
        label.fix_in_frame()
        label.to_edge(DOWN)
        label.set_backstroke(BLACK, 5)

        self.play(Write(label))
        self.wait()


class ShellTimesThickness(InteractiveScene):
    def construct(self):
        # Test
        equation = Tex(R"V(\partial B^n) \times {1 \over n} = V(B^n)", font_size=72)
        shell_rect = SurroundingRectangle(equation[R"V(\partial B^n)"]).set_stroke(TEAL, 2)
        thickness_rect = SurroundingRectangle(equation[R"{1 \over n}"]).set_stroke(YELLOW, 2)
        volume_rect = SurroundingRectangle(equation[R"V(B^n)"]).set_stroke(BLUE, 2)
        rects = VGroup(shell_rect, thickness_rect, volume_rect)
        for rect in rects:
            rect.set_height(rects.get_height(), stretch=True)
            rect.align_to(rects, UP)

        labels = VGroup()
        for rect, text in zip(rects, ["Shell", "Thickness", "Total\nVolume"]):
            label = Text(text, font_size=48)
            label.match_color(rect)
            label.next_to(rect, DOWN)
            labels.add(label)
        # for label in labels:
        #     label.align_to(labels, DOWN)

        radius_label = Text("(When radius=1)", font_size=36)
        radius_label.set_color(GREY_C)
        radius_label.next_to(labels, DOWN, MED_LARGE_BUFF).match_x(equation)

        self.add(equation, radius_label)
        self.add(rects)
        self.add(labels)


class VolumeRatio(InteractiveScene):
    def construct(self):
        # 3D
        group = VGroup(
            Tex(R"{8  \cdot \big({4 \over 3} \pi \big) \over 2 \times 2 \times 2}"),
            Tex(R"="),
            Tex(R"{4 \over 3} \pi")
        )
        group.arrange(RIGHT, buff=0.25)
        self.add(group)

        # 100D
        self.clear()
        group = VGroup(
            Tex(R"{2^{100}  \cdot \big(\pi^{50} / 50! \big) \over 2^{100}}"),
            Tex(R"="),
            Tex(R"{\pi^{50} \over 50!}"),
        )
        group.arrange(RIGHT, buff=0.5)
        self.add(group)


class EndScreen(SideScrollEndScreen):
    scroll_time = 25
