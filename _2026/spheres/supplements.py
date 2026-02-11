from manim_imports_ext import *


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