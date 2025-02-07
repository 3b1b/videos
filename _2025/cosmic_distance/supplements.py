from manim_imports_ext import *
from _2025.cosmic_distance.planets import *


class TerenceLabel(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Terence Tao", font_size=72)
        title.to_edge(UP)
        title.set_fill(border_width=3)
        title.set_backstroke(BLACK, 5)

        subtitle = VGroup(
            Text("Professor of Mathematics at UCLA", font_size=42),
        )
        subtitle.next_to(title, DOWN, LARGE_BUFF, aligned_edge=LEFT)
        subtitle.set_fill(border_width=1.5)
        subtitle.set_backstroke(BLACK, 3)

        arrow = Arrow(title.get_bottom(), title.get_bottom() + (2.5, -1.0, 0), thickness=5, path_arc=45 * DEG)
        arrow.set_fill(WHITE, 1)
        arrow.set_backstroke(BLACK, 2)

        self.add(title)
        self.wait()
        self.play(FadeIn(subtitle, lag_ratio=0.1, run_time=2))
        self.wait()
        return

        self.play(Write(arrow, run_time=2))
        self.wait()


class IntroducingTao(InteractiveScene):
    def construct(self):
        # Add images
        with_erdos = ImageMobject("TerryTaoPaulErdos")
        at_imo = ImageMobject("TerryTaoIMO")
        fields_medal = ImageMobject("TerryTaoFieldsMedal")
        images = Group(with_erdos, at_imo, fields_medal)
        for image in images:
            image.set_height(3)

        images.arrange(RIGHT, buff=1.5)
        images.set_width(FRAME_WIDTH - 1)
        images[1].scale(1.25, about_edge=DOWN)

        labels = VGroup(
            Text("Age 10, with Paul Erdős"),
            Text("Age 13, Gold medal at the\nInternationalMath Olympiad"),
            Text("Age 31, Receiving the\n2006 Fields Medal"),
        )
        for image, label in zip(images, labels):
            label.scale(0.6)
            label.next_to(image, DOWN)
            self.play(
                FadeIn(image, 0.25 * UP, scale=1.2),
                FadeIn(label, lag_ratio=0.1)
            )
            self.wait(0.5)
        self.wait()


class PilesOfResearchTopics(InteractiveScene):
    def construct(self):
        folder = '/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2025/cosmic_distance/supplements/TerenceTaoPapers'
        images = Group()
        for n, filename in enumerate(os.listdir(folder)):
            image = ImageMobject(os.path.join(folder, filename))
            image.set_opacity(1.0)
            image.set_height(6)
            image.move_to((3 * n % 10) * 1.0 * RIGHT + n * 0.1 * DOWN)
            border = SurroundingRectangle(image, buff=0)
            border.set_stroke(GREY_C, 3)
            border.set_anti_alias_width(5)
            group = Group(image, border)

            images.add(group)

        images.center().to_edge(UP, buff=0.25)

        self.play(LaggedStartMap(FadeIn, images, shift=0.2 * UP, lag_ratio=0.5, run_time=12))


class AskingTaoForTopics(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer()
        morty.to_edge(DOWN).set_x(4)
        morty.body.insert_n_curves(100)

        tau = TauCreature("plain")
        tau[4].set_fill(border_width=1)
        tau.set_height(3)
        tau.to_edge(DOWN)
        tau.set_x(-4)

        self.add(morty)
        self.add(tau)

        self.play(morty.change("raise_right_hand"))
        self.play(Transform(tau, TauCreature("pondering").move_to(tau)))
        self.wait()
        self.play(Blink(morty))
        self.play(tau[:4].animate.stretch(0, 1, about_edge=DOWN), rate_func=squish_rate_func(there_and_back, 0.4, 0.6))
        self.wait()

        self.play(
            Transform(tau, TauCreature("hooray").move_to(tau)),
            FadeIn(tau.get_bubble("How about the\nCosmic Distance Ladder?", bubble_type=SpeechBubble), lag_ratio=0.1, run_time=2),
            morty.change("tease", look_at=ORIGIN),
        )
        self.play(Blink(morty))
        self.play(tau[:4].animate.stretch(0, 1, about_edge=DOWN), rate_func=squish_rate_func(there_and_back, 0.4, 0.6))


class TableOfContents(InteractiveScene):
    def construct(self):
        items = VGroup(
            Text("Rung 1: Earth"),
            Text("Rung 2: Moon"),
            Text("Rung 3: Sun"),
            Text("Rung 4: Shapes of orbits"),
            Text("Rung 5: Distances to planets"),
            Text("Rung 6: Speed of light"),
            Text("Rung 7: Nearby stars"),
            Text("Rung 8: Milky way"),
            Text("Rung 9: Nearby Galaxies"),
            Text("Rung 10: Distant galaxies"),
        )
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        items.set_height(FRAME_HEIGHT - 1)
        items.to_edge(LEFT)

        self.play(LaggedStartMap(FadeIn, items, shift=0.5 * LEFT, lag_ratio=0.1))
        self.wait()

        # Slow pan through
        items.save_state()
        highlight_points = Group()

        def update_item(item):
            y_diff = item.get_y() - item.highlight_point.get_y()
            alpha = np.exp(-(2 * y_diff)**2)
            item.set_height(interpolate(0.3, item.start_height, alpha), about_edge=LEFT)
            item.set_opacity(interpolate(0.5, 1, alpha))

        for item in items:
            item.start_height = item.get_height()
            item.highlight_point = Point(item.get_center())
            item.add_updater(update_item)

            highlight_points.add(item.highlight_point)

        self.play(
            *(item.highlight_point.animate.set_y(items[0].get_y()) for item in items),
            run_time=2
        )
        self.play(
            *(item.highlight_point.animate.set_y(-4) for item in items),
            run_time=20,
            rate_func=linear
        )
        self.play(
            *(item.highlight_point.animate.match_y(item) for item in items),
            run_time=2
        )
        self.wait()

        # Show the two parts
        brace1 = Brace(items[:4], RIGHT)
        brace2 = Brace(items[4:], RIGHT)

        self.play(
            GrowFromCenter(brace1),
            *(item.highlight_point.animate.shift(DOWN) for item in items[4:])
        )
        self.wait()
        self.play(
            *(item.highlight_point.animate.shift(DOWN) for item in items[:4]),
            *(item.highlight_point.animate.shift(UP) for item in items[4:]),
            ReplacementTransform(brace1, brace2),
        )
        self.wait()
        self.play(
            FadeOut(brace2),
            # *(item.highlight_point.animate.shift(DOWN) for item in items[4:]),
            # *(item.highlight_point.animate.shift(UP) for item in items[:1]),
        )
        # self.wait()
        items.clear_updaters()

        # One by one
        for n in range(len(items)):
            items.target = items.generate_target()
            for k, item in enumerate(items.target):
                if k == n:
                    item.set_height(0.45, about_edge=LEFT).set_opacity(1)
                else:
                    item.set_height(0.25, about_edge=LEFT).set_opacity(0.5)
            self.play(MoveToTarget(items))
            # self.wait()
        self.wait(2)


class MainCharacterTimeline(InteractiveScene):
    def construct(self):
        # Add the timeline
        frame = self.frame

        timeline = NumberLine(
            (-500, 2000, 10),
            tick_size=0.05,
            longer_tick_multiple=2,
            big_tick_spacing=100,
            unit_size=1 / 50
        )
        numbers =timeline.add_numbers(
            range(-500, 2100, 100),
            group_with_commas=False,
            font_size=20,
            buff=0.15
        )
        for number in numbers[:5]:
            number.remove(number[0])
            bce = Text("BCE")
            bce.set_height(0.75 * number.get_height())
            bce.next_to(number, RIGHT, buff=0.05, aligned_edge=DOWN)
            number.add(bce)
            number.shift(0.15 * LEFT)

        self.add(timeline)
        frame.move_to(timeline.n2p(-175))

        # Characters
        characters = [
            ("Aristotle", -384, -322, 0.2, BLUE_D),
            ("Eratosthenes", -276, -194, 0.2, BLUE_B),
            ("Aristarchus", -310, -230, 0.5, BLUE_C),
            ("Kepler", 1571, 1630, 0.2, RED_C),
            ("Copernicus", 1473, 1543, 0.2, RED_A),
            ("Brahe", 1546, 1601, 0.5, RED_E),
        ]
        character_labels = VGroup()
        for name, start, end, offset, color in characters:
            line = Line(timeline.n2p(start), timeline.n2p(end))
            line.set_stroke(color, 2)
            line.shift(offset * UP)
            name_mob = Text(name, font_size=24)
            name_mob.set_color(color)
            name_mob.next_to(line, UP, buff=0.05)
            dashes = VGroup(
                DashedLine(line.get_start(), timeline.n2p(start), dash_length=0.025),
                DashedLine(line.get_end(), timeline.n2p(end), dash_length=0.025),
            )
            dashes.set_stroke(color, 1)
            line_group = VGroup(line, name_mob, dashes)
            character_labels.add(line_group)

        images = Group(
            ImageMobject("Head_of_Aristotle"),
            ImageMobject("Eratosthenes"),
            Square().set_opacity(0),
            ImageMobject("Kepler"),
            ImageMobject("Copernicus"),
            ImageMobject("TychoBrahe"),
        )
        for image, character_label in zip(images, character_labels):
            image.set_height(2.0)
            image.next_to(character_label, UP)

        frame.set_height(5).move_to(timeline.n2p(-250) + UP)
        self.play(
            FadeIn(character_labels[0], lag_ratio=0.1),
        )
        self.wait()
        self.play(
            FadeIn(character_labels[1], lag_ratio=0.1),
            FadeIn(images[1], 0.5 * UP),
            frame.animate.set_height(6).move_to(timeline.n2p(-175) + UP).set_anim_args(run_time=3),
        )
        self.wait()
        self.play(
            FadeIn(character_labels[2], lag_ratio=0.1),
            FadeOut(images[1], 0.5 * RIGHT),
        )
        self.wait()

        # Up to Kepler
        self.play(
            frame.animate.match_x(timeline.n2p(1600)),
            UpdateFromAlphaFunc(frame, lambda m, a: m.set_height(interpolate(6, 12, there_and_back(a)))),
            run_time=3,
        )
        self.play(
            FadeIn(character_labels[3], lag_ratio=0.1),
            FadeIn(images[3], 0.5 * UP),
        )
        self.wait()
        self.play(
            FadeIn(character_labels[4], lag_ratio=0.1),
            FadeIn(images[4], 0.5 * UP),
        )
        self.wait()
        self.play(
            FadeIn(character_labels[5], lag_ratio=0.1),
            FadeIn(images[5], 0.5 * UP),
            images[4].animate.scale(0.5, about_edge=DL).shift(0.25 * RIGHT),
            images[3].animate.scale(0.5, about_edge=DR).shift(0.25 * DR),
        )
        self.wait()


class WhatVsHow(InteractiveScene):
    def construct(self):
        # Test
        titles = VGroup(
            Text("What"),
            Text("How"),
        )
        titles.scale(1.5)
        for title, vect in zip(titles, [LEFT, RIGHT]):
            title.move_to(vect * FRAME_WIDTH / 4)
            title.to_edge(UP, buff=0.25)

        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        self.play(
            FadeIn(titles[0], 0.5 * UP),
            ShowCreation(v_line),
        )
        self.wait()
        self.play(FadeIn(titles[1], 0.5 * UP))
        self.wait()


class ProjectionTheorem(InteractiveScene):
    def construct(self):
        # Add plane
        frame = self.frame

        width = 10
        plane = Square3D(side_length=2 * width)
        plane.set_color(GREY_E, 0.5)
        grid = NumberPlane(
            (-width, width), (-width, width),
        )
        grid.axes.set_stroke(WHITE, 0.5, 0.5)
        grid.background_lines.set_stroke(WHITE, 1, 0.25)
        grid.faded_lines.set_stroke(WHITE, 0.25, 0.1)

        frame.reorient(-30, 70, 0, 1.5 * OUT, 8.00)
        frame.add_ambient_rotation(2 * DEG)
        self.add(plane, grid)

        # Add sphere
        sphere = Sphere(true_normals=False)
        sphere.set_color(BLUE_E, 0.5)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere)
        mesh.set_stroke(WHITE, 1, 0.25)
        mesh.set_anti_alias_width(0)

        sphere_group = Group(sphere, mesh)
        sphere_group.move_to(2 * OUT)

        self.add(sphere_group)

        # Show many projections
        for _ in range(10):
            projection = self.show_projection(sphere_group)
            self.play(
                Rotate(sphere_group, random.uniform(0, TAU), axis=normalize(np.random.random(3))),
                FadeOut(projection)
            )

    def show_projection(self, sphere):
        # Test
        projection = sphere.copy()
        projection.stretch(5e-2, 2)
        projection.set_z(1e-2)

        circle = Circle(radius=1)
        circle.set_stroke(TEAL, 2)

        z = sphere.get_z()
        v_lines = VGroup(
            Line(circle.pfp(a) + z * OUT, circle.pfp(a))
            for a in np.arange(0, 1, 1.0 / 12.0)
        )
        v_lines.set_stroke(WHITE, 1, 0.25)
        v_lines.apply_depth_test()

        self.play(
            TransformFromCopy(sphere, projection),
            *map(ShowCreationThenFadeOut, v_lines),
            run_time=1
        )
        self.add(circle)
        self.wait()
        self.remove(circle)
        return projection


class RouleauxTriangle(InteractiveScene):
    def construct(self):
        # Add triangle
        arcs = VGroup(
            Arc(30 * DEG + x * 120 * DEG, 60 * DEG, radius=1)
            for x in range(3)
        )
        arcs[1].shift(arcs[0].get_end() - arcs[1].get_start())
        arcs[2].shift(arcs[0].get_start() - arcs[2].get_end())
        arcs.center()
        shape = VMobject()
        for arc in arcs:
            shape.append_vectorized_mobject(arc)

        shape.set_fill(BLUE_E, 1)
        shape.set_stroke(WHITE, 2)
        shape.set_height(3)
        shape.move_to(UP)

        shape.set_shading(0.25, 0.25, 0)

        self.add(shape)

        # Show the projection
        v_lines = DashedLine(3 * UP, 3 * DOWN).replicate(2)
        v_lines.arrange(RIGHT, buff=shape.get_width())
        v_lines.move_to(shape, UP)
        v_lines.set_stroke(GREY_C, 1)

        projection = shape.copy()
        projection.stretch(0, 1)
        projection.move_to(v_lines.get_bottom())

        self.play(
            *map(ShowCreation, v_lines),
            TransformFromCopy(shape, projection),
            run_time=2
        )
        self.wait()

        # Rotate
        shape_center = np.array([0, shape.get_y(), 0])
        shape.add_updater(lambda m: m.move_to(shape_center))
        self.play(
            Rotate(shape, PI, run_time=6),
            # self.frame.animate.set_gamma(PI),
            run_time=6
        )


class WellOfSyene(InteractiveScene):
    def construct(self):
        # Test
        image_path = Path(self.file_writer.output_directory, "../KurtArtwork/TheWellOfSyene.jpg").resolve()
        image = TexturedSurface(Square3D(resolution=(101, 101)), image_path)
        image.set_shape(FRAME_WIDTH, FRAME_HEIGHT)
        image.set_shading(0.1, 0.1, 0.1)
        self.add(image)

        # Make waves
        frame = self.frame
        center = np.array([-2.3, 0.84, 0])

        sun_spot = GlowDot([-2.36, 1.26, 0.0])
        sun_spot.set_color(WHITE)
        sun_spot.set_radius(0.7)

        def wave(x, y, z, t):
            dist = get_dist(center, [x, y, z])
            scale = 0.025 * np.exp(-2 * dist * dist)
            nudge1 = scale * math.sin(2 * TAU * x - 5 * TAU * t)
            nudge2 = 3 * scale * math.sin(3 * TAU * y - 5 * TAU * t)
            return (x, y + nudge1, z + nudge2)

        frame.reorient(0, 0, 0, (-1.64, 0.72, 0.0), 2.39)
        self.play(
            Homotopy(wave, image, rate_func=linear),
            frame.animate.to_default_state(),
            GrowFromCenter(sun_spot, time_span=(5.5, 6.5)),
            run_time=12,
        )


class NileLabels(InteractiveScene):
    def construct(self):
        # Add image
        image_path = Path(self.file_writer.output_directory, "Nile.png").resolve()
        image = ImageMobject(image_path)
        image.set_height(FRAME_HEIGHT)

        # Mark points
        alex_point = TrueDot((-5.61, -0.8, 0.0), color=MAROON_D, radius=0.1).make_3d()
        syene_point = TrueDot((2.78, 1.12, 0.0), color=BLUE_D, radius=0.1).make_3d()
        for dot in alex_point, syene_point:
            dot.set_shading(0.25, 0.1, 0.1)
        alex_label = Text("Alexandria", font_size=60, font="CMU Serif").match_color(alex_point)
        alex_label.next_to(alex_point, DR, SMALL_BUFF)
        syene_label = Text("Syene", font_size=60, font="CMU Serif").match_color(syene_point)
        syene_label.next_to(syene_point, UP, SMALL_BUFF)

        line = Line(alex_point, syene_point, buff=0)
        line.insert_n_curves(20)
        # line.set_stroke([MAROON_D, BLUE_D])
        line.set_stroke(GREY_A, 3)

        line_label = TexText(R"$\approx$ 5,000 Stadia")
        line_label.next_to(line.get_center(), UP, SMALL_BUFF)
        line_label.rotate(line.get_angle(), about_point=line.get_center())
        line_label.set_fill(border_width=2)

        group = VGroup(alex_label, syene_label, line_label)
        group.set_fill(border_width=3)
        group.set_backstroke(BLACK, 5)

        self.play(
            FadeIn(alex_point),
            FadeIn(alex_label, lag_ratio=0.1),
        )
        self.play(
            ShowCreation(line, run_time=2),
            FadeIn(syene_point, time_span=(1, 2)),
            FadeIn(syene_label, lag_ratio=0.1, time_span=(1, 2)),
        )
        self.wait()
        self.play(Write(line_label))
        self.wait()


class EarthSizeRatios(InteractiveScene):
    def construct(self):
        # Test
        equation = Tex(R"""
            {7^\circ \over 360^\circ} = 
            {\text{dist}(\text{Alexandria}, \text{Syene}) \over \text{Circumference of Earth}}
        """, font_size=42)
        equation.to_corner(UL)
        dist_term = equation[R"\text{dist}(\text{Alexandria}, \text{Syene})"][0]

        self.play(FadeIn(equation[R"7^\circ"], UL))
        self.play(Write(equation[R"\over 360^\circ}"]))
        self.wait()
        self.play(LaggedStart(
            Write(equation[R"="][0]),
            FadeTransformPieces(equation[R"7^\circ"][0].copy(), dist_term),
            FadeTransformPieces(equation[R"\over"][0].copy(), equation[R"\over"][1]),
            FadeTransformPieces(equation[R"360^\circ"][0].copy(), equation[R"\text{Circumference of Earth}"][0]),
            lag_ratio=0.1,
        ))
        self.wait()

        # Highlight dist
        rect = SurroundingRectangle(dist_term)
        self.play(ShowCreation(rect))
        self.play(FadeOut(rect))


class AccuracyLabel(InteractiveScene):
    def construct(self):
        label = VGroup(
            TexText("Estimate: $R_E = ...$"),
            TexText("True value: $R_E = 6{,}378$ km"),
        )


class MoonOrbitCalculation(InteractiveScene):
    def construct(self):
        # Pure ratio
        ratio_tex = R"{\text{28 days} \over \text{4 hours}}"
        new_ratio_tex = R"{\text{28 days} \over \text{3.5 hours}}"
        eq1, eq2 = (
            Tex(ratio_tex + R"= {2 \pi D_M \over 2 R_\text{E}}"),
            Tex(new_ratio_tex + R"= {2 \pi D_M \over 2 R_\text{E}}"),
        )
        for eq in [eq1, eq2]:
            eq.to_corner(UL)
        eq2.move_to(eq1, RIGHT)

        RMO = eq1[R"D_M"]
        RMO_rect = SurroundingRectangle(RMO, buff=0.05)
        RMO_rect.set_stroke(TEAL, 2)
        RMO_words = Text("Distance to the moon", font_size=48)
        RMO_words.next_to(RMO_rect)
        RMO_words.set_color(TEAL)

        self.add(eq1)
        self.wait()
        self.play(
            RMO.animate.set_color(TEAL),
            ShowCreation(RMO_rect),
            FadeIn(RMO_words, lag_ratio=0.1),
        )
        self.wait()
        self.play(
            Transform(eq1[R"28 days"][0], eq2[R"28 days"][0]),
            Transform(eq1[R"\over"][0], eq2[R"\over"][0]),
            Transform(eq1[R"4"], eq2[R"3.5"]),
        )
        self.wait()

        # Equation 2
        eq3 = Tex(
            R"\frac{1}{\pi} \left( " + new_ratio_tex + R"\right) R_\text{E} = D_M"
        )
        eq3.next_to(eq2, DOWN, LARGE_BUFF, aligned_edge=LEFT)
        RMO2 = eq3[R"D_M"]
        RMO2.set_color(TEAL)

        self.play(LaggedStart(
            TransformFromCopy(eq2[new_ratio_tex], eq3[new_ratio_tex]),
            FadeIn(eq3[R"\left("]),
            FadeIn(eq3[R"\right)"]),
            TransformFromCopy(eq2[R"\pi"], eq3[R"\frac{1}{\pi}"]),
            TransformFromCopy(eq2[R"R_\text{E}"], eq3[R"R_\text{E}"]),
            TransformFromCopy(eq2["="], eq3["="]),
            TransformFromCopy(eq2[R"D_M"], eq3[R"D_M"]),
            lag_ratio=0.3,
            run_time=3
        ))
        self.wait()

        # Note the fraction
        frac = eq3[R"\frac{1}{\pi} \left( " + new_ratio_tex + R"\right)"]
        rect = SurroundingRectangle(frac)
        rect.set_stroke(YELLOW, 2)
        value = Tex(R"\approx 61")
        value.next_to(rect, DOWN)
        value.match_color(rect)

        self.play(ShowCreation(rect))
        self.play(Write(value))
        self.wait()


class FullLunarEclipseDistance(InteractiveScene):
    def construct(self):
        # Add terms
        R_e = 1.5
        R_m = R_e * (MOON_RADIUS / EARTH_RADIUS)

        diam_line = Line(R_e * DOWN, R_e * UP)
        start_dot = Dot(R_e * DOWN, radius=0.04)
        end_dot = Dot((R_e + 2 * R_m) * UP, radius=0.04)

        diam_brace = Brace(diam_line, RIGHT)
        diam_brace.add(diam_brace.get_tex(R"2R_E", font_size=36))

        R_m_brace = Brace(Line(diam_line.get_top(), end_dot.get_center()), RIGHT)
        R_m_brace.add(R_m_brace.get_tex(R"2 R_M", font_size=36))

        start_words = Text("Leading edge start")
        end_words = Text("Leading edge end")

        for words, dot in [(start_words, start_dot), (end_words, end_dot)]:
            words.next_to(dot, LEFT, buff=LARGE_BUFF)
            arrow = Arrow(words, dot, buff=0.1)
            words.shift(0.05 * UP)
            words.add(arrow)
            VGroup(words, dot).set_color(TEAL)

        # Animations
        self.play(
            FadeIn(start_words, lag_ratio=0.1),
            FadeIn(start_dot),
        )
        self.wait(2)
        self.play(FadeIn(diam_brace))
        self.wait()
        self.play(
            FadeIn(end_words, lag_ratio=0.1),
            FadeIn(end_dot)
        )
        self.play(FadeIn(R_m_brace))


class ThreePointFiveCorrection(InteractiveScene):
    def construct(self):
        # Test
        four_hours = Text("4 hours")
        correction = Text("3.5 hours")
        correction.next_to(four_hours, UP, MED_LARGE_BUFF)
        cross = Line(LEFT, RIGHT).replace(four_hours, 0)
        cross.set_stroke(RED, 6)

        self.play(ShowCreation(cross))
        self.play(Write(correction))
        self.wait()


class ShowFourLittleCircles(InteractiveScene):
    def construct(self):
        # Add image
        image_path = Path(self.file_writer.output_directory, "LunarEclipseComposite.jpg").resolve()
        image = ImageMobject(str(image_path))
        image.set_height(FRAME_HEIGHT)
        self.add(image)

        # Circles
        lil_circle = Circle(radius=0.62)
        lil_circle.move_to([-1.00, -0.167, 0.])
        lil_circle.set_stroke(YELLOW, 2)
        lil_circle.set_anti_alias_width(10)

        four_circles = lil_circle.get_grid(4, 1, buff=0)
        four_circles.rotate(8 * DEG)
        four_circles.move_to([1.503, -0.043, 0.])

        self.play(ShowCreation(lil_circle))
        self.wait()
        self.play(LaggedStart(
            (TransformFromCopy(lil_circle, circ)
            for circ in four_circles),
            lag_ratio=0.25,
            run_time=3
        ))
        self.wait()


class AskAboutMoonrise(InteractiveScene):
    def construct(self):
        # Test
        question = Text("What does the duration\nof a moonrise tell you?", font_size=60)
        question.set_backstroke(BLACK, 3)
        self.play(Write(question, stroke_color=WHITE, run_time=3))
        self.wait()


class MoonSizeCalculation(InteractiveScene):
    def construct(self):
        # Test
        eq = Tex(
            R"{2 \text{ minutes} \over 24 \text{ hours}} = {2 R_M\over 2 \pi D_M}",
            t2c={"R_M": GREY_B, "D_M": GREY_B}
        )
        eq.to_corner(UL)
        eq.set_backstroke(BLACK, 2)

        self.play(FadeIn(eq[R"{2 \text{ minutes} \over 24 \text{ hours}}"]))
        self.wait()
        self.play(LaggedStart(
            FadeIn(eq["="][0]),
            TransformFromCopy(eq[R"2 \text{ minutes}"][0], eq["2 R_M"][0]),
            TransformFromCopy(eq[R"\over"][0], eq[R"\over"][1]),
            TransformFromCopy(eq[R"24 \text{ hours}"][0], eq[R"2 \pi D_M"][0]),
            lag_ratio=0.35
        ))
        self.wait()

        # Annotations
        top_rect = SurroundingRectangle(SurroundingRectangle(eq["R_M"], buff=0.01).set_stroke(width=1))
        low_rect = SurroundingRectangle(SurroundingRectangle(eq["D_M"], buff=0.01).set_stroke(width=1))
        top_label = TexText("Radius of the moon")
        top_label.next_to(top_rect, RIGHT)
        low_label = TexText("Distance to the moon")
        low_label.next_to(low_rect, RIGHT)

        VGroup(top_rect, low_rect, top_label, low_label).set_color(TEAL)

        self.play(
            ShowCreation(top_rect),
            FadeIn(top_label, lag_ratio=0.1),
        )
        self.wait()
        self.play(
            ReplacementTransform(top_rect, low_rect),
            FadeOut(top_label),
            FadeIn(low_label, lag_ratio=0.1),
        )
        self.wait()
        self.play(FadeOut(low_rect), FadeOut(low_label))
        self.wait()


class MoonSunRatios(InteractiveScene):
    def construct(self):
        # Write equation
        equation = Tex(
            R"{R_M \over D_M} = {R_S \over D_S} \approx \frac{1}{220}",
            t2c={
                "R_M": GREY_B,
                "D_M": GREY_B,
                "R_S": YELLOW,
                "D_S": YELLOW,
            }
        )
        equation.to_corner(UL)

        self.add(equation[R"{R_M \over D_M}"])
        self.wait()
        self.play(LaggedStart(
            Write(equation["="][0]),
            TransformFromCopy(equation["R_M"][0], equation["R_S"][0]),
            TransformFromCopy(equation[R"\over"][0], equation[R"\over"][1]),
            TransformFromCopy(equation["D_M"][0], equation["D_S"][0]),
            lag_ratio=0.35
        ))
        self.wait()
        self.play(Write(equation[R"\approx \frac{1}{220}"]))
        self.wait()


class ArrowBackAndForth(InteractiveScene):
    def construct(self):
        # Test
        arrow = Arrow(3 * LEFT, 3 * RIGHT, path_arc=-90 * DEG, thickness=6)
        self.play(Write(arrow))
        self.wait()
        self.play(arrow.animate.flip())
        self.wait()


class AngleLabel(InteractiveScene):
    def construct(self):
        # Test
        arc = Arc(180 * DEG, -10 * DEG, radius=2)
        theta = Tex(R"\theta")
        theta.next_to(arc, LEFT, aligned_edge=DOWN, buff=SMALL_BUFF)

        self.add(arc, theta)


class AristarchusDistanceEstimate(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"D_S": YELLOW, "D_M": GREY_B}
        guess = TexText("Aristarchus's Guess: $D_S = 20 D_M$", t2c=t2c)
        truth = TexText(R"True answer: $D_S \approx 370 D_M$", t2c=t2c)

        guess.to_edge(UP)
        truth.next_to(guess, DOWN)

        truth.shift((guess["D_S"].get_x() - truth["D_S"].get_x()) * RIGHT)

        self.play(FadeIn(guess, 0.25 * UP))
        self.wait()
        self.play(FadeIn(truth, 0.25 * DOWN))
        self.wait()


class AristarchusSunSizeEstimate(InteractiveScene):
    def construct(self):
        # Mostly copied from above
        t2c = {"R_S": YELLOW, "R_E": BLUE}
        guess = TexText("Aristarchus's Guess: $R_S = 7 R_E$", t2c=t2c)
        truth = TexText(R"True answer: $R_S \approx 109 R_E$", t2c=t2c)

        guess.to_edge(UP)
        truth.next_to(guess, DOWN)

        truth.shift((guess["R_S"].get_x() - truth["R_S"].get_x()) * RIGHT)

        self.play(FadeIn(guess, 0.25 * UP))
        self.wait()
        self.play(FadeIn(truth, 0.25 * DOWN))
        self.wait()


class CrossAndCheck(InteractiveScene):
    def construct(self):
        # Test
        cross = Cross(Square())
        cross.set_shape(4, 3)
        cross.to_edge(LEFT)
        cross.set_stroke(RED, (0, 10, 10, 10, 0))

        checkmark = Checkmark()
        checkmark.set_height(1)
        checkmark.set_fill(GREEN)
        checkmark.next_to(cross, UP)

        self.play(ShowCreation(cross))
        self.wait()
        self.play(FadeOut(cross))
        self.play(FadeIn(checkmark, UP))
        self.wait()


class OtherGreeks(InteractiveScene):
    def construct(self):
        # Test
        folder = '/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2025/cosmic_distance/KurtArtwork/'
        base_image = ImageMobject(Path(folder, 'AristarchusProposingEarthGoesAroundTheSun.jpg'))
        sub_images = Group(
            ImageMobject(Path(folder, "stencil-luma-masks", f"figure-{n}-masked.png"))
            for n in range(1, 10)
        )
        for image in [base_image, *sub_images]:
            image.set_height(FRAME_HEIGHT)

        base_image.set_opacity(0.25)
        self.add(base_image)
        self.add(sub_images[0])
        self.play(
            *(
                FadeIn(im, rate_func=there_and_back_with_pause, time_span=(0.25 * n, 0.25 * n + 7))
                for n, im in enumerate(sub_images)
            ),
            run_time=8
        )


class EqualAreas1(InteractiveScene):
    angle = 30 * DEG
    radius = 2
    run_time = 1

    def construct(self):
        # Test
        arc = self.get_slice(1 * DEG)
        self.play(
            UpdateFromAlphaFunc(arc, lambda m, a: m.become(self.get_slice(a * self.angle))),
            run_time=self.run_time
        )
        self.wait()

    def get_slice(self, angle):
        arc = Arc(0, angle, radius=self.radius)
        arc.add_line_to(ORIGIN)
        arc.add_line_to(arc.get_start())
        arc.set_fill(BLUE, 0.5)
        arc.set_stroke(width=0)
        return arc


class EqualAreas2(EqualAreas1):
    angle = 10 * DEG
    radius = 4
    run_time = 1


class EvenWithMathRight(InteractiveScene):
    def construct(self):
        # Test
        words = TexText(R"Even when you have the math right\\you don't necessarily get to the truth", font_size=36)
        words.to_corner(UL)
        self.play(Write(words, lag_ratio=0.1, run_time=2))
        self.wait()


class CopernicusConclusions(InteractiveScene):
    def construct(self):
        # Add title
        title = TexText("Copernicus", font_size=48)
        underline = Underline(title, buff=-0.05)
        title.add(underline)
        title.to_corner(UL)
        title.set_color(YELLOW)

        self.add(title)

        # Facts
        facts = BulletedList(
            R"proposed the planets\\go around the Sun",
            R"assumed orbits\\were circular",
            R"calculated all the\\orbital periods",
            buff=0.75,
            font_size=36
        )
        facts.next_to(title, DOWN, aligned_edge=LEFT, buff=MED_LARGE_BUFF)
        facts.shift(MED_SMALL_BUFF * RIGHT)

        for fact in facts:
            self.play(FadeIn(fact, RIGHT))
            self.wait()

        # Test
        self.play(
            self.frame.animate.reorient(0, 0, 0, (0.78, -0.58, 0.0), 9.00),
            facts[:2].animate.scale(0.75, about_edge=UL).set_opacity(0.5),
            facts[2].animate.scale(1.25, about_edge=UL).shift(0.5 * UP)
        )
        self.wait()

        # Show orbital periods
        periods = VGroup(
            Text("Mercury: 88 days"),
            Text("Venus: 225 days"),
            Text("Earth: 365 days"),
            Text("Mars: 687 days"),
            Text("Jupiter: 4,333 days"),
            Text("Saturn: 10,755 days"),
        )
        periods.scale(0.75)
        periods.arrange(DOWN, aligned_edge=LEFT)
        periods.next_to(facts, DOWN, buff=MED_SMALL_BUFF, aligned_edge=LEFT)
        periods.shift(RIGHT)

        self.play(
            LaggedStartMap(FadeIn, periods, shift=0.35 * RIGHT, lag_ratio=0.15)
        )
        self.add(periods)

        # Highlight a few periods
        arrow = Vector(LEFT)
        arrow.set_fill(WHITE, 0)
        arrow.next_to(periods[0], RIGHT)
        for i, color in zip(range(2, 5), [BLUE, RED, ORANGE]):
            self.play(
                periods[:i].animate.set_fill(WHITE, 0.5),
                periods[i + 1:].animate.set_fill(WHITE, 0.5),
                periods[i].animate.set_fill(color, 1),
                arrow.animate.set_fill(color, 1).next_to(periods[i], RIGHT, SMALL_BUFF)
            )
            self.wait()


class UniversalProblemSolvingTip(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Universal Problem Solving Tip #1", font_size=60)
        title.add(Underline(title, buff=-0.025))
        title.to_edge(UP)
        title.set_color(BLUE)
        self.add(title)

        words = TexText(R"If you can't solve a problem, try \\ to solve a simpler problem instead", font_size=60, isolate="simpler problem ")
        words.next_to(title, DOWN, buff=1.5)

        self.play(FadeIn(words, lag_ratio=0.1, run_time=2))
        self.wait()
        self.play(
            FlashUnder(words["simpler problem"], buff=-0.025),
            words["simpler problem"].animate.set_color(YELLOW)
        )
        self.wait()

