from manim_imports_ext import *


class RandomChordScene(Scene):
    title = ""
    radius = 3.5
    n_samples = 1000
    long_color = BLUE
    short_color = WHITE
    chord_width = 0.5
    chord_opacity = 0.35
    run_time = 20
    include_triangle = True

    def construct(self):
        circle = self.circle = Circle(radius=self.radius)
        circle.set_stroke(GREY_B, 3)
        circle.to_edge(LEFT)

        chords = self.get_chords(circle)
        flash_chords = chords.copy()
        flash_chords.set_stroke(width=3, opacity=1)
        indicators = Group(*map(self.get_method_indicator, chords))

        title = Text(self.title)
        title.set_x(FRAME_WIDTH / 4).to_edge(UP)

        triangle = self.get_triangle(circle)

        if not self.include_triangle:
            triangle.set_opacity(0)

        self.add(circle, title, triangle)

        for s, rt in (slice(0, 16), 8), (slice(18, None), self.run_time):
            fraction = self.get_fraction([c.long for c in chords[s]])
            fraction.match_x(title)
            fraction.match_y(circle)
            self.add(fraction)
            self.remove(triangle)
            self.play(
                ShowIncreasingSubsets(chords[s]),
                ShowSubmobjectsOneByOne(flash_chords[s]),
                ShowSubmobjectsOneByOne(indicators[s]),
                Animation(triangle),
                fraction.alpha_update,
                rate_func=linear,
                run_time=rt,
            )
            self.remove(flash_chords)
            self.remove(indicators)
            self.remove(fraction)
        self.add(fraction)
        self.wait()

    def get_fraction(self, data):
        tex = Tex(
            "{100", "\\over ", "100", "+", "100}", "= ", "0.500"
        )
        nl1 = Integer(100, edge_to_fix=ORIGIN)  # Number of long chords
        nl2 = Integer(100, edge_to_fix=RIGHT)
        ns = Integer(100, edge_to_fix=LEFT)  # Number of short chords
        ratio = DecimalNumber(0, num_decimal_places=4)
        fraction = VGroup(
            nl1.move_to(tex[0]),
            tex[1],
            nl2.move_to(tex[2]),
            tex[3],
            ns.move_to(tex[4]),
            tex[5],
            ratio.move_to(tex[6], LEFT),
        )

        def update_fraction(frac, alpha):
            subdata = data[:int(np.floor(alpha * len(data)))]
            n_long = sum(subdata)
            n_short = len(subdata) - n_long
            frac[0].set_value(n_long)
            frac[0].set_color(self.long_color)
            frac[2].set_value(n_long)
            frac[2].set_color(self.long_color)
            frac[4].set_value(n_short)
            frac[4].set_color(self.short_color)

            if len(subdata) == 0:
                frac[-2:].set_opacity(0)
            else:
                frac[-2:].set_opacity(1)
                frac[6].set_value(n_long / len(subdata))
            return frac

        fraction.alpha_update = UpdateFromAlphaFunc(
            fraction, update_fraction
        )
        return fraction

    def get_chords(self, circle, chord_generator=None):
        if chord_generator is None:
            chord_generator = self.get_random_chord
        tri_len = np.sqrt(3) * circle.get_width() / 2
        chords = VGroup(*(
            chord_generator(circle)
            for x in range(self.n_samples)
        ))
        for chord in chords:
            chord.long = (chord.get_length() > tri_len)
            chord.set_color(
                self.long_color if chord.long else self.short_color
            )
        chords.set_stroke(
            width=self.chord_width,
            opacity=self.chord_opacity
        )
        return chords

    def get_triangle(self, circle):
        verts = [circle.pfp(a) for a in np.arange(0, 1, 1 / 3)]
        triangle = Polygon(*verts)
        triangle.rotate(-PI / 6, about_point=circle.get_center())
        triangle.set_stroke(RED, 2, 1)
        return triangle

    def get_random_chord(self, circle):
        return NotImplemented

    def get_method_indicator(self, chord):
        return NotImplemented


class PairOfPoints(RandomChordScene):
    title = "Random pair of circle points"

    @staticmethod
    def get_random_chord(circle):
        return Line(
            circle.pfp(random.random()),
            circle.pfp(random.random()),
        )

    @staticmethod
    def get_method_indicator(chord):
        dots = DotCloud([
            chord.get_start(),
            chord.get_end(),
        ])
        dots.set_glow_factor(2)
        dots.set_radius(0.25)
        dots.set_color(YELLOW)
        return dots


class CenterPoint(RandomChordScene):
    title = "Random point in circle"

    @staticmethod
    def get_random_chord(circle):
        x = y = 1
        while x * x + y * y > 1:
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
        return CenterPoint.chord_from_xy(x, y, circle)

    @staticmethod
    def chord_from_xy(x, y, circle):
        n2 = x * x + y * y
        temp_x = math.sqrt(n2)
        temp_y = math.sqrt(1 - n2)
        line = Line(
            [temp_x, -temp_y, 0],
            [temp_x, temp_y, 0],
        )
        line.rotate(angle_of_vector([x, y, 0]), about_point=ORIGIN)
        line.scale(circle.get_width() / 2, about_point=ORIGIN)
        line.shift(circle.get_center())
        return line

    @staticmethod
    def get_method_indicator(chord):
        dots = DotCloud([chord.get_center()])
        dots.set_glow_factor(2)
        dots.set_radius(0.25)
        dots.set_color(YELLOW)
        return dots


class RadialPoint(CenterPoint):
    title = "Random point along radial line"

    @staticmethod
    def get_random_chord(circle):
        angle = random.uniform(0, TAU)
        dist = random.uniform(0, 1)
        return CenterPoint.chord_from_xy(
            dist * math.cos(angle),
            dist * math.sin(angle),
            circle
        )

    @staticmethod
    def get_method_indicator(chord):
        dot = super().get_method_indicator(chord)
        line = Line(self.circle.get_center(), dot.get_center())
        line.set_length(self.radius, about_point=line.get_start())
        line.set_stroke(YELLOW, 1)
        return Group(dot, line)


class CompareFirstTwoMethods(RandomChordScene):
    chord_width = 0.1
    chord_opacity = 0.8

    def construct(self):
        circles = Circle().get_grid(1, 2, buff=1)
        circles.set_height(5)
        circles.to_edge(DOWN, buff=LARGE_BUFF)
        circles.set_stroke(GREY_B, 3)

        titles = VGroup(
            Text("Choose a pair of points"),
            Text("Chose a center point"),
        )
        for title, circle in zip(titles, circles):
            title.next_to(circle, UP, MED_LARGE_BUFF)

        chord_groups = VGroup(
            self.get_chords(circles[0], PairOfPoints.get_random_chord),
            self.get_chords(circles[1], CenterPoint.get_random_chord),
        )
        flash_chord_groups = chord_groups.copy()
        flash_chord_groups.set_stroke(width=3, opacity=1)
        indicator_groups = Group(
            Group(*(PairOfPoints.get_method_indicator(c) for c in chord_groups[0])),
            Group(*(CenterPoint.get_method_indicator(c) for c in chord_groups[1])),
        )

        self.add(circles, titles)

        self.play(
            *map(ShowIncreasingSubsets, chord_groups),
            *map(ShowSubmobjectsOneByOne, flash_chord_groups),
            *map(ShowSubmobjectsOneByOne, indicator_groups),
            rate_func=linear,
            run_time=15,
        )
        self.play(
            *map(FadeOut, (
                flash_chord_groups[0][-1],
                flash_chord_groups[1][-1],
                indicator_groups[0][-1],
                indicator_groups[1][-1],
            )),
        )

        self.wait()


class SparseWords(Scene):
    def construct(self):
        words = Text("Sparser in the middle")
        words.to_edge(DOWN, buff=MED_LARGE_BUFF)
        arrow = Arrow(
            words.get_top(),
            [3.5, -0.5, 0],
        )
        arrow.set_color(YELLOW)
        words.set_color(YELLOW)

        self.play(
            Write(words, run_time=1),
            ShowCreation(arrow),
        )
        self.wait()


class PortionOfRadialLineInTriangle(Scene):
    def construct(self):
        # Circle and triangle
        circle = Circle(radius=3.5)
        circle.set_stroke(GREY_B, 4)
        circle.rotate(-PI / 6)
        triangle = Polygon(*(
            circle.pfp(a) for a in np.arange(0, 1, 1 / 3)
        ))
        triangle.set_stroke(GREEN, 2)
        center_dot = Dot(circle.get_center().copy(), radius=0.04)
        center_dot.set_fill(GREY_C, 1)

        self.add(circle, center_dot, triangle)

        # Radial line
        radial_line = Line(circle.get_center().copy(), circle.pfp(1 / 6))
        radial_line.set_stroke(WHITE, 2)
        elbow = Elbow()
        elbow.rotate(PI + radial_line.get_angle(), about_point=ORIGIN)
        elbow.shift(radial_line.pfp(0.5) - ORIGIN)
        elbow.match_style(radial_line)
        half_line = radial_line.copy()
        half_line.pointwise_become_partial(radial_line, 0, 0.5)
        half_line.set_stroke(RED, 2)
        half_label = Tex("\\frac{1}{2}", font_size=30)
        half_label.next_to(half_line.get_center(), DR, SMALL_BUFF)
        half_label.set_color(RED)

        self.play(ShowCreation(radial_line))
        self.play(ShowCreation(elbow))
        self.wait()
        self.play(
            ShowCreation(half_line),
            FadeIn(half_label, 0.2 * DOWN),
        )

        # Show sample chords along line
        dot = TrueDot()
        dot.set_glow_factor(5)
        dot.set_radius(0.5)
        dot.set_color(YELLOW)
        dot.move_to(radial_line.pfp(0.1))
        circle.rotate(PI / 6)

        def get_chord():
            p = dot.get_center().copy()
            p /= (circle.get_width() / 2)
            chord = CenterPoint.chord_from_xy(p[0], p[1], circle)
            chord.set_stroke(BLUE, 4)
            return chord

        chord = always_redraw(get_chord)

        self.play(
            FadeIn(dot),
            GrowFromCenter(chord)
        )
        self.wait()
        for alpha in (0.5, 0.01, 0.4):
            self.play(
                dot.animate.move_to(radial_line.pfp(alpha)),
                run_time=4,
            )

        self.add(dot)

        # Embed
        self.embed()


class RandomPointsFromVariousSpaces(Scene):
    def construct(self):
        # Interval
        interval = UnitInterval()
        interval.add_numbers()
        top_words = TexText("Choose a random$^{*}$ ", "number between 0 and 1")
        top_words.to_edge(UP)
        subwords = TexText(
            "($^{*}$Implicitly: According to a \\emph{uniform} distribution)",
            color=GREY_A,
            tex_to_color_map={"\\emph{uniform}": YELLOW},
            font_size=36
        )
        subwords.next_to(top_words, DOWN, buff=MED_LARGE_BUFF)

        dots = DotCloud([
            interval.n2p(random.random())
            for x in range(100)
        ])
        dots.set_radius(0.5)
        dots.set_glow_factor(10)
        dots.set_color(YELLOW)
        dots.add_updater(lambda m: m)
        turn_animation_into_updater(
            ShowCreation(dots, rate_func=linear, run_time=10)
        )

        self.add(interval, dots)
        self.play(Write(top_words, run_time=1))
        self.wait(2)
        self.play(FadeIn(subwords, 0.5 * DOWN))
        self.wait(6)

        # Circle
        circle = Circle()
        circle.set_stroke(BLUE, 2)
        circle.set_height(5)
        circle.next_to(subwords, DOWN, buff=MED_LARGE_BUFF)
        dots.clear_updaters()
        dots.generate_target(use_deepcopy=True)
        dots.target.set_points([
            circle.pfp(interval.p2n(p))
            for p in dots.get_points()
        ])
        dots.target.set_radius(0)

        circle_words = Text("point on a circle")
        circle_words.move_to(top_words[1], UL)
        circle_words.set_color(BLUE)
        top_words[0].generate_target()
        VGroup(top_words[0].target, circle_words).set_x(0)

        self.play(
            MoveToTarget(dots, run_time=1),
            TransformFromCopy(
                Line(interval.n2p(0), interval.n2p(1)).set_stroke(GREY_B, 1),
                circle,
                run_time=1,
            ),
            FadeOut(interval),
            FadeOut(top_words[1], UP),
            FadeIn(circle_words, UP),
            MoveToTarget(top_words[0]),
        )

        dots.set_points([
            circle.pfp(random.random())
            for x in range(25)
        ])
        dots.set_radius(0.5)
        dots.set_glow_factor(5)
        dots.add_updater(lambda m: m)
        self.add(dots)
        self.play(ShowCreation(dots, rate_func=linear, run_time=2))

        # Sphere
        sphere_words = Text("point on a sphere")
        sphere_words.move_to(circle_words, LEFT)
        sphere_words.set_color(BLUE)

        sphere = Sphere()
        sphere.set_color(BLUE_D)
        sphere.set_opacity(0.5)
        sphere.set_shadow(0.5)
        sphere.set_gloss(0.5)
        sphere.set_reflectiveness(0.5)
        sphere.replace(circle)
        sphere.rotate(0.05 * PI)
        sphere.rotate(0.4 * PI, LEFT)
        sphere.sort_faces_back_to_front()

        mesh = SurfaceMesh(sphere)
        mesh.set_stroke(WHITE, 1, 0.5)

        self.play(
            FadeOut(dots),
            FadeOut(circle, run_time=2),
            ShowCreation(sphere, run_time=2),
            ShowCreation(mesh, lag_ratio=0.1, run_time=2),
            FadeOut(circle_words, UP),
            FadeIn(sphere_words, UP),
        )

        dots.set_points([
            random.choice(sphere.get_points())
            for x in range(100)
        ])
        dots.set_radius(0.25)
        self.play(ShowCreation(dots, run_time=10, rate_func=linear))

        # Ambiguity
        underline = Underline(
            subwords.get_part_by_tex("uniform"),
            buff=0
        )
        underline.set_stroke(YELLOW, width=[0, 2, 2, 2, 0])
        underline.scale(1.25)
        underline.insert_n_curves(50)
        question = Text("Is this well-defined?")
        question.to_edge(RIGHT).shift(UP)
        arrow = Arrow(underline, question, buff=0.1)
        arrow.set_color(YELLOW)

        turn_animation_into_updater(
            ShowCreation(dots, run_time=10, rate_func=lambda t: 0.5 + 0.5 * t),
        )

        dots.clear_updaters()
        dots.set_points(dots.get_points()[12:16])
        dots.set_radius(0.5)
        dots.add_updater(lambda m: m)

        self.play(
            ShowCreation(underline),
            ShowCreation(arrow),
            FadeIn(question, shift=0.5 * DR),
            ShowCreation(dots, run_time=4, rate_func=linear)
        )
        self.wait()


class CoinFlips(Scene):
    CONFIG = {
        "random_seed": 2,
    }

    def construct(self):
        n_rows = 15
        n_cols = 40
        heads = [bool(random.randint(0, 1)) for x in range(n_rows * n_cols)]
        coins = VGroup(*(self.get_coin(h) for h in heads))
        coins.arrange_in_grid(n_rows, n_cols, buff=SMALL_BUFF)
        coins.set_width(FRAME_WIDTH - 0.5)
        coins.to_edge(DOWN)

        eq = Tex(
            "{\\# \\text{Heads} \\over \\# \\text{Flips}} = ",
            "{Num \\over Den}", "=", "0.500",
            isolate={"Num", "Den", "\\# \\text{Heads}"}
        )
        eq.set_color_by_tex("Heads", RED)
        num = Integer(100, edge_to_fix=ORIGIN)
        den = Integer(100, edge_to_fix=ORIGIN)
        dec = DecimalNumber(0.5, num_decimal_places=3, edge_to_fix=LEFT)
        num_i = eq.index_of_part_by_tex("Num")
        den_i = eq.index_of_part_by_tex("Den")
        dec_i = eq.index_of_part_by_tex("0.500")
        num.replace(eq[num_i], dim_to_match=1)
        den.replace(eq[den_i], dim_to_match=1)
        dec.replace(eq[dec_i], dim_to_match=1)
        eq.replace_submobject(num_i, num)
        eq.replace_submobject(den_i, den)
        eq.replace_submobject(dec_i, dec)
        eq.to_edge(UP)

        def update_eq(eq, alpha):
            n_flips = max(int(alpha * len(heads)), 1)
            n_heads = sum(heads[:n_flips])
            num.set_value(n_heads).set_color(RED)
            den.set_value(n_flips)
            dec.set_value(n_heads / n_flips)
            return eq

        for sm in eq:
            sm.add_updater(lambda m: m)

        words = TexText("What does\\\\this approach?", font_size=36)
        words.to_corner(UR)
        arrow = Arrow(words, dec)
        VGroup(words, arrow).set_color(YELLOW)

        self.add(coins, *eq)
        srf = squish_rate_func(smooth, 0.3, 0.4)
        self.play(
            ShowIncreasingSubsets(coins, rate_func=linear),
            UpdateFromAlphaFunc(Mobject(), update_eq, rate_func=linear),
            Write(words, rate_func=srf),
            ShowCreation(arrow, rate_func=srf),
            run_time=18
        )
        self.wait(2)

    def get_coin(self, heads=True, radius=0.25):
        circle = Dot(radius=radius)
        circle.set_fill(RED_E if heads else BLUE_E)
        circle.set_stroke(WHITE, 0.5, 0.5)
        symbol = Tex("H" if heads else "T")
        symbol.set_height(radius)
        symbol.move_to(circle)
        return VGroup(circle, symbol)


class ChordsInSpaceWithCircle(RandomChordScene):
    def construct(self):
        # Introduce chords
        n_lines = 500
        big_circle = Circle(radius=FRAME_WIDTH + FRAME_HEIGHT)
        lines = VGroup(*(
            RadialPoint.get_random_chord(big_circle)
            for x in range(n_lines)
        ))
        lines.set_stroke(WHITE, 0.5, 0.5)

        circle = Circle(radius=2)
        circle.set_stroke(WHITE, 2)

        triangle = Polygon(*(circle.pfp(a) for a in np.arange(0, 1, 1 / 3)))
        triangle.rotate(-PI / 6, about_point=circle.get_center())
        triangle.set_stroke(YELLOW, 3)
        triangle = VGroup(
            triangle.copy().set_stroke(BLACK, 8),
            triangle.copy(),
        )

        self.play(Write(lines, lag_ratio=(1 / n_lines), run_time=6))
        self.wait()

        def get_chords():
            return VGroup(*(
                self.line_to_chord(line, circle)
                for line in lines
            ))

        chords = get_chords()
        chords.save_state(RED)
        chords.set_stroke(WHITE, 1)
        self.play(
            ShowCreation(circle),
            VFadeIn(chords),
        )
        self.wait()
        self.play(ShowCreation(triangle, lag_ratio=0))
        circle.add(triangle)
        self.wait()

        # Show colors
        key = TexText(
            "Blue $\\Rightarrow$ Chord > \\text{Triangle side}\\\\"
            "Red $\\Rightarrow$ Chord < \\text{Triangle side}\\\\",
            tex_to_color_map={
                "Blue": BLUE,
                "Red": RED,
            }
        )
        key.to_edge(UP)
        key.set_backstroke(width=5)
        key[len(key) // 2:].align_to(key[:len(key) // 2], RIGHT)

        self.play(
            Restore(chords),
            FadeIn(key),
        )
        self.wait(2)

        # Move around
        chords.add_updater(lambda m: m.become(get_chords()))
        self.add(chords, circle)
        self.play(circle.animate.to_edge(RIGHT), run_time=6)
        self.play(circle.animate.to_edge(LEFT), run_time=6)
        self.wait()
        self.play(
            circle.animate.set_height(7, about_edge=LEFT),
            FadeOut(key, rate_func=squish_rate_func(smooth, 0, 0.5)),
            run_time=4
        )
        self.play(circle.animate.set_height(4, about_edge=RIGHT), run_time=4)

        # Show center point method
        big_circle = Circle(radius=3.25)
        big_circle.set_stroke(GREY_B, 2)
        big_circle.to_edge(DOWN)

        title = Text("Random center point method")
        title.to_edge(UP, buff=MED_SMALL_BUFF)

        chords.clear_updaters()
        self.play(
            FadeOut(chords),
            FadeOut(lines),
            FadeIn(big_circle),
            FadeIn(title),
            circle.animate.set_height(0.5).to_corner(DR),
        )
        triangle.set_stroke(width=2)

        n_lines = 1000
        lines.become(VGroup(*(
            CenterPoint.get_random_chord(big_circle)
            for x in range(n_lines)
        )))
        lines.set_stroke(WHITE, self.chord_width, self.chord_opacity)
        flash_lines = lines.copy()
        flash_lines.set_stroke(width=3, opacity=1)
        indicators = Group(*map(CenterPoint.get_method_indicator, lines))

        self.play(
            ShowIncreasingSubsets(lines),
            ShowSubmobjectsOneByOne(flash_lines),
            ShowSubmobjectsOneByOne(indicators),
            rate_func=linear,
            run_time=8,
        )
        self.play(
            FadeOut(indicators[-1]),
            FadeOut(flash_lines[-1]),
        )
        self.wait()

        # Put circle back inside
        self.play(
            circle.animate.set_height(2).move_to(big_circle),
            run_time=1
        )
        chords.become(get_chords().set_stroke(opacity=0.7))
        self.add(chords, circle)
        self.play(VFadeIn(chords))
        self.wait()
        chords.add_updater(lambda m: m.become(get_chords().set_stroke(opacity=0.7)))
        self.play(
            circle.animate.move_to(big_circle, RIGHT),
            run_time=10,
            rate_func=there_and_back,
        )
        self.wait()

    def line_to_chord(self, line, circle, stroke_width=1):
        r = circle.get_radius()
        tangent = normalize(line.get_vector())
        normal = rotate_vector(tangent, PI / 2)
        center = circle.get_center()
        d1 = np.dot(line.get_start() - center, normal)
        if d1 > r:
            return VectorizedPoint(center + r * normal)
        d2 = np.sqrt(r**2 - d1**2)
        chord = Line(
            center + d1 * normal - d2 * tangent,
            center + d1 * normal + d2 * tangent,
        )
        chord.set_stroke(
            (BLUE if chord.get_length() > math.sqrt(3) * r else RED),
            width=stroke_width,
        )
        return chord


class TransitiveSymmetries(Scene):
    def construct(self):
        circle = Circle(radius=3)
        circle.set_stroke(GREY_B, 2)
        circle.rotate(PI / 2)

        dots = DotCloud([circle.pfp(a) for a in np.linspace(0, 1, 49)])
        dots.set_glow_factor(5)
        dots.set_radius(0.5)
        dots.set_color(WHITE)
        dots.set_opacity(0.5)

        dot = dots.copy()
        dot.set_points([dots.get_points()[0]])
        dot.set_color(YELLOW)
        dot.set_opacity(1)

        top_words = Text("By acting on any one point...")
        top_words.match_x(dot)
        top_words.to_edge(UP, buff=MED_SMALL_BUFF)
        low_words = Text("...you can map\nto any other")
        low_words.to_edge(RIGHT).shift(2 * DOWN)

        self.add(circle, top_words, dot)
        dots.add_updater(lambda m: m)
        self.play(
            Rotate(
                Group(circle, dot), TAU,
                about_point=circle.get_center(),
            ),
            ShowCreation(
                dots,
                rate_func=lambda a: smooth(a * (1 - 1 / 48) + 1 / 48)
            ),
            Write(low_words, rate_func=squish_rate_func(smooth, 0.3, 0.5)),
            run_time=10,
        )


class NonTransitive(Scene):
    def construct(self):
        circle = Circle(radius=3)
        circle.set_stroke(GREY_B, 2)

        words = TexText(
            "Rotational symmetries do \\emph{not} act\\\\"
            "transitively on the space of all chords",
            tex_to_color_map={"\\emph{not}": RED},
        )
        words.to_edge(UP, MED_SMALL_BUFF)
        circle.next_to(words, DOWN, MED_SMALL_BUFF)

        chord1 = Line(circle.pfp(0.4), circle.pfp(0.5))
        chord1.set_stroke(RED, 3)
        chord1_shadow = chord1.copy().set_stroke(opacity=0.5)
        chord2 = Line(circle.pfp(0.8), circle.pfp(0.25))
        chord2.set_stroke(BLUE, 3)

        left_words = Text("No action on\nthis chord...")
        right_words = Text("...will ever map\nto this chord.")
        left_words.to_edge(LEFT).shift(UP)
        right_words.to_edge(RIGHT).shift(DOWN)

        left_arrow = Arrow(left_words, chord1.get_center(), buff=0.1)
        right_arrow = Arrow(right_words.get_left(), chord2.get_center(), buff=0.1)

        VGroup(left_words, left_arrow).set_color(RED)
        VGroup(right_words, right_arrow).set_color(BLUE)

        chords = VGroup(*(
            RadialPoint.get_random_chord(circle)
            for x in range(100)
        ))
        chords.set_stroke(WHITE, 1, 0.5)

        group = VGroup(circle, chords)

        self.add(words)
        self.play(
            Rotate(group, TAU, about_point=circle.get_center(), run_time=6)
        )
        self.wait()

        self.add(chord1_shadow, chord1)
        self.play(
            FadeOut(chords),
            ShowCreation(chord1_shadow),
            ShowCreation(chord1),
            FadeIn(left_words),
            ShowCreation(left_arrow),
        )
        group = VGroup(circle, chord1)
        self.add(group, left_arrow)
        self.play(
            Rotate(
                group, TAU,
                about_point=circle.get_center(),
            ),
            FadeIn(
                VGroup(right_words, right_arrow, chord2),
                rate_func=squish_rate_func(smooth, 0.3, 0.4),
            ),
            run_time=10,
        )

        # Embed
        self.embed()


class RandomSpherePoint(Scene):
    def construct(self):
        # Setup
        frame = self.camera.frame
        frame.reorient(-20, 70)
        frame.add_updater(lambda m, dt: m.increment_theta(0.015 * dt))

        grid = NumberPlane(
            (-6, 6), (-4, 4),
            background_line_style={
                "stroke_color": GREY_B,
                "stroke_width": 1,
                "stroke_opacity": 0.5,
            },
            axis_config={
                "stroke_width": 1,
            }
        )
        grid.scale(2)
        plane = Rectangle()
        plane.set_stroke(width=0)
        plane.set_gloss(0.5)
        plane.set_fill(GREY_C, 0.5)
        plane.replace(grid, stretch=True)
        plane.add(grid)
        plane.shift(2 * IN)
        self.add(plane)

        sphere_radius = 2
        sphere = Sphere(radius=sphere_radius)
        sphere.set_color(BLUE_E)
        sphere.set_opacity(1.0)
        sphere.move_to(1.5 * OUT)

        mesh = SurfaceMesh(sphere)
        mesh.set_stroke(width=0.5, opacity=0.5)
        mesh.apply_depth_test()
        mesh_shadow = mesh.copy().set_stroke(opacity=0.1)
        mesh_shadow.deactivate_depth_test()

        sphere = Group(sphere, mesh, mesh_shadow)
        self.add(*sphere)

        # Random point
        words = TexText("Choose a random$^{*}$\\\\point on a sphere")
        words.fix_in_frame()
        words.to_corner(UL)

        technicality = TexText(
            "$^{*}$From a distribution that's\\\\",
            "invariant under rotational symmetries.",
            font_size=30
        )
        technicality[1].set_color(YELLOW)
        technicality.fix_in_frame()
        technicality.to_corner(UR)
        technicality.to_edge(RIGHT, buff=MED_SMALL_BUFF)

        def random_sphere_point():
            p = [1, 1, 1]
            while get_norm(p) > 1:
                p = np.random.uniform(-1, 1, 3)
            p = sphere_radius * normalize(p)
            return p + sphere.get_center()

        dot = TrueDot()
        dot.set_radius(0.5)
        dot.set_color(YELLOW)
        dot.set_glow_factor(5)
        dot.move_to(random_sphere_point())

        self.add(words)
        dot.set_radius(0)
        self.play(dot.animate.set_radius(0.5))

        for x in range(11):
            dot.move_to(random_sphere_point())
            dot.set_opacity(random.choice([0.25, 1.0]))
            if x == 5:
                self.play(FadeIn(technicality), run_time=0.5)
            else:
                self.wait(0.5)
        self.play(FadeOut(dot))

        # Little patch
        band = sphere[0].copy()
        band.pointwise_become_partial(sphere[0], 0.7, 0.8)
        patch = band.copy()
        patch.pointwise_become_partial(band, 0.7, 0.75, axis=0)
        patch.set_color(TEAL)
        patch.set_opacity(0.8)
        patch.deactivate_depth_test()

        dot.move_to(patch)

        def show_dots_in_patch(n=2000):
            dots = DotCloud([
                random_sphere_point()
                for x in range(n)
            ])
            dots.set_color(WHITE)
            dots.set_radius(0.02)
            dots.set_glow_factor(1)
            dots.set_opacity([
                0.25 if point[1] > 0 else 1.0
                for point in dots.get_points()
            ])
            dots.add_updater(lambda m: m)
            self.play(ShowCreation(dots, run_time=6, rate_func=linear))
            self.wait()
            self.play(FadeOut(dots))

        self.play(GrowFromCenter(patch))
        self.wait()
        show_dots_in_patch()
        self.wait()

        # patch_copy = patch.copy()
        self.play(*(
            Rotate(
                sm, PI / 3, axis=OUT + RIGHT,
                about_point=sphere.get_center(),
                run_time=2,
            )
            for sm in (*sphere, patch)
        ))
        show_dots_in_patch()
        self.wait()

        # Embed
        self.embed()


class CorrectionInsert(Scene):
    def construct(self):
        # Words
        title = TexText("Looking for an unambiguous ``uniform'' distribution?")
        title.to_edge(UP, buff=0.25)
        kw = dict(
            # font_size=36,
            # color=GREY_A,
            t2c={"compact": YELLOW, "transitively": BLUE},
            t2s={"compact": ITALIC},
        )
        conditions = VGroup(
            Text("1) Find a symmetry which acts transitively", **kw),
            Text("2) The space must be compact", **kw),
        )
        conditions.arrange(DOWN, aligned_edge=LEFT)
        conditions.next_to(title, DOWN, MED_LARGE_BUFF)

        conditions.to_edge(UP)

        # Add shapes
        radius = 1.5

        interval = UnitInterval(width=4)
        interval.add_numbers([0, 0.5, 1.0], num_decimal_places=1)

        circle = Circle(radius=radius)
        circle.set_stroke(GREY_B, 2)

        sphere = Sphere(radius=radius)
        sphere.set_color(BLUE_D, 1)
        sphere.set_opacity(0.5)
        mesh = SurfaceMesh(sphere)
        mesh.set_stroke(width=0.5)
        mesh_shadow = mesh.copy().set_stroke(opacity=0.35)
        sphere_group = Group(mesh_shadow, sphere, mesh)
        sphere_group.rotate(80 * DEGREES, LEFT)
        sphere_group.rotate(2.5 * DEGREES, OUT)
        sphere.sort_faces_back_to_front()

        shapes = Group(interval, circle, sphere_group)
        shapes.arrange(RIGHT, buff=LARGE_BUFF)
        shapes.set_y(-1)

        # Dots
        interval_dots = DotCloud([interval.pfp(random.random()) for n in range(100)])
        circle_dots = DotCloud([circle.pfp(random.random()) for n in range(100)])
        sphere_dots = DotCloud([
            sphere.pfp(math.acos(random.random()) / PI)
            for n in range(300)
        ])

        all_dots = Group(interval_dots, circle_dots, sphere_dots)
        for dots in all_dots:
            dots.set_glow_factor(5)
            dots.set_radius(0.1)
            dots.set_color(YELLOW)
            dots.add_updater(lambda m: m)
        sphere_dots.set_opacity(np.random.choice([1, 0.2], sphere_dots.get_num_points()))
        circle_dots.set_radius(0.15)

        # Animations
        self.add(conditions[0])
        self.add(shapes)
        self.add(*dots)

        self.play(*(ShowCreation(dots, rate_func=linear, run_time=15) for dots in all_dots))
        self.wait(4)
        self.play(Write(conditions[1][:2]))
        self.wait(2)
        self.play(Write(conditions[1][2:]))
        self.wait()

        # Compact
        rects = Rectangle(height=4.75, width=6).get_grid(1, 2, buff=0.5)
        rects.set_stroke(GREY, 1)
        rects.to_edge(DOWN)

        rect_titles = VGroup(Text("Compact"), Text("Not compact"))
        for rt, r, color in zip(rect_titles, rects, [YELLOW, RED]):
            rt.scale(0.7)
            rt.next_to(r, UP, buff=SMALL_BUFF)
            rt.set_color(color)

        compact_spaces = Group(
            Group(interval, interval_dots),
            Group(circle, circle_dots),
            Group(*sphere_group, sphere_dots)
        )
        compact_spaces.generate_target()
        compact_spaces.target.arrange(DOWN)
        compact_spaces.target.set_height(0.9 * rects[0].get_height())
        compact_spaces.target.move_to(rects[0])

        movers = []
        for cs, cst in zip(compact_spaces, compact_spaces.target):
            for m, mt in zip(cs, cst):
                m.generate_target()
                m.target.replace(mt)
                movers.append(m)

        self.play(
            *map(MoveToTarget, movers),
            *map(ShowCreation, rects),
            *map(FadeIn, rect_titles),
        )
        self.wait()

        # Non-compact
        reals = NumberLine((-5, 5), include_tip=True)
        reals.ticks.remove(reals.ticks[0])
        reals.add(reals.copy().flip())
        reals.add_numbers(font_size=20)
        reals.set_width(0.9 * rects[1].get_width())
        reals.next_to(rects[1].get_top(), DOWN, buff=LARGE_BUFF)
        reals_label = Text("Real number line", font_size=30)
        reals_label.next_to(reals, UP, SMALL_BUFF)

        plane = Square(4)
        plane.set_stroke(width=0)
        plane.set_fill(GREY_D, 1)
        plane.set_gloss(0.5)
        plane.set_reflectiveness(0.4)
        arrows = VGroup(*(
            Arrow(v, 2 * v, stroke_width=2)
            for v in compass_directions(8)
        ))
        group = VGroup(plane, arrows)
        group.rotate(80 * DEGREES, LEFT)
        group.set_width(4.5)
        group.next_to(reals, DOWN, buff=1.5)
        group.shift(0.25 * LEFT)

        self.play(
            ShowCreation(reals),
            FadeIn(reals_label),
        )
        self.wait()
        self.play(
            GrowFromCenter(plane),
            *map(ShowCreation, arrows),
        )
        self.wait(3)
