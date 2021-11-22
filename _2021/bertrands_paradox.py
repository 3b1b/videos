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

    def construct(self):
        circle = self.circle = Circle(radius=self.radius)
        circle.set_stroke(GREY_B, 3)
        circle.to_edge(LEFT)

        chords = self.get_chords(circle)
        flash_chords = chords.copy()
        flash_chords.set_stroke(width=3, opacity=1)
        indicators = Group(*map(self.get_method_indicator, chords))

        title = Text(self.title)
        title.to_corner(UR)

        self.add(circle, title)
        for s, rt in (slice(0, 16), 8), (slice(18, None), self.run_time):
            fraction = self.get_fraction([c.long for c in chords[s]])
            fraction.match_x(title)
            fraction.match_y(circle)
            self.add(fraction)
            self.play(
                ShowIncreasingSubsets(chords[s]),
                ShowSubmobjectsOneByOne(flash_chords[s]),
                ShowSubmobjectsOneByOne(indicators[s]),
                fraction.alpha_update,
                rate_func=linear,
                run_time=rt,
            )
            self.remove(flash_chords)
            self.remove(indicators)
            self.remove(fraction)
        self.add(fraction)
        self.wait()

        # Embed
        self.embed()

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

    def get_chords(self, circle):
        tri_len = np.sqrt(3) * self.radius
        chords = VGroup(*(
            self.get_random_chord(circle)
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

    def get_random_chord(self, circle):
        return NotImplemented

    def get_method_indicator(self, chord):
        return NotImplemented


class PairOfPoints(RandomChordScene):
    title = "Random pair of circle points"

    def get_random_chord(self, circle):
        return Line(
            circle.pfp(random.random()),
            circle.pfp(random.random()),
        )

    def get_method_indicator(self, chord):
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

    def get_random_chord(self, circle):
        x = y = 1
        while x * x + y * y > 1:
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
        return self.chord_from_xy(x, y, circle)

    def chord_from_xy(self, x, y, circle):
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

    def get_method_indicator(self, chord):
        dots = DotCloud([chord.get_center()])
        dots.set_glow_factor(2)
        dots.set_radius(0.25)
        dots.set_color(YELLOW)
        return dots


class RadialPoint(CenterPoint):
    title = "Random point along radial line"

    def get_random_chord(self, circle):
        angle = random.uniform(0, TAU)
        dist = random.uniform(0, 1)
        return self.chord_from_xy(
            dist * math.cos(angle),
            dist * math.sin(angle),
            circle
        )

    def get_method_indicator(self, chord):
        dot = super().get_method_indicator(chord)
        line = Line(self.circle.get_center(), dot.get_center())
        line.set_length(self.radius, about_point=line.get_start())
        line.set_stroke(YELLOW, 1)
        return Group(dot, line)
