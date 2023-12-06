from manim_imports_ext import *



EQUATION_T2C = {
    R"\vec{\textbf{x}}(t)": RED,
    R"\vec{\textbf{F}}(t)": YELLOW,
    R"\vec{\textbf{a}}(t)": YELLOW,
    R"\frac{d^2 \vec{\textbf{x}}}{dt^2}(t) ": YELLOW,
    R"\vec{\textbf{E}}_0": BLUE,
    R"\omega_r": PINK,
    R"\omega_l": TEAL,
}

def get_bordered_thumbnail(path, height=3, stroke_width=3):
    image = ImageMobject(path)
    thumbnail = Group(
        SurroundingRectangle(image, buff=0).set_stroke(WHITE, stroke_width),
        image,
    )
    thumbnail.set_height(height)
    return thumbnail


class HoldUpAlbumCover(TeacherStudentsScene):
    background_color = interpolate_color(GREY_E, BLACK, 0.5)

    def construct(self):
        # Album
        album = ImageMobject("Dark-side-of-the-moon")
        album.set_height(3.5)
        album.move_to(self.hold_up_spot, DOWN)

        self.play(
            self.teacher.change("raise_right_hand", album),
            self.change_students("hooray", "coin_flip_2", "tease", look_at=album)
        )
        self.play(
            FadeIn(album),
        )
        self.wait(2)

        # Enlarge album
        full_rect = FullScreenRectangle()
        full_rect.set_fill(BLACK, 1)
        prism_image = ImageMobject("True-prism")
        true_prism = Group(
            SurroundingRectangle(prism_image, buff=0).set_stroke(WHITE, 10),
            prism_image
        )

        album.save_state()
        album.target = album.generate_target()
        album.target.set_height(5.5)
        album.target.next_to(ORIGIN, RIGHT, buff=0.5)
        true_prism.match_height(album.target).scale(0.98)
        true_prism.next_to(ORIGIN, LEFT, buff=0.5)

        titles = VGroup(
            Text("Genuine simulation"),
            Text("Pink Floyd"),
        )
        for title, mob in zip(titles, [true_prism, album.target]):
            title.move_to(mob)
            title.to_edge(UP, buff=0.5)


        self.add(full_rect, album, true_prism)

        self.play(
            FadeIn(full_rect),
            MoveToTarget(album),
            FadeInFromPoint(true_prism, album.get_center()),
            run_time=2,
        )
        self.remove(self.pi_creatures)
        self.play(FadeIn(titles))
        self.wait()

        # Why is this white
        white_point = np.array([3.25, 0.62, 0])
        red_arrow = Arrow(white_point + UL, white_point, buff=0.1, stroke_width=7)
        red_arrow.set_color(RED)
        why_white = Text("Why is this white?!")
        why_white.next_to(red_arrow, UP, SMALL_BUFF)
        why_white.match_x(album)
        why_white.set_color(RED)

        self.play(
            FadeIn(why_white, lag_ratio=0.1),
            GrowArrow(red_arrow),
        )
        self.wait()

        # Discrete
        roygbiv = Text("ROYGBV")
        roygbiv.rotate(-10 * DEGREES)
        roygbiv.move_to(white_point).shift(1.5 * RIGHT + 0.25 * UP)
        rainbow = VGroup(*(
            Arc(PI, -PI, radius=radius, stroke_width=10)
            for radius in np.linspace(1.5, 2.0, 6)
        ))
        rainbow.reverse_submobjects()
        rainbow.move_to(album, DOWN).shift(0.05 * UP)
        for mob in roygbiv, rainbow:
            mob.set_submobject_colors_by_gradient(
                RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE
            )

        self.play(
            FadeIn(roygbiv, lag_ratio=0.5),
            ShowCreation(rainbow, lag_ratio=0.5),
            run_time=4
        )
        self.wait()

        # Return
        self.play(LaggedStartMap(
            FadeOut, Group(
                why_white, red_arrow,
                roygbiv, rainbow,
                true_prism, *titles,
            ),
            shift=DOWN,
            scale=0.5,
            lag_ratio=0.1
        ))
        self.add(self.pi_creatures, full_rect, album)
        self.play(
            Restore(album),
            FadeOut(full_rect),
        )
        self.wait()

        # Ask question
        question = Text("Why exactly does\nthis work?")
        question.move_to(album, UP)
        arrow = Arrow(question.get_bottom(), 2 * RIGHT + UP)
        arrow.set_stroke(opacity=0)

        self.play(
            FadeOut(album, UP),
            FadeIn(question, UP),
            self.teacher.change("confused", arrow.get_end()),
            self.change_students("erm", "hesitant", "sassy", look_at=arrow.get_end())
        )
        self.play(GrowArrow(arrow))
        self.wait(5)

        # Standard explanation
        title = Text("Standard explanation:", font_size=60)
        title.to_edge(UP, buff=MED_SMALL_BUFF)

        self.play(
            FadeTransform(question, title),
            FadeOut(arrow, LEFT),
            self.teacher.change("raise_right_hand", look_at=3 * UP),
            self.change_students("pondering", "pondering", "pondering", look_at=3 * UP),
        )
        self.wait(5)


class DefineIndexOfRefraction(InteractiveScene):
    def construct(self):
        # Speeds
        ior = "1.52"
        speed1 = Tex(R"\text{Speed} = c \approx 3\times 10^8 \, {\text{m} / \text{s}}")
        speed2 = Tex(R"\text{Speed} \approx c / " + ior)
        speed2[ior].set_color(PINK)
        for speed, x in zip([speed1, speed2], [-1, 1]):
            speed.set_x(x * FRAME_WIDTH / 4)
            speed.set_y(-2)
            self.play(FadeIn(speed, lag_ratio=0.1))
            self.wait()

        # Index of refraction
        num_words = R"\text{Speed in a vacuum}"
        den_words = R"\text{Speed in a glass}"
        equation = Tex(
            fR"{{{num_words} \over {den_words}}} \approx {ior}",
            font_size=36
        )
        equation[den_words].set_color(BLUE)
        equation[ior].set_color(PINK).scale(1.5, about_edge=LEFT)
        equation.to_corner(UL)

        name = TexText("``Index of refraction''")
        name.next_to(equation, DOWN, buff=LARGE_BUFF)
        arrow = Arrow(name, equation[ior], buff=0.2)

        self.play(
            TransformMatchingShapes(speed1["Speed"].copy(), equation[num_words]),
            TransformMatchingShapes(speed2["Speed"].copy(), equation[den_words]),
            Write(equation[R"\over"]),
            Write(equation[R"\approx"]),
            TransformFromCopy(speed2[ior], equation[ior]),
        )
        self.wait()
        self.play(Write(name), GrowArrow(arrow))
        self.wait()


class LightDoesntHaveTreads(TeacherStudentsScene):
    def construct(self):
        # Test
        stds = self.students
        self.play(
            stds[0].change("confused", self.screen),
            stds[1].change("erm", self.screen),
            stds[2].says("But, light doesn't\nhave treads...", mode="sassy", bubble_direction=LEFT),
            self.teacher.change("guilty"),
        )
        self.wait(4)


class LookAtScrolling(InteractiveScene):
    random_seed = 1

    def construct(self):
        morty = Mortimer()
        morty.set_height(3)
        morty.to_edge(RIGHT)

        self.play(morty.change("pondering", 3 * DOWN))
        for _ in range(4):
            self.play(morty.change(
                random.choice(["tease", "pondering"]),
                3 * UP
            ))
            if random.random() < 0.3:
                self.play(Blink(morty))
            else:
                self.wait()
            self.play(morty.animate.look_at(3 * DOWN))
            if random.random() < 0.3:
                self.play(Blink(morty))
            else:
                self.wait()


class WhySlowAtAll(TeacherStudentsScene):
    def construct(self):
        # Test
        stds = self.students
        self.play(
            stds[0].change("pondering", self.screen),
            stds[1].says("Why would light\nslow down at all?", mode="confused", bubble_direction=LEFT, look_at=self.screen),
            stds[2].change("erm", self.screen),
            self.teacher.change("tease"),
        )
        self.wait(2)
        self.play(self.change_students("hesitant", "maybe", "pondering", look_at=self.screen))
        self.wait(3)


class TankAnnotations(InteractiveScene):
    def construct(self):
        # self.add(ImageMobject("TankStill").set_height(FRAME_HEIGHT))
        point = LEFT
        words = Text("First contact!", font_size=24)
        words.next_to(point, DOWN)
        self.play(
            Flash(point),
            FadeIn(words, run_time=0.5)
        )
        self.wait()


class ReactToStandardExplanation(InteractiveScene):
    samples = 4

    def construct(self):
        # Phrases
        phrases = VGroup(
            Text("Light slows down in glass\ndifferent amounts for different colors", alignment="LEFT"),
            TexText(R"Slowing down $\Rightarrow$ bending"),
            TexText(R"Therefore colors refract at different angles"),
        )
        phrases.set_fill(border_width=0)

        for phrase, y in zip(phrases, [1, 0, -1]):
            phrase.set_y(y * FRAME_HEIGHT / 3)
            phrase.set_x(-2.0, LEFT)

        self.add(phrases)
        self.wait()

        # Ask question
        p1 = phrases[0]["Light slows down in glass"]
        p2 = phrases[0]["different amounts for different colors"]
        slows_down = phrases[0]["slows down"]

        rect = SurroundingRectangle(p1, buff=0.05)
        rect.set_stroke(YELLOW, 2)

        why = Text("Why?!")
        why.set_color(YELLOW)
        why.next_to(rect, RIGHT)
        how = Text("How?")
        how.match_style(why)
        how.next_to(slows_down, UP)
        discovered = Text("This should feel discovered")
        discovered.match_style(why)
        discovered.next_to(phrases[0], DOWN, buff=0.5)

        morty = Mortimer(height=2)
        morty.to_edge(RIGHT)

        self.play(
            ShowCreation(rect),
            Write(why),
            p2.animate.set_opacity(0.25),
            phrases[1:].animate.set_opacity(0.25),
            VFadeIn(morty),
            morty.change("maybe", rect),
        )
        self.play(Blink(morty))
        self.wait()
        self.play(
            FadeTransform(why, how),
            rect.animate.surround(slows_down),
            morty.change("angry", rect),
        )
        self.play(Blink(morty))
        self.wait()
        why.next_to(p2, DOWN)
        self.play(
            rect.animate.surround(p2),
            p2.animate.set_opacity(1),
            p1.animate.set_opacity(0.25),
            FadeTransform(how, why),
            morty.change("confused", rect),
        )
        self.play(Blink(morty))
        self.wait()

        self.remove(why)
        self.play(
            rect.animate.surround(phrases[0], buff=0.25),
            phrases[0].animate.set_opacity(1),
            FadeTransformPieces(why.copy(), discovered, remover=True),
            morty.change("tease", rect),
        )
        self.add(discovered)
        self.play(Blink(morty))
        self.wait()

        # Ask for better explanation of bending
        new_why = Text("Better explanation?")
        new_why.match_style(why)
        new_why.next_to(phrases[1], UP)
        self.play(
            phrases[0].animate.set_opacity(0.25),
            phrases[1].animate.set_opacity(1),
            rect.animate.surround(phrases[1]),
            FadeTransform(discovered, new_why),
            morty.change("pondering", phrases[1]),
        )
        self.play(Blink(morty))
        self.wait()

        # Back to central question
        why.next_to(p1, UP)
        self.play(
            rect.animate.surround(p1),
            p1.animate.set_opacity(1),
            phrases[1].animate.set_opacity(0.25),
            FadeTransform(new_why, why),
            morty.change("raise_right_hand")
        )
        self.play(Blink(morty))
        self.wait()


class SnellPuzzle(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Puzzle", font_size=60).set_color(YELLOW)
        underline = Underline(title, stretch_factor=2, stroke_color=YELLOW)
        puzzle = TexText(R"""
            Can you find an equation \\
            relating $\lambda_1$, $\lambda_2$, $\theta_1$ and $\theta_2$?
        """)
        puzzle.next_to(underline, DOWN, buff=0.5)
        group = VGroup(title, underline, puzzle)
        rect = SurroundingRectangle(group, buff=0.25)
        rect.set_fill(BLACK, 1)
        rect.set_stroke(WHITE, 0)
        group.add_to_back(rect)
        group.to_corner(UL)
        self.add(group)


class SnellComparrisonBackdrop(InteractiveScene):
    def construct(self):
        self.add(FullScreenRectangle().set_fill(GREY_E, 1))
        rects = Rectangle(6, 7).replicate(2)
        rects.set_fill(BLACK, 1)
        rects.set_stroke(WHITE, 1)
        rects[0].set_x(-FRAME_WIDTH / 4)
        rects[1].set_x(FRAME_WIDTH / 4)
        self.add(rects)


class KeyPoints(InteractiveScene):
    def construct(self):
        key_points = VGroup(
            Text("Key point #1: Phase kicks"),
            Text("Key point #2: Layer oscillations"),
            Text("Key point #3: Resonance"),
        )
        key_points.set_submobject_colors_by_gradient(BLUE, TEAL, YELLOW)
        key_points.scale(1.25)
        key_points.to_edge(UP, buff=0.25)
        key_points.set_backstroke(BLACK, 4)
        key_points.to_edge(UP)

        self.add(key_points[0])
        self.wait()
        for kp1, kp2 in zip(key_points, key_points[1:]):
            self.play(
                FadeOut(kp1, 0.5 * UP),
                FadeIn(kp2, 0.5 * UP),
            )
            self.wait()


class AsideOnWaveTerminology(TeacherStudentsScene):
    def construct(self):
        # Test
        title = Text("Wave terminology", font_size=60)
        title.to_corner(UR)
        title.to_edge(UP, buff=0.25)
        title.add(Underline(title))
        title.set_color(YELLOW)
        terms = VGroup(
            Text("Phase"),
            Text("Frequency"),
            Text("Wave number"),
            Text("Amplitude"),
        )
        terms.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        terms.next_to(title, DOWN)
        terms.shift(0.5 * LEFT)

        self.add(title, terms)

        self.play(
            self.teacher.change("raise_right_hand", terms),
            self.change_students("pondering", "tease", "hesitant", look_at=terms),
            LaggedStartMap(FadeIn, terms, shift=0.25 * DOWN, lag_ratio=0.5),
        )
        self.wait(3)


class NewQuestion(InteractiveScene):
    def construct(self):
        # Questions
        text_kw = dict(font_size=48)
        questions = VGroup(
            Text("Why does light\nslow down in glass?", **text_kw),
            Text("Why does light's interaction\nwith a layer of glass\nkick back its phase?", **text_kw),
            Text("How much does\nlight slow down?", **text_kw),
            Text("How strong is\nthe phase kick?", **text_kw),
        )
        q1, q2, q3, q4 = questions
        for question, x in zip(questions, [-1, 1, -1, 1]):
            question.set_x(x * FRAME_WIDTH / 4)
            question.set_y(2.75)

        q3.shift(0.5 * RIGHT)
        q4.shift(0.5 * LEFT)

        bad_ap = q2["'"][0]
        good_ap = TexText("'")
        good_ap.replace(bad_ap, dim_to_match=1)
        bad_ap.become(good_ap)

        arrow1 = Arrow(q1, q2)
        arrow2 = Arrow(q3, q4)

        for question in [q1, q3]:
            question.set_opacity(0.5)
            question.save_state()
            question.set_x(0)
            question.set_opacity(1)

        # First pair
        self.play(FadeIn(q1))
        self.wait()
        self.play(
            ShowCreation(arrow1),
            Restore(q1),
            TransformMatchingStrings(q1.copy(), q2, run_time=1),
        )
        self.wait()
        self.play(
            FlashUnder(q2["layer of glass"], color=BLUE),
            q2["layer of glass"].animate.set_color(BLUE)
        )
        self.wait()
        self.play(
            FlashUnder(q2["phase"]),
            q2["phase"].animate.set_color(YELLOW)
        )
        self.wait()

        # Second pair
        self.play(
            FadeOut(q1, UP),
            FadeIn(q3, UP),
            FadeOut(arrow1),
            FadeOut(q2),
        )
        self.wait()
        self.play(
            ShowCreation(arrow2),
            Restore(q3),
            TransformMatchingStrings(q3.copy(), q4, run_time=1),
        )
        self.wait()


class HoldUpLastVideo(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        image = ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/barber_pole/Thumbnails/Part2Thumbnail4.png")
        tn = Group(
            SurroundingRectangle(image, buff=0).set_stroke(WHITE, 3),
            image,
        )
        tn.set_height(3)
        tn.move_to(self.hold_up_spot, DOWN)
        tn.shift(0.25 * DOWN)

        label = Text("Last video")
        arrow = Vector(RIGHT)
        arrow.next_to(tn, LEFT)
        label.next_to(arrow, LEFT)

        self.play(
            self.teacher.change("raise_right_hand", tn),
            FadeIn(tn, UP),
            self.change_students("pondering", "erm", "tease", look_at=tn)
        )
        self.play(
            FadeIn(label, lag_ratio=0.1),
            GrowArrow(arrow)
        )
        self.wait(4)


class WhyDoesAddingCauseAShift(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            stds[2].says("But, why?", look_at=morty.eyes, mode="raise_left_hand", bubble_direction=LEFT),
            self.change_students("pondering", "pondering", look_at=self.screen),
            morty.change("guilty", self.screen),
        )
        self.wait()
        self.play(stds[2].change("confused", self.screen))
        self.wait(2)
        self.play(morty.change("tease", stds[2].eyes))
        self.wait()
        self.play(
            morty.change("raise_right_hand", 3 * UR),
            stds[2].debubble(look_at=3 * UR),
            self.change_students("erm", "hesitant", look_at=3 * UR)
        )
        self.wait(3)


class AnnoateWaveSumPhaseShift(InteractiveScene):
    def construct(self):
        image = ImageMobject("WaveSumPhaseShift")
        image.set_height(FRAME_HEIGHT).center()
        # self.add(image)

        # Test
        wave_len = 4.0
        point1 = (-1.31, 2.33, 0)
        point2 = (-2.3, 0, 0)

        line = DashedLine(UP, DOWN)
        lines1 = line.get_grid(1, 6, h_buff=wave_len / 2)
        lines1.set_height(1.0, stretch=True)
        lines1.set_stroke(YELLOW, 3)
        lines1.move_to(point1 + wave_len * LEFT / 2, LEFT)

        lines2 = lines1.copy()
        lines2.move_to(point2 + wave_len * LEFT / 2, LEFT)

        arrow = Arrow(
            lines2[1].get_top(),
            lines1[1].get_bottom(),
            buff=0.2
        )
        label = TexText(R"$90^\circ$ behind")
        label.next_to(arrow.get_center(), RIGHT)

        self.play(LaggedStartMap(ShowCreation, lines2, lag_ratio=0, run_time=0.5))
        self.wait()
        self.play(
            TransformFromCopy(lines2, lines1),
            GrowArrow(arrow),
            FadeIn(label, lag_ratio=0.1),
        )
        self.wait()


class PreviewQuarterBehindReason(InteractiveScene):
    def construct(self):
        # Plane
        plane = ComplexPlane(
            (-3, 3), (-3, 3),
            background_line_style=dict(stroke_width=2, stroke_color=BLUE),
            faded_line_style=dict(stroke_width=1, stroke_opacity=0.3, stroke_color=BLUE),
        )
        plane.set_height(7)
        self.add(plane)

        # Add little vectors
        z1 = 1 + complex(-4e-3, -0.05)
        amp = 0.08
        zs = np.array([amp * z1**n for n in range(1000)])
        vects = VGroup(*(
            Vector(plane.n2p(z), stroke_width=2)
            for z in zs
        ))
        vects.set_submobject_colors_by_gradient(YELLOW, RED)
        for v1, v2 in zip(vects, vects[1:]):
            v2.shift(v1.get_end() - v2.get_start())

        big_vect = Vector(stroke_width=5)
        big_vect.add_updater(lambda m: m.put_start_and_end_on(
            plane.n2p(0), vects[-1].get_end() if len(vects) > 0 else plane.n2p(amp)
        ))

        self.add(big_vect)
        self.play(
            ShowIncreasingSubsets(vects, rate_func=linear, run_time=8)
        )
        self.wait()


class ElectronLabel(InteractiveScene):
    def construct(self):
        # Test
        label = Text("An electron, say")
        label.set_backstroke(BLACK, 5)
        label.next_to(ORIGIN, UR, buff=LARGE_BUFF)
        arrow = Arrow(label, ORIGIN)
        self.play(
            FadeIn(label, lag_ratio=0.1),
            GrowArrow(arrow)
        )
        self.wait()


class IsThatTrue(TeacherStudentsScene):
    def construct(self):
        # Ask
        stds = self.students
        morty = self.teacher
        law = Tex(R"\vec{\textbf{F}}(t) = -k \vec{\textbf{x}}(t)", t2c=EQUATION_T2C)
        law.move_to(self.hold_up_spot, DOWN)
        morty.change_mode("raise_right_hand")
        self.add(law)
        self.play(
            stds[0].change("pondering", self.screen),
            stds[1].says("Is that accurate?", mode="sassy", look_at=self.screen),
            stds[2].change("erm", self.screen),
        )
        self.wait(2)

        # Answer
        self.play(
            morty.says(TexText(
                R"For small $\vec{\textbf{x}}$, \\ accurate enough",
                t2c={R"$\vec{\textbf{x}}$": RED}
            )),
            self.change_students("thinking", "erm", "happy", look_at=morty.eyes),
            law.animate.to_edge(UP),
        )
        self.wait()

        # Name restoring force
        label = TexText("``Linear restoring force''")
        label.next_to(law, DOWN)
        self.play(
            FadeInFromPoint(label, morty.get_corner(UL)),
            self.change_students(look_at=label),
            morty.debubble("raise_right_hand", look_at=label),
            stds[1].debubble("pondering", look_at=label),
        )
        self.wait(2)

        # True law
        group = VGroup(law, label)
        true_law = Tex(
            R"""
            \text{True force: }
            F = -k x + c_2 x^2 + c_3 x^3 + c_4 x^4 + \cdots
            """,
            t2c={"F": YELLOW, "x": RED},
            font_size=60
        )
        true_law.move_to(UP)
        rect = SurroundingRectangle(true_law["F = -k x"], buff=0.25)
        rect.set_stroke(BLUE, 1)

        self.play(
            group.animate.set_height(1).to_corner(UR),
            FadeIn(true_law, lag_ratio=0.1, run_time=2),
            morty.change("hesitant"),
            self.change_students("pondering", "pondering", "pondering", look_at=true_law)
        )
        self.wait(2)
        self.play(
            ShowCreation(rect),
            true_law[R" + c_2 x^2 + c_3 x^3 + c_4 x^4 + \cdots"].animate.set_opacity(0.2),
            morty.change("tease"),
        )
        self.wait(2)


class VelocityZero(InteractiveScene):
    def construct(self):
        label = Tex(R"(\text{Velocity} = 0)")
        label.set_color(PINK)
        self.play(FadeIn(label, 0.5 * UP))
        self.wait()


class FrequencyVsAngularFrequency(TeacherStudentsScene):
    def construct(self):
        # Correction
        morty = self.teacher
        stds = self.students

        rf_words = Text("Resonant frequency")
        raf_words = Text("Resonant angular frequency")
        omega = Tex(R"\omega_r", font_size=72)
        omega.set_color(PINK)
        omega.move_to(self.hold_up_spot, DOWN)
        for words in rf_words, raf_words:
            words.next_to(omega, UP)

        cross = Cross(rf_words)
        self.add(rf_words, omega)
        morty.change_mode("raise_right_hand")
        self.play(
            ShowCreation(cross),
            self.change_students("sassy", "hesitant", "angry")
        )
        rf_words.add(cross)
        rf_words.target = rf_words.generate_target()
        rf_words.target.next_to(raf_words, UP, LARGE_BUFF)
        rf_words.target.set_fill(opacity=0.5)
        rf_words.target[-1].set_opacity(0)

        self.wait()
        self.play(
            MoveToTarget(rf_words),
            TransformMatchingStrings(
                rf_words.copy(), raf_words,
                run_time=1,
            ),
            self.change_students("pondering", "hesitant", "hesitant")
        )

        # Show cycles
        plane = ComplexPlane(
            (-1, 1), (-1, 1),
            background_line_style=dict(stroke_width=1, stroke_color=BLUE),
            faded_line_style=dict(stroke_width=0.5, stroke_opacity=0.5, stroke_color=BLUE),
        )
        plane.set_height(4)
        plane.to_edge(DOWN)
        circle = Circle(radius=plane.x_axis.get_unit_size())
        circle.set_stroke(TEAL, 2)
        circle.move_to(plane.get_origin())

        f_words = Text("Frequency", font_size=60)
        af_words = Text("Angular Frequency", font_size=60)
        for words, x in [(f_words, -1), (af_words, 1)]:
            words.set_x(x * FRAME_WIDTH  / 4)
            words.to_edge(UP, buff=MED_SMALL_BUFF)

        t2c = {R"\omega": PINK, "f": TEAL}
        f_eq = Tex(R"f = {\text{Cycles} \over \text{Seconds}}", t2c=t2c)
        f_eq.next_to(f_words, DOWN, MED_LARGE_BUFF)
        f_eq["f"].set_color(YELLOW)
        af_eq = Tex(R"\omega = {\text{Radians} \over \text{Seconds}}", t2c=t2c)
        af_eq.next_to(af_words, DOWN, MED_LARGE_BUFF)
        af_eq[R"\omega"].set_color(PINK)

        angle_tracker = ValueTracker()
        angle_tracker.add_updater(lambda m, dt: m.increment_value(0.25 * dt * TAU))
        vect = Vector(RIGHT).set_color(YELLOW)
        vect.add_updater(lambda m: m.put_start_and_end_on(
            plane.n2p(0),
            plane.n2p(np.exp(complex(0, angle_tracker.get_value())))
        ))

        self.add(angle_tracker)
        self.play(
            FadeOut(self.background),
            LaggedStartMap(FadeOut, self.pi_creatures, shift=DOWN),
            ShowCreation(plane, lag_ratio=0.1),
            FadeIn(circle),
            TransformMatchingStrings(rf_words, f_words),
            TransformMatchingStrings(raf_words, af_words),
            FadeTransform(omega, af_eq[R"\omega"]),
            FadeIn(af_eq[1:]),
            FadeIn(f_eq),
            VFadeIn(vect)
        )
        self.wait(4)

        # Show specific value
        f_value_eq = Tex("f = 0.25", t2c=t2c)
        af_value_eq = Tex(R"\omega = 2\pi f = 1.57", t2c=t2c)
        values = VGroup(f_value_eq, af_value_eq)
        values.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        values.next_to(plane, LEFT, MED_LARGE_BUFF)

        self.play(FadeIn(f_value_eq))
        self.wait(8)
        self.play(FadeIn(af_value_eq))

        # Show angle label
        arc_angle_tracker = ValueTracker()
        get_arc_angle = arc_angle_tracker.get_value 
        arc = always_redraw(lambda: Arc(
            0, get_arc_angle(),
            arc_center=plane.n2p(0),
            radius=0.5,
        ))
        arc_angle_label = DecimalNumber(0, unit=R"\text{ Radians}")
        arc_angle_label.add_updater(lambda m: m.set_value(get_arc_angle()))
        arc_angle_label.add_updater(lambda m: m.next_to(arc.get_center(), UR))
        arc_angle_label.add_updater(lambda m: m[4:].shift(SMALL_BUFF * RIGHT))


        self.add(arc)
        self.play(
            arc_angle_tracker.animate.set_value(TAU / 4),
            VFadeIn(arc_angle_label),
        )
        arc.clear_updaters()
        self.wait(10)
        self.play(
            VFadeOut(arc),
            VFadeOut(arc_angle_label),
            FadeOut(values),
        )

        # Show cosine
        plane.add(circle)
        plane.generate_target()
        plane.target.to_edge(LEFT)
        axes = Axes((0, 16), (-1, 1), width=9, height=0.75 * plane.get_height())
        axes.next_to(plane.target, RIGHT)
        graph = axes.get_graph(lambda x: np.cos(TAU * 0.25 * x))
        graph.set_stroke(YELLOW, 2)

        cos_eq = Tex(R"\cos(\omega t) = \cos(2\pi f t)", t2c=t2c)
        cos_eq.next_to(graph, UP)
        omega_arrow = Vector(0.5 * DOWN).set_color(PINK)
        omega_arrow.next_to(cos_eq[R"\omega"], UP, SMALL_BUFF)

        dot = GlowDot()
        dot.add_updater(lambda m: m.move_to(plane.n2p(math.cos(angle_tracker.get_value()))))
        line = Line(UP, DOWN)
        line.set_stroke(YELLOW, 1)
        line.add_updater(lambda m: m.put_start_and_end_on(
            vect.get_end(), dot.get_center()
        ))

        self.play(
            MoveToTarget(plane),
            FadeIn(axes, 2 * LEFT)
        )
        self.wait(4 * (1 - (angle_tracker.get_value() / TAU) % 1))
        self.add(dot, line)
        self.play(
            ShowCreation(graph, rate_func=linear),
            FadeIn(cos_eq, time_span=(0, 1)),
            ShowCreation(omega_arrow, time_span=(4, 5)),
            run_time=16
        )


class UnequalFrequencies(InteractiveScene):
    def construct(self):
        # Test
        # self.add(ImageMobject("WaveEqStill").set_height(FRAME_HEIGHT))

        word_eq = TexText(R"Freq. of light $\ne$ Resonant freq.")
        eq = Tex(R"\omega_l \ne \omega_r")
        word_lhs = word_eq["Freq. of light"]
        word_rhs = word_eq["Resonant freq."]
        word_lhs.set_color(TEAL)
        word_rhs.set_color(PINK)
        eq[R"\omega_l"].set_color(TEAL)
        eq[R"\omega_r"].set_color(PINK)
        equations = VGroup(word_eq, eq)
        equations.arrange(DOWN)
        equations.to_corner(UR)

        oml_point = np.array([-1.2, 2.7, 0])
        arrow = Arrow(word_eq.get_left(), oml_point, path_arc=90 * DEGREES)
        arrow.set_color(TEAL)
        oml = eq[R"\omega_l"]
        oml_copy = oml.copy().move_to(oml_point)

        self.play(
            FadeIn(word_eq),
            ShowCreation(arrow),
        )
        self.wait()
        self.play(ReplacementTransform(oml_copy, oml))
        self.play(Write(eq[R"\ne \omega_r"]))
        self.wait()


class DisectDrivenEq(InteractiveScene):
    def construct(self):
        # Setup
        driven_eq = Tex(
            R"""
                \vec{\textbf{F}}(t) = - k \vec{\textbf{x}}(t)
                + \vec{\textbf{E}}_0 q \cos(\omega_l t)
            """,
            t2c=EQUATION_T2C
        )
        driven_eq.to_corner(UR, buff=0.6)

        lhs = driven_eq[R"\vec{\textbf{F}}(t) = - k \vec{\textbf{x}}(t)"]
        parts = VGroup(*(
            driven_eq[tex]
            for tex in [
                R"\vec{\textbf{E}}_0 q \cos(\omega_l t)",
                R"\omega_l",
                R"\vec{\textbf{E}}_0",
                R"q",
            ]
        ))
        light_part, omega_l, E0, q = parts
        words = VGroup(
            Text("Force from a\nlight wave"),
            Text("Angular frequency\nof the light (color)"),
            Text("Strength of\nthe wave"),
            Text("Charge of the\nelectron"),
        )
        words[0].scale(0.75)
        words[1:].scale(0.75)
        for part, word in zip(parts, words):
            word.next_to(part, DOWN, buff=0.35)
            word.shift_onto_screen()
            word.match_color(part[0][0])
        words[0].set_color(WHITE)

        rect = SurroundingRectangle(light_part)
        rect.set_stroke(TEAL, 2)

        # for expr in driven_eq, words[0]:
        #     bg = BackgroundRectangle(expr, buff=MED_SMALL_BUFF)
        #     bg.set_fill(BLACK, 1)
        #     self.add(bg)

        words[0].set_backstroke(BLACK, 8)

        # Animations
        self.play(FadeIn(lhs, UP))
        self.wait()
        self.play(
            Write(driven_eq["+"]),
            FadeIn(light_part, lag_ratio=0.1),
        )
        self.play(
            ShowCreation(rect),
            Write(words[0]),
        )
        self.wait()

        self.play(
            rect.animate.surround(omega_l),
            FadeTransform(*words[0:2]),
        )
        self.wait()
        self.play(
            rect.animate.surround(E0),
            FadeTransform(*words[1:3]),
        )
        self.wait()
        self.play(
            rect.animate.surround(q),
            FadeTransform(*words[2:4]),
        )
        self.wait()
        self.play(
            rect.animate.surround(light_part),
            FadeTransform(words[3], words[0]),
        )
        self.wait()


class AskAboutAmplitude(InteractiveScene):
    def construct(self):
        # Test
        lines = DashedLine(3 * LEFT, 3 * RIGHT).replicate(2)
        lines.arrange(DOWN, buff=0.8)
        brace = Brace(lines, LEFT)
        text = brace.get_text("What is this\nAmplitude?")
        text.set_backstroke(BLACK, 8)

        self.play(*map(ShowCreation, lines))
        self.play(
            GrowFromCenter(brace),
            Write(text),
        )
        self.wait()


class PiCreaturesReactingToEquation(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.remove(self.background)
        point = morty.get_corner(UL) + 2 * UP + LEFT
        self.play(
            morty.change("raise_left_hand", point),
            self.change_students("pondering", "thinking", "erm", look_at=point)
        )
        self.wait(3)
        self.play(self.change_students("thinking", "pondering", "pondering", look_at=point))
        self.wait(3)
        self.play(
            morty.says("Oh hey, it depends\non the frequency!", mode="hooray"),
            self.change_students("happy", "tease", "thinking", look_at=point)
        )
        self.wait(4)


class HighlightExpression(InteractiveScene):
    def construct(self):
        # Test
        expr = Tex(
            R"""
                \vec{\textbf{x}}(t) = 
                \frac{q ||\vec{\textbf{E}}_0||}{m\left(\omega_r^2-\omega_l^2\right)}
                \cos(\omega_l t)
            """,
            t2c=EQUATION_T2C
        )
        expr.to_edge(UP)
        rect1 = SurroundingRectangle(expr[R"\omega_r^2-\omega_l^2"])
        rect2 = SurroundingRectangle(expr[R"\frac{q ||\vec{\textbf{E}}_0||}{m\left(\omega_r^2-\omega_l^2\right)}"])

        self.add(expr)
        self.play(ShowCreation(rect1))
        self.wait()
        self.play(Transform(rect1, rect2))
        self.wait()
        self.play(FadeOut(rect2))


class Amplitude(InteractiveScene):
    def construct(self):
        # Show equation with annotations
        equation = Tex(R"""
            \text{Layer wiggle amplitude} = 
            \frac{q ||\vec{\textbf{E}}_0||}{m\left(\omega_r^2-\omega_l^2\right)}
        """, t2c=EQUATION_T2C)
        equation.to_corner(UL)

        lf_rect = SurroundingRectangle(equation[R"\omega_l"])
        rf_rect = SurroundingRectangle(equation[R"\omega_r"])

        lf_words = Text("Light frequency", font_size=36)
        rf_words = Text("Resonant frequency", font_size=36)
        lf_words.next_to(lf_rect, DOWN)
        rf_words.next_to(rf_rect, DOWN)

        VGroup(lf_rect, rf_rect).set_stroke(width=2)
        VGroup(lf_rect, lf_words).set_color(EQUATION_T2C[R"\omega_l"])
        VGroup(rf_rect, rf_words).set_color(EQUATION_T2C[R"\omega_r"])

        self.add(equation)
        self.wait()
        self.play(
            ShowCreation(lf_rect),
            FadeIn(lf_words, 0.5 * DOWN),
        )
        self.wait()
        self.play(
            ReplacementTransform(lf_rect, rf_rect),
            FadeTransform(lf_words, rf_words),
        )
        self.wait()
        self.play(FadeOut(rf_rect), FadeOut(rf_words))
        self.wait()


class GuessSteadyState(InteractiveScene):
    def construct(self):
        # Show prompt
        prompt = VGroup(
            Text("Guess that: "),
            Tex(R"""
                \vec{\textbf{x}}(t) =
                A \cos(\omega_l t)
            """, t2c=EQUATION_T2C)
        )
        prompt.arrange(RIGHT, buff=0.5)
        prompt.to_edge(UP)
        A_part = prompt[1]["A"]
        A_part.set_color(GREY_B)
        freq = prompt[1][R"\omega_l"]

        A_arrow = Vector(0.3 * UP, stroke_width=2).next_to(A_part, DOWN, buff=0.1)
        freq_arrow = Vector(0.3 * UP, stroke_width=2).next_to(freq, DOWN, buff=0.1)
        freq_arrow.set_color(TEAL)

        A_word = Text("Solve for\nthis", font_size=24)
        A_word.next_to(A_arrow, DOWN, buff=0.1)
        freq_word = Text("Light frequency", font_size=24)
        freq_word.next_to(freq_arrow, DOWN, buff=0.1)
        freq_word.shift(0.5 * RIGHT)
        freq_word.set_color(TEAL)
        A_word.align_to(freq_word, UP)

        rect = FullScreenRectangle()
        rect.set_fill(BLACK, 1)
        rect.set_height(2, about_edge=UP, stretch=True)

        self.add(rect)
        self.add(prompt)
        self.play(
            GrowArrow(A_arrow),
            FadeIn(A_word),
        )
        self.play(
            GrowArrow(freq_arrow),
            FadeIn(freq_word),
        )
        self.wait()

        # Answer
        amp_tex = R"\frac{q \vec{\textbf{E}}_0}{m\left(\omega_r^2-\omega_l^2\right)}"
        solution = Tex(
            fR"\vec{{\textbf{{x}}}}(t) = {amp_tex} \cos(\omega_l t)",
            t2c=EQUATION_T2C
        )
        solution.move_to(np.array([3.85, 0.65, 0.0]))

        self.play(
            rect.animate.set_height(FRAME_HEIGHT, about_edge=UP, stretch=True).set_anim_args(run_time=2),
            FadeOut(VGroup(
                prompt[0],
                A_arrow, A_word,
                freq_arrow, freq_word
            )),
            TransformMatchingTex(
                prompt[1], solution,
                matched_pairs=[
                    (prompt[1][R"\vec{\textbf{x}}(t)"], solution[R"\vec{\textbf{x}}(t)"]),
                    (A_part, solution[amp_tex])
                ]
            ),
        )
        self.wait()


class StrongerResonanceStrongerPhaseShift(InteractiveScene):
    def construct(self):
        words = TexText(R"Stronger resonance $\rightarrow$ Bigger phase kick", font_size=60)
        words.to_edge(UP)
        self.add(words)


class WhyIsThisTrue(TeacherStudentsScene):
    def construct(self):
        # Test
        stds = self.students
        morty = self.teacher
        self.screen.to_corner(UR)
        self.play(
            stds[2].says("Why is it\na phase shift?", look_at=self.screen),
            stds[1].change("confused", self.screen),
            stds[0].change("pondering", self.screen),
            morty.change("guilty", stds[0].eyes)
        )
        self.wait(2)
        self.play(
            self.change_students("pondering", "pleading", "maybe", look_at=self.screen),
            morty.change("hesitant", self.screen)
        )
        self.wait(3)


class QuestionsFromPatrons(InteractiveScene):
    def construct(self):
        # Questions
        title = Text("Viewer questions about the index of refraction", font_size=60)
        title.to_edge(UP)
        title.add(Underline(title))
        title.set_color(BLUE)

        questions = BulletedList(
            "Why does slowing imply bending?",
            "What causes birefringence?",
            "What's the end of the barber pole explanation?",
            "How can the index of refraction be lower than 1?",
            buff=0.75
        )
        questions.next_to(title, DOWN, buff=1.0)

        self.play(
            FadeIn(title, lag_ratio=0.1),
            LaggedStartMap(FadeIn, questions, shift=0.5 * DOWN, lag_ratio=0.25)
        )
        self.wait()

        # Higlight last three
        last_three = questions[1:]
        highlight_rect = SurroundingRectangle(last_three, buff=0.25)
        highlight_rect.set_stroke(YELLOW, 2)
        grey_rect = highlight_rect.copy().set_stroke(width=0).set_fill(GREY_E, 1)
        back_rect = FullScreenRectangle().set_fill(BLACK, 0.9)
        self.add(back_rect, grey_rect, highlight_rect, last_three)
        self.play(
            FadeIn(back_rect),
            FadeIn(grey_rect),
            DrawBorderThenFill(highlight_rect)
        )
        self.wait()
        self.play(FadeOut(back_rect), FadeOut(grey_rect), FadeOut(highlight_rect))

        # Each question
        background = FullScreenRectangle()
        background.set_fill(interpolate_color(GREY_E, BLACK, 0.5), 1)

        for question in questions:
            question.save_state()
            question.target = question.generate_target()
            question.target[0].scale(0, about_point=question[1].get_left()).set_opacity(0)
            question.target.center().to_edge(UP)

            self.add(background, question)
            self.play(
                FadeIn(background),
                MoveToTarget(question),
            )
            self.wait()
            self.play(
                Restore(question),
                FadeOut(background),
            )


class HoldUpMainVideo(TeacherStudentsScene):
    def construct(self):
        # Show previous video
        thumbnail = get_bordered_thumbnail("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/barber_pole/Thumbnails/PrismThumbnail2.png")
        thumbnail.move_to(self.hold_up_spot, DOWN)

        morty = self.teacher
        self.play(
            morty.change("raise_left_hand"),
            FadeIn(thumbnail, UP),
            self.change_students("happy", "guilty", "erm", look_at=thumbnail),
        )
        self.wait(2)

        # Key points
        key_points = VGroup(
            Text("Key point #1: Phase kicks"),
            Text("Key point #2: Layer oscillations"),
            Text("Key point #3: Resonance"),
        )
        key_points.scale(1.25)
        key_points.set_backstroke(BLACK, 4)
        key_points.to_edge(UP)
        key_points[0].set_fill(BLUE)
        key_points[0].save_state()
        key_points.set_fill(WHITE)
        key_points.scale(0.7)
        key_points.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        key_points.set_y(2).to_edge(RIGHT)
        thumbnail.generate_target()
        thumbnail.target.next_to(key_points, LEFT, buff=0.75)

        self.play(
            MoveToTarget(thumbnail),
            morty.change("raise_right_hand", key_points),
            self.change_students("pondering", "hesitant", "pondering", look_at=thumbnail.target),
            LaggedStartMap(FadeIn, key_points, shift=0.5 * LEFT, lag_ratio=0.5)
        )
        self.wait(2)
        self.play(
            Restore(key_points[0]),
            LaggedStartMap(FadeOut, key_points[1:], shift=DOWN),
            LaggedStartMap(FadeOut, self.pi_creatures, shift=DOWN),
            FadeOut(thumbnail, scale=0.5, shift=DOWN),
        )
        self.wait()


class HoldUpBarberPoleVideos(HoldUpMainVideo):
    def construct(self):
        # Show thumbnails
        folder = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/barber_pole/Thumbnails/"
        thumbnails = Group(*(
            get_bordered_thumbnail(os.path.join(folder, name), stroke_width=2)
            for name in ["CleanThumbnail", "Part2Thumbnail6"]
        ))
        thumbnails.arrange(RIGHT, buff=1.0)
        thumbnails.set_height(2.5)
        thumbnails.move_to(self.hold_up_spot, DR)

        morty = self.teacher
        self.play(
            morty.change("raise_right_hand", thumbnails),
            LaggedStartMap(FadeIn, thumbnails, shift=0.5 * UP, lag_ratio=0.5),
            self.change_students("maybe", "erm", "happy", look_at=thumbnails),
        )
        self.wait()
        self.play(
            self.change_students("confused", "maybe"),
            morty.animate.look_at(self.students)
        )
        self.wait()
        self.play(
            self.students[2].change("tease", thumbnails)
        )
        self.wait(2)


class MissingDetails(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Details I skipped:", font_size=72)
        title.to_edge(UP)
        title.add(Underline(title))
        title.set_color(BLUE)

        details = BulletedList(
            "There will be many resonant frequencies",
            "Our force law should include a damping term",
            "Why the wave produced by a plane of charges is a \\\\" + \
            "quarter-cycle out of phase with the incoming light",
            "Everything related to the magnetic field",
            "Whether accounting for quantum effects makes a difference",
        )
        details.next_to(title, DOWN, buff=0.5)
        details.set_fill(border_width=0)

        self.add(title)
        self.play(LaggedStartMap(FadeIn, details, shift=0.5 * DOWN, lag_ratio=0.35, run_time=3))
        self.wait()
        self.play(details.animate.fade_all_but(1))
        self.wait()


class ArrowOverInldexHalf(InteractiveScene):
    speed = 1.0
    index = 0.68
    bounds = (-3, 3)

    def construct(self):
        arrow = Vector(0.5 * DOWN)
        arrow.next_to(UP, UP)
        arrow.to_edge(LEFT, buff=0)
        x_min, x_max = self.bounds

        def update_arrow(arr, dt):
            step = dt * self.speed
            if x_min < arr.get_x() < x_max:
                step /= self.index
            arr.shift(step * RIGHT)

        arrow.add_updater(update_arrow)
        self.add(arrow)
        self.wait_until(lambda: arrow.get_x() > FRAME_WIDTH / 2 + 2 * arrow.get_width())


class SteadyStateSolutionCircledAmplitude(InteractiveScene):
    def construct(self):
        equation = Tex(
            R"""
                \vec{\textbf{x}}(t) = 
                \cos(\omega_l t) \cdot
                \frac{q ||\vec{\textbf{E}}_0||}{m\left(\omega_r^2-\omega_l^2\right)}
            """,
            t2c=EQUATION_T2C
        )
        equation.set_backstroke(BLACK, 5)

        rect = SurroundingRectangle(equation[R"\frac{q ||\vec{\textbf{E}}_0||}{m\left(\omega_r^2-\omega_l^2\right)}"])
        rect.set_stroke(TEAL)

        self.add(equation)
        self.play(ShowCreation(rect))
        self.wait()


class LimitToContinuity(InteractiveScene):
    def construct(self):
        arrow = Arrow(3 * UP, 3 * DOWN, buff=0, stroke_width=10, stroke_color=YELLOW)
        words = Text("Continuous in\nthe limit", font_size=60)
        words.next_to(arrow, RIGHT, buff=0.5)
        self.play(
            GrowArrow(arrow),
            FadeIn(words),
        )
        self.wait()


class KickForwardOrBackCondition(InteractiveScene):
    def construct(self):
        # First statement
        words = VGroup(
            TexText("If this $< 0$"),
            TexText("Refractive Index $< 1$", font_size=48),
        )
        rect = Rectangle(2, 1)
        rect.next_to(LEFT, LEFT, buff=MED_LARGE_BUFF).set_y(1.9)
        words[0].next_to(rect, UP, MED_SMALL_BUFF)
        words[1].set_y(1.5)
        words[1].set_x(FRAME_WIDTH / 4)

        arrow = Arrow(words[0].get_right(), words[1].get_left())
        arrow.set_stroke(YELLOW, 6)

        self.play(Write(words[0]))
        self.play(
            GrowArrow(arrow),
            FadeIn(words[1], RIGHT)
        )
        self.wait()

        # Freq reference
        freq_ineq = Tex(R"\text{If } \omega_l > \omega_r", t2c=EQUATION_T2C)
        freq_ineq.move_to(words[0])

        lf_rect = SurroundingRectangle(freq_ineq[R"\omega_l"])
        rf_rect = SurroundingRectangle(freq_ineq[R"\omega_r"])

        lf_words = Text("Light frequency", font_size=36)
        rf_words = Text("Resonant frequency", font_size=36)
        lf_words.next_to(lf_rect, UP)
        rf_words.next_to(rf_rect, UP)

        VGroup(lf_rect, rf_rect).set_stroke(width=2)
        VGroup(lf_rect, lf_words).set_color(EQUATION_T2C[R"\omega_l"])
        VGroup(rf_rect, rf_words).set_color(EQUATION_T2C[R"\omega_r"])

        words[0].generate_target()
        words[0].target.next_to(rect, RIGHT, SMALL_BUFF)

        self.play(
            # MoveToTarget(words[0]),
            # arrow.animate.put_start_and_end_on(
            #     words[0].target.get_right() + SMALL_BUFF * RIGHT,
            #     arrow.get_end(),
            # ),
            FadeOut(words[0], 0.5 * UP),
            FadeIn(freq_ineq, 0.5 * UP),
        )
        self.wait()
        self.play(
            FadeIn(lf_rect),
            FadeIn(lf_words, shift=0.25 * UP),
        )
        self.wait()
        self.play(
            ReplacementTransform(lf_rect, rf_rect),
            FadeTransform(lf_words, rf_words),
        )
        self.wait()
        self.play(FadeOut(rf_words), FadeOut(rf_rect))
        self.wait()


class ThisReallyDoesHappen(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.teacher.says("This really\ndoes happen!", mode="surprised"),
        )
        self.play(self.change_students("sassy", "confused", "pleading", look_at=self.screen))
        self.wait(4)


class LookAround(InteractiveScene):
    def construct(self):
        # Test
        circle = Circle(radius=2)
        circle.stretch(1.5, 0)
        arrows = VGroup(*(
            Arrow(v, 2 * v, buff=0, stroke_width=8, stroke_color=GREY_B)
            for alpha in np.arange(0, 1, 1 / 12)
            for v in [circle.pfp(alpha)]
        ))
        words1 = Text("Look\naround", font_size=90)
        words2 = Text("Most things\nare opaque", font_size=72)

        self.add(words1)
        self.play(LaggedStartMap(GrowArrow, arrows, lag_ratio=0.2))
        self.wait()
        self.play(FadeTransform(words1, words2))
        self.wait()


class CausalInfluence(InteractiveScene):
    def construct(self):
        words = TexText(R"No causal influence can \\ propagate faster than $c$", font_size=72)
        c = words["$c$"]
        c.set_color(YELLOW)
        rect = SurroundingRectangle(c)
        arrow = Vector(UP)
        arrow.set_color(YELLOW)
        arrow.next_to(rect, DOWN)

        self.add(words)
        self.wait()
        self.play(
            GrowArrow(arrow),
            ShowCreation(rect),
            FlashAround(c),
        )
        self.wait()


class RightLeftArrow(InteractiveScene):
    def construct(self):
        vect = Vector(DOWN)
        vect.to_edge(RIGHT)
        self.play(vect.animate.to_edge(LEFT), run_time=24, rate_func=linear)


class SameRotationRate(InteractiveScene):
    def construct(self):
        # Test
        words = Text("Same rate\nof rotation", font_size=60)
        words.to_edge(RIGHT)
        arrows = VGroup(*(
            Arrow(words.get_corner(corner), y * UP + 1.25 * RIGHT, buff=0.2)
            for y, corner in zip([2.75, 0, -2.75], [UL, LEFT, DL])
        ))
        self.play(
            Write(words),
            LaggedStartMap(GrowArrow, arrows)
        )
        self.wait()


class IOREndScreen(PatreonEndScreen):
    pass
