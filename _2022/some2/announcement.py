from manim_imports_ext import *


class IntersectionAndUnion(InteractiveScene):
    def construct(self):
        self.camera.frame.scale(1.1)
        # Title
        title = Text("Educational content creation by individuals", font_size=60)
        title.to_edge(UP)
        title_start = title.select_part("Educational content creation")
        by_ind = title.select_part("by individuals")
        by_ind.set_color(TEAL)
        self.add(title_start)

        self.play(
            Write(by_ind),
            FlashUnder(by_ind, run_time=3, color=TEAL, buff=0.1)
        )

        # Circles
        circles = Circle(radius=2.5).replicate(2)
        circles.arrange(RIGHT, buff=-2.0)
        circles.set_stroke(WHITE, width=0)
        circles[0].set_fill(BLUE_D, 0.5)
        circles[1].set_fill(TEAL_D, 0.5)
        circles.next_to(title, DOWN, LARGE_BUFF)

        circle_titles = VGroup(
            Text("People with\ngood lessons"),
            Text("People with the\ntime and know-how\nto produce videos,\ninteractives, etc.", alignment="LEFT"),
        )
        circle_titles.scale(0.75)
        for ct, circ, vect in zip(circle_titles, circles, [LEFT, RIGHT]):
            ct.next_to(circ, vect, buff=0, aligned_edge=UP)
            ct.match_color(circ)

        self.play(
            LaggedStartMap(Write, circle_titles, lag_ratio=0.25),
            LaggedStartMap(DrawBorderThenFill, circles, lag_ratio=0.25),
        )
        self.wait()

        # Intersection
        inter = Intersection(*circles)
        inter.set_stroke(WHITE, 3)
        inter.set_fill(TEAL, 1.0)
        inter.flip()
        inter.refresh_triangulation()
        self.play(
            DrawBorderThenFill(inter),
            circles.animate.set_fill(opacity=0.25)
        )
        self.wait()

        # Collaboration
        cross = Cross(by_ind)
        new_text = Text("from collaborations", font_size=60)
        new_text.set_color(YELLOW)
        new_text.move_to(by_ind, UL)
        title_start.generate_target()
        VGroup(title_start.target, new_text).set_x(0)

        union = Union(*circles)
        union.set_stroke(WHITE, 3)

        self.play(ShowCreation(cross))
        self.play(
            VGroup(by_ind, cross).animate.scale(0.5).shift(0.75 * DOWN),
            FadeIn(new_text, 0.5 * DOWN),
            MoveToTarget(title_start),
            FadeOut(inter),
            circles.animate.set_fill(opacity=0.7),
            ShowCreation(union)
        )
        self.wait(3)


class WinnerCategories(InteractiveScene):
    def construct(self):
        # Text
        title = Text("In choosing winners, we will select:", font_size=60)
        title.to_edge(UP)
        kw = dict(t2c={"non-video": BLUE, "video": RED, "collaboration": YELLOW})
        cats = VGroup(
            Text("At least one video entry", **kw),
            Text("At least one non-video entry", **kw),
            Text("At least one collaboration", **kw),
        )
        cats[1].select_part("non-video").set_color(BLUE)
        cats.arrange(DOWN, buff=0.75, aligned_edge=LEFT)
        cats.next_to(title, DOWN, LARGE_BUFF)
        details = Text("(See initial substack post for details)", font_size=36)
        details.next_to(cats, DOWN, LARGE_BUFF)
        details.set_fill(GREY_B)

        self.add(title)
        self.play(Write(cats[0], run_time=1))
        self.wait(0.5)
        self.play(TransformMatchingStrings(cats[0].copy(), cats[1]))
        self.wait(0.5)
        self.play(TransformMatchingStrings(cats[1].copy(), cats[2]))
        self.play(FlashAround(cats[2], run_time=2))
        self.wait()

        self.play(FadeIn(details, 0.5 * DOWN))
        self.wait()


class ComplainAboutGithub(TeacherStudentsScene):
    def construct(self):
        pis = self.students
        morty = self.teacher
        self.play(
            pis[1].says(
                "Aren't GitHub issues\nmeant for, like, tracking\nbugs with code?",
                mode="sassy"
            ),
            morty.change("guilty"),
            pis[0].change("confused"),
            pis[2].change("erm"),
        )
        self.wait(2)
        self.play(morty.change("shruggie"))
        self.wait(2)


class Triumverate(PiCreatureScene):
    def construct(self):
        # Introduce team
        morty, prof, artist = self.pi_creatures
        self.clear()
        self.add(morty)

        prof_label = Text("Domain expert", font_size=36).next_to(prof, DOWN)
        artist_label = Text("Artist/animator", font_size=36).next_to(artist, DOWN)
        morty_label = Text("Animations/Writing\n(Whatever is most helpful)", font_size=36)
        morty_label.next_to(morty, UP)

        self.play(
            FadeIn(prof_label, RIGHT),
            FadeIn(prof, RIGHT),
            morty.change("tease", prof.eyes)
        )
        self.play(prof.change("pondering"))
        self.wait(2)

        self.play(
            FadeIn(artist, RIGHT),
            FadeIn(artist_label, RIGHT),
            morty.change("tease", artist.eyes),
        )
        self.play(
            artist.change("hooray", morty.eyes),
            prof.change("thinking", artist.eyes),
        )
        self.wait(2)
        self.play(FadeIn(morty_label, UP))

        # Show video
        logo = Logo()
        logo.set_height(1.5)
        logo.move_to(UP)
        video = VideoIcon()
        video.set_color(RED_E)
        video.match_width(logo)
        video.next_to(logo, DOWN, LARGE_BUFF)

        self.play(
            Write(logo.iris_background, stroke_width=1),
            GrowFromCenter(logo.spike_layers, lag_ratio=0.1, run_time=2),
            Animation(logo.pupil),
            morty.change("thinking", logo),
            prof.change("pondering", logo),
            artist.change("tease", logo)
        )
        self.play(FadeIn(video, DOWN))

        # Footnote
        footnote = Text("""
            *Needless to say, such a collaboration would not be
            considered for winning SoME2, but I'd be happy to
            compensate collaborators here for their time.
        """, font_size=24, alignment="LEFT")
        footnote.to_corner(DR)
        self.play(FadeIn(footnote))
        self.wait(2)

    def create_pi_creatures(self):
        kw = dict(height=2.0)
        morty = Mortimer(**kw)
        prof = PiCreature(color=GREY_D, **kw)
        artist = PiCreature(color=TEAL, **kw)
        morty.to_edge(RIGHT, buff=1.5)
        VGroup(prof, artist).arrange(DOWN, buff=1.5).to_edge(LEFT, buff=1.5)

        return VGroup(morty, prof, artist)


class Winners(InteractiveScene):
    def construct(self):
        # Winners
        winners = PiCreature(color=YELLOW_D).replicate(5)
        for winner in winners:
            winner.body.set_gloss(0.5)
        winners.set_height(1.5)
        winners.arrange_to_fit_width(12)
        winners.set_opacity(0)

        brilliant_logo = SVGMobject("BrilliantLogo").set_height(1)
        brilliant_logo.insert_n_curves(100)
        brilliant_name = Text("Brilliant", font="Simplo Soft Medium")
        brilliant_name.set_height(brilliant_logo.get_height() * 0.5)
        brilliant_name.next_to(brilliant_logo, RIGHT)
        brilliant_logo.add(brilliant_name)
        brilliant_logo.set_fill(WHITE)
        brilliant_logo.set_x(0)
        brilliant_logo.to_edge(UP)

        cash = Text("$1,000").set_color(GREY_A).replicate(5)
        for c, pi in zip(cash, winners):
            c.next_to(pi, DOWN)

        self.play(LaggedStart(*(
            pi.change("hooray").set_opacity(1)
            for pi in winners
        )), lag_ratio=0.2)
        self.wait()
        self.play(Write(brilliant_logo, run_time=1))
        self.play(
            LaggedStart(*(pi.animate.look(DOWN) for pi in winners)),
            LaggedStartMap(FadeIn, cash, shift=DOWN, lag_ratio=0.5)
        )
        self.play(LaggedStartMap(Blink, winners))
        self.wait()


class EndingAnimation(InteractiveScene):
    def construct(self):
        self.add(FullScreenRectangle())
        frame = ScreenRectangle(height=4)
        frame.set_fill(BLACK, 1)
        frame.set_stroke(WHITE, 0)
        boundary = AnimatedBoundary(frame)
        self.add(frame)
        self.add(boundary)

        self.wait(20)


class Thumbnail(InteractiveScene):
    def construct(self):
        # text = Text("More\nMath\nPlease", alignment="LEFT")
        # text.set_height(6)
        # text.to_edge(LEFT, buff=1.0)
        text = Text("Summer\nof\nMath\nExposition", alignment="LEFT")
        text.set_height(FRAME_HEIGHT - 1)
        text.to_edge(LEFT)
        self.add(text)

        randy = Randolph(mode="thinking", color=YELLOW, height=5)
        randy.to_corner(UR, buff=1.0)
        randy.look_at(text)
        self.add(randy)

        two = Text("Round 2")
        two.set_height(1)
        two.set_color(YELLOW)
        two.next_to(text[-1], RIGHT, buff=0.7, aligned_edge=DOWN)
        self.add(two)

