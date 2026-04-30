from manim_imports_ext import *


class Banner(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        self.set_floor_plane("xz")

        n_walks = 30
        line_groups = VGroup()
        all_dots = Group()
        for n in range(n_walks):
            line, dots = self.get_random_walk()
            line.set_stroke(opacity=random.random()**2)
            line_groups.add(line)
            all_dots.add(dots)

        line_groups.set_stroke(flat=True)

        # Animate
        frame.set_field_of_view(0.1 * DEG)
        frame.reorient(45, -35, 0, (4.49, 2.29, -0.07), 5.96)
        self.play(
            LaggedStart(
                (ShowCreation(line, rate_func=linear)
                for line in line_groups),
                lag_ratio=0.1,
            ),
            LaggedStartMap(FadeIn, all_dots, lag_ratio=0.1, time_span=(0, 5)),
            frame.animate.reorient(0, 0, 0, (4.5, 2, 0), 5.96).set_anim_args(time_span=(7, 15)),
            run_time=15,
        )

    def get_random_walk(self, n_steps=25):
        line = VMobject()
        point = ORIGIN.copy()
        line.start_new_path(ORIGIN)

        choices = [UP, UP, UP, DOWN, LEFT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, IN, IN, OUT, OUT]

        all_points = [point.copy()]
        for n in range(n_steps):
            point += random.choice(choices)
            line.add_line_to(point)
            all_points.append(point.copy())

        line.set_stroke([BLUE_E, TEAL], width=3, opacity=0.5)
        dots = DotCloud(all_points)
        dots.set_color(WHITE, 0.1)

        return Group(line, dots)


class WriteNameToUrl(InteractiveScene):
    def construct(self):
        # Test
        full_name = Text("3b1b Talent", font_size=90)
        url = Text("3b1b.co/talent", font_size=90)
        url.shift((full_name["3"].get_y(UP) - url["3"].get_y(UP)) * UP)
        VGroup(full_name, url).set_backstroke(BLACK, 10)

        back_rect = BackgroundRectangle(url)
        back_rect.set_fill(BLACK, 0.9)
        back_rect.set_z_index(-1)

        self.play(Write(full_name, stroke_color=WHITE, run_time=2))
        self.wait()
        self.play(LaggedStart(
            FadeIn(back_rect),
            ReplacementTransform(full_name["3b1b"], url["3b1b"]),
            Write(url[".co/"].set_stroke(behind=True), stroke_color=WHITE),
            ReplacementTransform(full_name["Talent"], url["talent"]),
            lag_ratio=0.25,
        ))
        self.wait()


class TalentPartnerProperties(InteractiveScene):
    def construct(self):
        # Company seeking you
        company = SVGMobject("company_building")
        company.set_height(3)
        company.set_fill(GREY_B)
        company.set_shading(0.5, 0.5)
        company.to_edge(LEFT, buff=0.5)

        laptop = Laptop(width=2.5)
        laptop.rotate(80 * DEG, LEFT)
        laptop.rotate(80 * DEG, DOWN)
        laptop.center().to_edge(RIGHT, buff=1.0)
        randy = Randolph(mode="pondering", height=2.5)
        randy.next_to(laptop, LEFT, buff=0).shift(0.5 * DOWN)
        randy.align_to(company, DOWN)
        randy.look_at(laptop.screen)

        arrow = Arrow(randy.get_corner(DL), company.get_corner(DR), thickness=8, buff=0.5)
        arrow.match_y(randy)

        self.add(company)
        self.add(laptop)
        self.add(randy)
        for n in range(2):
            self.play(Blink(randy))
            self.wait()
        self.play(GrowArrow(arrow))
        self.play(randy.change("well", company))
        self.wait()
        self.play(Blink(randy))
        self.wait()

        # Others at the company
        others = VGroup(
            PiCreature(color=BLUE_D, mode="gracious"),
            PiCreature(color=BLUE_C, mode="tease"),
            PiCreature(color=BLUE_E, mode="well"),
        )
        others.set_height(0.8)
        others.arrange(RIGHT, buff=0.2)
        others.next_to(company, DOWN, buff=0.2)

        hearts = SuitSymbol("hearts").replicate(3)
        hearts.set_height(0.3)
        for heart, pi in zip(hearts, others):
            heart.move_to(pi.get_corner(UR))
            heart.shift(0.1 * RIGHT)
        hearts.shuffle()

        self.play(
            LaggedStartMap(FadeIn, others),
            randy.change("coin_flip_1", others),
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeIn, hearts, shift=0.25 * UP, lag_ratio=0.25),
            *(pi.animate.look_at(company) for pi in others)
        )
        self.play(Blink(others[1]))
        self.play(Blink(others[2]))
        self.wait()

        # Talking with them
        frame = self.frame

        morty = Mortimer().flip()
        morty.set_height(others.get_height() * 1.1)
        morty.next_to(others, LEFT, LARGE_BUFF, aligned_edge=DOWN)

        self.add(morty)
        self.play(
            FadeOut(randy, time_span=(0, 1)),
            frame.animate.reorient(0, 0, 0, (-6.75, -1.77, 0.0), 3.94),
            FadeOut(arrow),
            morty.change("well", others),
            FadeOut(hearts, lag_ratio=0.5),
            *(pi.animate.look_at(morty.eyes) for pi in others),
            run_time=3
        )
        bubble = morty.get_bubble("...", bubble_type=SpeechBubble)
        self.play(LaggedStart(
            Write(bubble),
            morty.change('speaking', others),
            others[0].change("tease"),
            others[1].change("happy"),
            others[1].change("happy"),
            lag_ratio=0.5
        ))
        self.play(Blink(morty))
        self.wait()

        # Response
        content = Text("...", font_size=12)
        new_bubble = others[0].get_bubble(
            content,
            bubble_type=SpeechBubble,
            direction=RIGHT
        )
        content.scale(3)
        self.play(
            others[0].change("hooray"),
            Write(new_bubble),
            FadeOut(bubble),
            morty.change('tease')
        )
        self.wait()
        self.play(Blink(others[0]))
        self.play(
            others[0].change("well", others[1].eyes),
            others[1].animate.look_at(others[0].eyes),
            FadeOut(new_bubble),
        )
        self.play(Blink(others[1]))
        self.wait()


class CareerFairBooths(InteractiveScene):
    def construct(self):
        # Add booths
        fair_words = Text("Virtual Career Fair", font_size=72)
        fair_words.to_edge(UP)

        booth = SVGMobject('booth')
        booth.set_height(1.35)
        booths = booth.get_grid(3, 2, h_buff=3.5, v_buff=0.75)
        booths.next_to(fair_words, DOWN, buff=LARGE_BUFF)
        for booth in booths:
            booth.set_color(random_bright_color(hue_range=(0.1, 0.2)))

        self.add(fair_words, booths)

        # Show job types
        job_types = BulletedList(
            "Senior roles",
            "New careers",
            "Internships",
            "Part-time",
            buff=0.75
        )
        job_types.move_to(booths).shift(0.5 * UP)

        for n, booth in enumerate(booths):
            booth.generate_target()
            booth.target.shift((-1)**(n + 1) * RIGHT)

        self.play(
            LaggedStartMap(FadeIn, job_types, lag_ratio=0.5, run_time=4),
            LaggedStartMap(MoveToTarget, booths, run_time=4)
        )
        self.wait()


class TalentContactCard(InteractiveScene):
    def construct(self):
        words = VGroup(
            VGroup(
                Text("Interested in joining?"),
                Text("Reach out to explore if your company is a good fit")
            ),
            VGroup(
                Text("Find a job through this page?"),
                Text("Let us know, we’d love to hear your story!")
            ),
        )
        for group in words:
            group[0].scale(1.25)
            group[1].scale(0.75)
            group[1].set_color(GREY_B)
            group.arrange(DOWN, buff=0.5)
        words.arrange(DOWN, buff=1.25)

        address = Text("talent@3blue1brown.com", font_size=48)
        address.set_color(BLUE)
        address.next_to(words, UP, LARGE_BUFF)

        # Test
        self.add(words[0][0])
        self.play(
            FadeIn(words[0][1], lag_ratio=0.1, run_time=1.5),
            FadeIn(address, 0.5 * UP),
        )
        self.play(LaggedStart(
            FadeIn(words[1][0]),
            FadeIn(words[1][1], lag_ratio=0.1, run_time=2)
        ))
        self.wait()


class IndustryTypes(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Opportunities in:", font_size=72)
        title.set_backstroke(BLACK, 3)
        title.add_to_back(Underline(title, buff=-0.05).set_stroke(BLUE))
        title.to_corner(UL)
        words = VGroup(
            Text("Education"),
            Text("AI Safety"),
            Text("Agentic code developement"),
            Text("Social engineering defense"),
            Text("Finance"),
            Text("Cryptography research"),
            Text("Software engineering"),
            Text("More..."),
        )
        words.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        words.set_fill(GREY_A)
        words.next_to(title, DOWN, aligned_edge=LEFT, buff=0.5).shift(RIGHT)

        self.frame.set_height(10, about_edge=UL)
        self.add(title, words)
        self.play(LaggedStartMap(FadeIn, words, shift=0.5 * UP, lag_ratio=0.25, run_time=3))
        self.wait()