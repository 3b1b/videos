from manim_imports_ext import *

class CoreDiagrm(InteractiveScene):
    def construct(self):
        # Titles
        frame = self.frame
        titles = VGroup(
            Text("What mathematicians do"),
            Text("Why math is valuable"),
            Text("Where funding comes from"),
        )
        titles.scale(0.75)
        for u, title in zip([-1, 0, 1], titles):
            title.move_to(u * RIGHT * FRAME_WIDTH / 3, UP)
        titles.to_edge(UP, buff=MED_SMALL_BUFF)

        v_lines = Line(DOWN, UP).set_height(2 * FRAME_HEIGHT).replicate(2)
        v_lines.arrange(RIGHT, buff=FRAME_WIDTH / 3)
        v_lines.set_stroke(GREY, 2)

        h_line = Line(LEFT, RIGHT).set_width(2 * FRAME_WIDTH)
        h_line.match_style(v_lines)
        h_line.next_to(titles, DOWN)

        # Stacks
        research_stack = Group(
            self.get_card("Proving theorems", "theorem_proving.png"),
            self.get_card("Asking questions", "question_asking.png"),
            self.get_card("Theory building", "theory_building.png"),
            self.get_card("Applying theory", "applying_theory.png"),
        )
        value_stack = Group(
            self.get_card("Applications\n(today and in the future)", "applications.png"),
            self.get_card("Building problem-solving\nmuscles", "problem_solving.png"),
            self.get_card("Aesthetic joy", "aesthetic_joy.png"),
        )
        funding_stack = Group(
            self.get_card("Teaching", "teaching.png"),
            self.get_card("Public funding", "public_funding.png"),
            self.get_card("Private foundations", "private_funding.png"),
            self.get_card("Industry", "industry.png"),
        )

        stacks = Group(research_stack, value_stack, funding_stack)
        for stack, title in zip(stacks, titles):
            stack.arrange_to_fit_height(6.5)
            stack.next_to(title, DOWN, buff=0.5)

        research_brace = Brace(research_stack, LEFT)
        communication_stack = Group(
            self.get_card("Writing", "writing.png"),
            self.get_card("Speaking", "speaking.png"),
            self.get_card("Outreach", "popularizing.png"),
        )
        communication_stack.arrange_to_fit_height(5.0)
        communication_stack.next_to(research_brace, LEFT)

        # Cover rectangles
        cover_rects = VGroup()
        colors = [BLUE, TEAL, GREEN]
        for stack, color in zip(stacks, colors):
            rect_stack = VGroup()
            for card in stack:
                rect = SurroundingRectangle(card)
                rect.round_corners()
                rect.set_fill(interpolate_color(color, BLACK, 0.5), 1)
                rect.set_stroke(color, 2)
                rect.card = card
                rect.add_updater(lambda m: m.set_height(m.card.get_height() * 1.2).move_to(m.card))
                rect_stack.add(rect)
            cover_rects.add(rect_stack)

        # Add titles and rects
        grid_lines = VGroup(h_line, *v_lines)
        self.play(Write(grid_lines))
        for i in range(3):
            self.play(
                Write(titles[i]),
                LaggedStartMap(GrowFromCenter, cover_rects[i], lag_ratio=0.25),
            )
            self.wait()

        # Reveal them all
        self.frame.set_field_of_view(5 * DEG)
        triptic_rects = FullScreenFadeRectangle().replicate(3)
        triptic_rects.stretch(1 / 3, 0).arrange(RIGHT, buff=0)
        triptic_rects.set_fill(BLACK, 0.0)
        for n, stack, rect_stack in zip(it.count(), stacks, cover_rects):
            title = titles[n]
            title.save_state()
            other_titles = VGroup(t for t in titles if t is not title)
            stack.save_state()
            stack.generate_target()
            stack.target.arrange_in_grid(n_cols=2, h_buff=0.75, v_buff=0.5)
            stack.target.set_height(5.5)
            stack.target.center().to_edge(DOWN)
            if len(stack) == 3:
                stack.target[2].set_x(0)

            self.add(stack, rect_stack)
            self.play(
                title.animate.scale(1.75).center().to_edge(UP),
                FadeOut(other_titles),
                MoveToTarget(stack),
                FadeOut(cover_rects[n + 1:]),
                FadeOut(stacks[:n]),
                FadeOut(grid_lines),
            )
            for card, rect in zip(stack, rect_stack):
                rect.clear_updaters()
                self.play(
                    Rotate(rect, 90 * DEG, axis=RIGHT, about_edge=UP),
                    VFadeOut(rect, time_span=(0.5, 1))
                )
                self.wait()
            self.play(
                Restore(stack),
                Restore(title),
                FadeIn(other_titles),
                FadeIn(grid_lines),
                FadeIn(cover_rects[n + 1:]),
                FadeIn(stacks[:n]),
            )

            if n == 0:
                self.play(
                    self.frame.animate.reorient(0, 0, 0, (-0.73, -0.29, 0.00), 8.97),
                    GrowFromCenter(research_brace),
                    LaggedStartMap(FadeIn, communication_stack, lag_ratio=0.5, run_time=3)
                )
                self.wait()
                self.play(
                    self.frame.animate.to_default_state(),
                    FadeOut(research_brace),
                    FadeOut(communication_stack),
                )

        # Teaching is the biggest
        funding_stack.generate_target()
        funding_stack.target[0].scale(2.0, about_edge=UP)
        funding_stack.target[1:].scale(0.7, about_edge=DOWN)

        self.play(MoveToTarget(funding_stack, run_time=3, rate_func=there_and_back_with_pause))
        self.wait()

        # Value and funding connections
        full_rect = FullScreenFadeRectangle()
        full_rect.set_fill(opacity=0.5)
        private_good_rect = SurroundingRectangle(value_stack[1:], buff=0.25)
        private_good_rect.set_stroke(TEAL, 3)

        self.add(full_rect, value_stack[1:])
        self.play(
            FadeIn(full_rect),
            ShowCreation(private_good_rect),
        )
        self.wait()
        self.play(FadeOut(full_rect), FadeOut(private_good_rect))
        self.show_connections(
            (value_stack[1], funding_stack[0]),
            (value_stack[2], funding_stack[0]),
        )
        self.show_connections(
            (value_stack[0], funding_stack[1]),
            (value_stack[0], funding_stack[2]),
            (value_stack[0], funding_stack[3]),
        )

        # Add communications

        # Proof machine
        self.play(self.frame.animate.set_height(4).move_to(research_stack[0]))
        self.show_ai_dominance(research_stack[0], "ai_theorem_proving.png")
        self.wait()
        self.play(self.frame.animate.to_default_state())
        self.play(LaggedStartMap(WiggleOutThenIn, value_stack))
        self.play(LaggedStartMap(WiggleOutThenIn, funding_stack))
        self.wait()
        self.show_connections(
            (research_stack[2], value_stack[0]),
            (research_stack[3], value_stack[0]),
        )
        self.show_connections(
            (research_stack[0], value_stack[1]),
            (research_stack[1], value_stack[1]),
            (research_stack[2], value_stack[1]),
            (research_stack[3], value_stack[1]),
        )
        self.show_connections(
            (value_stack[1], funding_stack[0]),
        )
        self.show_connections(
            (research_stack[0], value_stack[2]),
        )

        # What happens to communication here
        Group(research_brace, communication_stack).next_to(research_stack, LEFT, SMALL_BUFF)
        self.play(
            frame.animate.reorient(0, 0, 0, (-1.45, -0.53, 0.00), 9.59),
            GrowFromCenter(research_brace),
            LaggedStartMap(FadeIn, communication_stack),
        )
        comm_rect = SurroundingRectangle(communication_stack[0])
        comm_rect.set_stroke(YELLOW, 2)
        self.wait()
        self.play(ShowCreation(comm_rect))
        self.wait()
        self.play(comm_rect.animate.surround(communication_stack[1:], buff=0.25))
        self.wait()
        self.play(
            FadeOut(comm_rect),
            FadeOut(research_brace),
            FadeOut(communication_stack),
            frame.animate.to_default_state(),
        )

        # Conjecture machine
        self.play(self.frame.animate.set_height(4).move_to(research_stack[1]))
        self.show_ai_dominance(research_stack[1], "ai_question_asking.png")
        self.wait()
        self.play(self.frame.animate.to_default_state())

        self.show_connections(
            (research_stack[2], value_stack[0]),
            (research_stack[3], value_stack[0]),
        )
        self.show_connections(
            (research_stack[0], value_stack[1]),
            (research_stack[1], value_stack[1]),
            (research_stack[2], value_stack[1]),
            (research_stack[3], value_stack[1]),
        )

        # Emphasize the aesthetics
        joy_card = value_stack[2]
        joy_card.save_state()
        full_rect.scale(2)
        full_rect.set_fill(opacity=0.75)
        self.add(full_rect, joy_card)
        self.play(
            frame.animate.set_height(6).move_to(joy_card),
            joy_card.animate.scale(1.5),
            FadeIn(full_rect),
        )
        self.wait()
        self.play(
            FadeOut(full_rect),
            frame.animate.to_default_state(),
            Restore(joy_card),
        )

        # Theory machine
        self.play(self.frame.animate.set_height(4).move_to(research_stack[2]))
        self.show_ai_dominance(research_stack[2], "ai_theory_building.png")
        self.wait()
        self.play(self.frame.animate.to_default_state())

        self.show_connections(
            (research_stack[2], value_stack[0]),
            (research_stack[3], value_stack[0]),
        )
        self.show_connections(*(
            (research_card, value_card)
            for research_card in research_stack
            for value_card in value_stack[1:]
        ))
        self.show_connections(*(
            (value_card, funding_stack[0])
            for value_card in value_stack[1:]
        ))

        # Anything you can do I can do better
        self.play(self.frame.animate.set_height(4).move_to(research_stack[3]))
        self.show_ai_dominance(research_stack[3], "ai_applying_theory.png")
        self.wait()
        self.play(self.frame.animate.to_default_state())

        self.show_connections(*(
            (research_card, value_stack[0])
            for research_card in research_stack[2:]
        ))
        self.show_connections(*(
            (value_stack[0], funding_card)
            for funding_card in funding_stack[1:]
        ))
        self.show_connections(*(
            (value_card, funding_stack[1])
            for value_card in value_stack[1:]
        ))
        self.show_connections(
            (value_stack[2], funding_stack[1]),
            (value_stack[2], funding_stack[2]),
        )

    def get_card(self, words, image_name, image_height=1, font_size=24):
        file_path = self.get_image_folder() / image_name
        image = ImageMobject(str(file_path), height=image_height)
        text = Text(words, font_size=font_size)
        text.next_to(image, DOWN, buff=0.1)
        return Group(image, text)

    def show_ai_dominance(self, research_card, image_name):
        file_path = self.get_image_folder() / image_name
        pi_image = research_card[0]
        machine_image = ImageMobject(file_path, height=pi_image.get_height())
        ge = Tex(R">", font_size=72)
        machine_image = ImageMobject(file_path, height=pi_image.get_height())
        pi_image.generate_target()
        pi_image.target.next_to(ge, RIGHT, SMALL_BUFF)
        equation = Group(machine_image, ge, pi_image.target)
        equation.arrange(RIGHT, SMALL_BUFF)
        equation.move_to(pi_image)

        background = SurroundingRectangle(Group(research_card, machine_image, pi_image.target))
        background.round_corners()
        background.set_fill(RED, 0.2).set_stroke(width=0)

        self.add(background, research_card)
        self.play(LaggedStart(
            FadeIn(background),
            FadeIn(machine_image, 0.25 * RIGHT),
            FadeIn(ge, scale=2),
            MoveToTarget(pi_image),
        ))
        research_card.add_to_back(background)
        research_card.add(ge, machine_image)
        self.add(research_card)

    def get_image_folder(self):
        return Path(
            self.file_writer.get_output_file_rootname().parent.parent,
            "Claire Graphics",
        )

    def show_connections(self, *card_pairs, stroke_color=YELLOW, stroke_width=4):
        full_rect = FullScreenFadeRectangle()
        full_rect.set_fill(opacity=0.75)
        connections = VGroup()
        all_cards = set()
        for card1, card2 in card_pairs:
            line = CubicBezier(
                card1[0].get_right(),
                card1[0].get_right() + 2 * RIGHT,
                card2[0].get_left() + 2 * LEFT,
                card2[0].get_left(),
            )
            all_cards.add(card1)
            all_cards.add(card2)
            connections.add(line)
        connections.set_stroke(stroke_color, stroke_width)
        self.add(full_rect, *all_cards)
        self.play(
            FadeIn(full_rect),
            LaggedStartMap(ShowCreation, connections, lag_ratio=0.25)
        )
        self.wait()
        self.play(FadeOut(full_rect), FadeOut(connections))


class FundsToCompute(CoreDiagrm):
    def construct(self):
        # Test
        public_icon = ImageMobject(self.get_image_folder() / "public_funding")
        private_icon = ImageMobject(self.get_image_folder() / "private_funding")
        compute_blob = ImageMobject(self.get_image_folder() / "ai_applying_theory")

        compute_blob.set_height(6).to_edge(RIGHT)
        funding_icons = Group(public_icon, private_icon)
        funding_icons.set_height(2)
        funding_icons.to_edge(LEFT, buff=1)
        arrow = Arrow(funding_icons, compute_blob, thickness=6, fill_color=WHITE)
        compute_blob.shift(0.5 * DL)

        dollars = Text("$").replicate(20).set_fill(GREEN)
        def update_dollars(dollars):
            for dollar in dollars:
                x = dollar.get_x() - arrow.get_x()
                dollar.set_fill(opacity=np.exp(-x**2) * 3)

        dollars.next_to(arrow, UL)
        dollars.add_updater(update_dollars)
        dollars.generate_target()
        dollars.target.next_to(arrow, UR)

        dubious = Text("(seems unlikely)")
        likely = Text("(seems more likely)")
        words = VGroup(dubious, likely)
        words.scale(0.75).set_fill(GREY_B)
        words.next_to(arrow, DOWN, SMALL_BUFF)

        self.add(public_icon, compute_blob)
        self.add(dollars)
        self.play(
            GrowArrow(arrow),
            MoveToTarget(dollars, lag_ratio=0.03, run_time=6)
        )
        self.wait()
        self.play(FadeIn(dubious, 0.25 * DOWN))
        self.wait()
        self.play(LaggedStart(
            FadeOut(public_icon, UP),
            FadeIn(private_icon, UP),
            FadeOut(dubious, 0.5 * UP),
            FadeIn(likely, 0.5 * UP),
            lag_ratio=0.15
        ))
        self.wait()


class TheoremConjectureDefinitionQuote(InteractiveScene):
    def construct(self):
        # Test
        text = """
            “Good mathematicians prove theorems,
            great mathematicians make conjectures,
            the greatest mathematicians write definitions.”
        """
        quote = Text(
            text,
            alignment="LEFT",
            t2c={"prove theorems": BLUE, "make conjectures": TEAL, "write definitions": YELLOW}
        )
        quote.set_width(FRAME_WIDTH - 2)
        quote.move_to(UP)
        parts = VGroup(
            quote[substr.strip()]
            for substr in text.split("\n")
            if substr.strip()
        )
        for n in range(3):
            self.play(
                Write(parts[n]),
                parts[:n].animate.set_opacity(0.5)
            )
            self.wait()


class CardanosFormula(InteractiveScene):
    def construct(self):
        # Test
        equation = Tex(R"""
            x=\sqrt[3]{-\frac{q}{2}+\sqrt{\left(\frac{q}{2}\right)^2+\left(\frac{p}{3}\right)^3}}+\sqrt[3]{-\frac{q}{2}-\sqrt{\left(\frac{q}{2}\right)^2+\left(\frac{p}{3}\right)^3}}
        """)
        equation.to_edge(UP)
        inner_parts = equation[R"\left(\frac{q}{2}\right)^2+\left(\frac{p}{3}\right)^3"]
        inner_part_rects = VGroup(
            BackgroundRectangle(part, buff=0.1).set_fill(RED, 0.5).round_corners()
            for part in inner_parts
        )
        inner_part_rects.set_z_index(-1)
        rect_comments = VGroup(
            TexText(R"Can be $< 0$").next_to(rect, DOWN)
            for rect in inner_part_rects
        )
        rect_comments.set_fill(RED)

        self.play(FadeIn(equation, lag_ratio=0.1))
        self.wait()
        self.play(
            FadeIn(inner_part_rects),
            FadeIn(rect_comments),
        )
        self.wait()


class TitleCardBackdrop1(InteractiveScene):
    title = "How does math work?"

    def construct(self):
        title = Text(self.title, font_size=120)
        title.set_max_width(FRAME_WIDTH - 1)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(2)


class TitleCardBackdrop2(TitleCardBackdrop1):
    title = "A proof machine"


class TitleCardBackdrop3(TitleCardBackdrop1):
    title = "A conjecture machine"


class TitleCardBackdrop4(TitleCardBackdrop1):
    title = "A theory machine"


class TitleCardBackdrop5(TitleCardBackdrop1):
    title = "Anything you can do\nI can do better"


class TitleCardBackdrop6(TitleCardBackdrop1):
    title = "What must change?"


class CartesianCoordinates(InteractiveScene):
    def construct(self):
        # Test
        plane = NumberPlane()
        plane.set_z_index(-1)
        plane.fade(0.5)
        coord_pairs = [(1, 2), (5, 3), (3, -2)]
        points = [plane.c2p(*coords) for coords in coord_pairs]
        dots = VGroup(Dot().move_to(point) for point in points)
        triangle = Polygon(*points)
        triangle.set_stroke(TEAL, 5)
        labels = VGroup(
            Tex(str(coords)).next_to(point, normalize(point - triangle.get_center()), SMALL_BUFF)
            for coords, point in zip(coord_pairs, points)
        )

        circle = Circle(radius=1)
        circle.set_stroke(RED, 3)
        equation = Tex(R"x^2 + y^2 = 1")
        equation.next_to(circle, UL, buff=-0.1)

        words = VGroup(
            Text("Geometry"),
            Tex(R"\Leftrightarrow").rotate(90 * DEG),
            Text("Algebra")
        )
        words.arrange(DOWN)
        words.to_edge(LEFT)
        words[0].save_state()
        words[0].set_y(0)

        self.play(LaggedStart(
            FadeIn(words[0]),
            ShowCreation(triangle),
            ShowCreation(circle),
        ))
        self.wait()
        self.play(
            Restore(words[0]),
            Write(words[1]),
            FadeIn(words[2], 0.5 * DOWN),
            FadeIn(equation),
            Write(plane, lag_ratio=1e-3),
            LaggedStartMap(GrowFromCenter, dots, lag_ratio=0.25),
            LaggedStartMap(FadeIn, labels, shift=0.5 * UP, lag_ratio=0.25),
            run_time=2
        )
        self.wait()


class OrthogonalStatement(InteractiveScene):
    def construct(self):
        # Test
        text = Text("“What you’re saying is\northogonal to my point”")
        self.play(ShowIncreasingSubsets(text))
        self.wait()


class VlogEndScreen(SideScrollEndScreen):
    pass