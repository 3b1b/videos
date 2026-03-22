from manim_imports_ext import *


class WelchLabsIntroCard(InteractiveScene):
    title_words = "This is a guest video by Welch Labs"
    subtitle_words = ""
    logo_file = "WelchLabsLogo"
    banner_file = "WelchLabsBanner"
    bottom_words = "This is part of a series of guest videos, see the end for more details"

    def construct(self):
        # Add title and banners
        group = Group(
            VGroup(
                Text(self.title_words, font_size=48),
                Text(self.subtitle_words, font_size=36, fill_color=GREY_A),
            ).arrange(DOWN),
            ImageMobject(self.banner_file).set_width(0.7 * FRAME_WIDTH),
            Text(self.bottom_words, font_size=36),
        )
        group.arrange(DOWN, buff=MED_LARGE_BUFF)
        group[2].scale(1.1, about_edge=DOWN)
        group[-1].to_edge(DOWN, buff=MED_LARGE_BUFF)

        self.play(LaggedStart(
            Write(group[0], run_time=1, lag_ratio=5e-2),
            FadeIn(group[1], scale=1.25),
            FadeIn(group[2], lag_ratio=0.01),
            lag_ratio=0.05
        ))
        self.wait(2)
        self.play(
            LaggedStartMap(FadeOut, group, shift=1.0 * DOWN, lag_ratio=0.1)
        )


class Aleph0IntroCard(WelchLabsIntroCard):
    name = "Aleph 0"
    logo_file = "Aleph0Logo"
    banner_file = "Aleph0Banner"
    bottom_words = "This is the 2nd of 5 of guest videos this summer while I'm on leave"


class VilasIntroCard(WelchLabsIntroCard):
    title_words = "This is a guest video by Vilas Winstein"
    subtitle_words = "(a PhD candidate in probability at UC Berkeley)"
    banner_file = "vilas_winstein"
    logo_file = "SpectralCollectiveLogo"
    bottom_words = "This is the 3rd of 5 of guest videos this summer while I am on leave"


class SubManifolds(InteractiveScene):
    def construct(self):
        # Set up spaces
        blob = Circle(radius=1).stretch(2.4, 0, about_edge=LEFT)
        blob.shift(0.5 * RIGHT)
        blob.set_fill(BLUE, 0.5)
        blob.set_stroke(BLUE, 1)
        blobs = VGroup()
        for angle in np.arange(0, TAU, TAU / 5):
            new_blob = blob.copy()
            new_blob.scale(random.uniform(0.5, 1), about_edge=LEFT)
            new_blob.stretch(random.uniform(0.9, 1.3), 1)
            new_blob.rotate(angle + random.uniform(-0.5, 0.5), about_point=ORIGIN)
            new_blob.set_color(random_bright_color())
            blobs.add(new_blob)

        middle = Intersection(*blobs)
        middle.match_style(blob)
        middle.reverse_points()

        big_circle = Circle(radius=3.8)
        big_circle.stretch(1.5, 0)
        big_circle.set_fill(TEAL, 0.2)
        big_circle.set_stroke(TEAL, 1)

        # Label all videos
        all_videos_text = Text("Space of\nall videos")
        all_videos_text.next_to(big_circle.pfp(0.1), UR, SMALL_BUFF)
        self.add(all_videos_text)
        self.add(big_circle)

        # Show middle blob
        prompt_words = TexText(R"""
            Videos consistent with \\
            ``An astronaut on the moon\\
            riding a horse that turns\\
            into a giant cat''
        """, alignment="", font_size=24)
        prompt_words[len("Videos consistent with".replace(" ", "")):].set_color(BLUE_B)
        prompt_words.next_to(big_circle.pfp(3 / 8), DR, SMALL_BUFF)
        prompt_words.set_backstroke(BLACK, 5)
        arrow = Arrow(prompt_words.get_right(), middle.get_top(), buff=0.1, path_arc=-60 * DEG)

        self.play(LaggedStart(
            FadeIn(prompt_words, lag_ratio=0.1),
            Write(arrow),
            TransformFromCopy(big_circle, middle, run_time=2),
        ))
        self.wait()

        # No training data here
        no_training_words = Text("No training\ndata here", font_size=24)
        no_training_words.next_to(middle, RIGHT, SMALL_BUFF)
        no_training_words.set_z_index(1)
        no_training_words.set_backstroke(BLACK, 2)

        def get_training_data_examples(n_samples, min_scale=0.2):
            training_data = DotCloud(np.array([
                big_circle.pfp(random.random()) * random.uniform(min_scale, 1)
                for n in range(n_samples)
            ]))
            training_data.set_radius(0.04)
            training_data.make_3d()
            training_data.set_z_index(-1)
            return training_data

        training_data = get_training_data_examples(100)

        self.play(
            ShowCreation(training_data, run_time=3),
            Write(no_training_words, run_time=1),
        )
        self.wait()

        # Show other blobs
        blobs.set_fill(opacity=0.25)
        more_data = get_training_data_examples(1000)

        blob_words = VGroup(
            Text(word, font_size=24) for word in [
                "cats",
                "horses",
                "astronauts",
                "transformation",
                "on the\nmoon",
            ]
        )
        for word, blob in zip(blob_words, blobs):
            word.move_to(blob.get_center() * 1.5)
        blob_words.set_backstroke(BLACK, 2)

        self.play(
            ShowCreation(more_data, run_time=6),
            LaggedStartMap(Write, blobs),
            LaggedStartMap(FadeIn, blob_words),
            no_training_words.animate.scale(0.5).move_to(middle).set_backstroke(BLACK, 0).set_fill(BLACK),
            prompt_words.animate.set_backstroke(BLACK, 10),
        )
        self.wait()


class ComposingFeatures(InteractiveScene):
    def construct(self):
        pass


class IMOGoldOrganizations(InteractiveScene):
    def construct(self):
        # Test
        orgs = VGroup(
            Text("Google DeepMind"),
            Text("OpenAI"),
            Text("Harmonic"),
            Text("ByteDance"),
        )
        orgs.scale(1.5)
        orgs.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        orgs.to_edge(LEFT, buff=1)
        top_brace = Brace(orgs[:2], RIGHT)
        low_brace = Brace(orgs[2:], RIGHT)
        top_text = top_brace.get_text("Natural language", buff=0.5).set_color(BLUE_A)
        low_text = low_brace.get_text("Lean", buff=0.5).set_color(YELLOW_B)

        self.play(LaggedStartMap(FadeIn, orgs, shift=DOWN, lag_ratio=0.15))
        self.wait()
        self.play(LaggedStart(
            GrowFromCenter(top_brace),
            FadeIn(top_text, RIGHT),
            orgs[:2].animate.match_color(top_text),
            GrowFromCenter(low_brace),
            FadeIn(low_text, RIGHT),
            orgs[2:].animate.match_color(low_text),
        ))
        self.wait()


class AlephGeometryEndScreen(PatreonEndScreen):
    pass


class PhaseChangeEndScreen(PatreonEndScreen):
    def construct(self):
        # Attribution
        # background_color = interpolate_color("#FDF6E3", BLACK, 0)
        background_color = BLACK
        text_color = WHITE

        background_rect = FullScreenRectangle(fill_color=background_color)
        background_rect.set_z_index(-2)
        self.add(background_rect)

        elements = VGroup(
            ScreenRectangle(height=3),
            Circle(radius=1),
        )
        elements.arrange(RIGHT, buff=0.5)
        elements.set_width(6.5)
        elements.to_edge(RIGHT, buff=1.0)
        elements.set_y(1)
        elements.set_stroke(text_color, 1)
        # elements.insert_n_curves(100)

        attribution = Text("This guest video was\nproduced by Vilas Winstein", font_size=48, fill_color=text_color)
        watch = Text("Watch part 2 now on Spectral Collective", font_size=36).set_fill(text_color, 0.95)
        attribution.next_to(elements, DOWN, buff=MED_LARGE_BUFF)
        watch.next_to(elements, UP, buff=MED_SMALL_BUFF)

        self.add(elements, attribution, watch)

        # Patron scroll
        v_line = Line(DOWN, UP).set_height(FRAME_HEIGHT)
        v_line.next_to(elements, LEFT, LARGE_BUFF)
        v_line.set_y(0)
        v_line.set_stroke(text_color, 1)

        thanks = Text(
            "This channel is funded via Patreon\nSpecial thanks to these supporters",
            alignment="LEFT",
            font_size=32,
            fill_color=text_color
        )
        thanks.move_to(midpoint(v_line.get_center(), LEFT_SIDE))
        thanks.to_edge(UP)

        solid_rect = Square(side_length=8)
        solid_rect.set_fill(background_color, 1).set_stroke(text_color, 1)
        solid_rect.next_to(v_line, LEFT, 0)
        solid_rect.align_to(thanks, DOWN).shift(MED_SMALL_BUFF * DOWN)

        names = VGroup(map(Text, self.get_names()))
        names.scale(0.5)
        for name in names:
            name.set_width(min(name.get_width(), 2.0))
        names.set_fill(text_color)
        names.arrange_in_grid(n_cols=2, aligned_edge=LEFT)
        names.next_to(solid_rect, DOWN, buff=7).to_edge(LEFT)
        names.set_z_index(-1)

        self.add(solid_rect)
        self.add(v_line)
        self.add(thanks)
        self.add(names)
        self.play(
            names.animate.to_edge(DOWN, buff=1.5).set_anim_args(run_time=25, rate_func=linear),
            LaggedStartMap(
                VShowPassingFlash,
                elements.copy().set_stroke(WHITE, 5).insert_n_curves(1000),
                time_width=2,
                run_time=3
            ),
            FadeIn(elements, time_span=(1, 2))
        )


class EuclidEndScreen(SideScrollEndScreen):
    scroll_time = 30


class SeriesOfFiveVideos(InteractiveScene):
    def construct(self):
        # Add images
        self.add(FullScreenRectangle(fill_color=GREY_E))

        pure_images = Group(
            ImageMobject(filename)
            for filename in [
                "diffusion_TN",
                "alpha_geometry_TN",
                "phase_change_TN",
                "incomplete_cubes_TN",
                "euclid_TN",
            ]
        )
        borders = VGroup(SurroundingRectangle(image, buff=0) for image in pure_images)
        borders.set_stroke(WHITE, 3)

        images = Group(
            Group(border, image)
            for border, image in zip(borders, pure_images)
        )

        images.set_width(4)
        images.arrange_in_grid(n_cols=3, buff=1)
        images[3:].match_x(images[:3]).shift(MED_LARGE_BUFF * DOWN)
        images.set_width(FRAME_WIDTH - 1)

        # Add names
        names = VGroup(
            Text(f"Guest video by {text}", font_size=30)
            for text in [
                "Welch Labs",
                "Aleph0",
                "Vilas Winstein",
                "Paul Dancstep",
                "Ben Syversen",
            ]
        )
        for name, image in zip(names, images):
            name.next_to(image, UP, MED_SMALL_BUFF)
            image.add(name)

        # Animate in
        frame = self.frame
        frame.set_height(4).move_to(images[-1])
        self.add(images[-1])
        self.wait()
        self.play(
            frame.animate.to_default_state(),
            LaggedStartMap(FadeIn, images[3::-1], lag_ratio=0.25),
            run_time=3,
        )
        self.wait()

        # Swap out for topics
        titles = VGroup(
            Text("Diffusion models"),
            Text("AlphaGeometry"),
            Text("Statistical Mechanics"),
            Text("Group theory"),
            Text("Euclid"),
        )
        titles.scale(0.8)
        for title, name, image in zip(titles, names, images):
            title.move_to(name, DOWN)
            self.play(
                LaggedStart(FadeIn(title, 0.25 * UP), FadeOut(name, 0.25 * UP), lag_ratio=0.2),
                pure_images.animate.set_opacity(0.25),
                borders.animate.set_stroke(opacity=0.25),
            )
        self.wait()