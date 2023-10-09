from manim_imports_ext import *
from PIL import ImageFilter

class PitchShifterWrapper(VideoWrapper):
    animate_boundary = False
    title = "Making a Pitch Shifter by JentGent"


class ThumbnailProgression(InteractiveScene):
    def construct(self):
        # Make grid
        thumbnail_dir = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/SoME3/EntryThumbnails"
        names = [
            "Pixel Art Anti Aliasing",
            "The Math of Saving The Enola Gay",
            "Making a Pitch Shifter",
            "Cayley Graphs and Pretty Things",
            "The Longest Increasing Subsequence",
            "The Matrix Arcade: A Visual Explorable of Matrix Transformations",
            "Watching Neural Networks Learn",
            "Functions are Vectors",
            "The Art of Linear Programming",
            "The Mosaic Problem - How and Why to do Math for Fun",
            "Affording a Planet With Geometry",
            "When CAN'T Math Be Generalized?",
            "Rotation + Translation = Rotation. Animated proof",
            "Rethinking the real line",
            "How did the Ancient Egyptians find this volume without Algebra?",
            "A Subtle Aspect of Circular Motion",
            "Minimal Surfaces & the Calculus of Variations",
            "How does a computer-calculator compute logarithms?",
            "What Happens If We Add Fractions Incorrectly?",
            "Can you guess this shape from its shadows?",
            "Chasing Fixed Points: Greedy Gremlin's Trade-Off",
            "How Computers Use Numbers",
            "Mathematical Magic Mirrorball",
            "The Mathematics of String Art",
            "How Infinity Works",
        ]

        thumbnails = Group(*(
            ImageMobject(os.path.join(thumbnail_dir, name))
            for name in names
        ))
        framed_thumbnails = Group(*(
            Group(
                SurroundingRectangle(tn, buff=0).set_stroke(WHITE, 3),
                tn
            )
            for tn in thumbnails
        ))
        framed_thumbnails.set_height(2)
        framed_thumbnails.arrange_in_grid(h_buff=0.5, v_buff=0.25)
        framed_thumbnails.set_height(FRAME_HEIGHT - 1)

        # Covers
        covers = Group()
        cover_labels = VGroup()
        for thumbnail in thumbnails:
            folder, name = os.path.split(thumbnail.image_path)
            new_folder = guarantee_existence(os.path.join(folder, "blurred"))
            blur_path = os.path.join(new_folder, name)
            pil_img = thumbnail.image
            blurry_pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=75))
            with open(blur_path, 'w') as fp:
                blurry_pil_img.save(fp)
            blurry_image = ImageMobject(blur_path)
            blurry_image.replace(thumbnail)
            cover = Group(
                SurroundingRectangle(blurry_image, buff=0).set_stroke(WHITE, 3),
                blurry_image
            )
            label = Tex("?").move_to(cover)
            cover.add(label)
            covers.add(cover)
            cover_labels.add(label)

        # Shuffle them up and add labels
        indices = list(range(1, 25))
        # random.shuffle(indices)
        indices = [0, *indices]

        for n, cover in enumerate(covers):
            cover.save_state()
            cover.move_to(thumbnails[indices[n]])
            label = Integer(indices[n] + 1, font_size=60)
            label.set_height(0.6 * cover.get_height())
            label.move_to(cover)
            label.set_fill(WHITE)
            label.set_backstroke(BLACK, 7)
            cover[-1].become(label)
        cover_labels.set_stroke(background=True)

        # Show entries
        self.play(LaggedStart(*(
            FadeIn(covers[indices.index(n)])
            for n in range(len(indices))
        ), run_time=4, lag_ratio=0.1))

        self.wait()
        cover_labels.set_backstroke(BLACK, 12)

        # Highlight one
        fave = Text("(My personal\nFavorite)")
        fave.replace(framed_thumbnails[23], dim_to_match=0)
        fave.scale(0.8)
        fave.set_backstroke(BLACK, 3)

        self.play(FadeOut(cover_labels), FadeIn(fave))
        self.wait()
        self.play(FadeOut(fave), FadeIn(cover_labels))

        # Reveal first
        self.reveal(framed_thumbnails[0], names[0], covers[0])
        self.wait()

        # Shuffle remaining
        for cover in covers:
            cover.saved_state[-1].set_opacity(0)

        self.play(LaggedStart(*(
            Restore(cover, path_arc=30 * DEGREES)
            for cover in covers[1:]
        )), lag_ratio=0.01, run_time=2)

        # Reveal one by one
        for image, name, cover in list(zip(framed_thumbnails, names, covers))[1:]:
            self.reveal(image, name, cover)

        # Reveal winners
        winners = [
            "The Mathematics of String Art",
            "Minimal Surfaces & the Calculus of Variations",
            "Rethinking the real line",
            "Pixel Art Anti Aliasing",
            "How Computers Use Numbers",
        ]

        winner_ftns = Group(*(
            framed_thumbnails[names.index(winner)]
            for winner in winners
        ))
        to_fade = Group(*(
            ftn
            for ftn in framed_thumbnails
            if ftn not in winner_ftns
        ))
        winner_ftns.target = winner_ftns.generate_target()
        winner_ftns.target.scale(2)
        winner_ftns.target.arrange_in_grid(2, 3, h_buff=0.5, v_buff=1.5)
        winner_ftns.target[3:].match_x(winner_ftns.target[:3])
        winner_ftns.target.set_width(FRAME_WIDTH - 1)
        winner_ftns.target.to_edge(DOWN)
        for ftn in winner_ftns.target:
            ftn.scale(0.5)

        self.play(FadeOut(to_fade, lag_ratio=0.1, run_time=2))
        self.play(MoveToTarget(winner_ftns, run_time=2))
        self.wait()

        # Add winner names
        winner_names = VGroup(*(
            Text(name, font_size=36)
            for name in [
                "The Mathematics\nof String Art",
                "Minimal Surfaces & \n Calculus of Variations",
                "Rethinking the\nReal Line",
                "Pixel Art\nAnti-aliasing",
                "How Computers\nUse Numbers",
            ]
        ))
        for name, ftn in zip(winner_names, winner_ftns):
            ftn.target = ftn.generate_target()
            ftn.target.scale(2)
            name.next_to(ftn.target, UP)
            self.play(
                FadeIn(name, lag_ratio=0.1),
                MoveToTarget(ftn),
            )
            self.wait()

    def reveal(self, image, name, cover):
        fade_rect = FullScreenRectangle()
        fade_rect.set_fill(BLACK, opacity=0)
        image.save_state()
        title = Text(name, font_size=60)
        title.set_max_width(FRAME_WIDTH - 1)
        title.to_edge(UP)

        self.add(fade_rect, image, cover)
        self.play(FadeOut(cover))
        self.remove(cover)
        self.play(
            fade_rect.animate.set_fill(opacity=1.0),
            image.animate.set_height(6).next_to(title, DOWN),
            FadeInFromPoint(title, image.get_top()),
        )
        self.wait()
        self.play(
            FadeOut(fade_rect),
            Restore(image),
            FadeOutToPoint(title, image.saved_state.get_top()),
        )


class MetaThumbnail(InteractiveScene):
    def construct(self):
        thumbnail_dir = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/SoME3/EntryThumbnails"
        names = [
            "The Mathematics of String Art",
            "Rethinking the real line",
            "The Longest Increasing Subsequence",
            "Chasing Fixed Points: Greedy Gremlin's Trade-Off",
            "How Infinity Works",
            "Pixel Art Anti Aliasing",
            "The Math of Saving The Enola Gay",
            "Making a Pitch Shifter",
            "Functions are Vectors",
            "Cayley Graphs and Pretty Things",
            "The Matrix Arcade: A Visual Explorable of Matrix Transformations",
            "Watching Neural Networks Learn",
            "The Art of Linear Programming",
            "The Mosaic Problem - How and Why to do Math for Fun",
            "Affording a Planet With Geometry",
            "When CAN'T Math Be Generalized?",
            "Rotation + Translation = Rotation. Animated proof",
            "How did the Ancient Egyptians find this volume without Algebra?",
            "A Subtle Aspect of Circular Motion",
            "Minimal Surfaces & the Calculus of Variations",
            "How does a computer-calculator compute logarithms?",
            "What Happens If We Add Fractions Incorrectly?",
            "Can you guess this shape from its shadows?",
            "How Computers Use Numbers",
            "Signed Distance Functions",
            "A Study In Greyscale",
            "Conservative matrix fields",
            "Are Elo Systems Overrated?",
            "The essence of multivariable calculus",
            "The Principle of Least Action",
        ]

        thumbnails = Group(*(
            ImageMobject(os.path.join(thumbnail_dir, name))
            for name in names
        ))
        ftns = Group(*(
            Group(
                SurroundingRectangle(tn, buff=0).set_stroke(GREY_A, 2),
                tn
            )
            for tn in thumbnails
        ))

        # Rows
        groups = Group(
            ftns[:2],
            ftns[2:6],
            ftns[6:14],
            ftns[14:30],
        )
        last = self.frame.get_top()
        for n, group in enumerate(groups):
            group.scale(2**-n)
            group.arrange(RIGHT, buff=0.1)
            group.set_width(FRAME_WIDTH - 0.5)
            group.next_to(last, DOWN, buff=0.2)
            last = group.get_bottom()

        self.add(ftns)
        return

        # Fractal
        last_rect = FullScreenRectangle().set_height(7.25)
        for trip in zip(ftns[0::3], ftns[1::3], ftns[2::3]):
            triad = self.create_triad(*trip, last_rect)
            last_rect = triad[-1]

        self.add(ftns)

    def create_triad(self, im1, im2, im3, to_replace):
        group = Group(im1, im2, im3)
        for im in group:
            im.set_height(4)
        group.add(ScreenRectangle(height=4).set_opacity(0))
        group.arrange_in_grid(buff=0.25)
        group.replace(to_replace)
        return group


class GenerallyGreat(InteractiveScene):
    def construct(self):
        # Set up audience types
        audience_names = [
            "a child",
            "a middle school student",
            "a high school student",
            "an undergraduate in math",
            "an undergraduate in engineering",
            "a college student who doesn't like math",
            "a math Ph.D. student",
            "a professional in a technical field",
            "a person who hasn't thought about math for a decade",
        ]
        audiences = VGroup(*(
            Text(f"for {name}")
            for name in audience_names
        ))
        audiences.scale(0.9)
        audiences.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        audiences.to_corner(UR)
        dots = Tex(R"\vdots")
        dots.next_to(audiences, DOWN)
        dots.shift(3 * LEFT + 0.25 * DOWN)

        self.add(audiences, dots)

        # Quality
        GREEN_YELLOW = interpolate_color(GREEN, YELLOW, 0.25)
        qualities1 = VGroup(*map(Text, ["Good"] * len(audiences))).set_color(GREEN)
        qualities2 = VGroup(*map(Text, ["Pretty good"] * len(audiences))).set_color(YELLOW)
        qualities3 = VGroup(
            Text("Terrible").set_color(RED),
            Text("Bad").set_color(RED),
            Text("Okay").set_color(YELLOW),
            Text("Good").set_color(GREEN_YELLOW),
            Text("Great").set_color(GREEN),
            Text("Bad").set_color(RED),
            Text("Okay").set_color(YELLOW),
            Text("Good").set_color(GREEN_YELLOW),
            Text("Bad").set_color(RED),
        )

        for group in [qualities1, qualities2, qualities3]:
            group.scale(0.9)
            for quality, audience in zip(group, audiences):
                quality.next_to(audience, LEFT, buff=0.25)

        annotations = VGroup(
            Text("Not\nPossible", font_size=90, fill_color=RED, alignment="LEFT"),
            Text("Rare, but\nPossible", font_size=90, fill_color=YELLOW, alignment="LEFT"),
            Text("", font_size=90, fill_color=GREEN, alignment="LEFT"),
        )
        for annotation in annotations:
            annotation.to_corner(UR)

        rect = SurroundingRectangle(VGroup(qualities3[4], audiences[4]))
        rect.set_stroke(GREEN, 2)

        self.play(
            LaggedStartMap(FadeIn, qualities1),
            FadeIn(annotations[0])
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeOut, qualities1, shift=LEFT),
            LaggedStartMap(FadeIn, qualities2, shift=LEFT),
            FadeOut(annotations[0], LEFT),
            FadeIn(annotations[1], LEFT),
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeOut, qualities2, shift=LEFT),
            LaggedStartMap(FadeIn, qualities3, shift=LEFT),
            FadeIn(rect),
            FadeOut(annotations[1], LEFT),
            FadeIn(annotations[2], LEFT),
        )
        self.wait()


class OtherGreatOnes(InteractiveScene):
    def construct(self):
        self.add(FullScreenRectangle())
        names = [
           "Signed Distance Functions",
           "A Study In Greyscale",
           "Conservative matrix fields",
           "Are Elo Systems Overrated?",
           "The essence of multivariable calculus",
           "The Principle of Least Action",
        ]
        thumbnail_dir = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/SoME3/EntryThumbnails"

        pieces = Group(*(
            Group(
                Text(name),
                ImageMobject(
                    os.path.join(thumbnail_dir, name),
                    height=4
                )
            ).arrange(UP)
            for name in names
        ))
        pieces.arrange_in_grid(2, 3, buff=0.5)
        # pieces[3:].match_x(pieces[:3])
        pieces.set_width(FRAME_WIDTH - 1)
        pieces.to_edge(DOWN)

        title = Text("Some others I personally enjoyed a lot", font_size=60)
        title.to_edge(UP)

        self.add(title)
        self.play(LaggedStartMap(FadeIn, pieces, lag_ratio=0.5, run_time=4))
        self.wait()


class ManyThanks(InteractiveScene):
    def construct(self):
        morty = Mortimer()
        words = Text("Many thanks are in order")
        words.next_to(morty, DOWN)

        self.add(morty, words)
        self.play(morty.change("gracious"))
        self.play(Blink(morty))
        self.wait()


class Credits(InteractiveScene):
    def construct(self):
        # Credits
        credit_words = [
            ("Community management and logistics", "James Schloss"),
            ("Web development", "Frédéric Crozatier"),
            ("Guest judging, advising, and funding", "Jane Street"),
        ]
        credits = VGroup(*(
            VGroup(
                Text(role, font_size=36, fill_color=GREY_B),
                Text(name, font_size=48),
            ).arrange(DOWN)
            for role, name in credit_words
        ))
        credits.arrange(DOWN, buff=1.0)
        credits.to_edge(UP)

        logo = SVGMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/SoME3/main/images/janestreet_logo/LOGO_stacked_w_registration_white.svg")
        logo.set_height(1.75)
        logo.move_to(credits[2][1], UP)
        logo.shift(0.5 * DOWN)
        credits[2].replace_submobject(1, logo)

        for credit in credits:
            for part in credit.family_members_with_points():
                part.set_stroke(color=part.get_fill_color(), width=0.5, background=True)
            self.play(FadeIn(credit[0]), Write(credit[1]))
            self.wait()

        # Test


class EndScreen(PatreonEndScreen):
    thanks_words = """
        And, of course, many thanks to channel patrons
    """
