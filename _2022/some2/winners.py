from manim_imports_ext import *
import pandas as pd
import requests

from _2021.some1_winners import AllVideosOrdered


DOC_DIR = os.path.join(get_output_dir(), "2022/some2/winners/")


def get_entry_tile():
    rect = ScreenRectangle(height=1)
    rect.set_fill(GREY_D, 1)
    rect.set_stroke(WHITE, 1)
    rect.set_gloss(0)
    qmarks = Text("???")
    qmarks.move_to(rect)
    return Group(rect, qmarks)


def save_thumbnails(slugs):
    for slug in slugs:
        url = f"https://img.youtube.com/vi/{slug}/maxresdefault.jpg"
        img_data = requests.get(url).content
        with open(os.path.join(DOC_DIR, "thumbnails", f'{slug}.jpg'), 'wb') as handler:
            handler.write(img_data)


def url_to_image(self, url):
    if url.startswith("https://youtu.be/"):
        name = url.split("/")[-1]  # Get slug
    else:
        name = [p for p in url.split("/") if p][-1].split(".")[0]

    return ImageMobject(os.path.join(DOC_DIR, f"thumbnails/{name}"))


class ExampleChannels(InteractiveScene):
    def construct(self):
        # Channel titles
        self.add(FullScreenFadeRectangle(fill_color=GREY_E, fill_opacity=1))
        channels = [
            ("Eric Rowland", "8.9k", "3gyHKCDq1YA", 423816),
            ("A Well-Rested Dog", "15k", "5M2RWtD4EzI", 381953),
            ("polylog", "10k", "-64UT8yikng", 225508),
            ("A Bit Wiser", "6.5k", "_DaTsI42Wvo", 104138),
            ("SithDev", "7.3k", "6JxvKfSV9Ns", 134753),
            ("Another Roof", "11k", "dKtsjQtigag", 136517),
            ("Marc Evanstein", "3.1k", "8x374slJGuo", 161853),
            ("Armando Arredondo", "4.6k", "_lb1AxwXLaM", 272279),
            ("HexagonVideos", "2.8k", "-vxW42R47bc", 219047),
            ("SackVideo", "12k", "UJp4q2D2Nh0", 233851),
            ("Morphocular", "31k", "2dwQUUDt5Is", 281021),
        ]
        # save_thumbnails([c[2] for c in channels])

        tile = ScreenRectangle()
        tile.stretch(0.8, 0)
        tile.set_stroke(WHITE, 0)
        tile.set_fill(GREY_E, 0)
        globals().update(locals())
        tiles = Group(*(Group(tile.copy()) for x in range(9))).arrange_in_grid(3, 3, buff=0)
        tiles.set_height(FRAME_HEIGHT)
        for tile, channel in zip(tiles, channels):
            name, subs, slug, views = channel
            video = ImageMobject(os.path.join(DOC_DIR, "thumbnails", slug))
            video.set_height(tile.get_height() * 0.55)
            video.move_to(tile)

            title = Text(f"{name} ({subs} subs)", font_size=18, color=GREY_A)
            title.next_to(video, UP, SMALL_BUFF)

            count = VGroup(
                # Text(f"{subs} subs,", font_size=24),
                Integer(views, font_size=30),
                Text("Views", font_size=30)
            )
            count.arrange(RIGHT, buff=0.1, aligned_edge=UP)
            count.next_to(video, DOWN, buff=0.15)
            tile.add(title, video, count)

        self.add(tiles)

        # Show tiles
        self.play(
            LaggedStart(*(
                UpdateFromAlphaFunc(tile, lambda m, a: m.set_opacity(a))
                for tile in tiles
            ), lag_ratio=0.5),
            LaggedStart(*(
                CountInFrom(tile[-1][0], 0, run_time=2.5)
                for tile in tiles
            ), lag_ratio=0.2)
        )


class AllVideos(AllVideosOrdered):
    os.path.join(DOC_DIR, "some2_video_urls.txt")
    grid_size = 3
    time_per_image = 0.2


class IntroScene(InteractiveScene):
    def construct(self):
        # Intro
        logos = Group(ImageMobject("Leios"), Logo())
        for logo in logos:
            logo.set_height(1.5)
        logos.arrange(DOWN, buff=LARGE_BUFF)
        logos.to_edge(LEFT)
        title = Text("Summer of Math\nExposition", font_size=96)
        title.move_to(midpoint(logos.get_right(), RIGHT_SIDE))

        self.add(title)
        self.play(LaggedStart(
            FadeIn(logos[0], LEFT),
            Write(logos[1], run_time=1, lag_ratio=0.01, stroke_width=0.1, stroke_color=BLACK),
            lag_ratio=0.3,
        ))
        self.wait()

        # Compress name
        words = ["Summer", "of", "Math", "Exposition"]
        inits = VGroup()
        to_fade = VGroup()
        for word in words:
            part = title.select_part(word)
            inits.add(part[0])
            to_fade.add(*part[1:])

        inits.generate_target()
        inits.target.arrange(RIGHT, aligned_edge=DOWN, buff=SMALL_BUFF)
        inits.target.set_height(0.8)
        inits.target.set_stroke(WHITE, 1)
        inits.target.center().to_edge(UP)

        self.play(
            FadeOut(to_fade, lag_ratio=0.1, shift=UP, run_time=1.5),
            MoveToTarget(inits, time_span=(0.25, 1.5)),
            FadeOut(logos, LEFT)
        )

        # Add 2
        two = Text("2")
        two.match_height(inits)
        two.set_color(YELLOW)
        inits.generate_target()
        inits.target.shift(0.25 * LEFT)
        two.next_to(inits.target, RIGHT, buff=0.2, aligned_edge=DOWN)

        globals().update(locals())
        self.play(
            LaggedStart(*(
                FadeTransform(init.copy(), two)
                for init in inits
            ), lag_ratio=0.1),
            MoveToTarget(inits),
        )
        inits.add(two)
        self.wait()

        # View count
        count = VGroup(
            Integer(7000000, edge_to_fix=RIGHT),
            Text("collective views"),
        )
        count.arrange(RIGHT, aligned_edge=UP)
        count.set_height(0.6)
        count.to_corner(UR)

        self.play(
            inits.animate.set_height(0.6).to_corner(UL),
            FadeIn(count[1]),
            CountInFrom(count[0], 0, run_time=3),
        )
        self.add(count)
        count[0].edge_to_fix = LEFT
        count[0].add_updater(lambda m, dt: m.increment_value(int(dt * 1000)))
        self.wait(5)


class PeerReview(InteractiveScene):
    def construct(self):
        # Add everyone
        n_peers = 10
        peers = VGroup(*(
            PiCreature(color=random.choice([BLUE_E, BLUE_D, BLUE_C, BLUE_B]))
            for x in range(n_peers)
        ))
        peers.arrange(RIGHT)
        peers.set_width(FRAME_WIDTH - 1)
        peers.to_edge(DOWN)
        copies = VGroup(peers.copy().set_y(0), peers.copy().to_edge(UP, LARGE_BUFF))
        all_pis = VGroup(*peers, *it.chain(*copies))

        videos = VGroup(*(VideoIcon() for peer in all_pis))
        for video, pi in zip(videos, all_pis):
            video.set_height(0.25)
            video.set_color(random_bright_color())
            video.next_to(pi.get_corner(UR), UP, SMALL_BUFF)

        self.add(peers)
        pairs = list(zip(videos, all_pis))
        random.shuffle(pairs)
        self.play(LaggedStart(*(
            AnimationGroup(
                pi.change("raise_right_hand"),
                FadeIn(video, 0.5 * UP)
            )
            for video, pi in pairs
        )))

        # Entry box
        entry_box = Rectangle(4, 4)
        entry_box.set_fill(GREY_E, 1)
        entry_box.set_stroke(WHITE, 1)
        entry_box.to_edge(UP, buff=0.5)
        entry_box_label = Text("All entries")
        entry_box_label.next_to(entry_box.get_top(), DOWN, SMALL_BUFF)
        entry_box_label.set_backstroke(BLACK, 5)
        videos.generate_target()
        videos.target.arrange_in_grid(buff=0.25)
        videos.target.move_to(entry_box)

        self.add(entry_box, videos, entry_box_label)
        self.play(
            FadeIn(entry_box),
            FadeIn(entry_box_label),
            FadeOut(copies, lag_ratio=0.1),
            MoveToTarget(videos),
            LaggedStart(*(peer.change("tease", ORIGIN) for peer in peers))
        )

        # Peer review process
        peers.shuffle()
        for pi in peers:
            pi.watchlist = list(videos.copy())
            random.shuffle(pi.watchlist)

        self.play(LaggedStart(*(
            AnimationGroup(
                pi.watchlist[0].animate.next_to(pi.get_top(), UL, buff=0.1).shift(0.025 * RIGHT),
                pi.watchlist[1].animate.next_to(pi.get_top(), UR, buff=0.1).shift(0.025 * LEFT),
                pi.change("pondering", pi.get_top() + UP),
            )
            for pi in peers
        ), lag_ratio=0.1))

        for x in range(10):
            globals().update(locals())
            self.play(LaggedStart(*(
                AnimationGroup(
                    pi.change(
                        ["raise_right_hand", "raise_left_hand"][1 - bit],
                        pi.watchlist[x + bit],
                    ).set_anim_args(run_time=0.5),
                    pi.watchlist[x + bit].animate.scale(1.2),
                    pi.watchlist[x + 1 - bit].animate.scale(0.8),
                )
                for pi in peers
                for bit in [random.randint(0, 1)]
            ), lag_ratio=0.1))
            self.play(LaggedStart(*(
                AnimationGroup(
                    pi.change("sassy", pi.get_top() + UP).set_anim_args(run_time=0.5),
                    FadeOut(pi.watchlist[x], scale=0.5),
                    pi.watchlist[x + 1].animate.set_height(0.25).next_to(pi.get_top(), UL, buff=0.1).shift(0.025 * RIGHT),
                    pi.watchlist[x + 2].animate.next_to(pi.get_top(), UR, buff=0.1).shift(0.025 * LEFT),
                )
                for pi in peers
            )))

    def get_judgement_animation(self, pi, videos):
        vids = list(videos.copy())
        random.shuffle(vids)

        anims = [Animation(Mobject(), run_time=3 * random.random())]
        anims.append(AnimationGroup(
            vids[0].animate.next_to(pi.get_top(), UL, buff=0.1),
            vids[1].animate.next_to(pi.get_top(), UR, buff=0.1),
            pi.change("pondering", pi.get_top() + UP),
        ))
        for i in range(8):
            anims.append(pi.animate.change(random.choice(["raise_right_hand", "raise_left_hand"])))
            anims.append(AnimationGroup(
                FadeOut(vids[i], LEFT),
                vids[i + 1].animate.next_to(pi.get_top(), UL, buff=0.1),
                videos[i + 2].animate.next_to(pi.get_top(), UR, buff=0.1),
            ))

        return Succession(*anims)


class Over10KComparisons(InteractiveScene):
    def construct(self):
        # Start
        count = VGroup(
            Integer(10000),
            Text("Comparisons")
        )
        count.arrange(DOWN)
        count.scale(2)
        plus = Text("+").scale(2)
        plus.next_to(count[0], RIGHT, buff=SMALL_BUFF)

        self.play(
            CountInFrom(count[0], 0, run_time=3),
            Write(count[1], run_time=1),
        )
        self.add(plus)
        self.wait()


class YTStatement(InteractiveScene):
    def construct(self):
        # Start
        words = VGroup(
            Text("Shared viewer base"),
            Tex("\\Downarrow", font_size=96).set_stroke(WHITE, 0),
            Text("""
                YT is more likely to
                recommend one video
                to viewers of another.
            """, alignment="LEFT")
        )
        words.arrange(DOWN, buff=MED_LARGE_BUFF)

        self.add(words[0])
        self.play(
            Write(words[1]),
            FadeIn(words[2], DOWN)
        )
        self.wait()


class HistogramOfViews(InteractiveScene):
    data = [
        628083,
        396785,
        359233,
        301850,
        245930,
        243370,
        231732,
        212708,
        209343,
        208586,
        167442,
        153613,
        145692,
        135465,
        130783,
        126159,
        101919,
        93069,
        92110,
        89790,
        82160,
        81311,
        79598,
        67092,
        63373,
        63239,
        62246,
        56242,
        52840,
        47267,
        45942,
        43032,
        42558,
        41618,
        41201,
        39059,
        36675,
        36138,
        35999,
        35999,
        35805,
        34057,
        33137,
        33009,
        31064,
        30855,
        30616,
        30122,
        30016,
        28817,
    ]

    def construct(self):
        # Add axes
        axes = Axes((0, 50, 50), (0, 6e5, 1e5), width=12, height=6)
        axes.to_corner(DR)

        y_step = 100000
        axes.add_coordinate_labels(
            x_values=range(0),
            y_values=range(y_step, 7 * y_step, y_step),
        )
        self.add(axes)

        # Bars
        bars = VGroup(*(
            Rectangle(
                width=axes.x_axis.get_width() / len(self.data),
                height=value * axes.y_axis.unit_size,
            )
            for value in self.data
        ))
        bars.arrange(RIGHT, aligned_edge=DOWN, buff=0)
        bars.set_fill(opacity=1)
        bars.set_submobject_colors_by_gradient(BLUE_D, TEAL)
        bars.set_stroke(BLACK, 2)
        bars.move_to(axes.get_origin(), DL)
        bars.shift(0.01 * UR)
        self.add(bars)

        # Title
        title = Text(
            "Distribution of SoME2 views (top 50 shown)"
        )
        title.move_to(axes, UP)
        subtitle = Text("7,000,000+ total", font_size=36, fill_color=GREY_B)
        subtitle.next_to(title, DOWN)

        self.add(title, subtitle)

        # Animate bars
        self.remove(bars)
        bars.sort(lambda p: -p[0])
        self.play(LaggedStart(*(
            ReplacementTransform(bar.copy().stretch(0, 1, about_edge=DOWN), bar)
            for bar in bars
        )), lag_ratio=0.2)
        self.add(bars)

        # SoME1
        some1_words = Text("""
            2,000,000+
            Additional SoME1 views
            during this time
        """, font_size=36)
        box = SurroundingRectangle(some1_words, buff=MED_LARGE_BUFF)
        box.set_fill(GREY_E, 1)
        box.set_stroke(WHITE, 1)
        VGroup(box, some1_words).to_edge(RIGHT)

        self.play(
            FadeIn(box),
            Write(some1_words),
        )
        self.wait()


class ValueInSharedGoals(TeacherStudentsScene):
    def construct(self):
        # Start
        morty = self.teacher

        self.play(
            morty.says("Never underestimate\ncommunity and\ndeadlines"),
            self.change_students("happy", "tease", "thinking"),
        )
        self.wait(5)

        self.play(
            morty.debubble(mode="raise_right_hand", look_at=self.screen),
            self.change_students("pondering", "thinking", "erm", look_at=self.screen),
        )
        self.wait(2)

        # Show links
        links = VGroup(
            Text("explanaria.github.io/crystalgroups"),
            Text("thenumb.at/Autodiff"),
            Text("xperimex.com/blog/panorama-homography"),
            Text("calmcode.io/blog/inverse-turing-test.html"),
            Text("summbit.com/blog/bezier-curve-guide"),
            Text("lukelavalva.com/theoryofsliding"),
            Text("chessengines.org"),
            Tex("\\vdots")
        )
        links.arrange(DOWN, aligned_edge=LEFT)
        links[-1].match_x(links[-2])
        links.set_width(5)
        links.to_corner(UR)
        self.play(
            morty.change("raise_left_hand", links),
            self.change_students("tease", "happy", "thinking", look_at=links),
            FadeIn(links, lag_ratio=0.3, run_time=2)
        )
        self.wait(4)

        # Who won
        stds = self.students
        self.play(
            FadeOut(links, RIGHT),
            morty.change("guilty"),
            stds[0].says("But, like, who actually\nwon the contest?"),
            stds[1].change("erm", stds[0].eyes),
            stds[2].change("sassy", stds[0].eyes),
        )
        self.wait(3)


class OneHundredEntries(InteractiveScene):
    def construct(self):
        # Logos
        logos = Group(ImageMobject("Leios"), Logo())
        for logo in logos:
            logo.set_height(0.75)
        logos.arrange(RIGHT, buff=MED_LARGE_BUFF)
        logos.to_corner(UL)
        logos[1].flip().flip()

        # Winners
        title = Text("Winners", font_size=72)
        title.add(Underline(title).scale(1.5).insert_n_curves(20).set_stroke(width=(0, 3, 3, 3, 0)))
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        title.set_color(GREY_A)

        winners = VGroup(
            VideoIcon(),
            self.written_icon(),
            *(VideoIcon() for x in range(3))
        )
        winners.set_fill(BLUE_D)
        winners[1].set_fill(GREY_A)
        for winner in winners:
            winner.set_height(0.75)
        winners.arrange(DOWN, buff=MED_LARGE_BUFF)
        winners.next_to(title, DOWN, buff=MED_LARGE_BUFF)
        winners.shift(1.5 * LEFT)

        prizes = VGroup(*(
            VGroup(
                Text("$1,000 + ", color=GREEN),
                Randolph(mode="thinking", height=0.5, color=YELLOW)
            ).arrange(RIGHT, buff=MED_SMALL_BUFF).next_to(winner, RIGHT)
            for winner in winners
        ))

        self.add(title)
        self.play(LaggedStartMap(FadeIn, winners, shift=0.5 * DOWN, lag_ratio=0.1, run_time=1))
        self.play(LaggedStartMap(FadeIn, prizes, shift=0.25 * RIGHT, lag_ratio=0.1, run_time=1))
        self.wait()
        self.play(LaggedStartMap(GrowFromCenter, logos, lag_ratio=0.7, run_time=1))
        self.wait()

        # Large batches
        videos = VideoIcon().get_grid(10, 8, fill_rows_first=False)
        videos.set_height(6)
        videos.set_fill(BLUE_D)

        articles = self.written_icon().get_grid(10, 3, fill_rows_first=False)
        articles.remove(*articles[-5:])
        articles.match_height(videos)
        articles.next_to(videos, RIGHT, buff=2.0)

        VGroup(videos, articles).center().to_edge(DOWN, buff=MED_SMALL_BUFF)

        video_title = Text("80 top videos").next_to(videos, UP, buff=MED_LARGE_BUFF)
        article_title = Text("25 top non-video entries").next_to(articles, UP, buff=MED_LARGE_BUFF)

        self.play(
            LaggedStart(*(
                ReplacementTransform(vid, random.choice(videos))
                for vid in (winners[0], *winners[2:])
            )),
            ReplacementTransform(winners[1], random.choice(articles)),
            LaggedStartMap(FadeIn, videos),
            LaggedStartMap(FadeIn, articles),
            FadeOut(title, UP),
            LaggedStartMap(FadeOut, prizes, shift=UP),
        )
        self.add(videos, articles)
        self.play(
            Write(video_title),
            Write(article_title),
        )
        self.wait()

        # Mention added judges
        guest_words = Text("With guest judging help from:")
        guest_words.next_to(videos, LEFT, LARGE_BUFF).align_to(video_title, UP)
        guest_words.set_color(GREY_A)
        frame = self.camera.frame
        frame.save_state()

        judges = VGroup(
            Text("Alex Kontorovich"),
            Text("Henry Reich"),
            Text("Mithuna Yoganathan"),
            Text("Nicky Case"),
            Text("Tai-Danae Bradley"),
            Text("Deniz Ozbay"),
            Text("Jeffrey Wack"),
            Text("Fran Herr"),
        )
        judges.arrange(DOWN, aligned_edge=LEFT)
        judges.next_to(guest_words, DOWN, LARGE_BUFF)

        momath_brace = Brace(judges[-3:], RIGHT)
        momath_words = momath_brace.get_text("MoMath\nMuseum", buff=MED_SMALL_BUFF)
        momath = VGroup(momath_brace, momath_words)
        momath.set_color(YELLOW)

        self.play(
            frame.animate.scale(1.5, about_edge=RIGHT),
            FadeIn(guest_words),
            logos.animate.next_to(guest_words, UP),
            LaggedStartMap(FadeIn, judges, shift=0.5 * RIGHT, lag_ratio=0.5, run_time=6),
        )
        self.play(
            FadeIn(momath)
        )
        self.wait()

        # Return
        self.play(
            frame.animate.restore(),
            LaggedStart(*(
                FadeOut(mob, UL)
                for mob in [video_title, article_title, *logos, guest_words, *judges, momath]
            ), run_time=1)
        )
        self.wait()

    def written_icon(self):
        rect = Rectangle(height=0.1, width=1.5)
        rect.set_fill(WHITE, 1)
        rect.set_stroke(WHITE, 0)
        result = rect.get_grid(4, 1, buff=0.2)
        result[-1].stretch(0.5, 0, about_edge=LEFT)

        return result


class FeaturedContentFrame(InteractiveScene):
    title = "Example Title"
    thumbnail = "Lion"
    author = "author"
    frame_scale_factor = 0.75

    def construct(self):
        # Frames
        self.add(FullScreenRectangle())

        frame = ScreenRectangle()
        frame.set_fill(BLACK, 1)
        frame.set_stroke(WHITE, 2)
        frame.set_height(self.frame_scale_factor * FRAME_HEIGHT)
        frame.to_edge(DOWN, buff=MED_SMALL_BUFF)
        self.add(frame)

        # Title
        title = Text(self.title, font_size=66)
        title.set_max_width(FRAME_WIDTH - 1)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        author = Text("by " + self.author, font_size=48, color=GREY_A)
        author.move_to(midpoint(title.get_bottom(), frame.get_top()))

        if self.thumbnail:
            thumbnail = ImageMobject(self.thumbnail)
            thumbnail.replace(frame, dim_to_match=1)
        else:
            thumbnail = Mobject()

        self.play(
            Write(title, run_time=1),
            DrawBorderThenFill(frame),
            FadeIn(thumbnail),
        )
        self.play(FadeIn(author, shift=0.5 * DOWN))
        self.wait()


class FourCategories(InteractiveScene):
    frame_scale_factor = 0.75

    def setup(self):
        super().setup()
        df = pd.read_csv(os.path.join(DOC_DIR, "featured_entries.csv"))
        self.titles = df['Title']
        self.authors = df['Author']
        self.urls = df['Link']

    def construct(self):
        # Four categories
        regions = VGroup(*(ScreenRectangle() for x in range(4)))
        regions.set_height(FRAME_HEIGHT / 2)
        regions.arrange_in_grid(2, 2, buff=0)
        regions.set_stroke(width=0)
        regions.set_fill(opacity=0.3)
        regions.set_submobject_colors_by_gradient(
            BLUE_D, BLUE_C, BLUE_E, GREY_BROWN,
        )

        titles = VGroup(*(
            Text(word, font_size=60)
            for word in ["Motivation", "Clarity", "Novelty", "Memorability"]
        ))
        for title, region in zip(titles, regions):
            title.next_to(region.get_top(), DOWN, buff=MED_SMALL_BUFF)

        self.play(
            LaggedStartMap(FadeIn, regions, lag_ratio=0.5),
            LaggedStartMap(Write, titles, lag_ratio=0.8),
            run_time=3
        )
        self.wait()

        # Lesson tiles
        tile = self.get_entry_tile().set_height(0.5)
        regions[0].tiles = tile.get_grid(2, 5)
        regions[1].tiles = tile.get_grid(2, 2)
        regions[2].tiles = tile.get_grid(1, 1)
        regions[3].tiles = tile.get_grid(1, 2)
        self.tiles = VGroup()
        for region in regions:
            region.tiles.move_to(region)
            self.tiles.add(*region.tiles)

        self.play(LaggedStart(*(
            FadeIn(tile, scale=0.8)
            for tile in self.tiles
        )), run_time=5, lag_ratio=0.1)
        self.wait()

        # Accentuate titles
        icons = VGroup(*(
            SVGMobject(file)
            for file in ["teaching", "gem", "lightbulb", "memory"]
        ))
        icons.set_stroke(WHITE, width=0)
        icons.set_fill(WHITE, opacity=1)
        icons.set_color_by_gradient(GREY_A, BLUE_B, YELLOW, GREY_B)
        icons[2].set_opacity(0.8)
        icons[1].set_opacity(0.7)
        for title, icon in zip(titles, icons):
            icon.set_height(1.5 * title[0].get_height())
            icon.next_to(title, RIGHT).match_y(title[0])
            title.icon = icon

            self.play(
                FadeIn(icon, lag_ratio=0.25),
                FlashUnder(title),
            )
            title.add(icon)
        self.wait()

        # Relative importance
        self.play(LaggedStart(
            regions[:2].animate.stretch(1.4, 1, about_edge=UP),
            regions[2:].animate.stretch(0.6, 1, about_edge=DOWN),
            titles[:2].animate.shift(MED_SMALL_BUFF * DOWN),
            titles[2:].animate.shift(2 * DOWN),
            regions[0].tiles.animate.shift(DOWN),
            regions[1].tiles.animate.shift(DOWN),
            regions[2].tiles.animate.shift(1.4 * DOWN),
            regions[3].tiles.animate.shift(1.4 * DOWN),
            lag_ratio=0.025,
            run_time=2
        ))
        self.wait()

        # Isolate Motivation
        regions[0].save_state()
        titles[0].save_state()

        self.add(regions[0], regions[0].tiles)
        self.play(
            regions[0].animate.replace(FullScreenRectangle()).set_opacity(0),
            titles[0].animate.set_x(0),
            regions[0].tiles.animate.scale(2).center(),
            FadeOut(regions[1:]),
            FadeOut(titles[1:]),
            *(FadeOut(regions[i].tiles) for i in range(1, 4)),
        )

        # Subregions
        subregions = Rectangle(height=7, width=FRAME_WIDTH / 2).get_grid(1, 2, buff=0)
        subregions.to_edge(DOWN, buff=0)
        subregions.set_stroke(WHITE, 1)
        subregions.set_fill(GREY_E, 1)
        subtitles = VGroup(Text("Macro"), Text("Micro"))
        subtitles.scale(1.2)
        for st, sr in zip(subtitles, subregions):
            st.next_to(sr.get_top(), DOWN)

        descriptions = VGroup(
            Text("Have you generated a\ndesire to learn?"),
            Text("Is each new idea given\na good reason to be there?"),
        )
        descriptions.scale(0.8)
        for desc, st in zip(descriptions, subtitles):
            desc.next_to(st, DOWN, MED_LARGE_BUFF)

        self.play(
            Write(subregions),
            titles[0].animate.to_edge(UP, buff=SMALL_BUFF),
            regions[0].tiles[:7].animate.scale(0.9).arrange_in_grid(3, 3).move_to(subregions[0]).align_to(ORIGIN, UP),
            regions[0].tiles[7:].animate.scale(0.9).arrange_in_grid(2, 2).move_to(subregions[1]).align_to(ORIGIN, UP),
            LaggedStartMap(FadeIn, subtitles, shift=0.5 * DOWN)
        )
        self.wait()
        self.play(Write(descriptions[0]))

        for i in range(7):
            self.highlight_tile(i)

        self.play(Write(descriptions[1]))
        for i in range(7, 10):
            self.highlight_tile(i)

        # Intersection with clarity
        frame = self.camera.frame
        titles[1].save_state()
        c_region = subregions[0].copy()
        c_region.set_fill(regions[1].get_fill_color(), 0.2)
        c_region.set_stroke(BLUE, 3)
        c_region.next_to(subregions, RIGHT, buff=0.1)
        titles[1].match_y(titles[0])
        titles[1].match_x(c_region)

        regions[1].tiles.scale(regions[0].tiles[0].get_height() / regions[1].tiles[0].get_height())
        regions[1].tiles.arrange_in_grid(buff=0.3)
        regions[1].tiles.move_to(c_region)
        regions[1].tiles.align_to(regions[0].tiles[0], UP)

        self.play(
            frame.animate.scale(1.55).move_to(subregions[1]),
            FadeIn(titles[1]),
            FadeIn(c_region),
            FadeIn(regions[1].tiles),
        )
        self.wait()
        c_region.save_state()
        self.play(
            c_region.animate.stretch(2, 0, about_edge=RIGHT)
        )
        self.wait()

        # Attention and focus
        randy = Randolph()
        randy.next_to(subregions[0], UP)
        bucket = VMobject().set_points_as_corners([UL, LEFT, RIGHT, UR])
        bucket.set_stroke(WHITE, 1.5)
        bucket.set_width(1.0).set_height(1.5, stretch=True)
        bucket.next_to(randy.get_corner(UL), UP, buff=0)

        bucket_label = Text("Attention and focus", color=YELLOW)
        bucket_label.next_to(bucket, RIGHT, buff=MED_LARGE_BUFF, aligned_edge=UP)
        bucket_arrow = Arrow(
            bucket_label.get_corner(DL) + 0.7 * RIGHT + 0.1 * DOWN,
            bucket.get_right(),
            color=YELLOW
        )

        focus_dots = Group(*(
            GlowDot([
                interpolate(bucket.get_x(LEFT) + 0.1, bucket.get_x(RIGHT) - 0.1, random.random()),
                interpolate(bucket.get_y(DOWN) + 0.1, bucket.get_y(TOP) - 0.1, random.random()),
                0,
            ])
            for x in range(40)
        ))
        focus_dots.sort(lambda p: p[1])

        self.play(
            frame.animate.move_to(subregions[1], DOWN).shift(MED_SMALL_BUFF * DOWN),
            VFadeIn(randy),
            randy.change("tease"),
            FadeIn(bucket),
        )
        self.play(
            FadeIn(bucket_label),
            ShowCreation(bucket_arrow),
            FadeIn(focus_dots, lag_ratio=0.5),
        )
        self.wait()
        self.play(Blink(randy))
        self.wait()

        # Draining attention and focus
        randy_copy = randy.copy()
        randy_copy.generate_target()
        randy_copy.target.next_to(subregions.get_corner(UR), UR)
        randy_copy.target.change_mode("pondering").look(DOWN)

        bucket_copy = bucket.copy()
        bucket_copy.generate_target()
        bucket_copy.target.next_to(randy_copy.target.get_corner(UL), UP, buff=0)
        bucket_copy.set_opacity(0)

        focus_dots_copy = focus_dots.copy().set_opacity(0)

        self.play(
            MoveToTarget(randy_copy),
            MoveToTarget(bucket_copy),
            focus_dots_copy.animate.move_to(bucket_copy.target).set_opacity(1)
        )
        focus_dots_copy.sort(lambda p: -p[1])
        self.play(
            LaggedStartMap(FadeOut, focus_dots_copy, shift=0.2 * UP, scale=3, lag_ratio=0.4, run_time=6),
            randy_copy.change("erm", regions[1].tiles).set_anim_args(time_span=(4, 5))
        )
        self.wait()
        self.play(Blink(randy_copy))
        self.wait()

        # Transition to clarity
        self.play(
            titles[1].animate.center().to_edge(UP),
            regions[1].tiles.animate.scale(1.5).center(),
            c_region.animate.become(FullScreenRectangle(fill_color=BLACK, fill_opacity=0)),
            frame.animate.set_height(FRAME_HEIGHT).center(),
            *(
                FadeOut(mob, shift=LEFT, time_span=(0.05 * i, 0.05 * i + 1))
                for i, mob in enumerate([
                    randy, bucket, focus_dots, bucket_label,
                    randy_copy, bucket_copy,
                    titles[0], subregions,
                    subtitles, descriptions, *regions[0].tiles,
                ])
            ),
            run_time=2
        )

        for i in range(10, 14):
            self.highlight_tile(i)

        # Novelty
        regions[0].restore()
        regions[0].tiles.scale(0.7)
        regions[0].tiles.arrange_in_grid(4, 3)
        regions[0].tiles.move_to(regions[0]).shift(0.5 * DOWN)

        titles[0].restore()
        self.remove(titles[1])

        self.play(
            FadeIn(regions),
            *map(FadeIn, (titles[0], titles[2], titles[3])),
            *(FadeIn(regions[i].tiles) for i in [0, 2, 3]),
            regions[1].tiles.animate.scale(regions[0].tiles[0].get_height() / regions[1].tiles[0].get_height()).move_to(regions[1]),
            titles[1].animate.restore()
        )
        self.play(FlashAround(regions[2], buff=-0.01, run_time=3, time_width=2))

        # Isolate novelty
        self.og_mobjects = Group(*self.mobjects)
        self.og_mobjects.save_state()
        self.play(
            regions[2].animate.replace(FullScreenRectangle(), stretch=True).set_opacity(0),
            titles[2].animate.center().to_edge(UP),
            *map(FadeOut, [*titles[:2], titles[3], *regions[:2], regions[3], *self.tiles])
        )

        # Components of novelty
        rects = Rectangle().get_grid(1, 2, buff=0)
        rects.set_height(1.5)
        rects.set_width(12, stretch=True)
        rects.next_to(titles[2], DOWN, buff=2.0)
        rects.set_stroke(WHITE, 1)
        rects[0].set_fill(BLUE_E, 1)
        rects[1].set_fill(GREY_BROWN, 1)
        subtitles = VGroup(
            Text("Stylistic\noriginality"),
            Text("Content\noriginality"),
        )
        for st, rect in zip(subtitles, rects):
            st.rect = rect
            st.start_width = st.get_width()
            st.add_updater(lambda m: m.set_max_width(min(m.start_width, 0.8 * m.rect.get_width())))
            st.add_updater(lambda m: m.move_to(m.rect))

        lines = VGroup(*(Line(titles[2].get_bottom(), st.get_top(), buff=0.25) for st in subtitles))
        lines.set_stroke(WHITE, 2)

        subtitles.suspend_updating()
        self.play(
            LaggedStartMap(ShowCreation, lines, lag_ratio=0.5),
            LaggedStartMap(FadeInFromPoint, subtitles, point=titles[2].get_bottom(), lag_ratio=0.5),
        )
        subtitles.resume_updating()
        self.wait()
        self.add(*rects, *subtitles)
        self.play(*map(FadeIn, rects), FadeOut(lines))
        self.play(
            rects[0].animate.set_width(rects[0].get_width() - 5, stretch=True, about_edge=LEFT),
            rects[1].animate.set_width(rects[1].get_width() + 5, stretch=True, about_edge=RIGHT),
        )
        self.wait()

        self.og_mobjects.restore()
        self.clear()
        self.add(*self.og_mobjects)

        # Last couple
        for i in range(14, 17):
            self.highlight_tile(i)

    def get_entry_tile(self):
        return get_entry_tile()

    def highlight_tile(self, index):
        bg = FullScreenFadeRectangle(fill_color=GREY_E, fill_opacity=1)
        tile = self.tiles[index]
        title = Text(self.titles[index], font_size=60)
        author = Text("by " + self.authors[index], font_size=48, color=GREY_A)
        image = self.url_to_image(self.urls[index])

        image.set_width(0.97 * tile.get_width())
        image.move_to(tile)
        image.set_opacity(0.0)

        original_tile = tile.copy()
        tile.generate_target()
        tile.target[1].set_opacity(0)
        tile.target[0].set_fill(BLACK, 1)
        tile.target.set_height(self.frame_scale_factor * FRAME_HEIGHT)
        tile.target.center().to_edge(DOWN, buff=MED_SMALL_BUFF)

        title.set_max_width(FRAME_WIDTH - 1)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        author.move_to(midpoint(title.get_bottom(), tile.target.get_top()))

        bg.save_state()
        bg.replace(tile)
        bg.set_opacity(0)

        self.add(bg, tile, image, title)
        self.play(
            bg.animate.restore(),
            MoveToTarget(tile),
            image.animate.replace(tile.target).set_opacity(1.0),
            Write(title, time_span=(0.5, 2.0)),
        )
        # self.play()
        self.play(FadeIn(author, 0.5 * DOWN))
        self.wait()

        tile.replace_submobject(1, image)
        self.play(
            tile.animate.replace(original_tile),
            *map(FadeOut, (bg, title, author)),
        )

    def url_to_image(self, url):
        return url_to_image(url)


class WhoCares(TeacherStudentsScene):
    def construct(self):
        # Start
        morty = self.teacher
        stds = self.students

        equation = Tex("\\sum_{n=1}^N n^2 = \\frac{N(N + 1)(2N + 1)}{6}")
        equation.set_stroke(WHITE, 1)
        equation.move_to(self.hold_up_spot, DOWN).shift(UL)

        self.play(
            morty.change("raise_right_hand", look_at=equation),
            Write(equation),
            self.change_students("pondering", "pondering", "pondering", look_at=equation),
        )
        self.wait()
        self.play(self.change_students("sassy", "confused", "erm"))
        self.wait()
        self.play(
            stds[0].says("Why should\nI care?", mode="angry"),
            morty.change("guilty"),
        )
        self.wait()
        self.play(
            morty.says("Well...hmmm...", mode="sassy", look_at=equation),
            stds[1].change("maybe"),
        )
        self.wait(5)


class Overphilosophizing(TeacherStudentsScene):
    def construct(self):
        # Start
        morty = self.teacher

        rects = VGroup(*(
            Rectangle(width=w)
            for w in [2, 9, 2]
        ))
        rects.set_height(1.0, stretch=True)
        rects.arrange(RIGHT, buff=0)
        rects.set_stroke(WHITE, 1)
        rects.set_fill(opacity=0.5)
        rects.set_submobject_colors_by_gradient(YELLOW_E, BLUE, GREY_BROWN)
        rects.set_y(2)

        words = VGroup(*map(Text, ["Motivation", "Core lesson", "Broader\nconnections"]))
        words.scale(0.7)
        for word, rect in zip(words, rects):
            word.rect = rect
            word.start_width = word.get_width()
            word.add_updater(lambda m: m.set_max_width(min(m.rect.get_width() * 0.8, m.start_width)))
            word.add_updater(lambda m: m.move_to(m.rect))

        self.add(*rects, *words)

        # Shift sizes
        def left_shift(x, rects=rects, **kw):
            return (
                rects[0].animate.set_width(rects[0].get_width() + x, about_edge=LEFT, stretch=True).set_anim_args(**kw),
                rects[1].animate.set_width(rects[1].get_width() - x, about_edge=RIGHT, stretch=True).set_anim_args(**kw),
            )

        def right_shift(x, rects=rects, **kw):
            return (
                rects[1].animate.set_width(rects[1].get_width() + x, about_edge=LEFT, stretch=True).set_anim_args(**kw),
                rects[2].animate.set_width(rects[2].get_width() - x, about_edge=RIGHT, stretch=True).set_anim_args(**kw),
            )

        self.play(
            morty.change("guilty"),
            self.change_students("erm", "erm", "erm", look_at=rects),
        )
        self.wait()
        self.play(*left_shift(2, run_time=3))
        self.play(self.change_students("tired", "dejected", "tired", look_at=rects))
        self.wait()
        self.play(
            morty.change("raise_right_hand"),
            self.change_students("thinking", "thinking", "thinking", look_at=rects),
            *left_shift(-3),
        )
        self.wait()
        self.play(*right_shift(-1))

        # Examples beat philosophy
        exs = TexText("Examples", " > Sweeping statements", font_size=36)
        exs.to_edge(UP)
        exs.align_to(rects, LEFT)
        arrow = Arrow(exs[0].get_bottom(), rects[0].get_top(), buff=0.1)

        self.play(
            Write(exs),
            ShowCreation(arrow),
        )
        self.wait(4)


class MusicSidenote(InteractiveScene):
    def construct(self):
        # Randy
        randy = Randolph(height=2)
        randy.to_corner(DL)

        self.add(randy)

        # Add notes
        notes = SVGMobject("quarter-note").replicate(24)
        notes.arrange(RIGHT, buff=LARGE_BUFF)
        notes.set_width(FRAME_WIDTH + 5)
        notes.set_fill(GREY_B, 1)
        notes.set_stroke(WHITE, 0)
        for note in notes:
            note.shift(random.randint(0, 6) * 0.2 * UP)
        notes.center().to_edge(UP).to_edge(LEFT)

        self.play(
            LaggedStart(*(
                FadeIn(note, 0.2 * LEFT)
                for note in notes
            ), lag_ratio=0.3),
            randy.change("thinking", notes),
        )
        self.play(
            notes.animate.to_edge(RIGHT).set_anim_args(run_time=3),
            randy.change("tease"),
        )

        # Technical explanation
        outline = ScreenRectangle().set_height(4)
        outline.set_stroke(WHITE, 2)
        outline.next_to(randy, RIGHT, buff=1.0)
        outline.to_edge(DOWN, buff=1.0)

        self.play(
            ShowCreation(outline),
            randy.change("confused"),
        )
        self.play(Blink(randy))
        self.wait(2)

        self.play(
            LaggedStart(*(
                note.copy().animate.set_opacity(0).scale(0.5).move_to(randy.get_top())
                for note in notes
            ), lag_ratio=0.2),
            randy.change("concentrating")
        )
        self.play(Blink(randy))
        self.wait()


class NoveltySubdivision(InteractiveScene):
    def construct(self):
        pass


class NotAsUnique(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.says("It's not as\nunique these days", mode="shruggie"),
            self.change_students("happy", "tease", "happy")
        )
        self.wait(3)


class Winners(InteractiveScene):
    def construct(self):
        # Add title
        title = Text("Winners", font_size=72)
        underline = Underline(title)
        underline.scale(1.5)
        underline.insert_n_curves(20)
        underline.set_stroke(width=(0, 3, 3, 3, 0))
        title.add(underline)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        title.set_color(YELLOW)
        self.add(title)

        # Winner tiles
        winner_data = [
            ("a-767WnbaCQ", "Percolation: a Mathematical Phase Transition", "Spectral Collective"),
            ("crystalgroups", "Clear Crystal Conundrums", "Ex Planaria"),
            ("5M2RWtD4EzI", "This Is the Calculus They Won't Teach You", "A Well-Rested Dog"),
            ("gsZiJeaMO48", "How Realistic CGI Works (And How To Do It Way Faster)", "Joshua Maros"),
            ("6hVPNONm7xw", "The Coolest Hat Puzzle You've Probably Never Heard", "Going Null"),
        ]

        def get_winner_tile(slug, name, author):
            image = ImageMobject(os.path.join(DOC_DIR, "thumbnails", slug))
            image.set_height(2)
            label = Text(name)
            label.next_to(image, DOWN)
            sublabel = Text("by " + author, font_size=36)
            sublabel.next_to(image, DOWN)
            sublabel.set_color(GREY_A)
            return Group(image, sublabel)

        winners = Group(*(get_winner_tile(*d) for d in winner_data))
        winners.arrange_in_grid(2, 3, buff=0.5)
        winners[3:].match_x(winners[:3])
        winners.center().to_edge(DOWN)

        mystery_tiles = Group(*(
            get_entry_tile().replace(winner[0])
            for winner in winners
        ))

        first_outline = SurroundingRectangle(mystery_tiles[0], buff=0)
        first_outline.set_stroke(BLUE, 3)
        group_label = Text("Collaboration", color=BLUE)
        group_label.next_to(mystery_tiles[0], UP, SMALL_BUFF)

        self.play(LaggedStartMap(FadeIn, mystery_tiles, scale=2, lag_ratio=0.2))
        self.play(
            ShowCreation(first_outline),
            FadeIn(group_label, 0.2 * UP)
        )
        self.wait()
        for tile, winner in zip(mystery_tiles, winners):
            self.play(
                FadeOut(tile, scale=0.5),
                FadeIn(winner[0], scale=2),
                FadeIn(winner[1], DOWN, time_span=(0.5, 1.5))
            )
            self.add(winner)
            self.wait()

        # Honorable mentions
        hm_slugs = [
            "v_HeaeUUOnc",
            "piF6D6CQxUw",
            "QC3CjBZLHXs",
            "KufsL2VgELo",
            "zR_hpai3XkY",
            "HeBP3MG-WHg",
            "2dwQUUDt5Is",
            "3gyHKCDq1YA",
            "nK2jYk37Rlg",
            "l7bYY2U5ld8",
            "dwNxVpbEVcc",
            "CFBa2ezTQJQ",
            "gnUYoQ1pwes",
            "LUCvSsx6-EU",
            "-vxW42R47bc",
            "inverse-turing-test",
            "bezier-curve-guide",
            "does-every-game-have-a-winner",
            "Autodiff",
            "panorama-homography",
        ]
        random.shuffle(hm_slugs)

        to_fade = VGroup(title, first_outline, group_label, *(winner[1] for winner in winners))
        winner_tiles = Group(*(winner[0] for winner in winners))

        hm_tiles = Group(*(
            ImageMobject(os.path.join(DOC_DIR, "thumbnails", slug))
            for slug in hm_slugs
        ))
        winner_tiles.generate_target()
        tile_grid = Group(*winner_tiles.target, *hm_tiles)
        for tile in tile_grid:
            tile.set_width(1)
        tile_grid.arrange_in_grid(5, 5, buff=SMALL_BUFF)
        tile_grid.set_height(7.5)

        self.play(
            FadeOut(to_fade, run_time=2, lag_ratio=0.5),
            MoveToTarget(winner_tiles),
        )
        self.play(LaggedStartMap(FadeIn, hm_tiles, lag_ratio=0.5))

        # Wandering outline
        outline = SurroundingRectangle(winner_tiles[0], buff=0)
        outline.set_stroke(YELLOW, 3)

        self.play(FadeIn(outline))
        search_list = list(tile_grid[1:])
        random.shuffle(search_list)
        for tile in search_list[:12]:
            self.play(outline.animate.move_to(random.choice(tile_grid)))
            self.wait()


class Sponsors(InteractiveScene):
    def construct(self):
        # Images
        self.add(FullScreenFadeRectangle(fill_color=interpolate_color(GREY_A, WHITE, 0.5)))

        img_dir = os.path.join(DOC_DIR, "sponsor_logos")
        brilliant = ImageMobject(os.path.join(img_dir, "Brilliant"))
        risczero = ImageMobject(os.path.join(img_dir, "RiscZero"))
        google = SVGMobject(os.path.join(img_dir, "GoogleFonts"))
        google.set_stroke(width=0)
        pl = ImageMobject(os.path.join(img_dir, "ProtocolLabs"))

        logos = Group(brilliant, risczero, google, pl)
        for logo in logos:
            logo.set_width(5)

        logos.arrange(DOWN)
        logos[0].shift(0.35 * UP)
        logos[2:].shift(0.75 * DOWN)
        logos.center().to_corner(DL)
        logos.to_edge(DOWN, buff=0)

        # Contributions
        contributions = VGroup(
            Text("$1,000 for each winner").next_to(brilliant, RIGHT, buff=2.0),
            Text("$500 for each\nhonorable mention").next_to(logos[1:3], RIGHT, buff=2.0),
            Text("Operations costs").next_to(pl, RIGHT, buff=2.0),
        )
        contributions.set_fill(BLACK)
        contributions.set_stroke(BLACK, 0)
        brace = Brace(logos[1:3], RIGHT)
        brace.set_fill(BLACK, 1)
        arrows = VGroup()
        for logo, j in zip(logos, [0, 1, 1, 2]):
            arrows.add(Arrow(logo.get_right(), contributions[j].get_left(), color=BLACK))

        # Animations
        self.play(FadeIn(brilliant, 0.5 * DOWN))
        self.play(ShowCreation(arrows[0]), Write(contributions[0]))
        self.wait()

        self.play(LaggedStartMap(FadeIn, logos[1:3], shift=DOWN, lag_ratio=0.7))
        self.play(*map(ShowCreation, arrows[1:3]), Write(contributions[1]))
        self.wait()

        self.play(FadeIn(pl, DOWN))
        self.play(ShowCreation(arrows[3]), Write(contributions[2]))
        self.wait()


class EndScreen(PatreonEndScreen):
    CONFIG = {
        "title_text": "Playlist of all entries",
        "scroll_time": 30,
        "thanks_words": "Many additional thanks to these Patreon supporters"
    }
