from manim_imports_ext import *
from manimlib.logger import log
import urllib.request


WINNERS = [
    ("That weird light at the bottom of a mug — ENVELOPES", "Paralogical", "fJWnA4j0_ho"),
    ("Hiding Images in Plain Sight: The Physics Of Magic Windows", "Matt Ferraro", "CatInCausticImage"),
    ("The Beauty of Bézier Curves", "Freya Holmér", "aVwxzDHniEw"),
    ("What Is The Most Complicated Lock Pattern?", "Dr. Zye", "PKjbBQ0PBCQ"),
    ("Pick's theorem: The wrong, amazing proof", "spacematt", "uh-yRNqLpOg"),
]

HONORABLE_MENTIONS = [
    ("Dirac's belt trick, Topology, and Spin ½ particles", "Noah Miller", "ACZC_XEyg9U"),
    ("Galois-Free Guarantee! | The Insolubility of the Quintic", "Carl Turner", "BSHv9Elk1MU"),
    ("The Two Envelope Problem - a Mystifying Probability Paradox", "Formant", "_NGPncypY68"),
    ("The Math Behind Font Rasterization | How it Works", "GamesWithGabe", "LaYPoMPRSlk"),
    ("What is a Spinor?", "Mia Hughes", "SpinorArticle"),
    ("Understanding e", "Veli Peltola", "e_comic"),
    ("Ancient Multiplication Trick", "Inigo Quilez", "CsMrHzp850M"),
    ("对称多项式基本定理自我探究", "凡人忆拾", "FiveIntsTheorem"),
    ("Lehmer Factor Stencils", "Proof of Concept", "QzohwKT6TNA"),
    ("What is the limit of a sequence of graphs?", "Spectral Collective", "7Gj9BH4IZ-4"),
    ("Steiner's Porism: proving a cool animation", "Joseph Newton", "fKAyaP8IzlE"),
    ("Wait, Probabilities can be Negative?!", "Steven G", "std9EBbtOC0"),
    ("This random graph fact will blow your mind", "Snarky Math", "3QjZ31lj974"),
    ("Why is pi here? Estimating π by Buffon's n̶e̶e̶d̶l̶e noodle!", "Mihai Nica", "e-RUyCs9B08"),
    ("Introduction to Waves", "Rob Schlub", "IntroToWaves"),
    ("ComplexFunctions", "Treena", "Treena"),
    ("I spent an entire summer to find this spiral", "Sort of School", "n-e9C8g5x68"),
    ("HACKENBUSH: a window to a new world of math", "Owen Maitzen", "ZYj4NkeGPdM"),
    ("The Tale of the Lights Puzzle", "Throw Math At It", "9aZsABF-Vj4"),
    ("The BEST Way to Find a Random Point in a Circle", "nubDotDev", "4y_nmpv-9lI"),
    ("Secrets of the Fibonacci Tiles", "Eric Severson", "Ct7oltmdJrM"),
    ("The Tale of Three Triangles", "Robin Truax", "5nuYD2M2AX8"),
    ("How Karatsuba's algorithm gave us new ways to multiply", "Nemean", "cCKOl5li6YM"),
    ("Can you change a sum by rearranging its numbers? --- The Riemann Series Theorem", "Morphocular", "U0w0f0PDdPA"),
    ("Neural manifolds - The Geometry of Behaviour", "Artem Kirsanov", "QHj9uVmwA_0"),
]


def get_youtube_slugs(file="some1_video_urls.txt"):
    full_path = os.path.join(get_directories()["data"], file)
    slugs = []
    prefix = "https://youtu.be/"
    with open(full_path, "r") as fp:
        for url in fp.readlines():
            if not url.startswith(prefix):
                continue
            slugs.append(url[len(prefix):-1])
    return remove_list_redundancies(slugs)


def save_thumbnail_locally(slug, rewrite=False):
    file = yt_slug_to_image_file(slug)
    if os.path.exists(file) and not rewrite:
        return file
    suffixes = ["maxresdefault", "hqdefault", "mqdefault", "sddefault"]
    urls = [
        *(
            f"https://img.youtube.com/vi/{slug}/{suffix}.jpg"
            for suffix in suffixes
        ),
        *(
            f"https://i.ytimg.com/vi/{slug}/{suffix}.jpg"
            for suffix in suffixes
        )
    ]
    for url in urls:
        try:
            urllib.request.urlretrieve(url, file)
            return file
        except urllib.request.HTTPError:
            pass
    log.warning(f"Thumbnail not found: {slug}")


def save_thumbnails_locally(slugs):
    map(save_thumbnail_locally, slugs)


def yt_slug_to_image_file(slug):
    return os.path.join(
        get_raster_image_dir(),
        "some1_thumbnails",
        slug + ".jpg"
    )


class Introduction(TeacherStudentsScene):
    def construct(self):
        # Kill background
        self.clear()
        morty = self.teacher

        # Add title
        this_summer = Text("This Summer", font_size=72)
        this = this_summer.get_part_by_text("This")
        summer = this_summer.get_part_by_text("Summer")
        this_summer.to_edge(UP)
        some = Text("Summer of Math Exposition", font_size=72)
        some.to_edge(UP)
        some.shift(0.7 * RIGHT)
        this_summer.generate_target()
        this_summer.target.shift(some[0].get_left() - summer[0].get_left())

        logos = Group(
            ImageMobject("Leios"),
            Logo(),
        )
        for logo in logos:
            logo.set_height(1.0)
        logos.arrange(RIGHT, buff=2)
        logos.next_to(this_summer.target, RIGHT, buff=1.5)
        james_name = Text("James Schloss")
        james_name.set_color(GREY_A)
        james_name.next_to(logos[0], DOWN)

        logos.generate_target()
        logos.target.set_height(0.5)
        logos.target.arrange(RIGHT, buff=0.25)
        logos.target.replace(this_summer.target[:4], 0)

        self.add(this)
        self.wait(0.3)
        self.add(summer)
        self.wait(0.5)
        self.play(
            MoveToTarget(this_summer),
            FadeIn(james_name, lag_ratio=0.1),
            FadeIn(logos[0], LEFT),
        )
        self.play(
            Write(logos[1], lag_ratio=0.01, stroke_width=0.05, run_time=1),
            Animation(logos[1][-1].copy(), remover=True),
            FadeOut(james_name),
        )
        self.wait()
        self.play(
            FadeIn(some[len("Summer"):], lag_ratio=0.1),
            FadeOut(this),
            MoveToTarget(logos, path_arc=-PI / 3, run_time=2)
        )
        self.remove(this_summer)
        self.add(some)
        self.wait()

        title = Group(logos, some)

        # Mention blog post
        url = VGroup(
            Text("Full details: "),
            Text("https://3b1b.co/some1-results")
        )
        url[1].set_color(BLUE_C)
        url.arrange(RIGHT)
        url[0].shift(0.05 * UP)
        url.to_edge(UP)

        self.play(
            FadeOut(title, UP),
            FadeIn(url[0]),
            ShowIncreasingSubsets(url[1], rate_func=linear, run_time=2)
        )
        self.wait()

        # Key points
        dots = Dot().get_grid(3, 1)
        dots.arrange(DOWN, buff=1.5)
        dots.align_to(url, LEFT)
        dots.set_y(-0.5)

        points = VGroup(
            Text("Extremely open-ended"),
            Text("Promise to feature 4 to 5 winners"),
            Text("Brilliant offered $5k in cash prizes"),
        )
        colors = [GREEN_A, GREEN_B, GREEN_C]

        for dot, point, color in zip(dots, points, colors):
            point.next_to(dot, RIGHT)
            point.set_color(color)

        self.play(
            LaggedStartMap(
                FadeInFromPoint, dots,
                lambda m: (m, url.get_bottom()),
            ),
        )
        self.wait()

        for point in points:
            self.play(Write(point))
            if point is points[1]:
                self.play(
                    morty.change("hooray").look(ORIGIN),
                    FadeOut(BackgroundRectangle(morty, buff=0.25, fill_opacity=1))
                )
                self.play(Blink(morty))
                self.play(morty.change("tease", points[2]))
            else:
                self.wait()

        # Discuss winner
        winner_word = TexText("``Winners''", font_size=72)[0]
        winner_word.move_to(self.hold_up_spot, DOWN)
        pre_winner_word = points[1].get_part_by_text("winner").copy()

        self.students.flip().flip()

        self.add(self.background, pre_winner_word, *self.pi_creatures)
        self.play(
            LaggedStartMap(FadeOut, VGroup(
                url, dots, points,
            )),
            FadeIn(self.background),
            morty.change("hesitant", winner_word),
            FadeTransform(pre_winner_word, winner_word[2:9]),
            LaggedStartMap(FadeIn, self.students),
        )
        self.play(
            Write(winner_word[:2]),
            Write(winner_word[-2:]),
            self.change_students("sassy", "raise_right_hand", "raise_left_hand"),
        )
        self.wait(4)

        # Spirit of the event (start at 25)
        salt = Text("Take with a grain of salt")
        salt.to_edge(UP)
        strange = Text("(what a strange phrase...)", font_size=20)
        strange.set_color(GREY_B)
        strange.next_to(salt, RIGHT)
        arrow = Arrow(salt, winner_word)

        self.play(
            FadeIn(salt, lag_ratio=0.1),
            ShowCreation(arrow),
            self.change_students(
                "pondering", "pondering", "erm",
                look_at=salt,
            )
        )
        self.play(
            Blink(self.students[2]),
            FadeIn(strange, lag_ratio=0.1),
        )
        self.play(FadeOut(strange))
        self.wait()

        spirit = Text("The spirit of the event")
        spirit.to_edge(UP)
        bubble = ThoughtBubble(height=3, width=3)
        bubble.pin_to(self.students[0])
        bubble.shift(SMALL_BUFF * UR)
        bubble.add_content(Tex(
            r"|fg|_1 \leq |f|_p |g|_q",
            tex_to_color_map={
                "f": GREEN,
                "g": TEAL,
            }
        ))
        bubble.add(bubble.content)
        arrow.target = Arrow(spirit, bubble)
        video = VideoIcon()
        video.set_height(0.7)
        video.next_to(self.students, UP, MED_LARGE_BUFF).to_edge(LEFT)
        video.set_color(GREY_C)
        video.set_gloss(1)

        self.play(
            FadeOut(salt, 0.5 * UP),
            FadeIn(spirit, 0.5 * UP),
        )
        self.play(
            MoveToTarget(arrow),
            FadeIn(bubble),
            self.students[0].change("thinking", bubble),
            self.students[1].change("pondering", bubble),
            self.students[2].change("pondering", bubble),
            self.teacher.change("happy", bubble),
            winner_word.animate.scale(0.3).set_opacity(0.5).to_corner(UR),
        )
        self.play(
            self.students[0].change("raise_left_hand", video),
            FadeIn(video, 0.5 * UP),
        )
        self.wait()

        bubble_copies = bubble.replicate(2).scale(0.7)
        for bc, student in zip(bubble_copies, self.students[1:]):
            bc.move_to(student.get_corner(UR), DL)

        self.play(
            LaggedStart(*(
                TransformFromCopy(bubble, bc)
                for bc in bubble_copies
            ), lag_ratio=0.5),
            LaggedStart(*(
                student.change("thinking", UR)
                for student in self.students[1:]
            ), lag_ratio=0.5),
            run_time=2,
        )
        self.wait(3)


class ProsConsOfContext(Scene):
    def construct(self):
        title = Text("Framing as a contest", font_size=60)
        title.to_edge(UP)

        h_line = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        h_line.next_to(title, DOWN)
        v_line = Line(h_line.get_center(), FRAME_HEIGHT * DOWN / 2)

        pro_label = Text("Pros", font_size=60)
        pro_label.set_color(GREEN)
        pro_label.next_to(h_line, DOWN)
        pro_label.set_x(-FRAME_WIDTH / 4)
        con_label = Text("Cons", font_size=60)
        con_label.set_color(RED)
        con_label.next_to(h_line, DOWN)
        con_label.set_x(FRAME_WIDTH / 4)

        self.play(
            LaggedStart(
                ShowCreation(h_line),
                ShowCreation(v_line),
                lag_ratio=0.5,
            ),
            LaggedStart(
                FadeIn(title, 0.25 * UP),
                FadeIn(pro_label),
                FadeIn(con_label),
                lag_ratio=0.5,
            )
        )
        self.wait()

        pros = VGroup(
            Text("Clear deadline"),
            Text("Extra push for quality"),
        )
        pros.set_color(GREEN_A)
        pros.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        pros.next_to(pro_label, DOWN, LARGE_BUFF)

        con = Text("Suggests that\nsuccess = winning")
        con.set_color(RED_B)
        con.match_x(con_label)
        con.align_to(pros[0], UP)

        for words in (*pros, con):
            dot = Dot()
            dot.next_to(words[0], LEFT)
            words.add_to_back(dot)
            self.play(Write(words, run_time=2))
            self.wait()

        self.embed()


class FiltrationProcess(Scene):
    def construct(self):
        total = TexText("$>1{,}200(!)$ submissions")

        hundred = TexText("$\\sim 100$")
        hundred.next_to(total, RIGHT, buff=4)
        arrow = Arrow(total, hundred, stroke_width=5)
        VGroup(total, arrow, hundred).center().to_edge(UP)

        peer_words = Text("Peer review process", font_size=36)
        peer_words.set_color(BLUE)
        peer_words.next_to(arrow, DOWN, buff=SMALL_BUFF)
        peer_subwords = TexText(
            "(actually quite interesting\\\\see the blog post)",
            font_size=24,
            fill_color=GREY_B,
        )
        peer_subwords.next_to(peer_words, DOWN, MED_SMALL_BUFF)

        self.play(FadeIn(total, lag_ratio=0.1))
        self.wait()
        self.play(
            ShowCreation(arrow),
            FadeIn(peer_words, lag_ratio=0.1)
        )
        self.play(
            FadeIn(hundred),
            FadeIn(peer_subwords, 0.25 * DOWN)
        )
        self.wait()

        # Guest judges
        guest_words = Text("Guest judges:")
        guest_words.next_to(peer_subwords, DOWN, buff=LARGE_BUFF)
        guest_words.add(Underline(guest_words))
        guest_words.to_edge(RIGHT, buff=1.5)

        guests = VGroup(
            Text("Alex Kontorovich"),
            Text("Henry Reich"),
            Text("Nicky Case"),
            Text("Tai-Danae Bradley"),
            Text("Bukard Polster"),
            Text("Mithuna Yoganathan"),
            Text("Steven Strogatz"),
        )
        guests.scale(0.75)
        guests.set_color(GREY_B)
        guests.arrange(DOWN, aligned_edge=LEFT)
        guests.next_to(guest_words, DOWN, aligned_edge=LEFT)

        self.play(
            FadeIn(guest_words),
            LaggedStartMap(
                FadeIn, guests,
                shift=0.25 * DOWN,
                lag_ratio=0.5,
                run_time=4,
            )
        )
        self.wait()


class AllVideosOrdered(Scene):
    slug_file = "some1_video_urls_ordered.txt"
    grid_size = 3
    time_per_image = 0.2

    def construct(self):
        n = self.grid_size
        N = n**2
        time_per_image = self.time_per_image
        buffer_size = 2 * N

        slugs = get_youtube_slugs(self.slug_file)
        random.shuffle(slugs)
        log.info(f"Number of slugs: {len(slugs)}")
        image_slots = ScreenRectangle().get_grid(n, n, buff=0.2)
        image_slots.set_height(FRAME_HEIGHT)
        image_slots.set_stroke(WHITE, 1)

        images = buffer_size * [None]

        self.add(image_slots)

        def update_image(image, dt):
            image.time += dt
            image.set_opacity(clip(5 * image.time / (N * time_per_image), 0, 1))

        k = 0
        for slug in ProgressDisplay(slugs):
            slot = image_slots[k % N]
            save_thumbnail_locally(slug)
            file = yt_slug_to_image_file(slug)
            if not os.path.exists(file):
                continue
            if k > buffer_size:
                old_image = images[k % buffer_size]
                self.camera.release_texture(old_image.path)
                self.remove(old_image)

            image = ImageMobject(file)
            image.replace(slot, stretch=True)

            images[k % buffer_size] = image

            image.time = 0
            image.add_updater(update_image)
            self.add(image)
            self.wait(time_per_image)

            k += 1

            # Hack to keep OpenGL from tracking too many textures.
            # Should be fixed more intelligently in Camera
            if k % (buffer_size) == 0:
                self.camera.n_textures = buffer_size + 2


class RevealingTiles(Scene):
    def construct(self):
        self.five_winners()
        self.honorable_mentions()

    def five_winners(self):
        # title = Text("SoME1 Winners", font_size=72)
        title = TexText("Summer of Math Exposition\\\\", "Winners", font_size=72)
        title[1].scale(2, about_edge=UP)
        title[1].set_color(YELLOW)
        title.to_edge(UP, buff=0.2)
        # 
        # title.to_edge(UP)
        subtitle = Text("(in no particular order)", font_size=36)
        subtitle.next_to(title, DOWN, buff=0.2)
        subtitle.set_color(GREY_B)

        self.add(title)

        tiles = self.get_mystery_tile().replicate(5)
        colors = color_gradient([BLUE_B, BLUE_D], 5)
        for tile, color in zip(tiles, colors):
            # tile[0].set_stroke(color, 1)
            tile[0].set_stroke(YELLOW, 2)
        tiles.set_height(2)
        tiles.arrange_in_grid(2, 3, buff=MED_SMALL_BUFF)
        tiles[3:].match_x(tiles[:3])
        tiles.set_width(FRAME_WIDTH - 2)
        # tiles.to_edge(DOWN, buff=1)
        tiles.to_edge(DOWN, buff=0.25)

        reorder = list(range(5))
        random.seed(1)
        random.shuffle(reorder)

        # Animations
        self.play(
            Write(title),
            LaggedStartMap(GrowFromCenter, tiles, lag_ratio=0.15)
        )
        self.wait()
        self.play(
            FadeIn(subtitle, lag_ratio=0.2),
            LaggedStart(*(
                ApplyMethod(
                    tile.move_to, tiles[reorder[i]],
                    path_arc=PI / 2
                )
                for i, tile in enumerate(tiles)
            ), lag_ratio=0.25, run_time=2)
        )
        self.wait()

        #
        for tile, triple in zip(tiles, WINNERS):
            self.reveal_tile(tile, *triple)

        self.title = title
        self.subtitle = subtitle
        self.tiles = tiles

    def honorable_mentions(self):
        tiles = self.tiles

        new_title = TexText("Others you'll enjoy", font_size=72)
        new_title.to_edge(UP, buff=MED_SMALL_BUFF)

        tiles.generate_target()
        new_tile = self.get_mystery_tile()
        new_tile.replace(tiles[0])
        new_tiles = Group(
            *tiles.target,
            *new_tile.replicate(len(HONORABLE_MENTIONS))
        )
        new_tiles.arrange_in_grid(5, 6, buff=0.25)
        new_tiles.set_height(6.5)
        new_tiles.next_to(new_title, DOWN)

        self.play(
            FadeTransform(self.title, new_title),
            FadeOut(self.subtitle),
            MoveToTarget(tiles),
            LaggedStartMap(
                FadeInFromPoint, new_tiles[5:],
                lambda m: (m, ORIGIN),
                lag_ratio=0.1,
                run_time=5,
            )
        )
        self.wait()

        reorder_tail = list(range(5, len(new_tiles)))
        random.shuffle(reorder_tail)
        reorder = [*range(5), *reorder_tail]
        centers = [nt.get_center().copy() for nt in new_tiles]
        for i, tile in enumerate(new_tiles):
            tile.move_to(centers[reorder[i]])

        for tile, triple in zip(new_tiles[5:], HONORABLE_MENTIONS):
            self.reveal_tile(tile, *triple)

    def reveal_tile(self, tile, title, author, image_name):
        # Flip tile
        try:
            image_path = get_full_raster_image_path(image_name)
        except IOError:
            # Try to redownload best YT slug
            save_thumbnail_locally(image_name)
            image_path = yt_slug_to_image_file(image_name)
        # Trim to be 16x9
        arr = np.array(Image.open(image_path))
        h, w, _ = arr.shape
        nh = w * 9 // 16
        if nh < h:
            trimmed_arr = arr[(h - nh) // 2:(h - nh) // 2 + nh, :, :]
            Image.fromarray(trimmed_arr).save(image_path)

        image = ImageMobject(image_path)
        image.rotate(PI, RIGHT)
        image.replace(tile, dim_to_match=1)
        image.set_max_width(tile.get_width())

        self.play(
            Rotate(image, PI, RIGHT),
            Rotate(tile, PI, RIGHT),
            UpdateFromAlphaFunc(
                tile, lambda m, a: m.set_opacity(1 if a < 0.5 else 0),
            ),
            UpdateFromAlphaFunc(
                image, lambda m, a: m.set_opacity(0 if a < 0.5 else 1),
            ),
        )
        tile.remove(tile[1])
        tile[0].set_fill(BLACK, 1)
        tile.add(image)

        # Expand
        full_rect = FullScreenRectangle()
        full_rect.set_fill(GREY_E)

        title = Text(title)
        title.set_max_width(FRAME_WIDTH - 1)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        byline = Text(f"by {author}", font_size=30)
        byline.set_color(GREY_B)
        byline.next_to(title, DOWN, buff=0.2)

        tile.save_state()
        tile.generate_target()
        tile.target.set_height(6)
        tile.target.next_to(byline, DOWN)
        tile.saved_state[0].set_stroke(opacity=1)

        self.add(full_rect, tile)
        self.play(
            FadeIn(full_rect),
            MoveToTarget(tile, run_time=2),
            FadeInFromPoint(title, tile.get_top(), run_time=2),
        )
        self.play(FadeIn(byline, 0.25 * DOWN))
        self.wait()
        self.play(
            Restore(tile),
            FadeOut(full_rect),
            FadeOut(title, 0.75 * UP),
            FadeOut(byline, 1.0 * UP),
        )

    def get_mystery_tile(self):
        rect = ScreenRectangle(height=1)
        rect.set_stroke(BLUE_D, 1)
        rect.set_fill(GREY_D, 1)
        q_marks = Tex("???")
        q_marks.flip().flip()
        q_marks.set_fill(GREY_A)
        q_marks.move_to(rect)
        q_marks.shift(1e-3 * OUT)
        return Group(rect, q_marks)


class AlmostTooGood(TeacherStudentsScene):
    def construct(self):
        self.pi_creatures.flip().flip()
        self.teacher_says(
            TexText("Almost \\emph{too} good"),
            look_at=self.students[2].eyes,
            added_anims=[self.change_students("happy", "tease", "hesitant")],
        )
        self.wait(4)


class ViewRect(Scene):
    views = 596

    def construct(self):
        rect = Rectangle(3.5, 1)
        rect.set_height(0.25)
        rect.set_stroke(RED, 3)

        number = Integer(self.views, font_size=96)
        number.set_fill(RED)
        number.next_to(rect, DOWN, LARGE_BUFF)
        number.shift(3 * RIGHT)

        arrow = Arrow(rect, number)
        arrow.set_color(RED)

        self.play(
            ShowCreation(rect)
        )

        number.set_value(0)
        self.play(
            ShowCreation(arrow),
            # FadeInFromPoint(number, arrow.get_start())
            ChangeDecimalToValue(number, self.views, run_time=2),
            VFadeIn(number),
            UpdateFromFunc(number, lambda m: m.set_fill(RED))
        )
        self.wait()


class ViewRect700k(ViewRect):
    views = 795095


class SureSure(TeacherStudentsScene):
    def construct(self):
        self.students.flip().flip()
        self.teacher_says(
            "The point is \n not the winners",
            look_at=self.students[2].eyes,
            bubble_config={"height": 3, "width": 4}
        )
        self.play(PiCreatureSays(
            self.students[0],
            "Yeah, yeah, sure\n it isn't...",
            bubble_config={"height": 2, "width": 3}
        ))
        self.play_student_changes("sassy", "angry", "hesitant")
        self.play(self.teacher.change("guilty"))
        self.wait(4)


class Narrative(Scene):
    def construct(self):
        question = Text("What makes a good piece of exposition?", font_size=60)
        question.to_edge(UP)
        question.add(Underline(question))
        self.add(question)
        properties = VGroup(
            Text("Clarity"),
            Text("Motivation"),
            Text("Memorability"),
            Text("Narrative"),
        )
        properties.scale(60 / 48)
        properties.arrange(DOWN, buff=0.75, aligned_edge=LEFT)
        properties.next_to(question, DOWN, LARGE_BUFF)
        properties.set_fill(BLUE)
        narrative = properties[-1]

        for prop in properties:
            dot = Dot()
            dot.next_to(prop, LEFT)
            self.play(
                FadeIn(dot, scale=0.5),
                FadeIn(prop, lag_ratio=0.1)
            )

        rect = FullScreenFadeRectangle()
        rect.set_opacity(0.5)
        self.add(rect, narrative)
        self.play(
            FadeIn(rect),
            FlashAround(narrative, run_time=2, time_width=1),
            narrative.animate.set_color(YELLOW),
        )
        self.wait()


class Traction(Scene):
    def construct(self):
        slug_count_pairs = [
            ("PKjbBQ0PBCQ", "$\\sim$800,000"),
            ("ACZC_XEyg9U", "$\\sim$100,000"),
            ("4y_nmpv-9lI", "$\\sim$140,000"),
            ("cCKOl5li6YM", "$\\sim$400,000"),
        ]
        groups = Group()
        for slug, count in slug_count_pairs:
            rect = ScreenRectangle()
            rect.set_height(3)
            rect.set_stroke(BLUE, 2)
            image = ImageMobject(yt_slug_to_image_file(slug))
            image.replace(rect, 1)
            text = TexText(f"{count} views")
            text.next_to(image, DOWN, MED_SMALL_BUFF)
            groups.add(Group(image, rect, text))

        groups.arrange_in_grid(v_buff=MED_LARGE_BUFF, h_buff=LARGE_BUFF)
        groups.set_height(FRAME_HEIGHT - 1)

        for elem in groups:
            self.play(
                FadeIn(elem[:2]),
                FadeIn(elem[-1], 0.5 * DOWN, scale=2)
            )
        self.wait()


class EndScreen(PatreonEndScreen):
    pass
