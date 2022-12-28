from manim_imports_ext import *


class AmbientPermutations(Scene):
    def construct(self):
        # Test
        text = Text("abcde", font_size=72)
        text.arrange_to_fit_width(4.5)
        self.add(text)

        # Animate swaps
        n_swaps = 10
        perms = list(it.permutations(range(len(text))))

        for x in range(n_swaps):
            perm = random.choice(perms)
            letter_anims = []
            arrow_anims = []
            for i, letter in enumerate(text):
                target = text[perm[i]]
                letter_anims.append(letter.animate.move_to(target, DOWN).set_anim_args(path_arc=90 * DEGREES))
                if i < perm[i]:
                    arrow = Arrow(letter.get_bottom(), target.get_bottom(), path_arc=90 * DEGREES)
                elif i > perm[i]:
                    arrow = Arrow(letter.get_top(), target.get_top(), path_arc=90 * DEGREES)
                else:
                    arrow = VMobject()
                arrow.set_stroke(BLUE)
                arrow_anims.append(ShowCreationThenFadeOut(arrow, run_time=2))

            self.play(LaggedStart(*arrow_anims, *letter_anims, lag_ratio=0.05))
            text.sort()


class WriteName(InteractiveScene):
    def construct(self):
        self.play(Write(
            Text("Évariste Galois", font_size=90).to_corner(UL),
            run_time=3
        ))
        self.wait()


class TimelineTransition(InteractiveScene):
    def construct(self):
        pass


class OutpaintTransition(InteractiveScene):
    image_path = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2022/galois/artwork/chapter 1/wounded-by-pond/transition-to-hospital-2.png"

    def construct(self):
        image = ImageMobject(self.image_path)
        image.set_height(FRAME_HEIGHT)
        image.to_edge(LEFT, buff=0)
        self.add(image)

        # Pan
        self.play(
            image.animate.to_edge(RIGHT, buff=-1),
            run_time=8,
            rate_func=bezier([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        )
        image.to_edge(RIGHT, buff=-1)


class NightSkyOutpaintingTransition(InteractiveScene):
    image_path = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2022/galois/artwork/chapter 1/transition-to-night-sky/mural.png"

    def construct(self):
        # Add image
        image = ImageMobject(self.image_path)
        image.set_width(FRAME_WIDTH + 6)
        image.to_edge(DL, buff=0)
        image.shift(2 * DOWN + LEFT)
        self.add(image)

        # Pan
        image.generate_target()
        image.target.set_width(FRAME_WIDTH)
        image.target.center().to_edge(UP, buff=0)

        self.play(
            MoveToTarget(image),
            run_time=12,
            rate_func=bezier([0, 0, 0, 1, 1, 1])
        )
        self.wait()


class LastWordsQuote(InteractiveScene):
    def construct(self):
        # French quote
        fr_quote = Text(
            """
            Ne pleure pas, Alfred! J'ai besoin de
            tout mon courage pour mourir à vingt ans!
            """,
            alignment="LEFT",
            font="Better Grade",
            font_size=72
        )
        fr_quote.to_corner(UR)
        self.add(fr_quote)

        # Write english quote
        en_quote = Text(
            """
            Do not cry, Alfred! I
            need all my courage to die
            at twenty years of age!
            """,
            alignment="LEFT",
        )
        en_quote.match_width(fr_quote)
        en_quote.next_to(fr_quote, DOWN, buff=1)
        self.play(FadeIn(en_quote, lag_ratio=0.1, run_time=6))
        self.wait()


class InfamousCoquette(InteractiveScene):
    def construct(self):
        # Write quote
        quote = OldTexText(
            "``I die the victim of an infamous coquette and her two dupes.''",
        )
        quote.to_edge(UP)
        quote.set_width(FRAME_WIDTH - 1)
        quote.set_stroke(BLACK, 0)
        quote.set_fill(BLACK)
        self.play(Write(quote, run_time=3))
        self.wait()


class NightBeforeQuote(InteractiveScene):
    def construct(self):
        # Write
        quote = OldTexText("""
            ``You will publicly ask Jacobi\\\\
            or Gauss to give their opinion\\\\
            not on the truth but on the\\\\
            importance of the theorems.''
        """, alignment="")[0]

        quote.to_edge(RIGHT)
        self.play(FadeIn(quote, run_time=2, lag_ratio=0.025))
        self.wait()
        self.play(quote[33:-2].animate.set_color(YELLOW).set_anim_args(lag_ratio=0.5, run_time=3))
        self.wait(2)


class CauchyFourierPoisson(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle(fill_color="#211F22", fill_opacity=1))
        folder = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2022/galois/artwork/chapter 1/Portraits/"
        images = Group(
            ImageMobject(os.path.join(folder, "Cauchy-Portrait-Cutoff")),
            ImageMobject(os.path.join(folder, "Fourier-Portrait")),
            ImageMobject(os.path.join(folder, "Poisson-Portrait-Outpainted")),
        )
        for image in images:
            image.set_width(4)
        images[0].set_width(3.5)
        images.arrange(RIGHT, aligned_edge=DOWN)
        images.set_width(FRAME_WIDTH - 1)
        images.to_edge(UP, buff=LARGE_BUFF)

        # Names
        names = VGroup(
            Text("Augustin Cauchy"),
            Text("Joseph Fourier"),
            Text("Siméon Poisson"),
        )
        for name, image in zip(names, images):
            name.next_to(image, DOWN)
            name.set_fill(GREY_A)
        names[1:].shift(SMALL_BUFF * RIGHT)

        # Animations
        kw = dict(lag_ratio=0.5, run_time=3)
        self.play(
            LaggedStart(*(FadeIn(image, 0.5 * UP, scale=1.1) for image in images), **kw),
            LaggedStart(*(Write(name, rate_func=squish_rate_func(smooth, 0, 0.85)) for name in names), **kw),
        )
        self.wait()
