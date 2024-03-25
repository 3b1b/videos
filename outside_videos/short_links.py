from manim_imports_ext import *


def get_vertical_screen():
    screen = Rectangle(width=FRAME_HEIGHT * (9 / 16), height=FRAME_HEIGHT)
    screen.center()
    return screen


class CommentFlurry(InteractiveScene):
    base_folder = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/shorts/Custom Audio/lower_link/"
    sub_folder = ""
    random_seed = 4

    def construct(self):
        # Test
        folder = os.path.join(self.base_folder, self.sub_folder)
        images = Group()
        screen = get_vertical_screen()
        files = os.listdir(folder)

        for file in files:
            if not file.endswith(".png"):
                continue
            image = ImageMobject(os.path.join(folder, file))
            image.match_width(screen)
            image.scale(random.uniform(0.6, 0.8))
            image.move_to([
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-2, 2),
                0
            ])
            images.add(image)

        groups = Group()
        for image in images:
            rect = BackgroundRectangle(image, buff=0.025)
            rect.set_fill(BLACK, 0.75)
            group = Group(rect, image)
            groups.add(group)

        self.play(LaggedStartMap(FadeIn, groups, lag_ratio=0.5, shift=0.25 * UP, run_time=4))

        self.wait()


class CommentFlurryLinks(CommentFlurry):
    sub_folder = "OnLinks"


class CommentFlurryBlocks(CommentFlurry):
    sub_folder = "OnBlockCollisions"


class CommentFlurryPrisms(CommentFlurry):
    sub_folder = "OnPrisms"


class CommentFlurryFourier(CommentFlurry):
    sub_folder = "OnFourier"
    random_seed = 1


class CommentFlurryOtherQuestions(CommentFlurry):
    sub_folder = "OtherQuestions"


class PiLookingAtPhone(InteractiveScene):
    def construct(self):
        # Test
        screen = get_vertical_screen()
        # self.add(screen)

        randy = Randolph(height=2)
        randy.move_to(screen, LEFT)
        randy.shift(DOWN + 0.2 * RIGHT)
        phone = SVGMobject("iPhone")
        phone.set_height(3)
        phone.stretch(1.2, 0)
        phone.set_fill(GREY_A)
        phone.next_to(randy.get_corner(UR), RIGHT)

        self.add(randy)
        self.add(phone)

        for mode in ["pondering", "confused", "maybe", "tease"]:
            self.play(randy.change(mode, phone))
            self.play(Blink(randy))
            self.wait()


class ThisIsALink(InteractiveScene):
    def construct(self):
        # Test
        rect = Rectangle(3.5, 0.35)
        rect.set_stroke(YELLOW, 3)
        rect.insert_n_curves(100)
        rect.move_to(2 * UP)
        words = Text("This is\na link!", font_size=60)
        words.next_to(rect, DOWN, buff=2)
        arrow = Arrow(words, rect)
        arrow.shift(0.1 * UP)
        VGroup(words, arrow).set_color(YELLOW)

        self.add(words)
        self.play(
            VShowPassingFlash(rect.copy().set_stroke(width=5), time_width=1.5),
            ShowCreation(rect),
            ShowCreation(arrow),
        )
        self.wait()


class LinkHighlightOverlay(InteractiveScene):
    def construct(self):
        # # Test
        # image = ImageMobject("/Users/grant/Desktop/ShortSample.png")
        # image.set_height(FRAME_HEIGHT)
        # self.add(image)

        # Rect
        rect = VMobject()
        rect.set_points_as_corners([UR, UL, DL, DR])
        rect.set_shape(3.5, 0.5)
        rect.move_to([-2.125, -3.315,  0.], LEFT).shift(0 * UP)
        rect.insert_n_curves(100)
        rect.set_stroke(YELLOW, 4)

        self.play(VShowPassingFlash(rect, time_width=1.5, run_time=3))
