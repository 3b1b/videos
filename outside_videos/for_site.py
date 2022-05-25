from manim_imports_ext import *


class WhyPi(Scene):
    def construct(self):
        title = TexText("Why $\\pi$?")
        title.scale(3)
        title.to_edge(UP)

        formula1 = Tex(
            "1 +"
            "\\frac{1}{4} +"
            "\\frac{1}{9} +"
            "\\frac{1}{16} +"
            "\\frac{1}{25} + \\cdots"
            "=\\frac{\\pi^2}{6}"
        )
        formula1.set_color(YELLOW)
        formula1.set_width(FRAME_WIDTH - 2)
        formula1.next_to(title, DOWN, MED_LARGE_BUFF)

        formula2 = Tex(
            "1 -"
            "\\frac{1}{3} +"
            "\\frac{1}{5} -"
            "\\frac{1}{7} +"
            "\\frac{1}{9} - \\cdots"
            "=\\frac{\\pi}{4}"
        )
        formula2.set_color(BLUE_C)
        formula2.set_width(FRAME_WIDTH - 2)
        formula2.next_to(formula1, DOWN, LARGE_BUFF)

        self.add(title)
        self.add(formula1)
        self.add(formula2)


class GeneralExpositionIcon(Scene):
    def construct(self):
        title = TexText("What is \\underline{\\qquad \\qquad}?")
        title.scale(3)
        title.to_edge(UP)
        randy = Randolph()
        randy.change("pondering")
        randy.set_height(4.5)
        randy.to_edge(DOWN)
        randy.look_at(title[0][0])

        self.add(title)
        self.add(randy)


class GeometryIcon(Scene):
    def construct(self):
        im = ImageMobject("geometry_icon_base.jpg")
        im.set_height(FRAME_HEIGHT)
        im.scale(0.9, about_edge=DOWN)
        word = TexText("Geometry")
        word.scale(3)
        word.to_edge(UP)
        self.add(im, word)


class PhysicsIcon(Scene):
    def construct(self):
        im = ImageMobject("physics_icon_base.png")
        im.set_height(FRAME_HEIGHT)
        im.shift(UP)
        title = TexText("Physics")
        title.scale(3)
        title.to_edge(UP)

        self.add(im)
        self.add(title)


class SupportIcon(Scene):
    def construct(self):
        randy = Randolph(mode="coin_flip_2")
        morty = Mortimer(mode="gracious")
        pis = VGroup(randy, morty)
        pis.arrange(RIGHT, buff=3)
        pis.to_edge(DOWN)
        randy.make_eye_contact(morty)
        heart = SuitSymbol("hearts")
        heart.set_height(1)
        heart.next_to(randy, UR, buff=-0.5)
        heart.shift(0.5 * RIGHT)

        # rect = FullScreenFadeRectangle(opacity=0.85)

        # self.add(rect)
        self.add(pis)
        self.add(heart)


class SupportPitch1(Scene):
    CONFIG = {
        "camera_config": {
            "background_opacity": 0.85,
        },
        "mode1": "happy",
        "mode2": "hooray",
        "words1": "So what do\\\\you do?",
        "words2": "Oh, I make\\\\videos about\\\\math.",
    }

    def construct(self):
        randy = Randolph()
        randy.to_corner(DL)
        morty = Mortimer()
        morty.to_corner(DR)

        randy.change(self.mode1, morty.eyes)
        morty.change(self.mode2, randy.eyes)

        b1 = randy.get_bubble(
            self.words1,
            bubble_type=SpeechBubble,
            height=3,
            width=4,
        )
        b1.add(b1.content)
        b1.shift(0.25 * UP)
        b2 = morty.get_bubble(
            self.words2,
            bubble_type=SpeechBubble,
            height=3,
            width=4,
        )
        # b2.content.scale(0.9)
        b2.add(b2.content)
        b2.shift(0.25 * DOWN)

        self.add(randy)
        self.add(morty)
        self.add(b2)
        self.add(b1)


class SupportPitch2(SupportPitch1):
    CONFIG = {
        "mode1": "confused",
        "mode2": "speaking",
        "words1": "Wait, how does\\\\that work?",
        "words2": "People pay\\\\for them.",
    }


class SupportPitch3(SupportPitch1):
    CONFIG = {
        "mode1": "hesitant",
        "mode2": "coin_flip_2",
        "words1": "Oh, so like\\\\a paid course?",
        "words2": "Well, no,\\\\everything\\\\is free.",
    }


class SupportPitch4(SupportPitch1):
    CONFIG = {
        "mode1": "confused",
        "mode2": "hesitant",
        "words1": "Wait, what?",
        "words2": "I know,\\\\it's weird...",
    }


class RantPage(Scene):
    CONFIG = {
    }

    def construct(self):
        squares = VGroup(Square(), Square())
        squares.arrange(DOWN, buff=MED_SMALL_BUFF)
        squares.set_height(FRAME_HEIGHT - 0.5)
        squares.set_width(5, stretch=True)
        squares.set_stroke(WHITE, 2)
        squares.set_fill(BLACK, opacity=0.75)
        s1, s2 = squares

        # Group1
        morty = Mortimer(mode="maybe")
        for eye, pupil in zip(morty.eyes, morty.pupils):
            pupil.move_to(eye)
        morty.shift(MED_SMALL_BUFF * UL)
        words = TexText(
            "What were you\\\\expecting to be here?"
        )
        bubble = SpeechBubble(direction=RIGHT)
        bubble.match_style(s1)
        bubble.add_content(words)
        bubble.resize_to_content()
        bubble.add(bubble.content)
        bubble.pin_to(morty)
        group1 = VGroup(morty, bubble)
        group1.set_height(s1.get_height() - MED_SMALL_BUFF)
        group1.next_to(s1.get_corner(DR), UL, SMALL_BUFF)

        # Group 2
        morty = Mortimer(mode="surprised")
        morty.shift(MED_SMALL_BUFF * UL)
        words = TexText(
            "Go on!\\\\Give the rant!"
        )
        bubble = SpeechBubble(direction=RIGHT)
        bubble.match_style(s1)
        bubble.add_content(words)
        bubble.resize_to_content()
        bubble.add(bubble.content)
        bubble.pin_to(morty)
        group2 = VGroup(morty, bubble)
        group2.set_height(s2.get_height() - MED_SMALL_BUFF)
        group2.next_to(s2.get_corner(DR), UL, SMALL_BUFF)

        self.add(squares)
        self.add(group1)
        self.add(group2)


class ClipsLogo(Scene):
    def construct(self):
        logo = Logo()
        logo.set_height(FRAME_HEIGHT - 0.5)
        square = Square(stroke_width=0, fill_color=BLACK, fill_opacity=1)
        square.scale(5)
        square.rotate(45 * DEGREES)
        square.move_to(ORIGIN, LEFT)
        self.add(logo, square)
