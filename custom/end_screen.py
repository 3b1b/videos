import random
import os
import json
from tqdm import tqdm as ProgressDisplay
from pathlib import Path

from manimlib.animation.animation import Animation
from manimlib.animation.composition import Succession
from manimlib.animation.composition import LaggedStartMap
from manimlib.animation.composition import AnimationGroup
from manimlib.animation.creation import Write
from manimlib.animation.transform import ApplyMethod
from manimlib.animation.indication import VShowPassingFlash
from manimlib.animation.fading import FadeIn
from manimlib.animation.fading import FadeOut
from manimlib.animation.fading import VFadeIn
from manimlib.constants import *
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.frame import FullScreenRectangle
from manimlib.mobject.frame import ScreenRectangle
from manimlib.mobject.geometry import Circle
from manimlib.mobject.geometry import DashedLine
from manimlib.mobject.geometry import Line
from manimlib.mobject.geometry import Rectangle
from manimlib.mobject.geometry import Square
from manimlib.mobject.types.image_mobject import ImageMobject
from manimlib.mobject.svg.tex_mobject import TexText
from manimlib.mobject.svg.text_mobject import Text
from manimlib.mobject.svg.drawings import SpeechBubble
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.scene.scene import Scene
from manimlib.scene.interactive_scene import InteractiveScene
from manimlib.utils.directories import get_directories
from manimlib.utils.rate_functions import linear
from manimlib.utils.iterables import remove_list_redundancies
from manimlib.utils.space_ops import midpoint

from custom.characters.pi_creature import Mortimer
from custom.characters.pi_creature import Randolph
from custom.characters.pi_creature_animations import Blink


class PatreonEndScreen(InteractiveScene):
    title_text = "Clicky Stuffs"
    show_pis = True
    max_patron_group_size = 20
    patron_scale_val = 0.8
    n_patron_columns = 4
    max_patron_width = 5
    initial_names_top_y = -3.5
    final_names_bottom_y = 0
    randomize_order = False
    sort_alphabetically = False
    capitalize = True
    thanks_words = """
        These videos are unsponsored, instead funded by viewers.
        Special thanks to the ones listed below | 3b1b.co/support
    """
    scroll_time = 20

    def construct(self):
        # Set the top of the screen
        black_rect = Rectangle(
            fill_color=BLACK,
            fill_opacity=1,
            stroke_width=3,
            stroke_color=BLACK,
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
        )
        black_rect.to_edge(UP, buff=0)
        line = DashedLine(FRAME_X_RADIUS * LEFT, FRAME_X_RADIUS * RIGHT)
        line.move_to(ORIGIN)
        self.add(black_rect, line)

        # Add title
        title = self.title = Text(self.title_text)
        title.scale(1.5)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        self.add(title)

        pi_creatures = VGroup(Randolph(), Mortimer())
        for pi, vect in zip(pi_creatures, [LEFT, RIGHT]):
            pi.set_height(title.get_height())
            pi.change_mode("thinking")
            pi.look(DOWN)
            pi.next_to(title, vect, buff=MED_LARGE_BUFF)
        self.add(pi_creatures)
        if not self.show_pis:
            pi_creatures.set_opacity(0)

        # Add thanks
        thanks = Text(self.thanks_words)
        thanks.scale(0.8)
        thanks.next_to(line, DOWN, buff=MED_SMALL_BUFF)
        thanks.set_color(GREY_A)
        thanks["3b1b.co/support"].set_color(YELLOW)
        underline = Line(LEFT, RIGHT)
        underline.set_stroke(WHITE, 1)
        underline.match_width(thanks)
        underline.next_to(thanks, DOWN, MED_SMALL_BUFF)
        thanks.add(underline)
        black_rect.match_y(underline, DOWN)
        self.add(thanks, underline)

        # Build name list
        name_labels = self.get_name_mobjects()
        underline.match_width(name_labels)

        # Set movement
        self.scroll(name_labels, self.get_blinks(pi_creatures))

    def get_names(self):
        data_path = get_directories()["data"]

        # Get raw name list
        patron_path = Path(data_path, "top_patrons.txt")
        hardcoded_patron_path = Path(data_path, "hardcoded_patrons.txt")
        names = [
            *patron_path.read_text().split("\n"),
            *hardcoded_patron_path.read_text().split("\n"),
        ]

        # Apply substitutions
        substitution_path = Path(data_path, "patron_name_replacements.json")

        def normalize_name(name):
            return name.strip().lower()

        substitution_map = dict(
            (normalize_name(key), value)
            for key, value in json.loads(substitution_path.read_text()).items()
        )
        names = [
            substitution_map.get(normalize_name(name), name)
            for name in names
        ]

        # Remove duplicates and Anonymous
        names = list(dict.fromkeys(names))
        if "Anonymous" in names:
            names.remove("Anonymous")

        # Clean and sort
        if self.capitalize:
            names = [
                " ".join([s.capitalize() for s in name.split(" ")])
                for name in names
            ]
        if self.randomize_order:
            random.shuffle(names)
        if self.sort_alphabetically:
            names.sort()
        return names

    def get_name_mobjects(self, x=0, max_width=FRAME_WIDTH - 1):
        names = self.get_names()

        name_labels = VGroup(*map(
            Text, ProgressDisplay(names, leave=False, desc="Writing names")
        ))
        name_labels.scale(self.patron_scale_val)
        for label in name_labels:
            label.set_max_width(self.max_patron_width)

        name_labels.arrange_in_grid(
            n_cols=self.n_patron_columns,
            aligned_edge=LEFT,
            # V buff?
        )
        name_labels.set_max_width(max_width)
        name_labels.set_x(x)
        name_labels.set_y(self.initial_names_top_y, UP)

        return name_labels

    def get_blinks(self, pi_creatures):
        blink_anims = []
        blank_mob = Mobject()
        for x in range(self.scroll_time):
            if random.random() < 0.25:
                blink_anims.append(Blink(random.choice(pi_creatures)))
            else:
                blink_anims.append(Animation(blank_mob))
        return Succession(*blink_anims)

    def scroll(self, name_labels, *added_anims):
        for mobject in self.mobjects:
            mobject.fix_in_frame()
        for anim in added_anims:
            for mob in anim.mobject.get_family():
                anim.mobject.fix_in_frame()
        name_labels.unfix_from_frame()
        name_labels.set_z_index(-1)
        self.add(name_labels)

        # Frame shift
        distance = self.final_names_bottom_y - self.initial_names_top_y + name_labels.get_height()
        frame_shift = ApplyMethod(
            self.frame.shift, distance * DOWN,
            run_time=self.scroll_time,
            rate_func=linear,
        )

        self.play(frame_shift, *added_anims)


class SideScrollEndScreen(PatreonEndScreen):
    background_color = BLACK
    line_color = WHITE
    video_element_width = 4.6
    link_element_width = 2.2
    scroll_time = 21
    initial_buff_above_names = 6
    patron_scale_val = 0.5
    n_patron_columns = 2
    max_patron_width = 2
    thanks_text = "Special Thanks to\nthese channel supporters"
    thanks_style = dict(
        fill_color=YELLOW,
        font_size=36,
    )
    thanks_panel_margin = 0.25
    early_view_text = """
        Hey, psst, channel
        supporters get early
        views of new videos.
    """
    support_url = "https://3b1b.co/support"

    def construct(self):
        self.add_background_rectangle()
        self.add_element_outlines()
        self.add_thanks_and_names()

        # Scroll
        self.scroll(
            self.name_labels,
            LaggedStartMap(
                VShowPassingFlash,
                self.elements.copy().set_stroke(WHITE, 5).insert_n_curves(100),
                time_width=2,
                run_time=3,
                lag_ratio=0.25
            ),
            FadeIn(self.elements, time_span=(1, 2)),
            Write(self.link_text),
            self.get_early_view_comment(),
        )

    def add_background_rectangle(self):
        background_rect = FullScreenRectangle(fill_color=self.background_color)
        background_rect.set_z_index(-2)
        self.add(background_rect)

        panels = VGroup(
            ScreenRectangle().stretch(2 / 5, 0),
            ScreenRectangle().stretch(3 / 5, 0),
        )
        panels.arrange(RIGHT, buff=0)
        panels.set_stroke(width=0)
        panels.replace(background_rect, stretch=True)

        self.panels = panels

    def add_element_outlines(self):
        screen, channel, link = elements = VGroup(
            ScreenRectangle().set_width(self.video_element_width),
            Circle().set_width(self.link_element_width),
            Square().set_width(self.link_element_width),
        )
        screen.round_corners(radius=0.1)
        link.next_to(screen, RIGHT, MED_SMALL_BUFF)
        channel.next_to(screen, DOWN, buff=0.75, aligned_edge=LEFT)

        elements.move_to(self.panels[1])
        elements.set_stroke(self.line_color, 0)
        self.elements = elements

        self.add(self.elements)

        # Link text
        link_text = Text(self.support_url, font_size=30)
        link_text.next_to(link, UP)
        self.link_text = link_text
        self.add(link_text)

    def get_early_view_comment(self):
        link = self.elements[2]
        randy = Randolph(height=1.5)
        randy.flip()
        randy.to_corner(DR).shift(0.25 * UP)
        randy.always.fix_in_frame()  # Why?
        blank = VMobject()

        bubble = randy.get_bubble(
            Text(self.early_view_text, font_size=20),
            SpeechBubble,
            direction=RIGHT
        )
        bubble.fix_in_frame()

        return Succession(
            AnimationGroup(
                Write(bubble),
                randy.change("tease", self.link_text),
            ),
            Blink(randy),
            Animation(blank, run_time=1),
            bubble.animate.set_opacity(0),
            randy.change("raise_left_hand", look_at=self.link_text).set_anim_args(run_time=1),
            Animation(blank, run_time=2),
            Blink(randy),
            Animation(blank, run_time=3),
            Blink(randy),
            Animation(blank, run_time=2),
            Blink(randy),
            Animation(blank, run_time=3),
            Blink(randy),
        )

    def add_thanks_and_names(self):
        v_line = Line(DOWN, UP).set_height(FRAME_HEIGHT)
        v_line.move_to(self.panels[1].get_left())
        v_line.set_stroke(self.line_color, 1)

        thanks = Text(self.thanks_text, **self.thanks_style)
        thanks_panel = self.panels[0].copy()
        thanks_panel.set_fill(BLACK, 1)
        thanks_panel.set_height(
            thanks.get_height() + 2 * self.thanks_panel_margin,
            stretch=True,
            about_edge=UP,
        )
        thanks.move_to(thanks_panel)

        h_line = Line(LEFT, RIGHT)
        h_line.match_width(thanks_panel)
        h_line.move_to(thanks_panel, DOWN)
        h_line.set_stroke(WHITE, 1)

        name_labels = self.get_name_mobjects(
            x=thanks_panel.get_x(),
            max_width=thanks_panel.get_width() - 1
        )
        name_labels.set_z_index(-1)
        self.name_labels = name_labels

        self.add(thanks_panel, h_line, v_line)
        self.add(thanks, name_labels)
