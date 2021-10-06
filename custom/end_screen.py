import random
import os
import json

from manimlib.animation.animation import Animation
from manimlib.animation.composition import Succession
from manimlib.animation.transform import ApplyMethod
from manimlib.constants import *
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.geometry import DashedLine
from manimlib.mobject.geometry import Line
from manimlib.mobject.geometry import Rectangle
from manimlib.mobject.geometry import Square
from manimlib.mobject.svg.tex_mobject import TexText
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.scene.scene import Scene
from manimlib.utils.directories import get_directories
from manimlib.utils.rate_functions import linear

from custom.characters.pi_creature import Mortimer
from custom.characters.pi_creature import Randolph
from custom.characters.pi_creature_animations import Blink


class PatreonEndScreen(Scene):
    CONFIG = {
        "title_text": "Clicky Stuffs",
        "show_pis": True,
        "max_patron_group_size": 20,
        "patron_scale_val": 0.8,
        "n_patron_columns": 4,
        "max_patron_width": 5,
        "randomize_order": False,
        "capitalize": True,
        "name_y_spacing": 0.6,
        "thanks_words": "Funded by viewers, visit 3b1b.co/support to learn more",
        "scroll_time": 20,
    }

    def construct(self):
        # Add title
        title = self.title = TexText(self.title_text)
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

        # Set the top of the screen
        logo_box = Square(side_length=2.5)
        logo_box.to_corner(DOWN + LEFT, buff=MED_LARGE_BUFF)

        black_rect = Rectangle(
            fill_color=BLACK,
            fill_opacity=1,
            stroke_width=3,
            stroke_color=BLACK,
            width=FRAME_WIDTH,
            height=0.6 * FRAME_HEIGHT,
        )
        black_rect.to_edge(UP, buff=0)
        line = DashedLine(FRAME_X_RADIUS * LEFT, FRAME_X_RADIUS * RIGHT)
        line.move_to(ORIGIN)

        # Add thanks
        thanks = TexText(self.thanks_words)
        thanks.scale(0.9)
        thanks.next_to(black_rect.get_bottom(), UP, SMALL_BUFF)
        thanks.set_color(YELLOW)
        underline = Line(LEFT, RIGHT)
        underline.match_width(thanks)
        underline.scale(1.1)
        underline.next_to(thanks, DOWN, SMALL_BUFF)
        thanks.add(underline)

        # Build name list
        names = self.get_names()

        name_labels = VGroup(*map(TexText, names))
        name_labels.scale(self.patron_scale_val)
        for label in name_labels:
            if label.get_width() > self.max_patron_width:
                label.set_width(self.max_patron_width)
        columns = VGroup(*[
            VGroup(*name_labels[i::self.n_patron_columns])
            for i in range(self.n_patron_columns)
        ])
        column_x_spacing = 0.5 + max([c.get_width() for c in columns])

        for i, column in enumerate(columns):
            for n, name in enumerate(column):
                name.shift(n * self.name_y_spacing * DOWN)
                name.align_to(ORIGIN, LEFT)
            column.move_to(i * column_x_spacing * RIGHT, UL)
        columns.center()

        max_width = FRAME_WIDTH - 1
        if columns.get_width() > max_width:
            columns.set_width(max_width)
        underline.match_width(columns)
        columns.next_to(underline, DOWN, buff=3)

        # Set movement
        columns.generate_target()
        distance = columns.get_height() + 2
        wait_time = self.scroll_time
        frame = self.camera.frame
        frame_shift = ApplyMethod(
            frame.shift, distance * DOWN,
            run_time=wait_time,
            rate_func=linear,
        )
        blink_anims = []
        blank_mob = Mobject()
        for x in range(wait_time):
            if random.random() < 0.25:
                blink_anims.append(Blink(random.choice(pi_creatures)))
            else:
                blink_anims.append(Animation(blank_mob))
        blinks = Succession(*blink_anims)

        static_group = VGroup(black_rect, line, thanks, pi_creatures, title)
        static_group.fix_in_frame()
        self.add(columns, static_group)
        self.play(frame_shift, blinks)

    def get_names(self):
        patron_file = "patrons.txt"
        hardcoded_patron_file = "hardcoded_patrons.txt"
        names = []
        for file in patron_file, hardcoded_patron_file:
            full_path = os.path.join(get_directories()["data"], file)
            with open(full_path, "r") as fp:
                names.extend([
                    self.modify_patron_name(name.strip())
                    for name in fp.readlines()
                ])
        # Remove duplicates
        names = list(set(names))

        # Make sure these aren't missed
        if self.randomize_order:
            random.shuffle(names)
        else:
            names.sort()
        return names

    def modify_patron_name(self, name):
        path = os.path.join(
            get_directories()['data'],
            "patron_name_replacements.json"
        )
        with open(path) as fp:
            modification_map = json.load(fp)

        for n1, n2 in modification_map.items():
            name = name.replace("ā", "\\={a}")
            if name.lower() == n1.lower():
                name = n2
        if self.capitalize:
            name = " ".join(map(
                lambda s: s.capitalize(),
                name.split(" ")
            ))
        return name
