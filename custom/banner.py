from manimlib.constants import *
from manimlib.mobject.coordinate_systems import NumberPlane
from manimlib.mobject.svg.tex_mobject import TexText
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.frame import FullScreenFadeRectangle
from manimlib.scene.scene import Scene

from custom.characters.pi_creature import Mortimer
from custom.characters.pi_creature import Randolph


class Banner(Scene):
    CONFIG = {
        "camera_config": {
            "pixel_height": 1440,
            "pixel_width": 2560,
        },
        "pi_height": 1.25,
        "pi_bottom": 0.25 * DOWN,
        "use_date": False,
        "date": "Sunday, February 3rd",
        "message_height": 0.4,
        "add_supporter_note": False,
        "pre_date_text": "Next video on ",
    }

    def construct(self):
        # Background
        plane = NumberPlane(
            (-10, 10), (-14, 14),
            axis_config={"stroke_color": BLUE_A}
        )
        for line in plane.family_members_with_points():
            line.set_stroke(width=line.get_stroke_width() / 2)
        self.add(
            plane,
            FullScreenFadeRectangle(fill_opacity=0.25),
        )

        # Pis
        pis = self.get_pis()
        pis.set_height(self.pi_height)
        pis.arrange(RIGHT, aligned_edge=DOWN)
        pis.move_to(self.pi_bottom, DOWN)
        self.pis = pis
        self.add(pis)

        plane.move_to(pis.get_bottom() + SMALL_BUFF * DOWN)

        # Message
        message = self.get_message()
        message.set_height(self.message_height)
        message.next_to(pis, DOWN)
        message.set_stroke(BLACK, 10, background=True)
        self.add(message)

        # Suppoerter note
        if self.add_supporter_note:
            note = self.get_supporter_note()
            note.scale(0.5)
            message.shift((MED_SMALL_BUFF - SMALL_BUFF) * UP)
            note.next_to(message, DOWN, SMALL_BUFF)
            self.add(note)

        yellow_parts = [sm for sm in message if sm.get_color() == YELLOW]
        for pi in pis:
            if yellow_parts:
                pi.look_at(yellow_parts[-1])
            else:
                pi.look_at(message)

    def get_pis(self):
        return VGroup(
            Randolph(color=BLUE_E, mode="pondering"),
            Randolph(color=BLUE_D, mode="hooray"),
            Randolph(color=BLUE_C, mode="tease"),
            Mortimer(color=GREY_BROWN, mode="thinking")
        )

    def get_message(self):
        if self.use_date:
            return self.get_date_message()
        else:
            return self.get_probabalistic_message()

    def get_probabalistic_message(self):
        return TexText(
            "New video every day ",
            "(with probability 0.05)",
            tex_to_color_map={"Sunday": YELLOW},
        )

    def get_date_message(self):
        return TexText(
            self.pre_date_text,
            self.date,
            tex_to_color_map={self.date: YELLOW},
        )

    def get_supporter_note(self):
        return TexText(
            "(Available to supporters for review now)",
            color="#F96854",
        )


class CurrBanner(Banner):
    CONFIG = {
        "camera_config": {
            "pixel_height": 1440,
            "pixel_width": 2560,
        },
        "pi_height": 1.25,
        "pi_bottom": 0.25 * DOWN,
        "use_date": False,
        "date": "Wednesday, March 15th",
        "message_scale_val": 0.9,
        "add_supporter_note": False,
        "pre_date_text": "Next video on ",
    }

    def construct(self):
        super().construct()
        for pi in self.pis:
            pi.set_gloss(0.1)
