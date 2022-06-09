import os
import logging

import numpy as np

from manimlib.animation.composition import AnimationGroup
from manimlib.animation.fading import FadeTransform
from manimlib.animation.transform import ReplacementTransform
from manimlib.constants import *
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.geometry import Circle
from manimlib.mobject.svg.drawings import ThoughtBubble
from manimlib.mobject.svg.drawings import SpeechBubble
from manimlib.mobject.svg.svg_mobject import SVGMobject
from manimlib.mobject.svg.text_mobject import Text
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.utils.config_ops import digest_config
from manimlib.utils.directories import get_directories
from manimlib.utils.space_ops import get_norm
from manimlib.utils.space_ops import normalize

PI_CREATURE_SCALE_FACTOR = 0.5

LEFT_EYE_INDEX = 0
RIGHT_EYE_INDEX = 1
LEFT_PUPIL_INDEX = 2
RIGHT_PUPIL_INDEX = 3
BODY_INDEX = 4
MOUTH_INDEX = 5


class PiCreature(SVGMobject):
    CONFIG = {
        "color": BLUE_E,
        "file_name_prefix": "PiCreatures",
        "stroke_width": 0,
        "stroke_color": BLACK,
        "fill_opacity": 1.0,
        "height": 3,
        "corner_scale_factor": 0.75,
        "flip_at_start": False,
        "is_looking_direction_purposeful": False,
        "start_corner": None,
        "long_lines": True,
        # Range of proportions along body where arms are
        "right_arm_range": [0.55, 0.7],
        "left_arm_range": [0.34, 0.462],
        "pupil_to_eye_width_ratio": 0.4,
        "pupil_dot_to_pupil_width_ratio": 0.3,
        "path_string_config": {
            "long_lines": True,
            "should_subdivide_sharp_curves": False,
        }
    }

    def __init__(self, mode="plain", **kwargs):
        digest_config(self, kwargs)
        self.mode = mode
        self.bubble = None
        color = kwargs.pop("color", self.color)

        super().__init__(
            file_name=self.get_svg_file_path(mode),
            color=None,
            **kwargs
        )

        self.init_structure()
        self.set_color(color)
        if self.flip_at_start:
            self.flip()
        if self.start_corner is not None:
            self.to_corner(self.start_corner)
        self.refresh_triangulation()

    def get_svg_file_path(self, mode):
        folder = get_directories()["pi_creature_images"]
        path = os.path.join(folder, f"{mode}.svg")
        if os.path.exists(path):
            return path
        else:
            logging.log(
                logging.WARNING,
                f"No {self.file_name_prefix} design with mode {mode}",
            )
            folder = get_directories()["pi_creature_images"]
            return os.path.join(folder, "plain.svg")

    def init_structure(self):
        # Figma exports with superfluous parts, so this
        # hardcodes how to extract what we want.
        parts = self.submobjects
        self.eyes = self.draw_eyes(
            original_irises=VGroup(parts[2], parts[6]),
            original_pupils=VGroup(parts[8], parts[9])
        )
        self.body = parts[10]
        self.mouth = parts[11]
        self.mouth.insert_n_curves(10)
        self.set_submobjects([self.eyes, self.body, self.mouth])

    def align_data_and_family(self, mobject):
        # This ensures that after a transform into a different mode,
        # the pi creatures mode will be updated appropriately
        SVGMobject.align_data_and_family(self, mobject)
        if isinstance(mobject, PiCreature):
            self.mode = mobject.get_mode()

    def draw_eyes(self, original_irises, original_pupils):
        # Instead of what is drawn, make new circles.
        # This is mostly because the paths associated
        # with the eyes in all the drawings got slightly
        # messed up.
        eyes = VGroup()
        for iris, ref_pupil in zip(original_irises, original_pupils):
            pupil_r = iris.get_width() / 2
            pupil_r *= self.pupil_to_eye_width_ratio
            dot_r = pupil_r
            dot_r *= self.pupil_dot_to_pupil_width_ratio

            black = Circle(radius=pupil_r, color=BLACK)
            dot = Circle(radius=dot_r, color=WHITE)
            pupil = VGroup(black, dot)
            pupil.set_style(fill_opacity=1, stroke_width=0)
            pupil.move_to(ref_pupil)
            dot.shift(black.pfp(3 / 8) - dot.pfp(3 / 8))
            eye = VGroup(iris, pupil)
            eye.pupil = pupil
            eye.iris = iris
            eyes.add(eye)
        return eyes

    def set_color(self, color, recurse=False):
        self.body.set_fill(color)
        return self

    def get_color(self):
        return self.body.get_color()

    def change_mode(self, mode):
        new_self = self.__class__(mode=mode)
        new_self.match_style(self)
        new_self.match_height(self)
        if self.is_flipped() != new_self.is_flipped():
            new_self.flip()
        new_self.shift(self.eyes.get_center() - new_self.eyes.get_center())
        if hasattr(self, "purposeful_looking_direction"):
            new_self.look(self.purposeful_looking_direction)
        self.become(new_self)
        self.mode = mode
        return self

    def get_mode(self):
        return self.mode

    def look(self, direction):
        direction = normalize(direction)
        self.purposeful_looking_direction = direction
        for eye in self.eyes:
            iris, pupil = eye
            iris_center = iris.get_center()
            right = iris.get_right() - iris_center
            up = iris.get_top() - iris_center
            vect = direction[0] * right + direction[1] * up
            v_norm = get_norm(vect)
            pupil_radius = 0.5 * pupil.get_width()
            vect *= (v_norm - 0.75 * pupil_radius) / v_norm
            pupil.move_to(iris_center + vect)
        self.eyes[1].pupil.align_to(self.eyes[0].pupil, DOWN)
        return self

    def look_at(self, point_or_mobject):
        if isinstance(point_or_mobject, Mobject):
            point = point_or_mobject.get_center()
        else:
            point = point_or_mobject
        self.look(point - self.eyes.get_center())
        return self

    def get_looking_direction(self):
        vect = self.eyes[0].pupil.get_center() - self.eyes[0].get_center()
        return normalize(vect)

    def get_look_at_spot(self):
        return self.eyes.get_center() + self.get_looking_direction()

    def is_flipped(self):
        return self.eyes.submobjects[0].get_center()[0] > \
            self.eyes.submobjects[1].get_center()[0]

    def blink(self):
        eyes = self.eyes
        eye_bottom_y = eyes.get_y(DOWN)

        for eye_part in eyes.family_members_with_points():
            new_points = eye_part.get_points()
            new_points[:, 1] = eye_bottom_y
            eye_part.set_points(new_points)

        return self

    def to_corner(self, vect=None, **kwargs):
        if vect is not None:
            SVGMobject.to_corner(self, vect, **kwargs)
        else:
            self.scale(self.corner_scale_factor)
            self.to_corner(DOWN + LEFT, **kwargs)
        return self

    def get_bubble(self, content, bubble_type=ThoughtBubble, **bubble_config):
        bubble = bubble_type(**bubble_config)
        if len(content) > 0:
            if isinstance(content[0], str):
                content_mob = Text(content)
            else:
                content_mob = content
            bubble.add_content(content_mob)
            bubble.resize_to_content()
        bubble.pin_to(self)
        self.bubble = bubble
        return bubble

    def make_eye_contact(self, pi_creature):
        self.look_at(pi_creature.eyes)
        pi_creature.look_at(self.eyes)
        return self

    def shrug(self):
        self.change_mode("shruggie")
        points = self.mouth.get_points()
        top_mouth_point, bottom_mouth_point = [
            points[np.argmax(points[:, 1])],
            points[np.argmin(points[:, 1])]
        ]
        self.look(top_mouth_point - bottom_mouth_point)
        return self

    def get_arm_copies(self):
        body = self.body
        return VGroup(*[
            body.copy().pointwise_become_partial(body, *alpha_range)
            for alpha_range in (self.right_arm_range, self.left_arm_range)
        ])

    # Overrides

    def become(self, mobject):
        super().become(mobject)
        if isinstance(mobject, PiCreature):
            self.bubble = mobject.bubble
        return self

    # Animations

    def change(self, new_mode, look_at=None):
        animation = self.animate.change_mode(new_mode)
        if look_at is not None:
            animation = animation.look_at(look_at)
        return animation

    def says(self, content, mode="speaking", look_at=None, **kwargs):
        from custom.characters.pi_creature_animations import PiCreatureBubbleIntroduction
        return PiCreatureBubbleIntroduction(
            self, content,
            target_mode=mode,
            look_at=look_at,
            bubble_type=SpeechBubble,
            **kwargs,
        )

    def thinks(self, content, mode="thinking", look_at=None, **kwargs):
        from custom.characters.pi_creature_animations import PiCreatureBubbleIntroduction
        return PiCreatureBubbleIntroduction(
            self, content,
            target_mode=mode,
            look_at=look_at,
            bubble_type=ThoughtBubble,
            **kwargs,
        )

    def replace_bubble(self, content, mode="pondering", look_at=None, **kwargs):
        if self.bubble is None:
            return self.change(mode, look_at)
        old_bubble = self.bubble
        new_bubble = self.get_bubble(content, bubble_type=old_bubble.__class__, **kwargs)
        self.bubble = new_bubble
        return AnimationGroup(
            ReplacementTransform(old_bubble, new_bubble),
            FadeTransform(old_bubble.content, new_bubble.content),
            self.change(mode, look_at)
        )

    def debubble(self, mode="plain", look_at=None, **kwargs):
        if self.bubble is None:
            logging.log(
                logging.WARNING,
                f"Calling debubble on PiCreature with no bubble",
            )
            return self.change(mode, look_at)
        from custom.characters.pi_creature_animations import RemovePiCreatureBubble
        result = RemovePiCreatureBubble(
            self, target_mode=mode, look_at=look_at, **kwargs
        )
        self.bubble = None
        return result


def get_all_pi_creature_modes():
    result = []
    prefix = PiCreature.CONFIG["file_name_prefix"] + "_"
    suffix = ".svg"
    for file in os.listdir(PI_CREATURE_DIR):
        if file.startswith(prefix) and file.endswith(suffix):
            result.append(
                file[len(prefix):-len(suffix)]
            )
    return result


class Randolph(PiCreature):
    pass  # Nothing more than an alternative name


class Mortimer(PiCreature):
    CONFIG = {
        "color": GREY_BROWN,
        "flip_at_start": True,
    }


class Mathematician(PiCreature):
    CONFIG = {
        "color": GREY,
    }


class BabyPiCreature(PiCreature):
    CONFIG = {
        "scale_factor": 0.5,
        "eye_scale_factor": 1.2,
        "pupil_scale_factor": 1.3
    }

    def __init__(self, *args, **kwargs):
        PiCreature.__init__(self, *args, **kwargs)
        self.scale(self.scale_factor)
        self.shift(LEFT)
        self.to_edge(DOWN, buff=LARGE_BUFF)
        eyes = self.eyes
        eyes_bottom = eyes.get_bottom()
        eyes.scale(self.eye_scale_factor)
        eyes.move_to(eyes_bottom, aligned_edge=DOWN)
        looking_direction = self.get_looking_direction()
        for eye in eyes:
            eye.pupil.scale(self.pupil_scale_factor)
        self.look(looking_direction)


class TauCreature(PiCreature):
    CONFIG = {
        "file_name_prefix": "TauCreatures"
    }


class ThreeLeggedPiCreature(PiCreature):
    CONFIG = {
        "file_name_prefix": "ThreeLeggedPiCreatures"
    }


class Eyes(VMobject):
    CONFIG = {
        "height": 0.3,
        "thing_to_look_at": None,
        "mode": "plain",
    }

    def __init__(self, body, **kwargs):
        VMobject.__init__(self, **kwargs)
        self.body = body
        eyes = self.create_eyes()
        self.set_submobjects(eyes.submobjects)

    def create_eyes(self, mode=None, thing_to_look_at=None):
        if mode is None:
            mode = self.mode
        if thing_to_look_at is None:
            thing_to_look_at = self.thing_to_look_at
        self.thing_to_look_at = thing_to_look_at
        self.mode = mode
        looking_direction = None

        pi = PiCreature(mode=mode)
        eyes = VGroup(pi.eyes, pi.pupils)
        if self.submobjects:
            eyes.match_height(self)
            eyes.move_to(self, DOWN)
            looking_direction = self[1].get_center() - self[0].get_center()
        else:
            eyes.set_height(self.height)
            eyes.move_to(self.body.get_top(), DOWN)

        height = eyes.get_height()
        if thing_to_look_at is not None:
            pi.look_at(thing_to_look_at)
        elif looking_direction is not None:
            pi.look(looking_direction)
        eyes.set_height(height)

        return eyes

    def change_mode(self, mode, thing_to_look_at=None):
        new_eyes = self.create_eyes(
            mode=mode,
            thing_to_look_at=thing_to_look_at
        )
        self.set_submobjects(new_eyes.submobjects)
        return self

    def look_at(self, thing_to_look_at):
        self.change_mode(
            self.mode,
            thing_to_look_at=thing_to_look_at
        )
        return self

    def blink(self, **kwargs):  # TODO, change Blink
        bottom_y = self.get_bottom()[1]
        for submob in self:
            submob.apply_function(
                lambda p: [p[0], bottom_y, p[2]]
            )
        return self
