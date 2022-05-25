from manimlib.animation.animation import Animation
from manimlib.animation.composition import AnimationGroup
from manimlib.animation.fading import FadeOut
from manimlib.animation.creation import DrawBorderThenFill
from manimlib.animation.creation import Write
from manimlib.animation.transform import ApplyMethod
from manimlib.animation.transform import MoveToTarget
from manimlib.constants import *
from manimlib.mobject.mobject import Group
from manimlib.mobject.svg.drawings import SpeechBubble
from manimlib.utils.config_ops import digest_config
from manimlib.utils.rate_functions import squish_rate_func
from manimlib.utils.rate_functions import there_and_back

from custom.characters.pi_class import PiCreatureClass


class Blink(ApplyMethod):
    CONFIG = {
        "rate_func": squish_rate_func(there_and_back)
    }

    def __init__(self, pi_creature, **kwargs):
        ApplyMethod.__init__(self, pi_creature.blink, **kwargs)


class PiCreatureBubbleIntroduction(AnimationGroup):
    def __init__(
        self, pi_creature, content,
        target_mode="speaking",
        look_at=None,
        bubble_type=SpeechBubble,
        max_bubble_height=None,
        max_bubble_width=None,
        bubble_direction=None,
        bubble_config={},
        bubble_creation_class=DrawBorderThenFill,
        bubble_creation_kwargs={},
        content_introduction_class=Write,
        content_introduction_kwargs={},
        **kwargs,
    ):
        bubble_config["max_height"] = max_bubble_height
        bubble_config["max_width"] = max_bubble_width
        if bubble_direction is not None:
            bubble_config["direction"] = bubble_direction
        bubble = pi_creature.get_bubble(
            content, bubble_type=bubble_type,
            **bubble_config
        )
        Group(bubble, bubble.content).shift_onto_screen()

        super().__init__(
            pi_creature.change(target_mode, look_at),
            bubble_creation_class(bubble, **bubble_creation_kwargs),
            content_introduction_class(bubble.content, **content_introduction_kwargs),
            **kwargs
        )


class PiCreatureSays(PiCreatureBubbleIntroduction):
    CONFIG = {
        "target_mode": "speaking",
        "bubble_type": SpeechBubble,
    }


class RemovePiCreatureBubble(AnimationGroup):
    CONFIG = {
        "target_mode": "plain",
        "look_at": None,
        "remover": True,
    }

    def __init__(self, pi_creature, **kwargs):
        assert hasattr(pi_creature, "bubble")
        digest_config(self, kwargs, locals())

        pi_creature.generate_target()
        pi_creature.target.change_mode(self.target_mode)
        if self.look_at is not None:
            pi_creature.target.look_at(self.look_at)

        AnimationGroup.__init__(
            self,
            MoveToTarget(pi_creature),
            FadeOut(pi_creature.bubble),
            FadeOut(pi_creature.bubble.content),
        )

    def clean_up_from_scene(self, scene=None):
        AnimationGroup.clean_up_from_scene(self, scene)
        self.pi_creature.bubble = None
        if scene is not None:
            scene.add(self.pi_creature)


class FlashThroughClass(Animation):
    CONFIG = {
        "highlight_color": GREEN,
    }

    def __init__(self, mobject, mode="linear", **kwargs):
        if not isinstance(mobject, PiCreatureClass):
            raise Exception("FlashThroughClass mobject must be a PiCreatureClass")
        digest_config(self, kwargs)
        self.indices = list(range(mobject.height * mobject.width))
        if mode == "random":
            np.random.shuffle(self.indices)
        Animation.__init__(self, mobject, **kwargs)

    def interpolate_mobject(self, alpha):
        index = int(np.floor(alpha * self.mobject.height * self.mobject.width))
        for pi in self.mobject:
            pi.set_color(BLUE_E)
        if index < self.mobject.height * self.mobject.width:
            self.mobject[self.indices[index]].set_color(self.highlight_color)
