from __future__ import annotations

from manimlib.animation.composition import AnimationGroup
from manimlib.animation.fading import FadeOut
from manimlib.animation.creation import DrawBorderThenFill
from manimlib.animation.creation import Write
from manimlib.animation.transform import ApplyMethod
from manimlib.animation.transform import MoveToTarget
from manimlib.constants import *
from manimlib.mobject.mobject import Group
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.svg.drawings import SpeechBubble
from manimlib.utils.rate_functions import squish_rate_func
from manimlib.utils.rate_functions import there_and_back

from custom.characters.pi_creature import PiCreature

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable
    from manimlib.typing import Vect3

class Blink(ApplyMethod):
    def __init__(
        self,
        pi_creature: PiCreature,
        rate_func: Callable = squish_rate_func(there_and_back),
        **kwargs
    ):
        super().__init__(pi_creature.blink, rate_func=rate_func, **kwargs)


class PiCreatureBubbleIntroduction(AnimationGroup):
    def __init__(
        self,
        pi_creature: PiCreature,
        content: str,
        target_mode: str = "speaking",
        look_at: Mobject | Vect3 | None = None,
        bubble_type: type = SpeechBubble,
        bubble_direction: Vect3 | None = None,
        bubble_config=dict(),
        bubble_creation_class: type = DrawBorderThenFill,
        bubble_creation_kwargs: dict = dict(),
        content_introduction_class: type = Write,
        content_introduction_kwargs: dict = dict(),
        **kwargs,
    ):
        bubble_config = dict(bubble_config)
        if bubble_direction is not None:
            bubble_config["direction"] = bubble_direction
        bubble = pi_creature.get_bubble(
            content,
            bubble_type=bubble_type,
            **bubble_config
        )
        bubble.shift_onto_screen()

        super().__init__(
            pi_creature.change(target_mode, look_at),
            bubble_creation_class(bubble.body, **bubble_creation_kwargs),
            content_introduction_class(bubble.content, **content_introduction_kwargs),
            **kwargs
        )


class PiCreatureSays(PiCreatureBubbleIntroduction):
    def __init__(
        self,
        pi_creature: PiCreature,
        content: str,
        target_mode: str = "speaking",
        bubble_type: type = SpeechBubble,
        **kwargs,
    ):
        super().__init__(
            pi_creature, content,
            target_mode=target_mode,
            bubble_type=bubble_type,
            **kwargs
        )


class RemovePiCreatureBubble(AnimationGroup):
    def __init__(
        self,
        pi_creature: PiCreature,
        target_mode: str = "plain",
        look_at: Mobject | Vect3 | None = None,
        remover: bool = True,
        **kwargs
    ):
        assert hasattr(pi_creature, "bubble")
        self.pi_creature = pi_creature

        pi_creature.target = pi_creature.generate_target()
        pi_creature.target.change_mode(target_mode)
        if look_at is not None:
            pi_creature.target.look_at(look_at)
        anims = [MoveToTarget(pi_creature)]
        if pi_creature.bubble is not None:
            anims.append(FadeOut(pi_creature.bubble))

        super().__init__(*anims)

    def clean_up_from_scene(self, scene=None):
        AnimationGroup.clean_up_from_scene(self, scene)
        self.pi_creature.bubble = None
        if scene is not None:
            scene.add(self.pi_creature)
