lines = """
from manimlib.constants import *

from manimlib.animation.animation import *
from manimlib.animation.composition import *
from manimlib.animation.creation import *
from manimlib.animation.fading import *
from manimlib.animation.growing import *
from manimlib.animation.indication import *
from manimlib.animation.movement import *
from manimlib.animation.numbers import *
from manimlib.animation.rotation import *
from manimlib.animation.specialized import *
from manimlib.animation.transform import *
from manimlib.animation.transform_matching_parts import *
from manimlib.animation.update import *

from manimlib.camera.camera import *

from manimlib.mobject.coordinate_systems import *
from manimlib.mobject.changing import *
from manimlib.mobject.frame import *
from manimlib.mobject.functions import *
from manimlib.mobject.geometry import *
from manimlib.mobject.matrix import *
from manimlib.mobject.mobject import *
from manimlib.mobject.number_line import *
from manimlib.mobject.numbers import *
from manimlib.mobject.probability import *
from manimlib.mobject.shape_matchers import *
from manimlib.mobject.svg.brace import *
from manimlib.mobject.svg.drawings import *
from manimlib.mobject.svg.svg_mobject import *
from manimlib.mobject.svg.tex_mobject import *
from manimlib.mobject.svg.text_mobject import *
from manimlib.mobject.three_dimensions import *
from manimlib.mobject.types.image_mobject import *
from manimlib.mobject.types.point_cloud_mobject import *
from manimlib.mobject.types.surface import *
from manimlib.mobject.types.vectorized_mobject import *
from manimlib.mobject.types.dot_cloud import *
from manimlib.mobject.mobject_update_utils import *
from manimlib.mobject.value_tracker import *
from manimlib.mobject.vector_field import *

from manimlib.scene.scene import *
from manimlib.scene.three_d_scene import *

from manimlib.utils.bezier import *
from manimlib.utils.color import *
from manimlib.utils.config_ops import *
from manimlib.utils.customization import *
from manimlib.utils.debug import *
from manimlib.utils.directories import *
from manimlib.utils.images import *
from manimlib.utils.iterables import *
from manimlib.utils.file_ops import *
from manimlib.utils.paths import *
from manimlib.utils.rate_functions import *
from manimlib.utils.simple_functions import *
from manimlib.utils.sounds import *
from manimlib.utils.space_ops import *
from manimlib.utils.strings import *
"""

import time
import numpy as np
import random

def get_runtime(func):
    now = time.time()
    func()
    return np.round(1000 * (time.time() - now), 2)


if __name__ == "__main__":
    all_lines = lines.split("\n")
    random.shuffle(all_lines)
    total_time = 0
    for line in all_lines:
        try:
            rt = get_runtime(lambda: exec(line))
            total_time += rt
            if rt > 1:
                print(rt, "\t", line)
        except:
            pass

    print(total_time)
