from manim_imports_ext import *


class CleanBanner(Banner):
    message = " "


class ShortsBanner(Banner):
    message = "3b1b shorts"

    def get_pis(self):
        pis = VGroup(
            Randolph(color=BLUE_E, mode="pondering"),
            Randolph(color=BLUE_D, mode="hooray"),
            Randolph(color=BLUE_C, mode="tease"),
            Randolph(color=GREY_BROWN, mode="pondering")
        )
        height = 0.7
        for pi in pis:
            pi.set_height(1)
            pi.body.stretch(math.sqrt(height), 1, about_edge=UP)
            diff = height - pi.get_height()
            new_points = np.array(pi.body.get_points())
            new_points[3:26] += diff * DOWN
            new_points[63:82] += diff * DOWN
            new_points[2] += 0.05 * LEFT
            new_points[25] += 0.02 * DOWN
            pi.body.set_points(new_points)
            pi.mouth.shift(0.02 * UP)
        pis[3].flip()
        pis[1].mouth.shift(0.01 * UP)
        pis[1].eyes.shift(0.01 * UP)
        # for i, point in enumerate(pis[0].body.get_points()):
        #     pis[0].add(Integer(i).set_height(0.02).move_to(point))
        return pis
