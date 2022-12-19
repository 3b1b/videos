from manim_imports_ext import *


ASSET_DIR = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2022/galois/assets/"


class FlowerSymmetries(InteractiveScene):
    flower_height = 6.3

    def construct(self):
        # Add flower
        flower = ImageMobject(os.path.join(ASSET_DIR, "Flower"))
        flower.set_height(self.flower_height)
        self.add(flower)
        self.wait()

        # Several rotations
        angles = [45, 90, -45, 135, 180, -135]
        for angle in angles:
            arrow, label = self.get_arrow_and_label(flower, angle)
            self.play(
                Rotate(flower, angle * DEGREES),
                ShowCreation(arrow),
                FadeIn(label)
            )
            self.wait()
            self.play(FadeOut(label), FadeOut(arrow))

        # Show all rotations

    def get_vector_flower(self):
        flower_file = ImageMobject(os.path.join(ASSET_DIR, "Flower"))
        flower = SVGMobject(flower_file)
        flower.set_height(self.flower_height)
        flower.rotate(7 * DEGREES)
        flower.set_fill(GREY_A, 1)
        flower.set_gloss(1)
        flower.set_stroke(WHITE, 0)
        return flower

    def get_arrow_and_label(self, flower, degrees):
        radius = self.flower_height / 2 + MED_LARGE_BUFF
        arc = Arc(
            start_angle=0,
            angle=degrees * DEGREES - 3 * DEGREES,
            radius=radius,
            arc_center=flower.get_center(),
            stroke_width=5,
            color=BLUE,
        )
        arc.add_tip()
        label = Integer(degrees, unit="^\\circ")
        label.match_color(arc)
        point = arc.pfp(min([0.5, abs(45 / degrees)]))
        label.next_to(point, normalize(point - flower.get_center()))
        return arc, label
