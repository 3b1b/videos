from manim_imports_ext import *


class AdditionAnagram(Scene):
    def construct(self):
        words1 = Text("twelve + one", font_size=120)
        words2 = Text("eleven + two", font_size=120)
        VGroup(words1, words2).shift(DOWN)

        twelve = words1.get_part_by_text("twelve")
        one = words1.get_part_by_text("one")
        eleven = words2.get_part_by_text("eleven")
        two = words2.get_part_by_text("two")

        buff = 1.0
        dots1 = VGroup(
            *Dot().get_grid(3, 4).next_to(twelve, UP, buff=buff),
            Dot().next_to(one, UP, buff=buff)
        ).set_color(BLUE)
        dots2 = VGroup(
            *Dot().replicate(11).arrange_in_grid(3, 4).next_to(eleven, UP, buff=buff),
            *Dot().get_grid(2, 1).next_to(two, UP, buff=buff)
        ).set_color(BLUE)

        self.add(words1, dots1)
        self.wait()
        for w1, w2, d1, d2 in [(words1, words2, dots1, dots2), (words2, words1, dots2, dots1)]:
            self.clear()
            self.play(
                TransformMatchingShapes(
                    w1.copy(), w2.copy(),
                    path_arc=PI / 2,
                ),
                Transform(d1.copy(), d2.copy()),
                run_time=3
            )
            self.wait(2)
