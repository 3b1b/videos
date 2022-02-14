from manim_imports_ext import *


# Broken, but fixable
class ImagesMod256(Scene):
    CONFIG = {
    }

    def construct(self):
        lion = ImageMobject("Lion")
        lion.set_height(6)
        lion.to_edge(DOWN)
        integer = Integer(1)
        expression = VGroup(
            Tex("n \\rightarrow"),
            integer,
            Tex("\\times n \\mod 256")
        )
        expression.arrange(RIGHT)
        expression[-2:].shift(MED_SMALL_BUFF * RIGHT)
        integer.next_to(expression[-1], LEFT, SMALL_BUFF)
        expression.to_edge(UP)
        self.add(lion)
        self.add(expression)
        self.wait(0.5)
        self.remove(lion)

        total_time = 10
        wait_time = 1.0 / 15
        alpha_range = np.linspace(0, 1, int(total_time / wait_time))
        for start, end in (1, 128), (128, 257):
            for alpha in ProgressDisplay(alpha_range):
                m = int(interpolate(start, end, smooth(alpha)))
                im = Image.fromarray(lion.pixel_array)
                new_im = Image.eval(im, lambda n: (n * m) % 256)
                alt_lion = ImageMobject(np.array(new_im))
                alt_lion.replace(lion)
                self.add(alt_lion)

                new_int = Integer(m)
                new_int.next_to(expression[-1], LEFT, SMALL_BUFF)
                self.remove(integer)
                self.add(new_int)
                integer = new_int
                self.wait(1.0 / 15)
                self.remove(alt_lion)
        self.add(alt_lion)
        self.wait()
