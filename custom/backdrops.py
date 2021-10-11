from manimlib.constants import WHITE
from manimlib.constants import BLACK
from manimlib.constants import DOWN
from manimlib.constants import UP
from manimlib.constants import BLUE
from manimlib.scene.scene import Scene
from manimlib.mobject.frame import FullScreenRectangle
from manimlib.mobject.frame import ScreenRectangle
from manimlib.mobject.changing import AnimatedBoundary
from manimlib.mobject.svg.tex_mobject import TexText
from manimlib.animation.creation import Write

# from manimlib.mobject.svg.text_mobject import Text


class Spotlight(Scene):
    title = ""
    title_font_size = 60

    def construct(self):
        title = TexText(self.title, font_size=self.title_font_size)
        title.to_edge(UP)

        self.add(title)
        self.add(FullScreenRectangle())
        screen = ScreenRectangle()
        screen.set_height(6.0)
        screen.set_stroke(WHITE, 2)
        screen.set_fill(BLACK, 1)
        screen.to_edge(DOWN)
        animated_screen = AnimatedBoundary(screen)
        self.add(screen, animated_screen)
        self.wait(16)


class VideoWrapper(Scene):
    animate_boundary = True
    title = ""
    title_config = {
        "font_size": 60
    }
    wait_time = 32
    screen_height = 6.25

    def construct(self):
        self.add(FullScreenRectangle())
        screen = ScreenRectangle()
        screen.set_fill(BLACK, 1)
        screen.set_stroke(BLUE, 0)
        screen.set_height(self.screen_height)
        screen.to_edge(DOWN)

        if self.animate_boundary:
            screen = AnimatedBoundary(screen)
            wait_time = self.wait_time
        else:
            wait_time = 1

        self.add(screen)

        if self.title:
            title_text = TexText(
                self.title,
                **self.title_config,
            )
            title_text.set_max_width(screen.get_width())
            title_text.next_to(screen, UP)
            self.play(Write(title_text))

        self.wait(wait_time)
