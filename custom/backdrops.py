from manimlib.constants import WHITE
from manimlib.constants import BLACK
from manimlib.constants import DOWN
from manimlib.constants import UP
from manimlib.scene.scene import Scene
from manimlib.mobject.frame import FullScreenRectangle
from manimlib.mobject.frame import ScreenRectangle
from manimlib.mobject.changing import AnimatedBoundary
from manimlib.mobject.svg.tex_mobject import TexText

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
