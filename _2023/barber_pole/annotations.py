from manim_imports_ext import *


class AnnotateDemo(InteractiveScene):
    def construct(self):
        image = ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2023/barber_pole/images/DemoStill.jpg")
        image.set_height(FRAME_HEIGHT)
        plane = NumberPlane().fade(0.25)
        # self.add(image)
        # self.add(plane)

        # Label sugar
        sugar_label = Text("Sugar solution\n(0.75g sucrose/mL water)")
        sugar_label.move_to(2.5 * UP)
        sugar_label.set_backstroke(BLACK, 3)
        arrow_kw = dict(stroke_color=RED, stroke_width=10)
        sugar_arrow = Arrow(sugar_label, plane.c2p(0, 0.5), **arrow_kw)

        self.play(
            Write(sugar_label, lag_ratio=0.01, run_time=2),
            ShowCreation(sugar_arrow),
        )
        self.wait()

        # Label light
        light_label = Text("White light\n(unpolarized)")
        light_label.match_y(sugar_label)
        light_label.match_style(sugar_label)
        light_label.to_edge(RIGHT)
        light_arrow = Arrow(light_label, plane.c2p(4.75, 0.85), buff=0.1, **arrow_kw)

        self.play(
            FadeTransform(sugar_label, light_label),
            ReplacementTransform(sugar_arrow, light_arrow),
        )
        self.wait()

        # Label polarizer
        filter_label = Text("Linearly polarizing filter\n(variable angle)")
        filter_label.set_x(3.5).to_edge(UP)
        filter_label.match_style(sugar_label)
        filter_arrow = Arrow(filter_label, plane.c2p(3.4, 1.25), buff=0.1, **arrow_kw)

        self.play(
            FadeTransform(light_label, filter_label),
            ReplacementTransform(light_arrow, filter_arrow),
        )
        self.wait()
        self.play(
            filter_label.animate.set_x(-2.2).to_edge(UP, buff=0.5),
            filter_arrow.animate.set_x(-3.3),
        )
        self.wait()


class FocusOnWall(InteractiveScene):
    def construct(self):
        # Test
        words = Text("Notice what color comes out", font_size=72)
        words.set_backstroke(BLACK, 5)
        words.to_edge(UP)
        arrow = Arrow(
            words["Notice what"].get_bottom(), 6.5 * LEFT + 0.5 * UP,
            buff=0.5,
            stroke_color=RED, stroke_width=14,
        )
        self.play(
            FadeIn(words, 0.25 * UP),
            GrowArrow(arrow),
        )
        self.wait()


class ThisIsStillWhiteLight(InteractiveScene):
    def construct(self):
        # Test
        words = Text("This is still white light", font_size=60)
        words.to_edge(UP)
        arrow = Arrow(
            words["This"].get_bottom(), LEFT + 0.5 * UP,
            stroke_color=WHITE, stroke_width=8,
        )

        self.play(
            Write(words, run_time=1),
            FadeIn(arrow, RIGHT, run_time=1.5, rate_func=rush_into),
        )
        self.wait()


class BasicallyZ(InteractiveScene):
    def construct(self):
        rect = Rectangle(6.0, 2.0)
        rect.set_stroke(YELLOW, 2)
        rect.set_fill(YELLOW, 0.2)
        rect.to_edge(RIGHT, buff=0.25)
        words = Text("Essentially parallel\nto the z-axis")
        words.next_to(rect, UP)
        self.play(
            FadeIn(words, 0.25 * UP),
            FadeIn(rect)
        )
        self.wait()


class StrengthInDifferentDirectionsWithDecimal(InteractiveScene):
    def construct(self):
        line = Line(ORIGIN, 7 * RIGHT)
        line.set_stroke(TEAL, 4)
        arc = always_redraw(lambda: Arc(angle=line.get_angle(), radius=0.5))
        angle_label = Integer(0, unit=R"^\circ")
        angle_label.add_updater(lambda m: m.set_value(line.get_angle() / DEGREES))
        angle_label.add_updater(lambda m: m.set_height(clip(arc.get_height(), 0.01, 0.4)))
        angle_label.add_updater(lambda m: m.next_to(arc.pfp(0.3), normalize(arc.pfp(0.3)), SMALL_BUFF, aligned_edge=DOWN))

        strong_words = Text("Strongest in this direction")
        strong_words.next_to(line, UP)
        cos_temp_text = "cos(00*)=0.00"
        weak_words = Text(f"Weaker by a factor of {cos_temp_text}", font_size=36)
        cos_template = weak_words[cos_temp_text][0]
        cos_template.set_opacity(0)
        weak_words.next_to(line, UP)

        strong_words.set_backstroke(BLACK, 10)
        weak_words.set_backstroke(BLACK, 10)

        def get_cos_tex():
            cos_tex = Tex(R"\cos(10^\circ) = 0.00", font_size=36)
            cos_tex.make_number_changable("10", edge_to_fix=RIGHT).set_value(line.get_angle() / DEGREES)
            cos_tex.make_number_changable("0.00").set_value(math.cos(line.get_angle()))
            cos_tex.rotate(line.get_angle())
            cos_tex.move_to(weak_words[-len(cos_temp_text) + 1:])
            cos_tex.set_backstroke(BLACK, 10)
            return cos_tex

        cos_tex = always_redraw(get_cos_tex)

        # Test
        self.play(ShowCreation(line), Write(strong_words, run_time=1))
        self.wait(2)
        self.add(arc, angle_label, weak_words, cos_tex)
        self.remove(strong_words)
        rot_group = VGroup(line, weak_words)
        self.play(
            Rotate(rot_group, 30 * DEGREES, about_point=ORIGIN),
            run_time=3
        )
        self.wait(2)
        self.play(
            self.frame.animate.set_height(12),
            Rotate(rot_group, 30 * DEGREES, about_point=ORIGIN),
            run_time=3
        )
        self.wait(2)
        self.play(
            self.frame.animate.set_height(15),
            Rotate(rot_group, 30 * DEGREES, about_point=ORIGIN),
            run_time=3
        )
        self.wait(2)


class ERadEquation(InteractiveScene):
    def construct(self):
        equation = Tex(R"""
            \vec{E}_{\text{rad}}(\vec{r}, t) = 
            {-q \over 4\pi \epsilon_0 c^2}
            {1 \over ||\vec{r}||}
            \vec{a}_\perp(t - ||\vec{r}|| / c)
        """, font_size=36)
        lhs = equation[R"\vec{E}_{\text{rad}}(\vec{r}, t)"]
        lhs.set_color(BLUE)
        equation[R"\vec{a}_\perp("].set_color(PINK)
        equation[R")"][1].set_color(PINK)
        self.add(equation)


class XZLabel(InteractiveScene):
    def construct(self):
        xz_label = Tex("xz")
        x, z = xz_label
        x.next_to(ORIGIN, UP, SMALL_BUFF).to_edge(RIGHT, buff=0.2)
        z.next_to(ORIGIN, RIGHT, SMALL_BUFF).to_edge(UP, buff=0.2)
        self.add(xz_label)
