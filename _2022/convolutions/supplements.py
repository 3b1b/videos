from manim_imports_ext import *
from _2022.convolutions.main import *


class HoldUpLists(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        self.play(
            self.teacher.change("raise_right_hand", look_at=self.screen),
            self.change_students("pondering", "thinking", "pondering", look_at=self.screen),
        )
        self.wait()
        self.play(
            self.change_students("thinking", "thinking", "pondering", look_at=3 * UR),
            self.teacher.change("raise_left_hand", look_at=3 * UR)
        )
        self.wait(2)
        self.play(LaggedStartMap(FadeOut, self.pi_creatures, shift=2 * DOWN))


class FunToVisualize(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.teacher.says("Boy are they fun\nto visualize!", mode="hooray"),
            self.change_students("tease", "happy", "well")
        )
        self.wait(3)


class ILearnedSomething(TeacherStudentsScene):
    def construct(self):
        # Test
        implication = VGroup(
            Text("Making\nvisuals"),
            Tex(R"\Rightarrow", font_size=60),
            Text("Deeper\nunderstanding")
        )
        implication.arrange(RIGHT, buff=0.4)
        implication.move_to(self.hold_up_spot, DOWN)
        implication.shift_onto_screen()
        self.play(
            self.teacher.change("raise_right_hand", implication),
            FadeIn(implication, UP),
            self.change_students("confused", "hesitant", "tease", look_at=implication)
        )
        self.wait(5)


class NormalFunctionPreview(InteractiveScene):
    def construct(self):
        # Setup axes
        frame = self.camera.frame
        axes2d = Axes((-2, 2), (-1, 2), width=10, height=6)
        axes3d = ThreeDAxes((-2, 2), (-2, 2), (-1, 2), width=10, height=10, depth=6)
        axes3d.shift(axes2d.get_origin() - axes3d.get_origin())

        curve = axes2d.get_graph(lambda x: math.exp(-x**2))
        curve.set_stroke(BLUE)
        label = Tex("e^{-x^2}", font_size=60)
        label.next_to(curve.get_top(), UR).shift(RIGHT)
        label3d = Tex("e^{-x^2 - y^2}", font_size=60)
        label3d.move_to(label, DL)

        VGroup(axes2d, curve, label, label3d).rotate(PI / 2, RIGHT, about_point=axes3d.get_origin())
        frame.reorient(0, 90)

        self.add(axes2d)
        self.play(ShowCreation(curve), Write(label))

        # Show in 3d
        surface = axes3d.get_graph(lambda x, y: math.exp(-x * x - y * y))
        surface.always_sort_to_camera(self.camera)
        surface_mesh = SurfaceMesh(surface)
        surface_mesh.set_stroke(WHITE, 0.5, 0.5)

        self.add(axes3d, surface, curve)
        self.play(
            FadeIn(axes3d),
            frame.animate.reorient(-10, 70).set_anim_args(run_time=2),
            FadeIn(surface, time_span=(1, 2)),
            FadeIn(label3d),
            FadeOut(label),
            Rotate(curve, PI, run_time=3)
        )
        frame.add_updater(lambda m, dt: m.increment_theta(0.02 * dt))
        self.play(Write(surface_mesh))
        self.wait(8)


class JuliaVideoFrame(VideoWrapper):
    title = "Lecture on convolutions for image processing"


class Intimidation(InteractiveScene):
    def construct(self):
        pis = VGroup(*(Randolph(color=color) for color in (BLUE_C, BLUE_E, BLUE_D)))
        pis.arrange(DOWN, buff=LARGE_BUFF)
        pis.set_height(FRAME_HEIGHT - 1)
        pis.move_to(FRAME_WIDTH * LEFT / 4)

        self.play(LaggedStart(*(
            pi.change("pondering", look_at=3 * UR)
            for pi in pis
        )))
        self.play(LaggedStart(*(Blink(pi) for pi in pis)))
        self.wait()
        self.play(LaggedStart(*(
            pi.change(mode, look_at=3 * UR)
            for pi, mode in zip(pis, ("horrified", "maybe", "pleading"))
        )))
        for x in range(2):
            self.play(Blink(random.choice(pis)))
            self.wait()


class SideBySideForContinuousConv(InteractiveScene):
    def construct(self):
        self.add(FullScreenRectangle())
        squares = Square().replicate(2)
        squares.set_fill(BLACK, 1)
        squares.set_stroke(WHITE, 2)
        squares.set_height(6)
        squares.arrange(RIGHT, buff=0.5)
        squares.set_width(FRAME_WIDTH - 1)
        squares.to_edge(DOWN)
        self.add(squares)


class ThereIsAnother(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.says("There is another"),
            self.change_students("pondering", "thinking", "erm", look_at=self.screen)
        )
        self.wait(2)


class SharedInsights(InteractiveScene):
    def construct(self):
        # Rects
        rects = ScreenRectangle().get_grid(2, 2, h_buff=2.0, v_buff=2.0)
        rects.set_width(FRAME_WIDTH - 3)
        rects.set_stroke(WHITE, 2)
        rects.set_fill(BLACK, 1)
        self.add(FullScreenRectangle())
        self.add(rects)

        # Inter-relate
        kw = dict(stroke_width=5, stroke_color=YELLOW)
        arrows = VGroup(
            Arrow(rects[3].get_top(), rects[1].get_bottom(), **kw),
            Arrow(rects[3].get_corner(UL), rects[0].get_corner(DR), **kw),
            Arrow(rects[3].get_left(), rects[2].get_right(), **kw),
        )

        self.wait()
        self.play(LaggedStartMap(GrowArrow, arrows, lag_ratio=0.5))
        self.wait()


class HoldUpImageProcessing(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.change("raise_right_hand"),
            self.change_students("pleading", "hesitant", "horrified", look_at=2 * UP)
        )
        self.wait(2)
        self.play(self.change_students("pondering", "thinking", "pondering", look_at=2 * UP))
        self.wait(6)


class OtherVisualizations(TeacherStudentsScene):
    def construct(self):
        self.play(self.change_students("happy", "thinking", "tease", look_at=self.screen))
        self.play(
            self.teacher.says("Can you think of\nother visualizations?"),
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen),
        )
        self.wait(2)
        self.play(
            self.change_students("confused", "erm", "thinking")
        )
        self.wait(4)
        self.play(
            self.students[2].change("raise_right_hand", self.teacher.eyes),
            self.teacher.change("tease"),
        )
        self.wait(2)


class Boring(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(LaggedStart(
            self.students[2].says("Boring!", mode="dejected", look_at=self.teacher.eyes, bubble_direction=LEFT),
            self.teacher.change("hesitant"),
            self.students[0].change("hesitant", self.screen),
            self.students[1].change("guilty", self.screen),
            lag_ratio=0.25,
        ))
        self.wait(3)


class AskForExample(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.change("raise_right_hand", self.screen),
            self.change_students("pondering", "thinking", "confused", look_at=self.screen)
        )
        self.wait(3)
        self.play(
            self.students[2].says("Can we go do\na concrete example?", mode="raise_left_hand", bubble_direction=LEFT)
        )
        self.wait(3)


class MarioConvolutionLabel(BoxBlurMario):
    label_conv = True

    def construct(self):
        self.clear()
        pa = self.pixel_array
        ka = self.kernel_array
        ka.clear_updaters()
        pa.set_stroke(width=0.1)

        expr = VGroup(pa, Tex("*", font_size=72), ka)
        for arr in expr[::2]:
            arr.set_height(1.5)

        expr.arrange(RIGHT)
        expr.move_to(FRAME_WIDTH * RIGHT / 4).to_edge(UP, buff=0.9)

        conv_label = Text("Convolution", font_size=36)
        conv_label.next_to(expr, UP, buff=0.5)
        arrow = Arrow(conv_label, expr[1], buff=0.2)

        self.play(FadeIn(expr, lag_ratio=0.001))
        if self.label_conv:
            self.play(
                Write(conv_label),
                ShowCreation(arrow),
            )
        self.wait()
        kw = dict(rate_func=smooth, run_time=1)
        self.play(Rotate(ka, PI, **kw))
        squares = ka.copy()
        for square in squares:
            square.remove(square[0])
        self.add(squares)
        self.remove(ka)
        self.play(LaggedStart(*(
            Rotate(square[0], -PI, **kw)
            for square in ka
        )))
        self.wait(2)


class CatConvolutionLabel(MarioConvolutionLabel, GaussianBluMario):
    image_name = "PixelArtCat"
    kernel_tex = None

    def construct(self):
        MarioConvolutionLabel.construct(self)


class SobelKernelLabel(MarioConvolutionLabel, SobelFilter2):
    image_name = "BitRandy"
    kernel_tex = None

    def construct(self):
        for square in self.kernel_array:
            square.set_stroke(WHITE, 1)
            value = square[0].get_value()
            square.set_fill(rgb_to_color([
                max(-value, 0), max(value, 0), max(value, 0)
            ]), 1)
            square[0].set_stroke(width=0)
            square[0].set_fill(WHITE, 1)

        MarioConvolutionLabel.construct(self)


class SharpenKernelLabel(MarioConvolutionLabel, SharpenFilter):
    image_name = "KirbySmall"
    kernel_tex = None
    label_conv = False

    def construct(self):
        for square in self.kernel_array:
            square[0].scale(0.6)
        MarioConvolutionLabel.construct(self)


class SobelCatKernelLabel(SobelKernelLabel, SobelFilter1):
    image_name = "PixelArtCat"
    kernel_tex = None
    label_conv = False
    grayscale = False

    def get_kernel(self):
        return SobelFilter1.get_kernel(self)


class MakeAPrediction(TeacherStudentsScene):
    def construct(self):
        # Blah
        self.teacher_says("Try thinking\nthrough what\nwill happen")
        self.play(self.change_students("pondering", "confused", "thinking", look_at=self.screen))
        self.wait()
        self.play(self.teacher.change("tease"))
        self.wait(4)


class ThinkDifferently(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.change("raise_right_hand"),
            self.change_students("pondering", "confused", "thinking", look_at=self.screen),
        )
        self.wait()
        self.play(self.teacher.change("tease"))
        self.play(self.change_students("thinking", "pondering", "tease", look_at=self.screen))
        self.wait(4)


class ThisIsTheCoolPart(TeacherStudentsScene):
    def construct(self):
        # Blah
        self.play(
            self.teacher.says("Now for the\ncool part!", mode="hooray"),
            self.change_students("happy", "tease", "well")
        )
        self.wait(2)
        self.play(
            self.change_students("thinking", "thinking", "happy", look_at=self.screen),
            self.teacher.change("tease")
        )
        self.wait(3)


class MentionONSquared(InteractiveScene):
    def construct(self):
        # Blah
        morty = Mortimer(height=2)
        kw = dict(tex_to_color_map={"{N}": YELLOW})
        bigO = MTex(R"\mathcal{O}({N}^2)", **kw)
        explanation = VGroup(
            Text("# Operations"),
            Tex("=").rotate(PI / 2),
            Tex(
                R"\text{const}",
                R"\cdot {N}^2 +",
                R"\left(\substack{\text{stuff asymptotically} \\ \text{smaller than $N^2$}}\right)",
                **kw
            ),
        )
        explanation[2][0].set_color(GREY_B)
        explanation[2][-1].scale(0.6, about_edge=LEFT)
        explanation.arrange(DOWN)
        explanation.next_to(morty, UR, SMALL_BUFF)

        self.play(morty.says(bigO))
        self.play(Blink(morty))
        self.play(
            FadeIn(explanation, 0.5 * UP),
            morty.change("raise_left_hand", look_at=explanation)
        )
        for x in range(2):
            self.play(Blink(morty))
            self.wait(2)


class MentionLinearSystem(InteractiveScene):
    def construct(self):
        # Blah
        morty = Mortimer(height=3).flip().to_edge(DOWN)
        self.play(morty.says("And it's\nlinear!", mode="hooray"))
        self.play(Blink(morty))
        self.play(morty.change("tease"))
        for x in range(2):
            self.play(Blink(morty))
            self.wait(2)


class DumbIdea(TeacherStudentsScene):
    def construct(self):
        # Bad idea
        morty = self.teacher
        stds = self.students
        self.play(
            morty.change("happy", look_at=self.screen),
            self.change_students("pondering", "pondering", "frustrated", look_at=self.screen)
        )
        self.wait()
        self.student_says("Uh...", target_mode="hesitant", bubble_direction=LEFT, index=2)
        self.play(Blink(stds[2]))
        self.student_says(
            "That's idiotic!", target_mode="angry", index=2,
            added_anims=[morty.change("guilty"), stds[1].change("erm")]
        )
        self.wait(2)

        # Samples
        self.student_says(
            TexText(R"Calculating the samples\\is already $\mathcal{O}(N^2)$"),
            target_mode="concentrating",
            look_at=self.screen,
            added_anims=[stds[0].change("erm"), stds[1].change("hesitant")]
        )
        self.wait(5)
        self.student_says(
            "And so is solving\nthe linear system!",
            target_mode="surprised",
            look_at=morty.eyes,
            added_anims=[morty.change("tease"), stds[0].change("sassy"), stds[1].change("hesitant")]
        )
        self.wait()
        self.play(stds[2].change("angry"))
        self.wait(4)

        # Hold up new screen
        new_point = 3 * UR
        self.play(
            stds[2].debubble(mode="raise_right_hand", look_at=new_point),
            stds[0].change("pondering", look_at=new_point),
            stds[1].change("pondering", look_at=new_point),
        )
        self.wait(3)
        self.play(self.change_students("erm", "hesitant", "angry"))
        self.wait(3)

        # There's a trick
        self.play(
            morty.says("But there's\na trick"),
            self.change_students("tease", "happy", "hesitant")
        )
        self.wait(2)

        fft = Text("FFT")
        fft.next_to(stds[0].get_corner(UR), UP)
        self.play(
            stds[0].change("raise_right_hand"), FadeIn(fft, UP),
            morty.change("tease", fft),
            *(pi.animate.look_at(fft) for pi in self.students[1:]),
        )
        self.wait()
        self.student_says(
            "What...is that?",
            target_mode="dance_3",
            index=1,
        )
        self.play(morty.change("happy"))
        self.wait(2)
        self.play(stds[0].change("tease"))
        self.play(stds[1].change("maybe", stds[0].eyes))
        self.wait(2)
        self.play(morty.change("well"))
        self.wait(3)


class UhWhy(InteractiveScene):
    def construct(self):
        randy = Randolph(height=3)
        randy.to_corner()
        point = LEFT_SIDE + 2 * UR

        self.play(randy.says("Uh...friendlier?", mode="sassy"))
        self.play(Blink(randy))
        self.wait()
        self.play(randy.change("erm", look_at=point))
        self.play(Blink(randy))
        self.wait()
        self.play(randy.debubble(mode="pondering", look_at=point))
        for x in range(3):
            self.play(Blink(randy))
            self.wait(2)


class GenericScreen(VideoWrapper):
    pass


class EnthusiasticAboutRunTime(TeacherStudentsScene):
    def construct(self):
        run_time = Tex(R"\mathcal{O}\big(N \log(N)\big)")
        run_time.move_to(self.hold_up_spot, DOWN)
        self.play(
            self.teacher.change("raise_right_hand"),
            FadeIn(run_time, UP),
            self.change_students("thinking", "tease", "happy", look_at=run_time),
        )
        self.wait(2)
        self.play(
            self.teacher.says("That feels\nlike magic!", mode="hooray"),
            run_time.animate.to_edge(UP),
            self.change_students("tease", "happy", "tease")
        )
        self.wait(3)
