from manim_imports_ext import *
import math
import random

class Warmup(InteractiveScene):
    def construct(self):
        # Add a number line
        numberLine = NumberLine(
            x_range = (-10, 10),
            stroke_width = 5,
            include_numbers = True,
            line_to_number_direction = DOWN,
            line_to_number_buff = 0.3
        )
        xTracker = ValueTracker(math.pi)
        xDot = Dot(radius = 0.13).set_color(YELLOW).set_opacity(0.8)
        xLabel = Tex("x = 0.000", font_size = 60).set_color(YELLOW)
        xDot.add_updater(lambda m: m.move_to(numberLine.n2p(xTracker.get_value())))
        xLabel.add_updater(lambda m: m.next_to(xDot, UP))
        xValue = xLabel.make_number_changeable("0.000")
        xValue.add_updater(lambda m: m.set_value(xTracker.get_value()))
        self.add(numberLine, xDot, xLabel)

        for _ in range(5):
            self.play(xTracker.animate.set_value(random.uniform(-6, 6)))
        # Create the complex plane
        complexPlane = ComplexPlane(
            x_range = (-10, 10),
            y_range = (-10, 10),
            axis_config = {
                "stroke_width": 5,
                "line_to_number_buff": 0.1,
                "include_ticks": True
            },
            x_axis_config = {
                "line_to_number_direction": DR
            },
            y_axis_config = {
                "line_to_number_direction": DR
            }
        )
        complexPlane.add_coordinate_labels(font_size = 20)
        real_labels = VGroup(complexPlane.coordinate_labels[0], *complexPlane.coordinate_labels[:20]).copy()
        real_labels[9].set_opacity(0)
        complexPlane.shift(-complexPlane.c2p(0, 0))
        complexPlane.coordinate_labels[:21].set_opacity(0)
        self.add(complexPlane, complexPlane.get_axes(), numberLine, complexPlane.coordinate_labels)

        zTracker = ValueTracker([math.pi, math.e])
        zDot = xDot.copy()
        zDot.clear_updaters().move_to(complexPlane.n2p(zTracker.get_value()[0] + zTracker.get_value()[1]*1j))
        zLabel = VGroup(
            Tex(r"z\ = ", font_size = 46, tex_to_color_map = {"z": YELLOW, "=": WHITE}),
            DecimalNumber(3.142, font_size = 46, num_decimal_places = 3).set_color(RED),
            VGroup(
                Tex("+", font_size = 46),
                Tex("-", font_size = 46).set_opacity(0)
            ),
            DecimalNumber(2.718, font_size = 46, num_decimal_places = 3).set_color(GREEN),
            Tex("i", font_size = 46).set_color(GREEN)
        )
        def update_label(m):
            m.arrange(buff = 0.15)
            m[0].shift(DOWN*0.03)
            m[4].shift(DOWN*0.03 + LEFT*0.1)
            m.next_to(zDot, UP)
        update_label(zLabel)
        realValue = zLabel[1]
        imagValue = zLabel[3]


        complexPlaneText = TexText(r"Complex Plane", font_size = 85).to_corner(UL, buff = 0.3)
        xDot.clear_updaters()
        xLabel.clear_updaters()
        self.play(
            AnimationGroup(
                FadeIn(complexPlane, shift = UP*0.5),
                Write(complexPlaneText),
                AnimationGroup(
                    numberLine.ticks.animate.set_opacity(0),
                    numberLine.numbers.animate.become(real_labels)
                ),
                ReplacementTransform(xDot, zDot),
                ReplacementTransform(xLabel, zLabel)
            )
        )
        # Move the point around
        zDot.add_updater(lambda m: m.move_to(complexPlane.n2p(zTracker.get_value()[0] + zTracker.get_value()[1]*1j)))
        realValue.add_updater(lambda m: m.set_value(zTracker.get_value()[0]))
        imagValue.add_updater(lambda m: m.set_value(abs(zTracker.get_value()[1])))
        def update_symbol(m):
            if zTracker.get_value()[1] >= 0:
                m[0].set_opacity(1)
                m[1].set_opacity(0)
            else:
                m[0].set_opacity(0)
                m[1].set_opacity(1)
        zLabel[2].add_updater(update_symbol)
        zLabel.add_updater(update_label)
        rect = always_redraw(
            lambda:
                SurroundingRectangle(
                zLabel,
                fill_opacity = 0.7,
                fill_color = BLACK,
                stroke_width = 0,
                buff = 0.175
            ).round_corners(0.2)
        )
        realLine = VMobject()
        imagLine = VMobject()
        def update_real_line(m):
            m.become(DashedLine([0, zDot.get_y(), 0], zDot.get_center(), stroke_width = 7).set_color(RED))
        update_real_line(realLine)
        def update_imag_line(m):
            m.become(DashedLine([zDot.get_x(), 0, 0], zDot.get_center(), stroke_width = 7).set_color(GREEN))
        update_imag_line(imagLine)
        # using this trick a lot... not sure how to take care of z indexing in a more robust way.
        self.add(complexPlane, realLine, imagLine, zDot, rect, zLabel)
        self.play(FadeOut(complexPlaneText), FadeIn(rect), ShowCreation(realLine), ShowCreation(imagLine))
        imagLine.add_updater(update_imag_line)
        realLine.add_updater(update_real_line)

        arrow = Arrow(ORIGIN, DR, thickness = 5).set_color(PINK).set_opacity(0)
        arrow.add_updater(lambda m: m.next_to(zLabel, UL, buff = 0))
        sqrt_of_negative_1 = None
        self.camera.frame.save_state()
        for i in range(10):
            move_dot_anim = zTracker.animate.set_value(
                [random.uniform(-3, 3), random.uniform(-3, 3)]
            )
            if i == 3:
                sqrt_of_negative_1 = Tex(r"= \sqrt{-1}", font_size = 27).next_to(complexPlane.coordinate_labels[29], RIGHT, buff = 0.1).shift(UP*0.03)
                self.play(move_dot_anim, CircleIndicate(complexPlane.coordinate_labels[29]), self.camera.frame.animate.reorient(0, 0, 0, (np.float32(0.29), np.float32(0.52), np.float32(0.0)), 2.36))
                self.play(Write(sqrt_of_negative_1), run_time = 1.5)
            if i == 4:
                self.play(move_dot_anim, self.camera.frame.animate.restore(), run_time = 1.5)
            elif i == 5:
                self.play(move_dot_anim, arrow.animate.set_opacity(0.7), FadeOut(sqrt_of_negative_1), run_time = 1.5)
            elif i == 8:
                self.play(move_dot_anim, FadeOut(arrow), run_time = 1.5)
            else:
                self.play(move_dot_anim, run_time = 1.5)
        self.wait(2)

        # Clean up the grid
        zLabel.clear_updaters()
        realLine.clear_updaters()
        imagLine.clear_updaters()
        rect.clear_updaters()
        zArrow = Arrow().put_start_and_end_on(complexPlane.n2p(0), zDot.get_center()).set_color(YELLOW)
        self.play(
            FadeOut(VGroup(zLabel[0][1], zLabel[1:], rect)),
            ShrinkToCenter(realLine),
            ShrinkToCenter(imagLine),
            zLabel[0][0].animate.next_to(zDot, UP, buff = 0.1),
            GrowArrow(zArrow),
            zDot.animate.shift(0)
        )
        newZLabel = zLabel[0][0]
        newZLabel.add_updater(lambda m: m.next_to(zDot, UP, buff = 0.1))
        zArrow.add_updater(lambda m: m.put_start_and_end_on(complexPlane.n2p(0), zDot.get_center()))
        f_of_z = Tex(r"f(z) = 2 \cdot z", font_size = 90, tex_to_color_map = {"f": PINK, "z": YELLOW}).to_corner(UL, buff = 1)
        self.play(Write(f_of_z))

        # Apply functions
        grid_copy = VGroup(*complexPlane.background_lines, *complexPlane.faded_lines).copy().set_color(PINK)
        self.add(grid_copy, f_of_z)
        old_vector = zArrow.copy().clear_updaters().fade(0.5)
        self.play(
            FadeIn(old_vector),
            grid_copy.animate.scale(2),
            zTracker.animate.set_value(2*np.array(zTracker.get_value())),
            VGroup(zArrow, zDot, newZLabel).animate.set_color(PINK)
        , run_time = 3)