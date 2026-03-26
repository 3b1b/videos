from manim_imports_ext import *
from custom.characters import pi_creature
import math
import random


class Warmup(InteractiveScene):
    def construct(self):
        # Add a number line
        numberLine = NumberLine(
            x_range=(-10, 10),
            stroke_width=5,
            include_numbers=True,
            line_to_number_direction=DOWN,
            line_to_number_buff=0.3
        )
        xTracker = ValueTracker(math.pi)
        xDot = GlowDot(radius=0.13, glow_factor=1).set_color(YELLOW).set_opacity(0.8)
        xDotGroup = Group(TrueDot().set_color(YELLOW), xDot)
        xLabel = Tex("0.000", font_size=60).set_fill(color=YELLOW).set_stroke(BLACK, width=6, behind=True)
        xDotGroup.add_updater(lambda m: m.move_to(numberLine.n2p(xTracker.get_value())))
        xLabel.add_updater(lambda m: m.next_to(xDotGroup, UP))
        xValue = xLabel.make_number_changeable("0.000")
        xValue.add_updater(lambda m: m.set_value(xTracker.get_value()))
        self.add(numberLine, xDotGroup, xLabel)

        random_values = [4.137, 2.961, -0.984, -1.652, 3.230]
        for i in range(5):
            self.play(xTracker.animate.set_value(random_values[i]))

        # Create the imaginary axis
        complexPlane = ComplexPlane(
            x_range=(-10, 10),
            y_range=(-10, 10),
            axis_config={
                "stroke_width": 5,
                "line_to_number_buff": 0.1,
                "include_ticks": True
            },
            x_axis_config={
                "line_to_number_direction": DR
            },
            y_axis_config={
                "line_to_number_direction": DR
            }
        )
        complexPlane.add_coordinate_labels(font_size=20)
        real_labels = VGroup(complexPlane.coordinate_labels[0], *complexPlane.coordinate_labels[:20]).copy()
        real_labels[9].set_opacity(0)
        complexPlane.shift(-complexPlane.c2p(0, 0))
        complexPlane.coordinate_labels[:21].set_opacity(0)
        self.play(
            FadeIn(VGroup(complexPlane.get_y_axis(), complexPlane.coordinate_labels), shift=UP),
            numberLine.numbers[10].animate.set_opacity(0)
        )
        sqrt_of_negative_1 = Tex(
            r"= \sqrt{-1}", font_size=27
        ).next_to(
            complexPlane.coordinate_labels[29], RIGHT, buff=0.1
        ).shift(UP * 0.03)
        self.camera.frame.save_state()
        self.play(
            CircleIndicate(complexPlane.coordinate_labels[29]),
            self.camera.frame.animate.reorient(0, 0, 0, (np.float32(0.29), np.float32(0.52), np.float32(0.0)), 2.36)
        )
        self.play(Write(sqrt_of_negative_1), run_time=1.5)
        self.wait(2)
        self.play(self.camera.frame.animate.restore(), FadeOut(sqrt_of_negative_1))

        # Create the rest of the complex plane
        zTracker = ValueTracker([math.pi, math.e])
        zDot = xDot.copy()
        zDot.clear_updaters().move_to(complexPlane.n2p(zTracker.get_value()[0] + zTracker.get_value()[1] * 1j))
        zDotGroup = Group(TrueDot().move_to(zDot), zDot).set_color(YELLOW)
        zLabel = VGroup(
            Tex(r"z\ = ", font_size=46, tex_to_color_map={"z": YELLOW, "=": WHITE}),
            DecimalNumber(3.142, font_size=46, num_decimal_places=3).set_color(RED),
            VGroup(
                Tex("+", font_size=46),
                Tex("-", font_size=46).set_opacity(0)
            ),
            DecimalNumber(2.718, font_size=46, num_decimal_places=3).set_color(GREEN),
            Tex("i", font_size=46).set_color(GREEN)
        ).set_stroke(BLACK, width=6, behind=True)
        zLabel[0].set_opacity(0)

        def update_label(m):
            m.arrange(buff=0.15)
            m[0].shift(DOWN * 0.03)
            m[4].shift(DOWN * 0.03 + LEFT * 0.1)
            m.next_to(zDotGroup, UP, buff=0.05)
        update_label(zLabel)
        realValue = zLabel[1]
        imagValue = zLabel[3]

        complexPlaneText = TexText(r"Complex Plane", font_size=85).set_stroke(BLACK, width=6, behind=True).to_corner(UL, buff=0.3)
        xDot.clear_updaters()
        xLabel.clear_updaters()
        self.add(complexPlane[2])
        self.play(
            AnimationGroup(
                FadeIn(complexPlane[:2], shift=UP * 0.5),
                Write(complexPlaneText, stroke_color=WHITE, run_time=1.6),
                AnimationGroup(
                    numberLine.ticks.animate.set_opacity(0),
                    numberLine.numbers.animate.become(real_labels)
                ),
                ReplacementTransform(xDotGroup, zDotGroup),
                ReplacementTransform(xLabel, zLabel)
            ), run_time=2)

        # Move the point around
        zDotGroup.add_updater(lambda m: m.move_to(complexPlane.n2p(zTracker.get_value()[0] + zTracker.get_value()[1] * 1j)))
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
        realLine = VMobject()
        imagLine = VMobject()

        def update_real_line(m):
            m.become(
                DashedLine(
                    complexPlane.n2p(zTracker.get_value()[0]),
                    complexPlane.n2p(zTracker.get_value()[0] + zTracker.get_value()[1] * 1j),
                    stroke_width=2
                ).set_color(RED)
            )
        update_real_line(realLine)

        def update_imag_line(m):
            m.become(
                DashedLine(
                    complexPlane.n2p(zTracker.get_value()[1] * 1j),
                    complexPlane.n2p(zTracker.get_value()[0] + zTracker.get_value()[1] * 1j),
                    stroke_width=2
                ).set_color(GREEN)
            )
        update_imag_line(imagLine)
        # using this trick a lot... not sure how to take care of z indexing in a more robust way.
        self.add(complexPlane, realLine, imagLine, zDotGroup, zLabel)
        self.play(FadeOut(complexPlaneText), ShowCreation(realLine), ShowCreation(imagLine))
        imagLine.add_updater(update_imag_line)
        realLine.add_updater(update_real_line)

        arrow = Arrow(ORIGIN, DR, thickness=5).set_color(TEAL).set_opacity(0)
        arrow.add_updater(lambda m: m.next_to(zLabel, UL, buff=0))
        sqrt_of_negative_1 = None
        zEqualsOpacityTracker = ValueTracker(0)
        zLabel[0].add_updater(lambda m: m.set_opacity(zEqualsOpacityTracker.get_value()))
        for i in range(10):
            move_dot_anim = zTracker.animate.set_value(
                [random.uniform(-3.5, 3.5), random.uniform(-3.5, 3.5)] if i < 9 else
                [random.uniform(0, 1.75), random.uniform(-1.75, 0)]
            )
            if i == 3:
                self.play(
                    move_dot_anim,
                    arrow.animate(run_time=1).set_opacity(1),
                    zEqualsOpacityTracker.animate(run_time=1).set_value(1),
                    run_time=1.5
                )
            elif i == 6:
                self.play(move_dot_anim, FadeOut(arrow), run_time=1.5)
            else:
                self.play(move_dot_anim, run_time=1.5)

        # Visualize z as a vector
        zLabel.clear_updaters()
        realLine.clear_updaters()
        imagLine.clear_updaters()
        zArrow = Arrow().put_start_and_end_on(complexPlane.n2p(0), zDotGroup.get_center()).set_color(YELLOW)
        self.play(
            FadeOut(VGroup(zLabel[0][1], zLabel[1:])),
            ShrinkToCenter(realLine),
            ShrinkToCenter(imagLine),
            zLabel[0][0].animate.next_to(zDotGroup, UR, buff=0.05),
            GrowArrow(zArrow),
            zDot.animate.shift(0)
        )
        newZLabel = zLabel[0][0]
        newZLabel.add_updater(lambda m: m.next_to(zDotGroup, UR, buff=0.05))
        zArrow.add_updater(lambda m: m.put_start_and_end_on(complexPlane.n2p(0), zDotGroup.get_center()))

        # Apply f(z) = 2*z
        f_of_z = Tex(
            r"f(z) = 2 \cdot z", font_size=90, tex_to_color_map={"f": PINK, "z": YELLOW}
        ).to_corner(UL, buff=1).set_stroke(BLACK, width=10, behind=True)
        self.play(Write(f_of_z, stroke_color=WHITE))
        grid = VGroup(*complexPlane.background_lines, *complexPlane.faded_lines)
        grid_copy = VGroup(grid.copy(), complexPlane.get_axes().copy().set_opacity(1)).set_color(PINK)
        self.add(grid_copy, f_of_z)
        self.play(FadeIn(grid_copy))
        old_vector = zArrow.copy().clear_updaters().fade(0.25)
        self.play(
            FadeIn(old_vector, run_time=0.8),
            grid_copy.animate.scale(2),
            zTracker.animate.set_value(2 * np.array(zTracker.get_value())),
            VGroup(zArrow, newZLabel).animate.set_fill(color=PINK),
            zDotGroup.animate.set_color(PINK), run_time=3)
        self.wait(1)
        self.play(
            FadeOut(VGroup(grid_copy, f_of_z, old_vector)),
            VGroup(zArrow, newZLabel).animate.set_fill(color=YELLOW),
            zDotGroup.animate.set_color(YELLOW)
        )
        self.wait(1)

        # Run f(z) = i*z on z = 1 and z = i
        f_of_z = Tex(
            r"f(z) = i \cdot z", font_size=90, tex_to_color_map={"f": PINK, "z": YELLOW}
        ).to_corner(UL, buff=1).set_stroke(BLACK, width=10, behind=True)
        self.play(Write(f_of_z, stroke_color=WHITE))
        self.play(FadeOut(Group(zArrow, zDotGroup, newZLabel)))
        oneDot = Group(
            TrueDot(),
            zDot.copy().clear_updaters().center()
        ).set_color(YELLOW).move_to(complexPlane.n2p(1))
        iDot = oneDot.copy().move_to(complexPlane.n2p(1j))
        self.play(FadeIn(oneDot))
        oneTimesi = Tex(r"i \cdot 1 = i", font_size=50).shift(UP * 1.4 + RIGHT * 2.2).set_stroke(color=BLACK, width=7, behind=True)
        oneTimesi[2].set_fill(color=YELLOW)
        oneTimesi[4:].set_fill(color=PINK)
        iTimesi = Tex(r"i \cdot i = -1", font_size=50).shift(UP * 1.4 + LEFT * 2.2).set_stroke(color=BLACK, width=7, behind=True)
        iTimesi[2].set_fill(color=YELLOW)
        iTimesi[4:].set_fill(color=PINK)
        ninetyDegreesLabel1 = Tex(r"90^\circ", font_size=28).set_stroke(width=5, color=BLACK, behind=True).shift(UR * 1)
        ninetyDegreesLabel2 = ninetyDegreesLabel1.copy().center().shift(UL * 1)
        arrow1 = CurvedArrow(
            complexPlane.n2p(1), complexPlane.n2p(1j), angle=PI * 0.45, stroke_width=7
        ).set_color([YELLOW, PINK]).fade(0.2)
        arrow1.tip.set_color(PINK)
        self.play(VFadeIn(arrow1), Write(oneTimesi, stroke_color=WHITE))
        self.wait(3)
        self.play(FadeIn(iDot))
        arrow2 = CurvedArrow(
            complexPlane.n2p(1j), complexPlane.n2p(-1), angle=PI * 0.45, stroke_width=7
        ).set_color([YELLOW, PINK]).fade(0.2)
        arrow2.tip.set_color(PINK)
        self.play(VFadeIn(arrow2), Write(iTimesi, stroke_color=WHITE))
        self.play(Write(ninetyDegreesLabel1, stroke_color=WHITE), Write(ninetyDegreesLabel2, stroke_color=WHITE))
        self.wait(3)

        # Apply f(z) = i*z to the whole grid
        zTracker.set_value([0.4, -0.7])
        zDotGroup.update()
        self.play(
            AnimationGroup(
                FadeOut(Group(oneTimesi, iTimesi, arrow1, arrow2, oneDot, iDot, ninetyDegreesLabel1, ninetyDegreesLabel2)),
                FadeIn(Group(zArrow, zDot, newZLabel)), lag_ratio=0.7)
        )
        grid_copy = VGroup(grid.copy(), complexPlane.get_axes().copy().set_opacity(1)).set_color(PINK)
        self.add(grid_copy, f_of_z)
        self.play(Group(zArrow, zDotGroup, newZLabel).animate.shift(0), FadeIn(grid_copy))
        old_vector = zArrow.copy().clear_updaters().fade(0.25)
        angleTracker = ValueTracker(math.atan(zTracker.get_value()[1] / zTracker.get_value()[0]))

        def update_zTracker_based_on_angle(t):
            r = np.linalg.norm(np.array(zTracker.get_value()))
            t.set_value(np.array([r * math.cos(angleTracker.get_value()), r * math.sin(angleTracker.get_value())]))
        zTracker.add_updater(update_zTracker_based_on_angle)
        self.play(
            FadeIn(old_vector, run_time=0.8),
            Rotate(grid_copy, PI * 0.5),
            angleTracker.animate.increment_value(PI * 0.5),
            VGroup(zArrow, newZLabel).animate.set_fill(color=PINK),
            zDotGroup.animate.set_color(PINK), run_time=3)

        rightAngle = VGroup(
            Line(ORIGIN, RIGHT).set_color([YELLOW, interpolate_color(YELLOW, PINK, 0.5)]),
            Line(RIGHT, RIGHT + UP).set_color([interpolate_color(YELLOW, PINK, 0.5), PINK])
        ).center().set_stroke(
            width=4
        ).scale(
            0.3
        ).rotate(
            math.atan(-0.7 / 0.4) + PI * 0.5
        ).shift(
            0.2 * (zArrow.get_end() - zArrow.get_start() + old_vector.get_end() - old_vector.get_start())
        )
        self.add(rightAngle)
        self.play(ShowCreation(rightAngle))
        self.wait(2)
        self.play(
            FadeOut(VGroup(grid_copy, old_vector, rightAngle)),
            VGroup(zArrow, newZLabel).animate.set_fill(color=YELLOW),
            zDotGroup.animate.set_color(YELLOW)
        )
        self.wait(1)

        # Multiplying by an arbitrary complex constant
        cDot = Group(
            TrueDot(radius=0.1), zDot.copy().clear_updaters().center().set_radius(0.3).set_glow_factor(0.8)
        ).set_color(GREEN).move_to(complexPlane.n2p(-2 - 3j))
        cLabel = Tex("c = -2 - 3i").set_color(GREEN).set_stroke(width=5, color=BLACK, behind=True).next_to(cDot, LEFT)
        self.play(
            AnimationGroup(
                f_of_z.animate.set_opacity(0),
                FadeIn(cDot),
                Write(cLabel, stroke_color=WHITE), lag_ratio=0.3)
        )
        self.wait(1)

        # Show the transormation
        zDotGroup.clear_updaters()
        zTracker.clear_updaters()
        self.play(FadeIn(grid_copy), Group(f_of_z, zArrow, zDotGroup, newZLabel, cDot, cLabel).animate.shift(0))
        scale_factor = math.sqrt(2**2 + 3**2)
        zDotGroup.add_updater(lambda m: m.move_to(complexPlane.n2p(zTracker.get_value()[0] + zTracker.get_value()[1] * 1j)))
        self.play(
            grid_copy.animate.scale(scale_factor).set_color(PINK),
            zTracker.animate.set_value([zTracker.get_value()[0] * scale_factor, zTracker.get_value()[1] * scale_factor]),
            Group(zArrow, zDotGroup).animate.set_color(PINK),
            newZLabel.animate.set_fill(color=PINK), run_time=2)
        zDotGroup.clear_updaters()
        self.play(Rotate(Group(grid_copy, zDotGroup), angle=(-360 + 236.1) * DEG), run_time=2)
        self.play(FadeOut(grid_copy))
        self.wait(3)

        # Explain how to think about f(z) = c*z
        zeroTimesC = Tex(
            r"0 \cdot c = {0}",
            tex_to_color_map={"0": YELLOW, "c": GREEN, "{0}": PINK},
            font_size=90
        ).set_stroke(
            width=5, color=BLACK, behind=True
        ).to_corner(
            UR, buff=1.6
        ).set_opacity(0)
        self.play(Write(zeroTimesC, stroke_color=WHITE), run_time=2)
        self.wait(0.5)
        pin = SVGMobject("push_pin.svg").rotate(15 * DEG).scale(0.4).set_color(GREY).align_to(complexPlane.n2p(0), DR)
        pin.set_fill([GREY_D, GREY_B], 1)
        self.wait(1)
        self.play(FadeIn(pin, shift=RIGHT * 0.25 + DOWN * 0.5), FadeOut(Group(zDotGroup, zArrow, newZLabel)))
        self.wait(1)

        oneTimesC = Tex(
            r"1 \cdot c = {c}",
            tex_to_color_map={"1": YELLOW, "c": GREEN, "{c}": PINK},
            font_size=90
        ).set_stroke(
            width=5, color=BLACK, behind=True
        ).next_to(
            zeroTimesC, DOWN, buff=0.4
        ).align_to(
            zeroTimesC, LEFT
        ).set_opacity(0)
        self.play(Write(oneTimesC, stroke_color=WHITE), run_time=2)
        self.wait(0.5)
        grid_copy = VGroup(grid.copy(), complexPlane.get_axes().copy().set_opacity(1)).set_color(PINK)
        oneDot = Group(
            TrueDot(radius=0.1),
            zDot.copy().set_radius(0.2).clear_updaters().center()
        ).set_color(
            YELLOW
        ).move_to(
            complexPlane.n2p(1)
        )
        oneTail = TracingTail(oneDot, stroke_width=5, stroke_color=YELLOW, time_traced=3)
        self.add(grid_copy, Point(), f_of_z, oneTail, oneDot, cDot, cLabel)
        self.play(FadeIn(Group(grid_copy, oneDot)), Group(cDot, cLabel, f_of_z).animate.shift(0))
        self.play(
            grid_copy.animate.rotate(236.1 * DEG).scale(math.sqrt(3**2 + 2**2)),
            oneDot.animate.move_to(cDot).set_color(PINK),
            VGroup(zeroTimesC, oneTimesC).animate.shift(0), run_time=7)
        self.wait(5)
        self.play(FadeOut(Group(grid_copy, pin, oneDot, cDot, cLabel, f_of_z, zeroTimesC, oneTimesC)))

        # Show that shape is preserved
        f_of_z = Tex(
            r"f(z) = (a + bi) \cdot z",
            font_size=50, tex_to_color_map={"f": PINK, "z": YELLOW, "a": RED, "b": GREEN}
        ).to_corner(UL, buff=0.5)
        rect1 = BackgroundRectangle(f_of_z, buff=0.3)
        aTracker = ValueTracker(1)
        bTracker = ValueTracker(0)
        aSlider = Slider(aTracker, [-3, 3], var_name="a", width=4, arrow_color=RED)
        bSlider = Slider(bTracker, [-3, 3], var_name="b", width=4, arrow_color=GREEN)
        VGroup(aSlider, bSlider).arrange(DOWN)
        rect2 = BackgroundRectangle(VGroup(aSlider, bSlider), buff=0.3)
        VGroup(aSlider, bSlider, rect2).match_width(rect1).next_to(rect1, DOWN, buff=0)

        self.play(FadeOut(numberLine.numbers), FadeOut(complexPlane.coordinate_labels))
        grid_copy = VGroup(grid.copy(), complexPlane.get_axes().copy().set_opacity(1)).set_color(PINK)
        self.play(FadeIn(grid_copy), FadeIn(rect1), FadeIn(rect2), Write(f_of_z), FadeIn(aSlider), FadeIn(bSlider))
        self.remove(oneTail)

        piCreature = SVGMobject("pi_creature_plain.svg")
        piCreature.remove(piCreature[4], piCreature[0])
        piCreatureDizzy = SVGMobject("pi_creature_dizzy.svg")
        # self.add(piCreature)
        # self.add(VGroup(piCreature, piCreatureDizzy).arrange())
        piCreaturePartPairs = [
            (piCreature[1], piCreatureDizzy[0]),
            (piCreature[6], piCreatureDizzy[2]),
            (piCreature[8], piCreatureDizzy[4]),
            (piCreature[9], piCreatureDizzy[5]),
            (piCreature[4], piCreatureDizzy[1]),
            (piCreature[7], piCreatureDizzy[3])
        ]
        # for (part1, part2) in partPairs:
        #     self.play(Indicate(part1), Indicate(part2))
        # self.play(AnimationGroup(*[Transform(part1, part2) for (part1, part2) in partPairs]))
        shapes = VGroup(
            Square(fill_opacity=1).set_color(BLUE_A).set_stroke(width=5, color=BLACK),
            Circle(fill_opacity=1).set_color(BLUE_B).set_stroke(width=5, color=BLACK),
            Triangle(fill_opacity=1).set_color(BLUE_D).set_stroke(width=5, color=BLACK),
            piCreature
        )
        shapes.arrange_in_grid(buff=1).scale(0.5)
        shapes.add(piCreatureDizzy.move_to(piCreature).match_height(piCreature).set_opacity(0))
        self.play(AnimationGroup(*[Write(shape) for shape in shapes]))

        grid_copy.save_state()
        shapes.save_state()
        dizzyTracker = ValueTracker(0)

        def update_grid(m):
            grid_copy.restore()
            shapes.restore()
            VGroup(grid_copy, shapes).apply_function(
                lambda z: complexPlane.n2p((aTracker.get_value() + bTracker.get_value() * 1j) * complexPlane.p2n(z))
            )
            for i in range(len(piCreaturePartPairs)):
                part = piCreaturePartPairs[i][0]
                start = part.copy()
                end = piCreaturePartPairs[i][1].copy().set_opacity(1)
                start.align_data(end)
                part.align_data(start)
                part.interpolate(start, end, dizzyTracker.get_value())

        grid_copy.add_updater(update_grid)
        self.add(grid_copy, Point(), shapes, Point(), VGroup(rect1, rect2, f_of_z, aSlider, bSlider))
        random_values = [
            (random.uniform(-3, 3), random.uniform(-3, 3))
            for _ in range(5)
        ]
        for i in range(5):
            self.play(
                AnimationGroup(
                    AnimationGroup(
                        aTracker.animate(run_time=4).set_value(random_values[i][0]),
                        bTracker.animate(run_time=4).set_value(random_values[i][1])
                    ),
                    dizzyTracker.animate(run_time=2).set_value(1) if i == 1 else Point().animate.shift(0), lag_ratio=0.5)
            )


class F_Of_Z_Equals_C_Times_Z(InteractiveScene):
    def construct(self):
        f_of_z = Tex(
            r"f(z) = c \cdot z",
            tex_to_color_map={"f": PINK, "z": YELLOW, "c": GREEN},
            font_size=150
        )
        self.play(Write(f_of_z), run_time=2)


class ZeroTimesAnything(InteractiveScene):
    def construct(self):
        # Write 0*c = 0
        zeroTimesC = Tex(
            r"0 \cdot c = {0}",
            tex_to_color_map={"0": YELLOW, "c": GREEN, "{0}": PINK},
            font_size=150
        )
        self.play(Write(zeroTimesC), run_time=2)


class OneTimesAnything(InteractiveScene):
    def construct(self):
        # Write 1*c = c
        oneTimesC = Tex(
            r"1 \cdot c = {c}",
            tex_to_color_map={"1": YELLOW, "c": GREEN, "{c}": PINK},
            font_size=150
        )
        self.play(Write(oneTimesC), run_time=2)


class MoreComplicatedExamples1(InteractiveScene):
    def construct(self):
        # Show f(z) = z^2
        f_of_z = Tex(
            "f(z) = z^2", font_size=120, tex_to_color_map={"f": PINK, "z": YELLOW}
        ).set_stroke(width=10, color=BLACK, behind=True)
        self.play(Write(f_of_z, stroke_color=WHITE))
        self.wait(0.5)

        # Create two planes
        x_max = 4
        in_plane, out_plane = planes = VGroup(
            ComplexPlane((-x_max, x_max), (-x_max, x_max)),
            ComplexPlane((-x_max, x_max), (-x_max, x_max)),
        )
        planes.set_height(5)
        planes.arrange(RIGHT, buff=2)

        squares = Square().get_grid(2 * x_max * 5, 2 * x_max * 5, buff=0)
        squares.replace(in_plane)
        squares.set_stroke(WHITE, 1, 0.5)

        in_plane.add_coordinate_labels(
            list(range(-x_max, x_max + 1)) + [x * 1j for x in list(range(-x_max, x_max + 1)) if x != 0],
            font_size=16
        )
        out_plane.add_coordinate_labels(
            list(range(-x_max, x_max + 1)) + [x * 1j for x in list(range(-x_max, x_max + 1)) if x != 0],
            font_size=16
        )

        moving_plane = squares.copy()
        moving_plane.insert_n_curves(10)
        moving_plane.target = moving_plane.generate_target()
        moving_plane.target.apply_function(lambda p: out_plane.n2p(in_plane.p2n(p)**2))
        moving_plane.target.set_color(PINK)
        # moving_plane.set_clip_plane(RIGHT, 8)
        # moving_plane.target.set_clip_plane(RIGHT, 0)

        in_plane.set_stroke(GREY_D, 1)
        out_plane.set_stroke(GREY_D, 1)

        # Draw the input and output space
        zCopy = f_of_z[2].copy()
        self.play(
            zCopy.animate.scale(0.5).next_to(in_plane, UP),
            f_of_z.animate.scale(0.5).next_to(out_plane, UP),
            FadeIn(in_plane, shift=UP),
            FadeIn(out_plane, shift=UP), run_time=1.5)

        # Evaluate f(2)
        f_of_2 = Tex("f({2}) = {2}^2 = 4", font_size=20, tex_to_color_map={"f": PINK, "{2}": YELLOW}).align_to(in_plane, UP)
        twoDot = Group(TrueDot(), GlowDot()).move_to(in_plane.n2p(2))
        self.play(FadeIn(twoDot))
        twoDotResult = twoDot.copy().set_color(PINK).move_to(out_plane.n2p(4))
        self.play(Write(f_of_2), TransformFromCopy(twoDot, twoDotResult, path_arc=-PI * 0.3), run_time=3)

        # Evaluate f(i)
        f_of_i = Tex("f({i}) = {i}^2 = -1", font_size=20, tex_to_color_map={"f": PINK, "{i}": YELLOW}).next_to(f_of_2, DOWN)
        iDot = Group(TrueDot(), GlowDot()).move_to(in_plane.n2p(1j))
        self.play(FadeIn(iDot))
        iDotResult = iDot.copy().set_color(PINK).move_to(out_plane.n2p(-1))
        self.play(Write(f_of_i), TransformFromCopy(iDot, iDotResult, path_arc=-PI * 0.3), run_time=3)

        # Evaluate f(-1)
        f_of_negative_1 = Tex("f({-1}) = ({-1})^2 = 1", font_size=20, tex_to_color_map={"f": PINK, "{-1}": YELLOW}).next_to(f_of_i, DOWN)
        negativeOneDot = Group(TrueDot(), GlowDot()).move_to(in_plane.n2p(-1))
        self.play(FadeIn(negativeOneDot))
        negativeOneDotResult = negativeOneDot.copy().set_color(PINK).move_to(out_plane.n2p(1))
        self.play(Write(f_of_negative_1), TransformFromCopy(negativeOneDot, negativeOneDotResult, path_arc=-PI * 0.3), run_time=3)
        self.wait(2)
        self.play(
            FadeOut(
                Group(f_of_2, twoDot, twoDotResult, f_of_i, iDot, iDotResult, f_of_negative_1, negativeOneDot, negativeOneDotResult)
            )
        )

        # Show the transformation
        square_index = 895
        for i in range(2):
            moving_plane.save_state()
            self.add(in_plane, out_plane, moving_plane, Point(), f_of_z)
            self.camera.frame.save_state()
            self.play(FadeIn(moving_plane))
            if i == 1:
                square = moving_plane[square_index]
                moving_plane.target[square_index].set_stroke(width=5, color=BLUE, opacity=1)
                self.play(
                    self.camera.frame.animate.scale(0.5, about_point=square.get_center()),
                    square.animate.set_stroke(width=5, color=BLUE, opacity=1), run_time=3)
                self.wait(2)
            self.play(self.camera.frame.animate(run_time=4).restore(), MoveToTarget(moving_plane, run_time=6))
            self.wait(2)
            if i == 0:
                self.play(FadeOut(moving_plane))
                moving_plane.restore()

        # Write "conformal map"
        conformalMapText = TexText(
            "``conformal map''", font_size=60
        ).set_stroke(
            width=13, color=BLACK, behind=True
        ).next_to(out_plane, DOWN)
        self.play(Write(conformalMapText, stroke_color=WHITE))
        self.wait(2)
        self.play(FadeOut(VGroup(moving_plane, conformalMapText)))

        squares = Square().get_grid(x_max * 5, x_max * 5, buff=0).set_width(in_plane[0].get_width() * 0.5)
        squares.set_stroke(WHITE, 1, 0.5)

        moving_plane = squares.copy()
        moving_plane.insert_n_curves(10)

        # Try f(z) = z^3
        moving_plane.align_to(in_plane, UR)
        moving_plane.generate_target()

        def func(p):
            return out_plane.n2p(in_plane.p2n(p)**3)

        moving_plane.target.apply_function(func)
        square_index = 304
        moving_plane.target[square_index].set_stroke(width=5, color=BLUE, opacity=1)
        f_of_z_2 = Tex(
            "f(z) = z^3", tex_to_color_map={"f": PINK, "z": YELLOW}
        ).set_stroke(
            width=10, color=BLACK, behind=True
        ).match_height(
            f_of_z
        ).move_to(
            f_of_z
        )
        self.play(FadeOut(f_of_z), FadeIn(f_of_z_2), FadeIn(moving_plane))
        self.play(moving_plane[square_index].animate.set_stroke(width=5, color=BLUE, opacity=1))
        target_center_in_input = moving_plane[square_index].get_center()
        self.play(MoveToTarget(moving_plane), run_time=6)
        self.add(moving_plane, Point(), f_of_z_2)

        # Try f(z) = e^z - 2iz - 3/z
        # moving_plane.align_to(in_plane[0], DL).shift(DL*0.01)
        # moving_plane.generate_target()
        # func = lambda p: out_plane.n2p(
        #     math.e**in_plane.p2n(p) - 2j*in_plane.p2n(p) - 3/(in_plane.p2n(p) if abs(in_plane.p2n(p)) > 0 else 0.001)
        # )
        # moving_plane.target.apply_function(func)
        # square_index = 70
        # moving_plane.target[square_index].set_stroke(width = 5, color = BLUE, opacity = 1)
        # f_of_z_3 = Tex(
        #     r"f(z) = e^z - 2iz - \displaystyle\frac{3}{z}", tex_to_color_map = {"f": PINK, "z": YELLOW}
        # ).set_stroke(
        #     width = 6, color = BLACK, behind = True
        # ).match_height(
        #     f_of_z
        # ).scale(1.2).move_to(
        #     f_of_z
        # ).align_to(
        #     f_of_z, UP
        # )
        # self.remove(f_of_z)
        # self.add(f_of_z_3)
        # self.wait(1)
        # self.play(FadeIn(moving_plane))
        # self.add(moving_plane, Point(), f_of_z_3)
        # self.play(moving_plane[square_index].animate.set_stroke(width = 5, color = BLUE, opacity = 1))
        # target_center_in_input = moving_plane[square_index].get_center()
        # self.play(MoveToTarget(moving_plane), run_time = 6)
        # self.wait(2)

        # Zoom in on grid to show limiting behavior
        zoomed_in_planes = []
        frame = self.camera.frame
        initial_area = frame.get_width() * frame.get_height()

        for i in range(1, 7):
            grid_res = 20
            grid_width = in_plane[0].get_width() / (2**(i + 1))

            squares = Square().get_grid(grid_res, grid_res, buff=0)
            squares.set_width(grid_width)
            squares.move_to(target_center_in_input)

            plane = squares.copy()
            plane.insert_n_curves(5)
            plane.set_stroke(width=2 / (2**i), color=WHITE)
            plane.apply_function(func)

            def update_opacity(m, index=i):
                current_area = frame.get_width() * frame.get_height()
                start_a = initial_area / (4**index)
                end_a = initial_area / (4**(index + 1))

                if start_a == end_a:
                    alpha = 1
                else:
                    alpha = (current_area - start_a) / (end_a - start_a)

                alpha = max(0, min(1, alpha))
                m.set_stroke(opacity=alpha)

            plane.add_updater(update_opacity)
            zoomed_in_planes.append(plane)
            self.add(plane)

        self.play(
            FadeOut(moving_plane[square_index]),
            frame.animate.scale(
                2**-(len(zoomed_in_planes) + 1),
                about_point=moving_plane.target[square_index].get_center()
            ),
            run_time=12
        )
        self.wait(3)


class WhyComplexNumbers(InteractiveScene):
    def construct(self):
        # Create two planes
        x_max = 4
        in_plane, out_plane = planes = VGroup(
            NumberPlane((-x_max, x_max), (-x_max, x_max)),
            NumberPlane((-x_max, x_max), (-x_max, x_max)),
        )
        planes.set_height(5)
        planes.arrange(RIGHT, buff=2)

        squares = Square().get_grid(2 * x_max * 5, 2 * x_max * 5, buff=0).match_width(in_plane)
        squares.replace(in_plane)
        squares.set_stroke(WHITE, 1, 0.5)

        moving_plane = squares.copy()
        moving_plane.insert_n_curves(10)
        moving_plane.generate_target()

        def func(p):
            return out_plane.c2p(in_plane.p2c(p)[0] + in_plane.p2c(p)[1], 2 * in_plane.p2c(p)[0] * in_plane.p2c(p)[1])

        moving_plane.target.apply_function(func)
        # moving_plane.set_clip_plane(RIGHT, 8)
        # moving_plane.target.set_clip_plane(RIGHT, 0)

        in_plane.set_stroke(GREY_D, 1)
        out_plane.set_stroke(GREY_D, 1)

        # Draw the input and output space
        f_of_x_y = Tex(
            "f(x, y) = (x + y, 2xy)",
            font_size=55,
            tex_to_color_map={"f": PINK, "x": RED, "y": GREEN}
        ).set_stroke(
            width=15, color=BLACK, behind=True
        ).next_to(
            out_plane, UP
        )
        x_y_copy = f_of_x_y["(x, y)"].copy().next_to(
            in_plane, UP
        )
        self.add(in_plane, out_plane, moving_plane, Point(), x_y_copy, f_of_x_y)

        # Show the transformation
        square_index = 499
        moving_plane.save_state()
        self.camera.frame.save_state()
        self.play(FadeIn(moving_plane))
        square = moving_plane[square_index]
        moving_plane.target[square_index].set_stroke(width=6, color=BLUE, opacity=1)
        self.play(square.animate.set_stroke(width=6, color=BLUE, opacity=1))
        self.wait(2)
        target_center_in_input = moving_plane[square_index].get_center()
        self.play(self.camera.frame.animate(run_time=4).restore(), MoveToTarget(moving_plane, run_time=6))
        self.wait(1)

        # Zoom in on grid to show limiting behavior
        zoomed_in_planes = []
        frame = self.camera.frame
        initial_area = frame.get_width() * frame.get_height()

        for i in range(1, 7):
            grid_res = 20
            grid_width = in_plane.get_width() / (2**i)

            squares = Square().get_grid(grid_res, grid_res, buff=0)
            squares.set_width(grid_width)
            squares.move_to(target_center_in_input)

            plane = squares.copy()
            plane.insert_n_curves(5)
            plane.set_stroke(width=2 / (2**i), color=WHITE)
            plane.apply_function(func)

            def update_opacity(m, index=i):
                current_area = frame.get_width() * frame.get_height()
                start_a = initial_area / (4**index)
                end_a = initial_area / (4**(index + 1))

                if start_a == end_a:
                    alpha = 1
                else:
                    alpha = (current_area - start_a) / (end_a - start_a)

                alpha = max(0, min(1, alpha))
                m.set_stroke(opacity=alpha)

            plane.add_updater(update_opacity)
            zoomed_in_planes.append(plane)
            self.add(plane)

        self.play(
            FadeOut(moving_plane[square_index]),
            frame.animate.scale(
                2**-(len(zoomed_in_planes) + 1),
                about_point=moving_plane.target[square_index].get_center()
            ),
            run_time=12
        )
        self.wait(3)


class DerivativeMeaning(InteractiveScene):
    def construct(self):
        # Plot a function
        x_range = (-3, 3)
        y_range = (-3, 3)
        axes = Axes(
            x_range=x_range,
            y_range=y_range
        ).set_stroke(width=5, color=GREY)
        axes.add_axis_labels("x", "f(x)")
        axes.add_coordinate_labels(font_size=18, excluding=[])
        axes.coordinate_labels[0][x_range[1]].set_opacity(0)
        axes.coordinate_labels[1][y_range[1]].set_opacity(0)
        self.add(axes)
        self.wait(1)

        def func(x):
            return x**3 + 3 * x**2 - x - 3

        curve = ParametricCurve(
            lambda t: (axes.c2p(t, func(t))), (-3.5, x_range[1], 0.1)
        ).set_stroke(width=8, color=BLUE, opacity=0.7)
        f_of_x = Tex(
            r"f(x) = x^3 + 3x^2 - x - 4",
            font_size=40
        ).set_color(BLUE).to_corner(UR, buff=0.5).fix_in_frame()
        self.play(Write(f_of_x, run_time=2), ShowCreation(curve, run_time=4))
        self.wait(1)

        # Draw a set of local axes labeled dx and dy
        points = curve.get_points()
        points_index = 20
        localAxes = Axes(
            x_range=(-5, 5),
            y_range=(-5, 5)
        )
        localAxes.add_axis_labels("dx", "df(x)")
        localAxes.axis_labels.set_opacity(0)
        localAxes.set_stroke(width=3, color=GREY).scale(0.1).move_to(points[points_index]).shift(RIGHT * 0.014)
        self.bring_to_back(localAxes)
        self.play(FadeIn(localAxes))

        # Zoom in on part of the graph
        self.camera.frame.save_state()
        self.play(
            FadeOut(f_of_x),
            self.camera.frame.animate(run_time=7).scale(0.008).move_to(points[points_index] + DOWN * 0.01),
            localAxes.animate(run_time=7).set_stroke(width=5)
        )

        # Show small changes in x and y
        brace_x = Brace(Line(ORIGIN, RIGHT), DOWN).set_color(YELLOW).scale(0.011).align_to(localAxes.c2p(0, 0), UL)
        delta_x = Tex(r"\Delta x", font_size=0.5).set_color(YELLOW).next_to(brace_x, DOWN, buff=0.002)
        self.play(GrowFromEdge(brace_x, UP), Write(delta_x))
        brace_f_of_x = Brace(Line(ORIGIN, UP * 2.8), RIGHT).set_color(PINK).scale(0.011).align_to(brace_x.get_corner(UR), DL)
        delta_f_of_x = Tex(r"\Delta f(x)", font_size=0.5).set_color(PINK).next_to(brace_f_of_x, RIGHT, buff=0.002)
        self.play(GrowFromEdge(brace_f_of_x, LEFT), Write(delta_f_of_x))

        # Show that delta f(x) = delta x*c
        equation = Tex(
            r"\Delta f(x)", r"\ \approx\ ", r"\Delta x", r"\ \cdot\ {c}", r"\text{(at a small scale)}",
            tex_to_color_map={r"\Delta f(x)": PINK, r"\Delta x": YELLOW, "{c}": GREEN},
        )
        equation[r"\text{(at a small scale)}"].scale(0.85).next_to(equation[:-15], DOWN)

        smallScaleOpacityTracker = ValueTracker(0)

        def update_equation(m):
            m.set_height(
                0.46 * self.camera.frame.get_height() / FRAME_HEIGHT
            ).align_to(
                self.camera.frame, UL
            ).shift(
                DR * 0.5 * self.camera.frame.get_width() / FRAME_WIDTH
            )
            m[-15:].set_opacity(smallScaleOpacityTracker.get_value())
        equation.add_updater(update_equation)
        delta_f_of_x_copy = delta_f_of_x.copy()
        delta_x_copy = delta_x.copy()
        equation_copy = equation.copy()
        self.play(
            AnimationGroup(
                ReplacementTransform(delta_f_of_x_copy, equation_copy[r"\Delta f(x)"], path_arc=PI * 0.2),
                Write(equation_copy[r"\approx"]),
                ReplacementTransform(delta_x_copy, equation_copy[r"\Delta x"], path_arc=PI * 0.2),
                Write(equation_copy[r"\ \cdot\ {c}"]), lag_ratio=0.4), run_time=2.5)
        self.remove(delta_x_copy, delta_f_of_x_copy, equation_copy)
        self.add(equation)
        self.wait(2)
        rect = SurroundingRectangle(equation[:-15], buff=0.002, fill_opacity=0, stroke_width=4, stroke_color=TEAL)
        self.play(ShowCreation(rect, run_time=2))
        self.play(FadeOut(rect))
        self.wait(1)

        # Zoom back out to see the graph
        self.play(
            AnimationGroup(
                AnimationGroup(
                    FadeOut(VGroup(localAxes, brace_x, brace_f_of_x, delta_x, delta_f_of_x)),
                    smallScaleOpacityTracker.animate(run_time=4).set_value(1),
                    self.camera.frame.animate(run_time=7).restore()
                ),
                FadeIn(f_of_x), lag_ratio=0.8)
        )
        self.wait(0.5)

        # View the function as a transformation
        self.play(FadeOut(VGroup(axes.get_y_axis(), curve)), axes.coordinate_labels[0][x_range[1]].animate.set_opacity(1))
        input_space = axes.get_x_axis()
        input_space[1].set_color(YELLOW)
        n = 250
        x_min = 0
        x_max = 3
        dots = VGroup(*[
            Dot(radius=0.001).set_stroke(width=10).set_color(
                interpolate_color(YELLOW_E, YELLOW_A, i / n)
            ).move_to(
                input_space.n2p(x_min + (i / (n - 1)) * (x_max - x_min))
            )
            for i in range(n)
        ])
        self.play(AnimationGroup(*[FadeIn(dot, shift=DOWN) for dot in dots], lag_ratio=0.003))
        input_space_group = Group(input_space, Point(), dots)
        input_space_group.generate_target()
        output_space = input_space.copy()
        f_of_x_label = Tex("f(x)").set_color(PINK)
        f_of_x_label.set_height(
            input_space[1].get_height() / f_of_x_label[1].get_height()
        ).match_y(
            output_space[1]
        ).align_to(
            output_space[1], LEFT
        )
        output_space[1].become(f_of_x_label)
        Group(input_space_group.target, output_space).arrange(buff=0.7).set_width(FRAME_WIDTH * 0.98)

        input_space_text = TexText("Input Space", font_size=60).next_to(input_space_group.target[0], DOWN, buff=0.5).set_color(YELLOW)
        output_space_text = TexText("Output Space", font_size=60).next_to(output_space, DOWN, buff=0.5).set_color(PINK)
        equation.clear_updaters()
        equation.generate_target()
        equation.target.set_x(0).to_edge(DOWN, buff=0.7)
        equation.target[-15:].next_to(equation.target[:-15], RIGHT, buff=0.25)
        self.play(
            f_of_x.animate.match_x(output_space).set_color(WHITE).set_color_by_tex_to_color_map({"f": PINK, "x": YELLOW}),
            MoveToTarget(equation, run_time=2),
            AnimationGroup(
                MoveToTarget(input_space_group),
                FadeIn(output_space, shift=LEFT * 0.5),
                FadeIn(VGroup(input_space_text, output_space_text), shift=UP * 0.4), lag_ratio=0.5),
        )
        equationRect = BackgroundRectangle(equation, buff=0.1)
        equationGroup = VGroup(equationRect, equation)
        self.add(equationGroup)
        equationGroup.fix_in_frame()
        input_dots = dots.copy()
        self.play(
            AnimationGroup(*[
                dots[i].animate(
                    path_arc=-PI * 0.5
                ).move_to(
                    output_space.n2p(func(input_space.p2n(dots[i].get_center())))
                ).set_opacity(
                    0 if func(input_space.p2n(dots[i].get_center())) < x_range[0]
                    or func(input_space.p2n(dots[i].get_center())) > x_range[1]
                    else 1
                ).set_color(
                    interpolate_color(PINK, PURPLE_A, min(1, 2.5 * i / n))
                )
                for i in range(len(dots))
            ], lag_ratio=0.01), run_time=3)

        # Look at the spacing between the dots
        output_dot_index = 50
        self.play(FadeOut(f_of_x), self.camera.frame.animate(run_time=4).move_to(dots[output_dot_index]).scale(0.1))
        humps = VGroup(*[
            ArcBetweenPoints(
                dots[i].get_center(), dots[i + 1].get_center(), angle=PI * 0.9
            ).set_stroke(
                width=5, color=TEAL
            )
            for i in range(len(dots) - 1)
            # if output_space.p2n(dots[i].get_center()) > x_range[0] + 0.01
            # and output_space.p2n(dots[i].get_center()) < x_range[1] - 0.01
        ])
        self.play(AnimationGroup(*[GrowArrow(hump) for hump in humps], lag_ratio=0.025))
        self.wait(3)

        # Zoom in on a particular output
        patch_output = dots[output_dot_index - 3: output_dot_index + 4]
        self.play(self.camera.frame.animate.move_to(patch_output).scale(0.22), run_time=4)
        self.wait(2)

        # Find the corresponding input
        self.add(input_dots)
        camera_width = self.camera.frame.get_width()
        self.play(self.camera.frame.animate.restore(), run_time=1.5)
        patch_input = input_dots[output_dot_index - 3: output_dot_index + 4].copy()
        self.play(self.camera.frame.animate.set_width(camera_width).move_to(patch_input), run_time=2.5)

        # Find a patch from the input space
        self.play(patch_input.animate.shift(UP * 0.01))

        # Show delta xs
        input_patch_humps = VGroup(*[
            ArcBetweenPoints(
                patch_input[i].get_center(), patch_input[i + 1].get_center(), angle=-PI * 0.9
            ).set_stroke(
                width=5, color=TEAL
            )
            for i in range(len(patch_input) - 1)
        ])
        delta_xs = VGroup(*[
            Tex(r"\Delta x", font_size=0.6).set_color(YELLOW).next_to(
                input_patch_humps[i], UP, buff=0.004
            )
            for i in range(6)
        ])
        self.play(AnimationGroup(*[GrowArrow(hump) for hump in input_patch_humps], lag_ratio=0.025))
        self.play(AnimationGroup(*[FadeIn(delta_x, shift=UP * 0.005) for delta_x in delta_xs], lag_ratio=0.15))

        # Move the patch over to the output space
        self.camera.frame.add_updater(lambda m: m.match_x(patch_input).match_z(patch_input))
        self.play(
            VGroup(input_patch_humps, delta_xs, patch_input).animate(
                path_arc=-PI * 0.4,
                path_arc_axis=OUT * 0.1 + DOWN * 2
            ).shift(
                RIGHT * (patch_output[3].get_x() - patch_input[3].get_x())
            ), run_time=3)
        self.wait(1)

        # Show delta f(x)s
        delta_f_of_xs = VGroup(*[
            Tex(r"\Delta f(x)", font_size=1).set_color(PINK).next_to(
                humps[output_dot_index - 3 + i], DOWN, buff=0.01
            )
            for i in range(6)
        ])
        self.play(AnimationGroup(*[FadeIn(delta_f_of_x, shift=DOWN * 0.01) for delta_f_of_x in delta_f_of_xs], lag_ratio=0.15))

        # Line up the dots
        patch_input.generate_target()
        delta_x_value = patch_input[4].get_x() - patch_input[3].get_x()
        delta_f_of_x_value = (patch_output[-1].get_x() - patch_output[0].get_x()) / (len(patch_output) - 1)
        c = delta_f_of_x_value / delta_x_value
        for i in range(len(patch_input)):
            patch_input.target[i].set_x(patch_input[3].get_x() + c * delta_x_value * (i - 3))
        scaled_humps = VGroup(*[
            ArcBetweenPoints(
                patch_input.target[i].get_center(), patch_input.target[i + 1].get_center(), angle=-PI * 0.9
            ).set_stroke(
                width=5, color=TEAL
            )
            for i in range(len(patch_input) - 1)
        ])

        delta_x_times_cs = VGroup(*[
            Tex(r"\Delta x \cdot c", tex_to_color_map={"f": PINK, r"\Delta x": YELLOW, "c": GREEN}, font_size=1).next_to(
                scaled_humps[i], UP, buff=0.01
            )
            for i in range(6)
        ])
        self.play(
            MoveToTarget(patch_input),
            ReplacementTransform(input_patch_humps, scaled_humps),
            AnimationGroup(*[
                ReplacementTransform(delta_xs[i], delta_x_times_cs[i])
                for i in range(len(delta_xs))
            ]),
            self.camera.frame.animate.shift(DOWN * 0.02),
            equationGroup.animate.scale(1.08).shift(UP * 0.35), run_time=2)
        # equation = Tex(
        #     r"\Delta f(x)", r"\ \approx\ ", r"\Delta x", r"\ \cdot\ c",
        #     tex_to_color_map = {r"\Delta f(x)": PINK, r"\Delta x": YELLOW, "c": PINK},
        #     font_size = 1.6
        # ).next_to(delta_x_times_cs, UP, buff = 0.02)
        # self.play(Write(equation), run_time = 2)
        self.wait(2)

        # Zoom back out
        self.camera.frame.clear_updaters()
        self.remove(input_space_text, output_space_text)
        f_of_x.clear_updaters().set_opacity(0)
        self.add(f_of_x)
        self.play(
            self.camera.frame.animate(run_time=3).restore().scale(1.15).shift(UP * 0.6),
            FadeOut(VGroup(equation, delta_x_times_cs, scaled_humps, patch_input, input_dots, dots, humps, delta_f_of_xs)),
            f_of_x.animate.set_opacity(1).next_to(output_space, UP, buff=2.6).shift(LEFT * 0.7 + DOWN * 0.2)
        )

        # Turn the number lines into complex planes
        x_max = x_range[1]
        in_plane, out_plane = planes = VGroup(
            ComplexPlane((-x_max, x_max), (-x_max, x_max)),
            ComplexPlane((-x_max, x_max), (-x_max, x_max)),
        )
        in_plane.match_width(input_space[0])
        in_plane.move_to(input_space.ticks[3])
        out_plane.match_width(output_space[0])
        out_plane.move_to(output_space.ticks[3])
        in_plane.add_coordinate_labels(
            list(range(-x_max, x_max + 1)) + [x * 1j for x in list(range(-x_max, x_max + 1)) if x != 0],
            font_size=16
        )
        out_plane.add_coordinate_labels(
            list(range(-x_max, x_max + 1)) + [x * 1j for x in list(range(-x_max, x_max + 1)) if x != 0],
            font_size=16
        )

        in_plane.set_stroke(GREY_D, 1)
        out_plane.set_stroke(GREY_D, 1)

        f_of_z = Tex(
            "f(z) = z^3 + 3z^2 - z - 4",
            font_size=50,
            tex_to_color_map={"f": PINK, "z": YELLOW}
        ).set_stroke(
            width=10, color=BLACK, behind=True
        ).next_to(
            out_plane, UP, buff=0.5
        ).set_x(
            out_plane.c2p(0)[0]
        )
        self.play(
            AnimationGroup(
                FadeOut(input_space),
                ReplacementTransform(input_space.numbers, in_plane.coordinate_labels[:x_max * 2 + 1]),
                FadeIn(VGroup(in_plane[:4], in_plane.coordinate_labels[x_max * 2 + 1:]), shift=UP * 0.5)
            ),
            AnimationGroup(
                FadeOut(output_space),
                ReplacementTransform(output_space.numbers, out_plane.coordinate_labels[:x_max * 2 + 1]),
                FadeIn(VGroup(out_plane[:4], out_plane.coordinate_labels[x_max * 2 + 1:]), shift=UP * 0.5)
            ),
            FadeOut(f_of_x),
            FadeIn(f_of_z)
        )

        # Show the transformation
        grid_size = 3
        squares = Square().get_grid(grid_size, grid_size, buff=0)
        squares.replace(in_plane)
        squares.set_stroke(WHITE, 1, 0.5)
        squares.set_width(0.07).move_to(in_plane.n2p(-0.8 + 0.2j))

        moving_plane = squares.copy()
        input_patch = moving_plane.copy().set_fill(color=BLACK, opacity=0.3)
        moving_plane.insert_n_curves(10)
        moving_plane.target = moving_plane.generate_target()
        moving_plane.target.apply_function(lambda p: out_plane.n2p(func(in_plane.p2n(p))))
        moving_plane.target.set_color(PINK)

        square_index = 895
        self.add(in_plane, out_plane, moving_plane, Point(), f_of_z)
        self.camera.frame.save_state()
        destination = out_plane.n2p(func(in_plane.p2n(input_patch.get_center())))
        self.play(
            AnimationGroup(
                self.camera.frame.animate.scale(0.04, about_point=moving_plane.get_center()),
                FadeIn(moving_plane), lag_ratio=0.4), run_time=2)
        self.play(self.camera.frame.animate.move_to(destination).scale(3), MoveToTarget(moving_plane), run_time=6)
        self.wait(2)

        # Find the tiny patch of squares
        self.play(self.camera.frame.animate.restore(), run_time=1.5)
        self.play(self.camera.frame.animate.scale(0.01).move_to(input_patch), FadeIn(input_patch), run_time=3)

        # Show delta zs
        delta_z_arrows = VGroup(*[
            Arrow(
                input_patch[grid_size * grid_size // 2].get_center(),
                input_patch[i].get_center(),
                buff=0,
                thickness=0.06
            ).set_color(
                YELLOW
            ).scale(
                math.sqrt(0.5) if i in [0, 2, 6, 8] else 1,
                about_point=input_patch[grid_size * grid_size // 2].get_center()
            )
            for i in range(len(input_patch)) if i != grid_size * grid_size // 2
        ])
        delta_zs = VGroup(*[
            Tex(r"\Delta z", font_size=0.5).set_color(YELLOW).next_to(arrow.get_end(), arrow.get_end() - arrow.get_start(), buff=0.1)
            for arrow in delta_z_arrows
        ])
        delta_zs[1].shift(LEFT * 0.0025)
        # delta_zs[3].shift(UP*0.001)
        # delta_zs[4].shift(UP*0.002)
        delta_zs[6].shift(LEFT * 0.0025)
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    GrowArrow(delta_z_arrows[i]),
                    FadeIn(delta_zs[i]), lag_ratio=0.6)
                for i in range(len(delta_z_arrows))
            ], lag_ratio=0.1)
        )

        # Move the patch over to the output space
        self.camera.frame.add_updater(lambda m: m.move_to(input_patch))
        self.play(
            VGroup(input_patch, delta_z_arrows, delta_zs).animate(
                path_arc=-PI * 0.01
            ).move_to(
                destination
            ),
            self.camera.frame.animate.scale(3.8), run_time=3)
        self.wait(1)

        # Show delta f(z)s
        delta_f_of_z_arrows = VGroup(*[
            Arrow(
                moving_plane.target[grid_size * grid_size // 2].get_center(),
                moving_plane.target[i].get_center(),
                buff=0,
                thickness=0.12
            ).set_color(
                PINK
            ).scale(
                math.sqrt(0.5) if i in [0, 2, 6, 8] else 1,
                about_point=moving_plane.target[grid_size * grid_size // 2].get_center()
            )
            for i in range(len(moving_plane.target)) if i != grid_size * grid_size // 2
        ])
        delta_f_of_zs = VGroup(*[
            Tex(r"\Delta f(z)", font_size=1.5).set_color(PINK).next_to(arrow.get_end(), arrow.get_end() - arrow.get_start(), buff=0.04)
            for arrow in delta_f_of_z_arrows
        ])
        delta_f_of_zs[1].shift(RIGHT * 0.015)
        delta_f_of_zs[3].shift(UP * 0.003)
        delta_f_of_zs[4].shift(DOWN * 0.004)
        delta_f_of_zs[6].shift(LEFT * 0.02)
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    GrowArrow(delta_f_of_z_arrows[i]),
                    FadeIn(delta_f_of_zs[i]), lag_ratio=0.6)
                for i in range(len(delta_f_of_z_arrows))
            ], lag_ratio=0.1),
            VGroup(input_patch, delta_z_arrows, delta_zs).animate.shift(0)
        )

        # Write delta f(z) = delta z * c
        equation = Tex(
            r"\Delta f(z)", r"\ \approx\ ", r"\Delta z", r"\ \cdot\ c",
            tex_to_color_map={r"\Delta f(z)": PINK, r"\Delta z": YELLOW, "c": GREEN},
            font_size=3
        ).next_to(moving_plane.target, LEFT, buff=0.02)
        self.camera.frame.clear_updaters()
        self.play(self.camera.frame.animate.shift(LEFT * 0.1))
        self.play(Write(equation), run_time=1)

        # Line up the grids
        delta_z_times_cs = VGroup(*[
            Tex(r"\Delta z \cdot c", font_size=1.5, tex_to_color_map={r"\Delta z": YELLOW, "c": GREEN}).next_to(arrow.get_end(), arrow.get_end() - arrow.get_start(), buff=0.04)
            for arrow in delta_f_of_z_arrows
        ])
        delta_z_times_cs[1].shift(RIGHT * 0.015)
        delta_z_times_cs[3].shift(UP * 0.003)
        delta_z_times_cs[4].shift(DOWN * 0.004)
        delta_z_times_cs[6].shift(LEFT * 0.02)
        input_patch_group = VGroup(input_patch, delta_z_arrows)
        input_patch_group.generate_target()
        input_patch_group.target.scale(4).rotate(-PI * 0.019).set_opacity(1)
        input_patch_group.target[1].set_color(PINK)
        self.play(
            MoveToTarget(input_patch_group),
            AnimationGroup(*[ReplacementTransform(delta_zs[::-1][i], delta_z_times_cs[i]) for i in range(len(delta_zs))]), run_time=2)

        self.wait(3)
