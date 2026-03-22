import pandas as pd
import gzip
from matplotlib import colormaps

from manim_imports_ext import *
from _2025.cosmic_distance.planets import *


class StatsToVenus(InteractiveScene):
    def construct(self):
        # Test
        title = TexText(R"Nearest distance to Venus $\approx$ 39,000,000 km $\approx 6{,}200 \times R_E$")
        title.to_edge(UP)
        title.set_backstroke(BLACK, 3)

        self.play(Write(title["Nearest distance to Venus "]))
        self.wait()
        self.play(Write(title[R"$\approx$ 39,000,000 km"]))
        self.wait()
        self.play(Write(title[R"$\approx 6{,}200 \times R_E$"]))
        self.wait()


class AngleDeviationForVenusParallax(InteractiveScene):
    def construct(self):
        # Test
        title = TexText(R"Angle deviation $ = 2 \tan^{-1}(1 / 6200) \approx$ 1 arc-minute = $\displaystyle \frac{1}{60} \cdot 1^\circ$")
        title.to_edge(UP)

        self.play(Write(title))
        self.wait()


class SunInTheSky(InteractiveScene):
    def construct(self):
        # Add pictures
        frame = self.frame
        base_path = self.file_writer.get_output_file_rootname().parent.parent
        pure_sun = ImageMobject(str(Path(base_path, 'Paul Animations/6. Transit Of Venus/New Transit of Venus scenes/JustSunBackgroundRemoved.png')))
        clean_sun = ImageMobject(str(Path(base_path, 'Paul Animations/6. Transit Of Venus/New Transit of Venus scenes/JustSun.tif')))
        sky_image = ImageMobject(str(Path(base_path, 'supplements/SunAndThumb.jpg')))
        sky_image.set_height(FRAME_HEIGHT)
        for sun in pure_sun, clean_sun:
            sun.set_height(0.23)
            sun.move_to([-0.63, 1.36, 0.])

        # Add angle measure
        sun_arc_minutes = 32
        sun_disk = Circle(radius=0.086)
        sun_disk.move_to(pure_sun.get_center() + 0.0055 * RIGHT + 0.002 * UP)

        protractor = NumberLine([0, 60 * 10])
        protractor.rotate(90 * DEG)
        protractor.scale(sun_disk.get_height() / protractor.get_unit_size() / sun_arc_minutes)
        protractor.next_to(sun_disk, RIGHT, buff=0.15, aligned_edge=DOWN)

        protractor.ticks.align_to(protractor.get_center(), RIGHT)
        protractor.ticks.set_width(0.005, about_edge=RIGHT, stretch=True)
        protractor.ticks[::60].set_width(0.02, about_edge=RIGHT, stretch=True)

        deg_labels = VGroup(
            Tex(Rf"{n}^\circ", font_size=2).next_to(protractor.n2p(60 * n), RIGHT, buff=0.01)
            for n in range(5)
        )
        protractor.add(deg_labels)

        # Zoom in
        self.add(sky_image)
        self.add(clean_sun)
        self.add(protractor)

        self.play(
            frame.animate.set_height(1.75 * pure_sun.get_height()).move_to(pure_sun).shift([0.1, 0.08, 0]),
            FadeIn(protractor, time_span=(2.5, 3.5)),
            FadeIn(pure_sun, time_span=(3, 4)),
            FadeIn(clean_sun, time_span=(4, 5)),
            FadeOut(sky_image, time_span=(3, 4)),
            run_time=5,
        )
        self.wait()

        # Show height of sun
        sun_label = Text(f"{sun_arc_minutes} arc-minutes").scale(1 / 24)
        sun_label.set_color(YELLOW)
        point = protractor.n2p(sun_arc_minutes)
        sun_label.next_to(point, RIGHT, buff=0.01)

        dash_length = 0.0025
        dashed_line = DashedLine(point, point + 2 * sun_disk.get_width() * LEFT, dash_length=dash_length)
        dashed_line.set_stroke(YELLOW, 2)
        dashed_lines = VGroup(
            dashed_line.copy().align_to(protractor.n2p(0), DOWN),
            dashed_line,
        )

        self.play(
            FadeIn(sun_label),
            ShowCreation(dashed_lines),
        )
        self.wait()

        # Zoom into Venus
        venus = Dot(radius=0.25 * protractor.get_unit_size()).set_fill(BLACK)
        venus.set_stroke(BLACK, 2)
        venus.set_anti_alias_width(6)
        venus.move_to(sun_disk)
        venus.align_to(protractor.n2p(8), DOWN)

        venus_line = dashed_lines[0].copy()
        venus_line.set_stroke(TEAL)
        venus_line.match_y(venus)

        venus.save_state()
        venus.shift(0.5 * sun_disk.get_width() * LEFT)
        self.play(
            Restore(venus),
            frame.animate.reorient(0, 0, 0, (-0.5, 1.37, 0.0), 0.29),
            FadeIn(venus_line, time_span=(3, 4)),
            run_time=4
        )
        self.wait()

        # Shift up and down
        labels = VGroup(
            Text("Northern Hemisphere\nObservation"),
            Text("Sourthern Hemisphere\nObservation"),
        )
        labels.scale(1 / 36)
        labels.next_to(venus, UP, buff=protractor.get_unit_size())
        labels.set_fill(BLACK)
        shift_value = 0.5 * protractor.get_unit_size() * DOWN
        labels[1].shift(shift_value)

        self.play(FadeIn(labels[0]))
        for n in range(4):
            index = n % 2
            sign = -(2 * index - 1)

            self.play(
                FadeOut(labels[index]),
                FadeIn(labels[1 - index]),
                VGroup(venus, venus_line).animate.shift(sign * shift_value)
            )
            self.wait()

        # Have lines go over appropriate parts of the disk (7 plays)
        prop1 = 312.9 / 360
        prop2 = prop1 + 3 / 360
        p1 = sun_disk.pfp(prop1)
        p2 = sun_disk.pfp(prop2)
        lines = VGroup(
            DashedLine(point + 0.675 * sun_disk.get_width() * RIGHT, [protractor.n2p(0)[0], point[1], 0], dash_length=0.5 * dash_length).set_stroke(TEAL, 1)
            for point in [p1, p2]
        )

        self.play(
            FadeOut(pure_sun),
            FadeOut(clean_sun),
            FadeOut(venus_line),
            FadeOut(venus),
            FadeOut(labels),
            FadeIn(lines),
            sun_label.animate.scale(0.5, about_edge=LEFT),
            deg_labels[0].animate.scale(0.5, about_edge=LEFT),
        )
        self.wait()


class ShowDuration(InteractiveScene):
    max_time = 4 * 3600 + 20 * 20 + 35
    run_time = 12
    clock_color = BLACK

    def construct(self):
        # Increment clock
        clock = VGroup(
            Integer(0, min_total_width=2),
            Text("h"),
            Integer(0, min_total_width=2),
            Text("m"),
            Integer(0, min_total_width=2),
            Text("s"),
        )
        clock.arrange(RIGHT, buff=SMALL_BUFF, aligned_edge=DOWN)
        clock[1::2].shift(0.5 * SMALL_BUFF * LEFT)

        clock.set_fill(self.clock_color)
        self.add(clock)

        time_tracker = ValueTracker()

        def update_clock(clock):
            time = int(time_tracker.get_value())  # In seconds
            clock[4].set_value(time % 60)  # Seconds
            clock[2].set_value((time // 60) % 60)  # Minutes
            clock[0].set_value((time // 3600))  # Minutes

        clock.add_updater(update_clock)
        self.play(time_tracker.animate.set_value(self.max_time), rate_func=linear, run_time=self.run_time)
        self.wait()


class LongerDuration(ShowDuration):
    max_time = 4 * 3600 + 20 * 20 + 35
    run_time = 12


class SevenHourMarker(InteractiveScene):
    def construct(self):
        brace = Brace(Line(2 * LEFT, 2 * RIGHT))
        label = Tex(R"\sim 7 \text{ hours}")
        label[0].next_to(label[1], LEFT, buff=0.025)
        label.next_to(brace, DOWN)
        VGroup(brace, label).set_fill(BLACK)

        self.play(GrowFromCenter(brace), Write(label))
        self.wait()


class VenusTransitTimeline(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        timeline = NumberLine(
            (1500, 2000, 10),
            tick_size=0.05,
            longer_tick_multiple=2,
            big_tick_spacing=100,
            unit_size=1 / 25
        )
        numbers =timeline.add_numbers(
            range(1500, 2050, 50),
            group_with_commas=False,
            font_size=20,
            buff=0.15
        )

        self.add(timeline)

        # Show 1761
        years = [1761, 1769, 1874]
        dots = Group(
            GlowDot(timeline.n2p(year)).set_color(YELLOW)
            for year in years
        )
        arrow = Vector(0.65 * UP, thickness=2)
        arrow.set_color(YELLOW)
        arrow.next_to(dots[0], DOWN, buff=-SMALL_BUFF)
        year_label = Text(str(years[0]), font_size=24)
        year_label.set_color(YELLOW)
        year_label.next_to(arrow, DOWN, SMALL_BUFF)

        self.play(
            GrowArrow(arrow),
            Write(year_label),
            FadeIn(dots[0], UP),
            frame.animate.set_height(6).move_to(dots[0].get_center() + 0.5 * UP).set_anim_args(run_time=2)
        )
        self.wait()

        # Next
        next_arrows = VGroup(
            Arrow(d1.get_center(), d2.get_center(), path_arc=angle, buff=0.05)
            for d1, d2, angle in zip(dots, dots[1:], [-PI, -90 * DEG])
        )
        next_labels = VGroup(
            Text(f"+{n} years", font_size=16).next_to(na, UP, SMALL_BUFF)
            for na, n in zip(next_arrows, [8, 105])
        )

        for na, label, d1, d2 in zip(next_arrows, next_labels, dots, dots[1:]):
            self.play(
                FadeIn(na),
                FadeIn(label, lag_ratio=0.1),
                TransformFromCopy(d1, d2, path_arc=-90 * DEG),
            )
            self.wait()


class Count20Minutes(ShowDuration):
    max_time = 20 * 60
    run_time = 20
    clock_color = WHITE


class Antidisk(InteractiveScene):
    def construct(self):
        # Test
        rect = FullScreenRectangle()
        disk = Dot(radius=3)
        rect = Exclusion(rect, disk)
        rect.set_fill(GREEN_SCREEN, 1)
        rect.set_stroke(width=0)
        self.add(rect)


class LabelIo(InteractiveScene):
    def construct(self):
        # Background
        im = ImageMobject('/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2025/cosmic_distance/supplements2/JovianSystem.png')
        im.set_height(FRAME_HEIGHT)
        # self.add(im)
        # self.add(NumberPlane().fade(0.25))

        # Ellipse
        orbit = Circle()
        orbit.set_shape(2.75, 0.7)
        orbit.scale(2)  # For new edit
        orbit.rotate(10 * DEG)
        point = Point()
        point.move_to(orbit.get_start())

        label = Text("Io", font_size=36)
        buff = 0.05

        def update_label(label):
            coords = point.get_center()
            dist = get_norm(coords)
            label.move_to((dist + buff) * coords / dist)
            return label

        label.add_updater(update_label)

        self.add(label)
        for _ in range(3):
            self.play(
                MoveAlongPath(point, orbit, run_time=7, rate_func=linear),
            )


class TwoAU(InteractiveScene):
    def construct(self):
        # Test
        braces = VGroup(
            Brace(Line(ORIGIN, 3 * RIGHT), UP),
            Brace(Line(3 * LEFT, ORIGIN), UP),
        )
        labels = VGroup(brace.get_text("A.U.") for brace in braces)

        self.play(
            LaggedStartMap(GrowFromCenter, braces, lag_ratio=0.5),
            LaggedStartMap(FadeIn, labels, shift=0.25 * UP, lag_ratio=0.5),
        )
        self.wait()


class SpeedOfLightFrame(InteractiveScene):
    def construct(self):
        # Test
        # title = Text("Real-time\ndepiction\nof the\nspeed of\nlight", alignment="LEFT", font_size=60)
        title = Text("The speed\nof light in\nreal time", alignment="LEFT", font_size=60)
        title.to_edge(LEFT, buff=0.25)

        h_lines = Line(LEFT, RIGHT).set_width(11).replicate(2)
        h_lines.arrange(DOWN, buff=FRAME_HEIGHT / 3)
        h_lines.to_edge(RIGHT, buff=0)
        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        v_line.move_to(h_lines, LEFT)

        lines = VGroup(v_line, h_lines)
        lines.set_stroke(WHITE, 2)

        self.add(title)
        self.add(lines)


    def old(self):
        h_line = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        h_line.next_to(title, DOWN)
        h_line2 = h_line.copy()
        h_line2.set_y(-1)
        VGroup(h_line, h_line2).set_stroke(WHITE, 2)

        self.add(title, h_line, h_line2)
        self.wait()


class CompareLightSpeedEstimates(InteractiveScene):
    def construct(self):
        # Test
        words = VGroup(
            TexText("Huygens' Estimate: 212,000 km/s"),
            TexText("True speed: 299,792 km/s"),
        )
        words.arrange(DOWN, aligned_edge=RIGHT)
        words.to_corner(UL)
        for word in words:
            self.play(Write(word))
            self.wait()


class DemonstrateAnArcSecond(InteractiveScene):
    def construct(self):
        # Add circle
        frame = self.frame
        frame.set_height(90)
        frame.set_field_of_view(10 * DEG)
        radius = 35

        circle = Circle(radius=radius)
        circle.set_stroke(WHITE, 2)

        frame = self.frame
        deg_height = radius * DEG
        zero_point = circle.get_right()

        # Add degree tick marks
        deg_ticks = VGroup(
            self.get_tick_mark(circle, n * DEG, 1, max_aparent_length=1)
            for n in range(360)
        )
        deg10_ticks = VGroup(
            self.get_tick_mark(circle, n * DEG, 2.5, max_aparent_length=1)
            for n in range(0, 360, 10)
        )
        deg30_labels = VGroup(
            self.get_tick_label(tick, number)
            for tick, number in zip(deg10_ticks[0::3], range(0, 360, 30))
        )
        deg10_labels = VGroup(
            self.get_tick_label(tick, number, visibility_range=(60, 30))
            for tick, number in zip(
                [*deg10_ticks[-2:], *deg10_ticks[1:3]],
                [340, 350, 10, 20]
            )
        )
        deg_labels = VGroup(
            self.get_tick_label(tick, number, visibility_range=(12, 6))
            for tick, number in zip(deg_ticks[1:10], range(1, 10))
        )

        self.add(circle, deg_ticks, deg10_ticks)
        self.add(deg30_labels, deg10_labels, deg_labels)

        # Add arc minute labels
        arc_minute_ticks = VGroup(
            self.get_tick_mark(circle, n * DEG / 60, 2.5e-2, max_aparent_length=0.75)
            for n in range(0, 60)
        )
        arc_minute_labels = VGroup(
            self.get_tick_label(
                tick,
                number,
                unit="arc-minutes",
                frame_prop=0.025,
                visibility_range=(0.35 * deg_height, 0.15 * deg_height),
            )
            for tick, number in zip(arc_minute_ticks[1:20], range(1, 20))
        )

        self.add(arc_minute_ticks)
        self.add(arc_minute_labels)

        # Add arc second labels
        arc_second_ticks = VGroup(
            self.get_tick_mark(circle, n * DEG / 60 / 60, 3.5e-4, max_aparent_length=0.5)
            for n in range(0, 60)
        )
        arc_second_ticks.rotate(0.25 * DEG, about_point=zero_point)

        am_height = deg_height / 60
        arc_second_labels = VGroup(
            self.get_tick_label(
                tick,
                number,
                unit="arc-seconds",
                frame_prop=0.025,
                visibility_range=(0.35 * am_height, 0.15 * am_height),
                font_size=0.1
            )
            for tick, number in zip(arc_second_ticks[1:20], range(1, 20))
        )

        self.add(arc_second_ticks)
        self.add(arc_second_labels)

        # Zoom in to one degree
        sun = get_sun(radius=(16 / 60) * deg_height, big_glow_ratio=2)
        sun.shift(zero_point - sun[0].get_bottom())
        sun.shift(0.5 * deg_height * LEFT)

        self.play(
            frame.animate.set_height(1.5 * deg_height).move_to(circle.pfp(0.5 * DEG / TAU)),
            run_time=8
        )
        self.wait(2)
        if False:
            # For insertion
            sun[0].always_sort_to_camera(self.camera)
            self.play(FadeIn(sun, 0.1 * LEFT))
            self.wait()
            self.play(FadeOut(sun))


        # Zoom in to arc seconds
        self.play(
            frame.animate.set_height(1.5 * deg_height / 60).move_to(circle.pfp(0.5 * DEG / TAU / 60)),
            run_time=4
        )
        self.wait()

        op_tracker = ValueTracker(0)
        arc_second_labels.add_updater(lambda m: m.set_opacity(op_tracker.get_value()))
        self.play(
            VGroup(circle, arc_minute_ticks, arc_second_ticks).animate.scale(20, about_point=zero_point),
            op_tracker.animate.set_value(1).set_anim_args(time_span=(1, 3)),
            run_time=4
        )
        self.wait()

    def get_tick_mark(
        self,
        circle,
        angle,
        tick_length,
        max_aparent_length=0.5,
        stroke_color=WHITE,
        stroke_width=2
    ):
        tick = Line(ORIGIN, tick_length * RIGHT)
        tick.move_to(circle.get_right())
        tick.rotate(angle, about_point=circle.get_center())
        tick.set_stroke(WHITE, stroke_width)

        frame_prop = max_aparent_length / FRAME_WIDTH
        tick.add_updater(lambda m: m.set_length(
            min(frame_prop * self.frame.get_width(), tick_length)
        ))

        return tick

    def get_tick_label(
        self,
        tick,
        number,
        unit=R"^\circ",
        buff_ratio=0.5,
        font_size=360,
        visibility_range=(1000, 900),
        frame_prop=0.05,
    ):
        label = Integer(number, unit=unit, font_size=font_size)
        if not unit.startswith("^"):
            label[-1].shift(0.5 * label[0].get_width() * RIGHT)
            if number == 1:
                label[-1][-1].scale(0, about_edge=LEFT)

        direction = normalize(tick.get_vector())
        direction[abs(direction) < 0.2] = 0

        label_height = label.get_height()

        def update_label(label):
            frame_height = self.frame.get_height()

            # Opacity
            opacity = clip(inverse_interpolate(*visibility_range, frame_height), 0, 1)
            label.set_opacity(opacity)

            # Max height
            height = min(frame_prop * frame_height, label_height)
            label.set_height(height)

            # Location
            buff = buff_ratio * label[0].get_width()
            label.next_to(tick.get_end(), direction, buff=buff)
            return label

        label.add_updater(update_label)

        return label


class ArcMinuteLabels(InteractiveScene):
    def construct(self):
        # Test
        brace = Brace(Line(3 * DOWN, 3 * UP), LEFT)
        am_label = brace.get_text("60 arc-minutes")
        as_label = Text("60 arc-seconds", t2s={"seconds": ITALIC})
        as_label["seconds"].set_color(YELLOW)
        as_label.move_to(am_label, RIGHT)

        for label in [am_label, as_label]:
            self.clear()
            self.play(
                GrowFromCenter(brace),
                FadeIn(label, scale=3, shift=1 * LEFT)
            )
            self.wait()


class ConnectingLine(InteractiveScene):
    def construct(self):
        line = Line(2 * UP, 2 * DOWN)
        line.set_stroke(GREEN, 8)
        self.play(ShowCreation(line))
        self.play(line.animate.set_stroke(width=0), run_time=2)


class LightYearLabel(InteractiveScene):
    def construct(self):
        label = Text("4.25 Light years")
        self.play(Write(label))
        self.wait()
        self.play(FadeOut(label))


class CompareTwoStars(InteractiveScene):
    def construct(self):
        # Set up two stars
        frame = self.frame
        stars = Group(
            GlowDot(UR, color=WHITE, radius=0.2),
            GlowDot(DR, color=WHITE, radius=0.8),
        )
        randy = Randolph(mode="pondering", height=1)
        randy.shift(4 * LEFT - randy.eyes[1].get_center())
        randy.look_at(stars[0])

        frame.reorient(-88, 87, 0, (-4.0, -0.0, 0.0), 0.19)
        self.add(stars)
        self.wait()
        self.play(
            frame.animate.to_default_state(),
            FadeIn(randy, time_span=(1, 2)),
            run_time=3,
        )

        # Show observation lines
        lines = always_redraw(lambda: VGroup(
            DashedLine(randy.eyes[1].get_right(), stars[0].get_center()),
            DashedLine(randy.eyes[1].get_right(), stars[1].get_center()),
        ).set_stroke(WHITE, 1))

        self.play(*map(ShowCreation, lines))
        self.add(lines)
        self.wait()
        self.play(
            stars[0].animate.shift(lines[0].get_vector()).scale(4),
            run_time=2
        )
        self.play(Blink(randy))
        self.wait()


class InverseSquareLaw(InteractiveScene):
    def construct(self):
        # Initial area
        frame = self.frame
        frame.reorient(43, 75, 0)

        light = GlowDot(color=WHITE, radius=0.5)
        axes = ThreeDAxes()
        axes.set_stroke(width=1)
        self.add(axes)

        sphere = Sphere(radius=2)
        sphere.set_color(WHITE, 0.25)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere, resolution=(41, 21))
        mesh.set_stroke(WHITE, 1, 0.25)

        self.add(light)

        # Show eminating light
        big_sphere = sphere.copy().pointwise_become_partial(sphere, 0.5, 1)
        theta, phi = frame.get_euler_angles()[:2]
        big_sphere.rotate(phi, UP, about_point=ORIGIN)
        big_sphere.rotate(theta, IN, about_point=ORIGIN)
        big_sphere.always_sort_to_camera(self.camera)
        big_sphere.scale(3, about_point=ORIGIN)

        radiation, op_tracker = self.beaming_effect(big_sphere, opacity_range=(0.15, 0), n_components=20, speed=0.5)

        op_tracker.set_value(0)
        self.add(radiation)
        self.play(op_tracker.animate.set_value(1), run_time=2)
        self.wait(6)
        self.add(sphere, mesh, radiation)
        self.play(
            op_tracker.animate.set_value(0),
            FadeIn(sphere),
            FadeIn(mesh),
        )
        self.remove(radiation)
        self.wait()

        # Show patch
        patch = ParametricSurface(
            lambda u, v: sphere.uv_func(u, v),
            u_range=(0 * DEG, 9 * DEG),
            v_range=(90 * DEG, 99 * DEG),
        )
        patch.set_color(WHITE)

        beam, op_tracker = self.beaming_effect(patch)

        op_tracker.set_value(0)
        self.add(beam, sphere, mesh)
        self.play(
            op_tracker.animate.set_value(1),
            sphere.animate.set_opacity(0.1),
            mesh.animate.set_stroke(opacity=0.1),
            FadeIn(patch),
        )
        self.wait(2)
        self.play(frame.animate.reorient(72, 75, 0, (-0.12, -0.13, 0.0), 8), run_time=10)

        # Compare to full sphere
        sphere_highlight = sphere.copy()
        sphere_highlight.scale(1.01)
        sphere_highlight.set_color(BLUE, 0.2)

        self.play(ShowCreation(sphere_highlight, run_time=2))
        self.play(FadeOut(sphere_highlight))
        self.wait(3)

        # Grow the sphere
        patch.target = patch.generate_target()
        patch.target.scale(2, about_point=ORIGIN)

        division = VGroup(
            ParametricCurve(lambda t: 2 * sphere.uv_func(t, 94.5 * DEG), t_range=(0 * DEG, 9 * DEG, DEG)),
            ParametricCurve(lambda t: 2 * sphere.uv_func(4.5 * DEG, t), t_range=(90 * DEG, 99 * DEG, DEG)),
        )
        division.set_stroke(GREY_C, 1)

        sphere_group = Group(sphere, mesh)
        ghost_sphere = sphere_group.copy()
        ghost_sphere[0].set_opacity(0.05)
        ghost_sphere[1].set_stroke(opacity=0.05)
        ghost_sphere.scale(0.999)

        self.add(ghost_sphere)
        self.play(
            sphere_group.animate.scale(2),
            MoveToTarget(patch),
            frame.animate.reorient(65, 78, 0, ORIGIN, 8).set_anim_args(run_time=3),
            run_time=2
        )
        self.play(FadeIn(division))
        self.wait(6)

        self.play(
            patch.animate.scale(0.5, about_edge=IN + DOWN).set_opacity(0.5),
            FadeOut(division),
            run_time=2
        )
        self.play(
            frame.animate.reorient(103, 78, 0, (-0.0, 0.0, 0.0), 8.00),
            run_time=18,
        )

    def beaming_effect(self, piece, n_components=20, speed=0.5, opacity_range=(0.5, 0.25)):
        pieces = piece.replicate(n_components)
        d_alpha_range = np.arange(0, 1, 1.0 / n_components)
        radius = get_norm(piece.get_right())

        master_opacity_tracker = ValueTracker(1)

        def update_pieces(pieces):
            beam_time = radius / speed
            alpha = self.time / beam_time

            for subpiece, d_alpha in zip(pieces, d_alpha_range):
                sub_alpha = (alpha + d_alpha) % 1
                subpiece.become(piece)
                pre_opacity = interpolate(*opacity_range, sub_alpha)
                subpiece.set_opacity(pre_opacity * master_opacity_tracker.get_value())
                subpiece.scale(0.99 * sub_alpha, about_point=ORIGIN)

            pieces.sort(get_norm)

            return pieces

        pieces.add_updater(update_pieces)
        return pieces, master_opacity_tracker


class WriteInverseSquareLaw(InteractiveScene):
    def construct(self):
        # Test
        eq = Tex(R"\text{Apparent brightness} = \text{const} \cdot {\text{Absolute brightness} \over (\text{distance})^2}")
        eq.to_edge(UP)
        eq.set_color(GREY_B)

        self.add(eq)
        self.wait()
        words = ["Apparent brightness", "Absolute brightness", "distance"]
        colors = [YELLOW, WHITE, BLUE]
        for text, color in zip(words, colors):
            part = eq[text][0]
            self.play(
                FlashAround(part, color=color, buff=0.1),
                part.animate.set_color(color)
            )
            self.wait()


class MeasuringNearbyStars(InteractiveScene):
    def construct(self):
        # Add the sun
        frame = self.frame
        sun = GlowDot(color=WHITE, radius=0.1)
        sun.center()
        self.add(sun)

        frame.reorient(--270, 63, 0)
        dtheta_tracker = ValueTracker(5 * DEG)
        frame.add_updater(lambda m, dt: m.increment_theta(dtheta_tracker.get_value() * dt))
        self.add(frame)

        # Add the stars
        n_stars = 10000
        n_shown_stars = 1000
        data_file = '/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2025/cosmic_distance/Data/HYG_Data.gz'
        full_stellar_data, df = self.read_hyg_data(data_file)

        random.shuffle(full_stellar_data)
        stellar_data = full_stellar_data[:n_stars]
        abs_mags = stellar_data[:, 0]
        color_index = stellar_data[:, 1]
        rgbas = self.color_index_to_rgb(color_index)

        opacities = np.ones(n_stars)
        opacities[n_shown_stars:] = 0
        rgbas[n_shown_stars:, 3] = 0

        star_points = np.random.uniform(-1, 1, (n_stars, 3))
        distances = np.random.uniform(0.5, 36, n_stars)**0.5
        star_points *= (distances / np.linalg.norm(star_points, axis=1))[:, np.newaxis]
        stars = GlowDots(star_points, color=WHITE)

        radii = 0.1 * (abs_mags.max() - abs_mags) / (abs_mags.max() - abs_mags.min())
        stars.set_radii(radii)
        stars.set_opacity(opacities)

        self.add(stars)

        # Show distances to various stars
        last_group = VGroup()
        for n in range(10):
            line = Line(ORIGIN, random.choice(star_points))
            line.set_stroke(BLUE, 2)
            label = DecimalNumber(line.get_length() * 10, num_decimal_places=2, unit="L.Y.", font_size=24)
            label[-1].shift(SMALL_BUFF * RIGHT)
            label.set_color(BLUE)
            label.rotate(frame.get_phi(), RIGHT)
            label.rotate(frame.get_theta() + 2 * DEG, OUT)
            vect = normalize(np.cross(line.get_end(), frame.get_implied_camera_location()))
            label.next_to(line.pfp(0.33), vect, SMALL_BUFF)
            self.play(
                ShowCreation(line),
                FadeIn(label, shift=0.25 * OUT),
                FadeOut(last_group),
            )
            self.wait()
            last_group = VGroup(line, label)
        self.play(FadeOut(last_group))

        # Show the colors
        self.play(stars.animate.set_rgba_array(rgbas))
        self.wait(4)

        # Compile into a H.R. plot
        axes = Axes((0, 1, 0.1), (0, 1, 0.1), width=6, height=6)

        x_axis_label = Text("Color", font_size=36)
        x_axis_label.next_to(axes.x_axis, DOWN, SMALL_BUFF)
        y_axis_label = Text("Absolute\nbrightness", font_size=36)
        y_axis_label.next_to(axes.y_axis, LEFT, SMALL_BUFF)

        rand_x = np.random.random(n_stars)
        rand_y = np.random.random(n_stars)
        random_points = axes.c2p(rand_x, rand_y)

        color_alphas = inverse_interpolate(color_index.min(), color_index.max(), color_index)
        mag_alphas = inverse_interpolate(abs_mags.max(), abs_mags.min(), abs_mags)

        sorted_by_color = axes.c2p(color_alphas, rand_y)
        fully_sorted = axes.c2p(color_alphas, mag_alphas)

        new_radii = 0.5 * (radii + 0.05) / 1.5

        self.play(dtheta_tracker.animate.set_value(0))
        self.play(
            FadeIn(axes),
            frame.animate.to_default_state(),
            stars.animate.set_points(random_points).set_radii(new_radii).set_glow_factor(0).make_3d().set_opacity(0.5),
            FadeOut(sun),
            run_time=3
        )
        self.play(
            stars.animate.set_points(sorted_by_color).set_anim_args(run_time=5, path_arc=30 * DEG),
            Write(x_axis_label, run_time=2),
        )
        self.wait()
        self.play(
            stars.animate.set_points(fully_sorted).set_anim_args(run_time=5).set_opacity(0.25),
            Write(y_axis_label, run_time=2),
        )
        self.wait()

        # Circle the main sequence
        ms_circle = Circle().set_stroke(YELLOW, 2)
        ms_circle.set_shape(4.5, 1)
        ms_circle.rotate(-35 * DEG)
        ms_circle.move_to(axes.c2p(0.3, 0.35))

        ms_label = Text("Main sequence", font_size=30)
        ms_label.next_to(ms_circle.pfp(0.1), RIGHT)

        self.play(
            frame.animate.reorient(0, 0, 0, (-0.75, -1.45, 0.0), 5.71).set_anim_args(run_time=3),
            ShowCreation(ms_circle),
            Write(ms_label)
        )
        self.wait()

        # Name the diagram
        name = Text("Hertzsprung–Russell diagram")
        name.center().to_edge(UP)
        name.fix_in_frame()

        self.play(
            Write(name),
            frame.animate.reorient(0, 0, 0, 0.5 * UP, 9),
            run_time=2,
        )
        self.wait()

        # Move around an example star
        example_star = TrueDot().make_3d()
        glow = self.get_glow(example_star)

        def update_example_star(star):
            color_alpha, mag = axes.p2c(star.get_center())
            bv_index = interpolate(color_index.min(), color_index.max(), color_alpha)
            star.set_rgba_array(self.color_index_to_rgb(np.array([bv_index])))
            star.set_radius(0.25 * mag)

        example_star.add_updater(update_example_star)
        example_star.move_to(axes.c2p(0.5, 0.5))

        self.play(
            FadeOut(ms_label),
            FadeOut(ms_circle),
        )
        self.wait()
        self.play(
            stars.animate.set_opacity(0.02),
            FadeIn(example_star),
            FadeIn(glow),
        )
        for x in [-2.5, 0]:
            self.play(example_star.animate.set_x(x), run_time=1.5)

        for y in [2, 0]:
            self.play(example_star.animate.set_y(y), run_time=2)
        self.wait()

        # Place into the main sequence
        opacities = stars.get_opacities().copy()
        ur_values = np.dot(stars.get_points(), np.array([[1, 1.2, 0]]).T).flatten()
        in_ms = ur_values < -1.3
        opacities[in_ms] = 0.2

        ms_arrow = Arrow(UL, DR).rotate(10 * DEG)
        ms_arrow.set_color(GREY_B)
        ms_arrow.move_to(axes.c2p(0.25, 0.25))

        self.play(
            stars.animate.set_opacity(opacities),
            ShowCreationThenFadeOut(ms_circle),
            FadeIn(ms_label),
            example_star.animate.move_to(axes.c2p(0.1, 0.5)),
            run_time=2
        )
        self.play(GrowArrow(ms_arrow))
        self.wait()
        opacities[in_ms] = 0.025

        # Show radiation and shifting down
        self.play(stars.animate.set_opacity(opacities).set_anim_args(run_time=1))
        self.wait(6)
        self.play(
            example_star.animate.move_to(axes.c2p(0.5, 0.2)).set_anim_args(run_time=15),
        )
        self.wait()

        # Return full diagram
        self.play(
            FadeOut(ms_arrow),
            FadeOut(ms_label),
            FadeOut(example_star),
            FadeOut(glows),
            stars.animate.set_opacity(0.25),
        )
        self.wait()

        # Scan through color regions
        color_x_tracker = ValueTracker(0)
        opacities = stars.get_opacities().copy()

        def update_opacities(stars):
            mid_x = color_x_tracker.get_value()
            xs = stars.get_points()[:, 0]
            opacities[:] = 0.25 * np.exp(-15 * (xs - mid_x)**2) + 0.01
            stars.set_opacity(opacities)

        stars.add_updater(update_opacities)
        self.wait()
        for x in [-2.5, 1.5, 0]:
            self.play(color_x_tracker.animate.set_value(x), run_time=5)
        self.wait()

        stars.clear_updaters()
        self.play(stars.animate.set_opacity(0.25))
        self.wait()

        # Highlight other regions

    def read_hyg_data(self, file_path):
        """
        Read HYG Database from a gzipped file into a numpy array

        Parameters:
        file_path (str): Path to the HYG_Data.gz file

        Returns:
        numpy.ndarray: Array containing the stellar data
        pd.DataFrame: Original dataframe for reference if needed
        """
        # Read the gzipped CSV file
        with gzip.open(file_path, 'rt') as f:
            # Read into pandas first since the file is CSV formatted
            df = pd.read_csv(f)

        # For the H-R diagram, we primarily need:
        # - Color index (B-V)
        # - Absolute magnitude
        # Essential columns for H-R diagram
        essential_cols = ['absmag', 'ci']

        # Create numpy array from essential columns
        stellar_data = df[essential_cols].to_numpy()

        # Remove any rows with NaN values
        stellar_data = stellar_data[~np.isnan(stellar_data).any(axis=1)]

        return stellar_data, df

    def color_index_to_rgb(self, bv_index):
        alpha = inverse_interpolate(-0.2, 2.9, bv_index)
        red = "#FF0000"
        cmap = get_colormap_from_colors(["#0000FF", BLUE, WHITE, YELLOW, ORANGE, RED, * 4 * [red]])
        return cmap(alpha)

    def get_glow(self, star):
        glows = GlowDot().replicate(2)

        def update_glows(glows):
            for dot, delta_t in zip(glows, [0, 1]):
                alpha = (self.time + delta_t) % 2
                if alpha < 1:
                    dot.set_opacity(1)
                    dot.set_radius(interpolate(1, 8, alpha) * star.get_radius())
                else:
                    dot.set_opacity(interpolate(1, 0, alpha - 1))
                dot.set_color(star.get_color())
                dot.move_to(star)

        glows.add_updater(update_glows)
        return glows


class WriteHarvardComputer(InteractiveScene):
    def construct(self):
        words = Text("Harvard Computers", font_size=72)
        words.set_backstroke(BLACK, 8)
        words.to_corner(UL)
        self.play(Write(words))
        self.wait()


class LineToDistantStar(InteractiveScene):
    def construct(self):
        # Add Galaxy
        path = os.path.join(self.file_writer.get_output_file_rootname().parent, "GalaxyStill.png")
        galaxy = ImageMobject(path)
        galaxy.set_height(FRAME_HEIGHT)
        galaxy.center()
        self.add(galaxy)

        # Draw line
        sun_point = (-0.08, -0.74, 0.0)
        star_point = (0.6, 1.54, 0.0)
        line = Line(sun_point, star_point)
        line.set_stroke(TEAL, 4)
        line_label = TexText(R"$\sim$60,000 Light Years", font_size=36)
        line_label.next_to(line.get_center(), RIGHT)
        line_label.set_color(TEAL)
        line_label.set_backstroke(BLACK, 3)

        star = GlowDot(star_point, color=WHITE)
        star.f_always.set_radius(lambda: 0.1 * (2 + math.sin(self.time)))

        self.play(
            ShowCreation(line),
            FadeIn(line_label, lag_ratio=0.1, time_span=(1, 3)),
            galaxy.animate.set_opacity(0.75),
            FadeIn(star),
            run_time=3,
        )
        self.wait(12)


class SolarSpectrum(InteractiveScene):
    def construct(self):
        # Get data
        data_file = '/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2025/cosmic_distance/Data/solar_spectrum.csv'
        self.spectral_cmap = colormaps.get_cmap("Spectral")
        data = np.loadtxt(data_file, delimiter=',')

        wavelength = self.bucket_data(data[:, 0])
        avg_intensity = self.bucket_data(data[:, 1:4].mean(1))

        # Add axes
        axes = Axes((0, 2400, 100), (0, 1, 0.1), height=6, width=11)
        axes.to_corner(UL)

        x_label = Text("Wavelength (nm)", font_size=36)
        x_label.next_to(axes.x_axis.get_end(), UR, buff=SMALL_BUFF)
        x_label.shift_onto_screen(buff=SMALL_BUFF)
        x_coords = VGroup(
            Integer(n, font_size=24).next_to(axes.x_axis.n2p(n), DOWN, MED_SMALL_BUFF)
            for n in range(0, 2800, 400)
        )
        x_label.set_z_index(1)

        self.add(axes)
        self.add(x_label)
        self.add(x_coords)

        # Add lines
        lines = self.get_spectral_lines(axes, wavelength, avg_intensity)

        self.play(LaggedStart(
            (Write(line, stroke_color=WHITE, stroke_width=0.1)
            for line in lines),
            lag_ratio=0.005,
            run_time=5
        ))

        # Show the lump
        def func(wavelen):
            # Not even the right formula...
            T = 1
            c = 1
            h = 1
            freq = c / (wavelen / 2400)
            return (2 * h * freq**5) / (c**2) / (np.exp(h * freq / T) - 1)

        lump = axes.get_graph(func, x_range=(1e-3, 2400, 12))
        lump.set_height(1 * axes.y_axis.get_length(), about_edge=DOWN, stretch=True)

        self.play(VShowPassingFlash(lump, run_time=4, time_width=1.5))

        # Expand
        origin = axes.c2p(0, 0)
        self.play(
            lines.animate.stretch(2, 0, about_point=origin).set_stroke(width=1),
            axes.x_axis.animate.stretch(2, 0, about_point=origin),
            *(
                coord.animate.stretch(2, 0, about_point=origin).stretch(0.5, 0)
                for coord in x_coords
            ),
            run_time=3
        )

        # Highlight gaps
        gap_points = np.array([
            (-3.77, -1.1, 0.0),
            (-2.96, 1.09, 0.0),
            (-2.58, 2.51, 0.0),
            (-1.74, 3.0, 0.0),
        ])
        arrows = VGroup(
            Vector(0.5 * DOWN, thickness=2).set_color(WHITE).move_to(point, DOWN)
            for point in gap_points
        )

        self.play(LaggedStartMap(GrowArrow, arrows, lag_ratio=0.5))
        self.wait()

        # Show hydrogen spectrum
        spectral_wavelengths = np.array([
            122, 103, 97,  # Lyman
            656, 486, 430, 410,  # Balmer
            1875, 1282, 1094,  # Paschen
        ])
        spectral_lines = self.get_spectral_lines(axes, spectral_wavelengths, np.ones_like(spectral_wavelengths), stroke_width=3)
        for line in spectral_lines:
            line.set_stroke(interpolate_color(line.get_color(), WHITE, 0.25), width=2)
        # spectral_lines.set_height(6, about_edge=DOWN, stretch=True)

        hyd_words = Text("Spectral lines for Hydrogen", font_size=72)
        hyd_words.next_to(spectral_lines, UP, LARGE_BUFF)

        self.play(
            LaggedStartMap(ShowCreation, spectral_lines),
            FadeOut(arrows),
            lines.animate.set_stroke(opacity=0.2),
            self.frame.animate.reorient(0, 0, 0, (3.42, 0.49, 0.0), 12.00).set_anim_args(run_time=2),
            Write(hyd_words, run_time=3),
            x_label.animate.set_x(12.5),
        )
        self.wait()

        # Show red shift
        shift_value = 0.25 * RIGHT
        new_lines = spectral_lines.copy()
        new_lines.shift(shift_value)

        arrow = Vector(5 * RIGHT, thickness=10)
        arrow.move_to(hyd_words, DOWN)

        self.play(FadeOut(hyd_words), lines.animate.set_stroke(opacity=0.1))
        self.play(
            GrowArrow(arrow),
            TransformFromCopy(spectral_lines, new_lines),
            spectral_lines.animate.set_stroke(width=1, opacity=0.5),
        )
        self.wait()

        # Measure the shift
        brace = Brace(VGroup(spectral_lines[3], new_lines[3]), UP, SMALL_BUFF)
        self.play(GrowFromCenter(brace), FadeOut(arrow))
        self.play(
            brace.animate.stretch(3, 0, about_edge=LEFT),
            new_lines.animate.shift(2 * shift_value),
            rate_func=there_and_back,
            run_time=5
        )

    def bucket_data(self, arr, bucket_size=10):  # TODO, change
        full_size = len(arr) - (len(arr) % bucket_size)
        compressed = arr[:full_size].reshape((full_size // bucket_size, bucket_size)).min(1)
        return compressed

    def color_alpha_to_color(self, alpha):
        rgb = np.array(self.spectral_cmap(1 - alpha)[:3])
        brown = np.array([0.55, 0.27, 0.07])
        grey = np.array([0.15, 0.15, 0.15])
        if alpha > 1:
            factor = (np.exp(2 * (1 - alpha)) + 0.2) / 1.2
            rgb = interpolate(grey, rgb, factor)
        elif alpha < 0:
            rgb = interpolate(brown, rgb, np.exp(5 * alpha))
        return Color(rgb=rgb)

    def get_spectral_lines(self, axes, wavelengths, intensities, stroke_width=1):
        color_alpha = inverse_interpolate(380, 780, wavelengths)
        rel_intensities = intensities / intensities.max()
        return VGroup(
            Line(
                axes.c2p(lam, 0),
                axes.c2p(lam, y),
                stroke_color=self.color_alpha_to_color(ca),
                stroke_width=stroke_width
            )
            for lam, y, ca in zip(wavelengths, rel_intensities, color_alpha)
        )


class LeavittLabel(InteractiveScene):
    def construct(self):
        text = TexText(R"From Leavitt's\\1912 Paper")
        text.to_corner(UL)
        self.play(Write(text))
        self.wait()


class GalaxyFarFarAway(InteractiveScene):
    def construct(self):
        # Test
        words = Text("A Galaxy Far\n Far Away", font_size=120)
        words.set_color(YELLOW)
        words.set_backstroke(BLACK, 5)
        self.add(words)

        self.frame.reorient(0, 60, 0)
        self.play(words.animate.shift(20 * UP), run_time=5)


class GalacticSurveyData(InteractiveScene):
    def construct(self):
        # Gather data
        frame = self.frame
        data_file = '/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2025/cosmic_distance/Data/galactic_data.csv'
        # Columns are 'objID, ra, dec, redshift, distance_mpc'
        data = np.loadtxt(data_file, delimiter=',', skiprows=2)
        right_ascension = data[:, 1]
        declination = data[:, 2]
        distance = data[:, 4]

        cos_ra = np.cos(right_ascension * DEG)
        sin_ra = np.sin(right_ascension * DEG)
        cos_dec = np.cos(90 * DEG - declination * DEG)
        sin_dec = np.sin(90 * DEG - declination * DEG)

        cartesian = np.array([cos_dec * sin_ra, sin_dec * sin_ra, cos_ra]).T
        cartesian *= distance[:, np.newaxis]

        # Add dots
        dots = DotCloud(cartesian)
        dots.get_width()
        dots.get_height()
        dots.set_color(WHITE)
        dots.set_glow_factor(0.5)

        dots.clear_updaters()
        radii = np.random.random(len(distance))**2
        rad_factor = ValueTracker(0.003)
        dots.f_always.set_radii(lambda: max((rad_factor.get_value() * frame.get_height()**0.9), 0.1) * radii)

        self.add(dots)

        # Zoom out
        # frame.rleorient(-30, 146, 0, (1.89, 0.98, -10.17), 10.71)
        height_tracker = ValueTracker(10)
        angle_tracker = ValueTracker(np.array([-30, 146, 0]))
        center_tracker = ValueTracker(np.array([1.89, 0.98, -10.17]))

        frame.reorient(-22, 151, 0, (0.73, 0.79, -2.36), 13.45)
        self.play(
            frame.animate.reorient(-12, 129, 0, (-18.76, 7.73, -104.01), 439.75),
            rad_factor.animate.set_value(0.0025),
            run_time=12
        )
        self.wait()
        self.play(
            frame.animate.reorient(-115, 86, 0, ORIGIN, 1200),
            rad_factor.animate.set_value(0.002),
            run_time=10
        )
        self.play(
            frame.animate.reorient(107, 87, 0),
            run_time=15,
        )
        self.play(
            frame.animate.reorient(17, 100, 0, (-69.76, -40.26, -81.8), 186.35),
            rad_factor.animate.set_value(0.003),
            run_time=10,
        )
        self.play(
            frame.animate.reorient(-84, 85, 0, ORIGIN, 1100),
            rad_factor.animate.set_value(0.002),
            run_time=10,
        )


class RungsUpToGalaxies(InteractiveScene):
    def construct(self):
        # Test
        rungs = VGroup(
            Text("Distance to the sun"),
            Text("Stellar parallax"),
            Text("Main sequence fitting"),
            Text("Cepheid periods"),
            Text("Red shift"),
        )
        rungs.arrange(UP, buff=LARGE_BUFF, aligned_edge=LEFT)
        rungs.to_edge(LEFT)

        self.add(rungs)
        for n in [0, 1, 2, 4, 3, 4]:
            rungs.target = rungs.generate_target()
            for k, rung in enumerate(rungs.target):
                if n == k:
                    height, opacity = (0.5, 1)
                else:
                    height, opacity = (0.3, 0.35)
                rung.set_height(height, about_edge=LEFT)
                rung.set_opacity(opacity)
            self.play(MoveToTarget(rungs))
            self.wait()


class WriteLeGentil(InteractiveScene):
    def construct(self):
        name = Text("Guillaume Le Gentil", font_size=72)
        name.to_corner(UL)
        name.set_backstroke(BLACK, 3)
        self.play(Write(name, stroke_color=WHITE))
        self.wait()