from __future__ import annotations

from manim_imports_ext import *

from _2024.inscribed_rect.helpers import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from manimlib.typing import Vect3, Vect4


class LoopScene(InteractiveScene):
    def get_dot_group(
        self,
        vect_tracker: ValueTracker,
        loop_func: Callable[[float], Vect3],
        colors=None,
        radius: float = 0.05,
        glow_factor: float = 1.5,
    ):
        n = len(vect_tracker.get_value())
        if colors is None:
            colors = [random_bright_color() for _ in range(n)]

        dots = Group(
            get_special_dot(color, radius=radius, glow_factor=glow_factor)
            for _, color in zip(range(n), it.cycle(colors))
        )

        def update_dots(dots):
            for dot, value in zip(dots, vect_tracker.get_value()):
                dot.move_to(loop_func(value))

        dots.add_updater(update_dots)
        return dots

    def get_movable_pair(
        self,
        uv_tracker: ValueTracker,
        loop_func: Callable[[float], Vect3],
        colors=[YELLOW, PINK],
        radius: float = 0.05,
        glow_factor: float = 1.5,
    ):
        return self.get_dot_group(uv_tracker, loop_func, colors, radius, glow_factor)

    def get_movable_quad(
        self,
        abcd_tracker: ValueTracker,
        loop_func: Callable[[float], Vect3],
        colors=[YELLOW, MAROON_B, PINK, RED],
        radius: float = 0.05,
        glow_factor: float = 1.5,
    ):
        return self.get_dot_group(abcd_tracker, loop_func, colors, radius, glow_factor)

    def get_connecting_line(self, dot_pair, stroke_color=TEAL_B, stroke_width=2):
        d1, d2 = dot_pair
        if stroke_color is None:
            stroke_color = average_color(d1.get_color(), d2.get_color())

        line = Line().set_stroke(stroke_color, stroke_width)
        line.f_always.put_start_and_end_on(d1.get_center, d2.get_center)
        return line

    def get_midpoint_dot(self, dot_pair, color=TEAL_B):
        dot = dot_pair[0].copy()
        dot.f_always.move_to(dot_pair.get_center)
        dot.set_color(color)
        return dot

    def get_dot_polygon(self, dots, stroke_color=BLUE, stroke_width=3):
        polygon = Polygon(LEFT, RIGHT)
        polygon.set_stroke(stroke_color, stroke_width)
        polygon.add_updater(lambda m: m.set_points_as_corners(
            [*(d.get_center() for d in dots), dots[0].get_center()]
        ))
        return polygon

    def get_dot_labels(self, dots, label_texs, direction=UL, buff=0):
        result = VGroup()
        for dot, tex in zip(dots, label_texs):
            label = Tex(tex)
            label.match_color(dot[0])
            label.set_backstroke(BLACK, 3)
            label.always.next_to(dot[0], direction, buff=buff)
            result.add(label)
        return result


class StateThePuzzle(LoopScene):
    def construct(self):
        # Show the loop
        loop = get_example_loop(4)
        loop.set_height(7)
        loop.move_to(2 * RIGHT)
        curve_words = Text("Closed\nContinuous\nCurve", alignment="LEFT", font_size=72)
        curve_words.to_edge(LEFT)

        self.play(
            ShowCreation(loop),
            Write(curve_words, time_span=(3, 5)),
            run_time=9
        )

        # Show four points going to a square
        loop.insert_n_curves(50)
        loop_func = loop.quick_point_from_proportion
        quad_tracker = ValueTracker([0, 0, 0, 0])
        dots = self.get_movable_quad(quad_tracker, loop_func, colors=color_gradient([RED, PINK], 4), radius=0.075)
        square_params = find_rectangle(loop_func, target_angle=90 * DEG)

        polygon = self.get_dot_polygon(dots, stroke_color=YELLOW, stroke_width=5)
        inscribed_words = TexText(R"``Inscribed\\Square''", font_size=72)
        inscribed_words.to_edge(LEFT)

        self.add(dots)
        self.play(quad_tracker.animate.set_value(square_params), run_time=3)
        polygon.update()
        self.add(polygon, dots)
        self.play(ShowCreation(polygon, suspend_mobject_updating=True))
        self.play(
            Write(inscribed_words),
            FadeOut(curve_words, LEFT)
        )
        self.wait()

        # Alternate squares
        new_square_params = [
            [0.519, 0.308, 0.277, 0.177],
            [0.444, 0.105, 0.877, 0.650],
            [0.037, 0.739, 0.468, 0.372],
        ]
        dots.suspend_updating()
        for new_params in new_square_params:
            new_dots = dots.copy()
            new_dots.set_opacity(0)
            for dot, p in zip(new_dots, new_params):
                dot.move_to(loop_func(p))

            dots.set_opacity(0)
            self.play(Transform(dots, new_dots), run_time=2)
            dots.set_opacity(1)
            self.wait()

        # Ask question
        title = Text("Open Question", font_size=60)
        title.add(Underline(title))
        title.set_color(BLUE)
        question = Text("Do all closed\ncontinuous curves\nhave an inscribed\nsquare?", alignment="LEFT")
        question.next_to(title, DOWN)
        question.align_to(title[0], LEFT)
        question_group = VGroup(title, question)
        question_group.to_corner(UL, buff=MED_SMALL_BUFF)

        self.play(
            FadeIn(question_group, UP),
            FadeOut(inscribed_words, LEFT)
        )
        self.wait()

        # Ambiently animate to various different loops
        def true_find_square(loop_func, trg_angle=90 * DEG, cost_tol=1e-2, max_tries=8):
            ic = np.arange(0, 1, 0.25)
            min_params = ic
            min_cost = np.inf
            for x in range(max_tries):
                params, cost = find_rectangle(loop_func, target_angle=trg_angle, n_refinements=3, return_cost=True)
                if cost < min_cost:
                    min_params = params
                    min_cost = cost
                ic = np.random.random(4)
            return min_params

        new_loops = [
            get_example_loop(1),
            get_example_loop(2),
            Tex(R"\pi").family_members_with_points()[0],
            Tex(R"\epsilon").family_members_with_points()[0],
            get_example_loop(1),
        ]
        og_loop = loop.copy()
        for new_loop in new_loops:
            new_loop.insert_n_curves(50)
            new_loop.match_style(loop)
            new_loop.match_height(loop)
            new_loop.move_to(loop)

        dots.resume_updating()
        self.add(dots, polygon)
        for new_loop in new_loops:
            self.remove(dots, polygon)
            self.play(
                Transform(loop, new_loop),
                # UpdateFromFunc(
                #     quad_tracker,
                #     lambda m: m.set_value(true_find_square(loop_func))
                # ),
                run_time=1
            )
            self.add(dots, polygon)
            for _ in range(5):
                quad_tracker.set_value(find_rectangle(loop_func, np.random.random(4), target_angle=90 * DEG))
                self.wait(0.5)

        # Change question to rectangle
        square_word = question["square"]
        q_mark = question["?"]
        rect_word = Text("rectangle")
        rect_word.move_to(square_word, LEFT)
        rect_word.set_color(BLUE)
        red_line = Line(LEFT, RIGHT)
        red_line.replace(square_word, 0)
        red_line.set_stroke(RED, 8)

        self.play(
            FadeOut(title, UP),
            ShowCreation(red_line)
        )
        self.play(
            VGroup(square_word, red_line).animate.shift(0.75 * DOWN),
            Write(rect_word),
            q_mark.animate.next_to(rect_word, RIGHT, SMALL_BUFF, aligned_edge=UP),
        )
        self.wait()

        # Transition dots to rectangle
        rect_params = find_rectangle(loop_func, target_angle=45 * DEG)
        self.play(quad_tracker.animate.set_value(rect_params), run_time=4)
        self.wait()

        dots.suspend_updating()
        new_dots = dots.copy()
        for dot, p in zip(new_dots, rect_params):
            dot.move_to(loop_func(p))
        self.play(
            Transform(dots, new_dots),
            run_time=4
        )
        self.wait()
        dots.resume_updating()

        # More ambient transitioning
        for new_loop in [og_loop, *new_loops[1:3]]:
            self.play(
                Transform(loop, new_loop),
                UpdateFromFunc(
                    quad_tracker,
                    lambda m: m.set_value(true_find_square(loop_func, 60 * DEG))
                ),
                run_time=5
            )
            self.wait()


class ReframeToPairsOfPoints(LoopScene):
    def construct(self):
        # Add loop and dots
        loop = SVGMobject("xmas_tree").family_members_with_points()[0]
        loop.set_stroke(WHITE, 3)
        loop.set_fill(opacity=0)
        loop.insert_n_curves(100)
        loop.set_height(7)
        loop.to_edge(RIGHT)
        loop_func = loop.quick_point_from_proportion

        quad_tracker = ValueTracker(np.arange(0, 1, 0.25))
        dots = self.get_movable_quad(quad_tracker, loop_func, radius=0.075)
        labels = self.get_dot_labels(dots, "ABCD")
        labels.set_backstroke(BLACK, 3)
        rect_params = find_rectangle(loop_func, target_angle=55 * DEG)
        quad_tracker.set_value(rect_params + np.random.uniform(0.2, 0.2, 4))

        self.add(quad_tracker, loop, dots, labels)

        # Add words
        question1 = VGroup(
            Text("Find four points"),
            Tex("(A, B, C, D)"),
            Text("That form a rectangle"),
        )
        question2 = VGroup(
            Text("Find two pairs of points", t2s={"pairs": ITALIC}),
            Tex(
                R"\{\{A, C\}, \{B, D\}\}",
                t2c={"A": YELLOW, "B": RED, "C": YELLOW, "D": RED},
            ),
            Text("With the same midpoint\nand distance apart"),
        )
        for question in question1, question2:
            question.arrange(DOWN, buff=0.35)
            question.to_corner(UL)

        for char, label in zip("ABCD", labels):
            question1[1][char].match_style(label)

        self.add(question1)

        # Move to rectangle
        polygon = self.get_dot_polygon(dots)

        self.play(quad_tracker.animate.set_value(rect_params), run_time=8)
        polygon.update()
        self.play(
            ShowCreation(polygon, suspend_mobject_updating=True),
            loop.animate.set_stroke(width=2, opacity=0.5),
            run_time=2
        )
        self.wait()

        # Switch question
        line1 = self.get_connecting_line(dots[0::2]).set_stroke(YELLOW)
        line2 = self.get_connecting_line(dots[1::2]).set_stroke(RED)
        lines = VGroup(line1, line2)
        lines.update().suspend_updating()

        self.play(
            FadeOut(polygon),
            FadeOut(question1[2], DOWN),
            TransformMatchingStrings(question1[0], question2[0], key_map={"four": "two pairs of"}, mismatch_animation=FadeTransformPieces),
            TransformMatchingTex(question1[1], question2[1]),
            dots[2].animate.set_color(YELLOW),
            dots[1].animate.set_color(RED),
            labels[2].animate.set_fill(YELLOW),
            labels[1].animate.set_fill(RED),
        )
        self.wait()
        self.play(LaggedStartMap(ShowCreation, lines, lag_ratio=0.5))
        self.wait()

        # Show the midpoint
        mid_dot1 = self.get_midpoint_dot(dots[0::2])
        mid_dot2 = self.get_midpoint_dot(dots[1::2])
        mid_dot1.update()
        mid_dot2.update()
        arrow = Vector(RIGHT)
        arrow.match_color(mid_dot1[0])
        arrow.next_to(mid_dot1, LEFT, SMALL_BUFF)

        self.play(
            FadeIn(mid_dot1),
            GrowArrow(arrow),
            FadeIn(question2[2]["With the same midpoint"], lag_ratio=0.1)
        )
        self.play(
            FlashAround(question2[2]["midpoint"], color=TEAL),
            question2[2]["midpoint"].animate.set_color(TEAL),
        )
        self.wait()

        # Show the distance apart
        frame = self.frame
        new_lines = lines.copy()
        new_lines.clear_updaters()
        for line in new_lines:
            line.rotate(PI / 2 - line.get_angle())
        new_lines.arrange(RIGHT, buff=0.5)
        new_lines.next_to(loop, LEFT, buff=0.5)

        self.play(
            TransformFromCopy(lines, new_lines),
            Write(question2[2]["and distance apart"]),
            run_time=2
        )
        self.play(
            question2[2]["distance apart"].animate.set_color(YELLOW),
            FlashUnder(question2[2]["distance apart"]),
            run_time=1
        )
        self.wait()

        # Clear the loop and such
        dots.clear_updaters()
        dots[0].f_always.move_to(line1.get_start)
        dots[2].f_always.move_to(line1.get_end)
        dots[1].f_always.move_to(line2.get_start)
        dots[3].f_always.move_to(line2.get_end)

        self.play(
            LaggedStartMap(FadeOut, Group(mid_dot1, arrow, loop, new_lines)),
            question2[2].animate.set_opacity(0.25),
            line1.animate.scale(0.35).rotate(45 * DEG).shift(UP),
            line2.animate.scale(0.90).rotate(-30 * DEG).shift(DOWN),
            run_time=2
        )
        self.wait()

        # Match the midpoints
        for dot in mid_dot1, mid_dot2:
            dot.set_color(WHITE)
            dot.scale(0.5)

        target_midpoint = midpoint(line1.get_center(), line2.get_center())

        self.play(
            *map(FadeIn, [mid_dot1, mid_dot2]),
            question2[2]["With the same midpoint"].animate.set_fill(opacity=1),
        )
        self.play(
            line1.animate.move_to(target_midpoint),
            line2.animate.move_to(target_midpoint),
        )
        self.wait()

        # Match the distance
        self.play(
            line1.animate.set_length(line2.get_length()),
            question2[2]["and distance apart"].animate.set_opacity(1),
            run_time=2
        )
        self.wait()

        # Show various rectangles
        polygon.update()
        self.play(ShowCreation(polygon, suspend_mobject_updating=True))
        for line in [line2, line1, line2]:
            self.play(Rotate(line, 100 * DEG), run_time=4)
        self.wait()


class ShowTheSurface(LoopScene):
    def construct(self):
        # Axes and plane
        frame = self.frame
        axes, plane = self.get_axes_and_plane()
        frame.set_height(6)

        # Curve
        loop = get_example_loop()
        loop_func = get_quick_loop_func(loop)
        self.play(ShowCreation(loop, run_time=2))

        # Pair of points
        uv_tracker = ValueTracker([0, 0.5])
        dots = self.get_movable_pair(uv_tracker, loop_func, radius=0.075)
        connecting_line = self.get_connecting_line(dots)
        midpoint_dot = self.get_midpoint_dot(dots)

        self.add(uv_tracker)
        self.add(dots)
        self.add(connecting_line)

        self.play(
            UpdateFromFunc(uv_tracker, lambda m: m.set_value(np.random.random(2))),
        )
        self.add(dots, connecting_line)
        self.play(uv_tracker.animate.set_value([0.8, 0.5]), run_time=3)

        # Add coordinates
        coords = Tex("(x, y)", font_size=36)
        coords.set_backstroke(BLACK, 3)
        coords.set_fill(WHITE, 1)

        midpoint_dot.update()
        coords.always.next_to(midpoint_dot, UR, buff=-0.1)

        self.play(
            Write(coords, suspend_mobject_updating=True),
            FadeIn(midpoint_dot, scale=0.5)
        )
        self.wait()
        self.play(Write(plane, lag_ratio=0.01, run_time=2, stroke_width=1))
        self.wait()

        # Show the distance
        brace = Brace(Line(LEFT, RIGHT).set_width(connecting_line.get_length()), DOWN, buff=SMALL_BUFF)
        brace.rotate(connecting_line.get_angle(), about_point=ORIGIN)
        brace.shift(connecting_line.get_center())
        brace.set_color(GREY_B)
        d_label = Tex("d", font_size=36)
        d_label.move_to(brace.get_center() + 0.4 * normalize(brace.get_center() - midpoint_dot.get_center()))

        self.play(GrowFromCenter(brace), Write(d_label))

        # 3d coords into the corner
        coords_3d = Tex("(x, y, d)", font_size=36)
        coords_3d.next_to(self.frame.get_corner(UR), DL, MED_SMALL_BUFF)

        self.play(
            TransformFromCopy(coords[:4], coords_3d[:4]),
            TransformFromCopy(d_label, coords_3d[4:6]),
            TransformFromCopy(coords[4:], coords_3d[6:]),
            run_time=2
        )

        # Into 3d
        z_line = self.get_z_line(dots)
        top_dot = self.get_top_dot(z_line)
        top_dot.update()
        top_dot_coords = coords_3d.copy()
        top_dot_coords.unfix_from_frame()
        top_dot_coords.rotate(90 * DEG, RIGHT)
        top_dot_coords.scale(0.75)
        top_dot_coords.next_to(top_dot, OUT + RIGHT, buff=-0.05)

        self.play(
            frame.animate.reorient(5, 79, 0, (0.4, 0.01, 1.41), 5.07),
            FadeIn(axes),
            ReplacementTransform(coords_3d, top_dot_coords),
            TransformFromCopy(midpoint_dot, top_dot, suspend_mobject_updating=True),
            run_time=3
        )
        self.play(
            frame.animate.reorient(-21, 84, 0, (0.4, 0.01, 1.41), 5.07),
            run_time=5
        )
        self.play(
            frame.animate.reorient(0, 18, 0, (0.41, -0.13, 1.34), 4.76),
            run_time=2
        )
        self.play(FlashAround(coords, run_time=2, time_width=1.5))
        self.play(
            TransformFromCopy(connecting_line, z_line, suspend_mobject_updating=True, time_span=(4, 6)),
            frame.animate.reorient(-38, 88, 0, (1.19, -0.11, 1.48), 3.94),
            run_time=6,
        )
        self.play(frame.animate.reorient(-4, 88, 0, (1.19, -0.11, 1.48), 3.94), run_time=5)
        self.wait()

        # Show another pair of points
        uv_tracker2 = ValueTracker([0.2, 0.4])
        dots2 = self.get_movable_pair(uv_tracker2, loop_func, colors=[RED, MAROON_B])
        connecting_line2 = self.get_connecting_line(dots2)
        z_line2 = self.get_z_line(dots2)
        top_dot2 = self.get_top_dot(z_line2)

        dot_group1 = Group(dots, connecting_line, midpoint_dot, z_line, top_dot)
        dot_group2 = Group(dots2, connecting_line2, z_line2, top_dot2)

        self.play(
            frame.animate.reorient(12, 78, 0, (-0.48, 0.08, 1.15), 5.27),
            LaggedStartMap(FadeIn, dot_group2),
            LaggedStartMap(FadeOut, Group(d_label, brace, coords, top_dot_coords)),
            run_time=4
        )
        self.play(uv_tracker2.animate.set_value([0.1, 0.2]), run_time=8)
        self.wait()
        nudge = 0.01
        for _ in range(3):
            self.play(
                uv_tracker.animate.increment_value(np.random.uniform(-nudge, nudge, 2)),
                uv_tracker2.animate.increment_value(np.random.uniform(-nudge, nudge, 2)),
                run_time=2,
                rate_func=lambda t: wiggle(t, 7),
            )
        self.wait()

        # Show pair collision
        # ic = np.random.random(4)
        # print(list(ic.round(2)))
        rect_params = find_rectangle(
            loop_func,
            initial_condition=[0.54, 0.59, 0.73, 0.31],
            n_refinements=5,
            target_angle=64 * DEG
        )
        self.play(
            frame.animate.reorient(-18, 67, 0, (0.11, 0.07, 0.75), 4.63),
            uv_tracker.animate.set_value(rect_params[0::2]),
            uv_tracker2.animate.set_value(rect_params[1::2]),
            run_time=12
        )
        self.wait()

        # Show the rectangle
        rect_points = [loop_func(x) for x in rect_params]
        rect = Polygon(*rect_points)
        rect.set_stroke(YELLOW, 5)
        z_group = Group(z_line, z_line2, top_dot, top_dot2)
        self.play(
            FadeOut(z_group),
            FadeOut(axes),
            frame.animate.reorient(0, 0, 0, ORIGIN, 4.75),
            run_time=4
        )
        self.play(
            ShowCreation(rect),
            loop.animate.set_stroke(width=2)
        )
        self.wait(2)
        self.play(
            FadeOut(rect),
            FadeIn(z_group),
            FadeIn(axes),
            frame.animate.reorient(22, 85, 0, (-0.33, 0.45, 1.52), 6.78),
            run_time=3
        )

        # Set them both in motion
        self.set_uv_tracker_in_motion(uv_tracker, velocity=(-0.05, 0.1))
        self.set_uv_tracker_in_motion(uv_tracker2, velocity=(-0.025, 0.07))
        frame.add_ambient_rotation()

        for dot in [top_dot, top_dot2]:
            tail = TracingTail(dot, time_traced=5, stroke_color=BLUE, stroke_width=(0, 5))
            traced_path = TracedPath(dot.get_center, stroke_color=BLUE, stroke_width=1)
            dot.paths = VGroup(traced_path, tail)
            self.add(dot.paths)
        self.wait(30)

        # Surface
        surface, mesh, surface_func = self.get_surface_info(loop_func, surface_resolution=(301, 301))

        top_dot.paths.clear_updaters()
        top_dot2.paths.clear_updaters()
        self.play(
            FadeIn(mesh),
            FadeIn(surface),
            FadeOut(top_dot.paths),
            FadeOut(top_dot2.paths),
            frame.animate.reorient(-29, 81, 0, (0.14, 0.39, 2.1), 6.47).set_anim_args(run_time=8),
        )
        self.wait(15)

        # Remove dot groups
        self.play(
            FadeOut(dot_group1),
            FadeOut(dot_group2),
        )
        uv_tracker.clear_updaters()
        uv_tracker2.clear_updaters()
        self.wait()

        # Show surface cross sections
        z_tracker = ValueTracker(surface.get_z(OUT))
        top_mesh = mesh.copy()
        top_mesh.set_stroke(width=0.5, opacity=0.1)

        cross_plane = Square3D()
        cross_plane.set_color(WHITE, 0.1)
        cross_plane.replace(plane)
        cross_plane.f_always.set_z(z_tracker.get_value)

        surface.f_always.set_clip_plane(lambda: IN, z_tracker.get_value)
        top_mesh.f_always.set_clip_plane(lambda: OUT, lambda: -z_tracker.get_value())

        self.play(
            surface.animate.set_opacity(1),
            mesh.animate.set_stroke(width=0, opacity=0),
        )
        self.add(top_mesh)
        self.play(FadeIn(cross_plane))
        self.play(
            z_tracker.animate.set_value(0.25),
            surface.animate.set_color(BLUE_E, 1),
            run_time=8
        )
        self.wait(3)

        # Add dots and show the intersection point
        target_z = surface_func(*rect_params[0::2])[2]
        uv_tracker.set_value(rect_params[0::2] + np.random.random(2) * 0.2)
        uv_tracker2.set_value(rect_params[1::2] + np.random.random(2) * -0.2)

        self.play(
            FadeIn(dot_group1),
            FadeIn(dot_group2),
            surface.animate.set_opacity(0.75)
        )
        frame.clear_updaters()
        self.play(
            uv_tracker.animate.set_value(rect_params[0::2]),
            uv_tracker2.animate.set_value(rect_params[1::2]),
            FadeOut(z_line2, time_span=(2.5, 3)),
            FadeOut(top_dot2, time_span=(2.5, 3)),
            frame.animate.reorient(-3, 49, 0, (0.76, 0.62, 0.49), 6.46),
            run_time=3
        )

        # Show the rectangle again
        abcd_tracker = ValueTracker(rect_params)
        uv_tracker.f_always.set_value(lambda: abcd_tracker.get_value()[0::2])
        uv_tracker2.f_always.set_value(lambda: abcd_tracker.get_value()[1::2])

        rect = Rectangle()
        rect.f_always.set_points_as_corners(lambda: list(map(loop_func, abcd_tracker.get_value())))
        rect.always.close_path()

        self.add(abcd_tracker)
        self.play(
            ShowCreation(rect, suspend_mobject_updating=True),
        )
        self.wait()

        # Raise the cross section up to point
        self.play(
            z_tracker.animate.set_value(target_z),
            frame.animate.reorient(-3, 47, 0, (0.27, 0.98, 0.85), 3.80),
            top_mesh.animate.set_stroke(opacity=0.01),
            run_time=7,
        )
        # self.wait(15)  # Comment on the intersection
        self.wait()

        # Animate changing rectangle
        traced_path = TracingTail(top_dot, stroke_color=WHITE, time_traced=5)

        self.add(traced_path)
        self.play(
            frame.animate.reorient(-3, 48, 0, (0.15, 0.76, 0.6), 5.26),
            run_time=4,
        )
        z_tracker.f_always.set_value(top_dot.get_z)
        self.animate_to_rectangle_with_angle(abcd_tracker, loop_func, 80 * DEGREES, n_samples=10, param_range_per_step=0.02)
        z_tracker.clear_updaters()
        self.play(frame.animate.reorient(0, 44, 0, (0.02, 0.63, 0.47), 7.67), run_time=5)
        self.remove(traced_path)

        # Multiple self intersection points
        for _ in range(5):
            new_rect_params = find_rectangle(
                loop_func,
                initial_condition=np.random.random(4),
                target_angle=np.random.uniform(30 * DEG, 90 * DEG),
            )
            new_z = get_dist(*map(loop_func, new_rect_params[::2]))
            self.play(
                abcd_tracker.animate.set_value(new_rect_params),
                z_tracker.animate.set_value(new_z),
            )
            self.wait()

        # Raise the ceiling in full
        self.play(
            frame.animate.reorient(-1, 48, 0, (0.29, 1.22, 1.26), 8.99),
            z_tracker.animate.set_value(surface.get_z(OUT)),
            FadeOut(rect),
            FadeOut(dot_group2),
            run_time=10,
        )
        self.wait()
        self.remove(abcd_tracker)
        uv_tracker.clear_updaters()
        self.play(
            z_tracker.animate.set_value(0.25),
            top_mesh.animate.set_stroke(opacity=0.1),
            frame.animate.reorient(-7, 78, 0, (0.31, 1.16, 0.94), 7.21),
            run_time=6
        )

        # Point out what happens near the edge
        glow_tracker = ValueTracker(dots[0][0].get_glow_factor())
        radius_tracker = ValueTracker(dots[0][0].get_radius())
        for dot in [*dots, top_dot]:
            for part in dot:
                part.f_always.set_glow_factor(glow_tracker.get_value)
                part.f_always.set_radius(radius_tracker.get_value)
        v = uv_tracker.get_value()[1]
        self.play(
            uv_tracker.animate.set_value([v + 0.01, v]),
            glow_tracker.animate.set_value(0.25),
            radius_tracker.animate.set_value(0.025),
            frame.animate.reorient(-7, 78, 0, (0.1, 1.02, 0.18), 3.80),
            FadeOut(midpoint_dot),
            run_time=5
        )
        self.play(uv_tracker.animate.set_value([0.26, 0.25]), run_time=6)
        self.wait()
        self.play(
            uv_tracker.animate.set_value([0.25, 0.25]),
            frame.animate.reorient(-7, 78, 0, (-0.15, 1.0, -0.07), 2.80),
            z_tracker.animate.set_value(0.01),
            run_time=7
        )
        self.play(frame.animate.reorient(2, 70, 0, (0.05, 1.06, 0.05), 4.29), run_time=10)
        self.wait()

    def get_axes_and_plane(
        self,
        x_range=(-3, 3),
        y_range=(-3, 3),
        z_range=(0, 5),
        depth=4,
    ):
        axes = ThreeDAxes(x_range, y_range, z_range)
        axes.set_depth(depth, stretch=True, about_edge=IN)
        axes.set_stroke(GREY_B, 1)

        plane = NumberPlane(x_range, y_range)
        plane.background_lines.set_stroke(BLUE, 1, 0.75)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)
        plane.axes.match_style(axes)
        plane.set_z_index(-1)
        return axes, plane

    def get_z_line(self, dot_pair, stroke_color=TEAL_B, stroke_width=2):
        z_line = Line(IN, OUT).set_stroke(stroke_color, stroke_width)

        def update_z_line(z_line):
            point1 = dot_pair[0].get_center()
            point2 = dot_pair[1].get_center()
            midpoint = mid(point1, point2)
            top = midpoint + get_norm(point1 - point2) * OUT
            z_line.put_start_and_end_on(midpoint, top)

        z_line.add_updater(update_z_line)
        z_line.update()
        return z_line

    def get_top_dot(self, z_line, color=BLUE, radius=0.05, glow_factor=1.0):
        top_dot = Group(
            TrueDot(radius=radius).make_3d(),
            GlowDot(radius=radius * 2, glow_factor=glow_factor)
        )
        top_dot.set_color(color)
        top_dot.f_always.move_to(z_line.get_end)
        return top_dot

    def set_uv_tracker_in_motion(self, uv_tracker, velocity=(-0.05, 0.1)):
        velocity = np.array(velocity)

        def update_uv_tracker(uv_tracker, dt):
            new_value = uv_tracker.get_value() + dt * velocity
            uv_tracker.set_value(new_value % 1)

        uv_tracker.add_updater(update_uv_tracker)
        return uv_tracker

    def animate_to_rectangle_with_angle(
        self, abcd_tracker, loop_func, target_angle,
        n_samples=5,
        run_time=5,
        param_range_per_step=0.1,
        n_samples_per_range=10,
        n_refinements=3,
    ):
        # Find the sample points
        points = list(map(loop_func, abcd_tracker.get_value()))
        curr_angle = abs(angle_between_vectors(points[2] - points[0], points[3] - points[1]))
        if curr_angle > PI / 2:
            curr_angle = PI - curr_angle

        rectangle_range = [abcd_tracker.get_value()]
        for angle in np.linspace(curr_angle, target_angle, n_samples + 1)[1:]:
            rectangle_range.append(find_rectangle(
                loop_func=loop_func,
                initial_condition=rectangle_range[-1],
                target_angle=angle,
                initial_param_range=param_range_per_step,
                n_samples_per_range=n_samples_per_range,
                n_refinements=n_refinements,
            ))
        rectangle_range = np.array(rectangle_range)

        self.play(
            UpdateFromAlphaFunc(abcd_tracker, lambda m, a: m.set_value(smooth_index(rectangle_range, a))),
            run_time=run_time
        )

    def get_surface_info(
        self,
        loop_func: Callable[[float], Vect3],
        surface_color=BLUE,
        surface_opacity=0.25,
        surface_resolution=(101, 101),
        mesh_color=WHITE,
        mesh_stroke_width=0.5,
        mesh_stroke_opacity=0.1,
        mesh_resolution=(101, 101),
    ):
        surface_func = get_surface_func(loop_func)

        surface = ParametricSurface(
            get_half_parametric_func(surface_func),
            resolution=surface_resolution,
        )
        surface.set_color(surface_color, surface_opacity)
        surface.always_sort_to_camera(self.camera)

        full_surface = ParametricSurface(surface_func)
        mesh = SurfaceMesh(full_surface, resolution=mesh_resolution, normal_nudge=0)
        mesh.set_stroke(WHITE, mesh_stroke_width, mesh_stroke_opacity)
        mesh.deactivate_depth_test()

        return surface, mesh, surface_func


class ChangeTheSurface(ShowTheSurface):
    def construct(self):
        # Axes and plane
        frame = self.frame
        axes, plane = self.get_axes_and_plane()
        frame.reorient(-45, 83, 0, (0.08, 0.63, 2.3), 8.55)
        frame.add_ambient_rotation()
        self.add(axes, plane)

        # Show loops
        example_loops = VGroup(
            get_example_loop(1),
            get_example_loop(2),
            # get_example_loop(3),
            SVGMobject("gingerbread_outline")[0]
        )
        for loop in example_loops:
            loop.set_height(5)
            loop.set_stroke(WHITE, 4).set_fill(opacity=0)
        loop = example_loops[0].copy()

        surface = self.get_surface(loop)

        def update_surface(surface):
            surface.become(self.get_surface(loop))
            surface.always_sort_to_camera(self.camera)

        self.add(loop)
        self.add(surface)

        for next_loop in example_loops[1:]:
            self.play(
                Transform(loop, next_loop),
                UpdateFromFunc(surface, update_surface),
                run_time=8
            )
            self.wait(2)

        # Circle and ellipse
        circle = Circle(radius=2)
        circle.set_stroke(WHITE, 4)

        self.play(
            Transform(loop, circle),
            UpdateFromFunc(surface, update_surface),
            run_time=5
        )
        loop.become(circle)

        surface.always_sort_to_camera(self.camera)

        # Show all the recangles
        x_tracker = ValueTracker(0.125)
        get_x = x_tracker.get_value
        loop_func = loop.pfp

        uv_tracker1 = ValueTracker([0, 0])
        uv_tracker2 = ValueTracker([0, 0])
        uv_tracker1.add_updater(lambda m: m.set_value([get_x(), get_x() + 0.5]))
        uv_tracker2.add_updater(lambda m: m.set_value([0.5 - get_x(), 1.0 - get_x()]))

        dots1 = self.get_movable_pair(uv_tracker1, loop_func)
        dots2 = self.get_movable_pair(uv_tracker2, loop_func, colors=[RED, MAROON_B])
        line1 = self.get_connecting_line(dots1)
        line2 = self.get_connecting_line(dots2)
        z_line = self.get_z_line(dots1)
        top_dot = self.get_top_dot(z_line)
        rect = Rectangle()
        rect.set_stroke(YELLOW, 3)
        rect.f_always.set_points_as_corners(lambda: list(map(loop_func, [
            get_x(), 0.5 - get_x(), get_x() + 0.5, 1.0 - get_x(), get_x()
        ])))

        rect_group = Group(dots1, dots2, line1, line2, z_line, top_dot, rect)

        self.add(uv_tracker1, uv_tracker2)
        self.play(
            FadeIn(rect_group),
            frame.animate.reorient(-14, 31, 0, (-0.08, -0.56, 1.19), 8.04),
            run_time=3
        )
        for value in [0.24, 0.01, 0.125]:
            self.play(x_tracker.animate.set_value(value), run_time=6)
        self.wait()

        # Squish into an ellipse
        ellipse = circle.copy().stretch(0.5, 1)
        self.play(
            frame.animate.reorient(-11, 67, 0, (-0.08, -0.56, 1.19), 8.04),
            run_time=2
        )
        self.play(
            Transform(loop, ellipse),
            UpdateFromFunc(surface, update_surface),
            run_time=5
        )
        self.play(frame.animate.reorient(156, 77, 0, (-0.08, -0.56, 1.19), 8.04), run_time=10)
        self.wait(4)

        # Move the coordinates again
        self.play(frame.animate.reorient(174, 34, 0, (-0.08, -0.56, 1.19), 8.04), run_time=3)
        for value in [0.24, 0.01, 0.125]:
            self.play(x_tracker.animate.set_value(value), run_time=6)

    def get_surface(self, loop, surface_resolution=(301, 301), color=BLUE, opacity=0.5):
        # return Square3D().set_z(10)
        surface_func = get_surface_func(loop.quick_point_from_proportion)
        surface = ParametricSurface(
            get_half_parametric_func(surface_func),
            resolution=surface_resolution,
        )
        surface.sort_faces_back_to_front(self.camera.get_location())
        surface.set_color(color, opacity)
        return surface


class ParameterizeTheLoop(InteractiveScene):
    def construct(self):
        # Set up the loop
        loop = get_example_loop(width=5)
        loop.insert_n_curves(20)
        loop.to_edge(LEFT, buff=LARGE_BUFF)

        x_tracker = ValueTracker()
        get_x = x_tracker.get_value
        loop_x_group = self.get_loop_coord_group(loop, get_x)

        self.add(loop)
        self.add(loop_x_group)

        # Set up the unit interval
        interval = UnitInterval(width=6)
        interval.to_edge(RIGHT)
        interval.add_numbers()

        x_tip = ArrowTip(angle=-90 * DEG)
        x_tip.set_height(0.2)
        x_tip.set_color(YELLOW)
        x_tip.f_always.move_to(lambda: interval.n2p(get_x()), lambda: DOWN)
        int_x_label = DecimalNumber(font_size=24)
        int_x_label.set_color(YELLOW)
        int_x_label.always.next_to(x_tip, UP, buff=0.15)
        int_x_label.f_always.set_value(get_x)

        int_x_group = VGroup(x_tip, int_x_label)
        self.add(interval)
        self.add(int_x_group)

        # Animate changing x
        self.play(x_tracker.animate.set_value(1), run_time=12, rate_func=there_and_back)
        x_tracker.set_value(0)

        # Snip the loop
        snipped_loop = loop.copy()
        sl_points = np.array(loop.get_points())  # Snipped loop points
        sl_points[0] += 0.25 * LEFT
        sl_points[-1] += 0.25 * UP
        snipped_loop.set_points(sl_points)

        scissors = SVGMobject("scissors")
        scissors.set_color(GREY_B)
        scissors.rotate(45 * DEG)
        scissors.set_height(0.75)
        scissors_shift = np.array([0.7, -0.5, 0])
        scissors.move_to(loop.get_start() + scissors_shift)

        self.play(
            FadeOut(loop_x_group),
            FadeOut(int_x_group),
            FadeIn(scissors)
        )
        moving_loop = loop.copy()
        loop.set_stroke(opacity=0.25)
        self.play(
            Transform(moving_loop, snipped_loop, time_span=(0.5, 1.5)),
            scissors.animate.shift(-2 * scissors_shift),
            run_time=2
        )
        self.play(FadeOut(scissors))

        # Map it onto the unit interval
        line = Line(interval.n2p(0), interval.n2p(1))
        line.match_style(moving_loop)
        self.play(Transform(moving_loop, line, run_time=3, path_arc=-30 * DEG))
        self.wait()

        # Show coordinate moving around
        self.play(
            FadeIn(loop_x_group),
            FadeIn(int_x_group),
            loop.animate.set_stroke(opacity=1),
            FadeOut(moving_loop),
        )
        self.play(x_tracker.animate.set_value(1), run_time=5)
        self.wait()
        for value in [0, 1, 0]:
            x_tracker.set_value(value)
            self.wait()

        # Glue 0 to 1
        circular_interval = Circle(radius=TAU / interval.get_length())
        circular_interval.rotate(PI / 2)
        circular_interval.match_style(interval)
        circle_ticks = VGroup(
            Line(1.1 * point, 0.9 * point)
            for a in np.linspace(0, 1, 11)
            for point in [circular_interval.pfp(a)]
        )
        circle_numbers = interval.numbers.copy()
        for tick, number in zip(circle_ticks, circle_numbers):
            number.move_to(1.3 * tick.get_center())
        circle_numbers[-1].move_to(0.7 * circle_ticks[-1].get_center())

        circular_interval.add(circle_ticks, circle_numbers)
        circular_interval.move_to(interval)

        interval.save_state()
        self.play(
            FadeOut(int_x_group),
            Transform(interval, circular_interval, run_time=3),
        )
        self.play(FlashAround(VectorizedPoint(interval.get_start()), run_time=2))
        self.wait()
        self.play(Restore(interval, run_time=3))

        # Add a second point
        y_tracker = ValueTracker(0)
        get_y = y_tracker.get_value
        loop_y_group = self.get_loop_coord_group(loop, get_y, color=PINK, label_direction=UR)
        loop_y_group.update()

        self.add(loop_y_group)
        self.play(
            # FadeIn(loop_y_group, time_span=(0, 1)),
            x_tracker.animate.set_value(0.15),
            y_tracker.animate.set_value(0.25),
            run_time=2,
        )

        # Add a second axis
        x_axis = interval
        y_axis = UnitInterval()
        y_axis.set_width(interval.get_length())
        y_axis.rotate(90 * DEG)
        y_axis.add_numbers(direction=LEFT)

        y_axis.move_to(x_axis.n2p(0))
        y_axis.shift(0.25 * LEFT)

        int_y_group = int_x_group.copy()
        int_y_group.clear_updaters()
        int_y_group.set_color(PINK)
        y_tip, y_dec = int_y_group
        y_tip.rotate(-90 * DEG)
        y_tip.f_always.move_to(lambda: y_axis.n2p(get_y()), lambda: LEFT)
        y_dec.f_always.set_value(get_y)
        y_dec.always.next_to(y_tip, RIGHT, SMALL_BUFF)

        int_y_group.update()
        int_y_group.suspend_updating()
        self.play(
            FadeIn(y_axis),
            VFadeIn(int_x_group),
            x_axis.animate.shift(y_axis.n2p(0) - x_axis.n2p(0)),
            FadeTransformPieces(loop_y_group.copy(), int_y_group),
            run_time=2
        )
        int_y_group.resume_updating()
        self.wait()
        self.play(y_tracker.animate.set_value(0.84), run_time=3)
        self.play(x_tracker.animate.set_value(0.75), run_time=3)
        self.play(y_tracker.animate.set_value(0.65), run_time=3)

        # Show in the unit square
        axes = Axes((0, 1), (0, 1), width=x_axis.get_length(), height=y_axis.get_length())
        axes.shift(x_axis.n2p(0) - axes.c2p(0, 0))
        int_y_group.clear_updaters()
        int_x_group.clear_updaters()
        x_tip, x_dec = int_x_group

        square = Square()
        square.set_z_index(-1)
        square.set_stroke(GREY, 1)
        square.set_fill(GREY_E, 0.5)
        square.set_width(x_axis.get_length())
        square.move_to(x_axis.n2p(0), DL)

        square_point = get_special_dot(color=BLUE)
        square_point.f_always.move_to(lambda: axes.c2p(get_x(), get_y()))

        v_line = Line(DOWN, UP).set_stroke(WHITE, 1)
        h_line = Line(LEFT, RIGHT).set_stroke(WHITE, 1)
        v_line.add_updater(lambda m: m.put_start_and_end_on(
            axes.c2p(get_x(), 0), axes.c2p(get_x(), get_y())
        ))
        h_line.add_updater(lambda m: m.put_start_and_end_on(
            axes.c2p(0, get_y()), axes.c2p(get_x(), get_y())
        ))
        coord_lines = VGroup(v_line, h_line)

        coord_label = Tex(R"(0.00, 0.00)", font_size=24)
        coord_standins = coord_label.make_number_changeable("0.00", replace_all=True)
        coord_label.always.next_to(square_point, UR, buff=-0.1)
        coord_standins.set_opacity(0)

        coord_lines.update()
        coord_lines.suspend_updating()
        self.play(
            FadeIn(square),
            ShowCreation(v_line),
            ShowCreation(h_line),
            y_tip.animate.flip(UP, about_edge=LEFT),
            x_tip.animate.flip(RIGHT, about_edge=DOWN),
            x_dec.animate.move_to(coord_label[1]),
            y_dec.animate.move_to(coord_label[3]),
            FadeIn(coord_label),
            FadeIn(square_point, scale=0.5),
        )
        coord_lines.resume_updating()

        x_tip.f_always.match_x(lambda: x_axis.n2p(get_x()))
        y_tip.f_always.match_y(lambda: y_axis.n2p(get_y()))
        x_dec.f_always.set_value(get_x)
        y_dec.f_always.set_value(get_y)

        coord_label.replace_submobject(1, x_dec)
        coord_label.replace_submobject(3, y_dec)

        # Move coordinates
        xy_tracker = ValueTracker(np.array([get_x(), get_y()]))
        x_tracker.f_always.set_value(lambda: xy_tracker.get_value()[0])
        y_tracker.f_always.set_value(lambda: xy_tracker.get_value()[1])
        self.add(x_tracker, y_tracker)
        self.play(xy_tracker.animate.set_value([0.50, get_y()]), run_time=4)
        self.play(xy_tracker.animate.set_value([get_x(), 0.20]), run_time=4)
        self.play(xy_tracker.animate.set_value([0.05, get_y()]), run_time=4)
        np.random.seed(0)
        for _ in range(3):
            self.play(xy_tracker.animate.set_value(np.random.random(2)), run_time=4)

        # Highlight the x=0 and x=1 lines
        x_line_color = BLUE
        frame = self.frame
        left_edge = Line(DOWN, UP)
        left_edge.set_stroke(x_line_color, 5)
        left_edge.match_height(square)
        left_edge.move_to(square, LEFT)
        right_edge = left_edge.copy()
        right_edge.move_to(square, RIGHT)

        left_tips = ArrowTip(angle=90 * DEG).get_grid(3, 1, buff=1.0)
        left_tips.move_to(left_edge)
        left_tips.set_color(x_line_color)
        right_tips = left_tips.copy()
        right_tips.move_to(right_edge)

        self.play(xy_tracker.animate.set_value([0, 0]), run_time=2)
        self.play(
            frame.animate.set_height(9),
            ShowCreation(left_edge),
            xy_tracker.animate.set_value([0, 1]),
            run_time=12
        )
        xy_tracker.set_value([1, 0])
        self.play(
            ShowCreation(right_edge),
            xy_tracker.animate.set_value([1, 1]),
            run_time=8
        )
        self.wait()
        self.play(
            Write(left_tips),
            Write(right_tips),
        )

        v_arrows = VGroup(VGroup(left_edge, left_tips), VGroup(right_edge, right_tips))

        # Highlight y=0 and y=1 lines
        y_line_color = GREEN_SCREEN
        bottom_edge = Line(LEFT, RIGHT)
        bottom_edge.set_stroke(y_line_color, 5)
        bottom_edge.match_width(square)
        bottom_edge.move_to(square, DOWN)
        top_edge = bottom_edge.copy()
        top_edge.move_to(square, UP)

        bottom_tips = ArrowTip().get_grid(1, 3, buff=1.0)
        bottom_tips.move_to(bottom_edge)
        bottom_tips.set_color(y_line_color)
        top_tips = bottom_tips.copy()
        top_tips.move_to(top_edge)

        xy_tracker.set_value([0, 0])
        self.play(
            xy_tracker.animate.set_value([1, 0]),
            ShowCreation(bottom_edge),
            Write(bottom_tips, time_span=(2, 4)),
            run_time=4
        )
        xy_tracker.set_value([0, 1])
        self.play(
            xy_tracker.animate.set_value([1, 1]),
            ShowCreation(top_edge),
            Write(top_tips, time_span=(2, 4)),
            run_time=4
        )

        h_arrows = VGroup(VGroup(bottom_edge, bottom_tips), VGroup(top_edge, top_tips))

        # Fold into a torus
        def half_torus_func(u, v):
            return torus_func(u, 0.5 * v)

        surfaces = Group(
            TexturedSurface(ParametricSurface(func), "TorusTexture")
            for func in [square_func, tube_func, half_torus_func, torus_func]
        )
        for surface in surfaces:
            surface.set_shading(0.25, 0.25, 0)
            surface.set_opacity(0.75)

        target_z = 5
        square3d, tube, half_torus, torus = surfaces
        square3d.replace(square)

        surface = square3d.copy()
        surface.replace(square)
        surface.set_z(target_z)

        tube.set_width(surface.get_width() / PI)
        tube.match_height(surface, stretch=True)
        tube.move_to(surface, IN)

        torus.match_depth(tube)
        torus.move_to(tube)
        half_torus.match_width(torus)
        half_torus.move_to(torus, UP)

        cover_rect = SurroundingRectangle(Group(loop, loop_y_group))
        cover_rect.set_fill(BLACK, 1).set_stroke(width=0)

        self.add(surface)
        self.play(
            FadeIn(surface, shift=target_z * OUT),
            FadeIn(cover_rect),
            frame.animate.reorient(-13, 61, 0, (1.52, 1.67, 1.97), 15.41),
            run_time=3,
        )
        self.play(Transform(surface, tube), run_time=3)
        self.wait()
        self.play(Transform(surface, half_torus, path_arc=PI / 2), run_time=3)
        self.play(Transform(surface, torus, path_arc=PI / 2), run_time=3)
        self.wait()
        self.remove(surface)
        self.add(torus)

        # Put torus in position above the loop
        torus_point = TrueDot(color=BLUE)
        torus_point.f_always.move_to(lambda: torus.uv_to_point(get_x(), get_y()))
        torus_point.apply_depth_test()

        self.play(
            FadeOut(cover_rect),
            loop.animate.set_height(6).next_to(y_axis, LEFT, buff=1.5),
            frame.animate.reorient(0, 0, 0, (0.44, 1.84, 0.0), 13.21),
            torus.animate.set_height(7).rotate(50 * DEG, LEFT).move_to(6 * UP),
            torus.animate.set_height(7).rotate(50 * DEG, LEFT).move_to(6 * UP).match_x(square),
            v_arrows.animate.set_opacity(0.25),
            h_arrows.animate.set_opacity(0.25),
            coord_label.animate.scale(1.5),
            run_time=3
        )

        torus_mesh = SurfaceMesh(torus, resolution=(21, 21))
        torus_mesh.set_stroke(WHITE, 0.5, 0.5)
        self.add(torus_point, torus_mesh, torus)
        self.play(
            Write(torus_mesh, lag_ratio=0.01, stroke_width=0.5, run_time=1),
            FadeIn(torus_point),
        )

        target_xys = [
            [0.13, 0.25],
            [0.13, 0.65],
            [0.13, 0.35],
            [0.97, 0.35],
            [0.10, 0.35],
        ]
        for xy in target_xys:
            self.play(xy_tracker.animate.set_value(xy), run_time=4)

        # Wiggle the points
        for _ in range(3):
            self.play(
                xy_tracker.animate.increment_value(0.02 * np.random.uniform(-1, 1, 2)),
                run_time=2,
                rate_func=lambda t: wiggle(t, 7)
            )
            self.wait()

        # Fade back to square
        self.play(
            frame.animate.reorient(0, 0, 0, (-0.57, 0.46, 0.0), 10),
            FadeOut(torus, UP),
            FadeOut(torus_mesh, UP),
            FadeOut(torus_point, UP),
            run_time=2
        )
        self.wait()

        # Show (x, y) -> (y, x) pairs
        coord_ghosts = Group()
        double_arrows = VGroup()

        def get_coord_ghost():
            result = Group(square_point, coord_label).copy()
            result.clear_updaters()
            result.fade(0.25)
            coord_ghosts.add(result)
            return result

        def get_double_arrow():
            point1 = axes.c2p(get_x(), get_y())
            point2 = axes.c2p(get_y(), get_x())
            vect = normalize(point2 - point1)
            result = VGroup(
                Arrow(point1, point2).shift(0.1 * vect),
                Arrow(point2, point1).shift(-0.1 * vect),
            )
            result.set_stroke(GREY_C)
            double_arrows.add(result)
            return result

        def show_reflection():
            x_dot = loop_x_group[0]
            y_dot = loop_y_group[0]
            loop_x_group.suspend_updating()
            loop_y_group.suspend_updating()

            self.add(get_coord_ghost())
            self.play(
                GrowFromPoint(get_double_arrow(), square_point.get_center()),
                xy_tracker.animate.set_value([get_y(), get_x()]),
                Swap(x_dot, y_dot),
                run_time=1
            )
            self.play(Swap(x_dot, y_dot))
            self.wait()
            self.add(get_coord_ghost())

            loop_x_group.resume_updating().update()
            loop_y_group.resume_updating().update()
            self.add(loop_x_group, loop_y_group)
            loop_x_group.update()
            loop_y_group.update()

        for xy in [[0.1, 0.9], [0.8, 0.95]]:
            show_reflection()
            self.play(xy_tracker.animate.set_value(xy))
        show_reflection()

        # Show fold line
        fold_line = Line(axes.c2p(0, 0), axes.c2p(1, 1))
        fold_line.set_stroke(Color("red"), 2)

        self.play(
            ShowCreation(fold_line),
            *map(FadeOut, [v_line, h_line, coord_label, square_point, x_tip, y_tip]),
        )
        self.wait()
        self.play(
            FadeOut(coord_ghosts),
            FadeOut(double_arrows),
            v_arrows.animate.set_opacity(1),
            h_arrows.animate.set_opacity(1),
        )

        # Fold the square
        ul_triangle = Polygon(DL, UL, UR)
        dr_triangle = Polygon(DL, DR, UR)
        for triangle in [ul_triangle, dr_triangle]:
            triangle.replace(square)
            triangle.match_style(square)
            triangle.set_z_index(-1)
            triangle.set_shading(0.25, 0, 0)
            self.add(triangle)
        self.remove(square)

        self.play(
            Rotate(ul_triangle, PI, about_point=square.get_center(), axis=UR),
            Rotate(v_arrows[0], PI, about_point=square.get_center(), axis=UR),
            Rotate(h_arrows[1], PI, about_point=square.get_center(), axis=UR),
            run_time=2
        )
        self.remove(v_arrows)
        self.play(h_arrows.animate.set_color(PURPLE))

        folded_square = Group(dr_triangle, h_arrows, fold_line).copy()

        # Note the diagonal line again
        self.play(
            FadeIn(square_point),
            FadeIn(coord_label),
        )
        self.play(xy_tracker.animate.set_value([0.9, 0.9]), run_time=2)
        self.play(xy_tracker.animate.set_value([0.1, 0.1]), run_time=8)
        self.wait()

        # Comment on the tricky points (Probably edit out the actual transitions)
        self.play(xy_tracker.animate.set_value([0.1, 0]), run_time=2)
        self.play(FlashAround(coord_label))
        self.wait()
        self.play(xy_tracker.animate.set_value([1, 0]), run_time=2)
        self.play(xy_tracker.animate.set_value([1, 0.1]))
        self.play(FlashAround(coord_label))
        self.wait()
        self.play(xy_tracker.animate.set_value([0.9, 0]))
        self.play(FlashAround(coord_label))
        self.wait()
        self.play(xy_tracker.animate.set_value([1, 0]))
        self.play(xy_tracker.animate.set_value([1, 0.9]), run_time=2)
        self.play(FlashAround(coord_label))

        # Fade out axes and such
        self.play(
            LaggedStartMap(FadeOut, Group(
                loop_x_group[1], loop_y_group[1], coord_label, square_point,
                x_axis, y_axis,
            ))
        )
        self.wait()

        # Show the new cut
        cut_line = Line(square.get_center(), square.get_corner(DR))
        cut_line.set_color(YELLOW)
        cut_tips = bottom_tips.copy().set_color(YELLOW)
        cut_tips.rotate(-45 * DEGREES)
        cut_tips.move_to(cut_line)
        cut_arrow1 = VGroup(cut_line, cut_tips)
        cut_arrow2 = cut_arrow1.copy()

        d_tri = Polygon(LEFT, UP, RIGHT)
        d_tri.match_width(square).move_to(square, DOWN)
        r_tri = Polygon(DOWN, LEFT, UP)
        r_tri.match_height(square).move_to(square, RIGHT)
        for tri in d_tri, r_tri:
            tri.set_fill(GREY_D, 0.75)
            tri.set_stroke(width=0)

        fold_line1, fold_line2 = fold_line.replicate(2)
        fold_line1.put_start_and_end_on(square.get_corner(DL), square.get_center())
        fold_line2.put_start_and_end_on(square.get_center(), square.get_corner(UR))

        piece1 = VGroup(d_tri, h_arrows[0], fold_line1).copy()
        piece2 = VGroup(r_tri, h_arrows[1], fold_line2).copy()
        pieces = VGroup(piece1, piece2)
        to_remove = VGroup(h_arrows, fold_line)
        old_tris = VGroup(ul_triangle, dr_triangle)

        self.add(pieces, old_tris, fold_line)
        self.play(
            ShowCreation(cut_line),
            Write(cut_tips, time_span=(0.5, 1.5)),
            FadeIn(pieces),
            FadeOut(old_tris),
            run_time=1.5,
        )
        self.remove(to_remove)
        piece1.add(cut_arrow1)
        piece2.add(cut_arrow2)
        self.add(pieces)
        self.play(VGroup(piece1, piece2).animate.space_out_submobjects(1.5).move_to(square))
        self.wait()

        # Rearrange pieces
        pieces.target = pieces.generate_target()
        pieces.target[0].rotate(90 * DEGREES)
        pieces.target[1].flip()
        pieces.target.arrange(RIGHT, buff=0.5)
        pieces.target.move_to(square)

        self.play(MoveToTarget(pieces), run_time=2)
        self.wait()
        self.play(
            piece1.animate.shift(square.get_center() - piece1[0].get_right()),
            piece2.animate.shift(square.get_center() - piece2[0].get_left()),
        )
        self.play(
            piece1[1].animate.set_opacity(0),
            piece2[1].animate.set_opacity(0),
        )
        self.wait()

        # Fold into a Mobius strip
        custom_squish = bezier([0, 0.05, 0.95, 1])

        def smoothed_mobius_func(u, v):
            return mobius_strip_func(u, custom_squish(v))

        def get_partial_strip(upper_theta=1.0):
            result = ParametricSurface(lambda u, v: mobius_strip_func(
                u, custom_squish(v) * upper_theta / TAU
            ))
            result.scale(2, about_point=ORIGIN)
            result.shift((0, 4, 4))
            return result

        surfaces = Group(
            TexturedSurface(plain_surface, "MobiusStripTexture")
            for plain_surface in [
                ParametricSurface(square_func),
                get_partial_strip(2.0),
            ]
        )
        for surface in surfaces:
            surface.set_shading(0.25, 0.25, 0)
            surface.set_opacity(0.75)

        target_z = 4
        square3d, partial_strip = surfaces
        square3d.rotate(45 * DEG)
        square3d.replace(pieces)
        square3d.set_z(target_z)
        surface = square3d.copy()

        cover_rect.surround(loop, buff=0.2)

        self.play(
            FadeIn(cover_rect),
            FadeIn(surface, shift=target_z * OUT),
            frame.animate.reorient(2, 51, 0, (-0.35, 3.04, 0.42), 15.36),
            run_time=3
        )
        self.play(
            Transform(surface, partial_strip, run_time=2),
        )
        self.play(
            UpdateFromAlphaFunc(surface, lambda m, a: m.set_points(
                get_partial_strip(interpolate(2, TAU, smooth(a))).get_points()
            )),
            run_time=5
        )
        self.play(
            frame.animate.reorient(0, 42, 0, (-0.11, 2.48, 0.87), 13.94),
            Rotate(surface, PI, axis=RIGHT),
            run_time=8
        )
        mobius_strip = surface

        # Reintroduce coordiante plane
        self.play(
            frame.animate.reorient(0, 0, 0, (-1.02, 3.21, 0.0), 14.55),
            mobius_strip.animate.rotate(40 * DEG, LEFT).move_to(7.5 * UP),
            loop.animate.scale(1.25, about_edge=RIGHT),
            FadeOut(pieces),
            FadeIn(x_axis),
            FadeIn(y_axis),
            FadeIn(folded_square),
            FadeIn(square_point),
            FadeIn(coord_label),
            FadeIn(loop_x_group[1]),
            FadeIn(loop_y_group[1]),
        )

        # Show a point on the mobius strip
        strip_dot = TrueDot(color=BLUE)

        def update_strip_dot(dot):
            u, v = torus_uv_to_mobius_uv(get_x(), get_y())
            strip_dot.move_to(mobius_strip.uv_to_point(u, v))

        strip_dot.add_updater(update_strip_dot)

        self.play(FadeIn(strip_dot))

        xy_values = [
            [0.5, 0.25],
            [0.9, 0.1],
            [0.8, 0.7],
            [0.53, 0.12],
            [0.0, 0.0],
        ]
        for xy in xy_values:
            self.play(xy_tracker.animate.set_value(xy), run_time=3)

        self.play(xy_tracker.animate.set_value([0.99, 0.99]), run_time=8)
        self.play(xy_tracker.animate.set_value([0, 0]), run_time=8)

        # Map from torus to strip
        torus_group = Group(torus_mesh, torus)
        torus_group.next_to(square, UP, buff=2)

        torus_fold_line = ParametricCurve(lambda t: torus.uv_to_point(t, t), t_range=(0, 1, 0.01))
        torus_fold_line.set_stroke(RED, 1, 1)

        self.play(
            mobius_strip.animate.shift(7 * LEFT),
            GrowFromCenter(torus_group),
            FadeIn(torus_point),
        )
        self.play(
            ShowCreation(torus_fold_line),
            xy_tracker.animate.set_value([0.99, 0.99]),
            run_time=5
        )
        self.wait()

        # Animate torus squish
        squished_torus = TexturedSurface(
            ParametricSurface(lambda u, v: mobius_strip_func(*torus_uv_to_mobius_uv(u, v))),
            "TorusTexture",
        )
        squished_torus.replace(torus)
        squished_torus.rotate(40 * DEG, LEFT)
        squished_torus.set_opacity(0)

        squished_torus_mesh = SurfaceMesh(squished_torus, resolution=(21, 21))
        squished_torus_mesh.match_style(torus_mesh)
        squished_torus_mesh.set_stroke(opacity=0.35)
        squished_torus_mesh.make_jagged()

        new_fold_line = ParametricCurve(lambda t: squished_torus.uv_to_point(t, t), t_range=(0, 0.99, 0.01))

        self.play(
            Transform(torus, squished_torus),
            Transform(torus_mesh, squished_torus_mesh),
            torus_fold_line.animate.set_points(new_fold_line.get_points()),
            run_time=5,
        )
        self.wait()

    def get_loop_coord_group(self, loop, get_x, color=YELLOW, font_size=36, dot_to_num_buff=0.075, label_direction=UL):
        loop_dot = get_special_dot(color=color)
        loop_dot.f_always.move_to(lambda: loop.pfp(get_x()))

        loop_x_label = DecimalNumber(font_size=font_size)
        loop_x_label.match_color(loop_dot[1])
        loop_x_label.set_backstroke(BLACK, 3)
        loop_x_label.always.next_to(loop_dot[0], label_direction, buff=dot_to_num_buff)
        loop_x_label.f_always.set_value(get_x)

        return Group(loop_dot, loop_x_label)


class DiscussOrderOfPoints(LoopScene):
    def construct(self):
        # Add loops
        loop = get_example_loop(2)
        loop.set_height(6)
        loop.to_edge(DOWN, buff=0.25)
        loop_func = get_quick_loop_func(loop)
        self.add(loop)

        # Dots
        uv_tracker = ValueTracker([0.8, 0.4])
        dots = self.get_movable_pair(uv_tracker, loop_func, radius=0.1)
        line = self.get_connecting_line(dots)
        mid_dot = self.get_midpoint_dot(dots)
        mid_dot.update()

        A_label = Tex("A")
        B_label = Tex("B")
        labels = VGroup(A_label, B_label)
        for dot, label in zip(dots, labels):
            label.next_to(dot, UL, buff=-0.1)

        self.add(dots)
        self.add(labels)

        dots.clear_updaters()

        # Add question
        question = TexText(R"Is $(A, B)$ distinct from $(B, A)$?", font_size=60)
        question.to_edge(UP)
        self.play(Write(question))
        self.wait()

        # Swap points
        for _ in range(2):
            self.play(
                Swap(*dots),
                Swap(*labels),
                run_time=2
            )
            self.wait()

        # Show the same midpoint
        midpoint_word = Text("Same midpoint", font_size=36)
        midpoint_word.next_to(mid_dot, LEFT, buff=0)
        midpoint_word.set_color(TEAL)

        dist_word = Text("Same distance", font_size=36)
        dist_word.set_color(TEAL)
        dist_word.next_to(ORIGIN, DOWN, SMALL_BUFF)
        dist_word.rotate(line.get_angle(), about_point=ORIGIN)
        dist_word.shift(mid_dot.get_center())

        self.play(
            GrowFromCenter(line, suspend_mobject_updating=True),
            FadeIn(mid_dot),
            FadeIn(midpoint_word, lag_ratio=0.1),
        )
        self.play(Swap(*dots))
        self.play(
            TransformMatchingStrings(
                midpoint_word, dist_word,
                key_map={"midpoint": "distance"},
                run_time=1
            )
        )
        self.play(Swap(*dots))
        self.play(FadeOut(dist_word))
        self.wait()

        # Answer
        answer = Text("It shouldn't be!")
        answer.next_to(question, DOWN)
        answer.to_edge(RIGHT)
        answer.set_color(RED)

        self.play(FadeIn(answer, lag_ratio=0.1))
        self.wait()

        # Show trivial rectangle
        frame = self.frame
        angle = 60 * DEG
        question.fix_in_frame()
        answer.fix_in_frame()
        pair_group1 = Group(dots, line)
        pair_group1.clear_updaters()
        pair_group2 = pair_group1.copy()
        dots2 = pair_group2[0]
        dots2[0].set_color(PINK)
        dots2[1].set_color(YELLOW)
        pair_group2.rotate(angle, about_point=mid_dot.get_center())

        rect = self.get_dot_polygon(
            list(it.chain(*zip(dots, dots2)))
        )
        rect.update()

        self.play(
            FadeOut(loop),
            FadeIn(pair_group2),
            VFadeIn(rect),
            frame.animate.move_to(mid_dot.get_center() + 0.5 * UP).set_height(6)
        )
        self.play(
            Rotate(pair_group2, -angle, about_point=mid_dot.get_center()),
            run_time=5
        )
        self.wait()
        self.play(
            Rotate(pair_group2, -angle, about_point=mid_dot.get_center()),
            run_time=5
        )


class MapTheStripOntoTheSurface(ShowTheSurface):
    def construct(self):
        # Setup loop and surface
        frame = self.frame
        axes, plane = self.get_axes_and_plane()
        self.add(axes, plane)

        loop = SVGMobject("gingerbread_outline")[0]
        loop.insert_n_curves(50)
        loop.set_height(5)
        loop.set_stroke(WHITE, 3)
        loop.set_fill(opacity=0)
        loop_func = get_quick_loop_func(loop)
        self.add(loop)

        surface, mesh, surface_func = self.get_surface_info(loop_func)
        surface.set_opacity(0.1)

        # Setup a pair of points wandering the surface for a bit
        uv_tracker = ValueTracker([0, 0.5])
        self.set_uv_tracker_in_motion(uv_tracker)
        dots = self.get_movable_pair(uv_tracker, loop_func, radius=0.075)
        connecting_line = self.get_connecting_line(dots)
        z_line = self.get_z_line(dots)
        top_dot = self.get_top_dot(z_line)
        pair_group = Group(connecting_line, z_line, top_dot)
        pair_group.update()

        self.add(uv_tracker, dots)
        self.play(
            frame.animate.reorient(-38, 82, 0, (-0.02, 0.06, 1.73), 8.03),
            FadeIn(pair_group),
            FadeIn(surface),
            Write(mesh, stroke_width=1, lag_ratio=0.001, time_span=(2, 4)),
            run_time=4,
        )
        pair_group.add(dots)
        self.add(pair_group)
        frame.add_ambient_rotation()
        self.wait(20)

        # Show Mobius strip
        frame.clear_updaters()

        strip = ParametricSurface(mobius_strip_func)
        strip.set_shading(0.25, 0.25, 0)
        strip.set_color(BLUE, 0.35)
        edge_parts = VGroup(
            ParametricCurve(lambda t: mobius_strip_func(0, t)),
            ParametricCurve(lambda t: mobius_strip_func(1, t)),
        )
        edge = edge_parts[0].copy().append_vectorized_mobject(edge_parts[1])
        edge.make_smooth()
        edge.set_stroke(Color("red"), 0)

        def uv_to_strip_point(u, v):
            return strip.uv_to_point(*torus_uv_to_mobius_uv(u, v))

        strip_point = get_special_dot(color=TEAL)
        strip_point.f_always.move_to(
            lambda: uv_to_strip_point(*uv_tracker.get_value())
        )

        strip_group = Group(strip, edge)
        strip_group.set_height(6)
        strip_group.next_to(surface, LEFT, buff=3)
        strip_group.rotate(10 * DEGREES, RIGHT)

        self.play(
            frame.animate.reorient(0, 68, 0, (-3.7, -0.17, 1.95), 9.56),
            FadeIn(strip_group),
            FadeIn(strip_point),
            run_time=3
        )
        self.wait(20)

        # Show the full mapping
        uv_tracker.clear_updaters()
        pre_surface = ParametricSurface(
            get_half_parametric_func(uv_to_strip_point)
        )
        pre_surface.match_style(strip)
        pre_mesh = SurfaceMesh(
            ParametricSurface(uv_to_strip_point),
            resolution=mesh.resolution,
            normal_nudge=0
        )
        pre_mesh.set_stroke(WHITE, 0.25, 0.1)

        moving_surface = surface.copy()
        moving_mesh = mesh.copy()

        self.play(FadeOut(strip_point), FadeOut(pair_group))
        self.remove(surface, mesh)
        self.play(
            Transform(moving_surface, pre_surface),
            Transform(moving_mesh, pre_mesh),
            frame.animate.reorient(-25, 71, 0, (-4.86, 0.35, 1.64), 9.84),
            run_time=10
        )
        self.wait()
        self.play(
            ReplacementTransform(moving_surface, surface),
            ReplacementTransform(moving_mesh, mesh),
            frame.animate.reorient(36, 68, 0, (-3.94, 0.76, 0.7), 10.14),
            run_time=10
        )
        self.play(
            frame.animate.reorient(0, 60, 0, (-3.7, 0.73, 0.64), 10.44),
            FadeIn(strip_point),
            FadeIn(pair_group),
            run_time=5,
        )

        # Trace the edge
        u, v = uv_tracker.get_value()
        self.play(
            uv_tracker.animate.set_value([int(2 * u), int(2 * v)]),
            loop.animate.set_stroke(opacity=0.2),
            run_time=2
        )
        uv_tracker.set_value([0, 0])
        edge.set_stroke(width=3)
        loop_copy = loop.copy().set_stroke(opacity=1)
        self.play(
            ShowCreation(edge),
            ShowCreation(loop_copy),
            uv_tracker.animate.set_value([0.999, 0.999]),
            run_time=5,
        )
        self.remove(loop_copy)
        loop.set_stroke(opacity=1)

        # Show the mapping with the edge
        edge_image = loop.copy().match_style(edge)
        moving_edge = edge.copy()

        moving_group = Group(moving_surface, moving_mesh, moving_edge)
        pre_group = Group(pre_surface, pre_mesh, edge)
        post_group = Group(surface, mesh, edge_image)

        moving_group.become(pre_group)

        self.play(
            FadeIn(moving_surface),
            FadeIn(moving_mesh),
            FadeOut(strip_point),
            FadeOut(pair_group),
        )
        self.play(
            Transform(moving_group, post_group),
            frame.animate.reorient(18, 65, 0, (-3.85, 0.76, 0.85), 10.44),
            run_time=10
        )
        self.wait()

        # Show another back and forth, but with colliding points
        rect_params = find_rectangle(loop_func)
        pre_collision_dots, post_collision_dots = coll_dot_groups = Group(
            get_special_dot(color, radius=0.075)
            for color in [YELLOW, GREEN_SCREEN]
        ).replicate(2)
        coll_dot_groups.deactivate_depth_test()
        for func, dot_group in zip([uv_to_strip_point, surface_func], coll_dot_groups):
            for dot, uv in zip(dot_group, [rect_params[0::2], rect_params[1::2]]):
                dot.move_to(func(*uv))

        moving_dots = pre_collision_dots.copy()

        self.play(
            Transform(moving_group, pre_group),
            frame.animate.reorient(-32, 70, 0, (-3.85, 0.76, 0.85), 10.44),
            run_time=5
        )
        self.play(FadeIn(pre_collision_dots))
        self.wait()
        self.play(
            Transform(moving_group, post_group),
            Transform(moving_dots, post_collision_dots),
            frame.animate.reorient(1, 59, 0, (-4.08, 0.88, 1.17), 9.71),
            run_time=5
        )

        # Show show rectangle for this collision
        uv_tracker1 = uv_tracker
        uv_tracker1.set_value(rect_params[0::2])
        pair_group1 = pair_group

        uv_tracker2 = ValueTracker(rect_params[1::2])
        dots2 = self.get_movable_pair(uv_tracker2, loop_func, colors=[RED, MAROON_B])
        connecting_line2 = self.get_connecting_line(dots2)
        pair_group2 = Group(dots2, connecting_line2)

        rectangle = self.get_dot_polygon(Group(dots[0], dots2[0], dots[1], dots2[1]))
        rectangle.set_stroke(YELLOW, 5)
        dots.update()
        dots2.update()
        rectangle.update()
        rectangle.suspend_updating()

        self.play(
            FadeOut(moving_group),
            FadeIn(pair_group1),
            FadeIn(pair_group2),
            mesh.animate.set_stroke(opacity=0.05),
            frame.animate.reorient(0, 38, 0, (-2.92, 0.03, 1.12), 6.65),
            run_time=3,
        )
        self.play(
            ShowCreation(rectangle, time_span=(0, 2)),
            frame.animate.reorient(36, 68, 0, (-2.89, -0.15, 0.74), 8.11),
            run_time=8
        )


class AmbientRectangleSearch(ShowTheSurface):
    def construct(self):
        # Setup loop
        frame = self.frame
        axes, plane = self.get_axes_and_plane()
        self.add(plane)

        loop = get_example_loop(2)
        loop.insert_n_curves(50)
        loop.set_height(5)
        loop.set_stroke(WHITE, 3)
        loop.set_fill(opacity=0)
        loop_func = get_quick_loop_func(loop)
        self.add(loop)

        # Dots
        abcd_tracker = ValueTracker(np.random.random(4))
        dots = self.get_movable_quad(abcd_tracker, loop_func, radius=0.1)
        polygon = self.get_dot_polygon(dots)
        polygon.set_stroke(YELLOW, 2)

        self.add(dots, polygon)

        # Various rectangles
        for _ in range(20):
            rect_params = find_rectangle(
                loop_func,
                initial_condition=np.random.random(4),
                target_angle=np.random.uniform(0, 90 * DEG),
            )
            self.play(abcd_tracker.animate.set_value(rect_params), run_time=2)
            self.wait()


class GenericLoopPair(ShowTheSurface):
    def construct(self):
        loop = Circle(radius=2.5)
        loop.set_stroke(WHITE, 3)
        loop.set_fill(opacity=0)
        loop_func = get_quick_loop_func(loop)
        self.add(loop)

        uv_tracker = ValueTracker([0, 0.5])
        # self.set_uv_tracker_in_motion(uv_tracker, velocity=(0.1 * PI / 2, -0.1))
        dots = self.get_movable_pair(uv_tracker, loop_func, radius=0.075, colors=[YELLOW, YELLOW])
        connecting_line = self.get_connecting_line(dots)
        connecting_line.set_stroke(YELLOW)
        pair_group = Group(dots, connecting_line)

        self.add(uv_tracker, pair_group)

        # Wait
        for _ in range(20):
            self.play(uv_tracker.animate.set_value(np.random.random(2)), run_time=2)
            self.wait(0.5)


class SudaneseBand(InteractiveScene):
    def construct(self):
        # Tranform mobius to Sudanese
        frame = self.frame
        sudanese_band = self.get_full_surface(sudanese_band_func)
        strip = self.get_full_surface(alt_mobius_strip_func)
        strip.set_height(6)
        og_strip = strip.copy()
        sudanese_band.set_height(6)

        frame.reorient(28, 75, 0, ORIGIN, 8)
        self.add(strip)
        self.play(frame.animate.increment_theta(180 * DEG), run_time=12)
        self.play(
            frame.animate.reorient(99, 102, 0),
            Transform(strip, sudanese_band, time_span=(0, 6)),
            run_time=10
        )
        self.wait(30)  # Examine

        # Back to strip
        self.play(
            Transform(strip, og_strip),
            run_time=6
        )

    def get_full_surface(self, surface_func, resolution=(101, 101)):
        surface = ParametricSurface(surface_func, resolution=resolution)
        surface.set_color(BLUE_E, 1)
        surface.set_shading(0.25, 0.25, 0)

        mesh = VGroup(
            *SurfaceMesh(surface, resolution=(21, 21), normal_nudge=1e-3),
            *SurfaceMesh(surface, resolution=(21, 21), normal_nudge=-1e-3),
        )
        mesh.set_stroke(WHITE, 0.5, 0.2)

        edge = VGroup(
            ParametricCurve(lambda t: surface_func(0, t), (0, 1, 0.01)),
            ParametricCurve(lambda t: surface_func(1, t), (0, 1, 0.01)),
        )
        edge.set_stroke(Color("red"), 2)
        edge.apply_depth_test()

        return Group(surface, mesh, edge)


class ShowSurfaceReflection(ShowTheSurface):
    def construct(self):
        # Setup loop and surface
        frame = self.frame
        axes, plane = self.get_axes_and_plane()
        self.add(axes, plane)

        loop = get_example_loop(2)
        loop.insert_n_curves(50)
        loop.set_height(5)
        loop.set_stroke(WHITE, 3)
        loop.set_fill(opacity=0)
        loop_func = get_quick_loop_func(loop)
        self.add(loop)

        surface, mesh, surface_func = self.get_surface_info(loop_func)
        surface.set_opacity(0.1)
        surface_group = Group(surface, mesh)

        self.add(surface_group)
        frame.reorient(-36, 82, 0, (0.09, 1.17, 2.23), 11.39)
        self.play(frame.animate.reorient(36, 85, 0, (-0.68, 0.78, 2.39), 11.39), run_time=6)

        # Show the reflection
        reflection = surface_group.copy()
        reflection.stretch(-1, 2, about_point=axes.c2p(0, 0, 0))
        ghost_surface = surface_group.copy()
        ghost_surface.set_opacity(0)

        self.play(
            ReplacementTransform(ghost_surface, reflection),
            frame.animate.reorient(33, 81, 0, (-0.15, 0.61, 0.29), 14.60),
            run_time=5
        )
        self.add(reflection, surface_group)

        # Emphasize the ege
        frame.add_ambient_rotation(2 * DEG)
        low_edge = loop.copy()
        low_edge.set_color(RED)
        shift_vect = 0.25 * OUT

        self.play(
            surface_group.animate.shift(shift_vect),
            loop.animate.set_stroke(Color("red")).shift(shift_vect),
            low_edge.animate.shift(-shift_vect),
            reflection.animate.shift(-shift_vect),
            FadeOut(axes),
            FadeOut(plane),
            run_time=2,
        )
        self.wait(1)
        self.play(
            FadeOut(loop, shift=-shift_vect),
            FadeOut(low_edge, shift=shift_vect),
            surface_group.animate.shift(-shift_vect),
            reflection.animate.shift(shift_vect),
            run_time=2
        )
        self.wait(12)


class ConstructKleinBottle(InteractiveScene):
    def construct(self):
        # Add arrow diagram
        square = Square()
        square.set_fill(GREY_E, 1).set_stroke(BLACK, width=0)
        square.set_height(4)

        dr_tri = Polygon(DL, DR, UR)
        dr_tri.match_style(square)
        dr_tri.replace(square)

        mobius_diagram = VGroup(
            dr_tri,
            self.get_tri_arrow(square.get_corner(DL), square.get_corner(DR)),
            self.get_tri_arrow(square.get_corner(DR), square.get_corner(UR)),
            Line(square.get_corner(DL), square.get_corner(UR)).set_stroke(Color("red"), 3)
        )

        mobius_label = Text("Möbius Strip", font_size=60)
        mobius_label.next_to(mobius_diagram, UR)
        mobius_label.shift(DOWN + 0.25 * RIGHT)
        mobius_arrow = Arrow(
            mobius_label.get_bottom(),
            mobius_diagram.get_center() + 0.5 * DR,
            path_arc=-90 * DEG,
            thickness=5
        )
        mobius_arrow.set_z_index(1)

        self.add(mobius_diagram)
        self.play(
            FadeIn(mobius_label),
            FadeIn(mobius_arrow),
        )
        self.wait()

        # Show a reflection
        reflection = mobius_diagram.copy()
        reflection.flip(UR, about_point=square.get_center())
        reflection.shift(UL)
        reflection[1:3].set_color(PINK)

        reflection_label = Text("Reflected\nMöbius Strip", font_size=60)
        reflection_label.next_to(reflection, LEFT, aligned_edge=DOWN)
        reflection_arrow = Arrow(
            reflection_label.get_top(),
            reflection.get_center() + 0.5 * LEFT + 0.25 * UP,
            path_arc=-90 * DEG,
            thickness=5
        )

        self.play(
            LaggedStart(
                TransformMatchingStrings(mobius_label.copy(), reflection_label, run_time=1),
                TransformFromCopy(mobius_diagram, reflection),
                TransformFromCopy(mobius_arrow, reflection_arrow),
                lag_ratio=0.1
            ),
            run_time=2
        )
        self.wait()

        # Glue along boundary
        glue_label = Text("Glue the boundaries", font_size=36)
        glue_label.next_to(ORIGIN, DOWN, SMALL_BUFF)
        glue_label.rotate(45 * DEG, about_point=ORIGIN)
        glue_label.shift(square.get_center())

        self.play(
            LaggedStart(
                FadeOut(reflection_arrow),
                FadeOut(mobius_arrow),
                reflection.animate.shift(DR),
                reflection_label.animate.scale(0.75).next_to(square, LEFT, MED_SMALL_BUFF),
                mobius_label.animate.scale(0.75).next_to(square, RIGHT, MED_SMALL_BUFF),
            ),
            FadeIn(glue_label, lag_ratio=0.1),
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeOut, VGroup(glue_label, reflection_label, mobius_label)),
            mobius_diagram[-1].animate.set_stroke(width=0),
            reflection[-1].animate.set_stroke(width=0),
        )

        # Cut along diagonal
        teal_arrows = mobius_diagram[1:3]
        pink_arrows = reflection[1:3]
        yellow_arrows = self.get_tri_arrow(square.get_corner(UL), square.get_corner(DR), color=YELLOW).replicate(2)
        ur_tri = Polygon(DR, UR, UL)
        dl_tri = Polygon(DR, DL, UL)
        for tri in [ur_tri, dl_tri]:
            tri.match_style(square)
            tri.replace(square)

        ur_group = VGroup(ur_tri, teal_arrows[1], pink_arrows[1])
        dl_group = VGroup(dl_tri, teal_arrows[0], pink_arrows[0])

        self.remove(mobius_diagram, reflection)
        self.add(ur_group)
        self.add(dl_group)

        self.play(*(Write(arrow, stroke_color=YELLOW) for arrow in yellow_arrows))
        ur_group.add(yellow_arrows[0])
        dl_group.add(yellow_arrows[1])
        self.play(VGroup(ur_group, dl_group).animate.space_out_submobjects(3))
        self.wait()

        # Flip and glue
        frame = self.frame
        self.play(
            dl_group.animate.next_to(ORIGIN, UP, 0.5),
            ur_group.animate.flip(UR).next_to(ORIGIN, DOWN, 0.5),
            frame.animate.set_height(10),
            run_time=2,
        )
        self.wait()
        self.play(
            ur_group.animate.shift(-ur_tri.get_top()),
            dl_group.animate.shift(-dl_tri.get_bottom()),
        )
        self.play(teal_arrows.animate.set_stroke(width=0).set_fill(opacity=0))

        # Shear back into square
        pre_square = square.copy()
        pre_square.apply_matrix(np.matrix([[1, -1], [0, 1]]).T)
        pre_square.move_to(VGroup(dl_tri, dr_tri), UP)

        trg_yellow_arrows = VGroup(
            self.get_tri_arrow(square.get_corner(DR), square.get_corner(DL), color=YELLOW).flip(RIGHT),
            self.get_tri_arrow(square.get_corner(UL), square.get_corner(UR), color=YELLOW),
        )

        self.remove(ur_tri, dl_tri)
        self.add(pre_square, pink_arrows, yellow_arrows)
        self.play(
            Transform(pre_square, square),
            Transform(yellow_arrows, trg_yellow_arrows),
            pink_arrows[0].animate.move_to(square.get_left()),
            pink_arrows[1].animate.move_to(square.get_right()),
            frame.animate.set_height(8),
            run_time=2
        )
        self.wait()

        # Fold into half tube
        klein_func = self.get_kelin_bottle_func()
        near_smooth = bezier([0, 0.1, 0.9, 1])
        surfaces = Group(
            TexturedSurface(ParametricSurface(func), "KleinBottleTexture")
            for func in [
                square_func,
                tube_func,
                lambda u, v: torus_func(u, 0.5 * v),
                lambda u, v: klein_func(u, 0.5 * near_smooth(v)),
            ]
        )
        for surface in surfaces:
            surface.set_opacity(0.9)
            surface.set_shading(0.3, 0.2, 0)
        square3d, tube, half_torus, half_klein = surfaces
        square3d.replace(square)
        square3d.shift(4 * OUT)
        moving_surface = square3d.copy()

        tube.set_width(square.get_width() / PI)
        tube.set_height(square.get_height(), stretch=True)
        tube.move_to(square3d)

        half_torus.match_depth(tube)
        half_torus.move_to(tube)

        self.play(
            FadeIn(moving_surface, shift=square3d.get_z() * OUT),
            frame.animate.reorient(0, 56, 0, (0.07, 0.52, 2.39), 11.25),
            run_time=3,
        )
        self.play(Transform(moving_surface, tube), run_time=4)
        self.wait()
        self.play(Transform(moving_surface, half_torus, path_arc=PI / 2), run_time=4)
        self.wait()
        self.play(Transform(moving_surface, half_klein), run_time=4)

        # Transition to full Klein Bottle
        klein_diagram = VGroup(pre_square, pink_arrows, yellow_arrows)
        v_upper_bound = 0.85
        self.play(
            UpdateFromAlphaFunc(moving_surface, lambda m, a: m.match_points(
                ParametricSurface(lambda u, v: klein_func(u, interpolate(0.5, 1, a) * near_smooth(v)))
            ).set_opacity(interpolate(0.9, 0.75, a))),
            klein_diagram.animate.set_x(-5),
            frame.animate.reorient(0, 46, 0, (-0.71, -0.11, 1.71), 10.87),
            run_time=8
        )
        self.wait()
        self.play(
            klein_diagram.animate.next_to(moving_surface, LEFT, buff=2),
            frame.animate.reorient(0, 0, 0, (-3.2, 0.03, 0.0), 12.58),
            run_time=4
        )
        self.wait()

    def get_tri_arrow(self, start, end, color=TEAL, stroke_width=3, tip_width=0.35):
        line = Line(start, end)
        line.set_stroke(color, stroke_width)
        tips = ArrowTip().replicate(3)
        tips.set_fill(color)
        tips.set_width(tip_width)
        tips.rotate(line.get_angle())
        for alpha, tip in zip(np.linspace(0.2, 0.8, 3), tips):
            tip.move_to(line.pfp(alpha))

        return VGroup(line, tips)

    def get_kelin_bottle_func(self, width=4, z=4):
        # Test kelin func
        ref_svg = SVGMobject("KleinReference")[0]
        ref_svg.make_smooth(approx=False)
        ref_svg.add_line_to(ref_svg.get_start())
        ref_svg.set_stroke(WHITE, 3)
        ref_svg.set_width(width)
        ref_svg.rotate(PI)
        ref_svg.set_z(4)
        ref_svg.insert_n_curves(100)

        # curve_func = get_quick_loop_func(ref_svg)
        curve_func = ref_svg.quick_point_from_proportion
        radius_func = bezier([1, 1, 0.5, 0.3, 0.3, 0.3, 1.0])
        tan_alpha_func = bezier([1, 1, 0, 0, 0, 0, 1, 1])
        v_alpha_func = squish_rate_func(smooth, 0.25, 0.75)

        def pre_klein_func(u, v):
            dv = 1e-2
            c_point = curve_func(v)
            c_prime = normalize((curve_func(v + dv) - curve_func(v - dv)) / (2 * dv))
            tangent_alpha = tan_alpha_func(v)
            # tangent = interpolate(c_prime, UP if v < 0.5 else DOWN, tangent_alpha)
            tangent = interpolate(c_prime, interpolate(UP, DOWN, v_alpha_func(v)), tangent_alpha)

            perp = normalize(cross(tangent, OUT))
            radius = radius_func(v)

            return c_point + radius * (math.cos(TAU * u) * OUT - math.sin(TAU * u) * perp)

        v_upper_bound = 0.85

        def true_kelin_func(u, v):
            if v <= v_upper_bound:
                return pre_klein_func(u, v)
            else:
                alpha = inverse_interpolate(v_upper_bound, 1, v)
                return interpolate(pre_klein_func(u, v_upper_bound), pre_klein_func(1 - u, 0), alpha)

        return true_kelin_func


class PuzzleOverMobiusDiagram(ConstructKleinBottle):
    def construct(self):
        # Test
        triangle = Polygon(DL, DR, UR)
        triangle.set_fill(GREY_E, 1)
        triangle.set_stroke(WHITE, 0)
        triangle.set_height(4)
        triangle.to_edge(UP, buff=1)

        low_arrow = self.get_tri_arrow(triangle.get_corner(DL), triangle.get_corner(DR), stroke_width=4)
        right_arrow = self.get_tri_arrow(triangle.get_corner(DR), triangle.get_corner(UR), stroke_width=4)
        edge = Line(triangle.get_corner(DL), triangle.get_corner(UR))
        edge.set_stroke(Color("red"), 3)

        self.add(triangle, edge, low_arrow, right_arrow)
        self.wait()

        low_arrow.save_state()
        self.play(
            triangle.animate.set_opacity(0.25),
            edge.animate.set_opacity(0.1),
            low_arrow.animate.rotate(90 * DEG).next_to(right_arrow, RIGHT)
        )
        self.wait()
        self.play(low_arrow.animate.move_to(right_arrow))
        self.play(low_arrow.animate.next_to(right_arrow, RIGHT))
        self.play(
            Restore(low_arrow),
            triangle.animate.set_fill(opacity=1),
            edge.animate.set_stroke(opacity=1),
        )

        # Show the dots
        dots = GlowDot(color=BLUE, radius=0.25, glow_factor=1.5).replicate(2)
        top_dot = dots[0].copy()
        top_dot.move_to(right_arrow.get_top())
        for _ in range(3):
            dots[0].move_to(low_arrow.get_left())
            dots[1].move_to(right_arrow.get_bottom())
            self.add(top_dot)
            self.wait(1 / 30)
            self.remove(top_dot)
            self.play(
                dots[0].animate.move_to(low_arrow.get_right()),
                dots[1].animate.move_to(right_arrow.get_top()),
                run_time=3,
                rate_func=linear
            )


class ShowAngleInformation(ShowTheSurface):
    def construct(self):
        # Setup (fairly heavily copied from above)
        frame = self.frame
        axes, plane = self.get_axes_and_plane()
        plane.fade(0.5)
        frame.set_height(6)
        frame.set_x(1)
        self.add(plane)

        loop = get_example_loop()
        loop.set_height(5.5).center()
        loop_func = get_quick_loop_func(loop)
        self.add(loop)

        square_params = find_rectangle(loop_func, target_angle=90 * DEG)
        uv_tracker = ValueTracker(square_params[0::2])
        dots = self.get_movable_pair(uv_tracker, loop_func, radius=0.05)
        connecting_line = self.get_connecting_line(dots)
        connecting_line.update().suspend_updating()
        midpoint_dot = self.get_midpoint_dot(dots)
        midpoint_dot.update()
        pair_group = Group(dots, connecting_line, midpoint_dot)

        self.add(pair_group)

        # Corner coordinates
        corner_coords = Tex("(x, y, d)")
        corner_coords.next_to(frame.get_corner(UR), DL)
        self.add(corner_coords)

        # Add coordinates
        coords = Tex("(x, y)")
        coords.set_backstroke(BLACK, 5)
        coords.set_fill(WHITE, 1)

        coords.next_to(midpoint_dot, DR, buff=-0.1)
        self.play(
            Write(coords),
            FlashAround(corner_coords["x, y"], time_width=1.5),
            run_time=1.5
        )

        # Show the distance
        brace = Brace(Line(LEFT, RIGHT).set_width(connecting_line.get_length()), DOWN, buff=SMALL_BUFF)
        brace.rotate(connecting_line.get_angle() + PI, about_point=ORIGIN)
        brace.shift(connecting_line.get_center())
        brace.set_fill(GREY, 1)
        d_label = Tex("d")
        d_label.set_backstroke(BLACK, 5)
        d_label.move_to(brace.get_center() + 0.5 * normalize(brace.get_center() - midpoint_dot.get_center()))

        self.play(
            GrowFromCenter(brace),
            Write(d_label),
            FlashAround(corner_coords["d"], time_width=1.5, run_time=1.5)
        )
        self.wait()

        # Show the angle
        cross_point = connecting_line.pfp(inverse_interpolate(
            connecting_line.get_y(DOWN), connecting_line.get_y(UP), -1
        ))
        h_line = Line(LEFT, RIGHT)
        h_line.set_width(4)
        h_line.move_to(cross_point + RIGHT)
        h_line.set_stroke(WHITE, 2)
        arc = Arc(0, connecting_line.get_angle(), radius=0.45, arc_center=cross_point)
        theta = Tex(R"\theta")
        theta.next_to(arc.pfp(0.5), RIGHT, SMALL_BUFF).shift(SMALL_BUFF * UP)

        self.play(
            ShowCreation(h_line),
            ShowCreation(arc),
            Write(theta),
        )

        # New corner coords
        new_corner_coords = Tex(R"(x, y, d, \theta)")
        new_corner_coords.move_to(corner_coords, RIGHT)
        self.play(
            *(
                ReplacementTransform(corner_coords[substr], new_corner_coords[substr])
                for substr in ["(x, y, d", ")"]
            ),
            TransformFromCopy(theta, new_corner_coords[R"\theta"]),
            Write(new_corner_coords[R","][-1]),
        )
        self.add(new_corner_coords)
        self.wait()

        # Show the other pair
        uv_tracker2 = ValueTracker(square_params[1::2])
        dots2 = self.get_movable_pair(uv_tracker2, loop_func, radius=0.05, colors=[RED, MAROON_B])
        connecting_line2 = self.get_connecting_line(dots2)
        pair_group2 = Group(dots2, connecting_line2)
        midpoint = midpoint_dot.get_center()

        self.play(
            ReplacementTransform(
                connecting_line.copy().clear_updaters(),
                connecting_line2,
                path_arc=-90 * DEG,
                suspend_mobject_updating=True
            ),
            Rotate(brace, -90 * DEG, about_point=midpoint),
            d_label.animate.rotate(-90 * DEGREES, about_point=midpoint).rotate(90 * DEG),
            coords.animate.next_to(midpoint_dot, DL, buff=-0.1),
            loop.animate.set_stroke(opacity=0.5)
        )
        self.play(FadeIn(dots2))
        self.wait()

        # Show the right angle
        cross_point2 = connecting_line2.pfp(inverse_interpolate(
            connecting_line2.get_start()[1], connecting_line2.get_end()[1], 0
        ))
        h_line2 = Line(cross_point2, cross_point2 + 2 * RIGHT)
        h_line2.match_style(h_line)
        arc2 = Arc(0, PI + connecting_line2.get_angle(), radius=0.45, arc_center=cross_point2)
        theta_plus = Tex(R"\theta + 90^\circ", font_size=42)
        theta_plus.next_to(arc2.pfp(0.3), UR, buff=SMALL_BUFF)

        elbow = Elbow()
        elbow.rotate(connecting_line2.get_angle() + 90 * DEG, about_point=ORIGIN)
        elbow.shift(midpoint_dot.get_center())
        perp_label = Tex(R"90^\circ", font_size=36)
        perp_label.next_to(elbow.pfp(0.5), UP, SMALL_BUFF)

        self.play(
            FadeOut(brace),
            FadeOut(d_label),
            ShowCreation(elbow),
            Write(perp_label),
        )

        # New pair corner label
        shift_vect = 0.5 * RIGHT
        low_corner_coords = Tex(R"(x, y, d, \theta + 90^\circ)")
        low_corner_coords.next_to(new_corner_coords, DOWN, aligned_edge=RIGHT)
        low_corner_coords.shift(shift_vect)

        self.play(
            TransformMatchingStrings(new_corner_coords.copy(), low_corner_coords),
            new_corner_coords.animate.shift(shift_vect),
            frame.animate.shift(shift_vect),
        )
        self.wait()

        # Show the square
        square = self.get_dot_polygon(
            list(it.chain(*zip(dots, dots2)))
        )
        square.update().suspend_updating()
        square.set_stroke(YELLOW, 3)
        self.play(ShowCreation(square))
        self.wait()

        # Comment on four dimension


class RectanglesOfAllAspectRatios(LoopScene):
    def construct(self):
        # Loop
        loop = self.get_loop()
        loop.set_height(6.5)
        loop.to_corner(DR)
        loop.set_stroke(WHITE, 1).set_fill(opacity=0)
        loop_func = get_quick_loop_func(loop)
        self.add(loop)

        # Dots
        angle_tracker = ValueTracker(90 * DEG)
        get_angle = angle_tracker.get_value
        square_params = find_rectangle(
            loop_func,
            initial_condition=np.arange(0.5, 1, 0.1),
            target_angle=angle_tracker.get_value()
        )
        abcd_tracker = ValueTracker(square_params)
        dots = self.get_movable_quad(abcd_tracker, loop_func, radius=0.075)
        rect = self.get_dot_polygon(dots)
        rect.set_stroke(YELLOW)

        rect_group = Group(abcd_tracker, dots, rect)

        self.add(rect_group)

        # Show aspect ratio
        def get_aspect_ratio():
            phi = get_angle() / 2
            return math.cot(phi)

        sample_rect = Square(side_length=2)
        sample_rect_center = 4 * LEFT + DOWN
        sample_rect_size_factor = 2.5
        sample_rect.set_stroke(YELLOW)

        def update_sample_rect(rect):
            phi = get_angle() / 2
            sf = sample_rect_size_factor
            rect.set_shape(sf * math.cos(phi), sf * math.sin(phi))
            rect.move_to(sample_rect_center)

        sample_rect.add_updater(update_sample_rect)

        aspect_ratio_label = TexText("Aspect ratio = 1:1")
        aspect_ratio_label.next_to(sample_rect, UP).shift(MED_SMALL_BUFF * LEFT)
        ar_label = aspect_ratio_label["1:1"][0]

        self.add(aspect_ratio_label[:-3])
        self.add(ar_label)
        self.add(sample_rect)
        self.wait()

        # Change angle
        def true_find_rect(loop_func, angle, max_tries=10, max_cost=1e-2):
            for _ in range(max_tries):
                result, cost = find_rectangle(
                    loop_func,
                    initial_condition=np.random.random(4),
                    target_angle=angle,
                    n_refinements=4,
                    return_cost=True
                )
                if cost < max_cost:
                    return result
            return result

        aspect_ratio_pairs = self.get_aspect_ratio_pairs()
        for tex, ratio in aspect_ratio_pairs:
            label = Tex(tex)
            label.move_to(ar_label, LEFT)
            if ratio < 1:
                ratio = 1.0 / ratio
            angle_tracker.set_value(2 * math.atan(1.0 / ratio))
            abcd_tracker.set_value(true_find_rect(loop_func, get_angle()))
            ar_label.set_submobjects(list(label))
            self.wait(1 / 3)

    def get_loop(self):
        return get_example_loop(4)

    def get_aspect_ratio_pairs(self):
        return [
            ("16 : 9", 16 / 9),
            ("4 : 3", 4 / 3),
            ("10 : 1", 10 / 1),
            (R"\sqrt{2} : 1", math.sqrt(2)),
            (R"5 : 2", 5 / 2),
            (R"\pi : 1", math.pi),
            (R"\varphi : 1", (1 + math.sqrt(5)) / 2),  # Golden ratio
            (R"e : 1", math.e),
            ("21 : 9", 21 / 9),  # Ultrawide
            (R"\sqrt{3} : 1", math.sqrt(3)),
            ("3 : 2", 3 / 2),    # Classic photography
            ("1 : 1", 1 / 1),    # Square
            (R"2\pi : 1", 2 * math.pi),
            ("7 : 5", 7 / 5),
            (R"\ln(10) : 1", math.log(10)),
            ("8 : 5", 8 / 5),
            ("12 : 5", 12 / 5),
            (R"\sqrt{5} : 1", math.sqrt(5)),
            ("9 : 16", 9 / 16),  # Portrait 16:9
            ("2 : 3", 2 / 3),    # Portrait photography
            (R"\pi : 2", math.pi / 2),
            ("7 : 3", 7 / 3),
            (R"2\sqrt{2} : 1", 2 * math.sqrt(2)),
            ("15 : 4", 15 / 4),
            (R"\sqrt{7} : 1", math.sqrt(7)),
            ("11 : 4", 11 / 4),
            (R"e : 2", math.e / 2),
            ("13 : 5", 13 / 5),
            ("8 : 3", 8 / 3),
            (R"\pi^2 : 9", math.pi ** 2 / 9)
        ]


class AllAspectRatioXMaxTree(RectanglesOfAllAspectRatios):
    def get_loop(self):
        return SVGMobject("xmas_tree").family_members_with_points()[0]

    def get_aspect_ratio_pairs(self):
        return [
            (R"\sqrt{10} : 1", math.sqrt(10)),
            (R"\pi : \sqrt{2}", math.pi / math.sqrt(2)),
            ("17 : 10", 17 / 10),
            (R"\ln(2) : 1", math.log(2)),
            ("14 : 3", 14 / 3),
            (R"3\sqrt{3} : 2", 3 * math.sqrt(3) / 2),
            ("19 : 6", 19 / 6),
            (R"e^2 : 10", math.e ** 2 / 10),
            ("25 : 16", 25 / 16),
            (R"\varphi^2 : 2", ((1 + math.sqrt(5)) / 2) ** 2 / 2),
            ("23 : 9", 23 / 9),
            (R"\sqrt{6} : 1", math.sqrt(6)),
            ("18 : 5", 18 / 5),
            (R"2\pi : 3", 2 * math.pi / 3),
            ("16 : 10", 16 / 10),
            (R"\sqrt{13} : 2", math.sqrt(13) / 2),
            ("22 : 7", 22 / 7),  # Common π approximation ratio
            (R"e : \sqrt{5}", math.e / math.sqrt(5)),
            ("20 : 11", 20 / 11),
            (R"\ln(5) : 1", math.log(5)),
            ("27 : 16", 27 / 16),
            (R"3\varphi : 4", 3 * ((1 + math.sqrt(5)) / 2) / 4),
            ("24 : 11", 24 / 11),
            (R"\sqrt{15} : 2", math.sqrt(15) / 2),
            ("21 : 13", 21 / 13),
            (R"5\pi : 8", 5 * math.pi / 8),
            ("26 : 15", 26 / 15),
            (R"\sqrt{17} : 3", math.sqrt(17) / 3),
            ("29 : 12", 29 / 12),
            (R"e\pi : 10", math.e * math.pi / 10)
        ]


class AllAspectRatioPi(RectanglesOfAllAspectRatios):
    def get_loop(self):
        return Tex(R"\pi").family_members_with_points()[0]


class TrackTheAngle(ShowTheSurface):
    initial_uv = [0.55, 0.75]
    second_uv = [0.6, 0.66]
    limiting_uv = [0.659, 0.66]

    def construct(self):
        # Initial setup
        frame = self.frame
        frame.set_height(6.5)
        frame.set_x(1.5)
        loop, pair_group = self.setup_loop()
        uv_tracker, dots, connecting_line = pair_group
        uv_tracker.set_value(self.initial_uv)
        pair_group.update()

        self.remove(pair_group)

        # Show smoothness
        x_tracker = ValueTracker(0.5)
        get_x = x_tracker.get_value
        dx = 1e-3
        line = Line(LEFT, RIGHT)
        line.set_stroke(TEAL, 4)
        line.f_always.put_start_and_end_on(
            lambda: loop.quick_point_from_proportion(get_x()),
            lambda: loop.quick_point_from_proportion(get_x() + dx)
        )
        line.always.set_length(3)

        self.add(line)
        self.play(x_tracker.animate.set_value(0.9), run_time=10)

        # Temp
        x_tracker.set_value(0)
        self.play(x_tracker.animate.set_value(0.99), run_time=25, rate_func=linear)

        # Angle label
        angle_label = self.get_angle_label(connecting_line)
        theta_label = always_redraw(lambda: self.get_theta_label(connecting_line))

        self.play(
            VFadeIn(theta_label, suspend_mobject_updating=True),
            VFadeIn(angle_label, suspend_mobject_updating=True),
            FadeOut(line),
            FadeIn(pair_group),
            loop.animate.set_stroke(WHITE, 1, 0.5)
        )

        # Move points
        self.play(
            uv_tracker.animate.set_value(self.second_uv),
            run_time=4
        )
        self.play(
            uv_tracker.animate.set_value(self.limiting_uv),
            run_time=15,
            rate_func=bezier([0, 0, 1, 1, 1, 1, 1, 1, 1]),
        )

    def setup_loop(self):
        # Setup (fairly heavily copied from above)
        axes, plane = self.get_axes_and_plane()
        plane.fade(0.5)
        self.add(plane)

        loop = self.get_loop()
        loop.set_height(5.5).center()
        loop_func = loop.quick_point_from_proportion
        self.add(loop)

        uv_tracker = ValueTracker([0, 0.5])
        dots = self.get_movable_pair(uv_tracker, loop_func, radius=0.075)
        connecting_line = self.get_connecting_line(dots)
        connecting_line.always.set_length(20)
        pair_group = Group(uv_tracker, dots, connecting_line)
        pair_group.update()

        self.add(pair_group)

        return loop, pair_group

    def get_loop(self):
        return get_example_loop(4)

    def get_angle_label(self, line):
        label = Tex(R"\theta = 100.0^\circ")
        label.next_to(self.frame.get_corner(UR), DL)
        angle = label.make_number_changeable("100.0", edge_to_fix=RIGHT)
        angle.f_always.set_value(lambda: (line.get_angle() % PI) / DEG)
        return label

    def get_theta_label(self, line):
        h_line = Line(ORIGIN, RIGHT)
        line_center = line.get_center()
        h_line.move_to(line_center, LEFT)
        angle = line.get_angle() % PI
        arc = Arc(0, angle, radius=0.25, arc_center=line_center)
        theta = Tex(R"\theta", font_size=36)
        try:
            arc_center = arc.pfp(0.5)
        except Exception:
            arc_center = arc.get_center()
        theta.move_to(arc_center + 0.8 * (arc_center - line_center))
        return VGroup(h_line, arc, theta)


class TrackTheAngleForFractal(TrackTheAngle):
    def get_loop(self, n_iters=7):
        triangle = Triangle()
        triangle.set_height(4)
        a, b, c = triangle.get_vertices()
        snowflake = VMobject().set_points_as_corners(np.vstack([
            self.get_koch_line_points(a, c, n_iters),
            self.get_koch_line_points(c, b, n_iters),
            self.get_koch_line_points(b, a, n_iters),
        ]))
        snowflake.set_stroke(WHITE, 1)

        return snowflake

    def get_koch_line_points(self, start, end, n_iters=7):
        """
        Return points for a Koch snowflake portion,
        not including the end
        """
        a, b, c, d = np.linspace(start, end, 4)
        tip = b + rotate_vector(c - b, 60 * DEG)
        if n_iters == 0:
            return np.array([a, b, tip, c])
        return np.vstack([
            self.get_koch_line_points(a, b, n_iters - 1),
            self.get_koch_line_points(b, tip, n_iters - 1),
            self.get_koch_line_points(tip, c, n_iters - 1),
            self.get_koch_line_points(c, d, n_iters - 1),
        ])


class KochZoom(TrackTheAngleForFractal):
    def construct(self):
        snowflake = self.get_loop(9)
        self.add(snowflake)
        self.play(
            self.frame.animate.reorient(0, 0, 0, (-1.12, 1.0, 0.0), 0.23),
            rate_func=bezier([0, 0, 1, 1, 1, 1, 1, 1]),
            run_time=10
        )


class MobiusStripsAndKleinBottlesIn4D(ConstructKleinBottle):
    def construct(self):
        # Add surfaces
        four_d = Text("A certain\n4D space")
        cloud = ThoughtBubble(four_d, bulge_radius=0.15)[0][-1]
        cloud.add(four_d)
        cloud.set_width(4)
        cloud.to_edge(RIGHT).shift(2 * DOWN)
        self.add(cloud)

        strip = ParametricSurface(mobius_strip_func)
        strip.rotate(45 * DEG, LEFT)
        strip.set_width(3)

        bottle = ParametricSurface(self.get_kelin_bottle_func())
        bottle.rotate(30 * DEG, LEFT)
        bottle.set_height(3)
        bottle.always_sort_to_camera(self.camera)

        surfaces = Group(strip, bottle)
        surfaces.arrange(RIGHT, buff=1.0)
        surfaces.set_width(5)
        surfaces.to_corner(UR)

        for surface in surfaces:
            surface.set_color(BLUE_D, 0.5)
            surface.set_shading(0.4, 0.3, 0)
            mesh = SurfaceMesh(surface, resolution=(11, 51))
            mesh.set_stroke(WHITE, 0.5, 0.2)
            mesh.deactivate_depth_test()

            arrow = Arrow(surface, cloud.get_top(), thickness=5)

            self.play(
                FadeIn(surface),
                Write(mesh, stroke_width=0.5, run_time=2),
                GrowArrow(arrow),
            )


class MusicalIntervalsAsPairs(InteractiveScene):
    def construct(self):
        # Add piano
        piano = Piano()[:39]
        piano.center()
        piano.set_width(FRAME_WIDTH)
        piano.set_shading(0.2, 0.1, 0)

        keys = piano[15:27]
        key_labels = VGroup(map(Tex, [
            R"C",
            R"C^{\#}",
            R"D",
            R"D^{\#}",
            R"E",
            R"F",
            R"F^{\#}",
            R"G",
            R"G^{\#}",
            R"A",
            R"A^{\#}",
            R"B",
        ]))
        key_labels.scale(0.6)
        key_labels.set_stroke(WHITE, 1)
        for key, label in zip(keys, key_labels):
            label.next_to(key.get_bottom(), UP, buff=0.1)

        self.add(piano)

        # Highlight random key pairs
        random.seed(0)
        indices = list(range(12))
        keys.save_state()
        for _ in range(24):
            i, j = random.sample(indices, 2)
            keys[i].set_color(TEAL)
            keys[j].set_color(TEAL)
            self.add(key_labels[i])
            self.add(key_labels[j])
            self.wait(0.5)
            # self.play_notes(i, j, 0.5)  # Only used for screen recording
            self.remove(key_labels)
            keys.restore()

        # Show the circle
        circle = Circle(radius=3)
        circle.set_stroke(WHITE, 3)
        circle.flip(axis=UR)

        key_labels.target = key_labels.generate_target()
        dots = Group()
        for label, alpha in zip(key_labels.target, np.arange(0, 1, 1 / 12)):
            point = circle.pfp(alpha)
            label.move_to(1.1 * point)
            dots.add(GlowDot(point, color=TEAL, radius=0.3))

        self.play(
            FadeOut(piano[:15]),
            keys.animate.set_opacity(0.5),
            FadeOut(piano[27:]),
            VFadeIn(key_labels),
        )
        self.remove(key_labels)
        self.play(
            ShowCreation(circle),
            LaggedStart(
                (FadeTransform(Group(key), dot)
                for key, dot in zip(keys, dots)),
                lag_ratio=0.1,
                group_type=Group,
            ),
            MoveToTarget(key_labels, lag_ratio=0.01),
            run_time=2
        )

        # Show the random connections again
        random.seed(0)
        line = Line().set_stroke(TEAL, 3)
        self.add(line)

        for _ in range(24):
            i, j = random.sample(indices, 2)
            line.put_start_and_end_on(
                dots[i].get_center(),
                dots[j].get_center(),
            )
            self.wait(1 / 3)

    def play_notes(self, i, j, duration=0.5, sample_rate=44100):
        """
        Play two notes simultaneously, specified as half steps above middle C.

        Parameters:
        i (int): Half steps above middle C for first note
        j (int): Half steps above middle C for second note
        duration (float): Length of time to play in seconds
        sample_rate (int): Number of samples per second
        """
        import sounddevice as sd
        # Middle C is 261.63 Hz
        base_freq = 261.63

        # Calculate frequencies using equal temperament formula
        freq1 = base_freq * (2 ** (i / 12))
        freq2 = base_freq * (2 ** (j / 12))

        # Generate time array
        t = np.linspace(0, duration, int(sample_rate * duration), False)

        # Generate sine waves for each note
        note1 = np.sin(2 * np.pi * freq1 * t)
        note2 = np.sin(2 * np.pi * freq2 * t)

        # Combine notes and normalize
        combined = (note1 + note2) / 2

        # Play the sound
        sd.play(combined, sample_rate)
        sd.wait()  # Wait until the sound has finished playing
