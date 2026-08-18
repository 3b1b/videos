from manim_imports_ext import *


class ChordQuestion(InteractiveScene):
    random_seed = 6

    def construct(self):
        # Add circle
        radius = 2.2
        circle = Circle(radius=radius)
        circle.set_stroke(BLUE, 2)
        circle.to_edge(DOWN)
        self.add(circle)

        # Choose 10 random chords
        chord_counter, chord_count_tracker = self.get_counter(R"1 \text{ chords}")
        chord_counter.to_edge(UP)

        chord_groups = Group(
            self.get_random_chord(circle, color=random_bright_color(hue_range=(0.1, 0.2)))
            for n in range(10)
        )

        self.add(chord_counter)
        self.play(
            self.draw_chord(chord_groups[0]),
            FadeIn(chord_counter),
        )
        self.wait()
        self.play(
            ShowIncreasingSubsets(chord_groups[1:], int_func=np.floor),
            chord_count_tracker.animate.set_value(10),
            run_time=2,
        )
        self.wait()

        # Count intersection points
        intersect_counter, intersection_count_tracker = self.get_counter(R"1 \text{ intersections}")
        intersect_counter.next_to(chord_counter, DOWN, MED_LARGE_BUFF)
        intersect_counter.shift(0.25 * RIGHT)
        intersect_counter.set_color(RED)

        circle_dots = Group(chord_group[:2] for chord_group in chord_groups)
        chords = VGroup(chord_group[2] for chord_group in chord_groups)

        intersection_points = self.get_all_intersections(chords, circle, intersection_radius=0.075)

        self.add(intersection_points)
        self.add(intersect_counter)
        self.play(
            FadeOut(circle_dots, time_span=(0, 0.5)),
            ShowIncreasingSubsets(intersection_points, run_time=2, rate_func=rush_from),
            UpdateFromFunc(intersection_count_tracker, lambda m: m.set_value(len(intersection_points))),
            chords.animate.set_stroke(width=1),
            run_time=2
        )
        self.wait()

        # Make it 100
        added_chords = VGroup(
            self.get_random_chord(circle, color=random_bright_color(hue_range=(0.1, 0.2)))[-1]
            for n in range(90)
        )
        added_chords.set_stroke(width=1)
        all_chords = VGroup(*chords, *added_chords)
        cover_rect = BackgroundRectangle(intersect_counter)
        cover_rect.set_fill(BLACK, 0.7)
        cover_rect.stretch(2, 0)

        all_intersections = self.get_all_intersections(all_chords, circle)
        intersection_cloud = DotCloud([dot.get_center() for dot in all_intersections])
        intersection_cloud.set_color(RED).set_radius(0.025)
        intersection_cloud.make_3d().set_shading(0.1, 0.1, 0.1)

        self.play(
            chords.animate.set_stroke(width=1),
            LaggedStartMap(ShowCreation, added_chords, lag_ratio=2 / len(added_chords)),
            chord_count_tracker.animate.set_value(100),
            FadeIn(cover_rect),
            FadeOut(intersection_points)
        )
        self.wait()
        self.play(
            ShowCreation(intersection_cloud),
            intersection_count_tracker.animate.set_value(intersection_cloud.get_num_points()),
            all_chords.animate.set_stroke(opacity=0.5),
            FadeOut(cover_rect),
        )
        self.wait()

        # Clear the board
        frame = self.frame
        chord_counter.clear_updaters()
        intersect_counter.clear_updaters()
        self.play(
            FadeOut(chord_counter),
            FadeOut(intersect_counter),
            FadeOut(all_chords),
            FadeOut(intersection_cloud),
            frame.animate.move_to(circle)
        )
        self.wait()

        # Choose a random point
        point0, point0_anim = self.random_point_animations(circle, YELLOW)

        self.play(point0_anim)

        # Note uniformity
        points = Group(
            self.get_point(YELLOW).move_to(circle.pfp(a))
            for a in np.arange(0, 1, 0.01)
        )
        arc = Arc(-20 * DEG, 40 * DEG, radius=radius, arc_center=circle.get_center())
        arc.set_stroke(RED, 5)
        arc.rotate(angle_of_vector(point0.get_center() - circle.get_center()), about_point=circle.get_center())

        self.play(FadeIn(points, lag_ratio=0.1))
        self.play(FadeOut(points))
        self.wait()
        self.add(arc, point0)
        self.play(ShowCreation(arc))
        self.wait()
        self.play(FadeOut(arc))

        # Choose a second point, make a chord
        point1, point1_anim = self.random_point_animations(circle, YELLOW)
        self.play(point1_anim)
        self.wait()

        chord = Line(point0.get_center(), point1.get_center())
        chord.set_stroke(YELLOW, 2)
        self.play(ShowCreation(chord))
        self.wait()

        # Now choose a few more like this
        remaining_points = Group()
        remaining_anims = []
        for n in range(9):
            color = random_bright_color(hue_range=(0.1, 0.2))
            for k in range(2):
                point, anim = self.random_point_animations(circle, color=color)
                remaining_points.add(point)
                remaining_anims.append(anim)

        self.play(LaggedStart(*remaining_anims, lag_ratio=0.02, run_time=3))

        remaining_chords = VGroup()
        for p0, p1 in zip(remaining_points[0::2], remaining_points[1::2]):
            remaining_chords.add(
                Line(p0.get_center(), p1.get_center(), color=p0[0].get_color())
            )
        remaining_chords.set_stroke(width=2)

        self.play(ShowCreation(remaining_chords, lag_ratio=0.1))
        self.wait()

        # Show intersections again
        all_chords = VGroup(chord, *remaining_chords)
        all_points = Group(point0, point1, *remaining_points)
        intersection_points = self.get_all_intersections(all_chords, circle)

        self.play(
            FadeOut(all_points),
            FadeIn(intersection_points),
            all_chords.animate.set_stroke(width=1)
        )
        self.wait()

        # Cycle through cases with 10
        self.remove(all_chords, intersection_points)
        self.cycle_though_examples(circle, 10)
        self.add(all_chords, intersection_points)

        # Back to 100
        all_chords.set_stroke(width=1)
        self.play(
            ShowCreation(added_chords, lag_ratio=0.01),
            FadeOut(intersection_points),
            FadeOut(chords),
        )
        self.play(FadeIn(intersection_cloud))
        self.wait()

        # Cycle through
        self.remove(all_chords, added_chords, intersection_cloud)
        self.cycle_though_examples(circle, 100, chord_width=1, cloud=True)
        self.wait()

    def random_point_animations(self, circle, color=YELLOW, min_turns=1, run_time=2):
        prop_tracker = ValueTracker(random.random())
        point = self.get_point(color)
        point.add_updater(lambda m: m.move_to(circle.pfp(prop_tracker.get_value() % 1)))

        anim = AnimationGroup(
            prop_tracker.animate.increment_value(min_turns + random.random()),
            FadeIn(point),
            run_time=run_time
        )
        return point, anim

    def get_point(self, color, radius=0.05, glow_radius_factor=4):
        point = Group(
            TrueDot(radius=radius, color=color),
            GlowDot(radius=radius * glow_radius_factor, color=color),
            Circle(radius=radius).set_stroke(BLACK, 1),
        )
        return point

    def get_random_point(self, circle, color=YELLOW, **kwargs):
        return self.get_point(color, **kwargs).move_to(circle.pfp(random.random()))

    def get_random_chord(self, circle, color=YELLOW):
        p0 = self.get_random_point(circle, color)
        p1 = self.get_random_point(circle, color)
        line = Line(p0.get_center(), p1.get_center())
        line.set_stroke(color, 2)
        return Group(p0, p1, line)

    def draw_chord(self, chord_group):
        p0, p1, line = chord_group
        return AnimationGroup(
            ShowCreation(line),
            FadeIn(p0, scale=4),
            FadeIn(p1, scale=4),
        )

    def get_counter(self, tex, value_str="1"):
        value_str = tex[0]
        counter = Tex(tex)
        count_tracker = ValueTracker(int(value_str))
        count_dec = counter.make_number_changeable(value_str, edge_to_fix=UR)
        count_dec.add_updater(lambda m: m.set_value(count_tracker.get_value()))
        counter[-1].add_updater(lambda m: m.set_opacity(float(int(count_tracker.get_value()) != 1)))
        return counter, count_tracker

    def get_all_intersections(self, chords, circle, intersection_color=RED, intersection_radius=0.05):
        intersection_points = Group()

        for l1, l2 in it.combinations(chords, 2):
            try:
                intersection = line_intersection(l1.get_start_and_end(), l2.get_start_and_end())
            except Exception:
                intersection = np.array([np.inf, 0, 0])
            if get_norm(intersection - circle.get_center()) < circle.get_radius():
                int_point = self.get_point(intersection_color, radius=intersection_radius)
                int_point.move_to(intersection)
                intersection_points.add(int_point)
        return intersection_points

    def cycle_though_examples(
        self,
        circle,
        n_chords=10,
        n_cycles=20,
        wait_time=0.25,
        hue_range=(0.1, 0.2),
        chord_width=2,
        cloud=False
    ):
        for n in range(n_cycles):
            chords = VGroup(
                Line(
                    *[circle.pfp(random.random()) for x in range(2)],
                    color=random_bright_color(hue_range=hue_range)
                )
                for k in range(n_chords)
            )
            chords.set_stroke(width=chord_width)
            int_points = self.get_all_intersections(chords, circle)
            if cloud:
                point_cloud = DotCloud([dot.get_center() for dot in int_points])
                point_cloud.set_color(RED).set_radius(0.025)
                point_cloud.make_3d().set_shading(0.1, 0.1, 0.1)
                int_points = point_cloud
                chords.set_stroke(opacity=0.5)
            self.add(chords, int_points)
            self.wait(wait_time)
            self.remove(chords, int_points)


class StudentComplains(TeacherStudentsScene):
    def construct(self):
        # Test
        self.remove(self.background)
        stds = self.students
        morty = self.teacher
        self.play(
            stds[2].says("Hang on, hang on,\nhang on!", mode="angry", bubble_direction=LEFT, run_time=1),
            stds[0].change("erm", stds[2].eyes),
            stds[1].change("guilty", stds[2].eyes),
            morty.change("guilty"),
        )
        self.wait()
        self.play(
            FadeOut(stds[2].bubble),
            stds[2].says("Bertrand's\nParadox!", run_time=1, bubble_direction=LEFT),
            morty.change("tease"),
        )
        self.wait()
