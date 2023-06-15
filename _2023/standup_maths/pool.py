from manim_imports_ext import *


class PoolTableReflections(InteractiveScene):
    def construct(self):
        # Add  table
        table = ImageMobject("pool_table")
        table.set_height(4.0)
        buff = 0.475

        ball = TrueDot(radius=0.1)
        ball.set_color(GREY_A)
        ball.set_shading(1, 1, 1)
        ball.move_to(table)

        self.add(table, ball)

        # Show inner table
        frame = self.frame
        frame.set_height(6)

        irt = Rectangle(  # Inner rect template
            width=table.get_width() - 2 * buff,
            height=table.get_height() - 2 * buff,
        )
        irt.move_to(table)
        inner_rect = VMobject()
        inner_rect.start_new_path(irt.get_right())
        for corner in [UR, UL, DL, DR]:
            inner_rect.add_line_to(irt.get_corner(corner))
        inner_rect.add_line_to(irt.get_right())
        inner_rect.set_stroke(RED, 3)
        inner_rect.insert_n_curves(20)

        self.play(ball.animate.move_to(inner_rect.get_start()))
        self.play(
            ShowCreation(inner_rect, run_time=4),
            UpdateFromFunc(ball, lambda m: m.move_to(inner_rect.get_end()))
        )
        self.wait()

        # Show reflections
        table_group1 = Group(table, inner_rect, ball)
        new_origin = inner_rect.get_corner(UR)

        self.play(
            frame.animate.set_height(8).move_to(new_origin),
            table.animate.set_opacity(0.5),
            ball.animate.shift(DL)
        )
        table_group2 = table_group1.copy()
        table_group2[2].set_opacity(0.5)
        table_group2[0].set_opacity(0.1)
        self.play(
            Rotate(table_group2, axis=UP, about_point=inner_rect.get_right()),
            run_time=2
        )
        self.wait()

        table_group3 = table_group2.copy()
        self.play(
            Rotate(table_group3, axis=LEFT, about_point=table_group2[1].get_top()),
            run_time=2
        )
        self.wait()

        def get_reflection(point, dims=[0]):
            vect = point - new_origin
            for dim in dims:
                vect[dim] *= -1
            return new_origin + vect

        table_group2[2].add_updater(lambda m: m.move_to(get_reflection(ball.get_center())))
        table_group3[2].add_updater(lambda m: m.move_to(get_reflection(ball.get_center(), dims=[0, 1])))

        # Move ball around
        kw = dict(rate_func=there_and_back, run_time=4)
        self.play(ball.animate.shift(2 * UP), **kw)
        self.play(ball.animate.shift(2 * LEFT), **kw)
        self.wait()

        # Show a trajectory
        def line_to_trajectory(line, n_reflections=2):
            p0 = find_intersection(
                line.get_start(), line.get_vector(),
                new_origin, DOWN,
            )
            p1 = find_intersection(
                line.get_start(), line.get_vector(),
                new_origin, RIGHT,
            )

            trajectory = VMobject()
            if n_reflections == 1:
                trajectory.set_points_as_corners([
                    line.get_start(),
                    p1,
                    get_reflection(line.get_end(), [1])
                ])
            else:
                trajectory.set_points_as_corners([
                    line.get_start(), p0,
                    get_reflection(p1),
                    get_reflection(line.get_end(), [0, 1])
                ])
            trajectory.match_style(line)
            trajectory.insert_n_curves(100)

            return trajectory

        ur_corner = table_group3[1].get_corner(UR)
        straight_line = Line(ball.get_center(), ur_corner)
        straight_line.set_stroke(YELLOW, 2)
        traj_one_ref = line_to_trajectory(straight_line, 1)
        trajectory = line_to_trajectory(straight_line)

        self.play(
            ShowCreation(trajectory),
            UpdateFromFunc(ball, lambda m: m.move_to(trajectory.get_end())),
            run_time=4,
        )
        self.wait()
        self.play(
            TransformFromCopy(trajectory, traj_one_ref),
            trajectory.animate.set_stroke(opacity=0.5),
        )
        self.wait()
        self.play(
            MoveAlongPath(ball, trajectory),
            run_time=4,
        )
        self.wait()
        self.play(
            TransformFromCopy(traj_one_ref, straight_line),
            traj_one_ref.animate.set_stroke(opacity=0.5),
        )
        self.wait()
        self.play(
            MoveAlongPath(ball, trajectory),
            run_time=4,
        )
        self.wait()
        self.play(
            FadeOut(traj_one_ref),
            trajectory.animate.set_stroke(opacity=1)
        )

        # Alternate line
        alt_line = Line(straight_line.get_start(), straight_line.get_end() + 2 * DOWN)
        alt_line.match_style(straight_line)

        self.play(
            Transform(straight_line, alt_line),
            UpdateFromFunc(trajectory, lambda m: m.become(
                line_to_trajectory(straight_line)
            )),
            UpdateFromFunc(trajectory, lambda m: m.become(
                line_to_trajectory(straight_line)
            )),
            UpdateFromFunc(ball, lambda m: m.move_to(trajectory.get_end())),
            run_time=4,
            rate_func=there_and_back
        )
        self.wait()
        self.play(*map(FadeOut, [straight_line, trajectory]))

        # Diamond points
        vect1 = np.array([0.1, 0.315, 0.])
        vect2 = np.array([0.309, 0.1, 0.])
        low_diamond_points = np.linspace(
            inner_rect.get_corner(DL) + vect1 * [-1, -1, 0],
            inner_rect.get_corner(DR) + vect1 * [1, -1, 0],
            9,
        )
        side_diamond_points = np.linspace(
            inner_rect.get_corner(DR) + vect2 * [1, -1, 0],
            inner_rect.get_corner(UR) + vect2,
            5,
        )

        low_dots = GlowDots(low_diamond_points, radius=0.1)
        side_dots = GlowDots(side_diamond_points, radius=0.1)

        low_labels = VGroup(*(
            Integer(value, font_size=24).next_to(point, DOWN)
            for value, point in zip(
                range(80, -10, -10),
                low_diamond_points,
            )
        ))
        side_labels = VGroup(*(
            Integer(value, font_size=24).next_to(point, RIGHT)
            for value, point in zip(
                range(90, 0, -20),
                side_diamond_points,
            )
        ))

        low_nl = NumberLine((0, 80))
        low_nl.put_start_and_end_on(low_diamond_points[-1], low_diamond_points[0])

        side_nl = NumberLine((10, 90))
        side_nl.put_start_and_end_on(side_diamond_points[-1], side_diamond_points[0])

        self.play(
            FadeIn(low_dots, lag_ratio=0.5),
            FadeIn(low_labels, lag_ratio=0.5),
            frame.animate.set_height(10),
            *(
                tg[1].animate.set_stroke(width=1)
                for tg in [table_group1, table_group2, table_group3]
            ),
            run_time=2,
        )
        self.play(
            FadeIn(side_dots, lag_ratio=0.5),
            FadeIn(side_labels, lag_ratio=0.5),
        )
        self.wait()

        # Show the trick lines
        def get_diamond_line(x, y, length=25):
            line = Line(low_nl.n2p(x), side_nl.n2p(y))
            line.scale(length / line.get_length(), about_point=line.get_start())
            line.set_stroke(YELLOW, 2)
            return line

        def n_to_lines(n):
            return VGroup(*(
                get_diamond_line(x, n - x)
                for x in range(70, 0, -10)
            )).set_stroke(YELLOW)

        lines = n_to_lines(80)
        trajs = VGroup()
        for line in lines:
            traj = line_to_trajectory(line)
            traj.line = line
            traj.add_updater(lambda t: t.become(line_to_trajectory(
                t.line
            )).set_color(BLUE))
            trajs.add(traj)

        self.play(LaggedStartMap(FadeIn, lines, lag_ratio=0.75, run_time=4))
        self.wait()
        self.play(
            frame.animate.set_height(14, about_edge=DL),
            run_time=2,
        )
        self.wait()

        trajs.suspend_updating()
        for n in range(len(lines)):
            self.play(
                lines[:n].animate.set_stroke(opacity=0.2),
                lines[n].animate.set_stroke(opacity=1),
                lines[n + 1:].animate.set_stroke(opacity=0.2),
                FadeIn(trajs[n]),
                trajs[:n].animate.set_stroke(opacity=0.2),
                run_time=0.5,
            )
        self.play(
            lines.animate.set_stroke(opacity=1),
            trajs.animate.set_stroke(opacity=1),
        )
        trajs.resume_updating()
        self.wait()

        # Show just one line
        line = lines[-1]
        traj = trajs[-1]

        self.play(
            lines[:-1].animate.set_stroke(opacity=0.15),
            trajs[:-1].animate.set_stroke(opacity=0.15),
            frame.animate.set_height(12, about_edge=DL).move_to(new_origin),
        )
        self.play(
            UpdateFromAlphaFunc(
                line, lambda m, a: m.set_points(get_diamond_line(
                    interpolate(10, 70, there_and_back(a)),
                    interpolate(70, 10, there_and_back(a)),
                ).get_points()),
            ),
            run_time=8,
        )
        self.wait()

        self.play(
            lines.animate.set_stroke(opacity=1),
            trajs.animate.set_stroke(opacity=1),
        )

        # Transition between different n
        for n in [70, 60, 90, 100, 80]:
            self.play(
                Transform(lines, n_to_lines(n)),
                run_time=3
            )
            self.wait()

        # Show outward rays from the corner
        def get_clean_lines(point):
            result = VGroup(*(
                Line(p, point)
                for p in low_diamond_points[1:-1]
            ))
            result.set_stroke(YELLOW, 2)
            return result

        lines.save_state()
        for point in [ur_corner, ur_corner + DOWN, ur_corner, ur_corner + 2 * LEFT, ur_corner]:
            self.play(
                Transform(lines, get_clean_lines(point)),
                run_time=2
            )
            self.wait()
        self.play(Restore(lines))
        self.wait()
