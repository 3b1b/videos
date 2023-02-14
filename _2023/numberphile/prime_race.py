from manim_imports_ext import *
import sympy


class PrimeRace(InteractiveScene):
    race_length = 26863

    def construct(self):
        ONE_COLOR = BLUE
        THREE_COLOR = RED
        colors = [ONE_COLOR, THREE_COLOR]

        # Labels
        labels = VGroup(
            Tex(R"p \equiv 1 \mod 4", t2c={"1": ONE_COLOR}),
            Tex(R"p \equiv 3 \mod 4", t2c={"3": THREE_COLOR}),
        )
        labels.arrange(DOWN, buff=0.75)
        labels.to_edge(LEFT)

        h_line = Line(LEFT, RIGHT)
        h_line.set_width(100)
        h_line.to_edge(LEFT)

        v_line = Line(UP, DOWN)
        v_line.set_height(1.5 * labels.get_height())
        v_line.next_to(labels, RIGHT)
        v_line.set_y(0)
        VGroup(h_line, v_line).set_stroke(width=2)

        self.add(h_line, v_line)
        self.add(labels)

        # Start the race
        primes: list[int] = list(sympy.primerange(3, self.race_length))
        team1 = VGroup()
        team3 = VGroup()
        teams = [team1, team3]
        blocks = []
        for prime in primes:
            index = int((prime % 4) == 3)
            square = Square(side_length=1)
            square.set_fill(colors[index], 0.5)
            square.set_stroke(colors[index], 1.0)
            p_mob = Integer(prime)
            p_mob.set_max_width(0.8 * square.get_width())
            block = VGroup(square, p_mob)

            teams[index].add(block)
            blocks.append(block)

        for team, label in zip(teams, labels):
            team.arrange(RIGHT, buff=0)
            team.next_to(v_line, RIGHT, buff=SMALL_BUFF)
            team.match_y(label)

        h_line.set_width(teams[1].get_width() + 10, about_edge=LEFT)

        for block in blocks[:10]:
            self.play(FadeIn(block[0]), Write(block[1]))

        # Next sets
        frame = self.frame
        frame.target = frame.generate_target()
        frame.target.scale(1.75, about_edge=LEFT)
        self.play(
            LaggedStartMap(
                FadeIn, VGroup(*blocks[10:30]),
                lag_ratio=0.9,
            ),
            MoveToTarget(frame, rate_func=rush_into),
            run_time=12,
        )

        # Last set
        curr = 30
        tups = [
            (200, 10, linear, 1.25),
            (len(blocks) - 100, 60, linear, 1.25),
            (len(blocks) - 1, 5, smooth, 0.8)
        ]

        for index, rt, func, sf in tups:
            frame.target = frame.generate_target()
            frame.target.scale(sf)
            frame.target.set_x(blocks[index].get_right()[0] + 1)
            self.play(
                ShowIncreasingSubsets(VGroup(*blocks[curr:index])),
                MoveToTarget(frame),
                run_time=rt,
                rate_func=func,
            )
            curr = index

        blocks = VGroup(*blocks)
        self.add(blocks)
        self.play(frame.animate.set_height(8, about_point=blocks.get_right() + 2 * LEFT), run_time=3)
        self.wait()


class RaceGraph(InteractiveScene):
    race_length = 26863
    y_range = (-4, 30, 2)

    def construct(self):
        # Compute differences
        primes: list[int] = list(sympy.primerange(3, self.race_length))
        diffs = [0]
        colors = [WHITE]
        y_max = self.y_range[1]
        for p in primes:
            diff = diffs[-1] + (p % 4 - 2)
            diffs.append(diff)
            if diff < 0:
                colors.append(BLUE)
            elif diff < y_max / 2:
                colors.append(interpolate_color(WHITE, RED, clip(2 * diff / y_max, 0, 1)))
            else:
                colors.append(interpolate_color(RED, RED_E, clip(2 * (diff - y_max / 2) / y_max, 0, 1)))

        # Axes and graph
        x_unit = 10
        axes = Axes(
            x_range=(0, len(primes) + 10, x_unit),
            y_range=self.y_range,
            width=len(primes) / x_unit / 1.6,
            height=7,
            axis_config=dict(tick_size=0.05),
        )
        axes.to_edge(LEFT, buff=0.8)

        y_label = TexText(R"\#Team3 $-$ \#Team1")
        y_label["Team3"].set_color(RED)
        y_label["Team1"].set_color(BLUE)
        y_label.next_to(axes.y_axis.get_top(), RIGHT)
        y_label.fix_in_frame()

        axes.y_axis.add_numbers(font_size=14, buff=0.15)

        graph = VMobject()
        graph.set_points_as_corners([
            axes.c2p(i, diff)
            for i, diff in enumerate(diffs)
        ])
        graph.set_stroke(colors, width=3)

        self.add(axes)
        self.add(y_label)
        self.add(graph)

        # Set blocking rectangle
        rect = FullScreenRectangle()
        rect.set_fill(BLACK, 1)
        rect.set_stroke(BLACK, 0)
        rect.match_x(axes.c2p(0, 0), LEFT)

        def set_x_shift(x, anims=[], **kwargs):
            self.play(
                rect.animate.match_x(axes.c2p(x, 0), LEFT),
                *anims,
                **kwargs
            )

        self.clear()
        self.add(graph, rect, axes.x_axis, axes.y_axis, y_label)

        # First blocks, total runtime should be 12
        frame = self.frame
        frame.save_state()
        zoom_point = axes.c2p(0, 0)
        frame.set_height(3, about_point=zoom_point)

        for x in range(10):
            set_x_shift(x + 1)

        for x in range(10, 30):
            set_x_shift(x + 1, run_time = 6 / 20)
            self.wait(6 / 20)

        # Next block
        set_x_shift(
            200,
            anims=[Restore(frame)],
            run_time=10, rate_func=linear
        )

        # Squish the graph
        full_width = get_norm(axes.c2p(200, 0) - axes.c2p(0, 0))
        origin = axes.c2p(0, 0)

        prime_label = Integer(primes[200])
        prime_label.next_to(rect, LEFT, buff=0.7)
        prime_label.to_edge(DOWN, buff=0.3)
        prime_label.fix_in_frame()
        self.add(prime_label)

        def set_x_squish(x1, x2, **kwargs):
            group = VGroup(axes.x_axis, graph)
            self.add(group, rect)

            self.play(
                UpdateFromAlphaFunc(
                    group,
                    lambda m, a: m.stretch(
                        full_width / get_norm(axes.c2p(interpolate(x1, x2, a), 0) - origin),
                        0,
                        about_point=origin,
                    ),
                ),
                UpdateFromAlphaFunc(
                    prime_label,
                    lambda m, a: m.set_value(
                        primes[min(int(interpolate(x1, x2, a)), len(primes) - 1)]
                    )
                ),
                **kwargs
            )

        set_x_squish(200, len(primes) - 100, rate_func=linear, run_time=60)
        set_x_squish(len(primes) - 100, len(primes), rate_func=smooth, run_time=5)

        # Reset frame
        self.play(
            VGroup(axes, graph).animate.set_width(
                len(primes) / 10, about_point=axes.c2p(len(primes), 0), stretch=True
            ),
            FadeOut(rect),
            run_time=6,
        )
        self.wait()



class LongRaceGraph(RaceGraph):
    race_length = 650000
    y_range = (-10, 100, 5)
