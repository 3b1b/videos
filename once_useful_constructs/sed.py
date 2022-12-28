from manim_imports_ext import *

import csv
import time
from datetime import datetime


class SEDTest(MovingCameraScene):
    def construct(self):
        file_name1 = "/Users/grant/Desktop/SED_launch_data.csv"
        file_name2 = "/Users/grant/Desktop/SED_scrub_data.csv"

        times = []
        heart_rates = []
        with open(file_name1, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                try:
                    values = row[0].split(",")
                    timestamp = str(values[8])
                    heart_rate = int(values[10])
                    dt = datetime.fromisoformat(timestamp)
                    curr_time = time.mktime(dt.timetuple())

                    times.append(curr_time)
                    heart_rates.append(heart_rate)
                except ValueError:
                    continue

        times = np.array(times)
        times -= times[0]
        heart_rates = np.array(heart_rates)

        average_over = 100
        hr_averages = np.array([
            np.mean(heart_rates[i:i + average_over])
            for i in range(len(heart_rates) - average_over)
        ])
        prop = 10
        shown_times = times[::prop]
        # shown_heart_rates = heart_rates[::prop]
        shown_heart_rates = hr_averages[::prop]

        min_time = np.min(times)
        max_time = np.max(times)
        min_HR = np.min(heart_rates)
        max_HR = np.max(heart_rates)

        axes = Axes(
            x_min=-1,
            x_max=12,
            y_min=0,
            y_max=130,
            y_axis_config={
                "unit_size": 1.0 / 25,
                "tick_frequency": 10,
            }
        )
        axes.to_corner(UL)
        axes.set_stroke(width=2)

        def c2p(t, h):
            t_coord = t / 20  # 20 minute intervals
            return axes.coords_to_point(t_coord, h)

        # x_axis_labels = VGroup()
        # for t in range(0, 190, 60):
        #     point = c2p(t, 0)
        #     label = Integer(t)
        #     label.next_to(point, DOWN, MED_SMALL_BUFF)
        #     x_axis_labels.add(label)
        # axes.x_axis.add(x_axis_labels)
        # x_label = OldTexText("Time (minutes)")
        # x_label.next_to(axes.x_axis, UP, SMALL_BUFF)
        # x_label.to_edge(RIGHT)
        # axes.x_axis.add(x_label)

        y_axis_labels = VGroup()
        for y in range(50, 150, 50):
            point = axes.coords_to_point(0, y)
            label = Integer(y)
            label.next_to(point, LEFT)
            y_axis_labels.add(label)
        axes.y_axis.add(y_axis_labels)
        y_label = OldTexText("Heart rates")
        y_label.next_to(axes.y_axis, RIGHT, aligned_edge=UP)
        axes.y_axis.add(y_label)

        def point_to_color(point):
            hr = axes.y_axis.point_to_number(point)
            ratio = (hr - 50) / (120 - 50)
            if ratio < 0.5:
                return interpolate_color(BLUE_D, GREEN, 2 * ratio)
            else:
                return interpolate_color(GREEN, RED, 2 * ratio - 1)

        def get_v_line(t, label=None, **kwargs):
            line = DashedLine(c2p(t, 0), c2p(t, 120), **kwargs)
            line.set_stroke(width=2)
            if label is not None:
                label_mob = OldTexText(label)
                label_mob.next_to(line, UP)
                label_mob.set_color(WHITE)
                line.label = label_mob
            return line

        points = []
        for t, hr in zip(shown_times, shown_heart_rates):
            points.append(c2p(t / 60, hr))
        lines = VGroup()
        for p1, p2, p3, p4 in zip(points, points[1:], points[2:], points[3:]):
            line = Line(p1, p2)
            line.set_points_smoothly([p1, p2, p3, p4])
            line.set_color((
                point_to_color(p1),
                point_to_color(p2),
            ))
            line.set_sheen_direction(line.get_vector())
            lines.add(line)
        lines.set_stroke(width=2)

        def get_lines_after_value(lines, t):
            result = VGroup()
            for line in lines:
                line_t = axes.x_axis.point_to_number(line.get_center())
                line_t *= 20
                if t - 7 < line_t < t + 7:
                    result.add(line)

                # if t < line_t:
                #     alpha = 1 - smooth(abs(line_t - t) / 20)
                # elif line_t > t - 5:
                #     alpha = 1 - (t - line_t) / 5
                # line.set_stroke(
                #     width=interpolate(2, 10, alpha),
                #     opacity=interpolate(0.5, 1, alpha),
                # )
            return result

        # base_line = Line(
        #     c2p(0, 55), c2p(180, 55),
        # )
        # base_line_label = OldTexText(
        #     "(Felipe's resting HR)"
        # )
        # base_line_label.next_to(base_line, DOWN)
        # base_line_label.to_edge(RIGHT)


        # 22:22, yt launch time
        # 18:32, T-minus 4
        # 28:22, separation

        launch_time = 116

        for mark in axes.x_axis.tick_marks:
            mark.shift(c2p(0, 0) - c2p(4, 0))
            if mark.get_center()[0] < c2p(0, 0)[0]:
                mark.fade(1)

        times_and_words = [
            (launch_time - 60, "T-minus\\\\1 hour"),
            (launch_time, "Launch!"),
            (launch_time + 60, "1 hour \\\\ into flight"),
        ]
        time_labels = VGroup()
        for t, words in times_and_words:
            point = c2p(t, 0)
            tick = Line(DOWN, UP)
            tick.set_height(0.5)
            tick.move_to(point)
            label = OldTexText(words)
            label.next_to(tick, DOWN)
            time_labels.add(VGroup(tick, label))

        tm4_time = launch_time - 4
        tm4_line = get_v_line(tm4_time, "T-minus 4 (aka Game on!)")
        tm4_line.label.shift(2 * RIGHT)
        # self.add(tm4_line)

        sep1_time = launch_time + 5 + (35 / 60)
        sep1_line = get_v_line(sep1_time, "First stage separation")

        sep2_time = launch_time + 37  # Second stage separation
        sep2_line = get_v_line(sep2_time, "Second stage separation")

        sep3_time = launch_time + 43
        sep3_line = get_v_line(sep3_time, "Third stage (probe) separation")

        frame = self.camera.frame

        from _2017.eoc.uncertainty import FalconHeavy
        rocket = FalconHeavy()
        rocket.logo.set_fill(WHITE, opacity=0).scale(0)
        rocket.set_height(1)
        rocket.move_to(c2p(launch_time, 0))

        time_line_pairs = [
            (tm4_time, tm4_line),
            (sep1_time, sep1_line),
            (sep2_time, sep2_line),
            (sep3_time, sep3_line),
        ]

        # Introduce
        self.play(Write(axes), run_time=1)
        self.play(
            LaggedStartMap(FadeInFromLarge, lines, run_time=5, lag_ratio=0.1)
        )
        self.play(LaggedStartMap(FadeInFromDown, time_labels))
        self.wait()

        # Point indications
        curr_line = get_v_line(launch_time)
        last_label = VectorizedPoint()
        self.play(
            ShowCreation(curr_line),
            rocket.to_edge, UP,
            UpdateFromAlphaFunc(
                rocket,
                lambda r, a: r.set_fill(
                    opacity=min(2 - 2 * a, 1),
                ),
                remover=True
            ),
            run_time=2
        )
        for t, line in time_line_pairs:
            post_t_lines = get_lines_after_value(lines, t).copy()
            flicker = LaggedStartMap(
                UpdateFromAlphaFunc, post_t_lines,
                lambda mob: (
                    mob,
                    lambda l, a: l.set_stroke(
                        width=interpolate(2, 10, a),
                        opacity=interpolate(0.5, 1, a),
                    )
                ),
                rate_func=there_and_back,
                run_time=1.5
            )
            self.play(
                flicker,
                lines.set_stroke, {"opacity": 0.5},
                # ApplyFunction(
                #     lambda m: emphasize_lines_near_value(m, t),
                #     lines,
                #     run_time=1,
                #     lag_ratio=0.5,
                # ),
                Transform(curr_line, line),
                FadeInFromDown(line.label),
                FadeOut(last_label, DOWN),
            )
            for x in range(3):
                self.play(flicker)
            last_label = line.label
            # self.show_frame()

        # T -4 spike
        # Separation of arrays
        # Deployment of arrays
        # Indication of power positive
