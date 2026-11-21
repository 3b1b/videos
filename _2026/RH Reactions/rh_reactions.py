from manimlib import *
import textwrap


class Timeline(Group):
    def __init__(
        self, start_year, end_year, initial_year, *args,
        y=-2, center_month_width=2.4, edge_month_width=0.3, bulge_width=0.2, label_focus=0.11,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.start_year = start_year
        self.end_year = end_year
        self.y = y
        self.center_month_width = center_month_width
        self.edge_month_width = edge_month_width
        self.bulge_width = bulge_width
        self.label_focus = label_focus
        self.year_tracker = ValueTracker(initial_year)

        self.axis = Line(FRAME_WIDTH * LEFT / 2, FRAME_WIDTH * RIGHT / 2, stroke_width=2).set_y(self.y)

        month_names = [
            "Jan.", "Feb.", "Mar.", "Apr.", "May", "Jun.",
            "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."
        ]
        self.ticks = VGroup()
        self.month_labels = VGroup()
        for year in range(self.start_year, self.end_year):
            for index, name in enumerate(month_names):
                value = year + index / 12
                tick_size = 0.2 if index == 0 else 0.1
                tick = Line(tick_size * DOWN, tick_size * UP, stroke_width=2)
                tick.value = value
                self.ticks.add(tick)

                month_label = Text(f"{name} {year}", font_size=24)
                month_label.value = value
                month_label.base_height = month_label.get_height()
                self.month_labels.add(month_label)

        self.add(self.axis, self.ticks, self.month_labels)
        self.update_positions()
        self.add_updater(lambda tl: tl.update_positions())

        self.markers = Group()

    def warp(self, offset):
        edge_slope = 12 * self.edge_month_width
        bulge_slope = 12 * (self.center_month_width - self.edge_month_width)
        return edge_slope * offset + bulge_slope * self.bulge_width * np.tanh(offset / self.bulge_width)

    def n2p(self, value):
        return np.array([self.warp(value - self.year_tracker.get_value()), self.y, 0])

    def update_positions(self):
        for tick in self.ticks:
            tick.move_to(self.n2p(tick.value))
        for label in self.month_labels:
            point = self.n2p(label.value)
            focal_value = np.exp(-self.label_focus * point[0]**2)
            label.set_height(label.base_height * (0.6 + 0.4 * focal_value))
            label.next_to(point, DOWN, 0.3)
            label.set_fill(opacity=focal_value, color=interpolate_color(WHITE, YELLOW, focal_value**2))
        return self

    def center_on_year(self, year):
        self.year_tracker.set_value(year)
        return self.update_positions()

    def get_month_label(self, year, month):
        return self.month_labels[(year - self.start_year) * 12 + (month - 1)]

    def get_marker(self, marker_text, year, font_size=60, shift=0, fade_months=None):
        text = TexText(marker_text, font_size=font_size).next_to(self.n2p(year), UP).shift(UP * 2.5 + shift)
        ul = Underline(text)
        heading = VGroup(text, ul)

        dot = Dot(radius=0.12).move_to(self.n2p(year)).set_z_index(1)
        line = Line(dot.get_center(), ul.get_center(), buff=0, stroke_width=3)

        marker = Group(dot, line, heading)
        marker.create = AnimationGroup(
            FadeIn(dot),
            ShowCreation(line),
            Write(text),
            GrowFromPoint(ul, ul.get_left()), lag_ratio=0.2)
        marker.year_tracker = ValueTracker(year)
        marker.add_updater(lambda m: m.shift(self.n2p(m.year_tracker.get_value()) - dot.get_center()))

        if fade_months is not None:
            def update_marker_opacity(m):
                months_away = 12 * abs(m.year_tracker.get_value() - self.year_tracker.get_value())
                alpha = np.clip((fade_months + 1 - months_away) / 2, 0, 1)
                m.set_opacity(smooth(alpha))
            marker.add_updater(update_marker_opacity)

        self.markers.add(marker)

        marker.text = text
        marker.ul = ul
        marker.heading = heading
        marker.dot = dot
        marker.line = line

        return marker


def month_value(year, month):
    return year + (month - 1) / 12


class TimelineProgression(InteractiveScene):
    def construct(self):
        dates = [
            (2025, 6), (2025, 8), (2025, 8), (2026, 10), (2026, 5), (2026, 3),
            (2025, 8), (2025, 8), (2025, 8), (2026, 9), (2026, 6),
        ]

        # Start in June 2025
        timeline = Timeline(2020, 2031, month_value(*dates[0]))
        rh_marker = timeline.get_marker(
            R"Riemann Hypothesis \\ Solved",
            month_value(2026, 11),
            font_size=48,
            shift=DOWN * 0.8,
            fade_months=3
        )
        rh_marker.text.set_color(YELLOW)
        self.add(timeline, rh_marker)
        self.wait(1)

        # Hop from date to date
        for year, month in dates[1:]:
            self.play(timeline.year_tracker.animate.set_value(month_value(year, month)), run_time=2)
            self.wait(1)


class EndCard(InteractiveScene):
    def construct(self):
        paragraph_texts = [
            "On November 20, 2026, Levant Alpoge announced a proof of the Generalized Riemann Hypothesis, "
            "predominantly generated by AI. It is 1.3GB in lean, and Alpoge estimates that it would exceed 250k "
            "pages if translated to natural language.",
            "Shortly thereafter, Alessandro Mascullo posted about his own (purely human) proposed proof. "
            "He was planning for much more time to have it reviewed, but given the circumstances, "
            "posted the partial draft.",
            "Many thanks to Daria Ivanova for organizing and conducting most of these interviews, "
            "and kicking off the whole project."
        ]
        paragraphs = VGroup(*(
            Text("\n".join(textwrap.wrap(paragraph_text, 64)), font_size=30, alignment="LEFT")
            for paragraph_text in paragraph_texts
        ))
        paragraphs.arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        paragraphs.set_height(min(paragraphs.get_height(), FRAME_HEIGHT - 1))
        paragraphs.set_width(min(paragraphs.get_width(), FRAME_WIDTH - 1))
        paragraphs.center()

        def ambient_fade(paragraph):
            paragraph.fade_value *= 0.992
            paragraph.set_fill(opacity=paragraph.fade_value)

        # Write each paragraph, then let it drift away
        for paragraph in paragraphs:
            self.play(Write(paragraph), run_time=max(1, 0.015 * len(paragraph)))
            paragraph.fade_value = 1.0
            paragraph.add_updater(ambient_fade)
            self.wait(1)
        self.wait(10)