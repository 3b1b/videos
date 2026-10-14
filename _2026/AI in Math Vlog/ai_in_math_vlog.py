from manim_imports_ext import *
import glob

class QAResponses(InteractiveScene):
    BACKGROUND_COLOR = "#111111"

    MARGIN = 0.6
    MIN_WIDTH = 4.0
    MAX_WIDTH = 6
    GRID_JITTER_FRACTION = 0.4

    BORDER_COLOR = WHITE
    BORDER_WIDTH = 1

    GROW_TIME = 1.6
    BACKSWING_FRACTION = 0.4
    OVERSHOOT_SCALE = 1.06
    LAG_RATIO = 0.18

    PULSE_AMPLITUDE = 0.01
    PULSE_FREQ_RANGE = (0.25, 0.35)
    BREATHE_TIME = 7

    FADE_OUT_TOTAL_TIME = 2.5
    FADE_OUT_LAG_RATIO = 0.6

    def get_image_update_func(self, tiny_image, overshoot_image, final_image,
                               tiny_border, overshoot_border, final_border,
                               start_delay, rise_time, settle_time,
                               pulse_amplitude, pulse_freq):
        rise_end = start_delay + rise_time
        settle_end = rise_end + settle_time
        center = final_image.get_center()

        def update_func(mob, alpha, total_run_time):
            t = alpha * total_run_time
            image_mob, border_mob = mob.submobjects

            if t <= start_delay:
                image_mob.become(tiny_image)
                border_mob.become(tiny_border)
            elif t < rise_end:
                p = rush_from((t - start_delay) / rise_time)
                image_mob.interpolate(tiny_image, overshoot_image, p)
                border_mob.interpolate(tiny_border, overshoot_border, p)
            elif t < settle_end:
                p = smooth((t - rise_end) / settle_time)
                image_mob.interpolate(overshoot_image, final_image, p)
                border_mob.interpolate(overshoot_border, final_border, p)
            else:
                breathe_t = t - settle_end
                scale_factor = 1 / (1 + pulse_amplitude * np.sin(TAU * pulse_freq * breathe_t))
                for sub, final in [(image_mob, final_image), (border_mob, final_border)]:
                    sub.become(final)
                    sub.scale(scale_factor, about_point=center)

        return update_func

    def get_fade_out_updater(self, final_image, final_border, center,
                              initial_breathe_t, fade_start_delay, fade_time,
                              pulse_amplitude, pulse_freq):
        state = {"elapsed": 0.0}

        def updater(mob, dt):
            state["elapsed"] += dt
            elapsed = state["elapsed"]
            t = initial_breathe_t + elapsed
            image_mob, border_mob = mob.submobjects

            scale_factor = 1 / (1 + pulse_amplitude * np.sin(TAU * pulse_freq * t))

            if elapsed < fade_start_delay:
                opacity = 1.0
            else:
                opacity = 1.0 - min((elapsed - fade_start_delay) / fade_time, 1.0)

            image_mob.become(final_image)
            image_mob.scale(scale_factor, about_point=center)
            image_mob.set_opacity(opacity)

            border_mob.become(final_border)
            border_mob.scale(scale_factor, about_point=center)
            border_mob.set_stroke(opacity=opacity)

        return updater

    def construct(self):
        # Setup
        self.add(FullScreenRectangle(fill_color=self.BACKGROUND_COLOR, fill_opacity=1, stroke_width=0))

        images_dir = "q_and_a_responses"
        image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))

        images = [ImageMobject(path) for path in image_paths]
        random.shuffle(images)

        n = len(images)
        grid_width = FRAME_WIDTH - 2 * self.MARGIN
        grid_height = FRAME_HEIGHT - 2 * self.MARGIN
        aspect = FRAME_WIDTH / FRAME_HEIGHT
        n_cols = max(1, int(round(np.sqrt(n * aspect))))
        n_rows = max(1, int(np.ceil(n / n_cols)))
        cell_width = grid_width / n_cols
        cell_height = grid_height / n_rows

        cell_centers = []
        for row in range(n_rows):
            for col in range(n_cols):
                x = -grid_width / 2 + cell_width / 2 + col * cell_width
                y = grid_height / 2 - cell_height / 2 - row * cell_height
                cell_centers.append(np.array([x, y, 0.0]))
        random.shuffle(cell_centers)

        final_targets = []
        for img, cell_center in zip(images, cell_centers):
            img.set_width(random.uniform(self.MIN_WIDTH, self.MAX_WIDTH))

            border = Rectangle(width=img.get_width(), height=img.get_height())
            border.set_stroke(self.BORDER_COLOR, width=self.BORDER_WIDTH)
            border.set_fill(opacity=0)
            border.move_to(img.get_center())

            half_w = img.get_width() / 2
            half_h = img.get_height() / 2

            jitter_x = random.uniform(-cell_width * self.GRID_JITTER_FRACTION, cell_width * self.GRID_JITTER_FRACTION)
            jitter_y = random.uniform(-cell_height * self.GRID_JITTER_FRACTION, cell_height * self.GRID_JITTER_FRACTION)

            x_min = -FRAME_WIDTH / 2 + self.MARGIN + half_w
            x_max = FRAME_WIDTH / 2 - self.MARGIN - half_w
            y_min = -FRAME_HEIGHT / 2 + self.MARGIN + half_h
            y_max = FRAME_HEIGHT / 2 - self.MARGIN - half_h
            x_min, x_max = min(x_min, x_max), max(x_min, x_max)
            y_min, y_max = min(y_min, y_max), max(y_min, y_max)

            x = np.clip(cell_center[0] + jitter_x, x_min, x_max)
            y = np.clip(cell_center[1] + jitter_y, y_min, y_max)

            pos = np.array([x, y, 0.0])
            img.move_to(pos)
            border.move_to(pos)

            final_targets.append(Group(img, border))

        rise_time = self.GROW_TIME * (1 - self.BACKSWING_FRACTION)
        settle_time = self.GROW_TIME * self.BACKSWING_FRACTION

        max_delay = (n - 1) * self.LAG_RATIO * self.GROW_TIME
        total_run_time = max_delay + self.GROW_TIME + self.BREATHE_TIME

        # Add the images
        mobs = []
        pulse_freqs = []
        settle_ends = []
        animations = []
        for i, final_target in enumerate(final_targets):
            final_image, final_border = final_target.submobjects
            center = final_target.get_center()

            overshoot_image = final_image.copy().scale(self.OVERSHOOT_SCALE, about_point=center)
            overshoot_border = final_border.copy().scale(self.OVERSHOOT_SCALE, about_point=center)

            tiny_image = final_image.copy().scale(0.001, about_point=center).set_opacity(0)
            tiny_border = final_border.copy().scale(0.001, about_point=center).set_opacity(0)

            start_delay = i * self.LAG_RATIO * self.GROW_TIME
            settle_end = start_delay + rise_time + settle_time

            pulse_freq = random.uniform(*self.PULSE_FREQ_RANGE)

            pulse_freqs.append(pulse_freq)
            settle_ends.append(settle_end)

            mob = Group(tiny_image.copy(), tiny_border.copy())
            self.add(mob)
            mobs.append(mob)

            update_func = self.get_image_update_func(
                tiny_image, overshoot_image, final_image,
                tiny_border, overshoot_border, final_border,
                start_delay,
                rise_time,
                settle_time,
                self.PULSE_AMPLITUDE,
                pulse_freq,
            )
            animations.append(
                UpdateFromAlphaFunc(
                    mob,
                    lambda m, a, f=update_func: f(m, a, total_run_time),
                    run_time=total_run_time,
                    rate_func=linear,
                )
            )

        self.play(*animations)

        # Fade everything out in quick succession
        fade_time = self.FADE_OUT_TOTAL_TIME / (1 + (n - 1) * self.FADE_OUT_LAG_RATIO)

        fade_order = list(range(n))
        random.shuffle(fade_order)

        for fade_index, i in enumerate(fade_order):
            final_image, final_border = final_targets[i].submobjects
            center = final_targets[i].get_center()
            fade_start_delay = fade_index * self.FADE_OUT_LAG_RATIO * fade_time
            initial_breathe_t = total_run_time - settle_ends[i]

            mobs[i].add_updater(
                self.get_fade_out_updater(
                    final_image, final_border, center,
                    initial_breathe_t, fade_start_delay, fade_time,
                    self.PULSE_AMPLITUDE, pulse_freqs[i],
                )
            )

        self.wait(self.FADE_OUT_TOTAL_TIME)


class PositionVsVelocity(InteractiveScene):
    T_MAX = 4.0
    GROWTH_RATE = 0.7
    BASE_AMPLITUDE = 0.15
    PREDICTED_OFFSET = 1.2

    PANEL_WIDTH = 5.0
    PANEL_HEIGHT = 6.2

    POSITION_COLOR = BLUE
    VELOCITY_COLOR = GREEN

    RUN_TIME = 6

    def get_position_func(self, position_offset):
        def y(t):
            return self.BASE_AMPLITUDE * (np.exp(self.GROWTH_RATE * t) - 1) + position_offset
        return y

    def get_velocity_func(self):
        def v(t):
            return self.BASE_AMPLITUDE * self.GROWTH_RATE * np.exp(self.GROWTH_RATE * t)
        return v

    def get_panel(self, label_text, x_shift, y_max):
        axes = Axes(
            x_range=[0, self.T_MAX * 1.05, self.T_MAX],
            y_range=[0, y_max, y_max],
            width=self.PANEL_WIDTH,
            height=self.PANEL_HEIGHT,
            axis_config=dict(include_ticks=False, include_tip=False),
        )
        axes.move_to(x_shift * RIGHT + DOWN * 0.4)

        axis_labels = VGroup()

        title = Text(label_text, font_size=36)
        title.move_to(x_shift * RIGHT + UP * 3.2)

        return axes, axis_labels, title

    def construct(self):
        # Setup
        t_tracker = ValueTracker(0)
        v_func = self.get_velocity_func()

        y_max = (
            self.get_position_func(self.PREDICTED_OFFSET)(self.T_MAX)
            + v_func(self.T_MAX)
        )

        left_axes, left_labels, left_title = self.get_panel("Predicted Capability", -FRAME_WIDTH / 4, y_max)
        right_axes, right_labels, right_title = self.get_panel("Real Capability", FRAME_WIDTH / 4, y_max)

        divider = Line(UP * FRAME_HEIGHT / 2, DOWN * FRAME_HEIGHT / 2, stroke_color=GREY_C, stroke_width=1)

        self.add(left_axes, left_labels, left_title)
        self.add(right_axes, right_labels, right_title)
        self.add(divider)

        panels = [
            (left_axes, self.get_position_func(self.PREDICTED_OFFSET)),
            (right_axes, self.get_position_func(0)),
        ]

        for axes, y_func in panels:
            graph = always_redraw(lambda axes=axes, y_func=y_func: axes.get_graph(
                y_func,
                x_range=[0, max(t_tracker.get_value(), 1e-6)],
                color=self.POSITION_COLOR,
                stroke_width=4,
            ))
            position_arrow = always_redraw(lambda axes=axes, y_func=y_func: Arrow(
                axes.c2p(t_tracker.get_value(), 0),
                axes.c2p(t_tracker.get_value(), y_func(t_tracker.get_value())),
                buff=0,
                stroke_width=3,
            ).set_color(self.POSITION_COLOR))
            velocity_arrow = always_redraw(lambda axes=axes, y_func=y_func: Arrow(
                axes.c2p(t_tracker.get_value(), y_func(t_tracker.get_value())),
                axes.c2p(
                    t_tracker.get_value(),
                    y_func(t_tracker.get_value()) + v_func(t_tracker.get_value()),
                ),
                buff=0,
                stroke_width=3,
            ).set_color(self.VELOCITY_COLOR))
            glow_dot = always_redraw(lambda axes=axes, y_func=y_func: Group(
                TrueDot(
                    axes.c2p(t_tracker.get_value(), y_func(t_tracker.get_value())),
                    color=self.POSITION_COLOR,
                ),
                GlowDot(
                    axes.c2p(t_tracker.get_value(), y_func(t_tracker.get_value())),
                    color=self.POSITION_COLOR,
                ),
            ))

            position_label = always_redraw(lambda axes=axes, y_func=y_func: Tex(
                "x(t)", font_size=30,
            ).set_color(self.POSITION_COLOR).next_to(
                axes.c2p(t_tracker.get_value(), y_func(t_tracker.get_value()) / 2),
                LEFT, buff=0.15,
            ))
            velocity_label = always_redraw(lambda axes=axes, y_func=y_func: Tex(
                r"\frac{dx(t)}{dt}", font_size=30,
            ).set_color(self.VELOCITY_COLOR).next_to(
                axes.c2p(
                    t_tracker.get_value(),
                    y_func(t_tracker.get_value()) + v_func(t_tracker.get_value()) / 2,
                ),
                RIGHT, buff=0.15,
            ))

            self.add(graph, position_arrow, velocity_arrow, glow_dot, position_label, velocity_label)

        # Animate time advancing
        self.play(t_tracker.animate.set_value(self.T_MAX), run_time=self.RUN_TIME, rate_func=linear)
        self.wait()


class PositionVsVelocityV2(InteractiveScene):
    REALITY_COLOR = BLUE
    MEDIA_COLOR = YELLOW

    AXIS_Y = 2.2
    AXIS_LEFT = -5.5
    AXIS_RIGHT = 5.5
    REALITY_X = -1.0
    MEDIA_X = 4.0

    ARROW_START_X = -5.0
    PERCEIVED_ARROW_Y = -1.0
    REALITY_ARROW_Y = -3.0
    PERCEIVED_ARROW_LENGTH = 3.0
    REALITY_ARROW_LENGTH = 7.0

    def construct(self):
        # Divider separating the capability axis up top from the two arrows below
        divider = Line(LEFT * FRAME_WIDTH / 2, RIGHT * FRAME_WIDTH / 2, stroke_color=GREY_C, stroke_width=1)
        self.add(divider)

        # A single axis representing "AI capability in math," increasing to the right
        axis = Arrow(
            np.array([self.AXIS_LEFT, self.AXIS_Y, 0.0]),
            np.array([self.AXIS_RIGHT, self.AXIS_Y, 0.0]),
            buff=0, stroke_width=3,
        ).set_color(GREY_B)
        axis_label = Text("AI Capability in Math", font_size=50)
        axis_label.to_edge(UP, buff=0.4)
        self.play(GrowArrow(axis), Write(axis_label))

        # Mark where reality currently sits on that axis
        reality_dot = GlowDot(np.array([self.REALITY_X, self.AXIS_Y, 0.0]), color=self.REALITY_COLOR)
        reality_label = Text("Reality", font_size=32, color=self.REALITY_COLOR)
        reality_label.next_to(reality_dot, DOWN, buff=0.3)
        self.play(FadeIn(reality_dot), Write(reality_label))

        # Mark where media portrayal sits -- further along, ahead of reality
        media_dot = GlowDot(np.array([self.MEDIA_X, self.AXIS_Y, 0.0]), color=self.MEDIA_COLOR)
        media_label = Text("Portrayed in Media", font_size=32, color=self.MEDIA_COLOR)
        media_label.next_to(media_dot, DOWN, buff=0.3)
        self.play(FadeIn(media_dot), Write(media_label))

        self.wait()

        # Below: two arrows showing how long each narrative implies progress takes
        perceived_arrow = Arrow(
            np.array([self.ARROW_START_X, self.PERCEIVED_ARROW_Y, 0.0]),
            np.array([self.ARROW_START_X + self.PERCEIVED_ARROW_LENGTH, self.PERCEIVED_ARROW_Y, 0.0]),
            buff=0,
            thickness=5
        ).set_color(self.MEDIA_COLOR)

        reality_arrow = Arrow(
            np.array([self.ARROW_START_X, self.REALITY_ARROW_Y, 0.0]),
            np.array([self.ARROW_START_X + self.REALITY_ARROW_LENGTH, self.REALITY_ARROW_Y, 0.0]),
            buff=0,
            thickness=5
        ).set_color(self.REALITY_COLOR)
        VGroup(perceived_arrow, reality_arrow).set_x(0)
        perceived_label = Text("Perceived", font_size=28, color=self.MEDIA_COLOR)
        perceived_label.next_to(perceived_arrow, UP, buff=0.2)
        reality_arrow_label = Text("Reality", font_size=28, color=self.REALITY_COLOR)
        reality_arrow_label.next_to(reality_arrow, UP, buff=0.2)

        self.play(GrowArrow(perceived_arrow), Write(perceived_label))
        self.play(GrowArrow(reality_arrow), Write(reality_arrow_label))

        self.wait(2)