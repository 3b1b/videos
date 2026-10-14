from manim_imports_ext import *
import glob

class QAResponses(InteractiveScene):
    BACKGROUND_COLOR = "#111111"

    MARGIN = 0.6
    MIN_WIDTH = 4.0
    MAX_WIDTH = 6
    MAX_ROTATION = 10 * DEGREES
    GRID_JITTER_FRACTION = 0.4

    BORDER_COLOR = WHITE
    BORDER_WIDTH = 1

    GROW_TIME = 1.6
    BACKSWING_FRACTION = 0.4
    OVERSHOOT_SCALE = 1.06
    LAG_RATIO = 0.18

    PULSE_AMPLITUDE = 0.01
    PULSE_FREQ_RANGE = (0.25, 0.35)
    ROTATE_AMPLITUDE = 2 * DEGREES
    ROTATE_FREQ_RANGE = (0.1, 0.2)
    ROTATE_RAMP_TIME = 1.5
    BREATHE_TIME = 7

    FADE_OUT_TOTAL_TIME = 2
    FADE_OUT_LAG_RATIO = 0.2

    def get_image_update_func(self, tiny_image, overshoot_image, final_image,
                               tiny_border, overshoot_border, final_border,
                               start_delay, rise_time, settle_time,
                               pulse_amplitude, pulse_freq,
                               rotate_amplitude, rotate_freq, rotate_phase):
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
                ramp = smooth(min(breathe_t / self.ROTATE_RAMP_TIME, 1.0))
                rotate_angle = rotate_amplitude * ramp * np.sin(TAU * rotate_freq * breathe_t + rotate_phase)
                for sub, final in [(image_mob, final_image), (border_mob, final_border)]:
                    sub.become(final)
                    sub.scale(scale_factor, about_point=center)
                    sub.rotate(rotate_angle, about_point=center)

        return update_func

    def get_fade_out_updater(self, final_image, final_border, center,
                              initial_breathe_t, fade_start_delay, fade_time,
                              pulse_amplitude, pulse_freq,
                              rotate_amplitude, rotate_freq, rotate_phase):
        state = {"elapsed": 0.0}

        def updater(mob, dt):
            state["elapsed"] += dt
            elapsed = state["elapsed"]
            t = initial_breathe_t + elapsed
            image_mob, border_mob = mob.submobjects

            scale_factor = 1 / (1 + pulse_amplitude * np.sin(TAU * pulse_freq * t))
            ramp = smooth(min(t / self.ROTATE_RAMP_TIME, 1.0))
            rotate_angle = rotate_amplitude * ramp * np.sin(TAU * rotate_freq * t + rotate_phase)

            if elapsed < fade_start_delay:
                opacity = 1.0
            else:
                opacity = 1.0 - min((elapsed - fade_start_delay) / fade_time, 1.0)

            image_mob.become(final_image)
            image_mob.scale(scale_factor, about_point=center)
            image_mob.rotate(rotate_angle, about_point=center)
            image_mob.set_opacity(opacity)

            border_mob.become(final_border)
            border_mob.scale(scale_factor, about_point=center)
            border_mob.rotate(rotate_angle, about_point=center)
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

            angle = random.uniform(-self.MAX_ROTATION, self.MAX_ROTATION)
            img.rotate(angle)
            border.rotate(angle)

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
        rotate_freqs = []
        rotate_phases = []
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
            rotate_freq = random.uniform(*self.ROTATE_FREQ_RANGE)
            rotate_phase = random.uniform(0, TAU)

            pulse_freqs.append(pulse_freq)
            rotate_freqs.append(rotate_freq)
            rotate_phases.append(rotate_phase)
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
                self.ROTATE_AMPLITUDE,
                rotate_freq,
                rotate_phase,
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
                    self.ROTATE_AMPLITUDE, rotate_freqs[i], rotate_phases[i],
                )
            )

        self.wait(self.FADE_OUT_TOTAL_TIME)