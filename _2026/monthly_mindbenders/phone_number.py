from manim_imports_ext import *
from decimal import Context, Decimal
from scipy.spatial.transform import Rotation, Slerp


class BigInteger(Integer):
    # Integer formats its value through np.round, which overflows past ~19 digits.
    # Formatting the Python int directly keeps arbitrarily large values exact.
    def get_num_string(self, number) -> str:
        return self.get_formatter().format(int(number)).replace("-", "–")


class PhoneNumberScene(InteractiveScene):
    def __init__(self, n=1234567890, k=10, **kwargs):
        self.n = n
        self.k = k
        super().__init__(**kwargs)

    def setup(self):
        super().setup()
        phone_number = Integer(self.n, group_with_commas=False, text_config={"font": "Consolas"}).set_color(BLUE_D)
        self.multiple_value = ValueTracker(2)
        multiple = BigInteger(0, group_with_commas=False, text_config={"font": "Consolas"}).set_color(BLUE_B)
        multiple.add_updater(
            lambda m: m.set_value(int(self.multiple_value.get_value()))
            .next_to(phone_number, DOWN)
            .align_to(phone_number, RIGHT)
        )
        times_symbol = Tex(R"\times").next_to(phone_number, LEFT, buff=0.3).match_y(multiple)
        equals_symbol = Line().set_width(VGroup(times_symbol, multiple).get_width() * 1.1).next_to(VGroup(times_symbol, multiple), DOWN)
        result = BigInteger(0, group_with_commas=False, text_config={"font": "Consolas"}).next_to(equals_symbol, DOWN).set_color(YELLOW)
        result.add_updater(
            lambda r: r.set_value(self.n * int(self.multiple_value.get_value())).align_to(phone_number, RIGHT)
        )
        multiplication_group = VGroup(phone_number, times_symbol, multiple, equals_symbol, result)
        multiplication_group.scale(1.1).center()
        self.multiplication_group = multiplication_group

    def play_setup(self, start_with_phone_number_format=False):
        phone_number, times_symbol, multiple, equals_symbol, result = self.multiplication_group
        digit_positions = [d.get_x() for d in phone_number]

        if start_with_phone_number_format:
            # Lay the number out as xxx-xxx-xxxx: the middle group stays put, and the
            # outer groups shift out by one digit width to make room for the dashes
            pitch = (digit_positions[-1] - digit_positions[0]) / (len(phone_number) - 1)
            dashes = self.get_phone_number_dashes(phone_number)
            dashes[0].set_x(0.5 * (digit_positions[2] - pitch + digit_positions[3]))
            dashes[1].set_x(0.5 * (digit_positions[5] + digit_positions[6] + pitch))
            characters = [*phone_number[:3], dashes[0], *phone_number[3:6], dashes[1], *phone_number[6:]]
            char_positions = [
                *(x - pitch for x in digit_positions[:3]),
                dashes[0].get_x(),
                *digit_positions[3:6],
                dashes[1].get_x(),
                *(x + pitch for x in digit_positions[6:]),
            ]
            dashes.set_opacity(0)
            self.add(dashes)
        else:
            characters = list(phone_number)
            char_positions = digit_positions

        # Type the characters in, calculator style
        phone_number.set_opacity(0)
        self.add(phone_number)
        for i in range(1, len(characters) + 1):
            for j, char in enumerate(characters[:i][::-1]):
                char.set_x(char_positions[-(j + 1)])
                char.set_opacity(1)
            self.wait(0.1)
        self.wait(0.7)

        if start_with_phone_number_format:
            # Fade out the dashes and close up the gaps
            self.play(
                FadeOut(dashes, scale=0.5),
                *(digit.animate.set_x(x) for digit, x in zip(phone_number, digit_positions)),
                run_time=0.8,
            )
            self.wait(0.4)

        self.play(
            AnimationGroup(
                GrowFromCenter(times_symbol),
                FadeIn(multiple),
                ShowCreation(equals_symbol),
                FadeIn(result, run_time=0.7),
                lag_ratio=0.4
            )
        )

    def get_phone_number_dashes(self, phone_number):
        # Take the dash from "0-0" so it sits at the right height relative to the digits
        template = Text("0-0", font_size=phone_number.get_font_size(), **phone_number.text_config)
        template.set_color(phone_number.get_color())
        template.set_y(phone_number.get_y(DOWN), DOWN)
        return VGroup(template[1].copy(), template[1].copy())

    def set_camera_target_position(
        self,
        theta_degrees=None,
        phi_degrees=None,
        gamma_degrees=None,
        center=None,
        height=None,
        drift_time=2.0,
    ):
        frame = self.camera.frame
        frame.clear_updaters()
        initial_orientation = frame.get_orientation()
        initial_height = frame.get_height()
        initial_eye = frame.get_implied_camera_location()
        target_frame = frame.copy()
        target_frame.reorient(theta_degrees, phi_degrees, gamma_degrees, center, height)
        target_orientation = target_frame.get_orientation()
        target_height = target_frame.get_height()
        target_eye = target_frame.get_implied_camera_location()
        fovy = frame.get_field_of_view()
        slerp = Slerp([0, 1], Rotation.concatenate([initial_orientation, target_orientation]))
        drift_time = max(drift_time, 1e-4)
        elapsed = 0.0

        def update_camera(f, dt):
            nonlocal elapsed
            elapsed += dt
            t = min(elapsed / drift_time, 1.0)
            alpha = smooth(t)
            current_orientation = slerp(alpha)
            current_height = interpolate(initial_height, target_height, alpha)
            current_eye = interpolate(initial_eye, target_eye, alpha)
            focal_distance = 0.5 * current_height / np.tan(0.5 * fovy)
            to_camera = current_orientation.as_matrix().T[2]
            current_center = current_eye - focal_distance * to_camera
            f.set_orientation(current_orientation)
            f.move_to(current_center)
            f.set_height(current_height)
            if t >= 1.0:
                f.remove_updater(update_camera)
        frame.add_updater(update_camera)


class LargeMultipleScene(PhoneNumberScene):
    """
    For multiples too big for the default layout: counts up in log space, keeps the
    digits in fixed columns, pushes the × and the line left as the multiple grows,
    and drifts the camera out so everything stays framed.
    """
    count_up_time = 15
    camera_lead = 2        # The camera finishes its drift this many seconds before the count
    width_fraction = 0.5   # Share of the frame width the final numbers span

    def setup(self):
        super().setup()
        phone_number, times_symbol, multiple, line, result = self.multiplication_group

        multiple.clear_updaters()
        result.clear_updaters()

        self.log_k = np.log10(float(self.k))
        self.log_multiple = ValueTracker(np.log10(2))

        self.digit_pitch = (phone_number[-1].get_x() - phone_number[0].get_x()) / (len(phone_number) - 1)
        self.right_edge = phone_number[-1].get_x() + 0.5 * self.digit_pitch
        self.multiple_bottom = multiple.get_y(DOWN)
        self.result_bottom = result.get_y(DOWN)

        self.times_home_x = times_symbol.get_x(RIGHT)
        phone_left = self.right_edge - len(phone_number) * self.digit_pitch
        self.times_buff = phone_left - self.times_home_x
        self.line_overhang = times_symbol.get_x(LEFT) - line.get_start()[0]
        self.line_end = line.get_end().copy()

        line.set_scale_stroke_with_zoom(True)

        self.update_layout()

    def count_up_to_k(self, move_camera=False, additional_anim=Animation(Mobject()), run_time=None):
        # Keep the layout live from here on
        self.multiplication_group.add_updater(lambda m: self.update_layout())
        self.add(self.multiplication_group)

        # Count up while the camera drifts to frame the final numbers
        if move_camera:
            center, height = self.get_final_framing()
            drift_time = max(self.count_up_time - self.camera_lead, 1)
            self.set_camera_target_position(center=center, height=height, drift_time=drift_time)
        self.play(
            self.log_multiple.animate(run_time=run_time if run_time is not None else self.count_up_time).set_value(self.log_k),
            additional_anim,
            rate_func=smooth,
        )

    def get_multiple(self) -> int:
        log_value = float(self.log_multiple.get_value())
        if log_value >= self.log_k - 1e-9:
            return self.k
        # Decimal keeps every digit genuine (floats would leave the low digits as rounding junk)
        return int(Context(prec=100).power(Decimal(10), Decimal(log_value)).to_integral_value())

    def update_layout(self, value=None):
        phone_number, times_symbol, multiple, line, result = self.multiplication_group
        if value is None:
            value = self.get_multiple()

        # Both rows sit in the digit columns, right-aligned with the phone number
        for number, number_value, bottom in [
            (multiple, value, self.multiple_bottom),
            (result, self.n * value, self.result_bottom),
        ]:
            number.set_value(number_value)
            for i, digit in enumerate(number.submobjects[::-1]):
                digit.set_x(self.right_edge - (i + 0.5) * self.digit_pitch)
            number.set_y(bottom, DOWN)

        # Once the multiple reaches the ×, push the × along, keeping the same buffer
        multiple_left = self.right_edge - len(multiple) * self.digit_pitch
        times_symbol.set_x(min(self.times_home_x, multiple_left - self.times_buff), RIGHT)

        # The left end of the line follows the ×
        line_start = self.line_end.copy()
        line_start[0] = times_symbol.get_x(LEFT) - self.line_overhang
        line.put_start_and_end_on(line_start, self.line_end)

    def get_final_framing(self):
        # Briefly lay out the final values to measure them, then restore the current ones
        self.update_layout(self.k)
        group = self.multiplication_group
        center = group.get_center()
        width = group.get_width() / self.width_fraction
        height = width * self.frame.get_height() / self.frame.get_width()
        height = max(height, self.frame.get_height())  # Only ever zoom out
        self.update_layout()
        return center, height


class Example1(PhoneNumberScene):
    def __init__(self, **kwargs):
        super().__init__(n=7299270073, k=10, **kwargs)

    def construct(self):
        # Set up the multiplication
        self.play_setup()

        # Set the multiple to its target value
        self.play(self.multiple_value.animate(rate_func=linear).set_value(self.k), run_time=2)


class Example2(PhoneNumberScene):
    def __init__(self, **kwargs):
        super().__init__(n=7299270073, k=137, **kwargs)

    def construct(self):
        # Set up the multiplication
        self.play_setup()

        # Set the multiple to its target value
        self.play(self.multiple_value.animate(rate_func=linear).set_value(self.k), run_time=2)


class Example3(PhoneNumberScene):
    def __init__(self, **kwargs):
        super().__init__(n=2849002849, k=39, **kwargs)

    def construct(self):
        # Set up the multiplication
        self.play_setup()

        # Set the multiple to its target value
        self.play(self.multiple_value.animate(rate_func=linear).set_value(self.k), run_time=2)


class Example4(PhoneNumberScene):
    def __init__(self, **kwargs):
        super().__init__(n=6537945880, k=16977825, **kwargs)

    def construct(self):
        # Set up the multiplication
        self.play_setup(start_with_phone_number_format=True)

        # Set the multiple to its target value
        self.play(self.multiple_value.animate(rate_func=linear).set_value(500), run_time=2)
        self.play(self.multiple_value.animate.set_value(self.k), run_time=4)


class Example5(LargeMultipleScene):
    count_up_time = 5
    camera_lead = 1

    def __init__(self, **kwargs):
        super().__init__(n=1089218911, k=102009899010201, **kwargs)

    def construct(self):
        # Set up the multiplication
        self.play_setup()

        # Show the last digit is 1, 3, 7, or 9
        last_digit = self.multiplication_group[0][-1]
        for _ in range(2):
            self.play(Indicate(last_digit, scale_factor=1, color=RED_D))
        self.play(last_digit.animate.set_color(RED_D), run_time=0.5)

        # Set the multiple to its target value
        self.count_up_to_k(
            additional_anim=self.camera.frame.animate(run_time=3).scale(2.5).shift(LEFT * 5),
            run_time=4
        )


class Example6(LargeMultipleScene):
    def __init__(self, **kwargs):
        super().__init__(n=7740740733, k=14354067, **kwargs)

    def construct(self):
        # Set up the multiplication
        self.play_setup()

        # Show the last digit is 1, 3, 7, or 9
        last_digit = self.multiplication_group[0][-1]
        for _ in range(2):
            self.play(Indicate(last_digit, scale_factor=1, color=RED_D))
        self.play(last_digit.animate.set_color(RED_D), run_time=0.5)

        # Set the multiple to its target value
        self.count_up_to_k(
            additional_anim=self.camera.frame.animate(run_time=3).scale(2.5).shift(LEFT * 5),
            run_time=4
        )


class Example7(LargeMultipleScene):
    count_up_time = 8
    camera_lead = 1

    def __init__(self, **kwargs):
        super().__init__(n=5157478847, k=215436872175912408761278456313, **kwargs)

    def construct(self):
        # Set up the multiplication
        self.play_setup()

        # Show the last digit is 1, 3, 7, or 9
        last_digit = self.multiplication_group[0][-1]
        for _ in range(2):
            self.play(Indicate(last_digit, scale_factor=1, color=RED_D))
        self.play(last_digit.animate.set_color(RED_D), run_time=0.5)

        # Set the multiple to its target value
        self.count_up_to_k(
            additional_anim=self.camera.frame.animate(run_time=3).scale(2.5).shift(LEFT * 5),
            run_time=4
        )


class Example8(LargeMultipleScene):
    def __init__(self, **kwargs):
        # super().__init__(n=9148148139, k=12145749, **kwargs)
        super().__init__(n=2849002849, k=39, **kwargs)

    def construct(self):
        # Set up the multiplication
        self.play_setup()

        # Show the last digit is 1, 3, 7, or 9
        last_digit = self.multiplication_group[0][-1]
        for _ in range(2):
            self.play(Indicate(last_digit, scale_factor=1, color=RED_D))
        self.play(last_digit.animate.set_color(RED_D), run_time=0.5)

        # Set the multiple to its target value
        self.count_up_to_k(
            additional_anim=self.camera.frame.animate(run_time=3).scale(2.5).shift(LEFT * 5),
            run_time=4
        )


class BruteForce(LargeMultipleScene):
    count_up_time = 15

    def __init__(self, **kwargs):
        super().__init__(n=3141592653, k=350144093331030307193999225017, **kwargs)

    def construct(self):
        # Set up the multiplication
        self.play_setup()

        # Count up to the target multiple
        self.count_up_to_k(True)
        self.wait(2)
