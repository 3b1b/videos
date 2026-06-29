from manim_imports_ext import *
import math
import random

PURE_MAGENTA = "#FF00FF"

class FancyCircumscribe(VShowPassingFlash):
    def __init__(self, mobject, time_width = 1.5, run_time = 3, **kwargs):
        rect = SurroundingRectangle(mobject).set_stroke(YELLOW, 3)
        rect.add_line_to(rect.get_corner(UL))
        rect.insert_n_curves(100)
        super().__init__(rect, time_width = time_width, run_time = run_time, **kwargs)


# class Robot(Group):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.face = SVGMobject("robot").set_color(TEAL)
#         self.left_eye = self.face[0]
#         self.right_eye = self.face[1]
#         self.mouth = self.face[2]
#         self.head = self.face[3]
#         self.left_eye.set_color([LIGHT_BROWN])
#         self.head_background = Rectangle(
#             width = 2, height = 1.5, fill_opacity = 1, fill_color = BLACK, stroke_width = 0
#         ).round_corners(
#             0.3*self.head.get_width()
#         ).match_width(
#             self.head
#         ).scale(
#             0.8
#         ).align_to(
#             self.head, DOWN
#         ).shift(
#             UP*0.05*self.head.get_width()
#         )
#         self.add(self.head_background)
#         self.add(self.face)

#         self.blinker = GlowDot().set_width(
#             self.head.get_width()*0.6
#         ).set_color(
#             YELLOW
#         ).move_to(
#             self.head.get_top() + DOWN*0.08*self.head.get_width()
#         )
#         self.blinker_opacity_tracker = ValueTracker(0)
#         self.blinker.add_updater(lambda m: m.set_opacity(self.blinker_opacity_tracker.get_value()))
#         self.add(self.blinker)

#         self.move_amount = 0.5

#     def create(self):
#         return AnimationGroup(
#             FadeIn(self.head_background),
#             *[DrawBorderThenFill(part) for part in self.face]
#         , lag_ratio = 0.1)

#     def blink_antenna(self):
#         return UpdateFromAlphaFunc(self.blinker_opacity_tracker, lambda m, a: m.set_value(1 - (2*a - 1)**2))

#     def execute_instruction(self, instruction_index):
#         direction = [UP, DOWN, LEFT, RIGHT][instruction_index]
#         return AnimationGroup(
#             self.blink_antenna(),
#             self.animate.shift(direction*self.move_amount*self.head.get_width())
#         , lag_ratio = 0.3)


class Robot(Group):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.overhead_image_vertical = ImageMobject("images/lunar_rover_assets/Sprite_Vertical.png")
        self.overhead_image_horizontal = ImageMobject("images/lunar_rover_assets/Sprite_Horizontal.png").set_opacity(0)
        self.add(self.overhead_image_vertical, self.overhead_image_horizontal)

        self.blinker = GlowDot().set_width(
            self.overhead_image_vertical.get_width()*0.3
        ).set_color(
            YELLOW
        ).move_to(
            self.overhead_image_vertical.get_bottom() + UP*0.193*self.overhead_image_vertical.get_width() + RIGHT*0.176*self.overhead_image_vertical.get_width()
        )
        self.blinker_opacity_tracker = ValueTracker(0)
        self.blinker.add_updater(lambda m: m.set_opacity(self.blinker_opacity_tracker.get_value()))
        self.add(self.blinker)

        self.move_amount = 0.5
        self.set_width(2.1249567493816737)

    def create(self):
        return FadeIn(self)

    def blink_antenna(self):
        return UpdateFromAlphaFunc(self.blinker_opacity_tracker, lambda m, a: m.set_value(1 - (2*a - 1)**2))

    def execute_instruction(self, instruction_index):
        direction = [UP, DOWN, LEFT, RIGHT][instruction_index]
        if direction[0] == 0:
            self.overhead_image_vertical.set_opacity(1)
            self.overhead_image_horizontal.set_opacity(0)
        else:
            self.overhead_image_vertical.set_opacity(0)
            self.overhead_image_horizontal.set_opacity(1)
        return AnimationGroup(
            self.blink_antenna(),
            self.animate.shift(direction*self.move_amount*self.overhead_image_vertical.get_width())
        , lag_ratio = 0.3)

class RobotTest(InteractiveScene):
    def construct(self):
        # Create the robot
        robot = Robot()
        self.play(robot.create())
        # for _ in range(5):
        #     self.play(robot.blink_antenna())
        for instruction in [0, 1, 2, 3, 0, 1, 2, 3]:
            self.play(robot.execute_instruction(instruction))

class InstructionArrow(SVGMobject):
    def __init__(self, direction = RIGHT, *args, **kwargs):
        super().__init__("images/arrow.svg", *args, **kwargs)
        if (direction == UP).all():
            self.rotate(PI/2)
        elif (direction == LEFT).all():
            self.rotate(PI)
        elif (direction == DOWN).all():
            self.rotate(3*PI/2)
        self.direction = direction


def generate_random_instructions(n, distribution, seed = 0):
    np.random.seed(seed)
    instructions = []
    for _ in range(n):
        x = np.random.random()
        for i in range(len(distribution)):
            if x < sum(distribution[:i + 1]):
                instructions.append(i)
                break
    return instructions

class RobotOnMoon(InteractiveScene):
    def construct(self):
        # Add the robot
        robot = Robot()
        self.play(robot.create(), run_time = 2)

        # Add the surface of the far away moon
        moon_surface = ImageMobject("images/far_away_moon.png").get_grid(20, 20, buff = 0).scale(5).set_opacity(0.4)
        self.bring_to_back(moon_surface)
        self.play(FadeIn(moon_surface), run_time = 2)

        # The robot roams around
        directions = [UP, DOWN, LEFT, RIGHT]
        instruction_set = VGroup(*[
            InstructionArrow(
                direction
            ).scale(
                0.5
            ).set_opacity(
                0
            )
            for direction in directions
        ])
        self.add(instruction_set)
        arrow_length = instruction_set[0].get_height()
        arrow_animation_tracker = ValueTracker(0)
        for arrow in instruction_set:
            def update_arrow(m):
                m.set_opacity(arrow_animation_tracker.get_value()*0.6)
                m.next_to(
                    robot.overhead_image_vertical, m.direction, buff = 0.8 if (m.direction == DOWN).all() else 1.2
                )
                if m.direction[1] == 0:
                    m.stretch_to_fit_width(max(arrow_animation_tracker.get_value()*arrow_length, 0.1))
                else:
                    m.stretch_to_fit_height(max(arrow_animation_tracker.get_value()*arrow_length, 0.1))
            arrow.add_updater(update_arrow)

        fractions = VGroup(
            Tex(R"\mathbf{1/2}", font_size = 200),
            Tex(R"\mathbf{1/4}", font_size = 200),
            Tex(R"\mathbf{1/8}", font_size = 200),
            Tex(R"\mathbf{1/8}", font_size = 200)
        ).set_color(WHITE)
        self.add(fractions)
        fraction_opacity_trackers = [ValueTracker(0) for fraction in fractions]
        fractions_master_opacity_tracker = ValueTracker(0)
        for i in range(len(fraction_opacity_trackers)):
            def update_fraction_opacity_tracker(m, i = i):
                m.set_value(max(0, min(1, 8*(fractions_master_opacity_tracker.get_value() - 0.25*i))))
            fraction_opacity_trackers[i].add_updater(update_fraction_opacity_tracker)
        self.add(*fraction_opacity_trackers, fractions_master_opacity_tracker)
        def update_fractions(m):
            fractions[0].next_to(instruction_set[0].get_top(), UP, buff = 2).set_opacity(fraction_opacity_trackers[0].get_value())
            fractions[1].next_to(instruction_set[1].get_bottom(), DOWN, buff = 2).set_opacity(fraction_opacity_trackers[1].get_value())
            fractions[2].next_to(instruction_set[2].get_left(), LEFT, buff = 2).set_opacity(fraction_opacity_trackers[2].get_value())
            fractions[3].next_to(instruction_set[3].get_right(), RIGHT, buff = 2).set_opacity(fraction_opacity_trackers[3].get_value())
        fractions.add_updater(update_fractions)

        distribution = [1/2, 1/4, 1/8, 1/8]
        self.gravitate_camera_towards(robot, target_zoom_level = 0.7)
        tail = TracingTail(robot.overhead_image_vertical, stroke_color = TEAL, time_traced = 5, stroke_width=5)
        self.add(tail)
        instructions = generate_random_instructions(750, distribution)
        arrow_draw_iter = 10
        zoom_out_iter = arrow_draw_iter + 6
        fractions_draw_iter = arrow_draw_iter + 19
        iters_to_draw_fractions = 55
        for i in range(len(instructions)):
            anims = [robot.execute_instruction(instructions[i])]
            if i == arrow_draw_iter:
                anims.append(arrow_animation_tracker.animate(rate_func = lambda a: 2*smooth(a*0.5), run_time = 1.3).set_value(0.5))
            if i == arrow_draw_iter + 1:
                anims.append(arrow_animation_tracker.animate(rate_func = lambda a: 2*(smooth((a*0.5 + 0.5)) - smooth(0.5)), run_time = 1.3).set_value(1))
            if fractions_draw_iter <= i < fractions_draw_iter + iters_to_draw_fractions:
                anims.append(
                    fractions_master_opacity_tracker.animate(
                        rate_func = linear
                    ).set_value(
                        ((i + 1) - fractions_draw_iter)/iters_to_draw_fractions
                    )
                )
            run_time = 1
            if i == zoom_out_iter:
                self.gravitate_camera_towards(robot, target_zoom_level = 0.25, zoom_gravity_constant = 0.01)
                self.bring_to_back(tail)
                tail.add_updater(lambda m: m.set_stroke(width = 3))
            if i > zoom_out_iter - 3:
                run_time = max(0.15, smooth(1 - 0.1*(i - (zoom_out_iter - 3))))
            self.play(AnimationGroup(*anims, run_time = run_time))

    def gravitate_camera_towards(
        self,
        mobject_or_func,
        x_pct=0.5,
        y_pct=0.5,
        target_zoom_level = None,
        friction=0.05,
        gravity_constant=0.01,
        zoom_gravity_constant = 0.03
    ):
        frame = self.camera.frame
        frame.clear_updaters()
        frame.velocity = np.zeros(3)
        frame.width_velocity = 0

        def update_camera(f, dt):
            if callable(mobject_or_func):
                target_center = mobject_or_func()
            else:
                target_center = mobject_or_func.get_center()
            
            offset_x = (x_pct - 0.5) * f.get_width()
            offset_y = (y_pct - 0.5) * f.get_height()
            
            desired_center = target_center - np.array([offset_x, offset_y, 0])
            current_center = f.get_center()
            
            direction = desired_center - current_center
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                acceleration = direction * gravity_constant * distance
                f.velocity += acceleration
                
            f.velocity *= (1 - friction)
            f.shift(f.velocity * dt)

            if target_zoom_level is not None:
                width_diff = (1/target_zoom_level)*FRAME_WIDTH - f.get_width()
                width_acceleration = width_diff * zoom_gravity_constant
                f.width_velocity += width_acceleration
                f.width_velocity *= (1 - friction)
                
                new_width = f.get_width() + f.width_velocity * dt
                f.set_width(new_width)

        frame.add_updater(update_camera)

class RobotOnMoonShort(RobotOnMoon):
    def construct(self):
        # Add the robot
        robot = Robot()
        self.play(robot.create(), run_time = 2)

        # Add the surface of the far away moon
        moon_surface = ImageMobject("images/far_away_moon.png").get_grid(20, 20, buff = 0).scale(5).set_opacity(0.4)
        self.bring_to_back(moon_surface)
        self.play(FadeIn(moon_surface), run_time = 2)

        # The robot roams around
        directions = [UP, DOWN, LEFT, RIGHT]
        instruction_set = VGroup(*[
            InstructionArrow(
                direction
            ).scale(
                0.5
            ).set_opacity(
                0
            )
            for direction in directions
        ])
        self.add(instruction_set)
        arrow_length = instruction_set[0].get_height()
        arrow_animation_tracker = ValueTracker(0)
        for arrow in instruction_set:
            def update_arrow(m):
                m.set_opacity(arrow_animation_tracker.get_value()*0.6)
                m.next_to(
                    robot.overhead_image_vertical, m.direction, buff = 0.5 if (m.direction == DOWN).all() else 0.9
                )
                if m.direction[1] == 0:
                    m.stretch_to_fit_width(max(arrow_animation_tracker.get_value()*arrow_length, 0.1))
                else:
                    m.stretch_to_fit_height(max(arrow_animation_tracker.get_value()*arrow_length, 0.1))
            arrow.add_updater(update_arrow)

        fractions = VGroup(
            Tex(R"\mathbf{1/2}", font_size = 160),
            Tex(R"\mathbf{1/4}", font_size = 160),
            Tex(R"\mathbf{1/8}", font_size = 160),
            Tex(R"\mathbf{1/8}", font_size = 160)
        ).set_color(WHITE)
        self.add(fractions)
        fraction_opacity_trackers = [ValueTracker(0) for fraction in fractions]
        fractions_master_opacity_tracker = ValueTracker(0)
        for i in range(len(fraction_opacity_trackers)):
            def update_fraction_opacity_tracker(m, i = i):
                m.set_value(max(0, min(1, 8*(fractions_master_opacity_tracker.get_value() - 0.25*i))))
            fraction_opacity_trackers[i].add_updater(update_fraction_opacity_tracker)
        self.add(*fraction_opacity_trackers, fractions_master_opacity_tracker)
        def update_fractions(m):
            fractions[0].next_to(instruction_set[0].get_top(), UP, buff = 0.8).set_opacity(fraction_opacity_trackers[0].get_value())
            fractions[1].next_to(instruction_set[1].get_bottom(), DOWN, buff = 0.8).set_opacity(fraction_opacity_trackers[1].get_value())
            fractions[2].next_to(instruction_set[2].get_left(), LEFT, buff = 0.8).set_opacity(fraction_opacity_trackers[2].get_value())
            fractions[3].next_to(instruction_set[3].get_right(), RIGHT, buff = 0.8).set_opacity(fraction_opacity_trackers[3].get_value())
        fractions.add_updater(update_fractions)

        distribution = [1/2, 1/4, 1/8, 1/8]
        self.gravitate_camera_towards(robot, target_zoom_level = 0.45)
        tail = TracingTail(robot.overhead_image_vertical, stroke_color = TEAL, time_traced = 5, stroke_width=5)
        self.add(tail)
        instructions = generate_random_instructions(200, distribution)
        arrow_draw_iter = 10
        zoom_out_iter = arrow_draw_iter + 6
        fractions_draw_iter = arrow_draw_iter + 9
        iters_to_draw_fractions = 40
        for i in range(len(instructions)):
            anims = [robot.execute_instruction(instructions[i])]
            if i == arrow_draw_iter:
                anims.append(arrow_animation_tracker.animate(rate_func = lambda a: 2*smooth(a*0.5), run_time = 1.3).set_value(0.5))
            if i == arrow_draw_iter + 1:
                anims.append(arrow_animation_tracker.animate(rate_func = lambda a: 2*(smooth((a*0.5 + 0.5)) - smooth(0.5)), run_time = 1.3).set_value(1))
            if fractions_draw_iter <= i < fractions_draw_iter + iters_to_draw_fractions:
                anims.append(
                    fractions_master_opacity_tracker.animate(
                        rate_func = linear
                    ).set_value(
                        ((i + 1) - fractions_draw_iter)/iters_to_draw_fractions
                    )
                )
            run_time = 1
            if i == zoom_out_iter:
                self.gravitate_camera_towards(robot, target_zoom_level = 0.22, zoom_gravity_constant = 0.01)
                self.bring_to_back(tail)
                tail.add_updater(lambda m: m.set_stroke(width = 3))
            if i > zoom_out_iter - 3:
                run_time = max(0.15, smooth(1 - 0.1*(i - (zoom_out_iter - 3))))
            self.play(AnimationGroup(*anims, run_time = run_time))


class RobotOnMoon2(RobotOnMoon):
    def construct(self):
        # Add the robot and the moon
        robot = Robot()
        moon_surface = ImageMobject("images/far_away_moon.png").get_grid(20, 20, buff = 0).scale(5).set_opacity(0.4)
        self.add(moon_surface, robot)
        self.gravitate_camera_towards(robot, gravity_constant = 0.1)

        # Move the robot around a bit randomly
        tail = TracingTail(robot.overhead_image_vertical, stroke_color = TEAL, time_traced = 5, stroke_width=5)
        self.add(tail)
        distribution = [1/2, 1/4, 1/8, 1/8]
        instructions = generate_random_instructions(18, distribution)
        for instruction in instructions:
            self.play(robot.execute_instruction(instruction))
            self.bring_to_back(tail)
        self.wait(6)


        # Move the robot according to the specific instructions from the encoding example
        instructions = [1, 0, 2, 0, 2, 0, 1, 3, 1, 1, 0, 0, 0, 3, 2, 0] + generate_random_instructions(40, distribution)
        for i in range(len(instructions)):
            self.play(robot.execute_instruction(instructions[i]), run_time = 1 if i < 3 else 0.35)
            self.bring_to_back(tail)
            if i < 3:
                self.wait(6)
        self.wait(6)

class RobotOnMoon3(RobotOnMoon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_distribution_version = True
    def construct(self):
        # Add the robot, the moon, and the arrows
        robot = Robot()
        moon_surface = ImageMobject("images/far_away_moon.png").get_grid(20, 20, buff = 0).scale(5).set_opacity(0.4)

        directions = [UP, DOWN, LEFT, RIGHT]
        instruction_set = VGroup(*[
            InstructionArrow(
                direction
            ).scale(
                0.5
            )
            for direction in directions
        ])
        self.add(instruction_set)
        arrow_length = instruction_set[0].get_height()
        arrow_animation_tracker = ValueTracker(1)
        for arrow in instruction_set:
            def update_arrow(m):
                m.set_opacity(arrow_animation_tracker.get_value()*0.6)
                m.next_to(
                    robot.overhead_image_vertical, m.direction, buff = 0.8 if (m.direction == DOWN).all() else 1.2
                )
                if m.direction[1] == 0:
                    m.stretch_to_fit_width(max(arrow_animation_tracker.get_value()*arrow_length, 0.1))
                else:
                    m.stretch_to_fit_height(max(arrow_animation_tracker.get_value()*arrow_length, 0.1))
            arrow.add_updater(update_arrow)

        fractions = VGroup(
            Tex(R"\mathbf{1/2}", font_size = 200),
            Tex(R"\mathbf{1/4}", font_size = 200),
            Tex(R"\mathbf{1/8}", font_size = 200),
            Tex(R"\mathbf{1/8}", font_size = 200)
        ).set_color(PINK)
        self.add(fractions)
        fraction_opacity_trackers = [ValueTracker(1) for fraction in fractions]
        self.add(*fraction_opacity_trackers)
        def update_fractions(m):
            fractions[0].next_to(instruction_set[0].get_top(), UP, buff = 2).set_opacity(fraction_opacity_trackers[0].get_value())
            fractions[1].next_to(instruction_set[1].get_bottom(), DOWN, buff = 2).set_opacity(fraction_opacity_trackers[1].get_value())
            fractions[2].next_to(instruction_set[2].get_left(), LEFT, buff = 2).set_opacity(fraction_opacity_trackers[2].get_value())
            fractions[3].next_to(instruction_set[3].get_right(), RIGHT, buff = 2).set_opacity(fraction_opacity_trackers[3].get_value())
        fractions.add_updater(update_fractions)

        self.add(moon_surface, robot, instruction_set, fractions)
        self.gravitate_camera_towards(robot, gravity_constant = 0.05, target_zoom_level = 0.3)
        self.wait(6)


        # Move the robot according to the old distribution, then change it to the new one
        tail_1 = TracingTail(robot.overhead_image_vertical, stroke_color = PINK, time_traced = 5, stroke_width=3)
        tail_2 = TracingTail(robot.overhead_image_vertical, stroke_color = GREEN, time_traced = 5, stroke_width=3)
        hide_tail_2 = lambda m: m.set_opacity(0)
        tail_2.add_updater(hide_tail_2)
        self.add(tail_1, tail_2)

        old_distribution = [1/2, 1/4, 1/8, 1/8]
        new_distribution = [1/8, 1/8, 1/4, 1/2]
        total_iters = 400
        distribution_change_iter = 100
        iters_to_remove_initial_fractions = 1
        new_fractions_draw_iter = distribution_change_iter + iters_to_remove_initial_fractions + 47
        iters_to_draw_fractions = 51
        pan_camera_down_iter = new_fractions_draw_iter + 40

        if self.initial_distribution_version:
            instructions = generate_random_instructions(total_iters, old_distribution)
        else:
            instructions = generate_random_instructions(distribution_change_iter + iters_to_remove_initial_fractions, old_distribution)
            instructions += generate_random_instructions(total_iters - len(instructions), new_distribution)

        fractions_master_opacity_tracker = ValueTracker(0)
        for i, instruction in enumerate(instructions):
            anims = [robot.execute_instruction(instruction)]
            if distribution_change_iter <= i < distribution_change_iter + iters_to_remove_initial_fractions and not self.initial_distribution_version:
                anims.append(
                    AnimationGroup(*[
                        t.animate(
                            rate_func = linear
                        ).set_value(
                            1 - (((i + 1) - distribution_change_iter)/iters_to_remove_initial_fractions)
                        )
                        for t in fraction_opacity_trackers
                    ])
                )
            if i == distribution_change_iter + iters_to_remove_initial_fractions and not self.initial_distribution_version:
                fractions.clear_updaters()
                def update_fractions(m):
                    fractions[0].next_to(
                        instruction_set[3].get_right(), RIGHT, buff = 2
                    ).set_opacity(fraction_opacity_trackers[3].get_value())
                    fractions[1].next_to(
                        instruction_set[2].get_left(), LEFT, buff = 2
                    ).set_opacity(fraction_opacity_trackers[2].get_value())
                    fractions[2].next_to(
                        instruction_set[0].get_top(), UP, buff = 2
                    ).set_opacity(fraction_opacity_trackers[0].get_value())
                    fractions[3].next_to(
                        instruction_set[1].get_bottom(), DOWN, buff = 2
                    ).set_opacity(fraction_opacity_trackers[1].get_value())
                fractions.add_updater(update_fractions)
                for i in range(len(fraction_opacity_trackers)):
                    def update_fraction_opacity_tracker(m, i = i):
                        m.set_value(max(0, min(1, 8*(fractions_master_opacity_tracker.get_value() - 0.25*i))))
                    fraction_opacity_trackers[i].add_updater(update_fraction_opacity_tracker)
                self.add(fractions_master_opacity_tracker)

            if i == distribution_change_iter:
                if self.initial_distribution_version:
                    self.gravitate_camera_towards(
                        lambda: robot.get_center() + RIGHT*25 + UP*15,
                        gravity_constant = 0.002,
                        target_zoom_level = 0.14,
                        zoom_gravity_constant = 0.02
                    )
                else:
                    self.remove(tail_1)
                    tail_2.remove_updater(hide_tail_2)
                    fractions.set_color(GREEN)
                    self.gravitate_camera_towards(
                        lambda: robot.get_center() + LEFT*18 + UP*10,
                        gravity_constant = 0.002,
                        target_zoom_level = 0.14,
                        zoom_gravity_constant = 0.02
                    )
            if new_fractions_draw_iter <= i < new_fractions_draw_iter + iters_to_draw_fractions and not self.initial_distribution_version:
                anims.append(
                    fractions_master_opacity_tracker.animate(
                        rate_func = linear
                    ).set_value(
                        ((i + 1) - new_fractions_draw_iter)/iters_to_draw_fractions
                    )
                )
            # if i == pan_camera_down_iter:
            #     if self.initial_distribution_version:
            #         self.gravitate_camera_towards(
            #             lambda: robot.get_center() + RIGHT*17 + UP*9.5,
            #             gravity_constant = 0.01,
            #             target_zoom_level = 0.2,
            #             zoom_gravity_constant = 0.01
            #         )
            #     else:
            #         tail.add_updater(lambda m: m.set_color(GREEN))
            #         self.gravitate_camera_towards(
            #             lambda: robot.get_center() + LEFT*14 + UP*8,
            #             gravity_constant = 0.01,
            #             target_zoom_level = 0.2,
            #             zoom_gravity_constant = 0.01
            #         )
            self.play(*anims, run_time = 0.15)

class RobotOnMoon4(RobotOnMoon3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_distribution_version = False


def random_distribution(n, thresh = 1/16):
    if n <= 0:
        return []
    if n * thresh > 1:
        raise ValueError("Threshold is too high to sum to 1")
    remaining_sum = 1 - (n * thresh)
    random_parts = np.random.dirichlet(np.ones(n), size=1).flatten()
    numbers = (random_parts * remaining_sum) + thresh
    return numbers.tolist()


from _2026.cross_entropy.distribution import *
class RobotEncodings(InteractiveScene):
    def construct(self):
        # Add mission_control and the robot
        mission_control = ImageMobject(
            "images/pi_creature_mission_control.png"
        ).set_width(2).to_edge(RIGHT, buff = 0.4).to_edge(DOWN, buff = 0.6)
        robot = ImageMobject(
            "images/lunar_rover_assets/stationary.png"
        ).match_height(mission_control).to_edge(LEFT, buff = 0.4).align_to(mission_control, DOWN)
        self.add(mission_control, robot)
        self.wait(2)

        # Create a stream of bits flowing towards the bot, and decode them into instructions by chunks of 2
        distribution = [1/2, 1/4, 1/8, 1/8]
        instructions = [1, 0, 2] # Start with DOWN, UP, LEFT = 100110 with perfect encoding
        instructions += generate_random_instructions(13, distribution, seed = 7) # 16 total instructions
        instructions[-3] = 3
        bit_string = ""
        for instruction in instructions:
            bit_string += f"{instruction:02b}"
        bit_buff = 0.1
        bits = VGroup(*[
            Tex(bit_string[i], font_size = 32)
            for i in range(len(bit_string))
        ]).arrange(buff = bit_buff).set_color(YELLOW).match_y(robot).align_to(mission_control, LEFT)
        bits_opacity_tracker = ValueTracker(0)
        def update_bits(m):
            for bit in m:
                bit.set_opacity(bits_opacity_tracker.get_value()*min(1, max(0, 0.8*(mission_control.get_left()[0] - bit.get_x()))))
                if bit.get_x() < robot.get_x():
                    bit.set_opacity(0)
            self.bring_to_front(robot)
        bits.add_updater(update_bits)

        rects = VGroup(*[
            SurroundingRectangle(bits[i:i + 2], stroke_width = 2, stroke_color = WHITE, buff = bit_buff*0.5)
            for i in range(0, len(bit_string), 2)
        ])
        for rect in rects:
            rect.stretch_to_fit_height(rects[0].get_height())
        rect_opacity_trackers = [ValueTracker(0) for _ in range(4)]
        def update_rects(m):
            for i, rect in enumerate(m):
                target_bits = bits[2*i:2*(i + 1)]
                rect.match_x(target_bits)
                rect.set_stroke(
                    opacity = rect_opacity_trackers[
                        instructions[i]
                    ].get_value()*min(1, max(0, 0.8 * (mission_control.get_left()[0] - (rect.get_x() + 0.2))))
                )
                if rect.get_x() < robot.get_x():
                    rect.set_stroke(opacity = 0)
        rects.add_updater(update_rects)

        arrows = VGroup(*[
            InstructionArrow([UP, DOWN, LEFT, RIGHT][instructions[i]]).scale(0.07).move_to(rects[i]).shift(UP*0.62)
            for i in range(len(instructions))
        ])
        arrows_opacity_tracker = ValueTracker(1)
        def update_arrows(m):
            for i, arrow in enumerate(m):
                target_bits = bits[2*i:2*(i + 1)]
                arrow.match_x(target_bits)
                opacity = min(1, max(0, 0.8*(mission_control.get_left()[0] - (arrow.get_x() + 0.2))))
                if arrow.get_x() < robot.get_right()[0]:
                    opacity = min(1, max(0, 1 - 1.2*(robot.get_right()[0] - arrow.get_x())))
                arrow.set_opacity(arrows_opacity_tracker.get_value()*opacity)
        arrows.add_updater(update_arrows)

        self.add(bits, rects, arrows)
        self.play(
            AnimationGroup(
                bits.animate(run_time = 40, rate_func = smooth).align_to(robot.get_right() + RIGHT*0.2, LEFT),
                bits_opacity_tracker.animate.set_value(1)
            , lag_ratio = 0.65, rate_func = linear)
        )

        # Show the naive encoding
        table = VGroup(*[
            VGroup(
                Tex(f"{i:02b}").set_color(YELLOW),
                Tex(":"),
                InstructionArrow(direction = [UP, DOWN, LEFT, RIGHT][i]).scale(0.15)
            ).arrange()
            for i in range(4)
        ])
        table.arrange(DOWN)
        for row in table:
            row.shift(RIGHT*(table[0][1].get_x() - row[1].get_x()))
        for i, row in enumerate(table):
            row[2].match_x(table[2][2])
            row.set_y(table[0].get_y() + i*(table[1].get_y() - table[0].get_y()))
        table.set_width(2).to_edge(RIGHT, buff = 2).to_edge(UP, buff = 0.7)
        table_box = SurroundingRectangle(table, buff = 0.15).set_color(WHITE).set_stroke(opacity = 0)

        arrows.save_state()
        for i, row in enumerate(table):
            anims = []
            anims.append(FadeIn(row))
            anims.append(rect_opacity_trackers[i].animate.set_value(1))
            if i == 0:
                anims.append(table_box.animate.set_stroke(opacity = 1))
            else:
                anims.append(rect_opacity_trackers[i - 1].animate(run_time = 1).set_value(0))
                anims.append(
                    AnimationGroup(*[
                        arrow.animate(run_time = 1).set_color(WHITE if instructions[j] == i else YELLOW)
                        for j, arrow in enumerate(arrows)
                    ])
                )
            self.play(AnimationGroup(*anims, run_time = 2))
        self.wait(1)
        self.play(arrows.animate.restore(), run_time = 2)
        self.wait(2)    

        self.play(AnimationGroup(*[t.animate.set_value(1) for t in rect_opacity_trackers]), run_time = 2)

        # Transform the table into a stacked bar diagram
        naive_chart = EntropyChart(
            distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]),
            probability_labels = [
                Tex(R"\frac{1}{" + ["2", "4", "8", "8"][i] + "}", font_size = 80)
                for i in range(4)
            ],
            bar_labels = [
                Tex(["00", "01", "10", "11"][i], font_size = 90)
                for i in range(4)
            ],
            bar_heights = [2, 2, 2, 2],
            width = 9,
            include_vertical_axis = False,
            segments_height = 1,
            fill_colors = [YELLOW_B, YELLOW_D]
        ).match_height(table).scale(0.5).match_y(table).align_to(table, RIGHT).shift(RIGHT)
        naive_chart.bar_labels.set_color(WHITE)
        naive_chart.update()
        self.wait(0.01)
        naive_chart.suspend_updating()
        self.play(
            AnimationGroup(
                FadeOut(VGroup(table_box, *[row[1] for row in table])),
                AnimationGroup(
                    FadeIn(naive_chart.segments.bars),
                    AnimationGroup(*[
                        ReplacementTransform(table[i][2], naive_chart.event_labels[i])
                        for i in range(len(table))
                    ]),
                    FadeIn(naive_chart.probability_labels),
                    AnimationGroup(*[
                        ReplacementTransform(table[i][0], naive_chart.bar_labels[i])
                        for i in range(len(table))
                    ])
                )
            , lag_ratio = 0.4)
        )
        self.play(
            AnimationGroup(
                naive_chart.create_bars(),
                naive_chart.bar_labels.animate.set_color(BLACK)
            , lag_ratio = 0.5)
        )
        self.wait(1.8)

        # Save the Naive example
        arrows.clear_updaters()
        rects.clear_updaters()
        bits.clear_updaters()
        naive_example_group = Group(robot, Group(arrows, rects, bits), mission_control, naive_chart)
        naive_example_group.generate_target()
        naive_example_group.target[-1].scale(1.3)
        naive_example_group.target.arrange(buff = 0.6).to_edge(UP, buff = 0.7).set_width(FRAME_WIDTH*0.96)
        naive_example_group.target[1].shift(
            DOWN*(naive_example_group.target[1][2].get_y() - naive_example_group.target.get_y())
        )
        naive_example_group.target[2].align_to(naive_example_group.target[0], DOWN)
        naive_example_group.target[3].shift(DOWN*0.4)
        naive_example_group.target.to_edge(UP, buff = 0.7).set_x(0)
        self.play(MoveToTarget(naive_example_group, path_arc = -PI*0.2))

        # Change the encoding to a Huffman code
        robot_2 = robot.copy()
        arrows_2 = arrows.copy().set_color(PINK)
        mission_control_2 = mission_control.copy()
        huffman_chart_init = EntropyChart(
            distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]).set_color(PINK),
            probability_labels = [
                Tex(R"\frac{1}{" + ["2", "4", "8", "8"][i] + "}", font_size = 80)
                for i in range(4)
            ],
            bar_labels = [
                Tex(["00", "01", "10", "11"][i], font_size = 90)
                for i in range(4)
            ],
            bar_heights = [2, 2, 2, 2],
            width = 9,
            height = 6,
            include_vertical_axis = False,
            segments_height = 1,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).match_width(naive_chart).match_x(naive_chart)
        huffman_chart_init.shift(DOWN*(huffman_chart_init.segments.get_bottom()[1] - mission_control_2.get_bottom()[1]))
        huffman_chart_init.bar_labels.set_color(WHITE)
        huffman_chart_init.update()
        self.wait(0.01)
        huffman_chart_init.suspend_updating()
        Group(robot_2, arrows_2, mission_control_2, huffman_chart_init).to_edge(DOWN, buff = 0.4)
        self.play(TransformFromCopy(Group(robot, arrows, mission_control, naive_chart), Group(robot_2, arrows_2, mission_control_2, huffman_chart_init)))
        self.wait(2)

        encoding = ["0", "10", "110", "111"]
        huffman_chart = EntropyChart(
            distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]).set_color(PINK),
            probability_labels = [
                Tex(R"\frac{1}{" + ["2", "4", "8", "8"][i] + "}", font_size = 80)
                for i in range(4)
            ],
            bar_labels = [
                Tex(encoding[i], font_size = 60)
                for i in range(4)
            ],
            bar_heights = [1, 2, 3, 3],
            width = 9,
            height = 6,
            include_vertical_axis = False,
            segments_height = 1,
            fill_colors = [RED, LIGHT_PINK]
        ).match_width(huffman_chart_init).match_x(huffman_chart_init)
        huffman_chart.bar_labels.set_color(WHITE)
        huffman_chart.update()
        self.wait(0.01)
        huffman_chart.align_to(huffman_chart_init, DOWN)

        for i in range(4):
            self.play(
                ReplacementTransform(huffman_chart_init.bars[i], huffman_chart.bars[i], suspend_mobject_updating = True),
                ReplacementTransform(huffman_chart_init.bar_labels[i], huffman_chart.bar_labels[i], suspend_mobject_updating = True)
            )
            self.wait(2)


        # Encode the new sequence with the Huffman code
        bit_string = ""
        for instruction in instructions:
            bit_string += encoding[instruction]
        bits_2 = VGroup(*[
            Tex(bit_string[i])
            for i in range(len(bit_string))
        ]).set_color(PINK).match_height(bits)
        bits_2.set_y(robot_2.get_y() + (bits.get_y() - robot.get_y()))
        def update_bits_2(m):
            for bit in m:
                if bit.get_x() < robot_2.get_x():
                    bit.set_opacity(0)
            self.bring_to_front(robot_2)
        bits_2.add_updater(update_bits_2)
        index = 0
        bit_groups = VGroup()
        for instruction, arrow in zip(instructions, arrows):
            code_word_length = 1 if instruction == 0 else 2 if instruction == 1 else 3
            target_bits = bits_2[index:index + code_word_length]
            target_bits.arrange(center = False, buff = 0).match_x(arrow)
            bit_groups.add(target_bits)
            index += code_word_length
        self.play(
            AnimationGroup(*[
                TransformMatchingShapes(huffman_chart.bar_labels[instructions[i]].copy(), grouping, path_arc = PI*0.1)
                for i, grouping in enumerate(bit_groups)
            ], lag_ratio = 0.08, run_time = 6)
        )
        self.wait(1)

        def update_arrows_2(m):
            index = 0
            for arrow, instruction in zip(m, instructions):
                code_word_length = 1 if instruction == 0 else 2 if instruction == 1 else 3
                target_bits = bits_2[index:index + code_word_length]
                arrow.match_x(target_bits)
                if arrow.get_x() < robot_2.get_right()[0]:
                    arrow.set_opacity(min(1, max(0, 1 - 1.2*(robot_2.get_right()[0] - arrow.get_x()))))
                index += code_word_length

        arrows_2.add_updater(update_arrows_2)

        rects_2 = VGroup(*[Rectangle() for _ in instructions])
        rects_2_opacity_tracker = ValueTracker(0)
        def update_rects_2(m):
            index = 0
            for i, rect in enumerate(m):
                instruction = instructions[i]
                code_word_length = 1 if instruction == 0 else 2 if instruction == 1 else 3
                target_bits = bits_2[index:index + code_word_length]
                rect.become(
                    SurroundingRectangle(
                        target_bits,
                        stroke_width = 2,
                        stroke_color = WHITE,
                        stroke_opacity = rects_2_opacity_tracker.get_value(),
                        buff = 0.5*(bits_2[1].get_left()[0] - bits_2[0].get_right()[0])
                    )
                )
                if i > 0:
                    rect.stretch_to_fit_height(rects_2[0].get_height())
                if rect.get_x() < robot_2.get_x():
                    rect.set_stroke(opacity = 0)
                index += code_word_length
        rects_2.add_updater(update_rects_2)
        self.add(rects_2)

        self.play(
            AnimationGroup(
                AnimationGroup(*[
                    bit_2.animate.match_x(bit)
                    for bit_2, bit in zip(bits_2, bits)
                ]),
                rects_2_opacity_tracker.animate.set_value(1)
            , lag_ratio = 0.5)
        )
        self.wait(0.5)
        bits_2.suspend_updating()
        rects_2.suspend_updating()
        arrows_2.suspend_updating()

        # Focus on the perfect example
        self.remove(huffman_chart_init)
        self.add(huffman_chart)
        perfect_example_group = Group(robot_2, Group(arrows_2, rects_2, bits_2), mission_control_2, huffman_chart)
        self.play(
            AnimationGroup(
                FadeOut(naive_example_group, shift = UP),
                perfect_example_group[-1].animate(path_arc = PI*0.2).scale(1.2).to_edge(UP, buff = 0.7).to_edge(RIGHT, buff = 0.5),
                perfect_example_group[:-1].animate.set_width(FRAME_WIDTH*0.9).set_x(0).to_edge(DOWN, buff = 0.4)
            , lag_ratio = 0.1, run_time = 2)
        )

        # Introduce "code word" term
        code_words_text = Tex(R"\text{code words}").next_to(huffman_chart, UP).set_color(WHITE).fix_in_frame()
        huffman_chart.clear_updaters()
        huffman_chart.fix_in_frame()
        huffman_chart.save_state()
        self.play(
            Write(code_words_text, run_time = 1.5),
            VGroup(huffman_chart.bars, huffman_chart.segments, huffman_chart.probability_labels).animate.fade(0.8)
        )

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale(0.7, about_point = robot_2.get_corner(DL) + DOWN*0.3).shift(UR*0.2),
            FadeOut(VGroup(arrows_2, rects_2))
        , run_time = 2)

        self.wait(1.2)
        self.play(huffman_chart.animate.restore(), run_time = 2)

        # Break down the first code word
        self.play(
            bits_2[0].animate.set_color(PURE_GREEN),
            bits_2[1:].animate.set_opacity(0.2)
        )  
        self.wait(1)
        huffman_chart.bar_labels.save_state()
        self.play(
            huffman_chart.bar_labels[0].animate.set_opacity(0.2),
            huffman_chart.bars[0].animate.set_opacity(0.2),
            huffman_chart.segments.bars[0].animate.set_opacity(0.2),
            huffman_chart.event_labels[0].animate.set_opacity(0.2),
            huffman_chart.probability_labels[0].animate.set_opacity(0.2),
            AnimationGroup(*[huffman_chart.bar_labels[i][0].animate.set_color(PURE_GREEN) for i in range(1, 4)])
        , run_time = 0.3)
        self.wait(1)
        self.play(bits_2[1].animate.set_opacity(1).set_color(PURE_GREEN))
        self.wait(1)
        self.play(
            huffman_chart.bar_labels[2:].animate.set_opacity(0.2),
            huffman_chart.bars[2:].animate.set_opacity(0.2),
            huffman_chart.segments.bars[2:].animate.set_opacity(0.2),
            huffman_chart.event_labels[2:].animate.set_opacity(0.2),
            huffman_chart.probability_labels[2:].animate.set_opacity(0.2),
            huffman_chart.bar_labels[1][1].animate.set_color(PURE_GREEN)
        , run_time = 0.3)
        self.play(FadeIn(VGroup(rects_2[0], arrows_2[0])))
        self.wait(0.5)

        # Do the second chunk
        self.play(
            AnimationGroup(*[
                bit.animate.set_color(PURE_GREEN if i == 2 else PINK).set_opacity(0.2 if i > 2 else 1)
                for i, bit in enumerate(bits_2)
            ]),
            huffman_chart.animate.restore()
        )
        self.wait(1)
        self.play(
            huffman_chart.bar_labels[1:].animate.set_opacity(0.2),
            huffman_chart.bars[1:].animate.set_opacity(0.2),
            huffman_chart.segments.bars[1:].animate.set_opacity(0.2),
            huffman_chart.event_labels[1:].animate.set_opacity(0.2),
            huffman_chart.probability_labels[1:].animate.set_opacity(0.2),
            huffman_chart.bar_labels[0][0].animate.set_color(PURE_GREEN)
        , run_time = 0.3)
        self.play(FadeIn(VGroup(rects_2[1], arrows_2[1])))

        # Do the third chunk
        self.play(
            AnimationGroup(*[
                bit.animate.set_color(PURE_GREEN if i == 3 else PINK).set_opacity(0.2 if i > 3 else 1)
                for i, bit in enumerate(bits_2)
            ]),
            huffman_chart.animate.restore()
        )
        self.wait(1)
        self.play(
            huffman_chart.bar_labels[0].animate.set_opacity(0.2),
            huffman_chart.bars[0].animate.set_opacity(0.2),
            huffman_chart.segments.bars[0].animate.set_opacity(0.2),
            huffman_chart.event_labels[0].animate.set_opacity(0.2),
            huffman_chart.probability_labels[0].animate.set_opacity(0.2),
            AnimationGroup(*[huffman_chart.bar_labels[i][0].animate.set_color(PURE_GREEN) for i in range(1, 4)])
        , run_time = 0.3)
        self.wait(0.5)
        self.play(bits_2[4].animate.set_color(PURE_GREEN).set_opacity(1))
        self.play(
            huffman_chart.bar_labels[1].animate.set_opacity(0.2),
            huffman_chart.bars[1].animate.set_opacity(0.2),
            huffman_chart.segments.bars[1].animate.set_opacity(0.2),
            huffman_chart.event_labels[1].animate.set_opacity(0.2),
            huffman_chart.probability_labels[1].animate.set_opacity(0.2),
            AnimationGroup(*[huffman_chart.bar_labels[i][1].animate.set_color(PURE_GREEN) for i in range(2, 4)])
        , run_time = 0.3)
        self.wait(0.5)
        self.play(bits_2[5].animate.set_color(PURE_GREEN).set_opacity(1))
        self.play(
            huffman_chart.bar_labels[3].animate.set_opacity(0.2),
            huffman_chart.bars[3].animate.set_opacity(0.2),
            huffman_chart.segments.bars[3].animate.set_opacity(0.2),
            huffman_chart.event_labels[3].animate.set_opacity(0.2),
            huffman_chart.probability_labels[3].animate.set_opacity(0.2),
            huffman_chart.bar_labels[2][2].animate.set_color(PURE_GREEN)
        , run_time = 0.3)
        self.play(FadeIn(VGroup(rects_2[2], arrows_2[2])))

        # Show all the chunks being decoded
        self.play(
            huffman_chart.animate.restore(),
            bits_2.animate.set_color(PINK).set_opacity(1),
            FadeIn(rects_2[3:]),
            FadeIn(arrows_2[3:].set_opacity(1))
        )
        bits_2.resume_updating()
        arrows_2.resume_updating()
        rects_2.resume_updating()
        self.add(bits_2, arrows_2, rects_2)
        self.play(
            self.camera.frame.animate(run_time = 3).restore(),
            rects_2_opacity_tracker.animate.set_value(1),
            bits_2.animate(run_time = 10).align_to(robot.get_center(), RIGHT)
        )

        # Fade out everything but the chart
        self.remove(bits_2, rects_2, arrows_2)
        self.play(
            FadeOut(code_words_text),
            FadeOut(Group(robot_2, mission_control_2), shift = DOWN),
            huffman_chart.animate.scale(1.2).align_to(huffman_chart, RIGHT).set_y(0)
        )
        self.wait(3)

        # Write two questions
        questions = BulletedList(
            "How much more efficient?",
            "Prove optimality",
            numbered = True,
            font_size = 40
        ).to_edge(LEFT, buff = 1)
        for question in questions:
            self.play(Write(question), run_time = 2)
            self.wait(3)

        # Center the chart
        self.play(
            FadeOut(questions, shift = LEFT*2),
            huffman_chart.animate.scale(1.2).set_x(0).to_edge(DOWN, buff = 1)
        , run_time = 2)

        # Calculate the efficiency of the Huffman code
        self.wait(2)
        weighted_sum_lines = VGroup(
            Tex(R"\frac{1}{2} \cdot 1", font_size = 30).next_to(huffman_chart.bars[0], UP),
            Tex(R"\frac{1}{4} \cdot 2", font_size = 30).next_to(huffman_chart.bars[1], UP),
            Tex(R"\frac{1}{8} \cdot 3", font_size = 30).next_to(huffman_chart.bars[2], UP),
            Tex(R"\frac{1}{8} \cdot 3", font_size = 30).next_to(huffman_chart.bars[3], UP)
        )
        self.play(
            AnimationGroup(
                TransformFromCopy(huffman_chart.probability_labels[0], weighted_sum_lines[0][:-2]),
                FadeIn(weighted_sum_lines[0][-2:], shift = UP*0.1)
            , run_time = 1.5)
        )
        self.wait(1.5)
        self.play(
            AnimationGroup(
                TransformFromCopy(huffman_chart.probability_labels[1], weighted_sum_lines[1][:-2]),
                FadeIn(weighted_sum_lines[1][-2:], shift = UP*0.1)
            , run_time = 1.5)
        )
        self.wait(2.5)
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    TransformFromCopy(label, line[:-2]),
                    FadeIn(line[-2:], shift = UP*0.1)
                , run_time = 1.5)
                for label, line in zip(huffman_chart.probability_labels[2:], weighted_sum_lines[2:])
            ], lag_ratio = 0.3)
        )
        self.wait(0.35)

        # Show the weighted sum result
        sum_result = Tex(
            R"\frac{1}{2} \cdot 1 + \frac{1}{4} \cdot 2 + \frac{1}{8} \cdot 3 + \frac{1}{8} \cdot 3 \\ = 1.75 \text{ bits}",
            font_size = 49
        )
        sum_result[-9:].align_to(sum_result[:-9], RIGHT)
        sum_result.to_edge(RIGHT, buff = 1.2)

        self.play(
            huffman_chart.animate.scale(0.8).to_edge(LEFT, buff = 1),
            TransformMatchingShapes(weighted_sum_lines, sum_result[:-9], path_arc = PI*0.2, run_time = 1.5)
        )
        self.wait(0.5)
        self.play(FadeIn(sum_result[R"= 1.75 \text{ bits}"]))

        # Compare to the naive encoding
        huffman_group = VGroup(huffman_chart, sum_result)
        naive_sum_result = Tex(
            R"\frac{1}{2} \cdot 2 + \frac{1}{4} \cdot 2 + \frac{1}{8} \cdot 2 + \frac{1}{8} \cdot 2 \\ = 2 \text{ bits}",
            font_size = 49
        )
        naive_sum_result[-6:].align_to(naive_sum_result[:-6], RIGHT)
        naive_sum_result.to_edge(RIGHT, buff = 1.2)
        naive_chart.match_width(huffman_chart).move_to(huffman_chart)
        for label in naive_chart.bar_labels:
            label.match_height(huffman_chart.bar_labels[0])
        naive_group = VGroup(naive_chart, naive_sum_result).to_edge(UP, buff = 1)

        huffman_group.generate_target()
        huffman_group.target[0].scale(0.7)
        huffman_group.target.arrange(buff = 1).to_edge(DOWN, buff = 0.7)
        naive_group[0].scale(0.7)
        naive_group.arrange(buff = 1).to_edge(UP, buff = 0.7)
        naive_group.to_edge(UP, buff = 0.7)
        
        huffman_group.save_state()
        self.play(
            MoveToTarget(huffman_group),
            FadeIn(naive_group, shift = DOWN)
        , run_time = 1.5)
        self.wait(3)

        # Bring the huffman code back to focus
        self.play(
            huffman_group.animate.restore(),
            FadeOut(naive_group, shift = UP)
        , run_time = 1.5)
        self.wait(3)

        # Put away the calculation and focus on the chart
        self.play(FadeOut(sum_result, shift = RIGHT*3), huffman_chart.animate.set_x(0).to_edge(DOWN, buff = 0.7), run_time = 1.5)
        self.wait(3)

        # Show a message that looks like random noise
        mission_control_and_robot_group = Group(
            robot.scale(0.8).to_edge(LEFT, buff = 0.7),
            mission_control.scale(0.8).to_edge(RIGHT, buff = 0.7)
        ).set_x(0).to_edge(UP, buff = 0.5)
        robot = mission_control_and_robot_group[0]
        mission_control = mission_control_and_robot_group[1]
        message = VGroup(*[
            Integer(0).set_color(YELLOW)
            for _ in range(42)
        ]).arrange(
            buff = 0.08
        ).set_width(
            mission_control_and_robot_group.get_width()*0.76
        ).match_y(
            mission_control
        ).match_x(
            mission_control_and_robot_group
        ).shift(
            RIGHT*0.1
        )
        bit_opacity_trackers = [ValueTracker(0) for _ in message]
        for i, bit in enumerate(message):
            def update_bit(m, i = i):
                m.set_value(random.choice([0, 1])).set_opacity(bit_opacity_trackers[i].get_value())
            bit.add_updater(update_bit)
        self.add(message)

        bit_string = "•"*len(message)
        dummy_message = VGroup(*[
            Text(b, font_size = 30) for b in bit_string]
        ).set_color(PINK).arrange(buff = 0.08).match_width(message).match_y(message).align_to(mission_control, LEFT)
        dummy_message_opacity_tracker = ValueTracker(0)
        def update_dummy_message(m):
            for bit in m:
                bit.set_opacity(dummy_message_opacity_tracker.get_value()*min(1, max(0, 0.8*(mission_control.get_left()[0] - bit.get_x()))))
                if bit.get_x() < robot.get_x():
                    bit.set_opacity(0)
            self.bring_to_front(robot)
        dummy_message.add_updater(update_dummy_message)
        self.add(dummy_message)

        self.play(
            FadeIn(mission_control_and_robot_group),
            dummy_message.animate(run_time = 6).move_to(message),
            dummy_message_opacity_tracker.animate(run_time = 2).set_value(1)
        )
        dummy_message.clear_updaters()

        self.play(
            FadeOut(dummy_message),
            AnimationGroup(*[t.animate.set_value(1) for t in bit_opacity_trackers])
        )
        self.wait(1)

        # Show the probability of the first bit being 0 vs 1
        arrow = Arrow(ORIGIN, UP).set_color(WHITE).next_to(message[0], DOWN)
        huffman_chart.save_state()
        self.play(
            huffman_chart.bars[1:].animate.set_opacity(0.2),
            huffman_chart.bar_labels[1:].animate.set_opacity(0.2),
            huffman_chart.segments.bars[1:].animate.set_opacity(0.2),
            huffman_chart.event_labels[1:].animate.set_opacity(0.2),
            huffman_chart.probability_labels[1:].animate.set_opacity(0.2)
        )
        self.wait(2)
        self.play(
            GrowArrow(arrow),
            AnimationGroup(*[t.animate.set_value(0.2) for t in bit_opacity_trackers[1:]]),
        )
        message[0].suspend_updating()
        message[0].set_value(0)
        self.play(message[0].animate.set_color(PURE_GREEN))
        self.wait(1.5)
        self.play(
            message[0].animate.set_value(1),
            huffman_chart.animate.restore(),
            huffman_chart.bars[0].animate.set_opacity(0.2),
            huffman_chart.bar_labels[0].animate.set_opacity(0.2),
            huffman_chart.segments.bars[0].animate.set_opacity(0.2),
            huffman_chart.event_labels[0].animate.set_opacity(0.2),
            huffman_chart.probability_labels[0].animate.set_opacity(0.2)
        )
        self.wait(3)

        # Show the probability of the second bit being 0 vs 1
        arrow2 = Arrow(ORIGIN, UP).set_color(WHITE).next_to(message[1], DOWN)
        self.play(
            FadeOut(arrow),
            huffman_chart.bars[2:].animate.set_opacity(0.2),
            huffman_chart.bar_labels[2:].animate.set_opacity(0.2),
            huffman_chart.segments.bars[2:].animate.set_opacity(0.2),
            huffman_chart.event_labels[2:].animate.set_opacity(0.2),
            huffman_chart.probability_labels[2:].animate.set_opacity(0.2)
        )
        self.wait(2)
        self.play(
            GrowArrow(arrow2),
            bit_opacity_trackers[1].animate.set_value(1),
        )
        message[1].suspend_updating()
        message[1].set_value(0)
        self.play(message[1].animate.set_color(PURE_GREEN))
        self.wait(1.2)
        self.play(
            message[1].animate.set_value(1),
            huffman_chart.animate.restore(),
            huffman_chart.bars[:2].animate.set_opacity(0.2),
            huffman_chart.bar_labels[:2].animate.set_opacity(0.2),
            huffman_chart.segments.bars[:2].animate.set_opacity(0.2),
            huffman_chart.event_labels[:2].animate.set_opacity(0.2),
            huffman_chart.probability_labels[:2].animate.set_opacity(0.2)
        )
        self.wait(3)

        # Show the probability of the third bit being 0 vs 1
        arrow3 = Arrow(ORIGIN, UP).set_color(WHITE).next_to(message[2], DOWN)
        self.play(
            FadeOut(arrow2),
            GrowArrow(arrow3),
            bit_opacity_trackers[2].animate.set_value(1),
            huffman_chart.bars[3].animate.set_opacity(0.2),
            huffman_chart.bar_labels[3].animate.set_opacity(0.2),
            huffman_chart.segments.bars[3].animate.set_opacity(0.2),
            huffman_chart.event_labels[3].animate.set_opacity(0.2),
            huffman_chart.probability_labels[3].animate.set_opacity(0.2)
        )
        message[2].suspend_updating()
        message[2].set_value(0)
        self.play(message[2].animate.set_color(PURE_GREEN))
        self.wait(1.2)
        self.play(
            message[2].animate.set_value(1),
            huffman_chart.animate.restore(),
            huffman_chart.bars[:3].animate.set_opacity(0.2),
            huffman_chart.bar_labels[:3].animate.set_opacity(0.2),
            huffman_chart.segments.bars[:3].animate.set_opacity(0.2),
            huffman_chart.event_labels[:3].animate.set_opacity(0.2),
            huffman_chart.probability_labels[:3].animate.set_opacity(0.2)
        )
        self.wait(1.2)
        self.play(message[:3].animate.set_color(YELLOW)),
        message[0].resume_updating()
        message[1].resume_updating()
        message[2].resume_updating()
        self.play(
            FadeOut(arrow3),
            huffman_chart.animate.restore(),
            AnimationGroup(*[t.animate.set_value(1) for t in bit_opacity_trackers])
        )
        self.wait(10)

        # Zoom in on the receiver
        self.camera.frame.save_state()
        huffman_chart.unfix_from_frame()
        self.play(self.camera.frame.animate.scale(0.75, about_point = robot.get_corner(UL)), run_time = 3)
        self.wait(5)

        # Zoom out to think about entire messages
        n = 6
        robot.generate_target()
        robot.target.scale(2)
        sample_n_bits = message[:n]
        for bit, val in zip(sample_n_bits, "110101"):
            bit.set_value(int(val))
        sample_n_bits.clear_updaters()
        sample_n_bits.generate_target()
        sample_n_bits.target.scale(1.2).next_to(robot.target, RIGHT, buff = 0.4)
        Group(robot.target, sample_n_bits.target).center()
        right_shift_tracker = ValueTracker(0)
        for bit in message[n:]:
            original_x = bit.get_x()
            def update_bit(m, original_x = original_x):
                m.set_x(original_x)
                m.shift(RIGHT*right_shift_tracker.get_value())
            bit.add_updater(update_bit)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    self.camera.frame.animate(run_time = 3).restore(),
                    FadeOut(Group(mission_control, huffman_chart), shift = DR*1.4, run_time = 1.3),
                    right_shift_tracker.animate(run_time = 1.3).set_value(4),
                    AnimationGroup(*[t.animate.set_value(0) for t in bit_opacity_trackers[n:]], run_time = 1.3)
                ),
                AnimationGroup(
                    MoveToTarget(robot, run_time = 3),
                    MoveToTarget(sample_n_bits, run_time = 3)
                )
            )
        )
        self.remove(message[n:])
        self.wait(2)

        # Brace the n bits
        brace = Brace(sample_n_bits, direction = DOWN)
        label = brace.get_tex(R"n \text{ bits}")
        self.play(GrowFromEdge(brace, UP), Write(label))

        # Show all of the 2^n messages
        messages = VGroup(*[
            Tex(F"{i:0{n}b}", font_size = 40).set_color(YELLOW)
            for i in range(2**n)
        ]).arrange_in_grid(n_cols = 4, h_buff = 0.3, v_buff = 0.1).align_to(sample_n_bits, LEFT)
        message_index = sum([2**i*sample_n_bits[n - 1 - i].get_value() for i in range(n)])
        messages[message_index].set_color(BLUE)
        new_brace = Brace(messages, direction = DOWN)
        new_label = new_brace.get_tex(R"n \text{ bits}")
        VGroup(messages, new_brace, new_label).set_y(0)
        two_to_then_n_brace = Brace(messages, direction = RIGHT)
        two_to_the_n_label = two_to_then_n_brace.get_tex(R"2^n\ n\text{-bit messages} \\ \text{(all equally likely)}")
        two_to_the_n_label.shift(DOWN*(two_to_the_n_label[0].get_y() - two_to_then_n_brace.get_y()))
        self.play(
            self.camera.frame.animate(run_time = 3).match_x(Group(robot, two_to_the_n_label)),
            FadeOut(VGroup(brace, label).fix_in_frame(), shift = DOWN),
            AnimationGroup(
                AnimationGroup(*[
                    TransformMatchingShapes(sample_n_bits[i], messages[message_index][i], run_time = 2)
                    for i in range(n)
                ]),
                AnimationGroup(*[FadeIn(message, shift = DOWN*0.3) for message in messages[:message_index]], lag_ratio = 0.02),
                AnimationGroup(*[FadeIn(message, shift = UP*0.3) for message in messages[message_index + 1:]], lag_ratio = 0.02),
                AnimationGroup(
                    GrowFromEdge(two_to_then_n_brace, LEFT),
                    Write(two_to_the_n_label[R"2^n\ n\text{-bit messages}"])
                )
            , lag_ratio = 0.2)
        )
        self.add(messages)
        self.wait(2)
        self.play(FadeIn(two_to_the_n_label["(all equally likely)"]))
        self.wait(2)

        # View the chart again
        self.play(
            FadeOut(VGroup(messages, two_to_then_n_brace, two_to_the_n_label)),
            FadeIn(
                huffman_chart.scale(0.8).next_to(
                    self.camera.frame.get_right(), LEFT, buff = robot.get_left()[0] - self.camera.frame.get_left()[0]
                ),
                shift = LEFT
            )
        )
        self.camera.frame.center()
        Group(robot, huffman_chart).move_to(self.camera.frame)

        # Show some sample messages
        message_instructions = [
            [0, 0, 0],
            [3],
            [1, 0]
        ]
        messages = VGroup(*[
            VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]).set_color(PINK) for i in row]).arrange(buff = 3)
            for row in message_instructions
        ]).arrange_in_grid(n_cols = 1, aligned_edge = RIGHT, buff = 5).set_width(2).next_to(robot, RIGHT, buff = 1.85)
        for message in messages:
            self.play(FadeIn(message))
            self.wait(1.5)
        self.wait(1)

        # Show their encodings
        encodings = VGroup(*[
            Tex(R":\ " + "".join([encoding[i] for i in row]), font_size = 50).set_color(PINK)
            for row in message_instructions
        ])
        for message, enc, in zip(messages, encodings):
            enc[":"].match_y(enc[-1])
            enc.next_to(message, RIGHT, buff = 0.3)
        for message in messages:
            message.generate_target()
        VGroup(*[message.target for message in messages], encodings).match_x(messages)
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    MoveToTarget(message, run_time = 1.5),
                    FadeIn(enc[":"], shift = LEFT, run_time = 1.5),
                    AnimationGroup(*[
                        TransformFromCopy(huffman_chart.bar_labels[row[j]], enc[1:][
                            sum([len(encoding[row[k]]) for k in range(j)]):sum([len(encoding[row[k]]) for k in range(j + 1)])
                        ], path_arc = PI*0.2)
                        for j in range(len(row))
                    ], lag_ratio = 0.1, run_time = 3)
                )
                for message, row, enc in zip(messages, message_instructions, encodings)
            ], lag_ratio = 0.05)
        )

        # Show the probability of each
        probability_calculations = VGroup(
            Tex(R"\frac{1}{2}\ \ \cdot\ \  \frac{1}{2}\ \  \cdot\ \  \frac{1}{2}", font_size = 30),
            Tex(R"\frac{1}{8}", font_size = 30),
            Tex(R"\frac{1}{4}\ \  \cdot\ \  \frac{1}{2}", font_size = 30)
        )
        for calculation, message in zip(probability_calculations, messages):
            calculation.next_to(message, UP)
        

        self.play(
            AnimationGroup(*[
                AnimationGroup(*[
                    TransformFromCopy(huffman_chart.probability_labels[row[j]], calculation[j*4: j*4 + 3], path_arc = PI*0.2)
                    for j in range(len(row))
                ], lag_ratio = 0.1)
                for calculation, row in zip(probability_calculations, message_instructions)
            ], lag_ratio = 0.1, run_time = 2.5)
        )
        self.play(AnimationGroup(*[FadeIn(calculation[R"\cdot"]) for calculation in probability_calculations]))
        self.wait(1.5)

        # Highlight the different sizes of the messages
        for _ in range(2):
            self.play(
                AnimationGroup(*[
                    AnimationGroup(*[
                        Indicate(arrow)
                        for arrow in message
                    ], lag_ratio = 0.3, run_time = 2)
                    for message in messages
                ])
            )

        # Highlight the same size of the encoded messages and relate it to the probability
        rect = SurroundingRectangle(VGroup(*[enc[1:] for enc in encodings]), stroke_width = 4, buff = 0.2, stroke_color = YELLOW)
        self.play(FadeIn(rect), run_time = 2)
        rects = VGroup(*[
            SurroundingRectangle(calculation, stroke_width = 3, buff = 0.1, stroke_color = YELLOW)
            for calculation in probability_calculations
        ])
        self.play(FadeIn(rects), run_time = 2)
        self.wait(2)


class RobotEncodingsV2(InteractiveScene):
    def construct(self):
        # Add mission_control and the robot
        mission_control = ImageMobject(
            "images/pi_creature_mission_control.png"
        ).set_width(2).to_edge(RIGHT, buff = 0.4).to_edge(DOWN, buff = 0.6)
        robot = ImageMobject(
            "images/lunar_rover_assets/stationary.png"
        ).match_height(mission_control).to_edge(LEFT, buff = 0.4).align_to(mission_control, DOWN)
        self.add(mission_control, robot)
        self.wait(2)

        # Create a stream of bits flowing towards the bot, and decode them into instructions by chunks of 2
        distribution = [1/2, 1/4, 1/8, 1/8]
        instructions = [1, 0, 2] # Start with DOWN, UP, LEFT = 100110 with perfect encoding
        instructions += generate_random_instructions(13, distribution, seed = 7) # 16 total instructions
        instructions[-3] = 3
        bit_string = ""
        for instruction in instructions:
            bit_string += f"{instruction:02b}"
        bit_buff = 0.1
        bits = VGroup(*[
            Tex(bit_string[i], font_size = 32)
            for i in range(len(bit_string))
        ]).arrange(buff = bit_buff).set_color(YELLOW).match_y(robot).align_to(mission_control, LEFT)
        bits_opacity_tracker = ValueTracker(0)
        def update_bits(m):
            for bit in m:
                bit.set_opacity(bits_opacity_tracker.get_value()*min(1, max(0, 0.8*(mission_control.get_left()[0] - bit.get_x()))))
                if bit.get_x() < robot.get_x():
                    bit.set_opacity(0)
            self.bring_to_front(robot)
        bits.add_updater(update_bits)

        rects = VGroup(*[
            SurroundingRectangle(bits[i:i + 2], stroke_width = 2, stroke_color = WHITE, buff = bit_buff*0.5)
            for i in range(0, len(bit_string), 2)
        ])
        for rect in rects:
            rect.stretch_to_fit_height(rects[0].get_height())
        rect_opacity_trackers = [ValueTracker(0) for _ in range(4)]
        def update_rects(m):
            for i, rect in enumerate(m):
                target_bits = bits[2*i:2*(i + 1)]
                rect.match_x(target_bits)
                rect.set_stroke(
                    opacity = rect_opacity_trackers[
                        instructions[i]
                    ].get_value()*min(1, max(0, 0.8 * (mission_control.get_left()[0] - (rect.get_x() + 0.2))))
                )
                if rect.get_x() < robot.get_x():
                    rect.set_stroke(opacity = 0)
        rects.add_updater(update_rects)

        arrows = VGroup(*[
            InstructionArrow([UP, DOWN, LEFT, RIGHT][instructions[i]]).scale(0.07).move_to(rects[i]).shift(UP*0.62)
            for i in range(len(instructions))
        ])
        arrows_opacity_tracker = ValueTracker(1)
        def update_arrows(m):
            for i, arrow in enumerate(m):
                target_bits = bits[2*i:2*(i + 1)]
                arrow.match_x(target_bits)
                opacity = min(1, max(0, 0.8*(mission_control.get_left()[0] - (arrow.get_x() + 0.2))))
                if arrow.get_x() < robot.get_right()[0]:
                    opacity = min(1, max(0, 1 - 1.2*(robot.get_right()[0] - arrow.get_x())))
                arrow.set_opacity(arrows_opacity_tracker.get_value()*opacity)
        arrows.add_updater(update_arrows)

        self.add(bits, rects, arrows)
        self.play(
            AnimationGroup(
                bits.animate(run_time = 40, rate_func = smooth).align_to(robot.get_right() + RIGHT*0.2, LEFT),
                bits_opacity_tracker.animate.set_value(1)
            , lag_ratio = 0.65, rate_func = linear)
        )

        # Show the naive encoding
        table = VGroup(*[
            VGroup(
                Tex(f"{i:02b}").set_color(YELLOW),
                Tex(":"),
                InstructionArrow(direction = [UP, DOWN, LEFT, RIGHT][i]).scale(0.15)
            ).arrange()
            for i in range(4)
        ])
        table.arrange(DOWN)
        for row in table:
            row.shift(RIGHT*(table[0][1].get_x() - row[1].get_x()))
        for i, row in enumerate(table):
            row[2].match_x(table[2][2])
            row.set_y(table[0].get_y() + i*(table[1].get_y() - table[0].get_y()))
        table.set_width(2).to_edge(RIGHT, buff = 2).to_edge(UP, buff = 0.7)
        table_box = SurroundingRectangle(table, buff = 0.15).set_color(WHITE).set_stroke(opacity = 0)

        arrows.save_state()
        for i, row in enumerate(table):
            anims = []
            anims.append(FadeIn(row))
            anims.append(rect_opacity_trackers[i].animate.set_value(1))
            if i == 0:
                anims.append(table_box.animate.set_stroke(opacity = 1))
            else:
                anims.append(rect_opacity_trackers[i - 1].animate(run_time = 1).set_value(0))
                anims.append(
                    AnimationGroup(*[
                        arrow.animate(run_time = 1).set_color(WHITE if instructions[j] == i else YELLOW)
                        for j, arrow in enumerate(arrows)
                    ])
                )
            self.play(AnimationGroup(*anims, run_time = 2))
        self.wait(1)
        self.play(arrows.animate.restore(), run_time = 2)
        self.wait(2)    

        self.play(AnimationGroup(*[t.animate.set_value(1) for t in rect_opacity_trackers]), run_time = 2)

        # Transform the table into a stacked bar diagram
        naive_chart = EntropyChart(
            distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]),
            probability_labels = [
                Tex(R"\frac{1}{" + ["2", "4", "8", "8"][i] + "}", font_size = 80)
                for i in range(4)
            ],
            bar_labels = [
                Tex(["00", "01", "10", "11"][i], font_size = 90)
                for i in range(4)
            ],
            bar_heights = [2, 2, 2, 2],
            width = 9,
            include_vertical_axis = False,
            segments_height = 1,
            fill_colors = [YELLOW_B, YELLOW_D]
        ).match_height(table).scale(0.5).match_y(table).align_to(table, RIGHT).shift(RIGHT)
        naive_chart.bar_labels.set_color(WHITE)
        naive_chart.update()
        self.wait(0.01)
        naive_chart.suspend_updating()
        self.play(
            AnimationGroup(
                FadeOut(VGroup(table_box, *[row[1] for row in table])),
                AnimationGroup(
                    FadeIn(naive_chart.segments.bars),
                    AnimationGroup(*[
                        ReplacementTransform(table[i][2], naive_chart.event_labels[i])
                        for i in range(len(table))
                    ]),
                    FadeIn(naive_chart.probability_labels),
                    AnimationGroup(*[
                        ReplacementTransform(table[i][0], naive_chart.bar_labels[i])
                        for i in range(len(table))
                    ])
                )
            , lag_ratio = 0.4)
        )
        self.play(
            AnimationGroup(
                naive_chart.create_bars(),
                naive_chart.bar_labels.animate.set_color(BLACK)
            , lag_ratio = 0.5)
        )
        self.wait(1.8)

        # Contrast the naive example vs the better one
        encoding = ["0", "10", "110", "111"]
        huffman_chart = EntropyChart(
            distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]).set_color(PINK),
            probability_labels = [
                Tex(R"\frac{1}{" + ["2", "4", "8", "8"][i] + "}", font_size = 80)
                for i in range(4)
            ],
            bar_labels = [
                Tex(encoding[i], font_size = 60)
                for i in range(4)
            ],
            bar_heights = [1, 2, 3, 3],
            width = 9,
            height = 6,
            include_vertical_axis = False,
            segments_height = 1,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).match_width(naive_chart)
        huffman_chart.bar_labels.set_color(WHITE)
        huffman_chart.update()
        self.wait(0.01)
        naive_chart.clear_updaters()
        huffman_chart.clear_updaters()

        naive_chart.generate_target()
        VGroup(naive_chart.target, huffman_chart).arrange(buff = 3)
        huffman_chart.align_to(naive_chart.target, DOWN)
        rects.clear_updaters()
        arrows.clear_updaters()
        bits.clear_updaters()
        naive_encoding_text = TexText("Naive").next_to(naive_chart.target, UP, buff = 1.4)
        new_encoding_text = TexText("Optimized").match_x(huffman_chart).match_y(naive_encoding_text)
        VGroup(naive_chart.target, huffman_chart, naive_encoding_text, new_encoding_text).set_y(0)
        arrow = Arrow(
            naive_chart.target.bars.get_right() + RIGHT*0.2,
            [huffman_chart.bars.get_left()[0] - 0.2, naive_chart.target.bars.get_y(), 0]
        )
        huffman_chart.bars.set_opacity(0.2)
        self.play(
            AnimationGroup(
                FadeOut(Group(robot, rects, arrows, bits, mission_control)),
                AnimationGroup(MoveToTarget(naive_chart, path_arc = PI*0.2), FadeIn(naive_encoding_text), lag_ratio = 0.8),
                GrowArrow(arrow),
                AnimationGroup(
                    AnimationGroup(*[
                        FadeIn(VGroup(segment, arrow, prob, bar))
                        for segment, arrow, prob, bar in zip(
                            huffman_chart.segments.bars,
                            huffman_chart.event_labels,
                            huffman_chart.probability_labels,
                            huffman_chart.bars
                        )
                    ], lag_ratio = 0.2),
                    FadeIn(new_encoding_text)
                , lag_ratio = 0.3)
            , lag_ratio = 0.4)
        , run_time = 6.5)
        self.wait(2)
        circ = Circle(
            radius = 0.47, fill_opacity = 0, stroke_width = 3, stroke_color = GREEN
        ).move_to(huffman_chart.probability_labels[0])
        self.play(ShowCreation(circ))
        self.play(FadeOut(circ))
        self.wait(2)

        for i in range(4):
            self.play(huffman_chart.bars[i].animate.set_opacity(1), FadeIn(huffman_chart.bar_labels[i]), run_time = 2)
            self.wait(2)
        self.wait(2)

        # Compare the longer strings from the new code with the naive code
        naive_chart.save_state()
        huffman_chart.save_state()
        arrow.save_state()
        self.play(
            VGroup(
                arrow,
                huffman_chart.bars[:2],
                huffman_chart.bar_labels[:2],
                huffman_chart.segments.bars[:2],
                huffman_chart.event_labels[:2],
                huffman_chart.probability_labels[:2]
            ).animate.fade(0.8),
            naive_chart.animate.fade(0.8)
        )
        self.wait(2)
        naive_chart.generate_target()
        naive_chart.target.restore()
        VGroup(
            naive_chart.target.bars[:2],
            naive_chart.target.bar_labels[:2],
            naive_chart.target.segments.bars[:2],
            naive_chart.target.event_labels[:2],
            naive_chart.target.probability_labels[:2]
        ).fade(0.8)
        self.play(MoveToTarget(naive_chart))
        self.wait(2)
        naive_chart.generate_target()
        naive_chart.target.restore()
        VGroup(
            naive_chart.target.bars[1:],
            naive_chart.target.bar_labels[1:],
            naive_chart.target.segments.bars[1:],
            naive_chart.target.event_labels[1:],
            naive_chart.target.probability_labels[1:]
        ).fade(0.8)
        huffman_chart.generate_target()
        huffman_chart.target.restore()
        VGroup(
            huffman_chart.target.bars[1:],
            huffman_chart.target.bar_labels[1:],
            huffman_chart.target.segments.bars[1:],
            huffman_chart.target.event_labels[1:],
            huffman_chart.target.probability_labels[1:]
        ).fade(0.8)
        self.play(MoveToTarget(naive_chart), MoveToTarget(huffman_chart))
        self.wait(2)
        self.play(arrow.animate.restore(), naive_chart.animate.restore(), huffman_chart.animate.restore(), run_time = 2)

        # Show the sample sequence of instructions again
        self.play(
            FadeOut(arrow, run_time = 1),
            VGroup(naive_encoding_text, new_encoding_text).animate(run_time = 2).to_edge(UP, buff = 0.5),
            VGroup(naive_chart, huffman_chart).animate(run_time = 2).shift(UP*1.7)
        )
        sample_instruction_arrows = arrows.copy().set_color(WHITE).set_width(FRAME_WIDTH*0.75).to_edge(DOWN, buff = 1.5).set_x(0)
        self.play(
            AnimationGroup(*[
                FadeIn(arrow, shift = UP*0.2)
                for arrow in sample_instruction_arrows
            ], lag_ratio = 0.1)
        )
        self.wait(2)

        # Highlight the ups, downs, lefts, and rights
        for direction in [UP, DOWN, LEFT, RIGHT]:
            self.play(
                AnimationGroup(*[
                    arrow.animate.set_opacity(1 if (arrow.direction == direction).all() else 0.1)
                    for arrow in sample_instruction_arrows
                ])
            )
            self.wait(2)
        self.play(sample_instruction_arrows.animate.set_opacity(1))

        # Show the two examples side by side
        arrows.clear_updaters()
        rects.clear_updaters()
        bits.clear_updaters()
        naive_chart.generate_target()
        naive_example_group = Group(robot, Group(arrows, rects, bits), mission_control, naive_chart.target)
        naive_example_group.scale(1.3)
        naive_example_group.arrange(buff = 0.6).to_edge(UP, buff = 0.7).set_width(FRAME_WIDTH*0.96)
        naive_example_group[1].shift(
            DOWN*(naive_example_group[1][2].get_y() - naive_example_group.get_y())
        )
        naive_example_group[2].align_to(naive_example_group[0], DOWN)
        naive_example_group[3].shift(DOWN*0.4)
        naive_example_group.to_edge(UP, buff = 0.7).set_x(0)

        robot_2 = robot.copy()
        arrows_2 = arrows.copy().set_color(PINK)
        mission_control_2 = mission_control.copy()
        huffman_chart.generate_target()
        huffman_chart.target.match_width(
            naive_chart.target
        ).match_x(
            naive_chart.target
        ).shift(
            DOWN*(huffman_chart.segments.get_bottom()[1] - mission_control_2.get_bottom()[1])
        )
        Group(robot_2, arrows_2, mission_control_2, huffman_chart.target).to_edge(DOWN, buff = 0.4)

        self.play(
            AnimationGroup(
                AnimationGroup(
                    AnimationGroup(
                        MoveToTarget(naive_chart, path_arc = -PI*0.2),
                        naive_encoding_text.animate.scale(0.7).next_to(arrows, DOWN, buff = 1),
                        AnimationGroup(
                            ReplacementTransform(sample_instruction_arrows.copy().set_opacity(0), naive_example_group[1][0]),
                            AnimationGroup(
                                FadeIn(naive_example_group[0]),
                                FadeIn(naive_example_group[1][1:]),
                                FadeIn(naive_example_group[2])
                            )
                        , lag_ratio = 0.5)
                    , run_time = 2.5),
                    AnimationGroup(
                        MoveToTarget(huffman_chart, path_arc = -PI*0.2),
                        new_encoding_text.animate.scale(0.7).set_opacity(0.1).next_to(arrows_2, DOWN, buff = 1),
                        AnimationGroup(
                            ReplacementTransform(sample_instruction_arrows, arrows_2),
                            AnimationGroup(
                                FadeIn(robot_2),
                                FadeIn(mission_control_2)
                            )
                        , lag_ratio = 0.5)
                    , run_time = 2.5)
                )
            , lag_ratio = 0.4)
        )

        # Encode the new sequence with the Huffman code
        bit_string = ""
        for instruction in instructions:
            bit_string += encoding[instruction]
        bits_2 = VGroup(*[
            Tex(bit_string[i])
            for i in range(len(bit_string))
        ]).set_color(PINK).match_height(bits)
        bits_2.set_y(robot_2.get_y() + (bits.get_y() - robot.get_y()))
        def update_bits_2(m):
            for bit in m:
                if bit.get_x() < robot_2.get_x():
                    bit.set_opacity(0)
            self.bring_to_front(robot_2)
        bits_2.add_updater(update_bits_2)
        index = 0
        bit_groups = VGroup()
        for instruction, arrow in zip(instructions, arrows):
            code_word_length = 1 if instruction == 0 else 2 if instruction == 1 else 3
            target_bits = bits_2[index:index + code_word_length]
            target_bits.arrange(center = False, buff = 0).match_x(arrow)
            bit_groups.add(target_bits)
            index += code_word_length
        self.play(
            new_encoding_text.animate(run_time = 2).set_opacity(1),
            AnimationGroup(*[
                TransformMatchingShapes(huffman_chart.bar_labels[instructions[i]].copy(), grouping, path_arc = PI*0.1)
                for i, grouping in enumerate(bit_groups)
            ], lag_ratio = 0.08, run_time = 6)
        )
        self.wait(1)

        def update_arrows_2(m):
            index = 0
            for arrow, instruction in zip(m, instructions):
                code_word_length = 1 if instruction == 0 else 2 if instruction == 1 else 3
                target_bits = bits_2[index:index + code_word_length]
                arrow.match_x(target_bits)
                if arrow.get_x() < robot_2.get_right()[0]:
                    arrow.set_opacity(min(1, max(0, 1 - 1.2*(robot_2.get_right()[0] - arrow.get_x()))))
                index += code_word_length

        arrows_2.add_updater(update_arrows_2)

        rects_2 = VGroup(*[Rectangle() for _ in instructions])
        rects_2_opacity_tracker = ValueTracker(0)
        def update_rects_2(m):
            index = 0
            for i, rect in enumerate(m):
                instruction = instructions[i]
                code_word_length = 1 if instruction == 0 else 2 if instruction == 1 else 3
                target_bits = bits_2[index:index + code_word_length]
                rect.become(
                    SurroundingRectangle(
                        target_bits,
                        stroke_width = 2,
                        stroke_color = WHITE,
                        stroke_opacity = rects_2_opacity_tracker.get_value(),
                        buff = 0.5*(bits_2[1].get_left()[0] - bits_2[0].get_right()[0])
                    )
                )
                if i > 0:
                    rect.stretch_to_fit_height(rects_2[0].get_height())
                if rect.get_x() < robot_2.get_x():
                    rect.set_stroke(opacity = 0)
                index += code_word_length
        rects_2.add_updater(update_rects_2)
        self.add(rects_2)

        self.play(
            AnimationGroup(
                AnimationGroup(*[
                    bit_2.animate.match_x(bit)
                    for bit_2, bit in zip(bits_2, bits)
                ]),
                rects_2_opacity_tracker.animate.set_value(1)
            , lag_ratio = 0.5)
        )
        self.wait(0.5)
        bits_2.suspend_updating()
        rects_2.suspend_updating()
        arrows_2.suspend_updating()

        # Focus on the perfect example
        self.add(huffman_chart)
        perfect_example_group = Group(robot_2, Group(arrows_2, rects_2, bits_2), mission_control_2, huffman_chart)
        self.play(
            AnimationGroup(
                FadeOut(Group(naive_example_group[:-1], naive_chart), shift = UP, run_time = 2),
                FadeOut(VGroup(naive_encoding_text, new_encoding_text), shift = DOWN*0.2, run_time = 1),
                perfect_example_group[-1].animate(path_arc = PI*0.2, run_time = 2).scale(1.2).to_edge(UP, buff = 1).to_edge(RIGHT, buff = 1),
                perfect_example_group[:-1].animate(run_time = 2).set_width(FRAME_WIDTH*0.9).set_x(0).to_edge(DOWN, buff = 0.4)
            , lag_ratio = 0.1)
        )

        # Introduce "code word" term
        code_words_text = Tex(R"\text{code words}").next_to(huffman_chart, UP).set_color(WHITE).fix_in_frame()
        huffman_chart.clear_updaters()
        huffman_chart.fix_in_frame()
        huffman_chart.bar_labels.add_updater(lambda m: self.bring_to_front(m))
        huffman_chart.save_state()
        self.play(
            Write(code_words_text, run_time = 1.5),
            VGroup(huffman_chart.bars, huffman_chart.segments, huffman_chart.probability_labels).animate.fade(0.8)
        )

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale(0.7, about_point = robot_2.get_corner(DL) + DOWN*0.3).shift(UR*0.2),
            FadeOut(VGroup(arrows_2, rects_2))
        , run_time = 2)

        self.wait(1.2)
        huffman_chart.clear_updaters()
        self.play(huffman_chart.animate.restore(), run_time = 2)

        # Break down the first code word
        self.play(
            bits_2[0].animate.set_color(PURE_GREEN),
            bits_2[1:].animate.set_opacity(0.2)
        )  
        self.wait(1)
        huffman_chart.bar_labels.save_state()
        self.play(
            huffman_chart.bar_labels[0].animate.set_opacity(0.2),
            huffman_chart.bars[0].animate.set_opacity(0.2),
            huffman_chart.segments.bars[0].animate.set_opacity(0.2),
            huffman_chart.event_labels[0].animate.set_opacity(0.2),
            huffman_chart.probability_labels[0].animate.set_opacity(0.2),
            AnimationGroup(*[huffman_chart.bar_labels[i][0].animate.set_color(PURE_GREEN) for i in range(1, 4)])
        , run_time = 0.3)
        self.wait(1)
        self.play(bits_2[1].animate.set_opacity(1).set_color(PURE_GREEN))
        self.wait(1)
        self.play(
            huffman_chart.bar_labels[2:].animate.set_opacity(0.2),
            huffman_chart.bars[2:].animate.set_opacity(0.2),
            huffman_chart.segments.bars[2:].animate.set_opacity(0.2),
            huffman_chart.event_labels[2:].animate.set_opacity(0.2),
            huffman_chart.probability_labels[2:].animate.set_opacity(0.2),
            huffman_chart.bar_labels[1][1].animate.set_color(PURE_GREEN)
        , run_time = 0.3)
        self.play(FadeIn(VGroup(rects_2[0], arrows_2[0])))
        self.wait(0.5)

        # Do the second chunk
        self.play(
            AnimationGroup(*[
                bit.animate.set_color(PURE_GREEN if i == 2 else PINK).set_opacity(0.2 if i > 2 else 1)
                for i, bit in enumerate(bits_2)
            ]),
            huffman_chart.animate.restore()
        )
        self.wait(1)
        self.play(
            huffman_chart.bar_labels[1:].animate.set_opacity(0.2),
            huffman_chart.bars[1:].animate.set_opacity(0.2),
            huffman_chart.segments.bars[1:].animate.set_opacity(0.2),
            huffman_chart.event_labels[1:].animate.set_opacity(0.2),
            huffman_chart.probability_labels[1:].animate.set_opacity(0.2),
            huffman_chart.bar_labels[0][0].animate.set_color(PURE_GREEN)
        , run_time = 0.3)
        self.play(FadeIn(VGroup(rects_2[1], arrows_2[1])))

        # Do the third chunk
        self.play(
            AnimationGroup(*[
                bit.animate.set_color(PURE_GREEN if i == 3 else PINK).set_opacity(0.2 if i > 3 else 1)
                for i, bit in enumerate(bits_2)
            ]),
            huffman_chart.animate.restore()
        )
        self.wait(1)
        self.play(
            huffman_chart.bar_labels[0].animate.set_opacity(0.2),
            huffman_chart.bars[0].animate.set_opacity(0.2),
            huffman_chart.segments.bars[0].animate.set_opacity(0.2),
            huffman_chart.event_labels[0].animate.set_opacity(0.2),
            huffman_chart.probability_labels[0].animate.set_opacity(0.2),
            AnimationGroup(*[huffman_chart.bar_labels[i][0].animate.set_color(PURE_GREEN) for i in range(1, 4)])
        , run_time = 0.3)
        self.wait(0.5)
        self.play(bits_2[4].animate.set_color(PURE_GREEN).set_opacity(1))
        self.play(
            huffman_chart.bar_labels[1].animate.set_opacity(0.2),
            huffman_chart.bars[1].animate.set_opacity(0.2),
            huffman_chart.segments.bars[1].animate.set_opacity(0.2),
            huffman_chart.event_labels[1].animate.set_opacity(0.2),
            huffman_chart.probability_labels[1].animate.set_opacity(0.2),
            AnimationGroup(*[huffman_chart.bar_labels[i][1].animate.set_color(PURE_GREEN) for i in range(2, 4)])
        , run_time = 0.3)
        self.wait(0.5)
        self.play(bits_2[5].animate.set_color(PURE_GREEN).set_opacity(1))
        self.play(
            huffman_chart.bar_labels[3].animate.set_opacity(0.2),
            huffman_chart.bars[3].animate.set_opacity(0.2),
            huffman_chart.segments.bars[3].animate.set_opacity(0.2),
            huffman_chart.event_labels[3].animate.set_opacity(0.2),
            huffman_chart.probability_labels[3].animate.set_opacity(0.2),
            huffman_chart.bar_labels[2][2].animate.set_color(PURE_GREEN)
        , run_time = 0.3)
        self.play(FadeIn(VGroup(rects_2[2], arrows_2[2])))

        # Show all the chunks being decoded
        self.play(
            huffman_chart.animate.restore(),
            bits_2.animate.set_color(PINK).set_opacity(1),
            FadeIn(rects_2[3:]),
            FadeIn(arrows_2[3:].set_opacity(1))
        )
        bits_2.resume_updating()
        arrows_2.resume_updating()
        rects_2.resume_updating()
        self.add(bits_2, arrows_2, rects_2)
        self.play(
            self.camera.frame.animate(run_time = 3).restore(),
            rects_2_opacity_tracker.animate.set_value(1),
            bits_2.animate(run_time = 7.5).align_to(robot.get_center(), RIGHT)
        )

        # Show prefix rule
        huffman_chart.save_state()
        code_word_1 = Tex("10", font_size = 80)
        code_word_2 = Tex("100", font_size = 80).next_to(code_word_1, DOWN, buff = 0.8).align_to(code_word_1, LEFT)
        VGroup(code_word_1, code_word_2).center()
        background = Group(robot_2, code_words_text, mission_control_2)
        background.save_state()
        self.play(Group(huffman_chart, background).animate.fade(0.9))
        self.play(FadeIn(code_word_1))
        self.play(FadeIn(code_word_2))
        self.play(code_word_1[:2].animate.set_color(RED), code_word_2[:2].animate.set_color(RED))

        # Contemplate adding a 5th instruction
        fifth_instruction_chart = EntropyChart(
            [40/90, 20/90, 10/90, 10/90, 10/90],
            event_labels = VGroup(
                *[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)],
                SVGMobject("clockwise.svg").scale(2).set_stroke(width = 2, behind = True)
            ).set_color(PINK),
            probability_labels = None,
            bar_labels = [
                Tex(encoding[i], font_size = 60)
                for i in range(4)
            ] + [Tex("100", font_size = 60)],
            bar_heights = [1, 2, 3, 3, 3],
            width = 9,
            height = 13.5,
            include_vertical_axis = False,
            segments_height = 1,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        )
        fifth_instruction_chart.clear_updaters()
        fifth_instruction_chart.fix_in_frame().match_width(huffman_chart).match_x(huffman_chart).align_to(huffman_chart, UP)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    background.animate.restore(),
                    code_word_1.animate.match_width(fifth_instruction_chart.bar_labels[1]).move_to(fifth_instruction_chart.bar_labels[1]).set_color(BLACK),
                    code_word_2.animate.match_width(fifth_instruction_chart.bar_labels[-1]).move_to(fifth_instruction_chart.bar_labels[-1]).set_color(BLACK)
                ),
                AnimationGroup(
                    ReplacementTransform(huffman_chart.bars[:4], fifth_instruction_chart.bars[:4]),
                    ReplacementTransform(huffman_chart.bar_labels[:4], fifth_instruction_chart.bar_labels[:4]),
                    ReplacementTransform(huffman_chart.segments.bars[:4], fifth_instruction_chart.segments.bars[:4]),
                    ReplacementTransform(huffman_chart.event_labels[:4], fifth_instruction_chart.event_labels[:4]),
                    FadeOut(huffman_chart.probability_labels),
                    FadeIn(
                        VGroup(
                            fifth_instruction_chart.bars[-1],
                            fifth_instruction_chart.bar_labels[-1],
                            fifth_instruction_chart.segments.bars[-1],
                            fifth_instruction_chart.event_labels[-1]
                        ),
                        shift = LEFT*0.5
                    )
                )
            , lag_ratio = 0.3, run_time = 2)
        )

        # Show how it fails
        bit_string = "10•••••••••••••••"
        bits_3 = VGroup(*[
            Text(b, font_size = 40) for b in bit_string]
        ).set_color(PINK).arrange(buff = bit_buff).align_to(mission_control_2, LEFT)
        for bit in bits_3[2:]:
            bit.scale(0.6)
        bits_3.match_y(bits_2)
        bits_opacity_tracker = ValueTracker(0)
        def update_bits_3(m):
            for bit in m:
                bit.set_opacity(bits_opacity_tracker.get_value()*min(1, max(0, 0.8*(mission_control_2.get_left()[0] - bit.get_x()))))
                if bit.get_x() < robot_2.get_x():
                    bit.set_opacity(0)
            self.bring_to_front(robot_2)
        bits_3.add_updater(update_bits_3)
        self.add(bits_3)

        self.play(
            bits_3.animate(run_time = 6).align_to(robot_2.get_right(), LEFT).shift(RIGHT*0.3),
            bits_opacity_tracker.animate(run_time = 2).set_value(1)
        )
        bits_3.suspend_updating()
        self.wait(2)

        fifth_instruction_chart.save_state()
        self.play(
            bits_3[:2].animate.set_color(PURE_GREEN),
            fifth_instruction_chart.animate.fade(0.8),
            VGroup(
                fifth_instruction_chart.bars[1],
                fifth_instruction_chart.bar_labels[1],
                fifth_instruction_chart.segments.bars[1],
                fifth_instruction_chart.event_labels[1]
            ).animate.shift(0)
        )
        self.wait(2)
        self.play(
            bits_3[2].animate.become(Integer(0).match_height(bits_3[1]).match_y(bits_3[1]).match_x(bits_3[2]).set_color(YELLOW)),
            fifth_instruction_chart.animate.restore().fade(0.8),
            VGroup(
                fifth_instruction_chart.bars[-1],
                fifth_instruction_chart.bar_labels[-1],
                fifth_instruction_chart.segments.bars[-1],
                fifth_instruction_chart.event_labels[-1]
            ).animate.set_opacity(1)
        )
        self.wait(2)

        # Robot gets confused
        question_mark = Tex(
            "?", font_size = 80
        ).set_color(
            LIGHT_BROWN
        ).next_to(robot_2, UP, buff = 0)
        self.play(FadeIn(question_mark, shift = DOWN*0.05))
        self.wait(1)

        # Bring back the original chart
        huffman_chart.restore()
        self.play(
            FadeOut(VGroup(bits_3, question_mark)),
            ReplacementTransform(fifth_instruction_chart.bars[:4], huffman_chart.bars[:4]),
            ReplacementTransform(fifth_instruction_chart.bar_labels[:4], huffman_chart.bar_labels[:4]),
            ReplacementTransform(fifth_instruction_chart.segments.bars[:4], huffman_chart.segments.bars[:4]),
            ReplacementTransform(fifth_instruction_chart.event_labels[:4], huffman_chart.event_labels[:4]),
            FadeIn(huffman_chart.probability_labels),
            FadeOut(
                VGroup(
                    fifth_instruction_chart.bars[-1],
                    fifth_instruction_chart.segments.bars[-1],
                    fifth_instruction_chart.event_labels[-1],
                    fifth_instruction_chart.bar_labels[-1]
                ),
                shift = RIGHT*0.5
            )
        , run_time = 2)

        # Fade out everything but the chart
        self.remove(bits_2, rects_2, arrows_2)
        self.play(
            FadeOut(code_words_text),
            FadeOut(Group(robot_2, mission_control_2), shift = DOWN),
            huffman_chart.animate(path_arc = PI*0.1).scale(1.2).align_to(huffman_chart, RIGHT).center()
        , run_time = 2)
        self.wait(3)


        # Calculate the efficiency of the Huffman code
        self.wait(2)
        weighted_sum_lines = VGroup(
            Tex(R"\frac{1}{2} \cdot 1", font_size = 30).next_to(huffman_chart.bars[0], UP),
            Tex(R"\frac{1}{4} \cdot 2", font_size = 30).next_to(huffman_chart.bars[1], UP),
            Tex(R"\frac{1}{8} \cdot 3", font_size = 30).next_to(huffman_chart.bars[2], UP),
            Tex(R"\frac{1}{8} \cdot 3", font_size = 30).next_to(huffman_chart.bars[3], UP)
        )
        self.play(
            AnimationGroup(
                TransformFromCopy(huffman_chart.probability_labels[0], weighted_sum_lines[0][:-2]),
                FadeIn(weighted_sum_lines[0][-2:], shift = UP*0.1)
            , run_time = 1.5)
        )
        self.wait(1.5)
        self.play(
            AnimationGroup(
                TransformFromCopy(huffman_chart.probability_labels[1], weighted_sum_lines[1][:-2]),
                FadeIn(weighted_sum_lines[1][-2:], shift = UP*0.1)
            , run_time = 1.5)
        )
        self.wait(2.5)
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    TransformFromCopy(label, line[:-2]),
                    FadeIn(line[-2:], shift = UP*0.1)
                , run_time = 1.5)
                for label, line in zip(huffman_chart.probability_labels[2:], weighted_sum_lines[2:])
            ], lag_ratio = 0.3)
        )
        self.wait(0.35)

        # Show the weighted sum result
        sum_result = Tex(
            R"\frac{1}{2} \cdot 1 + \frac{1}{4} \cdot 2 + \frac{1}{8} \cdot 3 + \frac{1}{8} \cdot 3 \\ = 1.75 \text{ bits}",
            font_size = 49
        )
        sum_result[-9:].align_to(sum_result[:-9], RIGHT)
        sum_result.to_edge(RIGHT, buff = 1.2)

        self.play(
            huffman_chart.animate.scale(0.8).to_edge(LEFT, buff = 1),
            TransformMatchingShapes(weighted_sum_lines, sum_result[:-9], path_arc = PI*0.2, run_time = 1.5)
        )
        self.wait(0.5)
        self.play(FadeIn(sum_result[R"= 1.75 \text{ bits}"]))

        # Compare to the naive encoding
        huffman_group = VGroup(huffman_chart, sum_result)
        naive_sum_result = Tex(
            R"\frac{1}{2} \cdot 2 + \frac{1}{4} \cdot 2 + \frac{1}{8} \cdot 2 + \frac{1}{8} \cdot 2 \\ = 2 \text{ bits}",
            font_size = 49
        )
        naive_sum_result[-6:].align_to(naive_sum_result[:-6], RIGHT)
        naive_sum_result.to_edge(RIGHT, buff = 1.2)
        naive_chart.match_width(huffman_chart).move_to(huffman_chart)
        for label in naive_chart.bar_labels:
            label.match_height(huffman_chart.bar_labels[0])
        naive_group = VGroup(naive_chart, naive_sum_result).to_edge(UP, buff = 1)

        huffman_group.generate_target()
        huffman_group.target[0].scale(0.7)
        huffman_group.target.arrange(buff = 1).to_edge(DOWN, buff = 0.7)
        naive_group[0].scale(0.7)
        naive_group.arrange(buff = 1).to_edge(UP, buff = 0.7)
        naive_group.to_edge(UP, buff = 0.7)
        
        huffman_group.save_state()
        self.play(
            MoveToTarget(huffman_group),
            FadeIn(naive_group, shift = DOWN)
        , run_time = 1.5)
        self.wait(3)

        # Bring the huffman code back to focus
        self.play(
            huffman_group.animate.restore(),
            FadeOut(naive_group, shift = UP)
        , run_time = 1.5)
        self.wait(3)

        # Put away the calculation and focus on the chart
        self.play(FadeOut(sum_result, shift = RIGHT*3), huffman_chart.animate.set_x(0).to_edge(DOWN, buff = 0.7), run_time = 1.5)
        self.wait(3)

        # Show a message that looks like random noise
        mission_control_and_robot_group = Group(
            robot.scale(0.8).to_edge(LEFT, buff = 0.7),
            mission_control.scale(0.8).to_edge(RIGHT, buff = 0.7)
        ).set_x(0).to_edge(UP, buff = 0.5)
        robot = mission_control_and_robot_group[0]
        mission_control = mission_control_and_robot_group[1]
        message = VGroup(*[
            Integer(0).set_color(YELLOW)
            for _ in range(42)
        ]).arrange(
            buff = 0.08
        ).set_width(
            mission_control_and_robot_group.get_width()*0.76
        ).match_y(
            mission_control
        ).match_x(
            mission_control_and_robot_group
        ).shift(
            RIGHT*0.1
        )
        bit_opacity_trackers = [ValueTracker(0) for _ in message]
        for i, bit in enumerate(message):
            def update_bit(m, i = i):
                m.set_value(random.choice([0, 1])).set_opacity(bit_opacity_trackers[i].get_value())
            bit.add_updater(update_bit)
        self.add(message)
        self.play(
            AnimationGroup(*[t.animate.set_value(1) for t in bit_opacity_trackers]),
            FadeIn(mission_control_and_robot_group)
        )
        self.wait(1)

        # Show the probability of the first bit being 0 vs 1
        arrow = Arrow(ORIGIN, UP).set_color(WHITE).next_to(message[0], DOWN)
        huffman_chart.save_state()
        self.play(
            GrowArrow(arrow),
            AnimationGroup(*[t.animate.set_value(0.2) for t in bit_opacity_trackers[1:]]),
            huffman_chart.bars[1:].animate.set_opacity(0.2),
            huffman_chart.bar_labels[1:].animate.set_opacity(0.2),
            huffman_chart.segments.bars[1:].animate.set_opacity(0.2),
            huffman_chart.event_labels[1:].animate.set_opacity(0.2),
            huffman_chart.probability_labels[1:].animate.set_opacity(0.2)
        )
        message[0].suspend_updating()
        message[0].set_value(0)
        self.play(message[0].animate.set_color(PURE_GREEN))
        self.wait(1.5)
        self.play(
            message[0].animate.set_value(1),
            huffman_chart.animate.restore(),
            huffman_chart.bars[0].animate.set_opacity(0.2),
            huffman_chart.bar_labels[0].animate.set_opacity(0.2),
            huffman_chart.segments.bars[0].animate.set_opacity(0.2),
            huffman_chart.event_labels[0].animate.set_opacity(0.2),
            huffman_chart.probability_labels[0].animate.set_opacity(0.2)
        )
        self.wait(3)

        # Show the probability of the second bit being 0 vs 1
        arrow2 = Arrow(ORIGIN, UP).set_color(WHITE).next_to(message[1], DOWN)
        self.play(
            FadeOut(arrow),
            GrowArrow(arrow2),
            bit_opacity_trackers[1].animate.set_value(1),
            huffman_chart.bars[2:].animate.set_opacity(0.2),
            huffman_chart.bar_labels[2:].animate.set_opacity(0.2),
            huffman_chart.segments.bars[2:].animate.set_opacity(0.2),
            huffman_chart.event_labels[2:].animate.set_opacity(0.2),
            huffman_chart.probability_labels[2:].animate.set_opacity(0.2)
        )
        message[1].suspend_updating()
        message[1].set_value(0)
        self.play(message[1].animate.set_color(PURE_GREEN))
        self.wait(1.2)
        self.play(
            message[1].animate.set_value(1),
            huffman_chart.animate.restore(),
            huffman_chart.bars[:2].animate.set_opacity(0.2),
            huffman_chart.bar_labels[:2].animate.set_opacity(0.2),
            huffman_chart.segments.bars[:2].animate.set_opacity(0.2),
            huffman_chart.event_labels[:2].animate.set_opacity(0.2),
            huffman_chart.probability_labels[:2].animate.set_opacity(0.2)
        )
        self.wait(3)

        # Show the probability of the third bit being 0 vs 1
        arrow3 = Arrow(ORIGIN, UP).set_color(WHITE).next_to(message[2], DOWN)
        self.play(
            FadeOut(arrow2),
            GrowArrow(arrow3),
            bit_opacity_trackers[2].animate.set_value(1),
            huffman_chart.bars[3].animate.set_opacity(0.2),
            huffman_chart.bar_labels[3].animate.set_opacity(0.2),
            huffman_chart.segments.bars[3].animate.set_opacity(0.2),
            huffman_chart.event_labels[3].animate.set_opacity(0.2),
            huffman_chart.probability_labels[3].animate.set_opacity(0.2)
        )
        message[2].suspend_updating()
        message[2].set_value(0)
        self.play(message[2].animate.set_color(PURE_GREEN))
        self.wait(1.2)
        self.play(
            message[2].animate.set_value(1),
            huffman_chart.animate.restore(),
            huffman_chart.bars[:3].animate.set_opacity(0.2),
            huffman_chart.bar_labels[:3].animate.set_opacity(0.2),
            huffman_chart.segments.bars[:3].animate.set_opacity(0.2),
            huffman_chart.event_labels[:3].animate.set_opacity(0.2),
            huffman_chart.probability_labels[:3].animate.set_opacity(0.2)
        )
        self.wait(1.2)
        self.play(message[:3].animate.set_color(YELLOW)),
        message[0].resume_updating()
        message[1].resume_updating()
        message[2].resume_updating()
        self.play(
            FadeOut(arrow3),
            huffman_chart.animate.restore(),
            AnimationGroup(*[t.animate.set_value(1) for t in bit_opacity_trackers])
        )
        self.wait(3)

        # Define information
        huffman_chart.generate_target()
        huffman_chart.target.scale(0.8).to_corner(DL, buff = 1)
        arrow = Arrow(huffman_chart.get_top(), huffman_chart.get_top() + UP*2.7).set_color(WHITE)
        self.play(GrowArrow(arrow), run_time = 2)
        self.wait(5)
        message_probability = Tex(
            R"\mathbf{P}(\text{any $n$-bit message}) = \frac{1}{2^n}"
        ).next_to(huffman_chart.target, RIGHT, buff = 1).align_to(huffman_chart, UP).to_edge(RIGHT, buff = 2).shift(DOWN*0.3)
        new_arrow = Arrow(huffman_chart.target.get_corner(UR) + UR*0.4, arrow.get_end(), buff = 0)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    MoveToTarget(huffman_chart, run_time = 2),
                    arrow.animate(run_time = 2.3).become(new_arrow)
                ),
                Write(message_probability), run_time = 4.5
            , lag_ratio = 0.4)
        )
        self.wait(5)
        message_probability_line_2 = Tex("p = 2^{-n}").next_to(message_probability, DOWN, buff = 0.3)
        message_probability_line_2.shift(RIGHT*(message_probability[18].get_x() - message_probability_line_2["="].get_x()))
        self.play(
            TransformFromCopy(message_probability[:18], message_probability_line_2["p"], run_time = 1.5),
            TransformMatchingShapes(message_probability[18:].copy(), message_probability_line_2["= 2^{-n}"], run_time = 1.5)
        )
        self.wait(1.2)
        log_of_both_sides = Tex(R"\log_2(p) = \log_2(2^{-n})").match_y(message_probability_line_2)
        log_of_both_sides.shift(RIGHT*(message_probability_line_2["="].get_x() - log_of_both_sides["="].get_x()))
        self.play(TransformMatchingShapes(message_probability_line_2, log_of_both_sides), run_time = 1)
        log_of_both_sides_simplified = Tex(R"\log_2(p) = -n").match_y(log_of_both_sides)
        log_of_both_sides_simplified.shift(RIGHT*(log_of_both_sides["="].get_x() - log_of_both_sides_simplified["="].get_x()))
        self.play(
            TransformMatchingShapes(log_of_both_sides[R"\log_2(p) ="], log_of_both_sides_simplified[R"\log_2(p) ="]),
            FadeOut(VGroup(log_of_both_sides[R"\log_2(2"], log_of_both_sides[")"][1])),
            ReplacementTransform(log_of_both_sides["-n"], log_of_both_sides_simplified["-n"])
        , run_time = 1)
        negated = Tex(R"\text{Information} =  -\log_2(p) = {n}").match_y(log_of_both_sides_simplified)
        negated["Information"].set_color(TEAL)
        negated.shift(RIGHT*(log_of_both_sides_simplified["="].get_x() - negated["="][1].get_x()))
        self.play(
            TransformMatchingShapes(log_of_both_sides_simplified[R"\log_2(p) ="], negated[R"\log_2(p) ="]),
            TransformMatchingShapes(log_of_both_sides_simplified[R"n"], negated[R"{n}"]),
            TransformMatchingShapes(log_of_both_sides_simplified[R"-"], negated[R"-"], path_arc = PI*0.4)
        )
        self.wait(6)
        rect = SurroundingRectangle(negated[R"-\log_2(p)"], stroke_width = 3, stroke_color = TEAL)
        self.play(ShowCreation(rect), run_time = 2)
        self.wait(4)
        # self.wait(16)
        # self.play(Write(negated[R"\text{Information} ="], run_time = 1.2))
        # self.wait(7.5)

        # Draw a graph of -log_2 p
        x_range = [0, 1]
        y_range = [0, 10]
        axes = Axes(
            x_range = x_range,
            y_range = y_range,
            width = 6,
            height = 5
        ).to_edge(RIGHT, buff = 3)
        p_label = Tex("p").next_to(axes, DOWN, buff = 0.5).set_opacity(0)
        labels = axes.add_coordinate_labels(
            x_values = [0.001, 1],
            y_values = [0, 2, 4, 6, 8, 10]
        )
        def func(x):
            return -math.log2(x)
        curve = ParametricCurve(
            lambda t: (axes.c2p(t, func(t))), (0.001, x_range[1], 0.001)
        ).set_stroke(width=4, color=TEAL, opacity=0.9)
        negated[R"\text{Information} ="].set_opacity(0)
        information_definition = negated[R"\text{Information} = -\log_2(p)"]
        self.play(
            AnimationGroup(
                AnimationGroup(
                    FadeOut(mission_control_and_robot_group),
                    AnimationGroup(*[t.animate.set_value(0) for t in bit_opacity_trackers]),
                    FadeOut(arrow),
                    FadeOut(huffman_chart),
                    FadeOut(message_probability),
                    FadeOut(rect),
                    FadeOut(negated["= {n}"])
                ),
                information_definition.animate.scale(1.1).next_to(axes, LEFT, buff = 0.6),
                AnimationGroup(
                    FadeIn(VGroup(axes, p_label)),
                    ShowCreation(curve, run_time = 1)
                , lag_ratio = 0.5)
            , lag_ratio = 0.6)
        , run_time = 3)
        self.remove(message)

        # Track values of p vs -log_2(p)
        p_tracker = ValueTracker(0.314)
        p_dot = Group(GlowDot(), TrueDot()).set_color(YELLOW)
        p_dot.add_updater(lambda m: m.move_to(axes.c2p(p_tracker.get_value(), func(p_tracker.get_value()))))
        dashed_line = always_redraw(lambda: DashedLine(p_dot.get_center(), [curve.get_left()[0], p_dot.get_y(), 0]).set_color(YELLOW))
        p_value_triangle = Triangle(fill_opacity = 0.7, fill_color = YELLOW, stroke_width = 0).set_width(0.25).stretch_to_fit_height(0.3)
        p_value_triangle.add_updater(lambda m: m.move_to(axes.c2p(p_tracker.get_value(), 0)).align_to([0, axes.get_x_axis()[0].get_y(), 0], UP))
        p_display = Tex("p = 0.00", font_size = 30).set_stroke(color = BLACK, width = 2, behind = True)
        p_display.add_updater(lambda m: m.next_to(p_value_triangle, DOWN))
        p_value = p_display.make_number_changeable("0.00")
        p_value.add_updater(lambda m: m.set_value(p_tracker.get_value()))
        self.play(
            FadeIn(p_dot, suspend_mobject_updating = True),
            ShowCreation(dashed_line, suspend_mobject_updating = True),
            FadeIn(VGroup(p_value_triangle, p_display))
        , run_time = 0.9)
        bits_text = TexText("(bits)").next_to(information_definition, DOWN, buff = 0.3)
        bits_display = TexText("Information $=$ 0.00 bits", font_size = 38)
        bits_display["Information"].set_color(TEAL)
        display_group = VGroup(bits_display)
        display_group.arrange(DOWN)
        bits_display.shift(LEFT*(bits_display["="].get_x() - p_display["="].get_x()))
        display_group.align_to(axes, UR)
        bits_value = bits_display.make_number_changeable("0.00")
        bits_value.add_updater(lambda m: m.set_value(func(p_tracker.get_value())))
        display_opacity_tracker = ValueTracker(0)
        display_group.add_updater(lambda m: m.set_opacity(display_opacity_tracker.get_value()))
        self.add(display_group)
        values = [random.random() for _ in range(3)] + [0.019, 0.789] + [random.random() for _ in range(17)]
        for i, value in enumerate(values):
            anims = [p_tracker.animate.set_value(value)]
            if i == 1:
                anims.append(
                    AnimationGroup(
                        self.camera.frame.animate(run_time = 2).match_x(VGroup(information_definition, axes)),
                        information_definition.animate(run_time = 2).set_opacity(1)
                    , lag_ratio = 0.4)
                )
            if i == 12:
                anims.append(AnimationGroup(FadeIn(bits_text), display_opacity_tracker.animate(run_time = 2).set_value(1)))
            if i == 17:
                # Show the distinction between the different meanings of "bits"
                length = 13
                discrete_bits = VGroup(*[
                    Integer(random.choice([0, 1]), font_size = 60) for _ in range(length)
                ]).arrange(buff = 0.1).set_color(YELLOW).next_to(information_definition, LEFT, buff = 0)
                num_bits_label = Tex(str(length) + R"\text{ bits}").next_to(discrete_bits, DOWN)
                anims.append(
                    AnimationGroup(
                        FadeOut(VGroup(negated[R"\text{Information} ="], bits_text), run_time = 1),
                        self.camera.frame.animate(run_time = 2.5).scale(1.5).move_to(VGroup(discrete_bits, axes)).shift(LEFT*0.5),
                        FadeIn(VGroup(discrete_bits, num_bits_label), shift = RIGHT, run_time = 2)
                    )
                )
            if i > 17:
                # Update the discrete bit count
                length = random.randint(2, 15)
                discrete_bits.become(
                    VGroup(*[
                        Integer(random.choice([0, 1]), font_size = 60) for _ in range(length)
                    ]).arrange(buff = 0.1).set_color(YELLOW).move_to(discrete_bits)
                )
                num_bits_label.become(Tex(str(length) + R"\text{ bits}").next_to(discrete_bits, DOWN))
            self.play(*anims, run_time = 2.5)
            self.wait(0.8)
        self.wait(1)

class NaiveEncodingBitsTwoMeanings(InteractiveScene):
    def construct(self):
        # Show the dual meaning of "bits" for the naive encoding
        distribution = [1/2, 1/4, 1/8, 1/8]
        naive_chart = EntropyChart(
            distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]),
            probability_labels = [
                Tex(R"\frac{1}{" + ["2", "4", "8", "8"][i] + "}", font_size = 40)
                for i in range(4)
            ],
            bar_labels = [
                Tex(["00", "01", "10", "11"][i], font_size = 50)
                for i in range(4)
            ],
            bar_heights = [2, 2, 2, 2],
            width = 6,
            height = 7,
            include_vertical_axis = False,
            segments_height = 0.6,
            fill_colors = [YELLOW_B, YELLOW_D]
        ).shift(DOWN)
        naive_encoding_text = TexText("Naive encoding:", font_size = 60).next_to(naive_chart.bars, UP, buff = 0.7)
        self.play(naive_chart.create(), FadeIn(naive_encoding_text, run_time = 2))
        naive_chart.clear_updaters()
        self.wait(2)
        self.play(VGroup(naive_chart, naive_encoding_text).animate.scale(0.83).to_edge(LEFT, buff = 1.4).set_y(0.5))

        arrow1 = Arrow(LEFT*2, RIGHT*2).next_to(naive_chart.bars, RIGHT, buff = 0.6)
        arrow2 = arrow1.copy().match_y(naive_chart.probability_labels)
        arrow1_label = TexText("code word length (bits)", font_size = 31).next_to(arrow1, UP)
        arrow2_label = Tex(R"-\log_2 p_i \text{ (bits)}", font_size = 31).next_to(arrow2, UP)
        self.play(GrowArrow(arrow1), Write(arrow1_label), run_time = 2)
        code_word_lengths = Tex(R"2,\ 2,\ 2,\ 2,", font_size = 40).next_to(arrow1, RIGHT, buff = 0.6).set_color(YELLOW)
        code_word_lengths[","].set_color(WHITE)
        code_word_lengths[-1].set_opacity(0)
        self.play(
            AnimationGroup(*[
                FadeIn(code_word_lengths["2,"][i])
                for i in range(4)
            ], lag_ratio = 0.3)
        )

        self.play(GrowArrow(arrow2), Write(arrow2_label), run_time = 2)
        informations = Tex(R"1,\ 2,\ 3,\ 3,", font_size = 40).next_to(arrow2, RIGHT, buff = 0.6).set_color(YELLOW)
        informations[","].set_color(WHITE)
        informations[-1].set_opacity(0)
        self.play(
            AnimationGroup(*[
                FadeIn(informations[2*i:2*i + 2])
                for i in range(4)
            ], lag_ratio = 0.3)
        )

class PerfectEncodingsAndEntropyDefinition(InteractiveScene):
    def construct(self):
        # Bring back the huffman chart to show that it's a perfect encoding
        distribution = [1/2, 1/4, 1/8, 1/8]
        encoding = ["0", "10", "110", "111"]
        full_huffman_chart = EntropyChart(
            distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]).set_color(PINK),
            probability_labels = [
                Tex(R"\frac{1}{" + ["2", "4", "8", "8"][i] + "}", font_size = 40)
                for i in range(4)
            ],
            bar_labels = [
                Tex(encoding[i], font_size = 40)
                for i in range(4)
            ],
            bar_heights = [1, 2, 3, 3],
            width = 10,
            height = 4,
            segments_height = 1,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).move_to(self.camera.frame).shift(DOWN*0.5)
        full_huffman_chart.bar_labels.set_color(WHITE)
        self.add(full_huffman_chart)
        self.wait(1.5)

        # Compare the message length to the information
        self.play(
            AnimationGroup(*[
                Indicate(VGroup(bar, label), suspend_mobject_updating = True)
                for bar, label in zip(full_huffman_chart.bars, full_huffman_chart.bar_labels)
            ], lag_ratio = 0.2),
            full_huffman_chart.bar_labels.animate.shift(0)
        , run_time = 3)
        self.wait(1)
        information_labels = VGroup(*[
            Tex(
                str(num_bits) + R"\text{ bit" + ("s" if num_bits > 1 else "") + "}",
                font_size = 40
            ).set_stroke(
                width = 2, color = BLACK, behind = True
            ).next_to(full_huffman_chart.bars[i], UP)
            for i, num_bits in enumerate([1, 2, 3, 3])
        ])
        for label in information_labels:
            self.play(FadeIn(label, shift = UP*0.3))
            self.wait(0.2)
        self.wait(3)

        # Generalize 
        distribution = random_distribution(7)
        general_chart = EntropyChart(
            random_distribution(7),
            event_labels = VGroup(*[
                Tex(
                    (("m_" + str(i)) if i < len(distribution) - 2 else R"\ldots" if i == len(distribution) - 2 else "m_n")
                ).scale(0.8).set_color(BLACK)
                for i in range(len(distribution))
            ]),
            probability_labels = VGroup(*[
                Tex(
                    (("p_" + str(i)) if i < len(distribution) - 2 else R"\ldots" if i == len(distribution) - 2 else "p_n")
                )
                for i in range(len(distribution))
            ]),
            width = 10,
            height = 4,
            segments_height = 0.4,
            fit_event_labels_to_height = False,
            fill_colors = [YELLOW_B, YELLOW_D]
        ).match_width(full_huffman_chart).match_x(full_huffman_chart).align_to(full_huffman_chart, UP)
        general_chart.bars.add_updater(lambda m: m.set_stroke(width = 1))
        general_chart.segments.bars.set_stroke(width = 1)
        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate(run_time = 5).match_x(general_chart.segments),
            AnimationGroup(
                FadeOut(VGroup(full_huffman_chart, full_huffman_chart.event_labels, information_labels), suspend_mobject_updating = True),
                AnimationGroup(*[
                    AnimationGroup(GrowFromCenter(segment), FadeIn(e_label), FadeIn(p_label))
                    for segment, e_label, p_label in zip(
                        general_chart.segments.bars, general_chart.event_labels, general_chart.probability_labels
                    )
                ], lag_ratio = 0.2, suspend_mobject_updating = True)
            , lag_ratio = 0.5, run_time = 2.5)
        )

        # Move around the probabilities and write the definition of entropy
        self.add(general_chart)
        general_chart.save_state()
        bars_opacity_tracker = ValueTracker(0)
        general_chart.bars.add_updater(lambda m: m.set_opacity(bars_opacity_tracker.get_value()))
        general_chart.bars.add_updater(lambda m: self.bring_to_front(m))
        general_chart.vertical_axis.set_opacity(0)
        general_chart.vertical_axis_label.set_opacity(0)
        general_chart.reference_lines.set_opacity(0)
        for i in range(30):
            anims = [general_chart.set_distribution(random_distribution(7))]
            if i == 2:
                brace = Brace(general_chart.segments, UP)
                total_width_text = brace.get_tex(R"\text{Total width} = 1")
                anims.append(
                    AnimationGroup(
                        GrowFromEdge(brace, DOWN),
                        Write(total_width_text)
                    , run_time = 2)
                )
            if i == 3:
                anims.append(
                    AnimationGroup(
                        FadeOut(VGroup(brace, total_width_text)),
                        bars_opacity_tracker.animate.set_value(1)
                    )
                )
            if i == 5:
                anims.append(
                    AnimationGroup(
                        self.camera.frame.animate.restore().shift(UP*0.5),
                        VGroup(
                            general_chart.vertical_axis,
                            general_chart.vertical_axis_label,
                            general_chart.reference_lines
                        ).animate.set_opacity(1)
                    )
                )
            if i == 7:
                # Add the area formula
                weighted_sum_formula = Tex(
                    R"\text{Area}() = \sum_i p_i (-\log_2 p_i) = \text{Avg. information}"
                ).next_to(general_chart, UP)
                bars_copy = general_chart.bars.copy()
                bars_copy.clear_updaters().set_opacity(1)
                bars_copy.generate_target()
                bars_copy_height = bars_copy.get_height()
                scale_factor = weighted_sum_formula[4].get_height()/bars_copy_height
                bars_copy.target.stretch(0.5, 0).scale(scale_factor).next_to(weighted_sum_formula[4], RIGHT)
                weighted_sum_formula[5:].next_to(bars_copy.target, RIGHT)
                weighted_sum_formula[5:].shift(DOWN*(weighted_sum_formula[5].get_y() - weighted_sum_formula[4].get_y()))
                VGroup(weighted_sum_formula, bars_copy.target).set_x(0)
                # VGroup(part1, bars_copy.target).match_x(general_chart.reference_lines)
                self.play(
                    AnimationGroup(
                        Write(weighted_sum_formula),
                        MoveToTarget(bars_copy)
                    , run_time = 6)
                )
                def update_area_bars(m):
                    m.become(
                        general_chart.bars.copy()
                        .clear_updaters()
                        .set_opacity(1)
                        .stretch(0.5, 0)
                        .scale(scale_factor)
                        .next_to(weighted_sum_formula[4], RIGHT)
                        .align_to(m, DOWN)
                    )
                bars_copy.add_updater(update_area_bars)
            # if i == 10:
            #     part1.generate_target()
            #     part1.target.restore()
            #     bars_copy.generate_target()
            #     bars_copy.target.shift(part1.target.get_center() - part1.get_center())
            #     anims.append(
            #         AnimationGroup(
            #             AnimationGroup(MoveToTarget(part1), MoveToTarget(bars_copy)),
            #             Write(part2, run_time = 2.5)
            #         , lag_ratio = 0.6)
            #     )
            # if i == 18:
            #     entropy_text = TexText(
            #         "Shannon Entropy ($H$)"
            #     ).next_to(
            #         weighted_sum_formula[R"\text{Avg. information}"], DOWN, buff = 1
            #     )
            #     entropy_text["H"].set_color(BLUE)
            #     rect = BackgroundRectangle(entropy_text["Entropy"], buff = 0.2)
            #     rect2 = BackgroundRectangle(entropy_text["Shannon Entropy"], buff = 0.2)
            #     rect3 = BackgroundRectangle(entropy_text, buff = 0.2)
            #     arrow = always_redraw(
            #         lambda: Arrow(
            #             entropy_text["Entropy"].get_top() + UP*0.1,
            #             weighted_sum_formula[R"\text{Avg. information}"].get_bottom() + DOWN*0.1
            #         , buff = 0.1)
            #     )
            #     arrow.suspend_updating()
            #     anims.append(
            #         AnimationGroup(
            #             FadeIn(rect, run_time = 1),
            #             Write(entropy_text["Entropy"], run_time = 1),
            #             GrowArrow(arrow, run_time = 1.2)
            #         )
            #     )
            # if i == 19:
            #     anims.append(
            #         AnimationGroup(
            #             rect.animate(run_time = 0.7).become(rect2),
            #             FadeIn(entropy_text["Shannon"]),
            #             entropy_text["Entropy"].animate.shift(0),
            #             arrow.animate.shift(0)
            #         , lag_ratio = 0.4)
            #     )
            # if i == 20:
            #     anims.append(
            #         AnimationGroup(
            #             rect.animate.become(rect3),
            #             FadeIn(entropy_text["($H$)"]),
            #             entropy_text["Shannon Entropy"].animate.shift(0),
            #             arrow.animate.shift(0)
            #         , lag_ratio = 0.6)
            #     )
            if i == 24:
                entropy_display = Tex(R"\text{Entropy} = 0.00 \text{ bits}").next_to(general_chart.reference_lines.get_corner(UL), DR)
                entropy_display_opacity_tracker = ValueTracker(0)
                entropy_display.add_updater(lambda m: m.set_opacity(entropy_display_opacity_tracker.get_value()))
                entropy_value = entropy_display.make_number_changeable("0.00")
                entropy_value.add_updater(
                    lambda m: m.set_value(
                        sum([t.get_value()*-math.log2(t.get_value()) for t in general_chart.distribution_trackers])
                    )
                )
                entropy_display.add_updater(lambda m: self.bring_to_front(m))
                rect4 = BackgroundRectangle(entropy_display, buff = 0.2)
                self.add(entropy_display)
                anims.append(AnimationGroup(FadeIn(rect4), entropy_display_opacity_tracker.animate.set_value(1)))

            self.play(*anims, run_time = 3)
        # self.remove(entropy_text)
        # self.add(entropy_display.set_opacity(1))

        # Show a uniform distribution
        event_labels_opacity_tracker = ValueTracker(1)
        general_chart.segments.add_updater(lambda m: m.labels.set_opacity(event_labels_opacity_tracker.get_value()))
        self.play(
            general_chart.probability_labels.animate.set_opacity(0),
            event_labels_opacity_tracker.animate.set_value(0),
            general_chart.set_distribution([1/7 for _ in range(7)])
        , run_time = 3)
        self.wait(2)

        # Show a squished distribution
        big_prob = 0.789
        leftover = 1 - big_prob
        leftover_distibution = random_distribution(6, thresh = (2**-7)/leftover)
        leftover_distibution = [p*leftover for p in leftover_distibution]
        distribution = [big_prob] + leftover_distibution
        self.play(general_chart.set_distribution(distribution), run_time = 6)
        self.wait(4)

        # Indicate the most probable event with little information
        general_chart.save_state()
        rect4.save_state()
        entropy_display.save_state()
        bars_copy.save_state()
        weighted_sum_formula.save_state()
        general_chart.suspend_updating()
        entropy_display.suspend_updating()
        bars_copy.suspend_updating()
        weighted_sum_formula.suspend_updating()
        self.play(
            VGroup(
                general_chart.bars[1:],
                general_chart.segments.bars[1:],
                general_chart.reference_lines,
                general_chart.vertical_axis,
                general_chart.vertical_axis_label,
                rect4,
                entropy_display,
                bars_copy,
                weighted_sum_formula
            ).animate.fade(0.8)
        , run_time = 2)
        arrow = Arrow(ORIGIN, DOWN).set_color(TEAL).next_to(general_chart.bars[0], UP)
        label = TexText(
            R"high probability $\Longleftrightarrow$ low information",
            font_size = 40
        ).set_color(TEAL).next_to(arrow, UP, buff = 0.15)
        self.play(GrowArrow(arrow), FadeIn(label, run_time = 1.2))
        self.wait(2)
        self.play(
            general_chart.animate.restore(),
            rect4.animate.restore(),
            entropy_display.animate.restore(),
            bars_copy.animate.restore(),
            weighted_sum_formula.animate.restore(),
            FadeOut(VGroup(arrow, label))
        , run_time = 2)
        general_chart.resume_updating()
        self.add(rect4)
        entropy_display.resume_updating()
        bars_copy.resume_updating()
        weighted_sum_formula.resume_updating()
        self.wait(2)

        # Divide the probability space into more chunks
        entropy_value.clear_updaters()
        rect4.add_updater(lambda m: self.bring_to_front(m))
        entropy_display.add_updater(lambda m: self.bring_to_front(m))
        prev_chart = general_chart
        for i in range(1, 3):
            new_distribution = []
            for p in distribution:
                new_distribution += [p/2**i for _ in range(2**i)]
            new_chart = EntropyChart(
                new_distribution,
                event_labels = None,
                probability_labels = None,
                width = 10,
                height = 4,
                segments_height = 0.4,
                fit_event_labels_to_height = False,
                fill_colors = [YELLOW_B, YELLOW_D]
            ).match_width(general_chart).match_x(general_chart).align_to(general_chart, UP)
            new_chart.bars.add_updater(lambda m: m.set_stroke(width = 0.3/2**i))
            new_chart.segments.bars.set_stroke(width = 0.5/2**i)
            general_chart.clear_updaters()
            bars_copy.clear_updaters()
            self.play(
                FadeOut(prev_chart, suspend_mobject_updating = True, run_time = 3),
                FadeIn(new_chart, suspend_mobject_updating = True, run_time = 3),
                bars_copy.animate(run_time = 3).become(
                    new_chart.bars.copy()
                    .clear_updaters()
                    .set_opacity(1)
                    .stretch(0.5, 0)
                    .scale(scale_factor)
                    .next_to(weighted_sum_formula[4], RIGHT)
                    .align_to(bars_copy, DOWN)
                ),
                entropy_value.animate(run_time = 1.3).set_value(
                    sum([t.get_value()*-math.log2(t.get_value()) for t in new_chart.distribution_trackers])
                )
            )
            self.wait(2)
            prev_chart = new_chart

        def update_area_bars(m):
            m.become(
                new_chart.bars.copy()
                .clear_updaters()
                .set_opacity(1)
                .stretch(0.5, 0)
                .scale(scale_factor)
                .next_to(weighted_sum_formula[4], RIGHT)
                .align_to(m, DOWN)
            )
        bars_copy.add_updater(update_area_bars)

        # Squish and then spread out the new distribution
        entropy_value.add_updater(
            lambda m: m.set_value(
                sum([t.get_value()*-math.log2(t.get_value()) for t in new_chart.distribution_trackers])
            )
        )
        big_prob_1 = 0.42
        big_prob_2 = 0.178
        big_prob_3 = 0.32
        leftover = 1 - big_prob_1 - big_prob_2 - big_prob_3
        leftover_distibution = random_distribution(25, thresh = (2**-10)/leftover)
        leftover_distibution = [p*leftover for p in leftover_distibution]
        distribution = [big_prob_1, big_prob_2, big_prob_3] + leftover_distibution
        self.play(new_chart.set_distribution(distribution), run_time = 6)
        self.wait(1)

        uniform_distribution = [1/28 for _ in range(28)]
        self.play(new_chart.set_distribution(uniform_distribution), run_time = 6)
        self.wait(1)

        # Show some more random distributions
        for _ in range(5):
            self.play(new_chart.set_distribution(random_distribution(28, thresh = 2**-8)), run_time = 3)



class PerfectEncodingsAndEntropyDefinitionV2(InteractiveScene):
    def construct(self):
        # Bring back the huffman chart to show that it's a perfect encoding
        distribution = [1/2, 1/4, 1/8, 1/8]
        encoding = ["0", "10", "110", "111"]
        full_huffman_chart = EntropyChart(
            distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]).set_color(PINK),
            probability_labels = [
                Tex(R"\frac{1}{" + ["2", "4", "8", "8"][i] + "}", font_size = 40)
                for i in range(4)
            ],
            bar_labels = [
                Tex(encoding[i], font_size = 40)
                for i in range(4)
            ],
            bar_heights = [1, 2, 3, 3],
            width = 9,
            height = 3,
            segments_height = 1,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).center().shift(DOWN*0.7)
        full_huffman_chart.bar_labels.set_color(WHITE)
        full_huffman_chart.vertical_axis.set_opacity(0)
        full_huffman_chart.vertical_axis_label.set_opacity(0)
        full_huffman_chart.reference_lines.set_opacity(0)
        self.camera.frame.match_x(full_huffman_chart.bars)
        self.add(full_huffman_chart)
        self.wait(1.5)

        # Calculate the entropy
        weighted_sum_lines = VGroup(
            Tex(R"\frac{1}{2} \cdot 1", font_size = 36).next_to(full_huffman_chart.bars[0], UP),
            Tex(R"\frac{1}{4} \cdot 2", font_size = 36).next_to(full_huffman_chart.bars[1], UP),
            Tex(R"\frac{1}{8} \cdot 3", font_size = 36).next_to(full_huffman_chart.bars[2], UP),
            Tex(R"\frac{1}{8} \cdot 3", font_size = 36).next_to(full_huffman_chart.bars[3], UP)
        )
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    TransformFromCopy(label, line[:-2]),
                    FadeIn(line[-2:], shift = UP*0.1)
                , run_time = 1.5)
                for label, line in zip(full_huffman_chart.probability_labels, weighted_sum_lines)
            ], lag_ratio = 0.3)
        )
        self.wait(0.35)

        # Show the weighted sum result
        sum_result = Tex(
            R"\frac{1}{2} \cdot 1 + \frac{1}{4} \cdot 2 + \frac{1}{8} \cdot 3 + \frac{1}{8} \cdot 3 = 1.75 \text{ bits}",
            font_size = 45
        ).to_edge(UP, buff = 1.5).fix_in_frame()

        self.play(TransformMatchingShapes(weighted_sum_lines, sum_result[:-9], path_arc = PI*0.2, run_time = 1.5))
        self.play(FadeIn(sum_result[R"= 1.75 \text{ bits}"]))
        self.wait(2)

        # Compare the message length to the information
        information_labels = VGroup(*[
            Tex(
                str(num_bits) + R"\text{ bit" + ("s" if num_bits > 1 else "") + "}",
                font_size = 40
            ).set_stroke(
                width = 3, color = BLACK, behind = True
            ).next_to(full_huffman_chart.bars[i], UP)
            for i, num_bits in enumerate([1, 2, 3, 3])
        ])
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    Indicate(VGroup(bar, label), scale_factor = 1.1, suspend_mobject_updating = True),
                    FadeIn(info_label, shift = UP*0.1)
                )
                for bar, label, info_label in zip(full_huffman_chart.bars, full_huffman_chart.bar_labels, information_labels)
            ], lag_ratio = 0.2),
            full_huffman_chart.bar_labels.animate.shift(0)
        , run_time = 1.3)

        refernce_lines_opacity_tracker = ValueTracker(0)
        full_huffman_chart.reference_lines.add_updater(lambda m: self.bring_to_back(m.set_opacity(refernce_lines_opacity_tracker.get_value())))
        self.play(
            sum_result.animate.scale(0.8).to_edge(UP, buff = 0.7).match_x(full_huffman_chart.reference_lines),
            VGroup(
                full_huffman_chart.vertical_axis,
                full_huffman_chart.vertical_axis_label
            ).animate.set_opacity(1),
            refernce_lines_opacity_tracker.animate.set_value(1),
            self.camera.frame.animate.match_x(full_huffman_chart),
            VGroup(
                full_huffman_chart.bars,
                full_huffman_chart.bar_labels
            ).animate.shift(0)
        , run_time = 1)

        full_huffman_chart.suspend_updating()
        full_huffman_chart.save_state()
        information_labels.save_state()
        reference_lines_order = [1, 2, 3, 3]
        for i in range(4):
            self.play(
                VGroup(
                    information_labels[i],
                    full_huffman_chart.bars[i],
                    full_huffman_chart.bar_labels[i],
                    full_huffman_chart.segments.bars[i],
                    full_huffman_chart.event_labels[i],
                    full_huffman_chart.probability_labels[i],
                ).animate.set_opacity(1),
                full_huffman_chart.reference_lines[reference_lines_order[i]].animate.set_opacity(1).set_color(YELLOW),
                VGroup(
                    information_labels[:i],
                    full_huffman_chart.bars[:i],
                    full_huffman_chart.bar_labels[:i],
                    full_huffman_chart.segments.bars[:i],
                    full_huffman_chart.event_labels[:i],
                    full_huffman_chart.probability_labels[:i],
                    information_labels[i + 1:],
                    full_huffman_chart.bars[i + 1:],
                    full_huffman_chart.bar_labels[i + 1:],
                    full_huffman_chart.segments.bars[i + 1:],
                    full_huffman_chart.event_labels[i + 1:],
                    full_huffman_chart.probability_labels[i + 1:]
                ).animate.set_opacity(0.1),
                full_huffman_chart.reference_lines[:reference_lines_order[i]].animate.set_opacity(0.1).set_color(WHITE),
                full_huffman_chart.reference_lines[reference_lines_order[i] + 1:].animate.set_opacity(0.1).set_color(WHITE)
            , run_time = 0.8)
        self.play(
            full_huffman_chart.animate.restore(),
            information_labels.animate.restore()
        )
        full_huffman_chart.resume_updating()

        # Generalize 
        distribution = random_distribution(7)
        general_chart = EntropyChart(
            random_distribution(7),
            event_labels = VGroup(*[
                Tex(
                    (("s_" + str(i)) if i < len(distribution) - 2 else R"\ldots" if i == len(distribution) - 2 else "s_n")
                ).scale(0.8).set_color(BLACK)
                for i in range(len(distribution))
            ]),
            probability_labels = VGroup(*[
                Tex(
                    (("p_" + str(i)) if i < len(distribution) - 2 else R"\ldots" if i == len(distribution) - 2 else "p_n")
                )
                for i in range(len(distribution))
            ]),
            width = 9,
            height = 3,
            segments_height = 0.4,
            fit_event_labels_to_height = False,
            fill_colors = [YELLOW_B, YELLOW_D]
        ).match_width(full_huffman_chart).match_x(full_huffman_chart).align_to(full_huffman_chart, UP)
        general_chart.bars.add_updater(lambda m: m.set_stroke(width = 1))
        general_chart.segments.bars.set_stroke(width = 1)
        self.camera.frame.save_state()
        full_huffman_chart.suspend_updating()
        self.play(
            # self.camera.frame.animate(run_time = 5).match_x(general_chart.segments),
            AnimationGroup(
                FadeOut(
                    VGroup(
                        sum_result,
                        information_labels,
                        full_huffman_chart.bars,
                        full_huffman_chart.bar_labels,
                        full_huffman_chart.segments.bars,
                        full_huffman_chart.event_labels,
                        full_huffman_chart.probability_labels
                    )
                , suspend_mobject_updating = True),
                AnimationGroup(*[
                    AnimationGroup(GrowFromCenter(segment), GrowFromEdge(bar, DOWN), FadeIn(p_label))
                    for segment, bar, p_label in zip(
                        general_chart.segments.bars, general_chart.bars, general_chart.probability_labels
                    )
                ], lag_ratio = 0.2, suspend_mobject_updating = True),
                AnimationGroup(
                    FadeOut(VGroup(full_huffman_chart.vertical_axis, full_huffman_chart.vertical_axis_label, full_huffman_chart.reference_lines)),
                    FadeIn(VGroup(general_chart.vertical_axis, general_chart.vertical_axis_label, general_chart.reference_lines))
                ),
                AnimationGroup(*[
                    FadeIn(symbol)
                    for symbol in general_chart.event_labels
                ], lag_ratio = 0.3)
            , lag_ratio = 0.5, run_time = 6)
        )

        # Move around the probabilities and write the definition of entropy
        self.add(general_chart)
        general_chart.save_state()
        bars_opacity_tracker = ValueTracker(0.8)
        general_chart.bars.add_updater(lambda m: m.set_opacity(bars_opacity_tracker.get_value()))
        general_chart.bars.add_updater(lambda m: self.bring_to_front(m))
        for i in range(20):
            anims = [general_chart.set_distribution(random_distribution(7))]
            if i == 1:
                weighted_sum_formula = Tex(
                    R"\text{Avg. information} = \sum_i p_i (-\log_2 p_i)"
                ).next_to(general_chart.reference_lines, UP)
                anims.append(Write(weighted_sum_formula, run_time = 9))
            if i == 5:
                anims.append(bars_opacity_tracker.animate.set_value(0.1))
            if i == 7:
                brace = Brace(general_chart.segments, UP)
                total_width_text = brace.get_tex(R"\text{Total width} = 1")
                anims.append(
                    AnimationGroup(
                        GrowFromEdge(brace, DOWN),
                        Write(total_width_text)
                    , run_time = 2)
                )
            if i == 8:
                anims.append(
                    AnimationGroup(
                        FadeOut(VGroup(brace, total_width_text)),
                        bars_opacity_tracker.animate.set_value(1)
                    )
                )
            if i == 10:
                anims.append(FancyCircumscribe(general_chart.vertical_axis_label, run_time=5))
            if i == 13:
                # Add the area formula
                weighted_sum_formula_area = Tex(
                    R"\text{Avg. information} = \sum_i p_i (-\log_2 p_i) = \text{Area}()"
                ).move_to(weighted_sum_formula)
                bars_copy = general_chart.bars.copy()
                bars_copy.clear_updaters().set_opacity(1)
                bars_copy.generate_target()
                bars_copy_height = bars_copy.get_height()
                scale_factor = weighted_sum_formula_area[-2].get_height()/bars_copy_height
                bars_copy.target.stretch(0.5, 0).scale(scale_factor).next_to(weighted_sum_formula_area[-2], RIGHT)
                weighted_sum_formula_area[-1].next_to(bars_copy.target, RIGHT)
                weighted_sum_formula_area[-1].shift(DOWN*(weighted_sum_formula_area[-1].get_y() - weighted_sum_formula_area[-2].get_y()))
                VGroup(weighted_sum_formula_area, bars_copy.target).set_x(0)
                self.play(
                    AnimationGroup(
                        AnimationGroup(
                            ReplacementTransform(weighted_sum_formula, weighted_sum_formula_area[:-len("=Area()")], run_time = 1.5),
                            Write(weighted_sum_formula_area[-len("=Area()"):], run_time = 2)
                        , lag_ratio = 0.6),
                        MoveToTarget(bars_copy, run_time = 3)
                    )
                )
                def update_area_bars(m):
                    m.become(
                        general_chart.bars.copy()
                        .clear_updaters()
                        .set_opacity(1)
                        .stretch(0.5, 0)
                        .scale(scale_factor)
                        .next_to(weighted_sum_formula_area[-2], RIGHT)
                        .align_to(m, DOWN)
                    )
                bars_copy.add_updater(update_area_bars)
            if i == 18:
                # Write it as H(P)
                weighted_sum_formula_h = Tex(
                    R"H(P) = \sum_i p_i (-\log_2 p_i) = \text{Area}()"
                ).move_to(weighted_sum_formula_area).align_to(weighted_sum_formula_area, RIGHT)
                weighted_sum_formula_h[:-1].align_to(weighted_sum_formula_area[:-1], RIGHT)
                weighted_sum_formula_h[-1].next_to(bars_copy.target, RIGHT)
                weighted_sum_formula_h[-1].shift(DOWN*(weighted_sum_formula_h[-1].get_y() - weighted_sum_formula_h[-2].get_y()))
                weighted_sum_formula_h.match_x(general_chart.reference_lines)
                entropy_text = TexText("``Entropy''", font_size = 25).next_to(weighted_sum_formula_h["H(P)"], DOWN)
                anims.append(
                    AnimationGroup(
                        AnimationGroup(
                            weighted_sum_formula_area[len("Avg.information"):].animate.align_to(weighted_sum_formula_h["="][0], LEFT),
                            ReplacementTransform(weighted_sum_formula_area["Avg. information"], weighted_sum_formula_h["H(P)"])
                        , run_time = 2),
                        FadeIn(entropy_text, run_time = 2)
                    , lag_ratio = 0.4)
                )
            if i == 19:
                entropy_display = Tex(
                    R"= 0.00 \text{ bits}", font_size = 32
                ).next_to(
                    weighted_sum_formula_h[-2:], DOWN, buff = 0.17
                )
                entropy_display_opacity_tracker = ValueTracker(0)
                entropy_display.add_updater(lambda m: m.set_opacity(entropy_display_opacity_tracker.get_value()))
                entropy_value = entropy_display.make_number_changeable("0.00")
                entropy_value.add_updater(
                    lambda m: m.set_value(
                        sum([t.get_value()*-math.log2(t.get_value()) for t in general_chart.distribution_trackers])
                    )
                )
                entropy_display.add_updater(lambda m: self.bring_to_front(m))
                self.add(entropy_display)
                anims.append(entropy_display_opacity_tracker.animate.set_value(1))

            self.play(*anims, run_time = 3)
        self.remove(weighted_sum_formula_area)
        self.add(weighted_sum_formula_h)

        # Show a uniform distribution
        event_labels_opacity_tracker = ValueTracker(1)
        general_chart.segments.add_updater(lambda m: m.labels.set_opacity(event_labels_opacity_tracker.get_value()))
        self.play(
            general_chart.probability_labels.animate.set_opacity(0),
            event_labels_opacity_tracker.animate.set_value(0),
            general_chart.set_distribution([1/7 for _ in range(7)])
        , run_time = 3)
        self.wait(2)

        # Show a squished distribution
        big_prob = 0.789
        leftover = 1 - big_prob
        leftover_distibution = random_distribution(6, thresh = (2**-7)/leftover)
        leftover_distibution = [p*leftover for p in leftover_distibution]
        distribution = [big_prob] + leftover_distibution
        self.play(general_chart.set_distribution(distribution), run_time = 6)
        self.wait(4)

        # Indicate the most probable event with little information
        general_chart.save_state()
        entropy_display.save_state()
        bars_copy.save_state()
        weighted_sum_formula_h.save_state()
        entropy_text.save_state()
        general_chart.suspend_updating()
        entropy_display.suspend_updating()
        bars_copy.suspend_updating()
        weighted_sum_formula.suspend_updating()
        self.play(
            VGroup(
                general_chart.bars[1:],
                general_chart.segments.bars[1:],
                general_chart.reference_lines,
                general_chart.vertical_axis,
                general_chart.vertical_axis_label,
                entropy_display,
                bars_copy,
                weighted_sum_formula_h,
                entropy_text
            ).animate.fade(0.8)
        , run_time = 2)
        arrow = Arrow(ORIGIN, DOWN).set_color(TEAL).next_to(general_chart.bars[0], UP)
        label = TexText(
            R"high probability $\Longleftrightarrow$ low information",
            font_size = 40
        ).set_color(TEAL).next_to(arrow, UP, buff = 0.15)
        self.play(GrowArrow(arrow), FadeIn(label, run_time = 1.2))
        self.wait(2)
        self.play(
            general_chart.animate.restore(),
            entropy_display.animate.restore(),
            bars_copy.animate.restore(),
            weighted_sum_formula_h.animate.restore(),
            entropy_text.animate.restore(),
            FadeOut(VGroup(arrow, label))
        , run_time = 2)
        general_chart.resume_updating()
        entropy_display.resume_updating()
        bars_copy.resume_updating()
        weighted_sum_formula_area.resume_updating()
        self.wait(2)

        # Divide the probability space into more chunks
        entropy_value.clear_updaters()
        entropy_display.add_updater(lambda m: self.bring_to_front(m))
        prev_chart = general_chart
        for i in range(1, 3):
            new_distribution = []
            for p in distribution:
                new_distribution += [p/2**i for _ in range(2**i)]
            new_chart = EntropyChart(
                new_distribution,
                event_labels = None,
                probability_labels = None,
                width = 9,
                height = 3,
                segments_height = 0.4,
                fit_event_labels_to_height = False,
                fill_colors = [YELLOW_B, YELLOW_D]
            ).match_width(general_chart).match_x(general_chart).align_to(general_chart, UP)
            new_chart.bars.add_updater(lambda m: m.set_stroke(width = 0.3/2**i))
            new_chart.segments.bars.set_stroke(width = 0.5/2**i)
            general_chart.clear_updaters()
            bars_copy.clear_updaters()
            self.play(
                FadeOut(prev_chart, suspend_mobject_updating = True, run_time = 3),
                FadeIn(new_chart, suspend_mobject_updating = True, run_time = 3),
                bars_copy.animate(run_time = 3).become(
                    new_chart.bars.copy()
                    .clear_updaters()
                    .set_opacity(1)
                    .stretch(0.5, 0)
                    .scale(scale_factor)
                    .next_to(weighted_sum_formula_area[-2], RIGHT)
                    .align_to(bars_copy, DOWN)
                ),
                entropy_value.animate(run_time = 1.3).set_value(
                    sum([t.get_value()*-math.log2(t.get_value()) for t in new_chart.distribution_trackers])
                )
            )
            self.wait(2)
            prev_chart = new_chart

        def update_area_bars(m):
            m.become(
                new_chart.bars.copy()
                .clear_updaters()
                .set_opacity(1)
                .stretch(0.5, 0)
                .scale(scale_factor)
                .next_to(weighted_sum_formula_area[-2], RIGHT)
                .align_to(m, DOWN)
            )
        bars_copy.add_updater(update_area_bars)

        # Squish and then spread out the new distribution
        entropy_value.add_updater(
            lambda m: m.set_value(
                sum([t.get_value()*-math.log2(t.get_value()) for t in new_chart.distribution_trackers])
            )
        )
        big_prob_1 = 0.42
        big_prob_2 = 0.178
        big_prob_3 = 0.32
        leftover = 1 - big_prob_1 - big_prob_2 - big_prob_3
        leftover_distibution = random_distribution(25, thresh = (2**-10)/leftover)
        leftover_distibution = [p*leftover for p in leftover_distibution]
        distribution = [big_prob_1, big_prob_2, big_prob_3] + leftover_distibution
        self.play(new_chart.set_distribution(distribution), run_time = 6)
        self.wait(1)

        uniform_distribution = [1/28 for _ in range(28)]
        self.play(new_chart.set_distribution(uniform_distribution), run_time = 6)
        self.wait(1)

        # Show some more random distributions
        for _ in range(10):
            self.play(new_chart.set_distribution(random_distribution(28, thresh = 2**-8)), run_time = 3)



class CrossEntropyDefinition(InteractiveScene):
    def construct(self):
        # Show the chart for the first distribution
        encoding = ["0", "10", "110", "111"]
        first_distribution = [1/2, 1/4, 1/8, 1/8]
        first_distribution_chart = EntropyChart(
            first_distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]),
            probability_labels = [
                Tex(R"\frac{1}{" + ["2", "4", "8", "8"][i] + "}", font_size = 65)
                for i in range(4)
            ],
            bar_labels = [
                Tex(encoding[i], font_size = 57)
                for i in range(4)
            ],
            bar_heights = [1, 2, 3, 3],
            width = 12,
            height = 4.5,
            include_vertical_axis = False,
            segments_height = 1,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).scale(0.3).to_corner(UL, buff = 0.7).shift(DOWN*0.7)
        first_distribution_chart.update()
        first_distribution_chart.clear_updaters()
        first_distribution_chart.bar_labels.set_color(WHITE)
        self.play(first_distribution_chart.create(), run_time = 2)
        self.wait(2)
        self.play(first_distribution_chart.animate.set_x(-FRAME_WIDTH*0.25), run_time = 2.5)
        self.wait(2)

        # Build the segments for the second chart
        second_distribution = [1/8, 1/8, 1/4, 1/2]
        second_distribution_chart = EntropyChart(
            second_distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]),
            probability_labels = [
                Tex(R"\frac{1}{" + ["8", "8", "4", "2"][i] + "}", font_size = 65)
                for i in range(4)
            ],
            bar_labels = [
                Tex(encoding[i], font_size = 57)
                for i in range(4)
            ],
            bar_heights = [1, 2, 3, 3],
            width = 12,
            height = 4.5,
            include_vertical_axis = False,
            segments_height = 1,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).scale(0.3).to_edge(UP, buff = 1.4).set_x(FRAME_WIDTH*0.25)
        second_distribution_chart.update()
        second_distribution_chart.clear_updaters()
        second_distribution_chart.bar_labels.set_color(WHITE)
        for segment, e_label, p_label in list(zip(
            second_distribution_chart.segments.bars,
            second_distribution_chart.event_labels,
            second_distribution_chart.probability_labels
        )):
            self.play(
                AnimationGroup(
                    GrowFromCenter(segment),
                    FadeIn(e_label),
                    FadeIn(p_label),
                    suspend_mobject_updating = True
                , run_time = 2)
            )
            self.wait(2)
        self.wait(4)

        # The bars hop over from the old distribution to the new one
        self.play(
            AnimationGroup(*[
                TransformFromCopy(VGroup(bar1, label1), VGroup(bar2, label2), run_time = 3)
                for bar1, label1, bar2, label2 in list(zip(
                    first_distribution_chart.bars,
                    first_distribution_chart.bar_labels,
                    second_distribution_chart.bars,
                    second_distribution_chart.bar_labels
                ))[::-1]
            ], lag_ratio = 0.3)
        )
        self.wait(2)

        # Center everything
        self.play(
            VGroup(first_distribution_chart, second_distribution_chart).animate.scale(1.5).arrange(buff = 2).shift(DOWN*0.5)
        , run_time = 2.5)

        # Calculate the cross entropy
        weighted_sum_lines = VGroup(
            Tex(R"\frac{1}{8} \cdot 1", font_size = 26).next_to(second_distribution_chart.bars[0], UP),
            Tex(R"\frac{1}{8} \cdot 2", font_size = 26).next_to(second_distribution_chart.bars[1], UP),
            Tex(R"\frac{1}{4} \cdot 3", font_size = 26).next_to(second_distribution_chart.bars[2], UP),
            Tex(R"\frac{1}{2} \cdot 3", font_size = 26).next_to(second_distribution_chart.bars[3], UP)
        )
        for line in weighted_sum_lines:
            line[:3].set_color(GREEN)
            line[4:].set_color(PINK)

        self.play(
            AnimationGroup(
                TransformFromCopy(second_distribution_chart.probability_labels[0], weighted_sum_lines[0][:-2]),
                FadeIn(weighted_sum_lines[0][-2:], shift = UP*0.1)
            , run_time = 1.5)
        )
        self.wait(1.5)
        self.play(
            AnimationGroup(
                TransformFromCopy(second_distribution_chart.probability_labels[1], weighted_sum_lines[1][:-2]),
                FadeIn(weighted_sum_lines[1][-2:], shift = UP*0.1)
            , run_time = 1.5)
        )
        self.wait(2.5)
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    TransformFromCopy(label, line[:-2]),
                    FadeIn(line[-2:], shift = UP*0.1)
                , run_time = 1.5)
                for label, line in zip(second_distribution_chart.probability_labels[2:], weighted_sum_lines[2:])
            ], lag_ratio = 0.3)
        )

        self.wait(1)
        sum_result = Tex(
            R"\frac{1}{8} \cdot 1 + \frac{1}{8} \cdot 2 + \frac{1}{4} \cdot 3 + \frac{1}{2} \cdot 3 \\ = 2.625 \text{ bits}",
            font_size = 32,
            tex_to_color_map = {
                R"\frac{1}{8}": GREEN,
                R"\frac{1}{4}": GREEN,
                R"\frac{1}{2}": GREEN,
                " 1 ": PINK,
                " 2 ": PINK,
                " 3 ": PINK
            }
        ).next_to(second_distribution_chart, UP, buff = 0.3)

        self.play(TransformMatchingShapes(weighted_sum_lines, sum_result[:-10], path_arc = PI*0.2, run_time = 1.5))
        self.wait(0.5)
        self.play(FadeIn(sum_result[R"= 2.625 \text{ bits}"]))
        self.wait(2)

        # Write "cross entropy"
        cross_entropy_text = TexText("Cross Entropy:").set_fill(color = [PINK, GREEN]).next_to(sum_result, UP)
        for i, letter in enumerate(cross_entropy_text):
            letter.set_color(interpolate_color(PINK, GREEN, i/(len(cross_entropy_text) - 1)))
        self.play(Write(cross_entropy_text, run_time = 2.5))
        self.wait(0.5)
        rect1 = SurroundingRectangle(
            first_distribution_chart.probability_labels, stroke_width = 2, stroke_color = PINK
        ).stretch_to_fit_width(first_distribution_chart.bars.get_width()).match_x(first_distribution_chart.bars)
        self.play(FadeIn(rect1), run_time = 1.5)
        self.wait(3)
        rect2 = SurroundingRectangle(
            second_distribution_chart.probability_labels, stroke_width = 2, stroke_color = GREEN
        ).stretch_to_fit_width(second_distribution_chart.bars.get_width()).match_x(second_distribution_chart.bars)
        self.play(ReplacementTransform(rect1, rect2), run_time = 2.5)
        self.wait(2)
        self.play(FadeOut(rect2))
        self.wait(2)

        # Label the two charts with p and q
        p_chart = EntropyChart(
            second_distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]).scale(0.6) for i in range(4)]),
            probability_labels = VGroup(*[Tex(f"p_{i + 1}", font_size = 42) for i in range(4)]),
            bar_labels = [
                Tex(encoding[i], font_size = 27)
                for i in range(4)
            ],
            bar_heights = [1, 2, 3, 3],
            width = second_distribution_chart.get_width(),
            height = 3,
            include_vertical_axis = False,
            segments_height = 0.5,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        )
        p_chart.suspend_updating()
        p_chart.bar_labels.set_color(WHITE)

        q_chart = EntropyChart(
            first_distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]).scale(0.6) for i in range(4)]),
            probability_labels = VGroup(*[Tex(f"q_{i + 1}", font_size = 42) for i in range(4)]),
            bar_labels = [
                Tex(encoding[i], font_size = 27)
                for i in range(4)
            ],
            bar_heights = [1, 2, 3, 3],
            width = first_distribution_chart.get_width(),
            height = 3,
            include_vertical_axis = False,
            segments_height = 0.5,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        )
        q_chart.suspend_updating()
        q_chart.bar_labels.set_color(WHITE)

        p_chart.match_x(second_distribution_chart).align_to(second_distribution_chart.bars, UP)
        q_chart.match_x(first_distribution_chart).align_to(first_distribution_chart.bars, UP)
        self.play(ReplacementTransform(second_distribution_chart, p_chart, suspend_mobject_updating = True), run_time = 2)
        self.wait(2)
        self.play(ReplacementTransform(first_distribution_chart, q_chart, suspend_mobject_updating = True), run_time = 2)
        self.wait(3)

        # Show the encoding and the original distribution
        arrow = Arrow(q_chart.bars, p_chart.bars)
        arrow_label = TexText("encoding", font_size = 27).next_to(arrow, UP, buff = 0.1)
        arrow_label_2 = TexText(R"codeword length: \\ $-\log_2 q_i$", font_size = 20).next_to(arrow, DOWN, buff = 0.1)
        arrow_label_2["q_i"].set_color(PINK)
        q_chart.save_state()
        p_chart.save_state()
        self.play(
            VGroup(
                p_chart.segments,
                p_chart.probability_labels
            ).animate.fade(0.8),
            GrowArrow(arrow, run_time = 1.4),
            Write(arrow_label, run_time = 1.5)
        )
        self.wait(4)
        self.play(FadeIn(arrow_label_2))
        self.wait(4)
        self.play(
            FadeOut(VGroup(arrow, arrow_label, arrow_label_2)),
            q_chart.animate.fade(0.8),
            p_chart.animate.restore(),
            p_chart.bars.animate.fade(0.8),
            p_chart.bar_labels.animate.fade(0.8)
        )
        self.wait(1)
        self.play(q_chart.animate.restore(), p_chart.animate.restore(), run_time = 2)
        general_equation = TexText(
            R"Avg. bits per instruction: \\[0.1in] $\displaystyle\sum_i p_i (-\log_2 q_i)$",
            font_size = 40,
            tex_to_color_map = {
                "p_i": GREEN,
                "q_i": PINK
            }
        ).next_to(second_distribution_chart, UP, buff = 0.3)
        self.play(FadeOut(VGroup(cross_entropy_text, sum_result)), FadeIn(general_equation), run_time = 1.5)
        self.wait(2)

        # Replace "Avg. bits per instruction" with "Cross Entropy(Q, P)"
        cross_entropy_text = TexText(
            "``cross entropy of Q relative to P''",
            tex_to_color_map = {"Q": PINK, "P": GREEN}
        ).match_height(
            general_equation["Avg. bits per instruction:"]
        ).move_to(
            general_equation["Avg. bits per instruction:"]
        )
        self.play(FadeOut(general_equation["Avg. bits per instruction:"]), FadeIn(cross_entropy_text))
        self.wait(3)

        # Show special notation
        full_sum = general_equation[len("Avg.bitsperinstruction:"):]
        self.play(
            FadeOut(VGroup(q_chart, p_chart), run_time = 1.6, shift = DOWN*2),
            VGroup(cross_entropy_text, full_sum).animate(run_time = 2).set_y(0).to_edge(LEFT, buff = 2)
        )
        notations = BulletedList(
            R"$H(P, Q)$",
            R"$H(P \parallel Q)$",
            R"$H_Q(P)$",
            R"$\mathbb{E}_P[-\log Q]$",
            R"$\langle -\log Q \rangle_P$",
            tex_to_color_map = {"Q": PINK, "P": GREEN}
        ).to_edge(RIGHT, buff = 2)
        brace = Brace(notations, LEFT)
        self.play(
            GrowFromEdge(brace, RIGHT),
            AnimationGroup(*[FadeIn(line, shift = DOWN*0.3) for line in notations], lag_ratio = 0.2)
        , run_time = 3)
        self.wait(2)

        # Focus on the full sum
        self.play(
            FadeOut(VGroup(brace, notations), shift = RIGHT*3),
            VGroup(cross_entropy_text, full_sum).animate.scale(1.2).center()
        , run_time = 2)
        self.wait(2)

        # Show where the spacing of the bars and the heights of the bars come from
        self.play(VGroup(cross_entropy_text, full_sum).animate.to_edge(UP, buff = 0.7))
        np.random.seed(0)
        Q = [0.4, 0.1, 0.08, 0.15, 0.27]
        P = [0.1, 0.2, 0.3, 0.35, 0.05]
        q_entropy_chart = EntropyChart(
            Q,
            event_labels = None,
            probability_labels = VGroup(*[
                Tex(
                    (("q_" + str(i + 1)) if i < len(Q) - 2 else R"\ldots" if i == len(P) - 2 else "q_n"),
                    font_size = 40
                )
                for i in range(len(Q))
            ]),
            bar_labels = None,
            width = 5,
            height = 3.5,
            include_vertical_axis = False,
            segments_height = 0.2,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK],
            stroke_width = 1
        )
        p_cross_entropy_chart = EntropyChart(
            P,
            event_labels = None,
            probability_labels = VGroup(*[
                Tex(
                    (("p_" + str(i + 1)) if i < len(P) - 2 else R"\ldots" if i == len(P) - 2 else "p_n"),
                    font_size = 40
                )
                for i in range(len(P))
            ]),
            bar_labels = None,
            bar_heights = [-math.log2(q) for q in Q],
            width = 5,
            height = 3.5,
            include_vertical_axis = False,
            segments_height = 0.2,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [PURE_MAGENTA, LIGHT_PINK],
            stroke_width = 1
        )
        q_entropy_chart.generate_target()
        VGroup(q_entropy_chart.target, p_cross_entropy_chart).arrange(buff = 1.5).to_edge(DOWN, buff = 1)
        q_entropy_chart.to_edge(DOWN, buff = 1)
        self.play(q_entropy_chart.create())
        self.wait(1.5)
        self.play(MoveToTarget(q_entropy_chart), run_time = 1.5)

        self.play(
            AnimationGroup(*[
                TransformFromCopy(bar1, bar2, run_time = 3)
                for bar1, bar2 in list(zip(
                    q_entropy_chart.bars,
                    p_cross_entropy_chart.bars,
                ))[::-1]
            ], lag_ratio = 0.3),
            AnimationGroup(*[
                FadeIn(VGroup(segment, prob))
                for segment, prob in list(zip(
                    p_cross_entropy_chart.segments.bars,
                    p_cross_entropy_chart.probability_labels
                ))[::-1]
            ])
        , run_time = 4)
        self.add(p_cross_entropy_chart)

        # Change the distribution P to be more similar to Q
        Almost_Q = [0.34, 0.16, 0.03, 0.22, 0.25]
        self.play(p_cross_entropy_chart.set_distribution(Almost_Q), run_time = 2)
        self.wait(2)

        # Show more inefficient distributions
        self.play(p_cross_entropy_chart.set_distribution(random_distribution(5)), run_time = 2)
        self.wait(1)
        self.play(p_cross_entropy_chart.set_distribution(P), run_time = 2)
        for _ in range(3):
            self.play(p_cross_entropy_chart.set_distribution(random_distribution(5)), run_time = 2)



class KLDivergenceDefinition(InteractiveScene):
    def construct(self):
        # Add the charts and the entropy calculation
        encoding = ["0", "10", "110", "111"]
        first_distribution = [1/2, 1/4, 1/8, 1/8]
        first_distribution_chart = EntropyChart(
            first_distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]),
            probability_labels = [
                Tex(R"\frac{1}{" + ["2", "4", "8", "8"][i] + "}", font_size = 65)
                for i in range(4)
            ],
            bar_labels = [
                Tex(encoding[i], font_size = 57)
                for i in range(4)
            ],
            bar_heights = [1, 2, 3, 3],
            width = 9,
            height = 3.6,
            include_vertical_axis = False,
            segments_height = 1,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).scale(0.38)
        first_distribution_chart.update()
        first_distribution_chart.clear_updaters()
        first_distribution_chart.bar_labels.set_color(WHITE)
        self.add(first_distribution_chart)

        second_distribution = [1/8, 1/8, 1/4, 1/2]
        second_distribution_chart = EntropyChart(
            second_distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]),
            probability_labels = [
                Tex(R"\frac{1}{" + ["8", "8", "4", "2"][i] + "}", font_size = 65)
                for i in range(4)
            ],
            bar_labels = [
                Tex(encoding[i], font_size = 57)
                for i in range(4)
            ],
            bar_heights = [1, 2, 3, 3],
            width = 9,
            height = 3.6,
            include_vertical_axis = False,
            segments_height = 1,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).scale(0.38)
        second_distribution_chart.update()
        second_distribution_chart.clear_updaters()
        second_distribution_chart.bar_labels.set_color(WHITE)
        first_distribution_chart.to_edge(UP, buff = 1.1).to_edge(LEFT, buff = 1)
        second_distribution_chart.to_edge(UP, buff = 1.1).to_edge(RIGHT, buff = 1)
        arrow = Arrow(
            first_distribution_chart.bars.get_right(),
            second_distribution_chart.bars.get_left()
        )
        self.add(second_distribution_chart, arrow)
        self.wait(4)

        sum_result = Tex(
            R"\frac{1}{8} \cdot 1 + \frac{1}{8} \cdot 2 + \frac{1}{4} \cdot 3 + \frac{1}{2} \cdot 3 = 2.625 \text{ bits}",
            font_size = 20,
            tex_to_color_map = {
                R"\frac{1}{8}": GREEN,
                R"\frac{1}{4}": GREEN,
                R"\frac{1}{2}": GREEN,
                " 1 ": PINK,
                " 2 ": PINK,
                " 3 ": PINK
            }
        ).next_to(second_distribution_chart, UP, buff = 0.2)
        self.add(sum_result)

        Q_brace = Brace(first_distribution_chart.segments, DOWN).shift(DOWN*0.65)
        Q_label = Q_brace.get_tex("Q").set_color(PINK)
        self.add(Q_brace, Q_label)
        P_brace = Brace(second_distribution_chart.segments, DOWN).shift(DOWN*0.65)
        P_label = P_brace.get_tex("P").set_color(GREEN)
        self.add(P_brace, P_label)

        # Add the entropy chart for P
        second_distribution_entropy_chart = EntropyChart(
            second_distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]),
            probability_labels = [
                Tex(R"\frac{1}{" + ["8", "8", "4", "2"][i] + "}", font_size = 65)
                for i in range(4)
            ],
            bar_labels = [
                Tex(encoding[i], font_size = 57)
                for i in range(4)
            ],
            bar_heights = [3, 3, 2, 1],
            width = 9,
            height = 3.6,
            include_vertical_axis = False,
            segments_height = 1,
            fill_colors = [GREEN_B, GREEN_D]
        ).scale(0.38)
        second_distribution_entropy_chart.update()
        second_distribution_entropy_chart.clear_updaters()
        second_distribution_entropy_chart.bar_labels.set_opacity(0)
        second_distribution_entropy_chart.to_edge(DOWN, buff = 0.3).to_edge(RIGHT, buff = 1)
        copy = second_distribution_chart.copy()
        copy.set_opacity(0)
        entropy_sum = Tex(
            R"\frac{1}{8} \cdot 3 + \frac{1}{8} \cdot 3 + \frac{1}{4} \cdot 2 + \frac{1}{2} \cdot 1 = 1.75 \text{ bits}",
            font_size = 20,
            tex_to_color_map = {
                R"\frac{1}{8}": GREEN,
                R"\frac{1}{4}": GREEN,
                R"\frac{1}{2}": GREEN,
                " 1 ": GREEN,
                " 2 ": GREEN,
                " 3 ": GREEN
            }
        ).next_to(second_distribution_entropy_chart, UP, buff = 0.2)
        kl_divergence_equation = Tex(
            R"\left(\displaystyle\sum_i p_i \cdot -\log_2 q_i\right) - \left(\displaystyle\sum_i p_i \cdot -\log_2 p_i\right)",
            font_size = 45, tex_to_color_map = {"p_i": GREEN, "q_i": PINK}
        ).to_corner(DL, buff = 1)
        self.play(
            Write(kl_divergence_equation, run_time = 3.5),
            AnimationGroup(
                ReplacementTransform(copy, second_distribution_entropy_chart, run_time = 2),
                FadeIn(entropy_sum)
            , lag_ratio = 0.8)
        )

        # Highlight the cross entropy chart and the original distribution chart
        first_distribution_chart.bar_labels.add_updater(lambda m: self.bring_to_front(m))
        second_distribution_chart.bar_labels.add_updater(lambda m: self.bring_to_front(m))
        second_distribution_entropy_chart.bar_labels.add_updater(lambda m: self.bring_to_front(m))
        for _ in range(3):
            self.play(
                AnimationGroup(*[
                    AnimationGroup(Indicate(bar1, scale_factor = 1.1), Indicate(bar2, scale_factor = 1.1))
                    for bar1, bar2 in zip(first_distribution_chart.bars, second_distribution_chart.bars)
                ], lag_ratio = 0.2)
            )
            self.wait(0.5)
        for _ in range(3):
            self.play(
                AnimationGroup(*[
                    Indicate(bar, scale_factor = 1.1)
                    for bar in second_distribution_entropy_chart.bars
                ], lag_ratio = 0.2)
            )
            self.wait(0.5)

        # Clean up and show the numerical value of the KL divergence
        first_distribution_chart.clear_updaters()
        second_distribution_chart.clear_updaters()
        second_distribution_entropy_chart.clear_updaters()
        chart_group = VGroup(
            VGroup(second_distribution_chart, sum_result),
            VGroup(second_distribution_entropy_chart, entropy_sum)
        )
        chart_group.generate_target()
        chart_group.target.scale(1.3).arrange(buff = 2).to_edge(UP, buff = 1.5)
        kl_divergence_equation.save_state()
        kl_divergence_equation.generate_target()
        kl_divergence_equation.target[:14].match_x(chart_group.target[0])
        kl_divergence_equation.target[14].match_x(chart_group.target)
        kl_divergence_equation.target[15:].match_x(chart_group.target[1])
        self.play(
            AnimationGroup(
                AnimationGroup(
                    FadeOut(VGroup(first_distribution_chart, arrow, Q_brace, Q_label)),
                    FadeOut(VGroup(P_brace, P_label))
                , run_time = 1.2),
                AnimationGroup(
                    MoveToTarget(chart_group, path_arc = PI*0.2),
                    MoveToTarget(kl_divergence_equation)
                )
            , lag_ratio = 0.4)
        , run_time = 3)
        rect = SurroundingRectangle(entropy_sum[R"1.75 \text{ bits}"])
        self.play(ShowCreation(rect), run_time = 1.5)
        self.wait(2)
        self.play(FadeOut(rect), run_time = 2)

        subtraction = Tex(R"2.625 \text{ bits} - 1.75 \text{ bits} = 0.875 \text{ bits}", font_size = 30).to_edge(UP, buff = 0.7)
        subtraction[R"0.875 \text{ bits}"].set_color(YELLOW)
        self.play(
            AnimationGroup(
                TransformFromCopy(sum_result[R"2.625 \text{ bits}"], subtraction[R"2.625 \text{ bits}"], run_time = 1.2),
                Write(subtraction["-"], run_time = 0.4),
                TransformFromCopy(entropy_sum[R"1.75 \text{ bits}"], subtraction[R"1.75 \text{ bits}"], run_time = 2),
                Write(subtraction[R"= 0.875 \text{ bits}"], run_time = 2)
            , lag_ratio = 0.6)
        )

        # Write "KL Divergence"
        kl_divergence_text = TexText("Kullback-Leibler Divergence:", font_size = 60).to_edge(UP, buff = 1)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    FadeOut(
                        VGroup(subtraction, sum_result, entropy_sum, second_distribution_chart, second_distribution_entropy_chart)
                    , shift = UP*3),
                    kl_divergence_equation.animate.restore().center()
                , run_time = 2),
                Write(kl_divergence_text, run_time = 2)
            , lag_ratio = 0.8)
        )
        self.wait(1.6)
        kl_divergence_text_shortened = TexText("KL Divergence:").match_height(kl_divergence_text).move_to(kl_divergence_text)
        self.play(
            TransformMatchingShapes(kl_divergence_text["Kullback-Leibler"], kl_divergence_text_shortened["KL"]),
            TransformMatchingShapes(kl_divergence_text["Divergence:"], kl_divergence_text_shortened["Divergence:"])
        , run_time = 1.3)
        self.wait(2)

        # Show the more compact formula
        self.play(kl_divergence_equation.animate.shift(UP))
        kl_divergence_equation_compact = Tex(
            R"= \displaystyle\sum_i p_i \cdot -\log_2 \frac{p_i}{q_i}",
            font_size = 45, tex_to_color_map = {"p_i": GREEN, "q_i": PINK}
        ).shift(DOWN)
        self.play(FadeIn(kl_divergence_equation_compact))
        self.wait(4)

        # Go back to original definition
        self.play(FadeOut(kl_divergence_equation_compact, shift = DOWN), kl_divergence_equation.animate.shift(DOWN), run_time = 2)
        self.wait(1)
        self.play(Indicate(kl_divergence_equation[:14], scale_factor = 1.1), run_time = 1.5)
        self.play(Indicate(kl_divergence_equation[15:], scale_factor = 1.1), run_time = 1.5)
        self.wait(1.5)
        self.play(Indicate(kl_divergence_equation[:14], scale_factor = 1.1), run_time = 1.5)
        self.wait(1)
        self.play(Indicate(kl_divergence_equation[15:], scale_factor = 1.1), run_time = 1.5)
        self.wait(2)

        # Clean up
        self.play(
            VGroup(kl_divergence_text_shortened, kl_divergence_equation).animate(path_arc = PI*0.2).arrange().to_edge(UP, buff = 1)
        , run_time = 1.5)
        Q_label = Tex("Q", font_size = 60).set_color(PINK).to_edge(DOWN, buff = 1).set_x(-FRAME_WIDTH*0.25)
        P_label = Tex("P", font_size = 60).set_color(GREEN).to_edge(DOWN, buff = 1).set_x(FRAME_WIDTH*0.25).align_to(Q_label, UP)
        Q = [0.8, 0.1, 0.02, 0.05, 0.03]
        P = [0.05, 0.5, 0.2, 0.15, 0.1]
        q_entropy_chart = EntropyChart(
            Q,
            event_labels = None,
            probability_labels = None,
            bar_labels = None,
            width = 5,
            height = 2.5,
            include_vertical_axis = False,
            segments_height = 0.2,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).next_to(Q_label, UP, buff = 0.5)
        p_cross_entropy_chart = EntropyChart(
            P,
            event_labels = None,
            probability_labels = None,
            bar_labels = None,
            bar_heights = [-math.log2(q) for q in Q],
            width = 5,
            height = 2.5,
            include_vertical_axis = False,
            segments_height = 0.2,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).next_to(P_label, UP, buff = 0.5)
        p_entropy_chart = EntropyChart(
            P,
            event_labels = None,
            probability_labels = None,
            bar_labels = None,
            width = 5,
            height = 2.5,
            include_vertical_axis = False,
            segments_height = 0.2,
            fill_colors = [GREEN_B, GREEN_D]
        ).next_to(P_label, UP, buff = 0.5)
        q_entropy_chart.clear_updaters()
        p_cross_entropy_chart.clear_updaters()
        p_entropy_chart.clear_updaters()
        self.play(FadeIn(Q_label), q_entropy_chart.create())
        self.wait(2)
        self.play(FadeIn(P_label), p_cross_entropy_chart.create())
        self.wait(2)
        self.play(ReplacementTransform(p_cross_entropy_chart, p_entropy_chart), run_time = 2)




class CrossEntropyDefinitionV2(InteractiveScene):
    def construct(self):
        # Show the chart for the first distribution
        encoding = ["0", "10", "110", "111"]
        first_distribution = [1/2, 1/4, 1/8, 1/8]
        first_distribution_chart = EntropyChart(
            first_distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]),
            probability_labels = [
                Tex(R"\frac{1}{" + ["2", "4", "8", "8"][i] + "}", font_size = 65)
                for i in range(4)
            ],
            bar_labels = [
                Tex(encoding[i], font_size = 57)
                for i in range(4)
            ],
            bar_heights = [1, 2, 3, 3],
            width = 12,
            height = 4.5,
            include_vertical_axis = False,
            segments_height = 1,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).scale(0.3).to_corner(UL, buff = 0.7).shift(DOWN*0.7)
        first_distribution_chart.update()
        first_distribution_chart.clear_updaters()
        first_distribution_chart.bar_labels.set_color(WHITE)
        self.play(first_distribution_chart.create(), run_time = 2)
        self.wait(2)
        self.play(first_distribution_chart.animate.set_x(-FRAME_WIDTH*0.25), run_time = 2.5)
        self.wait(2)

        # Build the segments for the second chart
        second_distribution = [1/8, 1/8, 1/4, 1/2]
        second_distribution_chart = EntropyChart(
            second_distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]),
            probability_labels = [
                Tex(R"\frac{1}{" + ["8", "8", "4", "2"][i] + "}", font_size = 65)
                for i in range(4)
            ],
            bar_labels = [
                Tex(encoding[i], font_size = 57)
                for i in range(4)
            ],
            bar_heights = [1, 2, 3, 3],
            width = 12,
            height = 4.5,
            include_vertical_axis = False,
            segments_height = 1,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).scale(0.3).to_edge(UP, buff = 1.4).set_x(FRAME_WIDTH*0.25)
        second_distribution_chart.update()
        second_distribution_chart.clear_updaters()
        second_distribution_chart.bar_labels.set_color(WHITE)
        for segment, e_label, p_label in list(zip(
            second_distribution_chart.segments.bars,
            second_distribution_chart.event_labels,
            second_distribution_chart.probability_labels
        )):
            self.play(
                AnimationGroup(
                    GrowFromCenter(segment),
                    FadeIn(e_label),
                    FadeIn(p_label),
                    suspend_mobject_updating = True
                , run_time = 2)
            )
            self.wait(2)
        self.wait(4)

        # The bars hop over from the old distribution to the new one
        self.play(
            AnimationGroup(*[
                TransformFromCopy(VGroup(bar1, label1), VGroup(bar2, label2), run_time = 3)
                for bar1, label1, bar2, label2 in list(zip(
                    first_distribution_chart.bars,
                    first_distribution_chart.bar_labels,
                    second_distribution_chart.bars,
                    second_distribution_chart.bar_labels
                ))[::-1]
            ], lag_ratio = 0.3)
        )
        self.wait(2)

        # Center everything
        self.play(
            VGroup(first_distribution_chart, second_distribution_chart).animate.scale(1.3).arrange(buff = 2).shift(DOWN*0.5)
        , run_time = 2.5)

        # Calculate the cross entropy
        weighted_sum_lines = VGroup(
            Tex(R"\frac{1}{8} \cdot 1", font_size = 23).next_to(second_distribution_chart.bars[0], UP),
            Tex(R"\frac{1}{8} \cdot 2", font_size = 23).next_to(second_distribution_chart.bars[1], UP),
            Tex(R"\frac{1}{4} \cdot 3", font_size = 23).next_to(second_distribution_chart.bars[2], UP),
            Tex(R"\frac{1}{2} \cdot 3", font_size = 23).next_to(second_distribution_chart.bars[3], UP)
        )
        for line in weighted_sum_lines:
            line[:3].set_color(GREEN)
            line[4:].set_color(PINK)

        self.play(
            AnimationGroup(
                TransformFromCopy(second_distribution_chart.probability_labels[0], weighted_sum_lines[0][:-2]),
                FadeIn(weighted_sum_lines[0][-2:], shift = UP*0.1)
            , run_time = 1.5)
        )
        self.wait(1.5)
        self.play(
            AnimationGroup(
                TransformFromCopy(second_distribution_chart.probability_labels[1], weighted_sum_lines[1][:-2]),
                FadeIn(weighted_sum_lines[1][-2:], shift = UP*0.1)
            , run_time = 1.5)
        )
        self.wait(2.5)
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    TransformFromCopy(label, line[:-2]),
                    FadeIn(line[-2:], shift = UP*0.1)
                , run_time = 1.5)
                for label, line in zip(second_distribution_chart.probability_labels[2:], weighted_sum_lines[2:])
            ], lag_ratio = 0.3)
        )

        self.wait(1)
        sum_result = Tex(
            R"\frac{1}{8} \cdot 1 + \frac{1}{8} \cdot 2 + \frac{1}{4} \cdot 3 + \frac{1}{2} \cdot 3 \\ = 2.625 \text{ bits}",
            font_size = 32,
            tex_to_color_map = {
                R"\frac{1}{8}": GREEN,
                R"\frac{1}{4}": GREEN,
                R"\frac{1}{2}": GREEN,
                " 1 ": PINK,
                " 2 ": PINK,
                " 3 ": PINK
            }
        ).next_to(second_distribution_chart, UP, buff = 0.3)

        self.play(TransformMatchingShapes(weighted_sum_lines, sum_result[:-10], path_arc = PI*0.2, run_time = 1.5))
        self.wait(0.5)
        self.play(FadeIn(sum_result[R"= 2.625 \text{ bits}"]))
        self.wait(2)

        # Write "cross entropy"
        cross_entropy_text = TexText("Cross Entropy:").set_fill(color = [PINK, GREEN]).next_to(sum_result, UP)
        for i, letter in enumerate(cross_entropy_text):
            letter.set_color(interpolate_color(PINK, GREEN, i/(len(cross_entropy_text) - 1)))
        self.play(Write(cross_entropy_text, run_time = 2.5))
        self.wait(0.5)
        rect1 = SurroundingRectangle(
            first_distribution_chart.probability_labels, stroke_width = 2, stroke_color = PINK
        ).stretch_to_fit_width(first_distribution_chart.bars.get_width()).match_x(first_distribution_chart.bars)
        self.play(FadeIn(rect1), run_time = 1.5)
        self.wait(3)
        rect2 = SurroundingRectangle(
            second_distribution_chart.probability_labels, stroke_width = 2, stroke_color = GREEN
        ).stretch_to_fit_width(second_distribution_chart.bars.get_width()).match_x(second_distribution_chart.bars)
        self.play(ReplacementTransform(rect1, rect2), run_time = 2.5)
        self.wait(2)
        self.play(FadeOut(rect2))
        self.wait(2)

        # Label the two charts with p and q
        p_chart = EntropyChart(
            second_distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]).scale(0.6) for i in range(4)]),
            probability_labels = VGroup(*[Tex(f"p_{i + 1}", font_size = 42) for i in range(4)]),
            bar_labels = [
                Tex(encoding[i], font_size = 27)
                for i in range(4)
            ],
            bar_heights = [1, 2, 3, 3],
            width = second_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = False,
            segments_height = 0.4,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        )
        p_chart.suspend_updating()
        p_chart.bar_labels.set_color(WHITE)

        q_chart = EntropyChart(
            first_distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]).scale(0.6) for i in range(4)]),
            probability_labels = VGroup(*[Tex(f"q_{i + 1}", font_size = 42) for i in range(4)]),
            bar_labels = [
                Tex(encoding[i], font_size = 27)
                for i in range(4)
            ],
            width = first_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = False,
            segments_height = 0.4,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        )
        q_chart.suspend_updating()
        q_chart.bar_labels.set_color(WHITE)

        p_chart.match_x(second_distribution_chart).align_to(second_distribution_chart.bars, UP)
        q_chart.match_x(first_distribution_chart).align_to(first_distribution_chart.bars, UP)
        self.play(
            FadeOut(VGroup(cross_entropy_text, sum_result)),
            ReplacementTransform(second_distribution_chart, p_chart, suspend_mobject_updating = True)
        , run_time = 2)
        self.wait(2)
        self.play(ReplacementTransform(first_distribution_chart, q_chart, suspend_mobject_updating = True), run_time = 2)
        self.wait(3)

        # Generalize the charts
        p_chart_general = EntropyChart(
            second_distribution,
            event_labels = VGroup(*[
                Tex(
                    (("s_" + str(i + 1)) if i < len(second_distribution) - 2 else R"\ldots" if i == len(second_distribution) - 2 else "s_n"),
                    font_size = 35
                ).set_color(BLACK)
                for i in range(len(second_distribution))
            ]),
            probability_labels = VGroup(*[
                Tex(
                    (("p_" + str(i + 1)) if i < len(second_distribution) - 2 else R"\ldots" if i == len(second_distribution) - 2 else "p_n"),
                    font_size = 42
                )
                for i in range(len(second_distribution))
            ]),
            bar_labels = VMobject(),
            bar_heights = [1, 2, 3, 3],
            width = second_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = True,
            vertical_axis_label_text = R"\begin{gathered}-\log_2 q_i \\ \ (\text{bits})\end{gathered}",
            vertical_axis_font_size = 30,
            segments_height = 0.4,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        )
        p_chart_general.suspend_updating()

        q_chart_general = EntropyChart(
            first_distribution,
            event_labels = VGroup(*[
                Tex(
                    (("s_" + str(i + 1)) if i < len(first_distribution) - 2 else R"\ldots" if i == len(first_distribution) - 2 else "s_n"),
                    font_size = 35
                ).set_color(BLACK)
                for i in range(len(first_distribution))
            ]),
            probability_labels = VGroup(*[
                Tex(
                    (("q_" + str(i + 1)) if i < len(first_distribution) - 2 else R"\ldots" if i == len(first_distribution) - 2 else "q_n"),
                    font_size = 42
                )
                for i in range(len(first_distribution))
            ]),
            bar_labels = VMobject(),
            width = first_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = True,
            vertical_axis_label_text = R"\begin{gathered}-\log_2 q_i \\ \ (\text{bits})\end{gathered}",
            vertical_axis_font_size = 30,
            segments_height = 0.4,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        )
        q_chart_general.suspend_updating()

        p_chart_general.shift(p_chart.bars.get_center() - p_chart_general.bars.get_center())
        q_chart_general.shift(q_chart.bars.get_center() - q_chart_general.bars.get_center())
        self.play(
            ReplacementTransform(
                VGroup(q_chart.bars, q_chart.segments, q_chart.probability_labels),
                VGroup(q_chart_general.bars, q_chart_general.segments, q_chart_general.probability_labels)
            , suspend_mobject_updating = True),
            ReplacementTransform(
                VGroup(p_chart.bars, p_chart.segments, p_chart.probability_labels),
                VGroup(p_chart_general.bars, p_chart_general.segments, p_chart_general.probability_labels)
            , suspend_mobject_updating = True),
            FadeOut(VGroup(q_chart.bar_labels, p_chart.bar_labels))
        , run_time = 2)
        self.wait(2)

        self.remove(q_chart, p_chart)
        self.add(q_chart_general, p_chart_general)
        VGroup(
            q_chart_general.vertical_axis,
            q_chart_general.vertical_axis_label,
            q_chart_general.reference_lines,
            p_chart_general.vertical_axis,
            p_chart_general.vertical_axis_label,
            p_chart_general.reference_lines
        ).set_opacity(0)

        # Show different possibilities for what the symbols could mean
        alternate_symbols_q_1 = VGroup(*[
            InstructionArrow([UP, DOWN, LEFT, RIGHT][i]).scale(0.08)
            for i in range(4)
        ])
        alternate_symbols_p_1 = alternate_symbols_q_1.copy()
        for i, (q_symb, p_symb) in enumerate(zip(alternate_symbols_q_1, alternate_symbols_p_1)):
            q_symb.move_to(q_chart_general.event_labels[i])
            p_symb.move_to(p_chart_general.event_labels[i])
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    FadeOut(label),
                    FadeIn(alternate_label)
                )
                for label, alternate_label in zip(q_chart_general.event_labels, alternate_symbols_q_1)
            ], lag_ratio = 0.2),
            AnimationGroup(*[
                AnimationGroup(
                    FadeOut(label),
                    FadeIn(alternate_label)
                )
                for label, alternate_label in zip(p_chart_general.event_labels, alternate_symbols_p_1)
            ], lag_ratio = 0.2)
        )
        self.wait(0.8)

        alternate_symbols_q_2 = VGroup(*[
            TexText(["a", "b", "c", "d"][i], font_size = 35).set_color(BLACK)
            for i in range(4)
        ])
        alternate_symbols_p_2 = alternate_symbols_q_2.copy()
        for i, (q_symb, p_symb) in enumerate(zip(alternate_symbols_q_2, alternate_symbols_p_2)):
            q_symb.next_to(q_chart_general.probability_labels[0], UP, buff = 0.345).match_x(q_chart_general.probability_labels[i])
            p_symb.next_to(p_chart_general.probability_labels[0], UP, buff = 0.345).match_x(p_chart_general.probability_labels[i])
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    FadeOut(label),
                    FadeIn(alternate_label)
                )
                for label, alternate_label in zip(alternate_symbols_q_1, alternate_symbols_q_2)
            ], lag_ratio = 0.2),
            AnimationGroup(*[
                AnimationGroup(
                    FadeOut(label),
                    FadeIn(alternate_label)
                )
                for label, alternate_label in zip(alternate_symbols_p_1, alternate_symbols_p_2)
            ], lag_ratio = 0.2)
        )
        self.wait(0.8)

        alternate_symbols_q_3 = VGroup(*[
            Tex(["e", R"\wedge", R"\pi", "i"][i], font_size = 40 if i != 1 else 20).set_color(BLACK)
            for i in range(4)
        ])
        alternate_symbols_p_3 = alternate_symbols_q_3.copy()
        for i, (q_symb, p_symb) in enumerate(zip(alternate_symbols_q_3, alternate_symbols_p_3)):
            q_symb.next_to(q_chart_general.probability_labels[0], UP, buff = 0.33 if i != 1 else 0.455).match_x(q_chart_general.probability_labels[i])
            p_symb.next_to(p_chart_general.probability_labels[0], UP, buff = 0.33 if i != 1 else 0.455).match_x(p_chart_general.probability_labels[i])
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    FadeOut(label),
                    FadeIn(alternate_label)
                )
                for label, alternate_label in zip(alternate_symbols_q_2, alternate_symbols_q_3)
            ], lag_ratio = 0.2),
            AnimationGroup(*[
                AnimationGroup(
                    FadeOut(label),
                    FadeIn(alternate_label)
                )
                for label, alternate_label in zip(alternate_symbols_p_2, alternate_symbols_p_3)
            ], lag_ratio = 0.2)
        )
        self.wait(0.8)

        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    FadeOut(alternate_label),
                    FadeIn(label)
                )
                for alternate_label, label in zip(alternate_symbols_q_3, q_chart_general.event_labels)
            ], lag_ratio = 0.2),
            AnimationGroup(*[
                AnimationGroup(
                    FadeOut(alternate_label),
                    FadeIn(label)
                )
                for alternate_label, label in zip(alternate_symbols_p_3, p_chart_general.event_labels)
            ], lag_ratio = 0.2)
        )
        self.wait(0.8)


        # Show bar heights for q chart
        q_chart = q_chart_general
        p_chart = p_chart_general
        self.play(FancyCircumscribe(VGroup(q_chart.bars, q_chart.probability_labels)))

        p_chart.save_state()
        self.play(
            self.camera.frame.animate.match_x(VGroup(q_chart, p_chart)).shift(UP*0.7),
            AnimationGroup(
                p_chart.animate.fade(0.9),
                AnimationGroup(
                    Write(q_chart.vertical_axis_label.set_opacity(1), run_time = 2),
                    ShowCreation(q_chart.vertical_axis.set_opacity(1)),
                    AnimationGroup(*[
                        ShowCreation(line.set_opacity(1))
                        for line in q_chart.reference_lines
                    ], lag_ratio = 0.1)
                )
            , lag_ratio = 0.3)
        , run_time = 2)
        self.wait(2)
        Group(*self.mobjects).shift(-self.camera.frame.get_center())
        self.camera.frame.center()

        # Show the avg. bits per instruction for Q
        q_bits_per_instruction = TexText(
            R"Avg. bits per instruction: \\[0.1in] $\displaystyle\sum_i q_i (-\log_2 q_i)$",
            font_size = 35,
            tex_to_color_map = {
                "q_i": PINK
            }
        ).next_to(q_chart.bars, UP, buff = 1)
        self.play(Write(q_bits_per_instruction), run_time = 3)
        self.wait(1.5)

        q_bits_per_instruction.save_state()
        q_chart.save_state()
        for i in range(4):
            anims = [
                AnimationGroup(
                    VGroup(
                        q_chart.segments.bars[i + 1:],
                        q_chart.event_labels[i + 1:],
                        q_chart.probability_labels[i + 1:],
                        q_chart.segments.bars[:i],
                        q_chart.event_labels[:i],
                        q_chart.probability_labels[:i]
                    ).animate.set_opacity(0.2),
                    VGroup(
                        q_chart.segments.bars[i],
                        q_chart.event_labels[i],
                        q_chart.probability_labels[i]
                    ).animate.set_opacity(1),
                )
            ]
            if i == 0:
                anims.append(
                    AnimationGroup(
                        q_bits_per_instruction.animate.fade(0.8),
                        q_bits_per_instruction["q_i"][0].animate.set_opacity(1),
                        q_chart.bars.animate.fade(0.8)
                    )
                )
            self.play(AnimationGroup(*anims))

        for i in range(4):
            anims = [
                AnimationGroup(
                    VGroup(
                        q_chart.bars[i + 1:],
                        q_chart.bars[:i],
                        q_chart.bar_labels[:i],
                    ).animate.set_opacity(0.2),
                    q_chart.bars[i].animate.set_opacity(1),
                )
            ]
            if i == 0:
                anims.append(
                    AnimationGroup(
                        q_bits_per_instruction[-8:-1].animate.set_opacity(1),
                        q_bits_per_instruction["q_i"][0].animate.fade(0.8),
                        VGroup(
                            q_chart.segments.bars[3],
                            q_chart.event_labels[3],
                            q_chart.probability_labels[3]
                        ).animate.fade(0.8)
                    )
                )
            self.play(AnimationGroup(*anims))
        self.play(q_chart.animate.restore(), q_bits_per_instruction.animate.restore(), run_time = 2)
        self.wait(2)

        # Label the equation with "entropy"
        entropy_text = TexText(
            "Entropy of Q",
            tex_to_color_map = {"Q": PINK}
        ).match_height(
            q_bits_per_instruction["Avg. bits per instruction:"]
        ).move_to(
            q_bits_per_instruction["Avg. bits per instruction:"]
        )
        self.play(FadeOut(q_bits_per_instruction["Avg. bits per instruction:"]), FadeIn(entropy_text))
        self.wait(2)

        # Highlight the area of the diagram
        q_entropy_formula = q_bits_per_instruction[len("Avg.bitsperinstruction:"):]

        self.play(AnimationGroup(*[Indicate(bar, scale_factor = 1.1) for bar in q_chart.bars], lag_ratio = 0.1, run_time = 2.5))

        # Show why the area represents the expression
        entropy_text.save_state()
        q_entropy_formula.save_state()
        q_chart.save_state()
        for i in range(4):
            anims = [
                AnimationGroup(
                    VGroup(
                        q_chart.segments.bars[i + 1:],
                        q_chart.event_labels[i + 1:],
                        q_chart.probability_labels[i + 1:],
                        q_chart.segments.bars[:i],
                        q_chart.event_labels[:i],
                        q_chart.probability_labels[:i]
                    ).animate.set_opacity(0.2),
                    VGroup(
                        q_chart.segments.bars[i],
                        q_chart.event_labels[i],
                        q_chart.probability_labels[i]
                    ).animate.set_opacity(1),
                )
            ]
            if i == 0:
                anims.append(
                    AnimationGroup(
                        entropy_text.animate.fade(0.8),
                        q_entropy_formula.animate.fade(0.8),
                        q_bits_per_instruction["q_i"][0].animate.set_opacity(1),
                        q_chart.bars.animate.fade(0.8)
                    )
                )
            self.play(AnimationGroup(*anims), run_time = 0.62)

        for i in range(4):
            anims = [
                AnimationGroup(
                    VGroup(
                        q_chart.bars[i + 1:],
                        q_chart.bars[:i],
                        q_chart.bar_labels[:i],
                    ).animate.set_opacity(0.2),
                    q_chart.bars[i].animate.set_opacity(1),
                )
            ]
            if i == 0:
                anims.append(
                    AnimationGroup(
                        q_bits_per_instruction[-8:-1].animate.set_opacity(1),
                        q_bits_per_instruction["q_i"][0].animate.fade(0.8),
                        VGroup(
                            q_chart.segments.bars[3],
                            q_chart.event_labels[3],
                            q_chart.probability_labels[3]
                        ).animate.fade(0.8)
                    )
                )
            self.play(AnimationGroup(*anims), run_time = 0.62)
        self.play(q_chart.animate.restore(), q_entropy_formula.animate.restore(), entropy_text.animate.restore(), run_time = 1.2)
        self.bring_to_back(q_chart.reference_lines, p_chart.reference_lines)

        # Switch focus to the p chart
        self.play(VGroup(p_chart.segments, p_chart.probability_labels).animate.set_opacity(1))

        # Show the encoding and the original distribution
        self.play(
            AnimationGroup(
                AnimationGroup(
                    Write(p_chart.vertical_axis_label.set_opacity(1), run_time = 2),
                    ShowCreation(p_chart.vertical_axis.set_opacity(1)),
                    AnimationGroup(*[
                        ShowCreation(line.set_opacity(1))
                        for line in p_chart.reference_lines
                    ], lag_ratio = 0.1)
                ),
                AnimationGroup(*[
                    TransformFromCopy(bar1, bar2, run_time = 3)
                    for bar1, bar2 in list(zip(
                        q_chart.bars.copy().set_opacity(0.8),
                        p_chart.bars.copy().set_opacity(0.8),
                    ))[::-1]
                ], lag_ratio = 0.3)
            , lag_ratio = 0.7)
        )
        self.clear()
        p_chart.bars.set_opacity(0.8)
        self.add(q_chart, p_chart, entropy_text, q_entropy_formula)
        self.bring_to_back(q_chart.reference_lines, p_chart.reference_lines)

        # Write cross entropy equation
        general_equation = TexText(
            R"Avg. bits per instruction: \\[0.1in] $\displaystyle\sum_i p_i (-\log_2 q_i)$",
            font_size = 35,
            tex_to_color_map = {
                "p_i": GREEN,
                "q_i": PINK
            }
        ).next_to(p_chart.bars, UP, buff = 1)
        self.play(Write(general_equation), run_time = 3)
        self.wait(2)
        self.play(Flash(general_equation["p_i"], flash_radius = 0.5))

        # Replace "Avg. bits per instruction" with "Cross Entropy(Q, P)"
        self.play(FancyCircumscribe(general_equation))
        cross_entropy_text = TexText(
            "Cross Entropy of Q relative to P",
            tex_to_color_map = {"Q": PINK, "P": GREEN}
        ).match_height(
            general_equation["Avg. bits per instruction:"]
        ).move_to(
            general_equation["Avg. bits per instruction:"]
        )
        self.play(FadeOut(general_equation["Avg. bits per instruction:"]), FadeIn(cross_entropy_text), run_time = 2)
        self.wait(3)

        # Show special notation
        full_sum = general_equation[len("Avg.bitsperinstruction:"):]
        cross_entropy_group = VGroup(cross_entropy_text, full_sum)
        cross_entropy_group.save_state()
        self.play(
            FadeOut(
                VGroup(
                    q_chart,
                    p_chart,
                    entropy_text,
                    q_bits_per_instruction[len("Avg.bitsperinstruction:"):]
                )
            , run_time = 1.6, shift = DOWN*3),
            cross_entropy_group.animate(run_time = 2, path_arc = PI*0.2).set_y(0).to_edge(LEFT, buff = 2)
        )
        notations = BulletedList(
            R"$H(P, Q)$",
            R"$H(P \parallel Q)$",
            R"$H_Q(P)$",
            R"$\mathbb{E}_P[-\log Q]$",
            R"$\langle -\log Q \rangle_P$",
            tex_to_color_map = {"Q": PINK, "P": GREEN}
        ).to_edge(RIGHT, buff = 2.2)
        brace = Brace(notations, LEFT)
        self.play(
            GrowFromEdge(brace, RIGHT),
            AnimationGroup(*[FadeIn(line, shift = DOWN*0.3) for line in notations], lag_ratio = 0.2)
        , run_time = 3)
        self.wait(1)

        # Put everything back
        self.play(
            FadeOut(VGroup(brace, notations), shift = RIGHT*5, run_time = 1.5),
            FadeIn(
                VGroup(
                    q_chart,
                    p_chart,
                    entropy_text,
                    q_bits_per_instruction[len("Avg.bitsperinstruction:"):]
                )
            , run_time = 2, shift = UP*3),
            cross_entropy_group.animate(run_time = 2, path_arc = -PI*0.2).restore()
        )
        self.bring_to_back(q_chart.reference_lines, p_chart.reference_lines)
        self.wait(1)

        # Highlight the diagram
        self.play(FancyCircumscribe(p_chart, run_time=5))

        # Show the widths and heights of each bar in the cross entropy diagram
        self.play(FancyCircumscribe(p_chart.probability_labels), FancyCircumscribe(general_equation["p_i"]))
        self.play(FancyCircumscribe(p_chart.vertical_axis_label, run_time = 5), FancyCircumscribe(general_equation[-8:-1], run_time = 5))
        self.wait(3)

        # Create all four charts
        q_entropy_chart = q_chart
        qp_cross_entropy_chart = p_chart
        p_entropy_chart = EntropyChart(
            second_distribution,
            event_labels = VGroup(*[
                Tex(
                    (("s_" + str(i + 1)) if i < len(second_distribution) - 2 else R"\ldots" if i == len(second_distribution) - 2 else "s_n"),
                    font_size = 35
                ).set_color(BLACK)
                for i in range(len(second_distribution))
            ]),
            probability_labels = VGroup(*[
                Tex(
                    (("p_" + str(i + 1)) if i < len(second_distribution) - 2 else R"\ldots" if i == len(second_distribution) - 2 else "p_n"),
                    font_size = 42
                )
                for i in range(len(second_distribution))
            ]),
            bar_labels = None,
            width = second_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = True,
            vertical_axis_label_text = R"\begin{gathered}-\log_2 p_i \\ \ (\text{bits})\end{gathered}",
            vertical_axis_font_size = 30,
            segments_height = 0.4,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [GREEN_B, GREEN_D]
        ).move_to(qp_cross_entropy_chart)

        pq_cross_entropy_chart = EntropyChart(
            first_distribution,
            event_labels = VGroup(*[
                Tex(
                    (("s_" + str(i + 1)) if i < len(first_distribution) - 2 else R"\ldots" if i == len(first_distribution) - 2 else "s_n"),
                    font_size = 35
                ).set_color(BLACK)
                for i in range(len(first_distribution))
            ]),
            probability_labels = VGroup(*[
                Tex(
                    (("q_" + str(i + 1)) if i < len(first_distribution) - 2 else R"\ldots" if i == len(first_distribution) - 2 else "q_n"),
                    font_size = 42
                )
                for i in range(len(first_distribution))
            ]),
            bar_labels = None,
            bar_heights = [1, 2, 3, 3],
            width = first_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = True,
            vertical_axis_label_text = R"\begin{gathered}-\log_2 p_i \\ \ (\text{bits})\end{gathered}",
            vertical_axis_font_size = 30,
            segments_height = 0.4,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK],
            bar_fill_colors = [GREEN_B, GREEN_D]
        ).move_to(q_entropy_chart)
        q_entropy_text = entropy_text
        q_entropy_formula = q_entropy_formula
        q_entropy_group = VGroup(q_entropy_text, q_entropy_formula, q_entropy_chart)
        qp_cross_entropy_text = cross_entropy_text
        qp_cross_entropy_formula = full_sum
        qp_cross_entropy_group = VGroup(qp_cross_entropy_text, qp_cross_entropy_formula, qp_cross_entropy_chart)
        p_entropy_text = TexText(
            "Entropy of P",
            tex_to_color_map = {"P": GREEN}
        ).match_height(
            q_entropy_text
        ).move_to(
            qp_cross_entropy_text
        )
        p_entropy_formula = Tex(
            R"\displaystyle\sum_i p_i (-\log_2 p_i)",
            tex_to_color_map = {
                "p_i": GREEN
            }
        ).match_height(q_entropy_formula).move_to(qp_cross_entropy_formula)
        p_entropy_group = VGroup(p_entropy_text, p_entropy_formula, p_entropy_chart)
        pq_cross_entropy_text = TexText(
            "Cross Entropy of P relative to Q",
            tex_to_color_map = {"P": GREEN, "Q": PINK}
        ).match_height(
            q_entropy_text
        ).move_to(
            q_entropy_text
        )
        pq_cross_entropy_formula = Tex(
            R"\displaystyle\sum_i q_i (-\log_2 p_i)",
            tex_to_color_map = {
                "p_i": GREEN,
                "q_i": PINK
            }
        ).match_height(q_entropy_formula).move_to(q_entropy_formula)
        pq_cross_entropy_group = VGroup(pq_cross_entropy_text, pq_cross_entropy_formula, pq_cross_entropy_chart)

        # Show the 2x2 grid of charts
        q_entropy_group.generate_target()
        qp_cross_entropy_group.generate_target()
        VGroup(q_entropy_group.target, qp_cross_entropy_group.target).arrange(buff = 3)
        pq_cross_entropy_group.match_x(q_entropy_group.target)
        p_entropy_group.match_x(qp_cross_entropy_group.target)
        VGroup(
            VGroup(q_entropy_group.target, qp_cross_entropy_group.target),
            VGroup(pq_cross_entropy_group, p_entropy_group)
        ).arrange(DOWN, buff = 1.6)
        VGroup(p_entropy_chart.segments, p_entropy_chart.probability_labels).set_opacity(0)
        self.play(
            self.camera.frame.animate(run_time = 2).scale(1.5),
            AnimationGroup(
                MoveToTarget(q_entropy_group, run_time = 2),
                MoveToTarget(qp_cross_entropy_group, run_time = 2)
            ),
            AnimationGroup(
                TransformFromCopy(
                    VGroup(qp_cross_entropy_chart.segments, qp_cross_entropy_chart.probability_labels),
                    VGroup(p_entropy_chart.segments.copy(), p_entropy_chart.probability_labels.copy()).set_opacity(1)
                , run_time = 2),
                p_entropy_chart.create(),
                Write(VGroup(p_entropy_text, p_entropy_formula))
            , lag_ratio = 0.7)
        )
        self.clear()
        VGroup(p_entropy_chart.segments, p_entropy_chart.probability_labels).set_opacity(1)
        self.add(q_entropy_group, qp_cross_entropy_group, p_entropy_group)
        self.bring_to_back(q_entropy_chart.reference_lines, qp_cross_entropy_chart.reference_lines)
        # q_entropy_chart.add_updater(lambda m: m.set_stroke(width = 1))
        # qp_cross_entropy_chart.add_updater(lambda m: m.set_stroke(width = 1))
        # pq_cross_entropy_chart.add_updater(lambda m: m.set_stroke(width = 1))
        # p_entropy_chart.add_updater(lambda m: m.set_stroke(width = 1))

        # q_entropy_chart.resume_updating()
        # qp_cross_entropy_chart.resume_updating()
        # q_entropy_chart.update()
        # qp_cross_entropy_chart.update()

        # pq_cross_entropy_chart.update()
        # p_entropy_chart.update()
        self.wait(2)

        # Change the distribution p
        n = len(qp_cross_entropy_chart.distribution_trackers)
        dashed_lines_1 = VGroup(*[
            DashedLine(stroke_width = 2)
            for _ in range(n + 1)
        ])
        line_opacity_tracker_1 = ValueTracker(0)
        def update_dashed_lines(m):
            for i, line in enumerate(m):
                line.put_start_and_end_on(
                    p_entropy_chart.segments.bars[i].get_corner(DL) if i < n else p_entropy_chart.segments.bars[-1].get_corner(DR),
                    qp_cross_entropy_chart.segments.bars[i].get_corner(DL) if i < n else qp_cross_entropy_chart.segments.bars[-1].get_corner(DR)
                ).set_opacity(line_opacity_tracker_1.get_value())
        self.add(dashed_lines_1)
        self.bring_to_back(dashed_lines_1)
        dashed_lines_1.add_updater(update_dashed_lines)
        qp_cross_entropy_chart.resume_updating()
        for i in range(4):
            new_distribution = random_distribution(4, thresh = 1/8)
            anims = [
                AnimationGroup(
                    p_entropy_chart.set_distribution(new_distribution),
                    qp_cross_entropy_chart.set_distribution(new_distribution)
                , run_time = 2)
            ]
            if i == 0:
                anims.append(line_opacity_tracker_1.animate(run_time = 1.2).set_value(0.4))
            self.play(*anims)
        self.wait(2)

        # Change the distribution q
        dashed_lines_2 = VGroup(*[
            DashedLine(stroke_width = 2)
            for _ in range(n + 1)
        ])
        line_opacity_tracker_2 = ValueTracker(0)
        def update_dashed_lines(m):
            for i, line in enumerate(m):
                line.put_start_and_end_on(
                    q_entropy_chart.bars[i - 1].get_corner(UL) if i > 0 else q_entropy_chart.bars[0].get_corner(DL),
                    qp_cross_entropy_chart.bars[i - 1].get_corner(UL) if i > 0 else qp_cross_entropy_chart.bars[0].get_corner(DL)
                ).set_opacity(line_opacity_tracker_2.get_value())
        self.add(dashed_lines_2)
        self.bring_to_back(dashed_lines_2)
        dashed_lines_2.add_updater(update_dashed_lines)
        q_entropy_chart.resume_updating()
        for i in range(5):
            new_distribution = random_distribution(4, thresh = 1/8)
            anims = [
                AnimationGroup(
                    q_entropy_chart.set_distribution(new_distribution),
                    AnimationGroup(*[
                        h.animate.set_value(-math.log2(q))
                        for h, q in zip(qp_cross_entropy_chart.bar_heights, new_distribution)
                    ])
                , run_time = 2)
            ]
            if i == 0:
                anims.append(
                    AnimationGroup(
                        line_opacity_tracker_1.animate(run_time = 1).set_value(0),
                        line_opacity_tracker_2.animate(run_time = 1.2).set_value(0.4)
                    )
                )
            self.play(*anims)
        self.wait(2)

        # Add the pq cross entropy diagram
        q_entropy_chart.suspend_updating()
        for t1, t2 in zip(pq_cross_entropy_chart.distribution_trackers, q_entropy_chart.distribution_trackers):
            t1.set_value(t2.get_value())
        for h, t in zip(pq_cross_entropy_chart.bar_heights, p_entropy_chart.distribution_trackers):
            h.set_value(-math.log2(t.get_value()))
        pq_cross_entropy_chart.update()

        VGroup(pq_cross_entropy_chart.segments, pq_cross_entropy_chart.probability_labels, pq_cross_entropy_chart.bars).set_opacity(0)
        self.play(
            AnimationGroup(
                TransformFromCopy(
                    VGroup(q_entropy_chart.segments, q_entropy_chart.probability_labels),
                    VGroup(pq_cross_entropy_chart.segments.copy(), pq_cross_entropy_chart.probability_labels.copy()).set_opacity(1)
                , run_time = 2),
                AnimationGroup(*[
                    TransformFromCopy(bar1, bar2.copy().set_opacity(1))
                    for bar1, bar2 in zip(p_entropy_chart.bars, pq_cross_entropy_chart.bars)
                ], lag_ratio = 0.1, run_time = 2),
                pq_cross_entropy_chart.create(),
                Write(VGroup(pq_cross_entropy_text, pq_cross_entropy_formula))
            , lag_ratio = 0.7)
        )
        self.wait(2)
        self.clear()
        q_entropy_chart.resume_updating()
        VGroup(pq_cross_entropy_chart.segments, pq_cross_entropy_chart.probability_labels).set_opacity(1)
        pq_cross_entropy_chart.bars.set_opacity(0.8)
        self.add(q_entropy_group, qp_cross_entropy_group, p_entropy_group, pq_cross_entropy_group, dashed_lines_1, dashed_lines_2)
        self.bring_to_back(q_entropy_chart.reference_lines, qp_cross_entropy_chart.reference_lines)

        # Add the last two sets of dashed lines
        dashed_lines_3 = VGroup(*[
            DashedLine(stroke_width = 2)
            for _ in range(n + 1)
        ])
        line_opacity_tracker_3 = ValueTracker(0)
        def update_dashed_lines(m):
            for i, line in enumerate(m):
                line.put_start_and_end_on(
                    pq_cross_entropy_chart.segments.bars[i].get_corner(DL) if i < n else pq_cross_entropy_chart.segments.bars[-1].get_corner(DR),
                    q_entropy_chart.segments.bars[i].get_corner(DL) if i < n else q_entropy_chart.segments.bars[-1].get_corner(DR)
                ).set_opacity(line_opacity_tracker_3.get_value())
        self.add(dashed_lines_3)
        self.bring_to_back(dashed_lines_3)
        dashed_lines_3.add_updater(update_dashed_lines)

        dashed_lines_4 = VGroup(*[
            DashedLine(stroke_width = 2)
            for _ in range(n + 1)
        ])
        line_opacity_tracker_4 = ValueTracker(0)
        def update_dashed_lines(m):
            for i, line in enumerate(m):
                line.put_start_and_end_on(
                    pq_cross_entropy_chart.bars[i - 1].get_corner(UL) if i > 0 else pq_cross_entropy_chart.bars[0].get_corner(DL),
                    p_entropy_chart.bars[i - 1].get_corner(UL) if i > 0 else p_entropy_chart.bars[0].get_corner(DL)
                ).set_opacity(line_opacity_tracker_4.get_value())
        self.add(dashed_lines_4)
        self.bring_to_back(dashed_lines_4)
        dashed_lines_4.add_updater(update_dashed_lines)

        # Change both distributions
        for i in range(5):
            new_distribution_1 = random_distribution(4, thresh = 1/8) if i % 2 == 1 else [t.get_value() for t in q_entropy_chart.distribution_trackers]
            new_distribution_2 = random_distribution(4, thresh = 1/8) if i % 2 == 0 else [t.get_value() for t in p_entropy_chart.distribution_trackers]
            anims = [
                AnimationGroup(
                    q_entropy_chart.set_distribution(new_distribution_1),
                    qp_cross_entropy_chart.set_distribution(new_distribution_2),
                    p_entropy_chart.set_distribution(new_distribution_2),
                    pq_cross_entropy_chart.set_distribution(new_distribution_1),
                    AnimationGroup(*[
                        h.animate.set_value(-math.log2(q))
                        for h, q in zip(qp_cross_entropy_chart.bar_heights, new_distribution_1)
                    ]),
                    AnimationGroup(*[
                        h.animate.set_value(-math.log2(p))
                        for h, p in zip(pq_cross_entropy_chart.bar_heights, new_distribution_2)
                    ])
                , run_time = 2)
            ]
            if i == 0:
                anims.append(
                    AnimationGroup(
                        line_opacity_tracker_1.animate(run_time = 1.2).set_value(0.4),
                        line_opacity_tracker_3.animate(run_time = 1.2).set_value(0.4),
                        line_opacity_tracker_4.animate(run_time = 1.2).set_value(0.4)
                    )
                )
            if i == 2:
                anims.append(
                    AnimationGroup(*[
                        t.animate.set_value(0)
                        for t in [line_opacity_tracker_1, line_opacity_tracker_2, line_opacity_tracker_3, line_opacity_tracker_4]
                    ])
                )
            # if i == 3:
            #     rect1 = SurroundingRectangle(qp_cross_entropy_formula, stroke_width = 2)
            #     rect2 = SurroundingRectangle(pq_cross_entropy_formula, stroke_width = 2)
            #     anims.append(FadeIn(VGroup(rect1, rect2), run_time = 1.5))
            self.play(*anims)
            if i == 2:
                self.remove(dashed_lines_1, dashed_lines_2, dashed_lines_3, dashed_lines_4)

        # Simplify to distributions over two events
        distribution_1 = [0.5, 0.5]
        distribution_2 = [0.9, 0.1]
        q_entropy_chart_two_events = EntropyChart(
            distribution_1,
            event_labels = VGroup(*[
                Tex("s_" + str(i + 1), font_size = 35).set_color(BLACK)
                for i in range(2)
            ]),
            probability_labels = VGroup(*[
                Tex(str(distribution_1[i]), font_size = 42)
                for i in range(2)
            ]),
            bar_labels = None,
            width = first_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = True,
            vertical_axis_label_text = R"\begin{gathered}-\log_2 q_i \\ \ (\text{bits})\end{gathered}",
            vertical_axis_font_size = 30,
            segments_height = 0.4,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).move_to(q_entropy_chart)
        q_entropy_chart_two_events.suspend_updating()

        qp_cross_entropy_chart_two_events = EntropyChart(
            distribution_2,
            event_labels = VGroup(*[
                Tex("s_" + str(i + 1), font_size = 35).set_color(BLACK)
                for i in range(2)
            ]),
            probability_labels = VGroup(*[
                Tex(str(distribution_2[i]), font_size = 42)
                for i in range(2)
            ]),
            bar_labels = None,
            bar_heights = [-math.log2(q) for q in distribution_1],
            width = second_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = True,
            vertical_axis_label_text = R"\begin{gathered}-\log_2 q_i \\ \ (\text{bits})\end{gathered}",
            vertical_axis_font_size = 30,
            segments_height = 0.4,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).move_to(qp_cross_entropy_chart)
        qp_cross_entropy_chart_two_events.suspend_updating()

        p_entropy_chart_two_events = EntropyChart(
            distribution_2,
            event_labels = VGroup(*[
                Tex("s_" + str(i + 1), font_size = 35).set_color(BLACK)
                for i in range(2)
            ]),
            probability_labels = VGroup(*[
                Tex(str(distribution_2[i]), font_size = 42)
                for i in range(2)
            ]),
            bar_labels = None,
            width = second_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = True,
            vertical_axis_label_text = R"\begin{gathered}-\log_2 p_i \\ \ (\text{bits})\end{gathered}",
            vertical_axis_font_size = 30,
            segments_height = 0.4,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [GREEN_B, GREEN_D]
        ).move_to(p_entropy_chart)
        p_entropy_chart_two_events.suspend_updating()

        pq_cross_entropy_chart_two_events = EntropyChart(
            distribution_1,
            event_labels = VGroup(*[
                Tex("s_" + str(i + 1), font_size = 35).set_color(BLACK)
                for i in range(2)
            ]),
            probability_labels = VGroup(*[
                Tex(str(distribution_1[i]), font_size = 42)
                for i in range(2)
            ]),
            bar_labels = None,
            bar_heights = [-math.log2(p) for p in distribution_2],
            width = first_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = True,
            vertical_axis_label_text = R"\begin{gathered}-\log_2 p_i \\ \ (\text{bits})\end{gathered}",
            vertical_axis_font_size = 30,
            segments_height = 0.4,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK],
            bar_fill_colors = [GREEN_B, GREEN_D]
        ).move_to(pq_cross_entropy_chart)
        pq_cross_entropy_chart_two_events.suspend_updating()
        VGroup(
            q_entropy_chart_two_events,
            qp_cross_entropy_chart_two_events,
            p_entropy_chart_two_events,
            pq_cross_entropy_chart_two_events
        ).set_stroke(width = 0.5)

        # Replace the charts
        self.play(
            FadeOut(VGroup(q_entropy_chart, qp_cross_entropy_chart, p_entropy_chart, pq_cross_entropy_chart), suspend_mobject_updating = True),
            FadeIn(VGroup(q_entropy_chart_two_events, qp_cross_entropy_chart_two_events, p_entropy_chart_two_events, pq_cross_entropy_chart_two_events))
        )
        self.wait(2)
        self.play(
            FancyCircumscribe(q_entropy_chart_two_events.probability_labels, run_time = 5),
            FancyCircumscribe(pq_cross_entropy_chart_two_events.probability_labels, run_time = 5)
        )
        self.wait(1)
        self.play(
            FancyCircumscribe(qp_cross_entropy_chart_two_events.probability_labels, run_time = 5),
            FancyCircumscribe(p_entropy_chart_two_events.probability_labels, run_time = 5)
        )
        self.wait(2)

        # Explain the entropy of Q
        self.camera.frame.save_state()
        group1 = VGroup(q_entropy_text, q_entropy_formula, q_entropy_chart_two_events)
        group2 = VGroup(qp_cross_entropy_text, qp_cross_entropy_formula, qp_cross_entropy_chart_two_events)
        group3 = VGroup(p_entropy_text, p_entropy_formula, p_entropy_chart_two_events)
        group4 = VGroup(pq_cross_entropy_text, pq_cross_entropy_formula, pq_cross_entropy_chart_two_events)
        self.play(self.camera.frame.animate.scale(0.5).move_to(group1).shift(RIGHT*0.4), run_time = 2)
        information_1 = Tex(
            R"-\log_2 0.5 = 1", font_size = 30
        ).set_color(PURE_MAGENTA).next_to(q_entropy_chart_two_events.bars[0], UP, buff = 0.1)
        information_2 = Tex(
            R"-\log_2 0.5 = 1", font_size = 30
        ).set_color(LIGHT_PINK).next_to(q_entropy_chart_two_events.bars[1], UP, buff = 0.1)
        self.play(Write(information_1), Write(information_2))
        self.wait(2)

        # Explain the entropy of P
        v = group3.get_center() - group1.get_center()
        v_perp = [-v[1], v[0], 0]
        self.play(
            self.camera.frame.animate(path_arc = PI*0.8, path_arc_axis = v_perp).move_to(
                group3
            )
        , run_time = 2.5)
        information_3 = Tex(
            R"-\log_2 0.9 \approx 0.15", font_size = 30
        ).set_color(GREEN_B).next_to(p_entropy_chart_two_events.bars[0], UP, buff = 0.1)
        information_4 = Tex(
            R"-\log_2 0.1 \approx 3.32", font_size = 30
        ).set_color(GREEN_D).next_to(p_entropy_chart_two_events.bars[1], UP, buff = 0.1).align_to(p_entropy_chart_two_events.bars[1], RIGHT)
        self.play(Write(information_3))
        self.wait(2)
        self.play(Write(information_4))

        # Prepare for calculation
        self.play(FadeOut(VGroup(information_1, information_2, information_3, information_4)), self.camera.frame.animate.restore(), run_time = 2)
        dividing_line = Line(self.camera.frame.get_left(), self.camera.frame.get_right())
        self.play(FadeOut(VGroup(group1, group3)), ShowCreation(dividing_line))

        # Calculate cross entropy of Q relative to P
        calculation_1 = Tex(
            R"0.9(-\log_2 0.5) \\ +\ 0.1(-\log_2 0.5) \\ \approx " +
            str(round(sum([p*-math.log2(q) for q, p in zip(distribution_1, distribution_2)]), 2)),
            font_size = 80,
            tex_to_color_map = {"0.9": GREEN, "0.1": GREEN, "0.5": PINK}
        ).match_y(VGroup(qp_cross_entropy_text, qp_cross_entropy_chart)).match_x(pq_cross_entropy_chart).shift(RIGHT*2)
        calculation_1[28:].set_color(TEAL)
        self.play(Write(calculation_1), run_time = 3)
        self.wait(0.5)
        self.play(FancyCircumscribe(VGroup(calculation_1["0.9"], calculation_1["0.1"]), run_time = 5))


        # Calculate cross entropy of P relative to Q
        calculation_2 = Tex(
            R"0.5(-\log_2 0.9) \\ +\ 0.5(-\log_2 0.1) \\ \approx " +
            str(round(sum([q*-math.log2(p) for q, p in zip(distribution_1, distribution_2)]), 2)),
            font_size = 80,
            tex_to_color_map = {"0.9": GREEN, "0.1": GREEN, "0.5": PINK}
        ).match_y(VGroup(pq_cross_entropy_text, pq_cross_entropy_chart)).match_x(qp_cross_entropy_chart).shift(LEFT*2)
        calculation_2[28:].set_color(TEAL)
        self.play(Write(calculation_2), run_time = 3)

        # Compare the results
        comparision = Tex(
            R"1.0 \neq 1.74", font_size = 80
        ).set_color(
            TEAL
        ).set_color_by_tex(
            R"\neq", PURE_RED
        )
        background = VGroup(group2, group4, calculation_1, calculation_2)
        background.save_state()
        self.play(
            dividing_line.animate.set_opacity(0.15),
            background.animate.fade(0.8),
            AnimationGroup(
                AnimationGroup(
                    TransformFromCopy(calculation_1["1.0"], comparision["1.0"], path_arc = -PI*0.2),
                    TransformFromCopy(calculation_2["1.74"], comparision["1.74"], path_arc = PI*0.2)
                ),
                FadeIn(comparision[R"\neq"])
            , lag_ratio = 0.8)
        )
        self.wait(1)
        self.play(
            FadeOut(dividing_line, run_time = 1),
            FadeOut(comparision, run_time = 1),
            FadeIn(group1, run_time = 2),
            FadeIn(group3, run_time = 2),
            background.animate(run_time = 2).restore(),
            FadeOut(calculation_1, run_time = 1.5),
            FadeOut(calculation_2, run_time = 1.5)
        )
        self.wait(2)

        # Generalize the two-event distributions
        q_entropy_chart_two_events_general = EntropyChart(
            distribution_1,
            event_labels = VGroup(*[
                Tex("s_" + str(i + 1), font_size = 35).set_color(BLACK)
                for i in range(2)
            ]),
            probability_labels = VGroup(*[
                Tex(F"q_{str(i + 1)}", font_size = 42)
                for i in range(2)
            ]),
            bar_labels = None,
            width = first_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = True,
            vertical_axis_label_text = R"\begin{gathered}-\log_2 q_i \\ \ (\text{bits})\end{gathered}",
            vertical_axis_font_size = 30,
            segments_height = 0.4,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).move_to(q_entropy_chart)

        qp_cross_entropy_chart_two_events_general = EntropyChart(
            distribution_2,
            event_labels = VGroup(*[
                Tex("s_" + str(i + 1), font_size = 35).set_color(BLACK)
                for i in range(2)
            ]),
            probability_labels = VGroup(*[
                Tex(F"p_{str(i + 1)}", font_size = 42)
                for i in range(2)
            ]),
            bar_labels = None,
            bar_heights = [-math.log2(q) for q in distribution_1],
            width = second_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = True,
            vertical_axis_label_text = R"\begin{gathered}-\log_2 q_i \\ \ (\text{bits})\end{gathered}",
            vertical_axis_font_size = 30,
            segments_height = 0.4,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).move_to(qp_cross_entropy_chart)

        p_entropy_chart_two_events_general = EntropyChart(
            distribution_2,
            event_labels = VGroup(*[
                Tex("s_" + str(i + 1), font_size = 35).set_color(BLACK)
                for i in range(2)
            ]),
            probability_labels = VGroup(*[
                Tex(F"p_{str(i + 1)}", font_size = 42)
                for i in range(2)
            ]),
            bar_labels = None,
            width = second_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = True,
            vertical_axis_label_text = R"\begin{gathered}-\log_2 p_i \\ \ (\text{bits})\end{gathered}",
            vertical_axis_font_size = 30,
            segments_height = 0.4,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [GREEN_B, GREEN_D]
        ).move_to(p_entropy_chart)

        pq_cross_entropy_chart_two_events_general = EntropyChart(
            distribution_1,
            event_labels = VGroup(*[
                Tex("s_" + str(i + 1), font_size = 35).set_color(BLACK)
                for i in range(2)
            ]),
            probability_labels = VGroup(*[
                Tex(F"q_{str(i + 1)}", font_size = 42)
                for i in range(2)
            ]),
            bar_labels = None,
            bar_heights = [-math.log2(p) for p in distribution_2],
            width = first_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = True,
            vertical_axis_label_text = R"\begin{gathered}-\log_2 p_i \\ \ (\text{bits})\end{gathered}",
            vertical_axis_font_size = 30,
            segments_height = 0.4,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK],
            bar_fill_colors = [GREEN_B, GREEN_D]
        ).move_to(pq_cross_entropy_chart)
        q_entropy_chart_two_events_general.add_updater(lambda m: m.set_stroke(width = 0.5))
        qp_cross_entropy_chart_two_events_general.add_updater(lambda m: m.set_stroke(width = 0.5))
        p_entropy_chart_two_events_general.add_updater(lambda m: m.set_stroke(width = 0.5))
        pq_cross_entropy_chart_two_events_general.add_updater(lambda m: m.set_stroke(width = 0.5))

        self.play(
            AnimationGroup(
                FadeIn(
                    VGroup(
                        q_entropy_chart_two_events_general,
                        qp_cross_entropy_chart_two_events_general,
                        p_entropy_chart_two_events_general,
                        pq_cross_entropy_chart_two_events_general
                    )
                , suspend_mobject_updating = True),
                FadeOut(
                    VGroup(
                        q_entropy_chart_two_events,
                        qp_cross_entropy_chart_two_events,
                        p_entropy_chart_two_events,
                        pq_cross_entropy_chart_two_events
                    )
                )
            , lag_ratio = 0.2)
        )
        q_entropy_chart = q_entropy_chart_two_events_general
        qp_cross_entropy_chart = qp_cross_entropy_chart_two_events_general
        p_entropy_chart = p_entropy_chart_two_events_general
        pq_cross_entropy_chart = pq_cross_entropy_chart_two_events_general

        # Play around with the distrbutions
        for i in range(20):
            new_distribution_1 = random_distribution(2, thresh = 1/8) if i % 2 == 1 else [t.get_value() for t in q_entropy_chart.distribution_trackers]
            new_distribution_2 = random_distribution(2, thresh = 1/8) if i % 2 == 0 else [t.get_value() for t in p_entropy_chart.distribution_trackers]
            anims = [
                AnimationGroup(
                    q_entropy_chart.set_distribution(new_distribution_1),
                    qp_cross_entropy_chart.set_distribution(new_distribution_2),
                    p_entropy_chart.set_distribution(new_distribution_2),
                    pq_cross_entropy_chart.set_distribution(new_distribution_1),
                    AnimationGroup(*[
                        h.animate.set_value(-math.log2(q))
                        for h, q in zip(qp_cross_entropy_chart.bar_heights, new_distribution_1)
                    ]),
                    AnimationGroup(*[
                        h.animate.set_value(-math.log2(p))
                        for h, p in zip(pq_cross_entropy_chart.bar_heights, new_distribution_2)
                    ])
                , run_time = 2)
            ]

            if i == 3:
                # Highlight q_1 and add the number line
                circ_opacity_tracker_1 = ValueTracker(0)
                circ1 = always_redraw(
                    lambda: Circle(
                        stroke_width = 2,
                        stroke_color = YELLOW,
                        stroke_opacity = circ_opacity_tracker_1.get_value()
                    ).surround(
                        q_entropy_chart.probability_labels[0],
                        buff = 0.2
                    )
                )
                circ2 = always_redraw(
                    lambda: Circle(
                        stroke_width = 2,
                        stroke_color = YELLOW,
                        stroke_opacity = circ_opacity_tracker_1.get_value()
                    ).surround(
                        pq_cross_entropy_chart.probability_labels[0],
                        buff = 0.2
                    )
                )
                self.add(circ1, circ2)
                anims.append(circ_opacity_tracker_1.animate.set_value(1))

                number_line_q = NumberLine(
                    [0, 1, 0.1],
                    include_numbers = True,
                    numbers_to_exclude = [0.1*i for i in range(1, 10)],
                    width = q_entropy_chart.segments.get_width()
                ).next_to(q_entropy_chart.bars, DOWN, buff = 1.6)
                self.add(number_line_q)
                q1_dot = Group(TrueDot(), GlowDot()).set_color(PURE_MAGENTA)
                q1_dot.add_updater(lambda m: m.move_to(number_line_q.n2p(q_entropy_chart.distribution_trackers[0].get_value())))
                q1_triangle = Triangle(fill_opacity = 0.7, fill_color = PURE_MAGENTA, stroke_width = 0).stretch(1.5, 1).set_width(0.15)
                q1_triangle.add_updater(
                    lambda m: m.next_to(number_line_q.n2p(q_entropy_chart.distribution_trackers[0].get_value()), DOWN, buff = 0)
                )
                q1_label = Tex("q_1", font_size = 42).set_color(PURE_MAGENTA)
                q1_label.add_updater(lambda m: m.next_to(q1_triangle, DOWN, buff = 0.1))
                number_line_q_group = Group(number_line_q, q1_dot, q1_triangle, q1_label)
                number_line_q_opacity_tracker = ValueTracker(0)
                number_line_q_group.add_updater(lambda m: m.set_opacity(number_line_q_opacity_tracker.get_value()))
                self.add(number_line_q_group)
                anims.append(number_line_q_opacity_tracker.animate.set_value(1))
            if i == 4:
                # Highlight q_2
                anims.append(circ_opacity_tracker_1.animate.set_value(0))

                circ_opacity_tracker_2 = ValueTracker(0)
                circ3 = always_redraw(
                    lambda: Circle(
                        stroke_width = 2,
                        stroke_color = YELLOW,
                        stroke_opacity = circ_opacity_tracker_2.get_value()
                    ).surround(
                        q_entropy_chart.probability_labels[1],
                        buff = 0.2
                    )
                )
                circ4 = always_redraw(
                    lambda: Circle(
                        stroke_width = 2,
                        stroke_color = YELLOW,
                        stroke_opacity = circ_opacity_tracker_2.get_value()
                    ).surround(
                        pq_cross_entropy_chart.probability_labels[1],
                        buff = 0.2
                    )
                )
                self.add(circ3, circ4)
                anims.append(circ_opacity_tracker_2.animate.set_value(1))
            if i == 5:
                # Replace q_2 with 1 - q_1
                anims.append(circ_opacity_tracker_2.animate.set_value(0))

                new_q2_label_1 = always_redraw(
                    lambda: Tex(
                        "1 - q_1", font_size = 42
                    ).set_color(
                        LIGHT_PINK
                    ).move_to(
                        q_entropy_chart.probability_labels[1]
                    ).set_opacity(
                        1 - circ_opacity_tracker_2.get_value()
                    )
                )
                self.add(new_q2_label_1)
                q_entropy_chart.probability_labels[1].add_updater(lambda m: m.set_opacity(circ_opacity_tracker_2.get_value()))

                new_q2_label_2 = always_redraw(
                    lambda: Tex(
                        "1 - q_1", font_size = 42
                    ).set_color(
                        LIGHT_PINK
                    ).move_to(
                        pq_cross_entropy_chart.probability_labels[1]
                    ).set_opacity(
                        1 - circ_opacity_tracker_2.get_value()
                    )
                )
                self.add(new_q2_label_2)
                pq_cross_entropy_chart.probability_labels[1].add_updater(lambda m: m.set_opacity(circ_opacity_tracker_2.get_value()))
            if i == 7:
                # Highlight the p distirbution
                rect_opacity_tracker = ValueTracker(0)
                rect1 = always_redraw(
                    lambda: SurroundingRectangle(
                        qp_cross_entropy_chart.probability_labels,
                        buff = 0.1,
                        stroke_width = 1.5,
                        stroke_color = YELLOW,
                        stroke_opacity = rect_opacity_tracker.get_value()
                    )
                )
                rect2 = always_redraw(
                    lambda: SurroundingRectangle(
                        p_entropy_chart.probability_labels,
                        buff = 0.1,
                        stroke_width = 1.5,
                        stroke_color = YELLOW,
                        stroke_opacity = rect_opacity_tracker.get_value()
                    )
                )
                self.add(rect1, rect2)
                anims.append(rect_opacity_tracker.animate.set_value(1))

                number_line_p = NumberLine(
                    [0, 1, 0.1],
                    include_numbers = True,
                    numbers_to_exclude = [0.1*i for i in range(1, 10)],
                    width = qp_cross_entropy_chart.segments.get_width()
                ).next_to(qp_cross_entropy_chart.bars, DOWN, buff = 1.6)
                self.add(number_line_p)
                p1_dot = Group(TrueDot(), GlowDot()).set_color(GREEN_B)
                p1_dot.add_updater(lambda m: m.move_to(number_line_p.n2p(qp_cross_entropy_chart.distribution_trackers[0].get_value())))
                p1_triangle = Triangle(
                    fill_opacity = 0.7, fill_color = GREEN_B, stroke_width = 0
                ).stretch(1.5, 1).set_width(0.15).flip(axis = RIGHT)
                p1_triangle.add_updater(
                    lambda m: m.next_to(number_line_p.n2p(qp_cross_entropy_chart.distribution_trackers[0].get_value()), UP, buff = 0)
                )
                p1_label = Tex("p_1", font_size = 42).set_color(GREEN_B)
                p1_label.add_updater(lambda m: m.next_to(p1_triangle, UP, buff = 0.1))
                number_line_p_group = Group(number_line_p, p1_dot, p1_triangle, p1_label)
                number_line_p_opacity_tracker = ValueTracker(0)
                number_line_p_group.add_updater(lambda m: m.set_opacity(number_line_p_opacity_tracker.get_value()))
                self.add(number_line_p_group)
                anims.append(number_line_p_opacity_tracker.animate.set_value(1))
            if i == 9:
                # Replace the p_2 labels with 1 - p_1
                anims.append(rect_opacity_tracker.animate.set_value(0))

                dummy_p = Tex("p", font_size = 42)
                new_p2_label_1 = always_redraw(
                    lambda: Tex(
                        "1 - p_1", font_size = 42
                    ).set_color(
                        GREEN_D
                    ).move_to(
                        qp_cross_entropy_chart.probability_labels[1]
                    ).set_opacity(
                        1 - rect_opacity_tracker.get_value()
                    ).scale(qp_cross_entropy_chart.probability_labels[0][0].get_height()/dummy_p.get_height())
                )
                self.add(new_p2_label_1)
                qp_cross_entropy_chart.probability_labels[1].add_updater(lambda m: m.set_opacity(rect_opacity_tracker.get_value()))

                new_p2_label_2 = always_redraw(
                    lambda: Tex(
                        "1 - p_1", font_size = 42
                    ).set_color(
                        GREEN_D
                    ).move_to(
                        p_entropy_chart.probability_labels[1]
                    ).set_opacity(
                        1 - rect_opacity_tracker.get_value()
                    )
                )
                self.add(new_p2_label_2)
                p_entropy_chart.probability_labels[1].add_updater(lambda m: m.set_opacity(rect_opacity_tracker.get_value()))
            self.play(*anims)
            if i == 4:
                self.remove(circ1, circ2)
            if i == 5:
                self.remove(circ3, circ4)
            if i == 9:
                self.remove(rect1, rect2)
        self.wait(1)

        # Switch to graphs
        q1_dot.add_updater(lambda m: self.bring_to_front(m))
        p1_dot.add_updater(lambda m: self.bring_to_front(m))
        max_y = 4
        axes = Axes(
            x_range = [0, 1, 0.1],
            y_range = [0, max_y],
            width = number_line_q.get_width(),
            height = 5
        )
        coordinate_labels_x = axes.add_coordinate_labels(
            x_values = [0, 1],
            y_values = [],
            excluding = []
        ),
        coordinate_labels_y = axes.add_coordinate_labels(
            x_values = [],
            y_values = [i for i in range(max_y + 1)],
            excluding = [],
            direction = LEFT
        )
        axes.scale(1.3).align_to(number_line_q, RIGHT).shift(RIGHT*2.3).align_to(self.camera.frame.get_bottom(), DOWN).shift(UP*2)
        axes.get_x_axis().set_opacity(0)

        group1 = VGroup(q_entropy_text, q_entropy_formula, q_entropy_chart)
        group2 = VGroup(qp_cross_entropy_text, qp_cross_entropy_formula, qp_cross_entropy_chart)
        group3 = VGroup(p_entropy_text, p_entropy_formula, p_entropy_chart)
        group4 = VGroup(pq_cross_entropy_text, pq_cross_entropy_formula, pq_cross_entropy_chart)
        number_line_q.generate_target()
        number_line_q.target.match_width(axes.get_x_axis()).move_to(axes.get_x_axis())
        y_axis_label = qp_cross_entropy_formula.copy().scale(1.3).next_to(axes.get_y_axis(), LEFT, buff = 0.3)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    FadeOut(VGroup(q_entropy_text, q_entropy_formula, q_entropy_chart, new_q2_label_1), run_time = 1, suspend_mobject_updating = True),
                    FadeOut(VGroup(new_q2_label_2, new_p2_label_2, group3, group4), shift = DOWN*4, run_time = 1, suspend_mobject_updating = True),
                    AnimationGroup(
                        AnimationGroup(
                            MoveToTarget(number_line_q, run_time = 2),
                            VGroup(
                                qp_cross_entropy_text, qp_cross_entropy_formula, qp_cross_entropy_chart, number_line_p, new_p2_label_1
                            ).animate(run_time = 2).scale(1.3).shift(RIGHT*0.5).align_to(axes, DOWN),
                            TransformFromCopy(qp_cross_entropy_formula, y_axis_label, run_time = 1.5)
                        ),
                        FadeIn(axes, shift = DOWN*5 + LEFT, run_time = 1.5)
                    , lag_ratio = 0.2)
                )
            , lag_ratio = 0.8)
        )
        self.remove(new_q2_label_1, new_q2_label_2, new_p2_label_2)
        self.wait(2)

        # Fix P in place
        pin = SVGMobject("push_pin.svg").rotate(35 * DEG).scale(0.325).set_color(GREY).align_to(
            number_line_p.n2p(qp_cross_entropy_chart.distribution_trackers[0].get_value()), DR
        )
        pin.set_fill([GREY_D, GREY_B], 1)
        self.play(FadeIn(pin, shift=RIGHT * 0.25 + DOWN * 0.125, run_time = 2))
        self.wait(2)

        # Create the graph
        # epsilon = 5.6e-17
        epsilon = 1e-2
        infinity = 10
        def f(q1):
            if q1 < epsilon:
                f_of_epsilon = f(epsilon)
                return ((f_of_epsilon - infinity)/epsilon)*(q1 - epsilon) + f_of_epsilon
            if q1 > 1 - epsilon:
                f_of_1_minus_epsilon = f(1 - epsilon)
                return ((infinity - f_of_1_minus_epsilon)/epsilon)*(q1 - (1 - epsilon)) + f_of_1_minus_epsilon
            return sum([t.get_value()*-math.log2(q) for t, q in zip(qp_cross_entropy_chart.distribution_trackers, [q1, 1 - q1])])

        min_step = 0.001
        max_step = 0.01
        def calculate_step(p):
            # return -4*max_step*(p - 0.5)**2 + max_step
            if 0 <= p < 0.5:
                return 2*max_step*(p - 0.5) + max_step
            return -2*max_step*(p - 0.5) + max_step
        def get_curve():
            step = clip(calculate_step(qp_cross_entropy_chart.distribution_trackers[0].get_value()), min_step, max_step)
            return axes.get_graph(
                f, x_range = [0, 1, step]
            ).set_stroke(
                width = 2, color = PINK
            ).insert_n_curves(100)
        graph = always_redraw(get_curve)
        graph.suspend_updating()
        graph.insert_n_curves(10000)
        f_of_q_dot = Group(TrueDot(radius = 0.1), GlowDot(glow_factor = 1)).set_color(PURE_MAGENTA)
        f_of_q_dot.add_updater(
            lambda m: m.move_to(
                axes.c2p(
                    q_entropy_chart.distribution_trackers[0].get_value(),
                    f(q_entropy_chart.distribution_trackers[0].get_value())
                )
            )
        )
        self.play(
            AnimationGroup(
                ShowCreation(graph, suspend_mobject_updating = True, run_time = 3),
                FadeIn(f_of_q_dot, suspend_mobject_updating = True)
            , lag_ratio = 0.8)
        )
        f_of_q_dot.add_updater(lambda m: self.bring_to_front(m))

        # Change q_1
        for i in range(17):
            if i == 2:
                q1 = 0.2718
            elif i == 3:
                q1 = 0.123
            elif i == 4 or i == 10 or i == 16:
                q1 = qp_cross_entropy_chart.distribution_trackers[0].get_value()
            else:
                q1 = random.uniform(1/32, 1)
            distribution = [q1, 1 - q1]
            anims = [
                AnimationGroup(
                    q_entropy_chart.set_distribution(distribution),
                    AnimationGroup(*[
                        h.animate.set_value(-math.log2(q))
                        for h, q in zip(qp_cross_entropy_chart.bar_heights, distribution)
                    ])
                , run_time = 2)
            ]
            if i == 1:
                min_value_arrow = Arrow(ORIGIN, DOWN*1.8, thickness = 5).next_to(
                    axes.c2p(
                        qp_cross_entropy_chart.distribution_trackers[0].get_value(),
                        f(qp_cross_entropy_chart.distribution_trackers[0].get_value())
                    ),
                    UP, buff = 0
                )
                min_value_text = Tex(R"\text{min} = \displaystyle\sum_i p_i (-\log_2 p_i)", font_size = 35).next_to(min_value_arrow, UP, buff = -0.1)
                min_value_text["p_i"].set_color(GREEN)
                min_text = min_value_text["min"]
                min_value_text.shift(RIGHT*(min_value_arrow.get_x() - min_text.get_x()))
                anims.append(
                    AnimationGroup(
                        FadeIn(min_value_arrow, shift = DOWN*0.3, run_time = 1.5),
                        Write(min_text, run_time = 1.5)
                    , lag_ratio = 0.2)
                )
            if i == 3:
                number_line_p_group_copy = Group(number_line_p, pin, p1_dot, p1_triangle, p1_label).copy().clear_updaters()
                number_line_p_copy = number_line_p_group_copy[0]
                pin_copy = number_line_p_group_copy[1]
                p1_dot_copy = number_line_p_group_copy[2]
                p1_triangle_copy = number_line_p_group_copy[3]
                p1_label_copy = number_line_p_group_copy[4]
                p1_dot_copy.add_updater(
                    lambda m: m.move_to(number_line_p_copy.n2p(qp_cross_entropy_chart.distribution_trackers[0].get_value()))
                )
                p1_triangle_copy.add_updater(
                    lambda m: m.next_to(number_line_p_copy.n2p(qp_cross_entropy_chart.distribution_trackers[0].get_value()), UP, buff = 0)
                )
                p1_label_copy.add_updater(lambda m: m.next_to(p1_triangle_copy, UP, buff = 0.1))
                number_line_p_group_copy.suspend_updating()

                anims.append(
                    number_line_p_group_copy.animate(
                        run_time = 2, path_arc = PI*0.2
                    ).match_width(
                        number_line_q
                    ).match_x(
                        number_line_q.align_to(number_line_q, DOWN)
                    )
                )
            if i == 4:
                anims.append(FadeOut(number_line_p_group_copy[0]))
            # if i == 16:
            #     anims.append(Write(min_value_text[len("min"):], run_time = 2))

            self.play(*anims)

        # Clean up
        epsilon_2 = 0.001
        def g(p1):
            if p1 < epsilon_2 or p1 > 1 - epsilon_2:
                return 0
            return sum([p*-math.log2(p) for p in [p1, 1 - p1]])
        min_value_dot = f_of_q_dot.copy().clear_updaters().set_color(GREEN).set_opacity(0.5)
        min_value_dot.add_updater(
            lambda m: m.move_to(
                axes.c2p(
                    qp_cross_entropy_chart.distribution_trackers[0].get_value(),
                    g(qp_cross_entropy_chart.distribution_trackers[0].get_value())
                )
            )
        )
        q1_label.clear_updaters()
        f_of_q_dot.clear_updaters()
        self.remove(q1_dot)
        self.play(
            FadeOut(Group(min_text, min_value_arrow, f_of_q_dot, q1_triangle), suspend_mobject_updating = True),
            FadeOut(Group(pin, pin_copy), shift=LEFT * 0.25 + UP * 0.125),
            q1_label.animate.next_to(axes.get_x_axis()[0], RIGHT),
            FadeIn(min_value_dot, suspend_mobject_updating = True)
        )

        # Trace the minimum value of the graph
        number_line_p_group_copy.resume_updating()
        path = TracingTail(min_value_dot, time_traced = 5, stroke_color = GREEN)
        self.add(path)
        self.wait(3)
        graph.resume_updating()
        self.play(qp_cross_entropy_chart.set_distribution([1 - epsilon_2, epsilon_2]), run_time = 4)
        self.play(qp_cross_entropy_chart.set_distribution([epsilon_2, 1 - epsilon_2]), run_time = 4)
        graph.suspend_updating()

        # Replace the traced path with the full graph of the entropy of P
        p_entropy_graph = axes.get_graph(
            g, x_range = [0, 1, 0.01]
        ).set_stroke(
            width = 2, color = GREEN
        ).insert_n_curves(100)
        # self.wait(1)
        # self.remove(path)
        self.play(FadeIn(p_entropy_graph), run_time = 2)
        # self.wait(1)

        # Move around the distribution some more and write the entropy formula for p next to the new graph
        graph.resume_updating()
        for i in range(20):
            anims = [qp_cross_entropy_chart.set_distribution(random_distribution(2, thresh = 1/8))]
            if i == 0:
                entropy_of_p = min_value_text[len("min="):].scale(0.85).next_to(
                    p_entropy_graph, RIGHT, buff = -1
                ).shift(
                    UP*0.7
                ).set_stroke(
                    width = 3, color = BLACK, behind = True
                )
                self.play(Write(entropy_of_p, stroke_color = WHITE))
            self.play(*anims, run_time = 4)


class CrossEntropyDefinitionV3(InteractiveScene):
    def construct(self):
        # Show the chart for the first distribution
        encoding = ["0", "10", "110", "111"]
        first_distribution = [1/2, 1/4, 1/8, 1/8]
        first_distribution_chart = EntropyChart(
            first_distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]),
            probability_labels = [
                Tex(R"\frac{1}{" + ["2", "4", "8", "8"][i] + "}", font_size = 65)
                for i in range(4)
            ],
            bar_labels = [
                Tex(encoding[i], font_size = 57)
                for i in range(4)
            ],
            bar_heights = [1, 2, 3, 3],
            width = 12,
            height = 4.5,
            include_vertical_axis = False,
            segments_height = 1,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK],
            stroke_width = 1
        ).scale(0.3).to_corner(UL, buff = 0.7).shift(DOWN*0.7)
        first_distribution_chart.update()
        first_distribution_chart.clear_updaters()
        first_distribution_chart.bar_labels.set_color(WHITE)
        self.play(first_distribution_chart.create(), run_time = 2)
        self.wait(2)
        self.play(first_distribution_chart.animate.set_x(-FRAME_WIDTH*0.25), run_time = 2.5)
        self.wait(2)

        # Build the segments for the second chart
        second_distribution = [1/8, 1/8, 1/4, 1/2]
        second_distribution_chart = EntropyChart(
            second_distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]),
            probability_labels = [
                Tex(R"\frac{1}{" + ["8", "8", "4", "2"][i] + "}", font_size = 65)
                for i in range(4)
            ],
            bar_labels = [
                Tex(encoding[i], font_size = 57)
                for i in range(4)
            ],
            bar_heights = [1, 2, 3, 3],
            width = 12,
            height = 4.5,
            include_vertical_axis = False,
            segments_height = 1,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [PURE_MAGENTA, LIGHT_PINK],
            stroke_width = 1
        ).scale(0.3).to_edge(UP, buff = 1.4).set_x(FRAME_WIDTH*0.25)
        second_distribution_chart.update()
        second_distribution_chart.clear_updaters()
        second_distribution_chart.bar_labels.set_color(WHITE)
        for segment, e_label, p_label in list(zip(
            second_distribution_chart.segments.bars,
            second_distribution_chart.event_labels,
            second_distribution_chart.probability_labels
        )):
            self.play(
                AnimationGroup(
                    GrowFromCenter(segment),
                    FadeIn(e_label),
                    FadeIn(p_label),
                    suspend_mobject_updating = True
                , run_time = 2)
            )
            self.wait(2)
        self.wait(4)

        # The bars hop over from the old distribution to the new one
        self.play(
            AnimationGroup(*[
                TransformFromCopy(VGroup(bar1, label1), VGroup(bar2, label2), run_time = 3)
                for bar1, label1, bar2, label2 in list(zip(
                    first_distribution_chart.bars,
                    first_distribution_chart.bar_labels,
                    second_distribution_chart.bars,
                    second_distribution_chart.bar_labels
                ))[::-1]
            ], lag_ratio = 0.3)
        )
        self.wait(2)

        # Center everything
        self.play(
            VGroup(first_distribution_chart, second_distribution_chart).animate.scale(1.3).arrange(buff = 2).shift(DOWN*0.5)
        , run_time = 2.5)

        # Calculate the cross entropy
        weighted_sum_lines = VGroup(
            Tex(R"\frac{1}{8} \cdot 1", font_size = 23).next_to(second_distribution_chart.bars[0], UP),
            Tex(R"\frac{1}{8} \cdot 2", font_size = 23).next_to(second_distribution_chart.bars[1], UP),
            Tex(R"\frac{1}{4} \cdot 3", font_size = 23).next_to(second_distribution_chart.bars[2], UP),
            Tex(R"\frac{1}{2} \cdot 3", font_size = 23).next_to(second_distribution_chart.bars[3], UP)
        )
        for line in weighted_sum_lines:
            line[:3].set_color(GREEN)
            line[4:].set_color(PINK)

        self.play(
            AnimationGroup(
                TransformFromCopy(second_distribution_chart.probability_labels[0], weighted_sum_lines[0][:-2]),
                FadeIn(weighted_sum_lines[0][-2:], shift = UP*0.1)
            , run_time = 1.5)
        )
        self.wait(1.5)
        self.play(
            AnimationGroup(
                TransformFromCopy(second_distribution_chart.probability_labels[1], weighted_sum_lines[1][:-2]),
                FadeIn(weighted_sum_lines[1][-2:], shift = UP*0.1)
            , run_time = 1.5)
        )
        self.wait(2.5)
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    TransformFromCopy(label, line[:-2]),
                    FadeIn(line[-2:], shift = UP*0.1)
                , run_time = 1.5)
                for label, line in zip(second_distribution_chart.probability_labels[2:], weighted_sum_lines[2:])
            ], lag_ratio = 0.3)
        )

        self.wait(1)
        sum_result = Tex(
            R"\frac{1}{8} \cdot 1 + \frac{1}{8} \cdot 2 + \frac{1}{4} \cdot 3 + \frac{1}{2} \cdot 3 \\ = 2.625 \text{ bits}",
            font_size = 32,
            tex_to_color_map = {
                R"\frac{1}{8}": GREEN,
                R"\frac{1}{4}": GREEN,
                R"\frac{1}{2}": GREEN,
                " 1 ": PINK,
                " 2 ": PINK,
                " 3 ": PINK
            }
        ).next_to(second_distribution_chart, UP, buff = 0.3)

        self.play(TransformMatchingShapes(weighted_sum_lines, sum_result[:-10], path_arc = PI*0.2, run_time = 1.5))
        self.wait(0.5)
        self.play(FadeIn(sum_result[R"= 2.625 \text{ bits}"]))
        self.wait(2)

        # Write "cross entropy"
        cross_entropy_text = TexText("Cross Entropy:").set_fill(color = [PINK, GREEN]).next_to(sum_result, UP)
        for i, letter in enumerate(cross_entropy_text):
            letter.set_color(interpolate_color(PINK, GREEN, i/(len(cross_entropy_text) - 1)))
        self.play(Write(cross_entropy_text, run_time = 2.5))
        self.wait(0.5)
        rect1 = SurroundingRectangle(
            first_distribution_chart.probability_labels, stroke_width = 2, stroke_color = PINK
        ).stretch_to_fit_width(first_distribution_chart.bars.get_width()).match_x(first_distribution_chart.bars)
        self.play(FadeIn(rect1), run_time = 1.5)
        self.wait(3)
        rect2 = SurroundingRectangle(
            second_distribution_chart.probability_labels, stroke_width = 2, stroke_color = GREEN
        ).stretch_to_fit_width(second_distribution_chart.bars.get_width()).match_x(second_distribution_chart.bars)
        self.play(ReplacementTransform(rect1, rect2), run_time = 2.5)
        self.wait(2)
        self.play(FadeOut(rect2))
        self.wait(2)

        # Label the two charts with p and q
        p_chart = EntropyChart(
            second_distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]).scale(0.6) for i in range(4)]),
            probability_labels = VGroup(*[Tex(f"p_{i + 1}", font_size = 42) for i in range(4)]),
            bar_labels = [
                Tex(encoding[i], font_size = 27)
                for i in range(4)
            ],
            bar_heights = [1, 2, 3, 3],
            width = second_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = False,
            segments_height = 0.4,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [PURE_MAGENTA, LIGHT_PINK],
            stroke_width = 1
        )
        p_chart.suspend_updating()
        p_chart.bar_labels.set_color(WHITE)

        q_chart = EntropyChart(
            first_distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]).scale(0.6) for i in range(4)]),
            probability_labels = VGroup(*[Tex(f"q_{i + 1}", font_size = 42) for i in range(4)]),
            bar_labels = [
                Tex(encoding[i], font_size = 27)
                for i in range(4)
            ],
            width = first_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = False,
            segments_height = 0.4,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK],
            stroke_width = 1
        )
        q_chart.suspend_updating()
        q_chart.bar_labels.set_color(WHITE)

        p_chart.match_x(second_distribution_chart).align_to(second_distribution_chart.bars, UP)
        q_chart.match_x(first_distribution_chart).align_to(first_distribution_chart.bars, UP)
        self.play(
            FadeOut(VGroup(cross_entropy_text, sum_result)),
            ReplacementTransform(second_distribution_chart, p_chart, suspend_mobject_updating = True)
        , run_time = 2)
        self.wait(2)
        self.play(ReplacementTransform(first_distribution_chart, q_chart, suspend_mobject_updating = True), run_time = 2)
        self.wait(3)

        # Generalize the charts
        p_chart_general = EntropyChart(
            second_distribution,
            event_labels = VGroup(*[
                Tex(
                    (("s_" + str(i + 1)) if i < len(second_distribution) - 2 else R"\ldots" if i == len(second_distribution) - 2 else "s_n"),
                    font_size = 35
                ).set_color(BLACK)
                for i in range(len(second_distribution))
            ]),
            probability_labels = VGroup(*[
                Tex(
                    (("p_" + str(i + 1)) if i < len(second_distribution) - 2 else R"\ldots" if i == len(second_distribution) - 2 else "p_n"),
                    font_size = 42
                )
                for i in range(len(second_distribution))
            ]),
            bar_labels = VMobject(),
            bar_heights = [1, 2, 3, 3],
            width = second_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = True,
            vertical_axis_label_text = R"\begin{gathered}-\log_2 q_i \\ \ (\text{bits})\end{gathered}",
            vertical_axis_font_size = 30,
            segments_height = 0.4,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [PURE_MAGENTA, LIGHT_PINK],
            stroke_width = 1
        )
        p_chart_general.suspend_updating()

        q_chart_general = EntropyChart(
            first_distribution,
            event_labels = VGroup(*[
                Tex(
                    (("s_" + str(i + 1)) if i < len(first_distribution) - 2 else R"\ldots" if i == len(first_distribution) - 2 else "s_n"),
                    font_size = 35
                ).set_color(BLACK)
                for i in range(len(first_distribution))
            ]),
            probability_labels = VGroup(*[
                Tex(
                    (("q_" + str(i + 1)) if i < len(first_distribution) - 2 else R"\ldots" if i == len(first_distribution) - 2 else "q_n"),
                    font_size = 42
                )
                for i in range(len(first_distribution))
            ]),
            bar_labels = VMobject(),
            width = first_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = True,
            vertical_axis_label_text = R"\begin{gathered}-\log_2 q_i \\ \ (\text{bits})\end{gathered}",
            vertical_axis_font_size = 30,
            segments_height = 0.4,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK],
            stroke_width = 1
        )
        q_chart_general.suspend_updating()

        p_chart_general.shift(p_chart.bars.get_center() - p_chart_general.bars.get_center())
        q_chart_general.shift(q_chart.bars.get_center() - q_chart_general.bars.get_center())
        self.play(
            ReplacementTransform(
                VGroup(q_chart.bars, q_chart.segments, q_chart.probability_labels),
                VGroup(q_chart_general.bars, q_chart_general.segments, q_chart_general.probability_labels)
            , suspend_mobject_updating = True),
            ReplacementTransform(
                VGroup(p_chart.bars, p_chart.segments, p_chart.probability_labels),
                VGroup(p_chart_general.bars, p_chart_general.segments, p_chart_general.probability_labels)
            , suspend_mobject_updating = True),
            FadeOut(VGroup(q_chart.bar_labels, p_chart.bar_labels))
        , run_time = 2)
        self.wait(2)

        self.remove(q_chart, p_chart)
        self.add(q_chart_general, p_chart_general)
        VGroup(
            q_chart_general.vertical_axis,
            q_chart_general.vertical_axis_label,
            q_chart_general.reference_lines,
            p_chart_general.vertical_axis,
            p_chart_general.vertical_axis_label,
            p_chart_general.reference_lines
        ).set_opacity(0)

        # Show different possibilities for what the symbols could mean
        alternate_symbols_q_1 = VGroup(*[
            InstructionArrow([UP, DOWN, LEFT, RIGHT][i]).scale(0.08)
            for i in range(4)
        ])
        alternate_symbols_p_1 = alternate_symbols_q_1.copy()
        for i, (q_symb, p_symb) in enumerate(zip(alternate_symbols_q_1, alternate_symbols_p_1)):
            q_symb.move_to(q_chart_general.event_labels[i])
            p_symb.move_to(p_chart_general.event_labels[i])
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    FadeOut(label),
                    FadeIn(alternate_label)
                )
                for label, alternate_label in zip(q_chart_general.event_labels, alternate_symbols_q_1)
            ], lag_ratio = 0.2),
            AnimationGroup(*[
                AnimationGroup(
                    FadeOut(label),
                    FadeIn(alternate_label)
                )
                for label, alternate_label in zip(p_chart_general.event_labels, alternate_symbols_p_1)
            ], lag_ratio = 0.2)
        )
        self.wait(0.8)

        alternate_symbols_q_2 = VGroup(*[
            TexText(["a", "b", "c", "d"][i], font_size = 35).set_color(BLACK)
            for i in range(4)
        ])
        alternate_symbols_p_2 = alternate_symbols_q_2.copy()
        for i, (q_symb, p_symb) in enumerate(zip(alternate_symbols_q_2, alternate_symbols_p_2)):
            q_symb.next_to(q_chart_general.probability_labels[0], UP, buff = 0.345).match_x(q_chart_general.probability_labels[i])
            p_symb.next_to(p_chart_general.probability_labels[0], UP, buff = 0.345).match_x(p_chart_general.probability_labels[i])
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    FadeOut(label),
                    FadeIn(alternate_label)
                )
                for label, alternate_label in zip(alternate_symbols_q_1, alternate_symbols_q_2)
            ], lag_ratio = 0.2),
            AnimationGroup(*[
                AnimationGroup(
                    FadeOut(label),
                    FadeIn(alternate_label)
                )
                for label, alternate_label in zip(alternate_symbols_p_1, alternate_symbols_p_2)
            ], lag_ratio = 0.2)
        )
        self.wait(0.8)

        alternate_symbols_q_3 = VGroup(*[
            Tex(["e", R"\wedge", R"\pi", "i"][i], font_size = 40 if i != 1 else 20).set_color(BLACK)
            for i in range(4)
        ])
        alternate_symbols_p_3 = alternate_symbols_q_3.copy()
        for i, (q_symb, p_symb) in enumerate(zip(alternate_symbols_q_3, alternate_symbols_p_3)):
            q_symb.next_to(q_chart_general.probability_labels[0], UP, buff = 0.33 if i != 1 else 0.455).match_x(q_chart_general.probability_labels[i])
            p_symb.next_to(p_chart_general.probability_labels[0], UP, buff = 0.33 if i != 1 else 0.455).match_x(p_chart_general.probability_labels[i])
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    FadeOut(label),
                    FadeIn(alternate_label)
                )
                for label, alternate_label in zip(alternate_symbols_q_2, alternate_symbols_q_3)
            ], lag_ratio = 0.2),
            AnimationGroup(*[
                AnimationGroup(
                    FadeOut(label),
                    FadeIn(alternate_label)
                )
                for label, alternate_label in zip(alternate_symbols_p_2, alternate_symbols_p_3)
            ], lag_ratio = 0.2)
        )
        self.wait(0.8)

        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    FadeOut(alternate_label),
                    FadeIn(label)
                )
                for alternate_label, label in zip(alternate_symbols_q_3, q_chart_general.event_labels)
            ], lag_ratio = 0.2),
            AnimationGroup(*[
                AnimationGroup(
                    FadeOut(alternate_label),
                    FadeIn(label)
                )
                for alternate_label, label in zip(alternate_symbols_p_3, p_chart_general.event_labels)
            ], lag_ratio = 0.2)
        )
        self.wait(0.8)


        # Show bar heights for q chart
        q_chart = q_chart_general
        p_chart = p_chart_general
        q_chart.resume_updating()
        p_chart.resume_updating()
        self.play(
            q_chart.segments.bars.animate.stretch(0.2, 1).align_to(q_chart.segments.bars, DOWN),
            p_chart.segments.bars.animate.stretch(0.2, 1).align_to(p_chart.segments.bars, DOWN),
            AnimationGroup(*[label.animate.scale(0.01).shift(DOWN*0.2).set_opacity(0) for label in q_chart.event_labels]),
            AnimationGroup(*[label.animate.scale(0.01).shift(DOWN*0.2).set_opacity(0) for label in p_chart.event_labels])
        , run_time = 1)
        self.remove(q_chart.event_labels, p_chart.event_labels)
        VGroup(
            q_chart.vertical_axis, q_chart.vertical_axis_label, q_chart.reference_lines
        ).shift(
            DOWN*(q_chart.reference_lines[0][0].get_y() - q_chart.segments.bars.get_top()[1])
        )
        VGroup(
            p_chart.vertical_axis, p_chart.vertical_axis_label, p_chart.reference_lines
        ).shift(
            DOWN*(p_chart.reference_lines[0][0].get_y() - p_chart.segments.bars.get_top()[1])
        )
        q_chart.suspend_updating()
        p_chart.suspend_updating()

        self.play(FancyCircumscribe(VGroup(q_chart.bars, q_chart.probability_labels)))

        p_chart.save_state()
        self.play(
            self.camera.frame.animate.match_x(VGroup(q_chart, p_chart)).shift(UP*0.7),
            AnimationGroup(
                p_chart.animate.fade(0.9),
                AnimationGroup(
                    Write(q_chart.vertical_axis_label.set_opacity(1), run_time = 2),
                    ShowCreation(q_chart.vertical_axis.set_opacity(1)),
                    AnimationGroup(*[
                        ShowCreation(line.set_opacity(1))
                        for line in q_chart.reference_lines
                    ], lag_ratio = 0.1)
                )
            , lag_ratio = 0.3)
        , run_time = 2)
        self.wait(1)
        Group(*self.mobjects).shift(-self.camera.frame.get_center())
        self.camera.frame.center()

        # Show the avg. bits per instruction for Q
        q_bits_per_instruction = TexText(
            R"Avg. bits per instruction: \\[0.1in] $\displaystyle\sum_i q_i (-\log_2 q_i)$",
            font_size = 35,
            tex_to_color_map = {
                "q_i": PINK
            }
        ).next_to(q_chart.bars, UP, buff = 1.4)
        self.play(Write(q_bits_per_instruction), run_time = 3)
        self.wait(1.5)

        q_bits_per_instruction.save_state()
        q_chart.save_state()
        for i in range(4):
            anims = [
                AnimationGroup(
                    VGroup(
                        q_chart.segments.bars[i + 1:],
                        q_chart.event_labels[i + 1:],
                        q_chart.probability_labels[i + 1:],
                        q_chart.segments.bars[:i],
                        q_chart.event_labels[:i],
                        q_chart.probability_labels[:i]
                    ).animate.set_opacity(0.2),
                    VGroup(
                        q_chart.segments.bars[i],
                        q_chart.event_labels[i],
                        q_chart.probability_labels[i]
                    ).animate.set_opacity(1),
                )
            ]
            if i == 0:
                anims.append(
                    AnimationGroup(
                        q_bits_per_instruction.animate.fade(0.8),
                        q_bits_per_instruction["q_i"][0].animate.set_opacity(1),
                        q_chart.bars.animate.fade(0.8)
                    )
                )
            self.play(AnimationGroup(*anims))

        for i in range(4):
            anims = [
                AnimationGroup(
                    VGroup(
                        q_chart.bars[i + 1:],
                        q_chart.bars[:i],
                        q_chart.bar_labels[:i],
                    ).animate.set_opacity(0.2),
                    q_chart.bars[i].animate.set_opacity(1),
                )
            ]
            if i == 0:
                anims.append(
                    AnimationGroup(
                        q_bits_per_instruction[-8:-1].animate.set_opacity(1),
                        q_bits_per_instruction["q_i"][0].animate.fade(0.8),
                        VGroup(
                            q_chart.segments.bars[3],
                            q_chart.event_labels[3],
                            q_chart.probability_labels[3]
                        ).animate.fade(0.8)
                    )
                )
            self.play(AnimationGroup(*anims))
        self.play(q_chart.animate.restore(), q_bits_per_instruction.animate.restore(), run_time = 2)
        self.wait(2)

        # Label the equation with "entropy"
        entropy_text = TexText(
            "Entropy of Q",
            tex_to_color_map = {"Q": PINK}
        ).match_height(
            q_bits_per_instruction["Avg. bits per instruction:"]
        ).move_to(
            q_bits_per_instruction["Avg. bits per instruction:"]
        )
        self.play(FadeOut(q_bits_per_instruction["Avg. bits per instruction:"]), FadeIn(entropy_text))
        self.wait(2)

        # Highlight the area of the diagram
        q_entropy_formula = q_bits_per_instruction[len("Avg.bitsperinstruction:"):]
        self.play(AnimationGroup(*[Indicate(bar, scale_factor = 1.1) for bar in q_chart.bars], lag_ratio = 0.1, run_time = 2.5))

        # Show why the area represents the expression
        entropy_text.save_state()
        q_entropy_formula.save_state()
        q_chart.save_state()
        for i in range(4):
            anims = [
                AnimationGroup(
                    VGroup(
                        q_chart.segments.bars[i + 1:],
                        q_chart.event_labels[i + 1:],
                        q_chart.probability_labels[i + 1:],
                        q_chart.segments.bars[:i],
                        q_chart.event_labels[:i],
                        q_chart.probability_labels[:i]
                    ).animate.set_opacity(0.2),
                    VGroup(
                        q_chart.segments.bars[i],
                        q_chart.event_labels[i],
                        q_chart.probability_labels[i]
                    ).animate.set_opacity(1),
                )
            ]
            if i == 0:
                anims.append(
                    AnimationGroup(
                        entropy_text.animate.fade(0.8),
                        q_entropy_formula.animate.fade(0.8),
                        q_bits_per_instruction["q_i"][0].animate.set_opacity(1),
                        q_chart.bars.animate.fade(0.8)
                    )
                )
            self.play(AnimationGroup(*anims), run_time = 0.62)

        for i in range(4):
            anims = [
                AnimationGroup(
                    VGroup(
                        q_chart.bars[i + 1:],
                        q_chart.bars[:i],
                        q_chart.bar_labels[:i],
                    ).animate.set_opacity(0.2),
                    q_chart.bars[i].animate.set_opacity(1),
                )
            ]
            if i == 0:
                anims.append(
                    AnimationGroup(
                        q_bits_per_instruction[-8:-1].animate.set_opacity(1),
                        q_bits_per_instruction["q_i"][0].animate.fade(0.8),
                        VGroup(
                            q_chart.segments.bars[3],
                            q_chart.event_labels[3],
                            q_chart.probability_labels[3]
                        ).animate.fade(0.8)
                    )
                )
            self.play(AnimationGroup(*anims), run_time = 0.62)
        self.play(q_chart.animate.restore(), q_entropy_formula.animate.restore(), entropy_text.animate.restore(), run_time = 1.2)
        self.bring_to_back(q_chart.reference_lines, p_chart.reference_lines)

        # Switch focus to the p chart
        self.play(VGroup(p_chart.segments, p_chart.probability_labels).animate.set_opacity(1))

        # Show the encoding and the original distribution
        self.play(
            AnimationGroup(
                AnimationGroup(
                    Write(p_chart.vertical_axis_label.set_opacity(1), run_time = 2),
                    ShowCreation(p_chart.vertical_axis.set_opacity(1)),
                    AnimationGroup(*[
                        ShowCreation(line.set_opacity(1))
                        for line in p_chart.reference_lines
                    ], lag_ratio = 0.1)
                ),
                AnimationGroup(*[
                    TransformFromCopy(bar1, bar2, run_time = 3)
                    for bar1, bar2 in list(zip(
                        q_chart.bars.copy().set_opacity(0.8),
                        p_chart.bars.copy().set_opacity(0.8),
                    ))[::-1]
                ], lag_ratio = 0.3)
            , lag_ratio = 0.7)
        )
        self.clear()
        p_chart.bars.set_opacity(0.8)
        self.add(q_chart, p_chart, entropy_text, q_entropy_formula)
        self.bring_to_back(q_chart.reference_lines, p_chart.reference_lines)

        # Write cross entropy equation
        general_equation = TexText(
            R"Avg. bits per instruction: \\[0.1in] $\displaystyle\sum_i p_i (-\log_2 q_i)$",
            font_size = 35,
            tex_to_color_map = {
                "p_i": GREEN,
                "q_i": PINK
            }
        ).next_to(p_chart.bars, UP, buff = 1.4)
        self.play(Write(general_equation), run_time = 3)
        self.wait(2)
        self.play(Flash(general_equation["p_i"], flash_radius = 0.5))

        # Replace "Avg. bits per instruction" with "Cross Entropy(Q, P)"
        self.play(FancyCircumscribe(general_equation))
        cross_entropy_text = TexText(
            "Cross Entropy of Q relative to P",
            tex_to_color_map = {"Q": PINK, "P": GREEN}
        ).match_height(
            general_equation["Avg. bits per instruction:"]
        ).move_to(
            general_equation["Avg. bits per instruction:"]
        )
        self.play(FadeOut(general_equation["Avg. bits per instruction:"]), FadeIn(cross_entropy_text), run_time = 2)
        self.wait(3)

        # Show special notation
        full_sum = general_equation[len("Avg.bitsperinstruction:"):]
        cross_entropy_group = VGroup(cross_entropy_text, full_sum)
        cross_entropy_group.save_state()
        self.play(
            FadeOut(
                VGroup(
                    q_chart,
                    p_chart,
                    entropy_text,
                    q_bits_per_instruction[len("Avg.bitsperinstruction:"):]
                )
            , run_time = 1.6, shift = DOWN*3),
            cross_entropy_group.animate(run_time = 2, path_arc = PI*0.2).set_y(0).to_edge(LEFT, buff = 2)
        )
        notations = BulletedList(
            R"$H(P, Q)$",
            R"$H(P \parallel Q)$",
            R"$H_Q(P)$",
            R"$\mathbb{E}_P[-\log Q]$",
            R"$\langle -\log Q \rangle_P$",
            tex_to_color_map = {"Q": PINK, "P": GREEN}
        ).to_edge(RIGHT, buff = 2.2)
        brace = Brace(notations, LEFT)
        self.play(
            GrowFromEdge(brace, RIGHT),
            AnimationGroup(*[FadeIn(line, shift = DOWN*0.3) for line in notations], lag_ratio = 0.2)
        , run_time = 3)
        self.wait(1)

        # Put everything back
        self.play(
            FadeOut(VGroup(brace, notations), shift = RIGHT*5, run_time = 1.5),
            FadeIn(
                VGroup(
                    q_chart,
                    p_chart,
                    entropy_text,
                    q_bits_per_instruction[len("Avg.bitsperinstruction:"):]
                )
            , run_time = 2, shift = UP*3),
            cross_entropy_group.animate(run_time = 2, path_arc = -PI*0.2).restore()
        )
        self.bring_to_back(q_chart.reference_lines, p_chart.reference_lines)
        self.wait(1)

        # Highlight the diagram
        self.play(FancyCircumscribe(p_chart, run_time=5))

        # Show the widths and heights of each bar in the cross entropy diagram
        self.play(FancyCircumscribe(p_chart.probability_labels), FancyCircumscribe(general_equation["p_i"]))
        self.play(FancyCircumscribe(p_chart.vertical_axis_label, run_time = 5), FancyCircumscribe(general_equation[-8:-1], run_time = 5))
        self.wait(3)


        # Change the distribution Q
        q_entropy_chart = q_chart
        qp_cross_entropy_chart = p_chart
        q_entropy_chart.remove(q_entropy_chart.event_labels)
        qp_cross_entropy_chart.remove(qp_cross_entropy_chart.event_labels)
        q_entropy_chart.resume_updating()
        qp_cross_entropy_chart.resume_updating()

        def update_qp_cross_entropy_bar_heights(m):
            for t, h in zip(q_entropy_chart.distribution_trackers, m.bar_heights):
                h.set_value(-math.log2(t.get_value()))
        qp_cross_entropy_chart.add_updater(update_qp_cross_entropy_bar_heights)
        self.add(q_entropy_chart, qp_cross_entropy_chart)


        for _ in range(5):
            new_distribution = random_distribution(4, thresh = 1/8)
            self.play(q_entropy_chart.set_distribution(new_distribution), run_time = 2)
        self.wait(2)


        # Simplify to distributions over two events
        distribution_1 = [0.5, 0.5]
        distribution_2 = [0.9, 0.1]
        q_entropy_chart_two_events = EntropyChart(
            distribution_1,
            event_labels = None,
            probability_labels = VGroup(*[
                DecimalNumber(distribution_1[i], font_size = 30)
                for i in range(2)
            ]),
            bar_labels = None,
            width = first_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = True,
            vertical_axis_label_text = R"\begin{gathered}-\log_2 q_i \\ \ (\text{bits})\end{gathered}",
            vertical_axis_font_size = 30,
            segments_height = q_entropy_chart.segments.get_height(),
            fill_colors = [PURE_MAGENTA, LIGHT_PINK],
            stroke_width = 1
        ).move_to(q_entropy_chart)
        q_entropy_chart_two_events.suspend_updating()

        qp_cross_entropy_chart_two_events = EntropyChart(
            distribution_2,
            event_labels = None,
            probability_labels = VGroup(*[
                DecimalNumber(distribution_2[i], font_size = 30)
                for i in range(2)
            ]),
            bar_labels = None,
            bar_heights = [-math.log2(q) for q in distribution_1],
            width = second_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = True,
            vertical_axis_label_text = R"\begin{gathered}-\log_2 q_i \\ \ (\text{bits})\end{gathered}",
            vertical_axis_font_size = 30,
            segments_height = qp_cross_entropy_chart.segments.get_height(),
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [PURE_MAGENTA, LIGHT_PINK],
            stroke_width = 1
        ).move_to(qp_cross_entropy_chart)
        qp_cross_entropy_chart_two_events.suspend_updating()

        # Replace the charts
        self.play(
            FadeOut(VGroup(q_entropy_chart, qp_cross_entropy_chart), suspend_mobject_updating = True),
            FadeIn(VGroup(q_entropy_chart_two_events, qp_cross_entropy_chart_two_events))
        )
        self.wait(2)
        self.play(FancyCircumscribe(q_entropy_chart_two_events.probability_labels, run_time = 5))
        self.wait(1)
        self.play(FancyCircumscribe(qp_cross_entropy_chart_two_events.probability_labels, run_time = 5))
        self.wait(2)

        # Calculate the entropy of Q
        q_entropy_text = entropy_text
        information_1 = Tex(
            R"-\log_2 0.50 = 1", font_size = 30
        ).set_color(PURE_MAGENTA).next_to(q_entropy_chart_two_events.bars[0], UP, buff = 0.1)
        information_2 = Tex(
            R"-\log_2 0.50 = 1", font_size = 30
        ).set_color(LIGHT_PINK).next_to(q_entropy_chart_two_events.bars[1], UP, buff = 0.1)
        self.play(Write(information_1), Write(information_2))
        self.wait(2)

        q_entropy_formula_extension = Tex(R"0.50 \cdot 1 + 0.50 \cdot 1 = 1", font_size = 35).next_to(q_entropy_formula, DOWN)
        q_entropy_formula_extension["0.50"][0].set_color(PURE_MAGENTA)
        q_entropy_formula_extension["1"][0].set_color(PURE_MAGENTA)
        q_entropy_formula_extension["0.50"][1].set_color(LIGHT_PINK)
        q_entropy_formula_extension["1"][1].set_color(LIGHT_PINK)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    TransformMatchingShapes(
                        q_entropy_chart_two_events.probability_labels[0].copy(),
                        q_entropy_formula_extension["0.50"][0],
                        path_arc = PI*0.1
                    , run_time = 1.3),
                    GrowFromCenter(q_entropy_formula_extension[R"\cdot"][0]),
                    TransformMatchingShapes(information_1[-1].copy(), q_entropy_formula_extension["1"][0], path_arc = PI*0.1, run_time = 1.3),
                    Write(q_entropy_formula_extension["+"]),
                    TransformMatchingShapes(
                        q_entropy_chart_two_events.probability_labels[1].copy(),
                        q_entropy_formula_extension["0.50"][1],
                        path_arc = PI*0.1
                    , run_time = 1.3),
                    GrowFromCenter(q_entropy_formula_extension[R"\cdot"][1]),
                    TransformMatchingShapes(information_2[-1].copy(), q_entropy_formula_extension["1"][1], path_arc = PI*0.1, run_time = 1.3)
                , lag_ratio = 0.3),
                Write(q_entropy_formula_extension[-2:])
            , lag_ratio = 0.85)
        )

        # Calculate the cross entropy of Q relative to P
        qp_cross_entropy_text = cross_entropy_text
        qp_cross_entropy_formula = full_sum
        info_copy_1 = Tex("1", font_size = 30).set_color(PURE_MAGENTA).next_to(qp_cross_entropy_chart_two_events.bars[0], UP, buff = 0.1)
        info_copy_2 = Tex("1", font_size = 30).set_color(LIGHT_PINK).next_to(qp_cross_entropy_chart_two_events.bars[1], UP, buff = 0.1)
        self.play(
            AnimationGroup(*[
                TransformMatchingShapes(info["1"].copy(), info_copy)
                for info, info_copy in zip([information_2, information_1], [info_copy_2, info_copy_1])
            ], lag_ratio = 0.5)
        , run_time = 2)
        self.wait(2)

        qp_cross_entropy_formula_extension = Tex(R"0.90 \cdot 1 + 0.10 \cdot 1 = 1", font_size = 35).next_to(qp_cross_entropy_formula, DOWN)
        qp_cross_entropy_formula_extension["0.90"].set_color(GREEN_B)
        qp_cross_entropy_formula_extension["1"][0].set_color(PURE_MAGENTA)
        qp_cross_entropy_formula_extension["0.10"].set_color(GREEN_D)
        qp_cross_entropy_formula_extension["1"][2].set_color(LIGHT_PINK)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    TransformMatchingShapes(
                        qp_cross_entropy_chart_two_events.probability_labels[0].copy(),
                        qp_cross_entropy_formula_extension["0.90"],
                        path_arc = PI*0.1
                    , run_time = 1.3),
                    GrowFromCenter(qp_cross_entropy_formula_extension[R"\cdot"][0]),
                    TransformMatchingShapes(info_copy_1.copy(), qp_cross_entropy_formula_extension["1"][0], path_arc = PI*0.1, run_time = 1.3),
                    Write(qp_cross_entropy_formula_extension["+"]),
                    TransformMatchingShapes(
                        qp_cross_entropy_chart_two_events.probability_labels[1].copy(),
                        qp_cross_entropy_formula_extension["0.10"],
                        path_arc = PI*0.1
                    , run_time = 1.3),
                    GrowFromCenter(qp_cross_entropy_formula_extension[R"\cdot"][1]),
                    TransformMatchingShapes(info_copy_2.copy(), qp_cross_entropy_formula_extension["1"][2], path_arc = PI*0.1, run_time = 1.3)
                , lag_ratio = 0.2),
                Write(qp_cross_entropy_formula_extension[-2:])
            , lag_ratio = 0.85)
        )
        self.wait(1)
        self.play(
            FadeOut(
                VGroup(
                    information_1,
                    information_2,
                    q_entropy_formula_extension,
                    info_copy_1,
                    info_copy_2,
                    qp_cross_entropy_formula_extension
                )
            )
        )

        # Swap the distributions
        q_entropy_chart_two_events.resume_updating()
        qp_cross_entropy_chart_two_events.resume_updating()
        for label, t in zip(q_entropy_chart_two_events.probability_labels, q_entropy_chart_two_events.distribution_trackers):
            label.add_updater(lambda m, t = t: m.set_value(t.get_value()))
        for label, t in zip(qp_cross_entropy_chart_two_events.probability_labels, qp_cross_entropy_chart_two_events.distribution_trackers):
            label.add_updater(lambda m, t = t: m.set_value(t.get_value()))

        def update_qp_cross_entropy_bar_heights(m):
            for t, h in zip(q_entropy_chart_two_events.distribution_trackers, m.bar_heights):
                h.set_value(-math.log2(t.get_value()))
        qp_cross_entropy_chart_two_events.add_updater(update_qp_cross_entropy_bar_heights)

        self.play(
            q_entropy_chart_two_events.set_distribution([0.9, 0.1]),
            qp_cross_entropy_chart_two_events.set_distribution([0.5, 0.5]),
            AnimationGroup(*[h.animate.set_value(-math.log2(q)) for h, q in zip(qp_cross_entropy_chart_two_events.bar_heights, [0.9, 0.1])])
        , run_time = 2)
        self.wait(2)


        # Calculate the new entropy of Q
        q_entropy_text = entropy_text
        information_1_value = round(-math.log2(0.9), 2)
        information_2_value = round(-math.log2(0.1), 2)
        entropy_value = round(0.9*-math.log2(0.9) + 0.1*-math.log2(0.1), 2)
        information_1 = Tex(
            RF"-\log_2 0.90 = {information_1_value}", font_size = 30
        ).set_color(PURE_MAGENTA).next_to(q_entropy_chart_two_events.bars[0], UP, buff = 0.07)
        information_2 = Tex(
            RF"-\log_2 0.10 = {information_2_value}", font_size = 30
        ).set_color(LIGHT_PINK).next_to(q_entropy_chart_two_events.bars[1], UP, buff = 0).align_to(q_entropy_chart_two_events.bars[1], RIGHT)
        self.play(Write(information_1, run_time = 2))
        self.wait(1)
        self.play(Write(information_2, run_time = 2))
        self.wait(2)

        q_entropy_formula_extension = Tex(
            RF"0.90 \cdot {information_1_value} + 0.10 \cdot {information_2_value} = {entropy_value}",
            font_size = 35
        ).next_to(q_entropy_formula, DOWN)
        q_entropy_formula_extension["0.90"].set_color(PURE_MAGENTA)
        q_entropy_formula_extension[str(information_1_value)].set_color(PURE_MAGENTA)
        q_entropy_formula_extension["0.10"].set_color(LIGHT_PINK)
        q_entropy_formula_extension[str(information_2_value)].set_color(LIGHT_PINK)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    TransformMatchingShapes(
                        q_entropy_chart_two_events.probability_labels[0].copy(),
                        q_entropy_formula_extension["0.90"],
                        path_arc = PI*0.1
                    , run_time = 1.3),
                    GrowFromCenter(q_entropy_formula_extension[R"\cdot"][0]),
                    TransformMatchingShapes(
                        information_1[-4:].copy(),
                        q_entropy_formula_extension[str(information_1_value)],
                        path_arc = PI*0.1,
                        run_time = 1.3
                    ),
                    Write(q_entropy_formula_extension["+"]),
                    TransformMatchingShapes(
                        q_entropy_chart_two_events.probability_labels[1].copy(),
                        q_entropy_formula_extension["0.10"],
                        path_arc = PI*0.1
                    , run_time = 1.3),
                    GrowFromCenter(q_entropy_formula_extension[R"\cdot"][1]),
                    TransformMatchingShapes(
                        information_2[-4:].copy(),
                        q_entropy_formula_extension[str(information_2_value)],
                        path_arc = PI*0.1,
                        run_time = 1.3
                    )
                , lag_ratio = 0.2),
                Write(q_entropy_formula_extension[-5:])
            , lag_ratio = 0.85)
        )
        self.play(FancyCircumscribe(q_entropy_formula_extension[-4:]))

        # Calculate the new cross entropy of Q relative to P
        qp_cross_entropy_text = cross_entropy_text
        qp_cross_entropy_formula = full_sum
        info_copy_1 = Tex(
            str(information_1_value), font_size = 30
        ).set_color(
            PURE_MAGENTA
        ).next_to(
            qp_cross_entropy_chart_two_events.bars[0], UP, buff = 0.1
        )
        info_copy_2 = Tex(
            str(information_2_value), font_size = 30
        ).set_color(
            LIGHT_PINK
        ).next_to(
            qp_cross_entropy_chart_two_events.bars[1], UP, buff = 0.1
        )
        self.play(
            AnimationGroup(*[
                TransformMatchingShapes(info[-4:].copy(), info_copy)
                for info, info_copy in zip([information_2, information_1], [info_copy_2, info_copy_1])
            ], lag_ratio = 0.5)
        , run_time = 2)
        self.wait(2)

        cross_entropy_value = round(-math.log2(0.9)*0.5 + -math.log2(0.1)*0.5, 2)
        qp_cross_entropy_formula_extension = Tex(
            RF"0.50 \cdot {information_1_value} + 0.50 \cdot {information_2_value} = {cross_entropy_value}",
            font_size = 35
        ).next_to(qp_cross_entropy_formula, DOWN)
        qp_cross_entropy_formula_extension["0.50"].set_color(GREEN_B)
        qp_cross_entropy_formula_extension[str(information_1_value)].set_color(PURE_MAGENTA)
        qp_cross_entropy_formula_extension["0.50"].set_color(GREEN_D)
        qp_cross_entropy_formula_extension[str(information_2_value)].set_color(LIGHT_PINK)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    TransformMatchingShapes(
                        qp_cross_entropy_chart_two_events.probability_labels[0].copy(),
                        qp_cross_entropy_formula_extension["0.50"][0],
                        path_arc = PI*0.1
                    , run_time = 1.3),
                    GrowFromCenter(qp_cross_entropy_formula_extension[R"\cdot"][0]),
                    TransformMatchingShapes(
                        info_copy_1.copy(),
                        qp_cross_entropy_formula_extension[str(information_1_value)],
                        path_arc = PI*0.1,
                        run_time = 1.3
                    ),
                    Write(qp_cross_entropy_formula_extension["+"]),
                    TransformMatchingShapes(
                        qp_cross_entropy_chart_two_events.probability_labels[1].copy(),
                        qp_cross_entropy_formula_extension["0.50"][1],
                        path_arc = PI*0.1
                    , run_time = 1.3),
                    GrowFromCenter(qp_cross_entropy_formula_extension[R"\cdot"][1]),
                    TransformMatchingShapes(
                        info_copy_2.copy(),
                        qp_cross_entropy_formula_extension[str(information_2_value)],
                        path_arc = PI*0.1,
                        run_time = 1.3
                    )
                , lag_ratio = 0.2),
                Write(qp_cross_entropy_formula_extension[-5:])
            , lag_ratio = 0.85)
        )
        self.wait(1)
        self.play(
            FadeOut(
                VGroup(
                    information_1,
                    information_2,
                    q_entropy_formula_extension,
                    info_copy_1,
                    info_copy_2,
                    qp_cross_entropy_formula_extension
                )
            )
        )

        # Generalize the two-event distributions
        distribution_1 = [t.get_value() for t in q_entropy_chart_two_events.distribution_trackers]
        q_entropy_chart_two_events_general = EntropyChart(
            distribution_1,
            event_labels = None,
            probability_labels = VGroup(*[
                Tex(F"q_{str(i + 1)}", font_size = 42)
                for i in range(2)
            ]),
            bar_labels = None,
            width = first_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = True,
            vertical_axis_label_text = R"\begin{gathered}-\log_2 q_i \\ \ (\text{bits})\end{gathered}",
            vertical_axis_font_size = 30,
            segments_height = q_entropy_chart_two_events.segments.get_height(),
            fill_colors = [PURE_MAGENTA, LIGHT_PINK],
            stroke_width = 1
        ).move_to(q_entropy_chart_two_events)

        qp_cross_entropy_chart_two_events_general = EntropyChart(
            distribution_2,
            event_labels = None,
            probability_labels = VGroup(*[
                Tex(F"p_{str(i + 1)}", font_size = 42)
                for i in range(2)
            ]),
            bar_labels = None,
            bar_heights = [-math.log2(q) for q in distribution_1],
            width = second_distribution_chart.get_width(),
            height = 2.3,
            include_vertical_axis = True,
            vertical_axis_label_text = R"\begin{gathered}-\log_2 q_i \\ \ (\text{bits})\end{gathered}",
            vertical_axis_font_size = 30,
            segments_height = q_entropy_chart_two_events.segments.get_height(),
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [PURE_MAGENTA, LIGHT_PINK],
            stroke_width = 1
        ).move_to(qp_cross_entropy_chart_two_events)
        q_entropy_chart_two_events_general.add_updater(lambda m: m.set_stroke(width = 0.5))
        qp_cross_entropy_chart_two_events_general.add_updater(lambda m: m.set_stroke(width = 0.5))

        self.play(
            AnimationGroup(
                FadeIn(
                    VGroup(
                        q_entropy_chart_two_events_general,
                        qp_cross_entropy_chart_two_events_general
                    )
                , suspend_mobject_updating = True),
                FadeOut(
                    VGroup(
                        q_entropy_chart_two_events,
                        qp_cross_entropy_chart_two_events
                    )
                , suspend_mobject_updating = True)
            , lag_ratio = 0.2)
        )
        q_entropy_chart = q_entropy_chart_two_events_general
        qp_cross_entropy_chart = qp_cross_entropy_chart_two_events_general

        # Play around with the distrbutions
        def update_qp_cross_entropy_bar_heights(m):
            for t, h in zip(q_entropy_chart.distribution_trackers, m.bar_heights):
                h.set_value(-math.log2(t.get_value()))
        qp_cross_entropy_chart.add_updater(update_qp_cross_entropy_bar_heights)        

        for i in range(12):
            new_distribution_1 = (
                random_distribution(2, thresh = 1/8) if i % 2 == 1 else
                [t.get_value() for t in q_entropy_chart.distribution_trackers]
            )
            new_distribution_2 = (
                random_distribution(2, thresh = 1/8) if i % 2 == 0 else
                [t.get_value() for t in qp_cross_entropy_chart.distribution_trackers]
            )
            anims = [
                AnimationGroup(
                    q_entropy_chart.set_distribution(new_distribution_1),
                    qp_cross_entropy_chart.set_distribution(new_distribution_2)
                , run_time = 2)
            ]

            if i == 3:
                # Highlight q_1 and add the number line
                circ_opacity_tracker_1 = ValueTracker(0)
                circ = always_redraw(
                    lambda: Circle(
                        stroke_width = 2,
                        stroke_color = YELLOW,
                        stroke_opacity = circ_opacity_tracker_1.get_value()
                    ).surround(
                        q_entropy_chart.probability_labels[0],
                        buff = 0.2
                    )
                )
                self.add(circ)
                anims.append(circ_opacity_tracker_1.animate.set_value(1))

                number_line_q = NumberLine(
                    [0, 1, 0.1],
                    include_numbers = True,
                    numbers_to_exclude = [0.1*i for i in range(1, 10)],
                    width = q_entropy_chart.segments.get_width()
                ).next_to(q_entropy_chart.bars, DOWN, buff = 1.6)
                self.add(number_line_q)
                q1_dot = Group(TrueDot(), GlowDot()).set_color(PURE_MAGENTA)
                q1_dot.add_updater(lambda m: m.move_to(number_line_q.n2p(q_entropy_chart.distribution_trackers[0].get_value())))
                q1_triangle = Triangle(fill_opacity = 0.7, fill_color = PURE_MAGENTA, stroke_width = 0).stretch(1.5, 1).set_width(0.15)
                q1_triangle.add_updater(
                    lambda m: m.next_to(number_line_q.n2p(q_entropy_chart.distribution_trackers[0].get_value()), DOWN, buff = 0)
                )
                q1_label = Tex("q_1", font_size = 42).set_color(PURE_MAGENTA)
                q1_label.add_updater(lambda m: m.next_to(q1_triangle, DOWN, buff = 0.1))
                number_line_q_group = Group(number_line_q, q1_dot, q1_triangle, q1_label)
                number_line_q_opacity_tracker = ValueTracker(0)
                number_line_q_group.add_updater(lambda m: m.set_opacity(number_line_q_opacity_tracker.get_value()))
                self.add(number_line_q_group)
                anims.append(number_line_q_opacity_tracker.animate.set_value(1))
                anims.append(self.camera.frame.animate.shift(DOWN*0.5))
            if i == 4:
                # Highlight q_2
                anims.append(circ_opacity_tracker_1.animate.set_value(0))

                circ_opacity_tracker_2 = ValueTracker(0)
                circ2 = always_redraw(
                    lambda: Circle(
                        stroke_width = 2,
                        stroke_color = YELLOW,
                        stroke_opacity = circ_opacity_tracker_2.get_value()
                    ).surround(
                        q_entropy_chart.probability_labels[1],
                        buff = 0.2
                    )
                )
                self.add(circ2)
                anims.append(circ_opacity_tracker_2.animate.set_value(1))
            if i == 5:
                # Replace q_2 with 1 - q_1
                anims.append(circ_opacity_tracker_2.animate.set_value(0))

                new_q2_label = always_redraw(
                    lambda: Tex(
                        "1 - q_1", font_size = 42
                    ).set_color(
                        LIGHT_PINK
                    ).move_to(
                        q_entropy_chart.probability_labels[1]
                    ).set_opacity(
                        1 - circ_opacity_tracker_2.get_value()
                    )
                )
                self.add(new_q2_label)
                q_entropy_chart.probability_labels[1].add_updater(lambda m: m.set_opacity(circ_opacity_tracker_2.get_value()))
            if i == 7:
                # Highlight the p distirbution
                rect_opacity_tracker = ValueTracker(0)
                rect = always_redraw(
                    lambda: SurroundingRectangle(
                        qp_cross_entropy_chart.probability_labels,
                        buff = 0.1,
                        stroke_width = 1.5,
                        stroke_color = YELLOW,
                        stroke_opacity = rect_opacity_tracker.get_value()
                    )
                )
                self.add(rect)
                anims.append(rect_opacity_tracker.animate.set_value(1))

                number_line_p = NumberLine(
                    [0, 1, 0.1],
                    include_numbers = True,
                    numbers_to_exclude = [0.1*i for i in range(1, 10)],
                    width = qp_cross_entropy_chart.segments.get_width()
                ).next_to(qp_cross_entropy_chart.bars, DOWN, buff = 1.6)
                self.add(number_line_p)
                p1_dot = Group(TrueDot(), GlowDot()).set_color(GREEN_B)
                p1_dot.add_updater(lambda m: m.move_to(number_line_p.n2p(qp_cross_entropy_chart.distribution_trackers[0].get_value())))
                p1_triangle = Triangle(
                    fill_opacity = 0.7, fill_color = GREEN_B, stroke_width = 0
                ).stretch(1.5, 1).set_width(0.15).flip(axis = RIGHT)
                p1_triangle.add_updater(
                    lambda m: m.next_to(number_line_p.n2p(qp_cross_entropy_chart.distribution_trackers[0].get_value()), UP, buff = 0)
                )
                p1_label = Tex("p_1", font_size = 42).set_color(GREEN_B)
                p1_label.add_updater(lambda m: m.next_to(p1_triangle, UP, buff = 0.1))
                number_line_p_group = Group(number_line_p, p1_dot, p1_triangle, p1_label)
                number_line_p_opacity_tracker = ValueTracker(0)
                number_line_p_group.add_updater(lambda m: m.set_opacity(number_line_p_opacity_tracker.get_value()))
                self.add(number_line_p_group)
                anims.append(number_line_p_opacity_tracker.animate.set_value(1))
            if i == 9:
                # Replace the p_2 labels with 1 - p_1
                anims.append(rect_opacity_tracker.animate.set_value(0))

                dummy_p = Tex("p", font_size = 42)
                new_p2_label = always_redraw(
                    lambda: Tex(
                        "1 - p_1", font_size = 42
                    ).set_color(
                        GREEN_D
                    ).move_to(
                        qp_cross_entropy_chart.probability_labels[1]
                    ).set_opacity(
                        1 - rect_opacity_tracker.get_value()
                    ).scale(qp_cross_entropy_chart.probability_labels[0][0].get_height()/dummy_p.get_height())
                )
                self.add(new_p2_label)
                qp_cross_entropy_chart.probability_labels[1].add_updater(lambda m: m.set_opacity(rect_opacity_tracker.get_value()))
            self.play(*anims)
            if i == 4:
                self.remove(circ)
            if i == 5:
                self.remove(circ2)
            if i == 9:
                self.remove(rect)
        self.wait(1)

        # Switch to graphs
        q1_dot.add_updater(lambda m: self.bring_to_front(m))
        p1_dot.add_updater(lambda m: self.bring_to_front(m))
        max_y = 4
        axes = Axes(
            x_range = [0, 1, 0.1],
            y_range = [0, max_y],
            width = number_line_q.get_width(),
            height = 5
        )
        coordinate_labels_x = axes.add_coordinate_labels(
            x_values = [0, 1],
            y_values = [],
            excluding = []
        ),
        coordinate_labels_y = axes.add_coordinate_labels(
            x_values = [],
            y_values = [i for i in range(max_y + 1)],
            excluding = [],
            direction = LEFT
        )
        axes.align_to(number_line_q, RIGHT).to_edge(DOWN, buff = 1).to_edge(RIGHT, buff = -0.5)
        axes.get_x_axis().set_opacity(0)

        group1 = VGroup(q_entropy_text, q_entropy_formula, q_entropy_chart)
        group2 = VGroup(qp_cross_entropy_text, qp_cross_entropy_formula, qp_cross_entropy_chart)
        number_line_q.generate_target()
        number_line_q.target.match_width(axes.get_x_axis()).move_to(axes.get_x_axis())
        y_axis_label = qp_cross_entropy_formula.copy().next_to(axes.get_y_axis(), LEFT, buff = 0.3)
        self.play(
            self.camera.frame.animate(run_time = 2).scale(1.2).center(),
            AnimationGroup(
                AnimationGroup(
                    FadeOut(
                        VGroup(
                            q_entropy_text,
                            q_entropy_formula,
                            q_entropy_chart,
                            new_q2_label
                        ),
                        shift = UL*2
                    , run_time = 1, suspend_mobject_updating = True),
                    FadeOut(new_q2_label, shift = UL*2, run_time = 1, suspend_mobject_updating = True),
                    AnimationGroup(
                        AnimationGroup(
                            MoveToTarget(number_line_q, run_time = 2, path_arc = PI*0.4),
                            VGroup(
                                qp_cross_entropy_text, qp_cross_entropy_formula, qp_cross_entropy_chart, number_line_p, new_p2_label
                            ).animate(run_time = 2).align_to(axes, DOWN).to_edge(LEFT, buff = -0.5),
                            TransformFromCopy(qp_cross_entropy_formula, y_axis_label, run_time = 1.5)
                        ),
                        FadeIn(axes, shift = DOWN*5 + LEFT, run_time = 1.5)
                    , lag_ratio = 0.2)
                )
            , lag_ratio = 0.8)
        )
        self.remove(new_q2_label)
        self.wait(2)

        # Fix P in place
        pin = SVGMobject("push_pin.svg").rotate(35 * DEG).scale(0.325).set_color(GREY).align_to(
            number_line_p.n2p(qp_cross_entropy_chart.distribution_trackers[0].get_value()), DR
        )
        pin.set_fill([GREY_D, GREY_B], 1)
        self.play(FadeIn(pin, shift=RIGHT * 0.25 + DOWN * 0.125, run_time = 2))
        self.wait(2)

        # Create the graph
        # epsilon = 5.6e-17
        epsilon = 1e-2
        infinity = 10
        def f(q1):
            if q1 < epsilon:
                f_of_epsilon = f(epsilon)
                return ((f_of_epsilon - infinity)/epsilon)*(q1 - epsilon) + f_of_epsilon
            if q1 > 1 - epsilon:
                f_of_1_minus_epsilon = f(1 - epsilon)
                return ((infinity - f_of_1_minus_epsilon)/epsilon)*(q1 - (1 - epsilon)) + f_of_1_minus_epsilon
            return sum([t.get_value()*-math.log2(q) for t, q in zip(qp_cross_entropy_chart.distribution_trackers, [q1, 1 - q1])])

        min_step = 0.001
        max_step = 0.01
        def calculate_step(p):
            # return -4*max_step*(p - 0.5)**2 + max_step
            if 0 <= p < 0.5:
                return 2*max_step*(p - 0.5) + max_step
            return -2*max_step*(p - 0.5) + max_step
        def get_curve():
            step = clip(calculate_step(qp_cross_entropy_chart.distribution_trackers[0].get_value()), min_step, max_step)
            return axes.get_graph(
                f, x_range = [0, 1, step]
            ).set_stroke(
                width = 2, color = PINK
            ).insert_n_curves(100)
        graph = always_redraw(get_curve)
        graph.suspend_updating()
        graph.insert_n_curves(10000)
        f_of_q_dot = Group(TrueDot(radius = 0.1), GlowDot(glow_factor = 1)).set_color(PURE_MAGENTA)
        f_of_q_dot.add_updater(
            lambda m: m.move_to(
                axes.c2p(
                    q_entropy_chart.distribution_trackers[0].get_value(),
                    f(q_entropy_chart.distribution_trackers[0].get_value())
                )
            )
        )
        self.play(
            AnimationGroup(
                ShowCreation(graph, suspend_mobject_updating = True, run_time = 3),
                FadeIn(f_of_q_dot, suspend_mobject_updating = True)
            , lag_ratio = 0.8)
        )
        f_of_q_dot.add_updater(lambda m: self.bring_to_front(m))

        # Change q_1
        for i in range(17):
            if i == 2:
                q1 = 0.2718
            elif i == 3:
                q1 = 0.123
            elif i == 4 or i == 14 or i == 16:
                q1 = qp_cross_entropy_chart.distribution_trackers[0].get_value()
            elif i == 5:
                q1 = 0.789
            else:
                q1 = random.uniform(1/32, 1)
            distribution = [q1, 1 - q1]
            anims = [
                q_entropy_chart.set_distribution(distribution)
            ]
            if i == 1:
                min_value_arrow = Arrow(ORIGIN, DOWN*1.8, thickness = 5).next_to(
                    axes.c2p(
                        qp_cross_entropy_chart.distribution_trackers[0].get_value(),
                        f(qp_cross_entropy_chart.distribution_trackers[0].get_value())
                    ),
                    UP, buff = 0
                )
                min_value_text = Tex(R"\text{min} = \displaystyle\sum_i p_i (-\log_2 p_i)", font_size = 35).next_to(min_value_arrow, UP, buff = -0.1)
                min_value_text["p_i"].set_color(GREEN)
                min_text = min_value_text["min"]
                min_value_text.shift(RIGHT*(min_value_arrow.get_x() - min_text.get_x()))
                anims.append(
                    AnimationGroup(
                        FadeIn(min_value_arrow, shift = DOWN*0.3, run_time = 1.5),
                        Write(min_text, run_time = 1.5)
                    , lag_ratio = 0.2)
                )
            if i == 3:
                number_line_p_group_copy = Group(number_line_p, pin, p1_dot, p1_triangle, p1_label).copy().clear_updaters()
                number_line_p_copy = number_line_p_group_copy[0]
                pin_copy = number_line_p_group_copy[1]
                p1_dot_copy = number_line_p_group_copy[2]
                p1_triangle_copy = number_line_p_group_copy[3]
                p1_label_copy = number_line_p_group_copy[4]
                p1_dot_copy.add_updater(
                    lambda m: m.move_to(number_line_p_copy.n2p(qp_cross_entropy_chart.distribution_trackers[0].get_value()))
                )
                p1_triangle_copy.add_updater(
                    lambda m: m.next_to(number_line_p_copy.n2p(qp_cross_entropy_chart.distribution_trackers[0].get_value()), UP, buff = 0)
                )
                p1_label_copy.add_updater(lambda m: m.next_to(p1_triangle_copy, UP, buff = 0.1))
                number_line_p_group_copy.suspend_updating()

                anims.append(
                    number_line_p_group_copy.animate(
                        run_time = 2, path_arc = PI*0.2
                    ).match_width(
                        number_line_q
                    ).match_x(
                        number_line_q.align_to(number_line_q, DOWN)
                    )
                )
            if i == 4:
                anims.append(FadeOut(number_line_p_group_copy[0]))
            # if i == 16:
            #     anims.append(Write(min_value_text[len("min"):], run_time = 2))

            self.play(*anims, run_time = 2)

        # Clean up
        epsilon_2 = 0.001
        def g(p1):
            if p1 < epsilon_2 or p1 > 1 - epsilon_2:
                return 0
            return sum([p*-math.log2(p) for p in [p1, 1 - p1]])
        min_value_dot = f_of_q_dot.copy().clear_updaters().set_color(GREEN).set_opacity(0.5)
        min_value_dot.add_updater(
            lambda m: m.move_to(
                axes.c2p(
                    qp_cross_entropy_chart.distribution_trackers[0].get_value(),
                    g(qp_cross_entropy_chart.distribution_trackers[0].get_value())
                )
            )
        )
        q1_label.clear_updaters()
        f_of_q_dot.clear_updaters()
        self.remove(q1_dot)
        self.play(
            FadeOut(Group(min_text, min_value_arrow, f_of_q_dot, q1_triangle), suspend_mobject_updating = True),
            FadeOut(Group(pin, pin_copy), shift=LEFT * 0.25 + UP * 0.125),
            q1_label.animate.next_to(axes.get_x_axis()[0], RIGHT),
            FadeIn(min_value_dot, suspend_mobject_updating = True)
        )

        # Trace the minimum value of the graph
        VGroup(qp_cross_entropy_text, qp_cross_entropy_formula).add_updater(lambda m: self.bring_to_front(m))
        def update_qp_cross_entropy_bar_heights(m):
            for t, h in zip(q_entropy_chart.distribution_trackers, m.bar_heights):
                h.set_value(-math.log2(t.get_value()))
        qp_cross_entropy_chart.add_updater(update_qp_cross_entropy_bar_heights)

        number_line_p_group_copy.resume_updating()
        path = TracingTail(min_value_dot, time_traced = 5, stroke_color = GREEN)
        self.add(path)
        self.wait(3)
        graph.resume_updating()
        self.play(
            qp_cross_entropy_chart.set_distribution([1 - epsilon_2, epsilon_2]),
            q_entropy_chart.set_distribution([1 - epsilon_2, epsilon_2])
        , run_time = 4)
        self.play(
            qp_cross_entropy_chart.set_distribution([epsilon_2, 1 - epsilon_2]),
            q_entropy_chart.set_distribution([epsilon_2, 1 - epsilon_2])
        , run_time = 4)
        graph.suspend_updating()

        # Replace the traced path with the full graph of the entropy of P
        p_entropy_graph = axes.get_graph(
            g, x_range = [0, 1, 0.01]
        ).set_stroke(
            width = 2, color = GREEN
        ).insert_n_curves(100)
        # self.wait(1)
        # self.remove(path)
        self.play(FadeIn(p_entropy_graph), run_time = 2)
        # self.wait(1)

        # Move around the distribution some more and write the entropy formula for p next to the new graph
        graph.resume_updating()
        for i in range(20):
            distribution = random_distribution(2, thresh = 1/8)
            anims = [
                AnimationGroup(
                    qp_cross_entropy_chart.set_distribution(distribution),
                    q_entropy_chart.set_distribution(distribution)
                )
            ]
            if i == 0:
                entropy_of_p = min_value_text[len("min="):].scale(0.85).next_to(
                    p_entropy_graph, RIGHT, buff = -1.3
                ).shift(
                    UP*0.9
                ).set_stroke(
                    width = 3, color = BLACK, behind = True
                )
                graph.suspend_updating()
                self.play(Write(entropy_of_p, stroke_color = WHITE))
                graph.resume_updating()
            self.play(*anims, run_time = 4)





class InformationOfRobotInstructions(InteractiveScene):
    def construct(self):
        # Write the information of each instruction
        information_calculations = VGroup(*[
            Tex(R"\text{Information}\Big(\Big) = -\log_2\left(\frac{1}{" + str([2, 4, 8, 8][i]) + R"}\right) = " + str([1, 2, 3, 3][i]))
            for i in range(4)
        ]).arrange(DOWN, buff = 0.3)
        for calculation in information_calculations:
            calculation.align_to(information_calculations[0], LEFT)
        arrows = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)])
        for calculation, arrow in zip(information_calculations, arrows):
            arrow.scale(0.2)
            arrow.next_to(calculation[11], RIGHT)
            calculation[12:].align_to(arrow.get_right() + RIGHT*0.2, LEFT)
            self.play(FadeIn(calculation), FadeIn(arrow))
            self.wait(2)

class InformationEntropyAndCrossEntropy(InteractiveScene):
    def construct(self):
        # Write the definitions of information, entropy, cross entropy, and KL divergence
        definitions = VGroup(
            Tex(
                R"\text{Information}(\text{event}) = -\log_2(p_\text{event})",
                font_size = 60, tex_to_color_map = {"event": YELLOW, R"p_\text{event}": YELLOW}
            ),
            Tex(
                R"\text{Entropy}(P) = \displaystyle\sum_i p_i \cdot -\log_2 p_i",
                font_size = 60, tex_to_color_map = {"P": GREEN, "p_i": GREEN}
            ),
            Tex(
                R"\text{Cross Entropy}(Q, P) = \displaystyle\sum_i p_i \cdot -\log_2 q_i",
                font_size = 60, tex_to_color_map = {"P": GREEN, "Q": PINK, "p_i": GREEN, "q_i": PINK}
            )
        ).arrange_in_grid(n_cols = 1, buff = 0.7)
        for definition in definitions:
            definition.shift(LEFT*(definition["="].get_x() - definitions[0]["="].get_x()))
        definitions.center()
        part1 = definitions[:2]
        part1.save_state()
        part1[1].shift(DOWN*0.5)
        part1.set_y(0)
        self.play(FadeIn(definitions[0]))
        self.wait(1)
        self.play(FadeIn(definitions[1]))
        self.wait(2)

        self.play(
            part1.animate.restore(),
            FadeIn(definitions[2], shift = UP*0.6)
        , run_time = 2)

class CrossEntropyIsAsymmetric(InteractiveScene):
    def construct(self):
        # Show the entropy of Q vs the cross entropy of Q, P
        Q = [0.95, 0.05]
        P = [0.5, 0.5]
        cross_entropy_q_p = round(sum([p*-math.log2(q) for p, q in zip(P, Q)]), 2)
        cross_entropy_p_q = round(sum([q*-math.log2(p) for p, q in zip(P, Q)]), 2)
        Q_entropy_chart = EntropyChart(
            Q,
            event_labels = None,
            probability_labels = VGroup(*[Tex(f"q_{i + 1}", font_size = 42) for i in range(len(Q))]),
            bar_labels = None,
            bar_heights = [-math.log2(q) for q in Q],
            width = 4,
            height = 3,
            include_vertical_axis = False,
            segments_height = 0.3,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).to_corner(UL, buff = 0.5)
        Q_entropy_chart.clear_updaters()

        QP_cross_entropy_chart = EntropyChart(
            P,
            event_labels = None,
            probability_labels = VGroup(*[Tex(f"p_{i + 1}", font_size = 42) for i in range(len(Q))]),
            bar_labels = None,
            bar_heights = [-math.log2(q) for q in Q],
            width = 4,
            height = 3,
            include_vertical_axis = False,
            segments_height = 0.3,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).to_corner(UR, buff = 0.5)
        QP_cross_entropy_chart.clear_updaters()

        self.play(Q_entropy_chart.create())
        arrow1 = Arrow(Q_entropy_chart, QP_cross_entropy_chart, buff = 0.4)
        cross_entropy_qp = Tex(
            R"\displaystyle\sum_i p_i (-\log_2 q_i) \approx " + str(cross_entropy_q_p) + R"\text{ bits}",
            font_size = 30, tex_to_color_map = {"p_i": GREEN, "q_i": PINK}
        ).next_to(arrow1, UP)
        self.play(GrowArrow(arrow1, run_time = 1.5), Write(cross_entropy_qp), QP_cross_entropy_chart.create())
        self.wait(0.5)


        # Show the entropy of P vs the cross entropy of P, Q
        P_entropy_chart = EntropyChart(
            P,
            event_labels = None,
            probability_labels = VGroup(*[Tex(f"p_{i + 1}", font_size = 42) for i in range(len(Q))]),
            bar_labels = None,
            bar_heights = [-math.log2(p) for p in P],
            width = 4,
            height = 3,
            include_vertical_axis = False,
            segments_height = 0.3,
            fill_colors = [GREEN_B, GREEN_D]
        ).to_corner(DR, buff = 0.5)
        P_entropy_chart.clear_updaters()

        PQ_cross_entropy_chart = EntropyChart(
            Q,
            event_labels = None,
            probability_labels = VGroup(*[Tex(f"q_{i + 1}", font_size = 42) for i in range(len(Q))]),
            bar_labels = None,
            bar_heights = [-math.log2(p) for p in P],
            width = 4,
            height = 3,
            include_vertical_axis = False,
            segments_height = 0.3,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK],
            bar_fill_colors = [GREEN_B, GREEN_D]
        ).to_corner(DL, buff = 0.5)
        PQ_cross_entropy_chart.clear_updaters()

        self.play(TransformFromCopy(QP_cross_entropy_chart, P_entropy_chart), run_time = 1.5)
        arrow2 = Arrow(P_entropy_chart, PQ_cross_entropy_chart, buff = 0.4)
        cross_entropy_pq = Tex(
            R"\displaystyle\sum_i q_i (-\log_2 p_i) \approx " + str(cross_entropy_p_q) + R"\text{ bits}",
            font_size = 30, tex_to_color_map = {"p_i": GREEN, "q_i": PINK}
        ).next_to(arrow2, UP)
        self.play(GrowArrow(arrow2, run_time = 1.5), FadeIn(cross_entropy_pq), PQ_cross_entropy_chart.create())

        self.wait(3)




class EntropyChart(VGroup):
    def __init__(
        self,
        initial_distribution,
        event_labels = None,
        probability_labels = "default",
        bar_labels = None,
        bar_heights = None,
        width = 6,
        height = 6,
        segments_height = 0.5,
        stroke_width = 3,
        fit_event_labels_to_height = True,
        include_vertical_axis = True,
        vertical_axis_label_text = R"\text{Information } \\ (-\log_2 p_i \text{ bits})",
        vertical_axis_font_size = 42,
        fill_colors = (BLUE_E, TEAL_E),
        bar_fill_colors = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.stroke_width = stroke_width
        self.distribution_trackers = [ValueTracker(p) for p in initial_distribution]
        self.segments = StackedProbDistribution(
            initial_distribution,
            labels = event_labels,
            width = width,
            height = segments_height,
            fit_labels_to_height = fit_event_labels_to_height,
            fill_colors = (BLUE_E, TEAL_E),
            stroke_width = self.stroke_width
        )
        def update_segments(m):
            m.set_distribution([t.get_value() for t in self.distribution_trackers])
        self.segments.add_updater(update_segments)
        self.add(self.segments)
        if event_labels is not None:
            self.event_labels = self.segments.labels
        else:
            self.event_labels = None

        self.width = width
        self.height = height

        i = 0
        include_labels_every = 1
        self.include_vertical_axis = include_vertical_axis
        if self.include_vertical_axis:
            self.reference_lines = VGroup()
            while i*self.height/self.width <= self.height:
                group = VGroup()
                line = Line(
                    ORIGIN, RIGHT*self.segments.get_width(),
                     stroke_width = 1.3,
                ).align_to(
                    self.segments, UL
                ).shift(
                    UP*i*self.height/self.width
                ).set_opacity(0.4)
                group.add(line)
                label = Tex(str(i), font_size = 20).next_to(line)
                if i % include_labels_every == 0:
                    group.add(label)
                self.reference_lines.add(group)
                
                i += 1
            self.add(self.reference_lines)

        self.bar_heights = [ValueTracker(h) for h in bar_heights] if bar_heights is not None else None
        def get_bars():
            colors = fill_colors if bar_fill_colors is None else bar_fill_colors
            bars = VGroup()
            for (i, segment), t in zip(enumerate(self.segments.bars), self.distribution_trackers):
                bar_height = -math.log2(t.get_value()) if self.bar_heights is None else self.bar_heights[i].get_value()
                bar = Rectangle(
                    width = segment.get_width(),
                    height = bar_height*self.height/self.width,
                    fill_opacity = 0.8,
                    fill_color = interpolate_color(colors[0], colors[1], i/(len(self.distribution_trackers) - 1))
                ).set_stroke(
                    width = self.stroke_width, color = WHITE
                ).next_to(
                    segment, UP, buff = 0
                )
                bar.height = bar_height
                bars.add(bar)
            return bars
        self.bars = always_redraw(get_bars)
        self.add(self.bars)

        if bar_labels is not None:
            self.bar_labels = VGroup(*bar_labels)
            self.add(self.bar_labels)
            for i, label in enumerate(self.bar_labels):
                label.set_color(BLACK)
            def update_labels(m):
                for label, bar in zip(m, self.bars):
                    label.move_to(bar)
            self.bar_labels.add_updater(update_labels)
        else:
            self.bar_labels = None


        if probability_labels == "default":
            self.probability_labels = VGroup(*[Tex(F"p_{i}") for i in range(len(self.distribution_trackers))])
        elif probability_labels is not None:
            self.probability_labels = VGroup(*probability_labels)
        else:
            self.probability_labels = None
        if self.probability_labels is not None:            
            for i, label in enumerate(self.probability_labels):
                label.set_color(interpolate_color(fill_colors[0], fill_colors[1], i/(len(self.distribution_trackers) - 1)))
            def update_labels(m):
                for label, segment in zip(m, self.segments.bars):
                    label.next_to(segment, DOWN).match_y(m[0])
            self.probability_labels.add_updater(update_labels)
            self.add(self.probability_labels)

        if self.include_vertical_axis:
            self.vertical_axis = Line(ORIGIN, UP*self.height).align_to(self.segments.get_corner(UL), DL)
            self.vertical_axis_label = Tex(
                vertical_axis_label_text, font_size = vertical_axis_font_size
            ).next_to(self.vertical_axis, LEFT)
            self.vertical_axis_label["Information"].match_x(self.vertical_axis_label[len("Information"):])
            self.add(self.vertical_axis, self.vertical_axis_label)


    def set_distribution(self, distribution):
        return AnimationGroup(*[
            t.animate.set_value(p)
            for p, t in zip(distribution, self.distribution_trackers)
        ])
        self.segments.set_distribution(distribution)

    def create_bars(self):
        return AnimationGroup(*[
            UpdateFromAlphaFunc(
                bar,
                lambda m, a, t = t, i = i: m.stretch_to_fit_height(
                    max(0.0001, a)*(-math.log2(t.get_value()) if self.bar_heights is None else self.bar_heights[i].get_value())*self.height/self.width,
                    about_point = m.get_bottom()
                ).set_stroke(
                    width = self.stroke_width*a
                ),
                suspend_mobject_updating = True
            )
            for i, (bar, t) in enumerate(zip(self.bars, self.distribution_trackers))
        ], lag_ratio = 0.2)

    def create_reference_lines(self):
        anims = []
        for line in self.reference_lines:
            anim = [ShowCreation(line[0])]
            if len(line) > 1:
                anim.append(Write(line[1]))
            anims.append(AnimationGroup(*anim, lag_ratio = 0.8))
        return AnimationGroup(*anims, lag_ratio = 0.04)

    def create(self):
        anims_1 = []
        anims_2 = []
        for i in range(len(self.segments.bars)):
            fadeins = []
            fadeins.append(FadeIn(self.segments.bars[i]))
            if self.event_labels is not None:
                fadeins.append(FadeIn(self.event_labels[i]))
            if self.probability_labels is not None:
                fadeins.append(FadeIn(self.probability_labels[i]))
            anims_1.append(AnimationGroup(*fadeins))
        anims_2.append(self.create_bars())
        if self.bar_labels is not None:
            anims_2.append(
                AnimationGroup(*[
                    FadeIn(bar_label)
                    for bar_label in self.bar_labels
                ])
            )
        if self.include_vertical_axis:
            anims_2.append(
                AnimationGroup(
                    Write(self.vertical_axis_label),
                    ShowCreation(self.vertical_axis),
                    self.create_reference_lines()
                )
            )
        return AnimationGroup(AnimationGroup(*anims_1), AnimationGroup(*anims_2, lag_ratio = 0.5))


class EntropyChartTest(InteractiveScene):
    def construct(self):
        # Create the entropy chart
        chart = EntropyChart(
            [1/2, 1/4, 1/8, 1/8],
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]),
            # probability_labels = [
            #     Tex(R"\frac{1}{" + ["2", "4", "8", "8"][i] + "}", font_size = 40)
            #     for i in range(4)
            # ],
            width = 5,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        ).to_edge(DOWN, buff = 1)
        self.add(chart)
        self.play(chart.set_distribution([1/4, 1/8, 1/2, 1/8]))


class WhyLogsAreNice(InteractiveScene):
    def construct(self):
        # Write some rules of logs
        equation = Tex(
            R"-\log_2(p_1 \cdot p_2) = -(\log_2 p_1 + \log_2 p_2)",
            font_size = 62,
            tex_to_color_map = {"p_1": BLUE, "p_2": GREEN}
        )
        self.play(Write(equation), run_time = 5)
        self.wait(3)
        self.play(FadeOut(equation))

        p = 0.000000001
        log_p = round(-math.log2(p), 2)
        small_prob = Tex(
            F"p = {p:.9f}", tex_to_color_map = {"p": BLUE}, font_size = 70
        )
        small_prob_log = Tex(
            R"-\log_2 p \approx " + str(log_p), tex_to_color_map = {"p": BLUE}, font_size = 70
        ).match_height(small_prob).move_to(small_prob).shift(DOWN)
        self.play(FadeIn(small_prob))
        self.wait(2)
        small_prob.generate_target()
        small_prob.target.shift(UP*0.5)
        arrow = CurvedArrow(small_prob.target.get_left() + LEFT*0.3 + DOWN*0.2, small_prob_log.get_left() + LEFT*0.3).set_color(YELLOW)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    MoveToTarget(small_prob),
                    FadeIn(small_prob_log, shift = UP)
                , run_time = 2),
                FadeIn(arrow, run_time = 1.5)
            , lag_ratio = 0.7)
        )


class RandomNoiseMeaning(InteractiveScene):
    def construct(self):
        # Demonstrate that "random noise" means a bunch of independent coin tosses for each bit
        message = VGroup(*[
            Integer(0).set_color(YELLOW)
            for _ in range(37)
        ]).arrange(
            buff = 0.08
        ).set_width(
            FRAME_WIDTH*0.9
        )
        self.add(message)

        coins = Group(*[
            Coin(
                color = BLUE_E, tails_color = LIGHT_BROWN, numeric_labels = True
            )
            for b in message
        ]).arrange(buff = 0.2).match_width(message).next_to(message, DOWN)
        for coin in coins:
            if random.random() > 0.5:
                coin.flip()
        self.add(coins)

        for bit, coin in zip(message, coins):
            def update_bit(m, coin = coin):
                m.set_value(1 if coin.is_heads() else 0)
            bit.add_updater(update_bit)

        coin_list = list(coins)
        for _ in range(45):
            random.shuffle(coin_list)
            self.play(
                AnimationGroup(*[
                    coin.animate.flip() if random.random() > 0.5 else Rotate(coin, axis = RIGHT, angle = random.choice([-1, 1])*2*PI)
                    for coin in coin_list
                ], lag_ratio = 0.01)
            )


class Coin(Group):
    def __init__(
        self,
        disk_resolution = (150, 50),
        height = 1,
        depth = 0.1,
        color = GOLD_D,
        tails_color = RED,
        include_labels = True,
        numeric_labels = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        res = disk_resolution
        self.top = Disk3D(resolution=res)
        self.bottom = self.top.copy()
        self.top.shift(OUT)
        self.bottom.shift(IN)
        self.edge = Cylinder(height=2, resolution=(res[1], 2))
        self.add(self.top, self.bottom, self.edge)
        self.rotate(90 * DEGREES, OUT)
        self.set_color(color)
        self.bottom.set_color(tails_color)

        if include_labels:
            chars = "10" if numeric_labels else "HT"
            labels = VGroup(*[TexText(c, depth_test = True) for c in chars])
            for label, vect in zip(labels, [OUT, IN]):
                label.shift(1.02 * vect)
                label.set_height(1.2)
            labels[1].rotate(PI, RIGHT)
            # labels.apply_depth_test()
            labels.set_stroke(width=0)
            self.add(*labels)
            self.labels = labels

        self.set_height(height)
        self.set_depth(depth, stretch=True)

        def update_label_opacities(m):
            if m.is_heads():
                m.labels[0].set_opacity(1)
                m.labels[1].set_opacity(0)
            else:
                m.labels[0].set_opacity(0)
                m.labels[1].set_opacity(1)
        self.add_updater(update_label_opacities)

    def is_heads(self):
        return self.top.get_center()[2] > self.bottom.get_center()[2]

    def flip(self, axis=RIGHT):
        super().flip(axis)
        return self



class AmbientDecodingInstructions(InteractiveScene):
    def construct(self):
        # Add mission_control and the robot
        mission_control = ImageMobject(
            "images/pi_creature_mission_control.png"
        ).set_width(2).to_edge(RIGHT, buff = 0.4)
        robot = ImageMobject(
            "images/lunar_rover_assets/stationary.png"
        ).match_height(mission_control).to_edge(LEFT, buff = 0.4).align_to(mission_control, DOWN)
        self.add(mission_control, robot)

        # Create a stream of instructions flowing towards the bot, and decode them into instructions by chunks of 2
        distribution = [1/2, 1/4, 1/8, 1/8]
        instructions = generate_random_instructions(100, distribution, seed = 7)

        arrows = VGroup(*[
            InstructionArrow([UP, DOWN, LEFT, RIGHT][instructions[i]]).scale(0.2).match_y(robot).set_color(PINK)
            for i in range(len(instructions))
        ]).arrange(buff = 0.5).align_to(mission_control, LEFT)
        self.add(arrows)
        def update_arrows(m):
            for i, arrow in enumerate(m):
                opacity = min(1, max(0, 0.8*(mission_control.get_left()[0] - (arrow.get_x() + 0.2))))
                if arrow.get_x() < robot.get_right()[0]:
                    opacity = min(1, max(0, 1 - 1.2*(robot.get_right()[0] - arrow.get_x())))
                arrow.set_opacity(opacity)
        arrows.add_updater(update_arrows)
        encoding = ["0", "10", "110", "111"]
        bits = VGroup(*[
            Tex(encoding[instruction], font_size = 70).set_color(PINK)
            for instruction in instructions
        ])
        position_tracker = ValueTracker(0)
        interp_x_start = 1
        interp_x_end = 0.6
        for arrow, grouping in zip(arrows, bits):
            grouping.move_to(arrow)

            trans = Transform(arrow, grouping, lag_ratio = 0.8)
            trans.begin()
            interpolation_tracker = ValueTracker(0)
            original_x = arrow.get_x()
            def update_arrow(m, original_x = original_x, interpolation_tracker = interpolation_tracker, trans = trans):
                interpolation_tracker.set_value(max(0, min(1, (interp_x_start - m.get_x())/(interp_x_start - interp_x_end))))
                trans.interpolate_mobject(interpolation_tracker.get_value())
                m.set_x(original_x - position_tracker.get_value())
            arrow.add_updater(update_arrow)
        robot.set_opacity(0)
        mission_control.set_opacity(0)
        self.play(position_tracker.animate(run_time = 80, rate_func = linear).set_value(100))

class InterpolateTest(InteractiveScene):
    def construct(self):
        # Test Mobject interpolation
        mob1 = InstructionArrow(LEFT).scale(0.3).shift(RIGHT*6)
        mob2 = Tex("010").shift(LEFT).move_to(mob1)
        
        self.add(mob1)
        
        trans = Transform(mob1, mob2)
        trans.begin()
        interpolation_tracker = ValueTracker(0)
        position_tracker = ValueTracker(0)
        original_x = mob1.get_x()
        x_start = 1
        x_end = -1
        def update_arrow(m):
            interpolation_tracker.set_value(max(0, min(1, (x_start - m.get_x())/(x_start - x_end))))
            trans.interpolate_mobject(interpolation_tracker.get_value())
            m.set_x(original_x - position_tracker.get_value())
        mob1.add_updater(update_arrow)
        self.play(position_tracker.animate(run_time = 8, rate_func = linear).set_value(10))

class KLDivergencePreview(InteractiveScene):
    def construct(self):
        # Add the charts
        Q = [0.4, 0.1, 0.08, 0.15, 0.27]
        P = [0.1, 0.2, 0.3, 0.35, 0.05]
        # p_cross_entropy_chart = EntropyChart(
        #     P,
        #     event_labels = None,
        #     probability_labels = None,
        #     bar_labels = None,
        #     bar_heights = [-math.log2(q) for q in Q],
        #     width = 5,
        #     height = 3.5,
        #     include_vertical_axis = False,
        #     segments_height = 0.2,
        #     fill_colors = [GREEN_B, GREEN_D],
        #     bar_fill_colors = [RED, LIGHT_PINK]
        # )
        # p_cross_entropy_chart.to_edge(DOWN, buff = 1)
        # p_cross_entropy_chart.add_updater(lambda m: m.set_opacity(0))
        # self.add(p_cross_entropy_chart)

        p_entropy_chart = EntropyChart(
            P,
            event_labels = None,
            probability_labels = VGroup(*[
                Tex(
                    (("p_" + str(i)) if i < len(P) - 2 else R"\ldots" if i == len(P) - 2 else "p_n"),
                    font_size = 40
                )
                for i in range(len(P))
            ]),
            bar_labels = None,
            width = 5,
            height = 3.5,
            include_vertical_axis = False,
            segments_height = 0.2,
            fill_colors = [YELLOW_B, YELLOW_B],
            bar_fill_colors = [YELLOW_B, YELLOW_B]
        )
        p_entropy_chart.to_edge(DOWN, buff = 1).set_stroke(width = 1)
        p_entropy_chart.add_updater(lambda m: m.bars.set_fill(opacity = 0.1).set_stroke(width = 1))
        self.add(p_entropy_chart)

        def get_kl_divergence_bars():
            current_distribution = [t.get_value() for t in p_entropy_chart.distribution_trackers]
            bar_heights = [math.log2(p/q) for p, q in zip(current_distribution, Q)]
            bars = EntropyChart(
                current_distribution,
                event_labels = None,
                probability_labels = None,
                bar_labels = None,
                bar_heights = bar_heights,
                width = 5,
                height = 3.5,
                include_vertical_axis = False,
                segments_height = 0.2
            ).bars.set_stroke(width = 1)
            bars.clear_updaters()
            for bar, height, p_entropy_bar in zip(bars, bar_heights, p_entropy_chart.bars):
                bar.align_to(p_entropy_bar.get_top(), DOWN)
                if height > 0:
                    bar.set_fill(color = GREEN)
                else:
                    bar.set_fill(color = RED)
            return bars
        kl_divergence_chart_bars = always_redraw(get_kl_divergence_bars)
        self.add(kl_divergence_chart_bars)

        # Change the distribution a little
        for _ in range(15):
            self.play(p_entropy_chart.set_distribution(random_distribution(5)), run_time = 2)
            self.wait(1)

class SamplingSymbols(InteractiveScene):
    def construct(self):
        # Add the distribution chart the number line, and the entropy chart
        distribution = [0.1, 0.5, 0.2, 0.05, 0.15]
        distribution_chart = StackedProbDistribution(
            distribution,
            labels = VGroup(*[
                Tex("s_" + str(i + 1)).scale(0.8).set_color(BLACK)
                for i in range(len(distribution))
            ]),
            width = FRAME_WIDTH*0.8,
            height = 0.5,
            fit_labels_to_height = True,
            fill_colors = (BLUE_E, TEAL_E),
            stroke_width = 1.5
        ).shift(UP*1.5)
        number_line = NumberLine(
            (0, 1, 1),
            width = distribution_chart.get_width(),
            include_ticks = True,
            include_numbers = True,
            stroke_width = 1.5
        ).next_to(distribution_chart, DOWN, buff = 0)
        number_line[0].stretch_to_fit_width(distribution_chart.get_width())
        number_line.shift(RIGHT*(distribution_chart.get_x() - number_line[0].get_x()))
        self.add(distribution_chart, number_line)
        triangle = Triangle(
            stroke_width = 0, fill_opacity = 0.8, fill_color = YELLOW
        ).set_width(0.3).stretch(1.2, 1).align_to([0, number_line[0].get_y(), 0], UP)
        x_tracker = ValueTracker(0.314)
        triangle.add_updater(lambda m: m.set_x(number_line.n2p(x_tracker.get_value())[0]))
        self.add(triangle)
        random_value = DecimalNumber(
            num_decimal_places = 3, font_size = 17
        ).set_stroke(
            width = 2, color = BLACK, behind = True, opacity = 0.8
        )
        random_value.add_updater(lambda m: m.set_value(x_tracker.get_value()).next_to(triangle, DOWN, buff = 0.1))
        self.add(random_value)

        entropy_chart = EntropyChart(
            distribution,
            event_labels = VGroup(*[
                Tex("s_" + str(i + 1)).scale(0.8).set_color(BLACK)
                for i in range(len(distribution))
            ]),
            probability_labels = None,
            bar_labels = None,
            width = FRAME_WIDTH*0.8,
            height = 8,
            include_vertical_axis = False,
            segments_height = 0.5,
            fill_colors = [GREEN_B, GREEN_D]
        ).to_edge(DOWN, buff = 0.7)
        self.add(entropy_chart)


        # Sample some instructions at random
        buff = 0.1
        string = VGroup().to_edge(UP, buff = 1.5).align_to(distribution_chart, LEFT).shift(LEFT*buff)
        random.seed(7)
        for _ in range(21):
            x_tracker.set_value(random.random())
            self.wait(0.2)

            symbol = None
            for bar, label in zip(distribution_chart.bars, distribution_chart.labels):
                if bar.get_left()[0] <= triangle.get_x() < bar.get_right()[0]:
                    symbol = label.copy()
                    break
            symbol_move_run_time = 0.01
            self.play(
                Indicate(bar, scale_factor = 1, run_time = 0.3),
                symbol.animate(run_time = symbol_move_run_time).next_to(string, RIGHT, buff = buff).set_color(WHITE)
            )
            string.add(symbol)

            self.wait(0.3)


class PiCreatureSmiles(InteractiveScene):
    def construct(self):
        # Randy smiles
        randy = Randolph("happy", flip_at_start = True)
        self.play(FadeIn(randy, shift = LEFT))
        self.wait(0.5)
        self.play(randy.change("hooray", self.camera.frame.get_corner(UL)))

class RobotDistribution(InteractiveScene):
    def construct(self):
        # Show the main distribution for the robot
        chart = EntropyChart(
            [1/2, 1/4, 1/8, 1/8],
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]),
            probability_labels = [
                Tex(R"\frac{1}{" + ["2", "4", "8", "8"][i] + "}", font_size = 40)
                for i in range(4)
            ],
            bar_labels = None,
            width = 9,
            include_vertical_axis = False,
            segments_height = 0.5,
            fill_colors = [YELLOW_B, YELLOW_D]
        )
        distribution = VGroup(chart.segments, chart.probability_labels)
        self.add(distribution)
        self.wait(2)
        self.play(distribution.animate.scale(0.5).to_corner(UR, buff = 0.6).set_stroke(width = 1))


class MoreComplicatedCrossEntropyExampleAndKLDivergence(InteractiveScene):
    def construct(self):
        # Create the charts
        Q = [0.4, 0.1, 0.08, 0.15, 0.27]
        P = [0.1, 0.2, 0.3, 0.35, 0.05]
        q_entropy_chart = EntropyChart(
            Q,
            event_labels = None,
            probability_labels = VGroup(*[
                Tex(
                    (("q_" + str(i + 1)) if i < len(Q) - 2 else R"\ldots" if i == len(P) - 2 else "q_n"),
                    font_size = 40
                )
                for i in range(len(Q))
            ]),
            bar_labels = None,
            width = 5,
            height = 3.5,
            include_vertical_axis = False,
            segments_height = 0.2,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK],
            stroke_width = 1
        )
        qp_cross_entropy_chart = EntropyChart(
            P,
            event_labels = None,
            probability_labels = VGroup(*[
                Tex(
                    (("p_" + str(i + 1)) if i < len(P) - 2 else R"\ldots" if i == len(P) - 2 else "p_n"),
                    font_size = 40
                )
                for i in range(len(P))
            ]),
            bar_labels = None,
            bar_heights = [-math.log2(q) for q in Q],
            width = 5,
            height = 3.5,
            include_vertical_axis = False,
            segments_height = 0.2,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [PURE_MAGENTA, LIGHT_PINK],
            stroke_width = 1
        )
        VGroup(q_entropy_chart, qp_cross_entropy_chart).arrange(buff = 1.5)

        # Add P
        self.camera.frame.save_state()
        self.camera.frame.match_x(qp_cross_entropy_chart.segments.bars)
        self.play(
            AnimationGroup(*[
                AnimationGroup(GrowFromCenter(segment), FadeIn(p_label))
                for segment, p_label in zip(
                    qp_cross_entropy_chart.segments.bars, qp_cross_entropy_chart.probability_labels
                )
            ], lag_ratio = 0.2, suspend_mobject_updating = True)
        , run_time = 2)

        # Add Q
        self.play(
            self.camera.frame.animate.restore(),
            AnimationGroup(*[
                AnimationGroup(GrowFromCenter(segment), FadeIn(p_label))
                for segment, p_label in zip(
                    q_entropy_chart.segments.bars, q_entropy_chart.probability_labels
                )
            ], lag_ratio = 0.2, suspend_mobject_updating = True)
        , run_time = 1.5)

        # Wiggle around Q
        q_entropy_chart.suspend_updating()
        self.add(q_entropy_chart.segments)
        num_wiggles = 3
        for i in range(num_wiggles):
            self.play(
                q_entropy_chart.segments.animate.set_distribution(
                    random_distribution(5) if i < num_wiggles - 1 else Q
                ).set_stroke(width = 3 if i < num_wiggles - 1 else 1)
            , run_time = 0.35)
        q_entropy_chart.resume_updating()

        # Add the bars and write the cross entropy formula
        cross_entropy_formula = Tex(
            R"\sum_i p_i(-\log_2 q_i) \approx 0.00 \text{bits}", font_size = 40, tex_to_color_map = {"p_i": GREEN, "q_i": PINK}
        ).to_edge(UP, buff = 0.8).match_x(qp_cross_entropy_chart.bars)
        cross_entropy_value = cross_entropy_formula.make_number_changeable("0.00")
        cross_entropy_value.add_updater(
            lambda m: m.set_value(
                sum([
                    p.get_value()*-math.log2(q.get_value())
                    for p, q in zip(qp_cross_entropy_chart.distribution_trackers, q_entropy_chart.distribution_trackers)
                ])
            )
        )
        self.play(q_entropy_chart.create_bars(), qp_cross_entropy_chart.create_bars(), FadeIn(cross_entropy_formula))
        self.add(q_entropy_chart, qp_cross_entropy_chart)

        # Change the distribution Q to be more similar to P
        # Almost_P = [0.06, 0.24, 0.3, 0.3, 0.1]
        # self.play(
        #     q_entropy_chart.set_distribution(Almost_P),
        #     AnimationGroup(*[
        #         h.animate.set_value(-math.log2(q))
        #         for h, q in zip(qp_cross_entropy_chart.bar_heights, Almost_P)
        #     ])
        # , run_time = 2)
        # self.wait(2)

        # Change P
        num_changes = 2
        for i in range(num_changes):
            new_distribution = random_distribution(5) if i < num_changes - 1 else P
            self.play(
                q_entropy_chart.set_distribution(new_distribution),
                AnimationGroup(*[
                    h.animate.set_value(-math.log2(q))
                    for h, q in zip(qp_cross_entropy_chart.bar_heights, new_distribution)
                ])
            , run_time = 1.5)

        # q_i changes to p_i
        p_i = Tex("p_i", font_size = 40).set_color(GREEN).move_to(cross_entropy_formula["q_i"])
        rect = SurroundingRectangle(cross_entropy_formula, stroke_width = 2, stroke_color = YELLOW)
        entropy_of_p = TexText("Entropy of P", font_size = 30).set_color_by_tex("P", GREEN).next_to(rect, DOWN)
        self.play(
            AnimationGroup(
                self.camera.frame.animate.scale(0.75, about_point = self.camera.frame.get_corner(UR)),
                AnimationGroup(
                    FadeOut(cross_entropy_formula["q_i"]),
                    FadeIn(p_i)
                )
            , lag_ratio = 0.5),
            FadeIn(rect), Write(entropy_of_p)
        , run_time = 2.5)

        # Zoom back out and show some more random distributions
        for i in range(8):
            new_distribution = random_distribution(5)
            anims = [
                AnimationGroup(
                    q_entropy_chart.set_distribution(new_distribution),
                    AnimationGroup(*[
                        h.animate.set_value(-math.log2(q))
                        for h, q in zip(qp_cross_entropy_chart.bar_heights, new_distribution)
                    ])
                )
            ]
            if i == 0:
                anims.append(
                    AnimationGroup(
                        self.camera.frame.animate.restore(),
                        FadeOut(VGroup(rect, entropy_of_p, p_i)),
                        FadeIn(cross_entropy_formula["q_i"])
                    )
                )
            self.play(*anims, run_time = 3)
            self.wait(0.75)

        # Transition to KL Divergence
        kl_divergence_formula = Tex(
            R"\left(\sum_i p_i(-\log_2 q_i)\right) - \left(\sum_i p_i(-\log_2 p_i)\right)",
            font_size = 45, tex_to_color_map = {"p_i": GREEN, "q_i": PINK}
        )
        self.play(
            FadeOut(cross_entropy_formula[-9:], run_time = 2),
            FadeOut(VGroup(q_entropy_chart, qp_cross_entropy_chart), shift = DOWN*4, suspend_mobject_updating = True, run_time = 2),
            AnimationGroup(
                TransformMatchingShapes(
                    cross_entropy_formula[R"\sum_i p_i(-\log_2 q_i)"],
                    kl_divergence_formula[R"\left(\sum_i p_i(-\log_2 q_i)\right)"]
                , run_time = 2),
                Write(kl_divergence_formula[R"- \left(\sum_i p_i(-\log_2 p_i)\right)"], run_time = 2)
            , lag_ratio = 0.7)
        )
        self.add(kl_divergence_formula)

        # Put labels
        cross_entropy_label = TexText(
            R"Cross Entropy of \\ Q relative to P",
            tex_to_color_map = {"Q": PINK, "P": GREEN},
            font_size = 30
        ).next_to(
            kl_divergence_formula[R"\left(\sum_i p_i(-\log_2 q_i)\right)"], DOWN, buff = 0.3
        )
        self.play(FadeIn(cross_entropy_label, shift = UP*0.1))
        self.wait(1)
        entropy_label = TexText(
            R"Entropy of P",
            tex_to_color_map = {"Q": PINK, "P": GREEN},
            font_size = 30
        ).next_to(
            kl_divergence_formula[R"\left(\sum_i p_i(-\log_2 p_i)\right)"], DOWN, buff = 0.3
        ).match_y(cross_entropy_label)
        self.play(FadeIn(entropy_label, shift = UP*0.1))

        # Write "KL Divergence"
        kl_divergence_text = TexText(
            "Kullback-Leibler Divergence:",
            font_size = 60,
            tex_to_color_map = {"K": YELLOW, "L": YELLOW}
        ).shift(UP*2)
        self.play(
            AnimationGroup(
                VGroup(kl_divergence_formula, cross_entropy_label, entropy_label).animate.shift(DOWN*0.5),
                Write(kl_divergence_text, run_time = 2)
            , lag_ratio = 0.8)
        )
        self.wait(0.5)
        kl_divergence_text_shortened = TexText("KL Divergence:").match_height(kl_divergence_text).move_to(kl_divergence_text)
        kl_divergence_text_shortened[:2].set_color(YELLOW)
        self.play(
            TransformMatchingShapes(kl_divergence_text["Kullback-Leibler"], kl_divergence_text_shortened["KL"]),
            TransformMatchingShapes(kl_divergence_text["Divergence:"], kl_divergence_text_shortened["Divergence:"])
        , run_time = 1.3)
        self.wait(2)


class RobotNewOptimalCode(InteractiveScene):
    def construct(self):
        # Add the chart
        encoding = ["111", "110", "10", "0"]
        second_distribution = [1/8, 1/8, 1/4, 1/2]
        second_distribution_chart = EntropyChart(
            second_distribution,
            event_labels = VGroup(*[InstructionArrow([UP, DOWN, LEFT, RIGHT][i]) for i in range(4)]),
            probability_labels = [
                Tex(R"\frac{1}{" + ["8", "8", "4", "2"][i] + "}", font_size = 65)
                for i in range(4)
            ],
            bar_labels = [
                Tex(encoding[i], font_size = 57)
                for i in range(4)
            ],
            width = 12*1.12,
            height = 4.5*1.12,
            include_vertical_axis = False,
            segments_height = 1,
            fill_colors = [GREEN_B, GREEN_D]
        ).scale(0.3)
        second_distribution_chart.update()
        second_distribution_chart.clear_updaters()
        second_distribution_chart.bar_labels.set_color(WHITE)
        self.play(second_distribution_chart.create())

class KLDivergenceDemo(InteractiveScene):
    def construct(self):
        # Add the charts
        Q = [0.4, 0.1, 0.08, 0.15, 0.27]
        P = [0.1, 0.2, 0.3, 0.35, 0.05]

        p_entropy_chart = EntropyChart(
            P,
            event_labels = None,
            probability_labels = VGroup(*[
                Tex(
                    (("p_" + str(i + 1)) if i < len(P) - 2 else R"\ldots" if i == len(P) - 2 else "p_n"),
                    font_size = 40
                )
                for i in range(len(P))
            ]),
            bar_labels = None,
            width = 5,
            height = 3.5,
            include_vertical_axis = False,
            segments_height = 0.2,
            fill_colors = [GREEN_B, GREEN_D],
            bar_fill_colors = [YELLOW_B, YELLOW_B]
        )
        p_entropy_chart.set_stroke(width = 1)
        p_entropy_chart.add_updater(lambda m: m.bars.set_fill(opacity = 0.1).set_stroke(width = 1))
        self.add(p_entropy_chart)

        q_entropy_chart = EntropyChart(
            Q,
            event_labels = None,
            probability_labels = VGroup(*[
                Tex(
                    (("q_" + str(i + 1)) if i < len(P) - 2 else R"\ldots" if i == len(P) - 2 else "q_n"),
                    font_size = 40
                )
                for i in range(len(P))
            ]),
            bar_labels = None,
            width = 5,
            height = 3.5,
            include_vertical_axis = False,
            segments_height = 0.2,
            fill_colors = [PURE_MAGENTA, LIGHT_PINK]
        )
        q_entropy_chart.set_stroke(width = 1)
        q_entropy_chart.add_updater(lambda m: m.bars.set_stroke(width = 1))
        self.add(q_entropy_chart)
        VGroup(q_entropy_chart, p_entropy_chart).arrange(buff = 2)
        q_entropy_chart.to_edge(DOWN, buff = 0.7)
        p_entropy_chart.to_edge(DOWN, buff = 0.7)

        def get_kl_divergence_bars():
            current_distribution = [t.get_value() for t in p_entropy_chart.distribution_trackers]
            bar_heights = [math.log2(p/q.get_value()) for p, q in zip(current_distribution, q_entropy_chart.distribution_trackers)]
            bars = EntropyChart(
                current_distribution,
                event_labels = None,
                probability_labels = None,
                bar_labels = None,
                bar_heights = bar_heights,
                width = 5,
                height = 3.5,
                include_vertical_axis = False,
                segments_height = 0.2
            ).bars.set_stroke(width = 1)
            bars.clear_updaters()
            for bar, height, p_entropy_bar in zip(bars, bar_heights, p_entropy_chart.bars):
                bar.match_x(p_entropy_bar).align_to(p_entropy_bar.get_top(), DOWN)
                if height > 0:
                    bar.set_fill(color = PURE_GREEN)
                else:
                    bar.set_fill(color = PURE_RED)
            return bars
        kl_divergence_chart_bars = always_redraw(get_kl_divergence_bars)
        self.add(kl_divergence_chart_bars)

        # Add the equation
        kl_divergence_formula = Tex(
            R"\begin{gathered}\text{KL Divergence of Q relative to P:} \\ \left(\sum_i p_i(-\log_2 q_i)\right) - \left(\sum_i p_i(-\log_2 p_i)\right)\end{gathered}",
            font_size = 30, tex_to_color_map = {"Q": PINK, "P": GREEN, "p_i": GREEN, "q_i": PINK}
        ).next_to(p_entropy_chart, UP, buff = 1.2)
        self.add(kl_divergence_formula)

        # Change the distribution a little
        for i in range(15):
            new_distribution = random_distribution(5) if i != 1 else P
            anims = [q_entropy_chart.set_distribution(new_distribution)]
            if i == 3:
                kl_divergence_formula.save_state()
                kl_divergence_formula_2 = Tex(
                    R"\begin{gathered}\text{KL Divergence of P relative to Q:} \\ \left(\sum_i q_i(-\log_2 p_i)\right) - \left(\sum_i q_i(-\log_2 q_i)\right)\end{gathered}",
                    font_size = 30, tex_to_color_map = {"Q": PINK, "P": GREEN, "p_i": GREEN, "q_i": PINK}
                ).next_to(p_entropy_chart, UP, buff = 1.2)
                kl_divergence_formula.generate_target()
                VGroup(kl_divergence_formula.target, kl_divergence_formula_2).scale(1.05).arrange(buff = 1.3)
                rect = SurroundingRectangle(kl_divergence_formula_2, buff = 0.2, stroke_width = 3, stroke_color = YELLOW)
                p_entropy_chart.suspend_updating()
                charts_group = VGroup(q_entropy_chart, p_entropy_chart, kl_divergence_chart_bars)
                charts_group.save_state()
                charts_group.suspend_updating()
                self.play(
                    charts_group.animate(run_time = 1.5).shift(DL*2).set_opacity(0),
                    AnimationGroup(
                        MoveToTarget(kl_divergence_formula),
                        Write(kl_divergence_formula_2, run_time = 2),
                        FadeIn(rect)
                    , lag_ratio = 0.6, run_time = 4)
                )
                self.wait(1)
                not_equal = Tex(R"\neq", font_size = 50).move_to(VGroup(kl_divergence_formula, kl_divergence_formula_2))
                self.play(FadeOut(rect), FadeIn(not_equal))
                self.wait(2)
                self.play(
                    FadeOut(VGroup(not_equal, kl_divergence_formula_2), shift = UP*3 + RIGHT*5),
                    kl_divergence_formula.animate.restore(),
                    charts_group.animate(run_time = 2).shift(UR*2).restore()
                , run_time = 2)
                charts_group.resume_updating()
            if i == 5:
                kl_divergence_formula_compact = Tex(
                    R"= \displaystyle\sum_i p_i \left(\log_2 \frac{p_i}{q_i}\right)",
                    font_size = 30, tex_to_color_map = {"p_i": GREEN, "q_i": PINK}
                ).next_to(kl_divergence_formula, DOWN, buff = 0.25).align_to(kl_divergence_formula, LEFT)
                anims.append(Write(kl_divergence_formula_compact))
            if i == 8:
                rect = SurroundingRectangle(
                    VGroup(
                        kl_divergence_formula[R"\left(\sum_i p_i(-\log_2 q_i)\right) - \left(\sum_i p_i(-\log_2 p_i)\right)"],
                        kl_divergence_formula_compact
                    ), buff = 0.1, stroke_width = 3, stroke_color = YELLOW
                )
                anims.append(FadeIn(rect))
            if i == 10:
                anims.append(FadeOut(VGroup(kl_divergence_formula_compact, rect)))
            if i == 11:
                anims.append(FancyCircumscribe(VGroup(p_entropy_chart, kl_divergence_chart_bars)))
            self.play(*anims, run_time = 2)
            self.wait(1)


class DistanceBetweenDistributions(InteractiveScene):
    def construct(self):
        # Add the two distributions
        distribution = random_distribution(7)
        Q = StackedProbDistribution(
            distribution,
            labels = None,
            width = 8,
            height = 0.4,
            fill_colors = (LIGHT_PINK, PURE_MAGENTA),
            stroke_width = 3
        ).to_edge(LEFT, buff = 1.5).shift(UP*1.2)
        probability_labels_1 = VGroup(*[
            Tex(
                (("q_" + str(i + 1)) if i < len(Q.bars) - 2 else R"\ldots" if i == len(Q.bars) - 2 else "q_n"),
                font_size = 35
            ).set_color(interpolate_color(LIGHT_PINK, PURE_MAGENTA, i//(len(Q.bars) - 1)))
            for i in range(len(Q.bars))
        ])
        for label, segment in zip(probability_labels_1, Q.bars):
            label.add_updater(lambda m, segment = segment: m.next_to(segment, DOWN))

        P = StackedProbDistribution(
            distribution,
            labels = None,
            width = 8,
            height = 0.4,
            fill_colors = (GREEN_B, GREEN_D),
            stroke_width = 3
        ).to_edge(LEFT, buff = 1.5).shift(DOWN*1.2)
        probability_labels_2 = VGroup(*[
            Tex(
                (("p_" + str(i + 1)) if i < len(P.bars) - 2 else R"\ldots" if i == len(Q.bars) - 2 else "p_n"),
                font_size = 35
            ).set_color(interpolate_color(GREEN_B, GREEN_D, i//(len(Q.bars) - 1)))
            for i in range(len(P.bars))
        ])
        for label, segment in zip(probability_labels_2, P.bars):
            label.add_updater(lambda m, segment = segment: m.next_to(segment, DOWN))

        self.add(Q, probability_labels_1, P, probability_labels_2)

        # Add Q and P symbols
        def get_kl_divergence(Q, P):
            Q_total_width = Q.get_width()
            Q_current_distribution = [bar.get_width()/Q_total_width for bar in Q.bars]
            P_total_width = P.get_width()
            P_current_distribution = [bar.get_width()/P_total_width for bar in P.bars]
            return sum([p*math.log2(p/q) for q, p in zip(Q_current_distribution, P_current_distribution)])
        multiplier = 30
        Q_symbol = VGroup(Tex("Q", font_size = 60), Tex("-").stretch(0.5, 0)).arrange(RIGHT).set_color(PINK)
        P_symbol = VGroup(Tex("P", font_size = 60), Tex("-").stretch(0.5, 0)).arrange(LEFT).set_color(GREEN)
        VGroup(Q_symbol, P_symbol).arrange(buff = 0).to_edge(RIGHT, buff = 2)
        self.add(Q_symbol, P_symbol)

        # Add Dashed lines
        dashed_lines = VGroup(*[
            DashedLine(P.bars.get_bottom(), Q.bars.get_top())
            for _ in range(len(distribution) + 1)
        ]).set_color(GREY).set_opacity(0.6).align_to(P.bars, DOWN)
        for line, bar in zip(dashed_lines, P.bars):
            line.set_x(bar.get_left()[0])
        dashed_lines[-1].set_x(P.bars[-1].get_right()[0])
        self.add(dashed_lines)

        # Change Q to be closer / further away from P
        for _ in range(15):
            Q.generate_target()
            Q.target.set_distribution(random_distribution(7, thresh = 1/10))
            self.play(
                MoveToTarget(Q),
                Q_symbol.animate.set_y(multiplier*get_kl_divergence(Q.target, P)),
                P_symbol.animate.set_y(-multiplier*get_kl_divergence(Q.target, P))
            , run_time = 1.5)
            self.wait(0.5)

class WaysToMeasureProbabilityDistributionDifferences(InteractiveScene):
    def construct(self):
        # Pi creature is confused thinking about many different possible formulas for measuring sorts of "distance" between distributions
        randy = Randolph(flip_at_start = True).to_edge(RIGHT, buff = 1).look(LEFT)
        self.add(randy)
        self.wait(1)

        # Add the list of formulas
        formulas = BulletedList(
            R"L1: $\sum_i |p_i - q_i|$",
            R"L2: $\sum_i (p_i - q_i)^2$",
            R"Bhattacharyya: $-\ln \sum_i \sqrt{p_i q_i}$",
            R"Hellinger: $\frac{1}{2} \sum_i \left(\sqrt{p_i} - \sqrt{q_i}\right)^2$",
            R"Cross Entropy: $\sum_i p_i (-\log_2 q_i)$",
            R"KL Divergence: $\sum_i p_i \left(\log_2\frac{p_i}{q_i}\right)$",
            font_size = 45,
            tex_to_color_map = {"p_i": GREEN, "q_i": PINK}
        ).to_edge(LEFT, buff = 1)
        self.play(
            AnimationGroup(
                AnimationGroup(*[FadeIn(formula, shift = DOWN*0.4) for formula in formulas], lag_ratio = 0.2, run_time = 2.5),
                randy.change("confused")
            , lag_ratio = 0.2)
        )
        self.wait(1)
        self.play(Blink(randy))
        self.wait(1)

        self.play(randy.change("pondering", formulas[-2:]), FancyCircumscribe(formulas[-2:], run_time = 5))

        # Focus on Cross Entropy and KL Divergence
        self.play(
            AnimationGroup(
                AnimationGroup(*[
                    FadeOut(formula, shift = UP)
                    for formula in formulas[:-2]
                ], lag_ratio = 0.1),
                AnimationGroup(
                    formulas[-2:].animate.scale(1.1).set_y(0).to_edge(LEFT, buff = 1),
                    randy.change("thinking", LEFT)
                )
            , lag_ratio = 0.32)
        , run_time = 2)
        self.play(Blink(randy))
        self.wait(2)