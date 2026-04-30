from manim_imports_ext import *
from custom.characters import pi_creature
import math
import random


class Robot(Group):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.face = SVGMobject("robot").set_color(TEAL)
        self.left_eye = self.face[0]
        self.right_eye = self.face[1]
        self.mouth = self.face[2]
        self.head = self.face[3]
        self.left_eye.set_color([LIGHT_BROWN])
        self.head_background = Rectangle(
            width = 2, height = 1.5, fill_opacity = 1, fill_color = BLACK, stroke_width = 0
        ).round_corners(
            0.3*self.head.get_width()
        ).match_width(
            self.head
        ).scale(
            0.8
        ).align_to(
            self.head, DOWN
        ).shift(
            UP*0.05*self.head.get_width()
        )
        self.add(self.head_background)
        self.add(self.face)

        self.blinker = GlowDot().set_width(
            self.head.get_width()*0.6
        ).set_color(
            YELLOW
        ).move_to(
            self.head.get_top() + DOWN*0.08*self.head.get_width()
        )
        self.blinker_opacity_tracker = ValueTracker(0)
        self.blinker.add_updater(lambda m: m.set_opacity(self.blinker_opacity_tracker.get_value()))
        self.add(self.blinker)

        self.move_amount = 0.5

    def create(self):
        return AnimationGroup(
            FadeIn(self.head_background),
            *[DrawBorderThenFill(part) for part in self.face]
        , lag_ratio = 0.1)

    def blink_antenna(self):
        return UpdateFromAlphaFunc(self.blinker_opacity_tracker, lambda m, a: m.set_value(1 - (2*a - 1)**2))

    def execute_instruction(self, instruction_index):
        glowdot = GlowDot().set_color(YELLOW).move_to(self.head.get_top())
        direction = [UP, DOWN, LEFT, RIGHT][instruction_index]
        return AnimationGroup(
            self.blink_antenna(),
            self.animate.shift(direction*self.move_amount*self.head.get_width())
        , lag_ratio = 0.3)


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


def generate_random_instructions(n, distribution):
    instructions = []
    for _ in range(n):
        x = random.random()
        for i in range(len(distribution)):
            if x < sum(distribution[:i + 1]):
                instructions.append(i)
                break
    return instructions

class RobotEncoding(InteractiveScene):
    def construct(self):
        # Add the robot
        robot = Robot()
        self.play(robot.create(), run_time = 2)

        # Add the surface of the far away moon
        moon_surface = ImageMobject("images/far_away_moon.png").get_grid(20, 20, buff = 0).scale(5).set_opacity(0.4)
        self.bring_to_back(moon_surface)
        self.play(FadeIn(moon_surface), run_time = 2)

        # The robot moves around a bit
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
                    robot.head_background, m.direction, buff = 0.8 if not (m.direction == UP).all() else 1.2
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
        tail = TracingTail(robot.get_bottom, stroke_color = TEAL, time_traced = 5, stroke_width=5)
        self.add(tail)
        instructions = generate_random_instructions(200, distribution)
        arrow_draw_iter = 10
        zoom_out_iter = arrow_draw_iter + 8
        fractions_draw_iter = arrow_draw_iter + 85
        iters_to_draw_fractions = 60
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
                self.gravitate_camera_towards(robot, target_zoom_level = 0.25, camera_gravity_constant = 0.01)
                self.bring_to_back(tail)
                tail.add_updater(lambda m: m.set_stroke(width = 3))
            if i > zoom_out_iter - 3:
                run_time = max(0.15, smooth(1 - 0.1*(i - (zoom_out_iter - 3))))
            self.play(AnimationGroup(*anims, run_time = run_time))


        return


        # for instruction in instruction_set:
        #     instruction.align_to(instruction_set[0], LEFT)
        robot.generate_target()
        Group(instruction_set, robot.target).arrange(buff = 1)
        self.play(
            AnimationGroup(*[
                FadeIn(instruction, shift = RIGHT)
                for instruction in instruction_set
            ], lag_ratio = 0.1),
            MoveToTarget(robot)
        , run_time = 1.7)

        # Move the instructions to Earth and the Robot to the Moon
        self.play(instruction_set.animate.scale(0.9).to_edge(LEFT, buff = 2.5), robot.animate.scale(0.6).to_edge(RIGHT, buff = 2.5))

        # Send some instructions
        def send_instruction(instruction_index, run_time = 1.5):
            copy = instruction_set[instruction_index].copy()
            self.play(
                AnimationGroup(
                    Succession(
                        copy.animate(path_arc = -PI*0.3).move_to(robot.blinker).scale(0),
                        FadeOut(copy)
                    ),
                    robot.execute_instruction(instruction_index)
                , lag_ratio = 0.4)
            , run_time = run_time)
            self.remove(copy)
        instructions = [0, 0, 3, 1, 0]
        for instruction in instructions:
            send_instruction(instruction, run_time = 1.5)

        # Send many instructions with a skewed distribution (UP: 50%, DOWN: 25%, LEFT: %12.5, RIGHT: 12.5%)
        width_pct = instruction_set.get_width()/self.camera.frame.get_width()
        x_pct = (instruction_set.get_left()[0] - self.camera.frame.get_left()[0])/self.camera.frame.get_width()
        y_pct = (instruction_set.get_bottom()[1] - self.camera.frame.get_bottom()[1])/self.camera.frame.get_height()
        def reposition_instructions_set(m):
            m.set_width(self.camera.frame.get_width()*width_pct)
            m.align_to([
                self.camera.frame.get_left()[0] + self.camera.frame.get_width()*x_pct,
                self.camera.frame.get_bottom()[1] + self.camera.frame.get_height()*y_pct,
                0
            ], DL)
        instruction_set.add_updater(reposition_instructions_set)

        self.gravitate_camera_towards(
            robot,
            x_pct = 0.75,
            y_pct = 0.5,
            target_zoom_level = 1.2
        )


        instructions = generate_random_instructions(100, distribution)
        for instruction in instructions:
            send_instruction(instruction, run_time = 0.2)

    def gravitate_camera_towards(
        self,
        mobject,
        x_pct=0.5,
        y_pct=0.5,
        target_zoom_level = None,
        friction=0.05,
        gravity_constant=0.01,
        camera_gravity_constant = 0.03
    ):
        frame = self.camera.frame
        frame.clear_updaters()
        frame.velocity = np.zeros(3)
        frame.width_velocity = 0

        def update_camera(f, dt):
            target_center = mobject.get_center()
            
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
                width_acceleration = width_diff * camera_gravity_constant
                f.width_velocity += width_acceleration
                f.width_velocity *= (1 - friction)
                
                new_width = f.get_width() + f.width_velocity * dt
                f.set_width(new_width)

        frame.add_updater(update_camera)

class InstructionsToRobot(InteractiveScene):
    def construct(self):
        # Add randy and the robot
        randy = Randolph().set_width(2).to_edge(LEFT, buff = 1).to_edge(DOWN, buff = 1)
        robot = Robot().set_width(2).to_edge(RIGHT, buff = 1).to_edge(DOWN, buff = 1)
        self.add(randy, robot)
        self.wait(1)
        self.play(Blink(randy))

        # Create a stream of bits flowing towards the bot
        distribution = [1/2, 1/4, 1/8, 1/8]
        instructions = generate_random_instructions(100, distribution)
        bit_string = ""
        for instruction in instructions:
            bit_string += f"{instruction:02b}"
        bit_buff = 0.15
        bits = VGroup(*[
            Tex(bit_string[i], font_size = 60)
            for i in range(len(bit_string))
        ]).arrange(buff = 0.15).set_color(YELLOW).match_y(robot.head_background).align_to(randy, RIGHT)
        def update_bits(m):
            for bit in m:
                bit.set_opacity(min(1, max(0, 0.5*(bit.get_x() - randy.get_right()[0]))))
                if bit.get_x() > robot.get_x():
                    bit.set_opacity(0)
            self.bring_to_front(robot)
        bits.add_updater(update_bits)

        rects = VGroup(*[
            SurroundingRectangle(bits[i:i + 2], stroke_width = 2, stroke_color = WHITE, buff = bit_buff*0.5)
            for i in range(0, len(bit_string), 2)
        ])
        rect_opacity_tracker = ValueTracker(0)
        def update_rects(m):
            for i, rect in enumerate(m):
                target_bits = bits[2*i:2*(i + 1)]
                rect.match_x(target_bits)
                rect.set_stroke(
                    opacity = rect_opacity_tracker.get_value()*min(1, max(0, 0.5 * (rect.get_x() - 0.2 - randy.get_right()[0])))
                )
                if rect.get_x() > robot.get_x():
                    rect.set_stroke(opacity = 0)
        rects.add_updater(update_rects)

        arrows = VGroup(*[
            InstructionArrow([UP, DOWN, LEFT, RIGHT][instructions[i]]).scale(0.11).move_to(rects[i]).shift(UP*0.62)
            for i in range(len(instructions))
        ])
        arrows_opacity_tracker = ValueTracker(0)
        def update_arrows(m):
            for i, arrow in enumerate(m):
                target_bits = bits[2*i:2*(i + 1)]
                arrow.match_x(target_bits)
                opacity = min(1, max(0, 0.5*(arrow.get_x() - 0.2 - randy.get_right()[0])))
                if arrow.get_x() > robot.get_left()[0]:
                    opacity = min(1, max(0, 1 - 1.2*(arrow.get_x() - robot.get_left()[0])))
                arrow.set_opacity(arrows_opacity_tracker.get_value()*opacity)
        arrows.add_updater(update_arrows)

        self.add(bits, rects, arrows)

        self.play(
            AnimationGroup(
                bits.animate(run_time = 15, rate_func = linear).shift(RIGHT*15),
                AnimationGroup(
                    Blink(randy),
                    Succession(*[
                        robot.blink_antenna()
                        for _ in range(len(instructions))
                    ]),
                    AnimationGroup(
                        rect_opacity_tracker.animate.set_value(1),
                        arrows_opacity_tracker.animate.set_value(1)
                    , lag_ratio = 1.2),
                )
            , lag_ratio = 0.52)
        )