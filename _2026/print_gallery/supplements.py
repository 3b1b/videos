from manim_imports_ext import *


class DistributeZoomOverlay(InteractiveScene):
    def construct(self):
        # Add frames
        frames = Square(side_length=6).replicate(2)
        frames.arrange(RIGHT, buff=1.5)
        frames.set_stroke(WHITE, 2)
        left_frame, right_frame = frames
        left_frame.set_stroke(BLACK)
        right_frame.set_stroke(opacity=0)

        self.add(left_frame)

        # Add zoom arrows
        small_left = left_frame.copy()
        small_left.scale(1 / 16)
        corner_arrows = VGroup(
            Arrow(left_frame.get_corner(vect), left_frame.get_center(), buff=0.3, thickness=8)
            for vect in compass_directions(4, DL)
        )
        corner_arrows.set_color(TEAL)
        zoom_label = Tex(R"\times 16", font_size=72)
        zoom_label.move_to(corner_arrows[:2])

        self.play(
            *map(GrowArrow, corner_arrows),
            TransformFromCopy(left_frame, small_left),
            Write(zoom_label)
        )
        self.wait()

        # Distributed
        corner_squares = Square(side_length=2).replicate(4)
        for corner, square in zip(compass_directions(4, DL), corner_squares):
            square.move_to(right_frame, corner)

        rot_arrows = VGroup(
            Arrow(s2, s1, buff=0.1, thickness=5, fill_color=TEAL)
            for s1, s2 in adjacent_pairs(corner_squares)
        )

        two_x_labels = VGroup(
            Tex(R"\times 2").next_to(arrow, np.round(rotate_vector(arrow.get_vector(), 90 * DEG)), buff=0)
            for arrow in rot_arrows
        )

        corner_arrow_ghosts = corner_arrows.copy().set_fill(opacity=0.35)
        zoom_label_ghost = zoom_label.copy().set_fill(opacity=0.35)

        self.remove(zoom_label)
        self.play(
            small_left.animate.set_stroke(opacity=0.35),
            FadeIn(right_frame),
            ReplacementTransform(corner_arrows, rot_arrows),
            *(
                FadeTransform(zoom_label.copy(), two_x_label)
                for two_x_label in two_x_labels
            ),
            run_time=2
        )
        self.wait()