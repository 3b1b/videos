from manim_imports_ext import *
from scipy.spatial.transform import Rotation, Slerp


class Pegs(InteractiveScene):
    samples = 4

    def construct(self):
        frame = self.frame

        # Create the peg board and the pegs
        peg_board = self.get_peg_board(radius=30)
        self.add(peg_board)

        pegs = Group(
            self.get_peg(0, 1),
            self.get_peg(0, 0),
            self.get_peg(1, 0)
        ).set_z_index(5)
        for peg in pegs:
            peg.save_state()

        circles = VGroup(*[
            Circle(radius=0.08, fill_opacity=1, fill_color=BLACK, stroke_width=3, stroke_color=TEAL).move_to([i % 2, i // 2, 0])
            for i in range(4)
        ]).shift(OUT * 0.01)
        circles.set_z_index(1)
        self.add(circles)
        self.add(circles)

        # Introduce pegs
        square = Square(side_length=1, stroke_width=1)
        square.move_to(ORIGIN, DL).shift(OUT * 0.02)
        square.set_stroke(TEAL, 5)
        square.apply_depth_test()
        square.set_z_index(0)

        self.play(
            frame.animate(run_time=5).reorient(-25, 49, 0, (0.49, 0.14, 0.05), 6.96),
            LaggedStartMap(FadeIn, pegs, shift=IN * 0.5, lag_ratio=0.2, time_span=(2.5, 4)),
            ShowCreation(square, time_span=(3.5, 5))
        )
        frame.save_state()

        # Do a jump
        peg1, peg2, peg3 = pegs

        self.set_camera_target_position(24, 52, 0, (1.13, 0.46, 0.20), 6.02, drift_time=10)
        self.wait(4)
        self.play(self.illustrated_peg_jump(peg2, peg3, run_time=6))
        self.wait(2)

        # Have two pegs walk down a line
        self.set_camera_target_position(17, 48, 0, (-2.42, 1.35, -1.54), 11.95, drift_time=7)
        jumps = []
        for i in range(5):
            pegs_pair = [peg2, peg3] if i % 2 == 0 else [peg3, peg2]
            self.play(self.illustrated_peg_jump(*pegs_pair, run_time=1.5))
        self.wait(2)

        # Show diagonal jumps
        self.play(
            frame.animate(run_time=3).reorient(-19, 47, 0, (-2.71, 1.47, -1.52), 14.85),
            self.illustrated_peg_jump(peg1, peg3, run_time=5)
        )
        self.play(
            frame.animate(run_time=3).reorient(40, 48, 0, (-3.54, 0.91, -0.76), 9.61),
            self.illustrated_peg_jump(peg1, peg2, run_time=5)
        )

        # Move the pegs back to their initial positions
        self.play(
            frame.animate.restore(),
            AnimationGroup(*[peg.animate(path_arc=-PI, path_arc_axis=DOWN).restore() for peg in pegs], lag_ratio=0.2), run_time=2)
        self.wait(1)

        # Do a bunch of random jumps
        random.seed(2)
        num_jumps = 20
        self.set_camera_target_position(42, 59, 0, (0.49, 0.14, 0.05), 6.96, drift_time=num_jumps * 1.2)
        previous_jump = None
        for i in range(num_jumps):
            first_peg, second_peg = self.choose_weighted_jump(pegs, exclude_jump=previous_jump)
            anims = [self.illustrated_peg_jump(first_peg, second_peg, run_time=1, time_traced=1)]
            if i == 4:
                anims.append(circles[3].animate.set_stroke(color=RED, width=6))
            self.play(*anims)
            previous_jump = (first_peg, second_peg)

    def illustrated_peg_jump(self, peg1, peg2, color=YELLOW, run_time=3, time_traced=4):
        line = DashedLine(peg2.get_center(), peg1.get_center(), dash_length=0.025)
        line.set_z(0.03).set_z_index(3)
        line.set_color(color)
        line.apply_depth_test()
        moving_line = line.copy().set_opacity(0)
        self.add(moving_line)
        traced_path = TracingTail(moving_line.get_end, stroke_color=color, time_traced=time_traced).set_z_index(4)
        self.add(traced_path)

        axis = np.cross(peg1.get_center() - peg2.get_center(), OUT)

        return Succession(
            LaggedStart(
                moving_line.animate.set_opacity(1),
                Rotate(moving_line, PI, axis=axis, run_time=2, about_point=line.get_start()),
                self.simple_peg_jump(peg1, peg2),
                lag_ratio=0.3
            ),
            FadeOut(VGroup(line, moving_line), run_time=0.4), run_time=run_time)

    def simple_peg_jump(self, peg1, peg2, height=0.5, run_time=2):
        center = peg2.get_center()
        radius_vect = peg1.get_center() - center
        semi_circ = ParametricCurve(
            lambda t: center + math.cos(t) * radius_vect + math.sin(t) * OUT,
            t_range=(0, PI, 0.01)
        )
        semi_circ.shift(height * OUT)
        line1, line2 = VGroup(
            Line(point + height * IN, point)
            for point in semi_circ.get_start_and_end()
        )
        path = line1.copy()
        path.append_vectorized_mobject(semi_circ)
        path.append_vectorized_mobject(line2.reverse_points())

        path.set_stroke(RED)
        path.insert_n_curves(100)

        return MoveAlongPath(peg1, path, run_time=run_time)

    def get_peg_board(self, radius=10):
        holes = Group(
            Dot().move_to((x, y, 0)).reverse_points()
            for x in range(-radius, radius)
            for y in range(-radius, radius)
        )
        board = SurroundingRectangle(holes)
        for hole in holes:
            board.append_vectorized_mobject(hole)

        board.set_fill(GREY_D, 0.5).set_stroke(WHITE, 1)
        return board

    def get_peg(self, x, y, ball_radius=0.15, base_radius=0.07, base_height=0.25):
        top = Sphere(radius=ball_radius)
        base = Cone(radius=base_radius, height=1)
        base.rotate(PI, RIGHT)
        base.move_to(top.get_center(), OUT)
        peg = Group(top, base)
        peg.set_color(GREY_B)
        peg.set_shading(0.5, 0.5, 0.1)
        z = -base_height + ball_radius
        peg.move_to((x, y, z))
        return peg

    def choose_weighted_jump(self, pegs, target_xy=np.array([1, 1]), bias=0.3, exclude_jump=None):
        candidates = []
        for jumper in pegs:
            for pivot in pegs:
                if jumper is pivot:
                    continue
                if exclude_jump is not None and jumper is exclude_jump[0] and pivot is exclude_jump[1]:
                    continue
                landing_xy = (2 * pivot.get_center() - jumper.get_center())[:2]
                dist = np.linalg.norm(landing_xy - target_xy)
                candidates.append((dist, jumper, pivot))
        candidates.sort(key=lambda c: c[0])
        weights = [bias ** rank for rank in range(len(candidates))]
        _, jumper, pivot = random.choices(candidates, weights=weights, k=1)[0]
        return jumper, pivot

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
