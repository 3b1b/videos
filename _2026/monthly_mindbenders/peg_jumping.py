from manim_imports_ext import *


class PegTest(InteractiveScene):
    samples = 4

    def construct(self):
        frame = self.frame

        peg_board = self.get_peg_board()
        self.add(peg_board)

        pegs = Group(
            self.get_peg(0, 0),
            self.get_peg(1, 0),
            self.get_peg(0, 1),
        )

        # Introduce pegs
        square = Square(side_length=1)
        square.move_to(ORIGIN, DL)
        square.set_stroke(TEAL, 3)
        square.apply_depth_test()

        self.play(
            frame.animate(run_time=4).reorient(27, 64, 0, (0.11, 0.10, 0.12), 4.50),
            LaggedStartMap(FadeIn, pegs, shift=IN, lag_ratio=0.5, time_span=(3, 5)),
        )
        self.wait()

        # Do some jumps
        peg1, peg2, peg3 = pegs

        self.illustrated_peg_jump(peg1, peg2)
        self.wait()
        self.illustrated_peg_jump(peg3, peg2)
        self.wait()

    def illustrated_peg_jump(self, peg1, peg2, color=YELLOW):
        line = DashedLine(peg2.get_center(), peg1.get_center(), dash_length=0.025)
        line.set_z(0)
        line.set_color(color)
        line.apply_depth_test()
        moving_line = line.copy()
        traced_path = TracingTail(moving_line.get_end, stroke_color=color, time_traced=3)

        axis = np.cross(peg1.get_center() - peg2.get_center(), OUT)

        self.play(ShowCreation(line, run_time=0.5))
        self.wait()
        self.add(traced_path)
        self.play(LaggedStart(
            Rotate(moving_line, PI, axis=axis, run_time=2, about_point=line.get_start()),
            self.simple_peg_jump(peg1, peg2),
            lag_ratio=0.7,
        ))
        self.wait()
        self.play(FadeOut(VGroup(line, moving_line)))

    def simple_peg_jump(self, peg1, peg2, height=0.5, run_time=2):
        # Test
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
