from manim_imports_ext import *


class MugToTorus(ThreeDScene):
    def construct(self):
        frame = self.camera.frame
        frame.reorient(-20, 60)

        R1, R2 = (2, 0.75)

        def torus_func(u, v):
            v1 = np.array([-math.sin(u), 0, math.cos(u)])
            v2 = math.cos(v) * v1 + math.sin(v) * UP
            return R1 * v1 + R2 * v2

        def cylinder_func(u, v):
            return (math.cos(v), math.sin(v), u)

        left_half_torus = ParametricSurface(
            torus_func,
            u_range=(-PI / 2, PI + PI / 2),
            v_range=(0, TAU),
        )
        right_half_torus = ParametricSurface(
            torus_func,
            u_range=(PI, TAU),
            v_range=(0, TAU),
        )
        cylinder = ParametricSurface(
            cylinder_func,
            u_range=(PI, TAU),
            v_range=(0, TAU),
        )
        cylinder.set_width(3)
        cylinder.set_depth(5, stretch=True)
        cylinder.move_to(ORIGIN, LEFT)

        disk = Disk3D(resolution=(2, 50))
        disk.match_width(cylinder)
        disk.move_to(cylinder, IN)
        disk.scale(1.001)
        low_disk = disk.copy()

        for mob in (left_half_torus, right_half_torus, cylinder, low_disk, disk):
            mob.set_color(GREY)
            mob.set_gloss(0.7)

        left_half_torus.save_state()
        left_half_torus.set_depth(3, about_point=ORIGIN)

        self.add(left_half_torus)
        self.add(cylinder)
        self.add(low_disk, disk)
        self.add(frame)

        self.play(disk.animate.move_to(cylinder, OUT), run_time=2)

        for mob in (disk, low_disk):
            mob.generate_target()
            mob.target.rotate(90 * DEGREES, DOWN)
            mob.target.set_depth(2 * R2)
        disk.target.move_to(right_half_torus, OUT + LEFT)
        low_disk.target.rotate(PI, UP)
        low_disk.target.move_to(right_half_torus, IN + LEFT)

        self.play(
            MoveToTarget(disk),
            MoveToTarget(low_disk),
            Transform(cylinder, right_half_torus),
            Restore(left_half_torus, rate_func=squish_rate_func(smooth, 0, 0.75)),
            frame.animate.reorient(-10, 80),
            run_time=5,
        )
        self.wait()
