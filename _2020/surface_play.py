from manim_imports_ext import *


class EarthMorph(Scene):
    CONFIG = {
        "camera_class": ThreeDCamera,
    }

    def construct(self):
        torus1 = Torus(r1=1, r2=1)
        torus2 = Torus(r1=3, r2=1)
        sphere = Sphere(radius=3, resolution=torus1.resolution)
        earths = [
            TexturedSurface(surface, "EarthTextureMap", "NightEarthTextureMap")
            for surface in [sphere, torus1, torus2]
        ]
        for mob in earths:
            mob.mesh = SurfaceMesh(mob)
            mob.mesh.set_stroke(BLUE, 1, opacity=0.5)

        earth = earths[0]

        self.camera.frame.set_euler_angles(
            theta=-30 * DEGREES,
            phi=70 * DEGREES,
        )

        self.add(earth)
        self.play(ShowCreation(earth.mesh, lag_ratio=0.01, run_time=3))
        for mob in earths:
            mob.add(mob.mesh)
        earth.save_state()
        self.play(Rotate(earth, PI / 2), run_time=2)
        for mob in earths[1:]:
            mob.rotate(PI / 2)

        self.play(
            Transform(earth, earths[1]),
            run_time=3
        )

        light = self.camera.light_source
        frame = self.camera.frame

        self.play(
            Transform(earth, earths[2]),
            frame.increment_phi, -10 * DEGREES,
            frame.increment_theta, -20 * DEGREES,
            run_time=3
        )
        frame.add_updater(lambda m, dt: m.increment_theta(-0.1 * dt))
        self.add(light)
        light.save_state()
        self.play(light.move_to, 3 * IN, run_time=5)
        self.play(light.shift, 10 * OUT, run_time=5)
        self.wait(4)
