from manim_imports_ext import *


def stereo_project_point(point, axis=0, r=1, max_norm=10000):
    point = fdiv(point * r, point[axis] + r)
    point[axis] = 0
    norm = get_norm(point)
    if norm > max_norm:
        point *= max_norm / norm
    return point


def sudanese_band_func(eta, phi):
    z1 = math.sin(eta) * np.exp(complex(0, phi))
    z2 = math.cos(eta) * np.exp(complex(0, phi / 2))
    r4_point = np.array([z1.real, z1.imag, z2.real, z2.imag])
    r4_point[:3] = rotate_vector(r4_point[:3], PI / 3, axis=[1, 1, 1])
    return stereo_project_point(r4_point, axis=0)[1:]


def mobius_strip_func(u, phi):
    vect = rotate_vector(RIGHT, phi / 2, axis=UP)
    vect = rotate_vector(vect, phi, axis=OUT)
    ref_point = np.array([np.cos(phi), np.sin(phi), 0])
    return ref_point + 0.7 * (u - 0.5) * vect


def reversed_band(band_func):
    return lambda x, phi: band_func(x, -phi)


def get_full_surface(band_func, x_range):
    surface = ParametricSurface(
        band_func, x_range, (0, TAU),
    )
    surface.set_color(BLUE_D)
    surface.set_shadow(0.5)
    surface.add_updater(lambda m: m.sort_faces_back_to_front(DOWN))
    # surface = TexturedSurface(surface, "EarthTextureMap", "NightEarthTextureMap")
    # surface = TexturedSurface(surface, "WaterColor")
    # inv_surface = ParametricSurface(
    #     reversed_band(band_func), x_range[::-1], (0, TAU),
    # )
    m1, m2 = meshes = VGroup(
        SurfaceMesh(surface, normal_nudge=1e-3),
        SurfaceMesh(surface, normal_nudge=-1e-3),
    )
    bound = VGroup(
        ParametricCurve(lambda t: band_func(x_range[0], t), (0, TAU)),
        ParametricCurve(lambda t: band_func(x_range[1], t), (0, TAU)),
    )
    bound.set_stroke(RED, 3)
    bound.apply_depth_test()
    meshes.set_stroke(WHITE, 0.5, 0.5)
    return Group(surface, m1, m2, bound)
    return Group(surface, bound)


def get_sudanese_band(circle_on_xy_plane=False):
    s_band = get_full_surface(
        sudanese_band_func,
        (0, PI),
    )
    angle = angle_of_vector(s_band[-1][0].get_start() - s_band[-1][1].get_start())
    s_band.rotate(PI / 2 - angle)
    if circle_on_xy_plane:
        s_band.rotate(90 * DEGREES, DOWN)
    s_band.shift(-s_band[-1][0].get_start())
    return s_band


class SudaneseBand(ThreeDScene):
    circle_on_xy_plane = True

    def construct(self):
        frame = self.camera.frame
        frame.reorient(-45, 70)
        frame.add_updater(
            lambda m, dt: m.increment_theta(2 * dt * DEGREES)
        )
        self.add(frame)

        s_band = get_sudanese_band(self.circle_on_xy_plane)
        m_band = get_full_surface(mobius_strip_func, (0, 1))
        for band in s_band, m_band:
            band.set_height(6)

        # self.play(ShowCreation(m_band[0]))
        # self.play(
        #     FadeIn(m_band[1]),
        #     FadeIn(m_band[2]),
        #     ShowCreation(m_band[3]),
        # )
        self.add(m_band)
        self.wait()
        m_band.save_state()
        self.play(
            Transform(m_band, s_band),
            run_time=8,
        )
        # self.wait()
        self.play(frame.animate.reorient(-30, 110), run_time=4)
        # self.play(frame.animate.reorient(-30, 70), run_time=3)
        self.wait(2)
        frame.clear_updaters()
        self.play(
            m_band.animate.restore(),
            frame.animate.reorient(-45, 70),
            run_time=8,
        )

        # self.embed()


class SudaneseBandToKleinBottle(ThreeDScene):
    def construct(self):
        frame = self.camera.frame
        frame.reorient(-70, 70)
        # frame.add_updater(
        #     lambda m, dt: m.increment_theta(2 * dt * DEGREES)
        # )
        # self.add(frame)

        s_band = get_sudanese_band()
        s_band[1:3].set_opacity(0)
        circ = s_band[-1]
        s_band.shift(-circ.get_center())
        sb_copy = s_band.copy()

        self.add(s_band)
        self.play(
            Rotate(sb_copy, PI, axis=RIGHT, about_point=ORIGIN),
            run_time=4
        )
        self.play(frame.animate.reorient(360 - 70, 70), run_time=15)
        self.wait()

        self.embed()
