from manim_imports_ext import *


# Helpers
def project_to_xy_plane(p1, p2):
    """
    Draw a line from source to p1 to p2.  Where does it
    intersect the xy plane?
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    if z2 < z1:
        z2 = z1 + 1e-2  # TODO, bad hack
    vect = p2 - p1
    return p1 - (z2 / vect[2]) * vect


def flat_project(point):
    return [*point[:2], 0]


def get_pre_shadow(mobject, opacity):
    result = mobject.deepcopy()
    result.clear_updaters()

    color = interpolate_color(
        mobject[0].get_fill_color(), BLACK,
        mobject[0].get_fill_opacity()
    )
    # color = BLACK
    for sm in result.family_members_with_points():
        sm.set_fill(color, opacity=opacity)
        sm.set_stroke(BLACK, 0.5, opacity=opacity)
        sm.set_gloss(sm.get_gloss() * 0.5)
        sm.set_shadow(0)
        sm.set_reflectiveness(0)
    return result


def update_shadow(shadow, mobject, light_source):
    lp = light_source.get_center() if light_source is not None else None

    def project(point):
        if lp is None:
            return flat_project(point)
        else:
            return project_to_xy_plane(lp, point)

    for sm, mm in zip(shadow.family_members_with_points(), mobject.family_members_with_points()):
        sm.set_points(np.apply_along_axis(project, 1, mm.get_points()))


def get_shadow(mobject, light_source=None, opacity=0.5):
    shadow = get_pre_shadow(mobject, opacity)
    shadow.add_updater(lambda s: update_shadow(s, mobject, light_source))
    return shadow


def get_boundary_points(shadow, n_points=20):
    points = shadow.get_points_defining_boundary()
    return np.array([
        points[np.argmax(np.dot(points, vect.T))]
        for vect in compass_directions(n_points)
    ])


def get_area(planar_mobject):
    boundary = get_boundary_points(planar_mobject, 100)
    xs = boundary[:, 0]
    ys = boundary[:, 1]
    dxs = np.append(xs[-1], xs[:-1]) - xs
    dys = np.append(ys[-1], ys[:-1]) - ys
    return abs(sum([
        0.5 * (x * dy - y * dx)
        for x, dx, y, dy in zip(xs, dxs, ys, dys)
    ]))


# Scenes

class ShadowScene(ThreeDScene):
    object_center = [0, 0, 3]
    frame_center = [0, 0, 2]
    area_label_center = [0, -1.5, 0]
    surface_area = 6.0
    num_reorientations = 10
    plane_dims = (20, 20)
    plane_style = {
        "stroke_width": 0,
        "fill_color": GREY_A,
        "fill_opacity": 0.5,
        "gloss": 0.5,
        "shadow": 0.2,
    }
    inf_light = False
    glow_radius = 10
    glow_factor = 10

    def setup(self):
        self.camera.frame.reorient(-30, 75)
        self.camera.frame.move_to(self.frame_center)
        self.add_plane()
        self.add_solid()
        self.add_shadow()
        self.setup_light_source()

    def add_plane(self):
        width, height = self.plane_dims
        plane = self.plane = Rectangle(width, height)
        plane.set_style(**self.plane_style)

        grid = NumberPlane(
            x_range=(-width // 2, width // 2, 2),
            y_range=(-height // 2, height // 2, 2),
            background_line_style={
                "stroke_color": GREY_B,
                "stroke_width": 1,
            },
            faded_line_ratio=4,
        )
        grid.axes.match_style(grid.background_lines)
        grid.set_flat_stroke(True)
        plane.add(grid)
        self.add(plane)

    def add_solid(self):
        self.solid = self.get_object()
        self.solid.move_to(self.object_center)
        self.solid.add_updater(lambda m: self.sort_to_camera(m))
        self.add(self.solid)

    def get_object(self):
        cube = VCube()
        cube.deactivate_depth_test()
        cube.set_height(2)
        cube.set_stroke(WHITE, 0.5)
        cube.set_fill(BLUE_E, 0.8)
        cube.set_reflectiveness(0.3)
        cube.set_gloss(0.1)
        cube.set_shadow(0.5)
        # Wrap in group so that strokes and fills
        # are rendered in separate passes
        cube = self.cube = Group(*cube)
        return cube

    def add_shadow(self):
        light_source = None if self.inf_light else self.camera.light_source
        shadow = get_shadow(self.solid, light_source)

        self.add(shadow, self.solid)
        self.shadow = shadow

    def setup_light_source(self):
        self.light = self.camera.light_source
        glow = self.glow = TrueDot(
            radius=self.glow_radius,
            glow_factor=self.glow_factor,
        )
        glow.set_color(interpolate_color(YELLOW, WHITE, 0.5))
        glow.add_updater(lambda m: m.move_to(self.light))
        self.add(glow)

    def sort_to_camera(self, mobject):
        cl = self.camera.get_location()
        mobject.sort(lambda p: -get_norm(p - cl))
        for sm in mobject:
            sm.refresh_unit_normal()
        return mobject

    # TODO
    def add_shadow_area_label(self):
        text = TexText("Shadow area: ")
        decimal = DecimalNumber(0)
        label = VGroup(text, decimal)
        label.arrange(RIGHT)
        label.scale(1.5)
        label.move_to(self.area_label_center - decimal.get_center())
        self.shadow_area_label = label
        self.shadow_area_decimal = decimal

        # def update_decimal(decimal):
        #     # decimal.set_value(get_area(self.shadow))
        #     self.add_fixed_in_frame_mobjects(decimal)

        # decimal.add_updater(update_decimal)
        decimal.add_updater(
            lambda d: d.set_value(get_area(self.shadow))
        )
        decimal.add_updater(
            lambda d: self.add_fixed_in_frame_mobjects(d)
        )

        # self.add_fixed_orientation_mobjects(label)
        self.add_fixed_in_frame_mobjects(label)
        self.add(label)
        self.add(decimal)

    # TODO
    def add_surface_area_label(self):
        text = TexText("Surface area: ")
        decimal = DecimalNumber(self.surface_area)
        label = VGroup(text, decimal)
        label.arrange(RIGHT)
        label.scale(1.25)
        label.set_fill(YELLOW)
        label.set_background_stroke(width=3)
        label.next_to(self.obj3d, RIGHT, LARGE_BUFF)
        label.shift(MED_LARGE_BUFF * IN)
        self.surface_area_label = label
        self.add_fixed_orientation_mobjects(label)

    # TODO
    def get_average_label(self):
        rect = SurroundingRectangle(
            self.shadow_area_decimal,
            buff=SMALL_BUFF,
            color=RED,
        )
        words = TexText(
            "Average", "=",
            "$\\frac{\\text{Surface area}}{4}$"
        )
        words.scale(1.5)
        words[0].match_color(rect)
        words[2].set_color(self.surface_area_label[0].get_fill_color())
        words.set_background_stroke(width=3)
        words.next_to(
            rect, DOWN,
            index_of_submobject_to_align=0,
        )
        # words.shift(MED_LARGE_BUFF * LEFT)
        return VGroup(rect, words)


class IntroduceShadow(ShadowScene):
    def construct(self):
        light = self.light
        cube = self.cube

        self.play(
            light.animate.next_to(cube, OUT, 2),
            run_time=5,
        )
        self.wait()

        self.embed()


class ShowInfinitelyFarLightSource(ShadowScene):
    CONFIG = {
        "num_reorientations": 1,
        "camera_center": [0, 0, 1],
    }

    def construct(self):
        self.force_skipping()
        ShowShadows.construct(self)
        self.revert_to_original_skipping_status()

        self.add_light_source_based_shadow_updater()
        self.add_light()
        self.move_light_around()
        self.show_vertical_lines()

    def add_light(self):
        light = self.light = self.get_light()
        light_source = self.camera.light_source
        light.move_to(light_source)
        light_source.add_updater(lambda m: m.move_to(light))
        self.add(light_source)
        self.add_fixed_orientation_mobjects(light)

    def move_light_around(self):
        light = self.light
        self.add_foreground_mobjects(self.shadow_area_label)
        self.play(
            light.move_to, 5 * OUT + DOWN,
            run_time=3
        )
        self.play(Rotating(
            light, angle=TAU, about_point=5 * OUT,
            rate_func=smooth, run_time=3
        ))
        self.play(
            light.move_to, 30 * OUT,
            run_time=3,
        )
        self.remove(light)

    def show_vertical_lines(self):
        lines = self.get_vertical_lines()
        obj3d = self.obj3d
        shadow = self.shadow
        target_obj3d = obj3d.copy()
        target_obj3d.become(shadow)
        target_obj3d.match_style(obj3d)
        target_obj3d.set_shade_in_3d(False)
        source_obj3d = obj3d.copy()
        source_obj3d.set_shade_in_3d(False)
        source_obj3d.fade(1)

        self.play(LaggedStartMap(ShowCreation, lines))
        self.wait()
        self.add(source_obj3d, lines)
        self.play(
            ReplacementTransform(source_obj3d, target_obj3d),
            run_time=2
        )
        self.add(target_obj3d, lines)
        self.play(FadeOut(target_obj3d),)
        self.wait()
        lines.add_updater(lambda m: m.become(self.get_vertical_lines()))
        for x in range(5):
            self.randomly_reorient()

    def add_light_source_based_shadow_updater(self):
        shadow = self.shadow
        light_source = self.camera.light_source
        obj3d = self.obj3d
        center = obj3d.get_center()

        def update(shadow):
            lsp = light_source.get_center()
            proj_center = get_xy_plane_projection_point(lsp, center)
            c_to_lsp = lsp - center
            unit_c_to_lsp = normalize(c_to_lsp)
            rotation = rotation_matrix(
                angle=np.arccos(np.dot(unit_c_to_lsp, OUT)),
                axis=normalize(np.cross(unit_c_to_lsp, OUT))
            )
            new_shadow = get_shadow(
                self.obj3d.copy().apply_matrix(rotation)
            )
            shadow.become(new_shadow)
            shadow.scale(get_norm(lsp) / get_norm(c_to_lsp))
            shadow.move_to(proj_center)
            return shadow
        shadow.add_updater(update)

    def get_light(self):
        n_rings = 40
        radii = np.linspace(0, 2, n_rings)
        rings = VGroup(*[
            Annulus(inner_radius=r1, outer_radius=r2)
            for r1, r2 in zip(radii, radii[1:])
        ])
        opacities = np.linspace(1, 0, n_rings)**1.5
        for opacity, ring in zip(opacities, rings):
            ring.set_fill(YELLOW, opacity)
            ring.set_stroke(YELLOW, width=0.1, opacity=opacity)
        return rings

    def get_vertical_lines(self):
        shadow = self.shadow
        points = get_boundary_points(shadow, 10)
        # half_points = [(p1 + p2) / 2 for p1, p2 in adjacent_pairs(points)]
        # points = np.append(points, half_points, axis=0)
        light_source = self.light.get_center()
        lines = VGroup(*[
            DashedLine(light_source, point)
            for point in points
        ])
        lines.set_shade_in_3d(True)
        for line in lines:
            line.remove(*line[:int(0.8 * len(line))])
            line[-10:].set_shade_in_3d(False)
            line.set_stroke(YELLOW, 1)
        return lines


class CylinderShadows(ShadowScene):
    CONFIG = {
        "surface_area": 2 * PI + 2 * PI * 2,
        "area_label_center": [0, -2, 0],
    }

    def get_object(self):
        height = 2
        cylinder = ParametricSurface(
            lambda u, v: np.array([
                np.cos(TAU * v),
                np.sin(TAU * v),
                height * (1 - u)
            ]),
            resolution=(6, 32)
        )
        # circle = Circle(radius=1)
        circle = ParametricSurface(
            lambda u, v: np.array([
                (v + 0.01) * np.cos(TAU * u),
                (v + 0.01) * np.sin(TAU * u),
                0,
            ]),
            resolution=(16, 8)
        )
        # circle.set_fill(GREEN, opacity=0.5)
        for surface in cylinder, circle:
            surface.set_fill_by_checkerboard(GREEN, GREEN_E, opacity=1.0)
            # surface.set_fill(GREEN, opacity=0.5)
        cylinder.add(circle)
        cylinder.add(circle.copy().flip().move_to(height * OUT))
        cylinder.set_shade_in_3d(True)
        cylinder.set_stroke(width=0)
        cylinder.scale(1.003)
        return cylinder


class PrismShadows(ShadowScene):
    CONFIG = {
        "surface_area": 3 * np.sqrt(3) / 2 + 3 * (np.sqrt(3) * 2),
        "object_center": [0, 0, 3],
        "area_label_center": [0, -2.25, 0],
    }

    def get_object(self):
        height = 2
        prism = VGroup()
        triangle = RegularPolygon(3)
        verts = triangle.get_anchors()[:3]
        rects = [
            Polygon(v1, v2, v2 + height * OUT, v1 + height * OUT)
            for v1, v2 in adjacent_pairs(verts)
        ]
        prism.add(triangle, *rects)
        prism.add(triangle.copy().shift(height * OUT))
        triangle.reverse_points()
        prism.set_shade_in_3d(True)
        prism.set_fill(PINK, 0.8)
        prism.set_stroke(WHITE, 1)
        return prism


class TheseFourPiAreSquare(PiCreatureScene):
    def construct(self):
        pass

    def create_pi_creatures(self):
        pass
