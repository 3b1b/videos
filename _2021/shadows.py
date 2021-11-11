from manim_imports_ext import *
import scipy.spatial


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
    if isinstance(result, Group) and isinstance(result[0], VMobject):
        result = VGroup(*result)
    result.clear_updaters()

    for sm in result.family_members_with_points():
        # color = interpolate_color(sm.get_color(), BLACK, sm.get_opacity())
        color = interpolate_color(sm.get_color(), BLACK, opacity)
        sm.set_color(color)
        sm.set_opacity(opacity)
        if isinstance(sm, VMobject):
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
        if isinstance(sm, VMobject) and sm.get_unit_normal()[2] < 0:
            sm.reverse_points()
        sm.set_fill(opacity=mm.get_fill_opacity())


def get_shadow(mobject, light_source=None, opacity=0.7):
    shadow = get_pre_shadow(mobject, opacity)
    shadow.add_updater(lambda s: update_shadow(s, mobject, light_source))
    return shadow


def get_area(shadow):
    return 0.5 * sum(
        get_norm(sm.get_area_vector())
        for sm in shadow.get_family()
    )


def get_convex_hull(mobject):
    points = mobject.get_all_points()
    hull = scipy.spatial.ConvexHull(points[:, :2])
    return points[hull.vertices]


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
    object_style = {
        "stroke_color": WHITE,
        "stroke_width": 0.5,
        "fill_color": BLUE_E,
        "fill_opacity": 0.7,
        "reflectiveness": 0.3,
        "gloss": 0.1,
        "shadow": 0.5,
    }
    inf_light = False
    glow_radius = 10
    glow_factor = 10
    area_label_center = [-2, -1, 0]
    unit_size = 2

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
        self.solid = self.get_solid()
        self.solid.move_to(self.object_center)
        self.solid.add_updater(lambda m: self.sort_to_camera(m))
        self.add(self.solid)

    def get_solid(self):
        cube = VCube()
        cube.deactivate_depth_test()
        cube.set_height(2)
        cube.set_style(**self.object_style)
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
        if self.inf_light:
            self.light.move_to(100 * OUT)
        else:
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

    def get_shadow_area_label(self):
        text = TexText("Shadow area: ")
        decimal = DecimalNumber(0)
        decimal.add_updater(lambda d: d.set_value(
            get_area(self.shadow) / (self.unit_size**2)
        ))

        label = VGroup(text, decimal)
        label.arrange(RIGHT)
        label.move_to(self.area_label_center - decimal.get_center())
        label.fix_in_frame()
        label.set_stroke(BLACK, 3, background=True)
        return label

    def begin_ambient_rotation(self, mobject, speed=0.2):
        mobject.rot_axis = np.array([1, 1, 1])

        def update_mob(mob, dt):
            mob.rotate(speed * dt, mob.rot_axis)
            mob.rot_axis = rotate_vector(mob.rot_axis, speed * dt, OUT)
            return mob
        mobject.add_updater(update_mob)
        return mobject

    def get_shadow_outline(self, stroke_width=1):
        outline = VMobject()
        outline.set_stroke(WHITE, stroke_width)
        outline.add_updater(lambda m: m.set_points_as_corners(get_convex_hull(self.shadow)).close_path())
        return outline

    def get_light_lines(self, outline=None, n_lines=100, only_vertices=False):
        if outline is None:
            outline = self.get_shadow_outline()
        lp = self.light.get_center()

        def update_lines(lines):
            if only_vertices:
                points = outline.get_vertices()
            else:
                points = [outline.pfp(a) for a in np.linspace(0, 1, n_lines)]
            for line, point in zip(lines, points):
                if self.inf_light:
                    line.put_start_and_end_on(lp, point)
                else:
                    line.put_start_and_end_on(point + 10 * OUT, point)

        line = Line()
        line.insert_n_curves(5)
        light_lines = line.replicate(n_lines)
        light_lines.set_stroke(YELLOW, 0.5, 0.1)
        light_lines.add_updater(update_lines)
        return light_lines

    def randomly_reorient(self, run_time=1):
        # axis = normalize(np.random.random(3))
        # angle = PI + np.random.random() * PI
        self.solid.rot_axis = normalize(np.random.random(3))
        self.solid.rot_time = 0

        def update(mob, time):
            dt = time - mob.rot_time
            mob.rot_time = time
            mob.rot_axis = rotate_vector(mob.rot_axis, 5 * dt, normalize(np.random.random(3)))
            mob.rotate(TAU * dt, mob.rot_axis)

        # self.play(Rotate(self.solid, angle, axis), run_time=run_time)
        self.play(UpdateFromAlphaFunc(self.solid, update), run_time=run_time)


class IntroduceShadow(ShadowScene):
    area_label_center = [-2.5, -2, 0]
    plane_dims = (30, 20)

    def construct(self):
        # Setup
        light = self.light
        cube = self.solid
        shadow = self.shadow
        outline = self.get_shadow_outline()
        frame = self.camera.frame
        cube.scale(0.945)  # Hack to make the appropriate area 1

        # Ambient rotation
        frame.add_updater(lambda f, dt: f.increment_theta(0.01 * dt))
        light.move_to([-2, 2, 10])

        area_label = self.get_shadow_area_label()
        question = TexText(
            "Puzzle: Find the average\\\\area of a cube's shadow",
            font_size=48,
        )
        question.to_corner(UL)
        question.fix_in_frame()

        # Introductory animations
        self.shadow.update()
        self.play(
            FadeIn(question, 0.5 * UP),
            *(
                LaggedStartMap(DrawBorderThenFill, mob, lag_ratio=0.1, run_time=3)
                for mob in (cube, shadow)
            )
        )
        area_label.update()
        outline.update()
        self.play(
            FadeIn(area_label, lag_ratio=0.1),
            ShowCreation(outline),
        )
        self.begin_ambient_rotation(cube)
        self.wait(8)

        # Ask questions
        questions = VGroup(
            Text("Where is the light?"),
            TexText("``Average'' in what sense?"),
        )
        questions.set_color(TEAL)
        questions.arrange(DOWN, MED_LARGE_BUFF)
        questions.to_corner(UR)
        questions.fix_in_frame()

        light_lines = always_redraw(lambda: self.get_light_lines(outline))

        self.play(
            LaggedStartMap(
                FadeIn, questions,
                shift=0.5 * DOWN,
                lag_ratio=0.5
            ),
            ShowCreation(
                light_lines,
                lag_ratio=0.01,
                run_time=2
            )
        )
        self.play(
            light.animate.next_to(cube, OUT, 2),
            frame.animate.set_height(12).set_z(3),
            run_time=3,
        )
        self.play(light.animate.shift(1.0 * IN), run_time=2)
        self.wait()
        self.play(light.animate.shift(2 * OUT), run_time=2)

        # Long ambient rotation
        self.wait(18)

        # Light infinitely far away
        underlines = VGroup(*(
            Underline(q, buff=-0.05) for q in questions
        ))
        underlines.set_stroke(YELLOW, 1)
        underlines.fix_in_frame()

        self.play(
            questions[0].animate.set_color(YELLOW),
            questions[1].animate.set_opacity(0.2),
            ShowCreation(underlines[0]),
        )
        light_points = (
            [-2, 2, 6],
            [4, 2, 5.5],
            [4, -2, 6.5],
            [0, 0, 8],
            [0, 0, 75],
        )
        for point in light_points:
            self.play(light.animate.move_to(point), run_time=2)
            self.wait()
        self.wait(3)

        # Flat projection
        cube.clear_updaters()
        cube.add_updater(lambda m: self.sort_to_camera(m))
        cube_copy = cube.deepcopy()
        shadow_copy = get_pre_shadow(cube_copy, 0.75)
        shadow_copy.apply_function(lambda p: [*p[:2], 0])
        self.play(LaggedStart(*(
            ReplacementTransform(c.copy().fade(1), s)
            for c, s in zip(cube_copy, shadow_copy)
        )), lag_ratio=0.9, run_time=2)
        self.play(FadeOut(shadow_copy))
        self.wait(2)

        # Square projection
        top_face = cube[np.argmax([f.get_z() for f in cube])]
        normal_vect = top_face.get_unit_normal()
        theta = np.arccos(normal_vect[2])
        axis = normalize(rotate_vector([*normal_vect[:2], 0], PI / 2, OUT))

        self.play(
            Rotate(cube, -theta, axis),
            run_time=2,
        )
        top_face = cube[np.argmax([f.get_z() for f in cube])]
        verts = top_face.get_vertices()
        vect = verts[3] - verts[2]
        angle = angle_of_vector(vect)
        self.play(Rotate(cube, -angle, OUT))
        self.wait()

        corner = cube.get_corner(DL + OUT)
        edge_lines = VGroup(
            Line(corner, cube.get_corner(DR + OUT)),
            Line(corner, cube.get_corner(UL + OUT)),
            Line(corner, cube.get_corner(DL + IN)),
        )
        edge_lines.set_stroke(RED, 2)
        s_labels = Tex("s").replicate(3)
        s_labels.set_color(RED)
        s_labels.rotate(PI / 2, RIGHT)
        s_labels.set_stroke(BLACK, 3, background=True)
        for label, line, vect in zip(s_labels, edge_lines, [OUT, LEFT, LEFT]):
            label.next_to(line, vect, buff=SMALL_BUFF)
        s_labels[1].next_to(edge_lines[1], OUT)
        s_labels[2].next_to(edge_lines[2], LEFT)

        s_squared = Tex("s^2")
        s_squared.match_style(s_labels[0])
        s_squared.move_to(self.shadow)

        frame.generate_target()
        frame.target.reorient(10, 60)
        frame.target.scale(0.8)

        self.play(
            LaggedStartMap(ShowCreation, edge_lines),
            LaggedStartMap(FadeIn, s_labels, scale=2),
            MoveToTarget(frame, run_time=3)
        )
        self.wait()
        self.play(
            TransformFromCopy(s_labels[:2], s_squared),
        )
        self.wait(2)

        rect = SurroundingRectangle(area_label)
        rect.fix_in_frame()
        rect.set_stroke(YELLOW, 3)
        s_eq = Tex("s = 1")
        s_eq.next_to(area_label, DOWN)
        s_eq.set_color(RED)
        s_eq.set_stroke(BLACK, 3, background=True)
        s_eq.fix_in_frame()

        self.play(ShowCreation(rect))
        self.play(FadeIn(s_eq, 0.5 * DOWN))
        self.wait()
        self.play(LaggedStart(*map(FadeOut, (
            rect, s_eq, *edge_lines, *s_labels, s_squared,
        ))))
        self.wait()

        # Hexagonal orientation
        axis = UL
        angle = np.arccos(1 / math.sqrt(3))
        area_label.suspend_updating()
        self.play(
            Rotate(cube, -angle, axis),
            frame.animate.reorient(-10, 70),
            ChangeDecimalToValue(area_label[1], math.sqrt(3)),
            UpdateFromFunc(area_label[1], lambda m: m.fix_in_frame()),
            run_time=2
        )
        self.add(area_label)

        hex_area_label = Tex("\\sqrt{3} s^2")
        hex_area_label.set_color(RED)
        hex_area_label.move_to(self.shadow)
        self.play(Write(hex_area_label))
        self.wait(2)
        area_label.resume_updating()
        self.play(
            FadeOut(hex_area_label),
            Rotate(cube, 4, RIGHT)
        )
        self.wait(3)

        # Talk about averages
        light_lines.clear_updaters()
        self.play(
            FadeOut(underlines[0]),
            FadeOut(light_lines),
            ShowCreation(underlines[1]),
            questions[0].animate.set_opacity(0.1),
            questions[1].animate.set_fill(YELLOW, 1),
        )
        self.wait()

        samples = VGroup(VectorizedPoint())
        samples.next_to(questions, DOWN, buff=0.5)
        samples.shift(RIGHT)
        self.add(samples)
        for x in range(7):
            self.randomly_reorient()
            sample = area_label[1].copy()
            sample.clear_updaters()
            sample.fix_in_frame()
            self.play(sample.animate.next_to(samples, DOWN))
            samples.add(sample)

        v_dots = Tex("\\vdots")
        v_dots.next_to(samples, DOWN)
        v_dots.fix_in_frame()
        samples.add(v_dots)
        brace = Brace(samples, LEFT)
        brace.fix_in_frame()
        brace.next_to(samples, LEFT, SMALL_BUFF)
        text = TexText(
            "Take the mean.", "\\\\What does that\\\\approach?",
            font_size=30
        )
        text[0].shift(MED_SMALL_BUFF * UP)
        text.next_to(brace, LEFT)
        text.fix_in_frame()
        VGroup(text, brace).set_stroke(BLACK, 3, background=True)

        self.play(
            GrowFromCenter(brace),
            FadeIn(text),
            Write(v_dots),
        )
        self.wait()

        for x in range(7):
            self.randomly_reorient()
            self.wait()


class FocusOnOneFace(ShadowScene):
    inf_light = True

    def construct(self):
        # Some random tumbling
        cube = self.solid
        shadow = self.shadow
        frame = self.camera.frame

        words = VGroup(
            Text("Just one orientation"),
            Text("Just one face"),
        )
        words.fix_in_frame()
        words.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        words.to_corner(UL)

        for x in range(3):
            self.wait()
            self.randomly_reorient()
        self.play(FadeIn(words[0], scale=0.75, run_time=0.5))
        self.wait()

        # Just one face
        index = np.argmax([f.get_z() for f in cube])
        face = cube[index]
        prev_opacity = face.get_fill_opacity()
        cube.generate_target()
        cube.target.space_out_submobjects(2, about_point=face.get_center())
        cube.target.set_opacity(0)
        cube.target[index].set_opacity(prev_opacity)

        self.shadow.set_stroke(width=0)
        self.play(
            MoveToTarget(cube),
            FadeIn(words[1]),
        )
        self.play(
            frame.animate.reorient(-10, 65),
            run_time=3,
        )
        frame.add_updater(lambda f, dt: f.increment_theta(0.01 * dt))

        self.solid = face
        self.remove(shadow)
        self.add_shadow()
        shadow = self.shadow

        # Ask about area
        area_q = Text("Area?")
        area_q.add_updater(lambda m: m.move_to(shadow))
        self.play(Write(area_q))
        self.wait()

        # Orient straight up
        unit_normal = face.get_unit_normal()
        axis = rotate_vector(normalize([*unit_normal[:2], 0]), PI / 2, OUT)
        angle = np.arccos(unit_normal[2])
        face.generate_target()
        face.target.rotate(-angle, axis)
        face.target.move_to(3 * OUT)
        face.target.rotate(-PI / 4, OUT)
        self.play(MoveToTarget(face))

        light_lines = self.get_light_lines(n_lines=4, outline=shadow, only_vertices=True)
        light_lines.set_stroke(YELLOW, 1, 0.5)

        self.play(
            frame.animate.set_phi(70 * DEGREES),
            FadeIn(light_lines, lag_ratio=0.5),
            TransformFromCopy(face, face.deepcopy().set_opacity(0).set_z(0), remover=True),
            run_time=3,
        )
        self.wait(3)
        self.play(
            Rotate(face, PI / 2, UP),
            FadeOut(area_q, scale=0),
            run_time=3,
        )
        self.wait(3)
        self.play(
            Rotate(face, -PI / 3, UP),
            UpdateFromAlphaFunc(light_lines, lambda m, a: m.set_opacity(0.5 * (1 - a)), remover=True),
            run_time=2,
        )

        # Show normal vector
        z_axis = VGroup(
            Line(ORIGIN, face.get_center()),
            Line(face.get_center(), 10 * OUT),
        )
        z_axis.set_stroke(WHITE, 1)

        normal_vect = Vector()
        get_fc = face.get_center
        get_un = face.get_unit_normal

        def get_theta():
            return np.arccos(get_un()[2])

        normal_vect.add_updater(lambda v: v.put_start_and_end_on(
            get_fc(), get_fc() + get_un(),
        ))
        arc = always_redraw(lambda: Arc(
            start_angle=PI / 2,
            angle=-get_theta(),
            radius=0.5,
            stroke_width=2,
        ).rotate(PI / 2, RIGHT, about_point=ORIGIN).shift(get_fc()))
        theta = Tex("\\theta", font_size=30)
        theta.rotate(PI / 2, RIGHT)
        theta.add_updater(lambda m: m.move_to(
            get_fc() + 1.3 * (arc.pfp(0.5) - get_fc())
        ))
        theta.add_updater(lambda m: m.set_width(min(0.123, max(0.01, arc.get_width()))))

        self.play(ShowCreation(normal_vect))
        self.wait()
        self.add(z_axis[0], face, z_axis[1], normal_vect)
        self.play(*map(FadeIn, z_axis))
        self.play(
            FadeIn(theta, 0.5 * OUT), ShowCreation(arc),
            frame.animate.reorient(-10),
        )
        self.wait(5)

        # Show shadow area in the corner
        axes = Axes(
            (0, 180, 22.5), (0, 1, 0.25),
            width=5,
            height=2,
            axis_config={
                "include_tip": False,
                "tick_size": 0.05,
                "numbers_to_exclude": [],
            },
        )
        axes.to_corner(UR, buff=MED_SMALL_BUFF)
        axes.x_axis.add_numbers([0, 45, 90, 135, 180], unit="^\\circ")
        y_label = TexText("Shadow's area", font_size=24)
        y_label.next_to(axes.y_axis.get_top(), RIGHT, MED_SMALL_BUFF)
        ly_label = Tex("s^2", font_size=24)
        ly_label.next_to(axes.y_axis.get_top(), LEFT, SMALL_BUFF)
        ly_label.shift(0.05 * UP)
        axes.add(y_label, ly_label)
        axes.fix_in_frame()

        graph = axes.get_graph(
            lambda x: math.cos(x * DEGREES),
            x_range=(0, 90),
        )
        graph.set_stroke(RED, 3)
        graph.fix_in_frame()

        question = Text("Can you guess?", font_size=36)
        question.to_corner(UR)
        question.set_color(RED)

        dot = Dot(color=RED)
        dot.scale(0.5)
        dot.move_to(axes.c2p(0, 1))
        dot.fix_in_frame()

        self.play(
            FadeIn(axes),
            Rotate(face, -get_theta(), UP, run_time=2),
        )
        self.play(FadeIn(dot, shift=2 * UP + RIGHT))
        self.wait(2)
        self.add(graph, axes)
        self.play(
            MoveAlongPath(dot, graph.deepcopy()),
            ShowCreation(graph),
            Rotate(face, PI / 2, UP),
            run_time=5
        )
        self.play(frame.animate.reorient(15), run_time=2)
        self.play(frame.animate.reorient(-15), run_time=4)

        self.embed()


# Older scenes
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
