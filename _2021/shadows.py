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
        color = interpolate_color(sm.get_color(), BLACK, opacity)
        sm.set_color(color)
        sm.set_opacity(opacity)
        if isinstance(sm, VMobject):
            sm.set_stroke(
                interpolate_color(sm.get_stroke_color(), BLACK, opacity)
            )
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


def sort_to_camera(mobject, camera_frame):
    cl = camera_frame.get_implied_camera_location()
    mobject.sort(lambda p: -get_norm(p - cl))
    for sm in mobject:
        sm.refresh_unit_normal()
    return mobject


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

        grid = NumberPlane(
            x_range=(-width // 2, width // 2, 2),
            y_range=(-8, height // 2, 2),
            background_line_style={
                "stroke_color": GREY_B,
                "stroke_width": 1,
            },
            faded_line_ratio=4,
        )
        grid.shift(-grid.get_origin())
        grid.set_width(width)
        grid.axes.match_style(grid.background_lines)
        grid.set_flat_stroke(True)
        grid.insert_n_curves(3)

        plane = Rectangle()
        plane.replace(grid, stretch=True)
        plane.set_style(**self.plane_style)
        plane.set_stroke(width=0)
        self.plane = plane

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
        return sort_to_camera(mobject, self.camera.frame)

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

    def begin_ambient_rotation(self, mobject, speed=0.2, about_point=None):
        mobject.rot_axis = np.array([1, 1, 1])

        def update_mob(mob, dt):
            mob.rotate(speed * dt, mob.rot_axis, about_point=about_point)
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

        def update_lines(lines):
            lp = self.light.get_center()
            if only_vertices:
                points = outline.get_vertices()
            else:
                points = [outline.pfp(a) for a in np.linspace(0, 1, n_lines)]
            for line, point in zip(lines, points):
                if self.inf_light:
                    line.set_points_as_corners([point + 10 * OUT, point])
                else:
                    line.set_points_as_corners([lp, point])

        line = Line(IN, OUT)
        light_lines = line.replicate(n_lines)
        light_lines.set_stroke(YELLOW, 0.5, 0.1)
        light_lines.add_updater(update_lines)
        return light_lines

    def randomly_reorient(self, mobject=None, run_time=1, angle=TAU, about_point=None):
        if mobject is None:
            mobject = self.solid

        mobject.rot_axis = normalize(np.random.random(3))
        mobject.rot_time = 0

        def update(mob, time):
            dt = time - mob.rot_time
            mob.rot_time = time
            mob.rot_axis = rotate_vector(mob.rot_axis, 5 * dt, normalize(np.random.random(3)))
            mob.rotate(angle * dt, mob.rot_axis, about_point=about_point)

        self.play(UpdateFromAlphaFunc(mobject, update), run_time=run_time)


class IntroduceShadow(ShadowScene):
    area_label_center = [-2.5, -2, 0]
    plane_dims = (32, 20)

    def construct(self):
        # Setup
        light = self.light
        light.move_to([-2, 2, 10])
        cube = self.solid
        cube.scale(0.945)  # Hack to make the appropriate area 1
        shadow = self.shadow
        outline = self.get_shadow_outline()
        frame = self.camera.frame
        frame.add_updater(lambda f, dt: f.increment_theta(0.01 * dt))  # Ambient rotation
        area_label = self.get_shadow_area_label()
        light_lines = self.get_light_lines(outline)

        # Introductory animations
        self.shadow.update()
        self.play(
            *(
                LaggedStartMap(DrawBorderThenFill, mob, lag_ratio=0.1, run_time=3)
                for mob in (cube, shadow)
            )
        )
        self.wait(1)
        self.play(
            light.animate.next_to(cube, LEFT + OUT, buff=2),
            run_time=2,
        )
        light_lines.update()
        area_label.update()
        self.play(
            FadeIn(area_label, lag_ratio=0.1),
            ShowCreation(outline, run_time=3, rate_func=linear),
            ShowCreation(light_lines, lag_ratio=0.01, run_time=3),
        )
        self.wait(2)

        # Change size and orientation
        self.play(
            cube.animate.scale(0.5),
            run_time=2,
            rate_func=there_and_back,
        )
        self.randomly_reorient(run_time=2, angle=PI)
        self.wait()
        self.play(
            light.animate.set_x(0),
            run_time=5,
        )
        self.wait()
        self.begin_ambient_rotation(cube)
        self.play(light.animate.shift(IN), run_time=2)
        self.wait()
        self.play(light.animate.shift(OUT), run_time=2)
        self.wait(2)

        # Ask question
        question = TexText(
            "Puzzle: Find the average\\\\area of a cube's shadow",
            font_size=48,
        )
        question.to_corner(UL)
        question.fix_in_frame()
        subquestion = Text("(Averaged over all orientations)")
        subquestion.match_width(question)
        subquestion.next_to(question, DOWN, MED_LARGE_BUFF)
        subquestion.set_fill(GREY_B)
        subquestion.fix_in_frame()

        self.play(FadeIn(question, UP))
        self.wait()
        self.play(Write(subquestion))
        self.wait(8)

        # Where is the light?
        light_comment = Text("Where is the light?")
        light_comment.set_color(YELLOW)
        light_comment.to_corner(UR)
        light_comment.fix_in_frame()

        self.play(FadeIn(light_comment, 0.5 * UP))
        self.play(
            light.animate.next_to(cube, OUT, 1.0),
            run_time=3,
        )
        self.play(light.animate.shift(2 * OUT + 4 * RIGHT), run_time=4)
        self.wait(2)
        self.play(
            frame.animate.set_height(12).set_z(5),
            light.animate.next_to(cube, OUT, buff=7),
            run_time=3,
        )
        self.wait()
        self.play(light.animate.move_to(75 * OUT), run_time=3)
        self.wait()
        self.play(
            frame.animate.set_height(8).set_z(2),
            LaggedStart(*map(FadeOut, (question, subquestion, light_comment))),
            run_time=2
        )

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

        self.play(Rotate(cube, -theta, axis))
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
        frame.target.set_height(6.5)

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

        diagonal = Line(cube.get_nadir(), cube.get_zenith())
        diagonal.set_stroke(WHITE, 2)
        diagonal.scale(2)
        diagonal.move_to(ORIGIN, IN)
        self.add(diagonal, cube)
        self.play(ShowCreation(diagonal))

        hex_area_label = Tex("\\sqrt{3} s^2")
        hex_area_label.set_color(RED)
        hex_area_label.move_to(self.shadow)
        hex_area_label.shift(0.35 * DOWN)
        self.play(Write(hex_area_label))
        self.wait(10)
        area_label.resume_updating()
        self.play(
            Uncreate(diagonal),
            FadeOut(hex_area_label),
            Rotate(cube, 4, RIGHT)
        )

        # Talk about averages
        light_lines.clear_updaters()
        self.play(
            FadeOut(light_lines),
            FadeIn(question, 0.5 * UP),
            ApplyMethod(frame.set_height, 8, run_time=2)
        )
        self.wait()
        self.play(FadeIn(subquestion, 0.5 * UP))
        self.wait()

        samples = VGroup(VectorizedPoint())
        samples.to_corner(UR)
        samples.shift(1.5 * LEFT)
        self.add(samples)
        for x in range(9):
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

        for x in range(2):
            self.wait()
            self.randomly_reorient()
        self.play(FadeIn(words[0], scale=0.75, run_time=0.5))
        self.wait()

        # Just one face
        cube.update()
        index = np.argmax([f.get_z() for f in cube])
        face = cube[index]
        prev_opacity = face.get_fill_opacity()
        cube.generate_target(use_deepcopy=True)
        cube.target.clear_updaters()
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

        def get_un():
            return face.get_unit_normal(recompute=True)

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
        theta.set_backstroke()
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
            UpdateFromFunc(dot, lambda d: d.move_to(graph.get_end())),
            ShowCreation(graph),
            Rotate(face, PI / 2, UP),
            run_time=5
        )
        self.play(frame.animate.reorient(45), run_time=2)
        self.play(frame.animate.reorient(5), run_time=4)

        # Show vertical plane
        plane = Rectangle(width=self.plane.get_width(), height=5)
        plane.insert_n_curves(100)
        plane.set_fill(WHITE, 0.25)
        plane.set_stroke(width=0)
        plane.apply_depth_test()

        plane.rotate(PI / 2, RIGHT)
        plane.move_to(ORIGIN, IN)
        plane.save_state()
        plane.stretch(0, 2, about_edge=IN)

        face.apply_depth_test()
        self.shadow.apply_depth_test()

        self.play(
            LaggedStartMap(FadeOut, VGroup(*words, graph, axes, dot)),
            Restore(plane, run_time=3)
        )
        self.play(Rotate(face, -60 * DEGREES, UP, run_time=2))

        # Slice up face
        face_copy = face.deepcopy()
        face_copy.rotate(-get_theta(), UP)
        face_copy.move_to(ORIGIN)

        n_slices = 25
        rects = Rectangle().replicate(n_slices)
        rects.arrange(DOWN, buff=0)
        rects.replace(face_copy, stretch=True)
        slices = VGroup(*(Intersection(face_copy, rect) for rect in rects))
        slices.match_style(face_copy)
        slices.set_stroke(width=0)
        slices.rotate(get_theta(), UP)
        slices.move_to(face)
        slices.apply_depth_test()
        slices.save_state()
        slice_outlines = slices.copy()
        slice_outlines.set_stroke(RED, 1)
        slice_outlines.set_fill(opacity=0)
        slice_outlines.deactivate_depth_test()

        frame.clear_updaters()
        self.play(
            frame.animate.set_euler_angles(PI / 2, get_theta()),
            FadeOut(VGroup(theta, arc)),
            run_time=2
        )
        self.play(ShowCreation(slice_outlines, lag_ratio=0.05))

        self.remove(face)
        self.add(slices)
        self.remove(self.shadow)
        self.solid = slices
        self.add_shadow()
        self.shadow.set_stroke(width=0)
        self.add(normal_vect, plane, slice_outlines)

        slices.insert_n_curves(10)
        slices.generate_target()
        for sm in slices.target:
            sm.stretch(0.5, 1)
        self.play(
            MoveToTarget(slices),
            FadeOut(slice_outlines),
            run_time=2
        )
        self.wait(2)

        # Focus on one slice
        long_slice = slices[len(slices) // 2].deepcopy()
        line = Line(long_slice.get_corner(LEFT + OUT), long_slice.get_corner(RIGHT + IN))
        line.scale(0.97)
        line.set_stroke(BLUE, 3)

        frame.generate_target()
        frame.target.reorient(0, 90)
        frame.target.set_height(6)
        frame.target.move_to(2.5 * OUT)
        self.shadow.clear_updaters()
        self.play(
            MoveToTarget(frame),
            *map(FadeIn, (theta, arc)),
            FadeOut(plane),
            FadeOut(slices),
            FadeOut(self.shadow),
            FadeIn(line),
            run_time=2,
        )
        self.wait()

        # Analyze slice
        shadow = line.copy()
        shadow.stretch(0, 2, about_edge=IN)
        shadow.set_stroke(BLUE_E)
        vert_line = Line(line.get_start(), shadow.get_start())
        vert_line.set_stroke(GREY_B, 3)

        shadow_label = Text("Shadow")
        shadow_label.set_fill(BLUE_E)
        shadow_label.set_backstroke()
        shadow_label.rotate(PI / 2, RIGHT)
        shadow_label.next_to(shadow, IN, SMALL_BUFF)

        self.play(
            TransformFromCopy(line, shadow),
            FadeIn(shadow_label, 0.5 * IN),
        )
        self.wait()
        self.play(ShowCreation(vert_line))
        self.wait()

        top_theta_group = VGroup(
            z_axis[1].copy(),
            arc.copy().clear_updaters(),
            theta.copy().clear_updaters(),
            Line(*normal_vect.get_start_and_end()).match_style(z_axis[1].copy()),
        )
        self.play(
            top_theta_group.animate.move_to(line.get_start(), LEFT + IN)
        )

        elbow = Elbow(angle=-get_theta())
        elbow.set_stroke(WHITE, 2)
        ul_arc = Arc(
            radius=0.4,
            start_angle=-get_theta(),
            angle=-(PI / 2 - get_theta())
        )
        ul_arc.match_style(elbow)
        supl = Tex("90^\\circ - \\theta", font_size=24)
        supl.next_to(ul_arc, DOWN, SMALL_BUFF, aligned_edge=LEFT)
        supl.set_backstroke()
        supl[0][:3].shift(SMALL_BUFF * RIGHT / 2)

        ul_angle_group = VGroup(elbow, ul_arc, supl)
        ul_angle_group.rotate(PI / 2, RIGHT, about_point=ORIGIN)
        ul_angle_group.shift(line.get_start())

        dr_arc = Arc(
            radius=0.4,
            start_angle=PI,
            angle=-get_theta(),
        )
        dr_arc.match_style(ul_arc)
        dr_arc.rotate(PI / 2, RIGHT, about_point=ORIGIN)
        dr_arc.shift(line.get_end())
        dr_theta = Tex("\\theta", font_size=24)
        dr_theta.rotate(PI / 2, RIGHT)
        dr_theta.next_to(dr_arc, LEFT, SMALL_BUFF)
        dr_theta.shift(SMALL_BUFF * OUT / 2)

        self.play(ShowCreation(elbow))
        self.play(
            ShowCreation(ul_arc),
            FadeTransform(top_theta_group[2].copy(), supl),
        )
        self.play(
            TransformFromCopy(ul_arc, dr_arc),
            TransformFromCopy(supl[0][4].copy().set_stroke(width=0), dr_theta[0][0]),
        )
        self.wait()

        # Highlight lower right
        rect = Rectangle(0.8, 0.5)
        rect.set_stroke(YELLOW, 2)
        rect.rotate(PI / 2, RIGHT)
        rect.move_to(dr_theta, LEFT).shift(SMALL_BUFF * LEFT)

        self.play(
            ShowCreation(rect),
            top_theta_group.animate.fade(0.8),
            ul_angle_group.animate.fade(0.8),
        )
        self.wait()

        # Show cosine
        cos_formula = Tex(
            "\\cos(\\theta)", "=",
            "{\\text{Length of }", "\\text{shadow}",
            "\\over",
            "\\text{Length of }", "\\text{slice}"
            "}",
        )
        cos_formula[2:].scale(0.75, about_edge=LEFT)
        cos_formula.to_corner(UR)
        cos_formula.fix_in_frame()

        lower_formula = Tex(
            "\\text{shadow}", "=",
            "\\cos(\\theta)", "\\cdot", "\\text{slice}"
        )
        lower_formula.match_width(cos_formula)
        lower_formula.next_to(cos_formula, DOWN, MED_LARGE_BUFF)
        lower_formula.fix_in_frame()

        for tex in cos_formula, lower_formula:
            tex.set_color_by_tex("shadow", BLUE_D)
            tex.set_color_by_tex("slice", BLUE_B)

        self.play(Write(cos_formula))
        self.wait()
        self.play(TransformMatchingTex(
            VGroup(*(cos_formula[i].copy() for i in [0, 1, 3, 6])),
            lower_formula,
            path_arc=PI / 4,
        ))
        self.wait()

        # Bring full face back
        frame.generate_target()
        frame.target.reorient(20, 75)
        frame.target.set_height(6)
        frame.target.set_z(2)

        line_shadow = get_shadow(line)
        line_shadow.set_stroke(BLUE_E, opacity=0.5)

        self.solid = face
        self.add_shadow()
        self.add(z_axis[0], face, z_axis[1], line, normal_vect, theta, arc)
        self.play(
            MoveToTarget(frame, run_time=5),
            FadeIn(face, run_time=3),
            FadeIn(self.shadow, run_time=3),
            FadeIn(line_shadow, run_time=3),
            LaggedStart(*map(FadeOut, [
                top_theta_group, ul_angle_group, rect,
                dr_theta, dr_arc,
                vert_line, shadow, shadow_label,
            ]), run_time=4),
        )
        frame.add_updater(lambda f, dt: f.increment_theta(0.01 * dt))
        self.wait(2)

        # Show perpendicular
        perp = Line(
            face.pfp(binary_search(
                lambda a: face.pfp(a)[2],
                face.get_center()[2], 0, 0.5,
            )),
            face.pfp(binary_search(
                lambda a: face.pfp(a)[2],
                face.get_center()[2], 0.5, 1.0,
            )),
        )
        perp.set_stroke(RED, 3)
        perp_shadow = get_shadow(perp)
        perp_shadow.set_stroke(RED_E, 3, opacity=0.2)

        self.add(perp, normal_vect, arc)
        self.play(
            ShowCreation(perp),
            ShowCreation(perp_shadow),
        )
        face.add(line)
        self.play(Rotate(face, 45 * DEGREES, UP), run_time=3)
        self.play(Rotate(face, -55 * DEGREES, UP), run_time=3)
        self.play(Rotate(face, 20 * DEGREES, UP), run_time=2)

        # Give final area formula
        final_formula = Tex(
            "\\text{Area}(", "\\text{shadow}", ")",
            "=",
            "|", "\\cos(\\theta)", "|", "s^2"
        )
        final_formula.set_color_by_tex("shadow", BLUE_D)
        final_formula.match_width(lower_formula)
        final_formula.next_to(lower_formula, DOWN, MED_LARGE_BUFF)
        final_formula.fix_in_frame()
        final_formula.get_parts_by_tex("|").set_opacity(0)
        final_formula.set_stroke(BLACK, 3, background=True)
        rect = SurroundingRectangle(final_formula)
        rect.set_stroke(YELLOW, 2)
        rect.fix_in_frame()

        self.play(Write(final_formula))
        self.play(ShowCreation(rect))
        final_formula.add(rect)
        self.wait(10)

        # Absolute value
        face.remove(line)
        self.play(
            frame.animate.shift(0.5 * DOWN + RIGHT).reorient(10),
            LaggedStart(*map(FadeOut, [cos_formula, lower_formula])),
            FadeIn(graph),
            FadeIn(axes),
            FadeOut(line),
            FadeOut(line_shadow),
            FadeOut(perp),
            FadeOut(perp_shadow),
            final_formula.animate.shift(2 * DOWN),
            run_time=2
        )
        self.play(
            Rotate(face, PI / 2 - get_theta(), UP),
            run_time=2
        )

        new_graph = axes.get_graph(
            lambda x: math.cos(x * DEGREES),
            (90, 180),
        )
        new_graph.match_style(graph)
        new_graph.fix_in_frame()
        self.play(
            Rotate(face, PI / 2, UP),
            ShowCreation(new_graph),
            run_time=5,
        )
        self.play(
            Rotate(face, -PI / 4, UP),
            run_time=2,
        )
        self.wait(3)

        alt_normal = normal_vect.copy()
        alt_normal.clear_updaters()
        alt_normal.rotate(PI, UP, about_point=face.get_center())
        alt_normal.set_color(YELLOW)

        self.add(alt_normal, face, normal_vect, arc, theta)
        self.play(ShowCreation(alt_normal))
        self.wait()
        self.play(FadeOut(alt_normal))

        new_graph.generate_target()
        new_graph.target.flip(RIGHT)
        new_graph.target.move_to(graph.get_end(), DL)

        self.play(
            MoveToTarget(new_graph),
            final_formula.get_parts_by_tex("|").animate.set_opacity(1),
        )
        self.play(
            final_formula.animate.next_to(axes, DOWN)
        )
        self.wait()
        self.play(Rotate(face, -PI / 2, UP), run_time=5)
        self.wait(10)


class DiscussLinearity(Scene):
    def construct(self):
        pass


# This should maybe include changing shapes and sizes
class AmbientFaceRotation(ShadowScene):
    inf_light = True
    show_3d_perspective = True

    def construct(self):
        # Setup
        cube = self.solid
        frame = self.camera.frame
        frame.set_height(6)
        frame.add_updater(lambda f, dt: f.increment_theta(dt * 0.01))
        light = self.light
        light.move_to(75 * OUT)

        index = np.argmax([f.get_z() for f in cube])
        self.solid = face = cube[index]
        fc = 2.5 * OUT
        face.move_to(fc)
        self.remove(cube, self.shadow)
        self.add(face)
        self.add_shadow()
        shadow = self.shadow
        shadow_fill_opacity = shadow.get_fill_opacity()
        shadow.add_updater(lambda s: s.set_fill(opacity=shadow_fill_opacity))

        if self.show_3d_perspective:
            z_axis = VGroup(
                Line(ORIGIN, fc),
                Line(fc, 10 * OUT),
            )
            z_axis.set_stroke(WHITE, 1)
            self.add(z_axis[0], face, z_axis[1])

            orientation_arrows = VGroup(
                Vector(RIGHT, stroke_color=RED),
                Vector(UP, stroke_color=GREEN),
                Vector(OUT, stroke_color=BLUE),
            )
            orientation_arrows.shift(face.get_center())

            face.add(orientation_arrows[:2])
            face = Group(face, orientation_arrows[2])
            face.add_updater(lambda m: self.sort_to_camera(m))
            self.add(face)
            self.add(get_shadow(orientation_arrows))
        else:
            frame.reorient(0, 0)
            frame.set_height(3)
            frame.clear_updaters()
            fc = 10 * OUT
            face.move_to(fc)

        # Ambient rotation
        self.begin_ambient_rotation(face, about_point=fc)
        self.wait(30)


class AmbientFaceRotationShadowView(AmbientFaceRotation):
    show_3d_perspective = False


class AllPossibleOrientations(ShadowScene):
    inf_light = True
    plane_dims = (16, 12)

    def construct(self):
        # Setup
        frame = self.camera.frame
        frame.reorient(-10, 80)
        frame.set_height(5)
        frame.add_updater(lambda f, dt: f.increment_theta(0.02 * dt))
        face = self.solid
        square, normal_vect = face
        self.solid = square
        self.remove(self.shadow)
        self.add_shadow()
        self.shadow.deactivate_depth_test()
        self.solid = face
        fc = square.get_center()

        # Sphere points
        sphere = Sphere(radius=1)
        sphere.set_color(WHITE, 0.5)
        sphere.move_to(fc)
        sphere.always_sort_to_camera(self.camera)

        n_lat_lines = 20
        sphere_points = np.array([
            sphere.uv_func(phi, theta)
            for theta in np.linspace(0, PI, n_lat_lines)
            for phi in random.random() + np.linspace(
                0, TAU, int(2 * n_lat_lines * math.sin(theta))
            )
        ])
        sphere_points[:, 2] *= -1
        original_sphere_points = sphere_points.copy()
        sphere_points += fc

        sphere_dots = DotCloud(sphere_points)
        sphere_dots.set_radius(0.025)
        sphere_dots.set_glow_factor(0.5)
        sphere_dots.make_3d()
        sphere_dots.apply_depth_test()
        sphere_dots.add_updater(lambda m: m)

        face.save_state()

        z_to_vector

        N = len(original_sphere_points)
        self.play(
            ShowCreation(sphere_dots),
            UpdateFromAlphaFunc(
                face,
                lambda m, a: m.apply_matrix(
                    rotation_between_vectors(
                        normal_vect.get_vector(),
                        original_sphere_points[int(a * (N - 1))],
                    ),
                    about_point=fc
                )
            ),
            run_time=10,
            rate_func=linear,
        )

        self.add(sphere_dots)

        # Embed
        self.embed()

    def get_solid(self):
        face = Square(side_length=2)
        face.set_style(**self.object_style)
        face.set_stroke(width=0)
        normal = Vector(OUT)
        normal.shift(2e-2 * OUT)
        face = VGroup(face, normal)
        face.set_stroke(background=True)
        face.apply_depth_test()
        return face
