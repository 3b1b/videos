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
    limited_plane_extension = 0
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
            y_range=(-height // 2, height // 2, 2),
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
        if self.limited_plane_extension > 0:
            plane.set_height(height // 2 + self.limited_plane_extension, about_edge=UP, stretch=True)
        self.plane = plane

        plane.add(grid)
        self.add(plane)

    def add_solid(self):
        self.solid = self.get_solid()
        self.solid.move_to(self.object_center)
        self.add(self.solid)

    def get_solid(self):
        cube = VCube()
        cube.deactivate_depth_test()
        cube.set_height(2)
        cube.set_style(**self.object_style)
        # Wrap in group so that strokes and fills
        # are rendered in separate passes
        cube = self.cube = Group(*cube)
        cube.add_updater(lambda m: self.sort_to_camera(m))
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
        decimal = DecimalNumber(100)

        label = VGroup(text, decimal)
        label.arrange(RIGHT)
        label.move_to(self.area_label_center - decimal.get_center())
        label.fix_in_frame()
        label.set_backstroke()
        decimal.add_updater(lambda d: d.set_value(
            get_area(self.shadow) / (self.unit_size**2)
        ).set_backstroke())
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

    def random_toss(self, mobject=None, angle=TAU, about_point=None, **kwargs):
        if mobject is None:
            mobject = self.solid

        mobject.rot_axis = normalize(np.random.random(3))
        mobject.rot_time = 0

        def update(mob, time):
            dt = time - mob.rot_time
            mob.rot_time = time
            mob.rot_axis = rotate_vector(mob.rot_axis, 5 * dt, normalize(np.random.random(3)))
            mob.rotate(angle * dt, mob.rot_axis, about_point=about_point)

        self.play(
            UpdateFromAlphaFunc(mobject, update),
            **kwargs
        )

    def randomly_reorient(self, solid=None, about_point=None):
        solid = self.solid if solid is None else solid
        solid.rotate(
            random.uniform(0, TAU),
            axis=normalize(np.random.uniform(-1, 1, 3)),
            about_point=about_point,
        )
        return solid

    def init_frame_rotation(self, factor=0.0025, max_speed=0.01):
        frame = self.camera.frame
        frame.d_theta = 0

        def update_frame(frame, dt):
            frame.d_theta += -factor * frame.get_theta()
            frame.increment_theta(clip(
                factor * frame.d_theta,
                -max_speed * dt,
                max_speed * dt
            ))

        frame.add_updater(update_frame)
        return frame


class IntroduceShadow(ShadowScene):
    area_label_center = [-2.5, -2, 0]
    plane_dims = (28, 20)

    def construct(self):
        # Setup
        light = self.light
        light.move_to([0, 0, 20])
        self.add(light)
        cube = self.solid
        cube.scale(0.945)  # Hack to make the appropriate area 1
        shadow = self.shadow
        outline = self.get_shadow_outline()
        frame = self.camera.frame
        frame.add_updater(lambda f, dt: f.increment_theta(0.01 * dt))  # Ambient rotation
        area_label = self.get_shadow_area_label()
        light_lines = self.get_light_lines(outline)

        # Question
        question = TexText(
            "Puzzle: Find the average\\\\area of a cube's shadow",
            font_size=48,
        )
        question.to_corner(UL)
        question.fix_in_frame()
        subquestion = Text("(Averaged over all orientations)")
        subquestion.match_width(question)
        subquestion.next_to(question, DOWN, MED_LARGE_BUFF)
        subquestion.set_fill(BLUE_D)
        subquestion.fix_in_frame()
        subquestion.set_backstroke()

        # Introductory animations
        self.shadow.update()
        self.play(
            FadeIn(question, UP),
            *(
                LaggedStartMap(DrawBorderThenFill, mob, lag_ratio=0.1, run_time=3)
                for mob in (cube, shadow)
            )
        )
        self.random_toss(run_time=3, angle=TAU)

        # Change size and orientation
        outline.update()
        area_label.update()
        self.play(
            FadeIn(area_label),
            ShowCreation(outline),
        )
        self.play(
            cube.animate.scale(0.5),
            run_time=2,
            rate_func=there_and_back,
        )
        self.random_toss(run_time=2, angle=PI)
        self.wait()
        self.begin_ambient_rotation(cube)
        self.play(FadeIn(subquestion, 0.5 * DOWN))
        self.wait(7)

        # Where is the light?
        light_comment = Text("Where is the light?")
        light_comment.set_color(YELLOW)
        light_comment.to_corner(UR)
        light_comment.set_backstroke()
        light_comment.fix_in_frame()

        cube.clear_updaters()
        cube.add_updater(lambda m: self.sort_to_camera(cube))
        self.play(
            FadeIn(light_comment, 0.5 * UP),
            light.animate.next_to(cube, OUT, buff=1.5),
            run_time=2,
        )
        light_lines.update()
        self.play(
            ShowCreation(light_lines, lag_ratio=0.01, run_time=3),
        )
        self.play(
            light.animate.shift(1.0 * IN),
            rate_func=there_and_back,
            run_time=3
        )
        self.play(
            light.animate.shift(4 * RIGHT),
            run_time=10,
            rate_func=there_and_back,
        )
        self.wait()

        # Light straight above
        self.play(
            frame.animate.set_height(12).set_z(4),
            light.animate.set_z(10),
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
        verts = np.array([*cube[0].get_vertices(), *cube[5].get_vertices()])
        vert_dots = DotCloud(verts)
        vert_dots.set_glow_factor(0.5)
        vert_dots.set_color(WHITE)
        proj_dots = vert_dots.copy()
        proj_dots.apply_function(flat_project)
        proj_dots.set_color(GREY_B)
        vert_proj_lines = VGroup(*(
            DashedLine(*pair)
            for pair in zip(verts, proj_dots.get_points())
        ))
        vert_proj_lines.set_stroke(WHITE, 1, 0.5)

        point = verts[np.argmax(verts[:, 0])]
        xyz_label = Tex("(x, y, z)")
        xy0_label = Tex("(x, y, 0)")
        for label in xyz_label, xy0_label:
            label.rotate(PI / 2, RIGHT)
            label.set_backstroke()
        xyz_label.next_to(point, RIGHT)
        xy0_label.next_to(flat_project(point), RIGHT)

        vert_dots.save_state()
        vert_dots.set_glow_factor(5)
        vert_dots.set_radius(0.5)
        vert_dots.set_opacity(0)
        self.play(
            Restore(vert_dots),
            Write(xyz_label),
        )
        self.wait()
        self.play(
            TransformFromCopy(
                cube.deepcopy().clear_updaters().set_opacity(0.5),
                shadow.deepcopy().clear_updaters().set_opacity(0),
                remover=True
            ),
            TransformFromCopy(vert_dots, proj_dots),
            TransformFromCopy(xyz_label, xy0_label),
            *map(ShowCreation, vert_proj_lines),
        )
        self.wait(3)
        self.play(LaggedStart(*map(FadeOut, (
            vert_dots, vert_proj_lines, proj_dots,
            xyz_label, xy0_label
        ))))

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
        self.begin_ambient_rotation(cube)
        self.play(
            FadeOut(light_lines),
            FadeIn(question, 0.5 * UP),
            ApplyMethod(frame.set_height, 8, run_time=2)
        )
        self.play(FadeIn(subquestion, 0.5 * UP))
        self.wait(7)

        cube.clear_updaters()
        cube.add_updater(lambda m: self.sort_to_camera(m))
        samples = VGroup(VectorizedPoint())
        samples.to_corner(UR)
        samples.shift(1.5 * LEFT)
        self.add(samples)
        for x in range(9):
            self.random_toss()
            sample = area_label[1].copy()
            sample.clear_updaters()
            sample.fix_in_frame()
            self.play(
                sample.animate.next_to(samples, DOWN),
                run_time=0.5
            )
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

        for x in range(10):
            self.random_toss()
            self.wait()


class AskAboutAveraging(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        sts = self.students
        tch = self.teacher
        self.play(
            PiCreatureBubbleIntroduction(
                sts[2], TexText("What does that\\\\mean, exactly?"),
                target_mode="hesitant",
                look_at_arg=self.screen,
                bubble_kwargs={"direction": LEFT}
            ),
            LaggedStart(
                sts[0].animate.change("confused", self.screen),
                sts[1].animate.change("pondering", self.screen),
                tch.animate.change("tease", sts[2].eyes),
            )
        )
        self.wait(4)
        self.teacher_says(
            TexText("Ah, very good\\\\question!"),
            target_mode="hooray",
            added_anims=[
                sts[0].animate.change("maybe", tch.eyes),
                sts[1].animate.change("thinking", tch.eyes),
            ]
        )
        self.wait(3)

        # Embed
        self.embed()


class StartSimple(Scene):
    def construct(self):
        # Words
        title = Text("Universal problem-solving advice")
        title.set_width(FRAME_WIDTH - 4)
        title.to_edge(UP)
        title.set_color(BLUE)
        title.set_backstroke()
        line = Underline(title, buff=-0.035)
        line.set_width(FRAME_WIDTH - 1)
        line.set_color(BLUE_B)
        line.set_stroke(width=[0, 3, 3, 3, 0])
        line.insert_n_curves(101)

        words = Text(
            "Start with the simplest non-trivial\n"
            "variant of the problem you can."
        )
        words.next_to(line, DOWN, LARGE_BUFF)

        # Shapes
        cube = VCube()
        cube.deactivate_depth_test()
        cube.set_color(BLUE_E)
        cube.set_opacity(0.75)
        cube.set_stroke(WHITE, 0.5, 0.5)
        cube.set_height(2)
        cube.rotate(PI / 10, [1, 2, 0])
        cube.sort(lambda p: p[2])
        cube = Group(*cube)
        cube.set_gloss(1)

        arrow = Arrow(LEFT, RIGHT)
        face = cube[np.argmax([f.get_z() for f in cube])].copy()
        group = Group(cube, arrow, face)
        group.arrange(RIGHT, buff=MED_LARGE_BUFF)
        group.next_to(words, DOWN, LARGE_BUFF)

        self.camera.light_source.set_x(-4)

        self.play(
            ShowCreation(line),
            Write(title, run_time=1),
        )
        self.play(
            FadeIn(words, DOWN),
        )
        self.play(
            LaggedStart(*map(DrawBorderThenFill, cube))
        )
        self.wait()
        self.play(
            ShowCreation(arrow),
            TransformFromCopy(cube[-1], face)
        )
        self.wait()


class FocusOnOneFace(ShadowScene):
    inf_light = True
    limited_plane_extension = 8

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
        average_words = Text("Average over all orientations")
        average_words.move_to(words[0], LEFT)
        average_words.fix_in_frame()
        self.add(average_words)

        self.random_toss(run_time=3, rate_func=linear)
        self.play(
            FadeIn(words[0], 0.75 * UP),
            FadeOut(average_words, 0.75 * UP),
            run_time=0.5,
        )
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
            FlashAround(words[1], rate_func=squish_rate_func(smooth, 0.2, 0.5)),
            FlashAround(words[0], rate_func=squish_rate_func(smooth, 0.5, 0.8)),
            run_time=5,
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
        y_label.set_backstroke()
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
        z_axis.apply_depth_test()
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


class NotQuiteRight(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        self.teacher_says(
            "Not quite right...",
            target_mode="hesitant",
            bubble_kwargs={"height": 3, "width": 4},
            added_anims=[
                self.get_student_changes(
                    "pondering", "thinking", "erm",
                    look_at_arg=self.screen,
                )
            ]
        )
        self.wait(4)


class DiscussLinearity(Scene):
    def construct(self):
        # Set background
        background = FullScreenRectangle()
        self.add(background)
        panels = Rectangle(4, 4).replicate(3)
        panels.set_fill(BLACK, 1)
        panels.set_stroke(WHITE, 2)
        panels.set_height(FRAME_HEIGHT - 1)
        panels.arrange(RIGHT, buff=LARGE_BUFF)
        panels.set_width(FRAME_WIDTH - 1)
        panels.center()
        self.add(panels)

        # Arrows
        arrows = VGroup(*(
            Arrow(
                p1.get_top(), p2.get_top(), path_arc=-0.6 * PI
            ).scale(0.75, about_edge=DOWN)
            for p1, p2 in zip(panels, panels[1:])
        ))
        arrows.space_out_submobjects(0.8)
        arrows.rotate(PI, RIGHT, about_point=panels.get_center())
        arrow_labels = VGroup(
            Text("Rotation", font_size=30),
            Text("Flat projection", font_size=30),
        )
        arrow_labels.set_backstroke()
        for arrow, label in zip(arrows, arrow_labels):
            label.next_to(arrow.pfp(0.5), UP, buff=0.35)

        shape_labels = VGroup(
            Text("Some shape"),
            Text("Any shape"),
        )
        shape_labels.next_to(panels[0].get_top(), UP, SMALL_BUFF)

        self.play(Write(shape_labels[0], run_time=1))
        self.wait()
        self.play(
            FadeTransform(
                shape_labels[0].get_part_by_text("Some"),
                shape_labels[1].get_part_by_text("Any"),
            ),
            FadeTransform(
                shape_labels[0].get_part_by_text("shape"),
                shape_labels[1].get_part_by_text("shape"),
            ),
        )
        self.wait(2)

        for arrow, label in zip(arrows, arrow_labels):
            self.play(
                ShowCreation(arrow),
                FadeIn(label, lag_ratio=0.1)
            )
            self.wait()

        # Linear!
        lin_text = Text(
            "All linear transformations!",
            t2c={"linear": YELLOW}
        )
        lin_text.next_to(panels, UP, MED_SMALL_BUFF)

        self.play(
            FadeOut(shape_labels[1]),
            FadeIn(lin_text, lag_ratio=0.1)
        )
        self.wait()

        # Stretch words
        uniform_words = Text("Uniform stretching here", font_size=36).replicate(2)
        for words, panel in zip(uniform_words, panels[0::2]):
            words.next_to(panel.get_top(), DOWN, SMALL_BUFF)
            words.set_color(YELLOW)
            words.set_backstroke()
            self.play(
                FadeIn(words, lag_ratio=0.1),
            )
            self.wait()

        # Transition
        lin_part = lin_text.get_part_by_text("linear")
        lin_copies = lin_part.copy().replicate(2)
        lin_copies.scale(0.6)
        for lin_copy, arrow in zip(lin_copies, arrows):
            lin_copy.next_to(arrow.pfp(0.5), DOWN, buff=0.15)

        self.play(
            TransformFromCopy(lin_part.replicate(2), lin_copies),
            LaggedStart(
                FadeOut(lin_text, lag_ratio=0.1),
                *map(FadeOut, uniform_words)
            )
        )

        # Areas
        area_labels = VGroup(
            Text("Area(shape)", t2c={"shape": BLUE}),
            Text("Area(shadow)", t2c={"shadow": BLUE_E}),
        )
        area_exprs = VGroup(
            Tex("A").set_color(BLUE),
            Tex("(\\text{some factor})", "\\cdot ", "A"),
        )
        area_exprs[1][2].set_color(BLUE)
        area_exprs[1][0].set_color(GREY_C)
        equals = VGroup()
        for label, expr, panel in zip(area_labels, area_exprs, panels[0::2]):
            label.match_x(panel)
            label.to_edge(UP, buff=MED_SMALL_BUFF)
            eq = Tex("=")
            eq.rotate(PI / 2)
            eq.next_to(label, DOWN, buff=0.15)
            equals.add(eq)
            expr.next_to(eq, DOWN, buff=0.15)

        self.play(
            *map(Write, area_labels),
            run_time=1
        )
        self.play(
            *(FadeIn(eq, 0.5 * DOWN) for eq in equals),
            *(FadeIn(expr, DOWN) for expr in area_exprs),
        )
        self.wait()

        f_rot = Tex("f(\\text{Rot})")
        f_rot.set_color(GREY_B)
        times_A = area_exprs[1][1:]
        f_rot.next_to(times_A, LEFT, buff=0.2)
        times_A.generate_target()
        VGroup(f_rot, times_A.target).match_x(panels[2])

        self.play(
            FadeTransform(area_exprs[1][0], f_rot),
            MoveToTarget(times_A)
        )
        self.play(ShowCreationThenFadeAround(f_rot, run_time=2))
        self.wait(1)

        # Cross out right
        cross = Cross(VGroup(equals[1], f_rot, times_A))
        cross.insert_n_curves(20)
        self.play(ShowCreation(cross))
        self.wait(3)


class AmbientShapeRotationPreimage(ShadowScene):
    inf_light = False
    display_mode = "preimage_only"  # Or "full_3d" or "shadow_only"
    rotate_in_3d = True
    only_show_shadow = False

    def construct(self):
        # Setup
        display_mode = self.display_mode
        frame = self.camera.frame
        frame.set_height(6)
        light = self.light
        light.move_to(75 * OUT)

        shape = self.solid
        fc = 2.5 * OUT
        shape.move_to(fc)

        self.solid.rotate(-0.5 * PI)
        self.solid.insert_n_curves(20)
        preimage = self.solid.deepcopy()
        preimage.move_to(ORIGIN)
        rotated = self.solid

        self.remove(self.shadow)
        shadow = rotated.deepcopy()
        shadow.set_fill(interpolate_color(BLUE_E, BLACK, 0.5), 0.7)
        shadow.set_stroke(BLACK, 1)

        def update_shadow(shadow):
            shadow.set_points(
                np.apply_along_axis(
                    lambda p: project_to_xy_plane(self.light.get_center(), p),
                    1, rotated.get_points()
                )
            )
            shadow.refresh_triangulation()
            return shadow

        shadow.add_updater(update_shadow)

        rotated.axis_tracker = VectorizedPoint(RIGHT)
        rotated.angle_tracker = ValueTracker(0)
        rotated.rot_speed_tracker = ValueTracker(0.15)

        def update_rotated(mob, dt):
            mob.set_points(preimage.get_points())
            mob.shift(fc)
            mob.refresh_triangulation()
            axis = mob.axis_tracker.get_location()
            angle = mob.angle_tracker.get_value()
            speed = mob.rot_speed_tracker.get_value()
            mob.axis_tracker.rotate(speed * dt, axis=OUT, about_point=ORIGIN)
            mob.angle_tracker.increment_value(speed * dt)
            mob.rotate(angle, axis, about_point=fc)
            return rotated

        rotated.add_updater(update_rotated)

        # Conditionals
        if display_mode == "full_3d":
            preimage.set_opacity(0)
            self.add(shadow)
            self.add(rotated)

            z_axis = VGroup(
                Line(ORIGIN, fc),
                Line(fc, 10 * OUT),
            )
            z_axis.set_stroke(WHITE, 1)
            self.add(z_axis[0], rotated, z_axis[1])

            orientation_arrows = VGroup(
                Vector(RIGHT, stroke_color=RED_D),
                Vector(UP, stroke_color=GREEN_D),
                Vector(OUT, stroke_color=BLUE_D),
            )

            orientation_arrows.set_stroke(opacity=0.85)
            orientation_arrows.shift(fc)
            orientation_arrows.save_state()
            orientation_arrows.add_updater(lambda m: m.restore().rotate(
                rotated.angle_tracker.get_value(),
                rotated.axis_tracker.get_location(),
            ))
            orientation_arrows.add_updater(lambda m: m.shift(fc - m[0].get_start()))
            orientation_arrows.apply_depth_test()
            self.add(orientation_arrows)

            proj_lines = always_redraw(lambda: VGroup(*(
                Line(
                    rotated.pfp(a),
                    flat_project(rotated.pfp(a))
                ).set_stroke(WHITE, 0.5, 0.2)
                for a in np.linspace(0, 1, 100)
            )))
            self.add(proj_lines)

            frame.reorient(20, 70)
            frame_speed = -0.02
            frame.add_updater(lambda f, dt: f.increment_theta(frame_speed * dt))
        elif display_mode == "shadow_only":
            frame.reorient(0, 0)
            frame.set_height(3)
            rotated.set_opacity(0)
            preimage.set_opacity(0)
            self.glow.set_opacity(0.2)
            self.add(rotated)
            self.add(shadow)
        elif display_mode == "preimage_only":
            self.glow.set_opacity(0)
            self.remove(self.plane)
            self.add(preimage)
            frame.reorient(0, 0)
            frame.set_height(3)
            rotated.set_opacity(0)

        # Change shape
        self.wait(2)
        cat = SVGMobject("cat_outline").family_members_with_points()[0]
        dog = SVGMobject("dog_outline").family_members_with_points()[0]
        dog.insert_n_curves(87)
        for mob in cat, dog:
            mob.match_style(preimage)
            mob.replace(preimage, dim_to_match=0)
        self.play(
            Transform(preimage, cat, run_time=4),
        )
        cat.insert_n_curves(87)
        preimage.become(cat)
        self.wait(7)

        # Stretch
        self.play(rotated.rot_speed_tracker.animate.set_value(0))
        rotated.rot_speed = 0
        for axis in (0, 1):
            self.play(
                preimage.animate.stretch(3, axis),
                rate_func=there_and_back,
                run_time=4
            )
        self.wait(10)
        self.play(rotated.rot_speed_tracker.animate.set_value(0.1))
        self.wait(10)

        # More shape changes
        self.play(
            preimage.animate.scale(2),
            rate_func=there_and_back,
            run_time=3,
        )
        self.play(
            preimage.animate.become(dog),
            path_arc=PI,
            rate_func=there_and_back_with_pause,
            run_time=5,
        )
        self.wait(6)

        # Bring light source closer
        self.play(rotated.rot_speed_tracker.animate.set_value(0))
        anims = [
            light.animate.move_to(4 * OUT)
        ]

        angle = rotated.angle_tracker.get_value()
        angle_anim = rotated.angle_tracker.animate.set_value(np.round(angle / TAU, 0) * TAU)

        if self.display_mode == "full_3d":
            light_lines = self.get_light_lines(shadow)
            lso = light_lines[0].get_stroke_opacity()
            pso = proj_lines[0].get_stroke_opacity()
            proj_lines.clear_updaters()
            anims += [
                UpdateFromAlphaFunc(proj_lines, lambda m, a: m.set_stroke(opacity=pso * (1 - a))),
                UpdateFromAlphaFunc(light_lines, lambda m, a: m.set_stroke(opacity=lso * a)),
                angle_anim,
                frame.animate.reorient(20, 70),
            ]
            frame.clear_updaters()
        if self.display_mode == "shadow_only":
            anims += [
                frame.animate.set_height(8),
                angle_anim,
            ]
        self.play(*anims, run_time=4)
        self.wait()
        rotated.axis_tracker.move_to(UP)
        self.play(
            rotated.angle_tracker.animate.set_value(70 * DEGREES),
            run_time=3
        )
        self.play(
            preimage.animate.stretch(1.5, 0),
            rate_func=there_and_back,
            run_time=5,
        )
        anims = [rotated.axis_tracker.animate.move_to(RIGHT)]
        if self.display_mode == "full_3d":
            anims.append(frame.animate.reorient(-20, 70))
        self.play(*anims, run_time=2)
        self.play(
            preimage.animate.stretch(2, 1),
            rate_func=there_and_back,
            run_time=7,
        )

        # More ambient motion
        self.play(rotated.rot_speed_tracker.animate.set_value(0.1))
        self.wait(30)

    def get_solid(self):
        face = Square(side_length=2)
        face.set_fill(BLUE, 0.5)
        face.set_stroke(WHITE, 1)
        return face


class AmbientShapeRotationFull3d(AmbientShapeRotationPreimage):
    display_mode = "full_3d"


class AmbientShapeRotationShadowOnly(AmbientShapeRotationPreimage):
    display_mode = "shadow_only"


class IsntThatObvious(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            TexText("Isn't that obvious?"),
            bubble_kwargs={
                "height": 3,
                "width": 4,
                "direction": LEFT,
            },
            target_mode="angry",
            look_at_arg=self.screen,
            added_anims=[LaggedStart(
                self.teacher.animate.change("guilty"),
                self.students[0].animate.change("pondering", self.screen),
                self.students[1].animate.change("erm", self.screen),
            )]
        )
        self.wait(2)
        self.play(
            self.students[0].animate.change("hesitant"),
        )
        self.wait(2)


class WonderAboutAverage(Scene):
    def construct(self):
        randy = Randolph()
        randy.to_edge(DOWN)
        randy.look(RIGHT)
        self.play(PiCreatureBubbleIntroduction(
            randy, TexText("How do you think\\\\about this average"),
            target_mode="confused",
            run_time=2
        ))
        for x in range(2):
            self.play(Blink(randy))
            self.wait(2)


class SingleFaceRandomRotation(ShadowScene):
    initial_wait_time = 0
    inf_light = True
    n_rotations = 1
    total_time = 60
    plane_dims = (8, 8)
    frame_rot_speed = 0.02
    theta0 = -20 * DEGREES
    CONFIG = {"random_seed": 0}

    def construct(self):
        np.random.seed(self.random_seed)
        frame = self.camera.frame
        frame.set_height(5.0)
        frame.set_z(1.75)
        frame.set_theta(self.theta0)
        face = self.solid
        face.shift(0.25 * IN)
        fc = face.get_center()
        z_axis = VGroup(Line(ORIGIN, fc), Line(fc, 10 * OUT))
        z_axis.set_stroke(WHITE, 0.5)
        self.add(z_axis[0], face, z_axis[1])

        arrows = VGroup(
            Line(ORIGIN, RIGHT, color=RED_D),
            Line(ORIGIN, UP, color=GREEN_D),
            VGroup(
                Vector(OUT, stroke_width=4, stroke_color=BLACK),
                Vector(OUT, stroke_width=3, stroke_color=BLUE_D),
            )
        )
        arrows[:2].set_stroke(width=2)
        arrows.set_stroke(opacity=0.8)
        arrows.shift(fc)
        arrows.set_stroke(opacity=0.8)
        face.add(arrows[:2])

        face = Group(face, arrows[2])
        face.add_updater(lambda m: self.sort_to_camera(face))

        arrow_shadow = get_shadow(arrows)
        arrow_shadow.set_stroke(width=1)
        arrow_shadow[2].set_stroke(width=[1, 1, 4, 0])
        self.add(arrow_shadow)

        self.add(z_axis[0], face, z_axis[1])

        frame.add_updater(lambda f, dt: f.increment_theta(self.frame_rot_speed * dt))
        self.wait(self.initial_wait_time)
        for x in range(self.n_rotations):
            self.random_toss(
                face,
                about_point=fc,
                angle=3 * PI,
                # run_time=1.5,
                run_time=8,
                rate_func=smooth,
            )
            self.wait()

        self.wait(self.total_time - 2 - self.initial_wait_time)

    def get_solid(self):
        face = Square(side_length=2)
        face.set_fill(BLUE_E, 0.75)
        face.set_stroke(WHITE, 0.5)
        return face


class RandomRotations1(SingleFaceRandomRotation):
    initial_wait_time = 1
    theta0 = -30 * DEGREES
    CONFIG = {"random_seed": 10}


class RandomRotations2(SingleFaceRandomRotation):
    initial_wait_time = 1.5
    theta0 = -25 * DEGREES
    CONFIG = {"random_seed": 4}


class RandomRotations3(SingleFaceRandomRotation):
    initial_wait_time = 2
    theta0 = -20 * DEGREES
    CONFIG = {"random_seed": 5}


class RandomRotations4(SingleFaceRandomRotation):
    initial_wait_time = 2.5
    theta0 = -15 * DEGREES
    CONFIG = {"random_seed": 6}


class AlicesFaceAverage(Scene):
    def construct(self):
        # Background
        background = FullScreenRectangle()
        self.add(background)

        panels = Rectangle(2, 2.5).replicate(5)
        panels.set_stroke(WHITE, 1)
        panels.set_fill(BLACK, 1)
        dots = Tex("\\dots")
        panels.replace_submobject(3, dots)
        panels.arrange(RIGHT, buff=0.25)
        panels.set_width(FRAME_WIDTH - 1)
        panels.move_to(2 * DOWN, DOWN)
        self.add(panels)
        panels = VGroup(*panels[:-2], panels[-1])

        # Label the rotations
        indices = ["1", "2", "3", "n"]
        rot_labels = VGroup(*(
            Tex(f"R_{i}") for i in indices
        ))
        for label, panel in zip(rot_labels, panels):
            label.set_height(0.3)
            label.next_to(panel, DOWN)

        rot_words = Text("Sequence of random rotations")
        rot_words.next_to(rot_labels, DOWN, MED_LARGE_BUFF)

        self.play(Write(rot_words, run_time=2))
        self.wait(2)
        self.play(LaggedStartMap(
            FadeIn, rot_labels,
            shift=0.25 * DOWN,
            lag_ratio=0.5
        ))
        self.wait()

        # Show the shadow areas
        font_size = 30
        fra_labels = VGroup(*(
            Tex(
                f"f(R_{i})", "\\cdot ", "A",
                tex_to_color_map={"A": BLUE},
                font_size=font_size
            )
            for i in indices
        ))

        DARK_BLUE = interpolate_color(BLUE_D, BLUE_E, 0.5)
        area_shadow_labels = VGroup(*(
            Tex(
                "\\text{Area}(", "\\text{Shadow}_" + i, ")",
                tex_to_color_map={"\\text{Shadow}_" + i: DARK_BLUE},
                font_size=font_size
            )
            for i in indices
        ))
        s_labels = VGroup(*(
            Tex(
                f"S_{i}", "=",
                tex_to_color_map={f"S_{i}": DARK_BLUE},
                font_size=font_size
            )
            for i in indices
        ))
        label_arrows = VGroup()

        for fra, area, s_label, panel in zip(fra_labels, area_shadow_labels, s_labels, panels):
            fra.next_to(panel, UP, SMALL_BUFF)
            area.next_to(fra, UP)
            area.to_edge(UP, buff=LARGE_BUFF)
            label_arrows.add(Arrow(area, fra, buff=0.2, stroke_width=3))

            fra.generate_target()
            eq = VGroup(s_label, fra.target)
            eq.arrange(RIGHT, buff=SMALL_BUFF)
            eq.move_to(fra, DOWN)

        self.add(area_shadow_labels)
        self.add(fra_labels)
        self.add(label_arrows)

        lr = 0.2
        self.play(
            LaggedStartMap(FadeIn, area_shadow_labels, lag_ratio=lr),
            LaggedStartMap(ShowCreation, label_arrows, lag_ratio=lr),
            LaggedStartMap(FadeIn, fra_labels, shift=DOWN, lag_ratio=lr),
        )
        self.wait()
        self.play(
            LaggedStart(*(
                FadeTransform(area, area_s)
                for area, area_s in zip(area_shadow_labels, s_labels)
            ), lag_ratio=lr),
            LaggedStartMap(MoveToTarget, fra_labels, lag_ratio=lr),
            LaggedStartMap(Uncreate, label_arrows, lag_ratio=lr),
        )

        # Show average
        sample_average = Tex(
            "\\text{Sample average}", "=",
            "\\frac{1}{n}", "\\left(",
            "f(R_1)", "\\cdot ", "A", "+",
            "f(R_2)", "\\cdot ", "A", "+",
            "f(R_3)", "\\cdot ", "A", "+",
            "\\cdots ",
            "f(R_n)", "\\cdot ", "A",
            "\\right)",
            font_size=font_size
        )
        sample_average.set_color_by_tex("A", BLUE)
        sample_average.to_edge(UP, buff=MED_SMALL_BUFF)
        for tex in ["\\left", "\\right"]:
            part = sample_average.get_part_by_tex(tex)
            part.scale(1.5)
            part.stretch(1.5, 1)

        self.play(FadeIn(sample_average[:2]))
        self.play(
            TransformMatchingShapes(
                fra_labels.copy(),
                sample_average[4:-1]
            )
        )
        self.wait()
        self.play(
            Write(VGroup(*sample_average[2:4], sample_average[-1]))
        )
        self.wait()

        # Factor out A
        sample_average.generate_target()
        cdots = sample_average.target.get_parts_by_tex("\\cdot", substring=False)
        As = sample_average.target.get_parts_by_tex("A", substring=False)
        new_pieces = VGroup(*(
            sm for sm in sample_average.target
            if sm.get_tex() not in ["A", "\\cdot"]
        ))
        new_A = As[0].copy()
        new_cdot = cdots[0].copy()
        new_pieces.insert_submobject(2, new_cdot)
        new_pieces.insert_submobject(2, new_A)
        new_pieces.arrange(RIGHT, buff=SMALL_BUFF)
        new_pieces.move_to(sample_average)
        for group, target in (As, new_A), (cdots, new_cdot):
            for sm in group:
                sm.replace(target)
            group[1:].set_opacity(0)

        self.play(LaggedStart(
            *(
                FlashAround(mob, time_width=3)
                for mob in sample_average.get_parts_by_tex("A")
            ),
            lag_ratio=0.1,
            run_time=2
        ))
        self.play(MoveToTarget(sample_average, path_arc=-PI / 5))
        self.wait()

        # True average
        brace = Brace(new_pieces[4:], DOWN, buff=SMALL_BUFF, font_size=30)
        lim = Tex("n \\to \\infty", font_size=30)
        lim.next_to(brace, DOWN)
        VGroup(brace, lim).set_color(YELLOW)
        sample = sample_average[0][:len("Sample")]
        cross = Cross(sample)
        cross.insert_n_curves(20)
        cross.scale(1.5)

        self.play(
            FlashAround(sample_average[2:], run_time=3, time_width=1.5)
        )
        self.play(
            FlashUnder(sample_average[:2], color=RED),
            run_time=2
        )
        self.play(
            GrowFromCenter(brace),
            FadeIn(lim, 0.25 * DOWN),
            ShowCreation(cross)
        )
        self.play(
            LaggedStart(*map(FadeOut, [
                *fra_labels, *s_labels, *panels, dots, *rot_labels, rot_words
            ]))
        )
        self.wait()

        # Some constant
        rect = SurroundingRectangle(
            VGroup(new_pieces[4:], brace, lim),
            buff=SMALL_BUFF,
        )
        rect.set_stroke(YELLOW, 1)
        rect.stretch(0.98, 0)
        words = Text("Some constant")
        words.next_to(rect, DOWN)
        subwords = Text("Independent of the size and shape of the 2d piece")
        subwords.scale(0.5)
        subwords.next_to(words, DOWN)
        subwords.set_fill(GREY_A)

        self.play(
            ShowCreation(rect),
            FadeIn(words, 0.25 * DOWN)
        )
        self.wait()
        self.play(Write(subwords))
        self.wait()


class ManyShadows(SingleFaceRandomRotation):
    plane_dims = (4, 4)
    limited_plane_extension = 2

    def construct(self):
        self.clear()
        self.camera.frame.reorient(0, 0)

        plane = self.plane
        face = self.solid
        shadow = self.shadow

        n_rows = 3
        n_cols = 10

        planes = plane.replicate(n_rows * n_cols)
        for n, plane in zip(it.count(1), planes):
            face.rotate(angle=random.uniform(0, TAU), axis=normalize(np.random.uniform(-1, 1, 3)))
            shadow.update()
            sc = shadow.deepcopy()
            sc.clear_updaters()
            sc.set_fill(interpolate_color(BLUE_E, BLACK, 0.5), 0.75)
            plane.set_gloss(0)
            plane.add_to_back(sc)
            area = DecimalNumber(get_norm(sc.get_area_vector() / 4.0), font_size=56)
            label = VGroup(Tex(f"f(R_{n}) = "), area)
            label.arrange(RIGHT)
            label.set_width(0.8 * plane.get_width())
            label.next_to(plane, UP, SMALL_BUFF)
            label.set_color(WHITE)
            plane.add(label)

        planes.arrange_in_grid(n_rows, n_cols, buff=LARGE_BUFF)
        planes.set_width(15)
        planes.to_edge(DOWN)
        planes.update()

        self.play(
            LaggedStart(
                *(
                    FadeIn(plane, scale=1.1)
                    for plane in planes
                ),
                lag_ratio=0.6, run_time=10
            )
        )
        self.wait()

        self.embed()


class AllPossibleOrientations(ShadowScene):
    inf_light = True
    limited_plane_extension = 6
    plane_dims = (12, 8)

    def construct(self):
        # Setup
        frame = self.camera.frame
        frame.reorient(-20, 80)
        frame.set_height(5)
        frame.d_theta = 0

        def update_frame(frame, dt):
            frame.d_theta += -0.0025 * frame.get_theta()
            frame.increment_theta(clip(0.0025 * frame.d_theta, -0.01 * dt, 0.01 * dt))

        frame.add_updater(update_frame)
        face = self.solid
        square, normal_vect = face
        normal_vect.set_flat_stroke()
        self.solid = square
        self.remove(self.shadow)
        self.add_shadow()
        self.shadow.deactivate_depth_test()
        self.solid = face
        fc = square.get_center().copy()

        # Sphere points
        sphere = Sphere(radius=1)
        sphere.set_color(GREY_E, 0.7)
        sphere.move_to(fc)
        sphere.always_sort_to_camera(self.camera)

        n_lat_lines = 20
        theta_step = PI / n_lat_lines
        sphere_points = np.array([
            sphere.uv_func(phi, theta + theta_step * (phi / TAU))
            for theta in np.arange(0, PI, theta_step)
            for phi in np.linspace(
                0, TAU, int(2 * n_lat_lines * math.sin(theta)) + 1
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

        sphere_words = TexText("All normal vectors = Sphere")
        uniform_words = TexText("All points equally likely")
        for words in [sphere_words, uniform_words]:
            words.fix_in_frame()
            words.to_edge(UP)

        # Trace sphere
        N = len(original_sphere_points)
        self.play(FadeIn(sphere_words))
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
            run_time=15,
            rate_func=smooth,
        )
        self.play(
            FadeOut(sphere_words, UP),
            FadeIn(uniform_words, UP),
        )
        last_dot = Mobject()
        for x in range(20):
            point = random.choice(sphere_points)
            dot = TrueDot(
                point,
                radius=1,
                glow_factor=10,
                color=YELLOW,
            )
            face.apply_matrix(rotation_between_vectors(
                normal_vect.get_vector(),
                point - fc
            ), about_point=fc)
            self.add(dot)
            self.play(FadeOut(last_dot, run_time=0.25))
            self.wait(0.25)
            last_dot = dot
        self.play(FadeOut(last_dot))
        self.wait()

        # Sphere itself
        sphere_mesh = SurfaceMesh(sphere, resolution=(21, 11))
        sphere_mesh.set_stroke(BLUE_E, 1, 1)
        for sm in sphere_mesh.get_family():
            sm.uniforms["anti_alias_width"] = 0
        v1 = normal_vect.get_vector()
        normal_vect.scale(0.99, about_point=fc)
        v2 = DR + OUT
        self.play(
            Rotate(
                face, angle_between_vectors(v1, v2),
                axis=normalize(cross(v1, v2))
            ),
            UpdateFromAlphaFunc(
                self.plane, lambda m, a: square.scale(0.9).set_opacity(0.5 - a * 0.5)
            ),
        )
        self.play(
            ShowCreation(sphere_mesh, lag_ratio=0.5),
            FadeIn(sphere),
            sphere_dots.animate.set_radius(0),
            run_time=2,
        )
        self.remove(sphere_dots)

        # Show patch
        patch = ParametricSurface(
            sphere.uv_func,
            # u_range=(0.86 * TAU, 0.91 * TAU),
            # v_range=(0.615 * PI, 0.71 * PI),
            u_range=(0.85 * TAU, 0.9 * TAU),
            v_range=(0.6 * PI, 0.7 * PI),
        )
        patch.shift(fc)
        patch.set_color(YELLOW, 0.75)
        patch.always_sort_to_camera(self.camera)
        self.add(patch, sphere)

        self.play(
            ShowCreation(patch),
            frame.animate.reorient(10, 75),
        )

        # Probability expression
        patch_copy = patch.deepcopy()
        sphere_copy = sphere.deepcopy()
        sphere_copy.set_color(GREY_D, 0.7)
        for mob in patch_copy, sphere_copy:
            mob.apply_matrix(frame.get_inverse_camera_rotation_matrix())
            mob.fix_in_frame()
            mob.center()
        patch_copy2 = patch_copy.copy()

        prob = Group(*Tex(
            "P(", "0.", ")", "=", "{Num ", "\\over ", "Den}",
            font_size=60
        ))
        prob.fix_in_frame()
        prob.to_corner(UR)
        prob.shift(DOWN)
        for i, mob in [(1, patch_copy), (4, patch_copy2), (6, sphere_copy)]:
            mob.replace(prob[i], dim_to_match=1)
            prob.replace_submobject(i, mob)
        sphere_copy.scale(3, about_edge=UP)

        self.play(FadeIn(prob, lag_ratio=0.1))
        self.wait()
        for i in (4, 6):
            self.play(ShowCreationThenFadeOut(
                SurroundingRectangle(prob[i], stroke_width=2).fix_in_frame()
            ))
            self.wait()

        # Non-specified orientation
        self.play(
            LaggedStart(*map(FadeOut, (sphere, sphere_mesh, patch, *prob, uniform_words)))
        )
        self.play(
            square.animate.set_fill(opacity=0.5),
            frame.animate.reorient(-30),
            run_time=3,
        )
        self.play(
            Rotate(square, TAU, normal_vect.get_vector()),
            run_time=8,
        )
        self.wait()

        # Show theta
        def get_normal():
            return normal_vect.get_vector()

        def get_theta():
            return np.arccos(get_normal()[2] / get_norm(get_normal()))

        def get_arc():
            result = Arc(PI / 2, -get_theta(), radius=0.25)
            result.rotate(PI / 2, RIGHT, about_point=ORIGIN)
            result.rotate(angle_of_vector([*get_normal()[:2], 0]), OUT, about_point=ORIGIN)
            result.shift(fc)
            result.set_stroke(WHITE, 1)
            result.apply_depth_test()
            return result

        arc = always_redraw(get_arc)

        theta = Tex("\\theta", font_size=20)
        theta.rotate(PI / 2, RIGHT)
        theta.set_backstroke(width=2)
        theta.add_updater(lambda m: m.next_to(arc.pfp(0.5), OUT + RIGHT, buff=0.05))

        z_axis = Line(ORIGIN, 10 * OUT)
        z_axis.set_stroke(WHITE, 1)
        z_axis.apply_depth_test()

        self.add(z_axis, face, theta, arc)
        self.play(
            ShowCreation(z_axis),
            ShowCreation(arc),
            FadeIn(theta, 0.5 * OUT),
        )
        self.wait()

        # Show shadow area
        shadow_area = TexText("Shadow area =", "$|\\cos(\\theta)|s^2$")
        shadow_area.fix_in_frame()
        shadow_area.to_edge(RIGHT)
        shadow_area.set_y(-3)
        shadow_area.set_backstroke()

        self.play(
            Write(shadow_area, run_time=3),
            Rotate(face, TAU, normal_vect.get_vector(), run_time=10),
        )
        self.wait(4)

        shadow_area[1].generate_target()
        shadow_area[1].target.to_corner(UR, buff=MED_LARGE_BUFF)
        shadow_area[1].target.shift(LEFT)
        brace = Brace(shadow_area[1].target, DOWN)
        brace_text = TexText("How do you average this\\\\over the sphere?", font_size=36)
        brace_text.next_to(brace, DOWN, SMALL_BUFF)
        brace.fix_in_frame()
        brace_text.fix_in_frame()

        self.play(
            GrowFromCenter(brace),
            MoveToTarget(shadow_area[1]),
            FadeOut(shadow_area[0]),
            square.animate.set_fill(opacity=0),
        )
        face.generate_target()
        face.target[1].set_length(0.98, about_point=fc)
        sphere.set_opacity(0.35)
        sphere_mesh.set_stroke(width=0.5)
        self.play(
            MoveToTarget(face),
            FadeIn(brace_text, 0.5 * DOWN),
            Write(sphere_mesh, run_time=2, stroke_width=1),
            FadeIn(sphere),
        )

        # Sum expression
        def update_theta_ring(ring):
            theta = get_theta()
            phi = angle_of_vector([*get_normal()[:2], 0])
            ring.set_width(max(2 * 1.01 * math.sin(theta), 1e-3))
            ring.rotate(phi - angle_of_vector([*ring.get_start()[:2], 0]))
            ring.move_to(fc + math.cos(theta) * OUT)
            return ring

        theta_ring = Circle()
        theta_ring.set_stroke(YELLOW, 2)
        theta_ring.apply_depth_test()
        theta_ring.uniforms["anti_alias_width"] = 0

        loose_sum = Tex(
            "\\sum_{\\theta \\in [0, \\pi]}",
            "P(\\theta)",
            "\\cdot ",
            "|\\cos(\\theta)|s^2"
        )
        loose_sum.fix_in_frame()
        loose_sum.next_to(brace_text, DOWN, LARGE_BUFF)
        loose_sum.to_edge(RIGHT)
        prob_words = TexText("How likely is a given value of $\\theta$?", font_size=36)
        prob_words.fix_in_frame()
        prob_words.next_to(loose_sum[1], DOWN)
        prob_words.to_edge(RIGHT, buff=MED_SMALL_BUFF)

        finite_words = Text("If finite...")
        finite_words.next_to(brace_text, DOWN, LARGE_BUFF).fix_in_frame()
        self.add(finite_words)
        face.rotate(-angle_of_vector([*get_normal()[:2], 0]))
        face.shift(fc - normal_vect.get_start())
        for d_theta in (*[-0.2] * 10, *[0.2] * 10):
            face.rotate(d_theta, np.cross(get_normal(), OUT), about_point=fc)
            self.wait(0.25)

        self.play(
            Write(loose_sum.get_part_by_tex("P(\\theta)")),
            FadeIn(prob_words, 0.5 * DOWN),
            FadeOut(finite_words),
            ApplyMethod(frame.set_x, 1, run_time=2)
        )
        update_theta_ring(theta_ring)
        self.add(theta_ring, sphere)
        self.play(
            Rotate(face, TAU, OUT, about_point=fc, run_time=4),
            ShowCreation(theta_ring, run_time=4),
        )
        theta_ring.add_updater(update_theta_ring)
        self.wait()
        self.play(
            FadeTransform(shadow_area[1].copy(), loose_sum.get_part_by_tex("cos")),
            Write(loose_sum.get_part_by_tex("\\cdot")),
            FadeOut(prob_words, 0.5 * DOWN)
        )
        self.wait(2)
        self.play(
            Write(loose_sum[0], run_time=2),
            run_time=3,
        )
        face.rotate(get_theta(), axis=np.cross(get_normal(), OUT), about_point=fc)
        for x in np.arange(0.2, PI, 0.2):
            face.rotate(0.2, UP, about_point=fc)
            self.wait(0.5)
        self.wait(5)

        # Continuous
        sum_brace = Brace(loose_sum[0], DOWN, buff=SMALL_BUFF)
        continuum = TexText("Continuum\\\\(uncountably infinite)", font_size=36)
        continuum.next_to(sum_brace, DOWN, SMALL_BUFF)
        zero = Tex('0')
        zero.next_to(loose_sum[1], DOWN, buff=1.5)
        zero.shift(1.5 * RIGHT)
        zero_arrow = Arrow(loose_sum[1], zero, buff=SMALL_BUFF)
        nonsense_brace = Brace(loose_sum, UP)
        nonsense = nonsense_brace.get_text("Not really a sensible expression", font_size=36)

        for mob in [sum_brace, continuum, zero, zero_arrow, nonsense_brace, nonsense]:
            mob.fix_in_frame()
            mob.set_color(YELLOW)
        VGroup(nonsense_brace, nonsense).set_color(RED)

        face.start_time = self.time
        face.clear_updaters()
        face.add_updater(lambda f, dt: f.rotate(
            angle=0.25 * dt * math.cos(0.1 * (self.time - f.start_time)),
            axis=np.cross(get_normal(), OUT),
            about_point=fc,
        ).shift(fc - f[1].get_start()))

        self.play(
            GrowFromCenter(sum_brace),
            FadeIn(continuum, 0.5 * DOWN)
        )
        self.wait(4)
        self.play(
            ShowCreation(zero_arrow),
            GrowFromPoint(zero, zero_arrow.get_start()),
        )
        self.wait(2)
        inf_sum_group = VGroup(
            nonsense_brace, nonsense,
            sum_brace, continuum,
            zero_arrow, zero,
            loose_sum,
        )
        top_part = inf_sum_group[:2]
        top_part.set_opacity(0)
        self.play(
            inf_sum_group.animate.to_corner(UR),
            FadeOut(VGroup(brace, brace_text, shadow_area[1])),
            run_time=2,
        )
        top_part.set_fill(opacity=1)
        self.play(
            GrowFromCenter(nonsense_brace),
            Write(nonsense),
        )
        self.wait(10)

        # Swap for an integral
        integral = Tex(
            "\\int_0^\\pi ",
            "p(\\theta)",
            "\\cdot ",
            "|\\cos(\\theta)| s^2",
            "d\\theta",
        )
        integral.shift(loose_sum[-1].get_right() - integral[-1].get_right())
        integral.fix_in_frame()

        self.play(LaggedStart(*map(FadeOut, inf_sum_group[:-1])))
        self.play(
            TransformMatchingShapes(
                loose_sum[0], integral[0],
                fade_transform_mismatches=True,

            )
        )
        self.play(
            FadeTransformPieces(loose_sum[1:4], integral[1:4]),
            Write(integral[4])
        )
        self.wait(5)
        face.clear_updaters()
        self.wait(5)

        # Show 2d slice
        back_half_sphere = Sphere(u_range=(0, PI))
        back_half_sphere.match_color(sphere)
        back_half_sphere.set_opacity(sphere.get_opacity())
        back_half_sphere.shift(fc)
        back_half_mesh = SurfaceMesh(back_half_sphere, resolution=(11, 11))
        back_half_mesh.set_stroke(BLUE_D, 1, 0.75)

        circle = Circle()
        circle.set_stroke(TEAL, 1)
        circle.rotate(PI / 2, RIGHT)
        circle.move_to(fc)

        frame.clear_updaters()
        theta_ring.deactivate_depth_test()
        theta_ring.uniforms.pop("anti_alias_width")
        theta_ring.set_stroke(width=1)
        self.play(
            FadeOut(sphere),
            sphere_mesh.animate.set_stroke(opacity=0.25),
            FadeIn(circle),
            theta_ring.animate.set_stroke(width=1),
            frame.animate.reorient(-6, 87).set_height(4),
            integral.animate.set_height(0.5).set_opacity(0).to_corner(UR),
            run_time=2,
        )
        self.remove(integral)

        # Finite sample
        def get_tick_marks(theta_samples, tl=0.05):
            return VGroup(*(
                Line((1 - tl / 2) * p, (1 + tl / 2) * p).shift(fc)
                for theta in theta_samples
                for p in [np.array([math.sin(theta), 0, math.cos(theta)])]
            )).set_stroke(YELLOW, 1)

        theta_samples = np.linspace(0, PI, sphere_mesh.resolution[0])
        dtheta = theta_samples[1] - theta_samples[0]
        tick_marks = get_tick_marks(theta_samples)

        def set_theta(face, theta):
            face.apply_matrix(rotation_between_vectors(
                normal_vect.get_vector(), OUT
            ), about_point=fc)
            face.rotate(theta, UP, about_point=fc)

        self.play(
            ShowIncreasingSubsets(tick_marks[:-1]),
            UpdateFromAlphaFunc(
                face, lambda f, a: set_theta(face, theta_samples[int(a * (len(theta_samples) - 2))])
            ),
            run_time=4
        )
        self.add(tick_marks)
        self.wait(2)

        tsi = 6  # theta sample index
        dt_line = Line(tick_marks[tsi].get_center(), tick_marks[tsi + 1].get_center())
        dt_brace = Brace(
            Line(ORIGIN, RIGHT), UP
        )
        dt_brace.scale(0.5)
        dt_brace.set_width(dt_line.get_length(), stretch=True)
        dt_brace.rotate(PI / 2, RIGHT)
        dt_brace.rotate(theta_samples[tsi], UP)
        dt_brace.move_to(dt_line)
        dt_brace.shift(SMALL_BUFF * normalize(dt_line.get_center() - fc))
        dt_label = Tex("\\Delta\\theta", font_size=24)
        dt_label.rotate(PI / 2, RIGHT)
        dt_label.next_to(dt_brace, OUT + RIGHT, buff=0.05)

        self.play(
            Write(dt_brace),
            Write(dt_label),
            run_time=1,
        )
        sphere.set_opacity(0.1)
        self.play(
            frame.animate.reorient(10, 70),
            Rotate(face, -get_theta() + theta_samples[tsi], UP, about_point=fc),
            sphere_mesh.animate.set_stroke(opacity=0.5),
            FadeIn(sphere),
            run_time=3
        )
        frame.add_updater(update_frame)
        self.wait()

        # Latitude band
        def get_band(index):
            band = Sphere(
                u_range=(0, TAU), v_range=theta_samples[index:index + 2],
                prefered_creation_axis=1,
            )
            band.set_color(YELLOW, 0.5)
            band.stretch(-1, 2, about_point=ORIGIN)
            band.shift(fc)
            return band

        band = get_band(tsi)

        self.add(band, sphere_mesh, sphere)
        self.play(
            ShowCreation(band),
            Rotate(face, dtheta, UP, about_point=fc),
            run_time=3,
        )
        self.play(Rotate(face, -dtheta, UP, about_point=fc), run_time=3)
        self.wait(2)

        area_question = Text("Area of this band?")
        area_question.set_color(YELLOW)
        area_question.fix_in_frame()
        area_question.set_y(1.75)
        area_question.to_edge(RIGHT, buff=2.5)
        self.play(Write(area_question))
        self.wait()

        random_points = [sphere.pfp(random.random()) - fc for x in range(30)]
        random_points.append(normal_vect.get_end() - fc)
        glow_dots = Group(*(TrueDot(p) for p in random_points))
        for dot in glow_dots:
            dot.shift(fc)
            dot.set_radius(0.2)
            dot.set_color(BLUE)
            dot.set_glow_factor(2)

        theta_ring.suspend_updating()
        last_dot = VectorizedPoint()
        for dot in glow_dots:
            face.apply_matrix(rotation_between_vectors(
                get_normal(), dot.get_center() - fc,
            ), about_point=fc)
            self.add(dot)
            self.play(FadeOut(last_dot), run_time=0.25)
            last_dot = dot
        self.play(FadeOut(last_dot))
        self.wait()

        # Find the area of the band
        frame.clear_updaters()
        self.play(
            frame.animate.reorient(-7.5, 78),
            sphere_mesh.animate.set_stroke(opacity=0.2),
            band.animate.set_opacity(0.2),
        )

        one = Tex("1", font_size=24)
        one.rotate(PI / 2, RIGHT)
        one.next_to(normal_vect.get_center(), IN + RIGHT, buff=0.05)
        radial_line = Line(
            [0, 0, normal_vect.get_end()[2]],
            normal_vect.get_end()
        )
        radial_line.set_stroke(BLUE, 2)
        r_label = Tex("r", font_size=20)
        sin_label = Tex("\\sin(\\theta)", font_size=16)
        for label in r_label, sin_label:
            label.rotate(PI / 2, RIGHT)
            label.next_to(radial_line, OUT, buff=0.05)
            label.set_color(BLUE)
            label.set_backstroke()

        self.play(Write(one))
        self.wait()
        self.play(
            TransformFromCopy(normal_vect, radial_line),
            FadeTransform(one.copy(), r_label)
        )
        self.wait()
        self.play(FadeTransform(r_label, sin_label))
        self.wait()

        band_area = Tex("2\\pi \\sin(\\theta)", "\\Delta\\theta")
        band_area.next_to(area_question, DOWN, LARGE_BUFF)
        band_area.set_backstroke()
        band_area.fix_in_frame()
        circ_label, dt_copy = band_area
        circ_brace = Brace(circ_label, DOWN, buff=SMALL_BUFF)
        circ_words = circ_brace.get_text("Circumference")
        approx = Tex("\\approx")
        approx.rotate(PI / 2)
        approx.move_to(midpoint(band_area.get_top(), area_question.get_bottom()))
        VGroup(circ_brace, circ_words, approx).set_backstroke().fix_in_frame()

        self.play(
            frame.animate.reorient(10, 60),
        )
        theta_ring.update()
        self.play(
            ShowCreation(theta_ring),
            Rotate(face, TAU, OUT, about_point=fc),
            FadeIn(circ_label, 0.5 * DOWN, rate_func=squish_rate_func(smooth, 0, 0.5)),
            GrowFromCenter(circ_brace),
            Write(circ_words),
            run_time=3,
        )
        self.wait()
        self.play(frame.animate.reorient(-5, 75))
        self.play(FadeTransform(area_question[-1], approx))
        area_question.remove(area_question[-1])
        self.play(Write(dt_copy))
        self.wait(3)

        # Probability of falling in band
        prob = Tex(
            "P(\\text{Vector} \\text{ in } \\text{Band})", "=",
            "{2\\pi \\sin(\\theta) \\Delta\\theta", "\\over", " 4\\pi}",
            tex_to_color_map={
                "\\text{Vector}": GREY_B,
                "\\text{Band}": YELLOW,
            }
        )
        prob.fix_in_frame()
        prob.to_edge(RIGHT)
        prob.set_y(1)
        prob.set_backstroke()
        numer = prob.get_part_by_tex("\\sin")
        numer_rect = SurroundingRectangle(numer, buff=0.05)
        numer_rect.set_stroke(YELLOW, 1)
        numer_rect.fix_in_frame()
        area_question.generate_target()
        area_question.target.match_width(numer_rect)
        area_question.target.next_to(numer_rect, UP, SMALL_BUFF)
        denom_rect = SurroundingRectangle(prob.get_part_by_tex("4\\pi"), buff=0.05)
        denom_rect.set_stroke(BLUE, 2)
        denom_rect.fix_in_frame()
        denom_label = TexText("Surface area of\\\\a unit sphere")
        denom_label.scale(area_question.target[0].get_height() / denom_label[0][0].get_height())
        denom_label.set_color(BLUE)
        denom_label.next_to(denom_rect, DOWN, SMALL_BUFF)
        denom_label.fix_in_frame()

        i = prob.index_of_part_by_tex("sin")
        self.play(
            FadeTransform(band_area, prob.get_part_by_tex("sin"), remover=True),
            MoveToTarget(area_question),
            FadeIn(prob[:i]),
            FadeIn(prob[i + 1:]),
            FadeIn(numer_rect),
            *map(FadeOut, [approx, circ_brace, circ_words]),
            frame.animate.set_x(1.5),
        )
        self.add(prob)
        self.remove(band_area)
        self.wait()
        self.play(
            ShowCreation(denom_rect),
            FadeIn(denom_label, 0.5 * DOWN),
        )
        sc = sphere.copy().flip(UP).scale(1.01).set_color(BLUE, 0.5)
        self.add(sc, sphere_mesh)
        self.play(ShowCreation(sc), run_time=3)
        self.play(FadeOut(sc))
        self.wait()

        # Expression for average
        sphere_group = Group(
            sphere, sphere_mesh, theta_ring, band,
            circle, radial_line, sin_label, one, tick_marks,
            dt_brace, dt_label,
        )

        average_eq = Tex(
            "\\text{Average shadow} \\\\",
            "\\sum_{\\theta}",
            "{2\\pi", "\\sin(\\theta)", " \\Delta\\theta", "\\over", " 4\\pi}",
            "\\cdot", "|\\cos(\\theta)|", "s^2"
        )
        average_eq.fix_in_frame()
        average_eq.move_to(prob).to_edge(UP)
        average_eq[0].scale(1.25)
        average_eq[0].shift(MED_SMALL_BUFF * UP)
        average_eq[0].match_x(average_eq[1:])

        new_prob = average_eq[2:7]
        prob_rect = SurroundingRectangle(new_prob)
        prob_rect.set_stroke(YELLOW, 2)
        prob_rect.fix_in_frame()

        self.play(
            FadeIn(average_eq[:1]),
            FadeIn(prob_rect),
            prob[:5].animate.match_width(prob_rect).next_to(prob_rect, DOWN, buff=0.15),
            FadeTransform(prob[-3:], new_prob),
            *map(FadeOut, [prob[5], numer_rect, denom_rect, area_question, denom_label])
        )
        self.wait()
        self.play(
            FadeOut(sphere_group),
            FadeIn(average_eq[-3:]),
            UpdateFromAlphaFunc(face, lambda f, a: f[0].set_fill(opacity=0.5 * a))
        )
        self.wait()
        band.set_opacity(0.5)
        bands = Group(*(get_band(i) for i in range(len(theta_samples) - 1)))
        sphere_mesh.set_stroke(opacity=0.5)
        self.add(sphere_mesh, sphere, bands)
        self.play(
            FadeIn(average_eq[1]),
            UpdateFromAlphaFunc(face, lambda f, a: f[0].set_fill(opacity=0.5 * (1 - a))),
            FadeIn(sphere),
            FadeIn(tick_marks),
            FadeIn(sphere_mesh),
            LaggedStartMap(
                FadeIn, bands,
                rate_func=there_and_back,
                lag_ratio=0.5,
                run_time=8,
                remover=True
            ),
        )

        # Simplify
        average2 = Tex(
            "{2\\pi", "\\over", "4\\pi}", "s^2",
            "\\sum_{\\theta}",
            "\\sin(\\theta)", "\\Delta\\theta",
            "\\cdot", "|\\cos(\\theta)|"
        )
        average2.fix_in_frame()
        average2.move_to(average_eq[1:], RIGHT)
        half = Tex("1 \\over 2")
        pre_half = average2[:3]
        half.move_to(pre_half, RIGHT)
        half_rect = SurroundingRectangle(pre_half, buff=SMALL_BUFF)
        half_rect.set_stroke(RED, 1)
        VGroup(half, half_rect).fix_in_frame()

        self.play(
            FadeOut(prob_rect),
            FadeOut(prob[:5]),
            *(
                FadeTransform(average_eq[i], average2[j], path_arc=10 * DEGREES)
                for i, j in [
                    (1, 4),
                    (2, 0),
                    (3, 5),
                    (4, 6),
                    (5, 1),
                    (6, 2),
                    (7, 7),
                    (8, 8),
                    (9, 3),
                ]
            ),
            run_time=2,
        )
        self.play(ShowCreation(half_rect))
        self.play(
            FadeTransform(pre_half, half),
            FadeOut(half_rect),
        )
        sin, dt, dot, cos = average2[5:]
        tail = VGroup(cos, dot, sin, dt)
        tail.generate_target()
        tail.target.arrange(RIGHT, buff=SMALL_BUFF)
        tail.target.move_to(tail, LEFT)
        tail.target[-1].align_to(sin[0], DOWN)
        self.play(
            MoveToTarget(tail, path_arc=PI / 2),
        )
        self.wait(2)

        integral = Tex("\\int_0^\\pi ")
        integral.next_to(tail, LEFT, SMALL_BUFF)
        integral.fix_in_frame()
        dtheta = Tex("d\\theta").fix_in_frame()
        dtheta.move_to(tail[-1], LEFT)

        average_copy = VGroup(half, average2[3:]).copy()
        average_copy.set_backstroke()
        self.play(
            VGroup(half, average2[3]).animate.next_to(integral, LEFT, SMALL_BUFF),
            FadeTransform(average2[4], integral),
            FadeTransform(tail[-1], dtheta),
            average_copy.animate.shift(2.5 * DOWN),
            frame.animate.set_phi(80 * DEGREES),
        )
        self.wait()
        self.play(LaggedStart(
            ShowCreationThenFadeOut(SurroundingRectangle(average_copy[1][-3]).fix_in_frame()),
            ShowCreationThenFadeOut(SurroundingRectangle(dtheta).fix_in_frame()),
            lag_ratio=0.5
        ))
        self.wait()

        # The limit
        brace = Brace(average_copy, UP, buff=SMALL_BUFF)
        brace_text = brace.get_text(
            "What does this approach for finer subdivisions?",
            font_size=30
        )
        arrow = Arrow(integral.get_bottom(), brace_text)
        VGroup(brace, brace_text, arrow).set_color(YELLOW).fix_in_frame()
        brace_text.set_backstroke()

        self.play(
            GrowFromCenter(brace),
            ShowCreation(arrow),
            FadeIn(brace_text, lag_ratio=0.1)
        )

        for n in range(1, 4):
            new_ticks = get_tick_marks(
                np.linspace(0, PI, sphere_mesh.resolution[0] * 2**n),
                tl=0.05 / n
            )
            self.play(
                ShowCreation(new_ticks),
                FadeOut(tick_marks),
                run_time=2,
            )
            self.wait()
            tick_marks = new_ticks

        # Make room for computation
        face[0].set_fill(BLUE_D, opacity=0.75)
        face[0].set_stroke(WHITE, 0.5, 1)
        rect = Rectangle(fill_color=BLACK, fill_opacity=1, stroke_width=0)
        rect.replace(self.plane, stretch=True)
        rect.stretch(4 / 12, dim=0, about_edge=RIGHT)
        rect.scale(1.01)
        top_line = VGroup(half, average2[3], integral, tail[:-1], dtheta)
        self.add(face[0], sphere)
        self.play(
            LaggedStart(*map(FadeOut, [arrow, brace_text, brace, average_copy])),
            # UpdateFromAlphaFunc(face, lambda f, a: f[0].set_fill(opacity=0.5 * a)),
            GrowFromCenter(face[0], remover=True),
            frame.animate.set_height(6).set_x(3.5),
            FadeIn(rect),
            FadeOut(tick_marks),
            top_line.animate.set_width(4).to_edge(UP).to_edge(RIGHT, buff=LARGE_BUFF),
            FadeOut(average_eq[0], UP),
            run_time=2,
        )
        self.add(face, sphere)
        self.begin_ambient_rotation(face, about_point=fc, speed=0.1)

        # Computation
        new_lines = VGroup(
            Tex("{1 \\over 2} s^2 \\cdot 2 \\int_0^{\\pi / 2} \\cos(\\theta)\\sin(\\theta)\\,d\\theta"),
            Tex("{1 \\over 2} s^2 \\cdot \\int_0^{\\pi / 2} \\sin(2\\theta)\\,d\\theta"),
            Tex("{1 \\over 2} s^2 \\cdot \\left[ -\\frac{1}{2} \\cos(2\\theta) \\right]_0^{\\pi / 2}"),
            Tex("{1 \\over 2} s^2 \\cdot \\left(-\\left(-\\frac{1}{2}\\right) - \\left(-\\frac{1}{2}\\right)\\right)"),
            Tex("{1 \\over 2} s^2"),
        )
        new_lines.scale(top_line.get_height() / new_lines[0].get_height())
        kw = {"buff": 0.35, "aligned_edge": LEFT}
        new_lines.arrange(DOWN, **kw)
        new_lines.next_to(top_line, DOWN, **kw)
        new_lines.fix_in_frame()

        annotations = VGroup(
            TexText("To avoid the annoying absolute value, just\\\\cover the northern hemisphere and double it."),
            TexText("Trig identity: $\\sin(2\\theta) = 2\\cos(\\theta)\\sin(\\theta)$"),
            TexText("Antiderivative"),
            TexText("Try not to get lost in\\\\the sea of negatives..."),
            TexText("Whoa, that turned out nice!"),
        )
        annotations.fix_in_frame()
        annotations.set_color(YELLOW)
        annotations.scale(0.5)

        rect = SurroundingRectangle(new_lines[-1], buff=SMALL_BUFF)
        rect.set_stroke(YELLOW, 2).fix_in_frame()

        for note, line in zip(annotations, new_lines):
            note.next_to(line, LEFT, MED_LARGE_BUFF)

        self.play(
            LaggedStartMap(FadeIn, new_lines, lag_ratio=0.7),
            LaggedStartMap(FadeIn, annotations, lag_ratio=0.7),
            run_time=5,
        )
        self.wait(20)
        self.play(
            new_lines[:-1].animate.set_opacity(0.5),
            annotations[:-1].animate.set_opacity(0.5),
            ShowCreation(rect),
        )
        self.wait(10)

    def get_solid(self):
        face = Square(side_length=2)
        face.set_fill(BLUE, 0.5)
        face.set_stroke(width=0)
        normal = Vector(OUT)
        normal.shift(2e-2 * OUT)
        face = VGroup(face, normal)
        face.set_stroke(background=True)
        face.apply_depth_test()
        return face


class ThreeCamps(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        # Setup
        teacher = self.teacher
        students = self.students

        image = ImageMobject("Shadows_Integral_Intro")
        image.center().set_height(FRAME_HEIGHT)
        image.generate_target()
        image.target.replace(self.screen)
        self.screen.set_stroke(WHITE, 1)
        self.screen.save_state()
        self.screen.replace(image).set_stroke(width=0)

        self.play(
            LaggedStart(*(
                student.animate.change("pondering", image.target)
                for student in students
            ), run_time=2, lag_ratio=0.2),
            teacher.animate.change("tease")
        )
        self.wait()

        # Reactions
        phrases = [
            Text("How fun!", font_size=40),
            Text("Wait, what?", font_size=40),
            Text("Okay give\nme a sec...", font_size=35),
        ]
        modes = ["hooray", "erm", "confused"]
        heights = np.linspace(2.0, 2.5, 3)
        for student, phrase, mode, height in zip(reversed(students), phrases, modes, heights):
            self.play(
                PiCreatureSays(
                    student, phrase, target_mode=mode,
                    look_at_arg=image,
                    bubble_kwargs={
                        "direction": LEFT,
                        "width": 3,
                        "height": height,
                    },
                    bubble_class=ThoughtBubble,
                    run_time=2
                )
            )
        self.wait(4)

        # Next
        integral = Tex("\\int_0^\\pi \\dots d\\theta")
        integral.move_to(self.hold_up_spot, DOWN)

        self.play(
            LaggedStart(*(
                FadeOut(VGroup(student.bubble, student.bubble.content))
                for student in students
            )),
            LaggedStart(*(
                student.animate.change("pondering", integral)
                for student in students
            )),
            FadeIn(integral, UP),
            teacher.animate.change("raise_right_hand", integral),
        )
        self.wait(3)

        # Embed
        self.embed()


class TwoToOneCover(ShadowScene):
    inf_light = True
    plane_dims = (20, 12)
    limited_plane_extension = 8
    highlighted_face_color = YELLOW

    def construct(self):
        # Setup
        frame = self.camera.frame
        frame.reorient(-20, 75)
        frame.set_z(3)
        self.init_frame_rotation()

        cube = self.solid
        cube.add_updater(lambda m: self.sort_to_camera(m))
        cube.rotate(PI / 3, normalize([3, 4, 5]))

        outline = self.get_shadow_outline()

        # Inequality
        t2c = {
            "Shadow": GREY_B,
            "Cube": BLUE_D,
            "Face$_i$": YELLOW,
        }
        ineq = TexText(
            "Area(Shadow(Cube))",
            " $<$ ",
            " $\\displaystyle \\sum_{i=1}^6$ ",
            "Area(Shadow(Face$_i$))",
            tex_to_color_map=t2c,
            isolate=["(", ")"],
        )
        ineq.to_edge(UP, buff=MED_SMALL_BUFF)
        ineq.fix_in_frame()

        lhs = ineq.slice_by_tex(None, "<")
        lt = ineq.get_part_by_tex("<")
        rhs = ineq.slice_by_tex("sum", None)
        af_label = ineq[-7:]
        lhs.save_state()
        lhs.set_x(0)

        # Shadow of the cube
        wireframe = cube.copy()
        for face in wireframe:
            face.set_fill(opacity=0)
            face.set_stroke(WHITE, 1)
        wireframe_shadow = wireframe.copy()
        wireframe_shadow.apply_function(flat_project)
        wireframe_shadow.set_gloss(0)
        wireframe_shadow.set_reflectiveness(0)
        wireframe_shadow.set_shadow(0)
        for face in wireframe_shadow:
            face.set_stroke(GREY_D, 1)

        self.play(
            ShowCreation(wireframe, lag_ratio=0.1),
            Write(lhs[2:-1])
        )
        self.play(TransformFromCopy(wireframe, wireframe_shadow))
        self.play(*map(FadeOut, (wireframe, wireframe_shadow)))
        self.wait()
        self.play(
            FadeIn(lhs[:2]), FadeIn(lhs[-1]),
            Write(outline),
            VShowPassingFlash(
                outline.copy().set_stroke(YELLOW, 4),
                time_width=1.5
            ),
        )
        self.wait()

        # Show faces and shadows
        cube.save_state()
        faces, face_shadows = self.get_faces_and_face_shadows()
        faces[:3].set_opacity(0.1)
        face_shadow_lines = VGroup(*(
            VGroup(*(
                Line(v1, v2)
                for v1, v2 in zip(f.get_vertices(), fs.get_vertices())
            ))
            for f, fs in zip(faces, face_shadows)
        ))
        face_shadow_lines.set_stroke(YELLOW, 0.5, 0.5)

        self.play(
            Restore(lhs),
            FadeIn(af_label, shift=0.5 * RIGHT)
        )
        self.play(
            *(
                LaggedStart(*(
                    VFadeInThenOut(sm)
                    for sm in reversed(mobject)
                ), lag_ratio=0.5, run_time=6)
                for mobject in [faces, face_shadows, face_shadow_lines]
            ),
        )
        self.play(
            ApplyMethod(cube.space_out_submobjects, 1.7, rate_func=there_and_back_with_pause, run_time=8),
            ApplyMethod(frame.reorient, 20, run_time=8),
            Write(lt),
            Write(rhs[0]),
        )
        self.wait(2)

        # Show a given pair of faces
        face_pair = faces[:2].copy()

        # Half of sum
        eq_half = Tex("=", "\\frac{1}{2}")
        eq_half.move_to(lt, LEFT)
        eq_half.fix_in_frame()

        self.play(
            FadeOut(lt, UP),
            Write(eq_half),
            rhs.animate.next_to(eq_half, RIGHT, buff=0.15),
        )

        # Show the double cover

        # Embed
        self.embed()

    def get_faces_and_face_shadows(self):
        faces = self.solid.deepcopy()
        VGroup(*faces).set_fill(self.highlighted_face_color)

        shadows = get_pre_shadow(faces, opacity=0.7)
        shadows.apply_function(flat_project)
        return faces, shadows
