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
    # return [*point[:2], 0]
    return [*point[:2], 0.05 * point[2]]  # TODO


def get_pre_shadow(mobject, opacity):
    result = mobject.deepcopy()
    if isinstance(result, Group) and all((isinstance(sm, VMobject) for sm in mobject)):
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
        if isinstance(sm, VMobject):
            sm.set_fill(opacity=mm.get_fill_opacity())
        else:
            sm.set_opacity(mm.get_opacity())


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
        if isinstance(sm, VMobject):
            sm.refresh_unit_normal()
    return mobject


def cube_sdf(point, cube):
    c = cube.get_center()
    vect = point - c
    face_vects = [face.get_center() - c for face in cube]
    return max(*(
        abs(np.dot(fv, vect) / np.dot(fv, fv))
        for fv in face_vects
    )) - 1


def is_in_cube(point, cube):
    return cube_sdf(point, cube) < 0


def get_overline(mob):
    overline = Underline(mob).next_to(mob, UP, buff=0.05)
    overline.set_stroke(WHITE, 2)
    return overline


def get_key_result(solid_name, color=BLUE):
    eq = Tex(
        "\\text{Area}\\big(\\text{Shadow}(\\text{" + solid_name + "})\\big)",
        "=",
        "\\frac{1}{2}", "{c}", "\\cdot",
        "(\\text{Surface area})",
        tex_to_color_map={
            "\\text{Shadow}": GREY_B,
            f"\\text{{{solid_name}}}": color,
            "\\text{Solid}": BLUE,
            "{c}": RED,
        }
    )
    eq.add_to_back(get_overline(eq[:5]))
    return eq


def get_surface_area(solid):
    return sum(get_norm(f.get_area_vector()) for f in solid)


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

    def begin_ambient_rotation(self, mobject, speed=0.2, about_point=None, initial_axis=[1, 1, 1]):
        mobject.rot_axis = np.array(initial_axis)

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

    def random_toss(self, mobject=None, angle=TAU, about_point=None, meta_speed=5, **kwargs):
        if mobject is None:
            mobject = self.solid

        mobject.rot_axis = normalize(np.random.random(3))
        mobject.rot_time = 0

        def update(mob, time):
            dt = time - mob.rot_time
            mob.rot_time = time
            mob.rot_axis = rotate_vector(mob.rot_axis, meta_speed * dt, normalize(np.random.random(3)))
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


class SimpleWriting(Scene):
    text = ""
    font = "Better Grade"
    color = WHITE
    font_size = 48

    def construct(self):
        words = Text(self.text, font=self.font, font_size=self.font_size)
        words.set_color(self.color)
        self.play(Write(words))
        self.wait()


class AliceName(SimpleWriting):
    text = "Alice"
    font_size = 72


class BobName(SimpleWriting):
    text = "Bob"
    font = "Kalam"


class BobWords(SimpleWriting):
    font = "Kalam"
    font_size = 24
    words1 = "Embraces calculations"
    words2 = "Loves specifics"

    def construct(self):
        words = VGroup(*(
            Text(text, font=self.font, font_size=self.font_size)
            for text in (self.words1, self.words2)
        ))
        words.arrange(DOWN)

        for word in words:
            self.play(Write(word))
            self.wait()


class AliceWords(BobWords):
    font = "Better Grade"
    words1 = "Procrastinates calculations"
    words2 = "Seeks generality"
    font_size = 48


class AskAboutConditions(SimpleWriting):
    text = "Which properties matter?"


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
            run_time=5
        )
        self.play(
            Rotate(light, PI, about_point=light.get_z() * OUT),
            run_time=8,
        )
        self.play(light.animate.shift(4 * RIGHT), run_time=5)
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

        self.wait(2)
        frame.save_state()
        cube_opacity = cube[0].get_fill_opacity()
        cube.save_state()
        angle = angle_of_vector(outline.get_anchors()[-1] - outline.get_anchors()[-2])
        self.play(
            frame.animate.reorient(0, 0),
            cube.animate.rotate(-angle).set_opacity(0.2),
            run_time=3,
        )
        frame.suspend_updating()
        outline_copy = outline.copy().clear_updaters()
        outline_copy.set_stroke(RED, 5)
        title = Text("Regular hexagon")
        title.set_color(RED)
        title.next_to(outline_copy, UP)
        title.set_backstroke()
        self.play(
            ShowCreationThenFadeOut(outline_copy),
            Write(title, run_time=1),
        )
        self.play(
            FadeOut(title),
            Restore(frame),
            cube.animate.set_opacity(cube_opacity).rotate(angle),
            run_time=3,
        )
        frame.resume_updating()

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

        self.play_student_changes(
            "maybe", "thinking", "erm",
            look_at=self.screen,
            added_anims=[self.teacher.change("raise_right_hand", self.screen)]
        )
        self.wait(3)
        self.play(
            PiCreatureBubbleIntroduction(
                sts[2], TexText("What does that\\\\mean, exactly?"),
                target_mode="hesitant",
                look_at=self.screen,
                bubble_config={"direction": LEFT}
            ),
            LaggedStart(
                sts[0].change("confused", self.screen),
                sts[1].change("pondering", self.screen),
                tch.change("tease", sts[2].eyes),
            )
        )
        self.wait(4)
        self.student_says(
            "Can we do an experiment?",
            target_mode="raise_left_hand",
            index=1,
        )
        self.wait(4)
        self.student_says(
            TexText("But what defines a\\\\``random'' toss?"),
            look_at=self.screen,
            target_mode="hesitant",
            index=2,
            added_anims=[
                self.teacher.change("guilty"),
                self.students[0].change("erm"),
            ]
        )
        self.wait(4)
        self.play(LaggedStart(
            self.students[0].change("pondering", self.screen),
            self.students[1].change("maybe", self.screen),
            self.teacher.change("tease", self.screen),
        ))
        self.wait(2)
        self.teacher_says(TexText("Hold off until\\\\the end"))
        self.wait(3)
        self.play_student_changes(
            "thinking", "tease", "pondering",
            look_at=self.screen,
            added_anims=[self.teacher.change("tease", self.students)]
        )
        self.wait(4)


class MeanCalculation(Scene):
    def construct(self):
        values = [1.55, 1.33, 1.46, 1.34, 1.50, 1.26, 1.42, 1.54, 1.51]
        nums = VGroup(*(
            DecimalNumber(x)
            for x in values
        ))
        nums.arrange(DOWN, aligned_edge=LEFT)
        nums.to_corner(UR, buff=LARGE_BUFF).shift(0.5 * LEFT)

        self.add(nums)

        mean_label = Text("Mean", font_size=36)
        mean_label.set_color(GREEN)
        mean_label.set_backstroke()
        mean_arrow = Vector(0.25 * UR)
        mean_arrow.match_color(mean_label)
        mean_arrow.next_to(mean_label, UR, SMALL_BUFF)
        mean_label.add(mean_arrow)

        for n in range(len(nums)):
            brace = Brace(nums[:n + 1], LEFT, buff=SMALL_BUFF)
            mean = DecimalNumber(np.mean(values[:n + 1]))
            mean.next_to(brace, LEFT)
            mean.match_color(mean_label)
            VGroup(brace, mean).set_backstroke()
            mean_label.next_to(mean, DL, SMALL_BUFF)

            self.add(brace, mean, mean_label)
            self.wait(0.5)
            self.remove(brace, mean)
        self.add(brace, mean)
        self.wait()

        # Embed
        self.embed()


class DescribeSO3(ShadowScene):
    def construct(self):
        frame = self.camera.frame
        frame.set_z(1)
        frame.reorient(0)
        cube = self.solid
        cube.set_opacity(0.95)
        cube.move_to(ORIGIN)
        self.remove(self.plane)
        self.remove(self.shadow)

        x_point = VectorizedPoint(cube.get_right())
        y_point = VectorizedPoint(cube.get_top())
        z_point = VectorizedPoint(cube.get_zenith())
        cube.add(x_point, y_point, z_point)

        def get_matrix():
            return np.array([
                x_point.get_center(),
                y_point.get_center(),
                z_point.get_center(),
            ]).T

        def get_mat_mob():
            matrix = DecimalMatrix(
                get_matrix(),
                element_to_mobject_config=dict(
                    num_decimal_places=2,
                    edge_to_fix=LEFT,
                    include_sign=True,
                ),
                h_buff=2.0,
                element_alignment_corner=LEFT,
            )
            matrix.fix_in_frame()
            matrix.set_height(1.25)
            brackets = matrix.get_brackets()
            brackets[1].move_to(brackets[0].get_center() + 3.45 * RIGHT)
            matrix.to_corner(UL)
            return matrix

        matrix = always_redraw(get_mat_mob)
        self.add(matrix)

        # Space of orientations
        self.begin_ambient_rotation(cube, speed=0.4)
        self.wait(2)

        question = Text("What is the space of all orientations?")
        question.to_corner(UR)
        question.fix_in_frame()
        SO3 = Tex("SO(3)")
        SO3.next_to(question, DOWN)
        SO3.set_color(BLUE)
        SO3.fix_in_frame()

        self.play(Write(question))
        self.wait(2)
        self.play(FadeIn(SO3, DOWN))
        self.wait(2)
        self.play(SO3.animate.next_to(matrix, DOWN, MED_LARGE_BUFF))
        self.wait(5)

        new_question = Text(
            "What probability distribution are we placing\n"
            "on the space of all orientations?",
            t2c={"probability distribution": YELLOW},
            t2s={"probability distribution": ITALIC},
        )
        new_question.match_width(question)
        new_question.move_to(question, UP)
        new_question.fix_in_frame()

        n = len("the space of all orientations?")
        self.play(
            FadeTransform(question[-n:], new_question[-n:]),
            FadeOut(question[:-n]),
            FadeIn(new_question[:-n]),
        )
        self.wait()
        cube.clear_updaters()

        N = 15
        cube_field = cube.get_grid(N, N)
        cube_field.set_height(10)

        for n, c in enumerate(cube_field):
            c.rotate(PI * (n // N) / N, axis=RIGHT)
            c.rotate(PI * (n % N) / N, axis=UP)
            for face in c:
                face.set_stroke(width=0)
            self.sort_to_camera(c)

        matrix.clear_updaters()
        self.play(
            FadeTransform(cube, cube_field[0]),
            LaggedStartMap(FadeIn, cube_field, run_time=15, lag_ratio=0.1)
        )
        self.add(cube_field)
        self.wait()


class PauseAndPonder(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)

        self.teacher_says(
            TexText("The goal is\\\\not speed."),
            added_anims=[self.change_students(
                "tease", "well", "pondering",
                look_at=self.screen
            )]
        )
        self.wait(2)
        self.play(
            RemovePiCreatureBubble(self.teacher, target_mode="tease"),
            PiCreatureBubbleIntroduction(
                self.students[2],
                Lightbulb(),
                bubble_type=ThoughtBubble,
                bubble_creation_class=lambda m: FadeIn(m, lag_ratio=0.1),
                bubble_config=dict(
                    height=3,
                    width=3,
                    direction=LEFT,
                ),
                target_mode="thinking",
                look_at=self.screen,
            )
        )
        self.wait(3)
        self.teacher_says(
            "Pause and ponder!",
            target_mode="well",
            added_anims=[self.change_students(
                "pondering", "tease", "thinking"
            )],
            run_time=1
        )
        self.wait(5)

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
        words.next_to(line, DOWN, MED_SMALL_BUFF)
        rect = BackgroundRectangle(words, fill_opacity=1, buff=SMALL_BUFF)
        words.set_backstroke(width=5)

        # Shapes
        cube = VCube()
        cube.deactivate_depth_test()
        cube.set_color(BLUE_E)
        cube.set_opacity(0.75)
        cube.set_stroke(WHITE, 0.5, 0.5)
        cube.set_height(2)
        cube.rotate(PI / 4, [1, 2, 0])
        cube.sort(lambda p: p[2])
        cube = Group(*cube)
        cube.set_gloss(1)

        arrow = Arrow(LEFT, RIGHT)
        face = cube[np.argmax([f.get_z() for f in cube])].copy()
        group = Group(cube, arrow, face)
        group.arrange(RIGHT, buff=MED_LARGE_BUFF)
        group.next_to(words, DOWN, LARGE_BUFF)
        group.set_width(2)
        group.to_edge(RIGHT)
        group.set_y(0)

        self.camera.light_source.set_x(-4)

        self.play(
            ShowCreation(line),
            Write(title, run_time=1),
        )
        self.wait()
        self.play(
            FadeIn(rect),
            FadeIn(words, lag_ratio=0.1),
            run_time=2
        )
        self.wait()
        self.play(FlashAround(words.get_part_by_text("non-trivial"), run_time=2))
        self.wait()
        self.play(
            LaggedStart(*map(DrawBorderThenFill, cube)),
            ShowCreation(arrow),
            TransformFromCopy(cube[-1], face)
        )
        self.wait(3)


class FocusOnOneFace(ShadowScene):
    inf_light = True
    limited_plane_extension = 10

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

        # Vary Theta
        frame.reorient(2)
        face.rotate(-35 * DEGREES, get_un(), about_point=face.get_center())
        self.play(
            Rotate(face, 50 * DEGREES, UP),
            rate_func=there_and_back,
            run_time=8,
        )

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
            bubble_config={"height": 3, "width": 4},
            added_anims=[
                self.change_students(
                    "pondering", "thinking", "erm",
                    look_at=self.screen,
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

        # self.play(Write(shape_labels[0], run_time=1))
        # self.wait()

        for arrow, label in zip(arrows, arrow_labels):
            self.play(
                ShowCreation(arrow),
                FadeIn(label, lag_ratio=0.1)
            )
            self.wait()

        # Linear!
        lin_text = Text(
            "Both are linear transformations!",
            t2c={"linear": YELLOW}
        )
        lin_text.next_to(panels, UP, MED_SMALL_BUFF)

        self.play(FadeIn(lin_text, lag_ratio=0.1))
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

        # Determinant
        factor = area_exprs[1].get_part_by_tex('factor')
        rect = SurroundingRectangle(factor, buff=SMALL_BUFF)
        rect.set_stroke(YELLOW, 2)

        rot = Matrix([["v_1", "w_1"], ["v_2", "w_2"], ["v_3", "w_3"]], h_buff=1.0)
        rot.set_column_colors(GREEN, RED)
        proj = Matrix([["1", "0", "0"], ["0", "1", "0"]], h_buff=0.6)
        prod = VGroup(proj, rot)
        prod.arrange(RIGHT, buff=SMALL_BUFF)
        prod.set_height(0.8)
        det = Tex(
            "\\text{det}", "\\Big(", "\\Big)",
            tex_to_color_map={
                "\\text{det}": YELLOW,
                # "rot": BLUE_D,
                # "proj": BLUE_B,
            },
            font_size=36
        )
        det[1:].match_height(prod, stretch=True)
        det.to_edge(UP)
        prod.next_to(det[1], RIGHT, SMALL_BUFF)
        det[2].next_to(prod, RIGHT, SMALL_BUFF)
        det.add(prod)
        det.center().to_edge(UP, buff=0.25)
        det_rect = SurroundingRectangle(det, buff=SMALL_BUFF)
        det_rect.set_stroke(YELLOW, 1)

        rot_brace = Brace(rot, DOWN, buff=SMALL_BUFF)
        details = Text("Need to work out rotation matrix...", font_size=20)
        details.next_to(rot_brace, DOWN, SMALL_BUFF)
        details.set_color(GREY_A)

        arrow = Arrow(rect.get_corner(UL), det.get_right())
        arrow.set_color(YELLOW)

        self.play(ShowCreation(rect))
        self.play(
            FadeTransform(rect.copy(), det_rect),
            FadeTransform(factor.copy(), det),
            ShowCreation(arrow)
        )
        self.wait()
        self.play(
            FadeOut(det_rect),
            GrowFromCenter(rot_brace),
            FadeIn(details),
        )
        self.wait()
        self.play(LaggedStart(*map(FadeOut, (
            *det, rot_brace, details
        )), lag_ratio=0.3, run_time=2))

        # Any shape
        ind_words = Text("Independent of the shape!", font_size=30)
        ind_words.move_to(det)
        ind_words.set_color(GREEN)

        self.play(
            arrow.animate.match_points(Arrow(factor.get_corner(UL), ind_words.get_corner(DR))),
            Write(ind_words, run_time=1),
        )
        self.wait()
        self.play(LaggedStart(*map(FadeOut, (ind_words, arrow, rect))))
        self.wait()

        # Cross out right
        cross = Cross(VGroup(equals[1], f_rot, times_A))
        cross.insert_n_curves(20)
        self.play(ShowCreation(cross))
        self.wait(3)


class Matrices(Scene):
    def construct(self):
        self.add(FullScreenRectangle())

        kw = {
            "v_buff": 0.7,
            "bracket_v_buff": 0.15,
            "bracket_h_buff": 0.15,
        }
        matrices = VGroup(
            Matrix([["v_1", "w_1"], ["v_2", "w_2"], ["v_3", "w_3"]], h_buff=1.0, **kw),
            Matrix([["1", "0", "0"], ["0", "1", "0"]], h_buff=0.6, **kw),
        )
        matrices.set_color(GREY_A)
        matrices[0].set_column_colors(GREEN, RED)
        matrices.arrange(LEFT, buff=SMALL_BUFF)
        matrices.scale(0.5)
        mat_product = matrices[:2].copy()

        vectors = VGroup(
            Matrix([["x_0"], ["y_0"]], **kw),
            Matrix([["x_1"], ["y_1"], ["z_1"]], **kw),
            Matrix([["x_2"], ["y_2"]], **kw),
        )

        for vect, x in zip(vectors, [-6, 0, 6]):
            vect.set_x(x)
            vect.set_y(2.5)

        arrows = VGroup(
            Arrow(vectors[0], vectors[1]),
            Arrow(vectors[1], vectors[2]),
            Arrow(vectors[0], vectors[2]),
        )

        for mat, arrow in zip((*matrices[:2], mat_product), arrows):
            mat.next_to(arrow, UP, SMALL_BUFF)

        # Animations
        self.add(vectors[0])
        for i in range(2):
            self.play(
                FadeTransform(vectors[i].copy(), vectors[i + 1]),
                ShowCreation(arrows[i]),
                FadeIn(matrices[i], 0.5 * RIGHT)
            )
            self.wait()

        self.play(
            Transform(arrows[0], arrows[2]),
            Transform(arrows[1], arrows[2]),
            Transform(matrices, mat_product),
            FadeOut(vectors[1], scale=0),
        )


class DefineDeterminant(Scene):
    def construct(self):
        # Planes
        plane = NumberPlane((-2, 2), (-3, 3))
        plane.set_height(FRAME_HEIGHT)
        planes = VGroup(plane, plane.deepcopy())
        planes[0].to_edge(LEFT, buff=0)
        planes[1].to_edge(RIGHT, buff=0)
        planes[1].set_stroke(GREY_A, 1, 0.5)
        planes[1].faded_lines.set_opacity(0)

        titles = VGroup(
            Text("Input"),
            Text("Output"),
        )
        for title, plane in zip(titles, planes):
            title.next_to(plane.get_top(), DOWN)
            title.add_background_rectangle()

        self.add(planes)

        # Area
        square = Square()
        square.set_stroke(YELLOW, 2)
        square.set_fill(YELLOW, 0.5)
        square.replace(Line(planes[0].c2p(-1, -1), planes[0].c2p(1, 1)))
        area_label = TexText("Area", "=", "$A$")
        area_label.set_color_by_tex("$A$", YELLOW)
        area_label.next_to(square, UP)
        area_label.add_background_rectangle()
        self.play(
            DrawBorderThenFill(square),
            FadeIn(area_label, 0.25 * UP, rate_func=squish_rate_func(smooth, 0.5, 1))
        )
        self.wait()

        # Arrow
        arrow = Arrow(*planes)
        arrow_label = Text("Linear transformation", font_size=30)
        arrow_label.next_to(arrow, UP)
        mat_mob = Matrix([["a", "b"], ["c", "d"]], h_buff=0.7, v_buff=0.7)
        mat_mob.set_height(0.7)
        mat_mob.next_to(arrow, DOWN)

        # Apply matrix
        matrix = [
            [0.5, 0.4],
            [0.25, 0.75],
        ]

        for mob in planes[0], square:
            mob.output = mob.deepcopy()
            mob.output.apply_matrix(matrix, about_point=planes[0].c2p(0, 0))
            mob.output.move_to(planes[1].get_center())

        planes[0].output.set_stroke(width=1, opacity=1)
        planes[0].output.faded_lines.set_opacity(0)

        self.play(
            ReplacementTransform(planes[0].copy().fade(1), planes[0].output, run_time=2),
            ReplacementTransform(square.copy().fade(1), square.output, run_time=2),
            ShowCreation(arrow),
            FadeIn(arrow_label, 0.25 * RIGHT),
            FadeIn(mat_mob, 0.25 * RIGHT),
        )
        self.wait()

        # New area
        new_area_label = Tex(
            "\\text{Area} = ", "{c}", "\\cdot", "{A}",
            tex_to_color_map={
                "{c}": RED,
                "{A}": YELLOW,
            }
        )
        new_area_label.add_background_rectangle()
        new_area_label.next_to(square.output, UP)
        new_area_label.shift(0.5 * RIGHT)

        mmc = mat_mob.copy()
        mmc.scale(1.5)
        det = VGroup(get_det_text(mmc), mmc)
        det.set_height(new_area_label.get_height() * 1.2)
        det.move_to(new_area_label.get_part_by_tex("c"), RIGHT)
        det.match_y(new_area_label[-1])
        det_name = TexText("``Determinant''", font_size=36)
        det_name.next_to(det, UP, MED_LARGE_BUFF)
        det_name.set_color(RED)
        det_name.add_background_rectangle()

        self.play(FadeTransform(area_label.copy(), new_area_label))
        self.wait()
        self.play(
            FadeTransform(mat_mob.copy(), det),
            FadeTransform(new_area_label.get_part_by_tex("c"), det_name),
            new_area_label[1].animate.next_to(det, LEFT, SMALL_BUFF).match_y(new_area_label[1]),
        )
        self.wait()


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
            self.init_frame_rotation()
            # frame_speed = -0.02
            # frame.add_updater(lambda f, dt: f.increment_theta(frame_speed * dt))
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

        # Just hang around
        self.wait(15)

        # Change to cat
        cat = SVGMobject("cat_outline").family_members_with_points()[0]
        dog = SVGMobject("dog_outline").family_members_with_points()[0]
        dog.insert_n_curves(87)
        for mob in cat, dog:
            mob.match_style(preimage)
            mob.replace(preimage, dim_to_match=0)
        pass

        # Stretch
        self.play(rotated.rot_speed_tracker.animate.set_value(0))
        rotated.rot_speed = 0
        for axis, diag in zip((0, 1, 0, 1), (False, False, True, True)):
            preimage.generate_target()
            if diag:
                preimage.target.rotate(PI / 4)
            preimage.target.stretch(2, axis)
            if diag:
                preimage.target.rotate(-PI / 4)
            self.play(
                MoveToTarget(preimage),
                rate_func=there_and_back,
                run_time=4
            )
        self.wait(5)
        self.play(rotated.rot_speed_tracker.animate.set_value(0.1))

        # Change shape
        cat = SVGMobject("cat_outline").family_members_with_points()[0]
        dog = SVGMobject("dog_outline").family_members_with_points()[0]
        dog.insert_n_curves(87)
        for mob in cat, dog:
            mob.match_style(preimage)
            mob.replace(preimage, dim_to_match=0)
        self.play(Transform(preimage, cat, run_time=4))
        cat.insert_n_curves(87)
        preimage.become(cat)
        self.wait(2)

        # More shape changes
        self.play(
            preimage.animate.scale(2),
            rate_func=there_and_back,
            run_time=3,
        )
        self.play(
            preimage.animate.become(dog),
            path_arc=PI,
            rate_func=there_and_back_with_pause,  # Or rather, with paws...
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
                frame.animate.reorient(20, 70).set_height(8),
            ]
            frame.clear_updaters()
        if self.display_mode == "shadow_only":
            anims += [
                frame.animate.set_height(10),
                angle_anim,
            ]
        self.play(*anims, run_time=4)
        self.wait()
        rotated.axis_tracker.move_to(UP)
        self.play(
            rotated.angle_tracker.animate.set_value(70 * DEGREES + TAU),
            run_time=2
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
        self.remove(self.background)
        self.student_says(
            TexText("Isn't that obvious?"),
            bubble_config={
                "height": 3,
                "width": 4,
                "direction": LEFT,
            },
            target_mode="angry",
            look_at=self.screen,
            added_anims=[LaggedStart(
                self.teacher.change("guilty"),
                self.students[0].change("pondering", self.screen),
                self.students[1].change("erm", self.screen),
            )]
        )
        self.wait(2)
        self.play(
            self.students[0].change("hesitant"),
        )
        self.wait(2)


class StretchLabel(Scene):
    def construct(self):
        label = VGroup(
            Vector(0.5 * LEFT),
            Tex("1.5 \\times"),
            Vector(0.5 * RIGHT)
        )
        label.set_color(YELLOW)
        label.arrange(RIGHT, buff=SMALL_BUFF)

        self.play(
            *map(ShowCreation, label[::2]),
            Write(label[1]),
        )
        self.wait()


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

    def setup(self):
        super().setup()
        np.random.seed(self.random_seed)
        frame = self.camera.frame
        frame.set_height(5.0)
        frame.set_z(1.75)
        frame.set_theta(self.theta0)
        face = self.solid
        face.shift(0.25 * IN)
        fc = face.get_center()
        z_axis = self.z_axis = VGroup(Line(ORIGIN, fc), Line(fc, 10 * OUT))
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
        self.face = self.solid = face

        arrow_shadow = get_shadow(arrows)
        arrow_shadow.set_stroke(width=1)
        arrow_shadow[2].set_stroke(width=[1, 1, 4, 0])
        self.add(arrow_shadow)

        self.add(z_axis[0], face, z_axis[1])

    def construct(self):
        frame = self.camera.frame
        face = self.face

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


class AverageFaceShadow(SingleFaceRandomRotation):
    inf_light = True
    plane_dims = (16, 8)
    n_samples = 50

    def construct(self):
        # Random shadows
        self.camera.frame.set_height(6)
        face = self.face
        shadow = self.shadow
        shadow.add_updater(lambda m: m.set_fill(BLACK, 0.25))
        shadow.update()
        point = face[0].get_center()
        shadows = VGroup()
        n_samples = self.n_samples
        self.remove(self.z_axis)

        self.init_frame_rotation()
        self.add(shadows)
        for n in range(n_samples):
            self.randomly_reorient(face, about_point=point)
            if n == n_samples - 1:
                normal = next(
                    sm.get_unit_normal()
                    for sm in face.family_members_with_points()
                    if isinstance(sm, VMobject) and sm.get_fill_opacity() > 0
                )
                mat = z_to_vector(normal)
            #     face.apply_matrix(np.linalg.inv(mat), about_point=point)
            shadow.update()
            sc = shadow.copy()
            sc.clear_updaters()
            shadows.add(sc)
            shadows.set_fill(BLACK, 1.5 / len(shadows))
            shadows.set_stroke(opacity=10 / len(shadows))
            self.wait(0.1)

        # Fade out shadow
        self.remove(shadow)
        sc = shadow.copy().clear_updaters()
        self.play(FadeOut(sc))
        self.wait()

        # Scaling
        self.play(
            face.animate.scale(0.5, about_point=point),
            shadows.animate.scale(0.5, about_point=ORIGIN),
            run_time=3,
            rate_func=there_and_back,
        )
        for axis in [0, 1]:
            self.play(
                face.animate.stretch(2, axis, about_point=point),
                shadows.animate.stretch(2, axis, about_point=ORIGIN),
                run_time=3,
                rate_func=there_and_back,
            )
        self.wait()

        # Ambient rotations 106 plays
        self.play(
            self.camera.frame.animate.reorient(-10).shift(2 * LEFT),
        )
        self.add(shadow)
        for n in range(100):
            self.randomly_reorient(face, about_point=point)
            self.wait(0.2)


class AverageCatShadow(AverageFaceShadow):
    n_samples = 50

    def setup(self):
        super().setup()
        self.replace_face()

    def replace_face(self):
        face = self.face

        shape = self.get_shape().family_members_with_points()[0]
        shape.match_style(face[0])
        shape.replace(face[0])

        face[0].set_points(shape.get_points())
        face[0].set_gloss(0.25)
        face[0][0].set_gloss(0)

        self.solid = face
        self.remove(self.shadow)
        self.add_shadow()

    def get_shape(self):
        return SVGMobject("cat_outline")


class AveragePentagonShadow(AverageCatShadow):
    def get_shape(self):
        return RegularPolygon(5)


class AverageShadowAnnotation(Scene):
    def construct(self):
        # Many shadows
        many_shadows = Text("Many shadows")
        many_shadows.move_to(3 * DOWN)

        self.play(Write(many_shadows))
        self.wait(2)

        # Formula
        # shape_name = "2d shape"
        shape_name = "Square"
        t2c = {
            "Shadow": GREY_B,
            shape_name: BLUE,
            "$c$": RED,
        }
        formula = VGroup(
            TexText(
                f"Area(Shadow({shape_name}))",
                tex_to_color_map=t2c,
            ),
            Tex("=").rotate(PI / 2),
            TexText(
                "$c$", " $\\cdot$", f"(Area({shape_name}))",
                tex_to_color_map=t2c
            )
        )
        overline = get_overline(formula[0])
        formula[0].add(overline)
        formula.arrange(DOWN)
        formula.to_corner(UL)

        self.play(FadeTransform(many_shadows, formula[0]))
        self.wait()
        self.play(
            VShowPassingFlash(
                overline.copy().insert_n_curves(100).set_stroke(YELLOW, 5),
                time_width=0.75,
                run_time=2,
            )
        )
        self.wait()
        self.play(
            Write(formula[1]),
            FadeIn(formula[2], DOWN)
        )
        self.wait()

        # Append half
        half = Tex("\\frac{1}{2}")
        half.set_color(RED)
        c = formula[2].get_part_by_tex("$c$")
        half.move_to(c, RIGHT)

        self.play(
            FadeOut(c, 0.5 * UP),
            FadeIn(half, 0.5 * UP),
        )
        self.wait()


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


class ComingUp(VideoWrapper):
    title = "Bob will compute this directly"
    wait_time = 10
    animate_boundary = False


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

        n_lat_lines = 40
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
        sphere_dots.set_radius(0.0125)
        sphere_dots.set_glow_factor(0.5)
        sphere_dots.make_3d()
        sphere_dots.apply_depth_test()
        sphere_dots.add_updater(lambda m: m)

        sphere_lines = VGroup(*(
            Line(sphere.get_center(), p)
            for p in sphere_dots.get_points()
        ))
        sphere_lines.set_stroke(WHITE, 1, 0.05)

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
            ShowIncreasingSubsets(sphere_lines),
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
            run_time=30,
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
            self.add(dot)
            self.play(
                face.animate.apply_matrix(rotation_between_vectors(
                    normal_vect.get_vector(),
                    point - fc
                ), about_point=fc),
                FadeOut(last_dot, run_time=0.25),
                FadeIn(dot),
                run_time=0.5,
            )
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
        frame.reorient(-5)
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
            FadeOut(sphere_lines),
            frame.animate.reorient(0),
            run_time=3,
        )
        self.remove(sphere_dots)

        # Show patch
        def get_patch(u, v, delta_u=0.05, delta_v=0.1):
            patch = ParametricSurface(
                sphere.uv_func,
                u_range=(u * TAU, (u + delta_u) * TAU),
                v_range=(v * PI, (v + delta_v) * PI),
            )
            patch.shift(fc)
            patch.set_color(YELLOW, 0.75)
            patch.always_sort_to_camera(self.camera)
            return patch

        patch = get_patch(0.85, 0.6)
        self.add(patch, sphere)

        self.play(
            ShowCreation(patch),
            frame.animate.reorient(10, 75),
            run_time=2,
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

        # Many patches
        patches = Group(
            get_patch(0.65, 0.5),
            get_patch(0.55, 0.8),
            get_patch(0.85, 0.8),
            get_patch(0.75, 0.4, 0.1, 0.2),
        )

        patch.deactivate_depth_test()
        self.add(sphere, patch)
        for new_patch in patches:
            self.play(
                Transform(patch, new_patch),
            )
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

        factor = 1
        theta_samples = np.linspace(0, PI, factor * sphere_mesh.resolution[0])
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

        tsi = factor * 6  # theta sample index
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


class AskAboutAverageCosValue(AllPossibleOrientations):
    def construct(self):
        self.remove(self.solid)
        self.remove(self.shadow)
        frame = self.camera.frame
        frame.set_height(5)
        frame.reorient(-5, 80)
        frame.shift(2 * RIGHT)
        self.init_frame_rotation()

        # Copy pasting from above...not great
        fc = 3 * OUT
        sphere = Sphere(radius=1)
        sphere.set_color(GREY_E, 0.25)
        sphere.move_to(fc)
        sphere.always_sort_to_camera(self.camera)

        sphere_mesh = SurfaceMesh(sphere, resolution=(21, 11))
        sphere_mesh.set_stroke(BLUE_E, 1, 0.5)

        for sm in sphere_mesh.get_family():
            sm.uniforms["anti_alias_width"] = 0

        self.add(sphere, sphere_mesh)

        normal_vect = Arrow(sphere.get_center(), sphere.pfp(0.2), buff=0)

        def randomly_place_vect():
            theta = random.uniform(0.1, PI - 0.1)
            phi = random.uniform(-PI / 4, PI / 4) + random.choice([0, PI])
            point = fc + np.array([
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            ])
            normal_vect.put_start_and_end_on(sphere.get_center(), point)

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
        theta.add_updater(lambda m: m.next_to(
            arc.pfp(0.5), arc.pfp(0.5) - fc, buff=0.05)
        )

        z_axis = Line(ORIGIN, 10 * OUT)
        z_axis.set_stroke(WHITE, 1)
        z_axis.apply_depth_test()

        self.add(z_axis, normal_vect, arc, theta)
        self.add(sphere_mesh, sphere)

        # Show random samples
        question = TexText("What's the mean?")
        question.to_corner(UR)
        question.to_edge(UP, buff=MED_SMALL_BUFF)
        question.fix_in_frame()
        arrow = Arrow(question, question.get_center() + DOWN)
        arrow.fix_in_frame()

        values = VGroup()
        lhss = VGroup()

        self.add(question, arrow)

        for n in range(15):
            randomly_place_vect()
            lhs = Tex("|\\cos(\\theta_{" + str(n + 1) + "})| = ", font_size=30)
            value = DecimalNumber(abs(math.cos(get_theta())), font_size=30)
            value.next_to(values, DOWN)
            for mob in lhs, value:
                mob.fix_in_frame()
                mob.set_backstroke()
            values.add(value)
            values.next_to(arrow, DOWN)
            lhs.next_to(value, LEFT, buff=SMALL_BUFF)
            lhss.add(lhs)

            self.add(values, lhss)
            self.wait(0.5)


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
                student.change("pondering", image.target)
                for student in students
            ), run_time=2, lag_ratio=0.2),
            teacher.change("tease")
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
                    look_at=image,
                    bubble_config={
                        "direction": LEFT,
                        "width": 3,
                        "height": height,
                    },
                    bubble_type=ThoughtBubble,
                    run_time=2
                )
            )
        self.wait(4)

        # Let's go over the definition
        integral = Tex("\\int_0^\\pi \\dots d\\theta")
        integral.move_to(self.hold_up_spot, DOWN)
        brace = Brace(integral, UP)
        words = TexText("Let's go over the definition", font_size=36)
        words.next_to(brace, UP, SMALL_BUFF)
        words2 = TexText("It can't hurt, right?", font_size=36)
        words2.move_to(words)
        VGroup(brace, words, words2).set_color(YELLOW)

        self.play(
            LaggedStart(*(
                FadeOut(VGroup(student.bubble, student.bubble.content))
                for student in reversed(students)
            )),
            LaggedStart(*(
                student.change("pondering", integral)
                for student in students
            )),
            FadeIn(integral, UP),
            teacher.change("raise_right_hand", integral),
        )
        self.play(
            GrowFromCenter(brace),
            Write(words)
        )
        self.wait(2)
        self.play(
            words.animate.shift(0.75 * UP).set_opacity(0.5),
            FadeIn(words2, 0.2 * UP),
            LaggedStart(
                self.teacher.change("shruggie"),
                self.students[0].change("sassy", words2),
                self.students[1].change("thinking", words2),
                self.students[2].change("well", words2),
            )
        )
        self.wait(2)
        self.play(self.teacher.change("speaking", words2))
        self.wait(3)


class ParticularValuesUnhelpfulOverlay(Scene):
    def construct(self):
        # Particular value
        expr = Tex("P(\\theta =", "\\pi / 4", ")", "=", "0")
        expr.set_color_by_tex("\\pi / 4", YELLOW)
        brace = Brace(expr.get_part_by_tex("\\pi / 4"), UP, buff=SMALL_BUFF)
        brace.stretch(0.5, 1, about_edge=DOWN)
        words = Text("Some specific value", font_size=24)
        words.next_to(brace, UP, SMALL_BUFF)
        VGroup(brace, words).set_color(YELLOW)
        VGroup(expr, brace, words).to_corner(UR)

        self.play(FadeIn(expr, lag_ratio=1))
        self.play(
            GrowFromCenter(brace),
            FadeIn(words, shift=0.2 * UP),
        )
        self.wait()

        # Unhelpful
        question = TexText("What are you going\\\\to do with that?", font_size=24)
        question.next_to(expr, DOWN, LARGE_BUFF)
        arrow = Arrow(question, expr.get_part_by_tex("0"), buff=SMALL_BUFF)

        self.play(
            FadeIn(question),
            ShowCreation(arrow),
        )
        self.wait()

        # New expr
        range_expr = Tex(
            "P(\\pi / 4 < \\theta < \\pi / 4 + \\Delta\\theta) > 0",
            tex_to_color_map={
                "\\pi / 4": YELLOW,
                "\\Delta\\theta": GREY_A,
            },
            font_size=40
        )
        range_expr.move_to(expr, RIGHT)
        new_brace = Brace(range_expr.slice_by_tex("\\pi / 4", ")"), UP, buff=SMALL_BUFF)
        new_words = Text("Range of values", font_size=24)
        new_words.next_to(new_brace, UP, SMALL_BUFF)
        VGroup(new_brace, new_words).set_color(YELLOW)

        self.play(
            FadeOut(question),
            FadeOut(arrow),
            TransformMatchingShapes(expr, range_expr),
            FadeTransform(brace, new_brace),
            FadeTransform(words, new_words),
        )
        self.wait()


class SurfaceAreaOfSphere(Scene):
    def construct(self):
        sphere = Sphere(radius=3)
        sphere.set_color(BLUE_E, 1)
        sphere.always_sort_to_camera(self.camera)
        sphere.rotate(5 * DEGREES, OUT)
        sphere.rotate(80 * DEGREES, LEFT)
        sphere.move_to(0.5 * DOWN)

        sphere_mesh = SurfaceMesh(sphere)
        sphere_mesh.set_stroke(WHITE, 0.5, 0.5)

        equation = Tex(
            "\\text{Surface area} = 4\\pi R^2",
            tex_to_color_map={
                "R": YELLOW,
                "\\text{Surface area}": BLUE,
            },
        )
        equation.to_edge(UP)

        self.add(equation, sphere, sphere_mesh)
        self.play(
            Write(sphere_mesh, lag_ratio=0.02, stroke_width=1),
            ShowCreation(sphere, rate_func=squish_rate_func(smooth, 0.25, 1)),
            run_time=5,
        )
        self.wait()


class IntegralOverlay(Scene):
    def construct(self):
        integral = Tex("\\int_0^\\pi")
        integral.set_color(YELLOW)

        self.play(Write(integral, run_time=2))
        self.play(integral.animate.set_color(WHITE))
        self.play(LaggedStart(*(
            FlashAround(sm, time_width=3)
            for sm in integral[0][:0:-1]
        ), lag_ratio=0.5, run_time=3))
        self.wait()


class AlicesInsights(Scene):
    def construct(self):
        title = TexText("Alice's insights", font_size=72)
        title.to_edge(UP)
        title.set_backstroke()
        underline = Underline(title, buff=-0.05)
        underline.scale(1.5)
        underline.insert_n_curves(50)
        underline.set_stroke(WHITE, width=[0, *4 * [3], 0], opacity=1)
        self.add(underline, title)

        kw = dict(
            t2c={
                "double cover": YELLOW,
                "mean": RED,
                "means": RED,
                "sum": BLUE,
                "Sum": BLUE,
                "Average": RED,
            }
        )
        insights = VGroup(
            Text("1. The face shadows double cover the cube shadow", **kw),
            # Text("2. The mean of the sum is the sum of the means", **kw),
            Text("2. Average(Sum(Face shadows)) = Sum(Average(Face shadow))", **kw),
            Text("3. Use a sphere to deduce the unknown constant", **kw),
        )
        insights.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        insights.next_to(underline, DOWN, LARGE_BUFF)
        insights.to_edge(LEFT)

        self.play(LaggedStart(*(
            FadeIn(insight[:2], 0.25 * DOWN)
            for insight in insights
        )))
        self.wait()
        for insight in insights:
            self.play(FadeIn(insight[2:], lag_ratio=0.1))
            self.wait()


class HalfBathedInLight(ShadowScene):
    def construct(self):
        frame = self.camera.frame
        frame.set_height(12)
        frame.add_updater(lambda m, dt: m.increment_theta(0.05 * dt))
        cube = self.solid
        light = self.light
        light.next_to(cube, OUT, 2)
        self.add(light)
        self.remove(self.plane)
        self.shadow.add_updater(lambda m: m.set_opacity(0))
        cube.move_to(OUT)
        cube.set_opacity(0.95)
        cube.rotate(PI / 2, DL)
        # cube.add_updater(lambda m: self.sort_to_camera(cube))
        cube.update()
        cube.clear_updaters()

        light_lines = self.get_light_lines()
        light_lines.add_updater(lambda m: m.set_stroke(YELLOW, 2))
        self.add(light_lines, light)

        self.wait(2)
        for s, color in zip([slice(3, None), slice(0, 3)], [WHITE, GREY_D]):
            cube.generate_target()
            sorted_cube = Group(*cube.target)
            sorted_cube.sort(lambda p: p[2])
            sorted_cube[s].space_out_submobjects(2)
            sorted_cube[s].set_color(color)
            self.play(
                MoveToTarget(cube),
                rate_func=there_and_back_with_pause,
                run_time=3,
            )
        self.wait(5)

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
        for face in cube:
            face.set_fill(opacity=0.9)
            face.set_reflectiveness(0.1)
            face.set_gloss(0.2)
        cube.add_updater(lambda m: self.sort_to_camera(m))
        cube.rotate(PI / 3, normalize([3, 4, 5]))
        shadow = self.shadow
        outline = self.get_shadow_outline()

        # Inequality
        ineq = self.get_top_expression("$<$")
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
        face_pair = Group(faces[3], faces[5]).copy()
        face_pair[1].set_color(RED)
        face_pair.save_state()
        fp_shadow = get_shadow(face_pair)

        self.add(fp_shadow)
        self.play(
            FadeOut(cube),
            *map(VFadeOut, shadow),
            FadeOut(outline),
            *map(Write, face_pair),
        )
        self.wait(1)
        self.play(Rotate(
            face_pair, PI / 2, DOWN,
            about_point=cube.get_center(),
            run_time=3,
        ))
        self.wait()
        self.random_toss(face_pair, about_point=cube.get_center(), run_time=6)
        fp_shadow.clear_updaters()
        self.play(
            FadeIn(cube),
            *map(VFadeIn, shadow),
            FadeOut(face_pair, scale=0),
            FadeOut(fp_shadow, scale=0),
        )
        self.add(shadow)

        # Half of sum
        new_expression = self.get_top_expression(" = ", "$\\displaystyle \\frac{1}{2}$")
        new_expression.fix_in_frame()
        eq_half = VGroup(
            new_expression.get_part_by_tex("="),
            new_expression.get_part_by_tex("frac"),
        )

        cube.save_state()
        cube.generate_target(use_deepcopy=True)
        cube.target.clear_updaters()
        z_sorted_faces = Group(*sorted(list(cube.target), key=lambda f: f.get_z()))
        z_sorted_faces[:3].shift(2 * LEFT)
        z_sorted_faces[3:].shift(2 * RIGHT)

        cube.clear_updaters()
        self.play(
            MoveToTarget(cube),
            ApplyMethod(frame.reorient, 0, run_time=2)
        )
        self.play(
            FadeOut(lt, UP),
            Write(eq_half),
            ReplacementTransform(rhs, new_expression[-len(rhs):]),
            ReplacementTransform(lhs, new_expression[:len(lhs)]),
        )
        self.remove(ineq)
        self.add(new_expression)
        self.wait(2)
        anims = []
        for part in z_sorted_faces:
            pc = part.copy()
            pc.set_fill(YELLOW, 0.25)
            pc_shadow = get_shadow(pc)
            pc_shadow.clear_updaters()
            pc_shadow.match_style(pc)
            lines = VGroup(*(
                Line(v, flat_project(v))
                for v in pc.get_vertices()
            ))
            lines.set_stroke(YELLOW, 1, 0.1)
            anims.append(AnimationGroup(
                VFadeInThenOut(pc),
                VFadeInThenOut(pc_shadow),
                VFadeInThenOut(lines),
            ))
        self.play(LaggedStart(*anims, lag_ratio=0.4, run_time=6))
        self.play(Restore(cube))

        # Show the double cover
        shadow_point = shadow.get_bottom() + [0.5, 0.75, 0]
        dot = GlowDot(shadow_point)
        line = DashedLine(
            shadow_point + 5 * OUT, shadow_point,
            dash_length=0.025
        )
        line.set_stroke(YELLOW, 1)

        def update_line(line):
            line.move_to(dot.get_center(), IN)
            for dash in line:
                dist = cube_sdf(dash.get_center(), cube)
                dash.set_stroke(
                    opacity=interpolate(0.1, 1.0, clip(10 * dist, -0.5, 0.5) + 0.5)
                )
                dash.inside = (dist < 0)

        line.add_updater(update_line)

        entry_point = next(dash for dash in line if dash.inside).get_center()
        exit_point = next(dash for dash in reversed(line) if dash.inside).get_center()
        arrows = VGroup(*(
            Arrow(point + RIGHT, point, buff=0.1)
            for point in (entry_point, exit_point)
        ))

        self.play(ShowCreation(line, rate_func=rush_into))
        self.play(FadeIn(dot, scale=10, rate_func=rush_from, run_time=0.5))
        self.wait()
        self.play(
            Rotate(
                dot, TAU,
                about_point=ORIGIN,
                run_time=6,
            ),
        )
        self.wait()
        for arrow in arrows:
            self.play(ShowCreation(arrow))
        self.wait()
        self.play(FadeOut(arrows))

        cube.add_updater(lambda m: self.sort_to_camera(m))
        self.random_toss(cube, angle=1.5 * TAU, run_time=8)

        # Just show wireframe
        cube.save_state()
        cube.generate_target()
        for sm in cube.target:
            sm.set_fill(opacity=0)
            sm.set_stroke(WHITE, 2)
        outline = self.get_shadow_outline()
        outline.rotate(PI)
        self.play(
            MoveToTarget(cube),
            dot.animate.move_to(outline.get_start())
        )
        self.play(MoveAlongPath(dot, outline, run_time=8))
        self.wait(2)
        self.play(Restore(cube))
        self.play(dot.animate.move_to(outline.get_center()), run_time=2)

        # Make room for equation animations
        # Start here for new scene
        area_label = self.get_shadow_area_label()
        area_label.shift(1.75 * DOWN + 2.25 * RIGHT)
        area_label[0].scale(0, about_edge=RIGHT)
        area_label.scale(0.7)
        outline.update()
        line.clear_updaters()
        self.play(
            FadeOut(dot),
            FadeOut(line),
            ShowCreation(outline),
            VFadeIn(area_label),
        )

        # Single out a face
        self.remove(new_expression)
        self.wait(2)
        face = cube[np.argmax([f.get_z() for f in cube])].copy()
        face.set_color(YELLOW)
        face_shadow = get_shadow(face)
        face_shadow_area = DecimalNumber(get_norm(face_shadow.get_area_vector()) / (self.unit_size**2))
        face_shadow_area.scale(0.65)
        face_shadow_area.move_to(area_label)
        face_shadow_area.shift(flat_project(face.get_center() - cube.get_center()))
        face_shadow_area.shift(SMALL_BUFF * UR)
        face_shadow_area.fix_in_frame()

        cube.save_state()
        self.remove(cube)
        self.play(
            *(
                f.animate.set_fill(opacity=0)
                for f in cube
            ),
            FadeOut(outline),
            FadeOut(area_label),
            Write(face),
            FadeIn(face_shadow),
            FadeIn(face_shadow_area),
        )
        self.wait(2)
        self.play(
            Restore(cube),
            *map(FadeOut, (face, face_shadow, face_shadow_area)),
            *map(FadeIn, (outline, area_label)),
        )

        # Show simple rotations
        for x in range(2):
            self.random_toss(cube)
            self.wait()
        self.wait()

        # Many random orientations
        for x in range(40):
            self.randomly_reorient(cube)
            self.wait(0.25)

        # Show sum of faces again (play 78)
        self.random_toss(cube, 2 * TAU, run_time=8, meta_speed=10)
        self.wait()

        cube.save_state()
        sff = 1.5
        self.play(
            VFadeOut(outline),
            VFadeOut(area_label),
            cube.animate.space_out_submobjects(sff)
        )
        for x in range(3):
            self.random_toss(cube, angle=PI)
            self.wait()
        self.play(cube.animate.space_out_submobjects(1 / sff))

        # Mean shadow of a single face
        cube_style = cube[0].get_style()

        def isolate_face_anims(i, color=YELLOW):
            return (
                shadow.animate.set_fill(
                    interpolate_color(color, BLACK, 0.75)
                ),
                *(
                    f.animate.set_fill(
                        color if f is cube[i] else BLUE,
                        0.75 if f is cube[i] else 0,
                    )
                    for f in cube
                )
            )

        def tour_orientations():
            self.random_toss(cube, 2 * TAU, run_time=5, meta_speed=10)

        self.play(*isolate_face_anims(5))
        tour_orientations()
        for i, color in ((4, GREEN), (3, RED)):
            self.play(
                *isolate_face_anims(i, color),
            )
            tour_orientations()
        cube.update()
        self.play(
            *(
                f.animate.set_style(**cube_style)
                for f in cube
            ),
            shadow.animate.set_fill(interpolate_color(BLUE, BLACK, 0.85)),
            VFadeIn(outline),
            VFadeIn(area_label),
        )
        self.add(cube)

        # Ambient rotation
        self.add(cube)
        self.begin_ambient_rotation(cube, speed=0.4)
        self.wait(20)

    def get_top_expression(self, *mid_tex, n_faces=6):
        t2c = {
            "Shadow": GREY_B,
            "Cube": BLUE_D,
            "Face$_j$": YELLOW,
        }
        ineq = TexText(
            "Area(Shadow(Cube))",
            *mid_tex,
            " $\\displaystyle \\sum_{j=1}^" + f"{{{n_faces}}}" + "$ ",
            "Area(Shadow(Face$_j$))",
            tex_to_color_map=t2c,
            isolate=["(", ")"],
        )
        ineq.to_edge(UP, buff=MED_SMALL_BUFF)
        return ineq

    def get_faces_and_face_shadows(self):
        faces = self.solid.deepcopy()
        VGroup(*faces).set_fill(self.highlighted_face_color)

        shadows = get_pre_shadow(faces, opacity=0.7)
        shadows.apply_function(flat_project)
        return faces, shadows


class ConvexityPrelude(Scene):
    def construct(self):
        square = Square(side_length=3)
        square.rotate(-PI / 4)
        square.flip()
        square.set_stroke(BLUE, 2)

        points = [square.pfp(1 / 8), square.pfp(7 / 8)]

        beam = VGroup(
            Line(points[0] + 3 * UP, points[0], stroke_width=3),
            Line(points[0], points[1], stroke_width=2),
            Line(points[1], points[1] + 3 * DOWN, stroke_width=1),
        )

        beam.set_stroke(YELLOW)

        words = TexText("2 intersections\\\\", "(almost always)")
        words[1].scale(0.7, about_edge=UP).set_color(GREY_B)
        words.to_edge(LEFT)
        arrows = VGroup(*(
            Arrow(words[0].get_right(), point)
            for point in points
        ))

        dots = GlowDot()
        dots.set_points(points)
        dots.set_color(WHITE)
        dots.set_radius(0.2)
        dots.set_glow_factor(3)

        self.add(square)
        self.add(words)
        self.play(
            ShowCreation(beam),
            FadeIn(arrows, lag_ratio=0.7),
            FadeIn(dots, lag_ratio=0.7),
        )
        self.wait()

        question1 = Text("Why is this true?")
        question2 = Text("Is this true for all shapes?")
        question2.next_to(question1, DOWN, aligned_edge=LEFT)
        VGroup(question1, question2).to_corner(UR)

        self.play(Write(question1, run_time=1))
        self.wait()
        self.play(FadeIn(question2, 0.25 * DOWN))
        self.wait()

        # Convexity
        square.generate_target()
        convex_shapes = VGroup(
            square.target,
            RegularPolygon(5, color=TEAL),
            Rectangle(2, 1, color=TEAL_E),
            RegularPolygon(6, color=GREEN),
            Circle(color=GREEN_B),
        )
        convex_shapes[2].apply_matrix([[1, 0.5], [0, 1]])
        convex_shapes[3].shift(2 * RIGHT).apply_complex_function(np.exp)
        convex_shapes[3].make_jagged()
        for shape in convex_shapes:
            shape.set_height(1)
            shape.set_stroke(width=2)
        convex_shapes.arrange(DOWN)
        convex_shapes.set_height(6)

        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        h_line = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        h_line.set_y(3)
        VGroup(v_line, h_line).set_stroke(WHITE, 2)
        convex_title = Text("Convex")
        non_convex_title = Text("Non-convex")
        for title, vect in zip([convex_title, non_convex_title], [LEFT, RIGHT]):
            title.scale(1.5)
            title.next_to(h_line, UP)
            title.shift(vect * FRAME_WIDTH / 4)

        convex_shapes.next_to(h_line, DOWN)
        convex_shapes.match_x(convex_title)

        self.play(
            MoveToTarget(square),
            FadeIn(convex_shapes[1:], lag_ratio=0.5),
            FadeTransform(beam, v_line),
            ShowCreation(h_line),
            FadeIn(convex_title),
            LaggedStart(*map(FadeOut, (
                dots, arrows, words,
                question1, question2,
            ))),
        )
        self.wait()

        # Non-convex shapes
        pi = Tex("\\pi").family_members_with_points()[0]
        pent = RegularPolygon(5)
        pent.set_points_as_corners([ORIGIN, *pent.get_vertices()[1:], ORIGIN])
        n_mob = Tex("N").family_members_with_points()[0]
        nc_shapes = VGroup(pi, pent, n_mob)
        nc_shapes.set_fill(opacity=0)
        nc_shapes.set_stroke(width=2)
        nc_shapes.set_submobject_colors_by_gradient(RED, PINK)
        for shape in nc_shapes:
            shape.set_height(1)
        nc_shapes.arrange(DOWN)
        nc_shapes.replace(convex_shapes, dim_to_match=1)
        nc_shapes.match_x(non_convex_title)

        self.play(
            Write(non_convex_title, run_time=1),
            LaggedStartMap(FadeIn, nc_shapes),
        )
        self.wait()

        # Embed
        self.embed()


class DefineConvexity(Scene):
    def construct(self):
        # Shape
        definition = "A set is convex if the line connecting any\n"\
                     "two points within it is contained in the set"
        set_color = BLUE
        line_color = GREEN
        title = Text(
            definition,
            t2c={
                "convex": set_color,
                "line connecting any\ntwo points": line_color,
            },
            t2s={"convex": ITALIC},
        )
        title.to_edge(UP)
        title.set_opacity(0.2)

        shape = Square(4.5)
        shape.set_fill(set_color, 0.25)
        shape.set_stroke(set_color, 2)
        shape.next_to(title, DOWN, LARGE_BUFF)

        self.add(title)
        self.play(
            title.get_part_by_text("A set is convex").animate.set_opacity(1),
            Write(shape)
        )
        self.wait()

        # Show two points
        line = Line(shape.pfp(0.1), shape.pfp(0.5))
        line.scale(0.7)
        line.set_stroke(line_color, 2)
        dots = DotCloud(line.get_start_and_end())
        dots.set_color(line_color)
        dots.make_3d(0.5)
        dots.save_state()
        dots.set_radius(0)
        dots.set_opacity(0)

        self.play(
            title[definition.index("if"):definition.index("is", 16)].animate.set_opacity(1),
            Restore(dots),
        )
        self.play(ShowCreation(line))
        self.wait()
        self.play(
            title[definition.index("is", 16):].animate.set_opacity(1),
        )

        # Alternate places
        dots.add_updater(lambda m: m.set_points(line.get_start_and_end()))

        def show_sample_lines(n=5):
            for x in range(5):
                self.play(
                    line.animate.put_start_and_end_on(
                        shape.pfp(random.random()),
                        shape.pfp(random.random()),
                    ).scale(random.random(), about_point=shape.get_center())
                )
                self.wait(0.5)

        show_sample_lines()

        # Letter π
        def tex_to_shape(tex):
            result = Tex(tex).family_members_with_points()[0]
            result.match_style(shape)
            result.match_height(shape)
            result.move_to(shape)
            result_points = result.get_points().copy()
            index = np.argmax([np.dot(p, UR) for p in result_points])
            index = 3 * (index // 3)
            result.set_points([*result_points[index:], *result_points[:index]])
            return result

        pi = tex_to_shape("\\pi")
        letter_c = tex_to_shape("\\textbf{C}")
        letter_c.insert_n_curves(50)

        line.generate_target()
        line.target.put_start_and_end_on(*(
            pi.get_boundary_point(v) for v in (DL, DR)
        ))
        line.target.scale(0.9)
        not_convex_label = Text("Not convex!", color=RED)
        not_convex_label.next_to(pi, LEFT)

        shape.insert_n_curves(100)
        self.play(
            Transform(shape, pi, path_arc=PI / 2),
            MoveToTarget(line),
            run_time=2,
        )
        self.play(
            FadeIn(not_convex_label, scale=2)
        )
        self.wait()
        shape.insert_n_curves(80)
        self.play(
            Transform(shape, letter_c),
            line.animate.put_start_and_end_on(*(
                letter_c.pfp(a) for a in (0, 0.5)
            )),
            run_time=2,
        )
        self.play(UpdateFromAlphaFunc(
            line,
            lambda l, a: l.put_start_and_end_on(
                letter_c.pfp(0.4 * a), l.get_end()
            ),
            run_time=6,
        ))
        self.wait()

        # Polygon
        convex_label = TexText("Convex \\ding{51}")
        convex_label.set_color(YELLOW)
        convex_label.move_to(not_convex_label)
        polygon = RegularPolygon(7)
        polygon.match_height(shape)
        polygon.move_to(shape)
        polygon.match_style(shape)

        self.remove(not_convex_label)
        self.play(
            FadeTransform(shape, polygon),
            line.animate.put_start_and_end_on(
                polygon.get_vertices()[1],
                polygon.get_vertices()[-1],
            ),
            TransformMatchingShapes(not_convex_label.copy(), convex_label),
            run_time=2
        )
        self.wait()

        polygon.generate_target()
        new_tip = 2 * line.get_center() - polygon.get_start()
        polygon.target.set_points_as_corners([
            new_tip, *polygon.get_vertices()[1:], new_tip
        ])
        self.play(
            MoveToTarget(polygon),
            TransformMatchingShapes(convex_label, not_convex_label),
        )
        self.wait()

        # Show light beam
        beam = DashedLine(ORIGIN, FRAME_WIDTH * RIGHT)
        beam.set_stroke(YELLOW, 1)
        beam.next_to(line, DOWN, MED_SMALL_BUFF)
        flash_line = Line(beam.get_start(), beam.get_end())
        flash_line.set_stroke(YELLOW, 5)
        flash_line.insert_n_curves(100)
        self.play(
            ShowCreation(beam),
            VShowPassingFlash(flash_line),
            run_time=1,
        )
        self.wait()


class NonConvexDoughnut(ShadowScene):
    def construct(self):
        # Setup
        frame = self.camera.frame

        self.remove(self.solid, self.shadow)
        torus = Torus()
        torus.set_width(4)
        torus.set_z(3)
        torus.set_color(BLUE_D)
        torus.set_opacity(0.7)
        torus.set_reflectiveness(0)
        torus.set_shadow(0.5)
        torus.always_sort_to_camera(self.camera)

        shadow = get_shadow(torus)
        shadow.always_sort_to_camera(self.camera)

        self.add(torus)
        self.add(shadow)
        self.play(ShowCreation(torus), run_time=2)
        self.play(Rotate(torus, PI / 2, LEFT), run_time=2)
        self.wait()

        # Light beam
        dot = GlowDot(0.25 * LEFT)
        line = DashedLine(dot.get_center(), dot.get_center() + 10 * OUT)
        line.set_stroke(YELLOW, 1)
        line_shadow = line.copy().set_stroke(opacity=0.1)

        self.add(line, torus, line_shadow)
        self.play(
            FadeIn(dot),
            ShowCreation(line),
            ShowCreation(line_shadow),
            ApplyMethod(frame.reorient, 20, run_time=7)
        )
        self.wait()


class ShowGridSum(TwoToOneCover):
    def construct(self):
        # Setup
        self.clear()
        shape_name = "Cube"
        n_faces = 6
        # shape_name = "Dodec."
        # n_faces = 12

        frame = self.camera.frame
        frame.reorient(0, 0)
        frame.move_to(ORIGIN)
        equation = self.get_top_expression(" = ", "$\\displaystyle \\frac{1}{2}$", n_faces=n_faces)
        self.add(equation)

        lhs = equation.slice_by_tex(None, "=")
        summand = equation.slice_by_tex("sum", None)[1:]

        # Abbreviate
        t2c = {
            f"\\text{{{shape_name}}}": BLUE,
            "\\text{F}_j": YELLOW,
            "\\text{Face}": YELLOW,
            "\\text{Total}": GREEN,
            "S": GREY_B,
            "{c}": RED,
        }
        kw = {
            "tex_to_color_map": t2c
        }
        lhs.alt1 = Tex(f"S(\\text{{{shape_name}}})", **kw)
        summand.alt1 = Tex("S(\\text{F}_j)", **kw)

        def get_s_cube_term(i="i"):
            return Tex(f"S\\big(R_{{{i}}}", f"(\\text{{{shape_name}}})\\big)", **kw)

        def get_s_f_term(i="i", j="j"):
            result = Tex(
                f"S\\big(R_{{{i}}}",
                "(", "\\text{F}_{" + str(j) + "}", ")",
                "\\big)",
                **kw
            )
            result[3].set_color(YELLOW)
            return result

        lhs.alt2 = get_s_cube_term(1)
        summand.alt2 = get_s_f_term(1)

        for mob, vect in (lhs, RIGHT), (summand, LEFT):
            mob.brace = Brace(mob, DOWN, buff=SMALL_BUFF)
            mob.alt1.next_to(mob.brace, DOWN)
            self.play(
                GrowFromCenter(mob.brace),
                FadeIn(mob.alt1, shift=0.5 * DOWN),
            )
            self.wait()
            self.play(
                FadeOut(mob.brace, scale=0.5),
                FadeOut(mob, shift=0.5 * UP),
                mob.alt1.animate.move_to(mob, vect),
            )
            mob.alt2.move_to(mob, vect)
        self.wait()

        for mob in lhs, summand:
            self.play(TransformMatchingShapes(mob.alt1, mob.alt2))
            self.wait()

        # Add up many rotations
        lhss = VGroup(
            get_s_cube_term(1),
            get_s_cube_term(2),
            get_s_cube_term(3),
            Tex("\\vdots"),
            get_s_cube_term("n"),
        )
        buff = 0.6
        lhss.arrange(DOWN, buff=buff)
        lhss.move_to(lhs.alt2, UP)

        self.remove(lhs.alt2)
        self.play(
            LaggedStart(*(
                ReplacementTransform(lhs.alt2.copy(), target)
                for target in lhss
            )),
            frame.animate.set_height(10, about_edge=UP),
            run_time=2,
        )

        # Show empirical mean
        h_line = Line(LEFT, RIGHT)
        h_line.set_width(lhss.get_width() + 0.75)
        h_line.next_to(lhss, DOWN, MED_SMALL_BUFF, aligned_edge=RIGHT)
        plus = Tex("+")
        plus.align_to(h_line, LEFT).shift(0.1 * RIGHT)
        plus.match_y(lhss[-1])
        total = Text("Total", font_size=60)
        total.set_color(GREEN)
        total.next_to(h_line, DOWN, buff=0.35)
        total.match_x(lhss)

        mean_sa = Tex(
            f"S(\\text{{{shape_name}}})", "=", "\\frac{1}{n}",
            "\\sum_{i=1}^n S\\big(R_i(" + f"\\text{{{shape_name}}})\\big)",
            **kw,
        )

        mean_sa.add_to_back(get_overline(mean_sa.slice_by_tex(None, "=")))
        mean_sa.next_to(total, DOWN, LARGE_BUFF, aligned_edge=RIGHT)

        corner_rect = SurroundingRectangle(mean_sa, buff=0.25)
        corner_rect.set_stroke(WHITE, 2)
        corner_rect.set_fill(GREY_E, 1)
        corner_rect.move_to(frame, DL)
        corner_rect.shift(0.025 * UR)
        mean_sa.move_to(corner_rect)

        sum_part = mean_sa.slice_by_tex("sum")
        sigma = sum_part[0]
        sigma.save_state()
        lhss_rect = SurroundingRectangle(lhss)
        lhss_rect.set_stroke(BLUE, 2)
        sigma.next_to(lhss_rect, LEFT)
        sum_group = VGroup(lhss, lhss_rect)

        self.play(
            Write(lhss_rect),
            Write(sigma),
        )
        self.wait()
        self.add(corner_rect, sigma)
        self.play(
            FadeIn(corner_rect),
            *(
                FadeTransform(term.copy(), sum_part[1:])
                for term in lhss
            ),
            Restore(sigma),
        )
        self.play(Write(mean_sa.get_part_by_tex("frac")))
        self.wait()
        self.play(
            FadeIn(mean_sa.slice_by_tex(None, "frac")),
        )
        self.wait()

        # Create grid
        sf = get_s_f_term
        grid_terms = [
            [sf(1, 1), sf(1, 2), Tex("\\dots"), sf(1, n_faces)],
            [sf(2, 1), sf(2, 2), Tex("\\dots"), sf(2, n_faces)],
            [sf(3, 1), sf(3, 2), Tex("\\dots"), sf(3, n_faces)],
            [Tex("\\vdots"), Tex("\\vdots"), Tex("\\ddots"), Tex("\\vdots")],
            [sf("n", 1), sf("n", 2), Tex("\\dots"), sf("n", n_faces)],
        ]
        grid = VGroup(*(VGroup(*row) for row in grid_terms))
        for lhs, row in zip(lhss, grid):
            for i in range(len(row) - 1, 0, -1):
                is_dots = "dots" in row[0].get_tex()
                sym = VectorizedPoint() if is_dots else Tex("+")
                row.insert_submobject(i, sym)
            row.arrange(RIGHT, buff=MED_SMALL_BUFF)
            for m1, m2 in zip(row, grid[0]):
                m1.match_x(m2)
                m1.match_y(lhs)
            if not is_dots:
                parens = Tex("[]", font_size=72)[0]
                parens.set_stroke(width=2)
                parens.set_color(BLUE_B)
                parens[0].next_to(row, LEFT, buff=SMALL_BUFF)
                parens[1].next_to(row, RIGHT, buff=SMALL_BUFF)
                row.add(*parens)
                eq_half = Tex("=", "\\frac{1}{2}")
                eq_half[1].match_height(parens)
                eq_half.next_to(parens[0], LEFT, MED_SMALL_BUFF)
                row.add(*eq_half)

        grid.set_x(frame.get_right()[0] - 1.5, RIGHT)

        self.remove(summand.alt2)
        self.play(
            sum_group.animate.set_x(grid.get_left()[0] - MED_SMALL_BUFF, RIGHT),
            TransformMatchingShapes(
                VGroup(
                    equation.get_part_by_tex("frac"),
                    equation.get_part_by_tex("="),
                ),
                grid[0][-4:],
            ),
            LaggedStart(*(
                FadeTransform(summand.alt2.copy(), part)
                for part in grid[0][0:7:2]
            )),
            FadeOut(equation.get_part_by_tex("sum"), scale=0.25),
            Write(grid[0][1:7:2]),  # Plus signs
        )
        self.wait(2)
        self.play(FadeTransform(grid[0].copy(), grid[1]))
        self.play(FadeTransform(grid[1].copy(), grid[2]))
        self.play(FadeTransform(grid[2].copy(), grid[4]), FadeIn(grid[3]))
        self.wait(2)

        # Average along columns
        cols = VGroup(*(
            VGroup(*(row[i] for row in grid)).copy()
            for i in [0, 2, 6]
        ))
        col_rects = VGroup(*(
            SurroundingRectangle(col, buff=SMALL_BUFF)
            for col in cols
        ))
        col_rects.set_stroke(YELLOW, 1)

        mean_face = Tex("S(\\text{Face})", **kw)
        mean_face.add_to_back(get_overline(mean_face))
        mean_face.next_to(grid, DOWN, buff=2)
        mean_face_words = TexText("Average shadow\\\\of one face")
        mean_face_words.move_to(mean_face, UP)

        arrows = VGroup(*(
            Arrow(rect.get_bottom(), mean_face)
            for rect in col_rects
        ))
        arrow_labels = Tex("\\frac{1}{n} \\sum \\cdots", font_size=30).replicate(3)
        for arrow, label in zip(arrows, arrow_labels):
            vect = rotate_vector(normalize(arrow.get_vector()), PI / 2)
            label.next_to(arrow.pfp(0.5), vect, SMALL_BUFF)

        self.add(cols[0])
        self.play(
            grid.animate.set_opacity(0.4),
            ShowCreation(col_rects[0])
        )
        self.wait()
        self.play(
            ShowCreation(arrows[0]),
            FadeIn(arrow_labels[0]),
            FadeIn(mean_face_words, DR),
        )
        self.wait()
        self.play(
            mean_face_words.animate.scale(0.7).next_to(mean_face, DOWN, MED_LARGE_BUFF),
            FadeIn(mean_face, scale=2),
        )
        self.wait()
        for i in [1, 2]:
            self.play(
                FadeOut(cols[i - 1]),
                FadeIn(cols[i]),
                *(
                    ReplacementTransform(group[i - 1], group[i])
                    for group in (col_rects, arrows, arrow_labels)
                )
            )
            self.wait()
        self.wait()

        # Reposition
        frame.generate_target()
        frame.target.align_to(total, DOWN)
        frame.target.shift(0.5 * DOWN)
        frame.target.scale(1.15)
        frame.target.align_to(lhss, LEFT).shift(0.25 * LEFT)

        self.play(LaggedStart(
            VGroup(corner_rect, mean_sa).animate.scale(1.25).move_to(
                frame.target, UL
            ).shift(0.1 * DR),
            MoveToTarget(frame),
            FadeOut(mean_face_words),
            FadeOut(mean_face),
            grid.animate.set_opacity(1),
            *(
                FadeOut(group[-1])
                for group in (cols, col_rects, arrows, arrow_labels)
            ),
            run_time=3
        ))
        mean_sa.refresh_bounding_box()  # ??

        # Show final result
        rhss = VGroup(
            Tex("=", "\\frac{1}{2}", "\\sum_{j=1}^" + f"{{{n_faces}}}", " S(\\text{F}_j})", **kw),
            Tex("=", "\\frac{1}{2}", "\\sum_{j=1}^" + f"{{{n_faces}}}", " {c}", "\\cdot ", "A(\\text{F}_j)", **kw),
            Tex("=", "\\frac{1}{2}", "{c}", "\\cdot ", "(\\text{Surface area})", **kw),
        )
        rhss[0].add(get_overline(rhss[0].slice_by_tex("S")))
        rhss[2][-2].set_color(WHITE)

        rhss.arrange(RIGHT)
        rhss.next_to(mean_sa, RIGHT)

        corner_rect.generate_target()
        corner_rect.target.set_width(
            frame.get_width() - 0.2,
            stretch=True,
            about_edge=LEFT,
        )

        grid_rect = SurroundingRectangle(grid, buff=SMALL_BUFF)
        grid_rect.set_stroke(YELLOW, 1)
        grid_rect.set_fill(YELLOW, 0.25)

        rects = VGroup(
            SurroundingRectangle(mean_sa.slice_by_tex("frac")),
            SurroundingRectangle(rhss[0][1:])
        )
        for rect in rects:
            rect.match_height(corner_rect.target, stretch=True)
            rect.match_y(corner_rect.target)
        rects[0].set_color(BLUE)
        rects[1].set_color(YELLOW)
        rects.set_stroke(width=2)
        rects.set_fill(opacity=0.25)

        rows_first = Text("Rows first")
        rows_first.next_to(rects[0], DOWN)
        rows_first.match_color(rects[0])
        cols_first = Text("Columns first")
        cols_first.next_to(rects[1], DOWN)
        cols_first.match_color(rects[1])

        self.add(corner_rect, mean_sa)
        self.play(
            MoveToTarget(corner_rect),
            Write(rhss[0])
        )
        self.add(grid_rect, grid)
        self.play(VFadeInThenOut(grid_rect, run_time=2))
        self.wait()

        self.play(
            Write(rects[0]),
            Write(rows_first),
        )
        self.wait()
        self.play(
            Write(rects[1]),
            Write(cols_first),
        )
        self.wait()
        self.play(LaggedStart(*map(FadeOut, (
            *rects, rows_first, cols_first
        ))))

        self.play(
            TransformMatchingShapes(rhss[0].copy(), rhss[1])
        )
        self.wait()
        self.play(
            FadeTransform(rhss[1].copy(), rhss[2]),
        )
        self.wait()

        key_part = rhss[2][1:]
        final_rect = SurroundingRectangle(key_part)
        final_rect.set_stroke(YELLOW, 1)
        self.play(
            corner_rect.animate.set_stroke(width=0).scale(1.1),
            FlashAround(key_part, time_width=1.5),
            FadeIn(final_rect),
            run_time=2,
        )


class LimitBrace(Scene):
    def construct(self):
        brace = Brace(Line().set_width(3), UP)
        tex = brace.get_tex("n \\to \\infty")
        VGroup(brace, tex).set_color(TEAL)
        VGroup(brace, tex).set_backstroke()
        self.play(
            GrowFromCenter(brace),
            FadeIn(tex, shift=0.25 * UP)
        )
        self.wait()


class FromRowsToColumns(Scene):
    def construct(self):
        n = 5
        grid = Dot().get_grid(n, n, buff=MED_LARGE_BUFF)

        grids = grid.get_grid(1, 2, buff=3)

        buff = 0.2
        row_rects = VGroup(*(
            SurroundingRectangle(grids[0][k:k + n], buff=buff)
            for k in range(0, n * n, n)
        ))
        col_rects = VGroup(*(
            SurroundingRectangle(grids[1][k::n], buff=buff)
            for k in range(n)
        ))
        rects = VGroup(row_rects, col_rects)
        rects.set_fill(opacity=0.25)
        rects.set_stroke(width=2)
        row_rects.set_color(BLUE)
        col_rects.set_color(YELLOW)

        # plus_template = Tex("+")
        # plus_template.match_height(grids[0][0])
        # for grid in grids:
        #     plusses = VGroup()
        #     for k, dot in enumerate(grid):
        #         if k % n != n - 1:
        #             pc = plus_template.copy()
        #             pc.move_to(midpoint(dot.get_center(), grid[k + 1].get_center()))
        #             plusses.add(pc)
        #     grid.add(plusses)

        arrow = Arrow(grids[0], grids[1], buff=0.5)

        self.add(grids[0])
        self.play(FadeIn(row_rects, lag_ratio=0.2))
        self.wait()
        self.play(
            TransformFromCopy(grids[0], grids[1]),
            ShowCreation(arrow)
        )
        self.play(FadeIn(col_rects, lag_ratio=0.2))
        self.wait()


class ComplainAboutProgress(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            TexText("Wait, is that all\\\\we've accomplished?"),
            target_mode="angry",
        )
        self.play_student_changes(
            "guilty", "erm",
            look_at=self.students[2].eyes,
            added_anims=[self.teacher.change("guilty")],
        )
        self.wait(4)


class SupposedlyObviousProportionality(ShadowScene):
    solid_name = "Cube"

    def construct(self):
        # Setup
        cube = self.solid
        shadow = self.shadow
        frame = self.camera.frame
        frame.set_z(3)
        frame.reorient(-20, 80)
        self.init_frame_rotation()
        light = self.light
        light.next_to(cube, OUT, 50)
        equation = get_key_result(self.solid_name)
        equation.fix_in_frame()
        equation.to_edge(UP)
        self.add(equation)

        # Rotation
        self.begin_ambient_rotation(cube)
        self.wait()

        # Ask about constant
        question = TexText(
            "What is $c$?!",
            font_size=72,
            tex_to_color_map={"$c$": RED}
        )
        question.fix_in_frame()
        question.next_to(equation, DOWN, LARGE_BUFF)
        question.to_edge(RIGHT, buff=LARGE_BUFF)
        c_arrow = Arrow(
            equation.get_part_by_tex("{c}"),
            question.get_corner(UL),
        )
        c_arrow.set_color(RED)
        c_arrow.fix_in_frame()

        self.play(
            ShowCreation(c_arrow),
            Write(question)
        )
        self.wait(4)

        # "Obvious"
        obvious_words = TexText("Isn't this obvious?", font_size=36)
        obvious_words.set_color(GREY_A)
        obvious_words.match_y(question)
        obvious_words.to_edge(LEFT)
        obvious_arrow = Arrow(
            obvious_words, equation.get_corner(DL) + RIGHT
        )
        VGroup(obvious_words, obvious_arrow).fix_in_frame()

        self.play(
            TransformMatchingShapes(question, obvious_words),
            ReplacementTransform(c_arrow, obvious_arrow),
        )
        self.wait()

        # 2d quantities
        cube.clear_updaters()

        shadow_label = TexText("2D")
        shadow_label.move_to(shadow)
        face_labels = VGroup()
        for face in cube[3:]:
            lc = shadow_label.copy()
            normal = face.get_unit_normal()
            lc.rotate(angle_of_vector(flat_project(normal)) + PI / 2)
            lc.apply_matrix(
                rotation_between_vectors(OUT, normal)
            )
            lc.move_to(face)
            face.label = lc
            face_labels.add(lc)

        self.play(
            Write(shadow_label),
            Write(face_labels),
            run_time=2,
        )
        self.wait()
        scalars = [cube, face_labels, shadow_label]
        self.play(
            *(
                mob.animate.scale(0.5)
                for mob in scalars
            ),
            rate_func=there_and_back,
            run_time=3
        )
        self.play(
            *(
                mob.animate.stretch(2, 0)
                for mob in scalars
            ),
            rate_func=there_and_back,
            run_time=3
        )
        self.wait()

        # No not really
        no_words = Text("No, not really", font_size=30)
        no_words.set_color(RED)
        no_words.fix_in_frame()
        no_words.next_to(obvious_words, DOWN, MED_LARGE_BUFF)

        self.play(
            FadeIn(no_words, 0.25 * DOWN),
            FadeOut(shadow_label),
            FadeOut(face_labels),
        )

        # Move light
        self.play(
            light.animate.next_to(cube, OUT + RIGHT, 2),
            run_time=4,
            rate_func=rush_from,
        )
        for s in (1.5, 0.25, 2 / 0.75):
            self.play(
                cube.animate.scale(s),
                run_time=2,
            )

        # To finish
        cube.add_updater(lambda m: self.sort_to_camera(m))
        self.begin_ambient_rotation(cube)
        self.play(
            light.animate.next_to(cube, OUT, 50),
            LaggedStart(*map(FadeOut, (
                no_words, obvious_words, obvious_arrow,
            ))),
            run_time=3,
        )
        self.wait(33)


class LurkingAssumption(VideoWrapper):
    title = "There's a subtle hidden assumption..."
    wait_time = 4
    animate_boundary = False


class WhatIsC(Scene):
    def construct(self):
        words = TexText(
            "What is $c$?!",
            tex_to_color_map={"$c$": RED},
            font_size=72,
        )
        self.play(Write(words))
        self.play(FlashUnder(words, color=RED))
        self.wait()


class BobsFinalAnswer(Scene):
    def construct(self):
        answer = Tex(
            "S(\\text{Cube})", "=",
            "\\frac{1}{2}", "\\cdot", "{\\frac{1}{2}}",
            "(\\text{Surface area})", "=",
            "\\frac{1}{4} \\big(6s^2\\big)", "=",
            "\\frac{3}{2} s^2",
            tex_to_color_map={
                "\\text{Cube}": BLUE,
                "{\\frac{1}{2}}": RED,
            }
        )
        answer.add_to_back(get_overline(answer[:3]))
        equals = answer.get_parts_by_tex("=")
        eq_indices = list(map(answer.index_of_part, equals))

        eq1 = answer[:eq_indices[1]].deepcopy()
        eq2 = answer[:eq_indices[2]].deepcopy()
        eq3 = answer.deepcopy()
        for eq in eq1, eq2, eq3:
            eq.to_edge(RIGHT)
            eq.shift(1.25 * UP)

        self.play(FadeIn(eq1, DOWN))
        self.wait()
        for m1, m2 in (eq1, eq2), (eq2, eq3):
            self.play(
                FadeIn(m2[len(m1):]),
                m1.animate.move_to(m2, LEFT),
            )
            self.remove(m1)
            self.add(m2)
            self.wait()

        rect = SurroundingRectangle(eq3[-1])
        rect.set_stroke(YELLOW, 2)
        self.play(
            ShowCreation(rect),
            FlashAround(eq3[-1], stroke_width=5, time_width=1.5, run_time=1.5)
        )
        self.wait()


class ShowSeveralConvexShapes(Scene):
    def construct(self):
        frame = self.camera.frame
        frame.reorient(0, 70)

        dodec = Dodecahedron()
        dodec.set_fill(BLUE_D)

        spike = VGroup()
        hexagon = RegularPolygon(6)
        spike.add(hexagon)
        for v1, v2 in adjacent_pairs(hexagon.get_vertices()):
            spike.add(Polygon(v1, v2, 2 * OUT))
        spike.set_fill(BLUE_E)

        blob = Group(Sphere())
        blob.stretch(0.5, 0)
        blob.stretch(0.5, 1)
        blob.set_color(BLUE_E)
        blob.apply_function(
            lambda p: [*(2 - p[2]) * p[:2], p[2]]
        )
        blob.set_color(BLUE_E)

        cylinder = Group(Cylinder())
        cylinder.set_color(GREY_BROWN)
        cylinder.rotate(PI / 4, UP)

        examples = Group(*(
            Group(*mob)
            for mob in (dodec, spike, blob, cylinder)
        ))

        for ex in examples:
            ex.set_depth(2)
            ex.deactivate_depth_test()
            ex.set_gloss(0.5)
            ex.set_shadow(0.5)
            ex.set_reflectiveness(0.2)
            ex.rotate(20 * DEGREES, OUT)
            sort_to_camera(ex, self.camera.frame)
            for sm in ex:
                if isinstance(sm, VMobject):
                    sm.set_stroke(WHITE, 1)
                    sm.set_fill(opacity=0.9)
                else:
                    sm.always_sort_to_camera(self.camera)

        examples.arrange(RIGHT, buff=LARGE_BUFF)

        self.play(LaggedStart(*(
            LaggedStartMap(Write, ex, run_time=1)
            if isinstance(ex[0], VMobject)
            else ShowCreation(ex[0])
            for ex in examples
        ), lag_ratio=0.5, run_time=8))
        self.wait()


class KeyResult(Scene):
    def construct(self):
        eq1, eq2 = [
            get_key_result(word)
            for word in ("Cube", "Solid")
        ]
        VGroup(eq1, eq2).to_edge(UP)

        self.play(FadeIn(eq1, lag_ratio=0.1))
        self.wait()

        cube_underline = Underline(eq2.get_part_by_tex("Solid"), buff=0.05)
        cube_underline.set_stroke(BLUE, 1)
        general_words = Text("Assume convex", font_size=36)
        general_words.set_color(BLUE)
        general_words.next_to(cube_underline, DOWN, buff=1.0)
        general_words.shift(LEFT)
        general_arrow = Arrow(general_words, cube_underline.get_center(), buff=0.1)

        self.play(
            ShowCreation(cube_underline),
            FadeIn(general_words),
            ShowCreation(general_arrow),
            FadeTransformPieces(eq1, eq2),
        )
        self.wait()

        const_rect = SurroundingRectangle(VGroup(
            eq.get_part_by_tex("frac"),
            eq.get_part_by_tex("{c}")
        ), buff=SMALL_BUFF)
        const_rect.set_stroke(RED, 1)

        const_words = Text("Universal constant!", font_size=36)
        const_words.set_color(RED)
        const_words.match_y(general_words)
        const_words.set_x(const_rect.get_x() + 1)
        const_arrow = Arrow(const_words, const_rect, buff=0.1)

        self.play(
            ShowCreation(const_rect),
            FadeIn(const_words),
            ShowCreation(const_arrow),
        )
        self.wait()


class ShadowsOfDodecahedron(ShadowScene):
    inf_light = True

    def construct(self):
        # Setup
        self.camera.frame.set_height(7)
        self.camera.frame.shift(OUT)
        dodec = self.solid
        dodec.scale(5 / dodec[0].get_arc_length())
        outline = self.get_shadow_outline()
        area = DecimalNumber(font_size=36)
        area.move_to(outline)
        area.add_updater(lambda m: m.set_value(get_norm(outline.get_area_vector()) / (self.unit_size**2)))
        area.add_updater(lambda m: m.fix_in_frame())
        area.move_to(3.15 * DOWN)

        self.init_frame_rotation()

        ssf = 1.5
        self.wait()
        self.play(dodec.animate.space_out_submobjects(ssf))
        self.play(Rotate(dodec, PI, axis=RIGHT, run_time=6))
        self.play(dodec.animate.space_out_submobjects(1 / ssf))
        self.begin_ambient_rotation(dodec)

        self.play(
            VFadeIn(outline),
            VFadeIn(area),
        )

        # Add dot and line
        dot = GlowDot(0.5 * DR)
        line = DashedLine(10 * OUT, ORIGIN)
        line.set_stroke(YELLOW, 1)

        def dodec_sdf(point):
            return max(*(
                np.dot(point - pent.get_center(), pent.get_unit_normal())
                for pent in dodec
            ))

        def update_line(line):
            line.move_to(dot.get_center(), IN)
            for dash in line:
                dist = dodec_sdf(dash.get_center())
                dash.set_stroke(
                    opacity=interpolate(0.1, 1.0, clip(10 * dist, -0.5, 0.5) + 0.5)
                )
                dash.inside = (dist < 0)

        line.add_updater(update_line)

        self.play(ShowCreation(line, rate_func=rush_into))
        self.play(FadeIn(dot, rate_func=rush_from))

        # Just wait
        for n in range(8):
            self.play(dot.animate.move_to(midpoint(
                outline.get_center(),
                outline.pfp(random.random()),
            )), run_time=5)

    def get_solid(self):
        solid = self.get_solid_no_style()
        solid.set_stroke(WHITE, 1)
        solid.set_fill(self.solid_fill_color, 0.8)
        solid.set_gloss(0.1)
        solid.set_shadow(0.4)
        solid.set_reflectiveness(0.4)
        group = Group(*solid)
        group.deactivate_depth_test()
        group.add_updater(lambda m: self.sort_to_camera(m))
        return group

    def get_solid_no_style(self):
        dodec = Dodecahedron()
        dodec.scale((32 / get_surface_area(dodec))**0.5)
        return dodec


class AmbientDodecahedronShadow(ShadowsOfDodecahedron):
    solid_name = "Dodecahedron"
    solid_fill_color = BLUE_E
    name_color = BLUE_D

    def construct(self):
        self.camera.frame.reorient(20, 80)
        self.camera.frame.set_z(3)

        eq = get_key_result(self.solid_name, color=self.name_color)
        eq.to_edge(UP)
        eq.fix_in_frame()
        self.add(eq)

        self.init_frame_rotation()
        self.play(LaggedStart(*map(Write, self.solid)))
        self.add(self.solid)

        outline = self.get_shadow_outline()
        area_label = self.get_shadow_area_label()
        # area_label.scale(0.7)
        area_label.move_to(2.75 * DOWN).to_edge(LEFT)
        area_label.add_updater(lambda m: m.fix_in_frame())
        surface_area = get_surface_area(self.solid)
        surface_area /= (self.unit_size**2)
        sa_label = VGroup(Text("Surface area: "), DecimalNumber(surface_area))
        sa_label.arrange(RIGHT)
        sa_label.match_y(area_label)
        sa_label.to_edge(RIGHT)
        sa_label.set_backstroke()
        sa_label.fix_in_frame()

        self.play(
            *map(VFadeIn, (outline, area_label, sa_label))
        )
        self.add(outline)
        self.add(area_label)

        self.begin_ambient_rotation(self.solid, about_point=self.solid.get_center())
        self.wait(30)


class AmbientTriPrismSum(AmbientDodecahedronShadow):
    solid_name = "Triangular Prism"
    solid_fill_color = interpolate_color(TEAL_E, BLACK, 0.25)
    name_color = TEAL_D

    def get_solid_no_style(self):
        triangle = RegularPolygon(3)
        tri1, tri2 = triangle.replicate(2)
        tri2.shift(3 * OUT)
        sides = []
        verts1 = tri1.get_anchors()
        verts2 = tri2.get_anchors()
        for (a, b), (c, d) in zip(adjacent_pairs(verts1), adjacent_pairs(verts2)):
            sides.append(Polygon(a, b, d, c))
        result = VGroup(tri1, *sides, tri2)
        result.scale((16 / get_surface_area(result))**0.5)
        return result


class AmbientPyramidSum(AmbientDodecahedronShadow):
    solid_name = "Pyramid"
    solid_fill_color = GREY_BROWN
    name_color = interpolate_color(GREY_BROWN, WHITE, 0.5)

    def get_solid_no_style(self):
        base = Square(side_length=1)
        result = VGroup(base)
        for v1, v2 in adjacent_pairs(base.get_vertices()):
            result.add(Polygon(v1, v2, math.sqrt(3) * OUT / 2))
        result.set_height(2)
        return result


class AmbientCubeWithLabels(AmbientDodecahedronShadow):
    solid_name = "Cube"

    def get_solid_no_style(self):
        return VCube()


class DodecahedronFaceSum(Scene):
    def construct(self):
        expr = TexText(
            "Area(Shadow(Dodecahedron))", "=",
            "$\\displaystyle \\frac{1}{2}$",
            " $\\displaystyle \\sum_{j=1}^{12}$ ",
            "Area(Shadow(Face$_j$))",
            tex_to_color_map={
                "Shadow": GREY_B,
                "Dodecahedron": BLUE_D,
                "Face$_j$": YELLOW,
            }
        )
        expr.to_edge(UP, buff=MED_SMALL_BUFF)

        self.add(expr.slice_by_tex(None, "="))
        self.wait()
        self.play(FadeIn(expr.slice_by_tex("="), shift=0.25 * UP))
        self.wait()
        self.play(FlashAround(expr.get_part_by_tex("frac"), run_time=2))
        self.play(FlashUnder(expr[-5:], run_time=2))
        self.wait(2)


class SphereShadow(ShadowScene):
    inf_light = True

    def construct(self):
        frame = self.camera.frame
        frame.set_height(7)
        frame.shift(OUT)
        sphere = self.solid
        shadow = self.shadow
        # shadow[1].always_sort_to_camera(self.camera)
        shadow_circle = Circle()
        shadow_circle.set_fill(BLACK, 0.8)
        shadow_circle.replace(shadow)
        shadow_circle.set_stroke(WHITE, 1)
        self.add(shadow_circle)

        self.begin_ambient_rotation(
            sphere, speed=0.3,
            initial_axis=[-1, -1, 0.5]
        )
        self.wait(60)

    def get_solid(self):
        ep = 1e-3
        sphere = Sphere(
            radius=1.5,
            u_range=(ep, TAU - ep),
            v_range=(ep, PI - ep),
        )
        sphere = TexturedSurface(sphere, "EarthTextureMap", "NightEarthTextureMap")
        sphere.set_opacity(1)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere)
        mesh.set_stroke(WHITE, 0.5, 0.25)
        return Group(sphere, mesh)


class SphereInfo(Scene):
    def construct(self):
        kw = {
            "tex_to_color_map": {
                "{c}": RED,
                "=": WHITE,
                "R": BLUE
            },
            "font_size": 36
        }
        shadow = Tex("\\text{Average shadow area} = \\pi R^2", **kw)
        surface = Tex("\\text{Surface area} = 4 \\pi R^2", **kw)
        conclusion = Tex("\\frac{1}{2}", "{c}", "=", "\\frac{1}{4}", **kw)

        eqs = VGroup(shadow, surface, conclusion)
        eqs.arrange(DOWN, buff=MED_LARGE_BUFF)
        for eq in eqs:
            eq.shift(eq.get_part_by_tex("=").get_x() * LEFT)
        eqs.to_corner(UR)

        for eq in eqs[:2]:
            eq[0].set_color(GREY_A)

        for eq in eqs:
            self.play(FadeIn(eq, lag_ratio=0.1))
            self.wait()

        self.play(eqs[2].animate.scale(2, about_edge=UP))
        rect = SurroundingRectangle(eqs[2])
        rect.set_stroke(YELLOW, 2)
        self.play(ShowCreation(rect))
        self.wait()


class PiRSquared(Scene):
    def construct(self):
        form = Tex("\\pi R^2")[0]
        form[1].set_color(BLUE)
        self.play(Write(form))
        self.wait()


class SwapConstantForFourth(Scene):
    def construct(self):
        eq = get_key_result("Dodecahedron")
        eq.to_edge(UP)
        parts = VGroup(
            eq.get_part_by_tex("frac"),
            eq.get_part_by_tex("{c}")
        )
        fourth = Tex("\\frac{1}{4}")
        fourth.move_to(parts, LEFT)
        fourth.set_color(RED)

        self.add(eq)
        self.wait()
        self.play(
            FadeOut(parts, UP),
            FadeIn(fourth, UP),
            eq.slice_by_tex("\\cdot").animate.next_to(fourth, RIGHT),
        )
        self.wait()


class ButSpheresAreSmooth(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            TexText("But spheres don't\\\\have flat faces!"),
            target_mode="angry",
            index=2,
            added_anims=[self.teacher.change("guilty")]
        )
        self.play_student_changes(
            "erm", "hesitant", "angry",
            look_at=self.screen,
        )
        self.wait(6)


class RepeatedRelation(Scene):
    def construct(self):
        # Relations
        relation = VGroup(
            Text("Average shadow"),
            Tex("=").rotate(PI / 2),
            Tex("\\frac{1}{2}", "c", "\\cdot (", "\\text{Surface area}", ")")
        )
        relation[0].set_color(GREY_A)
        relation[2][1].set_color(RED)
        relation[2][3].set_color(BLUE)
        relation.arrange(DOWN)
        relation.scale(0.6)
        repeats = relation.get_grid(1, 4, buff=0.8)
        repeats.to_edge(LEFT, buff=MED_LARGE_BUFF)
        repeats.shift(0.5 * DOWN)

        for repeat in repeats:
            self.play(FadeIn(repeat[0], lag_ratio=0.1))
            self.play(
                Write(repeat[1]),
                FadeIn(repeat[2], 0.5 * DOWN)
            )
        self.wait()

        # Limit
        limit = Tex(
            "\\lim_{|F| \\to 0}",
            "\\left(", "{\\text{Average shadow}", "\\over ", "\\text{Surface area}}", "\\right)",
            "=", "\\frac{1}{2}", "{c}",
        )
        limit.set_color_by_tex("Average shadow", GREY_A)
        limit.set_color_by_tex("Surface area", BLUE)
        limit.set_color_by_tex("{c}", RED)
        limit.move_to(2.5 * DOWN)
        limit.match_x(repeats)

        new_rhs = Tex("=", "{\\pi R^2", "\\over", "4\\pi R^2}")
        new_rhs.set_color_by_tex("\\pi R^2", GREY_A)
        new_rhs.set_color_by_tex("4\\pi R^2", BLUE)
        new_rhs.move_to(limit.get_part_by_tex("="), LEFT)

        self.play(Write(limit))
        self.wait()
        self.play(
            limit.slice_by_tex("=").animate.next_to(new_rhs, RIGHT),
            GrowFromCenter(new_rhs)
        )
        self.wait()


class SimpleCross(Scene):
    def construct(self):
        lines = VGroup(
            Line(UP, DOWN).set_height(FRAME_HEIGHT),
            Line(LEFT, RIGHT).set_width(FRAME_WIDTH),
        )
        self.play(ShowCreation(lines, lag_ratio=0.5))
        self.wait()


# Not needed?
class AmbientCubeTurningIntoNewShapes(Scene):
    def construct(self):
        pass


class PopularizaitonVsDoing(Scene):
    def construct(self):
        # Words
        popular = Text("Popularization of math")
        doing = Text("Doing math")
        words = VGroup(popular, doing)
        words.arrange(DOWN, buff=3)
        words.move_to(UP)

        self.play(FadeIn(popular, UP))
        self.wait()
        self.play(
            TransformFromCopy(
                popular.get_part_by_text("math"),
                doing.get_part_by_text("math"),
            ),
            Write(doing[:len("Doing")])
        )
        self.wait()

        # Bars
        width = 8
        bar = Rectangle(width, 0.5)
        bar.set_stroke(WHITE, 1)
        bar.next_to(popular, DOWN)
        left_bar, right_bar = bar.replicate(2)
        left_bar.set_fill(BLUE_E, 1)
        right_bar.set_fill(RED_E, 1)
        left_bar.stretch(0.5, 0, about_edge=LEFT)
        right_bar.stretch(0.5, 0, about_edge=RIGHT)

        left_brace = always_redraw(lambda: Brace(left_bar, DOWN, buff=SMALL_BUFF))
        right_brace = always_redraw(lambda: Brace(right_bar, DOWN, buff=SMALL_BUFF))
        left_label = Text("Insights", font_size=30, color=GREY_B)
        right_label = Text("Computations", font_size=30, color=GREY_B)
        always(left_label.next_to, left_brace, DOWN, SMALL_BUFF)
        always(right_label.next_to, right_brace, DOWN, SMALL_BUFF)

        bar_group = VGroup(
            bar,
            left_bar, right_bar,
            left_brace, right_brace,
            left_label, right_label,
        )

        def set_bar_alpha(alpha, **kwargs):
            self.play(
                left_bar.animate.set_width(alpha * width, about_edge=LEFT, stretch=True),
                right_bar.animate.set_width((1 - alpha) * width, about_edge=RIGHT, stretch=True),
                **kwargs
            )

        self.play(FadeIn(bar_group, lag_ratio=0.1))
        set_bar_alpha(0.95, run_time=2)
        self.wait()
        self.add(bar_group.deepcopy().clear_updaters())
        self.play(
            bar_group.animate.shift(doing.get_center() - popular.get_center())
        )
        set_bar_alpha(0.05, run_time=5)
        self.wait()

        # Embed
        self.embed()


class MultipleMathematicalBackgrounds(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        labels = VGroup(
            TexText("$\\le$ High school"),
            TexText("$\\approx$ Undergrad"),
            TexText("$\\ge$ Ph.D."),
        )
        for student, label in zip(self.students, labels):
            label.scale(0.7)
            label.next_to(student, UP)

        words = TexText("Explanation doesn't vary\\\\with backgrounds")
        words.to_edge(UP)

        lines = VGroup(*(
            DashedLine(words, label, buff=0.5)
            for label in labels
        ))
        lines.set_stroke(WHITE, 2)

        self.add(words)
        self.play(
            self.teacher.change("raise_right_hand"),
            self.change_students(
                "pondering", "thinking", "pondering",
                look_at=self.teacher.eyes,
            ),
        )
        self.play(
            LaggedStartMap(ShowCreation, lines, lag_ratio=0.5),
            LaggedStartMap(FadeIn, labels, lag_ratio=0.5),
        )
        self.wait(3)
        self.play(
            self.teacher.change("dejected").look(UP),
            self.change_students("hesitant", "well", "thinking"),
            LaggedStartMap(FadeOut, lines, scale=0.5),
            FadeOut(words, DOWN),
        )
        self.wait(4)

        # Different levels
        kw = {"font_size": 30}
        methods = VGroup(
            TexText("Calculus\\\\primer", **kw),
            TexText("Quickly show\\\\key steps", **kw),
            TexText("Describe as a\\\\measure on SO(3)", **kw),
        )
        new_lines = VGroup()
        colors = [GREEN_B, GREEN_C, GREEN_D]
        for method, label, color in zip(methods, labels, colors):
            method.move_to(label)
            method.shift(2.5 * UP)
            method.set_color(color)
            line = DashedLine(method, label, buff=0.25)
            line.set_stroke(color, 2)
            new_lines.add(line)

        self.play(
            self.teacher.change("raise_right_hand"),
            self.change_students(
                "erm", "pondering", "thinking",
                look_at=self.students.get_center() + 4 * UP
            ),
            LaggedStartMap(FadeIn, methods, lag_ratio=0.5),
            LaggedStartMap(ShowCreation, new_lines, lag_ratio=0.5),
        )
        self.wait(4)


class WatchingAVideo(Scene):
    def construct(self):
        self.add(FullScreenRectangle())
        randy = Randolph()
        randy.to_corner(DL)
        screen = ScreenRectangle(height=5)
        screen.set_fill(BLACK, 1)
        screen.to_corner(UR)

        def blink_wait(n=1):
            for x in range(n):
                self.wait()
                self.play(Blink(randy))
                self.wait()

        self.add(screen)
        self.add(randy)
        self.play(randy.change("pondering", screen))
        blink_wait()
        self.play(randy.change("thinking", screen))
        blink_wait()
        self.play(randy.change("hesitant", screen))
        blink_wait(2)


class CleverProofExample(Scene):
    def construct(self):
        initial_sum = Tex("1^2 + 2^2 + 3^2 + \\cdots + n^2")
        tripple_tris, final_tri = self.get_triangle_sums()

        initial_sum.set_width(10)
        self.play(FadeIn(initial_sum, lag_ratio=0.1))
        self.wait()

        tripple_tris.set_width(10)
        tripple_tris.to_edge(DOWN, buff=2)
        tris = tripple_tris[0]
        tri = tris[0]
        tri.save_state()
        tri.set_height(4)
        tri.center().to_edge(DOWN, buff=1)
        self.play(
            initial_sum.animate.set_width(8).to_edge(UP),
            FadeIn(tri, lag_ratio=0.1)
        )
        self.wait()
        self.play(
            Restore(tri),
            FadeIn(tripple_tris[1:])
        )
        for i in (0, 1):
            bt1 = tris[i].copy()
            bt1.generate_target()
            bt1.target.rotate(120 * DEGREES)
            bt1.target.replace(tris[i + 1])
            bt1.target.set_opacity(0)
            tris[i + 1].save_state()
            tris[i + 1].rotate(-120 * DEGREES)
            tris[i + 1].replace(tris[i])
            tris[i + 1].set_opacity(0)
            self.play(
                MoveToTarget(bt1, remover=True),
                Restore(tris[i + 1])
            )
        self.wait()

        final_tri.set_height(3)
        final_tri.move_to(tripple_tris, UP)
        initial_sum.generate_target()
        eq = Tex("=").scale(2)
        tripple_tris.generate_target()
        top_row = VGroup(initial_sum.target, eq, tripple_tris.target)
        top_row.arrange(RIGHT, buff=0.5)
        top_row.set_width(FRAME_WIDTH - 1)
        top_row.center().to_edge(UP)

        self.play(
            MoveToTarget(initial_sum),
            FadeIn(eq),
            MoveToTarget(tripple_tris),
            FadeTransform(tripple_tris.copy(), final_tri)
        )
        self.wait()

        final1 = Tex("= \\frac{2n + 1}{3} (1 + 2 + 3 + \\cdots + n)")
        final2 = Tex("= \\frac{2n + 1}{3} \\frac{(n + 1)n}{2}")
        final3 = Tex("= \\frac{(2n + 1)(n + 1)(n)}{6}")
        final_tri.generate_target()
        final_tri.target.set_height(2).to_edge(LEFT)
        for final in (final1, final2, final3):
            final.next_to(final_tri.target, RIGHT)

        self.play(
            MoveToTarget(final_tri),
            Write(final1),
        )
        self.wait()
        self.play(
            FadeOut(final1, UP),
            FadeIn(final2, UP),
        )
        self.wait()
        self.play(
            FadeOut(final2, UP),
            FadeIn(final3, UP),
        )
        self.play(VGroup(final_tri, final3).animate.set_x(0))
        self.wait()

        # Embed
        self.embed()

    def get_triangle_sums(self):
        dl_dots = Tex("\\vdots").rotate(-30 * DEGREES)
        dr_dots = Tex("\\vdots").rotate(30 * DEGREES)
        blank = Integer(0).set_opacity(0)
        n = Tex("n")
        np1 = Tex("(2n + 1)")
        dots = Tex("\\dots")
        tri1 = VGroup(
            Integer(1).replicate(1),
            Integer(2).replicate(2),
            Integer(3).replicate(3),
            VGroup(dl_dots, blank.copy(), blank.copy(), dr_dots),
            VGroup(n.copy(), n.copy(), dots, n.copy(), n.copy()),
        )
        tri2 = VGroup(
            n.replicate(1),
            VGroup(dl_dots, n).copy(),
            VGroup(Integer(3), blank, dr_dots).copy(),
            VGroup(Integer(2), Integer(3), dots, n).copy(),
            VGroup(Integer(1), Integer(2), Integer(3), dots, n).copy(),
        )
        tri3 = VGroup(
            n.replicate(1),
            VGroup(n, dr_dots).copy(),
            VGroup(dl_dots, blank, Integer(3)).copy(),
            VGroup(n, dots, Integer(3), Integer(2)).copy(),
            VGroup(n, dots, Integer(3), Integer(2), Integer(1)).copy(),
        )

        sum_tri = VGroup(
            np1.replicate(1),
            np1.replicate(2),
            np1.replicate(3),
            VGroup(dl_dots, *blank.replicate(6), dr_dots).copy(),
            VGroup(np1.copy(), np1.copy(), dots.copy(), np1.copy(), np1.copy()),
        )

        tris = VGroup(tri1, tri2, tri3)
        for tri in (*tris, sum_tri):
            for row in tri:
                row.arrange(RIGHT, buff=0.5)
            tri.arrange(DOWN, buff=0.5)
        tris.arrange(RIGHT, buff=2.0)
        tris.set_width(6)
        plusses = VGroup(
            Tex("+").move_to(tris[:2]),
            Tex("+").move_to(tris[1:]),
        )
        parens = Tex("()")[0]
        parens.stretch(2, 1)
        parens.match_height(tris)
        parens[0].next_to(tris, LEFT)
        parens[1].next_to(tris, RIGHT)

        frac = Tex("\\frac{1}{3}")
        frac.next_to(parens, LEFT)

        lhs = VGroup(tris, frac, parens, plusses)

        parens_copy = parens.copy()
        sum_tri.match_height(lhs)
        sum_tri.move_to(tris, LEFT)
        parens_copy[1].next_to(sum_tri, RIGHT)

        rhs = VGroup(frac.copy(), sum_tri, parens_copy)
        rhs.next_to(lhs, DOWN, LARGE_BUFF, aligned_edge=LEFT)
        # eq = Tex("=")
        # eq.next_to(rhs, LEFT)

        return VGroup(lhs, rhs)


class BlendOfMindsets(Scene):
    def construct(self):
        Text("Calculate specifics")
        Text("Understand generalities")
        Text("You need both")


class ListernerEmail(Scene):
    def construct(self):
        # Letter
        rect = Rectangle(4, 7)
        rect.set_stroke(WHITE, 2)
        rect.set_fill("#060606", 1)
        lines = Line(LEFT, RIGHT).get_grid(15, 1)
        lines.set_width(0.8 * rect.get_width())
        lines.arrange(DOWN)
        lines.set_height(0.7 * rect.get_height(), stretch=True)
        for n in [3, 8, -1]:
            lines[n].stretch(0.5, 0, about_edge=LEFT)
            if n > 0:
                lines[n + 1].set_opacity(0)
        lines.move_to(rect)

        salutation = Text("Hi Prof. Kontorovich,", font_size=30)
        salutation.next_to(lines, UP, aligned_edge=LEFT)
        lines.shift(0.2 * DOWN)

        letter = VGroup(rect, lines, salutation)

        self.add(rect)
        self.play(
            Write(salutation, run_time=1),
            ShowCreation(lines, rate_func=linear, run_time=3, lag_ratio=0.5),
        )
        self.add(letter)
        self.wait()
        self.play(letter.animate.to_edge(LEFT))

        # Phrases
        phrases = VGroup(
            Text("I’m a PhD student..."),
            Text(
                "...I had noticed my mathematical capabilities\n"
                "starting to fade (to which I attributed getting\n"
                "older and not being as sharp)..."
            ),
            Text(
                "...I realized that the entire problem, for me at least,\n"
                "was entirely about my lack of problems and drills."
            ),
        )

        phrases.arrange(DOWN, buff=2.0, aligned_edge=LEFT)
        phrases.set_width(8)
        phrases.next_to(letter, RIGHT, LARGE_BUFF)

        highlights = VGroup()
        for i, w in [(0, 1), (5, 3), (11, 2.5)]:
            hrect = Rectangle(w, 0.1)
            hrect.set_stroke(width=0)
            hrect.set_fill(YELLOW, 0.5)
            hrect.move_to(lines[i], LEFT)
            highlights.add(hrect)

        highlights[0].shift(1.5 * RIGHT)
        highlights[2].align_to(lines, RIGHT)

        hlines = VGroup()

        for highlight, phrase in zip(highlights, phrases):
            hlines.add(VGroup(
                DashedLine(highlight.get_corner(UR), phrase.get_corner(UL), buff=0.1),
                DashedLine(highlight.get_corner(DR), phrase.get_corner(DL), buff=0.1),
            ))
        hlines.set_stroke(YELLOW, 1)

        for i in range(3):
            self.play(
                FadeIn(highlights[i]),
                *map(ShowCreation, hlines[i]),
                GrowFromPoint(phrases[i], highlights[i].get_right())
            )
            self.wait(2)

        # Embed
        self.embed()


class FamousMathematicians(Scene):
    im_height = 3.5

    def construct(self):
        # Portraits
        images = Group(
            ImageMobject("Newton"),
            ImageMobject("Euler"),
            ImageMobject("Gauss"),
            ImageMobject("Fourier"),
            ImageMobject("Riemann_cropped"),
            ImageMobject("Cauchy"),
            ImageMobject("Noether"),
            ImageMobject("Ramanujan"),
        )
        names = VGroup(
            Text("Isaac Newton"),
            Text("Leonhard Euler"),
            Text("Carl Friedrich Gauss"),
            Text("Joseph Fourier"),
            Text("Bernhard Riemann"),
            Text("Augustin Cauchy"),
            Text("Emmy Noether"),
            Text("Srinivasa Ramanujan"),
        )
        im_groups = Group()
        for im, name in zip(images, names):
            im.set_height(self.im_height)
            name.scale(0.6)
            name.set_color(GREY_A)
            name.next_to(im, DOWN)
            im_groups.add(Group(im, name))

        # im_groups.arrange(RIGHT, aligned_edge=UP, buff=LARGE_BUFF)
        im_groups.arrange_in_grid(2, 4, aligned_edge=UP, buff=LARGE_BUFF)
        im_groups.set_width(FRAME_WIDTH - 2)
        im_groups.to_edge(LEFT)
        dots = Tex("\\dots", font_size=72).replicate(2)
        dots[0].next_to(images[-5], RIGHT, MED_LARGE_BUFF)
        dots[1].next_to(images[-1], RIGHT, MED_LARGE_BUFF)

        self.play(
            LaggedStart(*map(FadeIn, (*im_groups, dots)), lag_ratio=0.25),
            run_time=5
        )
        self.wait()

        self.play(
            im_groups[0].animate.set_height(6).center().to_edge(LEFT),
            LaggedStart(*(
                FadeOut(mob, DR)
                for mob in (*im_groups[1:], dots)
            ), lag_ratio=0.25),
            run_time=2,
        )
        self.wait()

        # Papers (do in editor)


class InventingMath(Scene):
    def construct(self):
        pass


class AmbientHourglass(ShadowScene):
    inf_light = True

    def construct(self):
        frame = self.camera.frame
        frame.set_z(3)

        self.init_frame_rotation()
        self.remove(self.solid, self.shadow)

        qint_func = bezier([0, 1, -1.25, 1, 0])

        def func(u, v):
            qf = qint_func(v)
            x = qf * math.cos(u)
            y = qf * math.sin(u)
            x = np.sign(x) * abs(x)**0.5
            y = np.sign(y) * abs(y)**0.5
            return [x, y, 0.5 - v]

        ep = 1e-6
        hourglass = ParametricSurface(func, (0, TAU), (0 + ep, 1 - ep))

        hourglass.set_depth(2)
        hourglass.set_z(3)
        hourglass.set_color(BLUE_D)
        hourglass.set_opacity(0.5)
        hourglass.set_reflectiveness(0.1)
        hourglass.set_gloss(0.1)
        hourglass.set_shadow(0.5)
        hourglass.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(hourglass)
        mesh.set_flat_stroke(False)
        mesh.set_stroke(BLUE_B, 0.2, 0.5)
        mesh_shadow = mesh.copy()
        mesh_shadow.deactivate_depth_test()
        solid_group = Group(mesh_shadow, hourglass, mesh)

        shadow = self.shadow = get_shadow(solid_group)
        shadow[1].always_sort_to_camera(self.camera)

        self.add(solid_group, shadow)

        for x in range(30):
            self.random_toss(solid_group)
            self.wait()

        self.begin_ambient_rotation(
            solid_group,
            speed=0.5,
            initial_axis=[1, 0, 1],
        )
        self.wait(35)


class QuantifyConvexity(Scene):
    def construct(self):
        # Ask question
        nonconvex = Text("Non-convex")
        nonconvex.to_edge(UP)
        nonconvex.set_color(RED)
        question = Text("Can we quantify this?")
        question.next_to(nonconvex, DOWN, buff=1.5)
        question.to_edge(LEFT)
        arrow = Arrow(question, nonconvex.get_corner(DL))

        self.play(Write(nonconvex))
        self.wait()
        self.play(
            FadeIn(question, 0.5 * DOWN),
            ShowCreation(arrow),
        )
        self.wait()

        # Binary choice
        double_arrow = Tex("\\leftrightarrow")
        double_arrow.move_to(nonconvex)
        convex = Text("Convex")
        convex.set_color(GREEN)
        convex.next_to(double_arrow, RIGHT)

        self.play(
            nonconvex.animate.next_to(double_arrow, LEFT),
            Write(double_arrow),
            FadeIn(convex, shift=0.25 * RIGHT),
            Uncreate(arrow),
            FadeOut(question, 0.5 * DOWN),
        )
        self.wait()

        # Spectrum
        interval = UnitInterval(width=7)
        interval.add_numbers()
        interval.to_corner(UL, buff=LARGE_BUFF)

        self.play(
            FadeTransform(double_arrow, interval),
            convex.animate.scale(0.5).next_to(interval.n2p(1), UP),
            nonconvex.animate.scale(0.5).next_to(interval.n2p(0), UP),
        )
        self.wait()

        # Fraction
        shadow = get_key_result("Solid").slice_by_tex(None, "=")
        shadow.add(shadow[0].copy())
        shadow.remove(shadow[0])
        four_shadow = VGroup(Tex("4 \\cdot"), shadow)
        four_shadow.arrange(RIGHT, buff=SMALL_BUFF)
        sa = Text("Surface area")
        frac = VGroup(
            four_shadow,
            Line().match_width(four_shadow).set_stroke(width=2),
            sa
        )
        frac.arrange(DOWN)
        frac.set_width(3)
        frac.to_corner(UR)
        frac.match_y(interval)

        self.play(Write(shadow))
        self.play(FadeIn(four_shadow[0]))
        self.wait()
        self.play(ShowCreation(frac[1]))
        self.play(FadeIn(sa))
        self.wait()

        # Dot
        dot = GlowDot()
        dot.scale(2)
        dot.move_to(interval.n2p(1))

        self.play(FadeIn(dot, RIGHT))
        self.wait()
        self.play(dot.animate.move_to(interval.n2p(0.6)), run_time=2)
        self.wait()


class GoalsOfMath(TeacherStudentsScene):
    def construct(self):
        words = Text("The goal of math\nis to answer questions")
        words.move_to(self.hold_up_spot, DOWN)
        words.to_edge(RIGHT, buff=2.0)
        aq = words.get_part_by_text("answer questions")
        aq.set_color(BLUE)
        dni = Text(
            "develop new ideas",
            t2c={"new ideas": YELLOW},
            t2s={"new ideas": ITALIC},
        )
        dni.move_to(aq, LEFT)

        self.play(
            self.teacher.change("raise_right_hand", words),
            self.change_students(*3 * ["pondering"], look_at=words),
            Write(words)
        )
        self.wait(2)
        self.add(aq, self.teacher)
        self.play(
            aq.animate.shift(0.5 * DOWN).set_opacity(0.2),
            Write(dni),
            self.teacher.change("well", words),
            self.change_students(*3 * ["thinking"], look_at=words)
        )
        self.wait(3)


class InfatuationWithGenerality(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            TexText("Why are mathematicians\\\\obsessed with abstractions?"),
            index=0,
            added_anims=[
                self.students[1].change("tease"),
                self.students[2].change("pondering"),
            ]
        )
        self.play(
            self.teacher.change("well"),
        )
        self.wait(6)


class NumberphileFrame(VideoWrapper):
    animate_boundary = True
    title = "Bertrand's Paradox (with Numberphile)"
    title_config = {
        "font_size": 48
    }
    wait_time = 16


class ByLine(Scene):
    def construct(self):
        lines = VGroup(
            TexText("Artwork by\\\\", "Kurt Bruns"),
            TexText("Music by\\\\", "Vince Rubinetti"),
            TexText("Other stuff\\\\", "Grant Sanderson"),
        )
        for line in lines:
            line[0].set_color(GREY_B)
            line[1].scale(1.2, about_edge=UP)

        lines.arrange(DOWN, buff=1.5)
        self.add(lines)


class EndScreen(PatreonEndScreen):
    pass


class ThumbnailBackground(ShadowScene):
    plane_dims = (32, 20)

    def construct(self):
        frame = self.camera.frame
        frame.reorient(0)
        cube = self.solid
        cube.set_shadow(0.5)
        light = self.light

        light.next_to(cube, OUT, buff=2)
        light.shift(2 * LEFT)

        light.move_to(50 * OUT)

        gc = self.glow.replicate(10)
        gc.set_opacity(0.3)
        gc.clear_updaters()
        gc.arrange(RIGHT).match_width(cube)
        gc.move_to(6 * OUT)
        self.add(gc)

        outline = self.get_shadow_outline()
        light_lines = self.get_light_lines(outline)
        self.add(outline, light_lines)
        self.randomly_reorient(cube)
        self.randomly_reorient(cube)
        self.wait()
