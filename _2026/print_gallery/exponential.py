from manim_imports_ext import *


# Helper functions for image manipulation

def get_texture_folder():
    return Path(get_directories()['base'], "videos", "2026", "print_gallery", "textures")


def get_print_gallery_log_image_path():
    return Path(get_texture_folder(), "PrintGalleryLog.png")


def get_pi_house_log_image_path(small=False):
    file_name = "PiHouseLogSmall.png" if small else "PiHouseLog.png"
    return Path(get_texture_folder(), file_name)


def get_log_image(log_image_path, scale_factor, resolution=(51, 101)):
    log_image = TexturedSurface(
        Square3D(resolution=resolution),
        log_image_path,
    )
    log_image.set_shading(0, 0, 0)
    log_image.deactivate_depth_test()
    log_image.set_shape(math.log(scale_factor), TAU)
    return log_image


def get_droste_from_log_image_path(log_image_path, scale_factor=256, n_iterations=5, height=7.5):
    log_image = get_log_image(log_image_path, scale_factor)
    log_images = log_image.get_grid(1, n_iterations, buff=0)
    log_images.move_to(ORIGIN, DR)
    log_images.apply_complex_function(np.exp)
    log_images.set_height(height)
    return log_images


def get_rectified_print_gallery(**kwargs):
    return get_droste_from_log_image_path(get_print_gallery_log_image_path(), **kwargs)


def get_rectified_pi_house(scale_factor=16, **kwargs):
    return get_droste_from_log_image_path(get_pi_house_log_image_path(), scale_factor=scale_factor, **kwargs)


def get_image_slice(image: TexturedSurface, delta_u: float, u_min: float):
    """
    Takes in a TexturedSurface for a rectangular image, returns a vertical slice.
    Think of 'u' as the coordinate for the horizontal direction.
    delta_u gives the width of the slice, u_min gives its position
    """
    nu, nv = image.resolution
    image_slice = TexturedSurface(
        Square3D(resolution=(int(delta_u * nu) + 1, nv)),
        image.texture_paths["LightTexture"]
    )
    image_slice.set_shading(0, 0, 0)
    image_slice.deactivate_depth_test()
    image_slice.set_image_coords_by_uv_func(
        lambda u, v: (u_min + delta_u * u, v)
    )
    width, height, depth = image.get_shape()
    image_slice.set_shape(delta_u * width, height)
    image_slice.move_to(image, DL).shift(u_min * width * RIGHT)
    return image_slice


def upper_cut_log(z):
    raw_log = np.log(z)
    return complex(raw_log.real, raw_log.imag % TAU)


def apply_func_between_planes(mob, func, src_plane, trg_plane):
    mob.shift(-src_plane.get_origin())
    mob.scale(1.0 / src_plane.get_unit_size(), about_point=ORIGIN)
    mob.apply_complex_function(func)
    mob.scale(trg_plane.get_unit_size(), about_point=ORIGIN)
    mob.shift(trg_plane.get_origin())
    return mob


def get_nested_square_grid(n_rows=8, n_recursions=6, height=4, stroke_color=WHITE, stroke_width=2, stroke_width_decay_factor=0.85, scale_factor=2):
    grid = Square().get_grid(n_rows, n_rows, buff=0)
    grid.center()
    grid.set_height(height)
    grid.set_stroke(stroke_color, stroke_width)
    grid.remove(*[
        square for square in grid
        if np.all(np.abs(square.get_center()) < (height / 2) / scale_factor)
    ])
    result = VGroup(grid)
    for n in range(n_recursions):
        result.add(result[-1].copy().scale((1 / scale_factor), about_point=ORIGIN))
        result[-1].set_stroke(width=stroke_width * stroke_width_decay_factor**(n + 1))

    return result

# Scenes


class PrintGalleryZoomStages(InteractiveScene):
    def construct(self):
        # Add Droste image and zoom in two iterations
        droste_image = get_rectified_print_gallery()
        droste_image.scale(256).center()
        self.add(droste_image)

        exp_tracker = ValueTracker(0)
        frame = self.frame
        frame.add_updater(lambda m: m.set_height(FRAME_HEIGHT * np.exp(exp_tracker.get_value())))

        dec = DecimalNumber()
        dec.fix_in_frame()
        dec.to_corner(DL)
        dec.f_always.set_value(exp_tracker.get_value)

        exp_tracker.set_value(0.5)
        self.wait()
        for value in [-0.5, -1.87, -3.86, -5.0]:
            self.play(exp_tracker.animate.set_value(value), run_time=2)
            self.wait()

        # Full zoom
        exp_tracker.set_value(-5 + math.log(256))
        self.play(
            exp_tracker.animate.set_value(-5),
            run_time=10
        )


class ZoomWithColors(InteractiveScene):
    def construct(self):
        # Add Droste image and zoom in two iterations
        droste_image = get_rectified_print_gallery()
        droste_image.scale(256).center()
        droste_image.set_opacity(0.25)
        self.disable_interaction(droste_image)
        self.add(droste_image)

        exp_tracker = ValueTracker(0.5)
        frame = self.frame
        frame.add_updater(lambda m: m.set_height(FRAME_HEIGHT * np.exp(exp_tracker.get_value())))

        dec = DecimalNumber()
        dec.fix_in_frame()
        dec.to_corner(DL)
        dec.f_always.set_value(exp_tracker.get_value)

        # Add outlines
        root = Path(
            self.file_writer.get_output_file_rootname().parent.parent,
            "Paul Segments/SVGs for Print Gallery Outlines/Straight"
        )
        person_outline, person_mask, frame, building, arcade = images = Group(
            ImageMobject(str(Path(root, name + ".png")))
            for name in ["PersonOutline", "PersonMask", "Frame", "Building", "Arcade"]
        )
        person = Group(
            person_mask.set_opacity(0.2),
            person_outline,
        )
        person.set_height(8.8).move_to(np.array([-2.41928692, -5.89659999, 0.]))
        frame.set_height(7.599240).move_to(np.array([-0.65032366, -0.76692351, 0.]))
        building.set_height(1.2807975113391876).move_to(np.array([-0.75733024, 0.36813092, 0.]))
        arcade.set_height(0.24800574034452438).move_to(np.array([0.0865169, 0.01073422, 0.]))
        small_person = person.copy().scale(1 / 256, about_point=ORIGIN)

        self.add(frame)
        self.add(person)
        self.add(building)
        self.add(arcade)
        self.add(small_person)

        # Test
        exp_tracker.set_value(1)
        self.play(
            exp_tracker.animate.set_value(1 - 1 * math.log(256)).set_anim_args(rate_func=linear),
            FadeOut(person, time_span=(0, 2)),
            FadeIn(frame, time_span=(0, 5), rate_func=there_and_back_with_pause),
            FadeIn(building, time_span=(3, 10), rate_func=there_and_back_with_pause),
            FadeIn(arcade, time_span=(7, 12), rate_func=there_and_back_with_pause),
            FadeIn(small_person, time_span=(9, 12)),
            run_time=12
        )


class NoteThe256(InteractiveScene):
    def construct(self):
        # Set up Droste
        frame = self.frame
        droste_image = get_rectified_print_gallery()
        droste_image.scale(256).center()

        drost_image_ghost = droste_image.copy().set_opacity(0.3)

        frame_exp_tracker = ValueTracker(0)
        box_exp_tracker = ValueTracker(0)
        frame.add_updater(lambda m: m.set_height(FRAME_HEIGHT * np.exp(-frame_exp_tracker.get_value())))

        box = ScreenRectangle()
        box.set_height(FRAME_HEIGHT)
        box.set_stroke(RED, 4)
        droste_image.always.clip_to_box(box)

        self.add(drost_image_ghost, droste_image)
        self.wait()

        # Set up box label
        box_label = Tex(R"\times 256.0")
        dec = box_label.make_number_changeable("256.0", font_size=72)

        def get_scale_factor():
            return np.exp(box_exp_tracker.get_value())

        def update_box_label(box_label):
            box_label[1].set_value(np.exp(box_exp_tracker.get_value()))
            box_label.set_width(box.get_width() * 0.5)
            box_label.next_to(box, UP, buff=0.1 * box.get_height())

        box_label.add_updater(update_box_label)
        dec.f_always.set_value(get_scale_factor)
        box.add_updater(lambda m: m.set_height((1 / get_scale_factor()) * FRAME_HEIGHT))

        box_exp_tracker.set_value(0)
        self.add(box, box_label, droste_image)
        self.play(
            box_exp_tracker.animate.set_value(math.log(256)).set_anim_args(time_span=(0, 10)),
            frame_exp_tracker.animate.set_value(math.log(256) - 1.5).set_anim_args(time_span=(4, 12)),
        )
        self.wait()
        self.play(frame_exp_tracker.animate.set_value(0), run_time=2)
        self.wait()


class CreatePiHouseLog(InteractiveScene):
    def construct(self):
        # Read from rectified image as polar coordinates
        piece = TexturedSurface(
            Square3D(resolution=(1000, 1000)),
            Path(get_texture_folder(), "UltraHighResPiHouse.png"),
        )
        piece.set_height(FRAME_HEIGHT)
        piece.set_shading(0, 0, 0)

        def to_polar(u, v):
            z1 = np.exp(complex(-(1 - v) * np.log(16), TAU * u))
            z2 = (z1 + complex(1, 1)) / 2
            return (z2.real, z2.imag)

        piece.set_image_coords_by_uv_func(to_polar)
        piece.set_shape(FRAME_WIDTH, FRAME_HEIGHT)
        self.add(piece)


class UsePiCreatureLog(InteractiveScene):
    def construct(self):
        # Test

        pass


class PalleteOfFunctions(InteractiveScene):
    def construct(self):
        # Add example
        frame = self.frame
        frame.set_height(9)

        examples = VGroup(
            self.get_input_output_grid(R"{c} \cdot z", lambda z: (1.5 + 0.8j) * z),
            self.get_input_output_grid("z^2", lambda z: z**2),
            self.get_input_output_grid("z^3", lambda z: z**3),
            self.get_input_output_grid(
                "e^z", np.exp,
                in_x_range=(-7, 7),
                out_x_range=(-7, 7),
                in_grid_dims=(25, 16),
                in_grid_height=6.25,
                in_grid_z0=-2,
            ),
            log_example := self.get_input_output_grid(
                R"\ln(z)", np.log,
                in_x_range=(-2, 2),
                out_x_range=(-4, 4),
                in_grid_dims=(16, 16),
                in_grid_height=4,
                in_grid_z0=-2 - 2j,
            ),
            self.get_input_output_grid(
                R"\cos(z)", np.cos,
                in_grid_dims=(8, 16),
                in_grid_z0=-2,
            ),
            self.get_input_output_grid(
                R"\sin(z)", np.sin,
                in_grid_dims=(8, 16),
                in_grid_z0=-2,
            ),
            self.get_input_output_grid(R"\zeta(z)", lambda z: z),
            self.get_input_output_grid(R"\wp(z)", lambda z: z),
            self.get_input_output_grid(R"j(\tau)", lambda z: z),
        )
        for n, example in enumerate(examples):
            example.shift((3 - 3 * n) * UP - example[1].get_center())
        examples[5:].shift(15 * UP + 15 * RIGHT)

        self.add(examples)

        # Tweak log example
        in_plane = log_example[1]
        out_plane = log_example[3]
        log_grid = get_nested_square_grid(n_recursions=6)
        log_grid.replace(in_plane)
        log_grid.insert_n_curves(10)
        out_grid = log_grid.copy()
        for piece in out_grid.family_members_with_points():
            piece.scale(0.99)
        apply_func_between_planes(out_grid, np.log, in_plane, out_plane)
        for piece in out_grid.family_members_with_points():
            piece.scale(1.0 / 0.99)
        log_grid.set_stroke(BLUE, 1)
        out_grid.set_stroke(PINK, 1)

        log_example.replace_submobject(-2, log_grid)
        log_example.replace_submobject(-1, out_grid)

        # Cover up with question marks
        cover_rects = VGroup()
        all_q_marks = VGroup()
        for example in examples[3:]:
            planes = example[1:4]
            rect = BackgroundRectangle(planes, buff=MED_SMALL_BUFF)
            rect.set_fill(GREY_E, 1)
            q_marks = VGroup(
                Tex(R"?").set_height(0.5 * planes[0].get_height()).move_to(piece)
                for piece in planes
            )

            cover_rects.add(rect)
            all_q_marks.add(q_marks)

        self.add(cover_rects)
        self.add(all_q_marks)
        self.play(
            frame.animate.reorient(0, 0, 0, (7, -3, 0.0), 15.5),
            run_time=2
        )

        # Highlight exp and log
        exp_example, log_example = exp_log = examples[3:5]

        rect = SurroundingRectangle(exp_log, buff=0.5)

        self.play(
            ShowCreation(rect),
            FadeOut(cover_rects[:2]),
            FadeOut(all_q_marks[:2]),
        )
        self.play(
            frame.animate.reorient(0, 0, 0, (0.64, -7.71, 0.0), 7.26),
            examples[:3].animate.fade(0.8),
            FadeOut(examples[5:]),
            FadeOut(cover_rects[2:], time_span=(3, 4)),
            FadeOut(all_q_marks[2:], time_span=(3, 4)),
            rect.animate.set_stroke(width=1),
            run_time=4,
        )
        self.wait()

        # Emphasize the log
        self.remove(out_grid)
        self.play(
            FadeOut(out_grid.copy(), time_span=(0, 1)),
            TransformFromCopy(log_grid, out_grid),
            run_time=4,
        )
        self.wait()
        return

        # Focus on exp
        self.play(
            rect.animate.surround(exp_example, buff=0.5),
            log_example.animate.fade(0.8),
            frame.animate.match_y(exp_example).set_height(6.5),
            run_time=2
        )
        self.wait()

    def get_input_output_grid(
        self,
        tex_label,
        func,
        in_x_range=(-2, 2),
        out_x_range=(-4, 4),
        in_grid_dims=(8, 8),
        in_grid_height=2,
        in_grid_z0=0,
        plane_height=2,
        in_grid_color=BLUE,
        out_grid_color=PINK
    ):
        func_label = Tex(tex_label, font_size=72)
        in_plane, out_plane = planes = VGroup(
            ComplexPlane(in_x_range, in_x_range, faded_line_ratio=0),
            ComplexPlane(out_x_range, out_x_range, faded_line_ratio=0),
        )
        for plane in planes:
            plane.set_height(plane_height)
            plane.background_lines.set_stroke(BLUE, 1, 0.5)
            plane.add_coordinate_labels(font_size=16 / in_x_range[1], buff=0.1 / in_x_range[1])

        group = VGroup(func_label, in_plane, Vector(RIGHT), out_plane)
        group.arrange(RIGHT)
        func_label.shift(0.5 * LEFT)

        in_grid = Square().get_grid(*in_grid_dims, buff=0)
        in_grid.set_height(in_plane.get_unit_size() * in_grid_height)
        in_grid.move_to(in_plane.n2p(in_grid_z0), DL)
        in_grid.insert_n_curves(20)
        out_grid = apply_func_between_planes(in_grid.copy(), func, in_plane, out_plane)
        out_grid.always.clip_to_box(out_plane)
        in_grid.set_stroke(in_grid_color, 1)
        out_grid.set_stroke(out_grid_color, 1)

        group.add(in_grid, out_grid)
        return group


class TheExponential(InteractiveScene):
    def construct(self):
        # Set up input and output space, top an bottom
        in_line = NumberLine((-5, 5), unit_size=1.5)
        in_line.move_to(2 * UP)
        in_line.add_numbers(font_size=20)

        out_line = NumberLine((-21, 21), unit_size=1.5)
        out_line.move_to(2 * DOWN)
        out_line.add_numbers(font_size=20)

        arrow_buff = 0.75
        func_arrow = Arrow(in_line.n2p(0), out_line.n2p(1), buff=arrow_buff, thickness=6)
        func_label = Tex(R"e^{x}", font_size=72)
        func_label.next_to(func_arrow.get_center(), UR, buff=MED_SMALL_BUFF)
        og_func_arrow = func_arrow.copy()

        self.add(in_line, out_line)

        # Show example dots
        sample_xs = np.arange(-4, 4.25, 0.25)
        in_dots = Group(*[TrueDot(in_line.n2p(x)) for x in sample_xs])
        out_dots = Group(*[TrueDot(out_line.n2p(np.exp(x))) for x in sample_xs])
        all_dots = Group(in_dots, out_dots)
        for dots in all_dots:
            for dot in dots:
                dot.set_radius(0.1)
                dot.make_3d()
                dot.deactivate_depth_test()
                dot.set_glow_factor(0.15)
            dots.set_submobject_colors_by_gradient(YELLOW, BLUE, interp_by_hsl=True)

        self.play(LaggedStartMap(FadeIn, in_dots, lag_ratio=0.1, run_time=1))
        self.play(
            GrowArrow(func_arrow),
            FadeIn(func_label, 0.25 * func_arrow.get_vector()),
            TransformFromCopy(in_dots, out_dots, lag_ratio=0.01, run_time=2),
        )
        self.wait()

        # Show a few real-valued inputs
        frame = self.frame
        x_tracker = ValueTracker(0)
        get_x = x_tracker.get_value

        def get_y():
            return np.exp(get_x())

        in_dot = in_dots[-1].copy()
        out_dot = out_dots[0].copy()
        in_dot.add_updater(lambda m: m.move_to(in_line.n2p(get_x())))
        out_dot.add_updater(lambda m: m.move_to(out_line.n2p(get_y())))

        active_func_arrow = func_arrow.copy()
        active_func_arrow.add_updater(lambda m: m.set_points_by_ends(
            in_line.n2p(get_x()),
            out_line.n2p(get_y()),
            buff=0.5
        ))

        active_func_label = Tex(R"e^{+0.00} = 1.000")
        active_func_label.make_number_changeable("+0.00", include_sign=True).f_always.set_value(get_x)
        active_func_label.make_number_changeable("1.000").f_always.set_value(get_y)
        active_func_label.always.next_to(active_func_arrow.get_center(), UR, MED_SMALL_BUFF)

        output_labels = VGroup(
            Tex(Rf"e^{{{n}}}", font_size=36).next_to(out_line.n2p(np.exp(n)), UP, MED_SMALL_BUFF)
            for n in range(4)
        )

        all_dots.target = all_dots.generate_target()
        all_dots.target.set_opacity(0.25)
        for dot, opacity in zip(all_dots.target[1][:12], np.linspace(0, 0.25, 12)):
            dot.set_opacity(opacity)

        self.play(
            MoveToTarget(all_dots),
            ReplacementTransform(func_arrow, active_func_arrow),
            FadeTransformPieces(func_label, active_func_label),
            FadeIn(in_dot),
            FadeIn(out_dot),
        )
        self.wait()
        self.play(FlashAround(in_dot))
        self.play(TransformFromCopy(in_dot, out_dot, suspend_mobject_updating=True, path_arc=30 * DEG))
        self.add(output_labels[0])
        self.play(FlashAround(out_dot))
        self.wait()
        self.play(
            x_tracker.animate.set_value(1),
            frame.animate.set_x(5),
            run_time=5
        )
        self.add(output_labels[1])
        self.wait()
        self.play(
            x_tracker.animate.set_value(2),
            frame.animate.set_x(5),
        )
        self.add(output_labels[2])
        self.wait()
        self.play(
            x_tracker.animate.set_value(3),
            frame.animate.set_height(16).set_x(17),
            run_time=2
        )
        self.add(output_labels[3])
        self.wait()

        # Show negative inputs
        self.play(
            x_tracker.animate.set_value(-1),
            frame.animate.set_height(8).move_to(ORIGIN),
            run_time=8
        )
        self.play(x_tracker.animate.set_value(-4), run_time=6)
        self.wait()
        self.play(FadeOut(Group(in_dots, out_dots, in_dot, out_dot, active_func_label, output_labels)))

        # Transition to a side-by-side view
        x_max = 7
        in_plane, out_plane = [
            ComplexPlane(
                (-x_max, x_max),
                (-x_max, x_max),
                unit_size=0.4,
                faded_line_ratio=0,
                background_line_style=dict(stroke_color=BLUE_D, stroke_width=1),
                axis_config=dict(stroke_color=GREY_A, stroke_width=1),
            )
            for _ in range(2)
        ]
        in_plane.to_edge(LEFT)
        out_plane.to_edge(RIGHT)
        func_arrow.set_points_by_ends(in_plane.get_right(), out_plane.get_left(), buff=0.25)
        func_label = Tex(R"e^{z}")
        func_label.next_to(func_arrow, DOWN, SMALL_BUFF)

        for line, plane, in [(in_line, in_plane), (out_line, out_plane)]:
            line.target = line.generate_target()
            scale_factor = plane.x_axis.get_unit_size() / line.get_unit_size()
            shift_vect = plane.n2p(0) - line.n2p(0)
            line.target.scale(scale_factor)
            line.target.shift(shift_vect)
            line.target.set_opacity(0)
            plane.add_coordinate_labels(font_size=12, buff=0.05)
            plane.save_state()
            plane.scale(1.0 / scale_factor)
            plane.shift(-shift_vect)
            plane.set_opacity(0)

        active_func_arrow.clear_updaters()
        self.play(
            LaggedStart(
                AnimationGroup(MoveToTarget(in_line), Restore(in_plane)),
                AnimationGroup(MoveToTarget(out_line), Restore(out_plane)),
                lag_ratio=0.3
            ),
            ReplacementTransform(active_func_arrow, func_arrow),
            FadeIn(func_label, time_span=(1.5, 2)),
            run_time=2
        )
        self.wait()

        # Add input and output dots
        z_tracker = ComplexValueTracker(0)
        get_z = z_tracker.get_value

        def get_w():
            return np.exp(get_z())

        in_dot.clear_updaters()
        out_dot.clear_updaters()

        in_dot.set_radius(0.06)
        out_dot.set_radius(0.06)
        in_dot.add_updater(lambda m: m.move_to(in_plane.n2p(get_z())))
        out_dot.add_updater(lambda m: m.move_to(out_plane.n2p(get_w())))

        z_label = VGroup(Tex(R"z = "), DecimalNumber(complex(1, 1)))
        z_label.arrange(RIGHT)
        z_label.set_height(0.2)
        z_label[1].shift(0.05 * UP)
        z_label.set_backstroke(BLACK, 3)
        z_label.set_z_index(1)
        z_label[1].f_always.set_value(get_z)
        z_label.always.next_to(in_dot, UR, buff=0.05)

        self.play(
            FadeIn(in_dot),
            FadeIn(out_dot),
            FadeIn(z_label),
            in_plane.background_lines.animate.set_stroke(opacity=0.5),
            out_plane.background_lines.animate.set_stroke(opacity=0.5),
        )

        # Walk up the imaginary line
        def apply_exp_to_input_shape(mob):
            mob.apply_points_function(lambda ps: np.array([
                out_plane.n2p(np.exp(in_plane.p2n(p)))
                for p in ps
            ]), about_point=ORIGIN)
            return mob

        def get_line_circle_pair(x, line_color=BLUE, circle_color=YELLOW, stroke_width=2):
            v_line = Line(in_plane.n2p(x), in_plane.n2p(complex(x, TAU)))
            v_line.set_stroke(line_color, stroke_width)
            out_circle = apply_exp_to_input_shape(v_line.copy().insert_n_curves(100).set_stroke(circle_color))
            return v_line, out_circle

        v_lines = VGroup()
        out_circles = VGroup()
        x_values = [0, 1, 2, -1, -2]
        for x in x_values:
            v_line, out_circle = get_line_circle_pair(x)

            self.play(z_tracker.animate.set_value(x))
            self.play(
                z_tracker.animate.increment_value(complex(0, TAU)),
                ShowCreation(v_line),
                ShowCreation(out_circle),
                run_time=6
            )
            self.wait()
            self.play(z_tracker.animate.set_value(x), run_time=2)

            v_lines.add(v_line)
            out_circles.add(out_circle)

        self.play(z_tracker.animate.set_value(2), run_time=2)
        self.wait()

        # Show one unit to one radian
        z_tracker.set_value(complex(2, TAU))
        z_unit = in_plane.get_unit_size()
        left_rect = Rectangle(2 * z_unit, z_unit)
        left_rect.move_to(in_plane.n2p(0), DL)
        left_rect.insert_n_curves(100)
        left_rect.set_stroke(TEAL, 2)
        left_rect.set_fill(TEAL, 0.3)
        brace = Brace(left_rect, RIGHT, SMALL_BUFF)
        brace.stretch(0.5, 0, about_edge=LEFT)
        brace_label = brace.get_tex("+i", font_size=30)
        left_rect.save_state()
        left_rect.stretch(1e-2, 1, about_edge=DOWN)

        rad_label = Text("+1 Radian", font_size=30)
        rad_label.add_updater(lambda m: m.move_to(
            out_plane.n2p(np.exp(z_tracker.get_value() - 0.5j - 0.3))
        ))
        rad_label.set_backstroke(BLACK, 3)

        sector = always_redraw(
            lambda: apply_func_between_planes(left_rect.copy(), np.exp, in_plane, out_plane)
        )

        two_pi_brace = Brace(v_lines, RIGHT, SMALL_BUFF)
        two_pi_label = two_pi_brace.get_tex(R"+2\pi i")

        self.play(
            z_tracker.animate.set_value(2),
            frame.animate.reorient(0, 0, 0, (-2.7, 0.42, 0.0), 2.48),
            run_time=1,
        )
        self.play(
            Restore(left_rect),
            GrowFromPoint(brace, brace.get_bottom()),
            FadeIn(brace_label, 0.1 * UP),
            z_tracker.animate.increment_value(1j),
            run_time=2
        )
        self.wait()
        sector.update()
        rad_label.update()
        rad_label.suspend_updating()
        self.play(
            TransformFromCopy(left_rect, sector),
            FadeTransformPieces(brace_label, rad_label),
            frame.animate.reorient(0, 0, 0, (5.8, 1.11, 0.0), 3.22),
            run_time=3
        )
        self.wait()
        rad_label.resume_updating()
        left_group = VGroup(left_rect, brace, brace_label)
        for n in range(5):
            self.play(
                frame.animate.to_default_state(),
                z_tracker.animate.increment_value(1j),
                left_group.animate.shift(z_unit * UP),
                run_time=1,
            )
            self.wait()
        self.play(
            z_tracker.animate.increment_value((TAU - 6) * 1j),
            FadeOut(left_rect),
            FadeOut(sector),
            FadeOut(rad_label),
            ReplacementTransform(brace, two_pi_brace),
            FadeTransform(brace_label, two_pi_label),
        )
        self.wait()
        self.play(
            FadeOut(two_pi_brace),
            FadeOut(two_pi_label),
        )

        # Note the height of the lines
        brace = Brace(v_lines, LEFT, buff=SMALL_BUFF)
        brace_label = brace.get_tex(R"2\pi")

        self.play(
            GrowFromCenter(brace),
            Write(brace_label),
        )
        self.wait()

        # Map lines to circles
        v_lines.sort(lambda p: p[0])
        out_circles.submobjects.sort(key=lambda m: m.get_width())
        out_circles.note_changed_family(only_changed_order=True)
        circle_ghosts = out_circles.copy()
        circle_ghosts.set_stroke(opacity=0.25)

        self.add(circle_ghosts)
        self.play(
            FadeOut(out_circles),
            LaggedStartMap(ShowCreation, v_lines, lag_ratio=0.1),
            run_time=2
        )
        self.play(LaggedStart(
            (TransformFromCopy(line, circle, path_arc=(0, 0.3 * PI))
            for line, circle in zip(v_lines, out_circles)),
            lag_ratio=0.25,
            run_time=5
        ))
        self.wait()

        # Emphasize spacing by 1
        top_arrows = VGroup(
            Arrow(vl1.get_top(), vl2.get_top(), path_arc=-180 * DEG, thickness=2, buff=0.1, fill_color=TEAL)
            for vl1, vl2 in zip(v_lines, v_lines[1:])
        )
        top_labels = VGroup(
            Tex(R"+1", font_size=20).next_to(arrow, UP, buff=0.05)
            for arrow in top_arrows
        )

        self.play(
            LaggedStartMap(Write, top_arrows, lag_ratio=0.25),
            LaggedStartMap(FadeIn, top_labels, shift=0.1 * UP, lag_ratio=0.25),
        )
        self.wait()

        # Emphasize scale factor of e
        out_arrows = VGroup()
        scale_labels = VGroup()
        for c1, c2 in zip(out_circles[2:], out_circles[3:]):
            arrows = VGroup(
                Arrow(c1.pfp(a), c2.pfp(a), buff=0.1 * c1.get_width(), thickness=3)
                for a in np.arange(0, 1, 1.0 / 8)
            )
            arrows.set_fill(YELLOW_E)
            arrows.set_backstroke(BLACK, 3)
            out_arrows.add(arrows)
            scale_label = Tex(R"\times e")
            scale_label.set_max_width(0.8 * arrows[0].get_width())
            scale_label.next_to(arrows[0], UP, SMALL_BUFF)
            scale_labels.add(scale_label)

        circle_ghosts.set_stroke(opacity=0.5)
        v_line_ghosts = v_lines.copy().set_stroke(opacity=0.5)

        self.add(circle_ghosts, v_line_ghosts)
        self.play(
            FadeOut(out_circles[:2]),
            FadeOut(out_circles[3:]),
            FadeOut(v_lines[:2]),
            FadeOut(v_lines[3:]),
            z_tracker.animate.set_value(0),
        )
        self.wait()
        for n in [0, 1]:
            self.play(
                LaggedStartMap(GrowArrow, out_arrows[n], lag_ratio=1e-2),
                FadeIn(scale_labels[n], shift=0.1 * RIGHT, scale=2),
                TransformFromCopy(out_circles[n + 2], out_circles[n + 3]),
                TransformFromCopy(v_lines[n + 2], v_lines[n + 3]),
                z_tracker.animate.set_value(n + 1),
                run_time=3
            )
            self.wait()

        # Clean up the board
        self.play(
            FadeOut(VGroup(out_arrows, scale_labels, brace, brace_label, top_arrows, top_labels), lag_ratio=0.02),
            FadeOut(Group(in_dot, out_dot, z_label), lag_ratio=0.1),
            FadeOut(v_line_ghosts),
            FadeOut(circle_ghosts),
            v_lines.animate.set_stroke(opacity=1),
            out_circles.animate.set_stroke(opacity=1),
        )

        # Show an example grid from where the v-lines have been drawn
        grid_density = 4
        grid_width = (len(v_lines) - 1) * grid_density
        in_grid = Square().get_grid(int(TAU * grid_density), grid_width, buff=0)
        in_grid.sort(lambda p: np.dot(p, (0.1, 1, 0)))
        in_grid.match_width(v_lines)
        in_grid.move_to(v_lines, DOWN)
        in_grid.set_stroke(width=1)
        in_grid.set_submobject_colors_by_gradient(BLUE, YELLOW, interp_by_hsl=True)
        in_grid_ghost = in_grid.copy()
        in_grid_ghost.set_stroke(width=1, opacity=0.5)

        out_grid = apply_exp_to_input_shape(in_grid.copy().insert_n_curves(10))

        self.play(
            Write(in_grid),
            out_circles.animate.set_stroke(WHITE, 1),
            v_lines.animate.set_stroke(WHITE, 1),
        )
        self.wait()
        self.remove(in_grid)
        self.add(in_grid_ghost)
        self.play(
            out_plane.background_lines.animate.set_stroke(opacity=0.25),
            *(
                ReplacementTransform(in_square.copy(), out_square, path_arc=0.35 * z.imag + 0.05 * (z.real + 2))
                for in_square, out_square in zip(in_grid, out_grid)
                for z in [in_plane.p2n(in_square.get_center())]
            ),
            run_time=3
        )
        self.wait()

        # Cycle through squares
        solid_in_grid = in_grid.copy()
        solid_in_grid.sort(lambda p: np.dot(p, (1, 0.01, 0)))
        for square in solid_in_grid:
            square.set_fill(square.get_stroke_color(), 0.5)
            square.set_stroke(width=2)
        solid_out_grid = apply_exp_to_input_shape(solid_in_grid.copy().insert_n_curves(10))

        self.play(
            ShowSubmobjectsOneByOne(solid_in_grid),
            ShowSubmobjectsOneByOne(solid_out_grid),
            run_time=10,
            rate_func=lambda a: (0.5 + 0.5 * a),
        )
        self.remove(solid_in_grid, solid_out_grid)
        self.wait()

        # Re-emphasize the lines
        v_lines.set_stroke(BLUE, 3)
        out_circles.set_stroke(YELLOW, 3)

        self.play(
            *(
                LaggedStartMap(ShowCreation, group, lag_ratio=0.5, run_time=3)
                for group in [v_lines, out_circles]
            ),
            in_grid_ghost.animate.set_stroke(opacity=0.25),
            out_grid.animate.set_stroke(opacity=0.25),
        )
        self.play(LaggedStart(
            (TransformFromCopy(line, circle, path_arc=(0, 0.25 * PI))
            for line, circle in zip(v_lines, out_circles)),
            lag_ratio=0.1,
            run_time=3
        ))
        self.wait()

        # Full grid
        full_grid = Square3D(resolution=(10, 10)).get_grid(4 * x_max, 4 * x_max, buff=0)
        full_grid.match_width(in_plane)
        full_grid.move_to(in_plane)
        full_grid.set_shading(0, 0, 0)
        for n, square in enumerate(full_grid):
            row = n // (4 * x_max)
            col = n % (4 * x_max)
            parity = (row + col) % 2
            square.set_color([GREY_C, GREY_E][parity], 1)
        full_grid.sort(lambda p: np.dot(p, (0.01, 1, 0)))

        full_v_lines = VGroup(
            Line(in_plane.c2p(x, -x_max), in_plane.c2p(x, x_max)).insert_n_curves(100)
            for x in range(-x_max, x_max + 1)
        )
        full_v_lines.set_stroke(RED, 3)
        full_v_lines.set_submobject_colors_by_gradient(YELLOW, RED)
        full_v_lines.apply_depth_test()

        self.play(
            *map(FadeOut, [in_grid_ghost, v_lines, out_grid, out_circles]),
            ShowCreation(full_v_lines, lag_ratio=0.25, run_time=5),
        )
        self.wait()
        self.play(FadeIn(full_grid, lag_ratio=1e-3, run_time=1))
        self.wait()

        # Roll up into a cylinder
        def cylinder_func(points, threshold=0):
            xs = in_plane.x_axis.p2n(points)
            ys = in_plane.y_axis.p2n(points)
            theta = threshold - ys
            scale_factor = np.exp(0.01 * ys)
            spiral_points = np.array([xs.T, threshold - scale_factor * np.sin(theta), 1 - scale_factor * np.cos(theta)]).T
            result = np.array([xs, ys, np.zeros_like(xs)]).T
            curved_indices = (ys < threshold)
            result[curved_indices] = spiral_points[curved_indices]
            return result

        threshold_tracker = ValueTracker(-x_max)

        def update_rolled_cylinder(rolled_cylinder):
            for sm1, sm2 in zip(rolled_cylinder, rolled_cylinder.saved_state):
                sm1.match_points(sm2)
                sm1.match_color(sm2)
            threshold = threshold_tracker.get_value()
            rolled_cylinder.apply_points_function(lambda ps: cylinder_func(ps, threshold))
            rolled_cylinder.match_width(full_grid)
            rolled_cylinder.shift(full_grid[-1].get_end() - rolled_cylinder[-1].get_end())
            return rolled_cylinder

        rolled_cylinder = full_grid.copy()
        rolled_cylinder.save_state()
        rolled_v_lines = full_v_lines.copy()
        rolled_v_lines.save_state()

        self.remove(full_grid, full_v_lines)
        self.add(rolled_cylinder)
        self.play(
            frame.animate.reorient(58, 70, 0, (-1.74, 2.71, 0.02), 6.40),
            UpdateFromFunc(rolled_cylinder, update_rolled_cylinder),
            UpdateFromFunc(rolled_v_lines, update_rolled_cylinder),
            threshold_tracker.animate.set_value(x_max).set_anim_args(time_span=(2, 7)),
            run_time=8,
        )
        self.wait()

        # Show circle circumference
        circle = Circle(radius=in_plane.x_axis.get_unit_size())
        circle.set_color(RED)
        circle.flip(RIGHT)
        circle.rotate(90 * DEG, UP)
        circle.move_to(rolled_cylinder, RIGHT)
        circle.set_stroke(WHITE, 5)

        circle_center = circle.get_center()
        circum_dec = DecimalNumber(0, font_size=24)
        circum_dec.save_state()
        circum_tracker = ValueTracker(0)

        def update_circum_dec(circum_dec):
            circum_dec.restore()
            if circum_tracker.get_value() >= TAU - 1e-5:
                circum_dec.become(Tex(R"2\pi", font_size=36))
            else:
                circum_dec.set_value(circum_tracker.get_value())
            circum_dec.rotate(90 * DEG, RIGHT).rotate(90 * DEG, OUT)
            circum_dec.move_to(circle_center + 1.5 * (circle.get_end() - circle_center))

        circum_dec.add_updater(update_circum_dec)

        self.add(circum_dec)
        self.play(
            ShowCreation(circle),
            circum_tracker.animate.set_value(TAU),
            run_time=3
        )
        circum_dec.update()
        circum_dec.clear_updaters()
        self.wait()
        self.play(FadeOut(circle), FadeOut(circum_dec))
        self.wait()

        # Squish the cylinder
        cylinder = Group(rolled_cylinder, rolled_v_lines)
        cylinder.apply_depth_test()

        cylinder.target = cylinder.generate_target()
        cylinder.target.rotate(90 * DEG, DOWN)
        cylinder.target.next_to(out_plane, OUT, buff=1)

        def squish(points):
            xs, ys, zs = points.T
            xs, ys = normalize_along_axis(np.array([xs, ys]).T, 1).T
            radius = np.exp(zs)
            return np.array([radius * xs, radius * ys, np.zeros_like(xs)]).T

        central_cylinder = cylinder.target.copy()
        central_cylinder.set_width(2)
        central_cylinder.center()

        for mob in [cylinder.target, central_cylinder]:
            sorted_pieces = list(mob.family_members_with_points())
            sorted_pieces.sort(key=lambda m: m.get_z())
            for piece in sorted_pieces[int(0.82 * len(sorted_pieces)):]:
                piece.fade(1)

        central_cylinder.apply_points_function(squish, about_point=ORIGIN)
        central_cylinder.scale(out_plane.x_axis.get_unit_size(), about_point=ORIGIN)
        central_cylinder.shift(out_plane.get_center())

        self.play(
            MoveToTarget(cylinder),
            frame.animate.reorient(-1, 72, 0, (3.42, 0.82, 1.64), 11.83),
            run_time=8
        )
        central_cylinder.deactivate_depth_test()
        cylinder.deactivate_depth_test()

        self.play(Transform(cylinder, central_cylinder, run_time=3))
        self.wait()

        # Show input and output space again
        cylinder.set_clip_plane(RIGHT, 20)
        out_plane.background_lines.set_stroke(BLUE, 1, 1)
        in_plane.background_lines.set_stroke(BLUE, 1, 1)
        full_grid.set_opacity(0.5)
        v_lines.deactivate_depth_test()

        self.add(full_grid, out_plane, in_plane, full_v_lines)
        self.play(
            frame.animate.to_default_state(),
            cylinder.animate.set_clip_plane(RIGHT, -1).set_anim_args(time_span=(0, 1.5)),
            FadeIn(out_plane),
            FadeIn(full_grid, time_span=(2, 3)),
            FadeIn(full_v_lines, time_span=(2, 3)),
            FadeIn(in_plane),
            run_time=3
        )
        self.wait()
        self.play(
            ShowCreation(full_v_lines, lag_ratio=0.1),
            ShowCreation(rolled_v_lines, lag_ratio=0.1),
            run_time=3
        )
        self.wait()

        # Clear the board
        self.play(
            FadeOut(Group(full_grid, full_v_lines, cylinder)),
            in_plane.background_lines.animate.set_stroke(BLUE, 1, 0.5),
            out_plane.background_lines.animate.set_stroke(BLUE, 1, 0.5),
        )
        self.wait()

        # Show three points colliding
        z_range = [-TAU * 1j, 0, TAU * 1j] + [
            unit * n * TAU * 1j
            for n in range(2, int(200 / TAU))
            for unit in [-1, 1]
        ]
        input_dots = VGroup(
            Dot(in_plane.n2p(z))
            for z in z_range
        )
        in_dot_labels = VGroup(Tex(R"-2\pi i"), Tex(R"0"), Tex(R"2 \pi i"))
        for dot, label, vect in zip(input_dots, in_dot_labels, [LEFT, 0.5 * UL, LEFT]):
            label.next_to(dot, vect, SMALL_BUFF)

        out_dot_one = Dot(out_plane.n2p(1))
        out_dot_label = Tex(R"1")
        out_dot_label.next_to(out_dot_one, UR, SMALL_BUFF)

        in_plane_label = Tex(R"z").next_to(in_plane, UP)
        out_plane_label = Tex(R"e^{z}").next_to(out_plane, UP)

        dot_arrows = VGroup(
            Arrow(input_dot, out_dot_one, thickness=4, fill_color=RED, fill_opacity=0.75)
            for input_dot in input_dots
        )
        dot_arrows.set_fill(opacity=0.75)

        self.play(
            LaggedStartMap(FadeIn, input_dots),
            LaggedStartMap(FadeIn, in_dot_labels),
            in_plane.coordinate_labels[14:].animate.set_opacity(0),
            ReplacementTransform(func_label, out_plane_label),
            TransformFromCopy(func_label[1], in_plane_label[0]),
            func_arrow.animate.match_y(out_plane_label),
        )
        self.wait()
        self.play(
            LaggedStartMap(GrowArrow, dot_arrows[:3], lag_ratio=0.25),
            LaggedStart(
                (TransformFromCopy(input_dot, out_dot_one)
                for input_dot in input_dots[:3]),
                lag_ratio=0.25,
            )
        )
        self.wait()

        # Flip arrows
        self.play(
            *(
                Transform(
                    arrow,
                    arrow.copy().rotate(PI),
                    rate_func=there_and_back_with_pause,
                    run_time=3,
                    path_arc=45 * DEG,
                )
                for arrow in [*dot_arrows[:3], func_arrow]
            )
        )
        self.wait()

        # Equations
        equations = VGroup(
            Tex(R"e^{-2\pi i} = 1"),
            Tex(R"e^{0} = 1"),
            Tex(R"e^{2\pi i} = 1"),
        )
        z_tracker.set_value(0)
        for equation, y in zip(equations, [-2, 0.35, 2]):
            equation.move_to(y * UP)

        self.play(
            FadeIn(equations[1]),
            FadeIn(in_dot),
            FadeIn(out_dot),
            dot_arrows[0].animate.set_opacity(0.1),
            dot_arrows[2].animate.set_opacity(0.1),
        )
        self.wait()
        self.play(
            z_tracker.animate.set_value(complex(0, TAU)),
            LaggedStart(
                FadeIn(equations[2], UP),
                FadeOut(equations[1], UP),
                lag_ratio=0.1,
            ),
            dot_arrows[1].animate.set_opacity(0.1),
            dot_arrows[2].animate.set_opacity(1),
            run_time=2
        )
        self.wait()
        self.play(
            z_tracker.animate.set_value(complex(0, -TAU)),
            LaggedStart(
                FadeIn(equations[0], DOWN),
                FadeOut(equations[2], DOWN),
                lag_ratio=0.1,
            ),
            dot_arrows[2].animate.set_opacity(0.1),
            dot_arrows[0].animate.set_opacity(1),
        )
        self.wait()

        # Show infinite sequence
        big_in_plane = ComplexPlane((-x_max, x_max), (-10, 200), faded_line_ratio=0)
        big_in_plane.axes.match_style(in_plane.axes)
        big_in_plane.background_lines.match_style(in_plane.background_lines)
        big_in_plane.replace(in_plane, dim_to_match=0)
        big_in_plane.x_axis.add_numbers(range(-x_max + 1, x_max), font_size=12, buff=0.05)

        new_in_dot_labels = VGroup(
            Tex(Rf"{n} \pi i").next_to(dot, LEFT)
            for dot, n in zip(input_dots[4::2], range(4, 30, 2))
        )

        dot_arrows[3:].set_opacity(0)

        corner_eq = Tex(R"e^{2\pi i n} = 1", font_size=60)
        corner_eq.fix_in_frame()
        corner_eq.to_corner(UR)

        self.play(
            dot_arrows.animate.set_opacity(0.5).set_anim_args(lag_ratio=0.01, run_time=3),
            FadeIn(new_in_dot_labels, lag_ratio=0.1, time_span=(1, 3)),
            FadeIn(corner_eq),
            FadeOut(equations[0]),
            FadeIn(big_in_plane),
            FadeOut(in_plane_label),
            FadeOut(func_arrow),
            FadeOut(in_plane, time_span=(0.5, 1)),
            frame.animate.reorient(6, 62, 0, (-0.01, -0.01, -0.01), 8.00).set_anim_args(run_time=3)
        )
        self.wait()
        for _ in range(10):
            self.play(
                z_tracker.animate.increment_value(TAU * 1j),
                dot_arrows.animate.set_opacity(0.2),
                run_time=2,
            )
        self.play(FadeOut(corner_eq))

        # Flip the arrows
        exp_arrow = Arrow(in_plane, out_plane, thickness=6)
        exp_label = Tex(R"e^{z}", font_size=72, t2c={"z": BLUE})
        log_label = Tex(R"\ln(w)", font_size=72, t2c={"w": PINK})
        exp_label.next_to(exp_arrow, UP, SMALL_BUFF)
        log_label.next_to(exp_arrow, DOWN, SMALL_BUFF)

        self.add(exp_arrow)
        self.add(exp_label)

        z_tracker.set_value(-4 * TAU * 1j)
        z_tracker.clear_updaters()
        z_tracker.add_updater(lambda m, dt: m.increment_value(complex(0, PI * dt)))
        self.add(z_tracker)

        frame.reorient(5, 60, 0, (-0.38, -0.12, -0.11), 8.22)
        self.wait(5)
        self.play(
            Rotate(exp_arrow, PI, time_span=(1, 2)),
            ReplacementTransform(exp_label, log_label, path_arc=PI, time_span=(1, 2)),
            LaggedStart(
                (arrow.animate.rotate(PI).set_opacity(0.4).set_anim_args(path_arc=30 * DEG)
                for arrow in dot_arrows),
                lag_ratio=(1.0 / len(dot_arrows)),
            ),
            frame.animate.reorient(0, 51, 0, (-0.01, -0.01, -0.01)),
            run_time=3
        )
        self.play(
            frame.animate.reorient.reorient(0, 62, 0, (-0.01, -0.01, -0.01), 8.22),
            run_time=12
        )
        self.wait(12)


class TheNaturalLog(InteractiveScene):
    def construct(self):
        # Set up input and output planes
        plane_style = dict(
            background_line_style=dict(
                stroke_color=BLUE,
                stroke_width=1,
                stroke_opacity=0.5,
            ),
            faded_line_style=dict(
                stroke_color=BLUE,
                stroke_width=1,
                stroke_opacity=0.1
            )
        )
        z_plane = ComplexPlane((-3, 3), (-1, 7), unit_size=0.8, **plane_style)
        big_z_plane = ComplexPlane((-10, 3), (-20, 20), unit_size=0.8, **plane_style)
        w_plane = ComplexPlane((-3, 3), (-3, 3), unit_size=0.85, **plane_style)

        z_plane.to_edge(LEFT)
        big_z_plane.shift(z_plane.get_origin() - big_z_plane.get_origin())
        w_plane.to_edge(RIGHT)

        for plane in [z_plane, big_z_plane, w_plane]:
            plane.add_coordinate_labels(font_size=16, buff=0.05)

        self.add(z_plane, w_plane)

        # Set up example lines and circles
        v_lines = VGroup(
            Line(z_plane.n2p(x), z_plane.n2p(x + TAU * 1j))
            for x in np.linspace(1, -1, 9)
        )
        v_lines.set_stroke(WHITE, 1)
        v_lines.set_submobject_colors_by_gradient(YELLOW, RED)

        def apply_func_to_z_space(mob, func):
            return apply_func_between_planes(mob, func, z_plane, w_plane)

        circles = apply_func_to_z_space(v_lines.insert_n_curves(100).copy(), np.exp)

        # Add arrows
        right_arrow = Arrow(z_plane, w_plane, thickness=5, fill_color=GREY_B)
        right_arrow.set_y(1)
        left_arrow = right_arrow.copy().rotate(PI)
        left_arrow.set_y(-1)

        right_arrow_label = Tex(R"z \rightarrow e^z", t2c={"z": BLUE})
        left_arrow_label = Tex(R"\ln(w) \leftarrow w", t2c={"w": PINK})
        right_arrow_label.next_to(right_arrow, UP, SMALL_BUFF)
        left_arrow_label.next_to(left_arrow, DOWN, SMALL_BUFF)

        self.add(right_arrow)
        self.add(right_arrow_label)

        self.play(FadeIn(v_lines))
        self.play(TransformFromCopy(v_lines, circles, lag_ratio=0.1, run_time=10, path_arc=(0, 0.8 * PI)))
        self.wait()
        self.play(
            TransformFromCopy(right_arrow, left_arrow, path_arc=-PI / 2),
            FadeTransformPieces(VGroup(*reversed(right_arrow_label.copy())), left_arrow_label, path_arc=-PI / 2),
            FadeOut(v_lines),
        )
        self.play(
            TransformFromCopy(circles, v_lines, lag_ratio=0.1, run_time=3, path_arc=(0, -0.8 * PI)),
        )
        self.wait()

        # Add Droste image
        frame = self.frame
        log_image = get_log_image(get_pi_house_log_image_path(), 16)
        log_image.scale(z_plane.get_unit_size())
        log_image.move_to(z_plane.n2p(math.log(3)), DR)

        droste_image = get_droste_from_log_image_path(get_pi_house_log_image_path(), 16)
        droste_image.match_height(w_plane)
        droste_image.move_to(w_plane)

        w_plane.target = w_plane.generate_target()
        w_plane.target.x_axis.set_stroke(BLACK)
        w_plane.target.y_axis.set_stroke(BLACK)
        w_plane.target.background_lines.set_stroke(GREY_C)
        w_plane.target.faded_lines.set_stroke(GREY_C)
        w_plane.target.coordinate_labels.set_fill(BLACK)
        w_plane.target.coordinate_labels[2].set_opacity(0)

        w_plane.set_z_index(1)
        self.play(
            FadeOut(circles),
            FadeOut(v_lines),
            FadeIn(droste_image),
            MoveToTarget(w_plane),
        )
        self.wait()
        self.play(
            frame.animate.scale(1 / 16, about_point=w_plane.get_origin()),
            run_time=7,
            rate_func=lambda t: there_and_back_with_pause(t, 1.0 / 7),
        )

        # Create slices
        delta_u = 0.025
        n_log_slice_repetitions = 2
        log_image_slices = Group(
            get_image_slice(log_image, delta_u, u)
            for n in range(n_log_slice_repetitions)
            for u in np.arange(0, 1, delta_u)[::-1]
        )
        log_image_slices.arrange(LEFT, buff=0)
        log_image_slices.move_to(log_image, DR)
        droste_image_rings = apply_func_to_z_space(log_image_slices.copy(), np.exp)

        # Set up in dot and out dot
        z_tracker = ComplexValueTracker()

        def get_z():
            return z_tracker.get_value()

        def get_w():
            return np.exp(z_tracker.get_value())

        z_dot = Group(GlowDot(), TrueDot()).set_color(BLUE)
        w_dot = Group(GlowDot(), TrueDot()).set_color(PINK)
        z_dot.f_always.move_to(lambda: z_plane.n2p(get_z()))
        w_dot.f_always.move_to(lambda: w_plane.n2p(get_w()))

        traced_paths = self.get_traced_paths([z_dot, w_dot])

        # Draw circle
        z_tracker.set_value(z_plane.p2n(log_image_slices[0].get_bottom()))
        self.play(
            FadeIn(z_dot),
            FadeIn(w_dot),
            droste_image.animate.set_opacity(0.25)
        )
        self.add(traced_paths)
        self.play(z_tracker.animate.increment_value(TAU * 1j), run_time=2)
        traced_paths.suspend_updating()
        self.wait()

        # Show image ring
        droste_ring = droste_image_rings[0]
        log_slice = log_image_slices[0]
        path_func = path_along_arc(arc_angle=np.array([-0.35 * z_plane.y_axis.p2n(p) for p in log_slice.get_points()]))

        self.play(
            ShowCreation(droste_ring),
            FadeOut(traced_paths[1]),
        )
        self.wait()
        self.play(
            TransformFromCopy(droste_ring, log_slice, path_func=path_func),
            FadeOut(traced_paths[0], time_span=(0, 1)),
            FadeOut(z_dot),
            FadeOut(w_dot),
            run_time=3
        )
        self.wait()
        traced_paths.clear_points()

        # Circle e times smaller
        circle = Circle().replace(droste_ring)
        line = Line(DOWN, UP).match_height(log_slice).move_to(log_slice, DR)
        VGroup(circle, line).set_stroke(WHITE, 2)

        index = int(1 / math.log(16) / delta_u)
        small_ring = droste_image_rings[index].copy()
        small_log_slice = log_image_slices[index].copy()

        in_arrows = VGroup(
            Arrow(3 * v, np.exp(-1) * 3 * v, thickness=4, fill_color=WHITE)
            for v in compass_directions(8)
        )
        in_arrows.replace(circle).scale(0.9)
        in_arrows.set_z_index(2)
        one_over_e_label = Tex(R"\times 1 / e", font_size=36)
        one_over_e_label.next_to(in_arrows[0], UP, buff=0)

        traced_paths.set_stroke(WHITE, 2)
        self.play(
            line.animate.shift(z_plane.get_unit_size() * LEFT),
            circle.animate.scale(np.exp(-1), about_point=w_plane.get_origin()),
            Write(one_over_e_label),
            *map(GrowArrow, in_arrows),
        )
        self.play(ShowCreation(small_ring), FadeOut(circle))
        self.play(
            TransformFromCopy(small_ring, small_log_slice, path_func=path_func),
            FadeOut(line, time_span=(2, 3)),
            run_time=3
        )
        one_arrow = Arrow(log_slice.get_top(), line.get_top(), path_arc=120 * DEG, buff=0.1)
        one_arrow_label = Tex(R"-1", font_size=36).next_to(one_arrow, UP, SMALL_BUFF)
        self.play(
            GrowArrow(one_arrow),
            FadeIn(one_arrow_label, 0.1 * UP),
        )
        self.wait()
        self.play(FadeOut(VGroup(one_arrow, one_arrow_label, in_arrows, one_over_e_label)))

        # Show all the rings in between, and down towards the center
        droste_image_rings.set_opacity(1)
        v_lines = VGroup(
            Line(piece.get_corner(DL), piece.get_corner(UL)).set_stroke(BLACK, 1, 0.5)
            for piece in log_image_slices
        )
        black_circles = VGroup(
            Circle().replace(ring).set_stroke(BLACK, width=np.clip(circle.get_width(), 0, 1), opacity=0.5)
            for ring in droste_image_rings
        )
        log_image_row = log_image.get_grid(1, 4, buff=0)
        log_image_row.move_to(log_image, RIGHT)
        log_image_row.set_opacity(0.9)
        log_image_slices.set_opacity(0.9)

        self.play(
            LaggedStartMap(FadeIn, droste_image_rings, lag_ratio=0.5),
            LaggedStartMap(FadeIn, log_image_slices, lag_ratio=0.5),
            LaggedStartMap(FadeIn, v_lines, lag_ratio=0.5),
            LaggedStartMap(FadeIn, black_circles, lag_ratio=0.5),
            FadeOut(small_ring, time_span=(2, 4)),
            FadeOut(small_log_slice, time_span=(2, 4)),
            run_time=25
        )
        self.remove(log_image_slices, droste_image_rings)
        droste_image.set_opacity(1)
        self.add(log_image_row, droste_image, v_lines, black_circles)
        self.play(
            v_lines.animate.set_stroke(opacity=0.25),
            black_circles.animate.set_stroke(opacity=0.25),
        )
        self.wait()

        # Reference leftward repetition
        self.play(
            frame.animate.set_x(-4.8),
            v_lines.animate.set_opacity(0),
            run_time=7,
            rate_func=there_and_back_with_pause,
        )

        # Show a labeled z_value
        z_tracker.set_value(1.0)
        z_label = VGroup(Tex(R"z = "), DecimalNumber(complex(0)))
        z_label.scale(0.5)
        z_label.arrange(RIGHT, buff=SMALL_BUFF)
        z_label[1].shift(0.02 * UP)
        z_label[1].f_always.set_value(get_z)
        z_label.always.next_to(z_dot, UR, buff=-0.1)
        z_label.set_backstroke(BLACK, 5)
        exp_z_label = Tex(R"e^z")
        exp_z_label.set_z_index(3)
        exp_z_label.set_backstroke(BLACK, 4)
        exp_z_label.always.next_to(w_dot, UR, buff=-0.1)

        traced_paths = self.get_traced_paths([z_dot, w_dot], colors=[BLUE, PINK])
        traced_paths.set_z_index(2)

        brace = Brace(log_image, RIGHT, SMALL_BUFF)
        brace_label = brace.get_tex(R"2\pi")

        w_dot[0].set_color(PINK)
        w_dot[1].set_color(MAROON_E)
        z_dot[1].set_color(BLUE_E)

        self.play(
            FadeIn(z_dot),
            FadeIn(z_label),
        )
        self.play(
            TransformFromCopy(z_dot, w_dot, suspend_mobject_updating=True),
            TransformFromCopy(z_label[0], exp_z_label),
        )
        self.add(traced_paths)
        self.play(
            z_tracker.animate.increment_value(TAU * 1j),
            run_time=3
        )
        self.play(
            GrowFromCenter(brace),
            FadeIn(brace_label, 0.25 * RIGHT),
        )
        self.wait()

        # Grow to 4πi
        log_image_slices.add(*log_image_slices.copy())
        log_image_slices.arrange(LEFT, buff=0)
        log_image_slices.move_to(log_image.get_corner(UR), DR)

        big_z_plane.set_z_index(-1)

        circle_points = log_image_slices[1].copy()
        circle_points.set_z_index(3)
        circle_points.save_state()
        apply_func_to_z_space(circle_points, np.exp)

        self.play(
            z_tracker.animate.increment_value(TAU * 1j),
            frame.animate.reorient(0, 0, 0, (0, 2, 0.0), 13),
            FadeOut(v_lines),
            FadeOut(black_circles),
            FadeOut(z_plane, time_span=(2, 4)),
            FadeIn(big_z_plane, time_span=(2, 4)),
            run_time=12
        )
        traced_paths[1].add_updater(lambda m: m.set_stroke(width=1.5))
        traced_paths.suspend_updating()
        self.play(
            ShowCreation(circle_points, lag_ratio=0),
            droste_image.animate.set_opacity(0.5),
        )
        self.play(
            Restore(circle_points, run_time=2),
            traced_paths[0].animate.set_stroke(width=0.5)
        )
        self.wait()
        self.play(
            ShowCreation(log_image_slices, lag_ratio=0.03, run_time=5),
            FadeOut(circle_points, time_span=(1, 2)),
            traced_paths[0].animate.set_stroke(width=2),
            droste_image.animate.set_opacity(1),
        )
        self.add(log_image_slices)
        self.wait()
        traced_paths.resume_updating()

        # Add more tiles below
        log_image_tiles = log_image.get_grid(7, 6, buff=0)
        log_image_tiles.move_to(log_image, RIGHT)
        log_image_tiles.set_opacity(0.8)

        self.remove(log_image_row, log_image_slices, droste_image)
        self.add(log_image_tiles, droste_image)
        self.wait()
        self.play(
            frame.animate.set_y(-1),
            z_tracker.animate.increment_value(-3 * TAU * 1j),
            run_time=12
        )
        self.wait()

        # Place point on the pi
        traced_paths.clear_updaters()
        v_line, pink_circle = traced_paths
        v_line.add_updater(lambda m: m.match_x(z_dot))
        pink_circle.add_updater(lambda m: m.set_width(2 * get_norm(w_dot.get_center() - w_plane.get_origin())).move_to(w_plane.get_origin()))

        self.add(log_image_tiles, z_label, z_dot)
        self.play(
            z_tracker.animate.set_value(complex(0.5, -2.3)),
            frame.animate.set_y(0),
            run_time=3
        )
        self.play(FlashAround(z_label[0]))
        self.play(
            w_plane.animate.scale(2, about_point=w_plane.get_left()),
            droste_image.animate.scale(2, about_point=w_plane.get_left()),
            run_time=2
        )
        self.wait()
        self.play(TransformFromCopy(z_dot, w_dot, suspend_mobject_updating=True))
        self.play(FlashAround(exp_z_label))
        self.wait()

        # Show the pi creatures
        droste_randy = SVGMobject("EscherPiCreature")
        droste_randy.set_z_index(3)
        droste_randy.set_fill(border_width=1)
        droste_randy.set_height(1.9)
        droste_randy.move_to(w_plane.c2p(-1.24, -1.34))

        log_randy = droste_randy.copy()
        log_randy.apply_points_function(lambda ps: np.array([
            z_plane.n2p(np.log(w_plane.p2n(p)))
            for p in ps
        ]), about_point=ORIGIN)
        z_unit = z_plane.get_unit_size()
        log_randys = VGroup(
            log_randy,
            log_randy.copy().shift(TAU * z_unit * UP),
            log_randy.copy().shift(2 * TAU * z_unit * UP),
        )
        brace_group = VGroup(brace, brace_label)

        log_randy_rect = SurroundingRectangle(log_randys[0]).set_stroke(YELLOW, 3)

        self.play(
            FadeOut(z_label),
            FadeOut(exp_z_label),
            FadeOut(z_dot),
            FadeOut(w_dot),
            v_line.animate.set_stroke(width=1),
            pink_circle.animate.set_stroke(width=1),
            log_image_tiles.animate.set_opacity(0.25),
            droste_image.animate.set_opacity(0.25),
            w_plane.axes.animate.set_stroke(WHITE),
            w_plane.coordinate_labels.animate.set_fill(WHITE),
            FadeIn(log_randy),
            FadeIn(droste_randy),
        )
        self.play(FadeIn(log_randy_rect, scale=0.25, run_time=1, rate_func=rush_into))
        self.play(FlashAround(log_randys[0], run_time=2, stroke_width=6))
        self.wait()
        self.play(
            TransformFromCopy(log_randy, log_randys[1]),
            log_randy_rect.animate.surround(log_randys[1]),
            brace_group.animate.set_y(log_randy.get_y(), DOWN),
        )
        self.play(FlashAround(log_randys[1], run_time=2, stroke_width=6))
        self.play(
            TransformFromCopy(log_randys[1], log_randys[2]),
            log_randy_rect.animate.surround(log_randys[2]),
        )
        self.play(FlashAround(log_randys[2], run_time=2, stroke_width=6))
        self.wait()
        self.play(FadeOut(log_randy_rect))
        self.play(
            LaggedStart(
                (TransformFromCopy(lr, droste_randy)
                for lr in reversed(log_randys)),
                lag_ratio=0.25,
                run_time=4
            )
        )
        self.wait()

        # Show a band of values
        log_image_row = log_image_tiles.copy()
        log_image_row.set_opacity(1)
        limit_box = Rectangle(20, TAU)
        z_unit = z_plane.get_unit_size()
        limit_box.scale(z_unit)
        limit_box.move_to(z_plane.n2p(3), DR)
        limit_box.set_stroke(width=0)
        log_image_row.always.clip_to_box(limit_box)

        self.play(FadeIn(log_image_row))
        self.wait()
        self.play(
            limit_box.animate.shift(z_unit * PI * DOWN),
            run_time=5,
            rate_func=there_and_back
        )
        self.play(limit_box.animate.scale(5), run_time=6)
        self.remove(limit_box)
        self.play(FadeOut(log_image_row))
        self.wait()

        # Droste randy to log randys
        long_log_randys = VGroup(
            *log_randys,
            log_randy.copy().shift(z_unit * TAU * DOWN),
            log_randy.copy().shift(2 * z_unit * TAU * DOWN),
        )
        long_log_randys.save_state()
        w_dot.set_z_index(3)
        z_dot.set_z_index(3)
        circle = Circle(radius=get_norm(w_dot.get_center() - w_plane.get_origin()))
        circle.move_to(w_plane.get_origin())
        circle.set_stroke(PINK, 2)
        circle.rotate(np.log(get_w()).imag)
        z_tracker.save_state()

        self.play(
            FadeIn(w_dot),
            FadeIn(z_dot),
            FlashAround(w_dot),
        )
        for n in range(2):
            added_anims = [ShowCreation(circle)] if n == 0 else []
            self.play(
                z_tracker.animate.increment_value(TAU * 1j),
                Rotate(droste_randy, TAU, about_point=w_plane.get_origin()),
                long_log_randys.animate.shift(z_unit * TAU * UP),
                *added_anims,
                run_time=3,
            )
            long_log_randys.restore()
        self.play(FadeOut(z_dot), FadeOut(w_dot), FadeOut(circle))
        z_tracker.restore()

        # Comment on repetition to the left
        left_rep_arrow = Vector(4 * LEFT, thickness=8, fill_color=RED)
        left_rep_arrow.next_to(log_randys[1], UL)

        self.play(
            droste_image.animate.set_opacity(0.75),
            log_image_tiles.animate.set_opacity(0.75),
        )
        self.play(GrowArrow(left_rep_arrow))
        self.wait()
        self.play(FadeOut(left_rep_arrow))

        # Shift left
        small_droste_randy = droste_randy.copy()
        z_dot.set_opacity(0)
        w_dot.set_opacity(0)
        self.add(z_dot, w_dot)
        self.remove(droste_randy)
        self.play(
            log_randys.animate.shift(math.log(16) * z_plane.get_unit_size() * LEFT),
            UpdateFromFunc(small_droste_randy, lambda m: m.become(apply_func_to_z_space(log_randy.copy(), np.exp))),
            z_tracker.animate.increment_value(-math.log(16)),
            FadeOut(brace_group, time_span=(0, 1)),
            frame.animate.reorient(0, 0, 0, (0, 0.75, 0.0), 11),
            run_time=10
        )
        self.wait()
        self.play(
            frame.animate.reorient(0, 0, 0, (6.59, -0.04, 0.0), 0.38).set_anim_args(
                rate_func=lambda t: there_and_back_with_pause(t, 1 / 7),
                run_time=7
            ),
            FadeOut(left_arrow)
        )
        self.play(
            droste_image.animate.set_opacity(0.25),
            log_image_tiles.animate.set_opacity(0.25),
        )

        # Show exp and log properties
        func_labels = VGroup(right_arrow_label, right_arrow, left_arrow, left_arrow_label)
        t2c = {"z_1": BLUE, "z_2": BLUE_D, "w_1": PINK, "w_2": MAROON_B}
        exp_rule = Tex(R"e^{z_1 + z_2} = e^{z_1} e^{z_2}", t2c=t2c)
        log_rule = Tex(R"\ln(w_1 w_2) = \ln(w_1) + \ln(w_2)", t2c=t2c, font_size=40)
        rules = VGroup(exp_rule, log_rule)
        rules.scale(1.25)
        rules.arrange(DOWN, buff=LARGE_BUFF)
        rules.set_x(-0.1).set_y(-1.2)

        rect = SurroundingRectangle(exp_rule["z_1 + z_2"], buff=0.1)
        rect.set_stroke(YELLOW, 2)

        self.play(
            func_labels.animate.arrange(DOWN).align_to(w_plane, UP),
            Write(exp_rule, time_span=(0.5, 1.5)),
            Write(log_rule, time_span=(1.0, 2.0)),
        )
        self.play(ShowCreation(rect))
        self.wait()
        self.play(rect.animate.surround(exp_rule[R"e^{z_1} e^{z_2}"]))
        self.wait()
        self.play(rect.animate.surround(log_rule[R"w_1 w_2"]))
        self.wait()
        self.play(rect.animate.surround(log_rule[R"\ln(w_1) + \ln(w_2)"]))
        self.wait()
        self.play(FadeOut(rect))
        self.wait()

        # Show example point w -> 16w
        w_dot = Dot()
        w_dot.set_z_index(4)
        w_dot.set_color(RED)
        w_dot.move_to(small_droste_randy[10].get_corner(DL))
        w = w_plane.p2n(w_dot.get_center())

        w16_dot = w_dot.copy()
        w16_dot.move_to(w_plane.n2p(16 * w))

        w_label = Tex(R"w")
        w_label.set_color(RED)
        w_label.set_backstroke(BLACK, 2)
        w_label.next_to(w_dot, DL, buff=0, aligned_edge=UP)

        w16_label = Tex(R"w \cdot 16")
        w16_label.match_style(w_label)
        w16_label.next_to(w16_dot, UL, buff=0)

        scale_arrow = Arrow(w_label, w16_label, buff=0.1, path_arc=45 * DEG, thickness=5)
        scale_arrow.set_fill(RED)
        times_16_label = Tex(R"\times 16")
        times_16_label.next_to(scale_arrow.pfp(0.8), UL, SMALL_BUFF)

        w_label.save_state()
        w_dot.save_state()
        w_dot.scale(1 / 8)
        w_label.scale(1 / 4)
        w_label.next_to(w_dot, DL, buff=0.01)

        self.play(FadeOut(pink_circle))
        self.play(
            frame.animate.reorient(0, 0, 0, (6.08, -0.13, 0.0), 1.37),
            FadeIn(w_dot, 0.1 * UR, time_span=(1, 2)),
            FadeIn(w_label, 0.1 * UR, time_span=(1, 2)),
            FadeOut(v_line),
            FadeOut(rules),
            run_time=2,
        )
        self.wait()
        w_plane.coordinate_labels[1].set_opacity(0)
        self.play(
            frame.animate.reorient(0, 0, 0, (0.7, 0.69, 0.0), 11.37),
            Restore(w_dot),
            Restore(w_label),
            TransformFromCopy(w_dot, w16_dot),
            TransformMatchingTex(w_label.copy(), w16_label),
            TransformFromCopy(small_droste_randy, droste_randy),
            GrowArrow(scale_arrow),
            FadeIn(times_16_label, time_span=(1, 2), shift=0.25 * scale_arrow.get_vector()),
            run_time=3,
        )
        self.wait()

        # Show corresponding log(w) -> log(w) + log(16)
        log_w_dot = Dot()
        log_w_dot.match_style(w_dot)
        log_w_dot.move_to(z_plane.n2p(np.log(w) + TAU * 1j))
        log_w_label = Tex(R"\ln(w)")
        log_w_label.next_to(log_w_dot, DOWN, SMALL_BUFF)
        log_w_label.shift(SMALL_BUFF * LEFT)
        log_w_label.match_style(w_label)

        shift_vect = math.log(16) * z_plane.get_unit_size() * RIGHT
        shifted_log_dot = log_w_dot.copy().shift(shift_vect)
        shifted_log_randys = log_randys.copy().shift(shift_vect)
        shift_log_label = Tex(R"\ln(w) + \ln(16)")
        shift_log_label[R"\ln(w)"].set_fill(RED)
        shift_log_label.set_backstroke(BLACK, 3)
        shift_log_label.move_to(log_w_label, LEFT).shift(shift_vect)

        shift_arrow = Vector(shift_vect, thickness=5)
        shift_arrow.next_to(VGroup(log_randys[1], shifted_log_randys[1]), UP, SMALL_BUFF)
        shift_arrow_label = Tex(R"+ \ln(16)")
        shift_arrow_label.next_to(shift_arrow, UP, buff=0)
        shift_arrow_label.shift(3 * SMALL_BUFF * LEFT)
        shift_arrow_label.set_backstroke(BLACK, 3)

        self.play(
            TransformFromCopy(w_dot, log_w_dot, path_arc=-15 * DEG),
            TransformMatchingTex(w_label.copy(), log_w_label, path_arc=-15 * DEG),
            run_time=2,
        )
        self.wait()
        self.play(
            TransformFromCopy(log_w_dot, shifted_log_dot),
            TransformFromCopy(log_randys, shifted_log_randys),
            FadeTransform(log_w_label.copy(), shift_log_label[R"\ln(w)"][0]),
            FadeIn(shift_log_label[R"+ \ln(16)"][0], shift=shift_vect),
            TransformFromCopy(small_droste_randy, droste_randy),
            TransformFromCopy(w_dot, w16_dot),
            GrowArrow(shift_arrow, time_span=(1, 2)),
            FadeIn(shift_arrow_label, shift=0.25 * RIGHT, time_span=(1, 2)),
            run_time=3
        )
        self.wait()

        # Show tile and annulus
        top_brace = Brace(log_image, UP, SMALL_BUFF)
        shift_arrow_label.target = shift_arrow_label.generate_target()
        shift_arrow_label.target[0].scale(0, about_point=shift_arrow_label[1].get_left())
        shift_arrow_label.target.next_to(log_image, UP)
        shift_arrow_label.target.set_backstroke(BLACK, 5)
        shift_arrow_label.target.next_to(top_brace, UP, SMALL_BUFF)
        log_image.set_opacity(1)
        brace_group.next_to(log_image, RIGHT, SMALL_BUFF)

        fundamental_rect = SurroundingRectangle(log_image, buff=0)
        fundamental_rect.set_stroke(WHITE, 1)

        annulus = apply_func_to_z_space(log_image.copy(), np.exp)
        annulus.save_state()
        log_image_group = Group(log_image, top_brace, brace_group, shift_arrow_label)

        self.play(
            FadeOut(VGroup(w_dot, w_label, w16_dot, w16_label, scale_arrow, times_16_label)),
            FadeOut(VGroup(log_w_dot, log_w_label, shifted_log_dot, shift_log_label, shift_arrow)),
            FadeOut(VGroup(small_droste_randy, droste_randy, log_randys, shifted_log_randys)),
            MoveToTarget(shift_arrow_label),
            GrowFromCenter(top_brace),
            FadeIn(brace_group),
            FadeOut(big_z_plane.coordinate_labels[-15:]),
            FadeIn(log_image),
            run_time=2
        )
        self.play(ShowCreation(fundamental_rect))
        self.wait()
        self.play(
            ShowCreation(log_image),
            ShowCreation(annulus),
            w_plane.coordinate_labels.animate.set_opacity(0),
            w_plane.axes.animate.set_stroke(WHITE, 1, 0.25),
            run_time=3
        )
        self.wait()
        log_image_group.save_state()
        for n in range(3):
            self.play(
                log_image_group.animate.shift(-shift_vect),
                annulus.animate.scale(1 / 16, about_point=w_plane.get_origin()),
                frame.animate.reorient(0, 0, 0, (0.67, 0.14, 0.0), 13.59),
                run_time=3,
            )
            self.wait()
        self.play(
            Restore(log_image_group),
            Restore(annulus),
            frame.animate.reorient(0, 0, 0, (1.72, 0.49, 0.0), 13.42),
            run_time=2
        )

        # Show the boundary
        colors = color_gradient([RED, YELLOW], 4)
        v_lines = VGroup(
            Line(
                log_image.get_corner(DR),
                log_image.get_corner(UR)
            ).shift(-0.25 * n * shift_vect)
            for n in range(20)
        )
        v_lines.set_stroke(WHITE, 3)
        v_lines.set_submobject_colors_by_gradient(RED, YELLOW, GREEN, BLUE, interp_by_hsl=True)

        circles = apply_func_to_z_space(v_lines.copy().insert_n_curves(100), np.exp)
        indic_arrow = Vector(DL, thickness=6).set_fill(RED)
        indic_arrow.next_to(v_lines.get_corner(DR), UR, buff=SMALL_BUFF)

        log_image_tiles.set_clip_plane(LEFT, 0)
        v_lines.set_clip_plane(LEFT, 0)
        droste_image.set_clip_plane(RIGHT, 0)
        circles.set_clip_plane(RIGHT, 0)

        func_labels.set_backstroke(BLACK, 5)
        func_labels.set_z_index(5)
        self.add(func_labels)

        self.play(
            FadeOut(VGroup(top_brace, shift_arrow_label, brace_group)),
            log_image_tiles.animate.set_opacity(0.9),
            droste_image.animate.set_opacity(0.9),
            FadeOut(Group(log_image, annulus), time_span=(0.9, 1)),
            FadeOut(fundamental_rect),
        )
        self.play(
            indic_arrow.animate.shift(v_lines[0].get_vector()),
            ShowCreation(v_lines[0]),
            ShowCreation(circles[0]),
            run_time=3
        )
        self.wait()
        self.play(
            LaggedStartMap(ShowCreation, v_lines[1:], lag_ratio=0.1, run_time=2),
            LaggedStartMap(ShowCreation, circles[1:], lag_ratio=0.1, run_time=2),
        )
        self.wait()
        self.play(FadeOut(indic_arrow))

        # Expand
        def update_droste(mob):
            mob.set_width(2 * w_plane.get_unit_size() * np.exp(z_plane.p2n(log_image_tiles.get_right()).real))
            mob.move_to(w_plane.get_origin())
            return mob

        self.play(
            Group(log_image_tiles, v_lines).animate.shift(4 * RIGHT),
            UpdateFromFunc(Group(droste_image, circles), update_droste),
            run_time=10
        )
        self.play(FadeOut(v_lines), FadeOut(circles))
        self.wait()
        self.play(
            log_image_tiles.animate.shift(TAU * z_plane.get_unit_size() * UP),
            Rotate(droste_image, TAU, about_point=w_plane.get_origin()),
            run_time=6
        )
        log_image_tiles.shift(shift_vect * RIGHT)
        self.play(
            log_image_tiles.animate.shift(shift_vect * LEFT),
            UpdateFromFunc(droste_image, update_droste),
            run_time=6
        )
        self.wait()

    def get_traced_paths(self, dots, colors=None, stroke_width=3):
        if colors is None:
            colors = [dot.family_members_with_points()[0].get_color() for dot in dots]
        return VGroup(
            TracedPath(dot.get_center, stroke_color=color, stroke_width=stroke_width)
            for dot, color in zip(dots, colors)
        )


class CreatingTheSpiral(InteractiveScene):
    # log_image_resolution = (21, 51)
    log_image_resolution = (51, 101)
    droste_scale_adjustment = 1.5
    droste_scale_factor = 16
    fixed_point = 4.0j
    droste_plane_x_range = (-1, 1)
    log_plane_x_range = (-5, 5)

    def setup(self):
        super().setup()

        # Planes
        self.planes = self.get_four_planes()
        self.dark_planes = self.get_four_dark_planes()
        a_plane, b_plane, c_plane, d_plane = self.planes

        # Arrows
        self.log_arrow_group, self.exp_arrow_group = self.get_log_exp_arrow_groups()

        # Droste image
        droste_image = get_droste_from_log_image_path(self.get_log_image_path(), self.droste_scale_factor)
        droste_image.set_height(2 * a_plane.get_unit_size() * self.droste_scale_adjustment)
        droste_image.move_to(a_plane)
        droste_image.clip_to_box(a_plane)
        self.droste_image = droste_image

        # Log image
        self.const = complex(0, TAU) / complex(math.log(self.droste_scale_factor), TAU)

        self.log_image_tiles = self.get_log_image_tiles(b_plane, 9, 9, resolution=(2, 2))
        # self.pre_transform_log = self.get_log_image_tiles(b_plane, 3, 3, resolution=self.log_image_resolution)
        self.pre_transform_log = self.get_log_image_tiles(b_plane, 5, 5, resolution=self.log_image_resolution)

        self.var_const = ComplexValueTracker(self.const)
        self.var_fixed_point = ComplexValueTracker(self.fixed_point)

        # Blank spot
        blank_dot = Dot()
        blank_dot.set_width(0.4)
        blank_dot.set_color(WHITE)
        blank_spot = Group(GlowDot(color=WHITE, radius=2 * blank_dot.get_width()), blank_dot)
        blank_spot.move_to(d_plane.get_center())
        self.blank_spot = blank_spot

    def construct(self):
        # Load up local variables
        planes = self.planes
        dark_planes = self.dark_planes
        a_plane, b_plane, c_plane, d_plane = planes
        dark_a_plane, dark_b_plane, dark_c_plane, dark_d_plane = dark_planes

        log_arrow_group = self.log_arrow_group
        exp_arrow_group = self.exp_arrow_group
        log_arrow, log_label = log_arrow_group
        exp_arrow, exp_label = exp_arrow_group

        droste_image = self.droste_image
        log_image_tiles = self.log_image_tiles
        blank_spot = self.blank_spot

        const = self.const
        fixed_point = self.fixed_point
        var_const = self.var_const
        var_fixed_point = self.var_fixed_point

        rot_func = self.rot_func
        get_rot_tiles = self.get_rot_tiles
        get_final_image = self.get_final_image

        # Show start and end goal
        frame = self.frame
        frame.set_field_of_view(1 * DEG)
        frame.match_x(a_plane)

        func_arrow = Arrow(a_plane.get_right(), d_plane.get_right(), path_arc=-120 * DEG, thickness=5)
        fz, eq, q_marks = question = VGroup(Tex(R"f(z)"), Tex(R"="), Tex(R"???"))
        question.arrange(RIGHT, SMALL_BUFF)
        question.next_to(func_arrow, RIGHT)

        var_const.set_value(1)
        final_image = get_final_image()
        droste_image.save_state()

        self.add(func_arrow, question)
        self.add(a_plane, d_plane)
        self.play(
            FadeIn(droste_image),
            FadeIn(dark_a_plane),
            FadeOut(a_plane),
        )
        self.play(droste_image.animate.scale(16), run_time=4)
        droste_image.restore()
        self.play(
            FadeIn(final_image),
            FadeIn(dark_d_plane),
            FadeOut(d_plane),
            FadeIn(blank_spot),
        )
        self.play(
            UpdateFromFunc(final_image, lambda m: m.become(get_final_image())),
            var_const.animate.set_value(const),
            run_time=5
        )
        self.wait()

        # Show the four steps
        var_const.set_value(1)
        rot_log_tiles = get_rot_tiles()

        mult_arrow = Arrow(b_plane.get_left(), c_plane.get_left(), path_arc=90 * DEG, thickness=5)
        kw = dict(t2c={"{z}": BLUE})
        rot_label = Text("Rotate \n & Scale", font_size=36)
        rot_label.next_to(mult_arrow, LEFT, SMALL_BUFF)
        mult_arrow_group = VGroup(mult_arrow, rot_label)

        self.play(
            ReplacementTransform(func_arrow, log_arrow, time_span=(0, 1)),
            Write(log_label),
            FadeIn(b_plane),
            frame.animate.reorient(0, 0, 0, (0.01, 1.96, 0.0), 5.04),
            FadeOut(question, time_span=(1, 2)),
            run_time=2,
        )
        self.play(
            FadeOut(droste_image.copy()),
            ShowCreation(droste_image, lag_ratio=0),
            ShowCreation(log_image_tiles, lag_ratio=0),
            FadeOut(b_plane),
            FadeIn(dark_b_plane),
            run_time=3
        )
        self.wait()
        self.play(
            FadeIn(rot_log_tiles),
            FadeIn(dark_c_plane),
            Write(mult_arrow),
            FadeIn(rot_label, 0.25 * DOWN),
            frame.animate.to_default_state(),
            run_time=2,
        )
        self.play(
            var_const.animate.set_value(const),
            UpdateFromFunc(rot_log_tiles, lambda m: m.become(get_rot_tiles())),
            run_time=4,
        )
        self.wait()

        # Final exp
        exp_mover = self.get_log_image_tiles(c_plane, 3, 3)
        exp_mover.set_opacity(0.5)
        apply_func_between_planes(exp_mover, rot_func, c_plane, c_plane)
        box = SurroundingRectangle(c_plane, buff=0)
        box.set_stroke(width=0)
        exp_mover.add_updater(lambda m: m.clip_to_box(box))
        self.play(
            GrowArrow(exp_arrow, time_span=(0, 1)),
            Write(exp_label, time_span=(0, 1)),
            Transform(
                exp_mover,
                apply_func_between_planes(exp_mover.copy(), np.exp, c_plane, d_plane),
                time_span=(1, 3)
            ),
            box.animate.move_to(d_plane).set_anim_args(time_span=(1, 3)),
            run_time=3,
        )
        self.remove(exp_mover)
        self.wait()

        # Show the line from big to small
        plane_covers = VGroup(
            BackgroundRectangle(plane, buff=0.05).set_fill(BLACK, 0.9)
            for plane in planes
        )

        randy = SVGMobject("EscherPiCreature")
        randy.set_backstroke(BLACK, 3)
        randy.set_height(0.9)
        randy.move_to(a_plane.c2p(-0.62, -0.95), DOWN)
        randy.set_z_index(2)

        randy_box = SurroundingRectangle(randy, buff=0.05)
        randy_box.set_stroke(YELLOW, 2)

        randy_group = VGroup(randy, randy_box)
        small_randy, small_box = small_group = randy_group.copy().scale(1 / 16, about_point=a_plane.get_origin())
        box_lines = VGroup(
            Line(
                randy_box.get_corner(corner),
                small_box.get_corner(corner)
            ).match_style(small_box)
            for corner in [UL, DR]
        )

        randy_line = Line(randy.get_center(), small_randy.get_center())
        randy_line.set_stroke([RED, RED_E], width=[10, 3])
        randy_line_dots = VGroup(
            Dot(randy_line.get_start(), radius=0.05).set_fill(RED),
            Dot(randy_line.get_end(), radius=0.01).set_fill(RED_E),
        )

        self.play(
            frame.animate.reorient(0, 0, 0, (2.52, 1.16, 0.0), 1.74),
            run_time=3,
        )
        self.play(
            FadeIn(randy),
            ShowCreation(randy_box),
            droste_image.animate.set_opacity(0.5),
        )
        self.wait()
        self.play(
            TransformFromCopy(randy_group, small_group),
            ShowCreation(box_lines, lag_ratio=0),
            run_time=2
        )
        self.wait()
        self.play(GrowFromCenter(randy_line_dots[0]))
        self.play(
            FadeOut(randy_box),
            FadeOut(small_box),
            FadeOut(box_lines),
            TransformFromCopy(*randy_line_dots),
            ShowCreation(randy_line),
        )
        self.wait()

        # Show the corresponding loop
        self.add(plane_covers[1:3])
        log_arrow_group.set_fill(opacity=0.2)
        exp_arrow_group.set_fill(opacity=0.2)
        mult_arrow_group.set_fill(opacity=0.2)

        final_randy = apply_func_between_planes(
            randy.copy(), lambda z: np.exp(rot_func(np.log(z) + TAU * 1j)), a_plane, d_plane,
        )
        loop = Circle(radius=0.9 * d_plane.get_unit_size())
        loop.flip(RIGHT).rotate(225 * DEG)
        loop.move_to(d_plane)
        loop.set_stroke([RED, RED_E], 8)

        self.remove(randy_line, randy_line_dots)
        self.play(
            frame.animate.reorient(0, 0, 0, (4.43, -0.86, 0.0), 6.44),
            final_image.animate.set_opacity(0.1),
            TransformFromCopy(randy, final_randy),
            Animation(randy_line),
            Animation(randy_line_dots),
            run_time=2
        )
        self.play(
            TransformFromCopy(randy_line, loop, run_time=3, path_arc=[0, -0.5 * PI]),
            TransformFromCopy(small_randy, final_randy, run_time=3, path_arc=-0.5 * PI)
        )
        self.wait()
        self.play(
            ShowCreation(loop),
            ShowCreation(randy_line),
            run_time=2
        )
        self.wait()
        self.play(
            TransformFromCopy(loop, randy_line),
            TransformFromCopy(final_randy, randy),
            TransformFromCopy(final_randy, small_randy),
            rate_func=there_and_back_with_pause,
            run_time=6
        )
        self.add(randy_line)
        self.wait()

        # Show randy in log image
        b_unit = b_plane.get_unit_size()
        log_randy = apply_func_between_planes(randy.copy(), np.log, a_plane, b_plane)
        log_randys = VGroup(
            log_randy.copy().shift(n * b_unit * TAU * UP)
            for n in range(5)
        )
        log_randys.set_backstroke(BLACK, 2)
        small_log_randys = log_randys.copy().shift(b_unit * math.log(16) * LEFT)

        big_box = SurroundingRectangle(log_randys[:2])
        small_box = SurroundingRectangle(small_log_randys[:2])
        VGroup(big_box, small_box).set_stroke(BLUE_B, 2)
        big_box_label = Text("Big", font_size=24).next_to(big_box, UP, SMALL_BUFF)
        small_box_label = Text("Small", font_size=24).next_to(small_box, UP, SMALL_BUFF)
        big_box_label.align_to(small_box_label, UP)

        self.play(
            frame.animate.reorient(0, 0, 0, (-0.32, 1.82, 0.0), 4.54),
            log_arrow_group.animate.set_opacity(1),
            FadeOut(plane_covers[1]),
            log_image_tiles.animate.set_opacity(0.5),
            run_time=2,
        )
        self.play(
            TransformFromCopy(randy.replicate(len(log_randys)), log_randys),
            log_image_tiles.animate.set_opacity(0.25),
            run_time=2
        )
        self.play(
            ShowCreation(big_box),
            Write(big_box_label)
        )
        self.wait()
        self.play(
            TransformFromCopy(randy, small_randy),
            TransformFromCopy(log_randys, small_log_randys),
            TransformFromCopy(big_box, small_box),
            run_time=2
        )
        self.play(
            Write(small_box_label),
        )
        self.wait()

        log_randys = log_randys[:3]
        small_log_randys = small_log_randys[:3]
        old_randy_line = randy_line
        self.remove(*log_randys[2:])
        self.remove(*small_log_randys[2:])

        # Show the log lines
        log_lines = VGroup(
            Line(br.get_center(), sr.get_center()).set_stroke(RED, 3)
            for br, sr in zip(log_randys, small_log_randys)
        )
        old_log_lines = log_lines.copy()
        randy_line_style = randy_line.get_style()

        def get_true_randy_line():
            result = log_lines[1].copy()
            if result.get_num_points() < 100:
                result.insert_n_curves(100)
            apply_func_between_planes(result, np.exp, b_plane, a_plane)
            result.set_style(**randy_line_style)
            return result

        self.remove(old_randy_line, randy_line_dots)
        randy_line = get_true_randy_line()

        randy_line_indicator = GlowDot(randy_line.get_start(), color=RED, radius=0.25)
        log_indicators = Group(
            GlowDot(color=RED).move_to(lr) for lr in log_randys
        )
        line_indicators = Group(randy_line_indicator, log_indicators)

        self.play(
            UpdateFromFunc(randy_line_indicator, lambda m: m.move_to(randy_line.get_end())),
            log_indicators.animate.move_to(log_lines, LEFT),
            ShowCreation(log_lines, lag_ratio=0),
            ShowCreation(randy_line, lag_ratio=0),
            run_time=2
        )
        self.play(
            FadeOut(line_indicators),
        )
        self.wait()

        # Diagonal line
        diag_line = Line(log_randys[1].get_center(), small_log_randys[0].get_center())
        diag_line.match_style(log_lines[0])
        v_shift = b_unit * TAU * UP
        diag_lines = VGroup(
            diag_line.copy().shift(-v_shift),
            diag_line,
            diag_line.copy().shift(v_shift)
        )
        for lines in log_lines, diag_lines:
            lines.insert_n_curves(100)
            lines.clip_to_box(b_plane)

        test_line = apply_func_between_planes(log_lines[1].copy(), np.exp, b_plane, a_plane)

        self.play(
            ReplacementTransform(log_lines, diag_lines, run_time=3),
            UpdateFromFunc(randy_line, lambda m: m.match_points(get_true_randy_line())),
        )
        self.wait()
        self.play(big_box.animate.surround(log_randys[1]))
        self.wait()
        self.play(
            small_box.animate.surround(small_log_randys[0]),
            small_box_label.animate.next_to(small_log_randys[0], DOWN, buff=0.2),
        )
        self.wait()

        # Move along curled line
        log_randy = log_randys[1].copy()

        def get_exp_log_randy():
            return apply_func_between_planes(log_randy.copy(), np.exp, b_plane, a_plane)

        exp_log_randy = get_exp_log_randy()

        self.play(FadeIn(log_randy), FadeIn(exp_log_randy))
        self.play(
            log_randy.animate.move_to(small_log_randys[0]),
            UpdateFromFunc(exp_log_randy, lambda m: m.become(get_exp_log_randy())),
            run_time=3
        )
        self.play(FadeOut(log_randy), FadeOut(exp_log_randy))
        self.wait()

        # Zoom out to describe the goal
        plane_rect = SurroundingRectangle(d_plane)
        plane_rect.set_stroke(YELLOW, 4)

        v_line = apply_func_between_planes(diag_line.copy(), rot_func, b_plane, c_plane)
        v_line.deactivate_clip_plane()
        v_line.set_stroke(RED, 5)

        v_line_brace = Brace(v_line, RIGHT, SMALL_BUFF)
        v_line_label = v_line_brace.get_tex(R"2\pi")
        v_line_brace_group = VGroup(v_line_brace, v_line_label)
        v_line_brace_group.set_backstroke(BLACK, 5)

        rot_log_randys = VGroup(log_randys[1].copy(), small_log_randys[0].copy())
        apply_func_between_planes(rot_log_randys, rot_func, b_plane, c_plane)

        diag_arrow, down_arrow = red_arrows = VGroup(
            Arrow(line.get_start(), line.get_end(), buff=0, thickness=4, fill_color=RED)
            for line in [diag_line, v_line]
        )

        self.play(frame.animate.to_default_state(), run_time=2)
        self.play(ShowCreation(plane_rect))
        self.wait()
        self.play(
            plane_rect.animate.surround(c_plane),
            TransformFromCopy(loop, v_line),
            TransformFromCopy(final_randy.replicate(2), rot_log_randys),
            exp_arrow_group.animate.set_fill(opacity=1),
            FadeOut(plane_covers[2]),
            rot_log_tiles.animate.set_opacity(0.5),
            run_time=2
        )
        self.play(
            GrowFromCenter(v_line_brace),
            Write(v_line_label, stroke_color=WHITE),
        )
        self.wait()
        self.play(
            plane_rect.animate.surround(b_plane).stretch(1.05, 1, about_edge=DOWN),
            FadeOut(diag_lines),
            FadeIn(diag_arrow),
            FadeOut(VGroup(log_randys[0], small_log_randys[1])),
            log_image_tiles.animate.set_opacity(0.5),
        )
        self.wait()
        self.play(
            plane_rect.animate.surround(VGroup(b_plane, c_plane)).stretch(1.02, 1, about_edge=DOWN),
            TransformFromCopy(diag_arrow, down_arrow),
            mult_arrow_group.animate.set_fill(opacity=1),
            run_time=2,
        )
        self.wait()

        # Talk about complex constant
        kw = dict(t2c={"{z}": BLUE, "{c}": YELLOW})
        func_label = VGroup(
            Tex(R"{z}", **kw),
            Tex(R"\downarrow"),
            Tex(R"{c} \cdot {z}", **kw),
        )
        func_label.arrange(DOWN)
        func_label.next_to(mult_arrow, LEFT)
        true_output = Tex(R"{c} \cdot ({z} - z_0) + z_0", **kw)
        true_output.move_to(func_label[2], RIGHT)

        pivot_dot = Dot(b_plane.n2p(fixed_point)).set_color(TEAL)
        pivot_dot.set_stroke(WHITE, 1)
        pivot_dot_label = Tex(R"z_0")
        pivot_dot_label.set_backstroke(BLACK, 3)
        pivot_dot_label.next_to(pivot_dot, RIGHT, SMALL_BUFF)

        self.play(
            frame.animate.set_x(-2),
            FadeOut(rot_label, 0.5 * DOWN),
            FadeIn(func_label, 0.5 * DOWN),
        )
        self.wait()
        self.play(plane_rect.animate.surround(randy))
        self.wait()
        self.play(plane_rect.animate.surround(log_randys[1]))
        self.wait()
        self.play(
            FadeOut(VGroup(log_randys[1], small_log_randys[0], big_box, big_box_label, small_box, small_box_label, plane_rect)),
            diag_arrow.animate.put_start_and_end_on(pivot_dot.get_center(), diag_arrow.get_end()),
            FadeIn(pivot_dot),
        )
        self.play(
            log_image_tiles.animate.rotate(70 * DEG, about_point=diag_arrow.get_start()),
            diag_arrow.animate.rotate(70 * DEG, about_point=diag_arrow.get_start()),
            rate_func=there_and_back,
            run_time=4
        )
        self.play(Write(pivot_dot_label, stroke_color=WHITE))
        self.wait()
        self.play(
            TransformMatchingTex(func_label[2], true_output),
            TransformFromCopy(pivot_dot_label.replicate(2), true_output["z_0"]),
            func_label[:2].animate.match_x(true_output),
        )
        self.wait()

        func_label = VGroup(*func_label[:2], true_output)
        self.add(func_label)

        # Write the value for c
        c_value = Tex(R"{c} = \frac{2\pi i}{\ln(16) + 2 \pi i}", **kw)
        c_value.next_to(func_label, DOWN)

        log16_line = DashedLine(small_log_randys[0].get_center(), log_randys[0].get_center())
        two_pi_i_line = DashedLine(log_randys[0].get_center(), log_randys[1].get_center())
        two_pi_i_line.shift(pivot_dot.get_center() - log_randys[1].get_center())
        two_pi_i_line.set_z_index(3)

        log_16_label = Tex(R"\ln(16)", font_size=24)
        log_16_label.next_to(log16_line, DOWN, SMALL_BUFF)
        two_pi_i_label = Tex(R"2\pi i", font_size=24)
        two_pi_i_label.next_to(two_pi_i_line, RIGHT, SMALL_BUFF)
        VGroup(two_pi_i_label, log_16_label).set_backstroke(BLACK, 3)

        self.play(
            ShowCreation(log16_line),
            FadeIn(log_16_label),
        )
        self.play(
            ShowCreation(two_pi_i_line),
            FadeIn(two_pi_i_label),
        )
        self.wait()
        self.play(
            func_label.animate.shift(0.7 * UP),
            FadeIn(c_value, 0.5 * DOWN),
        )
        self.wait()

        # Clear the board
        faders = VGroup(
            randy, small_randy, randy_line, # pivot_dot, pivot_dot_label,
            log16_line, two_pi_i_line, log_16_label, two_pi_i_label,
            rot_log_randys,
            plane_rect, diag_arrow, v_line, down_arrow, v_line_brace_group,
            loop, final_randy,
        )

        self.remove(blank_spot)
        self.play(
            LaggedStartMap(FadeOut, faders, lag_ratio=0.1),
            droste_image.animate.set_opacity(1).set_anim_args(time_span=(0, 1)),
            log_image_tiles.animate.set_opacity(1).set_anim_args(time_span=(0.25, 1.25)),
            rot_log_tiles.animate.set_opacity(1).set_anim_args(time_span=(0.5, 1.5)),
            final_image.animate.set_opacity(1).set_anim_args(time_span=(0.75, 1.75)),
            FadeIn(blank_spot),
            run_time=2,
        )
        self.wait()

        # Show the factor c on its own plane
        mult_plane, c_group = self.get_mult_plane(c_plane)
        mult_plane.match_x(func_label)
        c_dot, c_vect, c_label = c_group

        self.play(
            FadeOut(c_value, DOWN),
            FadeIn(mult_plane, DOWN),
            FadeIn(c_group, DOWN),
            func_label.animate.shift(0.25 * UP)
        )

        # Play
        var_const.clear_updaters().add_updater(lambda m: m.set_value(mult_plane.p2n(c_dot.get_center())))
        rot_log_tiles.clear_updaters().add_updater(lambda m: m.become(get_rot_tiles()))
        final_image.clear_updaters().add_updater(lambda m: m.become(get_final_image()))
        for mob in self.mobjects:
            self.disable_interaction(mob)
        self.enable_interaction(c_dot)

        self.play(Rotating(c_dot, about_point=c_dot.get_center() + 0.25 * DOWN, run_time=10))
        self.wait()  # TODO, actually interact with this over a longer period
        self.play(c_dot.animate.move_to(mult_plane.n2p(const)), run_time=2)

        rot_log_tiles.clear_updaters()
        final_image.clear_updaters()

        # Comment on the center dot
        plane_rect.surround(d_plane)
        plane_covers.set_fill(opacity=0.5)

        self.pre_transform_log.save_state()
        final_image.save_state()
        rot_log_tiles.save_state()

        self.pre_transform_log.become(self.get_log_image_tiles(b_plane, 9, 9, resolution=self.log_image_resolution))
        self.log_image_tiles.become(self.get_log_image_tiles(b_plane, 15, 15, resolution=(2, 2)))
        rot_log_tiles.become(self.get_rot_tiles())
        final_image.become(get_final_image())

        self.play(
            ShowCreation(plane_rect),
            droste_image.animate.set_opacity(0.5),
            FadeIn(plane_covers[:3]),
        )
        self.wait()
        self.play(
            plane_rect.animate.surround(blank_spot[1]),
            frame.animate.reorient(0, 0, 0, (2.21, -1.99, 0.0), 3.48).set_anim_args(run_time=2),
        )
        self.wait()
        self.play(FadeOut(blank_spot))
        self.wait()
        self.play(
            FadeOut(plane_rect),
            FadeOut(plane_covers[2]),
            frame.animate.reorient(0, 0, 0, (-0.07, -1.85, 0.0), 4.89),
        )
        self.wait()
        self.play(
            rot_log_tiles.animate.shift(c_plane.n2p(4 * const * math.log(16)) - c_plane.get_origin()),
            self.pre_transform_log.animate.shift(4 * b_unit * math.log(16) * RIGHT),
            UpdateFromFunc(
                final_image,
                lambda m: m.become(get_final_image())
            ),
            run_time=12,
        )
        self.wait()

        self.pre_transform_log.restore()
        final_image.restore()
        rot_log_tiles.restore()

        # Zoom back out
        self.play(
            droste_image.animate.set_opacity(1),
            FadeOut(plane_covers[:2]),
            frame.animate.to_default_state().set_x(-2),
            run_time=2
        )
        self.wait()

        # Trying to use horizontal line
        self.play(
            ShowCreation(old_randy_line),
            ShowCreation(old_log_lines),
        )
        self.wait()

        var_const.clear_updaters().add_updater(lambda m: m.set_value(mult_plane.p2n(c_dot.get_center())))
        rot_log_tiles.clear_updaters().add_updater(lambda m: m.become(get_rot_tiles()))
        final_image.clear_updaters().add_updater(lambda m: m.become(get_final_image()))

        self.play(
            c_dot.animate.move_to(mult_plane.n2p(complex(0, TAU / math.log(16)))),
            run_time=3
        )
        rot_log_tiles.clear_updaters()
        final_image.clear_updaters()
        for factor, time in [(0.5, 2), (5e-7, 8)]:
            self.play(
                final_image.animate.scale(factor, about_point=d_plane.get_origin()),
                rot_log_tiles.animate.shift(b_unit * math.log(factor) * RIGHT),
                run_time=time
            )
        self.wait()

    def get_log_image_path(self):
        return get_pi_house_log_image_path()

    def get_four_planes(
        self,
        axes_color=WHITE,
        line_color=BLUE,
        line_width=1,
        line_opacity=1,
        faded_line_ratio=4,
        faded_line_opacity=0.25,
        h_buff=2.0,
        v_buff=0.75,
        include_coordinates=True
    ):
        kw = dict(faded_line_ratio=faded_line_ratio)
        planes = VGroup(
            ComplexPlane(self.droste_plane_x_range, self.droste_plane_x_range, **kw),
            ComplexPlane(self.log_plane_x_range, self.log_plane_x_range, **kw),
            ComplexPlane(self.log_plane_x_range, self.log_plane_x_range, **kw),
            ComplexPlane(self.droste_plane_x_range, self.droste_plane_x_range, **kw),
        )
        for plane in planes:
            plane.set_height(3.25)
            plane.axes.set_stroke(axes_color)
            plane.background_lines.set_stroke(line_color, line_width, line_opacity)
            plane.faded_lines.set_stroke(line_color, line_width, faded_line_opacity)
        planes.arrange_in_grid(2, 2, h_buff=h_buff, v_buff=v_buff)
        planes[0].match_x(planes[3])
        planes[1].match_x(planes[2])

        if include_coordinates:
            for plane in planes:
                plane.add_coordinate_labels(font_size=12, buff=0.05)

        return planes

    def get_four_dark_planes(self):
        result = self.get_four_planes(
            axes_color=BLACK,
            line_color=BLACK,
            line_opacity=0.25,
            faded_line_ratio=0,
            include_coordinates=False
        )
        result.set_z_index(1)
        return result

    def get_log_exp_arrow_groups(self):
        a_plane, b_plane, c_plane, d_plane = self.planes
        arrow_kw = dict(thickness=5, fill_color=GREY_B)
        tex_kw = dict(font_size=36, t2c={"z": BLUE, "w": PINK})
        log_arrow = Arrow(a_plane, b_plane, **arrow_kw)
        exp_arrow = Arrow(c_plane, d_plane, **arrow_kw)
        log_label = Tex(R"\ln(w) \leftarrow w", **tex_kw)
        log_label.next_to(log_arrow, UP, buff=SMALL_BUFF)
        exp_label = Tex(R"z \to e^z", **tex_kw)
        exp_label.next_to(exp_arrow, UP, buff=SMALL_BUFF)

        log_arrow_group = VGroup(log_arrow, log_label)
        exp_arrow_group = VGroup(exp_arrow, exp_label)

        return log_arrow_group, exp_arrow_group

    def get_log_image_tiles(self, plane, n_rows=None, n_cols=None, resolution=(21, 51)):
        log_image = get_log_image(self.get_log_image_path(), self.droste_scale_factor, resolution)
        log_image.scale(plane.get_unit_size())

        if n_cols is None:
            n_cols = int(np.ceil(plane.get_width() / log_image.get_width())) + 2
        if n_rows is None:
            n_rows = int(np.ceil(plane.get_height() / log_image.get_height())) + 2

        log_image_tiles = log_image.get_grid(n_rows, n_cols, buff=0)
        log_image_tiles.center()
        log_image_tiles.sort(lambda p: p[1])
        log_image_tiles.move_to(plane.n2p(math.log(self.droste_scale_adjustment)), DR)
        log_image_tiles.shift((n_rows // 2 + 1) * log_image.get_height() * DOWN)
        log_image_tiles.shift((n_cols // 2) * log_image.get_width() * RIGHT)
        log_image_tiles.clip_to_box(plane)
        return log_image_tiles

    def rot_func(self, z):
        z0 = self.var_fixed_point.get_value()
        c = self.var_const.get_value()
        return c * (z - z0) + z0

    def get_rot_tiles(self):
        b_plane, c_plane = self.planes[1:3]
        result = apply_func_between_planes(
            self.log_image_tiles.copy(), self.rot_func, b_plane, c_plane
        )
        result.clip_to_box(c_plane)
        return result

    def get_final_image(self):
        b_plane = self.planes[1]
        d_plane = self.planes[3]
        result = apply_func_between_planes(
            self.pre_transform_log.copy(), lambda z: np.exp(self.rot_func(z)), b_plane, d_plane
        )
        result.clip_to_box(d_plane)
        return result

    def get_mult_arrow_group(self):
        b_plane, c_plane = self.planes[1:3]
        mult_arrow = Arrow(
            b_plane.get_left(),
            c_plane.get_left(),
            path_arc=90 * DEG,
            thickness=5
        )
        kw = dict(t2c={"{z}": BLUE, "c": YELLOW})
        mult_label = VGroup(
            Tex(R"{z}", **kw),
            Tex(R"\downarrow", **kw),
            Tex(R"{c} \cdot ({z} - z_0) + z_0", **kw),
        )
        mult_label.arrange(DOWN)
        mult_label.next_to(mult_arrow, LEFT)

        return VGroup(mult_arrow, mult_label)

    def get_mult_plane(self, ref_plane, x_range=(-2, 2)):
        mult_plane = ComplexPlane(x_range, x_range)
        mult_plane.match_width(ref_plane)
        mult_plane.next_to(ref_plane, LEFT)

        c_dot = Group(GlowDot(), TrueDot()).set_color(YELLOW)
        c_dot.move_to(mult_plane.n2p(self.var_const.get_value()))
        c_vect = Vector(fill_color=YELLOW)
        c_label = Tex(R"c").set_color(YELLOW)
        c_label.always.next_to(c_dot, RIGHT, buff=0)
        c_vect.f_always.put_start_and_end_on(mult_plane.get_origin, c_dot.get_center)
        c_group = Group(c_dot, c_vect, c_label)

        return Group(mult_plane, c_group)


class FourPannelsWithPrintGallery(CreatingTheSpiral):
    log_image_resolution = (41, 101)
    droste_scale_factor = 256
    droste_scale_adjustment = 120
    fixed_point = 0.2 + 3.8j
    log_plane_x_range = (-7, 7)

    def construct(self):
        # Load up local variables
        planes = self.planes
        dark_planes = self.dark_planes
        a_plane, b_plane, c_plane, d_plane = planes
        dark_a_plane, dark_b_plane, dark_c_plane, dark_d_plane = dark_planes

        log_arrow_group = self.log_arrow_group
        exp_arrow_group = self.exp_arrow_group
        log_arrow, log_label = log_arrow_group
        exp_arrow, exp_label = exp_arrow_group

        droste_image = self.droste_image
        log_image_tiles = self.log_image_tiles
        blank_spot = self.blank_spot

        const = self.const
        fixed_point = self.fixed_point
        var_const = self.var_const
        var_fixed_point = self.var_fixed_point

        rot_func = self.rot_func
        get_rot_tiles = self.get_rot_tiles
        get_final_image = self.get_final_image

        # Edit dark planes
        for dark_plane in dark_planes:
            dark_plane.background_lines.set_stroke(BLUE_D, 2)
            dark_plane.axes.set_stroke(BLUE_D, 3)

        # Introduce droste_image
        frame = self.frame
        full_screen = FullScreenRectangle()
        droste_image.add(droste_image[-1].copy().scale(self.droste_scale_factor))
        droste_image.set_opacity(0.5)
        droste_image.save_state()
        droste_image.set_opacity(1)
        droste_image.clip_to_box(full_screen)
        droste_image.center()

        start_width = 1.5 * FRAME_WIDTH
        droste_image.set_width(self.droste_scale_factor * start_width)

        self.add(droste_image)
        self.play(
            UpdateFromAlphaFunc(
                droste_image,
                lambda m, a: m.set_width(start_width * np.exp(a * math.log(50)))
            ),
            droste_image.animate.scale(50),
            run_time=5
        )
        self.play(
            Restore(droste_image),
            FadeIn(a_plane),
            frame.animate.set_height(4).move_to(a_plane),
            run_time=2
        )
        self.add(planes)
        self.play(
            droste_image.animate.scale(1.0 / self.droste_scale_factor, about_point=a_plane.get_origin()),
            rate_func=lambda t: np.log(np.exp(smooth(t))),
            run_time=5
        )
        droste_image.restore()
        self.play(
            droste_image.animate.set_opacity(1),
            FadeOut(a_plane),
            FadeIn(dark_a_plane),
        )
        self.wait()

        # Show log
        frame = self.frame
        self.play(
            GrowArrow(log_arrow),
            FadeIn(log_label, LEFT),
            FadeIn(log_image_tiles),
            FadeIn(dark_b_plane),
            FadeOut(b_plane),
            frame.animate.reorient(0, 0, 0, (-0.09, 2.03, 0.0), 4.90).set_anim_args(run_time=2),
        )
        self.play(
            ShowCreation(droste_image, lag_ratio=0),
            ShowCreation(log_image_tiles, lag_ratio=0),
            FadeOut(droste_image.copy(), rate_func=rush_from),
            FadeOut(log_image_tiles.copy(), rate_func=rush_from),
            run_time=5
        )
        self.wait()

        # Show tile size
        b_unit = b_plane.get_unit_size()
        tile_box = Rectangle(math.log(self.droste_scale_factor), TAU)
        tile_box.scale(b_unit)
        tile_box.move_to(b_plane.get_origin(), DL)
        tile_box.set_stroke(YELLOW, 3)
        tile_box.set_z_index(2)

        tile_box_labels = VGroup(
            Tex(R"2\pi", font_size=24).next_to(tile_box, LEFT, SMALL_BUFF),
            Tex(R"\ln(256)", font_size=24).next_to(tile_box, UP, SMALL_BUFF),
        )
        tile_box_labels.set_backstroke(BLACK, 3)

        highlighted_tiles = log_image_tiles.copy()
        highlighted_tiles.always.clip_to_box(tile_box)

        self.add(highlighted_tiles, tile_box)
        self.play(
            ShowCreation(tile_box),
            log_image_tiles.animate.set_opacity(0.2),
            Write(tile_box_labels)
        )
        self.wait()
        self.play(
            VGroup(tile_box, tile_box_labels).animate.shift(tile_box.get_width() * LEFT)
        )
        self.wait()
        self.play(
            FadeOut(tile_box),
            FadeOut(tile_box_labels),
            FadeOut(highlighted_tiles),
            log_image_tiles.animate.set_opacity(1),
        )
        self.wait()

        # Show rotation
        mult_arrow, mult_label = self.get_mult_arrow_group()

        mult_plane, c_group = self.get_mult_plane(c_plane)
        mult_plane.match_x(mult_label)
        c_dot, c_vect, c_label = c_group
        c_dot.add_updater(lambda m: m.move_to(mult_plane.n2p(var_const.get_value())))

        rot_log_tiles = self.get_rot_tiles()

        self.play(
            frame.animate.set_height(FRAME_HEIGHT).move_to(2 * LEFT),
            LaggedStart(
                FadeIn(mult_arrow),
                FadeIn(mult_label),
                FadeIn(rot_log_tiles),
                FadeOut(c_plane),
                FadeIn(dark_c_plane),
                lag_ratio=0.1,
            ),
            run_time=2,
        )
        self.wait()
        self.play(
            FadeIn(mult_plane, 0.5 * DOWN),
            FadeIn(c_group, 0.5 * DOWN),
            mult_label.animate.next_to(mult_plane, UP),
        )
        self.play(
            var_const.animate.set_value(1),
            UpdateFromFunc(rot_log_tiles, lambda m: m.become(self.get_rot_tiles())),
            run_time=7,
            rate_func=lambda t: there_and_back_with_pause(t, 1 / 7),
        )
        self.wait()

        # Show final image
        self.pre_transform_log = self.get_log_image_tiles(b_plane, 5, 5, resolution=self.log_image_resolution)
        final_image = self.get_final_image()

        self.play(
            GrowArrow(exp_arrow),
            FadeIn(exp_label, 0.5 * RIGHT),
            ShowCreation(final_image, lag_ratio=0, run_time=5),
            ShowCreation(rot_log_tiles, lag_ratio=0, run_time=5),
            FadeOut(rot_log_tiles.copy()),
            FadeOut(d_plane),
            FadeIn(dark_d_plane),
        )
        self.wait()
        self.play(
            var_const.animate.set_value(1),
            UpdateFromFunc(rot_log_tiles, lambda m: m.become(self.get_rot_tiles())),
            UpdateFromFunc(final_image, lambda m: m.become(self.get_final_image())),
            run_time=7,
            rate_func=lambda t: there_and_back_with_pause(t, 1 / 7),
        )

        # Zoom in on the final image
        self.play(
            frame.animate.set_height(3.5).move_to(d_plane),
            dark_d_plane.animate.set_stroke(opacity=0.25),
            run_time=2,
        )
        self.wait()

        # Show the twisted Droste zoom
        rot_log_tiles.become(self.get_rot_tiles())
        final_image.become(get_final_image())
        b_unit = b_plane.get_unit_size()
        const = var_const.get_value()

        self.play(
            frame.animate.set_height(4.9).move_to(VGroup(c_plane, d_plane)),
            dark_d_plane.animate.set_stroke(opacity=0.25),
            run_time=2,
        )
        self.wait()
        self.play(
            rot_log_tiles.animate.shift(c_plane.n2p(1 * const * math.log(256)) - c_plane.get_origin()),
            self.pre_transform_log.animate.shift(1 * b_unit * math.log(256) * RIGHT),
            UpdateFromFunc(
                final_image,
                lambda m: m.become(get_final_image())
            ),
            run_time=5,
            rate_func=linear,
        )
        return

        # Change the final image based on constant

        # Zoom in on final image (Old)
        self.pre_transform_log = self.get_log_image_tiles(b_plane, 7, 7, resolution=self.log_image_resolution)
        final_image.become(self.get_final_image())


        exp_tracker = ValueTracker(0)
        ref_height = frame.get_height()
        frame.add_updater(lambda m: m.set_height(ref_height * np.exp(exp_tracker.get_value())).move_to(d_plane))

        self.play(
            exp_tracker.animate.set_value(-9.5),
            run_time=15
        )

    def get_log_image_path(self):
        return get_print_gallery_log_image_path()


class TransformPrintGalleryFromStraightToLoop(FourPannelsWithPrintGallery):
    droste_scale_adjustment = 150 / 256
    fixed_point = (3.8 + TAU) * 1j

    def construct(self):
        # Load up local variables
        planes = self.planes
        dark_planes = self.dark_planes
        a_plane, b_plane, c_plane, d_plane = planes
        dark_a_plane, dark_b_plane, dark_c_plane, dark_d_plane = dark_planes

        droste_image = self.droste_image
        blank_spot = self.blank_spot

        const = self.const
        var_const = self.var_const

        # Edit dark planes
        for dark_plane in dark_planes:
            dark_plane.background_lines.set_stroke(BLUE_D, 2)
            dark_plane.axes.set_stroke(BLUE_D, 3)

        self.add(dark_a_plane, dark_d_plane)

        # Change
        frame = self.frame
        frame.replace(d_plane, 1)

        self.var_fixed_point.set_value(3.8j + TAU * 1j)
        self.pre_transform_log = always_redraw(lambda: self.get_log_image_tiles(b_plane, 5, 5, resolution=self.log_image_resolution))
        final_image = always_redraw(self.get_final_image)

        self.add(final_image)
        self.add(self.pre_transform_log)
        self.var_const.set_value(1)
        self.play(
            self.var_const.animate.set_value(const),
            run_time=8,
        )


class FourStepsWithGrid(CreatingTheSpiral):
    droste_scale_factor = 16

    def construct(self):
        # Load up local variables
        planes = self.planes
        dark_planes = self.dark_planes
        a_plane, b_plane, c_plane, d_plane = planes
        dark_a_plane, dark_b_plane, dark_c_plane, dark_d_plane = dark_planes

        log_arrow_group = self.log_arrow_group
        exp_arrow_group = self.exp_arrow_group
        log_arrow, log_label = log_arrow_group
        exp_arrow, exp_label = exp_arrow_group

        droste_image = self.droste_image
        log_image_tiles = self.log_image_tiles
        blank_spot = self.blank_spot

        const = self.const
        fixed_point = self.fixed_point
        var_const = self.var_const
        var_fixed_point = self.var_fixed_point

        rot_func = self.rot_func
        get_rot_tiles = self.get_rot_tiles
        get_final_image = self.get_final_image

        # Add labels
        mult_arrow_group = self.get_mult_arrow_group()
        mult_arrow_group[1].scale(0.5, about_edge=RIGHT)
        for plane in [b_plane, c_plane]:
            plane.background_lines.set_stroke(BLUE, 1, 0.25)
            plane.faded_lines.set_stroke(width=0)
        self.add(planes)
        self.add(log_arrow_group)
        self.add(exp_arrow_group)
        self.add(mult_arrow_group)

        # Add nested square grid
        frame = self.frame
        grid = self.get_grid()

        self.play(
            frame.animate.set_height(4).move_to(a_plane),
            run_time=2
        )
        self.play(LaggedStartMap(FadeIn, grid, scale=1.2, lag_ratio=0.2, run_time=2))
        self.wait()

        # Take the log
        log_grid = grid.copy()
        log_grid.target = log_grid.generate_target()
        for piece in log_grid.target.family_members_with_points():
            piece.scale(0.99)
        apply_func_between_planes(log_grid.target, np.log, a_plane, b_plane)
        for piece in log_grid.target.family_members_with_points():
            piece.scale(1.0 / 0.99)

        box = Square()
        box.replace(a_plane)
        box.set_stroke(opacity=0)
        log_grid.always.clip_to_box(box)
        ab_plane = VGroup(a_plane, b_plane)

        self.play(
            MoveToTarget(log_grid),
            box.animate.move_to(b_plane),
            frame.animate.set_width(ab_plane.get_width() + 1).move_to(ab_plane),
            run_time=3
        )
        log_grid.clear_updaters()
        log_grid.clip_to_box(b_plane)
        self.wait()

        # Make the log tiles
        n_recursions = 12
        b_unit = b_plane.get_unit_size()
        x_shift = b_unit * n_recursions * math.log(2) * RIGHT
        y_shift = b_unit * TAU * UP
        log_tiles = VGroup(
            log_grid.copy().shift(x * x_shift + y * y_shift)
            for x in [1, 0]
            for y in range(-1, 2)
        )
        for tile in log_tiles[:3]:
            tile.set_submobject_colors_by_gradient(RED, BLUE)

        log_grid.clip_to_box(b_plane)
        log_tiles.clip_to_box(b_plane)

        self.remove(log_grid)
        self.play(
            LaggedStart(
                (TransformFromCopy(log_grid.copy().set_stroke(opacity=0.1), tile)
                for tile in log_tiles),
                lag_ratio=0.1,
                run_time=2
            )
        )
        self.wait()

        # Add rotated tiles
        rot_log_tiles = log_tiles.copy()
        apply_func_between_planes(rot_log_tiles, rot_func, b_plane, c_plane)
        rot_log_tiles.clip_to_box(c_plane)

        cd_plane = VGroup(c_plane, d_plane)
        mover = log_tiles.copy()
        box.set_fill(BLACK, 0)
        box.replace(b_plane)
        mover.always.clip_to_box(box)

        self.play(
            box.animate.replace(c_plane),
            ReplacementTransform(mover, rot_log_tiles),
            frame.animate.move_to(cd_plane),
            run_time=2
        )
        self.wait()

        # Show exponential
        rot_log_tiles_subset = VGroup(
            piece
            for piece in rot_log_tiles.family_members_with_points()
            if -PI - 0.01 < c_plane.y_axis.p2n(piece.get_center()) < PI + 0.01
        )
        rot_log_tiles_subset.sort(lambda p: -p[1])
        final_grid = apply_func_between_planes(rot_log_tiles_subset.copy(), np.exp, c_plane, d_plane)
        final_grid.clip_to_box(d_plane)

        final_grid_highlight = final_grid.copy().set_fill(TEAL, 0.75)
        rot_log_tiles_subset_highlight = rot_log_tiles_subset.copy().set_fill(TEAL, 0.75)

        kw = dict(run_time=5, lag_ratio=0.002)
        self.play(
            LaggedStartMap(FadeIn, final_grid, **kw),
            LaggedStartMap(VFadeInThenOut, final_grid_highlight, **kw),
            LaggedStartMap(VFadeInThenOut, rot_log_tiles_subset_highlight, **kw),
        )

        # Zoom in to relevant parts
        self.play(frame.animate.to_default_state(), run_time=2)
        self.wait()
        for plane in [a_plane, d_plane]:
            self.play(frame.animate.set_height(4).move_to(plane), run_time=2)
            self.wait()
            self.play(frame.animate.to_default_state(), run_time=2)
            self.wait()

        # Highlight tiny square
        frame = self.frame
        squares = VGroup(
            grid[2][20],
            grid[1][10],
            grid[3][0],
        ).copy()
        squares.set_fill(RED, 1)
        squares_image = apply_func_between_planes(squares.copy(), lambda z: np.exp(rot_func(np.log(z))), a_plane, d_plane)

        self.play(
            FadeIn(squares),
            VGroup(grid, log_tiles, rot_log_tiles, final_grid).animate.set_stroke(WHITE),
            frame.animate.set_height(3.8).move_to(a_plane),
            run_time=2
        )
        self.wait()
        self.play(
            TransformFromCopy(squares, squares_image, path_arc=15 * DEG),
            frame.animate.move_to(d_plane),
            run_time=3
        )
        self.wait()
        self.play(FadeOut(squares_image), FadeOut(squares))

        # Highlight more tiny squares
        final_grid.deactivate_clip_plane()
        self.remove(exp_arrow_group)
        self.remove(d_plane)
        n_samples = 30
        lower_squares = VGroup(*grid[2:6].copy().family_members_with_points())
        for square in lower_squares:
            square.scale(0.99)
        apply_func_between_planes(lower_squares, lambda z: np.exp(rot_func(np.log(z))), a_plane, d_plane)
        for square in lower_squares:
            square.scale(1.0 / 0.99)
        lower_squares.remove(*(
            s for s in lower_squares
            if np.max(np.abs(d_plane.p2c(s.get_center()))) > 0.8
        ))
        lower_squares.set_fill(TEAL, 1)
        samples = VGroup(*random.sample(list(lower_squares), n_samples))
        square_labels = VGroup(
            Text("square").replace(square, 0).scale(0.6)
            for square in samples
        )
        square_labels.set_backstroke(BLACK, 2)

        self.play(
            LaggedStartMap(VFadeInThenOut, samples, lag_ratio=0.2),
            LaggedStartMap(VFadeInThenOut, square_labels, lag_ratio=0.2),
            frame.animate.set_height(2).move_to(d_plane.get_origin()),
            run_time=8
        )
        self.wait()


    def get_four_planes(self, *args, faded_line_ratio=3, **kwargs):
        return super().get_four_planes(
            *args, faded_line_ratio=faded_line_ratio, **kwargs
        )

    def get_grid(self, n_recursions=12, n_rows=8, density_insertion=6, colors=(BLUE, YELLOW)):
        grid = get_nested_square_grid(n_rows=n_rows, n_recursions=n_recursions, scale_factor=2)
        grid.replace(self.planes[0])
        grid.set_stroke(width=1)
        grid.insert_n_curves(density_insertion)
        grid.set_submobject_colors_by_gradient(*colors, interp_by_hsl=True)
        return grid


class ModifyFinalGrid(FourStepsWithGrid):
    def construct(self):
        # Load up local variables
        planes = self.planes
        a_plane, b_plane, c_plane, d_plane = planes

        var_const = self.var_const
        rot_func = self.rot_func

        # Add labels
        self.add(a_plane, d_plane)

        # Add grid
        frame = self.frame
        grid = self.get_grid(n_recursions=20, colors=[BLUE, YELLOW, RED, BLUE])
        self.add(grid)

        # Modify the grid
        frame.set_height(d_plane.get_height()).move_to(d_plane)
        scale_factor_log_tracker = ValueTracker(0)

        def full_func(z):
            return np.exp(rot_func(np.log(z)))

        def get_final_grid():
            g = grid.copy()
            scale_factor_log = scale_factor_log_tracker.get_value()
            var_const.set_value(complex(0, TAU) / (complex(scale_factor_log, TAU)))
            for piece in g.family_members_with_points():
                piece.scale(0.95)
            result = apply_func_between_planes(g, full_func, a_plane, d_plane)
            for piece in g.family_members_with_points():
                piece.scale(1.0 / 0.95)
            result.clip_to_box(d_plane)
            return result

        final_grid = always_redraw(get_final_grid)

        self.add(final_grid)
        for scale_factor in [256, 16, 81]:
            self.play(
                scale_factor_log_tracker.animate.set_value(math.log(scale_factor)),
                run_time=5
            )
            self.wait()


class CreateMeshWarp(FourStepsWithGrid):
    log_image_resolution = (51, 101)

    def construct(self):
        # Load up local variables
        planes = self.planes
        a_plane, b_plane, c_plane, d_plane = planes
        droste_image = self.droste_image
        rot_func = self.rot_func

        # Prep images
        ad_planes = VGroup(a_plane, d_plane)
        bc_planes = VGroup(b_plane, c_plane)
        ad_planes.arrange(RIGHT, buff=1)
        bc_planes.arrange(RIGHT, buff=1)
        bc_planes.next_to(ad_planes, UP, buff=1)
        for plane in ad_planes:
            plane.faded_lines.set_opacity(0)

        self.pre_transform_log = self.get_log_image_tiles(b_plane, 3, 3, resolution=self.log_image_resolution)
        final_image = self.get_final_image()
        final_image.set_opacity(0.5)
        droste_image.set_opacity(0.75)
        droste_image.clip_to_box(a_plane)
        droste_image.move_to(a_plane.get_center())

        self.add(ad_planes)
        self.add(droste_image)
        self.add(final_image)
        self.frame.set_height(4.5)

        # Add nested square grid
        n_recursions = 12
        n_rows = 32

        grid = get_nested_square_grid(n_rows=n_rows, n_recursions=n_recursions, scale_factor=2)
        grid.replace(a_plane)
        grid.scale(4)

        # Controlled output grid
        w_period = rot_func(-TAU * 1j)
        rot_shift = c_plane.n2p(w_period) - c_plane.get_origin()
        c_unit = c_plane.get_unit_size()
        min_radius = 1 / 16
        flat_grid = VGroup(*grid.family_members_with_points().copy())
        flat_grid.set_stroke(WHITE, 1)
        log_grid = flat_grid.copy()

        for square in log_grid:
            square.scale(0.99)
        apply_func_between_planes(log_grid, upper_cut_log, a_plane, b_plane)
        for square in log_grid:
            square.scale(1.0 / 0.99)
        rot_log_grid = log_grid.copy()
        apply_func_between_planes(rot_log_grid, rot_func, b_plane, c_plane)
        # for square in rot_log_grid:
        #     if c_plane.p2n(square.get_center()).real < math.log(min_radius):
        #         square.shift(rot_shift)
        #     if c_plane.p2n(square.get_center()).imag > 0.75 * TAU:
        #         # square.shift(c_unit * TAU * DOWN)
        #         square.shift(rot_shift)
        warp_grid = rot_log_grid.copy()
        apply_func_between_planes(warp_grid, np.exp, c_plane, d_plane)

        removed_indices = []
        for n, square1, square2 in zip(it.count(), flat_grid, warp_grid):
            w_coords = d_plane.p2c(square2.get_center())
            if np.any(np.abs(w_coords) > 1.05):
                removed_indices.append(n)

        all_grids = [warp_grid, log_grid, rot_log_grid, flat_grid]

        for grid in all_grids:
            grid.remove(*(grid[n] for n in removed_indices))

        self.add(flat_grid)
        self.add(warp_grid)

        self.add(bc_planes)
        self.add(log_grid)
        self.add(rot_log_grid)
        self.frame.reorient(0, 0, 0, (0.23, 2.24, 0.0), 8.97)

        # Sort the grids
        neg_cz_imag_values = [-c_plane.y_axis.p2n(piece.get_center()) for piece in rot_log_grid]
        arg_sort = np.argsort(neg_cz_imag_values)

        for grid in all_grids:
            grid.set_submobjects([grid[i] for i in arg_sort])

        self.play(
            FadeIn(flat_grid, lag_ratio=0.5, run_time=10),
            FadeIn(log_grid, lag_ratio=0.5, run_time=10),
            FadeIn(rot_log_grid, lag_ratio=0.5, run_time=10),
            FadeIn(warp_grid, lag_ratio=0.5, run_time=10),
        )

        # Export
        for grid, plane in [(flat_grid, a_plane), (warp_grid, d_plane)]:
            grid.shift(-plane.get_origin())
            grid.scale(2.0, about_point=ORIGIN)
            grid.set_stroke(BLACK, 1)

        flat_grid_path = Path('/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2026/print_gallery/exponential/flat_grid.svg')
        warp_grid_path = Path('/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2026/print_gallery/exponential/warp_grid.svg')
        flat_grid_path.write_text(vmobject_to_svg(flat_grid))
        warp_grid_path.write_text(vmobject_to_svg(warp_grid))


class TileToDoughnut(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        file = get_print_gallery_log_image_path()
        tile = TexturedSurface(Square3D(resolution=(101, 101)), file)
        tile.set_shape(math.log(256), TAU).center()
        tiles = tile.get_grid(11, 11, buff=0)
        tiles.set_opacity(0.8)
        borders = VGroup(
            SurroundingRectangle(piece, buff=0)
            for piece in tiles
        )
        borders.set_stroke(RED, 1)
        tile.shift(1e-2 * OUT)

        def uv_func(u: float, v: float) -> np.ndarray:
            r1 = 2
            r2 = 1
            P = np.array([math.cos(v), math.sin(v), 0])
            return (r1 - r2 * math.cos(u)) * P - r2 * math.sin(u) * OUT

        torus = TexturedSurface(
            ParametricSurface(uv_func, resolution=(101, 101), u_range=(0, TAU), v_range=(0, TAU)),
            file
        )
        torus.set_z(3)

        frame.set_height(12)
        self.add(tiles, borders)
        self.play(
            tiles.animate.set_opacity(0.5),
            FadeIn(tile)
        )
        self.play(
            tiles.animate.set_opacity(0.5).set_anim_args(run_time=4),
            ReplacementTransform(tile, torus, time_span=(1, 5)),
            frame.animate.reorient(16, 45, 0, (0.01, 0.08, 0.14), 13.11).set_anim_args(run_time=6),
        )
        self.play(
            frame.animate.reorient(-39, 47, 0, (0.01, 0.08, 0.14), 13.11),
            run_time=8
        )
        self.wait()
