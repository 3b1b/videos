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


# Scenes

class PrintGalleryZoom(InteractiveScene):
    def construct(self):
        # Add Droste image and zoom in two iterations
        droste_image = get_rectified_print_gallery()
        droste_image.scale(256).center()
        self.add(droste_image)

        exp_tracker = ValueTracker(0)
        frame = self.frame
        frame.add_updater(lambda m: m.set_height(FRAME_HEIGHT * np.exp(exp_tracker.get_value())))

        self.play(
            exp_tracker.animate.set_value(-1 * math.log(256)),
            rate_func=linear,
            run_time=30
        )


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
            run_time=3
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

        self.play(
            LaggedStart(
                AnimationGroup(MoveToTarget(in_line), Restore(in_plane)),
                AnimationGroup(MoveToTarget(out_line), Restore(out_plane)),
                lag_ratio=0.3
            ),
            ReplacementTransform(active_func_arrow, func_arrow, suspend_mobject_updating=True),
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
        for x in [0, 1, 2, -1, -2]:
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
        grid_density = 3
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
            ReplacementTransform(in_grid.copy(), out_grid, lag_ratio=0.5 / len(in_grid), run_time=4),
            out_plane.background_lines.animate.set_stroke(opacity=0.25),
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
        for _ in range(4):
            self.play(
                z_tracker.animate.increment_value(TAU * 1j),
                dot_arrows.animate.set_opacity(0.2),
                run_time=2,
            )


class TheNaturalLog(InteractiveScene):
    use_high_res_log_image = True

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
            mob.shift(-z_plane.get_origin())
            mob.scale(1.0 / z_plane.x_axis.get_unit_size(), about_point=ORIGIN)
            mob.apply_complex_function(func)
            mob.scale(w_plane.x_axis.get_unit_size(), about_point=ORIGIN)
            mob.shift(w_plane.get_origin())
            return mob

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
        small_image = not self.use_high_res_log_image
        log_image = get_log_image(get_pi_house_log_image_path(small=small_image), 16)
        log_image.scale(z_plane.get_unit_size())
        log_image.move_to(z_plane.n2p(math.log(3)), DR)

        droste_image = get_droste_from_log_image_path(get_pi_house_log_image_path(small=small_image), 16)
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

        z_dot = Group(TrueDot(), GlowDot()).set_color(BLUE)
        w_dot = Group(TrueDot(), GlowDot()).set_color(PINK)
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
        log_image_row.set_opacity(0.8)
        log_image_slices.set_opacity(0.8)

        self.play(
            LaggedStartMap(FadeIn, droste_image_rings, lag_ratio=0.5),
            LaggedStartMap(FadeIn, log_image_slices, lag_ratio=0.5),
            LaggedStartMap(FadeIn, v_lines, lag_ratio=0.5),
            LaggedStartMap(FadeIn, black_circles, lag_ratio=0.5),
            FadeOut(small_ring, time_span=(2, 4)),
            FadeOut(small_log_slice, time_span=(2, 4)),
            run_time=15
        )
        self.remove(log_image_slices, droste_image_rings)
        droste_image.set_opacity(1)
        self.add(log_image_row, droste_image, v_lines, black_circles)
        self.play(
            v_lines.animate.set_stroke(opacity=0.25),
            black_circles.animate.set_stroke(opacity=0.25),
        )
        self.wait()

        # Show a labeled z_value
        z_tracker.set_value(1.0)
        z_label = VGroup(Tex(R"z = "), DecimalNumber(complex(0)))
        z_label.scale(0.5)
        z_label.arrange(RIGHT, buff=SMALL_BUFF)
        z_label[1].shift(0.02 * UP)
        z_label[1].f_always.set_value(get_z)
        z_label.always.next_to(z_dot, UR, buff=-0.1)
        z_label.set_backstroke(BLACK, 5)
        traced_paths = self.get_traced_paths([z_dot, w_dot])
        traced_paths[1].set_stroke(width=1.5)

        brace = Brace(log_image, RIGHT, SMALL_BUFF)
        brace_label = brace.get_tex(R"2\pi")

        self.play(
            FadeIn(z_dot),
            FadeIn(w_dot),
            FadeIn(z_label),
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

        self.play(
            z_tracker.animate.increment_value(TAU * 1j),
            frame.animate.reorient(0, 0, 0, (0, 2, 0.0), 13),
            FadeOut(v_lines),
            FadeOut(black_circles),
            FadeOut(z_plane, time_span=(2, 4)),
            FadeIn(big_z_plane, time_span=(2, 4)),
            run_time=6
        )
        traced_paths.set_z_index(2)
        traced_paths[1].add_updater(lambda m: m.set_stroke(width=1.5))
        self.play(ShowCreation(log_image_slices, lag_ratio=0.01, run_time=5))
        self.wait()

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
            run_time=4
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
            frame.animate.set_y(0)
        )
        self.play(FlashAround(z_dot))
        self.play(
            w_plane.animate.scale(2, about_point=w_plane.get_left()),
            droste_image.animate.scale(2, about_point=w_plane.get_left()),
            run_time=2
        )
        self.wait()
        self.play(TransformFromCopy(z_dot, w_dot, suspend_mobject_updating=True))
        self.play(FlashAround(w_dot))
        self.wait()

        # Show the pi creatures
        droste_randy = Randolph(mode="happy", color="#4B66C9")
        droste_randy.set_z_index(3)
        droste_randy.set_fill(border_width=1)
        droste_randy.set_height(2)
        droste_randy.move_to(w_plane.c2p(-1.25, -1.35))

        log_randy = droste_randy.copy()
        log_randy.apply_points_function(lambda ps: np.array([
            z_plane.n2p(np.log(w_plane.p2n(p)))
            for p in ps
        ]), about_point=ORIGIN)
        log_randys = VGroup(
            log_randy,
            log_randy.copy().shift(TAU * z_plane.get_unit_size() * UP),
            log_randy.copy().shift(2 * TAU * z_plane.get_unit_size() * UP),
        )
        brace_group = VGroup(brace, brace_label)

        self.play(
            FadeOut(z_label),
            FadeOut(z_dot),
            FadeOut(w_dot),
            v_line.animate.set_stroke(width=1),
            pink_circle.animate.set_stroke(width=1),
            log_image_tiles.animate.set_opacity(0.25),
            droste_image.animate.set_opacity(0.25),
            w_plane.axes.animate.set_stroke(WHITE),
            w_plane.coordinate_labels.animate.set_fill(WHITE),
            FadeIn(log_randy)
        )
        self.play(FlashAround(log_randy, run_time=2))
        self.play(
            TransformFromCopy(log_randy, log_randys[1]),
            brace_group.animate.set_y(log_randy.get_y(), DOWN),
        )
        self.play(FlashAround(log_randys[1], run_time=2))
        self.play(TransformFromCopy(log_randys[1], log_randys[2]))
        self.play(FlashAround(log_randys[2], run_time=2))
        self.wait()
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
        self.play(FadeIn(log_image_row))
        self.wait()
        self.play(FadeOut(log_image_row))

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

        # Show exp and log properties
        func_labels = VGroup(right_arrow_label, right_arrow, left_arrow, left_arrow_label)
        t2c = {"z_1": BLUE, "z_2": BLUE_D, "w_1": PINK, "w_2": MAROON_B}
        exp_rule = Tex(R"e^{z_1 + z_2} = e^{z_1} e^{z_2}", t2c=t2c)
        log_rule = Tex(R"\ln(w_1 w_2) = \ln(w_1) + \ln(w_2)", t2c=t2c, font_size=40)
        rules = VGroup(exp_rule, log_rule)
        rules.scale(1.25)
        rules.arrange(DOWN, buff=LARGE_BUFF)
        rules.set_x(-0.1).set_y(-1.2)

        self.play(
            func_labels.animate.arrange(DOWN).align_to(w_plane, UP),
            Write(exp_rule, time_span=(0.5, 1.5)),
            Write(log_rule, time_span=(1.0, 2.0)),
        )
        for rule in [exp_rule, log_rule]:
            self.play(FlashAround(rule, run_time=2))

        self.wait()

        # Show example point w -> 16w
        arm_index = 170
        w_dot = Dot()
        w_dot.set_z_index(4)
        w_dot.set_color(RED)
        w_dot.move_to(small_droste_randy.body.get_points()[arm_index])
        w = w_plane.p2n(w_dot.get_center())

        w16_dot = w_dot.copy()
        w16_dot.move_to(w_plane.n2p(16 * w))

        w_label = Tex(R"w")
        w_label.set_color(RED)
        w_label.set_backstroke(BLACK, 2)
        w_label.next_to(w_dot, DL, SMALL_BUFF, aligned_edge=UP)

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

        self.play(
            frame.animate.reorient(0, 0, 0, (6.08, -0.13, 0.0), 1.37),
            FadeIn(w_dot, 0.1 * UR, time_span=(1, 2)),
            FadeIn(w_label, 0.1 * UR, time_span=(1, 2)),
            FadeOut(pink_circle),
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
            FadeIn(times_16_label, time_span=(2, 3)),
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
        for n in range(2):
            self.play(
                log_image_group.animate.shift(-shift_vect),
                annulus.animate.scale(1 / 16, about_point=w_plane.get_origin()),
                run_time=3,
            )
            self.wait()
        self.play(
            log_image_group.animate.shift(2 * shift_vect),
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

    def get_traced_paths(self, dots):
        result = VGroup()
        for dot in dots:
            path = TracedPath(dot.get_center, stroke_color=dot.family_members_with_points()[0].get_color(), stroke_width=3)
            result.add(path)
        return result
