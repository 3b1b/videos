from manim_imports_ext import *
from _2021.matrix_exp import *
import matplotlib.pyplot as plt


def get_intensity_colors(values, cmap_name='viridis'):
    """
    Convert a value between 0 and 1 to a color using a perceptually uniform colormap.

    Args:
        value (np.array): Array of values between 0 and 1
        cmap_name (str): Name of colormap (default: 'viridis')

    Returns:
        np.array: RGB color values as (r,g,b) where each component is between 0 and 1
    """
    cmap = plt.get_cmap(cmap_name)
    return cmap(values)[:, :3]  # Only return RGB, exclude alpha


class TexScratchPad(InteractiveScene):
    def construct(self):
        # ODE
        tex = Tex(R"""
            \left[\begin{array}{c} x'(t) \\ y'(t) \end{array}\right] =
            \left[\begin{array}{cc} 1 & 2 \\ 3 & 1 \end{array}\right]
            \left[\begin{array}{c} x(t) \\ y(t) \end{array}\right]
        """)

        # ODE in new coordinates
        tex = Tex(
            R"""
            \left[\begin{array}{c} \tilde{x}'(t) \\ \tilde{y}'(t) \end{array}\right] =
            \left[\begin{array}{cc} \lambda_1 & 0 \\ 0 & \lambda_2 \end{array}\right]
            \left[\begin{array}{c} \tilde{x}(t) \\ \tilde{y}(t) \end{array}\right]
            """,
            t2c={R"\lambda_1": TEAL, R"\lambda_2": YELLOW}
        )

        # Solution to ODE in new coordinates
        tex = Tex(
            R"""
            \tilde{x}(t) = \tilde{x}_0 e^{\lambda_1 t} \\
            \tilde{y}(t) = \tilde{y}_0 e^{\lambda_2 t} \\
            """,
            t2c={R"\lambda_1": TEAL, R"\lambda_2": YELLOW}
        )
        self.add(tex)

        # Meaning of the alternate coordinates
        tex = Tex(
            R"""
            x \hat{\textbf{\i}} + y \hat{\textbf{\j}} =
            \tilde{x} \vec{\textbf{v}}_1 + \tilde{y} \vec{\textbf{v}}_2
            """,
            t2c={
                R"\lambda_1": TEAL,
                R"\lambda_2": YELLOW,
                R"\vec{\textbf{v}}_1": TEAL,
                R"\vec{\textbf{v}}_2": YELLOW,
                R"\hat{\textbf{\i}}": GREEN,
                R"\hat{\textbf{\j}}": RED,
            }
        )
        self.add(tex)

        # Change of basis matrix
        mat_tex = Tex(
            R"""
            \left[\begin{array}{cc} a & b \\ c & d \end{array}\right]
            """,
        )
        old_cols = VGroup(
            mat_tex[1:4:2],
            mat_tex[2:5:2],
        )
        old_cols.set_opacity(0)
        new_cols = VGroup(
            Tex(R"\vec{\textbf{v}}_1").set_color(TEAL),
            Tex(R"\vec{\textbf{v}}_2").set_color(YELLOW),
        )
        for new_col, old_col in zip(new_cols, old_cols):
            lines = Line(UP, DOWN).set_height(0.35).replicate(2)
            lines[0].next_to(new_col, UP, SMALL_BUFF)
            lines[1].next_to(new_col, DOWN, SMALL_BUFF)
            new_col.add(lines)
            new_col.match_height(mat_tex)
            new_col.move_to(old_col)
        new_cols[1].align_to(new_cols[0], UP)
        cob_matrix = VGroup(mat_tex, new_cols)

        og_matrix = Tex(
            R"""
            \left[\begin{array}{cc} a & b \\ c & d \end{array}\right]
            """,
        )

        inv_cob = cob_matrix.copy()
        inv_cob.add(Tex(R"-1", font_size=24).next_to(inv_cob, RIGHT, SMALL_BUFF, aligned_edge=UP))

        diag_matrix = Tex(
            R"""
            \left[\begin{array}{cc} \lambda_1 & 0 \\ 0 & \lambda_2 \end{array}\right]
            """,
            t2c={R"\lambda_1": TEAL, R"\lambda_2": YELLOW}
        )

        expr = VGroup(
            inv_cob, og_matrix, cob_matrix,
            Tex("="), diag_matrix
        )
        expr.arrange(RIGHT, buff=MED_SMALL_BUFF)

        self.add(expr)

        # Show powers
        diag_matrix = Tex(
            R"""
            \left[\begin{array}{cc} \lambda_1 & 0 \\ 0 & \lambda_2 \end{array}\right]^n = 
            \left[\begin{array}{cc} \lambda_1^n & 0 \\ 0 & \lambda_2^n \end{array}\right]
            """,
            t2c={R"\lambda_1": TEAL, R"\lambda_2": YELLOW}
        )
        self.add(diag_matrix)

        # Solve
        diag_matrix = Tex(
            R"""
            \left[\begin{array}{cc} 0 & 1 \\ 1 & 1 \end{array}\right]^n = 
            \left[\begin{array}{cc} \lambda_1^n & 0 \\ 0 & \lambda_2^n \end{array}\right]
            """,
            t2c={R"\lambda_1": TEAL, R"\lambda_2": YELLOW}
        )
        self.add(diag_matrix)

        # Write eigenvectors
        eigen1_equation = Tex(
            R"""
            \left[\begin{array}{cc} 0 & 1 \\ 1 & 1 \end{array}\right]
            \left[\begin{array}{c} 1  \\ \lambda_1 \end{array}\right] =
            \left[\begin{array}{c} \lambda_1  \\ \lambda_1 + 1 \end{array}\right] =
            \left[\begin{array}{c} \lambda_1  \\ \lambda_1^2 \end{array}\right] =
            \lambda_1 \left[\begin{array}{c} 1  \\ \lambda_1 \end{array}\right] =
            """,
            t2c={R"\lambda_1": TEAL, R"\lambda_2": YELLOW}
        )
        eigen1 = Tex(
            R"""
            \vec{\textbf{v}}_1 = \left[\begin{array}{c} 1  \\ \lambda_1 \end{array}\right]
            """,
            t2c={R"\lambda_1": TEAL, R"\lambda_2": YELLOW}
        )
        eigen2 = Tex(
            R"""
            \vec{\textbf{v}}_2 = \left[\begin{array}{c} 1 \\ \lambda_2 \end{array}\right]
            """,
            t2c={R"\lambda_1": TEAL, R"\lambda_2": YELLOW}
        )
        self.add(eigen2)

        # Eigen matrix
        fib_cob = Tex(
            R"""
            S = \left[\begin{array}{cc} 1 & 1  \\ \lambda_1 & \lambda_2 \end{array}\right]
            """,
            t2c={R"\lambda_1": TEAL, R"\lambda_2": YELLOW}
        )
        self.add(fib_cob)

        # Fibonacci expression
        fib_expr = Tex(
            R"""
            A^n \left[\begin{array}{c} 0 \\ 1 \end{array}\right] =
            \left[\begin{array}{c} F_n \\ F_{n + 1} \end{array}\right]
            """,
            t2c={R"\lambda_1": TEAL, R"\lambda_2": YELLOW}
        )
        self.add(fib_expr)

        # Fib formula
        fib_formula = Tex(
            R"""
            F_n = \frac{\lambda_1^n - \lambda_2^n}{\lambda_1 - \lambda_2}
            """,
            t2c={R"\lambda_1": TEAL, R"\lambda_2": YELLOW}
        )
        self.add(fib_formula)

        # Translation
        diag_matrix = Tex(
            R"""
            S \left[\begin{array}{cc} \lambda_1^n & 0 \\ 0 & \lambda_2^n \end{array}\right] S^{-1} = 
            A^n
            """,
            t2c={R"\lambda_1": TEAL, R"\lambda_2": YELLOW}
        )
        self.add(diag_matrix)

        # Matrix
        tex = Tex(R"""
            \left[\begin{array}{cc} 3 & 1 \\ 0 & 2 \end{array}\right]
        """)
        tex[1:4:2].set_color(GREEN)
        tex[2:5:2].set_color(RED)
        self.add(tex)

        # Basis
        basis_syms = VGroup(
            Tex(R"\hat{\textbf{\i}}").set_color(GREEN),
            Tex(R"\hat{\textbf{\j}}").set_color(RED),
        )
        basis_syms.arrange(RIGHT)
        self.add(basis_syms)

        # Equation
        tex = Tex(R"""
            \left[\begin{array}{c} x \\ y \end{array}\right] =
            \left[\begin{array}{c} 1 \\ 1 + \sqrt{5} \end{array}\right]
        """, t2c={R"\lambda": TEAL})
        self.add(tex)

        # List bases
        tex = Tex(R"""
            \left[\begin{array}{c} 1 \\ 0 \end{array}\right],
            \left[\begin{array}{c} 0 \\ 1 \end{array}\right]
        """)
        self.add(tex)

        # Diagonal matrix
        tex = Tex(
            R"""
                \left[\begin{array}{cc} \lambda_1 & 0 \\ 0 & \lambda_2 \end{array}\right]
            """,
            t2c={R"\lambda_1": TEAL, R"\lambda_2": YELLOW}
        )
        self.add(tex)


def get_vector_field_and_stream_lines(
    func, coordinate_system,
    # Vector config
    vector_stroke_width=5,
    vector_opacity=0.5,
    density=4,
    # Streamline config
    sample_freq=5,
    n_samples_per_line=10,
    solution_time=1,
    arc_len=3,  # Does nothing
    time_width=0.5,
    line_color=WHITE,
    line_width=3,
    line_opacity=1.0,
):
    # Vector field
    vector_field = VectorField(
        func, axes,
        density=density,
        stroke_width=vector_stroke_width,
        stroke_opacity=vector_opacity,
    )

    # Streamlines
    stream_lines = StreamLines(
        func, axes,
        density=sample_freq,
        n_samples_per_line=n_samples_per_line,
        solution_time=solution_time,
        magnitude_range=vector_field.magnitude_range,
        color_by_magnitude=False,
        stroke_color=line_color,
        stroke_width=line_width,
        stroke_opacity=line_opacity,
    )
    animated_lines = AnimatedStreamLines(
        stream_lines,
        line_anim_config=dict(time_width=time_width),
        rate_multiple=0.25,
    )

    return vector_field, animated_lines


class VectorFieldSolution(InteractiveScene):
    def construct(self):
        # Add axes
        mat = np.array([[1, 2], [3, 1]])
        # mat = np.array([[2, 0], [0, -1]])
        axes = NumberPlane((-4, 4), (-2, 2), faded_line_ratio=1)
        axes.set_height(FRAME_HEIGHT)
        axes.background_lines.set_stroke(BLUE, 1)
        axes.faded_lines.set_stroke(BLUE, 0.5, 0.5)
        axes.add_coordinate_labels(font_size=36)

        def func(v):
            return 0.5 * np.dot(v, mat.T)

        self.add(axes)

        # Calculate eigen vectors
        eigenvalues, eigenvectors = np.linalg.eig(mat)
        eigenlines = VGroup(
            Line(-v, v).set_length(10)
            for v in eigenvectors.T
        )
        eigenlines.set_stroke(TEAL, 5)

        # Show the flow
        config = dict()
        # config = dict(step_multiple=0.5, vector_stroke_width=8)
        vector_field, animated_lines = get_vector_field_and_stream_lines(func, axes, **config)

        self.add(vector_field, animated_lines)
        vector_field.set_stroke(opacity=1)
        self.play(vector_field.animate.set_stroke(opacity=0.5))
        self.wait(10)
        self.play(ShowCreation(eigenlines))
        self.wait(10)


class Transformation(InteractiveScene):
    def construct(self):
        # Apply matrix
        mat = np.array([[1, 2], [3, 1]])

        ghost_plane = NumberPlane(faded_line_ratio=0)
        ghost_plane.set_stroke(GREY, 1)
        plane = self.get_plane()
        basis = VGroup(
            self.get_updated_vector((1, 0), plane, GREEN),
            self.get_updated_vector((0, 1), plane, RED),
        )

        self.add(ghost_plane, plane, basis)
        self.play(
            plane.animate.apply_matrix(mat),
            run_time=4
        )
        self.wait()

        # Fade out
        self.play(FadeOut(VGroup(ghost_plane, plane, basis)))

        # Show matrix in new coordinate system
        eigenvalues, eigenvectors = np.linalg.eig(mat)
        eigenplane = self.get_plane()
        eigenplane.apply_matrix(eigenvectors, about_point=ORIGIN)

        eigenbasis = VGroup(
            self.get_updated_vector((1, 0), eigenplane, TEAL),
            self.get_updated_vector((0, 1), eigenplane, YELLOW),
        )

        self.add(eigenplane, eigenbasis)
        self.play(
            eigenplane.animate.apply_matrix(mat),
            run_time=4
        )
        self.wait()

    def get_plane(self, x_range=(-16, 16), y_range=(-8, 8)):
        return NumberPlane(x_range, y_range, faded_line_ratio=1)

    def get_updated_vector(self, coords, coord_system, color=YELLOW, thickness=4, **kwargs):
        vect = Vector(RIGHT, fill_color=color, thickness=thickness, **kwargs)
        vect.add_updater(lambda m: m.put_start_and_end_on(
            coord_system.get_origin(),
            coord_system.c2p(*coords),
        ))
        return vect
