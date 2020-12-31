from big_ol_pile_of_manim_imports import *


class QuaternionsDescribingRotation(EulerAnglesAndGimbal):
    CONFIG = {
        "use_lightweight_axes": True,
        "quaternions_and_imaginary_part_labels": [
            ([1, 1, 0, 0], "{i}"),
            # ([1, 0, 1, 0], "{j}"),
            # ([0, 0, 0, 1], "{k}"),
            # ([1, 1, 1, 1], "\\left({{i} + {j} + {k} \\over \\sqrt{3}}\\right)"),
        ],
    }

    def construct(self):
        self.setup_position()
        self.show_rotations()

    def show_rotations(self):
        for quat, ipl in self.quaternions_and_imaginary_part_labels:
            quat = normalize(quat)
            axis = quat[1:]
            angle = 2 * np.arccos(quat[0])
            label = self.get_label(angle, ipl)

            prism = RandyPrism()
            prism.scale(2)

            self.play(
                LaggedStartMap(FadeInFromDown, label),
                FadeIn(prism),
            )
            self.play(Rotate(
                prism,
                angle=angle, axis=axis,
                run_time=3,
                about_point=ORIGIN,
            ))


    #
    def get_label(self, angle, imaginary_part_label):
        deg = int(angle / DEGREES)
        ipl = imaginary_part_label
        kwargs = {
            "tex_to_color_map": {
                "{i}": I_COLOR,
                "{j}": J_COLOR,
                "{k}": K_COLOR,
            }
        }
        p_label = TexMobject(
            "x{i} + y{j} + z{k}", **kwargs
        )
        arrow = TexMobject(
            "\\rightarrow"
        )
        q_label = TexMobject(
            "\\big(\\cos(%d^\\circ) + \\sin(%d^\\circ)%s \\big)" % (deg, deg, ipl),
            **kwargs
        )
        inner_p_label = TexMobject(
            "\\left(x{i} + y{j} + z{k} \\right)",
            **kwargs
        )
        q_inv_label = TexMobject(
            "\\big(\\cos(-%d^\\circ) + \\sin(-%d^\\circ)%s \\big)" % (deg, deg, ipl),
            **kwargs
        )
        equation = VGroup(
            p_label, arrow, q_label, inner_p_label, q_inv_label
        )
        equation.arrange(RIGHT, buff=SMALL_BUFF)
        equation.set_width(FRAME_WIDTH - 1)
        equation.to_edge(UP)

        parts_text_colors = [
            (p_label, "\\text{3d point}", YELLOW),
            (q_label, "q", PINK),
            (inner_p_label, "\\text{3d point}", YELLOW),
            (q_inv_label, "q^{-1}", PINK),
        ]
        braces = VGroup()
        for part, text, color in parts_text_colors:
            brace = Brace(part, DOWN, buff=SMALL_BUFF)
            label = brace.get_tex(text, buff=MED_SMALL_BUFF)
            label.set_color(color)
            brace.add(label)
            braces.add(brace)
        braces[-1][-1].shift(0.2 * UR)

        equation.add_to_back(BackgroundRectangle(equation))
        equation.braces = braces
        equation.add(*braces)

        self.add_fixed_in_frame_mobjects(equation)
        self.add_fixed_in_frame_mobjects(braces)
        return equation