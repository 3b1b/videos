from __future__ import annotations

import itertools as it

import numpy as np

from manimlib.constants import DEFAULT_MOBJECT_TO_MOBJECT_BUFF
from manimlib.constants import LEFT, RIGHT
from manimlib.constants import WHITE
from manimlib.mobject.shape_matchers import BackgroundRectangle
from manimlib.mobject.svg.tex_mobject import Tex
from manimlib.mobject.svg.tex_mobject import TexText
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence, Union
    import numpy.typing as npt
    from manimlib.typing import ManimColor, VectNArray, Self


VECTOR_LABEL_SCALE_FACTOR = 0.8


def matrix_to_tex_string(matrix: npt.ArrayLike) -> str:
    matrix = np.array(matrix).astype("str")
    if matrix.ndim == 1:
        matrix = matrix.reshape((matrix.size, 1))
    n_rows, n_cols = matrix.shape
    prefix = R"\left[ \begin{array}{%s}" % ("c" * n_cols)
    suffix = R"\end{array} \right]"
    rows = [
        " & ".join(row)
        for row in matrix
    ]
    return prefix + R" \\ ".join(rows) + suffix


def matrix_to_mobject(matrix: npt.ArrayLike) -> Tex:
    return Tex(matrix_to_tex_string(matrix))


def vector_coordinate_label(
    vector_mob: VMobject,
    integer_labels: bool = True,
    n_dim: int = 2,
    color: ManimColor = WHITE
) -> Matrix:
    vect = np.array(vector_mob.get_end())
    if integer_labels:
        vect = np.round(vect).astype(int)
    vect = vect[:n_dim]
    vect = vect.reshape((n_dim, 1))
    label = Matrix(vect, add_background_rectangles_to_entries=True)
    label.scale(VECTOR_LABEL_SCALE_FACTOR)

    shift_dir = np.array(vector_mob.get_end())
    if shift_dir[0] >= 0:  # Pointing right
        shift_dir -= label.get_left() + DEFAULT_MOBJECT_TO_MOBJECT_BUFF * LEFT
    else:  # Pointing left
        shift_dir -= label.get_right() + DEFAULT_MOBJECT_TO_MOBJECT_BUFF * RIGHT
    label.shift(shift_dir)
    label.set_color(color)
    label.rect = BackgroundRectangle(label)
    label.add_to_back(label.rect)
    return label


def get_det_text(
    matrix: Matrix,
    determinant: int | str | None = None,
    background_rect: bool = False,
    initial_scale_factor: int = 2
) -> VGroup:
    parens = Tex("()")
    parens.scale(initial_scale_factor)
    parens.stretch_to_fit_height(matrix.get_height())
    l_paren, r_paren = parens.split()
    l_paren.next_to(matrix, LEFT, buff=0.1)
    r_paren.next_to(matrix, RIGHT, buff=0.1)
    det = TexText("det")
    det.scale(initial_scale_factor)
    det.next_to(l_paren, LEFT, buff=0.1)
    if background_rect:
        det.add_background_rectangle()
    det_text = VGroup(det, l_paren, r_paren)
    if determinant is not None:
        eq = Tex("=")
        eq.next_to(r_paren, RIGHT, buff=0.1)
        result = Tex(str(determinant))
        result.next_to(eq, RIGHT, buff=0.2)
        det_text.add(eq, result)
    return det_text
