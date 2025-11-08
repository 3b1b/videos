"""
Natural Group Name: Geometric Series Visualization

Educational Objectives:
- To demonstrate infinite geometric series convergence through visual area
- To build intuition for why 1/2 + 1/4 + 1/8 + ... = 1
- To show how geometric series can be understood through repeated subdivision
- To connect abstract algebraic formulas to concrete visual representations

Story Arc & Intent:
The animation provides a visual proof of the geometric series sum formula by showing
how successive terms in the series correspond to progressively smaller regions that
fill a finite area. This makes the abstract concept of infinite sums tangible.

Narrative Flow:
- Hook/Opening: A square is presented, representing the total (1)
- Development: The square is subdivided repeatedly, with each piece representing a term
- Build-up: Terms are labeled and their sum is shown accumulating
- Climax: The visual shows that all pieces exactly fill the original square
- Resolution: The algebraic formula is connected to the visual representation
- Extension: The pattern is generalized to other ratios

Technical Implementation Notes:
- Scene Classes: GeometricSeriesSquare, GeneralGeometricSeries, AlgebraicFormula
- Key Visual Elements: Rectangles, subdivision lines, area labels, equations
- Animation Techniques: Successive subdivision, color highlighting, formula derivation
- Mathematical Concepts: Infinite series, limits, geometric progressions

Dependency Chain:
All scenes are independent. They use basic Rectangle, Text, and Tex mobjects from
manimlib. No custom utilities required.

Rendering Instructions:
To render these scenes, use the manim command-line tool:
- For a single scene: manimgl geometric_series_visual.py GeometricSeriesSquare
- For all scenes in sequence: iterate through SCENE_ORDER
"""

# ============================================================
# 1. IMPORTS
# ============================================================
from manimlib import *
import numpy as np

# ============================================================
# 2. CONFIGURATION AND CONSTANTS
# ============================================================
# All constants are from manimlib.constants and used directly in the scenes.
#
# Color Scheme:
# - BLUE: Primary color for the first/largest term (1/2)
# - TEAL: Second term (1/4)
# - GREEN: Third term (1/8)
# - YELLOW: Fourth term (1/16)
# - Gradient colors for subsequent terms
# - WHITE: Borders and text
# - GREY_A: De-emphasized elements
#
# Key Manim Constants Used:
# - UP, DOWN, LEFT, RIGHT: Direction vectors
# - ORIGIN: Center point (0, 0, 0)
# - SMALL_BUFF, MED_SMALL_BUFF: Spacing constants
# - DEGREES: Angle conversion

# Square configuration
SQUARE_SIZE = 6.0
SERIES_COLORS = [BLUE, TEAL, GREEN, YELLOW, MAROON_B, RED_B, PURPLE_B, PINK]

# ============================================================
# 3. UTILITY FUNCTIONS
# ============================================================

def get_series_term_color(index):
    """
    Get the color for the nth term in the geometric series visualization.

    Args:
        index: Term index (0-indexed)

    Returns:
        Color for the term
    """
    if index < len(SERIES_COLORS):
        return SERIES_COLORS[index]
    else:
        # For terms beyond our color list, interpolate
        return interpolate_color(SERIES_COLORS[-1], GREY_A, 0.5)

def create_term_label(numerator, denominator, font_size=36):
    """
    Create a fraction label for a series term.

    Args:
        numerator: Numerator of the fraction
        denominator: Denominator of the fraction
        font_size: Font size for the label

    Returns:
        Tex mobject representing the fraction
    """
    return Tex(
        f"\\frac{{{numerator}}}{{{denominator}}}",
        font_size=font_size
    )

# ============================================================
# 4. BASE CLASSES AND HELPERS
# ============================================================
# No custom base classes needed

# ============================================================
# 5. SCENE IMPLEMENTATIONS
# ============================================================

class GeometricSeriesSquare(InteractiveScene):
    """
    Part 1: Visual proof that 1/2 + 1/4 + 1/8 + ... = 1 using a square.

    Narrative purpose:
        To provide a concrete, visual proof of the most famous geometric series,
        making the infinite sum tangible and obvious through spatial reasoning.

    Mathematical content:
        Demonstrates that the infinite geometric series with first term 1/2 and
        ratio 1/2 sums to exactly 1 by showing perfect subdivision of a unit square.

    Visual approach:
        Start with a square. Repeatedly divide it in half, with each piece representing
        the next term in the series. The pieces perfectly tile the square, showing the
        sum equals the whole.
    """
    def construct(self):
        # ========================================
        # SETUP: Title and initial square
        # ========================================
        title = Text("Geometric Series: Visual Proof", font_size=48)
        title.to_edge(UP)

        self.play(FadeIn(title, shift=DOWN))
        self.wait()

        # ========================================
        # CREATE: The unit square
        # ========================================
        square = Square(side_length=SQUARE_SIZE)
        square.set_stroke(WHITE, 3)
        square.set_fill(GREY_A, 0.3)
        square.shift(0.5 * DOWN)

        one_label = Tex("1", font_size=72)
        one_label.move_to(square)

        self.play(
            DrawBorderThenFill(square),
            FadeIn(one_label, scale=1.5)
        )
        self.wait()

        # ========================================
        # QUESTION: Pose the series
        # ========================================
        series_eq = Tex(
            R"\frac{1}{2} + \frac{1}{4} + \frac{1}{8} + \frac{1}{16} + \cdots = \, ?",
            font_size=42
        )
        series_eq.next_to(square, DOWN, buff=1)

        self.play(
            FadeOut(one_label),
            Write(series_eq)
        )
        self.wait()

        # ========================================
        # SUBDIVIDE: Show the series terms visually
        # ========================================
        # Remove the original square, we'll build it piece by piece
        self.play(FadeOut(square))

        # Create rectangles for each term
        # Term 1: 1/2 (left half of square)
        rect1 = Rectangle(width=SQUARE_SIZE/2, height=SQUARE_SIZE)
        rect1.move_to(square.get_center() + (SQUARE_SIZE/4) * LEFT)
        rect1.set_fill(BLUE, 0.7)
        rect1.set_stroke(WHITE, 2)

        label1 = create_term_label(1, 2, font_size=48)
        label1.move_to(rect1)

        self.play(
            DrawBorderThenFill(rect1),
            FadeIn(label1, scale=1.2)
        )
        self.wait()

        # Term 2: 1/4 (top-right quarter)
        rect2 = Rectangle(width=SQUARE_SIZE/2, height=SQUARE_SIZE/2)
        rect2.move_to(square.get_center() + (SQUARE_SIZE/4) * RIGHT + (SQUARE_SIZE/4) * UP)
        rect2.set_fill(TEAL, 0.7)
        rect2.set_stroke(WHITE, 2)

        label2 = create_term_label(1, 4, font_size=40)
        label2.move_to(rect2)

        self.play(
            DrawBorderThenFill(rect2),
            FadeIn(label2, scale=1.2)
        )
        self.wait()

        # Term 3: 1/8 (bottom-right eighth, upper half)
        rect3 = Rectangle(width=SQUARE_SIZE/2, height=SQUARE_SIZE/4)
        rect3.move_to(square.get_center() + (SQUARE_SIZE/4) * RIGHT + (SQUARE_SIZE/8) * DOWN)
        rect3.set_fill(GREEN, 0.7)
        rect3.set_stroke(WHITE, 2)

        label3 = create_term_label(1, 8, font_size=32)
        label3.move_to(rect3)

        self.play(
            DrawBorderThenFill(rect3),
            FadeIn(label3, scale=1.2)
        )
        self.wait()

        # Term 4: 1/16 (remaining space, upper half)
        rect4 = Rectangle(width=SQUARE_SIZE/2, height=SQUARE_SIZE/8)
        rect4.move_to(square.get_center() + (SQUARE_SIZE/4) * RIGHT + (SQUARE_SIZE*3/16) * DOWN)
        rect4.set_fill(YELLOW, 0.7)
        rect4.set_stroke(WHITE, 2)

        label4 = create_term_label(1, 16, font_size=24)
        label4.move_to(rect4)

        self.play(
            DrawBorderThenFill(rect4),
            FadeIn(label4, scale=1.2)
        )
        self.wait()

        # Term 5 and beyond: Show pattern continues
        rect5 = Rectangle(width=SQUARE_SIZE/2, height=SQUARE_SIZE/16)
        rect5.move_to(square.get_center() + (SQUARE_SIZE/4) * RIGHT + (SQUARE_SIZE*7/32) * DOWN)
        rect5.set_fill(MAROON_B, 0.7)
        rect5.set_stroke(WHITE, 2)

        dots = Tex(R"\vdots", font_size=32)
        dots.move_to(rect5.get_center() + 0.3 * DOWN)

        self.play(
            DrawBorderThenFill(rect5),
            FadeIn(dots)
        )
        self.wait()

        # ========================================
        # RESOLUTION: Show the sum equals 1
        # ========================================
        answer = Tex(R"= 1", font_size=42, color=YELLOW)
        answer.next_to(series_eq, RIGHT, buff=0.3)

        # Draw border around the complete square
        border = square.copy()
        border.set_fill(opacity=0)
        border.set_stroke(YELLOW, 5)

        self.play(
            Write(answer),
            ShowCreation(border),
            run_time=2
        )
        self.wait(2)

        # ========================================
        # CLEANUP
        # ========================================
        everything = VGroup(
            rect1, rect2, rect3, rect4, rect5,
            label1, label2, label3, label4, dots,
            border, series_eq, answer, title
        )
        self.play(FadeOut(everything), run_time=1)
        self.wait()


class AlgebraicFormula(InteractiveScene):
    """
    Part 2: Derive the algebraic formula for geometric series.

    Narrative purpose:
        To connect the visual intuition from Part 1 to the algebraic formula,
        showing how the geometric series sum formula works mathematically.

    Mathematical content:
        Derives the formula S = a/(1-r) for a geometric series with first term a
        and common ratio r, using the standard algebraic manipulation.

    Visual approach:
        Step-by-step algebraic derivation with clear visual highlighting of each
        step. Connect back to the specific example from Part 1.
    """
    def construct(self):
        # ========================================
        # SETUP: Title
        # ========================================
        title = Text("Geometric Series Formula", font_size=48)
        title.to_edge(UP)

        self.play(FadeIn(title, shift=DOWN))
        self.wait()

        # ========================================
        # GENERAL SERIES: Define the general form
        # ========================================
        general_series = Tex(
            R"S = a + ar + ar^2 + ar^3 + \cdots",
            font_size=48
        )
        general_series.shift(1.5 * UP)

        self.play(Write(general_series))
        self.wait()

        # ========================================
        # DERIVATION: Multiply by r
        # ========================================
        multiply_step = Tex(
            R"rS = ar + ar^2 + ar^3 + ar^4 + \cdots",
            font_size=48
        )
        multiply_step.next_to(general_series, DOWN, buff=0.7)

        self.play(TransformMatchingStrings(general_series.copy(), multiply_step))
        self.wait()

        # ========================================
        # DERIVATION: Subtract the equations
        # ========================================
        subtraction = Tex(
            R"S - rS = a",
            font_size=48
        )
        subtraction.next_to(multiply_step, DOWN, buff=0.7)

        self.play(Write(subtraction))
        self.wait()

        # ========================================
        # DERIVATION: Factor and solve
        # ========================================
        factor_step = Tex(
            R"S(1 - r) = a",
            font_size=48
        )
        factor_step.next_to(subtraction, DOWN, buff=0.7)

        self.play(TransformMatchingStrings(subtraction.copy(), factor_step))
        self.wait()

        # ========================================
        # FINAL FORMULA
        # ========================================
        final_formula = Tex(
            R"S = \frac{a}{1 - r}",
            font_size=60,
            color=YELLOW
        )
        final_formula.next_to(factor_step, DOWN, buff=1)

        box = SurroundingRectangle(final_formula, buff=0.2, color=YELLOW, stroke_width=3)

        self.play(
            TransformMatchingStrings(factor_step.copy(), final_formula),
            ShowCreation(box)
        )
        self.wait(2)

        # ========================================
        # APPLICATION: Apply to our example
        # ========================================
        # Fade out derivation steps
        derivation_steps = VGroup(general_series, multiply_step, subtraction, factor_step)
        self.play(
            FadeOut(derivation_steps),
            VGroup(final_formula, box).animate.shift(2 * UP)
        )

        # Show our specific example
        example = Tex(
            R"a = \frac{1}{2}, \quad r = \frac{1}{2}",
            font_size=42
        )
        example.next_to(final_formula, DOWN, buff=1)

        self.play(Write(example))
        self.wait()

        # Calculate the result
        calculation = Tex(
            R"S = \frac{1/2}{1 - 1/2} = \frac{1/2}{1/2} = 1",
            font_size=48
        )
        calculation.next_to(example, DOWN, buff=0.7)

        self.play(Write(calculation))
        self.wait(2)

        # ========================================
        # FINALE: Emphasize the result
        # ========================================
        checkmark = Tex(R"\checkmark", font_size=72, color=GREEN)
        checkmark.next_to(calculation, RIGHT, buff=0.5)

        self.play(FadeIn(checkmark, scale=2))
        self.wait(2)


class GeneralGeometricSeries(InteractiveScene):
    """
    Part 3: Explore geometric series with different ratios.

    Narrative purpose:
        To generalize the visual intuition by showing how the series behaves
        with different common ratios, reinforcing the formula S = a/(1-r).

    Mathematical content:
        Demonstrates how changing the ratio r affects convergence speed and
        the final sum value, using the formula to predict results.

    Visual approach:
        Show multiple examples with different ratios side by side or in sequence.
        Use visual area representations to show how different ratios affect the sum.
    """
    def construct(self):
        # ========================================
        # SETUP: Title
        # ========================================
        title = Text("Different Ratios, Different Sums", font_size=48)
        title.to_edge(UP)

        self.play(FadeIn(title, shift=DOWN))
        self.wait()

        # ========================================
        # FORMULA REMINDER
        # ========================================
        formula = Tex(
            R"S = \frac{a}{1 - r}",
            font_size=42
        )
        formula.next_to(title, DOWN, buff=0.5)

        self.play(Write(formula))
        self.wait()

        # ========================================
        # EXAMPLE 1: r = 1/3
        # ========================================
        example1_label = Tex(
            R"a = 1, \quad r = \frac{1}{3}",
            font_size=36
        )
        example1_label.shift(2 * LEFT + 0.5 * UP)

        example1_sum = Tex(
            R"S = \frac{1}{1 - 1/3} = \frac{3}{2}",
            font_size=36
        )
        example1_sum.next_to(example1_label, DOWN, buff=0.5)

        # Visual representation: bars showing terms
        bar_height = 1.5
        bar1_width = 2.0  # Represents sum of 1.5
        bar1 = Rectangle(width=bar1_width, height=bar_height)
        bar1.next_to(example1_sum, DOWN, buff=0.5)
        bar1.set_fill(BLUE, 0.6)
        bar1.set_stroke(WHITE, 2)

        sum_label1 = Tex(R"= \frac{3}{2}", font_size=32)
        sum_label1.next_to(bar1, RIGHT, buff=0.3)

        self.play(
            Write(example1_label),
            Write(example1_sum)
        )
        self.wait()
        self.play(
            DrawBorderThenFill(bar1),
            FadeIn(sum_label1)
        )
        self.wait()

        # ========================================
        # EXAMPLE 2: r = 2/3
        # ========================================
        example2_label = Tex(
            R"a = 1, \quad r = \frac{2}{3}",
            font_size=36
        )
        example2_label.shift(2 * RIGHT + 0.5 * UP)

        example2_sum = Tex(
            R"S = \frac{1}{1 - 2/3} = 3",
            font_size=36
        )
        example2_sum.next_to(example2_label, DOWN, buff=0.5)

        # Visual: larger bar for larger sum
        bar2_width = 4.0  # Represents sum of 3
        bar2 = Rectangle(width=bar2_width, height=bar_height)
        bar2.next_to(example2_sum, DOWN, buff=0.5)
        bar2.set_fill(GREEN, 0.6)
        bar2.set_stroke(WHITE, 2)

        sum_label2 = Tex(R"= 3", font_size=32)
        sum_label2.next_to(bar2, RIGHT, buff=0.3)

        self.play(
            Write(example2_label),
            Write(example2_sum)
        )
        self.wait()
        self.play(
            DrawBorderThenFill(bar2),
            FadeIn(sum_label2)
        )
        self.wait()

        # ========================================
        # INSIGHT: Larger r means larger sum
        # ========================================
        insight = Text(
            "Larger ratio → Larger sum\n(but must have r < 1 for convergence)",
            font_size=32,
            color=YELLOW
        )
        insight.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(insight, shift=UP))
        self.wait(3)

        # ========================================
        # CLEANUP
        # ========================================
        everything = VGroup(
            example1_label, example1_sum, bar1, sum_label1,
            example2_label, example2_sum, bar2, sum_label2,
            insight, formula, title
        )
        self.play(FadeOut(everything), run_time=1)
        self.wait()


# ============================================================
# 6. SCENE EXECUTION ORDER
# ============================================================
# The order of scenes that tells the story.
#
# Scene 1 (GeometricSeriesSquare):
#   - Visual proof using square subdivision
#   - Shows 1/2 + 1/4 + 1/8 + ... = 1
#   - Builds spatial intuition for infinite sums
#
# Scene 2 (AlgebraicFormula):
#   - Derives the general formula S = a/(1-r)
#   - Connects algebra to visual intuition
#   - Verifies the formula on the specific example
#
# Scene 3 (GeneralGeometricSeries):
#   - Explores series with different ratios
#   - Shows how r affects the sum
#   - Reinforces the formula with multiple examples

SCENE_ORDER = [
    GeometricSeriesSquare,      # Part 1: Visual proof with square
    AlgebraicFormula,           # Part 2: Algebraic derivation
    GeneralGeometricSeries,     # Part 3: General exploration
]
