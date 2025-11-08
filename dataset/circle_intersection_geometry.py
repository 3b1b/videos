"""
Natural Group Name: Circle Intersection Geometry

Educational Objectives:
- To visualize the geometric relationship between two overlapping circles
- To demonstrate boolean operations (intersection and union) on geometric shapes
- To explore how changing the distance between centers affects the overlapping region
- To build intuition for set theory concepts through visual geometry

Story Arc & Intent:
The animation explores the fundamental geometry of two overlapping circles, demonstrating
how boolean operations can be applied to geometric objects. This serves as a visual
foundation for understanding set theory, Venn diagrams, and geometric probability.

Narrative Flow:
- Hook/Opening: Two circles are introduced and positioned to create an overlap
- Development: The intersection region is highlighted, showing the shared area between circles
- Exploration: The union of the circles is shown, demonstrating the combined area
- Extension: Dynamic adjustment shows how different overlaps create different proportions
- Resolution: Clear visual understanding of intersection vs. union operations

Technical Implementation Notes:
- Scene Classes: CircleIntersection, CircleUnion, DynamicOverlap
- Key Visual Elements: Circles, Intersection regions, Union regions, Fill colors
- Animation Techniques: DrawBorderThenFill, color transitions, opacity changes
- Mathematical Concepts: Set theory, boolean operations, geometric intersection

Dependency Chain:
All scenes are independent and use only basic manimlib primitives. No custom utility
classes are required. Scenes inherit from Scene or InteractiveScene.

Rendering Instructions:
To render these scenes, use the manim command-line tool:
- For a single scene: manimgl circle_intersection_geometry.py CircleIntersection
- For all scenes: iterate through SCENE_ORDER list
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
# - BLUE_D: Primary circle color (represents first set)
# - TEAL_D: Secondary circle color (represents second set)
# - TEAL: Intersection highlight color (overlap of both sets)
# - WHITE: Border/stroke color for emphasis
# - GREY_A: De-emphasized elements
#
# Key Manim Constants Used:
# - RIGHT, LEFT, UP, DOWN: Unit direction vectors for positioning
# - LARGE_BUFF, MED_LARGE_BUFF: Standard spacing constants
# - DEGREES: Conversion factor for degree-based rotations

# Configuration for circle intersection scenes
CIRCLE_RADIUS = 2.5
CIRCLE_SEPARATION = -2.0  # Negative value means circles overlap
CIRCLE_STROKE_WIDTH = 3
FILL_OPACITY_NORMAL = 0.5
FILL_OPACITY_DIMMED = 0.25
FILL_OPACITY_BRIGHT = 1.0

# ============================================================
# 3. UTILITY FUNCTIONS
# ============================================================

def get_circle_pair(radius=CIRCLE_RADIUS, separation=CIRCLE_SEPARATION):
    """
    Create a pair of circles arranged horizontally with specified separation.

    Args:
        radius: Radius of each circle
        separation: Distance between circle centers (negative = overlap)

    Returns:
        VGroup containing two Circle objects
    """
    circles = VGroup(*[Circle(radius=radius) for _ in range(2)])
    circles.arrange(RIGHT, buff=separation)
    return circles

def setup_circle_styles(circles, colors=(BLUE_D, TEAL_D), opacity=FILL_OPACITY_NORMAL):
    """
    Apply consistent styling to a pair of circles.

    Args:
        circles: VGroup of circles to style
        colors: Tuple of fill colors for each circle
        opacity: Fill opacity value
    """
    circles[0].set_fill(colors[0], opacity)
    circles[1].set_fill(colors[1], opacity)
    circles.set_stroke(WHITE, width=0)
    return circles

# ============================================================
# 4. BASE CLASSES AND HELPERS
# ============================================================
# No custom base classes needed - using standard Scene and InteractiveScene

# ============================================================
# 5. SCENE IMPLEMENTATIONS
# ============================================================

class CircleIntersection(InteractiveScene):
    """
    Part 1: Introduce two overlapping circles and highlight their intersection.

    Narrative purpose:
        To introduce the concept of geometric intersection through two overlapping circles,
        building visual intuition for set intersection operations.

    Mathematical content:
        Demonstrates the intersection of two circular regions, showing the area that
        belongs to both circles simultaneously.

    Visual approach:
        Progressive reveal: first show the individual circles, then highlight their
        shared region. Use color and opacity changes to emphasize the intersection.
    """
    def construct(self):
        # ========================================
        # SETUP: Create title
        # ========================================
        title = Text("Intersection of Two Circles", font_size=48)
        title.to_edge(UP)

        self.play(FadeIn(title, shift=DOWN))
        self.wait()

        # ========================================
        # CREATE: Two overlapping circles
        # ========================================
        circles = get_circle_pair()
        setup_circle_styles(circles)

        # Position circles in the center, below the title
        circles.next_to(title, DOWN, buff=LARGE_BUFF)

        # Labels for the circles
        labels = VGroup(
            Text("Set A", font_size=36),
            Text("Set B", font_size=36),
        )

        # Position labels outside the circles
        labels[0].next_to(circles[0], LEFT, buff=0.5)
        labels[1].next_to(circles[1], RIGHT, buff=0.5)
        labels[0].set_color(BLUE)
        labels[1].set_color(TEAL)

        # ========================================
        # ANIMATE: Draw the circles
        # ========================================
        self.play(
            LaggedStart(
                DrawBorderThenFill(circles[0]),
                DrawBorderThenFill(circles[1]),
                lag_ratio=0.3
            ),
            run_time=2
        )
        self.play(
            FadeIn(labels[0], shift=RIGHT),
            FadeIn(labels[1], shift=LEFT),
        )
        self.wait()

        # ========================================
        # HIGHLIGHT: The intersection region
        # ========================================
        # Create the intersection shape
        intersection = Intersection(circles[0], circles[1])
        intersection.set_stroke(WHITE, CIRCLE_STROKE_WIDTH)
        intersection.set_fill(TEAL, FILL_OPACITY_BRIGHT)

        # Dim the original circles to emphasize the intersection
        self.play(
            DrawBorderThenFill(intersection),
            circles.animate.set_fill(opacity=FILL_OPACITY_DIMMED),
            run_time=2
        )
        self.wait()

        # ========================================
        # LABEL: The intersection
        # ========================================
        intersection_label = Tex(R"A \cap B", font_size=48)
        intersection_label.move_to(intersection.get_center())
        intersection_label.set_color(YELLOW)

        self.play(FadeIn(intersection_label, scale=1.5))
        self.wait(2)

        # ========================================
        # CLEANUP: Fade out for transition
        # ========================================
        self.play(
            FadeOut(VGroup(circles, intersection, labels, intersection_label, title)),
            run_time=1
        )
        self.wait()


class CircleUnion(InteractiveScene):
    """
    Part 2: Show the union of two overlapping circles.

    Narrative purpose:
        To contrast intersection with union, showing how the union operation combines
        both circular regions into a single composite shape.

    Mathematical content:
        Demonstrates the union of two circular regions, showing the total area covered
        by either or both circles.

    Visual approach:
        Build on the intersection concept by showing how union encompasses everything.
        Use border highlighting to show the combined boundary.
    """
    def construct(self):
        # ========================================
        # SETUP: Create title
        # ========================================
        title = Text("Union of Two Circles", font_size=48)
        title.to_edge(UP)

        self.play(FadeIn(title, shift=DOWN))
        self.wait()

        # ========================================
        # CREATE: Two overlapping circles
        # ========================================
        circles = get_circle_pair()
        setup_circle_styles(circles, opacity=FILL_OPACITY_DIMMED)
        circles.next_to(title, DOWN, buff=LARGE_BUFF)

        # ========================================
        # SHOW: Individual circles first
        # ========================================
        self.play(
            LaggedStart(
                FadeIn(circles[0]),
                FadeIn(circles[1]),
                lag_ratio=0.3
            ),
            run_time=1.5
        )
        self.wait()

        # ========================================
        # CREATE AND HIGHLIGHT: The union
        # ========================================
        union = Union(circles[0], circles[1])
        union.set_stroke(WHITE, CIRCLE_STROKE_WIDTH)
        union.set_fill(BLUE, FILL_OPACITY_NORMAL)

        self.play(
            FadeOut(circles),
            DrawBorderThenFill(union),
            run_time=2
        )
        self.wait()

        # ========================================
        # LABEL: The union
        # ========================================
        union_label = Tex(R"A \cup B", font_size=48)
        union_label.move_to(union.get_center())
        union_label.set_color(YELLOW)

        self.play(FadeIn(union_label, scale=1.5))
        self.wait(2)

        # ========================================
        # CLEANUP
        # ========================================
        self.play(
            FadeOut(VGroup(union, union_label, title)),
            run_time=1
        )
        self.wait()


class CompareIntersectionAndUnion(InteractiveScene):
    """
    Part 3: Side-by-side comparison of intersection and union.

    Narrative purpose:
        To solidify understanding by showing intersection and union simultaneously,
        making the distinction clear through direct visual comparison.

    Mathematical content:
        Direct comparison of set intersection (A ∩ B) vs set union (A ∪ B),
        demonstrating the fundamental difference between these operations.

    Visual approach:
        Split screen showing both operations on the same pair of circles.
        Use consistent colors and labels to enable easy comparison.
    """
    def construct(self):
        # ========================================
        # SETUP: Create title
        # ========================================
        title = Text("Intersection vs Union", font_size=48)
        title.to_edge(UP)

        self.play(Write(title))
        self.wait()

        # ========================================
        # CREATE: Two sets of circles (left and right)
        # ========================================
        left_circles = get_circle_pair()
        right_circles = get_circle_pair()

        setup_circle_styles(left_circles)
        setup_circle_styles(right_circles)

        # Position circles
        left_circles.scale(0.8).shift(3.5 * LEFT + 0.5 * DOWN)
        right_circles.scale(0.8).shift(3.5 * RIGHT + 0.5 * DOWN)

        # ========================================
        # LABELS: Intersection and Union titles
        # ========================================
        left_title = Text("Intersection", font_size=36, color=TEAL)
        right_title = Text("Union", font_size=36, color=BLUE)

        left_title.next_to(left_circles, UP, buff=0.7)
        right_title.next_to(right_circles, UP, buff=0.7)

        # ========================================
        # ANIMATE: Show both sets of circles
        # ========================================
        self.play(
            LaggedStart(
                DrawBorderThenFill(left_circles),
                DrawBorderThenFill(right_circles),
                lag_ratio=0.2
            ),
            run_time=2
        )
        self.play(
            FadeIn(left_title, shift=DOWN),
            FadeIn(right_title, shift=DOWN),
        )
        self.wait()

        # ========================================
        # CREATE: Intersection and Union shapes
        # ========================================
        intersection = Intersection(left_circles[0], left_circles[1])
        intersection.set_stroke(WHITE, CIRCLE_STROKE_WIDTH)
        intersection.set_fill(TEAL, FILL_OPACITY_BRIGHT)

        union = Union(right_circles[0], right_circles[1])
        union.set_stroke(WHITE, CIRCLE_STROKE_WIDTH)
        union.set_fill(BLUE, FILL_OPACITY_NORMAL)

        # ========================================
        # HIGHLIGHT: Both operations simultaneously
        # ========================================
        self.play(
            DrawBorderThenFill(intersection),
            left_circles.animate.set_fill(opacity=FILL_OPACITY_DIMMED),
            DrawBorderThenFill(union),
            FadeOut(right_circles),
            run_time=2.5
        )
        self.wait()

        # ========================================
        # LABELS: Mathematical notation
        # ========================================
        intersection_label = Tex(R"A \cap B", font_size=36, color=YELLOW)
        union_label = Tex(R"A \cup B", font_size=36, color=YELLOW)

        intersection_label.next_to(left_circles, DOWN, buff=0.7)
        union_label.next_to(union, DOWN, buff=0.7)

        self.play(
            FadeIn(intersection_label, scale=1.2),
            FadeIn(union_label, scale=1.2),
        )
        self.wait(3)

        # ========================================
        # FINALE: Hold the comparison
        # ========================================
        self.wait(2)


class DynamicCircleOverlap(InteractiveScene):
    """
    Part 4: Interactive demonstration of how overlap distance affects intersection.

    Narrative purpose:
        To show how the degree of overlap between circles affects the size of their
        intersection, building dynamic intuition for the relationship.

    Mathematical content:
        As the distance between circle centers changes, the intersection area changes
        continuously from zero (no overlap) to maximum (perfect overlap).

    Visual approach:
        Animate one circle moving relative to the other, with the intersection
        region updating in real-time to show the changing overlap.
    """
    def construct(self):
        # ========================================
        # SETUP: Create title and initial circles
        # ========================================
        title = Text("Dynamic Circle Overlap", font_size=48)
        title.to_edge(UP)

        self.add(title)

        # Create fixed and moving circles
        fixed_circle = Circle(radius=CIRCLE_RADIUS)
        fixed_circle.set_fill(BLUE_D, FILL_OPACITY_NORMAL)
        fixed_circle.set_stroke(WHITE, 0)
        fixed_circle.shift(1.5 * LEFT)

        moving_circle = Circle(radius=CIRCLE_RADIUS)
        moving_circle.set_fill(TEAL_D, FILL_OPACITY_NORMAL)
        moving_circle.set_stroke(WHITE, 0)
        moving_circle.shift(1.5 * RIGHT)

        circles = VGroup(fixed_circle, moving_circle)
        circles.shift(0.5 * DOWN)

        # ========================================
        # SHOW: Initial state
        # ========================================
        self.play(
            DrawBorderThenFill(fixed_circle),
            DrawBorderThenFill(moving_circle),
        )
        self.wait()

        # ========================================
        # ANIMATE: Move circles together
        # ========================================
        # Move the right circle to the left to increase overlap
        self.play(
            moving_circle.animate.shift(3 * LEFT),
            run_time=3,
            rate_func=smooth
        )
        self.wait()

        # ========================================
        # HIGHLIGHT: Maximum overlap
        # ========================================
        # At this point, circles are mostly overlapping
        # Create and show the intersection
        final_intersection = Intersection(fixed_circle, moving_circle)
        final_intersection.set_stroke(WHITE, CIRCLE_STROKE_WIDTH)
        final_intersection.set_fill(TEAL, FILL_OPACITY_BRIGHT)

        self.play(
            DrawBorderThenFill(final_intersection),
            circles.animate.set_fill(opacity=FILL_OPACITY_DIMMED),
            run_time=1.5
        )
        self.wait()

        # ========================================
        # REVERSE: Move circles apart
        # ========================================
        self.play(
            FadeOut(final_intersection),
            circles.animate.set_fill(opacity=FILL_OPACITY_NORMAL),
            run_time=1
        )

        self.play(
            moving_circle.animate.shift(5 * RIGHT),
            run_time=3,
            rate_func=smooth
        )
        self.wait()

        # ========================================
        # FINALE: No overlap
        # ========================================
        no_overlap_text = Text("No intersection when circles don't overlap", font_size=32)
        no_overlap_text.next_to(circles, DOWN, buff=1)
        no_overlap_text.set_color(GREY_A)

        self.play(FadeIn(no_overlap_text, shift=UP))
        self.wait(2)


# ============================================================
# 6. SCENE EXECUTION ORDER
# ============================================================
# The order of scenes that tells the story.
#
# Scene 1 (CircleIntersection):
#   - Introduces two overlapping circles
#   - Highlights the intersection region
#   - Teaches the concept of set intersection visually
#
# Scene 2 (CircleUnion):
#   - Shows the union of two circles
#   - Contrasts with intersection by showing the combined region
#   - Teaches the concept of set union visually
#
# Scene 3 (CompareIntersectionAndUnion):
#   - Side-by-side comparison of both operations
#   - Reinforces understanding through direct visual contrast
#   - Solidifies the distinction between intersection and union
#
# Scene 4 (DynamicCircleOverlap):
#   - Interactive demonstration of changing overlap
#   - Builds dynamic intuition for how overlap affects intersection
#   - Shows edge cases (no overlap, maximum overlap)

SCENE_ORDER = [
    CircleIntersection,           # Part 1: Introduce intersection
    CircleUnion,                  # Part 2: Introduce union
    CompareIntersectionAndUnion,  # Part 3: Direct comparison
    DynamicCircleOverlap,         # Part 4: Dynamic exploration
]
