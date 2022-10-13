from manim_imports_ext import *


class VennDiagram(InteractiveScene):
    def construct(self):
        # Add circles
        radius = 3.0
        c1, c2 = circles = Circle(radius=radius).replicate(2)
        c1.set_stroke(BLUE, 3)
        c2.set_stroke(YELLOW, 3)
        c1.move_to(radius * LEFT / 2)
        c2.move_to(radius * RIGHT / 2)
        circles.to_edge(DOWN)
        self.add(circles)
        self.wait()

        # Labels
        l1 = Text("""
            People who take the
            geometry of keynote
            slides way too
            literally
        """, alignment="LEFT")
        l1.to_corner(UL)
        l2 = Text("""
            People who enjoy facts
            about 4-dimensional
            geometry
        """, alignment="RIGHT")
        l2.to_corner(UR)
        labels = VGroup(l1, l2)

        self.play(
            FadeIn(l1),
            c1.animate.set_fill(BLUE, 0.5)
        )
        self.wait()
        self.play(
            FadeIn(l2),
            c2.animate.set_fill(YELLOW, 0.5)
        )

        # Show centers
        rad_line = Line(c1.get_center(), c2.get_center())
        rad_line_label = Text("radius")
        rad_line_label.next_to(rad_line, UP, SMALL_BUFF)
        dot1, dot2 = dots = VGroup(*(Dot(c.get_center()) for c in circles))

        self.remove(l1, l2)
        circles.set_fill(opacity=0)
        self.add(dots, rad_line, rad_line_label)
        self.wait()

        # Ask about proportion
        arc = Arc(radius=radius, start_angle=TAU / 3, angle=TAU / 3, arc_center=c2.get_center())
        arc.set_stroke(YELLOW, 6)
        c2.match_style(c1)
        c1.set_stroke(GREY_B, 3)
        self.remove(rad_line_label)
        self.add(arc)
        self.wait()

        question = Text("What proportion of the circle?", font_size=60)
        question.to_corner(UL)
        question.to_edge(UP, buff=MED_SMALL_BUFF)
        arrow = Arrow(
            question.get_bottom() + LEFT,
            arc.pfp(0.3),
            buff=0.2,
            stroke_width=8,
            color=WHITE
        )

        self.add(question, arrow)
        self.wait()

        # Show hexagon
        hexagon = RegularPolygon(6)
        hexagon.replace(c2, dim_to_match=0)
        hexagon.add(*(
            Line(v1, v2)
            for v1, v2 in zip(hexagon.get_vertices()[:3], hexagon.get_vertices()[3:])
        ))
        hexagon.set_stroke(GREY_A, 2)
        answer = Tex("1 / 3", font_size=72, color=YELLOW)
        answer.next_to(question, RIGHT, buff=LARGE_BUFF)
        self.add(hexagon, answer)
        self.wait()


class Spheres(InteractiveScene):
    def construct(self):
        # Add spheres
        radius = 2.5
        s1, s2 = spheres = Sphere(radius=radius).replicate(2)
        s1.shift(radius * LEFT / 2)
        s2.shift(radius * RIGHT / 2)
        spheres.set_opacity(0.35)
        spheres.set_color(BLUE)
        meshes = VGroup()
        for sphere in spheres:
            sphere.always_sort_to_camera(self.camera)
            meshes.add(SurfaceMesh(sphere))
        meshes.set_stroke(width=1, opacity=0.25)

        cap = Sphere(v_range=(0, PI / 3), radius=radius)
        cap.rotate(90 * DEGREES, axis=UP, about_point=ORIGIN)
        cap.shift(s2.get_center())
        cap.set_color(YELLOW)
        cap.set_opacity(0.8)
        cap.always_sort_to_camera(self.camera)

        self.camera.frame.reorient(-20, 60)
        self.add(cap, spheres, meshes)
