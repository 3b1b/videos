import importlib.util

from manim_imports_ext import *


def _load_sibling(name):
    """
    Import a module from this file's directory. Needed since the directory
    name starts with a digit, so it cannot be reached by a dotted import.
    """
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parent / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_geography = _load_sibling("imo_2025_geography")

IMO_2025_PARTICIPANTS = _geography.IMO_2025_PARTICIPANTS
HOST_LATITUDE = _geography.HOST_LATITUDE
HOST_LONGITUDE = _geography.HOST_LONGITUDE


def lat_lon_to_uv(latitude, longitude):
    """
    Convert a (latitude, longitude) pair in degrees to the (u, v) parameters
    of a Sphere, whose uv_func sends u to the angle about the z-axis and v to
    the angle down from the north pole.

    A TexturedSurface built over that Sphere maps u linearly across the width
    of the image and v up its height, so this matches a standard equirect-
    angular map, with the left edge of the image at longitude 180W.
    """
    u = (longitude + 180) * DEGREES
    v = (latitude + 90) * DEGREES
    return (u, v)


def lat_lon_to_point(latitude, longitude, sphere):
    """
    Point on a given Sphere (or TexturedSurface over one) corresponding to a
    (latitude, longitude) pair in degrees.
    """
    u, v = lat_lon_to_uv(latitude, longitude)
    return sphere.uv_func(u, v) + sphere.get_center()


class FlightsFromAroundTheGlobe(InteractiveScene):
    def construct(self):
        # Add earth
        frame = self.frame
        light = self.camera.light_source
        light.move_to([10, 0, 5])
        radius = 3
        day_texture = "EarthTextureMap"
        night_texture = "NightEarthTextureMap"
        globe = TexturedSurface(Sphere(radius=3), day_texture, night_texture)
        mesh = SurfaceMesh(globe, resolution=(41, 21))
        mesh.set_stroke(WHITE, 0.5, 0.25)
        mesh.rotate(90 * DEG)

        frame.reorient(-163, 64, 0)
        self.add(globe, mesh)

        # Add start and end points
        student_height = 0.07
        dot_radius = 0.025
        dot_glow_factor = 0.1
        students = Group()
        for country, n_participants, lat, lon, color in IMO_2025_PARTICIPANTS:
            point = lat_lon_to_point(lat, lon, globe)
            for n in range(n_participants):
                randy = Randolph(mode="pondering", color=color, height=student_height)
                randy.rotate(angle_of_vector(point) + 90 * DEG)
                randy.move_to(globe.get_zenith())
                randy.apply_matrix(z_to_vector(normalize(point)), about_point=ORIGIN)
                students.add(randy)
        target_point = lat_lon_to_point(HOST_LATITUDE, HOST_LONGITUDE, globe)
        students.shuffle()
        students.apply_depth_test()

        arcs = VGroup(
            self.get_arc_between_points(dot.get_center(), target_point).match_color(dot)
            for dot in students
        )
        arcs.set_stroke(width=1, opacity=0.5)
        arcs.apply_depth_test()

        self.add(students)

        # Animate
        lag_ratio = 1 / len(students)
        start_orientation = frame.get_euler_angles() / DEG
        target_orientation = np.array([64, 110, 0])
        end_dot = GlowDot(target_point, color="#FFCD00")
        location_title = Text("Sunshine Coast, Australia", font_size=24)
        location_title.add_updater(lambda m: m.next_to(frame.to_fixed_frame_point(end_dot.get_center()), UR, SMALL_BUFF))
        location_title.fix_in_frame()

        self.play(
            LaggedStart(
                (MoveAlongPath(student, arc)
                for student, arc in zip(students, arcs)),
                group=students,
                lag_ratio=lag_ratio,
            ),
            LaggedStart(
                (VShowPassingFlash(arc.copy().set_stroke(width=2, opacity=0.5), time_width=2)
                for arc in arcs),
                lag_ratio=lag_ratio
            ),
            UpdateFromAlphaFunc(
                frame,
                lambda m, a: m.reorient(
                    *interpolate(start_orientation, target_orientation, a),
                    interpolate(ORIGIN, np.array((-0.13, 0.06, -0.39)), a),
                    interpolate(11, 6, a),
                ),
            ),
            Rotate(light, -75 * DEG, axis=OUT, about_point=light.get_z() * OUT),
            FadeIn(end_dot, time_span=(6, 12)),
            VFadeIn(location_title, time_span=(6, 12)),
            run_time=12
        )
        self.play(FadeOut(students))
        self.wait()

    def get_arc_between_points(self, p1, p2, n_anchors=100):
        lerp_points = np.array([
            interpolate(p1, p2, alpha)
            for alpha in np.linspace(0, 1, n_anchors)
        ])
        radius = get_norm(p1)
        normalized_points = np.array([radius * normalize(p) for p in lerp_points])
        result = VMobject().set_points_smoothly(
            lerp_points + 1.2 * (normalized_points - lerp_points)
        )
        return result


class ProblemIntro(InteractiveScene):
    def construct(self):
        # Test
        folder = Path(
            self.file_writer.get_output_file_rootname().parent.parent,
            "Assets"
        )
        pages = Group(
            ImageMobject(Path(folder / "IMO-2025-problems-eng-1.png")),
            ImageMobject(Path(folder / "IMO-2025-problems-eng-2.png")),
        )
        pages.set_height(7).arrange(RIGHT, buff=LARGE_BUFF)

        # Test 2
        p6_statement = Text(
            """
            Problem 6.   Consider a 2025 x 2025 grid of unit squares. Matilda wishes to place on the grid some
            rectangular tiles, possibly of different sizes, such that each side of every tile lies on a grid line and
            every unit square is covered by at most one tile.

            Determine the minimum number of tiles Matilda needs to place so that each row and each column
            of the grid has exactly one unit square that is not covered by any tile.
            """,
            alignment="LEFT",
        )
        p6_statement.set_fill(border_width=0)
        p6_statement["Problem 6."].set_fill(border_width=0.5)
        p6_statement[re.compile(r"Determine.*\n.*")].shift(0.35 * UP)
        pre_size_text = p6_statement["2025 x 2025"]
        size_text = Tex(R"2025 \times 2025")
        size_text.replace(pre_size_text)
        for m1, m2 in zip(pre_size_text.family_members_with_points(), size_text.family_members_with_points()):
            m1.set_points(m2.get_points())
        p6_statement.set_width(4.25)
        p6_statement.move_to((2.99, -0.75, 0), UP)
        back_rect = SurroundingRectangle(p6_statement)
        back_rect.set_stroke(width=0)
        back_rect.set_fill(WHITE, 1)
        p6_statement.set_fill(BLACK)

        self.add(pages, back_rect, p6_statement)
        self.wait()

        # Fade colors
        parts = VGroup(
            p6_statement[text]
            for text in [
                "Problem 6.",
                "Consider a 2025 x 2025 grid of unit squares",
                "Matilda wishes to place on the grid some",
                "rectangular tiles, possibly of different sizes,",
                "such that each side of every tile lies on a grid line and",
                "every unit square is covered by at most one tile.",
                "Determine the minimum number of tiles Matilda needs to place",
                "so that each row",
                "and each column",
                "of the grid has exactly one unit square that is not covered by any tile.",
            ]
        ).copy()
        parts.set_fill(WHITE)
        self.play(
            self.frame.animate(run_time=3).reorient(0, 0, 0, (2.98, -1.9, 0.00), 2.55),
            pages.animate(run_time=3).set_opacity(opacity=0.0),
            back_rect.animate(time_span=(1, 3)).set_fill(BLACK),
            FadeOut(p6_statement, time_span=(1, 3)),
            FadeIn(parts, time_span=(1, 3), lag_ratio=3e-3),
        )
        # parts.set_backstroke(BLACK, 2)
        self.remove(back_rect)
        self.wait()

        # Highlight part-by-part
        last_highlights = VGroup()

        def highlight(indices, other_opacity=0.2, last_highlights=last_highlights):
            highlights = VGroup(
                # Underline(parts[n], buff=-0.01, stretch_factor=1).set_stroke(TEAL, 3)
                SurroundingRectangle(parts[n], buff=0.015).set_fill(TEAL_D, 0.5).set_stroke(width=0)
                for n in indices
            )
            highlights.save_state()
            for highlight in highlights:
                highlight.stretch(0, 0, about_edge=LEFT)
                highlight.set_opacity(0)
            highlights.set_z_index(-1)
            result = AnimationGroup(
                Restore(highlights, lag_ratio=0.5),
                *(parts[n].animate.set_fill(opacity=1, border_width=1) for n in indices),
                *(FadeOut(highlight) for highlight in last_highlights),
                *(parts[n].animate.set_fill(border_width=0) for n in range(0, indices[0])),
                run_time=2
            )
            last_highlights.set_submobjects(highlights)
            return result

        self.play(
            highlight([1]),
            parts[2:6].animate.set_opacity(0.2),
            parts[6:].animate.set_opacity(0),
        )
        self.wait()
        self.play(highlight([2, 3]))
        self.wait()
        self.play(highlight([4, 5]))
        self.wait()
        self.play(
            self.frame.animate(run_time=2).set_y(-2.37),
            highlight([6]),
            parts[7:].animate.set_opacity(0.2),
        )
        self.wait()
        self.play(
            highlight([7, 9])
        )
        self.wait(2)
        highlight = last_highlights[0]
        self.play(
            highlight.animate.surround(parts[8], buff=0.015),
            parts[7].animate.set_fill(border_width=0),
            parts[8].animate.set_fill(opacity=1, border_width=1),
        )
        self.wait()


class Timeline(InteractiveScene):
    def construct(self):
        timeline = NumberLine(
            (2017, 2030, 1 / 12),
            unit_size=4,
            tick_size=0.1,
            longer_tick_multiple=2,
            big_tick_spacing=1,
        )
        timeline.set_y(-2)
        year_labels = timeline.add_numbers(
            range(2017, 2030),
            group_with_commas=False,
        )

        def update_year_label(label):
            year = label.get_value()
            label.next_to(timeline.n2p(year), DOWN, 0.4)
            focal_value = np.exp(-0.2 * label.get_x()**2)
            label.set_height(
                0.25 + 0.25 * focal_value,
                about_edge=UP
            )
            label.set_fill(opacity=(0.5 + 0.5 * focal_value))

        for label in year_labels:
            label.add_updater(update_year_label)

        def center_on_year(year, run_time=2):
            shift_value = timeline.n2p(year)[0] * LEFT
            return timeline.animate(run_time=run_time).shift(shift_value)

        self.add(timeline)
        self.play(center_on_year(2026))
        self.wait()
        self.play(center_on_year(2023))
        self.wait()
        self.play(center_on_year(2025))


# Mathlib's formalization of IMO 2024 Q3, whose module docstring states the problem in
# English, and whose source both states it in Lean and proves it
LEAN_FILE = Path(__file__).parent / "Imo2024Q3.lean"
# Consolas, which Code reaches for by default, has none of ℕ ↦ ∀ ∈ ⟨ ⟩; Menlo carries every
# glyph this file uses but ⊔ and ⋃, so the source stays in one font at one width
LEAN_FONT = "Menlo"


def lean_slice(source, start, end=None):
    """
    The stretch of a Lean source from where one snippet of it appears up to where another
    does, or to the end of it where no end is named.
    """
    index = source.index(start)
    return source[index:source.index(end, index) if end else len(source)].strip("\n")


def lean_code(code, **kwargs):
    """
    Lean source, highlighted as the Lean 4 it is: pygments answers to the name "lean" with
    its Lean 3 lexer, which leaves this file's declarations unhighlighted.
    """
    return Code(code, language="lean4", font=LEAN_FONT, alignment="LEFT", **kwargs)


def source_blocks(code):
    """Each paragraph of a source, as the line it begins on and the lines themselves"""
    blocks = []
    lines = []
    for number, line in enumerate(code.split("\n")):
        if line.strip():
            lines.append(line)
        elif lines:
            blocks.append((number - len(lines), "\n".join(lines)))
            lines = []
    if lines:
        blocks.append((number + 1 - len(lines), "\n".join(lines)))
    return blocks


class SourceColumn(VGroup):
    """
    Every glyph of a source, which moves as one whether or not all of it is showing.

    An animation which reveals a group a piece at a time takes whatever it has not reached
    yet out of the group as it goes, see ShowIncreasingSubsets. A group scrolled while that
    runs would move only what is showing and leave the rest where it stood: each glyph would
    start moving as it appeared, and every line would break wherever the reveal had got to,
    its tail a line below its head. So a shift here moves every glyph, showing or not.
    """

    def __init__(self, *glyphs):
        super().__init__(*glyphs)
        self.glyphs = list(glyphs)

    def shift(self, vector):
        for glyph in self.glyphs:
            glyph.shift(vector)
        return self


def lean_column(code, font_size=36):
    """
    Lean source as a column of blocks, a paragraph of it to each, rather than as one mobject
    holding the lot. Text is laid out on a canvas of fixed size, which a thousand lines at a
    readable font size overrun: what does not fit is dropped, and what is left comes back
    folded into the room there was, which reads as a source full of stray line breaks.

    Each block is put where its own line numbers say, so the blank lines between paragraphs
    and the indentation within them come out as the file has them. What fixes where a block
    begins is a bar given a line of its own in front of it, and taken away once it has: the
    box around a block's glyphs reaches as far as its tallest, which differs from paragraph
    to paragraph, so stacking those boxes would leave each one a little out.

    Comes back flat, a glyph to a submobject, as a single Code mobject would, so that
    showing it a piece at a time shows it a glyph at a time.
    """
    glyph = lean_code("x", font_size=font_size)
    line_height = lean_code("x\nx", font_size=font_size).get_height() - glyph.get_height()
    column = VGroup()
    for number, block in source_blocks(code):
        mob = lean_code("|\n" + block, font_size=font_size)
        anchor = mob[0]
        mob.shift((number - 1) * line_height * DOWN - anchor.get_corner(DL))
        mob.remove(anchor)
        column.add(mob)
    return SourceColumn(*column.family_members_with_points())


class Lean(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        source = LEAN_FILE.read_text()
        english = self.get_english_statement(source)
        statement = self.get_lean_statement(source)
        proof = self.get_lean_proof(source)

        titles = VGroup(
            Text("Problem\n(English)"),
            Text("Problem\n(Lean)").set_color(TEAL),
            Text("Proof\n(Lean)").set_color(TEAL),
        )
        pieces = VGroup(english, statement, proof)
        for title, piece, x in zip(titles, pieces, [-1, 0, 1]):
            title.set_x(x * FRAME_WIDTH / 3)
            title.to_edge(UP, buff=MED_LARGE_BUFF)
            piece.set_width(4)
            piece.next_to(title, DOWN, LARGE_BUFF)

        arrow = Arrow(titles[0], titles[1], thickness=6, buff=0.3)
        arrow.set_fill((WHITE, TEAL), gradient_direction=RIGHT)

        # Show initial translation
        frame.reorient(0, 0, 0, (-2.30, 1.76, 0.00), 5.71)
        self.add(titles[0], pieces[0])
        self.play(LaggedStart(
            GrowArrow(arrow),
            TransformMatchingStrings(
                titles[0].copy(), titles[1],
                key_map={"English": "Lean"},
                run_time=1
            ),
            TransformFromCopy(english, statement, lag_ratio=1e-4),
            run_time=2,
            lag_ratio=0.25
        ))
        self.wait(2)

        # Show proof
        arrow2 = Arrow(titles[1], titles[2], thickness=6, buff=0.3)
        arrow2.set_fill(TEAL)

        back_rect = VGroup(
            Rectangle(proof.get_width() + 0.5, 1.5).set_fill(BLACK, 1),
            Rectangle(proof.get_width() + 0.5, 1.5).set_fill(BLACK, opacity=(0, 1), gradient_direction=UP),
        )
        back_rect.arrange(DOWN, buff=0)
        back_rect.move_to(proof).to_edge(UP, buff=0)
        back_rect.set_stroke(width=0)

        proof.set_z_index(-1)
        proof.add_updater(lambda m, dt: m.shift(2 * dt * UP))

        self.add(back_rect)
        self.play(
            GrowArrow(arrow2),
            TransformMatchingStrings(titles[1].copy(), titles[2], run_time=1),
            ShowIncreasingSubsets(proof, run_time=10, rate_func=linear),
            frame.animate.to_default_state(),
        )
        self.wait()

    def get_english_statement(self, source):
        statement = lean_slice(source, "Let $a_1", "We follow Solution 1")
        pre_result = VGroup(*(
            TexText(" ".join(paragraph.split()), alignment="", font_size=36)
            for paragraph in statement.split("\n\n")
        ))
        pre_result.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        return VGroup(*pre_result.family_members_with_points())

    def get_lean_statement(self, source):
        definitions = lean_slice(source, "/-- The condition of the problem.", "/-! ###")
        claim, _ = self.get_theorem(source)
        return lean_code("\n\n".join([definitions, claim]))

    def get_lean_proof(self, source):
        development = lean_slice(source, "/-! ### Definitions", "theorem result")
        _, tactics = self.get_theorem(source)
        return lean_column("\n\n".join([development, tactics]))

    def get_theorem(self, source):
        """The final theorem, as what it claims and how it is proved"""
        theorem = lean_slice(source, "theorem result", "end Imo2024Q3")
        claim, _, tactics = theorem.partition(":= by")
        return claim.strip() + " := by", tactics.strip("\n")


class KScaling(InteractiveScene):
    def construct(self):
        # Expression
        expression = Tex(R"k^2 + 2k - 3", font_size=60)
        expression.to_edge(UP)

        k_tracker = ValueTracker(5)
        get_k = lambda: int(k_tracker.get_value())

        self.add(expression)

        # Blocks
        block_size = 0.15
        block_template = Square(side_length=block_size).set_stroke(WHITE, 1).set_fill(BLUE, 1)
        block_template = VGroup(
            Square(block_size).set_stroke(WHITE, 1).set_fill(BLUE, 0.5),
            Cross(Square(0.8 * block_size)).set_stroke(RED, 2)
        )
        block_top = 1.0 * UP

        def get_k_squared_blocks():
            k = int(get_k())
            squares = block_template.get_grid(k, k)
            squares.set_max_width(4.25)
            squares.move_to(4.5 * LEFT + block_top, UP).shift_onto_screen()
            return squares

        def get_k_blocks():
            k = int(get_k())
            squares = block_template.get_grid(2, k)
            squares.set_max_width(4.25)
            squares.move_to(RIGHT).shift_onto_screen()
            squares.align_to(block_top, UP)
            return squares

        k_squared_blocks = get_k_squared_blocks()
        k_blocks = get_k_blocks()

        neg_3 = block_template.get_grid(1, 3)
        neg_3.set_stroke(RED, 2).set_fill(opacity=0)
        neg_3.move_to(5.5 * RIGHT).align_to(k_squared_blocks, UP)

        # Add k label
        k_label = Tex(R"k = 0")
        k_label.to_edge(UP, buff=0.25)
        num_part = k_label.make_number_changeable("0", edge_to_fix=LEFT)
        num_part.add_updater(lambda m: m.set_value(get_k()))

        number_line = NumberLine(
            (0, 50),
            tick_size=0.05,
            longer_tick_multiple=2,
            big_tick_spacing=10,
            width=6
        )
        number_line.to_edge(UP, buff=0.75)
        k_tip = ArrowTip(angle=-90 * DEG).set_height(0.2)
        k_tip.set_fill(TEAL)
        k_tip.add_updater(lambda m: m.move_to(number_line.n2p(get_k()), DOWN))
        k_label.add_updater(lambda m: m.next_to(k_tip, UP, SMALL_BUFF, aligned_edge=LEFT))

        # Animate
        terms = [expression["k^2"], expression["+ 2k"], expression["- 3"]]
        self.play(
            LaggedStart(
                terms[0].animate.next_to(k_squared_blocks, UP),
                terms[1].animate.next_to(k_blocks, UP),
                terms[2].animate.next_to(neg_3, UP),
                lag_ratio=0.5,
            ),
            LaggedStart(
                ShowIncreasingSubsets(k_squared_blocks, suspend_mobject_updating=True),
                ShowIncreasingSubsets(k_blocks, suspend_mobject_updating=True),
                ShowIncreasingSubsets(neg_3),
                lag_ratio=0.5,
            ),
            FadeIn(k_tip, time_span=(1, 2)),
            FadeIn(k_label, time_span=(1, 2)),
            FadeIn(number_line, time_span=(1, 2)),
            run_time=2
        )

        self.play(
            k_tracker.animate.set_value(30),
            UpdateFromFunc(k_squared_blocks, lambda m: m.become(get_k_squared_blocks())),
            UpdateFromFunc(k_blocks, lambda m: m.become(get_k_blocks())),
            run_time=5
        )
        self.wait()

        # Highlight parts
        frame = self.frame
        rect = SurroundingRectangle(VGroup(expression["k^2"], k_squared_blocks))
        k = get_k()

        edges1 = VGroup(
            Line(
                block.get_corner(DR),
                block.get_corner(UR)
            )
            for block in k_squared_blocks
        )
        edges2 = VGroup(
            VMobject().set_points_as_corners([
                block.get_corner(DR),
                block.get_corner(UR),
                block.get_corner(UL),
            ])
            for block in k_squared_blocks[:2 * k]
        )
        edges1.set_stroke(YELLOW, 4)
        edges2.set_stroke(YELLOW, 4)

        k_squared_blocks.target = k_squared_blocks.generate_target()
        for block in k_squared_blocks.target:
            block[0].set_opacity(0.25)
            block[1].set_stroke(width=1)

        self.play(
            MoveToTarget(k_squared_blocks),
            FadeIn(edges1, lag_ratio=0.01, time_span=(1, 3)),
            frame.animate.reorient(0, 0, 0, (-4.52, 0.34, 0.00), 2.83),
            terms[0].animate.scale(0.75, about_edge=DOWN),
            run_time=3,
        )
        self.wait()

        terms[0].target = terms[0].generate_target()
        terms[0].target.shift(0.25 * LEFT)
        terms[1].target = terms[1].generate_target()
        terms[1].target.scale(0.75, about_edge=DOWN)
        terms[1].target.next_to(terms[0].target, RIGHT, SMALL_BUFF, aligned_edge=DOWN)

        k_squared_blocks.target = k_squared_blocks.generate_target()
        for block in k_squared_blocks.target[:2 * k]:
            block[0].set_opacity(0.5)

        self.play(
            MoveToTarget(terms[0]),
            MoveToTarget(terms[1]),
            MoveToTarget(k_squared_blocks),
            VGroup(k_squared_blocks[2 * k:], edges1[2 * k:]).animate.shift(0.25 * DOWN),
            run_time=2,
        )
        self.play(
            ShowCreation(edges2, lag_ratio=0.01),
            k_blocks.animate.set_opacity(0.25)
        )
        self.wait(2)


class LuongQuote(InteractiveScene):
    def construct(self):
        # Add the quote
        quote = Text(
            """
            “We didn’t really have a way to teach
            the model to be patient. It didn’t take
            the time to understand the problem,
            to get a feel for the problem,
            to not try to solve the problem.”
            """,
            alignment="left"
        ).set_color(YELLOW)
        quote_bg = quote.copy().set_color("#111111")
        self.add(quote_bg)
        self.play(FadeIn(quote["""“We didn’t really have a way to teach
            the model to be patient."""], lag_ratio=0.1, run_time=3))
        self.wait(0.2)
        self.play(FadeIn(quote["""It didn’t take
            the time to understand the problem,"""], lag_ratio=0.1, run_time=2.5),
                  quote["""“We didn’t really have a way to teach
            the model to be patient."""].animate.set_color(WHITE)
                  )
        self.play(
            FadeIn(quote["""to get a feel for the problem,"""], lag_ratio=0.1, run_time=1.5),
            quote["""It didn’t take
            the time to understand the problem,"""].animate.set_color(WHITE)
        )
        self.wait(0.1)
        self.play(
            FadeIn(quote["""to not try to solve the problem.”"""], lag_ratio=0.1, run_time=1.5),
            quote["""to get a feel for the problem,"""].animate.set_color(WHITE)
        )
        self.wait(0.1)
        self.play(quote["""to not try to solve the problem.”"""].animate.set_color(WHITE))