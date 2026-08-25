from manim_imports_ext import *


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
