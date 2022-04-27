from manim_imports_ext import *


def get_set_tex(values, max_shown=7, **kwargs):
    if len(values) > max_shown:
        value_mobs = [
            *map(Integer, values[:max_shown - 2]),
            MTex("\\dots"),
            Integer(values[-1]),
        ]
    else:
        value_mobs = list(map(Integer, values))

    commas = MTex(",").replicate(len(value_mobs) - 1)
    result = VGroup()
    result.add(MTex("\\{"))
    result.add(*it.chain(*zip(value_mobs, commas)))
    if len(value_mobs) > 0:
        result.add(value_mobs[-1].align_to(value_mobs[0], UP))
    result.add(MTex("\\}"))
    result.arrange(RIGHT, buff=SMALL_BUFF)
    if len(values) > 0:
        commas.set_y(value_mobs[0].get_y(DOWN))
    if len(values) > max_shown:
        result[-4].match_y(commas)
    result.values = values
    return result


def get_part_by_value(set_tex, value):
    try:
        return next(sm for sm in set_tex if isinstance(sm, Integer) and sm.get_value() == value)
    except StopIteration:
        return VMobject().move_to(set_tex)


def get_brackets(set_tex):
    return VGroup(set_tex[0], set_tex[-1])


def get_integer_parts(set_tex):
    result = VGroup(*(
        sm for sm in set_tex
        if isinstance(sm, Integer)
    ))
    if len(result) == 0:
        result.move_to(set_tex)
    return result


def get_commas(set_tex):
    result = set_tex[2:-1:2]
    if len(result) == 0:
        result.move_to(set_tex)
    return result


def set_tex_transform(set_tex1, set_tex2):
    bracket_anim = TransformFromCopy(
        get_brackets(set_tex1),
        get_brackets(set_tex2),
    )
    matching_anims = [
        TransformFromCopy(
            get_part_by_value(set_tex1, value),
            get_part_by_value(set_tex2, value),
        )
        for value in filter(
            lambda v: v in set_tex2.values,
            set_tex1.values,
        )
    ]
    mismatch_animations = [
        FadeInFromPoint(
            get_part_by_value(set_tex2, value),
            set_tex1.get_center()
        )
        for value in set(set_tex2.values).difference(set_tex1.values)
    ]
    anims = [bracket_anim, *matching_anims, *mismatch_animations]
    if len(set_tex2.values) > 1:
        commas = []
        for st in set_tex1, set_tex2:
            if len(st.values) > 1:
                commas.append(st[2:-1:2])
            else:
                commas.append(MTex(",").set_opacity(0).move_to(st, DOWN))
        comma_animations = TransformFromCopy(*commas)
        anims.append(comma_animations)
    return AnimationGroup(*anims)


def get_sum_group(set_tex, sum_color=YELLOW):
    height = set_tex.get_height()
    buff = 0.75 * height
    arrow = Vector(height * RIGHT)
    arrow.next_to(set_tex, RIGHT, buff=buff)
    sum_value = Integer(sum(set_tex.values))
    sum_value.set_color(sum_color)
    sum_value.set_height(0.66 * height)
    sum_value.next_to(arrow, RIGHT, buff=buff)

    return VGroup(arrow, sum_value)


def get_sum_animation(set_tex, sum_group, path_arc=-10 * DEGREES):
    arrow, sum_value = sum_group

    return AnimationGroup(
        GrowArrow(arrow),
        FadeTransform(
            get_integer_parts(set_tex).copy(),
            sum_value,
            path_arc=path_arc,
        ),
    )


def get_subset_highlights(set_tex, subset, stroke_color=YELLOW, stroke_width=2):
    result = VGroup()
    for value in subset:
        if value not in set_tex.values:
            continue
        rect = SurroundingRectangle(
            set_tex.get_part_by_tex(str(value)),
            buff=0.05,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
        )
        rect.round_corners(radius=rect.get_width() / 4)
        result.add(rect)
    return result


def get_subsets(full_set):
    return list(it.chain(*(
        it.combinations(full_set, k)
        for k in range(len(full_set) + 1)
    )))


def subset_to_int(subset):
    return sum(2**(v - 1) for v in subset)


def subset_sum_generating_function(full_set):
    pass


# Scenes

class ExampleWith5(InteractiveScene):
    def construct(self):
        # Show all subsets
        N = 5
        full_set = list(range(1, N + 1))

        set_tex = get_set_tex(full_set)
        set_tex.to_edge(UP)

        self.play(FadeIn(get_brackets(set_tex), scale=0.7))
        self.play(
            ShowIncreasingSubsets(get_integer_parts(set_tex)),
            Write(get_commas(set_tex)),
        )
        self.add(set_tex)
        self.wait()
        subsets = self.count_all_subsets(set_tex)

        # Show n choose k stacks
        stacks = self.get_subset_stacks(full_set)
        sum_stacks = self.get_subset_sums(stacks)

        anims = []
        for stack in stacks:
            for new_subset in stack:
                for subset in subsets:
                    if set(subset.values) == set(new_subset.values):
                        anims.append(FadeTransform(subset, new_subset))

        self.play(LaggedStart(*anims))
        self.wait()

        # Show their sums
        covered_sums = []
        for n in range(4):
            self.wait(note=f"Example sum {n} / 4")
            # Show sum based on what's in self.selection
            anims = []
            for stack, sum_stack in zip(stacks, sum_stacks):
                for subset, sum_mob in zip(stack, sum_stack):
                    if set(subset.get_family()).intersection(self.selection.get_family()):
                        if sum_mob not in covered_sums:
                            covered_sums.append(sum_mob)
                            anims.append(get_sum_animation(subset, sum_mob))
            self.clear_selection()
            self.play(LaggedStart(*anims))
        self.add(sum_stacks)

        # Isolate counts we care about
        self.highlight_multiple_of_5(stacks, sum_stacks)
        self.group_by_sum(stacks, sum_stacks)

        # Show generating function
        self.show_generating_function(set_tex)
        self.transition_to_full_generating_function(set_tex)

    def count_all_subsets(self, set_tex):
        equation = MTex("2 \\cdot 2 \\cdot 2 \\cdot 2 \\cdot 2 = 2^5 = 32")
        equation.set_width(4)
        equation.to_corner(UL)
        equation.set_color(YELLOW)
        subsets = VGroup(get_set_tex([]))
        for value in range(1, 6):
            value_mob = get_part_by_value(set_tex, value)
            marks = VGroup(Exmark(), Checkmark())
            marks.match_height(value_mob)
            marks.next_to(value_mob, DOWN)

            subsets.generate_target()
            new_subsets = VGroup(
                *subsets.target,
                *(
                    get_set_tex([*ss.values, value]).move_to(ss)
                    for ss in subsets
                ),
            )
            new_subsets.arrange_in_grid(
                n_rows=[1, 1, 2, 4, 4][value - 1],
                buff=MED_LARGE_BUFF
            )
            new_subsets.set_max_width(FRAME_WIDTH - 1)

            self.add(equation[:2 * value - 1])
            self.wait()
            self.add(marks[0])
            if value == 1:
                self.play(FadeIn(subsets, scale=0.5))
            else:
                self.play(LaggedStart(*(
                    FlashAround(ss, color=RED)
                    for ss in subsets
                )), lag_ratio=1 / len(subsets))
            self.wait()
            self.remove(subsets)
            self.play(
                FadeOut(marks[0], 0.5 * UP),
                FadeIn(marks[1], 0.5 * UP),
                MoveToTarget(subsets),
                *(
                    set_tex_transform(st1, st2)
                    for st1, st2 in zip(subsets, new_subsets[len(subsets):])
                )
            )
            self.add(new_subsets)
            self.wait()
            self.remove(marks[1])
            subsets.set_submobjects(list(new_subsets))
        self.play(
            Write(equation[9:]),
        )
        self.wait()
        self.play(FadeOut(equation))

        return subsets

    def highlight_multiple_of_5(self, stacks, sum_stacks):
        rects = VGroup()
        anims = []
        for stack, sum_stack in zip(stacks, sum_stacks):
            for set_tex, sum_group in zip(stack, sum_stack):
                if sum(set_tex.values) % 5 == 0:
                    rect = SurroundingRectangle(VGroup(set_tex, sum_group))
                    rect.value = sum(set_tex.values)
                    rects.add(rect)
                else:
                    anims.append(set_tex.animate.set_opacity(0.25))
                    anims.append(sum_group.animate.set_opacity(0.25))
        rects.set_stroke(TEAL, 2)
        for rect in rects:
            rect.round_corners()

        counter = Integer(0, font_size=72)
        counter.to_corner(UR)
        counter.set_color(TEAL)

        self.play(*anims, run_time=1)
        self.wait()
        self.play(
            FadeIn(rects, lag_ratio=0.9),
            ChangeDecimalToValue(counter, len(rects)),
            run_time=1.5
        )
        self.wait()

        self.highlight_rects = rects
        self.counter = counter

    def group_by_sum(self, stacks, sum_stacks):
        # Lock sums to subsets
        subset_groups = VGroup()
        for stack, sum_stack in zip(stacks, sum_stacks):
            for set_tex, sum_group in zip(stack, sum_stack):
                set_tex.sum_group = sum_group
                sum_group.set_tex = set_tex
                subset_groups.add(VGroup(set_tex, sum_group))

        # Reorganize
        common_sum_stacks = VGroup()
        max_sum = max(sum(ssg[0].values) for ssg in subset_groups)
        for n in range(max_sum + 1):
            stack = VGroup(*filter(
                lambda ssg: sum(ssg[0].values) == n,
                subset_groups
            ))
            common_sum_stacks.add(stack)

        common_sum_stacks.generate_target()
        csst = common_sum_stacks.target
        for stack in common_sum_stacks.target:
            stack.arrange(DOWN, aligned_edge=RIGHT, buff=SMALL_BUFF)

        csst.arrange_in_grid(4, 5, buff=MED_LARGE_BUFF, aligned_edge=RIGHT)
        csst[10:15].set_y(np.mean([csst[5].get_y(DOWN), csst[15].get_y(UP)]))
        csst.refresh_bounding_box()
        csst.set_width(FRAME_WIDTH - 1)
        csst.to_corner(DL)
        csst.set_opacity(1)

        # Create new rectangles
        common_sum_rects = VGroup()
        for stack in common_sum_stacks.target:
            rect = SurroundingRectangle(stack, buff=SMALL_BUFF)
            rect.round_corners(radius=0.05)
            rect.value = sum(stack[0][0].values)
            color = TEAL if rect.value % 5 == 0 else GREY_B
            rect.set_stroke(color, 1)
            common_sum_rects.add(rect)

        rect_anims = []
        for highlight_rect in self.highlight_rects:
            for rect in common_sum_rects:
                if rect.value == highlight_rect.value:
                    rect_anims.append(Transform(highlight_rect, rect))

        # Transition to common sum
        self.play(
            FadeOut(self.counter),
            MoveToTarget(common_sum_stacks),
            *rect_anims,
            run_time=2
        )
        self.play(
            FadeOut(self.highlight_rects),
            FadeIn(common_sum_rects),
        )
        self.wait()

        self.subset_groups = subset_groups
        self.common_sum_stacks = common_sum_stacks
        self.common_sum_rects = common_sum_rects

    def show_generating_function(self, set_tex):
        # Setup expressions
        css = self.common_sum_stacks
        csr = self.common_sum_rects
        subset_group = self.subset_group = VGroup(csr, css)

        factored_terms = "(1 + x)", "(1 + x^2)", "(1 + x^3)", "(1 + x^4)", "(1 + x^5)"
        factored = MTex("".join(factored_terms), isolate=factored_terms)
        expanded_terms = ["1"]
        for n in range(1, 16):
            k = len(css[n])
            expanded_terms.append((str(k) if k > 1 else "") + f"x^{{{n}}}")
        expanded = MTex("+".join(expanded_terms), isolate=["+", *expanded_terms])
        expanded.set_width(FRAME_WIDTH - 1)
        factored.next_to(set_tex, DOWN, MED_LARGE_BUFF)
        expanded.next_to(factored, DOWN, MED_LARGE_BUFF)

        self.play(Write(factored))
        self.wait()
        self.play(subset_group.animate.set_height(3.0, about_edge=DOWN))

        # Animate expansion
        fac_term_parts = [factored.get_part_by_tex(term) for term in factored_terms]
        expanded_terms = [expanded.get_part_by_tex(term) for term in expanded_terms]
        super_expanded = VGroup()
        super_expanded.next_to(factored, DOWN, MED_LARGE_BUFF)
        collection_anims = []
        for bits in it.product(*5 * [[0, 1]]):
            bits = list(reversed(bits))
            top_terms = [
                part[3:-1] if bit else part[1]
                for bit, part in zip(bits, fac_term_parts)
            ]
            top_rects = VGroup(*(
                # VHighlight(part, color_bounds=(BLUE, BLUE_E), max_stroke_addition=3.0)
                SurroundingRectangle(part).set_stroke(BLUE, 2).round_corners()
                for part in top_terms
            ))
            n = sum(b * k for k, b in zip(range(1, 6), bits))
            if n == 0:
                new_term = MTex("1", font_size=36)
                super_expanded.add(new_term)
            else:
                new_plus = MTex("+", font_size=36)
                new_term = MTex(f"x^{{{n}}}", font_size=36)
                super_expanded.add(new_plus, new_term)
                collection_anims.append(FadeOut(new_plus))
            super_expanded.arrange(RIGHT, aligned_edge=DOWN, buff=SMALL_BUFF)
            super_expanded.next_to(factored, DOWN, MED_LARGE_BUFF)
            super_expanded.to_edge(LEFT)
            if len(super_expanded) > 33:
                super_expanded[33:].next_to(
                    super_expanded[0], DOWN, MED_LARGE_BUFF, aligned_edge=LEFT
                )
            low_rect = SurroundingRectangle(new_term, buff=0.5 * SMALL_BUFF)
            low_rect.set_stroke(BLUE, 2).round_corners()
            collection_anims.append(
                FadeTransform(new_term, expanded_terms[n], path_arc=45 * DEGREES)
            )

            self.add(top_rects)
            self.add(super_expanded, low_rect)
            self.wait()
            self.remove(top_rects, low_rect)
        self.wait()

        # Reorganize to expanded
        self.play(
            LaggedStart(*collection_anims, path_arc=10 * DEGREES),
            LaggedStartMap(FadeIn, expanded.get_parts_by_tex("+")),
            ApplyMethod(
                subset_group.set_height, 4.0, dict(about_edge=DOWN),
                rate_func=squish_rate_func(smooth, 0.5, 1.0)
            ),
            run_time=3,
        )
        self.add(expanded)
        self.wait(note="Highlight multiples of 5")

        self.factored_func = factored
        self.expanded_func = expanded

    def transition_to_full_generating_function(self, set_tex):
        # Expressions
        factored = MTex(
            "f(x) = (1 + x)(1 + x^2)(1 + x^3) \\cdots \\left(1 + x^{1{,}999}\\right)\\left(1 + x^{2{,}000}\\right)",
        )
        expanded = MTex(
            "f(x) = 1+x+x^{2}+2 x^{3}+2 x^{4}+ 3x^{5} +\\cdots + x^{2{,}001{,}000}",
            isolate="+",
        )
        new_set_tex = get_set_tex(range(1, 2001))
        new_set_tex.move_to(set_tex)

        h_line = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        h_line.set_stroke(GREY_C, 1)
        h_line.set_y(0.5)

        factored_word = Text("Factored", font_size=60, color=BLUE_B)
        factored_word.next_to(set_tex, DOWN, MED_LARGE_BUFF)
        factored.next_to(factored_word, DOWN, MED_LARGE_BUFF)
        expanded_word = Text("Expanded", font_size=60, color=TEAL)
        expanded_word.next_to(h_line, DOWN, LARGE_BUFF)
        expanded.next_to(expanded_word, DOWN, MED_LARGE_BUFF)
        for mob in [factored, factored_word, expanded, expanded_word]:
            mob.to_edge(LEFT)

        self.play(
            TransformMatchingShapes(set_tex, new_set_tex),
            FadeTransform(self.factored_func, factored),
            FadeIn(factored_word, scale=2.0),
            FadeOut(self.expanded_func, 2 * DOWN),
            FadeOut(self.subset_group, DOWN),
        )
        self.wait()
        self.play(
            ShowCreation(h_line),
            FadeIn(expanded_word),
            Write(expanded)
        )
        self.wait()

        # Show example term
        n = 25
        subsets = list(filter(
            lambda s: sum(s) == n,
            get_subsets(range(1, n))
        ))
        coef = len(subsets)
        term = Tex(str(coef), f"x^{{{n}}}", "+", "\\cdots")
        term[:2].set_color(TEAL)
        term[2:].set_color(WHITE)
        tail = expanded[-11:]
        term.move_to(tail, DL)
        tail.generate_target()
        tail.target.next_to(term, RIGHT, buff=0.15, aligned_edge=DOWN)

        self.play(
            Write(term),
            MoveToTarget(tail),
        )
        self.wait()

        subset_mobs = VGroup(*map(get_set_tex, subsets))
        subset_mobs.arrange_in_grid(n_cols=10)
        subset_mobs.set_width(FRAME_WIDTH - 1)
        subset_mobs.to_edge(UP)
        subset_mobs.set_color(TEAL)
        top_rect = FullScreenFadeRectangle()
        top_rect.set_fill(BLACK, opacity=0.9)
        top_rect.set_height(4, about_edge=UP, stretch=True)

        self.play(
            FadeIn(top_rect),
            ShowIncreasingSubsets(subset_mobs, run_time=5)
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeOut, subset_mobs, shift=0.2 * UP),
            FadeOut(top_rect, rate_func=squish_rate_func(smooth, 0.6, 1)),
            run_time=3
        )
        self.wait()

    ##

    def get_subset_stacks(self, full_set, buff=3.5):
        stacks = VGroup(*(
            VGroup(*(
                get_set_tex(subset)
                for subset in it.combinations(full_set, k)
            ))
            for k in range(len(full_set) + 1)
        ))
        for stack in stacks:
            stack.arrange(DOWN)
        stacks.arrange(RIGHT, buff=buff, aligned_edge=DOWN)

        stacks[0].move_to(stacks[1]).align_to(stacks[2], UP)
        stacks[4].align_to(stacks[3], UP)
        stacks[5].match_x(stacks[4])

        stacks.set_max_height(FRAME_HEIGHT - 3)
        stacks.set_max_width(FRAME_WIDTH - 2)
        stacks.center().to_edge(DOWN)
        return stacks

    def get_subset_sums(self, stacks):
        return VGroup(*(
            VGroup(*(
                get_sum_group(set_tex)
                for set_tex in stack
            ))
            for stack in stacks
        ))


class DerivativeTricks(InteractiveScene):
    def construct(self):
        # Setup function
        def func(x, n=10):
            return np.product([1 + x**n for n in range(1, n + 1)])

        plane = NumberPlane(
            (-1, 1),
            (-10, 10, 5),
            width=10,
            faded_line_ratio=4,
        )
        plane.set_height(FRAME_HEIGHT)
        plane.to_edge(LEFT, buff=0)
        plane.add_coordinate_labels(x_values=[-1, 1], y_values=range(-10, 15, 5))
        for cl in plane.x_axis.numbers:
            cl.shift_onto_screen(buff=SMALL_BUFF)
        graph = plane.get_graph(func, x_range=(-1, 1, 0.05))
        graph.set_stroke(YELLOW, 2)
        self.disable_interaction(plane)

        tex_kw = dict(tex_to_color_map={"x": BLUE})
        factored = MTex("f(x) = (1 + x)(1 + x^2)(1 + x^3) \\cdots (1 + x^{2{,}000})", **tex_kw)
        factored.to_corner(UR)
        expanded = MTex("f(x) = 1+x+x^{2}+2 x^{3}+2 x^{4}+ 3x^{5} + \\cdots", **tex_kw)
        expanded.next_to(factored, DOWN, LARGE_BUFF, LEFT)

        self.add(factored, expanded)
        self.play(Write(plane, lag_ratio=0.01))
        self.play(ShowCreation(graph))
        self.wait()

        # Black box
        lhs = expanded[:len("f(x)=")]
        rhs = expanded[len("f(x)="):]
        box = SurroundingRectangle(rhs)
        box.set_stroke(WHITE, 1)
        box.set_fill(GREY_E, 1)
        q_marks = MTex("?").get_grid(1, 9, buff=0.7)
        q_marks.move_to(box)
        box.add(q_marks)

        self.play(FadeIn(box, lag_ratio=0.5))
        self.wait()

        # Plug in 0
        f0 = MTex("f(0) = 1", tex_to_color_map={"0": BLUE})
        f0.next_to(expanded, DOWN, LARGE_BUFF, aligned_edge=LEFT)
        dot = GlowDot(color=WHITE)
        dot.move_to(f0[-1])
        dot.set_opacity(0)
        self.play(FadeTransform(lhs.copy(), f0))
        self.play(dot.animate.move_to(plane.i2gp(0, graph)).set_opacity(1))
        self.wait(note="Move box out of the way")

        # Plug in 1
        f1 = MTex("f(1) = 2^{2{,}000}", tex_to_color_map={"2^{2{,}000}": TEAL, "1": BLUE})
        f1.next_to(f0, RIGHT, buff=2.0, aligned_edge=DOWN)
        self.play(
            TransformFromCopy(factored[:5], f1[:5]),
            FadeTransform(factored[5:].copy(), f1[5:]),
        )
        self.wait(note="Comment on factored form")

        # Take derivative at 0
        f_prime_0 = VGroup(MTex("f'(0) = "), Text("First coefficient"))
        f_prime_n = VGroup(MTex("\\frac{1}{n!} f^{(n)}(0) = "), MTex("n^{\\text{th} } \\text{ coefficient}"))
        last = f0
        for eq in [f_prime_0, f_prime_n]:
            eq.arrange(RIGHT, buff=MED_SMALL_BUFF)
            eq.next_to(last, DOWN, LARGE_BUFF, LEFT)
            last = eq

        tan_line = plane.get_graph(lambda x: x + 1)
        tan_line.set_stroke(PINK, 2, 0.8)

        self.play(FadeTransform(f0.copy(), f_prime_0))
        self.add(tan_line, dot)
        self.play(ShowCreation(tan_line))
        self.wait(note="Comment on derivative")

        self.play(FadeTransform(f_prime_0.copy(), f_prime_n))
        self.wait(note="Comment on this being a nightmare")

        crosses = VGroup(*map(Cross, (f_prime_0, f_prime_n)))
        crosses.insert_n_curves(20)
        self.play(ShowCreation(crosses))
        self.wait()
        self.play(LaggedStartMap(FadeOut, VGroup(f_prime_0, f_prime_n, *crosses)))

        # Clarify goal
        new_rhs = MTex("\\sum_{n = 0}^{\\infty} c_n x^n", **tex_kw)
        new_rhs.move_to(rhs, LEFT)

        self.add(rhs, box)
        self.play(LaggedStartMap(FadeOut, VGroup(*rhs, box), lag_ratio=0.1))
        self.play(Write(new_rhs))
        self.wait()

        examples = VGroup(f0, f1)
        parens = MTex("()")
        words = TexText("Some clever\\\\operation on $f$", font_size=36)
        words.set_color(WHITE)
        parens.match_height(words)
        parens[0].next_to(words, LEFT, buff=SMALL_BUFF)
        parens[1].next_to(words, RIGHT, buff=SMALL_BUFF)
        desire = VGroup(
            VGroup(parens, words),
            MTex("= c_0 + c_5 + c_{10} + \\cdots + c_{2{,}000}")
        )
        desire.arrange(RIGHT)
        desire.next_to(expanded, DOWN, 1.5, LEFT)

        self.play(
            examples.animate.next_to(desire, DOWN, LARGE_BUFF, LEFT),
            FadeIn(desire, DOWN)
        )
        self.wait()

        # Indicator on x^n
        rect = SurroundingRectangle(new_rhs.get_part_by_tex("x^n"), buff=0.05)
        rect.set_stroke(BLUE_B, 2)
        rect.round_corners()

        outcomes = VGroup(
            TexText("$1$ if $n \\equiv 0$ mod $5$", font_size=36, color=GREEN),
            TexText("$0$ if $n \\not\\equiv 0$ mod $5$", font_size=36, color=RED_D),
        )
        outcomes.arrange(DOWN, buff=0.75, aligned_edge=LEFT)
        outcomes.next_to(new_rhs, RIGHT, 1.5, UP)
        arrows = VGroup(*(
            Arrow(
                rect.get_right(), outcome.get_left(),
                buff=0.25,
                color=outcome.get_color(),
                stroke_width=2.0
            )
            for outcome in outcomes
        ))

        self.play(ShowCreation(rect))
        self.wait(0.25)
        for arrow, outcome in zip(arrows, outcomes):
            self.play(
                ShowCreation(arrow),
                GrowFromPoint(outcome, rect.get_right())
            )
            self.wait()


class FifthRootsOfUnity(InteractiveScene):
    def construct(self):
        # Setup plane
        plane = ComplexPlane((-2, 2), (-2, 2))
        plane.set_height(FRAME_HEIGHT - 0.5)
        plane.add_coordinate_labels(font_size=24)
        for coord in plane.coordinate_labels:
            coord.shift_onto_screen(buff=SMALL_BUFF)
            coord.set_fill(WHITE)
        plane.to_edge(LEFT, buff=0.1)
        self.disable_interaction(plane)

        unit_circle = Circle(radius=plane.x_axis.get_unit_size())
        unit_circle.move_to(plane.get_origin())
        unit_circle.set_stroke(GREY_C, 2)
        self.disable_interaction(unit_circle)

        complex_plane_title = Text("Complex plane", font_size=42)
        complex_plane_title.next_to(plane.get_corner(UL), DR, buff=SMALL_BUFF)
        complex_plane_title.set_backstroke(width=8)

        self.add(plane)
        self.add(unit_circle)
        self.add(complex_plane_title)

        # Add function label
        function = MTex(
            "f(x) = \\sum_{n = 0}^\\infty c_n x^n",
            tex_to_color_map={"x": BLUE}
        )
        function.move_to(midpoint(plane.get_right(), RIGHT_SIDE))
        function.to_edge(UP, buff=MED_SMALL_BUFF)
        self.add(function)

        # Roots of unity
        roots = [np.exp(complex(0, n * TAU / 5)) for n in range(5)]
        root_points = list(map(plane.n2p, roots))
        root_dots = Group(*(
            GlowDot(point)
            for point in root_points
        ))
        root_lines = VGroup(*(
            Arrow(
                plane.get_origin(), point, buff=0,
                stroke_width=2,
                stroke_color=YELLOW,
                stroke_opacity=0.7
            )
            for point in root_points
        ))

        arc = Arc(0, TAU / 5, radius=0.2, arc_center=plane.get_origin())
        arc.set_stroke(WHITE, 2)
        arc_label = MTex("2\\pi / 5", font_size=24)
        arc_label.next_to(arc.pfp(0.5), UR, buff=SMALL_BUFF)
        arc_label.set_color(GREY_A)

        root_kw = dict(tex_to_color_map={"\\zeta": YELLOW}, font_size=36)
        zeta_labels = VGroup(
            MTex("\\zeta^0 = 1", **root_kw),
            MTex("\\zeta", **root_kw),
            MTex("\\zeta^2", **root_kw),
            MTex("\\zeta^3", **root_kw),
            MTex("\\zeta^4", **root_kw),
        )
        zeta_labels.set_backstroke(width=8)
        for point, label in zip(root_points, zeta_labels):
            vect = normalize(point - plane.get_origin())
            if point is root_points[0]:
                vect = UR
            label.next_to(point, vect, buff=SMALL_BUFF)
        exp_rhs = MTex(" = e^{2\\pi i / 5}", **root_kw)
        trig_rhs = MTex("= \\cos(72^\\circ) + i\\cdot \\sin(72^\\circ)", **root_kw)
        last = zeta_labels[1]
        for rhs in exp_rhs, trig_rhs:
            rhs.set_backstroke(width=8)
            rhs.next_to(last, RIGHT, SMALL_BUFF)
            last = rhs
        exp_rhs.shift((trig_rhs[0].get_y() - exp_rhs[0].get_y()) * UP)

        self.play(FadeInFromPoint(root_dots[1], plane.get_origin()))
        self.play(Write(zeta_labels[1]))
        self.wait()
        self.play(
            ShowCreation(arc), Write(arc_label),
            TransformFromCopy(root_lines[0], root_lines[1], path_arc=-TAU / 5)
        )
        self.wait()
        self.play(Write(exp_rhs))
        self.wait()
        self.play(Write(trig_rhs))
        self.wait()
        self.play(FadeOut(trig_rhs))

        # Show all roots of unity
        for i in range(2, 6):
            globals().update(locals())
            self.play(*(
                TransformFromCopy(group[i - 1], group[i % 5], path_arc=-TAU / 5)
                for group in [root_lines, root_dots, zeta_labels]
            ))
            self.wait()

        # Name the roots of unity
        title = TexText("``Roots of unity''")
        title.set_color(YELLOW)
        title.match_y(plane)
        title.match_x(function)
        equation = Tex("x^5 = 1")
        equation.set_color(WHITE)
        equation.next_to(title, DOWN)

        self.play(
            Write(title),
            LaggedStart(*(
                FlashAround(zl, time_width=1.5)
                for zl in zeta_labels
            ), lag_ratio=0.1, run_time=2)
        )
        self.wait()
        self.play(FadeIn(equation, 0.5 * DOWN))
        self.wait()
        self.play(LaggedStartMap(FadeOut, VGroup(title, equation, arc, arc_label), shift=DOWN))

        # Key expression
        expr = MTex("+".join([f"f(\\zeta^{{{n}}})" for n in range(5)]), **root_kw)
        expr.next_to(function, DOWN, LARGE_BUFF)

        self.play(
            TransformMatchingShapes(function[:4].copy(), expr, run_time=1.5)
        )
        self.wait()

        # Examples, f(x) = x, f(x) = x^2, etc.
        ex_kw = dict(tex_to_color_map={"{x}": BLUE}, font_size=36)
        example_texts = [
            MTex(
                "\\text{Example: } f({x}) = {x}" + ("^" + str(n) if n > 1 else ""),
                **ex_kw
            )
            for n in range(1, 6)
        ]
        example_sums = [
            MTex(
                "+".join([f"\\zeta^{{{k * n}}}" for n in range(5)]) + ("=5" if k == 5 else "=0"),
                **root_kw
            )
            for k in range(1, 6)
        ]

        def update_root_lines(rl):
            for line, dot in zip(rl, root_dots):
                line.put_start_and_end_on(plane.get_center(), dot.get_center())
            return rl

        root_lines.add_updater(update_root_lines)

        for k, ex_text, ex_sum in zip(it.count(1), example_texts, example_sums):
            ex_text.next_to(expr, DOWN, LARGE_BUFF)
            ex_sum.next_to(ex_text, DOWN, LARGE_BUFF)

            if k == 1:
                self.play(Write(ex_text))
                self.wait()
                self.play(FadeTransform(expr.copy(), ex_sum))
                root_lines.save_state()
                self.wait(note="Move root vectors tip to tail (next animation they restore)")
                self.play(root_lines.animate.restore())
                self.wait()
            else:
                self.play(
                    FadeOut(example_texts[k - 2], 0.5 * UP),
                    FadeIn(ex_text, 0.5 * UP),
                    FadeOut(example_sums[k - 2])
                )
                self.wait()
                self.play(FadeTransform(expr.copy(), ex_sum))
                self.wait()
                # Show permutation
                arrows = VGroup(*(
                    Arrow(root_dots[n], root_dots[(k * n) % 5], buff=0, stroke_width=3)
                    for n in range(1, 5)
                ))
                arrows.set_opacity(0.8)
                for arrow in arrows:
                    self.play(ShowCreation(arrow))
                    self.wait()
                self.play(FadeOut(arrows))
                # Animate kth power
                self.animate_kth_power(plane, root_dots, k)
                self.wait()

        # Emphasize the upshot
        example = VGroup(example_texts[-1], example_sums[-1])
        example.generate_target()
        example.target.arrange(DOWN)
        example.target.match_x(expr)
        example.target.to_edge(DOWN)
        brace = Brace(expr, DOWN, color=GREY_B)

        func_kw = dict(tex_to_color_map={"x": BLUE})
        relations = VGroup(
            MTex("x^n \\rightarrow 0 \\qquad \\text{ if } 5 \\nmid n", **func_kw),
            MTex("x^n \\rightarrow 5 \\qquad \\text{ if } 5 \\mid n", **func_kw),
        )
        relations.arrange(DOWN)
        relations.next_to(brace, DOWN)

        self.play(
            GrowFromCenter(brace),
            MoveToTarget(example)
        )
        for relation in relations:
            self.play(Write(relation))
            self.wait()

        # Write answer expression
        relation_group = VGroup(expr, brace, relations)

        answer = MTex(
            "\\text{Answer} = \\frac{1}{5}\\sum_{k = 0}^4 f(\\zeta^k)",
            tex_to_color_map={"\\zeta": YELLOW}
        )
        answer.next_to(function, DOWN, LARGE_BUFF)
        answer_rect = SurroundingRectangle(answer)
        answer_rect.set_stroke(YELLOW, 2)

        self.play(
            FadeOut(example, DOWN),
            relation_group.animate.set_width(4).to_edge(DOWN),
            Write(answer)
        )
        self.play(
            FlashAround(answer),
            FadeIn(answer_rect)
        )
        self.add(answer_rect, answer)
        self.wait()

        # Bring back orignial definition
        factored = MTex(
            "f(x) = (1 + x)(1 + x^2)(1 + x^3)(1 + x^4)(1 + x^5)\\cdots\\left(1 + x^{2{,}000}\\right)",
            **func_kw
        )
        factored.to_edge(UP)

        lower_group = VGroup(
            VGroup(answer_rect, answer),
            relation_group,
        )
        lower_group.generate_target()
        lower_group.target[0].match_width(relation_group)
        lower_group.target.arrange(RIGHT, buff=MED_LARGE_BUFF)
        lower_group.target.to_corner(DR)

        plane_group = Group(
            plane, unit_circle,
            root_lines, root_dots, zeta_labels, exp_rhs,
            complex_plane_title
        )
        plane_group.generate_target()
        plane_group.target.set_height(4.5, about_edge=DL)

        self.play(
            Write(factored),
            function.animate.next_to(factored, DOWN, buff=0.4, aligned_edge=LEFT),
            MoveToTarget(lower_group),
            MoveToTarget(plane_group),
        )
        self.wait()

        # Evaluate f at zeta
        eq_kw = dict(
            tex_to_color_map={"\\zeta": YELLOW, "{z}": GREY_A},
        )
        f_zeta = MTex(
            "f(\\zeta) = \\Big("
            "(1+\\zeta)(1+\\zeta^{2})(1+\\zeta^{3})(1+\\zeta^{4})(1+\\zeta^{5})"
            "\\Big)^{400}",
            **eq_kw
        )
        f_zeta.next_to(factored, DOWN, aligned_edge=LEFT)

        self.play(
            TransformFromCopy(factored[:5], f_zeta[:5]),
            FadeOut(function, DOWN)
        )
        self.wait()
        self.play(TransformFromCopy(factored[5:34], f_zeta[6:35]))
        self.wait()
        self.play(Write(f_zeta[5]), Write(f_zeta[35:]))
        self.wait(note="Shift zeta values on next move")

        # Visualize roots moving
        shift_vect = plane.n2p(1) - plane.n2p(0)
        zp1_labels = VGroup(*(
            MTex(f"\\zeta^{{{n}}} + 1", **root_kw)
            for n in range(5)
        ))
        zp1_labels.match_height(zeta_labels[0])
        for zp1_label, z_label in zip(zp1_labels, zeta_labels):
            zp1_label.set_backstroke(width=5)
            zp1_label.move_to(z_label, DL)
            zp1_label.shift(shift_vect)

        new_circle = unit_circle.copy()
        new_circle.set_stroke(GREY_B, opacity=0.5)
        self.replace(unit_circle, unit_circle, new_circle)

        self.remove(zeta_labels)
        self.play(
            root_dots.animate.shift(shift_vect),
            new_circle.animate.shift(shift_vect),
            TransformFromCopy(zeta_labels, zp1_labels),
            FadeOut(exp_rhs),
            run_time=2,
        )
        self.wait()

        # Example value
        lhss = VGroup(f_zeta[6:11], f_zeta[11:17], f_zeta[17:23]).copy()
        lhss.generate_target()
        lhss.target.scale(36 / 48)
        lhss.target.arrange(DOWN, aligned_edge=LEFT)
        lhss.target.next_to(plane, RIGHT, LARGE_BUFF, UP)
        rhss = VGroup(*(
            MTex(f"= \\big(1 + \\cos({deg}^\\circ)\\big) + i\\cdot \\sin({deg}^\\circ)", font_size=36)
            for deg in 72 * np.arange(1, 4)
        ))
        for rhs, lhs in zip(rhss, lhss.target):
            rhs.next_to(lhs, RIGHT)

        dots = MTex("\\vdots")
        dots.next_to(lhss.target[-1], DOWN)

        crosses = VGroup(*(Cross(VGroup(lhs, rhs)) for lhs, rhs in zip(lhss.target, rhss)))
        crosses.insert_n_curves(20)

        self.play(MoveToTarget(lhss), Write(dots))
        self.play(Write(rhss))
        self.wait()
        self.play(ShowCreation(crosses))
        self.wait()
        self.play(*map(FadeOut, [lhss, rhss, crosses, dots]))

        # Setup for the trick
        box = Rectangle(
            height=plane.get_height(),
            width=abs(plane.get_right()[0] - RIGHT_SIDE[0]) - 1,
        )
        box.set_stroke(WHITE, 1)
        box.set_fill(GREY_E, 1)
        box.next_to(plane, RIGHT, buff=0.5)
        self.disable_interaction(box)
        trick_title = Text("The trick")
        trick_title.next_to(box.get_top(), DOWN, SMALL_BUFF)

        self.remove(zp1_labels)
        self.play(
            root_dots.animate.shift(-shift_vect),
            new_circle.animate.shift(-shift_vect),
            TransformFromCopy(zp1_labels, zeta_labels),
            FadeOut(lower_group, DOWN),
            FadeIn(box),
            Write(trick_title),
        )
        self.wait()

        # The trick
        root_kw["tex_to_color_map"]["{z}"] = GREY_A
        root_kw["tex_to_color_map"]["{-1}"] = GREY_A
        root_kw["tex_to_color_map"][" = "] = WHITE
        texs = [
            "{z}^5 - 1 \\,=\\, ({z} - \\zeta^0)({z} - \\zeta^1)({z} - \\zeta^2)({z} - \\zeta^3)({z} - \\zeta^4)",
            "({-1})^5 - 1 \\,=\\, ({-1} - \\zeta^0)({-1} - \\zeta^1)({-1} - \\zeta^2)({-1} - \\zeta^3)({-1} - \\zeta^4)",
            "2 \\,=\\, (1 + \\zeta^0)(1 + \\zeta^1)(1 + \\zeta^2)(1 + \\zeta^3)(1 + \\zeta^4)",
        ]
        equations = VGroup(*(MTex(tex, **root_kw) for tex in texs))
        equations[1].set_width(box.get_width() - 0.5)
        equations.arrange(DOWN, buff=0.75)
        equals_x = equations[0].get_part_by_tex("=").get_x()
        for eq in equations[1:]:
            eq.shift((equals_x - eq.get_part_by_tex("=").get_x()) * RIGHT)
        equations.next_to(trick_title, DOWN, MED_LARGE_BUFF)

        self.play(Write(equations[0]))
        self.wait()
        self.play(FadeTransform(equations[0].copy(), equations[1]))
        self.wait()
        self.play(FadeTransform(equations[1].copy(), equations[2]))
        self.wait()

        # Show value of 2
        brace = Brace(f_zeta[6:35], DOWN).set_color(WHITE)
        brace.stretch(0.75, 1, about_edge=UP)
        two_label = brace.get_tex("2").set_color(WHITE)
        self.play(GrowFromCenter(brace))
        self.play(TransformFromCopy(equations[2][0], two_label))
        self.wait()
        self.play(LaggedStartMap(FadeOut, VGroup(box, trick_title, *equations)))
        self.play(FadeIn(lower_group))
        self.wait()

        # Evaluate answer
        ans_group, expr_group = lower_group
        self.play(
            ans_group.animate.scale(0.5, about_edge=UL),
            expr_group.animate.scale(1.5, about_edge=DR),
        )
        self.wait()

        parts = [expr.get_part_by_tex(f"f(\\zeta^{{{n}}})") for n in range(5)]
        arrows = VGroup(*(
            Vector(0.5 * UP).next_to(part, UP, SMALL_BUFF)
            for part in parts
        ))
        values = VGroup(
            MTex("2^{2{,}000}", font_size=36),
            *(MTex("2^{400}", font_size=36) for x in range(4))
        )
        for value, arrow in zip(values, arrows):
            value.next_to(arrow, UP, SMALL_BUFF)

        self.play(
            ShowCreation(arrows[1]),
            FadeIn(values[1], 0.5 * UP)
        )
        self.wait()
        self.play(
            LaggedStartMap(ShowCreation, arrows[2:], lag_ratio=0.5),
            LaggedStartMap(FadeIn, values[2:], shift=0.5 * UP, lag_ratio=0.5),
        )
        self.wait()
        self.play(
            ShowCreation(arrows[0]),
            FadeIn(values[0], 0.5 * UP)
        )
        self.wait()

        expr_group.add(arrows, values)
        self.play(
            ans_group.animate.scale(2.0, about_edge=UL),
            expr_group.animate.scale(1 / 1.5, about_edge=DR),
        )

        # Final answer
        final_answer = MTex(
            "\\text{Answer} = \\frac{1}{5}\\Big("
            "2^{2{,}000} + 4 \\cdot 2^{400}\\Big)"
        )
        final_answer.align_to(answer, LEFT)
        final_answer.align_to(plane, UP)
        f_zeta_rhs = Tex("=2^{400}").set_color(WHITE)
        f_zeta_rhs.next_to(f_zeta, RIGHT)

        final_answer_rect = SurroundingRectangle(final_answer)
        final_answer_rect.set_stroke(YELLOW, 2)
        self.disable_interaction(final_answer_rect)

        self.play(
            Transform(answer[:10].copy(), final_answer[:10]),
            FadeTransform(answer[10:].copy(), VGroup(final_answer[10], final_answer[-1])),
            TransformMatchingShapes(two_label, f_zeta_rhs),
            FadeOut(brace),
            FadeOut(answer_rect),
        )
        self.wait()
        self.play(
            FadeTransform(values.copy(), final_answer[11:-1])
        )
        self.play(ShowCreation(final_answer_rect))
        self.wait()

    def animate_kth_power(self, plane, dots, k):
        # Try the rotation
        angles = np.angle([plane.p2n(dot.get_center()) for dot in dots])
        angles = angles % TAU
        paths = [
            ParametricCurve(
                lambda t: plane.n2p(np.exp(complex(0, interpolate(angle, k * angle, t)))),
                t_range=(0, 1, 0.01),
            )
            for angle in angles
        ]
        dots.save_state()

        self.play(*(
            MoveAlongPath(dot, path)
            for dot, path in zip(dots, paths)
        ), run_time=4)
        if k == 5:
            self.wait()
            self.play(dots.animate.restore())
        else:
            dots.restore()

        # moving_plane = ComplexPlane((-2, 2), (-2, 2))
        # moving_plane.replace(plane)
        # moving_plane.prepare_for_nonlinear_transform()
        # back_plane = plane.copy()
        # back_plane.set_color(GREY_C)
        # back_plane.fade(0.5)
        # self.add(back_plane)
        # self.bring_to_back(back_plane)
        # self.replace(plane, moving_plane)

        # def homotopy(x, y, z, t):
        #     z = plane.p2n([x, y, z])
        #     return plane.n2p(z**(1 + t * (k - 1)))

        # self.play(
        #     Homotopy(homotopy, dots),
        #     Homotopy(homotopy, moving_plane),
        # )
        # self.wait()
        # self.play(
        #     FadeOut(moving_plane),
        #     FadeIn(plane),
        # )
        # self.remove(back_plane)
        # self.replace(moving_plane, plane)


class GoThroughAllSubsets(InteractiveScene):
    n_searched = 5000

    def construct(self):
        question = TexText(
            "How many subsets of $\\{1, 2, \\dots 2{,}000\\}$ sum "
            "to a multiple of 5?",
            tex_to_color_map={"$\\{1, 2, \\dots, 2{,}000\\}$": BLUE}
        )
        question.to_edge(UP)
        self.add(question)

        answer = MTex("\\frac{1}{5}\\big(2^{2{,}000} + 4 \\cdot 2^{400}\\big)")
        answer.next_to(question, DOWN, MED_LARGE_BUFF)
        self.add(answer)

        arrow = MTex("\\downarrow")
        arrow.move_to(DOWN + 2 * LEFT)
        self.add(arrow)

        counter = VGroup(
            Text("Count so far: "), Integer(0, edge_to_fix=LEFT)
        )
        counter.arrange(RIGHT)
        counter.next_to(arrow, RIGHT, buff=3.0)
        self.add(counter)

        for n in range(self.n_searched):
            bits = str(bin(n))[2:]
            subset = []
            for i, bit in enumerate(reversed(bits)):
                if bit == '1':
                    subset.append(i)
            set_tex = get_set_tex(subset, max_shown=12)
            set_tex.next_to(arrow, UP)
            set_tex.set_color(BLUE)
            total = sum(subset)
            value = Integer(total)
            value.next_to(arrow, DOWN)
            to_add = [value, set_tex]
            if total % 5 == 0:
                counter[1].increment_value()
                counter[1].set_color(YELLOW)
                rect = SurroundingRectangle(value, buff=SMALL_BUFF)
                rect.round_corners()
                rect.set_stroke(YELLOW, 2)
                to_add.append(rect)
            self.add(*to_add)
            self.wait(1 / 30)
            self.remove(*to_add)

