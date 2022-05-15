from manim_imports_ext import *


def get_set_tex(values, max_shown=7, **kwargs):
    if len(values) > max_shown:
        value_mobs = [
            *map(Integer, values[:max_shown - 2]),
            MTex("\\dots"),
            Integer(values[-1], group_with_commas=False),
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
    for part in set_tex2:
        if isinstance(part, MTex) and part.get_tex() == "\\dots":
            anims.append(FadeInFromPoint(part, set_tex1.get_bottom()))
    return AnimationGroup(*anims)


def get_sum_wrapper(set_tex):
    wrapper = VGroup(
        Tex("\\text{sum}\\big(\\big) = ")[0],
        Integer(sum(set_tex.values))
    )
    wrapper.set_height(1.25 * set_tex.get_height())
    wrapper[0][:4].next_to(set_tex, LEFT, SMALL_BUFF)
    wrapper[0][4:].next_to(set_tex, RIGHT, SMALL_BUFF)
    wrapper[1].next_to(wrapper[0][-1], RIGHT, buff=0.2)
    return wrapper


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


def get_question_title():
    st = "$\\{1, 2, 3, 4, 5, \\dots, 2{,}000\\}$"
    question = TexText(
        f"Find the number of subsets of {st},\\\\"
        " the sum of whose elements is divisible by 5",
        isolate=[st]
    )
    set_tex = get_set_tex(range(1, 2001))
    set_tex.set_color(BLUE)
    set_tex.replace(question.get_part_by_tex(st))
    question.replace_submobject(1, set_tex)
    question.to_edge(UP)
    return question


# Scenes

class ProblemStatement(TeacherStudentsScene):
    def construct(self):
        # Title
        title = Text("Useless puzzles with useful lessons", font_size=60)
        title.to_edge(UP)
        title.set_color(BLUE_C)
        underline = Underline(title, buff=-0.05)
        underline.insert_n_curves(20)
        underline.set_stroke(width=[1, 3, 3, 3, 1])
        words = VGroup(*(
            title.get_part_by_text(text)
            for text in title.get_text().split(" ")
        ))

        morty = self.teacher

        self.play(
            LaggedStart(*(
                FadeInFromPoint(word, morty.get_corner(UL))
                for word in words
            )),
            morty.animate.change("raise_right_hand", title),
            self.get_student_changes("happy", "thinking", "happy", look_at_arg=title)
        )
        self.remove(words)
        self.add(title)
        self.play(ShowCreation(underline))
        self.wait(2)
        self.change_student_modes("pondering", "erm", "tease", look_at_arg=title)
        self.wait(3)

        # Statement
        statement = get_question_title()
        set_tex = statement[1]

        self.play(
            FadeIn(statement),
            self.get_student_changes(*3 * ["pondering"], look_at_arg=statement),
            FadeOut(VGroup(title, underline), UP)
        )
        self.wait()
        self.play(
            LaggedStart(*(
                FlashAround(int_part)
                for int_part in get_integer_parts(set_tex)
            ), lag_ratio=0.05, run_time=3),
            self.get_student_changes("erm", "sassy", "confused", look_at_arg=statement),
        )
        self.wait(2)

        # Example subsets
        subsets = VGroup(
            get_set_tex([3, 1, 4]),
            get_set_tex([2, 3, 5]),
            get_set_tex(range(1, 2000, 2)),
        )
        subsets.move_to(UP)
        sum_wrappers = VGroup(*map(get_sum_wrapper, subsets))
        mode_triples = [
            ("erm", "pondering", "plain"),
            ("pondering", "thinking", "pondering"),
            ("thinking", "happy", "thinking"),
        ]

        sum_groups = VGroup()
        for subset, sum_wrapper, mode_triple in zip(subsets, sum_wrappers, mode_triples):
            self.play(
                morty.animate.change("tease", subset),
                self.get_student_changes(*mode_triple, look_at_arg=subset),
                set_tex_transform(set_tex, subset),
            )
            elem_rects = VGroup(*(
                SurroundingRectangle(
                    get_part_by_value(set_tex, value)
                ).set_stroke(TEAL, 2).round_corners()
                for value in subset.values
            ))
            self.play(LaggedStartMap(ShowCreation, elem_rects, lag_ratio=0.5))

            self.wait()
            self.play(
                Write(sum_wrapper),
                morty.animate.change("hooray" if subset is subsets[-1] else "tease", subset),
            )
            self.wait(2)
            mark = Checkmark() if sum(subset.values) % 5 == 0 else Exmark()
            sum_wrapper.refresh_bounding_box()
            mark.next_to(sum_wrapper, RIGHT)
            self.play(Write(mark))
            self.wait()
            anims = [FadeOut(elem_rects)]

            sum_group = VGroup(subset, sum_wrapper, mark)

            if subset is subsets[0]:
                self.play(
                    sum_group.animate.scale(0.5).to_edge(LEFT).set_y(2),
                    *anims
                )
            else:
                self.play(
                    sum_group.animate.scale(0.5).next_to(sum_groups, DOWN, MED_LARGE_BUFF, LEFT),
                    *anims
                )
            sum_groups.add(sum_group)

        self.play(
            self.get_student_changes(
                "erm", "confused", "hesitant",
                look_at_arg=statement,
            )
        )

        # Who cares?
        statement.generate_target()
        statement.target.scale(0.5)
        statement.target.to_corner(UL)
        title.set_opacity(0)
        title.scale(0.5)
        title.generate_target()
        title.target.set_opacity(1)
        title.target.scale(2)
        title.target.next_to(self.hold_up_spot, UP, MED_LARGE_BUFF)
        title.target.to_edge(RIGHT)

        self.play(
            PiCreatureSays(
                self.students[2], "Who cares?", target_mode="angry",
                bubble_kwargs=dict(direction=LEFT),
            ),
            morty.animate.change("guilty"),
            MoveToTarget(statement),
            FadeOut(sum_groups, LEFT)
        )
        self.wait()
        self.play(
            RemovePiCreatureBubble(
                self.students[2],
                target_mode="plain",
                look_at_arg=ORIGIN,
            ),
            morty.animate.change("sassy", self.students[2].eyes),
            self.get_student_changes(
                "pondering", "pondering",
                look_at_arg=ORIGIN,
            ),
            MoveToTarget(title),
        )
        self.play(Blink(morty))
        self.play(
            FlashAround(title, time_width=1),
            morty.animate.change("raise_right_hand"),
        )
        self.wait()
        return

        # Start simpler
        small_set = get_set_tex(range(1, 6))
        small_set.to_edge(UP)

        case_words = Text("Start simple")
        case_words.next_to(small_set, DOWN)
        case_words.set_color(BLUE)

        self.play(
            statement.animate.set_y(0),
            set_tex_transform(set_tex, small_set),
            LaggedStartMap(
                FadeOut, self.pi_creatures, shift=DOWN,
            ),
            FadeOut(self.background),
            Write(case_words),
            run_time=2,
        )
        self.wait()
        self.play(FadeOut(statement), FadeOut(case_words))


class AnswerGuess(InteractiveScene):
    def construct(self):
        # Count total
        title = get_question_title()

        count = VGroup(Text("Total subsets: "), MTex("2^{2{,}000}"))
        count[0].set_color(TEAL)
        count.arrange(RIGHT, buff=MED_SMALL_BUFF, aligned_edge=DOWN)
        count.scale(60 / 48)
        count.next_to(title, DOWN, LARGE_BUFF)

        self.add(title)
        self.wait()
        self.play(Write(count[0]))
        self.play(
            GrowFromCenter(count[1][0]),
            FadeTransform(title[1][-6:-1].copy(), count[1][1:]),
        )
        self.wait()

        # Ask about why
        count_rect = SurroundingRectangle(count, buff=MED_SMALL_BUFF)
        count_rect.set_stroke(TEAL, 2)
        randy = Randolph(height=1.5)
        randy.flip()
        randy.next_to(count_rect, RIGHT, LARGE_BUFF).shift(0.5 * DOWN)
        randy.get_bubble(Text("Why?", font_size=24), direction=LEFT, height=1.0, width=1.0)

        self.play(
            ShowCreation(count_rect),
            VFadeIn(randy),
            randy.animate.change("maybe", count_rect),
            Write(randy.bubble),
            Write(randy.bubble.content),
        )
        self.play(Blink(randy))
        self.wait()
        self.play(randy.animate.change("thinking").look(DL))
        self.play(Blink(randy))
        randy.add(randy.bubble, randy.bubble.content)
        self.wait()
        self.play(*map(FadeOut, [randy, count_rect]))

        # Show total digits
        count.generate_target()
        count.target.to_edge(LEFT)
        eq = MTex("=")
        eq.next_to(count.target, RIGHT).match_y(count[1][0])

        total = VGroup(*(MTex(digit) for digit in str(2**2000)))
        total.arrange_in_grid(h_buff=SMALL_BUFF, v_buff=1.5 * SMALL_BUFF, n_cols=42)
        for n in range(len(total) - 3, -3, -3):
            comma = MTex(",")
            triplet = total[n:n + 3]
            triplet.arrange_to_fit_width(
                triplet.get_width() - 2 * comma.get_width(),
                about_edge=LEFT
            )
            comma.move_to(triplet.get_corner(DR) + 1.5 * comma.get_width() * RIGHT)
            total.insert_submobject(n + 3, comma)
        total[-1].set_opacity(0)
        total.set_width(7)
        total.next_to(eq, RIGHT).align_to(count, UP)

        self.play(MoveToTarget(count), Write(eq))
        self.play(ShowIncreasingSubsets(total, run_time=5))
        self.wait()
        self.play(
            FadeOut(total, 0.1 * DOWN, lag_ratio=0.01),
            FadeOut(eq),
        )

        # Guess 1 / 5
        guess = VGroup(Text("Guess: "), MTex("\\approx \\frac{1}{5} \\cdot 2^{2{,}000}"))
        guess[0].set_color(YELLOW)
        guess.arrange(RIGHT)
        guess.next_to(count, RIGHT, buff=2.5)
        guess.match_y(count[0])

        self.play(Write(guess[0]))
        self.play(TransformMatchingShapes(count[1].copy(), guess[1], path_arc=30 * DEGREES))
        self.wait()

        # Simplify
        small_set = get_set_tex(range(1, 6))
        small_set.to_edge(UP)
        simplify = Text("Simplify!")
        simplify.next_to(small_set, DOWN, MED_LARGE_BUFF)
        simplify.set_color(YELLOW)

        self.play(
            LaggedStartMap(
                FadeOut, VGroup(count, guess),
                lag_ratio=0.25,
                run_time=1,
            )
        )
        self.remove(title)
        self.play(
            FadeOut(title[0], scale=0.5),
            FadeOut(title[2], scale=0.5),
            set_tex_transform(title[1], small_set),
            FadeIn(simplify, scale=2)
        )
        self.wait()

        # Ask questions
        kw = dict(t2c={
            "construct": YELLOW,
            "organize": TEAL,
        })
        questions = VGroup(
            Text("What are all\nthe subsets?", **kw),
            Text("How do you\nconstruct them?", **kw),
            Text("How do you\norganize them?", **kw),
        )
        questions.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        questions.to_edge(LEFT).to_edge(DOWN, buff=LARGE_BUFF)

        self.play(FadeTransform(simplify, questions[0]))
        self.wait()
        self.play(Write(questions[1], run_time=1))
        self.wait()
        self.play(
            TransformFromCopy(
                questions[1].get_part_by_text("How do you"),
                questions[2].get_part_by_text("How do you"),
            ),
            TransformFromCopy(
                questions[1][-5:],
                questions[2][-5:],
            ),
            FadeTransform(
                questions[1].get_part_by_text("construct").copy(),
                questions[2].get_part_by_text("organize"),
            )
        )
        self.wait()


class GoThroughAllSubsets(InteractiveScene):
    n_searched = 2**6
    rate = 1

    def construct(self):
        full_set = get_set_tex(range(1, 2001), max_shown=25)
        full_set.set_width(0.9 * FRAME_WIDTH)
        full_set.center()
        full_set_elements = [sm for sm in full_set if isinstance(sm, Integer)]

        sum_mob = Integer(0)
        sum_mob.next_to(full_set, DOWN, buff=1.0)
        sum_lhs = Text("Sum = ")
        sum_lhs.next_to(sum_mob, LEFT)
        sum_label = VGroup(sum_lhs, sum_mob)
        sum_label.align_to(full_set, LEFT).shift(RIGHT)

        counter = VGroup(
            Text("Count so far: "), Integer(0, edge_to_fix=LEFT)
        )
        counter.arrange(RIGHT)
        counter.next_to(sum_mob, RIGHT)
        counter.align_to(full_set, RIGHT).shift(LEFT)

        for elem in full_set_elements:
            elem.rect = SurroundingRectangle(elem, buff=0.05)
            elem.rect.set_stroke(BLUE, 2)
            elem.rect.round_corners()
            elem.line = Line(elem.rect.get_bottom(), sum_mob.get_top(), buff=0.2)
            elem.line.set_stroke(BLUE, 2)
        self.add(full_set)
        self.add(counter)
        self.add(sum_label)

        group = VGroup(
            VGroup(),  # Rects
            VGroup(),  # Lines
            sum_mob,
            counter[1],
            VGroup(),  # Mark
        )

        n_tracker = ValueTracker(0)

        all_counts = np.cumsum([
            int(sum((int(bit) * n for n, bit in zip(it.count(1), reversed(bin(n)[2:])))) % 5 == 0)
            for n in range(self.n_searched)
        ])

        def update_group(group):
            n = int(n_tracker.get_value())
            rects, lines, sum_mob, count, mark = group

            bits = str(bin(n))[2:]
            elems = []
            subset = []
            for i, elem, bit in zip(it.count(1), full_set_elements, reversed(bits)):
                if bit == '1':
                    elems.append(elem)
                    subset.append(i)

            total = sum(subset)
            sum_mob.set_value(total)

            rects.set_submobjects([elem.rect for elem in elems])
            lines.set_submobjects([elem.line for elem in elems])

            if total % 5 == 0:
                count.set_value(all_counts[n])
                count.set_color(GREEN)
                sum_mob.set_color(GREEN)
                check = Checkmark()
                check.next_to(sum_mob, RIGHT)
                mark.set_submobjects([check])
            else:
                sum_mob.set_color(WHITE)
                mark.set_submobjects([])

        self.add(group, n_tracker)
        self.play(
            n_tracker.animate.set_value(self.n_searched),
            UpdateFromFunc(group, update_group),
            run_time=self.n_searched / self.rate,
            rate_func=linear,
        )


class GoThroughAllSubsetsFast(GoThroughAllSubsets):
    n_searched = 2**13
    rate = 120


class TwoThousandBinaryChoices(InteractiveScene):
    def construct(self):
        # Build parts
        full_set = get_set_tex(range(1, 2001), max_shown=25)
        full_set.set_width(0.9 * FRAME_WIDTH)
        full_set.center()
        self.add(full_set)

        elems = get_integer_parts(full_set)
        elem_rects = VGroup(*(
            SurroundingRectangle(elem, buff=0.05).round_corners()
            for elem in elems
        ))
        elem_rects.set_stroke(BLUE, 2)

        twos = VGroup(*(
            Integer(2).match_width(elems[1]).next_to(er, UP)
            for er in elem_rects
        ))
        dots = VGroup(*(
            MTex("\\cdot").move_to(midpoint(t1.get_right(), t2.get_left()))
            for t1, t2 in zip(twos, twos[1:])
        ))
        dots.replace_submobject(-1, MTex("\\dots").move_to(dots[-1]))
        dots.add_to_back(VMobject())

        choices = Text("choices").match_height(twos)
        VGroup(twos, dots, choices).set_color(TEAL)

        # Label choices
        words = Text("2,000 Binary choices")
        words.set_color(TEAL)
        words.next_to(full_set, DOWN, buff=MED_LARGE_BUFF)

        self.play(
            LaggedStartMap(ShowCreationThenFadeOut, elem_rects, lag_ratio=0.05),
            FadeIn(words, 0.5 * DOWN)
        )
        self.wait()

        # Go through all choices
        self.add(choices)
        for rect, two, dot in zip(elem_rects, twos, dots):
            choices.next_to(two, RIGHT, buff=0.2)
            marks = VGroup(Checkmark(), Exmark())
            marks.scale(0.5)
            marks.next_to(rect, DOWN, SMALL_BUFF)
            self.add(two, dot)
            include = random.random() < 0.5
            if include:
                self.add(rect)
                marks.submobjects.reverse()
            self.play(ShowSubmobjectsOneByOne(marks.copy(), remover=True, run_time=0.5))
            self.add(marks[-1])
            self.wait(0.25)
        self.remove(choices)
        self.wait()


class ExampleWith5(InteractiveScene):
    elem_colors = color_gradient([BLUE_B, BLUE_D], 5)

    def construct(self):
        # Add full set
        N = 5
        full_set = list(range(1, N + 1))
        set_tex = get_set_tex(full_set)
        set_tex.to_edge(UP)
        self.add(set_tex)
        self.wait()

        # Construct all subsets
        subsets = self.construct_all_subsets(set_tex)

        # Show n choose k stacks
        stacks = self.get_subset_stacks(full_set)

        anims = []
        for stack in stacks:
            for new_subset in stack:
                for subset in subsets:
                    if set(subset.values) == set(new_subset.values):
                        anims.append(FadeTransform(subset, new_subset))

        self.play(LaggedStart(*anims, lag_ratio=0.05))
        self.wait()

        # Show their sums
        sum_stacks = self.get_subset_sums(stacks)
        covered_sums = []
        n_selections = 6
        for n in range(n_selections):
            self.wait(note=f"Example sum {n + 1} / {n_selections}")
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

        # Group by sum
        self.highlight_rects = self.get_highlight_rects(stacks, sum_stacks)
        self.highlight_rects.set_stroke(width=0)
        self.group_by_sum(stacks, sum_stacks)
        self.wait()

        # Count Answer
        highlights = VGroup()
        for stack in self.common_sum_stacks[::5]:
            for subset in stack:
                highlights.add(VHighlight(subset, max_stroke_addition=10))

        answer_word, answer_count = answer = VGroup(
            Text("Answer: "),
            Integer(0)
        )
        answer.arrange(RIGHT)
        answer.next_to(self.common_sum_stacks[0], UP, LARGE_BUFF)
        answer_count.set_color(TEAL)

        self.add(highlights, self.common_sum_stacks, answer)
        self.play(
            ShowIncreasingSubsets(highlights),
            UpdateFromFunc(answer_count, lambda m: m.set_value(len(highlights))),
            run_time=1.5
        )
        self.play(FadeOut(highlights))

        # Contrast with 1/5
        count = TexText(
            "Total subsets:", " $2^5 = $", " $32$"
        )
        count.set_color_by_tex("32", BLUE)
        count.match_x(set_tex).match_y(answer)
        counter = Integer(32)
        counter.set_color(BLUE)
        counter.replace(count[-1])

        all_highlights = VGroup()
        for stack in self.common_sum_stacks:
            for subset in stack:
                all_highlights.add(VHighlight(subset, max_stroke_addition=10))

        fifth_arrow = Vector(RIGHT)
        fifth_arrow.next_to(count, RIGHT)
        fifth = MTex("\\times 1 / 5", font_size=24)
        fifth.next_to(fifth_arrow, UP, buff=SMALL_BUFF)
        fifth_rhs = DecimalNumber(32 / 5, num_decimal_places=1)
        fifth_rhs.next_to(fifth_arrow, RIGHT)

        self.add(all_highlights, self.common_sum_stacks)
        self.play(
            FadeIn(count[:2]),
            ShowIncreasingSubsets(all_highlights, run_time=2),
            UpdateFromFunc(counter, lambda m: m.set_value(len(all_highlights))),
        )
        self.play(FadeOut(all_highlights))
        self.wait()
        self.play(
            GrowArrow(fifth_arrow),
            Write(fifth),
        )
        self.wait()
        self.play(FadeTransform(count[-1].copy(), fifth_rhs, path_arc=-60 * DEGREES))
        self.wait()
        self.play(LaggedStartMap(FadeOut, VGroup(
            answer, *count[:2], counter, fifth_arrow, fifth, fifth_rhs,
        )))

        # Show generating function
        self.show_generating_function(set_tex)
        self.transition_to_full_generating_function(set_tex)

    def construct_all_subsets(self, set_tex):
        # Preview binary choices
        value_parts = VGroup(*(
            get_part_by_value(set_tex, v) for v in range(1, 6)
        ))
        rects = VGroup(*(
            SurroundingRectangle(vp, buff=0.1).round_corners()
            for vp in value_parts
        ))
        rects.set_stroke(BLUE, 2)

        words = Text("5 binary choices")
        words.next_to(set_tex, DOWN, buff=1.5)
        lines = VGroup(*(
            Line(words.get_top(), rect.get_bottom(), buff=0.15)
            for rect in rects
        ))
        lines.match_style(rects)

        def match_n(rects, n):
            bits = it.chain(
                str(bin(n)[-1:1:-1]),
                it.repeat("0")
            )
            for rect, bit in zip(rects, bits):
                rect.set_stroke(opacity=float(bit == "1"))

        self.add(rects)
        self.play(
            Write(words),
            Write(lines),
            UpdateFromAlphaFunc(
                rects, lambda r, a: match_n(r, int(31 * a)),
                run_time=4,
                rate_func=linear,
            )
        )
        self.wait()
        self.play(
            FadeOut(rects),
            LaggedStartMap(Uncreate, lines),
            FadeOut(words, 0.1 * DOWN),
        )

        # Show construction
        subsets = VGroup(get_set_tex([]))
        for value in range(1, 6):
            value_mob = get_part_by_value(set_tex, value)
            marks = VGroup(Exmark(), Checkmark())
            marks.match_height(value_mob)
            marks.next_to(value_mob, DOWN)

            subsets.generate_target()
            added_subsets = VGroup(*(
                get_set_tex([*ss.values, value]).move_to(ss)
                for ss in subsets
            ))
            for ss in added_subsets:
                self.color_set_tex(ss)
                get_integer_parts(ss)[-1].set_opacity(0)

            vect = [RIGHT, DOWN, RIGHT, DOWN, RIGHT][value - 1]
            buff = [2.25, 0.75, 2.0, 0.75, 1.0][value - 1]
            added_subsets.next_to(subsets, vect, buff=buff)
            new_subsets = VGroup(*subsets.target, *added_subsets)
            new_subsets.set_max_width(FRAME_WIDTH - 1)
            new_subsets.center()
            subsets_copy = subsets.copy()
            for ssc, nss in zip(subsets_copy, added_subsets):
                ssc.match_height(nss)
                ssc.move_to(nss)

            self.wait()
            elem = get_part_by_value(set_tex, value)
            if value == 1:
                self.play(set_tex_transform(set_tex, subsets[0]))
                self.add(subsets)
                self.wait()
            self.play(
                MoveToTarget(subsets, path_arc=30 * DEGREES),
                ReplacementTransform(
                    subsets.copy(),
                    subsets_copy,
                    path_arc=30 * DEGREES,
                )
            )
            self.remove(subsets_copy)
            self.play(
                LaggedStart(*(
                    Transform(
                        elem.copy(),
                        get_integer_parts(st)[-1].copy().set_opacity(1),
                        remover=True,
                    )
                    for st in added_subsets
                ), lag_ratio=0.1),
                *(
                    set_tex_transform(st1, st2)
                    for st1, st2 in zip(subsets_copy, added_subsets)
                ),
                elem.animate.set_color(self.elem_colors[value - 1]),
                FlashAround(elem, color=self.elem_colors[value - 1]),
            )
            self.remove(subsets_copy, new_subsets)
            subsets.set_submobjects(list(new_subsets))
            self.add(subsets)
            subsets.set_opacity(1)
            self.wait()
        self.wait()

        # Equation
        equation = MTex("2 \\cdot 2 \\cdot 2 \\cdot 2 \\cdot 2 = 2^5 = 32")
        equation.set_width(4)
        equation.to_corner(UL)
        equation.set_color(YELLOW)
        self.play(Write(equation))
        self.wait()
        self.play(FadeOut(equation))

        return subsets

    def get_highlight_rects(self, stacks, sum_stacks):
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
        return rects

    def highlight_multiple_of_5(self, stacks, sum_stacks):
        # Blah
        rects = self.get_highlight_rects(stacks, sum_stacks)

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
        lower_group = self.lower_group = VGroup(csr, css)

        factored_terms = "(1 + x^1)", "(1 + x^2)", "(1 + x^3)", "(1 + x^4)", "(1 + x^5)"
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
        self.play(lower_group.animate.set_height(3.0, about_edge=DOWN))

        # Emphasize 5 binary choices
        parts = VGroup(*(
            factored.get_part_by_tex(term)
            for term in factored_terms
        ))
        rects = VGroup(*(
            SurroundingRectangle(part, buff=0.05).round_corners()
            for part in parts
        ))
        rects.set_stroke(BLUE, 2)
        words = Text("5 binary choices", color=BLUE)
        words.next_to(rects, DOWN, MED_LARGE_BUFF)

        self.play(
            LaggedStartMap(
                VFadeInThenOut, rects,
                lag_ratio=0.25,
                run_time=4,
            ),
            Write(words),
        )
        self.play(FadeOut(words))

        # Animate expansion
        fac_term_parts = [factored.get_part_by_tex(term) for term in factored_terms]
        expanded_parts = [expanded.get_part_by_tex(term) for term in expanded_terms]
        super_expanded = VGroup()
        super_expanded.next_to(factored, DOWN, MED_LARGE_BUFF)
        collection_anims = []

        subset_groups = self.subset_groups
        subset_groups.submobjects.sort(
            key=lambda ssg: sum(ssg[0].values)
        )

        for subset_group in subset_groups:
            bits = [i + 1 in subset_group[0].values for i in range(5)]
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
                FadeTransform(new_term, expanded_parts[n], path_arc=10 * DEGREES)
            )

            self.add(top_rects)
            self.add(super_expanded, low_rect)
            subset_groups.set_opacity(0.25)
            subset_group.set_opacity(1)
            self.wait()
            self.remove(top_rects, low_rect)
        self.wait()

        # Reorganize to expanded
        lower_group.generate_target()
        lower_group.target.set_height(4, about_edge=DOWN)
        lower_group.target[1].set_opacity(1)
        self.play(
            LaggedStart(*collection_anims),
            LaggedStartMap(FadeIn, expanded.get_parts_by_tex("+")),
            MoveToTarget(
                lower_group,
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
            "f(x) = (1 + x^1)(1 + x^2)(1 + x^3) \\cdots \\left(1 + x^{1{,}999}\\right)\\left(1 + x^{2{,}000}\\right)",
        )
        expanded = MTex(
            "f(x) = 1+x+x^{2}+2 x^{3}+2 x^{4}+ 3x^{5} +\\cdots + x^{2{,}001{,}000}",
            isolate="+",
        )
        new_set_tex = get_set_tex(range(1, 2001))
        new_set_tex.move_to(set_tex)
        self.color_set_tex(new_set_tex)
        get_integer_parts(new_set_tex)[-1].set_color(
            interpolate_color(BLUE_E, BLUE_D, 0.5)
        )

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
            FadeOut(self.lower_group, DOWN),
        )
        self.wait()
        self.play(
            ShowCreation(h_line),
            FadeIn(expanded_word),
            Write(expanded)
        )
        self.wait()

        # Emphasize scale
        words = TexText("Imagine collecting $2^{2{,}000}$ terms!")
        words.set_color(RED)
        words.next_to(h_line, DOWN)
        words.to_edge(RIGHT)
        morty = Mortimer(height=2)
        morty.next_to(words, DOWN, MED_LARGE_BUFF)
        morty.to_edge(RIGHT)

        self.play(
            VFadeIn(morty),
            morty.animate.change("surprised", expanded),
            Write(words)
        )
        self.play(Blink(morty))
        self.wait()
        self.play(FadeOut(words), FadeOut(morty))

        # Show example term
        n = 25
        subsets = list(filter(
            lambda s: sum(s) == n,
            get_subsets(range(1, n + 1))
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

        term_rect = SurroundingRectangle(term[:2])
        term_rect.round_corners()
        term_rect.set_stroke(YELLOW, 2)
        term_words = Text("Is there a snazzy\nway to deduce this?", font_size=36)
        term_words.next_to(term_rect, DOWN)
        term_words.set_color(YELLOW)

        self.play(
            FadeIn(top_rect),
            ShowIncreasingSubsets(subset_mobs, run_time=5)
        )
        self.wait()
        self.play(
            ShowCreation(term_rect),
            Write(term_words),
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeOut, subset_mobs, shift=0.2 * UP),
            FadeOut(top_rect, rate_func=squish_rate_func(smooth, 0.6, 1)),
            run_time=3
        )
        self.wait()

    ##

    def color_set_tex(self, set_tex):
        for value in set_tex.values:
            elem = get_part_by_value(set_tex, value)
            if value - 1 < len(self.elem_colors):
                elem.set_color(self.elem_colors[value - 1])

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
            for ss in stack:
                self.color_set_tex(ss)
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


class ShowHypercubeConstruction(InteractiveScene):
    def construct(self):
        # Intro
        v1 = np.array([1.5, 1.5, 0.5])
        v2 = np.array([-2.5, 0.5, 1.5])
        vects = 1.5 * np.array([
            RIGHT, UP, OUT,
            v1, v2,
            1.5 * (v1 + 2 * OUT), 2 * (v2 + 2 * UP),
            5 * v1, 5 * v2,
        ])
        colors = [GREEN_D, TEAL, BLUE_D, RED, PINK] * 2
        curr_sets = VGroup(get_set_tex([]))
        lines = VGroup()
        frame = self.camera.frame

        self.play(Write(curr_sets))
        self.wait()

        # Grow new sets
        # indices = list(range(1, 6))
        indices = list(range(1, 10))
        for n, vect, color in zip(indices, vects, colors):
            new_sets = VGroup(*(
                get_set_tex([*st.values, n])
                for st in curr_sets
            ))
            for ns in new_sets:
                for value, c in zip(indices, colors):
                    if value in ns.values:
                        get_part_by_value(ns, value).set_color(c)
                ns.set_backstroke()
                if n >= 3:
                    ns.rotate(70 * DEGREES, RIGHT)
            curr_set_copies = curr_sets.copy()
            curr_sets.generate_target()
            new_lines = VGroup()
            line_copies = VGroup()
            for ns, cst, csc in zip(new_sets, curr_sets.target, curr_set_copies):
                ns.move_to(cst)
                ns.shift(vect)
                cst.shift(-vect)
                csc.move_to(ns)
                new_lines.add(Line(
                    cst.get_center(), ns.get_center(),
                    # buff=MED_SMALL_BUFF + ns.length_over_dim(np.argmax(vect)),
                    buff=MED_SMALL_BUFF + ns.get_height(),
                    stroke_color=color,
                    stroke_width=2,
                ))
            line_copies = lines.copy()
            diff = new_sets[0].get_width() - curr_sets[0].get_width()
            for line in line_copies:
                if abs(line.get_vector()[0]) > 1e-2:
                    line.set_width(line.get_width() - diff)
            lines.generate_target()
            line_copies.shift(vect)
            lines.target.shift(-vect)
            anims = [
                MoveToTarget(curr_sets),
                TransformFromCopy(curr_sets, curr_set_copies),
                MoveToTarget(lines),
                TransformFromCopy(lines, line_copies),
                *map(GrowFromCenter, new_lines),
            ]
            if n == 3:
                anims.append(ApplyMethod(
                    frame.reorient, 10, 70,
                    run_time=2,
                ))
                for st in [*curr_set_copies, *curr_sets.target]:
                    st.rotate(70 * DEGREES, RIGHT)
                frame.add_updater(lambda m, dt: frame.increment_theta(0.01 * dt))
            if n >= 4:
                anims.append(frame.animate.scale(1.5).reorient(-10, 70).move_to(2 * IN))
            self.play(*anims)
            self.remove(curr_set_copies)
            self.play(*(
                set_tex_transform(csc, ns)
                for csc, ns in zip(curr_set_copies, new_sets)
            ))
            self.wait()

            curr_sets.set_submobjects([*curr_sets, *new_sets])
            curr_sets.set_backstroke()
            lines.set_submobjects([*lines, *line_copies, *new_lines])
            self.add(lines, curr_sets)
            if n == 3:
                for i in range(1, 4):
                    self.highlight_direction(curr_sets, lines, i)
        self.wait(5)

    def highlight_direction(self, set_texs, lines, index):
        anims = []
        for line in lines:
            if np.argmax(line.get_vector()) == index - 1:
                anims.append(ShowCreation(line))
        self.play(LaggedStart(*anims, lag_ratio=0.25, run_time=1))
        anims = []
        for st in set_texs:
            if index in st.values:
                part = get_part_by_value(st, index)
                outline = part.family_members_with_points()[0].copy()
                outline.set_fill(opacity=0)
                outline.set_stroke(YELLOW, 5)
                anims.append(VShowPassingFlash(outline, time_width=2))
        self.play(LaggedStart(*anims, lag_ratio=0.1, run_time=2))
        self.wait()


class PolynomialConstruction(InteractiveScene):
    def construct(self):
        # (1 + x)
        set_group = VGroup(
            get_set_tex([]),
            get_set_tex([1]),
        )
        set_group.scale(0.5)
        curr_poly = MTex("(1 + x^1)")
        curr_poly[0].set_opacity(0)
        curr_poly[-1].set_opacity(0)

        for st, sym in zip(set_group, curr_poly[1::2]):
            st.next_to(sym, DOWN, MED_LARGE_BUFF)

        self.add(curr_poly, set_group)

        # Expand via subset iteration process
        for n in range(2, 8):
            cp_l, cp_r = curr_poly.replicate(2)
            sg_l, sg_r = set_group.replicate(2)
            plus = MTex("+")
            cp_l.next_to(plus, LEFT, LARGE_BUFF)
            cp_r.next_to(plus, RIGHT, LARGE_BUFF)
            sg_l.match_x(cp_l)
            sg_r.match_x(cp_r)

            self.remove(curr_poly, set_group)
            added_anims = []
            if n == 5:
                added_anims.append(self.camera.frame.animate.set_height(10))
            if n >= 6:
                added_anims.append(self.camera.frame.animate.scale(1.4))
                VGroup(cp_l, sg_l).shift(LEFT)
                VGroup(cp_r, sg_r).shift(RIGHT)
            self.play(
                TransformFromCopy(curr_poly, cp_l),
                TransformFromCopy(curr_poly, cp_r),
                TransformFromCopy(set_group, sg_l),
                TransformFromCopy(set_group, sg_r),
                *added_anims
            )
            self.wait()
            new_term = MTex(f"x^{n}")
            new_poly = MTex("".join([
                f"\\left(1 + x^{k}\\right)"
                for k in range(1, n + 1)
            ]))
            new_term.next_to(cp_r, LEFT, SMALL_BUFF)
            new_term.align_to(cp_r.family_members_with_points()[1], DOWN)
            new_sets = VGroup(*(
                get_set_tex([*st.values, n])
                for st in sg_r
            ))
            anims = [Write(new_term), Write(plus)]
            for ns, s in zip(new_sets, sg_r):
                ns.match_height(s)
            new_sets.arrange_in_grid(
                n_cols=2**int(np.ceil(np.log2(len(new_sets)) / 2))
            )
            new_sets.move_to(sg_r)
            for ns, s in zip(new_sets, sg_r):
                anims.append(set_tex_transform(s, ns))
            if n == 2:
                anims.append(cp_r.animate.set_opacity(1))

            self.remove(sg_r)
            self.play(*anims)
            self.wait()

            new_set_group = VGroup(*sg_l, *new_sets)
            new_set_group.generate_target()
            new_set_group.target.arrange_in_grid(
                n_cols=2**int(np.ceil(np.log2(len(new_set_group)) / 2))
            )
            new_set_group.target.next_to(new_poly, DOWN, MED_LARGE_BUFF)

            group = VGroup(cp_l, cp_r, plus, new_term)
            self.play(group.animate.shift(2 * UP))
            group_copy = group.copy()

            self.add(group_copy)
            self.play(
                LaggedStart(
                    Transform(
                        cp_l, new_poly[:-6].copy().set_opacity(0),
                        path_arc=30 * DEGREES,
                        remover=True,
                    ),
                    ReplacementTransform(new_term, new_poly[-3:-1]),
                    ReplacementTransform(cp_r, new_poly[:-6], path_arc=30 * DEGREES),
                    ReplacementTransform(plus, new_poly[-4], path_arc=45 * DEGREES),
                    Write(new_poly[-6:-4]),
                    Write(new_poly[-1]),
                ),
                MoveToTarget(new_set_group),
                run_time=2,
            )
            self.add(new_poly)
            set_group = new_set_group
            curr_poly = new_poly
            self.play(FadeOut(group_copy))
            self.wait()


class EvaluationTricks(InteractiveScene):
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
        expanded = VGroup(
            MTex("f(x) = ", **tex_kw),
            MTex("\\sum_{n = 0}^{N} c_n x^n", **tex_kw),
            MTex("= 1+x+x^{2}+2 x^{3}+2 x^{4}+ \\cdots", **tex_kw)
        )
        expanded.arrange(RIGHT, buff=0.2)
        expanded.next_to(factored, DOWN, LARGE_BUFF, LEFT)

        factored_label = Text("What we know", color=TEAL_B)
        expanded_label = Text("What we want", color=TEAL_C)
        for label, expr in [(factored_label, factored), (expanded_label, expanded)]:
            label.next_to(expr, LEFT, LARGE_BUFF)
            expr.arrow = Arrow(label, expr)

        self.add(factored)
        self.play(
            Write(factored_label),
            ShowCreation(factored.arrow),
        )
        self.wait()
        self.play(FadeTransform(factored.copy(), expanded))
        self.play(
            Write(expanded_label),
            ShowCreation(expanded.arrow),
        )
        self.wait()

        # Black box
        lhs = expanded[0]
        rhs = expanded[2]
        box = SurroundingRectangle(rhs[1:])
        box.set_stroke(WHITE, 1)
        box.set_fill(GREY_E, 1)
        q_marks = MTex("?").get_grid(1, 7, buff=0.7)
        q_marks.move_to(box)
        box.add(q_marks)

        self.play(FadeIn(box, lag_ratio=0.25, run_time=2))
        self.wait()

        # Show example evaluations
        x_tracker = ValueTracker(0.5)
        get_x = x_tracker.get_value
        dot = GlowDot(color=WHITE)
        dot.add_updater(lambda m: m.move_to(plane.i2gp(get_x(), graph)))
        line = Line(DOWN, UP).set_stroke(WHITE, 1)
        line.add_updater(lambda l: l.put_start_and_end_on(
            plane.c2p(get_x(), 0),
            plane.i2gp(get_x(), graph)
        ))

        self.play(
            Write(plane, lag_ratio=0.01),
            LaggedStartMap(FadeOut, VGroup(factored_label, expanded_label, factored.arrow, expanded.arrow)),
        )
        self.play(ShowCreation(graph))
        self.wait()
        self.play(
            ShowCreation(line),
            FadeInFromPoint(dot, line.get_start()),
        )
        self.play(x_tracker.animate.set_value(-0.5), run_time=2)
        self.play(x_tracker.animate.set_value(0.7), run_time=2)
        self.wait()

        # Plug in 0
        f0 = MTex("f(0) = 1", tex_to_color_map={"0": BLUE})
        f0[-1].set_opacity(0)
        f0.next_to(expanded, DOWN, LARGE_BUFF, aligned_edge=LEFT)

        c0_rhs = MTex("= c_0")
        c0_rhs.next_to(f0, RIGHT)
        c0_rhs.shift(0.05 * DOWN)

        self.play(FadeTransform(lhs.copy(), f0))
        self.play(x_tracker.animate.set_value(0))
        self.wait(note="Move box out of the way")
        f0.set_opacity(1)
        self.play(Write(f0[-1]))
        self.add(f0)
        self.wait()
        self.play(Write(c0_rhs))
        f0.add(c0_rhs)

        # Take derivative at 0
        dkw = dict(tex_to_color_map={"0": BLUE})
        f_prime_0 = MTex("f'(0) = c_1", **dkw)
        f_prime_n = MTex("\\frac{1}{n!} f^{(n)}(0) = c_n", **dkw)
        f_prime_0.next_to(f0, RIGHT, buff=1.5)
        f_prime_n.next_to(f_prime_0, DOWN, MED_LARGE_BUFF, LEFT)

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

        # Plug in 1
        f1 = MTex(
            "f({1}) \\,=\\, 2^{2{,}000} \\,=\\, c_0 + c_1 + c_2 + c_3 + \\cdots + c_N",
            tex_to_color_map={
                "2^{2{,}000}": TEAL,
                "{1}": BLUE,
                "=": WHITE,
            }
        )
        f1.move_to(f0, LEFT)
        self.play(
            TransformFromCopy(factored[:5], f1[:5]),
            f0.animate.shift(2 * DOWN),
        )
        self.wait()
        self.play(
            Write(f1[5:11]),
            x_tracker.animate.set_value(1)
        )
        self.wait(note="Comment on factored form")
        self.play(Write(f1[11:]))
        self.add(f1)

        # Plug in -1
        fm1 = self.load_mobject("f_of_neg1.mob")
        # fm1 = MTex(
        #     "f({-1}) \\,=\\, {0} \\,=\\,"
        #     "c_0 - c_1 + c_2 - c_3 + \\cdots + c_N",
        #     tex_to_color_map={
        #         "{-1}": RED,
        #         "{0}": TEAL,
        #         "=": WHITE,
        #     }
        # )
        fm1.next_to(f1, DOWN, LARGE_BUFF, LEFT)

        self.play(
            TransformMatchingShapes(f1[:5].copy(), fm1[:5]),
            FadeOut(f0, DOWN)
        )
        self.wait()
        self.play(
            Write(fm1[5:7]),
            ApplyMethod(x_tracker.set_value, -1, run_time=3)
        )
        self.wait()
        self.play(Write(fm1[7:]))
        self.wait()

        # Show filtration expression
        f1_group = VGroup(f1, fm1)
        self.play(
            FadeOut(expanded),
            FadeOut(box),
            f1_group.animate.move_to(expanded, UL),
        )

        h_line = Line(LEFT, RIGHT).match_width(f1_group)
        h_line.set_stroke(GREY_B, 3)
        h_line.next_to(f1_group, DOWN, MED_LARGE_BUFF)
        h_line.stretch(1.05, 0, about_edge=LEFT)

        filter_expr = MTex(
            "{1 \\over 2} \\Big(f({1}) + f({-1})\\Big)"
            "= c_0 + c_2 + c_4 + \\cdots + c_{N}",
            tex_to_color_map={
                "{1}": BLUE,
                "{-1}": RED,
            }
        )
        filter_expr.next_to(h_line, DOWN, MED_LARGE_BUFF)
        filter_expr.align_to(f1_group, LEFT)

        self.play(
            ShowCreation(h_line),
            TransformFromCopy(f1[:4], filter_expr[4:8]),
            TransformFromCopy(fm1[:4], filter_expr[9:14]),
            Write(filter_expr[:4]),
            Write(filter_expr[8]),
            Write(filter_expr[14:16]),
        )
        self.wait()
        self.play(Write(filter_expr[16:]))
        self.wait()

        # Clarify goal
        parens = MTex("()")
        words = TexText("Some clever\\\\evaluation of $f$", font_size=36)
        words.set_color(WHITE)
        parens.match_height(words)
        parens[0].next_to(words, LEFT, buff=SMALL_BUFF)
        parens[1].next_to(words, RIGHT, buff=SMALL_BUFF)
        desire = VGroup(
            VGroup(parens, words),
            MTex("= c_0 + c_5 + c_{10} + \\cdots + c_{N}")
        )
        desire.arrange(RIGHT)
        desire.next_to(expanded, DOWN, 1.0, LEFT)

        self.play(
            FadeIn(expanded),
            FadeOut(f1_group, DOWN),
            Uncreate(h_line),
            filter_expr.animate.to_edge(DOWN),
        )
        self.wait()
        self.play(
            FadeIn(desire, DOWN),
        )
        self.wait()

        # Indicator on x^n
        new_rhs = expanded[1]
        rect = SurroundingRectangle(new_rhs.get_part_by_tex("x^n"), buff=0.05)
        rect.set_stroke(BLUE_B, 2)
        rect.round_corners()

        outcomes = VGroup(
            TexText("$1$ if $\\; 5 \\mid n$", font_size=36, color=GREEN),
            TexText("$0$ if $\\; 5 \\nmid n$", font_size=36, color=RED_D),
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

        self.play(
            ShowCreation(rect),
            FadeOut(expanded[2]),
        )
        self.wait(0.25)
        for arrow, outcome in zip(arrows, outcomes):
            self.play(
                ShowCreation(arrow),
                GrowFromPoint(outcome, rect.get_right())
            )
            self.wait()


class MotivateRootsOfUnity(InteractiveScene):
    def construct(self):
        # Add generating function
        pieces = ["f(x)", "="]
        n_range = list(range(0, 7))
        for n in n_range:
            pieces.extend([f"c_{{{n}}}", f"x^{{{n}}}", "+"])
        pieces.extend(["\\cdots"])
        polynomial = MTex("\\,".join(pieces), isolate=pieces)
        polynomial.to_edge(UP)

        exp_parts = VGroup(*(
            polynomial.get_part_by_tex(f"x^{{{n}}}")
            for n in n_range
        ))
        coef_parts = VGroup(*(
            polynomial.get_part_by_tex(f"c_{{{n}}}")
            for n in n_range
        ))

        fm1 = MTex(
            "f({-1}) = c_0 - c_1 + c_2 - c_3 + c_4 - c_5 + c_6 - \\cdots",
            tex_to_color_map={"{-1}": RED}
        )
        fm1.next_to(polynomial, DOWN, buff=1.5, aligned_edge=LEFT)

        self.add(polynomial)

        # Real number line
        line = NumberLine((-4, 4), unit_size=2)
        line.move_to(DOWN)
        line.add_numbers()
        self.play(Write(line))
        self.wait()

        # Show oscillation
        v1 = Arrow(line.n2p(0), line.n2p(1), buff=0, color=BLUE)
        vm1 = Arrow(line.n2p(0), line.n2p(-1), buff=0, color=RED)
        vect = v1.copy()
        neg1_powers = [
            MTex(f"(-1)^{{{n}}}")
            for n in range(8)
        ]
        for n, neg1_power in enumerate(neg1_powers):
            v = [v1, vm1][n % 2]
            neg1_power.next_to(v, UP)

        power = neg1_powers[0].copy()

        self.play(Write(fm1[:6]))
        self.wait()
        self.play(FlashAround(exp_parts[0]))
        self.play(
            FadeTransform(exp_parts[0].copy(), vect),
            FadeTransform(coef_parts[0].copy(), power),
            Write(fm1[6:8]),
        )
        self.wait()
        for n in range(1, 7):
            self.play(FlashAround(exp_parts[n]))
            self.play(
                Transform(vect, [v1, vm1][n % 2], path_arc=PI),
                Transform(power, neg1_powers[n], path_arc=PI / 2),
            )
            self.wait()
            self.play(
                FadeTransform(
                    coef_parts[n].copy(),
                    fm1[5 + 3 * n:8 + 3 * n]
                )
            )
            self.wait()
        self.play(
            Write(fm1[-4:]),
            FadeOut(power),
        )
        self.wait()

        # Bring in complex plane
        plane_unit = 1.75
        plane = ComplexPlane(
            (-3, 3), (-2, 2),
            axis_config=dict(unit_size=plane_unit)
        )
        plane.move_to(line.n2p(0) - plane.n2p(0))
        plane.add_coordinate_labels(font_size=24)

        plane_label = Text("Complex plane", font_size=36)
        plane_label.set_backstroke()
        plane_label.next_to(plane.get_corner(UL), DR, SMALL_BUFF)

        self.add(plane, polynomial, vect)
        sf = plane_unit / line.get_unit_size()
        self.play(
            Write(plane, lag_ratio=0.01),
            FadeOut(fm1),
            line.animate.scale(sf).set_opacity(0),
            vect.animate.set_color(YELLOW).scale(sf, about_edge=LEFT),
        )
        self.remove(line)
        self.play(Write(plane_label))
        self.wait()

        # Preview roots of unity
        unit_circle = Circle(radius=plane_unit)
        unit_circle.move_to(plane.n2p(0))
        unit_circle.set_stroke(GREY, 2)

        points = [
            plane.n2p(np.exp(complex(0, angle)))
            for angle in np.arange(0, TAU, TAU / 5)
        ]
        dots = GlowDots(points)

        pentagon = Polygon(*points)
        pentagon.set_stroke(TEAL, 1)
        pentagon.set_fill(TEAL, 0.25)
        pentagon.save_state()
        pentagon.set_opacity(0)

        self.add(pentagon, unit_circle, vect, dots)
        self.play(
            ShowCreation(unit_circle),
            ShowCreation(dots)
        )
        self.wait()
        self.play(pentagon.animate.restore())
        self.wait()

        last_rect = VMobject()
        for n in range(11):
            if n == 0:
                continue
            if n < len(exp_parts):
                exp_part = exp_parts[n]
            else:
                exp_part = polynomial[-3:]
            rect = SurroundingRectangle(exp_part, buff=0.05)
            rect.round_corners()
            rect.set_stroke(YELLOW, 2)
            self.play(
                FadeOut(last_rect),
                ShowCreation(rect),
                ApplyMethod(
                    vect.put_start_and_end_on,
                    plane.n2p(0), points[n % 5],
                    path_arc=TAU / 5,
                )
            )
            last_rect = rect
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

        # Setup roots
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
        self.disable_interaction(root_dots)

        pentagon = Polygon(*root_points)
        pentagon.set_stroke(TEAL, 2)
        pentagon.set_fill(TEAL, 0.25)
        self.add(pentagon)

        # Add function label
        function = MTex(
            "f(x) = \\sum_{n = 0}^N c_n x^n",
            tex_to_color_map={"x": BLUE}
        )
        function.move_to(midpoint(plane.get_right(), RIGHT_SIDE))
        function.to_edge(UP, buff=MED_SMALL_BUFF)
        self.add(function)

        # Roots of unity
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
        zeta_labels.set_backstroke(width=4)
        for point, label in zip(root_points, zeta_labels):
            vect = normalize(point - plane.get_origin())
            if point is root_points[0]:
                vect = UR
            label.next_to(point, vect, buff=SMALL_BUFF)
        exp_rhs = MTex(" = e^{2\\pi i / 5}", **root_kw)
        trig_rhs = MTex("= \\cos(72^\\circ) + i\\cdot \\sin(72^\\circ)", **root_kw)
        last = zeta_labels[1]
        for rhs in exp_rhs, trig_rhs:
            rhs.set_backstroke(width=4)
            rhs.next_to(last, RIGHT, SMALL_BUFF)
            last = rhs
        exp_rhs.shift((trig_rhs[0].get_y() - exp_rhs[0].get_y()) * UP)

        self.play(
            FadeInFromPoint(
                root_dots[1], plane.n2p(1),
                path_arc=TAU / 5,
            ),
            pentagon.animate.set_fill(opacity=0.1)
        )
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
            self.play(*(
                TransformFromCopy(group[i - 1], group[i % 5], path_arc=-TAU / 5)
                for group in [root_lines, root_dots, zeta_labels]
            ))
            self.wait()

        # Name the roots of unity
        title = TexText("``Fifth roots of unity''")
        title.set_color(YELLOW)
        title.match_y(plane)
        title.match_x(function)
        equation = Tex("z^5 = 1")
        equation.set_color(WHITE)
        equation.next_to(title, DOWN)

        self.play(
            Write(title),
            LaggedStart(*(
                FlashAround(zl, time_width=1.5)
                for zl in zeta_labels
            ), lag_ratio=0.1, run_time=3)
        )
        self.wait()
        self.play(FadeIn(equation, 0.5 * DOWN))
        self.wait()
        self.play(
            LaggedStartMap(FadeOut, VGroup(title, equation), shift=DOWN),
            FadeOut(VGroup(arc, arc_label)),
        )

        # Key expression
        expr = MTex("+".join([f"f(\\zeta^{{{n}}})" for n in range(5)]), **root_kw)
        expr.next_to(function, DOWN, LARGE_BUFF)

        self.play(
            TransformMatchingShapes(function[:4].copy(), expr, run_time=1.5),
            FadeOut(pentagon),
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
                self.animate_kth_power(
                    plane,
                    root_dots, k,
                )
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
            "c_0 + c_5 + c_{10} + \\cdots"
            "=\\frac{1}{5}\\sum_{k = 0}^4 f(\\zeta^k)",
            tex_to_color_map={"\\zeta": YELLOW}
        )
        answer.set_width(5)
        answer.next_to(function, DOWN, LARGE_BUFF)
        answer_rect = SurroundingRectangle(answer, buff=0.2)
        answer_rect.round_corners()
        answer_rect.set_stroke(YELLOW, 2)
        self.disable_interaction(answer_rect)

        self.play(
            FadeOut(example, DOWN),
            relation_group.animate.set_width(4.5).to_edge(DOWN),
            Write(answer)
        )
        self.play(
            VShowPassingFlash(answer_rect.copy()),
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
        lower_group.target.arrange(RIGHT, buff=MED_LARGE_BUFF)
        lower_group.target.set_width(8.5)
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
        zp1_labels[0].next_to(root_dots[0].get_center() + shift_vect, UL, SMALL_BUFF)

        new_circle = unit_circle.copy()
        new_circle.set_stroke(GREY_B, opacity=0.5)
        self.disable_interaction(new_circle)
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

        # Estimate answer
        estimate = MTex(
            "= 2 \\cdot |\\zeta + 1|^2 \\cdot |\\zeta^2 + 1|^2",
            **eq_kw
        )
        estimate.next_to(plane, RIGHT, buff=1.5, aligned_edge=UP)

        self.play(Write(estimate[1]))
        self.wait()
        self.play(
            Write(estimate[0]),
            Write(estimate[2:9]),
        )
        self.wait()
        self.play(
            Write(estimate[9:]),
        )
        self.wait()

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

        self.play(
            FadeOut(estimate),
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
        self.remove(arrows, values)
        expr_group.add(*arrows, *values)

        # Rescale answer group
        plane_group = Group(
            plane, unit_circle, new_circle,
            root_lines, root_dots, zp1_labels,
            complex_plane_title,
        )

        ans_group.generate_target()
        ans_group.target.set_width(6.5, about_edge=LEFT)
        ans_group.target.next_to(brace, DOWN)
        ans_group.target.shift(1 * LEFT)
        ans_group.target[0].set_stroke(opacity=0)

        f_zeta_rhs = Tex("=2^{400}").set_color(WHITE)
        f_zeta_rhs.next_to(f_zeta, RIGHT)

        self.play(
            MoveToTarget(ans_group),
            expr_group.animate.scale(1 / 1.5, about_edge=DR),
            plane_group.animate.scale(0.7, about_edge=DL),
            TransformMatchingShapes(two_label, f_zeta_rhs),
            FadeOut(brace),
            run_time=2
        )
        self.wait()

        # Final answer
        final_answer = MTex(
            "= \\frac{1}{5}\\Big("
            "2^{2{,}000} + 4 \\cdot 2^{400}\\Big)"
        )
        final_answer.next_to(answer, RIGHT)

        final_answer_rect = SurroundingRectangle(final_answer[1:], buff=0.2)
        final_answer_rect.round_corners()
        final_answer_rect.set_stroke(YELLOW, 2)
        self.disable_interaction(final_answer_rect)

        self.play(
            Write(final_answer[:5]),
            Write(final_answer[-1]),
        )
        self.wait()
        self.play(
            FadeTransform(values.copy(), final_answer[5:-1])
        )
        self.play(ShowCreation(final_answer_rect))
        self.wait(note="Comment on dominant term")

        # Smaller case
        box = Rectangle(
            width=(RIGHT_SIDE[0] - plane.get_right()[0]) - 1,
            height=plane.get_height()
        )
        box.to_corner(DR)
        box.align_to(plane, DOWN)
        box.set_stroke(WHITE, 1)
        box.set_fill(GREY_E, 1)
        set_tex = get_set_tex(range(1, 6))
        set_tex.next_to(box.get_left(), RIGHT, SMALL_BUFF)

        rhs = MTex(
            "\\rightarrow \\frac{1}{5}"
            "\\Big(2^5 + 4 \\cdot 2^1\\Big)"
            "= \\frac{1}{5}(32 + 8)"
            "=8"
        )
        rhs.scale(0.9)
        rhs.next_to(set_tex, RIGHT)

        self.play(
            FadeOut(expr_group),
            Write(box),
            Write(set_tex),
        )
        self.wait()
        self.play(
            Write(rhs[0]),
            FadeTransform(final_answer.copy(), rhs[1:13])
        )
        self.wait()
        self.play(Write(rhs[13:]))
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
        # pentagon.add_updater(lambda p: p.set_points_as_corners([
        #     d.get_center() for d in [*dots, dots[0]]
        # ]))

        self.play(*(
            MoveAlongPath(dot, path)
            for dot, path in zip(dots, paths)
        ), run_time=4)
        if k == 5:
            self.wait()
            self.play(dots.animate.restore())
        else:
            # self.play(FadeOut(pentagon))
            dots.restore()
            # self.play(FadeInFromDown(pentagon))
        # pentagon.clear_updaters()

    def get_highlight(self, mobject):
        result = super().get_highlight(mobject)
        if isinstance(mobject, Arrow):
            result.set_stroke(width=result.get_stroke_width())
        return result


class JustifyLinearity(InteractiveScene):
    n_terms = 8

    def construct(self):
        # Create polynomials
        x_poly = self.get_polynomial("{x}", BLUE)
        x_poly.to_edge(UP)

        zeta_polys = VGroup(*(
            self.get_polynomial("\\zeta", YELLOW, exp_multiple=n)
            for n in range(5)
        ))
        zeta_polys.arrange(DOWN, buff=0.65, aligned_edge=LEFT)
        zeta_polys.next_to(x_poly, DOWN, LARGE_BUFF)
        x_poly.align_to(zeta_polys, LEFT)
        polys = VGroup(x_poly, *zeta_polys)

        # Align parts
        low_poly = polys[-1]
        for poly in polys[:-1]:
            for i1, i2 in zip(poly.sep_indices, low_poly.sep_indices):
                shift = (low_poly[i2].get_x(LEFT) - poly[i1].get_x(LEFT)) * RIGHT
                poly[i1].shift(shift / 2)
                poly[i1 + 1:].shift(shift)

        # Prepare for grid sum
        lhss = VGroup(*(
            poly[:poly.sep_indices[0]]
            for poly in zeta_polys
        ))
        lhss.save_state()
        plusses = MTex("+", font_size=36).replicate(len(lhss) - 1)
        group = VGroup(*it.chain(*zip(lhss, plusses)))
        group.add(lhss[-1])
        group.arrange(RIGHT, buff=MED_SMALL_BUFF)
        group.next_to(x_poly, DOWN, LARGE_BUFF)

        plusses.generate_target()
        for plus, lhs1, lhs2 in zip(plusses.target, lhss.saved_state, lhss.saved_state[1:]):
            plus.move_to(midpoint(lhs1.get_bottom(), lhs2.get_top()))
            plus.scale(0.7)

        self.add(x_poly)
        self.play(Write(group))
        self.wait()
        self.play(
            lhss.animate.restore(),
            MoveToTarget(plusses),
            path_arc=-90 * DEGREES,
        )
        self.wait()

        # Write all right hand sides
        anims = []
        for poly in zeta_polys:
            anims.append(FadeTransformPieces(
                x_poly[x_poly.sep_indices[0]:].copy(),
                poly[poly.sep_indices[0]:],
            ))
        self.play(LaggedStart(*anims, lag_ratio=0.5, run_time=3))
        self.wait()

        # Highlight columns
        lower_sum = VGroup()
        lp = polys[-1]
        for j in range(len(x_poly.sep_indices) - 1):
            polys.generate_target()
            col = VGroup()
            for poly in polys.target:
                col.add(poly[poly.sep_indices[j] + 1:poly.sep_indices[j + 1]])
            polys.target.set_fill(opacity=0.5)
            col.set_fill(opacity=1)

            if j % 5 == 0:
                term = MTex(f"5c_{j}", font_size=36)
            else:
                term = MTex("0", font_size=36)
            i1, i2 = lp.sep_indices[j:j + 2]
            term.next_to(VGroup(lp[i1], lp[i2]), DOWN, buff=0.7)
            plus = MTex("+", font_size=36)
            plus.next_to(lp[i1], DOWN, buff=0.7)
            if j == 0:
                plus.set_opacity(0)
            else:
                plus.move_to(midpoint(lower_sum.get_right(), term.get_left()))
            lower_sum.add(plus, term)

            self.play(
                FlashAround(col, run_time=2, time_width=2),
                MoveToTarget(polys),
                plusses.animate.set_fill(opacity=0.5),
            )
            self.wait()
            self.play(
                Write(plus),
                *(
                    FadeTransform(piece.copy(), term)
                    for piece in col[1:]
                )
            )
            self.wait()
        self.play(polys.animate.set_opacity(1))
        self.wait()

    def get_polynomial(self, input_tex, color, exp_multiple=None, font_size=36):
        if exp_multiple is not None:
            tex = f"f({input_tex}^{exp_multiple}) ="
        else:
            tex = f"f({input_tex}) = "
            exp_multiple = 1
        tex += "c_0 + "
        for n in range(1, self.n_terms):
            tex += f"c_{{{n}}}{input_tex}^{{{exp_multiple * n}}} + "
        tex += "\\cdots"
        tex = tex.replace("+", "\\,+\\,")
        tex = tex.replace("=", "\\,=\\,")
        result = MTex(
            tex,
            tex_to_color_map={
                input_tex: color,
                "+": WHITE,
                "=": WHITE,
            },
            font_size=font_size,
        )
        result.sep_indices = [
            result.submobjects.index(part.family_members_with_points()[0])
            for part in (*result.get_parts_by_tex("="), *result.get_parts_by_tex("+"))
        ]
        # result.arrange(RIGHT, aligned_edge=DOWN)
        return result


class ExpandOutKeyEquation(InteractiveScene):
    def construct(self):
        pass


class ReflectOnNumericalAnswer(InteractiveScene):
    def construct(self):
        pass


class RotationalSum(InteractiveScene):
    n_iterations = 40

    def construct(self):
        # Intro
        outer_r = 3
        inner_r = 2.0
        circle = Circle(radius=outer_r)
        circle.set_stroke(GREY_B, 2)
        points = outer_r * compass_directions(5)
        values = VGroup(*(
            Integer(0, edge_to_fix=ORIGIN).move_to(point)
            for point in points
        ))
        values[0].edge_to_fix = LEFT
        rects = VGroup(*(BackgroundRectangle(v) for v in values))
        rects.set_fill(BLACK, 1)

        def update_rects(rects):
            for rect, value in zip(rects, values):
                rect.replace(value, stretch=True)
                rect.set_width(value.get_width() + SMALL_BUFF)

        rects.add_updater(update_rects)

        def update_values(values):
            for v in values:
                v.set_max_width(1, about_edge=v.edge_to_fix)

        values.add_updater(update_values)

        values[0].increment_value()

        self.add(circle)
        self.add(rects, values)

        # Add labels
        labels = VGroup(*(
            TexText(f"sum $\\equiv$ {n}\\\\mod 5", font_size=30)
            for n in range(5)
        ))
        colors = [YELLOW, *color_gradient([BLUE_B, BLUE_D], 4)]
        for label, rect, color in zip(labels, rects, colors):
            label.set_fill(color)
            label.add_background_rectangle()
            # label.next_to(rect, DOWN, SMALL_BUFF)
            label.next_to(rect, normalize(rect.get_center()), SMALL_BUFF)
        labels[0].next_to(rects[0], DR, SMALL_BUFF)

        self.add(labels)

        # Corner set
        set_tex = get_set_tex([])
        set_tex.scale(0.75)
        set_tex.to_corner(UL)
        set_tex.shift(LEFT)
        self.camera.frame.move_to(LEFT)

        self.add(set_tex)

        # Iterative all
        for n in range(1, self.n_iterations + 1):
            value_copies = values.copy()
            to_add = [0] * 5
            for i, vc in enumerate(value_copies):
                new_i = (i + n) % 5
                vc.move_to((inner_r / outer_r) * points[new_i])
                to_add[new_i] = vc.get_value()
                vc.target = VectorizedPoint(values[new_i].get_center())
                vc.target.set_opacity(0)

            reduced_n = n % 5
            path_arc = min(reduced_n * TAU / 5, TAU / 3)

            rot_label = Text(f"Add {n}", font_size=30)
            arcs = VGroup(Arc(0, path_arc), Arc(TAU / 2, path_arc))
            for arc in arcs:
                arc.add_tip(width=0.1, length=0.1)
            if reduced_n == 0:
                arcs.set_opacity(0)

            new_set_tex = get_set_tex([*set_tex.values, n])
            new_set_tex.match_height(set_tex)
            new_set_tex.move_to(set_tex, LEFT)

            self.remove(set_tex)
            self.play(
                TransformFromCopy(
                    values, value_copies,
                    path_arc=-path_arc
                ),
                FadeIn(rot_label),
                TransformMatchingTex(
                    set_tex, new_set_tex,
                    fade_transform_mismatches=(len(set_tex) == len(new_set_tex)),
                    remover=True,
                ),
                *map(ShowCreation, arcs)
            )
            set_tex = new_set_tex
            self.add(set_tex)

            self.play(*map(FadeOut, [rot_label, *arcs]))
            self.play(
                *(
                    ChangeDecimalToValue(
                        v, v.get_value() + ta, time_span=(0.5, 1),
                    )
                    for v, ta in zip(values, to_add)
                ),
                *(
                    MoveToTarget(vc, remover=True)
                    for v, vc in zip(values, value_copies)
                ),
            )
            self.wait()


# Quick filler


class TopicRolidex(InteractiveScene):
    def construct(self):
        words = VGroup(
            Text("Algebra"),
            Text("Number theory"),
            Text("Analysis"),
            Text("Topology"),
            Text("Geometry"),
            Text("Combinatorics"),
            Text("Group theory"),
        )
        words.set_submobject_colors_by_gradient(BLUE, GREEN, YELLOW)
        words.arrange(DOWN, aligned_edge=LEFT)
        words.next_to(ORIGIN, DR)
        words.shift(UP)

        def update_words(ws):
            for word in ws:
                word.set_opacity(1.0 - clip(1.25 * abs(word.get_y()), 0, 1))

        words.add_updater(update_words)

        lhs = Text("Exampe in ")
        lhs.center()
        words.next_to(lhs, RIGHT, submobject_to_align=words[0])
        self.add(lhs)

        self.add(words)
        self.play(
            words.animate.shift(-words[-2].get_y() * UP),
            UpdateFromFunc(words, update_words),
            run_time=4
        )
        self.wait()


class IMOLogo(InteractiveScene):
    def construct(self):
        # Intro
        logo = ImageMobject("IMO_logo")
        logo.set_width(3)
        logo.to_corner(UL, buff=0.25)
        us_flag = ImageMobject("US_Flag")
        us_flag.set_height(0.5 * logo.get_height())
        us_flag.move_to(logo)

        title = Text("International\nMath\nOlympiad")
        title.next_to(logo, DOWN).to_edge(LEFT)

        self.play(FadeIn(us_flag, scale=0.75))
        self.wait()
        self.play(
            FadeOut(us_flag, UP),
            FadeIn(logo, UP),
            Write(title, run_time=1.5),
        )
        title.set_stroke(background=True)
        self.play(title.animate.set_backstroke(width=5))
        self.wait()


class BookQuestionHighlight(InteractiveScene):
    def construct(self):
        # Temp
        img = ImageMobject("/Users/grant/Dropbox/3Blue1Brown/videos/2022/puzzles/subsets/images/BookQuestion.jpg")
        img.set_height(FRAME_HEIGHT)
        self.disable_interaction(img)

        # Highlight
        rect = FullScreenFadeRectangle(fill_color=BLACK, fill_opacity=0.975)
        lil_rect = Rectangle(5, 0.55)
        lil_rect.move_to([-0.2, 0, 0])
        lil_rect.reverse_points()
        rect.append_vectorized_mobject(lil_rect)

        self.play(
            FlashAround(
                lil_rect,
                color=BLACK,
                stroke_width=5,
                time_width=1.5,
                buff=0,
                run_time=2,
            ),
            FadeIn(rect, time_span=(0.5, 2)),
        )
        self.wait()

        # Title
        title = get_question_title()
        title.set_backstroke(width=7)

        self.play(
            FadeTransform(lil_rect.copy().set_opacity(0), title)
        )
        self.wait()
        self.play(
            LaggedStart(*(
                FlashAround(int_part)
                for int_part in get_integer_parts(title[1])
            ), lag_ratio=0.05, run_time=3),
        )
        self.wait(2)

        last_part = title[-1][-12:]
        self.play(
            FlashUnder(last_part, color=TEAL, buff=0, time_width=1.5, run_time=2),
            last_part.animate.set_fill(TEAL),
        )
        self.wait()


class ConfusedPi(InteractiveScene):
    def construct(self):
        randy = Randolph(mode="erm", height=2)
        self.play(randy.animate.change("maybe").look(UP))
        self.play(Blink(randy))
        self.wait()
        self.play(randy.animate.change("pondering").look(DL))
        self.play(Blink(randy))
        self.wait(2)


class HowManyIntotal(TeacherStudentsScene):
    def construct(self):
        self.teacher_says(
            TexText("How many total\\\\subsets are there?"),
            bubble_kwargs=dict(width=4, height=3),
            added_anims=[self.get_student_changes(
                "happy", "pondering", "tease",
                look_at_arg=self.screen,
            )],
            run_time=2,
        )
        self.wait(4)


class IsThereAGeometry(InteractiveScene):
    def construct(self):
        randy = Randolph(height=1.5)
        randy.shift(RIGHT)

        self.play(PiCreatureBubbleIntroduction(
            randy, TexText("Is there a\\\\geometric structure?"),
            target_mode="pondering",
            look_at_arg=[10, -3, 0],
            bubble_class=ThoughtBubble,
            bubble_kwargs=dict(width=3, height=1.75)
        ))
        for x in range(2):
            self.play(Blink(randy))
            self.wait(2)
            self.play(randy.animate.change("thinking"))


class SimpleQuestionTitle(InteractiveScene):
    def construct(self):
        self.add(get_question_title())


class FinalAnswerTitle(InteractiveScene):
    def construct(self):
        answer = MTex("\\frac{1}{5}\\big(2^{2{,}000} + 4 \\cdot 2^{400}\\big)")
        self.add(answer)


class ComingUpWrapper(VideoWrapper):
    title = "Generating functions"
    wait_time = 0

    def construct(self):
        super().construct()
        self.wait(10)
        kw = self.title_config
        new_title = TexText("A Complex trick".replace("C", "$\\mathds{C}$"), **kw)
        parens = TexText("(secretly Fourier)", **kw)
        new_title.move_to(self.title_text, DOWN)
        new_title.generate_target()
        group = VGroup(new_title.target, parens)
        group.arrange(RIGHT, MED_LARGE_BUFF)
        group.move_to(self.title_text, DOWN)

        self.boundary.clear_updaters()
        self.play(
            FadeOut(self.title_text, UP),
            FadeIn(new_title, UP),
        )
        self.wait(4)
        self.play(
            Write(parens),
            MoveToTarget(new_title)
        )
        self.wait(10)


class IsItHelpful(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            "Is that helpful here?",
            student_index=2,
            added_anims=[LaggedStart(
                self.students[0].animate.change("confused", self.screen),
                self.students[1].animate.change("maybe", self.screen),
                lag_ratio=0.5,
            )]
        )
        self.wait(2)
        self.teacher_says(
            "Honestly...\nnot really.",
            target_mode="shruggie",
            added_anims=[self.get_student_changes(
                "sassy", "erm",
                look_at_arg=self.teacher.eyes,
            )]
        )
        self.play(self.students[2].animate.change("angry"))
        self.wait(3)


class AskAboutOrganization(InteractiveScene):
    def construct(self):
        pass


class NoteToPatrons(InteractiveScene):
    def construct(self):
        morty = Mortimer()
        morty.to_corner(DR)
        title = Text("(Note for patrons)", font_size=60)
        title.to_edge(UP)

        self.play(
            morty.animate.change("raise_right_hand"),
            FadeInFromPoint(title, morty.get_corner(UL)),
        )
        modes = ["happy", "tease", "shruggie", "gracious"]
        for x in range(70):
            if random.random() < 0.3:
                self.play(Blink(morty))
            elif random.random() < 0.1:
                self.play(morty.animate.change(
                    random.choice(modes),
                    title,
                ))
            else:
                self.wait()


class QuestionMorph(InteractiveScene):
    def construct(self):
        question = TexText(
            "How many subsets are there\\\\",
            "with a sum divisible by 5", "?"
        )
        self.play(Write(question))
        self.wait(2)
        self.play(
            question[1].animate.scale(0.75).set_opacity(0.5).to_corner(DL),
            question[2].animate.next_to(question[0], RIGHT, SMALL_BUFF),
        )
        self.wait()


class SimpleRect(InteractiveScene):
    def construct(self):
        rect = Rectangle(2, 0.5)
        rect.set_stroke(YELLOW, 2)
        self.play(ShowCreation(rect))
        self.wait()
        self.play(FadeOut(rect))


class FirstTrick(TeacherStudentsScene):
    def construct(self):
        self.teacher_says(
            "Here's our\nfirst trick",
            target_mode="tease",
            run_time=1
        )
        self.change_student_modes(
            "pondering", "thinking", "sassy",
            look_at_arg=self.screen
        )
        self.wait(4)

        # Here
        self.student_says(
            TexText("Huh? What is $x$?"),
            student_index=2,
            target_mode="confused",
            look_at_arg=self.teacher.eyes,
        )
        self.change_student_modes(
            "erm", "pondering",
            look_at_arg=self.screen,
        )
        self.wait(4)


class GeneratingFunctions(InteractiveScene):
    def construct(self):
        # Title
        title = Text("Generating functions!", font_size=72)
        title.to_edge(UP)
        title.set_color(BLUE)
        underline = Underline(title, buff=-0.05)
        underline.scale(1.25)
        underline.insert_n_curves(20)
        underline.set_stroke(BLUE_B, width=[0, 3, 3, 3, 0])
        self.play(
            Write(title),
            ShowCreation(underline),
            run_time=1
        )

        # First poly
        poly = MTex(
            "1+1 x^{1}+1 x^{2}+2 x^{3}+2 x^{4}+"
            "3 x^{5}+4 x^{6}+5x^{7}+6x^{8}+\\cdots"
        )
        poly.next_to(underline, DOWN, LARGE_BUFF)
        coefs = VGroup(poly[0], poly[2], *poly[6:-4:4])

        subsets = get_subsets(list(range(1, 9)))
        subset_groups = VGroup().replicate(len(coefs))
        for subset in subsets:
            index = sum(subset)
            if index <= 8:
                subset_groups[index].add(get_set_tex(subset))

        subset_groups.set_width(0.7)
        subset_groups.set_color(BLUE_B)

        self.play(FadeIn(poly, DOWN))
        self.wait()

        rects = VGroup()
        for ssg, coef in zip(subset_groups, coefs):
            ssg.arrange(DOWN, buff=SMALL_BUFF)
            ssg.next_to(coef, DOWN, buff=MED_LARGE_BUFF)
            rect = SurroundingRectangle(coef, buff=0.1)
            rect.round_corners()
            rect.set_stroke(BLUE, 1)
            rects.add(rect)
            coef.set_color(BLUE_B)

            self.add(rect, ssg)
            self.play(ShowIncreasingSubsets(
                ssg,
                int_func=np.ceil,
                run_time=0.5,
                rate_func=linear,
            ))
            self.wait(0.5)
            self.remove(rect)

        # Fibbonacci poly
        fib_poly = self.load_mobject("fib_poly.mob")
        # fib_poly = MTex(
        #     "1+1 x^{1}+2 x^{2}+3 x^{3}+5 x^{4}+"
        #     "8 x^{5}+13 x^{6}+21x^{7}+34x^{8}+\\cdots"
        # )
        fib_poly.match_width(poly)
        fib_poly.next_to(subset_groups, DOWN, LARGE_BUFF)
        fib_poly.align_to(poly, LEFT)

        rhs = Tex("=\\frac{x}{1-x-x^{2}}")
        rhs.next_to(fib_poly, DOWN, MED_LARGE_BUFF)

        h_line = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        h_line.set_stroke(GREY, 1)
        h_line.next_to(fib_poly, UP, MED_LARGE_BUFF)

        self.play(
            ShowCreation(h_line),
            Write(fib_poly),
        )
        self.wait()
        self.play(FadeIn(rhs, DOWN))
        self.wait()


class AskAboutTaylorSeries(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            TexText("Can we use derivatives\\\\in some way?"),
            student_index=0,
        )
        self.play(
            self.teacher.animate.change("tease"),
            self.get_student_changes(
                None, "confused", "thinking",
                look_at_arg=self.students[0].bubble,
            ),
        )
        self.wait(4)


class ToTheComplexPlane(InteractiveScene):
    def construct(self):
        plane = ComplexPlane()
        plane.add_coordinate_labels()
        label = Text("Complex plane")
        label.to_corner(UL, buff=MED_SMALL_BUFF)
        label.set_backstroke(width=10)
        poly = MTex(
            "f(x) = (1 + x)(1 + x^2)\\cdots\\left(1+x^{2{,}000}\\right)",
            tex_to_color_map={"x": YELLOW}
        )
        poly.scale(0.75)
        poly.next_to(label, DOWN, MED_LARGE_BUFF, LEFT)
        poly.set_backstroke(width=8)

        self.play(
            Write(plane),
            Write(label),
        )
        self.play(FadeIn(poly, DOWN))
        self.wait()


class StudentSaysWhat(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            "I'm sorry, what?!",
            target_mode="angry"
        )
        self.play(
            self.teacher.animate.change("happy"),
            self.get_student_changes("confused", "sassy", "angry")
        )
        self.look_at(self.screen)
        self.wait(4)


class HangInThere(TeacherStudentsScene):
    def construct(self):
        self.change_student_modes(
            "sad", "tired", "depressed",
            look_at_arg=self.screen,
        )
        self.teacher_says(
            "Hang in\nthere!",
            target_mode="surprised",
        )
        self.change_student_modes(
            "plain", "tease", "happy",
            look_at_arg=self.screen,
        )
        self.play(self.teacher.animate.change("tease"))
        self.wait(4)


class RootOfUnityRearranging(InteractiveScene):
    def construct(self):
        kw = dict(tex_to_color_map={"\\zeta": YELLOW})
        expressions = VGroup(
            MTex("z^5 = 1", **kw),
            MTex("z^5 - 1 = 0", **kw),
            MTex("z^5 - 1 = (z - \\zeta^0)(z - \\zeta^1)(z - \\zeta^2)(z - \\zeta^3)(z - \\zeta^4)", **kw),
        )
        expressions[:2].scale(1.5, about_edge=UP)
        subtexts = VGroup(
            MTex("\\text{Solutions: } \\zeta^0,\\, \\zeta^1,\\, \\zeta^2,\\, \\zeta^3,\\, \\zeta^4", **kw),
            MTex("\\text{Roots: } \\zeta^0,\\, \\zeta^1,\\, \\zeta^2,\\, \\zeta^3,\\, \\zeta^4", **kw),
        )
        subtexts.next_to(expressions, DOWN, LARGE_BUFF)

        self.play(FadeIn(expressions[0]))
        self.play(Write(subtexts[0]))
        self.wait()
        self.play(
            TransformMatchingShapes(*expressions[:2]),
            FadeTransform(*subtexts[:2]),
        )
        self.wait()
        self.play(
            expressions[1].animate.move_to(expressions[2], LEFT)
        )
        self.play(
            FadeOut(expressions[1][-1]),
            FadeTransform(expressions[1][:5], expressions[2][:5]),
            TransformMatchingShapes(subtexts[1], expressions[2][5:]),
        )
        self.wait()


class Thumbnail(InteractiveScene):
    def construct(self):
        title = TexText("Useless puzzles with\\\\useful lessons")
        title.set_width(0.6 * FRAME_WIDTH)
        title.to_corner(DR)
        title.set_color(BLUE)
        self.add(title)

        st = "$\\big\\{1, 2, 3, \\dots, 2000\\big\\}$"
        question = TexText(
            f"How many subsets of {st}\\\\"
            "have a sum which is divisible by 5?",
            tex_to_color_map={st: YELLOW}
        )
        question.set_width(FRAME_WIDTH - 1)
        question.to_edge(UP)
        self.add(question)

        randy = Randolph()
        randy.to_corner(DL)
        randy.change("thinking", question)
        self.add(randy)
