from manim_imports_ext import *


def get_set_tex(values, max_shown=7, **kwargs):
    value_strs = list(map(str, values))
    if len(value_strs) > max_shown:
        value_strs = [
            *value_strs[:max_shown - 2],
            "\\dots",
            value_strs[-1]
        ]

    result = MTex(
        "\\{" + ",".join(value_strs) + "\\}",
        isolate=("\\{", "\\}", ",", *value_strs),
        **kwargs,
    )
    result.values = values
    return result


def get_brackets(set_tex):
    return VGroup(set_tex[0], set_tex[-1])


def get_integer_parts(set_tex):
    result = VGroup(*(
        set_tex.get_part_by_tex(str(value))
        for value in set_tex.values
    ))
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
            set_tex1.get_part_by_tex(str(value)),
            set_tex2.get_part_by_tex(str(value)),
        )
        for value in filter(
            lambda v: v in set_tex2.values,
            set_tex1.values,
        )
    ]
    mismatch_animations = [
        FadeInFromPoint(
            set_tex2.get_part_by_tex(str(value)),
            set_tex1.get_center()
        )
        for value in set(set_tex2.values).difference(set_tex1.values)
    ]
    n = min(len(set_tex1.values), len(set_tex2.values))
    commas = []
    for st in set_tex1, set_tex2:
        try:
            commas.append(st.get_parts_by_tex(",")[:n - 1])
        except ValueError:
            commas.append(VGroup().move_to(st))
    comma_animations = TransformFromCopy(*commas)
    return AnimationGroup(
        bracket_anim, *matching_anims, *mismatch_animations, comma_animations
    )


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

class ExampleWith5(Scene):
    def construct(self):
        N = 5
        full_set = list(range(1, N + 1))

        set_tex = get_set_tex(full_set)
        set_tex.to_edge(UP)
        stacks = self.get_subset_stacks(full_set)
        sum_stacks = self.get_subset_sums(stacks)

        self.show_all_subsets(set_tex, stacks, sum_stacks)
        self.highlight_multiple_of_5(stacks, sum_stacks)
        self.group_by_sum(stacks, sum_stacks)
        self.show_generating_function(set_tex)

        self.embed()

    def show_all_subsets(self, set_tex, stacks, sum_stacks):
        self.add(set_tex)

        # Null set
        null_set = stacks[0][0]
        null_set.save_state()
        null_set.match_height(set_tex)
        null_set.center()
        null_sum = get_sum_group(null_set)

        self.play(set_tex_transform(set_tex, null_set))
        self.wait()
        self.play(get_sum_animation(null_set, null_sum))
        self.wait()

        # Singletons
        self.play(
            Restore(null_set),
            ReplacementTransform(null_sum, sum_stacks[0][0]),
        )
        self.wait()
        self.play(
            LaggedStart(*(
                set_tex_transform(set_tex, singleton)
                for singleton in stacks[1]
            ), lag_ratio=0.01),
        )
        self.wait()
        self.play(
            LaggedStart(*(
                get_sum_animation(st, sg)
                for st, sg in zip(stacks[1], sum_stacks[1])
            ), lag_ratio=0.01)
        )
        self.wait()

        # Pairs, triplets
        for k in range(2, len(set_tex.values)):
            stack = stacks[k]
            sum_groups = sum_stacks[k]
            highlights = VGroup(*(
                get_subset_highlights(set_tex, subset)
                for subset in it.combinations(set_tex.values, k)
            ))
            colored_stack = stack.copy().set_color(YELLOW)

            self.play(
                ShowIncreasingSubsets(stack, int_func=np.ceil),
                ShowSubmobjectsOneByOne(highlights),
                ShowSubmobjectsOneByOne(colored_stack),
                run_time=len(stack) / 2,
                rate_func=linear
            )
            self.wait(1)
            self.remove(highlights, colored_stack)

            self.play(
                LaggedStart(*(
                    get_sum_animation(st, sg)
                    for st, sg, in zip(stack, sum_groups)
                ), lag_ratio=0.2),
            )
            self.wait()

        # Full set
        self.play(FadeTransform(set_tex.copy(), stacks[-1]))
        self.wait()
        self.play(get_sum_animation(
            stacks[-1][0],
            sum_stacks[-1][0],
        ))
        self.wait()

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
                    anims.append(set_tex.animate.set_opacity(0.5))
                    anims.append(sum_group.animate.set_opacity(0.5))
        rects.set_color(TEAL)

        counter = Integer(0, font_size=72)
        counter.to_corner(UR)
        counter.set_color(TEAL)

        self.play(*anims, run_time=1)

        self.add(counter)
        for rect in rects:
            self.add(rect)
            counter.increment_value()
            self.wait(0.5)
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

        csst.arrange(RIGHT, buff=MED_LARGE_BUFF, aligned_edge=DOWN)
        csst[9:].next_to(csst[:9], DOWN, buff=1.5, aligned_edge=LEFT)
        csst.refresh_bounding_box()
        csst.set_width(FRAME_WIDTH - 1)
        csst.move_to(2 * DOWN, DOWN).to_edge(LEFT)

        # Create new rectangles
        common_sum_rects = VGroup()
        for stack in common_sum_stacks.target:
            rect = SurroundingRectangle(stack, buff=0.5 * SMALL_BUFF)
            rect.value = sum(stack[0][0].values)
            color = TEAL if rect.value % 5 == 0 else GREY_B
            rect.set_stroke(color, 1)
            common_sum_rects.add(rect)

        rect_anims = []
        for highlight_rect in self.highlight_rects:
            for rect in common_sum_rects:
                if rect.value == highlight_rect.value:
                    rect_anims.append(Transform(highlight_rect, rect))

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
        pass

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


class ExampleWith10(Scene):
    def construct(self):
        pass
