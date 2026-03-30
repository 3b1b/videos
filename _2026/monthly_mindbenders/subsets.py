from manim_imports_ext import *


class Subsets(InteractiveScene):
    def construct(self):
        # Add pi creatures
        randy, morty = pis = VGroup(Randolph(), Mortimer())
        pis.arrange(RIGHT)

        self.play(LaggedStart(
            VFadeIn(randy),
            randy.change("tease", morty.eyes),
            VFadeIn(morty),
            morty.change("hesitant", randy.eyes),
            lag_ratio=0.5,
            run_time=1.5
        ))
        self.play(Blink(randy))

        # Add numbers
        number_grid = VGroup(Integer(n) for n in range(1, 101))
        number_grid.arrange_in_grid(10, 10, v_buff=0.3, h_buff=0.1)
        number_grid.set_width(5)
        number_grid.to_edge(UP)

        self.play(
            FadeIn(number_grid, lag_ratio=0.01, shift=0.05 * UP, run_time=2),
            randy.change("pondering", number_grid).set_height(1).next_to(number_grid, DOWN, 1.25, LEFT),
            morty.change("raise_right_hand").set_height(1.5).next_to(number_grid, DOWN, 0.75, RIGHT),
        )
        self.wait()

        # Choose a random subset
        sample_list = random.sample(list(number_grid), 10)
        sample_list.sort(key=lambda m: m.get_value())
        sample = VGroup(*sample_list)
        number_grid.remove(*sample)
        sample_rects = VGroup(
            SurroundingRectangle(num, buff=0.1)
            for num in sample
        )
        sample_rects.set_stroke(TEAL, 2)

        self.play(
            number_grid.animate.set_fill(opacity=0.25).set_anim_args(lag_ratio=0.005),
            sample.animate.set_fill(TEAL).set_anim_args(lag_ratio=0.2),
            Write(sample_rects),
            run_time=3
        )
        self.play(Blink(morty))
        self.wait()

        # Organize
        sample_groups = VGroup(
            VGroup(num, rect)
            for num, rect in zip(sample, sample_rects)
        )
        sample_groups.target = sample_groups.generate_target()
        sample_groups.target.arrange_in_grid(2, 5)
        sample_groups.target.set_width(number_grid.get_width())
        sample_groups.target.next_to(pis, UP, MED_LARGE_BUFF)

        self.play(
            MoveToTarget(sample_groups),
            FadeOut(number_grid),
            randy.change("hesitant", sample_groups),
            morty.change("tease", randy.eyes),
        )
        self.play(
            Blink(randy),
            FadeOut(sample_rects),
        )
        self.wait()
        self.play(
            randy.animate.set_height(1.25, about_edge=DL),
            morty.animate.set_height(0.75, about_edge=DR),
        )
        self.wait()

        # Find a collision
        groups = self.find_a_match(sample).copy()
        group_rects = self.get_group_rects(groups)

        self.play(
            randy.change("raise_right_hand"),
            sample.animate.set_fill(WHITE, 0.5),
            groups[0].animate.set_fill(group_rects[0].get_color(), 1),
            Write(group_rects[0], run_time=1.5, lag_ratio=0.2),
        )
        self.play(
            groups[1].animate.set_fill(group_rects[1].get_color(), 1),
            Write(group_rects[1], run_time=1.5, lag_ratio=0.2),
        )
        self.wait()

        # Show the same sum
        top_groups, plusses, equals = equation = self.get_top_sum(groups)

        self.play(LaggedStart(
            TransformFromCopy(groups[0], top_groups[0]),
            Write(plusses[0]),
            FadeIn(equals, UP),
            TransformFromCopy(groups[1], top_groups[1]),
            Write(plusses[1]),
            randy.change("happy"),
            morty.change("pondering")
        ))
        self.play(Blink(randy))
        self.wait()

        self.play(
            FadeOut(groups),
            FadeOut(group_rects),
            FadeOut(equation),
        )

        # Show some unequal subsets
        for n in range(40):
            subsets = VGroup(
                VGroup(*random.sample(list(sample), random.randint(2, 5)))
                for n in range(2)
            )
            intersection = [
                mob for mob in sample
                if mob in subsets[0] and mob in subsets[1]
            ]
            for subset in subsets:
                subset.remove(*intersection)
            if len(subsets[0]) == 0 or len(subsets[0]) == 0:
                break

            subsets = subsets.copy()
            rects = self.get_group_rects(subsets, colors=[RED, PINK])
            for subset, rect in zip(subsets, rects):
                subset.set_fill(rect.get_color(), 1)

            equation = self.get_top_sum(subsets)

            self.add(subsets, rects, equation)
            randy.change_mode("hesitant")
            self.wait(0.5)
            self.remove(subsets, rects, equation)

    def find_a_match(self, group):
        value_to_mob = {mob.get_value(): mob for mob in group}
        values = list(value_to_mob.keys())
        sums_to_subsets = dict()
        match = ()
        subsets = it.chain(*(
            it.combinations(values, n)
            for n in range(len(values))
        ))
        for subset in subsets:
            sub_sum = sum(subset)
            if sub_sum in sums_to_subsets:
                match = (sums_to_subsets[sub_sum], subset)
                break
            sums_to_subsets[sub_sum] = subset

        clean_match = [
            set(match[0]).difference(match[1]),
            set(match[1]).difference(match[0]),
        ]

        return VGroup(
            VGroup(
                value_to_mob[value]
                for value in match
            )
            for match in clean_match
        )

    def get_group_rects(self, groups, colors=[BLUE, GREEN]):
        group_rects = VGroup(
            VGroup(SurroundingRectangle(mob) for mob in group)
            for group in groups
        )
        for rect, color in zip(group_rects, colors):
            rect.set_stroke(color, 2)
        return group_rects

    def get_top_sum(self, groups):
        top_groups = groups.copy()
        sums = []
        for group in top_groups:
            group.arrange(RIGHT, buff=0.5)
            sums.append(sum(m.get_value() for m in group))
        top_groups.arrange(DOWN, buff=1)
        top_groups.to_edge(UP)

        equals = Tex(R"=")
        equals.rotate(90 * DEG)
        equals.move_to(top_groups)
        if sums[0] != sums[1]:
            slash = Line(equals.get_bottom(), equals.get_top())
            slash.set_stroke(RED, 5)
            slash.rotate(45 * DEG)
            equals.add(slash)

        plusses = VGroup()
        for group in top_groups:
            plus_line = VGroup()
            for m1, m2 in zip(group, group[1:]):
                plus = Tex(R"+").move_to(midpoint(m1.get_right(), m2.get_left()))
                plus_line.add(plus)
            plusses.add(plus_line)

        return VGroup(top_groups, plusses, equals)


class Subsets2(Subsets):
    random_seed = 1


class Subsets3(Subsets):
    random_seed = 2
