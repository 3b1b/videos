from manim_imports_ext import *
from _2022.wordle.scenes import *


class HeresTheThing(TeacherStudentsScene):
    def construct(self):
        thumbnail = ImageMobject(os.path.join(
            self.file_writer.output_directory,
            self.file_writer.get_default_module_directory(),
            os.pardir,
            "images",
            "Thumbnail",
        ))
        thumbnail = Group(
            SurroundingRectangle(thumbnail, buff=0, stroke_color=WHITE, stroke_width=2),
            thumbnail
        )
        thumbnail.set_height(3)
        thumbnail.move_to(self.hold_up_spot, DOWN)

        self.play(
            self.teacher.animate.change("raise_right_hand"),
            FadeIn(thumbnail, UP),
        )
        self.change_student_modes(
            "raise_left_hand", "pondering", "erm",
            look_at_arg=self.teacher.eyes,
        )
        self.wait()
        self.teacher_says(
            TexText("Right, here's\\\\the thing...", font_size=40),
            target_mode="hesitant",
            bubble_kwargs=dict(width=3.5, height=2.5),
            run_time=1,
            added_anims=[
                self.get_student_changes("dance_3", "frustrated", "sassy"),
                thumbnail.animate.to_corner(UL),
            ]
        )
        self.look_at(thumbnail)
        cross = Cross(Text("CRANE"))
        cross.insert_n_curves(20)
        cross.move_to(thumbnail, UR)
        cross.shift([-0.25, -0.22, 0])
        self.play(ShowCreation(cross))
        self.wait(3)


class WriteTheTitle(Scene):
    def construct(self):
        title = Text("Solving Wordle using\ninformation theory", font_size=60)
        title.get_part_by_text('information theory').set_color(BLUE)
        self.play(Write(title))
        self.wait()


class Confessions(Scene):
    def construct(self):
        titles = VGroup(*(
            Text(f"Confession {n}", font_size=60)
            for n in range(1, 4)
        ))
        titles.arrange(DOWN, buff=2, aligned_edge=LEFT)
        titles.center()
        for title in titles:
            title.save_state()
            title.generate_target()

        phrases = VGroup(
            TexText("Bug in the code", font_size=48),
            TexText("About the opening scene...", font_size=48),
            TexText("Maybe don't use entropy?", font_size=48),
        )

        self.play(LaggedStart(*(
            FadeIn(title, RIGHT)
            for title in titles
        ), lag_ratio=0.3))
        self.wait()

        last_phrase = VMobject()
        for title, phrase in zip(titles, phrases):
            others = VGroup(*(
                t for t in titles
                if t is not title
            ))
            others_target = VGroup(*(t.target for t in others))
            for t in others_target:
                t.set_height(0.2)
                t.set_opacity(0.5)
            others_target.arrange(DOWN, buff=0.75)
            others_target.to_edge(LEFT)

            phrase.set_color(BLUE)
            phrase.to_edge(UP, buff=1.5)

            self.play(
                FadeOut(last_phrase),
                title.animate.restore().set_height(0.5).center().to_edge(UP),
                *map(MoveToTarget, others),
            )
            self.play(FadeIn(phrase))
            self.wait(2)

            last_phrase = phrase


class WhatWasTheBug(TeacherStudentsScene):
    def construct(self):
        self.student_says("What as the bug?")
        self.play(
            self.get_student_changes("raise_left_hand", "pondering", "raise_right_hand"),
            self.teacher.animate.change("tired")
        )
        self.wait(3)


class HowWordleColoringWorks(WordleScene):
    def construct(self):
        # Rows
        guess_row = self.grid[0].copy()
        answer_row = self.grid[0].copy()
        rows = VGroup(guess_row, answer_row)
        rows.arrange(DOWN, buff=LARGE_BUFF)
        words = VGroup()
        for row in rows:
            row.word = VGroup()
            self.add(row.word)
            words.add(row.word)

        self.add(rows)
        self.remove(self.grid)

        # Add labels
        guess_label = TexText("Guess $\\rightarrow$")
        answer_label = TexText("Answer $\\rightarrow$")
        labels = VGroup(guess_label, answer_label)
        for label, row in zip(labels, rows):
            label.next_to(row, LEFT, buff=MED_LARGE_BUFF)

        def get_box_to_box_path(box1, box2):
            p1 = box1.get_bottom()
            p2 = box2.get_top()
            v = 0.3 * get_norm(p2 - p1) * DOWN
            return CubicBezier(p1, p1 + v, p2 - v, p2)

        def get_grey_label(i):
            result = VGroup(
                Vector(0.5 * UP),
                TexText("No 2nd `E'"),
            )
            result.arrange(UP, SMALL_BUFF)
            result.next_to(guess_row[i], UP, SMALL_BUFF)
            result.set_color(GREY_B)
            return result

        def generate_group_target(group):
            target = group.copy()
            target.scale(0.3)
            for sm in target.get_family():
                if sm.get_stroke_width() > 0:
                    sm.set_stroke(width=1.5)
            return target

        # And answer, yellow grey
        guess = "speed"
        answer = "abide"
        self.play(Write(guess_label), run_time=1)
        self.write_word_in_row(guess, guess_row)
        self.wait()
        answer_row.generate_target()
        answer_row.target.set_stroke(GREEN, 3)
        self.play(
            Write(answer_row.target, remover=True),
            Write(answer_label),
            run_time=1
        )
        answer_row.match_style(answer_row.target)
        self.write_word_in_row(answer, answer_row,)
        self.wait()

        colors = self.get_colors(get_pattern(guess, answer))
        self.animate_color_change(guess_row, guess_row.word, colors)
        self.wait()

        # Explain two e's
        path = get_box_to_box_path(guess_row[2], answer_row[4])
        gl = get_grey_label(3)

        self.play(ShowCreation(path))
        self.play(Write(gl))
        self.wait()
        group1 = VGroup(
            guess_row.copy(),
            answer_row.copy(),
            guess_row.word.copy(),
            answer_row.word.copy(),
            path, gl,
        )

        answer_row.word.set_submobjects([])

        vect = 1.5 * DOWN
        self.play(
            Transform(group1, generate_group_target(group1).to_corner(UL)),
            labels.animate.shift(vect),
            rows.animate.set_fill(BLACK, 0).shift(vect),
            guess_row.word.animate.shift(vect),
        )

        # General transition
        def change_answer(answer):
            self.write_word_in_row(answer, answer_row)
            colors = self.get_colors(get_pattern(guess, answer))
            self.animate_color_change(guess_row, guess_row.word, colors)
            self.wait()

        # Two yellows
        change_answer("erase")
        path1 = get_box_to_box_path(guess_row[2], answer_row[0])
        path2 = get_box_to_box_path(guess_row[3], answer_row[4])

        self.play(ShowCreation(path1))
        self.play(ShowCreation(path2))
        self.wait()

        group2 = VGroup(
            *rows.copy(),
            *words.copy(),
            path1, path2,
        )
        answer_row.word.set_submobjects([])
        group2.target = generate_group_target(group2)
        group2.target.next_to(group1, RIGHT, buff=LARGE_BUFF, aligned_edge=DOWN)
        self.play(
            guess_row.animate.set_fill(BLACK, 0),
            MoveToTarget(group2),
        )

        # Green grey
        change_answer("steal")
        path = get_box_to_box_path(guess_row[2], answer_row[2])
        gl = get_grey_label(3)
        self.play(ShowCreation(path))
        self.play(Write(gl))
        self.wait()

        group3 = VGroup(
            *rows.copy(),
            *words.copy(),
            path, gl,
        )
        answer_row.word.set_submobjects([])

        group3.target = generate_group_target(group3)
        group3.target.next_to(group2, RIGHT, buff=LARGE_BUFF, aligned_edge=DOWN)
        self.play(
            guess_row.animate.set_fill(BLACK, 0),
            MoveToTarget(group3)
        )

        # Green yellow
        change_answer("crepe")
        path1 = get_box_to_box_path(guess_row[2], answer_row[2])
        path2 = get_box_to_box_path(guess_row[3], answer_row[4])

        self.play(ShowCreation(path1))
        self.play(ShowCreation(path2))
        self.wait()

        group4 = VGroup(*rows, *words, path1, path2)

        # Organize all groups
        groups = VGroup(group1, group2, group3, group4)
        groups.generate_target()
        groups.target[-1].become(generate_group_target(group4))
        groups.target.arrange(RIGHT, buff=LARGE_BUFF, aligned_edge=DOWN)
        groups.target.set_width(FRAME_WIDTH - 1)
        groups.target.to_edge(UP)

        self.play(
            MoveToTarget(groups, run_time=2),
            FadeOut(labels)
        )

        # Compare to bug
        low_groups = groups.deepcopy()
        low_groups.to_edge(DOWN, buff=LARGE_BUFF)

        paths = VGroup()
        for lg in low_groups:
            # Remove grey labels
            lg.remove(lg[-1])
            # Make fourth square grey
            lg[0][3].set_fill(WordleScene.color_map[1])
            new_path = get_box_to_box_path(lg[0][3], VectorizedPoint(lg[-1].get_end()))
            new_path.set_stroke(width=1.5)
            lg.add(new_path)
            paths.add(*lg[-2:])
            paths.set_opacity(0)

        bug_box = SurroundingRectangle(low_groups, buff=0.25)
        bug_box.set_stroke(RED, 3)
        bug_label = Text("Buggy behavior")
        bug_label.next_to(bug_box, UP)
        bug_label.set_color(RED)

        self.play(
            FadeTransform(groups.copy(), low_groups),
            FadeIn(bug_box),
            Write(bug_label),
        )
        paths.set_stroke(opacity=1)
        self.play(LaggedStartMap(ShowCreation, paths, lag_ratio=0.5, run_time=3))
        self.wait()

    def animate_color_change(self, row, word, colors):
        super().animate_color_change(row, word, colors)
        self.add(word)

    def write_word_in_row(self, word, row, added_anims=[]):
        word_mob = VGroup(*(
            self.get_letter_in_square(letter, square)
            for letter, square in zip(word, row)
        ))
        self.play(
            ShowIncreasingSubsets(word_mob, run_time=0.5),
            *added_anims,
        )
        self.remove(word_mob)
        row.word.set_submobjects(list(word_mob))

    def delete_word_from_row(self, row, added_anims=[]):
        self.play(FadeOut(row.word, lag_ratio=0.2), *added_anims)
        row.word.set_submobjects([])
        self.add(row.word)


class MatrixOfPatterns(Scene):
    n_shown_words = 50

    def construct(self):
        words = get_word_list(short=True)
        words = random.sample(words, self.n_shown_words)
        words.sort()
        word_mobs = VGroup(*(Text(word, font="Consolas", font_size=30) for word in words))
        top_row = word_mobs.copy().arrange(RIGHT, buff=0.35)
        left_col = word_mobs.copy().arrange(DOWN, buff=0.35)
        top_row.to_corner(UL).shift(1.25 * RIGHT)
        left_col.to_corner(UL).shift(0.5 * DOWN)

        h_line = Line(LEFT, RIGHT).match_width(top_row).scale(1.2)
        v_line = Line(UP, DOWN).match_height(left_col).scale(1.2)
        h_line.next_to(top_row, DOWN, SMALL_BUFF)
        v_line.next_to(left_col, RIGHT)
        lines = VGroup(h_line, v_line)
        lines.set_stroke(GREY, 1)

        pattern_mobs = VGroup()
        for n in range(2 * self.n_shown_words):
            for k in range(n + 1):
                if n - k > len(words) - 1 or k > len(words) - 1:
                    continue
                w1 = left_col[n - k]
                w2 = top_row[k]
                pattern = get_pattern(w1.text, w2.text)
                pm = WordleScene.patterns_to_squares([pattern])[0]
                pm.set_stroke(WHITE, 0.5)
                pm.match_width(w1)
                pm.match_y(w1)
                pm.match_x(w2, UP)
                pattern_mobs.add(pm)

        frame = self.camera.frame

        self.add(top_row, left_col)
        self.add(h_line, v_line)

        self.play(
            ApplyMethod(
                frame.set_height, 25, dict(about_edge=UL),
                rate_func=squish_rate_func(smooth, 0.1, 1),
            ),
            ShowIncreasingSubsets(pattern_mobs),
            run_time=20,
        )
        self.wait()


class SaletBruteForceDistribution(ShowScoreDistribution):
    data_file = "salet_brute_force.json"


class CraneBruteForceDistribution(ShowScoreDistribution):
    data_file = "crane_brute_force.json"


class CrateBruteForceDistribution(ShowScoreDistribution):
    data_file = "crate_brute_force.json"


class TraceBruteForceDistribution(ShowScoreDistribution):
    data_file = "trace_brute_force.json"


class LeastDistribution(ShowScoreDistribution):
    data_file = "least_results.json"


class SlateDistribution(ShowScoreDistribution):
    data_file = "slate_results.json"


class WearyDistribution(ShowScoreDistribution):
    data_file = "weary_results.json"


class AdieuDistribution(ShowScoreDistribution):
    data_file = "adieu_results.json"


class AudioDistribution(ShowScoreDistribution):
    data_file = "audio_results.json"


class LastVideoWrapper(VideoWrapper):
    title = "Last video"


class SneakAttackInformationTheory(Scene):
    def construct(self):
        randy = Randolph()
        randy.to_edge(DOWN)
        phone = ImageMobject("wordle-phone")
        phone.set_height(4)
        phone.next_to(randy, RIGHT)
        phone.shift(UP)
        phone_outline = SVGMobject("wordle-phone")
        phone_outline.set_fill(opacity=0)
        phone_outline.set_stroke(width=0)
        phone_outline.replace(phone).scale(0.8)

        self.play(
            randy.animate.change("thinking", phone),
            FadeIn(phone),
            Write(phone_outline, stroke_width=1, run_time=2),
        )
        self.play(Blink(randy))
        self.wait()

        # surprise
        words = VGroup(
            Text("Surprise!"),
            Text("Information theory!"),
        )
        words.scale(1.25)
        words.arrange(DOWN)
        words.to_corner(UL)
        ent_formula = Tex("H = \\sum_{x} -p(x)\\log_2\\big(p(x)\\right)")
        ent_formula.set_color(TEAL)
        ent_formula.next_to(words, DOWN, LARGE_BUFF)

        self.play(
            FadeIn(words[0], scale=2),
            randy.animate.change("horrified", words),
            run_time=0.5
        )
        self.play(
            FadeIn(words[1], scale=2),
            run_time=0.5,
        )
        self.play(Blink(randy))
        self.play(
            Write(ent_formula),
            randy.animate.change("erm", ent_formula),
        )
        self.play(Blink(randy))
        self.wait(2)
        self.play(Blink(randy))
        self.wait(2)


class HowAreYouFindingTheBest(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            TexText("What exactly is this\\\\``final analysis''?"),
            student_index=0,
        )
        self.play(
            self.students[1].animate.change("pondering"),
            self.students[2].animate.change("raise_left_hand"),
            self.teacher.animate.change("happy")
        )
        self.wait(2)

        self.teacher_says(TexText("I'm glad\\\\you asked!"), target_mode="hooray")
        self.wait(3)


class SaletDistThin(SaletBruteForceDistribution):
    axes_config = dict(
        x_range=(0, 6),
        y_range=(0, 1, 0.1),
        width=4.5,
        height=6,
    )
    bar_count_font_size = 24


class CrateDistThin(SaletDistThin):
    data_file = "crate_brute_force.json"


class TraceDistThin(SaletDistThin):
    data_file = "trace_brute_force.json"


class FromEstimatedProbabilityOfInclusionToExact(DistributionOverWord):
    def construct(self):
        super().construct()
        bars = self.bars
        decs = self.decs
        word_mobs = self.word_mobs

        true_answers = get_word_list(short=True)
        anims = []
        height = 1.5
        for bar, dec, word_mob in zip(bars, decs, word_mobs):
            value = float(word_mob.text in true_answers)
            anims.extend([
                bar.animate.set_height(value * height, stretch=True, about_edge=DOWN),
                ChangeDecimalToValue(dec, 100 * value),
            ])

        self.play(*anims, run_time=2)
        self.wait()

        self.embed()


class SearchThroughFirstGuessDistributions(HowLookTwoAheadWorks):
    def construct(self):
        # Setup
        all_words = get_word_list()
        possibilities = get_word_list()
        priors = self.get_priors()

        template = self.get_word_mob("soare").to_edge(LEFT)

        # Show first guess
        sample = random.sample(all_words, 300)
        for word in sample:
            guess = self.get_word_mob(word)
            guess.move_to(template, RIGHT)
            pattern_array = self.get_pattern_array(guess, possibilities, priors)
            prob_bars = self.get_prob_bars(pattern_array.pattern_mobs)
            EI_label = self.get_entropy_label(guess, pattern_array.distribution)

            self.add(guess)
            self.add(pattern_array)
            self.add(prob_bars)
            self.add(EI_label)
            self.wait(0.05)
            self.clear()


class TwoStepLookAheadWithSlane(HowLookTwoAheadWorks):
    first_guess = "slane"
    n_shown_trials = 120
    transition_time = 0.01

    def get_priors(self):
        return get_true_wordle_prior()


class TwoStepLookAheadWithSoare(TwoStepLookAheadWithSlane):
    first_guess = "soare"


class TwoStepLookAheadWithCrane(TwoStepLookAheadWithSlane):
    first_guess = "crane"


class TwoStepLookAheadWithSalet(TwoStepLookAheadWithSlane):
    first_guess = "salet"


class SaletSecondGuessMap(HowLookTwoAheadWorks):
    first_guess = "salet"

    def construct(self):
        path = os.path.join(get_directories()['data'], 'wordle', 'salet_with_brute_force_guess_map.json')
        with open(path) as fp:
            self.guess_map = json.load(fp)

        self.all_words = get_word_list()
        possibilities = get_word_list(short=True)
        priors = self.priors = self.get_priors()

        guess1 = self.get_word_mob(self.first_guess)
        guess1.to_edge(LEFT)
        pattern_array1 = self.get_pattern_array(guess1, possibilities, priors)

        arrows, guess2s = self.get_next_guess_group(guess1.text, pattern_array1, possibilities)

        self.add(guess1)
        self.add(pattern_array1)
        self.play(
            LaggedStartMap(ShowCreation, arrows, lag_ratio=0.5),
            LaggedStartMap(FadeIn, guess2s, lambda m: (m, RIGHT), lag_ratio=0.5),
            run_time=2,
        )
        self.wait()

        for pm, guess2 in zip(pattern_array1.pattern_mobs, guess2s):
            bucket = get_possible_words(guess1.text, pm.pattern, possibilities)
            pattern_array2 = self.get_pattern_array(guess2, bucket, priors)
            ng_group = self.get_next_guess_group(
                guess2.text, pattern_array2, bucket,
                prefix=guess1.text + "".join(map(str, pattern_to_int_list(pm.pattern))),
            )

            self.add(pattern_array2, ng_group)
            self.wait(0.5)
            self.remove(pattern_array2, ng_group)

    def get_next_guess_group(self, guess, pattern_array, possibilities, prefix=""):
        arrows = VGroup()
        guesses = VGroup()
        for pm in pattern_array.pattern_mobs:
            ps = "".join(map(str, pattern_to_int_list(pm.pattern)))
            key = prefix + guess + ps
            if key in self.guess_map:
                next_guess = self.guess_map[key]
            else:
                next_guess = optimal_guess(
                    self.all_words,
                    get_possible_words(guess, pm.pattern, possibilities),
                    self.priors,
                    optimize_for_uniform_distribution=True
                )
            next_guess_mob = self.get_word_mob(next_guess)
            next_guess_mob.scale(0.8)
            arrow = Vector(0.5 * RIGHT, stroke_width=3)
            arrow.next_to(pm, RIGHT, buff=SMALL_BUFF)
            next_guess_mob.next_to(arrow, RIGHT, buff=SMALL_BUFF)

            arrows.add(arrow)
            guesses.add(next_guess_mob)

        return VGroup(arrows, guesses)


class RankingUpdates(Scene):
    def construct(self):
        # Titles
        kw = dict(tex_to_color_map={"$E[I]$": TEAL}, font_size=30)
        titles = VGroup(
            TexText("Highest $E[I]$\\\\(one step)", **kw),
            TexText("Highest $E[I]$\\\\(two steps)", **kw),
            TexText("Lowest average\\\\scores", **kw),
        )
        for x, title in zip(range(-1, 2), titles):
            title.set_x(x * FRAME_WIDTH / 3)
            title.to_edge(UP, buff=MED_SMALL_BUFF)
        titles.to_edge(UP)

        h_line = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        h_line.set_stroke(GREY_B, 1)
        h_line.next_to(titles, DOWN)
        h_line.set_x(0)

        v_lines = VGroup(*(
            Line(UP, DOWN).set_height(FRAME_HEIGHT).set_x(x * FRAME_WIDTH / 6)
            for x in (-1, 1)
        ))
        v_lines.match_style(h_line)

        self.add(h_line)
        self.add(v_lines)

        # Lists
        n_shown = 15

        def get_column(rows, title, row_height=0.2, buff=0.2):
            for row in rows:
                row.move_to(ORIGIN, LEFT)
            rows.set_height(row_height)
            for i, row in enumerate(rows):
                row.shift(i * (row_height + buff) * DOWN)
            rows.next_to(h_line, DOWN)

            numbers = VGroup()
            for i, row in zip(it.count(1), rows):
                num = Integer(i, unit=".")
                num.match_height(row)
                num.next_to(row, LEFT)
                numbers.add(num)

            result = VGroup(numbers, rows)
            result.match_x(title)
            result.rows = rows
            result.numbers = numbers
            return result

        # One step
        with open(os.path.join(get_directories()["data"], "wordle", "best_entropies.json")) as fp:
            rows1 = VGroup()
            for word, ent in json.load(fp)[:n_shown]:
                row = VGroup(
                    Text(word, font="Consolas"),
                    Tex("\\rightarrow"),
                    DecimalNumber(ent, color=TEAL),
                )
                row.arrange(RIGHT)
                rows1.add(row)

        # Two step
        with open(os.path.join(get_directories()["data"], "wordle", "best_double_entropies.json")) as fp:
            rows2 = VGroup()
            for word, ent1, ent2 in json.load(fp)[:n_shown]:
                row = VGroup(
                    Text(word, font="Consolas"),
                    Tex("\\rightarrow"),
                    DecimalNumber(ent1, color=TEAL),
                    Tex("+"),
                    DecimalNumber(ent2, color=TEAL),
                    Tex("="),
                    DecimalNumber(ent1 + ent2, color=TEAL),
                )
                row.arrange(RIGHT)
                rows2.add(row)

        # Score
        answers = get_word_list(short=True)
        with open(os.path.join(get_directories()["data"], "wordle", "best_scores_with_entropy_method.json")) as fp:
            rows3 = VGroup()
            for word, score, dist in json.load(fp)[:n_shown]:
                row = VGroup(
                    Text(word, font="Consolas"),
                    Tex("\\rightarrow"),
                    DecimalNumber(score / len(answers), num_decimal_places=3, color=BLUE),
                )
                row.arrange(RIGHT)
                rows3.add(row)

        all_rows = [rows1, rows2, rows3]
        col1, col2, col3 = cols = [
            get_column(rows, title)
            for rows, title in zip(all_rows, titles)
        ]
        for row in col2.rows:
            row[-2:].align_to(col2.rows[0][-2:], LEFT)

        # Animations
        self.add(titles[0])
        self.play(FadeIn(col1, lag_ratio=0.1, run_time=2))
        self.wait()

        last_col = col1
        for title, col in zip(titles[1:], cols[1:]):
            pre_words = VGroup(*(r[0] for r in last_col.rows))
            words = VGroup(*(r[0] for r in col.rows))
            mover_anims = []
            fade_anims = []
            for word in words:
                has_pre = False
                for pre_word in pre_words:
                    if word.text == pre_word.text:
                        has_pre = True
                        mover_anims.append(TransformFromCopy(pre_word, word))
                if not has_pre:
                    fade_anims.append(FadeInFromPoint(word, last_col.get_bottom()))

            for row in col.rows:
                fade_anims.append(FadeIn(row[1:]))

            self.play(
                FadeIn(title),
                FadeIn(col.numbers, lag_ratio=0.1),
                LaggedStart(*mover_anims, run_time=2, lag_ratio=0.2),
            )
            self.play(LaggedStart(*fade_anims))
            self.add(col)
            self.wait()
            last_col = col


class WeCanDoBetter(Scene):
    def construct(self):
        morty = Mortimer()
        morty.to_corner(DR)
        self.play(PiCreatureSays(
            morty, TexText("We can do\\\\better!"),
            bubble_kwargs=dict(width=3.5, height=2.5, fill_opacity=0.95),
        ))
        for x in range(2):
            self.play(Blink(morty))
            self.wait(2)


class WhydYouHaveToRuinIt(TeacherStudentsScene):
    def construct(self):
        self.student_says(
            TexText("Why'd you have to\\\\ruin Wordle!"),
            target_mode="pleading",
            added_anims=[self.teacher.animate.change("guilty")]
        )
        self.change_student_modes("sassy", "angry", "pleading")
        self.wait(3)
        self.teacher_says(
            TexText("But ``salet'' is probably\\\\not the best for us"),
            added_anims=[self.get_student_changes("confused", "sassy", "hesitant")]
        )
        self.wait(3)


class ForgetTheBestWord(Scene):
    def construct(self):
        self.add(FullScreenRectangle())

        randy = Randolph()
        randy.to_corner(DL)
        text = TexText("Wait, was it ``slane'',\\\\``salet'' or ``soare''?")

        self.play(PiCreatureBubbleIntroduction(
            randy, text,
            bubble_class=ThoughtBubble,
            target_mode="confused",
        ))
        self.play(Blink(randy))
        self.wait()

        randy.bubble.add(text)
        self.play(
            FadeOut(randy.bubble, scale=0.25, shift=2 * DL),
            randy.animate.change("erm")
        )
        self.wait()
        self.play(randy.animate.change("thinking", UR))
        for x in range(2):
            self.play(Blink(randy))
            self.wait(2)


class Thumbnail2(Thumbnail):
    def construct(self):
        super().construct()
        title = self.title
        self.rows.to_edge(DOWN, buff=MED_SMALL_BUFF)

        crane = title.get_part_by_text("CRANE")
        strike = Line(
            crane.get_corner(DL),
            crane.get_corner(UR),
        )
        strike.set_stroke(RED, 15)
        strike.scale(1.2)
        strike.insert_n_curves(100)
        strike.set_stroke(width=(2, 15, 15, 2))
        self.add(strike)

        oops = Text("Here's the thing...", font="Consolas", font_size=90)
        oops.set_color(RED)
        oops.next_to(title, DR, buff=MED_LARGE_BUFF)
        oops.shift_onto_screen()
        self.add(oops)


class EndScreen2(EndScreen):
    pass
