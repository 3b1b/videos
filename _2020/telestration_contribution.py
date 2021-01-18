from manim_imports_ext import *


class TelestrationContribution(Scene):
    def construct(self):
        # Object creators
        def get_beer():
            beer = SVGMobject(file_name="beer")
            beer.set_stroke(width=0)
            beer[0].set_fill(GREY_C)
            beer[1].set_fill(WHITE)
            beer[2].set_fill("#ff9900")
            return beer

        def get_muscle():
            muscle = SVGMobject("muscle")
            muscle.set_fill(GREY_BROWN)
            muscle.set_stroke(WHITE, 2)
            return muscle

        def get_cat():
            cat = SVGMobject("sitting_cat")
            cat.set_fill(GREY_C)
            cat.set_stroke(WHITE, 0)
            return cat

        def get_fat_cat():
            cat = SVGMobject("fat_cat")
            cat.flip()
            cat.set_stroke(WHITE, 0)
            cat.set_fill(GREY_C, 1)

            return cat

        def get_person():
            person = SVGMobject("person")
            person.set_fill(GREY_C, 1)
            person.set_stroke(WHITE, 1)
            return person

        # Beer makes you stronger
        beer = get_beer()
        arrow = Tex("\\Rightarrow")
        arrow.set_width(1)
        muscle = get_muscle()
        imply_group = VGroup(beer, arrow, muscle)
        imply_group.arrange(RIGHT, buff=0.5)

        news = Rectangle(height=7, width=6)
        news.set_fill(GREY_E, 1)
        imply_group.set_width(news.get_width() - 1)
        imply_group.next_to(news.get_top(), DOWN)
        lines = VGroup(*[Line(LEFT, RIGHT) for x in range(12)])
        lines.arrange(DOWN, buff=0.3)
        lines.set_width(news.get_width() - 1, stretch=True)
        lines.next_to(imply_group, DOWN, MED_LARGE_BUFF)
        lines[-1].stretch(0.5, 0, about_edge=LEFT)
        news.add(lines)

        q_marks = Tex("???")[0]
        q_marks.space_out_submobjects(1.5)
        q_marks.replace(imply_group, dim_to_match=1)

        self.add(news)
        self.play(Write(q_marks))
        self.wait()
        beer.save_state()
        beer.move_to(imply_group)
        self.play(
            FadeOut(q_marks, lag_ratio=0.1),
            FadeInFromDown(beer)
        )
        self.play(
            Restore(beer),
            FadeIn(arrow, 0.2 * LEFT),
            DrawBorderThenFill(muscle)
        )
        news.add(imply_group)
        self.wait(2)

        # Doubt
        randy = Randolph()
        randy.to_corner(DL)
        randy.change("confused")
        bangs = Tex("!?!")
        bangs.scale(2)
        bangs.next_to(randy, UP)

        self.play(
            FadeIn(randy),
            news.scale, 0.8, {"about_edge": UP},
            news.shift, RIGHT,
        )
        self.play(Blink(randy))
        self.play()
        self.play(
            randy.change, "angry", imply_group,
            Write(bangs, run_time=1)
        )
        self.wait()
        self.play(Blink(randy))
        self.wait()

        # Axes
        axes = Axes(
            x_min=0,
            x_max=15,
            y_min=0,
            y_max=10,
        )
        axes.center()
        axes.set_height(FRAME_HEIGHT - 1)

        news.remove(imply_group)
        news.remove(lines)
        beer.generate_target()
        beer.target.set_height(0.75)
        beer.target.next_to(axes.x_axis.get_end(), UR, SMALL_BUFF)
        self.play(
            FadeOut(news),
            LaggedStartMap(
                FadeOutAndShift, VGroup(randy, bangs, arrow, muscle),
                lambda m: (m, DOWN)
            ),
            Uncreate(lines),
            MoveToTarget(beer, run_time=2),
            ShowCreation(axes),
        )

        # Cat labels
        lil_cat = get_cat()
        lil_cat.set_height(0.5)
        lil_cat.next_to(axes.c2p(0, 0), LEFT, aligned_edge=DOWN)

        fat_cat = get_fat_cat()
        fat_cat.set_height(1.5)
        fat_cat.next_to(axes.c2p(0, 10), LEFT, aligned_edge=UP)

        self.play(FadeIn(lil_cat))
        self.play(TransformFromCopy(lil_cat, fat_cat))

        # Data
        data = VGroup()
        n_data_points = 50
        for x in np.linspace(1, 15, n_data_points):
            x += np.random.random() - 0.5
            y = (x * 10 / 15) + (np.random.random() - 0.5) * 5
            if y < 0.5:
                y = 0.5
            if y > 15:
                y -= 1
            dot = Dot(axes.c2p(x, y))
            dot.set_height(0.1)
            data.add(dot)

        data.set_color(BLUE)

        line = Line(axes.c2p(0, 0.5), axes.c2p(15, 10))

        self.play(ShowIncreasingSubsets(data, run_time=4))
        self.play(ShowCreation(line))
        self.wait()

        graph = VGroup(axes, lil_cat, fat_cat, beer, data, line)
        graph.save_state()

        # Write article
        article = Rectangle(height=4, width=3)
        article.set_fill(GREY_E, 1)
        article.to_edge(RIGHT)

        arrow = Vector(RIGHT)
        arrow.set_color(YELLOW)
        arrow.next_to(article, LEFT)

        lines = VGroup(*[Line(LEFT, RIGHT) for x in range(20)])
        lines.arrange(DOWN)
        for line in (lines[9], lines[19]):
            line.stretch(random.random() * 0.7, 0, about_edge=LEFT)
        lines[10:].shift(SMALL_BUFF * DOWN)
        lines.set_height(article.get_height() - 1, stretch=True)
        lines.set_width(article.get_width() - 0.5, stretch=True)
        lines.move_to(article)

        self.play(
            DrawBorderThenFill(article),
            ShowCreation(lines, run_time=2),
            ShowCreation(arrow),
            graph.set_height, 3,
            graph.next_to, arrow, LEFT,
        )
        article.add(lines)
        self.wait()

        new_article = article.copy()
        new_arrow = arrow.copy()

        likes = VGroup(*[SVGMobject("like") for x in range(3)])
        likes.set_stroke(width=0)
        likes.set_fill(BLUE)
        likes.arrange(RIGHT)
        likes.match_width(new_article)
        likes.next_to(new_article, UP)

        self.play(VGroup(graph, arrow, article).next_to, new_arrow, LEFT)
        self.play(
            ShowCreation(new_arrow),
            TransformFromCopy(article, new_article, path_arc=30 * DEGREES),
        )
        self.play(LaggedStartMap(FadeInFrom, likes, lambda m: (m, DOWN)))
        self.wait()
        self.add(new_article, graph)

        new_article.generate_target()
        new_article.target.set_height(FRAME_HEIGHT, stretch=True)
        new_article.target.set_width(FRAME_WIDTH, stretch=True)
        new_article.target.center()
        new_article.target.set_stroke(width=0)

        self.play(
            Restore(graph, run_time=2),
            MoveToTarget(new_article, run_time=2),
            FadeOut(arrow),
            FadeOut(new_arrow),
            FadeOut(article),
            FadeOut(likes),
        )
        self.wait()

        # Replace cats with people
        lil_person = get_person()
        lil_person.replace(lil_cat, dim_to_match=1)
        big_person = get_person()
        big_person.replace(fat_cat, dim_to_match=1)

        self.play(
            FadeOut(lil_cat, LEFT),
            FadeIn(lil_person, RIGHT),
        )
        self.play(
            FadeOut(fat_cat, LEFT),
            FadeIn(big_person, RIGHT),
        )
        self.wait()

        cross = Cross(big_person)
        self.play(ShowCreation(cross))

        muscle = get_muscle()
        muscle.set_width(1.5)
        muscle.move_to(big_person)

        self.play(
            FadeIn(muscle, RIGHT),
            FadeOut(big_person, LEFT),
            FadeOut(cross, LEFT),
        )
        self.wait()

        cross = Cross(new_article)
        cross.set_stroke(RED, 30)
        self.play(ShowCreation(cross))
        self.wait()
