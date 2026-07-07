from manim_imports_ext import *


class CrossEntropyKLDivergenceTitles(InteractiveScene):
    def construct(self):
        # Test
        titles = VGroup(
            Text("Cross-Entropy", font_size=60),
            Text("KL Divergence", font_size=60),
        )
        titles.to_edge(UP)
        kw = dict(t2c={"p_i": GREEN, "q_i": PINK})
        formulas = VGroup(
            Tex(R"\sum_i p_i \big(-\log(q_i)\big)", **kw),
            Tex(R"\sum_i p_i \log\left({p_i / q_i}\right)", **kw),
        )

        for x, title, formula in zip([-1, 1], titles, formulas):
            title.set_x(x * FRAME_WIDTH / 4)
            formula.next_to(title, DOWN)

            self.play(LaggedStart(
                Write(title),
                FadeIn(formula, 0.5 * DOWN),
                lag_ratio=0.25
            ))
            self.wait()


class AskWhyLogs(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        randy.flip()
        randy.to_edge(RIGHT)

        self.play(randy.says("Why logs?", mode="maybe", look_at=ORIGIN))
        self.play(Blink(randy))
        self.wait()
        self.play(randy.change("confused"))
        self.wait()


class PreviousVideos(InteractiveScene):
    def construct(self):
        # Test
        images = Group(
            ImageMobject("https://img.youtube.com/vi/IHZwWFHWa-w/maxresdefault.jpg"),
            ImageMobject("https://img.youtube.com/vi/Ilg3gGewQ5U/maxresdefault.jpg"),
        )
        thumbnails = Group(
            Group(
                SurroundingRectangle(image, buff=0).set_stroke(WHITE, 3),
                image
            )
            for image in images
        )
        thumbnails.set_width(4)
        thumbnails.arrange(DOWN, buff=1.5)
        thumbnails.to_corner(UL).to_edge(LEFT, buff=LARGE_BUFF)
        titles = VGroup(Text("Gradient Descent"), Text("Backpropagation"))

        for title, tn in zip(titles, thumbnails):
            title.next_to(tn, DOWN)
            self.play(LaggedStart(
                FadeIn(tn, 0.5 * DOWN),
                FadeIn(title, 1.0 * DOWN),
                lag_ratio=0.2
            ))
        self.wait()


class YouTheEngineer(InteractiveScene):
    def construct(self):
        # test
        randy = Randolph()
        randy.to_edge(LEFT)

        self.play(randy.change("erm", RIGHT))
        self.play(Blink(randy))
        self.wait(2)
        self.play(randy.change("hesitant", RIGHT))
        self.play(Blink(randy))
        self.wait()
        self.play(Blink(randy))
        self.wait(2)
        self.play(randy.change("pondering", self.frame.get_corner(UL)))
        self.play(Blink(randy))

        # Away
        self.play(
            randy.change("horrified"),
            VFadeOut(randy),
            self.frame.animate.shift(2 * UP)
        )
        self.wait()


class BillionsOfInputs(InteractiveScene):
    def construct(self):
        # Simple function
        func = Tex(R"L(x, y)", font_size=90, t2c={"L": RED})
        self.play(Write(func))
        self.wait()

        # Many labels
        lp = func["L("][0]
        rp = func[")"][0]
        xy = func["x, y"][0]

        lp.target = lp.generate_target()
        rp.target = rp.generate_target()
        lp.target.shift(2 * LEFT)
        rp.target.shift(2 * RIGHT)
        min_x = lp.target.get_x(RIGHT)
        max_x = rp.target.get_x(LEFT)

        w_i_template = Tex(R"w_{0}", font_size=48)
        subscr = w_i_template.make_number_changeable("0", edge_to_fix=LEFT)
        all_ws = VGroup()
        n_terms = 200
        for n in range(n_terms):
            subscr.set_value(n)
            all_ws.add(w_i_template.copy())
        all_ws[-1].remove(all_ws[-1][-1])
        all_ws.arrange(RIGHT)
        for mob in all_ws:
            mob.add(Text(",").next_to(mob, RIGHT, buff=0.05).match_y(mob.get_corner(DL)))

        def update_opacities(xs):
            for mob in xs:
                x = mob.get_x()
                alpha = min(
                    clip(inverse_interpolate(min_x, min_x + LARGE_BUFF, x), 0, 1),
                    clip(inverse_interpolate(max_x, max_x - LARGE_BUFF, x), 0, 1),
                )
                mob.set_fill(opacity=alpha)

        all_ws.add_updater(update_opacities)
        all_ws.next_to(lp.target, RIGHT, SMALL_BUFF)

        self.play(
            MoveToTarget(lp),
            MoveToTarget(rp),
            FadeOut(xy),
            VFadeIn(all_ws, suspend_mobject_updating=True),
        )
        self.play(
            all_ws.animate.next_to(rp, LEFT),
            run_time=20,
            rate_func=lambda t: t**3,
        )
        self.wait()

