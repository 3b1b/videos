from manim_imports_ext import *
from _2023.clt.main import *


# Galton board

class GaltonBoard(InteractiveScene):
    random_seed = 1
    pegs_per_row = 15
    n_rows = 5
    spacing = 1.0
    top_buff = 1.0
    peg_radius = 0.1
    ball_radius = 0.1
    bucket_floor_buff = 1.0
    bucket_style = dict(
        fill_color=GREY_D,
        fill_opacity=1.0,
    )
    stack_ratio = 1.0
    fall_factor = 0.6
    # clink_sound = "click.wav"
    clink_sound = "plate.wav"

    def setup(self):
        super().setup()
        self.ball_template = Sphere(
            radius=self.ball_radius,
            color=YELLOW,
            resolution=(51, 26),
        )
        self.ball_template.rotate(90 * DEGREES, RIGHT)
        self.ball_template.set_shading(0.25, 0.5, 0.5)
        # ball = TrueDot(radius=self.ball_radius, color=color)
        # ball.make_3d()

    def construct(self):
        # Setup
        pegs = self.get_pegs()
        buckets = self.get_buckets(pegs)

        self.play(
            LaggedStartMap(Write, buckets),
            LaggedStartMap(Write, pegs),
        )
        self.wait()

        # Initial flurry
        balls = self.drop_n_balls(25, pegs, buckets, sound=True)
        self.wait()
        balls.reverse_submobjects()
        self.play(FadeOut(balls, lag_ratio=0.05))
        for bucket in buckets:
            bucket.balls.clear()
            bucket.bottom.match_y(bucket[0].get_bottom())

        # Single ball bouncing, step-by-step
        ball = self.get_ball()
        bits = np.random.randint(0, 2, self.n_rows)
        full_trajectory, pieces = self.random_trajectory(ball, pegs, buckets, bits)

        all_arrows = VGroup()
        for piece, bit in zip(pieces, bits):
            ball.move_to(piece.get_end())
            pm_arrows = self.get_pm_arrows(ball)
            self.play(self.falling_anim(ball, piece))
            self.add_single_clink_sound()
            self.play(FadeIn(pm_arrows, lag_ratio=0.1))
            all_arrows.add(pm_arrows)
            self.wait()
            self.play(pm_arrows[1 - bit].animate.set_opacity(0.25))
        for piece in pieces[-2:]:
            self.play(self.falling_anim(ball, piece))
        self.wait()

        # Add up arrows
        corner_sum_anim, corner_sum_fade = self.show_corner_sum(all_arrows, bits)
        self.play(corner_sum_anim)
        self.wait()

        # Show buckets as sums
        sums = range(-self.pegs_per_row + 2, self.pegs_per_row, 2)
        sum_labels = VGroup(*(
            Integer(s, font_size=24, include_sign=True) for s in sums
        ))
        for bucket, label in zip(buckets, sum_labels):
            label.next_to(bucket, DOWN, SMALL_BUFF)

        sum_labels.set_stroke(WHITE, 1)
        self.play(Write(sum_labels))
        self.wait()

        for bucket, label in zip(buckets, sum_labels):
            bucket.add(label)

        self.play(
            FadeOut(all_arrows, lag_ratio=0.025),
            corner_sum_fade
        )

        # Show a few more trajectories with cumulative sum
        for x in range(3):
            ball = self.get_ball()
            bits = np.random.randint(0, 2, self.n_rows)
            full_trajectory, pieces = self.random_trajectory(ball, pegs, buckets, bits)

            all_arrows = VGroup()

            self.add(all_arrows)
            for piece, bit in zip(pieces, bits):
                ball.move_to(piece.get_end())
                arrows = self.get_pm_arrows(ball)
                all_arrows.add(arrows)
                self.play(self.falling_anim(ball, piece))
                self.add_single_clink_sound()
                arrows[1 - bit].set_opacity(0.25)

            corner_sum_anim, corner_sum_fade = self.show_corner_sum(all_arrows, bits)
            self.play(self.falling_anim(ball, pieces[-2]))
            self.add_single_clink_sound()
            self.play(
                self.falling_anim(ball, pieces[-1]),
                corner_sum_anim,
            )
            self.wait()
            self.play(
                FadeOut(all_arrows, lag_ratio=0.025),
                corner_sum_fade
            )

        # Show a flurry
        self.drop_n_balls(25, pegs, buckets)

        # Fade out irrelevant parts
        n = self.pegs_per_row // 2
        to_fade = VGroup()
        peg_triangle = VGroup()
        for row in range(self.n_rows):
            r2 = row // 2
            low = n - r2
            high = n + 1 + r2 + (row % 2)
            to_fade.add(pegs[row][:low])
            to_fade.add(pegs[row][high:])
            peg_triangle.add(pegs[row][low:high])

        to_fade.add(buckets[:n - 3])
        to_fade.add(buckets[n + 3:])

        self.play(to_fade.animate.set_opacity(0.25), lag_ratio=0.01)

        # Show relevant probabilities
        point = peg_triangle[0][0].get_top() + MED_SMALL_BUFF * UP
        v1 = peg_triangle[1][0].get_center() - peg_triangle[0][0].get_center()
        v2 = peg_triangle[1][1].get_center() - peg_triangle[1][0].get_center()

        def get_peg_label(n, k, split=False):
            kw = dict(font_size=16)
            if n == 0:
                label = Tex("1", font_size=24)
            elif split and 0 < k < n:
                label = VGroup(
                    Tex(f"{choose(n - 1, k - 1)} \\over {2**n}", **kw),
                    Tex(f" + {{{choose(n - 1, k)} \\over {2**n}}}", **kw),
                )
                label.arrange(RIGHT, buff=0.75 * label[0].get_width())
            else:
                label = VGroup(Tex(f"{choose(n, k)} \\over {2**n}", **kw))
            label.move_to(point + n * v1 + k * v2)
            return label

        last_labels = VGroup(get_peg_label(0, 0))
        self.play(FadeIn(last_labels))
        for n in range(1, self.n_rows + 1):
            split_labels = VGroup(*(get_peg_label(n, k, split=True) for k in range(n + 1)))
            unsplit_labels = VGroup(*(get_peg_label(n, k, split=False) for k in range(n + 1)))
            anims = [
                TransformFromCopy(last_labels[0], split_labels[0]),
                TransformFromCopy(last_labels[-1], split_labels[-1])
            ]
            for k in range(1, n):
                anims.append(TransformFromCopy(last_labels[k - 1], split_labels[k][0]))
                anims.append(TransformFromCopy(last_labels[k], split_labels[k][1]))

            self.play(*anims)
            self.play(*(
                FadeTransformPieces(sl1, sl2)
                for sl1, sl2 in zip(split_labels, unsplit_labels)
            ))

            last_labels = unsplit_labels

        # Larger flurry
        all_balls = Group()
        for bucket in buckets:
            bucket.bottom.match_y(bucket[0].get_bottom())
            all_balls.add(*bucket.balls)
            bucket.balls.clear()
        self.play(LaggedStartMap(FadeOut, all_balls, run_time=1))

        self.stack_ratio = 0.125
        np.random.seed(0)
        self.drop_n_balls(250, pegs, buckets, lr_factor=2)
        self.wait(2)

    def get_pegs(self):
        row = VGroup(*(
            Dot(radius=self.peg_radius).shift(x * self.spacing * RIGHT)
            for x in range(self.pegs_per_row)
        ))
        rows = VGroup(*(
            row.copy().shift(y * self.spacing * DOWN * math.sqrt(3) / 2)
            for y in range(self.n_rows)
        ))
        rows[1::2].shift(0.5 * self.spacing * LEFT)

        rows.set_fill(GREY_C, 1)
        rows.set_shading(0.5, 0.5)
        rows.center()
        rows.to_edge(UP, buff=self.top_buff)

        return rows

    def get_buckets(self, pegs):
        # Buckets
        points = [dot.get_center() for dot in pegs[-1]]
        height = 0.5 * FRAME_HEIGHT + pegs[-1].get_y() - self.bucket_floor_buff

        buckets = VGroup()
        for point in points:
            # Fully construct bucket here
            width = 0.5 * self.spacing - self.ball_radius
            buff = 0.7
            p0 = point + 0.5 * self.spacing * DOWN + buff * width * RIGHT
            p1 = p0 + height * DOWN
            p2 = p1 + (1 - buff) * width * RIGHT
            y = point[1] - 0.5 * self.spacing * math.sqrt(3) + self.ball_radius
            p3 = p2[0] * RIGHT + y * UP
            side1 = VMobject().set_points_as_corners([p0, p1, p2, p3, p0])
            side1.set_stroke(WHITE, 0)
            side1.set_style(**self.bucket_style)

            side2 = side1.copy()
            side2.flip(about_point=point)
            side2.reverse_points()
            side2.shift(self.spacing * RIGHT)

            floor = Line(side1.get_corner(DR), side2.get_corner(DL))
            floor.set_stroke(GREY_D, 1)
            bucket = VGroup(side1, side2, floor)
            bucket.set_shading(0.25, 0.25)

            # Add  bottom reference
            bucket.bottom = VectorizedPoint(floor.get_center())
            bucket.add(bucket.bottom)

            # Keep track of balls
            bucket.balls = Group()

            buckets.add(bucket)

        self.add(buckets)

        return buckets

    def get_ball_arrows(self, ball, labels, sub_labels=[], colors=[RED, BLUE]):
        arrows = VGroup()
        for vect, color, label in zip([LEFT, RIGHT], colors, labels):
            arrow = Vector(
                0.5 * self.spacing * vect,
                tip_width_ratio=3,
                stroke_color=color
            )
            arrow.next_to(ball, vect, buff=0.1)
            arrows.add(arrow)
            text = TexText(label, font_size=28)
            text.next_to(arrow, UP, SMALL_BUFF)
            arrow.add(text)
        # Possibly add smaller labels
        for arrow, label in zip(arrows, sub_labels):
            text = Text(label, font_size=16)
            text.next_to(arrow, DOWN, SMALL_BUFF)
            arrow.add(text)
        return arrows

    def get_fifty_fifty_arrows(self, ball):
        return self.get_ball_arrows(ball, ["50%", "50%"])

    def get_pm_arrows(self, ball, show_prob=True):
        return self.get_ball_arrows(
            ball, ["$-1$", "$+1$"],
            sub_labels=(["50%", "50%"] if show_prob else [])
        )

    def show_corner_sum(self, pm_arrows, bits, font_size=48):
        # Test
        parts = VGroup(*(
            arrow[bit][0].copy()
            for arrow, bit in zip(pm_arrows, bits)
        ))
        parts.target = parts.generate_target()
        parts.target.arrange(RIGHT, buff=0.1)
        parts.target.scale(font_size / 28)
        parts.target.to_edge(UP, buff=MED_SMALL_BUFF)
        parts.target.to_edge(LEFT)

        anim1 = MoveToTarget(parts, lag_ratio=0.01)

        sum_term = Tex(f"= {2 * sum(bits) - len(bits)}", font_size=font_size)
        sum_term.next_to(parts.target, RIGHT, buff=0.1, aligned_edge=UP)

        anim2 = LaggedStart(*(
            ReplacementTransform(
                part.copy().set_opacity(0),
                sum_term,
                path_arc=-30 * DEGREES
            )
            for part in parts.target
        ))

        return Succession(anim1, anim2), FadeOut(VGroup(parts, sum_term))

    def get_ball(self, color=YELLOW_E):
        ball = TrueDot(radius=self.ball_radius, color=color)
        ball.make_3d()
        ball.set_shading(0.5, 0.5, 0.2)
        return ball

    def single_bounce_trajectory(self, ball, peg, direction):
        sgn = np.sign(direction[0])
        trajectory = FunctionGraph(
            lambda x: -x * (x - 1),
            x_range=(0, 2, 0.2),
        )
        p1 = peg.get_top()
        p2 = p1 + self.spacing * np.array([sgn * 0.5, -0.5 * math.sqrt(3), 0])
        vect = trajectory.get_end() - trajectory.get_start()
        for i in (0, 1):
            trajectory.stretch((p2 - p1)[i] / vect[i], i)
        trajectory.shift(p1 - trajectory.get_start() + 0.5 * ball.get_height() * UP)

        return trajectory

    def random_trajectory(self, ball, pegs, buckets, bits=None):
        index = len(pegs[0]) // 2
        radius = ball.get_height() / 2
        peg = pegs[0][index]

        top_line = ParametricCurve(lambda t: t**2 * DOWN)
        top_line.move_to(peg.get_top() + radius * UP, DOWN)

        bounces = []
        if bits is None:
            bits = np.random.randint(0, 2, self.n_rows)
        for row, bit in enumerate(bits):
            peg = pegs[row][index]
            bounces.append(self.single_bounce_trajectory(ball, peg, [LEFT, RIGHT][bit]))
            index += bit
            if row % 2 == 1:
                index -= 1
        bucket = buckets[index + (0 if self.n_rows % 2 == 0 else -1)]
        final_line = Line(
            bounces[-1].get_end(),
            bucket.bottom.get_center() + self.ball_radius * UP
        )
        final_line.insert_n_curves(int(8 * final_line.get_length()))
        bucket.bottom.shift(2 * self.ball_radius * self.stack_ratio * UP)
        bucket.balls.add(ball)

        result = VMobject()
        pieces = VGroup(top_line, *bounces, final_line)
        for vmob in pieces:
            if result.get_num_points() > 0:
                vmob.shift(result.get_end() - vmob.get_start())
            result.append_vectorized_mobject(vmob)

        return result, pieces

    def falling_anim(self, ball, trajectory):
        return MoveAlongPath(
            ball, trajectory,
            rate_func=linear,
            run_time=self.fall_factor * trajectory.get_arc_length()
        )

    def add_single_clink_sound(self, time_offset=0, gain=-20):
        self.add_sound(
            sound_file=self.clink_sound.replace("click", "click" + str(random.randint(1, 12))),
            time_offset=time_offset,
            gain=gain,
        )

    def add_falling_clink_sounds(self, trajectory_pieces, time_offset=0, gain=-20):
        total_len = trajectory_pieces[0].get_arc_length()
        for piece in trajectory_pieces[1:-1]:
            self.add_single_clink_sound(time_offset + self.fall_factor * total_len, gain)
            total_len += piece.get_arc_length()

    def drop_n_balls(self, n, pegs, buckets, lr_factor=1, sound=False):
        # Test
        balls = Group(*(self.get_ball() for x in range(n)))
        trajs = [
            self.random_trajectory(ball, pegs, buckets)
            for ball in balls
        ]
        anims = (
            self.falling_anim(ball, traj[0])
            for ball, traj in zip(balls, trajs)
        )
        full_anim = LaggedStart(*anims, lag_ratio=lr_factor / n)

        # Add sounds
        if sound:
            start_times = [tup[1] for tup in full_anim.anims_with_timings]
            for time, traj in zip(start_times, trajs):
                self.add_falling_clink_sounds(traj[1], time + 0.00 * random.random(), gain=-30)

        self.play(full_anim)

        return balls


class EmphasizeMultipleSums(GaltonBoard):
    def construct(self):
        pegs = self.get_pegs()
        buckets = self.get_buckets(pegs)
        self.add(pegs, buckets)

        # Show a trajectories with cumulative sum
        for x in range(20):
            ball = self.get_ball()
            bits = np.random.randint(0, 2, self.n_rows)
            full_trajectory, pieces = self.random_trajectory(ball, pegs, buckets, bits)

            all_arrows = VGroup()

            self.add(all_arrows)
            for piece, bit in zip(pieces, bits):
                ball.move_to(piece.get_end())
                arrows = self.get_pm_arrows(ball)
                all_arrows.add(arrows)
                self.play(self.falling_anim(ball, piece))
                self.add_single_clink_sound()
                arrows[1 - bit].set_opacity(0.25)

            self.play(self.falling_anim(ball, pieces[-2]))
            self.add_single_clink_sound()
            self.play(
                self.falling_anim(ball, pieces[-1]),
                FadeOut(all_arrows)
            )


class GaltonTrickle(GaltonBoard):
    def construct(self):
        frame = self.frame

        pegs = self.get_pegs()
        buckets = self.get_buckets(pegs)
        self.add(pegs, buckets)

        ball = self.get_ball()
        peg = pegs[0][len(pegs[0]) // 2]
        ball.move_to(peg.get_top(), DOWN)
        arrows = self.get_pm_arrows(ball)

        frame.set_height(3, about_edge=UP)

        # Drops
        n = 25

        balls = Group(*(self.get_ball() for x in range(n)))
        all_bits = [np.random.randint(0, 2, self.n_rows) for x in range(n)]
        trajs = [
            self.random_trajectory(ball, pegs, buckets, bits)
            for ball, bits in zip(balls, all_bits)
        ]
        falling_anims = (
            self.falling_anim(ball, traj[0])
            for ball, traj in zip(balls, trajs)
        )

        arrow_copies = VGroup()
        for bits in all_bits:
            ac = arrows.copy()
            ac[1 - bits[0]].set_opacity(0.2)
            arrow_copies.add(ac)

        rt = 60
        arrows.set_opacity(1)
        self.add(arrows)
        self.play(
            LaggedStart(*falling_anims, lag_ratio=0.4, run_time=rt),
            # ShowSubmobjectsOneByOne(arrow_copies, run_time=1.0 * rt),
        )
        self.wait()


class BiggerGaltonBoard(GaltonBoard):
    random_seed = 0
    pegs_per_row = 30
    n_rows = 13
    spacing = 0.5
    top_buff = 0.5
    peg_radius = 0.025
    ball_radius = 0.05
    bucket_floor_buff = 0.5
    stack_ratio = 0.1
    n_balls = 800

    def construct(self):
        # Setup
        pegs = self.get_pegs()
        buckets = self.get_buckets(pegs)
        self.add(pegs, buckets)

        # Drop!
        self.drop_n_balls(self.n_balls, pegs, buckets, lr_factor=2)
        self.wait()

        # Show low bell cuve
        full_rect = FullScreenFadeRectangle()
        full_rect.set_fill(BLACK, 0.5)
        balls = self.mobjects[-1]
        curve = FunctionGraph(lambda x: gauss_func(x, 0, 1))
        curve.set_stroke(YELLOW)
        curve.move_to(balls, DOWN)
        curve.match_height(balls, stretch=True, about_edge=DOWN)
        formula = Tex(R"{1 \over \sqrt{2\pi}} e^{-x^2 / 2}", font_size=60)
        formula.move_to(balls, LEFT)
        formula.shift(1.25 * LEFT)
        formula.set_backstroke(width=8)

        self.add(full_rect, balls)
        self.play(
            FadeIn(full_rect),
            ShowCreation(curve, run_time=2),
            Write(formula)
        )
        self.wait()


class SingleDropBigGaltonBoard(BiggerGaltonBoard):
    spacing = 0.55
    ball_radius = 0.075

    def construct(self):
        # Setup
        pegs = self.get_pegs()
        buckets = self.get_buckets(pegs)
        self.add(pegs, buckets)

        # Single ball bouncing, step-by-step
        ball = self.get_ball()
        full_trajectory, pieces = self.random_trajectory(ball, pegs, buckets)
        self.add_falling_clink_sounds(pieces)
        self.play(self.falling_anim(ball, full_trajectory))
        self.wait()


class NotIdenticallyDistributed(GaltonBoard):
    def construct(self):
        # Setup
        pegs = self.get_pegs()
        buckets = self.get_buckets(pegs)
        self.add(pegs, buckets)

        # Arrows to show distributions
        max_arrow_len = 0.5

        def get_peg_arrow(peg, angle, length, color=RED_E):
            vect = np.array([-math.sin(angle), math.cos(angle), 0])
            arrow = FillArrow(
                ORIGIN, length * vect,
                buff=0,
                fill_color=color,
                tip_width_ratio=3,
                thickness=0.025,
            )
            arrow.shift(peg.get_center() + vect * peg.get_radius())
            arrow.set_fill(opacity=0.8 * length / max_arrow_len)
            return arrow

        def get_bounce_distribution(peg, sigma=30 * DEGREES):
            ds = sigma / 2
            angles = np.arange(-2 * sigma, 2 * sigma + ds, ds)
            denom = math.sqrt(2 * PI) * sigma
            arrows = VGroup(*(
                get_peg_arrow(peg, angle, denom * gauss_func(angle, 0, sigma) * max_arrow_len)
                for angle in angles
            ))
            return arrows

        # Show many distributions
        all_dists = VGroup(*(
            get_bounce_distribution(peg)
            for row in pegs
            for peg in row
        ))

        all_dists.set_fill(RED_E, 0.8)
        self.play(LaggedStart(*(
            LaggedStartMap(GrowArrow, dist)
            for dist in all_dists
        )))
        self.wait()

        # Zoom in to top one
        ball = self.get_ball()
        peg1 = pegs[0][len(pegs[0]) // 2]
        peg2 = pegs[1][len(pegs[1]) // 2]
        frame = self.frame
        peg1_dist = get_bounce_distribution(peg1)
        peg2_dist = get_bounce_distribution(peg2)
        peg1_dist.rotate(30 * DEGREES, about_point=peg1.get_center())
        peg2_dist.rotate(-30 * DEGREES, about_point=peg2.get_center())

        full_trajectory, pieces = self.random_trajectory(ball, pegs, buckets, [0, 1, 0, 0, 0])
        pieces[0].move_to(peg1.pfp(3 / 8) + ball.get_radius() * UP, DOWN)
        pieces[1].stretch(0.7, 0)
        pieces[1].shift(pieces[0].get_end() - pieces[1].get_start())
        pieces[2].stretch(0.9, 0)
        pieces[2].stretch(0.97, 1)
        pieces[2].shift(pieces[1].get_end() - pieces[2].get_start())

        self.play(
            frame.animate.set_height(3, about_edge=UP),
            FadeOut(all_dists, lag_ratio=0.01),
            self.falling_anim(ball, pieces[0]),
            run_time=2,
        )
        self.add(peg1_dist, ball)
        self.play(LaggedStartMap(FadeIn, peg1_dist))
        self.wait()
        self.play(self.falling_anim(ball, pieces[1]), run_time=1)
        self.play(LaggedStartMap(FadeIn, peg2_dist))
        self.wait()
        self.play(self.falling_anim(ball, pieces[2]), run_time=1)
        self.wait(2)

