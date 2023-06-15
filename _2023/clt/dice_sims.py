from manim_imports_ext import *
from _2023.clt.main import *

# Dice Simulations


class DiceSimulation(InteractiveScene):
    n_dice = 10
    n_samples = 3000
    distribution = [1 / 6] * 6
    die_distribution_config = dict(
        axes_config=dict(width=3.5, height=1.5),
        y_range=(0, 0.5, 0.25),
    )
    brick_height = 0.2
    initial_brick_color = YELLOW
    full_dist_color = GREEN_D
    spread = 22
    dice_width = 5

    def setup(self):
        super().setup()
        self.add_die_distribution()
        self.add_sum_axes()
        self.add_sample_label()
        self.buckets = dict()
        self.all_bricks = VGroup()

    def construct(self):
        # Slow samples
        for _ in range(3):
            self.run_one_sum()

        # Speedier samples
        for _ in range(10):
            self.run_one_sum(run_time=0.5, transition_time=0.5)

        # First few instant samples
        for _ in range(200):
            self.run_one_sum(transition_time=1 / 30, still_frame=True)

        # Thousands of samples
        self.scale_brick_height(0.1)
        for _ in range(self.n_samples - int(self.sample_label.count.get_value())):
            self.run_one_sum(transition_time=1 / 30, still_frame=True)

        self.play(self.all_bricks.animate.set_fill(color=self.full_dist_color))

    def add_die_distribution(self, animate=False):
        die_dist = get_die_distribution_chart(
            self.distribution,
            **self.die_distribution_config,
        )
        die_dist.to_corner(UL)
        die_dist.shift(0.25 * DOWN)

        axes, bars, dice = die_dist

        if animate:
            self.play(
                Write(axes),
                LaggedStartMap(GrowFromEdge, bars, edge=DOWN),
                Write(dice),
            )

        self.add(die_dist)

        self.die_dist = die_dist

    def add_sum_axes(self):
        # (Value, probability) pairs
        vps = list(zip(it.count(1), self.distribution))
        mu = sum(v * p for v, p in vps)
        var = sum(p * (v - mu)**2 for v, p in vps)

        n = self.n_dice
        x_mid = int(n * mu)
        spread = self.spread or int(2.5 * math.sqrt(var * n))

        x_range = (x_mid - spread, x_mid + spread, 1)
        axes = Axes(
            x_range,
            (0, int(math.sqrt(self.n_samples))),
            width=FRAME_WIDTH - 1,
            height=4,
        )

        # dots = Tex(R"\dots", font_size=24)
        # dots.next_to(axes.x_axis, LEFT, SMALL_BUFF)
        # axes.y_axis.set_x(dots.get_left()[0] - SMALL_BUFF)
        # axes.add(dots)

        x_axis = axes.x_axis
        x_axis.add_numbers(font_size=16)
        x_axis.numbers.remove(x_axis.numbers[-1])
        x_axis.numbers.shift(
            0.5 * x_axis.get_unit_size() * RIGHT + \
            0.1 * UP
        )

        # axes.center()
        # axes.to_edge(DOWN)

        x_axis.center()
        x_axis.to_edge(DOWN)

        self.add(x_axis)

        self.sum_axes = axes
        self.sum_axis = x_axis

    def add_sample_label(self):
        label = TexText(R"\# Sums = 0", font_size=36)
        label.set_fill(GREY_B)
        label.to_edge(LEFT)
        label.count = label.make_number_changable("0")
        label.count.edge_to_fix = LEFT

        self.add(label)

        self.sample_label = label

    def run_one_sum(self, run_time=3.0, transition_time=1.0, still_frame=False):
        # Setup sample
        values = list(range(1, 7))
        bars = self.die_dist.bars
        samples = np.random.choice(values, size=self.n_dice, p=self.distribution)
        dice = VGroup(*(DieFace(sample) for sample in samples))
        dice.arrange_in_grid(n_cols=5)
        dice.set_width(self.dice_width)
        dice.to_corner(UR)

        bar_highlights = VGroup(*(
            bars[sample - 1].copy()
            for sample in samples
        ))
        bar_highlights.set_fill(YELLOW, 1)

        face_counts = [0] * len(self.distribution)
        for sample in samples:
            face_counts[sample - 1] += 1

        # Tips
        tips = get_sample_markers(bars, samples)

        def get_sum():
            return sum(d.value for d in dice)

        # Sum label
        sum_label = TexText(f"Sum = 0")
        sum_label.count = sum_label.make_number_changable(0)
        sum_label.next_to(dice, DOWN, buff=MED_LARGE_BUFF)
        sum_label.count.set_value(get_sum())

        # Mark the sample
        brick = self.get_brick()
        s = get_sum()
        if s not in self.buckets:
            self.buckets[s] = VGroup(VectorizedPoint(
                self.sum_axis.n2p(s + 0.5)
            ))
        bucket = self.buckets[s]

        brick.next_to(bucket, UP, buff=0)

        count_copy = sum_label.count.copy()
        count_copy.target = bucket[0]
        for number in self.sum_axis.numbers:
            if number.get_value() == sum_label.count.get_value():
                sum_label.count.target = number
                break

        # Animate!
        if still_frame:
            self.all_bricks.set_fill(color=self.full_dist_color)
            self.sample_label.count.increment_value()
            self.add(dice, brick, sum_label, tips)
            self.wait(transition_time)
            self.remove(dice, sum_label, tips)
        else:
            self.play(
                ShowIncreasingSubsets(dice, int_func=np.ceil),
                ShowIncreasingSubsets(tips, int_func=np.ceil),
                ShowSubmobjectsOneByOne(bar_highlights, remover=True),
                UpdateFromFunc(sum_label, lambda m: m.count.set_value(get_sum())),
                self.all_bricks.animate.set_fill(color=self.full_dist_color),
                run_time=run_time,
            )
            self.wait(transition_time)
            self.play(
                FadeInFromPoint(brick, count_copy.get_center()),
                MoveToTarget(count_copy),
                FadeOut(sum_label, lag_ratio=0.1),
                FadeOut(dice, lag_ratio=0.1),
                FadeOut(tips, lag_ratio=0.1),
                run_time=transition_time
            )
            self.sample_label.count.increment_value()
            self.wait(transition_time)

        self.buckets[s].add(brick)
        self.all_bricks.add(brick)
        self.add(self.all_bricks)

    def scale_brick_height(self, scale_factor):
        self.brick_height *= scale_factor
        bricks = self.all_bricks
        bricks.target = bricks.generate_target()
        bricks.target.stretch(scale_factor, 1, about_edge=DOWN)
        bricks.target.set_stroke(width=scale_factor * bricks[0].get_stroke_width())
        self.play(MoveToTarget(bricks), run_time=3)
        self.wait()

    def get_brick(self):
        return Rectangle(
            stroke_width=self.brick_height * 5,
            stroke_color=BLACK,
            fill_color=self.initial_brick_color,
            fill_opacity=1,
            height=self.brick_height,
            width=0.8 * self.sum_axis.get_unit_size()
        )


class DiceSimulationAlt1(DiceSimulation):
    random_seed = 1


class DiceSimulationAlt2(DiceSimulation):
    random_seed = 2


class DiceSimulationAlt3(DiceSimulation):
    random_seed = 3


class DiceSimulationAlt4(DiceSimulation):
    random_seed = 4


class DiceSimulationAlt5(DiceSimulation):
    random_seed = 5

    
class LargerDiceSimulation(DiceSimulation):
    n_samples = 3000
    brick_height = 0.02
    random_seed = 1

    def construct(self):
        # Larger sample
        for _ in range(self.n_samples):
            self.run_one_sum(transition_time=1 / 30, still_frame=True)


class SimulationWithUShapedDistribution(DiceSimulation):
    random_seed = 1
    distribution = U_SHAPED_DISTRIBUTION
    brick_height = 0.2
    n_samples = 3000

    def construct(self):
        # Transition
        tmp_dist = self.distribution
        tmp_die_dist = self.die_dist
        self.remove(tmp_die_dist)
        self.distribution = self.__class__.__base__.distribution
        self.add_die_distribution()
        self.play(Transform(self.die_dist, tmp_die_dist, run_time=3))

        self.distribution = tmp_dist

        # First few samples
        for _ in range(2):
            self.run_one_sum()
        for _ in range(6):
            self.run_one_sum(run_time=0.5, transition_time=0.5)
        for _ in range(300):
            self.run_one_sum(transition_time=1 / 30, still_frame=True)

        # Thousands of samples
        self.scale_brick_height(0.1)
        for _ in range(self.n_samples - int(self.sample_label.count.get_value())):
            self.run_one_sum(transition_time=1 / 30, still_frame=True)

        self.play(self.all_bricks.animate.set_fill(color=self.full_dist_color))


class LargerUSimulation(SimulationWithUShapedDistribution):
    n_samples = 10000
    brick_height = 0.01

    def construct(self):
        # Thousands of samples
        for _ in range(self.n_samples):
            self.run_one_sum(transition_time=1 / 30, still_frame=True)

    def get_brick(self):
        return super().get_brick().set_stroke(width=0)


class SteeperUDistributionSimulation(SimulationWithUShapedDistribution):
    distribution = STEEP_U_SHAPED_DISTRIBUTION


class SimulationWithSteepUShapedDistribution(LargerUSimulation):
    distribution = [0.4, 0.075, 0.025, 0.025, 0.075, 0.4]
    n_samples = 3000
    brick_height = 0.02


class SimulationWithExpDistribution(SimulationWithUShapedDistribution):
    random_seed = 1
    distribution = EXP_DISTRIBUTION


class SimulationWithExpDistribution2(SimulationWithExpDistribution):
    random_seed = 2


class SimulationWithExpDistribution2Dice(SimulationWithExpDistribution):
    n_dice = 2
    brick_height = 0.13
    dice_width = 2

    def get_brick(self):
        return super().get_brick().set_stroke(width=0)


class SimulationWithRandomDistribution(SimulationWithUShapedDistribution):
    random_seed = 1
    n_dice = 15
    distribution = [0.05, 0.17, 0.28, 0.05, 0.18, 0.27]


class SimulationWithExpDistribution5Dice(SimulationWithExpDistribution):
    n_dice = 5
    brick_height = 0.15


class SimulationWithExpDistribution15Dice(SimulationWithExpDistribution):
    n_dice = 15

