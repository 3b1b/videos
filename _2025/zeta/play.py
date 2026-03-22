from manim_imports_ext import *
from sympy import primerange
from sympy import factorint
import mpmath


def get_first_n_primes(n=100_000):
    pass
    """Generate the first n prime numbers."""
    primes = []
    prime_gen = primerange(1, float('inf'))
    for _ in range(n):
        primes.append(next(prime_gen))
    return primes


def get_zeta_zeros(n_zeros=10):
    return [complex(mpmath.zetazero(k + 1)) for k in range(n_zeros)]


class ZetaSum(InteractiveScene):
    n_vects = 100
    max_N = 100_000
    show_c = True
    sum_tex = R"\sum_{n=1}^\infty \frac{1}{n^{s}}"
    exp_tex = R"\cdot e^{\minus {c} n}"

    def setup(self):
        super().setup()
        self.exp_const_tracker = ValueTracker(0)
        self.s_plane, self.out_plane = self.add_planes()
        self.s_group = self.add_s_group(self.s_plane)
        self.s_tracker = self.s_group[0]
        self.sum_label = self.add_sum_label()

        self.generate_sample_ns()

    def construct(self):
        # Set up
        s_tracker, s_dot, s_label = self.s_group
        exp_const_tracker = self.exp_const_tracker
        zeta_zeros = get_zeta_zeros()

        # Add sum
        self.add_partial_sum_path()

        # Move to various values
        self.play(s_tracker.animate.set_value(zeta_zeros[0]), run_time=3)

        # Change exponential constant
        self.play(exp_const_tracker.animate.set_value(0.01), run_time=3)

    def add_planes(self):
        planes = VGroup(
            ComplexPlane((-1, 3), (-40, 40)),
            ComplexPlane((-3, 3), (-3, 3)),
        )
        planes[0].set_width(4)
        planes[1].set_width(6)
        planes.arrange(RIGHT, buff=2.0)
        planes.set_y(-0.5).to_edge(LEFT)
        for plane in planes:
            plane.add_coordinate_labels(font_size=16)
        self.add(planes)
        return planes

    def add_s_group(self, s_plane):
        s_tracker = ComplexValueTracker(1)
        get_s = s_tracker.get_value
        s_dot = Group(TrueDot(), GlowDot())
        s_dot.set_color(YELLOW)
        s_dot.add_updater(lambda m: m.move_to(s_plane.n2p(get_s())))
        s_label = Tex(R"s").set_color(YELLOW)
        s_label.always.next_to(s_dot[0], UR, SMALL_BUFF)

        s_group = Group(s_tracker, s_dot, s_label)
        self.add(s_group)
        return s_group

    def add_sum_label(self):
        out_plane = self.out_plane
        get_exp_const = self.exp_const_tracker.get_value

        sum_label = Tex(
            self.sum_tex + (self.exp_tex if self.show_c else ""),
            t2c={"{s}": YELLOW, "{c}": RED},
            font_size=42
        )
        sum_label.next_to(out_plane, UP, aligned_edge=LEFT)
        self.add(sum_label)

        if self.show_c:
            exp_const_label = Tex(R"c = 0.0000", t2c={"c": RED})
            exp_const_label.next_to(sum_label, RIGHT, LARGE_BUFF)
            c_dec = exp_const_label.make_number_changeable("0.0000")
            c_dec.add_updater(lambda m: m.set_value(get_exp_const()))
            sum_label.add(exp_const_label)

        return sum_label

    def add_partial_sum_path(self):
        self.partial_sum_path = self.get_partial_sum_path()
        self.vect_sum = self.get_vect_sum(self.partial_sum_path)

        self.add(self.partial_sum_path)
        self.add(self.vect_sum)

    def generate_sample_ns(self):
        self.sample_ns = np.arange(1, self.max_N + 1)

    def get_summands(self, s):
        exp_const = self.exp_const_tracker.get_value()
        weights = np.exp(-exp_const * self.sample_ns)
        return weights * (self.sample_ns**(-s))

    def get_partial_sum_path(self, stroke_color=TEAL, stroke_width=1):
        out_plane = self.out_plane

        path = VMobject()

        def update_path(path):
            summands = self.get_summands(self.s_tracker.get_value())
            partial_sums = np.hstack([[0], np.cumsum(summands)])
            points = out_plane.n2p(partial_sums)
            path.set_points_as_corners(points)

        path.add_updater(update_path)
        path.set_stroke(TEAL, 1)
        return path

    def get_vect_sum(self, path, vect_colors=[TEAL, GREEN], thickness=3):
        vects = VGroup(
            Vector(RIGHT, thickness=thickness, fill_color=color)
            for n, color in zip(range(self.n_vects), it.cycle(vect_colors))
        )
        vects.set_fill(border_width=1)

        def update_vects(vects):
            points = path.get_anchors()
            for vect, p0, p1 in zip(vects, points, points[1:]):
                vect.put_start_and_end_on(p0, p1)

        vects.add_updater(update_vects)
        return vects


class ZetaLogDerivSum(ZetaSum):
    sum_tex = R"-\frac{\zeta'(s)}{\zeta(s)} = \sum_{\substack{p \, \text{prime} \\ k \in \mathds{N}}} \log(p) \left(\frac{1}{p^{k}} \right)^s"
    exp_tex = R"\cdot e^{\minus {c} p^k}"
    show_c = False
    max_N = 1_000_000

    def construct(self):
        # Set up
        self.frame.reorient(0, 0, 0, RIGHT, 10)
        s_tracker = self.s_tracker
        exp_const_tracker = self.exp_const_tracker
        zeta_zeros = get_zeta_zeros()

        # Add sum
        partial_sum_path = self.get_partial_sum_path()
        vect_sum = self.get_vect_sum(partial_sum_path)

        self.add(partial_sum_path)
        self.add(vect_sum)

        # Move to various values
        self.play(s_tracker.animate.set_value(-1 + 5j))
        self.play(s_tracker.animate.set_value(0.5 + 5j), run_time=3)
        self.play(s_tracker.animate.set_value(zeta_zeros[5]), run_time=5)
        self.play(s_tracker.animate.increment_value(-0.25), run_time=3)

        self.play(exp_const_tracker.animate.set_value(0.1), run_time=3)

    def generate_sample_ns(self):
        base_weights = []
        sample_ns = []
        for n in range(1, self.max_N + 1):
            factors = factorint(n)
            if len(factors) == 1:
                sample_ns.append(n)
                p, k = list(factors.items())[0]
                base_weights.append(np.log(p))
        self.base_weights = np.array(base_weights)
        self.sample_ns = np.array(sample_ns)

    def get_summands(self, s):
        exp_const = self.exp_const_tracker.get_value()
        weights = self.base_weights * np.exp(-exp_const * self.sample_ns)
        return weights * (self.sample_ns**(-s))

