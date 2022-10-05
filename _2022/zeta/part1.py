from manim_imports_ext import *
import sympy


PRIME_COLOR = YELLOW


def get_primes(max_n=100):
    pass


def get_prime_colors():
    pass


# Scenes

class Intro(InteractiveScene):
    def construct(self):
        pass


class ShowPrimeDensity(InteractiveScene):
    def construct(self):
        # Setup
        frame = self.camera.frame

        x_max = 150
        numberline = NumberLine((0, x_max), width=0.7 * x_max)
        numberline.to_edge(LEFT)
        numberline.add_numbers()

        def get_number_animation(n):
            if sympy.isprime(n):
                return self.get_prime_animation(numberline, n)
            else:
                return Animation(Mobject(), remover=True)

        self.add(numberline)

        # Show the first 100 primes
        all_prime_animations = []
        soft_smooth = bezier([0, 0, 1, 1])
        for n in range(x_max):
            if sympy.isprime(n):
                t = smooth()
                all_prime_animations.append(self.get_prime_animation(
                    numberline, n,
                    run_time=11,
                    time_span=(t, t + 1)
                ))


        frame.generate_target()
        frame.target.move_to(numberline.n2p(100))
        frame.target.scale(2)

        self.play(
            LaggedStart(*all_prime_animations, lag_ratio=0.25),
            MoveToTarget(
                frame,
                rate_func=smooth,
                time_span=(2, 10),
            ),
            run_time=10,
        )

        # Zoom out to 1 million

        # Zoom out to 1 trillion

        # Mention logarithm rule

        # Ask about log rule

    def get_prime_animation(self, numberline, prime, **kwargs):
        numberline.get_unit_size()
        point = numberline.n2p(prime)
        dot = GlowDot(point, color=PRIME_COLOR)
        dot.set_glow_factor(0)
        dot.set_radius(0.5)
        dot.set_opacity(0)
        dot.generate_target()
        dot.target.set_glow_factor(2)
        dot.target.set_opacity(1)
        dot.target.set_radius(DEFAULT_GLOW_DOT_RADIUS)

        arrow = Vector(DOWN, color=PRIME_COLOR)
        arrow.move_to(point, DOWN)

        return AnimationGroup(
            ShowCreation(arrow),
            MoveToTarget(dot, rate_func=rush_into),
            numberline.numbers[prime].animate.set_color(PRIME_COLOR),
            **kwargs
        )


class ShowLogarithmicWeighting(InteractiveScene):
    def construct(self):
        pass