from manim_imports_ext import *


CLAIRE_ASSETS_DIRECTORY = os.path.join(
    os.path.dirname(manim_config.file_writer.output_directory),
    "Claire_Fruit_Basket_Assets"
)


class Apple(ImageMobject):
    def __init__(self):
        super().__init__(
            os.path.join(
                CLAIRE_ASSETS_DIRECTORY,
                F"apple_{random.randint(1, 2)}.png"
            )
        )
        self.set_height(2).set_z_index(1)


class Pear(ImageMobject):
    def __init__(self):
        super().__init__(
            os.path.join(
                CLAIRE_ASSETS_DIRECTORY,
                F"pear_{random.randint(1, 2)}.png"
            )
        )
        self.set_height(2).set_z_index(1)


class Kumquat(ImageMobject):
    def __init__(self):
        super().__init__(
            os.path.join(
                CLAIRE_ASSETS_DIRECTORY,
                F"kumquat_{random.randint(1, 2)}.png"
            )
        )
        self.set_height(2).set_z_index(1)


class FruitBunch(Group):
    def __init__(
        self,
        n_fruits=12,
        n_fruits_variation=3,
        fruit_height=2,
        size_variation=0.08,
        max_angle=12 * DEGREES,
        aspect_ratio_range=(1.2, 1.4),
        packing_density=0.4,
        max_attempts=5000,
    ):
        super().__init__()
        n_fruits = n_fruits + random.randint(-n_fruits_variation, n_fruits_variation)
        radius = 0.5 * fruit_height
        aspect_ratio = random.uniform(*aspect_ratio_range)
        a = radius * np.sqrt(n_fruits / (packing_density * aspect_ratio))
        b = aspect_ratio * a
        harmonics = [
            (k, random.uniform(-0.08, 0.08), random.uniform(0, TAU))
            for k in range(2, 5)
        ]

        def boundary_scale(theta):
            return 1 + sum(
                amplitude * np.cos(k * theta + phase)
                for k, amplitude, phase in harmonics
            )

        placed = []
        attempts = 0
        while len(placed) < n_fruits and attempts < max_attempts:
            attempts += 1
            x = random.uniform(-1.3, 1.3) * a
            y = random.uniform(-1.3, 1.3) * b
            theta = np.arctan2(y / b, x / a)
            if (x / a) ** 2 + (y / b) ** 2 > boundary_scale(theta) ** 2:
                continue
            r = radius * random.uniform(1 - size_variation, 1 + size_variation)
            if any(np.hypot(x - px, y - py) < r + pr for px, py, pr in placed):
                continue
            placed.append((x, y, r))

        for x, y, r in placed:
            fruit = random.choice([Apple, Pear, Kumquat])()
            fruit.set_height(2 * r)
            fruit.rotate(random.uniform(-max_angle, max_angle))
            fruit.move_to([x, y, 0])
            self.add(fruit)

        if placed:
            self.center()

    def arrange_in_mound(
        self,
        max_width=None,
        horizontal_spacing=0.7,
        vertical_spacing=0.4,
        dome_ratio=0.45,
        min_row_ratio=0.35,
        jitter=0.08,
    ):
        fruits = list(self.submobjects)
        if not fruits:
            return self
        random.shuffle(fruits)
        avg_width = np.mean([fruit.get_width() for fruit in fruits])
        avg_height = np.mean([fruit.get_height() for fruit in fruits])
        dx = horizontal_spacing * avg_width
        dy = vertical_spacing * avg_height
        if max_width is None:
            max_width = avg_width + dx * np.ceil(np.sqrt(2 * len(fruits)))
        half_span = max(0.5 * (max_width - avg_width), 0)
        dome_height = dome_ratio * max_width

        positions = []
        remaining = len(fruits)
        row = 0
        while remaining > 0:
            y = row * dy
            row_ratio = np.sqrt(max(1 - (y / dome_height) ** 2, 0))
            row_half = half_span * max(row_ratio, min_row_ratio)
            n = min(int(2 * row_half // dx) + 1, remaining)
            used_half = min(row_half, 0.5 * (n - 1) * dx)
            xs = np.linspace(-used_half, used_half, n) if n > 1 else [0]
            for x in xs:
                positions.append((
                    x + random.uniform(-jitter, jitter) * avg_width,
                    y + random.uniform(-jitter, jitter) * avg_height,
                ))
            remaining -= n
            row += 1

        for fruit, (x, y) in zip(fruits, positions):
            fruit.move_to([x, y, 0])
        return self


class FruitBaskets(InteractiveScene):
    def construct(self):
        # Add the initial families
        margin = 2
        x = 4
        y = 2
        family_positions = [
            (-x + 0.7, y - 0.1, 0),
            (x - 0.9, y + 0.2, 0),
            (-x, -y + 0.1, 0),
            (x - 0.4, -y - 0.1, 0)
        ]
        initial_families = Group(*[
            Group(
                ImageMobject(
                    os.path.join(
                        CLAIRE_ASSETS_DIRECTORY,
                        F"blanket_{i}.png"
                    )
                ),
                ImageMobject(
                    os.path.join(
                        CLAIRE_ASSETS_DIRECTORY,
                        F"family_{i}.png"
                    )
                )
            ).set_height(3.5).move_to(family_positions[i - 1])
            for i in range(1, 5)
        ])
        self.play(
            AnimationGroup(*[
                FadeIn(family)
                for family in initial_families
            ], lag_ratio=0.3)
        )

        # Zoom out to see all the families
        families = Group(*[
            Group(
                ImageMobject(
                    os.path.join(
                        CLAIRE_ASSETS_DIRECTORY,
                        F"blanket_{i + 1 if i < 4 else random.randint(1, 4)}.png"
                    )
                ),
                ImageMobject(
                    os.path.join(
                        CLAIRE_ASSETS_DIRECTORY,
                        F"family_{i + 1 if i < 4 else random.randint(1, 4)}.png"
                    )
                )
            ).set_height(3.5)
            for i in range(100)
        ]).arrange_in_grid(n_rows=10, n_cols=10).align_to(initial_families, UL)
        rest_of_families = list(families[4:])
        random.shuffle(rest_of_families)
        self.play(
            self.camera.frame.animate.move_to(families).set_height(families.get_height() * 1.1),
            AnimationGroup(
                *[
                    ReplacementTransform(initial_family, new_family)
                    for initial_family, new_family in zip(initial_families, families[:4])
                ],
                *[
                    FadeIn(family)
                    for family in rest_of_families
                ],
                lag_ratio=0.01,
                run_time=3
            )
        )

        # Show some representative baskets
        num_representative_baskets = 5
        baskets = Group(*[
            Group(
                ImageMobject(
                    os.path.join(
                        CLAIRE_ASSETS_DIRECTORY,
                        "basket_back.png"
                    )
                ).set_z_index(1),
                ImageMobject(
                    os.path.join(
                        CLAIRE_ASSETS_DIRECTORY,
                        "basket_front.png"
                    )
                ).shift(DOWN * 0.2 + RIGHT * 0.2).set_z_index(3)
            )
            if i != num_representative_baskets - 1 else
            Tex(R"\cdots", font_size=400)
            for i in range(num_representative_baskets + 1)
        ]).set_height(
            10
        ).arrange(
            buff=1
        ).match_y(
            families
        ).align_to(
            families, LEFT
        ).shift(
            LEFT * 10
        )
        self.play(
            families.animate.set_opacity(0.2),
            AnimationGroup(*[FadeIn(basket) for basket in baskets], lag_ratio=0.1)
        )

        # Fill the baskets with fruits
        bunches = Group(*[
            FruitBunch().next_to(basket, UP, buff=2)
            for basket in baskets if not isinstance(basket, Tex)
        ])
        for fruit_type in [Apple, Pear, Kumquat]:
            fruits = [
                fruit
                for bunch in bunches
                for fruit in bunch
                if isinstance(fruit, fruit_type)
            ]
            self.play(
                AnimationGroup(
                    *[FadeIn(fruit) for fruit in fruits],
                )
            )
        for bunch, basket in zip(bunches, list(baskets[:-2]) + [baskets[-1]]):
            bunch.generate_target()
            bunch.target.arrange_in_mound(max_width=basket[1].get_width() * 0.85)
            bunch.target.move_to(
                basket[0].get_top() + DOWN * 1.3 * np.mean([fruit.get_height() for fruit in bunch]),
                aligned_edge=DOWN
            )
            bunch.target.match_x(basket[1])
            order = sorted(range(len(bunch)), key=lambda i: -bunch.target[i].get_y())
            bunch.set_submobjects([bunch[i] for i in order])
            bunch.target.set_submobjects([bunch.target[i] for i in order])
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    basket[1].animate.set_opacity(1),
                    MoveToTarget(bunch)
                )
                for bunch, basket in zip(bunches, list(baskets[:-2]) + [baskets[-1]])
            ], lag_ratio=0.1)
        )
