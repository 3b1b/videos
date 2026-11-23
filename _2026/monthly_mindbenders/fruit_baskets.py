from manim_imports_ext import *
from scipy.spatial.transform import Slerp
from scipy.optimize import linear_sum_assignment


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

    def place_in_basket(self, basket):
        self.arrange_in_mound(max_width=basket[1].get_width() * 0.85)
        self.move_to(
            basket[0].get_top() + DOWN * 1.4 * np.mean([fruit.get_height() for fruit in self]),
            aligned_edge=DOWN
        )
        self.match_x(basket[1])
        self.set_submobjects(sorted(self.submobjects, key=lambda fruit: -fruit.get_y()))
        for rank, fruit in enumerate(self):
            fruit.set_z_index(1 + 2 * (rank + 1) / (len(self) + 1))
        return self


def get_best_selection(counts, n_selected, n_restarts=20, max_iterations=1000):
    counts = np.array(counts, dtype=float)
    proportions = counts / counts.sum(axis=0)
    n_total = len(counts)
    best_selection = None
    best_score = None
    for _ in range(n_restarts):
        selection = np.zeros(n_total, dtype=bool)
        selection[random.sample(range(n_total), n_selected)] = True
        current = proportions[selection].sum(axis=0)
        for _ in range(max_iterations):
            selected_indices = np.where(selection)[0]
            unselected_indices = np.where(~selection)[0]
            swapped = (
                current[None, None, :]
                - proportions[selected_indices][:, None, :]
                + proportions[unselected_indices][None, :, :]
            )
            scores = swapped.min(axis=2) + 1e-6 * swapped.sum(axis=2)
            out_index, in_index = np.unravel_index(np.argmax(scores), scores.shape)
            if scores[out_index, in_index] <= current.min() + 1e-6 * current.sum() + 1e-12:
                break
            selection[selected_indices[out_index]] = False
            selection[unselected_indices[in_index]] = True
            current = swapped[out_index, in_index]
        score = current.min() + 1e-6 * current.sum()
        if best_score is None or score > best_score:
            best_score = score
            best_selection = selection.copy()
    return set(np.where(best_selection)[0].tolist())


BASKET_SPOTS = {
    1: (-0.25, 0.25),
    2: (0, -0.2),
    3: (-0.2, -0.27),
    4: (0.24, 0.0),
}


def get_basket():
    return Group(
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
    ).set_height(10)


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
        initial_families[3][1].shift(LEFT * 0.15 * initial_families[3][0].get_width())
        self.play(
            AnimationGroup(*[
                FadeIn(family)
                for family in initial_families
            ], lag_ratio=0.3)
        )

        # Zoom out to see all the families
        family_variants = [i + 1 if i < 4 else random.randint(1, 4) for i in range(100)]
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
                        F"family_{family_variants[i]}.png"
                    )
                )
            ).set_height(3.5)
            for i in range(100)
        ]).arrange_in_grid(n_rows=10, n_cols=10).align_to(initial_families, UL)
        for family, variant in zip(families, family_variants):
            if variant == 4:
                family[1].shift(LEFT * 0.15 * family[0].get_width())
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
                basket[0].get_top() + DOWN * 1.4 * np.mean([fruit.get_height() for fruit in bunch]),
                aligned_edge=DOWN
            )
            bunch.target.match_x(basket[1])
            order = sorted(range(len(bunch)), key=lambda i: -bunch.target[i].get_y())
            bunch.set_submobjects([bunch[i] for i in order])
            bunch.target.set_submobjects([bunch.target[i] for i in order])
            for rank, (fruit, target_fruit) in enumerate(zip(bunch, bunch.target)):
                z_index = 1 + 2 * (rank + 1) / (len(bunch) + 1)
                fruit.set_z_index(z_index)
                target_fruit.set_z_index(z_index)
        self.play(
            AnimationGroup(*[
                AnimationGroup(
                    basket[1].animate.set_opacity(1),
                    MoveToTarget(bunch)
                )
                for bunch, basket in zip(bunches, list(baskets[:-2]) + [baskets[-1]])
            ], lag_ratio=0.1)
        )

        # Hand a basket to every family
        representative_baskets = list(zip(list(baskets[:-2]) + [baskets[-1]], bunches))
        family_order = random.sample(range(len(families)), len(families))
        frame = self.camera.frame
        off_screen_radius = 0.5 * np.hypot(frame.get_width(), frame.get_height())

        def get_basket_spot(index):
            family = families[index]
            dx, dy = BASKET_SPOTS[family_variants[index]]
            return family.get_center() + dx * family.get_width() * RIGHT + dy * family.get_height() * UP

        delivery_animations = []
        for (basket, bunch), index in zip(representative_baskets, family_order):
            scale_factor = families[index].get_width() / 3 / basket.get_width()
            center = Group(basket, bunch).get_center()
            offset = get_basket_spot(index) - center
            delivery_animations.append(AnimationGroup(
                basket.animate.scale(scale_factor, about_point=center).shift(offset),
                bunch.animate.scale(scale_factor, about_point=center).shift(offset)
            ))
        new_baskets = Group()
        for index in family_order[len(representative_baskets):]:
            basket = get_basket().match_width(representative_baskets[0][0])
            bunch = FruitBunch().place_in_basket(basket)
            new_basket = Group(basket, bunch).set_opacity(0)
            start = frame.get_center() + rotate_vector(RIGHT, random.uniform(0, TAU)) * (
                off_screen_radius + 0.5 * np.hypot(new_basket.get_width(), new_basket.get_height())
            )
            new_basket.move_to(start)
            new_baskets.add(new_basket)
            delivery_animations.append(
                new_basket.animate.scale(
                    families[index].get_width() / 3 / basket.get_width()
                ).move_to(
                    get_basket_spot(index)
                ).set_opacity(1)
            )
        self.add(new_baskets)
        new_deliveries = delivery_animations[len(representative_baskets):]
        random.shuffle(new_deliveries)
        delivery_animations[len(representative_baskets):] = new_deliveries
        self.play(
            AnimationGroup(*delivery_animations, lag_ratio=0.008, run_time=3),
            FadeOut(baskets[-2]),
            families.animate.set_opacity(1)
        )
        baskets = Group(*[mob for pair in representative_baskets for mob in pair], *new_baskets)
        self.wait(1)

        # Abstract the view
        self.camera.light_source.move_to(families).shift(UP * 5 + OUT * 10)
        prisms = Group(*[
            Prism(
                width=families[0].get_width(),
                height=families[0].get_width(),
                depth=0.1
            ).set_color(GREY_D).move_to(family)
            for family in families
        ])
        basket_images = Group(
            *[basket for basket, bunch in representative_baskets],
            *[new_basket[0] for new_basket in new_baskets]
        )

        bunch_families = list(zip(
            [bunch for basket, bunch in representative_baskets] + [new_basket[1] for new_basket in new_baskets],
            family_order
        ))
        cube_size = prisms[0].get_width() / 3
        cube_colors = {"Kumquat": ORANGE, "Pear": GREEN, "Apple": RED}
        stack_order = {"Kumquat": 0, "Pear": 1, "Apple": 2}
        cubes = Group()
        family_cubes = [Group() for family in families]
        cube_transforms = []
        for bunch, index in bunch_families:
            stacked_fruits = sorted(bunch, key=lambda fruit: stack_order[type(fruit).__name__])
            for level, fruit in enumerate(stacked_fruits):
                cube = Cube(side_length=cube_size, color=cube_colors[type(fruit).__name__])
                cube.move_to(families[index].get_center() + OUT * (0.05 + (level + 0.5) * cube_size))
                cubes.add(cube)
                family_cubes[index].add(cube)
                cube_transforms.append(FadeTransform(fruit, cube))

        self.play(
            self.camera.frame.animate(run_time=2).reorient(-25, 61, 0, (26.84, -24.62, -2.63), 55.06),
            FadeOut(Group(families, basket_images)),
            FadeIn(prisms),
            AnimationGroup(*cube_transforms, run_time=2)
        )

        # Spotlight random picks of 51
        stack_prisms = Group()
        fruit_prisms = {fruit_name: Group() for fruit_name in stack_order}
        for bunch, index in bunch_families:
            family_stack = Group()
            level = 0
            for fruit_name in sorted(stack_order, key=stack_order.get):
                count = sum(1 for fruit in bunch if type(fruit).__name__ == fruit_name)
                if count == 0:
                    continue
                fruit_prism = Prism(
                    width=cube_size,
                    height=cube_size,
                    depth=count * cube_size,
                    color=cube_colors[fruit_name]
                ).move_to(families[index].get_center() + OUT * (0.05 + (level + 0.5 * count) * cube_size))
                family_stack.add(fruit_prism)
                fruit_prisms[fruit_name].add(fruit_prism)
                level += count
            family_cubes[index] = family_stack
            stack_prisms.add(family_stack)
        self.remove(cubes)
        self.add(stack_prisms)
        camera_location = frame.get_implied_camera_location()
        highlights = [
            Square(side_length=prism.get_width())
            .set_stroke(YELLOW, 2)
            .set_fill(opacity=0)
            .move_to(prism.get_center() + OUT * 0.06)
            .apply_depth_test()
            for prism in prisms
        ]
        self.set_camera_target_position(19, 33, 0, (23.79, -20.06, -0.40), 73.47, drift_time=4)
        family_fruit_counts = [[0] * len(stack_order) for family in families]
        for bunch, index in bunch_families:
            for fruit in bunch:
                family_fruit_counts[index][stack_order[type(fruit).__name__]] += 1
        final_selection = get_best_selection(family_fruit_counts, 51)
        for round_index in range(10):
            if round_index < 9:
                selected = set(random.sample(range(len(families)), 51))
            else:
                selected = final_selection
            for index in range(len(families)):
                opacity = 1 if index in selected else 0.1
                prisms[index].set_opacity(opacity)
                family_cubes[index].set_opacity(opacity)
            dimmed = sorted(
                (index for index in range(len(families)) if index not in selected),
                key=lambda index: -get_norm(prisms[index].get_center() - camera_location)
            )
            self.remove(*highlights)
            self.add(
                *[mob for index in selected for mob in (prisms[index], family_cubes[index])],
                *[highlights[index] for index in selected],
                *[mob for index in dimmed for mob in (prisms[index], family_cubes[index])]
            )
            self.wait(0.5)

        # Show the comparison
        def get_cluster_offsets(n, spacing):
            row_height = spacing * np.sqrt(3) / 2
            extent = int(np.ceil(np.sqrt(n))) + 2
            points = [
                np.array([(col + 0.5 * (row % 2)) * spacing, row * row_height, 0])
                for row in range(-extent, extent + 1)
                for col in range(-extent, extent + 1)
            ]
            points = sorted(points, key=get_norm)[:n]
            centroid = np.mean(points, axis=0)
            return [point - centroid for point in points]

        def assign_to_cluster(indices, center, offsets):
            targets = [center + offset for offset in offsets]
            costs = np.array([
                [get_norm(prisms[index].get_center() - target) for target in targets]
                for index in indices
            ])
            rows, cols = linear_sum_assignment(costs)
            return {indices[row]: targets[col] for row, col in zip(rows, cols)}

        def order_for_transparency(opaque, translucent):
            eye = frame.get_implied_camera_location()
            self.add(*opaque, *sorted(translucent, key=lambda mob: -get_norm(mob.get_center() - eye)))

        tile_width = prisms[0].get_width()
        spacing = 1.6 * tile_width
        camera_right = frame.get_orientation().as_matrix()[:, 0].copy()
        camera_right[2] = 0
        camera_right /= get_norm(camera_right)
        grid_center = prisms.get_center()
        grid_center[2] = 0
        selected_indices = list(selected)
        unselected_indices = list(dimmed)
        left_offsets = get_cluster_offsets(len(selected_indices), spacing)
        right_offsets = get_cluster_offsets(len(unselected_indices), spacing)
        left_radius = max(get_norm(offset) for offset in left_offsets) + tile_width * (1 / np.sqrt(2) + 0.3)
        right_radius = max(get_norm(offset) for offset in right_offsets) + tile_width * (1 / np.sqrt(2) + 0.3)
        cluster_gap = 2.5 * tile_width
        left_center = grid_center - camera_right * (left_radius + 0.5 * cluster_gap)
        right_center = grid_center + camera_right * (right_radius + 0.5 * cluster_gap)
        left_targets = assign_to_cluster(selected_indices, left_center, left_offsets)
        right_targets = assign_to_cluster(unselected_indices, right_center, right_offsets)

        selection_circle = Circle(radius=left_radius).set_stroke(YELLOW, 5)
        selection_circle.move_to(left_center + OUT * 0.06).apply_depth_test()
        move_animations = []
        for index, target in left_targets.items():
            shift = target - prisms[index].get_center()
            move_animations += [
                prisms[index].animate.shift(shift),
                family_cubes[index].animate.shift(shift),
                FadeOut(highlights[index], shift=shift)
            ]
        for index, target in right_targets.items():
            shift = target - prisms[index].get_center()
            move_animations += [
                prisms[index].animate.shift(shift),
                family_cubes[index].animate.shift(shift)
            ]
        self.add(
            *[mob for index in selected_indices for mob in (prisms[index], family_cubes[index])],
            *[highlights[index] for index in selected_indices],
            selection_circle,
            *[mob for index in unselected_indices for mob in (prisms[index], family_cubes[index])]
        )
        self.play(
            self.camera.frame.animate.reorient(21, 87, 0, (21.08, -16.44, 13.06), 79.01),
            *move_animations,
            ShowCreation(selection_circle, time_span=(1, 2)),
            run_time=2)
        unselected_mobs = [mob for index in unselected_indices for mob in (prisms[index], family_cubes[index])]
        order_for_transparency(
            [*[mob for index in selected_indices for mob in (prisms[index], family_cubes[index])], selection_circle],
            unselected_mobs
        )
        self.play(*[mob.animate.set_opacity(1) for mob in unselected_mobs])
        self.wait()

        # Spotlight one fruit at a time
        fruit_spotlights = [
            ("Apple", ["Pear", "Kumquat"]),
            ("Pear", ["Apple", "Kumquat"]),
            ("Kumquat", ["Apple", "Pear"]),
        ]
        comparison_symbol = Tex(R"\ge", font_size=70).fix_in_frame()
        for i, (shown_name, faded_names) in enumerate(fruit_spotlights):
            order_for_transparency(
                [*prisms, selection_circle, *fruit_prisms[shown_name]],
                [prism for name in faded_names for prism in fruit_prisms[name]]
            )
            self.play(
                FadeIn(comparison_symbol, time_span=(0.3, 1)) if i == 0 else comparison_symbol.animate.shift(DOWN * 0.5),
                fruit_prisms[shown_name].animate.set_opacity(1),
                *[fruit_prisms[name].animate.set_opacity(0.05) for name in faded_names]
            )
            self.wait()

    def set_camera_target_position(
        self,
        theta_degrees=None,
        phi_degrees=None,
        gamma_degrees=None,
        center=None,
        height=None,
        drift_time=2.0,
    ):
        frame = self.camera.frame
        self.add(frame)
        frame.clear_updaters()
        initial_orientation = frame.get_orientation()
        initial_height = frame.get_height()
        initial_eye = frame.get_implied_camera_location()
        target_frame = frame.copy()
        target_frame.reorient(theta_degrees, phi_degrees, gamma_degrees, center, height)
        target_orientation = target_frame.get_orientation()
        target_height = target_frame.get_height()
        target_eye = target_frame.get_implied_camera_location()
        fovy = frame.get_field_of_view()
        slerp = Slerp([0, 1], Rotation.concatenate([initial_orientation, target_orientation]))
        drift_time = max(drift_time, 1e-4)
        elapsed = 0.0

        def update_camera(f, dt):
            nonlocal elapsed
            elapsed += dt
            t = min(elapsed / drift_time, 1.0)
            alpha = smooth(t)
            current_orientation = slerp(alpha)
            current_height = interpolate(initial_height, target_height, alpha)
            current_eye = interpolate(initial_eye, target_eye, alpha)
            focal_distance = 0.5 * current_height / np.tan(0.5 * fovy)
            to_camera = current_orientation.as_matrix().T[2]
            current_center = current_eye - focal_distance * to_camera
            f.set_orientation(current_orientation)
            f.move_to(current_center)
            f.set_height(current_height)
            if t >= 1.0:
                f.remove_updater(update_camera)
        frame.add_updater(update_camera)
