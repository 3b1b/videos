from manim_imports_ext import *
import random

SPRITES_DIRECTORY = os.path.join(
    os.path.dirname(manim_config.file_writer.output_directory),
    "Mitchell-Animations", "Manim Pixel Art v02"
)


class TileFlip(Animation):
    def __init__(self, tile, axis=RIGHT, depth_margin=0.2, **kwargs):
        self.axis = normalize(np.array(axis, dtype=float))
        self.depth_margin = depth_margin
        super().__init__(tile, **kwargs)

    def begin(self):
        self.center = self.mobject.get_center().copy()
        half_extent = max(self.mobject.get_width(), self.mobject.get_height()) / 2
        self.depth = half_extent * (1 + self.depth_margin)
        super().begin()

    def interpolate_mobject(self, alpha):
        angle = PI * self.rate_func(alpha)

        pairs = zip(
            self.mobject.family_members_with_points(),
            self.starting_mobject.family_members_with_points()
        )
        for sm1, sm2 in pairs:
            for key in sm1.pointlike_data_keys:
                sm1.data[key][:] = sm2.data[key]

        self.mobject.rotate(angle, axis=self.axis, about_point=self.center)
        self.mobject.shift(IN * self.depth * np.sin(angle))


class Tile(Group):
    def __init__(self, parity=True, finish_line=False, has_monster=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finish_line = finish_line
        self.has_monster = has_monster

        self.top = TexturedSurface(
            Square3D(side_length=1),
            os.path.join(SPRITES_DIRECTORY, "MTilv2-1.png" if parity else "MTilv2-3.png"),
            texture_filter="nearest"
        )
        self.bot = TexturedSurface(
            Square3D(side_length=1),
            os.path.join(SPRITES_DIRECTORY, "MTilv2-4.png" if has_monster else "MTilv2-5.png" if finish_line else "MTilv2-2.png"),
            texture_filter="nearest"
        )
        self.add(self.top, self.bot)
        self.arrange(IN, buff=0.001)
        self.set_shading(0, 0, 0)

    def reveal(self, axis=RIGHT, run_time=0.5, **kwargs):
        if self.top.get_z() > self.bot.get_z():
            return TileFlip(self, axis=axis, run_time=run_time, **kwargs)
        else:
            return Animation(Mobject(), run_time=run_time, **kwargs)

    def set_has_monster(self, has_monster):
        if has_monster == self.has_monster:
            return
        self.has_monster = has_monster
        new_bot = TexturedSurface(
            Square3D(side_length=1),
            os.path.join(SPRITES_DIRECTORY, "MTilv2-4.png" if has_monster else "MTilv2-5.png" if self.finish_line else "MTilv2-2.png"),
            texture_filter="nearest"
        )
        new_bot.data['point'][:] = self.bot.data['point']
        new_bot.set_shading(0, 0, 0)
        self.replace_submobject(self.submobjects.index(self.bot), new_bot)
        self.bot = new_bot


class Turbo(Sprite):
    FRAME_DURATION = 0.1
    FRAMES_PER_ANIMATION = 8
    ANIMATION_DURATION = FRAMES_PER_ANIMATION * FRAME_DURATION
    LAST_FRAME_OFFSET = (FRAMES_PER_ANIMATION - 1) * FRAME_DURATION

    RIGHT_START = 0 * ANIMATION_DURATION
    DOWN_START = 1 * ANIMATION_DURATION
    LEFT_START = 2 * ANIMATION_DURATION
    UP_START = 3 * ANIMATION_DURATION
    DEATH_START = 4 * ANIMATION_DURATION
    DEATH_END = DEATH_START + LAST_FRAME_OFFSET

    def __init__(self, grid, *args, **kwargs):
        self.grid = grid
        self.current_position = [0, 0]
        self.previous_position = (0, 0)
        super().__init__(
            os.path.join(SPRITES_DIRECTORY, "MTurbv2-Turbo.gif"),
            height=grid.get_tile(0, 0).get_height() * 0.8,
            *args,
            **kwargs
        )
        self.move_to(
            self.grid.get_tile(0, 0)
        ).align_to(
            self.grid.get_tile(0, 0).get_zenith(), IN
        ).shift(
            OUT * 0.02
        )

        self.time_tracker = ValueTracker(self.RIGHT_START)
        # Hacky fix below: if self.time_tracker.add_updater(lambda t: self.set_time(t.get_value())) is used,
        # it causes the tracker to get stuck oscillating between the two keyframe values, since the other .animates
        # create copies of the tracker.
        self.time_tracker.add_updater(lambda _: self.set_time(self.time_tracker.get_value()))

        self.opacity_tracker = ValueTracker(1)
        self.opacity_tracker.add_updater(lambda _: self.set_opacity(self.opacity_tracker.get_value()))
        self.add(self.time_tracker, self.opacity_tracker)

    def move_to_position(self, i, j, **kwargs):
        target_tile = self.grid.get_tile(i, j)
        self.current_position = [i, j]
        return self.animate(**kwargs).match_x(target_tile).match_y(target_tile)

    def move_to_start(self, **kwargs):
        target_tile = self.grid.get_tile(0, 0)
        start_center = self.get_center()
        end_center = target_tile.get_center()
        move_vector = end_center - start_center

        arc_axis = np.cross(move_vector, IN)

        self.time_tracker.set_value(self.RIGHT_START)
        return self.move_to_position(
            0, 0,
            path_arc=PI * 0.8,
            path_arc_axis=arc_axis,
            **kwargs
        )

    def get_neighbor(self, direction):
        i, j = self.current_position
        if (direction == UP).all():
            return (i, j - 1)
        if (direction == RIGHT).all():
            return (i + 1, j)
        if (direction == DOWN).all():
            return (i, j + 1)
        if (direction == LEFT).all():
            return (i - 1, j)
        raise ValueError("Direction of movement must be UP, RIGHT, DOWN, or LEFT")

    def get_direction_start(self, direction):
        if (direction == RIGHT).all():
            return self.RIGHT_START
        if (direction == DOWN).all():
            return self.DOWN_START
        if (direction == LEFT).all():
            return self.LEFT_START
        if (direction == UP).all():
            return self.UP_START
        raise ValueError("Direction of movement must be UP, RIGHT, DOWN, or LEFT")

    def move(self, direction, run_time=0.5):
        i, j = self.current_position
        self.previous_position = (i, j)
        new_position = self.get_neighbor(direction)

        walk_start = self.get_direction_start(direction)
        walk_end = walk_start + self.LAST_FRAME_OFFSET
        self.time_tracker.set_value(walk_start)

        walk_in = AnimationGroup(
            self.move_to_position(*new_position, run_time=run_time),
            self.time_tracker.animate(run_time=run_time).set_value(walk_end)
        )

        if self.grid.is_monster(*new_position):
            return walk_in

        reveal_anim = self.grid.reveal_tile(*new_position, axis=[new_position[1] - j, new_position[0] - i, 0], run_time=run_time)
        return AnimationGroup(walk_in, reveal_anim, lag_ratio=0.3)

    def bounce_and_die(self):
        original_tile = self.grid.get_tile(*self.previous_position)
        current_x, current_y = self.get_x(), self.get_y()
        target_x, target_y = original_tile.get_x(), original_tile.get_y()

        def bounce_update(mob, alpha):
            mob.set_x(interpolate(current_x, target_x, alpha))
            mob.set_y(interpolate(current_y, target_y, alpha))
        bounce_back = UpdateFromAlphaFunc(self, bounce_update, run_time=0.3)

        death_anim = UpdateFromAlphaFunc(
            self.time_tracker,
            lambda mob, alpha: mob.set_value(interpolate(self.DEATH_START, self.DEATH_END, alpha)),
            run_time=0.6
        )
        return AnimationGroup(bounce_back, death_anim)


class Monster(Sprite):
    FRAME_DURATION = 0.3
    BOB_START = 0 * FRAME_DURATION
    BOB_END = 1 * FRAME_DURATION
    X_START = 2 * FRAME_DURATION
    BURROW_START = 3 * FRAME_DURATION

    def __init__(self, grid, i, j, *args, **kwargs):
        self.grid = grid
        super().__init__(
            os.path.join(SPRITES_DIRECTORY, "MMonv2-Monster.gif"),
            height=self.grid.get_tile(i, j).get_height() * 0.8,
            *args,
            **kwargs
        )
        self.move_to(
            self.grid.get_tile(i, j)
        ).align_to(
            self.grid.get_tile(i, j).get_zenith(), IN
        ).shift(
            OUT * 0.04
        )
        self.set_time(self.BOB_START)


class TurboGrid(Group):
    def __init__(self, n, monster_positions=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.monster_positions = monster_positions

        self.tiles = Group(*[
            Tile(
                parity=((i % (n - 1)) + (i // (n - 1))) % 2,
                finish_line=i // (n - 1) == n - 1,
                has_monster=(i % (n - 1), i // (n - 1)) in self.monster_positions
            )
            for i in range(n * (n - 1))
        ]).arrange_in_grid(
            n_rows=n, n_cols=n - 1, buff=0
        ).set_z_index(0)
        self.get_row(0).flip(axis=RIGHT)
        self.get_row(self.n - 1).flip(axis=RIGHT)
        self.add(self.tiles)

        self.turbo = Turbo(self).set_z_index(100)
        self.monsters = Group(*[
            Monster(self, i, j)
            for (i, j) in self.monster_positions
        ]).set_z_index(200)
        self.add(self.turbo, self.monsters)

    def create(self, run_time=None):
        tiles_sorted_from_center = sorted(self.tiles, key=lambda t: np.linalg.norm(t.get_center() - self.tiles.get_center()))
        kwargs = {} if run_time is None else {"run_time": run_time}
        return AnimationGroup(
            *[
                FadeIn(tile, shift=IN * 0.3)
                for tile in tiles_sorted_from_center
            ],
            self.turbo.shift(OUT * 0.3).animate.shift(IN * 0.3),
            self.turbo.opacity_tracker.set_value(0).animate.set_value(1),
            FadeIn(self.monsters), lag_ratio=0.1, **kwargs)

    def get_tile(self, i, j):
        if i < 0:
            raise IndexError("Tile column index is negative")
        if j < 0:
            raise IndexError("Tile row index is negative")
        if i >= self.n - 1:
            raise IndexError("Tile column index is greater than the number of columns")
        if j >= self.n:
            raise IndexError("Tile row index is greater than the number of rows")
        return self.tiles[i + j * (self.n - 1)]

    def reveal_tile(self, i, j, axis=RIGHT, run_time=0.5):
        return self.get_tile(i, j).reveal(axis=axis, run_time=run_time)

    def get_col(self, i):
        return Group(*[self.get_tile(i, j) for j in range(self.n)])

    def get_row(self, j):
        return Group(*[self.get_tile(i, j) for i in range(self.n - 1)])

    def is_monster(self, i, j):
        return (i, j) in self.monster_positions

    def get_monster(self, i, j):
        for monster, pos in zip(self.monsters, self.monster_positions):
            if pos == (i, j):
                return monster
        raise LookupError(F"No monster at position ({i}, {j})")

    def sync_tile_monster_flags(self):
        for i in range(self.n - 1):
            for j in range(self.n):
                self.get_tile(i, j).set_has_monster((i, j) in self.monster_positions)

    def reveal_monster(self, i, j, run_time=3, reveal_free_alleys=True):
        monster = self.get_monster(i, j)
        monster_tile = self.get_tile(i, j)

        def dist_to_monster_tile(tile):
            return np.linalg.norm(tile.get_center() - monster_tile.get_center())
        monster_col = sorted(self.get_col(i), key=dist_to_monster_tile)
        monster_row = sorted(self.get_row(j), key=dist_to_monster_tile)
        monster_col.remove(monster_tile)
        monster_row.remove(monster_tile)
        monster.set_opacity(1).set_time(monster.BOB_END)
        anims = [monster_tile.reveal()]
        if reveal_free_alleys:
            anims.append(
                AnimationGroup(*[
                    t.reveal()
                    for t in monster_row
                ], lag_ratio=0.1),
                AnimationGroup(*[
                    t.reveal()
                    for t in monster_col
                ], lag_ratio=0.1)
            )
        return AnimationGroup(*anims, lag_ratio=0.2)


class TurboController:
    def __init__(self, scene):
        self.scene = scene
        self.n = scene.grid.n
        self.move_speed = 1
        self.reveal_free_alleys = True

    @property
    def position(self):
        return tuple(self.scene.turbo.current_position)

    @property
    def col(self):
        return self.position[0]

    @property
    def row(self):
        return self.position[1]

    def move(self, direction):
        target = self.scene.turbo.get_neighbor(direction)
        if not self.scene.move_turbo(direction, 0.5 / self.move_speed, reveal_free_alleys=self.reveal_free_alleys):
            self.last_monster_pos = target
            return False
        return True

    def move_to_col(self, col):
        direction = RIGHT if col > self.col else LEFT
        for _ in range(abs(col - self.col)):
            if not self.move(direction):
                return False
        return True

    def move_to_row(self, row):
        direction = DOWN if row > self.row else UP
        for _ in range(abs(row - self.row)):
            if not self.move(direction):
                return False
        return True

    def try_col(self, col):
        self.move_to_col(col)
        return self.move_to_row(self.n - 1)


def get_random_monster_positions(n, hardcoded_monsters=[], free_column=None):
    monster_positions = []
    remaining_rows = set(range(1, n - 1))
    remaining_columns = set(range(n - 1))
    for (i, j) in hardcoded_monsters:
        monster_positions.append((i, j))
        remaining_columns.remove(i)
        remaining_rows.remove(j)
    if free_column is None:
        free_column = random.choice(list(remaining_columns))
    remaining_columns.remove(free_column)
    for j in remaining_rows:
        i = random.choice(list(remaining_columns))
        monster_positions.append((i, j))
        remaining_columns.remove(i)
    return monster_positions


def get_monster_staircase(n):
    return [(i, i + 1) for i in range(n - 2)]


def get_monster_staircase_inverted(n):
    return [(i, n - (i + 2)) for i in range(n - 2)]


class TurboScene(InteractiveScene):
    def __init__(self, n, monster_positions, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.grid = TurboGrid(n, monster_positions)
        self.grid.set_height(FRAME_HEIGHT * 0.9)
        self.turbo = self.grid.turbo

    def move_turbo(self, direction, *args, reveal_free_alleys=True, **kwargs):
        self.play(self.turbo.move(direction, *args, **kwargs))
        position = tuple(self.turbo.current_position)
        if not self.grid.is_monster(*position):
            return True
        monster = self.grid.get_monster(*position)
        self.play(
            AnimationGroup(
                self.turbo.bounce_and_die(),
                self.grid.reveal_monster(*position, reveal_free_alleys=reveal_free_alleys)
            )
        )
        monster.set_time(monster.X_START)
        self.play(self.turbo.move_to_start())
        return False


class TurboTest(TurboScene, ThreeDScene):
    def __init__(self, *args, **kwargs):
        n = 6
        super().__init__(n, get_monster_staircase(n), *args, **kwargs)

    def construct(self):
        # Set the camera
        self.camera.frame.reorient(26, 58, 0, (-0.19, -0.72, -0.82), 8.21)

        # Add the grid
        grid, turbo = self.grid, self.turbo
        self.play(
            self.camera.frame.animate.reorient(-8, 38, 0, (-0.56, -0.79, -0.65), 9.93),
            grid.create(), run_time=4)

        # Label the number of rows with "N"
        brace = Brace(grid.tiles, LEFT, buff=0.4)
        label = brace.get_tex("N", font_size=100)
        self.play(
            Succession(
                AnimationGroup(
                    GrowFromEdge(brace, RIGHT),
                    Write(label), run_time=1),
                Animation(VMobject(), run_time=4),
                FadeOut(VGroup(brace, label), run_time=1)
            ),
            self.camera.frame.animate(run_time=30, rate_func=there_and_back).reorient(13, 47, 0, (0.82, -0.71, -0.92), 9.93)
        )

        # Reset the camera to an overhead position
        self.play(self.camera.frame.animate.reorient(0, 0, 0, (0, 0, 0), 7), run_time=2)

        # Move turbo
        moves = [RIGHT, RIGHT, RIGHT, DOWN, DOWN, DOWN]
        for direction in moves:
            self.move_turbo(direction)


class ProblemStatementPart1(InteractiveScene):
    def construct(self):
        # Write the problem statement
        raw_text = R"""
            Turbo the snail plays a game on a board \\
            with $2024$ rows and $2023$ columns.\hfill\break

            There are hidden monsters in $2022$ of \\
            the cells. Initially, Turbo does not \\
            know where any of the monsters are, but \\
            he knows that there is exactly one \\
            monster in each row except the first \\
            row and the last row, and that each \\
            column contains at most one monster.
        """

        problem_statement = TexText(raw_text, alignment="\\raggedright").set_height(FRAME_HEIGHT * 0.5).set_y(0).to_edge(LEFT, buff=1).set_z_index(1)

        # Split into lines by clustering glyphs on their y-coordinate
        glyphs = sorted(problem_statement.submobjects, key=lambda m: -m.get_y())
        tol = 0.5 * np.median([g.get_height() for g in glyphs])

        line_groups = [[glyphs[0]]]
        for glyph in glyphs[1:]:
            line_y = np.mean([g.get_y() for g in line_groups[-1]])
            if abs(glyph.get_y() - line_y) < tol:
                line_groups[-1].append(glyph)
            else:
                line_groups.append([glyph])

        lines = VGroup(
            VGroup(*sorted(group, key=lambda m: m.get_x()))
            for group in line_groups
        )

        self.play(AnimationGroup(*[FadeIn(line, shift=DOWN * 0.3) for line in lines], lag_ratio=0.1), run_time=2)

        # Highlight the number of columns and monsters
        rect1 = SurroundingRectangle(
            problem_statement["$2023$ columns."],
            buff=0.08,
            stroke_width=0
        ).set_opacity(0.8).round_corners(0.07).set_color(TEAL_E).set_z_index(0)
        self.play(
            DrawBorderThenFill(rect1, run_time=1.5),
            problem_statement["$2023$ columns."].animate(lag_ratio=0.3).set_color(YELLOW)
        )

        rect2 = SurroundingRectangle(
            problem_statement[66:88],
            buff=0.08,
            stroke_width=0
        ).set_opacity(0.8).round_corners(0.07).set_color(TEAL_E).set_z_index(0)
        rect3 = SurroundingRectangle(
            problem_statement[88:97],
            buff=0.08,
            stroke_width=0
        ).set_opacity(0.8).round_corners(0.07).set_color(TEAL_E).set_z_index(0)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    DrawBorderThenFill(rect2, run_time=1.5),
                    problem_statement[66:88].animate(lag_ratio=0.3).set_color(YELLOW)
                ),
                AnimationGroup(
                    DrawBorderThenFill(rect3, run_time=1.5),
                    problem_statement[88:97].animate(lag_ratio=0.3).set_color(YELLOW)
                ),
                lag_ratio=0.1
            )
        )
        self.wait(1)

        # Highlight the fact that each column has at most one monster
        rect4 = SurroundingRectangle(
            problem_statement[233:237],
            buff=0.08,
            stroke_width=0
        ).set_opacity(0.8).round_corners(0.07).set_color(TEAL_E).set_z_index(0)
        rect5 = SurroundingRectangle(
            problem_statement[237:268],
            buff=0.08,
            stroke_width=0
        ).set_opacity(0.8).round_corners(0.07).set_color(TEAL_E).set_z_index(0)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    DrawBorderThenFill(rect4, run_time=1.5),
                    problem_statement[233:237].animate(lag_ratio=0.3).set_color(YELLOW)
                ),
                AnimationGroup(
                    DrawBorderThenFill(rect5, run_time=1.5),
                    problem_statement[237:268].animate(lag_ratio=0.3).set_color(YELLOW)
                ),
                lag_ratio=0.1
            )
        )


class BruteForce(TurboScene):
    def __init__(self, *args, **kwargs):
        n = 15
        random.seed(3)
        super().__init__(n, get_random_monster_positions(n, hardcoded_monsters=[(0, 2)], free_column=10), *args, **kwargs)

    def construct(self):
        # Set the camera
        self.camera.frame.save_state()
        self.camera.frame.reorient(
            -23, 53, 0,
            self.grid.get_center(),
            self.grid.get_height() * 0.6
        )

        # Add the grid
        brace = Brace(self.grid, UP)
        label = brace.get_tex("N - 1")
        self.play(
            AnimationGroup(
                AnimationGroup(
                    self.camera.frame.animate(run_time=3.5).restore().scale(1.1, about_point=self.grid.get_bottom()),
                    self.grid.create(run_time=3.5),
                ),
                AnimationGroup(
                    GrowFromEdge(brace, DOWN, run_time=2),
                    Write(label, run_time=1)
                ),
                lag_ratio=0.6
            )
        )

        # Show the initial positions of the monsters
        num_monsters = len(self.grid.monsters)
        monster_numbers = Group()
        reveal_anims = []
        for i, monster in enumerate(self.grid.monsters):
            number = Tex(
                R"\dots" if i == num_monsters - 2 else
                "N - 2" if i == num_monsters - 1 else
                str(i + 1),
                font_size=40
            ).set_color(
                RED_E
            ).set_stroke(
                width=5, color=BLACK, behind=True
            ).next_to(
                monster, UP, buff=0.275 if i == num_monsters - 2 else 0.15
            )
            monster_numbers.add(number)
            reveal_anims.append(
                AnimationGroup(
                    monster.animate_set_time(monster.BOB_END),
                    FadeIn(number, shift=UP * 0.2)
                )
            )
        self.play(AnimationGroup(*reveal_anims, lag_ratio=0.08))
        self.wait(0.3)

        # Show how each column has at most one monster
        rect = Rectangle(
            width=self.grid.get_col(0).get_width(),
            height=self.grid.get_col(0).get_height(),
            fill_opacity=0.5,
            fill_color=YELLOW,
            stroke_width=0
        ).move_to(
            self.grid.get_col(0)
        ).shift(
            OUT * 0.02
        )
        self.play(FadeIn(rect), run_time=0.6)
        free_columns = set(range(self.grid.n - 1))
        for (i, j) in self.grid.monster_positions:
            free_columns.remove(i)
        free_column = list(free_columns)[0]
        for col in range(1, free_column + 1):
            self.play(rect.animate.match_x(self.grid.get_col(col)), run_time=0.6)

        self.play(
            AnimationGroup(
                AnimationGroup(
                    FadeOut(brace),
                    FadeOut(label),
                    FadeOut(monster_numbers),
                    AnimationGroup(*[
                        monster.animate.set_opacity(0.7)
                        for monster in self.grid.monsters[::-1]
                    ])
                ),
                self.camera.frame.animate(run_time=2).restore()
            ),
        )

        # Execute the strategy
        turbo = TurboController(self)
        turbo.reveal_free_alleys = False
        turbo.move_speed = 10
        n = self.grid.n

        def brute_force():
            for col in range(n - 1):
                if turbo.try_col(col):
                    return
        brute_force()
        self.wait(2)

        # Change the positions of the monsters and show the new free column
        new_positions = get_random_monster_positions(n, free_column=n - 2)
        new_grid = TurboGrid(n, new_positions)
        new_grid.match_height(self.grid).move_to(self.grid)
        self.remove(self.grid, rect)
        self.add(new_grid)
        self.wait(2)

        self.grid = new_grid
        self.turbo = new_grid.turbo

        # Show the monsters and the free column
        shuffled_monsters = list(self.grid.monsters)
        random.shuffle(shuffled_monsters)
        rect.match_x(self.grid.get_col(n - 2))
        self.play(
            AnimationGroup(*[
                monster.animate_set_time(monster.BOB_END)
                for monster in shuffled_monsters
            ], lag_ratio=0.1),
            FadeIn(rect),
            self.camera.frame.animate(run_time=1).restore().scale(1.1, about_point=self.grid.get_bottom()),
            AnimationGroup(
                GrowFromEdge(brace, DOWN, run_time=1),
                Write(label, run_time=1.5)
            ),
        )
        self.wait(1)

        # Hide the monsters again
        self.play(
            AnimationGroup(*[
                monster.animate.set_opacity(0.7)
                for monster in self.grid.monsters[::-1]
            ])
        )

        # Run the strategy again
        turbo = TurboController(self)
        turbo.reveal_free_alleys = False
        turbo.move_speed = 25
        brute_force()
        self.wait(2)


class GetUnderneath(TurboScene):
    def __init__(self, *args, **kwargs):
        random.seed(1)
        n = 15
        # super().__init__(n, get_monster_staircase(n), *args, **kwargs)
        # super().__init__(n, get_monster_staircase_inverted(n), *args, **kwargs)
        super().__init__(n, get_random_monster_positions(n, hardcoded_monsters=[(0, 2)]), *args, **kwargs)

    def construct(self):
        # Add the grid
        self.add(self.grid, self.turbo)

        # # Show the initial positions of the monsters
        # self.play(
        #     AnimationGroup(
        #         *[
        #             monster.animate_set_time(monster.BOB_END)
        #             for monster in self.grid.monsters
        #         ],
        #         lag_ratio=0.3)
        # )
        # self.wait(0.3)

        # # Hide the monsters again
        # self.play(
        #     AnimationGroup(
        #         *[
        #             monster.animate_set_time(monster.BOB_START)
        #             for monster in self.grid.monsters[::-1]
        #         ],
        #         lag_ratio=0.3
        #     )
        # )

        # Execute the strategy
        turbo = TurboController(self)
        n = self.grid.n

        def get_underneath():
            # Try a column
            col = 0
            if turbo.try_col(col):
                return

            while True:
                # Move just before the last monster
                turbo.move_to_col(turbo.last_monster_pos[0])
                turbo.move_to_row(turbo.last_monster_pos[1] - 1)

                # Attempt to pass the monster on the right
                moves = [RIGHT, DOWN, DOWN, LEFT]
                found_monster = False
                for move in moves:
                    if not turbo.move(move):
                        found_monster = True
                        break
                # If successful, move to the bottom row
                if not found_monster:
                    turbo.move_to_row(n - 1)
                    return

        get_underneath()
        self.wait(2)

        # Change the positions of the monsters
        new_positions = get_random_monster_positions(n, hardcoded_monsters=[(0, 2), (1, 1)])
        new_grid = TurboGrid(n, new_positions)
        new_grid.match_height(self.grid).move_to(self.grid)
        self.remove(self.grid)
        self.add(new_grid)
        self.wait(1)

        self.grid = new_grid
        self.turbo = new_grid.turbo
        for monster in self.grid.monsters:
            monster.set_time(monster.BOB_END)
        self.wait(1)

        # Change the positions of the monsters to the second case
        new_positions = get_random_monster_positions(n, hardcoded_monsters=[(0, 2), (1, 3)])
        new_grid = TurboGrid(n, new_positions)
        new_grid.match_height(self.grid).move_to(self.grid)
        self.remove(self.grid)
        self.add(new_grid)
        self.wait(1)

        self.grid = new_grid
        self.turbo = new_grid.turbo
        for monster in self.grid.monsters:
            monster.set_time(monster.BOB_END)
        self.wait(1)

        # Hide the monsters
        for monster in self.grid.monsters:
            monster.set_time(monster.BOB_START)

        # Try the strategy again
        get_underneath()
        self.wait(2)

        # Try the strategy on many random arrangements
        for _ in range(10):
            new_positions = get_random_monster_positions(n)
            new_grid = TurboGrid(n, new_positions)
            new_grid.match_height(self.grid).move_to(self.grid)
            self.remove(self.grid)
            self.add(new_grid)

            self.grid = new_grid
            self.turbo = new_grid.turbo

            for monster in self.grid.monsters:
                monster.set_time(monster.BOB_END)
            self.wait(1)

            get_underneath()
        self.wait(2)

        # Try the strategy again with a diagonal wall of monsters
        new_positions = get_monster_staircase(n)
        new_grid = TurboGrid(n, new_positions)
        new_grid.match_height(self.grid).move_to(self.grid)
        self.remove(self.grid)
        self.add(new_grid)

        self.grid = new_grid
        self.turbo = new_grid.turbo

        self.play(
            AnimationGroup(
                *[
                    monster.animate_set_time(monster.BOB_END)
                    for monster in self.grid.monsters
                ],
                lag_ratio=0.3)
        )
        self.wait(0.3)

        brace = Brace(self.grid, UP)
        label = brace.get_tex("N - 1")
        self.play(
            AnimationGroup(
                self.camera.frame.animate(run_time=1).scale(1.1, about_point=self.grid.get_bottom()),
                AnimationGroup(
                    GrowFromEdge(brace, DOWN, run_time=1),
                    Write(label, run_time=1)
                ),
                lag_ratio=0.2
            )
        )

        get_underneath()


class QuadrantExplorer(TurboScene):
    def __init__(self, *args, **kwargs):
        random.seed(1)
        n = 32
        # super().__init__(n, get_monster_staircase(n), *args, **kwargs)
        # super().__init__(n, get_monster_staircase_inverted(n), *args, **kwargs)
        super().__init__(n, get_random_monster_positions(n), *args, **kwargs)

    def construct(self):
        # Add the grid
        self.add(self.grid, self.turbo)

        # Turbo tries an arbitrary column
        turbo = TurboController(self)
        turbo.move_speed = 3
        n = self.grid.n

        turbo.try_col((n - 1) // 2)
