from manim_imports_ext import *
import random
from scipy.spatial.transform import Slerp

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
        self.position = [i, j]
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


class DummyMonster(Monster):
    def __init__(self, *args, **kwargs):
        super().__init__(TurboGrid(2), 0, 0, *args, **kwargs)
        self.set_time(self.BOB_END)
        self.center()


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
                AnimationGroup(
                    AnimationGroup(*[
                        t.reveal(axis=UP if t.get_x() > monster_tile.get_x() else DOWN)
                        for t in monster_row
                    ], lag_ratio=0.1),
                    AnimationGroup(*[
                        t.reveal(axis=LEFT if t.get_y() > monster_tile.get_y() else RIGHT)
                        for t in monster_col
                    ], lag_ratio=0.1)
                )
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


def get_partial_monster_staircase(min_x, max_x, min_y, max_y):
    return [(min_x + i, min_y + i) for i in range(min(max_x - min_x + 1, max_y - min_y + 1))]


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

    def reset_grid(self, n=None, monster_positions=[]):
        if n is None:
            n = self.grid.n
        new_grid = TurboGrid(n, monster_positions)
        new_grid.match_height(self.grid).move_to(self.grid)
        self.clear()
        self.add(new_grid)

        self.grid = new_grid
        self.turbo = new_grid.turbo

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
                lag_ratio=0.3
            )
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


def perfect_quadrant_explorer_num_cols(k):
    if k <= 1:
        return 1
    return 2 * perfect_quadrant_explorer_num_cols(k - 1) + 1


def perfect_quadrant_explorer_num_rows(k):
    return perfect_quadrant_explorer_num_cols(k) + 1


class QuadrantExplorerPart1(TurboScene):
    def __init__(self, *args, **kwargs):
        random.seed(1)
        n = perfect_quadrant_explorer_num_rows(5)
        # super().__init__(n, get_monster_staircase(n), *args, **kwargs)
        # super().__init__(n, get_monster_staircase_inverted(n), *args, **kwargs)
        random.seed(2)
        super().__init__(n, get_random_monster_positions(n, hardcoded_monsters=[((n - 1) // 2, 12)]), *args, **kwargs)

    def construct(self):
        # Add the grid
        self.add(self.grid, self.turbo)

        # Turbo tries the middle column
        turbo = TurboController(self)
        turbo.move_speed = 2
        n = self.grid.n

        turbo.try_col((n - 1) // 2)
        self.wait(2)

        # Highlight the four quadrants
        colors = [RED, GREEN, YELLOW, BLUE]
        rectangle_points = [
            [
                self.grid.get_tile(
                    0,
                    0
                ).get_corner(UL),
                self.grid.get_tile(
                    turbo.last_monster_pos[0] - 1,
                    turbo.last_monster_pos[1] - 1
                ).get_corner(DR)
            ],
            [
                self.grid.get_tile(
                    turbo.last_monster_pos[0] + 1,
                    0
                ).get_corner(UL),
                self.grid.get_tile(
                    n - 2,
                    turbo.last_monster_pos[1] - 1
                ).get_corner(DR)
            ],
            [
                self.grid.get_tile(
                    0,
                    turbo.last_monster_pos[1]
                ).get_corner(UL),
                self.grid.get_tile(
                    turbo.last_monster_pos[0] - 1,
                    n - 2
                ).get_corner(DR)
            ],
            [
                self.grid.get_tile(
                    turbo.last_monster_pos[0] + 1,
                    turbo.last_monster_pos[1]
                ).get_corner(UL),
                self.grid.get_tile(
                    n - 2,
                    n - 2
                ).get_corner(DR)
            ]
        ]
        rects = VGroup(*[
            Rectangle(
                width=pts[1][0] - pts[0][0],
                height=pts[1][1] - pts[0][1],
                fill_opacity=0.4,
                fill_color=[color],
                stroke_width=0
            ).align_to(pts[0], UL)
            for pts, color in zip(rectangle_points, colors)
        ]).match_z(self.grid.tiles)
        self.play(LaggedStartMap(FadeIn, rects, lag_ratio=0.4))
        self.wait(2)

        # Zoom in on the lower-left quadrant
        self.camera.frame.save_state()
        lane2 = SurroundingRectangle(
            self.grid.get_row(turbo.last_monster_pos[1])[:turbo.last_monster_pos[0]],
            fill_opacity=0.5,
            fill_color=YELLOW,
            stroke_width=0,
            buff=0
        ).match_z(self.grid.tiles)
        self.play(
            self.camera.frame.animate.scale(0.6).move_to(rects[2]),
            FadeOut(rects),
            self.turbo.move_to_position(0, turbo.last_monster_pos[1]),
            FadeIn(lane2),
            run_time=2
        )
        self.wait(1)

        # Luckily get to the bottom (lower-left quadrant)
        self.grid.tiles.save_state()
        turbo.move_speed = 4
        moves = [
            RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, DOWN, RIGHT, DOWN, DOWN,
            LEFT, LEFT, LEFT, LEFT, DOWN, DOWN, RIGHT, RIGHT, DOWN, DOWN,
            RIGHT, DOWN, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, DOWN,
            DOWN, DOWN, LEFT, LEFT, LEFT, LEFT, LEFT, LEFT, UP, LEFT, LEFT,
            LEFT, DOWN, DOWN, DOWN, RIGHT, DOWN, DOWN, RIGHT, DOWN, DOWN,
            RIGHT, RIGHT, RIGHT, RIGHT, DOWN, DOWN
        ]
        for direction in moves:
            turbo.move(direction)
        self.wait(1)

        # Repeat for the lower-right
        self.wait(1)
        lane1 = lane2.copy().move_to(self.grid.get_row(turbo.last_monster_pos[1])[turbo.last_monster_pos[0] + 1:]).match_z(self.grid.tiles)
        self.play(
            self.grid.tiles.animate(run_time=2).restore(),
            self.camera.frame.animate(path_arc=-PI * 0.3, path_arc_axis=DOWN, run_time=2).move_to(rects[3]),
            FadeOut(lane2, run_time=2),
            self.turbo.move_to_position(turbo.last_monster_pos[0] + 1, turbo.last_monster_pos[1], run_time=2),
            self.turbo.time_tracker.animate(run_time=0.5).set_value(self.turbo.RIGHT_START),
            FadeIn(lane1, run_time=2)
        )

        moves = [
            RIGHT, RIGHT, RIGHT, DOWN, DOWN, DOWN, RIGHT, RIGHT, RIGHT,
            RIGHT, DOWN, DOWN, DOWN, DOWN, LEFT, LEFT, DOWN, DOWN, DOWN,
            DOWN, DOWN, DOWN, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, UP, UP,
            UP, RIGHT, RIGHT, DOWN, DOWN, DOWN, DOWN, DOWN, DOWN, LEFT,
            LEFT, LEFT, DOWN, DOWN, DOWN
        ]
        for direction in moves:
            turbo.move(direction)
        self.wait(1)

        # Show a diagonal wall of monsters blocking off the section, moving down iteratively until there's a free column
        original_monster_pos = turbo.last_monster_pos
        for i in range(5):
            hardcoded_monsters = get_partial_monster_staircase(original_monster_pos[0] + 1, n - 2, original_monster_pos[1] + 1 + i, n - 2)
            hardcoded_monsters += [(original_monster_pos[0], original_monster_pos[1] + i)]
            self.reset_grid(monster_positions=get_random_monster_positions(n, hardcoded_monsters=hardcoded_monsters))
            self.play(self.grid.reveal_monster(*hardcoded_monsters[-1]), run_time=0.001)
            monster = self.grid.get_monster(*hardcoded_monsters[-1])
            monster.set_time(monster.X_START)
            self.play(self.turbo.move_to_position(original_monster_pos[0] + 1, original_monster_pos[1] + i), run_time=0.001)
            if i == 0:
                self.play(
                    AnimationGroup(
                        *[
                            monster.animate_set_time(monster.BOB_END)
                            for monster in self.grid.monsters[:len(hardcoded_monsters) - 1]
                        ],
                        lag_ratio=0.3
                    )
                )
                self.wait(1)
                self.play(self.camera.frame.animate.restore())
            else:
                for monster in self.grid.monsters[:len(hardcoded_monsters) - 1]:
                    monster.set_time(monster.BOB_END)
                self.wait(1)

        # Highlight the empty column
        lane3 = SurroundingRectangle(
            self.grid.get_col(n - 2)[hardcoded_monsters[-1][1] + 1:],
            fill_opacity=0.5,
            fill_color=YELLOW,
            stroke_width=0,
            buff=0
        ).match_z(self.grid.tiles)
        self.play(FadeIn(lane3))
        self.wait(2)

        # Show the same thing on the left side
        lane4 = SurroundingRectangle(
            self.grid.get_col(hardcoded_monsters[-1][0] - 1)[hardcoded_monsters[-1][1] + 1:],
            fill_opacity=0.5,
            fill_color=YELLOW,
            stroke_width=0,
            buff=0
        ).match_z(self.grid.tiles)
        self.grid.monsters.generate_target()
        self.grid.monsters.target[:len(hardcoded_monsters) - 1].shift(
            LEFT * (self.grid.monsters[0].get_x() - self.grid.get_col(0).get_x())
        )
        self.play(
            FadeOut(lane3),
            self.turbo.animate.match_x(self.grid.monsters.target[0]),
            MoveToTarget(self.grid.monsters)
        )
        self.play(FadeIn(lane4))

        # Show that there are more columns than there are monsters
        brace1 = Brace(lane1, UP).align_to(self.grid.get_row(hardcoded_monsters[-1][1]).get_top(), DOWN)
        brace2 = Brace(lane2, UP).align_to(self.grid.get_row(hardcoded_monsters[-1][1]).get_top(), DOWN)
        label = Group(Tex(R"\# \text{cols} > \#"), DummyMonster()).arrange(buff=0.1).scale(0.8).next_to(brace1, UP, buff=0.2)
        label1 = Group(BackgroundRectangle(label, buff=0.15).round_corners(0.2), label)
        label2 = label1.copy().match_x(brace2)
        self.play(
            GrowFromEdge(brace1, DOWN),
            GrowFromEdge(brace2, DOWN),
            AnimationGroup(FadeIn(label1[0], run_time=2), Write(label1[1][0]), FadeIn(label1[1][1]), lag_ratio=0.3),
            AnimationGroup(FadeIn(label2[0], run_time=2), Write(label2[1][0]), FadeIn(label2[1][1]), lag_ratio=0.3)
        )
        self.wait(2)

        # Set up the initial monster in the lower half again
        original_monster_pos = hardcoded_monsters[-1]
        for i in [0, -1, -2, -3, -4, -5, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3]:
            hardcoded_monsters = [(original_monster_pos[0], original_monster_pos[1] + i)]
            self.reset_grid(monster_positions=get_random_monster_positions(n, hardcoded_monsters=hardcoded_monsters))
            monster = self.grid.get_monster(*hardcoded_monsters[0])
            self.play(self.grid.reveal_monster(*hardcoded_monsters[-1]), run_time=0.001)
            monster.set_time(monster.X_START)
            self.wait(0.1)

        # Turbo tries to get around to the left
        self.camera.frame.save_state()
        turbo.move_speed = 5
        turbo.move_to_col(hardcoded_monsters[0][0])
        self.set_camera_target_position(0, 0, 0, (-0.01, -1.64, 0.02), 4.17, drift_time=3)
        turbo.move_to_row(hardcoded_monsters[0][1] - 1)
        self.wait(0.5)
        turbo.move_speed = 1
        turbo.move(LEFT)
        turbo.move(DOWN)

        # Highlight the lane to the left
        monster_tile = self.grid.get_tile(*hardcoded_monsters[0])
        lane = lane1.copy().match_y(monster_tile).align_to(monster_tile.get_left(), RIGHT)
        self.play(FadeIn(lane))
        self.wait(0.5)

        # Highlight the full lower-left quadrant
        quadrant = lane.copy().stretch_to_fit_height(
            self.grid.get_col(0)[hardcoded_monsters[0][1]:].get_height()
        ).align_to(
            lane, UP
        ).set_opacity(
            0.35
        ).set_fill(
            opacity=0.1, color=YELLOW
        ).set_stroke(
            width=5, color=YELLOW, opacity=1
        ).scale(1.01)
        self.play(ReplacementTransform(lane, quadrant))
        self.wait(1)

        # Show the case where turbo gets blocked
        hardcoded_monsters += [(hardcoded_monsters[0][0] - 1, hardcoded_monsters[0][1] - 1)]
        hardcoded_monsters += [
            (hardcoded_monsters[0][0] + i + 1, num)
            for i, num in enumerate([23, 27, 25, 30, 20, 28, 29, 22, 21, 24])
        ]
        new_grid = TurboGrid(n, get_random_monster_positions(n, hardcoded_monsters=hardcoded_monsters))
        new_grid.match_height(self.grid).move_to(self.grid)
        self.clear()
        self.add(new_grid)

        self.grid = new_grid
        self.turbo = new_grid.turbo

        monster = self.grid.get_monster(*hardcoded_monsters[0])
        monster.set_time(monster.X_START)
        self.grid.get_col(hardcoded_monsters[0][0])[1:-1].flip(axis=UP)
        self.grid.get_row(hardcoded_monsters[0][1]).flip(axis=RIGHT)
        self.grid.get_tile(*hardcoded_monsters[0]).flip(axis=UP)
        self.play(self.turbo.move_to_position(hardcoded_monsters[0][0], hardcoded_monsters[0][1] - 1, run_time=0.001))
        self.wait(1)

        turbo.move(LEFT)
        turbo.move_speed = 100
        turbo.move_to_col(hardcoded_monsters[0][0])
        turbo.move_speed = 3
        turbo.move_to_row(hardcoded_monsters[0][1] - 1)
        turbo.move_speed = 1
        turbo.move(RIGHT)
        turbo.move(DOWN)

        # Highlight the lane to the right
        lane = lane1.copy().match_y(monster_tile).align_to(monster_tile.get_right(), LEFT)
        self.play(FadeIn(lane))
        self.wait(0.5)

        # Highlight the full lower-right quadrant
        quadrant = lane.copy().stretch_to_fit_height(
            self.grid.get_col(0)[hardcoded_monsters[0][1]:].get_height()
        ).align_to(
            lane, UP
        ).set_fill(
            opacity=0.1, color=YELLOW
        ).set_stroke(
            width=5, color=YELLOW, opacity=1
        ).scale(1.01)
        self.play(ReplacementTransform(lane, quadrant))
        self.wait(2)

        # Show the maximum size of the subproblems
        self.wait(2)
        brace = Brace(quadrant, UP, buff=0)
        label = brace.get_tex(R"\le \frac{N}{2}", font_size=30).set_stroke(width=10, color=BLACK, behind=True)
        self.play(AnimationGroup(GrowFromEdge(brace, DOWN), Write(label), lag_ratio=0.6), run_time=2)

        # Brute force the right side
        def brute_force():
            for col in range(hardcoded_monsters[0][0] + 1, n - 1):
                if turbo.try_col(col):
                    return
                turbo.move_to_col(hardcoded_monsters[0][0])
                turbo.move_to_row(hardcoded_monsters[0][1] - 1)
                turbo.move(RIGHT)
                turbo.move(DOWN)
        turbo.move_speed = 10
        brute_force()


class QuadrantExplorerPart2(TurboScene):
    def __init__(self, *args, **kwargs):
        random.seed(1)
        n = perfect_quadrant_explorer_num_rows(5)
        # super().__init__(n, get_monster_staircase(n), *args, **kwargs)
        # super().__init__(n, get_monster_staircase_inverted(n), *args, **kwargs)
        random.seed(2)
        super().__init__(n, get_random_monster_positions(n, free_column=(n - 1) // 2), *args, **kwargs)

    def construct(self):
        # Add the grid
        self.add(self.grid, self.turbo)

        # Turbo tries the middle column
        turbo = TurboController(self)
        turbo.move_speed = 2
        n = self.grid.n

        turbo.move_speed = 10
        turbo.try_col((n - 1) // 2)
        self.wait(2)

        # Reset and run again, finding a monster in the lower half
        self.reset_grid(monster_positions=get_random_monster_positions(n, hardcoded_monsters=[((n - 1) // 2, 19)]))
        turbo.try_col((n - 1) // 2)
        self.wait(1)

        # Highlight the lower half
        lower_half_rect = Rectangle(
            width=self.grid.get_width(),
            height=self.grid.get_col(0)[n // 2:].get_height(),
            fill_opacity=0.4,
            fill_color=YELLOW,
            stroke_width=0
        ).match_x(self.grid).align_to(self.grid.get_col(0)[n // 2:], UP)
        self.play(FadeIn(lower_half_rect))
        self.play(FadeOut(lower_half_rect))

        # Try getting around the monster on the left
        turbo.move_to_col((n - 1) // 2)
        turbo.move_to_row(turbo.last_monster_pos[1] - 1)
        turbo.move_speed = 2
        turbo.move(LEFT)
        turbo.move(DOWN)
        self.wait(1)

        # Highlight the lower-left quadrant
        def get_quadrants(last_monster_pos=turbo.last_monster_pos):
            rectangle_points = [
                [
                    self.grid.get_tile(
                        0,
                        0
                    ).get_corner(UL),
                    self.grid.get_tile(
                        last_monster_pos[0] - 1,
                        last_monster_pos[1] - 1
                    ).get_corner(DR)
                ],
                [
                    self.grid.get_tile(
                        last_monster_pos[0] + 1,
                        0
                    ).get_corner(UL),
                    self.grid.get_tile(
                        n - 2,
                        last_monster_pos[1] - 1
                    ).get_corner(DR)
                ],
                [
                    self.grid.get_tile(
                        0,
                        last_monster_pos[1]
                    ).get_corner(UL),
                    self.grid.get_tile(
                        last_monster_pos[0] - 1,
                        n - 2
                    ).get_corner(DR)
                ],
                [
                    self.grid.get_tile(
                        last_monster_pos[0] + 1,
                        last_monster_pos[1]
                    ).get_corner(UL),
                    self.grid.get_tile(
                        n - 2,
                        n - 2
                    ).get_corner(DR)
                ]
            ]
            return VGroup(*[
                Rectangle(
                    width=pts[1][0] - pts[0][0],
                    height=pts[1][1] - pts[0][1],
                    fill_opacity=0.4,
                    fill_color=YELLOW,
                    stroke_width=0
                ).align_to(pts[0], UL)
                for pts in rectangle_points
            ]).match_z(self.grid.tiles)
        quadrants = get_quadrants()
        self.play(FadeIn(quadrants[2]))
        self.wait(1)

        # Reset to a version where turbo gets blocked on the left
        original_monster_pos = turbo.last_monster_pos
        self.reset_grid(monster_positions=[original_monster_pos, (original_monster_pos[0] - 1, original_monster_pos[1] - 1)])

        monster = self.grid.get_monster(*original_monster_pos)
        monster.set_time(monster.X_START)
        self.grid.get_col(original_monster_pos[0])[1:-1].flip(axis=UP)
        self.grid.get_row(original_monster_pos[1]).flip(axis=RIGHT)
        self.grid.get_tile(*original_monster_pos).flip(axis=UP)
        self.play(self.turbo.move_to_position(original_monster_pos[0], original_monster_pos[1] - 1, run_time=0.001))
        self.wait(0.5)

        turbo.move(LEFT)
        turbo.move_speed = 100
        turbo.move_to_col(original_monster_pos[0])
        turbo.move_to_row(original_monster_pos[1] - 1)
        turbo.move_speed = 1
        turbo.move(RIGHT)
        turbo.move(DOWN)

        # Highlight the lower-right quadrant
        self.play(FadeIn(quadrants[3]))
        self.wait(2)

        # Move the initial monster above the halfway point
        for i in range(7):
            hardcoded_monsters = [(original_monster_pos[0], original_monster_pos[1] - i)]
            self.reset_grid(monster_positions=get_random_monster_positions(n, hardcoded_monsters=hardcoded_monsters))
            monster = self.grid.get_monster(*hardcoded_monsters[0])
            monster.set_time(monster.X_START)
            self.grid.get_col(hardcoded_monsters[0][0])[1:-1].flip(axis=UP)
            self.grid.get_row(hardcoded_monsters[0][1]).flip(axis=RIGHT)
            self.grid.get_tile(*hardcoded_monsters[0]).flip(axis=UP)
            self.wait(0.2)

        # Show the key property
        half_row_1 = self.grid.get_row(hardcoded_monsters[0][1])[:hardcoded_monsters[0][0]]
        brace1 = Brace(
            half_row_1, DOWN
        ).align_to(
            half_row_1.get_top(), UP
        )
        half_row_2 = self.grid.get_row(hardcoded_monsters[0][1])[hardcoded_monsters[0][0] + 1:]
        brace2 = Brace(
            half_row_2, DOWN
        ).align_to(
            half_row_2.get_top(), UP
        )
        label = Group(Tex(R"\# \text{cols} > \#"), DummyMonster()).arrange(buff=0.1).scale(0.8).next_to(brace1, DOWN, buff=0.2)
        label1 = Group(BackgroundRectangle(label, buff=0.15).round_corners(0.2), label)
        label2 = label1.copy().match_x(brace2)
        quadrants = get_quadrants(last_monster_pos=hardcoded_monsters[0])
        self.play(
            FadeIn(quadrants[0]),
            FadeIn(quadrants[1]),
            GrowFromEdge(brace1, UP),
            GrowFromEdge(brace2, UP),
            AnimationGroup(FadeIn(label1[0], run_time=2), Write(label1[1][0]), FadeIn(label1[1][1]), lag_ratio=0.3),
            AnimationGroup(FadeIn(label2[0], run_time=2), Write(label2[1][0]), FadeIn(label2[1][1]), lag_ratio=0.3)
        )
        self.wait(2)

        # Show the monster positions and the free column
        free_column_1 = 4
        free_column_2 = 25
        lane1 = quadrants[0].copy().surround(self.grid.get_col(free_column_1)[:hardcoded_monsters[0][1]], buff=0)
        lane2 = quadrants[1].copy().surround(self.grid.get_col(free_column_2)[:hardcoded_monsters[0][1]], buff=0)
        self.play(
            ReplacementTransform(quadrants[0], lane1),
            ReplacementTransform(quadrants[1], lane2),
            AnimationGroup(
                *[
                    monster.animate_set_time(monster.BOB_END)
                    for monster in self.grid.monsters if monster.position[1] < hardcoded_monsters[0][1]
                ],
                lag_ratio=0.3
            )
        )
        self.wait(1)

        # Focus back on the quadrants
        self.play(
            FadeOut(Group(brace1, label1, brace2, label2, lane1, lane2)),
            AnimationGroup(
                *[
                    monster.animate_set_time(monster.BOB_START)
                    for monster in self.grid.monsters[::-1] if monster.position[1] < hardcoded_monsters[0][1]
                ],
                lag_ratio=0.1
            ),
            self.camera.frame.animate(run_time=3).reorient(0, 0, 0, (0.01, 1.79, 0.00), 4.16)
        )
        self.wait(2)

        # Pretend turbo gets to the middle row
        self.play(self.turbo.move_to_position(10, hardcoded_monsters[0][1], path_arc=-PI * 0.3, run_time=2))
        self.wait(1)

        # Turbo paces back and forth as he contemplates getting to the bottom
        self.camera.frame.save_state()
        self.set_camera_target_position(0, 0, 0, (0.03, -1.47, 0.00), 4.47, drift_time=3)
        turbo.move_speed = 5
        turbo.move_to_col(hardcoded_monsters[0][0] - 1)
        turbo.move_to_col(0)
        turbo.move_to_col(hardcoded_monsters[0][0] - 1)
        self.wait(1)

        # Reset
        self.play(
            self.camera.frame.animate(run_time=2).restore(),
            self.turbo.move_to_position(0, 0, path_arc=PI * 0.3, run_time=3)
        )

        # Turbo finds a monster
        turbo.move_speed = 7
        turbo.move_to_col(3)
        turbo.move_to_row(hardcoded_monsters[0][1])

        # Pretend turbo gets to the middle row again
        self.play(self.turbo.move_to_position(10, hardcoded_monsters[0][1], path_arc=-PI * 0.3, run_time=2))

        # Turbo uses the free column to get to the bottom
        self.camera.frame.save_state()
        self.set_camera_target_position(0, 0, 0, (0.03, -1.47, 0.00), 4.47, drift_time=3)
        turbo.move_speed = 5
        turbo.move_to_col(turbo.last_monster_pos[0])
        turbo.move_to_row(n - 1)
        self.wait(1)

        # Show the case where no monster is found
        self.play(
            self.camera.frame.animate(run_time=2).restore(),
            self.turbo.move_to_position(0, 0, path_arc=PI * 0.3, run_time=2),
            self.turbo.time_tracker.animate.set_value(self.turbo.RIGHT_START)
        )
        self.wait(1)
        self.reset_grid(monster_positions=hardcoded_monsters + [(3, 7)])
        self.play(self.grid.reveal_monster(*hardcoded_monsters[0]), run_time=0.001)
        monster = self.grid.get_monster(*hardcoded_monsters[0])
        monster.set_time(monster.X_START)
        self.grid.save_state()
        self.wait(1)

        turbo.move_speed = 10
        free_col = 11
        turbo.move_to_col(free_col)
        turbo.move_to_row(hardcoded_monsters[0][1])
        self.wait(0.5)

        # Do a comprehensive sweep to find a monster
        for row in range(hardcoded_monsters[0][1], 0, -1):
            if not turbo.move_to_row(row):
                break
            if row % 2 == 0:
                if not turbo.move_to_col(0):
                    break
            else:
                if not turbo.move_to_col(hardcoded_monsters[0][0] - 1):
                    break

        # Use it to get to the bottom
        turbo.move_to_col(free_col)
        self.set_camera_target_position(0, 0, 0, (0.03, -1.47, 0.00), 4.47, drift_time=2)
        turbo.move_to_row(hardcoded_monsters[0][1])
        turbo.move_to_col(turbo.last_monster_pos[0])
        turbo.move_to_row(n - 1)
        self.wait(1)

        # Show the case where no monster is found in the upper left at all
        monster = self.grid.get_monster(*turbo.last_monster_pos)
        monster.set_time(monster.BOB_START)
        self.play(
            self.camera.frame.animate(run_time=2).restore(),
            self.grid.animate(run_time=3).restore(),
            self.turbo.move_to_position(0, 0, path_arc=PI * 0.3, run_time=2.3),
            self.turbo.time_tracker.animate.set_value(self.turbo.RIGHT_START)
        )
        self.reset_grid(
            monster_positions=hardcoded_monsters + [(25, 6)] + get_partial_monster_staircase(
                0, hardcoded_monsters[0][0] - 1, hardcoded_monsters[0][1] + 1, n - 2
            )
        )
        monster = self.grid.get_monster(*hardcoded_monsters[0])
        self.play(
            self.grid.reveal_monster(*hardcoded_monsters[0]),
            monster.animate_set_time(monster.X_START),
            run_time=0.001
        )

        turbo.move_to_col(free_col)
        turbo.move_to_row(hardcoded_monsters[0][1])
        for row in range(hardcoded_monsters[0][1], 0, -1):
            if not turbo.move_to_row(row):
                break
            if row % 2 == 0:
                if not turbo.move_to_col(0):
                    break
            else:
                if not turbo.move_to_col(hardcoded_monsters[0][0] - 1):
                    break
        self.wait(2)

        # Show the lower section being blocked off
        self.play(
            self.camera.frame.animate.reorient(0, 0, 0, (0.03, -1.47, 0.00), 4.47),
            AnimationGroup(*[
                monster.animate_set_time(monster.BOB_END)
                for monster in self.grid.monsters[2:]
            ], lag_ratio=0.3),
            run_time=2
        )
        self.wait(1)

        # Focus back on the upper quadrants
        self.play(
            self.camera.frame.animate.restore(),
            AnimationGroup(*[
                FadeOut(monster)
                for monster in self.grid.monsters[2:][::-1]
            ]),
            run_time=2
        )
        self.wait(1)

        # Turbo finds the monster in the upper-right instead, and uses it to get to the bottom
        turbo.move_speed = 1
        turbo.move(UP)
        turbo.move(RIGHT)
        turbo.move(RIGHT)
        self.wait(1)

        free_col = 27
        turbo.move_speed = 10
        turbo.move_to_col(free_col)
        turbo.move_to_row(hardcoded_monsters[0][1])

        for row in range(hardcoded_monsters[0][1], 0, -1):
            if not turbo.move_to_row(row):
                break
            if row % 2 == 0:
                if not turbo.move_to_col(hardcoded_monsters[0][0] + 1):
                    break
            else:
                if not turbo.move_to_col(n - 2):
                    break
        turbo.move_to_col(free_col)
        self.set_camera_target_position(0, 0, 0, (0.03, -1.47, 0.00), 4.47, drift_time=2)
        turbo.move_to_row(hardcoded_monsters[0][1])
        turbo.move_to_col(turbo.last_monster_pos[0])
        turbo.move_to_row(n - 1)
        self.wait(1)


class QuadrantExplorerPart3(TurboScene):
    def __init__(self, *args, **kwargs):
        random.seed(1)
        n = perfect_quadrant_explorer_num_rows(5)
        # super().__init__(n, get_monster_staircase(n), *args, **kwargs)
        # super().__init__(n, get_monster_staircase_inverted(n), *args, **kwargs)
        random.seed(2)
        super().__init__(n, [], *args, **kwargs)

    def construct(self):
        # Add the grid
        turbo = TurboController(self)
        turbo.move_speed = 10
        n = self.grid.n
        hardcoded_monsters = [((n - 1) // 2, 12)]
        hardcoded_monsters += [
            (0, 4),
            (1, 5),
            (2, 3),
            (3, 7),
            (4, 9),
            (5, 6),
            (6, 8),
            (7, 11)
        ]
        self.reset_grid(monster_positions=get_random_monster_positions(n, hardcoded_monsters=hardcoded_monsters))
        monster = self.grid.get_monster(*hardcoded_monsters[0])
        self.play(
            self.grid.reveal_monster(*hardcoded_monsters[0]),
            monster.animate_set_time(monster.X_START),
            run_time=0.001
        )

        # Highlight the lower half
        lower_half_rect = Rectangle(
            width=self.grid.get_width(),
            height=self.grid.get_col(0)[n // 2:].get_height(),
            fill_opacity=0.4,
            fill_color=YELLOW,
            stroke_width=0
        ).match_x(self.grid).align_to(self.grid.get_col(0)[n // 2:], UP)
        self.play(FadeIn(lower_half_rect))
        self.play(lower_half_rect.animate.set_color(RED))
        self.play(FadeOut(lower_half_rect))
        self.wait(2)

        # Try to brute force the upper-left quadrant
        def brute_force():
            for col in range(hardcoded_monsters[0][0] - 1):
                turbo.move_to_col(col)
                if turbo.move_to_row(hardcoded_monsters[0][1]):
                    return
        brute_force()
        self.wait(2)

        # Get underneath one of the monsters and go to the bottom
        turbo.move_to_col(3)
        turbo.move_to_row(n - 1)
        self.wait(1)

        # Do the case where no monster is found
        hardcoded_monsters = [((n - 1) // 2, 12), (7, 3)]
        self.reset_grid(monster_positions=hardcoded_monsters)
        monster = self.grid.get_monster(*hardcoded_monsters[0])
        self.play(
            self.grid.reveal_monster(*hardcoded_monsters[0]),
            monster.animate_set_time(monster.X_START),
            run_time=0.001
        )
        self.wait(1)
        turbo.move_to_col(3)
        turbo.move_to_row(hardcoded_monsters[0][1])
        self.wait(0.5)

        for row in range(hardcoded_monsters[0][1], 0, -1):
            if not turbo.move_to_row(row):
                break
            if row % 2 == 0:
                if not turbo.move_to_col(0):
                    break
            else:
                if not turbo.move_to_col(hardcoded_monsters[0][0] - 1):
                    break
        turbo.move_to_col(3)
        turbo.move_to_row(hardcoded_monsters[0][1])
        turbo.move_to_col(turbo.last_monster_pos[0])
        turbo.move_to_row(n - 1)

        # Show the case where no monster is found in the upper left at all
        self.reset_grid(
            monster_positions=[hardcoded_monsters[0]] + [(25, 6)] + get_partial_monster_staircase(
                0, hardcoded_monsters[0][0] - 1, hardcoded_monsters[0][1] + 1, n - 2
            )
        )
        monster = self.grid.get_monster(*hardcoded_monsters[0])
        self.play(
            self.grid.reveal_monster(*hardcoded_monsters[0]),
            monster.animate_set_time(monster.X_START),
            run_time=0.001
        )

        turbo.move_to_col(3)
        turbo.move_to_row(hardcoded_monsters[0][1])
        for row in range(hardcoded_monsters[0][1], 0, -1):
            if not turbo.move_to_row(row):
                break
            if row % 2 == 0:
                if not turbo.move_to_col(0):
                    break
            else:
                if not turbo.move_to_col(hardcoded_monsters[0][0] - 1):
                    break

        # Turbo finds the monster in the upper-right instead, and uses it to get to the bottom
        turbo.move_speed = 4
        turbo.move(UP)
        turbo.move(RIGHT)
        turbo.move(RIGHT)

        free_col = 27
        turbo.move_speed = 10
        turbo.move_to_col(free_col)
        turbo.move_to_row(hardcoded_monsters[0][1])

        for row in range(hardcoded_monsters[0][1], 0, -1):
            if not turbo.move_to_row(row):
                break
            if row % 2 == 0:
                if not turbo.move_to_col(hardcoded_monsters[0][0] + 1):
                    break
            else:
                if not turbo.move_to_col(n - 2):
                    break
        turbo.move_to_col(free_col)
        turbo.move_to_row(hardcoded_monsters[0][1])
        turbo.move_to_col(turbo.last_monster_pos[0])
        turbo.move_to_row(n - 1)
        self.wait(1)
