#!/usr/bin/env python

from imports_3b1b import *
from from_3b1b.old.zeta import zeta

from from_3b1b.old.mug import HappyHolidays
from from_3b1b.old.clacks.question import BlocksAndWallExampleMass1e2
from from_3b1b.old.clacks.solution1 import CircleDiagramFromSlidingBlocks1e2

# from IPython import embed

# Test add_smooth_curve_to

# Fix VMobject.fade and VMobject.fade_to

# Fixed rounded_corners to that instead of add_line_to, it
# inserts many anchor points

# AnimationGroup() works when empty, but scene doesn't
# like getting an animation with no run_time


def get_primes(n_min, n_max):
    result = []
    for x in range(max(n_min, 2), n_max):
        has_factor = False
        for p in range(2, int(np.sqrt(x)) + 1):
            if x % p == 0:
                has_factor = True
                break
        if not has_factor:
            result.append(x)
    return result


def all_combinations(nums, exprs, op_map):
    if len(nums) == 1:
        return [(nums[0], exprs[0])]

    combo_lists = []
    for i in range(len(nums) - 1):
        for op_name, oper in op_map.items():
            try:
                new_num = oper(nums[i], nums[i + 1])
            except ZeroDivisionError:
                new_num = np.inf
            new_expr = f"({exprs[i]}{op_name}{exprs[i + 1]})"

            new_nums = [*nums[:i], new_num, *nums[i + 2:]]
            new_exprs = [*exprs[:i], new_expr, *exprs[i + 2:]]
            combo_lists.append(all_combinations(new_nums, new_exprs, op_map))
    return it.chain(*combo_lists)


def countdown(nums, target):
    op_map = {
        "+": op.add,
        "-": op.sub,
        "*": op.mul,
        "/": op.truediv,
    }
    result = []
    for perm in it.permutations(nums):
        exprs = list(map(str, perm))
        for value, expr in all_combinations(perm, exprs, op_map):
            if value == target:
                result.append(expr)
    return result


def duplicates_exist(elems):
    return len(elems) != len(set(elems))


def proportion_with_duplicates(list_size, max_n, num_samples=1000):
    results = np.array([
        duplicates_exist(np.random.randint(0, max_n, list_size))
        for x in range(num_samples)
    ])
    return sum(results) / len(results)


def a(n, x):
    if n == 0:
        return 1
    return n * a(n - 1, x) + x**n


def get_square_diffs(N):
    diffs = [
        x**2 - y**2
        for x in range(N + 1)
        for y in range(x)
        if (x - y) % 2 == 1
    ]
    diffs.sort()
    non_diffs = [
        x
        for x in range(1, N**2 + 1, 2)
        if x not in diffs
    ]
    return diffs, non_diffs


def get_family2(mob):
    result = [mob]
    for sm in mob.submobjects:
        result.extend(get_family(sm))
    return result


def flatten(vmobject):
    new_vmob = VMobject()
    new_vmob.match_style(vmobject)  # Instead, match rgba arrays
    new_vmob.lock_triangulation()
    new_vmob.points = np.array(vmobject.get_all_points())

    new_vmob.saved_orientation = vmobject.get_orientation()
    triangulations = []
    last = 0
    for sm in vmobject.get_family():
        tris = np.array(sm.get_triangulation(sm.get_orientation()))
        new_tri = last + tris
        triangulations.append(new_tri)
        last += len(sm.points)

    new_vmob.saved_triangulation = np.hstack(triangulations).astype(int)
    return new_vmob


def faststack(arrays):
    return np.frombuffer(
        b''.join(arrays),
        dtype=arrays[0].dtype,
    )


def alt_smooth(t):
    s = 1 - t
    return 10 * (s**2) * (t**3) + 5 * s * (t**4) + t**5


def naive_exp(A, n_terms=100):
    return sum((
        np.linalg.matrix_power(A, n) / math.factorial(n)
        for n in range(n_terms)
    ))


class Test(Scene):
    def construct(self):
        pis = VGroup(
            Randolph(mode="hooray", color=BLUE_D),
            Randolph(mode="pondering", color=BLUE_C),
            Randolph(mode="sassy", color=BLUE_E),
            Mortimer(mode="tease"),
        )
        pis.arrange(RIGHT)
        self.add(pis)
        return

        plane = NumberPlane(
            axis_config={"stroke_color": BLUE},
            faded_line_ratio=3,
        )
        self.add(plane)


class TwoErrorGrids(Scene):
    def construct(self):
        grid = VGroup(*[Square() for x in range(16)])
        grid.arrange_in_grid(buff=0)
        grid.set_stroke(WHITE, 1)
        grid.set_height(1)

        grids = VGroup(*[grid.copy() for x in range(16)])
        grids.arrange_in_grid(buff=MED_LARGE_BUFF)
        grids.set_height(7)
        grids.to_edge(RIGHT)

        self.add(grids)

        vects = [
            np.array(tup)
            for tup in it.product(*[[0, 1]] * 4)
        ]

        def vect_to_int(vect):
            return sum([b * (1 << i) for i, b in enumerate(reversed(vect))])

        for vect in vects:
            label = VGroup(*map(Integer, vect))
            label.arrange(RIGHT, buff=SMALL_BUFF)
            label.to_edge(LEFT)
            self.add(label)

            error_int = vect_to_int(vect)
            for i, grid in enumerate(grids):
                grid[i].set_fill(YELLOW, 1)
                grid[i ^ error_int].set_fill(TEAL, 1)

            self.wait()
            grids.set_fill(opacity=0)
            self.remove(label)


class NeuralNetImageAgain(Scene):
    def construct(self):
        layers = VGroup()
        for length in [16, 16, 16, 10]:
            circs = VGroup(*[
                Circle(radius=1)
                for x in range(length)
            ])
            circs.arrange(DOWN, buff=0.5)
            circs.set_stroke(WHITE, 2)
            layers.add(circs)
        layers.set_height(6.5)
        layers.arrange(RIGHT, buff=2.5)

        dots = TexMobject("\\vdots")
        dots.move_to(layers[0])
        layers[0][:8].next_to(dots, UP, MED_SMALL_BUFF)
        layers[0][8:].next_to(dots, DOWN, MED_SMALL_BUFF)

        for layer in layers[1:3]:
            for node in layer:
                node.set_fill(WHITE, opacity=random.random())
        layers[3][6].set_fill(WHITE, 0.9)

        all_edges = VGroup()
        for l1, l2 in zip(layers, layers[1:]):
            edges = VGroup()
            for n1, n2 in it.product(l1, l2):
                edge = Line(
                    n1.get_center(), n2.get_center(),
                    buff=n1.get_height() / 2
                )
                edge.set_stroke(WHITE, 1, opacity=0.75)
                # edge.set_stroke(
                #     color=random.choice([BLUE, RED]),
                #     width=3 * random.random()**6,
                #     # opacity=0.5
                # )
                edges.add(edge)
            all_edges.add(edges)

        network = VGroup(all_edges, layers, dots)

        brace = Brace(network, LEFT)

        self.add(network)
        self.add(brace)


class EarthMorph(Scene):
    CONFIG = {
        "camera_config": {
            "apply_depth_test": True,
            "samples": 8,
        }
    }

    def construct(self):
        torus1 = Torus(r1=1, r2=1)
        torus2 = Torus(r1=3, r2=1)
        sphere = Sphere(radius=3, resolution=torus1.resolution)
        earths = [
            TexturedSurface(surface, "EarthTextureMap", "NightEarthTextureMap")
            for surface in [sphere, torus1, torus2]
        ]
        for mob in earths:
            mob.mesh = SurfaceMesh(mob)
            mob.mesh.set_stroke(BLUE, 1, opacity=0.5)

        earth = earths[0]

        self.camera.frame.set_rotation(
            theta=-30 * DEGREES,
            phi=70 * DEGREES,
        )

        self.add(earth)
        self.play(ShowCreation(earth.mesh, lag_ratio=0.01, run_time=3))
        for mob in earths:
            mob.add(mob.mesh)
        earth.save_state()
        self.play(Rotate(earth, PI / 2), run_time=2)
        for mob in earths[1:]:
            mob.rotate(PI / 2)

        self.play(
            Transform(earth, earths[1]),
            run_time=3
        )

        light = self.camera.light_source
        frame = self.camera.frame

        self.play(
            Transform(earth, earths[2]),
            frame.increment_phi, -10 * DEGREES,
            frame.increment_theta, -20 * DEGREES,
            run_time=3
        )
        frame.add_updater(lambda m, dt: m.increment_theta(-0.1 * dt))
        self.add(light)
        light.save_state()
        self.play(light.move_to, 3 * IN, run_time=5)
        self.play(light.shift, 10 * OUT, run_time=5)
        self.wait(4)


class ProbDiagram(Scene):
    def construct(self):
        square = Square(side_length=1)
        square.move_to(ORIGIN, DL)
        square.set_stroke(width=0)
        square.set_fill(BLUE_E, 1)

        frame = self.camera.frame
        frame.set_height(1.5)
        frame.move_to(square)

        tri = Polygon(ORIGIN, UP, UR)
        tri.set_fill(BLUE_E, 1)
        tri.set_stroke(width=0)

        tris = VGroup(tri)
        N = 1000
        for n in range(1, N):
            tri = Polygon(
                ORIGIN,
                RIGHT + UP / n,
                RIGHT + UP / (n + 1),
            )
            tri.set_stroke(width=0)
            color = BLUE_E if (n % 2 == 0) else RED_D
            tri.set_fill(color, 1)
            tris.add(tri)

        self.add(tris)


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
        arrow = TexMobject("\\Rightarrow")
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

        q_marks = TexMobject("???")[0]
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
        bangs = TexMobject("!?!")
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


class ClassroomHooray(TeacherStudentsScene):
    CONFIG = {
        "background_color": BLACK,
    }

    def construct(self):
        self.look_at(self.screen)
        self.play(self.teacher.change, "raise_right_hand")
        self.change_all_student_modes("hooray", look_at_arg=self.screen)
        self.wait(4)


class LogarithmicSpiral(Scene):
    def construct(self):
        exp_n_tracker = ValueTracker(1)
        group = VGroup()

        def update_group(gr):
            n = 3 * int(np.exp(exp_n_tracker.get_value()))
            gr.set_submobjects(self.get_group(n))

        group.add_updater(update_group)

        self.add(group)
        self.play(
            exp_n_tracker.set_value, 7,
            # exp_n_tracker.set_value, 4,
            run_time=10,
            rate_func=linear,
        )
        self.wait()

    def get_group(self, n, n_spirals=50):
        n = int(n)
        theta = TAU / n
        R = 10

        lines = VGroup(*[
            VMobject().set_points_as_corners([ORIGIN, R * v])
            for v in compass_directions(n)
        ])
        lines.set_stroke(WHITE, min(1, 25 / n))

        # points = [3 * RIGHT]
        # transform = np.array(rotation_matrix_transpose(90 * DEGREES + theta, OUT))
        # transform *= math.sin(theta)
        points = [RIGHT]
        transform = np.array(rotation_matrix_transpose(90 * DEGREES, OUT))
        transform *= math.tan(theta)

        for x in range(n_spirals * n):
            p = points[-1]
            dp = np.dot(p, transform)
            points.append(p + dp)

        vmob = VMobject()
        vmob.set_points_as_corners(points)

        vmob.scale(math.tan(theta), about_point=ORIGIN)

        vmob.set_stroke(BLUE, clip(1000 / n, 1, 3))

        return VGroup(lines, vmob)


class FakeAreaManipulation(Scene):
    CONFIG = {
        "unit": 0.5
    }

    def construct(self):
        unit = self.unit
        group1, group2 = groups = self.get_diagrams()
        for group in groups:
            group.set_width(10 * unit, stretch=True)
            group.set_height(12 * unit, stretch=True)
            group.move_to(3 * DOWN, DOWN)
            group[2].append_points(3 * [group[2].get_left() + LEFT])
            group[3].append_points(3 * [group[3].get_right() + RIGHT])

        grid = NumberPlane(
            x_min=-30,
            x_max=30,
            y_min=-30,
            y_max=30,
            faded_line_ratio=0,
        )
        grid.set_stroke(width=1)
        grid.scale(unit)
        grid.shift(3 * DOWN - grid.c2p(0, 0))

        vertex_dots = VGroup(
            Dot(group1.get_top()),
            Dot(group1.get_corner(DR)),
            Dot(group1.get_corner(DL)),
        )

        self.add(grid)
        self.add(group1)
        self.add(vertex_dots)

        # group1.save_state()

        kw = {
            "lag_ratio": 0.1,
            "run_time": 2,
            "rate_func": bezier([0, 0, 1, 1]),
        }
        path_arc_factors = [-1, 1, 0, 0, -1, 1]
        for target in (group2, group1.copy()):
            self.play(group1.space_out_submobjects, 1.2)
            self.play(*[
                Transform(
                    sm1, sm2,
                    path_arc=path_arc_factors[i] * 60 * DEGREES,
                    **kw
                )
                for i, sm1, sm2 in zip(it.count(), group1, target)
            ])
            group1.unlock_shader_data()
            self.wait(2)

        lines = VGroup(
            Line(group1.get_top(), group1.get_corner(DR)),
            Line(group1.get_top(), group1.get_corner(DL)),
        )
        lines.set_stroke(YELLOW, 2)

        frame = self.camera.frame
        frame.save_state()

        self.play(ShowCreation(lines, lag_ratio=0))
        self.play(
            frame.scale, 0.15,
            frame.move_to, group1[1].get_corner(DR),
            run_time=4,
        )
        self.wait(3)
        self.play(Restore(frame, run_time=2))

        # Another switch
        self.play(*[
            Transform(sm1, sm2, **kw)
            for i, sm1, sm2 in zip(it.count(), group1, group2)
        ])

        # Another zooming
        self.play(
            frame.scale, 0.15,
            frame.move_to, group1[1].get_corner(UL),
            run_time=4,
        )
        self.wait(2)
        self.play(Restore(frame, run_time=4))

        self.embed()

    def get_diagrams(self):
        unit = self.unit

        tri1 = Polygon(2 * LEFT, ORIGIN, 5 * UP)
        tri2 = tri1.copy()
        tri2.flip()
        tri2.next_to(tri1, RIGHT, buff=0)
        tris = VGroup(tri1, tri2)
        tris.scale(unit)
        tris.move_to(3 * UP, UP)
        tris.set_stroke(width=0)
        tris.set_fill(BLUE_D)
        tris[1].set_color(BLUE_C)

        ell = Polygon(
            ORIGIN,
            4 * RIGHT,
            4 * RIGHT + 2 * UP,
            2 * RIGHT + 2 * UP,
            2 * RIGHT + 5 * UP,
            5 * UP,
        )
        ell.scale(unit)
        ells = VGroup(ell, ell.copy().rotate(PI).shift(2 * unit * UP))
        ells.next_to(tris, DOWN, buff=0)

        ells.set_stroke(width=0)
        ells.set_fill(GREY)
        ells[1].set_fill(GREY_BROWN)

        big_tri = Polygon(ORIGIN, 3 * LEFT, 7 * UP)
        big_tri.set_stroke(width=0)
        big_tri.scale(unit)

        big_tri.move_to(ells.get_corner(DL), DR)
        big_tris = VGroup(big_tri, big_tri.copy().rotate(PI, UP, about_point=ORIGIN))

        big_tris[0].set_fill(RED_E, 1)
        big_tris[1].set_fill(RED_C, 1)
        full_group = VGroup(*tris, *ells, *big_tris)
        full_group.set_height(5, about_edge=UP)

        alt_group = full_group.copy()

        alt_group[0].move_to(alt_group, DL)
        alt_group[1].move_to(alt_group, DR)
        alt_group[4].move_to(alt_group[0].get_corner(UR), DL)
        alt_group[5].move_to(alt_group[1].get_corner(UL), DR)
        alt_group[2].rotate(90 * DEGREES)
        alt_group[2].move_to(alt_group[1].get_corner(DL), DR)
        alt_group[2].rotate(-90 * DEGREES)
        alt_group[2].move_to(alt_group[0].get_corner(DR), DL)
        alt_group[3].move_to(alt_group[1].get_corner(DL), DR)

        full_group.set_opacity(0.75)
        alt_group.set_opacity(0.75)

        return full_group, alt_group


class AskAboutCircleProportion(Scene):
    def construct(self):
        R = 2.5
        circle = Circle(radius=R)
        circles = VGroup(circle, circle.copy())
        circles[0].move_to(R * LEFT / 2)
        circles[1].move_to(R * RIGHT / 2)
        circles[0].set_stroke(WHITE, 2)
        circles[1].set_stroke(BLUE, 4)

        dots = VGroup()
        for circle in circles:
            dots.add(Dot(circle.get_center()))

        arc = Arc(
            radius=R,
            start_angle=TAU / 3,
            angle=TAU / 3,
        )
        arc.set_stroke(YELLOW, 4)
        arc.move_arc_center_to(circles[1].get_center())

        question = TextMobject("What proportion of the circle?")
        question.set_height(0.6)
        question.to_corner(UL)
        arrow = Arrow(
            question.get_bottom() + LEFT,
            arc.point_from_proportion(0.25),
        )

        self.add(circles)
        self.add(dots)
        self.add(arc)
        self.add(question)
        self.add(arrow)

        answer = TexMobject("1/3")
        answer.set_height(0.9)
        answer.set_color(YELLOW)
        answer.next_to(question, RIGHT, LARGE_BUFF)
        self.add(answer)


class BorweinIntegrals(Scene):
    def construct(self):
        ints = VGroup(
            TexMobject(
                "\\int_0^\\infty",
                "\\frac{\\sin(x)}{x}",
                "dx = \\frac{\\pi}{2}"
            ),
            TexMobject(
                "\\int_0^\\infty",
                "\\frac{\\sin(x)}{x}",
                "\\frac{\\sin(x/3)}{x/3}",
                "dx = \\frac{\\pi}{2}"
            ),
            TexMobject(
                "\\int_0^\\infty",
                "\\frac{\\sin(x)}{x}",
                "\\frac{\\sin(x/3)}{x/3}",
                "\\frac{\\sin(x/5)}{x/5}",
                "dx = \\frac{\\pi}{2}"
            ),
            TexMobject("\\vdots"),
            TexMobject(
                "\\int_0^\\infty",
                "\\frac{\\sin(x)}{x}",
                "\\frac{\\sin(x/3)}{x/3}",
                "\\frac{\\sin(x/5)}{x/5}",
                "\\dots",
                "\\frac{\\sin(x/13)}{x/13}",
                "dx = \\frac{\\pi}{2}"
            ),
            TexMobject(
                "\\int_0^\\infty",
                "\\frac{\\sin(x)}{x}",
                "\\frac{\\sin(x/3)}{x/3}",
                "\\frac{\\sin(x/5)}{x/5}",
                "\\dots",
                "\\frac{\\sin(x/13)}{x/13}",
                "\\frac{\\sin(x/15)}{x/15}",
                "dx = \\frac{\\pi}{2}",
                "- 0.0000000000231006..."
            ),
        )

        ints.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=RIGHT)
        ints.set_height(FRAME_HEIGHT - 1)
        ints[-1][:-1].align_to(ints[:-1], RIGHT)
        ints[-1][-1].next_to(ints[-1][:-1], RIGHT, SMALL_BUFF)
        ints[3].shift(SMALL_BUFF * LEFT)
        ints.center()

        for integral in ints:
            integral.set_color_by_tex("\\sin(x)", BLUE)
            integral.set_color_by_tex("x/3", TEAL)
            integral.set_color_by_tex("x/5", GREEN)
            integral.set_color_by_tex("x/13", YELLOW)
            integral.set_color_by_tex("x/15", RED_B)

        self.add(ints)


class Test2(TeacherStudentsScene):
    def construct(self):
        self.teacher_says(
            "Hi there everyone!"
        )
        self.wait(2)
        self.embed()


class LogoTest(Scene):
    def construct(self):
        logo = Logo(**{
            "iris_background_blue": "#78C0E3",
            "iris_background_brown": "#8C6239",
            # "blue_spike_colors": [
            #     "#528EA3",
            #     "#3E6576",
            #     "#224C5B",
            #     BLACK,
            # ],
            "blue_spike_colors": [
                BLUE_E,
                interpolate_color("#74C0E3", BLACK, 0.25),
                # BLUE_D,
                BLACK,
            ],
            "brown_spike_colors": [
                "#754C24",
                "#603813",
                "#42210b",
                BLACK,
            ],
            "n_spike_layers": 2,
            "n_spikes": 4,
            "spike_angle": TAU / 14,
        })
        self.add(logo)


class CurrBanner(Banner):
    CONFIG = {
        "camera_config": {
            "pixel_height": 1440,
            "pixel_width": 2560,
        },
        "pi_height": 1.25,
        "pi_bottom": 0.25 * DOWN,
        "use_date": False,
        "date": "Wednesday, March 15th",
        "message_scale_val": 0.9,
        "add_supporter_note": False,
        "pre_date_text": "Next video on ",
    }

    def construct(self):
        super().construct()
        for pi in self.pis:
            pi.set_gloss(0.1)

    # def get_date_message(self):
    #     result = TextMobject(
    #         "Last ", "live ", " lecture 12pm PDT, Friday May 22nd"
    #     )
    #     result.set_color_by_tex("live", Color("red"))
    #     self.add(result)
    #     return result


# For Quanta article

class Clacks1(BlocksAndWallExampleMass1e2):
    CONFIG = {
        "counter_label": "Number of collisions: ",
        "sliding_blocks_config": {
            "block1_config": {
                "mass": 1e0,
                "velocity": -2.0,
                "sheen_factor": 0.0,
                "stroke_width": 1,
                "fill_color": "#cccccc",
            },
            "block2_config": {
                "fill_color": "#cccccc",
                "sheen_factor": 0.0,
                "stroke_width": 1,
            },
        },
        "wait_time": 15,
    }


class Clacks100(Clacks1):
    CONFIG = {
        "sliding_blocks_config": {
            "block1_config": {
                "mass": 1e2,
                "fill_color": "#ff6d58",
                "velocity": -0.5,
                "distance": 5,
            },
        },
        "wait_time": 33,
    }


class Clacks1e4(Clacks1):
    CONFIG = {
        "sliding_blocks_config": {
            "block1_config": {
                "mass": 1e4,
                "fill_color": "#44c5ae",
                "distance": 5,
                "velocity": -0.7,
            },
        },
        "wait_time": 32,
    }


class Clacks1e6(Clacks1):
    CONFIG = {
        "sliding_blocks_config": {
            "block1_config": {
                "mass": 1e6,
                "fill_color": "#2fb9de",
                "velocity": -0.5,
                "distance": 5,
            },
        },
        "wait_time": 26,
    }


class SlowClacks100(Clacks1):
    CONFIG = {
        "sliding_blocks_config": {
            "block1_config": {
                "mass": 1e2,
                "fill_color": "#ff6666",
                "velocity": -0.25,
                "distance": 4.5,
            },
        },
        "wait_time": 65,
    }


class Clacks100VectorEvolution(CircleDiagramFromSlidingBlocks1e2):
    CONFIG = {
        "BlocksAndWallSceneClass": SlowClacks100,
        "show_dot": False,
        "show_vector": True,
    }


# For patreon images

class CircleDivisionImage(Scene):
    CONFIG = {
        "random_seed": 0,
        "n": 0,
    }

    def construct(self):
        # tex = TexMobject("e^{\\tau i}")
        # tex = TexMobject("\\sin(2\\theta) \\over \\sin(\\theta)\\cos(\\theta)")
        # tex = TexMobject("")

        # tex.set_height(FRAME_HEIGHT - 2)
        # if tex.get_width() > (FRAME_WIDTH - 2):
        #     tex.set_width(FRAME_WIDTH - 2)
        # self.add(tex)

        n = self.n
        # angles = list(np.arange(0, TAU, TAU / 9))
        # for i in range(len(angles)):
        #     angles[i] += 1 * np.random.random()

        # random.shuffle(angles)
        # angles = angles[:n + 1]
        # angles.sort()

        # arcs = VGroup(*[
        #     Arc(
        #         start_angle=a1,
        #         angle=(a2 - a1),
        #     )
        #     for a1, a2 in zip(angles, angles[1:])
        # ])
        # arcs.set_height(FRAME_HEIGHT - 1)
        # arcs.set_stroke(YELLOW, 3)

        circle = Circle()
        circle.set_stroke(YELLOW, 5)
        circle.set_height(FRAME_HEIGHT - 1)

        alphas = np.arange(0, 1, 1 / 10)
        alphas += 0.025 * np.random.random(10)
        # random.shuffle(alphas)
        alphas = alphas[:n + 1]

        points = [circle.point_from_proportion(3 * alpha % 1) for alpha in alphas]

        dots = VGroup(*[Dot(point) for point in points])
        for dot in dots:
            dot.scale(1.5)
            dot.set_stroke(BLACK, 2, background=True)
        dots.set_color(BLUE)
        lines = VGroup(*[
            Line(p1, p2)
            for p1, p2 in it.combinations(points, 2)
        ])
        lines.set_stroke(WHITE, 3)

        self.add(circle, lines, dots)


class PatronImage1(CircleDivisionImage):
    CONFIG = {"n": 0}


class PatronImage2(CircleDivisionImage):
    CONFIG = {"n": 1}


class PatronImage4(CircleDivisionImage):
    CONFIG = {"n": 2}


class PatronImage8(CircleDivisionImage):
    CONFIG = {"n": 3}


class PatronImage16(CircleDivisionImage):
    CONFIG = {"n": 4}


class PatronImage31(CircleDivisionImage):
    CONFIG = {"n": 5}


class PatronImage57(CircleDivisionImage):
    CONFIG = {"n": 6}


class PatronImage99(CircleDivisionImage):
    CONFIG = {"n": 7}


class PatronImage163(CircleDivisionImage):
    CONFIG = {"n": 8}


class PatronImage256(CircleDivisionImage):
    CONFIG = {"n": 9}


# For Q&A Video
class Questions(Scene):
    def construct(self):
        kw = {
            "alignment": ""
        }
        TextMobject.CONFIG.update(kw)
        questions = VGroup(
            TextMobject(
                "Who is your favorite mathematician?"
            ),
            TextMobject(
                "A teenage kid walks up to you and says they\\\\",
                "hate maths. What do you tell/show them?"
            ),
            TextMobject(
                "What advice would you want to give to give a\\\\",
                "math enthusiast suffering from an anxiety\\\\",
                "disorder, clinical depression and ADHD?",
            ),
            TextMobject(
                "Is Ben, Ben and Blue still a thing?"
            ),
            TextMobject(
                "Favorite podcasts?"
            ),
            TextMobject(
                "Hey Grant, if you had, both, the responsibility and\\\\",
                "opportunity to best introduce the world of mathematics\\\\",
                "to curious and intelligent minds before they are shaped\\\\",
                "by the antiquated, disempowering and demotivational\\\\",
                "education system of today, what would you do?  (Asking\\\\",
                "because I will soon be a father).\\\\",
            ),
            TextMobject(
                "What's something you think could've\\\\",
                "been discovered long before it was\\\\",
                "actually discovered?",
            ),
            TextMobject(
                "Can we fix math on Wikipedia? Really serious\\\\",
                "here. I constantly go there after your vids for\\\\",
                "a bit of deeper dive and learn - nothing more, ever.\\\\",
                "Compared to almost any topic in the natural sciences\\\\",
                "or physics where at least I get an outline of where\\\\",
                "to go next. It's such a shame."
            ),
        )

        last_question = VMobject()
        for question in questions:
            question.set_width(FRAME_WIDTH - 1)
            self.play(
                FadeInFromDown(question),
                FadeOut(last_question, UP)
            )
            last_question = question
            self.wait(2)


class MathematicianPlusX(Scene):
    def construct(self):
        text = TextMobject(
            "Side note:\\\\",
            "``The Mathematician + X''\\\\",
            "would make a great band name.",
        )
        text.set_width(FRAME_WIDTH - 1)
        self.add(text)


class NoClearCutPath(Scene):
    def construct(self):
        path1 = VMobject()
        path1.start_new_path(3 * LEFT)
        path1.add_line_to(ORIGIN)
        path1.append_points([
            ORIGIN,
            2.5 * RIGHT,
            2.5 * RIGHT + 3 * UP,
            5 * RIGHT + 3 * UP,
        ])
        path2 = path1.copy()
        path2.rotate(PI, axis=RIGHT, about_point=ORIGIN)
        paths = VGroup(path1, path2)
        paths.to_edge(LEFT)
        paths.set_stroke(WHITE, 2)

        labels = VGroup(
            TextMobject("Pure mathematicians"),
            TextMobject("Applied mathematicians"),
        )
        for label, path in zip(labels, paths):
            label.next_to(path.get_end(), RIGHT)

        animations = []
        n_animations = 20
        colors = [BLUE_C, BLUE_D, BLUE_E, GREY_BROWN]
        for x in range(n_animations):
            dot = self.get_dot(random.choice(colors))
            path = random.choice(paths)
            dot.move_to(path.get_start())
            anim = Succession(
                # FadeIn(dot),
                MoveAlongPath(dot, path, run_time=4, rate_func=lambda t: smooth(t, 2)),
                FadeOut(dot),
            )
            animations.append(anim)

        alt_path = VMobject()
        alt_path.start_new_path(paths.get_left())
        alt_path.add_line_to(paths.get_left() + 3 * RIGHT)
        alt_path.add_line_to(2 * UP)
        alt_path.add_line_to(2 * DOWN + RIGHT)
        alt_path.add_line_to(2 * RIGHT)
        alt_path.add_line_to(3 * RIGHT)
        alt_path.add_line_to(8 * RIGHT)
        alt_path.make_smooth()
        alt_path.set_stroke(YELLOW, 3)
        dashed_path = DashedVMobject(alt_path, num_dashes=100)

        alt_dot = self.get_dot(YELLOW)
        alt_dot.move_to(alt_path.get_start())
        alt_dot_anim = MoveAlongPath(
            alt_dot,
            alt_path,
            run_time=10,
            rate_func=linear,
        )

        self.add(paths)
        self.add(labels)
        self.play(
            LaggedStart(
                *animations,
                run_time=10,
                lag_ratio=1 / n_animations,
            ),
            ShowCreation(
                dashed_path,
                rate_func=linear,
                run_time=10,
            ),
            alt_dot_anim,
        )
        self.wait()

    def get_dot(self, color):
        dot = Dot()
        dot.scale(1.5)
        dot.set_color(color)
        dot.set_stroke(BLACK, 2, background=True)
        return dot


class Cumulative(Scene):
    def construct(self):
        colors = list(Color(BLUE_B).range_to(BLUE_D, 20))
        rects = VGroup(*[
            Rectangle(
                height=0.3, width=4,
                fill_color=random.choice(colors),
                fill_opacity=1,
                stroke_color=WHITE,
                stroke_width=1,
            )
            for i, color in zip(range(20), it.cycle(colors))
        ])
        rects.arrange(UP, buff=0)

        check = TexMobject("\\checkmark").set_color(GREEN)
        cross = TexMobject("\\times").set_color(RED)
        checks, crosses = [
            VGroup(*[
                mob.copy().next_to(rect, RIGHT, SMALL_BUFF)
                for rect in rects
            ])
            for mob in [check, cross]
        ]

        rects.set_fill(opacity=0)
        self.add(rects)

        for i in range(7):
            self.play(
                rects[i].set_fill, None, 1.0,
                Write(checks[i]),
                run_time=0.5,
            )
        self.play(FadeOut(rects[7], 2 * LEFT))
        self.play(
            LaggedStartMap(Write, crosses[8:]),
            LaggedStartMap(
                ApplyMethod, rects[8:],
                lambda m: (m.set_fill, None, 0.2),
            )
        )
        self.wait()
        rects[7].set_opacity(1)
        self.play(
            FadeIn(rects[7], 2 * LEFT),
            FadeIn(checks[7], 2 * LEFT),
        )
        self.play(
            LaggedStartMap(
                ApplyMethod, rects[8:],
                lambda m: (m.set_fill, None, 1.0),
            ),
            LaggedStart(*[
                ReplacementTransform(cross, check)
                for cross, check in zip(crosses[8:], checks[8:])
            ])
        )
        self.wait()


class HolidayStorePromotionTime(HappyHolidays):
    def construct(self):
        title = TextMobject("Holiday store promotion time!")
        title.set_width(FRAME_WIDTH - 1)
        title.to_edge(UP)
        self.add(title)

        HappyHolidays.construct(self)


# Random play
class SumRotVectors(Scene):
    CONFIG = {
        "n_vects": 100,
    }

    def construct(self):
        plane = ComplexPlane()
        circle = Circle(color=YELLOW)

        t_tracker = ValueTracker(0)
        get_t = t_tracker.get_value

        vects = always_redraw(lambda: self.get_vects(get_t()))

        self.add(plane, circle)
        self.add(t_tracker, vects)
        self.play(
            t_tracker.set_value, 1,
            run_time=10,
            rate_func=linear,
        )
        # vects.clear_updaters()
        self.play(ShowIncreasingSubsets(vects, run_time=5))

    def get_vects(self, t):
        vects = VGroup()
        last_tip = ORIGIN
        for n in range(self.n_vects):
            vect = Vector(RIGHT)
            vect.rotate(n * TAU * t, about_point=ORIGIN)
            vect.shift(last_tip)
            last_tip = vect.get_end()
            vects.add(vect)

        vects.set_submobject_colors_by_gradient(BLUE, GREEN, YELLOW, RED)
        vects.set_opacity(0.5)
        return vects


class ZetaTest(Scene):
    def construct(self):
        plane = ComplexPlane()
        self.add(plane)
        plane.scale(0.2)

        s = complex(0.5, 14.135)
        N = int(1e7)
        lines = VGroup()
        color = it.cycle([BLUE, RED])
        r = int(1e3)
        for k in range(1, N + 1, r):
            c = sum([
                l**(-s) * (N - l + 1) / N
                for l in range(k, k + r)
            ])
            line = Line(plane.n2p(0), plane.n2p(c))
            line.set_color(next(color))
            if len(lines) > 0:
                line.shift(lines[-1].get_end())
            lines.add(line)

        self.add(lines)
        self.add(
            Dot(lines[-1].get_end(), color=YELLOW),
            Dot(
                center_of_mass([line.get_end() for line in lines]),
                color=YELLOW
            ),
        )


class BigCross(Scene):
    def construct(self):
        rect = FullScreenFadeRectangle()
        big_cross = Cross(rect)
        big_cross.set_stroke(width=30)
        self.add(big_cross)


class Eoc1Thumbnail(GraphScene):
    CONFIG = {

    }

    def construct(self):
        title = TextMobject(
            "The Essence of\\\\Calculus",
            tex_to_color_map={
                "\\emph{you}": YELLOW,
            },
        )
        subtitle = TextMobject("Chapter 1")
        subtitle.match_width(title)
        subtitle.scale(0.75)
        subtitle.next_to(title, DOWN)
        # title.add(subtitle)
        title.set_width(FRAME_WIDTH - 2)
        title.to_edge(UP)
        title.set_stroke(BLACK, 8, background=True)
        # answer = TextMobject("...yes")
        # answer.to_edge(DOWN)

        axes = Axes(
            x_min=-1,
            x_max=5,
            y_min=-1,
            y_max=5,
            y_axis_config={
                "include_tip": False,
            },
            x_axis_config={
                "unit_size": 2,
            },
        )
        axes.set_width(FRAME_WIDTH - 1)
        axes.center().to_edge(DOWN)
        axes.shift(DOWN)
        self.x_axis = axes.x_axis
        self.y_axis = axes.y_axis
        self.axes = axes

        graph = self.get_graph(self.func)
        rects = self.get_riemann_rectangles(
            graph,
            x_min=0, x_max=4,
            dx=0.2,
        )
        rects.set_submobject_colors_by_gradient(BLUE, GREEN)
        rects.set_opacity(1)
        rects.set_stroke(BLACK, 1)

        self.add(axes)
        self.add(graph)
        self.add(rects)
        self.add(title)
        # self.add(answer)

    def func(slef, x):
        return 0.35 * ((x - 2)**3 - 2 * (x - 2) + 6)


class HenryAnimation(Scene):
    CONFIG = {
        "mu_color": BLUE_E,
        "nu_color": RED_E,
        "camera_config": {
            "background_color": WHITE,
        },
        "tex_config": {
            "color": BLACK,
            "background_stroke_width": 0,
        },
    }

    def construct(self):
        eq1 = self.get_field_eq("\\mu", "\\nu")
        indices = list(filter(
            lambda t: t[0] <= t[1],
            it.product(range(4), range(4))
        ))
        sys1, sys2 = [
            VGroup(*[
                self.get_field_eq(i, j, simple=simple)
                for i, j in indices
            ])
            for simple in (True, False)
        ]
        for sys in sys1, sys2:
            sys.arrange(DOWN, buff=MED_LARGE_BUFF)
            sys.set_height(FRAME_HEIGHT - 0.5)
            sys2.center()

        sys1.next_to(ORIGIN, RIGHT)

        eq1.generate_target()
        group = VGroup(eq1.target, sys1)
        group.arrange(RIGHT, buff=2)
        arrows = VGroup(*[
            Arrow(
                eq1.target.get_right(), eq.get_left(),
                buff=0.2,
                color=BLACK,
                stroke_width=2,
                tip_length=0.2,
            )
            for eq in sys1
        ])

        self.play(FadeIn(eq1, DOWN))
        self.wait()
        self.play(
            MoveToTarget(eq1),
            LaggedStart(*[
                GrowArrow(arrow)
                for arrow in arrows
            ]),
        )
        self.play(
            LaggedStart(*[
                TransformFromCopy(eq1, eq)
                for eq in sys1
            ], lag_ratio=0.2),
        )
        self.wait()

        #
        sys1.generate_target()
        sys1.target.to_edge(LEFT)
        sys2.to_edge(RIGHT)
        new_arrows = VGroup(*[
            Arrow(
                e1.get_right(), e2.get_left(),
                buff=SMALL_BUFF,
                color=BLACK,
                stroke_width=2,
                tip_length=0.2,
            )
            for e1, e2 in zip(sys1.target, sys2)
        ])
        self.play(
            MoveToTarget(sys1),
            MaintainPositionRelativeTo(arrows, sys1),
            MaintainPositionRelativeTo(eq1, sys1),
            VFadeOut(arrows),
            VFadeOut(eq1),
        )

        #
        sys1_rects, sys2_rects = [
            VGroup(*map(self.get_rects, sys))
            for sys in [sys1, sys2]
        ]
        self.play(
            LaggedStartMap(FadeIn, sys1_rects),
            LaggedStartMap(GrowArrow, new_arrows),
            run_time=1,
        )
        self.play(
            TransformFromCopy(sys1_rects, sys2_rects),
            TransformFromCopy(sys1, sys2),
        )
        self.play(
            FadeOut(sys1_rects),
            FadeOut(sys2_rects),
        )
        self.wait()

    def get_field_eq(self, mu, nu, simple=True):
        mu = "{" + str(mu) + " }"  # Deliberate space
        nu = "{" + str(nu) + "}"
        config = dict(self.tex_config)
        config["tex_to_color_map"] = {
            mu: self.mu_color,
            nu: self.nu_color,
        }
        if simple:
            tex_args = [
                ("R_{%s%s}" % (mu, nu),),
                ("-{1 \\over 2}",),
                ("g_{%s%s}" % (mu, nu),),
                ("R",),
                ("=",),
                ("8\\pi T_{%s%s}" % (mu, nu),),
            ]
        else:
            tex_args = [
                (
                    "\\left(",
                    "\\partial_\\rho \\Gamma^{\\rho}_{%s%s} -" % (mu, nu),
                    "\\partial_%s \\Gamma^{\\rho}_{\\rho%s} +" % (nu, mu),
                    "\\Gamma^{\\rho}_{\\rho\\lambda}",
                    "\\Gamma^{\\lambda}_{%s%s} -" % (nu, mu),
                    "\\Gamma^{\\rho}_{%s \\lambda}" % nu,
                    "\\Gamma^{\\lambda}_{\\rho %s}" % mu,
                    "\\right)",
                ),
                ("-{1 \\over 2}",),
                ("g_{%s%s}" % (mu, nu),),
                (
                    "g^{\\alpha \\beta}",
                    "\\left(",
                    "\\partial_\\rho \\Gamma^{\\rho}_{\\beta \\alpha} -"
                    "\\partial_\\beta \\Gamma^{\\rho}_{\\rho\\alpha} +",
                    "\\Gamma^{\\rho}_{\\rho\\lambda}",
                    "\\Gamma^{\\lambda}_{\\beta\\alpha} -"
                    "\\Gamma^{\\rho}_{\\beta \\lambda}",
                    "\\Gamma^{\\lambda}_{\\rho \\alpha}",
                    "\\right)",
                ),
                ("=",),
                ("8\\pi T_{%s%s}" % (mu, nu),),
            ]

        result = VGroup(*[
            TexMobject(*args, **config)
            for args in tex_args
        ])
        result.arrange(RIGHT, buff=SMALL_BUFF)
        return result

    def get_rects(self, equation):
        return VGroup(*[
            SurroundingRectangle(
                equation[i],
                buff=0.025,
                color=color,
                stroke_width=1,
            )
            for i, color in zip(
                [0, 3],
                [GREY, GREY]
            )
        ])


class PrimePiEPttern(Scene):
    def construct(self):
        self.add(FullScreenFadeRectangle(fill_color=WHITE, fill_opacity=1))

        tex0 = TexMobject(
            "\\frac{1}{1^2}", "+"
            "\\frac{1}{2^2}", "+"
            "\\frac{1}{3^2}", "+"
            "\\frac{1}{4^2}", "+"
            "\\frac{1}{5^2}", "+"
            "\\frac{1}{6^2}", "+"
            # "\\frac{1}{7^2}", "+"
            # "\\frac{1}{8^2}", "+"
            # "\\frac{1}{9^2}", "+"
            # "\\frac{1}{10^2}", "+"
            # "\\frac{1}{11^2}", "+"
            # "\\frac{1}{12^2}", "+"
            "\\cdots",
            "=",
            "\\frac{\\pi^2}{6}",
        )
        self.alter_tex(tex0)
        # self.add(tex0)

        tex1 = TexMobject(
            "\\underbrace{\\frac{1}{1^2}}_{\\text{kill}}", "+",
            "\\underbrace{\\frac{1}{2^2}}_{\\text{keep}}", "+",
            "\\underbrace{\\frac{1}{3^2}}_{\\text{keep}}", "+",
            "\\underbrace{\\frac{1}{4^2}}_{\\times (1 / 2)}", "+",
            "\\underbrace{\\frac{1}{5^2}}_{\\text{keep}}", "+",
            "\\underbrace{\\frac{1}{6^2}}_{\\text{kill}}", "+",
            "\\underbrace{\\frac{1}{7^2}}_{\\text{keep}}", "+",
            "\\underbrace{\\frac{1}{8^2}}_{\\times (1 / 3)}", "+",
            "\\underbrace{\\frac{1}{9^2}}_{\\times (1 / 2)}", "+",
            "\\underbrace{\\frac{1}{10^2}}_{\\text{kill}}", "+",
            "\\underbrace{\\frac{1}{11^2}}_{\\text{keep}}", "+",
            "\\underbrace{\\frac{1}{12^2}}_{\\text{kill}}", "+",
            "\\cdots",
            # "=",
            # "\\frac{\\pi^2}{6}"
        )
        self.alter_tex(tex1)
        tex1.set_color_by_tex("kill", RED)
        tex1.set_color_by_tex("keep", GREEN_E)
        tex1.set_color_by_tex("times", BLUE_D)

        self.add(tex1)
        return

        # tex1 = TexMobject(
        #     "\\underbrace{\\frac{1}{1}}_{\\text{kill}}", "+",
        #     "\\underbrace{\\frac{-1}{3}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{1}{5}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{-1}{7}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{1}{9}}_{\\times (1 / 2)}", "+",
        #     "\\underbrace{\\frac{-1}{11}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{1}{13}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{-1}{15}}_{\\text{kill}}", "+",
        #     "\\underbrace{\\frac{1}{17}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{-1}{19}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{1}{21}}_{\\text{kill}}", "+",
        #     "\\underbrace{\\frac{-1}{23}}_{\\text{keep}}", "+",
        #     "\\underbrace{\\frac{1}{25}}_{\\times (1 / 2)}", "+",
        #     "\\underbrace{\\frac{-1}{27}}_{\\times (1 / 3)}", "+",
        #     "\\cdots",
        #     "=",
        #     "\\frac{\\pi}{4}"
        # )
        # self.alter_tex(tex1)
        # VGroup(
        #     tex1[2 * 0],
        #     tex1[2 * 7],
        #     tex1[2 * 10],
        # ).set_color(RED)
        # VGroup(
        #     tex1[2 * 1],
        #     tex1[2 * 2],
        #     tex1[2 * 3],
        #     tex1[2 * 5],
        #     tex1[2 * 6],
        #     tex1[2 * 8],
        #     tex1[2 * 9],
        #     tex1[2 * 11],
        # ).set_color(GREEN_E)
        # VGroup(
        #     tex1[2 * 4],
        #     tex1[2 * 12],
        #     tex1[2 * 13],
        # ).set_color(BLUE_D)

        # self.add(tex1)

        # tex2 = TexMobject(
        #     "\\frac{-1}{3}", "+",
        #     "\\frac{1}{5}", "+",
        #     "\\frac{-1}{7}", "+",
        #     "\\frac{1}{2}", "\\cdot", "\\frac{1}{9}", "+",
        #     "\\frac{-1}{11}", "+",
        #     "\\frac{1}{13}", "+",
        #     "\\frac{1}{17}", "+",
        #     "\\frac{-1}{19}", "+",
        #     "\\frac{-1}{23}", "+",
        #     "\\frac{1}{2}", "\\cdot", "\\frac{1}{25}", "+",
        #     "\\frac{1}{3}", "\\cdot", "\\frac{-1}{27}", "+",
        #     "\\cdots",
        # )
        # self.alter_tex(tex2)
        # VGroup(
        #     tex2[2 * 0],
        #     tex2[2 * 1],
        #     tex2[2 * 2],
        #     tex2[2 * 5],
        #     tex2[2 * 6],
        #     tex2[2 * 7],
        #     tex2[2 * 8],
        #     tex2[2 * 9],
        # ).set_color(GREEN_E)
        # VGroup(
        #     tex2[2 * 3],
        #     tex2[2 * 4],
        #     tex2[2 * 10],
        #     tex2[2 * 11],
        #     tex2[2 * 12],
        #     tex2[2 * 13],
        # ).set_color(BLUE_D)

        tex2 = TexMobject(
            "\\frac{1}{2^2}", "+",
            "\\frac{1}{3^2}", "+",
            "\\frac{1}{2}", "\\cdot", "\\frac{1}{4^2}", "+",
            "\\frac{1}{5^2}", "+",
            "\\frac{1}{7^2}", "+",
            "\\frac{1}{3}", "\\cdot", "\\frac{1}{8^2}", "+",
            "\\frac{1}{2}", "\\cdot", "\\frac{1}{9^2}", "+",
            "\\frac{1}{11^2}", "+",
            "\\frac{1}{13^2}", "+",
            "\\frac{1}{4}", "\\cdot", "\\frac{1}{16^2}", "+",
            "\\cdots",
        )
        self.alter_tex(tex2)
        VGroup(
            tex2[2 * 0],
            tex2[2 * 1],
            tex2[2 * 4],
            tex2[2 * 5],
            tex2[2 * 10],
            tex2[2 * 11],
        ).set_color(GREEN_E)
        VGroup(
            tex2[2 * 2],
            tex2[2 * 3],
            tex2[2 * 6],
            tex2[2 * 7],
            tex2[2 * 8],
            tex2[2 * 9],
            tex2[2 * 12],
            tex2[2 * 13],
        ).set_color(BLUE_D)
        self.add(tex2)

        exp = TexMobject(
            "e^{\\left(",
            "0" * 30,
            "\\right)}",
            "= \\frac{\\pi^2}{6}"
        )
        self.alter_tex(exp)
        exp[1].set_opacity(0)
        tex2.replace(exp[1], dim_to_match=0)

        self.add(exp, tex2)

    def alter_tex(self, tex):
        tex.set_color(BLACK)
        tex.set_stroke(BLACK, 0, background=True)
        tex.set_width(FRAME_WIDTH - 1)


class NewMugThumbnail(Scene):
    def construct(self):
        title = TexMobject(
            "V - E + F = 0",
            tex_to_color_map={"0": YELLOW},
        )
        title.scale(3)
        title.to_edge(UP)
        image = ImageMobject("sci_youtubers_thumbnail")
        image.set_height(5.5)
        image.next_to(title, DOWN)
        self.add(title, image)


class Vertical3B1B(Scene):
    def construct(self):
        words = TextMobject(
            "3", "Blue", "1", "Brown",
        )
        words.scale(2)
        words[::2].scale(1.2)
        buff = 0.2
        words.arrange(
            DOWN,
            buff=buff,
            aligned_edge=LEFT,
        )
        words[0].match_x(words[1][0])
        words[2].match_x(words[3][0])
        self.add(words)

        logo = Logo()
        logo.next_to(words, LEFT)
        self.add(logo)

        VGroup(logo, words).center()


class ZetaSpiral(Scene):
    def construct(self):
        max_t = 50
        spiral = VGroup(*[
            ParametricCurve(
                lambda t: complex_to_R3(
                    zeta(complex(0.5, t))
                ),
                t_min=t1,
                t_max=t2 + 0.1,
            )
            for t1, t2 in zip(it.count(), range(1, max_t))
        ])
        spiral.set_stroke(width=0, background=True)
        # spiral.set_color_by_gradient(BLUE, GREEN, YELLOW, RED)
        # spiral.set_color_by_gradient(BLUE, YELLOW)
        spiral.set_color(YELLOW)

        width = 10
        for piece in spiral:
            piece.set_stroke(width=width)
            width *= 0.98
            dot = Dot()
            dot.scale(0.25)
            dot.match_color(piece)
            dot.set_stroke(BLACK, 1, background=True)
            dot.move_to(piece.get_start())
            # piece.add(dot)

        label = TexMobject(
            "\\zeta\\left(0.5 + i{t}\\right)",
            tex_to_color_map={"{t}": YELLOW},
            background_stroke_width=0,
        )
        label.scale(1.5)
        label.next_to(spiral, DOWN)

        group = VGroup(spiral, label)
        group.set_height(FRAME_HEIGHT - 1)
        group.center()

        self.add(group)


# PRODUCTION_QUALITY_CAMERA_CONFIG = {
#     "pixel_height": 2560,
#     "pixel_width": 2560,
#     "frame_rate": 60,
# }

class PendulumPhaseSpace(Scene):
    def construct(self):
        axes = Axes(
            x_min=-PI,
            x_max=PI,
            y_min=-2,
            y_max=2,
            x_axis_config={
                "tick_frequency": PI / 4,
                "unit_size": FRAME_WIDTH / TAU,
                "include_tip": False,
            },
            y_axis_config={
                "unit_size": 4,
                "tick_frequency": 0.25,
            },
        )

        def func(point, mu=0.1, k=0.5):
            theta, theta_dot = axes.p2c(point)
            return axes.c2p(
                theta_dot,
                -mu * theta_dot - k * np.sin(theta),
            )

        field = VectorField(
            func,
            delta_x=1.5 * FRAME_WIDTH / 12,
            delta_y=1.5,
            y_min=-6,
            y_max=6,
            length_func=lambda norm: 1.25 * sigmoid(norm),
            max_magnitude=4,
            vector_config={
                "tip_length": 0.75,
                "max_tip_length_to_length_ratio": 0.35,
            },
        )
        field.set_stroke(width=12)
        colors = list(Color(BLUE).range_to(RED, 5))
        for vect in field:
            mag = get_norm(1.18 * func(vect.get_start()))
            vect.set_color(colors[min(int(mag), len(colors) - 1)])

        line = VMobject()
        line.start_new_path(axes.c2p(
            -3 * TAU / 4,
            1.75,
        ))

        dt = 0.1
        t = 0
        total_time = 60

        while t < total_time:
            t += dt
            last_point = line.get_last_point()
            new_point = last_point + dt * func(last_point)
            if new_point[0] > FRAME_WIDTH / 2:
                new_point = last_point + FRAME_WIDTH * LEFT
                line.start_new_path(new_point)
            else:
                line.add_smooth_curve_to(new_point)

        line.set_stroke(WHITE, 6)

        # self.add(axes)
        self.add(field)
        # line.set_stroke(BLACK)
        # self.add(line)


class TenDThumbnail(Scene):
    def construct(self):
        square = Square()
        square.set_height(3.5)
        square.set_stroke(YELLOW, 5)
        r = square.get_width() / 2
        circles = VGroup(*[
            Circle(radius=r).move_to(corner)
            for corner in square.get_vertices()
        ])
        circles.set_stroke(BLUE, 5)
        circles.set_fill(BLUE, 0.5)
        circles.set_sheen(0.5, UL)
        lil_circle = Circle(
            radius=(np.sqrt(2) - 1) * r
        )
        lil_circle.set_stroke(YELLOW, 3)
        lil_circle.set_fill(YELLOW, 0.5)

        group = VGroup(circles, lil_circle, square)
        group.to_edge(LEFT)
        square.scale(2)

        words = TextMobject(
            "What\\\\"
            "about\\\\"
            "in 10D?\\\\"
            # "dimensions?"
        )
        words.set_height(5)
        words.to_edge(RIGHT)

        arrow = Arrow(
            words[0][0].get_left(),
            lil_circle.get_center(),
            path_arc=90 * DEGREES,
            buff=0.5,
        )
        arrow.set_color(RED)
        arrow.set_stroke(width=12)
        arrow_group = VGroup(
            arrow.copy().set_stroke(BLACK, 16),
            arrow,
        )

        self.add(group)
        self.add(words)
        self.add(arrow_group)


class WhyPi(Scene):
    def construct(self):
        title = TextMobject("Why $\\pi$?")
        title.scale(3)
        title.to_edge(UP)

        formula1 = TexMobject(
            "1 +"
            "\\frac{1}{4} +"
            "\\frac{1}{9} +"
            "\\frac{1}{16} +"
            "\\frac{1}{25} + \\cdots"
            "=\\frac{\\pi^2}{6}"
        )
        formula1.set_color(YELLOW)
        formula1.set_width(FRAME_WIDTH - 2)
        formula1.next_to(title, DOWN, MED_LARGE_BUFF)

        formula2 = TexMobject(
            "1 -"
            "\\frac{1}{3} +"
            "\\frac{1}{5} -"
            "\\frac{1}{7} +"
            "\\frac{1}{9} - \\cdots"
            "=\\frac{\\pi}{4}"
        )
        formula2.set_color(BLUE_C)
        formula2.set_width(FRAME_WIDTH - 2)
        formula2.next_to(formula1, DOWN, LARGE_BUFF)

        self.add(title)
        self.add(formula1)
        self.add(formula2)


class GeneralExpositionIcon(Scene):
    def construct(self):
        title = TextMobject("What is \\underline{\\qquad \\qquad}?")
        title.scale(3)
        title.to_edge(UP)
        randy = Randolph()
        randy.change("pondering")
        randy.set_height(4.5)
        randy.to_edge(DOWN)
        randy.look_at(title[0][0])

        self.add(title)
        self.add(randy)


class GeometryIcon(Scene):
    def construct(self):
        im = ImageMobject("/Users/grant/Desktop/maxresdefault (9).jpg")
        im.set_height(FRAME_HEIGHT)
        im.scale(0.9, about_edge=DOWN)
        word = TextMobject("Geometry")
        word.scale(3)
        word.to_edge(UP)
        self.add(im, word)


class PhysicsIcon(Scene):
    def construct(self):
        im = ImageMobject("/Users/grant/Desktop/maxresdefault (10).png")
        im.set_height(FRAME_HEIGHT)
        im.shift(UP)
        title = TextMobject("Physics")
        title.scale(3)
        title.to_edge(UP)

        self.add(im)
        self.add(title)


class SupportIcon(Scene):
    def construct(self):
        randy = Randolph(mode="coin_flip_2")
        morty = Mortimer(mode="gracious")
        pis = VGroup(randy, morty)
        pis.arrange(RIGHT, buff=3)
        pis.to_edge(DOWN)
        randy.make_eye_contact(morty)
        heart = SuitSymbol("hearts")
        heart.set_height(1)
        heart.next_to(randy, UR, buff=-0.5)
        heart.shift(0.5 * RIGHT)

        # rect = FullScreenFadeRectangle(opacity=0.85)

        # self.add(rect)
        self.add(pis)
        self.add(heart)


class SupportPitch1(Scene):
    CONFIG = {
        "camera_config": {
            "background_opacity": 0.85,
        },
        "mode1": "happy",
        "mode2": "hooray",
        "words1": "So what do\\\\you do?",
        "words2": "Oh, I make\\\\videos about\\\\math.",
    }

    def construct(self):
        randy = Randolph()
        randy.to_corner(DL)
        morty = Mortimer()
        morty.to_corner(DR)

        randy.change(self.mode1, morty.eyes)
        morty.change(self.mode2, randy.eyes)

        b1 = randy.get_bubble(
            self.words1,
            bubble_class=SpeechBubble,
            height=3,
            width=4,
        )
        b1.add(b1.content)
        b1.shift(0.25 * UP)
        b2 = morty.get_bubble(
            self.words2,
            bubble_class=SpeechBubble,
            height=3,
            width=4,
        )
        # b2.content.scale(0.9)
        b2.add(b2.content)
        b2.shift(0.25 * DOWN)

        self.add(randy)
        self.add(morty)
        self.add(b2)
        self.add(b1)


class SupportPitch2(SupportPitch1):
    CONFIG = {
        "mode1": "confused",
        "mode2": "speaking",
        "words1": "Wait, how does\\\\that work?",
        "words2": "People pay\\\\for them.",
    }


class SupportPitch3(SupportPitch1):
    CONFIG = {
        "mode1": "hesitant",
        "mode2": "coin_flip_2",
        "words1": "Oh, so like\\\\a paid course?",
        "words2": "Well, no,\\\\everything\\\\is free.",
    }


class SupportPitch4(SupportPitch1):
    CONFIG = {
        "mode1": "confused",
        "mode2": "hesitant",
        "words1": "Wait, what?",
        "words2": "I know,\\\\it's weird...",
    }


class RantPage(Scene):
    CONFIG = {
    }

    def construct(self):
        squares = VGroup(Square(), Square())
        squares.arrange(DOWN, buff=MED_SMALL_BUFF)
        squares.set_height(FRAME_HEIGHT - 0.5)
        squares.set_width(5, stretch=True)
        squares.set_stroke(WHITE, 2)
        squares.set_fill(BLACK, opacity=0.75)
        s1, s2 = squares

        # Group1
        morty = Mortimer(mode="maybe")
        for eye, pupil in zip(morty.eyes, morty.pupils):
            pupil.move_to(eye)
        morty.shift(MED_SMALL_BUFF * UL)
        words = TextMobject(
            "What were you\\\\expecting to be here?"
        )
        bubble = SpeechBubble(direction=RIGHT)
        bubble.match_style(s1)
        bubble.add_content(words)
        bubble.resize_to_content()
        bubble.add(bubble.content)
        bubble.pin_to(morty)
        group1 = VGroup(morty, bubble)
        group1.set_height(s1.get_height() - MED_SMALL_BUFF)
        group1.next_to(s1.get_corner(DR), UL, SMALL_BUFF)

        # Group 2
        morty = Mortimer(mode="surprised")
        morty.shift(MED_SMALL_BUFF * UL)
        words = TextMobject(
            "Go on!\\\\Give the rant!"
        )
        bubble = SpeechBubble(direction=RIGHT)
        bubble.match_style(s1)
        bubble.add_content(words)
        bubble.resize_to_content()
        bubble.add(bubble.content)
        bubble.pin_to(morty)
        group2 = VGroup(morty, bubble)
        group2.set_height(s2.get_height() - MED_SMALL_BUFF)
        group2.next_to(s2.get_corner(DR), UL, SMALL_BUFF)

        self.add(squares)
        self.add(group1)
        self.add(group2)


#####

# TODO, this seems useful.  Put it somewhere
def get_count_mobs_on_points(points, height=0.1):
    result = VGroup()
    for n, point in enumerate(points):
        count = Integer(n)
        count.set_height(height)
        count.move_to(point)
        result.add(count)
    return result


def exp(A, max_n=30):
    factorial = 1
    A = np.array(A)
    result = np.zeros(A.shape)
    power = np.identity(A.shape[0])
    for k in range(max_n):
        result = result + (power / factorial)
        power = np.dot(power, A)
        factorial *= (k + 1)
    return result


def get_arrows(squares):
    result = VGroup()
    for square in squares:
        corners = square.get_vertices()
        if len(corners) >= 2:
            p0, p1 = corners[:2]
            angle = angle_of_vector(p1 - p0)
        else:
            angle = 0
        arrow = Vector(RIGHT)
        arrow.rotate(angle)
        arrow.move_to(square.get_center() + RIGHT)
        arrow.match_color(square)
        result.add(arrow)
    return result


class ClipsLogo(Scene):
    def construct(self):
        logo = Logo()
        logo.set_height(FRAME_HEIGHT - 0.5)
        square = Square(stroke_width=0, fill_color=BLACK, fill_opacity=1)
        square.scale(5)
        square.rotate(45 * DEGREES)
        square.move_to(ORIGIN, LEFT)
        self.add(logo, square)


class PowersOfTwo(MovingCameraScene):
    def construct(self):
        R = 3
        circle = Circle(radius=R)
        circle.set_stroke(BLUE, 2)
        n = 101
        dots = VGroup()
        numbers = VGroup()
        points = [
            rotate_vector(R * DOWN, k * TAU / n)
            for k in range(n)
        ]
        dots = VGroup(*[
            Dot(point, radius=0.03, color=YELLOW)
            for point in points
        ])
        numbers = VGroup(*[
            Integer(k).scale(0.2).move_to((1.03) * point)
            for k, point in enumerate(points)
        ])
        lines = VGroup(*[
            Line(points[k], points[(2 * k) % n])
            for k in range(n)
        ])
        lines.set_stroke(RED, 2)
        arrows = VGroup(*[
            Arrow(
                n1.get_bottom(), n2.get_bottom(),
                path_arc=90 * DEGREES,
                buff=0.02,
                max_tip_length_to_length_ratio=0.1,
            )
            for n1, n2 in zip(numbers, numbers[::2])
        ])
        transforms = [
            Transform(
                numbers[k].copy(), numbers[(2 * k) % n].copy(),
            )
            for k in range(n)
        ]

        title = TexMobject(
            "\\mathds{Z} / (101 \\mathds{Z})"
        )
        title.scale(2)

        frame = self.camera_frame
        frame.save_state()

        self.add(circle, title)
        self.play(
            LaggedStart(*map(FadeInFromLarge, dots)),
            LaggedStart(*[
                FadeIn(n, -normalize(n.get_center()))
                for n in numbers
            ]),
            run_time=2,
        )
        self.play(
            frame.scale, 0.25,
            {"about_point": circle.get_bottom() + SMALL_BUFF * DOWN}
        )
        n_examples = 6
        for k in range(1, n_examples):
            self.play(
                ShowCreation(lines[k]),
                transforms[k],
                ShowCreation(arrows[k])
            )
        self.play(
            frame.restore,
            FadeOut(arrows[:n_examples])
        )
        self.play(
            LaggedStart(*map(ShowCreation, lines[n_examples:])),
            LaggedStart(*transforms[n_examples:]),
            FadeOut(title, rate_func=squish_rate_func(smooth, 0, 0.5)),
            run_time=10,
            lag_ratio=0.01,
        )
        self.play(
            LaggedStart(*[
                ShowCreationThenFadeOut(line.copy().set_stroke(PINK, 3))
                for line in lines
            ]),
            run_time=3
        )
        self.wait(4)


class Cardiod(Scene):
    def construct(self):
        r = 1
        big_circle = Circle(color=BLUE, radius=r)
        big_circle.set_stroke(width=1)
        big_circle.rotate(-PI / 2)
        time_tracker = ValueTracker()
        get_time = time_tracker.get_value

        def get_lil_circle():
            lil_circle = big_circle.copy()
            lil_circle.set_color(YELLOW)
            time = get_time()
            lil_circle.rotate(time)
            angle = 0.5 * time
            lil_circle.shift(
                rotate_vector(UP, angle) * 2 * r
            )
            return lil_circle
        lil_circle = always_redraw(get_lil_circle)

        cardiod = ParametricCurve(
            lambda t: op.add(
                rotate_vector(UP, 0.5 * t) * (2 * r),
                -rotate_vector(UP, 1.0 * t) * r,
            ),
            t_min=0,
            t_max=(2 * TAU),
        )
        cardiod.set_color(MAROON_B)

        dot = Dot(color=RED)
        dot.add_updater(lambda m: m.move_to(lil_circle.get_start()))

        self.add(big_circle, lil_circle, dot, cardiod)
        for color in [RED, PINK, MAROON_B]:
            self.play(
                ShowCreation(cardiod.copy().set_color(color)),
                time_tracker.increment_value, TAU * 2,
                rate_func=linear,
                run_time=6,
            )


class UsualBanner(Banner):
    CONFIG = {
        # "date": "Wednesday, April 3rd",
        # "use_date": True,
    }


class HindiBanner(Banner):
    CONFIG = {
        "message_scale_val": 0.9,
    }

    def get_probabalistic_message(self):
        # return TextMobject("3Blue1Brown")
        result = TextMobject("3Blue1Brown", "XXX")
        result[1].set_opacity(0)
        return result


class HindiLogo(Scene):
    def construct(self):
        logo = Logo()
        logo.set_height(3)
        words = TextMobject("")
        words.set_width(logo.get_width() * 0.9)
        words.move_to(logo)
        words.shift(SMALL_BUFF * UP)
        words.set_stroke(BLACK, 5, background=True)
        self.add(logo, words)


class PatronBanner(Banner):
    CONFIG = {
        "date": "Sunday, December 22nd",
        "use_date": True,
        "add_supporter_note": True,
    }


def vs_func(point):
    x, y, z = point
    return np.array([
        np.cos(y) * x,
        y - x,
        0,
    ])


class ColorTest(Scene):
    CONFIG = {
        "color_group_width": 2,
        "color_group_height": 2,
    }

    def construct(self):
        self.add(TextMobject("Some title").scale(1.5).to_edge(UP))

        blues = self.get_color_group("BLUE")
        greens = self.get_color_group("GREEN")
        browns = self.get_color_group("BROWN")
        teals = self.get_color_group("TEAL")
        pinks = self.get_color_group("PINK")
        reds = self.get_color_group("RED")
        yellows = self.get_color_group("YELLOW")
        greys = self.get_color_group("GREY")

        color_groups = VGroup(
            blues, teals, greens, greys,
            reds, pinks, yellows, browns,
        )
        color_groups.arrange_in_grid(n_rows=2, buff=LARGE_BUFF)
        self.add(color_groups)

        # tone_groups = VGroup(*[
        #     VGroup(*[group[i] for group in color_groups])
        #     for i in range(5)
        # ])
        # for group in tone_groups:
        #     group.arrange(DOWN, buff=0.0)

        # tone_groups.arrange(RIGHT, buff=MED_LARGE_BUFF)

        # for group in color_groups:
        #     self.add_equation(group)

        # self.add(tone_groups)

    def add_equation(self, group):
        eq = TexMobject("e^{\\pi\\sqrt{163}}")
        eq.match_color(group[2])
        eq.next_to(group, DOWN, buff=SMALL_BUFF)
        eq.set_stroke(width=0, background=True)
        self.add(eq)

    def get_color_group(self, name):
        colors = [
            globals().get(name + "_{}".format(c))
            for c in "ABCDE"
        ]
        group = VGroup(*[
            Rectangle(
                stroke_width=0,
                fill_opacity=1,
            ).set_color(color)
            for color in colors
        ])
        group.arrange(DOWN, buff=0.1)
        group.set_width(self.color_group_width, stretch=True)
        group.set_height(self.color_group_height, stretch=True)
        return group
