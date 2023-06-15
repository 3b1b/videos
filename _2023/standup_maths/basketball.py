from manim_imports_ext import *
import pandas as pd


def year_to_file_name(year):
    year_str = str(year)[-2:]
    year_p1_str = str(year + 1)[-2:]
    return f"/Users/grant/Downloads/allShots/nbaShots{year_str}_{year_p1_str}.csv"


def load_data(year):
    frame = pd.read_csv(
        year_to_file_name(year),
        usecols=[
            'LOC_X',
            'LOC_Y',
            'SHOT_MADE_FLAG',
        ]
    )
    coords = np.array(frame[["LOC_X", "LOC_Y"]])
    zero_free_coords = coords[~(coords == [0, 0]).all(1)]

    return zero_free_coords


def get_dots(axes, coords):
    dots = DotCloud(axes.c2p(*coords.T))
    dots.set_color(YELLOW)
    dots.set_glow_factor(0)
    dots.set_radius(0.01)
    dots.shift(0.01 * OUT)
    dots.set_opacity(0.5)
    return dots


def get_bars(axes, coords, resolution=(50, 94)):
    # Test
    # resolution = (10, 20)
    # resolution = (25, 47)
    x_min, x_max = axes.x_range
    y_min, y_max = axes.y_range

    int_coords = (((coords - [x_min, y_min]) / [x_max - x_min, y_max - y_min]) * resolution).astype(int)
    int_coords = int_coords[::50]

    xs = np.linspace(x_min, x_max, resolution[0])
    ys = np.linspace(y_min, y_max, resolution[1])

    factor = 200 / len(int_coords)

    boxes = VGroup()
    for i, j in it.product(range(len(xs) - 1), range(len(ys) - 1)):
        n_in_range = (int_coords == (i, j)).all(1).sum()
        x1, x2 = xs[i:i + 2]
        y1, y2 = ys[j:j + 2]
        line = Line(axes.c2p(x1, y1), axes.c2p(x2, y2))
        box = VCube()
        box.remove(box[-1])
        box.set_fill(RED, 1)
        box.set_stroke(BLACK, 0.5, 0.5)
        box.set_width(0.35 * line.get_width(), stretch=True)
        box.set_height(0.35 * line.get_height(), stretch=True)
        box.set_depth(factor * n_in_range, stretch=True)
        box.move_to(line, IN)
        if n_in_range < 1:
            box.set_opacity(0)
        boxes.add(box)

    boxes.set_fill(BLUE_E)
    boxes.set_stroke(WHITE, 1)

    return boxes


class ShotHistory(InteractiveScene, ThreeDScene):
    def construct(self):
        self.always_depth_test = False
        frame = self.frame
        frame.reorient(-65, 70)

        # Court
        court_rect = Rectangle(50, 94)
        court_rect.set_height(10)
        court_rect.set_fill(GREY, 1)
        court_rect.set_stroke(width=0)
        court_rect.move_to(IN)
        court = ImageMobject("basketball-court")
        court.rotate(90 * DEGREES)
        court.replace(court_rect, stretch=True)
        axes = Axes((-250, 250), (-50, 900))
        axes.replace(court, stretch=True)
        self.add(court)

        # Year label
        year_label = TexText("Year: 2000", font_size=72)
        year_mob = year_label.make_number_changable("2000", group_with_commas=False)
        year_label.fix_in_frame()
        year_label.to_edge(UP)
        self.add(year_label)
        year_mob.add_updater(lambda m: m.fix_in_frame())

        # Create plots
        all_dots = Group()
        all_bars = Group()
        year_range = (1997, 2022)
        low_point = court_rect.get_bottom() + 0.6 * UP

        for year in ProgressDisplay(range(*year_range)):
            try:
                coords = load_data(year)
            except FileNotFoundError:
                continue
            all_dots.add(get_dots(axes, coords))
            all_bars.add(get_bars(axes, coords))
            for bar in all_bars[-1]:
                dist = get_norm(bar.get_nadir() - low_point)
                if dist < 0.5:
                    bar.set_opacity(0)

        for bars in all_bars:
            bars.sort(lambda p: np.dot(p, frame.get_implied_camera_location()))

        all_dots.save_state()
        all_bars.save_state()

        # Show all data
        self.remove(all_dots, all_bars)
        all_dots.restore()
        all_bars.restore()

        self.play(
            frame.animate.reorient(-100),
            ShowSubmobjectsOneByOne(all_dots, rate_func=linear, int_func=np.floor),
            ShowSubmobjectsOneByOne(all_bars, rate_func=linear, int_func=np.floor),
            UpdateFromAlphaFunc(year_mob, lambda m, a: m.set_value(
                integer_interpolate(year_range[0], year_range[1] - 1, a)[0]
            ), rate_func=linear),
            run_time=20,
        )
        self.play(
            frame.animate.reorient(-65),
            run_time=10
        )

        # Reset states
        all_bars.restore()
        all_dots.restore()
        curr_bars = all_bars[-1].copy()
        curr_dots = all_dots[-1].copy()

        self.remove(*all_bars, *all_dots)
        self.add(curr_dots, curr_bars)

        # Roll back to 2000
        def update_dots(dots):
            dots.set_submobjects(
                all_dots[int(year_mob.get_value() - year_range[0])]
            )

        def update_bars(bars):
            bars.set_submobjects(
                all_bars[int(year_mob.get_value() - year_range[0])]
            )

        self.play(
            UpdateFromFunc(curr_dots, update_dots),
            UpdateFromFunc(curr_bars, update_bars),
            ChangeDecimalToValue(year_mob, 2000),
            self.frame.animate.reorient(-117, 69, 0).move_to([-0.54, -2.5, 0.31]).set_height(5.50),
            run_time=3,
        )
        self.wait(2)
        self.play(
            self.frame.animate.reorient(42, 98, 0).move_to([-0.54, -2.5, 0.31]).set_height(5.50),
            run_time=3,
        )
        self.wait()
        self.play(
            self.frame.animate.reorient(-29, 95, 0).move_to([-0.57, -2.46, 0.52]).set_height(6.10),
            run_time=3,
        )

        # Up to 2010
        self.play(
            UpdateFromFunc(curr_dots, update_dots),
            UpdateFromFunc(curr_bars, update_bars),
            ChangeDecimalToValue(year_mob, 2010),
            self.frame.animate.reorient(-159, 84, 0).move_to([-0.41, -2.56, 0.76]).set_height(6.73),
            run_time=4,
        )
        self.play(
            self.frame.animate.reorient(-232, 84, 0).move_to([-0.41, -2.56, 0.76]).set_height(6.73),
            run_time=15,
        )
        self.frame.reorient(128)
        self.wait()

        # Up to 2020
        self.play(
            UpdateFromFunc(curr_dots, update_dots),
            UpdateFromFunc(curr_bars, update_bars),
            ChangeDecimalToValue(year_mob, 2020),
            self.frame.animate.reorient(0, 31, 0).move_to([-0.37, -2.11, 0.71]).set_height(7.26),
            run_time=4,
        )
        self.wait()
        self.play(
            self.frame.animate.reorient(-34, 80, 0).move_to([-0.37, -2.11, 0.71]).set_height(7.26),
            run_time=5,
        )
        self.play(
            self.frame.animate.reorient(29, 70, 0).move_to([0.43, -1.87, 0.95]).set_height(7.49),
            run_time=25,
        )
        self.wait()
