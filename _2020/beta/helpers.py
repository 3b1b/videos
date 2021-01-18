from manim_imports_ext import *
import scipy.stats


CMARK_TEX = "\\text{\\ding{51}}"
XMARK_TEX = "\\text{\\ding{55}}"

COIN_COLOR_MAP = {
    "H": BLUE_E,
    "T": RED_E,
}


class Histogram(Group):
    CONFIG = {
        "height": 5,
        "width": 10,
        "y_max": 1,
        "y_axis_numbers_to_show": range(20, 120, 20),
        "y_axis_label_height": 0.25,
        "y_tick_freq": 0.2,
        "x_label_freq": 1,
        "include_h_lines": True,
        "h_line_style": {
            "stroke_width": 1,
            "stroke_color": GREY_B,
            # "draw_stroke_behind_fill": True,
        },
        "bar_style": {
            "stroke_width": 1,
            "stroke_color": WHITE,
            "fill_opacity": 1,
        },
        "bar_colors": [BLUE, GREEN]
    }

    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)
        self.data = data

        self.add_axes()
        if self.include_h_lines:
            self.add_h_lines()
        self.add_bars(data)
        self.add_x_axis_labels()
        self.add_y_axis_labels()

    def add_axes(self):
        n_bars = len(self.data)
        axes_config = {
            "x_min": 0,
            "x_max": n_bars,
            "x_axis_config": {
                "unit_size": self.width / n_bars,
                "include_tip": False,
            },
            "y_min": 0,
            "y_max": self.y_max,
            "y_axis_config": {
                "unit_size": self.height / self.y_max,
                "include_tip": False,
                "tick_frequency": self.y_tick_freq,
            },
        }
        axes = Axes(**axes_config)
        axes.center()
        self.axes = axes
        self.add(axes)

    def add_h_lines(self):
        axes = self.axes
        axes.h_lines = VGroup()
        for tick in axes.y_axis.tick_marks:
            line = Line(**self.h_line_style)
            line.match_width(axes.x_axis)
            line.move_to(tick.get_center(), LEFT)
            axes.h_lines.add(line)
        axes.add(axes.h_lines)

    def add_bars(self, data):
        self.bars = self.get_bars(data)
        self.add(self.bars)

    def add_x_axis_labels(self):
        axes = self.axes
        axes.x_labels = VGroup()
        for x, bar in list(enumerate(self.bars))[::self.x_label_freq]:
            label = Integer(x)
            label.set_height(0.25)
            label.next_to(bar, DOWN)
            axes.x_labels.add(label)
        axes.add(axes.x_labels)

    def add_y_axis_labels(self):
        axes = self.axes
        labels = VGroup()
        for value in self.y_axis_numbers_to_show:
            label = Integer(value, unit="\\%")
            label.set_height(self.y_axis_label_height)
            label.next_to(axes.y_axis.n2p(0.01 * value), LEFT)
            labels.add(label)
        axes.y_labels = labels
        axes.y_axis.add(labels)

    # Bar manipulations
    def get_bars(self, data):
        portions = np.array(data).astype(float)
        total = portions.sum()
        if total == 0:
            portions[:] = 0
        else:
            portions /= total
        bars = VGroup()
        for x, prop in enumerate(portions):
            bar = Rectangle()
            width = get_norm(self.axes.c2p(1, 0) - self.axes.c2p(0, 0))
            height = get_norm(self.axes.c2p(0, 1) - self.axes.c2p(0, 0))
            bar.set_width(width)
            bar.set_height(height * prop, stretch=True)
            bar.move_to(self.axes.c2p(x, 0), DL)
            bars.add(bar)

        bars.set_submobject_colors_by_gradient(*self.bar_colors)
        bars.set_style(**self.bar_style)
        return bars


# Images of randomness


def get_random_process(choices, shuffle_time=2, total_time=3, change_rate=0.05,
                       h_buff=0.1, v_buff=0.1):
    content = choices[0]

    container = Square()
    container.set_opacity(0)
    container.set_width(content.get_width() + 2 * h_buff, stretch=True)
    container.set_height(content.get_height() + 2 * v_buff, stretch=True)
    container.move_to(content)
    container.add(content)
    container.time = 0
    container.last_change_time = 0

    def update(container, dt):
        container.time += dt

        t = container.time
        change = all([
            (t % total_time) < shuffle_time,
            container.time - container.last_change_time > change_rate
        ])
        if change:
            mob = container.submobjects[0]
            new_mob = random.choice(choices)
            new_mob.match_height(mob)
            new_mob.move_to(container, DL)
            new_mob.shift(2 * np.random.random() * h_buff * RIGHT)
            new_mob.shift(2 * np.random.random() * v_buff * UP)
            container.set_submobjects([new_mob])
            container.last_change_time = container.time

    container.add_updater(update)
    return container


def get_die_faces():
    dot = Dot()
    dot.set_width(0.15)
    dot.set_color(BLUE_B)

    square = Square()
    square.round_corners(0.25)
    square.set_stroke(WHITE, 2)
    square.set_fill(GREY_E, 1)
    square.set_width(0.6)

    edge_groups = [
        (ORIGIN,),
        (UL, DR),
        (UL, ORIGIN, DR),
        (UL, UR, DL, DR),
        (UL, UR, ORIGIN, DL, DR),
        (UL, UR, LEFT, RIGHT, DL, DR),
    ]

    arrangements = VGroup(*[
        VGroup(*[
            dot.copy().move_to(square.get_bounding_box_point(ec))
            for ec in edge_group
        ])
        for edge_group in edge_groups
    ])
    square.set_width(1)

    faces = VGroup(*[
        VGroup(square.copy(), arrangement)
        for arrangement in arrangements
    ])
    faces.arrange(RIGHT)

    return faces


def get_random_die(**kwargs):
    return get_random_process(get_die_faces(), **kwargs)


def get_random_card(height=1, **kwargs):
    cards = DeckOfCards()
    cards.set_height(height)
    return get_random_process(cards, **kwargs)


# Coins
def get_coin(symbol, color=None):
    if color is None:
        color = COIN_COLOR_MAP.get(symbol, GREY_E)
    coin = VGroup()
    circ = Circle()
    circ.set_fill(color, 1)
    circ.set_stroke(WHITE, 1)
    circ.set_height(1)
    label = TexText(symbol)
    label.set_height(0.5 * circ.get_height())
    label.move_to(circ)
    coin.add(circ, label)
    coin.symbol = symbol
    return coin


def get_random_coin(**kwargs):
    return get_random_process([get_coin("H"), get_coin("T")], **kwargs)


def get_prob_coin_label(symbol="H", color=None, p=0.5, num_decimal_places=2):
    label = Tex("P", "(", "00", ")", "=",)
    coin = get_coin(symbol, color)
    template = label.get_part_by_tex("00")
    coin.replace(template)
    label.replace_submobject(label.index_of_part(template), coin)
    rhs = DecimalNumber(p, num_decimal_places=num_decimal_places)
    rhs.next_to(label, RIGHT, buff=MED_SMALL_BUFF)
    label.add(rhs)
    return label


def get_q_box(mob):
    box = SurroundingRectangle(mob)
    box.set_stroke(WHITE, 1)
    box.set_fill(GREY_E, 1)
    q_marks = Tex("???")
    max_width = 0.8 * box.get_width()
    max_height = 0.8 * box.get_height()

    if q_marks.get_width() > max_width:
        q_marks.set_width(max_width)

    if q_marks.get_height() > max_height:
        q_marks.set_height(max_height)

    q_marks.move_to(box)
    box.add(q_marks)
    return box


def get_coin_grid(bools, height=6):
    coins = VGroup(*[
        get_coin("H" if heads else "T")
        for heads in bools
    ])
    coins.arrange_in_grid()
    coins.set_height(height)
    return coins


def get_prob_positive_experience_label(include_equals=False,
                                       include_decimal=False,
                                       include_q_mark=False):
    label = Tex(
        "P", "(", "00000", ")",
    )

    pe = TexText("Positive\\\\experience")
    pe.set_color(GREEN)
    pe.replace(label[2], dim_to_match=0)
    label.replace_submobject(2, pe)
    VGroup(label[1], label[3]).match_height(
        pe, stretch=True, about_edge=DOWN,
    )
    if include_equals:
        eq = Tex("=").next_to(label, RIGHT)
        label.add(eq)
    if include_decimal:
        decimal = DecimalNumber(0.95)
        decimal.next_to(label, RIGHT)
        decimal.set_color(YELLOW)
        label.decimal = decimal
        label.add(decimal)
    if include_q_mark:
        q_mark = Tex("?")
        q_mark.relative_mob = label[-1]
        q_mark.add_updater(
            lambda m: m.next_to(m.relative_mob, RIGHT, SMALL_BUFF)
        )
        label.add(q_mark)

    return label


def get_beta_dist_axes(y_max=20, y_unit=2, label_y=False, **kwargs):
    config = {
        "x_min": 0,
        "x_max": 1,
        "x_axis_config": {
            "unit_size": 0.1,
            "tick_frequency": 0.1,
            "include_tip": False,
        },
        "y_min": 0,
        "y_max": y_max,
        "y_axis_config": {
            "unit_size": 1,
            "tick_frequency": y_unit,
            "include_tip": False,
        },
    }
    result = Axes(**config)
    origin = result.c2p(0, 0)
    kw = {
        "about_point": origin,
        "stretch": True,
    }
    result.x_axis.set_width(11, **kw)
    result.y_axis.set_height(6, **kw)

    x_vals = np.arange(0, 1, 0.2) + 0.2
    result.x_axis.add_numbers(
        *x_vals,
        number_config={"num_decimal_places": 1}
    )

    if label_y:
        result.y_axis.add_numbers(
            *np.arange(y_unit, y_max, y_unit)
        )
        label = TexText("Probability density")
        label.scale(0.5)
        label.next_to(result.y_axis.get_top(), UR, SMALL_BUFF)
        label.next_to(result.y_axis, UP, SMALL_BUFF)
        label.align_to(result.y_axis.numbers, LEFT)
        result.add(label)
        result.y_axis_label = label

    result.to_corner(DR, LARGE_BUFF)

    return result


def scaled_pdf_axes(scale_factor=3.5):
    axes = get_beta_dist_axes(
        label_y=True,
        y_unit=1,
    )
    axes.y_axis.numbers.set_submobjects([
        *axes.y_axis.numbers[:5],
        *axes.y_axis.numbers[4::5]
    ])
    sf = scale_factor
    axes.y_axis.stretch(sf, 1, about_point=axes.c2p(0, 0))
    for number in axes.y_axis.numbers:
        number.stretch(1 / sf, 1)
    axes.y_axis_label.to_edge(LEFT)
    axes.y_axis_label.add_background_rectangle(opacity=1)
    axes.set_stroke(background=True)
    return axes


def close_off_graph(axes, graph):
    x_max = axes.x_axis.p2n(graph.get_end())
    graph.add_line_to(axes.c2p(x_max, 0))
    graph.add_line_to(axes.c2p(0, 0))
    return graph


def get_beta_graph(axes, n_plus, n_minus, **kwargs):
    dist = scipy.stats.beta(n_plus + 1, n_minus + 1)
    graph = axes.get_graph(dist.pdf, **kwargs)
    close_off_graph(axes, graph)
    graph.set_stroke(BLUE, 2)
    graph.set_fill(BLUE_E, 1)
    return graph


def get_beta_label(n_plus, n_minus, point=ORIGIN):
    template = TexText("Beta(", "00", ",", "00", ")")
    template.scale(1.5)
    a_label = Integer(n_plus + 1)
    a_label.set_color(GREEN)
    b_label = Integer(n_minus + 1)
    b_label.set_color(RED)

    for i, label in (1, a_label), (3, b_label):
        label.match_height(template[i])
        label.move_to(template[i], DOWN)
        template.replace_submobject(i, label)
    template.save_state()
    template.arrange(RIGHT, buff=0.15)
    for t1, t2 in zip(template, template.saved_state):
        t1.align_to(t2, DOWN)

    return template


def get_plusses_and_minuses(n_rows=15, n_cols=20, p=0.95):
    result = VGroup()
    for x in range(n_rows * n_cols):
        if random.random() < p:
            mob = Tex(CMARK_TEX)
            mob.set_color(GREEN)
            mob.is_plus = True
        else:
            mob = Tex(XMARK_TEX)
            mob.set_color(RED)
            mob.is_plus = False
        mob.set_width(1)
        result.add(mob)

    result.arrange_in_grid(n_rows, n_cols)
    result.set_width(5.5)
    return result


def get_checks_and_crosses(bools, width=12):
    result = VGroup()
    for positive in bools:
        if positive:
            mob = Tex(CMARK_TEX)
            mob.set_color(GREEN)
        else:
            mob = Tex(XMARK_TEX)
            mob.set_color(RED)
        mob.positive = positive
        mob.set_width(0.5)
        result.add(mob)
    result.arrange(RIGHT, buff=MED_SMALL_BUFF)
    result.set_width(width)
    return result


def get_underlines(marks):
    underlines = VGroup()
    for mark in marks:
        underlines.add(Underline(mark))
    for line in underlines:
        line.align_to(underlines[-1], DOWN)
    return underlines


def get_random_checks_and_crosses(n=50, s=0.95, width=12):
    return get_checks_and_crosses(
        bools=(np.random.random(n) < s),
        width=width
    )


def get_random_num_row(s, n=10):
    values = np.random.random(n)
    nums = VGroup()
    syms = VGroup()
    for x, value in enumerate(values):
        num = DecimalNumber(value)
        num.set_height(0.25)
        num.move_to(x * RIGHT)
        num.positive = (num.get_value() < s)
        if num.positive:
            num.set_color(GREEN)
            sym = Tex(CMARK_TEX)
        else:
            num.set_color(RED)
            sym = Tex(XMARK_TEX)
        sym.match_color(num)
        sym.match_height(num)
        sym.positive = num.positive
        sym.next_to(num, UP)

        nums.add(num)
        syms.add(sym)

    row = VGroup(nums, syms)
    row.nums = nums
    row.syms = syms
    row.n_positive = sum([m.positive for m in nums])

    row.set_width(10)
    row.center().to_edge(UP)
    return row


def get_prob_review_label(n_positive, n_negative, s=0.95):
    label = Tex(
        "P(",
        f"{n_positive}\\,{CMARK_TEX}", ",\\,",
        f"{n_negative}\\,{XMARK_TEX}",
        "\\,|\\,",
        "s = {:.2f}".format(s),
        ")",
    )
    label.set_color_by_tex_to_color_map({
        CMARK_TEX: GREEN,
        XMARK_TEX: RED,
        "0.95": YELLOW,
    })
    return label


def get_binomial_formula(n, k, p):
    n_mob = Integer(n, color=WHITE)
    k_mob = Integer(k, color=GREEN)
    nmk_mob = Integer(n - k, color=RED)
    p_mob = DecimalNumber(p, color=YELLOW)

    n_str = "N" * len(n_mob)
    k_str = "K" * len(k_mob)
    p_str = "P" * len(k_mob)
    nmk_str = "M" * len(nmk_mob)

    formula = Tex(
        "\\left(",
        "{" + n_str,
        "\\over",
        k_str + "}",
        "\\right)",
        "(", p_str, ")",
        "^{" + k_str + "}",
        "(1 - ", p_str, ")",
        "^{" + nmk_str + "}",
    )
    parens = VGroup(formula[0], formula[4])
    parens.space_out_submobjects(0.7)
    formula.remove(formula.get_part_by_tex("\\over"))
    pairs = (
        (n_mob, n_str),
        (k_mob, k_str),
        (nmk_mob, nmk_str),
        (p_mob, p_str),
    )
    for mob, tex in pairs:
        parts = formula.get_parts_by_tex(tex)
        for part in parts:
            mob_copy = mob.copy()
            i = formula.index_of_part_by_tex(tex)
            mob_copy.match_height(part)
            mob_copy.move_to(part, DOWN)
            formula.replace_submobject(i, mob_copy)

    terms = VGroup(
        formula[:4],
        formula[4:7],
        formula[7],
        formula[8:11],
        formula[11],
    )
    ys = [term.get_y() for term in terms]
    terms.arrange(RIGHT, buff=SMALL_BUFF)
    terms[0].shift(SMALL_BUFF * LEFT)
    for term, y in zip(terms, ys):
        term.set_y(y)

    return formula


def get_check_count_label(nc, nx, include_rect=True):
    result = VGroup(
        Integer(nc),
        Tex(CMARK_TEX, color=GREEN),
        Integer(nx),
        Tex(XMARK_TEX, color=RED),
    )
    result.arrange(RIGHT, buff=SMALL_BUFF)
    result[2:].shift(SMALL_BUFF * RIGHT)

    if include_rect:
        rect = SurroundingRectangle(result)
        rect.set_stroke(WHITE, 1)
        rect.set_fill(GREY_E, 1)
        result.add_to_back(rect)

    return result


def reverse_smooth(t):
    return smooth(1 - t)


def get_region_under_curve(axes, graph, min_x, max_x):
    props = [
        binary_search(
            function=lambda a: axes.x_axis.p2n(graph.pfp(a)),
            target=x,
            lower_bound=axes.x_min,
            upper_bound=axes.x_max,
        )
        for x in [min_x, max_x]
    ]
    region = graph.copy()
    region.pointwise_become_partial(graph, *props)
    region.add_line_to(axes.c2p(max_x, 0))
    region.add_line_to(axes.c2p(min_x, 0))
    region.add_line_to(region.get_start())

    region.set_stroke(GREEN, 2)
    region.set_fill(GREEN, 0.5)

    region.axes = axes
    region.graph = graph
    region.min_x = min_x
    region.max_x = max_x

    return region
