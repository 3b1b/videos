from manim_imports_ext import *


# Related to pi creatures
class Car(SVGMobject):
    CONFIG = {
        "file_name": "Car",
        "height": 1,
        "color": GREY_B,
        "light_colors": [BLACK, BLACK],
    }

    def __init__(self, **kwargs):
        SVGMobject.__init__(self, **kwargs)

        path = self.submobjects[0]
        subpaths = path.get_subpaths()
        path.clear_points()
        for indices in [(0, 1), (2, 3), (4, 6, 7), (5,), (8,)]:
            part = VMobject()
            for index in indices:
                part.append_points(subpaths[index])
            path.add(part)

        self.set_height(self.height)
        self.set_stroke(color=WHITE, width=0)
        self.set_fill(self.color, opacity=1)

        from videos.characters.pi_creature import Randolph
        randy = Randolph(mode="happy")
        randy.set_height(0.6 * self.get_height())
        randy.stretch(0.8, 0)
        randy.look(RIGHT)
        randy.move_to(self)
        randy.shift(0.07 * self.height * (RIGHT + UP))
        self.randy = self.pi_creature = randy
        self.add_to_back(randy)

        orientation_line = Line(self.get_left(), self.get_right())
        orientation_line.set_stroke(width=0)
        self.add(orientation_line)
        self.orientation_line = orientation_line

        for light, color in zip(self.get_lights(), self.light_colors):
            light.set_fill(color, 1)
            light.is_subpath = False

        self.add_treds_to_tires()

    def move_to(self, point_or_mobject):
        vect = rotate_vector(
            UP + LEFT, self.orientation_line.get_angle()
        )
        self.next_to(point_or_mobject, vect, buff=0)
        return self

    def get_front_line(self):
        return DashedLine(
            self.get_corner(UP + RIGHT),
            self.get_corner(DOWN + RIGHT),
            color=DISTANCE_COLOR,
            dash_length=0.05,
        )

    def add_treds_to_tires(self):
        for tire in self.get_tires():
            radius = tire.get_width() / 2
            center = tire.get_center()
            tred = Line(
                0.7 * radius * RIGHT, 1.1 * radius * RIGHT,
                stroke_width=2,
                color=BLACK
            )
            tred.rotate(PI / 5, about_point=tred.get_end())
            for theta in np.arange(0, 2 * np.pi, np.pi / 4):
                new_tred = tred.copy()
                new_tred.rotate(theta, about_point=ORIGIN)
                new_tred.shift(center)
                tire.add(new_tred)
        return self

    def get_tires(self):
        return VGroup(self[1][0], self[1][1])

    def get_lights(self):
        return VGroup(self.get_front_light(), self.get_rear_light())

    def get_front_light(self):
        return self[1][3]

    def get_rear_light(self):
        return self[1][4]


class MoveCar(ApplyMethod):
    CONFIG = {
        "moving_forward": True,
        "run_time": 5,
    }

    def __init__(self, car, target_point, **kwargs):
        self.check_if_input_is_car(car)
        self.target_point = target_point
        super().__init__(car.move_to, target_point, **kwargs)

    def check_if_input_is_car(self, car):
        if not isinstance(car, Car):
            raise Exception("MoveCar must take in Car object")

    def begin(self):
        super().begin()
        car = self.mobject
        distance = get_norm(op.sub(
            self.target_mobject.get_right(),
            self.starting_mobject.get_right(),
        ))
        if not self.moving_forward:
            distance *= -1
        tire_radius = car.get_tires()[0].get_width() / 2
        self.total_tire_radians = -distance / tire_radius

    def interpolate_mobject(self, alpha):
        ApplyMethod.interpolate_mobject(self, alpha)
        if alpha == 0:
            return
        radians = alpha * self.total_tire_radians
        for tire in self.mobject.get_tires():
            tire.rotate(radians)


class PartyHat(SVGMobject):
    CONFIG = {
        "file_name": "party_hat",
        "height": 1.5,
        "pi_creature": None,
        "stroke_width": 0,
        "fill_opacity": 1,
        "frills_colors": [MAROON_B, PURPLE],
        "cone_color": GREEN,
        "dots_colors": [YELLOW],
    }
    NUM_FRILLS = 7
    NUM_DOTS = 6

    def __init__(self, **kwargs):
        SVGMobject.__init__(self, **kwargs)
        self.set_height(self.height)
        if self.pi_creature is not None:
            self.next_to(self.pi_creature.eyes, UP, buff=0)

        self.frills = VGroup(*self[:self.NUM_FRILLS])
        self.cone = self[self.NUM_FRILLS]
        self.dots = VGroup(*self[self.NUM_FRILLS + 1:])

        self.frills.set_color_by_gradient(*self.frills_colors)
        self.cone.set_color(self.cone_color)
        self.dots.set_color_by_gradient(*self.dots_colors)


class SunGlasses(SVGMobject):
    CONFIG = {
        "file_name": "sunglasses",
        "glasses_width_to_eyes_width": 1.1,
    }

    def __init__(self, pi_creature, **kwargs):
        SVGMobject.__init__(self, **kwargs)
        self.set_stroke(WHITE, width=0)
        self.set_fill(GREY, 1)
        self.set_width(
            self.glasses_width_to_eyes_width * pi_creature.eyes.get_width()
        )
        self.move_to(pi_creature.eyes, UP)


class Headphones(SVGMobject):
    CONFIG = {
        "file_name": "headphones",
        "height": 2,
        "y_stretch_factor": 0.5,
        "color": GREY,
    }

    def __init__(self, pi_creature=None, **kwargs):
        digest_config(self, kwargs)
        SVGMobject.__init__(self, file_name=self.file_name, **kwargs)
        self.stretch(self.y_stretch_factor, 1)
        self.set_height(self.height)
        self.set_stroke(width=0)
        self.set_fill(color=self.color)
        if pi_creature is not None:
            eyes = pi_creature.eyes
            self.set_height(3 * eyes.get_height())
            self.move_to(eyes, DOWN)
            self.shift(DOWN * eyes.get_height() / 4)


class Guitar(SVGMobject):
    CONFIG = {
        "file_name": "guitar",
        "height": 2.5,
        "fill_color": GREY_D,
        "fill_opacity": 1,
        "stroke_color": WHITE,
        "stroke_width": 0.5,
    }


# Cards
class DeckOfCards(VGroup):
    def __init__(self, **kwargs):
        possible_values = list(map(str, list(range(1, 11)))) + ["J", "Q", "K"]
        possible_suits = ["hearts", "diamonds", "spades", "clubs"]
        VGroup.__init__(self, *[
            PlayingCard(value=value, suit=suit, **kwargs)
            for value in possible_values
            for suit in possible_suits
        ])


class PlayingCard(VGroup):
    CONFIG = {
        "value": None,
        "suit": None,
        "key": None,  # String like "8H" or "KS"
        "height": 2,
        "height_to_width": 3.5 / 2.5,
        "card_height_to_symbol_height": 7,
        "card_width_to_corner_num_width": 10,
        "card_height_to_corner_num_height": 10,
        "color": GREY_A,
        "turned_over": False,
        "possible_suits": ["hearts", "diamonds", "spades", "clubs"],
        "possible_values": list(map(str, list(range(2, 11)))) + ["J", "Q", "K", "A"],
    }

    def __init__(self, key=None, **kwargs):
        VGroup.__init__(self, **kwargs)

        self.key = key
        self.add(Rectangle(
            height=self.height,
            width=self.height / self.height_to_width,
            stroke_color=WHITE,
            stroke_width=2,
            fill_color=self.color,
            fill_opacity=1,
        ))
        if self.turned_over:
            self.set_fill(GREY_D)
            self.set_stroke(GREY_B)
            contents = VectorizedPoint(self.get_center())
        else:
            value = self.get_value()
            symbol = self.get_symbol()
            design = self.get_design(value, symbol)
            corner_numbers = self.get_corner_numbers(value, symbol)
            contents = VGroup(design, corner_numbers)
            self.design = design
            self.corner_numbers = corner_numbers
        self.add(contents)

    def get_value(self):
        value = self.value
        if value is None:
            if self.key is not None:
                value = self.key[:-1]
            else:
                value = random.choice(self.possible_values)
        value = str(value).upper()
        if value == "1":
            value = "A"
        if value not in self.possible_values:
            raise Exception("Invalid card value")

        face_card_to_value = {
            "J": 11,
            "Q": 12,
            "K": 13,
            "A": 14,
        }
        try:
            self.numerical_value = int(value)
        except Exception:
            self.numerical_value = face_card_to_value[value]
        return value

    def get_symbol(self):
        suit = self.suit
        if suit is None:
            if self.key is not None:
                suit = dict([
                    (s[0].upper(), s)
                    for s in self.possible_suits
                ])[self.key[-1].upper()]
            else:
                suit = random.choice(self.possible_suits)
        if suit not in self.possible_suits:
            raise Exception("Invalud suit value")
        self.suit = suit
        symbol_height = float(self.height) / self.card_height_to_symbol_height
        symbol = SuitSymbol(suit, height=symbol_height)
        return symbol

    def get_design(self, value, symbol):
        if value == "A":
            return self.get_ace_design(symbol)
        if value in list(map(str, list(range(2, 11)))):
            return self.get_number_design(value, symbol)
        else:
            return self.get_face_card_design(value, symbol)

    def get_ace_design(self, symbol):
        design = symbol.copy().scale(1.5)
        design.move_to(self)
        return design

    def get_number_design(self, value, symbol):
        num = int(value)
        n_rows = {
            2: 2,
            3: 3,
            4: 2,
            5: 2,
            6: 3,
            7: 3,
            8: 3,
            9: 4,
            10: 4,
        }[num]
        n_cols = 1 if num in [2, 3] else 2
        insertion_indices = {
            5: [0],
            7: [0],
            8: [0, 1],
            9: [1],
            10: [0, 2],
        }.get(num, [])

        top = self.get_top() + symbol.get_height() * DOWN
        bottom = self.get_bottom() + symbol.get_height() * UP
        column_points = [
            interpolate(top, bottom, alpha)
            for alpha in np.linspace(0, 1, n_rows)
        ]

        design = VGroup(*[
            symbol.copy().move_to(point)
            for point in column_points
        ])
        if n_cols == 2:
            space = 0.2 * self.get_width()
            column_copy = design.copy().shift(space * RIGHT)
            design.shift(space * LEFT)
            design.add(*column_copy)
        design.add(*[
            symbol.copy().move_to(
                center_of_mass(column_points[i:i + 2])
            )
            for i in insertion_indices
        ])
        for symbol in design:
            if symbol.get_center()[1] < self.get_center()[1]:
                symbol.rotate(np.pi)
        return design

    def get_face_card_design(self, value, symbol):
        from videos.characters.pi_creature import PiCreature
        sub_rect = Rectangle(
            stroke_color=BLACK,
            fill_opacity=0,
            height=0.9 * self.get_height(),
            width=0.6 * self.get_width(),
        )
        sub_rect.move_to(self)

        # pi_color = average_color(symbol.get_color(), GREY)
        pi_color = symbol.get_color()
        if Color(pi_color) == Color(BLACK):
            pi_color = GREY_D
        pi_mode = {
            "J": "plain",
            "Q": "thinking",
            "K": "hooray"
        }[value]
        pi_creature = PiCreature(
            mode=pi_mode,
            color=pi_color,
        )
        pi_creature.set_width(0.8 * sub_rect.get_width())
        if value in ["Q", "K"]:
            prefix = "king" if value == "K" else "queen"
            crown = SVGMobject(file_name=prefix + "_crown")
            crown.set_stroke(width=0)
            crown.set_fill(YELLOW, 1)
            crown.stretch_to_fit_width(0.5 * sub_rect.get_width())
            crown.stretch_to_fit_height(0.17 * sub_rect.get_height())
            crown.move_to(pi_creature.eyes.get_center(), DOWN)
            pi_creature.add_to_back(crown)
            to_top_buff = 0
        else:
            to_top_buff = SMALL_BUFF * sub_rect.get_height()
        pi_creature.next_to(sub_rect.get_top(), DOWN, to_top_buff)
        # pi_creature.shift(0.05*sub_rect.get_width()*RIGHT)

        pi_copy = pi_creature.copy()
        pi_copy.rotate(np.pi, about_point=sub_rect.get_center())

        return VGroup(sub_rect, pi_creature, pi_copy)

    def get_corner_numbers(self, value, symbol):
        value_mob = TexText(value)
        width = self.get_width() / self.card_width_to_corner_num_width
        height = self.get_height() / self.card_height_to_corner_num_height
        value_mob.set_width(width)
        value_mob.stretch_to_fit_height(height)
        value_mob.next_to(
            self.get_corner(UP + LEFT), DOWN + RIGHT,
            buff=MED_LARGE_BUFF * width
        )
        value_mob.set_color(symbol.get_color())
        corner_symbol = symbol.copy()
        corner_symbol.set_width(width)
        corner_symbol.next_to(
            value_mob, DOWN,
            buff=MED_SMALL_BUFF * width
        )
        corner_group = VGroup(value_mob, corner_symbol)
        opposite_corner_group = corner_group.copy()
        opposite_corner_group.rotate(
            np.pi, about_point=self.get_center()
        )

        return VGroup(corner_group, opposite_corner_group)


class SuitSymbol(SVGMobject):
    CONFIG = {
        "height": 0.5,
        "fill_opacity": 1,
        "stroke_width": 0,
        "red": "#D02028",
        "black": BLACK,
    }

    def __init__(self, suit_name, **kwargs):
        digest_config(self, kwargs)
        suits_to_colors = {
            "hearts": self.red,
            "diamonds": self.red,
            "spades": self.black,
            "clubs": self.black,
        }
        if suit_name not in suits_to_colors:
            raise Exception("Invalid suit name")
        SVGMobject.__init__(self, file_name=suit_name, **kwargs)

        color = suits_to_colors[suit_name]
        self.set_stroke(width=0)
        self.set_fill(color, 1)
        self.set_height(self.height)


# Logos
class AoPSLogo(SVGMobject):
    CONFIG = {
        "file_name": "aops_logo",
        "height": 1.5,
    }

    def __init__(self, **kwargs):
        SVGMobject.__init__(self, **kwargs)
        self.set_stroke(WHITE, width=0)
        colors = [BLUE_E, "#008445", GREEN_B]
        index_lists = [
            (10, 11, 12, 13, 14, 21, 22, 23, 24, 27, 28, 29, 30),
            (0, 1, 2, 3, 4, 15, 16, 17, 26),
            (5, 6, 7, 8, 9, 18, 19, 20, 25)
        ]
        for color, index_list in zip(colors, index_lists):
            for i in index_list:
                self.submobjects[i].set_fill(color, opacity=1)

        self.set_height(self.height)
        self.center()


class BitcoinLogo(SVGMobject):
    CONFIG = {
        "file_name": "Bitcoin_logo",
        "height": 1,
        "fill_color": "#f7931a",
        "inner_color": WHITE,
        "fill_opacity": 1,
        "stroke_width": 0,
    }

    def __init__(self, **kwargs):
        SVGMobject.__init__(self, **kwargs)
        self[0].set_fill(self.fill_color, self.fill_opacity)
        self[1].set_fill(self.inner_color, 1)
