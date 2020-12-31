# -*- coding: utf-8 -*-
# !/usr/bin/env python

from big_ol_pile_of_manim_imports import *


class ShowHilbertCurve(Scene):
    # To be run with space dimensions
    # DEFAULT_PIXEL_WIDTH = 1080*4
    # DEFAULT_PIXEL_HEIGHT  = 1920*4
    CONFIG = {
        "FractalClass": HilbertCurve,
        "orders": [3, 5, 7],
        "stroke_widths": [20, 15, 7],
    }

    def construct(self):
        curves = VGroup(*[
            self.FractalClass(
                order=order,
            ).scale(scale_factor)
            for order, scale_factor in zip(
                self.orders,
                np.linspace(1, 2, 3)
            )
        ])
        for curve, stroke_width in zip(curves, self.stroke_widths):
            curve.set_stroke(width=stroke_width)
        curves.arrange(DOWN, buff=LARGE_BUFF)
        curves.set_height(FRAME_HEIGHT - 1)
        self.add(*curves)


class ShowFlowSnake(ShowHilbertCurve):
    CONFIG = {
        "FractalClass": FlowSnake,
        "orders": [2, 3, 4],
        "stroke_widths": [20, 15, 10],
    }


class FlippedSierpinski(Sierpinski):
    def __init__(self, *args, **kwargs):
        Sierpinski.__init__(self, *args, **kwargs)
        self.rotate(np.pi, RIGHT, about_point=ORIGIN)


class ShowSierpinski(ShowHilbertCurve):
    CONFIG = {
        "FractalClass": FlippedSierpinski,
        "orders": [3, 6, 9],
        "stroke_widths": [20, 15, 6],
    }


LEXICON = ["Acne", "Acre", "Addendum", "Advertise", "Aircraft", "Aisle", "Alligator", "Alphabetize", "America", "Ankle", "Apathy", "Applause", "Applesauce", "Application", "Archaeologist", "Aristocrat", "Arm", "Armada", "Asleep", "Astronaut", "Athlete", "Atlantis", "Aunt", "Avocado", "Baby-Sitter", "Backbone", "Bag", "Baguette", "Bald", "Balloon", "Banana", "Banister", "Baseball", "Baseboards", "Basketball", "Bat", "Battery", "Beach", "Beanstalk", "Bedbug", "Beer", "Beethoven", "Belt", "Bib", "Bicycle", "Big", "Bike", "Billboard", "Bird", "Birthday", "Bite", "Blacksmith", "Blanket", "Bleach", "Blimp", "Blossom", "Blueprint", "Blunt", "Blur", "Boa", "Boat", "Bob", "Bobsled", "Body", "Bomb", "Bonnet", "Book", "Booth", "Bowtie", "Box", "Boy", "Brainstorm", "Brand", "Brave", "Bride", "Bridge", "Broccoli", "Broken", "Broom", "Bruise", "Brunette", "Bubble", "Buddy", "Buffalo", "Bulb", "Bunny", "Bus", "Buy", "Cabin", "Cafeteria", "Cake", "Calculator", "Campsite", "Can", "Canada", "Candle", "Candy", "Cape", "Capitalism", "Car", "Cardboard", "Cartography", "Cat", "Cd", "Ceiling", "Cell", "Century", "Chair", "Chalk", "Champion", "Charger", "Cheerleader", "Chef", "Chess", "Chew", "Chicken", "Chime", "China", "Chocolate", "Church", "Circus", "Clay", "Cliff", "Cloak", "Clockwork", "Clown", "Clue", "Coach", "Coal", "Coaster", "Cog", "Cold", "College", "Comfort", "Computer", "Cone", "Constrictor", "Continuum", "Conversation", "Cook", "Coop", "Cord", "Corduroy", "Cot", "Cough", "Cow", "Cowboy", "Crayon", "Cream", "Crisp", "Criticize", "Crow", "Cruise", "Crumb", "Crust", "Cuff", "Curtain", "Cuticle", "Czar", "Dad", "Dart", "Dawn", "Day", "Deep", "Defect", "Dent", "Dentist", "Desk", "Dictionary", "Dimple", "Dirty", "Dismantle", "Ditch", "Diver", "Doctor", "Dog", "Doghouse", "Doll", "Dominoes", "Door", "Dot", "Drain", "Draw", "Dream", "Dress", "Drink", "Drip", "Drums", "Dryer", "Duck", "Dump", "Dunk", "Dust", "Ear", "Eat", "Ebony", "Elbow", "Electricity", "Elephant", "Elevator", "Elf", "Elm", "Engine", "England", "Ergonomic", "Escalator", "Eureka", "Europe", "Evolution", "Extension", "Eyebrow", "Fan", "Fancy", "Fast", "Feast", "Fence", "Feudalism", "Fiddle", "Figment", "Finger", "Fire", "First", "Fishing", "Fix", "Fizz", "Flagpole", "Flannel", "Flashlight", "Flock", "Flotsam", "Flower", "Flu", "Flush", "Flutter", "Fog", "Foil", "Football", "Forehead", "Forever", "Fortnight", "France", "Freckle", "Freight", "Fringe", "Frog", "Frown", "Gallop", "Game", "Garbage", "Garden", "Gasoline", "Gem", "Ginger", "Gingerbread", "Girl", "Glasses", "Goblin", "Gold", "Goodbye", "Grandpa", "Grape", "Grass", "Gratitude", "Gray", "Green", "Guitar", "Gum", "Gumball", "Hair", "Half", "Handle", "Handwriting", "Hang", "Happy", "Hat", "Hatch", "Headache", "Heart", "Hedge", "Helicopter", "Hem", "Hide", "Hill", "Hockey", "Homework", "Honk", "Hopscotch", "Horse", "Hose", "Hot", "House", "Houseboat", "Hug", "Humidifier", "Hungry", "Hurdle", "Hurt", "Hut", "Ice", "Implode", "Inn", "Inquisition", "Intern", "Internet", "Invitation", "Ironic", "Ivory", "Ivy", "Jade", "Japan", "Jeans", "Jelly", "Jet", "Jig", "Jog", "Journal", "Jump", "Key", "Killer", "Kilogram", "King", "Kitchen", "Kite", "Knee", "Kneel", "Knife", "Knight", "Koala", "Lace", "Ladder", "Ladybug", "Lag", "Landfill", "Lap", "Laugh", "Laundry", "Law", "Lawn",
           "Lawnmower", "Leak", "Leg", "Letter", "Level", "Lifestyle", "Ligament", "Light", "Lightsaber", "Lime", "Lion", "Lizard", "Log", "Loiterer", "Lollipop", "Loveseat", "Loyalty", "Lunch", "Lunchbox", "Lyrics", "Machine", "Macho", "Mailbox", "Mammoth", "Mark", "Mars", "Mascot", "Mast", "Matchstick", "Mate", "Mattress", "Mess", "Mexico", "Midsummer", "Mine", "Mistake", "Modern", "Mold", "Mom", "Monday", "Money", "Monitor", "Monster", "Mooch", "Moon", "Mop", "Moth", "Motorcycle", "Mountain", "Mouse", "Mower", "Mud", "Music", "Mute", "Nature", "Negotiate", "Neighbor", "Nest", "Neutron", "Niece", "Night", "Nightmare", "Nose", "Oar", "Observatory", "Office", "Oil", "Old", "Olympian", "Opaque", "Opener", "Orbit", "Organ", "Organize", "Outer", "Outside", "Ovation", "Overture", "Pail", "Paint", "Pajamas", "Palace", "Pants", "Paper", "Paper", "Park", "Parody", "Party", "Password", "Pastry", "Pawn", "Pear", "Pen", "Pencil", "Pendulum", "Penis", "Penny", "Pepper", "Personal", "Philosopher", "Phone", "Photograph", "Piano", "Picnic", "Pigpen", "Pillow", "Pilot", "Pinch", "Ping", "Pinwheel", "Pirate", "Plaid", "Plan", "Plank", "Plate", "Platypus", "Playground", "Plow", "Plumber", "Pocket", "Poem", "Point", "Pole", "Pomp", "Pong", "Pool", "Popsicle", "Population", "Portfolio", "Positive", "Post", "Princess", "Procrastinate", "Protestant", "Psychologist", "Publisher", "Punk", "Puppet", "Puppy", "Push", "Puzzle", "Quarantine", "Queen", "Quicksand", "Quiet", "Race", "Radio", "Raft", "Rag", "Rainbow", "Rainwater", "Random", "Ray", "Recycle", "Red", "Regret", "Reimbursement", "Retaliate", "Rib", "Riddle", "Rim", "Rink", "Roller", "Room", "Rose", "Round", "Roundabout", "Rung", "Runt", "Rut", "Sad", "Safe", "Salmon", "Salt", "Sandbox", "Sandcastle", "Sandwich", "Sash", "Satellite", "Scar", "Scared", "School", "Scoundrel", "Scramble", "Scuff", "Seashell", "Season", "Sentence", "Sequins", "Set", "Shaft", "Shallow", "Shampoo", "Shark", "Sheep", "Sheets", "Sheriff", "Shipwreck", "Shirt", "Shoelace", "Short", "Shower", "Shrink", "Sick", "Siesta", "Silhouette", "Singer", "Sip", "Skate", "Skating", "Ski", "Slam", "Sleep", "Sling", "Slow", "Slump", "Smith", "Sneeze", "Snow", "Snuggle", "Song", "Space", "Spare", "Speakers", "Spider", "Spit", "Sponge", "Spool", "Spoon", "Spring", "Sprinkler", "Spy", "Square", "Squint", "Stairs", "Standing", "Star", "State", "Stick", "Stockholder", "Stoplight", "Stout", "Stove", "Stowaway", "Straw", "Stream", "Streamline", "Stripe", "Student", "Sun", "Sunburn", "Sushi", "Swamp", "Swarm", "Sweater", "Swimming", "Swing", "Tachometer", "Talk", "Taxi", "Teacher", "Teapot", "Teenager", "Telephone", "Ten", "Tennis", "Thief", "Think", "Throne", "Through", "Thunder", "Tide", "Tiger", "Time", "Tinting", "Tiptoe", "Tiptop", "Tired", "Tissue", "Toast", "Toilet", "Tool", "Toothbrush", "Tornado", "Tournament", "Tractor", "Train", "Trash", "Treasure", "Tree", "Triangle", "Trip", "Truck", "Tub", "Tuba", "Tutor", "Television", "Twang", "Twig", "Twitterpated", "Type", "Unemployed", "Upgrade", "Vest", "Vision", "Wag", "Water", "Watermelon", "Wax", "Wedding", "Weed", "Welder", "Whatever", "Wheelchair", "Whiplash", "Whisk", "Whistle", "White", "Wig", "Will", "Windmill", "Winter", "Wish", "Wolf", "Wool", "World", "Worm", "Wristwatch", "Yardstick", "Zamboni", "Zen", "Zero", "Zipper", "Zone", "Zoo", ]


class ShowWordGrid(Scene):
    CONFIG = {
        "num_blue_agents": 10,
        "num_red_agents": 9,
        "assassin": "",
        "random_seed": 1,
    }

    def construct(self):
        self.string_to_tex_mob = {}
        self.setup_grid()
        self.get_master_layout()

    def setup_grid(self):
        word_matrix = self.get_word_matrix()
        tex_mob_matrix = np.array([
            list(map(TextMobject, row))
            for row in word_matrix
        ])
        n_rows, n_cols = tex_mob_matrix.shape
        vert_shift = (2.0 * FRAME_Y_RADIUS - 1) / n_rows
        horiz_shift = (FRAME_WIDTH - 1) / n_cols
        rects = VGroup()
        for y, row in enumerate(tex_mob_matrix):
            for x, tex_mob in enumerate(row):
                tex_mob.shift(
                    (y - n_rows / 2) * vert_shift * UP +
                    (x - n_cols / 2) * horiz_shift * RIGHT
                )
                tex_mob.scale_in_place(0.8)
                self.add(tex_mob)
                rect = Rectangle()
                rect.replace(tex_mob, stretch=True)
                rect.set_stroke(width=0)
                rects.add(rect)
                tex_mob.rect = rect
                word = tex_mob.get_tex_string().split(" ")[-1]
                word = word.lower()
                self.string_to_tex_mob[word] = tex_mob
        VGroup(rects, *tex_mob_matrix.flatten()).center()

    def get_word_matrix(self):
        return np.array(random.sample(LEXICON, 25)).reshape((5, 5))

    def cover_words(self, *words, **kwargs):
        if self.assassin.lower() in [w.lower() for w in words]:
            for x in range(3):
                play_error_sound()
        rects = VGroup(*[
            self.string_to_tex_mob[word.lower()].rect
            for word in words
        ])
        if "color" in kwargs:
            rects.set_fill(kwargs["color"], opacity=0.6)
        self.add(rects)
        return self

    def get_master_layout(self):
        words = set(self.string_to_tex_mob.keys())
        blue_agents = random.sample(words, self.num_blue_agents)
        words.difference_update(blue_agents)
        red_agents = random.sample(words, self.num_red_agents)
        words.difference_update(red_agents)
        assassin = random.choice(list(words))
        words.difference_update([assassin])
        bystandards = words

        self.cover_words(*blue_agents, color=BLUE)
        self.cover_words(*red_agents, color=RED)
        self.cover_words(*bystandards, color=YELLOW)
        self.cover_words(assassin, color=GREY)
        self.update_frame()
        self.save_image(name="MasterLayout")
        for mob in list(self.string_to_tex_mob.values()):
            self.remove(mob.rect)

        self.assassin = assassin

    def show(self):
        self.show_frame()


eight_dollar_patrons_names = [
    "Janel",
]


class EightDollarPatronAnimation(ComplexTransformationScene):
    CONFIG = {
        "patron_name": "Janel",
        "function": lambda z: 0.2 * (z**3),
        "default_apply_complex_function_kwargs": {
            "run_time": 5,
        },
        "output_directory": os.path.join(VIDEO_DIR, "EightDollarPatrons"),
        "include_coordinate_labels": False,
        "vert_start_color": YELLOW,  # TODO
        "vert_end_color": PINK,
        "horiz_start_color": GREEN,
        "horiz_end_color": BLUE,
        "use_multicolored_plane": True,
        # "plane_config" : {
        #     "unit_size" : 1.5,
        # },
    }

    def construct(self):
        name = self.patron_name
        self.clear()
        self.frames = []
        self.setup()
        self.add_transformable_plane()
        self.plane.fade()

        name_mob = TextMobject(name)
        name_mob.set_width(4)
        name_mob.next_to(ORIGIN, UP, LARGE_BUFF)
        self.start_vect = name_mob.get_center()
        for submob in name_mob.family_members_with_points():
            submob.insert_n_curves(100)
        name_mob_copy = name_mob.copy()

        self.play(Write(name_mob))
        self.play(
            self.get_rotation(name_mob),
            run_time=5,
        )
        self.wait()
        self.add_transformable_mobjects(name_mob)
        self.apply_complex_function(self.function)
        self.wait()
        self.play(
            self.get_post_transform_rotation(name_mob, name_mob_copy),
            run_time=10
        )
        self.wait(3)

    def get_rotation(self, name_mob):
        return UpdateFromAlphaFunc(
            name_mob,
            lambda mob, alpha: mob.move_to(rotate_vector(
                self.start_vect, 2 * np.pi * alpha
            ))
        )

    def get_post_transform_rotation(self, name_mob, name_mob_copy):
        simple_rotation = self.get_rotation(name_mob_copy)

        def update(name_mob, alpha):
            simple_rotation.update(alpha)
            new_name = simple_rotation.mobject.copy()
            new_name.apply_complex_function(self.function)
            Transform(name_mob, new_name).update(1)
            return name_mob
        return UpdateFromAlphaFunc(name_mob, update)


if __name__ == "__main__":
    for name in eight_dollar_patrons_names:
        scene = EightDollarPatronAnimation(
            name=name,
            write_to_movie=True
        )
