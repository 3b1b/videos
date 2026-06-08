from manim_imports_ext import *
import json

class LanguageTextExample(VGroup):
    def __init__(self, example_text = "Lorem Ipsum"):
        super().__init__()
        self.file_icon = SVGMobject("../images/file_icon.svg").set_width(1)
        self.example_text = Text(
            example_text, font = "CMU Serif", font_size = 25
        ).next_to(self.file_icon, DOWN).set_color("#BBBBBB").set_stroke(width = 4, color = BLACK, behind = True)
        self.add(self.file_icon, self.example_text)


class BuildTree(InteractiveScene):
    def construct(self):
        # Instantiate a list of canonical texts
        texts = {
            "Czech": [
                "Babička",
                "R.U.R.",
                "Máj"
            ],
            "English": [
                "Hamlet",
                "Paradise Lost",
                "Emma",
                "1984"
            ],
            "French": [
                "Les Misérables",
                "Le Petit Prince"
            ],
            "Scottish Gaelic": [
                "Dàin do Eimhir",
                "An t-Ogha Mór",
                "An Sgàthach"
            ],
            "German": [
                "Faust",
                "Die Verwandlung"
            ],
            "Italian": [
                "Divina Commedia",
                "Il Principe",
                "I Promessi Sposi",
                "Il Gattopardo"
            ],
            "Polish": [
                "Pan Tadeusz",
                "Quo Vadis",
                "Ferdydurke"
            ],
            "Spanish": [
                "Don Quijote",
                "Ficciones",
                "Rayuela",
                "Bodas de sangre"
            ],
            "Swedish": [
                "Fröken Julie",
                "Pippi Långstrump"
            ],
            "Welsh": [
                "Y Gododdin",
                "Un Nos Ola Leuad",
                "Chwalfa"
            ]
        }

        # Create the file icons for each example
        bunches = VGroup()
        for language in texts.keys():
            r = 1
            theta = random.uniform(0, 2*PI)
            bunch = VGroup()
            for text in texts[language]:
                file = LanguageTextExample(text).scale(0.7).shift(RIGHT*r*math.cos(theta) + UP*r*math.sin(theta))
                bunch.add(file)
                theta += 2*PI/len(texts[language])
            bunches.add(bunch)
        bunches.arrange_in_grid(n_cols = 5, v_buff = 2, h_buff = 0.8).set_width(FRAME_WIDTH*0.95)
        # self.add(bunches)
        # for bunch in bunches:
        #     for file in bunch:
        #         self.bring_to_front(file.example_text)
        all_files = []
        for bunch in bunches:
            for file in bunch:
                all_files.append(file)
        random.shuffle(all_files)

        self.play(
            AnimationGroup(*[
                FadeIn(file, shift = UP*0.3)
                for file in all_files
            ], lag_ratio = 0.1)
        , run_time = 3)

        # Label each bunch with its language
        braces = VGroup()
        labels = VGroup()
        for i, bunch in enumerate(bunches):
            brace = Brace(bunch, DOWN).set_color("#888888")
            name = list(texts.keys())[i]
            label = SVGMobject(F"flags/{name}.svg").set_width(0.5).next_to(brace, DOWN)
            label.name = name
            if 1 <= i < 5:
                VGroup(brace, label).align_to(braces[0], UP)
            if 6 <= i < 10:
                VGroup(brace, label).align_to(braces[5], UP)
            braces.add(brace)
            labels.add(label)
        self.play(
            self.camera.frame.animate.match_y(VGroup(*all_files, braces, labels)),
            AnimationGroup(*[
                AnimationGroup(
                    GrowFromEdge(brace, UP),
                    FadeIn(label, shift = DOWN*0.1)
                , lag_ratio = 0.1)
                for brace, label in zip(braces, labels)
            ], lag_ratio = 0.1)
        )
        self.camera.frame.center()
        VGroup(*all_files, braces, labels).center()

        # Create the full tree of language lineage
        UNNAMED_COLOR = WHITE
        colors = [UNNAMED_COLOR, BLUE_D, UNNAMED_COLOR, GREEN_B, UNNAMED_COLOR, TEAL_D, BLUE_B, GREEN_D, UNNAMED_COLOR, TEAL_B, BLUE_E, UNNAMED_COLOR]
        with open('language_families.json', 'r', encoding='utf-8') as file:
            families_data = json.load(file)["families"]
        with open('languages.txt', 'r', encoding='utf-8') as file:
            leaves_data = file.read().splitlines()
        families = VGroup()
        leaves = VGroup()
        for family_data, color in zip(families_data, colors):
            family = VGroup()
            for language in family_data["languages"]:
                name_text = Text(language["name"]).set_color(color)
                family.add(name_text)
                leaves.add(name_text)
            families.add(family)
        leaves.arrange(DOWN)
        for leaf in leaves:
            leaf.align_to(leaves[0], LEFT)
        leaves.set_height(FRAME_HEIGHT*0.95)
        family_braces = VGroup(*[
            Brace(family, RIGHT).set_color(color)
            for family, color in zip(families, colors)
        ]).shift(RIGHT*0.1)
        for brace in family_braces:
            brace.align_to(family_braces[0], LEFT)
        family_labels = VGroup(*[
            brace.get_text(family_data["name"], font_size = 20).set_color(color)
            for brace, family_data, color in zip(family_braces, families_data, colors)
        ])
        for i in range(len(colors)):
            if colors[i] == UNNAMED_COLOR:
                VGroup(family_braces[i], family_labels[i]).set_opacity(0)
        VGroup(leaves, family_braces, family_labels).to_edge(RIGHT, buff = 1)

        # Transform the file icons into the tree
        tree = LanguageTree("tree_layout_adjusted.json", node_radius = 0.02).next_to(leaves, LEFT, buff = 0.08)
        leaf_nodes = tree.node_group[:len(leaves)]
        tree.shift(DOWN*(leaf_nodes.get_y() - leaves.get_y()))

        self.play(
            AnimationGroup(
                AnimationGroup(*[
                    AnimationGroup(
                        VGroup(bunch, brace).animate(run_time = 1.5).scale(0.001).move_to(leaves[leaves_data.index(label.name)]),
                        TransformMatchingShapes(label, leaves[leaves_data.index(label.name)], run_time = 1.5)
                    )
                    for bunch, brace, label in zip(bunches, braces, labels)
                ], lag_ratio = 0.08),
                AnimationGroup(*[
                    FadeIn(leaves[i])
                    for i in range(len(leaves)) if i not in [leaves_data.index(label.name) for label in labels]
                ], lag_ratio = 0.01)
            , lag_ratio = 0.6)
        )
        self.remove(bunches, braces)
        self.wait(1)

        self.play(
            AnimationGroup(
                FadeIn(tree.edge_group, lag_ratio = 0.1, run_time = 5),
                FadeIn(tree.node_group, run_time = 1.5)
            , lag_ratio = 0.9)
        )


class LanguageTree(VGroup):
    def __init__(
            self,
            json_path,
            node_radius=0.1,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.json_path = json_path
        self.node_radius = node_radius
        
        with open(json_path, "r") as f:
            self.graph_data = json.load(f)
            
        self.node_group = VGroup()
        self.edge_group = VGroup()
        self.nodes_dict = {} 
        self.edges_dict = {}
        
        self.create_nodes()
        self.create_edges()
        
        self.add(self.edge_group, self.node_group)
        self.center()

    def get_node(self, label):
        return self.nodes_dict[label]

    def get_edge(self, u_label, v_label):
        return self.edges_dict[(u_label, v_label)]


    def create_nodes(self, show_node_labels = True):
        for node_data in self.graph_data["nodes"]:
            label_text = node_data["label"]
            pos = np.array(node_data["position"])
            
            node = Dot(pos, radius=self.node_radius).set_color(WHITE)
            self.nodes_dict[label_text] = node
            self.node_group.add(node)

    def create_edges(self):
        for edge_data in self.graph_data["edges"]:
            u_label = edge_data["start_node"]
            v_label = edge_data["end_node"]
            angle = edge_data["angle"]
            
            edge = ArcBetweenPoints(ORIGIN, RIGHT, angle=angle).set_color("#BBBBBB").set_stroke(width=1)
            
            def update_edge(m, u_l=u_label, v_l=v_label, ang=angle):
                u = self.nodes_dict[u_l]
                v = self.nodes_dict[v_l]
                
                u_p, v_p = self.get_anchored_points(
                    u.get_center(), 
                    v.get_center(), 
                    ang,
                    u.get_width()/2,
                    v.get_width()/2
                )
                
                # Create a temporary arrow to steal the path points from
                new_arc = CurvedArrow(u.get_center(), v.get_center(), angle=ang).set_color("#BBBBBB").set_stroke(width=1)
                edge.put_start_and_end_on(new_arc.get_start(), new_arc.get_end())
                

            edge.u_label = u_label
            edge.v_label = v_label

            edge.add_updater(update_edge)
            self.edge_group.add(edge)


    def get_anchored_points(self, u_c, v_c, angle, u_r, v_r):
        u_rad = u_r + (u_r/self.node_radius)*0.1
        v_rad = v_r + (v_r/self.node_radius)*0.1
        temp_arc = ArcBetweenPoints(u_c, v_c, angle=angle)
        total_len = temp_arc.get_arc_length()
        if total_len < (u_rad + v_rad):
            return u_c, v_c
        return temp_arc.point_from_proportion(u_rad / total_len), temp_arc.point_from_proportion(1 - v_rad / total_len)

    # def create(self):
    #     return AnimationGroup(
    #         AnimationGroup(*[
    #             AnimationGroup(
    #                 Pop(node.dot),
    #                 GrowFromEdgeWarp(node.label, DOWN, run_up_dist=0.1, fade_in=True) if node.label else DoNothing()
    #             )
    #             for node in self.node_group
    #         ], lag_ratio=0.1),
    #         AnimationGroup(*[
    #             AnimationGroup(
    #                 GrowArrow(edge.arrow),
    #                 GrowFromEdgeWarp(edge.weight_label, DOWN, run_up_dist=0.1, fade_in=True) if edge.weight_label else DoNothing()
    #             )
    #             for edge in self.edge_group
    #         ], lag_ratio=0.1)
    #     , lag_ratio=0.4)