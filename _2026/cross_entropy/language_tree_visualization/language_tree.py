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


class BuildTreeV2(InteractiveScene):
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
                FadeIn(tree.edge_group, lag_ratio = 0.1, run_time = 10),
                FadeIn(tree.node_group, run_time = 3)
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


class BasicIdea(InteractiveScene):
    def construct(self):
        # Add two documents
        document1 = VGroup(*[
            Rectangle(
                width = 5,
                height = 1,
                fill_opacity = 1,
                fill_color = interpolate_color(TEAL_B, TEAL_E, random.random()),
                stroke_width = 1,
                stroke_color = WHITE
            )
            for _ in range(8)
        ]).arrange(DOWN, buff = 0)
        document1_label = TexText("Document A", font_size = 80).next_to(document1, UP, buff = 0.4)
        document2 = document1.copy()
        for rect in document2:
            rect.set_fill(color = interpolate_color(GREEN_B, GREEN_E, random.random()))
        document2_label = TexText("Document B", font_size = 80).next_to(document2, UP, buff = 0.4)
        document1_group = VGroup(document1, document1_label)
        document2_group = VGroup(document2, document2_label)
        documents = VGroup(document1_group, document2_group)
        documents.arrange(buff = 2).set_height(4)
        self.play(
            AnimationGroup(*[
                FadeIn(doc, shift = UP*0.5)
                for doc in documents
            ], lag_ratio = 0.4)
        )
        self.play(documents.animate.scale(0.9).to_edge(LEFT, buff = 1))

        # Take a small snippet of B and append it to a copy of A
        a_copy = document1.copy()
        snippet = document2[4:7].copy()
        new_doc = VGroup(a_copy, snippet).arrange(DOWN, buff = 0).set_x(0.5*(FRAME_WIDTH*0.5 + document2.get_right()[0]))
        self.play(
            AnimationGroup(
                TransformFromCopy(document2[4:7], snippet, path_arc = PI*0.3),
                TransformFromCopy(document1, a_copy, path_arc = -PI*0.3)
            , lag_ratio = 0.9)
        , run_time = 4)

        # "Compress" it
        new_doc.generate_target()
        new_doc.target.set_stroke(width = 0)
        new_doc.target[0].stretch(0.15, 1)
        new_doc.target[1].stretch(0.25, 1)
        new_doc.target.arrange(DOWN, buff = 0).set_width(2.2).move_to(new_doc)
        gzip_ab = TexText("GZIP(AB)", font_size = 35, tex_to_color_map = {"A": BLUE, "B": GREEN}).next_to(new_doc.target, UP)
        self.play(AnimationGroup(MoveToTarget(new_doc), Write(gzip_ab), lag_ratio = 0.8))
        self.wait(1)

        # Compress A on its own
        a_copy_2 = document1.copy().match_x(new_doc).to_edge(DOWN, buff = 1)
        self.play(
            VGroup(new_doc, gzip_ab).animate.to_edge(UP, buff = 2),
            TransformFromCopy(document1, a_copy_2, path_arc = PI*0.3)
        , run_time = 2)
        self.wait(0.5)
        a_copy_2.generate_target()
        a_copy_2.target.set_stroke(width = 0).stretch(0.15, 1).set_width(2.2)
        gzip_a = TexText("GZIP(A)", font_size = 35, tex_to_color_map = {"A": BLUE, "B": TEAL}).next_to(a_copy_2.target, UP)
        self.play(AnimationGroup(MoveToTarget(a_copy_2), Write(gzip_a), lag_ratio = 0.8))
        self.wait(2)

        # Compare the sizes
        compressed_docs_group = VGroup(VGroup(gzip_ab, new_doc), VGroup(gzip_a, a_copy_2))
        compressed_docs_group.generate_target()
        compressed_docs_group.target.scale(1.3).arrange(buff = 3)
        compressed_docs_group.target[1].align_to(compressed_docs_group.target[0], UP)
        self.play(
            FadeOut(VGroup(document1_group, document2_group), shift = LEFT*4),
            MoveToTarget(compressed_docs_group, path_arc = PI*0.3)
        , run_time = 1.5)
        difference_equation = VGroup(
            Line(ORIGIN, UP*4).next_to(compressed_docs_group[0], LEFT, buff = 0.35),
            Line(ORIGIN, UP*4).next_to(compressed_docs_group[0], RIGHT, buff = 0.35),
            Tex("-", font_size = 90),
            Line(ORIGIN, UP*4).next_to(compressed_docs_group[1], LEFT, buff = 0.35).set_y(0),
            Line(ORIGIN, UP*4).next_to(compressed_docs_group[1], RIGHT, buff = 0.35).set_y(0)
        )
        self.play(Write(difference_equation))
        self.wait(2)

        # Highlight the snippet of B
        self.play(AnimationGroup(*[Indicate(rect, scale_factor = 1.05) for rect in new_doc[1]], lag_ratio = 0.1), run_time = 2)
        self.wait(1)

        # Highlight the main part of A
        self.play(AnimationGroup(*[Indicate(rect, scale_factor = 1.05) for rect in new_doc[0]], lag_ratio = 0.1), run_time = 2)
        self.wait(1)

        # Decrease the "linguistic difference"
        new_doc.generate_target()
        new_doc.target[1].stretch(0.75, 1).next_to(new_doc.target[0], DOWN, buff = 0)
        for rect in new_doc.target[1]:
            rect.set_fill(color = interpolate_color(BLUE_A, BLUE_E, random.random()))
        self.play(MoveToTarget(new_doc), run_time = 2)
        self.wait(2)

        # Increase the "linguistic difference"
        new_doc.generate_target()
        new_doc.target[1].stretch(3, 1).next_to(new_doc.target[0], DOWN, buff = 0)
        for rect in new_doc.target[1]:
            rect.set_fill(color = interpolate_color(RED_A, RED_E, random.random()))
        self.play(MoveToTarget(new_doc), run_time = 2)
        self.wait(2)


from scipy.spatial import ConvexHull
class SurroundingEllipse(Ellipse):
    def __init__(
        self,
        mobject,
        buff=0.2,
        color=YELLOW,
        **kwargs
    ):
        super().__init__(color=color, **kwargs)
        self.buff = buff
        self.surround(mobject)

    def surround(self, mobject, buff=None):
        self.mobject = mobject
        if buff is not None:
            self.buff = buff
            
        points = mobject.get_all_points()
        if len(points) < 2:
            base_ellipse = Ellipse(
                width=mobject.get_width() + 2 * self.buff,
                height=mobject.get_height() + 2 * self.buff
            )
            base_ellipse.move_to(mobject.get_center())
            self.set_points(base_ellipse.get_points())
            return self

        P = np.unique(points[:, :2], axis=0)
        
        centroid = np.mean(P, axis=0)
        eps = 1e-4
        dummy_points = np.array([
            [centroid[0] - eps, centroid[1] - eps],
            [centroid[0] + eps, centroid[1] - eps],
            [centroid[0] + eps, centroid[1] + eps],
            [centroid[0] - eps, centroid[1] + eps]
        ])
        P = np.vstack([P, dummy_points])

        try:
            hull = ConvexHull(P)
            P = P[hull.vertices]
        except:
            pass

        N = len(P)
        d = 2
        Q = np.vstack((P.T, np.ones(N)))
        
        err = 1.0
        tol = 0.005
        u = np.ones(N) / N
        
        max_iters = 500
        iters = 0
        
        while err > tol and iters < max_iters:
            X = Q @ np.diag(u) @ Q.T
            try:
                X_inv = np.linalg.inv(X)
            except np.linalg.LinAlgError:
                break
                
            M = np.diag(Q.T @ X_inv @ Q)
            j = np.argmax(M)
            maximum = M[j]
            step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
            new_u = (1.0 - step_size) * u
            new_u[j] += step_size
            err = np.linalg.norm(new_u - u)
            u = new_u
            iters += 1
            
        c = (P.T @ u).T
        U = np.diag(u)
        
        try:
            A = np.linalg.inv(P.T @ U @ P - np.outer(c, c)) / d
            eigenvalues, eigenvectors = np.linalg.eigh(A)
            
            width = 2 / np.sqrt(eigenvalues[0]) + 2 * self.buff
            height = 2 / np.sqrt(eigenvalues[1]) + 2 * self.buff
            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        except:
            width = mobject.get_width() + 2 * self.buff
            height = mobject.get_height() + 2 * self.buff
            angle = 0
            c = mobject.get_center()[:2]

        center = np.array([c[0], c[1], np.mean(points[:, 2])])
        
        base_ellipse = Ellipse(width=width, height=height)
        base_ellipse.rotate(angle)
        base_ellipse.move_to(center)
        self.set_points(base_ellipse.get_points())
        
        return self

    def is_inside(self, point):
            center = self.get_center()
            
            p0 = self.point_from_proportion(0)
            p1 = self.point_from_proportion(0.25)
            
            v1 = p0 - center
            v2 = p1 - center
            
            a = np.linalg.norm(v1)
            b = np.linalg.norm(v2)
            
            angle = np.arctan2(v1[1], v1[0])
            
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            
            cos_a = np.cos(-angle)
            sin_a = np.sin(-angle)
            
            rot_x = dx * cos_a - dy * sin_a
            rot_y = dx * sin_a + dy * cos_a
            
            if a == 0 or b == 0:
                return False
                
            return (rot_x ** 2) / (a ** 2) + (rot_y ** 2) / (b ** 2) <= 1


class DistancesBetweenDocuments(InteractiveScene):
    def construct(self):
        # Load the data
        with open('languages.txt', 'r', encoding='utf-8') as file:
            languages = file.read().splitlines()

        # Load JSON and build a dynamic lookup map for each language's category
        with open('language_families.json', 'r', encoding='utf-8') as file:
            family_data = json.load(file)
        
        lang_to_category = {}
        for family in family_data["families"]:
            category_name = family["name"]
            for lang_obj in family["languages"]:
                lang_to_category[lang_obj["name"]] = category_name

        # Define macro-similarity groups (Super-families) to sort clusters near each other
        macro_groups = {
            "Romance": "Indo-European",
            "Germanic": "Indo-European",
            "Slavic": "Indo-European",
            "Celtic": "Indo-European",
            "Baltic": "Indo-European",
            "Urgofinnic": "Uralic",
            "Altaic": "Turkic"
        }

        # Create file icons for each of the languages
        self.camera.frame.scale(0.5)
        buff = 1
        files = VGroup(*[
            LanguageTextExample(language).scale(0.3).move_to(
                [
                    random.uniform(-FRAME_WIDTH*0.5 + buff, FRAME_WIDTH*0.5 - buff),
                    random.uniform(-FRAME_HEIGHT*0.5 + buff, FRAME_HEIGHT*0.5 - buff),
                    0
                ]
            )
            for language in languages
        ])
        

        self.play(
            AnimationGroup(*[
                FadeIn(file, shift = UP*0.3)
                for file in files
            ], lag_ratio = 0.05)
        , run_time = 3)

        # Add the physics engine
        for file in files:
            lang_name = file.example_text.text
            file.category = lang_to_category.get(lang_name, lang_name)
            file.macro_category = macro_groups.get(file.category, "Other")
            file.velocity = np.zeros(3)

        def update_physics(mobjects, dt):
            if dt == 0:
                return
            n = len(mobjects)
            positions = [mob.get_center() for mob in mobjects]
            forces = [np.zeros(3) for _ in range(n)]
            
            # Step 1: Compute the current center of mass for each family dynamically
            family_centers = {}
            family_counts = {}
            for i, mob in enumerate(mobjects):
                cat = mob.category
                if cat not in family_centers:
                    family_centers[cat] = np.zeros(3)
                    family_counts[cat] = 0
                family_centers[cat] += positions[i]
                family_counts[cat] += 1
            
            for cat in family_centers:
                family_centers[cat] /= family_counts[cat]

            # Step 2: Apply tracking pulls toward family centroids & macro-category pulls
            for i, mob in enumerate(mobjects):
                center = family_centers[mob.category]
                to_center = center - positions[i]
                dist_to_center = np.linalg.norm(to_center) + 1e-5
                
                # Base pull to immediate family center
                pull_strength = 22.5 if dist_to_center < 2.0 else 37.5
                forces[i] += (to_center / dist_to_center) * dist_to_center * pull_strength

            # Step 3: Handle pair-wise repulsions and structural macro-attractions
            for i in range(n):
                for j in range(i + 1, n):
                    diff = positions[j] - positions[i]
                    dist = np.linalg.norm(diff) + 1e-5
                    direction = diff / dist
                    
                    same_category = (mobjects[i].category == mobjects[j].category)
                    
                    if not same_category:
                        same_macro = (mobjects[i].macro_category == mobjects[j].macro_category) and (mobjects[i].macro_category != "Other")
                        
                        if same_macro:
                            # Reduced macro-attraction to let related clusters breathe a bit more
                            macro_pull = (dist - 2.5) * 0.65
                            forces[i] += macro_pull * direction
                            forces[j] -= macro_pull * direction
                        
                        # Increased inter-category repulsion radius and strength for clearer group separation
                        if dist < 4.2:
                            repulsion = ((4.2 - dist) ** 2) * 3.5
                            forces[i] -= repulsion * direction
                            forces[j] += repulsion * direction
                    
                    # Increased micro-repulsion threshold and coefficient for cleaner structural padding
                    if dist < 1.7:
                        micro_repulsion = ((1.7 - dist) ** 2) * 52.5
                        forces[i] -= micro_repulsion * direction
                        forces[j] += micro_repulsion * direction
                        
                # Balanced centripetal screen-centering gravity
                forces[i] -= positions[i] * 0.175

            for i, mob in enumerate(mobjects):
                mob.velocity += forces[i] * dt
                mob.velocity *= 0.55
                mob.shift(mob.velocity * dt)

        self.add(files)
        files.add_updater(update_physics)

        # Let the physics engine emergently push the files into language categories
        self.play(self.camera.frame.animate.scale(2.5).shift(UP*0.6), run_time = 10)
        files.remove_updater(update_physics)

        # Draw a surrounding ellipse around each valid cluster and label it
        groups_by_category = {}
        for file in files:
            cat = file.category
            if "unnamed" not in cat.lower():
                if cat not in groups_by_category:
                    groups_by_category[cat] = VGroup()
                groups_by_category[cat].add(file)

        ellipses = VGroup()
        labels = VGroup()

        for cat, group in groups_by_category.items():
            ellipse = SurroundingEllipse(group, buff=0.2, stroke_width = 2).set_color(BLUE)
            label = Text(cat).scale(0.5).next_to(ellipse, UP, buff=0.2)
            label.name = cat
            ellipses.add(ellipse)
            labels.add(label)

        self.play(
            AnimationGroup(*[
                ShowCreation(ellipse)
                for ellipse in ellipses
            ], lag_ratio=0.1),
            AnimationGroup(*[
                FadeIn(label, shift=UP*0.2)
                for label in labels
            ], lag_ratio=0.1),
            run_time=2
        )
        self.wait(2)

        # Create the full tree layout structure on the right side
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
                
        # Group targets together and orient to the right
        tree_text_layout = VGroup(leaves, family_braces, family_labels)
        tree_text_layout.to_edge(RIGHT, buff = 1)

        # Build phylogenetic tree graphical layout adjacent to labels
        tree = LanguageTree("tree_layout_adjusted.json", node_radius = 0.02).next_to(leaves, LEFT, buff = 0.08)
        leaf_nodes = tree.node_group[:len(leaves)]
        tree.shift(DOWN*(leaf_nodes.get_y() - leaves.get_y()))

        # Reset camera center smoothly to encapsulate everything
        all_scene_elements = VGroup(files, tree, tree_text_layout)
        
        # Determine mapping relationships from floating file components to target terminal text positions
        text_transforms = []
        for file in files:
            lang_name = file.example_text.text
            if lang_name in leaves_data:
                target_leaf = leaves[leaves_data.index(lang_name)]
                
                # Transform the text directly into its slot on the tree layout
                text_transforms.append(file.example_text.animate.match_width(target_leaf).move_to(target_leaf).set_fill(color = target_leaf.get_color()))
                
                # Fade out all non-text components (the card background, outline, etc.)
                other_parts = [part for part in file if part != file.example_text]
                if other_parts:
                    text_transforms.append(VGroup(*other_parts).animate.scale(0.001).move_to(target_leaf))

        # Identify which cluster titles dynamically match up with final tree category designations
        label_transforms = []
        remaining_labels = VGroup()
        
        for label in labels:
            matched = False
            for f_label in family_labels:
                if f_label.text == label.name and f_label.get_opacity() > 0:
                    label_transforms.append(label.animate.match_width(f_label).move_to(f_label).set_fill(color = f_label.get_color()))
                    matched = True
                    break
            if not matched:
                remaining_labels.add(label)

        # Execute structural reorganization animation
        self.play(
            AnimationGroup(
                AnimationGroup(
                    self.camera.frame.animate(run_time = 1.5).center(),
                    FadeOut(ellipses, run_time = 0.8),
                    FadeOut(remaining_labels),
                    AnimationGroup(*text_transforms, run_time = 1.5),
                    AnimationGroup(*label_transforms, run_time = 1.5)
                ),
                AnimationGroup(*[
                    GrowFromEdge(brace, LEFT, run_time = 0.8)
                    for brace in family_braces if brace.get_opacity() > 0
                ], lag_ratio = 0.05)
            , lag_ratio = 0.8)
        )

        # Progressively structuralize tree trunk, branches, and connection points
        self.play(
            AnimationGroup(
                FadeIn(tree.edge_group, lag_ratio = 0.1, run_time = 8),
                FadeIn(tree.node_group, run_time = 2),
                lag_ratio = 0.8
            )
        )
        self.wait(2)