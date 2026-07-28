from manimlib import *
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog


class TreeEditor(Scene):
    def construct(self):
        self.add(SVGMobject("Tree.svg").set_color(WHITE).set_stroke(width=3).set_height(FRAME_HEIGHT * 0.95).stretch_to_fit_width(FRAME_WIDTH - 5).to_edge(RIGHT, buff=4))
        self.nodes = VGroup()
        self.edges = VGroup()
        self.preview_edge = Line(ORIGIN, ORIGIN, color=WHITE, stroke_opacity=0)

        self.add(self.edges, self.nodes, self.preview_edge)

        self.selected_obj = None
        self.is_creating_edge = False

    def on_key_press(self, symbol, modifiers):
        ctrl_held = bool(modifiers & 2)

        if ctrl_held:
            if symbol in [ord('s'), 115]:
                self.export_to_json()
                return
            if symbol in [ord('o'), 111]:
                self.load_from_json()
                return

    def on_mouse_press(self, point, button, mods):
        is_right_click = (button == 2 or button == 4)
        ctrl_held = bool(mods & 2)
        alt_held = bool(mods & 4)

        if is_right_click:
            clicked_node = self.get_node_at_point(point)
            if clicked_node:
                to_remove = [e for e in self.edges if e.u == clicked_node or e.v == clicked_node]
                for e in to_remove:
                    self.edges.remove(e)
                    self.remove(e)
                self.nodes.remove(clicked_node)
                self.remove(clicked_node)
                return

            for e in self.edges:
                if np.linalg.norm(e.handle.get_center() - point) < 0.4 * self.camera.frame.get_width() / FRAME_WIDTH:
                    self.edges.remove(e)
                    self.remove(e)
                    return
            return

        for edge_group in self.edges:
            if np.linalg.norm(edge_group.handle.get_center() - point) < 0.4 * self.camera.frame.get_width() / FRAME_WIDTH:
                self.selected_obj = edge_group.handle
                return

        if alt_held:
            for node in self.nodes:
                if np.linalg.norm(node.label.get_center() - point) < 0.4 * self.camera.frame.get_width() / FRAME_WIDTH:
                    self.selected_obj = node.label
                    self.selected_obj.parent_node = node
                    return

        clicked_node = self.get_node_at_point(point)
        if clicked_node:
            self.selected_obj = clicked_node
            if ctrl_held:
                self.is_creating_edge = True
            return

        if button == 1 and not ctrl_held:
            new_node = self.create_node(point)
            self.nodes.add(new_node)
            self.selected_obj = new_node

    def on_mouse_drag(self, point, d_point, buttons, mods):
        if self.selected_obj is None:
            return

        if self.is_creating_edge:
            self.preview_edge.set_stroke(opacity=0.5)
            self.preview_edge.put_start_and_end_on(self.selected_obj.dot.get_center(), point)
        elif hasattr(self.selected_obj, "parent_node"):
            center = self.selected_obj.parent_node.dot.get_center()
            direction = normalize(point - center)
            self.selected_obj.move_to(center + direction * 0.5)
        elif hasattr(self.selected_obj, "is_handle"):
            self.selected_obj.move_to(point)
            self.refresh_edge(self.selected_obj.edge_group, snapping=True)
        else:
            self.selected_obj.move_to(point)
            for e in self.edges:
                if e.u == self.selected_obj or e.v == self.selected_obj:
                    self.refresh_edge(e, follow_node=True)

    def on_mouse_release(self, point, button, mods):
        if self.is_creating_edge:
            self.preview_edge.set_stroke(opacity=0)
            target = self.get_node_at_point(point)
            if target and target != self.selected_obj:
                self.add_edge(self.selected_obj, target)
        self.selected_obj = None
        self.is_creating_edge = False

    def style_edge(self, arc):
        arc.set_stroke(width=6).set_color(YELLOW)
        return arc

    def add_edge(self, u, v):
        edge_group = VGroup()
        edge_group.u, edge_group.v = u, v
        edge_group.angle = 0
        u_p, v_p = self.get_anchored_points(u.dot.get_center(), v.dot.get_center(), 0)
        arc = ArcBetweenPoints(u_p, v_p, angle=0)
        self.style_edge(arc)
        handle = Dot(arc.point_from_proportion(0.5), radius=0.05).set_color(YELLOW)
        handle.is_handle, handle.edge_group = True, edge_group
        edge_group.add(arc, handle)
        edge_group.arc, edge_group.handle = arc, handle
        self.edges.add(edge_group)

    def refresh_edge(self, edge_group, follow_node=False, snapping=False):
        u_c, v_c = edge_group.u.dot.get_center(), edge_group.v.dot.get_center()
        if not follow_node:
            edge_group.angle = self.calculate_angle(u_c, v_c, edge_group.handle.get_center())
        u_p, v_p = self.get_anchored_points(u_c, v_c, edge_group.angle)
        new_arc = ArcBetweenPoints(u_p, v_p, angle=edge_group.angle)
        self.style_edge(new_arc)
        edge_group.arc.become(new_arc)
        if follow_node or snapping:
            edge_group.handle.move_to(new_arc.point_from_proportion(0.5))

    def calculate_angle(self, start, end, mid):
        chord = end - start
        dist = np.linalg.norm(chord)
        if dist < 0.1:
            return 0
        perp_vec = np.array([chord[1], -chord[0], 0]) / dist
        height = np.dot(mid - (start + end) / 2, perp_vec)
        return 4 * np.clip(height, -dist * 0.8, dist * 0.8) / dist

    def get_node_at_point(self, point):
        for node in self.nodes:
            if np.linalg.norm(node.dot.get_center() - point) < 0.4 * self.camera.frame.get_width() / FRAME_WIDTH:
                return node
        return None

    def create_node(self, point):
        current_labels = {node.label.text for node in self.nodes}
        index = 0
        while True:
            candidate = self.get_label_name(index)
            if candidate not in current_labels:
                name = candidate
                break
            index += 1
        dot = Dot(point, radius=0.05).set_color("#F5F5DC")
        label = Text(name, font_size=15, font="CMU Serif").set_color(BLUE).next_to(dot, UP, buff=0.1)
        node = VGroup(dot, label)
        node.dot, node.label = dot, label
        return node

    def get_label_name(self, n):
        name = ""
        while n >= 0:
            name = chr(n % 26 + 65) + name
            n = (n // 26) - 1
        return name

    def get_anchored_points(self, u_c, v_c, angle, radius=0):
        temp_arc = ArcBetweenPoints(u_c, v_c, angle=angle)
        t_len = temp_arc.get_arc_length()
        if t_len == 0:
            return u_c, v_c
        return temp_arc.point_from_proportion(radius / t_len), temp_arc.point_from_proportion(1 - radius / t_len)

    def export_to_json(self):
        data = {"nodes": [], "edges": []}
        for node in self.nodes:
            offset = node.label.get_center() - node.dot.get_center()
            data["nodes"].append({
                "label": node.label.text,
                "position": node.dot.get_center().tolist(),
                "label_offset": offset.tolist()
            })
        for edge in self.edges:
            data["edges"].append({
                "start_node": edge.u.label.text,
                "end_node": edge.v.label.text,
                "angle": edge.angle
            })

        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(defaultextension=".json")
        root.destroy()
        if file_path:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)

    def load_from_json(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        root.destroy()

        if not file_path:
            return

        with open(file_path, "r") as f:
            data = json.load(f)

        for e in list(self.edges):
            self.remove(e)
        for n in list(self.nodes):
            self.remove(n)
        self.nodes.submobjects = []
        self.edges.submobjects = []

        node_map = {}
        for n_data in data["nodes"]:
            pos = np.array(n_data["position"])
            offset = np.array(n_data.get("label_offset", [0, 0.1, 0]))

            dot = Dot(pos, radius=0.05).set_color("#F5F5DC")
            label = Text(n_data["label"], font_size=15, font="CMU Serif").set_color(BLUE)
            label.move_to(pos + offset)

            node = VGroup(dot, label)
            node.dot, node.label = dot, label

            self.nodes.add(node)
            node_map[n_data["label"]] = node

        for e_data in data["edges"]:
            u = node_map.get(e_data["start_node"])
            v = node_map.get(e_data["end_node"])
            if not u or not v:
                continue

            edge_group = VGroup()
            edge_group.u, edge_group.v = u, v
            edge_group.angle = e_data["angle"]

            u_p, v_p = self.get_anchored_points(u.dot.get_center(), v.dot.get_center(), edge_group.angle)
            arc = ArcBetweenPoints(u_p, v_p, angle=edge_group.angle)
            self.style_edge(arc)

            handle = Dot(arc.point_from_proportion(0.5), radius=0.05).set_color(YELLOW)
            handle.is_handle, handle.edge_group = True, edge_group

            edge_group.add(arc, handle)
            edge_group.arc, edge_group.handle = arc, handle

            self.edges.add(edge_group)
