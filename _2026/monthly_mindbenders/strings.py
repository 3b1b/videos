from manim_imports_ext import *

LOOP_RAINBOW = [
    RED,
    YELLOW,
    GREEN,
    TEAL,
    BLUE,
    "#7744FF",
    PURPLE,
    PINK,
    MAROON,
]


class Strings(InteractiveScene):
    random_seed = (
        8  # use once_useful_constructs/find_strings_seeds.py to find/generate more seeds
    )
    n_anchors = 9
    box_dims = (4.0, 3.0, 2.0)

    def construct(self):
        frame = self.frame
        n = 10
        n_box_joins = 4

        join_seq = self.compute_join_sequence(n)
        join_seq = self.ensure_box_self_loop(join_seq, n_box_joins)

        box = self.get_open_box()
        strings, end_dots = self.get_floor_strings(n)
        bh = self.box_dims[2]

        frame.reorient(20, 65)
        self.add(strings, end_dots)
        self.play(FadeIn(box), run_time=0.5)

        drop_anims = []
        for i in range(n):
            span = (i * 0.15, 0.6 + i * 0.15)
            for m in [strings[i], end_dots[2 * i], end_dots[2 * i + 1]]:
                drop_anims.append(
                    m.animate.restore().set_anim_args(
                        time_span=span,
                        rate_func=rush_into,
                    )
                )
        self.play(
            *drop_anims,
            frame.animate.reorient(-10, 65),
            run_time=0.6 + (n - 1) * 0.15,
        )
        self.play(frame.animate.reorient(15, 55), run_time=1.5)
        self.wait(0.5)

        self.remove(strings, end_dots)
        for i in range(n):
            self.add(strings[i])
            self.add(end_dots[2 * i])
            self.add(end_dots[2 * i + 1])

        piece_map = {}
        for i in range(n):
            piece = {
                "mob": strings[i],
                "start_eid": 2 * i,
                "end_eid": 2 * i + 1,
                "size": 1,
                "dots": {
                    2 * i: end_dots[2 * i],
                    2 * i + 1: end_dots[2 * i + 1],
                },
            }
            piece_map[2 * i] = piece
            piece_map[2 * i + 1] = piece

        def pin_dots_to_endpoints(piece):
            mob = piece["mob"]
            for d_eid, dot in piece["dots"].items():
                if d_eid == piece["start_eid"]:
                    dot.f_always.move_to(lambda: mob.get_anchors()[0])
                else:
                    dot.f_always.move_to(lambda: mob.get_anchors()[-1])
            return list(piece["dots"].values())

        def remove_updaters(dots):
            for dot in dots:
                dot.clear_updaters()

        box_loops = []
        for k in range(n_box_joins):
            e1, e2, forms_loop, _ = join_seq[k]
            pa = piece_map[e1]
            pb = piece_map[e2]
            d1 = pa["dots"][e1]
            d2 = pb["dots"][e2]

            self.highlight_endpoints(d1, d2)

            lift_height = bh + 1.0
            lift_anims = []
            seen_mobs = set()
            pieces_to_lift = [(pa, e1)]
            if pa is not pb:
                pieces_to_lift.append((pb, e2))

            dot_updaters = []
            for piece in [pa, pb] if pa is not pb else [pa]:
                dot_updaters.extend(pin_dots_to_endpoints(piece))
            for piece, eid in pieces_to_lift:
                mob = piece["mob"]
                if id(mob) in seen_mobs:
                    continue
                seen_mobs.add(id(mob))
                lift_anims.append(
                    self.make_pull_lift_anim(
                        mob,
                        eid,
                        piece,
                        lift_height,
                    )
                )
            self.play(*lift_anims, run_time=1.0)
            remove_updaters(dot_updaters)
            self.wait(0.3)

            if forms_loop:
                mid = (d1.get_center() + d2.get_center()) / 2
                color = LOOP_RAINBOW[len(box_loops) % len(LOOP_RAINBOW)]
                radius = 0.25 + 0.08 * pa["size"]
                loop_mob = Circle(radius=radius)
                loop_mob.move_to(mid)
                loop_mob.set_stroke(color, 4)

                source = pa["mob"].copy()
                self.remove(pa["mob"])
                self.add(source)
                d1.f_always.move_to(source.get_start)
                d2.f_always.move_to(source.get_end)
                self.play(
                    Transform(source, loop_mob),
                    run_time=1.2,
                )
                d1.clear_updaters()
                d2.clear_updaters()
                self.remove(source)
                self.add(loop_mob)
                self.play(FadeOut(d1), FadeOut(d2), run_time=0.3)

                drop_vec = (-bh - loop_mob.get_center()[2]) * OUT
                self.play(
                    loop_mob.animate.shift(drop_vec),
                    run_time=0.8,
                    rate_func=rush_into,
                )
                self.wait(0.5)

                box_loops.append(loop_mob)
                for eid in list(pa["dots"].keys()):
                    piece_map.pop(eid, None)

            else:
                mid = (d1.get_center() + d2.get_center()) / 2
                shift_a = mid - d1.get_center()
                shift_b = mid - d2.get_center()

                dot_updaters = []
                for piece in [pa, pb] if pa is not pb else [pa]:
                    dot_updaters.extend(pin_dots_to_endpoints(piece))
                slide_anims = [pa["mob"].animate.shift(shift_a)]
                if pa is not pb:
                    slide_anims.append(pb["mob"].animate.shift(shift_b))
                self.play(*slide_anims, run_time=0.8)
                remove_updaters(dot_updaters)

                pts_a = pa["mob"].get_points().copy()
                pts_b = pb["mob"].get_points().copy()
                if e1 == pa["start_eid"]:
                    pts_a = pts_a[::-1].copy()
                if e2 == pb["end_eid"]:
                    pts_b = pts_b[::-1].copy()
                all_pts = np.concatenate([pts_a, pts_b[1:]])
                connected = VMobject()
                connected.set_points(all_pts)
                connected.set_stroke(ORANGE, 3)

                free_a = pa["start_eid"] if e1 == pa["end_eid"] else pa["end_eid"]
                free_b = pb["start_eid"] if e2 == pb["end_eid"] else pb["end_eid"]
                da = pa["dots"][free_a]
                db = pb["dots"][free_b]

                self.remove(pa["mob"], pb["mob"])
                self.add(connected)

                da.move_to(connected.get_anchors()[0])
                db.move_to(connected.get_anchors()[-1])

                self.play(
                    Flash(mid, color=YELLOW, line_length=0.12, flash_radius=0.25),
                    FadeOut(d1),
                    FadeOut(d2),
                    run_time=0.5,
                )

                drop_vec = (-bh - connected.get_center()[2]) * OUT
                merged = {
                    "mob": connected,
                    "start_eid": free_a,
                    "end_eid": free_b,
                    "size": pa["size"] + pb["size"],
                    "dots": {free_a: da, free_b: db},
                }
                dot_updaters = pin_dots_to_endpoints(merged)
                self.play(
                    connected.animate.shift(drop_vec),
                    run_time=0.8,
                    rate_func=rush_into,
                )
                remove_updaters(dot_updaters)
                self.wait(0.5)

                piece_map[free_a] = merged
                piece_map[free_b] = merged
                piece_map.pop(e1, None)
                piece_map.pop(e2, None)

        all_box_mobs = Group(box)
        seen = set()
        for piece in piece_map.values():
            pid = id(piece)
            if pid not in seen:
                seen.add(pid)
                all_box_mobs.add(piece["mob"])
                for d in piece["dots"].values():
                    all_box_mobs.add(d)
        for lm in box_loops:
            all_box_mobs.add(lm)

        self.play(
            FadeOut(all_box_mobs),
            frame.animate.reorient(0, 0).set_height(8),
            run_time=1.5,
        )

        chains_data = self.get_chain_state(n, join_seq[:n_box_joins])
        self.init_organized(chains_data, base_height=0.5)

        self.play(
            LaggedStartMap(
                FadeIn,
                VGroup(*(c["line"] for c in self.org_chains)),
                lag_ratio=0.08,
            ),
            LaggedStartMap(
                FadeIn,
                VGroup(*(d for c in self.org_chains for d in c["dots"].values())),
                lag_ratio=0.04,
            ),
            run_time=1.5,
        )

        loop_counter = Integer(0, font_size=60)
        loop_label = Text("Loops:", font_size=36)
        counter_group = VGroup(loop_label, loop_counter)
        counter_group.arrange(RIGHT, buff=0.3)
        counter_group.to_corner(UR)
        self.play(FadeIn(counter_group))

        for k in range(n_box_joins, n):
            e1, e2, forms_loop, _ = join_seq[k]
            self.animate_organized_join(e1, e2, forms_loop, loop_counter)

        self.wait(2)

        old_mobs = VGroup(
            *(c["line"] for c in self.org_chains),
            *(d for c in self.org_chains for d in c["dots"].values()),
            *self.org_loops,
            counter_group,
        )
        self.play(FadeOut(old_mobs), run_time=1.0)

        n2 = 50
        join_seq_50 = self.compute_join_sequence(n2)
        chains_50 = self.get_chain_state(n2, [])

        self.init_organized(
            chains_50,
            base_height=0.08,
            x_range=(-6.5, 6.5),
            dot_radius=0.035,
            stroke_width=2,
        )

        all_50_lines = VGroup(*(c["line"] for c in self.org_chains))
        all_50_dots = VGroup(*(d for c in self.org_chains for d in c["dots"].values()))

        counter_50 = Integer(0, font_size=60)
        label_50 = Text("Loops:", font_size=36)
        cg_50 = VGroup(label_50, counter_50)
        cg_50.arrange(RIGHT, buff=0.3)
        cg_50.to_corner(UR)

        self.play(
            FadeIn(all_50_lines),
            FadeIn(all_50_dots),
            FadeIn(cg_50),
            run_time=1.0,
        )

        for k in range(n2):
            e1, e2, forms_loop, _ = join_seq_50[k]
            self.animate_organized_join(
                e1,
                e2,
                forms_loop,
                counter_50,
                fast=False,
            )

        self.wait(3)

    def init_organized(
        self,
        chains_data,
        base_height=0.5,
        x_range=(-5, 5),
        y_center=1.0,
        loop_y=-2.0,
        dot_radius=0.07,
        stroke_width=4,
    ):
        self.org_chains = []
        self.org_end_map = {}
        self.org_loops = []
        self.org_loop_count = 0
        self.org_y = y_center
        self.org_loop_y = loop_y
        self.org_bh = base_height
        self.org_xr = x_range
        self.org_dr = dot_radius
        self.org_sw = stroke_width

        xs = self._chain_xs(len(chains_data))
        for i, cd in enumerate(chains_data):
            x = xs[i]
            h = base_height * cd["size"]
            top = self.org_y + h / 2
            bot = self.org_y - h / 2

            line = Line(
                np.array([x, bot, 0]),
                np.array([x, top, 0]),
            ).set_stroke(ORANGE, stroke_width)
            td = Dot(np.array([x, top, 0]), radius=dot_radius, color=WHITE).set_z_index(
                1
            )
            bd = Dot(np.array([x, bot, 0]), radius=dot_radius, color=WHITE).set_z_index(
                1
            )

            chain = {
                "line": line,
                "ends": [cd["ends"][0], cd["ends"][1]],
                "dots": {cd["ends"][0]: td, cd["ends"][1]: bd},
                "size": cd["size"],
            }
            self.org_chains.append(chain)
            self.org_end_map[cd["ends"][0]] = chain
            self.org_end_map[cd["ends"][1]] = chain

    def animate_organized_join(self, e1, e2, forms_loop, counter, fast=False):
        ca = self.org_end_map[e1]
        cb = self.org_end_map[e2]
        d1 = ca["dots"][e1]
        d2 = cb["dots"][e2]

        main_rt = 0.25 if fast else 0.5
        lift_y = 1.5

        if not fast:
            self.highlight_endpoints(d1, d2)

        if forms_loop:
            self.org_loop_count += 1
            color = LOOP_RAINBOW[(self.org_loop_count - 1) % len(LOOP_RAINBOW)]

            radius = 0.15 + 0.03 * ca["size"]
            if self.org_loops:
                lx = self.org_loops[-1].get_right()[0] + radius + 0.3
            else:
                lx = self.org_xr[0] + 1.0

            circle = Circle(radius=radius)
            circle.set_stroke(color, 3)
            circle.move_to(np.array([lx, self.org_loop_y, 0]))

            self.org_chains.remove(ca)
            del self.org_end_map[e1]
            del self.org_end_map[e2]

            xs = self._chain_xs(len(self.org_chains))

            if not fast:
                mid = ca["line"].get_center()
                circle_raised = Circle(radius=radius)
                circle_raised.set_stroke(color, 3)
                circle_raised.move_to(mid)

                d1.f_always.move_to(ca["line"].get_start)
                d2.f_always.move_to(ca["line"].get_end)
                self.play(
                    ReplacementTransform(ca["line"], circle_raised),
                    run_time=0.5,
                )
                d1.clear_updaters()
                d2.clear_updaters()
                self.play(FadeOut(d1), FadeOut(d2), run_time=0.2)

                anims = [
                    circle_raised.animate.move_to(circle.get_center()).set_anim_args(
                        path_arc=-PI / 3
                    ),
                    counter.animate.set_value(self.org_loop_count),
                ]
                self._add_redistribute_anims(anims, xs)
                self.play(*anims, run_time=main_rt)
                self.org_loops.append(circle_raised)

            else:
                anims = [
                    ReplacementTransform(ca["line"], circle),
                    FadeOut(d1),
                    FadeOut(d2),
                    counter.animate.set_value(self.org_loop_count),
                ]
                self._add_redistribute_anims(anims, xs)
                self.play(*anims, run_time=main_rt)
                self.org_loops.append(circle)

        else:
            oa = [e for e in ca["ends"] if e != e1][0]
            ob = [e for e in cb["ends"] if e != e2][0]
            da, db = ca["dots"][oa], cb["dots"][ob]
            new_size = ca["size"] + cb["size"]

            ia = self.org_chains.index(ca)
            ib = self.org_chains.index(cb)
            self.org_chains.remove(ca)
            self.org_chains.remove(cb)

            nc = {
                "line": None,
                "ends": [oa, ob],
                "dots": {oa: da, ob: db},
                "size": new_size,
            }
            ins = min(ia, ib, len(self.org_chains))
            self.org_chains.insert(ins, nc)

            xs = self._chain_xs(len(self.org_chains))
            ni = self.org_chains.index(nc)
            xn = xs[ni]
            hn = self.org_bh * new_size
            tn = self.org_y + hn / 2
            bn = self.org_y - hn / 2

            new_line = Line(
                np.array([xn, bn, 0]),
                np.array([xn, tn, 0]),
            ).set_stroke(ORANGE, self.org_sw)
            nc["line"] = new_line

            del self.org_end_map[e1]
            del self.org_end_map[e2]
            self.org_end_map[oa] = nc
            self.org_end_map[ob] = nc

            if not fast:
                mid_x = (d1.get_center()[0] + d2.get_center()[0]) / 2
                raised_y = self.org_y + lift_y

                h_a = abs(ca["line"].get_end()[1] - ca["line"].get_start()[1])
                h_b = abs(cb["line"].get_end()[1] - cb["line"].get_start()[1])

                meet_y = raised_y
                meet_point = np.array([mid_x, meet_y, 0])

                a_bot = meet_y
                a_top = meet_y + h_a
                da_y = a_top

                b_bot = meet_y - h_b
                b_top = meet_y
                db_y = b_bot

                ca["line"].generate_target()
                ca["line"].target.put_start_and_end_on(
                    np.array([mid_x, a_bot, 0]),
                    np.array([mid_x, a_top, 0]),
                )
                cb["line"].generate_target()
                cb["line"].target.put_start_and_end_on(
                    np.array([mid_x, b_bot, 0]),
                    np.array([mid_x, b_top, 0]),
                )

                self.play(
                    MoveToTarget(ca["line"]),
                    MoveToTarget(cb["line"]),
                    d1.animate.move_to(meet_point),
                    d2.animate.move_to(meet_point),
                    da.animate.move_to(np.array([mid_x, da_y, 0])),
                    db.animate.move_to(np.array([mid_x, db_y, 0])),
                    run_time=0.4,
                )

                connected = Line(
                    np.array([mid_x, b_bot, 0]),
                    np.array([mid_x, a_top, 0]),
                ).set_stroke(ORANGE, self.org_sw)

                self.remove(ca["line"], cb["line"])
                self.add(connected)

                self.play(
                    Flash(meet_point, color=YELLOW, line_length=0.1, flash_radius=0.2),
                    FadeOut(d1),
                    FadeOut(d2),
                    run_time=0.25,
                )

                connected.generate_target()
                connected.target.put_start_and_end_on(
                    np.array([xn, bn, 0]),
                    np.array([xn, tn, 0]),
                )
                anims = [
                    MoveToTarget(connected),
                    da.animate.move_to(np.array([xn, tn, 0])),
                    db.animate.move_to(np.array([xn, bn, 0])),
                ]
                self._add_redistribute_anims(anims, xs, skip=nc)
                self.play(*anims, run_time=main_rt)
                nc["line"] = connected

            else:
                anims = [
                    ReplacementTransform(ca["line"], new_line),
                    FadeOut(cb["line"]),
                    FadeOut(d1),
                    FadeOut(d2),
                    da.animate.move_to(np.array([xn, tn, 0])),
                    db.animate.move_to(np.array([xn, bn, 0])),
                ]
                self._add_redistribute_anims(anims, xs, skip=nc)
                self.play(*anims, run_time=main_rt)

    def _add_redistribute_anims(self, anims, xs, skip=None):
        for i, chain in enumerate(self.org_chains):
            if chain is skip:
                continue
            x = xs[i]
            h = self.org_bh * chain["size"]
            t = self.org_y + h / 2
            b = self.org_y - h / 2
            chain["line"].generate_target()
            chain["line"].target.put_start_and_end_on(
                np.array([x, b, 0]),
                np.array([x, t, 0]),
            )
            anims.append(MoveToTarget(chain["line"]))
            for j, eid in enumerate(chain["ends"]):
                pos = np.array([x, t if j == 0 else b, 0])
                anims.append(chain["dots"][eid].animate.move_to(pos))

    def _chain_xs(self, n):
        if n <= 0:
            return []
        if n == 1:
            return [0.0]
        return np.linspace(self.org_xr[0], self.org_xr[1], n).tolist()

    def compute_join_sequence(self, n):
        parent = list(range(2 * n))
        rank = [0] * (2 * n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True

        for i in range(n):
            union(2 * i, 2 * i + 1)

        loose = list(range(2 * n))
        random.shuffle(loose)

        seq = []
        lc = 0
        while len(loose) >= 2:
            a = loose.pop(random.randrange(len(loose)))
            b = loose.pop(random.randrange(len(loose)))
            fl = find(a) == find(b)
            union(a, b)
            if fl:
                lc += 1
            seq.append((a, b, fl, lc))
        return seq

    def get_chain_state(self, n, partial):
        parent = list(range(2 * n))
        rank = [0] * (2 * n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True

        for i in range(n):
            union(2 * i, 2 * i + 1)

        consumed = set()
        for entry in partial:
            union(entry[0], entry[1])
            consumed.add(entry[0])
            consumed.add(entry[1])

        loose = [e for e in range(2 * n) if e not in consumed]

        root_ends = {}
        for eid in loose:
            root_ends.setdefault(find(eid), []).append(eid)

        root_strings = {}
        for eid in range(2 * n):
            root_strings.setdefault(find(eid), set()).add(eid // 2)

        return [
            {"ends": ends, "size": len(root_strings[root])}
            for root, ends in root_ends.items()
            if len(ends) == 2
        ]

    def make_pull_lift_anim(self, string, grabbed_eid, piece, lift_height):
        start_anchors = string.get_anchors().copy()
        na = len(start_anchors)
        grab_from_start = grabbed_eid == piece["start_eid"]

        def update(mob, alpha):
            new_anchors = []
            for j in range(na):
                if grab_from_start:
                    frac = j / max(na - 1, 1)
                else:
                    frac = 1.0 - j / max(na - 1, 1)
                local_alpha = np.clip((alpha - frac * 0.5) / (1.0 - frac * 0.5), 0, 1)
                lift = smooth(local_alpha) * lift_height
                pos = start_anchors[j] + lift * OUT
                new_anchors.append(pos)
            mob.set_points_smoothly(new_anchors)

        return UpdateFromAlphaFunc(string, update)

    def highlight_endpoints(self, d1, d2):
        def make_ring_anim(dot):
            r0 = 0.35
            r1 = dot.get_width() / 2
            ring = Circle(radius=r0)
            ring.set_stroke(YELLOW, 3)
            ring.move_to(dot.get_center())
            ring.always.move_to(dot)

            def update(m, a):
                r = r0 + (r1 - r0) * smooth(a)
                m.set_width(2 * r)
                m.set_opacity(1 - smooth(a))

            return ring, UpdateFromAlphaFunc(ring, update)

        ring1, anim1 = make_ring_anim(d1)
        ring2, anim2 = make_ring_anim(d2)
        self.add(ring1, ring2)
        self.play(
            d1.animate.set_color(YELLOW),
            d2.animate.set_color(YELLOW),
            anim1,
            anim2,
            run_time=0.5,
        )
        self.remove(ring1, ring2)
        self.wait(0.3)

    def ensure_box_self_loop(self, seq, n_box_joins):
        for i in range(len(seq)):
            if seq[i][2]:
                if i >= n_box_joins - 1:
                    continue
                loop_entry = seq[i]
                seq.pop(i)
                seq.insert(n_box_joins - 1, loop_entry)
                return seq
        return seq

    def get_open_box(self):
        bw, bd, bh = self.box_dims
        w, h = bw / 2, bd / 2
        top = [
            np.array([-w, -h, 0.0]),
            np.array([w, -h, 0.0]),
            np.array([w, h, 0.0]),
            np.array([-w, h, 0.0]),
        ]
        bot = [p + bh * IN for p in top]

        edges = VGroup()
        for i in range(4):
            edges.add(Line(top[i], top[(i + 1) % 4]))
        for i in range(4):
            edges.add(Line(bot[i], bot[(i + 1) % 4]))
        for i in range(4):
            edges.add(Line(top[i], bot[i]))
        edges.set_stroke(WHITE, 2)

        def quad(c0, c1, c2, c3, opacity=0.1):
            poly = Polygon(c0, c1, c2, c3)
            poly.set_fill(GREY_D, opacity=opacity)
            poly.set_stroke(width=0)
            poly.set_z_index(-10)
            return poly

        bottom_face = quad(bot[0], bot[1], bot[2], bot[3], opacity=0.15)
        back = quad(top[2], top[3], bot[3], bot[2])
        left = quad(top[3], top[0], bot[0], bot[3])
        right = quad(top[1], top[2], bot[2], bot[1])

        return Group(bottom_face, back, left, right, edges)

    def get_floor_strings(self, n):
        na = self.n_anchors
        bw, bd, bh = self.box_dims
        strings = VGroup()
        end_dots = VGroup()
        floor_z = -bh

        for i in range(n):
            cx = random.uniform(-bw * 0.35, bw * 0.35)
            cy = random.uniform(-bd * 0.35, bd * 0.35)
            center = np.array([cx, cy, floor_z])

            angle = random.uniform(-PI, PI)
            half_len = 0.25 + 0.35 * random.random()
            d = rotate_vector(half_len * RIGHT, angle)
            p1, p2 = center - d, center + d
            perp = rotate_vector(UP, angle)

            amp_flat = 0.04 + 0.08 * random.random()
            f1 = 1.5 + 2.0 * random.random()
            f2 = 2.5 + 3.0 * random.random()
            phase = random.uniform(0, TAU)
            flat_anchors = []
            for j in range(na):
                t = j / (na - 1)
                p = p1 + t * (p2 - p1)
                wave = (
                    amp_flat
                    * (np.sin(f1 * PI * t + phase) + 0.5 * np.sin(f2 * PI * t))
                    * perp
                )
                flat_anchors.append(p + wave)

            dz = bh + 0.5 + 0.3 * i
            air_center = center + dz * OUT
            spread = half_len * 1.2
            amp_3d = 0.15 + 0.2 * random.random()
            f3 = 1.0 + 2.0 * random.random()
            f4 = 1.5 + 2.0 * random.random()
            f5 = 1.0 + 1.5 * random.random()
            phase2 = random.uniform(0, TAU)
            phase3 = random.uniform(0, TAU)
            air_anchors = []
            for j in range(na):
                t = j / (na - 1)
                base = air_center + (t - 0.5) * 2 * spread * d / half_len
                offset = np.array(
                    [
                        amp_3d * np.sin(f3 * PI * t + phase),
                        amp_3d * np.sin(f4 * PI * t + phase2),
                        amp_3d * np.sin(f5 * PI * t + phase3),
                    ]
                )
                air_anchors.append(base + offset)

            string = VMobject()
            string.set_points_smoothly(air_anchors)
            string.set_stroke(ORANGE, 3)

            flat_string = VMobject()
            flat_string.set_points_smoothly(flat_anchors)
            flat_string.set_stroke(ORANGE, 3)
            string.saved_state = flat_string

            dot1 = Dot(air_anchors[0], radius=0.06, color=WHITE)
            dot2 = Dot(air_anchors[-1], radius=0.06, color=WHITE)
            flat_dot1 = Dot(flat_anchors[0], radius=0.06, color=WHITE)
            flat_dot2 = Dot(flat_anchors[-1], radius=0.06, color=WHITE)
            dot1.saved_state = flat_dot1
            dot2.saved_state = flat_dot2

            strings.add(string)
            end_dots.add(dot1)
            end_dots.add(dot2)

        return strings, end_dots
