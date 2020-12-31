from imports_3b1b import *


class NameAnimation(Scene):
    CONFIG = {
        "patron_name": "Grant Sanderson"
    }

    def construct(self):
        name_mob = TextMobject(self.patron_name)
        name_mob.scale(2)
        max_width = FRAME_WIDTH - 2
        if name_mob.get_width() > max_width:
            name_mob.set_width(max_width)
        name_strokes = VGroup()
        for mob in name_mob.family_members_with_points():
            mob.insert_n_curves(20)
            anchors1, handles1, handles2, anchors2 = mob.get_anchors_and_handles()
            for a1, h1, h2, a2 in zip(anchors1, handles1, handles2, anchors2):
                stroke = VMobject()
                stroke.set_points([a1, h1, h2, a2])
                stroke.set_stroke(WHITE, width=2)
                name_strokes.add(stroke)
                stroke.save_state()

        from _2017.eoc.div_curl import four_swirls_function
        from _2017.eoc.div_curl import VectorField
        from _2017.eoc.div_curl import move_submobjects_along_vector_field
        func = four_swirls_function
        vector_field = VectorField(func)
        vector_field.submobjects.sort(
            key=lambda a: a.get_length()
        )
        flow = move_submobjects_along_vector_field(name_strokes, func)

        self.add_foreground_mobjects(name_strokes)
        self.play(Write(name_strokes))
        self.play(LaggedStartMap(GrowArrow, vector_field))
        self.add(flow)
        self.wait(60)
        self.remove(flow)
        self.play(
            FadeOut(vector_field),
            LaggedStartMap(
                ApplyMethod, name_strokes,
                lambda m: (m.restore,),
                lag_ratio=0.2
            ),
            run_time=5,
        )
        self.wait()

# patron_names = [
#     "Tobias",
#     "Monku",
#     "Kyle Begovich",
#     "Oliver Dunk",
#     "Ofir Kedar",
#     "Xiyu Cai",
#     "Anoki",
#     "Lucas Dziesinski",
#     "Christian J\\\"ah",
# ]

# if __name__ == "__main__":
#     for name in patron_names:
#         no_whitespace_name = name.replace(" ", "")
#         scene = NameAnimation(
#             patron_name=name,
#             name=no_whitespace_name + "_AsFluidFLow",
#             write_to_movie=True,
#             camera_config=PRODUCTION_QUALITY_CAMERA_CONFIG,
#         )