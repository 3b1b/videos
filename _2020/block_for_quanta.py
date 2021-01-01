from _2019.clacks.question import BlocksAndWallExampleMass1e2
from _2019.clacks.solution1 import CircleDiagramFromSlidingBlocks1e2


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
