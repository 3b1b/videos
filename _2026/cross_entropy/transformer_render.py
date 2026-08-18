"""
Manim reconstruction of the nanoGPT transformer visualization.

This is a native-manim rebuild of the point-cloud layout defined by the Python
SOP embedded in `houdini_nanogpt/nanogpt_2.hiplc`. Nothing is imported from the
Houdini file itself (its geometry is procedural and only exists once Houdini
"cooks" the graph); instead the *generating algorithm* was recovered and
re-implemented here so the whole thing lives in manim.

Structure (top -> bottom, i.e. decreasing y, matching the original):

    input embeddings
      -> for each of N_BLOCKS transformer blocks:
             ln1 -> multi-head q/k/v -> attention matrices -> attn out
             -> projection -> attention residual
             -> ln2 -> mlp fc -> mlp hidden -> mlp act -> mlp proj -> residual
      -> final ln -> lm_head -> logits -> softmax -> output tokens

Each "layer" is a small (rows x cols) grid of points, exactly as in
`create_layer_points`. Every point carries a `layer_depth` (its order along the
forward pass); `forward_pass()` sweeps a brightness wave along that axis to
indicate information propagating through the network.

The weight/activation values that drive the greyscale shading are synthesized
deterministically so the scene builds instantly with no torch dependency. To
drive it with the real model instead, replace `_synth_values` with a lookup into
actual GPT-2 tensors (see next_char.py, which already loads GPT-2).
"""
from __future__ import annotations

import numpy as np

from manimlib import *

# ---------------------------------------------------------------------------
# Layout constants (verbatim from the nanogpt_2.hiplc Python SOP)
# ---------------------------------------------------------------------------
SPACING = 0.06
LAYER_SPACING_X = 3.0
LAYER_SPACING_Y = 2.0
LAYER_SPACING_Z = 2.0
BLOCK_SPACING_Y = 4.0
RESIDUAL_X = 5.0
INPUT_LETTER_SPACING = 1.0

# Visualization dimensions (the SOP shows only a slice of the full tensors)
VIS_EMBD = 32
VIS_HEADS = 6
SEQ_LEN = 4
VIS_VOCAB = 16
N_EMBD = 384
N_HEAD = 6
HEAD_DIM = N_EMBD // N_HEAD
VIS_MLP_DIM = min(4 * N_EMBD, 128)
N_BLOCKS = 6

# Highlight color for the forward-pass flash
FLASH_COLOR = np.array([0.25, 0.65, 1.0])


# ---------------------------------------------------------------------------
# Greyscale color models (from weight_to_greyscale / activation_to_greyscale)
# ---------------------------------------------------------------------------
def _weight_rgba(vals: np.ndarray) -> np.ndarray:
    t = np.clip(np.abs(vals) / 0.1, 0, 1)
    g = 1.0 - t * 0.8
    return np.stack([g, g, g, np.ones_like(g)], axis=-1)


def _activation_rgba(vals: np.ndarray) -> np.ndarray:
    t = np.clip(vals, 0, 1)
    g = 0.2 + t * 0.8
    return np.stack([g, g, g, np.ones_like(g)], axis=-1)


def _plain_rgba(n: int) -> np.ndarray:
    return np.tile(np.array([0.1, 0.1, 0.1, 1.0]), (n, 1))


def _synth_values(name: str, kind: str, n: int) -> np.ndarray:
    """Deterministic stand-in values so the shading looks organic without torch."""
    rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
    if kind == "weight":
        return rng.normal(0.0, 0.06, n)
    if kind == "activation":
        return np.abs(rng.normal(0.0, 0.45, n))
    return np.zeros(n)


# ---------------------------------------------------------------------------
# Grid geometry (from create_layer_points)
# ---------------------------------------------------------------------------
def _grid_points(dims: tuple[int, int], center, sp: float) -> np.ndarray:
    rows, cols = dims
    total_w = cols * sp
    total_h = rows * sp
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    xs = center[0] + cc * sp - total_w / 2
    ys = center[1] - rr * sp + total_h / 2
    zs = np.full(rr.shape, center[2], dtype=float)
    return np.stack([xs, ys, zs], axis=-1).reshape(-1, 3).astype(float)


class TransformerModel(Group):
    """
    The full transformer as one mobject: a stack of DotClouds, one per section
    (input, each block, output), grouped so you can grab any part and play.

        model = TransformerModel()
        self.add(model)
        self.play(model.forward_pass())        # flash a forward pass
        model.blocks[2].set_opacity(0.3)        # dim block 2
        model.sections["input"]                 # the input DotCloud
    """

    def __init__(
        self,
        n_blocks: int = N_BLOCKS,
        scale: float = 0.15,
        glow_factor: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_blocks = n_blocks
        self.model_scale = scale
        self.glow_factor = glow_factor

        # Filled during the build:
        self._layers: list[dict] = []          # flat list of layer specs
        self._section_order: list[str] = []
        self._cur_section = "input"
        self._depth_map: dict[str, int] = {}
        self._depth_counter = 0

        self._assign_depths()
        self._build_layers()
        self.max_depth = max((L["depth"] for L in self._layers), default=1)

        # Group layers into one DotCloud per section
        self.sections: dict[str, DotCloud] = {}
        self._section_depths: dict[str, np.ndarray] = {}
        self._section_base_rgba: dict[str, np.ndarray] = {}
        self._assemble()

        self.blocks = [self.sections[f"block{b}"] for b in range(n_blocks)]

    # -- depth bookkeeping (from assign_depth) --------------------------------
    def _assign(self, name: str) -> int:
        self._depth_map[name] = self._depth_counter
        self._depth_counter += 1
        return self._depth_map[name]

    def _assign_depths(self) -> None:
        for name in ["raw_input", "tokenized", "token_embedding",
                     "position_embedding", "input_embedding"]:
            self._assign(name)
        for b in range(self.n_blocks):
            p = f"b{b}_"
            for suffix in ["ln1", "qkv", "attn", "attn_out", "attn_residual",
                           "ln2", "mlp_fc", "mlp_act", "mlp_proj", "mlp_residual"]:
                self._assign(p + suffix)
        for name in ["final_ln", "lm_head", "logits", "softmax", "output"]:
            self._assign(name)

    # -- layer accumulation ---------------------------------------------------
    def _add_layer(self, name, dims, center, depth_key, kind="plain",
                   pscale=1.0, sp=None):
        rows, cols = dims
        if rows <= 0 or cols <= 0:
            return
        sp = SPACING if sp is None else sp
        pts = _grid_points(dims, center, sp)
        n = len(pts)
        depth = self._depth_map[depth_key]

        vals = _synth_values(name, kind, n)
        if kind == "weight":
            rgba = _weight_rgba(vals)
        elif kind == "activation":
            rgba = _activation_rgba(vals)
        else:
            rgba = _plain_rgba(n)

        radius = np.full(n, sp * 0.75 * pscale)

        self._layers.append(dict(
            section=self._cur_section, name=name, depth=depth,
            points=pts, rgba=rgba, radius=radius,
        ))

    # -- the build (mirrors SECTIONS 11-13 of the SOP) ------------------------
    def _build_layers(self) -> None:
        y = 0.0

        # ---- SECTION 11: input subnet ----
        self._cur_section = "input"
        # raw input + tokenized: a few fat points per sequence position
        for i in range(SEQ_LEN):
            self._add_layer(f"raw_input_{i}", (1, 1),
                            (0, y, i * INPUT_LETTER_SPACING), "raw_input", pscale=3.0)
        y -= LAYER_SPACING_Y
        for i in range(SEQ_LEN):
            self._add_layer(f"tokenized_{i}", (1, 1),
                            (0, y, i * INPUT_LETTER_SPACING), "tokenized", pscale=2.0)
        y -= LAYER_SPACING_Y
        self._add_layer("token_embedding", (VIS_VOCAB, VIS_EMBD),
                        (-5, y, 0), "token_embedding", kind="weight")
        self._add_layer("position_embedding", (SEQ_LEN, VIS_EMBD),
                        (5, y, 0), "position_embedding", kind="weight")
        y -= LAYER_SPACING_Y
        self._add_layer("input_embedding", (SEQ_LEN, VIS_EMBD),
                        (RESIDUAL_X, y, 0), "input_embedding", kind="activation")
        y -= LAYER_SPACING_Y

        # ---- SECTION 12: transformer blocks ----
        for b in range(self.n_blocks):
            self._cur_section = f"block{b}"
            y = self._build_attention_block(b, y)
            y = self._build_mlp_block(b, y)
            y -= BLOCK_SPACING_Y

        # ---- SECTION 13: output subnet ----
        self._cur_section = "output"
        self._add_layer("final_ln_beta", (VIS_EMBD, 1),
                        (-LAYER_SPACING_X * 2, y, -1), "final_ln", kind="weight")
        self._add_layer("final_ln_gamma", (VIS_EMBD, 1),
                        (-LAYER_SPACING_X * 1.5, y, 0), "final_ln", kind="weight")
        self._add_layer("final_layer_norm", (SEQ_LEN, VIS_EMBD),
                        (0, y, 1), "final_ln", kind="activation")
        y -= LAYER_SPACING_Y
        self._add_layer("lm_head_weights", (VIS_EMBD, VIS_VOCAB),
                        (-1, y, -1), "lm_head", kind="weight")
        y -= LAYER_SPACING_Y
        self._add_layer("logits", (SEQ_LEN, VIS_VOCAB),
                        (0, y, 1), "logits", kind="activation")
        y -= LAYER_SPACING_Y
        self._add_layer("logits_softmax", (SEQ_LEN, VIS_VOCAB),
                        (0, y, 0), "softmax", kind="activation")
        y -= LAYER_SPACING_Y * 1.5
        for i in range(SEQ_LEN):
            self._add_layer(f"output_{i}", (1, 1),
                            (0, y, i * INPUT_LETTER_SPACING), "output", pscale=3.0)

    def _build_attention_block(self, b: int, start_y: float) -> float:
        y = start_y
        p = f"b{b}_"

        # Layer norm
        self._add_layer(p + "ln_beta", (VIS_EMBD, 1),
                        (-LAYER_SPACING_X * 2, y, 0), p + "ln1", kind="weight")
        self._add_layer(p + "ln_gamma", (VIS_EMBD, 1),
                        (-LAYER_SPACING_X * 1.5, y, 0), p + "ln1", kind="weight")
        self._add_layer(p + "layer_norm", (VIS_EMBD, SEQ_LEN),
                        (0, y, 0), p + "ln1", kind="activation")
        self._add_layer(p + "ln_agg", (SEQ_LEN, 2),
                        (LAYER_SPACING_X * 1.2, y, 0), p + "ln1",
                        sp=SPACING * 1.5, pscale=1.5)
        y -= LAYER_SPACING_Y

        # Multi-head Q/K/V
        head_z_spacing = LAYER_SPACING_Z * 0.8
        z_start = -(VIS_HEADS - 1) * head_z_spacing / 2
        for h in range(VIS_HEADS):
            z = z_start + h * head_z_spacing
            # Q at z-0.3/-0.5, K at z, V at z+0.3/+0.5
            for tag, dz_b, dz_w, dz_v in [("q", -0.3, -0.5, -0.3),
                                          ("k", 0.0, 0.0, 0.0),
                                          ("v", 0.3, 0.5, 0.3)]:
                self._add_layer(f"{p}{tag}_bias_h{h}", (VIS_EMBD, 1),
                                (-LAYER_SPACING_X * 1.5, y, z + dz_b),
                                p + "qkv", kind="weight")
                self._add_layer(f"{p}{tag}_weights_h{h}", (VIS_EMBD, VIS_EMBD),
                                (-LAYER_SPACING_X * 0.8, y, z + dz_w),
                                p + "qkv", kind="weight")
                self._add_layer(f"{p}{tag}_vectors_h{h}", (SEQ_LEN, VIS_EMBD),
                                (LAYER_SPACING_X * 0.5, y, z + dz_v),
                                p + "qkv", kind="activation")
        y -= LAYER_SPACING_Y

        # Attention matrices (one per head)
        for h in range(VIS_HEADS):
            z = z_start + h * head_z_spacing
            self._add_layer(f"{p}attention_matrix_h{h}", (SEQ_LEN, SEQ_LEN),
                            (0, y, z), p + "attn", kind="activation",
                            sp=SPACING * 2, pscale=2.0)
            self._add_layer(f"{p}attn_softmax_h{h}", (SEQ_LEN, SEQ_LEN),
                            (LAYER_SPACING_X * 1, y, z), p + "attn",
                            kind="activation", sp=SPACING * 2, pscale=2.0)
        y -= LAYER_SPACING_Y

        # V output
        self._add_layer(p + "v_output", (SEQ_LEN, VIS_EMBD),
                        (0, y, 0), p + "attn_out", kind="activation")
        y -= LAYER_SPACING_Y

        # Projection
        self._add_layer(p + "projection_bias", (VIS_EMBD, 1),
                        (-LAYER_SPACING_X * 1.5, y, 0), p + "attn_out", kind="weight")
        self._add_layer(p + "projection_weights", (VIS_EMBD, VIS_EMBD),
                        (-LAYER_SPACING_X * 0.8, y, 0), p + "attn_out", kind="weight")
        self._add_layer(p + "attention_output", (SEQ_LEN, VIS_EMBD),
                        (LAYER_SPACING_X * 0.5, y, 0), p + "attn_out", kind="activation")
        y -= LAYER_SPACING_Y

        # Attention residual
        self._add_layer(p + "attention_residual", (SEQ_LEN, VIS_EMBD),
                        (RESIDUAL_X, y, 0), p + "attn_residual", kind="activation")
        return y

    def _build_mlp_block(self, b: int, start_y: float) -> float:
        y = start_y - LAYER_SPACING_Y
        p = f"b{b}_"

        self._add_layer(p + "mlp_ln_beta", (VIS_EMBD, 1),
                        (1.5, y - LAYER_SPACING_Y, 0), p + "ln2", kind="weight")
        self._add_layer(p + "mlp_ln_gamma", (VIS_EMBD, 1),
                        (2, y - LAYER_SPACING_Y, 0), p + "ln2", kind="weight")
        self._add_layer(p + "mlp_layer_norm", (VIS_EMBD, SEQ_LEN),
                        (2.5, y - LAYER_SPACING_Y, -2), p + "ln2", kind="activation")
        self._add_layer(p + "mlp_ln_agg", (SEQ_LEN, 2),
                        (2.5, y, 0), p + "ln2", sp=SPACING * 1.5, pscale=1.5)
        self._add_layer(p + "mlp_bias", (1, VIS_MLP_DIM),
                        (-3, y - 0.5, 1), p + "mlp_fc", kind="weight")
        y -= LAYER_SPACING_Y

        self._add_layer(p + "mlp_weights", (VIS_EMBD, VIS_MLP_DIM),
                        (-3, y, -1), p + "mlp_fc", kind="weight")
        y -= LAYER_SPACING_Y

        self._add_layer(p + "mlp", (SEQ_LEN, VIS_MLP_DIM),
                        (-3, y, 0), p + "mlp_fc", kind="activation")
        y -= LAYER_SPACING_Y

        self._add_layer(p + "mlp_activation", (SEQ_LEN, VIS_MLP_DIM),
                        (-3, y, -1), p + "mlp_act", kind="activation")
        y -= LAYER_SPACING_Y

        self._add_layer(p + "mlp_projection_weights", (VIS_EMBD, VIS_MLP_DIM),
                        (-3, y, 1), p + "mlp_proj", kind="weight")
        self._add_layer(p + "mlp_projection_bias", (VIS_EMBD, 1),
                        (-8, y, 0), p + "mlp_proj", kind="weight")
        self._add_layer(p + "mlp_result", (SEQ_LEN, VIS_EMBD),
                        (2.5, y, 0), p + "mlp_proj", kind="activation")
        self._add_layer(p + "mlp_residual", (SEQ_LEN, VIS_EMBD),
                        (RESIDUAL_X, y - 2, 0), p + "mlp_residual", kind="activation")
        return y - 2

    # -- assemble DotClouds ---------------------------------------------------
    def _assemble(self) -> None:
        by_section: dict[str, list[dict]] = {}
        for L in self._layers:
            by_section.setdefault(L["section"], []).append(L)

        for section, layers in by_section.items():
            points = np.concatenate([L["points"] for L in layers]) * self.model_scale
            rgba = np.concatenate([L["rgba"] for L in layers])
            radius = np.concatenate([L["radius"] for L in layers]) * self.model_scale
            depths = np.concatenate([
                np.full(len(L["points"]), L["depth"]) for L in layers
            ]).astype(float)

            dc = DotCloud(points, glow_factor=self.glow_factor)
            dc.data["rgba"][:] = rgba
            dc.set_radii(radius)
            dc.apply_depth_test()

            self.sections[section] = dc
            self._section_depths[section] = depths
            self._section_base_rgba[section] = rgba.copy()
            self.add(dc)

    # -- forward-pass flash ---------------------------------------------------
    def set_wave(self, center: float | None, width: float = 4.0, amp: float = 1.6):
        """
        Brighten points whose layer_depth is near `center`, fading with a
        gaussian of the given `width`. Pass center=None to reset to base colors.
        """
        for section, dc in self.sections.items():
            base = self._section_base_rgba[section]
            if center is None:
                dc.data["rgba"][:] = base
                continue
            depths = self._section_depths[section]
            t = np.exp(-((depths - center) / width) ** 2)[:, None]
            rgb = base[:, :3] + amp * t * FLASH_COLOR[None, :]
            dc.data["rgba"][:, :3] = np.clip(rgb, 0, 1)
            dc.data["rgba"][:, 3] = base[:, 3]
        return self

    def forward_pass(self, run_time: float = 5.0, width: float = 4.0,
                     amp: float = 1.6, **kwargs):
        """Animation: a flash of activity sweeping top -> bottom through depth."""
        span = self.max_depth + 3 * width

        def update(mob, alpha):
            center = -2 * width + alpha * span
            mob.set_wave(center, width=width, amp=amp)

        return UpdateFromAlphaFunc(
            self, update, run_time=run_time, rate_func=linear, **kwargs
        )


class TransformerForwardPass(Scene):
    def construct(self):
        model = TransformerModel()
        self.add(model)

        # The model is long and thin, so the natural "full transformer,
        # top-to-bottom, with perspective" shot looks down its length: input
        # near, output receding to a vanishing point (phi ~ 76 = view roughly
        # along the y axis, theta gives a 3/4 angle). Tweak all of this live.
        self.frame.reorient(-15, 76, 0, center=model.get_center(), height=12)

        # Show one forward pass, then hand control over for interactive play.
        self.play(model.forward_pass(run_time=6))
        self.wait()

        # Interactive: e.g. `play(model.forward_pass())`, reorient the frame,
        # `model.blocks[3].set_opacity(0.2)`, etc. (Skipped when rendering to
        # a file, where there is no interactive window.)
        if self.window is not None:
            self.embed()
