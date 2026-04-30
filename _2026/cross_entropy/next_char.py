from __future__ import annotations

import pickle
import sys

import torch
import numpy as np
from functools import lru_cache

CHAR_ALPHABET = "abcdefghijklmnopqrstuvwxyz .,;!?"

NANOGPT_DIR = "/Users/grant/cs/nanoGPT"
NANOGPT_CKPT = "/Users/grant/cs/nanoGPT/out-wiki-char/ckpt.pt"


_nano_model = None
_nano_encode = None
_nano_decode = None
_nano_vocab = None  # list of chars in order


def _ensure_nano_loaded():
    global _nano_model, _nano_encode, _nano_decode, _nano_vocab
    if _nano_model is not None:
        return

    if NANOGPT_DIR not in sys.path:
        sys.path.insert(0, NANOGPT_DIR)
    from model import GPT, GPTConfig  # noqa: PLC0415

    checkpoint = torch.load(NANOGPT_CKPT, map_location="cpu", weights_only=False)

    # strip compile-wrapper prefix if present
    state_dict = checkpoint["model"]
    for k in list(state_dict):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)

    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    model.load_state_dict(state_dict)
    model.eval()
    _nano_model = model

    meta_path = f"{NANOGPT_DIR}/data/wiki_char/meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi = meta["stoi"]
    itos = meta["itos"]
    _nano_encode = lambda s: [stoi[c] for c in s if c in stoi]
    _nano_decode = lambda ids: "".join(itos[i] for i in ids)
    _nano_vocab = [itos[i] for i in range(len(itos))]


@lru_cache()
def get_next_char_distribution(prefix: str, alphabet: str = CHAR_ALPHABET) -> np.ndarray:
    """
    Return a probability distribution over `alphabet` using the nanoGPT
    wiki_char model. Because this model was trained character-by-character
    (each token = one character), the logits directly give P(next character).
    """
    _ensure_nano_loaded()

    alpha_lower = alphabet.lower()

    ids = _nano_encode(prefix)[-_nano_model.config.block_size:]
    if ids:
        input_ids = torch.tensor([ids])
        with torch.no_grad():
            logits = _nano_model(input_ids)[0][0, -1]  # (vocab_size,)
        probs = torch.softmax(logits, dim=-1).numpy()
    else:
        probs = np.ones(len(_nano_vocab)) / len(_nano_vocab)

    char_probs = np.zeros(len(alphabet))
    for vocab_idx, char in enumerate(_nano_vocab):
        idx = alpha_lower.find(char.lower())
        if idx != -1:
            char_probs[idx] += probs[vocab_idx]

    total = char_probs.sum()
    if total > 0:
        char_probs /= total

    return char_probs
