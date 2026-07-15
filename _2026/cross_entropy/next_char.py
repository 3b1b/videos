from __future__ import annotations

import hashlib
import heapq
import json
import os
import pickle
import sys

import torch
import numpy as np
from collections import Counter
from functools import lru_cache

# CHAR_ALPHABET = "abcdefghijklmnopqrstuvwxyz .,'!?"
CHAR_ALPHABET = "abcdefghijklmnopqrstuvwxyz .,'!?0123456789-"

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


_gpt2_tokenizer = None
_gpt2_model = None


def _ensure_gpt2_loaded():
    global _gpt2_tokenizer, _gpt2_model
    if _gpt2_model is not None:
        return
    from transformers import GPT2Tokenizer, GPT2LMHeadModel  # noqa: PLC0415
    _gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
    _gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True)
    _gpt2_model.eval()


def gpt2_predict_next_token(text: str, n_shown: int = 7):
    """
    Top-`n_shown` next-token predictions from a local GPT-2 instance.
    Returns (tokens, probs) where tokens is a list of decoded strings and
    probs is a numpy array of their probabilities.
    """
    _ensure_gpt2_loaded()

    input_ids = _gpt2_tokenizer.encode(
        text, add_special_tokens=False, return_tensors="pt"
    )
    with torch.no_grad():
        logits = _gpt2_model(input_ids).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, n_shown)
    tokens = [_gpt2_tokenizer.decode([i]) for i in top_indices.tolist()]
    return tokens, top_probs.numpy()


def gpt2_token_probability(text: str, token: str) -> float:
    """
    Probability GPT-2 assigns to `token` following `text`.

    `token` is a raw string, which may encode to one or more GPT-2 tokens.
    When it spans multiple tokens, the joint probability of the continuation
    is returned (the product of successive next-token probabilities).
    """
    _ensure_gpt2_loaded()

    input_ids = _gpt2_tokenizer.encode(
        text, add_special_tokens=False, return_tensors="pt"
    )
    token_ids = _gpt2_tokenizer.encode(token, add_special_tokens=False)

    probability = 1.0
    with torch.no_grad():
        for token_id in token_ids:
            logits = _gpt2_model(input_ids).logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            probability *= probs[token_id].item()
            next_id = torch.tensor([[token_id]])
            input_ids = torch.cat([input_ids, next_id], dim=1)
    return probability


_qwen_tokenizer = None
_qwen_model = None

# Modern (2024) open-source stand-in for GPT-2. At ~0.5B params it is a few
# times larger than GPT-2 base (124M), but still small enough to run on CPU,
# and it uses a byte-level BPE tokenizer analogous to GPT-2's. This is the
# *base* model (raw next-token prediction over natural text), matching how the
# gpt2_* functions above behave.
QWEN_MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def _ensure_qwen_loaded():
    global _qwen_tokenizer, _qwen_model
    if _qwen_model is not None:
        return
    from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: PLC0415
    _qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    _qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, dtype=torch.float32
    )
    _qwen_model.eval()


def qwen_predict_next_token(text: str, n_shown: int = 7):
    """
    Top-`n_shown` next-token predictions from a local Qwen2.5-0.5B instance.
    Returns (tokens, probs) where tokens is a list of decoded strings and
    probs is a numpy array of their probabilities.

    Drop-in replacement for gpt2_predict_next_token using a modern model.
    """
    _ensure_qwen_loaded()

    input_ids = _qwen_tokenizer.encode(
        text, add_special_tokens=False, return_tensors="pt"
    )
    with torch.no_grad():
        logits = _qwen_model(input_ids).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, n_shown)
    tokens = [_qwen_tokenizer.decode([i]) for i in top_indices.tolist()]
    return tokens, top_probs.numpy()


def qwen_token_probability(text: str, token: str) -> float:
    """
    Probability Qwen2.5-0.5B assigns to `token` following `text`.

    `token` is a raw string, which may encode to one or more Qwen tokens.
    When it spans multiple tokens, the joint probability of the continuation
    is returned (the product of successive next-token probabilities).

    Drop-in replacement for gpt2_token_probability using a modern model.
    """
    _ensure_qwen_loaded()

    input_ids = _qwen_tokenizer.encode(
        text, add_special_tokens=False, return_tensors="pt"
    )
    token_ids = _qwen_tokenizer.encode(token, add_special_tokens=False)

    probability = 1.0
    with torch.no_grad():
        for token_id in token_ids:
            logits = _qwen_model(input_ids).logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            probability *= probs[token_id].item()
            next_id = torch.tensor([[token_id]])
            input_ids = torch.cat([input_ids, next_id], dim=1)
    return probability


def qwen_sample_completions(
    prefix: str,
    n: int = 200,
    max_new_tokens: int = 8,
    temperature: float = 0.9,
    top_p: float = 0.95,
    seed: int = 0,
    stop_at_newline: bool = True,
    use_cache: bool = True,
) -> list[str]:
    """
    Sample `n` completions of `prefix` from Qwen2.5-0.5B, each returned as the
    full string (prefix included) continued by up to `max_new_tokens` sampled
    tokens.

    Results are deterministic given `seed` and cached to a JSON file next to
    this module (keyed by all sampling parameters), so repeated scene renders
    don't re-run the model. Pass use_cache=False to force regeneration.
    """
    key = (prefix, n, max_new_tokens, temperature, top_p, seed, stop_at_newline)
    digest = hashlib.md5(repr(key).encode()).hexdigest()[:10]
    cache_path = os.path.join(
        os.path.dirname(__file__), f"qwen_completions_{digest}.json"
    )
    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)["completions"]

    _ensure_qwen_loaded()
    torch.manual_seed(seed)

    input_ids = _qwen_tokenizer.encode(
        prefix, add_special_tokens=False, return_tensors="pt"
    )
    pad_id = _qwen_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = _qwen_tokenizer.eos_token_id

    completions: list[str] = []
    batch_size = 50
    with torch.no_grad():
        while len(completions) < n:
            k = min(batch_size, n - len(completions))
            batch_ids = input_ids.repeat(k, 1)
            outputs = _qwen_model.generate(
                batch_ids,
                attention_mask=torch.ones_like(batch_ids),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_id,
            )
            for row in outputs:
                text = _qwen_tokenizer.decode(row, skip_special_tokens=True)
                if stop_at_newline:
                    text = text.split("\n")[0]
                completions.append(text.rstrip())

    completions = completions[:n]

    if use_cache:
        payload = {
            "model": QWEN_MODEL_NAME,
            "prefix": prefix,
            "params": {
                "n": n, "max_new_tokens": max_new_tokens,
                "temperature": temperature, "top_p": top_p, "seed": seed,
                "stop_at_newline": stop_at_newline,
            },
            "completions": completions,
        }
        with open(cache_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return completions


HUFFMAN_TABLE_PATH = os.path.join(
    os.path.dirname(__file__), "huffman_table.json"
)


def _wiki_char_text() -> str:
    """Decode the nanoGPT wiki_char train.bin back into a string."""
    meta_path = f"{NANOGPT_DIR}/data/wiki_char/meta.pkl"
    bin_path = f"{NANOGPT_DIR}/data/wiki_char/train.bin"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    itos = meta["itos"]
    ids = np.fromfile(bin_path, dtype=np.uint16)
    return "".join(itos[int(i)] for i in ids)


def build_huffman_table(
    alphabet: str = CHAR_ALPHABET,
    save_path: str | None = HUFFMAN_TABLE_PATH,
) -> dict[str, str]:
    """
    Build a Huffman code table for `alphabet` using character frequencies
    from the lowercased nanoGPT wiki_char training corpus. Characters not
    in `alphabet` are ignored. Returns a dict mapping char -> bitstring.
    If `save_path` is given, also writes the result as JSON (sorted by
    code length, then char).
    """
    text = _wiki_char_text().lower()
    counts = Counter(c for c in text if c in alphabet)
    # Ensure every alphabet char appears, so all are encodable
    for c in alphabet:
        counts.setdefault(c, 1)

    # Heap entries: (weight, tie_breaker, node). Node is either a
    # single-char string (leaf) or a (left, right) tuple.
    heap: list[tuple[int, int, object]] = []
    for i, (ch, w) in enumerate(counts.items()):
        heapq.heappush(heap, (w, i, ch))
    next_id = len(heap)
    while len(heap) > 1:
        w1, _, n1 = heapq.heappop(heap)
        w2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (w1 + w2, next_id, (n1, n2)))
        next_id += 1
    root = heap[0][2]

    codes: dict[str, str] = {}

    def walk(node, prefix: str) -> None:
        if isinstance(node, str):
            codes[node] = prefix or "0"
            return
        left, right = node
        walk(left, prefix + "0")
        walk(right, prefix + "1")

    walk(root, "")

    if save_path is not None:
        ordered = dict(sorted(codes.items(), key=lambda kv: (len(kv[1]), kv[0])))
        payload = {
            "source": "nanoGPT/data/wiki_char/train.bin (lowercased)",
            "alphabet": alphabet,
            "total_chars_counted": sum(counts.values()),
            "counts": {c: counts[c] for c in ordered},
            "codes": ordered,
        }
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return codes


@lru_cache(maxsize=1)
def load_huffman_table(alphabet: str = CHAR_ALPHABET) -> dict[str, str]:
    """
    Load the cached Huffman table from disk, building it if missing.
    Rebuilds if the cached alphabet doesn't match.
    """
    if os.path.exists(HUFFMAN_TABLE_PATH):
        with open(HUFFMAN_TABLE_PATH) as f:
            payload = json.load(f)
        if payload.get("alphabet") == alphabet:
            return payload["codes"]
    return build_huffman_table(alphabet)


def total_information(input_text: str, context: str = " ") -> float:
    """
    Total information in bits the nanoGPT wiki_char model assigns to
    `input_text` given `context`: sum of -log2(P(c | context + earlier chars))
    over each character c in `input_text`. Slides a window across the full
    sequence so inputs longer than the model's block_size are handled.
    """
    _ensure_nano_loaded()

    context_ids = _nano_encode(context)
    target_ids = _nano_encode(input_text)
    if not target_ids:
        return 0.0

    full_ids = context_ids + target_ids
    n = len(full_ids)
    block_size = _nano_model.config.block_size
    stride = max(1, block_size // 2)

    i = max(1, len(context_ids))
    total_nats = 0.0
    while i < n:
        j = min(i + stride, n)
        a = max(0, j - block_size)
        window = full_ids[a:j]
        input_tensor = torch.tensor([window])
        with torch.no_grad():
            logits = _nano_model(input_tensor, input_tensor)[0][0]
        log_probs = torch.log_softmax(logits, dim=-1)

        positions = torch.arange(i - 1 - a, j - 1 - a)
        targets = torch.tensor(full_ids[i:j])
        total_nats += -log_probs[positions, targets].sum().item()
        i = j

    return float(total_nats / np.log(2))
