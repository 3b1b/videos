import collections
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import _2026.cross_entropy.next_char as next_char

ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,'!?"
SAMPLE_CHARS = 5_000_000
NGRAM_WORDS = 8
MIN_COUNT = 5
N_SCORED = 80
N_SHOWN = 25

GUESSES = [
    "is a species of sea snail, a marine gastropod mollusk in the family",
    "a marine gastropod mollusk in the family Conidae, the cone snails and their allies.",
    "is a village in the administrative district of Gmina",
    "is a moth of the family Noctuidae. It is found in",
    "is a species of beetle in the family",
    "is a genus of flowering plants in the family",
    "This article about a location in",
    "You can help Wikipedia by expanding it.",
    "is a census-designated place in",
    "The population was",
]


def find_frequent_phrases(text):
    words = text.split()
    counts = collections.Counter(
        " ".join(words[i:i + NGRAM_WORDS])
        for i in range(len(words) - NGRAM_WORDS)
    )
    allowed = set(ALPHABET)
    phrases = [
        (count, phrase)
        for phrase, count in counts.items()
        if count >= MIN_COUNT and set(phrase) <= allowed
    ]
    phrases.sort(key=lambda pair: pair[0] * len(pair[1]), reverse=True)
    return phrases[:N_SCORED]


def score(phrase):
    bits = next_char.total_information(phrase, " ")
    return bits, bits / len(phrase)


if __name__ == "__main__":
    next_char._ensure_nano_loaded()
    print(f"Model context length: {next_char._nano_model.config.block_size} characters\n")

    print("Hand-picked guesses")
    for phrase in GUESSES:
        bits, per_char = score(phrase)
        print(f"  {per_char:5.2f} bits/char  {bits:6.1f} bits  {len(phrase):3d} chars  {phrase!r}")

    text = next_char._wiki_char_text()[:SAMPLE_CHARS]
    results = []
    for count, phrase in find_frequent_phrases(text):
        bits, per_char = score(phrase)
        results.append((per_char, bits, count, phrase))
    results.sort()

    print(f"\nMost predictable frequent {NGRAM_WORDS}-word phrases in the first {SAMPLE_CHARS:,} training characters")
    for per_char, bits, count, phrase in results[:N_SHOWN]:
        print(f"  {per_char:5.2f} bits/char  {bits:6.1f} bits  x{count:<5d} {phrase!r}")