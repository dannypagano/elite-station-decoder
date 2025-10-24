import itertools
import multiprocessing as mp
from collections import Counter
import time

# --- Morse definitions ---
MORSE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
    'Z': '--..', ' ': '/'
}

STOPWORDS = ["THE", "AND", "OF", "TO", "IN", "IT", "IS", "BE", "AS", "AT", "BY"]

def text_to_morse(text):
    return ''.join(MORSE_DICT.get(c, '') for c in text if c in MORSE_DICT)

def collapse_runs_to_morse(cipher, mapping):
    return ''.join(mapping.get(ch, '?') for ch in cipher)

def evaluate_fit(cipher, morse_target):
    # sliding window match between cipher Morse and target Morse
    max_score = 0
    for i in range(len(cipher) - len(morse_target) + 1):
        segment = cipher[i:i + len(morse_target)]
        matches = sum(a == b for a, b in zip(segment, morse_target))
        score = matches / len(morse_target)
        if score > max_score:
            max_score = score
    return max_score

def generate_candidate_mappings(symbols):
    for dots in itertools.combinations(symbols, 3):
        dots = set(dots)
        for dashes in itertools.combinations([s for s in symbols if s not in dots], 3):
            dashes = set(dashes)
            spaces = [s for s in symbols if s not in dots | dashes]
            yield {s: '.' for s in dots} | {s: '-' for s in dashes} | {s: '/' for s in spaces}

import multiprocessing as mp
import itertools
import time

# Global shared data for workers
CIPHERTEXT = None
MORSE_TARGETS = None

def init_worker(ciphertext, morse_targets):
    global CIPHERTEXT, MORSE_TARGETS
    CIPHERTEXT = ciphertext
    MORSE_TARGETS = morse_targets

def evaluate_mapping(mapping):
    morse_guess = ''.join(mapping.get(ch, '?') for ch in CIPHERTEXT)
    best_local = max(
        sum(a == b for a, b in zip(morse_guess[i:i+len(t)], t)) / len(t)
        for t in MORSE_TARGETS
        for i in range(len(CIPHERTEXT) - len(t) + 1)
    )
    return (mapping, best_local)

import random

def generate_candidate_mappings(symbols, limit=100_000, seed=None):
    """Randomly sample mappings from 36 symbols → {'.', '-', 'x'}"""
    rng = random.Random(seed)
    morse_symbols = ['.', '-', 'x']
    for _ in range(limit):
        yield {s: rng.choice(morse_symbols) for s in symbols}


def pollux_solver_parallel(ciphertext,
                           n_processes=8,
                           score_threshold=0.98,
                           max_evaluations=100000,
                           verbose=False):
    symbols = sorted(set(ciphertext))
    morse_targets = [
        ".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..",
        ".---", "-.-", ".-..", "--", "-.", "---", ".--.", "--.-", ".-.",
        "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."
    ]
    morse_targets += [''.join(t) for t in itertools.permutations(['.', '-'], 3)]

    print(f"[INFO] Cipher has {len(symbols)} symbols: {symbols}")
    print(f"[INFO] {len(morse_targets)} morse targets generated")

    best_mapping, best_score = None, -float('inf')
    start_time = time.time()
    counter = 0

    mapping_gen = generate_candidate_mappings(symbols)

    with mp.get_context("spawn").Pool(
        processes=n_processes,
        initializer=init_worker,
        initargs=(ciphertext, morse_targets),
        maxtasksperchild=50
    ) as pool:
        for mapping, score in pool.imap_unordered(evaluate_mapping, mapping_gen, chunksize=200):
            counter += 1
            if score > best_score:
                best_mapping, best_score = mapping, score
                if verbose:
                    print(f"[DEBUG] New best {best_score:.4f} at eval {counter}")

            if score_threshold and best_score >= score_threshold:
                print(f"[INFO] Score threshold reached ({best_score:.4f}), stopping early.")
                pool.terminate()
                break

            if max_evaluations and counter >= max_evaluations:
                print(f"[INFO] Max evaluations reached ({max_evaluations}), stopping early.")
                pool.terminate()
                break

    elapsed = time.time() - start_time
    print(f"[INFO] Completed in {elapsed:.2f}s, evaluated {counter} mappings.")
    print(f"[RESULT] Best score: {best_score:.4f}")

    return best_mapping, best_score


if __name__ == "__main__":
    with open("message.txt", "r", encoding="utf-8") as f:
        ciphertext = f.read().strip()

    best_mapping, score = pollux_solver_parallel(ciphertext=ciphertext,
    n_processes=12,
    score_threshold=0.95,
    max_evaluations=50000,
    verbose=True
)

    print("Best mapping:", best_mapping)
    print("Best score:", score)