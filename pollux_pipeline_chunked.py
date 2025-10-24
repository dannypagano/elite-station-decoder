#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pollux Cipher Combined Pipeline (Decoder + Optimizer)
- Threaded scan and optimizer
- GPU/NumPy-accelerated scoring (PyTorch CUDA/MPS or NumPy)
- Bounded merge iterations and pool sizes
- Config-driven (no CLI flags)
"""


# ======= CONFIG (edit me) =======
CONFIG = {
    # ---------- Inputs ----------
    "corpus": "./corpus",                      # Folder with *.txt files for building n-grams
    "ciphertext": None,                        # Paste ciphertext here (if set, overrides file)
    "ciphertext_file": "./cipher.txt",         # File path used when ciphertext is None

    # ---------- Concurrency ----------
    "threads": 8,                              # Thread workers for optimizer & scan (good default on M3 Air)
    "max_workers": 8,                          # Decoder scanning threads for chunked search

    # ---------- N-gram model (smaller = faster for quick tests) ----------
    "nmin": 1,                                 # Minimum n-gram length
    "nmax": 3,                                 # Maximum n-gram length (≤3 is much faster than 4–5)
    "topk": 2000,                              # Keep top-K most frequent n-grams
    "min_count": 3,                            # Drop n-grams below this frequency (noise filter)
    "min_token_len": 3,                        # Drop tokens shorter than this many characters

    # ---------- Decoder scanning ----------
    "chunk_size": 20000,                       # Cipher chunk size; 11k fits in one chunk (fine to keep)
    "max_per_word_candidates": 5,              # Cap: keep at most this many candidate windows per word

    # ---------- Merge (key limits to keep it bounded) ----------
    "max_merge_iter": 5,                       # Hard cap on merge iterations
    "max_merged_pool": 20000,                  # After each merge iteration, keep only top-N mappings

    # ---------- Constraints from decoder -> optimizer ----------
    "take_top_cands": 12,                      # Use top-N decoder mappings to form symbol constraints
    "require_unanimous": True,                 # Only keep a constraint if all top-N agree

    # ---------- Quick model (chunked search space) ----------
    "quick_chunk_evals": 20000,                # Evaluations per chunk in the quick pass
    "quick_targets_top_ngram": 120,            # Build small target set from top-N n-grams (fast)
    "quick_threshold": 0.982,                  # Early-stop threshold inside each quick chunk
    "quick_top_per_chunk": 5,                  # Keep this many best mappings from each chunk
    "quick_global_top": 30,                    # After all chunks, keep this many best for refinement

    # ---------- Refinement phase ----------
    "refine_rounds": 3,                        # Number of refinement rounds
    "refine_neighbors_per_map": 2000,          # Neighbors to try per seed mapping each round
    "refine_threshold": 0.985,                 # Early-stop threshold in refinement
    "refine_max_evals": 120000,                # Total evals cap in refinement

    # ---------- Full optimizer (fallback / legacy single-stage) ----------
    "opt_max_evals": 20000,                    # Max mapping evaluations if calling legacy optimizer directly
    "opt_threshold": 0.983,                    # Early-stop threshold for legacy optimizer

    # ---------- Target set for legacy/full optimizer ----------
    "targets_top_ngram": 200,                  # Use Morse of top-N n-grams as additional targets
}
# =================================


# ==== BEGIN Embedded: PolluxDecoder.py ====
#!/usr/bin/env python3
"""
Optimized decode pipeline:
- builds n-grams from corpus
- parallel chunk scanning for candidate windows
- aggressive merging of partial mappings
- decode + fast Aho-Corasick scoring of decoded candidates
- export top results to CSV

Drop-in, pure-Python (no external libraries required).
"""

import os
import glob
import csv
import pprint
import time
from datetime import datetime
from collections import defaultdict, Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple

# ---------------------------
# Configuration
# ---------------------------
CORPUS_FOLDER = r"corpus"
MESSAGE_PATH = r"message.txt"
OUTPUT_DIR = r"output"
NGRAM_RANGE = (1, 5)
TOP_K_NGRAMS = 5000
MIN_COUNT = 2
MIN_TOKEN_LEN = 3

CHUNK_SIZE = 20000
MAX_WORKERS = 8

EXPORT_K = 30000

# ---------------------------
# Morse dictionary + allowed chars
# ---------------------------
MORSE = {
    'a': '.-', 'b': '-...', 'c': '-.-.', 'd': '-..', 'e': '.',
    'f': '..-.', 'g': '--.', 'h': '....', 'i': '..', 'j': '.---',
    'k': '-.-', 'l': '.-..', 'm': '--', 'n': '-.', 'o': '---',
    'p': '.--.', 'q': '--.-', 'r': '.-.', 's': '...', 't': '-',
    'u': '..-', 'v': '...-', 'w': '.--', 'x': '-..-', 'y': '-.--',
    'z': '--..',
    '0': '-----','1': '.----','2': '..---','3': '...--','4': '....-',
    '5': '.....','6': '-....','7': '--...','8': '---..','9': '----.',
    '.': '.-.-.-', ',': '--..--', '?': '..--..', '!': '-.-.--'
}

ALLOWED_CHARS = set(MORSE.keys()) | {" "}
LETTER_SEP = '|'
WORD_SEP = '||'

# ---------------------------
# Lightweight Aho-Corasick (pure Python)
# ---------------------------
# Build automaton from patterns -> returns an object with `find_all(text)` yielding (end_index, matched_pattern)
class AhoCorasick:
    def __init__(self, patterns: List[str]):
        # patterns should be lowercased
        self._build_trie(patterns)

    def _build_trie(self, patterns: List[str]):
        # trie: each node is dict char -> node_index
        # output: node_index -> list of patterns ending at that node
        self.trie = []
        self.out = []
        self.fail = []
        self.trie.append({})  # root 0
        self.out.append([])
        self.fail.append(0)
        for pat in patterns:
            node = 0
            for ch in pat:
                if ch not in self.trie[node]:
                    self.trie[node][ch] = len(self.trie)
                    self.trie.append({})
                    self.out.append([])
                    self.fail.append(0)
                node = self.trie[node][ch]
            self.out[node].append(pat)
        # build failure links BFS
        q = deque()
        for ch, nxt in self.trie[0].items():
            self.fail[nxt] = 0
            q.append(nxt)
        while q:
            r = q.popleft()
            for ch, s in self.trie[r].items():
                q.append(s)
                state = self.fail[r]
                while state and ch not in self.trie[state]:
                    state = self.fail[state]
                self.fail[s] = self.trie[state].get(ch, 0)
                self.out[s].extend(self.out[self.fail[s]])

    def find_all(self, text: str):
        """Yield (index_of_end, matched_pattern) for every match found."""
        node = 0
        for i, ch in enumerate(text):
            while node and ch not in self.trie[node]:
                node = self.fail[node]
            node = self.trie[node].get(ch, 0)
            if self.out[node]:
                for pat in self.out[node]:
                    yield (i, pat)

# ---------------------------
# Utilities: n-grams, morse pattern, window checking
# ---------------------------
def clean_and_build_ngrams(folder: str, n_range=(1,3), top_k=5000, min_count=2, min_token_len=2):
    """Build n-grams from all .txt files in folder. Returns (words_list, freqs, diagnostics)."""
    print(f"[INFO] Building n-grams from: {folder}")
    t0 = time.time()
    raw_text = []
    files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    if not files:
        print("[WARN] No files found in corpus folder.")
    for path in files:
        print(f"  loading {os.path.basename(path)}")
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                raw_text.append(fh.read().lower())
        except Exception as e:
            print(f"  warning reading {path}: {e}")
    text = " ".join(raw_text)
    # restrict characters
    filtered = "".join(ch if ch in ALLOWED_CHARS else " " for ch in text)
    tokens = [t for t in filtered.split() if len(t) >= min_token_len and any(c.isalnum() for c in t)]
    print(f"  tokens after cleaning: {len(tokens)}")
    ngram_counts = Counter()
    for n in range(n_range[0], n_range[1] + 1):
        print(f"  generating {n}-grams...")
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i + n])
            ngram_counts[ngram] += 1
    # filter by min_count and take top_k
    filtered_counts = {k: v for k, v in ngram_counts.items() if v >= min_count}
    most_common = Counter(filtered_counts).most_common(top_k)
    words = [w for w, _ in most_common]
    freqs = dict(most_common)
    diagnostics = {"raw_tokens": len(text.split()), "filtered_tokens": len(tokens), "unique_ngrams": len(filtered_counts)}
    print(f"[DONE] n-grams built: {len(words)} items in {time.time()-t0:.1f}s")
    return words, freqs, diagnostics

def word_to_bordered_morse(word: str) -> str:
    # word is expected lowercase and may contain spaces for multiword ngrams.
    parts = []
    for ch in word:
        if ch == " ":
            # when joining multi-word ngrams we will join by WORD_SEP later.
            parts.append(WORD_SEP[:-1])  # add one '|' to mark separation; we'll join tokens at call site
        elif ch in MORSE:
            parts.append(MORSE[ch])
        else:
            # unknown char -> skip
            pass
    # The caller will usually build pattern per token; we provide helper below.
    return parts  # list of morse tokens and separators fragment

def morse_pattern_for_phrase(phrase: str) -> str:
    """
    Convert phrase (possibly multiword) to a morse class pattern string,
    WITHOUT adding leading or trailing WORD_SEP. Caller decides whether to
    add boundaries when scanning.
    """
    tokens = phrase.split()
    parts = []
    for token in tokens:
        letters = [MORSE[ch] for ch in token if ch in MORSE]
        if not letters:
            continue
        parts.append(LETTER_SEP.join(letters))
    if not parts:
        return ""
    # join word tokens with WORD_SEP (which is '||') but do NOT append trailing WORD_SEP here
    return WORD_SEP.join(parts)

def window_candidate(window: str, morse_pat: str) -> Dict[str, str] or None:
    """Return partial mapping dict glyph->class if window matches morse pattern, else None."""
    if not morse_pat or len(window) != len(morse_pat):
        return None
    # reject repeat adjacent glyphs
    for i in range(len(window)-1):
        if window[i] == window[i+1]:
            return None
    mapping = {}
    for g, m in zip(window, morse_pat):
        cls = {'|':'S', '.':'D', '-':'H'}.get(m)
        if cls is None:
            return None
        if g in mapping and mapping[g] != cls:
            return None
        mapping[g] = cls
    return mapping

def merge_mappings(m1: Dict[str,str], m2: Dict[str,str]) -> Dict[str,str] or None:
    merged = dict(m1)
    for k, v in m2.items():
        if k in merged and merged[k] != v:
            return None
        merged[k] = v
    return merged
from collections import defaultdict

def fast_merge_candidate_sets(candidate_sets):
    print("[INFO] Fast merging candidates by glyph overlap...")
    grouped = defaultdict(list)
    for cset in candidate_sets:
        for word, cands in cset.items():
            for c in cands:
                key = tuple(sorted(c['mapping'].keys()))
                grouped[key].append(c['mapping'])
    merged = []
    for group_key, maps in grouped.items():
        merged_dict = {}
        for m in maps:
            merged_dict.update(m)
        merged.append({'mapping': merged_dict, 'count': len(maps)})
    print(f"[DONE] Reduced {sum(len(v) for v in grouped.values())} → {len(merged)} mappings")
    return merged

# ---------------------------
# Parallel chunk processing
# ---------------------------
def process_chunk_worker(args):
    """Worker wrapper to be picklable for ProcessPoolExecutor."""
    chunk, offset, words, patterns = args
    return process_chunk(chunk, offset, words, patterns)

def process_chunk(cipher_chunk: str, offset: int, words: List[str], patterns: Dict[str,str]):
    """Scan one ciphertext chunk for n-gram matches using precomputed morse patterns.
       Returns a dict: word -> list of candidate dicts with window, mapping, start.
    """
    candidates = defaultdict(list)
    # iterate words; patterns dict contains morse pattern string and its length
    total = len(words)
    for idx, w in enumerate(words, start=1):
        if idx % 1000 == 0 or idx == total:
            print(f"    [chunk {offset}] scanning word {idx}/{total}")
        pat = patterns[w]
        if not pat:
            continue
        L = len(pat)
        # iterate sliding windows
        # micro-optimizations: localize variables
        wc = cipher_chunk
        for i in range(len(wc) - L + 1):
            win = wc[i:i+L]
            m = window_candidate(win, pat)
            if m:
                candidates[w].append({"window": win, "mapping": m, "start": offset + i})
    
        # Cap per-word candidate windows to keep merging tractable
        try:
            _MAX_PER = CONFIG.get("max_per_word_candidates", 5)
        except NameError:
            _MAX_PER = 5
        if isinstance(candidates, dict) and _MAX_PER:
            for _w, _items in list(candidates.items()):
                candidates[_w] = sorted(_items, key=lambda x: x.get("start", 0))[:_MAX_PER]

return candidates

def parallel_scan(ciphertext: str, words: List[str], patterns: Dict[str,str], chunk_size=20000, max_workers=4):
    """Split ciphertext into chunks and process in parallel, returning list of candidate dicts."""
    print(f"[INFO] Parallel scan: ciphertext length {len(ciphertext)}, chunk_size {chunk_size}, workers {max_workers}")
    args_list = []
    for i in range(0, len(ciphertext), chunk_size):
        args_list.append((ciphertext[i:i+chunk_size], i, words, patterns))
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_chunk_worker, a) for a in args_list]
        for fut in as_completed(futures):
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                print(f"[!] worker error: {e}")
    return results

# ---------------------------
# Merge candidate sets (aggressive merging)
# ---------------------------
def merge_candidate_sets(cand_sets: List[Dict[str, List[Dict]]], min_merge_gain=1):
    print("[INFO] Merging candidate mappings...")
    t0 = time.time()
    partial_mappings = defaultdict(lambda: {"count": 0, "examples": []})
    for all_candidates in cand_sets:
        for word, cand_list in all_candidates.items():
            for cand in cand_list:
                sig = tuple(sorted(cand["mapping"].items()))
                partial_mappings[sig]["count"] += 1
                partial_mappings[sig]["examples"].append({"word": word, **cand})
    # build initial pool
    pool = [{"mapping": dict(sig), "count": info["count"], "examples": info["examples"]} for sig, info in partial_mappings.items()]
    changed = True
    iteration = 0
    # greedy pairwise merging: merge b into a when they are consistent and increase mapping size
    while changed and iteration < CONFIG.get('max_merge_iter', 5):
        iteration += 1
        print(f"  merge iteration {iteration}, pool size {len(pool)}")
        changed = False
        new_pool = []
        used = set()
        for i, a in enumerate(pool):
            if i in used:
                continue
            merged_map = dict(a["mapping"])
            total_count = a["count"]
            examples = list(a["examples"])
            for j in range(i+1, len(pool)):
                if j in used:
                    continue
                b = pool[j]
                m = merge_mappings(merged_map, b["mapping"])
                if m and len(m) - len(merged_map) >= min_merge_gain:
                    merged_map = m
                    total_count += b["count"]
                    examples.extend(b["examples"])
                    used.add(j)
                    changed = True
            new_pool.append({"mapping": merged_map, "count": total_count, "examples": examples})
        pool = new_pool
    # deduplicate and keep the most complete versions
    merged_final = {}
    for item in pool:
        sig = tuple(sorted(item["mapping"].items()))
        if sig not in merged_final or len(item["mapping"]) > len(merged_final[sig]["mapping"]):
            merged_final[sig] = item
    max_size = max((len(v["mapping"]) for v in merged_final.values()), default=0)
    filtered = [v for v in merged_final.values() if len(v["mapping"]) >= max_size - 1]
    print(f"[DONE] Merging finished in {time.time()-t0:.1f}s, final candidates: {len(filtered)}")
    return filtered

# ---------------------------
# Fast decoding + scoring using Aho-Corasick
# ---------------------------
def weighted_word_score(word_counts, word_freqs,
                        min_length=3, length_exponent=5,
                        single_letter_cap=5, single_letter_weight=0.1, alpha=0.7):
    score = 0
    for w, c in word_counts.items():
        raw_len = len(w.replace(" ",""))
        freq = word_freqs.get(w, 1)
        if raw_len < min_length:
            capped = min(c, single_letter_cap)
            score += single_letter_weight * (capped**2) * (freq ** alpha)
            continue
        score += (c**2) * (freq ** alpha) * (raw_len ** length_exponent)
    return score
def morse_to_plain(decoded_morse: str) -> str:
    """
    Convert a decoded_morse string (symbols '.', '-', '|' with LETTER_SEP and WORD_SEP)
    into plain text, where WORD_SEP -> a single space, and LETTER_SEP separates morse letters.
    Preserves leading/trailing word spaces if present.
    """
    rev_morse = {v: k for k, v in MORSE.items()}
    out_tokens = []
    i = 0
    L = len(decoded_morse)
    # We'll treat WORD_SEP ('||') as an explicit space token
    while i < L:
        # if we see a WORD_SEP at this position, append a space and advance by 2
        if i + 1 < L and decoded_morse[i:i+2] == WORD_SEP:
            out_tokens.append(" ")   # preserve explicit space token
            i += 2
            continue
        # otherwise collect up to next WORD_SEP
        j = decoded_morse.find(WORD_SEP, i)
        if j == -1:
            seg = decoded_morse[i:]
            i = L
        else:
            seg = decoded_morse[i:j]
            i = j
        # seg now contains letters separated by LETTER_SEP ('|')
        letters = []
        for morse_letter in seg.split(LETTER_SEP):
            if not morse_letter:
                continue
            letters.append(rev_morse.get(morse_letter, "_"))
        if letters:
            out_tokens.append("".join(letters).lower())
    # join tokens but avoid merging the explicit spaces with surrounding words incorrectly:
    # tokens list may contain words and " " items; join with '' then normalize multi-space to single space
    joined = "".join(t if t == " " else t for t in out_tokens)
    # normalize runs of spaces to a single space, and strip only trailing spaces (but preserve a leading space if present)
    # preserve leading space if present
    has_lead_space = joined.startswith(" ")
    normalized = " ".join(joined.split())
    if has_lead_space and not normalized.startswith(" "):
        normalized = " " + normalized
    return normalized

def decode_and_score(ciphertext: str, merged_mappings: List[Dict], words: List[str], word_freqs: Dict[str,int]):
    print(f"[INFO] Decoding {len(merged_mappings)} candidate mappings and scoring...")
    t0 = time.time()
    patterns = [w.lower() for w in words]
    automaton = AhoCorasick(patterns)

    rev_morse = {v: k for k, v in MORSE.items()}
    class_to_symbol = {"D": ".", "H": "-", "S": "|"}

    results = []
    total = len(merged_mappings)
    for idx, m in enumerate(merged_mappings, start=1):
        if idx % 10 == 0 or idx == total:
            print(f"  scoring mapping {idx}/{total}")
        mapping = m["mapping"]
        decoded_class = "".join(mapping.get(g, "_") for g in ciphertext)
        decoded_morse = "".join(class_to_symbol.get(c, "_") for c in decoded_class)
        decoded_text = morse_to_plain(decoded_morse)

        # --- deduplicate overlapping word matches ---
        matches = []
        for end_idx, pat in automaton.find_all(decoded_text):
            start_idx = end_idx - len(pat) + 1
            matches.append((start_idx, end_idx, pat))
        matches.sort(key=lambda x: (x[0], -(x[1]-x[0])))

        non_overlapping = []
        last_end = -1
        for s, e, pat in matches:
            if s > last_end:
                non_overlapping.append(pat)
                last_end = e

        # count only non-overlapping words
        word_counts = Counter(non_overlapping)

        score = weighted_word_score(word_counts, word_freqs)
        results.append({
            "mapping": mapping,
            "decoded_text": decoded_text,
            "word_counts": word_counts,
            "total_hits": sum(word_counts.values()),
            "score": score
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"[DONE] Decoding & scoring in {time.time()-t0:.1f}s")
    return results


# ---------------------------
# CSV export
# ---------------------------
def export_results(rows: List[Dict], words: List[str], export_dir: str, export_k: int = 1000):
    os.makedirs(export_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(export_dir, f"decoded_results_{timestamp}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["rank", "total_matches", "word_hits", "glyph_mapping", "top_words", "decoded_preview"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rank, r in enumerate(rows[:export_k], start=1):
            mapping = r["mapping"]
            decoded_text = r["decoded_text"]
            word_counts = r["word_counts"]
            writer.writerow({
                "rank": rank,
                "total_matches": r.get("total_hits", 0),
                "word_hits": sum(word_counts.values()),
                "glyph_mapping": "; ".join(f"{k}:{v}" for k, v in mapping.items()),
                "top_words": ", ".join(f"{w}({c})" for w, c in word_counts.most_common(10)),
                "decoded_preview": decoded_text
            })
    print(f"[INFO] Exported results to: {path}")
    return path

# ---------------------------
# Main runner
# ---------------------------
def main():
    # 1) build n-grams
    words, word_freqs, diag = clean_and_build_ngrams(CORPUS_FOLDER, n_range=NGRAM_RANGE,
                                                     top_k=TOP_K_NGRAMS, min_count=MIN_COUNT,
                                                     min_token_len=MIN_TOKEN_LEN)
    print("Corpus diagnostics:", diag)
    print("Sample words:", words[:20])

    # 2) load ciphertext
    with open(MESSAGE_PATH, "r", encoding="utf-8") as fh:
        ciphertext = fh.read().strip().replace("\n", "").replace(" ", "")
    print(f"[INFO] ciphertext length = {len(ciphertext)}")

    # 3) precompute morse pattern strings for each word
    print("[INFO] Precomputing morse patterns for words...")
    patterns = {}
    for i, w in enumerate(words, start=1):
        if i % 2000 == 0:
            print(f"  patterns computed: {i}/{len(words)}")
        core = morse_pattern_for_phrase(w)
        if core:
            # require a word boundary before and after when scanning windows
            patterns[w] = WORD_SEP + core + WORD_SEP
        else:
            patterns[w] = ""

    # 4) parallel scan in chunks
    print("[INFO] Starting parallel chunk scan...")
    start = time.time()
    cand_sets = parallel_scan(ciphertext, words, patterns, chunk_size=CHUNK_SIZE, max_workers=MAX_WORKERS)
    print(f"[DONE] Parallel scan took {time.time()-start:.1f}s; collected {len(cand_sets)} chunk results")

    # 5) merge candidate sets into hypothesized mappings
    merged = fast_merge_candidate_sets(cand_sets)
    print(f"[INFO] merged candidate count: {len(merged)}")

    # 6) decode + score using Aho-Corasick
    ranked = decode_and_score(ciphertext, merged, words, word_freqs)

    # 7) export
    export_path = export_results(ranked, words, OUTPUT_DIR, export_k=EXPORT_K)

    # 8) print top few
    TOP_K = min(10, len(ranked))
    for rnk in range(TOP_K):
        r = ranked[rnk]
        print(f"\n--- Rank {rnk+1} | score {r['score']} | hits {r['total_hits']} ---")
        print("Top matched words:", r["word_counts"].most_common(10))
        print("Decoded preview:", r["decoded_text"][:300])

    return export_path


# [Main block stripped to EOF]

# ==== END Embedded: PolluxDecoder.py ====

# ==== BEGIN Embedded: polluxcrypto.py ====
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



# [Main block stripped to EOF]

# ==== END Embedded: polluxcrypto.py ====

# ==== BEGIN Glue & Pipeline (no CLI) ====
import itertools
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional high-performance backends
try:
    import torch
    _TORCH = True
    _TORCH_DEVICE = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )
except Exception:
    torch = None
    _TORCH = False
    _TORCH_DEVICE = "cpu"

try:
    import numpy as _np
    from numpy.lib.stride_tricks import sliding_window_view as _sliding_window_view
    _NUMPY = True
except Exception:
    _np = None
    _sliding_window_view = None
    _NUMPY = False

CLASS_TO_SYMBOL = {"D": ".", "H": "-", "S": "/"}
CONFIG.setdefault("threads", 8)

def class_map_to_symbol_map(class_map):
    return {k: CLASS_TO_SYMBOL[v] for k, v in class_map.items() if v in CLASS_TO_SYMBOL}

def constraints_from_candidates(merged_list, top=10, require_unanimous=True):
    votes = defaultdict(lambda: Counter())
    take = merged_list[:max(1, top)]
    for item in take:
        for sym, cls in item["mapping"].items():
            votes[sym][cls] += 1
    constraints = {}
    for sym, cnts in votes.items():
        best_cls, best_votes = cnts.most_common(1)[0]
        if require_unanimous:
            if best_votes == len(take):
                constraints[sym] = CLASS_TO_SYMBOL.get(best_cls, None)
        else:
            constraints[sym] = CLASS_TO_SYMBOL.get(best_cls, None)
    return {k:v for k,v in constraints.items() if v is not None}

def decoder_build_patterns(words):
    pat = {}
    for w in words:
        p = morse_pattern_for_phrase(w)
        if p:
            pat[w] = p
    return pat

def convert_morse_classes_to_optimizer(morse_str):
    # decoder uses '|' for letter sep and '||' for word sep; optimizer uses '/' for any sep
    return morse_str.replace('||', '//').replace('|', '/')

# ---- High-performance scorer preparation ----
_SYMBOL_TO_INT = {'.': 1, '-': 2, '/': 0}  # 3-class encoding

def _encode_morse_str_to_ints(s: str):
    return [_SYMBOL_TO_INT.get(ch, 0) for ch in s]

def _prep_targets_torch(morse_targets):
    groups = defaultdict(list)
    for t in morse_targets:
        ints = _encode_morse_str_to_ints(t)
        if ints:
            groups[len(ints)].append(ints)
    stacked = {}
    for L, seqs in groups.items():
        T = torch.tensor(seqs, dtype=torch.int16, device=_TORCH_DEVICE)
        stacked[L] = T
    return stacked

def _prep_targets_numpy(morse_targets):
    groups = defaultdict(list)
    for t in morse_targets:
        ints = _encode_morse_str_to_ints(t)
        if ints:
            groups[len(ints)].append(ints)
    stacked = {}
    for L, seqs in groups.items():
        stacked[L] = _np.asarray(seqs, dtype=_np.int16)
    return stacked

# Globals used by evaluators
CIPHERTEXT = None
MORSE_TARGETS = None
_TARGETS_TORCH = None
_TARGETS_NUMPY = None

def prepare_targets(morse_targets):
    global MORSE_TARGETS, _TARGETS_TORCH, _TARGETS_NUMPY
    MORSE_TARGETS = morse_targets
    _TARGETS_TORCH = _prep_targets_torch(morse_targets) if _TORCH else None
    _TARGETS_NUMPY = _prep_targets_numpy(morse_targets) if _NUMPY else None

def _mapping_to_morse_int_array(mapping, ciphertext):
    return _np.fromiter((
        {'.':1,'-':2,'/':0}.get(mapping.get(ch, '/'), 0) for ch in ciphertext
    ), dtype=_np.int16, count=len(ciphertext)) if _NUMPY else None

def _score_torch(morse_guess_np):
    g = torch.from_numpy(morse_guess_np).to(_TORCH_DEVICE)
    best = torch.tensor(0.0, device=_TORCH_DEVICE)
    N = g.shape[0]
    for L, T in _TARGETS_TORCH.items():
        if L > N:
            continue
        W = g.unfold(0, L, 1)            # [S, L]
        eq = (W.unsqueeze(1) == T.unsqueeze(0))  # [S, K, L]
        matches = eq.sum(dim=-1)         # [S, K]
        local_best = matches.max().to(torch.float32) / float(L)
        best = torch.maximum(best, local_best)
    return float(best.item())

def _score_numpy(morse_guess_np):
    N = morse_guess_np.shape[0]
    best = 0.0
    for L, T in _TARGETS_NUMPY.items():
        if L > N:
            continue
        W = _sliding_window_view(morse_guess_np, window_shape=L)  # [S, L]
        eq = (W[:, None, :] == T[None, :, :])                     # [S, K, L]
        matches = eq.sum(axis=2)                                  # [S, K]
        local_best = matches.max() / float(L)
        if local_best > best:
            best = float(local_best)
    return best

def evaluate_mapping(mapping):
    """GPU/NumPy-accelerated evaluator with thread-safe globals preparation."""
    if _NUMPY:
        morse_guess_np = _mapping_to_morse_int_array(mapping, CIPHERTEXT)
    else:
        morse_guess_np = None
    if _TORCH and _TARGETS_TORCH is not None and morse_guess_np is not None:
        return (mapping, _score_torch(morse_guess_np))
    elif _NUMPY and _TARGETS_NUMPY is not None and morse_guess_np is not None and _sliding_window_view is not None:
        return (mapping, _score_numpy(morse_guess_np))
    else:
        # Fallback simple scorer
        morse_guess = ''.join(mapping.get(ch, '/') for ch in CIPHERTEXT)
        best_local = 0.0
        for t in MORSE_TARGETS:
            L = len(t)
            for i in range(len(CIPHERTEXT) - L + 1):
                segment = morse_guess[i:i+L]
                matches = sum(a == b for a, b in zip(segment, t))
                score = matches / L
                if score > best_local:
                    best_local = score
        return (mapping, best_local)

def generate_candidate_mappings_constrained(symbols, constraints, dots_total=3, dashes_total=3):
    sym_set = list(symbols)
    pre_dots  = {s for s, v in constraints.items() if v == '.'}
    pre_dash  = {s for s, v in constraints.items() if v == '-'}
    pre_space = {s for s, v in constraints.items() if v == '/'}

    if len(pre_dots) > dots_total or len(pre_dash) > dashes_total:
        return
        yield

    rem_dots = dots_total - len(pre_dots)
    rem_dash = dashes_total - len(pre_dash)

    undecided = [s for s in sym_set if s not in constraints]
    for dots_choice in itertools.combinations(undecided, rem_dots):
        rem_after_dots = [s for s in undecided if s not in dots_choice]
        for dash_choice in itertools.combinations(rem_after_dots, rem_dash):
            mapping = {}
            for s in pre_dots:  mapping[s] = '.'
            for s in pre_dash:  mapping[s] = '-'
            for s in pre_space: mapping[s] = '/'
            for s in dots_choice: mapping[s] = '.'
            for s in dash_choice: mapping[s] = '-'
            for s in rem_after_dots:
                if s not in dash_choice:
                    mapping[s] = '/'
            yield mapping

def optimize_with_constraints(ciphertext, morse_targets, constraints, n_processes=8, max_evaluations=200000, threshold=0.99, verbose=True):
    """Multithreaded optimizer with GPU/NumPy-accelerated scoring."""
    symbols = sorted(set(ciphertext))
    global CIPHERTEXT
    CIPHERTEXT = ciphertext
    prepare_targets(morse_targets)

    best_mapping, best_score = None, -1.0
    counter = 0
    start = time.time()

    threads = max(1, int(CONFIG.get("threads", n_processes)))
    inflight_target = threads * 6

    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = []
        gen = generate_candidate_mappings_constrained(symbols, constraints)
        exhausted = False

        while True:
            while not exhausted and len(futures) < inflight_target and (not max_evaluations or counter < max_evaluations):
                try:
                    mapping = next(gen)
                except StopIteration:
                    exhausted = True
                    break
                counter += 1
                futures.append(ex.submit(evaluate_mapping, mapping))

            if not futures:
                break

            take = min(len(futures), threads)
            for fut in as_completed(futures[:take]):
                futures.remove(fut)
                mapping, score = fut.result()
                if score > best_score:
                    best_mapping, best_score = mapping, score
            if verbose:
                print(f"[opt] total {counter}, best={best_score:.4f} [{_TORCH_DEVICE if _TORCH else ('numpy' if _NUMPY else 'python')}]")

            if threshold and best_score >= threshold:
                if verbose:
                    print(f"[opt] Threshold {threshold:.3f} reached at {counter} evals.")
                for f in futures: f.cancel()
                futures.clear()
                break

            if exhausted and not futures:
                break

    if verbose:
        print(f"[opt] Done in {time.time()-start:.2f}s, evaluations={counter}, best={best_score:.4f}")
    return best_mapping, best_score

def parallel_scan_threaded(ciphertext, words, patterns, chunk_size=20000, max_workers=4):
    """Threaded version of decoder.parallel_scan to avoid multiprocessing on macOS."""
    args_list = []
    for i in range(0, len(ciphertext), chunk_size):
        args_list.append((ciphertext[i:i+chunk_size], i, words, patterns))

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(process_chunk, chunk, offset, words, patterns) for (chunk, offset, _, _) in args_list]
        for fut in as_completed(futs):
            try:
                res = fut.result()
                # Apply per-word cap (defensive)
                _MAX_PER = CONFIG.get("max_per_word_candidates", 5)
                if _MAX_PER:
                    for _word, _items in list(res.items()):
                        res[_word] = sorted(_items, key=lambda x: x.get("start", 0))[:_MAX_PER]
                results.append(res)
            except Exception as e:
                print(f"[!] thread worker error: {e}")
    return results

def decode_with_mapping(ciphertext, mapping):
    inv = {'.': '.', '-': '-', '/': '|'}
    morse_seq = ''.join(inv.get(mapping.get(ch, '/'), '|') for ch in ciphertext)
    return morse_to_plain(morse_seq), morse_seq


# ----- Chunked quick-pass and refinement optimizer -----
import random

def build_targets_quick(words):
    base = [
        ".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..",
        ".---", "-.-", ".-..", "--", "-.", "---", ".--.", "--.-", ".-.",
        "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."
    ]
    base += [''.join(t) for t in itertools.permutations(['.', '-'], 3)]
    n = CONFIG.get('quick_targets_top_ngram', 120)
    ngram_targets = [convert_morse_classes_to_optimizer(morse_pattern_for_phrase(w)) for w in words[:n]]
    return base + [t for t in ngram_targets if t]

def evaluate_quick_chunks(ciphertext, symbols, constraints, morse_targets, per_chunk_evals, top_per_chunk, threshold, threads):
    global CIPHERTEXT
    CIPHERTEXT = ciphertext
    prepare_targets(morse_targets)

    best_overall = []
    gen = generate_candidate_mappings_constrained(symbols, constraints)
    done = False
    while not done:
        chunk = []
        for _ in range(per_chunk_evals):
            try:
                chunk.append(next(gen))
            except StopIteration:
                done = True
                break
        if not chunk:
            break

        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = [ex.submit(evaluate_mapping, m) for m in chunk]
            scored = []
            for fut in as_completed(futs):
                m, s = fut.result()
                scored.append((m, s))

            scored.sort(key=lambda x: x[1], reverse=True)
            best_overall.extend(scored[:top_per_chunk])

        unique = {}
        for m, s in best_overall:
            key = tuple(sorted(m.items()))
            if key not in unique or s > unique[key][1]:
                unique[key] = (m, s)
        best_overall = sorted(unique.values(), key=lambda x: x[1], reverse=True)[:CONFIG.get("quick_global_top", 30)]
    return best_overall

def neighbors_preserving_counts(mapping, symbols):
    dots = [s for s,v in mapping.items() if v == '.']
    dashes = [s for s,v in mapping.items() if v == '-']
    spaces = [s for s,v in mapping.items() if v == '/']
    all_syms = list(symbols)

    ops = []
    if len(dots) and len(dashes):
        ops.append(('.', '-'))
    if len(dots) and len(spaces):
        ops.append(('.', '/'))
    if len(dashes) and len(spaces):
        ops.append(('-', '/'))

    while True:
        if not ops:
            return
            yield
        a_cls, b_cls = random.choice(ops)
        a_pool = [s for s in all_syms if mapping.get(s) == a_cls]
        b_pool = [s for s in all_syms if mapping.get(s) == b_cls]
        if not a_pool or not b_pool:
            continue
        a = random.choice(a_pool)
        b = random.choice(b_pool)
        if a == b:
            continue
        new_map = dict(mapping)
        new_map[a], new_map[b] = new_map[b], new_map[a]
        yield new_map

def refine_mappings(ciphertext, symbols, seeds, words, threads, rounds, neighbors_per_map, threshold, max_evals):
    base_targets = [
        ".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..",
        ".---", "-.-", ".-..", "--", "-.", "---", ".--.", "--.-", ".-.",
        "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."
    ]
    base_targets += [''.join(t) for t in itertools.permutations(['.', '-'], 3)]
    ngram_targets = [convert_morse_classes_to_optimizer(morse_pattern_for_phrase(w)) for w in words[:CONFIG.get('targets_top_ngram', 200)]]
    morse_targets = base_targets + [t for t in ngram_targets if t]

    global CIPHERTEXT
    CIPHERTEXT = ciphertext
    prepare_targets(morse_targets)

    leaderboard = list(seeds)
    leaderboard.sort(key=lambda x: x[1], reverse=True)
    best_mapping, best_score = (leaderboard[0] if leaderboard else ({}, -1.0))

    evals = 0
    for r in range(rounds):
        if evals >= max_evals:
            break
        proposals = []
        for (m, s) in leaderboard[:min(len(leaderboard), 10)]:
            gen = neighbors_preserving_counts(m, symbols)
            for _ in range(neighbors_per_map):
                proposals.append(next(gen, None))
        proposals = [p for p in proposals if p]
        if not proposals:
            break

        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = [ex.submit(evaluate_mapping, m) for m in proposals]
            for fut in as_completed(futs):
                m, s = fut.result()
                evals += 1
                if s > best_score:
                    best_mapping, best_score = m, s
                leaderboard.append((m, s))
                if threshold and best_score >= threshold:
                    break
            if threshold and best_score >= threshold:
                break

        # dedupe + sort
        uniq = {}
        for m, s in leaderboard:
            k = tuple(sorted(m.items()))
            if k not in uniq or s > uniq[k][1]:
                uniq[k] = (m, s)
        leaderboard = sorted(uniq.values(), key=lambda x: x[1], reverse=True)[:CONFIG.get("quick_global_top", 30)]

    return best_mapping, best_score, leaderboard
def main():
    cfg = CONFIG

    # Load ciphertext
    if cfg["ciphertext"] is not None and len(str(cfg["ciphertext"]).strip()) > 0:
        ciphertext = str(cfg["ciphertext"]).strip()
    else:
        with open(cfg["ciphertext_file"], 'r', encoding='utf-8', errors='ignore') as fh:
            ciphertext = fh.read().strip()

    # 1) Build n-grams
    words, word_freqs, diag = clean_and_build_ngrams(cfg["corpus"],
                                                     n_range=(cfg["nmin"], cfg["nmax"]),
                                                     top_k=cfg["topk"],
                                                     min_count=cfg["min_count"],
                                                     min_token_len=cfg["min_token_len"])
    print(f"[pipeline] ngrams: {len(words)}, diagnostics: {diag}")

    # 2) Build patterns and scan (threaded)
    patterns = decoder_build_patterns(words)
    print("[pipeline] scanning ciphertext (threaded)...")
    cand_sets = parallel_scan_threaded(ciphertext, words, patterns,
                                       chunk_size=cfg["chunk_size"],
                                       max_workers=cfg["max_workers"])

    # 3) Merge candidates and decode+score
    merged = merge_candidate_sets(cand_sets)
    results = decode_and_score(ciphertext, merged, words, word_freqs)
    print(f"[pipeline] decoder produced {len(results)} scored mappings")
    top = results[:cfg["take_top_cands"]]

    # Save CSV (best effort)
    try:
        export_results(results, words, export_dir='decoder_exports', export_k=200)
    except Exception as e:
        print(f"[warn] export failed: {e}")

    # 4) Build constraints from top decoder mappings
    constraints = constraints_from_candidates(top, top=cfg["take_top_cands"],
                                              require_unanimous=cfg["require_unanimous"])
    print(f"[pipeline] constraints (from decoder): {constraints}")

    # 5) Build morse targets for optimizer
    base_targets = [
        ".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..",
        ".---", "-.-", ".-..", "--", "-.", "---", ".--.", "--.-", ".-.",
        "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."
    ]
    base_targets += [''.join(t) for t in itertools.permutations(['.', '-'], 3)]
    ngram_targets = [convert_morse_classes_to_optimizer(morse_pattern_for_phrase(w)) for w in words[:CONFIG.get('targets_top_ngram', 200)]]
    morse_targets = base_targets + [t for t in ngram_targets if t]

    # 6) Chunked quick-pass + refinement
    symbols = sorted(set(ciphertext))
    quick_targets = build_targets_quick(words)
    quick_best = evaluate_quick_chunks(
        ciphertext, symbols, constraints, quick_targets,
        per_chunk_evals=cfg.get('quick_chunk_evals', 20000),
        top_per_chunk=cfg.get('quick_top_per_chunk', 5),
        threshold=cfg.get('quick_threshold', 0.982),
        threads=cfg.get('threads', 8)
    )
    print(f"[chunked] gathered {len(quick_best)} candidates from quick pass")
    best_map, best_score, _ = refine_mappings(
        ciphertext, symbols, quick_best, words,
        threads=cfg.get('threads', 8),
        rounds=cfg.get('refine_rounds', 3),
        neighbors_per_map=cfg.get('refine_neighbors_per_map', 2000),
        threshold=cfg.get('refine_threshold', 0.985),
        max_evals=cfg.get('refine_max_evals', 120000)
    )

    # 7) Decode final plaintext
    if best_map:
        plaintext, morse_seq = decode_with_mapping(ciphertext, best_map)
    else:
        best_map, best_score = {}, 0.0
        plaintext, morse_seq = "", ""

    print("\n=== PIPELINE RESULT ===")
    print(f"Device: {_TORCH_DEVICE if _TORCH else ('numpy' if _NUMPY else 'python')}  |  Threads: {CONFIG.get('threads')}")
    print(f"Best optimizer score: {best_score:.4f}")
    print(f"Best mapping: {best_map}")
    print(f"Decoded preview (first 500 chars):\n{plaintext[:500]}")

if __name__ == "__main__":
    main()
# ==== END Glue & Pipeline (no CLI) ====
