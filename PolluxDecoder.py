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
    while changed:
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

if __name__ == "__main__":
    main()
