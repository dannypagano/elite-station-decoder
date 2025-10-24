#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pollux Cipher decoder — multithreaded + GPU-aware (CUDA/ROCm/MPS) with:
- length-weighted Morse n-gram scoring (longer n-grams count more)
- parallel random-start hill-climb search (CPU threads)
- consensus + crossover + repair + local refinement across iterations
- hotspot-centered preview (shows the region with the densest n-gram matches)
- priors to avoid slash-heavy (separator-heavy) mappings
- CSV export of top candidates (initial + refined), with hotspot-centered previews per row

Optimized defaults for:
- Debian VM on Windows, 16 GB RAM
- AMD Ryzen 7 5800X3D (8C/16T) -> threads=16
- AMD RX 6750 XT via PCIe passthrough (ROCm PyTorch) -> GPU scorer if available

If ROCm/CUDA/MPS isn't available, falls back to NumPy; else pure Python.
"""

from __future__ import annotations
import os, re, math, string, random, csv
import concurrent.futures as cf
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

# =======================
# Environment & Backends
# =======================
# Max out BLAS threads for NumPy (tune if you see oversubscription)
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

_TORCH = False
_NUMPY = False
_TORCH_DEVICE = "cpu"
_TORCH_BACKEND = "none"   # "cuda", "rocm", "mps", or "none"

try:
    import torch
    _TORCH = True
    # CUDA (NVIDIA) usually reports cuda; ROCm (AMD) often appears under cuda device with torch.version.hip present
    if torch.cuda.is_available():
        _TORCH_DEVICE = "cuda"
        _TORCH_BACKEND = "cuda" if getattr(torch.version, "hip", None) is None else "rocm"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _TORCH_DEVICE = "mps"
        _TORCH_BACKEND = "mps"
except Exception:
    torch = None
    _TORCH = False

try:
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view as _sliding
    _NUMPY = True
except Exception:
    np = None
    _sliding = None
    _NUMPY = False

# =======================
# Config (balanced ~1h target, 1200-char preview)
# =======================
CONFIG = {
    # ---------- Inputs ----------
    "ciphertext": None,                 # Paste ciphertext (string) here; if None, read from file
    "ciphertext_file": "./cipher.txt",  # Used when ciphertext is None
    "corpus_dir": "./corpus",           # Directory containing *.txt files

    # ---------- Parallel search ----------
    "threads": 16,                      # ThreadPool size (5800X3D -> 16 logical threads)
    "random_starts": 56,                # Starts per class-split (breadth)
    "hill_steps": 3000,                 # Swap attempts per start (depth)
    "seed": 42,                         # RNG seed (or None)

    # ---------- Class-count splits (dots, dashes) ----------
    # The remaining symbols become '/'. Keep richer splits to avoid slash soup.
    "class_splits": [(6,6), (7,7), (5,6), (6,5), (4,4)],

    # ---------- Language model weights ----------
    "lm_word_weight": 1.0,              # Word-frequency log likelihood
    "lm_char_weight": 1.0,              # Char-bigram log likelihood
    "word_min_len": 2,                  # Ignore short words (except 'a', 'i')

    # ---------- Morse n-gram targets ----------
    "targets_top_words": 2500,          # Top-N corpus words to convert to Morse targets
    "targets_min_len": 1,               # Minimum Morse length of target
    "targets_max_len": 8,               # Max Morse length (keeps sliding windows tractable)
    "ngram_weight_alpha": 1.4,          # Weight(L) = L ** alpha (favor longer n-grams)
    "ngram_weight_norm": True,          # Normalize weights across lengths

    # ---------- Early stop ----------
    "early_stop_score": None,           # Use None to disable absolute threshold; rely on search depth

    # ---------- Priors to avoid separator soup ----------
    "sep_target_ratio": 0.30,           # target fraction of '/' in morse (~0.25–0.35 reasonable)
    "sep_penalty_strength": 150.0,      # penalty weight for exceeding sep target
    "dot_ratio_target": 0.60,           # target fraction of dots among signals (dots+dashes)
    "dot_ratio_strength": 50.0,         # penalty weight for dot:dash deviation
    "sep_run_penalty_strength": 3.0,    # per extra '/' in a run (discourage //// sequences)

    # ---------- Consensus refinement ----------
    "topK_global": 40,                  # Best K from parallel phase
    "consensus_rounds": 3,              # Iterative refinement passes
    "consensus_top_from_each_round": 40,# Keep this many after each round
    "parent_pool": 16,                  # Parents considered for crossover
    "children_per_pair": 2,             # Children spawned per parent pair
    "neighbors_per_map": 900,           # Local neighbor proposals per mapping per round
    "repair_max_swaps": 150,            # Max swaps during class-count repair
    "consensus_symbol_min_vote": 0.0,   # Minimal vote to accept a symbol class

    # ---------- Preview (hotspot-centered) ----------
    "preview_chars": 1200,              # Length of final plaintext preview (expanded)
    "preview_window_morse": 2000,       # Morse window around hotspot to convert for preview (wider)

    # ---------- Logging ----------
    "log_every": 250,                   # Print progress from starts at most this often (currently quiet)

    # ---------- CSV export ----------
    "export_csv": True,                 # turn CSV export on/off
    "export_top_k": 50,                 # how many candidates to write
    "export_path_initial": "candidates_initial.csv",
    "export_path_refined": "candidates_refined.csv",
    "export_preview_chars": 240,        # per-row preview chars (hotspot-centered)
    "export_preview_window_morse": 800, # narrower window for CSV previews
}

CONFIG.update({
    # Richer class splits (exact counts): ~14/14 signals leaves ~8 seps (≈22%)
    "class_splits": [(14,14), (13,13), (15,14), (14,15), (12,12), (15,15)],

    # Strengthen Morse n-gram signal
    "targets_top_words": 3000,
    "targets_max_len": 10,
    "ngram_weight_alpha": 1.5,

    # Make separator ratio penalty symmetric (we’ll enable it below)
    "sep_target_ratio": 0.24,            # aim ~24% '/' overall
    "sep_penalty_strength": 380.0,       # stronger penalty
    # Dot:dash realism (dots slightly more common)
    "dot_ratio_target": 0.60,
    "dot_ratio_strength": 80.0,
    # Run-length penalty for '/////'
    "sep_run_penalty_strength": 5.0,

    # NEW: reward valid Morse letters (pulls toward real decodes)
    "valid_letter_reward": 600.0,        # scale ~200–600
})


ALNUM = set(string.ascii_lowercase + string.digits)

# =======================
# Morse tables
# =======================
MORSE_TABLE = {
    'a': '.-',   'b': '-...', 'c': '-.-.', 'd': '-..',  'e': '.',
    'f': '..-.', 'g': '--.',  'h': '....', 'i': '..',   'j': '.---',
    'k': '-.-',  'l': '.-..', 'm': '--',   'n': '-.',   'o': '---',
    'p': '.--.', 'q': '--.-', 'r': '.-.',  's': '...',  't': '-',
    'u': '..-',  'v': '...-', 'w': '.--',  'x': '-..-', 'y': '-.--',
    'z': '--..',
    '0': '-----','1': '.----','2': '..---','3': '...--','4': '....-',
    '5': '.....','6': '-....','7': '--...','8': '---..','9': '----.',
}
MORSE_TO_CHAR = {v: k for k, v in {k.upper(): v for k, v in MORSE_TABLE.items()}.items()}

# =======================
# Corpus model
# =======================
def clean_text(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r'[^a-z0-9\s]+', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

def load_corpus_texts(corpus_dir: str) -> List[str]:
    out = []
    for root, _, files in os.walk(corpus_dir):
        for f in files:
            if f.lower().endswith(".txt"):
                try:
                    with open(os.path.join(root, f), "r", encoding="utf-8", errors="ignore") as fh:
                        out.append(fh.read())
                except Exception:
                    pass
    return out

def build_word_freq_and_char_bigrams(corpus_dir: str) -> Tuple[Counter, Counter]:
    texts = load_corpus_texts(corpus_dir)
    cleaned = clean_text(" ".join(texts))
    words = cleaned.split()
    word_freq = Counter(words)
    chars = " " + cleaned + " "
    bigrams = Counter(zip(chars, chars[1:]))
    return word_freq, bigrams

def log_prob_words(text: str, word_freq: Counter, word_min_len=2) -> float:
    words = clean_text(text).split()
    total = sum(word_freq.values()) + len(word_freq)
    lp = 0.0
    for w in words:
        if w in ("a", "i") or len(w) >= word_min_len:
            lp += math.log((word_freq[w] + 1) / total)
    return lp

def log_prob_char_bigrams(text: str, bigr: Counter) -> float:
    t = " " + clean_text(text) + " "
    total = sum(bigr.values())
    V = max(1, len({a for a, _ in bigr.keys()} | {b for _, b in bigr.keys()}))
    denom = total + V * V
    lp = 0.0
    for ab in zip(t, t[1:]):
        lp += math.log((bigr.get(ab, 0) + 1) / denom)
    return lp

# =======================
# Morse targets (weighted)
# =======================
def word_to_morse(word: str) -> str:
    return '/'.join(MORSE_TABLE.get(ch, '') for ch in word if ch in MORSE_TABLE)

def build_morse_targets(word_freq: Counter, top_n=2500, min_len=1, max_len=8) -> List[str]:
    words = [w for w, _ in word_freq.most_common(top_n)]
    seqs = []
    for w in words:
        m = word_to_morse(w)
        if not m:
            continue
        L = len(m)
        if min_len <= L <= max_len:
            seqs.append(m)
    letters = list({v for v in MORSE_TABLE.values()})
    seqs.extend(letters)
    return seqs

class MorseTargets:
    """Pre-encoded Morse targets grouped by length, with length weights."""
    def __init__(self, targets: List[str], alpha: float = 1.4, normalize: bool = True):
        self.alpha = alpha
        self.normalize = normalize
        self.by_len = defaultdict(list)
        for t in targets:
            self.by_len[len(t)].append(t)

        self.enc = {'.': 1, '-': 2, '/': 0}
        self.weights = {}
        self.torch_by_len = {}
        self.numpy_by_len = {}

        for L, seqs in self.by_len.items():
            self.weights[L] = float(L ** self.alpha)
            if _TORCH:
                T = torch.tensor([[self.enc.get(ch, 0) for ch in s] for s in seqs],
                                 dtype=torch.int16, device=_TORCH_DEVICE)
                self.torch_by_len[L] = T
            if _NUMPY:
                A = np.asarray([[self.enc.get(ch, 0) for ch in s] for s in seqs], dtype=np.int16)
                self.numpy_by_len[L] = A

        if self.normalize and self.weights:
            total = sum(self.weights.values())
            for L in self.weights:
                self.weights[L] /= total

# =======================
# Morse parsing & hotspot preview
# =======================
def morse_to_plaintext(morse: str) -> str:
    if not morse:
        return ""
    # Normalize separators: 3+ '/' => word break; 1–2 '/' => letter break
    morse = re.sub(r'/+', lambda m: ' /// ' if len(m.group(0)) >= 3 else ' / ', morse)
    out_words = []
    for tk in morse.strip().split(' /// '):
        letters = []
        for chunk in tk.strip().split(' / '):
            if not chunk:
                continue
            letters.append(MORSE_TO_CHAR.get(chunk.replace(' ', ''), '?'))
        out_words.append("".join(letters))
    return " ".join(out_words)

def apply_mapping(cipher: str, mapping: Dict[str, str]) -> str:
    # unknowns -> '/' (separator), conservative to avoid false dots/dashes
    return "".join(mapping.get(ch, '/') for ch in cipher)

def morse_string_to_ints(m: str) -> "np.ndarray":
    enc = {'.':1,'-':2,'/':0}
    return np.fromiter((enc.get(ch, 0) for ch in m), dtype=np.int16, count=len(m))

def hotspot_index_from_targets(morse: str, tgt: MorseTargets) -> int:
    """
    Build a per-position heat (weighted best-match across lengths) and return
    the index with maximum heat. Uses GPU/NumPy if available.
    """
    if not _NUMPY:
        return max(0, len(morse)//2)

    m_int = morse_string_to_ints(morse)
    heat = np.zeros(max(1, len(m_int)), dtype=np.float32)

    if _TORCH and tgt.torch_by_len:
        g = torch.from_numpy(m_int).to(_TORCH_DEVICE)
        for L, T in tgt.torch_by_len.items():
            if L > g.numel():
                continue
            W = g.unfold(0, L, 1)                        # [S, L]
            eq = (W.unsqueeze(1) == T.unsqueeze(0))      # [S, K, L]
            frac = eq.sum(dim=-1).float() / float(L)     # [S, K]
            best = frac.max(dim=1).values                # [S]
            add = (best * tgt.weights[L]) / float(L)
            add_cpu = add.detach().cpu().numpy()
            inc = np.zeros(len(heat)+1, dtype=np.float32)
            inc[:len(add_cpu)] += add_cpu
            inc[L:] -= add_cpu
            heat += np.cumsum(inc[:-1])
    else:
        for L, A in tgt.numpy_by_len.items():
            if L > m_int.shape[0]:
                continue
            W = _sliding(m_int, window_shape=L)          # [S, L]
            eq = (W[:, None, :] == A[None, :, :])        # [S, K, L]
            frac = eq.sum(axis=2) / float(L)             # [S, K]
            best = frac.max(axis=1)                      # [S]
            add = (best * tgt.weights[L]) / float(L)     # [S]
            inc = np.zeros(len(heat)+1, dtype=np.float32)
            inc[:len(add)] += add.astype(np.float32)
            inc[L:] -= add.astype(np.float32)
            heat += np.cumsum(inc[:-1])

    return int(np.argmax(heat))

def preview_centered_on_hotspot(cipher: str, mapping: Dict[str,str], tgt: MorseTargets,
                                window_morse: int, preview_chars: int) -> Tuple[str, str, int]:
    """
    Make a preview centered on the hotspot (max weighted n-gram density).
    Returns (plaintext_preview, morse_slice, hotspot_index).
    """
    morse = apply_mapping(cipher, mapping)
    if len(morse) == 0:
        return "", "", 0
    hotspot = hotspot_index_from_targets(morse, tgt)

    half = max(10, window_morse // 2)
    lo = max(0, hotspot - half)
    hi = min(len(morse), hotspot + half)
    morse_slice = morse[lo:hi]
    plain_slice = morse_to_plaintext(morse_slice)
    return plain_slice[:preview_chars], morse_slice, hotspot

# =======================
# Scoring (with anti-slash priors)
# =======================
def score_morse_gpu(morse_int: "np.ndarray", tgt: MorseTargets) -> float:
    g = torch.from_numpy(morse_int).to(_TORCH_DEVICE)
    accum = torch.tensor(0.0, device=_TORCH_DEVICE)
    for L, T in tgt.torch_by_len.items():
        if L > g.numel():
            continue
        W = g.unfold(0, L, 1)                   # [S, L]
        eq = (W.unsqueeze(1) == T.unsqueeze(0)) # [S, K, L]
        frac = eq.sum(dim=-1).float() / float(L)# [S, K]
        local = frac.max() * tgt.weights[L]     # scalar
        accum += local
    return float(accum.item())

def score_morse_numpy(morse_int: "np.ndarray", tgt: MorseTargets) -> float:
    accum = 0.0
    for L, A in tgt.numpy_by_len.items():
        if L > morse_int.shape[0]:
            continue
        W = _sliding(morse_int, window_shape=L)      # [S, L]
        eq = (W[:, None, :] == A[None, :, :])        # [S, K, L]
        frac = eq.sum(axis=2) / float(L)             # [S, K]
        local = float(frac.max()) * tgt.weights[L]
        accum += local
    return accum

def score_mapping(cipher: str,
                  mapping: Dict[str, str],
                  word_freq: Counter, bigr: Counter,
                  lm_word_w: float, lm_char_w: float,
                  word_min_len: int,
                  tgt: MorseTargets) -> Tuple[float, str, str]:
    morse = apply_mapping(cipher, mapping)
    plain = morse_to_plaintext(morse)
    score = 0.0

    # Language model components
    if lm_word_w:
        score += lm_word_w * log_prob_words(plain, word_freq, word_min_len)
    if lm_char_w:
        score += lm_char_w * log_prob_char_bigrams(plain, bigr)

    # Weighted Morse n-gram component (GPU/NumPy preferred)
    if _NUMPY:
        m_int = morse_string_to_ints(morse)
        if _TORCH and tgt.torch_by_len:
            score += score_morse_gpu(m_int, tgt)
        else:
            score += score_morse_numpy(m_int, tgt)
    else:
        # slow fallback (pure Python)
        enc = {'.':1,'-':2,'/':0}
        m_int = [enc.get(ch, 0) for ch in morse]
        accum = 0.0
        for L, seqs in tgt.by_len.items():
            if L > len(m_int):
                continue
            best_local = 0.0
            for s in seqs:
                t = [enc.get(ch, 0) for ch in s]
                for i in range(0, len(m_int)-L+1):
                    frac = sum(1 for a, b in zip(m_int[i:i+L], t) if a == b) / float(L)
                    if frac > best_local:
                        best_local = frac
            accum += best_local * tgt.weights[L]
        score += accum

    # -------- Priors / constraints --------
    if morse:
        sep = morse.count('/')
        dot = morse.count('.')
        dash = morse.count('-')
        total = max(1, sep + dot + dash)

        # Symmetric separator ratio penalty (too many or too few '/')
        sep_ratio = sep / total
        target_sep = CONFIG.get("sep_target_ratio", 0.24)
        sep_strength = CONFIG.get("sep_penalty_strength", 380.0)  # matches your CONFIG
        score -= sep_strength * (sep_ratio - target_sep) ** 2

        # Dot:Dash ratio penalty among signals
        sig = dot + dash
        if sig > 0:
            dot_ratio = dot / sig
            dot_target = CONFIG.get("dot_ratio_target", 0.60)
            dot_strength = CONFIG.get("dot_ratio_strength", 80.0)
            score -= dot_strength * (dot_ratio - dot_target) ** 2

        # Run-length penalty for long separator streaks
        run_strength = CONFIG.get("sep_run_penalty_strength", 5.0)
        if run_strength > 0.0:
            extra = 0
            curr = 0
            for ch in morse:
                if ch == '/':
                    curr += 1
                    if curr > 1:
                        extra += 1
                else:
                    curr = 0
            score -= run_strength * extra

        # Reward for valid Morse letters (non-empty chunks that decode cleanly)
        valid_reward = CONFIG.get("valid_letter_reward", 600.0)
        if valid_reward > 0.0:
            chunks = [c for c in morse.split('/') if c]  # non-empty runs of . and -
            if chunks:
                valid = sum(1 for c in chunks if c in MORSE_TO_CHAR)
                frac = valid / len(chunks)
                score += valid_reward * (frac ** 2)

    return score, plain, morse

# =======================
# Search primitives (random starts + hill climb + parallel executor)
# =======================
def random_mapping(symbols: List[str], n_dot: int, n_dash: int) -> Dict[str, str]:
    """Generate a random mapping of symbols to dot/dash/sep classes."""
    s = symbols[:]
    random.shuffle(s)
    return {c: ('.' if i < n_dot else '-' if i < n_dot + n_dash else '/') for i, c in enumerate(s)}

def neighbors_swaps(mapping: Dict[str, str], symbols: List[str], k: int = 1) -> List[Dict[str, str]]:
    """Generate k neighboring mappings by swapping assignments of random symbols."""
    out = []
    for _ in range(k):
        a, b = random.sample(symbols, 2)
        if mapping[a] == mapping[b]:
            continue
        nm = dict(mapping)
        nm[a], nm[b] = nm[b], nm[a]
        out.append(nm)
    return out

def run_start(cipher: str, symbols: List[str], split: Tuple[int, int],
              word_freq: Counter, bigr: Counter, cfg: dict, tgt: MorseTargets):
    """Perform a single hill-climb search from a random mapping for a given (dots, dashes) split."""
    n_dot, n_dash = split
    mapping = random_mapping(symbols, n_dot, n_dash)
    best_score, best_plain, best_morse = score_mapping(
        cipher, mapping, word_freq, bigr,
        cfg["lm_word_weight"], cfg["lm_char_weight"], cfg["word_min_len"], tgt
    )
    best_map = mapping
    for step in range(cfg["hill_steps"]):
        for cand in neighbors_swaps(best_map, symbols, k=1):
            s, p, m = score_mapping(
                cipher, cand, word_freq, bigr,
                cfg["lm_word_weight"], cfg["lm_char_weight"], cfg["word_min_len"], tgt
            )
            if s > best_score:
                best_map, best_score, best_plain, best_morse = cand, s, p, m
                if cfg["early_stop_score"] is not None and s >= cfg["early_stop_score"]:
                    return (best_score, best_map, best_plain, best_morse)
    return (best_score, best_map, best_plain, best_morse)

def pollux_search_all(cipher: str, symbols: List[str],
                      word_freq: Counter, bigr: Counter,
                      cfg: dict, tgt: MorseTargets):
    """Run multiple random-start hill climbs in parallel across class splits."""
    results = []
    with cf.ThreadPoolExecutor(max_workers=cfg["threads"]) as ex:
        futs = []
        for split in cfg["class_splits"]:
            if sum(split) > len(symbols):
                continue
            for _ in range(cfg["random_starts"]):
                futs.append(ex.submit(run_start, cipher, symbols, split, word_freq, bigr, cfg, tgt))
        for fut in cf.as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                print("[warn] worker failed:", e)
    results.sort(key=lambda x: x[0], reverse=True)
    return results  # list of (score, mapping, plaintext, morse)

# =======================
# Consensus + crossover + repair (fixed unpacking)
# =======================
def mapping_class_counts(mapping: Dict[str, str]) -> Tuple[int,int,int]:
    c = Counter(mapping.values())
    return c.get('.', 0), c.get('-', 0), c.get('/', 0)

def consensus_mapping(symbols: List[str], candidates: List[Tuple[float, Dict[str,str]]],
                      target_counts: Tuple[int,int,int], min_vote=0.0) -> Dict[str,str]:
    wsum = sum(max(0.0, s) for s, _ in candidates) + 1e-9
    votes = {sym: {'.':0.0, '-':0.0, '/':0.0} for sym in symbols}
    for score, m in candidates:
        w = max(0.0, score) / wsum
        for sym in symbols:
            votes[sym][m.get(sym, '/')] += w

    choice = {}
    for sym in symbols:
        cls, v = max(votes[sym].items(), key=lambda kv: kv[1])
        choice[sym] = cls if v >= min_vote else '/'

    # greedy repair to exact class counts
    need_dot, need_dash, need_sep = target_counts
    have_dot, have_dash, have_sep = mapping_class_counts(choice)

    def adjust(src_cls, dst_cls, needed):
        nonlocal choice
        if needed <= 0:
            return
        pool = [sym for sym, c in choice.items() if c == src_cls]
        pool.sort(key=lambda s: (votes[s][src_cls] - votes[s][dst_cls]))  # smallest loss first
        i = 0
        while needed > 0 and i < len(pool):
            s = pool[i]; i += 1
            choice[s] = dst_cls
            needed -= 1

    if have_dot < need_dot:
        adjust('/', '.', need_dot - have_dot)
    if have_dash < need_dash:
        adjust('/', '-', need_dash - have_dash)
    have_dot, have_dash, have_sep = mapping_class_counts(choice)
    if have_dot > need_dot:
        adjust('.', '/', have_dot - need_dot)
    if have_dash > need_dash:
        adjust('-', '/', have_dash - need_dash)

    return choice

def crossover_child(parentA: Dict[str,str], parentB: Dict[str,str],
                    symbols: List[str], votes=None) -> Dict[str,str]:
    child = {}
    for s in symbols:
        a = parentA.get(s, '/'); b = parentB.get(s, '/')
        if a == b:
            child[s] = a
        else:
            if votes:
                va = votes[s].get(a, 0.0); vb = votes[s].get(b, 0.0)
                child[s] = a if va >= vb else b
            else:
                child[s] = a if random.random() < 0.5 else b
    return child

def repair_class_counts(mapping: Dict[str,str], target_counts: Tuple[int,int,int],
                        symbols: List[str], max_swaps=150) -> Dict[str,str]:
    """
    Repair to exact dot/dash/sep counts by reassigning symbols.
    Preference order: fill from '/' pool first to dots/dashes, then trim excess.
    """
    m = dict(mapping)
    need_dot, need_dash, need_sep = target_counts
    have_dot, have_dash, have_sep = mapping_class_counts(m)

    def pool(cls): return [s for s, c in m.items() if c == cls]

    swaps = 0
    # Fill deficits from '/' pool first
    if have_dot < need_dot and swaps < max_swaps:
        from_sep = pool('/')
        take = min(len(from_sep), need_dot - have_dot, max_swaps - swaps)
        for s in from_sep[:take]: m[s] = '.'; swaps += 1
    if have_dash < need_dash and swaps < max_swaps:
        from_sep = pool('/')
        take = min(len(from_sep), need_dash - have_dash, max_swaps - swaps)
        for s in from_sep[:take]: m[s] = '-'; swaps += 1

    # Recompute and trim excess by pushing to '/'
    have_dot, have_dash, have_sep = mapping_class_counts(m)
    if have_dot > need_dot and swaps < max_swaps:
        from_dot = pool('.')
        take = min(have_dot - need_dot, max_swaps - swaps)
        for s in from_dot[:take]: m[s] = '/'; swaps += 1
    if have_dash > need_dash and swaps < max_swaps:
        from_dash = pool('-')
        take = min(have_dash - need_dash, max_swaps - swaps)
        for s in from_dash[:take]: m[s] = '/'; swaps += 1

    # Final assert: exact counts if possible
    # (If max_swaps too small, we might still be off by 1; bump max_swaps if you ever see that.)
    return m


def refine_consensus(cipher: str, symbols: List[str], splits: List[Tuple[int,int]],
                     word_freq: Counter, bigr: Counter, cfg: dict, tgt: MorseTargets,
                     initial_results: List[Tuple[float, Dict[str,str], str, str]]):
    """
    Iterative refinement:
      - take topK mappings
      - build consensus for nearby class-count splits
      - spawn crossover children + repair
      - hill-climb neighbors
      - re-rank, dedupe, and repeat
    """
    pool: List[Tuple[float, Dict[str,str], str, str]] = initial_results[:cfg["topK_global"]]
    symbols_sorted = sorted(symbols)

    def dedupe_keep_best(items: List[Tuple[float, Dict[str,str], str, str]],
                         cap: int) -> List[Tuple[float, Dict[str,str], str, str]]:
        seen = {}
        for s, m, p, morse in items:
            key = tuple(sorted(m.items()))
            if key not in seen or s > seen[key][0]:
                seen[key] = (s, m, p, morse)
        return sorted(seen.values(), key=lambda x: x[0], reverse=True)[:cap]

    best = max(pool, key=lambda x: x[0])

    for _round in range(cfg["consensus_rounds"]):
        new_candidates: List[Tuple[float, Dict[str,str], str, str]] = []

        for (nd, nh) in splits:
            # Pick parents whose dot/dash counts roughly match this split
            filtered_pairs: List[Tuple[float, Dict[str,str]]] = []
            for s, m, p, morse in pool:
                d, h, q = mapping_class_counts(m)
                if abs(d - nd) <= 2 and abs(h - nh) <= 2:
                    filtered_pairs.append((s, m))
            if not filtered_pairs:
                continue

            target_counts = (nd, nh, max(0, len(symbols_sorted) - nd - nh))

            # ----- Consensus (weighted vote) -----
            consensus_map = consensus_mapping(symbols_sorted, filtered_pairs, target_counts,
                                              min_vote=cfg["consensus_symbol_min_vote"])
            consensus_map = repair_class_counts(consensus_map, target_counts, symbols_sorted,
                                                cfg["repair_max_swaps"])
            s0, p0, morse0 = score_mapping(cipher, consensus_map, word_freq, bigr,
                                           cfg["lm_word_weight"], cfg["lm_char_weight"],
                                           cfg["word_min_len"], tgt)
            new_candidates.append((s0, consensus_map, p0, morse0))

            # ----- Votes to guide crossover -----
            wsum = sum(max(0.0, sc) for sc, _m in filtered_pairs) + 1e-9
            votes = {sym: {'.': 0.0, '-': 0.0, '/': 0.0} for sym in symbols_sorted}
            for sc, mp in filtered_pairs:
                w = max(0.0, sc) / wsum
                for sym in symbols_sorted:
                    votes[sym][mp.get(sym, '/')] += w

            # ----- Parent pool (mappings only) and pairs -----
            parent_maps: List[Dict[str,str]] = [m for (_sc, m) in filtered_pairs][:cfg["parent_pool"]]
            n_par = len(parent_maps)
            if n_par >= 2:
                pairs = [(parent_maps[i], parent_maps[j]) for i in range(n_par) for j in range(i+1, n_par)]
                random.shuffle(pairs)
                max_pairs = max(1, min(len(pairs), cfg["parent_pool"] // 2))
                pairs = pairs[:max_pairs]
            else:
                pairs = []

            # ----- Children via crossover + repair -----
            children: List[Dict[str,str]] = []
            for pa, pb in pairs:
                for _ in range(cfg["children_per_pair"]):
                    child = crossover_child(pa, pb, symbols_sorted, votes=votes)
                    child = repair_class_counts(child, target_counts, symbols_sorted, cfg["repair_max_swaps"])
                    children.append(child)

            # ----- Local neighbor refinement (mapping stays attached) -----
            seeds = [consensus_map] + children
            with cf.ThreadPoolExecutor(max_workers=cfg["threads"]) as ex:
                futs: List[Tuple[Dict[str,str], "cf.Future"]] = []
                for m0 in seeds:
                    futs.append((m0, ex.submit(
                        score_mapping, cipher, m0, word_freq, bigr,
                        cfg["lm_word_weight"], cfg["lm_char_weight"], cfg["word_min_len"], tgt
                    )))
                    for nm in neighbors_swaps(m0, symbols_sorted, k=cfg["neighbors_per_map"]):
                        futs.append((nm, ex.submit(
                            score_mapping, cipher, nm, word_freq, bigr,
                            cfg["lm_word_weight"], cfg["lm_char_weight"], cfg["word_min_len"], tgt
                        )))
                for m_candidate, fut in futs:
                    try:
                        s, p, morse = fut.result()
                        new_candidates.append((s, m_candidate, p, morse))
                    except Exception:
                        pass  # ignore failed evaluations

        # Merge old & new, dedupe by mapping, keep top N
        pool = dedupe_keep_best(pool + new_candidates, cfg["consensus_top_from_each_round"])
        if pool and pool[0][0] > best[0]:
            best = pool[0]

    return best, pool  # (score, mapping, plaintext, morse), final pool list

# =======================
# CSV export helpers (hotspot-centered previews)
# =======================
def _mapping_counts(mapping: Dict[str,str]) -> Tuple[int,int,int]:
    c = Counter(mapping.values())
    return c.get('.',0), c.get('-',0), c.get('/',0)

def export_candidates_csv(path: str,
                          results: List[Tuple[float, Dict[str,str], str, str]],
                          cipher: str,
                          tgt: MorseTargets,
                          max_rows: int = 50,
                          preview_chars: int = 240,
                          preview_window_morse: int = 800) -> None:
    rows = []
    for score, mapping, _plaintext_full, _morse_full in results[:max_rows]:
        d,h,s = _mapping_counts(mapping)
        # hotspot-centered short preview for CSV
        prev_plain, _, _ = preview_centered_on_hotspot(
            cipher, mapping, tgt,
            window_morse=preview_window_morse,
            preview_chars=preview_chars
        )
        rows.append({
            "score": f"{score:.6f}",
            "dots": d, "dashes": h, "seps": s,
            "mapping": " ".join(f"{k}:{v}" for k,v in sorted(mapping.items())),
            "preview": (prev_plain or "").replace("\n"," "),
        })
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["score","dots","dashes","seps","mapping","preview"])
        w.writeheader(); w.writerows(rows)

# =======================
# Main
# =======================
def main():
    cfg = CONFIG
    if cfg["ciphertext"] is not None:
        ciphertext = cfg["ciphertext"].strip().lower()
    else:
        with open(cfg["ciphertext_file"], "r", encoding="utf-8", errors="ignore") as fh:
            ciphertext = fh.read().strip().lower()

    # Keep only [a-z0-9]
    ciphertext = "".join(ch for ch in ciphertext if ch in ALNUM)
    symbols = sorted(set(ciphertext))

    print(f"[INFO] Torch={_TORCH} backend={_TORCH_BACKEND} device={_TORCH_DEVICE} | NumPy={_NUMPY} | Threads={cfg['threads']}")

    # Build corpus models & Morse targets
    print("[INFO] Building corpus models …")
    word_freq, bigr = build_word_freq_and_char_bigrams(cfg["corpus_dir"])
    print(f"  unique words={len(word_freq):,}  |  char bigrams={len(bigr):,}")

    print("[INFO] Building Morse targets …")
    targets = build_morse_targets(word_freq,
                                  top_n=cfg["targets_top_words"],
                                  min_len=cfg["targets_min_len"],
                                  max_len=cfg["targets_max_len"])
    tgt = MorseTargets(targets, alpha=cfg["ngram_weight_alpha"], normalize=cfg["ngram_weight_norm"])
    if targets:
        lens = sorted(len(x) for x in targets)
        print(f"  target lengths: {lens[:5]} … {lens[-5:]} (total={len(targets)})")
    else:
        print("  [warn] no targets built; scoring will rely on LM components and priors only")

    # Parallel search (gather many results)
    print("[INFO] Searching (parallel random starts) …")
    all_results = pollux_search_all(ciphertext, symbols, word_freq, bigr, cfg, tgt)
    if not all_results:
        print("[ERR] No results produced.")
        return
    best_initial = all_results[0]
    print(f"[INFO] Best initial score: {best_initial[0]:.4f}")

    # Consensus + refinement iterations
    print("[INFO] Refining with consensus + crossover + repair …")
    best_tuple, final_pool = refine_consensus(
        ciphertext, symbols, cfg["class_splits"], word_freq, bigr, cfg, tgt, all_results
    )
    best_score, best_map, best_plain, best_morse = best_tuple

    # Hotspot-centered preview (final)
    preview_plain, preview_morse, hotspot = preview_centered_on_hotspot(
        ciphertext, best_map, tgt,
        window_morse=cfg["preview_window_morse"],
        preview_chars=cfg["preview_chars"]
    )

    # Export CSVs (optional) — with hotspot-centered per-row previews
    if cfg.get("export_csv", False):
        try:
            export_candidates_csv(
                cfg.get("export_path_initial", "candidates_initial.csv"),
                all_results,
                cipher=ciphertext,
                tgt=tgt,
                max_rows=cfg.get("export_top_k", 50),
                preview_chars=cfg.get("export_preview_chars", 240),
                preview_window_morse=cfg.get("export_preview_window_morse", 800),
            )
        except Exception as e:
            print("[warn] could not export initial CSV:", e)
        try:
            export_candidates_csv(
                cfg.get("export_path_refined", "candidates_refined.csv"),
                final_pool,
                cipher=ciphertext,
                tgt=tgt,
                max_rows=cfg.get("export_top_k", 50),
                preview_chars=cfg.get("export_preview_chars", 240),
                preview_window_morse=cfg.get("export_preview_window_morse", 800),
            )
        except Exception as e:
            print("[warn] could not export refined CSV:", e)

    print("\n=== RESULT ===")
    print(f"Device: {('torch:'+_TORCH_BACKEND+':'+_TORCH_DEVICE) if _TORCH else ('numpy' if _NUMPY else 'python')}")
    print(f"Score:  {best_score:.4f}")
    print(f"Mapping (symbol→class): {best_map}")
    print(f"[Hotspot] morse index: {hotspot}  |  morse slice (first 200): {preview_morse[:200]}")
    print(f"Plaintext preview (centered on hotspot, up to {cfg['preview_chars']} chars):\n{preview_plain}")

if __name__ == "__main__":
    if CONFIG["seed"] is not None:
        random.seed(CONFIG["seed"])
    main()
