"""Phase 7b — classical pattern-mining family (PLAN.md §11, Block 57).

The one ML family that was entirely missing: frequent-itemset + sequential-pattern
mining (Apriori, ECLAT, FP-Growth, PrefixSpan, GSP, SPADE) — "repeating trade
setups / frequent market behaviours / event-sequence extraction".

The bridge to the generic (T, D) interface (Rule 23): pattern mining works on
*itemsets/sequences*, not numeric arrays — so each node first **symbolizes** the
series (per-step change → one of k quantile symbols), turns sliding windows into
**transactions** (sets, for itemset miners) and **sequences** (ordered tuples, for
sequential miners), then mines. Each emits the `pattern` belief type (registered in
the v0.1 interlingua) with the top patterns + supports + a next-symbol guess.

Light-stack (pure Python/numpy). Genuine, distinct algorithms — not wrappers.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

from ..belief import Belief
from ..node import Node
from ..registry import register

ALPHABET_K = 5
WINDOW = 4


def _symbolize(x: np.ndarray, k: int = ALPHABET_K):
    d = np.diff(x)
    if len(d) < k:
        return [], [f"s{i}" for i in range(k)]
    edges = np.quantile(d, np.linspace(0, 1, k + 1)[1:-1])
    syms = np.digitize(d, edges)
    return [f"s{int(s)}" for s in syms], [f"s{i}" for i in range(k)]


def _transactions(symbols, w=WINDOW):
    return [set(symbols[i:i + w]) for i in range(len(symbols) - w + 1)]


def _sequences(symbols, w=WINDOW):
    return [tuple(symbols[i:i + w]) for i in range(len(symbols) - w + 1)]


def _min_sup(n: int) -> int:
    return max(2, int(0.15 * n))


def _top(freq: Dict, k: int = 12):
    items = sorted(freq.items(), key=lambda kv: -kv[1])[:k]
    return [{"items": list(p) if not isinstance(p, tuple) else list(p), "support": int(s)}
            for p, s in items]


def _emit(self, x, freq, alphabet, predicted_next, algo, multi=False):
    pats = _top(freq)
    conf = min(0.9, len(pats) / 15.0)
    return Belief("pattern", {"patterns": pats, "n_patterns": len(freq),
                              "support": _min_sup(max(1, len(x))), "alphabet": alphabet,
                              "predicted_next": predicted_next, "algorithm": algo,
                              "series": np.asarray(x).tolist()}, conf, self.name)


# ----------------------------------------------------- frequent itemsets
def _apriori(transactions, min_sup) -> Dict[frozenset, int]:
    def sup(s):
        return sum(1 for t in transactions if s <= t)
    items = sorted({i for t in transactions for i in t})
    freq, L = {}, []
    for i in items:
        fs = frozenset([i])
        if sup(fs) >= min_sup:
            freq[fs] = sup(fs); L.append(fs)
    k = 1
    while L and k < 4:
        cand = {a | b for a in L for b in L if len(a | b) == k + 1}
        L = []
        for c in cand:
            s = sup(c)
            if s >= min_sup:
                freq[c] = s; L.append(c)
        k += 1
    return freq


def _eclat(transactions, min_sup) -> Dict[frozenset, int]:
    tid = defaultdict(set)
    for ti, t in enumerate(transactions):
        for it in t:
            tid[it].add(ti)
    freq = {}
    items = sorted([(i, ts) for i, ts in tid.items() if len(ts) >= min_sup])

    def rec(prefix, items_tids):
        for idx, (it, ts) in enumerate(items_tids):
            newp = prefix + [it]
            freq[frozenset(newp)] = len(ts)
            suffix = [(it2, ts & ts2) for it2, ts2 in items_tids[idx + 1:]
                      if len(ts & ts2) >= min_sup]
            if suffix and len(newp) < 3:
                rec(newp, suffix)
    rec([], items)
    return freq


def _fpgrowth(transactions, min_sup) -> Dict[frozenset, int]:
    cnt = Counter(i for t in transactions for i in t)
    freqitems = {i for i, c in cnt.items() if c >= min_sup}
    if not freqitems:
        return {}
    rank = {it: r for r, it in enumerate(sorted(freqitems, key=lambda i: (-cnt[i], i)))}
    root = [None, 0, None, {}]
    headers = defaultdict(list)
    for t in transactions:
        node = root
        for it in sorted([i for i in t if i in freqitems], key=lambda i: rank[i]):
            if it in node[3]:
                child = node[3][it]; child[1] += 1
            else:
                child = [it, 1, node, {}]; node[3][it] = child; headers[it].append(child)
            node = child
    result: Dict[frozenset, int] = {}

    def mine(hdrs, suffix):
        for it in sorted(hdrs, key=lambda i: cnt.get(i, 0)):
            support = sum(n[1] for n in hdrs[it])
            if support < min_sup:
                continue
            newsuffix = [it] + suffix
            result[frozenset(newsuffix)] = support
            condpat = []
            for n in hdrs[it]:
                path, p = [], n[2]
                while p[0] is not None:
                    path.append(p[0]); p = p[2]
                condpat += [path[::-1]] * n[1]
            ccnt = Counter(i for pat in condpat for i in pat)
            cfreq = {i for i, c in ccnt.items() if c >= min_sup}
            if cfreq and len(newsuffix) < 4:
                croot = [None, 0, None, {}]
                chdr = defaultdict(list)
                for pat in condpat:
                    node = croot
                    for xx in [i for i in pat if i in cfreq]:
                        if xx in node[3]:
                            ch = node[3][xx]; ch[1] += 1
                        else:
                            ch = [xx, 1, node, {}]; node[3][xx] = ch; chdr[xx].append(ch)
                        node = ch
                mine(chdr, newsuffix)
    mine(headers, [])
    return result


# ----------------------------------------------------- sequential patterns
def _is_subseq(sub, seq) -> bool:
    it = iter(seq)
    return all(any(s == x for x in it) for s in sub)


def _prefixspan(sequences, min_sup, maxlen=4) -> Dict[tuple, int]:
    result: Dict[tuple, int] = {}

    def rec(prefix, projected):
        c = Counter()
        for suf in projected:
            for sym in set(suf):
                c[sym] += 1
        for sym, n in c.items():
            if n >= min_sup:
                newp = prefix + (sym,)
                result[newp] = n
                if len(newp) < maxlen:
                    proj = [suf[suf.index(sym) + 1:] for suf in projected if sym in suf]
                    rec(newp, proj)
    rec((), [tuple(s) for s in sequences])
    return result


def _gsp(sequences, min_sup, maxlen=4) -> Dict[tuple, int]:
    seqs = [tuple(s) for s in sequences]
    items = sorted({i for s in seqs for i in s})

    def sup(c):
        return sum(1 for s in seqs if _is_subseq(c, s))
    L = [(i,) for i in items if sup((i,)) >= min_sup]
    result = {c: sup(c) for c in L}
    k = 1
    while L and k < maxlen:
        cand = {a + (b[-1],) for a in L for b in L}
        L = []
        for c in cand:
            s = sup(c)
            if s >= min_sup:
                result[c] = s; L.append(c)
        k += 1
    return result


def _spade(sequences, min_sup, maxlen=4) -> Dict[tuple, int]:
    seqs = [tuple(s) for s in sequences]
    idl = defaultdict(list)
    for sid, s in enumerate(seqs):
        for pos, sym in enumerate(s):
            idl[sym].append((sid, pos))

    def support(il):
        return len({sid for sid, _ in il})
    freq1 = {(it,): il for it, il in idl.items() if support(il) >= min_sup}
    result = {p: support(il) for p, il in freq1.items()}

    def join(il_a, il_b):
        bypos = defaultdict(list)
        for sid, pos in il_b:
            bypos[sid].append(pos)
        out = []
        for sid, pos in il_a:
            for p2 in bypos.get(sid, []):
                if p2 > pos:
                    out.append((sid, p2)); break
        return out
    cur, k = freq1, 1
    while cur and k < maxlen:
        nxt = {}
        for pa, ila in cur.items():
            for pb, ilb in freq1.items():
                joined = join(ila, ilb)
                if support(joined) >= min_sup:
                    nxt[pa + (pb[-1],)] = joined
        result.update({p: support(il) for p, il in nxt.items()})
        cur, k = nxt, k + 1
    return result


def _predict_next(sequential_freq, symbols, w=WINDOW):
    """Most frequent symbol that extends the current trailing context."""
    if not symbols:
        return None
    ctx = tuple(symbols[-(w - 1):])
    best, best_sup = None, 0
    for pat, sup in sequential_freq.items():
        if len(pat) == len(ctx) + 1 and pat[:-1] == ctx and sup > best_sup:
            best, best_sup = pat[-1], sup
    return best or Counter(symbols).most_common(1)[0][0]


# --------------------------------------------------------------- nodes
class _MinerNode(Node):
    layer = "pattern"

    def _prep(self, X):
        x = X[:, 0].astype(float)
        symbols, alphabet = _symbolize(x)
        return x, symbols, alphabet


@register
class AprioriNode(_MinerNode):
    """Apriori — level-wise frequent itemsets over windowed symbol transactions."""
    node_type = "apriori_patterns"

    def _predict(self, X):
        x, sym, alpha = self._prep(X)
        tx = _transactions(sym)
        freq = _apriori(tx, _min_sup(len(tx))) if tx else {}
        return _emit(self, x, freq, alpha, None, "apriori")


@register
class EclatNode(_MinerNode):
    """ECLAT — depth-first frequent itemsets via vertical tid-set intersection."""
    node_type = "eclat_patterns"

    def _predict(self, X):
        x, sym, alpha = self._prep(X)
        tx = _transactions(sym)
        freq = _eclat(tx, _min_sup(len(tx))) if tx else {}
        return _emit(self, x, freq, alpha, None, "eclat")


@register
class FPGrowthNode(_MinerNode):
    """FP-Growth — frequent itemsets via an FP-tree + conditional pattern bases."""
    node_type = "fpgrowth_patterns"

    def _predict(self, X):
        x, sym, alpha = self._prep(X)
        tx = _transactions(sym)
        freq = _fpgrowth(tx, _min_sup(len(tx))) if tx else {}
        return _emit(self, x, freq, alpha, None, "fpgrowth")


@register
class PrefixSpanNode(_MinerNode):
    """PrefixSpan — projection-based sequential pattern mining (ordered motifs)."""
    node_type = "prefixspan_patterns"

    def _predict(self, X):
        x, sym, alpha = self._prep(X)
        seqs = _sequences(sym)
        freq = _prefixspan(seqs, _min_sup(len(seqs))) if seqs else {}
        return _emit(self, x, freq, alpha, _predict_next(freq, sym), "prefixspan")


@register
class GSPNode(_MinerNode):
    """GSP — level-wise sequential pattern mining (Generalized Sequential Patterns)."""
    node_type = "gsp_patterns"

    def _predict(self, X):
        x, sym, alpha = self._prep(X)
        seqs = _sequences(sym)
        freq = _gsp(seqs, _min_sup(len(seqs))) if seqs else {}
        return _emit(self, x, freq, alpha, _predict_next(freq, sym), "gsp")


@register
class SPADENode(_MinerNode):
    """SPADE — sequential pattern mining via vertical id-lists + temporal joins."""
    node_type = "spade_patterns"

    def _predict(self, X):
        x, sym, alpha = self._prep(X)
        seqs = _sequences(sym)
        freq = _spade(seqs, _min_sup(len(seqs))) if seqs else {}
        return _emit(self, x, freq, alpha, _predict_next(freq, sym), "spade")
