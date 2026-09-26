"""Blocking / candidate generation for entity resolution.

Memory-optimized:
 - Only 6 keys per entity (fewer than before) to keep index small
 - No reservoir sampling (_counts dict removed) — simpler truncation
 - Strings only in index (_norm_names, _addr_nums_str), no frozensets
 - max_per_key=80: posting lists capped to limit memory

Memory budget for 10M records:
  _index dict (~9M unique keys):  ~1.4 GB
  _norm_names (10M):              ~0.8 GB
  _addr_nums_str (10M):           ~0.4 GB
  Total index:                    ~2.6 GB
  (DataFrames already in memory:  ~3.7 GB)
  Grand total:                    ~6.3 GB (within 8.5 GB)
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from normalize import (
    extract_address_numbers,
    get_char_ngrams,
    get_name_tokens,
    normalize_address,
    normalize_name,
)


def _jaccard_frozen(a: frozenset, b: frozenset) -> float:
    u = a | b
    return (len(a & b) / len(u)) if u else 1.0


def _tokens_and_ngrams(norm_name: str):
    toks = get_name_tokens(norm_name)
    ng = frozenset(get_char_ngrams(''.join(toks[:5]), 3))
    return frozenset(toks), ng


def _make_keys_6(country: str, norm_name: str, addr_nums: List[str],
                 norm_addr: str) -> List[str]:
    """Generate at most 6 blocking keys per entity."""
    c = (country or 'xx').strip().lower()[:4]
    toks = get_name_tokens(norm_name)
    ng_list = get_char_ngrams(''.join(toks[:5]), 3)
    addr_words = [t for t in norm_addr.split() if len(t) > 2 and not t.isdigit()]
    keys: List[str] = []

    # 1. First significant name token
    if toks:
        keys.append(f"{c}|n1|{toks[0]}")

    # 2. Sorted pair of first two tokens (handles word-order swaps)
    if len(toks) >= 2:
        keys.append(f"{c}|n2|{'_'.join(sorted(toks[:2]))}")

    # 3. One representative char trigram (pick alphabetically smallest for determinism)
    if ng_list:
        keys.append(f"{c}|ng|{min(ng_list)}")

    # 4. First address number (for transliteration cases where name doesn't match)
    if addr_nums:
        keys.append(f"{c}|an|{addr_nums[0]}")

        # 5. Address number + first address word (more specific)
        if addr_words:
            keys.append(f"{c}|an_w|{addr_nums[0]}_{addr_words[0]}")

    # 6. Pair of address numbers (very specific, low false-positive)
    if len(addr_nums) >= 2:
        keys.append(f"{c}|an2|{'_'.join(sorted(addr_nums[:2]))}")

    return keys[:6]  # hard cap


def _compute(name: str, address: str, country: str):
    """Returns (keys, norm_name, addr_nums_list, norm_addr)."""
    norm_name = normalize_name(name)
    norm_addr = normalize_address(address)
    addr_nums = extract_address_numbers(norm_addr)
    keys = _make_keys_6(country, norm_name, addr_nums, norm_addr)
    return keys, norm_name, addr_nums, norm_addr


class BlockingIndex:
    """Memory-efficient inverted index (strings only, no frozensets, no counts)."""

    def __init__(self, max_per_key: int = 80):
        self.max_per_key = max_per_key
        self._index: Dict[str, List[str]] = defaultdict(list)
        self._norm_names: Dict[str, str] = {}
        self._addr_nums_str: Dict[str, str] = {}
        self._tok_cache: Dict[str, Tuple[frozenset, frozenset]] = {}

    def add(self, entity_id: str, name: str, address: str, country: str) -> None:
        keys, norm_name, addr_nums, _ = _compute(name, address, country)
        self._norm_names[entity_id] = norm_name
        self._addr_nums_str[entity_id] = ','.join(addr_nums)
        seen: Set[str] = set()
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            lst = self._index[key]
            if len(lst) < self.max_per_key:
                lst.append(entity_id)

    def build_from_df(self, df, id_col='entity_id', name_col='business_name',
                      addr_col='business_address', country_col='country',
                      verbose: bool = True) -> None:
        t0 = time.time()
        total = len(df)
        for i, row in enumerate(df.itertuples(index=False), 1):
            self.add(
                str(getattr(row, id_col)),
                str(getattr(row, name_col, '') or ''),
                str(getattr(row, addr_col, '') or ''),
                str(getattr(row, country_col, '') or ''),
            )
            if verbose and i % 1_000_000 == 0:
                print(f"    indexed {i:,}/{total:,} ({time.time()-t0:.0f}s)", flush=True)

    def query_and_score(self, name: str, address: str, country: str,
                        top_k: int = 50) -> List[Tuple[str, float]]:
        keys, norm_name, addr_nums, _ = _compute(name, address, country)
        s1_toks, s1_ngs = _tokens_and_ngrams(norm_name)
        s1_nums = frozenset(addr_nums)

        raw: Set[str] = set()
        for key in keys:
            raw.update(self._index.get(key, []))

        if not raw:
            return []

        scored: List[Tuple[float, str]] = []
        for eid in raw:
            cached = self._tok_cache.get(eid)
            if cached is None:
                c_norm = self._norm_names.get(eid, '')
                cached = _tokens_and_ngrams(c_norm)
                self._tok_cache[eid] = cached
            c_toks, c_ngs = cached
            c_addr_str = self._addr_nums_str.get(eid, '')
            c_nums = frozenset(c_addr_str.split(',')) if c_addr_str else frozenset()

            name_j = _jaccard_frozen(s1_toks, c_toks)
            ng_j = _jaccard_frozen(s1_ngs, c_ngs)
            num_b = 0.15 if (s1_nums and c_nums and s1_nums & c_nums) else 0.0
            scored.append((0.5 * name_j + 0.35 * ng_j + num_b, eid))

        scored.sort(reverse=True)
        return [(eid, sc) for sc, eid in scored[:top_k]]
