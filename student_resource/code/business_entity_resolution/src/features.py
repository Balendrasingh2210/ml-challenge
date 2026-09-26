"""Feature extraction for entity pair classification.

EntityRep stores only string fields (norm_name, norm_addr, country).
Frozensets are computed on-the-fly inside extract_features_from_reps.
Memory: ~200 bytes per entity vs ~850 bytes with precomputed frozensets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from rapidfuzz import fuzz

from normalize import (
    extract_address_numbers,
    get_char_ngrams,
    get_name_tokens,
    normalize_address,
    normalize_name,
)


@dataclass(slots=True)
class EntityRep:
    """Compact entity representation — strings only, frozensets computed on demand."""
    norm_name: str
    norm_addr: str
    country: str


def make_rep(name: str, address: str, country: str) -> EntityRep:
    return EntityRep(
        normalize_name(name),
        normalize_address(address),
        (country or '').strip().lower(),
    )


def build_reps(df, name_col='business_name', addr_col='business_address',
               country_col='country', id_col='entity_id') -> Dict[str, EntityRep]:
    """Build EntityRep dict for all rows in df."""
    reps: Dict[str, EntityRep] = {}
    for row in df.itertuples(index=False):
        eid = str(getattr(row, id_col))
        reps[eid] = EntityRep(
            normalize_name(str(getattr(row, name_col, '') or '')),
            normalize_address(str(getattr(row, addr_col, '') or '')),
            (str(getattr(row, country_col, '') or '')).strip().lower(),
        )
    return reps


def _jaccard(a, b) -> float:
    u = a | b
    return (len(a & b) / len(u)) if u else 1.0


def extract_features_from_reps(r1: EntityRep, r2: EntityRep) -> List[float]:
    """Extract 20 numeric features from two EntityRep objects."""
    # ── Name features ──────────────────────────────────────────────────────
    n1, n2 = r1.norm_name[:120], r2.norm_name[:120]

    # Word-token Jaccard
    t1 = frozenset(get_name_tokens(r1.norm_name))
    t2 = frozenset(get_name_tokens(r2.norm_name))
    f_name_tok_j = _jaccard(t1, t2)

    # Char-trigram Jaccard
    ng1 = frozenset(get_char_ngrams(''.join(list(t1)[:5]), 3))
    ng2 = frozenset(get_char_ngrams(''.join(list(t2)[:5]), 3))
    f_name_ng_j = _jaccard(ng1, ng2)

    # rapidfuzz string ratios
    f_name_ratio = fuzz.ratio(n1, n2) / 100.0
    f_name_tsort = fuzz.token_sort_ratio(n1, n2) / 100.0
    f_name_tset = fuzz.token_set_ratio(n1, n2) / 100.0
    f_name_partial = fuzz.partial_ratio(n1, n2) / 100.0

    # Length ratio
    l1, l2 = len(r1.norm_name), len(r2.norm_name)
    f_name_len = min(l1, l2) / max(l1, l2, 1)

    # Common prefix (contiguous)
    mp = min(l1, l2, 20)
    cp = 0
    for i in range(mp):
        if r1.norm_name[i] == r2.norm_name[i]:
            cp += 1
        else:
            break
    f_name_pfx = cp / max(mp, 1)

    f_name_any_tok = float(bool(t1 & t2))
    f_name_avg = (f_name_tok_j + f_name_ng_j + f_name_ratio + f_name_tsort + f_name_tset) / 5.0

    # ── Address features ───────────────────────────────────────────────────
    a1, a2 = r1.norm_addr[:120], r2.norm_addr[:120]

    at1 = frozenset(r1.norm_addr.split())
    at2 = frozenset(r2.norm_addr.split())
    f_addr_tok_j = _jaccard(at1, at2)

    ang1 = frozenset(get_char_ngrams(r1.norm_addr[:50], 3))
    ang2 = frozenset(get_char_ngrams(r2.norm_addr[:50], 3))
    f_addr_ng_j = _jaccard(ang1, ang2)

    f_addr_ratio = fuzz.ratio(a1, a2) / 100.0
    f_addr_tsort = fuzz.token_sort_ratio(a1, a2) / 100.0
    f_addr_partial = fuzz.partial_ratio(a1, a2) / 100.0

    an1 = frozenset(extract_address_numbers(r1.norm_addr))
    an2 = frozenset(extract_address_numbers(r2.norm_addr))
    if an1 and an2:
        f_addr_num_j = _jaccard(an1, an2)
        f_addr_common_num = float(bool(an1 & an2))
    else:
        f_addr_num_j = 0.0
        f_addr_common_num = 0.0

    f_s1_has_addr = float(len(r1.norm_addr) > 3)
    f_s2_has_addr = float(len(r2.norm_addr) > 3)

    # ── Country ───────────────────────────────────────────────────────────
    f_country = float(r1.country == r2.country)

    return [
        f_name_tok_j, f_name_ng_j, f_name_ratio, f_name_tsort, f_name_tset, f_name_partial,
        f_name_len, f_name_pfx, f_name_any_tok, f_name_avg,
        f_addr_tok_j, f_addr_ng_j, f_addr_ratio, f_addr_tsort, f_addr_partial,
        f_addr_num_j, f_addr_common_num, f_s1_has_addr, f_s2_has_addr,
        f_country,
    ]


FEATURE_NAMES = [
    'name_tok_j', 'name_ng_j', 'name_ratio', 'name_tsort', 'name_tset', 'name_partial',
    'name_len', 'name_pfx', 'name_any_tok', 'name_avg',
    'addr_tok_j', 'addr_ng_j', 'addr_ratio', 'addr_tsort', 'addr_partial',
    'addr_num_j', 'addr_common_num', 's1_has_addr', 's2_has_addr',
    'country_match',
]
