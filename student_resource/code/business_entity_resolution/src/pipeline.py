"""Main pipeline: blocking → feature extraction → LightGBM → output.

Memory-efficient design (streaming approach):
 - Build blocking index by STREAMING through S2+S3 files (no large DataFrames)
 - After blocking, read S2+S3 again to build reps ONLY for unique candidates
 - Peak memory ≈ 5.5 GB (index) vs 8+ GB with full DataFrames
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import BlockingIndex
from features import FEATURE_NAMES, EntityRep, build_reps, extract_features_from_reps

# ── Globals for forked workers ────────────────────────────────────────────────
_IDX: BlockingIndex | None = None
_C_REPS: Dict[str, EntityRep] = {}
_S1_REPS: Dict[str, EntityRep] = {}
_MODEL: lgb.Booster | None = None


def _init_blocking(idx):
    global _IDX
    _IDX = idx


def _init_inference(c_reps, s1_reps, model):
    global _C_REPS, _S1_REPS, _MODEL
    _C_REPS = c_reps
    _S1_REPS = s1_reps
    _MODEL = model


def _blocking_chunk(chunk, top_k):
    return [(sid, _IDX.query_and_score(nm, ad, ct, top_k=top_k))
            for sid, nm, ad, ct in chunk]


def _inference_chunk(chunk, threshold):
    results = []
    for sid, cands in chunk:
        r1 = _S1_REPS.get(sid)
        if not cands or r1 is None:
            results.append((sid, [])); continue
        X, eids = [], []
        for eid, _ in cands:
            r2 = _C_REPS.get(eid)
            if r2:
                X.append(extract_features_from_reps(r1, r2))
                eids.append(eid)
        if X:
            probs = _MODEL.predict(np.array(X, dtype=np.float32))
            matched = [e for e, p in zip(eids, probs) if p >= threshold]
        else:
            matched = []
        results.append((sid, matched))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_s1(data_dir: str, split: str) -> pd.DataFrame:
    return pd.read_csv(f"{data_dir}/{split}/{split}_source1.tsv", sep='\t', dtype=str).fillna('')


def read_gt(data_dir: str) -> pd.DataFrame:
    return pd.read_csv(f"{data_dir}/train/train_ground_truth.tsv", sep='\t', dtype=str).fillna('')


def stream_s2s3(data_dir: str, split: str, chunksize: int = 200_000):
    """Yield chunks of the combined S2+S3 DataFrame for the given split."""
    for src in ('source2', 'source3'):
        path = f"{data_dir}/{split}/{split}_{src}.tsv"
        for chunk in pd.read_csv(path, sep='\t', dtype=str, chunksize=chunksize):
            yield chunk.fillna('')


def build_index_streaming(data_dir: str, split: str, max_per_key: int,
                          chunksize: int = 200_000) -> BlockingIndex:
    """Build BlockingIndex by streaming S2+S3 files — never loads full DF."""
    t0 = time.time()
    print(f"  Building index (streaming) ...", flush=True)
    idx = BlockingIndex(max_per_key=max_per_key)
    n_total = 0
    for chunk in stream_s2s3(data_dir, split, chunksize):
        idx.build_from_df(chunk, verbose=False)
        n_total += len(chunk)
        if n_total % 1_000_000 == 0:
            print(f"    indexed {n_total:,} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  Index built: {n_total:,} records, {len(idx._index):,} keys in {time.time()-t0:.1f}s",
          flush=True)
    return idx


def build_reps_streaming(data_dir: str, split: str, target_ids: Set[str],
                         chunksize: int = 200_000) -> Dict[str, EntityRep]:
    """Build EntityRep for a subset of S2+S3 entities by streaming."""
    from normalize import normalize_name, normalize_address
    reps: Dict[str, EntityRep] = {}
    remaining = set(target_ids)
    for chunk in stream_s2s3(data_dir, split, chunksize):
        mask = chunk.entity_id.isin(remaining)
        if mask.any():
            sub = chunk[mask]
            reps.update(build_reps(sub))
            remaining -= set(sub.entity_id)
        if not remaining:
            break
    return reps


def build_gt_map(gt: pd.DataFrame) -> Dict[str, Set[str]]:
    gt_map: Dict[str, Set[str]] = defaultdict(set)
    for _, row in gt.iterrows():
        mids = str(row.get('matched_entity_ids', '')).strip()
        if mids and mids.lower() not in ('', 'nan'):
            for m in mids.split(','):
                m = m.strip()
                if m:
                    gt_map[row['source1_entity_id']].add(m)
    return gt_map


# ─────────────────────────────────────────────────────────────────────────────
# Blocking
# ─────────────────────────────────────────────────────────────────────────────

def parallel_block(s1: pd.DataFrame, idx: BlockingIndex, top_k: int,
                   n_workers: int) -> Dict[str, List[Tuple[str, float]]]:
    t0 = time.time()
    total = len(s1)
    print(f"  Blocking {total:,} S1 (k={top_k}, workers={n_workers}) ...", flush=True)
    rows = list(zip(s1.entity_id, s1.business_name, s1.business_address, s1.country))
    chunk_size = max(500, total // (n_workers * 12))
    chunks = [rows[i:i+chunk_size] for i in range(0, total, chunk_size)]

    results: Dict[str, List[Tuple[str, float]]] = {}
    if n_workers <= 1:
        for chunk in chunks:
            for sid, nm, ad, ct in chunk:
                results[sid] = idx.query_and_score(nm, ad, ct, top_k=top_k)
    else:
        ctx = mp.get_context('fork')
        with ctx.Pool(n_workers, initializer=_init_blocking, initargs=(idx,)) as pool:
            for batch in pool.starmap(_blocking_chunk, [(c, top_k) for c in chunks]):
                for sid, scored in batch:
                    results[sid] = scored
                done = len(results)
                if done % 300_000 < chunk_size:
                    print(f"    {done:,}/{total:,} ({time.time()-t0:.0f}s)", flush=True)

    avg = sum(len(v) for v in results.values()) / max(len(results), 1)
    print(f"  Blocking done: {time.time()-t0:.1f}s, avg_cands={avg:.1f}", flush=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Training pairs
# ─────────────────────────────────────────────────────────────────────────────

def build_training_features(
    s1_ids: List[str], s1_reps: Dict[str, EntityRep], c_reps: Dict[str, EntityRep],
    gt_map: Dict[str, Set[str]], candidates: Dict[str, List[Tuple[str, float]]],
    neg_ratio: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    print(f"  Building training features for {len(s1_ids):,} entities ...", flush=True)
    rng = random.Random(42)
    X_rows, y_rows = [], []
    n_pos = n_neg = n_miss = 0

    for sid in s1_ids:
        r1 = s1_reps.get(sid)
        if r1 is None:
            continue
        cands = candidates.get(sid, [])
        true = gt_map.get(sid, set())
        cand_ids = [e for e, _ in cands]
        cand_set = set(cand_ids)

        pos = [(e, 1) for e in cand_ids if e in true]
        for e in list(true - cand_set)[:2]:
            if e in c_reps:
                pos.append((e, 1)); n_miss += 1

        neg_pool = [e for e in cand_ids if e not in true]
        neg = [(e, 0) for e in rng.sample(neg_pool, min(len(neg_pool),
               max(len(pos), 1) * neg_ratio))] if neg_pool else []

        for e, label in pos + neg:
            r2 = c_reps.get(e)
            if r2 is None:
                continue
            X_rows.append(extract_features_from_reps(r1, r2))
            y_rows.append(label)
            if label:
                n_pos += 1
            else:
                n_neg += 1

    print(f"  {n_pos:,} pos, {n_neg:,} neg, {n_miss:,} miss-recov in {time.time()-t0:.1f}s", flush=True)
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int8)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def train_lgbm(X: np.ndarray, y: np.ndarray) -> lgb.Booster:
    t0 = time.time()
    print(f"  Training LightGBM on {len(X):,} pairs ...", flush=True)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    pos_w = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
    params = {
        'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
        'num_leaves': 127, 'min_child_samples': 20,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'verbose': -1, 'n_jobs': -1, 'scale_pos_weight': pos_w,
    }
    ds = lgb.Dataset(X_tr, y_tr, feature_name=FEATURE_NAMES)
    dsv = lgb.Dataset(X_val, y_val, reference=ds)
    model = lgb.train(params, ds, num_boost_round=600, valid_sets=[dsv],
                      callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(100)])
    print(f"  {model.num_trees()} trees in {time.time()-t0:.1f}s", flush=True)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# F_0.5 evaluation
# ─────────────────────────────────────────────────────────────────────────────

def f05(pred: Dict[str, Set[str]], true: Dict[str, Set[str]], ids: List[str]) -> float:
    scores = []
    for sid in ids:
        p, t = pred.get(sid, set()), true.get(sid, set())
        if not p and not t:
            scores.append(1.0)
        elif not p or not t:
            scores.append(0.0)
        else:
            tp = len(p & t)
            pr = tp / len(p); rc = tp / len(t)
            scores.append((1.25 * pr * rc) / (0.25 * pr + rc) if (pr + rc) > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def parallel_infer(
    s1_ids: List[str], candidates: Dict[str, List[Tuple[str, float]]],
    s1_reps: Dict[str, EntityRep], c_reps: Dict[str, EntityRep],
    model: lgb.Booster, threshold: float, n_workers: int,
) -> Dict[str, List[str]]:
    t0 = time.time()
    total = len(s1_ids)
    print(f"  Inference: {total:,} entities, threshold={threshold:.2f}, workers={n_workers}", flush=True)

    items = [(sid, candidates.get(sid, [])) for sid in s1_ids]
    chunk_size = max(500, total // (n_workers * 12))
    chunks = [items[i:i+chunk_size] for i in range(0, total, chunk_size)]

    results: Dict[str, List[str]] = {}
    if n_workers <= 1:
        for chunk in chunks:
            for sid, cands in chunk:
                r1 = s1_reps.get(sid)
                if not cands or r1 is None:
                    results[sid] = []; continue
                X, eids = [], []
                for eid, _ in cands:
                    r2 = c_reps.get(eid)
                    if r2:
                        X.append(extract_features_from_reps(r1, r2))
                        eids.append(eid)
                if X:
                    probs = model.predict(np.array(X, dtype=np.float32))
                    results[sid] = [e for e, p in zip(eids, probs) if p >= threshold]
                else:
                    results[sid] = []
    else:
        ctx = mp.get_context('fork')
        with ctx.Pool(n_workers, initializer=_init_inference,
                      initargs=(c_reps, s1_reps, model)) as pool:
            for batch in pool.starmap(_inference_chunk, [(c, threshold) for c in chunks]):
                for sid, matched in batch:
                    results[sid] = matched
                done = len(results)
                if done % 300_000 < chunk_size:
                    print(f"    {done:,}/{total:,} ({time.time()-t0:.0f}s)", flush=True)

    n_matched = sum(1 for v in results.values() if v)
    print(f"  Inference done: {time.time()-t0:.1f}s, {n_matched:,}/{total:,} matched", flush=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def write_outputs(s1: pd.DataFrame, candidates, predictions, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    mr_rows, cp_rows = [], []
    for row in s1.itertuples(index=False):
        sid = row.entity_id
        preds = list(dict.fromkeys(predictions.get(sid, [])))
        cands = list(dict.fromkeys(e for e, _ in candidates.get(sid, [])))
        cand_set = set(cands)
        for p in preds:
            if p not in cand_set:
                cands.append(p)
        mr_rows.append({'source1_entity_id': sid, 'matched_entity_ids': ','.join(preds)})
        cp_rows.append({'source1_entity_id': sid, 'candidate_entity_ids': ','.join(cands)})
    pd.DataFrame(mr_rows).to_csv(f"{out_dir}/matching_results.tsv", sep='\t', index=False)
    pd.DataFrame(cp_rows).to_csv(f"{out_dir}/candidate_pairs.tsv", sep='\t', index=False)
    n_m = sum(1 for r in mr_rows if r['matched_entity_ids'])
    print(f"  {len(mr_rows):,} rows written: {n_m:,} matched, {len(mr_rows)-n_m:,} singletons", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='dataset')
    ap.add_argument('--output-dir', default='output')
    ap.add_argument('--top-k', type=int, default=60)
    ap.add_argument('--threshold', type=float, default=0.45)
    ap.add_argument('--max-per-key', type=int, default=80)
    ap.add_argument('--train-s1-cap', type=int, default=400_000)
    ap.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args()

    t_start = time.time()
    print("=" * 60)
    print("Business Entity Resolution Pipeline")
    print(f"  workers={args.workers}  top_k={args.top_k}  max_per_key={args.max_per_key}")
    print("=" * 60)

    # ── [1] Load training S1 + GT ─────────────────────────────────────────────
    print("\n[1] Loading training S1 and ground truth ...")
    s1_tr = read_s1(args.data_dir, 'train')
    gt = read_gt(args.data_dir)
    gt_map = build_gt_map(gt)
    del gt; gc.collect()

    rng_np = np.random.default_rng(42)
    n_tr = min(len(s1_tr), args.train_s1_cap)
    s1_tr_sample = s1_tr.iloc[rng_np.choice(len(s1_tr), n_tr, replace=False)].reset_index(drop=True)
    del s1_tr; gc.collect()
    print(f"  S1 sample: {n_tr:,} entities")

    # ── [2] Build training blocking index (streaming) ─────────────────────────
    print("\n[2] Building training blocking index ...")
    idx_tr = build_index_streaming(args.data_dir, 'train', args.max_per_key)

    # ── [3] Generate training candidates ──────────────────────────────────────
    print("\n[3] Generating training candidates ...")
    tr_cands = parallel_block(s1_tr_sample, idx_tr, top_k=args.top_k, n_workers=args.workers)
    del idx_tr; gc.collect()

    # ── [4] Build entity reps (streaming) ─────────────────────────────────────
    print("\n[4] Building entity reps ...")
    t0 = time.time()
    unique_cand_ids = set(e for cs in tr_cands.values() for e, _ in cs)
    # Also include GT positives that may have been missed by blocking
    for sid in s1_tr_sample.entity_id:
        unique_cand_ids.update(gt_map.get(sid, set()))

    c_reps_tr = build_reps_streaming(args.data_dir, 'train', unique_cand_ids)
    s1_reps_tr = build_reps(s1_tr_sample)
    print(f"  {len(c_reps_tr):,} candidate reps, {len(s1_reps_tr):,} S1 reps [{time.time()-t0:.1f}s]")

    # ── [5] Training features ──────────────────────────────────────────────────
    print("\n[5] Building training features ...")
    s1_tr_ids = list(s1_tr_sample.entity_id)
    X, y = build_training_features(s1_tr_ids, s1_reps_tr, c_reps_tr, gt_map, tr_cands)

    # ── [6] Train model ────────────────────────────────────────────────────────
    print("\n[6] Training LightGBM ...")
    model = train_lgbm(X, y)
    del X, y; gc.collect()

    # ── [7] Threshold tuning ───────────────────────────────────────────────────
    print("\n[7] Tuning threshold ...")
    n_val = min(15_000, n_tr // 5)
    val_ids = s1_tr_ids[-n_val:]
    val_cands = {sid: tr_cands[sid] for sid in val_ids if sid in tr_cands}
    val_gt = {sid: gt_map.get(sid, set()) for sid in val_ids}

    best_thresh, best_f = args.threshold, 0.0
    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        pm = parallel_infer(val_ids, val_cands, s1_reps_tr, c_reps_tr, model,
                            threshold=thresh, n_workers=1)
        score = f05({k: set(v) for k, v in pm.items()}, val_gt, val_ids)
        print(f"  threshold={thresh:.2f}  F_0.5={score:.4f}")
        if score > best_f:
            best_f, best_thresh = score, thresh
    print(f"  Best: threshold={best_thresh:.2f}, F_0.5={best_f:.4f}")
    del tr_cands, c_reps_tr, s1_reps_tr, s1_tr_sample; gc.collect()

    # ── [8] Load test S1 ──────────────────────────────────────────────────────
    print("\n[8] Loading test S1 ...")
    s1_te = read_s1(args.data_dir, 'test')
    print(f"  Test S1: {len(s1_te):,} entities")

    # ── [9] Build test blocking index (streaming) ─────────────────────────────
    print("\n[9] Building test blocking index ...")
    idx_te = build_index_streaming(args.data_dir, 'test', args.max_per_key)

    # ── [10] Generate test candidates ─────────────────────────────────────────
    print("\n[10] Generating test candidates ...")
    te_cands = parallel_block(s1_te, idx_te, top_k=args.top_k, n_workers=args.workers)
    del idx_te; gc.collect()

    # ── [11] Build test entity reps (streaming) ────────────────────────────────
    print("\n[11] Building test entity reps ...")
    t0 = time.time()
    unique_te_ids = set(e for cs in te_cands.values() for e, _ in cs)
    c_reps_te = build_reps_streaming(args.data_dir, 'test', unique_te_ids)
    s1_reps_te = build_reps(s1_te)
    print(f"  {len(c_reps_te):,} candidate reps, {len(s1_reps_te):,} S1 reps [{time.time()-t0:.1f}s]")

    # ── [12] Test inference ───────────────────────────────────────────────────
    print("\n[12] Running test inference ...")
    te_preds = parallel_infer(list(s1_te.entity_id), te_cands, s1_reps_te, c_reps_te,
                              model, threshold=best_thresh, n_workers=args.workers)

    # ── [13] Write outputs ────────────────────────────────────────────────────
    print("\n[13] Writing outputs ...")
    write_outputs(s1_te, te_cands, te_preds, args.output_dir)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done in {total:.1f}s ({total/60:.1f} min)")
    print(f"Val F_0.5={best_f:.4f} @ threshold={best_thresh:.2f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
