# Task: Add semantic embedding features to close the leaderboard gap

You are a fresh Claude Code session (or continuing one) with no memory of prior work
on this repo unless told otherwise. This file is self-contained — read fully before
acting. Work on this `feature/llm-embeddings` branch only; do not touch `main`
(reserved as the frozen, already-submitted baseline at F_0.5≈0.7288).

## Context — why this work exists

The existing pipeline (blocking → 20 hand-crafted fuzzy-string features → LightGBM →
threshold tuning, in `student_resource/code/business_entity_resolution/src/`)
produced a validated, submitted baseline: **F_0.5 ≈ 0.7288** on our own held-out
training split, 94.2% match rate on the real 1,732,544-entity test set, passed
`validate_submission.py --check-ids`.

The actual competition leaderboard shows top teams clustered at **F_0.5 ≈ 0.991**
(screenshot shared by the user showed rank 1 through 12 all near that number). That's
a ~25-30 point gap.

**Key diagnostic clue:** LightGBM training hit **AUC ≈ 0.999** on validation with only
150k training entities — the classifier is already near-perfectly separating
positive/negative pairs *among the candidates it's shown*. This points at **blocking
recall ceiling** (true matches never reaching the classifier because blocking's
token/n-gram index missed them) as the likely dominant bottleneck, with feature
quality on semantically-hard pairs (transliteration, cross-lingual, paraphrase, and
the France test entities — a country entirely absent from training) as a secondary
factor. A cheap diagnostic to measure blocking recall ceiling directly (Phase 0
below) was proposed but skipped by the user in favor of going straight to Phase 1a;
consider running it if Phase 1a's gain is smaller than hoped, since it would tell you
definitively whether to invest further in blocking (Phase 1b) instead.

## What's explicitly allowed (re-confirm by reading `problemStatement.md` yourself,
particularly lines ~120-170, before making any licensing assumptions)

> "Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters."

Fair-play rules prohibit calling **external APIs/services/databases to look up
business identities** (commercial entity-resolution APIs, geocoding APIs, government
registries, "any external data augmentation from internet sources"). Using a
permissively-licensed **pretrained model's weights, run locally/offline** (no network
calls at inference/scoring time — downloading the weights once during setup is fine)
is different from that prohibition and is explicitly anticipated by the
license/parameter-count clause. `Documentation_template.md` lists "Hybrid" as an
expected approach type. This is why an embedding or small-LLM component is the
intended next step, not a rules risk.

## Already validated during initial investigation (don't redo this part)

- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  (Apache-2.0, 118M params, 384-dim output, supports 50+ languages) was
  downloaded and smoke-tested locally. Similarity results were strong and directly
  targeted the weakness of fuzzy-string matching:
  - `"Acme Corporation"` vs `"Acme Corp"` → 0.987
  - `"Acme Corporation"` vs `"ACME CORP PVT LTD"` → 0.915 (fuzzy-string methods
    struggle more with this one — legal-suffix + case changes)
  - `"Sharma Enterprises Pvt Ltd"` vs `"Sharma Enterprise"` → 0.858
  - Unrelated business-name pairs → 0.28–0.50 (well separated)
  - This is a strong existing candidate model; re-evaluate `intfloat/multilingual-e5-small`
    (MIT, 118M) too if you want a second option, but don't assume you need to start over.
- **CPU throughput measured:** ~382 texts/sec (batch of 3000, batch_size default).
  At full scale (~8-9M unique candidate+S1 entities × 2 fields [name, address] ≈
  ~17-18M texts for the test stage alone, similar order for training) this is
  **~14+ hours on CPU alone — not viable.** GPU acceleration is required.
- **GPU status when this was last touched:** local machine has an NVIDIA RTX 4050
  (6GB VRAM, driver supports CUDA 13.0) but `pip install torch --index-url
  https://download.pytorch.org/whl/cu121` did NOT result in `torch.cuda.is_available()`
  returning True — this was not debugged further due to time pressure (an
  approaching submission deadline forced a stop). **Fix this first** — likely need
  a CUDA build matching the installed driver more precisely, or a clean venv to
  avoid a stale CPU-only torch shadowing the new install. Alternative: use Kaggle's
  free GPU quota (30 hrs, completely unused so far this session, T4/P100 typically)
  instead of fighting local CUDA setup — Kaggle has already proven reliable for
  long-running work in this project (see git log / `kaggle_notebook.ipynb`).

## Critical memory-safety constraint (learned expensively overnight — do not repeat)

**Do NOT embed all ~10M+ records per source file.** Earlier debugging on this exact
project hit real out-of-memory kernel crashes (`nbclient.exceptions.DeadKernelError`)
from holding too much per-entity state in memory across forked worker processes.
Only compute embeddings for entities that already enter `c_reps`/`s1_reps` in
`pipeline.py` — i.e. piggyback on the exact same scope as the existing
`build_reps`/`build_reps_streaming` calls (these already narrow the ~10M-record raw
files down to just the entities that survived blocking as actual candidates, or are
S1 query entities — typically millions, not tens of millions). Store embeddings as
`float16` (384-dim × 2 bytes ≈ 768 bytes/entity — cheap at this scope) or on the
`EntityRep` dataclass itself.

## Integration points (read these files fully before editing)

- `student_resource/code/business_entity_resolution/src/features.py`:
  - `EntityRep` dataclass (currently `norm_name: str, norm_addr: str, country: str`,
    `slots=True` for memory efficiency) — extend with optional embedding fields, or
    maintain a parallel `Dict[str, Tuple[np.ndarray, np.ndarray]]` (name_embed,
    addr_embed) keyed by entity_id if you'd rather not touch the dataclass shape.
  - `extract_features_from_reps(r1, r2)` — currently returns a 20-float list; add
    2 more (cosine similarity of name embeddings, cosine similarity of address
    embeddings) and update the parallel `FEATURE_NAMES` list at the bottom of the
    file. Keep the existing 20 untouched — this is additive, not a replacement,
    to minimize risk of regressing what's already working.
- `student_resource/code/business_entity_resolution/src/pipeline.py`:
  - After each `build_reps`/`build_reps_streaming` call (there are several — for
    training S1, training candidates, test S1, test candidates), add one batched
    embedding-encode pass. Dedupe identical normalized strings first (many
    businesses share generic terms/abbreviated addresses) to avoid redundant
    encoding work.
  - The existing `--workers` CLI flag and the `_mp_context()` platform-aware
    fork/spawn helper (already fixed for Windows compatibility) are unrelated to
    this change — embedding encoding is a separate, single-process (or GPU-batched)
    step, not part of the blocking/inference multiprocessing pools.

## Phase 1b (only if Phase 1a's gain disappoints, or the recall-ceiling diagnostic
confirms blocking itself is the bottleneck)

Building an approximate-nearest-neighbor index (e.g. `faiss`) over the *full*
~10M-record index per source, as a second blocking channel alongside the existing
token/n-gram inverted index, would catch matches invisible to token-based blocking
(no shared tokens/n-grams at all). This reintroduces the full-scale memory problem
Phase 1a specifically avoids — mitigate with an on-disk or quantized (IVF-PQ) faiss
index rather than in-RAM float vectors, and/or only apply ANN blocking as a fallback
for S1 entities whose token-based candidate set came back empty or suspiciously
small. Don't start this until Phase 1a is measured.

## Verification

1. Re-run the existing threshold-tuning loop (unchanged) with the new feature set on
   the same held-out validation methodology used for the baseline; compare F_0.5
   against 0.7288 before/after.
2. Full pipeline re-run must still pass `utils/validate_submission.py --check-ids`.
3. Do not overwrite the `main` branch's submitted baseline files — if this branch's
   result is better, that becomes a *new* candidate submission (subject to whatever
   the competition's resubmission policy allows), not an automatic replacement.
4. Be honest in any results reporting: closing the full ~25-30 point gap to ~0.99 is
   a large ask that may not be fully achievable via this path alone — measure and
   report the real delta rather than assuming the target score.
