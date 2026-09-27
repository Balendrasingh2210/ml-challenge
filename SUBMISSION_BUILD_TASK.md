# Task: Assemble the final submission zip (baseline, F_0.5≈0.7288)

You are a fresh Claude Code session with no memory of prior work on this repo. This
file is fully self-contained — read it carefully before doing anything.

## Background

This repo (`ml-challenge`) implements a business entity-resolution pipeline for an
ML challenge: match business records across 3 noisy data sources using blocking +
hand-crafted similarity features + a LightGBM classifier. The pipeline already ran
successfully end-to-end on Kaggle and produced validated output files. **Your job is
only to assemble the final submission zip from what already exists — do not modify,
retrain, or re-run the pipeline.** A separate branch (`feature/llm-embeddings`) is
being used concurrently to improve the model; stay off that branch and don't touch
`student_resource/code/business_entity_resolution/src/*.py` — those files must remain
byte-identical to what actually produced the results below, for reproducibility.

Work on the `main` branch. Commit the new files you create (README.md,
requirements.txt, filled-in Documentation_template.md) directly to `main` — this
branch is reserved as the frozen, submission-ready baseline, so it's safe to commit
here without any conflict with the embeddings work happening elsewhere.

## What already exists (do not regenerate)

- **Output files** (already downloaded from Kaggle, already validated — see below):
  - `C:\Users\harsh\Desktop\AmazonMLChallenge\student_resource\output\matching_results.tsv` (106.4 MB)
  - `C:\Users\harsh\Desktop\AmazonMLChallenge\student_resource\output\candidate_pairs.tsv` (1.36 GB)
  - These directories are gitignored (too large for git) — reference them by absolute
    path, don't try to `git add` them.
- **Source code** (already correct, already produced the above outputs):
  `student_resource/code/business_entity_resolution/src/{blocking.py, features.py,
  normalize.py, pipeline.py}`
- **Validator**: `student_resource/utils/validate_submission.py` (stdlib only, no deps)
- **Challenge spec**: `C:\Users\harsh\Desktop\AmazonMLChallenge\submission_format.md`
  and `C:\Users\harsh\Desktop\AmazonMLChallenge\problemStatement.md` — read both in
  full before proceeding; they define the exact required zip structure and rules.
- **Methodology template** (currently blank): `student_resource/Documentation_template.md`

## Known facts about the run that produced these outputs (for the methodology doc)

- **Approach type:** Hybrid (Blocking + Classifier)
- **Blocking:** inverted index over 6 keys per entity — first significant name
  token; sorted pair of first two name tokens (handles word-order swaps); one
  representative char-trigram; first address number; address-number + first
  address word; pair of address numbers. `max_per_key=80`, `top_k=60` candidates
  per Source-1 entity kept after Jaccard-based scoring (name-token Jaccard 0.5
  weight + char-trigram Jaccard 0.35 weight + address-number-overlap bonus 0.15).
- **Features (20 total):** name-token Jaccard, name char-trigram Jaccard,
  rapidfuzz ratio/token_sort_ratio/token_set_ratio/partial_ratio, name length
  ratio, common-prefix ratio, any-shared-token flag, average of the above 5;
  same set of address features (token Jaccard, char-trigram Jaccard, ratio,
  token_sort_ratio, partial_ratio, address-number Jaccard, common-number flag,
  has-address flags for both sides); country-match flag. See `FEATURE_NAMES` in
  `features.py` for the exact list/order.
- **Model:** LightGBM binary classifier, `num_leaves=127`, `learning_rate=0.05`,
  `scale_pos_weight` set to the pos/neg class ratio, trained with early stopping
  (40 rounds patience) on features from `TRAIN_S1_CAP=150,000` sampled Source-1
  training entities (reduced from the full 400k available, as a deliberate
  speed/reliability tradeoff after multiple debugging iterations — noted honestly
  in the doc, not hidden).
- **Threshold tuning:** grid search over {0.30, 0.35, ..., 0.65} on a held-out
  15,000-entity validation slice; best was **threshold=0.65, F_0.5=0.7288**.
- **Test-set results:** 1,732,544 total Source-1 test entities; 1,631,691 matched
  (94.2%), 100,853 singletons (5.8%). Passed `validate_submission.py --check-ids`
  (strict mode) with zero blocking issues.
- **Known limitation to mention honestly:** validation F_0.5 (0.7288) was measured
  on a held-out slice of training data, not the real test set — the actual
  leaderboard score may differ, particularly for France entities (a country
  present in test but absent from training).

## What you need to produce

1. **`student_resource/code/business_entity_resolution/README.md`** — how to
   reproduce end-to-end: install deps from requirements.txt, then run
   `pipeline.py` with the exact CLI flags used (`--top-k 60 --max-per-key 80
   --threshold 0.45 --train-s1-cap 150000 --workers 1`, data-dir pointing at a
   folder with `train/` and `test/` TSVs). Keep it short and accurate — don't
   invent steps that don't exist in the code.
2. **`student_resource/code/business_entity_resolution/requirements.txt`** —
   pinned: `pandas`, `lightgbm==4.7.0`, `rapidfuzz==3.14.6`, `scikit-learn==1.9.1`,
   `numpy`. Verify these are the actual versions the code was run with if you can
   (check installed local environment) rather than trusting this list blindly.
3. **Fill in `student_resource/Documentation_template.md`** using the facts above
   — write real prose, not placeholder brackets. Keep the existing section
   structure/headings.
4. **Run the validator** to reconfirm before packaging:
   ```
   cd student_resource
   python3 utils/validate_submission.py \
       --matching  "C:\Users\harsh\Desktop\AmazonMLChallenge\student_resource\output\matching_results.tsv" \
       --candidate "C:\Users\harsh\Desktop\AmazonMLChallenge\student_resource\output\candidate_pairs.tsv" \
       --test-dir  dataset/test --check-ids
   ```
   Must print `PASS`. If it doesn't, STOP and report — do not package a failing submission.
5. **Assemble the zip** at
   `C:\Users\harsh\Desktop\AmazonMLChallenge\<team_name>_submission.zip` with
   exactly this structure (see `submission_format.md` for the authoritative spec):
   ```
   <team_name>_submission.zip
   ├── output/
   │   ├── matching_results.tsv
   │   └── candidate_pairs.tsv
   ├── code/
   │   └── business_entity_resolution/
   │       ├── src/                    (copy of the four .py files, unmodified)
   │       ├── README.md
   │       └── requirements.txt
   └── Documentation_template.md        (filled in)
   ```
   Ask the user what `<team_name>` should be if it's not otherwise obvious from
   the repo/account context.

## Verification

- `validate_submission.py --check-ids` PASS (step 4 above) before zipping.
- After zipping, spot check: `unzip -l` the archive and confirm the exact
  directory structure above, and that `output/matching_results.tsv` inside the
  zip is byte-identical (or at least same size) to the source file.
- Report the final zip path and size back to the user when done.
