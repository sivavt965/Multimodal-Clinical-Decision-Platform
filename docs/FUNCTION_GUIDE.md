# Function Guide

This guide explains the maintained baseline path in plain engineering terms.
It is meant to help a reviewer understand which files matter before opening
the code.

## Backend Inference

- `backend/engine/inference.py::run_cxr_inference(...)`: loads a CXR image,
  applies the same resize and tensor conversion used by the baseline model,
  runs DenseNet121, and returns calibrated finding probabilities.
- `backend/engine/inference.py::_compute_gradcam(...)`: attaches hooks to the
  final convolutional block, backpropagates the selected class score, and turns
  the resulting activations into a heatmap.
- `backend/engine/inference.py::_run_mc_dropout(...)`: repeats inference with
  dropout active so the platform can expose predictive uncertainty instead of
  only point probabilities.
- `backend/engine/inference.py::_extract_gap_embedding(...)`: captures the
  DenseNet global-average-pooled feature vector used for FAISS retrieval.

## Vector Search

- `backend/engine/vector_store.py::VectorStore.for_model(...)`: chooses the FAISS index
  and id-map paths for a named embedding family.
- `backend/engine/vector_store.py::VectorStore.add_to_index(...)`: normalizes new
  embeddings and appends them with the matching case ids.
- `backend/engine/vector_store.py::VectorStore.search_similar(...)`: normalizes the
  query vector, runs nearest-neighbor search, and maps FAISS rows back to case
  ids plus similarity scores.

## Baseline Training

- `mimic_project/src/training/train_baseline.py::parse_args()`: defines the
  reproducible training contract for image-only CXR training.
- `mimic_project/src/training/train_baseline.py::make_preferential_mask(...)`:
  gives uncertain labels lower weight while keeping observed positives and
  negatives in the loss.
- `mimic_project/src/training/train_baseline.py::masked_bce_with_logits(...)`:
  applies binary cross-entropy only where label weights are non-zero.
- `mimic_project/src/training/train_baseline.py::run_one_epoch(...)`: central
  train/eval loop; it is the best first read for understanding metrics and
  checkpoint behavior.
- `mimic_project/src/models/densenet121.py::DenseNet121.forward(...)`: returns
  eight CheXpert-style logits from a 3-channel 512x512 CXR tensor.

## Authentication And Audit

- `backend/auth.py`: verifies Supabase JWTs, maps the Supabase identity to the
  local `users` table, and exposes role guards used by protected endpoints.
- `backend/database.py::log_audit(...)`: records mutating clinical actions and
  updates `last_active_at`; failures are intentionally non-fatal so audit
  logging cannot crash the clinical workflow.

## Optional Symile Runtime

The public repo keeps only the minimal runtime files needed by
`backend/engine/symile_encoder.py`. Full Symile training, preprocessing, and
paper reproduction code belongs upstream:
[rajesh-lab/symile](https://github.com/rajesh-lab/symile).
