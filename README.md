# Multimodal Clinical Decision Support Platform

A role-based clinical decision support system that combines chest X-ray analysis (DenseNet121 + Grad-CAM + MC Dropout uncertainty), MIMIC-IV laboratory values, and ECG metadata. Built on the Symile-MIMIC dataset.

**Stack:** FastAPI · Next.js 13 · Supabase (PostgreSQL + Storage) · PyTorch · FAISS · TailwindCSS

## What it does

- **Phase A** — Tabular risk scoring from ECG + 50 MIMIC-IV lab biomarkers
- **Phase B** — CXR classification (8 CheXpert findings) with Grad-CAM heatmaps and MC Dropout uncertainty
- **Multimodal similarity** — FAISS retrieval over 1024-d DenseNet GAP embeddings to surface comparable historical cases
- **RBAC** — Four roles (radiologist, ward doctor, clinical admin, system admin) with role-gated UI surfaces and an audit trail

## Architecture at a glance

```
frontend/   Next.js 13 App Router · TypeScript · Tailwind · Zustand
backend/    FastAPI · Supabase client · DenseNet121 inference · FAISS index
mimic_project/  ML training, model definitions, inference engine, notebooks
```

Roles are currently routed through a dev-only role switcher (`useUserRole` reads `cdss_dev_role` from localStorage). Real Supabase Auth is the next planned phase.

---

## Prerequisites

- Python 3.14 (standard, **not** the freethreaded `python3.14t` build — numpy is incompatible)
- Node.js 18+
- A free Supabase project (https://supabase.com)
- DenseNet121 weights (`baseline_best.pt`, 773 MB) — see `mimic_project/` training pipeline or contact the maintainer
- (Optional) The 40 demo MIMIC-CXR images, downloadable from PhysioNet via `backend/download_urls.txt`

---

## Setup

### 1. Supabase project

1. Create a new Supabase project. Note the URL and copy the **anon** + **service_role** keys (Settings → API).
2. Open the SQL editor and run the two schema files in order:
   - `backend/supabase_schema/schema.sql`
   - `backend/supabase_schema/002_users_audit_outcome.sql`

### 2. Backend

```bash
cd backend
python -m venv venv
# activate venv (Windows: venv\Scripts\activate, *nix: source venv/bin/activate)
pip install -r requirements.txt

cp .env.example .env
# Fill SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_ROLE_KEY, CXR_WEIGHTS_PATH
```

Run the one-time data setup (in this order):

```bash
py -V:3.14 migrate_to_supabase.py    # seed cases/patients/predictions from local_db.json
py -V:3.14 seed_demo_users.py        # 4 demo users (one per role)
py -V:3.14 seed_audit_log.py         # ~6 demo audit entries (optional)
py -V:3.14 batch_infer.py            # run DenseNet inference on the 40 demo cases (~7 min on CPU)
```

Start the API:

```bash
py -V:3.14 -m uvicorn main:app --port 8000 --reload
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 — it redirects to `/dashboard`.

### 4. (Optional) Real CXR images

The 40 demo images are not in the repo. To populate them:

1. Log into PhysioNet and follow the URLs in `backend/download_urls.txt`
2. Save all `.jpg` files to your `~/Downloads` folder
3. Run `py -V:3.14 backend/rename_downloads.py` — copies/renames into `frontend/public/mock-data/dicoms/`
4. Run `py -V:3.14 backend/trim_to_40.py` — verifies images, trims dataset, regenerates inference

---

## Roles & demo users

The dev role switcher (top-right header) lets you act as any of:

| Role | User | Sees |
|---|---|---|
| **Ward Doctor** | Dr. Ben Johnson | Full multimodal flow: Early Risk, CXR, ECG, Similar Cases, Before vs. After. Owns clinical decisions. |
| **Radiologist** | Dr. Alice Smith | My Queue (triage view); CXR Analysis (full controls + Flag Critical); Patient Summary (read-only) |
| **Clinical Admin** | Sarah Lee | Register Patient (4-step wizard), Upload Data, Case Status (no clinical findings) |
| **System Admin** | System Operator | Live metrics, **User Management** (Add/Role/Deactivate), **Audit Log**, login activity |

The `/admin` route is hidden from the nav for non–system-admin roles and redirects to `/dashboard` on direct access.

---

## Scripts reference

| Script | Purpose |
|---|---|
| `backend/migrate_to_supabase.py` | One-time JSON → Postgres migration |
| `backend/seed_demo_users.py` | Insert 4 demo users (one per role) |
| `backend/seed_audit_log.py` | Insert ~6 demo audit entries |
| `backend/batch_infer.py` | Run DenseNet inference on all cases |
| `backend/rename_downloads.py` | Copy/rename PhysioNet downloads into the app |
| `backend/trim_to_40.py` | Trim dataset to 40 real-image cases |
| `mimic_project/run_gradcam.py` | Standalone Grad-CAM CLI |
| `mimic_project/verify_ckpt.py` | Sanity-check a model checkpoint |

---

## Retraining the embeddings (advanced)

The application currently uses **1024-d DenseNet121 GAP features** for FAISS similarity. The original Symile architecture proposes **448-d multimodal embeddings** (ECG 128 + CXR 256 + Lab 64) trained with the Symile contrastive loss. The full retraining pipeline is vendored in two locations:

### Symile contrastive library — `symile-main/symile-main/`

MIT licensed (rajesh-lab, 2023). Contains:

- `symile/loss.py`, `symile/similarity.py` — the core contrastive loss + multi-modal similarity functions
- `experiments/main.py` — training entry point
- `experiments/models/symile_mimic_model.py` — the multimodal architecture
- `experiments/data_processing/symile_mimic/` — canonical data preprocessing for MIMIC

```bash
cd symile-main/symile-main
poetry install              # uses the vendored pyproject.toml + poetry.lock
poetry run python experiments/main.py --help
```

### Application-side training — `mimic_project/`

The repo also contains the **DenseNet baseline + FiLM-gated multimodal** training that produced the current `baseline_best.pt` checkpoint:

- `mimic_project/src/training/train_baseline.py` — single-modality CXR baseline
- `mimic_project/src/training/train_mm.py` — FiLM multimodal training
- `mimic_project/src/models/model_mm_film_gated.py` — architecture
- `mimic_project/src/preprocessing/` — preprocessing helpers
- `mimic_project/inference_engine/` — what `backend/` imports for inference

### Dataset preprocessing — `code/`

The Symile-MIMIC release's official preprocessing scripts (`process_mimic_data.py`, `create_dataset_splits.py`, `process_and_save_tensors.py`). These produce the `train.csv`/`val.csv`/`test.csv` splits and the `.npy` tensor files referenced in the data card.

---

## Audit log

Eight mutating endpoints write to `audit_log` automatically:

`case.create`, `case.complete`, `case.flag_critical`, `cxr.upload`, `cxr.reinfer`, `gradcam.regenerate`, `ecg.upload`, `labs.upload`, `faiss.reload`, `user.create`, `user.update`

The actor is identified by `X-User-Id` + `X-User-Role` headers (sent by the frontend from `useUserRole()`). Server-side `database.log_audit()` is best-effort and never raises. As a side effect it bumps `users.last_active_at` so System Admin sees live presence.

---

## What's intentionally not in the repo

- `backend/.env` — secrets
- `node_modules/`, `venv/`, `.next/`, `__pycache__/` — install artefacts
- `mimic_project/models/*.pt` — model weights (too large; download separately)
- `frontend/public/mock-data/dicoms/`, `heatmaps/` — generated/downloaded images
- Top-level CSV splits (`train.csv`, `val.csv`, `test.csv`) — derive from MIMIC-IV
- ML output dirs (`mimic_project/00_meta..08_selpred`, `results/`, `plots/`)

See `.gitignore` for the full list.

---

## License

See `LICENSE.txt` for the upstream Symile-MIMIC dataset license. Application code in `frontend/` and `backend/` is for clinical demonstration purposes only — **not for production clinical use without further validation**.
