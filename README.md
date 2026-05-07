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

Authentication uses **Supabase Auth** (email + password). After sign-in, the frontend attaches an `Authorization: Bearer <jwt>` header on every API call; the FastAPI backend verifies the token via `supabase.auth.get_user()` (works with both legacy HS256 and the current ECC P-256 signing keys) and bridges the auth identity to the application `users` table by email. A localStorage dev-shim (`cdss_dev_role`) is preserved as a fallback so local development still works without a live Supabase session.

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
py -V:3.14 seed_demo_users.py        # 4 demo users in the `users` table (one per role)
py -V:3.14 seed_auth_users.py        # create matching Supabase Auth accounts (password: Demo1234!)
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

Open http://localhost:3000 — it redirects to `/login`. Sign in with one of the four demo accounts (e.g. `dr.smith@hospital.org` / `Demo1234!`) to enter the role-gated workspace.

### 4. (Optional) Real CXR images

The 40 demo images are not in the repo. To populate them:

1. Log into PhysioNet and follow the URLs in `backend/download_urls.txt`
2. Save all `.jpg` files to your `~/Downloads` folder
3. Run `py -V:3.14 backend/rename_downloads.py` — copies/renames into `frontend/public/mock-data/dicoms/`
4. Run `py -V:3.14 backend/trim_to_40.py` — verifies images, trims dataset, regenerates inference

---

## Roles & demo users

Sign in at `/login` to enter the role-gated workspace. After authentication, the header shows a static identity chip (name + role badge) — no dev role-switcher.

| Role | User | Email (login) | Sees |
|---|---|---|---|
| **Ward Doctor** | Dr. Ben Johnson | `dr.johnson@hospital.org` | Full multimodal flow: Early Risk, CXR, ECG, Similar Cases, Before vs. After. Owns clinical decisions. |
| **Radiologist** | Dr. Alice Smith | `dr.smith@hospital.org` | My Queue (triage view); CXR Analysis (full controls + Flag Critical); Patient Summary (read-only) |
| **Clinical Admin** | Sarah Lee | `sarah.lee@hospital.org` | Register Patient (4-step wizard), Upload Data, Case Status (no clinical findings) |
| **System Admin** | System Operator | `ops@hospital.org` | Live metrics, **User Management** (Add/Role/Deactivate), **Audit Log**, login activity |

Default password for all demo accounts: `Demo1234!` (set by `seed_auth_users.py`; rotate before deploying).

The `/admin` route is hidden from the nav for non–system-admin roles and redirects to `/dashboard` on direct access. Server-side, every audited endpoint also calls `require_role(...)` so the gating cannot be bypassed by editing the client.

---

## Scripts reference

| Script | Purpose |
|---|---|
| `backend/migrate_to_supabase.py` | One-time JSON → Postgres migration |
| `backend/seed_demo_users.py` | Insert 4 demo users into the `users` table (one per role) |
| `backend/seed_auth_users.py` | Create matching Supabase Auth accounts so demo users can sign in |
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

The actor is identified by the verified Supabase JWT (`Authorization: Bearer <token>`); the backend calls `supabase.auth.get_user(token)` to validate it, extracts the `email` claim, and looks up the application `users` row by email to resolve the app-level UUID and role. The pre-auth `X-User-Id` + `X-User-Role` header shim is still accepted as a fallback when no Bearer token is present (used by local-dev runs). Server-side `database.log_audit()` is best-effort and never raises; as a side effect it bumps `users.last_active_at` so System Admin sees live presence.

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

## License & Attribution

This repository contains code and references to data under three different licenses. Read all three before redistributing.

### Application code (`frontend/`, `backend/`)
No open-source license declared — **all rights reserved** by default. For clinical demonstration purposes only; not for production clinical use without further validation.

### Symile contrastive learning library (`symile-main/`)
**MIT License** — Copyright © 2023 [rajesh-lab](https://github.com/rajesh-lab/symile). License text preserved in `symile-main/symile-main/LICENSE`. The vendored copy is unmodified; for the canonical version see the upstream repo.

### MIMIC-IV / MIMIC-CXR-JPG / Symile-MIMIC dataset and derivatives
**PhysioNet Credentialed Health Data License v1.5.0** — Copyright © 2025 MIT Laboratory for Computational Physiology. Full text in `LICENSE.txt`. This applies to:
- Anything imported from the dataset (subject_ids, study_ids, hadm_ids, lab values, DICOM paths)
- The preprocessing pipelines in `code/` and `symile-main/symile-main/experiments/data_processing/symile_mimic/`
- The DenseNet checkpoint trained on MIMIC-CXR (`mimic_project/models/baseline_best.pt`)
- Any inference outputs that map back to MIMIC subjects

> **Data redistribution notice:** Per clauses 1–3 of the PhysioNet license, MIMIC-derived data must not be shared with non-credentialed users. This repo deliberately **omits**: `backend/local_db.json`, `demo_form_values.json`, `frontend/public/mock-data/dicoms/`, model weights, and the `*.csv` splits. Each of these contains real MIMIC identifiers and must be regenerated locally from your own credentialed PhysioNet access.

---

## References & citations

If you use this codebase for research, please cite the relevant data sources, models, and methods below. Datasets are credentialed; redistribution is governed by the PhysioNet Data Use Agreement (see License above).

### Data sources

**MIMIC-IV electronic health records** — admissions, diagnoses, and the 50 lab biomarkers used by the Phase A risk model (`itemid` columns + `_percentile` rankings).

```bibtex
@article{johnson2023mimiciv,
  title   = {MIMIC-IV, a freely accessible electronic health record dataset},
  author  = {Johnson, Alistair E. W. and Bulgarelli, Lucas and Shen, Lu and Gow, Brian and Pollard, Tom J. and Horng, Steven and Celi, Leo Anthony and Mark, Roger G.},
  journal = {Scientific Data},
  volume  = {10},
  number  = {1},
  pages   = {1},
  year    = {2023},
  doi     = {10.1038/s41597-022-01899-x}
}
```

**MIMIC-CXR-JPG** — frontal chest radiographs with CheXpert-style structured labels; the source for the 8-class baseline classifier and Grad-CAM heatmaps.

```bibtex
@article{johnson2019mimiccxrjpg,
  title   = {MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs},
  author  = {Johnson, Alistair E. W. and Pollard, Tom J. and Greenbaum, Nathaniel R. and Lungren, Matthew P. and Deng, Chih-ying and Peng, Yifan and Lu, Zhiyong and Mark, Roger G. and Berkowitz, Seth J. and Horng, Steven},
  journal = {arXiv preprint arXiv:1901.07042},
  year    = {2019}
}

@inproceedings{johnson2019mimiccxrdb,
  title     = {MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports},
  author    = {Johnson, Alistair E. W. and Pollard, Tom J. and Berkowitz, Seth J. and Greenbaum, Nathaniel R. and Lungren, Matthew P. and Deng, Chih-ying and Mark, Roger G. and Horng, Steven},
  journal   = {Scientific Data},
  volume    = {6},
  number    = {1},
  pages     = {317},
  year      = {2019},
  doi       = {10.1038/s41597-019-0322-0}
}
```

**MIMIC-IV-ECG** — 12-lead diagnostic ECGs (10s @ 500 Hz) linked to MIMIC-IV admissions; the source for the (N, 1, 5000, 12) ECG tensors used by the multimodal model.

```bibtex
@misc{gow2023mimicivecg,
  title        = {MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset},
  author       = {Gow, Brian and Pollard, Tom and Nathanson, Larry A. and Johnson, Alistair and Moody, Benjamin and Fernandes, Chrystinne and Greenbaum, Nathaniel and Berkowitz, Seth and Horng, Steven and Mark, Roger},
  howpublished = {PhysioNet},
  year         = {2023},
  doi          = {10.13026/4nqg-sb35}
}
```

**Symile-MIMIC** — the curated tri-modal release (CXR + ECG + 50 labs) and train/val/test/val_retrieval splits this project consumes verbatim.

```bibtex
@misc{saporta2024symilemimic,
  title        = {Symile-MIMIC: A Multimodal Clinical Dataset of Chest X-rays, Electrocardiograms, and Blood Labs from MIMIC-IV},
  author       = {Saporta, Adriel and Puli, Aahlad and Goldstein, Mark and Ranganath, Rajesh},
  howpublished = {PhysioNet},
  year         = {2024}
}
```

**PhysioNet** — the platform hosting all four datasets above; cite the seminal infrastructure paper when describing data provenance.

```bibtex
@article{goldberger2000physionet,
  title   = {PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals},
  author  = {Goldberger, Ary L. and Amaral, Luis A. N. and Glass, Leon and Hausdorff, Jeffrey M. and Ivanov, Plamen Ch. and Mark, Roger G. and Mietus, Joseph E. and Moody, George B. and Peng, Chung-Kang and Stanley, H. Eugene},
  journal = {Circulation},
  volume  = {101},
  number  = {23},
  pages   = {e215--e220},
  year    = {2000},
  doi     = {10.1161/01.CIR.101.23.e215}
}
```

### Multimodal contrastive learning

**Symile** — the model-agnostic contrastive objective implemented in `symile-main/`; we use the public 24,576-d (3 × 8,192) embedding for FAISS retrieval.

```bibtex
@inproceedings{saporta2024symile,
  title     = {Contrasting with Symile: Simple Model-Agnostic Representation Learning for Unlimited Modalities},
  author    = {Saporta, Adriel and Puli, Aahlad and Goldstein, Mark and Ranganath, Rajesh},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  url       = {https://github.com/rajesh-lab/symile}
}
```

### Computer vision / model architecture

**DenseNet121** — the backbone of the CXR baseline classifier (`mimic_project/src/models/`); the 1024-d global-average-pooling output is also indexed in the DenseNet FAISS store.

```bibtex
@inproceedings{huang2017densenet,
  title     = {Densely Connected Convolutional Networks},
  author    = {Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q.},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2017},
  doi       = {10.1109/CVPR.2017.243}
}
```

**Grad-CAM** — the heatmap technique used in `engine/inference.py` to localise CXR findings.

```bibtex
@inproceedings{selvaraju2017gradcam,
  title     = {Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization},
  author    = {Selvaraju, Ramprasaath R. and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year      = {2017},
  doi       = {10.1109/ICCV.2017.74}
}
```

**CheXpert label schema** — the 14-finding label taxonomy adopted by MIMIC-CXR-JPG and used as our classification targets.

```bibtex
@inproceedings{irvin2019chexpert,
  title     = {CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison},
  author    = {Irvin, Jeremy and Rajpurkar, Pranav and Ko, Michael and others},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year      = {2019},
  doi       = {10.1609/aaai.v33i01.3301590}
}
```

### Uncertainty & calibration

**Monte-Carlo Dropout** — the basis of the per-prediction uncertainty bars (we run N=10 stochastic forward passes per case).

```bibtex
@inproceedings{gal2016dropout,
  title     = {Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning},
  author    = {Gal, Yarin and Ghahramani, Zoubin},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2016}
}
```

**Temperature scaling** — the post-hoc calibration step applied to the baseline classifier (T = 1.2518; ECE 0.040 → 0.027).

```bibtex
@inproceedings{guo2017calibration,
  title     = {On Calibration of Modern Neural Networks},
  author    = {Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2017}
}
```

### Similarity search

**FAISS** — the IndexFlatL2 used for both the 1024-d DenseNet store and the 24,576-d Symile multimodal store.

```bibtex
@article{johnson2019faiss,
  title   = {Billion-scale similarity search with GPUs},
  author  = {Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal = {IEEE Transactions on Big Data},
  volume  = {7},
  number  = {3},
  pages   = {535--547},
  year    = {2019},
  doi     = {10.1109/TBDATA.2019.2921572}
}
```

### Software stack (informational)

- **PyTorch** — Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS 2019.
- **FastAPI** — Tiangolo S., *FastAPI*, https://fastapi.tiangolo.com (BSD-3).
- **Next.js** — Vercel, *Next.js*, https://nextjs.org (MIT).
- **Supabase** — open-source Firebase alternative, https://supabase.com (Apache-2.0).
