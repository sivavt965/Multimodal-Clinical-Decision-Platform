# =============================================================================
# engine/vector_store.py
# In-process FAISS index for CXR embedding similarity search.
# =============================================================================
"""
Architecture
------------
A VectorStore holds:
  - A FAISS IndexFlatL2 (exact L2 nearest-neighbour, no training required).
  - A parallel list `_id_map` that maps integer FAISS row-indices → case UUIDs.

The platform supports **two distinct embedding spaces**, each with its own
VectorStore instance and on-disk index file. They must never be merged because
their dimensionalities and semantics are different:

  * DenseNet (current)  : 1024-d L2-normalised GAP vectors from DenseNet121.
                          Captures CXR-only features.
                          File: faiss_index.bin (legacy default name).

  * Symile               : 24576-d multimodal embeddings produced by the
                          Symile-MIMIC contrastive checkpoint
                          (symile_mimic_model.ckpt). The vector is the
                          concatenation of three 8192-d per-modality reps
                          (CXR + ECG + Labs), L2-normalised.
                          File: faiss_symile_index.bin.
                          Producer: engine.symile_encoder.run_symile_inference.

Use `VectorStore.for_model("densenet")` or `for_model("symile")` to get the
right singleton. The bare `VectorStore()` constructor keeps backward
compatibility with the existing DenseNet index.

Thread safety
-------------
`add_to_index` is called from background thread-pool workers (one per new case).
A threading.Lock guards all mutating operations so concurrent writes are safe.

Persistence
-----------
The index is held in RAM for the lifetime of the uvicorn process.  On restart
it is rebuilt lazily as new cases are ingested (or pre-populated by the
`rebuild_from_db` helper below).  For durable storage call `save()` / `load()`.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-model embedding dimensions — DO NOT mix these in one index.
# ---------------------------------------------------------------------------
DENSENET_EMBEDDING_DIM: int = 1024  # DenseNet121 GAP (current production path)
# Symile checkpoint uses d=8192 per modality (CXR/ECG/Labs all project to 8192).
# We concatenate the three reps for retrieval, so the indexed vector is 3*d.
SYMILE_PER_MODALITY_DIM: int = 8192
SYMILE_EMBEDDING_DIM:    int = SYMILE_PER_MODALITY_DIM * 3   # 24576

# Backward-compat alias — existing callers reference EMBEDDING_DIM and assume
# DenseNet. Keep this until they migrate to the model-specific names.
EMBEDDING_DIM: int = DENSENET_EMBEDDING_DIM

# Per-model index file paths
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_INDEX_FILES = {
    "densenet": (_BACKEND_DIR / "faiss_index.bin",        _BACKEND_DIR / "faiss_id_map.npy"),
    "symile":   (_BACKEND_DIR / "faiss_symile_index.bin", _BACKEND_DIR / "faiss_symile_id_map.npy"),
}
_MODEL_DIMS = {
    "densenet": DENSENET_EMBEDDING_DIM,
    "symile":   SYMILE_EMBEDDING_DIM,
}

# Legacy defaults still pointing at DenseNet for back-compat
_DEFAULT_INDEX_PATH, _DEFAULT_MAP_PATH = _INDEX_FILES["densenet"]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class VectorStore:
    """Thread-safe per-model FAISS L2 index keyed by case UUID.

    One instance per embedding model — DenseNet (1024-d) and Symile (24576-d)
    are kept separate because their distance metrics aren't comparable.
    """

    _instances: dict[str, "VectorStore"] = {}
    # Reentrant: for_model() holds the lock and calls __init__() which would
    # otherwise deadlock on a plain Lock when re-acquiring.
    _class_lock: threading.RLock = threading.RLock()

    @classmethod
    def for_model(cls, model_name: str) -> "VectorStore":
        """Return the singleton VectorStore for the named embedding model."""
        if model_name not in _MODEL_DIMS:
            raise ValueError(
                f"Unknown embedding model '{model_name}'. "
                f"Known: {sorted(_MODEL_DIMS.keys())}"
            )
        if model_name not in cls._instances:
            with cls._class_lock:
                if model_name not in cls._instances:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    inst._model_name = model_name
                    # __init__ doesn't auto-fire when we create instances
                    # via super().__new__; do it explicitly so _dim/_index
                    # are set before the first method call.
                    inst.__init__()
                    cls._instances[model_name] = inst
        return cls._instances[model_name]

    def __new__(cls) -> "VectorStore":
        # Backward-compat: bare VectorStore() returns the DenseNet singleton.
        return cls.for_model("densenet")

    def __init__(self) -> None:
        if self._initialized:  # type: ignore[attr-defined]
            return
        with VectorStore._class_lock:
            if self._initialized:
                return
            # _model_name is set by for_model() before __init__ runs
            name = getattr(self, "_model_name", "densenet")
            dim = _MODEL_DIMS[name]
            self._dim: int = dim
            self._index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dim)
            self._id_map: List[str] = []          # row-index → case_id
            self._op_lock = threading.Lock()      # guards writes
            # Debounced-save bookkeeping: every add_to_index() flips _dirty,
            # and save_if_needed() throttles disk writes to once per interval.
            self._dirty: bool = False
            self._last_save_ts: float = 0.0
            self._initialized = True
            logger.info(
                "[VectorStore] Initialised FAISS IndexFlatL2 model=%s dim=%d",
                name, dim,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self._index.ntotal

    def add_to_index(self, case_id: str, embedding: np.ndarray) -> None:
        """
        Add or update a single 1024-d embedding in the FAISS index.
        If case_id already exists, replaces its entry to prevent duplicates.

        Parameters
        ----------
        case_id   : UUID string — used to resolve FAISS row indices back to cases.
        embedding : Raw numpy array of shape (1024,) or (1, 1024).
                    Will be L2-normalised internally.
        """
        vec = _coerce_embedding(embedding, self._dim)  # → (1, dim) float32
        vec = normalize(vec, norm="l2")                # unit sphere

        with self._op_lock:
            # Remove existing entry for this case_id to avoid duplicates on reinfer
            if case_id in self._id_map:
                old_row = self._id_map.index(case_id)
                self._id_map.pop(old_row)
                # Rebuild index without the old vector — reconstruct all other vectors.
                # np.vstack([]) raises, so handle the "old vector was the only one" case.
                kept = [
                    self._index.reconstruct(i)
                    for i in range(self._index.ntotal)
                    if i != old_row
                ]
                self._index.reset()
                if kept:
                    self._index.add(np.vstack(kept))  # type: ignore[arg-type]

            self._index.add(vec)                    # type: ignore[arg-type]
            self._id_map.append(case_id)
            self._dirty = True

        logger.info(
            "[VectorStore] Updated case %s (index size now %d)", case_id, self.size
        )

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        exclude_case_id: Optional[str] = None,
    ) -> List[dict]:
        """
        Return the *top_k* most similar cases by L2 distance.

        Parameters
        ----------
        query_embedding  : (1024,) or (1, 1024) float32 numpy array.
        top_k            : Number of neighbours to retrieve.
        exclude_case_id  : Optional case UUID to exclude from results
                           (typically the query case itself).

        Returns
        -------
        List of dicts: [{"case_id": str, "distance": float, "rank": int}, ...]
        sorted ascending by distance (most similar first).
        """
        vec = _coerce_embedding(query_embedding, self._dim)
        vec = normalize(vec, norm="l2")

        # FAISS objects are not thread-safe; hold the same lock writers do
        # so a concurrent add_to_index reset/rebuild can't return mid-state rows.
        with self._op_lock:
            if self._index.ntotal == 0:
                logger.warning("[VectorStore] Index is empty — returning no results.")
                return []

            k = min(top_k + (1 if exclude_case_id else 0), self._index.ntotal)
            distances, indices = self._index.search(vec, k)  # type: ignore[arg-type]
            id_map_snapshot = list(self._id_map)

        results: List[dict] = []
        for rank, (dist, idx) in enumerate(
            zip(distances[0].tolist(), indices[0].tolist()), start=1
        ):
            if idx < 0 or idx >= len(id_map_snapshot):
                continue
            cid = id_map_snapshot[idx]
            if cid == exclude_case_id:
                continue
            results.append({"case_id": cid, "distance": float(dist), "rank": rank})
            if len(results) >= top_k:
                break

        # Debug logging for retrieval verification (Epic 3.2)
        logger.info(
            "[VectorStore] search_similar: exclude=%s, k=%d, raw_indices=%s, raw_distances=%s",
            exclude_case_id, top_k,
            indices[0].tolist()[:top_k + 1],
            [f"{d:.4f}" for d in distances[0].tolist()[:top_k + 1]],
        )
        for r in results:
            logger.info(
                "[VectorStore]   → rank=%d  case_id=%s  L2_dist=%.4f",
                r["rank"], r["case_id"], r["distance"],
            )

        return results

    def get_embedding(self, case_id: str) -> Optional[np.ndarray]:
        """
        Reconstruct the stored vector for *case_id* from the FAISS index.

        Returns None if the case is not in the index.
        Note: only works with IndexFlatL2 (supports reconstruct).
        """
        with self._op_lock:
            try:
                row = self._id_map.index(case_id)
            except ValueError:
                return None
            vec = np.zeros((1, self._dim), dtype=np.float32)
            self._index.reconstruct(row, vec[0])   # type: ignore[attr-defined]
            return vec

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _default_paths(self) -> tuple[Path, Path]:
        """Per-model default index + id-map file paths."""
        return _INDEX_FILES[self._model_name]

    def save(
        self,
        index_path: Optional[Path] = None,
        map_path: Optional[Path] = None,
    ) -> None:
        """Persist the FAISS index and ID map to disk (per-model path by default)."""
        ip_default, mp_default = self._default_paths()
        index_path = index_path or ip_default
        map_path   = map_path or mp_default
        with self._op_lock:
            faiss.write_index(self._index, str(index_path))
            np.save(str(map_path), np.array(self._id_map))
            self._dirty = False
            self._last_save_ts = time.monotonic()
        logger.info("[VectorStore] Saved index (%d vectors) → %s", self.size, index_path)

    # Default debounce window for save_if_needed() — saves at most once per
    # this many seconds when called from the per-case API path.
    _DEFAULT_SAVE_DEBOUNCE_SEC: float = 30.0

    def save_if_needed(
        self,
        min_interval_sec: float = _DEFAULT_SAVE_DEBOUNCE_SEC,
        force: bool = False,
    ) -> bool:
        """
        Persist the index ONLY if there are unsaved changes AND enough time has
        elapsed since the last save (or force=True).

        Cheap to call repeatedly from the inference hot-path — does no disk I/O
        when the index is clean or the debounce window hasn't elapsed.

        Returns True if a save actually happened, False otherwise.
        """
        if not self._dirty:
            return False
        if not force and (time.monotonic() - self._last_save_ts) < min_interval_sec:
            return False
        self.save()
        return True

    def load(
        self,
        index_path: Optional[Path] = None,
        map_path: Optional[Path] = None,
    ) -> None:
        """Restore the FAISS index and ID map from disk (per-model path by default)."""
        ip_default, mp_default = self._default_paths()
        index_path = index_path or ip_default
        map_path   = map_path or mp_default
        if not index_path.exists() or not map_path.exists():
            logger.warning("[VectorStore] No persisted index found at %s", index_path)
            return
        with self._op_lock:
            loaded_index = faiss.read_index(str(index_path))
            if loaded_index.d != self._dim:
                # Loading a 1024-d file into the 448-d singleton (or vice-versa)
                # would silently break every subsequent add_to_index. Refuse it.
                raise ValueError(
                    f"[VectorStore] Refusing to load index with dim={loaded_index.d} "
                    f"into '{self._model_name}' store (expected dim={self._dim}). "
                    f"File: {index_path}"
                )
            self._index = loaded_index
            self._id_map = np.load(str(map_path), allow_pickle=True).tolist()
        logger.info("[VectorStore] Loaded index (%d vectors) from %s", self.size, index_path)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _coerce_embedding(arr: np.ndarray, dim: int = DENSENET_EMBEDDING_DIM) -> np.ndarray:
    """Ensure the array is shape (1, dim) float32. Default dim = DenseNet."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape != (1, dim):
        raise ValueError(
            f"Expected embedding of shape ({dim},) or (1, {dim}), "
            f"got {arr.shape}"
        )
    return arr


def get_vector_store() -> VectorStore:
    """Module-level accessor — lazy init on first call."""
    return VectorStore()
