# =============================================================================
# engine/vector_store.py
# In-process FAISS index for CXR embedding similarity search.
# =============================================================================
"""
Architecture
------------
One singleton VectorStore holds:
  - A FAISS IndexFlatL2 (exact L2 nearest-neighbour, no training required).
  - A parallel list `_id_map` that maps integer FAISS row-indices → case UUIDs.

Embeddings are 1024-dimensional L2-normalised GAP vectors from DenseNet121.

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
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dimension constant — must match the DenseNet121 GAP output (1024-d)
# ---------------------------------------------------------------------------
EMBEDDING_DIM: int = 1024

# Default path for persisting the FAISS index between restarts
_DEFAULT_INDEX_PATH = Path(__file__).resolve().parent.parent / "faiss_index.bin"
_DEFAULT_MAP_PATH   = Path(__file__).resolve().parent.parent / "faiss_id_map.npy"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class VectorStore:
    """Thread-safe singleton FAISS L2 index keyed by case UUID."""

    _instance: "VectorStore | None" = None
    _class_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "VectorStore":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:  # type: ignore[attr-defined]
            return
        with VectorStore._class_lock:
            if self._initialized:
                return
            self._index: faiss.IndexFlatL2 = faiss.IndexFlatL2(EMBEDDING_DIM)
            self._id_map: List[str] = []          # row-index → case_id
            self._op_lock = threading.Lock()      # guards writes
            self._initialized = True
            logger.info(
                "[VectorStore] Initialised FAISS IndexFlatL2 (dim=%d)", EMBEDDING_DIM
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
        vec = _coerce_embedding(embedding)          # → (1, 1024) float32
        vec = normalize(vec, norm="l2")             # unit sphere

        with self._op_lock:
            # Remove existing entry for this case_id to avoid duplicates on reinfer
            if case_id in self._id_map:
                old_row = self._id_map.index(case_id)
                self._id_map.pop(old_row)
                # Rebuild index without the old vector — reconstruct all other vectors
                all_vecs = np.vstack([
                    self._index.reconstruct(i)
                    for i in range(self._index.ntotal)
                    if i != old_row
                ])
                self._index.reset()
                if len(all_vecs) > 0:
                    self._index.add(all_vecs)       # type: ignore[arg-type]

            self._index.add(vec)                    # type: ignore[arg-type]
            self._id_map.append(case_id)

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
        if self._index.ntotal == 0:
            logger.warning("[VectorStore] Index is empty — returning no results.")
            return []

        vec = _coerce_embedding(query_embedding)
        vec = normalize(vec, norm="l2")

        # Retrieve extra candidates to allow filtering of the excluded case
        k = min(top_k + (1 if exclude_case_id else 0), self._index.ntotal)

        distances, indices = self._index.search(vec, k)  # type: ignore[arg-type]

        results: List[dict] = []
        for rank, (dist, idx) in enumerate(
            zip(distances[0].tolist(), indices[0].tolist()), start=1
        ):
            if idx < 0 or idx >= len(self._id_map):
                continue
            cid = self._id_map[idx]
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
            vec = np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
            self._index.reconstruct(row, vec[0])   # type: ignore[attr-defined]
            return vec

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(
        self,
        index_path: Path = _DEFAULT_INDEX_PATH,
        map_path: Path = _DEFAULT_MAP_PATH,
    ) -> None:
        """Persist the FAISS index and ID map to disk."""
        faiss.write_index(self._index, str(index_path))
        np.save(str(map_path), np.array(self._id_map))
        logger.info("[VectorStore] Saved index (%d vectors) → %s", self.size, index_path)

    def load(
        self,
        index_path: Path = _DEFAULT_INDEX_PATH,
        map_path: Path = _DEFAULT_MAP_PATH,
    ) -> None:
        """Restore the FAISS index and ID map from disk."""
        if not index_path.exists() or not map_path.exists():
            logger.warning("[VectorStore] No persisted index found at %s", index_path)
            return
        with self._op_lock:
            self._index = faiss.read_index(str(index_path))
            self._id_map = np.load(str(map_path), allow_pickle=True).tolist()
        logger.info("[VectorStore] Loaded index (%d vectors) from %s", self.size, index_path)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _coerce_embedding(arr: np.ndarray) -> np.ndarray:
    """Ensure the array is shape (1, EMBEDDING_DIM) float32."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape != (1, EMBEDDING_DIM):
        raise ValueError(
            f"Expected embedding of shape ({EMBEDDING_DIM},) or (1, {EMBEDDING_DIM}), "
            f"got {arr.shape}"
        )
    return arr


def get_vector_store() -> VectorStore:
    """Module-level accessor — lazy init on first call."""
    return VectorStore()
