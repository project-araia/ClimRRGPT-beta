import os
import json
import numpy as np
import pickle
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import re
import math
import mmap

import sys

# Set up paths relative to REPO_ROOT
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Configuration
DB_DIR = REPO_ROOT / "data"
DENSE_PATH = DB_DIR / "dense_ivfpq.faiss"
SPARSE_PATH = DB_DIR / "sparse_bm25.pkl"
MANIFEST_PATH = DB_DIR / "manifest.json"

# Hybrid Search Parameters
DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5
DENSE_TOP_K = 20
SPARSE_TOP_K = 20
FINAL_K = 10

# Load manifest
with open(MANIFEST_PATH, "r") as f:
    manifest = json.load(f)

EMBEDDING_MODEL = manifest.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B")

# Global variables
_model = None
_index = None
_bm25 = None
_chunk_texts = None
_chunk_metadata = None


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def normalize_scores(
    raw_scores: dict[int, float], higher_is_better: bool
) -> dict[int, float]:
    if not raw_scores:
        return {}
    values = np.asarray(list(raw_scores.values()), dtype=np.float32)
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if math.isclose(max_val, min_val, rel_tol=0.0, abs_tol=1e-12):
        return {key: 1.0 for key in raw_scores}
    if higher_is_better:
        return {
            key: (value - min_val) / (max_val - min_val)
            for key, value in raw_scores.items()
        }
    return {
        key: (max_val - value) / (max_val - min_val)
        for key, value in raw_scores.items()
    }


def load_resources():
    global _model, _index, _bm25, _chunk_texts, _chunk_metadata

    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    if _index is None:
        print(f"Loading dense index (MMAP): {DENSE_PATH.name}...")
        # FAISS supports memory mapping for some index types
        _index = faiss.read_index(
            str(DENSE_PATH), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
        )
        _index.nprobe = 32

    if _bm25 is None:
        print(f"Loading sparse index: {SPARSE_PATH.name}...")
        # To handle 3GB+ files on limited RAM, we use a buffered read
        # Note: pickle.load() unfortunately requires significant RAM to reconstruct the object.
        # If this still fails, we may need to refactor the build process to save data in chunks or use a proper DB.
        with open(SPARSE_PATH, "rb") as f:
            # Using a larger buffer for potentially faster reading
            sparse_payload = pickle.load(f)

        _token_corpus = sparse_payload.get("token_corpus", [])
        _chunk_texts = sparse_payload.get("chunk_texts", [])
        _chunk_metadata = sparse_payload.get("chunk_metadata", []) or [{}] * len(
            _chunk_texts
        )

        print("Initializing BM25 object...")
        _bm25 = BM25Okapi(_token_corpus)


def get_doi_by_title(title):
    url = "https://api.crossref.org/works"
    params = {"query.title": title}
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            items = data.get("message", {}).get("items", [])
            if items:
                return items[0].get("DOI")
    except Exception:
        pass
    return None


def mla_citation(title, doi):
    if doi:
        if not doi.startswith("http"):
            doi = f"https://doi.org/{doi}"
        return f'"{title}." {doi}'
    return f'"{title}."'


def search(query, k=FINAL_K):
    load_resources()

    query_vec = _model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    query_vec = np.asarray(query_vec, dtype=np.float32)

    dense_scores_arr, dense_ids_arr = _index.search(query_vec, DENSE_TOP_K)
    dense_scores_arr = dense_scores_arr[0]
    dense_ids_arr = dense_ids_arr[0]

    sparse_scores_all = _bm25.get_scores(tokenize(query))
    sparse_ids_top = np.argsort(sparse_scores_all)[::-1][:SPARSE_TOP_K]

    dense_raw = {}
    for score, idx in zip(dense_scores_arr, dense_ids_arr):
        idx = int(idx)
        if idx >= 0:
            dense_raw[idx] = float(score)

    sparse_raw = {}
    for idx in sparse_ids_top:
        idx = int(idx)
        sparse_raw[idx] = float(sparse_scores_all[idx])

    dense_norm = normalize_scores(dense_raw, higher_is_better=True)
    sparse_norm = normalize_scores(sparse_raw, higher_is_better=True)

    candidates = list(set(dense_raw.keys()) | set(sparse_raw.keys()))
    results = []

    for idx in candidates:
        h_score = DENSE_WEIGHT * dense_norm.get(
            idx, 0.0
        ) + SPARSE_WEIGHT * sparse_norm.get(idx, 0.0)

        results.append(
            {
                "idx": idx,
                "text": _chunk_texts[idx],
                "metadata": _chunk_metadata[idx],
                "h_score": h_score,
            }
        )

    results = sorted(results, key=lambda x: x["h_score"], reverse=True)
    return results[:k]


def literature_search(query):
    results = search(query, k=FINAL_K)

    output_message = ""
    references = []
    seen_titles = set()

    for res in results:
        text = res["text"]
        meta = res["metadata"]
        title = meta.get("title", "Unknown Title")

        output_message += f"Source: {title}\n"
        output_message += f"Content: {text}\n\n"

        if title not in seen_titles:
            doi = get_doi_by_title(title)
            ref = mla_citation(title, doi)
            references.append(f"{ref}\n\n")
            seen_titles.add(title)

    return output_message, references


if __name__ == "__main__":
    test_query = "coastal flooding"
    print(f"Testing hybrid search for: '{test_query}'")
    msg, refs = literature_search(test_query)
    print("\n--- MESSAGE ---\n")
    print(msg)
    print("\n--- REFERENCES ---\n")
    print("".join(refs))
