import os
import json
import numpy as np
import pickle
from pathlib import Path
from usearch.index import Index
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_file_list, DATA_DIR

DEFAULT_LIMIT = 100000

parser = argparse.ArgumentParser(description='Build USearch index from sectionized JSON files.')
parser.add_argument('--data_dir', type=str, default=str(DATA_DIR), help='Path to the directory containing sectionized JSON files')
parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT, help='Number of documents to index')
args = parser.parse_args()

DATA_DIR = args.data_dir
LIMIT = args.limit

# Load a sentence transformer model
print("Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='mps')

d = 384

# Build a USearch HNSW index
print("Building USearch index...")
index = Index(ndim=d, metric='cos')

# Batched encoding with streaming index population
BATCH_SIZE = 2500
paper_ids = []
total_encoded = 0

print(f"Processing up to {LIMIT} documents in batches of {BATCH_SIZE}...")
files = get_file_list(DATA_DIR, LIMIT)

for batch_start in tqdm(range(0, len(files), BATCH_SIZE)):
    batch_files = files[batch_start:batch_start + BATCH_SIZE]
    batch_texts = []
    batch_ids = []
    
    for filename in batch_files:
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
        
        text_parts = []
        for key, value in data.items():
            if isinstance(value, str):
                text_parts.append(value)
            elif isinstance(value, list):
                text_parts.append(' '.join([str(v) for v in value]))
        
        batch_texts.append(' '.join(text_parts))
        batch_ids.append(filename.replace('.json', ''))
    
    document_embeddings = model.encode(batch_texts, show_progress_bar=False)
    
    start_idx = total_encoded
    end_idx = total_encoded + len(batch_ids)
    index.add(np.arange(start_idx, end_idx), document_embeddings)
    
    paper_ids.extend(batch_ids)
    total_encoded = end_idx

print(f"Encoded {total_encoded} documents in {(total_encoded + BATCH_SIZE - 1) // BATCH_SIZE} batches.")

# Save to disk
index.save(str(REPO_ROOT / 'src' / 'literature' / 'data' / 'database.usearch'))

# Save ID mapping
mapping_path = REPO_ROOT / 'src' / 'literature' / 'data' / 'id_to_paper_id.pkl'
with open(mapping_path, 'wb') as f:
    pickle.dump(paper_ids, f)

print(f"Index built with {len(index)} vectors of dimension {d}.")
print(f"ID mapping saved to {mapping_path}")
