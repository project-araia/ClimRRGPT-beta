import os
import json
import numpy as np
import pickle
from usearch.index import Index
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse

# Path to the sectionized directory
DEFAULT_DATA_DIR = './src/literature/data/data/titanv_all_terms_results_v2_2026-03-26_12:13:28_sectionized'
DEFAULT_LIMIT = 10000

parser = argparse.ArgumentParser(description='Build USearch index from sectionized JSON files.')
parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR, help='Path to the directory containing sectionized JSON files')
parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT, help='Number of documents to index')
args = parser.parse_args()

DATA_DIR = args.data_dir
LIMIT = args.limit

# Load data from JSON files
combined_texts = []
paper_ids = []

print(f"Loading up to {LIMIT} documents from {DATA_DIR}...")
files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
files = sorted(files)[:LIMIT]

for filename in tqdm(files):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    # Combine all text fields
    # Values can be strings or potentially lists/dicts if not cleaned, but usually strings in this dataset
    text_parts = []
    for key, value in data.items():
        if isinstance(value, str):
            text_parts.append(value)
        elif isinstance(value, list):
            text_parts.append(' '.join([str(v) for v in value]))
            
    combined_text = ' '.join(text_parts)
    combined_texts.append(combined_text)
    paper_ids.append(filename.replace('.json', ''))

# Load a sentence transformer model
print("Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='mps')

# Encode documents
print(f"Encoding {len(combined_texts)} documents...")
document_embeddings = model.encode(combined_texts, show_progress_bar=True)

# Dimension of vectors
d = document_embeddings.shape[1]

# Build a USearch HNSW index
print("Building USearch index...")
index = Index(ndim=d, metric='cos')
index.add(np.arange(len(document_embeddings)), document_embeddings)

# Save to disk
index.save('./src/literature/data/climate_index.usearch')

# Save ID mapping
mapping_path = './src/literature/data/id_to_paper_id.pkl'
with open(mapping_path, 'wb') as f:
    pickle.dump(paper_ids, f)

print(f"Index built with {len(index)} vectors of dimension {d}.")
print(f"ID mapping saved to {mapping_path}")
