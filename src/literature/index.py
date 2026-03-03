import numpy as np
import pickle
from usearch.index import Index

# Load embeddings
document_embeddings = pickle.load(open('./src/literature/data/document_embeddings.pkl', 'rb'))
document_embeddings = document_embeddings.astype(np.float32)

# Dimension of vectors
d = document_embeddings.shape[1]

# Build a USearch HNSW index (cosine distance matches sentence-transformer norms well)
index = Index(ndim=d, metric='cos')
index.add(np.arange(len(document_embeddings)), document_embeddings)

# Save to disk
index.save('./src/literature/data/climate_index.usearch')
print(f"Index built with {len(index)} vectors of dimension {d}.")
