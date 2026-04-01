import os
import json
import numpy as np
import pickle
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer
from usearch.index import Index

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import find_json_path, DATA_DIR

MAPPING_PATH = DATA_DIR.parent.parent / 'id_to_paper_id.pkl'
INDEX_PATH = DATA_DIR.parent.parent / 'database.usearch'

# Load mapping and index
with open(MAPPING_PATH, 'rb') as f:
    id_to_paper_id = pickle.load(f)

# Load a sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2', device='mps')

# Load the USearch index
index = Index(ndim=384, metric='cos')
index.load(INDEX_PATH)

def get_doi_by_title(title):
    url = "https://api.crossref.org/works"
    params = {"query.title": title}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            items = data.get("message", {}).get("items", [])
            if items:
                return items[0].get("DOI")
    except Exception:
        pass
    return "No results found"

def search(query, k=5):
    query_vector = model.encode([query]).astype(np.float32)
    matches = index.search(query_vector, k)
    
    indices = matches.keys.flatten().tolist()
    
    results = []
    for idx in indices:
        paper_id = id_to_paper_id[idx]
        json_path = find_json_path(paper_id, DATA_DIR)
        
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            data['paper_id'] = paper_id
            results.append(data)
            
    return results

def MLA_citation(title, doi):
    '''
    Generates a simplified MLA citation omitting authors and year as they are not present in current source.
    '''
    if doi != 'No results found' and doi != 'Failed to fetch data' and doi:
        return f"\"{title}.\" {doi}"
    else:
        return f"\"{title}.\""

def literature_search(query):
    results = search(query, k=3)
    
    for result in results:
        try:
            result['doi'] = get_doi_by_title(result['title'])
        except Exception:
            result['doi'] = 'Failed to fetch data'
        # Simplified DOI validation for the temporary build
        if result['doi'] != 'No results found' and result['doi'] != 'Failed to fetch data':
            result['doi'] = f"https://doi.org/{result['doi']}"
    
    message = ""
    references = []
    for result in results:
        message += f"Title: {result.get('title', 'Unknown Title')}\n\n"
        
        # Sections handling - skip title and abstract for specific section listing
        sections = {k: v for k, v in result.items() if k not in ['title', 'abstract', 'paper_id', 'doi']}
        
        if 'abstract' in result:
            message += f"Abstract: {result['abstract']}\n\n"
            
        for sec_title, sec_content in sections.items():
            message += f"**{sec_title}**:\n {sec_content}\n\n"
            
        references.append(f"{MLA_citation(result.get('title'), result.get('doi'))}\n\n")
        
    return message, references

if __name__ == "__main__":
    query = "coastal flooding"
    msg, refs = literature_search(query)
    print("SEARCH RESULTS:\n")
    print(msg)
    print("REFERENCES:\n")
    print("".join(refs))
