import os
import json
import numpy as np
import pickle
import requests
from sentence_transformers import SentenceTransformer
from usearch.index import Index

# Configuration
DATA_DIR = './src/literature/data/data/titanv_all_terms_results_v2_2026-03-26_12:13:28_sectionized'
MAPPING_PATH = './src/literature/data/id_to_paper_id.pkl'
INDEX_PATH = './src/literature/data/climate_index.usearch'

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
    
    # matches.keys contains the index positions
    indices = matches.keys.flatten().tolist()
    
    results = []
    for idx in indices:
        paper_id = id_to_paper_id[idx]
        json_path = os.path.join(DATA_DIR, f"{paper_id}.json")
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
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
        result['doi'] = get_doi_by_title(result['title'])
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
            message += f"{sec_title}: {sec_content}\n\n"
            
        references.append(f"{MLA_citation(result.get('title'), result.get('doi'))}\n\n")
        
    return message, references

if __name__ == "__main__":
    query = "wildfire mitigation strategies for bridge construction in wildfire-prone areas"
    msg, refs = literature_search(query)
    print("SEARCH RESULTS:\n")
    print(msg)
    print("REFERENCES:\n")
    print("".join(refs))
