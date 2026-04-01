import os
import yaml
import time
from pathlib import Path
import streamlit as st
TEXT_CURSOR = "▕"

def load_config(path):
    """
    This function loads the config file.
    """
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def add_appendix(response: str, appendix_path: str):
    """
    This function adds the examples to the response.
    
    Args:
        response (str): The response string.
        appendix_path (str): The path to the appendix markdown file.
    """
    with open(appendix_path, "r") as f:
        appendix = f.read()
    response += appendix
    return response


def create_text_stream(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)

def stream_static_text(text):
    stream_text = create_text_stream(text)
    st.write_stream(stream_text)

def get_file_list(data_dir, limit=None):
    shard_dirs = sorted([d for d in os.listdir(data_dir) if d.isdigit()])
    files = []
    for shard in shard_dirs:
        shard_files = [os.path.join(shard, f) for f in os.listdir(data_dir / shard) if f.endswith('.json')]
        files.extend(shard_files)
    files = sorted(files)
    if limit:
        files = files[:limit]
    return files

def find_json_path(paper_id, data_dir):
    json_file = f"{paper_id}.json"
    shard_dirs = sorted([d for d in os.listdir(data_dir) if d.isdigit()])
    for shard in shard_dirs:
        path = data_dir / shard / json_file
        if path.exists():
            return path
    return data_dir / json_file
