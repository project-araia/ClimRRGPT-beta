import os
import sys
import time
from pathlib import Path

import streamlit as st
import yaml

TEXT_CURSOR = "▕"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "src" / "literature" / "data" / "data" / "resilience_dataset_4-1"


def load_config(path=None):
    """
    This function loads a config file. If path is None, it loads the global config.yml.
    """
    if path is None:
        path = REPO_ROOT / "config.yml"

    if not Path(path).exists():
        return {}

    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


GLOBAL_CONFIG = load_config()


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
        shard_files = [
            os.path.join(shard, f)
            for f in os.listdir(data_dir / shard)
            if f.endswith(".json")
        ]
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
