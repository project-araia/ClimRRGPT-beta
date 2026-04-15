from src.utils import find_json_path, get_file_list, load_config


def test_load_config_missing(tmp_path):
    # Test loading from a non-existent path
    path = tmp_path / "non_existent.yml"
    config = load_config(path)
    assert config == {}


def test_get_file_list(tmp_path):
    # Create mock directory structure
    data_dir = tmp_path / "data"
    shard1 = data_dir / "001"
    shard1.mkdir(parents=True)
    (shard1 / "paper1.json").write_text("{}")
    (shard1 / "paper2.json").write_text("{}")

    shard2 = data_dir / "002"
    shard2.mkdir(parents=True)
    (shard2 / "paper3.json").write_text("{}")
    (shard2 / "not_a_paper.txt").write_text("...")

    files = get_file_list(data_dir)
    assert len(files) == 3
    assert "001/paper1.json" in files
    assert "002/paper3.json" in files


def test_find_json_path(tmp_path):
    data_dir = tmp_path / "data"
    shard = data_dir / "042"
    shard.mkdir(parents=True)
    json_file = shard / "123.456.json"
    json_file.write_text("{}")

    path = find_json_path("123.456", data_dir)
    assert path == json_file

    # Test not found
    path_not_found = find_json_path("missing", data_dir)
    assert path_not_found == data_dir / "missing.json"
