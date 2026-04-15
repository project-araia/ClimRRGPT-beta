from src.literature.search import mla_citation, normalize_scores, tokenize


def test_tokenize():
    text = "Hello, World! This is a test 123."
    tokens = tokenize(text)
    assert tokens == ["hello", "world", "this", "is", "a", "test", "123"]


def test_normalize_scores_empty():
    assert normalize_scores({}, True) == {}


def test_normalize_scores_higher_is_better():
    raw = {1: 10.0, 2: 20.0, 3: 15.0}
    norm = normalize_scores(raw, True)
    assert norm[1] == 0.0
    assert norm[2] == 1.0
    assert norm[3] == 0.5


def test_normalize_scores_lower_is_better():
    raw = {1: 10.0, 2: 20.0, 3: 15.0}
    norm = normalize_scores(raw, False)
    assert norm[1] == 1.0
    assert norm[2] == 0.0
    assert norm[3] == 0.5


def test_mla_citation():
    assert (
        mla_citation("Climate Change", "10.1000/123")
        == '"Climate Change." https://doi.org/10.1000/123'
    )
    assert mla_citation("Wildfires", None) == '"Wildfires."'


def test_search_mocked(mocker):
    # Mocking load_resources to prevent actual loading
    mocker.patch("src.literature.search.load_resources")

    # Mock global variables that would be loaded by load_resources
    mocker.patch("src.literature.search._model")
    mocker.patch("src.literature.search._index")
    mocker.patch("src.literature.search._bm25")
    mocker.patch("src.literature.search._chunk_texts", ["text1", "text2"])
    mocker.patch(
        "src.literature.search._chunk_metadata", [{"title": "t1"}, {"title": "t2"}]
    )

    # Mock the search functions since they depend on the complex FAISS/BM25 objects
    mock_results = [
        {"idx": 0, "text": "text1", "metadata": {"title": "t1"}, "h_score": 0.9}
    ]
    mocker.patch("src.literature.search.search", return_value=mock_results)

    from src.literature.search import literature_search

    msg, refs = literature_search("query")

    assert "Source: t1" in msg
    assert "Content: text1" in msg
    assert len(refs) > 0
