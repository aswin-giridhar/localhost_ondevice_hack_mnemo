from mnemo import config


def test_settings_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMO_DATA_DIR", str(tmp_path))
    s = config.load_settings()
    assert s.llm_models[0] == "phi4-mini"
    assert s.embed_model == "nomic-embed-text"
    assert s.memory_backend in ("cognee", "lance")
    assert config.data_path("traces.jsonl", s).parent == tmp_path
