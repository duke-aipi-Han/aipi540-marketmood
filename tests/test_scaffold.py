from marketmood.config import load_config
from marketmood.text_processing import build_text_input


def test_config_loads_project_name() -> None:
    config = load_config("config.yaml")
    assert config.values["project"]["name"] == "marketmood"


def test_ticker_masked_text_replaces_cashtags() -> None:
    text = build_text_input("$AAPL and $TSLA look strong", "AAPL", "ticker_masked")
    assert "$AAPL" not in text
    assert "$TSLA" not in text
    assert text.count("$TICKER") == 3
