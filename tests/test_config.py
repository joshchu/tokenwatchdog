"""Config loading/validation tests."""

from __future__ import annotations

import pytest

from tokenwatchdog.config import ConfigError, load_config


def _write(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_defaults_load_with_no_file(tmp_path):
    cfg = load_config(tmp_path / "config.toml")
    assert cfg.predictor.model == "auto"
    assert cfg.thresholds.burn_alert_within_hours == 1.0


def test_renamed_key_gets_an_actionable_hint_not_a_bare_unknown_key_error(tmp_path):
    """Regression: renaming a config field must not turn into an opaque
    crash for anyone with an existing config.toml — the error should say
    what the key became, not just that it's unrecognized."""
    path = _write(tmp_path, "[thresholds]\nburn_margin_hours = 0.25\n")
    with pytest.raises(ConfigError, match="burn_margin_hours.*burn_alert_within_hours"):
        load_config(path)


def test_genuinely_unknown_key_still_fails_loud(tmp_path):
    path = _write(tmp_path, "[thresholds]\nnot_a_real_key = 1\n")
    with pytest.raises(ConfigError, match="not_a_real_key"):
        load_config(path)


def test_working_hours_start_must_be_before_end(tmp_path):
    path = _write(
        tmp_path, '[working_hours]\nenabled = true\nstart = "17:00"\nend = "09:00"\n'
    )
    with pytest.raises(ConfigError):
        load_config(path)


def test_codex_token_prices_default_to_unconfigured(tmp_path):
    cfg = load_config(tmp_path / "config.toml")
    assert cfg.codex.input_price_per_million_usd == 0.0
    assert cfg.codex.output_price_per_million_usd == 0.0


def test_negative_codex_token_price_fails_loud(tmp_path):
    path = _write(tmp_path, "[codex]\ninput_price_per_million_usd = -1.0\n")
    with pytest.raises(ConfigError, match="input_price_per_million_usd"):
        load_config(path)


def test_claude_token_prices_default_to_unconfigured(tmp_path):
    cfg = load_config(tmp_path / "config.toml")
    assert cfg.claude.input_price_per_million_usd == 0.0
    assert cfg.claude.output_price_per_million_usd == 0.0
    assert cfg.claude.cache_write_price_per_million_usd == 0.0
    assert cfg.claude.cache_read_price_per_million_usd == 0.0


def test_negative_claude_token_price_fails_loud(tmp_path):
    path = _write(tmp_path, "[claude]\ncache_read_price_per_million_usd = -0.1\n")
    with pytest.raises(ConfigError, match="cache_read_price_per_million_usd"):
        load_config(path)
