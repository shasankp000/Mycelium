"""
Tests for the Phase 2 configuration module.
"""


import pytest

from phase2_validation.config.phase2_config import (
    Phase2Config,
    _env_bool,
    _env_float,
    _env_int,
    _env_str,
)


class TestEnvHelpers:
    def test_env_int_default(self):
        assert _env_int("P2_TEST_NOTSET_INT", 42) == 42

    def test_env_int_from_env(self, monkeypatch):
        monkeypatch.setenv("P2_TEST_INT", "99")
        assert _env_int("P2_TEST_INT", 0) == 99

    def test_env_int_invalid(self, monkeypatch):
        monkeypatch.setenv("P2_TEST_INT_BAD", "not_a_number")
        with pytest.raises(ValueError, match="must be an integer"):
            _env_int("P2_TEST_INT_BAD", 0)

    def test_env_float_default(self):
        assert _env_float("P2_TEST_NOTSET_FLOAT", 3.14) == 3.14

    def test_env_float_from_env(self, monkeypatch):
        monkeypatch.setenv("P2_TEST_FLOAT", "2.71")
        assert _env_float("P2_TEST_FLOAT", 0.0) == 2.71

    def test_env_float_invalid(self, monkeypatch):
        monkeypatch.setenv("P2_TEST_FLOAT_BAD", "nope")
        with pytest.raises(ValueError, match="must be a float"):
            _env_float("P2_TEST_FLOAT_BAD", 0.0)

    def test_env_str_default(self):
        assert _env_str("P2_TEST_NOTSET_STR", "hi") == "hi"

    def test_env_str_from_env(self, monkeypatch):
        monkeypatch.setenv("P2_TEST_STR", "world")
        assert _env_str("P2_TEST_STR", "hi") == "world"

    def test_env_bool_default_true(self):
        assert _env_bool("P2_TEST_NOTSET_BOOL", True) is True

    def test_env_bool_default_false(self):
        assert _env_bool("P2_TEST_NOTSET_BOOL", False) is False

    def test_env_bool_from_env_true(self, monkeypatch):
        for val in ("true", "1", "yes", "True", "YES"):
            monkeypatch.setenv("P2_TEST_BOOL", val)
            assert _env_bool("P2_TEST_BOOL", False) is True

    def test_env_bool_from_env_false(self, monkeypatch):
        monkeypatch.setenv("P2_TEST_BOOL", "false")
        assert _env_bool("P2_TEST_BOOL", True) is False


class TestPhase2Config:
    def test_default_values(self):
        config = Phase2Config()
        assert config.input_min_length == 10
        assert config.input_max_length == 50000
        assert config.input_min_tokens == 3
        assert config.input_max_tokens == 10000
        assert config.similarity_threshold == 0.45
        assert config.log_level == "INFO"

    def test_validate_defaults_ok(self):
        config = Phase2Config()
        errors = config.validate()
        assert errors == []

    def test_validate_bad_min_length(self):
        config = Phase2Config(input_min_length=-1)
        errors = config.validate()
        assert any("input_min_length" in e for e in errors)

    def test_validate_max_less_than_min(self):
        config = Phase2Config(
            input_min_length=100, input_max_length=10
        )
        errors = config.validate()
        assert any("input_max_length" in e for e in errors)

    def test_validate_bad_tokens(self):
        config = Phase2Config(
            input_min_tokens=-1, input_max_tokens=-2
        )
        errors = config.validate()
        assert len(errors) >= 2

    def test_validate_thresholds_out_of_range(self):
        config = Phase2Config(
            language_confidence_threshold=1.5,
            similarity_threshold=-0.1,
            tag_confidence_threshold=2.0,
            spam_score_threshold=-1.0,
            rejection_score_threshold=5.0,
            coherence_min_score=1.1,
        )
        errors = config.validate()
        assert len(errors) == 6

    def test_validate_bad_max_tags(self):
        config = Phase2Config(max_tags=0)
        errors = config.validate()
        assert any("max_tags" in e for e in errors)

    def test_validate_bad_timeout(self):
        config = Phase2Config(pipeline_timeout_seconds=-1.0)
        errors = config.validate()
        assert any("pipeline_timeout" in e for e in errors)

    def test_validate_empty_languages(self):
        config = Phase2Config(supported_languages=[])
        errors = config.validate()
        assert any("supported_languages" in e for e in errors)

    def test_validate_bad_log_level(self):
        config = Phase2Config(log_level="VERBOSE")
        errors = config.validate()
        assert any("log_level" in e for e in errors)

    def test_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "input_min_length: 20\n"
            "input_max_length: 10000\n"
            "log_level: DEBUG\n"
        )
        config = Phase2Config.from_yaml(str(yaml_file))
        assert config.input_min_length == 20
        assert config.input_max_length == 10000
        assert config.log_level == "DEBUG"

    def test_from_yaml_unknown_key(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("unknown_key: 123\n")
        config = Phase2Config.from_yaml(str(yaml_file))
        assert not hasattr(config, "unknown_key")

    def test_from_yaml_empty(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("")
        config = Phase2Config.from_yaml(str(yaml_file))
        assert config.input_min_length == 10
